import React, { useRef, useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { TransformChangeV2, KVMetrics } from '../../types/transform';
import { TrendingUp, AlertCircle, Hash } from 'lucide-react';
import clsx from 'clsx';

interface KVPrefixHeatmapProps {
  kvMetrics: KVMetrics;
  changes: TransformChangeV2[];
  className?: string;
  detailed?: boolean;
}

interface HeatmapCell {
  row: number;
  col: number;
  value: number;
  type: 'jaccard' | 'volatility' | 'head_edit' | 'tail_edit' | 'frequency';
  label: string;
  intensity: number; // 0-1 scale for color intensity
}

/**
 * KVPrefixHeatmap - Heatmap visualization of KV prefix impact analysis
 * 
 * Features:
 * - Jaccard similarity visualization for prefix matching
 * - Volatility metrics showing change frequency
 * - Head/tail edit analysis for positional impact
 * - Interactive cells with detailed tooltips
 * - Color-coded intensity based on impact level
 * - Real-time updates as metrics change
 */
export const KVPrefixHeatmap: React.FC<KVPrefixHeatmapProps> = ({
  kvMetrics,
  changes,
  className,
  detailed = false
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedCell, setSelectedCell] = useState<HeatmapCell | null>(null);
  
  // Process changes and metrics into heatmap data
  const heatmapData = useMemo<HeatmapCell[]>(() => {
    const cells: HeatmapCell[] = [];
    
    // Group changes by provider and model to create a matrix view
    const providerModelMap = new Map<string, Map<string, TransformChangeV2[]>>();
    
    changes.forEach(change => {
      const provider = change.context.target_provider || 'unknown';
      const model = change.context.target_model || 'default';
      
      if (!providerModelMap.has(provider)) {
        providerModelMap.set(provider, new Map());
      }
      if (!providerModelMap.get(provider)!.has(model)) {
        providerModelMap.get(provider)!.set(model, []);
      }
      providerModelMap.get(provider)!.get(model)!.push(change);
    });

    const providers = Array.from(providerModelMap.keys());
    const allModels = new Set<string>();
    providerModelMap.forEach(models => {
      models.forEach((_, model) => allModels.add(model));
    });
    const models = Array.from(allModels);

    // Generate heatmap cells
    providers.forEach((provider, providerIdx) => {
      models.forEach((model, modelIdx) => {
        const changesForCell = providerModelMap.get(provider)?.get(model) || [];
        
        if (changesForCell.length === 0) {
          // Empty cell
          cells.push({
            row: providerIdx,
            col: modelIdx,
            value: 0,
            type: 'frequency',
            label: `${provider}-${model}`,
            intensity: 0
          });
          return;
        }

        // Calculate metrics for this provider-model combination
        // const avgPerformance = changesForCell.reduce((sum, change) => 
        //   sum + (change.metadata.performance?.duration_us || 0), 0) / changesForCell.length;
        
        const sizeChanges = changesForCell
          .map(change => change.metadata.performance)
          .filter(perf => perf && perf.input_size_bytes > 0)
          .map(perf => (perf!.output_size_bytes - perf!.input_size_bytes) / perf!.input_size_bytes);
        
        const avgSizeChange = sizeChanges.length > 0 
          ? sizeChanges.reduce((sum, change) => sum + change, 0) / sizeChanges.length 
          : 0;

        // Jaccard similarity (simulated based on size changes)
        const jaccardScore = Math.max(0, 1 - Math.abs(avgSizeChange));
        cells.push({
          row: providerIdx,
          col: modelIdx,
          value: jaccardScore,
          type: 'jaccard',
          label: `${provider}-${model} (Jaccard)`,
          intensity: jaccardScore
        });

        // Volatility based on change frequency and performance variance
        const performanceVariance = changesForCell.length > 1 
          ? d3.variance(changesForCell.map(c => c.metadata.performance?.duration_us || 0)) || 0
          : 0;
        const volatility = Math.min(1, performanceVariance / 1000000); // Normalize
        cells.push({
          row: providerIdx,
          col: modelIdx + models.length,
          value: volatility,
          type: 'volatility',
          label: `${provider}-${model} (Volatility)`,
          intensity: volatility
        });

        // Head edits (changes affecting early tokens)
        const headEditCount = changesForCell.filter(change => {
          const changeType = Object.keys(change.change_type)[0];
          return changeType.includes('system') || changeType.includes('prelude');
        }).length;
        const headEditIntensity = headEditCount / Math.max(1, changesForCell.length);
        cells.push({
          row: providerIdx,
          col: modelIdx + models.length * 2,
          value: headEditCount,
          type: 'head_edit',
          label: `${provider}-${model} (Head Edits)`,
          intensity: headEditIntensity
        });

        // Tail edits (changes affecting later tokens)
        const tailEditCount = changesForCell.filter(change => {
          const changeType = Object.keys(change.change_type)[0];
          return changeType.includes('content') || changeType.includes('enhanced');
        }).length;
        const tailEditIntensity = tailEditCount / Math.max(1, changesForCell.length);
        cells.push({
          row: providerIdx,
          col: modelIdx + models.length * 3,
          value: tailEditCount,
          type: 'tail_edit',
          label: `${provider}-${model} (Tail Edits)`,
          intensity: tailEditIntensity
        });
      });
    });

    return cells;
  }, [changes, kvMetrics]);

  // D3 heatmap visualization
  useEffect(() => {
    if (!svgRef.current || heatmapData.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const container = svg.node()?.getBoundingClientRect();
    if (!container) return;

    const margin = { top: 60, right: 40, bottom: 40, left: 100 };
    const width = container.width - margin.left - margin.right;
    const height = container.height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Get dimensions
    const maxRow = Math.max(...heatmapData.map(d => d.row)) + 1;
    const maxCol = Math.max(...heatmapData.map(d => d.col)) + 1;

    const cellWidth = width / maxCol;
    const cellHeight = height / maxRow;

    // Color scale based on cell type
    const colorScale = (type: string, intensity: number) => {
      const alpha = Math.max(0.1, intensity);
      switch (type) {
        case 'jaccard':
          return d3.interpolateBlues(alpha);
        case 'volatility':
          return d3.interpolateOranges(alpha);
        case 'head_edit':
          return d3.interpolateReds(alpha);
        case 'tail_edit':
          return d3.interpolateGreens(alpha);
        case 'frequency':
          return d3.interpolatePurples(alpha);
        default:
          return d3.interpolateGreys(alpha);
      }
    };

    // Draw cells
    g.selectAll('.cell')
      .data(heatmapData)
      .enter().append('rect')
      .attr('class', 'cell')
      .attr('x', d => d.col * cellWidth)
      .attr('y', d => d.row * cellHeight)
      .attr('width', cellWidth)
      .attr('height', cellHeight)
      .attr('fill', d => colorScale(d.type, d.intensity))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('stroke-width', 2).attr('stroke', '#000000');
        
        const tooltip = d3.select('body').append('div')
          .attr('class', 'heatmap-tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <div><strong>${d.label}</strong></div>
            <div>Value: ${d.value.toFixed(3)}</div>
            <div>Type: ${d.type.replace('_', ' ')}</div>
            <div>Intensity: ${(d.intensity * 100).toFixed(1)}%</div>
          `);

        const [mouseX, mouseY] = d3.pointer(event, document.body);
        tooltip
          .style('left', mouseX + 10 + 'px')
          .style('top', mouseY - 10 + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke-width', 1).attr('stroke', '#ffffff');
        d3.selectAll('.heatmap-tooltip').remove();
      })
      .on('click', function(_, d) {
        setSelectedCell(selectedCell === d ? null : d);
      });

    // Column headers (metric types)
    const metricTypes = ['Jaccard', 'Volatility', 'Head Edits', 'Tail Edits'];
    metricTypes.forEach((type, idx) => {
      g.append('text')
        .attr('class', 'col-header')
        .attr('x', (idx * cellWidth * 2) + (cellWidth * 2) / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .style('fill', '#374151')
        .text(type);
    });

    // Row headers (providers)
    const providers = [...new Set(changes.map(c => c.context.target_provider || 'unknown'))];
    providers.forEach((provider, idx) => {
      g.append('text')
        .attr('class', 'row-header')
        .attr('x', -10)
        .attr('y', (idx * cellHeight) + cellHeight / 2)
        .attr('text-anchor', 'end')
        .attr('dy', '0.35em')
        .style('font-size', '12px')
        .style('font-weight', '500')
        .style('fill', '#374151')
        .text(provider);
    });

    // Legend
    if (detailed) {
      const legend = g.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${width - 120}, 20)`);

      const legendData = [
        { type: 'jaccard', color: '#3b82f6', label: 'Jaccard' },
        { type: 'volatility', color: '#f97316', label: 'Volatility' },
        { type: 'head_edit', color: '#ef4444', label: 'Head Edits' },
        { type: 'tail_edit', color: '#10b981', label: 'Tail Edits' }
      ];

      legendData.forEach((item, idx) => {
        const legendRow = legend.append('g')
          .attr('transform', `translate(0, ${idx * 20})`);

        legendRow.append('rect')
          .attr('width', 15)
          .attr('height', 15)
          .attr('fill', item.color);

        legendRow.append('text')
          .attr('x', 20)
          .attr('y', 12)
          .style('font-size', '11px')
          .style('fill', '#374151')
          .text(item.label);
      });
    }

  }, [heatmapData, detailed, selectedCell]);

  return (
    <div className={clsx('bg-white dark:bg-gray-900', className)}>
      {detailed && (
        <div className="mb-4 flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            KV Prefix Impact Analysis
          </h3>
          <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300">
            <Hash size={16} />
            <span>Hover cells for details</span>
          </div>
        </div>
      )}

      <div className="flex-1 relative">
        <svg
          ref={svgRef}
          className="w-full h-full"
          style={{ minHeight: detailed ? '400px' : '250px' }}
        />
      </div>

      {/* KV Metrics Summary */}
      <div className={clsx(
        'mt-4 grid gap-4',
        detailed ? 'grid-cols-4' : 'grid-cols-2'
      )}>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
          <div className="flex items-center">
            <TrendingUp className="text-blue-600 dark:text-blue-400" size={16} />
            <div className="ml-2">
              <div className="text-sm font-medium text-blue-900 dark:text-blue-100">
                Prefix Jaccard
              </div>
              <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                {kvMetrics.prefixJaccard.toFixed(3)}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-3">
          <div className="flex items-center">
            <AlertCircle className="text-orange-600 dark:text-orange-400" size={16} />
            <div className="ml-2">
              <div className="text-sm font-medium text-orange-900 dark:text-orange-100">
                Volatility
              </div>
              <div className="text-lg font-bold text-orange-600 dark:text-orange-400">
                {kvMetrics.volatility.toFixed(3)}
              </div>
            </div>
          </div>
        </div>

        {detailed && (
          <>
            <div className="bg-red-50 dark:bg-red-900/20 rounded p-3">
              <div className="flex items-center">
                <Hash className="text-red-600 dark:text-red-400" size={16} />
                <div className="ml-2">
                  <div className="text-sm font-medium text-red-900 dark:text-red-100">
                    Head Edits
                  </div>
                  <div className="text-lg font-bold text-red-600 dark:text-red-400">
                    {kvMetrics.headEdits}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
              <div className="flex items-center">
                <Hash className="text-green-600 dark:text-green-400" size={16} />
                <div className="ml-2">
                  <div className="text-sm font-medium text-green-900 dark:text-green-100">
                    Tail Edits
                  </div>
                  <div className="text-lg font-bold text-green-600 dark:text-green-400">
                    {kvMetrics.tailEdits || 0}
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {selectedCell && (
        <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
            Selected Cell Details
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600 dark:text-gray-300">Label:</span>
              <span className="ml-2 font-medium">{selectedCell.label}</span>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-300">Type:</span>
              <span className="ml-2 font-medium">{selectedCell.type.replace('_', ' ')}</span>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-300">Value:</span>
              <span className="ml-2 font-medium">{selectedCell.value.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-300">Intensity:</span>
              <span className="ml-2 font-medium">{(selectedCell.intensity * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};