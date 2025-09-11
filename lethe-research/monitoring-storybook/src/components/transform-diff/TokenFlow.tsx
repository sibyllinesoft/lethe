import React, { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';
import { sankey, sankeyLinkHorizontal, SankeyNode, SankeyLink } from 'd3-sankey';
import { TransformChangeV2, TokenMetrics } from '../../types/transform';
import { ArrowRight, ArrowUp, ArrowDown } from 'lucide-react';
import clsx from 'clsx';

interface TokenFlowProps {
  tokenMetrics: TokenMetrics;
  changes: TransformChangeV2[];
  className?: string;
  detailed?: boolean;
}

interface FlowData {
  nodes: Array<{ id: string; name: string; category: 'source' | 'transform' | 'destination' }>;
  links: Array<{ source: string; target: string; value: number; type: 'added' | 'removed' | 'modified' }>;
}

/**
 * TokenFlow - Sankey diagram showing token allocation changes through transformations
 * 
 * Features:
 * - Visual representation of token flow from input to output
 * - Color-coded links showing added, removed, and modified tokens
 * - Interactive tooltips with detailed metrics
 * - Efficiency indicators and delta calculations
 * - Real-time updates as new changes arrive
 */
export const TokenFlow: React.FC<TokenFlowProps> = ({
  tokenMetrics,
  changes,
  className,
  detailed = false
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  // Process changes into flow data
  const flowData = useMemo<FlowData>(() => {
    // Group changes by type to understand token flow patterns
    const changesByType = changes.reduce((acc, change) => {
      const changeType = Object.keys(change.change_type)[0];
      if (!acc[changeType]) {
        acc[changeType] = [];
      }
      acc[changeType].push(change);
      return acc;
    }, {} as Record<string, TransformChangeV2[]>);

    const nodes: FlowData['nodes'] = [
      { id: 'input', name: 'Input Tokens', category: 'source' },
      { id: 'output', name: 'Output Tokens', category: 'destination' }
    ];

    const links: FlowData['links'] = [];

    // Add transformation nodes based on change types
    Object.entries(changesByType).forEach(([changeType, changeList]) => {
      const nodeId = `transform_${changeType}`;
      const nodeName = formatChangeType(changeType);
      
      nodes.push({
        id: nodeId,
        name: nodeName,
        category: 'transform'
      });

      // Calculate token flow through this transformation
      const avgInputSize = changeList.reduce((sum, change) => 
        sum + (change.metadata.performance?.input_size_bytes || 0), 0) / changeList.length;
      const avgOutputSize = changeList.reduce((sum, change) => 
        sum + (change.metadata.performance?.output_size_bytes || 0), 0) / changeList.length;

      // Estimate token counts (assuming ~4 chars per token)
      const inputTokens = Math.round(avgInputSize / 4);
      const outputTokens = Math.round(avgOutputSize / 4);

      // Input to transformation
      links.push({
        source: 'input',
        target: nodeId,
        value: inputTokens,
        type: 'modified'
      });

      // Transformation to output
      if (outputTokens > inputTokens) {
        links.push({
          source: nodeId,
          target: 'output',
          value: outputTokens,
          type: 'added'
        });
      } else if (outputTokens < inputTokens) {
        links.push({
          source: nodeId,
          target: 'output',
          value: outputTokens,
          type: 'removed'
        });
      } else {
        links.push({
          source: nodeId,
          target: 'output',
          value: outputTokens,
          type: 'modified'
        });
      }
    });

    // If no transformations, show direct flow
    if (Object.keys(changesByType).length === 0) {
      links.push({
        source: 'input',
        target: 'output',
        value: tokenMetrics.after,
        type: tokenMetrics.delta > 0 ? 'added' : tokenMetrics.delta < 0 ? 'removed' : 'modified'
      });
    }

    return { nodes, links };
  }, [changes, tokenMetrics]);

  // D3 Sankey visualization
  useEffect(() => {
    if (!svgRef.current || flowData.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const container = svg.node()?.getBoundingClientRect();
    if (!container) return;

    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    const width = container.width - margin.left - margin.right;
    const height = container.height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Create sankey layout
    const sankeyLayout = sankey<SankeyNode<any, any>, SankeyLink<any, any>>()
      .nodeWidth(15)
      .nodePadding(10)
      .extent([[1, 1], [width - 1, height - 5]]);

    // Transform data for d3-sankey
    const sankeyData = {
      nodes: flowData.nodes.map(node => ({ ...node })),
      links: flowData.links.map(link => ({
        ...link,
        source: flowData.nodes.findIndex(n => n.id === link.source),
        target: flowData.nodes.findIndex(n => n.id === link.target)
      }))
    };

    const { nodes, links } = sankeyLayout(sankeyData as any);

    // Color scales
    const nodeColorScale = (category: string) => {
      switch (category) {
        case 'source': return '#3b82f6'; // blue-500
        case 'transform': return '#8b5cf6'; // violet-500
        case 'destination': return '#10b981'; // emerald-500
        default: return '#6b7280'; // gray-500
      }
    };

    const linkColorScale = (type: string) => {
      switch (type) {
        case 'added': return '#10b981'; // emerald-500
        case 'removed': return '#ef4444'; // red-500
        case 'modified': return '#f59e0b'; // amber-500
        default: return '#6b7280'; // gray-500
      }
    };

    // Draw links
    g.append('g')
      .selectAll('.link')
      .data(links)
      .enter().append('path')
      .attr('class', 'link')
      .attr('d', sankeyLinkHorizontal() as any)
      .attr('stroke', (d: any) => linkColorScale(d.type))
      .attr('stroke-width', (d: any) => Math.max(1, d.width))
      .attr('fill', 'none')
      .attr('opacity', 0.5)
      .on('mouseover', function(event, d: any) {
        d3.select(this).attr('opacity', 0.8);
        
        const tooltip = d3.select('body').append('div')
          .attr('class', 'tokenflow-tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <div><strong>${d.source.name} → ${d.target.name}</strong></div>
            <div>Tokens: ${d.value}</div>
            <div>Type: ${d.type}</div>
          `);

        const [mouseX, mouseY] = d3.pointer(event, document.body);
        tooltip
          .style('left', mouseX + 10 + 'px')
          .style('top', mouseY - 10 + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.5);
        d3.selectAll('.tokenflow-tooltip').remove();
      });

    // Draw nodes
    g.append('g')
      .selectAll('.node')
      .data(nodes)
      .enter().append('rect')
      .attr('class', 'node')
      .attr('x', (d: any) => d.x0)
      .attr('y', (d: any) => d.y0)
      .attr('height', (d: any) => d.y1 - d.y0)
      .attr('width', (d: any) => d.x1 - d.x0)
      .attr('fill', (d: any) => nodeColorScale(d.category))
      .attr('opacity', 0.8)
      .on('mouseover', function(event, d: any) {
        d3.select(this).attr('opacity', 1);
        
        const tooltip = d3.select('body').append('div')
          .attr('class', 'tokenflow-tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <div><strong>${d.name}</strong></div>
            <div>Category: ${d.category}</div>
            <div>Value: ${d.value || 'N/A'}</div>
          `);

        const [mouseX, mouseY] = d3.pointer(event, document.body);
        tooltip
          .style('left', mouseX + 10 + 'px')
          .style('top', mouseY - 10 + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8);
        d3.selectAll('.tokenflow-tooltip').remove();
      });

    // Add node labels
    g.append('g')
      .selectAll('.node-label')
      .data(nodes)
      .enter().append('text')
      .attr('class', 'node-label')
      .attr('x', (d: any) => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
      .attr('y', (d: any) => (d.y1 + d.y0) / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', (d: any) => d.x0 < width / 2 ? 'start' : 'end')
      .style('font-size', detailed ? '12px' : '10px')
      .style('font-weight', '500')
      .style('fill', '#374151')
      .text((d: any) => d.name);

  }, [flowData, detailed]);

  // Calculate efficiency metrics
  const efficiency = tokenMetrics.before > 0 ? (tokenMetrics.after / tokenMetrics.before) : 1;
  const isEfficient = efficiency >= 0.9 && efficiency <= 1.1; // Within 10% is considered efficient

  return (
    <div className={clsx('bg-white dark:bg-gray-900', className)}>
      {detailed && (
        <div className="mb-4 flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Token Flow Analysis
          </h3>
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-emerald-500 rounded mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">Added</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-500 rounded mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">Removed</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-amber-500 rounded mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">Modified</span>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 relative">
        <svg
          ref={svgRef}
          className="w-full h-full"
          style={{ minHeight: detailed ? '300px' : '200px' }}
        />
      </div>

      {/* Token metrics summary */}
      <div className={clsx(
        'mt-4 grid gap-4',
        detailed ? 'grid-cols-4' : 'grid-cols-2'
      )}>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
          <div className="flex items-center">
            <ArrowRight className="text-blue-600 dark:text-blue-400" size={16} />
            <div className="ml-2">
              <div className="text-sm font-medium text-blue-900 dark:text-blue-100">
                Input Tokens
              </div>
              <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                {tokenMetrics.before.toLocaleString()}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
          <div className="flex items-center">
            <ArrowRight className="text-green-600 dark:text-green-400" size={16} />
            <div className="ml-2">
              <div className="text-sm font-medium text-green-900 dark:text-green-100">
                Output Tokens
              </div>
              <div className="text-lg font-bold text-green-600 dark:text-green-400">
                {tokenMetrics.after.toLocaleString()}
              </div>
            </div>
          </div>
        </div>

        {detailed && (
          <>
            <div className={clsx(
              'rounded p-3',
              tokenMetrics.delta > 0 
                ? 'bg-emerald-50 dark:bg-emerald-900/20' 
                : tokenMetrics.delta < 0
                ? 'bg-red-50 dark:bg-red-900/20'
                : 'bg-gray-50 dark:bg-gray-800'
            )}>
              <div className="flex items-center">
                {tokenMetrics.delta > 0 ? (
                  <ArrowUp className="text-emerald-600 dark:text-emerald-400" size={16} />
                ) : tokenMetrics.delta < 0 ? (
                  <ArrowDown className="text-red-600 dark:text-red-400" size={16} />
                ) : (
                  <ArrowRight className="text-gray-600 dark:text-gray-400" size={16} />
                )}
                <div className="ml-2">
                  <div className={clsx(
                    'text-sm font-medium',
                    tokenMetrics.delta > 0 
                      ? 'text-emerald-900 dark:text-emerald-100'
                      : tokenMetrics.delta < 0
                      ? 'text-red-900 dark:text-red-100'
                      : 'text-gray-900 dark:text-gray-100'
                  )}>
                    Delta
                  </div>
                  <div className={clsx(
                    'text-lg font-bold',
                    tokenMetrics.delta > 0 
                      ? 'text-emerald-600 dark:text-emerald-400'
                      : tokenMetrics.delta < 0
                      ? 'text-red-600 dark:text-red-400'
                      : 'text-gray-600 dark:text-gray-400'
                  )}>
                    {tokenMetrics.delta > 0 ? '+' : ''}{tokenMetrics.delta.toLocaleString()}
                  </div>
                </div>
              </div>
            </div>

            <div className={clsx(
              'rounded p-3',
              isEfficient 
                ? 'bg-green-50 dark:bg-green-900/20'
                : 'bg-yellow-50 dark:bg-yellow-900/20'
            )}>
              <div className="flex items-center">
                <div className={clsx(
                  'w-4 h-4 rounded-full mr-2',
                  isEfficient ? 'bg-green-500' : 'bg-yellow-500'
                )} />
                <div>
                  <div className={clsx(
                    'text-sm font-medium',
                    isEfficient 
                      ? 'text-green-900 dark:text-green-100'
                      : 'text-yellow-900 dark:text-yellow-100'
                  )}>
                    Efficiency
                  </div>
                  <div className={clsx(
                    'text-lg font-bold',
                    isEfficient 
                      ? 'text-green-600 dark:text-green-400'
                      : 'text-yellow-600 dark:text-yellow-400'
                  )}>
                    {(efficiency * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Helper function to format change types
function formatChangeType(changeType: string): string {
  return changeType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}