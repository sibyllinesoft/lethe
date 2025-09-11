import React, { useRef, useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { TransformChangeV2, ChangeHistogramData } from '../../types/transform';
import { AlertTriangle, TrendingUp } from 'lucide-react';
import clsx from 'clsx';

interface ChangeHistogramProps {
  changes: TransformChangeV2[];
  className?: string;
  detailed?: boolean;
  onBarClick?: (changeType: string) => void;
}

/**
 * ChangeHistogram - D3-powered bar chart visualization of change type frequency
 * 
 * Features:
 * - 15+ granular change types from TransformChangeV2
 * - Severity color coding (low: green, medium: yellow, high: orange, critical: red)
 * - Performance impact visualization
 * - Success rate indicators
 * - Interactive tooltips and click handlers
 * - Responsive design with smooth animations
 */
export const ChangeHistogram: React.FC<ChangeHistogramProps> = ({
  changes,
  className,
  detailed = false,
  onBarClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedBar, setSelectedBar] = useState<string | null>(null);
  
  // Process changes into histogram data
  const histogramData = useMemo<ChangeHistogramData[]>(() => {
    const changeTypeMap = new Map<string, {
      count: number;
      totalPerformanceImpact: number;
      successCount: number;
      severity: 'low' | 'medium' | 'high' | 'critical';
    }>();

    changes.forEach(change => {
      const changeType = Object.keys(change.change_type)[0];
      const performance = change.metadata.performance;
      const isSuccess = !change.metadata.performance || change.metadata.confidence_score !== 0;
      
      if (!changeTypeMap.has(changeType)) {
        changeTypeMap.set(changeType, {
          count: 0,
          totalPerformanceImpact: 0,
          successCount: 0,
          severity: determineSeverity(changeType)
        });
      }

      const data = changeTypeMap.get(changeType)!;
      data.count++;
      data.totalPerformanceImpact += performance?.duration_us || 0;
      if (isSuccess) data.successCount++;
    });

    return Array.from(changeTypeMap.entries()).map(([changeType, data]) => ({
      changeType: formatChangeType(changeType),
      count: data.count,
      severity: data.severity,
      avgPerformanceImpact: data.totalPerformanceImpact / data.count,
      successRate: (data.successCount / data.count) * 100
    })).sort((a, b) => b.count - a.count);
  }, [changes]);

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || histogramData.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = detailed 
      ? { top: 20, right: 30, bottom: 100, left: 60 }
      : { top: 20, right: 20, bottom: 60, left: 40 };
    
    const container = svg.node()?.getBoundingClientRect();
    if (!container) return;

    const width = container.width - margin.left - margin.right;
    const height = container.height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Scales
    const xScale = d3.scaleBand()
      .domain(histogramData.map(d => d.changeType))
      .range([0, width])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(histogramData, d => d.count) || 1])
      .range([height, 0]);

    // Color scale based on severity
    const colorScale = (severity: string) => {
      switch (severity) {
        case 'critical': return '#dc2626'; // red-600
        case 'high': return '#ea580c'; // orange-600
        case 'medium': return '#ca8a04'; // yellow-600
        case 'low': return '#16a34a'; // green-600
        default: return '#6b7280'; // gray-500
      }
    };

    // Bars
    const bars = g.selectAll('.bar')
      .data(histogramData)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.changeType) || 0)
      .attr('width', xScale.bandwidth())
      .attr('y', height)
      .attr('height', 0)
      .attr('fill', d => colorScale(d.severity))
      .attr('opacity', 0.8)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1);
        
        // Tooltip
        const tooltip = d3.select('body').append('div')
          .attr('class', 'histogram-tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <div><strong>${d.changeType}</strong></div>
            <div>Count: ${d.count}</div>
            <div>Success Rate: ${d.successRate.toFixed(1)}%</div>
            <div>Avg Performance: ${(d.avgPerformanceImpact / 1000).toFixed(2)}ms</div>
            <div>Severity: ${d.severity}</div>
          `);

        const [mouseX, mouseY] = d3.pointer(event, document.body);
        tooltip
          .style('left', mouseX + 10 + 'px')
          .style('top', mouseY - 10 + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8);
        d3.selectAll('.histogram-tooltip').remove();
      })
      .on('click', function(_, d) {
        setSelectedBar(selectedBar === d.changeType ? null : d.changeType);
        onBarClick?.(d.changeType);
      });

    // Animate bars
    bars.transition()
      .duration(750)
      .delay((_, i) => i * 50)
      .attr('y', d => yScale(d.count))
      .attr('height', d => height - yScale(d.count));

    // X axis
    const xAxis = g.append('g')
      .attr('transform', `translate(0, ${height})`)
      .call(d3.axisBottom(xScale));

    xAxis.selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)')
      .style('font-size', detailed ? '12px' : '10px');

    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('font-size', detailed ? '12px' : '10px');

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('fill', '#374151')
      .text('Change Count');

    // Success rate indicators (small bars on top)
    if (detailed) {
      const successScale = d3.scaleLinear()
        .domain([0, 100])
        .range([0, 20]);

      g.selectAll('.success-bar')
        .data(histogramData)
        .enter().append('rect')
        .attr('class', 'success-bar')
        .attr('x', d => (xScale(d.changeType) || 0) + xScale.bandwidth() * 0.1)
        .attr('width', xScale.bandwidth() * 0.8)
        .attr('y', d => yScale(d.count) - 25)
        .attr('height', d => successScale(d.successRate))
        .attr('fill', '#10b981')
        .attr('opacity', 0.6);
    }

  }, [histogramData, detailed, selectedBar, onBarClick]);

  // Summary statistics
  const totalChanges = histogramData.reduce((sum, d) => sum + d.count, 0);
  const avgSuccessRate = histogramData.length > 0 
    ? histogramData.reduce((sum, d) => sum + d.successRate, 0) / histogramData.length 
    : 0;
  const criticalChanges = histogramData.filter(d => d.severity === 'critical');

  return (
    <div className={clsx('bg-white dark:bg-gray-900', className)}>
      {detailed && (
        <div className="mb-4 flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Change Type Distribution
          </h3>
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-600 rounded mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">Critical</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-orange-600 rounded mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">High</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-600 rounded mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">Medium</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-600 rounded mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">Low</span>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 relative">
        <svg
          ref={svgRef}
          className="w-full h-full"
          style={{ minHeight: detailed ? '400px' : '200px' }}
        />
      </div>

      {detailed && (
        <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
            <div className="font-medium text-blue-900 dark:text-blue-100">
              Total Changes
            </div>
            <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
              {totalChanges}
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
            <div className="font-medium text-green-900 dark:text-green-100 flex items-center">
              <TrendingUp size={16} className="mr-1" />
              Success Rate
            </div>
            <div className="text-xl font-bold text-green-600 dark:text-green-400">
              {avgSuccessRate.toFixed(1)}%
            </div>
          </div>

          <div className="bg-red-50 dark:bg-red-900/20 rounded p-3">
            <div className="font-medium text-red-900 dark:text-red-100 flex items-center">
              <AlertTriangle size={16} className="mr-1" />
              Critical Issues
            </div>
            <div className="text-xl font-bold text-red-600 dark:text-red-400">
              {criticalChanges.reduce((sum, d) => sum + d.count, 0)}
            </div>
          </div>
        </div>
      )}

      {selectedBar && (
        <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-800 rounded">
          <div className="text-sm font-medium text-gray-900 dark:text-white">
            Selected: {selectedBar}
          </div>
        </div>
      )}
    </div>
  );
};

// Helper functions
function determineSeverity(changeType: string): 'low' | 'medium' | 'high' | 'critical' {
  // Critical: Security and validation issues
  if (changeType.includes('validation') || changeType.includes('sanitized')) {
    return 'critical';
  }
  
  // High: System modifications and structural changes
  if (changeType.includes('system') || changeType.includes('structure') || changeType.includes('encoding')) {
    return 'high';
  }
  
  // Medium: Content rewrites and enhancements
  if (changeType.includes('rewritten') || changeType.includes('enhanced') || changeType.includes('converted')) {
    return 'medium';
  }
  
  // Low: Additions and no-ops
  return 'low';
}

function formatChangeType(changeType: string): string {
  return changeType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}