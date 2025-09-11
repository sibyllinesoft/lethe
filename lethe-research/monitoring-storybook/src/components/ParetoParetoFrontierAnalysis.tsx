import React, { useState, useEffect, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import { scaleLinear, scaleOrdinal, schemeCategory10 } from 'd3-scale';
import { line, curveMonotoneX } from 'd3-shape';
import { select } from 'd3-selection';
import { zoom, zoomIdentity } from 'd3-zoom';
import { TrendingUp, Settings, Download, RefreshCw, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

/**
 * Complexity vs K2 Pareto Frontier Analysis Dashboard
 * 
 * Analyzes the trade-off between transform complexity and K2 coefficient values,
 * identifying optimal configurations that minimize complexity while maintaining
 * performance quality. Displays Pareto-optimal solutions and performance boundaries.
 */

interface ParetoParetoPoint {
  complexity: number;
  k2Coefficient: number;
  performanceScore: number;
  difficultyGateScore: number;
  provider: string;
  modelSize: '256' | '768';
  timestamp: Date;
  isOptimal: boolean;
  dominatedBy?: string[];
  metadata: {
    requestId: string;
    scenario: string;
    successRate: number;
    avgLatency: number;
    tokenEfficiency: number;
  };
}

interface ParetoAnalysisMetrics {
  frontierSize: number;
  dominationRatio: number;
  optimalityGap: number;
  convergenceRate: number;
  stabilityIndex: number;
}

interface ParetoFrontierAnalysisProps {
  data: ParetoParetoPoint[];
  realTimeEnabled?: boolean;
  onOptimalConfigurationSelected?: (config: ParetoParetoPoint) => void;
  onExport?: (format: 'png' | 'svg' | 'json' | 'csv') => void;
  className?: string;
}

export const ParetoFrontierAnalysis: React.FC<ParetoFrontierAnalysisProps> = ({
  data = [],
  realTimeEnabled = false,
  onOptimalConfigurationSelected,
  onExport,
  className
}) => {
  const [selectedPoint, setSelectedPoint] = useState<ParetoParetoPoint | null>(null);
  const [viewMode, setViewMode] = useState<'frontier' | 'evolution' | 'sensitivity'>('frontier');
  const [filterProvider, setFilterProvider] = useState<string>('all');
  const [filterModelSize, setFilterModelSize] = useState<string>('all');
  const [zoomLevel, setZoomLevel] = useState(1);

  const svgRef = React.useRef<SVGSVGElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Pareto frontier calculation
  const paretoAnalysis = useMemo(() => {
    if (!data.length) return { frontierPoints: [], dominated: [], metrics: null };

    // Calculate Pareto frontier
    const sortedData = [...data].sort((a, b) => a.complexity - b.complexity);
    const frontierPoints: ParetoParetoPoint[] = [];
    const dominated: ParetoParetoPoint[] = [];

    for (const point of sortedData) {
      let isDominated = false;
      const dominatingPoints: string[] = [];

      // Check if this point is dominated by any frontier point
      for (const frontierPoint of frontierPoints) {
        if (frontierPoint.complexity <= point.complexity && 
            frontierPoint.k2Coefficient >= point.k2Coefficient &&
            (frontierPoint.complexity < point.complexity || frontierPoint.k2Coefficient > point.k2Coefficient)) {
          isDominated = true;
          dominatingPoints.push(frontierPoint.metadata.requestId);
        }
      }

      if (!isDominated) {
        // Remove any existing frontier points that are dominated by this point
        for (let i = frontierPoints.length - 1; i >= 0; i--) {
          const existingPoint = frontierPoints[i];
          if (point.complexity <= existingPoint.complexity && 
              point.k2Coefficient >= existingPoint.k2Coefficient &&
              (point.complexity < existingPoint.complexity || point.k2Coefficient > existingPoint.k2Coefficient)) {
            dominated.push({ ...existingPoint, isOptimal: false });
            frontierPoints.splice(i, 1);
          }
        }
        frontierPoints.push({ ...point, isOptimal: true });
      } else {
        dominated.push({ ...point, isOptimal: false, dominatedBy: dominatingPoints });
      }
    }

    // Calculate analysis metrics
    const metrics: ParetoAnalysisMetrics = {
      frontierSize: frontierPoints.length,
      dominationRatio: dominated.length / data.length,
      optimalityGap: frontierPoints.length > 1 
        ? (Math.max(...frontierPoints.map(p => p.k2Coefficient)) - Math.min(...frontierPoints.map(p => p.k2Coefficient))) / Math.max(...data.map(p => p.k2Coefficient))
        : 0,
      convergenceRate: calculateConvergenceRate(frontierPoints),
      stabilityIndex: calculateStabilityIndex(frontierPoints)
    };

    return { frontierPoints, dominated, metrics };
  }, [data]);

  // Filter data based on selected criteria
  const filteredData = useMemo(() => {
    return data.filter(point => {
      if (filterProvider !== 'all' && point.provider !== filterProvider) return false;
      if (filterModelSize !== 'all' && point.modelSize !== filterModelSize) return false;
      return true;
    });
  }, [data, filterProvider, filterModelSize]);

  // Visualization setup
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !filteredData.length) return;

    const svg = select(svgRef.current);
    const container = containerRef.current;
    const { width, height } = container.getBoundingClientRect();

    const margin = { top: 40, right: 120, bottom: 60, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.bottom - margin.top;

    svg.selectAll('*').remove();

    // Scales
    const xScale = scaleLinear()
      .domain(d3.extent(filteredData, d => d.complexity) as [number, number])
      .range([0, innerWidth])
      .nice();

    const yScale = scaleLinear()
      .domain(d3.extent(filteredData, d => d.k2Coefficient) as [number, number])
      .range([innerHeight, 0])
      .nice();

    const colorScale = scaleOrdinal(schemeCategory10)
      .domain(Array.from(new Set(filteredData.map(d => d.provider))));

    const sizeScale = scaleLinear()
      .domain(d3.extent(filteredData, d => d.performanceScore) as [number, number])
      .range([4, 12]);

    // Main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickSize(-innerHeight)
        .tickFormat('' as any))
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale)
        .tickSize(-innerWidth)
        .tickFormat('' as any))
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    // Pareto frontier line
    if (paretoAnalysis.frontierPoints.length > 1) {
      const frontierLine = line<ParetoParetoPoint>()
        .x(d => xScale(d.complexity))
        .y(d => yScale(d.k2Coefficient))
        .curve(curveMonotoneX);

      const sortedFrontier = paretoAnalysis.frontierPoints.sort((a, b) => a.complexity - b.complexity);

      g.append('path')
        .datum(sortedFrontier)
        .attr('fill', 'none')
        .attr('stroke', '#ff6b6b')
        .attr('stroke-width', 3)
        .attr('stroke-dasharray', '5,5')
        .attr('d', frontierLine)
        .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))');

      // Frontier area fill
      const area = d3.area<ParetoParetoPoint>()
        .x(d => xScale(d.complexity))
        .y0(innerHeight)
        .y1(d => yScale(d.k2Coefficient))
        .curve(curveMonotoneX);

      g.append('path')
        .datum(sortedFrontier)
        .attr('fill', 'url(#gradientFill)')
        .attr('opacity', 0.1)
        .attr('d', area);

      // Gradient definition
      const defs = svg.append('defs');
      const gradient = defs.append('linearGradient')
        .attr('id', 'gradientFill')
        .attr('gradientUnits', 'userSpaceOnUse')
        .attr('x1', 0).attr('y1', innerHeight)
        .attr('x2', 0).attr('y2', 0);

      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#ff6b6b')
        .attr('stop-opacity', 0);

      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', '#ff6b6b')
        .attr('stop-opacity', 0.3);
    }

    // Data points
    const points = g.selectAll('.data-point')
      .data(filteredData)
      .enter().append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => xScale(d.complexity))
      .attr('cy', d => yScale(d.k2Coefficient))
      .attr('r', d => sizeScale(d.performanceScore))
      .attr('fill', d => d.isOptimal ? '#ff6b6b' : (colorScale(d.provider) as string))
      .attr('stroke', d => d.isOptimal ? '#ffffff' : 'none')
      .attr('stroke-width', d => d.isOptimal ? 2 : 0)
      .style('opacity', d => d.isOptimal ? 1 : 0.6)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        // Tooltip
        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${xScale(d.complexity)},${yScale(d.k2Coefficient)})`);

        const rect = tooltip.append('rect')
          .attr('x', 10)
          .attr('y', -60)
          .attr('width', 200)
          .attr('height', 80)
          .attr('fill', 'rgba(0,0,0,0.8)')
          .attr('rx', 4);

        const text = tooltip.append('text')
          .attr('x', 20)
          .attr('y', -40)
          .attr('fill', 'white')
          .attr('font-size', '12px');

        text.append('tspan')
          .attr('x', 20)
          .attr('dy', '0')
          .text(`${d.provider} (${d.modelSize})`);

        text.append('tspan')
          .attr('x', 20)
          .attr('dy', '14')
          .text(`Complexity: ${d.complexity.toFixed(2)}`);

        text.append('tspan')
          .attr('x', 20)
          .attr('dy', '14')
          .text(`K2: ${d.k2Coefficient.toFixed(3)}`);

        text.append('tspan')
          .attr('x', 20)
          .attr('dy', '14')
          .text(`Performance: ${(d.performanceScore * 100).toFixed(1)}%`);

        if (d.isOptimal) {
          text.append('tspan')
            .attr('x', 20)
            .attr('dy', '14')
            .attr('fill', '#ff6b6b')
            .text('✓ Pareto Optimal');
        }
      })
      .on('mouseout', function() {
        g.selectAll('.tooltip').remove();
      })
      .on('click', function(event, d) {
        setSelectedPoint(d);
        if (onOptimalConfigurationSelected && d.isOptimal) {
          onOptimalConfigurationSelected(d);
        }
      });

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 40)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('Transform Complexity Score');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -50)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('K2 Coefficient Value');

    // Legend
    const legend = g.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${innerWidth + 20}, 20)`);

    const providers = Array.from(new Set(filteredData.map(d => d.provider)));
    
    providers.forEach((provider, i) => {
      const legendGroup = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      legendGroup.append('circle')
        .attr('r', 6)
        .attr('fill', colorScale(provider) as string);

      legendGroup.append('text')
        .attr('x', 12)
        .attr('y', 4)
        .text(provider)
        .style('font-size', '12px');
    });

    // Pareto frontier legend
    legend.append('g')
      .attr('transform', `translate(0, ${providers.length * 20 + 10})`)
      .call(g => {
        g.append('circle')
          .attr('r', 6)
          .attr('fill', '#ff6b6b')
          .attr('stroke', '#ffffff')
          .attr('stroke-width', 2);

        g.append('text')
          .attr('x', 12)
          .attr('y', 4)
          .text('Pareto Optimal')
          .style('font-size', '12px')
          .style('font-weight', 'bold');
      });

  }, [filteredData, paretoAnalysis, viewMode]);

  // Helper functions
  const calculateConvergenceRate = (frontierPoints: ParetoParetoPoint[]): number => {
    if (frontierPoints.length < 2) return 0;
    
    const sorted = frontierPoints.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
    let improvements = 0;
    
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].performanceScore > sorted[i-1].performanceScore) {
        improvements++;
      }
    }
    
    return improvements / (sorted.length - 1);
  };

  const calculateStabilityIndex = (frontierPoints: ParetoParetoPoint[]): number => {
    if (frontierPoints.length < 2) return 1;
    
    const complexities = frontierPoints.map(p => p.complexity);
    const mean = complexities.reduce((a, b) => a + b, 0) / complexities.length;
    const variance = complexities.reduce((acc, x) => acc + Math.pow(x - mean, 2), 0) / complexities.length;
    
    return 1 / (1 + Math.sqrt(variance));
  };

  const handleExport = (format: 'png' | 'svg' | 'json' | 'csv') => {
    if (onExport) {
      onExport(format);
    }
  };

  const resetView = () => {
    setZoomLevel(1);
    setSelectedPoint(null);
  };

  return (
    <div className={clsx('bg-white rounded-lg border shadow-sm', className)}>
      {/* Header */}
      <div className="p-4 border-b bg-gray-50 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            <h3 className="text-lg font-semibold text-gray-900">
              Complexity vs K2 Pareto Frontier Analysis
            </h3>
            {realTimeEnabled && (
              <div className="flex items-center gap-1 text-green-600 text-sm">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Live</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={resetView}
              className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-md text-sm transition-colors"
            >
              Reset View
            </button>
            <button
              onClick={() => handleExport('png')}
              className="px-3 py-1.5 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-md text-sm transition-colors flex items-center gap-1"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="p-4 border-b bg-gray-50">
        <div className="flex flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">View:</label>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as any)}
              className="border rounded px-2 py-1 text-sm"
            >
              <option value="frontier">Pareto Frontier</option>
              <option value="evolution">Evolution</option>
              <option value="sensitivity">Sensitivity</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Provider:</label>
            <select
              value={filterProvider}
              onChange={(e) => setFilterProvider(e.target.value)}
              className="border rounded px-2 py-1 text-sm"
            >
              <option value="all">All Providers</option>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="google">Google</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Model Size:</label>
            <select
              value={filterModelSize}
              onChange={(e) => setFilterModelSize(e.target.value)}
              className="border rounded px-2 py-1 text-sm"
            >
              <option value="all">All Sizes</option>
              <option value="256">256 Dimensions</option>
              <option value="768">768 Dimensions</option>
            </select>
          </div>
        </div>
      </div>

      {/* Metrics Dashboard */}
      {paretoAnalysis.metrics && (
        <div className="p-4 border-b bg-blue-50">
          <div className="grid grid-cols-5 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {paretoAnalysis.metrics.frontierSize}
              </div>
              <div className="text-sm text-gray-600">Optimal Solutions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {(paretoAnalysis.metrics.dominationRatio * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Dominated Points</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {(paretoAnalysis.metrics.optimalityGap * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Optimality Gap</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(paretoAnalysis.metrics.convergenceRate * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Convergence Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-indigo-600">
                {(paretoAnalysis.metrics.stabilityIndex * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Stability Index</div>
            </div>
          </div>
        </div>
      )}

      {/* Visualization */}
      <div className="relative">
        <div ref={containerRef} className="w-full h-96">
          <svg
            ref={svgRef}
            className="w-full h-full"
            style={{ minHeight: '400px' }}
          />
        </div>
      </div>

      {/* Selected Point Details */}
      {selectedPoint && (
        <div className="p-4 border-t bg-gray-50">
          <h4 className="font-medium text-gray-900 mb-2">Selected Configuration</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="font-medium">Provider:</span> {selectedPoint.provider}
            </div>
            <div>
              <span className="font-medium">Model Size:</span> {selectedPoint.modelSize}
            </div>
            <div>
              <span className="font-medium">Complexity:</span> {selectedPoint.complexity.toFixed(3)}
            </div>
            <div>
              <span className="font-medium">K2 Coefficient:</span> {selectedPoint.k2Coefficient.toFixed(3)}
            </div>
            <div>
              <span className="font-medium">Performance:</span> {(selectedPoint.performanceScore * 100).toFixed(1)}%
            </div>
            <div>
              <span className="font-medium">Success Rate:</span> {(selectedPoint.metadata.successRate * 100).toFixed(1)}%
            </div>
          </div>
          {selectedPoint.isOptimal && (
            <div className="mt-2 text-sm text-green-600 font-medium">
              ✓ This configuration is Pareto optimal
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ParetoFrontierAnalysis;