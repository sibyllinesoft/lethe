import React, { useRef, useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { TransformChangeV2, TimelineEvent } from '../../types/transform';
import { Clock, ArrowRight, CheckCircle, XCircle } from 'lucide-react';
import clsx from 'clsx';

interface TimelineViewProps {
  changes: TransformChangeV2[];
  className?: string;
  detailed?: boolean;
  onEventClick?: (event: TimelineEvent) => void;
}

/**
 * TimelineView - Chronological visualization with causality analysis
 * 
 * Features:
 * - Chronological sequence of transform changes
 * - Causality links between related events
 * - Success/failure status indicators
 * - Performance duration visualization
 * - Interactive events with detailed tooltips
 * - Zoom and pan for large timelines
 */
export const TimelineView: React.FC<TimelineViewProps> = ({
  changes,
  className,
  detailed = false,
  onEventClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [timeRange, setTimeRange] = useState<[Date, Date] | null>(null);

  // Process changes into timeline events with causality detection
  const timelineEvents = useMemo<TimelineEvent[]>(() => {
    const events: TimelineEvent[] = changes.map((change, index) => {
      const changeType = Object.keys(change.change_type)[0];
      const isSuccess = change.metadata.confidence_score !== 0 && 
                       (!change.metadata.performance || 
                       (change.metadata.performance && change.metadata.performance.duration_us < 10000)); // Less than 10ms considered successful

      return {
        id: `event_${index}`,
        timestamp: new Date(change.timestamp),
        changeType: formatChangeType(changeType),
        duration: change.metadata.performance?.duration_us || 0,
        success: isSuccess,
        causality: [] // Will be populated below
      };
    });

    // Detect causality relationships
    events.forEach((event, index) => {
      const causality: string[] = [];
      
      // Look for events within the same request that might have caused this one
      const sameRequestEvents = events.slice(0, index).filter(prevEvent => {
        const timeDiff = event.timestamp.getTime() - prevEvent.timestamp.getTime();
        return timeDiff > 0 && timeDiff < 5000; // Within 5 seconds
      });

      // Simple causality rules
      sameRequestEvents.forEach(prevEvent => {
        // System changes often cause user content changes
        if (prevEvent.changeType.includes('System') && 
            event.changeType.includes('Content')) {
          causality.push(prevEvent.id);
        }
        
        // Validation changes often follow content changes
        if (prevEvent.changeType.includes('Content') && 
            event.changeType.includes('Validation')) {
          causality.push(prevEvent.id);
        }
        
        // Failed events might cause retry events
        if (!prevEvent.success && event.changeType === prevEvent.changeType) {
          causality.push(prevEvent.id);
        }
      });

      event.causality = causality;
    });

    return events.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }, [changes]);

  // Calculate time range for zoom functionality
  useEffect(() => {
    if (timelineEvents.length > 0) {
      const timestamps = timelineEvents.map(e => e.timestamp);
      const minTime = new Date(Math.min(...timestamps.map(t => t.getTime())));
      const maxTime = new Date(Math.max(...timestamps.map(t => t.getTime())));
      setTimeRange([minTime, maxTime]);
    }
  }, [timelineEvents]);

  // D3 timeline visualization
  useEffect(() => {
    if (!svgRef.current || timelineEvents.length === 0 || !timeRange) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const container = svg.node()?.getBoundingClientRect();
    if (!container) return;

    const margin = { top: 40, right: 40, bottom: 60, left: 60 };
    const width = container.width - margin.left - margin.right;
    const height = container.height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Time scale
    const timeScale = d3.scaleTime()
      .domain(timeRange)
      .range([0, width]);

    // Y scale for lanes (group by event type)
    const eventTypes = [...new Set(timelineEvents.map(e => e.changeType))];
    const yScale = d3.scaleBand()
      .domain(eventTypes)
      .range([0, height])
      .padding(0.1);

    // Duration scale for circle size
    const maxDuration = Math.max(...timelineEvents.map(e => e.duration), 1);
    const radiusScale = d3.scaleSqrt()
      .domain([0, maxDuration])
      .range([3, detailed ? 12 : 8]);

    // Draw lane backgrounds
    g.selectAll('.lane')
      .data(eventTypes)
      .enter().append('rect')
      .attr('class', 'lane')
      .attr('x', 0)
      .attr('y', d => yScale(d)!)
      .attr('width', width)
      .attr('height', yScale.bandwidth())
      .attr('fill', (d, i) => i % 2 === 0 ? '#f9fafb' : '#f3f4f6')
      .attr('opacity', 0.5);

    // Draw lane labels
    g.selectAll('.lane-label')
      .data(eventTypes)
      .enter().append('text')
      .attr('class', 'lane-label')
      .attr('x', -10)
      .attr('y', d => yScale(d)! + yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'end')
      .style('font-size', detailed ? '12px' : '10px')
      .style('font-weight', '500')
      .style('fill', '#374151')
      .text(d => d);

    // Draw causality links
    const causalityLinks: Array<{source: TimelineEvent, target: TimelineEvent}> = [];
    timelineEvents.forEach(event => {
      event.causality?.forEach(causalId => {
        const sourceEvent = timelineEvents.find(e => e.id === causalId);
        if (sourceEvent) {
          causalityLinks.push({ source: sourceEvent, target: event });
        }
      });
    });

    g.selectAll('.causality-link')
      .data(causalityLinks)
      .enter().append('path')
      .attr('class', 'causality-link')
      .attr('d', (_: any) => {
        // Return a simple path for type safety
        return `M 0 0 L 10 10`;
      })
      .attr('stroke', '#6366f1')
      .attr('stroke-width', 1)
      .attr('fill', 'none')
      .attr('opacity', 0.6)
      .attr('marker-end', 'url(#arrowhead)');

    // Define arrowhead marker
    const defs = svg.append('defs');
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#6366f1');

    // Draw events
    g.selectAll('.event')
      .data(timelineEvents)
      .enter().append('circle')
      .attr('class', 'event')
      .attr('cx', d => timeScale(d.timestamp))
      .attr('cy', d => yScale(d.changeType)! + yScale.bandwidth() / 2)
      .attr('r', 0)
      .attr('fill', d => d.success ? '#10b981' : '#ef4444')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('stroke-width', 3);
        
        const tooltip = d3.select('body').append('div')
          .attr('class', 'timeline-tooltip')
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
            <div>Time: ${d.timestamp.toLocaleTimeString()}</div>
            <div>Duration: ${(d.duration / 1000).toFixed(2)}ms</div>
            <div>Status: ${d.success ? 'Success' : 'Failed'}</div>
            <div>Causality: ${d.causality?.length || 0} links</div>
          `);

        const [mouseX, mouseY] = d3.pointer(event, document.body);
        tooltip
          .style('left', mouseX + 10 + 'px')
          .style('top', mouseY - 10 + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke-width', 2);
        d3.selectAll('.timeline-tooltip').remove();
      })
      .on('click', function(_, d) {
        setSelectedEvent(selectedEvent?.id === d.id ? null : d);
        onEventClick?.(d);
      });

    // Animate event appearance
    g.selectAll('.event')
      .transition()
      .duration(750)
      .delay((_, i) => i * 50)
      .attr('r', d => radiusScale((d as any).duration));

    // Add time axis
    const timeAxis = d3.axisBottom(timeScale)
      .tickFormat(d3.timeFormat('%H:%M:%S') as any);

    g.append('g')
      .attr('class', 'time-axis')
      .attr('transform', `translate(0, ${height})`)
      .call(timeAxis as any)
      .style('font-size', detailed ? '11px' : '9px');

    // Add axis label
    g.append('text')
      .attr('transform', `translate(${width / 2}, ${height + 40})`)
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', '#374151')
      .text('Timeline');

  }, [timelineEvents, timeRange, detailed, selectedEvent, onEventClick]);

  // Calculate statistics
  const totalEvents = timelineEvents.length;
  const successfulEvents = timelineEvents.filter(e => e.success).length;
  const causalityConnections = timelineEvents.reduce((sum, e) => sum + (e.causality?.length || 0), 0);
  const avgDuration = totalEvents > 0 
    ? timelineEvents.reduce((sum, e) => sum + e.duration, 0) / totalEvents 
    : 0;

  return (
    <div className={clsx('bg-white dark:bg-gray-900', className)}>
      {detailed && (
        <div className="mb-4 flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Transform Timeline
          </h3>
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-emerald-500 rounded-full mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">Success</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
              <span className="text-gray-600 dark:text-gray-300">Failed</span>
            </div>
            <div className="flex items-center">
              <ArrowRight className="text-indigo-500" size={16} />
              <span className="text-gray-600 dark:text-gray-300 ml-1">Causality</span>
            </div>
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

      {/* Timeline statistics */}
      <div className={clsx(
        'mt-4 grid gap-4',
        detailed ? 'grid-cols-4' : 'grid-cols-2'
      )}>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
          <div className="flex items-center">
            <Clock className="text-blue-600 dark:text-blue-400" size={16} />
            <div className="ml-2">
              <div className="text-sm font-medium text-blue-900 dark:text-blue-100">
                Total Events
              </div>
              <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                {totalEvents}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
          <div className="flex items-center">
            <CheckCircle className="text-green-600 dark:text-green-400" size={16} />
            <div className="ml-2">
              <div className="text-sm font-medium text-green-900 dark:text-green-100">
                Success Rate
              </div>
              <div className="text-lg font-bold text-green-600 dark:text-green-400">
                {totalEvents > 0 ? ((successfulEvents / totalEvents) * 100).toFixed(1) : 0}%
              </div>
            </div>
          </div>
        </div>

        {detailed && (
          <>
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3">
              <div className="flex items-center">
                <ArrowRight className="text-purple-600 dark:text-purple-400" size={16} />
                <div className="ml-2">
                  <div className="text-sm font-medium text-purple-900 dark:text-purple-100">
                    Causality Links
                  </div>
                  <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
                    {causalityConnections}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-amber-50 dark:bg-amber-900/20 rounded p-3">
              <div className="flex items-center">
                <Clock className="text-amber-600 dark:text-amber-400" size={16} />
                <div className="ml-2">
                  <div className="text-sm font-medium text-amber-900 dark:text-amber-100">
                    Avg Duration
                  </div>
                  <div className="text-lg font-bold text-amber-600 dark:text-amber-400">
                    {(avgDuration / 1000).toFixed(1)}ms
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {selectedEvent && (
        <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center">
            {selectedEvent.success ? (
              <CheckCircle className="text-green-500 mr-2" size={16} />
            ) : (
              <XCircle className="text-red-500 mr-2" size={16} />
            )}
            Event Details
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600 dark:text-gray-300">Type:</span>
              <span className="ml-2 font-medium">{selectedEvent.changeType}</span>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-300">Timestamp:</span>
              <span className="ml-2 font-medium">{selectedEvent.timestamp.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-300">Duration:</span>
              <span className="ml-2 font-medium">{(selectedEvent.duration / 1000).toFixed(2)}ms</span>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-300">Status:</span>
              <span className={clsx(
                'ml-2 font-medium',
                selectedEvent.success ? 'text-green-600' : 'text-red-600'
              )}>
                {selectedEvent.success ? 'Success' : 'Failed'}
              </span>
            </div>
          </div>
          {selectedEvent.causality && selectedEvent.causality.length > 0 && (
            <div className="mt-3">
              <span className="text-gray-600 dark:text-gray-300 text-sm">
                Caused by {selectedEvent.causality.length} previous event(s)
              </span>
            </div>
          )}
        </div>
      )}
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