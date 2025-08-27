import React from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { PromptPerformance } from '../types/monitoring';
import { ChartCard, performanceColors, designTokens, Text, Badge } from '../design-system';

interface PerformanceBubbleChartProps {
  data: PromptPerformance[];
  height?: number;
  className?: string;
}

/**
 * Bubble chart showing prompt performance metrics
 * Based on PromptDashboard.create_prompt_performance_chart()
 * Now using sophisticated graphite design system colors
 * X-axis: Average execution time
 * Y-axis: Execution count
 * Bubble size: Success rate
 * Color: Quality score
 */
export const PerformanceBubbleChart: React.FC<PerformanceBubbleChartProps> = ({ 
  data, 
  height = 500,
  className 
}) => {
  // Transform data for scatter plot
  const scatterData = data.map((item, index) => ({
    x: item.avg_execution_time,
    y: item.execution_count,
    z: item.success_rate,
    quality: item.avg_quality_score || 0,
    promptId: item.prompt_id,
    errorCount: item.error_count,
    responseLength: item.avg_response_length
  }));

  const getColor = (quality: number): string => {
    if (quality >= 0.9) return performanceColors.excellent;
    if (quality >= 0.8) return performanceColors.good;
    if (quality >= 0.7) return performanceColors.fair;
    if (quality >= 0.6) return performanceColors.poor;
    return performanceColors.critical;
  };

  const formatExecutionTime = (value: number) => {
    return `${Math.round(value)}ms`;
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div 
          style={{
            backgroundColor: designTokens.colors.graphite[900],
            color: designTokens.colors.graphite[100],
            border: `1px solid ${designTokens.colors.graphite[700]}`,
            borderRadius: '8px',
            boxShadow: designTokens.shadows.large,
            padding: '12px',
            fontSize: '12px',
            fontFamily: designTokens.typography.fontFamily.primary.join(', ')
          }}
        >
          <Text color="secondary" weight="semibold" className="mb-2 text-graphite-100">
            {data.promptId}
          </Text>
          <div className="space-y-1">
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Executions:</span> {data.y.toLocaleString()}
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Avg Time:</span> {formatExecutionTime(data.x)}
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Success Rate:</span> {data.z.toFixed(1)}%
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Quality Score:</span> {data.quality.toFixed(3)}
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Avg Length:</span> {Math.round(data.responseLength)} chars
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Errors:</span> {data.errorCount}
            </Text>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <ChartCard 
      title="Prompt Performance Overview"
      subtitle="Quality metrics and execution patterns"
      className={className}
    >
      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: performanceColors.excellent }}
            />
            <Text size="sm" color="secondary">Excellent (&gt;0.9)</Text>
          </div>
          <div className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: performanceColors.good }}
            />
            <Text size="sm" color="secondary">Good (0.8-0.9)</Text>
          </div>
          <div className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: performanceColors.fair }}
            />
            <Text size="sm" color="secondary">Fair (0.7-0.8)</Text>
          </div>
          <div className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: performanceColors.poor }}
            />
            <Text size="sm" color="secondary">Poor (0.6-0.7)</Text>
          </div>
          <div className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: performanceColors.critical }}
            />
            <Text size="sm" color="secondary">Critical (&lt;0.6)</Text>
          </div>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={designTokens.colors.graphite[200]}
            strokeWidth={1}
          />
          <XAxis 
            type="number" 
            dataKey="x"
            name="Avg Execution Time"
            unit="ms"
            tickFormatter={formatExecutionTime}
            stroke={designTokens.colors.graphite[500]}
            fontSize={11}
            fontFamily={designTokens.typography.fontFamily.primary.join(', ')}
            axisLine={false}
            tickLine={false}
          />
          <YAxis 
            type="number" 
            dataKey="y"
            name="Total Executions"
            stroke={designTokens.colors.graphite[500]}
            fontSize={11}
            fontFamily={designTokens.typography.fontFamily.primary.join(', ')}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Scatter data={scatterData}>
            {scatterData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={getColor(entry.quality)}
                fillOpacity={0.7}
                stroke={getColor(entry.quality)}
                strokeWidth={1}
                r={Math.max(6, Math.min(24, entry.z / 4))} // Size based on success rate
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div className="space-y-1">
          <Text size="sm" weight="medium" color="primary">Chart Guide:</Text>
          <Text size="sm" color="secondary">X-axis: Average execution time (lower is better)</Text>
          <Text size="sm" color="secondary">Y-axis: Total executions (higher indicates more usage)</Text>
        </div>
        <div className="space-y-1">
          <Text size="sm" weight="medium" color="primary">Visual Encoding:</Text>
          <Text size="sm" color="secondary">Bubble size: Success rate (larger = higher success)</Text>
          <Text size="sm" color="secondary">Color: Quality score (green = excellent, red = critical)</Text>
        </div>
      </div>
    </ChartCard>
  );
};

export default PerformanceBubbleChart;