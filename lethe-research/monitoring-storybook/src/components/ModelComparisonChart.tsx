import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart
} from 'recharts';
import { ModelComparison } from '../types/monitoring';
import { ChartCard, chartColors, designTokens, Text } from '../design-system';

interface ModelComparisonChartProps {
  data: ModelComparison[];
  height?: number;
  className?: string;
}

/**
 * Model performance comparison chart
 * Matches the visualization from PromptDashboard.create_model_comparison_chart()
 * Combines bar chart for execution time with line chart for quality score
 */
export const ModelComparisonChart: React.FC<ModelComparisonChartProps> = ({ 
  data, 
  height = 400,
  className 
}) => {
  const formatModelName = (name: string) => {
    // Shorten model names for better display
    return name
      .replace('gpt-4-turbo-preview', 'GPT-4 Turbo')
      .replace('gpt-3.5-turbo', 'GPT-3.5')
      .replace('claude-3-opus-20240229', 'Claude Opus')
      .replace('claude-3-sonnet-20240229', 'Claude Sonnet')
      .replace('claude-3-haiku-20240307', 'Claude Haiku')
      .replace('gemini-pro', 'Gemini Pro');
  };

  const chartData = data.map(item => ({
    ...item,
    model_display: formatModelName(item.model_name),
    quality_score_scaled: (item.avg_quality_score || 0) * 1000 // Scale quality score for dual axis
  }));

  const formatExecutionTime = (value: number) => {
    return `${Math.round(value)}ms`;
  };

  const formatQualityScore = (value: number) => {
    return (value / 1000).toFixed(3);
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
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
          <Text weight="semibold" className="mb-2 text-graphite-100">
            {data.model_name}
          </Text>
          <div className="space-y-1">
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Executions:</span> {data.execution_count.toLocaleString()}
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Avg Time:</span> {formatExecutionTime(data.avg_execution_time)}
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Quality Score:</span> {(data.avg_quality_score || 0).toFixed(3)}
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Avg Length:</span> {Math.round(data.avg_response_length)} chars
            </Text>
            <Text size="sm" className="text-graphite-200">
              <span className="font-medium">Errors:</span> {data.error_count}
            </Text>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <ChartCard 
      title="Model Performance Comparison"
      subtitle="Execution time vs quality score analysis"
      className={className}
    >
      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-sm" 
              style={{ backgroundColor: chartColors.primary[0] }}
            />
            <Text size="sm" color="secondary">Execution Time (bars)</Text>
          </div>
          <div className="flex items-center gap-2">
            <div 
              className="w-3 h-1 rounded-full" 
              style={{ backgroundColor: chartColors.error[0] }}
            />
            <Text size="sm" color="secondary">Quality Score (line)</Text>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={designTokens.colors.graphite[200]}
            strokeWidth={1}
          />
          <XAxis 
            dataKey="model_display" 
            stroke={designTokens.colors.graphite[500]}
            fontSize={11}
            fontFamily={designTokens.typography.fontFamily.primary.join(', ')}
            angle={-45}
            textAnchor="end"
            height={80}
            axisLine={false}
            tickLine={false}
          />
          <YAxis 
            yAxisId="left"
            orientation="left"
            stroke={designTokens.colors.graphite[500]}
            fontSize={11}
            fontFamily={designTokens.typography.fontFamily.primary.join(', ')}
            tickFormatter={formatExecutionTime}
            axisLine={false}
            tickLine={false}
          />
          <YAxis 
            yAxisId="right"
            orientation="right"
            stroke={designTokens.colors.graphite[500]}
            fontSize={11}
            fontFamily={designTokens.typography.fontFamily.primary.join(', ')}
            tickFormatter={formatQualityScore}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          
          <Bar 
            yAxisId="left"
            dataKey="avg_execution_time" 
            fill={chartColors.primary[0]}
            radius={[3, 3, 0, 0]}
          />
          
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="quality_score_scaled"
            stroke={chartColors.error[0]}
            strokeWidth={2.5}
            dot={{ fill: designTokens.colors.surface.primary, stroke: chartColors.error[0], strokeWidth: 2, r: 4 }}
            activeDot={{ fill: chartColors.error[0], stroke: designTokens.colors.surface.primary, strokeWidth: 2, r: 6 }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-6 space-y-1">
        <Text size="sm" color="secondary">
          <span className="font-medium text-graphite-700">Left axis (bars):</span> Average execution time in milliseconds
        </Text>
        <Text size="sm" color="secondary">
          <span className="font-medium text-graphite-700">Right axis (line):</span> Average quality score (0-1 scale)
        </Text>
        <Text size="sm" color="secondary">
          <span className="font-medium text-graphite-700">Goal:</span> Lower execution time with higher quality score
        </Text>
      </div>
    </ChartCard>
  );
};

export default ModelComparisonChart;