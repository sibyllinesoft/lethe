import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { TimelineDataPoint } from '../types/monitoring';
import { ChartCard, chartColors, designTokens } from '../design-system';

interface TimelineChartProps {
  data: TimelineDataPoint[];
  height?: number;
  className?: string;
}

/**
 * Timeline chart showing execution metrics over time
 * Matches the visualization from PromptDashboard.create_timeline_chart()
 * Now using sophisticated graphite design system styling
 */
export const TimelineChart: React.FC<TimelineChartProps> = ({ 
  data, 
  height = 400,
  className 
}) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const formatExecutionTime = (value: number) => {
    return `${Math.round(value)}ms`;
  };

  const formatNumber = (value: number) => {
    return value.toLocaleString();
  };

  return (
    <ChartCard 
      title="Execution Timeline" 
      subtitle="Performance metrics over time"
      className={className}
    >
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={designTokens.colors.graphite[200]}
            strokeWidth={1}
          />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatDate}
            stroke={designTokens.colors.graphite[500]}
            fontSize={11}
            fontFamily={designTokens.typography.fontFamily.primary.join(', ')}
            axisLine={false}
            tickLine={false}
            dy={10}
          />
          <YAxis 
            stroke={designTokens.colors.graphite[500]}
            fontSize={11}
            fontFamily={designTokens.typography.fontFamily.primary.join(', ')}
            axisLine={false}
            tickLine={false}
            dx={-10}
          />
          <Tooltip 
            formatter={(value: number, name: string) => {
              switch (name) {
                case 'total_executions':
                  return [formatNumber(value), 'Total Executions'];
                case 'avg_execution_time':
                  return [formatExecutionTime(value), 'Avg Execution Time'];
                case 'avg_response_length':
                  return [`${Math.round(value)} chars`, 'Avg Response Length'];
                case 'errors':
                  return [formatNumber(value), 'Errors'];
                default:
                  return [value, name];
              }
            }}
            labelFormatter={(date: string) => `Date: ${formatDate(date)}`}
            contentStyle={{
              backgroundColor: designTokens.colors.graphite[900],
              color: designTokens.colors.graphite[100],
              border: `1px solid ${designTokens.colors.graphite[700]}`,
              borderRadius: '8px',
              boxShadow: designTokens.shadows.large,
              fontSize: '12px',
              fontFamily: designTokens.typography.fontFamily.primary.join(', ')
            }}
          />
          <Legend 
            wrapperStyle={{
              paddingTop: '16px',
              fontSize: '12px',
              fontFamily: designTokens.typography.fontFamily.primary.join(', ')
            }}
          />
          <Line 
            type="monotone" 
            dataKey="total_executions" 
            stroke={chartColors.primary[0]} 
            strokeWidth={2.5}
            dot={{ fill: designTokens.colors.surface.primary, stroke: chartColors.primary[0], strokeWidth: 2, r: 4 }}
            activeDot={{ fill: chartColors.primary[0], stroke: designTokens.colors.surface.primary, strokeWidth: 2, r: 6 }}
            name="Total Executions"
          />
          <Line 
            type="monotone" 
            dataKey="avg_execution_time" 
            stroke={chartColors.primary[4]} 
            strokeWidth={2.5}
            dot={{ fill: designTokens.colors.surface.primary, stroke: chartColors.primary[4], strokeWidth: 2, r: 4 }}
            activeDot={{ fill: chartColors.primary[4], stroke: designTokens.colors.surface.primary, strokeWidth: 2, r: 6 }}
            name="Avg Execution Time (ms)"
          />
          <Line 
            type="monotone" 
            dataKey="errors" 
            stroke={chartColors.error[0]} 
            strokeWidth={2.5}
            dot={{ fill: designTokens.colors.surface.primary, stroke: chartColors.error[0], strokeWidth: 2, r: 4 }}
            activeDot={{ fill: chartColors.error[0], stroke: designTokens.colors.surface.primary, strokeWidth: 2, r: 6 }}
            name="Errors"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
};

export default TimelineChart;