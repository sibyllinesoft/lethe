import type { Meta, StoryObj } from '@storybook/react';
import { MonitoringDashboard } from './MonitoringDashboard';
import { mockDataset } from '../data/mockData';

const meta: Meta<typeof MonitoringDashboard> = {
  title: 'Pages/MonitoringDashboard',
  component: MonitoringDashboard,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Complete monitoring dashboard interface combining all components into a tabbed, responsive dashboard for prompt monitoring analytics.'
      }
    }
  },
  tags: ['autodocs']
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    summaryStats: mockDataset.summaryStats,
    timelineData: mockDataset.timelineData,
    promptPerformance: mockDataset.promptPerformance,
    modelComparison: mockDataset.modelComparison,
    executionComparison: mockDataset.executionComparison,
    cliOutputs: mockDataset.cliOutputs
  }
};

export const HighVolumeSystem: Story = {
  args: {
    summaryStats: {
      total_executions: 125678,
      unique_prompts: 456,
      success_rate: 98.7,
      avg_execution_time_ms: 189,
      recent_executions_24h: 2341
    },
    timelineData: [
      { date: '2024-01-09', total_executions: 1234, avg_execution_time: 156, avg_response_length: 1100, errors: 12 },
      { date: '2024-01-10', total_executions: 1456, avg_execution_time: 167, avg_response_length: 1200, errors: 8 },
      { date: '2024-01-11', total_executions: 1789, avg_execution_time: 145, avg_response_length: 1150, errors: 5 },
      { date: '2024-01-12', total_executions: 2012, avg_execution_time: 134, avg_response_length: 1080, errors: 3 },
      { date: '2024-01-13', total_executions: 1987, avg_execution_time: 142, avg_response_length: 1220, errors: 7 },
      { date: '2024-01-14', total_executions: 2134, avg_execution_time: 128, avg_response_length: 1300, errors: 2 },
      { date: '2024-01-15', total_executions: 2341, avg_execution_time: 119, avg_response_length: 1250, errors: 1 }
    ],
    promptPerformance: Array.from({ length: 15 }, (_, i) => ({
      prompt_id: `high_volume_prompt_${String(i + 1).padStart(3, '0')}`,
      execution_count: Math.floor(Math.random() * 1000) + 500,
      avg_execution_time: Math.floor(Math.random() * 300) + 100,
      avg_response_length: Math.floor(Math.random() * 800) + 600,
      avg_quality_score: 0.85 + Math.random() * 0.15,
      last_used: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      error_count: Math.floor(Math.random() * 10),
      success_rate: 95 + Math.random() * 5
    })),
    modelComparison: mockDataset.modelComparison,
    executionComparison: mockDataset.executionComparison,
    cliOutputs: mockDataset.cliOutputs
  }
};

export const StartupSystem: Story = {
  args: {
    summaryStats: {
      total_executions: 342,
      unique_prompts: 15,
      success_rate: 92.1,
      avg_execution_time_ms: 567,
      recent_executions_24h: 23
    },
    timelineData: [
      { date: '2024-01-09', total_executions: 12, avg_execution_time: 890, avg_response_length: 800, errors: 2 },
      { date: '2024-01-10', total_executions: 18, avg_execution_time: 756, avg_response_length: 950, errors: 1 },
      { date: '2024-01-11', total_executions: 25, avg_execution_time: 634, avg_response_length: 1100, errors: 0 },
      { date: '2024-01-12', total_executions: 31, avg_execution_time: 598, avg_response_length: 1050, errors: 3 },
      { date: '2024-01-13', total_executions: 28, avg_execution_time: 612, avg_response_length: 980, errors: 1 },
      { date: '2024-01-14', total_executions: 34, avg_execution_time: 534, avg_response_length: 1200, errors: 0 },
      { date: '2024-01-15', total_executions: 23, avg_execution_time: 567, avg_response_length: 1150, errors: 1 }
    ],
    promptPerformance: Array.from({ length: 5 }, (_, i) => ({
      prompt_id: `startup_prompt_${String(i + 1).padStart(2, '0')}`,
      execution_count: Math.floor(Math.random() * 50) + 10,
      avg_execution_time: Math.floor(Math.random() * 500) + 300,
      avg_response_length: Math.floor(Math.random() * 600) + 400,
      avg_quality_score: 0.75 + Math.random() * 0.25,
      last_used: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      error_count: Math.floor(Math.random() * 5),
      success_rate: 85 + Math.random() * 15
    })),
    modelComparison: [
      {
        model_name: 'gpt-3.5-turbo',
        execution_count: 234,
        avg_execution_time: 456,
        avg_response_length: 890,
        avg_quality_score: 0.82,
        error_count: 12
      },
      {
        model_name: 'claude-3-haiku-20240307',
        execution_count: 108,
        avg_execution_time: 234,
        avg_response_length: 1100,
        avg_quality_score: 0.87,
        error_count: 3
      }
    ],
    executionComparison: mockDataset.executionComparison,
    cliOutputs: mockDataset.cliOutputs
  }
};