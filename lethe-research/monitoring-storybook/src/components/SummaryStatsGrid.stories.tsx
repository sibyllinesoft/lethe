import type { Meta, StoryObj } from '@storybook/react';
import { SummaryStatsGrid } from './SummaryStatsGrid';
import { mockDataset } from '../data/mockData';

const meta: Meta<typeof SummaryStatsGrid> = {
  title: 'Components/SummaryStatsGrid',
  component: SummaryStatsGrid,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'Grid of summary statistics cards showing key metrics from the prompt monitoring system.'
      }
    }
  },
  tags: ['autodocs']
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    stats: mockDataset.summaryStats
  }
};

export const HighPerformance: Story = {
  args: {
    stats: {
      total_executions: 45678,
      unique_prompts: 234,
      success_rate: 99.2,
      avg_execution_time_ms: 156,
      recent_executions_24h: 892
    }
  }
};

export const LowPerformance: Story = {
  args: {
    stats: {
      total_executions: 1234,
      unique_prompts: 45,
      success_rate: 87.3,
      avg_execution_time_ms: 2340,
      recent_executions_24h: 23
    }
  }
};

export const StartupMetrics: Story = {
  args: {
    stats: {
      total_executions: 156,
      unique_prompts: 12,
      success_rate: 94.8,
      avg_execution_time_ms: 890,
      recent_executions_24h: 34
    }
  }
};