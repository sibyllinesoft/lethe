import type { Meta, StoryObj } from '@storybook/react';
import { ModelComparisonChart } from './ModelComparisonChart';
import { mockDataset } from '../data/mockData';

const meta: Meta<typeof ModelComparisonChart> = {
  title: 'Components/ModelComparisonChart',
  component: ModelComparisonChart,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'Combined bar and line chart comparing performance metrics across different AI models, showing execution time (bars) and quality scores (line).'
      }
    }
  },
  tags: ['autodocs'],
  argTypes: {
    height: {
      control: { type: 'range', min: 300, max: 800, step: 50 }
    }
  }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    data: mockDataset.modelComparison,
    height: 400
  }
};

export const OpenAIModels: Story = {
  args: {
    data: [
      {
        model_name: 'gpt-4-turbo-preview',
        execution_count: 1456,
        avg_execution_time: 234,
        avg_response_length: 1234,
        avg_quality_score: 0.94,
        error_count: 12
      },
      {
        model_name: 'gpt-4',
        execution_count: 987,
        avg_execution_time: 456,
        avg_response_length: 1456,
        avg_quality_score: 0.92,
        error_count: 18
      },
      {
        model_name: 'gpt-3.5-turbo',
        execution_count: 2134,
        avg_execution_time: 189,
        avg_response_length: 890,
        avg_quality_score: 0.85,
        error_count: 34
      }
    ],
    height: 400
  }
};

export const AnthropicModels: Story = {
  args: {
    data: [
      {
        model_name: 'claude-3-opus-20240229',
        execution_count: 567,
        avg_execution_time: 345,
        avg_response_length: 1567,
        avg_quality_score: 0.96,
        error_count: 8
      },
      {
        model_name: 'claude-3-sonnet-20240229',
        execution_count: 892,
        avg_execution_time: 278,
        avg_response_length: 1234,
        avg_quality_score: 0.91,
        error_count: 15
      },
      {
        model_name: 'claude-3-haiku-20240307',
        execution_count: 1234,
        avg_execution_time: 156,
        avg_response_length: 789,
        avg_quality_score: 0.87,
        error_count: 23
      }
    ],
    height: 400
  }
};

export const MixedPerformance: Story = {
  args: {
    data: [
      {
        model_name: 'high-speed-model',
        execution_count: 1000,
        avg_execution_time: 89,
        avg_response_length: 600,
        avg_quality_score: 0.75,
        error_count: 45
      },
      {
        model_name: 'balanced-model',
        execution_count: 800,
        avg_execution_time: 234,
        avg_response_length: 1200,
        avg_quality_score: 0.89,
        error_count: 12
      },
      {
        model_name: 'high-quality-model',
        execution_count: 500,
        avg_execution_time: 567,
        avg_response_length: 1800,
        avg_quality_score: 0.97,
        error_count: 3
      },
      {
        model_name: 'problematic-model',
        execution_count: 200,
        avg_execution_time: 2340,
        avg_response_length: 1000,
        avg_quality_score: 0.62,
        error_count: 67
      }
    ],
    height: 500
  }
};

export const SingleModel: Story = {
  args: {
    data: [
      {
        model_name: 'gpt-4-turbo-preview',
        execution_count: 1456,
        avg_execution_time: 234,
        avg_response_length: 1234,
        avg_quality_score: 0.94,
        error_count: 12
      }
    ],
    height: 300
  }
};

export const CompactHeight: Story = {
  args: {
    data: mockDataset.modelComparison.slice(0, 4),
    height: 350
  }
};