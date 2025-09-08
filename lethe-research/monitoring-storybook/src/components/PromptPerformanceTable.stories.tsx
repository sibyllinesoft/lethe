import type { Meta, StoryObj } from '@storybook/react';
import { PromptPerformanceTable } from './PromptPerformanceTable';
import { mockDataset, generateMockPromptPerformance } from '../data/mockData';

const meta: Meta<typeof PromptPerformanceTable> = {
  title: 'Components/PromptPerformanceTable',
  component: PromptPerformanceTable,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'Interactive data table showing prompt performance metrics with sorting, filtering, and search functionality.'
      }
    }
  },
  tags: ['autodocs']
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    data: mockDataset.promptPerformance
  }
};

export const LargeDataset: Story = {
  args: {
    data: generateMockPromptPerformance(50)
  }
};

export const SmallDataset: Story = {
  args: {
    data: generateMockPromptPerformance(3)
  }
};

export const HighPerformancePrompts: Story = {
  args: {
    data: [
      {
        prompt_id: 'optimized_code_generation_v3',
        execution_count: 1234,
        avg_execution_time: 89,
        avg_response_length: 1456,
        avg_quality_score: 0.96,
        last_used: '2024-01-15T14:30:00Z',
        error_count: 0,
        success_rate: 100
      },
      {
        prompt_id: 'fast_text_summarization_v2',
        execution_count: 987,
        avg_execution_time: 123,
        avg_response_length: 678,
        avg_quality_score: 0.94,
        last_used: '2024-01-15T13:45:00Z',
        error_count: 2,
        success_rate: 99.8
      },
      {
        prompt_id: 'efficient_translation_engine',
        execution_count: 756,
        avg_execution_time: 156,
        avg_response_length: 890,
        avg_quality_score: 0.92,
        last_used: '2024-01-15T12:20:00Z',
        error_count: 1,
        success_rate: 99.9
      }
    ]
  }
};

export const ProblematicPrompts: Story = {
  args: {
    data: [
      {
        prompt_id: 'slow_legacy_analyzer_v1',
        execution_count: 234,
        avg_execution_time: 4567,
        avg_response_length: 1200,
        avg_quality_score: 0.67,
        last_used: '2024-01-14T16:30:00Z',
        error_count: 45,
        success_rate: 80.8
      },
      {
        prompt_id: 'unreliable_content_generator',
        execution_count: 123,
        avg_execution_time: 3456,
        avg_response_length: 890,
        avg_quality_score: 0.58,
        last_used: '2024-01-15T09:15:00Z',
        error_count: 32,
        success_rate: 74.0
      },
      {
        prompt_id: 'deprecated_old_template',
        execution_count: 67,
        avg_execution_time: 5678,
        avg_response_length: 1500,
        avg_quality_score: 0.45,
        last_used: '2024-01-13T11:00:00Z',
        error_count: 18,
        success_rate: 73.1
      }
    ]
  }
};

export const EmptyState: Story = {
  args: {
    data: []
  }
};