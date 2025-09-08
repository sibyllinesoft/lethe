import type { Meta, StoryObj } from '@storybook/react';
import { PerformanceBubbleChart } from './PerformanceBubbleChart';
import { mockDataset, generateMockPromptPerformance } from '../data/mockData';

const meta: Meta<typeof PerformanceBubbleChart> = {
  title: 'Components/PerformanceBubbleChart',
  component: PerformanceBubbleChart,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'Bubble chart visualization showing prompt performance with execution time vs count, bubble size representing success rate, and color indicating quality score.'
      }
    }
  },
  tags: ['autodocs'],
  argTypes: {
    height: {
      control: { type: 'range', min: 400, max: 800, step: 50 }
    }
  }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    data: mockDataset.promptPerformance,
    height: 500
  }
};

export const ManyPrompts: Story = {
  args: {
    data: generateMockPromptPerformance(25),
    height: 600
  }
};

export const FewPrompts: Story = {
  args: {
    data: generateMockPromptPerformance(5),
    height: 400
  }
};

export const CompactView: Story = {
  args: {
    data: mockDataset.promptPerformance.slice(0, 8),
    height: 400
  }
};

export const HighPerformancePrompts: Story = {
  args: {
    data: [
      {
        prompt_id: 'optimized_code_gen_v3',
        execution_count: 234,
        avg_execution_time: 156,
        avg_response_length: 1200,
        avg_quality_score: 0.95,
        last_used: '2024-01-15T14:30:00Z',
        error_count: 0,
        success_rate: 100
      },
      {
        prompt_id: 'fast_text_analysis_v2',
        execution_count: 187,
        avg_execution_time: 89,
        avg_response_length: 800,
        avg_quality_score: 0.92,
        last_used: '2024-01-15T13:45:00Z',
        error_count: 1,
        success_rate: 99.5
      },
      {
        prompt_id: 'efficient_summarization',
        execution_count: 156,
        avg_execution_time: 234,
        avg_response_length: 450,
        avg_quality_score: 0.94,
        last_used: '2024-01-15T12:20:00Z',
        error_count: 0,
        success_rate: 100
      }
    ],
    height: 450
  }
};

export const ProblematicPrompts: Story = {
  args: {
    data: [
      {
        prompt_id: 'slow_complex_analysis',
        execution_count: 45,
        avg_execution_time: 3456,
        avg_response_length: 2100,
        avg_quality_score: 0.67,
        last_used: '2024-01-14T16:30:00Z',
        error_count: 12,
        success_rate: 73.3
      },
      {
        prompt_id: 'unreliable_generator_v1',
        execution_count: 78,
        avg_execution_time: 2234,
        avg_response_length: 890,
        avg_quality_score: 0.58,
        last_used: '2024-01-15T09:15:00Z',
        error_count: 18,
        success_rate: 76.9
      },
      {
        prompt_id: 'legacy_prompt_old',
        execution_count: 23,
        avg_execution_time: 4567,
        avg_response_length: 1500,
        avg_quality_score: 0.45,
        last_used: '2024-01-13T11:00:00Z',
        error_count: 8,
        success_rate: 65.2
      }
    ],
    height: 450
  }
};