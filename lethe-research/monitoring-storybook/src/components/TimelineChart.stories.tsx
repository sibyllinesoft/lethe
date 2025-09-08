import type { Meta, StoryObj } from '@storybook/react';
import { TimelineChart } from './TimelineChart';
import { mockDataset, generateMockTimelineData } from '../data/mockData';

const meta: Meta<typeof TimelineChart> = {
  title: 'Components/TimelineChart',
  component: TimelineChart,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'Interactive timeline chart showing execution metrics over time with line visualizations for executions, response time, and errors.'
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
    data: mockDataset.timelineData,
    height: 400
  }
};

export const OneWeek: Story = {
  args: {
    data: generateMockTimelineData(7),
    height: 400
  }
};

export const OneMonth: Story = {
  args: {
    data: generateMockTimelineData(30),
    height: 500
  }
};

export const CompactHeight: Story = {
  args: {
    data: mockDataset.timelineData,
    height: 300
  }
};

export const TallChart: Story = {
  args: {
    data: mockDataset.timelineData,
    height: 600
  }
};

export const HighActivity: Story = {
  args: {
    data: [
      { date: '2024-01-10', total_executions: 450, avg_execution_time: 234, avg_response_length: 1200, errors: 2 },
      { date: '2024-01-11', total_executions: 523, avg_execution_time: 198, avg_response_length: 1100, errors: 1 },
      { date: '2024-01-12', total_executions: 612, avg_execution_time: 167, avg_response_length: 1250, errors: 0 },
      { date: '2024-01-13', total_executions: 578, avg_execution_time: 189, avg_response_length: 1180, errors: 3 },
      { date: '2024-01-14', total_executions: 634, avg_execution_time: 156, avg_response_length: 1300, errors: 1 },
      { date: '2024-01-15', total_executions: 689, avg_execution_time: 145, avg_response_length: 1350, errors: 0 },
      { date: '2024-01-16', total_executions: 701, avg_execution_time: 134, avg_response_length: 1400, errors: 2 }
    ],
    height: 450
  }
};