import type { Meta, StoryObj } from '@storybook/react';
import { ChangeHistogram } from './ChangeHistogram';
import { mockDataScenarios } from '../../utils/mockData';

const meta = {
  title: 'Transform Diff/Components/ChangeHistogram',
  component: ChangeHistogram,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
# Change Histogram

D3-powered bar chart visualization of transform change type frequency with severity color coding.

## Features
- 15+ granular change types from TransformChangeV2 schema
- Severity color coding (low: green, medium: yellow, high: orange, critical: red)
- Performance impact visualization
- Success rate indicators
- Interactive tooltips with detailed metrics
- Responsive design with smooth animations
        `
      }
    }
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ChangeHistogram>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    changes: mockDataScenarios.standard.changes,
    detailed: false,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '600px', height: '300px' }}>
        <Story />
      </div>
    ),
  ],
};

export const Detailed: Story = {
  args: {
    changes: mockDataScenarios.standard.changes,
    detailed: true,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '800px', height: '500px' }}>
        <Story />
      </div>
    ),
  ],
};

export const HighComplexity: Story = {
  args: {
    changes: mockDataScenarios.highComplexity.changes,
    detailed: true,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '800px', height: '500px' }}>
        <Story />
      </div>
    ),
  ],
};

export const Minimal: Story = {
  args: {
    changes: mockDataScenarios.minimal.changes,
    detailed: false,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '400px', height: '250px' }}>
        <Story />
      </div>
    ),
  ],
};

export const Empty: Story = {
  args: {
    changes: [],
    detailed: true,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '600px', height: '400px' }}>
        <Story />
      </div>
    ),
  ],
};