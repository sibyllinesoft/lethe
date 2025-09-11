import type { Meta, StoryObj } from '@storybook/react';
import { TokenFlow } from './TokenFlow';
import { mockDataScenarios } from '../../utils/mockData';

const meta = {
  title: 'Transform Diff/Components/TokenFlow',
  component: TokenFlow,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
# Token Flow

Sankey diagram visualization showing token allocation changes through transformations.

## Features
- Visual representation of token flow from input to output
- Color-coded links showing added, removed, and modified tokens
- Interactive tooltips with detailed metrics
- Efficiency indicators and delta calculations
- Real-time updates as new changes arrive
        `
      }
    }
  },
  tags: ['autodocs'],
} satisfies Meta<typeof TokenFlow>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    tokenMetrics: mockDataScenarios.standard.tokenMetrics,
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
    tokenMetrics: mockDataScenarios.standard.tokenMetrics,
    changes: mockDataScenarios.standard.changes,
    detailed: true,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '800px', height: '400px' }}>
        <Story />
      </div>
    ),
  ],
};

export const TokenIncrease: Story = {
  args: {
    tokenMetrics: {
      before: 1000,
      after: 1350,
      delta: 350,
      efficiency: 1.35
    },
    changes: mockDataScenarios.standard.changes,
    detailed: true,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '800px', height: '400px' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'Scenario where transformations result in token increase (content expansion).'
      }
    }
  }
};

export const TokenDecrease: Story = {
  args: {
    tokenMetrics: {
      before: 1500,
      after: 1200,
      delta: -300,
      efficiency: 0.8
    },
    changes: mockDataScenarios.standard.changes,
    detailed: true,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '800px', height: '400px' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'Scenario where transformations result in token reduction (content compression).'
      }
    }
  }
};

export const HighEfficiency: Story = {
  args: {
    tokenMetrics: {
      before: 1000,
      after: 995,
      delta: -5,
      efficiency: 0.995
    },
    changes: mockDataScenarios.minimal.changes,
    detailed: true,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '800px', height: '400px' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'High-efficiency transformation with minimal token changes.'
      }
    }
  }
};