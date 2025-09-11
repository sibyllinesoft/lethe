import type { Meta, StoryObj } from '@storybook/react';
import { DifficultyGatePanel } from './DifficultyGatePanel';
import { mockDataScenarios } from '../../utils/mockData';

const meta = {
  title: 'Transform Diff/Components/DifficultyGatePanel',
  component: DifficultyGatePanel,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
# Difficulty Gate Panel

Real-time difficulty gate analysis and recommendations for transform complexity management.

## Features
- Change entropy calculation (Shannon entropy of change types)
- Rollback frequency tracking
- Edit depth analysis
- Dynamic K2 cap recommendations
- Dimension selection guidance (256 vs 768)
- Visual complexity indicators with color-coded metrics
        `
      }
    }
  },
  tags: ['autodocs'],
} satisfies Meta<typeof DifficultyGatePanel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    metrics: mockDataScenarios.standard.difficultyMetrics,
    changes: mockDataScenarios.standard.changes,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '400px' }}>
        <Story />
      </div>
    ),
  ],
};

export const LowComplexity: Story = {
  args: {
    metrics: {
      changeEntropy: 1.2,
      rollbackFrequency: 0.02,
      editDepth: 0.15,
      complexityScore: 0.25,
      recommendedK2Cap: 4096,
      recommendedDimension: 256
    },
    changes: mockDataScenarios.minimal.changes,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '400px' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'Low complexity scenario with recommended standard settings.'
      }
    }
  }
};

export const MediumComplexity: Story = {
  args: {
    metrics: {
      changeEntropy: 2.1,
      rollbackFrequency: 0.12,
      editDepth: 0.28,
      complexityScore: 0.55,
      recommendedK2Cap: 2048,
      recommendedDimension: 256
    },
    changes: mockDataScenarios.standard.changes,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '400px' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'Medium complexity scenario triggering moderate K2 cap recommendations.'
      }
    }
  }
};

export const HighComplexity: Story = {
  args: {
    metrics: mockDataScenarios.highComplexity.difficultyMetrics,
    changes: mockDataScenarios.highComplexity.changes,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '400px' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'High complexity scenario with critical warnings and aggressive K2 cap reduction.'
      }
    }
  }
};

export const CriticalComplexity: Story = {
  args: {
    metrics: {
      changeEntropy: 3.8,
      rollbackFrequency: 0.35,
      editDepth: 0.65,
      complexityScore: 0.85,
      recommendedK2Cap: 1024,
      recommendedDimension: 768
    },
    changes: mockDataScenarios.highComplexity.changes,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '400px' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'Critical complexity scenario requiring immediate intervention and minimal token limits.'
      }
    }
  }
};

export const NoData: Story = {
  args: {
    metrics: {
      changeEntropy: 0,
      rollbackFrequency: 0,
      editDepth: 0,
      complexityScore: 0,
      recommendedK2Cap: 4096,
      recommendedDimension: 256
    },
    changes: [],
  },
  decorators: [
    (Story) => (
      <div style={{ width: '400px' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'Empty state with no data, showing default recommendations.'
      }
    }
  }
};