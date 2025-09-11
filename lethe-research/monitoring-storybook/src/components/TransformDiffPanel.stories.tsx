import type { Meta, StoryObj } from '@storybook/react';
import { TransformDiffPanel } from './TransformDiffPanel';
import { mockDataScenarios, addRealtimeChange } from '../utils/mockData';
import { useState, useEffect } from 'react';

const meta = {
  title: 'Transform Diff/Panel',
  component: TransformDiffPanel,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
# Transform Diff Panel

A comprehensive real-time visualization panel for analyzing prompt transformation changes.

## Features

- **Real-time WebSocket Integration**: Live updates with connection status indicators
- **Multiple Visualization Types**: 
  - Change Histogram: Bar chart with 15+ granular change types and severity coding
  - Token Flow: Sankey diagram showing token allocation changes
  - KV Prefix Heatmap: Impact analysis with Jaccard similarity and volatility metrics
  - Timeline View: Chronological sequence with causality links
- **Difficulty Gate Integration**: Dynamic K2 cap adjustment and dimension selection
- **Interactive Filtering**: By provider, request ID, time window, change types, severity
- **Export Functionality**: PNG, SVG, and JSON formats
- **Performance Optimized**: <100ms render time for 1000+ changes

## Architecture

Built with React 18, D3.js for visualizations, and custom WebSocket integration.
Designed for operational utility - helping engineers quickly understand what changed and why.
        `
      }
    }
  },
  tags: ['autodocs'],
} satisfies Meta<typeof TransformDiffPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

// Basic story with minimal data
export const Default: Story = {
  args: {
    changes: mockDataScenarios.minimal.changes,
    tokenMetrics: mockDataScenarios.minimal.tokenMetrics,
    kvMetrics: mockDataScenarios.minimal.kvMetrics,
    difficultyGateMetrics: mockDataScenarios.minimal.difficultyMetrics,
    realTimeEnabled: false,
  },
};

// Standard dataset for typical usage
export const StandardDataset: Story = {
  args: {
    changes: mockDataScenarios.standard.changes,
    tokenMetrics: mockDataScenarios.standard.tokenMetrics,
    kvMetrics: mockDataScenarios.standard.kvMetrics,
    difficultyGateMetrics: mockDataScenarios.standard.difficultyMetrics,
    realTimeEnabled: false,
  },
  parameters: {
    docs: {
      description: {
        story: 'Standard dataset with ~50 changes over 4 hours. Represents typical operational load.'
      }
    }
  }
};

// Large dataset for performance testing
export const LargeDataset: Story = {
  args: {
    changes: mockDataScenarios.large.changes,
    tokenMetrics: mockDataScenarios.large.tokenMetrics,
    kvMetrics: mockDataScenarios.large.kvMetrics,
    difficultyGateMetrics: mockDataScenarios.large.difficultyMetrics,
    realTimeEnabled: false,
  },
  parameters: {
    docs: {
      description: {
        story: 'Large dataset with 500+ changes over 24 hours. Tests performance optimization and rendering efficiency.'
      }
    }
  }
};

// High complexity scenario
export const HighComplexity: Story = {
  args: {
    changes: mockDataScenarios.highComplexity.changes,
    tokenMetrics: mockDataScenarios.highComplexity.tokenMetrics,
    kvMetrics: mockDataScenarios.highComplexity.kvMetrics,
    difficultyGateMetrics: mockDataScenarios.highComplexity.difficultyMetrics,
    realTimeEnabled: false,
  },
  parameters: {
    docs: {
      description: {
        story: 'High complexity scenario with elevated rollback frequency, performance variance, and change entropy. Triggers difficulty gate recommendations.'
      }
    }
  }
};

// Real-time simulation story
export const RealTimeSimulation: Story = {
  render: (args) => {
    const [changes, setChanges] = useState(mockDataScenarios.realTime.changes);
    const [tokenMetrics, setTokenMetrics] = useState(mockDataScenarios.realTime.tokenMetrics);
    const [kvMetrics, setKVMetrics] = useState(mockDataScenarios.realTime.kvMetrics);
    const [difficultyMetrics, setDifficultyMetrics] = useState(mockDataScenarios.realTime.difficultyMetrics);

    useEffect(() => {
      const interval = setInterval(() => {
        setChanges(prev => {
          const newChanges = addRealtimeChange(prev);
          
          // Update metrics based on new changes
          const mockScenario = mockDataScenarios.realTime;
          mockScenario.changes = newChanges;
          
          setTokenMetrics(mockScenario.tokenMetrics);
          setKVMetrics(mockScenario.kvMetrics);
          setDifficultyMetrics(mockScenario.difficultyMetrics);
          
          return newChanges;
        });
      }, 3000); // Add new change every 3 seconds

      return () => clearInterval(interval);
    }, []);

    return (
      <TransformDiffPanel
        {...args}
        changes={changes}
        tokenMetrics={tokenMetrics}
        kvMetrics={kvMetrics}
        difficultyGateMetrics={difficultyMetrics}
        realTimeEnabled={true}
        websocketUrl="ws://localhost:8080/transforms" // Mock URL
      />
    );
  },
  args: {
    onExport: (format) => {
      console.log(`Exporting in ${format} format`);
      return Promise.resolve();
    },
    onFilter: (filters) => {
      console.log('Filter changed:', filters);
    }
  },
  parameters: {
    docs: {
      description: {
        story: `
Real-time simulation that adds new transform changes every 3 seconds. 
Demonstrates:
- WebSocket connection simulation
- Live metric updates
- Dynamic visualization updates
- Real-time difficulty gate analysis
        `
      }
    }
  }
};

// Empty state story
export const EmptyState: Story = {
  args: {
    changes: [],
    tokenMetrics: {
      before: 0,
      after: 0,
      delta: 0,
      efficiency: 1
    },
    kvMetrics: {
      prefixJaccard: 0,
      volatility: 0,
      headEdits: 0,
      tailEdits: 0
    },
    difficultyGateMetrics: {
      changeEntropy: 0,
      rollbackFrequency: 0,
      editDepth: 0,
      complexityScore: 0,
      recommendedK2Cap: 4096,
      recommendedDimension: 256
    },
    realTimeEnabled: false,
  },
  parameters: {
    docs: {
      description: {
        story: 'Empty state with no transform changes. Shows how the panel handles zero data gracefully.'
      }
    }
  }
};

// Filtered view story
export const FilteredView: Story = {
  render: (args) => {
    const [filters, setFilters] = useState({
      provider: ['openai'],
      timeWindow: {
        start: new Date(Date.now() - 2 * 60 * 60 * 1000), // Last 2 hours
        end: new Date()
      }
    });

    return (
      <TransformDiffPanel
        {...args}
        onFilter={(newFilters) => {
          console.log('Filters updated:', newFilters);
          setFilters(newFilters);
        }}
      />
    );
  },
  args: {
    changes: mockDataScenarios.standard.changes,
    tokenMetrics: mockDataScenarios.standard.tokenMetrics,
    kvMetrics: mockDataScenarios.standard.kvMetrics,
    difficultyGateMetrics: mockDataScenarios.standard.difficultyMetrics,
    realTimeEnabled: false,
  },
  parameters: {
    docs: {
      description: {
        story: 'Pre-filtered view showing only OpenAI provider data from the last 2 hours. Demonstrates filter functionality.'
      }
    }
  }
};

// Export functionality story
export const ExportFunctionality: Story = {
  args: {
    changes: mockDataScenarios.standard.changes,
    tokenMetrics: mockDataScenarios.standard.tokenMetrics,
    kvMetrics: mockDataScenarios.standard.kvMetrics,
    difficultyGateMetrics: mockDataScenarios.standard.difficultyMetrics,
    realTimeEnabled: false,
    onExport: async (format) => {
      console.log(`Exporting visualization in ${format.toUpperCase()} format...`);
      
      // Simulate export processing time
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Simulate successful export
      alert(`Successfully exported visualization as ${format.toUpperCase()}`);
    }
  },
  parameters: {
    docs: {
      description: {
        story: 'Demonstrates export functionality. Click the Export button to try different formats (PNG, SVG, JSON).'
      }
    }
  }
};

// Dark theme story
export const DarkTheme: Story = {
  args: {
    changes: mockDataScenarios.standard.changes,
    tokenMetrics: mockDataScenarios.standard.tokenMetrics,
    kvMetrics: mockDataScenarios.standard.kvMetrics,
    difficultyGateMetrics: mockDataScenarios.standard.difficultyMetrics,
    realTimeEnabled: false,
  },
  parameters: {
    backgrounds: { default: 'dark' },
    docs: {
      description: {
        story: 'Dark theme variant of the transform diff panel. All visualizations support both light and dark modes.'
      }
    }
  },
  decorators: [
    (Story) => (
      <div className="dark">
        <Story />
      </div>
    ),
  ],
};

// Mobile responsive story
export const MobileView: Story = {
  args: {
    changes: mockDataScenarios.minimal.changes,
    tokenMetrics: mockDataScenarios.minimal.tokenMetrics,
    kvMetrics: mockDataScenarios.minimal.kvMetrics,
    difficultyGateMetrics: mockDataScenarios.minimal.difficultyMetrics,
    realTimeEnabled: false,
  },
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
    docs: {
      description: {
        story: 'Mobile-responsive view with adapted layout for smaller screens. Sidebar collapses and visualizations stack vertically.'
      }
    }
  }
};