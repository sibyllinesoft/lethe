import type { Meta, StoryObj } from '@storybook/react';
import { ParetoFrontierAnalysis } from './ParetoParetoFrontierAnalysis';

// Mock data generator for Pareto analysis
const generateParetoData = (size: number = 100) => {
  const providers = ['openai', 'anthropic', 'google'];
  const modelSizes: ('256' | '768')[] = ['256', '768'];
  const scenarios = ['simple', 'moderate', 'complex', 'extreme'];
  
  return Array.from({ length: size }, (_, i) => {
    const complexity = 0.1 + Math.random() * 0.8; // 0.1 to 0.9
    const k2Base = Math.max(0.1, 1.0 - complexity + (Math.random() - 0.5) * 0.3); // Inverse relationship with noise
    const k2Coefficient = Math.max(0.05, Math.min(1.0, k2Base));
    const performanceScore = Math.max(0.6, 1.0 - (complexity * 0.3) + (Math.random() - 0.5) * 0.2);
    
    return {
      complexity,
      k2Coefficient,
      performanceScore,
      difficultyGateScore: complexity * 0.8 + Math.random() * 0.2,
      provider: providers[Math.floor(Math.random() * providers.length)],
      modelSize: modelSizes[Math.floor(Math.random() * modelSizes.length)],
      timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000), // Last 7 days
      isOptimal: false, // Will be calculated by the component
      metadata: {
        requestId: `req_${i.toString().padStart(4, '0')}`,
        scenario: scenarios[Math.floor(Math.random() * scenarios.length)],
        successRate: 0.85 + Math.random() * 0.14, // 85-99%
        avgLatency: 50 + Math.random() * 200, // 50-250ms
        tokenEfficiency: 0.7 + Math.random() * 0.29 // 70-99%
      }
    };
  });
};

// Generate specialized datasets
const balancedDataset = generateParetoData(150);
const complexityFocusedDataset = generateParetoData(80).map(point => ({
  ...point,
  complexity: Math.pow(Math.random(), 0.5) * 0.9 + 0.1, // Skew towards higher complexity
  k2Coefficient: Math.max(0.1, point.k2Coefficient - 0.2)
}));

const performanceFocusedDataset = generateParetoData(120).map(point => ({
  ...point,
  k2Coefficient: Math.pow(Math.random(), 0.7) * 0.8 + 0.2, // Skew towards higher K2
  complexity: Math.max(0.1, point.complexity - 0.1)
}));

const meta: Meta<typeof ParetoFrontierAnalysis> = {
  title: 'Analytics/ParetoFrontierAnalysis',
  component: ParetoFrontierAnalysis,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
# Complexity vs K2 Pareto Frontier Analysis

Advanced visualization dashboard for analyzing the trade-off between transform complexity and K2 coefficient values.
Identifies Pareto-optimal configurations that minimize complexity while maintaining quality performance.

## Key Features

- **Pareto Frontier Calculation**: Automatically identifies optimal configurations
- **Real-time Updates**: Live data streaming with performance metrics
- **Interactive Filtering**: Filter by provider, model size, and time ranges
- **Multiple View Modes**: Frontier, evolution, and sensitivity analysis
- **Export Capabilities**: PNG, SVG, JSON, and CSV exports
- **Performance Metrics**: Convergence rate, stability index, and optimality gap analysis

## Use Cases

- **Configuration Optimization**: Find optimal complexity-performance trade-offs
- **Provider Comparison**: Compare different providers' Pareto frontiers
- **Model Size Analysis**: Analyze 256 vs 768 dimension performance
- **Historical Trend Analysis**: Track Pareto frontier evolution over time
        `
      }
    }
  },
  argTypes: {
    realTimeEnabled: {
      control: { type: 'boolean' },
      description: 'Enable real-time data updates'
    },
    onOptimalConfigurationSelected: {
      action: 'optimal configuration selected',
      description: 'Callback when user selects an optimal point'
    },
    onExport: {
      action: 'exported',
      description: 'Callback when user exports data'
    }
  }
};

export default meta;
type Story = StoryObj<typeof meta>;

// Default story with balanced dataset
export const Default: Story = {
  args: {
    data: balancedDataset,
    realTimeEnabled: false,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Default Pareto frontier analysis with a balanced dataset showing typical complexity-K2 trade-offs across multiple providers.'
      }
    }
  }
};

// Real-time simulation story
export const RealTimeSimulation: Story = {
  args: {
    data: balancedDataset,
    realTimeEnabled: true,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Real-time simulation showing live updates to the Pareto frontier as new data points arrive. Demonstrates dynamic recalculation of optimal configurations.'
      }
    }
  },
  play: async ({ canvasElement }) => {
    // Simulate real-time updates by periodically adding new data points
    const canvas = canvasElement;
    
    const simulateDataUpdate = () => {
      const event = new CustomEvent('paretoDataUpdate', {
        detail: generateParetoData(5) // Add 5 new points
      });
      canvas.dispatchEvent(event);
    };

    // Start simulation
    const interval = setInterval(simulateDataUpdate, 3000);
    
    // Clean up after 30 seconds
    setTimeout(() => clearInterval(interval), 30000);
  }
};

// Complexity-focused analysis
export const ComplexityFocused: Story = {
  args: {
    data: complexityFocusedDataset,
    realTimeEnabled: false,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Analysis focused on high-complexity scenarios, showing how the Pareto frontier behaves when dealing with challenging transform operations.'
      }
    }
  }
};

// Performance-focused analysis
export const PerformanceFocused: Story = {
  args: {
    data: performanceFocusedDataset,
    realTimeEnabled: false,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Analysis focused on high-performance configurations, emphasizing optimal K2 coefficient values with manageable complexity levels.'
      }
    }
  }
};

// Provider comparison story
export const ProviderComparison: Story = {
  args: {
    data: [
      ...generateParetoData(50).map(p => ({ ...p, provider: 'openai' })),
      ...generateParetoData(50).map(p => ({ ...p, provider: 'anthropic', complexity: p.complexity * 1.1, k2Coefficient: p.k2Coefficient * 0.95 })),
      ...generateParetoData(50).map(p => ({ ...p, provider: 'google', complexity: p.complexity * 0.9, k2Coefficient: p.k2Coefficient * 1.05 }))
    ],
    realTimeEnabled: false,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Comparison of Pareto frontiers across different providers, showing how each provider performs in the complexity-performance trade-off space.'
      }
    }
  }
};

// Model size analysis story
export const ModelSizeAnalysis: Story = {
  args: {
    data: [
      ...generateParetoData(75).map(p => ({ 
        ...p, 
        modelSize: '256' as const,
        complexity: p.complexity * 0.8, // 256 generally less complex
        performanceScore: p.performanceScore * 0.95 // Slightly lower performance
      })),
      ...generateParetoData(75).map(p => ({ 
        ...p, 
        modelSize: '768' as const,
        complexity: p.complexity * 1.2, // 768 more complex
        performanceScore: Math.min(1.0, p.performanceScore * 1.05) // Better performance
      }))
    ],
    realTimeEnabled: false,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Analysis comparing 256-dimension vs 768-dimension models, showing the trade-offs between complexity and performance for different model sizes.'
      }
    }
  }
};

// Sparse data story
export const SparseData: Story = {
  args: {
    data: generateParetoData(25),
    realTimeEnabled: false,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Pareto analysis with sparse data, demonstrating how the component handles smaller datasets and edge cases.'
      }
    }
  }
};

// Large dataset performance test
export const PerformanceTest: Story = {
  args: {
    data: generateParetoData(1000),
    realTimeEnabled: false,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Performance test with 1000+ data points to demonstrate scalability and rendering efficiency. Tests visualization performance under load.'
      }
    }
  }
};

// Interactive demo with all features
export const InteractiveDemo: Story = {
  args: {
    data: balancedDataset,
    realTimeEnabled: true,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Interactive demonstration showcasing all features: real-time updates, filtering, view modes, export functionality, and optimal configuration selection.'
      }
    }
  },
  play: async ({ args, canvasElement }) => {
    // Demonstrate interactive features
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Log interactions for demonstration
    console.log('Pareto Frontier Analysis Demo:');
    console.log('- Click points to select configurations');
    console.log('- Use filters to focus on specific providers or model sizes');
    console.log('- Export visualizations using the export button');
    console.log('- Observe real-time updates to the Pareto frontier');
  }
};

// Empty state story
export const EmptyState: Story = {
  args: {
    data: [],
    realTimeEnabled: false,
    className: 'w-full h-full'
  },
  parameters: {
    docs: {
      description: {
        story: 'Empty state handling when no data is available, showing how the component gracefully handles edge cases.'
      }
    }
  }
};