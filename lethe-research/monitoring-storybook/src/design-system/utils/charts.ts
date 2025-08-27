/**
 * Chart Utilities for Data Visualization
 * Provides consistent colors, themes, and configurations for charts
 */

import { designTokens } from '../tokens';

// Chart Color Palette - Sophisticated and accessible
export const chartColors = {
  primary: [
    '#6366f1', // Indigo
    '#8b5cf6', // Violet
    '#06b6d4', // Cyan
    '#84cc16', // Lime
    '#f59e0b', // Amber
    '#ec4899', // Pink
    '#10b981', // Emerald
    '#f97316', // Orange
  ],
  
  success: [
    '#10b981',
    '#059669',
    '#047857',
    '#065f46',
  ],
  
  warning: [
    '#f59e0b',
    '#d97706',
    '#b45309',
    '#92400e',
  ],
  
  error: [
    '#ef4444',
    '#dc2626',
    '#b91c1c',
    '#991b1b',
  ],
  
  neutral: [
    '#6b7280',
    '#4b5563',
    '#374151',
    '#1f2937',
  ],

  // Gradient combinations for sophisticated visualizations
  gradients: {
    primary: ['#6366f1', '#8b5cf6'],
    success: ['#10b981', '#059669'],
    warning: ['#f59e0b', '#d97706'],
    error: ['#ef4444', '#dc2626'],
    cool: ['#06b6d4', '#8b5cf6'],
    warm: ['#f59e0b', '#ec4899'],
    ocean: ['#06b6d4', '#10b981'],
    sunset: ['#f97316', '#ec4899'],
  }
} as const;

// Get chart color by index with automatic cycling
export const getChartColor = (index: number, palette: keyof typeof chartColors = 'primary') => {
  const colors = chartColors[palette];
  return colors[index % colors.length];
};

// Chart configuration defaults
export const chartConfig = {
  // Common chart options
  defaults: {
    maintainAspectRatio: false,
    responsive: true,
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
    plugins: {
      legend: {
        display: true,
        position: 'bottom' as const,
        labels: {
          usePointStyle: true,
          pointStyle: 'circle',
          font: {
            family: designTokens.typography.fontFamily.primary.join(', '),
            size: 12,
          },
          color: designTokens.colors.graphite[600],
          padding: 16,
        },
      },
      tooltip: {
        backgroundColor: designTokens.colors.graphite[900],
        titleColor: designTokens.colors.graphite[50],
        bodyColor: designTokens.colors.graphite[100],
        borderColor: designTokens.colors.graphite[700],
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        font: {
          family: designTokens.typography.fontFamily.primary.join(', '),
          size: 12,
        },
        displayColors: true,
        usePointStyle: true,
      },
    },
    scales: {
      x: {
        grid: {
          display: true,
          color: designTokens.colors.graphite[200],
          lineWidth: 1,
        },
        ticks: {
          font: {
            family: designTokens.typography.fontFamily.primary.join(', '),
            size: 11,
          },
          color: designTokens.colors.graphite[500],
          padding: 8,
        },
        border: {
          display: false,
        },
      },
      y: {
        grid: {
          display: true,
          color: designTokens.colors.graphite[200],
          lineWidth: 1,
        },
        ticks: {
          font: {
            family: designTokens.typography.fontFamily.primary.join(', '),
            size: 11,
          },
          color: designTokens.colors.graphite[500],
          padding: 8,
        },
        border: {
          display: false,
        },
      },
    },
  },

  // Line chart specific options
  line: {
    elements: {
      line: {
        borderWidth: 2.5,
        tension: 0.1, // Subtle curve
        fill: false,
      },
      point: {
        radius: 4,
        hoverRadius: 6,
        borderWidth: 2,
        backgroundColor: designTokens.colors.surface.primary,
      },
    },
  },

  // Area chart options (filled line charts)
  area: {
    elements: {
      line: {
        borderWidth: 2.5,
        tension: 0.1,
        fill: true,
      },
      point: {
        radius: 3,
        hoverRadius: 5,
        borderWidth: 2,
        backgroundColor: designTokens.colors.surface.primary,
      },
    },
  },

  // Bar chart options
  bar: {
    borderRadius: {
      topLeft: 4,
      topRight: 4,
    },
    borderSkipped: false,
    maxBarThickness: 60,
    categoryPercentage: 0.8,
    barPercentage: 0.7,
  },

  // Bubble chart options
  bubble: {
    elements: {
      point: {
        radius: (context: any) => {
          const value = context.parsed.r;
          return Math.max(4, Math.min(20, value));
        },
        hoverRadius: (context: any) => {
          const value = context.parsed.r;
          return Math.max(6, Math.min(24, value * 1.2));
        },
        borderWidth: 2,
        backgroundColor: (context: any) => {
          const color = getChartColor(context.datasetIndex);
          return color + '80'; // 50% opacity
        },
        borderColor: (context: any) => getChartColor(context.datasetIndex),
      },
    },
  },

  // Doughnut/Pie chart options
  doughnut: {
    cutout: '60%',
    borderWidth: 0,
    borderRadius: 2,
    spacing: 2,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          boxWidth: 12,
          boxHeight: 12,
          usePointStyle: true,
        },
      },
    },
  },
} as const;

// Utility functions for creating chart data
export const createGradient = (
  ctx: CanvasRenderingContext2D,
  chartArea: { top: number; bottom: number; left: number; right: number },
  colorStart: string,
  colorEnd: string,
  direction: 'vertical' | 'horizontal' = 'vertical'
) => {
  const gradient = direction === 'vertical'
    ? ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom)
    : ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
  
  gradient.addColorStop(0, colorStart);
  gradient.addColorStop(1, colorEnd);
  
  return gradient;
};

// Create dataset with design system colors
export const createDataset = (
  label: string,
  data: number[],
  type: 'line' | 'bar' | 'area' = 'line',
  colorIndex = 0
) => {
  const color = getChartColor(colorIndex);
  
  const baseDataset = {
    label,
    data,
    borderColor: color,
    backgroundColor: type === 'area' ? `${color}20` : color,
  };

  switch (type) {
    case 'line':
      return {
        ...baseDataset,
        ...chartConfig.line.elements,
        fill: false,
      };
    
    case 'area':
      return {
        ...baseDataset,
        ...chartConfig.area.elements,
        fill: true,
        backgroundColor: `${color}20`,
      };
    
    case 'bar':
      return {
        ...baseDataset,
        ...chartConfig.bar,
        borderColor: 'transparent',
        borderWidth: 0,
      };
    
    default:
      return baseDataset;
  }
};

// Performance metrics color mapping
export const performanceColors = {
  excellent: chartColors.success[0], // Green
  good: chartColors.success[1],      // Dark green
  fair: chartColors.warning[0],      // Amber
  poor: chartColors.warning[1],      // Dark amber
  critical: chartColors.error[0],    // Red
} as const;

// Status colors for system monitoring
export const statusColors = {
  online: chartColors.success[0],
  offline: chartColors.neutral[0],
  error: chartColors.error[0],
  warning: chartColors.warning[0],
  pending: chartColors.warning[1],
} as const;

// Time series chart utilities
export const timeSeriesConfig = {
  ...chartConfig.defaults,
  scales: {
    ...chartConfig.defaults.scales,
    x: {
      ...chartConfig.defaults.scales.x,
      type: 'time' as const,
      time: {
        displayFormats: {
          hour: 'HH:mm',
          day: 'MMM DD',
          week: 'MMM DD',
          month: 'MMM YYYY',
        },
      },
      adapters: {
        date: {},
      },
    },
  },
} as const;