/**
 * Layout Utilities
 * Consistent layout patterns and spacing utilities
 */

import { designTokens } from '../tokens';

// Grid utilities
export const layoutUtils = {
  // Responsive grid configurations
  grid: {
    cols: {
      1: 'grid-cols-1',
      2: 'grid-cols-1 md:grid-cols-2',
      3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
      4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
      auto: 'grid-cols-auto-fit',
    },
    gaps: {
      sm: 'gap-3',
      md: 'gap-4',
      lg: 'gap-6',
      xl: 'gap-8',
    }
  },

  // Flex utilities
  flex: {
    center: 'flex items-center justify-center',
    between: 'flex items-center justify-between',
    start: 'flex items-center justify-start',
    end: 'flex items-center justify-end',
    col: 'flex flex-col',
    colCenter: 'flex flex-col items-center justify-center',
  },

  // Container sizes
  container: {
    sm: 'max-w-2xl mx-auto',
    md: 'max-w-4xl mx-auto', 
    lg: 'max-w-6xl mx-auto',
    xl: 'max-w-7xl mx-auto',
    full: 'max-w-full mx-auto',
  },

  // Spacing utilities
  spacing: {
    section: 'py-12 md:py-16 lg:py-20',
    component: 'py-6 md:py-8',
    element: 'py-3 md:py-4',
  },

  // Dashboard-specific layouts
  dashboard: {
    sidebar: 'w-64 flex-shrink-0',
    main: 'flex-1 min-w-0',
    content: 'p-6 md:p-8',
    header: 'h-16 border-b border-graphite-200 flex items-center justify-between px-6',
  },

  // Card grid patterns
  cardGrid: {
    metrics: 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6',
    charts: 'grid grid-cols-1 lg:grid-cols-2 gap-6',
    list: 'space-y-4',
  }
} as const;

// Responsive breakpoints (matching Tailwind defaults but documented)
export const breakpoints = {
  sm: '640px',
  md: '768px', 
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
} as const;

// Z-index scale
export const zIndex = {
  hide: -1,
  auto: 'auto',
  base: 0,
  docked: 10,
  dropdown: 1000,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  tooltip: 1600,
} as const;

// Common layout patterns as utility functions
export const getGridCols = (count: 1 | 2 | 3 | 4) => {
  return layoutUtils.grid.cols[count];
};

export const getContainerSize = (size: keyof typeof layoutUtils.container) => {
  return layoutUtils.container[size];
};

// Layout composition helpers
export const createStackLayout = (spacing: 'sm' | 'md' | 'lg' = 'md') => {
  const spacingMap = {
    sm: 'space-y-3',
    md: 'space-y-4', 
    lg: 'space-y-6',
  };
  
  return `flex flex-col ${spacingMap[spacing]}`;
};

export const createClusterLayout = (gap: 'sm' | 'md' | 'lg' = 'md') => {
  const gapMap = {
    sm: 'gap-2',
    md: 'gap-3',
    lg: 'gap-4',
  };
  
  return `flex flex-wrap ${gapMap[gap]}`;
};

// Responsive utilities
export const responsive = {
  show: {
    sm: 'hidden sm:block',
    md: 'hidden md:block',
    lg: 'hidden lg:block',
  },
  hide: {
    sm: 'sm:hidden',
    md: 'md:hidden', 
    lg: 'lg:hidden',
  },
} as const;