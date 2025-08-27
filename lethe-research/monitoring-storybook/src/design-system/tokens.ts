/**
 * Design Tokens for Lethe Monitoring Design System
 * Graphite Theme - Sophisticated minimal design system
 */

export const designTokens = {
  // Color System - Graphite Palette
  colors: {
    // Primary Graphite Scale
    graphite: {
      50: '#f9fafb',    // Lightest - backgrounds
      100: '#f3f4f6',   // Light backgrounds
      200: '#e5e7eb',   // Subtle borders
      300: '#d1d5db',   // Muted borders
      400: '#9ca3af',   // Disabled text
      500: '#6b7280',   // Secondary text
      600: '#4b5563',   // Primary text
      700: '#374151',   // Dark text
      800: '#1f2937',   // Darker text
      900: '#111827',   // Darkest - high contrast
      950: '#030712',   // Deepest - terminals
    },
    
    // Semantic Colors
    success: {
      50: '#f0fdfa',
      100: '#ccfbf1',
      500: '#10b981',
      600: '#059669',
      700: '#047857',
    },
    
    warning: {
      50: '#fefce8',
      100: '#fef3c7',
      500: '#eab308',
      600: '#ca8a04',
      700: '#a16207',
    },
    
    error: {
      50: '#fef2f2',
      100: '#fee2e2',
      500: '#ef4444',
      600: '#dc2626',
      700: '#b91c1c',
    },
    
    // Data Visualization Colors - Sophisticated palette
    data: {
      primary: '#6366f1',    // Indigo
      secondary: '#8b5cf6',  // Violet  
      tertiary: '#06b6d4',   // Cyan
      quaternary: '#84cc16', // Lime
      accent1: '#f59e0b',    // Amber
      accent2: '#ec4899',    // Pink
      accent3: '#10b981',    // Emerald
      accent4: '#f97316',    // Orange
    },
    
    // Surface Colors
    surface: {
      primary: '#ffffff',
      secondary: '#f9fafb',
      tertiary: '#f3f4f6',
      elevated: '#ffffff',
      overlay: 'rgba(17, 24, 39, 0.8)',
    },
    
    // Interactive States
    interactive: {
      hover: 'rgba(107, 114, 128, 0.08)',
      pressed: 'rgba(107, 114, 128, 0.12)',
      focus: '#6366f1',
      disabled: '#f3f4f6',
    }
  },

  // Typography System
  typography: {
    fontFamily: {
      primary: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      mono: ['JetBrains Mono', 'SF Mono', 'Monaco', 'Consolas', 'monospace'],
    },
    
    fontSize: {
      xs: '0.75rem',     // 12px
      sm: '0.875rem',    // 14px
      base: '1rem',      // 16px
      lg: '1.125rem',    // 18px
      xl: '1.25rem',     // 20px
      '2xl': '1.5rem',   // 24px
      '3xl': '1.875rem', // 30px
      '4xl': '2.25rem',  // 36px
    },
    
    fontWeight: {
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.625,
    },
    
    letterSpacing: {
      tight: '-0.025em',
      normal: '0',
      wide: '0.025em',
      wider: '0.05em',
      widest: '0.1em',
    }
  },

  // Spacing System (8px base)
  spacing: {
    px: '1px',
    0: '0',
    1: '0.25rem',  // 4px
    2: '0.5rem',   // 8px
    3: '0.75rem',  // 12px
    4: '1rem',     // 16px
    5: '1.25rem',  // 20px
    6: '1.5rem',   // 24px
    8: '2rem',     // 32px
    10: '2.5rem',  // 40px
    12: '3rem',    // 48px
    16: '4rem',    // 64px
    20: '5rem',    // 80px
    24: '6rem',    // 96px
  },

  // Border Radius
  borderRadius: {
    none: '0',
    sm: '0.25rem',   // 4px
    base: '0.5rem',  // 8px
    md: '0.75rem',   // 12px
    lg: '1rem',      // 16px
    xl: '1.5rem',    // 24px
    full: '9999px',
  },

  // Shadows - Sophisticated elevation system
  shadows: {
    none: 'none',
    subtle: '0 1px 2px 0 rgba(17, 24, 39, 0.05)',
    soft: '0 1px 3px 0 rgba(17, 24, 39, 0.08), 0 1px 2px 0 rgba(17, 24, 39, 0.04)',
    medium: '0 4px 6px -1px rgba(17, 24, 39, 0.08), 0 2px 4px -1px rgba(17, 24, 39, 0.04)',
    large: '0 10px 15px -3px rgba(17, 24, 39, 0.08), 0 4px 6px -2px rgba(17, 24, 39, 0.04)',
    elevated: '0 20px 25px -5px rgba(17, 24, 39, 0.08), 0 10px 10px -5px rgba(17, 24, 39, 0.04)',
    inner: 'inset 0 2px 4px 0 rgba(17, 24, 39, 0.06)',
  },

  // Animation & Transitions
  animation: {
    duration: {
      fast: '150ms',
      normal: '200ms',
      slow: '300ms',
      slower: '500ms',
    },
    
    easing: {
      linear: 'linear',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
      easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      subtle: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
    }
  },

  // Layout & Grid
  layout: {
    containerMaxWidth: {
      sm: '640px',
      md: '768px',
      lg: '1024px',
      xl: '1280px',
      '2xl': '1536px',
    },
    
    gridGap: {
      sm: '0.75rem',  // 12px
      md: '1rem',     // 16px
      lg: '1.5rem',   // 24px
      xl: '2rem',     // 32px
    }
  },

  // Component-specific tokens
  components: {
    card: {
      padding: '1.5rem',
      borderWidth: '1px',
      borderRadius: '0.75rem',
    },
    
    button: {
      paddingX: {
        sm: '0.75rem',
        md: '1rem', 
        lg: '1.25rem',
      },
      paddingY: {
        sm: '0.5rem',
        md: '0.625rem',
        lg: '0.75rem',
      },
      borderRadius: '0.5rem',
    },
    
    input: {
      padding: '0.75rem',
      borderRadius: '0.5rem',
      borderWidth: '1px',
    },
    
    table: {
      cellPadding: '0.75rem 1rem',
      headerPadding: '1rem',
      borderRadius: '0.75rem',
    }
  }
} as const;

// Type-safe token access
export type ColorToken = keyof typeof designTokens.colors;
export type SpacingToken = keyof typeof designTokens.spacing;
export type TypographyToken = keyof typeof designTokens.typography;

// Helper functions for token usage
export const getColor = (path: string) => {
  const parts = path.split('.');
  let value: any = designTokens.colors;
  for (const part of parts) {
    value = value[part];
  }
  return value || path;
};

export const getSpacing = (token: SpacingToken) => {
  return designTokens.spacing[token];
};

export const getShadow = (level: keyof typeof designTokens.shadows) => {
  return designTokens.shadows[level];
};