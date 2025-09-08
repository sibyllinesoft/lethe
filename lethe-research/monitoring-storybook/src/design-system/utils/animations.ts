/**
 * Animation Utilities
 * Consistent animation patterns and easing functions
 */

import { designTokens } from '../tokens';

// Animation configurations
export const animations = {
  // Durations (in CSS)
  duration: {
    instant: '0ms',
    fast: '150ms',
    normal: '200ms',
    slow: '300ms',
    slower: '500ms',
    slowest: '1000ms',
  },

  // Easing functions
  easing: {
    linear: 'linear',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    subtle: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
    spring: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },

  // Pre-defined transitions
  transitions: {
    all: 'all 200ms cubic-bezier(0.4, 0, 0.2, 1)',
    colors: 'color, background-color, border-color 200ms cubic-bezier(0.4, 0, 0.2, 1)',
    shadow: 'box-shadow 200ms cubic-bezier(0.4, 0, 0.2, 1)',
    transform: 'transform 200ms cubic-bezier(0.4, 0, 0.2, 1)',
    opacity: 'opacity 150ms cubic-bezier(0.4, 0, 0.2, 1)',
  },

  // CSS Keyframes (as CSS-in-JS objects)
  keyframes: {
    fadeIn: {
      from: { opacity: 0 },
      to: { opacity: 1 },
    },
    
    fadeOut: {
      from: { opacity: 1 },
      to: { opacity: 0 },
    },
    
    slideUp: {
      from: { 
        opacity: 0,
        transform: 'translateY(10px)',
      },
      to: { 
        opacity: 1,
        transform: 'translateY(0)',
      },
    },
    
    slideDown: {
      from: { 
        opacity: 0,
        transform: 'translateY(-10px)',
      },
      to: { 
        opacity: 1,
        transform: 'translateY(0)',
      },
    },
    
    slideLeft: {
      from: { 
        opacity: 0,
        transform: 'translateX(10px)',
      },
      to: { 
        opacity: 1,
        transform: 'translateX(0)',
      },
    },
    
    slideRight: {
      from: { 
        opacity: 0,
        transform: 'translateX(-10px)',
      },
      to: { 
        opacity: 1,
        transform: 'translateX(0)',
      },
    },
    
    scaleIn: {
      from: { 
        opacity: 0,
        transform: 'scale(0.95)',
      },
      to: { 
        opacity: 1,
        transform: 'scale(1)',
      },
    },
    
    scaleOut: {
      from: { 
        opacity: 1,
        transform: 'scale(1)',
      },
      to: { 
        opacity: 0,
        transform: 'scale(0.95)',
      },
    },
    
    pulse: {
      '0%, 100%': { opacity: 1 },
      '50%': { opacity: 0.6 },
    },
    
    bounce: {
      '0%, 100%': { transform: 'translateY(0)' },
      '50%': { transform: 'translateY(-4px)' },
    },
    
    spin: {
      from: { transform: 'rotate(0deg)' },
      to: { transform: 'rotate(360deg)' },
    },
    
    // Loading animations
    skeleton: {
      '0%': { backgroundPosition: '200% 0' },
      '100%': { backgroundPosition: '-200% 0' },
    },
    
    shimmer: {
      '0%': { transform: 'translateX(-100%)' },
      '100%': { transform: 'translateX(100%)' },
    },
  },

  // Pre-configured animation classes
  classes: {
    fadeIn: 'animate-fade-in',
    fadeOut: 'animate-fade-out',
    slideUp: 'animate-slide-up',
    slideDown: 'animate-slide-down',
    slideLeft: 'animate-slide-left',
    slideRight: 'animate-slide-right',
    scaleIn: 'animate-scale-in',
    scaleOut: 'animate-scale-out',
    pulse: 'animate-pulse',
    bounce: 'animate-bounce',
    spin: 'animate-spin',
  },

  // Stagger utilities for sequential animations
  stagger: {
    children: (delay = 50) => `animation-delay: ${delay}ms`,
    sequence: (index: number, delay = 100) => `animation-delay: ${index * delay}ms`,
  },

  // Hover animations
  hover: {
    lift: 'transition-transform duration-200 hover:-translate-y-1',
    scale: 'transition-transform duration-200 hover:scale-105',
    glow: 'transition-shadow duration-200 hover:shadow-large',
    fadeIn: 'transition-opacity duration-200 hover:opacity-100',
  },

  // Loading states
  loading: {
    spinner: 'animate-spin',
    pulse: 'animate-pulse',
    bounce: 'animate-bounce',
    skeleton: 'animate-skeleton',
  },

  // Focus animations
  focus: {
    ring: 'transition-shadow duration-150 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
    scale: 'transition-transform duration-150 focus:scale-105',
  },
} as const;

// CSS custom properties for animations (to be used in CSS)
export const animationCSSVars = {
  '--duration-fast': '150ms',
  '--duration-normal': '200ms', 
  '--duration-slow': '300ms',
  '--easing-ease-out': 'cubic-bezier(0, 0, 0.2, 1)',
  '--easing-ease-in-out': 'cubic-bezier(0.4, 0, 0.2, 1)',
  '--easing-subtle': 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
} as const;

// Utility functions
export const createTransition = (
  property: string,
  duration: keyof typeof animations.duration = 'normal',
  easing: keyof typeof animations.easing = 'easeInOut'
) => {
  return `${property} ${animations.duration[duration]} ${animations.easing[easing]}`;
};

export const createDelay = (delay: number, unit: 'ms' | 's' = 'ms') => {
  return `${delay}${unit}`;
};

// Animation presets for common UI patterns
export const presets = {
  // Card hover effects
  cardHover: {
    transition: createTransition('all'),
    ':hover': {
      transform: 'translateY(-2px)',
      boxShadow: designTokens.shadows.large,
    },
  },

  // Button press effects
  buttonPress: {
    transition: createTransition('transform', 'fast'),
    ':active': {
      transform: 'scale(0.98)',
    },
  },

  // Modal entrance
  modalEnter: {
    animationName: 'slideUp',
    animationDuration: animations.duration.normal,
    animationTimingFunction: animations.easing.easeOut,
  },

  // Page transitions
  pageTransition: {
    animationName: 'fadeIn',
    animationDuration: animations.duration.slow,
    animationTimingFunction: animations.easing.easeOut,
  },
} as const;