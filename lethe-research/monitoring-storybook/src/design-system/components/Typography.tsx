import React from 'react';
import { clsx } from 'clsx';

// Typography Component Types
interface TypographyProps {
  children: React.ReactNode;
  className?: string;
  as?: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' | 'p' | 'span' | 'div';
}

interface HeadingProps extends TypographyProps {
  level: 1 | 2 | 3 | 4 | 5 | 6;
}

interface TextProps extends TypographyProps {
  variant?: 'body' | 'caption' | 'label' | 'code';
  size?: 'xs' | 'sm' | 'base' | 'lg' | 'xl';
  weight?: 'light' | 'normal' | 'medium' | 'semibold' | 'bold';
  color?: 'primary' | 'secondary' | 'muted' | 'success' | 'warning' | 'error';
}

// Heading Component
export const Heading: React.FC<HeadingProps> = ({ 
  children, 
  level, 
  className,
  as 
}) => {
  const Component = as || (`h${level}` as const);
  
  const baseStyles = 'font-semibold text-graphite-900 leading-tight tracking-tight';
  
  const sizeStyles = {
    1: 'text-4xl',
    2: 'text-3xl',
    3: 'text-2xl',
    4: 'text-xl',
    5: 'text-lg',
    6: 'text-base font-medium',
  };

  return (
    <Component className={clsx(baseStyles, sizeStyles[level], className)}>
      {children}
    </Component>
  );
};

// Text Component
export const Text: React.FC<TextProps> = ({
  children,
  variant = 'body',
  size = 'base',
  weight = 'normal',
  color = 'primary',
  className,
  as = 'p'
}) => {
  const Component = as;

  const baseStyles = 'leading-normal';
  
  const variantStyles = {
    body: '',
    caption: 'text-sm',
    label: 'text-sm font-medium',
    code: 'font-mono text-sm bg-graphite-100 px-1 py-0.5 rounded',
  };
  
  const sizeStyles = {
    xs: 'text-xs',
    sm: 'text-sm',
    base: 'text-base',
    lg: 'text-lg',
    xl: 'text-xl',
  };
  
  const weightStyles = {
    light: 'font-light',
    normal: 'font-normal',
    medium: 'font-medium',
    semibold: 'font-semibold',
    bold: 'font-bold',
  };
  
  const colorStyles = {
    primary: 'text-graphite-700',
    secondary: 'text-graphite-600',
    muted: 'text-graphite-500',
    success: 'text-success',
    warning: 'text-warning',
    error: 'text-error',
  };

  return (
    <Component 
      className={clsx(
        baseStyles,
        variantStyles[variant],
        sizeStyles[size],
        weightStyles[weight],
        colorStyles[color],
        className
      )}
    >
      {children}
    </Component>
  );
};

// Display Component for large numbers/metrics
interface DisplayProps extends TypographyProps {
  size?: 'lg' | 'xl' | '2xl' | '3xl';
  color?: 'primary' | 'secondary' | 'muted';
}

export const Display: React.FC<DisplayProps> = ({
  children,
  size = 'xl',
  color = 'primary',
  className,
  as = 'div'
}) => {
  const Component = as;
  
  const baseStyles = 'font-bold leading-none tracking-tight';
  
  const sizeStyles = {
    lg: 'text-2xl',
    xl: 'text-3xl',
    '2xl': 'text-4xl', 
    '3xl': 'text-5xl',
  };
  
  const colorStyles = {
    primary: 'text-graphite-900',
    secondary: 'text-graphite-700',
    muted: 'text-graphite-600',
  };

  return (
    <Component 
      className={clsx(
        baseStyles,
        sizeStyles[size],
        colorStyles[color],
        className
      )}
    >
      {children}
    </Component>
  );
};

// Label Component for form labels and metric labels
interface LabelProps extends TypographyProps {
  size?: 'sm' | 'base';
  weight?: 'normal' | 'medium' | 'semibold';
  color?: 'primary' | 'secondary' | 'muted';
  uppercase?: boolean;
}

export const Label: React.FC<LabelProps> = ({
  children,
  size = 'sm',
  weight = 'medium',
  color = 'secondary',
  uppercase = false,
  className,
  as = 'label'
}) => {
  const Component = as;
  
  const baseStyles = 'leading-normal';
  
  const sizeStyles = {
    sm: 'text-sm',
    base: 'text-base',
  };
  
  const weightStyles = {
    normal: 'font-normal',
    medium: 'font-medium',
    semibold: 'font-semibold',
  };
  
  const colorStyles = {
    primary: 'text-graphite-700',
    secondary: 'text-graphite-600',
    muted: 'text-graphite-500',
  };

  return (
    <Component 
      className={clsx(
        baseStyles,
        sizeStyles[size],
        weightStyles[weight],
        colorStyles[color],
        uppercase && 'uppercase tracking-wider',
        className
      )}
    >
      {children}
    </Component>
  );
};

// Code Component for inline and block code
interface CodeProps extends TypographyProps {
  variant?: 'inline' | 'block';
  language?: string;
}

export const Code: React.FC<CodeProps> = ({
  children,
  variant = 'inline',
  language,
  className,
  as
}) => {
  const Component = as || (variant === 'block' ? 'pre' : 'code');
  
  const baseStyles = 'font-mono';
  
  const variantStyles = {
    inline: 'text-sm bg-graphite-100 text-graphite-800 px-1.5 py-0.5 rounded border',
    block: 'text-sm bg-graphite-50 text-graphite-800 p-4 rounded-lg border overflow-x-auto',
  };

  return (
    <Component 
      className={clsx(
        baseStyles,
        variantStyles[variant],
        className
      )}
      data-language={language}
    >
      {variant === 'block' && (
        <code>{children}</code>
      )}
      {variant === 'inline' && children}
    </Component>
  );
};