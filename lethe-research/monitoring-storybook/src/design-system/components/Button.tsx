import React from 'react';
import { clsx } from 'clsx';
import { Loader2 } from 'lucide-react';

// Button Component Types
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'outline' | 'danger';
  size?: 'xs' | 'sm' | 'base' | 'lg';
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
  children: React.ReactNode;
}

interface ButtonGroupProps {
  children: React.ReactNode;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
  spacing?: 'none' | 'sm' | 'md';
}

// Button Component
export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'base',
  loading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  disabled,
  className,
  children,
  ...props
}) => {
  const baseStyles = [
    'inline-flex items-center justify-center',
    'font-medium leading-none',
    'transition-all duration-200',
    'focus-ring',
    'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
    fullWidth && 'w-full'
  ];

  const sizeStyles = {
    xs: 'px-2.5 py-1.5 text-xs gap-1 rounded',
    sm: 'px-3 py-2 text-sm gap-1.5 rounded',
    base: 'px-4 py-2.5 text-base gap-2 rounded-md',
    lg: 'px-6 py-3 text-lg gap-2.5 rounded-md',
  };

  const variantStyles = {
    primary: [
      'bg-graphite-900 text-white',
      'hover:bg-graphite-800',
      'active:bg-graphite-950',
      'shadow-soft',
      'hover:shadow-medium',
    ],
    secondary: [
      'bg-surface-primary text-graphite-700',
      'border border-graphite-300',
      'hover:bg-graphite-50',
      'active:bg-graphite-100',
      'shadow-soft',
      'hover:shadow-medium',
    ],
    ghost: [
      'text-graphite-600',
      'hover:bg-graphite-100',
      'active:bg-graphite-200',
    ],
    outline: [
      'border-2 border-graphite-300 text-graphite-700 bg-transparent',
      'hover:border-graphite-400 hover:bg-graphite-50',
      'active:border-graphite-500 active:bg-graphite-100',
    ],
    danger: [
      'bg-error-600 text-white',
      'hover:bg-error-700',
      'active:bg-error-800',
      'shadow-soft',
      'hover:shadow-medium',
    ],
  };

  return (
    <button
      className={clsx(
        baseStyles,
        sizeStyles[size],
        variantStyles[variant],
        className
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : (
        leftIcon && <span className="flex-shrink-0">{leftIcon}</span>
      )}
      
      {children}
      
      {!loading && rightIcon && (
        <span className="flex-shrink-0">{rightIcon}</span>
      )}
    </button>
  );
};

// Icon Button Component (for buttons with only icons)
interface IconButtonProps extends Omit<ButtonProps, 'leftIcon' | 'rightIcon' | 'children'> {
  icon: React.ReactNode;
  'aria-label': string;
}

export const IconButton: React.FC<IconButtonProps> = ({
  icon,
  variant = 'ghost',
  size = 'base',
  className,
  ...props
}) => {
  const sizeStyles = {
    xs: 'p-1',
    sm: 'p-1.5',
    base: 'p-2',
    lg: 'p-3',
  };

  return (
    <Button
      variant={variant}
      size={size}
      className={clsx('!px-0 !py-0', sizeStyles[size], className)}
      {...props}
    >
      {icon}
    </Button>
  );
};

// Button Group Component
export const ButtonGroup: React.FC<ButtonGroupProps> = ({
  children,
  orientation = 'horizontal',
  spacing = 'sm',
  className
}) => {
  const baseStyles = 'inline-flex items-center';
  
  const orientationStyles = {
    horizontal: 'flex-row',
    vertical: 'flex-col',
  };
  
  const spacingStyles = {
    none: 'gap-0',
    sm: orientation === 'horizontal' ? 'gap-2' : 'gap-1',
    md: orientation === 'horizontal' ? 'gap-3' : 'gap-2',
  };

  return (
    <div className={clsx(
      baseStyles,
      orientationStyles[orientation],
      spacingStyles[spacing],
      className
    )}>
      {children}
    </div>
  );
};

// Button variants for specific use cases
export const PrimaryButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="primary" {...props} />
);

export const SecondaryButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="secondary" {...props} />
);

export const GhostButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="ghost" {...props} />
);

export const OutlineButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="outline" {...props} />
);

export const DangerButton: React.FC<Omit<ButtonProps, 'variant'>> = (props) => (
  <Button variant="danger" {...props} />
);