import React from 'react';
import { clsx } from 'clsx';

// Card Component Types
interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  shadow?: 'none' | 'subtle' | 'soft' | 'medium' | 'large' | 'elevated';
  border?: boolean;
  borderRadius?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
  hover?: boolean;
  interactive?: boolean;
  as?: 'div' | 'article' | 'section';
}

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'sm' | 'md' | 'lg';
}

interface CardBodyProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'sm' | 'md' | 'lg';
}

interface CardFooterProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'sm' | 'md' | 'lg';
  border?: boolean;
}

// Main Card Component
export const Card: React.FC<CardProps> = ({
  children,
  className,
  padding = 'md',
  shadow = 'soft',
  border = true,
  borderRadius = 'lg',
  hover = false,
  interactive = false,
  as: Component = 'div'
}) => {
  const baseStyles = [
    'bg-surface-primary',
    'transition-all duration-200',
    interactive && 'cursor-pointer',
  ];

  const paddingStyles = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };

  const shadowStyles = {
    none: 'shadow-none',
    subtle: 'shadow-subtle',
    soft: 'shadow-soft',
    medium: 'shadow-medium',
    large: 'shadow-large',
    elevated: 'shadow-elevated',
  };

  const borderRadiusStyles = {
    none: 'rounded-none',
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    xl: 'rounded-xl',
  };

  const hoverStyles = hover && [
    'hover:shadow-medium',
    'hover:-translate-y-0.5',
  ];

  const interactiveStyles = interactive && [
    'hover:shadow-large',
    'active:translate-y-0',
    'focus-ring',
  ];

  return (
    <Component
      className={clsx(
        baseStyles,
        paddingStyles[padding],
        shadowStyles[shadow],
        borderRadiusStyles[borderRadius],
        border && 'border border-graphite-200',
        hoverStyles,
        interactiveStyles,
        className
      )}
      tabIndex={interactive ? 0 : undefined}
      role={interactive ? 'button' : undefined}
    >
      {children}
    </Component>
  );
};

// Card Header Component
export const CardHeader: React.FC<CardHeaderProps> = ({
  children,
  className,
  padding = 'md'
}) => {
  const paddingStyles = {
    sm: 'p-4 pb-2',
    md: 'p-6 pb-3',
    lg: 'p-8 pb-4',
  };

  return (
    <div className={clsx(
      'border-b border-graphite-200',
      paddingStyles[padding],
      className
    )}>
      {children}
    </div>
  );
};

// Card Body Component
export const CardBody: React.FC<CardBodyProps> = ({
  children,
  className,
  padding = 'md'
}) => {
  const paddingStyles = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };

  return (
    <div className={clsx(paddingStyles[padding], className)}>
      {children}
    </div>
  );
};

// Card Footer Component
export const CardFooter: React.FC<CardFooterProps> = ({
  children,
  className,
  padding = 'md',
  border = true
}) => {
  const paddingStyles = {
    sm: 'p-4 pt-2',
    md: 'p-6 pt-3',
    lg: 'p-8 pt-4',
  };

  return (
    <div className={clsx(
      paddingStyles[padding],
      border && 'border-t border-graphite-200',
      className
    )}>
      {children}
    </div>
  );
};

// Specialized Card Components

// Metric Card - for displaying key performance indicators
interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
  icon?: React.ReactNode;
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  icon,
  className
}) => {
  const trendColors = {
    up: 'text-success-600',
    down: 'text-error-600',
    stable: 'text-graphite-500',
  };

  const formatValue = (val: string | number) => {
    if (typeof val === 'number') {
      return val.toLocaleString();
    }
    return val;
  };

  return (
    <Card className={className} hover>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-graphite-600 leading-none">
          {title}
        </h3>
        {icon && (
          <div className="text-graphite-400 flex-shrink-0">
            {icon}
          </div>
        )}
      </div>
      
      <div className="mb-2">
        <p className="text-3xl font-bold text-graphite-900 leading-none">
          {formatValue(value)}
        </p>
      </div>
      
      {subtitle && (
        <p className="text-sm text-graphite-500 mb-3">
          {subtitle}
        </p>
      )}
      
      {(trend || trendValue) && (
        <div className="flex items-center gap-1">
          {trend && (
            <div className={clsx('text-sm', trendColors[trend])}>
              {trend === 'up' && '↗'}
              {trend === 'down' && '↘'}
              {trend === 'stable' && '→'}
            </div>
          )}
          {trendValue && (
            <span className={clsx('text-sm font-medium', trend ? trendColors[trend] : 'text-graphite-600')}>
              {trendValue}
            </span>
          )}
        </div>
      )}
    </Card>
  );
};

// Chart Card - for data visualization components
interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  headerActions?: React.ReactNode;
}

export const ChartCard: React.FC<ChartCardProps> = ({
  title,
  subtitle,
  children,
  className,
  headerActions
}) => {
  return (
    <Card className={className} padding="none">
      <CardHeader className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-graphite-900">
            {title}
          </h3>
          {subtitle && (
            <p className="text-sm text-graphite-600 mt-1">
              {subtitle}
            </p>
          )}
        </div>
        {headerActions && (
          <div className="flex items-center gap-2">
            {headerActions}
          </div>
        )}
      </CardHeader>
      
      <CardBody>
        {children}
      </CardBody>
    </Card>
  );
};

// Status Card - for system status and alerts
interface StatusCardProps {
  title: string;
  status: 'success' | 'warning' | 'error' | 'info';
  message?: string;
  children?: React.ReactNode;
  className?: string;
}

export const StatusCard: React.FC<StatusCardProps> = ({
  title,
  status,
  message,
  children,
  className
}) => {
  const statusStyles = {
    success: {
      bg: 'bg-success-50',
      border: 'border-success-200',
      text: 'text-success-700',
      dot: 'bg-success-500',
    },
    warning: {
      bg: 'bg-warning-50',
      border: 'border-warning-200',
      text: 'text-warning-700',
      dot: 'bg-warning-500',
    },
    error: {
      bg: 'bg-error-50',
      border: 'border-error-200',
      text: 'text-error-700',
      dot: 'bg-error-500',
    },
    info: {
      bg: 'bg-graphite-50',
      border: 'border-graphite-200',
      text: 'text-graphite-700',
      dot: 'bg-graphite-500',
    },
  };

  const styles = statusStyles[status];

  return (
    <div className={clsx(
      'border rounded-lg p-4',
      styles.bg,
      styles.border,
      className
    )}>
      <div className="flex items-start gap-3">
        <div className={clsx('w-2 h-2 rounded-full mt-2 flex-shrink-0', styles.dot)} />
        <div className="flex-1">
          <h4 className={clsx('font-medium mb-1', styles.text)}>
            {title}
          </h4>
          {message && (
            <p className={clsx('text-sm', styles.text)}>
              {message}
            </p>
          )}
          {children && (
            <div className="mt-3">
              {children}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};