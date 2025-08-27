import React from 'react';
import { clsx } from 'clsx';
import { X } from 'lucide-react';

// Badge Component Types
interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
  rounded?: boolean;
  removable?: boolean;
  onRemove?: () => void;
  className?: string;
}

interface StatusBadgeProps {
  status: 'online' | 'offline' | 'pending' | 'error' | 'warning';
  showDot?: boolean;
  className?: string;
  children?: React.ReactNode;
}

interface MetricBadgeProps {
  value: string | number;
  label?: string;
  trend?: 'up' | 'down' | 'stable';
  className?: string;
}

// Main Badge Component
export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'default',
  size = 'md',
  rounded = true,
  removable = false,
  onRemove,
  className
}) => {
  const baseStyles = [
    'inline-flex items-center font-medium',
    'transition-all duration-200',
    rounded ? 'rounded-full' : 'rounded',
  ];

  const sizeStyles = {
    sm: 'px-2 py-0.5 text-xs gap-1',
    md: 'px-2.5 py-0.5 text-sm gap-1.5',
    lg: 'px-3 py-1 text-base gap-2',
  };

  const variantStyles = {
    default: [
      'bg-graphite-100 text-graphite-700',
      'border border-graphite-200',
    ],
    success: [
      'bg-success-100 text-success-700',
      'border border-success-200',
    ],
    warning: [
      'bg-warning-100 text-warning-700', 
      'border border-warning-200',
    ],
    error: [
      'bg-error-100 text-error-700',
      'border border-error-200',
    ],
    info: [
      'bg-blue-100 text-blue-700',
      'border border-blue-200',
    ],
    secondary: [
      'bg-graphite-600 text-white',
      'shadow-soft',
    ],
  };

  return (
    <span className={clsx(
      baseStyles,
      sizeStyles[size],
      variantStyles[variant],
      className
    )}>
      {children}
      
      {removable && (
        <button
          onClick={onRemove}
          className={clsx(
            'ml-1 -mr-1 flex-shrink-0',
            'hover:bg-black/10 rounded-full p-0.5',
            'transition-colors duration-150'
          )}
          aria-label="Remove"
        >
          <X className="w-3 h-3" />
        </button>
      )}
    </span>
  );
};

// Status Badge Component
export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  showDot = true,
  className,
  children
}) => {
  const statusConfig = {
    online: {
      variant: 'success' as const,
      dot: 'bg-success-500',
      label: 'Online',
    },
    offline: {
      variant: 'default' as const,
      dot: 'bg-graphite-400',
      label: 'Offline',
    },
    pending: {
      variant: 'warning' as const,
      dot: 'bg-warning-500',
      label: 'Pending',
    },
    error: {
      variant: 'error' as const,
      dot: 'bg-error-500',
      label: 'Error',
    },
    warning: {
      variant: 'warning' as const,
      dot: 'bg-warning-500',
      label: 'Warning',
    },
  };

  const config = statusConfig[status];
  const displayText = children || config.label;

  return (
    <Badge variant={config.variant} className={className}>
      {showDot && (
        <span className={clsx(
          'w-2 h-2 rounded-full flex-shrink-0',
          config.dot,
          status === 'pending' && 'animate-pulse'
        )} />
      )}
      {displayText}
    </Badge>
  );
};

// Metric Badge Component
export const MetricBadge: React.FC<MetricBadgeProps> = ({
  value,
  label,
  trend,
  className
}) => {
  const formatValue = (val: string | number) => {
    if (typeof val === 'number') {
      return val.toLocaleString();
    }
    return val;
  };

  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <span className="text-success-600">↗</span>;
      case 'down':
        return <span className="text-error-600">↘</span>;
      case 'stable':
        return <span className="text-graphite-500">→</span>;
      default:
        return null;
    }
  };

  return (
    <Badge variant="default" size="md" className={className}>
      {label && (
        <span className="text-graphite-500">
          {label}:
        </span>
      )}
      <span className="font-semibold">
        {formatValue(value)}
      </span>
      {trend && getTrendIcon()}
    </Badge>
  );
};

// Specialized Badge variants for common use cases
export const SuccessBadge: React.FC<Omit<BadgeProps, 'variant'>> = (props) => (
  <Badge variant="success" {...props} />
);

export const WarningBadge: React.FC<Omit<BadgeProps, 'variant'>> = (props) => (
  <Badge variant="warning" {...props} />
);

export const ErrorBadge: React.FC<Omit<BadgeProps, 'variant'>> = (props) => (
  <Badge variant="error" {...props} />
);

export const InfoBadge: React.FC<Omit<BadgeProps, 'variant'>> = (props) => (
  <Badge variant="info" {...props} />
);

// Count Badge - for notifications and counters
interface CountBadgeProps {
  count: number;
  max?: number;
  showZero?: boolean;
  className?: string;
}

export const CountBadge: React.FC<CountBadgeProps> = ({
  count,
  max = 99,
  showZero = false,
  className
}) => {
  if (count === 0 && !showZero) {
    return null;
  }

  const displayCount = count > max ? `${max}+` : count.toString();

  return (
    <Badge
      variant="error"
      size="sm"
      className={clsx('min-w-[1.25rem] h-5 justify-center px-1', className)}
    >
      {displayCount}
    </Badge>
  );
};

// Tag Badge - for labels and categories
interface TagBadgeProps {
  children: React.ReactNode;
  color?: string;
  removable?: boolean;
  onRemove?: () => void;
  className?: string;
}

export const TagBadge: React.FC<TagBadgeProps> = ({
  children,
  color,
  removable = false,
  onRemove,
  className
}) => {
  const colorStyles = color ? {
    backgroundColor: `${color}20`,
    borderColor: `${color}40`,
    color: color,
  } : {};

  return (
    <Badge
      variant={color ? undefined : 'default'}
      removable={removable}
      onRemove={onRemove}
      className={className}
      style={colorStyles}
    >
      {children}
    </Badge>
  );
};