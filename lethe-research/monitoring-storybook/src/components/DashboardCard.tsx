import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { clsx } from 'clsx';
import { DashboardCardProps } from '../types/monitoring';
import { MetricCard } from '../design-system';

/**
 * Dashboard card component for displaying key metrics
 * Based on the summary statistics from PromptDashboard.get_summary_stats()
 * Now using the sophisticated graphite design system
 */
export const DashboardCard: React.FC<DashboardCardProps> = ({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  icon,
  className
}) => {
  // Use the new MetricCard component from the design system
  return (
    <MetricCard
      title={title}
      value={value}
      subtitle={subtitle}
      trend={trend}
      trendValue={trendValue}
      icon={icon}
      className={className}
    />
  );
};

export default DashboardCard;