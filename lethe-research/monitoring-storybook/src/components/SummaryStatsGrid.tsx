import React from 'react';
import { 
  Activity, 
  Target, 
  CheckCircle, 
  Zap, 
  Clock 
} from 'lucide-react';
import { DashboardCard } from './DashboardCard';
import { SummaryStats } from '../types/monitoring';
import { layoutUtils } from '../design-system';

interface SummaryStatsGridProps {
  stats: SummaryStats;
  className?: string;
}

/**
 * Grid of summary statistics cards
 * Matches the data structure from PromptDashboard.get_summary_stats()
 * Now using sophisticated graphite design system layout
 */
export const SummaryStatsGrid: React.FC<SummaryStatsGridProps> = ({ 
  stats, 
  className 
}) => {
  const formatExecutionTime = (ms: number): string => {
    if (ms < 1000) {
      return `${Math.round(ms)}ms`;
    }
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className={`${layoutUtils.cardGrid.metrics} ${className || ''}`}>
      <DashboardCard
        title="Total Executions"
        value={stats.total_executions.toLocaleString()}
        subtitle="All-time prompt executions"
        icon={<Activity className="h-8 w-8" />}
        trend="up"
        trendValue="+12% this week"
      />
      
      <DashboardCard
        title="Unique Prompts"
        value={stats.unique_prompts}
        subtitle="Distinct prompt templates"
        icon={<Target className="h-8 w-8" />}
        trend="stable"
        trendValue="2 new this week"
      />
      
      <DashboardCard
        title="Success Rate"
        value={`${stats.success_rate.toFixed(1)}%`}
        subtitle="Successful executions"
        icon={<CheckCircle className="h-8 w-8" />}
        trend={stats.success_rate >= 95 ? 'up' : stats.success_rate >= 90 ? 'stable' : 'down'}
        trendValue={stats.success_rate >= 95 ? 'Excellent' : stats.success_rate >= 90 ? 'Good' : 'Needs attention'}
      />
      
      <DashboardCard
        title="Avg Response Time"
        value={formatExecutionTime(stats.avg_execution_time_ms)}
        subtitle="Mean execution time"
        icon={<Zap className="h-8 w-8" />}
        trend={stats.avg_execution_time_ms < 500 ? 'up' : stats.avg_execution_time_ms < 1000 ? 'stable' : 'down'}
        trendValue={stats.avg_execution_time_ms < 500 ? 'Fast' : stats.avg_execution_time_ms < 1000 ? 'Good' : 'Slow'}
      />
      
      <DashboardCard
        title="Recent Activity"
        value={stats.recent_executions_24h}
        subtitle="Last 24 hours"
        icon={<Clock className="h-8 w-8" />}
        trend="up"
        trendValue="Active"
      />
    </div>
  );
};

export default SummaryStatsGrid;