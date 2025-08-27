import React, { useState } from 'react';
import { 
  Activity, 
  BarChart3, 
  Terminal, 
  FileText, 
  Settings,
  RefreshCw,
  Search
} from 'lucide-react';
import { SummaryStatsGrid } from './SummaryStatsGrid';
import { TimelineChart } from './TimelineChart';
import { PerformanceBubbleChart } from './PerformanceBubbleChart';
import { ModelComparisonChart } from './ModelComparisonChart';
import { PromptPerformanceTable } from './PromptPerformanceTable';
import { ExecutionDetailView } from './ExecutionDetailView';
import { CLITerminal } from './CLITerminal';
import { CLIOutputCard } from './CLIOutputCard';
import { 
  SummaryStats,
  TimelineDataPoint,
  PromptPerformance,
  ModelComparison,
  ExecutionComparison,
  CLIOutput 
} from '../types/monitoring';

interface MonitoringDashboardProps {
  summaryStats: SummaryStats;
  timelineData: TimelineDataPoint[];
  promptPerformance: PromptPerformance[];
  modelComparison: ModelComparison[];
  executionComparison: ExecutionComparison;
  cliOutputs: {
    status: CLIOutput;
    list: CLIOutput;
    analyze: CLIOutput;
  };
  className?: string;
}

type TabType = 'overview' | 'performance' | 'executions' | 'cli' | 'settings';

/**
 * Main monitoring dashboard component
 * Combines all monitoring visualizations into a comprehensive interface
 * Based on the PromptDashboard class functionality
 */
export const MonitoringDashboard: React.FC<MonitoringDashboardProps> = ({
  summaryStats,
  timelineData,
  promptPerformance,
  modelComparison,
  executionComparison,
  cliOutputs,
  className
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const tabs: Array<{
    id: TabType;
    label: string;
    icon: React.ReactNode;
    description: string;
  }> = [
    {
      id: 'overview',
      label: 'Overview',
      icon: <Activity className="h-5 w-5" />,
      description: 'Summary statistics and trends'
    },
    {
      id: 'performance',
      label: 'Performance',
      icon: <BarChart3 className="h-5 w-5" />,
      description: 'Charts and analytics'
    },
    {
      id: 'executions',
      label: 'Executions',
      icon: <FileText className="h-5 w-5" />,
      description: 'Detailed execution data'
    },
    {
      id: 'cli',
      label: 'CLI',
      icon: <Terminal className="h-5 w-5" />,
      description: 'Command-line interface'
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <Settings className="h-5 w-5" />,
      description: 'Configuration options'
    }
  ];

  const handleRefresh = async () => {
    setIsRefreshing(true);
    // Simulate refresh delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsRefreshing(false);
  };

  return (
    <div className={`min-h-screen bg-gray-50 ${className}`}>
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center">
                <Search className="h-6 w-6 mr-2" />
                Lethe Prompt Monitor
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Real-time monitoring and analytics for prompt executions
              </p>
            </div>
            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              {isRefreshing ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="px-6">
          <div className="flex space-x-8 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-1 py-4 border-b-2 text-sm font-medium whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="px-6 py-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Summary Stats */}
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Summary Statistics
              </h2>
              <SummaryStatsGrid stats={summaryStats} />
            </section>

            {/* Timeline Chart */}
            <section>
              <TimelineChart data={timelineData} />
            </section>

            {/* Quick Actions */}
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Recent Status
                </h3>
                <CLIOutputCard output={cliOutputs.status} compact />
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Top Prompts
                </h3>
                <div className="space-y-2">
                  {promptPerformance.slice(0, 3).map((prompt, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 truncate">
                        {prompt.prompt_id}
                      </span>
                      <span className="text-xs text-gray-500">
                        {prompt.execution_count}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  System Health
                </h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Success Rate</span>
                    <span className="text-sm font-medium text-green-600">
                      {summaryStats.success_rate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Avg Response</span>
                    <span className="text-sm font-medium text-blue-600">
                      {Math.round(summaryStats.avg_execution_time_ms)}ms
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">24h Activity</span>
                    <span className="text-sm font-medium text-purple-600">
                      {summaryStats.recent_executions_24h}
                    </span>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* Performance Tab */}
        {activeTab === 'performance' && (
          <div className="space-y-6">
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Performance Analytics
              </h2>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <PerformanceBubbleChart data={promptPerformance} />
                <ModelComparisonChart data={modelComparison} />
              </div>
            </section>

            <section>
              <PromptPerformanceTable data={promptPerformance} />
            </section>
          </div>
        )}

        {/* Executions Tab */}
        {activeTab === 'executions' && (
          <div className="space-y-6">
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Execution Details
              </h2>
              <ExecutionDetailView comparison={executionComparison} />
            </section>
          </div>
        )}

        {/* CLI Tab */}
        {activeTab === 'cli' && (
          <div className="space-y-6">
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Command Line Interface
              </h2>
              <CLITerminal 
                outputs={[cliOutputs.status, cliOutputs.list, cliOutputs.analyze]} 
                interactive={true}
              />
            </section>

            <section>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Recent Commands
              </h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <CLIOutputCard output={cliOutputs.list} />
                <CLIOutputCard output={cliOutputs.analyze} />
              </div>
            </section>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div className="space-y-6">
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Configuration Settings
              </h2>
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-base font-medium text-gray-900 mb-4">
                      Database Configuration
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Database Path
                        </label>
                        <input
                          type="text"
                          value="experiments/prompt_tracking.db"
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                          readOnly
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Auto-cleanup (days)
                        </label>
                        <input
                          type="number"
                          value="30"
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-base font-medium text-gray-900 mb-4">
                      Monitoring Settings
                    </h3>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="text-sm font-medium text-gray-900">
                            Real-time Updates
                          </h4>
                          <p className="text-sm text-gray-500">
                            Automatically refresh data every 30 seconds
                          </p>
                        </div>
                        <input
                          type="checkbox"
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          defaultChecked
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="text-sm font-medium text-gray-900">
                            Performance Alerts
                          </h4>
                          <p className="text-sm text-gray-500">
                            Notify when execution time exceeds threshold
                          </p>
                        </div>
                        <input
                          type="checkbox"
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          defaultChecked
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-base font-medium text-gray-900 mb-4">
                      Export Options
                    </h3>
                    <div className="flex space-x-4">
                      <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
                        Export as JSON
                      </button>
                      <button className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors">
                        Export as CSV
                      </button>
                      <button className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors">
                        Generate Report
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
};

export default MonitoringDashboard;