import { useState } from 'react';
import DiffViewer from './DiffViewer';
import { LLMCall } from '@ctx-run/core/types';

interface CallComparison {
  metricsDiff: {
    duration: { difference: number; percentageChange: number };
    totalTokens?: { difference: number; percentageChange: number };
    cost?: { difference: number; percentageChange: number };
  };
  requestDiff?: any;
  responseDiff?: any;
}

interface CompareViewProps {
  calls: LLMCall[];
  comparison: CallComparison;
  onClose?: () => void;
}

type CompareTab = 'metrics' | 'request' | 'response';

export function CompareView({ calls, comparison, onClose }: CompareViewProps) {
  const [activeTab, setActiveTab] = useState<CompareTab>('metrics');

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatCost = (cost?: number) => {
    if (!cost) return 'N/A';
    return `$${cost.toFixed(4)}`;
  };

  const formatPercentChange = (change: number) => {
    const sign = change > 0 ? '+' : '';
    const color = change > 0 ? 'text-red-600' : change < 0 ? 'text-green-600' : 'text-gray-600';
    return (
      <span className={color}>
        {sign}{change.toFixed(1)}%
      </span>
    );
  };

  const tabs = [
    { id: 'metrics', label: 'Metrics Comparison' },
    { id: 'request', label: 'Request Diff' },
    { id: 'response', label: 'Response Diff' },
  ] as const;

  return (
    <div className="bg-white rounded-lg shadow-lg max-w-7xl mx-auto">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">
              Compare Calls ({calls.length})
            </h2>
            <div className="mt-1 text-sm text-gray-500">
              {calls.map((call, index) => (
                <span key={call.id} className="mr-4">
                  Call {index + 1}: {call.provider}/{call.model}
                </span>
              ))}
            </div>
          </div>
          
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <span className="sr-only">Close</span>
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8 px-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-3 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="p-6">
        {activeTab === 'metrics' && (
          <div className="space-y-6">
            {/* Call Overview Table */}
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Call
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Provider/Model
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Tokens
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Cost
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {calls.map((call, index) => (
                    <tr key={call.id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Call {index + 1}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {call.provider}/{call.model}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          call.status >= 200 && call.status < 300
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {call.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatDuration(call.duration)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {call.totalTokens?.toLocaleString() || 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatCost(call.cost)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Metrics Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Duration</h4>
                <div className="space-y-1">
                  <div className="text-lg font-semibold">
                    {formatDuration(comparison.metricsDiff.duration.difference)}
                  </div>
                  <div className="text-sm">
                    {formatPercentChange(comparison.metricsDiff.duration.percentageChange)}
                  </div>
                </div>
              </div>

              {comparison.metricsDiff.totalTokens && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Total Tokens</h4>
                  <div className="space-y-1">
                    <div className="text-lg font-semibold">
                      {comparison.metricsDiff.totalTokens.difference > 0 ? '+' : ''}
                      {comparison.metricsDiff.totalTokens.difference.toLocaleString()}
                    </div>
                    <div className="text-sm">
                      {formatPercentChange(comparison.metricsDiff.totalTokens.percentageChange)}
                    </div>
                  </div>
                </div>
              )}

              {comparison.metricsDiff.cost && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Cost</h4>
                  <div className="space-y-1">
                    <div className="text-lg font-semibold">
                      {comparison.metricsDiff.cost.difference > 0 ? '+' : ''}
                      ${comparison.metricsDiff.cost.difference.toFixed(4)}
                    </div>
                    <div className="text-sm">
                      {formatPercentChange(comparison.metricsDiff.cost.percentageChange)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'request' && (
          <div>
            {comparison.requestDiff ? (
              <DiffViewer
                original={JSON.stringify(calls[0]?.requestBody, null, 2)}
                modified={JSON.stringify(calls[1]?.requestBody, null, 2)}
                language="json"
                height={600}
              />
            ) : (
              <div className="text-center py-12 text-gray-500">
                No request differences to show
              </div>
            )}
          </div>
        )}

        {activeTab === 'response' && (
          <div>
            {comparison.responseDiff ? (
              <DiffViewer
                original={JSON.stringify(calls[0]?.responseBody, null, 2)}
                modified={JSON.stringify(calls[1]?.responseBody, null, 2)}
                language="json"
                height={600}
              />
            ) : (
              <div className="text-center py-12 text-gray-500">
                No response differences to show
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}