import { useState } from 'react';
import type { LLMCall } from '@lethe/llm-analyzer-shared';
import { DiffViewer } from './DiffViewer';

interface CallViewProps {
  call: LLMCall;
  onClose?: () => void;
}

type Tab = 'overview' | 'request' | 'response' | 'raw';

export function CallView({ call, onClose }: CallViewProps) {
  const [activeTab, setActiveTab] = useState<Tab>('overview');

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatCost = (cost?: number) => {
    if (!cost) return 'N/A';
    return `$${cost.toFixed(4)}`;
  };

  const getStatusColor = (status: number) => {
    if (status >= 200 && status < 300) return 'bg-green-100 text-green-800';
    if (status >= 400) return 'bg-red-100 text-red-800';
    return 'bg-yellow-100 text-yellow-800';
  };

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'request', label: 'Request' },
    { id: 'response', label: 'Response' },
    { id: 'raw', label: 'Raw Data' },
  ] as const;

  return (
    <div className="bg-white rounded-lg shadow-lg max-w-6xl mx-auto">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <h2 className="text-xl font-semibold text-gray-900">
              Call Details
            </h2>
            <span className={`px-2 py-1 rounded-full text-sm font-medium ${getStatusColor(call.status)}`}>
              {call.status}
            </span>
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
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Basic Info */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <dt className="text-sm font-medium text-gray-500">Provider</dt>
                <dd className="mt-1 text-sm text-gray-900">{call.provider}</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Model</dt>
                <dd className="mt-1 text-sm text-gray-900">{call.model}</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Endpoint</dt>
                <dd className="mt-1 text-sm text-gray-900">{call.endpoint}</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Method</dt>
                <dd className="mt-1 text-sm text-gray-900">{call.method}</dd>
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <dt className="text-sm font-medium text-gray-500">Duration</dt>
                <dd className="mt-1 text-lg font-semibold text-gray-900">
                  {formatDuration(call.duration)}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Total Tokens</dt>
                <dd className="mt-1 text-lg font-semibold text-gray-900">
                  {call.totalTokens?.toLocaleString() || 'N/A'}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Input Tokens</dt>
                <dd className="mt-1 text-lg font-semibold text-gray-900">
                  {call.inputTokens?.toLocaleString() || 'N/A'}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Output Tokens</dt>
                <dd className="mt-1 text-lg font-semibold text-gray-900">
                  {call.outputTokens?.toLocaleString() || 'N/A'}
                </dd>
              </div>
            </div>

            {/* Cost */}
            <div>
              <dt className="text-sm font-medium text-gray-500">Cost</dt>
              <dd className="mt-1 text-2xl font-bold text-gray-900">
                {formatCost(call.cost)}
              </dd>
            </div>

            {/* Timestamp */}
            <div>
              <dt className="text-sm font-medium text-gray-500">Timestamp</dt>
              <dd className="mt-1 text-sm text-gray-900">
                {new Date(call.timestamp).toLocaleString()}
              </dd>
            </div>

            {/* Tags */}
            {call.tags.length > 0 && (
              <div>
                <dt className="text-sm font-medium text-gray-500 mb-2">Tags</dt>
                <div className="flex flex-wrap gap-2">
                  {call.tags.map((tag) => (
                    <span
                      key={tag}
                      className="inline-block px-2 py-1 text-sm bg-blue-100 text-blue-800 rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Error */}
            {call.error && (
              <div className="bg-red-50 border border-red-200 rounded-md p-4">
                <h4 className="text-sm font-medium text-red-800 mb-2">Error</h4>
                <p className="text-sm text-red-700">{call.error.message}</p>
                {call.error.stack && (
                  <pre className="mt-2 text-xs text-red-600 whitespace-pre-wrap">
                    {call.error.stack}
                  </pre>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'request' && (
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-2">Headers</h4>
              <DiffViewer
                content={JSON.stringify(call.requestHeaders, null, 2)}
                language="json"
                readOnly
              />
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-2">Body</h4>
              <DiffViewer
                content={JSON.stringify(call.requestBody, null, 2)}
                language="json"
                readOnly
              />
            </div>
          </div>
        )}

        {activeTab === 'response' && (
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-2">Headers</h4>
              <DiffViewer
                content={JSON.stringify(call.responseHeaders, null, 2)}
                language="json"
                readOnly
              />
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-2">Body</h4>
              <DiffViewer
                content={JSON.stringify(call.responseBody, null, 2)}
                language="json"
                readOnly
              />
            </div>
          </div>
        )}

        {activeTab === 'raw' && (
          <div>
            <DiffViewer
              content={JSON.stringify(call, null, 2)}
              language="json"
              readOnly
            />
          </div>
        )}
      </div>
    </div>
  );
}