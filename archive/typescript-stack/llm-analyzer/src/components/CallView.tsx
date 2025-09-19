import { useState } from 'react';
import type { CallPair } from '@lethe/types';
import { ContentViewer } from './DiffViewer';

interface CallViewProps {
  call: CallPair;
  onClose?: () => void;
}

type Tab = 'overview' | 'request' | 'response' | 'raw';

export function CallView({ call, onClose }: CallViewProps) {
  const [activeTab, setActiveTab] = useState<Tab>('overview');

  const formatTokens = (tokens?: number) => (tokens ? tokens.toLocaleString() : 'N/A');

  const tabs: Tab[] = ['overview', 'request', 'response', 'raw'];

  return (
    <div className="bg-white rounded-lg shadow-lg max-w-5xl mx-auto">
      <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Call Details</h2>
          <p className="text-sm text-gray-500">{new Date(call.timestamp).toLocaleString()}</p>
        </div>
        {onClose && (
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600" aria-label="Close">
            ✕
          </button>
        )}
      </div>

      <nav className="border-b border-gray-200 px-6 flex space-x-6">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`py-3 text-sm font-medium border-b-2 ${
              activeTab === tab
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </nav>

      <div className="p-6 space-y-4">
        {activeTab === 'overview' && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <InfoBlock label="Provider" value={call.provider} />
              <InfoBlock label="Model" value={call.model} />
              <InfoBlock label="Run" value={call.run_id} />
              <InfoBlock label="Query" value={call.query_id} />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <InfoBlock label="Status" value={call.status} />
              <InfoBlock label="Latency" value={`${call.latency_ms} ms`} />
              <InfoBlock label="Input Tokens" value={formatTokens(call.input_tokens)} />
              <InfoBlock label="Output Tokens" value={formatTokens(call.output_tokens)} />
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-1">Prompt</h3>
              <ContentViewer content={call.prompt} language="markdown" />
            </div>

            {call.completion && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-1">Completion</h3>
                <ContentViewer content={call.completion} language="markdown" />
              </div>
            )}

            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-1">Transform Changes</h3>
              {call.transform_changes.length ? (
                <ul className="list-disc list-inside text-sm text-gray-700">
                  {call.transform_changes.map((change) => (
                    <li key={change}>{change}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-gray-500">No transforms recorded</p>
              )}
            </div>
          </div>
        )}

        {activeTab === 'request' && (
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-2">Pre-transform payload</h4>
              <ContentViewer
                content={JSON.stringify(call.request.pre_transform.payload, null, 2)}
                language="json"
              />
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-2">Post-transform payload</h4>
              <ContentViewer
                content={JSON.stringify(call.request.post_transform.payload, null, 2)}
                language="json"
              />
            </div>
          </div>
        )}

        {activeTab === 'response' && (
          <div className="space-y-4">
            {call.response ? (
              <>
                <div>
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Status</h4>
                  <p className="text-sm text-gray-700">HTTP {call.response.status_code}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Performance</h4>
                  <ContentViewer
                    content={JSON.stringify(call.response.performance, null, 2)}
                    language="json"
                  />
                </div>
              </>
            ) : (
              <p className="text-sm text-gray-500">No response recorded for this call.</p>
            )}
          </div>
        )}

        {activeTab === 'raw' && (
          <ContentViewer content={JSON.stringify(call, null, 2)} language="json" />
        )}
      </div>
    </div>
  );
}

interface InfoBlockProps {
  label: string;
  value?: string | number;
}

function InfoBlock({ label, value }: InfoBlockProps) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-gray-500">{label}</dt>
      <dd className="mt-1 text-sm text-gray-900">{value ?? 'N/A'}</dd>
    </div>
  );
}
