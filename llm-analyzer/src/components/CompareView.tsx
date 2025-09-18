import { useState } from 'react';
import DiffViewer from './DiffViewer';
import type { CallPair, DiffResult } from '@lethe/types';

interface CompareViewProps {
  calls: CallPair[];
  comparison: DiffResult;
  onClose?: () => void;
}

type CompareTab = 'prompt' | 'output' | 'params';

export function CompareView({ calls, comparison, onClose }: CompareViewProps) {
  const [activeTab, setActiveTab] = useState<CompareTab>('prompt');

  const tabs: { id: CompareTab; label: string }[] = [
    { id: 'prompt', label: 'Prompt Diff' },
    { id: 'output', label: 'Output Diff' },
    { id: 'params', label: 'Parameter Changes' },
  ];

  return (
    <div className="bg-white rounded-lg shadow-lg max-w-5xl mx-auto">
      <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Compare Calls</h2>
          <p className="text-sm text-gray-500">
            {calls.map((call) => `${call.provider}/${call.model}`).join(' vs. ')}
          </p>
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
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`py-3 text-sm font-medium border-b-2 ${
              activeTab === tab.id
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="p-6">
        {activeTab === 'prompt' && (
          <DiffViewer
            title="Prompt"
            original={calls[0]?.prompt ?? ''}
            modified={calls[1]?.prompt ?? ''}
            language="markdown"
          />
        )}

        {activeTab === 'output' && (
          <DiffViewer
            title="Completion"
            original={calls[0]?.completion ?? ''}
            modified={calls[1]?.completion ?? ''}
            language="markdown"
          />
        )}

        {activeTab === 'params' && (
          <div className="space-y-4">
            {comparison.params_diff ? (
              Object.entries(comparison.params_diff).map(([key, value]) => (
                <div key={key} className="bg-gray-50 border border-gray-200 rounded-md p-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-1">{key}</h3>
                  <p className="text-sm text-gray-600">
                    before: <span className="font-semibold">{String(value.before)}</span>
                  </p>
                  <p className="text-sm text-gray-600">
                    after: <span className="font-semibold">{String(value.after)}</span>
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500">No parameter differences detected.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
