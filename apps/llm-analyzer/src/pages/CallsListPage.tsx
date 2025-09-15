import { useState } from 'react'
import { useUIStore } from '../store/ui'
import CallsList from '../components/CallsList'
import { CallPair } from '../types'

export default function CallsListPage() {
  const { compareMode, toggleCompareMode, clearSelection } = useUIStore()
  const [selectedCall, setSelectedCall] = useState<CallPair | null>(null)

  const handleCallSelect = (call: CallPair) => {
    setSelectedCall(call)
  }

  return (
    <div className="main-layout">
      {/* Sidebar */}
      <div className="sidebar">
        <CallsList 
          onCallSelect={handleCallSelect}
          selectedCallId={selectedCall?.id}
        />
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold">LLM Call Analysis</h1>
            <div className="flex gap-3">
              <button
                onClick={toggleCompareMode}
                className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                  compareMode
                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {compareMode ? 'Exit Compare' : 'Compare Mode'}
              </button>
              {compareMode && (
                <button
                  onClick={clearSelection}
                  className="px-4 py-2 rounded text-sm font-medium bg-gray-200 text-gray-700 hover:bg-gray-300"
                >
                  Clear Selection
                </button>
              )}
            </div>
          </div>

          {compareMode ? (
            <div className="text-center py-12">
              <h2 className="text-xl font-semibold mb-4">Compare Mode Active</h2>
              <p className="text-gray-600 mb-4">
                Select two calls from the sidebar to compare their transformations, 
                parameters, and outputs.
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-md mx-auto">
                <p className="text-sm text-blue-800">
                  💡 <strong>Tip:</strong> This is perfect for A/B testing different 
                  optimization strategies or comparing pre/post transformation effects.
                </p>
              </div>
            </div>
          ) : selectedCall ? (
            <div>
              <h2 className="text-xl font-semibold mb-4">Call Details</h2>
              
              {/* Call Info */}
              <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div>
                    <label className="text-sm font-medium text-gray-500">Query ID</label>
                    <div className="text-sm">{selectedCall.query_id}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500">Provider</label>
                    <div className="text-sm">{selectedCall.provider}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500">Model</label>
                    <div className="text-sm">{selectedCall.model}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500">Status</label>
                    <div className="text-sm">
                      <span className={`pill ${
                        selectedCall.status === 'success' ? 'pill-success' :
                        selectedCall.status === 'error' ? 'pill-error' : 'pill-pending'
                      }`}>
                        {selectedCall.status}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="metrics-strip">
                  <div className="metric">
                    <span className="metric-value">{selectedCall.latency_ms}ms</span> latency
                  </div>
                  <div className="metric">
                    <span className="metric-value">{selectedCall.input_tokens}</span> input tokens
                  </div>
                  <div className="metric">
                    <span className="metric-value">{selectedCall.output_tokens}</span> output tokens
                  </div>
                  <div className="metric">
                    <span className="metric-value">{selectedCall.temperature}</span> temperature
                  </div>
                  <div className="metric">
                    <span className="metric-value">{selectedCall.max_tokens}</span> max tokens
                  </div>
                </div>

                {selectedCall.transform_changes.length > 0 && (
                  <div className="mt-4">
                    <label className="text-sm font-medium text-gray-500">Applied Transformations</label>
                    <div className="flex gap-2 mt-1 flex-wrap">
                      {selectedCall.transform_changes.map((change, i) => (
                        <span key={i} className="pill">
                          {change}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Prompt */}
              <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
                <h3 className="text-lg font-medium mb-3">Prompt</h3>
                <div className="bg-gray-50 rounded p-4 font-mono text-sm whitespace-pre-wrap">
                  {selectedCall.prompt || 'No prompt captured'}
                </div>
              </div>

              {/* Response */}
              {selectedCall.completion && (
                <div className="bg-white rounded-lg border border-gray-200 p-6">
                  <h3 className="text-lg font-medium mb-3">Response</h3>
                  <div className="bg-gray-50 rounded p-4 font-mono text-sm whitespace-pre-wrap">
                    {selectedCall.completion}
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="mt-6 flex gap-3">
                <a
                  href={`/call/${selectedCall.id}`}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  View Full Details
                </a>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <h2 className="text-xl font-semibold mb-4">Welcome to LLM Analyzer</h2>
              <p className="text-gray-600 mb-4">
                Select a call from the sidebar to view its details, transformations, 
                and performance metrics.
              </p>
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 max-w-md mx-auto">
                <p className="text-sm text-gray-700">
                  📊 This tool helps you analyze LLM proxy logs to understand 
                  request patterns, optimization effects, and performance characteristics.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}