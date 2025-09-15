import { useParams } from 'react-router-dom'
import { useCall, usePrePostDiff } from '../hooks/api'
import DiffViewer, { JsonDiff } from '../components/DiffViewer'
import { useState } from 'react'

export default function CallViewPage() {
  const { id } = useParams<{ id: string }>()
  const [activeTab, setActiveTab] = useState<'overview' | 'pre-post' | 'raw'>('overview')
  
  const { data: call, isLoading, error } = useCall(id!)
  const { data: prePostDiff } = usePrePostDiff(id!)

  if (isLoading) {
    return <div className="loading">Loading call details...</div>
  }

  if (error || !call) {
    return <div className="error">Error loading call: {error?.message || 'Call not found'}</div>
  }

  return (
    <div className="container">
      <div className="header">
        <h1>Call Details: {call.query_id}</h1>
        <div className="flex gap-2">
          <span className="pill pill-provider">{call.provider}</span>
          <span className="pill pill-model">{call.model}</span>
          <span className={`pill ${call.status === 'success' ? 'pill-success' : 'pill-error'}`}>
            {call.status}
          </span>
        </div>
      </div>

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={`tab ${activeTab === 'pre-post' ? 'active' : ''}`}
          onClick={() => setActiveTab('pre-post')}
        >
          Pre/Post Transform
        </button>
        <button 
          className={`tab ${activeTab === 'raw' ? 'active' : ''}`}
          onClick={() => setActiveTab('raw')}
        >
          Raw Data
        </button>
      </div>

      <div className="main-content">
        {activeTab === 'overview' && (
          <div className="p-6">
            <div className="metrics-strip mb-6">
              <div className="metric">
                <span className="metric-value">{call.latency_ms}ms</span> latency
              </div>
              <div className="metric">
                <span className="metric-value">{call.input_tokens}</span> input tokens
              </div>
              <div className="metric">
                <span className="metric-value">{call.output_tokens}</span> output tokens
              </div>
              <div className="metric">
                <span className="metric-value">{call.temperature}</span> temperature
              </div>
            </div>

            <div className="bg-white rounded border p-4 mb-4">
              <h3 className="font-medium mb-2">Prompt</h3>
              <div className="bg-gray-50 p-3 rounded font-mono text-sm">
                {call.prompt}
              </div>
            </div>

            {call.completion && (
              <div className="bg-white rounded border p-4">
                <h3 className="font-medium mb-2">Response</h3>
                <div className="bg-gray-50 p-3 rounded font-mono text-sm">
                  {call.completion}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'pre-post' && prePostDiff && (
          <div className="p-6">
            <h3 className="text-lg font-medium mb-4">Transformation Analysis</h3>
            
            <div className="mb-6">
              <h4 className="font-medium mb-2">Size Changes</h4>
              <div className="bg-gray-50 p-3 rounded">
                <div>Before: {prePostDiff.size_diff?.pre_bytes} bytes</div>
                <div>After: {prePostDiff.size_diff?.post_bytes} bytes</div>
                <div>Change: +{prePostDiff.size_diff?.change_bytes} bytes ({prePostDiff.size_diff?.change_percent}%)</div>
              </div>
            </div>

            <div className="mb-6">
              <h4 className="font-medium mb-2">Applied Transformations</h4>
              <div className="flex gap-2">
                {prePostDiff.transformations?.map((transform: string, i: number) => (
                  <span key={i} className="pill">
                    {transform}
                  </span>
                ))}
              </div>
            </div>

            {prePostDiff.payload_diff && (
              <JsonDiff
                obj1={call.request?.pre_transform?.payload}
                obj2={call.request?.post_transform?.payload}
                title="Payload Diff (Pre → Post Transform)"
              />
            )}
          </div>
        )}

        {activeTab === 'raw' && (
          <div className="p-6">
            <h3 className="text-lg font-medium mb-4">Raw Call Data</h3>
            <div className="bg-gray-50 p-4 rounded overflow-auto">
              <pre className="text-xs">{JSON.stringify(call, null, 2)}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}