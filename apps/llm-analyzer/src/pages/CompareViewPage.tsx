import { useSearchParams } from 'react-router-dom'
import { useCallComparison, useCall } from '../hooks/api'
import DiffViewer, { JsonDiff, TextDiff } from '../components/DiffViewer'
import { useState } from 'react'

export default function CompareViewPage() {
  const [searchParams] = useSearchParams()
  const callIdA = searchParams.get('a')
  const callIdB = searchParams.get('b')
  const [activeTab, setActiveTab] = useState<'prompts' | 'context' | 'params' | 'performance'>('prompts')

  const { data: callA } = useCall(callIdA || '')
  const { data: callB } = useCall(callIdB || '')
  const { data: comparison, isLoading, error } = useCallComparison(callIdA || undefined, callIdB || undefined)

  if (!callIdA || !callIdB) {
    return (
      <div className="container">
        <div className="error">
          Missing call IDs for comparison. Please provide both call IDs as URL parameters.
        </div>
      </div>
    )
  }

  if (isLoading) {
    return <div className="loading">Loading comparison...</div>
  }

  if (error || !comparison) {
    return <div className="error">Error loading comparison: {error?.message || 'Comparison failed'}</div>
  }

  return (
    <div className="container">
      <div className="header">
        <h1>Call Comparison</h1>
        <div className="text-sm text-gray-600">
          Comparing {callIdA} vs {callIdB}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white rounded border p-4">
          <h3 className="font-medium mb-2">Call A</h3>
          <div className="text-sm space-y-1">
            <div>ID: {comparison.metadata?.callA?.id}</div>
            <div>Provider: {comparison.metadata?.callA?.provider}</div>
            <div>Model: {comparison.metadata?.callA?.model}</div>
            <div>Time: {new Date(comparison.metadata?.callA?.timestamp).toLocaleString()}</div>
          </div>
        </div>
        <div className="bg-white rounded border p-4">
          <h3 className="font-medium mb-2">Call B</h3>
          <div className="text-sm space-y-1">
            <div>ID: {comparison.metadata?.callB?.id}</div>
            <div>Provider: {comparison.metadata?.callB?.provider}</div>
            <div>Model: {comparison.metadata?.callB?.model}</div>
            <div>Time: {new Date(comparison.metadata?.callB?.timestamp).toLocaleString()}</div>
          </div>
        </div>
      </div>

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'prompts' ? 'active' : ''}`}
          onClick={() => setActiveTab('prompts')}
        >
          Prompts
        </button>
        <button 
          className={`tab ${activeTab === 'context' ? 'active' : ''}`}
          onClick={() => setActiveTab('context')}
        >
          Context
        </button>
        <button 
          className={`tab ${activeTab === 'params' ? 'active' : ''}`}
          onClick={() => setActiveTab('params')}
        >
          Parameters
        </button>
        <button 
          className={`tab ${activeTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveTab('performance')}
        >
          Performance
        </button>
      </div>

      <div className="main-content">
        {activeTab === 'prompts' && comparison.prompt_diff && (
          <div className="p-6">
            <h3 className="text-lg font-medium mb-4">Prompt Comparison</h3>
            <div className="bg-gray-50 p-4 mb-4 rounded">
              <div className="text-sm">
                Similarity: <span className="font-medium">{comparison.prompt_diff.similarity?.toFixed(1)}%</span>
              </div>
            </div>
            <div 
              className="border rounded p-4 text-sm"
              dangerouslySetInnerHTML={{ __html: comparison.prompt_diff.text || 'No differences' }}
            />
          </div>
        )}

        {activeTab === 'context' && (
          <div className="p-6">
            <h3 className="text-lg font-medium mb-4">Context Comparison</h3>
            
            <div className="mb-6">
              <h4 className="font-medium mb-2">Pre-Transform Context</h4>
              {comparison.context_diff?.pre?.hasChanges ? (
                <JsonDiff
                  obj1={callA?.pre_context || []}
                  obj2={callB?.pre_context || []}
                  title="Pre-Transform Context Diff"
                />
              ) : (
                <div className="text-gray-600 p-4 bg-gray-50 rounded">No differences in pre-transform context</div>
              )}
            </div>

            <div>
              <h4 className="font-medium mb-2">Post-Transform Context</h4>
              {comparison.context_diff?.post?.hasChanges ? (
                <JsonDiff
                  obj1={callA?.post_context || []}
                  obj2={callB?.post_context || []}
                  title="Post-Transform Context Diff"
                />
              ) : (
                <div className="text-gray-600 p-4 bg-gray-50 rounded">No differences in post-transform context</div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'params' && comparison.params_diff && (
          <div className="p-6">
            <h3 className="text-lg font-medium mb-4">Parameter Comparison</h3>
            {comparison.params_diff.hasChanges ? (
              <div className="bg-gray-50 p-4 rounded">
                <pre className="text-sm">{JSON.stringify(comparison.params_diff.delta, null, 2)}</pre>
              </div>
            ) : (
              <div className="text-gray-600 p-4 bg-gray-50 rounded">No parameter differences</div>
            )}
          </div>
        )}

        {activeTab === 'performance' && comparison.performance_diff && (
          <div className="p-6">
            <h3 className="text-lg font-medium mb-4">Performance Comparison</h3>
            {comparison.performance_diff.hasChanges ? (
              <div className="bg-gray-50 p-4 rounded">
                <pre className="text-sm">{JSON.stringify(comparison.performance_diff.delta, null, 2)}</pre>
              </div>
            ) : (
              <div className="text-gray-600 p-4 bg-gray-50 rounded">No performance differences</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}