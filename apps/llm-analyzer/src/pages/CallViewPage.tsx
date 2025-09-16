import { useParams } from 'react-router-dom'
import { useCall, usePrePostDiff } from '../hooks/api'
import { JsonDiff } from '../components/DiffViewer'
import { useState } from 'react'

// Helper function to check answer accuracy with improved matching and debugging
function checkAnswerAccuracy(response: string, goldenAnswer: string | string[]): { result: string, debug: any } {
  const goldenAnswers = Array.isArray(goldenAnswer) ? goldenAnswer : [goldenAnswer]
  const responseLower = response.toLowerCase().trim()
  const responseWords = responseLower.split(/\s+/)
  
  // Debug info
  const debug = {
    response_length: response.length,
    response_preview: response.substring(0, 200) + (response.length > 200 ? '...' : ''),
    golden_answers: goldenAnswers,
    exact_matches: [] as string[],
    partial_matches: [] as Array<{ answer: string; matched_words: string[]; total_words: number }>,
    word_matches: [] as string[]
  }
  
  // Check for exact matches (case-insensitive)
  const exactMatches = goldenAnswers.filter(answer => {
    const exactMatch = responseLower.includes(answer.toLowerCase())
    if (exactMatch) debug.exact_matches.push(answer)
    return exactMatch
  })
  
  // Check for partial matches (individual words)
  goldenAnswers.forEach(answer => {
    const answerWords = answer.toLowerCase().split(/[._\s]+/).filter(w => w.length > 1)
    const matchedWords = answerWords.filter(word => responseWords.includes(word))
    if (matchedWords.length > 0) {
      debug.partial_matches.push({ answer, matched_words: matchedWords, total_words: answerWords.length })
    }
  })
  
  // Check for word boundaries (more precise matching)
  const wordBoundaryMatches = goldenAnswers.filter(answer => {
    const escapedAnswer = answer.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const wordBoundaryRegex = new RegExp(`\\b${escapedAnswer}\\b`, 'i')
    const match = wordBoundaryRegex.test(response)
    if (match) debug.word_matches.push(answer)
    return match
  })
  
  let result = ""
  if (exactMatches.length === goldenAnswers.length) {
    result = "✅ CORRECT - All expected answers found"
  } else if (wordBoundaryMatches.length > 0) {
    result = `✅ CORRECT - Found ${wordBoundaryMatches.length}/${goldenAnswers.length} with word boundaries: ${wordBoundaryMatches.join(', ')}`
  } else if (exactMatches.length > 0) {
    result = `⚠️ PARTIAL - Found ${exactMatches.length}/${goldenAnswers.length} exact matches: ${exactMatches.join(', ')}`
  } else if (debug.partial_matches.length > 0) {
    const bestMatch = debug.partial_matches.reduce((best, current) => 
      current.matched_words.length > best.matched_words.length ? current : best
    )
    result = `⚠️ PARTIAL - Best match: ${bestMatch.matched_words.length}/${bestMatch.total_words} words from "${bestMatch.answer}"`
  } else {
    result = "❌ INCORRECT - Expected answers not found"
  }
  
  return { result, debug }
}

// Helper function to compress large prompts for display
function compressPrompt(prompt: string): string {
  if (prompt.length <= 2000) return prompt
  
  const sizeKB = (prompt.length / 1024).toFixed(1)
  const preview = prompt.substring(0, 400)
  const suffix = prompt.substring(prompt.length - 400)
  
  // Try to extract the question for context
  const questionMatch = prompt.match(/Question[:\s]*([^?\n]+\?)/i) || 
                       prompt.match(/Which\s+[^?\n]{10,60}\?/i)
  const question = questionMatch ? questionMatch[0].trim() : 'InfiniteBench task'
  
  return `${preview}\n\n[... COMPRESSED CONTEXT (${sizeKB}KB total) - ${question} ...]\n\n${suffix}`
}

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
              <div className="bg-gray-50 p-3 rounded font-mono text-sm whitespace-pre-wrap">
                {compressPrompt(call.prompt)}
              </div>
            </div>

            {call.completion && (
              <div className="bg-white rounded border p-4 mb-4">
                <h3 className="font-medium mb-2">Response</h3>
                <div className="bg-gray-50 p-3 rounded font-mono text-sm">
                  {call.completion}
                </div>
              </div>
            )}

            {/* Golden Answer Section */}
            <div className="bg-white rounded border p-4">
              <h3 className="font-medium mb-2">Expected Answer (Golden Standard)</h3>
              <div className="bg-yellow-50 border border-yellow-200 p-3 rounded">
                {call.request?.benchmark_metadata?.golden_answer ? (
                  <div className="font-mono text-sm">
                    {Array.isArray(call.request.benchmark_metadata.golden_answer) 
                      ? call.request.benchmark_metadata.golden_answer.join(', ')
                      : call.request.benchmark_metadata.golden_answer
                    }
                  </div>
                ) : (
                  <div className="text-gray-600 text-sm italic">
                    Golden answer not available for this benchmark
                  </div>
                )}
              </div>
              
              {call.request?.benchmark_metadata?.golden_answer && call.completion && (
                <div className="mt-3 space-y-2">
                  <div className="p-2 bg-gray-50 rounded text-xs">
                    <strong>Accuracy Check:</strong> {checkAnswerAccuracy(
                      call.completion, 
                      call.request.benchmark_metadata.golden_answer
                    ).result}
                  </div>
                  <details className="text-xs">
                    <summary className="cursor-pointer text-gray-600 hover:text-gray-800">Debug Information</summary>
                    <div className="mt-2 p-2 bg-gray-100 rounded">
                      <pre className="whitespace-pre-wrap font-mono text-xs">
                        {JSON.stringify(
                          checkAnswerAccuracy(
                            call.completion, 
                            call.request.benchmark_metadata.golden_answer
                          ).debug, 
                          null, 
                          2
                        )}
                      </pre>
                    </div>
                  </details>
                </div>
              )}
            </div>
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
              <div style={{ minHeight: '800px' }}>
                <JsonDiff
                  obj1={call.request?.pre_transform?.payload}
                  obj2={call.request?.post_transform?.payload}
                  title="Payload Diff (Pre → Post Transform)"
                  height={800}
                  disableCompression={true}
                />
              </div>
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