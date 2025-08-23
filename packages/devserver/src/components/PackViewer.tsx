import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Package, Clock, Users, FileText, AlertTriangle } from 'lucide-react'
import { api, type ContextPack } from '../lib/api'
import { formatTime } from '../lib/utils'

export function PackViewer() {
  const { sessionId, packId } = useParams<{ sessionId: string; packId: string }>()
  const [pack, setPack] = useState<ContextPack | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (sessionId && packId) {
      loadPack()
    }
  }, [sessionId, packId])

  const loadPack = async () => {
    // For now, we'll mock this since the API doesn't have a specific pack endpoint
    // In a real implementation, you'd add GET /api/sessions/:sessionId/packs/:packId
    try {
      setLoading(true)
      // Mock pack data - in reality this would come from the API
      const mockPack: ContextPack = {
        id: packId!,
        sessionId: sessionId!,
        query: "Sample query",
        summary: "This is a sample context pack for demonstration purposes.",
        keyEntities: ["Entity1", "Entity2", "Entity3"],
        claims: [
          { text: "This is a sample claim extracted from the context.", chunks: ["chunk1", "chunk2"] },
          { text: "Another important claim found in the conversation.", chunks: ["chunk3"] }
        ],
        contradictions: [
          { issue: "Conflicting information about implementation approach", chunks: ["chunk2", "chunk4"] }
        ],
        citations: {
          "ref1": { messageId: "msg1", span: [100, 200] },
          "ref2": { messageId: "msg2", span: [50, 150] }
        },
        debug: {
          hydeQueries: ["query1", "query2", "query3"],
          candidateCount: 25,
          rerankTime: 150,
          totalTime: 850
        }
      }
      setPack(mockPack)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load pack')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="animate-pulse">
          <div className="h-8 bg-slate-700 rounded w-1/4 mb-6"></div>
          <div className="space-y-4">
            <div className="bg-slate-800 p-6 rounded-lg">
              <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
              <div className="h-4 bg-slate-700 rounded w-full mb-2"></div>
              <div className="h-4 bg-slate-700 rounded w-2/3"></div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error || !pack) {
    return (
      <div className="p-8">
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-6">
          <h2 className="text-red-400 font-semibold mb-2">Error Loading Pack</h2>
          <p className="text-red-300">{error || 'Pack not found'}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <Link
            to={`/sessions/${sessionId}`}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Package className="w-8 h-8" />
              Context Pack
            </h1>
            <p className="text-slate-400">
              {sessionId} • {pack.id}
            </p>
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-white mb-2">Query</h2>
          <p className="text-slate-200">{pack.query}</p>
        </div>
      </div>

      <div className="space-y-8">
        {/* Debug Information */}
        {pack.debug && (
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Execution Metrics
            </h2>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">{pack.debug.hydeQueries.length}</div>
                <div className="text-sm text-slate-400">HyDE Queries</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">{pack.debug.candidateCount}</div>
                <div className="text-sm text-slate-400">Candidates</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">{formatTime(pack.debug.rerankTime)}</div>
                <div className="text-sm text-slate-400">Rerank Time</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-400">{formatTime(pack.debug.totalTime)}</div>
                <div className="text-sm text-slate-400">Total Time</div>
              </div>
            </div>

            <div>
              <h3 className="text-md font-medium text-white mb-2">Generated HyDE Queries</h3>
              <div className="space-y-2">
                {pack.debug.hydeQueries.map((query, i) => (
                  <div key={i} className="bg-slate-700 rounded-lg p-3 text-sm text-slate-300">
                    <span className="text-slate-500 mr-2">{i + 1}.</span>
                    {query}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Summary */}
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Summary
          </h2>
          <div className="bg-slate-700 rounded-lg p-4">
            <p className="text-slate-200 leading-relaxed">{pack.summary}</p>
          </div>
        </div>

        {/* Key Entities */}
        {pack.keyEntities.length > 0 && (
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Users className="w-5 h-5" />
              Key Entities
            </h2>
            <div className="flex flex-wrap gap-3">
              {pack.keyEntities.map((entity, i) => (
                <span 
                  key={i} 
                  className="bg-blue-900/30 text-blue-300 border border-blue-800 px-4 py-2 rounded-lg text-sm font-medium"
                >
                  {entity}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Claims */}
        {pack.claims.length > 0 && (
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Claims</h2>
            <div className="space-y-4">
              {pack.claims.map((claim, i) => (
                <div key={i} className="bg-green-900/10 border border-green-800/30 rounded-lg p-4">
                  <p className="text-green-100 mb-3 leading-relaxed">{claim.text}</p>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-green-400">Evidence:</span>
                    <div className="flex gap-2">
                      {claim.chunks.map((chunkId, j) => (
                        <span 
                          key={j}
                          className="bg-green-800/30 text-green-300 px-2 py-1 rounded text-xs font-mono"
                        >
                          {chunkId}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Contradictions */}
        {pack.contradictions.length > 0 && (
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Contradictions
            </h2>
            <div className="space-y-4">
              {pack.contradictions.map((contradiction, i) => (
                <div key={i} className="bg-red-900/10 border border-red-800/30 rounded-lg p-4">
                  <p className="text-red-100 mb-3 leading-relaxed">{contradiction.issue}</p>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-red-400">Conflicting sources:</span>
                    <div className="flex gap-2">
                      {contradiction.chunks.map((chunkId, j) => (
                        <span 
                          key={j}
                          className="bg-red-800/30 text-red-300 px-2 py-1 rounded text-xs font-mono"
                        >
                          {chunkId}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Citations */}
        {Object.keys(pack.citations).length > 0 && (
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Citations</h2>
            <div className="grid gap-4 md:grid-cols-2">
              {Object.entries(pack.citations).map(([citationId, citation]) => (
                <div key={citationId} className="bg-slate-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-slate-300">
                      Citation {citationId}
                    </span>
                    <Link 
                      to={`/sessions/${sessionId}`}
                      className="text-blue-400 hover:text-blue-300 text-sm transition-colors"
                    >
                      View Message
                    </Link>
                  </div>
                  <div className="space-y-1 text-sm">
                    <div>
                      <span className="text-slate-400">Message ID:</span>
                      <span className="text-slate-200 font-mono ml-2">{citation.messageId}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">Text Span:</span>
                      <span className="text-slate-200 font-mono ml-2">
                        {citation.span[0]} - {citation.span[1]}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}