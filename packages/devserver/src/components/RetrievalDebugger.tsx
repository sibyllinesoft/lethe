import React, { useState } from 'react'
import { Search, Play, Clock, Layers, Filter, BarChart3 } from 'lucide-react'

export function RetrievalDebugger() {
  const [query, setQuery] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [debugging, setDebugging] = useState(false)
  const [debugData, setDebugData] = useState<any>(null)

  const handleDebug = async () => {
    if (!query.trim() || !sessionId.trim()) return
    
    setDebugging(true)
    
    // Simulate debugging delay
    setTimeout(() => {
      // Mock debug data
      setDebugData({
        pipeline: [
          {
            stage: 'Query Planning',
            duration: 45,
            details: {
              planType: 'explore',
              confidence: 0.85
            }
          },
          {
            stage: 'HyDE Generation',
            duration: 320,
            details: {
              queriesGenerated: 3,
              model: 'xgen-small:4b'
            }
          },
          {
            stage: 'Vector Search',
            duration: 128,
            details: {
              candidates: 150,
              embeddingModel: 'all-MiniLM-L6-v2'
            }
          },
          {
            stage: 'BM25 Search',
            duration: 65,
            details: {
              candidates: 89,
              alpha: 0.7
            }
          },
          {
            stage: 'Hybrid Fusion',
            duration: 23,
            details: {
              finalCandidates: 75,
              fusionAlpha: 0.7,
              fusionBeta: 0.8
            }
          },
          {
            stage: 'Reranking',
            duration: 445,
            details: {
              model: 'ms-marco-MiniLM-L-12-v2',
              topK: 15
            }
          },
          {
            stage: 'Pack Generation',
            duration: 890,
            details: {
              chunks: 12,
              summarizationModel: 'xgen-small:4b'
            }
          }
        ],
        hydeQueries: [
          'How to implement authentication in a web application',
          'Best practices for user authentication and session management',
          'JWT vs session-based authentication comparison'
        ],
        candidates: [
          {
            id: 'chunk-1',
            text: 'To implement JWT authentication, you need to create a token service...',
            bm25Score: 8.4,
            vectorScore: 0.89,
            hybridScore: 0.92,
            rerankScore: 0.95,
            messageId: 'msg-123',
            kind: 'prose'
          },
          {
            id: 'chunk-2', 
            text: 'const authMiddleware = (req, res, next) => { const token = req.headers.authorization...',
            bm25Score: 7.1,
            vectorScore: 0.82,
            hybridScore: 0.87,
            rerankScore: 0.91,
            messageId: 'msg-124',
            kind: 'code'
          },
          // More candidates...
        ]
      })
      setDebugging(false)
    }, 1500)
  }

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
          <BarChart3 className="w-8 h-8" />
          Retrieval Debugger
        </h1>
        <p className="text-slate-400">
          Step-by-step visualization of the retrieval pipeline
        </p>
      </div>

      {/* Debug Form */}
      <div className="bg-slate-800 rounded-lg p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Session ID
            </label>
            <input
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="Enter session ID"
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Debug Query
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-3 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter query to debug..."
                className="w-full bg-slate-700 border border-slate-600 rounded-lg pl-10 pr-12 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={handleDebug}
                disabled={debugging || !query.trim() || !sessionId.trim()}
                className="absolute right-2 top-2 p-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed rounded-lg transition-colors"
              >
                {debugging ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                ) : (
                  <Play className="w-4 h-4 text-white" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {debugData && (
        <div className="space-y-8">
          {/* Pipeline Visualization */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Pipeline Stages
            </h2>
            
            <div className="space-y-4">
              {debugData.pipeline.map((stage: any, i: number) => (
                <div key={i} className="relative">
                  <div className="flex items-center justify-between bg-slate-700 rounded-lg p-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-semibold text-white">
                        {i + 1}
                      </div>
                      <div>
                        <h3 className="text-white font-medium">{stage.stage}</h3>
                        <div className="flex items-center gap-4 mt-1 text-sm text-slate-400">
                          <div className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {stage.duration}ms
                          </div>
                          {Object.entries(stage.details).map(([key, value]: [string, any]) => (
                            <div key={key}>
                              <span className="text-slate-500">{key}:</span>
                              <span className="text-slate-300 ml-1">{value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="text-2xl font-bold text-blue-400">{stage.duration}ms</div>
                    </div>
                  </div>
                  
                  {i < debugData.pipeline.length - 1 && (
                    <div className="absolute left-4 top-16 w-0.5 h-4 bg-slate-600"></div>
                  )}
                </div>
              ))}
            </div>

            <div className="mt-6 bg-slate-900 rounded-lg p-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400 mb-1">
                  {debugData.pipeline.reduce((sum: number, stage: any) => sum + stage.duration, 0)}ms
                </div>
                <div className="text-slate-400">Total Pipeline Duration</div>
              </div>
            </div>
          </div>

          {/* HyDE Queries */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Generated HyDE Queries</h2>
            <div className="space-y-3">
              {debugData.hydeQueries.map((hyde: string, i: number) => (
                <div key={i} className="bg-slate-700 rounded-lg p-3">
                  <div className="flex items-start gap-3">
                    <span className="text-slate-500 font-mono text-sm mt-0.5">{i + 1}.</span>
                    <p className="text-slate-200 text-sm">{hyde}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Candidates */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Filter className="w-5 h-5" />
              Top Candidates
            </h2>
            
            <div className="space-y-4">
              {debugData.candidates.map((candidate: any, i: number) => (
                <div key={i} className="bg-slate-700 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-white font-medium">#{i + 1}</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          candidate.kind === 'code' ? 'bg-green-900/30 text-green-300' :
                          candidate.kind === 'prose' ? 'bg-blue-900/30 text-blue-300' :
                          'bg-purple-900/30 text-purple-300'
                        }`}>
                          {candidate.kind}
                        </span>
                        <span className="text-slate-400 text-sm font-mono">{candidate.id}</span>
                      </div>
                      <p className="text-slate-200 text-sm font-mono">
                        {candidate.text.substring(0, 120)}...
                      </p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div className="text-center">
                      <div className="text-lg font-bold text-orange-400">{candidate.bm25Score}</div>
                      <div className="text-slate-400">BM25</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-blue-400">{candidate.vectorScore}</div>
                      <div className="text-slate-400">Vector</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-purple-400">{candidate.hybridScore}</div>
                      <div className="text-slate-400">Hybrid</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-green-400">{candidate.rerankScore}</div>
                      <div className="text-slate-400">Rerank</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}