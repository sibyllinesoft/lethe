import React, { useState, useEffect } from 'react'
import { Search, Send, Loader, Clock, Zap, Package } from 'lucide-react'
import { api, type Session, type QueryResult } from '../lib/api'
import { formatTime } from '../lib/utils'

export function QueryInterface() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [selectedSession, setSelectedSession] = useState('')
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<QueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadSessions()
  }, [])

  const loadSessions = async () => {
    try {
      const data = await api.getSessions()
      setSessions(data)
      if (data.length > 0 && !selectedSession) {
        setSelectedSession(data[0].id)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions')
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || !selectedSession) return

    try {
      setLoading(true)
      setError(null)
      const data = await api.query(query.trim(), selectedSession)
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Query Interface</h1>
        <p className="text-slate-400">
          Search conversation history with natural language queries
        </p>
      </div>

      {/* Query Form */}
      <div className="bg-slate-800 rounded-lg p-6 mb-8">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Session
            </label>
            <select
              value={selectedSession}
              onChange={(e) => setSelectedSession(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {sessions.map((session) => (
                <option key={session.id} value={session.id}>
                  {session.id} ({session.messageCount} messages)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Query
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-3 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question about the conversation..."
                className="w-full bg-slate-700 border border-slate-600 rounded-lg pl-10 pr-12 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                disabled={loading || !query.trim() || !selectedSession}
                className="absolute right-2 top-2 p-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed rounded-lg transition-colors"
              >
                {loading ? (
                  <Loader className="w-4 h-4 text-white animate-spin" />
                ) : (
                  <Send className="w-4 h-4 text-white" />
                )}
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-6 mb-8">
          <p className="text-red-400 font-semibold">Error</p>
          <p className="text-red-300">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Debug Info */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Query Execution</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-slate-400">Plan Type</p>
                <p className="text-white font-medium">{result.debug.planType}</p>
              </div>
              <div>
                <p className="text-slate-400">HyDE Queries</p>
                <p className="text-white font-medium">{result.debug.hydeQueries.length}</p>
              </div>
              <div>
                <p className="text-slate-400">Candidates</p>
                <p className="text-white font-medium">{result.debug.candidateCount}</p>
              </div>
              <div>
                <p className="text-slate-400">Total Time</p>
                <p className="text-white font-medium">{formatTime(result.debug.totalTime)}</p>
              </div>
            </div>

            {result.debug.hydeQueries.length > 0 && (
              <div className="mt-4">
                <p className="text-slate-400 text-sm mb-2">Generated HyDE Queries:</p>
                <div className="space-y-1">
                  {result.debug.hydeQueries.map((q, i) => (
                    <div key={i} className="bg-slate-700 rounded px-3 py-2 text-sm text-slate-300">
                      {q}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Context Pack */}
          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Package className="w-5 h-5" />
              Context Pack
            </h2>
            
            {/* Summary */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-slate-300 mb-2">Summary</h3>
              <div className="bg-slate-700 rounded-lg p-4">
                <p className="text-slate-200">{result.pack.summary}</p>
              </div>
            </div>

            {/* Key Entities */}
            {result.pack.keyEntities.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-slate-300 mb-2">Key Entities</h3>
                <div className="flex flex-wrap gap-2">
                  {result.pack.keyEntities.map((entity, i) => (
                    <span key={i} className="bg-blue-900/30 text-blue-300 px-3 py-1 rounded-full text-sm">
                      {entity}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Claims */}
            {result.pack.claims.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-slate-300 mb-2">Claims</h3>
                <div className="space-y-3">
                  {result.pack.claims.map((claim, i) => (
                    <div key={i} className="bg-green-900/20 border border-green-800 rounded-lg p-4">
                      <p className="text-green-100 mb-2">{claim.text}</p>
                      <p className="text-green-400 text-sm">
                        {claim.chunks.length} supporting chunk(s)
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Contradictions */}
            {result.pack.contradictions.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-slate-300 mb-2">Contradictions</h3>
                <div className="space-y-3">
                  {result.pack.contradictions.map((contradiction, i) => (
                    <div key={i} className="bg-red-900/20 border border-red-800 rounded-lg p-4">
                      <p className="text-red-100 mb-2">{contradiction.issue}</p>
                      <p className="text-red-400 text-sm">
                        {contradiction.chunks.length} conflicting chunk(s)
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Citations */}
            {Object.keys(result.pack.citations).length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-slate-300 mb-2">Citations</h3>
                <div className="space-y-2">
                  {Object.entries(result.pack.citations).map(([id, citation]) => (
                    <div key={id} className="bg-slate-700 rounded-lg p-3 text-sm">
                      <p className="text-slate-200">
                        <span className="text-slate-400">Message:</span> {citation.messageId}
                      </p>
                      <p className="text-slate-200">
                        <span className="text-slate-400">Span:</span> {citation.span[0]}-{citation.span[1]}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}