import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { MessageSquare, FileText, Package, Zap, ExternalLink } from 'lucide-react'
import { api, type Session } from '../lib/api'
import { formatDate, formatTime } from '../lib/utils'

export function SessionsView() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadSessions()
  }, [])

  const loadSessions = async () => {
    try {
      setLoading(true)
      const data = await api.getSessions()
      setSessions(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions')
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
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-slate-800 p-6 rounded-lg">
                <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
                <div className="h-4 bg-slate-700 rounded w-full mb-2"></div>
                <div className="h-4 bg-slate-700 rounded w-2/3"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-6">
          <h2 className="text-red-400 font-semibold mb-2">Error Loading Sessions</h2>
          <p className="text-red-300">{error}</p>
          <button
            onClick={loadSessions}
            className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white text-sm transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Sessions</h1>
        <p className="text-slate-400">
          Browse and analyze conversation history from your ctx-run workspace
        </p>
      </div>

      {sessions.length === 0 ? (
        <div className="bg-slate-800 rounded-lg p-12 text-center">
          <MessageSquare className="w-12 h-12 text-slate-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">No Sessions Found</h2>
          <p className="text-slate-400 mb-6">
            Import some conversation data to get started
          </p>
          <div className="bg-slate-700 rounded-lg p-4 text-left max-w-md mx-auto">
            <p className="text-sm font-medium text-white mb-2">Quick Start:</p>
            <code className="text-sm text-green-400 block mb-1">
              npx ctx-run ingest -s my-session --from chat.json
            </code>
            <code className="text-sm text-green-400 block mb-1">
              npx ctx-run index -s my-session
            </code>
            <code className="text-sm text-green-400 block">
              npx ctx-run serve --open
            </code>
          </div>
        </div>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {sessions.map((session) => (
            <SessionCard key={session.id} session={session} />
          ))}
        </div>
      )}
    </div>
  )
}

function SessionCard({ session }: { session: Session }) {
  const coveragePercentage = session.chunkCount > 0 
    ? (session.embeddingCount / session.chunkCount * 100).toFixed(1)
    : '0.0'
  
  const isFullyIndexed = session.embeddingCount === session.chunkCount && session.chunkCount > 0

  return (
    <Link
      to={`/sessions/${session.id}`}
      className="bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-slate-600 rounded-lg p-6 transition-all duration-200 group"
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white group-hover:text-blue-400 mb-1 transition-colors">
            {session.id}
          </h3>
          <p className="text-sm text-slate-400">
            {formatDate(session.updatedAt)}
          </p>
        </div>
        <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-slate-400" />
      </div>

      <div className="space-y-3">
        <div className="flex items-center gap-3 text-sm">
          <div className="flex items-center gap-2 text-blue-400">
            <MessageSquare className="w-4 h-4" />
            <span>{session.messageCount} messages</span>
          </div>
          <div className="flex items-center gap-2 text-green-400">
            <FileText className="w-4 h-4" />
            <span>{session.chunkCount} chunks</span>
          </div>
        </div>

        <div className="flex items-center gap-3 text-sm">
          <div className="flex items-center gap-2 text-purple-400">
            <Zap className="w-4 h-4" />
            <span>{session.embeddingCount} embeddings</span>
          </div>
          <div className="flex items-center gap-2 text-orange-400">
            <Package className="w-4 h-4" />
            <span>{session.packCount} packs</span>
          </div>
        </div>

        <div className="pt-2">
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="text-slate-400">Index Coverage</span>
            <span className={`font-medium ${isFullyIndexed ? 'text-green-400' : 'text-yellow-400'}`}>
              {coveragePercentage}%
            </span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${
                isFullyIndexed ? 'bg-green-500' : 'bg-yellow-500'
              }`}
              style={{ width: `${coveragePercentage}%` }}
            />
          </div>
        </div>
      </div>
    </Link>
  )
}