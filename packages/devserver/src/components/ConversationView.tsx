import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, User, Bot, FileText, Code, Terminal, Zap, Package } from 'lucide-react'
import { api, type Message, type Chunk } from '../lib/api'
import { formatDate, truncateText } from '../lib/utils'

export function ConversationView() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const [messages, setMessages] = useState<Message[]>([])
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showHeatMap, setShowHeatMap] = useState(false)

  useEffect(() => {
    if (sessionId) {
      loadSessionData()
    }
  }, [sessionId])

  const loadSessionData = async () => {
    if (!sessionId) return

    try {
      setLoading(true)
      const [messagesData, chunksData] = await Promise.all([
        api.getMessages(sessionId),
        api.getChunks(sessionId),
      ])
      
      setMessages(messagesData)
      setChunks(chunksData)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load session')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="animate-pulse">
          <div className="h-8 bg-slate-700 rounded w-1/4 mb-6"></div>
          <div className="space-y-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-slate-800 p-6 rounded-lg">
                <div className="h-4 bg-slate-700 rounded w-1/6 mb-4"></div>
                <div className="space-y-2">
                  <div className="h-4 bg-slate-700 rounded w-full"></div>
                  <div className="h-4 bg-slate-700 rounded w-3/4"></div>
                </div>
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
          <h2 className="text-red-400 font-semibold mb-2">Error Loading Session</h2>
          <p className="text-red-300">{error}</p>
        </div>
      </div>
    )
  }

  // Group chunks by message
  const chunksByMessage = chunks.reduce((acc, chunk) => {
    if (!acc[chunk.messageId]) {
      acc[chunk.messageId] = []
    }
    acc[chunk.messageId].push(chunk)
    return acc
  }, {} as Record<string, Chunk[]>)

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <Link
            to="/"
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-white">{sessionId}</h1>
            <p className="text-slate-400">
              {messages.length} messages • {chunks.length} chunks
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowHeatMap(!showHeatMap)}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              showHeatMap
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            <Zap className="w-4 h-4 inline mr-2" />
            DF/IDF Heat Map
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="space-y-6">
        {messages.map((message) => {
          const messageChunks = chunksByMessage[message.id] || []
          return (
            <MessageCard
              key={message.id}
              message={message}
              chunks={messageChunks}
              showHeatMap={showHeatMap}
            />
          )
        })}
      </div>
    </div>
  )
}

function MessageCard({ 
  message, 
  chunks, 
  showHeatMap 
}: { 
  message: Message
  chunks: Chunk[]
  showHeatMap: boolean 
}) {
  const isUser = message.role === 'user'
  
  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-4xl ${isUser ? 'order-2' : 'order-1'}`}>
        <div className={`rounded-lg p-6 ${
          isUser 
            ? 'bg-blue-900/20 border border-blue-800' 
            : 'bg-slate-800 border border-slate-700'
        }`}>
          {/* Header */}
          <div className="flex items-center gap-3 mb-4">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              isUser ? 'bg-blue-600' : 'bg-slate-600'
            }`}>
              {isUser ? (
                <User className="w-4 h-4 text-white" />
              ) : (
                <Bot className="w-4 h-4 text-white" />
              )}
            </div>
            <div>
              <p className="font-medium text-white">
                {isUser ? 'User' : 'Assistant'}
              </p>
              <p className="text-sm text-slate-400">
                {formatDate(message.timestamp)}
              </p>
            </div>
          </div>

          {/* Content */}
          <div className="prose prose-invert prose-slate max-w-none">
            <div className="text-slate-200 whitespace-pre-wrap">
              {message.content}
            </div>
          </div>

          {/* Chunks */}
          {chunks.length > 0 && (
            <div className="mt-6 pt-4 border-t border-slate-600">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="w-4 h-4 text-slate-400" />
                <span className="text-sm font-medium text-slate-300">
                  {chunks.length} chunks
                </span>
              </div>
              <div className="space-y-2">
                {chunks.map((chunk) => (
                  <ChunkPreview 
                    key={chunk.id} 
                    chunk={chunk} 
                    showHeatMap={showHeatMap}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
        isUser ? 'bg-blue-600 order-1' : 'bg-slate-600 order-2'
      }`}>
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </div>
    </div>
  )
}

function ChunkPreview({ 
  chunk, 
  showHeatMap 
}: { 
  chunk: Chunk
  showHeatMap: boolean 
}) {
  const kindIcons = {
    prose: FileText,
    code: Code,
    tool_result: Terminal,
    user_code: Code,
  }
  
  const kindColors = {
    prose: 'text-blue-400',
    code: 'text-green-400',
    tool_result: 'text-purple-400',
    user_code: 'text-orange-400',
  }

  const Icon = kindIcons[chunk.kind]
  
  return (
    <div className="bg-slate-700 rounded-lg p-3 text-sm">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${kindColors[chunk.kind]}`} />
        <span className="text-slate-300 font-medium">{chunk.kind}</span>
        <span className="text-slate-400">•</span>
        <span className="text-slate-400">{chunk.tokens} tokens</span>
        {chunk.hasEmbedding && (
          <>
            <span className="text-slate-400">•</span>
            <Zap className="w-3 h-3 text-yellow-400" />
            <span className="text-yellow-400 text-xs">embedded</span>
          </>
        )}
      </div>
      <div className={`text-slate-200 font-mono text-xs ${
        showHeatMap ? 'heatmap-active' : ''
      }`}>
        {truncateText(chunk.text, 200)}
      </div>
    </div>
  )
}