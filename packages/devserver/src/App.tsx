import React, { useState, useEffect } from 'react'
import { Routes, Route, HashRouter } from 'react-router-dom'
import { SessionsView } from './components/SessionsView'
import { ConversationView } from './components/ConversationView'
import { PackViewer } from './components/PackViewer'
import { ConfigEditor } from './components/ConfigEditor'
import { CheckpointsView } from './components/CheckpointsView'
import { Sidebar } from './components/Sidebar'
import { QueryInterface } from './components/QueryInterface'
import { RetrievalDebugger } from './components/RetrievalDebugger'

export function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Test API connection
    fetch('/api/health')
      .then(res => res.json())
      .then(() => setIsConnected(true))
      .catch(err => {
        setError(`Failed to connect to API: ${err.message}`)
        setIsConnected(false)
      })
  }, [])

  if (error) {
    return (
      <div className="min-h-screen bg-slate-900 text-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-400 mb-4">Connection Error</h1>
          <p className="text-slate-300 mb-4">{error}</p>
          <p className="text-sm text-slate-500">
            Make sure the ctx-run server is running: <code>npx ctx-run serve</code>
          </p>
        </div>
      </div>
    )
  }

  if (!isConnected) {
    return (
      <div className="min-h-screen bg-slate-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <p className="text-slate-300">Connecting to ctx-run server...</p>
        </div>
      </div>
    )
  }

  return (
    <HashRouter>
      <div className="min-h-screen bg-slate-900 text-white">
        <div className="flex">
          <Sidebar />
          
          <main className="flex-1 overflow-auto">
            <Routes>
              <Route path="/" element={<SessionsView />} />
              <Route path="/sessions/:sessionId" element={<ConversationView />} />
              <Route path="/sessions/:sessionId/pack/:packId" element={<PackViewer />} />
              <Route path="/query" element={<QueryInterface />} />
              <Route path="/debug" element={<RetrievalDebugger />} />
              <Route path="/config" element={<ConfigEditor />} />
              <Route path="/checkpoints" element={<CheckpointsView />} />
            </Routes>
          </main>
        </div>
      </div>
    </HashRouter>
  )
}