import React, { useState } from 'react'
import { Archive, Calendar, GitBranch, Plus, Trash2, Eye, Download } from 'lucide-react'
import { formatDate } from '../lib/utils'

interface Checkpoint {
  id: string
  label: string
  sessionId: string
  createdAt: number
  messageCount: number
  chunkCount: number
  description?: string
  size: number
}

export function CheckpointsView() {
  // Mock data - in reality this would come from the API
  const [checkpoints] = useState<Checkpoint[]>([
    {
      id: 'cp-001',
      label: 'After implementing auth',
      sessionId: 'auth-discussion',
      createdAt: Date.now() - 86400000, // 1 day ago
      messageCount: 45,
      chunkCount: 128,
      description: 'Snapshot after completing JWT authentication implementation',
      size: 2.4 * 1024 * 1024 // 2.4 MB
    },
    {
      id: 'cp-002',
      label: 'Database design complete',
      sessionId: 'db-planning',
      createdAt: Date.now() - 172800000, // 2 days ago
      messageCount: 32,
      chunkCount: 89,
      description: 'All database schemas and migrations finalized',
      size: 1.8 * 1024 * 1024 // 1.8 MB
    },
    {
      id: 'cp-003',
      label: 'API endpoints v1',
      sessionId: 'api-development',
      createdAt: Date.now() - 259200000, // 3 days ago
      messageCount: 67,
      chunkCount: 156,
      description: 'First version of all REST endpoints implemented',
      size: 3.1 * 1024 * 1024 // 3.1 MB
    }
  ])

  const [selectedCheckpoints, setSelectedCheckpoints] = useState<Set<string>>(new Set())
  const [showCreateForm, setShowCreateForm] = useState(false)

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const toggleCheckpoint = (id: string) => {
    const newSelected = new Set(selectedCheckpoints)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedCheckpoints(newSelected)
  }

  const handleCompare = () => {
    if (selectedCheckpoints.size === 2) {
      const [id1, id2] = Array.from(selectedCheckpoints)
      console.log(`Comparing checkpoints: ${id1} vs ${id2}`)
      // In a real app, this would navigate to a comparison view
    }
  }

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
              <Archive className="w-8 h-8" />
              Checkpoints
            </h1>
            <p className="text-slate-400">
              Capture and compare conversation snapshots
            </p>
          </div>
          
          <button
            onClick={() => setShowCreateForm(true)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            Create Checkpoint
          </button>
        </div>

        {/* Actions Bar */}
        {selectedCheckpoints.size > 0 && (
          <div className="bg-slate-800 rounded-lg p-4 mb-6">
            <div className="flex items-center justify-between">
              <div className="text-sm text-slate-300">
                {selectedCheckpoints.size} checkpoint{selectedCheckpoints.size > 1 ? 's' : ''} selected
              </div>
              
              <div className="flex items-center gap-3">
                {selectedCheckpoints.size === 2 && (
                  <button
                    onClick={handleCompare}
                    className="flex items-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm transition-colors"
                  >
                    <GitBranch className="w-4 h-4" />
                    Compare
                  </button>
                )}
                
                <button
                  onClick={() => setSelectedCheckpoints(new Set())}
                  className="flex items-center gap-2 px-3 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg text-sm transition-colors"
                >
                  Clear Selection
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold text-white mb-4">Create Checkpoint</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Session ID
                </label>
                <input
                  type="text"
                  placeholder="Enter session ID"
                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Label
                </label>
                <input
                  type="text"
                  placeholder="Descriptive label for this checkpoint"
                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Description (optional)
                </label>
                <textarea
                  rows={3}
                  placeholder="Additional details about this checkpoint"
                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                />
              </div>
            </div>
            
            <div className="flex items-center justify-end gap-3 mt-6">
              <button
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  // In a real app, this would call the API
                  console.log('Creating checkpoint...')
                  setShowCreateForm(false)
                }}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Checkpoints List */}
      {checkpoints.length === 0 ? (
        <div className="bg-slate-800 rounded-lg p-12 text-center">
          <Archive className="w-12 h-12 text-slate-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">No Checkpoints</h2>
          <p className="text-slate-400 mb-6">
            Create checkpoints to capture conversation snapshots
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {checkpoints.map((checkpoint) => (
            <CheckpointCard
              key={checkpoint.id}
              checkpoint={checkpoint}
              isSelected={selectedCheckpoints.has(checkpoint.id)}
              onToggleSelect={() => toggleCheckpoint(checkpoint.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function CheckpointCard({ 
  checkpoint, 
  isSelected, 
  onToggleSelect 
}: { 
  checkpoint: Checkpoint
  isSelected: boolean
  onToggleSelect: () => void
}) {
  return (
    <div className={`bg-slate-800 border rounded-lg p-6 transition-all ${
      isSelected ? 'border-blue-500 ring-1 ring-blue-500' : 'border-slate-700 hover:border-slate-600'
    }`}>
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-4">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={onToggleSelect}
            className="mt-1 w-4 h-4 text-blue-600 bg-slate-700 border-slate-600 rounded focus:ring-blue-500 focus:ring-2"
          />
          
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-lg font-semibold text-white">{checkpoint.label}</h3>
              <span className="text-slate-400 text-sm">•</span>
              <span className="text-blue-400 text-sm font-medium">{checkpoint.sessionId}</span>
            </div>
            
            {checkpoint.description && (
              <p className="text-slate-300 text-sm mb-3">{checkpoint.description}</p>
            )}
            
            <div className="flex items-center gap-6 text-sm text-slate-400">
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                {formatDate(checkpoint.createdAt)}
              </div>
              <div>
                {checkpoint.messageCount} messages
              </div>
              <div>
                {checkpoint.chunkCount} chunks
              </div>
              <div>
                {formatBytes(checkpoint.size)}
              </div>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
            title="View Details"
          >
            <Eye className="w-4 h-4" />
          </button>
          
          <button
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
            title="Export"
          >
            <Download className="w-4 h-4" />
          </button>
          
          <button
            className="p-2 text-red-400 hover:text-red-300 hover:bg-red-900/20 rounded-lg transition-colors"
            title="Delete"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}