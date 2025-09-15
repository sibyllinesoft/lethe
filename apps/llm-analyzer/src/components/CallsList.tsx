import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Virtuoso } from 'react-virtuoso'
import { format } from 'date-fns'
import { useCalls, useStats } from '../hooks/api'
import { useUIStore } from '../store/ui'
import { CallPair, CallsFilters } from '../types'
import clsx from 'clsx'

interface CallsListProps {
  onCallSelect?: (call: CallPair) => void
  selectedCallId?: string
}

export default function CallsList({ onCallSelect, selectedCallId }: CallsListProps) {
  const { selectedCallIds, setSelectedCallIds, compareMode } = useUIStore()
  const [filters, setFilters] = useState<CallsFilters>({ page: 1, limit: 100 })
  
  const { data: callsData, isLoading, error } = useCalls(filters)
  const { data: stats } = useStats()

  const calls = callsData?.calls || []

  const handleCallClick = (call: CallPair) => {
    if (compareMode) {
      if (selectedCallIds.includes(call.id)) {
        // Remove from selection
        setSelectedCallIds(selectedCallIds.filter(id => id !== call.id))
      } else if (selectedCallIds.length < 2) {
        // Add to selection
        setSelectedCallIds([...selectedCallIds, call.id])
      }
    } else {
      onCallSelect?.(call)
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return format(new Date(timestamp), 'HH:mm:ss.SSS')
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'pill-success'
      case 'error': return 'pill-error'
      case 'pending': return 'pill-pending'
      default: return ''
    }
  }

  const renderCallItem = (index: number, call: CallPair) => {
    const isSelected = selectedCallIds.includes(call.id) || call.id === selectedCallId
    
    return (
      <div
        key={call.id}
        className={clsx(
          'p-3 border-b border-gray-200 cursor-pointer hover:bg-gray-50',
          isSelected && 'bg-blue-50 border-blue-200'
        )}
        onClick={() => handleCallClick(call)}
      >
        <div className="flex justify-between items-start mb-2">
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-gray-900 truncate">
              {call.query_id}
            </div>
            <div className="text-xs text-gray-500">
              {formatTimestamp(call.timestamp)}
            </div>
          </div>
          <div className="flex-shrink-0">
            <span className={clsx('pill', getStatusColor(call.status))}>
              {call.status}
            </span>
          </div>
        </div>
        
        <div className="flex gap-2 mb-2 flex-wrap">
          <span className="pill pill-provider">{call.provider}</span>
          <span className="pill pill-model">{call.model}</span>
        </div>
        
        <div className="flex justify-between text-xs text-gray-500">
          <span>{call.latency_ms}ms</span>
          <span>{call.input_tokens + call.output_tokens} tokens</span>
        </div>
        
        {call.transform_changes.length > 0 && (
          <div className="mt-2 text-xs">
            <span className="text-gray-500">Transforms: </span>
            {call.transform_changes.slice(0, 2).map((change, i) => (
              <span key={i} className="pill" style={{ fontSize: '10px' }}>
                {change}
              </span>
            ))}
            {call.transform_changes.length > 2 && (
              <span className="text-gray-500">+{call.transform_changes.length - 2} more</span>
            )}
          </div>
        )}
        
        {!compareMode && (
          <Link 
            to={`/call/${call.id}`}
            className="text-xs text-blue-600 hover:text-blue-800 mt-1 inline-block"
            onClick={(e) => e.stopPropagation()}
          >
            View Details →
          </Link>
        )}
      </div>
    )
  }

  if (isLoading) {
    return <div className="loading">Loading calls...</div>
  }

  if (error) {
    return <div className="error">Error loading calls: {error.message}</div>
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-lg font-semibold">LLM Calls</h2>
          {compareMode && (
            <div className="text-sm text-gray-600">
              {selectedCallIds.length}/2 selected
            </div>
          )}
        </div>
        
        {stats && (
          <div className="text-sm text-gray-600">
            {stats.total_calls} total calls • {stats.providers.length} providers
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="grid grid-cols-2 gap-2 text-xs">
          <select
            value={filters.provider || ''}
            onChange={(e) => setFilters({ ...filters, provider: e.target.value || undefined })}
            className="px-2 py-1 border border-gray-300 rounded"
          >
            <option value="">All Providers</option>
            {stats?.providers.map((provider: string) => (
              <option key={provider} value={provider}>{provider}</option>
            ))}
          </select>
          
          <select
            value={filters.status || ''}
            onChange={(e) => setFilters({ ...filters, status: e.target.value || undefined })}
            className="px-2 py-1 border border-gray-300 rounded"
          >
            <option value="">All Statuses</option>
            <option value="success">Success</option>
            <option value="error">Error</option>
            <option value="pending">Pending</option>
          </select>
        </div>
      </div>

      {/* Compare Mode Actions */}
      {compareMode && selectedCallIds.length === 2 && (
        <div className="p-3 bg-blue-50 border-b border-blue-200">
          <Link
            to={`/compare?a=${selectedCallIds[0]}&b=${selectedCallIds[1]}`}
            className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700"
          >
            Compare Selected Calls
          </Link>
        </div>
      )}

      {/* Calls List */}
      <div className="flex-1 overflow-hidden">
        {calls.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            No calls found
          </div>
        ) : (
          <div className="overflow-y-auto">
            {calls.map((call, index) => renderCallItem(index, call))}
          </div>
        )}
      </div>
    </div>
  )
}