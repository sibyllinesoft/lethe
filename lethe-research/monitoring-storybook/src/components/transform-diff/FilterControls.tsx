import React, { useState } from 'react';
import { FilterOptions } from '../../types/transform';
import { Filter, X, Calendar, Server, Hash, List } from 'lucide-react';
import clsx from 'clsx';

interface FilterControlsProps {
  filters: FilterOptions;
  onFilterChange: (filters: FilterOptions) => void;
  availableProviders: string[];
  availableRequestIds: string[];
  availableChangeTypes: string[];
  className?: string;
}

/**
 * FilterControls - Interactive filtering interface for transform data
 * 
 * Features:
 * - Multi-select provider filtering
 * - Request ID filtering with search
 * - Time window selection with presets
 * - Change type filtering with categories
 * - Severity level filtering
 * - Clear filters functionality
 * - Real-time filter application
 */
export const FilterControls: React.FC<FilterControlsProps> = ({
  filters,
  onFilterChange,
  availableProviders,
  availableRequestIds,
  availableChangeTypes,
  className
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [timePreset, setTimePreset] = useState<'1h' | '4h' | '24h' | '7d' | 'custom'>('24h');

  // Handle provider filter changes
  const handleProviderChange = (provider: string, checked: boolean) => {
    const newProviders = checked
      ? [...(filters.provider || []), provider]
      : (filters.provider || []).filter(p => p !== provider);
    
    onFilterChange({
      ...filters,
      provider: newProviders.length > 0 ? newProviders : undefined
    });
  };

  // Handle request ID filter changes
  const handleRequestIdChange = (requestId: string, checked: boolean) => {
    const newRequestIds = checked
      ? [...(filters.requestId || []), requestId]
      : (filters.requestId || []).filter(id => id !== requestId);
    
    onFilterChange({
      ...filters,
      requestId: newRequestIds.length > 0 ? newRequestIds : undefined
    });
  };

  // Handle change type filter changes
  const handleChangeTypeChange = (changeType: string, checked: boolean) => {
    const newChangeTypes = checked
      ? [...(filters.changeTypes || []), changeType]
      : (filters.changeTypes || []).filter(type => type !== changeType);
    
    onFilterChange({
      ...filters,
      changeTypes: newChangeTypes.length > 0 ? newChangeTypes : undefined
    });
  };

  // Handle severity level filter changes
  const handleSeverityChange = (severity: 'low' | 'medium' | 'high' | 'critical', checked: boolean) => {
    const newSeverityLevels = checked
      ? [...(filters.severityLevel || []), severity]
      : (filters.severityLevel || []).filter(s => s !== severity);
    
    onFilterChange({
      ...filters,
      severityLevel: newSeverityLevels.length > 0 ? newSeverityLevels : undefined
    });
  };

  // Handle time preset changes
  const handleTimePresetChange = (preset: '1h' | '4h' | '24h' | '7d' | 'custom') => {
    setTimePreset(preset);
    
    if (preset === 'custom') return; // Let user set custom range
    
    const now = new Date();
    const start = new Date();
    
    switch (preset) {
      case '1h':
        start.setHours(now.getHours() - 1);
        break;
      case '4h':
        start.setHours(now.getHours() - 4);
        break;
      case '24h':
        start.setDate(now.getDate() - 1);
        break;
      case '7d':
        start.setDate(now.getDate() - 7);
        break;
    }

    onFilterChange({
      ...filters,
      timeWindow: { start, end: now }
    });
  };

  // Clear all filters
  const clearAllFilters = () => {
    onFilterChange({});
    setTimePreset('24h');
  };

  // Count active filters
  const activeFilterCount = Object.values(filters).filter(value => 
    Array.isArray(value) ? value.length > 0 : value !== undefined
  ).length;

  return (
    <div className={clsx('bg-white dark:bg-gray-900 rounded-lg', className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center">
          <Filter className="text-gray-600 dark:text-gray-300 mr-2" size={18} />
          <span className="font-medium text-gray-900 dark:text-white">Filters</span>
          {activeFilterCount > 0 && (
            <span className="ml-2 px-2 py-1 bg-blue-600 text-white text-xs rounded-full">
              {activeFilterCount}
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          {activeFilterCount > 0 && (
            <button
              onClick={clearAllFilters}
              className="text-red-600 hover:text-red-700 dark:text-red-400 text-sm flex items-center"
            >
              <X size={14} className="mr-1" />
              Clear All
            </button>
          )}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>

      {/* Filter content */}
      <div className={clsx('overflow-hidden transition-all duration-200', {
        'max-h-0': !isExpanded,
        'max-h-96 overflow-y-auto': isExpanded
      })}>
        <div className="p-4 space-y-4">
          {/* Time window filter */}
          <div>
            <div className="flex items-center mb-2">
              <Calendar size={16} className="text-gray-600 dark:text-gray-300 mr-2" />
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Time Window
              </label>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {(['1h', '4h', '24h', '7d'] as const).map(preset => (
                <button
                  key={preset}
                  onClick={() => handleTimePresetChange(preset)}
                  className={clsx(
                    'px-3 py-2 text-sm rounded-md transition-colors',
                    timePreset === preset
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  )}
                >
                  Last {preset.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Provider filter */}
          {availableProviders.length > 0 && (
            <div>
              <div className="flex items-center mb-2">
                <Server size={16} className="text-gray-600 dark:text-gray-300 mr-2" />
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Providers
                </label>
              </div>
              <div className="space-y-1">
                {availableProviders.map(provider => (
                  <label key={provider} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.provider?.includes(provider) || false}
                      onChange={(e) => handleProviderChange(provider, e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                      {provider}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Change type filter */}
          {availableChangeTypes.length > 0 && (
            <div>
              <div className="flex items-center mb-2">
                <List size={16} className="text-gray-600 dark:text-gray-300 mr-2" />
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Change Types
                </label>
              </div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {availableChangeTypes.map(changeType => (
                  <label key={changeType} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.changeTypes?.includes(changeType) || false}
                      onChange={(e) => handleChangeTypeChange(changeType, e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                      {formatChangeType(changeType)}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Severity level filter */}
          <div>
            <div className="flex items-center mb-2">
              <Hash size={16} className="text-gray-600 dark:text-gray-300 mr-2" />
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Severity Level
              </label>
            </div>
            <div className="space-y-1">
              {(['low', 'medium', 'high', 'critical'] as const).map(severity => (
                <label key={severity} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.severityLevel?.includes(severity) || false}
                    onChange={(e) => handleSeverityChange(severity, e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <div className="ml-2 flex items-center">
                    <div className={clsx(
                      'w-3 h-3 rounded-full mr-2',
                      severity === 'critical' && 'bg-red-500',
                      severity === 'high' && 'bg-orange-500',
                      severity === 'medium' && 'bg-yellow-500',
                      severity === 'low' && 'bg-green-500'
                    )} />
                    <span className="text-sm text-gray-700 dark:text-gray-300 capitalize">
                      {severity}
                    </span>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Request ID filter (limited to first few) */}
          {availableRequestIds.length > 0 && (
            <div>
              <div className="flex items-center mb-2">
                <Hash size={16} className="text-gray-600 dark:text-gray-300 mr-2" />
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Request IDs
                </label>
              </div>
              <div className="space-y-1 max-h-24 overflow-y-auto">
                {availableRequestIds.slice(0, 10).map(requestId => (
                  <label key={requestId} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.requestId?.includes(requestId) || false}
                      onChange={(e) => handleRequestIdChange(requestId, e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-700 dark:text-gray-300 font-mono">
                      {requestId.slice(0, 8)}...
                    </span>
                  </label>
                ))}
                {availableRequestIds.length > 10 && (
                  <div className="text-xs text-gray-500 dark:text-gray-400 italic">
                    ...and {availableRequestIds.length - 10} more
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Helper function to format change types
function formatChangeType(changeType: string): string {
  return changeType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}