import React from 'react';
import { ConnectionStatus } from '../../hooks/useWebSocket';
import { Wifi, WifiOff, Activity, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

interface RealTimeIndicatorProps {
  enabled: boolean;
  status: ConnectionStatus;
  changeCount: number;
  className?: string;
}

/**
 * RealTimeIndicator - Shows WebSocket connection status and real-time activity
 * 
 * Features:
 * - Connection status visualization
 * - Real-time change counter
 * - Animated indicators for active connections
 * - Color-coded status (green: connected, yellow: connecting, red: error)
 */
export const RealTimeIndicator: React.FC<RealTimeIndicatorProps> = ({
  enabled,
  status,
  changeCount,
  className
}) => {
  if (!enabled) {
    return (
      <div className={clsx('flex items-center text-gray-500 dark:text-gray-400', className)}>
        <WifiOff size={16} className="mr-2" />
        <span className="text-sm">Real-time disabled</span>
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (status) {
      case 'connected':
        return <Wifi size={16} className="text-green-600 dark:text-green-400" />;
      case 'connecting':
        return <Wifi size={16} className="text-yellow-600 dark:text-yellow-400 animate-pulse" />;
      case 'error':
        return <AlertCircle size={16} className="text-red-600 dark:text-red-400" />;
      case 'disconnected':
      default:
        return <WifiOff size={16} className="text-gray-600 dark:text-gray-400" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return 'Real-time active';
      case 'connecting':
        return 'Connecting...';
      case 'error':
        return 'Connection error';
      case 'disconnected':
      default:
        return 'Disconnected';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return 'text-green-700 dark:text-green-300';
      case 'connecting':
        return 'text-yellow-700 dark:text-yellow-300';
      case 'error':
        return 'text-red-700 dark:text-red-300';
      case 'disconnected':
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const isActive = status === 'connected';

  return (
    <div className={clsx('flex items-center space-x-3', className)}>
      {/* Connection status */}
      <div className="flex items-center">
        {getStatusIcon()}
        <span className={clsx('ml-2 text-sm font-medium', getStatusColor())}>
          {getStatusText()}
        </span>
      </div>

      {/* Activity indicator */}
      {isActive && (
        <div className="flex items-center">
          <div className="relative">
            <Activity size={16} className="text-blue-600 dark:text-blue-400" />
            {changeCount > 0 && (
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-green-500 rounded-full animate-ping" />
            )}
          </div>
          <span className="ml-2 text-sm text-gray-600 dark:text-gray-300">
            {changeCount.toLocaleString()} changes
          </span>
        </div>
      )}

      {/* Connection dot indicator */}
      <div className="flex items-center">
        <div className={clsx(
          'w-2 h-2 rounded-full',
          status === 'connected' && 'bg-green-500 animate-pulse',
          status === 'connecting' && 'bg-yellow-500 animate-pulse',
          status === 'error' && 'bg-red-500',
          status === 'disconnected' && 'bg-gray-400'
        )} />
        {isActive && (
          <div className="ml-1 text-xs text-gray-500 dark:text-gray-400">
            LIVE
          </div>
        )}
      </div>
    </div>
  );
};