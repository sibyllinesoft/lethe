import React, { useState } from 'react';
import { 
  Terminal, 
  Copy, 
  Check, 
  Clock,
  AlertCircle,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';
import { CLIOutput } from '../types/monitoring';
import { format } from 'date-fns';

interface CLIOutputCardProps {
  output: CLIOutput;
  compact?: boolean;
  className?: string;
}

/**
 * Card component displaying a single CLI command output
 * Useful for showing individual command results in a dashboard
 */
export const CLIOutputCard: React.FC<CLIOutputCardProps> = ({ 
  output, 
  compact = false,
  className 
}) => {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(!compact);

  const getStatusColor = (status: CLIOutput['status']) => {
    switch (status) {
      case 'success':
        return 'border-green-200 bg-green-50';
      case 'error':
        return 'border-red-200 bg-red-50';
      case 'warning':
        return 'border-yellow-200 bg-yellow-50';
      default:
        return 'border-gray-200 bg-gray-50';
    }
  };

  const getStatusIcon = (status: CLIOutput['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-600" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
      default:
        return <Terminal className="h-5 w-5 text-gray-600" />;
    }
  };

  const getTextColor = (status: CLIOutput['status']) => {
    switch (status) {
      case 'success':
        return 'text-green-800';
      case 'error':
        return 'text-red-800';
      case 'warning':
        return 'text-yellow-800';
      default:
        return 'text-gray-800';
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const formatOutput = (lines: string[]) => {
    return lines.map((line, index) => (
      <div key={index} className="whitespace-pre-wrap">
        {/* Special formatting for different line types */}
        {line.startsWith('🔍') || line.startsWith('📋') || line.startsWith('⚖️') ? (
          <div className="font-bold text-blue-700">{line}</div>
        ) : line.startsWith('=') || line.startsWith('-') ? (
          <div className="text-gray-500 text-xs">{line}</div>
        ) : line.startsWith('📊') || line.startsWith('🎯') || line.startsWith('✅') || 
             line.startsWith('⚡') || line.startsWith('🕐') || line.startsWith('💾') ||
             line.startsWith('📝') || line.startsWith('📈') || line.startsWith('📤') || 
             line.startsWith('📁') ? (
          <div className="text-green-700">{line}</div>
        ) : line.includes('Error:') || line.startsWith('❌') ? (
          <div className="text-red-700 font-medium">{line}</div>
        ) : line.includes('Warning:') || line.startsWith('⚠️') ? (
          <div className="text-yellow-700 font-medium">{line}</div>
        ) : line.startsWith('🧹') || line.startsWith('💡') ? (
          <div className="text-blue-600">{line}</div>
        ) : (
          <div className="text-gray-700">{line}</div>
        )}
      </div>
    ));
  };

  return (
    <div className={`border rounded-lg ${getStatusColor(output.status)} ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {getStatusIcon(output.status)}
            <div>
              <h3 className={`text-sm font-medium ${getTextColor(output.status)}`}>
                CLI Command
              </h3>
              <code className="text-xs text-gray-600 bg-white px-2 py-1 rounded mt-1 inline-block">
                {output.command}
              </code>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {output.duration_ms && (
              <div className="flex items-center text-xs text-gray-600">
                <Clock className="h-3 w-3 mr-1" />
                {output.duration_ms}ms
              </div>
            )}
            <button
              onClick={() => copyToClipboard(output.output.join('\n'))}
              className="text-gray-500 hover:text-gray-700 transition-colors"
              title="Copy output"
            >
              {copied ? (
                <Check className="h-4 w-4 text-green-600" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>

        {/* Timestamp */}
        <div className="mt-2 text-xs text-gray-500">
          Executed: {format(new Date(output.timestamp), 'PPp')}
        </div>
      </div>

      {/* Output Content */}
      {expanded && (
        <div className="p-4">
          <div className="bg-gray-900 rounded-md p-4 overflow-x-auto">
            <div className="text-sm font-mono text-gray-300 space-y-1">
              {formatOutput(output.output)}
            </div>
          </div>
        </div>
      )}

      {/* Compact Mode Toggle */}
      {compact && (
        <div className="px-4 pb-3">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-blue-600 hover:text-blue-800 underline"
          >
            {expanded ? 'Hide Output' : `Show Output (${output.output.length} lines)`}
          </button>
        </div>
      )}

      {/* Summary for compact mode */}
      {compact && !expanded && (
        <div className="px-4 pb-3">
          <div className="text-xs text-gray-600">
            {output.output.length} lines of output • Status: {output.status}
            {output.duration_ms && ` • ${output.duration_ms}ms`}
          </div>
        </div>
      )}
    </div>
  );
};

export default CLIOutputCard;