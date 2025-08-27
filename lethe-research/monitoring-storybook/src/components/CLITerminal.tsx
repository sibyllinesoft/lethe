import React, { useState, useRef, useEffect } from 'react';
import { 
  Terminal, 
  Copy, 
  Check, 
  Play, 
  Square,
  ChevronRight,
  Search
} from 'lucide-react';
import { CLIOutput } from '../types/monitoring';
import { format } from 'date-fns';
import { designTokens, Text } from '../design-system';

interface CLITerminalProps {
  outputs: CLIOutput[];
  interactive?: boolean;
  className?: string;
}

/**
 * Terminal-style component for displaying CLI outputs
 * Simulates the appearance of the actual CLI tool from PromptMonitorCLI
 */
export const CLITerminal: React.FC<CLITerminalProps> = ({ 
  outputs, 
  interactive = false,
  className 
}) => {
  const [currentCommand, setCurrentCommand] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom when new outputs are added
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [outputs]);

  const getStatusColor = (status: CLIOutput['status']) => {
    switch (status) {
      case 'success':
        return { color: designTokens.colors.success[400] };
      case 'error':
        return { color: designTokens.colors.error[400] };
      case 'warning':
        return { color: designTokens.colors.warning[400] };
      default:
        return { color: designTokens.colors.graphite[300] };
    }
  };

  const getStatusIcon = (status: CLIOutput['status']) => {
    switch (status) {
      case 'success':
        return '✅';
      case 'error':
        return '❌';
      case 'warning':
        return '⚠️';
      default:
        return '💡';
    }
  };

  const getLineIcon = (line: string) => {
    const iconSize = "h-4 w-4 mr-1 inline";
    
    if (line.startsWith('🔍')) return <Search className={iconSize} />;
    if (line.startsWith('📋')) return <ClipboardList className={iconSize} />;
    if (line.startsWith('📊')) return <BarChart3 className={iconSize} />;
    if (line.startsWith('🎯')) return <Target className={iconSize} />;
    if (line.startsWith('✅')) return <CheckCircle className={iconSize} />;
    if (line.startsWith('⚡')) return <Zap className={iconSize} />;
    if (line.startsWith('🕐')) return <Clock className={iconSize} />;
    if (line.startsWith('💾')) return <HardDrive className={iconSize} />;
    if (line.startsWith('❌')) return <X className={iconSize} />;
    if (line.startsWith('⚠️')) return <AlertTriangle className={iconSize} />;
    if (line.startsWith('💡')) return <Lightbulb className={iconSize} />;
    if (line.startsWith('📝')) return <FileText className={iconSize} />;
    if (line.startsWith('📈')) return <TrendingUp className={iconSize} />;
    if (line.startsWith('🧹')) return <Trash2 className={iconSize} />;
    if (line.startsWith('📤')) return <Upload className={iconSize} />;
    if (line.startsWith('📁')) return <FolderOpen className={iconSize} />;
    
    return null;
  };

  const removeEmojiFromLine = (line: string) => {
    return line.replace(/^[🔍📋📊🎯✅⚡🕐💾❌⚠️💡📝📈🧹📤📁]\s*/, '');
  };

  const copyToClipboard = async (text: string, index: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const simulateExecution = async () => {
    if (!currentCommand.trim() || isExecuting) return;
    
    setIsExecuting(true);
    // Simulate execution delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsExecuting(false);
    setCurrentCommand('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      simulateExecution();
    }
  };

  return (
    <div 
      className={`rounded-lg ${className}`}
      style={{
        backgroundColor: designTokens.colors.graphite[950],
        border: `1px solid ${designTokens.colors.graphite[700]}`,
        boxShadow: designTokens.shadows.large
      }}
    >
      {/* Terminal Header */}
      <div 
        className="terminal-header flex items-center justify-between px-4 py-3 rounded-t-lg"
        style={{
          backgroundColor: designTokens.colors.graphite[900],
          borderBottom: `1px solid ${designTokens.colors.graphite[700]}`
        }}
      >
        <div className="flex items-center gap-2">
          <Terminal className="h-5 w-5" style={{ color: designTokens.colors.success[400] }} />
          <Text 
            size="sm" 
            weight="medium" 
            style={{ color: designTokens.colors.graphite[300] }}
          >
            Lethe Prompt Monitor CLI
          </Text>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex gap-2">
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: designTokens.colors.error[500] }}
            />
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: designTokens.colors.warning[500] }}
            />
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: designTokens.colors.success[500] }}
            />
          </div>
        </div>
      </div>

      {/* Terminal Content */}
      <div 
        ref={terminalRef}
        className="terminal-dark p-4 rounded-b-lg overflow-y-auto max-h-96"
        style={{
          backgroundColor: designTokens.colors.graphite[950],
          color: designTokens.colors.graphite[300],
          fontFamily: designTokens.typography.fontFamily.mono.join(', '),
          fontSize: '14px'
        }}
      >
        {/* Welcome Message */}
        <div className="mb-4">
          <div className="mb-2 flex items-center" style={{ color: designTokens.colors.success[400] }}>
            <Search className="h-4 w-4 mr-2" />
            Lethe Prompt Monitoring System v1.0.0
          </div>
          <div 
            className="text-xs mb-2"
            style={{ color: designTokens.colors.graphite[500] }}
          >
            Type 'prompt-monitor --help' for available commands
          </div>
        </div>

        {/* Command Outputs */}
        {outputs.map((output, index) => (
          <div key={index} className="mb-6">
            {/* Command Header */}
            <div className="flex items-center justify-between group">
              <div className="flex items-center gap-2 mb-2">
                <span style={{ color: designTokens.colors.data.primary[0] }}>$</span>
                <span style={{ color: designTokens.colors.graphite[300] }}>{output.command}</span>
                <span style={getStatusColor(output.status)}>{getStatusIcon(output.status)}</span>
                {output.duration_ms && (
                  <span 
                    className="text-xs"
                    style={{ color: designTokens.colors.graphite[500] }}
                  >
                    ({output.duration_ms}ms)
                  </span>
                )}
              </div>
              <button
                onClick={() => copyToClipboard(output.output.join('\n'), index)}
                className="opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ color: designTokens.colors.graphite[500] }}
                onMouseOver={(e) => {
                  e.currentTarget.style.color = designTokens.colors.graphite[300];
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.color = designTokens.colors.graphite[500];
                }}
                title="Copy output"
              >
                {copiedIndex === index ? (
                  <Check className="h-4 w-4" style={{ color: designTokens.colors.success[400] }} />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </button>
            </div>

            {/* Command Output */}
            <div className="pl-4 space-y-1">
              {output.output.map((line, lineIndex) => (
                <div key={lineIndex} className="whitespace-pre-wrap">
                  {/* Special formatting for different line types */}
                  {line.startsWith('🔍') || line.startsWith('📋') || line.startsWith('⚖️') ? (
                    <div 
                      className="font-bold"
                      style={{ color: designTokens.colors.data.secondary[0] }}
                    >
                      {line}
                    </div>
                  ) : line.startsWith('=') || line.startsWith('-') ? (
                    <div style={{ color: designTokens.colors.graphite[600] }}>{line}</div>
                  ) : line.startsWith('📊') || line.startsWith('🎯') || line.startsWith('✅') || 
                       line.startsWith('⚡') || line.startsWith('🕐') || line.startsWith('💾') ||
                       line.startsWith('📝') || line.startsWith('📈') || line.startsWith('📤') || 
                       line.startsWith('📁') ? (
                    <div style={{ color: designTokens.colors.success[400] }}>{line}</div>
                  ) : line.includes('Error:') || line.startsWith('❌') ? (
                    <div style={{ color: designTokens.colors.error[400] }}>{line}</div>
                  ) : line.includes('Warning:') || line.startsWith('⚠️') ? (
                    <div style={{ color: designTokens.colors.warning[400] }}>{line}</div>
                  ) : line.startsWith('🧹') || line.startsWith('💡') ? (
                    <div style={{ color: designTokens.colors.data.secondary[0] }}>{line}</div>
                  ) : (
                    <div style={{ color: designTokens.colors.graphite[300] }}>{line}</div>
                  )}
                </div>
              ))}
            </div>

            {/* Timestamp */}
            <div 
              className="text-xs mt-2 pl-4"
              style={{ color: designTokens.colors.graphite[600] }}
            >
              {format(new Date(output.timestamp), 'yyyy-MM-dd HH:mm:ss')}
            </div>
          </div>
        ))}

        {/* Interactive Input */}
        {interactive && (
          <div className="flex items-center space-x-2">
            <span className="text-blue-400">$</span>
            <input
              type="text"
              value={currentCommand}
              onChange={(e) => setCurrentCommand(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter command..."
              className="flex-1 bg-transparent text-gray-300 outline-none placeholder-gray-600"
              disabled={isExecuting}
            />
            {isExecuting ? (
              <div className="flex items-center text-yellow-400">
                <Square className="h-4 w-4 mr-1 animate-pulse" />
                <span className="text-xs">Executing...</span>
              </div>
            ) : (
              <button
                onClick={simulateExecution}
                disabled={!currentCommand.trim()}
                className="text-green-400 hover:text-green-300 disabled:text-gray-600"
                title="Execute command"
              >
                <Play className="h-4 w-4" />
              </button>
            )}
          </div>
        )}

        {/* Command cursor */}
        {interactive && !isExecuting && (
          <div className="flex items-center space-x-1 mt-2">
            <ChevronRight className="h-3 w-3 text-gray-500 animate-pulse" />
            <span className="text-xs text-gray-500">
              Ready for next command
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default CLITerminal;