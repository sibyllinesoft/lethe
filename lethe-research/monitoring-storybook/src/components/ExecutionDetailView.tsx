import React, { useState } from 'react';
import { 
  Clock, 
  Zap, 
  MessageSquare,
  AlertCircle,
  CheckCircle,
  Hash,
  GitCommit,
  User,
  Settings,
  TrendingUp,
  TrendingDown,
  ChevronRight,
  ChevronDown
} from 'lucide-react';
import { ExecutionComparison } from '../types/monitoring';
import { format } from 'date-fns';
import { 
  Card, 
  MetricCard, 
  Text, 
  Heading, 
  Badge, 
  Code,
  designTokens, 
  performanceColors 
} from '../design-system';

interface ExecutionDetailViewProps {
  comparison: ExecutionComparison;
  className?: string;
}

/**
 * Detailed execution view with before/after comparison
 * Based on PromptDashboard.get_before_after_comparison() and get_detailed_execution()
 */
export const ExecutionDetailView: React.FC<ExecutionDetailViewProps> = ({ 
  comparison, 
  className 
}) => {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    prompt: false,
    response: false,
    parameters: false,
    environment: false
  });

  const execution = comparison.current_execution;
  
  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const formatTime = (ms: number): string => {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const formatDate = (dateString: string): string => {
    return format(new Date(dateString), 'PPp');
  };

  const getStatusBadgeVariant = (hasError: boolean): 'success' | 'error' => {
    return hasError ? 'error' : 'success';
  };

  const getStatusIcon = (hasError: boolean) => {
    return hasError 
      ? <AlertCircle className="h-5 w-5" style={{ color: designTokens.colors.error[500] }} />
      : <CheckCircle className="h-5 w-5" style={{ color: designTokens.colors.success[500] }} />;
  };

  const getChangeIcon = (change: string) => {
    if (change.includes('improved') || change.includes('increased')) {
      return <TrendingUp className="h-4 w-4" style={{ color: designTokens.colors.success[500] }} />;
    } else if (change.includes('decreased') || change.includes('slower')) {
      return <TrendingDown className="h-4 w-4" style={{ color: designTokens.colors.error[500] }} />;
    }
    return <AlertCircle className="h-4 w-4" style={{ color: designTokens.colors.data.primary[0] }} />;
  };

  const getQualityColor = (score: number): string => {
    if (score >= 0.9) return performanceColors.excellent;
    if (score >= 0.8) return performanceColors.good;
    if (score >= 0.7) return performanceColors.fair;
    if (score >= 0.6) return performanceColors.poor;
    return performanceColors.critical;
  };

  const SectionHeader = ({ 
    title, 
    section, 
    children 
  }: { 
    title: string; 
    section: string; 
    children: React.ReactNode; 
  }) => (
    <Card className="overflow-hidden">
      <button
        onClick={() => toggleSection(section)}
        className="w-full px-4 py-3 flex items-center justify-between transition-colors duration-200 rounded-t-lg"
        style={{
          backgroundColor: designTokens.colors.graphite[50],
        }}
        onMouseOver={(e) => {
          e.currentTarget.style.backgroundColor = designTokens.colors.graphite[100];
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.backgroundColor = designTokens.colors.graphite[50];
        }}
      >
        <Text weight="medium" color="primary">{title}</Text>
        {expandedSections[section] ? 
          <ChevronDown className="h-4 w-4" style={{ color: designTokens.colors.graphite[500] }} /> :
          <ChevronRight className="h-4 w-4" style={{ color: designTokens.colors.graphite[500] }} />
        }
      </button>
      {expandedSections[section] && (
        <div 
          className="p-4"
          style={{ borderTop: `1px solid ${designTokens.colors.graphite[200]}` }}
        >
          {children}
        </div>
      )}
    </Card>
  );

  return (
    <Card className={className}>
      {/* Header */}
      <div className="p-6" style={{ borderBottom: `1px solid ${designTokens.colors.graphite[200]}` }}>
        <div className="flex items-center justify-between mb-6">
          <Heading level={2} className="text-graphite-900">
            Execution Details
          </Heading>
          <Badge 
            variant={getStatusBadgeVariant(execution.error_occurred)}
            className="flex items-center gap-2"
          >
            {getStatusIcon(execution.error_occurred)}
            <span>
              {execution.error_occurred ? 'Failed' : 'Successful'}
            </span>
          </Badge>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricCard
            title="Execution ID"
            value={execution.execution_id.substring(0, 12) + '...'}
            subtitle="Unique identifier"
            icon={<Hash className="h-8 w-8" />}
            iconColor={designTokens.colors.data.primary[0]}
            backgroundColor={designTokens.colors.data.primary[0] + '10'}
          />
          
          <MetricCard
            title="Execution Time"
            value={formatTime(execution.execution_time_ms)}
            subtitle="Processing duration"
            icon={<Clock className="h-8 w-8" />}
            iconColor={designTokens.colors.data.primary[1]}
            backgroundColor={designTokens.colors.data.primary[1] + '10'}
          />
          
          <MetricCard
            title="Response Length"
            value={`${execution.response_length.toLocaleString()}`}
            subtitle="Characters generated"
            icon={<MessageSquare className="h-8 w-8" />}
            iconColor={designTokens.colors.success[600]}
            backgroundColor={designTokens.colors.success[50]}
          />
          
          <MetricCard
            title="Quality Score"
            value={execution.response_quality_score?.toFixed(3) || 'N/A'}
            subtitle="Response quality"
            icon={<Zap className="h-8 w-8" />}
            iconColor={execution.response_quality_score ? getQualityColor(execution.response_quality_score) : designTokens.colors.graphite[400]}
            backgroundColor={execution.response_quality_score ? getQualityColor(execution.response_quality_score) + '10' : designTokens.colors.graphite[50]}
          />
        </div>
      </div>

      {/* Execution Metadata */}
      <div className="p-6" style={{ borderBottom: `1px solid ${designTokens.colors.graphite[200]}` }}>
        <Heading level={3} className="text-graphite-900 mb-4">Execution Metadata</Heading>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <User className="h-4 w-4 flex-shrink-0" style={{ color: designTokens.colors.graphite[400] }} />
              <Text size="sm" color="secondary">Prompt ID:</Text>
              <Text size="sm" weight="medium" color="primary">{execution.prompt_id}</Text>
            </div>
            <div className="flex items-center gap-2">
              <Settings className="h-4 w-4 flex-shrink-0" style={{ color: designTokens.colors.graphite[400] }} />
              <Text size="sm" color="secondary">Model:</Text>
              <Text size="sm" weight="medium" color="primary">{execution.model_name}</Text>
            </div>
            <div className="flex items-center gap-2">
              <GitCommit className="h-4 w-4 flex-shrink-0" style={{ color: designTokens.colors.graphite[400] }} />
              <Text size="sm" color="secondary">Version:</Text>
              <Text size="sm" weight="medium" color="primary">{execution.prompt_version}</Text>
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 flex-shrink-0" style={{ color: designTokens.colors.graphite[400] }} />
              <Text size="sm" color="secondary">Timestamp:</Text>
              <Text size="sm" weight="medium" color="primary">{formatDate(execution.timestamp)}</Text>
            </div>
            <div className="flex items-center gap-2">
              <Hash className="h-4 w-4 flex-shrink-0" style={{ color: designTokens.colors.graphite[400] }} />
              <Text size="sm" color="secondary">Temperature:</Text>
              <Text size="sm" weight="medium" color="primary">{execution.temperature}</Text>
            </div>
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 flex-shrink-0" style={{ color: designTokens.colors.graphite[400] }} />
              <Text size="sm" color="secondary">Context Length:</Text>
              <Text size="sm" weight="medium" color="primary">{execution.context_length}</Text>
            </div>
          </div>
        </div>
      </div>

      {/* Changes Detected */}
      {comparison.changes_detected.length > 0 && (
        <div className="p-6" style={{ borderBottom: `1px solid ${designTokens.colors.graphite[200]}` }}>
          <Heading level={3} className="text-graphite-900 mb-4">Changes Detected</Heading>
          <div className="space-y-2">
            {comparison.changes_detected.map((change, index) => (
              <div 
                key={index} 
                className="flex items-center gap-3 p-3 rounded-lg"
                style={{ backgroundColor: designTokens.colors.data.primary[0] + '10' }}
              >
                {getChangeIcon(change)}
                <Text size="sm" color="primary">{change}</Text>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expandable Sections */}
      <div className="p-6 space-y-4">
        <SectionHeader title="Prompt Content" section="prompt">
          <div className="space-y-4">
            <div>
              <Text weight="medium" color="primary" className="mb-2">
                Prompt Text
              </Text>
              <div 
                className="rounded-md p-4"
                style={{ backgroundColor: designTokens.colors.graphite[50] }}
              >
                <Code className="text-sm whitespace-pre-wrap">
                  {execution.prompt_text}
                </Code>
              </div>
            </div>
            <div>
              <Text weight="medium" color="primary" className="mb-2">
                Prompt Hash
              </Text>
              <Code 
                className="px-2 py-1 rounded text-sm"
                style={{
                  backgroundColor: designTokens.colors.data.primary[0] + '20',
                  color: designTokens.colors.data.primary[0]
                }}
              >
                {execution.prompt_hash}
              </Code>
            </div>
          </div>
        </SectionHeader>

        <SectionHeader title="Response Content" section="response">
          <div className="space-y-4">
            <div>
              <Text weight="medium" color="primary" className="mb-2">
                Response Text
              </Text>
              <div 
                className="rounded-md p-4 max-h-96 overflow-y-auto"
                style={{ backgroundColor: designTokens.colors.graphite[50] }}
              >
                <Code className="text-sm whitespace-pre-wrap">
                  {execution.response_text || 'No response generated'}
                </Code>
              </div>
            </div>
            {execution.response_tokens && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Text weight="medium" color="primary" className="mb-1">
                    Response Tokens
                  </Text>
                  <Text size="sm" color="secondary">{execution.response_tokens.toLocaleString()}</Text>
                </div>
                <div>
                  <Text weight="medium" color="primary" className="mb-1">
                    Tokens per Second
                  </Text>
                  <Text size="sm" color="secondary">
                    {execution.tokens_per_second?.toFixed(1) || 'N/A'}
                  </Text>
                </div>
              </div>
            )}
          </div>
        </SectionHeader>

        <SectionHeader title="Model Parameters" section="parameters">
          <div 
            className="rounded-md p-4"
            style={{ backgroundColor: designTokens.colors.graphite[50] }}
          >
            <Code className="text-sm">
              {JSON.stringify(execution.model_parameters, null, 2)}
            </Code>
          </div>
        </SectionHeader>

        <SectionHeader title="Environment & Context" section="environment">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Environment Variables
              </label>
              <div className="bg-gray-50 rounded-md p-4">
                <pre className="text-sm text-gray-800">
                  {JSON.stringify(execution.environment, null, 2)}
                </pre>
              </div>
            </div>
            {execution.git_commit && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Git Commit
                </label>
                <code className="text-sm text-blue-600 bg-blue-50 px-2 py-1 rounded">
                  {execution.git_commit}
                </code>
              </div>
            )}
          </div>
        </SectionHeader>
      </div>

      {/* Similar Executions */}
      {comparison.similar_executions.length > 0 && (
        <div className="p-6 border-t border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Similar Executions</h3>
          <div className="space-y-2">
            {comparison.similar_executions.map((exec, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-4">
                  <code className="text-sm text-gray-600">{exec.execution_id?.substring(0, 12)}...</code>
                  <span className="text-sm text-gray-600">{exec.prompt_version}</span>
                  <span className="text-sm text-gray-600">
                    {exec.timestamp && formatDate(exec.timestamp)}
                  </span>
                </div>
                <div className="flex items-center space-x-4 text-sm">
                  <span className="text-gray-600">
                    {formatTime(exec.execution_time_ms || 0)}
                  </span>
                  <span className="text-gray-600">
                    {exec.response_length || 0} chars
                  </span>
                  {exec.response_quality_score && (
                    <span className="text-gray-600">
                      {exec.response_quality_score.toFixed(3)}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Information */}
      {execution.error_occurred && (
        <div 
          className="p-6" 
          style={{ 
            borderTop: `1px solid ${designTokens.colors.graphite[200]}`,
            backgroundColor: designTokens.colors.error[50] 
          }}
        >
          <Heading level={3} className="mb-4" style={{ color: designTokens.colors.error[900] }}>
            Error Information
          </Heading>
          <div className="space-y-3">
            <div>
              <Text weight="medium" className="mb-2" style={{ color: designTokens.colors.error[800] }}>
                Error Type
              </Text>
              <Badge variant="error">
                {execution.error_type}
              </Badge>
            </div>
            <div>
              <Text weight="medium" className="mb-2" style={{ color: designTokens.colors.error[800] }}>
                Error Message
              </Text>
              <div 
                className="rounded-md p-4"
                style={{ backgroundColor: designTokens.colors.error[100] }}
              >
                <Text size="sm" style={{ color: designTokens.colors.error[800] }}>
                  {execution.error_message}
                </Text>
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export default ExecutionDetailView;