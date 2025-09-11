/**
 * TypeScript interfaces for Transform Diff Panel
 * Based on TransformChangeV2 Rust schema
 */

export interface TransformChangeV2 {
  schema_version: number;
  timestamp: string; // ISO 8601 datetime
  change_type: ChangeType;
  metadata: ChangeMetadata;
  context: TransformContext;
}

export interface ChangeType {
  // System message manipulations
  system_prelude_added?: {
    position: MessagePosition;
    content_hash?: string;
  };
  system_prelude_prepended?: {
    existing_content_hash?: string;
    merged_content_hash?: string;
  };
  system_message_modified?: {
    modification_type: SystemModificationType;
    before_hash?: string;
    after_hash?: string;
  };
  
  // User content transformations
  user_content_rewritten?: {
    rewrite_type: ContentRewriteType;
    affected_messages: number;
    content_size_change: number;
  };
  user_message_enhanced?: {
    enhancement_type: EnhancementType;
    metadata: Record<string, string>;
  };
  
  // Legacy format handling
  legacy_prompt_rewritten?: {
    conversion_type: LegacyConversionType;
    prompt_length?: number;
  };
  legacy_format_converted?: {
    from_format: string;
    to_format: string;
    conversion_notes?: string;
  };
  
  // Content structure changes
  message_structure_normalized?: {
    original_structure: string;
    normalized_structure: string;
  };
  content_encoding_changed?: {
    from_encoding: string;
    to_encoding: string;
  };
  
  // Security and validation
  content_sanitized?: {
    sanitization_rules: string[];
    removed_elements: string[];
  };
  validation_applied?: {
    validation_type: ValidationType;
    validation_result: ValidationResult;
  };
  
  // No-op cases
  no_changes_applied?: {
    reason: NoChangeReason;
  };
}

export type MessagePosition = 
  | 'beginning'
  | 'end'
  | { before_index: number }
  | { after_index: number };

export type SystemModificationType =
  | 'content_appended'
  | 'content_prepended'
  | 'content_replaced'
  | 'content_merged';

export type ContentRewriteType =
  | 'language_normalization'
  | 'format_standardization'
  | 'content_expansion'
  | 'content_compression'
  | 'structural_reorganization';

export type EnhancementType =
  | 'context_enrichment'
  | 'metadata_addition'
  | 'quality_improvement'
  | 'accessibility_enhancement';

export type LegacyConversionType =
  | 'prompt_to_messages'
  | 'completion_to_chat'
  | 'text_to_structured';

export type ValidationType =
  | 'schema_validation'
  | 'content_validation'
  | 'security_validation'
  | 'length_validation';

export interface ValidationResult {
  passed?: null;
  passed_with_warnings?: string[];
  failed?: string[];
  corrected?: string[];
}

export type NoChangeReason =
  | 'already_optimal'
  | 'rewriting_disabled'
  | 'unsupported_format'
  | 'insufficient_data'
  | 'error_prevented';

export interface ChangeMetadata {
  performance?: TransformPerformanceMetrics;
  config_version?: string;
  config_hash?: string;
  confidence_score?: number;
  quality_score?: number;
  reversible: boolean;
  rollback_data?: string;
}

export interface TransformPerformanceMetrics {
  duration_us: number;
  memory_used_bytes?: number;
  cpu_cycles?: number;
  input_size_bytes: number;
  output_size_bytes: number;
}

export interface TransformContext {
  request_id?: string;
  target_provider?: string;
  target_model?: string;
  user_context?: string;
  feature_flags: string[];
  experiment_bucket?: string;
}

// Enhanced interfaces for visualization
export interface TokenMetrics {
  before: number;
  after: number;
  delta: number;
  efficiency: number; // ratio of after/before
}

export interface KVMetrics {
  prefixJaccard: number; // Jaccard similarity index for prefix matching
  volatility: number; // measure of change frequency
  headEdits: number; // number of edits in the first N tokens
  tailEdits: number; // number of edits in the last N tokens
}

export interface ChangeHistogramData {
  changeType: string;
  count: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  avgPerformanceImpact: number; // microseconds
  successRate: number; // percentage
}

export interface TimelineEvent {
  id: string;
  timestamp: Date;
  changeType: string;
  duration: number;
  success: boolean;
  causality?: string[]; // IDs of events that caused this one
}

export interface DifficultyGateMetrics {
  changeEntropy: number; // Shannon entropy of change types
  rollbackFrequency: number; // how often changes get rolled back
  editDepth: number; // average depth of edit operations
  complexityScore: number; // composite complexity metric
  recommendedK2Cap: number; // suggested token limit
  recommendedDimension: 256 | 768; // embedding dimension
}

export interface TransformDiffProps {
  changes: TransformChangeV2[];
  tokenMetrics: TokenMetrics;
  kvMetrics: KVMetrics;
  difficultyGateMetrics: DifficultyGateMetrics;
  onExport?: (format: 'png' | 'svg' | 'json') => void;
  onFilter?: (filters: FilterOptions) => void;
  realTimeEnabled?: boolean;
  websocketUrl?: string;
}

export interface FilterOptions {
  provider?: string[];
  requestId?: string[];
  timeWindow?: {
    start: Date;
    end: Date;
  };
  changeTypes?: string[];
  severityLevel?: ('low' | 'medium' | 'high' | 'critical')[];
}

export interface WebSocketMessage {
  type: 'transform_change' | 'metrics_update' | 'difficulty_gate_update';
  data: TransformChangeV2 | TokenMetrics | KVMetrics | DifficultyGateMetrics;
  timestamp: string;
}

// Utility types for change type extraction
export type ChangeTypeKey = keyof ChangeType;

export interface ChangeAnalysis {
  totalChanges: number;
  successRate: number;
  avgPerformanceImpact: number;
  topChangeTypes: Array<{
    type: string;
    count: number;
    percentage: number;
  }>;
  timeRange: {
    start: Date;
    end: Date;
  };
}