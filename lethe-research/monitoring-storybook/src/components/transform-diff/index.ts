/**
 * Transform Diff Panel - Component Exports
 */

export { TransformDiffPanel } from '../TransformDiffPanel';
export { ChangeHistogram } from './ChangeHistogram';
export { TokenFlow } from './TokenFlow';
export { KVPrefixHeatmap } from './KVPrefixHeatmap';
export { TimelineView } from './TimelineView';
export { DifficultyGatePanel } from './DifficultyGatePanel';
export { FilterControls } from './FilterControls';
export { ExportControls } from './ExportControls';
export { RealTimeIndicator } from './RealTimeIndicator';

// Re-export types for external usage
export type {
  TransformChangeV2,
  TokenMetrics,
  KVMetrics,
  DifficultyGateMetrics,
  TransformDiffProps,
  FilterOptions,
  WebSocketMessage,
  ChangeAnalysis
} from '../../types/transform';

// Re-export hooks
export { useWebSocket } from '../../hooks/useWebSocket';
export { useTransformAnalysis } from '../../hooks/useTransformAnalysis';

// Re-export utilities
export {
  mockDataScenarios,
  generateTransformChange,
  generateTransformChanges,
  addRealtimeChange
} from '../../utils/mockData';