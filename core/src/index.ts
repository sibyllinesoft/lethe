export { ContextOrchestrator, type OrchestrationResult } from './orchestrator.js';
export { MessageChunker, type ChunkingConfig } from './chunker.js';
export { DfIdfBuilder } from './dfidf.js';
export { RetrievalSystem, type SearchResult } from './retrieval.js';
export { AIIntegration, type AIConfig } from './ai-integration.js';

// Re-export types from shared types package
export type { 
  Config, 
  Candidate, 
  HydeResult, 
  ContextPack, 
  PlanType,
  LetheError,
  Result,
  PerformanceMetrics,
  TelemetryEvent
} from '@lethe/types';
