import type { DB } from '@lethe/sqlite';
import type { Embeddings } from '@lethe/embeddings';
import type { PlanSelection } from '../state/index.js';
import { getStateManager } from '../state/index.js';
import { generateHyde } from '../hyde/index.js';
import { buildContextPack } from '../summarize/index.js';
import { hybridRetrieval } from '../retrieval/index.js';

export interface EnhancedQueryOptions {
  db: DB;
  embeddings: Embeddings;
  sessionId: string;
  enableHyde?: boolean;
  enableSummarization?: boolean;
  enablePlanSelection?: boolean;
}

export interface EnhancedQueryResult {
  pack: any; // ContextPack from summarize
  plan: PlanSelection;
  hydeQueries?: string[];
  duration: {
    total: number;
    hyde?: number;
    retrieval: number;
    summarization?: number;
  };
  debug: {
    originalQuery: string;
    finalQueries: string[];
    retrievalCandidates: number;
    plan: PlanSelection;
  };
}

export async function enhancedQuery(
  originalQuery: string,
  options: EnhancedQueryOptions
): Promise<EnhancedQueryResult> {
  const startTime = Date.now();
  const { db, embeddings, sessionId, enableHyde = true, enableSummarization = true, enablePlanSelection = true } = options;

  console.log(`🚀 Enhanced query pipeline for: "${originalQuery}"`);
  
  // Step 1: Plan Selection
  let plan: PlanSelection;
  if (enablePlanSelection) {
    const stateManager = getStateManager(db);
    plan = stateManager.selectPlan(sessionId, originalQuery);
    console.log(`📋 Plan selected: ${plan.plan} - ${plan.reasoning}`);
  } else {
    plan = {
      plan: 'exploit',
      reasoning: 'Plan selection disabled',
      parameters: { hyde_k: 3, beta: 0.5, granularity: 'medium', k_final: 10 }
    };
  }

  // Step 2: HyDE Generation
  let finalQueries = [originalQuery];
  let hydeDuration = 0;
  let hydeQueries: string[] | undefined;
  
  if (enableHyde) {
    const hydeStart = Date.now();
    try {
      const hydeResult = await generateHyde(db, originalQuery);
      hydeQueries = hydeResult.queries;
      finalQueries = hydeResult.queries;
      hydeDuration = Date.now() - hydeStart;
      console.log(`🔍 HyDE generated ${hydeResult.queries.length} queries in ${hydeDuration}ms`);
    } catch (error) {
      console.warn(`HyDE generation failed: ${error}`);
      hydeDuration = Date.now() - hydeStart;
    }
  }

  // Step 3: Hybrid Retrieval with Plan Parameters
  const retrievalStart = Date.now();
  const retrievalConfig = {
    alpha: 0.7, // Keep default alpha
    beta: plan.parameters.beta || 0.5,
    gamma_kind_boost: { code: 0.1, text: 0.0 },
    rerank: true,
    diversify: true,
    k_initial: (plan.parameters.hyde_k || 3) * 10, // Scale up for initial retrieval
    k_final: plan.parameters.k_final || 10
  };

  const candidates = await hybridRetrieval(finalQueries, {
    db,
    embeddings,
    sessionId,
    config: retrievalConfig
  });
  
  const retrievalDuration = Date.now() - retrievalStart;
  console.log(`📚 Retrieved ${candidates.length} candidates in ${retrievalDuration}ms`);

  // Step 4: Context Pack Generation with Summarization
  let summarizationDuration = 0;
  let pack: any;
  
  if (enableSummarization) {
    const summarizeStart = Date.now();
    try {
      pack = await buildContextPack(db, sessionId, originalQuery, candidates, {
        granularity: plan.parameters.granularity || 'medium'
      });
      summarizationDuration = Date.now() - summarizeStart;
      console.log(`📝 Context pack created in ${summarizationDuration}ms`);
    } catch (error) {
      console.warn(`Summarization failed: ${error}`);
      summarizationDuration = Date.now() - summarizeStart;
      // Fallback to simple pack
      pack = createFallbackPack(sessionId, originalQuery, candidates);
    }
  } else {
    pack = createFallbackPack(sessionId, originalQuery, candidates);
  }

  // Step 5: Update State
  if (enablePlanSelection && pack) {
    const stateManager = getStateManager(db);
    stateManager.updateSessionState(sessionId, pack);
    console.log(`💾 Session state updated`);
  }

  const totalDuration = Date.now() - startTime;
  
  console.log(`✅ Enhanced query complete in ${totalDuration}ms`);

  return {
    pack,
    plan,
    hydeQueries,
    duration: {
      total: totalDuration,
      hyde: hydeDuration || undefined,
      retrieval: retrievalDuration,
      summarization: summarizationDuration || undefined
    },
    debug: {
      originalQuery,
      finalQueries,
      retrievalCandidates: candidates.length,
      plan
    }
  };
}

function createFallbackPack(sessionId: string, query: string, candidates: any[]) {
  return {
    id: `pack-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    session_id: sessionId,
    query,
    created_at: new Date().toISOString(),
    summary: `Retrieved ${candidates.length} relevant chunks using hybrid search`,
    key_entities: [],
    claims: [],
    contradictions: [],
    chunks: candidates.map(candidate => ({
      id: candidate.docId,
      score: candidate.score,
      kind: candidate.kind || 'text',
      text: candidate.text || ''
    })),
    citations: candidates.map((candidate, i) => ({
      id: i + 1,
      chunk_id: candidate.docId,
      relevance: candidate.score
    }))
  };
}

// Diagnostic function for CLI
export async function diagnoseEnhancedPipeline(db: DB): Promise<{
  ollama: { available: boolean; models: string[] };
  state: { sessions: string[] };
  config: any;
}> {
  const { testOllamaConnection } = await import('../ollama/index.js');
  
  // Test Ollama
  const ollamaTest = await testOllamaConnection(db);
  
  // Check state
  const stateManager = getStateManager(db);
  const sessions = stateManager.getAllSessions();
  
  // Check config
  const { getConfig } = await import('@lethe/sqlite');
  const config = {
    retrieval: getConfig(db, 'retrieval'),
    chunking: getConfig(db, 'chunking'),
    timeouts: getConfig(db, 'timeouts')
  };

  return {
    ollama: {
      available: ollamaTest.available,
      models: ollamaTest.models
    },
    state: {
      sessions
    },
    config
  };
}