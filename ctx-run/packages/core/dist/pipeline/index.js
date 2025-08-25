"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.enhancedQuery = enhancedQuery;
exports.diagnoseEnhancedPipeline = diagnoseEnhancedPipeline;
const index_js_1 = require("../state/index.js");
const index_js_2 = require("../hyde/index.js");
const index_js_3 = require("../summarize/index.js");
const index_js_4 = require("../retrieval/index.js");
const index_js_5 = require("../query-understanding/index.js");
const index_js_6 = require("../ml-prediction/index.js");
async function enhancedQuery(originalQuery, options) {
    const startTime = Date.now();
    const { db, embeddings, sessionId, enableHyde = true, enableSummarization = true, enablePlanSelection = true, enableQueryUnderstanding = true, queryUnderstandingConfig, recentTurns = [], 
    // Iteration 3: ML options
    enableMLPrediction = false, mlConfig = {}, 
    // Iteration 4: LLM reranking options
    llmRerankConfig = {} } = options;
    console.log(`🚀 Enhanced query pipeline for: "${originalQuery}"`);
    // Step 1: Query Understanding (Iteration 2) - runs BEFORE HyDE as per specification
    let processedQuery = originalQuery;
    let queryUnderstandingResult = undefined;
    let queryUnderstandingDuration = 0;
    if (enableQueryUnderstanding) {
        const queryUnderstandingStart = Date.now();
        try {
            const result = await (0, index_js_5.processQuery)(db, originalQuery, recentTurns, queryUnderstandingConfig);
            queryUnderstandingResult = {
                canonical_query: result.canonical_query,
                subqueries: result.subqueries,
                rewrite_success: result.rewrite_success,
                decompose_success: result.decompose_success,
                llm_calls_made: result.llm_calls_made,
                errors: result.errors
            };
            // Use rewritten query for downstream processing if successful
            processedQuery = result.canonical_query || originalQuery;
            queryUnderstandingDuration = Date.now() - queryUnderstandingStart;
            console.log(`🔄 Query understanding: ${result.rewrite_success ? 'rewritten' : 'unchanged'}, ${result.subqueries?.length || 0} subqueries in ${queryUnderstandingDuration}ms`);
        }
        catch (error) {
            queryUnderstandingDuration = Date.now() - queryUnderstandingStart;
            console.warn(`Query understanding failed: ${error}`);
            queryUnderstandingResult = {
                canonical_query: originalQuery,
                subqueries: [],
                rewrite_success: false,
                decompose_success: false,
                llm_calls_made: 0,
                errors: [String(error)]
            };
        }
    }
    // Step 2: Plan Selection (with ML enhancement in Iteration 3)
    let plan;
    let mlPrediction;
    let mlPredictionDuration = 0;
    if (enablePlanSelection) {
        // Step 2a: ML-enhanced plan selection (Iteration 3)
        if (enableMLPrediction && mlConfig.plan_learned) {
            const mlStart = Date.now();
            try {
                const mlPredictor = (0, index_js_6.getMLPredictor)({
                    fusion_dynamic: mlConfig.fusion_dynamic ?? false,
                    plan_learned: true
                });
                // Get session context for ML prediction
                const stateManager = (0, index_js_1.getStateManager)(db);
                const sessionContext = stateManager.getRecentContext(sessionId);
                const mlContext = {
                    contradictions: sessionContext.lastPackContradictions.length,
                    entity_overlap: sessionContext.entityCount > 0 ? 0.5 : 0.1, // Simplified
                };
                mlPrediction = await mlPredictor.predictParameters(processedQuery, mlContext);
                mlPredictionDuration = Date.now() - mlStart;
                if (mlPrediction.model_loaded && mlPrediction.plan) {
                    // Use ML-predicted plan with heuristic parameters
                    plan = {
                        plan: mlPrediction.plan,
                        reasoning: `ML-predicted plan based on query characteristics (${mlPrediction.prediction_time_ms.toFixed(1)}ms)`,
                        parameters: getParametersForPlan(mlPrediction.plan)
                    };
                    console.log(`🤖 ML predicted plan: ${mlPrediction.plan} - ${plan.reasoning}`);
                }
                else {
                    console.log('ML plan prediction failed, falling back to heuristic');
                    const stateManager = (0, index_js_1.getStateManager)(db);
                    plan = stateManager.selectPlan(sessionId, processedQuery);
                    console.log(`📋 Heuristic plan selected: ${plan.plan} - ${plan.reasoning}`);
                }
            }
            catch (error) {
                console.warn(`ML plan prediction error: ${error}, using heuristic fallback`);
                const stateManager = (0, index_js_1.getStateManager)(db);
                plan = stateManager.selectPlan(sessionId, processedQuery);
                console.log(`📋 Fallback plan selected: ${plan.plan} - ${plan.reasoning}`);
            }
        }
        else {
            // Step 2b: Traditional heuristic plan selection
            const stateManager = (0, index_js_1.getStateManager)(db);
            plan = stateManager.selectPlan(sessionId, processedQuery);
            console.log(`📋 Plan selected: ${plan.plan} - ${plan.reasoning}`);
        }
    }
    else {
        plan = {
            plan: 'exploit',
            reasoning: 'Plan selection disabled',
            parameters: { hyde_k: 3, beta: 0.5, granularity: 'medium', k_final: 10 }
        };
    }
    // Step 3: HyDE Generation (now uses processed query as input)
    let finalQueries = [processedQuery]; // Use processed query instead of original
    let hydeDuration = 0;
    let hydeQueries;
    const retrievalStart = Date.now();
    const retrievalConfig = {
        alpha: 0.7, // Keep default alpha (will be overridden by ML if enabled)
        beta: plan.parameters.beta || 0.5,
        gamma_kind_boost: { code: 0.1, text: 0.0 },
        rerank: true,
        diversify: true,
        diversify_method: 'semantic', // Use semantic diversification for Iteration 1
        k_initial: (plan.parameters.hyde_k || 3) * 10, // Scale up for initial retrieval
        k_final: plan.parameters.k_final || 10,
        // Iteration 3: ML fusion configuration
        fusion: {
            dynamic: Boolean(enableMLPrediction && mlConfig.fusion_dynamic)
        },
        // Iteration 4: LLM reranking configuration
        llm_rerank: {
            use_llm: llmRerankConfig.use_llm ?? false,
            llm_budget_ms: llmRerankConfig.llm_budget_ms ?? 1200,
            llm_model: llmRerankConfig.llm_model ?? 'llama3.2:1b',
            contradiction_enabled: llmRerankConfig.contradiction_enabled ?? false,
            contradiction_penalty: llmRerankConfig.contradiction_penalty ?? 0.15
        }
    };
    // Run HyDE and initial embedding preparation in parallel (now uses processed query)
    const hydePromise = enableHyde ?
        (0, index_js_2.generateHyde)(db, processedQuery).catch(error => {
            console.warn(`HyDE generation failed: ${error}`);
            return { queries: [processedQuery] };
        }) :
        Promise.resolve({ queries: [processedQuery] });
    const [hydeResult] = await Promise.all([hydePromise]);
    if (enableHyde) {
        hydeQueries = hydeResult.queries;
        finalQueries = hydeResult.queries;
        console.log(`🔍 HyDE generated ${hydeResult.queries.length} queries`);
    }
    // Step 3: Hybrid Retrieval with Plan Parameters
    const candidates = await (0, index_js_4.hybridRetrieval)(finalQueries, {
        db,
        embeddings,
        sessionId,
        config: retrievalConfig
    });
    const retrievalDuration = Date.now() - retrievalStart;
    console.log(`📚 Retrieved ${candidates.length} candidates in ${retrievalDuration}ms`);
    // Step 4: Context Pack Generation with Summarization (use processed query for context)
    let summarizationDuration = 0;
    let pack;
    if (enableSummarization) {
        const summarizeStart = Date.now();
        try {
            pack = await (0, index_js_3.buildContextPack)(db, sessionId, processedQuery, candidates, {
                granularity: plan.parameters.granularity || 'medium'
            });
            summarizationDuration = Date.now() - summarizeStart;
            console.log(`📝 Context pack created in ${summarizationDuration}ms`);
        }
        catch (error) {
            console.warn(`Summarization failed: ${error}`);
            summarizationDuration = Date.now() - summarizeStart;
            // Fallback to simple pack
            pack = createFallbackPack(sessionId, processedQuery, candidates);
        }
    }
    else {
        pack = createFallbackPack(sessionId, processedQuery, candidates);
    }
    // Step 5: Update State
    if (enablePlanSelection && pack) {
        const stateManager = (0, index_js_1.getStateManager)(db);
        stateManager.updateSessionState(sessionId, pack);
        console.log(`💾 Session state updated`);
    }
    const totalDuration = Date.now() - startTime;
    console.log(`✅ Enhanced query complete in ${totalDuration}ms`);
    return {
        pack,
        plan,
        hydeQueries,
        queryUnderstanding: queryUnderstandingResult,
        // Iteration 3: ML prediction results
        mlPrediction: mlPrediction ? {
            alpha: mlPrediction.alpha,
            beta: mlPrediction.beta,
            predicted_plan: mlPrediction.plan,
            prediction_time_ms: mlPrediction.prediction_time_ms,
            model_loaded: mlPrediction.model_loaded
        } : undefined,
        duration: {
            total: totalDuration,
            queryUnderstanding: queryUnderstandingDuration || undefined,
            hyde: hydeDuration || undefined,
            retrieval: retrievalDuration,
            summarization: summarizationDuration || undefined,
            mlPrediction: mlPredictionDuration || undefined
        },
        debug: {
            originalQuery,
            finalQueries,
            retrievalCandidates: candidates.length,
            plan,
            queryProcessingEnabled: enableQueryUnderstanding,
            rewriteFailureRate: queryUnderstandingResult ? (!queryUnderstandingResult.rewrite_success ? 1.0 : 0.0) : undefined,
            decomposeFailureRate: queryUnderstandingResult ? (!queryUnderstandingResult.decompose_success ? 1.0 : 0.0) : undefined,
            // Iteration 3: ML debug info
            mlPredictionEnabled: enableMLPrediction,
            staticAlpha: 0.7,
            staticBeta: plan.parameters.beta || 0.5,
            predictedAlpha: mlPrediction?.alpha,
            predictedBeta: mlPrediction?.beta
        }
    };
}
function createFallbackPack(sessionId, query, candidates) {
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
// Helper function to get parameters for ML-predicted plans
function getParametersForPlan(planType) {
    switch (planType) {
        case 'explore':
            return {
                hyde_k: 4,
                beta: 0.6, // Favor semantic search for exploration
                granularity: 'loose',
                k_final: 12
            };
        case 'verify':
            return {
                hyde_k: 5,
                beta: 0.4, // Slightly favor lexical for verification
                granularity: 'tight',
                k_final: 8
            };
        case 'exploit':
        default:
            return {
                hyde_k: 3,
                beta: 0.5, // Balanced approach
                granularity: 'medium',
                k_final: 10
            };
    }
}
// Diagnostic function for CLI
async function diagnoseEnhancedPipeline(db) {
    const { testOllamaConnection } = await Promise.resolve().then(() => __importStar(require('../ollama/index.js')));
    // Test Ollama
    const ollamaTest = await testOllamaConnection(db);
    // Check state
    const stateManager = (0, index_js_1.getStateManager)(db);
    const sessions = stateManager.getAllSessions();
    // Check config
    const { getConfig } = await Promise.resolve().then(() => __importStar(require('@lethe/sqlite')));
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
//# sourceMappingURL=index.js.map