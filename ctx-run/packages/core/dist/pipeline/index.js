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
async function enhancedQuery(originalQuery, options) {
    const startTime = Date.now();
    const { db, embeddings, sessionId, enableHyde = true, enableSummarization = true, enablePlanSelection = true } = options;
    console.log(`🚀 Enhanced query pipeline for: "${originalQuery}"`);
    // Step 1: Plan Selection
    let plan;
    if (enablePlanSelection) {
        const stateManager = (0, index_js_1.getStateManager)(db);
        plan = stateManager.selectPlan(sessionId, originalQuery);
        console.log(`📋 Plan selected: ${plan.plan} - ${plan.reasoning}`);
    }
    else {
        plan = {
            plan: 'exploit',
            reasoning: 'Plan selection disabled',
            parameters: { hyde_k: 3, beta: 0.5, granularity: 'medium', k_final: 10 }
        };
    }
    // Step 2: HyDE Generation
    let finalQueries = [originalQuery];
    let hydeDuration = 0;
    let hydeQueries;
    if (enableHyde) {
        const hydeStart = Date.now();
        try {
            const hydeResult = await (0, index_js_2.generateHyde)(db, originalQuery);
            hydeQueries = hydeResult.queries;
            finalQueries = hydeResult.queries;
            hydeDuration = Date.now() - hydeStart;
            console.log(`🔍 HyDE generated ${hydeResult.queries.length} queries in ${hydeDuration}ms`);
        }
        catch (error) {
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
    const candidates = await (0, index_js_4.hybridRetrieval)(finalQueries, {
        db,
        embeddings,
        sessionId,
        config: retrievalConfig
    });
    const retrievalDuration = Date.now() - retrievalStart;
    console.log(`📚 Retrieved ${candidates.length} candidates in ${retrievalDuration}ms`);
    // Step 4: Context Pack Generation with Summarization
    let summarizationDuration = 0;
    let pack;
    if (enableSummarization) {
        const summarizeStart = Date.now();
        try {
            pack = await (0, index_js_3.buildContextPack)(db, sessionId, originalQuery, candidates, {
                granularity: plan.parameters.granularity || 'medium'
            });
            summarizationDuration = Date.now() - summarizeStart;
            console.log(`📝 Context pack created in ${summarizationDuration}ms`);
        }
        catch (error) {
            console.warn(`Summarization failed: ${error}`);
            summarizationDuration = Date.now() - summarizeStart;
            // Fallback to simple pack
            pack = createFallbackPack(sessionId, originalQuery, candidates);
        }
    }
    else {
        pack = createFallbackPack(sessionId, originalQuery, candidates);
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