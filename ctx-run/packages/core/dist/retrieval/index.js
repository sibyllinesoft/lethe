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
exports.DEFAULT_HYBRID_CONFIG = void 0;
exports.bm25Search = bm25Search;
exports.bm25SearchWithDb = bm25SearchWithDb;
exports.vectorSearch = vectorSearch;
exports.vectorSearchWithDb = vectorSearchWithDb;
exports.hybridScore = hybridScore;
exports.hybridRetrieval = hybridRetrieval;
const sqlite_1 = require("@lethe/sqlite");
const index_js_1 = require("../reranker/index.js");
const index_js_2 = require("../diversifier/index.js");
// Tokenize text the same way as DF/IDF
function tokenize(text) {
    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(term => term.length > 1);
}
// Calculate BM25 score for a document
function calculateBM25(termFreqs, docLength, avgDocLength, termIdfMap, k1 = 1.2, b = 0.75) {
    let score = 0;
    for (const [term, tf] of Object.entries(termFreqs)) {
        const idf = termIdfMap[term] || 0;
        if (idf <= 0)
            continue;
        const numerator = tf * (k1 + 1);
        const denominator = tf + k1 * (1 - b + b * (docLength / avgDocLength));
        score += idf * (numerator / denominator);
    }
    return score;
}
async function bm25Search(queries, sessionId, k) {
    // This matches the package contract signature exactly
    throw new Error('bm25Search needs DB instance - will be fixed in CLI integration');
}
async function bm25SearchWithDb(db, queries, sessionId, k) {
    // Get all chunks for the session
    const chunks = (0, sqlite_1.getChunksBySession)(db, sessionId);
    if (chunks.length === 0)
        return [];
    // Get DF/IDF data for the session
    const dfidfData = (0, sqlite_1.getDFIdf)(db, sessionId);
    const termIdfMap = {};
    for (const entry of dfidfData) {
        termIdfMap[entry.term] = entry.idf;
    }
    // Calculate average document length
    const totalLength = chunks.reduce((sum, chunk) => sum + tokenize(chunk.text).length, 0);
    const avgDocLength = totalLength / chunks.length;
    // Combine all query terms
    const allQueryTerms = new Set();
    for (const query of queries) {
        const terms = tokenize(query);
        terms.forEach(term => allQueryTerms.add(term));
    }
    // Score each chunk
    const candidates = [];
    for (const chunk of chunks) {
        const docTerms = tokenize(chunk.text);
        const docLength = docTerms.length;
        // Calculate term frequencies for query terms only
        const termFreqs = {};
        for (const term of docTerms) {
            if (allQueryTerms.has(term)) {
                termFreqs[term] = (termFreqs[term] || 0) + 1;
            }
        }
        // Skip documents with no query terms
        if (Object.keys(termFreqs).length === 0)
            continue;
        const score = calculateBM25(termFreqs, docLength, avgDocLength, termIdfMap);
        if (score > 0) {
            candidates.push({ docId: chunk.id, score });
        }
    }
    // Sort by score descending and take top k
    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, k);
}
async function vectorSearch(qVec, k) {
    // Package contract signature - will delegate to sqlite
    throw new Error('vectorSearch needs DB instance - will be fixed in CLI integration');
}
async function vectorSearchWithDb(db, qVec, k) {
    // Delegate to sqlite's vectorSearch function
    const { vectorSearch } = await Promise.resolve().then(() => __importStar(require('@lethe/sqlite')));
    return await vectorSearch(db, qVec, k);
}
// Normalize scores to [0,1] range
function normalizeBM25Scores(candidates) {
    if (candidates.length === 0)
        return [];
    const maxScore = Math.max(...candidates.map(c => c.score));
    if (maxScore === 0)
        return candidates;
    return candidates.map(c => ({
        docId: c.docId,
        score: c.score / maxScore
    }));
}
function normalizeCosineScores(candidates) {
    return candidates.map(c => ({
        docId: c.docId,
        score: (c.score + 1) / 2 // Transform [-1,1] to [0,1]
    }));
}
function hybridScore(lexical, vector, config) {
    // Normalize scores
    const lexicalNorm = normalizeBM25Scores(lexical);
    const vectorNorm = normalizeCosineScores(vector);
    // Create maps for quick lookup
    const lexicalMap = new Map(lexicalNorm.map(c => [c.docId, c.score]));
    const vectorMap = new Map(vectorNorm.map(c => [c.docId, c.score]));
    // Get all unique document IDs
    const allDocIds = new Set([...lexicalMap.keys(), ...vectorMap.keys()]);
    const candidates = [];
    for (const docId of allDocIds) {
        const lexScore = lexicalMap.get(docId) || 0;
        const vecScore = vectorMap.get(docId) || 0;
        // Basic hybrid score (gamma boost will be added when we have chunk metadata)
        const hybridScore = config.alpha * lexScore + config.beta * vecScore;
        candidates.push({
            docId,
            score: hybridScore
        });
    }
    // Sort by hybrid score descending
    candidates.sort((a, b) => b.score - a.score);
    return candidates;
}
// Default configuration optimized for code retrieval
exports.DEFAULT_HYBRID_CONFIG = {
    alpha: 0.7, // Favor lexical matching for code
    beta: 0.3, // Some semantic understanding
    gamma_kind_boost: {
        'code': 1.2, // Boost code blocks
        'import': 1.1, // Boost import statements
        'function': 1.15, // Boost function definitions
        'error': 1.3 // Boost error messages
    },
    rerank: true,
    diversify: true,
    k_initial: 50, // Retrieve more initially
    k_final: 20 // Return top 20 after processing
};
// Complete hybrid retrieval pipeline
async function hybridRetrieval(queries, options) {
    const config = { ...exports.DEFAULT_HYBRID_CONFIG, ...options.config };
    const { db, embeddings, sessionId } = options;
    console.log(`Starting hybrid retrieval for ${queries.length} queries`);
    console.time('hybrid-retrieval');
    try {
        // Step 1: BM25 lexical search
        console.log('Step 1: BM25 lexical search...');
        const lexicalResults = await bm25SearchWithDb(db, queries, sessionId, config.k_initial);
        console.log(`BM25 found ${lexicalResults.length} candidates`);
        // Step 2: Vector semantic search
        console.log('Step 2: Vector semantic search...');
        let vectorResults = [];
        try {
            // Embed the combined query
            const queryText = queries.join(' ');
            const queryEmbeddings = await embeddings.embed([queryText]);
            if (queryEmbeddings.length > 0) {
                vectorResults = await vectorSearchWithDb(db, queryEmbeddings[0], config.k_initial);
                console.log(`Vector search found ${vectorResults.length} candidates`);
            }
        }
        catch (error) {
            console.warn(`Vector search failed: ${error}, continuing with lexical only`);
        }
        // Step 3: Hybrid scoring
        console.log('Step 3: Hybrid scoring...');
        let candidates = hybridScore(lexicalResults, vectorResults, config);
        console.log(`Hybrid scoring produced ${candidates.length} candidates`);
        // Step 4: Add text and metadata to candidates
        console.log('Step 4: Enriching candidates with text...');
        candidates = await enrichCandidatesWithText(db, candidates);
        // Step 5: Cross-encoder reranking (optional)
        if (config.rerank && candidates.length > 0) {
            console.log('Step 5: Cross-encoder reranking...');
            const reranker = await (0, index_js_1.getReranker)(true);
            const queryText = queries.join(' ');
            candidates = await reranker.rerank(queryText, candidates);
            console.log(`Reranking complete`);
        }
        // Step 6: Diversification (optional)
        if (config.diversify && candidates.length > config.k_final) {
            console.log('Step 6: Diversification...');
            const diversifier = await (0, index_js_2.getDiversifier)(true);
            candidates = await diversifier.diversify(candidates, config.k_final);
            console.log(`Diversification complete`);
        }
        else {
            // Just take top k if no diversification
            candidates = candidates.slice(0, config.k_final);
        }
        console.timeEnd('hybrid-retrieval');
        console.log(`Hybrid retrieval complete: ${candidates.length} final results`);
        return candidates;
    }
    catch (error) {
        console.timeEnd('hybrid-retrieval');
        console.error(`Hybrid retrieval failed: ${error}`);
        throw error;
    }
}
// Enrich candidates with text content and metadata
async function enrichCandidatesWithText(db, candidates) {
    const { getChunkById } = await Promise.resolve().then(() => __importStar(require('@lethe/sqlite')));
    const enrichedCandidates = [];
    for (const candidate of candidates) {
        try {
            const chunk = getChunkById(db, candidate.docId);
            if (chunk) {
                enrichedCandidates.push({
                    ...candidate,
                    text: chunk.text,
                    kind: chunk.kind
                });
            }
            else {
                // Keep candidate without text if chunk not found
                enrichedCandidates.push(candidate);
            }
        }
        catch (error) {
            console.warn(`Failed to enrich candidate ${candidate.docId}: ${error}`);
            enrichedCandidates.push(candidate);
        }
    }
    return enrichedCandidates;
}
//# sourceMappingURL=index.js.map