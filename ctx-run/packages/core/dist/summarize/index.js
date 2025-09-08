"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_SUMMARIZE_CONFIG = void 0;
exports.summarizeChunks = summarizeChunks;
exports.buildContextPack = buildContextPack;
exports.testSummarization = testSummarization;
const sqlite_1 = require("@lethe/sqlite");
const index_js_1 = require("../ollama/index.js");
exports.DEFAULT_SUMMARIZE_CONFIG = {
    model: 'xgen-small:4b',
    temperature: 0.2,
    maxTokens: 1024,
    timeoutMs: 10000,
    enabled: true,
    granularity: 'medium'
};
function buildSummarizePrompt(query, chunks, granularity) {
    // Build context from chunks
    const context = chunks
        .map((chunk, i) => `[${i + 1}] ${chunk.kind || 'text'}: ${chunk.text}`)
        .join('\n\n');
    const granularityInstructions = {
        loose: 'Be comprehensive and include related concepts. Include entities that might be tangentially related.',
        medium: 'Focus on directly relevant content. Include the most important entities and clear relationships.',
        tight: 'Be very specific and concise. Only include entities and claims directly mentioned in the context.'
    };
    return `You are an expert at analyzing technical documentation and code for information extraction and summarization.

Query: "${query}"

Context chunks:
${context}

Your task is to analyze this context and extract structured information. Pay special attention to:
- Technical entities (functions, classes, APIs, patterns, tools, libraries)
- Factual claims about how things work or behave
- Any contradictions or inconsistencies between chunks
- Code patterns and architectural decisions

${granularityInstructions[granularity]}

Format your response as JSON:
{
  "summary": "A 2-3 sentence summary addressing the query based on the context",
  "key_entities": [
    "technical_entity_1",
    "technical_entity_2",
    "important_concept_3"
  ],
  "claims": [
    "Factual statement 1 about how something works",
    "Factual statement 2 about behavior or implementation",
    "Factual statement 3 about relationships or dependencies"
  ],
  "contradictions": [
    "Description of any contradiction found between chunks (or empty array if none)"
  ]
}

Be specific and technical in your language. Avoid generic terms.`;
}
async function summarizeChunks(db, query, chunks, config) {
    const finalConfig = { ...exports.DEFAULT_SUMMARIZE_CONFIG, ...config };
    // Check if summarization is enabled
    if (!finalConfig.enabled) {
        console.debug('Summarization disabled in config');
        return createFallbackSummary(query, chunks);
    }
    // Override with database config if available
    try {
        const timeoutConfig = (0, sqlite_1.getConfig)(db, 'timeouts');
        if (timeoutConfig?.summarize_ms) {
            finalConfig.timeoutMs = timeoutConfig.summarize_ms;
        }
    }
    catch (error) {
        console.debug(`Could not load summarization timeout config: ${error}`);
    }
    if (chunks.length === 0) {
        return {
            summary: `No relevant chunks found for query: ${query}`,
            key_entities: [],
            claims: [],
            contradictions: [],
            citations: []
        };
    }
    try {
        const bridge = await (0, index_js_1.getOllamaBridge)(db);
        // Check if Ollama is available
        const isAvailable = await bridge.isAvailable();
        if (!isAvailable) {
            console.warn('Ollama not available, creating fallback summary');
            return createFallbackSummary(query, chunks);
        }
        const prompt = buildSummarizePrompt(query, chunks, finalConfig.granularity);
        console.debug('Generating summary...');
        const startTime = Date.now();
        const response = await bridge.generate({
            model: finalConfig.model,
            prompt,
            temperature: finalConfig.temperature,
            max_tokens: finalConfig.maxTokens
        });
        const duration = Date.now() - startTime;
        console.debug(`Summarization took ${duration}ms`);
        // Parse the JSON response with fallback
        const fallback = createFallbackSummaryData(query, chunks);
        const result = (0, index_js_1.safeParseJSON)(response.response, fallback);
        // Validate and clean up the result
        const cleanResult = validateSummarizeResult(result, fallback);
        // Create citations based on chunk scores
        const citations = chunks.map((chunk, i) => ({
            id: i + 1,
            chunk_id: chunk.docId,
            relevance: chunk.score
        }));
        console.debug(`Summary generated with ${cleanResult.key_entities.length} entities and ${cleanResult.claims.length} claims`);
        return {
            ...cleanResult,
            citations
        };
    }
    catch (error) {
        console.warn(`Summarization failed: ${error}, using fallback`);
        return createFallbackSummary(query, chunks);
    }
}
function createFallbackSummary(query, chunks) {
    const citations = chunks.map((chunk, i) => ({
        id: i + 1,
        chunk_id: chunk.docId,
        relevance: chunk.score
    }));
    // Extract basic entities from chunks
    const entities = new Set();
    chunks.forEach(chunk => {
        if (chunk.text) {
            // Simple entity extraction - look for technical terms
            const words = chunk.text
                .split(/\s+/)
                .filter(word => word.length >= 3 &&
                /^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(word) && // Valid identifier
                word !== word.toLowerCase() // Has capital letters
            );
            words.forEach(word => entities.add(word));
        }
    });
    return {
        summary: `Found ${chunks.length} relevant chunks for "${query}". Content includes ${chunks.map(c => c.kind).filter(Boolean).join(', ') || 'various types of'} information.`,
        key_entities: Array.from(entities).slice(0, 10),
        claims: [`Retrieved ${chunks.length} relevant chunks related to ${query}`],
        contradictions: [],
        citations
    };
}
function createFallbackSummaryData(query, chunks) {
    return {
        summary: `Analysis of ${chunks.length} chunks related to "${query}"`,
        key_entities: [],
        claims: [],
        contradictions: []
    };
}
function validateSummarizeResult(result, fallback) {
    const validated = { ...fallback };
    if (typeof result.summary === 'string' && result.summary.trim().length > 0) {
        validated.summary = result.summary.trim();
    }
    if (Array.isArray(result.key_entities)) {
        validated.key_entities = result.key_entities
            .filter((entity) => typeof entity === 'string' && entity.trim().length > 0)
            .map((entity) => entity.trim())
            .slice(0, 20); // Limit to 20 entities
    }
    if (Array.isArray(result.claims)) {
        validated.claims = result.claims
            .filter((claim) => typeof claim === 'string' && claim.trim().length > 0)
            .map((claim) => claim.trim())
            .slice(0, 15); // Limit to 15 claims
    }
    if (Array.isArray(result.contradictions)) {
        validated.contradictions = result.contradictions
            .filter((contradiction) => typeof contradiction === 'string' && contradiction.trim().length > 0)
            .map((contradiction) => contradiction.trim())
            .slice(0, 10); // Limit to 10 contradictions
    }
    return validated;
}
// Build a complete context pack from query and chunks
async function buildContextPack(db, sessionId, query, chunks, config) {
    const summary = await summarizeChunks(db, query, chunks, config);
    return {
        id: `pack-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        session_id: sessionId,
        query,
        created_at: new Date().toISOString(),
        summary: summary.summary,
        key_entities: summary.key_entities,
        claims: summary.claims,
        contradictions: summary.contradictions,
        chunks: chunks.map(chunk => ({
            id: chunk.docId,
            score: chunk.score,
            kind: chunk.kind || 'text',
            text: chunk.text || ''
        })),
        citations: summary.citations
    };
}
// Test function for CLI diagnostics
async function testSummarization(db, query = 'async error handling') {
    try {
        // Create test chunks
        const testChunks = [
            {
                docId: 'test-1',
                score: 0.9,
                text: 'async function handleError() { try { await operation(); } catch (error) { console.error(error); } }',
                kind: 'code'
            },
            {
                docId: 'test-2',
                score: 0.8,
                text: 'Error handling in async functions requires try-catch blocks to properly catch promises.',
                kind: 'text'
            }
        ];
        const startTime = Date.now();
        const result = await summarizeChunks(db, query, testChunks);
        const duration = Date.now() - startTime;
        return { success: true, result, duration };
    }
    catch (error) {
        return { success: false, error: error?.message || String(error) };
    }
}
//# sourceMappingURL=index.js.map