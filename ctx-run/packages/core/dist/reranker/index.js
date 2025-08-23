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
exports.NoOpReranker = exports.CrossEncoderReranker = void 0;
exports.getReranker = getReranker;
class CrossEncoderReranker {
    name;
    model = null;
    constructor(modelId = "Xenova/ms-marco-MiniLM-L-6-v2") {
        this.name = modelId;
    }
    async init() {
        if (!this.model) {
            try {
                console.log(`Loading cross-encoder model: ${this.name}`);
                // Dynamic import to handle ESM
                const { pipeline } = await Promise.resolve().then(() => __importStar(require('@xenova/transformers')));
                this.model = await pipeline('text-classification', this.name, {
                    local_files_only: false,
                });
                console.log(`Cross-encoder loaded successfully`);
            }
            catch (error) {
                console.warn(`Failed to load cross-encoder: ${error}`);
                // Graceful degradation - use simple text similarity
                this.model = null;
            }
        }
    }
    async rerank(query, candidates) {
        if (!this.model) {
            await this.init();
        }
        if (!this.model) {
            // Fallback: simple text similarity scoring
            console.log("Using fallback text similarity for reranking");
            return this.fallbackRerank(query, candidates);
        }
        try {
            console.log(`Reranking ${candidates.length} candidates with cross-encoder`);
            // Prepare query-document pairs
            const pairs = [];
            for (const candidate of candidates) {
                if (candidate.text) {
                    pairs.push(`${query} [SEP] ${candidate.text}`);
                }
            }
            if (pairs.length === 0) {
                return candidates; // No text to rerank
            }
            // Get relevance scores from cross-encoder
            const outputs = await this.model(pairs);
            // Update candidate scores with cross-encoder relevance
            const rerankedCandidates = [];
            let pairIndex = 0;
            for (let i = 0; i < candidates.length; i++) {
                const candidate = candidates[i];
                if (candidate.text) {
                    const output = outputs[pairIndex];
                    // Cross-encoder typically outputs [irrelevant, relevant] scores
                    const relevanceScore = Array.isArray(output) ?
                        (output.find(o => o.label === 'LABEL_1' || o.label === 'relevant')?.score || output[1]?.score || 0.5) :
                        (output.score || 0.5);
                    rerankedCandidates.push({
                        ...candidate,
                        score: relevanceScore // Replace with cross-encoder score
                    });
                    pairIndex++;
                }
                else {
                    // Keep original score if no text available
                    rerankedCandidates.push(candidate);
                }
            }
            // Sort by new relevance scores
            rerankedCandidates.sort((a, b) => b.score - a.score);
            console.log(`Reranking complete - score range: ${rerankedCandidates[0]?.score.toFixed(3)} to ${rerankedCandidates[rerankedCandidates.length - 1]?.score.toFixed(3)}`);
            return rerankedCandidates;
        }
        catch (error) {
            console.error(`Cross-encoder reranking failed: ${error}`);
            return this.fallbackRerank(query, candidates);
        }
    }
    fallbackRerank(query, candidates) {
        // Simple text similarity fallback
        const queryTerms = this.tokenize(query.toLowerCase());
        return candidates.map(candidate => {
            if (!candidate.text) {
                return candidate;
            }
            const docTerms = this.tokenize(candidate.text.toLowerCase());
            const overlap = queryTerms.filter(term => docTerms.includes(term));
            const similarityScore = overlap.length / Math.sqrt(queryTerms.length * docTerms.length);
            return {
                ...candidate,
                score: similarityScore * 0.5 + candidate.score * 0.5 // Blend with original score
            };
        }).sort((a, b) => b.score - a.score);
    }
    tokenize(text) {
        return text
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(term => term.length > 1);
    }
}
exports.CrossEncoderReranker = CrossEncoderReranker;
class NoOpReranker {
    name = "noop";
    async rerank(query, candidates) {
        return candidates; // Pass through unchanged
    }
}
exports.NoOpReranker = NoOpReranker;
async function getReranker(enabled = true) {
    if (!enabled) {
        return new NoOpReranker();
    }
    try {
        const reranker = new CrossEncoderReranker();
        await reranker.init();
        return reranker;
    }
    catch (error) {
        console.warn(`Reranker initialization failed: ${error}, using no-op`);
        return new NoOpReranker();
    }
}
//# sourceMappingURL=index.js.map