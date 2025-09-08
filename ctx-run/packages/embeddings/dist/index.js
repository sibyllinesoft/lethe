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
exports.FallbackEmbeddings = exports.OllamaEmbeddings = exports.TransformersJsEmbeddings = void 0;
exports.getProvider = getProvider;
class TransformersJsEmbeddings {
    name;
    dim;
    extractor = null;
    constructor(modelId = "Xenova/bge-small-en-v1.5") {
        this.name = modelId;
        this.dim = 384; // Default dimension for bge-small-en-v1.5
    }
    async init() {
        if (!this.extractor) {
            try {
                console.log(`Loading embeddings model: ${this.name}`);
                // Dynamic import to handle ESM
                const { pipeline } = await Promise.resolve().then(() => __importStar(require('@xenova/transformers')));
                this.extractor = await pipeline('feature-extraction', this.name, {
                    // Enable caching for offline use
                    local_files_only: false,
                });
                console.log(`Model loaded successfully (${this.dim}d)`);
            }
            catch (error) {
                console.warn(`Failed to load transformers model: ${error}`);
                // Mark as failed - embed() will return zero vectors
                this.extractor = null;
                throw error; // Re-throw to let getProvider handle it
            }
        }
    }
    async embed(texts) {
        if (!this.extractor) {
            await this.init();
        }
        console.log(`Embedding ${texts.length} texts`);
        // Process texts and get embeddings
        const outputs = await this.extractor(texts, {
            pooling: 'mean',
            normalize: true
        });
        // Convert to Float32Array[]
        const embeddings = [];
        if (outputs.dims && outputs.dims.length === 2) {
            // Batch processing - outputs is [batch_size, embed_dim]
            const [batch_size, embed_dim] = outputs.dims;
            const data = outputs.data;
            for (let i = 0; i < batch_size; i++) {
                const start = i * embed_dim;
                const embedding = new Float32Array(embed_dim);
                for (let j = 0; j < embed_dim; j++) {
                    embedding[j] = data[start + j];
                }
                embeddings.push(embedding);
            }
        }
        else {
            // Single text - convert to array format
            embeddings.push(new Float32Array(outputs.data));
        }
        return embeddings;
    }
}
exports.TransformersJsEmbeddings = TransformersJsEmbeddings;
class OllamaEmbeddings {
    name;
    dim;
    url;
    model;
    constructor(url = "http://localhost:11434", model = "nomic-embed-text") {
        this.url = url;
        this.model = model;
        this.name = `ollama:${model}`;
        this.dim = 768; // nomic-embed-text dimension
    }
    async embed(texts) {
        const embeddings = [];
        for (const text of texts) {
            try {
                const response = await fetch(`${this.url}/api/embeddings`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: text,
                    }),
                });
                if (!response.ok) {
                    throw new Error(`Ollama API error: ${response.status}`);
                }
                const result = await response.json();
                embeddings.push(new Float32Array(result.embedding));
            }
            catch (error) {
                console.error(`Failed to embed text via Ollama: ${error}`);
                // Return zero vector on failure
                embeddings.push(new Float32Array(this.dim));
            }
        }
        return embeddings;
    }
}
exports.OllamaEmbeddings = OllamaEmbeddings;
// Fallback embeddings that return zero vectors
class FallbackEmbeddings {
    name = "fallback";
    dim = 384;
    async embed(texts) {
        console.warn("Using fallback zero-vector embeddings - vector search will be disabled");
        return texts.map(() => new Float32Array(this.dim)); // Zero vectors
    }
}
exports.FallbackEmbeddings = FallbackEmbeddings;
async function getProvider(pref) {
    if (pref === "ollama") {
        // Test Ollama connectivity first
        try {
            const response = await fetch("http://localhost:11434/api/version", {
                method: 'GET',
                signal: AbortSignal.timeout(500), // 500ms timeout as per spec
            });
            if (response.ok) {
                console.log("Using Ollama embeddings");
                return new OllamaEmbeddings();
            }
        }
        catch {
            console.log("Ollama not available, falling back to TransformersJS");
        }
    }
    // Try TransformersJS
    console.log("Using TransformersJS embeddings");
    try {
        const provider = new TransformersJsEmbeddings();
        await provider.init();
        return provider;
    }
    catch (error) {
        console.warn("TransformersJS not available, using fallback embeddings (BM25 only)");
        return new FallbackEmbeddings();
    }
}
//# sourceMappingURL=index.js.map