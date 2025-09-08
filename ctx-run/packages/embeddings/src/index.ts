import type { Pipeline } from '@xenova/transformers';

export interface Embeddings {
    name: string;
    dim: number;
    embed(texts: string[]): Promise<Float32Array[]>;
}

export class TransformersJsEmbeddings implements Embeddings {
    name: string;
    dim: number;
    private extractor: Pipeline | null = null;

    constructor(modelId: string = "Xenova/bge-small-en-v1.5") {
        this.name = modelId;
        this.dim = 384; // Default dimension for bge-small-en-v1.5
    }

    async init(): Promise<void> {
        if (!this.extractor) {
            try {
                console.log(`Loading embeddings model: ${this.name}`);
                // Dynamic import to handle ESM
                const { pipeline } = await import('@xenova/transformers');
                this.extractor = await pipeline('feature-extraction', this.name, {
                    // Enable caching for offline use
                    local_files_only: false,
                }) as Pipeline;
                console.log(`Model loaded successfully (${this.dim}d)`);
            } catch (error) {
                console.warn(`Failed to load transformers model: ${error}`);
                // Mark as failed - embed() will return zero vectors
                this.extractor = null;
                throw error; // Re-throw to let getProvider handle it
            }
        }
    }

    async embed(texts: string[]): Promise<Float32Array[]> {
        if (!this.extractor) {
            await this.init();
        }

        console.log(`Embedding ${texts.length} texts`);
        
        // Process texts and get embeddings
        const outputs = await this.extractor!(texts, { 
            pooling: 'mean', 
            normalize: true 
        }) as any;
        
        // Convert to Float32Array[]
        const embeddings: Float32Array[] = [];
        
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
        } else {
            // Single text - convert to array format
            embeddings.push(new Float32Array(outputs.data));
        }
        
        return embeddings;
    }
}

export class OllamaEmbeddings implements Embeddings {
    name: string;
    dim: number;
    private url: string;
    private model: string;

    constructor(url: string = "http://localhost:11434", model: string = "nomic-embed-text") {
        this.url = url;
        this.model = model;
        this.name = `ollama:${model}`;
        this.dim = 768; // nomic-embed-text dimension
    }

    async embed(texts: string[]): Promise<Float32Array[]> {
        const embeddings: Float32Array[] = [];
        
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
            } catch (error) {
                console.error(`Failed to embed text via Ollama: ${error}`);
                // Return zero vector on failure
                embeddings.push(new Float32Array(this.dim));
            }
        }
        
        return embeddings;
    }
}

// Fallback embeddings that return zero vectors
export class FallbackEmbeddings implements Embeddings {
    name = "fallback";
    dim = 384;

    async embed(texts: string[]): Promise<Float32Array[]> {
        console.warn("Using fallback zero-vector embeddings - vector search will be disabled");
        return texts.map(() => new Float32Array(this.dim)); // Zero vectors
    }
}

export async function getProvider(pref?: "transformersjs" | "ollama"): Promise<Embeddings> {
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
        } catch {
            console.log("Ollama not available, falling back to TransformersJS");
        }
    }
    
    // Try TransformersJS
    console.log("Using TransformersJS embeddings");
    try {
        const provider = new TransformersJsEmbeddings();
        await provider.init();
        return provider;
    } catch (error) {
        console.warn("TransformersJS not available, using fallback embeddings (BM25 only)");
        return new FallbackEmbeddings();
    }
}