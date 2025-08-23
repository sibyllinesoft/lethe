import { pipeline, env } from '@xenova/transformers';
import type { EmbeddingProvider, EmbeddingConfig } from './types.js';

// Configure transformers.js to use local cache
env.cacheDir = './.transformers-cache';

export class TransformersJsEmbeddings implements EmbeddingProvider {
  name: string;
  dim: number;
  private model: any;
  private batchSize: number;
  private maxLength: number;

  constructor(config: EmbeddingConfig = { model: 'Xenova/bge-small-en-v1.5' }) {
    this.name = config.model;
    this.batchSize = config.batchSize ?? 8;
    this.maxLength = config.maxLength ?? 512;
    
    // Set dimension based on model
    const dimensions: Record<string, number> = {
      'Xenova/bge-small-en-v1.5': 384,
      'Xenova/all-MiniLM-L6-v2': 384,
      'Xenova/all-mpnet-base-v2': 768,
      'Xenova/e5-small-v2': 384,
      'Xenova/e5-base-v2': 768
    };
    
    this.dim = dimensions[config.model] ?? 384;
    this.model = null;
  }

  private async initModel(): Promise<void> {
    if (!this.model) {
      console.log(`Loading embedding model: ${this.name}...`);
      this.model = await pipeline('feature-extraction', this.name, {
        quantized: true,  // Use quantized model for better performance
      });
      console.log(`Model loaded successfully. Dimension: ${this.dim}`);
    }
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    await this.initModel();
    
    const embeddings: Float32Array[] = [];
    
    // Process in batches to manage memory
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const batchEmbeddings = await this.processBatch(batch);
      embeddings.push(...batchEmbeddings);
    }
    
    return embeddings;
  }

  private async processBatch(texts: string[]): Promise<Float32Array[]> {
    // Truncate texts to max length if needed
    const processedTexts = texts.map(text => 
      text.length > this.maxLength * 4 ? text.substring(0, this.maxLength * 4) : text
    );
    
    const outputs = await this.model(processedTexts, {
      pooling: 'mean',     // Mean pooling
      normalize: true,     // Normalize embeddings
    });
    
    const embeddings: Float32Array[] = [];
    
    if (Array.isArray(outputs)) {
      // Multiple texts - outputs is array of tensors
      for (const output of outputs) {
        embeddings.push(new Float32Array(output.data));
      }
    } else {
      // Single text or batch - outputs is tensor with shape [batch_size, dim]
      const { data, dims } = outputs;
      const [batchSize, dim] = dims;
      
      for (let i = 0; i < batchSize; i++) {
        const start = i * dim;
        const end = start + dim;
        embeddings.push(new Float32Array(data.slice(start, end)));
      }
    }
    
    return embeddings;
  }

  // Utility method for single text embedding
  async embedSingle(text: string): Promise<Float32Array> {
    const embeddings = await this.embed([text]);
    return embeddings[0];
  }

  // Compute cosine similarity between two embeddings
  static cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}