import axios, { AxiosInstance } from 'axios';
import type { EmbeddingProvider, EmbeddingConfig } from './types.js';

export interface OllamaConfig extends EmbeddingConfig {
  url?: string;
  timeout?: number;
}

export class OllamaEmbeddings implements EmbeddingProvider {
  name: string;
  dim: number;
  private client: AxiosInstance;
  private batchSize: number;

  constructor(config: OllamaConfig = { model: 'nomic-embed-text' }) {
    this.name = config.model;
    this.batchSize = config.batchSize ?? 4; // Smaller batches for Ollama
    
    // Set dimension based on model
    const dimensions: Record<string, number> = {
      'nomic-embed-text': 768,
      'all-minilm': 384,
      'mxbai-embed-large': 1024
    };
    
    this.dim = dimensions[config.model] ?? 768;
    
    this.client = axios.create({
      baseURL: config.url ?? 'http://localhost:11434',
      timeout: config.timeout ?? 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async isAvailable(): Promise<boolean> {
    try {
      const response = await this.client.get('/api/version');
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    const embeddings: Float32Array[] = [];
    
    // Process in batches
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const batchEmbeddings = await this.processBatch(batch);
      embeddings.push(...batchEmbeddings);
    }
    
    return embeddings;
  }

  private async processBatch(texts: string[]): Promise<Float32Array[]> {
    const embeddings: Float32Array[] = [];
    
    // Ollama processes one text at a time for embeddings
    for (const text of texts) {
      try {
        const response = await this.client.post('/api/embeddings', {
          model: this.name,
          prompt: text
        });
        
        if (response.data?.embedding) {
          embeddings.push(new Float32Array(response.data.embedding));
        } else {
          throw new Error(`Invalid response format from Ollama`);
        }
      } catch (error) {
        throw new Error(`Failed to get embedding from Ollama: ${error}`);
      }
    }
    
    return embeddings;
  }

  async embedSingle(text: string): Promise<Float32Array> {
    const embeddings = await this.embed([text]);
    return embeddings[0];
  }

  // Check if a model is available in Ollama
  async modelExists(): Promise<boolean> {
    try {
      const response = await this.client.get('/api/tags');
      const models = response.data?.models ?? [];
      return models.some((model: any) => model.name === this.name);
    } catch {
      return false;
    }
  }

  // Pull a model if it's not available
  async pullModel(): Promise<void> {
    try {
      const response = await this.client.post('/api/pull', {
        name: this.name
      }, {
        timeout: 300000 // 5 minute timeout for model pull
      });
      
      if (response.status !== 200) {
        throw new Error(`Failed to pull model ${this.name}`);
      }
    } catch (error) {
      throw new Error(`Failed to pull Ollama model: ${error}`);
    }
  }
}