export interface EmbeddingProvider {
  name: string;
  dim: number;
  embed(texts: string[]): Promise<Float32Array[]>;
}

export interface EmbeddingConfig {
  model: string;
  batchSize?: number;
  maxLength?: number;
}