export type { EmbeddingProvider, EmbeddingConfig } from './types.js';
export { TransformersJsEmbeddings } from './transformers-provider.js';
export { OllamaEmbeddings, type OllamaConfig } from './ollama-provider.js';
export { 
  getProvider, 
  getDefaultProvider, 
  getOllamaProvider,
  type ProviderType,
  type ProviderConfig 
} from './provider-factory.js';