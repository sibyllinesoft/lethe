import type { EmbeddingProvider } from './types.js';
import { TransformersJsEmbeddings } from './transformers-provider.js';
import { OllamaEmbeddings } from './ollama-provider.js';

export type ProviderType = 'transformersjs' | 'ollama';

export interface ProviderConfig {
  type?: ProviderType;
  model?: string;
  url?: string;
  batchSize?: number;
  timeout?: number;
}

export async function getProvider(config?: ProviderConfig): Promise<EmbeddingProvider> {
  const preferredType = config?.type ?? 'transformersjs';
  
  if (preferredType === 'ollama') {
    const ollama = new OllamaEmbeddings({
      model: config?.model ?? 'nomic-embed-text',
      url: config?.url,
      batchSize: config?.batchSize,
      timeout: config?.timeout
    });
    
    // Check if Ollama is available and model exists
    const isAvailable = await ollama.isAvailable();
    if (!isAvailable) {
      console.warn('Ollama not available, falling back to Transformers.js');
      return getTransformersJsProvider(config);
    }
    
    const modelExists = await ollama.modelExists();
    if (!modelExists) {
      console.log(`Model ${config?.model ?? 'nomic-embed-text'} not found, attempting to pull...`);
      try {
        await ollama.pullModel();
        console.log('Model pulled successfully');
      } catch (error) {
        console.warn(`Failed to pull model: ${error}. Falling back to Transformers.js`);
        return getTransformersJsProvider(config);
      }
    }
    
    return ollama;
  }
  
  return getTransformersJsProvider(config);
}

function getTransformersJsProvider(config?: ProviderConfig): TransformersJsEmbeddings {
  return new TransformersJsEmbeddings({
    model: config?.model ?? 'Xenova/bge-small-en-v1.5',
    batchSize: config?.batchSize,
  });
}

// Convenience function to get default provider (Transformers.js)
export async function getDefaultProvider(): Promise<EmbeddingProvider> {
  return getProvider({ type: 'transformersjs' });
}

// Convenience function to get Ollama provider with fallback
export async function getOllamaProvider(model?: string): Promise<EmbeddingProvider> {
  return getProvider({ type: 'ollama', model });
}