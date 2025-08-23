import { getConfig } from '@lethe/sqlite';
import type { DB } from '@lethe/sqlite';

export interface OllamaConfig {
  baseUrl: string;
  connectTimeoutMs: number;
  callTimeoutMs: number;
}

export interface OllamaRequest {
  model: string;
  prompt: string;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
}

export interface OllamaResponse {
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface OllamaBridge {
  generate(request: OllamaRequest): Promise<OllamaResponse>;
  isAvailable(): Promise<boolean>;
  getModels(): Promise<string[]>;
}

class OllamaBridgeImpl implements OllamaBridge {
  private config: OllamaConfig;

  constructor(config: OllamaConfig) {
    this.config = config;
  }

  async isAvailable(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.connectTimeoutMs);

      const response = await fetch(`${this.config.baseUrl}/api/tags`, {
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json'
        }
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      console.debug(`Ollama not available: ${error}`);
      return false;
    }
  }

  async getModels(): Promise<string[]> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.connectTimeoutMs);

      const response = await fetch(`${this.config.baseUrl}/api/tags`, {
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json'
        }
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return (data.models || []).map((model: any) => model.name);
    } catch (error: any) {
      console.warn(`Failed to get Ollama models: ${error?.message || error}`);
      return [];
    }
  }

  async generate(request: OllamaRequest): Promise<OllamaResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.callTimeoutMs);

    try {
      const response = await fetch(`${this.config.baseUrl}/api/generate`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...request,
          stream: false // Force non-streaming for simplicity
        })
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Validate response structure
      if (typeof data.response !== 'string') {
        throw new Error('Invalid Ollama response: missing response field');
      }

      return data as OllamaResponse;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Ollama request timeout after ${this.config.callTimeoutMs}ms`);
      }
      throw error;
    }
  }
}

// Default configuration
export const DEFAULT_OLLAMA_CONFIG: OllamaConfig = {
  baseUrl: 'http://localhost:11434',
  connectTimeoutMs: 500,
  callTimeoutMs: 10000
};

// Cache for bridge instances
const bridgeCache = new Map<string, OllamaBridge>();

export async function getOllamaBridge(db?: DB): Promise<OllamaBridge> {
  let config = DEFAULT_OLLAMA_CONFIG;

  // Override with database config if available
  if (db) {
    try {
      const timeoutConfig = getConfig(db, 'timeouts');
      if (timeoutConfig) {
        config = {
          ...config,
          connectTimeoutMs: timeoutConfig.ollama_connect_ms || config.connectTimeoutMs,
          callTimeoutMs: Math.max(
            timeoutConfig.hyde_ms || config.callTimeoutMs,
            timeoutConfig.summarize_ms || config.callTimeoutMs
          )
        };
      }
    } catch (error: any) {
      console.debug(`Could not load timeout config: ${error?.message || error}`);
    }
  }

  const cacheKey = JSON.stringify(config);
  
  if (!bridgeCache.has(cacheKey)) {
    bridgeCache.set(cacheKey, new OllamaBridgeImpl(config));
  }

  return bridgeCache.get(cacheKey)!;
}

// Helper function to safely parse JSON from Ollama responses
export function safeParseJSON<T>(text: string, fallback: T): T {
  try {
    // Clean up common JSON formatting issues from LLMs
    let cleaned = text.trim();
    
    // Remove markdown code blocks
    cleaned = cleaned.replace(/^```json?\s*|\s*```$/gm, '');
    
    // Find JSON object boundaries
    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');
    
    if (start !== -1 && end !== -1 && end > start) {
      cleaned = cleaned.substring(start, end + 1);
    }
    
    return JSON.parse(cleaned);
  } catch (error) {
    console.warn(`Failed to parse JSON from Ollama response: ${error}`);
    console.debug(`Raw text: ${text}`);
    return fallback;
  }
}

// Test function for CLI diagnostics
export async function testOllamaConnection(db?: DB): Promise<{
  available: boolean;
  models: string[];
  testGeneration?: boolean;
  error?: string;
}> {
  try {
    const bridge = await getOllamaBridge(db);
    
    const available = await bridge.isAvailable();
    if (!available) {
      return { available: false, models: [], error: 'Ollama service not reachable' };
    }

    const models = await bridge.getModels();
    
    // Test basic generation with a simple model if available
    let testGeneration = false;
    if (models.length > 0) {
      try {
        const testModel = models.find(m => m.includes('llama')) || models[0];
        const response = await bridge.generate({
          model: testModel,
          prompt: 'Hello, respond with just "OK"',
          temperature: 0,
          max_tokens: 10
        });
        testGeneration = response.response.includes('OK');
      } catch (error) {
        console.debug(`Test generation failed: ${error}`);
      }
    }

    return { available, models, testGeneration };
  } catch (error: any) {
    return { available: false, models: [], error: error?.message || String(error) };
  }
}