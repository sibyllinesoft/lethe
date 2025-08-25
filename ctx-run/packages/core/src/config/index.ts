import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

/**
 * External configuration structure for ctx-run
 * Externalized from database storage for easier management and version control
 */
export interface CtxConfig {
  version: string;
  description?: string;
  
  retrieval: {
    alpha: number;
    beta: number;
    gamma_kind_boost: {
      code: number;
      text: number;
    };
    fusion?: {
      dynamic: boolean;
    };
    llm_rerank?: {
      use_llm: boolean;
      llm_budget_ms: number;
      llm_model: string;
      contradiction_enabled: boolean;
      contradiction_penalty: number;
    };
  };
  
  chunking: {
    target_tokens: number;
    overlap: number;
    method?: string;
  };
  
  timeouts: {
    hyde_ms: number;
    summarize_ms: number;
    ollama_connect_ms: number;
    ml_prediction_ms?: number;
  };
  
  features?: {
    enable_hyde: boolean;
    enable_summarization: boolean;
    enable_plan_selection: boolean;
    enable_query_understanding: boolean;
    enable_ml_prediction: boolean;
    enable_state_tracking: boolean;
  };
  
  query_understanding?: {
    rewrite_enabled: boolean;
    decompose_enabled: boolean;
    max_subqueries: number;
    llm_model: string;
    temperature: number;
  };
  
  ml?: {
    prediction_service?: {
      enabled: boolean;
      host: string;
      port: number;
      timeout_ms: number;
      fallback_to_static: boolean;
    };
    models?: {
      plan_selector?: string;
      fusion_weights?: string;
      feature_extractor?: string;
    };
  };
  
  development?: {
    debug_enabled: boolean;
    profiling_enabled: boolean;
    log_level: string;
  };
}

/**
 * Default configuration fallback
 * Used when ctx.config.json is not found or has missing sections
 */
export const DEFAULT_CONFIG: CtxConfig = {
  version: "1.0.0",
  description: "Default ctx-run configuration",
  
  retrieval: {
    alpha: 0.7,
    beta: 0.5,
    gamma_kind_boost: {
      code: 0.1,
      text: 0.0
    },
    fusion: {
      dynamic: false
    },
    llm_rerank: {
      use_llm: false,
      llm_budget_ms: 1200,
      llm_model: "llama3.2:1b",
      contradiction_enabled: false,
      contradiction_penalty: 0.15
    }
  },
  
  chunking: {
    target_tokens: 320,
    overlap: 64,
    method: "semantic"
  },
  
  timeouts: {
    hyde_ms: 10000,
    summarize_ms: 10000,
    ollama_connect_ms: 500,
    ml_prediction_ms: 2000
  },
  
  features: {
    enable_hyde: true,
    enable_summarization: true,
    enable_plan_selection: true,
    enable_query_understanding: true,
    enable_ml_prediction: false,
    enable_state_tracking: true
  },
  
  query_understanding: {
    rewrite_enabled: true,
    decompose_enabled: true,
    max_subqueries: 3,
    llm_model: "llama3.2:1b",
    temperature: 0.1
  },
  
  ml: {
    prediction_service: {
      enabled: false,
      host: "127.0.0.1",
      port: 8080,
      timeout_ms: 2000,
      fallback_to_static: true
    },
    models: {
      plan_selector: "learned_plan_selector.joblib",
      fusion_weights: "dynamic_fusion_model.joblib",
      feature_extractor: "feature_extractor.json"
    }
  },
  
  development: {
    debug_enabled: false,
    profiling_enabled: false,
    log_level: "info"
  }
};

/**
 * Global configuration cache
 * Loaded once and reused to avoid filesystem overhead
 */
let configCache: CtxConfig | null = null;

/**
 * Load configuration from ctx.config.json with fallback to defaults
 * Supports both project-local and package-relative config files
 */
export function loadConfig(basePath?: string): CtxConfig {
  // Return cached config if available
  if (configCache) {
    return configCache;
  }
  
  // Configuration file paths to try (in order of priority)
  const configPaths = [
    // 1. User-specified base path
    basePath ? join(basePath, 'ctx.config.json') : null,
    // 2. Current working directory
    join(process.cwd(), 'ctx.config.json'),
    // 3. Project root (ctx-run directory)
    join(process.cwd(), 'ctx-run', 'ctx.config.json'),
    // 4. Parent of current working directory
    join(dirname(process.cwd()), 'ctx.config.json'),
    // 5. Package-relative fallback (for development)
    join(getPackageRoot(), '..', '..', 'ctx.config.json')
  ].filter(Boolean) as string[];
  
  let loadedConfig: Partial<CtxConfig> = {};
  let configFound = false;
  
  for (const configPath of configPaths) {
    if (existsSync(configPath)) {
      try {
        const configData = readFileSync(configPath, 'utf-8');
        loadedConfig = JSON.parse(configData);
        configFound = true;
        console.log(`📋 Loaded configuration from: ${configPath}`);
        break;
      } catch (error) {
        console.warn(`Warning: Failed to parse config at ${configPath}: ${error}`);
        continue;
      }
    }
  }
  
  if (!configFound) {
    console.log(`📋 No ctx.config.json found, using defaults`);
    console.log(`   Searched: ${configPaths.slice(0, 3).join(', ')}`);
  }
  
  // Deep merge with defaults to handle partial configurations
  configCache = mergeWithDefaults(loadedConfig);
  
  return configCache;
}

/**
 * Force reload configuration (clears cache)
 * Useful for testing or when config file changes during runtime
 */
export function reloadConfig(basePath?: string): CtxConfig {
  configCache = null;
  return loadConfig(basePath);
}

/**
 * Get current loaded configuration (cached)
 */
export function getConfig(): CtxConfig {
  return configCache || loadConfig();
}

/**
 * Deep merge loaded config with default config
 * Ensures all required fields are present even in partial configs
 */
function mergeWithDefaults(loaded: Partial<CtxConfig>): CtxConfig {
  return {
    version: loaded.version || DEFAULT_CONFIG.version,
    description: loaded.description || DEFAULT_CONFIG.description,
    
    retrieval: {
      ...DEFAULT_CONFIG.retrieval,
      ...loaded.retrieval,
      gamma_kind_boost: {
        ...DEFAULT_CONFIG.retrieval.gamma_kind_boost,
        ...loaded.retrieval?.gamma_kind_boost
      },
      fusion: {
        ...DEFAULT_CONFIG.retrieval.fusion!,
        ...loaded.retrieval?.fusion
      },
      llm_rerank: {
        ...DEFAULT_CONFIG.retrieval.llm_rerank!,
        ...loaded.retrieval?.llm_rerank
      }
    },
    
    chunking: {
      ...DEFAULT_CONFIG.chunking,
      ...loaded.chunking
    },
    
    timeouts: {
      ...DEFAULT_CONFIG.timeouts,
      ...loaded.timeouts
    },
    
    features: {
      ...DEFAULT_CONFIG.features!,
      ...loaded.features
    },
    
    query_understanding: {
      ...DEFAULT_CONFIG.query_understanding!,
      ...loaded.query_understanding
    },
    
    ml: {
      prediction_service: {
        ...DEFAULT_CONFIG.ml!.prediction_service!,
        ...loaded.ml?.prediction_service
      },
      models: {
        ...DEFAULT_CONFIG.ml!.models!,
        ...loaded.ml?.models
      }
    },
    
    development: {
      ...DEFAULT_CONFIG.development!,
      ...loaded.development
    }
  };
}

/**
 * Get package root directory (for relative path resolution)
 */
function getPackageRoot(): string {
  // Handle both CommonJS and ESM environments
  try {
    if (typeof __dirname !== 'undefined') {
      return __dirname;
    }
  } catch (e) {
    // __dirname not available in ESM
  }
  
  // Fallback to current working directory
  return process.cwd();
}

/**
 * Validate configuration structure and values
 * Throws descriptive errors for invalid configurations
 */
export function validateConfig(config: CtxConfig): void {
  // Version validation
  if (!config.version) {
    throw new Error('Configuration missing required "version" field');
  }
  
  // Retrieval validation
  if (config.retrieval.alpha < 0 || config.retrieval.alpha > 1) {
    throw new Error('retrieval.alpha must be between 0 and 1');
  }
  
  if (config.retrieval.beta < 0 || config.retrieval.beta > 1) {
    throw new Error('retrieval.beta must be between 0 and 1');
  }
  
  // Chunking validation
  if (config.chunking.target_tokens <= 0) {
    throw new Error('chunking.target_tokens must be positive');
  }
  
  if (config.chunking.overlap < 0 || config.chunking.overlap >= config.chunking.target_tokens) {
    throw new Error('chunking.overlap must be non-negative and less than target_tokens');
  }
  
  // Timeouts validation
  const timeoutFields = ['hyde_ms', 'summarize_ms', 'ollama_connect_ms'] as const;
  for (const field of timeoutFields) {
    if (config.timeouts[field] <= 0) {
      throw new Error(`timeouts.${field} must be positive`);
    }
  }
  
  // ML service validation
  if (config.ml?.prediction_service?.enabled) {
    const service = config.ml.prediction_service;
    if (service.port <= 0 || service.port > 65535) {
      throw new Error('ml.prediction_service.port must be a valid port number');
    }
    
    if (service.timeout_ms <= 0) {
      throw new Error('ml.prediction_service.timeout_ms must be positive');
    }
  }
  
  console.log('✅ Configuration validation passed');
}

/**
 * Create a default config file at the specified path
 * Used by the CLI init command to bootstrap configuration
 */
export function createDefaultConfigFile(targetPath: string): void {
  const configContent = JSON.stringify(DEFAULT_CONFIG, null, 2);
  
  try {
    const { writeFileSync } = require('fs');
    writeFileSync(targetPath, configContent);
    console.log(`📋 Created default configuration at: ${targetPath}`);
  } catch (error) {
    throw new Error(`Failed to create config file at ${targetPath}: ${error}`);
  }
}