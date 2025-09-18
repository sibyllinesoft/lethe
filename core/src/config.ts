/**
 * Configuration and Feature Flag Management System
 * 
 * Provides comprehensive configuration management with:
 * - Dynamic feature flags with type safety
 * - Real-time configuration validation
 * - Environment-aware settings
 * - Circuit breaker patterns for resilience
 * - Performance monitoring and SLA enforcement
 * - Hot-reload capabilities for configuration changes
 */

import { Result } from './types.js';
import { LetheError } from './errors.js';
import { JsonSchemaValidator } from './validation.js';

// Core configuration types
export interface LetheConfig {
  // System-wide settings
  system: {
    version: string;
    environment: 'development' | 'testing' | 'staging' | 'production';
    logLevel: 'debug' | 'info' | 'warn' | 'error';
    enableMetrics: boolean;
    maxConcurrentRequests: number;
    requestTimeoutMs: number;
    memoryLimitMB: number;
  };

  // Performance SLA targets
  performance: {
    targetP50Ms: number;
    targetP95Ms: number;
    maxMemoryMB: number;
    circuitBreakerThreshold: number;
    adaptiveScaling: boolean;
  };

  // Retrieval configuration
  retrieval: {
    defaultStrategy: 'window' | 'bm25' | 'vector' | 'hybrid';
    windowSize: number;
    overlapRatio: number;
    bm25Config: {
      k1: number;
      b: number;
      epsilon: number;
    };
    vectorConfig: {
      model: string;
      dimensions: number;
      similarity: 'cosine' | 'dot' | 'euclidean';
    };
    hybridConfig: {
      bm25Weight: number;
      vectorWeight: number;
      adaptiveWeights: boolean;
    };
  };

  // Chunking configuration
  chunking: {
    defaultStrategy: 'ast' | 'hierarchical' | 'propositional';
    maxChunkSize: number;
    minChunkSize: number;
    overlapTokens: number;
    preserveStructure: boolean;
    astConfig: {
      supportedLanguages: string[];
      preserveComments: boolean;
      includeMetadata: boolean;
    };
    hierarchicalConfig: {
      maxDepth: number;
      sectionHeaders: string[];
      semanticBoundaries: boolean;
    };
    propositionalConfig: {
      extractRelations: boolean;
      logicalConnectors: string[];
      confidenceThreshold: number;
    };
  };

  // Ranking and diversification
  ranking: {
    enableMetadataBoost: boolean;
    diversificationStrategy: 'entity' | 'semantic' | 'hybrid';
    maxDiversityRatio: number;
    rerankingTopK: number;
    fusionStrategy: 'rrf' | 'weighted' | 'learned';
  };

  // LLM integration
  llm: {
    provider: 'openai' | 'anthropic' | 'huggingface' | 'local';
    model: string;
    maxTokens: number;
    temperature: number;
    enableReranking: boolean;
    enableContradictionDetection: boolean;
    batchSize: number;
    rateLimitRpm: number;
  };

  // Feature flags
  features: FeatureFlags;
}

export interface FeatureFlags {
  // Core features
  enableAdvancedRetrieval: boolean;
  enableSemanticDiversification: boolean;
  enableLLMReranking: boolean;
  enableContradictionDetection: boolean;
  enableAdaptiveWeights: boolean;
  
  // Performance features
  enableCaching: boolean;
  enablePrefetching: boolean;
  enableBatching: boolean;
  enableCircuitBreaker: boolean;
  enableAutoScaling: boolean;
  
  // Experimental features
  enableMultiQueryGeneration: boolean;
  enableQueryDecomposition: boolean;
  enableIterativeRefinement: boolean;
  enableFeedbackLearning: boolean;
  enableCrossEncoderFusion: boolean;
  
  // Debug features
  enableDetailedLogging: boolean;
  enablePerformanceTracing: boolean;
  enableMemoryProfiling: boolean;
  enableConfigValidation: boolean;
  
  // Rollout controls
  rolloutPercentage: Record<string, number>;
  userSegments: Record<string, string[]>;
  environmentOverrides: Record<string, boolean>;
}

// Configuration schemas for validation
export const CONFIG_SCHEMAS = {
  system: {
    type: 'object',
    required: ['version', 'environment', 'logLevel'],
    properties: {
      version: { type: 'string', pattern: '^\\d+\\.\\d+\\.\\d+$' },
      environment: { type: 'string', enum: ['development', 'testing', 'staging', 'production'] },
      logLevel: { type: 'string', enum: ['debug', 'info', 'warn', 'error'] },
      enableMetrics: { type: 'boolean' },
      maxConcurrentRequests: { type: 'number', minimum: 1, maximum: 1000 },
      requestTimeoutMs: { type: 'number', minimum: 1000, maximum: 60000 },
      memoryLimitMB: { type: 'number', minimum: 512, maximum: 16384 }
    }
  },
  performance: {
    type: 'object',
    required: ['targetP50Ms', 'targetP95Ms', 'maxMemoryMB'],
    properties: {
      targetP50Ms: { type: 'number', minimum: 100, maximum: 10000 },
      targetP95Ms: { type: 'number', minimum: 200, maximum: 30000 },
      maxMemoryMB: { type: 'number', minimum: 512, maximum: 16384 },
      circuitBreakerThreshold: { type: 'number', minimum: 0.1, maximum: 1.0 },
      adaptiveScaling: { type: 'boolean' }
    }
  }
} as const;

// Feature flag evaluation context
export interface FeatureFlagContext {
  userId?: string;
  sessionId: string;
  environment: string;
  version: string;
  userSegment?: string;
  customAttributes?: Record<string, unknown>;
}

// Configuration change event
export interface ConfigChangeEvent {
  timestamp: Date;
  field: string;
  oldValue: unknown;
  newValue: unknown;
  source: 'manual' | 'automatic' | 'rollout';
  appliedBy?: string;
}

/**
 * Configuration Manager with feature flag support
 */
export class ConfigurationManager {
  private config: LetheConfig;
  private validator: JsonSchemaValidator;
  private changeListeners: Set<(event: ConfigChangeEvent) => void> = new Set();
  private configHistory: ConfigChangeEvent[] = [];
  private lastValidation: Date = new Date();
  private hotReloadEnabled: boolean = false;

  constructor(initialConfig: Partial<LetheConfig> = {}) {
    this.validator = new JsonSchemaValidator();
    this.config = this.mergeWithDefaults(initialConfig);
    
    // Validate initial configuration
    const validation = this.validateConfiguration(this.config);
    if (!validation.success) {
      throw new Error(`Invalid initial configuration: ${validation.error.message}`);
    }
  }

  /**
   * Get complete configuration
   */
  getConfig(): LetheConfig {
    return { ...this.config };
  }

  /**
   * Get configuration section
   */
  getSection<K extends keyof LetheConfig>(section: K): LetheConfig[K] {
    return { ...this.config[section] };
  }

  /**
   * Update configuration with validation
   */
  async updateConfig(updates: Partial<LetheConfig>): Promise<Result<void, LetheError>> {
    try {
      // Create merged config for validation
      const mergedConfig = this.deepMerge(this.config, updates);
      
      // Validate the merged configuration
      const validation = this.validateConfiguration(mergedConfig);
      if (!validation.success) {
        return {
          success: false,
          error: new LetheError(
            'CONFIGURATION_VALIDATION_ERROR',
            `Configuration validation failed: ${validation.error.message}`,
            { validation: validation.error }
          )
        };
      }

      // Track changes
      const changes = this.detectChanges(this.config, mergedConfig);
      
      // Apply changes
      this.config = mergedConfig;
      this.lastValidation = new Date();

      // Notify listeners
      for (const change of changes) {
        const event: ConfigChangeEvent = {
          ...change,
          timestamp: new Date(),
          source: 'manual'
        };
        this.configHistory.push(event);
        this.notifyListeners(event);
      }

      return { success: true, data: undefined };
    } catch (error) {
      return {
        success: false,
        error: new LetheError(
          'CONFIGURATION_UPDATE_ERROR',
          'Failed to update configuration',
          { originalError: error }
        )
      };
    }
  }

  /**
   * Evaluate feature flag with context
   */
  evaluateFeature(
    featureName: keyof FeatureFlags,
    context: FeatureFlagContext
  ): boolean {
    const featureValue = this.config.features[featureName];
    
    // Handle boolean flags
    if (typeof featureValue === 'boolean') {
      return this.applyRolloutLogic(featureName, featureValue, context);
    }

    // Handle complex flags (objects)
    if (typeof featureValue === 'object' && featureValue !== null) {
      // Apply environment overrides
      const envOverride = this.config.features.environmentOverrides[featureName];
      if (envOverride !== undefined) {
        return this.applyRolloutLogic(featureName, envOverride, context);
      }
    }

    return false;
  }

  /**
   * Get all active features for context
   */
  getActiveFeatures(context: FeatureFlagContext): Record<string, boolean> {
    const activeFeatures: Record<string, boolean> = {};
    
    for (const [featureName, _] of Object.entries(this.config.features)) {
      if (featureName === 'rolloutPercentage' || 
          featureName === 'userSegments' || 
          featureName === 'environmentOverrides') {
        continue;
      }
      
      activeFeatures[featureName] = this.evaluateFeature(
        featureName as keyof FeatureFlags, 
        context
      );
    }
    
    return activeFeatures;
  }

  /**
   * Apply rollout logic (percentage-based, segment-based)
   */
  private applyRolloutLogic(
    featureName: keyof FeatureFlags,
    baseValue: boolean,
    context: FeatureFlagContext
  ): boolean {
    if (!baseValue) {
      return false;
    }

    // Check user segment eligibility
    if (context.userSegment) {
      const segmentUsers = this.config.features.userSegments[featureName];
      if (segmentUsers && !segmentUsers.includes(context.userSegment)) {
        return false;
      }
    }

    // Apply percentage rollout
    const rolloutPercentage = this.config.features.rolloutPercentage[featureName] ?? 100;
    if (rolloutPercentage < 100) {
      const hash = this.hashString(context.sessionId + featureName);
      const userPercentile = hash % 100;
      return userPercentile < rolloutPercentage;
    }

    return true;
  }

  /**
   * Validate entire configuration
   */
  private validateConfiguration(config: LetheConfig): Result<void, Error> {
    try {
      // System validation
      const systemValidation = this.validator.validate(config.system, CONFIG_SCHEMAS.system);
      if (!systemValidation.success) {
        return { success: false, error: new Error(`System config invalid: ${systemValidation.error.message}`) };
      }

      // Performance validation
      const perfValidation = this.validator.validate(config.performance, CONFIG_SCHEMAS.performance);
      if (!perfValidation.success) {
        return { success: false, error: new Error(`Performance config invalid: ${perfValidation.error.message}`) };
      }

      // Cross-field validation
      if (config.performance.targetP50Ms >= config.performance.targetP95Ms) {
        return { success: false, error: new Error('P50 must be less than P95') };
      }

      if (config.chunking.minChunkSize >= config.chunking.maxChunkSize) {
        return { success: false, error: new Error('Min chunk size must be less than max chunk size') };
      }

      if (config.retrieval.hybridConfig.bm25Weight + config.retrieval.hybridConfig.vectorWeight !== 1.0) {
        return { success: false, error: new Error('Hybrid weights must sum to 1.0') };
      }

      return { success: true, data: undefined };
    } catch (error) {
      return { success: false, error: error as Error };
    }
  }

  /**
   * Deep merge configurations
   */
  private deepMerge(target: any, source: any): any {
    const result = { ...target };
    
    for (const key in source) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        result[key] = this.deepMerge(result[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }
    
    return result;
  }

  /**
   * Detect configuration changes
   */
  private detectChanges(oldConfig: LetheConfig, newConfig: LetheConfig): Array<{
    field: string;
    oldValue: unknown;
    newValue: unknown;
  }> {
    const changes: Array<{ field: string; oldValue: unknown; newValue: unknown }> = [];
    
    const compareObjects = (old: any, updated: any, path: string = '') => {
      for (const key in updated) {
        const fullPath = path ? `${path}.${key}` : key;
        
        if (typeof updated[key] === 'object' && updated[key] !== null && !Array.isArray(updated[key])) {
          compareObjects(old[key] || {}, updated[key], fullPath);
        } else if (old[key] !== updated[key]) {
          changes.push({
            field: fullPath,
            oldValue: old[key],
            newValue: updated[key]
          });
        }
      }
    };
    
    compareObjects(oldConfig, newConfig);
    return changes;
  }

  /**
   * Merge with default configuration
   */
  private mergeWithDefaults(config: Partial<LetheConfig>): LetheConfig {
    const defaults: LetheConfig = {
      system: {
        version: '1.0.0',
        environment: 'development',
        logLevel: 'info',
        enableMetrics: true,
        maxConcurrentRequests: 100,
        requestTimeoutMs: 30000,
        memoryLimitMB: 2048
      },
      performance: {
        targetP50Ms: 3000,
        targetP95Ms: 6000,
        maxMemoryMB: 1536,
        circuitBreakerThreshold: 0.5,
        adaptiveScaling: true
      },
      retrieval: {
        defaultStrategy: 'hybrid',
        windowSize: 512,
        overlapRatio: 0.1,
        bm25Config: {
          k1: 1.2,
          b: 0.75,
          epsilon: 0.25
        },
        vectorConfig: {
          model: 'sentence-transformers/all-MiniLM-L6-v2',
          dimensions: 384,
          similarity: 'cosine'
        },
        hybridConfig: {
          bm25Weight: 0.4,
          vectorWeight: 0.6,
          adaptiveWeights: true
        }
      },
      chunking: {
        defaultStrategy: 'ast',
        maxChunkSize: 1024,
        minChunkSize: 128,
        overlapTokens: 50,
        preserveStructure: true,
        astConfig: {
          supportedLanguages: ['typescript', 'javascript', 'python', 'rust', 'go'],
          preserveComments: true,
          includeMetadata: true
        },
        hierarchicalConfig: {
          maxDepth: 5,
          sectionHeaders: ['#', '##', '###', '####', '#####'],
          semanticBoundaries: true
        },
        propositionalConfig: {
          extractRelations: true,
          logicalConnectors: ['and', 'or', 'but', 'however', 'therefore', 'because'],
          confidenceThreshold: 0.7
        }
      },
      ranking: {
        enableMetadataBoost: true,
        diversificationStrategy: 'semantic',
        maxDiversityRatio: 0.3,
        rerankingTopK: 20,
        fusionStrategy: 'rrf'
      },
      llm: {
        provider: 'openai',
        model: 'gpt-3.5-turbo',
        maxTokens: 4096,
        temperature: 0.1,
        enableReranking: true,
        enableContradictionDetection: true,
        batchSize: 10,
        rateLimitRpm: 60
      },
      features: {
        // Core features
        enableAdvancedRetrieval: true,
        enableSemanticDiversification: true,
        enableLLMReranking: true,
        enableContradictionDetection: true,
        enableAdaptiveWeights: true,
        
        // Performance features
        enableCaching: true,
        enablePrefetching: false,
        enableBatching: true,
        enableCircuitBreaker: true,
        enableAutoScaling: false,
        
        // Experimental features
        enableMultiQueryGeneration: false,
        enableQueryDecomposition: false,
        enableIterativeRefinement: false,
        enableFeedbackLearning: false,
        enableCrossEncoderFusion: false,
        
        // Debug features
        enableDetailedLogging: false,
        enablePerformanceTracing: false,
        enableMemoryProfiling: false,
        enableConfigValidation: true,
        
        // Rollout controls
        rolloutPercentage: {},
        userSegments: {},
        environmentOverrides: {}
      }
    };

    return this.deepMerge(defaults, config);
  }

  /**
   * Simple hash function for consistent percentage-based rollouts
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Add configuration change listener
   */
  addChangeListener(listener: (event: ConfigChangeEvent) => void): void {
    this.changeListeners.add(listener);
  }

  /**
   * Remove configuration change listener
   */
  removeChangeListener(listener: (event: ConfigChangeEvent) => void): void {
    this.changeListeners.delete(listener);
  }

  /**
   * Notify all listeners of configuration changes
   */
  private notifyListeners(event: ConfigChangeEvent): void {
    this.changeListeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in config change listener:', error);
      }
    });
  }

  /**
   * Get configuration change history
   */
  getChangeHistory(limit: number = 100): ConfigChangeEvent[] {
    return this.configHistory.slice(-limit);
  }

  /**
   * Enable hot reload of configuration
   */
  enableHotReload(): void {
    this.hotReloadEnabled = true;
  }

  /**
   * Disable hot reload of configuration
   */
  disableHotReload(): void {
    this.hotReloadEnabled = false;
  }

  /**
   * Export configuration to JSON
   */
  exportConfig(): string {
    return JSON.stringify(this.config, null, 2);
  }

  /**
   * Import configuration from JSON
   */
  async importConfig(configJson: string): Promise<Result<void, LetheError>> {
    try {
      const importedConfig = JSON.parse(configJson);
      return await this.updateConfig(importedConfig);
    } catch (error) {
      return {
        success: false,
        error: new LetheError(
          'CONFIGURATION_IMPORT_ERROR',
          'Failed to import configuration',
          { originalError: error }
        )
      };
    }
  }
}

/**
 * Global configuration manager instance
 */
let globalConfigManager: ConfigurationManager | null = null;

/**
 * Initialize global configuration manager
 */
export function initializeConfig(config?: Partial<LetheConfig>): ConfigurationManager {
  if (globalConfigManager) {
    console.warn('Configuration manager already initialized. Use getConfigManager() instead.');
    return globalConfigManager;
  }
  
  globalConfigManager = new ConfigurationManager(config);
  return globalConfigManager;
}

/**
 * Get global configuration manager
 */
export function getConfigManager(): ConfigurationManager {
  if (!globalConfigManager) {
    throw new Error('Configuration manager not initialized. Call initializeConfig() first.');
  }
  return globalConfigManager;
}

/**
 * Convenience function to evaluate feature flags
 */
export function isFeatureEnabled(
  featureName: keyof FeatureFlags,
  context: FeatureFlagContext
): boolean {
  return getConfigManager().evaluateFeature(featureName, context);
}

/**
 * Convenience function to get configuration section
 */
export function getConfigSection<K extends keyof LetheConfig>(section: K): LetheConfig[K] {
  return getConfigManager().getSection(section);
}