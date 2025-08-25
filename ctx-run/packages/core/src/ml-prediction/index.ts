/**
 * Iteration 3: ML Prediction Integration - HTTP Service Version
 * 
 * This module provides TypeScript integration with HTTP-based ML service for:
 * 1. Dynamic alpha/beta parameter prediction
 * 2. Learned plan selection
 * 3. Fast feature extraction and caching
 */

import { join } from 'path';

export interface QueryFeatures {
  query_len: number;
  has_code_symbol: boolean;
  has_error_token: boolean;
  has_path_or_file: boolean;
  has_numeric_id: boolean;
  bm25_top1: number;
  ann_top1: number;
  overlap_ratio: number;
  hyde_k: number;
  word_count: number;
  char_count: number;
  complexity_score: number;
}

export interface MLPrediction {
  alpha: number;
  beta: number;
  plan: 'explore' | 'verify' | 'exploit';
  prediction_time_ms: number;
  model_loaded: boolean;
  features?: QueryFeatures;
}

export interface MLConfig {
  fusion_dynamic: boolean;
  plan_learned: boolean;
  models_dir?: string;
  fallback_alpha: number;
  fallback_beta: number;
  fallback_plan: 'explore' | 'verify' | 'exploit';
  prediction_timeout_ms: number;
  // HTTP service configuration
  service_url?: string;
  service_enabled?: boolean;
}

export class MLPredictor {
  private config: MLConfig;
  private isInitialized = false;
  private predictionCache = new Map<string, MLPrediction>();
  private lastLoadTime = 0;
  private serviceHealthy = false;

  constructor(config: Partial<MLConfig> = {}) {
    this.config = {
      fusion_dynamic: config.fusion_dynamic ?? false,
      plan_learned: config.plan_learned ?? false,
      models_dir: config.models_dir ?? 'models',
      fallback_alpha: config.fallback_alpha ?? 0.7,
      fallback_beta: config.fallback_beta ?? 0.5,
      fallback_plan: config.fallback_plan ?? 'exploit',
      prediction_timeout_ms: config.prediction_timeout_ms ?? 2000,
      service_url: config.service_url ?? 'http://127.0.0.1:8080',
      service_enabled: config.service_enabled ?? true
    };
  }

  /**
   * Initialize the ML predictor by checking service health
   */
  async initialize(): Promise<boolean> {
    if (this.isInitialized) {
      return this.serviceHealthy;
    }

    if (!this.config.fusion_dynamic && !this.config.plan_learned) {
      console.log('ML prediction disabled in config');
      this.isInitialized = true;
      this.serviceHealthy = false;
      return true;
    }

    if (!this.config.service_enabled) {
      console.log('ML service disabled, using fallback predictions');
      this.isInitialized = true;
      this.serviceHealthy = false;
      return true;
    }

    try {
      const startTime = Date.now();
      
      // Check service health
      const healthOk = await this.checkServiceHealth();
      
      this.lastLoadTime = Date.now() - startTime;
      
      if (healthOk) {
        console.log(`ML service healthy, initialized in ${this.lastLoadTime}ms`);
        this.isInitialized = true;
        this.serviceHealthy = true;
        return true;
      } else {
        console.warn('ML service not available, using fallbacks');
        this.isInitialized = true;
        this.serviceHealthy = false;
        return false;
      }
    } catch (error) {
      console.warn(`ML service initialization failed: ${error}, using fallbacks`);
      this.isInitialized = true;
      this.serviceHealthy = false;
      return false;
    }
  }

  /**
   * Predict optimal parameters for a query
   */
  async predictParameters(
    query: string, 
    context: {
      bm25_top1?: number;
      ann_top1?: number;
      overlap_ratio?: number;
      hyde_k?: number;
      contradictions?: number;
    } = {}
  ): Promise<MLPrediction> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    // Check cache first
    const cacheKey = this.generateCacheKey(query, context);
    if (this.predictionCache.has(cacheKey)) {
      return this.predictionCache.get(cacheKey)!;
    }

    try {
      const prediction = await this.predictParametersInternal(query, context);
      
      // Cache successful predictions
      this.predictionCache.set(cacheKey, prediction);
      
      return prediction;
    } catch (error) {
      console.warn(`ML prediction failed: ${error}, using fallback`);
      
      const fallback: MLPrediction = {
        alpha: this.config.fallback_alpha,
        beta: this.config.fallback_beta,
        plan: this.config.fallback_plan,
        prediction_time_ms: 0,
        model_loaded: false
      };
      
      return fallback;
    }
  }

  /**
   * Internal prediction using HTTP service
   */
  private async predictParametersInternal(
    query: string, 
    context: any
  ): Promise<MLPrediction> {
    if (!this.serviceHealthy) {
      // Return fallback prediction if service is not healthy
      return {
        alpha: this.config.fallback_alpha,
        beta: this.config.fallback_beta,
        plan: this.config.fallback_plan,
        prediction_time_ms: 0,
        model_loaded: false
      };
    }

    const startTime = Date.now();
    
    try {
      // Make HTTP request to prediction service
      const response = await this.makeHttpRequest('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          context
        }),
        timeout: this.config.prediction_timeout_ms
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      const totalTime = Date.now() - startTime;

      const prediction: MLPrediction = {
        alpha: this.config.fusion_dynamic ? result.alpha : this.config.fallback_alpha,
        beta: this.config.fusion_dynamic ? result.beta : this.config.fallback_beta,
        plan: this.config.plan_learned ? result.plan : this.config.fallback_plan,
        prediction_time_ms: totalTime,
        model_loaded: result.model_loaded ?? true,
        features: result.features
      };

      return prediction;

    } catch (error) {
      console.warn(`HTTP prediction failed: ${error}, using fallback`);
      
      return {
        alpha: this.config.fallback_alpha,
        beta: this.config.fallback_beta,
        plan: this.config.fallback_plan,
        prediction_time_ms: Date.now() - startTime,
        model_loaded: false
      };
    }
  }

  /**
   * Check if the ML service is healthy and available
   */
  private async checkServiceHealth(): Promise<boolean> {
    try {
      const response = await this.makeHttpRequest('/health', {
        method: 'GET',
        timeout: 5000
      });

      if (!response.ok) {
        return false;
      }

      const health = await response.json();
      return health.status === 'healthy';

    } catch (error) {
      console.debug(`Service health check failed: ${error}`);
      return false;
    }
  }

  /**
   * Make HTTP request to the ML service
   */
  private async makeHttpRequest(
    path: string, 
    options: {
      method: string;
      headers?: Record<string, string>;
      body?: string;
      timeout: number;
    }
  ): Promise<Response> {
    const url = `${this.config.service_url}${path}`;
    
    // Use fetch with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout);

    try {
      const response = await fetch(url, {
        method: options.method,
        headers: options.headers,
        body: options.body,
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      return response;

    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${options.timeout}ms`);
      }
      throw error;
    }
  }

  /**
   * Generate cache key for predictions
   */
  private generateCacheKey(query: string, context: any): string {
    const contextStr = JSON.stringify(context);
    return `${query.slice(0, 50)}:${contextStr}`;
  }

  /**
   * Clear prediction cache
   */
  clearCache(): void {
    this.predictionCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; lastLoadTime: number; serviceHealthy: boolean } {
    return {
      size: this.predictionCache.size,
      lastLoadTime: this.lastLoadTime,
      serviceHealthy: this.serviceHealthy
    };
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    this.predictionCache.clear();
    this.isInitialized = false;
    this.serviceHealthy = false;
  }
}

/**
 * Extract query features without ML (fallback for TypeScript-only feature extraction)
 */
export function extractBasicFeatures(query: string): Partial<QueryFeatures> {
  const queryLower = query.toLowerCase();
  
  // Basic text features
  const queryLen = query.length;
  const wordCount = query.split(/\s+/).length;
  const charCount = query.length;

  // Pattern-based features (simplified regex)
  const hasCodeSymbol = /[_a-zA-Z][\w]*\(|\b[A-Z][A-Za-z0-9]+::[A-Za-z0-9]+\b|(class|function|def|var|let|const|import|export)/.test(query);
  const hasErrorToken = /(exception|error|stack\s+trace|errno|\bE\d{2,}\b|failed|crashed|broken|bug|undefined|null|nan)/i.test(query);
  const hasPathOrFile = /\/[^\s]+\.[a-zA-Z0-9]+|[A-Za-z]:\\[^\s]+\.[a-zA-Z0-9]+|\b\w+\.(js|py|ts|java|cpp|h|css|html)\b/.test(query);
  const hasNumericId = /\b\d{3,}\b/.test(query);

  // Complexity score (simplified)
  let complexityScore = 0;
  if (wordCount > 10) complexityScore += 0.3;
  else if (wordCount > 5) complexityScore += 0.15;
  if (hasCodeSymbol) complexityScore += 0.25;
  if (hasErrorToken) complexityScore += 0.2;
  if (hasPathOrFile) complexityScore += 0.15;
  if (/\?/.test(query)) complexityScore += 0.1;
  if (/(how|why|what|when|where)/i.test(query)) complexityScore += 0.1;
  complexityScore = Math.min(complexityScore, 1.0);

  return {
    query_len: queryLen,
    has_code_symbol: hasCodeSymbol,
    has_error_token: hasErrorToken,
    has_path_or_file: hasPathOrFile,
    has_numeric_id: hasNumericId,
    word_count: wordCount,
    char_count: charCount,
    complexity_score: complexityScore
  };
}

// Create singleton instance for reuse
let globalMLPredictor: MLPredictor | null = null;

export function getMLPredictor(config?: Partial<MLConfig>): MLPredictor {
  if (!globalMLPredictor) {
    globalMLPredictor = new MLPredictor(config);
  }
  return globalMLPredictor;
}

export function initializeMLPrediction(config: Partial<MLConfig> = {}): Promise<boolean> {
  const predictor = getMLPredictor(config);
  return predictor.initialize();
}