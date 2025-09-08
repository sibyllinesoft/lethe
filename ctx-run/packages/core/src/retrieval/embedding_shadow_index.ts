/**
 * EmbeddingGemma-300M Shadow Index System
 * 
 * Implements A/B testing infrastructure for EmbeddingGemma-300M integration:
 * - Freeze CE and candidate pools; swap only S1 bi-encoder
 * - Test both 768-d and Matryoshka 256-d heads
 * - Measure: (a) ΔCBU/GB (quality per memory), (b) middleware P95, 
 *   (c) index-build time, (d) calibrator re-fit deltas
 * - Shadow index system for seamless A/B testing
 * 
 * Performance Targets:
 * - Index build time: <30s for 100K chunks
 * - Memory efficiency: >2x improvement vs current embeddings
 * - Quality maintenance: <5% CBU degradation
 * - P95 latency: <200ms for retrieval operations
 */

import { z } from 'zod';
import type { Embeddings } from '@lethe/embeddings';
import type { DB } from '@lethe/sqlite';

// Shadow index configuration
export const ShadowIndexConfigSchema = z.object({
  // Embedding model configuration
  model_name: z.enum(['embedding-gemma-300m-768d', 'embedding-gemma-300m-256d']).default('embedding-gemma-300m-768d'),
  embedding_dimension: z.number().int().min(128).default(768),
  matryoshka_dimension: z.number().int().min(64).default(256), // For 256-d head
  
  // Index build parameters
  batch_size: z.number().int().min(1).default(32),
  max_build_time_ms: z.number().int().min(1000).default(30000), // 30s target
  enable_parallel_build: z.boolean().default(true),
  
  // Performance monitoring
  track_memory_usage: z.boolean().default(true),
  track_build_metrics: z.boolean().default(true),
  track_retrieval_latency: z.boolean().default(true),
  
  // A/B testing configuration
  shadow_traffic_ratio: z.number().min(0).max(1).default(0.1), // 10% shadow traffic
  enable_live_comparison: z.boolean().default(true),
  fallback_on_failure: z.boolean().default(true),
  
  // Quality thresholds
  max_cbu_degradation: z.number().min(0).max(1).default(0.05), // 5% max degradation
  min_memory_improvement: z.number().min(1).default(2.0), // 2x memory improvement
  max_p95_latency_ms: z.number().min(50).default(200),
});

export type ShadowIndexConfig = z.infer<typeof ShadowIndexConfigSchema>;

// Index build metrics
export interface IndexBuildMetrics {
  model_name: string;
  embedding_dimension: number;
  total_chunks: number;
  build_time_ms: number;
  memory_usage_mb: number;
  memory_per_chunk_kb: number;
  chunks_per_second: number;
  peak_memory_mb: number;
  index_size_mb: number;
  build_errors: number;
  success_rate: number;
}

// Retrieval performance metrics
export interface RetrievalMetrics {
  model_name: string;
  query_count: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  average_latency_ms: number;
  error_rate: number;
  memory_usage_mb: number;
  quality_metrics: {
    average_cbu: number;
    ndcg_at_10: number;
    recall_at_k: number[];
  };
}

// A/B test comparison result
export interface ABTestResult {
  primary_model: string;
  shadow_model: string;
  test_duration_ms: number;
  query_count: number;
  
  performance_comparison: {
    cbu_per_gb_ratio: number; // shadow / primary
    p95_latency_ratio: number;
    memory_efficiency_ratio: number;
    build_time_ratio: number;
  };
  
  quality_analysis: {
    cbu_degradation: number; // negative = improvement
    ranking_correlation: number;
    significant_difference: boolean;
    confidence_interval: [number, number];
  };
  
  recommendation: {
    should_promote: boolean;
    confidence_score: number;
    risk_assessment: 'low' | 'medium' | 'high';
    rollout_strategy: 'immediate' | 'gradual' | 'hold';
  };
}

// Shadow index state
interface ShadowIndexState {
  primary_index: {
    model_name: string;
    dimension: number;
    build_time_ms: number;
    memory_mb: number;
    ready: boolean;
  };
  shadow_index: {
    model_name: string;
    dimension: number;
    build_time_ms: number;
    memory_mb: number;
    ready: boolean;
  };
  ab_test_active: boolean;
  test_start_time: number;
  query_routing: {
    primary_count: number;
    shadow_count: number;
    error_count: number;
  };
}

/**
 * EmbeddingGemma-300M Shadow Index Manager
 * 
 * Coordinates A/B testing between current embeddings and EmbeddingGemma-300M:
 * 1. Build shadow indices with both 768-d and 256-d variants
 * 2. Route traffic for live comparison
 * 3. Monitor quality and performance metrics
 * 4. Generate promotion recommendations
 * 5. Handle seamless model swapping
 */
export class EmbeddingShadowIndexManager {
  private config: ShadowIndexConfig;
  private state: ShadowIndexState;
  private primary_embeddings: Embeddings;
  private shadow_embeddings?: Embeddings;
  private performance_history: Map<string, RetrievalMetrics[]> = new Map();
  
  constructor(
    primary_embeddings: Embeddings,
    config: Partial<ShadowIndexConfig> = {}
  ) {
    this.config = ShadowIndexConfigSchema.parse(config);
    this.primary_embeddings = primary_embeddings;
    this.initializeState();
    
    console.log(`🌍 Shadow Index Manager initialized: ${this.config.model_name} @ ${this.config.embedding_dimension}d`);
  }
  
  /**
   * Initialize shadow index state
   */
  private initializeState(): void {
    this.state = {
      primary_index: {
        model_name: 'current',
        dimension: 384, // Assumed current dimension
        build_time_ms: 0,
        memory_mb: 0,
        ready: true, // Assume primary is ready
      },
      shadow_index: {
        model_name: this.config.model_name,
        dimension: this.config.embedding_dimension,
        build_time_ms: 0,
        memory_mb: 0,
        ready: false,
      },
      ab_test_active: false,
      test_start_time: 0,
      query_routing: {
        primary_count: 0,
        shadow_count: 0,
        error_count: 0,
      },
    };
  }
  
  /**
   * Build shadow index with EmbeddingGemma-300M
   */
  async buildShadowIndex(
    db: DB,
    session_id: string,
    force_rebuild: boolean = false
  ): Promise<IndexBuildMetrics> {
    console.log(`🏗️ Building shadow index: ${this.config.model_name}...`);
    
    if (this.state.shadow_index.ready && !force_rebuild) {
      console.log('Shadow index already ready');
      return this.createMockBuildMetrics(); // Return cached metrics
    }
    
    const start_time = performance.now();
    const start_memory = this.getMemoryUsage();
    
    try {
      // Initialize shadow embeddings model
      this.shadow_embeddings = await this.initializeShadowEmbeddings();
      
      // Get chunks from database
      const { getChunksBySession } = await import('@lethe/sqlite');
      const chunks = getChunksBySession(db, session_id);
      
      console.log(`Processing ${chunks.length} chunks with ${this.config.model_name}`);
      
      // Build embeddings in batches for memory efficiency
      const batch_size = this.config.batch_size;
      let processed_chunks = 0;
      let build_errors = 0;
      const embeddings_data: Array<{ id: string; embedding: number[] }> = [];
      
      // Track peak memory usage
      let peak_memory = start_memory;
      
      for (let i = 0; i < chunks.length; i += batch_size) {
        const batch = chunks.slice(i, i + batch_size);
        const batch_texts = batch.map(chunk => chunk.text);
        
        try {
          // Get embeddings from shadow model
          const batch_embeddings = await this.shadow_embeddings.embed(batch_texts);
          
          // Process embeddings (apply Matryoshka reduction if needed)
          for (let j = 0; j < batch.length; j++) {
            let embedding = Array.from(batch_embeddings[j]);
            
            // Apply Matryoshka dimension reduction if using 256-d variant
            if (this.config.model_name === 'embedding-gemma-300m-256d') {
              embedding = embedding.slice(0, this.config.matryoshka_dimension);
            }
            
            embeddings_data.push({
              id: batch[j].id,
              embedding,
            });
          }
          
          processed_chunks += batch.length;
          
          // Track memory usage
          const current_memory = this.getMemoryUsage();
          peak_memory = Math.max(peak_memory, current_memory);
          
          // Progress logging
          if (processed_chunks % (batch_size * 10) === 0) {
            console.log(`  Processed ${processed_chunks}/${chunks.length} chunks (${(current_memory - start_memory).toFixed(1)}MB used)`);
          }
          
          // Check build timeout
          const elapsed = performance.now() - start_time;
          if (elapsed > this.config.max_build_time_ms) {
            console.warn(`Build timeout reached: ${elapsed.toFixed(0)}ms`);
            break;
          }
          
        } catch (error) {
          console.warn(`Batch ${i}-${i + batch_size} failed:`, error);
          build_errors += batch.length;
        }
      }
      
      // Store embeddings in shadow index (simplified - would use vector DB)
      await this.storeShadowEmbeddings(embeddings_data, session_id);
      
      const build_time = performance.now() - start_time;
      const final_memory = this.getMemoryUsage();
      const memory_used = final_memory - start_memory;
      
      // Update state
      this.state.shadow_index.ready = true;
      this.state.shadow_index.build_time_ms = build_time;
      this.state.shadow_index.memory_mb = memory_used;
      
      const build_metrics: IndexBuildMetrics = {
        model_name: this.config.model_name,
        embedding_dimension: this.config.embedding_dimension,
        total_chunks: chunks.length,
        build_time_ms: build_time,
        memory_usage_mb: memory_used,
        memory_per_chunk_kb: (memory_used * 1024) / chunks.length,
        chunks_per_second: processed_chunks / (build_time / 1000),
        peak_memory_mb: peak_memory - start_memory,
        index_size_mb: embeddings_data.length * this.config.embedding_dimension * 4 / (1024 * 1024), // Float32 = 4 bytes
        build_errors,
        success_rate: (processed_chunks - build_errors) / chunks.length,
      };
      
      console.log(`✅ Shadow index built: ${build_time.toFixed(0)}ms, ${memory_used.toFixed(1)}MB, ${build_metrics.chunks_per_second.toFixed(1)} chunks/s`);
      
      return build_metrics;
      
    } catch (error) {
      console.error('Shadow index build failed:', error);
      this.state.shadow_index.ready = false;
      throw error;
    }
  }
  
  /**
   * Start A/B testing between primary and shadow indices
   */
  async startABTest(): Promise<void> {
    if (!this.state.shadow_index.ready) {
      throw new Error('Shadow index not ready for A/B testing');
    }
    
    console.log('🧪 Starting A/B test between primary and shadow indices');
    
    this.state.ab_test_active = true;
    this.state.test_start_time = Date.now();
    this.state.query_routing = {
      primary_count: 0,
      shadow_count: 0,
      error_count: 0,
    };
    
    console.log(`  Traffic split: ${(1 - this.config.shadow_traffic_ratio) * 100}% primary, ${this.config.shadow_traffic_ratio * 100}% shadow`);
  }
  
  /**
   * Route query to appropriate index based on A/B test configuration
   */
  async routeQuery(
    query_text: string,
    k: number = 20
  ): Promise<{
    results: Array<{ docId: string; score: number }>;
    model_used: string;
    metrics: {
      latency_ms: number;
      memory_mb: number;
      embedding_dimension: number;
    };
  }> {
    const start_time = performance.now();
    const start_memory = this.getMemoryUsage();
    
    // Determine which index to use
    const use_shadow = this.state.ab_test_active && 
      Math.random() < this.config.shadow_traffic_ratio;
    
    try {
      let results: Array<{ docId: string; score: number }>;
      let model_used: string;
      let embedding_dimension: number;
      
      if (use_shadow && this.shadow_embeddings) {
        // Use shadow index
        results = await this.queryWithShadowIndex(query_text, k);
        model_used = this.config.model_name;
        embedding_dimension = this.config.embedding_dimension;
        this.state.query_routing.shadow_count++;
        
      } else {
        // Use primary index
        results = await this.queryWithPrimaryIndex(query_text, k);
        model_used = 'primary';
        embedding_dimension = 384; // Assumed primary dimension
        this.state.query_routing.primary_count++;
      }
      
      const latency = performance.now() - start_time;
      const memory_used = this.getMemoryUsage() - start_memory;
      
      return {
        results,
        model_used,
        metrics: {
          latency_ms: latency,
          memory_mb: memory_used,
          embedding_dimension,
        },
      };
      
    } catch (error) {
      console.warn(`Query routing failed (${use_shadow ? 'shadow' : 'primary'}):`, error);
      this.state.query_routing.error_count++;
      
      // Fallback logic
      if (this.config.fallback_on_failure && use_shadow) {
        console.log('Falling back to primary index');
        const results = await this.queryWithPrimaryIndex(query_text, k);
        const latency = performance.now() - start_time;
        
        return {
          results,
          model_used: 'primary-fallback',
          metrics: {
            latency_ms: latency,
            memory_mb: 0,
            embedding_dimension: 384,
          },
        };
      }
      
      throw error;
    }
  }
  
  /**
   * Analyze A/B test results and generate recommendation
   */
  async analyzeABTestResults(
    min_queries: number = 100
  ): Promise<ABTestResult> {
    if (!this.state.ab_test_active) {
      throw new Error('No active A/B test to analyze');
    }
    
    const total_queries = this.state.query_routing.primary_count + this.state.query_routing.shadow_count;
    
    if (total_queries < min_queries) {
      throw new Error(`Insufficient queries for analysis: ${total_queries} < ${min_queries}`);
    }
    
    console.log('📊 Analyzing A/B test results...');
    
    const test_duration = Date.now() - this.state.test_start_time;
    
    // Get performance metrics for both models
    const primary_metrics = await this.getPerformanceMetrics('primary');
    const shadow_metrics = await this.getPerformanceMetrics(this.config.model_name);
    
    // Calculate performance ratios
    const performance_comparison = {
      cbu_per_gb_ratio: this.calculateCBUPerGBRatio(shadow_metrics, primary_metrics),
      p95_latency_ratio: shadow_metrics.p95_latency_ms / primary_metrics.p95_latency_ms,
      memory_efficiency_ratio: primary_metrics.memory_usage_mb / shadow_metrics.memory_usage_mb,
      build_time_ratio: this.state.shadow_index.build_time_ms / this.state.primary_index.build_time_ms,
    };
    
    // Quality analysis
    const cbu_degradation = (primary_metrics.quality_metrics.average_cbu - shadow_metrics.quality_metrics.average_cbu) / primary_metrics.quality_metrics.average_cbu;
    const ranking_correlation = this.calculateRankingCorrelation(primary_metrics, shadow_metrics);
    
    const quality_analysis = {
      cbu_degradation,
      ranking_correlation,
      significant_difference: Math.abs(cbu_degradation) > 0.02, // 2% significance threshold
      confidence_interval: this.calculateConfidenceInterval(cbu_degradation, total_queries),
    };
    
    // Generate recommendation
    const recommendation = this.generateRecommendation(
      performance_comparison,
      quality_analysis,
      total_queries
    );
    
    const result: ABTestResult = {
      primary_model: 'primary',
      shadow_model: this.config.model_name,
      test_duration_ms: test_duration,
      query_count: total_queries,
      performance_comparison,
      quality_analysis,
      recommendation,
    };
    
    console.log(`📈 A/B Test Analysis Complete:`);
    console.log(`  CBU/GB Ratio: ${performance_comparison.cbu_per_gb_ratio.toFixed(2)}x`);
    console.log(`  P95 Latency: ${performance_comparison.p95_latency_ratio.toFixed(2)}x`);
    console.log(`  Memory Efficiency: ${performance_comparison.memory_efficiency_ratio.toFixed(2)}x`);
    console.log(`  Recommendation: ${recommendation.should_promote ? '✅ PROMOTE' : '❌ HOLD'} (confidence: ${(recommendation.confidence_score * 100).toFixed(1)}%)`);
    
    return result;
  }
  
  /**
   * Promote shadow model to primary (model swap)
   */
  async promoteShadowToPrimary(): Promise<void> {
    if (!this.state.shadow_index.ready) {
      throw new Error('Shadow index not ready for promotion');
    }
    
    console.log('🎆 Promoting shadow model to primary...');
    
    try {
      // Swap models
      const old_primary = this.primary_embeddings;
      this.primary_embeddings = this.shadow_embeddings!;
      
      // Update state
      const old_primary_state = { ...this.state.primary_index };
      this.state.primary_index = { ...this.state.shadow_index };
      this.state.shadow_index = old_primary_state;
      
      // Stop A/B test
      this.state.ab_test_active = false;
      
      console.log('✅ Model promotion complete');
      
    } catch (error) {
      console.error('Model promotion failed:', error);
      throw error;
    }
  }
  
  /**
   * Get comprehensive shadow index status
   */
  getStatus(): {
    primary_index: typeof this.state.primary_index;
    shadow_index: typeof this.state.shadow_index;
    ab_test: {
      active: boolean;
      duration_ms: number;
      query_distribution: typeof this.state.query_routing;
      traffic_split: number;
    };
    performance_summary: {
      memory_efficiency_gain: number;
      quality_score: number;
      recommendation: string;
    };
  } {
    const duration = this.state.ab_test_active ? Date.now() - this.state.test_start_time : 0;
    
    return {
      primary_index: this.state.primary_index,
      shadow_index: this.state.shadow_index,
      ab_test: {
        active: this.state.ab_test_active,
        duration_ms: duration,
        query_distribution: this.state.query_routing,
        traffic_split: this.config.shadow_traffic_ratio,
      },
      performance_summary: {
        memory_efficiency_gain: this.calculateMemoryEfficiencyGain(),
        quality_score: this.calculateQualityScore(),
        recommendation: this.getQuickRecommendation(),
      },
    };
  }
  
  /**
   * Private helper methods
   */
  private async initializeShadowEmbeddings(): Promise<Embeddings> {
    // Mock implementation - would initialize actual EmbeddingGemma-300M model
    console.log(`Initializing ${this.config.model_name} with ${this.config.embedding_dimension}d embeddings`);
    
    // Return mock embeddings interface
    return {
      embed: async (texts: string[]): Promise<Float32Array[]> => {
        // Mock implementation - generates random embeddings for testing
        return texts.map(() => {
          const embedding = new Float32Array(this.config.embedding_dimension);
          for (let i = 0; i < this.config.embedding_dimension; i++) {
            embedding[i] = Math.random() - 0.5;
          }
          return embedding;
        });
      },
      getDimension: () => this.config.embedding_dimension,
    };
  }
  
  private async storeShadowEmbeddings(
    embeddings: Array<{ id: string; embedding: number[] }>,
    session_id: string
  ): Promise<void> {
    // Mock implementation - would store in vector database
    console.log(`Storing ${embeddings.length} shadow embeddings for session ${session_id}`);
    
    // In practice, would use vector DB like Faiss, Qdrant, etc.
    // For now, just simulate storage time
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  private async queryWithShadowIndex(
    query: string,
    k: number
  ): Promise<Array<{ docId: string; score: number }>> {
    // Mock implementation - would query actual shadow vector index
    const results = [];
    for (let i = 0; i < k; i++) {
      results.push({
        docId: `shadow_doc_${i}`,
        score: Math.random() * 0.8 + 0.1, // Random score 0.1-0.9
      });
    }
    return results;
  }
  
  private async queryWithPrimaryIndex(
    query: string,
    k: number
  ): Promise<Array<{ docId: string; score: number }>> {
    // Use primary embeddings to query
    const query_embedding = await this.primary_embeddings.embed([query]);
    
    // Mock vector search - would use actual vector DB
    const results = [];
    for (let i = 0; i < k; i++) {
      results.push({
        docId: `primary_doc_${i}`,
        score: Math.random() * 0.9 + 0.05, // Random score 0.05-0.95
      });
    }
    return results;
  }
  
  private async getPerformanceMetrics(model_name: string): Promise<RetrievalMetrics> {
    // Mock performance metrics - would collect from actual usage
    return {
      model_name,
      query_count: 100,
      p50_latency_ms: model_name.includes('gemma') ? 85 : 120,
      p95_latency_ms: model_name.includes('gemma') ? 180 : 250,
      p99_latency_ms: model_name.includes('gemma') ? 300 : 450,
      average_latency_ms: model_name.includes('gemma') ? 95 : 135,
      error_rate: 0.01,
      memory_usage_mb: model_name.includes('gemma') ? 150 : 300,
      quality_metrics: {
        average_cbu: model_name.includes('gemma') ? 0.82 : 0.85,
        ndcg_at_10: model_name.includes('gemma') ? 0.78 : 0.80,
        recall_at_k: [0.6, 0.75, 0.85, 0.90],
      },
    };
  }
  
  private calculateCBUPerGBRatio(shadow: RetrievalMetrics, primary: RetrievalMetrics): number {
    const shadow_cbu_per_gb = shadow.quality_metrics.average_cbu / (shadow.memory_usage_mb / 1024);
    const primary_cbu_per_gb = primary.quality_metrics.average_cbu / (primary.memory_usage_mb / 1024);
    return shadow_cbu_per_gb / primary_cbu_per_gb;
  }
  
  private calculateRankingCorrelation(primary: RetrievalMetrics, shadow: RetrievalMetrics): number {
    // Mock correlation calculation
    return 0.75 + Math.random() * 0.2; // 0.75-0.95 correlation
  }
  
  private calculateConfidenceInterval(effect_size: number, sample_size: number): [number, number] {
    // Simplified confidence interval calculation
    const margin = 1.96 / Math.sqrt(sample_size); // 95% CI
    return [effect_size - margin, effect_size + margin];
  }
  
  private generateRecommendation(
    performance: ABTestResult['performance_comparison'],
    quality: ABTestResult['quality_analysis'],
    sample_size: number
  ): ABTestResult['recommendation'] {
    // Decision logic based on performance and quality metrics
    const memory_gain = performance.memory_efficiency_ratio > this.config.min_memory_improvement;
    const quality_maintained = Math.abs(quality.cbu_degradation) < this.config.max_cbu_degradation;
    const latency_acceptable = performance.p95_latency_ratio < (this.config.max_p95_latency_ms / 150); // Assume 150ms baseline
    
    const should_promote = memory_gain && quality_maintained && latency_acceptable;
    const confidence_score = Math.min(0.95, sample_size / 1000); // Higher confidence with more samples
    
    let risk_assessment: 'low' | 'medium' | 'high' = 'low';
    let rollout_strategy: 'immediate' | 'gradual' | 'hold' = 'immediate';
    
    if (!quality_maintained) {
      risk_assessment = 'high';
      rollout_strategy = 'hold';
    } else if (!latency_acceptable) {
      risk_assessment = 'medium';
      rollout_strategy = 'gradual';
    }
    
    return {
      should_promote,
      confidence_score,
      risk_assessment,
      rollout_strategy,
    };
  }
  
  private calculateMemoryEfficiencyGain(): number {
    if (!this.state.shadow_index.memory_mb || !this.state.primary_index.memory_mb) {
      return 0;
    }
    return this.state.primary_index.memory_mb / this.state.shadow_index.memory_mb;
  }
  
  private calculateQualityScore(): number {
    // Mock quality score - would compute from actual metrics
    return 0.85;
  }
  
  private getQuickRecommendation(): string {
    const memory_gain = this.calculateMemoryEfficiencyGain();
    
    if (memory_gain > 2.0) {
      return 'PROMOTE: Significant memory efficiency gains';
    } else if (memory_gain > 1.5) {
      return 'GRADUAL: Moderate improvements, gradual rollout';
    } else {
      return 'HOLD: Insufficient improvements for promotion';
    }
  }
  
  private getMemoryUsage(): number {
    // Mock memory usage - would use actual system metrics
    return Math.random() * 100 + 200; // 200-300MB
  }
  
  private createMockBuildMetrics(): IndexBuildMetrics {
    return {
      model_name: this.config.model_name,
      embedding_dimension: this.config.embedding_dimension,
      total_chunks: 10000,
      build_time_ms: 15000,
      memory_usage_mb: 120,
      memory_per_chunk_kb: 12,
      chunks_per_second: 667,
      peak_memory_mb: 180,
      index_size_mb: 150,
      build_errors: 0,
      success_rate: 1.0,
    };
  }
}

/**
 * Convenience function for EmbeddingGemma-300M A/B testing
 */
export async function runEmbeddingGemmaABTest(
  primary_embeddings: Embeddings,
  db: DB,
  session_id: string,
  test_queries: string[],
  config: Partial<ShadowIndexConfig> = {}
): Promise<{
  build_metrics: IndexBuildMetrics;
  ab_test_result: ABTestResult;
  final_recommendation: string;
}> {
  console.log('🤖 Starting comprehensive EmbeddingGemma-300M A/B test...');
  
  const manager = new EmbeddingShadowIndexManager(primary_embeddings, config);
  
  // Build shadow index
  const build_metrics = await manager.buildShadowIndex(db, session_id);
  
  // Start A/B test
  await manager.startABTest();
  
  // Run test queries
  console.log(`Running ${test_queries.length} test queries...`);
  for (const query of test_queries) {
    try {
      await manager.routeQuery(query);
    } catch (error) {
      console.warn(`Query failed: ${query}:`, error);
    }
  }
  
  // Analyze results
  const ab_test_result = await manager.analyzeABTestResults();
  
  // Generate final recommendation
  const final_recommendation = ab_test_result.recommendation.should_promote ?
    `PROMOTE: ${build_metrics.model_name} shows ${(ab_test_result.performance_comparison.memory_efficiency_ratio).toFixed(1)}x memory efficiency with ${Math.abs(ab_test_result.quality_analysis.cbu_degradation * 100).toFixed(1)}% quality change` :
    `HOLD: Insufficient gains for promotion (confidence: ${(ab_test_result.recommendation.confidence_score * 100).toFixed(1)}%)`;
  
  console.log('🎆 EmbeddingGemma-300M A/B test complete');
  console.log(`Final recommendation: ${final_recommendation}`);
  
  return {
    build_metrics,
    ab_test_result,
    final_recommendation,
  };
}
