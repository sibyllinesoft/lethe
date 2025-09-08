/**
 * Production Validation System - Usage Examples
 * 
 * Comprehensive examples demonstrating production readiness validation
 * for the Lethe retrieval system.
 */

import { DB } from '@lethe/sqlite';
import type { Embeddings } from '@lethe/embeddings';
import { 
  hybridRetrieval, 
  ProductionReadinessOrchestrator,
  type ProductionReadinessConfig,
  type HybridConfig 
} from '../index.js';

// =============================================================================
// Example 1: Basic Production Validation Integration
// =============================================================================

export async function basicProductionValidation(
  db: DB, 
  embeddings: Embeddings, 
  sessionId: string
) {
  console.log('=== Basic Production Validation Example ===');
  
  const queries = [
    "implement async retry logic with exponential backoff",
    "handle database connection errors gracefully"
  ];
  
  // Enable production validation in retrieval config
  const config: Partial<HybridConfig> = {
    // Standard retrieval settings
    alpha: 0.7,
    beta: 0.3,
    k_initial: 50,
    k_final: 20,
    rerank: true,
    diversify: true,
    
    // Production validation settings
    production_validation: {
      enable_validation: true,
      enable_monitoring: true,
      
      // Quality gate thresholds
      dual_sanity_threshold: 0.005,    // <0.5% primal-dual gap
      ood_ece_threshold: 0.08,         // ≤8% expected calibration error
      win_rate_threshold: 0.80,        // ≥80% statistical power
      
      // Operational settings
      fail_fast_on_validation: false,   // Continue on validation failures
      enable_chaos_testing: true,       // Enable fault injection testing
      risk_budget_threshold: 0.10       // 10% monthly error budget
    }
  };
  
  try {
    // Run retrieval with production validation
    const results = await hybridRetrieval(queries, {
      db,
      embeddings,
      sessionId,
      config
    });
    
    console.log(`✅ Retrieval completed with validation: ${results.length} results`);
    return results;
    
  } catch (error) {
    console.error('❌ Retrieval failed:', error);
    throw error;
  }
}

// =============================================================================
// Example 2: Advanced Production Orchestrator
// =============================================================================

export async function advancedProductionOrchestrator(sessionId: string) {
  console.log('=== Advanced Production Orchestrator Example ===');
  
  // Comprehensive production readiness configuration
  const config: ProductionReadinessConfig = {
    session_id: sessionId,
    
    // Enable all subsystems
    enable_validation: true,
    enable_monitoring: true,
    enable_hierarchical_interleaving: true,
    enable_dpp_optimization: true,
    enable_embedding_gemma_trial: true,
    
    // Strict quality gates for production
    dual_sanity_threshold: 0.003,      // Stricter than default
    ood_ece_threshold: 0.06,           // Stricter calibration requirement
    win_rate_threshold: 0.85,          // Higher statistical power
    
    // Monitoring configuration
    cusum_threshold: 2.5,              // CUSUM change detection sensitivity
    lambda_drift_bounds: [-0.05, 0.05], // Tighter drift bounds
    risk_budget_threshold: 0.05,       // 5% monthly error budget
    
    // Chaos testing scenarios
    enable_chaos_testing: true,
    chaos_scenarios: [
      'closure_cycle_injection',
      'rank_collapse_simulation', 
      'kv_churn_spike',
      'embedding_corruption',
      'network_partition'
    ],
    
    // DPP optimization settings
    dpp_config: {
      enable_rank_tuning: true,
      target_efficiency: 15.0,          // ΔCBU/ms target
      group_split_threshold: 0.7,       // 70% contribution threshold
      max_optimization_iterations: 100,
      convergence_tolerance: 0.001
    },
    
    // EmbeddingGemma trial configuration
    embedding_trial_config: {
      trial_duration_days: 7,
      canary_traffic_percentage: 5,
      promotion_threshold_cbu: 0.10,    // ≥+10% ΔCBU/GB improvement
      promotion_threshold_latency: 5,   // ≥5ms p95 latency improvement
      rollback_threshold_error_rate: 0.01 // 1% error rate triggers rollback
    },
    
    // Fail-fast for critical environments
    fail_fast_on_validation: true
  };
  
  // Initialize orchestrator
  const orchestrator = new ProductionReadinessOrchestrator(config);
  
  try {
    // Simulate production readiness assessment
    const assessment = await orchestrator.assessProductionReadiness({
      query_text: "implement robust error handling with circuit breaker pattern",
      candidate_pool: [
        { docId: 'doc1', score: 0.9, text: 'Circuit breaker implementation...' },
        { docId: 'doc2', score: 0.8, text: 'Error handling best practices...' },
        { docId: 'doc3', score: 0.7, text: 'Retry logic with exponential backoff...' }
      ],
      retrieval_config: {
        alpha: 0.7,
        beta: 0.3,
        k_final: 20
      },
      system_metrics: {
        current_load: 0.65,
        memory_usage: 0.45,
        cpu_utilization: 0.50
      }
    });
    
    // Log detailed assessment results
    console.log('\n📊 Production Readiness Assessment:');
    console.log(`   Overall Readiness: ${assessment.overall_readiness ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`   Risk Score: ${(assessment.risk_assessment.overall_risk_score * 100).toFixed(1)}%`);
    
    if (assessment.validation_results) {
      const { validation_results } = assessment;
      
      if (validation_results.dual_sanity_check) {
        const dual = validation_results.dual_sanity_check;
        console.log(`   Dual Sanity: λ=${dual.lambda.toFixed(4)}, gap=${dual.primal_dual_gap.toFixed(3)} (${dual.monotonicity_satisfied ? '✅' : '❌'})`);
      }
      
      if (validation_results.ood_resilience) {
        const ood = validation_results.ood_resilience;
        console.log(`   OOD Resilience: ECE=${(ood.expected_calibration_error * 100).toFixed(1)}%, coverage=${(ood.mondrian_coverage * 100).toFixed(1)}% (${ood.coverage_achieved ? '✅' : '❌'})`);
      }
      
      if (validation_results.long_horizon_win_rate) {
        const win = validation_results.long_horizon_win_rate;
        console.log(`   Win Rate: ${(win.win_rate * 100).toFixed(1)}%, power=${(win.power_analysis.achieved_power * 100).toFixed(1)}% (${win.statistical_power_achieved ? '✅' : '❌'})`);
      }
    }
    
    if (!assessment.overall_readiness) {
      console.log(`   Failing Components: ${assessment.failing_components.join(', ')}`);
      console.log(`   Recommendations: ${assessment.recommendations.join('; ')}`);
    }
    
    return assessment;
    
  } catch (error) {
    console.error('❌ Production readiness assessment failed:', error);
    throw error;
  }
}

// =============================================================================
// Example 3: Continuous Monitoring Setup
// =============================================================================

export async function setupContinuousMonitoring(sessionId: string) {
  console.log('=== Continuous Monitoring Setup Example ===');
  
  const { ProductionMonitoringSystem } = await import('../monitoring_system.js');
  
  // Initialize monitoring system
  const monitoring = new ProductionMonitoringSystem({
    session_id: sessionId,
    cusum_threshold: 2.5,
    lambda_drift_bounds: [-0.1, 0.1],
    risk_budget_threshold: 0.10,
    alert_channels: ['console'], // In real usage: ['slack', 'email', 'pagerduty']
    monitoring_interval_ms: 60000 // Monitor every minute
  });
  
  // Set up alert handlers
  monitoring.onAlert('lambda_drift', async (alert) => {
    console.log('🚨 Lambda drift detected:', alert);
    // In production: await notifyOnCallTeam(alert);
  });
  
  monitoring.onAlert('risk_budget_exceeded', async (alert) => {
    console.log('🚨 Risk budget exceeded:', alert);
    // In production: await triggerIncidentResponse(alert);
  });
  
  monitoring.onAlert('cusum_change_detection', async (alert) => {
    console.log('🚨 System change detected:', alert);
    // In production: await investigateSystemChanges(alert);
  });
  
  try {
    // Start continuous monitoring
    await monitoring.startMonitoring();
    console.log('✅ Continuous monitoring started');
    
    // Simulate some monitoring data
    await simulateMonitoringData(monitoring);
    
    // Get current metrics
    const metrics = await monitoring.getCurrentMetrics();
    console.log('\n📊 Current Monitoring Metrics:');
    console.log('   Lambda:', metrics.current_lambda.toFixed(4));
    console.log('   Size Ratio:', metrics.current_size_ratio.toFixed(3));
    console.log('   CBU Rate:', metrics.current_cbu_rate.toFixed(2), 'CBU/s');
    console.log('   Risk Budget Used:', (metrics.risk_budget_used * 100).toFixed(1), '%');
    
    return monitoring;
    
  } catch (error) {
    console.error('❌ Monitoring setup failed:', error);
    throw error;
  }
}

async function simulateMonitoringData(monitoring: any) {
  // Simulate monitoring data over time
  const dataPoints = [
    { lambda: 1.2, size_ratio: 0.85, cbu_rate: 45.2, error_count: 0 },
    { lambda: 1.25, size_ratio: 0.87, cbu_rate: 46.8, error_count: 1 },
    { lambda: 1.22, size_ratio: 0.86, cbu_rate: 45.9, error_count: 0 },
    { lambda: 1.28, size_ratio: 0.89, cbu_rate: 48.1, error_count: 2 } // Drift detection
  ];
  
  for (const point of dataPoints) {
    await monitoring.recordMetrics({
      timestamp: Date.now(),
      lambda: point.lambda,
      size_ratio: point.size_ratio,
      cbu_rate: point.cbu_rate,
      error_count: point.error_count
    });
    
    // Small delay to simulate real-time data
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

// =============================================================================
// Example 4: Hierarchical Interleaving A/B Test
// =============================================================================

export async function runHierarchicalInterleavingTest(sessionId: string) {
  console.log('=== Hierarchical Interleaving A/B Test Example ===');
  
  const { HierarchicalInterleavingEngine } = await import('../hierarchical_interleaving.js');
  
  // Setup interleaving experiment
  const interleaving = new HierarchicalInterleavingEngine({
    atom_level_interleaving: true,
    cluster_pair_sessions: true,
    statistical_power_target: 0.85,
    minimum_effect_size: 0.05 // 5% minimum nDCG improvement
  });
  
  try {
    // Create experiment configuration
    const experiment = await interleaving.setupExperiment({
      experiment_name: 'new_ranking_algorithm_test',
      session_id: sessionId,
      baseline_system: 'current_production_v2.1',
      test_system: 'candidate_algorithm_v3.0',
      target_sessions: 10000,
      traffic_split: { baseline: 0.5, test: 0.5 },
      evaluation_metrics: ['ndcg_at_10', 'user_satisfaction', 'task_completion_rate']
    });
    
    console.log(`✅ Experiment setup complete: ${experiment.experiment_id}`);
    
    // Simulate interleaving sessions
    const sessions = await simulateInterleavingSessions(interleaving, experiment, 100);
    console.log(`📊 Simulated ${sessions.length} interleaving sessions`);
    
    // Analyze results
    const analysis = await interleaving.analyzeResults({
      experiment_id: experiment.experiment_id,
      minimum_sessions: 100,
      confidence_level: 0.95
    });
    
    console.log('\n📈 Interleaving Analysis Results:');
    console.log(`   Statistical Power: ${(analysis.statistical_power * 100).toFixed(1)}%`);
    console.log(`   nDCG@10 Improvement: ${(analysis.ndcg_improvement * 100).toFixed(2)}%`);
    console.log(`   Win Rate: ${(analysis.win_rate * 100).toFixed(1)}%`);
    console.log(`   Sessions Required: ${analysis.sessions_required}`);
    console.log(`   Significant: ${analysis.is_significant ? '✅ YES' : '❌ NO'}`);
    
    if (analysis.is_significant && analysis.ndcg_improvement > 0.05) {
      console.log('🚀 Recommendation: Promote test system to production');
    } else {
      console.log('⏳ Recommendation: Continue data collection or iterate on test system');
    }
    
    return analysis;
    
  } catch (error) {
    console.error('❌ Hierarchical interleaving failed:', error);
    throw error;
  }
}

async function simulateInterleavingSessions(
  interleaving: any, 
  experiment: any, 
  sessionCount: number
): Promise<any[]> {
  const sessions = [];
  
  for (let i = 0; i < sessionCount; i++) {
    const session = await interleaving.executeInterleavingSession({
      experiment_id: experiment.experiment_id,
      session_id: `sim_session_${i}`,
      query_sequence: [
        "implement error handling",
        "add logging functionality", 
        "create unit tests",
        "optimize performance"
      ],
      user_feedback: {
        satisfaction_score: Math.random() * 0.3 + 0.7, // 0.7-1.0 range
        task_completed: Math.random() > 0.2 // 80% completion rate
      }
    });
    
    sessions.push(session);
  }
  
  return sessions;
}

// =============================================================================
// Example 5: DPP Optimization Workflow
// =============================================================================

export async function runDPPOptimizationWorkflow() {
  console.log('=== DPP Optimization Workflow Example ===');
  
  const { DPPOptimizationEngine } = await import('../dpp_optimization.js');
  
  // Initialize DPP optimization engine
  const dpp = new DPPOptimizationEngine({
    enable_rank_tuning: true,
    target_efficiency: 15.0,
    group_split_threshold: 0.7,
    max_optimization_iterations: 100,
    convergence_tolerance: 0.001
  });
  
  try {
    // Simulate candidate embeddings
    const candidateEmbeddings = generateMockEmbeddings(50, 384);
    
    console.log(`🧮 Starting DPP optimization with ${candidateEmbeddings.length} candidates`);
    
    // Run optimization
    const optimization = await dpp.optimizeDiversityRanking({
      candidate_embeddings: candidateEmbeddings,
      efficiency_target: 15.0,
      max_iterations: 50
    });
    
    console.log('\n📊 DPP Optimization Results:');
    console.log(`   Optimal Rank: ${optimization.optimal_rank}`);
    console.log(`   Final Efficiency: ${optimization.final_efficiency.toFixed(2)} ΔCBU/ms`);
    console.log(`   Diversity Score: ${optimization.diversity_score.toFixed(3)}`);
    console.log(`   Iterations: ${optimization.iterations_completed}`);
    console.log(`   Convergence: ${optimization.converged ? '✅ YES' : '❌ NO'}`);
    
    // Performance curve analysis
    if (optimization.efficiency_curve.length > 0) {
      console.log('\n📈 Efficiency Curve (rank → ΔCBU/ms):');
      optimization.efficiency_curve.slice(0, 5).forEach(point => {
        console.log(`   Rank ${point.rank}: ${point.efficiency.toFixed(2)} ΔCBU/ms`);
      });
    }
    
    // Group split analysis
    if (optimization.group_split_results) {
      const splits = optimization.group_split_results;
      console.log(`\n🔄 Group Splits: ${splits.splits_performed} performed, ${splits.effectiveness.toFixed(3)} avg effectiveness`);
    }
    
    return optimization;
    
  } catch (error) {
    console.error('❌ DPP optimization failed:', error);
    throw error;
  }
}

function generateMockEmbeddings(count: number, dimensions: number): Array<{
  id: string;
  embedding: Float32Array;
  importance: number;
  group_id?: string;
}> {
  const embeddings = [];
  
  for (let i = 0; i < count; i++) {
    const embedding = new Float32Array(dimensions);
    for (let j = 0; j < dimensions; j++) {
      embedding[j] = Math.random() * 2 - 1; // Range [-1, 1]
    }
    
    embeddings.push({
      id: `candidate_${i}`,
      embedding,
      importance: Math.random(),
      group_id: i % 5 === 0 ? `group_${Math.floor(i / 5)}` : undefined
    });
  }
  
  return embeddings;
}

// =============================================================================
// Example 6: EmbeddingGemma Canary Trial
// =============================================================================

export async function runEmbeddingGemmaCanaryTrial(sessionId: string) {
  console.log('=== EmbeddingGemma Canary Trial Example ===');
  
  const { EmbeddingGemmaTrialEngine } = await import('../embedding_gemma_trial.js');
  
  // Initialize trial engine
  const trial = new EmbeddingGemmaTrialEngine({
    trial_duration_days: 7,
    canary_traffic_percentage: 5,
    promotion_threshold_cbu: 0.10,
    promotion_threshold_latency: 5,
    rollback_threshold_error_rate: 0.01,
    safety_monitoring_interval: 300000 // 5 minutes
  });
  
  try {
    // Start canary trial
    const canaryTrial = await trial.startCanaryTrial({
      trial_name: 'EmbeddingGemma-300M-Production-Trial',
      session_id: sessionId,
      baseline_model: 'current_embedding_model_v2.1',
      candidate_model: 'EmbeddingGemma-300M',
      target_metrics: {
        cbu_efficiency: 0.10,  // ≥10% improvement
        p95_latency: 5,        // ≥5ms improvement
        accuracy: 0.02         // ≥2% accuracy improvement
      }
    });
    
    console.log(`🚀 Canary trial started: ${canaryTrial.trial_id}`);
    
    // Simulate trial monitoring over time
    const monitoringResults = await simulateTrialMonitoring(trial, canaryTrial, 7);
    
    // Check promotion eligibility
    const promotionCheck = await trial.checkPromotionEligibility({
      trial_id: canaryTrial.trial_id
    });
    
    console.log('\n📊 Trial Monitoring Results:');
    console.log(`   Days Monitored: ${monitoringResults.days_monitored}`);
    console.log(`   CBU Efficiency Gain: ${(monitoringResults.cbu_efficiency_gain * 100).toFixed(1)}%`);
    console.log(`   P95 Latency Improvement: ${monitoringResults.p95_latency_improvement.toFixed(1)}ms`);
    console.log(`   Error Rate: ${(monitoringResults.error_rate * 100).toFixed(3)}%`);
    
    console.log('\n🎯 Promotion Decision:');
    console.log(`   Eligible for Promotion: ${promotionCheck.eligible ? '✅ YES' : '❌ NO'}`);
    console.log(`   Promotion Score: ${promotionCheck.promotion_score.toFixed(3)}`);
    
    if (promotionCheck.eligible) {
      console.log('🚀 Recommendation: Promote EmbeddingGemma to production');
      
      // Simulate promotion
      const promotion = await trial.promoteToProduction({
        trial_id: canaryTrial.trial_id,
        rollout_strategy: 'gradual', // gradual | immediate
        rollout_percentage_steps: [10, 25, 50, 100]
      });
      
      console.log(`✅ Promotion initiated: ${promotion.promotion_id}`);
    } else {
      console.log('⏳ Recommendation: Extend trial or investigate performance issues');
      console.log(`   Missing Requirements: ${promotionCheck.missing_requirements.join(', ')}`);
    }
    
    return { canaryTrial, monitoringResults, promotionCheck };
    
  } catch (error) {
    console.error('❌ EmbeddingGemma trial failed:', error);
    throw error;
  }
}

async function simulateTrialMonitoring(
  trial: any, 
  canaryTrial: any, 
  days: number
): Promise<any> {
  let cbuEfficiencyGain = 0;
  let p95LatencyImprovement = 0;
  let errorRate = 0.001; // Start with low error rate
  
  for (let day = 1; day <= days; day++) {
    // Simulate gradual improvement
    cbuEfficiencyGain = 0.05 + (day / days) * 0.08; // Reaches ~13% by end
    p95LatencyImprovement = 2 + (day / days) * 4;    // Reaches ~6ms by end
    errorRate = Math.max(0.0005, 0.002 - (day / days) * 0.0015); // Decreases over time
    
    await trial.recordDailyMetrics({
      trial_id: canaryTrial.trial_id,
      day: day,
      metrics: {
        cbu_efficiency_gain: cbuEfficiencyGain,
        p95_latency_improvement: p95LatencyImprovement,
        error_rate: errorRate,
        request_count: 50000 + Math.random() * 10000,
        user_satisfaction: 0.85 + Math.random() * 0.1
      }
    });
    
    console.log(`   Day ${day}: CBU=${(cbuEfficiencyGain*100).toFixed(1)}%, Latency=${p95LatencyImprovement.toFixed(1)}ms, Errors=${(errorRate*100).toFixed(3)}%`);
  }
  
  return {
    days_monitored: days,
    cbu_efficiency_gain: cbuEfficiencyGain,
    p95_latency_improvement: p95LatencyImprovement,
    error_rate: errorRate
  };
}

// =============================================================================
// Example 7: Complete Production Pipeline
// =============================================================================

export async function completeProductionPipeline(
  db: DB,
  embeddings: Embeddings,
  sessionId: string
) {
  console.log('=== Complete Production Pipeline Example ===');
  console.log('This example demonstrates the full production-ready retrieval pipeline with all systems enabled.');
  
  const queries = [
    "implement comprehensive error handling with circuit breaker pattern",
    "add structured logging with correlation IDs",
    "create integration tests with test containers"
  ];
  
  // Production-grade configuration
  const config: Partial<HybridConfig> = {
    // Core retrieval parameters
    alpha: 0.7,
    beta: 0.3,
    k_initial: 50,
    k_final: 20,
    rerank: true,
    diversify: true,
    diversify_method: 'semantic',
    
    // ML enhancements
    fusion: {
      dynamic: true
    },
    
    // LLM reranking
    llm_rerank: {
      use_llm: true,
      llm_budget_ms: 1500,
      llm_model: 'llama3.2:3b',
      contradiction_enabled: true,
      contradiction_penalty: 0.15
    },
    
    // Full production validation suite
    production_validation: {
      enable_validation: true,
      enable_monitoring: true,
      enable_hierarchical_interleaving: true,
      enable_dpp_optimization: true,
      enable_embedding_gemma_trial: true,
      
      // Strict production thresholds
      dual_sanity_threshold: 0.003,
      ood_ece_threshold: 0.06,
      win_rate_threshold: 0.85,
      
      // Operational hardening
      fail_fast_on_validation: false,
      enable_chaos_testing: true,
      risk_budget_threshold: 0.05,
      
      // Advanced monitoring
      cusum_threshold: 2.0,
      lambda_drift_bounds: [-0.05, 0.05],
      
      // Chaos testing scenarios
      chaos_scenarios: [
        'closure_cycle_injection',
        'rank_collapse_simulation',
        'kv_churn_spike',
        'embedding_corruption'
      ],
      
      // DPP optimization
      dpp_config: {
        enable_rank_tuning: true,
        target_efficiency: 18.0,
        group_split_threshold: 0.75,
        max_optimization_iterations: 150
      },
      
      // EmbeddingGemma trial
      embedding_trial_config: {
        trial_duration_days: 7,
        canary_traffic_percentage: 5,
        promotion_threshold_cbu: 0.12,
        promotion_threshold_latency: 6
      }
    }
  };
  
  console.log('🚀 Starting production pipeline with full validation suite...');
  const startTime = performance.now();
  
  try {
    // Execute complete production pipeline
    const results = await hybridRetrieval(queries, {
      db,
      embeddings,
      sessionId,
      config
    });
    
    const totalTime = performance.now() - startTime;
    
    console.log('\n✅ Production Pipeline Completed Successfully!');
    console.log(`   Results: ${results.length} candidates`);
    console.log(`   Total Time: ${totalTime.toFixed(1)}ms`);
    console.log(`   Avg Time per Query: ${(totalTime / queries.length).toFixed(1)}ms`);
    
    // Display top results
    console.log('\n🎯 Top Results:');
    results.slice(0, 3).forEach((result, i) => {
      console.log(`   ${i + 1}. ${result.docId} (score: ${result.score.toFixed(3)})`);
      if (result.text) {
        console.log(`      "${result.text.substring(0, 80)}..."`);
      }
    });
    
    return {
      results,
      totalTime,
      avgTimePerQuery: totalTime / queries.length,
      pipeline_validated: true
    };
    
  } catch (error) {
    console.error('❌ Production pipeline failed:', error);
    throw error;
  }
}

// =============================================================================
// Example 8: Error Scenarios and Handling
// =============================================================================

export async function demonstrateErrorHandling(sessionId: string) {
  console.log('=== Error Handling Demonstration ===');
  
  // Test various error scenarios
  const scenarios = [
    {
      name: 'Validation Threshold Violation',
      config: {
        enable_validation: true,
        dual_sanity_threshold: 0.001, // Very strict threshold
        fail_fast_on_validation: true
      }
    },
    {
      name: 'Risk Budget Exceeded',
      config: {
        enable_monitoring: true,
        risk_budget_threshold: 0.01, // Very low threshold
        fail_fast_on_validation: false
      }
    },
    {
      name: 'Chaos Test Failure',
      config: {
        enable_chaos_testing: true,
        chaos_scenarios: ['catastrophic_failure_simulation'],
        fail_fast_on_validation: true
      }
    }
  ];
  
  for (const scenario of scenarios) {
    console.log(`\n🧪 Testing Scenario: ${scenario.name}`);
    
    const orchestrator = new ProductionReadinessOrchestrator({
      session_id: `${sessionId}_${scenario.name.toLowerCase().replace(/\s+/g, '_')}`,
      ...scenario.config
    } as ProductionReadinessConfig);
    
    try {
      const assessment = await orchestrator.assessProductionReadiness({
        query_text: "test error handling scenario",
        candidate_pool: [
          { docId: 'test1', score: 0.5 },
          { docId: 'test2', score: 0.3 }
        ],
        retrieval_config: { alpha: 0.7, beta: 0.3 },
        system_metrics: {
          current_load: 0.9, // High load to trigger issues
          memory_usage: 0.8,
          cpu_utilization: 0.85
        }
      });
      
      if (assessment.overall_readiness) {
        console.log('   ✅ Scenario passed (unexpected)');
      } else {
        console.log('   ⚠️ Scenario failed as expected');
        console.log(`   Failing components: ${assessment.failing_components.join(', ')}`);
        console.log(`   Risk score: ${(assessment.risk_assessment.overall_risk_score * 100).toFixed(1)}%`);
      }
      
    } catch (error) {
      console.log(`   🛡️ Fail-fast triggered: ${error.message}`);
    }
  }
  
  console.log('\n✅ Error handling demonstration complete');
}

// =============================================================================
// Main Demo Function
// =============================================================================

export async function runAllExamples(db: DB, embeddings: Embeddings) {
  const sessionId = `production_demo_${Date.now()}`;
  
  console.log('🚀 Starting Complete Production Validation Examples');
  console.log(`Session ID: ${sessionId}\n`);
  
  try {
    // Run all examples
    await basicProductionValidation(db, embeddings, sessionId);
    await advancedProductionOrchestrator(sessionId);
    await setupContinuousMonitoring(sessionId);
    await runHierarchicalInterleavingTest(sessionId);
    await runDPPOptimizationWorkflow();
    await runEmbeddingGemmaCanaryTrial(sessionId);
    await completeProductionPipeline(db, embeddings, sessionId);
    await demonstrateErrorHandling(sessionId);
    
    console.log('\n🎉 All Production Validation Examples Completed Successfully!');
    
  } catch (error) {
    console.error('\n❌ Examples failed:', error);
    throw error;
  }
}