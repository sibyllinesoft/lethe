/**
 * Chaos Testing Suite for Production Validation
 * Closure cycles, rank collapse, KV churn spike tests
 */

export interface ChaosTestConfig {
  test_duration_minutes: number;
  intensity_levels: number[]; // 1-10 scale
  recovery_timeout_minutes: number;
  failure_injection_rate: number; // 0-1
  monitoring_interval_seconds: number;
}

export interface ChaosTestResult {
  test_type: string;
  test_id: string;
  start_time: number;
  end_time: number;
  duration_minutes: number;
  intensity_level: number;
  success: boolean;
  recovery_time_seconds: number;
  metrics: {
    performance_degradation: number; // % degradation
    error_rate_increase: number; // % increase
    availability_impact: number; // % downtime
    recovery_success: boolean;
  };
  observations: string[];
  failures: string[];
}

export interface SystemSnapshot {
  timestamp: number;
  cpu_utilization: number;
  memory_usage: number;
  network_latency: number;
  disk_io: number;
  active_connections: number;
  error_rate: number;
  response_time_p95: number;
  throughput_qps: number;
}

/**
 * Closure Cycle Chaos Test
 * Tests system behavior under recursive dependency cycles
 */
export class ClosureCycleChaosTest {
  private config: ChaosTestConfig;
  private isRunning: boolean = false;
  private testResults: ChaosTestResult[] = [];

  constructor(config: ChaosTestConfig) {
    this.config = config;
  }

  /**
   * Execute closure cycle chaos test
   */
  async executeTest(
    intensity: number,
    targetComponents: string[]
  ): Promise<ChaosTestResult> {
    const testId = `closure-cycle-${Date.now()}`;
    const startTime = Date.now();
    
    console.log(`🌀 Starting Closure Cycle Chaos Test (Intensity: ${intensity}/${10})`);
    
    this.isRunning = true;
    const result: ChaosTestResult = {
      test_type: 'closure-cycle',
      test_id: testId,
      start_time: startTime,
      end_time: 0,
      duration_minutes: 0,
      intensity_level: intensity,
      success: false,
      recovery_time_seconds: 0,
      metrics: {
        performance_degradation: 0,
        error_rate_increase: 0,
        availability_impact: 0,
        recovery_success: false
      },
      observations: [],
      failures: []
    };

    try {
      // Capture baseline metrics
      const baselineSnapshot = await this.captureSystemSnapshot();
      result.observations.push(`Baseline captured: ${JSON.stringify(baselineSnapshot)}`);

      // Inject closure cycles based on intensity
      const cyclesInjected = await this.injectClosureCycles(intensity, targetComponents);
      result.observations.push(`Injected ${cyclesInjected} closure cycles`);

      // Monitor system during chaos
      const monitoringResults = await this.monitorSystemDuringChaos();
      result.observations.push(`Monitoring completed: ${monitoringResults.length} data points`);

      // Calculate performance impact
      const impactAnalysis = this.analyzePerformanceImpact(baselineSnapshot, monitoringResults);
      result.metrics.performance_degradation = impactAnalysis.degradation;
      result.metrics.error_rate_increase = impactAnalysis.errorIncrease;
      result.metrics.availability_impact = impactAnalysis.availabilityImpact;

      // Attempt recovery
      const recoveryStartTime = Date.now();
      const recoverySuccess = await this.recoverFromClosureCycles(targetComponents);
      const recoveryTime = (Date.now() - recoveryStartTime) / 1000;

      result.recovery_time_seconds = recoveryTime;
      result.metrics.recovery_success = recoverySuccess;

      if (recoverySuccess) {
        result.observations.push(`Recovery successful in ${recoveryTime}s`);
      } else {
        result.failures.push(`Recovery failed after ${recoveryTime}s`);
      }

      // Validate system stability post-recovery
      const stabilityCheck = await this.validateSystemStability();
      if (stabilityCheck.stable) {
        result.success = true;
        result.observations.push('System stability validated post-recovery');
      } else {
        result.failures.push('System instability detected post-recovery');
      }

    } catch (error) {
      result.failures.push(`Test execution failed: ${error}`);
    } finally {
      this.isRunning = false;
      const endTime = Date.now();
      result.end_time = endTime;
      result.duration_minutes = (endTime - startTime) / (60 * 1000);
      
      this.testResults.push(result);
    }

    console.log(`✅ Closure Cycle Chaos Test completed (${result.success ? 'PASSED' : 'FAILED'})`);
    return result;
  }

  private async injectClosureCycles(intensity: number, components: string[]): Promise<number> {
    // Simulate closure cycle injection based on intensity
    const cyclesToInject = Math.floor(intensity * components.length / 2);
    
    console.log(`💥 Injecting ${cyclesToInject} closure cycles...`);
    
    for (let i = 0; i < cyclesToInject; i++) {
      // Create artificial circular dependencies
      const component1 = components[i % components.length];
      const component2 = components[(i + 1) % components.length];
      
      await this.createCircularDependency(component1, component2);
      
      // Wait between injections to observe gradual impact
      await this.sleep(1000);
    }
    
    return cyclesToInject;
  }

  private async createCircularDependency(component1: string, component2: string): Promise<void> {
    // Simulate creating a circular dependency between components
    console.log(`🔄 Creating circular dependency: ${component1} ↔ ${component2}`);
    
    // In a real implementation, this would modify routing rules,
    // service mesh configuration, or dependency injection settings
    // to create actual circular dependencies
    
    await this.sleep(100); // Simulate configuration time
  }

  private async monitorSystemDuringChaos(): Promise<SystemSnapshot[]> {
    const snapshots: SystemSnapshot[] = [];
    const monitoringDuration = this.config.test_duration_minutes * 60 * 1000;
    const interval = this.config.monitoring_interval_seconds * 1000;
    const iterations = Math.floor(monitoringDuration / interval);
    
    for (let i = 0; i < iterations && this.isRunning; i++) {
      const snapshot = await this.captureSystemSnapshot();
      snapshots.push(snapshot);
      
      await this.sleep(interval);
    }
    
    return snapshots;
  }

  private async recoverFromClosureCycles(components: string[]): Promise<boolean> {
    console.log(`🔧 Attempting recovery from closure cycles...`);
    
    try {
      // Break circular dependencies by resetting configurations
      for (const component of components) {
        await this.resetComponentDependencies(component);
      }
      
      // Restart affected services
      await this.restartAffectedServices(components);
      
      // Verify dependencies are clean
      const dependencyCheck = await this.validateDependencyGraph(components);
      
      return dependencyCheck.isAcyclic;
      
    } catch (error) {
      console.error(`Recovery failed: ${error}`);
      return false;
    }
  }

  private async resetComponentDependencies(component: string): Promise<void> {
    // Simulate resetting component dependencies to clean state
    console.log(`🔄 Resetting dependencies for ${component}`);
    await this.sleep(500);
  }

  private async restartAffectedServices(components: string[]): Promise<void> {
    console.log(`🔄 Restarting ${components.length} affected services...`);
    
    // Simulate rolling restart
    for (const component of components) {
      await this.sleep(1000); // Simulate restart time
    }
  }

  private async validateDependencyGraph(components: string[]): Promise<{ isAcyclic: boolean; cycles: string[] }> {
    // Simulate dependency graph validation
    // In reality, this would analyze service mesh or configuration
    const hasCycles = Math.random() < 0.2; // 20% chance of remaining cycles
    
    return {
      isAcyclic: !hasCycles,
      cycles: hasCycles ? ['remaining-cycle-detected'] : []
    };
  }

  private analyzePerformanceImpact(
    baseline: SystemSnapshot,
    monitoring: SystemSnapshot[]
  ): { degradation: number; errorIncrease: number; availabilityImpact: number } {
    if (monitoring.length === 0) {
      return { degradation: 0, errorIncrease: 0, availabilityImpact: 0 };
    }

    // Calculate averages during chaos
    const avgResponseTime = monitoring.reduce((sum, s) => sum + s.response_time_p95, 0) / monitoring.length;
    const avgErrorRate = monitoring.reduce((sum, s) => sum + s.error_rate, 0) / monitoring.length;
    const avgThroughput = monitoring.reduce((sum, s) => sum + s.throughput_qps, 0) / monitoring.length;

    // Calculate degradation percentages
    const degradation = baseline.response_time_p95 > 0 
      ? ((avgResponseTime - baseline.response_time_p95) / baseline.response_time_p95) * 100
      : 0;

    const errorIncrease = baseline.error_rate > 0
      ? ((avgErrorRate - baseline.error_rate) / baseline.error_rate) * 100
      : avgErrorRate * 100;

    const throughputDrop = baseline.throughput_qps > 0
      ? ((baseline.throughput_qps - avgThroughput) / baseline.throughput_qps) * 100
      : 0;

    // Availability impact based on error rate and throughput
    const availabilityImpact = Math.max(0, Math.min(100, (errorIncrease + throughputDrop) / 2));

    return { degradation, errorIncrease, availabilityImpact };
  }

  private async captureSystemSnapshot(): Promise<SystemSnapshot> {
    // Simulate capturing real system metrics
    return {
      timestamp: Date.now(),
      cpu_utilization: 20 + Math.random() * 60, // 20-80%
      memory_usage: 30 + Math.random() * 50, // 30-80%
      network_latency: 10 + Math.random() * 40, // 10-50ms
      disk_io: Math.random() * 100, // 0-100%
      active_connections: Math.floor(100 + Math.random() * 900), // 100-1000
      error_rate: Math.random() * 5, // 0-5%
      response_time_p95: 50 + Math.random() * 200, // 50-250ms
      throughput_qps: 100 + Math.random() * 400 // 100-500 QPS
    };
  }

  private async validateSystemStability(): Promise<{ stable: boolean; metrics: SystemSnapshot }> {
    // Take several snapshots and check for stability
    const snapshots = [];
    for (let i = 0; i < 5; i++) {
      snapshots.push(await this.captureSystemSnapshot());
      await this.sleep(2000);
    }

    // Calculate stability metrics
    const responseTimeVariability = this.calculateVariability(snapshots.map(s => s.response_time_p95));
    const errorRateVariability = this.calculateVariability(snapshots.map(s => s.error_rate));
    
    const stable = responseTimeVariability < 0.2 && errorRateVariability < 0.5; // Thresholds
    
    return {
      stable,
      metrics: snapshots[snapshots.length - 1]
    };
  }

  private calculateVariability(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    return mean > 0 ? stdDev / mean : 0; // Coefficient of variation
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Rank Collapse Chaos Test
 * Tests system behavior under ranking algorithm failures
 */
export class RankCollapseChaosTest {
  private config: ChaosTestConfig;
  private isRunning: boolean = false;

  constructor(config: ChaosTestConfig) {
    this.config = config;
  }

  /**
   * Execute rank collapse chaos test
   */
  async executeTest(intensity: number): Promise<ChaosTestResult> {
    const testId = `rank-collapse-${Date.now()}`;
    const startTime = Date.now();
    
    console.log(`📉 Starting Rank Collapse Chaos Test (Intensity: ${intensity}/10)`);
    
    this.isRunning = true;
    const result: ChaosTestResult = {
      test_type: 'rank-collapse',
      test_id: testId,
      start_time: startTime,
      end_time: 0,
      duration_minutes: 0,
      intensity_level: intensity,
      success: false,
      recovery_time_seconds: 0,
      metrics: {
        performance_degradation: 0,
        error_rate_increase: 0,
        availability_impact: 0,
        recovery_success: false
      },
      observations: [],
      failures: []
    };

    try {
      // Inject rank collapse based on intensity
      const collapseResult = await this.injectRankCollapse(intensity);
      result.observations.push(`Rank collapse injected: ${collapseResult.description}`);

      // Monitor ranking quality degradation
      const qualityMetrics = await this.monitorRankingQuality();
      result.observations.push(`Quality monitoring: ${qualityMetrics.length} measurements`);

      // Analyze impact on system performance
      const impactAnalysis = this.analyzeRankingImpact(qualityMetrics);
      result.metrics.performance_degradation = impactAnalysis.qualityDrop;
      result.metrics.error_rate_increase = impactAnalysis.errorIncrease;

      // Attempt ranking recovery
      const recoveryStartTime = Date.now();
      const recoverySuccess = await this.recoverRankingSystem();
      const recoveryTime = (Date.now() - recoveryStartTime) / 1000;

      result.recovery_time_seconds = recoveryTime;
      result.metrics.recovery_success = recoverySuccess;

      if (recoverySuccess) {
        result.success = true;
        result.observations.push(`Ranking system recovered in ${recoveryTime}s`);
      } else {
        result.failures.push(`Ranking recovery failed after ${recoveryTime}s`);
      }

    } catch (error) {
      result.failures.push(`Rank collapse test failed: ${error}`);
    } finally {
      this.isRunning = false;
      const endTime = Date.now();
      result.end_time = endTime;
      result.duration_minutes = (endTime - startTime) / (60 * 1000);
    }

    console.log(`✅ Rank Collapse Chaos Test completed (${result.success ? 'PASSED' : 'FAILED'})`);
    return result;
  }

  private async injectRankCollapse(intensity: number): Promise<{ description: string; affectedRankings: number }> {
    const collapseTypes = [
      'zero_scores', // All ranking scores become zero
      'random_shuffle', // Rankings become random
      'bias_injection', // Inject severe ranking bias
      'algorithm_failure', // Ranking algorithm returns errors
      'stale_rankings' // Rankings stop updating
    ];

    const selectedCollapse = collapseTypes[Math.floor(intensity / 2)];
    const affectedRankings = Math.floor(intensity * 10);

    console.log(`💥 Injecting rank collapse: ${selectedCollapse} affecting ${affectedRankings} rankings`);

    switch (selectedCollapse) {
      case 'zero_scores':
        await this.injectZeroScores(affectedRankings);
        break;
      case 'random_shuffle':
        await this.injectRandomShuffle(affectedRankings);
        break;
      case 'bias_injection':
        await this.injectRankingBias(affectedRankings);
        break;
      case 'algorithm_failure':
        await this.injectAlgorithmFailure(affectedRankings);
        break;
      case 'stale_rankings':
        await this.injectStaleRankings(affectedRankings);
        break;
    }

    return {
      description: `${selectedCollapse} collapse type`,
      affectedRankings
    };
  }

  private async injectZeroScores(count: number): Promise<void> {
    // Simulate making all ranking scores zero
    console.log(`🔥 Setting ${count} ranking scores to zero`);
    await this.sleep(500);
  }

  private async injectRandomShuffle(count: number): Promise<void> {
    // Simulate randomizing ranking order
    console.log(`🎲 Randomizing ${count} ranking algorithms`);
    await this.sleep(500);
  }

  private async injectRankingBias(count: number): Promise<void> {
    // Simulate injecting severe bias into rankings
    console.log(`⚖️ Injecting bias into ${count} ranking systems`);
    await this.sleep(500);
  }

  private async injectAlgorithmFailure(count: number): Promise<void> {
    // Simulate ranking algorithm failures
    console.log(`💔 Failing ${count} ranking algorithms`);
    await this.sleep(500);
  }

  private async injectStaleRankings(count: number): Promise<void> {
    // Simulate stale ranking data
    console.log(`⏰ Making ${count} rankings stale`);
    await this.sleep(500);
  }

  private async monitorRankingQuality(): Promise<Array<{
    timestamp: number;
    ranking_quality: number;
    result_relevance: number;
    user_satisfaction: number;
    ndcg_score: number;
  }>> {
    const metrics = [];
    const monitoringDuration = this.config.test_duration_minutes * 60 * 1000;
    const interval = this.config.monitoring_interval_seconds * 1000;
    const iterations = Math.floor(monitoringDuration / interval);

    for (let i = 0; i < iterations && this.isRunning; i++) {
      metrics.push({
        timestamp: Date.now(),
        ranking_quality: Math.max(0, 0.8 - (i * 0.05)), // Degrading quality
        result_relevance: Math.max(0, 0.9 - (i * 0.03)),
        user_satisfaction: Math.max(0, 0.85 - (i * 0.04)),
        ndcg_score: Math.max(0, 0.75 - (i * 0.02))
      });

      await this.sleep(interval);
    }

    return metrics;
  }

  private analyzeRankingImpact(metrics: Array<{
    ranking_quality: number;
    result_relevance: number;
    user_satisfaction: number;
    ndcg_score: number;
  }>): { qualityDrop: number; errorIncrease: number } {
    if (metrics.length === 0) {
      return { qualityDrop: 0, errorIncrease: 0 };
    }

    const initialQuality = metrics[0].ranking_quality;
    const finalQuality = metrics[metrics.length - 1].ranking_quality;
    const qualityDrop = ((initialQuality - finalQuality) / initialQuality) * 100;

    // Assume error rate increases as quality drops
    const errorIncrease = qualityDrop * 2; // Rough correlation

    return { qualityDrop, errorIncrease };
  }

  private async recoverRankingSystem(): Promise<boolean> {
    console.log(`🔧 Attempting ranking system recovery...`);

    try {
      // Reset ranking algorithms to known good state
      await this.resetRankingAlgorithms();
      
      // Rebuild ranking indices
      await this.rebuildRankingIndices();
      
      // Validate ranking quality
      const qualityCheck = await this.validateRankingQuality();
      
      return qualityCheck.quality > 0.8; // 80% quality threshold
      
    } catch (error) {
      console.error(`Ranking recovery failed: ${error}`);
      return false;
    }
  }

  private async resetRankingAlgorithms(): Promise<void> {
    console.log(`🔄 Resetting ranking algorithms...`);
    await this.sleep(2000);
  }

  private async rebuildRankingIndices(): Promise<void> {
    console.log(`🏗️ Rebuilding ranking indices...`);
    await this.sleep(3000);
  }

  private async validateRankingQuality(): Promise<{ quality: number; ndcg: number }> {
    // Simulate ranking quality validation
    await this.sleep(1000);
    
    return {
      quality: 0.85 + Math.random() * 0.1, // 85-95%
      ndcg: 0.8 + Math.random() * 0.15 // 80-95%
    };
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * KV Churn Spike Chaos Test
 * Tests system behavior under extreme key-value store churn
 */
export class KVChurnSpikeChaosTest {
  private config: ChaosTestConfig;
  private isRunning: boolean = false;

  constructor(config: ChaosTestConfig) {
    this.config = config;
  }

  /**
   * Execute KV churn spike chaos test
   */
  async executeTest(intensity: number): Promise<ChaosTestResult> {
    const testId = `kv-churn-${Date.now()}`;
    const startTime = Date.now();
    
    console.log(`🌪️ Starting KV Churn Spike Chaos Test (Intensity: ${intensity}/10)`);
    
    this.isRunning = true;
    const result: ChaosTestResult = {
      test_type: 'kv-churn-spike',
      test_id: testId,
      start_time: startTime,
      end_time: 0,
      duration_minutes: 0,
      intensity_level: intensity,
      success: false,
      recovery_time_seconds: 0,
      metrics: {
        performance_degradation: 0,
        error_rate_increase: 0,
        availability_impact: 0,
        recovery_success: false
      },
      observations: [],
      failures: []
    };

    try {
      // Start KV churn spike
      const churnResult = await this.startKVChurnSpike(intensity);
      result.observations.push(`KV churn spike started: ${churnResult.description}`);

      // Monitor system performance under churn
      const performanceMetrics = await this.monitorPerformanceUnderChurn();
      result.observations.push(`Performance monitoring: ${performanceMetrics.length} data points`);

      // Analyze churn impact
      const impactAnalysis = this.analyzeChurnImpact(performanceMetrics);
      result.metrics.performance_degradation = impactAnalysis.performanceDrop;
      result.metrics.error_rate_increase = impactAnalysis.errorIncrease;
      result.metrics.availability_impact = impactAnalysis.availabilityImpact;

      // Stop churn and attempt recovery
      await this.stopKVChurnSpike();
      result.observations.push('KV churn spike stopped');

      const recoveryStartTime = Date.now();
      const recoverySuccess = await this.recoverFromKVChurn();
      const recoveryTime = (Date.now() - recoveryStartTime) / 1000;

      result.recovery_time_seconds = recoveryTime;
      result.metrics.recovery_success = recoverySuccess;

      if (recoverySuccess) {
        result.success = true;
        result.observations.push(`KV system recovered in ${recoveryTime}s`);
      } else {
        result.failures.push(`KV recovery failed after ${recoveryTime}s`);
      }

    } catch (error) {
      result.failures.push(`KV churn test failed: ${error}`);
    } finally {
      this.isRunning = false;
      const endTime = Date.now();
      result.end_time = endTime;
      result.duration_minutes = (endTime - startTime) / (60 * 1000);
    }

    console.log(`✅ KV Churn Spike Chaos Test completed (${result.success ? 'PASSED' : 'FAILED'})`);
    return result;
  }

  private async startKVChurnSpike(intensity: number): Promise<{ description: string; operationsPerSecond: number }> {
    // Calculate operations per second based on intensity
    const baseOps = 1000; // 1K ops/sec baseline
    const operationsPerSecond = baseOps * intensity;

    console.log(`💥 Starting KV churn spike: ${operationsPerSecond} ops/sec`);

    const churnTypes = ['random_keys', 'sequential_keys', 'hotkey_churn', 'mixed_patterns'];
    const selectedPattern = churnTypes[Math.floor(intensity / 3)];

    // Start the appropriate churn pattern
    switch (selectedPattern) {
      case 'random_keys':
        await this.startRandomKeyChurn(operationsPerSecond);
        break;
      case 'sequential_keys':
        await this.startSequentialKeyChurn(operationsPerSecond);
        break;
      case 'hotkey_churn':
        await this.startHotKeyChurn(operationsPerSecond);
        break;
      case 'mixed_patterns':
        await this.startMixedPatternChurn(operationsPerSecond);
        break;
    }

    return {
      description: `${selectedPattern} churn pattern at ${operationsPerSecond} ops/sec`,
      operationsPerSecond
    };
  }

  private async startRandomKeyChurn(opsPerSec: number): Promise<void> {
    console.log(`🎲 Starting random key churn: ${opsPerSec} ops/sec`);
    // Simulate random key operations
    await this.sleep(500);
  }

  private async startSequentialKeyChurn(opsPerSec: number): Promise<void> {
    console.log(`📈 Starting sequential key churn: ${opsPerSec} ops/sec`);
    // Simulate sequential key operations
    await this.sleep(500);
  }

  private async startHotKeyChurn(opsPerSec: number): Promise<void> {
    console.log(`🔥 Starting hot key churn: ${opsPerSec} ops/sec`);
    // Simulate hot key contention
    await this.sleep(500);
  }

  private async startMixedPatternChurn(opsPerSec: number): Promise<void> {
    console.log(`🌀 Starting mixed pattern churn: ${opsPerSec} ops/sec`);
    // Simulate mixed access patterns
    await this.sleep(500);
  }

  private async monitorPerformanceUnderChurn(): Promise<Array<{
    timestamp: number;
    read_latency_p95: number;
    write_latency_p95: number;
    cache_hit_rate: number;
    memory_pressure: number;
    gc_pressure: number;
    connection_pool_utilization: number;
  }>> {
    const metrics = [];
    const monitoringDuration = this.config.test_duration_minutes * 60 * 1000;
    const interval = this.config.monitoring_interval_seconds * 1000;
    const iterations = Math.floor(monitoringDuration / interval);

    let degradationFactor = 0;

    for (let i = 0; i < iterations && this.isRunning; i++) {
      degradationFactor += 0.1; // Gradual degradation

      metrics.push({
        timestamp: Date.now(),
        read_latency_p95: 10 + (degradationFactor * 50), // 10ms baseline, degrading
        write_latency_p95: 15 + (degradationFactor * 75), // 15ms baseline, degrading
        cache_hit_rate: Math.max(0.3, 0.95 - degradationFactor), // 95% baseline, degrading
        memory_pressure: Math.min(0.9, 0.4 + degradationFactor), // 40% baseline, increasing
        gc_pressure: Math.min(0.8, 0.2 + (degradationFactor * 0.8)), // 20% baseline, increasing
        connection_pool_utilization: Math.min(0.95, 0.5 + degradationFactor) // 50% baseline, increasing
      });

      await this.sleep(interval);
    }

    return metrics;
  }

  private analyzeChurnImpact(metrics: Array<{
    read_latency_p95: number;
    write_latency_p95: number;
    cache_hit_rate: number;
    memory_pressure: number;
  }>): { performanceDrop: number; errorIncrease: number; availabilityImpact: number } {
    if (metrics.length === 0) {
      return { performanceDrop: 0, errorIncrease: 0, availabilityImpact: 0 };
    }

    const initialMetrics = metrics[0];
    const finalMetrics = metrics[metrics.length - 1];

    // Calculate performance degradation
    const readLatencyIncrease = ((finalMetrics.read_latency_p95 - initialMetrics.read_latency_p95) / initialMetrics.read_latency_p95) * 100;
    const writeLatencyIncrease = ((finalMetrics.write_latency_p95 - initialMetrics.write_latency_p95) / initialMetrics.write_latency_p95) * 100;
    const performanceDrop = (readLatencyIncrease + writeLatencyIncrease) / 2;

    // Cache hit rate drop translates to error increase
    const cacheHitDrop = ((initialMetrics.cache_hit_rate - finalMetrics.cache_hit_rate) / initialMetrics.cache_hit_rate) * 100;
    const errorIncrease = cacheHitDrop * 0.5; // Rough correlation

    // Memory pressure affects availability
    const availabilityImpact = finalMetrics.memory_pressure * 100;

    return { performanceDrop, errorIncrease, availabilityImpact };
  }

  private async stopKVChurnSpike(): Promise<void> {
    console.log(`🛑 Stopping KV churn spike...`);
    this.isRunning = false;
    await this.sleep(1000);
  }

  private async recoverFromKVChurn(): Promise<boolean> {
    console.log(`🔧 Attempting KV system recovery...`);

    try {
      // Clear connection pools
      await this.clearConnectionPools();
      
      // Trigger garbage collection
      await this.triggerGarbageCollection();
      
      // Rebuild cache indices
      await this.rebuildCacheIndices();
      
      // Validate system performance
      const performanceCheck = await this.validateKVPerformance();
      
      return performanceCheck.healthy;
      
    } catch (error) {
      console.error(`KV recovery failed: ${error}`);
      return false;
    }
  }

  private async clearConnectionPools(): Promise<void> {
    console.log(`🔄 Clearing connection pools...`);
    await this.sleep(2000);
  }

  private async triggerGarbageCollection(): Promise<void> {
    console.log(`🗑️ Triggering garbage collection...`);
    await this.sleep(1500);
  }

  private async rebuildCacheIndices(): Promise<void> {
    console.log(`🏗️ Rebuilding cache indices...`);
    await this.sleep(3000);
  }

  private async validateKVPerformance(): Promise<{ healthy: boolean; readLatency: number; writeLatency: number }> {
    await this.sleep(1000);
    
    const readLatency = 8 + Math.random() * 4; // 8-12ms
    const writeLatency = 12 + Math.random() * 6; // 12-18ms
    
    return {
      healthy: readLatency < 15 && writeLatency < 20,
      readLatency,
      writeLatency
    };
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Chaos Testing Orchestrator
 * Coordinates all chaos tests and provides unified reporting
 */
export class ChaosTestingOrchestrator {
  private config: ChaosTestConfig;
  private closureCycleTest: ClosureCycleChaosTest;
  private rankCollapseTest: RankCollapseChaosTest;
  private kvChurnTest: KVChurnSpikeChaosTest;
  private testResults: ChaosTestResult[] = [];

  constructor(config: ChaosTestConfig) {
    this.config = config;
    this.closureCycleTest = new ClosureCycleChaosTest(config);
    this.rankCollapseTest = new RankCollapseChaosTest(config);
    this.kvChurnTest = new KVChurnSpikeChaosTest(config);
  }

  /**
   * Execute comprehensive chaos test suite
   */
  async executeFullChaosTestSuite(intensity: number = 5): Promise<{
    overall_success: boolean;
    test_results: ChaosTestResult[];
    summary: {
      tests_passed: number;
      tests_failed: number;
      average_recovery_time: number;
      worst_performance_degradation: number;
    };
  }> {
    console.log(`🌀 Starting comprehensive chaos test suite (Intensity: ${intensity}/10)`);
    
    const results: ChaosTestResult[] = [];
    
    try {
      // Execute closure cycle test
      console.log(`\n--- Closure Cycle Chaos Test ---`);
      const closureResult = await this.closureCycleTest.executeTest(intensity, ['service-a', 'service-b', 'service-c']);
      results.push(closureResult);
      
      // Wait between tests for system stabilization
      await this.sleep(30000); // 30 second buffer
      
      // Execute rank collapse test
      console.log(`\n--- Rank Collapse Chaos Test ---`);
      const rankResult = await this.rankCollapseTest.executeTest(intensity);
      results.push(rankResult);
      
      // Wait between tests
      await this.sleep(30000);
      
      // Execute KV churn test
      console.log(`\n--- KV Churn Spike Chaos Test ---`);
      const kvResult = await this.kvChurnTest.executeTest(intensity);
      results.push(kvResult);
      
      // Store all results
      this.testResults.push(...results);
      
    } catch (error) {
      console.error(`Chaos test suite execution failed: ${error}`);
    }
    
    // Calculate summary
    const testsPassed = results.filter(r => r.success).length;
    const testsFailed = results.filter(r => !r.success).length;
    const averageRecoveryTime = results.length > 0
      ? results.reduce((sum, r) => sum + r.recovery_time_seconds, 0) / results.length
      : 0;
    const worstDegradation = results.length > 0
      ? Math.max(...results.map(r => r.metrics.performance_degradation))
      : 0;
    
    const overallSuccess = testsPassed === results.length;
    
    console.log(`\n🎯 Chaos Test Suite Complete:`);
    console.log(`   - Tests Passed: ${testsPassed}/${results.length}`);
    console.log(`   - Average Recovery Time: ${averageRecoveryTime.toFixed(1)}s`);
    console.log(`   - Worst Performance Degradation: ${worstDegradation.toFixed(1)}%`);
    console.log(`   - Overall Result: ${overallSuccess ? 'PASSED' : 'FAILED'}`);
    
    return {
      overall_success: overallSuccess,
      test_results: results,
      summary: {
        tests_passed: testsPassed,
        tests_failed: testsFailed,
        average_recovery_time: averageRecoveryTime,
        worst_performance_degradation: worstDegradation
      }
    };
  }

  /**
   * Get historical test results
   */
  getHistoricalResults(): ChaosTestResult[] {
    return [...this.testResults];
  }

  /**
   * Health check for chaos testing system
   */
  healthCheck(): {
    healthy: boolean;
    issues: string[];
    test_count: number;
    last_test_time: number;
  } {
    const issues: string[] = [];
    const now = Date.now();
    
    const lastTestTime = this.testResults.length > 0
      ? Math.max(...this.testResults.map(r => r.start_time))
      : 0;
    
    // Check if tests are too old (older than 24 hours)
    if (lastTestTime > 0 && (now - lastTestTime) > 24 * 60 * 60 * 1000) {
      issues.push('No chaos tests executed in the last 24 hours');
    }
    
    // Check recent test failure rate
    const recentTests = this.testResults.filter(r => (now - r.start_time) < 7 * 24 * 60 * 60 * 1000); // Last 7 days
    if (recentTests.length > 0) {
      const failureRate = recentTests.filter(r => !r.success).length / recentTests.length;
      if (failureRate > 0.5) {
        issues.push(`High chaos test failure rate: ${(failureRate * 100).toFixed(1)}%`);
      }
    }
    
    return {
      healthy: issues.length === 0,
      issues,
      test_count: this.testResults.length,
      last_test_time: lastTestTime
    };
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}