/**
 * Configuration Integration Example
 * 
 * Demonstrates how to integrate the new configuration management system
 * with the existing Lethe components for paper reinforcement.
 * 
 * Features shown:
 * - Environment-aware configuration setup
 * - Feature flag-driven component initialization
 * - Dynamic configuration updates during runtime
 * - Performance monitoring with SLA enforcement
 * - Circuit breaker pattern for resilience
 */

import { 
  ConfigurationManager, 
  LetheConfig, 
  FeatureFlagContext, 
  initializeConfig,
  getConfigManager,
  isFeatureEnabled,
  getConfigSection
} from '../config.js';
import { OrchestrationSystem } from '../orchestrate.js';
import { TelemetrySystem } from '../telemetry.js';
import { Result, PerformanceMetrics } from '../types.js';
import { LetheError } from '../errors.js';

/**
 * Environment-aware configuration setup
 */
export function setupConfigurationForEnvironment(
  environment: 'development' | 'testing' | 'staging' | 'production'
): ConfigurationManager {
  const baseConfig: Partial<LetheConfig> = {
    system: {
      environment,
      logLevel: environment === 'production' ? 'warn' : 'debug',
      enableMetrics: true,
      maxConcurrentRequests: environment === 'production' ? 500 : 100,
      requestTimeoutMs: environment === 'production' ? 10000 : 30000,
      memoryLimitMB: environment === 'production' ? 4096 : 2048
    },
    performance: {
      // Stricter SLAs for production
      targetP50Ms: environment === 'production' ? 2000 : 3000,
      targetP95Ms: environment === 'production' ? 4000 : 6000,
      maxMemoryMB: environment === 'production' ? 1024 : 1536,
      circuitBreakerThreshold: environment === 'production' ? 0.1 : 0.5,
      adaptiveScaling: environment === 'production'
    },
    features: {
      // Progressive feature enablement
      enableAdvancedRetrieval: true,
      enableSemanticDiversification: environment !== 'testing', // Disable in testing
      enableLLMReranking: environment === 'production' ? true : false, // Production only
      enableContradictionDetection: environment === 'production',
      enableCaching: environment !== 'development',
      enableCircuitBreaker: environment === 'production',
      enableDetailedLogging: environment === 'development',
      enablePerformanceTracing: environment !== 'production',
      
      // Rollout controls for gradual feature deployment
      rolloutPercentage: {
        enableLLMReranking: environment === 'staging' ? 50 : 100,
        enableContradictionDetection: environment === 'staging' ? 25 : 100
      },
      
      // Environment overrides
      environmentOverrides: {
        enableDetailedLogging: environment === 'development'
      }
    }
  };

  return initializeConfig(baseConfig);
}

/**
 * Feature-aware component factory
 */
export class LetheComponentFactory {
  private configManager: ConfigurationManager;
  private telemetrySystem: TelemetrySystem;

  constructor(configManager: ConfigurationManager) {
    this.configManager = configManager;
    this.telemetrySystem = new TelemetrySystem(
      configManager.getSection('telemetry')
    );
  }

  /**
   * Create orchestration system with feature-flag driven configuration
   */
  async createOrchestrationSystem(context: FeatureFlagContext): Promise<OrchestrationSystem> {
    const config = this.configManager.getConfig();
    
    // Build configuration based on active features
    const activeFeatures = this.configManager.getActiveFeatures(context);
    
    // Create orchestration system with dynamic configuration
    const orchestrationConfig = {
      ...config,
      // Override based on feature flags
      rerank: {
        ...config.rerank,
        use_llm: activeFeatures.enableLLMReranking,
        topk_out: activeFeatures.enableSemanticDiversification ? 
          Math.ceil(config.rerank.topk_out * 1.2) : // More results for diversification
          config.rerank.topk_out
      },
      diversify: {
        ...config.diversify,
        method: activeFeatures.enableSemanticDiversification ? 'semantic' : 'entity'
      },
      contradiction: {
        ...config.contradiction,
        enabled: activeFeatures.enableContradictionDetection
      }
    };

    return new OrchestrationSystem(orchestrationConfig);
  }

  /**
   * Get performance-aware configuration
   */
  getPerformanceAwareConfig(currentMetrics?: PerformanceMetrics): Partial<LetheConfig> {
    const config = this.configManager.getConfig();
    const performanceConfig = config.performance;
    
    if (!currentMetrics) {
      return config;
    }

    // Adaptive configuration based on current performance
    const adaptations: Partial<LetheConfig> = {};

    // If we're approaching memory limits, reduce batch sizes
    if (currentMetrics.memory_rss_mb > performanceConfig.maxMemoryMB * 0.8) {
      adaptations.rerank = {
        ...config.rerank,
        batch_size: Math.max(1, Math.floor(config.rerank.batch_size * 0.5)),
        llm_batch_size: Math.max(1, Math.floor((config.rerank.llm_batch_size || 5) * 0.5))
      };
    }

    // If latency is high, disable expensive features
    if (currentMetrics.latency_p95 > performanceConfig.targetP95Ms) {
      adaptations.features = {
        ...config.features,
        enableSemanticDiversification: false,
        enableLLMReranking: false
      };
    }

    // If CPU usage is high, reduce chunking complexity
    if (currentMetrics.cpu_usage_percent > 80) {
      adaptations.chunking = {
        ...config.chunking,
        strategy: 'basic', // Fall back to basic chunking
        target_tokens: Math.floor(config.chunking.target_tokens * 0.7)
      };
    }

    return adaptations;
  }
}

/**
 * Circuit breaker for resilient operation
 */
export class LetheCircuitBreaker {
  private configManager: ConfigurationManager;
  private failureCount: number = 0;
  private lastFailureTime: number = 0;
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';
  private readonly resetTimeoutMs: number = 60000; // 1 minute

  constructor(configManager: ConfigurationManager) {
    this.configManager = configManager;
  }

  /**
   * Execute operation with circuit breaker protection
   */
  async execute<T>(
    operation: () => Promise<Result<T, LetheError>>,
    context: FeatureFlagContext
  ): Promise<Result<T, LetheError>> {
    if (!isFeatureEnabled('enableCircuitBreaker', context)) {
      return await operation();
    }

    const performanceConfig = this.configManager.getSection('performance');
    const threshold = performanceConfig.circuitBreakerThreshold;

    // Check circuit state
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.resetTimeoutMs) {
        this.state = 'HALF_OPEN';
      } else {
        return {
          success: false,
          error: new LetheError(
            'CIRCUIT_BREAKER_OPEN',
            'Circuit breaker is open - operation not attempted',
            { 
              state: this.state,
              failureCount: this.failureCount,
              timeSinceLastFailure: Date.now() - this.lastFailureTime
            }
          )
        };
      }
    }

    try {
      const result = await operation();
      
      if (result.success) {
        // Success - reset circuit breaker
        this.failureCount = 0;
        this.state = 'CLOSED';
      } else {
        // Failure - increment counter
        this.failureCount++;
        this.lastFailureTime = Date.now();
        
        // Check if we should open the circuit
        const failureRate = this.failureCount / (this.failureCount + 1);
        if (failureRate >= threshold) {
          this.state = 'OPEN';
        }
      }
      
      return result;
    } catch (error) {
      this.failureCount++;
      this.lastFailureTime = Date.now();
      this.state = 'OPEN';
      
      return {
        success: false,
        error: new LetheError(
          'OPERATION_FAILED',
          'Operation failed with exception',
          { originalError: error }
        )
      };
    }
  }

  /**
   * Get circuit breaker status
   */
  getStatus() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      lastFailureTime: this.lastFailureTime
    };
  }
}

/**
 * Dynamic configuration manager that responds to runtime conditions
 */
export class DynamicConfigurationManager {
  private configManager: ConfigurationManager;
  private telemetrySystem: TelemetrySystem;
  private adaptationInterval: NodeJS.Timeout | null = null;

  constructor(configManager: ConfigurationManager) {
    this.configManager = configManager;
    this.telemetrySystem = new TelemetrySystem(
      configManager.getSection('telemetry')
    );
  }

  /**
   * Start adaptive configuration management
   */
  startAdaptiveManagement(intervalMs: number = 30000): void {
    if (this.adaptationInterval) {
      return; // Already running
    }

    this.adaptationInterval = setInterval(async () => {
      await this.adaptConfigurationBasedOnMetrics();
    }, intervalMs);
  }

  /**
   * Stop adaptive configuration management
   */
  stopAdaptiveManagement(): void {
    if (this.adaptationInterval) {
      clearInterval(this.adaptationInterval);
      this.adaptationInterval = null;
    }
  }

  /**
   * Adapt configuration based on current metrics
   */
  private async adaptConfigurationBasedOnMetrics(): Promise<void> {
    try {
      // Get recent performance metrics
      const recentMetrics = await this.telemetrySystem.getRecentMetrics(5 * 60 * 1000); // 5 minutes
      
      if (recentMetrics.length === 0) {
        return;
      }

      const avgMetrics = this.calculateAverageMetrics(recentMetrics);
      const performanceConfig = this.configManager.getSection('performance');
      
      // Determine if adaptation is needed
      const adaptations: Partial<LetheConfig> = {};
      let needsUpdate = false;

      // Memory pressure adaptation
      if (avgMetrics.memory_rss_mb > performanceConfig.maxMemoryMB * 0.9) {
        adaptations.chunking = {
          target_tokens: Math.floor(
            this.configManager.getSection('chunking').target_tokens * 0.8
          )
        };
        adaptations.rerank = {
          batch_size: Math.max(1, 
            Math.floor(this.configManager.getSection('rerank').batch_size * 0.7)
          )
        };
        needsUpdate = true;
      }

      // Latency adaptation
      if (avgMetrics.latency_p95 > performanceConfig.targetP95Ms * 1.1) {
        adaptations.features = {
          enableSemanticDiversification: false,
          enableLLMReranking: false
        };
        needsUpdate = true;
      }

      // Apply adaptations if needed
      if (needsUpdate) {
        const result = await this.configManager.updateConfig(adaptations);
        if (result.success) {
          console.log('Configuration adapted based on performance metrics', adaptations);
        } else {
          console.error('Failed to adapt configuration:', result.error);
        }
      }
    } catch (error) {
      console.error('Error in adaptive configuration management:', error);
    }
  }

  /**
   * Calculate average metrics from a set of performance measurements
   */
  private calculateAverageMetrics(metrics: PerformanceMetrics[]): PerformanceMetrics {
    const avg = {
      latency_p50: 0,
      latency_p95: 0,
      memory_rss_mb: 0,
      cpu_usage_percent: 0,
      params_count: 0,
      flops_count: 0,
      timestamp: Date.now()
    };

    for (const metric of metrics) {
      avg.latency_p50 += metric.latency_p50;
      avg.latency_p95 += metric.latency_p95;
      avg.memory_rss_mb += metric.memory_rss_mb;
      avg.cpu_usage_percent += metric.cpu_usage_percent;
      avg.params_count += metric.params_count || 0;
      avg.flops_count += metric.flops_count || 0;
    }

    const count = metrics.length;
    avg.latency_p50 /= count;
    avg.latency_p95 /= count;
    avg.memory_rss_mb /= count;
    avg.cpu_usage_percent /= count;
    avg.params_count /= count;
    avg.flops_count /= count;

    return avg;
  }
}

/**
 * Complete example usage
 */
export async function demonstrateConfigurationIntegration() {
  console.log('🚀 Starting Lethe Configuration Integration Demo');

  // 1. Setup environment-aware configuration
  const configManager = setupConfigurationForEnvironment('production');
  console.log('✅ Configuration manager initialized for production');

  // 2. Create feature context
  const context: FeatureFlagContext = {
    sessionId: 'demo-session-001',
    environment: 'production',
    version: '1.0.0',
    userSegment: 'premium',
    customAttributes: {
      region: 'us-east-1',
      tier: 'enterprise'
    }
  };

  // 3. Create component factory
  const factory = new LetheComponentFactory(configManager);
  const orchestrationSystem = await factory.createOrchestrationSystem(context);
  console.log('✅ Orchestration system created with feature-aware configuration');

  // 4. Setup circuit breaker
  const circuitBreaker = new LetheCircuitBreaker(configManager);
  console.log('✅ Circuit breaker initialized');

  // 5. Demonstrate feature flag evaluation
  const activeFeatures = configManager.getActiveFeatures(context);
  console.log('🎛️  Active Features:', Object.entries(activeFeatures)
    .filter(([_, enabled]) => enabled)
    .map(([name, _]) => name)
  );

  // 6. Setup dynamic configuration management
  const dynamicManager = new DynamicConfigurationManager(configManager);
  dynamicManager.startAdaptiveManagement(10000); // 10 second intervals
  console.log('✅ Dynamic configuration management started');

  // 7. Demonstrate configuration updates
  console.log('\n🔧 Testing configuration update...');
  const updateResult = await configManager.updateConfig({
    performance: {
      targetP50Ms: 1800 // Stricter target
    }
  });

  if (updateResult.success) {
    console.log('✅ Configuration updated successfully');
  } else {
    console.log('❌ Configuration update failed:', updateResult.error);
  }

  // 8. Demonstrate circuit breaker
  console.log('\n🛡️  Testing circuit breaker...');
  const testOperation = async (): Promise<Result<string, LetheError>> => {
    // Simulate a potentially failing operation
    if (Math.random() < 0.3) {
      return {
        success: false,
        error: new LetheError('SIMULATED_FAILURE', 'Simulated operation failure')
      };
    }
    return { success: true, data: 'Operation succeeded' };
  };

  for (let i = 0; i < 5; i++) {
    const result = await circuitBreaker.execute(testOperation, context);
    console.log(`Attempt ${i + 1}:`, result.success ? '✅ Success' : '❌ Failed');
  }

  console.log('🔍 Circuit breaker status:', circuitBreaker.getStatus());

  // 9. Show configuration change history
  const changeHistory = configManager.getChangeHistory(10);
  if (changeHistory.length > 0) {
    console.log('\n📋 Recent configuration changes:');
    changeHistory.forEach(change => {
      console.log(`  ${change.field}: ${change.oldValue} → ${change.newValue}`);
    });
  }

  // 10. Export current configuration
  const configExport = configManager.exportConfig();
  console.log('\n📤 Configuration exported (length):', configExport.length, 'characters');

  // Cleanup
  dynamicManager.stopAdaptiveManagement();
  console.log('\n🏁 Demo completed successfully');
}

// Example of A/B testing with feature flags
export class ABTestingManager {
  private configManager: ConfigurationManager;

  constructor(configManager: ConfigurationManager) {
    this.configManager = configManager;
  }

  /**
   * Setup A/B test for feature rollout
   */
  async setupABTest(
    featureName: keyof import('../config.js').FeatureFlags,
    controlPercentage: number = 50
  ): Promise<Result<void, LetheError>> {
    const updateResult = await this.configManager.updateConfig({
      features: {
        rolloutPercentage: {
          [featureName]: controlPercentage
        }
      }
    });

    return updateResult;
  }

  /**
   * Evaluate user's test group
   */
  getUserTestGroup(
    featureName: keyof import('../config.js').FeatureFlags,
    context: FeatureFlagContext
  ): 'control' | 'treatment' {
    const isEnabled = this.configManager.evaluateFeature(featureName, context);
    return isEnabled ? 'treatment' : 'control';
  }
}

// Run demo if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstrateConfigurationIntegration().catch(console.error);
}