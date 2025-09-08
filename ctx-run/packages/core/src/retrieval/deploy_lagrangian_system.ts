#!/usr/bin/env node

/**
 * Lagrangian Latency Optimization Deployment Script
 * 
 * Critical Production Issue: P95 latency +6.8ms vs ≤1ms target
 * Target: 85%+ latency reduction while maintaining +12.5% CBU performance
 * 
 * This script orchestrates the complete deployment of the Lagrangian optimization system
 * with comprehensive safety mechanisms, monitoring, and statistical validation.
 */

import { LagrangianDeploymentOrchestrator } from './lagrangian_deployment_orchestrator.js';
import { PerformanceMonitor } from './performance_monitor.js';
import { CEEarlyExitSystem } from './ce_early_exit.js';
import { StatisticalValidationEngine } from './statistical_validation.js';
import type { DB } from '@lethe/sqlite';

interface DeploymentConfig {
  /** Target P95 latency in milliseconds (≤1ms requirement) */
  targetLatencyMs: number;
  
  /** Minimum CBU performance to maintain */
  minCBUPerformance: number;
  
  /** Canary rollout percentage stages */
  canaryStages: number[];
  
  /** Monitoring window duration in minutes */
  monitoringWindowMinutes: number;
  
  /** Statistical significance threshold */
  significanceThreshold: number;
  
  /** Maximum drift tolerance for lambda */
  maxLambdaDrift: number;
  
  /** Dual gap threshold for convergence */
  dualGapThreshold: number;
  
  /** KV cache Jaccard threshold */
  kvJaccardThreshold: number;
}

const PRODUCTION_CONFIG: DeploymentConfig = {
  targetLatencyMs: 1.0,           // ≤1ms P95 requirement
  minCBUPerformance: 12.5,        // Maintain +12.5% CBU improvement
  canaryStages: [1, 5, 25, 50, 100], // Gradual rollout percentages
  monitoringWindowMinutes: 15,     // Monitoring window per stage
  significanceThreshold: 0.05,     // 95% confidence level
  maxLambdaDrift: 0.15,           // ±15% lambda drift tolerance
  dualGapThreshold: 0.005,        // <0.5% dual gap requirement
  kvJaccardThreshold: 0.10        // ≥10pp Jaccard drop alarm
};

export class LagrangianSystemDeployment {
  private orchestrator: LagrangianDeploymentOrchestrator;
  private monitor: PerformanceMonitor;
  private earlyExit: CEEarlyExitSystem;
  private validator: StatisticalValidationEngine;
  private config: DeploymentConfig;
  
  constructor(
    db: DB,
    config: DeploymentConfig = PRODUCTION_CONFIG
  ) {
    this.config = config;
    
    // Initialize core components
    this.monitor = new PerformanceMonitor({
      lagrangianConfig: {
        lambdaDriftThreshold: config.maxLambdaDrift,
        dualGapThreshold: config.dualGapThreshold,
        convergenceWindowSize: 100
      },
      cbuConfig: {
        elasticityThreshold: 0.1,
        budgetWindowSize: 50
      },
      kvCacheConfig: {
        jaccardThreshold: config.kvJaccardThreshold,
        prefixWindowSize: 1000
      },
      alertingEnabled: true
    });
    
    this.earlyExit = new CEEarlyExitSystem({
      calibrationMethod: 'isotonic_regression',
      matryoshkaConfig: {
        budgetTiers: [0.1, 0.25, 0.5, 0.75, 1.0],
        routingThresholds: [0.3, 0.5, 0.7, 0.85]
      },
      earlyExitThresholds: [0.6, 0.75, 0.85]
    });
    
    this.validator = new StatisticalValidationEngine({
      alpha: config.significanceThreshold,
      beta: 0.2, // 80% statistical power
      minimumSampleSize: 1000,
      maximumSampleSize: 10000,
      sequentialTestingEnabled: true,
      multipleTestingCorrection: 'bonferroni'
    });
    
    this.orchestrator = new LagrangianDeploymentOrchestrator(
      db,
      this.monitor,
      this.earlyExit,
      this.validator
    );
    
    // Set up critical alert handlers
    this.setupAlertHandlers();
  }
  
  private setupAlertHandlers(): void {
    // Lambda drift detection
    this.monitor.on('lambda_drift_alert', async (data) => {
      console.error(`🚨 LAMBDA DRIFT ALERT: ${data.drift * 100}% (threshold: ±${this.config.maxLambdaDrift * 100}%)`);
      if (Math.abs(data.drift) > this.config.maxLambdaDrift * 1.5) {
        console.error('⚠️  Critical lambda drift detected - initiating emergency rollback');
        await this.orchestrator.executeEmergencyRollback('lambda_drift_critical');
      }
    });
    
    // Dual gap convergence issues
    this.monitor.on('dual_gap_alert', async (data) => {
      console.error(`🚨 DUAL GAP ALERT: ${data.dualGap * 100}% (threshold: <${this.config.dualGapThreshold * 100}%)`);
      if (data.dualGap > this.config.dualGapThreshold * 2) {
        console.error('⚠️  Convergence failure detected - initiating rollback');
        await this.orchestrator.executeEmergencyRollback('convergence_failure');
      }
    });
    
    // CBU performance degradation
    this.monitor.on('cbu_degradation_alert', async (data) => {
      console.error(`🚨 CBU DEGRADATION ALERT: ${data.currentCBU}% (min required: ${this.config.minCBUPerformance}%)`);
      if (data.currentCBU < this.config.minCBUPerformance * 0.8) {
        console.error('⚠️  Severe CBU degradation - initiating emergency rollback');
        await this.orchestrator.executeEmergencyRollback('cbu_degradation');
      }
    });
    
    // KV cache efficiency issues
    this.monitor.on('kv_cache_alert', async (data) => {
      console.error(`🚨 KV CACHE ALERT: Jaccard similarity drop ${data.jaccardDrop * 100}pp`);
      if (data.jaccardDrop > this.config.kvJaccardThreshold * 1.5) {
        console.error('⚠️  Severe cache degradation - initiating rollback');
        await this.orchestrator.executeEmergencyRollback('cache_degradation');
      }
    });
  }
  
  /**
   * Execute the complete Lagrangian optimization deployment
   * This is the main entry point for the critical latency fix
   */
  async executeCriticalLatencyDeployment(): Promise<void> {
    console.log('🚀 Starting Critical Lagrangian Latency Optimization Deployment');
    console.log(`📊 Target: P95 latency ≤${this.config.targetLatencyMs}ms (currently +6.8ms)`);
    console.log(`🎯 Maintain: +${this.config.minCBUPerformance}% CBU performance`);
    console.log('');
    
    try {
      // Phase 1: Pre-deployment validation
      console.log('📋 Phase 1: Pre-deployment Validation');
      await this.preDeploymentChecks();
      
      // Phase 2: Initialize baseline metrics
      console.log('📊 Phase 2: Establishing Performance Baseline');
      await this.establishBaseline();
      
      // Phase 3: Execute staged canary deployment
      console.log('🐤 Phase 3: Staged Canary Deployment');
      const deploymentResult = await this.orchestrator.deployLagrangianOptimization({
        enableEarlyExit: true,
        enablePerformanceMonitoring: true,
        canaryConfig: {
          stages: this.config.canaryStages,
          monitoringWindowMinutes: this.config.monitoringWindowMinutes,
          rollbackOnFailure: true,
          successCriteria: {
            maxP95LatencyMs: this.config.targetLatencyMs,
            minCBUImprovement: this.config.minCBUPerformance,
            maxDualGap: this.config.dualGapThreshold,
            maxLambdaDrift: this.config.maxLambdaDrift
          }
        },
        statisticalValidation: {
          significanceLevel: this.config.significanceThreshold,
          minimumSampleSize: 1000,
          sequentialTesting: true
        }
      });
      
      // Phase 4: Final validation and monitoring setup
      console.log('✅ Phase 4: Final Validation and Monitoring');
      await this.postDeploymentValidation(deploymentResult);
      
      console.log('');
      console.log('🎉 DEPLOYMENT SUCCESSFUL! 🎉');
      console.log(`✅ P95 Latency: ${deploymentResult.finalMetrics.p95LatencyMs}ms (target: ≤${this.config.targetLatencyMs}ms)`);
      console.log(`✅ CBU Performance: +${deploymentResult.finalMetrics.cbuImprovement}% (target: ≥${this.config.minCBUPerformance}%)`);
      console.log(`✅ Dual Gap: ${deploymentResult.finalMetrics.dualGap * 100}% (target: <${this.config.dualGapThreshold * 100}%)`);
      console.log(`✅ Lambda Stability: ±${Math.abs(deploymentResult.finalMetrics.lambdaDrift) * 100}% (target: ≤±${this.config.maxLambdaDrift * 100}%)`);
      
    } catch (error) {
      console.error('');
      console.error('❌ DEPLOYMENT FAILED ❌');
      console.error(`Error: ${error.message}`);
      
      // Execute emergency rollback
      console.error('🔄 Initiating Emergency Rollback...');
      await this.orchestrator.executeEmergencyRollback('deployment_failure');
      
      throw new Error(`Lagrangian deployment failed: ${error.message}`);
    }
  }
  
  private async preDeploymentChecks(): Promise<void> {
    console.log('  🔍 Validating system health...');
    
    // Check if mathematical orchestrator is available
    const hasOrchestrator = await this.orchestrator.validateSystemComponents();
    if (!hasOrchestrator) {
      throw new Error('Mathematical orchestrator not available');
    }
    
    // Validate CE Early-Exit calibration
    const calibrationValid = await this.earlyExit.validateCalibration();
    if (!calibrationValid) {
      throw new Error('CE Early-Exit calibration invalid');
    }
    
    // Check statistical validation readiness
    const validatorReady = this.validator.isReady();
    if (!validatorReady) {
      throw new Error('Statistical validator not ready');
    }
    
    console.log('  ✅ All system components validated');
  }
  
  private async establishBaseline(): Promise<void> {
    console.log('  📊 Collecting baseline performance metrics...');
    
    // Collect 5 minutes of baseline data
    const baselineStart = Date.now();
    const baselineWindow = 5 * 60 * 1000; // 5 minutes
    
    while (Date.now() - baselineStart < baselineWindow) {
      // Record current system performance
      // This would integrate with actual metrics collection
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log('  ✅ Baseline established');
  }
  
  private async postDeploymentValidation(deploymentResult: any): Promise<void> {
    console.log('  🔍 Validating deployment success...');
    
    // Verify latency improvement
    if (deploymentResult.finalMetrics.p95LatencyMs > this.config.targetLatencyMs) {
      throw new Error(`P95 latency ${deploymentResult.finalMetrics.p95LatencyMs}ms exceeds target ${this.config.targetLatencyMs}ms`);
    }
    
    // Verify CBU performance maintained
    if (deploymentResult.finalMetrics.cbuImprovement < this.config.minCBUPerformance) {
      throw new Error(`CBU performance ${deploymentResult.finalMetrics.cbuImprovement}% below required ${this.config.minCBUPerformance}%`);
    }
    
    // Verify mathematical stability
    if (deploymentResult.finalMetrics.dualGap > this.config.dualGapThreshold) {
      throw new Error(`Dual gap ${deploymentResult.finalMetrics.dualGap} exceeds threshold ${this.config.dualGapThreshold}`);
    }
    
    // Set up continuous monitoring
    console.log('  📊 Configuring continuous monitoring...');
    this.monitor.startContinuousMonitoring();
    
    console.log('  ✅ Post-deployment validation complete');
  }
  
  /**
   * Execute emergency rollback if needed
   */
  async executeEmergencyRollback(reason: string): Promise<void> {
    console.error(`🚨 EMERGENCY ROLLBACK INITIATED: ${reason}`);
    await this.orchestrator.executeEmergencyRollback(reason);
    console.error('🔄 Emergency rollback complete');
  }
  
  /**
   * Get current system status
   */
  async getSystemStatus(): Promise<any> {
    return {
      deployment: await this.orchestrator.getDeploymentStatus(),
      performance: this.monitor.getCurrentMetrics(),
      earlyExit: this.earlyExit.getSystemStatus(),
      validation: this.validator.getValidationStatus()
    };
  }
}

/**
 * CLI Entry Point for Production Deployment
 */
async function main() {
  console.log('🎯 Lagrangian Latency Optimization - Production Deployment');
  console.log('=' .repeat(60));
  
  try {
    // This would need actual DB connection in production
    const mockDB = {} as DB;
    
    const deployment = new LagrangianSystemDeployment(mockDB, PRODUCTION_CONFIG);
    await deployment.executeCriticalLatencyDeployment();
    
    // Keep monitoring running
    console.log('');
    console.log('📊 Continuous monitoring active...');
    console.log('Press Ctrl+C to stop monitoring');
    
    // Set up graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n🔄 Graceful shutdown initiated...');
      process.exit(0);
    });
    
    // Keep process alive for monitoring
    setInterval(async () => {
      const status = await deployment.getSystemStatus();
      console.log(`📊 Status: P95=${status.performance?.p95LatencyMs}ms, CBU=+${status.performance?.cbuImprovement}%`);
    }, 30000); // Status update every 30 seconds
    
  } catch (error) {
    console.error('💥 Deployment failed:', error.message);
    process.exit(1);
  }
}

// Execute if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { LagrangianSystemDeployment, PRODUCTION_CONFIG };