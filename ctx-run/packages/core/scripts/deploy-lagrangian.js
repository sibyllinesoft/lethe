#!/usr/bin/env node

/**
 * Command Line Interface for Lagrangian Latency Optimization Deployment
 * 
 * Critical Production Issue: P95 latency +6.8ms vs ≤1ms target
 * 
 * Usage:
 *   npm run deploy:lagrangian
 *   node scripts/deploy-lagrangian.js
 *   node scripts/deploy-lagrangian.js --config production
 *   node scripts/deploy-lagrangian.js --dry-run
 */

import { LagrangianSystemDeployment, PRODUCTION_CONFIG } from '../src/retrieval/deploy_lagrangian_system.js';
import { createSQLiteDB } from '../src/state/index.js';
import { performance } from 'perf_hooks';
import process from 'process';

// CLI Arguments
const args = process.argv.slice(2);
const isDryRun = args.includes('--dry-run');
const configMode = args.find(arg => arg.startsWith('--config='))?.split('=')[1] || 'production';
const verbose = args.includes('--verbose');

// Configuration variants
const CONFIGS = {
  production: PRODUCTION_CONFIG,
  staging: {
    ...PRODUCTION_CONFIG,
    canaryStages: [10, 50, 100], // Faster rollout for staging
    monitoringWindowMinutes: 5,   // Shorter monitoring window
    targetLatencyMs: 2.0         // Less aggressive target for staging
  },
  development: {
    ...PRODUCTION_CONFIG,
    canaryStages: [50, 100],     // Very fast rollout for dev
    monitoringWindowMinutes: 2,   // Minimal monitoring
    targetLatencyMs: 5.0,        // Relaxed target for dev
    significanceThreshold: 0.1    // Lower confidence for faster testing
  }
};

function printHeader() {
  console.log('');
  console.log('🎯 LAGRANGIAN LATENCY OPTIMIZATION DEPLOYMENT');
  console.log('=' .repeat(80));
  console.log(`📊 Current Issue: P95 latency +6.8ms vs ≤1ms target`);
  console.log(`🎯 Target: 85%+ latency reduction while maintaining +12.5% CBU performance`);
  console.log(`⚙️  Configuration: ${configMode.toUpperCase()}`);
  console.log(`🧪 Mode: ${isDryRun ? 'DRY RUN (simulation only)' : 'LIVE DEPLOYMENT'}`);
  console.log('=' .repeat(80));
  console.log('');
}

function printHelp() {
  console.log('Lagrangian Latency Optimization Deployment CLI');
  console.log('');
  console.log('Usage:');
  console.log('  node scripts/deploy-lagrangian.js [options]');
  console.log('');
  console.log('Options:');
  console.log('  --config=MODE        Configuration mode (production, staging, development)');
  console.log('  --dry-run           Simulate deployment without making changes');
  console.log('  --verbose           Enable verbose logging');
  console.log('  --help              Show this help message');
  console.log('');
  console.log('Examples:');
  console.log('  node scripts/deploy-lagrangian.js --config=production');
  console.log('  node scripts/deploy-lagrangian.js --dry-run --verbose');
  console.log('  npm run deploy:lagrangian');
}

async function createDatabase() {
  try {
    // Initialize SQLite database
    const db = await createSQLiteDB(':memory:'); // Use in-memory for deployment script
    if (verbose) {
      console.log('✅ Database connection established');
    }
    return db;
  } catch (error) {
    console.error('❌ Failed to create database connection:', error.message);
    throw error;
  }
}

async function simulateDeployment(deployment, config) {
  console.log('🧪 SIMULATION MODE - No actual changes will be made');
  console.log('');
  
  const startTime = performance.now();
  
  try {
    console.log('📋 Phase 1: Pre-deployment Validation (Simulated)');
    console.log('  ✅ System components validated');
    console.log('  ✅ CE Early-Exit calibration verified');
    console.log('  ✅ Statistical validator ready');
    console.log('');
    
    console.log('📊 Phase 2: Baseline Establishment (Simulated)');
    console.log('  📊 Current P95 latency: 7.8ms (baseline +6.8ms over target)');
    console.log('  📊 Current CBU performance: +12.5%');
    console.log('  ✅ Baseline metrics established');
    console.log('');
    
    console.log('🐤 Phase 3: Staged Canary Deployment (Simulated)');
    for (let i = 0; i < config.canaryStages.length; i++) {
      const stage = config.canaryStages[i];
      console.log(`  🐤 Stage ${i + 1}: Rolling out to ${stage}% of traffic`);
      
      // Simulate monitoring window
      for (let j = 0; j < config.monitoringWindowMinutes; j++) {
        await new Promise(resolve => setTimeout(resolve, 200)); // Fast simulation
        const progress = Math.round((j + 1) / config.monitoringWindowMinutes * 100);
        if (verbose || j === config.monitoringWindowMinutes - 1) {
          console.log(`    📊 Monitoring ${progress}%: P95=${(7.8 - (i + 1) * 1.2).toFixed(1)}ms, CBU=+${12.5 + i * 0.2}%`);
        }
      }
      
      console.log(`    ✅ Stage ${i + 1} successful - metrics within targets`);
    }
    console.log('');
    
    console.log('✅ Phase 4: Final Validation (Simulated)');
    const finalLatency = 0.8; // Simulated final latency
    const finalCBU = 12.7; // Simulated final CBU performance
    
    console.log(`  ✅ Final P95 latency: ${finalLatency}ms (target: ≤${config.targetLatencyMs}ms)`);
    console.log(`  ✅ Final CBU performance: +${finalCBU}% (target: ≥${config.minCBUPerformance}%)`);
    console.log(`  ✅ Dual gap: 0.002% (target: <${config.dualGapThreshold * 100}%)`);
    console.log(`  ✅ Lambda stability: ±3% (target: ≤±${config.maxLambdaDrift * 100}%)`);
    console.log('');
    
    const deploymentTime = performance.now() - startTime;
    console.log('🎉 SIMULATION SUCCESSFUL! 🎉');
    console.log(`⏱️  Total deployment time: ${Math.round(deploymentTime)}ms`);
    console.log(`✅ Projected latency improvement: ${((7.8 - finalLatency) / 7.8 * 100).toFixed(1)}% reduction`);
    console.log(`✅ CBU performance maintained: +${finalCBU}%`);
    
  } catch (error) {
    console.error('❌ Simulation failed:', error.message);
    throw error;
  }
}

async function executeLiveDeployment(deployment) {
  console.log('🚀 LIVE DEPLOYMENT - Making actual system changes');
  console.log('⚠️  This will modify the production system');
  console.log('');
  
  // In a real deployment, we'd execute the actual deployment
  await deployment.executeCriticalLatencyDeployment();
}

async function main() {
  // Handle help command
  if (args.includes('--help') || args.includes('-h')) {
    printHelp();
    return;
  }
  
  printHeader();
  
  try {
    // Get configuration
    const config = CONFIGS[configMode];
    if (!config) {
      throw new Error(`Unknown configuration mode: ${configMode}. Available: ${Object.keys(CONFIGS).join(', ')}`);
    }
    
    if (verbose) {
      console.log('📋 Configuration Details:');
      console.log(`  Target P95 Latency: ≤${config.targetLatencyMs}ms`);
      console.log(`  Min CBU Performance: +${config.minCBUPerformance}%`);
      console.log(`  Canary Stages: ${config.canaryStages.join('% → ')}%`);
      console.log(`  Monitoring Window: ${config.monitoringWindowMinutes} minutes per stage`);
      console.log(`  Statistical Significance: ${config.significanceThreshold}`);
      console.log('');
    }
    
    // Initialize database
    console.log('🔌 Initializing database connection...');
    const db = await createDatabase();
    
    // Initialize deployment system
    console.log('⚙️  Initializing Lagrangian deployment system...');
    const deployment = new LagrangianSystemDeployment(db, config);
    
    if (verbose) {
      console.log('✅ Deployment system initialized');
      console.log('  📊 Performance monitor: Active');
      console.log('  🧠 CE Early-Exit system: Ready');
      console.log('  📈 Statistical validator: Configured');
      console.log('  🎛️  Deployment orchestrator: Standby');
      console.log('');
    }
    
    // Execute deployment or simulation
    if (isDryRun) {
      await simulateDeployment(deployment, config);
    } else {
      console.log('⚠️  WARNING: This is a LIVE deployment that will modify the system.');
      console.log('⚠️  Ensure you have proper backups and rollback procedures ready.');
      console.log('');
      
      // In production, you might want to add a confirmation prompt here
      await executeLiveDeployment(deployment);
    }
    
    console.log('');
    console.log('📋 Deployment Summary:');
    console.log(`  Configuration: ${configMode}`);
    console.log(`  Mode: ${isDryRun ? 'Simulation' : 'Live Deployment'}`);
    console.log(`  Target: P95 ≤${config.targetLatencyMs}ms, CBU ≥+${config.minCBUPerformance}%`);
    console.log(`  Status: ✅ SUCCESS`);
    
  } catch (error) {
    console.error('');
    console.error('💥 DEPLOYMENT FAILED 💥');
    console.error(`❌ Error: ${error.message}`);
    console.error('');
    
    if (verbose && error.stack) {
      console.error('Stack trace:');
      console.error(error.stack);
    }
    
    console.error('🔄 Recommendations:');
    console.error('  1. Check system logs for detailed error information');
    console.error('  2. Verify all prerequisites are met');
    console.error('  3. Consider running with --dry-run first');
    console.error('  4. Contact system administrator if issues persist');
    
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('');
  console.log('🔄 Deployment interrupted by user');
  console.log('⚠️  If this was a live deployment, check system status');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('');
  console.log('🔄 Deployment terminated');
  process.exit(0);
});

// Execute main function
main().catch((error) => {
  console.error('💥 Unexpected error:', error);
  process.exit(1);
});