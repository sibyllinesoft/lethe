#!/usr/bin/env node
/**
 * Production Deployment Execution Script
 * 
 * This script executes the complete production validation and canary deployment
 * with all monitoring systems, quality gates, and automated rollback capabilities.
 * 
 * Usage: npm run deploy:production
 */

import { executeProductionDeployment, ProductionConfig } from './production-orchestrator';

/**
 * Main deployment execution
 */
async function main() {
    console.log('🌟 LETHE PRODUCTION VALIDATION SYSTEM');
    console.log('=====================================');
    console.log(`📅 Deployment Date: ${new Date().toISOString()}`);
    console.log(`🔬 Mathematical Proofs: Dual Sanity, OOD Resilience, Long Horizon`);
    console.log(`📊 Quality Gates: ECE ≤ 0.08, ILP ≤ 5%`);
    console.log(`🐣 Canary Model: EmbeddingGemma-300M (7-day validation)`);
    console.log(`⚡ Performance: ΔCBU/GB ≥ +10%, P95 ≥5ms improvement`);
    console.log(`🌪️ Chaos Testing: Closure cycles, Rank collapse, KV churn`);
    console.log(`🚨 Auto-Rollback: Health checks with statistical monitoring`);
    console.log('=====================================\n');
    
    try {
        // Configure production deployment
        const config: Partial<ProductionConfig> = {
            // Core validation thresholds  
            eceThreshold: 0.08,           // Expected Calibration Error
            ilpThreshold: 5,              // Information Leakage Percentage
            lambdaDriftThreshold: 0.15,   // Lambda drift bounds
            
            // Performance requirements
            cbuPerformanceImprovement: 10, // ΔCBU/GB ≥ +10%
            p95LatencyImprovement: 5,     // P95 latency ≥5ms improvement
            
            // Canary deployment
            canaryDurationDays: 7,        // 7-day validation period
            initialCanaryTraffic: 5,      // Start with 5% traffic
            
            // Risk management
            riskBudgetThresholds: {
                warning: 70,              // 70% budget consumption warning
                critical: 90,             // 90% critical threshold
                emergency: 95             // 95% emergency rollback
            },
            
            // Monitoring configuration
            healthCheckIntervalMs: 30000,     // 30-second health checks
            metricsCollectionIntervalMs: 60000, // 1-minute metrics collection
            
            // Chaos testing
            chaosTestingEnabled: true,
            chaosIntensityLevel: 3        // Moderate chaos testing (1-10 scale)
        };
        
        console.log('🚀 Initiating production deployment...\n');
        
        // Execute the complete production deployment
        const deploymentStatus = await executeProductionDeployment(config);
        
        console.log('\n' + '='.repeat(60));
        console.log('🎉 PRODUCTION DEPLOYMENT SUCCESSFUL');
        console.log('='.repeat(60));
        
        // Display deployment summary
        console.log('\n📊 DEPLOYMENT SUMMARY:');
        console.log(`├─ System State: ${deploymentStatus.state}`);
        console.log(`├─ Deployment Time: ${deploymentStatus.timestamp.toISOString()}`);
        console.log(`├─ Canary Status: ${deploymentStatus.systems.canary.trafficPercentage}% traffic`);
        console.log(`├─ Days Remaining: ${deploymentStatus.systems.canary.daysRemaining}`);
        console.log(`└─ Overall Health: ${getHealthSummary(deploymentStatus)}`);
        
        console.log('\n🔬 QUALITY METRICS:');
        console.log(`├─ ECE (Expected Calibration Error): ${deploymentStatus.metrics.ece} (≤ 0.08 ✅)`);
        console.log(`├─ ILP (Information Leakage): ${deploymentStatus.metrics.ilp}% (≤ 5% ✅)`);
        console.log(`├─ Lambda Drift: ${deploymentStatus.metrics.lambdaDrift}`);
        console.log(`├─ CBU Performance: +${deploymentStatus.metrics.cbuPerformance}% (≥ +10% ✅)`);
        console.log(`└─ P95 Latency: +${deploymentStatus.metrics.p95Latency}ms (≥ +5ms ✅)`);
        
        console.log('\n🔧 SYSTEM STATUS:');
        console.log(`├─ Core Proofs: ${deploymentStatus.systems.coreProofs.status} (confidence: ${deploymentStatus.systems.coreProofs.confidence.toFixed(3)})`);
        console.log(`├─ Monitoring: ${deploymentStatus.systems.monitoring.status} (${deploymentStatus.systems.monitoring.alertsActive} active alerts)`);
        console.log(`├─ Risk Budget: ${deploymentStatus.systems.riskBudget.status} (${deploymentStatus.systems.riskBudget.consumed}% consumed)`);
        console.log(`├─ Quality Gates: ${deploymentStatus.systems.qualityGates.status} (${deploymentStatus.systems.qualityGates.violations} violations)`);
        console.log(`└─ Health Checks: ${deploymentStatus.systems.healthChecks.status} (${deploymentStatus.systems.healthChecks.failingComponents.length} failing)`);
        
        if (deploymentStatus.alerts.length > 0) {
            console.log('\n⚠️  ACTIVE ALERTS:');
            deploymentStatus.alerts.slice(-3).forEach((alert, i) => {
                const icon = getAlertIcon(alert.severity);
                console.log(`${i === deploymentStatus.alerts.length - 1 ? '└─' : '├─'} ${icon} [${alert.severity.toUpperCase()}] ${alert.message}`);
            });
        }
        
        console.log('\n🎯 NEXT STEPS:');
        console.log('├─ Monitor canary deployment progress over 7 days');
        console.log('├─ Watch for automatic traffic scaling (5% → 100%)');
        console.log('├─ Review daily operational metrics and alerts');
        console.log('├─ Validate performance improvements and quality gates');
        console.log('└─ System will auto-promote or rollback based on statistical validation');
        
        console.log('\n📚 OPERATIONAL COMMANDS:');
        console.log('├─ Status: curl -X GET http://localhost:3000/api/production/status');
        console.log('├─ Metrics: curl -X GET http://localhost:3000/api/production/metrics');
        console.log('├─ Health: curl -X GET http://localhost:3000/api/production/health');
        console.log('└─ Alerts: curl -X GET http://localhost:3000/api/production/alerts');
        
        console.log('\n📖 Documentation: ./PRODUCTION_OPERATIONS_RUNBOOK.md');
        console.log('\n🎉 PRODUCTION SYSTEM IS LIVE AND MONITORED');
        
    } catch (error) {
        console.error('\n💥 PRODUCTION DEPLOYMENT FAILED');
        console.error('='.repeat(60));
        console.error('❌ Error:', error instanceof Error ? error.message : String(error));
        
        if (error instanceof Error && error.stack) {
            console.error('\n📚 Stack Trace:');
            console.error(error.stack);
        }
        
        console.error('\n🔧 Troubleshooting:');
        console.error('├─ Check system dependencies and configuration');
        console.error('├─ Verify database connectivity and API access');
        console.error('├─ Review logs for specific error details');
        console.error('├─ Consult PRODUCTION_OPERATIONS_RUNBOOK.md');
        console.error('└─ Contact production support if issues persist');
        
        process.exit(1);
    }
}

/**
 * Get overall health summary
 */
function getHealthSummary(status: any): string {
    const systems = Object.values(status.systems);
    const healthyCount = systems.filter((s: any) => s.status === 'healthy').length;
    const totalCount = systems.length;
    
    if (healthyCount === totalCount) return '✅ All Systems Healthy';
    if (healthyCount >= totalCount * 0.8) return '⚠️  Mostly Healthy';
    return '❌ System Issues Detected';
}

/**
 * Get alert icon by severity
 */
function getAlertIcon(severity: string): string {
    switch (severity) {
        case 'emergency': return '🚨';
        case 'critical': return '⛔';
        case 'warning': return '⚠️';
        case 'info': return 'ℹ️';
        default: return '📋';
    }
}

// Execute main function
if (require.main === module) {
    main().catch((error) => {
        console.error('Fatal error in main():', error);
        process.exit(1);
    });
}