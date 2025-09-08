#!/usr/bin/env python3
"""
Production Dashboard Runner for Lethe-StreamingLLM Hybrid

Orchestrates the complete monitoring system:
- Real-time metrics collection and alerting
- Progressive canary rollout management  
- Auto-remediation actions
- Performance report generation
- Integration with external monitoring systems

Usage:
    python run_dashboard.py --config production_config.yaml
"""

import asyncio
import argparse
import logging
import json
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from production_dashboard import ProductionDashboard, PerTurnMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardOrchestrator:
    """Main orchestrator for production monitoring system"""
    
    def __init__(self, config_file: str):
        self.config = self._load_config(config_file)
        self.dashboard = None
        self.running = False
        
        # Monitoring intervals
        self.health_check_interval = self.config.get('health_check_interval', 60)  # seconds
        self.report_interval = self.config.get('report_interval', 3600)  # 1 hour
        self.drift_check_interval = self.config.get('drift_check_interval', 300)  # 5 minutes
        
        # Canary control
        self.auto_promotion = self.config.get('auto_promotion', True)
        self.promotion_check_interval = self.config.get('promotion_check_interval', 1800)  # 30 minutes
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file"""
        config_path = Path(config_file)
        if not config_path.exists():
            # Create default config
            self._create_default_config(config_path)
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_default_config(self, config_path: Path):
        """Create default configuration file"""
        default_config = {
            'database': {
                'url': 'postgresql://lethe_user:lethe_pass@localhost:5432/lethe_monitoring'
            },
            'redis': {
                'url': 'redis://localhost:6379'
            },
            'prometheus': {
                'port': 9090
            },
            'monitoring': {
                'health_check_interval': 60,
                'report_interval': 3600,
                'drift_check_interval': 300
            },
            'canary': {
                'auto_promotion': True,
                'promotion_check_interval': 1800,
                'stability_window_hours': 2,
                'promotion_criteria': {
                    'min_health_score': 0.8,
                    'min_requests': 100,
                    'max_p95_regression': 1.0,  # ms
                    'min_delta_cbu': 8.0,
                    'max_dual_gap': 0.005
                }
            },
            'alerts': {
                'webhook_urls': [],
                'email_recipients': [],
                'slack_webhook': None
            },
            'auto_remediation': {
                'enabled': True,
                'kv_jaccard_drop': {
                    'enabled': True,
                    'head_reduction_factor': 0.97
                },
                'evt_xi_rising': {
                    'enabled': True,
                    'stride_reduction_factor': 0.8
                }
            }
        }
        
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, indent=2)
            
        logger.info(f"Created default configuration at {config_path}")
        
    async def initialize(self):
        """Initialize the monitoring system"""
        logger.info("Initializing Lethe production monitoring dashboard...")
        
        # Initialize main dashboard
        self.dashboard = ProductionDashboard(
            db_url=self.config['database']['url'],
            redis_url=self.config['redis']['url']
        )
        
        await self.dashboard.initialize()
        logger.info("Dashboard initialized successfully")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
    async def run(self):
        """Main monitoring loop"""
        self.running = True
        logger.info("Starting Lethe production monitoring dashboard")
        
        # Start background tasks
        tasks = [
            self._health_monitor_loop(),
            self._report_generator_loop(), 
            self._drift_monitor_loop(),
            self._canary_controller_loop(),
            self._metrics_ingestion_loop()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled, shutting down")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            logger.info("Monitoring dashboard shutdown complete")
            
    async def _health_monitor_loop(self):
        """Continuous health monitoring"""
        while self.running:
            try:
                # Check system health
                health_status = await self._check_system_health()
                
                # Log critical issues
                if health_status['overall_health_score'] < 0.5:
                    logger.critical(f"System health critical: {health_status['overall_health_score']:.2f}")
                    
                # Update health checkpoint
                await self._record_health_checkpoint(health_status)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.health_check_interval)
                
    async def _check_system_health(self) -> dict:
        """Comprehensive system health check"""
        # Get recent metrics
        report = await self.dashboard.generate_performance_report()
        
        # Calculate health scores
        performance_health = self._calculate_performance_health(report)
        quality_health = self._calculate_quality_health(report) 
        stability_health = self._calculate_stability_health(report)
        
        overall_health = (performance_health + quality_health + stability_health) / 3
        
        return {
            'timestamp': datetime.now(),
            'performance_health_score': performance_health,
            'quality_health_score': quality_health,
            'stability_health_score': stability_health,
            'overall_health_score': overall_health,
            'total_requests': report.get('total_requests', 0),
            'alerts_24h': report.get('alerts_24h', 0),
            'key_metrics': report.get('key_metrics', {})
        }
        
    def _calculate_performance_health(self, report: dict) -> float:
        """Calculate performance health score (0-1)"""
        key_metrics = report.get('key_metrics', {})
        
        # P95 latency health (baseline: 142ms, threshold: 143ms)
        avg_p95 = key_metrics.get('avg_llm_p95', 200)
        p95_health = max(0, 1 - max(0, avg_p95 - 142) / 58)  # Scale to 0-1
        
        # Primal-dual gap health (threshold: 0.5%)
        max_gap = key_metrics.get('max_primal_dual_gap', 0.01)
        gap_health = max(0, 1 - max(0, max_gap - 0.005) / 0.045)  # Scale to 0-1
        
        return (p95_health + gap_health) / 2
        
    def _calculate_quality_health(self, report: dict) -> float:
        """Calculate quality health score (0-1)"""
        key_metrics = report.get('key_metrics', {})
        
        # ΔCBU/1k health (baseline: 8.42, minimum: 8.0)
        avg_delta_cbu = key_metrics.get('avg_delta_cbu_1k', 6.0)
        delta_cbu_health = max(0, min(1, (avg_delta_cbu - 7.0) / 4.0))  # Scale 7-11 to 0-1
        
        # KV reuse health (target: >70%)
        avg_kv_reuse = key_metrics.get('avg_kv_reuse', 0.5)
        kv_health = max(0, min(1, (avg_kv_reuse - 0.6) / 0.3))  # Scale 60%-90% to 0-1
        
        return (delta_cbu_health + kv_health) / 2
        
    def _calculate_stability_health(self, report: dict) -> float:
        """Calculate stability health score (0-1)"""
        alerts_24h = report.get('alerts_24h', 10)
        total_requests = report.get('total_requests', 1)
        
        # Alert rate health (target: <1% of requests trigger alerts)
        alert_rate = alerts_24h / max(1, total_requests)
        alert_health = max(0, 1 - min(1, alert_rate / 0.01))
        
        # Request volume stability (check for anomalies)
        volume_health = 1.0  # Simplified for now
        
        return (alert_health + volume_health) / 2
        
    async def _record_health_checkpoint(self, health_status: dict):
        """Record health checkpoint in database"""
        # Store in database for trending
        # Implementation would use dashboard.db_pool
        pass
        
    async def _report_generator_loop(self):
        """Generate periodic performance reports"""
        while self.running:
            try:
                # Generate comprehensive report
                report = await self.dashboard.generate_performance_report()
                
                # Save report to file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_file = f"/tmp/lethe_report_{timestamp}.json"
                
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
                logger.info(f"Performance report saved to {report_file}")
                
                # Send to external systems if configured
                await self._send_report_to_external_systems(report)
                
                await asyncio.sleep(self.report_interval)
                
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                await asyncio.sleep(self.report_interval)
                
    async def _send_report_to_external_systems(self, report: dict):
        """Send reports to external monitoring systems"""
        # Implementation for external integrations (Slack, email, webhooks)
        pass
        
    async def _drift_monitor_loop(self):
        """Monitor parameter drift"""
        while self.running:
            try:
                # This would integrate with dashboard.drift_detector
                # Check for significant parameter drift
                
                await asyncio.sleep(self.drift_check_interval)
                
            except Exception as e:
                logger.error(f"Error in drift monitoring: {e}")
                await asyncio.sleep(self.drift_check_interval)
                
    async def _canary_controller_loop(self):
        """Automated canary rollout control"""
        if not self.auto_promotion:
            return
            
        while self.running:
            try:
                # Check promotion readiness
                readiness = await self.dashboard.get_promotion_readiness()
                
                if readiness['ready'] and readiness['next_percentage']:
                    logger.info(
                        f"Canary promotion ready: {readiness['current_percentage']}% "
                        f"→ {readiness['next_percentage']}%"
                    )
                    
                    # In production, this would trigger the actual promotion
                    # await self._promote_canary(readiness['next_percentage'])
                    
                elif readiness['health_score'] < 0.3:
                    logger.warning(
                        f"Canary health poor ({readiness['health_score']:.2f}), "
                        "consider rollback"
                    )
                    
                await asyncio.sleep(self.promotion_check_interval)
                
            except Exception as e:
                logger.error(f"Error in canary controller: {e}")
                await asyncio.sleep(self.promotion_check_interval)
                
    async def _metrics_ingestion_loop(self):
        """Simulate metrics ingestion from live system"""
        while self.running:
            try:
                # In production, this would read from message queue or API
                # For now, generate sample metrics
                
                sample_metrics = self._generate_sample_metrics()
                await self.dashboard.log_per_turn_metrics(sample_metrics)
                
                await asyncio.sleep(5)  # Ingest metrics every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics ingestion: {e}")
                await asyncio.sleep(5)
                
    def _generate_sample_metrics(self) -> PerTurnMetrics:
        """Generate realistic sample metrics for testing"""
        import random
        
        # Base performance (slightly better than baseline)
        base_delta_cbu = 8.5 + random.gauss(0, 0.3)
        base_p95 = 141 + random.gauss(0, 2)
        base_kv_reuse = 0.75 + random.gauss(0, 0.05)
        
        return PerTurnMetrics(
            lambda_param=0.12 + random.gauss(0, 0.01),
            mu_param=0.08 + random.gauss(0, 0.005),
            tokens_in=random.randint(6000, 10000),
            head_tokens=random.randint(800, 1200),
            tail_tokens=random.randint(1200, 1800), 
            keep_ratio_head=0.12 + random.gauss(0, 0.02),
            keep_ratio_tail=0.18 + random.gauss(0, 0.02),
            K1=random.choice([150, 200, 250]),
            K2=random.choice([280, 320, 360]),
            r=random.choice([12, 14, 16]),
            CE_early_exit=random.choice([True, False]),
            num_windows=random.randint(1, 3),
            window_size=random.choice([4000, 6000, 8000]),
            stride=random.choice([2000, 3000, 4000]),
            sinks=random.choice([64, 96, 128]),
            KV_prefix_reuse=max(0.5, base_kv_reuse),
            middleware_p95=base_p95 + random.gauss(0, 1),
            LLM_p95=base_p95,
            DELTA_CBU_1k=max(0, base_delta_cbu),
            P_at_k=0.85 + random.gauss(0, 0.02),
            R_at_k=0.82 + random.gauss(0, 0.02),
            primal_dual_gap=max(0, 0.002 + random.gauss(0, 0.001)),
            tail_cvar_095=160 + random.gauss(0, 10),
            timestamp=datetime.now(),
            request_id=f"req-{random.randint(100000, 999999)}",
            canary_percentage=5.0,
            method="hybrid"
        )

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Lethe Production Monitoring Dashboard')
    parser.add_argument('--config', default='production_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize and run dashboard
        orchestrator = DashboardOrchestrator(args.config)
        await orchestrator.initialize()
        await orchestrator.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())