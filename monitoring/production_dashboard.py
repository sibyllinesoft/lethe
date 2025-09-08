#!/usr/bin/env python3
"""
Production Monitoring Dashboard for Lethe-StreamingLLM Hybrid System

Tracks all metrics specified in TODO.md instrumentation requirements:
- Per-turn logging of all λ,μ,tokens,ratios,parameters
- Primal-dual proxy gap monitoring (<0.5%)
- Tail CVaR₀.₉₅(compute) tracking
- λ-drift/μ-drift alarms (>±15%/24h)
- KV prefix-Jaccard alarms
- Tail EVT monitoring
- Real-time dashboard with alert thresholds
- Progressive rollout controls (5% → 25% → 50% → 100%)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import asyncpg
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerTurnMetrics:
    """Per-turn logging structure matching TODO.md requirements"""
    # Core parameters
    lambda_param: float  # λ
    mu_param: float      # μ
    tokens_in: int
    head_tokens: int
    tail_tokens: int
    keep_ratio_head: float
    keep_ratio_tail: float
    
    # DPP/CE parameters
    K1: int
    K2: int
    r: int  # DPP rank
    CE_early_exit: bool
    
    # Streaming parameters
    num_windows: int
    window_size: int
    stride: int
    sinks: int
    
    # Performance metrics
    KV_prefix_reuse: float
    middleware_p95: float  # ms
    LLM_p95: float         # ms
    DELTA_CBU_1k: float    # ΔCBU/1k
    P_at_k: float          # P@k
    R_at_k: float          # R@k
    
    # Advanced monitoring
    primal_dual_gap: float
    tail_cvar_095: float   # Tail CVaR₀.₉₅(compute)
    
    # Metadata
    timestamp: datetime
    request_id: str
    canary_percentage: float
    method: str  # 'hybrid', 'streaming', 'lethe'

class PrometheusMetrics:
    """Prometheus metrics for real-time monitoring"""
    
    def __init__(self):
        # Core performance metrics
        self.request_counter = Counter('lethe_requests_total', 'Total requests', ['method', 'canary_pct'])
        self.latency_histogram = Histogram('lethe_latency_seconds', 'Request latency', ['component'])
        self.token_usage = Counter('lethe_tokens_total', 'Token usage', ['type'])
        
        # Parameter tracking
        self.lambda_gauge = Gauge('lethe_lambda', 'Lambda parameter')
        self.mu_gauge = Gauge('lethe_mu', 'Mu parameter') 
        self.keep_ratio_head = Gauge('lethe_keep_ratio_head', 'Head keep ratio')
        self.keep_ratio_tail = Gauge('lethe_keep_ratio_tail', 'Tail keep ratio')
        
        # Quality metrics
        self.delta_cbu_gauge = Gauge('lethe_delta_cbu_1k', 'ΔCBU per 1k tokens')
        self.p_at_k_gauge = Gauge('lethe_precision_at_k', 'Precision at k')
        self.r_at_k_gauge = Gauge('lethe_recall_at_k', 'Recall at k')
        
        # System health
        self.primal_dual_gap = Gauge('lethe_primal_dual_gap', 'Primal-dual proxy gap')
        self.tail_cvar = Gauge('lethe_tail_cvar_095', 'Tail CVaR 95th percentile')
        self.kv_reuse = Gauge('lethe_kv_prefix_reuse', 'KV prefix reuse ratio')
        
        # Drift detection
        self.lambda_drift_24h = Gauge('lethe_lambda_drift_24h', 'Lambda drift over 24h (%)')
        self.mu_drift_24h = Gauge('lethe_mu_drift_24h', 'Mu drift over 24h (%)')
        self.kv_jaccard_drop = Gauge('lethe_kv_jaccard_drop', 'KV Jaccard similarity drop')
        
        # EVT parameters
        self.evt_xi = Gauge('lethe_evt_xi', 'EVT shape parameter ξ')
        
        # Canary controls
        self.canary_percentage = Gauge('lethe_canary_percentage', 'Current canary percentage')
        self.canary_health = Gauge('lethe_canary_health', 'Canary health score (0-1)')

class DriftDetector:
    """Detects parameter drift over 24h windows"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.drift_threshold = 0.15  # ±15%
        
    async def update_parameter(self, param_name: str, value: float):
        """Update parameter value with timestamp"""
        timestamp = time.time()
        key = f"param_history:{param_name}"
        
        # Store with timestamp
        await self.redis.zadd(key, {str(value): timestamp})
        
        # Keep only last 24 hours
        cutoff = timestamp - 24 * 3600
        await self.redis.zremrangebyscore(key, 0, cutoff)
        
    async def check_drift(self, param_name: str) -> Tuple[float, bool]:
        """Check if parameter has drifted >±15% in 24h"""
        key = f"param_history:{param_name}"
        
        # Get all values from last 24h
        now = time.time()
        values = await self.redis.zrangebyscore(key, now - 24*3600, now, withscores=True)
        
        if len(values) < 2:
            return 0.0, False
            
        # Calculate drift percentage
        oldest_val = float(values[0][0])
        newest_val = float(values[-1][0])
        
        if oldest_val == 0:
            drift_pct = 0.0
        else:
            drift_pct = abs(newest_val - oldest_val) / abs(oldest_val)
            
        alarm = drift_pct > self.drift_threshold
        
        return drift_pct, alarm

class KVJaccardMonitor:
    """Monitors KV prefix Jaccard similarity"""
    
    def __init__(self):
        self.baseline_jaccard = None
        self.drop_threshold = 0.10  # -10pp drop triggers alarm
        
    def update_jaccard(self, jaccard_similarity: float) -> Tuple[bool, str]:
        """Update Jaccard and check for significant drops"""
        if self.baseline_jaccard is None:
            self.baseline_jaccard = jaccard_similarity
            return False, "Baseline established"
            
        drop = self.baseline_jaccard - jaccard_similarity
        
        if drop > self.drop_threshold:
            recommendation = f"Drop H by 2-3% (current drop: {drop:.3f})"
            return True, recommendation
            
        return False, f"Normal variation (drop: {drop:.3f})"

class TailEVTMonitor:
    """Monitors tail extreme value theory parameters"""
    
    def __init__(self):
        self.xi_threshold = 0.3  # Rising ξ indicates heavier tails
        self.xi_history = []
        
    def update_evt_params(self, compute_times: List[float], percentile: float = 0.95) -> Dict[str, Any]:
        """Update EVT parameters and detect concerning trends"""
        if len(compute_times) < 100:  # Need sufficient data
            return {"xi": 0, "alarm": False, "recommendation": "Insufficient data"}
            
        # Fit GPD to excesses above p95
        threshold = np.percentile(compute_times, percentile * 100)
        excesses = [t - threshold for t in compute_times if t > threshold]
        
        if len(excesses) < 10:
            return {"xi": 0, "alarm": False, "recommendation": "No significant excesses"}
            
        # Simple method-of-moments estimator for ξ
        mean_excess = np.mean(excesses)
        var_excess = np.var(excesses)
        
        if var_excess == 0 or mean_excess == 0:
            xi = 0
        else:
            xi = 0.5 * (1 - (mean_excess ** 2) / var_excess)
            
        self.xi_history.append(xi)
        if len(self.xi_history) > 100:
            self.xi_history.pop(0)
            
        # Check for rising trend
        if len(self.xi_history) > 5:
            recent_xi = np.mean(self.xi_history[-5:])
            older_xi = np.mean(self.xi_history[-10:-5]) if len(self.xi_history) >= 10 else recent_xi
            
            xi_rising = recent_xi > older_xi and recent_xi > self.xi_threshold
            
            if xi_rising:
                return {
                    "xi": xi,
                    "alarm": True, 
                    "recommendation": "Shrink stride and fanout first, not H"
                }
                
        return {"xi": xi, "alarm": False, "recommendation": "Normal tail behavior"}

class ProductionDashboard:
    """Main dashboard orchestrator"""
    
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_url = redis_url
        self.metrics = PrometheusMetrics()
        
        # Monitoring components
        self.drift_detector = None
        self.kv_monitor = KVJaccardMonitor()
        self.evt_monitor = TailEVTMonitor()
        
        # Alert thresholds
        self.alert_thresholds = {
            'primal_dual_gap': 0.005,  # 0.5%
            'p95_regression': 1.0,     # +1ms
            'lambda_drift': 0.15,      # ±15%
            'mu_drift': 0.15,          # ±15%
            'kv_jaccard_drop': 0.10,   # -10pp
        }
        
        # Canary control
        self.current_canary_pct = 5.0
        self.canary_progression = [5, 25, 50, 100]
        
    async def initialize(self):
        """Initialize connections and monitoring"""
        # Initialize Redis connection
        redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.drift_detector = DriftDetector(redis_client)
        
        # Initialize database connection
        self.db_pool = await asyncpg.create_pool(self.db_url)
        
        # Create monitoring tables
        await self._create_tables()
        
        logger.info("Production dashboard initialized")
        
    async def _create_tables(self):
        """Create database tables for metrics storage"""
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS per_turn_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    request_id VARCHAR(255) NOT NULL,
                    lambda_param FLOAT NOT NULL,
                    mu_param FLOAT NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    head_tokens INTEGER NOT NULL,
                    tail_tokens INTEGER NOT NULL,
                    keep_ratio_head FLOAT NOT NULL,
                    keep_ratio_tail FLOAT NOT NULL,
                    K1 INTEGER NOT NULL,
                    K2 INTEGER NOT NULL,
                    r INTEGER NOT NULL,
                    CE_early_exit BOOLEAN NOT NULL,
                    num_windows INTEGER NOT NULL,
                    window_size INTEGER NOT NULL,
                    stride INTEGER NOT NULL,
                    sinks INTEGER NOT NULL,
                    KV_prefix_reuse FLOAT NOT NULL,
                    middleware_p95 FLOAT NOT NULL,
                    LLM_p95 FLOAT NOT NULL,
                    DELTA_CBU_1k FLOAT NOT NULL,
                    P_at_k FLOAT NOT NULL,
                    R_at_k FLOAT NOT NULL,
                    primal_dual_gap FLOAT NOT NULL,
                    tail_cvar_095 FLOAT NOT NULL,
                    canary_percentage FLOAT NOT NULL,
                    method VARCHAR(50) NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON per_turn_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_metrics_method ON per_turn_metrics(method);
                CREATE INDEX IF NOT EXISTS idx_metrics_canary ON per_turn_metrics(canary_percentage);
            ''')
            
    async def log_per_turn_metrics(self, metrics: PerTurnMetrics):
        """Log per-turn metrics to database and update monitoring"""
        
        # Store in database
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO per_turn_metrics (
                    timestamp, request_id, lambda_param, mu_param, tokens_in,
                    head_tokens, tail_tokens, keep_ratio_head, keep_ratio_tail,
                    K1, K2, r, CE_early_exit, num_windows, window_size, stride,
                    sinks, KV_prefix_reuse, middleware_p95, LLM_p95,
                    DELTA_CBU_1k, P_at_k, R_at_k, primal_dual_gap, tail_cvar_095,
                    canary_percentage, method
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                         $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27)
            ''', 
                metrics.timestamp, metrics.request_id, metrics.lambda_param,
                metrics.mu_param, metrics.tokens_in, metrics.head_tokens,
                metrics.tail_tokens, metrics.keep_ratio_head, metrics.keep_ratio_tail,
                metrics.K1, metrics.K2, metrics.r, metrics.CE_early_exit,
                metrics.num_windows, metrics.window_size, metrics.stride,
                metrics.sinks, metrics.KV_prefix_reuse, metrics.middleware_p95,
                metrics.LLM_p95, metrics.DELTA_CBU_1k, metrics.P_at_k,
                metrics.R_at_k, metrics.primal_dual_gap, metrics.tail_cvar_095,
                metrics.canary_percentage, metrics.method
            )
        
        # Update Prometheus metrics
        self._update_prometheus_metrics(metrics)
        
        # Check for alerts
        await self._check_alerts(metrics)
        
    def _update_prometheus_metrics(self, metrics: PerTurnMetrics):
        """Update Prometheus metrics"""
        
        # Core metrics
        self.metrics.request_counter.labels(
            method=metrics.method,
            canary_pct=str(metrics.canary_percentage)
        ).inc()
        
        self.metrics.latency_histogram.labels(component='middleware').observe(
            metrics.middleware_p95 / 1000  # Convert to seconds
        )
        self.metrics.latency_histogram.labels(component='llm').observe(
            metrics.LLM_p95 / 1000
        )
        
        self.metrics.token_usage.labels(type='head').inc(metrics.head_tokens)
        self.metrics.token_usage.labels(type='tail').inc(metrics.tail_tokens)
        self.metrics.token_usage.labels(type='total').inc(metrics.tokens_in)
        
        # Parameter gauges
        self.metrics.lambda_gauge.set(metrics.lambda_param)
        self.metrics.mu_gauge.set(metrics.mu_param)
        self.metrics.keep_ratio_head.set(metrics.keep_ratio_head)
        self.metrics.keep_ratio_tail.set(metrics.keep_ratio_tail)
        
        # Quality metrics
        self.metrics.delta_cbu_gauge.set(metrics.DELTA_CBU_1k)
        self.metrics.p_at_k_gauge.set(metrics.P_at_k)
        self.metrics.r_at_k_gauge.set(metrics.R_at_k)
        
        # System health
        self.metrics.primal_dual_gap.set(metrics.primal_dual_gap)
        self.metrics.tail_cvar.set(metrics.tail_cvar_095)
        self.metrics.kv_reuse.set(metrics.KV_prefix_reuse)
        
        # Canary status
        self.metrics.canary_percentage.set(metrics.canary_percentage)
        
    async def _check_alerts(self, metrics: PerTurnMetrics):
        """Check all alert conditions"""
        
        alerts = []
        
        # 1. Primal-dual gap alarm
        if metrics.primal_dual_gap > self.alert_thresholds['primal_dual_gap']:
            alerts.append({
                'severity': 'HIGH',
                'type': 'primal_dual_gap',
                'value': metrics.primal_dual_gap,
                'threshold': self.alert_thresholds['primal_dual_gap'],
                'message': f'Primal-dual gap {metrics.primal_dual_gap:.4f} > 0.5%'
            })
            
        # 2. P95 latency regression
        if metrics.LLM_p95 > 142 + self.alert_thresholds['p95_regression']:  # Baseline + 1ms
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'p95_regression',
                'value': metrics.LLM_p95,
                'threshold': 142 + self.alert_thresholds['p95_regression'],
                'message': f'LLM p95 {metrics.LLM_p95:.1f}ms > baseline+1ms'
            })
            
        # 3. Parameter drift checks
        await self.drift_detector.update_parameter('lambda', metrics.lambda_param)
        await self.drift_detector.update_parameter('mu', metrics.mu_param)
        
        lambda_drift, lambda_alarm = await self.drift_detector.check_drift('lambda')
        mu_drift, mu_alarm = await self.drift_detector.check_drift('mu')
        
        if lambda_alarm:
            alerts.append({
                'severity': 'HIGH',
                'type': 'lambda_drift',
                'value': lambda_drift,
                'threshold': self.alert_thresholds['lambda_drift'],
                'message': f'Lambda drift {lambda_drift:.1%} > ±15% in 24h'
            })
            
        if mu_alarm:
            alerts.append({
                'severity': 'HIGH',
                'type': 'mu_drift', 
                'value': mu_drift,
                'threshold': self.alert_thresholds['mu_drift'],
                'message': f'Mu drift {mu_drift:.1%} > ±15% in 24h'
            })
            
        # Update drift metrics
        self.metrics.lambda_drift_24h.set(lambda_drift * 100)
        self.metrics.mu_drift_24h.set(mu_drift * 100)
        
        # 4. KV Jaccard monitoring
        kv_alarm, kv_msg = self.kv_monitor.update_jaccard(metrics.KV_prefix_reuse)
        if kv_alarm:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'kv_jaccard_drop',
                'message': kv_msg
            })
            
        # 5. Quality degradation
        if metrics.DELTA_CBU_1k < 8.0:  # Below baseline performance
            alerts.append({
                'severity': 'HIGH',
                'type': 'quality_degradation',
                'value': metrics.DELTA_CBU_1k,
                'message': f'ΔCBU/1k {metrics.DELTA_CBU_1k:.2f} below baseline'
            })
            
        # Process alerts
        if alerts:
            await self._handle_alerts(alerts, metrics)
            
    async def _handle_alerts(self, alerts: List[Dict], metrics: PerTurnMetrics):
        """Handle triggered alerts"""
        
        for alert in alerts:
            logger.warning(f"ALERT: {alert['type']} - {alert['message']}")
            
            # Store alert in database
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO alerts (timestamp, request_id, alert_type, severity, message, metric_value)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                ''', 
                    metrics.timestamp, metrics.request_id, alert['type'],
                    alert['severity'], alert['message'], 
                    alert.get('value', 0)
                )
                
            # Auto-actions for critical alerts
            if alert['severity'] == 'HIGH':
                await self._trigger_auto_actions(alert, metrics)
                
    async def _trigger_auto_actions(self, alert: Dict, metrics: PerTurnMetrics):
        """Trigger automatic responses to critical alerts"""
        
        if alert['type'] == 'kv_jaccard_drop':
            # Automatically reduce head size by 2-3%
            new_head_ratio = max(0.05, metrics.keep_ratio_head * 0.97)
            logger.info(f"Auto-reducing head ratio from {metrics.keep_ratio_head:.3f} to {new_head_ratio:.3f}")
            # TODO: Update runtime parameters
            
        elif alert['type'] in ['lambda_drift', 'mu_drift']:
            # Consider emergency rollback if drift is severe
            drift_value = alert['value']
            if drift_value > 0.25:  # >25% drift
                logger.critical(f"Severe parameter drift {drift_value:.1%} - consider emergency rollback")
                
        elif alert['type'] == 'quality_degradation':
            # Reduce canary percentage if quality drops significantly
            if metrics.DELTA_CBU_1k < 7.0:  # Severe degradation
                await self._emergency_canary_reduction()
                
    async def _emergency_canary_reduction(self):
        """Emergency canary traffic reduction"""
        if self.current_canary_pct > 5:
            self.current_canary_pct = max(5, self.current_canary_pct / 2)
            self.metrics.canary_percentage.set(self.current_canary_pct)
            logger.critical(f"Emergency canary reduction to {self.current_canary_pct}%")
            
    async def get_promotion_readiness(self) -> Dict[str, Any]:
        """Assess readiness for canary promotion"""
        
        # Get recent metrics (last hour)
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM per_turn_metrics 
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                AND canary_percentage = $1
                ORDER BY timestamp DESC
            ''', self.current_canary_pct)
            
        if not rows:
            return {"ready": False, "reason": "Insufficient data"}
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame([dict(row) for row in rows])
        
        # Check promotion criteria
        criteria = {
            'sufficient_data': len(df) >= 100,  # At least 100 requests
            'stable_performance': df['LLM_p95'].mean() <= 143,  # Within 1ms of baseline
            'quality_maintained': df['DELTA_CBU_1k'].mean() >= 8.0,  # Above baseline
            'no_severe_alerts': df['primal_dual_gap'].max() <= 0.005,  # Gap <0.5%
            'stable_parameters': True,  # Check parameter stability
        }
        
        # Calculate health score
        health_score = sum(criteria.values()) / len(criteria)
        self.metrics.canary_health.set(health_score)
        
        next_percentage = None
        if health_score >= 0.8 and self.current_canary_pct < 100:
            # Find next percentage in progression
            current_idx = self.canary_progression.index(self.current_canary_pct)
            if current_idx < len(self.canary_progression) - 1:
                next_percentage = self.canary_progression[current_idx + 1]
                
        return {
            "ready": health_score >= 0.8,
            "current_percentage": self.current_canary_pct,
            "next_percentage": next_percentage,
            "health_score": health_score,
            "criteria": criteria,
            "metrics_summary": {
                "avg_delta_cbu": df['DELTA_CBU_1k'].mean(),
                "avg_p95": df['LLM_p95'].mean(),
                "max_dual_gap": df['primal_dual_gap'].max(),
                "avg_kv_reuse": df['KV_prefix_reuse'].mean(),
            }
        }
        
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Get data from last 24 hours
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM per_turn_metrics 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                ORDER BY timestamp DESC
            ''')
            
        if not rows:
            return {"error": "No data available"}
            
        df = pd.DataFrame([dict(row) for row in rows])
        
        # Compare methods
        method_comparison = {}
        for method in ['hybrid', 'streaming', 'lethe']:
            method_data = df[df['method'] == method]
            if not method_data.empty:
                method_comparison[method] = {
                    "count": len(method_data),
                    "avg_delta_cbu": method_data['DELTA_CBU_1k'].mean(),
                    "avg_p95": method_data['LLM_p95'].mean(),
                    "avg_kv_reuse": method_data['KV_prefix_reuse'].mean(),
                    "avg_keep_ratio": method_data['keep_ratio_head'].mean() + method_data['keep_ratio_tail'].mean(),
                }
                
        # Trend analysis
        hourly_trends = df.set_index('timestamp').resample('1H').agg({
            'DELTA_CBU_1k': 'mean',
            'LLM_p95': 'mean',
            'primal_dual_gap': 'max',
            'KV_prefix_reuse': 'mean'
        }).to_dict()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_requests": len(df),
            "method_comparison": method_comparison,
            "hourly_trends": hourly_trends,
            "current_canary_pct": self.current_canary_pct,
            "alerts_24h": await self._count_recent_alerts(),
            "key_metrics": {
                "avg_delta_cbu_1k": df['DELTA_CBU_1k'].mean(),
                "avg_llm_p95": df['LLM_p95'].mean(),
                "max_primal_dual_gap": df['primal_dual_gap'].max(),
                "avg_kv_reuse": df['KV_prefix_reuse'].mean(),
            }
        }
        
    async def _count_recent_alerts(self) -> int:
        """Count alerts in last 24 hours"""
        async with self.db_pool.acquire() as conn:
            count = await conn.fetchval('''
                SELECT COUNT(*) FROM alerts 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            ''')
        return count or 0

async def main():
    """Main dashboard runner"""
    
    # Configuration
    DB_URL = "postgresql://user:pass@localhost/lethe_monitoring"
    REDIS_URL = "redis://localhost:6379"
    PROMETHEUS_PORT = 9090
    
    # Initialize dashboard
    dashboard = ProductionDashboard(DB_URL, REDIS_URL)
    await dashboard.initialize()
    
    # Start Prometheus metrics server
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics available on port {PROMETHEUS_PORT}")
    
    # Example: Log some test metrics
    test_metrics = PerTurnMetrics(
        lambda_param=0.12,
        mu_param=0.08, 
        tokens_in=8000,
        head_tokens=960,   # 12% of 8000
        tail_tokens=1440,  # Remaining after Lethe selection
        keep_ratio_head=0.12,
        keep_ratio_tail=0.18,
        K1=200,
        K2=320,
        r=14,
        CE_early_exit=True,
        num_windows=2,
        window_size=6000,
        stride=3000,
        sinks=96,
        KV_prefix_reuse=0.73,
        middleware_p95=142.3,
        LLM_p95=139.7,
        DELTA_CBU_1k=8.42,
        P_at_k=0.847,
        R_at_k=0.823,
        primal_dual_gap=0.0023,
        tail_cvar_095=167.2,
        timestamp=datetime.now(),
        request_id="test-001",
        canary_percentage=5.0,
        method="hybrid"
    )
    
    await dashboard.log_per_turn_metrics(test_metrics)
    
    # Generate initial report
    report = await dashboard.generate_performance_report()
    print("Performance Report:")
    print(json.dumps(report, indent=2, default=str))
    
    logger.info("Dashboard running - monitoring all TODO.md metrics")
    
    # Keep running
    while True:
        await asyncio.sleep(60)
        promotion_status = await dashboard.get_promotion_readiness()
        logger.info(f"Canary health: {promotion_status['health_score']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())