-- Lethe Production Monitoring Database Setup
-- Creates all tables for comprehensive TODO.md instrumentation tracking

-- Database creation
CREATE DATABASE IF NOT EXISTS lethe_monitoring;
USE lethe_monitoring;

-- Per-turn metrics table (main instrumentation data)
CREATE TABLE IF NOT EXISTS per_turn_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id VARCHAR(255) NOT NULL,
    
    -- Core parameters (λ, μ, tokens)
    lambda_param FLOAT NOT NULL,
    mu_param FLOAT NOT NULL, 
    tokens_in INTEGER NOT NULL,
    head_tokens INTEGER NOT NULL,
    tail_tokens INTEGER NOT NULL,
    keep_ratio_head FLOAT NOT NULL,
    keep_ratio_tail FLOAT NOT NULL,
    
    -- DPP/CE parameters (K1/K2/r)
    K1 INTEGER NOT NULL,
    K2 INTEGER NOT NULL,
    r INTEGER NOT NULL, -- DPP rank
    CE_early_exit BOOLEAN NOT NULL,
    
    -- Streaming parameters
    num_windows INTEGER NOT NULL,
    window_size INTEGER NOT NULL,
    stride INTEGER NOT NULL,
    sinks INTEGER NOT NULL,
    
    -- Performance metrics
    KV_prefix_reuse FLOAT NOT NULL,
    middleware_p95 FLOAT NOT NULL, -- ms
    LLM_p95 FLOAT NOT NULL,        -- ms
    
    -- Quality metrics
    DELTA_CBU_1k FLOAT NOT NULL,   -- ΔCBU/1k
    P_at_k FLOAT NOT NULL,         -- P@k
    R_at_k FLOAT NOT NULL,         -- R@k
    
    -- Advanced monitoring (TODO.md requirements)
    primal_dual_gap FLOAT NOT NULL,      -- <0.5% threshold
    tail_cvar_095 FLOAT NOT NULL,        -- Tail CVaR₀.₉₅(compute)
    
    -- Metadata
    canary_percentage FLOAT NOT NULL,
    method VARCHAR(50) NOT NULL, -- 'hybrid', 'streaming', 'lethe'
    
    -- Indexing for time-series queries
    INDEX idx_metrics_timestamp (timestamp),
    INDEX idx_metrics_method (method),
    INDEX idx_metrics_canary (canary_percentage),
    INDEX idx_metrics_request (request_id),
    INDEX idx_metrics_compound (timestamp, method, canary_percentage)
);

-- Alert history table
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id VARCHAR(255),
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- 'critical', 'warning', 'info'
    message TEXT NOT NULL,
    metric_value FLOAT,
    threshold_value FLOAT,
    resolved_at TIMESTAMPTZ,
    auto_remediated BOOLEAN DEFAULT FALSE,
    
    INDEX idx_alerts_timestamp (timestamp),
    INDEX idx_alerts_type (alert_type),
    INDEX idx_alerts_severity (severity),
    INDEX idx_alerts_resolved (resolved_at)
);

-- Parameter drift tracking table
CREATE TABLE IF NOT EXISTS parameter_drift (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parameter_name VARCHAR(50) NOT NULL, -- 'lambda', 'mu'
    current_value FLOAT NOT NULL,
    baseline_value FLOAT NOT NULL,
    drift_percentage FLOAT NOT NULL, -- %
    window_hours INTEGER NOT NULL DEFAULT 24,
    alarm_triggered BOOLEAN DEFAULT FALSE,
    
    INDEX idx_drift_timestamp (timestamp),
    INDEX idx_drift_param (parameter_name),
    INDEX idx_drift_alarm (alarm_triggered)
);

-- KV cache performance tracking
CREATE TABLE IF NOT EXISTS kv_performance (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id VARCHAR(255) NOT NULL,
    jaccard_similarity FLOAT NOT NULL,
    jaccard_baseline FLOAT,
    jaccard_drop FLOAT,
    prefix_reuse_ratio FLOAT NOT NULL,
    cache_hit_rate FLOAT,
    cache_efficiency_score FLOAT,
    
    INDEX idx_kv_timestamp (timestamp),
    INDEX idx_kv_jaccard (jaccard_similarity)
);

-- Tail EVT (Extreme Value Theory) monitoring
CREATE TABLE IF NOT EXISTS tail_evt_analysis (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    compute_sample_size INTEGER NOT NULL,
    evt_xi_parameter FLOAT NOT NULL,        -- Shape parameter ξ
    evt_sigma_parameter FLOAT,              -- Scale parameter
    tail_threshold_p95 FLOAT NOT NULL,      -- p95 threshold
    excesses_count INTEGER NOT NULL,
    tail_cvar_095 FLOAT NOT NULL,           -- CVaR₀.₉₅
    heavy_tail_alarm BOOLEAN DEFAULT FALSE,
    recommended_action TEXT,
    
    INDEX idx_evt_timestamp (timestamp),
    INDEX idx_evt_xi (evt_xi_parameter),
    INDEX idx_evt_alarm (heavy_tail_alarm)
);

-- Canary rollout tracking
CREATE TABLE IF NOT EXISTS canary_rollout (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    canary_percentage FLOAT NOT NULL,
    traffic_split JSONB NOT NULL,          -- {'hybrid': 5, 'streaming': 95}
    health_score FLOAT NOT NULL,           -- 0-1 score
    promotion_criteria JSONB NOT NULL,     -- Detailed criteria check
    auto_promoted BOOLEAN DEFAULT FALSE,
    rollback_triggered BOOLEAN DEFAULT FALSE,
    rollback_reason TEXT,
    
    INDEX idx_canary_timestamp (timestamp),
    INDEX idx_canary_percentage (canary_percentage),
    INDEX idx_canary_health (health_score)
);

-- Performance comparison table (methods comparison)
CREATE TABLE IF NOT EXISTS method_comparison (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    comparison_window_hours INTEGER NOT NULL DEFAULT 1,
    
    -- Hybrid method metrics
    hybrid_delta_cbu FLOAT,
    hybrid_p95_latency FLOAT,
    hybrid_kv_reuse FLOAT,
    hybrid_request_count INTEGER DEFAULT 0,
    
    -- Streaming baseline metrics  
    streaming_delta_cbu FLOAT,
    streaming_p95_latency FLOAT,
    streaming_kv_reuse FLOAT,
    streaming_request_count INTEGER DEFAULT 0,
    
    -- Lethe-only metrics
    lethe_delta_cbu FLOAT,
    lethe_p95_latency FLOAT,
    lethe_kv_reuse FLOAT,
    lethe_request_count INTEGER DEFAULT 0,
    
    -- Comparison results
    hybrid_vs_streaming_improvement FLOAT, -- % improvement
    win_condition_met BOOLEAN DEFAULT FALSE,
    
    INDEX idx_comparison_timestamp (timestamp),
    INDEX idx_comparison_win (win_condition_met)
);

-- System health checkpoints
CREATE TABLE IF NOT EXISTS health_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checkpoint_type VARCHAR(50) NOT NULL, -- 'hourly', 'deployment', 'alert'
    
    -- Overall system metrics
    total_requests_last_hour INTEGER,
    error_rate_percentage FLOAT,
    avg_response_time FLOAT,
    p95_response_time FLOAT,
    p99_response_time FLOAT,
    
    -- Resource utilization
    cpu_usage_percentage FLOAT,
    memory_usage_percentage FLOAT,
    active_connections INTEGER,
    
    -- Quality metrics
    avg_delta_cbu FLOAT,
    avg_precision_at_k FLOAT,
    avg_recall_at_k FLOAT,
    
    -- Health scores
    performance_health_score FLOAT,  -- 0-1
    quality_health_score FLOAT,      -- 0-1
    stability_health_score FLOAT,    -- 0-1
    overall_health_score FLOAT,      -- 0-1
    
    INDEX idx_health_timestamp (timestamp),
    INDEX idx_health_type (checkpoint_type),
    INDEX idx_health_overall (overall_health_score)
);

-- Create views for common queries
CREATE VIEW v_recent_metrics AS
SELECT 
    timestamp,
    method,
    canary_percentage,
    lambda_param,
    mu_param,
    DELTA_CBU_1k,
    LLM_p95,
    KV_prefix_reuse,
    primal_dual_gap,
    keep_ratio_head + keep_ratio_tail as total_keep_ratio
FROM per_turn_metrics 
WHERE timestamp > NOW() - INTERVAL '4 hours'
ORDER BY timestamp DESC;

CREATE VIEW v_method_performance_summary AS
SELECT 
    method,
    canary_percentage,
    COUNT(*) as request_count,
    AVG(DELTA_CBU_1k) as avg_delta_cbu,
    AVG(LLM_p95) as avg_p95_latency,
    AVG(KV_prefix_reuse) as avg_kv_reuse,
    MAX(primal_dual_gap) as max_dual_gap,
    AVG(keep_ratio_head + keep_ratio_tail) as avg_keep_ratio
FROM per_turn_metrics 
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY method, canary_percentage
ORDER BY method, canary_percentage;

CREATE VIEW v_alert_summary AS
SELECT 
    alert_type,
    severity,
    COUNT(*) as alert_count,
    COUNT(CASE WHEN resolved_at IS NULL THEN 1 END) as active_count,
    MAX(timestamp) as last_occurrence,
    AVG(EXTRACT(EPOCH FROM (resolved_at - timestamp))) as avg_resolution_time_seconds
FROM alerts 
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY alert_type, severity
ORDER BY alert_count DESC;

-- Create functions for drift detection
CREATE OR REPLACE FUNCTION calculate_parameter_drift(
    param_name VARCHAR(50),
    current_val FLOAT,
    hours_back INTEGER DEFAULT 24
) RETURNS TABLE(drift_pct FLOAT, alarm_triggered BOOLEAN) AS $$
DECLARE
    baseline_val FLOAT;
    drift_percentage FLOAT;
    alarm_threshold FLOAT := 0.15; -- 15%
BEGIN
    -- Get baseline value from hours_back
    SELECT AVG(CASE 
        WHEN param_name = 'lambda' THEN lambda_param 
        WHEN param_name = 'mu' THEN mu_param 
        END)
    INTO baseline_val
    FROM per_turn_metrics 
    WHERE timestamp BETWEEN (NOW() - INTERVAL '1 hour' * (hours_back + 1)) 
                        AND (NOW() - INTERVAL '1 hour' * hours_back)
    AND CASE 
        WHEN param_name = 'lambda' THEN lambda_param IS NOT NULL
        WHEN param_name = 'mu' THEN mu_param IS NOT NULL
        ELSE false
        END;
    
    IF baseline_val IS NULL OR baseline_val = 0 THEN
        drift_pct := 0;
        alarm_triggered := false;
    ELSE
        drift_percentage := ABS(current_val - baseline_val) / ABS(baseline_val);
        drift_pct := drift_percentage;
        alarm_triggered := drift_percentage > alarm_threshold;
    END IF;
    
    RETURN QUERY SELECT drift_pct, alarm_triggered;
END;
$$ LANGUAGE plpgsql;

-- Insert initial baseline data (example)
INSERT INTO parameter_drift (parameter_name, current_value, baseline_value, drift_percentage) VALUES
    ('lambda', 0.12, 0.12, 0.0),
    ('mu', 0.08, 0.08, 0.0);

-- Create materialized view for fast dashboard queries
CREATE MATERIALIZED VIEW mv_dashboard_metrics AS
SELECT 
    date_trunc('minute', timestamp) as minute_bucket,
    method,
    canary_percentage,
    COUNT(*) as request_count,
    AVG(DELTA_CBU_1k) as avg_delta_cbu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LLM_p95) as p95_latency,
    AVG(KV_prefix_reuse) as avg_kv_reuse,
    MAX(primal_dual_gap) as max_dual_gap,
    AVG(lambda_param) as avg_lambda,
    AVG(mu_param) as avg_mu
FROM per_turn_metrics
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY date_trunc('minute', timestamp), method, canary_percentage
ORDER BY minute_bucket DESC;

-- Create index on materialized view
CREATE INDEX idx_mv_dashboard_bucket ON mv_dashboard_metrics(minute_bucket);
CREATE INDEX idx_mv_dashboard_method ON mv_dashboard_metrics(method, canary_percentage);

-- Refresh function for materialized view (call every minute)
CREATE OR REPLACE FUNCTION refresh_dashboard_metrics() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_metrics;
END;
$$ LANGUAGE plpgsql;