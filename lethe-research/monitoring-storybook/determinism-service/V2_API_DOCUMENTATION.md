# Determinism Service V2 API Documentation

## Overview

The Determinism Service V2 provides a comprehensive RESTful API for monitoring, validating, and optimizing prompt transformations with deterministic behavior guarantees. Built with Rust for memory safety and performance, the service includes real-time monitoring, machine learning integration, and advanced analytics capabilities.

**Base URL**: `http://localhost:PORT` (configurable via `SERVER_PORT` environment variable)  
**Authentication**: None (designed for trusted network environments)  
**Content-Type**: `application/json`  
**Rate Limiting**: Built-in circuit breaker protection

## Core Endpoints

### Health & Status

#### GET /health
**Description**: Service health check for load balancers and monitoring systems.

**Response**:
```json
{
  "status": "healthy"
}
```

**Status Codes**:
- `200 OK`: Service is healthy and operational
- `503 Service Unavailable`: Service is degraded or unhealthy

**Usage**:
```bash
curl -X GET http://localhost:3001/health
```

#### GET /determinism/status
**Description**: Overall system health and performance metrics with detailed component status.

**Response**:
```json
{
  "service_status": "operational",
  "uptime_seconds": 86400,
  "version": "v2.1.0",
  "components": {
    "determinism_sentinel": {
      "status": "healthy",
      "last_validation": "2025-09-10T16:30:00Z",
      "success_rate": 0.9997
    },
    "performance_budget": {
      "status": "healthy",
      "current_budget_utilization": 0.018,
      "circuit_breaker_state": "closed"
    },
    "database": {
      "status": "healthy",
      "connection_pool": {
        "active": 15,
        "idle": 35,
        "max": 50
      }
    },
    "learning_loop": {
      "status": "healthy",
      "active_experiments": 3,
      "model_accuracy": 0.847
    }
  },
  "performance_metrics": {
    "avg_response_time_ms": 1.8,
    "p95_response_time_ms": 4.2,
    "p99_response_time_ms": 8.1,
    "requests_per_second": 245.7
  },
  "alerts": []
}
```

**Status Codes**:
- `200 OK`: Status retrieved successfully

#### GET /determinism/metrics
**Description**: Detailed performance and determinism statistics for monitoring dashboards.

**Response**:
```json
{
  "timestamp": "2025-09-10T16:30:00Z",
  "determinism": {
    "success_rate": 0.9997,
    "total_replays": 15420,
    "failed_replays": 5,
    "avg_jitter_ms": 0.3,
    "max_jitter_ms": 0.8
  },
  "performance": {
    "budget_utilization": 0.018,
    "sampling_rate": 0.85,
    "circuit_breaker_trips": 0,
    "avg_latency_ms": 1.8,
    "throughput_ops_per_sec": 245.7
  },
  "invariants": {
    "canonical_json_violations": 0,
    "timestamp_violations": 2,
    "causality_violations": 0
  },
  "system": {
    "memory_usage_mb": 342.1,
    "cpu_usage_percent": 23.4,
    "disk_usage_mb": 1847.3
  }
}
```

**Status Codes**:
- `200 OK`: Metrics retrieved successfully

## Determinism Validation

#### POST /determinism/replay/{slice_id}
**Description**: Triggers determinism validation by running the same slice twice and comparing results with mathematical rigor.

**Parameters**:
- `slice_id` (path): Unique identifier for the processing slice to replay

**Request Body**: None

**Response**:
```json
{
  "slice_id": "user_prompt_transform_20250910_001",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-09-10T16:30:00Z",
  "validation_method": "sha256_canonical_json",
  "run1": {
    "duration_ms": 12.4,
    "result_hash": "a7b3c2d1e5f4a8b9c0d3e6f2a9b4c7d0e1f5a2b8c3d6e9f0a4b7c1d4e8f3a6b9",
    "timestamp": "2025-09-10T16:30:00.123Z",
    "memory_usage_mb": 23.4,
    "cpu_usage_percent": 15.2
  },
  "run2": {
    "duration_ms": 12.1,
    "result_hash": "a7b3c2d1e5f4a8b9c0d3e6f2a9b4c7d0e1f5a2b8c3d6e9f0a4b7c1d4e8f3a6b9",
    "timestamp": "2025-09-10T16:30:00.149Z",
    "memory_usage_mb": 23.6,
    "cpu_usage_percent": 15.0
  },
  "determinism_check": {
    "is_deterministic": true,
    "hash_match": true,
    "timestamp_jitter_ms": 0.3,
    "differences": [],
    "tolerance_met": true,
    "canonical_json_validation": {
      "utf8_nfc_normalized": true,
      "keys_sorted": true,
      "floats_rounded": true,
      "null_handling_consistent": true
    }
  },
  "performance_budget_check": {
    "budget_met": true,
    "p95_latency_ms": 1.8,
    "budget_threshold_ms": 2.0,
    "performance_ratio": 0.9,
    "sampling_rate": 1.0,
    "circuit_breaker_state": "closed"
  },
  "invariant_report": {
    "all_passed": true,
    "violations": [],
    "score": 1.0,
    "checks_performed": [
      "monotonic_timestamps",
      "causal_ordering",
      "resource_bounds",
      "output_consistency"
    ]
  }
}
```

**Status Codes**:
- `200 OK`: Replay completed successfully
- `400 Bad Request`: Invalid slice_id format
- `404 Not Found`: Slice not found
- `500 Internal Server Error`: Replay execution failed

**Usage**:
```bash
curl -X POST http://localhost:3001/determinism/replay/user_prompt_transform_20250910_001
```

## Dashboard & Monitoring

#### GET /dashboard/data
**Description**: Comprehensive data for monitoring dashboards with real-time metrics and historical trends.

**Response**:
```json
{
  "timestamp": "2025-09-10T16:30:00Z",
  "overview": {
    "total_requests_24h": 125840,
    "success_rate": 0.9997,
    "avg_response_time": 1.8,
    "active_experiments": 3
  },
  "determinism_metrics": {
    "hourly_validations": [
      {
        "hour": "2025-09-10T15:00:00Z",
        "total_replays": 642,
        "successful": 641,
        "failed": 1,
        "avg_jitter": 0.2
      }
    ],
    "success_rate_trend": [0.999, 0.9995, 0.9992, 0.9997],
    "current_streak": {
      "consecutive_successes": 15415,
      "duration_hours": 168.3
    }
  },
  "performance_trends": {
    "latency_p95": [1.9, 1.7, 1.8, 1.8],
    "throughput": [238.5, 251.2, 247.8, 245.7],
    "budget_utilization": [0.019, 0.017, 0.018, 0.018],
    "circuit_breaker_events": []
  },
  "learning_loop_status": {
    "active_variants": ["control", "v2_features", "isotonic_cal"],
    "current_best": "v2_features",
    "confidence": 0.92,
    "improvement": 0.15
  },
  "alerts": {
    "active": [],
    "recent": [
      {
        "id": "alert_001",
        "type": "performance_budget_warning",
        "message": "Budget utilization approaching threshold",
        "timestamp": "2025-09-10T14:22:00Z",
        "resolved": true,
        "resolution": "automatic_scaling"
      }
    ]
  },
  "system_health": {
    "memory_usage": 0.68,
    "cpu_usage": 0.23,
    "disk_usage": 0.45,
    "network_io": 12.4
  }
}
```

**Status Codes**:
- `200 OK`: Dashboard data retrieved successfully

## Learning Loop Integration

#### POST /learning/process
**Description**: Process transform changes through the learning loop with A/B testing and statistical analysis.

**Request Body**:
```json
{
  "changes": [
    {
      "change_type": "system_prelude_added",
      "content_delta": 45,
      "provider": "anthropic",
      "model": "claude-3-sonnet",
      "timestamp": "2025-09-10T16:30:00Z",
      "metadata": {
        "request_id": "req_12345",
        "user_id": "user_67890",
        "session_id": "sess_abcde"
      }
    }
  ],
  "scenario_type": "complex_reasoning"
}
```

**Response**:
```json
{
  "processing_id": "proc_550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-09-10T16:30:00Z",
  "variant_assignment": "v2_features",
  "processing_results": {
    "utility_score": 0.847,
    "confidence": 0.92,
    "feature_importance": {
      "v2_extraction": 0.34,
      "isotonic_calibration": 0.28,
      "ips_adjustment": 0.19,
      "lambda_mu_control": 0.19
    },
    "performance_impact": {
      "latency_change_ms": -0.3,
      "accuracy_improvement": 0.023,
      "complexity_score": 0.65
    }
  },
  "experiment_status": {
    "current_phase": "treatment",
    "sample_size": 15420,
    "statistical_power": 0.95,
    "significance_level": 0.05,
    "p_value": 0.001,
    "effect_size": 0.15,
    "confidence_interval": [0.12, 0.18]
  },
  "recommendations": {
    "continue_experiment": true,
    "estimated_completion": "2025-09-12T16:30:00Z",
    "next_decision_point": "2025-09-11T16:30:00Z",
    "risk_assessment": "low"
  }
}
```

**Status Codes**:
- `200 OK`: Changes processed successfully
- `400 Bad Request`: Invalid request format
- `422 Unprocessable Entity`: Invalid scenario type or changes
- `500 Internal Server Error`: Processing failed

#### GET /learning/metrics
**Description**: Learning loop performance metrics and experiment results.

**Response**:
```json
{
  "timestamp": "2025-09-10T16:30:00Z",
  "overall_performance": {
    "model_accuracy": 0.847,
    "prediction_confidence": 0.92,
    "convergence_rate": 0.023,
    "stability_index": 0.89
  },
  "active_experiments": {
    "total": 3,
    "by_status": {
      "running": 2,
      "analysis": 1,
      "completed": 0
    }
  },
  "feature_performance": {
    "v2_feature_extraction": {
      "accuracy": 0.863,
      "latency_ms": 2.1,
      "success_rate": 0.996,
      "adoption_rate": 0.67
    },
    "isotonic_calibration": {
      "accuracy": 0.851,
      "latency_ms": 1.9,
      "success_rate": 0.998,
      "adoption_rate": 0.72
    },
    "ips_adjustment": {
      "accuracy": 0.834,
      "latency_ms": 2.3,
      "success_rate": 0.994,
      "adoption_rate": 0.58
    },
    "lambda_mu_controller": {
      "accuracy": 0.829,
      "latency_ms": 1.7,
      "success_rate": 0.999,
      "adoption_rate": 0.81
    }
  },
  "training_metrics": {
    "total_samples": 458920,
    "recent_samples_24h": 12540,
    "cross_validation_score": 0.826,
    "feature_importance": {
      "complexity_score": 0.31,
      "provider_embedding": 0.24,
      "temporal_features": 0.22,
      "content_features": 0.23
    }
  }
}
```

**Status Codes**:
- `200 OK`: Metrics retrieved successfully

#### GET /learning/ab-test/results
**Description**: A/B testing results with statistical analysis and recommendations.

**Response**:
```json
{
  "timestamp": "2025-09-10T16:30:00Z",
  "experiment_overview": {
    "total_active": 2,
    "total_completed": 15,
    "success_rate": 0.87
  },
  "current_experiments": [
    {
      "id": "exp_v2_features_001",
      "name": "V2 Feature Extraction vs Baseline",
      "status": "running",
      "start_date": "2025-09-08T10:00:00Z",
      "estimated_end": "2025-09-12T16:00:00Z",
      "variants": {
        "control": {
          "name": "Baseline V1",
          "traffic_allocation": 0.3,
          "sample_size": 4626,
          "conversion_rate": 0.834,
          "confidence_interval": [0.821, 0.847]
        },
        "treatment": {
          "name": "V2 Features Enabled", 
          "traffic_allocation": 0.7,
          "sample_size": 10794,
          "conversion_rate": 0.863,
          "confidence_interval": [0.855, 0.871]
        }
      },
      "statistical_analysis": {
        "significance_level": 0.05,
        "power": 0.95,
        "p_value": 0.001,
        "effect_size": 0.029,
        "lift": 0.0348,
        "lift_confidence_interval": [0.018, 0.052],
        "minimum_detectable_effect": 0.01,
        "statistical_significance": true,
        "practical_significance": true
      },
      "recommendation": {
        "action": "deploy_treatment",
        "confidence": 0.95,
        "rationale": "Treatment shows significant improvement with high statistical confidence",
        "risk_level": "low",
        "estimated_impact": "+3.5% conversion improvement"
      }
    }
  ],
  "performance_summary": {
    "best_performing_variant": "v2_features",
    "average_lift": 0.0348,
    "experiments_with_significance": 13,
    "false_discovery_rate": 0.02
  }
}
```

**Status Codes**:
- `200 OK`: A/B test results retrieved successfully

#### POST /learning/training
**Description**: Add training data to the learning loop for model improvement.

**Request Body**:
```json
{
  "features": {
    "complexity_score": 0.65,
    "provider_id": 2,
    "model_size": 768,
    "content_length": 1520,
    "edit_distance": 45,
    "semantic_similarity": 0.89,
    "temporal_features": [0.1, 0.3, 0.8, 0.2],
    "contextual_features": {
      "domain": "technical",
      "urgency": "normal",
      "user_experience": "expert"
    }
  },
  "ground_truth_utility": 0.847,
  "scenario_type": "complex_reasoning"
}
```

**Response**:
```json
{
  "status": "training data added",
  "sample_id": "sample_550e8400-e29b",
  "timestamp": "2025-09-10T16:30:00Z",
  "validation": {
    "feature_validation": "passed",
    "utility_range_check": "passed", 
    "scenario_consistency": "passed"
  },
  "model_impact": {
    "samples_in_training_set": 458921,
    "estimated_accuracy_change": 0.00001,
    "next_retrain_time": "2025-09-11T02:00:00Z"
  }
}
```

**Status Codes**:
- `200 OK`: Training data added successfully
- `400 Bad Request`: Invalid feature format or utility value
- `422 Unprocessable Entity`: Data validation failed

#### POST /learning/performance
**Description**: Record performance metrics for model evaluation and optimization.

**Request Body**:
```json
{
  "metric_type": "accuracy",
  "value": 0.847,
  "timestamp": "2025-09-10T16:30:00Z",
  "variant": "v2_features",
  "context": {
    "scenario": "complex_reasoning",
    "sample_size": 1000,
    "evaluation_method": "cross_validation"
  }
}
```

**Response**:
```json
{
  "status": "performance recorded",
  "metric_id": "metric_550e8400-e29b",
  "timestamp": "2025-09-10T16:30:00Z",
  "aggregation": {
    "current_average": 0.845,
    "trend": "improving",
    "sample_count": 1524,
    "confidence_interval": [0.839, 0.851]
  }
}
```

**Status Codes**:
- `200 OK`: Performance metric recorded successfully
- `400 Bad Request`: Invalid metric format

## V2 Benchmark System

#### POST /benchmark/v2/execute
**Description**: Execute comprehensive V2 benchmark matrix with performance validation.

**Request Body** (optional):
```json
{
  "test_matrix": {
    "scenarios": ["simple", "moderate", "complex", "extreme"],
    "providers": ["openai", "anthropic", "google"],
    "model_sizes": [256, 768],
    "iterations": 100
  },
  "performance_gates": {
    "max_ece_score": 0.08,
    "p95_latency_baseline": 2.1,
    "max_p99_p95_ratio": 2.5,
    "max_proxy_gap": 0.005,
    "pool_fingerprint_tolerance": 0.001
  }
}
```

**Response**:
```json
{
  "status": "completed",
  "benchmark_id": "benchmark_550e8400-e29b",
  "timestamp": "2025-09-10T16:30:00Z",
  "execution_time_seconds": 342.7,
  "report": {
    "summary": {
      "total_tests": 2400,
      "passed": 2387,
      "failed": 13,
      "success_rate": 0.9946,
      "overall_score": 94.6
    },
    "performance_results": {
      "ece_score": {
        "average": 0.063,
        "p95": 0.078,
        "gate_status": "PASS"
      },
      "latency_metrics": {
        "p95_ms": 1.8,
        "p99_ms": 3.7,
        "p99_p95_ratio": 2.06,
        "gate_status": "PASS"
      },
      "proxy_gap": {
        "average": 0.0031,
        "maximum": 0.0047,
        "gate_status": "PASS"
      },
      "pool_fingerprint_stability": {
        "consistency_rate": 1.0,
        "hash_match_rate": 1.0,
        "gate_status": "PASS"
      }
    },
    "feature_analysis": {
      "v2_feature_extraction": {
        "improvement_over_v1": 0.147,
        "accuracy": 0.863,
        "latency_overhead_ms": 0.3
      },
      "isotonic_calibration": {
        "calibration_error_reduction": 0.234,
        "accuracy": 0.851,
        "latency_overhead_ms": 0.1
      },
      "ips_adjustment": {
        "bias_reduction": 0.182,
        "accuracy": 0.834,
        "latency_overhead_ms": 0.4
      },
      "lambda_mu_controller": {
        "stability_improvement": 0.195,
        "accuracy": 0.829,
        "latency_improvement_ms": -0.4
      }
    },
    "competitive_analysis": {
      "baseline_comparison": {
        "accuracy_lift": 0.089,
        "latency_reduction": 0.143,
        "resource_efficiency": 0.067
      },
      "industry_benchmarks": {
        "accuracy_percentile": 0.87,
        "latency_percentile": 0.92,
        "reliability_percentile": 0.95
      }
    }
  },
  "recommendations": {
    "production_readiness": "READY",
    "optimization_opportunities": [
      "Fine-tune isotonic calibration parameters",
      "Optimize V2 feature extraction caching",
      "Consider lambda-mu controller for edge cases"
    ],
    "risk_assessment": {
      "overall_risk": "LOW",
      "key_risks": [
        "Latency spike under extreme load",
        "Memory usage with large contexts"
      ],
      "mitigation_strategies": [
        "Implement adaptive sampling",
        "Add memory pressure monitoring"
      ]
    }
  }
}
```

**Status Codes**:
- `200 OK`: Benchmark completed successfully
- `400 Bad Request`: Invalid benchmark configuration
- `500 Internal Server Error`: Benchmark execution failed

#### POST /benchmark/v2/quick-test
**Description**: Execute quick V2 validation test for rapid feedback.

**Request Body**: None

**Response**:
```json
{
  "status": "passed",
  "message": "Quick V2 test completed successfully",
  "test_id": "quicktest_550e8400",
  "timestamp": "2025-09-10T16:30:00Z",
  "execution_time_ms": 2347,
  "results": {
    "basic_functionality": "PASS",
    "determinism_check": "PASS",
    "performance_gate": "PASS",
    "feature_validation": "PASS"
  },
  "metrics": {
    "response_time_ms": 1.6,
    "accuracy_score": 0.89,
    "determinism_score": 1.0,
    "resource_usage": "normal"
  }
}
```

**Status Codes**:
- `200 OK`: Quick test passed
- `500 Internal Server Error`: Quick test failed

## Error Responses

All endpoints return standardized error responses following RFC 7807 (Problem Details for HTTP APIs):

```json
{
  "type": "https://httpstatuses.com/400",
  "title": "Bad Request",
  "status": 400,
  "detail": "Invalid request format: missing required field 'scenario_type'",
  "instance": "/learning/process",
  "timestamp": "2025-09-10T16:30:00Z",
  "request_id": "req_550e8400-e29b"
}
```

**Common Error Types**:
- `400 Bad Request`: Invalid request format or parameters
- `401 Unauthorized`: Authentication required (if enabled)
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Valid format but semantically incorrect
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side processing error
- `503 Service Unavailable`: Service temporarily unavailable

## WebSocket Integration

For real-time updates, the service supports WebSocket connections:

**Endpoint**: `ws://localhost:PORT/ws`

**Message Types**:
- `determinism_update`: Real-time determinism validation results
- `performance_metrics`: Live performance monitoring data
- `learning_loop_event`: Learning loop experiment updates
- `alert`: System alerts and notifications

**Message Format**:
```json
{
  "type": "determinism_update",
  "timestamp": "2025-09-10T16:30:00Z",
  "data": {
    "slice_id": "slice_12345",
    "is_deterministic": true,
    "jitter_ms": 0.3,
    "performance_score": 0.95
  }
}
```

## SDK & Client Libraries

### Rust Client
```rust
use determinism_client::Client;

let client = Client::new("http://localhost:3001")?;
let status = client.get_status().await?;
let replay_result = client.replay_slice("slice_123").await?;
```

### Python Client
```python
from determinism_client import Client

client = Client("http://localhost:3001")
status = client.get_status()
replay_result = client.replay_slice("slice_123")
```

### JavaScript/TypeScript Client
```typescript
import { DeterminismClient } from 'determinism-client';

const client = new DeterminismClient('http://localhost:3001');
const status = await client.getStatus();
const replayResult = await client.replaySlice('slice_123');
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `3001` | Server port |
| `DATABASE_URL` | `postgresql://localhost:5432/determinism` | PostgreSQL connection string |
| `DB_MAX_CONNECTIONS` | `20` | Maximum database connections |
| `REPLAY_INTERVAL_SECONDS` | `3600` | Determinism replay interval |
| `TOLERANCE_MS` | `1` | Timestamp jitter tolerance |
| `PERFORMANCE_BUDGET_PERCENT` | `2.0` | P95 performance budget |
| `METRICS_PORT` | `9090` | Prometheus metrics port |
| `RUST_LOG` | `determinism_service=info` | Logging configuration |

### Production Configuration

```bash
# Production optimized settings
export SERVER_PORT=8080
export REPLAY_INTERVAL_SECONDS=1800
export MAX_CONCURRENT_REPLAYS=20
export DB_MAX_CONNECTIONS=50
export PERFORMANCE_BUDGET_PERCENT=1.5
export RUST_LOG=determinism_service=info,tower_http=warn
```

## Monitoring & Observability

### Prometheus Metrics

Available at `http://localhost:9090/metrics`:

- `determinism_success_rate` - Current success rate (0.0-1.0)
- `determinism_replay_duration_seconds` - Histogram of replay durations
- `performance_budget_violations_total` - Counter of budget violations
- `invariant_violations_total` - Counter by violation type
- `clock_skew_tolerance_exceeded_total` - Counter of timing violations
- `learning_loop_accuracy` - ML model accuracy gauge
- `ab_test_conversion_rate` - A/B test conversion rates
- `api_request_duration_seconds` - API endpoint latencies
- `database_connection_pool_size` - Connection pool metrics

### Health Checks

- **Liveness**: `/health` - Basic service availability
- **Readiness**: `/determinism/status` - Full system readiness
- **Component Health**: Individual component status in status endpoint

### Logging

Structured JSON logging with configurable levels:

```json
{
  "timestamp": "2025-09-10T16:30:00Z",
  "level": "INFO",
  "module": "determinism_service::determinism",
  "message": "Determinism validation completed",
  "fields": {
    "slice_id": "slice_123",
    "is_deterministic": true,
    "jitter_ms": 0.3,
    "duration_ms": 12.4
  }
}
```

## Security Considerations

1. **Network Security**: Designed for trusted network environments
2. **Input Validation**: All inputs validated and sanitized
3. **Resource Protection**: Circuit breakers prevent resource exhaustion
4. **Memory Safety**: Rust guarantees prevent buffer overflows
5. **Data Privacy**: No PII logged or stored
6. **Configuration Security**: Sensitive data via environment variables only

## Rate Limiting & Circuit Breakers

The service includes built-in protection mechanisms:

- **Circuit Breaker**: Automatically opens when error rate exceeds threshold
- **Adaptive Sampling**: Reduces load during high-traffic periods
- **Resource Monitoring**: Prevents memory and CPU exhaustion
- **Database Protection**: Connection pooling with overflow handling

## Changelog

### V2.1.0 (Current)
- Complete Rust refactor with memory safety
- V2 feature extraction with certificate caching
- Learning loop integration with A/B testing
- Determinism sentinel with hourly validation
- Performance budget enforcement
- Canonical JSON with mathematical guarantees
- Clock skew tolerance testing
- Delta-U training with ML optimization
- Lambda-Mu controller for adaptive performance

### V2.0.0
- Initial V2 architecture
- Basic learning loop integration
- Enhanced monitoring capabilities

## Support & Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce historical data retention, increase GC frequency
2. **Performance Budget Violations**: Check system load, increase budget temporarily
3. **Clock Skew Failures**: Verify NTP synchronization, check CPU load
4. **Determinism Failures**: Review processing logic for non-deterministic behavior

### Debug Configuration

```bash
RUST_LOG=determinism_service=debug,tower_http=debug cargo run
```

### Contact Information

- **Technical Support**: determinism-team@lethe.research
- **Bug Reports**: GitHub Issues
- **Security Issues**: security@lethe.research
- **General Questions**: support@lethe.research

---

*This documentation is automatically generated from the OpenAPI specification and kept in sync with the codebase. Last updated: 2025-09-10T16:30:00Z*