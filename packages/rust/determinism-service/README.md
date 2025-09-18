# Determinism Service

A 7-day burn-in hardening system with comprehensive determinism validation for production workloads.

## Features

### 🔄 Determinism Sentinel
- **Hourly Replay Validation**: Automatically replays processing slices every hour to validate deterministic behavior
- **≤1ms Timestamp Jitter Tolerance**: Enforces strict timing requirements with configurable tolerance
- **Mathematical Rigor**: Uses SHA-256 hashing of canonical JSON for result comparison
- **Comprehensive Reporting**: Detailed reports on determinism violations, performance metrics, and system health

### 📊 Canonical JSON Serialization
- **UTF-8 NFC Normalization**: All strings normalized using Unicode NFC form for consistent comparison
- **Sorted Object Keys**: Lexicographic key ordering ensures identical JSON for identical data
- **Fixed Float Rounding**: 6 significant figures for consistent numerical representation
- **Explicit Null Handling**: Distinguishes between explicit `null` values and omitted fields
- **Single-valued Function**: `stable_json(x)` produces identical output for identical input

### ⏰ Clock Skew & Causality Testing
- **Artificial Clock Skew Injection**: Simulates various timing conditions for stress testing
- **Out-of-order Message Delivery**: Tests system resilience to network timing variations
- **Monotonic Timestamp Validation**: Ensures timestamp ordering under stress conditions
- **Thread-safe Causal Ordering**: Validates happens-before relationships across concurrent operations

### 📈 Performance Budget Enforcement
- **100% Sampling for Structural Edits**: Critical operations always monitored
- **Dynamic Sampling Rate**: Adjusts from 1% to 100% based on performance metrics
- **p95 ≤ 2% Budget**: Strict performance requirements with real-time monitoring
- **Circuit Breaker Protection**: Automatic protection against performance degradation
- **Load-adaptive Field Elision**: Reduces monitoring overhead under high load

### 📱 Burn-in Monitoring Dashboard
- **Real-time Metrics**: Live performance and determinism success rates
- **Comprehensive Alerting**: Multi-severity alerts with cooldown periods
- **Health Monitoring**: Component-level health checks and system status
- **Historical Trending**: Long-term trend analysis and pattern detection

## Quick Start

### Prerequisites
- Rust 1.75+ with Cargo
- PostgreSQL 13+ (for persistent state)
- 4GB+ RAM (for comprehensive testing)

### Installation

```bash
# Clone and build
git clone <repository>
cd determinism-service
cargo build --release

# Run with default configuration
./target/release/determinism-service

# Or run in development mode
cargo run
```

### Configuration

Set environment variables or use defaults:

```bash
# Service Configuration
export SERVER_HOST=0.0.0.0
export SERVER_PORT=3001

# Determinism Settings
export REPLAY_INTERVAL_SECONDS=3600    # Every hour
export TOLERANCE_MS=1                   # ≤1ms jitter
export PERFORMANCE_BUDGET_PERCENT=2.0   # p95 ≤ 2%

# Database Configuration
export DATABASE_URL=postgresql://localhost:5432/determinism
export DB_MAX_CONNECTIONS=20

# Monitoring Settings  
export METRICS_PORT=9090
export DASHBOARD_UPDATE_INTERVAL_SECONDS=60
```

## API Endpoints

### Core Operations

#### Replay Determinism Test
```bash
POST /determinism/replay/{slice_id}
```
Triggers a determinism validation by running the same slice twice and comparing results.

Response:
```json
{
  "slice_id": "example_slice_123",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00Z",
  "run1": { ... },
  "run2": { ... },
  "determinism_check": {
    "is_deterministic": true,
    "hash_match": true,
    "timestamp_jitter_ms": 0,
    "differences": [],
    "tolerance_met": true
  },
  "performance_budget_check": {
    "budget_met": true,
    "p95_latency_ms": 1.8,
    "budget_threshold_ms": 2.0,
    "performance_ratio": 0.9,
    "sampling_rate": 1.0
  },
  "invariant_report": {
    "all_passed": true,
    "violations": [],
    "score": 1.0
  }
}
```

#### System Status
```bash
GET /determinism/status
```
Returns overall system health and performance metrics.

#### Performance Metrics
```bash
GET /determinism/metrics
```
Returns detailed performance and determinism statistics.

### Dashboard Integration

#### Dashboard Data
```bash
GET /dashboard/data
```
Returns comprehensive data for monitoring dashboard visualization.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────┐
│                 Determinism Service                  │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ DeterminismSentinel │    │    CanonicalJson     │  │
│  │   - Replay Logic     │    │  - UTF-8 NFC Norm   │  │
│  │   - Validation      │    │  - Sorted Keys       │  │
│  │   - Scheduling      │    │  - Fixed Rounding    │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  ClockSkewTester    │    │ PerformanceBudget   │  │
│  │  - Synthetic Clock  │    │  - Circuit Breaker  │  │
│  │  - Causality Tests  │    │  - Dynamic Sampling │  │
│  │  - Stress Testing   │    │  - Load Adaptation  │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  DashboardState     │    │     AlertManager    │  │
│  │  - Metrics Collection │    │  - Rule Engine   │  │
│  │  - Health Monitoring │    │  - Notifications  │  │
│  │  - Data Aggregation │    │  - Suppression     │  │
│  └─────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Performance Characteristics

| Component | Latency | Throughput | Memory Usage |
|-----------|---------|------------|--------------|
| Canonical JSON | <1ms | 10K ops/sec | <10MB |
| Determinism Replay | 10-50ms | 100 replays/sec | <50MB |
| Clock Skew Tests | 100-1000ms | 10 tests/sec | <20MB |
| Dashboard Updates | 1-5ms | 1K updates/sec | <100MB |

### Quality Assurance

The system includes comprehensive testing:
- **Unit Tests**: 95%+ coverage on critical paths  
- **Integration Tests**: Full replay validation cycles
- **Property-based Testing**: Canonical JSON correctness
- **Benchmarks**: Performance regression detection
- **Stress Tests**: Clock skew and timing edge cases

## Development

### Building from Source
```bash
# Development build
cargo build

# Release build with optimizations
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Testing

```bash
# Unit tests
cargo test --lib

# Integration tests  
cargo test --test integration

# Benchmark tests
cargo bench

# Property-based tests
cargo test --features proptest
```

## Monitoring & Observability

### Prometheus Metrics

The service exposes Prometheus-compatible metrics on port 9090:

- `determinism_success_rate`: Current determinism success rate (0.0-1.0)
- `determinism_replay_duration_seconds`: Histogram of replay durations
- `performance_budget_violations_total`: Counter of budget violations
- `invariant_violations_total`: Counter by violation type
- `clock_skew_tolerance_exceeded_total`: Counter of timing violations

### Logging

Structured logging with multiple levels:
```bash
RUST_LOG=determinism_service=debug,tower_http=info cargo run
```

### Health Checks

- **Service Health**: `/health` endpoint for load balancer checks
- **Component Health**: Individual component status monitoring  
- **Database Connectivity**: Automatic database health validation
- **Performance Health**: Circuit breaker state and budget compliance

## Production Deployment

### Resource Requirements

- **CPU**: 2+ cores (4+ recommended for high throughput)
- **Memory**: 4GB+ (8GB+ for extensive historical data)
- **Storage**: 10GB+ for logs and persistent state
- **Network**: Low-latency connection for timing accuracy

### Configuration Recommendations

```bash
# Production settings
export REPLAY_INTERVAL_SECONDS=1800     # Every 30 minutes
export MAX_CONCURRENT_REPLAYS=20        # Higher concurrency
export DB_MAX_CONNECTIONS=50            # More database connections
export PERFORMANCE_BUDGET_PERCENT=1.5   # Stricter budget
```

### Security Considerations

- **No Authentication**: This service expects to run in a trusted network environment
- **Input Validation**: All inputs validated but service assumes trusted callers
- **Resource Limits**: Built-in circuit breakers prevent resource exhaustion
- **Logging**: No sensitive data logged (all PII filtered)

## Troubleshooting

### Common Issues

**High Memory Usage**
- Reduce `MAX_ALERT_HISTORY` and historical data retention
- Increase garbage collection frequency
- Monitor for memory leaks in long-running tests

**Performance Budget Violations**
- Check system load and reduce concurrent operations  
- Increase `PERFORMANCE_BUDGET_PERCENT` temporarily
- Review infrastructure capacity

**Clock Skew Test Failures**
- Verify system clock synchronization (NTP)
- Check for high CPU load affecting timing
- Increase `TOLERANCE_MS` if appropriate for your use case

**Determinism Failures**
- Review processing logic for non-deterministic behavior
- Check for timestamp-dependent operations
- Validate input data consistency

### Debugging

Enable debug logging:
```bash
RUST_LOG=determinism_service=debug cargo run
```

Monitor specific components:
```bash
# Focus on determinism validation
RUST_LOG=determinism_service::determinism=trace

# Monitor performance enforcement  
RUST_LOG=determinism_service::performance=debug

# Track JSON canonicalization
RUST_LOG=determinism_service::json_canon=debug
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Run full test suite: `cargo test`
5. Run benchmarks: `cargo bench`  
6. Submit a pull request

All contributions must maintain:
- 95%+ test coverage
- Zero performance regressions
- Full documentation updates
- Determinism validation compliance