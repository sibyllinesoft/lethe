# Lethe Replication Pack

## One-Click Verification System

This package contains everything needed to independently verify Lethe's benchmark claims.

### Quick Start

```bash
# Extract package
unzip lethe-replication-pack-*.zip
cd lethe-replication-pack/

# Run complete replication
./lethe-bench replay --matrix matrix.yml

# Validate results only
./lethe-bench validate --results runs/

# Run adversarial tests
./lethe-bench adversarial --suite all
```

### Package Contents

- `docker-compose.yml` - Complete system stack
- `matrix.yml` - Benchmark configuration
- `lethe-bench` - CLI tool for replication
- `pools/` - Frozen candidate pools with checksums
- `validators/` - Fail-closed validation scripts
- `MANIFEST.json` + `MANIFEST.sig` - Cryptographically signed manifest

### Verification Process

1. **Manifest Integrity**: Verify cryptographic signatures
2. **Pool Consistency**: Validate frozen candidate pools
3. **Statistical Integrity**: Check CI brackets and significance
4. **Fairness Invariants**: Validate latency distributions
5. **Adversarial Robustness**: Test failure modes and recovery

### Expected Results

The replication should produce results within 5% variance of published figures:

- Lethe-Hybrid: ~14ms latency, 0.831 Macro P@5
- Statistical significance: p < 0.001 vs all competitors
- Adversarial degradation: < 30% in worst-case scenarios

### Troubleshooting

**Docker Issues**:
```bash
docker-compose down
docker system prune -f
docker-compose up -d --force-recreate
```

**Validation Failures**:
- Check `runs/validation_*.log` for details
- Ensure all services are healthy: `docker-compose ps`
- Verify manifest signature: `python validators/verify_manifest.py`

### Support

For replication issues, contact: replication@lethe.dev

This package is designed to be "fork-proof" - any deviation from published 
results indicates either:
1. Replication environment issues
2. Underlying system changes requiring investigation
