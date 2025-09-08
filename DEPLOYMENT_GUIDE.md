# Lethe Deployment Guide

## System Requirements

### Minimum Requirements
- CPU: 8 cores, 2.5GHz+
- RAM: 16GB
- Storage: 100GB SSD
- Docker: 20.10+
- Docker Compose: 2.0+

### Recommended Requirements  
- CPU: 16 cores, 3.0GHz+
- RAM: 32GB
- Storage: 500GB NVMe SSD
- Network: 1Gbps+ for distributed deployments

## Quick Deploy

### 1. Single-Node Deployment
```bash
# Clone replication package
wget https://releases.lethe.dev/replication-pack-latest.zip
unzip replication-pack-latest.zip
cd lethe-replication-pack/

# Start all services
docker-compose up -d

# Verify deployment
./lethe-bench validate --results runs/
```

### 2. Production Deployment
```bash
# Production compose file
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose scale lethe-hybrid=3
docker-compose scale weaviate=2

# Enable monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

## Configuration

### Lethe-Hybrid Configuration
```json
{
  "parameters": {
    "alpha": 0.6,
    "beta": 0.4, 
    "keep_ratio": 0.15,
    "lambda": 0.5,
    "mu": 0.7,
    "K2": 550,
    "reranker_weight": 0.3
  },
  "performance": {
    "max_qps": 1000,
    "timeout_ms": 30000,
    "circuit_breaker": true
  }
}
```

### Environment Variables
```bash
# Core settings
LETHE_MODE=hybrid                    # hybrid|streaming|db-hybrid
POOL_PATH=/app/pools/frozen_pool_v1.jsonl
CONFIG_PATH=/app/configs/hybrid.json

# Performance tuning
LETHE_MAX_QPS=1000
LETHE_TIMEOUT_MS=30000
LETHE_WORKERS=4

# Monitoring
LETHE_METRICS_ENABLED=true
LETHE_LOG_LEVEL=INFO
```

## Health Checks

### Service Health
```bash
# Check all services
docker-compose ps

# Individual health checks
curl http://localhost:8080/health     # Lethe
curl http://localhost:8081/v1/meta    # Weaviate  
curl http://localhost:19530/health    # Milvus
curl http://localhost:6070/           # Zoekt
```

### Performance Validation
```bash
# Run performance benchmark
./lethe-bench replay --matrix matrix.yml

# Check throughput curves
./lethe-bench throughput --duration 60s

# Validate latency SLA
./lethe-bench validate --sla p95_lt_50ms
```

## Monitoring & Observability

### Metrics Collection
- Prometheus metrics on `/metrics` endpoint
- Grafana dashboard at `http://localhost:3000`
- Alert manager for SLA violations

### Key Metrics
- `lethe_request_duration_seconds` - Request latency
- `lethe_requests_total` - Request count by status
- `lethe_relevance_score` - Quality metrics
- `lethe_pool_hit_ratio` - Cache efficiency

### Log Analysis
```bash
# View service logs
docker-compose logs -f lethe-hybrid

# Search for errors
docker-compose logs | grep ERROR

# Performance analysis
docker-compose logs | grep "latency_ms"
```

## Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check resource usage
docker stats

# Clean restart
docker-compose down
docker system prune -f
docker-compose up -d --force-recreate
```

**High latency:**
```bash
# Check system resources
htop
iostat -x 1

# Tune parameters
export LETHE_WORKERS=8
export LETHE_MAX_QPS=2000
docker-compose restart lethe-hybrid
```

**Validation failures:**
```bash
# Check validation logs
ls -la runs/validation_*.log

# Verify manifest integrity
python validators/verify_manifest.py

# Test individual components
./lethe-bench test --system lethe-hybrid --query "test query"
```

### Performance Tuning

**CPU Optimization:**
- Increase worker processes: `LETHE_WORKERS=<cpu_cores>`
- Enable CPU affinity in Docker
- Use performance CPU governor

**Memory Optimization:**
- Tune JVM heap for Milvus: `-Xmx8g`
- Configure Weaviate memory: `LIMIT_RESOURCES=8GB`
- Enable swap if needed: `swapon /swapfile`

**Storage Optimization:**
- Use SSD for index storage
- Enable filesystem compression
- Tune Docker storage driver

## Security

### Network Security
- Internal Docker network isolation
- TLS for external endpoints
- API key authentication

### Data Protection
- Encrypt data at rest
- Secure secret management
- Regular security updates

## Scaling

### Horizontal Scaling
```bash
# Scale Lethe instances
docker-compose scale lethe-hybrid=5

# Load balancer configuration
nginx-conf/upstream-lethe.conf
```

### Database Scaling
```bash
# Milvus cluster mode
export MILVUS_CLUSTER=true
docker-compose -f docker-compose.cluster.yml up -d

# Weaviate replication
export WEAVIATE_REPLICATION_FACTOR=3
```

## Backup & Recovery

### Data Backup
```bash
# Backup indexes
tar -czf lethe-indexes-$(date +%Y%m%d).tar.gz pools/ indexes/

# Database backup
docker exec milvus /backup.sh
docker exec weaviate /backup.sh
```

### Disaster Recovery
```bash
# Restore from backup
tar -xzf lethe-indexes-YYYYMMDD.tar.gz

# Restart services
docker-compose down
docker-compose up -d

# Validate restoration
./lethe-bench validate --results runs/
```

## Support

**Documentation:** https://docs.lethe.dev
**Issues:** https://github.com/lethe-ai/lethe/issues
**Community:** https://discord.gg/lethe-ai
**Enterprise:** enterprise@lethe.dev
