#!/bin/bash
# Lethe Determinism Service V2.1.0 - Monitoring Setup
# Generated on: 2025-09-10
# Review this script before execution

set -euo pipefail  # Exit on any error

echo "🔍 Checking system requirements..."

# Validation checks
check_requirements() {
    echo "Checking Docker/Kubernetes availability..."
    if command -v kubectl &> /dev/null; then
        echo "✅ Kubernetes CLI found"
        DEPLOYMENT_TYPE="kubernetes"
    elif command -v docker &> /dev/null; then
        echo "✅ Docker found"
        DEPLOYMENT_TYPE="docker"
    else
        echo "❌ Neither Docker nor Kubernetes found"
        exit 1
    fi
    
    echo "Checking system resources..."
    TOTAL_MEM=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    CPU_CORES=$(nproc)
    
    if [ "$TOTAL_MEM" -lt 8 ]; then
        echo "⚠️  Warning: Less than 8GB RAM detected ($TOTAL_MEM GB)"
    fi
    
    if [ "$CPU_CORES" -lt 4 ]; then
        echo "⚠️  Warning: Less than 4 CPU cores detected ($CPU_CORES cores)"
    fi
    
    echo "✅ System check complete: ${CPU_CORES} cores, ${TOTAL_MEM}GB RAM"
}

# Setup Prometheus configuration
setup_prometheus() {
    echo "📦 Setting up Prometheus configuration..."
    
    mkdir -p monitoring/prometheus
    
    cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules.yml"

scrape_configs:
  - job_name: 'determinism-service'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
      
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
EOF

    cat > monitoring/prometheus/rules.yml << 'EOF'
groups:
  - name: determinism-service
    rules:
      - alert: DeterminismRateBelow99_5
        expr: determinism_rate < 0.995
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Determinism rate below 99.5%"
          description: "Current rate: {{ $value }}%"
          
      - alert: ECEScoreAbove0_08
        expr: ece_score > 0.08
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ECE calibration score above threshold"
          description: "Current ECE: {{ $value }}"
          
      - alert: P95LatencyAbove200ms
        expr: p95_latency_ms > 200
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 200ms"
          description: "Current P95: {{ $value }}ms"
          
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker is OPEN"
          description: "Performance protection activated"
EOF

    echo "✅ Prometheus configuration created"
}

# Setup Grafana dashboards
setup_grafana() {
    echo "📊 Setting up Grafana dashboards..."
    
    mkdir -p monitoring/grafana
    
    # Main system dashboard
    cat > monitoring/grafana/determinism-overview.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Lethe Determinism Service V2.1.0 - Overview",
    "tags": ["lethe", "determinism", "v2"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "Determinism Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "determinism_rate",
            "legendFormat": "Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0.99,
            "max": 1.0,
            "unit": "percentunit"
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "ECE Calibration Score",
        "type": "stat",
        "targets": [
          {
            "expr": "ece_score",
            "legendFormat": "ECE"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 0.15,
            "unit": "short"
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "title": "P95 Latency",
        "type": "timeseries",
        "targets": [
          {
            "expr": "p95_latency_ms",
            "legendFormat": "P95 Latency"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ms"
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 4,
        "title": "Production Gates Status",
        "type": "table",
        "targets": [
          {
            "expr": "gate_status",
            "format": "table",
            "instant": true
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      }
    ],
    "refresh": "5s",
    "time": {
      "from": "now-1h",
      "to": "now"
    }
  }
}
EOF

    # Performance gates dashboard
    cat > monitoring/grafana/performance-gates.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Lethe Determinism Service - Performance Gates",
    "tags": ["lethe", "gates", "performance"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "All Gates Status",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(gate_status == 1)",
            "legendFormat": "GREEN Gates"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 12,
            "unit": "short"
          }
        },
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Continuous Validation Hours", 
        "type": "stat",
        "targets": [
          {
            "expr": "continuous_validation_hours",
            "legendFormat": "Hours"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 48,
            "unit": "h"
          }
        },
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0}
      },
      {
        "id": 3,
        "title": "Gate Trends (24h)",
        "type": "timeseries", 
        "targets": [
          {
            "expr": "rate(gate_transitions[1h])",
            "legendFormat": "Gate Changes/hour"
          }
        ],
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0}
      }
    ],
    "refresh": "30s",
    "time": {
      "from": "now-24h",
      "to": "now"
    }
  }
}
EOF

    echo "✅ Grafana dashboards created"
}

# Setup Docker monitoring stack
setup_docker_monitoring() {
    echo "🐳 Setting up Docker monitoring stack..."
    
    mkdir -p docker
    
    cat > docker/docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: lethe-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - lethe-monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: lethe-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning/dashboards
    networks:
      - lethe-monitoring

  node-exporter:
    image: prom/node-exporter:latest
    container_name: lethe-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - lethe-monitoring

  alertmanager:
    image: prom/alertmanager:latest
    container_name: lethe-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager:/etc/alertmanager
    networks:
      - lethe-monitoring

volumes:
  grafana_data:

networks:
  lethe-monitoring:
    driver: bridge
EOF

    echo "✅ Docker monitoring stack configuration created"
}

# Setup Kubernetes monitoring
setup_kubernetes_monitoring() {
    echo "☸️  Setting up Kubernetes monitoring..."
    
    mkdir -p k8s
    
    cat > k8s/monitoring-namespace.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: lethe-monitoring
  labels:
    name: lethe-monitoring
EOF

    cat > k8s/prometheus-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: lethe-monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: lethe-monitoring
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
EOF

    echo "✅ Kubernetes monitoring manifests created"
}

# Main setup function
main() {
    echo "🚀 Starting Lethe Determinism Service V2.1.0 monitoring setup..."
    
    check_requirements
    setup_prometheus
    setup_grafana
    
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        setup_docker_monitoring
        echo "🐳 To start monitoring stack: docker-compose -f docker/docker-compose.monitoring.yml up -d"
    elif [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
        setup_kubernetes_monitoring
        echo "☸️  To deploy monitoring: kubectl apply -f k8s/monitoring-namespace.yaml && kubectl apply -f k8s/"
    fi
    
    echo "✅ Monitoring setup complete!"
    echo ""
    echo "📊 Access URLs:"
    echo "- Grafana: http://localhost:3000 (admin/admin123)"
    echo "- Prometheus: http://localhost:9090"
    echo "- Node Exporter: http://localhost:9100"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Start the monitoring stack"
    echo "2. Run ./validate-deployment.sh to verify setup"
    echo "3. Deploy the determinism service"
    echo "4. Import Grafana dashboards from monitoring/grafana/"
}

# Execute main function
main "$@"