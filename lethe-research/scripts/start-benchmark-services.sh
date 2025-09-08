#!/bin/bash

# Benchmark Services Startup Script
# Starts all required services for comprehensive Lethe benchmarking

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INFRA_DIR="$PROJECT_ROOT/infra"

echo "🚀 Starting Lethe Benchmark Services"
echo "=====================================\n"

# Function to wait for service health
wait_for_service() {
    local service_name=$1
    local health_check=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if eval "$health_check" &>/dev/null; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 5
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start within timeout"
    return 1
}

# Create required directories
echo "📁 Creating volume directories..."
cd "$INFRA_DIR"
mkdir -p volumes/{etcd,minio,milvus,redis,ollama,benchmark-results,weaviate,opensearch,elasticsearch,vespa}

# Start core services first (lightweight)
echo "🔄 Starting core services..."
docker-compose -f docker-compose.benchmarks.yml up -d redis

# Wait for Redis
wait_for_service "Redis" "redis-cli -h localhost -p 6379 ping | grep -q PONG"

# Start Weaviate (most important for benchmarking)
echo "🔄 Starting Weaviate..."
docker-compose -f docker-compose.benchmarks.yml up -d weaviate

# Wait for Weaviate
wait_for_service "Weaviate" "curl -f http://localhost:8081/v1/.well-known/ready"

# Start Ollama for LLM inference
echo "🔄 Starting Ollama..."
docker-compose -f docker-compose.benchmarks.yml up -d ollama

# Wait for Ollama
wait_for_service "Ollama" "curl -f http://localhost:11434/api/tags"

# Start OpenSearch (lighter than full Elasticsearch)
echo "🔄 Starting OpenSearch..."
docker-compose -f docker-compose.benchmarks.yml up -d opensearch

# Wait for OpenSearch
wait_for_service "OpenSearch" "curl -f http://localhost:9200/_cluster/health"

# Start Milvus dependencies
echo "🔄 Starting Milvus dependencies..."
docker-compose -f docker-compose.benchmarks.yml up -d etcd minio

# Wait for dependencies
wait_for_service "etcd" "docker exec lethe-milvus-etcd etcdctl endpoint health"
wait_for_service "MinIO" "curl -f http://localhost:9000/minio/health/live"

# Start Milvus
echo "🔄 Starting Milvus..."
docker-compose -f docker-compose.benchmarks.yml up -d milvus-standalone

# Wait for Milvus
wait_for_service "Milvus" "curl -f http://localhost:9091/healthz"

# Start remaining services
echo "🔄 Starting remaining services..."
docker-compose -f docker-compose.benchmarks.yml up -d elasticsearch vespa

# Optional: Start benchmark runner if Dockerfile exists
if [ -f "$INFRA_DIR/../Dockerfile" ]; then
    echo "🔄 Building and starting benchmark runner..."
    docker-compose -f docker-compose.benchmarks.yml up -d benchmark-runner
else
    echo "ℹ️  No Dockerfile found - skipping benchmark runner container"
fi

# Display service status
echo "\n📊 Service Status:"
echo "==================="
docker-compose -f docker-compose.benchmarks.yml ps

# Display connection info
echo "\n🔗 Service Endpoints:"
echo "====================="
echo "Weaviate:     http://localhost:8081"
echo "Milvus:       localhost:19530"
echo "OpenSearch:   http://localhost:9200"
echo "Elasticsearch: http://localhost:9201"  
echo "Vespa:        http://localhost:8080"
echo "Redis:        localhost:6379"
echo "Ollama:       http://localhost:11434"
echo "MinIO:        http://localhost:9001 (admin: minioadmin/minioadmin)"

echo "\n✅ Benchmark services started successfully!"
echo "💡 Run the comprehensive evaluation with:"
echo "   cd $PROJECT_ROOT && python3 -m src.infinitebench.comprehensive_evaluation --baseline-families hybrid_vector_dbs learned_sparse"