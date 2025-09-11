#!/bin/bash
# Lethe Determinism Service V2.1.0 - Emergency Rollback Script
# Generated on: 2025-09-10
# Review this script before execution

set -euo pipefail  # Exit on any error

echo "🚨 Lethe Determinism Service Emergency Rollback"
echo "WARNING: This script will rollback to a previous version"

# Configuration
CURRENT_VERSION="v2.1.0"
ROLLBACK_VERSION="${1:-v2.0.9}"
SERVICE_NAME="determinism-service"
NAMESPACE="${NAMESPACE:-default}"
BACKUP_DIR="./backups"
TIMEOUT_SECONDS=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Detect deployment type
detect_deployment_type() {
    if kubectl get pods --no-headers 2>/dev/null | grep -q "$SERVICE_NAME"; then
        echo "kubernetes"
    elif docker ps --format "table {{.Names}}" | grep -q "$SERVICE_NAME"; then
        echo "docker"
    elif pgrep -f "$SERVICE_NAME" > /dev/null; then
        echo "native"
    else
        echo "unknown"
    fi
}

# Confirm rollback with user
confirm_rollback() {
    log_warn "🚨 ROLLBACK CONFIRMATION REQUIRED 🚨"
    echo ""
    echo "Current Version: $CURRENT_VERSION"
    echo "Rollback Target: $ROLLBACK_VERSION"
    echo "Service: $SERVICE_NAME"
    echo "Deployment Type: $DEPLOYMENT_TYPE"
    echo ""
    log_warn "This will:"
    echo "  1. Stop the current service"
    echo "  2. Restore previous version configuration"
    echo "  3. Restart with the previous version"
    echo "  4. Validate the rollback"
    echo ""
    
    read -p "Are you sure you want to proceed with rollback? (type 'ROLLBACK' to confirm): " confirmation
    
    if [ "$confirmation" != "ROLLBACK" ]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
    
    log_info "Rollback confirmed. Proceeding..."
}

# Create backup of current state
create_backup() {
    log_step "Creating backup of current deployment state..."
    
    mkdir -p "$BACKUP_DIR"
    local backup_file="$BACKUP_DIR/backup_${CURRENT_VERSION}_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    # Backup configuration files and current state
    case $DEPLOYMENT_TYPE in
        "kubernetes")
            kubectl get deployment $SERVICE_NAME -o yaml > "$BACKUP_DIR/deployment_backup.yaml" 2>/dev/null || true
            kubectl get service $SERVICE_NAME -o yaml > "$BACKUP_DIR/service_backup.yaml" 2>/dev/null || true
            kubectl get configmap ${SERVICE_NAME}-config -o yaml > "$BACKUP_DIR/configmap_backup.yaml" 2>/dev/null || true
            ;;
        "docker")
            docker inspect $SERVICE_NAME > "$BACKUP_DIR/docker_inspect_backup.json" 2>/dev/null || true
            ;;
        "native")
            cp -r . "$BACKUP_DIR/native_backup/" 2>/dev/null || true
            ;;
    esac
    
    # Create compressed backup
    tar -czf "$backup_file" -C "$BACKUP_DIR" . 2>/dev/null || true
    
    log_info "✅ Backup created: $backup_file"
}

# Rollback Kubernetes deployment
rollback_kubernetes() {
    log_step "Rolling back Kubernetes deployment..."
    
    # Check if deployment exists
    if ! kubectl get deployment $SERVICE_NAME -n $NAMESPACE > /dev/null 2>&1; then
        log_error "Deployment $SERVICE_NAME not found in namespace $NAMESPACE"
        return 1
    fi
    
    # Get rollback revision (previous version)
    local previous_revision=$(kubectl rollout history deployment/$SERVICE_NAME -n $NAMESPACE | tail -n 2 | head -n 1 | awk '{print $1}' || echo "")
    
    if [ -z "$previous_revision" ]; then
        log_warn "No previous revision found. Updating image tag to $ROLLBACK_VERSION"
        kubectl set image deployment/$SERVICE_NAME ${SERVICE_NAME}=determinism-service:$ROLLBACK_VERSION -n $NAMESPACE
    else
        log_info "Rolling back to revision $previous_revision"
        kubectl rollout undo deployment/$SERVICE_NAME --to-revision=$previous_revision -n $NAMESPACE
    fi
    
    # Wait for rollout to complete
    log_info "Waiting for rollout to complete..."
    kubectl rollout status deployment/$SERVICE_NAME -n $NAMESPACE --timeout=${TIMEOUT_SECONDS}s
    
    # Verify pods are running
    local ready_pods=$(kubectl get deployment $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    local desired_pods=$(kubectl get deployment $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    
    if [ "$ready_pods" == "$desired_pods" ] && [ "$ready_pods" -gt 0 ]; then
        log_info "✅ Kubernetes rollback completed successfully"
        return 0
    else
        log_error "❌ Kubernetes rollback failed: $ready_pods/$desired_pods pods ready"
        return 1
    fi
}

# Rollback Docker deployment
rollback_docker() {
    log_step "Rolling back Docker deployment..."
    
    # Stop current container
    if docker ps --format "table {{.Names}}" | grep -q "$SERVICE_NAME"; then
        log_info "Stopping current container..."
        docker stop $SERVICE_NAME || true
        docker rm $SERVICE_NAME || true
    fi
    
    # Check if rollback image exists
    if ! docker images | grep -q "determinism-service.*$ROLLBACK_VERSION"; then
        log_warn "Rollback image not found locally. Attempting to pull..."
        docker pull determinism-service:$ROLLBACK_VERSION || {
            log_error "Failed to pull rollback image"
            return 1
        }
    fi
    
    # Start with rollback version
    log_info "Starting container with version $ROLLBACK_VERSION..."
    
    docker run -d \
        --name $SERVICE_NAME \
        -p 8080:8080 \
        -e RUST_LOG=info \
        -e ENVIRONMENT=production \
        determinism-service:$ROLLBACK_VERSION
    
    # Wait for container to be healthy
    local attempts=0
    while [ $attempts -lt 30 ]; do
        if docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$SERVICE_NAME" | grep -q "healthy\|Up"; then
            log_info "✅ Docker rollback completed successfully"
            return 0
        fi
        sleep 5
        ((attempts++))
    done
    
    log_error "❌ Docker rollback failed: container not healthy after 150 seconds"
    return 1
}

# Rollback native deployment
rollback_native() {
    log_step "Rolling back native deployment..."
    
    # Stop current process
    local pids=$(pgrep -f "$SERVICE_NAME" || echo "")
    if [ -n "$pids" ]; then
        log_info "Stopping current service processes: $pids"
        kill $pids
        sleep 5
        
        # Force kill if still running
        local remaining_pids=$(pgrep -f "$SERVICE_NAME" || echo "")
        if [ -n "$remaining_pids" ]; then
            log_warn "Force killing processes: $remaining_pids"
            kill -9 $remaining_pids
        fi
    fi
    
    # Check if rollback binary exists
    local rollback_binary="./target/release/${SERVICE_NAME}-${ROLLBACK_VERSION}"
    if [ ! -f "$rollback_binary" ]; then
        log_warn "Rollback binary not found. Attempting to build from git tag..."
        
        # Try to checkout and build rollback version
        git fetch --tags 2>/dev/null || true
        if git rev-parse --verify "refs/tags/$ROLLBACK_VERSION" > /dev/null 2>&1; then
            git checkout $ROLLBACK_VERSION
            cargo build --release
            log_info "Built rollback version from source"
        else
            log_error "Cannot find rollback version $ROLLBACK_VERSION in git"
            return 1
        fi
    fi
    
    # Start rollback version
    log_info "Starting service with version $ROLLBACK_VERSION..."
    
    RUST_LOG=info ENVIRONMENT=production nohup ./target/release/determinism-service > service.log 2>&1 &
    
    # Wait for service to be ready
    local attempts=0
    while [ $attempts -lt 30 ]; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            log_info "✅ Native rollback completed successfully"
            return 0
        fi
        sleep 5
        ((attempts++))
    done
    
    log_error "❌ Native rollback failed: service not responding after 150 seconds"
    return 1
}

# Validate rollback success
validate_rollback() {
    log_step "Validating rollback success..."
    
    # Test health endpoint
    local health_attempts=0
    while [ $health_attempts -lt 10 ]; do
        if curl -s "http://localhost:8080/health" > /dev/null 2>&1; then
            log_info "✅ Health check: PASS"
            break
        fi
        sleep 3
        ((health_attempts++))
    done
    
    if [ $health_attempts -eq 10 ]; then
        log_error "❌ Health check: FAIL"
        return 1
    fi
    
    # Test version endpoint
    local version_response=$(curl -s "http://localhost:8080/version" 2>/dev/null || echo "{}")
    if echo "$version_response" | grep -q "$ROLLBACK_VERSION"; then
        log_info "✅ Version verification: PASS (confirmed $ROLLBACK_VERSION)"
    else
        log_warn "⚠️  Version verification: Cannot confirm version from response"
    fi
    
    # Test basic functionality
    if curl -s "http://localhost:8080/metrics" > /dev/null 2>&1; then
        log_info "✅ Metrics endpoint: PASS"
    else
        log_warn "⚠️  Metrics endpoint: Not responding"
    fi
    
    log_info "✅ Rollback validation completed"
    return 0
}

# Send rollback notification
send_notification() {
    local status="$1"
    local timestamp=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    
    log_step "Sending rollback notification..."
    
    local message
    if [ "$status" == "SUCCESS" ]; then
        message="🟢 ROLLBACK SUCCESSFUL: Lethe Determinism Service rolled back from $CURRENT_VERSION to $ROLLBACK_VERSION at $timestamp"
    else
        message="🔴 ROLLBACK FAILED: Lethe Determinism Service rollback from $CURRENT_VERSION to $ROLLBACK_VERSION failed at $timestamp"
    fi
    
    # Log to local file
    echo "$message" >> rollback.log
    
    # Here you would integrate with your notification systems
    # Examples:
    # - Slack webhook
    # - Email notification
    # - PagerDuty alert
    # - Teams webhook
    
    log_info "Notification logged to rollback.log"
    echo "$message"
}

# Main rollback function
main() {
    local start_time=$(date)
    
    log_info "🚨 Starting emergency rollback procedure..."
    log_info "Start time: $start_time"
    
    # Detect deployment type
    DEPLOYMENT_TYPE=$(detect_deployment_type)
    log_info "Detected deployment type: $DEPLOYMENT_TYPE"
    
    if [ "$DEPLOYMENT_TYPE" == "unknown" ]; then
        log_error "Cannot detect deployment type. Service may not be running."
        exit 1
    fi
    
    # Confirm with user
    confirm_rollback
    
    # Create backup
    create_backup
    
    # Perform rollback based on deployment type
    local rollback_success=false
    case $DEPLOYMENT_TYPE in
        "kubernetes")
            if rollback_kubernetes; then
                rollback_success=true
            fi
            ;;
        "docker")
            if rollback_docker; then
                rollback_success=true
            fi
            ;;
        "native")
            if rollback_native; then
                rollback_success=true
            fi
            ;;
        *)
            log_error "Unsupported deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    if [ "$rollback_success" == true ]; then
        # Validate rollback
        if validate_rollback; then
            local end_time=$(date)
            send_notification "SUCCESS"
            log_info "🎉 Emergency rollback completed successfully!"
            log_info "Start time: $start_time"
            log_info "End time: $end_time"
            log_info "Rollback from $CURRENT_VERSION to $ROLLBACK_VERSION completed"
            exit 0
        else
            send_notification "VALIDATION_FAILED"
            log_error "🚨 Rollback completed but validation failed!"
            exit 1
        fi
    else
        send_notification "FAILED"
        log_error "🚨 Emergency rollback failed!"
        log_error "Manual intervention required"
        exit 1
    fi
}

# Show help
show_help() {
    echo "Lethe Determinism Service Emergency Rollback Script"
    echo ""
    echo "Usage: $0 [ROLLBACK_VERSION]"
    echo ""
    echo "Arguments:"
    echo "  ROLLBACK_VERSION    Version to rollback to (default: v2.0.9)"
    echo ""
    echo "Environment Variables:"
    echo "  NAMESPACE          Kubernetes namespace (default: default)"
    echo ""
    echo "Examples:"
    echo "  $0                 # Rollback to v2.0.9"
    echo "  $0 v2.0.8         # Rollback to v2.0.8"
    echo "  NAMESPACE=prod $0 v2.0.9  # Rollback in prod namespace"
    echo ""
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac