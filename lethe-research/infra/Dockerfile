# Multi-stage Dockerfile for Lethe Research Infrastructure
# Security-hardened container with deterministic builds

# ===== Builder Stage =====
FROM python:3.11-alpine AS builder

# Install build dependencies with pinned versions
RUN apk add --no-cache \
    gcc=13.2.1_git20231014-r0 \
    musl-dev=1.2.4_git20230717-r4 \
    libffi-dev=3.4.4-r2 \
    openssl-dev=3.1.4-r5 \
    cargo=1.72.1-r0 \
    nodejs=20.10.0-r1 \
    npm=10.2.3-r0

# Create non-root user for build
RUN addgroup -g 1001 lethe && adduser -D -u 1001 -G lethe lethe

# Set working directory
WORKDIR /app

# Copy dependency files
COPY --chown=lethe:lethe infra/requirements.txt .
COPY --chown=lethe:lethe infra/package.json .
COPY --chown=lethe:lethe infra/package-lock.json .

# Switch to non-root user
USER lethe

# Install Python dependencies with hash verification
RUN pip install --no-cache-dir --user -r requirements.txt

# Install Node.js dependencies with integrity check
RUN npm ci --only=production

# ===== Runtime Stage =====
FROM python:3.11-alpine AS runtime

# Security hardening
RUN apk add --no-cache \
    dumb-init=1.2.5-r3 \
    curl=8.5.0-r0 \
    jq=1.7-r4 \
    gnupg=2.4.4-r0 \
    && rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 lethe && adduser -D -u 1001 -G lethe lethe

# Set security-hardened environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/home/lethe/.local/bin:$PATH" \
    ENVIRONMENT=production \
    SECURITY_MODE=strict

# Create application directories with correct permissions
WORKDIR /app
RUN chown lethe:lethe /app

# Copy built dependencies from builder stage
COPY --from=builder --chown=lethe:lethe /home/lethe/.local /home/lethe/.local
COPY --from=builder --chown=lethe:lethe /app/node_modules ./node_modules

# Copy application code
COPY --chown=lethe:lethe . .

# Switch to non-root user
USER lethe

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Security labels
LABEL maintainer="Lethe Research Team" \
      version="1.0.0" \
      security.scan="required" \
      compliance="hermetic-build"

# Expose port
EXPOSE 8080

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Default command
CMD ["python", "-m", "scripts.spinup_smoke"]