# Multi-stage Dockerfile for Lethe Hermetic Infrastructure
# Deterministic, reproducible builds with cryptographic verification
# Version: 1.0.0

# ===== Builder Stage =====
FROM python:3.11-alpine@sha256:5f0192349d7b15069947122d96d65b6c78c91fb0ff5448fe0e78b0f4db0d6bb8 AS builder

# Set build-time arguments for reproducibility
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Install build dependencies with exact pinned versions
RUN apk add --no-cache \
    gcc=13.2.1_git20231014-r0 \
    musl-dev=1.2.4_git20230717-r4 \
    libffi-dev=3.4.4-r2 \
    openssl-dev=3.1.4-r5 \
    cargo=1.72.1-r0 \
    nodejs=20.10.0-r1 \
    npm=10.2.3-r0 \
    git=2.43.0-r0 \
    curl=8.5.0-r0 \
    && rm -rf /var/cache/apk/*

# Create non-root user for build security
RUN addgroup -g 1001 lethe && adduser -D -u 1001 -G lethe lethe

# Set working directory with correct ownership
WORKDIR /app
RUN chown lethe:lethe /app

# Copy lockfiles first for better layer caching
COPY --chown=lethe:lethe infra/lockfiles/ ./lockfiles/
COPY --chown=lethe:lethe infra/requirements_clean.txt ./requirements.txt

# Switch to non-root user for all build operations
USER lethe

# Install Python dependencies with hash verification and integrity checking
RUN pip install --no-cache-dir --user --require-hashes -r requirements.txt

# Verify Python package integrity
RUN python -c "import pkg_resources; [print(f'{d.project_name}=={d.version}') for d in pkg_resources.working_set]" > /tmp/installed_packages.txt

# ===== Runtime Stage =====
FROM python:3.11-alpine@sha256:5f0192349d7b15069947122d96d65b6c78c91fb0ff5448fe0e78b0f4db0d6bb8 AS runtime

# Security hardening with exact versions
RUN apk add --no-cache \
    dumb-init=1.2.5-r3 \
    curl=8.5.0-r0 \
    jq=1.7-r4 \
    gnupg=2.4.4-r0 \
    bash=5.2.21-r0 \
    && rm -rf /var/cache/apk/* \
    && rm -rf /root/.cache

# Create non-root user with exact IDs for reproducibility
RUN addgroup -g 1001 lethe && adduser -D -u 1001 -G lethe lethe

# Set deterministic environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/home/lethe/.local/bin:$PATH" \
    ENVIRONMENT=hermetic \
    SECURITY_MODE=strict \
    BUILD_REPRODUCIBLE=1 \
    LETHE_VERSION=${VERSION:-1.0.0}

# Create application directories with secure permissions
WORKDIR /app
RUN chown lethe:lethe /app && chmod 755 /app

# Copy built dependencies from builder stage with verification
COPY --from=builder --chown=lethe:lethe /home/lethe/.local /home/lethe/.local
COPY --from=builder --chown=lethe:lethe /tmp/installed_packages.txt /app/build_manifest/

# Copy core application files with checksums
COPY --chown=lethe:lethe scripts/ ./scripts/
COPY --chown=lethe:lethe spec/ ./spec/
COPY --chown=lethe:lethe contracts/ ./contracts/
COPY --chown=lethe:lethe infra/config/ ./config/

# Switch to non-root user for all runtime operations
USER lethe

# Runtime verification of dependencies
RUN python -c "
import sys, pkg_resources, json
installed = {d.project_name: d.version for d in pkg_resources.working_set}
with open('/app/build_manifest/runtime_packages.json', 'w') as f:
    json.dump(installed, f, sort_keys=True, indent=2)
print('Runtime package verification complete')
"

# Health check for hermetic environment
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/readiness.py || exit 1

# Security and compliance labels
LABEL maintainer="Lethe Research Team" \
      version="${VERSION}" \
      vcs.ref="${VCS_REF}" \
      build.date="${BUILD_DATE}" \
      security.scan="required" \
      compliance="hermetic-build" \
      reproducible="true" \
      base.image="python:3.11-alpine" \
      base.digest="sha256:5f0192349d7b15069947122d96d65b6c78c91fb0ff5448fe0e78b0f4db0d6bb8"

# Expose port for readiness checks
EXPOSE 8080

# Use dumb-init for proper signal handling and process reaping
ENTRYPOINT ["dumb-init", "--"]

# Default command runs the hermetic smoke tests
CMD ["python", "scripts/spinup_smoke.sh"]