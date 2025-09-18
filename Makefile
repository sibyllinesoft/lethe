# Lethe Monorepo Makefile

.PHONY: help install build test lint ci-local clean

# Default command: show help
help:
	@echo "Lethe Monorepo Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install all dependencies using bun"
	@echo "  make build        Build all workspace packages"
	@echo "  make test         Run all unit and integration tests"
	@echo "  make lint         Lint all workspace packages"
	@echo "  make ci-local     Run the full CI pipeline locally using act"
	@echo "  make clean        Remove build artifacts and dependencies"
	@echo ""

# Install all dependencies for all workspace packages
install:
	@echo "📦 Installing all dependencies..."
	@echo "Dependencies are managed per-package. Use 'cd <package> && bun install' as needed."

# Build all packages in the workspace
build:
	@echo "🏗️ Building all packages..."
	@echo "Build individual packages with 'cd <package> && bun run build'."

# Run all tests
test:
	@echo "🧪 Running all tests..."
	bun test

# Lint all packages
lint:
	@echo "🔍 Linting all packages..."
	@echo "Lint individual packages with 'cd <package> && bun run lint'."

# Run the full CI pipeline locally using act (requires Docker)
ci-local:
	@echo "🤖 Running local CI with act..."
	act push --container-architecture linux/amd64

# Clean up the repository
clean:
	@echo "🧹 Cleaning up workspace..."
	rm -rf ./**/node_modules
	rm -rf ./**/dist
	rm -rf ./**/build
	rm -rf ./coverage
