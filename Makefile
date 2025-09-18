# -----------------------------------------------------------------------------
# Lethe Monorepo Makefile
# Provides a thin layer around Bun workspace commands so that contributors
# have a single entry point for common actions.
# -----------------------------------------------------------------------------

WORKSPACES := api-server cli core llm-analyzer tokenizer types

.PHONY: help install build test lint ci-local clean

help:
	@echo "Lethe Monorepo"
	@echo "Commands:"
	@echo "  make install   Install dependencies for all workspaces"
	@echo "  make build     Build every package"
	@echo "  make test      Run the Bun test suite"
	@echo "  make lint      Type-check all packages"
	@echo "  make ci-local  Execute the GitHub Actions workflow via act"
	@echo "  make clean     Remove generated artifacts"

install:
	@echo "📦 Installing workspace dependencies"
	@bun install

build:
	@for dir in $(WORKSPACES); do \
		printf "🏗️  Building %s\\n" $$dir; \
		(cd $$dir && bun run build); \
	done

lint:
	@echo "🔍 Type-checking workspace"
	@bunx tsc --noEmit

test:
	@echo "🧪 Running tests" 
	@bun test ./tests

ci-local:
	@echo "🤖 Running GitHub Actions workflow locally"
	@act push --container-architecture linux/amd64

clean:
	@echo "🧹 Cleaning workspace"
	@find . -name 'node_modules' -type d -prune -exec rm -rf {} +
	@find . -name 'dist' -type d -prune -exec rm -rf {} +
	@rm -rf ./**/bun.lockb
