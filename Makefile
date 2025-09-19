# -----------------------------------------------------------------------------
# Lethe Monorepo Makefile
# Convenience wrappers around common Cargo workflows for the Rust workspace.
# -----------------------------------------------------------------------------

.PHONY: help install build lint fmt clippy test ci-local clean

help:
	@echo "Lethe (Rust workspace)"
	@echo "Commands:"
	@echo "  make install   Fetch workspace dependencies"
	@echo "  make build     Build all crates"
	@echo "  make lint      Run fmt + clippy"
	@echo "  make test      Execute the full test suite"
	@echo "  make ci-local  Run lint and tests"
	@echo "  make clean     Remove build artifacts"

install:
	@echo "📦 Fetching Cargo dependencies"
	@cargo fetch

build:
	@echo "🏗️  Building workspace"
	@cargo build --workspace --all-targets

fmt:
	@echo "🧹 Formatting workspace"
	@cargo fmt --all

clippy:
	@echo "🔍 Running clippy"
	@cargo clippy --workspace --all-targets --all-features

lint: fmt
	@cargo fmt --all -- --check
	@cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	@echo "🧪 Running tests"
	@cargo test --workspace

ci-local:
	@$(MAKE) lint
	@$(MAKE) test

clean:
	@echo "🧽 Cleaning build artifacts"
	@cargo clean
