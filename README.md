# Lethe Monorepo

Lethe is a retrieval-augmented generation (RAG) platform that now ships as a
Rust-first workspace. The repository contains the production API, CLI, domain
logic, and high-performance search components that power the system end to end.

## Project Structure

- `packages/rust/lethe-core/` – primary workspace with the API, CLI, shared
  application state, and persistence layer.
- (removed) `packages/rust/hotpath/` – legacy placeholder bindings retired in favour of the Rust core pipeline.
- `archive/typescript-stack/` – frozen Bun/TypeScript implementation kept for
  historical reference. It is no longer maintained or included in the build.

## Prerequisites

- Rust toolchain 1.75+ with `cargo`, `rustfmt`, and `clippy` components
- PostgreSQL 15+ if you plan to run the API end to end
- `make` (optional; convenience wrapper around common Cargo commands)

## Getting Started

```bash
# Fetch dependencies and ensure toolchain is ready
cargo fetch

# Validate the codebase compiles
cargo check --workspace

# Run the full test suite
cargo test --workspace

# Start the API server (see --help for configuration flags)
cargo run -p lethe-api -- --host 0.0.0.0 --port 3000

# Use the CLI to ingest data or query the pipeline
cargo run -p lethe-cli -- ingest --file ./docs.json --session demo
cargo run -p lethe-cli -- query demo "latency metrics"
```

Configuration lives in `lethe.json` (or the path supplied via
`--config`). Secrets such as database URLs or LLM tokens should be provided via
environment variables.

## Tests & Quality Gates

The Rust workspace relies on standard tooling:

```bash
# Style checks
cargo fmt --all -- --check

# Linting
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Unit + integration tests
cargo test --workspace
```

Make targets mirror the same commands for convenience (`make lint`, `make
build`, `make test`).

## Repository Layout

```
.
├── packages/
│   ├── rust/
│   │   ├── lethe-core/        # API, CLI, domain, infrastructure crates
├── deployment/                # Infrastructure manifests and scripts
├── archive/
│   └── typescript-stack/      # Legacy Bun/TypeScript implementation (read-only)
├── Cargo.toml                 # Rust workspace definition
├── Makefile                   # Convenience wrappers for Cargo commands
└── README.md
```

## Legacy TypeScript Stack

The former Bun/TypeScript implementation has been moved to
`archive/typescript-stack/` and excluded from the active build. It remains
available for reference or archaeological purposes only. New work should target
the Rust crates described above.

## Contributing

1. Create a feature branch from `main`.
2. Run `cargo fmt`, `cargo clippy`, and `cargo test` before opening a PR.
3. Include tests for any behavioural change when practical.
4. Update documentation when you introduce new commands or configuration.

See `CONTRIBUTING.md` for additional guidelines.
