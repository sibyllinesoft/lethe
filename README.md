# Lethe Monorepo

Lethe is a feature-oriented monorepo for context-aware developer tooling. It contains:

- `@lethe/core` – the orchestration engine that builds conversation-aware context packs.
- `@lethe/cli` – a Bun-powered CLI for initializing workspaces, ingesting transcripts, querying context, and launching the API.
- `@lethe/api-server` – a minimal HTTP API that powers the web-based analyzer experience.
- `@lethe/llm-analyzer` – a React UI for exploring LLM call history and comparing prompts.
- `@lethe/tokenizer` and `@lethe/types` – shared utilities that keep implementations consistent.

## Prerequisites

- [Bun](https://bun.sh/) 1.1+
- `make` (installed by default on most UNIX-like systems)
- Docker (only required when running `make ci-local` with `act`)

## Getting Started

```bash
# Install all workspace dependencies
bun install
# (or run `make install`, which simply wraps the same command)

# Run the full lint/build/test pipeline
make lint
make build
make test

# Launch the API server and analyzer UI during development
bun run --cwd api-server src/dev.ts
bun run --cwd llm-analyzer dev
```

The CLI can bootstrap a local workspace and experiment with the core orchestration pipeline:

```bash
# Create a workspace in the current directory
bun run --cwd cli src/index.ts init

# Ingest a JSON transcript (array of { role, text })
bun run --cwd cli src/index.ts ingest --file ./transcript.json --session demo

# Ask for a context pack
bun run --cwd cli src/index.ts query demo "latency metrics"

# Start the API server through the CLI wrapper
bun run --cwd cli src/index.ts serve --port 3001
```

## Tests & Quality

The repository uses Bun's built-in test runner. Tests live under the top-level `tests/` directory and cover:

- Context orchestration behaviour (`tests/unit/core`)
- Tokenizer utilities (`tests/unit/tokenizer`)
- Workspace helpers and CLI entry points (`tests/unit/cli`)
- HTTP endpoints exposed by the API server (`tests/unit/api-server`)
- Shape checks for the shared `@lethe/types` package (`tests/unit/types`)

Run everything with:

```bash
make test
```

## Repository Layout

```
.
├── api-server/         # Bun + Elysia API serving the analyzer
├── cli/                # CLI for local workflows
├── core/               # Retrieval, scoring, and summarisation logic
├── llm-analyzer/       # React UI powered by the API server
├── tokenizer/          # Shared tokenizer utilities
├── types/              # Shared TypeScript interfaces
├── tests/              # Centralised test suites
├── deployment/         # Deployment assets (e.g. canary configs)
├── bunfig.toml         # Bun workspace configuration
└── Makefile            # Project automation entry points
```

## Continuous Integration

GitHub Actions runs the same steps as `make lint`, `make build`, and `make test`. You can execute the workflow locally with:

```bash
make ci-local
```

## Contributing

1. Fork the repository and create a feature branch.
2. Install dependencies via `make install`.
3. Add or update tests in `tests/` for any behaviour change.
4. Run `make build` and `make test` before opening a pull request.

See `CONTRIBUTING.md` for additional guidance.
