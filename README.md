# Lethe

A modern context-aware retrieval system built with TypeScript and Bun.

## Project Structure

This is a clean, feature-based monorepo organized for maintainability and modern development workflows:

```
lethe/
├── api-server/          # Lightweight API server for the analyzer UI
├── cli/                 # Command-line interface
├── core/                # Core retrieval logic and algorithms
├── llm-analyzer/        # React frontend for LLM call analysis
├── tokenizer/           # Standalone tokenizer utilities
├── types/               # Shared TypeScript types
├── tests/               # Centralized testing (unit/integration/e2e)
├── deployment/          # Deployment configurations
├── .github/workflows/   # CI/CD pipelines
├── bunfig.toml          # Bun workspace configuration
└── Makefile             # Development automation
```

## Quick Start

### Prerequisites

- [Bun](https://bun.sh/) - Fast all-in-one JavaScript runtime
- [Docker](https://docker.com/) - For local CI testing (optional)
- [act](https://github.com/nektos/act) - Local GitHub Actions testing (optional)

### Development Setup

```bash
# Install all dependencies
make install

# Build all packages
make build

# Run tests
make test

# Lint code
make lint

# Run local CI pipeline
make ci-local
```

### Package Development

Each package can be developed independently:

```bash
# Core package
cd core/
bun dev

# CLI package  
cd cli/
bun dev

# API server
cd api-server/
bun dev

# LLM analyzer (React app)
cd llm-analyzer/
bun dev
```

## Architecture

### Core Packages

- **@lethe/core** - Context retrieval algorithms and processing pipelines
- **@lethe/cli** - Command-line interface for context management
- **@lethe/api-server** - Lightweight server for the analyzer frontend
- **@lethe/llm-analyzer** - React application for analyzing LLM calls
- **@lethe/tokenizer** - Text tokenization utilities
- **@lethe/types** - Shared TypeScript type definitions

### Key Features

- **Modern Tooling** - Built with Bun for speed and simplicity
- **Type Safety** - Full TypeScript coverage with shared type definitions
- **Workspace Management** - Efficient dependency management across packages
- **Automated Testing** - Centralized test suite with coverage reporting
- **Local CI** - Test GitHub Actions locally with `act`
- **Clean Architecture** - Feature-based organization, no language silos

## Development Workflow

1. **Make Changes** - Edit code in any package
2. **Test Locally** - `make test` runs all tests
3. **Build** - `make build` compiles all packages
4. **Local CI** - `make ci-local` runs the full pipeline
5. **Commit** - Standard git workflow

## License

MIT License - see [LICENSE](LICENSE) for details.
