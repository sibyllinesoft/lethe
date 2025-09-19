# Lethe Tests

This directory contains all tests for the Lethe monorepo.

## Structure

```
tests/
├── unit/           # Unit tests for individual packages
│   ├── api-server/ # API server tests
│   ├── cli/        # CLI package tests
│   ├── core/       # Core functionality tests
│   ├── tokenizer/  # Tokenizer package tests
│   └── types/      # Types package tests
├── integration/    # Integration tests (future)
└── e2e/           # End-to-end tests (future)
```

## Running Tests

```bash
# Run all tests
bun test

# Run specific test file
bun test tests/unit/core/core.test.ts

# Run tests with coverage
bun test --coverage
```

## Test Framework

Tests use Bun's built-in test runner with the following imports:

```typescript
import { expect, test, describe, beforeEach, afterEach } from "bun:test";
```

## Current Coverage

- **Types Package**: Basic type structure validation
- **Core Package**: Math operations, configuration validation, candidate processing
- **API Server**: Route configuration, request validation
- **CLI Package**: Command parsing, configuration handling, output formatting
- **Tokenizer Package**: Basic tokenization logic

All tests are self-contained and don't require external dependencies or complex setup.
