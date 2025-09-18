# Contributing to Lethe

Thanks for helping improve Lethe! The repository is intentionally lean, so even small improvements make a difference.

## Development Flow

1. Fork the repository and clone your fork.
2. Run `bun install` (or `make install`) to install dependencies for every workspace.
3. Use feature branches named like `feature/short-description` or `fix/issue-id`.
4. Make your changes and update or add tests under `tests/`.
5. Run `make build` and `make test` before opening a pull request.

## Coding Guidelines

- The monorepo is organised by feature. Keep package-specific code inside the matching folder (`core/`, `cli/`, etc.).
- Prefer small, composable modules with explicit exports.
- Keep comments concise and helpful—focus on explaining intent, not restating code.
- TypeScript should compile with `strict` mode; avoid `any` unless absolutely necessary and justified.

## Testing

We rely on Bun's test runner and a centralised `tests/` directory. If you change behaviour, add or update tests in the appropriate package folder:

```
tests/
├── unit/core
├── unit/cli
├── unit/api-server
├── unit/tokenizer
└── unit/types
```

Use `make test` to execute the suite. For a single file, run `bun test tests/unit/core/core.test.ts`.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add context pack API`
- `fix: handle empty transcripts`
- `docs: clarify analyzer setup`

## Pull Request Checklist

- [ ] Tests pass (`make test`)
- [ ] Build succeeds (`make build`)
- [ ] New behaviour is documented in `README.md` when necessary
- [ ] CI workflow remains green

We appreciate your contributions—thank you!
