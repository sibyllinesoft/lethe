# Contributing to Lethe

Thanks for helping improve Lethe! The repository is intentionally lean, so even small improvements make a difference.

## Development Flow

1. Fork the repository and clone your fork.
2. Run `cargo fetch` (or `make install`) to prime the Rust workspace.
3. Use feature branches named like `feature/short-description` or `fix/issue-id`.
4. Make your changes and update or add tests under the relevant Rust crate.
5. Run `cargo fmt`, `cargo clippy`, and `cargo test` (or `make ci-local`) before opening a pull request.

## Coding Guidelines

- The Rust workspace is organised by bounded context. Keep changes within the appropriate crate (`api`, `cli`, `domain`, `infrastructure`, etc.).
- Prefer small, composable modules with explicit interfaces and thorough error handling.
- Keep comments concise and helpful—explain intent or edge cases rather than restating code.
- Follow idiomatic Rust patterns (`Result` returns, `?` propagation, builder structs for complex configuration).

## Testing

We rely on the standard Cargo toolchain. Please ensure:

- `cargo fmt --all -- --check` passes (style)
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes (lint)
- `cargo test --workspace` passes (unit + integration tests)

When you add new behaviour, include targeted tests in the relevant crate (e.g.
`packages/rust/lethe-core/crates/domain/tests` or inline module tests).

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add context pack API`
- `fix: handle empty transcripts`
- `docs: clarify analyzer setup`

## Pull Request Checklist

- [ ] Lint passes (`cargo fmt`, `cargo clippy`)
- [ ] Tests pass (`cargo test`)
- [ ] New behaviour is documented in `README.md` when necessary
- [ ] CI workflow remains green

We appreciate your contributions—thank you!
