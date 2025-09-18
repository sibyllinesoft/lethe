# Rust Packages

This directory collects all Rust-based components for Lethe so they can be
managed from a single location:

- `lethe-core/` – primary workspace hosting the API, CLI, shared domain logic,
  and infrastructure crates.
- `determinism-service/` – monitoring and benchmarking service previously under
  the research storybook.
- `hotpath/` – the native hot path bindings that expose the optimized search
  routines (Rust → N-API).

Each subproject keeps its original `Cargo.toml` (or workspace) so you can run
`cargo` commands from within each directory. Future additions should follow the
same pattern to keep Rust code discoverable.
