# Contributing to BayesianGPU

Thank you for your interest in contributing to BayesianGPU! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- **Rust 1.70+** with `wasm32-unknown-unknown` target
- **Node.js 18+**

### Setup

```bash
# Clone the repository
git clone https://github.com/mojavedataops/bayesiangpu-core.git
cd bayesiangpu-core

# Install Rust WASM target
rustup target add wasm32-unknown-unknown

# Build everything
cargo build
cd js && npm install && npm run build
```

## Development Workflow

### Rust Crates

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p bayesian-core
cargo test -p bayesian-diagnostics

# Run benchmarks
cargo bench -p bayesian-sampler

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --workspace -- -D warnings
```

### JavaScript SDK

```bash
cd js

# Build WASM and TypeScript
npm run build

# Run tests
npm test

# Watch mode for development
npm run test:watch
```

## Code Style

### Rust

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Run `cargo fmt` before committing
- All public items must have documentation
- Use `#[must_use]` for functions with important return values

### TypeScript

- Use TypeScript strict mode
- Document public APIs with JSDoc comments
- Use `const` by default, `let` only when necessary

## Testing

### Property-Based Testing

For numerical algorithms, use property-based testing:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_rhat_converged_chains(
        chain in prop::collection::vec(
            prop::collection::vec(-10.0..10.0f64, 50..100),
            2..6
        )
    ) {
        // Test properties here
    }
}
```

### Reference Comparisons

Compare implementations against reference libraries:

```rust
#[test]
fn test_ess_matches_arviz() {
    // Compare against ArviZ reference values
}
```

## Pull Request Process

1. **Create an issue** first for significant changes
2. **Fork and branch** from `main`
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run all checks** before submitting:
   ```bash
   cargo fmt --check
   cargo clippy --workspace -- -D warnings
   cargo test --workspace
   ```
6. **Create PR** with clear description

### PR Title Format

Use conventional commits:
- `feat: Add new distribution type`
- `fix: Correct ESS calculation for short chains`
- `docs: Update API reference`
- `test: Add property tests for R-hat`
- `refactor: Simplify leapfrog implementation`

## Architecture

### Crate Dependencies

```
bayesian-core (distributions)
    ↓
bayesian-rng (GPU random numbers)
    ↓
bayesian-sampler (HMC/NUTS)
    ↓
bayesian-diagnostics (R-hat, ESS)
    ↓
bayesian-wasm (browser bindings)
```

### Key Design Decisions

1. **Generic over backend**: Use Burn's backend traits for portability
2. **Tensor-based**: Operations use tensors for GPU acceleration
3. **Stateless samplers**: Samplers don't hold state between calls
4. **ArviZ compatibility**: Match ArviZ diagnostic formulas

## Areas for Contribution

### High Priority

- [ ] Additional distributions (StudentT, Cauchy, LogNormal)
- [ ] Mass matrix adaptation improvements
- [ ] WebGPU compute shader integration
- [ ] Documentation and tutorials

### Medium Priority

- [ ] Variational inference
- [ ] Model comparison (WAIC, LOO)
- [ ] Automatic reparameterization
- [ ] Browser-based visualization

### Good First Issues

Look for issues labeled `good-first-issue` in GitHub Issues.

## Questions?

- Open a GitHub Discussion for questions
- File an issue for bugs
- Email maintainers for security concerns

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
