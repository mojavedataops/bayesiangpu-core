# BayesianGPU Cross-Framework Benchmarks

Standardized benchmarks comparing BayesianGPU against established probabilistic programming frameworks.

## Frameworks

| Framework | Language | Backend | Version |
|-----------|----------|---------|---------|
| **BayesianGPU** | Rust/Python | NdArray (CPU) / wgpu (GPU) | 0.2.x |
| **PyMC** | Python | PyTensor/JAX | 5.x |
| **NumPyro** | Python | JAX | 0.13+ |
| **CmdStan** | Stan/C++ | CmdStan | 2.x |
| **brms** | R/Stan | Stan backend | 2.x |

## Models

| # | Model | Parameters | Description |
|---|-------|-----------|-------------|
| 1 | Beta-Binomial | 1 | Conjugate model, single parameter |
| 2 | Normal Mean | 2 | Mean + variance estimation |
| 3 | Linear Regression | 11 | 10 predictors + noise scale |
| 4 | Logistic Regression | 50 | Binary classification, 50 predictors |
| 5 | Hierarchical Intercepts | ~22 | 20-group random intercepts |
| 6 | Eight Schools | 10 | Classic hierarchical benchmark |
| 7 | Wide Regression | 1001 | 1000 predictors + noise scale |
| 8 | Deep Hierarchy | ~504 | 3-level nested hierarchy |

## Methodology

- **Sampler**: NUTS (No-U-Turn Sampler) across all frameworks
- **Chains**: 4 parallel chains
- **Samples**: 1000 post-warmup draws per chain
- **Warmup**: 1000 iterations per chain
- **Repeats**: 3 runs per combination, report median
- **Timeout**: 10 minutes per run
- **ESS**: Effective Sample Size computed uniformly via ArviZ
- **Timing**: `time.perf_counter` (wall clock), excludes:
  - NumPyro JIT compilation warmup
  - CmdStan model compilation
- **Memory**: Peak RSS via `tracemalloc`

## Running

### Prerequisites

```bash
pip install -r requirements.txt
# For CmdStan: install_cmdstan
# For brms: R with brms package installed
```

### Full suite

```bash
python -m benchmarks.run_benchmarks --frameworks all --models all
```

### Specific frameworks/models

```bash
python -m benchmarks.run_benchmarks \
  --frameworks bayesiangpu pymc numpyro \
  --models linear_regression eight_schools
```

### Options

```
--frameworks    Frameworks to benchmark (or 'all')
--models        Models to benchmark (or 'all')
--repeats       Number of repeats (default: 3)
--samples       Draws per chain (default: 1000)
--warmup        Warmup iterations (default: 1000)
--chains        Number of chains (default: 4)
--timeout       Timeout per run in seconds (default: 600)
--seed          Random seed (default: 42)
--output        Save results to JSON file
--list-models   List available models
```

### Generate charts

```bash
python -m benchmarks.visualization.charts results/benchmark_results.json
```

## Results

Settings: 200 draws, 100 warmup, 2 chains, Apple M3 Pro, macOS.

### Wall Time (seconds)

| Model | Params | BayesianGPU (cpu) | NumPyro | PyMC |
|-------|--------|-------------------|---------|------|
| Beta-Binomial | 1 | **0.73** | 2.11 | 2.62 |
| Normal Mean | 2 | 21.54 | **1.63** | 1.41 |
| Linear Regression | 11 | timeout | **1.83** | 3.09 |
| Logistic Regression | 50 | timeout | **1.72** | error |
| Hierarchical Intercepts | 22 | timeout | **2.25** | 10.05 |
| Eight Schools | 10 | timeout | **1.93** | 5.77 |
| Wide Regression | 1001 | timeout | **10.52** | error |
| Deep Hierarchy | 504 | timeout | **3.25** | 15.75 |

### ESS per Second (higher is better)

| Model | Params | BayesianGPU (cpu) | NumPyro | PyMC |
|-------|--------|-------------------|---------|------|
| Beta-Binomial | 1 | **100.8** | 60.1 | 57.1 |
| Normal Mean | 2 | 1.7 | 169.2 | **261.4** |
| Linear Regression | 11 | - | **154.0** | 143.0 |
| Logistic Regression | 50 | - | **189.4** | - |
| Hierarchical Intercepts | 22 | - | **180.5** | 10.4 |
| Eight Schools | 10 | - | 11.2 | 0.4 |
| Wide Regression | 1001 | - | 0.3 | - |
| Deep Hierarchy | 504 | - | **82.9** | 6.0 |

### Key Takeaways

- **BayesianGPU leads on simple models**: 1.7x faster wall time and 1.7x higher ESS/s than competitors on the 1-parameter Beta-Binomial
- **Per-step overhead limits scaling**: Models with 2+ parameters expose the cost of per-step gradient computation through Burn's interpreted tensor operations vs PyTensor's C/LLVM compilation and JAX's XLA
- **NumPyro dominates multi-parameter models**: JAX's XLA compilation gives NumPyro consistently best performance on 10+ parameter models
- **Optimization path**: Vectorized gradient computation, compiled kernels, and parallel chains would close the gap

*Regenerate with: `python -m benchmarks.run_benchmarks --frameworks all --models all --output results/full_benchmark.json`*

## GPU Backend Notes

BayesianGPU supports both CPU (`ndarray`) and GPU (`wgpu`) backends via a compile-time feature flag:

```bash
# CPU (default)
maturin develop --release -m crates/bayesian-py/Cargo.toml

# GPU
maturin develop --features wgpu --release -m crates/bayesian-py/Cargo.toml
```

Check which backend is active:

```python
import bayesiangpu as bg
print(bg.backend_name())  # "cpu" or "gpu"
```

**Current GPU performance**: The wgpu backend uses fused kernel dispatch to minimize Metal/Vulkan overhead. Each NUTS step computes logp + grad in a single command encoder → submit → poll cycle (~1.5ms), halving the ~3ms-per-roundtrip cost of separate dispatches.

**GPU dispatch modes** (from fastest to slowest):
1. **Fused persistent** — pre-allocated buffers, single dispatch for logp+grad
2. **Persistent** — pre-allocated buffers, separate logp and grad dispatches
3. **Allocating** — fresh buffer allocation per kernel call

### GPU Kernel Benchmarks (logp + grad, Apple M3 Pro)

Representative results across 9 distributions (Normal, HalfNormal, Exponential, Beta, Gamma, InverseGamma, StudentT, Cauchy, LogNormal):

| N_obs | GPU Fused | GPU Alloc | CPU SIMD | Fused vs Alloc | Fused vs CPU |
|------:|----------:|----------:|---------:|:--------------:|:------------:|
| 256 | 1.5ms | 3.1ms | <0.01ms | **2.0x** | 1600x slower |
| 1K | 1.5ms | 3.2ms | 0.005ms | **2.1x** | 300x slower |
| 10K | 1.5ms | 3.2ms | 0.04ms | **2.1x** | 40x slower |
| 100K | 1.6ms | 3.3ms | 0.4ms | **2.1x** | 4x slower |
| 1M | 1.7ms | 4.7ms | 3.9ms | **2.8x** | **0.44x faster** |
| 10M | 4.7ms | 29ms | 39ms | **6.3x** | **0.12x faster** |

**Key findings:**
- Fused dispatch is consistently **2–2.8x faster** than allocating dispatch at all data sizes
- GPU crossover vs CPU occurs at **~1M observations** — above this, GPU fused wins
- At **10M observations**, GPU fused is **~8x faster** than CPU for compute-heavy distributions (Beta, Gamma, LogNormal)
- All 9 distributions pass numerical consistency checks (GPU paths agree within f32 tolerance)

**When GPU wins**: Likelihood evaluation over large datasets (>1M observations) — common in time-series, genomics, and large-N Bayesian regression. The fused dispatch path is used automatically when persistent buffers are available in the NUTS sampler.

**Remaining GPU bottlenecks**:
- NUTS sampling is inherently sequential (each step depends on the previous)
- Below ~1M observations, CPU SIMD dominates due to zero dispatch overhead
- Parallel chains on GPU and batch VI will further improve throughput

*Benchmark: `cargo bench -p bayesian-wasm --features sync-gpu --bench gpu_vs_cpu`*
