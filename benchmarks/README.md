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

**Current GPU performance**: The wgpu backend is heavily optimized for likelihood evaluation throughput:
- **Single-pass fused shaders** compute logp + grad in one kernel (halving memory bandwidth)
- **Loop coarsening** (4 elements/thread) reduces workgroup count by 4x
- **Bind group caching** eliminates per-dispatch CPU allocation
- **Pre-computed normalization constants** (lgamma etc.) avoid redundant GPU math
- **Second-pass GPU reduction** for large partial sum arrays (N > 262K)
- **Multi-chain batched dispatch** runs all chains in a single submit+poll cycle
- **f64 accumulation** on CPU readback for precision at large N

**GPU dispatch modes** (from fastest to slowest):
1. **Single-pass fused** — one shader computes both logp+grad, pre-allocated buffers
2. **Two-pass fused** — separate logp and grad shaders, single submit+poll
3. **Persistent** — pre-allocated buffers, separate dispatches
4. **Allocating** — fresh buffer allocation per kernel call

### GPU Kernel Benchmarks (logp + grad, Apple M3 Pro)

Results across 9 distributions (Normal, HalfNormal, Exponential, Beta, Gamma, InverseGamma, StudentT, Cauchy, LogNormal). "GPU 1-pass" = single-pass fused with all optimizations.

| N_obs | GPU 1-pass | GPU Alloc | CPU SIMD | 1-pass vs Alloc | 1-pass vs CPU |
|------:|-----------:|----------:|---------:|:---------------:|:-------------:|
| 256 | 1.5ms | 3.0ms | <0.01ms | **2.0x** | CPU wins |
| 1K | 1.5ms | 3.1ms | 0.005ms | **2.1x** | CPU wins |
| 10K | 1.5ms | 3.2ms | 0.04ms | **2.1x** | CPU wins |
| 100K | 1.5ms | 3.4ms | 0.4ms | **2.3x** | CPU wins |
| 1M | 1.6ms | 4.7ms | 3.9ms | **2.9x** | **1.3x** |
| 10M | 2.5ms | 29ms | 42ms | **11.5x** | **16.8x** |

### Per-Distribution GPU Speedup at 10M Observations

| Distribution | GPU 1-pass | CPU | GPU vs CPU |
|---|---:|---:|:---:|
| Normal | 2.1ms | 23ms | **11.2x** |
| HalfNormal | 2.5ms | 20ms | **8.1x** |
| Exponential | 2.4ms | 20ms | **8.1x** |
| Beta | 2.4ms | 74ms | **30.8x** |
| Gamma | 2.5ms | 54ms | **21.6x** |
| InverseGamma | 2.8ms | 54ms | **19.7x** |
| StudentT | 2.5ms | 37ms | **14.9x** |
| Cauchy | 2.6ms | 36ms | **14.0x** |
| LogNormal | 2.8ms | 51ms | **18.1x** |

**Key findings:**
- Single-pass fused is **10–14x faster than allocating** dispatch at 10M observations
- GPU crossover vs CPU occurs at **~1M observations** — above this, GPU wins
- At **10M observations**, GPU is **8–31x faster** than CPU depending on distribution complexity
- Compute-heavy distributions (Beta, Gamma, InverseGamma) benefit most from GPU — lgamma hoisted to CPU eliminates redundant work
- All 9 distributions pass numerical consistency checks (GPU paths agree within f32 tolerance)

**When GPU wins**: Likelihood evaluation over large datasets (>1M observations) — common in time-series, genomics, and large-N Bayesian regression. The single-pass fused path is used automatically when persistent buffers are available in the NUTS sampler.

**Remaining GPU bottlenecks**:
- NUTS sampling is inherently sequential (each step depends on the previous)
- Below ~1M observations, CPU SIMD dominates due to zero dispatch overhead
- Multi-chain batched dispatch is available but requires sampler-level refactoring for full integration

*Benchmark: `cargo bench -p bayesian-wasm --features sync-gpu --bench gpu_vs_cpu`*
