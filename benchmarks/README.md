# BayesianGPU Cross-Framework Benchmarks

Standardized benchmarks comparing BayesianGPU against established probabilistic programming frameworks.

## Frameworks

| Framework | Language | Backend | Version |
|-----------|----------|---------|---------|
| **BayesianGPU** | Rust/Python | NdArray (CPU) | 0.2.x |
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

Results will be generated in `results/` after running benchmarks.

| Model | BayesianGPU | PyMC | NumPyro | CmdStan | brms |
|-------|------------|------|---------|---------|------|
| Beta-Binomial | - | - | - | - | - |
| Normal Mean | - | - | - | - | - |
| Linear Regression | - | - | - | - | - |
| Logistic Regression | - | - | - | - | - |
| Hierarchical Intercepts | - | - | - | - | - |
| Eight Schools | - | - | - | - | - |
| Wide Regression | - | - | - | - | - |
| Deep Hierarchy | - | - | - | - | - |

*Run `python -m benchmarks.run_benchmarks --frameworks all --models all --output results/results.json` to populate.*
