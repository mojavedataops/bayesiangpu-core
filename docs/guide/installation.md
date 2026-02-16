# Installation

BayesianGPU can be used in multiple ways depending on your platform and use case.

## Rust Library

Add the crates to your `Cargo.toml`:

```toml
[dependencies]
# Core MCMC functionality
bayesian-sampler = { git = "https://github.com/mojavedataops/bayesiangpu-core" }

# Diagnostics (R-hat, ESS)
bayesian-diagnostics = { git = "https://github.com/mojavedataops/bayesiangpu-core" }

# Optional: distribution library
bayesian-core = { git = "https://github.com/mojavedataops/bayesiangpu-core" }

# Optional: GPU random numbers
bayesian-rng = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
```

### Feature Flags

```toml
# GPU acceleration (default)
bayesian-sampler = { git = "...", features = ["wgpu"] }

# CPU only (smaller binary)
bayesian-sampler = { git = "...", default-features = false }
```

### Minimum Rust Version

Rust 1.70 or later is required.

```bash
rustup update stable
rustc --version  # Should be >= 1.70
```

## Docker Images

Pre-built images are available for NumPyro and brms.

### NumPyro (Python/JAX)

```bash
# Latest stable
docker pull bayesiangpu/numpyro:latest

# Specific version
docker pull bayesiangpu/numpyro:0.1.0
```

**Image details:**
- Base: `nvidia/cuda:12.4.1-cudnn9-devel-ubuntu24.04`
- Python: 3.12
- Includes: NumPyro, JAX (GPU), ArviZ
- Size: ~14GB

### brms (R/Stan)

```bash
# Latest stable
docker pull bayesiangpu/brms:latest

# With CUDA support
docker pull bayesiangpu/brms:cuda-12.4
```

**Image details:**
- Base: `rocker/r-ver:4.4`
- R: 4.4
- Includes: brms, CmdStan 2.37
- Size: ~5GB

### Running with GPU

```bash
# NumPyro with NVIDIA GPU
docker run --gpus all bayesiangpu/numpyro python model.py

# brms with GPU (OpenCL)
docker run --gpus all bayesiangpu/brms:cuda-12.4 Rscript model.R
```

## JavaScript SDK

::: warning Coming in v0.2.0
Browser support is blocked on WASM compilation issues.
:::

When available:

```bash
npm install bayesiangpu
```

Or via CDN:

```html
<script type="module">
  import { Model, sample } from 'https://unpkg.com/bayesiangpu@0.2.0';
</script>
```

## From Source

### Prerequisites

- Rust 1.70+ with `wasm32-unknown-unknown` target
- Docker (for images)
- Node.js 18+ (for docs)

### Clone and Build

```bash
# Clone
git clone https://github.com/mojavedataops/bayesiangpu-core.git
cd bayesiangpu-core

# Build Rust library
cargo build --release

# Run tests
cargo test --workspace

# Build Docker images (optional)
docker build -f images/python/Dockerfile.numpyro -t bayesiangpu/numpyro images/python
docker build -f images/r/Dockerfile.brms -t bayesiangpu/brms images/r
```

### Development Setup

```bash
# Install Rust WASM target
rustup target add wasm32-unknown-unknown

# Install Node dependencies for docs
cd docs && npm install
```

## Verification

### Rust

```bash
cargo test --workspace
# Should show: 129 tests passed
```

### Docker

```bash
# NumPyro
docker run --rm bayesiangpu/numpyro python -c "import numpyro; print(numpyro.__version__)"

# brms
docker run --rm bayesiangpu/brms R -e "library(brms); packageVersion('brms')"
```

### GPU

```bash
# Check CUDA availability
docker run --gpus all --rm bayesiangpu/numpyro python -c \
    "import jax; print('GPU:', jax.devices())"
```

## Troubleshooting

### Rust: "can't find crate"

Make sure you're using the git dependency, not a crates.io version (not published yet):

```toml
# Wrong
bayesian-sampler = "0.1"

# Correct
bayesian-sampler = { git = "https://github.com/mojavedataops/bayesiangpu-core" }
```

### Docker: "could not select device driver"

Install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Docker: Out of memory

The NumPyro image is large (~14GB). Ensure Docker has enough disk space:

```bash
docker system prune -a  # Clean old images
docker info | grep "Total Memory"  # Check available
```

## Next Steps

- [Getting Started Guide](/guide/getting-started)
- [API Reference](/reference/rust-api)
- [Examples](/examples/linear-regression)
