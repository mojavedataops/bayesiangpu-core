#!/bin/bash
#
# Container entrypoint for BayesianGPU NumPyro image
#
# Responsibilities:
# 1. Verify GPU availability
# 2. Configure JAX memory settings
# 3. Execute user command
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Unknown GPU")
            log_info "GPU detected: $GPU_INFO"
            return 0
        else
            log_warn "nvidia-smi available but GPU not accessible"
            return 1
        fi
    else
        log_warn "nvidia-smi not found - GPU may not be available"
        return 1
    fi
}

# Configure JAX based on environment
configure_jax() {
    # Default to GPU if available, fall back to CPU
    if check_gpu; then
        export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-gpu}"
        log_info "JAX configured for GPU"
    else
        export JAX_PLATFORM_NAME="cpu"
        log_warn "JAX configured for CPU (no GPU detected)"
    fi

    # Memory settings
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"

    log_info "Memory preallocate: $XLA_PYTHON_CLIENT_PREALLOCATE"
    log_info "Memory fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION"
}

# Main entrypoint
main() {
    log_info "Starting BayesianGPU NumPyro container"

    # Configure environment
    configure_jax

    # Create workspace if it doesn't exist
    mkdir -p /workspace

    # Execute the provided command
    if [ $# -eq 0 ]; then
        log_info "No command provided, starting Python REPL"
        exec python
    else
        log_info "Executing: $@"
        exec "$@"
    fi
}

main "$@"
