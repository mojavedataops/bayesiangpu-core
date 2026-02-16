#!/usr/bin/env python
"""
Verify GPU is accessible and JAX/NumPyro work correctly.

This script is used by Docker HEALTHCHECK to verify container health.
Exit codes:
  0 - Healthy (GPU accessible, NumPyro working)
  1 - Unhealthy (GPU not found or NumPyro error)
"""
import sys


def main():
    try:
        import jax

        # Check for GPU devices
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']

        if not gpu_devices:
            print("ERROR: No GPU devices found")
            print(f"Available devices: {devices}")
            sys.exit(1)

        # Quick NumPyro smoke test
        import numpyro
        import numpyro.distributions as dist
        from jax import random

        key = random.PRNGKey(0)
        samples = dist.Normal(0, 1).sample(key, (100,))

        # Verify samples are valid
        if samples.shape != (100,):
            print(f"ERROR: Unexpected sample shape: {samples.shape}")
            sys.exit(1)

        print(f"OK: GPU={gpu_devices[0]}, NumPyro working")
        print(f"  JAX version: {jax.__version__}")
        print(f"  NumPyro version: {numpyro.__version__}")
        sys.exit(0)

    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
