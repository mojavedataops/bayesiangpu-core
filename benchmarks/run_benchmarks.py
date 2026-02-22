#!/usr/bin/env python3
"""Cross-framework Bayesian inference benchmarks.

Usage:
    python run_benchmarks.py --frameworks bayesiangpu pymc --models all
    python run_benchmarks.py --frameworks all --models linear_regression eight_schools
    python run_benchmarks.py --list-models
"""
import argparse
import sys
from pathlib import Path


def get_adapter(name: str):
    """Get framework adapter by name."""
    if name == "bayesiangpu":
        from .adapters.bayesiangpu_adapter import BayesianGPUAdapter
        return BayesianGPUAdapter()
    elif name == "pymc":
        from .adapters.pymc_adapter import PyMCAdapter
        return PyMCAdapter()
    elif name == "numpyro":
        from .adapters.numpyro_adapter import NumPyroAdapter
        return NumPyroAdapter()
    elif name == "cmdstan":
        from .adapters.cmdstan_adapter import CmdStanAdapter
        return CmdStanAdapter()
    elif name == "brms":
        from .adapters.brms_adapter import BRMSAdapter
        return BRMSAdapter()
    else:
        raise ValueError(f"Unknown framework: {name}")


ALL_FRAMEWORKS = ["bayesiangpu", "pymc", "numpyro", "cmdstan", "brms"]


def main():
    parser = argparse.ArgumentParser(description="Cross-framework Bayesian inference benchmarks")
    parser.add_argument("--frameworks", nargs="+", default=["bayesiangpu"],
                        help="Frameworks to benchmark (or 'all')")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Models to benchmark (or 'all')")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--chains", type=int, default=4, help="Number of chains")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per run (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()

    # Import models to populate registry
    from .models import simple, medium, large  # noqa: F401
    from .models.registry import MODEL_REGISTRY

    if args.list_models:
        for name, model in MODEL_REGISTRY.items():
            print(f"  {name}: {model.description} ({model.num_params} params)")
        return

    # Resolve frameworks
    fw_names = ALL_FRAMEWORKS if "all" in args.frameworks else args.frameworks
    adapters = []
    for name in fw_names:
        try:
            adapters.append(get_adapter(name))
        except ImportError as e:
            print(f"Warning: {name} not available ({e})")

    if not adapters:
        print("No frameworks available. Install at least one.")
        sys.exit(1)

    # Resolve models
    model_names = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models

    print(f"Benchmarking {len(adapters)} frameworks x {len(model_names)} models x {args.repeats} repeats\n")

    from .runners.harness import run_benchmarks
    results = run_benchmarks(
        adapters, model_names,
        num_repeats=args.repeats,
        num_samples=args.samples,
        num_warmup=args.warmup,
        num_chains=args.chains,
        seed=args.seed,
        timeout=args.timeout,
    )

    from .runners.reporter import to_markdown_table, to_json
    print("\n" + to_markdown_table(results))

    if args.output:
        to_json(results, args.output)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
