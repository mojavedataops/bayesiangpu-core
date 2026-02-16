#!/usr/bin/env python3
"""Benchmark scipy distribution computations for comparison with GPU kernels."""

import time
import numpy as np
from scipy.stats import norm, halfnorm, expon, beta, gamma, t as studentt


def benchmark_single(func, args, iterations=10000):
    """Benchmark a single-value computation. Returns time in milliseconds."""
    # Warmup
    for _ in range(100):
        func(*args)

    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # milliseconds per call


def benchmark_batch(func, args, batch_size=1000, iterations=100):
    """Benchmark batched computation. Returns time in milliseconds per element."""
    # Create batched input (first arg is x, repeat it)
    x_batch = np.full(batch_size, args[0])
    batch_args = (x_batch,) + args[1:]

    # Warmup
    for _ in range(10):
        func(*batch_args)

    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        func(*batch_args)
    elapsed = time.perf_counter() - start

    total_ops = batch_size * iterations
    return elapsed / total_ops * 1000  # milliseconds per element


def main():
    print("=" * 60)
    print("SciPy Distribution Benchmark")
    print("=" * 60)
    print()

    # Test parameters
    test_cases = {
        "Normal": (lambda x, mu, sigma: norm.logpdf(x, loc=mu, scale=sigma), (1.5, 0.0, 1.0)),
        "HalfNormal": (lambda x, sigma: halfnorm.logpdf(x, scale=sigma), (1.5, 1.0)),
        "Exponential": (lambda x, lam: expon.logpdf(x, scale=1.0/lam), (1.0, 1.0)),
        "Beta": (lambda x, a, b: beta.logpdf(x, a, b), (0.5, 2.0, 2.0)),
        "Gamma": (lambda x, a, b: gamma.logpdf(x, a, scale=1.0/b), (1.0, 2.0, 1.0)),
        "StudentT": (lambda x, mu, sigma, nu: studentt.logpdf(x, nu, loc=mu, scale=sigma), (1.5, 0.0, 1.0, 3.0)),
    }

    print("Single-value performance (10,000 iterations):")
    print("-" * 60)
    print(f"{'Distribution':<15} {'Time (ms)':<15} {'Ops/sec':<15}")
    print("-" * 60)

    single_results = {}
    for name, (func, args) in test_cases.items():
        ms_per_call = benchmark_single(func, args)
        ops_per_sec = 1000 / ms_per_call
        single_results[name] = ms_per_call
        print(f"{name:<15} {ms_per_call:<15.5f} {ops_per_sec:<15,.0f}")

    print()
    print("Batched performance (1000 elements × 100 iterations):")
    print("-" * 60)
    print(f"{'Distribution':<15} {'Time/elem (ms)':<18} {'Throughput':<15}")
    print("-" * 60)

    batch_results = {}
    for name, (func, args) in test_cases.items():
        ms_per_elem = benchmark_batch(func, args)
        throughput = 1000 / ms_per_elem
        batch_results[name] = ms_per_elem
        print(f"{name:<15} {ms_per_elem:<18.8f} {throughput:<15,.0f} elem/s")

    print()
    print("=" * 60)
    print("Summary for GPU comparison (all in ms):")
    print("=" * 60)
    print()
    print("Copy these values to compare with browser GPU timings:")
    print()
    for name in test_cases:
        print(f"  {name}:")
        print(f"    scipy single:  {single_results[name]:.5f} ms")
        print(f"    scipy batched: {batch_results[name]:.8f} ms/elem")

    print()
    print("Note: GPU has ~2ms initialization overhead but should be faster")
    print("for batched operations due to parallel execution.")


if __name__ == "__main__":
    main()
