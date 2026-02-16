#!/usr/bin/env python3
"""Generate reference test vectors from scipy for GPU kernel validation."""

import json
from scipy.stats import norm, halfnorm, expon, beta, gamma, t as studentt
import numpy as np


def normal_reference(x: float, mu: float, sigma: float) -> dict:
    """Normal distribution log_prob and gradient."""
    log_prob = norm.logpdf(x, loc=mu, scale=sigma)
    # grad = -(x - mu) / sigma^2
    grad = -(x - mu) / (sigma ** 2)
    return {"x": x, "mu": mu, "sigma": sigma, "log_prob": float(log_prob), "grad": float(grad)}


def half_normal_reference(x: float, sigma: float) -> dict:
    """HalfNormal distribution log_prob and gradient (x >= 0)."""
    log_prob = halfnorm.logpdf(x, scale=sigma)
    # grad = -x / sigma^2
    grad = -x / (sigma ** 2)
    return {"x": x, "sigma": sigma, "log_prob": float(log_prob), "grad": float(grad)}


def exponential_reference(x: float, lam: float) -> dict:
    """Exponential distribution log_prob and gradient (x >= 0)."""
    # scipy uses scale = 1/lambda
    log_prob = expon.logpdf(x, scale=1.0/lam)
    # grad = -lambda (constant)
    grad = -lam
    return {"x": x, "lambda": lam, "log_prob": float(log_prob), "grad": float(grad)}


def beta_reference(x: float, alpha: float, beta_param: float) -> dict:
    """Beta distribution log_prob and gradient (0 < x < 1)."""
    log_prob = beta.logpdf(x, alpha, beta_param)
    # grad = (alpha - 1) / x - (beta - 1) / (1 - x)
    grad = (alpha - 1) / x - (beta_param - 1) / (1 - x)
    return {"x": x, "alpha": alpha, "beta": beta_param, "log_prob": float(log_prob), "grad": float(grad)}


def gamma_reference(x: float, alpha: float, beta_param: float) -> dict:
    """Gamma distribution log_prob and gradient (x > 0)."""
    # scipy uses shape=alpha, scale=1/beta
    log_prob = gamma.logpdf(x, alpha, scale=1.0/beta_param)
    # grad = (alpha - 1) / x - beta
    grad = (alpha - 1) / x - beta_param
    return {"x": x, "alpha": alpha, "beta": beta_param, "log_prob": float(log_prob), "grad": float(grad)}


def student_t_reference(x: float, mu: float, sigma: float, nu: float) -> dict:
    """Student's t distribution log_prob and gradient."""
    # scipy t distribution: loc=mu, scale=sigma, df=nu
    log_prob = studentt.logpdf(x, nu, loc=mu, scale=sigma)
    # grad = -(nu + 1) * z / (sigma * (nu + z^2)) where z = (x - mu) / sigma
    z = (x - mu) / sigma
    grad = -(nu + 1) * z / (sigma * (nu + z * z))
    return {"x": x, "mu": mu, "sigma": sigma, "nu": nu, "log_prob": float(log_prob), "grad": float(grad)}


def generate_test_vectors():
    """Generate test vectors for all distributions."""
    vectors = {
        "normal": [
            normal_reference(1.5, 0.0, 1.0),
            normal_reference(0.0, 0.0, 1.0),
            normal_reference(-2.0, 1.0, 2.0),
            normal_reference(3.5, -1.0, 0.5),
        ],
        "half_normal": [
            half_normal_reference(1.5, 1.0),
            half_normal_reference(0.5, 1.0),
            half_normal_reference(2.0, 0.5),
            half_normal_reference(0.1, 2.0),
        ],
        "exponential": [
            exponential_reference(1.0, 1.0),
            exponential_reference(0.5, 2.0),
            exponential_reference(2.0, 0.5),
            exponential_reference(0.1, 3.0),
        ],
        "beta": [
            beta_reference(0.5, 2.0, 2.0),
            beta_reference(0.3, 1.0, 3.0),
            beta_reference(0.7, 5.0, 2.0),
            beta_reference(0.1, 0.5, 0.5),
        ],
        "gamma": [
            gamma_reference(1.0, 2.0, 1.0),
            gamma_reference(0.5, 1.0, 2.0),
            gamma_reference(2.0, 3.0, 0.5),
            gamma_reference(0.1, 0.5, 1.0),
        ],
        "student_t": [
            student_t_reference(1.5, 0.0, 1.0, 3.0),
            student_t_reference(0.0, 0.0, 1.0, 5.0),
            student_t_reference(-1.0, 1.0, 2.0, 10.0),
            student_t_reference(2.0, -0.5, 0.5, 2.0),
        ],
    }
    return vectors


if __name__ == "__main__":
    vectors = generate_test_vectors()

    # Print as JSON for embedding in tests
    print("// Reference test vectors generated from scipy")
    print("// Run: python3 generate_reference.py")
    print("const REFERENCE_VECTORS = " + json.dumps(vectors, indent=2) + ";")
    print()

    # Also print human-readable format
    print("\n// Human-readable reference values:")
    for dist, cases in vectors.items():
        print(f"\n// {dist.upper()}")
        for case in cases:
            params = ", ".join(f"{k}={v}" for k, v in case.items() if k not in ["log_prob", "grad"])
            print(f"//   {params}")
            print(f"//     log_prob = {case['log_prob']:.10f}")
            print(f"//     grad     = {case['grad']:.10f}")
