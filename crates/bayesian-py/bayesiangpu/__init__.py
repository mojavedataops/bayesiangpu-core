"""
BayesianGPU - GPU-accelerated Bayesian inference in Python

This library provides a complete solution for Bayesian inference in Python,
using a Rust core for high performance.

Example:
    >>> from bayesiangpu import Model, Beta, Binomial, sample
    >>>
    >>> # Define a simple Beta-Binomial model
    >>> model = Model()
    >>> model.param('theta', Beta(1, 1))      # Prior: uniform on [0,1]
    >>> model.observe(Binomial(100, 'theta'), [65])  # Likelihood: 65 successes in 100 trials
    >>>
    >>> # Run inference
    >>> result = sample(model, num_samples=1000, num_chains=4)
    >>>
    >>> # Analyze results
    >>> summary = result.summarize('theta')
    >>> print(f"Mean: {summary.mean:.3f}")
    >>> print(f"95% CI: [{summary.q025:.3f}, {summary.q975:.3f}]")
"""

from bayesiangpu._core import (
    # Model class
    Model,
    Distribution,
    # Result types
    InferenceResult,
    Diagnostics,
    ParameterSummary,
    # Distribution factories - priors
    normal as Normal,
    half_normal as HalfNormal,
    beta as Beta,
    gamma as Gamma,
    uniform as Uniform,
    exponential as Exponential,
    student_t as StudentT,
    half_cauchy as HalfCauchy,
    cauchy as Cauchy,
    log_normal as LogNormal,
    multivariate_normal as MultivariateNormal,
    dirichlet as Dirichlet,
    multinomial as Multinomial,
    # Distribution factories - likelihoods
    bernoulli as Bernoulli,
    binomial as Binomial,
    poisson as Poisson,
    # Sampling functions
    sample,
    quick_sample,
    # Diagnostic functions
    is_converged,
    summarize_parameter,
    # Version
    __version__,
)

__all__ = [
    # Model
    "Model",
    "Distribution",
    # Results
    "InferenceResult",
    "Diagnostics",
    "ParameterSummary",
    # Priors
    "Normal",
    "HalfNormal",
    "Beta",
    "Gamma",
    "Uniform",
    "Exponential",
    "StudentT",
    "HalfCauchy",
    "Cauchy",
    "LogNormal",
    "MultivariateNormal",
    "Dirichlet",
    "Multinomial",
    # Likelihoods
    "Bernoulli",
    "Binomial",
    "Poisson",
    # Functions
    "sample",
    "quick_sample",
    "is_converged",
    "summarize_parameter",
    # Metadata
    "__version__",
]
