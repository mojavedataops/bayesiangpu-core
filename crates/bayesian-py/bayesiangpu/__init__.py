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

Linear regression example:
    >>> import numpy as np
    >>> from bayesiangpu import Model, Normal, HalfCauchy, LinearPredictor, sample
    >>>
    >>> X = np.random.randn(200, 3)
    >>> y = X @ [1.5, -0.8, 0.3] + np.random.randn(200) * 0.5
    >>>
    >>> model = Model()
    >>> model.param("beta", Normal(0, 10), size=3)
    >>> model.param("sigma", HalfCauchy(5))
    >>> model.observe(Normal(LinearPredictor(X, "beta"), "sigma"), y.tolist())
    >>>
    >>> result = sample(model, num_samples=2000)
"""

import numpy as np

from bayesiangpu._core import (
    # Model class (saved as _RustModel below)
    Model as _RustModel,
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
    lkj_corr as LKJCorr,
    # Distribution factories - likelihoods
    bernoulli as Bernoulli,
    binomial as Binomial,
    poisson as Poisson,
    # Sampling functions (saved with _rust_ prefix below)
    sample as _rust_sample,
    quick_sample as _rust_quick_sample,
    fit as _rust_fit,
    # Diagnostic functions
    is_converged,
    summarize_parameter,
    # ADVI result
    PyAdviResult as AdviResult,
    # Backend info
    backend_name,
    # Version
    __version__,
)

# ---------------------------------------------------------------------------
# LinearPredictor helper
# ---------------------------------------------------------------------------

# Module-level store for pending design matrices. LinearPredictor() puts data
# here and Model.observe() drains it into the `known` dict that Rust receives.
_PENDING_MATRICES: dict[str, list[float]] = {}


def LinearPredictor(X, param_name):
    """Create a linear predictor ``X @ param`` for use in observe().

    Args:
        X: Design matrix (n x p array-like).
        param_name: Name of the coefficient parameter.

    Returns:
        A dict that the Rust side parses as ``ParamValue::LinearPredictor``.

    Example:
        >>> X = np.random.randn(200, 10)
        >>> model.observe(Normal(LinearPredictor(X, "beta"), "sigma"), y)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    n, p = X.shape
    key = f"__X_{param_name}"
    _PENDING_MATRICES[key] = X.flatten().tolist()
    return {
        "__type": "LinearPredictor",
        "matrix_key": key,
        "param_name": param_name,
        "num_cols": p,
    }


# ---------------------------------------------------------------------------
# Python Model wrapper (composition over the Rust PyModel)
# ---------------------------------------------------------------------------

class Model:
    """Extended Model with LinearPredictor support.

    Wraps the Rust ``PyModel`` and intercepts ``observe()`` to inject any
    pending design-matrix data into the ``known`` dictionary.
    """

    def __init__(self):
        self._inner = _RustModel()

    def param(self, name, distribution, size=1):
        """Add a parameter with a prior distribution."""
        self._inner.param(name, distribution, size)
        return self

    def observe(self, distribution, data, known=None):
        """Observe data, auto-extracting LinearPredictor matrices into *known*."""
        global _PENDING_MATRICES
        if known is None:
            known = {}
        # Merge any pending LinearPredictor matrices
        known.update(_PENDING_MATRICES)
        _PENDING_MATRICES = {}
        self._inner.observe(distribution, data, known)
        return self

    # -- properties forwarded from the Rust model --

    @property
    def param_names(self):
        return self._inner.param_names

    @property
    def num_params(self):
        return self._inner.num_params

    @property
    def has_likelihood(self):
        return self._inner.has_likelihood

    def to_json(self):
        """Serialize model spec to JSON."""
        return self._inner.to_json()

    def spec(self):
        """Return the inner Rust model (for passing to sample/fit)."""
        return self._inner

    def __repr__(self):
        return repr(self._inner)


# ---------------------------------------------------------------------------
# Wrapped sampling / fitting functions
# ---------------------------------------------------------------------------

def _unwrap(model):
    """Return the underlying Rust PyModel regardless of wrapper."""
    return model._inner if hasattr(model, "_inner") else model


def sample(model, **kwargs):
    """Run MCMC sampling (NUTS).  Accepts both wrapped and raw models."""
    return _rust_sample(_unwrap(model), **kwargs)


def quick_sample(model, **kwargs):
    """Quick MCMC run with reduced defaults.  Accepts both wrapped and raw models."""
    return _rust_quick_sample(_unwrap(model), **kwargs)


def fit(model, **kwargs):
    """Run variational inference (ADVI).  Accepts both wrapped and raw models."""
    return _rust_fit(_unwrap(model), **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Model
    "Model",
    "Distribution",
    # Results
    "InferenceResult",
    "AdviResult",
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
    "LKJCorr",
    # Likelihoods
    "Bernoulli",
    "Binomial",
    "Poisson",
    # Functions
    "sample",
    "quick_sample",
    "fit",
    "is_converged",
    "summarize_parameter",
    # LinearPredictor
    "LinearPredictor",
    # Backend info
    "backend_name",
    # Metadata
    "__version__",
]
