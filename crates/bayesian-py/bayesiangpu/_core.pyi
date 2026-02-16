"""Type stubs for bayesiangpu._core"""

from typing import Dict, List, Optional, Tuple, Union

__version__: str

class Distribution:
    """Distribution specification"""
    @property
    def dist_type(self) -> str: ...
    @property
    def params(self) -> Dict[str, Union[float, str]]: ...

class Model:
    """Fluent builder for Bayesian models"""
    def __init__(self) -> None: ...
    def param(self, name: str, distribution: Distribution) -> "Model":
        """Add a parameter with a prior distribution"""
        ...
    def observe(self, distribution: Distribution, data: List[float]) -> "Model":
        """Set the likelihood with observed data"""
        ...
    @property
    def param_names(self) -> List[str]: ...
    @property
    def num_params(self) -> int: ...
    @property
    def has_likelihood(self) -> bool: ...
    def to_json(self) -> str: ...

class Diagnostics:
    """MCMC diagnostics"""
    @property
    def rhat(self) -> Dict[str, float]: ...
    @property
    def ess(self) -> Dict[str, float]: ...
    @property
    def divergences(self) -> int: ...

class ParameterSummary:
    """Summary statistics for a single parameter"""
    @property
    def name(self) -> str: ...
    @property
    def mean(self) -> float: ...
    @property
    def std(self) -> float: ...
    @property
    def q025(self) -> float: ...
    @property
    def q25(self) -> float: ...
    @property
    def q50(self) -> float: ...
    @property
    def q75(self) -> float: ...
    @property
    def q975(self) -> float: ...
    @property
    def rhat(self) -> float: ...
    @property
    def ess(self) -> float: ...
    def ci95(self) -> Tuple[float, float]: ...

class InferenceResult:
    """Complete inference result"""
    def get_samples(self, param: str) -> List[float]:
        """Get samples for a parameter (flattened across chains)"""
        ...
    def get_chain_samples(self, param: str) -> List[List[float]]:
        """Get samples organized by chain"""
        ...
    @property
    def param_names(self) -> List[str]: ...
    @property
    def diagnostics(self) -> Diagnostics: ...
    @property
    def num_samples(self) -> int: ...
    @property
    def num_warmup(self) -> int: ...
    @property
    def num_chains(self) -> int: ...
    @property
    def step_size(self) -> float: ...
    def summarize(self, param: str) -> ParameterSummary:
        """Compute summary statistics for a parameter"""
        ...
    def summary(self) -> Dict[str, ParameterSummary]:
        """Compute summary statistics for all parameters"""
        ...
    def is_converged(self) -> bool:
        """Check if all R-hat values are below 1.01"""
        ...
    def has_sufficient_ess(self, min_ess: float = 400.0) -> bool:
        """Check if all ESS values are above the minimum"""
        ...
    def warnings(self) -> List[str]:
        """Get warning messages for diagnostic issues"""
        ...
    def format_summary(self) -> str:
        """Format results as a summary table"""
        ...

# Distribution factory functions

def normal(loc: Union[float, str], scale: Union[float, str]) -> Distribution:
    """Normal (Gaussian) distribution"""
    ...

def half_normal(scale: float) -> Distribution:
    """Half-Normal distribution (positive values only)"""
    ...

def beta(alpha: float, beta: float) -> Distribution:
    """Beta distribution on (0, 1)"""
    ...

def gamma(shape: float, rate: float) -> Distribution:
    """Gamma distribution on (0, infinity)"""
    ...

def uniform(low: float, high: float) -> Distribution:
    """Uniform distribution"""
    ...

def exponential(rate: float) -> Distribution:
    """Exponential distribution"""
    ...

def student_t(
    df: float,
    loc: Optional[Union[float, str]] = None,
    scale: Optional[Union[float, str]] = None,
) -> Distribution:
    """Student's t distribution"""
    ...

def cauchy(loc: float = 0.0, scale: float = 1.0) -> Distribution:
    """Cauchy distribution"""
    ...

def log_normal(loc: Union[float, str], scale: Union[float, str]) -> Distribution:
    """Log-Normal distribution"""
    ...

def bernoulli(p: Union[float, str]) -> Distribution:
    """Bernoulli distribution for binary outcomes"""
    ...

def binomial(n: int, p: Union[float, str]) -> Distribution:
    """Binomial distribution for count of successes"""
    ...

def poisson(rate: Union[float, str]) -> Distribution:
    """Poisson distribution for count data"""
    ...

# Sampling functions

def sample(
    model: Model,
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    target_accept: float = 0.8,
    seed: int = 42,
) -> InferenceResult:
    """Run NUTS sampling on a model"""
    ...

def quick_sample(model: Model, seed: int = 42) -> InferenceResult:
    """Quick sampling with fewer iterations (good for testing)"""
    ...

# Diagnostic functions

def is_converged(result: InferenceResult) -> bool:
    """Check if inference result indicates good convergence"""
    ...

def summarize_parameter(result: InferenceResult, param: str) -> ParameterSummary:
    """Compute summary statistics for a parameter"""
    ...
