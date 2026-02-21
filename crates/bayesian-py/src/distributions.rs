//! Distribution types and factory functions for Python bindings

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A parameter value that can be either a number or a reference to another parameter
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Number(f64),
    Reference(String),
}

impl ParamValue {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(n) = obj.extract::<f64>() {
            Ok(ParamValue::Number(n))
        } else if let Ok(s) = obj.extract::<String>() {
            Ok(ParamValue::Reference(s))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Parameter must be a number or string reference",
            ))
        }
    }
}

#[allow(deprecated)]
impl IntoPy<PyObject> for ParamValue {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            ParamValue::Number(n) => n.into_py(py),
            ParamValue::Reference(s) => s.into_py(py),
        }
    }
}

/// Distribution specification
#[pyclass(name = "Distribution")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyDistribution {
    #[pyo3(get)]
    pub dist_type: String,
    pub params: HashMap<String, ParamValue>,
}

impl PyDistribution {
    /// Get a string representation (callable from Rust)
    pub fn repr_str(&self) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|(k, v)| {
                let val_str = match v {
                    ParamValue::Number(n) => format!("{}", n),
                    ParamValue::Reference(s) => format!("'{}'", s),
                };
                format!("{}={}", k, val_str)
            })
            .collect();
        format!("{}({})", self.dist_type, params.join(", "))
    }
}

#[pymethods]
impl PyDistribution {
    fn __repr__(&self) -> String {
        self.repr_str()
    }

    /// Get parameters as a dictionary
    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in &self.params {
            #[allow(deprecated)]
            dict.set_item(k, v.clone().into_py(py))?;
        }
        Ok(dict.into())
    }
}

// ============================================================================
// Distribution Factory Functions
// ============================================================================

/// Normal (Gaussian) distribution
///
/// Args:
///     loc: Mean (can be a number or parameter name)
///     scale: Standard deviation (can be a number or parameter name)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Normal(0, 1)           # Standard normal
///     >>> Normal("mu", "sigma")  # Parameters from model
#[pyfunction]
#[pyo3(signature = (loc, scale))]
pub fn normal(loc: &Bound<'_, PyAny>, scale: &Bound<'_, PyAny>) -> PyResult<PyDistribution> {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::from_py(loc)?);
    params.insert("scale".to_string(), ParamValue::from_py(scale)?);
    Ok(PyDistribution {
        dist_type: "Normal".to_string(),
        params,
    })
}

/// Half-Normal distribution (positive values only)
///
/// Useful for scale parameters.
///
/// Args:
///     scale: Scale parameter
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> HalfNormal(1)  # Half-normal with scale 1
#[pyfunction]
pub fn half_normal(scale: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("scale".to_string(), ParamValue::Number(scale));
    PyDistribution {
        dist_type: "HalfNormal".to_string(),
        params,
    }
}

/// Beta distribution
///
/// Defined on (0, 1), useful for probability parameters.
///
/// Args:
///     alpha: Shape parameter alpha (concentration1)
///     beta: Shape parameter beta (concentration0)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Beta(1, 1)   # Uniform on [0, 1]
///     >>> Beta(2, 5)   # Skewed toward 0
#[pyfunction]
pub fn beta(alpha: f64, beta: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("alpha".to_string(), ParamValue::Number(alpha));
    params.insert("beta".to_string(), ParamValue::Number(beta));
    PyDistribution {
        dist_type: "Beta".to_string(),
        params,
    }
}

/// Gamma distribution
///
/// Defined on (0, infinity), useful for positive parameters.
///
/// Args:
///     shape: Shape parameter (k)
///     rate: Rate parameter (1/scale)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Gamma(2, 1)  # Shape=2, rate=1
#[pyfunction]
pub fn gamma(shape: f64, rate: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("shape".to_string(), ParamValue::Number(shape));
    params.insert("rate".to_string(), ParamValue::Number(rate));
    PyDistribution {
        dist_type: "Gamma".to_string(),
        params,
    }
}

/// Uniform distribution
///
/// Args:
///     low: Lower bound
///     high: Upper bound
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Uniform(0, 10)  # Uniform on [0, 10]
#[pyfunction]
pub fn uniform(low: f64, high: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("low".to_string(), ParamValue::Number(low));
    params.insert("high".to_string(), ParamValue::Number(high));
    PyDistribution {
        dist_type: "Uniform".to_string(),
        params,
    }
}

/// Exponential distribution
///
/// For waiting times and durations.
///
/// Args:
///     rate: Rate parameter (inverse of mean)
///
/// Returns:
///     Distribution specification
#[pyfunction]
pub fn exponential(rate: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("rate".to_string(), ParamValue::Number(rate));
    PyDistribution {
        dist_type: "Exponential".to_string(),
        params,
    }
}

/// Student's t distribution
///
/// For robust inference with potential outliers.
///
/// Args:
///     df: Degrees of freedom
///     loc: Location parameter (default: 0)
///     scale: Scale parameter (default: 1)
///
/// Returns:
///     Distribution specification
#[pyfunction]
#[pyo3(signature = (df, loc=None, scale=None))]
pub fn student_t(
    df: f64,
    loc: Option<&Bound<'_, PyAny>>,
    scale: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyDistribution> {
    let mut params = HashMap::new();
    params.insert("df".to_string(), ParamValue::Number(df));

    if let Some(l) = loc {
        params.insert("loc".to_string(), ParamValue::from_py(l)?);
    } else {
        params.insert("loc".to_string(), ParamValue::Number(0.0));
    }

    if let Some(s) = scale {
        params.insert("scale".to_string(), ParamValue::from_py(s)?);
    } else {
        params.insert("scale".to_string(), ParamValue::Number(1.0));
    }

    Ok(PyDistribution {
        dist_type: "StudentT".to_string(),
        params,
    })
}

/// Half-Cauchy distribution (positive values only)
///
/// Heavy-tailed prior for scale parameters, widely recommended
/// for hierarchical models.
///
/// Args:
///     scale: Scale parameter (default: 1)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> HalfCauchy(5)  # Half-Cauchy with scale 5
#[pyfunction]
#[pyo3(signature = (scale=1.0))]
pub fn half_cauchy(scale: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("scale".to_string(), ParamValue::Number(scale));
    PyDistribution {
        dist_type: "HalfCauchy".to_string(),
        params,
    }
}

/// Cauchy distribution
///
/// Heavy-tailed distribution useful for weakly informative priors.
///
/// Args:
///     loc: Location parameter (default: 0)
///     scale: Scale parameter (default: 1)
///
/// Returns:
///     Distribution specification
#[pyfunction]
#[pyo3(signature = (loc=0.0, scale=1.0))]
pub fn cauchy(loc: f64, scale: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));
    PyDistribution {
        dist_type: "Cauchy".to_string(),
        params,
    }
}

/// Laplace (double exponential) distribution
///
/// Heavier tails than Normal, useful for sparsity-inducing priors.
///
/// Args:
///     loc: Location parameter (mean)
///     scale: Scale parameter (must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Laplace(0, 1)  # Standard Laplace
#[pyfunction]
pub fn laplace(loc: f64, scale: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));
    PyDistribution {
        dist_type: "Laplace".to_string(),
        params,
    }
}

/// Logistic distribution
///
/// Similar shape to Normal but with heavier tails.
///
/// Args:
///     loc: Location parameter (mean)
///     scale: Scale parameter (must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Logistic(0, 1)  # Standard Logistic
#[pyfunction]
pub fn logistic(loc: f64, scale: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));
    PyDistribution {
        dist_type: "Logistic".to_string(),
        params,
    }
}

/// Inverse Gamma distribution
///
/// Commonly used as a prior for variance parameters.
///
/// Args:
///     alpha: Shape parameter (must be positive)
///     beta: Scale parameter (must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> InverseGamma(2, 1)  # Shape=2, scale=1
#[pyfunction]
pub fn inverse_gamma(alpha: f64, beta: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("alpha".to_string(), ParamValue::Number(alpha));
    params.insert("beta".to_string(), ParamValue::Number(beta));
    PyDistribution {
        dist_type: "InverseGamma".to_string(),
        params,
    }
}

/// Chi-Squared distribution
///
/// Special case of Gamma distribution, used in hypothesis testing.
///
/// Args:
///     df: Degrees of freedom (must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> ChiSquared(5)  # 5 degrees of freedom
#[pyfunction]
pub fn chi_squared(df: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("df".to_string(), ParamValue::Number(df));
    PyDistribution {
        dist_type: "ChiSquared".to_string(),
        params,
    }
}

/// Truncated Normal distribution
///
/// Normal distribution constrained to [low, high].
///
/// Args:
///     loc: Location parameter (mean of underlying normal)
///     scale: Scale parameter (std dev, must be positive)
///     low: Lower bound
///     high: Upper bound
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> TruncatedNormal(0, 1, -2, 2)  # Normal(0,1) truncated to [-2, 2]
#[pyfunction]
pub fn truncated_normal(loc: f64, scale: f64, low: f64, high: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));
    params.insert("low".to_string(), ParamValue::Number(low));
    params.insert("high".to_string(), ParamValue::Number(high));
    PyDistribution {
        dist_type: "TruncatedNormal".to_string(),
        params,
    }
}

/// Log-Normal distribution
///
/// For positive values with multiplicative effects.
///
/// Args:
///     loc: Mean of the log (mu)
///     scale: Standard deviation of the log (sigma)
///
/// Returns:
///     Distribution specification
#[pyfunction]
#[pyo3(signature = (loc, scale))]
pub fn log_normal(loc: &Bound<'_, PyAny>, scale: &Bound<'_, PyAny>) -> PyResult<PyDistribution> {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::from_py(loc)?);
    params.insert("scale".to_string(), ParamValue::from_py(scale)?);
    Ok(PyDistribution {
        dist_type: "LogNormal".to_string(),
        params,
    })
}

// ============================================================================
// Likelihood Distributions
// ============================================================================

/// Bernoulli distribution
///
/// For binary outcomes (0 or 1).
///
/// Args:
///     p: Probability of success (can be a number or parameter name)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Bernoulli(0.5)       # 50% success rate
///     >>> Bernoulli("theta")   # Parameter from model
#[pyfunction]
pub fn bernoulli(p: &Bound<'_, PyAny>) -> PyResult<PyDistribution> {
    let mut params = HashMap::new();
    params.insert("p".to_string(), ParamValue::from_py(p)?);
    Ok(PyDistribution {
        dist_type: "Bernoulli".to_string(),
        params,
    })
}

/// Binomial distribution
///
/// For count of successes in n trials.
///
/// Args:
///     n: Number of trials
///     p: Probability of success (can be a number or parameter name)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Binomial(100, 0.5)      # 100 trials, 50% success
///     >>> Binomial(100, "theta")  # Parameter from model
#[pyfunction]
pub fn binomial(n: usize, p: &Bound<'_, PyAny>) -> PyResult<PyDistribution> {
    let mut params = HashMap::new();
    params.insert("n".to_string(), ParamValue::Number(n as f64));
    params.insert("p".to_string(), ParamValue::from_py(p)?);
    Ok(PyDistribution {
        dist_type: "Binomial".to_string(),
        params,
    })
}

/// Poisson distribution
///
/// For count data.
///
/// Args:
///     rate: Rate parameter (expected count)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Poisson(5)          # Rate of 5
///     >>> Poisson("lambda")   # Parameter from model
#[pyfunction]
pub fn poisson(rate: &Bound<'_, PyAny>) -> PyResult<PyDistribution> {
    let mut params = HashMap::new();
    params.insert("rate".to_string(), ParamValue::from_py(rate)?);
    Ok(PyDistribution {
        dist_type: "Poisson".to_string(),
        params,
    })
}

/// Multivariate Normal distribution
///
/// For multivariate continuous data with correlation structure.
///
/// Args:
///     mu: Mean vector (list of floats)
///     cov: Covariance matrix (list of lists) OR
///     scale_tril: Lower triangular Cholesky factor (list of lists)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> MultivariateNormal([0, 0], cov=[[1, 0.5], [0.5, 1]])
///     >>> MultivariateNormal([0, 0], scale_tril=[[1, 0], [0.5, 0.866]])
#[pyfunction]
#[pyo3(signature = (mu, cov=None, scale_tril=None))]
pub fn multivariate_normal(
    mu: Vec<f64>,
    cov: Option<Vec<Vec<f64>>>,
    scale_tril: Option<Vec<Vec<f64>>>,
) -> PyResult<PyDistribution> {
    let mut params = HashMap::new();

    // Store mu as JSON array
    let mu_json = serde_json::to_string(&mu).unwrap();
    params.insert("mu".to_string(), ParamValue::Number(0.0)); // placeholder
    params.insert("mu_json".to_string(), ParamValue::Reference(mu_json));

    if let Some(cov_mat) = cov {
        let cov_json = serde_json::to_string(&cov_mat).unwrap();
        params.insert("cov_json".to_string(), ParamValue::Reference(cov_json));
        params.insert(
            "parameterization".to_string(),
            ParamValue::Reference("covariance".to_string()),
        );
    } else if let Some(tril) = scale_tril {
        let tril_json = serde_json::to_string(&tril).unwrap();
        params.insert(
            "scale_tril_json".to_string(),
            ParamValue::Reference(tril_json),
        );
        params.insert(
            "parameterization".to_string(),
            ParamValue::Reference("cholesky".to_string()),
        );
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must provide either cov or scale_tril",
        ));
    }

    Ok(PyDistribution {
        dist_type: "MultivariateNormal".to_string(),
        params,
    })
}

/// Dirichlet distribution
///
/// Prior over probability simplexes (vectors that sum to 1).
/// Essential for topic modeling (LDA) and categorical data modeling.
///
/// Args:
///     alpha: Concentration vector (list of positive floats)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Dirichlet([1, 1, 1])       # Uniform over 3-simplex
///     >>> Dirichlet([10, 10, 10])    # Concentrated at center
///     >>> Dirichlet([0.1, 0.1, 0.1]) # Concentrated at corners
#[pyfunction]
pub fn dirichlet(alpha: Vec<f64>) -> PyResult<PyDistribution> {
    if alpha.len() < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Dirichlet requires at least 2 categories",
        ));
    }

    let mut params = HashMap::new();

    // Store alpha as JSON array
    let alpha_json = serde_json::to_string(&alpha).unwrap();
    params.insert("alpha_json".to_string(), ParamValue::Reference(alpha_json));
    params.insert("dim".to_string(), ParamValue::Number(alpha.len() as f64));

    Ok(PyDistribution {
        dist_type: "Dirichlet".to_string(),
        params,
    })
}

/// Multinomial distribution
///
/// Generalization of binomial to K categories. Models the number of
/// occurrences of K outcomes in n independent trials.
///
/// Args:
///     n: Number of trials (positive integer)
///     probs: Probability vector (list of non-negative floats summing to 1,
///            or can be a parameter reference for hierarchical models)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Multinomial(10, [0.2, 0.3, 0.5])   # 10 trials with fixed probs
///     >>> Multinomial(100, "theta")          # Parameter from model (e.g., Dirichlet)
#[pyfunction]
pub fn multinomial(n: usize, probs: &Bound<'_, PyAny>) -> PyResult<PyDistribution> {
    let mut params = HashMap::new();
    params.insert("n".to_string(), ParamValue::Number(n as f64));

    // Check if probs is a list/array or a string reference
    if let Ok(probs_vec) = probs.extract::<Vec<f64>>() {
        if probs_vec.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Multinomial requires at least 2 categories",
            ));
        }
        let probs_json = serde_json::to_string(&probs_vec).unwrap();
        params.insert("probs_json".to_string(), ParamValue::Reference(probs_json));
        params.insert(
            "dim".to_string(),
            ParamValue::Number(probs_vec.len() as f64),
        );
    } else if let Ok(ref_name) = probs.extract::<String>() {
        // Reference to another parameter (e.g., Dirichlet prior)
        params.insert("probs".to_string(), ParamValue::Reference(ref_name));
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "probs must be a list of floats or a string parameter reference",
        ));
    }

    Ok(PyDistribution {
        dist_type: "Multinomial".to_string(),
        params,
    })
}
