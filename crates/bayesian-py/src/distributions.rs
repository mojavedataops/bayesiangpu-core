//! Distribution types and factory functions for Python bindings

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A parameter value that can be either a number or a reference to another parameter
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Number(f64),
    LinearPredictor {
        #[serde(rename = "__type")]
        __type: String,
        matrix_key: String,
        param_name: String,
        num_cols: usize,
    },
    Reference(String),
}

impl ParamValue {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(n) = obj.extract::<f64>() {
            Ok(ParamValue::Number(n))
        } else if let Ok(s) = obj.extract::<String>() {
            Ok(ParamValue::Reference(s))
        } else if let Ok(dict) = obj.downcast::<pyo3::types::PyDict>() {
            if let Some(type_val) = dict.get_item("__type")? {
                let type_str: String = type_val.extract()?;
                if type_str == "LinearPredictor" {
                    let matrix_key: String = dict
                        .get_item("matrix_key")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing matrix_key")
                        })?
                        .extract()?;
                    let param_name: String = dict
                        .get_item("param_name")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing param_name")
                        })?
                        .extract()?;
                    let num_cols: usize = dict
                        .get_item("num_cols")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing num_cols")
                        })?
                        .extract()?;
                    return Ok(ParamValue::LinearPredictor {
                        __type: "LinearPredictor".to_string(),
                        matrix_key,
                        param_name,
                        num_cols,
                    });
                }
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unknown dict parameter type",
            ))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Parameter must be a number, string reference, or LinearPredictor dict",
            ))
        }
    }
}

#[allow(deprecated)]
impl IntoPy<PyObject> for ParamValue {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            ParamValue::Number(n) => n.into_py(py),
            ParamValue::LinearPredictor {
                __type,
                matrix_key,
                param_name,
                num_cols,
            } => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("__type", __type).unwrap();
                dict.set_item("matrix_key", matrix_key).unwrap();
                dict.set_item("param_name", param_name).unwrap();
                dict.set_item("num_cols", num_cols).unwrap();
                dict.into_py(py)
            }
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
                    ParamValue::LinearPredictor { param_name, .. } => {
                        format!("LinearPredictor({})", param_name)
                    }
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

/// Weibull distribution
///
/// Commonly used in reliability engineering and survival analysis.
///
/// Args:
///     shape: Shape parameter k (must be positive)
///     scale: Scale parameter lambda (must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Weibull(2, 1)  # Shape=2, scale=1
#[pyfunction]
#[pyo3(signature = (shape, scale))]
pub fn weibull(shape: f64, scale: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("shape".to_string(), ParamValue::Number(shape));
    params.insert("scale".to_string(), ParamValue::Number(scale));
    PyDistribution {
        dist_type: "Weibull".to_string(),
        params,
    }
}

/// Pareto distribution
///
/// Heavy-tailed distribution for modeling power-law phenomena.
///
/// Args:
///     alpha: Shape parameter (must be positive)
///     x_m: Scale/minimum parameter (must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Pareto(2, 1)  # Shape=2, minimum=1
#[pyfunction]
#[pyo3(signature = (alpha, x_m))]
pub fn pareto(alpha: f64, x_m: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("alpha".to_string(), ParamValue::Number(alpha));
    params.insert("x_m".to_string(), ParamValue::Number(x_m));
    PyDistribution {
        dist_type: "Pareto".to_string(),
        params,
    }
}

/// Gumbel distribution (Type-I extreme value)
///
/// Used for modeling the distribution of the maximum of samples.
///
/// Args:
///     loc: Location parameter (default: 0)
///     scale: Scale parameter (default: 1, must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Gumbel(0, 1)  # Standard Gumbel
#[pyfunction]
#[pyo3(signature = (loc=0.0, scale=1.0))]
pub fn gumbel(loc: f64, scale: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));
    PyDistribution {
        dist_type: "Gumbel".to_string(),
        params,
    }
}

/// Half Student's t distribution (positive values only)
///
/// Heavy-tailed prior for scale parameters, more robust than HalfNormal.
///
/// Args:
///     df: Degrees of freedom (must be positive)
///     scale: Scale parameter (default: 1, must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> HalfStudentT(3, 1)  # 3 degrees of freedom
#[pyfunction]
#[pyo3(signature = (df, scale=1.0))]
pub fn half_student_t(df: f64, scale: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("df".to_string(), ParamValue::Number(df));
    params.insert("scale".to_string(), ParamValue::Number(scale));
    PyDistribution {
        dist_type: "HalfStudentT".to_string(),
        params,
    }
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

/// Negative Binomial distribution
///
/// Models the number of failures before r successes.
///
/// Args:
///     r: Number of successes (positive)
///     p: Success probability (0 < p < 1)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> NegativeBinomial(5, 0.5)
#[pyfunction]
#[pyo3(signature = (r, p))]
pub fn negative_binomial(r: f64, p: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("r".to_string(), ParamValue::Number(r));
    params.insert("p".to_string(), ParamValue::Number(p));
    PyDistribution {
        dist_type: "NegativeBinomial".to_string(),
        params,
    }
}

/// Categorical distribution
///
/// Single draw from K categories with specified probabilities.
///
/// Args:
///     probs: Probability vector (list of non-negative floats summing to 1)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Categorical([0.2, 0.3, 0.5])
#[pyfunction]
pub fn categorical(probs: Vec<f64>) -> PyResult<PyDistribution> {
    if probs.len() < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Categorical requires at least 2 categories",
        ));
    }

    let mut params = HashMap::new();
    let probs_json = serde_json::to_string(&probs).unwrap();
    params.insert("probs_json".to_string(), ParamValue::Reference(probs_json));
    params.insert("dim".to_string(), ParamValue::Number(probs.len() as f64));

    Ok(PyDistribution {
        dist_type: "Categorical".to_string(),
        params,
    })
}

/// Geometric distribution
///
/// Models the number of failures before the first success.
///
/// Args:
///     p: Success probability (0 < p <= 1)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Geometric(0.3)
#[pyfunction]
#[pyo3(signature = (p))]
pub fn geometric(p: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("p".to_string(), ParamValue::Number(p));
    PyDistribution {
        dist_type: "Geometric".to_string(),
        params,
    }
}

/// Discrete Uniform distribution
///
/// Equal probability for each integer in [low, high].
///
/// Args:
///     low: Lower bound (integer)
///     high: Upper bound (integer)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> DiscreteUniform(1, 6)  # Fair die
#[pyfunction]
#[pyo3(signature = (low, high))]
pub fn discrete_uniform(low: f64, high: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("low".to_string(), ParamValue::Number(low));
    params.insert("high".to_string(), ParamValue::Number(high));
    PyDistribution {
        dist_type: "DiscreteUniform".to_string(),
        params,
    }
}

/// Beta-Binomial distribution
///
/// Compound distribution: Binomial with Beta-distributed success probability.
/// Models overdispersed binomial data.
///
/// Args:
///     n: Number of trials (positive integer)
///     alpha: First Beta shape parameter (positive)
///     beta: Second Beta shape parameter (positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> BetaBinomial(10, 2, 3)
#[pyfunction]
#[pyo3(signature = (n, alpha, beta))]
pub fn beta_binomial(n: f64, alpha: f64, beta: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("n".to_string(), ParamValue::Number(n));
    params.insert("alpha".to_string(), ParamValue::Number(alpha));
    params.insert("beta".to_string(), ParamValue::Number(beta));
    PyDistribution {
        dist_type: "BetaBinomial".to_string(),
        params,
    }
}

/// Zero-Inflated Poisson distribution
///
/// Models count data with excess zeros.
///
/// Args:
///     rate: Poisson rate parameter (lambda > 0)
///     zero_prob: Probability of structural zero (0 <= pi < 1)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> ZeroInflatedPoisson(3.0, 0.3)
#[pyfunction]
#[pyo3(signature = (rate, zero_prob))]
pub fn zero_inflated_poisson(rate: f64, zero_prob: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("rate".to_string(), ParamValue::Number(rate));
    params.insert("zero_prob".to_string(), ParamValue::Number(zero_prob));
    PyDistribution {
        dist_type: "ZeroInflatedPoisson".to_string(),
        params,
    }
}

/// Zero-Inflated Negative Binomial distribution
///
/// Models overdispersed count data with excess zeros.
///
/// Args:
///     r: Number of successes (positive)
///     p: Success probability (0 < p < 1)
///     zero_prob: Probability of structural zero (0 <= pi < 1)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> ZeroInflatedNegativeBinomial(5, 0.5, 0.3)
#[pyfunction]
#[pyo3(signature = (r, p, zero_prob))]
pub fn zero_inflated_neg_binomial(r: f64, p: f64, zero_prob: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("r".to_string(), ParamValue::Number(r));
    params.insert("p".to_string(), ParamValue::Number(p));
    params.insert("zero_prob".to_string(), ParamValue::Number(zero_prob));
    PyDistribution {
        dist_type: "ZeroInflatedNegativeBinomial".to_string(),
        params,
    }
}

/// Hypergeometric distribution
///
/// Models the number of successes in draws from a finite population
/// without replacement.
///
/// Args:
///     big_n: Population size (N >= 0)
///     big_k: Number of success states in population (0 <= K <= N)
///     n: Number of draws (0 <= n <= N)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> Hypergeometric(52, 13, 5)  # Drawing from deck of cards
#[pyfunction]
#[pyo3(signature = (big_n, big_k, n))]
pub fn hypergeometric(big_n: f64, big_k: f64, n: f64) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("big_n".to_string(), ParamValue::Number(big_n));
    params.insert("big_k".to_string(), ParamValue::Number(big_k));
    params.insert("n".to_string(), ParamValue::Number(n));
    PyDistribution {
        dist_type: "Hypergeometric".to_string(),
        params,
    }
}

/// LKJ Correlation distribution
///
/// Prior over correlation matrices parameterized by a single concentration
/// parameter eta. When eta = 1, the prior is uniform over valid correlation
/// matrices. When eta > 1, the prior favors the identity matrix.
///
/// The model stores D*(D-1)/2 unconstrained parameters that are transformed
/// to a Cholesky factor of a correlation matrix via partial correlations
/// (tanh transform).
///
/// Args:
///     dim: Dimension of the correlation matrix (D >= 2)
///     eta: Concentration parameter (must be positive)
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> LKJCorr(3, 1.0)  # Uniform over 3x3 correlation matrices
///     >>> LKJCorr(4, 2.0)  # Favor identity for 4x4
#[pyfunction]
pub fn lkj_corr(dim: usize, eta: f64) -> PyResult<PyDistribution> {
    if dim < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "LKJCorr requires dim >= 2",
        ));
    }
    if eta <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "LKJCorr requires eta > 0",
        ));
    }

    let mut params = HashMap::new();
    params.insert("dim".to_string(), ParamValue::Number(dim as f64));
    params.insert("eta".to_string(), ParamValue::Number(eta));

    Ok(PyDistribution {
        dist_type: "LKJCorr".to_string(),
        params,
    })
}

/// Ordered Logistic distribution
///
/// Models ordinal outcomes with K categories defined by K-1 cutpoints.
/// Used in ordinal regression models.
///
/// Args:
///     eta: Linear predictor (scalar)
///     cutpoints: Ordered vector of K-1 thresholds
///
/// Returns:
///     Distribution specification
///
/// Example:
///     >>> OrderedLogistic(0.5, [-1.0, 0.0, 1.0])
#[pyfunction]
#[pyo3(signature = (eta, cutpoints))]
pub fn ordered_logistic(eta: f64, cutpoints: Vec<f64>) -> PyDistribution {
    let mut params = HashMap::new();
    params.insert("eta".to_string(), ParamValue::Number(eta));
    let cutpoints_json = serde_json::to_string(&cutpoints).unwrap();
    params.insert(
        "cutpoints".to_string(),
        ParamValue::Reference(cutpoints_json),
    );
    params.insert(
        "k".to_string(),
        ParamValue::Number((cutpoints.len() + 1) as f64),
    );
    PyDistribution {
        dist_type: "OrderedLogistic".to_string(),
        params,
    }
}
