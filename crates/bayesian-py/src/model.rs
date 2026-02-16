//! Model builder for Python bindings

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::distributions::PyDistribution;

/// Prior specification for a parameter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prior {
    pub name: String,
    pub distribution: PyDistribution,
}

/// Likelihood specification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Likelihood {
    pub distribution: PyDistribution,
    pub observed: Vec<f64>,
}

/// Complete model specification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub priors: Vec<Prior>,
    pub likelihood: Option<Likelihood>,
}

/// Fluent builder for Bayesian models
///
/// Example:
///     >>> model = Model()
///     >>> model.param("mu", Normal(0, 10))
///     >>> model.param("sigma", HalfNormal(1))
///     >>> model.observe(Normal("mu", "sigma"), data)
#[pyclass(name = "Model")]
#[derive(Clone, Debug)]
pub struct PyModel {
    spec: ModelSpec,
}

#[pymethods]
impl PyModel {
    /// Create a new empty model
    #[new]
    pub fn new() -> Self {
        PyModel {
            spec: ModelSpec {
                priors: Vec::new(),
                likelihood: None,
            },
        }
    }

    /// Add a parameter with a prior distribution
    ///
    /// Args:
    ///     name: Parameter name (used to reference in likelihood)
    ///     distribution: Prior distribution
    ///
    /// Returns:
    ///     self for method chaining
    ///
    /// Example:
    ///     >>> model.param("theta", Beta(1, 1))
    pub fn param(&mut self, name: String, distribution: PyDistribution) -> PyResult<Self> {
        self.spec.priors.push(Prior { name, distribution });
        Ok(self.clone())
    }

    /// Set the likelihood (observed data) for the model
    ///
    /// Args:
    ///     distribution: Likelihood distribution
    ///     data: Observed data points
    ///
    /// Returns:
    ///     self for method chaining
    ///
    /// Example:
    ///     >>> model.observe(Binomial(100, "theta"), [65])
    pub fn observe(&mut self, distribution: PyDistribution, data: Vec<f64>) -> PyResult<Self> {
        self.spec.likelihood = Some(Likelihood {
            distribution,
            observed: data,
        });
        Ok(self.clone())
    }

    /// Get the list of parameter names
    #[getter]
    pub fn param_names(&self) -> Vec<String> {
        self.spec.priors.iter().map(|p| p.name.clone()).collect()
    }

    /// Get the number of parameters
    #[getter]
    pub fn num_params(&self) -> usize {
        self.spec.priors.len()
    }

    /// Check if the model has a likelihood
    #[getter]
    pub fn has_likelihood(&self) -> bool {
        self.spec.likelihood.is_some()
    }

    /// Convert model to JSON string (for internal use)
    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.spec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to serialize model: {}",
                e
            ))
        })
    }

    fn __repr__(&self) -> String {
        let params: Vec<String> = self
            .spec
            .priors
            .iter()
            .map(|p| format!("  {} ~ {}", p.name, p.distribution.repr_str()))
            .collect();

        let likelihood_str = if let Some(ref lik) = self.spec.likelihood {
            format!(
                "\nLikelihood:\n  data ~ {} (n={})",
                lik.distribution.repr_str(),
                lik.observed.len()
            )
        } else {
            "\nLikelihood: (not set)".to_string()
        };

        format!(
            "Model(\nPriors:\n{}\n{})",
            params.join("\n"),
            likelihood_str
        )
    }
}

impl PyModel {
    /// Get the internal model specification
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
}

impl Default for PyModel {
    fn default() -> Self {
        Self::new()
    }
}
