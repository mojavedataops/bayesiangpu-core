//! Model builder for Python bindings

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::distributions::PyDistribution;

/// Prior specification for a parameter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prior {
    pub name: String,
    pub distribution: PyDistribution,
    /// Number of elements for vector parameters (defaults to 1 for scalar)
    #[serde(default = "default_prior_size")]
    pub size: usize,
}

fn default_prior_size() -> usize {
    1
}

/// Likelihood specification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Likelihood {
    pub distribution: PyDistribution,
    pub observed: Vec<f64>,
    /// Per-observation known data (e.g., known standard deviations in Eight Schools)
    #[serde(default)]
    pub known: HashMap<String, Vec<f64>>,
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
    ///     size: Number of elements for vector parameters (defaults to 1)
    ///
    /// Returns:
    ///     self for method chaining
    ///
    /// Example:
    ///     >>> model.param("theta", Beta(1, 1))
    ///     >>> model.param("theta", Normal("mu", "tau"), size=8)
    #[pyo3(signature = (name, distribution, size=1))]
    pub fn param(
        &mut self,
        name: String,
        distribution: PyDistribution,
        size: usize,
    ) -> PyResult<Self> {
        self.spec.priors.push(Prior {
            name,
            distribution,
            size: size.max(1),
        });
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
    #[pyo3(signature = (distribution, data, known=None))]
    pub fn observe(
        &mut self,
        distribution: PyDistribution,
        data: Vec<f64>,
        known: Option<HashMap<String, Vec<f64>>>,
    ) -> PyResult<Self> {
        self.spec.likelihood = Some(Likelihood {
            distribution,
            observed: data,
            known: known.unwrap_or_default(),
        });
        Ok(self.clone())
    }

    /// Get the list of parameter names (expanded for vector params)
    #[getter]
    pub fn param_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for p in &self.spec.priors {
            let size = p.size.max(1);
            if size == 1 {
                names.push(p.name.clone());
            } else {
                for i in 0..size {
                    names.push(format!("{}[{}]", p.name, i));
                }
            }
        }
        names
    }

    /// Get the number of parameters (total dimension including vector params)
    #[getter]
    pub fn num_params(&self) -> usize {
        self.spec.priors.iter().map(|p| p.size.max(1)).sum()
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
