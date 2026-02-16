//! Inference result types for Python bindings

use pyo3::prelude::*;
use std::collections::HashMap;

/// MCMC diagnostics for a single parameter or overall
#[pyclass(name = "Diagnostics")]
#[derive(Clone, Debug)]
pub struct PyDiagnostics {
    /// R-hat convergence diagnostic (per parameter)
    #[pyo3(get)]
    pub rhat: HashMap<String, f64>,
    /// Effective sample size (per parameter)
    #[pyo3(get)]
    pub ess: HashMap<String, f64>,
    /// Number of divergent transitions
    #[pyo3(get)]
    pub divergences: usize,
}

#[pymethods]
impl PyDiagnostics {
    fn __repr__(&self) -> String {
        format!(
            "Diagnostics(divergences={}, rhat={:?}, ess={:?})",
            self.divergences, self.rhat, self.ess
        )
    }
}

/// Summary statistics for a single parameter
#[pyclass(name = "ParameterSummary")]
#[derive(Clone, Debug)]
pub struct PyParameterSummary {
    /// Parameter name
    #[pyo3(get)]
    pub name: String,
    /// Posterior mean
    #[pyo3(get)]
    pub mean: f64,
    /// Posterior standard deviation
    #[pyo3(get)]
    pub std: f64,
    /// 2.5th percentile (lower bound of 95% CI)
    #[pyo3(get)]
    pub q025: f64,
    /// 25th percentile
    #[pyo3(get)]
    pub q25: f64,
    /// Median (50th percentile)
    #[pyo3(get)]
    pub q50: f64,
    /// 75th percentile
    #[pyo3(get)]
    pub q75: f64,
    /// 97.5th percentile (upper bound of 95% CI)
    #[pyo3(get)]
    pub q975: f64,
    /// R-hat convergence diagnostic
    #[pyo3(get)]
    pub rhat: f64,
    /// Effective sample size
    #[pyo3(get)]
    pub ess: f64,
}

#[pymethods]
impl PyParameterSummary {
    fn __repr__(&self) -> String {
        format!(
            "ParameterSummary(name='{}', mean={:.3}, std={:.3}, 95% CI=[{:.3}, {:.3}], rhat={:.3}, ess={:.0})",
            self.name, self.mean, self.std, self.q025, self.q975, self.rhat, self.ess
        )
    }

    /// Get the 95% credible interval as a tuple
    pub fn ci95(&self) -> (f64, f64) {
        (self.q025, self.q975)
    }
}

/// Complete inference result
#[pyclass(name = "InferenceResult")]
#[derive(Clone, Debug)]
pub struct PyInferenceResult {
    /// Samples for each parameter (flattened across chains)
    samples: HashMap<String, Vec<f64>>,
    /// Samples organized by chain
    chain_samples: HashMap<String, Vec<Vec<f64>>>,
    /// MCMC diagnostics
    diagnostics: PyDiagnostics,
    /// Number of samples per chain
    num_samples: usize,
    /// Number of warmup iterations
    num_warmup: usize,
    /// Number of chains
    num_chains: usize,
    /// Final step size
    step_size: f64,
}

#[pymethods]
impl PyInferenceResult {
    /// Get samples for a parameter
    ///
    /// Args:
    ///     param: Parameter name
    ///
    /// Returns:
    ///     List of samples (flattened across chains)
    pub fn get_samples(&self, param: &str) -> PyResult<Vec<f64>> {
        self.samples.get(param).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Parameter '{}' not found in results",
                param
            ))
        })
    }

    /// Get samples for a parameter organized by chain
    ///
    /// Args:
    ///     param: Parameter name
    ///
    /// Returns:
    ///     List of lists, one per chain
    pub fn get_chain_samples(&self, param: &str) -> PyResult<Vec<Vec<f64>>> {
        self.chain_samples.get(param).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Parameter '{}' not found in results",
                param
            ))
        })
    }

    /// Get all parameter names
    #[getter]
    pub fn param_names(&self) -> Vec<String> {
        self.samples.keys().cloned().collect()
    }

    /// Get diagnostics
    #[getter]
    pub fn diagnostics(&self) -> PyDiagnostics {
        self.diagnostics.clone()
    }

    /// Get number of samples per chain
    #[getter]
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Get number of warmup iterations
    #[getter]
    pub fn num_warmup(&self) -> usize {
        self.num_warmup
    }

    /// Get number of chains
    #[getter]
    pub fn num_chains(&self) -> usize {
        self.num_chains
    }

    /// Get final step size
    #[getter]
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Compute summary statistics for a parameter
    ///
    /// Args:
    ///     param: Parameter name
    ///
    /// Returns:
    ///     ParameterSummary with mean, std, quantiles, rhat, and ess
    pub fn summarize(&self, param: &str) -> PyResult<PyParameterSummary> {
        let samples = self.get_samples(param)?;
        let n = samples.len();

        if n == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No samples available",
            ));
        }

        // Compute mean
        let mean = samples.iter().sum::<f64>() / n as f64;

        // Compute standard deviation
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = variance.sqrt();

        // Sort for quantiles
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let quantile = |p: f64| -> f64 {
            let idx = p * (n - 1) as f64;
            let lower = idx.floor() as usize;
            let upper = (lower + 1).min(n - 1);
            let weight = idx - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        };

        Ok(PyParameterSummary {
            name: param.to_string(),
            mean,
            std,
            q025: quantile(0.025),
            q25: quantile(0.25),
            q50: quantile(0.5),
            q75: quantile(0.75),
            q975: quantile(0.975),
            rhat: *self.diagnostics.rhat.get(param).unwrap_or(&f64::NAN),
            ess: *self.diagnostics.ess.get(param).unwrap_or(&f64::NAN),
        })
    }

    /// Compute summary statistics for all parameters
    ///
    /// Returns:
    ///     Dictionary mapping parameter names to ParameterSummary
    pub fn summary(&self) -> PyResult<HashMap<String, PyParameterSummary>> {
        let mut result = HashMap::new();
        for param in self.samples.keys() {
            result.insert(param.clone(), self.summarize(param)?);
        }
        Ok(result)
    }

    /// Check if all diagnostics indicate good convergence
    ///
    /// Returns:
    ///     True if all R-hat values are below 1.01
    pub fn is_converged(&self) -> bool {
        self.diagnostics
            .rhat
            .values()
            .all(|&r| !r.is_nan() && r < 1.01)
    }

    /// Check if effective sample size is sufficient
    ///
    /// Args:
    ///     min_ess: Minimum ESS required (default: 400)
    ///
    /// Returns:
    ///     True if all ESS values are above the minimum
    #[pyo3(signature = (min_ess=400.0))]
    pub fn has_sufficient_ess(&self, min_ess: f64) -> bool {
        self.diagnostics
            .ess
            .values()
            .all(|&e| !e.is_nan() && e >= min_ess)
    }

    /// Get warning messages for any diagnostic issues
    ///
    /// Returns:
    ///     List of warning messages
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check R-hat values
        for (param, &rhat) in &self.diagnostics.rhat {
            if rhat >= 1.1 {
                warnings.push(format!(
                    "{}: R-hat = {:.3} indicates poor convergence (should be < 1.01)",
                    param, rhat
                ));
            } else if rhat >= 1.01 {
                warnings.push(format!(
                    "{}: R-hat = {:.3} is marginal (ideally < 1.01)",
                    param, rhat
                ));
            }
        }

        // Check ESS values
        for (param, &ess) in &self.diagnostics.ess {
            if ess < 100.0 {
                warnings.push(format!(
                    "{}: ESS = {:.0} is too low (should be > 100)",
                    param, ess
                ));
            } else if ess < 400.0 {
                warnings.push(format!(
                    "{}: ESS = {:.0} is low (ideally > 400)",
                    param, ess
                ));
            }
        }

        // Check divergences
        let total_samples = self.num_samples * self.num_chains;
        let divergence_rate = self.diagnostics.divergences as f64 / total_samples as f64;
        if divergence_rate > 0.05 {
            warnings.push(format!(
                "{} divergences ({:.1}%) - consider reparameterizing",
                self.diagnostics.divergences,
                divergence_rate * 100.0
            ));
        } else if self.diagnostics.divergences > 0 {
            warnings.push(format!(
                "{} divergences detected",
                self.diagnostics.divergences
            ));
        }

        warnings
    }

    /// Format results as a summary table string
    ///
    /// Returns:
    ///     Formatted table string
    pub fn format_summary(&self) -> PyResult<String> {
        let summaries = self.summary()?;
        let mut params: Vec<_> = summaries.keys().collect();
        params.sort();

        if params.is_empty() {
            return Ok("No parameters in result".to_string());
        }

        let mut lines = Vec::new();

        // Header
        lines.push(format!(
            "{:15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "Parameter", "Mean", "Std", "2.5%", "Median", "97.5%", "R-hat", "ESS"
        ));
        lines.push("-".repeat(90));

        // Rows
        for param in params {
            let s = &summaries[param];
            lines.push(format!(
                "{:15} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>8.3} {:>8.0}",
                param, s.mean, s.std, s.q025, s.q50, s.q975, s.rhat, s.ess
            ));
        }

        Ok(lines.join("\n"))
    }

    fn __repr__(&self) -> String {
        format!(
            "InferenceResult(params={}, samples={}, chains={}, divergences={})",
            self.samples.len(),
            self.num_samples,
            self.num_chains,
            self.diagnostics.divergences
        )
    }

    fn __str__(&self) -> PyResult<String> {
        self.format_summary()
    }
}

impl PyInferenceResult {
    /// Create a new inference result
    pub fn new(
        samples: HashMap<String, Vec<f64>>,
        chain_samples: HashMap<String, Vec<Vec<f64>>>,
        diagnostics: PyDiagnostics,
        num_samples: usize,
        num_warmup: usize,
        num_chains: usize,
        step_size: f64,
    ) -> Self {
        PyInferenceResult {
            samples,
            chain_samples,
            diagnostics,
            num_samples,
            num_warmup,
            num_chains,
            step_size,
        }
    }
}

// ============================================================================
// Standalone Functions
// ============================================================================

/// Check if inference result indicates good convergence
///
/// Args:
///     result: The inference result
///
/// Returns:
///     True if all R-hat values are below 1.01
#[pyfunction]
pub fn is_converged(result: &PyInferenceResult) -> bool {
    result.is_converged()
}

/// Compute summary statistics for a parameter
///
/// Args:
///     result: The inference result
///     param: Parameter name
///
/// Returns:
///     ParameterSummary with mean, std, quantiles, rhat, and ess
#[pyfunction]
pub fn summarize_parameter(
    result: &PyInferenceResult,
    param: &str,
) -> PyResult<PyParameterSummary> {
    result.summarize(param)
}
