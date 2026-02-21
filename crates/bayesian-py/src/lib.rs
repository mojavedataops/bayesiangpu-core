//! Python bindings for BayesianGPU
//!
//! This crate provides Python bindings using PyO3 for the BayesianGPU
//! Bayesian inference library.
//!
//! # Example
//!
//! ```python
//! from bayesiangpu import Model, Normal, HalfNormal, Beta, Binomial, sample
//!
//! model = Model()
//! model.param("theta", Beta(1, 1))
//! model.observe(Binomial(100, "theta"), [65])
//!
//! result = sample(model, num_samples=1000, num_chains=4)
//! print(result.summary())
//! ```

use pyo3::prelude::*;

mod distributions;
mod model;
mod result;
mod sampler;

use distributions::*;
use model::PyModel;
use result::{PyDiagnostics, PyInferenceResult, PyParameterSummary};
use sampler::{fit, quick_sample, sample, PyAdviResult};

/// Python module for BayesianGPU
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Model class
    m.add_class::<PyModel>()?;

    // Distribution classes
    m.add_class::<PyDistribution>()?;

    // Result classes
    m.add_class::<PyInferenceResult>()?;
    m.add_class::<PyDiagnostics>()?;
    m.add_class::<PyParameterSummary>()?;

    // Distribution factory functions
    m.add_function(wrap_pyfunction!(normal, m)?)?;
    m.add_function(wrap_pyfunction!(half_normal, m)?)?;
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    m.add_function(wrap_pyfunction!(exponential, m)?)?;
    m.add_function(wrap_pyfunction!(student_t, m)?)?;
    m.add_function(wrap_pyfunction!(half_cauchy, m)?)?;
    m.add_function(wrap_pyfunction!(cauchy, m)?)?;
    m.add_function(wrap_pyfunction!(laplace, m)?)?;
    m.add_function(wrap_pyfunction!(logistic, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(chi_squared, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_normal, m)?)?;
    m.add_function(wrap_pyfunction!(log_normal, m)?)?;
    m.add_function(wrap_pyfunction!(weibull, m)?)?;
    m.add_function(wrap_pyfunction!(pareto, m)?)?;
    m.add_function(wrap_pyfunction!(gumbel, m)?)?;
    m.add_function(wrap_pyfunction!(half_student_t, m)?)?;
    m.add_function(wrap_pyfunction!(multivariate_normal, m)?)?;
    m.add_function(wrap_pyfunction!(dirichlet, m)?)?;
    m.add_function(wrap_pyfunction!(multinomial, m)?)?;

    // Likelihood distributions
    m.add_function(wrap_pyfunction!(bernoulli, m)?)?;
    m.add_function(wrap_pyfunction!(binomial, m)?)?;
    m.add_function(wrap_pyfunction!(poisson, m)?)?;

    // Discrete distributions
    m.add_function(wrap_pyfunction!(negative_binomial, m)?)?;
    m.add_function(wrap_pyfunction!(categorical, m)?)?;
    m.add_function(wrap_pyfunction!(geometric, m)?)?;
    m.add_function(wrap_pyfunction!(discrete_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(beta_binomial, m)?)?;
    m.add_function(wrap_pyfunction!(zero_inflated_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(zero_inflated_neg_binomial, m)?)?;
    m.add_function(wrap_pyfunction!(hypergeometric, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_logistic, m)?)?;

    // Sampling functions
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(quick_sample, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;

    // ADVI result class
    m.add_class::<PyAdviResult>()?;

    // Diagnostic functions
    m.add_function(wrap_pyfunction!(result::is_converged, m)?)?;
    m.add_function(wrap_pyfunction!(result::summarize_parameter, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
