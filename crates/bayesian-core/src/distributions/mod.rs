//! Probability distributions with autodiff support
//!
//! This module provides probability distributions that can be used with
//! the Burn framework's autodiff backend for gradient-based inference.

pub mod beta;
pub mod beta_binomial;
pub mod categorical;
pub mod cauchy;
pub mod chi_squared;
pub mod dirichlet;
pub mod dirichlet_multinomial;
pub mod discrete_uniform;
pub mod exponential;
pub mod gamma;
pub mod geometric;
pub mod gumbel;
pub mod half_cauchy;
pub mod half_normal;
pub mod half_student_t;
pub mod hypergeometric;
pub mod inverse_gamma;
pub mod laplace;
pub mod log_normal;
pub mod logistic;
pub mod multinomial;
pub mod multivariate_normal;
pub mod negative_binomial;
pub mod normal;
pub mod ordered_logistic;
pub mod pareto;
pub mod stick_breaking;
pub mod student_t;
pub mod truncated_normal;
pub mod uniform;
pub mod weibull;
pub mod zero_inflated_neg_binomial;
pub mod zero_inflated_poisson;

// Re-export distribution types
pub use beta::Beta;
pub use beta_binomial::BetaBinomial;
pub use categorical::Categorical;
pub use cauchy::Cauchy;
pub use chi_squared::ChiSquared;
pub use dirichlet::Dirichlet;
pub use dirichlet_multinomial::DirichletMultinomial;
pub use discrete_uniform::DiscreteUniform;
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use geometric::Geometric;
pub use gumbel::Gumbel;
pub use half_cauchy::HalfCauchy;
pub use half_normal::HalfNormal;
pub use half_student_t::HalfStudentT;
pub use hypergeometric::Hypergeometric;
pub use inverse_gamma::InverseGamma;
pub use laplace::Laplace;
pub use log_normal::LogNormal;
pub use logistic::Logistic;
pub use multinomial::{log_multinomial_coefficient, Multinomial};
pub use multivariate_normal::{mvn_from_covariance, mvn_from_precision, MultivariateNormal};
pub use negative_binomial::NegativeBinomial;
pub use normal::Normal;
pub use ordered_logistic::OrderedLogistic;
pub use pareto::Pareto;
pub use stick_breaking::{StickBreaking, GEM};
pub use student_t::StudentT;
pub use truncated_normal::TruncatedNormal;
pub use uniform::Uniform;
pub use weibull::Weibull;
pub use zero_inflated_neg_binomial::ZeroInflatedNegativeBinomial;
pub use zero_inflated_poisson::ZeroInflatedPoisson;

use burn::prelude::*;

/// Support of a probability distribution
#[derive(Debug, Clone, PartialEq)]
pub enum Support {
    /// Real line (-inf, inf)
    Real,
    /// Positive real numbers (0, inf)
    Positive,
    /// Unit interval (0, 1)
    UnitInterval,
    /// Non-negative integers 0, 1, 2, ...
    NonNegativeInteger,
    /// Simplex of dimension K (values sum to 1)
    Simplex(usize),
}

/// Trait for probability distributions with autodiff support
///
/// Distributions implement log probability density/mass functions that
/// can be differentiated through using Burn's autodiff backend.
pub trait Distribution<B: Backend> {
    /// Compute the log probability density/mass function
    ///
    /// # Arguments
    /// * `x` - Values at which to evaluate the log probability
    ///
    /// # Returns
    /// A tensor of the same shape as `x` containing log probabilities
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1>;

    /// Get the support of the distribution
    ///
    /// This is used to determine appropriate parameter transformations
    /// for unconstrained optimization.
    fn support(&self) -> Support;
}
