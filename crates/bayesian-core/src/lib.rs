//! BayesianGPU Core Library
//!
//! This crate provides the core probability distributions and model definitions
//! for GPU-accelerated Bayesian inference.

pub mod distributions;
pub mod math;
pub mod transforms;

// Re-export commonly used types
pub use distributions::beta::Beta;
pub use distributions::cauchy::Cauchy;
pub use distributions::dirichlet::Dirichlet;
pub use distributions::exponential::Exponential;
pub use distributions::gamma::Gamma;
pub use distributions::half_cauchy::HalfCauchy;
pub use distributions::half_normal::HalfNormal;
pub use distributions::log_normal::LogNormal;
pub use distributions::multinomial::{log_multinomial_coefficient, Multinomial};
pub use distributions::multivariate_normal::{
    mvn_from_covariance, mvn_from_precision, MultivariateNormal,
};
pub use distributions::normal::Normal;
pub use distributions::student_t::StudentT;
pub use distributions::uniform::Uniform;
pub use distributions::{Distribution, Support};

// Re-export math functions
pub use math::{digamma, ln_beta, ln_gamma};

// Re-export transforms
pub use transforms::{
    BoundedTransform, PositiveTransform, SimplexTransform, UnitIntervalTransform,
};
