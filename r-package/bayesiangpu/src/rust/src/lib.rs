//! BayesianGPU R bindings using extendr
//!
//! This module provides R bindings for Bayesian inference using NUTS sampling.

use extendr_api::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;

use bayesian_core::{Beta, Distribution, Exponential, Gamma, HalfCauchy, HalfNormal, Normal, Uniform};
use bayesian_diagnostics::{ess, rhat};
use bayesian_rng::GpuRng;
use bayesian_sampler::{
    BayesianModel, NutsConfig, NutsSampler,
    AdviConfig, MeanFieldAdvi, FullRankAdvi,
};

// Type alias for our backend
type RBackend = Autodiff<NdArray<f32>>;

// ============================================================================
// Distribution Types
// ============================================================================

/// A parameter value that can be either a number or a reference to another parameter
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Number(f64),
    Reference(String),
}

/// Distribution specification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RDistribution {
    pub dist_type: String,
    pub params: HashMap<String, ParamValue>,
}

/// Prior specification for a parameter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prior {
    pub name: String,
    pub distribution: RDistribution,
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
    pub distribution: RDistribution,
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

// ============================================================================
// Distribution Factory Functions
// ============================================================================

/// Create a Normal distribution specification
/// @export
#[extendr]
fn normal_dist(loc: Robj, scale: Robj) -> String {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), parse_param_value(&loc));
    params.insert("scale".to_string(), parse_param_value(&scale));

    let dist = RDistribution {
        dist_type: "Normal".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a HalfNormal distribution specification
/// @export
#[extendr]
fn half_normal_dist(scale: f64) -> String {
    let mut params = HashMap::new();
    params.insert("scale".to_string(), ParamValue::Number(scale));

    let dist = RDistribution {
        dist_type: "HalfNormal".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a HalfCauchy distribution specification
/// @export
#[extendr]
fn half_cauchy_dist(scale: f64) -> String {
    let mut params = HashMap::new();
    params.insert("scale".to_string(), ParamValue::Number(scale));

    let dist = RDistribution {
        dist_type: "HalfCauchy".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Beta distribution specification
/// @export
#[extendr]
fn beta_dist(alpha: f64, beta: f64) -> String {
    let mut params = HashMap::new();
    params.insert("alpha".to_string(), ParamValue::Number(alpha));
    params.insert("beta".to_string(), ParamValue::Number(beta));

    let dist = RDistribution {
        dist_type: "Beta".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Gamma distribution specification
/// @export
#[extendr]
fn gamma_dist(shape: f64, rate: f64) -> String {
    let mut params = HashMap::new();
    params.insert("shape".to_string(), ParamValue::Number(shape));
    params.insert("rate".to_string(), ParamValue::Number(rate));

    let dist = RDistribution {
        dist_type: "Gamma".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Uniform distribution specification
/// @export
#[extendr]
fn uniform_dist(low: f64, high: f64) -> String {
    let mut params = HashMap::new();
    params.insert("low".to_string(), ParamValue::Number(low));
    params.insert("high".to_string(), ParamValue::Number(high));

    let dist = RDistribution {
        dist_type: "Uniform".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create an Exponential distribution specification
/// @export
#[extendr]
fn exponential_dist(rate: f64) -> String {
    let mut params = HashMap::new();
    params.insert("rate".to_string(), ParamValue::Number(rate));

    let dist = RDistribution {
        dist_type: "Exponential".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Cauchy distribution specification
/// @export
#[extendr]
fn cauchy_dist(loc: f64, scale: f64) -> String {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));

    let dist = RDistribution {
        dist_type: "Cauchy".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Student's t distribution specification
/// @export
#[extendr]
fn student_t_dist(df: f64, loc: f64, scale: f64) -> String {
    let mut params = HashMap::new();
    params.insert("df".to_string(), ParamValue::Number(df));
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));

    let dist = RDistribution {
        dist_type: "StudentT".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a LogNormal distribution specification
/// @export
#[extendr]
fn log_normal_dist(loc: Robj, scale: Robj) -> String {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), parse_param_value(&loc));
    params.insert("scale".to_string(), parse_param_value(&scale));

    let dist = RDistribution {
        dist_type: "LogNormal".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Bernoulli distribution specification
/// @export
#[extendr]
fn bernoulli_dist(p: Robj) -> String {
    let mut params = HashMap::new();
    params.insert("p".to_string(), parse_param_value(&p));

    let dist = RDistribution {
        dist_type: "Bernoulli".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Binomial distribution specification
/// @export
#[extendr]
fn binomial_dist(n: i32, p: Robj) -> String {
    let mut params = HashMap::new();
    params.insert("n".to_string(), ParamValue::Number(n as f64));
    params.insert("p".to_string(), parse_param_value(&p));

    let dist = RDistribution {
        dist_type: "Binomial".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Poisson distribution specification
/// @export
#[extendr]
fn poisson_dist(rate: Robj) -> String {
    let mut params = HashMap::new();
    params.insert("rate".to_string(), parse_param_value(&rate));

    let dist = RDistribution {
        dist_type: "Poisson".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a MultivariateNormal distribution specification
///
/// @param mu Mean vector (numeric vector)
/// @param cov Covariance matrix (optional, matrix or nested list)
/// @param scale_tril Lower triangular Cholesky factor (optional, matrix or nested list)
/// @return JSON string with distribution specification
/// @export
#[extendr]
fn multivariate_normal_dist(mu: Robj, cov: Robj, scale_tril: Robj) -> String {
    let mut params = HashMap::new();

    // Parse mu as a vector
    let mu_vec: Vec<f64> = if let Some(r) = mu.as_real_vector() {
        r
    } else {
        vec![]
    };
    let mu_json = serde_json::to_string(&mu_vec).unwrap();
    params.insert("mu".to_string(), ParamValue::Number(0.0)); // placeholder
    params.insert("mu_json".to_string(), ParamValue::Reference(mu_json));

    // Check if cov is provided (not NULL)
    if !cov.is_null() {
        // Try to parse as matrix or list
        let cov_mat = parse_matrix(&cov);
        let cov_json = serde_json::to_string(&cov_mat).unwrap();
        params.insert("cov_json".to_string(), ParamValue::Reference(cov_json));
        params.insert(
            "parameterization".to_string(),
            ParamValue::Reference("covariance".to_string()),
        );
    } else if !scale_tril.is_null() {
        // Try to parse scale_tril as matrix or list
        let tril_mat = parse_matrix(&scale_tril);
        let tril_json = serde_json::to_string(&tril_mat).unwrap();
        params.insert(
            "scale_tril_json".to_string(),
            ParamValue::Reference(tril_json),
        );
        params.insert(
            "parameterization".to_string(),
            ParamValue::Reference("cholesky".to_string()),
        );
    } else {
        panic!("Must provide either cov or scale_tril for MultivariateNormal");
    }

    let dist = RDistribution {
        dist_type: "MultivariateNormal".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Dirichlet distribution specification
///
/// @param alpha Concentration vector (numeric vector of positive values)
/// @return JSON string with distribution specification
/// @export
#[extendr]
fn dirichlet_dist(alpha: Robj) -> String {
    let mut params = HashMap::new();

    // Parse alpha as a vector
    let alpha_vec: Vec<f64> = if let Some(r) = alpha.as_real_vector() {
        r
    } else {
        vec![]
    };

    if alpha_vec.len() < 2 {
        panic!("Dirichlet requires at least 2 categories");
    }

    let alpha_json = serde_json::to_string(&alpha_vec).unwrap();
    params.insert("alpha_json".to_string(), ParamValue::Reference(alpha_json));
    params.insert("dim".to_string(), ParamValue::Number(alpha_vec.len() as f64));

    let dist = RDistribution {
        dist_type: "Dirichlet".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Multinomial distribution specification
///
/// @param n Number of trials (positive integer)
/// @param probs Probability vector (can be a numeric vector or parameter name)
/// @return JSON string with distribution specification
/// @export
#[extendr]
fn multinomial_dist(n: i32, probs: Robj) -> String {
    let mut params = HashMap::new();
    params.insert("n".to_string(), ParamValue::Number(n as f64));

    // Check if probs is a vector or a string reference
    if let Some(probs_vec) = probs.as_real_vector() {
        if probs_vec.len() < 2 {
            panic!("Multinomial requires at least 2 categories");
        }
        let probs_json = serde_json::to_string(&probs_vec).unwrap();
        params.insert("probs_json".to_string(), ParamValue::Reference(probs_json));
        params.insert("dim".to_string(), ParamValue::Number(probs_vec.len() as f64));
    } else if let Some(ref_name) = probs.as_str() {
        // Reference to another parameter (e.g., Dirichlet prior)
        params.insert("probs".to_string(), ParamValue::Reference(ref_name.to_string()));
    } else {
        panic!("probs must be a numeric vector or a string parameter reference");
    }

    let dist = RDistribution {
        dist_type: "Multinomial".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Laplace distribution specification
/// @export
#[extendr]
fn laplace_dist(loc: f64, scale: f64) -> String {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));

    let dist = RDistribution {
        dist_type: "Laplace".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Logistic distribution specification
/// @export
#[extendr]
fn logistic_dist(loc: f64, scale: f64) -> String {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));

    let dist = RDistribution {
        dist_type: "Logistic".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create an Inverse Gamma distribution specification
/// @export
#[extendr]
fn inverse_gamma_dist(alpha: f64, beta: f64) -> String {
    let mut params = HashMap::new();
    params.insert("alpha".to_string(), ParamValue::Number(alpha));
    params.insert("beta".to_string(), ParamValue::Number(beta));

    let dist = RDistribution {
        dist_type: "InverseGamma".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Chi-Squared distribution specification
/// @export
#[extendr]
fn chi_squared_dist(df: f64) -> String {
    let mut params = HashMap::new();
    params.insert("df".to_string(), ParamValue::Number(df));

    let dist = RDistribution {
        dist_type: "ChiSquared".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Create a Truncated Normal distribution specification
/// @export
#[extendr]
fn truncated_normal_dist(loc: f64, scale: f64, low: f64, high: f64) -> String {
    let mut params = HashMap::new();
    params.insert("loc".to_string(), ParamValue::Number(loc));
    params.insert("scale".to_string(), ParamValue::Number(scale));
    params.insert("low".to_string(), ParamValue::Number(low));
    params.insert("high".to_string(), ParamValue::Number(high));

    let dist = RDistribution {
        dist_type: "TruncatedNormal".to_string(),
        params,
    };
    serde_json::to_string(&dist).unwrap()
}

/// Helper function to parse a matrix from R (matrix or list of lists)
fn parse_matrix(obj: &Robj) -> Vec<Vec<f64>> {
    // Try as a matrix first
    if obj.is_matrix() {
        if let Some(mat) = obj.as_real_vector() {
            // Get dimensions
            let dim = obj.get_attrib(Robj::from("dim")).unwrap();
            if let Some(dims) = dim.as_integer_vector() {
                let nrow = dims[0] as usize;
                let ncol = dims[1] as usize;
                // R stores matrices column-major
                let mut result = Vec::with_capacity(nrow);
                for i in 0..nrow {
                    let mut row = Vec::with_capacity(ncol);
                    for j in 0..ncol {
                        row.push(mat[j * nrow + i]);
                    }
                    result.push(row);
                }
                return result;
            }
        }
    }

    // Try as a list of vectors
    if let Some(list) = obj.as_list() {
        let mut result = Vec::new();
        for (_name, item) in list.iter() {
            if let Some(row) = item.as_real_vector() {
                result.push(row);
            }
        }
        return result;
    }

    vec![]
}

// Helper function to parse parameter values
fn parse_param_value(obj: &Robj) -> ParamValue {
    if let Some(n) = obj.as_real() {
        ParamValue::Number(n)
    } else if let Some(i) = obj.as_integer() {
        ParamValue::Number(i as f64)
    } else if let Some(s) = obj.as_str() {
        ParamValue::Reference(s.to_string())
    } else {
        ParamValue::Number(0.0)
    }
}

// ============================================================================
// Dynamic Model Implementation
// ============================================================================

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    let one = Tensor::<B, 1>::ones_like(&x);
    one.clone() / (one + x.neg().exp())
}

/// Dynamic model implementation that builds log_prob from a model spec
#[derive(Clone)]
struct DynamicModel {
    spec: ModelSpec,
    /// Size of each prior (1 for scalar, N for vector)
    prior_sizes: Vec<usize>,
    /// Offset of each prior in the flat parameter vector
    prior_offsets: Vec<usize>,
    device: NdArrayDevice,
}

impl DynamicModel {
    fn new(spec: ModelSpec) -> Self {
        let prior_sizes: Vec<usize> = spec.priors.iter().map(|p| p.size.max(1)).collect();
        let mut prior_offsets = Vec::with_capacity(prior_sizes.len());
        let mut offset = 0usize;
        for &sz in &prior_sizes {
            prior_offsets.push(offset);
            offset += sz;
        }

        DynamicModel {
            spec,
            prior_sizes,
            prior_offsets,
            device: NdArrayDevice::default(),
        }
    }

    fn scalar_tensor(&self, value: f64) -> Tensor<RBackend, 1> {
        Tensor::<RBackend, 1>::from_floats([value as f32], &self.device)
    }

    fn get_param_value(
        &self,
        value: &ParamValue,
        params: &Tensor<RBackend, 1>,
    ) -> Tensor<RBackend, 1> {
        match value {
            ParamValue::Number(n) => self.scalar_tensor(*n),
            ParamValue::Reference(name) => {
                let idx = self
                    .spec
                    .priors
                    .iter()
                    .position(|p| &p.name == name)
                    .expect("Parameter reference not found");
                let offset = self.prior_offsets[idx];
                params.clone().slice([offset..offset + 1])
            }
        }
    }

    /// Check if a distribution type has positive support (parameters stored in log space)
    fn is_positive_support_dist(dist_type: &str) -> bool {
        matches!(
            dist_type,
            "HalfCauchy" | "HalfNormal" | "Gamma" | "Exponential" | "LogNormal"
        )
    }

    /// Resolve a prior distribution parameter to a tensor value.
    /// If it's a number, return a scalar tensor.
    /// If it's a string reference, look up the named parameter in the full params tensor.
    /// For positive-support referenced parameters, apply exp() to get constrained value.
    fn resolve_prior_param(
        &self,
        value: &ParamValue,
        all_params: &Tensor<RBackend, 1>,
    ) -> Tensor<RBackend, 1> {
        match value {
            ParamValue::Number(n) => self.scalar_tensor(*n),
            ParamValue::Reference(name) => {
                let idx = self
                    .spec
                    .priors
                    .iter()
                    .position(|p| &p.name == name)
                    .expect("Parameter reference not found in prior");
                let offset = self.prior_offsets[idx];

                #[allow(clippy::single_range_in_vec_init)]
                let raw = all_params.clone().slice([offset..offset + 1]);

                // If the referenced parameter has a positive-support prior, it's stored in
                // log space (unconstrained). Apply exp() to get constrained value.
                let ref_dist_type = &self.spec.priors[idx].distribution.dist_type;
                if Self::is_positive_support_dist(ref_dist_type) {
                    raw.exp()
                } else {
                    raw
                }
            }
        }
    }

    fn compute_prior_logp(
        &self,
        dist: &RDistribution,
        value: &Tensor<RBackend, 1>,
        all_params: &Tensor<RBackend, 1>,
    ) -> Tensor<RBackend, 1> {
        match dist.dist_type.as_str() {
            "Normal" => {
                // loc and scale can be parameter references
                let loc_pv = dist.params.get("loc").cloned().unwrap_or(ParamValue::Number(0.0));
                let scale_pv = dist.params.get("scale").cloned().unwrap_or(ParamValue::Number(1.0));
                let loc_tensor = self.resolve_prior_param(&loc_pv, all_params);
                let scale_tensor = self.resolve_prior_param(&scale_pv, all_params);
                let normal = Normal::<RBackend>::new(loc_tensor, scale_tensor);
                normal.log_prob(value)
            }
            "HalfNormal" => {
                let scale = self.get_f64_param(&dist.params, "scale");
                let scale_tensor = self.scalar_tensor(scale);
                let half_normal = HalfNormal::<RBackend>::new(scale_tensor);
                let transformed = value.clone().exp();
                let logp = half_normal.log_prob(&transformed);
                logp + value.clone()
            }
            "Beta" => {
                let alpha = self.get_f64_param(&dist.params, "alpha");
                let beta_param = self.get_f64_param(&dist.params, "beta");
                let alpha_tensor = self.scalar_tensor(alpha);
                let beta_tensor = self.scalar_tensor(beta_param);
                let beta = Beta::<RBackend>::new(alpha_tensor, beta_tensor);
                let transformed = sigmoid(value.clone());
                let logp = beta.log_prob(&transformed);
                let sig = sigmoid(value.clone());
                let one = Tensor::<RBackend, 1>::ones_like(&sig);
                let jac = (sig.clone() * (one - sig.clone())).log();
                logp + jac
            }
            "Gamma" => {
                let shape = self.get_f64_param(&dist.params, "shape");
                let rate = self.get_f64_param(&dist.params, "rate");
                let shape_tensor = self.scalar_tensor(shape);
                let rate_tensor = self.scalar_tensor(rate);
                let gamma = Gamma::<RBackend>::new(shape_tensor, rate_tensor);
                let transformed = value.clone().exp();
                let logp = gamma.log_prob(&transformed);
                logp + value.clone()
            }
            "Uniform" => {
                let low = self.get_f64_param(&dist.params, "low");
                let high = self.get_f64_param(&dist.params, "high");
                let low_tensor = self.scalar_tensor(low);
                let high_tensor = self.scalar_tensor(high);
                let uniform = Uniform::<RBackend>::new(low_tensor, high_tensor);
                let range = high - low;
                let sig = sigmoid(value.clone());
                let transformed = sig.clone() * range as f32 + low as f32;
                let logp = uniform.log_prob(&transformed);
                let one = Tensor::<RBackend, 1>::ones_like(&sig);
                let jac = (sig.clone() * (one - sig) * range as f32).log();
                logp + jac
            }
            "Exponential" => {
                let rate = self.get_f64_param(&dist.params, "rate");
                let rate_tensor = self.scalar_tensor(rate);
                let exp_dist = Exponential::<RBackend>::new(rate_tensor);
                let transformed = value.clone().exp();
                let logp = exp_dist.log_prob(&transformed);
                logp + value.clone()
            }
            "HalfCauchy" => {
                let scale = self.get_f64_param(&dist.params, "scale");
                let scale_tensor = self.scalar_tensor(scale);
                let half_cauchy = HalfCauchy::<RBackend>::new(scale_tensor);
                let transformed = value.clone().exp();
                let logp = half_cauchy.log_prob(&transformed);
                logp + value.clone()
            }
            _ => Tensor::<RBackend, 1>::zeros([1], &self.device),
        }
    }

    /// Resolve a likelihood distribution parameter for a specific observation index.
    fn resolve_for_observation(
        &self,
        value: &ParamValue,
        obs_idx: usize,
        params: &Tensor<RBackend, 1>,
        known: &HashMap<String, Vec<f64>>,
    ) -> Tensor<RBackend, 1> {
        match value {
            ParamValue::Number(n) => self.scalar_tensor(*n),
            ParamValue::Reference(name) => {
                // Check known data first
                if let Some(known_vals) = known.get(name.as_str()) {
                    return self.scalar_tensor(known_vals.get(obs_idx).copied().unwrap_or(0.0));
                }
                // Check model parameters
                if let Some(idx) = self.spec.priors.iter().position(|p| &p.name == name) {
                    let offset = self.prior_offsets[idx];
                    let size = self.prior_sizes[idx];
                    let param_offset = if size > 1 && obs_idx < size {
                        offset + obs_idx
                    } else {
                        offset
                    };
                    #[allow(clippy::single_range_in_vec_init)]
                    let raw = params.clone().slice([param_offset..param_offset + 1]);
                    let ref_dist_type = &self.spec.priors[idx].distribution.dist_type;
                    if Self::is_positive_support_dist(ref_dist_type) {
                        raw.exp()
                    } else {
                        raw
                    }
                } else {
                    self.scalar_tensor(0.0)
                }
            }
        }
    }

    fn compute_likelihood_logp(
        &self,
        likelihood: &Likelihood,
        params: &Tensor<RBackend, 1>,
    ) -> Tensor<RBackend, 1> {
        let dist = &likelihood.distribution;
        let observed = &likelihood.observed;
        let known = &likelihood.known;

        match dist.dist_type.as_str() {
            "Bernoulli" => {
                let p = self.get_param_value(
                    dist.params.get("p").expect("Bernoulli needs p"),
                    params,
                );
                let p_constrained = sigmoid(p);

                let mut total_logp = Tensor::<RBackend, 1>::zeros([1], &self.device);
                let one = self.scalar_tensor(1.0);
                for &y in observed {
                    let y_tensor = self.scalar_tensor(y);
                    let log_p = p_constrained.clone().log();
                    let log_1_minus_p = (one.clone() - p_constrained.clone()).log();
                    let logp = y_tensor.clone() * log_p + (one.clone() - y_tensor) * log_1_minus_p;
                    total_logp = total_logp + logp;
                }
                total_logp
            }
            "Binomial" => {
                let n = self.get_f64_param(&dist.params, "n") as usize;
                let p = self.get_param_value(
                    dist.params.get("p").expect("Binomial needs p"),
                    params,
                );
                let p_constrained = sigmoid(p);

                let mut total_logp = Tensor::<RBackend, 1>::zeros([1], &self.device);
                let one = self.scalar_tensor(1.0);
                for &k in observed {
                    let k_tensor = self.scalar_tensor(k);
                    let n_minus_k = self.scalar_tensor((n as f64 - k) as f64);
                    let log_p = p_constrained.clone().log();
                    let log_1_minus_p = (one.clone() - p_constrained.clone()).log();
                    let logp = k_tensor * log_p + n_minus_k * log_1_minus_p;
                    total_logp = total_logp + logp;
                }
                total_logp
            }
            "Normal" => {
                let loc_pv = dist.params.get("loc").expect("Normal needs loc");
                let scale_pv = dist.params.get("scale").expect("Normal needs scale");

                // Per-observation resolution for loc and scale
                let mut total_logp = Tensor::<RBackend, 1>::zeros([1], &self.device);
                let half = self.scalar_tensor(0.5);
                for (j, &y) in observed.iter().enumerate() {
                    let loc_j = self.resolve_for_observation(loc_pv, j, params, known);
                    let scale_j = self.resolve_for_observation(scale_pv, j, params, known);
                    let y_tensor = self.scalar_tensor(y);
                    let diff = y_tensor - loc_j;
                    let z = diff / scale_j.clone();
                    let logp = -scale_j.log() - half.clone() * z.powf_scalar(2.0);
                    total_logp = total_logp + logp;
                }
                total_logp
            }
            "Poisson" => {
                let rate = self.get_param_value(
                    dist.params.get("rate").expect("Poisson needs rate"),
                    params,
                );
                let rate_constrained = rate.exp();

                let mut total_logp = Tensor::<RBackend, 1>::zeros([1], &self.device);
                for &k in observed {
                    let k_tensor = self.scalar_tensor(k);
                    let logp = k_tensor * rate_constrained.clone().log() - rate_constrained.clone();
                    total_logp = total_logp + logp;
                }
                total_logp
            }
            _ => Tensor::<RBackend, 1>::zeros([1], &self.device),
        }
    }

    fn get_f64_param(&self, params: &HashMap<String, ParamValue>, key: &str) -> f64 {
        match params.get(key) {
            Some(ParamValue::Number(n)) => *n,
            _ => panic!("Expected numeric parameter for {}", key),
        }
    }
}

impl BayesianModel<RBackend> for DynamicModel {
    fn dim(&self) -> usize {
        self.prior_sizes.iter().sum()
    }

    fn log_prob(&self, params: &Tensor<RBackend, 1>) -> Tensor<RBackend, 1> {
        let mut total_logp = Tensor::<RBackend, 1>::zeros([1], &self.device);

        for (i, prior) in self.spec.priors.iter().enumerate() {
            let offset = self.prior_offsets[i];
            let size = self.prior_sizes[i];
            for j in 0..size {
                let param_value = params.clone().slice([offset + j..offset + j + 1]);
                let logp = self.compute_prior_logp(&prior.distribution, &param_value, params);
                total_logp = total_logp + logp;
            }
        }

        if let Some(ref likelihood) = self.spec.likelihood {
            let logp = self.compute_likelihood_logp(likelihood, params);
            total_logp = total_logp + logp;
        }

        total_logp
    }

    fn param_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for (i, prior) in self.spec.priors.iter().enumerate() {
            let size = self.prior_sizes[i];
            if size == 1 {
                names.push(prior.name.clone());
            } else {
                for j in 0..size {
                    names.push(format!("{}[{}]", prior.name, j));
                }
            }
        }
        names
    }
}

// ============================================================================
// Sampling Function
// ============================================================================

/// Run NUTS sampling on a model specification (JSON string)
/// @export
#[extendr]
fn run_nuts_sampling(
    model_json: &str,
    num_samples: i32,
    num_warmup: i32,
    num_chains: i32,
    target_accept: f64,
    seed: i32,
) -> List {
    let spec: ModelSpec = serde_json::from_str(model_json).expect("Failed to parse model JSON");

    if spec.priors.is_empty() {
        panic!("Model must have at least one parameter");
    }

    let dynamic_model = DynamicModel::new(spec);
    let dim = dynamic_model.dim();
    let param_names = dynamic_model.param_names();
    let device = NdArrayDevice::default();

    let config = NutsConfig {
        num_samples: num_samples as usize,
        num_warmup: num_warmup as usize,
        max_tree_depth: 10,
        target_accept,
        init_step_size: 1.0,
    };

    let mut all_chain_samples: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for name in &param_names {
        all_chain_samples.insert(name.clone(), Vec::new());
    }

    let mut total_divergences = 0i32;
    let mut final_step_size = 0.0f64;

    for chain_idx in 0..num_chains {
        let chain_seed = (seed as u64) + (chain_idx as u64);
        let init_vals: Vec<f32> = (0..dim)
            .map(|i| {
                let x = ((chain_seed.wrapping_mul(1103515245).wrapping_add(i as u64 * 12345)) % 1000)
                    as f32;
                (x / 500.0 - 1.0) * 0.1
            })
            .collect();
        let init = Tensor::<RBackend, 1>::from_floats(init_vals.as_slice(), &device);

        let rng = GpuRng::<RBackend>::new(chain_seed, dim.max(64), &device);
        let mut sampler = NutsSampler::new(dynamic_model.clone(), config.clone(), rng);
        let result = sampler.sample(init);

        for (i, name) in param_names.iter().enumerate() {
            let chain_samples: Vec<f64> = result
                .samples
                .iter()
                .map(|s| {
                    let data: Vec<f32> = s.clone().into_data().to_vec().unwrap();
                    data[i] as f64
                })
                .collect();
            all_chain_samples.get_mut(name).unwrap().push(chain_samples);
        }

        total_divergences += result.divergences as i32;
        final_step_size = result.final_step_size;
    }

    // Flatten samples and compute diagnostics
    let mut flattened_samples: HashMap<String, Vec<f64>> = HashMap::new();
    let mut rhat_values: HashMap<String, f64> = HashMap::new();
    let mut ess_values: HashMap<String, f64> = HashMap::new();

    for name in &param_names {
        let chain_samples = all_chain_samples.get(name).unwrap();
        let flat: Vec<f64> = chain_samples.iter().flatten().copied().collect();
        flattened_samples.insert(name.clone(), flat);

        let rhat_val = rhat(chain_samples);
        let ess_val = ess(chain_samples);

        rhat_values.insert(name.clone(), rhat_val);
        ess_values.insert(name.clone(), ess_val);
    }

    // Build result list
    let mut samples_list = List::new(param_names.len());
    for (i, name) in param_names.iter().enumerate() {
        let samples = flattened_samples.get(name).unwrap();
        samples_list.set_elt(i, Doubles::from_values(samples.iter().copied()).into()).unwrap();
    }
    samples_list.set_names(param_names.clone()).unwrap();

    let mut rhat_list = List::new(param_names.len());
    for (i, name) in param_names.iter().enumerate() {
        rhat_list.set_elt(i, Rfloat::from(*rhat_values.get(name).unwrap()).into()).unwrap();
    }
    rhat_list.set_names(param_names.clone()).unwrap();

    let mut ess_list = List::new(param_names.len());
    for (i, name) in param_names.iter().enumerate() {
        ess_list.set_elt(i, Rfloat::from(*ess_values.get(name).unwrap()).into()).unwrap();
    }
    ess_list.set_names(param_names.clone()).unwrap();

    // Build chain samples (list of lists)
    let mut chains_list = List::new(param_names.len());
    for (i, name) in param_names.iter().enumerate() {
        let chain_samples = all_chain_samples.get(name).unwrap();
        let mut param_chains = List::new(chain_samples.len());
        for (j, chain) in chain_samples.iter().enumerate() {
            param_chains.set_elt(j, Doubles::from_values(chain.iter().copied()).into()).unwrap();
        }
        chains_list.set_elt(i, param_chains.into()).unwrap();
    }
    chains_list.set_names(param_names.clone()).unwrap();

    list!(
        samples = samples_list,
        chains = chains_list,
        rhat = rhat_list,
        ess = ess_list,
        divergences = total_divergences,
        step_size = final_step_size,
        num_samples = num_samples,
        num_warmup = num_warmup,
        num_chains = num_chains,
        param_names = param_names
    )
}

/// Run ADVI (Automatic Differentiation Variational Inference)
///
/// ADVI approximates the posterior with a Gaussian and optimizes the
/// Evidence Lower Bound (ELBO) using gradient descent.
///
/// @param model_json JSON string with model specification
/// @param method Either "mean_field" or "full_rank"
/// @param num_iterations Maximum number of optimization iterations
/// @param num_samples Number of Monte Carlo samples for gradient estimation
/// @param learning_rate Learning rate for Adam optimizer
/// @param seed Random seed
/// @return List with posterior approximation and diagnostics
/// @export
#[extendr]
fn run_advi(
    model_json: &str,
    method: &str,
    num_iterations: i32,
    num_samples: i32,
    learning_rate: f64,
    seed: i32,
) -> List {
    let spec: ModelSpec = serde_json::from_str(model_json).expect("Failed to parse model JSON");

    if spec.priors.is_empty() {
        panic!("Model must have at least one parameter");
    }

    let dynamic_model = DynamicModel::new(spec);
    let dim = dynamic_model.dim();
    let param_names = dynamic_model.param_names();
    let device = NdArrayDevice::default();

    let config = AdviConfig {
        num_iterations: num_iterations as usize,
        num_samples: num_samples as usize,
        learning_rate,
        ..AdviConfig::default()
    };

    let rng = GpuRng::<RBackend>::new(seed as u64, dim.max(64), &device);

    match method {
        "mean_field" => {
            let mut advi = MeanFieldAdvi::new(dynamic_model, config, rng);
            let result = advi.fit(None, None);

            // Build result list
            let mu = Doubles::from_values(result.mu.iter().copied());
            let sigma = Doubles::from_values(result.sigma.iter().copied());
            let elbo_history = Doubles::from_values(result.elbo_history.iter().copied());

            list!(
                mu = mu,
                sigma = sigma,
                elbo_history = elbo_history,
                final_elbo = result.final_elbo,
                iterations = result.iterations as i32,
                converged = result.converged,
                param_names = param_names,
                method = "mean_field"
            )
        }
        "full_rank" => {
            let mut advi = FullRankAdvi::new(dynamic_model, config, rng);
            let result = advi.fit(None, None);

            // Compute covariance and std
            let cov = result.covariance();
            let stds = result.std();

            // Flatten covariance for R matrix
            let cov_flat: Vec<f64> = cov.iter().flatten().copied().collect();

            let mu = Doubles::from_values(result.mu.iter().copied());
            let sigma = Doubles::from_values(stds.iter().copied());
            let covariance = Doubles::from_values(cov_flat.iter().copied());
            let elbo_history = Doubles::from_values(result.elbo_history.iter().copied());

            list!(
                mu = mu,
                sigma = sigma,
                covariance = covariance,
                elbo_history = elbo_history,
                final_elbo = result.final_elbo,
                iterations = result.iterations as i32,
                converged = result.converged,
                param_names = param_names,
                method = "full_rank",
                dim = dim as i32
            )
        }
        _ => {
            panic!("Unknown method: {}. Use 'mean_field' or 'full_rank'", method);
        }
    }
}

// Macro to generate exports
extendr_module! {
    mod bayesiangpu;
    fn normal_dist;
    fn half_normal_dist;
    fn half_cauchy_dist;
    fn beta_dist;
    fn gamma_dist;
    fn uniform_dist;
    fn exponential_dist;
    fn cauchy_dist;
    fn student_t_dist;
    fn log_normal_dist;
    fn bernoulli_dist;
    fn binomial_dist;
    fn poisson_dist;
    fn multivariate_normal_dist;
    fn dirichlet_dist;
    fn multinomial_dist;
    fn laplace_dist;
    fn logistic_dist;
    fn inverse_gamma_dist;
    fn chi_squared_dist;
    fn truncated_normal_dist;
    fn run_nuts_sampling;
    fn run_advi;
}
