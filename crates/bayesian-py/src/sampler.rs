//! MCMC sampling for Python bindings
//!
//! This module implements the sampling logic that translates Python model specs
//! into Rust model implementations and runs NUTS inference.

use pyo3::prelude::*;
use std::collections::HashMap;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;

use bayesian_core::{
    ln_gamma, Beta, Distribution, Exponential, Gamma, HalfCauchy, HalfNormal, Normal, Support,
    Uniform,
};
use bayesian_diagnostics::{ess, rhat};
use bayesian_rng::GpuRng;
use bayesian_sampler::{
    AdviConfig, BayesianModel, FullRankAdvi, FullRankResult, MeanFieldAdvi, MeanFieldResult,
    NutsConfig, NutsResult, NutsSampler,
};

use crate::distributions::{ParamValue, PyDistribution};
use crate::model::{Likelihood, ModelSpec, PyModel};
use crate::result::{PyDiagnostics, PyInferenceResult};

// Type alias for our backend
type PyBackend = Autodiff<NdArray<f32>>;

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

    /// Create a tensor from an f64 value
    fn scalar_tensor(&self, value: f64) -> Tensor<PyBackend, 1> {
        Tensor::<PyBackend, 1>::from_floats([value as f32], &self.device)
    }

    /// Get a parameter value, either as a constant or from the parameter vector
    fn get_param_value(
        &self,
        value: &ParamValue,
        params: &Tensor<PyBackend, 1>,
    ) -> Tensor<PyBackend, 1> {
        match value {
            ParamValue::Number(n) => self.scalar_tensor(*n),
            ParamValue::Reference(name) => {
                // Find the parameter by prior name and use offset
                let idx = self
                    .spec
                    .priors
                    .iter()
                    .position(|p| &p.name == name)
                    .expect("Parameter reference not found");
                let offset = self.prior_offsets[idx];

                #[allow(clippy::single_range_in_vec_init)]
                params.clone().slice([offset..offset + 1])
            }
        }
    }

    /// Check if a distribution type has positive support (parameters stored in log space)
    fn is_positive_support_dist(dist_type: &str) -> bool {
        matches!(
            dist_type,
            "HalfCauchy"
                | "HalfNormal"
                | "Gamma"
                | "Exponential"
                | "LogNormal"
                | "InverseGamma"
                | "ChiSquared"
        )
    }

    /// Resolve a prior distribution parameter to a tensor value.
    /// If it's a number, return a scalar tensor.
    /// If it's a string reference, look up the named parameter in the full params tensor.
    /// For positive-support referenced parameters, apply exp() to get constrained value.
    fn resolve_prior_param(
        &self,
        value: &ParamValue,
        all_params: &Tensor<PyBackend, 1>,
    ) -> Tensor<PyBackend, 1> {
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

    /// Compute log probability for a prior distribution
    fn compute_prior_logp(
        &self,
        dist: &PyDistribution,
        value: &Tensor<PyBackend, 1>,
        all_params: &Tensor<PyBackend, 1>,
    ) -> Tensor<PyBackend, 1> {
        match dist.dist_type.as_str() {
            "Normal" => {
                // loc and scale can be parameter references
                let loc_pv = dist
                    .params
                    .get("loc")
                    .cloned()
                    .unwrap_or(ParamValue::Number(0.0));
                let scale_pv = dist
                    .params
                    .get("scale")
                    .cloned()
                    .unwrap_or(ParamValue::Number(1.0));
                let loc_tensor = self.resolve_prior_param(&loc_pv, all_params);
                let scale_tensor = self.resolve_prior_param(&scale_pv, all_params);
                let normal = Normal::<PyBackend>::new(loc_tensor, scale_tensor);
                normal.log_prob(value)
            }
            "HalfNormal" => {
                let scale = self.get_f64_param(&dist.params, "scale");
                let scale_tensor = self.scalar_tensor(scale);
                let half_normal = HalfNormal::<PyBackend>::new(scale_tensor);
                // Apply exp transform for positive constraint
                let transformed = value.clone().exp();
                let logp = half_normal.log_prob(&transformed);
                // Add Jacobian: log|d/dx exp(x)| = x
                logp + value.clone()
            }
            "Beta" => {
                let alpha = self.get_f64_param(&dist.params, "alpha");
                let beta_param = self.get_f64_param(&dist.params, "beta");
                let alpha_tensor = self.scalar_tensor(alpha);
                let beta_tensor = self.scalar_tensor(beta_param);
                let beta = Beta::<PyBackend>::new(alpha_tensor, beta_tensor);
                // Apply logit^{-1} transform for [0,1] constraint
                let transformed = sigmoid(value.clone());
                let logp = beta.log_prob(&transformed);
                // Add Jacobian: log|sigmoid'(x)| = log(sigmoid(x) * (1 - sigmoid(x)))
                let sig = sigmoid(value.clone());
                let one = Tensor::<PyBackend, 1>::ones_like(&sig);
                let jac = (sig.clone() * (one - sig.clone())).log();
                logp + jac
            }
            "Gamma" => {
                let shape = self.get_f64_param(&dist.params, "shape");
                let rate = self.get_f64_param(&dist.params, "rate");
                let shape_tensor = self.scalar_tensor(shape);
                let rate_tensor = self.scalar_tensor(rate);
                let gamma = Gamma::<PyBackend>::new(shape_tensor, rate_tensor);
                // Apply exp transform for positive constraint
                let transformed = value.clone().exp();
                let logp = gamma.log_prob(&transformed);
                logp + value.clone()
            }
            "Uniform" => {
                let low = self.get_f64_param(&dist.params, "low");
                let high = self.get_f64_param(&dist.params, "high");
                let low_tensor = self.scalar_tensor(low);
                let high_tensor = self.scalar_tensor(high);
                let uniform = Uniform::<PyBackend>::new(low_tensor, high_tensor);
                // Apply scaled logit^{-1} transform
                let range = high - low;
                let sig = sigmoid(value.clone());
                let transformed = sig.clone() * range as f32 + low as f32;
                let logp = uniform.log_prob(&transformed);
                // Jacobian for scaled sigmoid
                let one = Tensor::<PyBackend, 1>::ones_like(&sig);
                let jac = (sig.clone() * (one - sig) * range as f32).log();
                logp + jac
            }
            "Exponential" => {
                let rate = self.get_f64_param(&dist.params, "rate");
                let rate_tensor = self.scalar_tensor(rate);
                let exp_dist = Exponential::<PyBackend>::new(rate_tensor);
                let transformed = value.clone().exp();
                let logp = exp_dist.log_prob(&transformed);
                logp + value.clone()
            }
            "HalfCauchy" => {
                let scale = self.get_f64_param(&dist.params, "scale");
                let scale_tensor = self.scalar_tensor(scale);
                let half_cauchy = HalfCauchy::<PyBackend>::new(scale_tensor);
                // Apply exp transform for positive constraint
                let transformed = value.clone().exp();
                let logp = half_cauchy.log_prob(&transformed);
                // Add Jacobian: log|d/dx exp(x)| = x
                logp + value.clone()
            }
            "Laplace" => {
                let loc = self.get_f64_param(&dist.params, "loc");
                let scale = self.get_f64_param(&dist.params, "scale");
                // log Laplace(x|loc,scale) = -ln(2*scale) - |x-loc|/scale
                let loc_t = self.scalar_tensor(loc);
                let scale_t = self.scalar_tensor(scale);
                let log_norm = -((2.0 * scale) as f32).ln();
                let diff = value.clone() - loc_t;
                let abs_diff = diff.abs();
                (abs_diff / scale_t).neg().add_scalar(log_norm)
            }
            "Logistic" => {
                let loc = self.get_f64_param(&dist.params, "loc");
                let scale = self.get_f64_param(&dist.params, "scale");
                // log Logistic(x|loc,scale) = -(x-loc)/scale - log(scale) - 2*log(1 + exp(-(x-loc)/scale))
                let z = value
                    .clone()
                    .sub_scalar(loc as f32)
                    .div_scalar(scale as f32);
                let log_scale = (scale as f32).ln();
                // softplus(-z) = log(1 + exp(-z))
                let neg_z = z.clone().neg();
                let sp = neg_z.exp().add_scalar(1.0).log();
                z.neg().sub_scalar(log_scale) - sp.mul_scalar(2.0)
            }
            "InverseGamma" => {
                // Positive support - value is in log space
                let alpha = self.get_f64_param(&dist.params, "alpha");
                let beta_param = self.get_f64_param(&dist.params, "beta");
                // x = exp(v), ln(x) = v, 1/x = exp(-v)
                let log_norm = (alpha * beta_param.ln() - ln_gamma(alpha)) as f32;
                let log_x = value.clone(); // since x = exp(v), log(x) = v
                let inv_x = value.clone().neg().exp(); // 1/x = exp(-v)
                let logp = log_x
                    .mul_scalar(-(alpha as f32 + 1.0))
                    .sub(inv_x.mul_scalar(beta_param as f32))
                    .add_scalar(log_norm);
                // Jacobian: log|d/dv exp(v)| = v
                logp + value.clone()
            }
            "ChiSquared" => {
                // Positive support - value is in log space
                // ChiSquared(k) = Gamma(k/2, 1/2)
                let k = self.get_f64_param(&dist.params, "k");
                let shape = k / 2.0;
                let rate = 0.5_f64;
                let transformed = value.clone().exp();
                let log_norm = (shape * rate.ln() - ln_gamma(shape)) as f32;
                let log_x = value.clone();
                let logp = log_x
                    .mul_scalar((shape - 1.0) as f32)
                    .sub(transformed.mul_scalar(rate as f32))
                    .add_scalar(log_norm);
                // Jacobian
                logp + value.clone()
            }
            "TruncatedNormal" => {
                let loc = self.get_f64_param(&dist.params, "loc");
                let scale = self.get_f64_param(&dist.params, "scale");
                let low = self.get_f64_param(&dist.params, "low");
                let high = self.get_f64_param(&dist.params, "high");
                // TruncatedNormal log_prob = Normal.log_prob(x) - log(Phi((high-loc)/scale) - Phi((low-loc)/scale))
                let z = value
                    .clone()
                    .sub_scalar(loc as f32)
                    .div_scalar(scale as f32);
                let log_norm_const = -0.5 * (2.0 * std::f64::consts::PI).ln() - scale.ln();
                let logp_normal = z
                    .powf_scalar(2.0)
                    .mul_scalar(-0.5)
                    .add_scalar(log_norm_const as f32);
                // Compute log normalizing constant using f64
                fn normal_cdf_f64(x: f64) -> f64 {
                    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
                }
                fn erf_approx(x: f64) -> f64 {
                    // Abramowitz and Stegun approximation
                    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                    let x = x.abs();
                    let t = 1.0 / (1.0 + 0.3275911 * x);
                    let poly = t
                        * (0.254829592
                            + t * (-0.284496736
                                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
                    sign * (1.0 - poly * (-x * x).exp())
                }
                let cdf_high = normal_cdf_f64((high - loc) / scale);
                let cdf_low = normal_cdf_f64((low - loc) / scale);
                let log_z = ((cdf_high - cdf_low).max(1e-10)).ln() as f32;
                logp_normal - log_z
            }
            _ => {
                // Default: just return 0 (improper uniform)
                Tensor::<PyBackend, 1>::zeros([1], &self.device)
            }
        }
    }

    /// Resolve a likelihood distribution parameter for a specific observation index.
    ///
    /// Resolution order:
    /// 1. Number -> return that constant as a tensor
    /// 2. String matching a `known` data key -> return known[key][obs_idx] as tensor
    /// 3. String matching a vector parameter with matching size -> return params[offset + obs_idx]
    ///    (with exp() transform if positive-support prior)
    /// 4. String matching a scalar parameter -> return params[offset]
    ///    (with exp() transform if positive-support prior)
    /// 5. Fallback -> return 0.0
    fn resolve_for_observation(
        &self,
        value: &ParamValue,
        obs_idx: usize,
        params: &Tensor<PyBackend, 1>,
        known: &HashMap<String, Vec<f64>>,
    ) -> Tensor<PyBackend, 1> {
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
                    // Apply constraint transform if needed
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

    /// Compute log likelihood
    fn compute_likelihood_logp(
        &self,
        likelihood: &Likelihood,
        params: &Tensor<PyBackend, 1>,
    ) -> Tensor<PyBackend, 1> {
        let dist = &likelihood.distribution;
        let observed = &likelihood.observed;
        let known = &likelihood.known;

        match dist.dist_type.as_str() {
            "Bernoulli" => {
                let p =
                    self.get_param_value(dist.params.get("p").expect("Bernoulli needs p"), params);
                // Transform to [0,1] - p is in unconstrained space
                let p_constrained = sigmoid(p);

                // Bernoulli log likelihood: sum(y * log(p) + (1-y) * log(1-p))
                let mut total_logp = Tensor::<PyBackend, 1>::zeros([1], &self.device);
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
                let p =
                    self.get_param_value(dist.params.get("p").expect("Binomial needs p"), params);
                let p_constrained = sigmoid(p);

                // Binomial log likelihood (ignoring constant)
                let mut total_logp = Tensor::<PyBackend, 1>::zeros([1], &self.device);
                let one = self.scalar_tensor(1.0);
                for &k in observed {
                    let k_tensor = self.scalar_tensor(k);
                    let n_minus_k = self.scalar_tensor(n as f64 - k);
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
                let mut total_logp = Tensor::<PyBackend, 1>::zeros([1], &self.device);
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
                let rate = self
                    .get_param_value(dist.params.get("rate").expect("Poisson needs rate"), params);
                // Transform to positive
                let rate_constrained = rate.exp();

                // Poisson log likelihood (ignoring factorial constant)
                let mut total_logp = Tensor::<PyBackend, 1>::zeros([1], &self.device);
                for &k in observed {
                    let k_tensor = self.scalar_tensor(k);
                    let logp = k_tensor * rate_constrained.clone().log() - rate_constrained.clone();
                    total_logp = total_logp + logp;
                }
                total_logp
            }
            _ => Tensor::<PyBackend, 1>::zeros([1], &self.device),
        }
    }

    /// Get a parameter as f64 from the params map
    fn get_f64_param(&self, params: &HashMap<String, ParamValue>, key: &str) -> f64 {
        match params.get(key) {
            Some(ParamValue::Number(n)) => *n,
            _ => panic!("Expected numeric parameter for {}", key),
        }
    }

    /// Get support type for a distribution
    #[allow(dead_code)]
    fn get_support(&self, dist: &PyDistribution) -> Support {
        match dist.dist_type.as_str() {
            "Normal" | "StudentT" | "Cauchy" | "Laplace" | "Logistic" | "TruncatedNormal" => {
                Support::Real
            }
            "HalfNormal" | "HalfCauchy" | "Gamma" | "Exponential" | "LogNormal"
            | "InverseGamma" | "ChiSquared" => Support::Positive,
            "Beta" | "Uniform" => Support::UnitInterval,
            _ => Support::Real,
        }
    }
}

impl BayesianModel<PyBackend> for DynamicModel {
    fn dim(&self) -> usize {
        self.prior_sizes.iter().sum()
    }

    fn log_prob(&self, params: &Tensor<PyBackend, 1>) -> Tensor<PyBackend, 1> {
        let mut total_logp = Tensor::<PyBackend, 1>::zeros([1], &self.device);

        // Add prior log probabilities (offset-based for vector params)
        for (i, prior) in self.spec.priors.iter().enumerate() {
            let offset = self.prior_offsets[i];
            let size = self.prior_sizes[i];
            for j in 0..size {
                #[allow(clippy::single_range_in_vec_init)]
                let param_value = params.clone().slice([offset + j..offset + j + 1]);
                let logp = self.compute_prior_logp(&prior.distribution, &param_value, params);
                total_logp = total_logp + logp;
            }
        }

        // Add likelihood if present
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

/// Run NUTS sampling on a model
///
/// Args:
///     model: The model to sample from
///     num_samples: Number of samples to draw (after warmup)
///     num_warmup: Number of warmup iterations
///     num_chains: Number of parallel chains
///     target_accept: Target acceptance probability (0-1)
///     seed: Random seed for reproducibility
///
/// Returns:
///     InferenceResult with samples and diagnostics
///
/// Example:
///     >>> model = Model()
///     >>> model.param("theta", Beta(1, 1))
///     >>> model.observe(Binomial(100, "theta"), [65])
///     >>> result = sample(model, num_samples=1000, num_chains=4)
#[pyfunction]
#[pyo3(signature = (model, num_samples=1000, num_warmup=1000, num_chains=4, target_accept=0.8, seed=42))]
pub fn sample(
    model: &PyModel,
    num_samples: usize,
    num_warmup: usize,
    num_chains: usize,
    target_accept: f64,
    seed: u64,
) -> PyResult<PyInferenceResult> {
    // Parse the model spec
    let spec = model.spec().clone();

    // Validate model
    if spec.priors.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Model must have at least one parameter",
        ));
    }

    // Create dynamic model
    let dynamic_model = DynamicModel::new(spec);
    let dim = dynamic_model.dim();
    let param_names = dynamic_model.param_names();
    let device = NdArrayDevice::default();

    // Configure NUTS
    let config = NutsConfig {
        num_samples,
        num_warmup,
        max_tree_depth: 10,
        target_accept,
        init_step_size: 1.0,
    };

    // Run sampling for each chain
    let mut all_chain_samples: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for name in &param_names {
        all_chain_samples.insert(name.clone(), Vec::new());
    }

    let mut total_divergences = 0;
    let mut final_step_size = 0.0;

    for chain_idx in 0..num_chains {
        // Initialize from different starting points
        let chain_seed = seed + chain_idx as u64;
        let init_vals: Vec<f32> = (0..dim)
            .map(|i| {
                // Simple pseudo-random initialization
                let x = ((chain_seed
                    .wrapping_mul(1103515245)
                    .wrapping_add(i as u64 * 12345))
                    % 1000) as f32;
                (x / 500.0 - 1.0) * 0.1 // Small values around 0
            })
            .collect();
        let init = Tensor::<PyBackend, 1>::from_floats(init_vals.as_slice(), &device);

        // Create RNG for this chain
        let rng = GpuRng::<PyBackend>::new(chain_seed, dim.max(64), &device);

        // Create sampler with cloned model
        let mut sampler = NutsSampler::new(dynamic_model.clone(), config.clone(), rng);

        // Run sampling
        let result: NutsResult<PyBackend> = sampler.sample(init);

        // Collect samples
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

        total_divergences += result.divergences;
        final_step_size = result.final_step_size;
    }

    // Flatten samples across chains and compute diagnostics
    let mut flattened_samples: HashMap<String, Vec<f64>> = HashMap::new();
    let mut rhat_values: HashMap<String, f64> = HashMap::new();
    let mut ess_values: HashMap<String, f64> = HashMap::new();

    for name in &param_names {
        let chain_samples = all_chain_samples.get(name).unwrap();

        // Flatten
        let flat: Vec<f64> = chain_samples.iter().flatten().copied().collect();
        flattened_samples.insert(name.clone(), flat);

        // Compute diagnostics
        let rhat_val = rhat(chain_samples);
        let ess_val = ess(chain_samples);

        rhat_values.insert(name.clone(), rhat_val);
        ess_values.insert(name.clone(), ess_val);
    }

    // Create diagnostics
    let diagnostics = PyDiagnostics {
        rhat: rhat_values,
        ess: ess_values,
        divergences: total_divergences,
    };

    Ok(PyInferenceResult::new(
        flattened_samples,
        all_chain_samples,
        diagnostics,
        num_samples,
        num_warmup,
        num_chains,
        final_step_size,
    ))
}

/// Quick sampling with fewer iterations (good for testing)
///
/// Args:
///     model: The model to sample from
///     seed: Random seed
///
/// Returns:
///     InferenceResult
#[pyfunction]
#[pyo3(signature = (model, seed=42))]
pub fn quick_sample(model: &PyModel, seed: u64) -> PyResult<PyInferenceResult> {
    sample(model, 500, 500, 2, 0.8, seed)
}

/// Python wrapper for ADVI result
#[pyclass]
#[derive(Clone)]
pub struct PyAdviResult {
    /// Variational mean (posterior mean estimate)
    #[pyo3(get)]
    pub mu: Vec<f64>,
    /// Variational standard deviation
    #[pyo3(get)]
    pub sigma: Vec<f64>,
    /// ELBO values during optimization
    #[pyo3(get)]
    pub elbo_history: Vec<f64>,
    /// Final ELBO value
    #[pyo3(get)]
    pub final_elbo: f64,
    /// Number of iterations run
    #[pyo3(get)]
    pub iterations: usize,
    /// Whether optimization converged
    #[pyo3(get)]
    pub converged: bool,
    /// Parameter names
    #[pyo3(get)]
    pub param_names: Vec<String>,
    /// Full covariance (for full-rank, None for mean-field)
    covariance: Option<Vec<Vec<f64>>>,
    /// Method used (mean_field or full_rank)
    #[pyo3(get)]
    pub method: String,
}

#[pymethods]
impl PyAdviResult {
    /// Get the posterior covariance matrix
    ///
    /// For mean-field ADVI, this returns a diagonal matrix.
    /// For full-rank ADVI, this returns the full covariance.
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        if let Some(ref cov) = self.covariance {
            cov.clone()
        } else {
            // Mean-field: diagonal covariance
            let dim = self.sigma.len();
            let mut cov = vec![vec![0.0; dim]; dim];
            for (i, sigma_i) in self.sigma.iter().enumerate().take(dim) {
                cov[i][i] = sigma_i * sigma_i;
            }
            cov
        }
    }

    /// Sample from the variational posterior
    ///
    /// Args:
    ///     n_samples: Number of samples to draw
    ///     seed: Random seed
    ///
    /// Returns:
    ///     Dict mapping parameter names to sample arrays
    pub fn sample(&self, n_samples: usize, seed: u64) -> HashMap<String, Vec<f64>> {
        let dim = self.mu.len();
        let mut result: HashMap<String, Vec<f64>> = HashMap::new();

        for name in &self.param_names {
            result.insert(name.clone(), Vec::with_capacity(n_samples));
        }

        // Simple LCG for random number generation
        let mut state = seed;
        let mut next_normal = || -> f64 {
            // Box-Muller transform using LCG
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let u1 = (state as f64) / (u64::MAX as f64);
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let u2 = (state as f64) / (u64::MAX as f64);

            let u1 = u1.max(1e-10); // Avoid log(0)
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        if let Some(ref cov) = self.covariance {
            // Full-rank: sample from MVN using Cholesky
            // First compute Cholesky of covariance
            let mut l = vec![vec![0.0; dim]; dim];
            #[allow(clippy::needless_range_loop)]
            for i in 0..dim {
                for j in 0..=i {
                    let mut sum = 0.0;
                    if i == j {
                        for l_jk in l[j].iter().take(j) {
                            sum += l_jk * l_jk;
                        }
                        l[j][j] = (cov[j][j] - sum).sqrt();
                    } else {
                        for k in 0..j {
                            sum += l[i][k] * l[j][k];
                        }
                        l[i][j] = (cov[i][j] - sum) / l[j][j];
                    }
                }
            }

            for _ in 0..n_samples {
                let epsilon: Vec<f64> = (0..dim).map(|_| next_normal()).collect();

                for (i, name) in self.param_names.iter().enumerate() {
                    let mut sample_val = self.mu[i];
                    for j in 0..=i {
                        sample_val += l[i][j] * epsilon[j];
                    }
                    result.get_mut(name).unwrap().push(sample_val);
                }
            }
        } else {
            // Mean-field: independent sampling
            for _ in 0..n_samples {
                for (i, name) in self.param_names.iter().enumerate() {
                    let epsilon = next_normal();
                    let sample_val = self.mu[i] + self.sigma[i] * epsilon;
                    result.get_mut(name).unwrap().push(sample_val);
                }
            }
        }

        result
    }

    /// Print summary statistics
    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("ADVI Result ({} method)", self.method),
            format!("  Converged: {}", self.converged),
            format!("  Iterations: {}", self.iterations),
            format!("  Final ELBO: {:.4}", self.final_elbo),
            String::from("\nParameter estimates:"),
        ];

        for (i, name) in self.param_names.iter().enumerate() {
            lines.push(format!(
                "  {}: mean={:.4}, std={:.4}",
                name, self.mu[i], self.sigma[i]
            ));
        }

        lines.join("\n")
    }
}

/// Run ADVI (Automatic Differentiation Variational Inference) on a model
///
/// ADVI approximates the posterior with a Gaussian and optimizes the
/// Evidence Lower Bound (ELBO) using gradient descent.
///
/// Args:
///     model: The model to fit
///     method: Either "mean_field" (diagonal covariance) or "full_rank" (full covariance)
///     num_iterations: Maximum number of optimization iterations
///     num_samples: Number of Monte Carlo samples for gradient estimation
///     learning_rate: Learning rate for Adam optimizer
///     seed: Random seed for reproducibility
///
/// Returns:
///     AdviResult with posterior approximation and diagnostics
///
/// Example:
///     >>> model = Model()
///     >>> model.param("theta", Beta(1, 1))
///     >>> model.observe(Binomial(100, "theta"), [65])
///     >>> result = fit(model, method="mean_field")
///     >>> print(result.summary())
#[pyfunction]
#[pyo3(signature = (model, method="mean_field", num_iterations=10000, num_samples=1, learning_rate=0.01, seed=42))]
pub fn fit(
    model: &PyModel,
    method: &str,
    num_iterations: usize,
    num_samples: usize,
    learning_rate: f64,
    seed: u64,
) -> PyResult<PyAdviResult> {
    // Parse the model spec
    let spec = model.spec().clone();

    // Validate model
    if spec.priors.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Model must have at least one parameter",
        ));
    }

    // Create dynamic model
    let dynamic_model = DynamicModel::new(spec);
    let dim = dynamic_model.dim();
    let param_names = dynamic_model.param_names();
    let device = NdArrayDevice::default();

    // Configure ADVI
    let config = AdviConfig {
        num_iterations,
        num_samples,
        learning_rate,
        ..AdviConfig::default()
    };

    // Create RNG
    let rng = GpuRng::<PyBackend>::new(seed, dim.max(64), &device);

    match method {
        "mean_field" => {
            let mut advi = MeanFieldAdvi::new(dynamic_model, config, rng);
            let result: MeanFieldResult = advi.fit(None, None);

            Ok(PyAdviResult {
                mu: result.mu,
                sigma: result.sigma,
                elbo_history: result.elbo_history,
                final_elbo: result.final_elbo,
                iterations: result.iterations,
                converged: result.converged,
                param_names,
                covariance: None,
                method: "mean_field".to_string(),
            })
        }
        "full_rank" => {
            let mut advi = FullRankAdvi::new(dynamic_model, config, rng);
            let result: FullRankResult = advi.fit(None, None);

            // Compute covariance and std from scale_tril before moving
            let cov = result.covariance();
            let sigma = result.std();

            Ok(PyAdviResult {
                mu: result.mu,
                sigma,
                elbo_history: result.elbo_history,
                final_elbo: result.final_elbo,
                iterations: result.iterations,
                converged: result.converged,
                param_names,
                covariance: Some(cov),
                method: "full_rank".to_string(),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown method: {}. Use 'mean_field' or 'full_rank'",
            method
        ))),
    }
}
