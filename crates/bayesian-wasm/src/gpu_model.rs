//! Standalone GPU model dispatch functions.
//!
//! These functions provide the GPU-accelerated logp + gradient computation
//! without requiring the full DynamicModel struct. They are used by Python
//! and R bindings to implement `logp_and_grad_direct()`.

use std::collections::HashMap;

use serde_json::Value;

use crate::gpu::kernels::GpuLikelihoodResult;
use crate::gpu::sync::GpuContextSync;
use crate::gpu::PersistentGpuBuffers;

/// Minimum data size to use GPU (below this, CPU is faster due to dispatch overhead)
pub const GPU_THRESHOLD: usize = 256;

/// Check if a distribution + data size qualifies for GPU dispatch
pub fn can_use_gpu(dist_type: &str, n_obs: usize) -> bool {
    if n_obs < GPU_THRESHOLD {
        return false;
    }
    matches!(
        dist_type,
        "Normal"
            | "HalfNormal"
            | "HalfCauchy"
            | "Exponential"
            | "Gamma"
            | "Beta"
            | "InverseGamma"
            | "StudentT"
            | "Cauchy"
            | "LogNormal"
            | "Bernoulli"
            | "Binomial"
            | "Poisson"
            | "Laplace"
            | "Logistic"
            | "ChiSquared"
            | "TruncatedNormal"
            | "Weibull"
            | "Pareto"
            | "Gumbel"
            | "HalfStudentT"
            | "NegativeBinomial"
            | "Categorical"
            | "Geometric"
            | "DiscreteUniform"
            | "BetaBinomial"
            | "ZeroInflatedPoisson"
            | "ZeroInflatedNegativeBinomial"
            | "Hypergeometric"
            | "OrderedLogistic"
    )
}

/// Lightweight model spec for GPU dispatch - contains just the fields needed
/// for parameter resolution and kernel selection.
#[derive(Clone)]
pub struct GpuModelSpec {
    /// Prior parameter names
    pub prior_names: Vec<String>,
    /// Offset of each prior in the flat parameter vector
    pub prior_offsets: Vec<usize>,
    /// Size of each prior (1 for scalar, N for vector)
    pub prior_sizes: Vec<usize>,
    /// Prior distribution specs: (dist_type, params HashMap)
    pub priors: Vec<(String, HashMap<String, Value>)>,
    /// Likelihood distribution type
    pub likelihood_dist_type: String,
    /// Likelihood distribution params
    pub likelihood_params: HashMap<String, Value>,
    /// Known data for hierarchical models
    pub known: HashMap<String, Vec<f64>>,
}

/// Compute total log_prob and gradient for all parameters using GPU.
///
/// This is the main entry point for Python/R bindings implementing `logp_and_grad_direct()`.
/// Returns (log_prob, gradients) where gradients.len() == params.len().
pub fn gpu_logp_and_grad(
    spec: &GpuModelSpec,
    params: &[f32],
    observed: &[f32],
    gpu_ctx: &GpuContextSync,
    buffers: &PersistentGpuBuffers,
) -> Result<(f64, Vec<f64>), String> {
    let dim: usize = spec.prior_sizes.iter().sum();
    let mut total_logp = 0.0_f64;
    let mut total_grad = vec![0.0_f64; dim];

    // 1. Prior contributions (CPU - per-parameter, not data-parallel)
    for (i, (dist_type, dist_params)) in spec.priors.iter().enumerate() {
        let offset = spec.prior_offsets[i];
        let size = spec.prior_sizes[i];
        for j in 0..size {
            let value = params[offset + j];
            let (lp, g) = compute_prior_logp_and_grad(
                value,
                dist_type,
                dist_params,
                &spec.prior_names,
                &spec.prior_offsets,
                params,
            );
            total_logp += lp;
            total_grad[offset + j] += g;
        }
    }

    // 2. Likelihood contribution (GPU)
    let result = compute_likelihood_gpu(
        &spec.prior_names,
        &spec.prior_offsets,
        &spec.prior_sizes,
        params,
        observed,
        &spec.likelihood_dist_type,
        &spec.likelihood_params,
        &spec.known,
        gpu_ctx,
        Some(buffers),
    )?;
    total_logp += result.log_prob;
    for (idx, grad) in result.param_grads {
        if idx < total_grad.len() {
            total_grad[idx] += grad;
        }
    }

    Ok((total_logp, total_grad))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get a parameter value as f64 from spec params
fn get_param_f64(params: &HashMap<String, Value>, key: &str, default: f64) -> f64 {
    params.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Resolve a parameter value and its index (if it references a model parameter)
fn resolve_param_value(
    prior_names: &[String],
    prior_offsets: &[usize],
    params: &[f32],
    spec_params: &HashMap<String, Value>,
    key: &str,
) -> (f32, Option<usize>) {
    let value = spec_params.get(key);
    match value {
        Some(Value::Number(n)) => {
            let val = n.as_f64().unwrap_or(0.0) as f32;
            (val, None)
        }
        Some(Value::String(s)) => {
            if let Some(idx) = prior_names.iter().position(|n| n == s) {
                let offset = prior_offsets[idx];
                (params[offset], Some(offset))
            } else {
                (0.0, None)
            }
        }
        _ => (0.0, None),
    }
}

/// Check if a likelihood parameter references a vector prior (size > 1).
/// Returns Some((prior_idx, offset, size)) if it does, None otherwise.
fn resolve_vector_param(
    prior_names: &[String],
    prior_offsets: &[usize],
    prior_sizes: &[usize],
    spec_params: &HashMap<String, Value>,
    key: &str,
) -> Option<(usize, usize, usize)> {
    if let Some(Value::String(s)) = spec_params.get(key) {
        if let Some(idx) = prior_names.iter().position(|n| n == s) {
            let size = prior_sizes[idx];
            if size > 1 {
                return Some((idx, prior_offsets[idx], size));
            }
        }
    }
    None
}

/// Resolve a prior parameter to an f32 value.
/// If the param is a string reference, look it up in the params array.
fn resolve_prior_param_f32(
    prior_names: &[String],
    prior_offsets: &[usize],
    spec_params: &HashMap<String, Value>,
    key: &str,
    default: f64,
    all_params: &[f32],
) -> f32 {
    match spec_params.get(key) {
        Some(Value::String(s)) => {
            if let Some(idx) = prior_names.iter().position(|n| n == s) {
                let offset = prior_offsets[idx];
                all_params[offset]
            } else {
                default as f32
            }
        }
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default) as f32,
        _ => default as f32,
    }
}

/// Build indexed kernel data for a vector parameter (hierarchical models).
fn build_group_index(
    known: &HashMap<String, Vec<f64>>,
    n_obs: usize,
    num_groups: usize,
) -> Option<(Vec<u32>, Vec<usize>, Vec<usize>)> {
    let group_data = known.get("group")?;
    if group_data.len() != n_obs {
        return None;
    }

    let group_idx: Vec<u32> = group_data.iter().map(|&g| g as u32).collect();

    let mut sort_order: Vec<usize> = (0..n_obs).collect();
    sort_order.sort_by_key(|&i| group_idx[i]);

    let mut group_boundaries = vec![0usize; num_groups + 1];
    for &idx in &sort_order {
        let g = group_idx[idx] as usize;
        if g < num_groups {
            group_boundaries[g + 1] += 1;
        }
    }
    for k in 1..=num_groups {
        group_boundaries[k] += group_boundaries[k - 1];
    }

    let sorted_group_idx: Vec<u32> = sort_order.iter().map(|&i| group_idx[i]).collect();

    Some((sorted_group_idx, sort_order, group_boundaries))
}

/// Log-gamma function using Lanczos approximation
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    let g = 7.0;
    let c = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0;
    let mut sum = c[0];
    for (i, &coeff) in c.iter().enumerate().skip(1) {
        sum += coeff / (x + i as f64);
    }

    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

// ============================================================================
// Prior Log-Probability and Gradient
// ============================================================================

/// Compute log_prior and its gradient for a single parameter value.
pub fn compute_prior_logp_and_grad(
    value: f32,
    dist_type: &str,
    dist_params: &HashMap<String, Value>,
    prior_names: &[String],
    prior_offsets: &[usize],
    all_params: &[f32],
) -> (f64, f64) {
    match dist_type {
        "Normal" => {
            let loc = resolve_prior_param_f32(
                prior_names,
                prior_offsets,
                dist_params,
                "loc",
                0.0,
                all_params,
            );
            let scale = resolve_prior_param_f32(
                prior_names,
                prior_offsets,
                dist_params,
                "scale",
                1.0,
                all_params,
            );
            let z = (value - loc) / scale;
            let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
            let log_prob = (log_norm - 0.5 * z * z) as f64;
            let grad = (-(value - loc) / (scale * scale)) as f64;
            (log_prob, grad)
        }
        "Beta" => {
            let alpha = get_param_f64(dist_params, "alpha", 1.0) as f32;
            let beta_param = get_param_f64(dist_params, "beta", 1.0) as f32;
            let x = value.clamp(1e-6, 1.0 - 1e-6);
            let log_beta = (ln_gamma(alpha as f64) + ln_gamma(beta_param as f64)
                - ln_gamma((alpha + beta_param) as f64)) as f32;
            let log_prob =
                ((alpha - 1.0) * x.ln() + (beta_param - 1.0) * (1.0 - x).ln() - log_beta) as f64;
            let grad = ((alpha - 1.0) / x - (beta_param - 1.0) / (1.0 - x)) as f64;
            (log_prob, grad)
        }
        "HalfNormal" => {
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let x = value.max(1e-10);
            let log_norm = (2.0f32).ln() - 0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
            let log_prob = (log_norm - 0.5 * (x / scale).powi(2)) as f64;
            let grad = (-x / (scale * scale)) as f64;
            (log_prob, grad)
        }
        "HalfCauchy" => {
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let x = value.max(1e-10);
            let log_norm = (2.0f32).ln() - std::f32::consts::PI.ln() - scale.ln();
            let log_prob = (log_norm - (1.0 + (x / scale).powi(2)).ln()) as f64;
            let grad = (-2.0 * x / (scale * scale + x * x)) as f64;
            (log_prob, grad)
        }
        "Gamma" => {
            let shape = get_param_f64(dist_params, "shape", 1.0) as f32;
            let rate = get_param_f64(dist_params, "rate", 1.0) as f32;
            let x = value.max(1e-10);
            let log_norm = shape * rate.ln() - ln_gamma(shape as f64) as f32;
            let log_prob = (log_norm + (shape - 1.0) * x.ln() - rate * x) as f64;
            let grad = ((shape - 1.0) / x - rate) as f64;
            (log_prob, grad)
        }
        "Exponential" => {
            let rate = get_param_f64(dist_params, "rate", 1.0) as f32;
            let x = value.max(1e-10);
            let log_prob = (rate.ln() - rate * x) as f64;
            let grad = (-rate) as f64;
            (log_prob, grad)
        }
        "Uniform" => {
            let low = get_param_f64(dist_params, "low", 0.0) as f32;
            let high = get_param_f64(dist_params, "high", 1.0) as f32;
            let log_prob = -((high - low).ln()) as f64;
            (log_prob, 0.0)
        }
        "Laplace" => {
            let loc = get_param_f64(dist_params, "loc", 0.0) as f32;
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let diff = value - loc;
            let log_prob = (-(2.0 * scale).ln() - diff.abs() / scale) as f64;
            let grad = if diff > 0.0 {
                -1.0 / scale
            } else if diff < 0.0 {
                1.0 / scale
            } else {
                0.0
            };
            (log_prob, grad as f64)
        }
        "Logistic" => {
            let loc = get_param_f64(dist_params, "loc", 0.0) as f32;
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let z = (value - loc) / scale;
            let exp_neg_z = (-z).exp();
            let log_prob = (-z - scale.ln() - 2.0 * (1.0 + exp_neg_z).ln()) as f64;
            let sig = 1.0 / (1.0 + exp_neg_z);
            let grad = ((1.0 - 2.0 * sig) / scale) as f64;
            (log_prob, grad)
        }
        "InverseGamma" => {
            let alpha = get_param_f64(dist_params, "alpha", 1.0) as f32;
            let beta_param = get_param_f64(dist_params, "beta", 1.0) as f32;
            let x = value.max(1e-10);
            let log_norm = alpha * beta_param.ln() - ln_gamma(alpha as f64) as f32;
            let log_prob = (log_norm - (alpha + 1.0) * x.ln() - beta_param / x) as f64;
            let grad = (-(alpha + 1.0) / x + beta_param / (x * x)) as f64;
            (log_prob, grad)
        }
        "ChiSquared" => {
            let k = get_param_f64(dist_params, "df", 1.0) as f32;
            let shape = k / 2.0;
            let rate = 0.5_f32;
            let x = value.max(1e-10);
            let log_norm = shape * rate.ln() - ln_gamma(shape as f64) as f32;
            let log_prob = (log_norm + (shape - 1.0) * x.ln() - rate * x) as f64;
            let grad = ((shape - 1.0) / x - rate) as f64;
            (log_prob, grad)
        }
        "TruncatedNormal" => {
            let loc = get_param_f64(dist_params, "loc", 0.0) as f32;
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let low = get_param_f64(dist_params, "low", f64::NEG_INFINITY) as f32;
            let high = get_param_f64(dist_params, "high", f64::INFINITY) as f32;
            if value < low || value > high {
                return (f64::NEG_INFINITY, 0.0);
            }
            let z = (value - loc) / scale;
            let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
            let log_prob_normal = (log_norm - 0.5 * z * z) as f64;
            fn normal_cdf_grad(x: f32) -> f32 {
                0.5 * (1.0 + erf_approx_grad(x / std::f32::consts::SQRT_2))
            }
            fn erf_approx_grad(x: f32) -> f32 {
                let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                let ax = x.abs();
                let t = 1.0 / (1.0 + 0.3275911 * ax);
                let poly = t
                    * (0.254_829_6_f32
                        + t * (-0.284_496_74_f32
                            + t * (1.421_413_7_f32 + t * (-1.453_152_f32 + t * 1.061_405_4_f32))));
                sign * (1.0 - poly * (-ax * ax).exp())
            }
            let cdf_high = normal_cdf_grad((high - loc) / scale);
            let cdf_low = normal_cdf_grad((low - loc) / scale);
            let log_z = (cdf_high - cdf_low).max(1e-10).ln();
            let log_prob = log_prob_normal - log_z as f64;
            let grad = (-(value - loc) / (scale * scale)) as f64;
            (log_prob, grad)
        }
        "Weibull" => {
            let k = get_param_f64(dist_params, "shape", 1.0) as f32;
            let lambda = get_param_f64(dist_params, "scale", 1.0) as f32;
            let x = value.max(1e-10);
            let x_over_lambda = x / lambda;
            let log_prob =
                ((k / lambda).ln() + (k - 1.0) * x_over_lambda.ln() - x_over_lambda.powf(k)) as f64;
            let grad = ((k - 1.0) / x - k * x_over_lambda.powf(k - 1.0) / lambda) as f64;
            (log_prob, grad)
        }
        "Pareto" => {
            let alpha = get_param_f64(dist_params, "alpha", 1.0) as f32;
            let x_m = get_param_f64(dist_params, "x_m", 1.0) as f32;
            if value < x_m {
                return (f64::NEG_INFINITY, 0.0);
            }
            let x = value.max(1e-10);
            let log_prob = (alpha.ln() + alpha * x_m.ln() - (alpha + 1.0) * x.ln()) as f64;
            let grad = (-(alpha + 1.0) / x) as f64;
            (log_prob, grad)
        }
        "Gumbel" => {
            let loc = get_param_f64(dist_params, "loc", 0.0) as f32;
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let z = (value - loc) / scale;
            let exp_neg_z = (-z).exp();
            let log_prob = (-z - exp_neg_z - scale.ln()) as f64;
            let grad = ((-1.0 + exp_neg_z) / scale) as f64;
            (log_prob, grad)
        }
        "HalfStudentT" => {
            let df = get_param_f64(dist_params, "df", 1.0) as f32;
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let x = value.max(1e-10);
            let log_norm = (2.0f32).ln() + ln_gamma(((df + 1.0) / 2.0) as f64) as f32
                - ln_gamma((df / 2.0) as f64) as f32
                - 0.5 * (df * std::f32::consts::PI).ln()
                - scale.ln();
            let z_sq = (x / scale).powi(2);
            let log_prob = (log_norm - ((df + 1.0) / 2.0) * (1.0 + z_sq / df).ln()) as f64;
            let grad = (-(df + 1.0) * x / (scale * scale * df + x * x)) as f64;
            (log_prob, grad)
        }
        "StudentT" => {
            let loc = get_param_f64(dist_params, "loc", 0.0) as f32;
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let df = get_param_f64(dist_params, "nu", 1.0) as f32;
            let z = (value - loc) / scale;
            let log_norm = ln_gamma(((df + 1.0) / 2.0) as f64) as f32
                - ln_gamma((df / 2.0) as f64) as f32
                - 0.5 * (df * std::f32::consts::PI).ln()
                - scale.ln();
            let log_prob = (log_norm - ((df + 1.0) / 2.0) * (1.0 + z * z / df).ln()) as f64;
            let grad = (-(df + 1.0) * z / (scale * (df + z * z))) as f64;
            (log_prob, grad)
        }
        "Cauchy" => {
            let loc = get_param_f64(dist_params, "loc", 0.0) as f32;
            let scale = get_param_f64(dist_params, "scale", 1.0) as f32;
            let z = (value - loc) / scale;
            let log_prob = (-(std::f32::consts::PI * scale).ln() - (1.0 + z * z).ln()) as f64;
            let grad = (-2.0 * z / (scale * (1.0 + z * z))) as f64;
            (log_prob, grad)
        }
        "LogNormal" => {
            let mu = get_param_f64(dist_params, "mu", 0.0) as f32;
            let sigma = get_param_f64(dist_params, "sigma", 1.0) as f32;
            let x = value.max(1e-10);
            let log_x = x.ln();
            let z = (log_x - mu) / sigma;
            let log_prob = (-log_x
                - 0.5 * (2.0 * std::f32::consts::PI * sigma * sigma).ln()
                - 0.5 * z * z) as f64;
            let grad = (-(1.0 + (log_x - mu) / (sigma * sigma)) / x) as f64;
            (log_prob, grad)
        }
        "NegativeBinomial" => {
            let r_param = get_param_f64(dist_params, "r", 1.0);
            let p = get_param_f64(dist_params, "p", 0.5);
            let k = value as f64;
            let log_prob = ln_gamma(k + r_param) - ln_gamma(k + 1.0) - ln_gamma(r_param)
                + r_param * p.ln()
                + k * (1.0 - p).ln();
            (log_prob, 0.0)
        }
        "Categorical" => (0.0, 0.0),
        "Geometric" => {
            let p = get_param_f64(dist_params, "p", 0.5);
            let k = value as f64;
            let log_prob = p.ln() + k * (1.0 - p).ln();
            (log_prob, 0.0)
        }
        "DiscreteUniform" => {
            let low = get_param_f64(dist_params, "low", 0.0);
            let high = get_param_f64(dist_params, "high", 10.0);
            let log_prob = -(high - low + 1.0).ln();
            (log_prob, 0.0)
        }
        "BetaBinomial" => {
            let n = get_param_f64(dist_params, "n", 10.0);
            let alpha = get_param_f64(dist_params, "alpha", 1.0);
            let beta_param = get_param_f64(dist_params, "beta", 1.0);
            let k = value as f64;
            let log_prob = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
                + ln_gamma(alpha + k)
                + ln_gamma(beta_param + n - k)
                - ln_gamma(alpha + beta_param + n)
                - ln_gamma(alpha)
                - ln_gamma(beta_param)
                + ln_gamma(alpha + beta_param);
            (log_prob, 0.0)
        }
        "ZeroInflatedPoisson" => {
            let rate = get_param_f64(dist_params, "rate", 1.0);
            let pi = get_param_f64(dist_params, "zero_prob", 0.0);
            let k = value as f64;
            let log_prob = if k == 0.0 {
                (pi + (1.0 - pi) * (-rate).exp()).ln()
            } else {
                (1.0 - pi).ln() + k * rate.ln() - rate - ln_gamma(k + 1.0)
            };
            (log_prob, 0.0)
        }
        "ZeroInflatedNegativeBinomial" => {
            let r_param = get_param_f64(dist_params, "r", 1.0);
            let p = get_param_f64(dist_params, "p", 0.5);
            let pi = get_param_f64(dist_params, "zero_prob", 0.0);
            let k = value as f64;
            let log_prob = if k == 0.0 {
                (pi + (1.0 - pi) * p.powf(r_param)).ln()
            } else {
                (1.0 - pi).ln() + ln_gamma(k + r_param) - ln_gamma(k + 1.0) - ln_gamma(r_param)
                    + r_param * p.ln()
                    + k * (1.0 - p).ln()
            };
            (log_prob, 0.0)
        }
        "Hypergeometric" => {
            let big_n = get_param_f64(dist_params, "big_n", 50.0);
            let big_k = get_param_f64(dist_params, "big_k", 25.0);
            let n = get_param_f64(dist_params, "n", 10.0);
            let k = value as f64;
            let ln_choose_val = |a: f64, b: f64| -> f64 {
                ln_gamma(a + 1.0) - ln_gamma(b + 1.0) - ln_gamma(a - b + 1.0)
            };
            let log_prob = ln_choose_val(big_k, k) + ln_choose_val(big_n - big_k, n - k)
                - ln_choose_val(big_n, n);
            (log_prob, 0.0)
        }
        "OrderedLogistic" => {
            let eta = get_param_f64(dist_params, "eta", 0.0);
            let cutpoints: Vec<f64> = dist_params
                .get("cutpoints")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            let j = value as usize;
            let n_cat = cutpoints.len() + 1;
            let prob = if j == 0 {
                if cutpoints.is_empty() {
                    1.0 / n_cat as f64
                } else {
                    1.0 / (1.0 + (-(cutpoints[0] - eta)).exp())
                }
            } else if j >= n_cat - 1 {
                if cutpoints.is_empty() {
                    1.0 / n_cat as f64
                } else {
                    let cum_prev = 1.0 / (1.0 + (-(cutpoints[cutpoints.len() - 1] - eta)).exp());
                    1.0 - cum_prev
                }
            } else {
                let cum_j = 1.0 / (1.0 + (-(cutpoints[j] - eta)).exp());
                let cum_j_minus_1 = 1.0 / (1.0 + (-(cutpoints[j - 1] - eta)).exp());
                cum_j - cum_j_minus_1
            };
            let log_prob = prob.max(1e-20).ln();
            (log_prob, 0.0)
        }
        _ => (0.0, 0.0),
    }
}

// ============================================================================
// Likelihood GPU Dispatch
// ============================================================================

/// Compute likelihood log_prob and gradients for ALL parameters using GPU.
///
/// Returns a `GpuLikelihoodResult` with log_prob and a vec of (param_index, gradient)
/// pairs for every distribution parameter that references a model parameter.
#[allow(clippy::too_many_arguments)]
pub fn compute_likelihood_gpu(
    prior_names: &[String],
    prior_offsets: &[usize],
    prior_sizes: &[usize],
    params: &[f32],
    observed: &[f32],
    dist_type: &str,
    dist_params: &HashMap<String, Value>,
    known: &HashMap<String, Vec<f64>>,
    gpu_ctx: &GpuContextSync,
    buffers: Option<&PersistentGpuBuffers>,
) -> Result<GpuLikelihoodResult, String> {
    match dist_type {
        "Normal" => {
            // Check if loc references a vector parameter (hierarchical model)
            if let Some((_prior_idx, vec_offset, vec_size)) =
                resolve_vector_param(prior_names, prior_offsets, prior_sizes, dist_params, "loc")
            {
                let (scale_val, scale_idx) =
                    resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
                let sigma = if scale_idx.is_some() {
                    scale_val
                } else {
                    scale_val.max(1e-6)
                };

                let theta: Vec<f32> = params[vec_offset..vec_offset + vec_size].to_vec();

                if let Some((sorted_group_idx, sort_order, group_boundaries)) =
                    build_group_index(known, observed.len(), vec_size)
                {
                    let y_sorted: Vec<f32> = sort_order.iter().map(|&i| observed[i]).collect();

                    let r = gpu_ctx.run_normal_indexed_reduce(
                        &y_sorted,
                        &theta,
                        &sorted_group_idx,
                        &group_boundaries,
                        sigma,
                    )?;

                    let mut grads = Vec::new();
                    for (k, &grad_k) in r.grad_theta.iter().enumerate() {
                        grads.push((vec_offset + k, grad_k));
                    }
                    if let Some(idx) = scale_idx {
                        grads.push((idx, r.grad_sigma));
                    }

                    return Ok(GpuLikelihoodResult {
                        log_prob: r.total_log_prob,
                        param_grads: grads,
                    });
                }
            }

            // Scalar loc path
            let (loc_val, loc_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "loc");
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
            let scale = if scale_idx.is_some() {
                scale_val
            } else {
                scale_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_normal_multi_grad_fused(b, loc_val, scale)?;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_normal_reduce(observed, loc_val, scale)? as f64;
                let gr = gpu_ctx.run_normal_grad_reduce(observed, loc_val, scale)? as f64;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, gr));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: grads,
                })
            }
        }
        "HalfNormal" => {
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");

            if let Some(b) = buffers {
                let r = gpu_ctx.run_half_normal_multi_grad_fused(b, scale_val)?;
                let mut grads = Vec::new();
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_half_normal_reduce(observed, scale_val)? as f64;
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: Vec::new(),
                })
            }
        }
        "Exponential" => {
            let (lambda_val, lambda_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "rate");

            if let Some(b) = buffers {
                let r = gpu_ctx.run_exponential_multi_grad_fused(b, lambda_val)?;
                let mut grads = Vec::new();
                if let Some(idx) = lambda_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_exponential_reduce(observed, lambda_val)? as f64;
                let gr = gpu_ctx.run_exponential_grad_reduce(observed, lambda_val)? as f64;
                let mut grads = Vec::new();
                if let Some(idx) = lambda_idx {
                    grads.push((idx, gr));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: grads,
                })
            }
        }
        "Gamma" => {
            let (alpha_val, alpha_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "shape");
            let (beta_val, beta_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "rate");
            let beta = if beta_idx.is_some() {
                beta_val
            } else {
                beta_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_gamma_multi_grad_fused(b, alpha_val, beta)?;
                let mut grads = Vec::new();
                if let Some(idx) = alpha_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = beta_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_gamma_reduce(observed, alpha_val, beta)? as f64;
                let gr = gpu_ctx.run_gamma_grad_reduce(observed, alpha_val, beta)? as f64;
                let mut grads = Vec::new();
                if let Some(idx) = alpha_idx {
                    grads.push((idx, gr));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: grads,
                })
            }
        }
        "Beta" => {
            let (alpha_val, alpha_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "alpha");
            let (beta_val, beta_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "beta");
            let beta = if beta_idx.is_some() {
                beta_val
            } else {
                beta_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_beta_multi_grad_fused(b, alpha_val, beta)?;
                let mut grads = Vec::new();
                if let Some(idx) = alpha_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = beta_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_beta_reduce(observed, alpha_val, beta)? as f64;
                let gr = gpu_ctx.run_beta_grad_reduce(observed, alpha_val, beta)? as f64;
                let mut grads = Vec::new();
                if let Some(idx) = alpha_idx {
                    grads.push((idx, gr));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: grads,
                })
            }
        }
        "InverseGamma" => {
            let (alpha_val, alpha_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "shape");
            let (beta_val, beta_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
            let beta = if beta_idx.is_some() {
                beta_val
            } else {
                beta_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_inverse_gamma_multi_grad_fused(b, alpha_val, beta)?;
                let mut grads = Vec::new();
                if let Some(idx) = alpha_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = beta_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_inverse_gamma_reduce(observed, alpha_val, beta)? as f64;
                let gr = gpu_ctx.run_inverse_gamma_grad_reduce(observed, alpha_val, beta)? as f64;
                let mut grads = Vec::new();
                if let Some(idx) = alpha_idx {
                    grads.push((idx, gr));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: grads,
                })
            }
        }
        "StudentT" => {
            let (loc_val, loc_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "loc");
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
            let (nu_val, nu_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "nu");
            let scale = if scale_idx.is_some() {
                scale_val
            } else {
                scale_val.max(1e-6)
            };
            let nu = if nu_idx.is_some() {
                nu_val
            } else {
                nu_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_student_t_multi_grad_fused(b, loc_val, scale, nu)?;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                if let Some(idx) = nu_idx {
                    grads.push((idx, r.total_grads[2] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_student_t_reduce(observed, nu_val, loc_val, scale)? as f64;
                let gr =
                    gpu_ctx.run_student_t_grad_reduce(observed, nu_val, loc_val, scale)? as f64;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, gr));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: grads,
                })
            }
        }
        "Cauchy" => {
            let (loc_val, loc_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "loc");
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
            let scale = if scale_idx.is_some() {
                scale_val
            } else {
                scale_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_cauchy_multi_grad_fused(b, loc_val, scale)?;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_cauchy_reduce(observed, loc_val, scale)? as f64;
                let gr = gpu_ctx.run_cauchy_grad_reduce(observed, loc_val, scale)? as f64;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, gr));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: grads,
                })
            }
        }
        "LogNormal" => {
            let (mu_val, mu_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "mu");
            let (sigma_val, sigma_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "sigma");
            let sigma = if sigma_idx.is_some() {
                sigma_val
            } else {
                sigma_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_lognormal_multi_grad_fused(b, mu_val, sigma)?;
                let mut grads = Vec::new();
                if let Some(idx) = mu_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = sigma_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_lognormal_reduce(observed, mu_val, sigma)? as f64;
                let gr = gpu_ctx.run_lognormal_grad_reduce(observed, mu_val, sigma)? as f64;
                let mut grads = Vec::new();
                if let Some(idx) = mu_idx {
                    grads.push((idx, gr));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: grads,
                })
            }
        }
        "Bernoulli" => {
            let (p_val, p_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "p");

            let log_prob = if let Some(b) = buffers {
                gpu_ctx.run_bernoulli_reduce_persistent(b, p_val)? as f64
            } else {
                gpu_ctx.run_bernoulli_reduce(observed, p_val)? as f64
            };
            let mut grad = 0.0f64;
            for &y in observed {
                if y == 1.0 {
                    grad += 1.0 / p_val as f64;
                } else {
                    grad -= 1.0 / (1.0 - p_val) as f64;
                }
            }
            let mut grads = Vec::new();
            if let Some(idx) = p_idx {
                grads.push((idx, grad));
            }
            Ok(GpuLikelihoodResult {
                log_prob,
                param_grads: grads,
            })
        }
        "Poisson" => {
            let (rate_val, rate_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "rate");

            let log_prob = if let Some(b) = buffers {
                gpu_ctx.run_poisson_reduce_persistent(b, rate_val)? as f64
            } else {
                gpu_ctx.run_poisson_reduce(observed, rate_val)? as f64
            };
            let mut grad = 0.0f64;
            for &y in observed {
                grad += y as f64 / rate_val as f64 - 1.0;
            }
            let mut grads = Vec::new();
            if let Some(idx) = rate_idx {
                grads.push((idx, grad));
            }
            Ok(GpuLikelihoodResult {
                log_prob,
                param_grads: grads,
            })
        }
        "Uniform" => {
            let (low_val, low_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "low");
            let (high_val, high_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "high");

            if let Some(b) = buffers {
                let r = gpu_ctx.run_uniform_multi_grad_fused(b, low_val, high_val)?;
                let mut grads = Vec::new();
                if let Some(idx) = low_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = high_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let lp = gpu_ctx.run_uniform_reduce(observed, low_val, high_val)? as f64;
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: Vec::new(),
                })
            }
        }
        "HalfCauchy" => {
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");

            if let Some(b) = buffers {
                let r = gpu_ctx.run_half_cauchy_multi_grad_fused(b, scale_val)?;
                let mut grads = Vec::new();
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let s = scale_val as f64;
                let lp: f64 = observed
                    .iter()
                    .map(|&x| {
                        let xf = x as f64;
                        if xf < 0.0 {
                            f64::NEG_INFINITY
                        } else {
                            (2.0 / (std::f64::consts::PI * s * (1.0 + (xf / s).powi(2)))).ln()
                        }
                    })
                    .sum();
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: Vec::new(),
                })
            }
        }
        "Laplace" => {
            let (loc_val, loc_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "loc");
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
            let scale = if scale_idx.is_some() {
                scale_val
            } else {
                scale_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_laplace_multi_grad_fused(b, loc_val, scale)?;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let mu = loc_val as f64;
                let s = scale as f64;
                let lp: f64 = observed
                    .iter()
                    .map(|&x| {
                        let xf = x as f64;
                        -(2.0 * s).ln() - (xf - mu).abs() / s
                    })
                    .sum();
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: Vec::new(),
                })
            }
        }
        "Logistic" => {
            let (loc_val, loc_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "loc");
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
            let scale = if scale_idx.is_some() {
                scale_val
            } else {
                scale_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_logistic_multi_grad_fused(b, loc_val, scale)?;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let mu = loc_val as f64;
                let s = scale as f64;
                let lp: f64 = observed
                    .iter()
                    .map(|&x| {
                        let z = (x as f64 - mu) / s;
                        -z - s.ln() - 2.0 * (1.0 + (-z).exp()).ln()
                    })
                    .sum();
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: Vec::new(),
                })
            }
        }
        "TruncatedNormal" => {
            let (loc_val, loc_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "loc");
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
            let scale = if scale_idx.is_some() {
                scale_val
            } else {
                scale_val.max(1e-6)
            };
            let low = get_param_f64(dist_params, "low", f64::NEG_INFINITY) as f32;
            let high = get_param_f64(dist_params, "high", f64::INFINITY) as f32;

            if let Some(b) = buffers {
                let r =
                    gpu_ctx.run_truncated_normal_multi_grad_fused(b, loc_val, scale, low, high)?;
                let mut grads = Vec::new();
                if let Some(idx) = loc_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let mu = loc_val as f64;
                let s = scale as f64;
                let lo = low as f64;
                let hi = high as f64;
                let cdf = |z: f64| -> f64 { 0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2)) };
                let log_norm = (cdf((hi - mu) / s) - cdf((lo - mu) / s)).max(1e-10).ln();
                let lp: f64 = observed
                    .iter()
                    .map(|&x| {
                        let xf = x as f64;
                        if xf < lo || xf > hi {
                            f64::NEG_INFINITY
                        } else {
                            -0.5 * (2.0 * std::f64::consts::PI).ln()
                                - s.ln()
                                - 0.5 * ((xf - mu) / s).powi(2)
                                - log_norm
                        }
                    })
                    .sum();
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: Vec::new(),
                })
            }
        }
        "Weibull" => {
            let (shape_val, shape_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "shape");
            let (scale_val, scale_idx) =
                resolve_param_value(prior_names, prior_offsets, params, dist_params, "scale");
            let scale = if scale_idx.is_some() {
                scale_val
            } else {
                scale_val.max(1e-6)
            };

            if let Some(b) = buffers {
                let r = gpu_ctx.run_weibull_multi_grad_fused(b, shape_val, scale)?;
                let mut grads = Vec::new();
                if let Some(idx) = shape_idx {
                    grads.push((idx, r.total_grads[0] as f64));
                }
                if let Some(idx) = scale_idx {
                    grads.push((idx, r.total_grads[1] as f64));
                }
                Ok(GpuLikelihoodResult {
                    log_prob: r.total_log_prob as f64,
                    param_grads: grads,
                })
            } else {
                let k = shape_val as f64;
                let lam = scale as f64;
                let lp: f64 = observed
                    .iter()
                    .map(|&x| {
                        let xf = x as f64;
                        if xf < 0.0 {
                            f64::NEG_INFINITY
                        } else {
                            k.ln() - k * lam.ln() + (k - 1.0) * xf.ln() - (xf / lam).powf(k)
                        }
                    })
                    .sum();
                Ok(GpuLikelihoodResult {
                    log_prob: lp,
                    param_grads: Vec::new(),
                })
            }
        }
        other => Err(format!("Unsupported distribution for GPU: {}", other)),
    }
}
