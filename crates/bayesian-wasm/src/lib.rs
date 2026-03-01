//! WASM bindings for BayesianGPU
//!
//! This crate provides WebAssembly bindings to expose the BayesianGPU
//! library to JavaScript/TypeScript applications running in the browser.
//!
//! # Features
//!
//! - `ndarray` (default): CPU backend, works everywhere
//! - `wgpu`: WebGPU backend for GPU acceleration (requires async init)

// The Burn tensor `.slice([range])` API uses single-element range arrays as its standard pattern.
#![allow(clippy::single_range_in_vec_init)]

#[cfg(any(feature = "ndarray", feature = "wgpu"))]
use bayesian_diagnostics::{ess, rhat};
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
use bayesian_rng::GpuRng;
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
use bayesian_sampler::{BayesianModel, MultiChainConfig, MultiChainSampler, NutsConfig};
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
use burn::prelude::*;
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
use std::collections::HashMap;
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
use wasm_bindgen::prelude::*;

#[cfg(any(feature = "ndarray", feature = "wgpu"))]
pub mod backend;
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
use backend::get_device_or_init;
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
pub use backend::{get_backend_type, init_backend, is_webgpu_available, WasmBackend, WasmDevice};

// Direct wgpu compute module (bypasses Burn for reliable browser GPU)
#[cfg(feature = "direct-gpu")]
pub mod gpu;

// Standalone GPU model dispatch functions (used by Python/R bindings)
#[cfg(feature = "sync-gpu")]
pub mod gpu_model;

// When the `console_error_panic_hook` feature is enabled, we can call the
// `set_panic_hook` function at least once during initialization, and then
// we will get better error messages if our code ever panics.
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Initialize the WASM module
///
/// This should be called once when the module is first loaded.
/// It sets up panic hooks for better error messages in the browser console.
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
#[wasm_bindgen(start)]
pub fn init() {
    set_panic_hook();
}

/// Get the library version
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// WASM binding code — requires a Burn backend (ndarray or wgpu)
// ============================================================================
#[cfg(any(feature = "ndarray", feature = "wgpu"))]
mod wasm_impl {
    use super::*;

    // ============================================================================
    // Model Specification Types (from JavaScript)
    // ============================================================================

    /// Distribution specification from JavaScript
    #[derive(Debug, Clone, Deserialize)]
    pub struct DistributionSpec {
        #[serde(rename = "type")]
        pub dist_type: String,
        pub params: HashMap<String, serde_json::Value>,
    }

    /// Prior specification
    #[derive(Debug, Clone, Deserialize)]
    pub struct Prior {
        pub name: String,
        pub distribution: DistributionSpec,
        /// Number of elements for vector parameters (defaults to 1 for scalar)
        #[serde(default = "default_prior_size")]
        pub size: usize,
    }

    fn default_prior_size() -> usize {
        1
    }

    /// Likelihood specification
    #[derive(Debug, Clone, Deserialize)]
    pub struct Likelihood {
        pub distribution: DistributionSpec,
        pub observed: Vec<f64>,
        /// Per-observation known data (e.g., known standard deviations in Eight Schools)
        #[serde(default)]
        pub known: HashMap<String, Vec<f64>>,
    }

    /// Model specification from JavaScript
    #[derive(Debug, Clone, Deserialize)]
    pub struct ModelSpec {
        pub priors: Vec<Prior>,
        pub likelihood: Likelihood,
    }

    /// Inference configuration from JavaScript
    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct InferenceConfig {
        #[serde(default = "default_num_samples")]
        pub num_samples: usize,
        #[serde(default = "default_num_warmup")]
        pub num_warmup: usize,
        #[serde(default = "default_num_chains")]
        pub num_chains: usize,
        #[serde(default = "default_target_accept")]
        pub target_accept: f64,
        #[serde(default = "default_seed")]
        pub seed: u64,
    }

    fn default_num_samples() -> usize {
        1000
    }
    fn default_num_warmup() -> usize {
        1000
    }
    fn default_num_chains() -> usize {
        4
    }
    fn default_target_accept() -> f64 {
        0.8
    }
    fn default_seed() -> u64 {
        42
    }

    impl Default for InferenceConfig {
        fn default() -> Self {
            Self {
                num_samples: default_num_samples(),
                num_warmup: default_num_warmup(),
                num_chains: default_num_chains(),
                target_accept: default_target_accept(),
                seed: default_seed(),
            }
        }
    }

    // ============================================================================
    // Inference Output Types (to JavaScript)
    // ============================================================================

    /// Diagnostics output
    #[derive(Debug, Clone, Serialize)]
    pub struct DiagnosticsOutput {
        pub rhat: HashMap<String, f64>,
        pub ess: HashMap<String, f64>,
        pub divergences: usize,
    }

    /// Configuration output (what was actually used)
    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ConfigOutput {
        pub num_samples: usize,
        pub num_warmup: usize,
        pub num_chains: usize,
        pub step_size: f64,
    }

    /// Full inference output
    #[derive(Debug, Clone, Serialize)]
    pub struct InferenceOutput {
        pub samples: HashMap<String, Vec<f64>>,
        pub diagnostics: DiagnosticsOutput,
        pub config: ConfigOutput,
    }

    // ============================================================================
    // Dynamic Model Implementation
    // ============================================================================

    /// A dynamically-built Bayesian model from a JSON specification.
    ///
    /// This model supports a subset of distributions and simple likelihood models.
    /// Parameters can reference other parameters by name (e.g., "theta" in likelihood).
    #[derive(Clone)]
    struct DynamicModel {
        /// Parameter names in order (one per prior, not expanded)
        prior_names: Vec<String>,
        /// Prior distributions (stored as closures would be complex, so we store specs)
        priors: Vec<DistributionSpec>,
        /// Size of each prior (1 for scalar, N for vector)
        prior_sizes: Vec<usize>,
        /// Offset of each prior in the flat parameter vector
        prior_offsets: Vec<usize>,
        /// Expanded parameter names (e.g., theta[0], theta[1], ...)
        param_names: Vec<String>,
        /// Likelihood specification
        likelihood: Likelihood,
        /// Device for tensor creation
        device: WasmDevice,
        /// Cached f32 observation data (avoids Vec<f64> -> Vec<f32> conversion per call)
        #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
        observed_f32: Vec<f32>,
        /// Persistent GPU buffers for reduce kernels (allocated once, reused per call).
        /// Wrapped in Arc so DynamicModel can derive Clone (wgpu::Buffer is not Clone).
        /// Safe because chains run sequentially and buffers are only mutated via queue.write_buffer.
        #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
        gpu_buffers: Option<std::sync::Arc<gpu::PersistentGpuBuffers>>,
        /// GPU context for accelerated likelihood computation (sync, native only)
        #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
        gpu_ctx: Option<std::sync::Arc<gpu::sync::GpuContextSync>>,
        /// Multi-chain GPU buffers for batched dispatch across chains.
        /// Shares observation data (x_buffer) with per-chain param/staging buffers.
        #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
        #[allow(dead_code)]
        multi_chain_buffers: Option<std::sync::Arc<gpu::sync::MultiChainGpuBuffers>>,
    }

    impl DynamicModel {
        fn new(spec: ModelSpec, device: WasmDevice) -> Self {
            let prior_names: Vec<String> = spec.priors.iter().map(|p| p.name.clone()).collect();
            let priors: Vec<DistributionSpec> =
                spec.priors.iter().map(|p| p.distribution.clone()).collect();
            let prior_sizes: Vec<usize> = spec.priors.iter().map(|p| p.size.max(1)).collect();

            // Compute offsets
            let mut prior_offsets = Vec::with_capacity(prior_sizes.len());
            let mut offset = 0usize;
            for &sz in &prior_sizes {
                prior_offsets.push(offset);
                offset += sz;
            }

            // Expand parameter names
            let mut param_names = Vec::with_capacity(offset);
            for (name, &sz) in prior_names.iter().zip(&prior_sizes) {
                if sz == 1 {
                    param_names.push(name.clone());
                } else {
                    for i in 0..sz {
                        param_names.push(format!("{}[{}]", name, i));
                    }
                }
            }

            // Use the global shared GPU context for accelerated likelihood computation.
            // A single wgpu Device avoids Metal resource conflicts under parallel access.
            #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
            let gpu_ctx = gpu::sync::GpuContextSync::global();

            // Pre-compute f32 observation data for GPU path (avoids per-call conversion)
            #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
            let observed_f32: Vec<f32> =
                spec.likelihood.observed.iter().map(|&x| x as f32).collect();

            // Allocate persistent GPU buffers if GPU context is available
            // 32 bytes covers all param structs including StudentTReduceParams (8 x f32)
            #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
            let gpu_buffers = gpu_ctx
                .as_ref()
                .map(|ctx| std::sync::Arc::new(ctx.create_persistent_buffers(&observed_f32, 32)));

            Self {
                prior_names,
                priors,
                prior_sizes,
                prior_offsets,
                param_names,
                likelihood: spec.likelihood,
                device,
                #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
                observed_f32,
                #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
                gpu_buffers,
                #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
                gpu_ctx,
                // Multi-chain buffers are lazily initialized via init_multi_chain_buffers()
                #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
                multi_chain_buffers: None,
            }
        }

        /// Get a parameter value as f64 from spec
        fn get_param_f64(
            &self,
            params: &HashMap<String, serde_json::Value>,
            key: &str,
            default: f64,
        ) -> f64 {
            params.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        }
    }

    impl BayesianModel<WasmBackend> for DynamicModel {
        fn dim(&self) -> usize {
            self.prior_sizes.iter().sum()
        }

        fn log_prob(&self, params: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            // Start with zero log probability
            let mut log_prob = Tensor::<WasmBackend, 1>::zeros([1], &self.device);

            // Add log prior for each parameter using tensor operations (offset-based)
            for (idx, spec) in self.priors.iter().enumerate() {
                let offset = self.prior_offsets[idx];
                let size = self.prior_sizes[idx];
                for j in 0..size {
                    let param_val = params.clone().slice([offset + j..offset + j + 1]);
                    let log_prior = self.log_prior_tensor(&param_val, spec, params);
                    log_prob = log_prob.add(log_prior);
                }
            }

            // Add log likelihood using tensor operations
            let log_lik = self.log_likelihood_tensor(params);
            log_prob = log_prob.add(log_lik);

            log_prob
        }

        fn param_names(&self) -> Vec<String> {
            self.param_names.clone()
        }

        /// GPU-accelerated log_prob + gradient computation
        ///
        /// When the model has a GPU context and the likelihood has enough observations
        /// (>= 256), this bypasses Burn's autodiff entirely:
        /// - Priors: analytical log_prob + gradient on CPU (few parameters)
        /// - Likelihood: GPU REDUCE kernels for log_prob + gradient (many observations)
        ///
        /// Falls back to None (autodiff) when GPU is unavailable or data is too small.
        #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
        fn logp_and_grad_direct(&self, params: &[f32]) -> Option<(f64, Vec<f64>)> {
            if let Some(ref gpu_ctx) = self.gpu_ctx {
                if self.can_use_gpu() {
                    // Use persistent buffers for the fast path (same device as internal gpu_ctx)
                    let observed_f32 = &self.observed_f32;
                    let spec = &self.likelihood.distribution;
                    let buffers = self.gpu_buffers.as_deref();

                    let mut total_log_prob: f64 = 0.0;
                    let mut gradients: Vec<f64> = vec![0.0; params.len()];

                    for (idx, prior_spec) in self.priors.iter().enumerate() {
                        let offset = self.prior_offsets[idx];
                        let size = self.prior_sizes[idx];
                        for j in 0..size {
                            let param_val = params[offset + j];
                            let (prior_logp, prior_grad) =
                                self.compute_prior_logp_and_grad(param_val, prior_spec, params);
                            total_log_prob += prior_logp;
                            gradients[offset + j] += prior_grad;
                        }
                    }

                    match self.compute_likelihood_gpu(
                        params,
                        observed_f32,
                        spec,
                        gpu_ctx.as_ref(),
                        buffers,
                    ) {
                        Ok(result) => {
                            total_log_prob += result.log_prob;
                            for (idx, grad) in result.param_grads {
                                gradients[idx] += grad;
                            }
                            return Some((total_log_prob, gradients));
                        }
                        Err(_) => return None,
                    }
                }
            }
            None
        }

        #[cfg(not(all(feature = "direct-gpu", feature = "sync-gpu")))]
        fn logp_and_grad_direct(&self, _params: &[f32]) -> Option<(f64, Vec<f64>)> {
            None
        }
    }

    impl DynamicModel {
        /// Resolve a prior distribution parameter to a tensor.
        /// If the param value is a number, return a scalar tensor.
        /// If it's a string, look up the named parameter in `all_params` and return its value.
        /// For the WASM generic model, parameters are in constrained space, so no transform needed.
        fn resolve_prior_param_tensor(
            &self,
            spec_params: &HashMap<String, serde_json::Value>,
            key: &str,
            default: f64,
            _all_params: &Tensor<WasmBackend, 1>,
        ) -> Tensor<WasmBackend, 1> {
            match spec_params.get(key) {
                Some(serde_json::Value::String(s)) => {
                    // Resolve by looking up the named parameter
                    if let Some(idx) = self.prior_names.iter().position(|n| n == s) {
                        let offset = self.prior_offsets[idx];
                        _all_params.clone().slice([offset..offset + 1])
                    } else {
                        Tensor::<WasmBackend, 1>::from_floats([default as f32], &self.device)
                    }
                }
                Some(serde_json::Value::Number(n)) => {
                    let val = n.as_f64().unwrap_or(default) as f32;
                    Tensor::<WasmBackend, 1>::from_floats([val], &self.device)
                }
                _ => Tensor::<WasmBackend, 1>::from_floats([default as f32], &self.device),
            }
        }

        /// Check if a prior distribution parameter is a string reference
        fn is_param_ref(spec_params: &HashMap<String, serde_json::Value>, key: &str) -> bool {
            matches!(spec_params.get(key), Some(serde_json::Value::String(_)))
        }

        /// Compute log prior using tensor operations (autodiff-compatible)
        fn log_prior_tensor(
            &self,
            value: &Tensor<WasmBackend, 1>,
            spec: &DistributionSpec,
            all_params: &Tensor<WasmBackend, 1>,
        ) -> Tensor<WasmBackend, 1> {
            match spec.dist_type.as_str() {
                "Normal" => {
                    // For Normal, loc and scale can be parameter references
                    if Self::is_param_ref(&spec.params, "loc")
                        || Self::is_param_ref(&spec.params, "scale")
                    {
                        let loc =
                            self.resolve_prior_param_tensor(&spec.params, "loc", 0.0, all_params);
                        let scale =
                            self.resolve_prior_param_tensor(&spec.params, "scale", 1.0, all_params);
                        // log N(x | loc, scale) = -0.5 * log(2*pi*scale^2) - 0.5 * ((x-loc)/scale)^2
                        let z = (value.clone() - loc) / scale.clone();
                        let log_norm = scale
                            .clone()
                            .powf_scalar(2.0)
                            .mul_scalar(2.0 * std::f32::consts::PI)
                            .log()
                            .mul_scalar(-0.5);
                        z.powf_scalar(2.0).mul_scalar(-0.5) + log_norm
                    } else {
                        let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                        let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                        let z = value.clone().sub_scalar(loc).div_scalar(scale);
                        let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                        z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm)
                    }
                }
                "Beta" => {
                    let alpha = self.get_param_f64(&spec.params, "alpha", 1.0) as f32;
                    let beta_param = self.get_param_f64(&spec.params, "beta", 1.0) as f32;
                    // log Beta(x | a, b) = (a-1)*log(x) + (b-1)*log(1-x) - log(B(a,b))
                    // Support: x in (0, 1), return -inf outside
                    let log_beta = (ln_gamma(alpha as f64) + ln_gamma(beta_param as f64)
                        - ln_gamma((alpha + beta_param) as f64))
                        as f32;
                    // Clamp to (epsilon, 1-epsilon) for numerical stability
                    let x_clamped = value.clone().clamp(1e-6, 1.0 - 1e-6);
                    let log_x = x_clamped.clone().log();
                    let log_1mx = x_clamped.neg().add_scalar(1.0).log();
                    log_x
                        .mul_scalar(alpha - 1.0)
                        .add(log_1mx.mul_scalar(beta_param - 1.0))
                        .sub_scalar(log_beta)
                }
                "HalfNormal" => {
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log HalfNormal(x | scale) = log(2) - 0.5*log(2*pi*scale^2) - 0.5*(x/scale)^2
                    let z = value.clone().div_scalar(scale);
                    let log_norm =
                        (2.0f32).ln() - 0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                    z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm)
                }
                "HalfCauchy" => {
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log HalfCauchy(x | scale) = log(2) - log(pi) - log(scale) - log(1 + (x/scale)^2)
                    let log_norm = (2.0f32).ln() - std::f32::consts::PI.ln() - scale.ln();
                    let z = value.clone().div_scalar(scale);
                    z.powf_scalar(2.0)
                        .add_scalar(1.0)
                        .log()
                        .neg()
                        .add_scalar(log_norm)
                }
                "Uniform" => {
                    let low = self.get_param_f64(&spec.params, "low", 0.0) as f32;
                    let high = self.get_param_f64(&spec.params, "high", 1.0) as f32;
                    let log_density = -(high - low).ln();
                    Tensor::<WasmBackend, 1>::from_floats([log_density], &self.device)
                }
                "Gamma" => {
                    let shape = self.get_param_f64(&spec.params, "shape", 1.0) as f32;
                    let rate = self.get_param_f64(&spec.params, "rate", 1.0) as f32;
                    // log Gamma(x | shape, rate) = shape*log(rate) - ln_gamma(shape) + (shape-1)*log(x) - rate*x
                    let log_norm = shape * rate.ln() - ln_gamma(shape as f64) as f32;
                    let x_clamped = value.clone().clamp(1e-10, f32::MAX);
                    let log_x = x_clamped.clone().log();
                    log_x
                        .mul_scalar(shape - 1.0)
                        .sub(x_clamped.mul_scalar(rate))
                        .add_scalar(log_norm)
                }
                "Exponential" => {
                    let rate = self.get_param_f64(&spec.params, "rate", 1.0) as f32;
                    // log Exponential(x | rate) = log(rate) - rate*x
                    let log_rate = rate.ln();
                    value.clone().mul_scalar(-rate).add_scalar(log_rate)
                }
                "StudentT" => {
                    let df = self.get_param_f64(&spec.params, "df", 1.0) as f32;
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log StudentT(x | df, loc, scale) =
                    //   ln_gamma((df+1)/2) - ln_gamma(df/2) - 0.5*log(df*pi) - log(scale)
                    //   - ((df+1)/2)*log(1 + ((x-loc)/scale)^2/df)
                    let log_norm = (ln_gamma(((df + 1.0) / 2.0) as f64)
                        - ln_gamma((df / 2.0) as f64)
                        - 0.5 * (df as f64 * std::f64::consts::PI).ln()
                        - (scale as f64).ln()) as f32;
                    let z = value.clone().sub_scalar(loc).div_scalar(scale);
                    let z_sq_over_df = z.powf_scalar(2.0).div_scalar(df);
                    z_sq_over_df
                        .add_scalar(1.0)
                        .log()
                        .mul_scalar(-((df + 1.0) / 2.0))
                        .add_scalar(log_norm)
                }
                "Cauchy" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log Cauchy(x | loc, scale) = -log(pi) - log(scale) - log(1 + ((x-loc)/scale)^2)
                    let log_norm = -(std::f32::consts::PI).ln() - scale.ln();
                    let z = value.clone().sub_scalar(loc).div_scalar(scale);
                    z.powf_scalar(2.0)
                        .add_scalar(1.0)
                        .log()
                        .neg()
                        .add_scalar(log_norm)
                }
                "LogNormal" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log LogNormal(x | loc, scale) = -log(x) - log(scale) - 0.5*log(2*pi) - 0.5*((log(x)-loc)/scale)^2
                    let log_norm = -scale.ln() - 0.5 * (2.0 * std::f32::consts::PI).ln();
                    let x_clamped = value.clone().clamp(1e-10, f32::MAX);
                    let log_x = x_clamped.log();
                    let z = log_x.clone().sub_scalar(loc).div_scalar(scale);
                    z.powf_scalar(2.0)
                        .mul_scalar(-0.5)
                        .sub(log_x)
                        .add_scalar(log_norm)
                }
                "Laplace" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log Laplace(x|loc,scale) = -ln(2*scale) - |x-loc|/scale
                    let log_norm = -(2.0 * scale).ln();
                    value
                        .clone()
                        .sub_scalar(loc)
                        .abs()
                        .div_scalar(scale)
                        .neg()
                        .add_scalar(log_norm)
                }
                "Logistic" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let z = value.clone().sub_scalar(loc).div_scalar(scale);
                    let log_scale = scale.ln();
                    // log f = -z - log(s) - 2*log(1 + exp(-z))
                    let neg_z = z.clone().neg();
                    let softplus = neg_z.clone().exp().add_scalar(1.0).log();
                    z.neg().sub_scalar(log_scale) - softplus.mul_scalar(2.0)
                }
                "InverseGamma" => {
                    let alpha = self.get_param_f64(&spec.params, "alpha", 1.0) as f32;
                    let beta_param = self.get_param_f64(&spec.params, "beta", 1.0) as f32;
                    let x_clamped = value.clone().clamp(1e-10, f32::MAX);
                    let log_norm = alpha * beta_param.ln() - ln_gamma(alpha as f64) as f32;
                    let log_x = x_clamped.clone().log();
                    // -beta/x
                    let neg_beta_over_x = x_clamped.recip().mul_scalar(-beta_param);
                    log_x
                        .mul_scalar(-(alpha + 1.0))
                        .add(neg_beta_over_x)
                        .add_scalar(log_norm)
                }
                "ChiSquared" => {
                    let k = self.get_param_f64(&spec.params, "df", 1.0) as f32;
                    // ChiSquared(df) = Gamma(df/2, 1/2)
                    let shape = k / 2.0;
                    let rate = 0.5_f32;
                    let log_norm = shape * rate.ln() - ln_gamma(shape as f64) as f32;
                    let x_clamped = value.clone().clamp(1e-10, f32::MAX);
                    let log_x = x_clamped.clone().log();
                    log_x
                        .mul_scalar(shape - 1.0)
                        .sub(x_clamped.mul_scalar(rate))
                        .add_scalar(log_norm)
                }
                "TruncatedNormal" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let low = self.get_param_f64(&spec.params, "low", f64::NEG_INFINITY) as f32;
                    let high = self.get_param_f64(&spec.params, "high", f64::INFINITY) as f32;
                    // Enforce truncation bounds
                    let v = value.clone().into_scalar();
                    if v < low || v > high {
                        return Tensor::<WasmBackend, 1>::from_floats(
                            [f32::NEG_INFINITY],
                            &self.device,
                        );
                    }
                    // Normal log_prob
                    let z = value.clone().sub_scalar(loc).div_scalar(scale);
                    let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                    let logp = z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm);
                    // Subtract log normalizing constant (CDF difference)
                    fn normal_cdf_tn(x: f32) -> f32 {
                        0.5 * (1.0 + erf_approx_tn(x / std::f32::consts::SQRT_2))
                    }
                    fn erf_approx_tn(x: f32) -> f32 {
                        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                        let ax = x.abs();
                        let t = 1.0 / (1.0 + 0.3275911 * ax);
                        let poly = t
                            * (0.254_829_6_f32
                                + t * (-0.284_496_74_f32
                                    + t * (1.421_413_7_f32
                                        + t * (-1.453_152_f32 + t * 1.061_405_4_f32))));
                        sign * (1.0 - poly * (-ax * ax).exp())
                    }
                    let cdf_high = normal_cdf_tn((high - loc) / scale);
                    let cdf_low = normal_cdf_tn((low - loc) / scale);
                    let log_z = (cdf_high - cdf_low).max(1e-10).ln();
                    logp.sub_scalar(log_z)
                }
                "Weibull" => {
                    let k = self.get_param_f64(&spec.params, "shape", 1.0) as f32;
                    let lambda = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log f(x) = ln(k/lambda) + (k-1)*ln(x/lambda) - (x/lambda)^k
                    let log_norm = (k / lambda).ln();
                    let x_clamped = value.clone().clamp(1e-10, f32::MAX);
                    let x_over_lambda = x_clamped.div_scalar(lambda);
                    let log_x_over_lambda = x_over_lambda.clone().log();
                    log_x_over_lambda
                        .mul_scalar(k - 1.0)
                        .sub(x_over_lambda.powf_scalar(k))
                        .add_scalar(log_norm)
                }
                "Pareto" => {
                    let alpha = self.get_param_f64(&spec.params, "alpha", 1.0) as f32;
                    let x_m = self.get_param_f64(&spec.params, "x_m", 1.0) as f32;
                    // Enforce x >= x_m
                    let v = value.clone().into_scalar();
                    if v < x_m {
                        return Tensor::<WasmBackend, 1>::from_floats(
                            [f32::NEG_INFINITY],
                            &self.device,
                        );
                    }
                    // log f(x) = ln(alpha) + alpha*ln(x_m) - (alpha+1)*ln(x)
                    let log_norm = alpha.ln() + alpha * x_m.ln();
                    let x_clamped = value.clone().clamp(1e-10, f32::MAX);
                    x_clamped
                        .log()
                        .mul_scalar(-(alpha + 1.0))
                        .add_scalar(log_norm)
                }
                "Gumbel" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log f(x) = -z - exp(-z) - ln(s) where z = (x - loc) / scale
                    let z = value.clone().sub_scalar(loc).div_scalar(scale);
                    let neg_z = z.clone().neg();
                    let exp_neg_z = neg_z.clone().exp();
                    z.neg().sub(exp_neg_z).sub_scalar(scale.ln())
                }
                "HalfStudentT" => {
                    let df = self.get_param_f64(&spec.params, "df", 1.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    // log f(x) = ln(2) + ln_gamma((df+1)/2) - ln_gamma(df/2) - 0.5*ln(df*pi) - ln(scale)
                    //            - ((df+1)/2)*ln(1 + (x/scale)^2/df)
                    let log_norm = (2.0f32).ln() + ln_gamma(((df + 1.0) / 2.0) as f64) as f32
                        - ln_gamma((df / 2.0) as f64) as f32
                        - 0.5 * (df * std::f32::consts::PI).ln()
                        - scale.ln();
                    let x_clamped = value.clone().clamp(1e-10, f32::MAX);
                    let z = x_clamped.div_scalar(scale);
                    let z_sq_over_df = z.powf_scalar(2.0).div_scalar(df);
                    z_sq_over_df
                        .add_scalar(1.0)
                        .log()
                        .mul_scalar(-((df + 1.0) / 2.0))
                        .add_scalar(log_norm)
                }
                "NegativeBinomial" => {
                    // log f(k) = ln_gamma(k+r) - ln_gamma(k+1) - ln_gamma(r) + r*ln(p) + k*ln(1-p)
                    let r_param = self.get_param_f64(&spec.params, "r", 1.0);
                    let p = self.get_param_f64(&spec.params, "p", 0.5);
                    // value is the observation cast to float
                    let k_val = value.clone().into_scalar() as f64;
                    let log_prob =
                        ln_gamma(k_val + r_param) - ln_gamma(k_val + 1.0) - ln_gamma(r_param)
                            + r_param * p.ln()
                            + k_val * (1.0 - p).ln();
                    Tensor::<WasmBackend, 1>::from_floats([log_prob as f32], &self.device)
                }
                "Categorical" => {
                    // Categorical is only meaningful as a likelihood; return improper uniform
                    Tensor::<WasmBackend, 1>::zeros([1], &self.device)
                }
                "Geometric" => {
                    // log f(k) = ln(p) + k*ln(1-p)
                    let p = self.get_param_f64(&spec.params, "p", 0.5);
                    let k_val = value.clone().into_scalar() as f64;
                    let log_prob = p.ln() + k_val * (1.0 - p).ln();
                    Tensor::<WasmBackend, 1>::from_floats([log_prob as f32], &self.device)
                }
                "DiscreteUniform" => {
                    // log f(k) = -ln(high - low + 1)
                    let low = self.get_param_f64(&spec.params, "low", 0.0);
                    let high = self.get_param_f64(&spec.params, "high", 10.0);
                    let k_val = value.clone().into_scalar() as f64;
                    if k_val < low || k_val > high || k_val.fract() != 0.0 {
                        return Tensor::<WasmBackend, 1>::from_floats(
                            [f32::NEG_INFINITY],
                            &self.device,
                        );
                    }
                    let log_prob = -(high - low + 1.0).ln();
                    Tensor::<WasmBackend, 1>::from_floats([log_prob as f32], &self.device)
                }
                "BetaBinomial" => {
                    // log f(k) = ln_gamma(n+1) - ln_gamma(k+1) - ln_gamma(n-k+1)
                    //          + ln_gamma(alpha+k) + ln_gamma(beta+n-k) - ln_gamma(alpha+beta+n)
                    //          - ln_gamma(alpha) - ln_gamma(beta) + ln_gamma(alpha+beta)
                    let n = self.get_param_f64(&spec.params, "n", 10.0);
                    let alpha = self.get_param_f64(&spec.params, "alpha", 1.0);
                    let beta_param = self.get_param_f64(&spec.params, "beta", 1.0);
                    let k_val = value.clone().into_scalar() as f64;
                    let log_prob =
                        ln_gamma(n + 1.0) - ln_gamma(k_val + 1.0) - ln_gamma(n - k_val + 1.0)
                            + ln_gamma(alpha + k_val)
                            + ln_gamma(beta_param + n - k_val)
                            - ln_gamma(alpha + beta_param + n)
                            - ln_gamma(alpha)
                            - ln_gamma(beta_param)
                            + ln_gamma(alpha + beta_param);
                    Tensor::<WasmBackend, 1>::from_floats([log_prob as f32], &self.device)
                }
                "ZeroInflatedPoisson" => {
                    // ZI-Poisson: pi = zero_prob, rate = rate
                    // k==0: log(pi + (1-pi)*exp(-rate))
                    // k>0:  log(1-pi) + k*log(rate) - rate - ln_gamma(k+1)
                    let rate = self.get_param_f64(&spec.params, "rate", 1.0);
                    let pi = self.get_param_f64(&spec.params, "zero_prob", 0.0);
                    let k_val = value.clone().into_scalar() as f64;
                    if k_val < 0.0 || k_val.fract() != 0.0 {
                        return Tensor::<WasmBackend, 1>::from_floats(
                            [f32::NEG_INFINITY],
                            &self.device,
                        );
                    }
                    let log_prob = if k_val == 0.0 {
                        (pi + (1.0 - pi) * (-rate).exp()).ln()
                    } else {
                        (1.0 - pi).ln() + k_val * rate.ln() - rate - ln_gamma(k_val + 1.0)
                    };
                    Tensor::<WasmBackend, 1>::from_floats([log_prob as f32], &self.device)
                }
                "ZeroInflatedNegativeBinomial" => {
                    // ZI-NegBin: pi = zero_prob, r = r, p = p
                    // k==0: log(pi + (1-pi)*p^r)
                    // k>0:  log(1-pi) + ln_gamma(k+r) - ln_gamma(k+1) - ln_gamma(r) + r*log(p) + k*log(1-p)
                    let r_param = self.get_param_f64(&spec.params, "r", 1.0);
                    let p = self.get_param_f64(&spec.params, "p", 0.5);
                    let pi = self.get_param_f64(&spec.params, "zero_prob", 0.0);
                    let k_val = value.clone().into_scalar() as f64;
                    if k_val < 0.0 || k_val.fract() != 0.0 {
                        return Tensor::<WasmBackend, 1>::from_floats(
                            [f32::NEG_INFINITY],
                            &self.device,
                        );
                    }
                    let log_prob = if k_val == 0.0 {
                        (pi + (1.0 - pi) * p.powf(r_param)).ln()
                    } else {
                        (1.0 - pi).ln() + ln_gamma(k_val + r_param)
                            - ln_gamma(k_val + 1.0)
                            - ln_gamma(r_param)
                            + r_param * p.ln()
                            + k_val * (1.0 - p).ln()
                    };
                    Tensor::<WasmBackend, 1>::from_floats([log_prob as f32], &self.device)
                }
                "Hypergeometric" => {
                    // ln_choose(K,k) + ln_choose(N-K, n-k) - ln_choose(N, n)
                    // where ln_choose(a,b) = ln_gamma(a+1) - ln_gamma(b+1) - ln_gamma(a-b+1)
                    let big_n = self.get_param_f64(&spec.params, "big_n", 50.0);
                    let big_k = self.get_param_f64(&spec.params, "big_k", 25.0);
                    let n = self.get_param_f64(&spec.params, "n", 10.0);
                    let k_val = value.clone().into_scalar() as f64;
                    // Check support: k must be in [max(0, n-(N-K)), min(K, n)]
                    let k_min = (n - (big_n - big_k)).max(0.0);
                    let k_max = big_k.min(n);
                    if k_val < k_min || k_val > k_max || k_val.fract() != 0.0 {
                        return Tensor::<WasmBackend, 1>::from_floats(
                            [f32::NEG_INFINITY],
                            &self.device,
                        );
                    }
                    fn ln_choose(a: f64, b: f64) -> f64 {
                        ln_gamma(a + 1.0) - ln_gamma(b + 1.0) - ln_gamma(a - b + 1.0)
                    }
                    let log_prob = ln_choose(big_k, k_val) + ln_choose(big_n - big_k, n - k_val)
                        - ln_choose(big_n, n);
                    Tensor::<WasmBackend, 1>::from_floats([log_prob as f32], &self.device)
                }
                "OrderedLogistic" => {
                    // Ordered logistic: P(Y=j | eta, c) from cumulative logistic
                    // cutpoints stored as JSON array in "cutpoints" param
                    let eta = self.get_param_f64(&spec.params, "eta", 0.0);
                    let cutpoints: Vec<f64> = spec
                        .params
                        .get("cutpoints")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default();
                    let v = value.clone().into_scalar();
                    if (v as f64) < 0.0 {
                        return Tensor::<WasmBackend, 1>::from_floats(
                            [f32::NEG_INFINITY],
                            &self.device,
                        );
                    }
                    let j = v as usize;
                    let n_cat = cutpoints.len() + 1;
                    // Compute cumulative probs: P(Y <= j) = sigmoid(c_j - eta)
                    let prob = if j == 0 {
                        // P(Y=0) = sigmoid(c_0 - eta)
                        if cutpoints.is_empty() {
                            1.0 / n_cat as f64
                        } else {
                            1.0 / (1.0 + (-(cutpoints[0] - eta)).exp())
                        }
                    } else if j >= n_cat - 1 {
                        // P(Y=K) = 1 - sigmoid(c_{K-1} - eta)
                        if cutpoints.is_empty() {
                            1.0 / n_cat as f64
                        } else {
                            let cum_prev =
                                1.0 / (1.0 + (-(cutpoints[cutpoints.len() - 1] - eta)).exp());
                            1.0 - cum_prev
                        }
                    } else {
                        // P(Y=j) = sigmoid(c_j - eta) - sigmoid(c_{j-1} - eta)
                        let cum_j = 1.0 / (1.0 + (-(cutpoints[j] - eta)).exp());
                        let cum_j_minus_1 = 1.0 / (1.0 + (-(cutpoints[j - 1] - eta)).exp());
                        cum_j - cum_j_minus_1
                    };
                    let log_prob = prob.max(1e-20).ln();
                    Tensor::<WasmBackend, 1>::from_floats([log_prob as f32], &self.device)
                }
                "LKJCorr" => {
                    // LKJ Correlation distribution on the Cholesky factor of a correlation matrix.
                    // Parameters: dim (matrix dimension D), eta (concentration > 0).
                    // The prior has size = D*(D-1)/2 unconstrained off-diagonal elements of L.
                    //
                    // Since this function is called once per element in the prior's parameter
                    // vector, but the LKJ density is a joint density over all D*(D-1)/2
                    // elements, we compute the full density and divide by size so that the
                    // sum across all element-wise calls yields the correct total.
                    let dim = self.get_param_f64(&spec.params, "dim", 2.0) as usize;
                    let eta = self.get_param_f64(&spec.params, "eta", 1.0);
                    let n_tri = dim * (dim - 1) / 2;

                    if n_tri == 0 {
                        // 1x1 correlation matrix: log density is 0
                        return Tensor::<WasmBackend, 1>::zeros([1], &self.device);
                    }

                    // Find the offset for this prior in all_params
                    let prior_offset = self
                        .priors
                        .iter()
                        .enumerate()
                        .find(|(_, p)| std::ptr::eq(*p, spec))
                        .map(|(idx, _)| self.prior_offsets[idx])
                        .unwrap_or(0);

                    // Extract the full Cholesky off-diagonal slice from all_params
                    let z_slice = all_params
                        .clone()
                        .slice([prior_offset..prior_offset + n_tri]);
                    let z_data: Vec<f32> = (0..n_tri)
                        .map(|k| z_slice.clone().slice([k..k + 1]).into_scalar())
                        .collect();

                    // Reconstruct the Cholesky factor L and compute sum of log(L[i,i])
                    // L is DxD lower triangular with:
                    //   L[i][j] = z_data[k] for j < i (off-diagonal), k in row-major order
                    //   L[i][i] = sqrt(1 - sum_{j<i} L[i][j]^2)
                    let mut log_diag_sum = 0.0_f64;
                    let mut k = 0usize;
                    for i in 0..dim {
                        let mut row_sq_sum = 0.0_f64;
                        for _j in 0..i {
                            let lij = z_data[k] as f64;
                            row_sq_sum += lij * lij;
                            k += 1;
                        }
                        if i > 0 {
                            let diag_sq = (1.0 - row_sq_sum).max(1e-20);
                            let log_diag = 0.5 * diag_sq.ln(); // log(sqrt(1 - sum)) = 0.5*log(1 - sum)
                                                               // Weight: (D - i - 1) from Jacobian + 2*(eta - 1) from LKJ density
                            let weight = (dim as f64 - i as f64 - 1.0) + 2.0 * (eta - 1.0);
                            log_diag_sum += weight * log_diag;
                        }
                    }

                    // Normalizing constant:
                    // log Z = sum_{k=1}^{D-1} [ k * log(2) + 2 * lnbeta(eta + (D-1-k)/2, eta + (D-1-k)/2) ]
                    // where lnbeta(a, a) = 2*lngamma(a) - lngamma(2a)
                    // Alternative: sum over i=1..D-1 of (D-i-1+2*(eta-1))*E[log L[i,i]] but we need the constant.
                    // log Z = sum_{i=2}^{D} [ (i-1)*log(2) + 2*ln_gamma(eta + (i-2)/2) - ln_gamma(2*eta + i - 2) ]
                    let mut log_normalizer = 0.0_f64;
                    for i in 2..=dim {
                        let a = eta + (i as f64 - 2.0) / 2.0;
                        log_normalizer +=
                            (i as f64 - 1.0) * 2.0_f64.ln() + 2.0 * ln_gamma(a) - ln_gamma(2.0 * a);
                    }

                    let full_logp = (log_diag_sum - log_normalizer) as f32;
                    let logp_per_element = full_logp / n_tri as f32;
                    Tensor::<WasmBackend, 1>::from_floats([logp_per_element], &self.device)
                }
                _ => {
                    // Unknown distribution, return 0
                    Tensor::<WasmBackend, 1>::zeros([1], &self.device)
                }
            }
        }

        /// Compute log likelihood using tensor operations (autodiff-compatible)
        fn log_likelihood_tensor(&self, params: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            let spec = &self.likelihood.distribution;
            let observed = &self.likelihood.observed;

            match spec.dist_type.as_str() {
                "Bernoulli" => {
                    let mut log_lik = Tensor::<WasmBackend, 1>::zeros([1], &self.device);
                    for (j, &y) in observed.iter().enumerate() {
                        let p_raw = self.resolve_for_observation(
                            &spec.params,
                            "p",
                            j,
                            params,
                            &self.likelihood.known,
                        );
                        // Apply sigmoid to get probability
                        let p = 1.0 / (1.0 + (-p_raw).exp());
                        let p = p.clamp(1e-10, 1.0 - 1e-10);
                        let logp = if y == 1.0 { p.ln() } else { (1.0 - p).ln() };
                        log_lik = log_lik.add_scalar(logp);
                    }
                    log_lik
                }
                "Binomial" => {
                    let n = self.get_param_f64(&spec.params, "n", 1.0) as usize;
                    let p = self.resolve_param_tensor(params, &spec.params, "p");
                    // log Binomial(k | n, p) = log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
                    let log_p = p.clone().clamp(1e-10, 1.0 - 1e-10).log();
                    let log_1mp = p.neg().add_scalar(1.0).clamp(1e-10, 1.0 - 1e-10).log();

                    let mut log_lik = Tensor::<WasmBackend, 1>::zeros([1], &self.device);
                    for &y in observed {
                        let k = y as usize;
                        let log_binom_coef = (ln_gamma((n + 1) as f64)
                            - ln_gamma((k + 1) as f64)
                            - ln_gamma((n - k + 1) as f64))
                            as f32;
                        log_lik = log_lik
                            .add(log_p.clone().mul_scalar(k as f32))
                            .add(log_1mp.clone().mul_scalar((n - k) as f32))
                            .add_scalar(log_binom_coef);
                    }
                    log_lik
                }
                "Normal" => {
                    // Per-observation resolution: loc and scale can be vector params, known data, or scalars
                    let mut log_lik = Tensor::<WasmBackend, 1>::zeros([1], &self.device);
                    for (j, &y) in observed.iter().enumerate() {
                        let loc_j = self.resolve_for_observation(
                            &spec.params,
                            "loc",
                            j,
                            params,
                            &self.likelihood.known,
                        );
                        let scale_j = self.resolve_for_observation(
                            &spec.params,
                            "scale",
                            j,
                            params,
                            &self.likelihood.known,
                        );
                        // log N(y | loc, scale) = -0.5*log(2*pi*scale^2) - 0.5*((y-loc)/scale)^2
                        let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale_j * scale_j).ln();
                        let z = (y as f32 - loc_j) / scale_j;
                        log_lik = log_lik.add(Tensor::<WasmBackend, 1>::from_floats(
                            [-0.5 * z * z + log_norm],
                            &self.device,
                        ));
                    }
                    log_lik
                }
                "Poisson" => {
                    let rate = self.resolve_param_tensor(params, &spec.params, "rate");
                    // log Poisson(k | rate) = k*log(rate) - rate - ln_gamma(k+1)
                    let log_rate = rate.clone().clamp(1e-10, f32::MAX).log();

                    let mut log_lik = Tensor::<WasmBackend, 1>::zeros([1], &self.device);
                    for &y in observed {
                        let k = y as usize;
                        let log_factorial = ln_gamma((k + 1) as f64) as f32;
                        log_lik = log_lik
                            .add(log_rate.clone().mul_scalar(k as f32))
                            .sub(rate.clone())
                            .sub_scalar(log_factorial);
                    }
                    log_lik
                }
                _ => Tensor::<WasmBackend, 1>::zeros([1], &self.device),
            }
        }

        /// Resolve a likelihood distribution parameter for a specific observation index.
        ///
        /// Resolution order:
        /// 1. Number literal -> return that constant
        /// 2. String matching a `known` data key -> return known[key][obs_idx]
        /// 3. String matching a vector parameter with matching size -> return params[offset + obs_idx]
        /// 4. String matching a scalar parameter -> return params[offset]
        /// 5. Fallback -> return 0.0
        fn resolve_for_observation(
            &self,
            spec_params: &HashMap<String, serde_json::Value>,
            key: &str,
            obs_idx: usize,
            params: &Tensor<WasmBackend, 1>,
            known: &HashMap<String, Vec<f64>>,
        ) -> f32 {
            match spec_params.get(key) {
                Some(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(0.0) as f32,
                Some(serde_json::Value::String(s)) => {
                    // Check known data first
                    if let Some(known_vals) = known.get(s.as_str()) {
                        return known_vals.get(obs_idx).copied().unwrap_or(0.0) as f32;
                    }
                    // Check model parameters
                    if let Some(idx) = self.prior_names.iter().position(|n| n == s) {
                        let offset = self.prior_offsets[idx];
                        let size = self.prior_sizes[idx];
                        let param_offset = if size > 1 && obs_idx < size {
                            offset + obs_idx
                        } else {
                            offset
                        };
                        let val: Vec<f32> = params
                            .clone()
                            .slice([param_offset..param_offset + 1])
                            .into_data()
                            .to_vec()
                            .unwrap();
                        val[0]
                    } else {
                        0.0
                    }
                }
                Some(serde_json::Value::Object(obj)) => {
                    // Check for LinearPredictor
                    if obj.get("__type").and_then(|v| v.as_str()) == Some("LinearPredictor") {
                        let matrix_key = obj
                            .get("matrix_key")
                            .and_then(|v| v.as_str())
                            .expect("LinearPredictor missing matrix_key");
                        let param_name = obj
                            .get("param_name")
                            .and_then(|v| v.as_str())
                            .expect("LinearPredictor missing param_name");
                        let num_cols = obj
                            .get("num_cols")
                            .and_then(|v| v.as_u64())
                            .expect("LinearPredictor missing num_cols")
                            as usize;

                        let x_flat = known.get(matrix_key).unwrap_or_else(|| {
                            panic!("Design matrix '{}' not found in known data", matrix_key)
                        });
                        let row_start = obs_idx * num_cols;
                        let mut dot = 0.0f32;

                        if let Some(param_idx) =
                            self.prior_names.iter().position(|n| n == param_name)
                        {
                            let offset = self.prior_offsets[param_idx];
                            for j in 0..num_cols {
                                let x_val = x_flat[row_start + j] as f32;
                                let beta_vals: Vec<f32> = params
                                    .clone()
                                    .slice([offset + j..offset + j + 1])
                                    .into_data()
                                    .to_vec()
                                    .unwrap();
                                dot += x_val * beta_vals[0];
                            }
                        }
                        return dot;
                    }
                    0.0
                }
                _ => 0.0,
            }
        }

        /// Resolve a parameter to a tensor (either constant or from params)
        fn resolve_param_tensor(
            &self,
            params: &Tensor<WasmBackend, 1>,
            spec_params: &HashMap<String, serde_json::Value>,
            key: &str,
        ) -> Tensor<WasmBackend, 1> {
            let value = spec_params.get(key);
            match value {
                Some(serde_json::Value::Number(n)) => {
                    let val = n.as_f64().unwrap_or(0.0) as f32;
                    Tensor::<WasmBackend, 1>::from_floats([val], &self.device)
                }
                Some(serde_json::Value::String(s)) => {
                    // Find parameter by prior name and return that slice (using offset)
                    if let Some(idx) = self.prior_names.iter().position(|n| n == s) {
                        let offset = self.prior_offsets[idx];
                        params.clone().slice([offset..offset + 1])
                    } else {
                        Tensor::<WasmBackend, 1>::zeros([1], &self.device)
                    }
                }
                Some(serde_json::Value::Object(_obj)) => {
                    // LinearPredictor requires per-observation resolution
                    // Return zero as fallback for scalar resolution
                    Tensor::<WasmBackend, 1>::zeros([1], &self.device)
                }
                _ => Tensor::<WasmBackend, 1>::zeros([1], &self.device),
            }
        }
    }

    // ============================================================================
    // GPU-Accelerated Model Methods
    // ============================================================================

    #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
    impl DynamicModel {
        /// Minimum data size to use GPU acceleration (below this, CPU is faster)
        const GPU_THRESHOLD: usize = 256;

        /// Check if this model can use GPU acceleration for the likelihood
        pub fn can_use_gpu(&self) -> bool {
            let data_size = self.likelihood.observed.len();
            if data_size < Self::GPU_THRESHOLD {
                return false;
            }

            // Check if the likelihood distribution has a GPU kernel
            matches!(
                self.likelihood.distribution.dist_type.as_str(),
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

        /// Compute log probability and gradient using GPU for likelihood (sync version)
        ///
        /// This method computes:
        /// - Prior log_prob and gradients via autodiff (few values, not worth GPU)
        /// - Likelihood log_prob and gradients via GPU reduce kernels (many values)
        ///
        /// Returns (log_prob, gradient_vector) or falls back to autodiff if GPU unavailable.
        #[allow(dead_code)] // Used by tests with external GPU contexts
        pub fn logp_and_grad_gpu(
            &self,
            params: &[f32],
            gpu_ctx: &gpu::sync::GpuContextSync,
        ) -> Result<(f64, Vec<f64>), String> {
            let observed_f32 = &self.observed_f32;
            let spec = &self.likelihood.distribution;

            // Step 1: Compute prior log_prob and gradients
            // For each prior, compute log_prior(param) and d log_prior / d param
            let mut total_log_prob: f64 = 0.0;
            let mut gradients: Vec<f64> = vec![0.0; params.len()];

            for (idx, prior_spec) in self.priors.iter().enumerate() {
                let offset = self.prior_offsets[idx];
                let size = self.prior_sizes[idx];
                for j in 0..size {
                    let param_val = params[offset + j];
                    let (prior_logp, prior_grad) =
                        self.compute_prior_logp_and_grad(param_val, prior_spec, params);
                    total_log_prob += prior_logp;
                    gradients[offset + j] += prior_grad;
                }
            }

            // Step 2: Compute likelihood log_prob and gradient using GPU
            // Find which parameter is the "location" parameter for the likelihood
            // Note: persistent buffers are NOT used here because the caller may pass
            // a different GPU context than the one the buffers were created on.
            // Use logp_and_grad_direct() for the fast persistent-buffer path.
            let result = self.compute_likelihood_gpu(params, observed_f32, spec, gpu_ctx, None)?;

            total_log_prob += result.log_prob;
            for (idx, grad) in result.param_grads {
                gradients[idx] += grad;
            }

            Ok((total_log_prob, gradients))
        }

        /// Resolve a prior parameter to an f32 value (for the GPU fast-path).
        /// If the param is a string reference, look it up in the params array.
        fn resolve_prior_param_f32(
            &self,
            spec_params: &HashMap<String, serde_json::Value>,
            key: &str,
            default: f64,
            all_params: &[f32],
        ) -> f32 {
            match spec_params.get(key) {
                Some(serde_json::Value::String(s)) => {
                    if let Some(idx) = self.prior_names.iter().position(|n| n == s) {
                        let offset = self.prior_offsets[idx];
                        all_params[offset]
                    } else {
                        default as f32
                    }
                }
                Some(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(default) as f32,
                _ => default as f32,
            }
        }

        /// Compute log_prior and its gradient for a single parameter
        fn compute_prior_logp_and_grad(
            &self,
            value: f32,
            spec: &DistributionSpec,
            all_params: &[f32],
        ) -> (f64, f64) {
            match spec.dist_type.as_str() {
                "Normal" => {
                    let loc = self.resolve_prior_param_f32(&spec.params, "loc", 0.0, all_params);
                    let scale =
                        self.resolve_prior_param_f32(&spec.params, "scale", 1.0, all_params);
                    let z = (value - loc) / scale;
                    let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                    let log_prob = (log_norm - 0.5 * z * z) as f64;
                    let grad = (-(value - loc) / (scale * scale)) as f64;
                    (log_prob, grad)
                }
                "Beta" => {
                    let alpha = self.get_param_f64(&spec.params, "alpha", 1.0) as f32;
                    let beta_param = self.get_param_f64(&spec.params, "beta", 1.0) as f32;
                    let x = value.clamp(1e-6, 1.0 - 1e-6);
                    let log_beta = (ln_gamma(alpha as f64) + ln_gamma(beta_param as f64)
                        - ln_gamma((alpha + beta_param) as f64))
                        as f32;
                    let log_prob = ((alpha - 1.0) * x.ln() + (beta_param - 1.0) * (1.0 - x).ln()
                        - log_beta) as f64;
                    let grad = ((alpha - 1.0) / x - (beta_param - 1.0) / (1.0 - x)) as f64;
                    (log_prob, grad)
                }
                "HalfNormal" => {
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let x = value.max(1e-10);
                    let log_norm =
                        (2.0f32).ln() - 0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                    let log_prob = (log_norm - 0.5 * (x / scale).powi(2)) as f64;
                    let grad = (-x / (scale * scale)) as f64;
                    (log_prob, grad)
                }
                "HalfCauchy" => {
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let x = value.max(1e-10);
                    let log_norm = (2.0f32).ln() - std::f32::consts::PI.ln() - scale.ln();
                    let log_prob = (log_norm - (1.0 + (x / scale).powi(2)).ln()) as f64;
                    // d/dx log p = -2x / (scale^2 + x^2)
                    let grad = (-2.0 * x / (scale * scale + x * x)) as f64;
                    (log_prob, grad)
                }
                "Gamma" => {
                    let shape = self.get_param_f64(&spec.params, "shape", 1.0) as f32;
                    let rate = self.get_param_f64(&spec.params, "rate", 1.0) as f32;
                    let x = value.max(1e-10);
                    let log_norm = shape * rate.ln() - ln_gamma(shape as f64) as f32;
                    let log_prob = (log_norm + (shape - 1.0) * x.ln() - rate * x) as f64;
                    let grad = ((shape - 1.0) / x - rate) as f64;
                    (log_prob, grad)
                }
                "Exponential" => {
                    let rate = self.get_param_f64(&spec.params, "rate", 1.0) as f32;
                    let x = value.max(1e-10);
                    let log_prob = (rate.ln() - rate * x) as f64;
                    let grad = (-rate) as f64;
                    (log_prob, grad)
                }
                "Uniform" => {
                    let low = self.get_param_f64(&spec.params, "low", 0.0) as f32;
                    let high = self.get_param_f64(&spec.params, "high", 1.0) as f32;
                    let log_prob = -((high - low).ln()) as f64;
                    (log_prob, 0.0) // Uniform has zero gradient
                }
                "Laplace" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
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
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let z = (value - loc) / scale;
                    let exp_neg_z = (-z).exp();
                    let log_prob = (-z - scale.ln() - 2.0 * (1.0 + exp_neg_z).ln()) as f64;
                    // grad = (1 - 2*sigmoid(z)) / scale
                    let sig = 1.0 / (1.0 + exp_neg_z);
                    let grad = ((1.0 - 2.0 * sig) / scale) as f64;
                    (log_prob, grad)
                }
                "InverseGamma" => {
                    let alpha = self.get_param_f64(&spec.params, "alpha", 1.0) as f32;
                    let beta_param = self.get_param_f64(&spec.params, "beta", 1.0) as f32;
                    let x = value.max(1e-10);
                    let log_norm = alpha * beta_param.ln() - ln_gamma(alpha as f64) as f32;
                    let log_prob = (log_norm - (alpha + 1.0) * x.ln() - beta_param / x) as f64;
                    // d/dx [-(alpha+1)*ln(x) - beta/x] = -(alpha+1)/x + beta/x^2
                    let grad = (-(alpha + 1.0) / x + beta_param / (x * x)) as f64;
                    (log_prob, grad)
                }
                "ChiSquared" => {
                    let k = self.get_param_f64(&spec.params, "df", 1.0) as f32;
                    let shape = k / 2.0;
                    let rate = 0.5_f32;
                    let x = value.max(1e-10);
                    let log_norm = shape * rate.ln() - ln_gamma(shape as f64) as f32;
                    let log_prob = (log_norm + (shape - 1.0) * x.ln() - rate * x) as f64;
                    let grad = ((shape - 1.0) / x - rate) as f64;
                    (log_prob, grad)
                }
                "TruncatedNormal" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let low = self.get_param_f64(&spec.params, "low", f64::NEG_INFINITY) as f32;
                    let high = self.get_param_f64(&spec.params, "high", f64::INFINITY) as f32;
                    // Enforce truncation bounds
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
                                    + t * (1.421_413_7_f32
                                        + t * (-1.453_152_f32 + t * 1.061_405_4_f32))));
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
                    let k = self.get_param_f64(&spec.params, "shape", 1.0) as f32;
                    let lambda = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let x = value.max(1e-10);
                    let x_over_lambda = x / lambda;
                    let log_prob = ((k / lambda).ln() + (k - 1.0) * x_over_lambda.ln()
                        - x_over_lambda.powf(k)) as f64;
                    // d/dx = (k-1)/x - k*(x/lambda)^(k-1)/lambda
                    let grad = ((k - 1.0) / x - k * x_over_lambda.powf(k - 1.0) / lambda) as f64;
                    (log_prob, grad)
                }
                "Pareto" => {
                    let alpha = self.get_param_f64(&spec.params, "alpha", 1.0) as f32;
                    let x_m = self.get_param_f64(&spec.params, "x_m", 1.0) as f32;
                    // Enforce x >= x_m
                    if value < x_m {
                        return (f64::NEG_INFINITY, 0.0);
                    }
                    let x = value.max(1e-10);
                    let log_prob = (alpha.ln() + alpha * x_m.ln() - (alpha + 1.0) * x.ln()) as f64;
                    // d/dx = -(alpha+1)/x
                    let grad = (-(alpha + 1.0) / x) as f64;
                    (log_prob, grad)
                }
                "Gumbel" => {
                    let loc = self.get_param_f64(&spec.params, "loc", 0.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let z = (value - loc) / scale;
                    let exp_neg_z = (-z).exp();
                    let log_prob = (-z - exp_neg_z - scale.ln()) as f64;
                    // d/dx = (-1 + exp(-z)) / scale
                    let grad = ((-1.0 + exp_neg_z) / scale) as f64;
                    (log_prob, grad)
                }
                "HalfStudentT" => {
                    let df = self.get_param_f64(&spec.params, "df", 1.0) as f32;
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let x = value.max(1e-10);
                    let log_norm = (2.0f32).ln() + ln_gamma(((df + 1.0) / 2.0) as f64) as f32
                        - ln_gamma((df / 2.0) as f64) as f32
                        - 0.5 * (df * std::f32::consts::PI).ln()
                        - scale.ln();
                    let z_sq = (x / scale).powi(2);
                    let log_prob = (log_norm - ((df + 1.0) / 2.0) * (1.0 + z_sq / df).ln()) as f64;
                    // d/dx = -(df+1)*x / (scale^2*df + x^2)
                    let grad = (-(df + 1.0) * x / (scale * scale * df + x * x)) as f64;
                    (log_prob, grad)
                }
                "NegativeBinomial" => {
                    let r_param = self.get_param_f64(&spec.params, "r", 1.0);
                    let p = self.get_param_f64(&spec.params, "p", 0.5);
                    let k = value as f64;
                    let log_prob = ln_gamma(k + r_param) - ln_gamma(k + 1.0) - ln_gamma(r_param)
                        + r_param * p.ln()
                        + k * (1.0 - p).ln();
                    (log_prob, 0.0) // Discrete: gradient is 0
                }
                "Categorical" => {
                    (0.0, 0.0) // Not meaningful as prior
                }
                "Geometric" => {
                    let p = self.get_param_f64(&spec.params, "p", 0.5);
                    let k = value as f64;
                    let log_prob = p.ln() + k * (1.0 - p).ln();
                    (log_prob, 0.0) // Discrete: gradient is 0
                }
                "DiscreteUniform" => {
                    let low = self.get_param_f64(&spec.params, "low", 0.0);
                    let high = self.get_param_f64(&spec.params, "high", 10.0);
                    let log_prob = -(high - low + 1.0).ln();
                    (log_prob, 0.0) // Discrete: gradient is 0
                }
                "BetaBinomial" => {
                    let n = self.get_param_f64(&spec.params, "n", 10.0);
                    let alpha = self.get_param_f64(&spec.params, "alpha", 1.0);
                    let beta_param = self.get_param_f64(&spec.params, "beta", 1.0);
                    let k = value as f64;
                    let log_prob = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
                        + ln_gamma(alpha + k)
                        + ln_gamma(beta_param + n - k)
                        - ln_gamma(alpha + beta_param + n)
                        - ln_gamma(alpha)
                        - ln_gamma(beta_param)
                        + ln_gamma(alpha + beta_param);
                    (log_prob, 0.0) // Discrete: gradient is 0
                }
                "ZeroInflatedPoisson" => {
                    let rate = self.get_param_f64(&spec.params, "rate", 1.0);
                    let pi = self.get_param_f64(&spec.params, "zero_prob", 0.0);
                    let k = value as f64;
                    let log_prob = if k == 0.0 {
                        (pi + (1.0 - pi) * (-rate).exp()).ln()
                    } else {
                        (1.0 - pi).ln() + k * rate.ln() - rate - ln_gamma(k + 1.0)
                    };
                    (log_prob, 0.0) // Discrete: gradient is 0
                }
                "ZeroInflatedNegativeBinomial" => {
                    let r_param = self.get_param_f64(&spec.params, "r", 1.0);
                    let p = self.get_param_f64(&spec.params, "p", 0.5);
                    let pi = self.get_param_f64(&spec.params, "zero_prob", 0.0);
                    let k = value as f64;
                    let log_prob = if k == 0.0 {
                        (pi + (1.0 - pi) * p.powf(r_param)).ln()
                    } else {
                        (1.0 - pi).ln() + ln_gamma(k + r_param)
                            - ln_gamma(k + 1.0)
                            - ln_gamma(r_param)
                            + r_param * p.ln()
                            + k * (1.0 - p).ln()
                    };
                    (log_prob, 0.0) // Discrete: gradient is 0
                }
                "Hypergeometric" => {
                    let big_n = self.get_param_f64(&spec.params, "big_n", 50.0);
                    let big_k = self.get_param_f64(&spec.params, "big_k", 25.0);
                    let n = self.get_param_f64(&spec.params, "n", 10.0);
                    let k = value as f64;
                    let ln_choose_val = |a: f64, b: f64| -> f64 {
                        ln_gamma(a + 1.0) - ln_gamma(b + 1.0) - ln_gamma(a - b + 1.0)
                    };
                    let log_prob = ln_choose_val(big_k, k) + ln_choose_val(big_n - big_k, n - k)
                        - ln_choose_val(big_n, n);
                    (log_prob, 0.0) // Discrete: gradient is 0
                }
                "OrderedLogistic" => {
                    let eta = self.get_param_f64(&spec.params, "eta", 0.0);
                    let cutpoints: Vec<f64> = spec
                        .params
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
                            let cum_prev =
                                1.0 / (1.0 + (-(cutpoints[cutpoints.len() - 1] - eta)).exp());
                            1.0 - cum_prev
                        }
                    } else {
                        let cum_j = 1.0 / (1.0 + (-(cutpoints[j] - eta)).exp());
                        let cum_j_minus_1 = 1.0 / (1.0 + (-(cutpoints[j - 1] - eta)).exp());
                        cum_j - cum_j_minus_1
                    };
                    let log_prob = prob.max(1e-20).ln();
                    (log_prob, 0.0) // Discrete: gradient is 0
                }
                "LKJCorr" => {
                    // LKJ Correlation distribution (f32 fast path).
                    // Same joint-density-divided-by-size strategy as the tensor path.
                    // For the gradient, we compute d(log_density)/d(z_m) where z_m = value
                    // is one off-diagonal element of the Cholesky factor.
                    let dim = self.get_param_f64(&spec.params, "dim", 2.0) as usize;
                    let eta = self.get_param_f64(&spec.params, "eta", 1.0);
                    let n_tri = dim * (dim - 1) / 2;

                    if n_tri == 0 {
                        return (0.0, 0.0);
                    }

                    // Find offset for this prior
                    let prior_offset = self
                        .priors
                        .iter()
                        .enumerate()
                        .find(|(_, p)| std::ptr::eq(*p, spec))
                        .map(|(idx, _)| self.prior_offsets[idx])
                        .unwrap_or(0);

                    let z: Vec<f64> = (0..n_tri)
                        .map(|k| all_params[prior_offset + k] as f64)
                        .collect();

                    // Determine which element index m this value corresponds to
                    // (find which z[m] == value)
                    let m = (0..n_tri)
                        .find(|&k| (all_params[prior_offset + k] - value).abs() < 1e-30)
                        .unwrap_or(0);

                    // Reconstruct Cholesky factor and compute log density + gradient
                    // Build row-squared-sums and diagonal entries
                    let mut row_sq_sum = vec![0.0_f64; dim];
                    let mut diag_sq = vec![1.0_f64; dim]; // L[i,i]^2
                    let mut k = 0usize;
                    for i in 0..dim {
                        let mut sq_sum = 0.0_f64;
                        for _j in 0..i {
                            sq_sum += z[k] * z[k];
                            k += 1;
                        }
                        row_sq_sum[i] = sq_sum;
                        if i > 0 {
                            diag_sq[i] = (1.0 - sq_sum).max(1e-20);
                        }
                    }

                    // Compute full log density
                    let mut log_diag_sum = 0.0_f64;
                    for (i, &dsq) in diag_sq.iter().enumerate().take(dim).skip(1) {
                        let weight = (dim as f64 - i as f64 - 1.0) + 2.0 * (eta - 1.0);
                        log_diag_sum += weight * 0.5 * dsq.ln();
                    }

                    // Normalizing constant
                    let mut log_normalizer = 0.0_f64;
                    for i in 2..=dim {
                        let a = eta + (i as f64 - 2.0) / 2.0;
                        log_normalizer +=
                            (i as f64 - 1.0) * 2.0_f64.ln() + 2.0 * ln_gamma(a) - ln_gamma(2.0 * a);
                    }

                    let full_logp = log_diag_sum - log_normalizer;
                    let logp_per_element = full_logp / n_tri as f64;

                    // Gradient: d(log_density)/d(z_m) where z_m = L[row_m][col_m]
                    // log_density includes weight_i * 0.5 * ln(1 - sum_{j<i} z_{ij}^2) for the row
                    // containing z_m. The derivative is:
                    //   weight_row * 0.5 * (-2 * z_m) / (1 - row_sq_sum[row])
                    //   = -weight_row * z_m / diag_sq[row]
                    // We divide the gradient by n_tri to match the per-element log_prob.

                    // Find which row this element belongs to
                    let mut row_of_m = 0usize;
                    let mut cumulative = 0usize;
                    for i in 1..dim {
                        if m < cumulative + i {
                            row_of_m = i;
                            break;
                        }
                        cumulative += i;
                    }
                    if row_of_m == 0 && dim > 1 {
                        row_of_m = dim - 1; // last row
                    }

                    let weight_row = (dim as f64 - row_of_m as f64 - 1.0) + 2.0 * (eta - 1.0);
                    let z_m = value as f64;
                    let grad = -weight_row * z_m / diag_sq[row_of_m];
                    // Divide by n_tri to match per-element log_prob
                    let grad_per_element = grad / n_tri as f64;

                    (logp_per_element, grad_per_element)
                }
                _ => (0.0, 0.0),
            }
        }

        /// Compute likelihood log_prob and gradients for ALL parameters using GPU.
        ///
        /// Returns a `GpuLikelihoodResult` with log_prob and a vec of (param_index, gradient)
        /// pairs for every distribution parameter that references a model parameter.
        fn compute_likelihood_gpu(
            &self,
            params: &[f32],
            observed: &[f32],
            spec: &DistributionSpec,
            gpu_ctx: &gpu::sync::GpuContextSync,
            buffers: Option<&gpu::PersistentGpuBuffers>,
        ) -> Result<gpu::kernels::GpuLikelihoodResult, String> {
            match spec.dist_type.as_str() {
                "Normal" => {
                    // Check if loc references a vector parameter (hierarchical model)
                    if let Some((_prior_idx, vec_offset, vec_size)) =
                        self.resolve_vector_param(&spec.params, "loc")
                    {
                        // Indexed kernel path: y[i] ~ Normal(theta[group[i]], sigma)
                        let (scale_val, scale_idx) =
                            self.resolve_param_value(params, &spec.params, "scale");
                        let sigma = if scale_idx.is_some() {
                            scale_val
                        } else {
                            scale_val.max(1e-6)
                        };

                        // Extract theta values from params
                        let theta: Vec<f32> = params[vec_offset..vec_offset + vec_size].to_vec();

                        // Build group index and sort observations
                        if let Some((sorted_group_idx, sort_order, group_boundaries)) =
                            self.build_group_index(vec_size)
                        {
                            let y_sorted: Vec<f32> =
                                sort_order.iter().map(|&i| observed[i]).collect();

                            let r = gpu_ctx.run_normal_indexed_reduce(
                                &y_sorted,
                                &theta,
                                &sorted_group_idx,
                                &group_boundaries,
                                sigma,
                            )?;

                            let mut grads = Vec::new();
                            // Per-group theta gradients
                            for (k, &grad_k) in r.grad_theta.iter().enumerate() {
                                grads.push((vec_offset + k, grad_k));
                            }
                            // Sigma gradient
                            if let Some(idx) = scale_idx {
                                grads.push((idx, r.grad_sigma));
                            }

                            return Ok(gpu::kernels::GpuLikelihoodResult {
                                log_prob: r.total_log_prob,
                                param_grads: grads,
                            });
                        }
                        // Fall through to scalar path if no group index available
                    }

                    // Scalar loc path (original)
                    let (loc_val, loc_idx) = self.resolve_param_value(params, &spec.params, "loc");
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: grads,
                        })
                    }
                }
                "HalfNormal" => {
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");

                    if let Some(b) = buffers {
                        let r = gpu_ctx.run_half_normal_multi_grad_fused(b, scale_val)?;
                        let mut grads = Vec::new();
                        if let Some(idx) = scale_idx {
                            grads.push((idx, r.total_grads[0] as f64));
                        }
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        let lp = gpu_ctx.run_half_normal_reduce(observed, scale_val)? as f64;
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: Vec::new(),
                        })
                    }
                }
                "Exponential" => {
                    let (lambda_val, lambda_idx) =
                        self.resolve_param_value(params, &spec.params, "rate");

                    if let Some(b) = buffers {
                        let r = gpu_ctx.run_exponential_multi_grad_fused(b, lambda_val)?;
                        let mut grads = Vec::new();
                        if let Some(idx) = lambda_idx {
                            grads.push((idx, r.total_grads[0] as f64));
                        }
                        Ok(gpu::kernels::GpuLikelihoodResult {
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: grads,
                        })
                    }
                }
                "Gamma" => {
                    let (alpha_val, alpha_idx) =
                        self.resolve_param_value(params, &spec.params, "shape");
                    let (beta_val, beta_idx) =
                        self.resolve_param_value(params, &spec.params, "rate");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: grads,
                        })
                    }
                }
                "Beta" => {
                    let (alpha_val, alpha_idx) =
                        self.resolve_param_value(params, &spec.params, "alpha");
                    let (beta_val, beta_idx) =
                        self.resolve_param_value(params, &spec.params, "beta");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: grads,
                        })
                    }
                }
                "InverseGamma" => {
                    let (alpha_val, alpha_idx) =
                        self.resolve_param_value(params, &spec.params, "shape");
                    let (beta_val, beta_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        let lp =
                            gpu_ctx.run_inverse_gamma_reduce(observed, alpha_val, beta)? as f64;
                        let gr = gpu_ctx.run_inverse_gamma_grad_reduce(observed, alpha_val, beta)?
                            as f64;
                        let mut grads = Vec::new();
                        if let Some(idx) = alpha_idx {
                            grads.push((idx, gr));
                        }
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: grads,
                        })
                    }
                }
                "StudentT" => {
                    let (loc_val, loc_idx) = self.resolve_param_value(params, &spec.params, "loc");
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");
                    let (nu_val, nu_idx) = self.resolve_param_value(params, &spec.params, "nu");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        let lp =
                            gpu_ctx.run_student_t_reduce(observed, nu_val, loc_val, scale)? as f64;
                        let gr = gpu_ctx
                            .run_student_t_grad_reduce(observed, nu_val, loc_val, scale)?
                            as f64;
                        let mut grads = Vec::new();
                        if let Some(idx) = loc_idx {
                            grads.push((idx, gr));
                        }
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: grads,
                        })
                    }
                }
                "Cauchy" => {
                    let (loc_val, loc_idx) = self.resolve_param_value(params, &spec.params, "loc");
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: grads,
                        })
                    }
                }
                "LogNormal" => {
                    let (mu_val, mu_idx) = self.resolve_param_value(params, &spec.params, "mu");
                    let (sigma_val, sigma_idx) =
                        self.resolve_param_value(params, &spec.params, "sigma");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: grads,
                        })
                    }
                }
                "Bernoulli" => {
                    let (p_val, p_idx) = self.resolve_param_value(params, &spec.params, "p");

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
                    Ok(gpu::kernels::GpuLikelihoodResult {
                        log_prob,
                        param_grads: grads,
                    })
                }
                "Poisson" => {
                    let (rate_val, rate_idx) =
                        self.resolve_param_value(params, &spec.params, "rate");

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
                    Ok(gpu::kernels::GpuLikelihoodResult {
                        log_prob,
                        param_grads: grads,
                    })
                }
                "Uniform" => {
                    let (low_val, low_idx) = self.resolve_param_value(params, &spec.params, "low");
                    let (high_val, high_idx) =
                        self.resolve_param_value(params, &spec.params, "high");

                    if let Some(b) = buffers {
                        let r = gpu_ctx.run_uniform_multi_grad_fused(b, low_val, high_val)?;
                        let mut grads = Vec::new();
                        if let Some(idx) = low_idx {
                            grads.push((idx, r.total_grads[0] as f64));
                        }
                        if let Some(idx) = high_idx {
                            grads.push((idx, r.total_grads[1] as f64));
                        }
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        let lp = gpu_ctx.run_uniform_reduce(observed, low_val, high_val)? as f64;
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: Vec::new(),
                        })
                    }
                }
                "HalfCauchy" => {
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");

                    if let Some(b) = buffers {
                        let r = gpu_ctx.run_half_cauchy_multi_grad_fused(b, scale_val)?;
                        let mut grads = Vec::new();
                        if let Some(idx) = scale_idx {
                            grads.push((idx, r.total_grads[0] as f64));
                        }
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        // CPU fallback for HalfCauchy logp
                        let s = scale_val as f64;
                        let lp: f64 = observed
                            .iter()
                            .map(|&x| {
                                let xf = x as f64;
                                if xf < 0.0 {
                                    f64::NEG_INFINITY
                                } else {
                                    (2.0 / (std::f64::consts::PI * s * (1.0 + (xf / s).powi(2))))
                                        .ln()
                                }
                            })
                            .sum();
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: Vec::new(),
                        })
                    }
                }
                "Laplace" => {
                    let (loc_val, loc_idx) = self.resolve_param_value(params, &spec.params, "loc");
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        // CPU fallback for Laplace logp
                        let mu = loc_val as f64;
                        let s = scale as f64;
                        let lp: f64 = observed
                            .iter()
                            .map(|&x| {
                                let xf = x as f64;
                                -(2.0 * s).ln() - (xf - mu).abs() / s
                            })
                            .sum();
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: Vec::new(),
                        })
                    }
                }
                "Logistic" => {
                    let (loc_val, loc_idx) = self.resolve_param_value(params, &spec.params, "loc");
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        // CPU fallback for Logistic logp
                        let mu = loc_val as f64;
                        let s = scale as f64;
                        let lp: f64 = observed
                            .iter()
                            .map(|&x| {
                                let z = (x as f64 - mu) / s;
                                -z - s.ln() - 2.0 * (1.0 + (-z).exp()).ln()
                            })
                            .sum();
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: Vec::new(),
                        })
                    }
                }
                "TruncatedNormal" => {
                    let (loc_val, loc_idx) = self.resolve_param_value(params, &spec.params, "loc");
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");
                    let scale = if scale_idx.is_some() {
                        scale_val
                    } else {
                        scale_val.max(1e-6)
                    };
                    let low = self.get_param_f64(&spec.params, "low", f64::NEG_INFINITY) as f32;
                    let high = self.get_param_f64(&spec.params, "high", f64::INFINITY) as f32;

                    if let Some(b) = buffers {
                        let r = gpu_ctx
                            .run_truncated_normal_multi_grad_fused(b, loc_val, scale, low, high)?;
                        let mut grads = Vec::new();
                        if let Some(idx) = loc_idx {
                            grads.push((idx, r.total_grads[0] as f64));
                        }
                        if let Some(idx) = scale_idx {
                            grads.push((idx, r.total_grads[1] as f64));
                        }
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        // CPU fallback for TruncatedNormal logp
                        let mu = loc_val as f64;
                        let s = scale as f64;
                        let lo = low as f64;
                        let hi = high as f64;
                        let cdf = |z: f64| -> f64 {
                            0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2))
                        };
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: Vec::new(),
                        })
                    }
                }
                "Weibull" => {
                    let (shape_val, shape_idx) =
                        self.resolve_param_value(params, &spec.params, "shape");
                    let (scale_val, scale_idx) =
                        self.resolve_param_value(params, &spec.params, "scale");
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: r.total_log_prob as f64,
                            param_grads: grads,
                        })
                    } else {
                        // CPU fallback for Weibull logp
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
                        Ok(gpu::kernels::GpuLikelihoodResult {
                            log_prob: lp,
                            param_grads: Vec::new(),
                        })
                    }
                }
                other => Err(format!("Unsupported distribution for GPU: {}", other)),
            }
        }

        /// Resolve a parameter value and its index (if it references a model parameter)
        fn resolve_param_value(
            &self,
            params: &[f32],
            spec_params: &HashMap<String, serde_json::Value>,
            key: &str,
        ) -> (f32, Option<usize>) {
            let value = spec_params.get(key);
            match value {
                Some(serde_json::Value::Number(n)) => {
                    let val = n.as_f64().unwrap_or(0.0) as f32;
                    (val, None) // Constant value, no gradient
                }
                Some(serde_json::Value::String(s)) => {
                    // Find parameter by prior name and use offset
                    if let Some(idx) = self.prior_names.iter().position(|n| n == s) {
                        let offset = self.prior_offsets[idx];
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
            &self,
            spec_params: &HashMap<String, serde_json::Value>,
            key: &str,
        ) -> Option<(usize, usize, usize)> {
            if let Some(serde_json::Value::String(s)) = spec_params.get(key) {
                if let Some(idx) = self.prior_names.iter().position(|n| n == s) {
                    let size = self.prior_sizes[idx];
                    if size > 1 {
                        return Some((idx, self.prior_offsets[idx], size));
                    }
                }
            }
            None
        }

        /// Build indexed kernel data for a vector parameter.
        ///
        /// For hierarchical models where observations map to groups via a `known` field
        /// (e.g., "group" in likelihood.known), this builds:
        /// - group_idx: per-observation group assignment (u32)
        /// - sort_order: indices that sort observations by group
        /// - group_boundaries: start index of each group in the sorted array (K+1 values)
        fn build_group_index(
            &self,
            num_groups: usize,
        ) -> Option<(Vec<u32>, Vec<usize>, Vec<usize>)> {
            // Look for a "group" field in likelihood.known
            let group_data = self.likelihood.known.get("group")?;
            let n = self.likelihood.observed.len();
            if group_data.len() != n {
                return None;
            }

            let group_idx: Vec<u32> = group_data.iter().map(|&g| g as u32).collect();

            // Sort indices by group
            let mut sort_order: Vec<usize> = (0..n).collect();
            sort_order.sort_by_key(|&i| group_idx[i]);

            // Compute group boundaries in sorted order
            let mut group_boundaries = vec![0usize; num_groups + 1];
            for &idx in &sort_order {
                let g = group_idx[idx] as usize;
                if g < num_groups {
                    group_boundaries[g + 1] += 1;
                }
            }
            // Convert counts to cumulative offsets
            for k in 1..=num_groups {
                group_boundaries[k] += group_boundaries[k - 1];
            }

            // Build sorted group_idx
            let sorted_group_idx: Vec<u32> = sort_order.iter().map(|&i| group_idx[i]).collect();

            Some((sorted_group_idx, sort_order, group_boundaries))
        }

        /// Initialize multi-chain GPU buffers for batched dispatch.
        ///
        /// Call this before sampling if you want all chains to share observation data
        /// on the GPU and dispatch in a single submit+poll cycle.
        #[allow(dead_code)]
        pub fn init_multi_chain_buffers(&mut self, num_chains: u32) {
            if let Some(ref ctx) = self.gpu_ctx {
                if self.can_use_gpu() {
                    self.multi_chain_buffers = Some(std::sync::Arc::new(
                        ctx.create_multi_chain_buffers(&self.observed_f32, 32, num_chains),
                    ));
                }
            }
        }

        /// Compute likelihood logp + grad for multiple chains in a single GPU dispatch.
        ///
        /// Each element of `chain_params_list` is the parameter vector for one chain.
        /// Returns Vec<GpuLikelihoodResult> with one entry per chain, matching
        /// the signature of `compute_likelihood_gpu`.
        ///
        /// Falls back to sequential per-chain computation if multi-chain buffers are
        /// not initialized or the distribution doesn't support batched dispatch.
        #[allow(dead_code)]
        pub fn compute_likelihood_multi_chain(
            &self,
            chain_params_list: &[&[f32]],
            observed: &[f32],
            spec: &DistributionSpec,
            gpu_ctx: &gpu::sync::GpuContextSync,
        ) -> Result<Vec<gpu::kernels::GpuLikelihoodResult>, String> {
            let num_chains = chain_params_list.len();
            let mc_buffers = self.multi_chain_buffers.as_deref();

            // Try batched dispatch for Normal distribution (most common case)
            if let Some(buffers) = mc_buffers {
                if buffers.num_chains as usize == num_chains && spec.dist_type.as_str() == "Normal"
                {
                    let scale = self.get_param_f64(&spec.params, "scale", 1.0) as f32;
                    let mut loc_indices = Vec::with_capacity(num_chains);
                    let params: Vec<gpu::NormalBatchParams> = chain_params_list
                        .iter()
                        .map(|params| {
                            let (loc_val, loc_idx) =
                                self.resolve_param_value(params, &spec.params, "loc");
                            loc_indices.push(loc_idx);
                            gpu::NormalBatchParams {
                                mu: loc_val,
                                sigma: scale,
                                count: observed.len() as u32,
                                _padding: 0,
                            }
                        })
                        .collect();

                    let fused_results = gpu_ctx.run_normal_multi_chain_fused(buffers, &params)?;

                    return Ok(fused_results
                        .into_iter()
                        .zip(loc_indices)
                        .map(|(r, idx)| {
                            let mut grads = Vec::new();
                            if let Some(i) = idx {
                                grads.push((i, r.total_grad as f64));
                            }
                            gpu::kernels::GpuLikelihoodResult {
                                log_prob: r.total_log_prob as f64,
                                param_grads: grads,
                            }
                        })
                        .collect());
                }
            }

            // Fallback: sequential per-chain computation using single-chain persistent buffers
            let mut results = Vec::with_capacity(num_chains);
            for params in chain_params_list {
                let r = self.compute_likelihood_gpu(
                    params,
                    observed,
                    spec,
                    gpu_ctx,
                    self.gpu_buffers.as_deref(),
                )?;
                results.push(r);
            }
            Ok(results)
        }
    }

    // ============================================================================
    // Helper Functions
    // ============================================================================

    /// Log-gamma function using Stirling's approximation
    fn ln_gamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }

        // Use Lanczos approximation coefficients
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

    /// Generate initial values for chains based on prior types
    fn generate_inits(
        priors: &[DistributionSpec],
        prior_sizes: &[usize],
        num_chains: usize,
        seed: u64,
        device: &WasmDevice,
    ) -> Vec<Tensor<WasmBackend, 1>> {
        let dim: usize = prior_sizes.iter().sum();
        let mut rng = GpuRng::<WasmBackend>::new(seed.wrapping_add(1000), dim, device);

        (0..num_chains)
            .map(|chain_idx| {
                // Generate sensible initial values based on prior type
                let mut inits = Vec::with_capacity(dim);
                for (prior_idx, prior) in priors.iter().enumerate() {
                    let size = prior_sizes.get(prior_idx).copied().unwrap_or(1);
                    let init_val = match prior.dist_type.as_str() {
                        "Beta" => {
                            // Beta: initialize in (0.2, 0.8) range
                            0.5 + 0.1 * (chain_idx as f32 - num_chains as f32 / 2.0)
                        }
                        "HalfNormal" | "HalfCauchy" | "Exponential" => {
                            // Positive distributions: initialize around 1
                            1.0 + 0.1 * chain_idx as f32
                        }
                        "Gamma" => {
                            // Gamma: positive, use shape/rate as hint
                            let shape = prior
                                .params
                                .get("shape")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32;
                            let rate = prior
                                .params
                                .get("rate")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32;
                            (shape / rate).max(0.1) + 0.1 * chain_idx as f32
                        }
                        "LogNormal" => {
                            // LogNormal: positive, initialize around exp(loc)
                            let loc = prior
                                .params
                                .get("loc")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32;
                            loc.exp().max(0.1) + 0.1 * chain_idx as f32
                        }
                        "Uniform" => {
                            // Uniform: middle of range
                            let low = prior
                                .params
                                .get("low")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32;
                            let high = prior
                                .params
                                .get("high")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32;
                            (low + high) / 2.0 + 0.05 * chain_idx as f32
                        }
                        "StudentT" | "Cauchy" => {
                            // Heavy-tailed: initialize near location
                            let loc = prior
                                .params
                                .get("loc")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32;
                            loc + 0.1 * (chain_idx as f32 - num_chains as f32 / 2.0)
                        }
                        "InverseGamma" => {
                            // InverseGamma: positive, use beta/(alpha-1) as mode hint
                            let alpha = prior
                                .params
                                .get("alpha")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32;
                            let beta_param = prior
                                .params
                                .get("beta")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32;
                            let mode = if alpha > 1.0 {
                                beta_param / (alpha + 1.0)
                            } else {
                                beta_param
                            };
                            mode.max(0.1) + 0.1 * chain_idx as f32
                        }
                        "ChiSquared" => {
                            // ChiSquared: positive, initialize around df (degrees of freedom)
                            let k = prior
                                .params
                                .get("df")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32;
                            k.max(0.1) + 0.1 * chain_idx as f32
                        }
                        "Laplace" | "Logistic" => {
                            // Real-valued, initialize near location
                            let loc = prior
                                .params
                                .get("loc")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32;
                            loc + 0.1 * (chain_idx as f32 - num_chains as f32 / 2.0)
                        }
                        "Weibull" | "Pareto" | "HalfStudentT" => {
                            // Positive distributions: initialize around 1
                            1.0 + 0.1 * chain_idx as f32
                        }
                        "Gumbel" => {
                            // Real-valued, initialize near location
                            let loc = prior
                                .params
                                .get("loc")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32;
                            loc + 0.1 * (chain_idx as f32 - num_chains as f32 / 2.0)
                        }
                        "TruncatedNormal" => {
                            // Initialize at midpoint of (low, high)
                            let low = prior
                                .params
                                .get("low")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(-1.0) as f32;
                            let high = prior
                                .params
                                .get("high")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0) as f32;
                            let loc = prior
                                .params
                                .get("loc")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(((low + high) / 2.0) as f64)
                                as f32;
                            // Clamp to (low, high) range
                            loc.clamp(low + 0.01, high - 0.01)
                                + 0.01 * (chain_idx as f32 - num_chains as f32 / 2.0)
                        }
                        "LKJCorr" => {
                            // LKJ Cholesky: off-diagonal elements of L near 0
                            // (identity correlation matrix). Small perturbation per chain.
                            0.01 * (chain_idx as f32 - num_chains as f32 / 2.0)
                        }
                        "NegativeBinomial"
                        | "Geometric"
                        | "DiscreteUniform"
                        | "BetaBinomial"
                        | "Categorical"
                        | "ZeroInflatedPoisson"
                        | "ZeroInflatedNegativeBinomial"
                        | "Hypergeometric"
                        | "OrderedLogistic" => {
                            // Discrete distributions: initialize at 0 or 1
                            1.0 + 0.1 * chain_idx as f32
                        }
                        _ => {
                            // Normal and others: small perturbation around 0
                            let noise: f32 = rng.normal(&[1]).into_scalar();
                            noise * 0.1
                        }
                    };
                    for _ in 0..size {
                        inits.push(init_val);
                    }
                }
                Tensor::<WasmBackend, 1>::from_floats(inits.as_slice(), device)
            })
            .collect()
    }

    // ============================================================================
    // Linear Regression Model
    // ============================================================================

    /// Linear regression model specification from JavaScript
    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LinearRegressionSpec {
        /// Design matrix X (n_obs x n_features), row-major
        pub predictors: Vec<Vec<f64>>,
        /// Response vector y
        pub response: Vec<f64>,
        /// Prior scale for coefficients (default: 10.0)
        #[serde(default = "default_coef_scale")]
        pub coef_prior_scale: f64,
        /// Prior scale for noise sigma (default: 1.0)
        #[serde(default = "default_sigma_scale")]
        pub sigma_prior_scale: f64,
    }

    fn default_coef_scale() -> f64 {
        10.0
    }
    fn default_sigma_scale() -> f64 {
        1.0
    }

    /// Bayesian linear regression model
    ///
    /// y ~ Normal(X @ beta, sigma)
    /// beta_i ~ Normal(0, coef_prior_scale)
    /// sigma ~ HalfNormal(sigma_prior_scale)
    #[derive(Clone)]
    struct LinearRegressionModel {
        /// Design matrix X (n_obs x n_features) as flat row-major
        x_flat: Vec<f32>,
        /// Response vector y
        y: Vec<f32>,
        /// Number of observations
        n_obs: usize,
        /// Number of features (coefficients)
        n_features: usize,
        /// Prior scale for coefficients
        coef_prior_scale: f32,
        /// Prior scale for sigma
        sigma_prior_scale: f32,
        /// Device for tensor creation
        device: WasmDevice,
    }

    impl LinearRegressionModel {
        fn new(spec: LinearRegressionSpec, device: WasmDevice) -> Result<Self, String> {
            let n_obs = spec.predictors.len();
            if n_obs == 0 {
                return Err("predictors cannot be empty".to_string());
            }
            let n_features = spec.predictors[0].len();
            if n_features == 0 {
                return Err("predictors must have at least one feature".to_string());
            }
            if spec.response.len() != n_obs {
                return Err(format!(
                    "response length ({}) must match number of rows in predictors ({})",
                    spec.response.len(),
                    n_obs
                ));
            }

            // Flatten design matrix to row-major
            let mut x_flat = Vec::with_capacity(n_obs * n_features);
            for row in &spec.predictors {
                if row.len() != n_features {
                    return Err("all predictor rows must have same length".to_string());
                }
                for &val in row {
                    x_flat.push(val as f32);
                }
            }

            let y: Vec<f32> = spec.response.iter().map(|&v| v as f32).collect();

            Ok(Self {
                x_flat,
                y,
                n_obs,
                n_features,
                coef_prior_scale: spec.coef_prior_scale as f32,
                sigma_prior_scale: spec.sigma_prior_scale as f32,
                device,
            })
        }

        /// Compute X @ beta using tensor operations
        fn compute_linear_predictor(
            &self,
            params: &Tensor<WasmBackend, 1>,
        ) -> Tensor<WasmBackend, 1> {
            // params = [beta_0, beta_1, ..., beta_{k-1}, sigma]
            // Extract beta coefficients (all but last param)
            let beta = params.clone().slice([0..self.n_features]);

            // Manual matrix-vector multiply: y_hat[i] = sum_j X[i,j] * beta[j]
            let mut y_hat_vec = Vec::with_capacity(self.n_obs);
            for i in 0..self.n_obs {
                let mut sum = 0.0f32;
                for j in 0..self.n_features {
                    let x_ij = self.x_flat[i * self.n_features + j];
                    // We need to extract beta[j] - but we're working with tensors
                    // For simplicity, we'll compute this differently
                    sum += x_ij; // placeholder, real impl below
                }
                y_hat_vec.push(sum);
            }

            // Better approach: use tensor operations
            // Create X as 2D tensor, beta as 1D, then matmul
            let x_tensor =
                Tensor::<WasmBackend, 1>::from_floats(self.x_flat.as_slice(), &self.device)
                    .reshape([self.n_obs, self.n_features]);

            // beta is shape [n_features], reshape to [n_features, 1] for matmul
            let beta_col = beta.reshape([self.n_features, 1]);

            // y_hat = X @ beta -> [n_obs, 1]
            x_tensor.matmul(beta_col).reshape([self.n_obs])
        }
    }

    impl BayesianModel<WasmBackend> for LinearRegressionModel {
        fn dim(&self) -> usize {
            // n_features coefficients + 1 sigma
            self.n_features + 1
        }

        fn log_prob(&self, params: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            // params = [beta_0, beta_1, ..., beta_{k-1}, log_sigma]
            // We use log_sigma to keep sigma positive via transform

            let mut log_prob = Tensor::<WasmBackend, 1>::zeros([1], &self.device);

            // 1. Priors on beta coefficients: beta_i ~ Normal(0, coef_prior_scale)
            for j in 0..self.n_features {
                let beta_j = params.clone().slice([j..j + 1]);
                // log N(beta_j | 0, scale) = -0.5 * log(2*pi*scale^2) - 0.5 * (beta_j/scale)^2
                let scale = self.coef_prior_scale;
                let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                let z = beta_j.div_scalar(scale);
                let log_prior = z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm);
                log_prob = log_prob.add(log_prior);
            }

            // 2. Prior on sigma: sigma ~ HalfNormal(sigma_prior_scale)
            // We parameterize as log_sigma, so sigma = exp(log_sigma)
            // log HalfNormal(sigma | scale) = log(2) - 0.5*log(2*pi*scale^2) - 0.5*(sigma/scale)^2
            // Plus Jacobian: log|d_sigma/d_log_sigma| = log_sigma (since sigma = exp(log_sigma))
            let log_sigma = params.clone().slice([self.n_features..self.n_features + 1]);
            let sigma = log_sigma.clone().exp();
            let scale = self.sigma_prior_scale;
            let log_norm = (2.0f32).ln() - 0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
            let z = sigma.clone().div_scalar(scale);
            let log_prior_sigma = z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm);
            // Add Jacobian for log transform
            let log_jacobian = log_sigma.clone();
            log_prob = log_prob.add(log_prior_sigma).add(log_jacobian);

            // 3. Likelihood: y ~ Normal(X @ beta, sigma)
            let y_hat = self.compute_linear_predictor(params);
            let y_tensor = Tensor::<WasmBackend, 1>::from_floats(self.y.as_slice(), &self.device);

            // residuals = y - y_hat
            let residuals = y_tensor.sub(y_hat);

            // log N(y | y_hat, sigma) = -0.5*n*log(2*pi) - n*log(sigma) - 0.5*sum((y-y_hat)^2)/sigma^2
            let n = self.n_obs as f32;
            let log_2pi = (2.0 * std::f32::consts::PI).ln();

            // -0.5 * n * log(2*pi)
            let term1 = -0.5 * n * log_2pi;

            // -n * log(sigma) = -n * log_sigma
            let term2 = log_sigma.clone().mul_scalar(-n);

            // -0.5 * sum(residuals^2) / sigma^2
            let sum_sq_resid = residuals.powf_scalar(2.0).sum().reshape([1]);
            let sigma_sq = sigma.powf_scalar(2.0);
            let term3 = sum_sq_resid.div(sigma_sq).mul_scalar(-0.5);

            log_prob = log_prob.add_scalar(term1).add(term2).add(term3);

            log_prob
        }

        fn param_names(&self) -> Vec<String> {
            let mut names = Vec::with_capacity(self.n_features + 1);
            for j in 0..self.n_features {
                names.push(format!("beta_{}", j));
            }
            names.push("sigma".to_string());
            names
        }

        fn transform(&self, unconstrained: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            // Transform log_sigma back to sigma for reporting
            let mut result_vec: Vec<f32> = Vec::with_capacity(self.dim());

            // Beta coefficients stay the same
            let beta_data: Vec<f32> = unconstrained
                .clone()
                .slice([0..self.n_features])
                .into_data()
                .to_vec()
                .unwrap();
            result_vec.extend(beta_data);

            // Transform log_sigma -> sigma
            let log_sigma_data: Vec<f32> = unconstrained
                .clone()
                .slice([self.n_features..self.n_features + 1])
                .into_data()
                .to_vec()
                .unwrap();
            let sigma = log_sigma_data[0].exp();
            result_vec.push(sigma);

            Tensor::<WasmBackend, 1>::from_floats(result_vec.as_slice(), &unconstrained.device())
        }
    }

    /// Generate initial values for linear regression
    fn generate_linear_regression_inits(
        n_features: usize,
        num_chains: usize,
        seed: u64,
        device: &WasmDevice,
    ) -> Vec<Tensor<WasmBackend, 1>> {
        let dim = n_features + 1; // beta + log_sigma
        let mut rng = GpuRng::<WasmBackend>::new(seed.wrapping_add(2000), dim, device);

        (0..num_chains)
            .map(|chain_idx| {
                let mut inits = Vec::with_capacity(dim);

                // Beta coefficients: small random values
                for _ in 0..n_features {
                    let noise: f32 = rng.normal(&[1]).into_scalar();
                    inits.push(noise * 0.1);
                }

                // log_sigma: initialize around log(1) = 0 with small perturbation
                let noise: f32 = rng.normal(&[1]).into_scalar();
                inits.push(noise * 0.1 + 0.1 * chain_idx as f32);

                Tensor::<WasmBackend, 1>::from_floats(inits.as_slice(), device)
            })
            .collect()
    }

    // ============================================================================
    // Prediction and Estimation Utilities
    // ============================================================================

    /// Input format for samples (subset of InferenceOutput)
    #[derive(Debug, Clone, Deserialize)]
    struct SamplesInput {
        samples: HashMap<String, Vec<f64>>,
    }

    /// Input format for credible interval probability
    #[derive(Debug, Clone, Deserialize)]
    struct CredibleIntervalConfig {
        #[serde(default = "default_prob")]
        prob: f64,
    }

    fn default_prob() -> f64 {
        0.95
    }

    /// Input for linear/logistic regression prediction
    #[derive(Debug, Clone, Deserialize)]
    struct PredictionInput {
        predictors: Vec<Vec<f64>>,
    }

    /// Input for hierarchical model prediction
    #[derive(Debug, Clone, Deserialize)]
    struct HierarchicalPredictionInput {
        predictors: Vec<Vec<f64>>,
        #[serde(rename = "groupIds")]
        group_ids: Vec<usize>,
    }

    /// Compute posterior mean for each parameter
    ///
    /// # Arguments
    /// * `samples_json` - JSON with {"samples": {"param_name": [values], ...}}
    ///
    /// # Returns
    /// JSON with {"param_name": mean_value, ...}
    #[wasm_bindgen]
    pub fn posterior_mean(samples_json: &str) -> String {
        let input: SamplesInput = match serde_json::from_str(samples_json) {
            Ok(inp) => inp,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid samples JSON: {}", e)
                })
                .to_string();
            }
        };

        let mut means: HashMap<String, f64> = HashMap::new();
        for (name, values) in input.samples.iter() {
            if values.is_empty() {
                means.insert(name.clone(), f64::NAN);
            } else {
                let sum: f64 = values.iter().sum();
                means.insert(name.clone(), sum / values.len() as f64);
            }
        }

        serde_json::to_string(&means).unwrap_or_else(|e| {
            serde_json::json!({"error": format!("Serialization error: {}", e)}).to_string()
        })
    }

    /// Compute quantile-based credible intervals for each parameter
    ///
    /// # Arguments
    /// * `samples_json` - JSON with {"samples": {"param_name": [values], ...}}
    /// * `config_json` - JSON with {"prob": 0.95} (optional, defaults to 0.95)
    ///
    /// # Returns
    /// JSON with {"param_name": [lower, upper], ...}
    #[wasm_bindgen]
    pub fn credible_interval(samples_json: &str, config_json: &str) -> String {
        let input: SamplesInput = match serde_json::from_str(samples_json) {
            Ok(inp) => inp,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid samples JSON: {}", e)
                })
                .to_string();
            }
        };

        let config: CredibleIntervalConfig = match serde_json::from_str(config_json) {
            Ok(cfg) => cfg,
            Err(_) => CredibleIntervalConfig { prob: 0.95 },
        };

        let alpha = 1.0 - config.prob;
        let lower_q = alpha / 2.0;
        let upper_q = 1.0 - alpha / 2.0;

        let mut intervals: HashMap<String, [f64; 2]> = HashMap::new();
        for (name, values) in input.samples.iter() {
            if values.is_empty() {
                intervals.insert(name.clone(), [f64::NAN, f64::NAN]);
            } else {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let n = sorted.len();
                let lower_idx = ((n as f64) * lower_q).floor() as usize;
                let upper_idx = ((n as f64) * upper_q).ceil() as usize;

                let lower = sorted[lower_idx.min(n - 1)];
                let upper = sorted[upper_idx.min(n - 1)];

                intervals.insert(name.clone(), [lower, upper]);
            }
        }

        serde_json::to_string(&intervals).unwrap_or_else(|e| {
            serde_json::json!({"error": format!("Serialization error: {}", e)}).to_string()
        })
    }

    /// Generate predictions for linear regression
    ///
    /// # Arguments
    /// * `samples_json` - JSON with {"samples": {"beta_0": [...], "beta_1": [...], ..., "sigma": [...]}}
    /// * `new_x_json` - JSON with {"predictors": [[x1_1, x1_2, ...], [x2_1, x2_2, ...], ...]}
    ///
    /// # Returns
    /// JSON with {"mean": [...], "lower": [...], "upper": [...]} for each new observation
    #[wasm_bindgen]
    pub fn predict_linear_regression(samples_json: &str, new_x_json: &str) -> String {
        let input: SamplesInput = match serde_json::from_str(samples_json) {
            Ok(inp) => inp,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid samples JSON: {}", e)
                })
                .to_string();
            }
        };

        let pred_input: PredictionInput = match serde_json::from_str(new_x_json) {
            Ok(inp) => inp,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid predictors JSON: {}", e)
                })
                .to_string();
            }
        };

        // Extract beta coefficients
        let n_predictors = pred_input.predictors.first().map(|r| r.len()).unwrap_or(0);
        let mut betas: Vec<&Vec<f64>> = Vec::new();
        for i in 0..n_predictors {
            let beta_name = format!("beta_{}", i);
            match input.samples.get(&beta_name) {
                Some(b) => betas.push(b),
                None => {
                    return serde_json::json!({
                        "error": format!("Missing {} in samples", beta_name)
                    })
                    .to_string();
                }
            }
        }

        let n_samples = betas.first().map(|b| b.len()).unwrap_or(0);
        if n_samples == 0 {
            return serde_json::json!({"error": "No samples found"}).to_string();
        }

        let n_obs = pred_input.predictors.len();
        let mut all_preds: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_obs];

        // For each sample, compute predictions
        #[allow(clippy::needless_range_loop)]
        for s in 0..n_samples {
            for (i, x_row) in pred_input.predictors.iter().enumerate() {
                let mut y_hat: f64 = 0.0;
                for (j, &x_val) in x_row.iter().enumerate() {
                    y_hat += betas[j][s] * x_val;
                }
                all_preds[i].push(y_hat);
            }
        }

        // Compute mean and 95% intervals for each observation
        let mut means: Vec<f64> = Vec::with_capacity(n_obs);
        let mut lowers: Vec<f64> = Vec::with_capacity(n_obs);
        let mut uppers: Vec<f64> = Vec::with_capacity(n_obs);

        for preds in all_preds.iter() {
            let mean = preds.iter().sum::<f64>() / preds.len() as f64;
            means.push(mean);

            let mut sorted = preds.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            lowers.push(sorted[(n as f64 * 0.025).floor() as usize]);
            uppers.push(sorted[((n as f64 * 0.975).ceil() as usize).min(n - 1)]);
        }

        serde_json::json!({
            "mean": means,
            "lower": lowers,
            "upper": uppers
        })
        .to_string()
    }

    /// Generate probability predictions for logistic regression
    ///
    /// # Arguments
    /// * `samples_json` - JSON with {"samples": {"beta_0": [...], "beta_1": [...], ...}}
    /// * `new_x_json` - JSON with {"predictors": [[x1_1, x1_2, ...], [x2_1, x2_2, ...], ...]}
    ///
    /// # Returns
    /// JSON with {"mean": [...], "lower": [...], "upper": [...]} for predicted probabilities
    #[wasm_bindgen]
    pub fn predict_logistic_regression(samples_json: &str, new_x_json: &str) -> String {
        let input: SamplesInput = match serde_json::from_str(samples_json) {
            Ok(inp) => inp,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid samples JSON: {}", e)
                })
                .to_string();
            }
        };

        let pred_input: PredictionInput = match serde_json::from_str(new_x_json) {
            Ok(inp) => inp,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid predictors JSON: {}", e)
                })
                .to_string();
            }
        };

        // Extract beta coefficients
        let n_predictors = pred_input.predictors.first().map(|r| r.len()).unwrap_or(0);
        let mut betas: Vec<&Vec<f64>> = Vec::new();
        for i in 0..n_predictors {
            let beta_name = format!("beta_{}", i);
            match input.samples.get(&beta_name) {
                Some(b) => betas.push(b),
                None => {
                    return serde_json::json!({
                        "error": format!("Missing {} in samples", beta_name)
                    })
                    .to_string();
                }
            }
        }

        let n_samples = betas.first().map(|b| b.len()).unwrap_or(0);
        if n_samples == 0 {
            return serde_json::json!({"error": "No samples found"}).to_string();
        }

        let n_obs = pred_input.predictors.len();
        let mut all_probs: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_obs];

        // For each sample, compute predicted probabilities
        #[allow(clippy::needless_range_loop)]
        for s in 0..n_samples {
            for (i, x_row) in pred_input.predictors.iter().enumerate() {
                let mut eta: f64 = 0.0;
                for (j, &x_val) in x_row.iter().enumerate() {
                    eta += betas[j][s] * x_val;
                }
                // Sigmoid with clamping for numerical stability
                let eta_clamped = eta.clamp(-20.0, 20.0);
                let prob = 1.0 / (1.0 + (-eta_clamped).exp());
                all_probs[i].push(prob);
            }
        }

        // Compute mean and 95% intervals for each observation
        let mut means: Vec<f64> = Vec::with_capacity(n_obs);
        let mut lowers: Vec<f64> = Vec::with_capacity(n_obs);
        let mut uppers: Vec<f64> = Vec::with_capacity(n_obs);

        for probs in all_probs.iter() {
            let mean = probs.iter().sum::<f64>() / probs.len() as f64;
            means.push(mean);

            let mut sorted = probs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            lowers.push(sorted[(n as f64 * 0.025).floor() as usize]);
            uppers.push(sorted[((n as f64 * 0.975).ceil() as usize).min(n - 1)]);
        }

        serde_json::json!({
            "mean": means,
            "lower": lowers,
            "upper": uppers
        })
        .to_string()
    }

    /// Generate predictions for hierarchical model
    ///
    /// # Arguments
    /// * `samples_json` - JSON with {"samples": {"alpha_0": [...], ..., "beta_0": [...], ...}}
    /// * `new_x_json` - JSON with {"predictors": [...], "groupIds": [...]}
    ///
    /// # Returns
    /// JSON with {"mean": [...], "lower": [...], "upper": [...]} for predictions
    #[wasm_bindgen]
    pub fn predict_hierarchical(samples_json: &str, new_x_json: &str) -> String {
        let input: SamplesInput = match serde_json::from_str(samples_json) {
            Ok(inp) => inp,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid samples JSON: {}", e)
                })
                .to_string();
            }
        };

        let pred_input: HierarchicalPredictionInput = match serde_json::from_str(new_x_json) {
            Ok(inp) => inp,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid prediction input JSON: {}", e)
                })
                .to_string();
            }
        };

        // Determine number of groups from samples
        let mut num_groups = 0;
        while input.samples.contains_key(&format!("alpha_{}", num_groups)) {
            num_groups += 1;
        }
        if num_groups == 0 {
            return serde_json::json!({"error": "No alpha parameters found in samples"})
                .to_string();
        }

        // Extract alpha samples
        let mut alphas: Vec<&Vec<f64>> = Vec::new();
        for i in 0..num_groups {
            alphas.push(input.samples.get(&format!("alpha_{}", i)).unwrap());
        }

        // Extract beta samples
        let n_predictors = pred_input.predictors.first().map(|r| r.len()).unwrap_or(0);
        let mut betas: Vec<&Vec<f64>> = Vec::new();
        for i in 0..n_predictors {
            let beta_name = format!("beta_{}", i);
            match input.samples.get(&beta_name) {
                Some(b) => betas.push(b),
                None => {
                    return serde_json::json!({
                        "error": format!("Missing {} in samples", beta_name)
                    })
                    .to_string();
                }
            }
        }

        let n_samples = alphas.first().map(|a| a.len()).unwrap_or(0);
        if n_samples == 0 {
            return serde_json::json!({"error": "No samples found"}).to_string();
        }

        let n_obs = pred_input.predictors.len();
        if pred_input.group_ids.len() != n_obs {
            return serde_json::json!({
                "error": format!("groupIds length ({}) doesn't match predictors length ({})",
                               pred_input.group_ids.len(), n_obs)
            })
            .to_string();
        }

        let mut all_preds: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_obs];

        // For each sample, compute predictions: y = alpha_g + X @ beta
        for s in 0..n_samples {
            for (i, x_row) in pred_input.predictors.iter().enumerate() {
                let group_id = pred_input.group_ids[i];
                if group_id >= num_groups {
                    // Skip invalid group IDs
                    continue;
                }

                let alpha_g = alphas[group_id][s];
                let mut y_hat = alpha_g;

                for (j, &x_val) in x_row.iter().enumerate() {
                    y_hat += betas[j][s] * x_val;
                }
                all_preds[i].push(y_hat);
            }
        }

        // Compute mean and 95% intervals for each observation
        let mut means: Vec<f64> = Vec::with_capacity(n_obs);
        let mut lowers: Vec<f64> = Vec::with_capacity(n_obs);
        let mut uppers: Vec<f64> = Vec::with_capacity(n_obs);

        for preds in all_preds.iter() {
            if preds.is_empty() {
                means.push(f64::NAN);
                lowers.push(f64::NAN);
                uppers.push(f64::NAN);
                continue;
            }

            let mean = preds.iter().sum::<f64>() / preds.len() as f64;
            means.push(mean);

            let mut sorted = preds.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            lowers.push(sorted[(n as f64 * 0.025).floor() as usize]);
            uppers.push(sorted[((n as f64 * 0.975).ceil() as usize).min(n - 1)]);
        }

        serde_json::json!({
            "mean": means,
            "lower": lowers,
            "upper": uppers
        })
        .to_string()
    }

    /// Run Bayesian linear regression
    ///
    /// # Arguments
    /// * `model_json` - JSON string with predictors, response, and optional prior scales
    /// * `config_json` - JSON string with inference configuration
    ///
    /// # Returns
    /// JSON string with samples (beta_0, beta_1, ..., sigma), diagnostics, and configuration
    #[wasm_bindgen]
    pub fn run_linear_regression(model_json: &str, config_json: &str) -> String {
        // Parse inputs
        let model_spec: LinearRegressionSpec = match serde_json::from_str(model_json) {
            Ok(spec) => spec,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model JSON: {}", e)
                })
                .to_string();
            }
        };

        let config: InferenceConfig = match serde_json::from_str(config_json) {
            Ok(cfg) => cfg,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid config JSON: {}", e)
                })
                .to_string();
            }
        };

        // Create device and model
        let device = get_device_or_init();

        let n_features = if model_spec.predictors.is_empty() {
            0
        } else {
            model_spec.predictors[0].len()
        };

        let model = match LinearRegressionModel::new(model_spec, device) {
            Ok(m) => m,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model spec: {}", e)
                })
                .to_string();
            }
        };

        let param_names = model.param_names();

        // Configure NUTS sampler
        let sampler_config = NutsConfig {
            num_samples: config.num_samples,
            num_warmup: config.num_warmup,
            max_tree_depth: 10,
            target_accept: config.target_accept,
            init_step_size: 0.1, // Smaller step size for regression
        };

        // Configure multi-chain sampling
        let multi_config = MultiChainConfig::new(config.num_chains, sampler_config, config.seed);
        let sampler = MultiChainSampler::new(model, multi_config);

        // Generate initial values
        let inits =
            generate_linear_regression_inits(n_features, config.num_chains, config.seed, &device);

        // Run sampling
        let result = sampler.sample(inits);

        // Extract samples by parameter (transform sigma)
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for (idx, name) in param_names.iter().enumerate() {
            let mut param_samples = result.get_param_samples_flat(idx);

            // Transform log_sigma -> sigma for the last parameter
            if name == "sigma" {
                param_samples = param_samples.iter().map(|&x| x.exp()).collect();
            }

            samples.insert(name.clone(), param_samples);
        }

        // Compute diagnostics
        let mut rhat_vals: HashMap<String, f64> = HashMap::new();
        let mut ess_vals: HashMap<String, f64> = HashMap::new();

        for (idx, name) in param_names.iter().enumerate() {
            let chains = result.get_param_samples(idx);
            rhat_vals.insert(name.clone(), rhat(&chains));
            ess_vals.insert(name.clone(), ess(&chains));
        }

        // Get average step size
        let step_sizes = result.final_step_sizes();
        let avg_step_size = if step_sizes.is_empty() {
            1.0
        } else {
            step_sizes.iter().sum::<f64>() / step_sizes.len() as f64
        };

        // Build output
        let output = InferenceOutput {
            samples,
            diagnostics: DiagnosticsOutput {
                rhat: rhat_vals,
                ess: ess_vals,
                divergences: result.total_divergences(),
            },
            config: ConfigOutput {
                num_samples: config.num_samples,
                num_warmup: config.num_warmup,
                num_chains: config.num_chains,
                step_size: avg_step_size,
            },
        };

        serde_json::to_string(&output).unwrap_or_else(|e| {
            serde_json::json!({
                "error": format!("Failed to serialize output: {}", e)
            })
            .to_string()
        })
    }

    // ============================================================================
    // Logistic Regression Model
    // ============================================================================

    /// Logistic regression model specification from JavaScript
    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LogisticRegressionSpec {
        /// Design matrix X (n_obs x n_features), row-major
        pub predictors: Vec<Vec<f64>>,
        /// Binary response vector y (0 or 1)
        pub response: Vec<f64>,
        /// Prior scale for coefficients (default: 2.5, weakly informative for logistic)
        #[serde(default = "default_logistic_coef_scale")]
        pub coef_prior_scale: f64,
    }

    fn default_logistic_coef_scale() -> f64 {
        2.5
    }

    /// Bayesian logistic regression model
    ///
    /// y ~ Bernoulli(p)
    /// p = sigmoid(X @ beta)
    /// beta_i ~ Normal(0, coef_prior_scale)
    #[derive(Clone)]
    struct LogisticRegressionModel {
        /// Design matrix X (n_obs x n_features) as flat row-major
        x_flat: Vec<f32>,
        /// Binary response vector y (0 or 1)
        y: Vec<f32>,
        /// Number of observations
        n_obs: usize,
        /// Number of features (coefficients)
        n_features: usize,
        /// Prior scale for coefficients
        coef_prior_scale: f32,
        /// Device for tensor creation
        device: WasmDevice,
    }

    impl LogisticRegressionModel {
        fn new(spec: LogisticRegressionSpec, device: WasmDevice) -> Result<Self, String> {
            let n_obs = spec.predictors.len();
            if n_obs == 0 {
                return Err("predictors cannot be empty".to_string());
            }
            let n_features = spec.predictors[0].len();
            if n_features == 0 {
                return Err("predictors must have at least one feature".to_string());
            }
            if spec.response.len() != n_obs {
                return Err(format!(
                    "response length ({}) must match number of rows in predictors ({})",
                    spec.response.len(),
                    n_obs
                ));
            }

            // Validate response is binary
            for &y in &spec.response {
                if y != 0.0 && y != 1.0 {
                    return Err(format!("response must be 0 or 1, got {}", y));
                }
            }

            // Flatten design matrix to row-major
            let mut x_flat = Vec::with_capacity(n_obs * n_features);
            for row in &spec.predictors {
                if row.len() != n_features {
                    return Err("all predictor rows must have same length".to_string());
                }
                for &val in row {
                    x_flat.push(val as f32);
                }
            }

            let y: Vec<f32> = spec.response.iter().map(|&v| v as f32).collect();

            Ok(Self {
                x_flat,
                y,
                n_obs,
                n_features,
                coef_prior_scale: spec.coef_prior_scale as f32,
                device,
            })
        }

        /// Compute X @ beta using tensor operations
        fn compute_linear_predictor(
            &self,
            params: &Tensor<WasmBackend, 1>,
        ) -> Tensor<WasmBackend, 1> {
            let x_tensor =
                Tensor::<WasmBackend, 1>::from_floats(self.x_flat.as_slice(), &self.device)
                    .reshape([self.n_obs, self.n_features]);
            let beta_col = params.clone().reshape([self.n_features, 1]);
            x_tensor.matmul(beta_col).reshape([self.n_obs])
        }

        /// Sigmoid function: 1 / (1 + exp(-x))
        fn sigmoid(&self, x: Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            // Numerically stable sigmoid: use clamp to avoid overflow
            let x_clamped = x.clamp(-20.0, 20.0);
            x_clamped.neg().exp().add_scalar(1.0).recip()
        }
    }

    impl BayesianModel<WasmBackend> for LogisticRegressionModel {
        fn dim(&self) -> usize {
            self.n_features
        }

        fn log_prob(&self, params: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            let mut log_prob = Tensor::<WasmBackend, 1>::zeros([1], &self.device);

            // 1. Priors on beta coefficients: beta_i ~ Normal(0, coef_prior_scale)
            for j in 0..self.n_features {
                let beta_j = params.clone().slice([j..j + 1]);
                let scale = self.coef_prior_scale;
                let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                let z = beta_j.div_scalar(scale);
                let log_prior = z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm);
                log_prob = log_prob.add(log_prior);
            }

            // 2. Likelihood: y ~ Bernoulli(sigmoid(X @ beta))
            let eta = self.compute_linear_predictor(params);
            let p = self.sigmoid(eta.clone());

            // Clamp p to avoid log(0)
            let p_clamped = p.clone().clamp(1e-7, 1.0 - 1e-7);
            let log_p = p_clamped.clone().log();
            let log_1_minus_p = p_clamped.neg().add_scalar(1.0).log();

            let y_tensor = Tensor::<WasmBackend, 1>::from_floats(self.y.as_slice(), &self.device);

            // log Bernoulli(y | p) = y * log(p) + (1-y) * log(1-p)
            // Sum over all observations
            let term1 = y_tensor.clone().mul(log_p);
            let term2 = y_tensor.neg().add_scalar(1.0).mul(log_1_minus_p);
            let log_lik = term1.add(term2).sum().reshape([1]);

            log_prob = log_prob.add(log_lik);

            log_prob
        }

        fn param_names(&self) -> Vec<String> {
            (0..self.n_features)
                .map(|j| format!("beta_{}", j))
                .collect()
        }
    }

    /// Generate initial values for logistic regression
    fn generate_logistic_regression_inits(
        n_features: usize,
        num_chains: usize,
        seed: u64,
        device: &WasmDevice,
    ) -> Vec<Tensor<WasmBackend, 1>> {
        let mut rng = GpuRng::<WasmBackend>::new(seed.wrapping_add(3000), n_features, device);

        (0..num_chains)
            .map(|_chain_idx| {
                let mut inits = Vec::with_capacity(n_features);
                for _ in 0..n_features {
                    let noise: f32 = rng.normal(&[1]).into_scalar();
                    inits.push(noise * 0.1);
                }
                Tensor::<WasmBackend, 1>::from_floats(inits.as_slice(), device)
            })
            .collect()
    }

    /// Run Bayesian logistic regression
    ///
    /// # Arguments
    /// * `model_json` - JSON string with predictors, binary response, and optional prior scale
    /// * `config_json` - JSON string with inference configuration
    ///
    /// # Returns
    /// JSON string with samples (beta_0, beta_1, ...), diagnostics, and configuration
    #[wasm_bindgen]
    pub fn run_logistic_regression(model_json: &str, config_json: &str) -> String {
        let model_spec: LogisticRegressionSpec = match serde_json::from_str(model_json) {
            Ok(spec) => spec,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model JSON: {}", e)
                })
                .to_string();
            }
        };

        let config: InferenceConfig = match serde_json::from_str(config_json) {
            Ok(cfg) => cfg,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid config JSON: {}", e)
                })
                .to_string();
            }
        };

        let device = get_device_or_init();
        let n_features = if model_spec.predictors.is_empty() {
            0
        } else {
            model_spec.predictors[0].len()
        };

        let model = match LogisticRegressionModel::new(model_spec, device) {
            Ok(m) => m,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model spec: {}", e)
                })
                .to_string();
            }
        };

        let param_names = model.param_names();

        let sampler_config = NutsConfig {
            num_samples: config.num_samples,
            num_warmup: config.num_warmup,
            max_tree_depth: 10,
            target_accept: config.target_accept,
            init_step_size: 0.1,
        };

        let multi_config = MultiChainConfig::new(config.num_chains, sampler_config, config.seed);
        let sampler = MultiChainSampler::new(model, multi_config);
        let inits =
            generate_logistic_regression_inits(n_features, config.num_chains, config.seed, &device);
        let result = sampler.sample(inits);

        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for (idx, name) in param_names.iter().enumerate() {
            let param_samples = result.get_param_samples_flat(idx);
            samples.insert(name.clone(), param_samples);
        }

        let mut rhat_vals: HashMap<String, f64> = HashMap::new();
        let mut ess_vals: HashMap<String, f64> = HashMap::new();
        for (idx, name) in param_names.iter().enumerate() {
            let chains = result.get_param_samples(idx);
            rhat_vals.insert(name.clone(), rhat(&chains));
            ess_vals.insert(name.clone(), ess(&chains));
        }

        let step_sizes = result.final_step_sizes();
        let avg_step_size = if step_sizes.is_empty() {
            1.0
        } else {
            step_sizes.iter().sum::<f64>() / step_sizes.len() as f64
        };

        let output = InferenceOutput {
            samples,
            diagnostics: DiagnosticsOutput {
                rhat: rhat_vals,
                ess: ess_vals,
                divergences: result.total_divergences(),
            },
            config: ConfigOutput {
                num_samples: config.num_samples,
                num_warmup: config.num_warmup,
                num_chains: config.num_chains,
                step_size: avg_step_size,
            },
        };

        serde_json::to_string(&output).unwrap_or_else(|e| {
            serde_json::json!({
                "error": format!("Failed to serialize output: {}", e)
            })
            .to_string()
        })
    }

    // ============================================================================
    // Hierarchical (Random Intercepts) Model
    // ============================================================================

    /// Hierarchical model specification from JavaScript
    ///
    /// Model structure:
    /// y_ij ~ Normal(alpha_j + X_ij @ beta, sigma)
    /// alpha_j ~ Normal(mu_alpha, sigma_alpha)  # group-level intercepts
    /// mu_alpha ~ Normal(0, mu_alpha_prior_scale)  # population mean
    /// sigma_alpha ~ HalfNormal(sigma_alpha_prior_scale)  # between-group variance
    /// beta ~ Normal(0, beta_prior_scale)  # fixed effects
    /// sigma ~ HalfNormal(sigma_prior_scale)  # residual variance
    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct HierarchicalModelSpec {
        /// Group indicators for each observation (0-indexed)
        pub group_ids: Vec<usize>,
        /// Design matrix X (n_obs x n_features), row-major (can be empty for intercept-only)
        #[serde(default)]
        pub predictors: Vec<Vec<f64>>,
        /// Response vector y
        pub response: Vec<f64>,
        /// Number of groups
        pub num_groups: usize,
        /// Prior scale for population mean mu_alpha (default: 10.0)
        #[serde(default = "default_mu_alpha_scale")]
        pub mu_alpha_prior_scale: f64,
        /// Prior scale for between-group std sigma_alpha (default: 5.0)
        #[serde(default = "default_sigma_alpha_scale")]
        pub sigma_alpha_prior_scale: f64,
        /// Prior scale for fixed effects beta (default: 10.0)
        #[serde(default = "default_beta_scale")]
        pub beta_prior_scale: f64,
        /// Prior scale for residual sigma (default: 1.0)
        #[serde(default = "default_hier_sigma_scale")]
        pub sigma_prior_scale: f64,
    }

    fn default_mu_alpha_scale() -> f64 {
        10.0
    }
    fn default_sigma_alpha_scale() -> f64 {
        5.0
    }
    fn default_beta_scale() -> f64 {
        10.0
    }
    fn default_hier_sigma_scale() -> f64 {
        1.0
    }

    /// Bayesian hierarchical (random intercepts) model
    ///
    /// Parameters (in order):
    /// - mu_alpha: population mean intercept
    /// - log_sigma_alpha: log of between-group std (transformed for positivity)
    /// - alpha_0, alpha_1, ..., alpha_{J-1}: group-level intercepts
    /// - beta_0, beta_1, ..., beta_{K-1}: fixed effect coefficients
    /// - log_sigma: log of residual std (transformed for positivity)
    #[derive(Clone)]
    struct HierarchicalModel {
        /// Group indicators for each observation
        group_ids: Vec<usize>,
        /// Design matrix X (n_obs x n_features) as flat row-major
        x_flat: Vec<f32>,
        /// Response vector y
        y: Vec<f32>,
        /// Number of observations
        n_obs: usize,
        /// Number of groups
        n_groups: usize,
        /// Number of fixed effect features
        n_features: usize,
        /// Prior scale for mu_alpha
        mu_alpha_prior_scale: f32,
        /// Prior scale for sigma_alpha
        sigma_alpha_prior_scale: f32,
        /// Prior scale for beta coefficients
        beta_prior_scale: f32,
        /// Prior scale for sigma
        sigma_prior_scale: f32,
        /// Device for tensor creation
        device: WasmDevice,
    }

    impl HierarchicalModel {
        fn new(spec: HierarchicalModelSpec, device: WasmDevice) -> Result<Self, String> {
            let n_obs = spec.response.len();
            if n_obs == 0 {
                return Err("response cannot be empty".to_string());
            }
            if spec.group_ids.len() != n_obs {
                return Err(format!(
                    "group_ids length ({}) must match response length ({})",
                    spec.group_ids.len(),
                    n_obs
                ));
            }
            if spec.num_groups == 0 {
                return Err("num_groups must be at least 1".to_string());
            }

            // Validate group_ids are in range
            for &gid in &spec.group_ids {
                if gid >= spec.num_groups {
                    return Err(format!(
                        "group_id {} is >= num_groups {}",
                        gid, spec.num_groups
                    ));
                }
            }

            // Determine number of features from predictors
            let n_features = if spec.predictors.is_empty() {
                0
            } else {
                if spec.predictors.len() != n_obs {
                    return Err(format!(
                        "predictors length ({}) must match response length ({})",
                        spec.predictors.len(),
                        n_obs
                    ));
                }
                spec.predictors[0].len()
            };

            // Flatten design matrix to row-major
            let mut x_flat = Vec::with_capacity(n_obs * n_features);
            if n_features > 0 {
                for row in &spec.predictors {
                    if row.len() != n_features {
                        return Err("all predictor rows must have same length".to_string());
                    }
                    for &val in row {
                        x_flat.push(val as f32);
                    }
                }
            }

            let y: Vec<f32> = spec.response.iter().map(|&v| v as f32).collect();

            Ok(Self {
                group_ids: spec.group_ids,
                x_flat,
                y,
                n_obs,
                n_groups: spec.num_groups,
                n_features,
                mu_alpha_prior_scale: spec.mu_alpha_prior_scale as f32,
                sigma_alpha_prior_scale: spec.sigma_alpha_prior_scale as f32,
                beta_prior_scale: spec.beta_prior_scale as f32,
                sigma_prior_scale: spec.sigma_prior_scale as f32,
                device,
            })
        }

        /// Get parameter indices
        /// Layout: [mu_alpha, log_sigma_alpha, alpha_0..alpha_{J-1}, beta_0..beta_{K-1}, log_sigma]
        fn mu_alpha_idx(&self) -> usize {
            0
        }
        fn log_sigma_alpha_idx(&self) -> usize {
            1
        }
        fn alpha_start_idx(&self) -> usize {
            2
        }
        #[allow(dead_code)]
        fn alpha_end_idx(&self) -> usize {
            2 + self.n_groups
        }
        fn beta_start_idx(&self) -> usize {
            2 + self.n_groups
        }
        #[allow(dead_code)]
        fn beta_end_idx(&self) -> usize {
            2 + self.n_groups + self.n_features
        }
        fn log_sigma_idx(&self) -> usize {
            2 + self.n_groups + self.n_features
        }
    }

    impl BayesianModel<WasmBackend> for HierarchicalModel {
        fn dim(&self) -> usize {
            // mu_alpha + log_sigma_alpha + n_groups alphas + n_features betas + log_sigma
            2 + self.n_groups + self.n_features + 1
        }

        fn log_prob(&self, params: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            let mut log_prob = Tensor::<WasmBackend, 1>::zeros([1], &self.device);

            // Extract parameters
            let mu_alpha = params
                .clone()
                .slice([self.mu_alpha_idx()..self.mu_alpha_idx() + 1]);
            let log_sigma_alpha = params
                .clone()
                .slice([self.log_sigma_alpha_idx()..self.log_sigma_alpha_idx() + 1]);
            let sigma_alpha = log_sigma_alpha.clone().exp();
            let log_sigma = params
                .clone()
                .slice([self.log_sigma_idx()..self.log_sigma_idx() + 1]);
            let sigma = log_sigma.clone().exp();

            // 1. Prior on mu_alpha ~ Normal(0, mu_alpha_prior_scale)
            {
                let scale = self.mu_alpha_prior_scale;
                let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                let z = mu_alpha.clone().div_scalar(scale);
                let log_prior = z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm);
                log_prob = log_prob.add(log_prior);
            }

            // 2. Prior on sigma_alpha ~ HalfNormal(sigma_alpha_prior_scale)
            // Plus Jacobian for log transform
            {
                let scale = self.sigma_alpha_prior_scale;
                let log_norm =
                    (2.0f32).ln() - 0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                let z = sigma_alpha.clone().div_scalar(scale);
                let log_prior = z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm);
                log_prob = log_prob.add(log_prior).add(log_sigma_alpha.clone());
                // Jacobian
            }

            // 3. Prior on each alpha_j ~ Normal(mu_alpha, sigma_alpha)
            for j in 0..self.n_groups {
                let idx = self.alpha_start_idx() + j;
                let alpha_j = params.clone().slice([idx..idx + 1]);
                // log N(alpha_j | mu_alpha, sigma_alpha)
                let z = alpha_j
                    .sub(mu_alpha.clone())
                    .div(sigma_alpha.clone().clamp(1e-6, f32::MAX));
                let log_sigma_alpha_clamped = sigma_alpha.clone().clamp(1e-6, f32::MAX).log();
                let log_prior = z
                    .powf_scalar(2.0)
                    .mul_scalar(-0.5)
                    .sub_scalar(0.5 * (2.0 * std::f32::consts::PI).ln())
                    .sub(log_sigma_alpha_clamped);
                log_prob = log_prob.add(log_prior);
            }

            // 4. Prior on beta coefficients ~ Normal(0, beta_prior_scale)
            for k in 0..self.n_features {
                let idx = self.beta_start_idx() + k;
                let beta_k = params.clone().slice([idx..idx + 1]);
                let scale = self.beta_prior_scale;
                let log_norm = -0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                let z = beta_k.div_scalar(scale);
                let log_prior = z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm);
                log_prob = log_prob.add(log_prior);
            }

            // 5. Prior on sigma ~ HalfNormal(sigma_prior_scale)
            // Plus Jacobian for log transform
            {
                let scale = self.sigma_prior_scale;
                let log_norm =
                    (2.0f32).ln() - 0.5 * (2.0 * std::f32::consts::PI * scale * scale).ln();
                let z = sigma.clone().div_scalar(scale);
                let log_prior = z.powf_scalar(2.0).mul_scalar(-0.5).add_scalar(log_norm);
                log_prob = log_prob.add(log_prior).add(log_sigma.clone()); // Jacobian
            }

            // 6. Likelihood: y_i ~ Normal(alpha_{g[i]} + X_i @ beta, sigma)
            // Compute linear predictor for each observation
            for i in 0..self.n_obs {
                let group_id = self.group_ids[i];
                let alpha_idx = self.alpha_start_idx() + group_id;
                let alpha_g = params.clone().slice([alpha_idx..alpha_idx + 1]);

                // Start with group intercept
                let mut mu_i = alpha_g;

                // Add fixed effects X_i @ beta if we have predictors
                if self.n_features > 0 {
                    for k in 0..self.n_features {
                        let x_ik = self.x_flat[i * self.n_features + k];
                        let beta_idx = self.beta_start_idx() + k;
                        let beta_k = params.clone().slice([beta_idx..beta_idx + 1]);
                        mu_i = mu_i.add(beta_k.mul_scalar(x_ik));
                    }
                }

                // log N(y_i | mu_i, sigma)
                let y_i = self.y[i];
                let z = mu_i
                    .neg()
                    .add_scalar(y_i)
                    .div(sigma.clone().clamp(1e-6, f32::MAX));
                let log_sigma_clamped = sigma.clone().clamp(1e-6, f32::MAX).log();
                let log_lik_i = z
                    .powf_scalar(2.0)
                    .mul_scalar(-0.5)
                    .sub_scalar(0.5 * (2.0 * std::f32::consts::PI).ln())
                    .sub(log_sigma_clamped);
                log_prob = log_prob.add(log_lik_i);
            }

            log_prob
        }

        fn param_names(&self) -> Vec<String> {
            let mut names = Vec::with_capacity(self.dim());
            names.push("mu_alpha".to_string());
            names.push("sigma_alpha".to_string());
            for j in 0..self.n_groups {
                names.push(format!("alpha_{}", j));
            }
            for k in 0..self.n_features {
                names.push(format!("beta_{}", k));
            }
            names.push("sigma".to_string());
            names
        }

        fn transform(&self, unconstrained: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            // Transform log_sigma_alpha -> sigma_alpha and log_sigma -> sigma
            let mut result_vec: Vec<f32> = Vec::with_capacity(self.dim());

            let data: Vec<f32> = unconstrained.clone().into_data().to_vec().unwrap();

            // mu_alpha stays the same
            result_vec.push(data[self.mu_alpha_idx()]);

            // Transform log_sigma_alpha -> sigma_alpha
            result_vec.push(data[self.log_sigma_alpha_idx()].exp());

            // alpha_j stay the same
            for j in 0..self.n_groups {
                result_vec.push(data[self.alpha_start_idx() + j]);
            }

            // beta_k stay the same
            for k in 0..self.n_features {
                result_vec.push(data[self.beta_start_idx() + k]);
            }

            // Transform log_sigma -> sigma
            result_vec.push(data[self.log_sigma_idx()].exp());

            Tensor::<WasmBackend, 1>::from_floats(result_vec.as_slice(), &unconstrained.device())
        }
    }

    /// Generate initial values for hierarchical model
    fn generate_hierarchical_inits(
        n_groups: usize,
        n_features: usize,
        num_chains: usize,
        seed: u64,
        device: &WasmDevice,
    ) -> Vec<Tensor<WasmBackend, 1>> {
        let dim = 2 + n_groups + n_features + 1;
        let mut rng = GpuRng::<WasmBackend>::new(seed.wrapping_add(4000), dim, device);

        (0..num_chains)
            .map(|chain_idx| {
                let mut inits = Vec::with_capacity(dim);

                // mu_alpha: near 0 with small perturbation
                let noise: f32 = rng.normal(&[1]).into_scalar();
                inits.push(noise * 0.5);

                // log_sigma_alpha: around log(1) = 0
                let noise: f32 = rng.normal(&[1]).into_scalar();
                inits.push(noise * 0.2 + 0.1 * chain_idx as f32);

                // alpha_j: near 0 with small perturbation
                for _ in 0..n_groups {
                    let noise: f32 = rng.normal(&[1]).into_scalar();
                    inits.push(noise * 0.5);
                }

                // beta_k: near 0 with small perturbation
                for _ in 0..n_features {
                    let noise: f32 = rng.normal(&[1]).into_scalar();
                    inits.push(noise * 0.1);
                }

                // log_sigma: around log(1) = 0
                let noise: f32 = rng.normal(&[1]).into_scalar();
                inits.push(noise * 0.2 + 0.1 * chain_idx as f32);

                Tensor::<WasmBackend, 1>::from_floats(inits.as_slice(), device)
            })
            .collect()
    }

    /// Run Bayesian hierarchical (random intercepts) model
    ///
    /// # Arguments
    /// * `model_json` - JSON string with groupIds, predictors, response, numGroups, and optional prior scales
    /// * `config_json` - JSON string with inference configuration
    ///
    /// # Returns
    /// JSON string with samples (mu_alpha, sigma_alpha, alpha_0..alpha_J, beta_0..beta_K, sigma),
    /// diagnostics, and configuration
    #[wasm_bindgen]
    pub fn run_hierarchical_model(model_json: &str, config_json: &str) -> String {
        let model_spec: HierarchicalModelSpec = match serde_json::from_str(model_json) {
            Ok(spec) => spec,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model JSON: {}", e)
                })
                .to_string();
            }
        };

        let config: InferenceConfig = match serde_json::from_str(config_json) {
            Ok(cfg) => cfg,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid config JSON: {}", e)
                })
                .to_string();
            }
        };

        let device = get_device_or_init();

        let n_groups = model_spec.num_groups;
        let n_features = if model_spec.predictors.is_empty() {
            0
        } else {
            model_spec.predictors[0].len()
        };

        let model = match HierarchicalModel::new(model_spec, device) {
            Ok(m) => m,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model spec: {}", e)
                })
                .to_string();
            }
        };

        let param_names = model.param_names();

        let sampler_config = NutsConfig {
            num_samples: config.num_samples,
            num_warmup: config.num_warmup,
            max_tree_depth: 10,
            target_accept: config.target_accept,
            init_step_size: 0.05, // Smaller step size for hierarchical models
        };

        let multi_config = MultiChainConfig::new(config.num_chains, sampler_config, config.seed);
        let sampler = MultiChainSampler::new(model, multi_config);
        let inits = generate_hierarchical_inits(
            n_groups,
            n_features,
            config.num_chains,
            config.seed,
            &device,
        );
        let result = sampler.sample(inits);

        // Extract samples and transform constrained parameters
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for (idx, name) in param_names.iter().enumerate() {
            let mut param_samples = result.get_param_samples_flat(idx);

            // Transform log-scale parameters back to natural scale
            if name == "sigma_alpha" || name == "sigma" {
                param_samples = param_samples.iter().map(|&x| x.exp()).collect();
            }

            samples.insert(name.clone(), param_samples);
        }

        let mut rhat_vals: HashMap<String, f64> = HashMap::new();
        let mut ess_vals: HashMap<String, f64> = HashMap::new();
        for (idx, name) in param_names.iter().enumerate() {
            let chains = result.get_param_samples(idx);
            rhat_vals.insert(name.clone(), rhat(&chains));
            ess_vals.insert(name.clone(), ess(&chains));
        }

        let step_sizes = result.final_step_sizes();
        let avg_step_size = if step_sizes.is_empty() {
            1.0
        } else {
            step_sizes.iter().sum::<f64>() / step_sizes.len() as f64
        };

        let output = InferenceOutput {
            samples,
            diagnostics: DiagnosticsOutput {
                rhat: rhat_vals,
                ess: ess_vals,
                divergences: result.total_divergences(),
            },
            config: ConfigOutput {
                num_samples: config.num_samples,
                num_warmup: config.num_warmup,
                num_chains: config.num_chains,
                step_size: avg_step_size,
            },
        };

        serde_json::to_string(&output).unwrap_or_else(|e| {
            serde_json::json!({
                "error": format!("Failed to serialize output: {}", e)
            })
            .to_string()
        })
    }

    // ============================================================================
    // Survival Analysis Models
    // ============================================================================

    /// Exponential survival model specification
    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ExponentialSurvivalSpec {
        /// Observed times (time-to-event or censoring time)
        pub times: Vec<f64>,
        /// Event indicators: 1 = event observed, 0 = right-censored
        pub events: Vec<u8>,
        /// Optional covariates (n_obs x n_features)
        #[serde(default)]
        pub predictors: Vec<Vec<f64>>,
        /// Prior scale for baseline rate (default: 1.0)
        #[serde(default = "default_rate_scale")]
        pub rate_prior_scale: f64,
        /// Prior scale for covariate coefficients (default: 1.0)
        #[serde(default = "default_survival_beta_scale")]
        pub beta_prior_scale: f64,
    }

    fn default_rate_scale() -> f64 {
        1.0
    }
    fn default_survival_beta_scale() -> f64 {
        1.0
    }

    /// Weibull survival model specification
    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct WeibullSurvivalSpec {
        /// Observed times
        pub times: Vec<f64>,
        /// Event indicators: 1 = event, 0 = censored
        pub events: Vec<u8>,
        /// Prior scale for Weibull scale parameter (default: 2.0)
        #[serde(default = "default_weibull_scale")]
        pub scale_prior_scale: f64,
        /// Prior scale for Weibull shape parameter (default: 1.0)
        #[serde(default = "default_weibull_shape")]
        pub shape_prior_scale: f64,
    }

    fn default_weibull_scale() -> f64 {
        2.0
    }
    fn default_weibull_shape() -> f64 {
        1.0
    }

    /// Exponential survival model
    ///
    /// For event: f(t|λ) = λ·exp(-λt)  →  log_lik = log(λ) - λt
    /// For censored: S(t|λ) = exp(-λt)  →  log_lik = -λt
    /// With covariates: λ_i = λ * exp(X_i @ β)
    #[derive(Clone)]
    struct ExponentialSurvivalModel {
        times: Vec<f32>,
        events: Vec<bool>,
        has_covariates: bool,
        x_flat: Vec<f32>,
        n_obs: usize,
        n_features: usize,
        rate_prior_scale: f32,
        beta_prior_scale: f32,
        device: WasmDevice,
    }

    impl ExponentialSurvivalModel {
        fn new(spec: ExponentialSurvivalSpec, device: WasmDevice) -> Result<Self, String> {
            let n_obs = spec.times.len();
            if n_obs == 0 {
                return Err("times cannot be empty".to_string());
            }
            if spec.events.len() != n_obs {
                return Err(format!(
                    "events length ({}) must match times length ({})",
                    spec.events.len(),
                    n_obs
                ));
            }

            let times: Vec<f32> = spec.times.iter().map(|&t| t as f32).collect();
            let events: Vec<bool> = spec.events.iter().map(|&e| e == 1).collect();

            // Validate times are positive
            for &t in &times {
                if t <= 0.0 {
                    return Err("all times must be positive".to_string());
                }
            }

            // Handle covariates
            let has_covariates = !spec.predictors.is_empty();
            let (x_flat, n_features) = if has_covariates {
                if spec.predictors.len() != n_obs {
                    return Err(format!(
                        "predictors rows ({}) must match times length ({})",
                        spec.predictors.len(),
                        n_obs
                    ));
                }
                let n_feat = spec.predictors[0].len();
                let mut flat = Vec::with_capacity(n_obs * n_feat);
                for row in &spec.predictors {
                    if row.len() != n_feat {
                        return Err("all predictor rows must have same length".to_string());
                    }
                    for &val in row {
                        flat.push(val as f32);
                    }
                }
                (flat, n_feat)
            } else {
                (vec![], 0)
            };

            Ok(Self {
                times,
                events,
                has_covariates,
                x_flat,
                n_obs,
                n_features,
                rate_prior_scale: spec.rate_prior_scale as f32,
                beta_prior_scale: spec.beta_prior_scale as f32,
                device,
            })
        }

        fn dim(&self) -> usize {
            // log_rate + (optional) beta coefficients
            1 + self.n_features
        }

        fn param_names(&self) -> Vec<String> {
            let mut names = vec!["rate".to_string()];
            for i in 0..self.n_features {
                names.push(format!("beta_{}", i));
            }
            names
        }
    }

    impl bayesian_sampler::model::BayesianModel<WasmBackend> for ExponentialSurvivalModel {
        fn dim(&self) -> usize {
            self.dim()
        }

        fn log_prob(&self, params: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            // params[0] = log_rate (unconstrained)
            // params[1..] = beta coefficients (if covariates)
            let log_rate = params.clone().slice([0..1]);
            let rate = log_rate.clone().exp();

            // Prior: rate ~ HalfNormal(rate_prior_scale)
            // In log-space: log_rate has Jacobian adjustment
            let prior_scale =
                Tensor::<WasmBackend, 1>::from_floats([self.rate_prior_scale], &self.device);

            // log p(rate) ∝ -rate² / (2σ²) + log_rate (Jacobian)
            let rate_sq = rate.clone().powf_scalar(2.0);
            let prior_var = prior_scale.clone().powf_scalar(2.0).mul_scalar(2.0);
            let log_prior_rate = rate_sq.div(prior_var).neg().add(log_rate.clone());

            // Beta priors (if covariates)
            let log_prior_beta = if self.has_covariates {
                let betas = params.clone().slice([1..1 + self.n_features]);
                // Normal(0, beta_prior_scale)
                let beta_sq_sum = betas.powf_scalar(2.0).sum();
                let beta_var = self.beta_prior_scale * self.beta_prior_scale * 2.0;
                beta_sq_sum.div_scalar(beta_var).neg()
            } else {
                Tensor::<WasmBackend, 1>::zeros([1], &self.device)
            };

            // Compute individual rates (with covariates if present)
            let individual_rates = if self.has_covariates {
                let betas = params.clone().slice([1..1 + self.n_features]);
                let betas_data: Vec<f32> = betas.to_data().to_vec().unwrap();

                let mut log_rates = Vec::with_capacity(self.n_obs);
                let base_log_rate: f32 = log_rate.clone().into_scalar();

                for i in 0..self.n_obs {
                    let mut eta = 0.0f32;
                    for (j, &beta_j) in betas_data.iter().enumerate().take(self.n_features) {
                        eta += self.x_flat[i * self.n_features + j] * beta_j;
                    }
                    log_rates.push(base_log_rate + eta);
                }
                log_rates
            } else {
                let base_log_rate: f32 = log_rate.clone().into_scalar();
                vec![base_log_rate; self.n_obs]
            };

            // Likelihood: sum over observations
            // event: log(λ_i) - λ_i * t_i
            // censored: -λ_i * t_i
            let mut log_lik = 0.0f32;
            for ((&log_rate_i, &t), &event) in individual_rates
                .iter()
                .zip(self.times.iter())
                .zip(self.events.iter())
                .take(self.n_obs)
            {
                let rate_i = log_rate_i.exp();

                if event {
                    // Event: log(λ) - λt
                    log_lik += log_rate_i - rate_i * t;
                } else {
                    // Censored: -λt (survival function)
                    log_lik += -rate_i * t;
                }
            }

            let log_lik_tensor = Tensor::<WasmBackend, 1>::from_floats([log_lik], &self.device);

            log_prior_rate.add(log_prior_beta).add(log_lik_tensor)
        }

        fn param_names(&self) -> Vec<String> {
            self.param_names()
        }
    }

    /// Weibull survival model
    ///
    /// PDF: f(t|λ,k) = (k/λ)(t/λ)^(k-1) exp(-(t/λ)^k)
    /// Survival: S(t|λ,k) = exp(-(t/λ)^k)
    /// λ = scale, k = shape
    #[derive(Clone)]
    struct WeibullSurvivalModel {
        times: Vec<f32>,
        events: Vec<bool>,
        n_obs: usize,
        scale_prior_scale: f32,
        shape_prior_scale: f32,
        device: WasmDevice,
    }

    impl WeibullSurvivalModel {
        fn new(spec: WeibullSurvivalSpec, device: WasmDevice) -> Result<Self, String> {
            let n_obs = spec.times.len();
            if n_obs == 0 {
                return Err("times cannot be empty".to_string());
            }
            if spec.events.len() != n_obs {
                return Err(format!(
                    "events length ({}) must match times length ({})",
                    spec.events.len(),
                    n_obs
                ));
            }

            let times: Vec<f32> = spec.times.iter().map(|&t| t as f32).collect();
            let events: Vec<bool> = spec.events.iter().map(|&e| e == 1).collect();

            for &t in &times {
                if t <= 0.0 {
                    return Err("all times must be positive".to_string());
                }
            }

            Ok(Self {
                times,
                events,
                n_obs,
                scale_prior_scale: spec.scale_prior_scale as f32,
                shape_prior_scale: spec.shape_prior_scale as f32,
                device,
            })
        }

        fn dim(&self) -> usize {
            2 // log_scale, log_shape
        }

        fn param_names(&self) -> Vec<String> {
            vec!["scale".to_string(), "shape".to_string()]
        }
    }

    impl bayesian_sampler::model::BayesianModel<WasmBackend> for WeibullSurvivalModel {
        fn dim(&self) -> usize {
            self.dim()
        }

        fn log_prob(&self, params: &Tensor<WasmBackend, 1>) -> Tensor<WasmBackend, 1> {
            // params[0] = log_scale (λ in log-space)
            // params[1] = log_shape (k in log-space)
            let log_scale = params.clone().slice([0..1]);
            let log_shape = params.clone().slice([1..2]);

            let scale_tensor = log_scale.clone().exp();
            let shape_tensor = log_shape.clone().exp();

            // Extract scalar values for likelihood computation
            let scale: f32 = scale_tensor.clone().into_scalar();
            let shape: f32 = shape_tensor.clone().into_scalar();

            // Priors: HalfNormal for both scale and shape (keep as tensors for autodiff)
            // log p(scale) ∝ -scale²/(2σ²) + log_scale (Jacobian)
            let scale_var = self.scale_prior_scale * self.scale_prior_scale * 2.0;
            let shape_var = self.shape_prior_scale * self.shape_prior_scale * 2.0;

            let log_prior_scale = scale_tensor
                .clone()
                .powf_scalar(2.0)
                .div_scalar(scale_var)
                .neg()
                .add(log_scale);

            let log_prior_shape = shape_tensor
                .powf_scalar(2.0)
                .div_scalar(shape_var)
                .neg()
                .add(log_shape);

            // Likelihood (computed as scalar, then converted to tensor)
            // Event: log f(t) = log(k) - k*log(λ) + (k-1)*log(t) - (t/λ)^k
            // Censored: log S(t) = -(t/λ)^k
            let mut log_lik = 0.0f32;
            for i in 0..self.n_obs {
                let t = self.times[i];
                let t_over_scale = t / scale;
                let t_over_scale_pow_k = t_over_scale.powf(shape);

                if self.events[i] {
                    // Event
                    log_lik += shape.ln() - shape * scale.ln() + (shape - 1.0) * t.ln()
                        - t_over_scale_pow_k;
                } else {
                    // Censored
                    log_lik += -t_over_scale_pow_k;
                }
            }

            let log_lik_tensor = Tensor::<WasmBackend, 1>::from_floats([log_lik], &self.device);

            log_prior_scale.add(log_prior_shape).add(log_lik_tensor)
        }

        fn param_names(&self) -> Vec<String> {
            self.param_names()
        }
    }

    /// Run exponential survival analysis
    #[wasm_bindgen]
    pub fn run_exponential_survival(model_json: &str, config_json: &str) -> String {
        let model_spec: ExponentialSurvivalSpec = match serde_json::from_str(model_json) {
            Ok(spec) => spec,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model JSON: {}", e)
                })
                .to_string();
            }
        };

        let config: InferenceConfig = match serde_json::from_str(config_json) {
            Ok(cfg) => cfg,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid config JSON: {}", e)
                })
                .to_string();
            }
        };

        let device = backend::get_device_or_init();
        let n_features = model_spec.predictors.first().map(|r| r.len()).unwrap_or(0);

        let model = match ExponentialSurvivalModel::new(model_spec, device) {
            Ok(m) => m,
            Err(e) => {
                return serde_json::json!({"error": e}).to_string();
            }
        };

        let param_names = model.param_names();

        // Configure NUTS sampler
        let sampler_config = NutsConfig {
            num_samples: config.num_samples,
            num_warmup: config.num_warmup,
            max_tree_depth: 10,
            target_accept: config.target_accept,
            init_step_size: 0.1,
        };

        // Configure multi-chain sampling
        let multi_config = MultiChainConfig::new(config.num_chains, sampler_config, config.seed);
        let sampler = MultiChainSampler::new(model, multi_config);

        // Generate initial values
        let inits = generate_survival_inits(n_features, config.num_chains, config.seed, &device);

        // Run sampling
        let result = sampler.sample(inits);

        // Extract samples by parameter (transform rate from log-space)
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for (idx, name) in param_names.iter().enumerate() {
            let mut param_samples = result.get_param_samples_flat(idx);

            // Transform log_rate -> rate for the first parameter
            if name == "rate" {
                param_samples = param_samples.iter().map(|&x| x.exp()).collect();
            }

            samples.insert(name.clone(), param_samples);
        }

        // Compute diagnostics
        let mut rhat_vals: HashMap<String, f64> = HashMap::new();
        let mut ess_vals: HashMap<String, f64> = HashMap::new();

        for (idx, name) in param_names.iter().enumerate() {
            let chains = result.get_param_samples(idx);
            rhat_vals.insert(name.clone(), rhat(&chains));
            ess_vals.insert(name.clone(), ess(&chains));
        }

        // Get average step size
        let step_sizes = result.final_step_sizes();
        let avg_step_size = if step_sizes.is_empty() {
            1.0
        } else {
            step_sizes.iter().sum::<f64>() / step_sizes.len() as f64
        };

        let output = InferenceOutput {
            samples,
            diagnostics: DiagnosticsOutput {
                rhat: rhat_vals,
                ess: ess_vals,
                divergences: result.total_divergences(),
            },
            config: ConfigOutput {
                num_samples: config.num_samples,
                num_warmup: config.num_warmup,
                num_chains: config.num_chains,
                step_size: avg_step_size,
            },
        };

        serde_json::to_string(&output).unwrap_or_else(|e| {
            serde_json::json!({"error": format!("Serialization error: {}", e)}).to_string()
        })
    }

    /// Generate initial values for survival models
    fn generate_survival_inits(
        n_features: usize,
        num_chains: usize,
        seed: u64,
        device: &WasmDevice,
    ) -> Vec<Tensor<WasmBackend, 1>> {
        let dim = 1 + n_features; // log_rate + betas
        let mut rng = GpuRng::<WasmBackend>::new(seed.wrapping_add(3000), dim, device);

        (0..num_chains)
            .map(|chain_idx| {
                let mut inits = Vec::with_capacity(dim);

                // log_rate: small random value
                let noise: f32 = rng.normal(&[1]).into_scalar();
                inits.push(noise * 0.1 + 0.1 * chain_idx as f32);

                // beta coefficients
                for _ in 0..n_features {
                    let noise: f32 = rng.normal(&[1]).into_scalar();
                    inits.push(noise * 0.1);
                }

                Tensor::<WasmBackend, 1>::from_floats(inits.as_slice(), device)
            })
            .collect()
    }

    /// Run Weibull survival analysis
    #[wasm_bindgen]
    pub fn run_weibull_survival(model_json: &str, config_json: &str) -> String {
        let model_spec: WeibullSurvivalSpec = match serde_json::from_str(model_json) {
            Ok(spec) => spec,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model JSON: {}", e)
                })
                .to_string();
            }
        };

        let config: InferenceConfig = match serde_json::from_str(config_json) {
            Ok(cfg) => cfg,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid config JSON: {}", e)
                })
                .to_string();
            }
        };

        let device = backend::get_device_or_init();
        let model = match WeibullSurvivalModel::new(model_spec, device) {
            Ok(m) => m,
            Err(e) => {
                return serde_json::json!({"error": e}).to_string();
            }
        };

        let param_names = model.param_names();

        // Configure NUTS sampler
        let sampler_config = NutsConfig {
            num_samples: config.num_samples,
            num_warmup: config.num_warmup,
            max_tree_depth: 10,
            target_accept: config.target_accept,
            init_step_size: 0.1,
        };

        // Configure multi-chain sampling
        let multi_config = MultiChainConfig::new(config.num_chains, sampler_config, config.seed);
        let sampler = MultiChainSampler::new(model, multi_config);

        // Generate initial values
        let inits = generate_weibull_inits(config.num_chains, config.seed, &device);

        // Run sampling
        let result = sampler.sample(inits);

        // Extract samples by parameter (transform from log-space)
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for (idx, name) in param_names.iter().enumerate() {
            let mut param_samples = result.get_param_samples_flat(idx);

            // Transform log_scale -> scale, log_shape -> shape
            param_samples = param_samples.iter().map(|&x| x.exp()).collect();

            samples.insert(name.clone(), param_samples);
        }

        // Compute diagnostics
        let mut rhat_vals: HashMap<String, f64> = HashMap::new();
        let mut ess_vals: HashMap<String, f64> = HashMap::new();

        for (idx, name) in param_names.iter().enumerate() {
            let chains = result.get_param_samples(idx);
            rhat_vals.insert(name.clone(), rhat(&chains));
            ess_vals.insert(name.clone(), ess(&chains));
        }

        // Get average step size
        let step_sizes = result.final_step_sizes();
        let avg_step_size = if step_sizes.is_empty() {
            1.0
        } else {
            step_sizes.iter().sum::<f64>() / step_sizes.len() as f64
        };

        let output = InferenceOutput {
            samples,
            diagnostics: DiagnosticsOutput {
                rhat: rhat_vals,
                ess: ess_vals,
                divergences: result.total_divergences(),
            },
            config: ConfigOutput {
                num_samples: config.num_samples,
                num_warmup: config.num_warmup,
                num_chains: config.num_chains,
                step_size: avg_step_size,
            },
        };

        serde_json::to_string(&output).unwrap_or_else(|e| {
            serde_json::json!({"error": format!("Serialization error: {}", e)}).to_string()
        })
    }

    /// Generate initial values for Weibull survival model
    fn generate_weibull_inits(
        num_chains: usize,
        seed: u64,
        device: &WasmDevice,
    ) -> Vec<Tensor<WasmBackend, 1>> {
        let dim = 2; // log_scale, log_shape
        let mut rng = GpuRng::<WasmBackend>::new(seed.wrapping_add(4000), dim, device);

        (0..num_chains)
            .map(|chain_idx| {
                // log_scale, log_shape: small random values
                let noise1: f32 = rng.normal(&[1]).into_scalar();
                let noise2: f32 = rng.normal(&[1]).into_scalar();
                Tensor::<WasmBackend, 1>::from_floats(
                    [noise1 * 0.1 + 0.1 * chain_idx as f32, noise2 * 0.1],
                    device,
                )
            })
            .collect()
    }

    // ============================================================================
    // Main Inference Entry Point
    // ============================================================================

    /// Run Bayesian inference on a model
    ///
    /// # Arguments
    /// * `model_json` - JSON string describing the model (priors and likelihood)
    /// * `config_json` - JSON string with inference configuration
    ///
    /// # Returns
    /// JSON string with samples, diagnostics, and configuration
    #[wasm_bindgen]
    pub fn run_inference(model_json: &str, config_json: &str) -> String {
        // Parse inputs
        let model_spec: ModelSpec = match serde_json::from_str(model_json) {
            Ok(spec) => spec,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid model JSON: {}", e)
                })
                .to_string();
            }
        };

        let config: InferenceConfig = match serde_json::from_str(config_json) {
            Ok(cfg) => cfg,
            Err(e) => {
                return serde_json::json!({
                    "error": format!("Invalid config JSON: {}", e)
                })
                .to_string();
            }
        };

        // Create device and model
        let device = get_device_or_init();

        // Extract prior specs and sizes for initialization before moving model_spec
        let prior_specs: Vec<DistributionSpec> = model_spec
            .priors
            .iter()
            .map(|p| p.distribution.clone())
            .collect();
        let prior_sizes: Vec<usize> = model_spec.priors.iter().map(|p| p.size.max(1)).collect();

        let model = DynamicModel::new(model_spec, device);
        let param_names = model.param_names();

        // Configure NUTS sampler
        let sampler_config = NutsConfig {
            num_samples: config.num_samples,
            num_warmup: config.num_warmup,
            max_tree_depth: 10,
            target_accept: config.target_accept,
            init_step_size: 1.0,
        };

        // Configure multi-chain sampling
        let multi_config = MultiChainConfig::new(config.num_chains, sampler_config, config.seed);
        let sampler = MultiChainSampler::new(model, multi_config);

        // Generate initial values based on prior types
        let inits = generate_inits(
            &prior_specs,
            &prior_sizes,
            config.num_chains,
            config.seed,
            &device,
        );

        // Run sampling
        let result = sampler.sample(inits);

        // Extract samples by parameter
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for (idx, name) in param_names.iter().enumerate() {
            let param_samples = result.get_param_samples_flat(idx);
            samples.insert(name.clone(), param_samples);
        }

        // Compute diagnostics
        let mut rhat_vals: HashMap<String, f64> = HashMap::new();
        let mut ess_vals: HashMap<String, f64> = HashMap::new();

        for (idx, name) in param_names.iter().enumerate() {
            let chains = result.get_param_samples(idx);
            rhat_vals.insert(name.clone(), rhat(&chains));
            ess_vals.insert(name.clone(), ess(&chains));
        }

        // Get average step size
        let step_sizes = result.final_step_sizes();
        let avg_step_size = if step_sizes.is_empty() {
            1.0
        } else {
            step_sizes.iter().sum::<f64>() / step_sizes.len() as f64
        };

        // Build output
        let output = InferenceOutput {
            samples,
            diagnostics: DiagnosticsOutput {
                rhat: rhat_vals,
                ess: ess_vals,
                divergences: result.total_divergences(),
            },
            config: ConfigOutput {
                num_samples: config.num_samples,
                num_warmup: config.num_warmup,
                num_chains: config.num_chains,
                step_size: avg_step_size,
            },
        };

        serde_json::to_string(&output).unwrap_or_else(|e| {
            serde_json::json!({
                "error": format!("Failed to serialize output: {}", e)
            })
            .to_string()
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_version() {
            let v = version();
            assert!(!v.is_empty());
            assert!(v.starts_with("0."));
        }

        #[test]
        fn test_ln_gamma() {
            // Test against known values
            // lgamma(1) = 0
            assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);
            // lgamma(2) = 0 (since 1! = 1)
            assert!((ln_gamma(2.0) - 0.0).abs() < 1e-10);
            // lgamma(3) = ln(2) = 0.693...
            assert!((ln_gamma(3.0) - std::f64::consts::LN_2).abs() < 1e-5);
            // lgamma(4) = ln(6) = 1.791...
            assert!((ln_gamma(4.0) - 1.791759).abs() < 1e-5);
        }

        #[test]
        fn test_run_inference_beta_binomial() {
            let model_json = r#"{
            "priors": [
                {
                    "name": "theta",
                    "distribution": {
                        "type": "Beta",
                        "params": {"alpha": 1, "beta": 1}
                    }
                }
            ],
            "likelihood": {
                "distribution": {
                    "type": "Binomial",
                    "params": {"n": 10, "p": "theta"}
                },
                "observed": [7]
            }
        }"#;

            let config_json = r#"{
            "numSamples": 100,
            "numWarmup": 50,
            "numChains": 2,
            "targetAccept": 0.8,
            "seed": 42
        }"#;

            let result = run_inference(model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Check we got samples
            assert!(parsed.get("samples").is_some());
            let samples = parsed.get("samples").unwrap();
            assert!(samples.get("theta").is_some());

            // Check we got diagnostics
            assert!(parsed.get("diagnostics").is_some());
            let diagnostics = parsed.get("diagnostics").unwrap();
            assert!(diagnostics.get("rhat").is_some());
            assert!(diagnostics.get("ess").is_some());
        }

        #[test]
        fn test_run_inference_invalid_json() {
            let result = run_inference("not valid json", "{}");
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
            assert!(parsed.get("error").is_some());
        }

        #[test]
        fn test_run_inference_gamma_poisson() {
            // Gamma-Poisson model: Gamma(2, 1) prior, observed counts [3, 4, 5, 3, 4]
            let model_json = r#"{
            "priors": [
                {
                    "name": "rate",
                    "distribution": {
                        "type": "Gamma",
                        "params": {"shape": 2, "rate": 1}
                    }
                }
            ],
            "likelihood": {
                "distribution": {
                    "type": "Poisson",
                    "params": {"rate": "rate"}
                },
                "observed": [3, 4, 5, 3, 4]
            }
        }"#;

            let config_json = r#"{
            "numSamples": 100,
            "numWarmup": 50,
            "numChains": 2,
            "seed": 123
        }"#;

            let result = run_inference(model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Check we got samples
            assert!(parsed.get("samples").is_some());
            let samples = parsed.get("samples").unwrap();
            assert!(samples.get("rate").is_some());

            // Posterior should be Gamma(2 + sum(obs), 1 + n) = Gamma(21, 6)
            // Posterior mean = 21/6 ≈ 3.5
            let rate_samples: Vec<f64> = samples
                .get("rate")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let mean: f64 = rate_samples.iter().sum::<f64>() / rate_samples.len() as f64;

            // Should be roughly around 3.5 (posterior mean)
            assert!(
                mean > 2.0 && mean < 5.0,
                "Posterior mean {} out of expected range",
                mean
            );
        }

        #[test]
        fn test_run_inference_exponential_prior() {
            // Exponential prior for scale parameter
            let model_json = r#"{
            "priors": [
                {
                    "name": "sigma",
                    "distribution": {
                        "type": "Exponential",
                        "params": {"rate": 1}
                    }
                }
            ],
            "likelihood": {
                "distribution": {
                    "type": "Normal",
                    "params": {"loc": 0, "scale": "sigma"}
                },
                "observed": [0.5, -0.3, 0.2]
            }
        }"#;

            let config_json = r#"{
            "numSamples": 50,
            "numWarmup": 25,
            "numChains": 2,
            "seed": 456
        }"#;

            let result = run_inference(model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Should run without error
            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );
            assert!(parsed.get("samples").is_some());
        }

        #[test]
        fn test_linear_regression() {
            // Simple linear regression: y = 2 + 3*x + noise
            // Generate data with known coefficients
            let x_vals: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
            let y_vals: Vec<f64> = x_vals.iter().map(|&x| 2.0 + 3.0 * x).collect();

            // Build predictors with intercept (column of 1s)
            let predictors: Vec<Vec<f64>> = x_vals.iter().map(|&x| vec![1.0, x]).collect();

            let model_json = serde_json::json!({
                "predictors": predictors,
                "response": y_vals,
                "coefPriorScale": 10.0,
                "sigmaPriorScale": 1.0
            })
            .to_string();

            let config_json = r#"{
            "numSamples": 200,
            "numWarmup": 100,
            "numChains": 2,
            "targetAccept": 0.8,
            "seed": 789
        }"#;

            let result = run_linear_regression(&model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Should run without error
            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );
            assert!(parsed.get("samples").is_some());

            let samples = parsed.get("samples").unwrap();

            // Check we got beta_0, beta_1, and sigma
            assert!(samples.get("beta_0").is_some(), "Missing beta_0");
            assert!(samples.get("beta_1").is_some(), "Missing beta_1");
            assert!(samples.get("sigma").is_some(), "Missing sigma");

            // Check posterior means are close to true values
            let beta_0: Vec<f64> = samples
                .get("beta_0")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let beta_1: Vec<f64> = samples
                .get("beta_1")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();

            let beta_0_mean: f64 = beta_0.iter().sum::<f64>() / beta_0.len() as f64;
            let beta_1_mean: f64 = beta_1.iter().sum::<f64>() / beta_1.len() as f64;

            // With no noise in data, posteriors should be very close to true values
            // True beta_0 = 2.0, beta_1 = 3.0
            assert!(
                (beta_0_mean - 2.0).abs() < 1.0,
                "beta_0 mean {} too far from 2.0",
                beta_0_mean
            );
            assert!(
                (beta_1_mean - 3.0).abs() < 1.0,
                "beta_1 mean {} too far from 3.0",
                beta_1_mean
            );
        }

        #[test]
        fn test_logistic_regression() {
            // Logistic regression with separable data
            // Generate data where x < 0 -> y = 0, x > 0 -> y = 1
            let x_vals: Vec<f64> = vec![-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0];
            let y_vals: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

            // Build predictors with intercept
            let predictors: Vec<Vec<f64>> = x_vals.iter().map(|&x| vec![1.0, x]).collect();

            let model_json = serde_json::json!({
                "predictors": predictors,
                "response": y_vals,
                "coefPriorScale": 2.5
            })
            .to_string();

            let config_json = r#"{
            "numSamples": 200,
            "numWarmup": 100,
            "numChains": 2,
            "targetAccept": 0.8,
            "seed": 321
        }"#;

            let result = run_logistic_regression(&model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Should run without error
            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );
            assert!(parsed.get("samples").is_some());

            let samples = parsed.get("samples").unwrap();

            // Check we got beta_0 and beta_1
            assert!(samples.get("beta_0").is_some(), "Missing beta_0");
            assert!(samples.get("beta_1").is_some(), "Missing beta_1");

            // beta_1 should be positive (higher x -> higher p)
            let beta_1: Vec<f64> = samples
                .get("beta_1")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let beta_1_mean: f64 = beta_1.iter().sum::<f64>() / beta_1.len() as f64;

            assert!(
                beta_1_mean > 0.0,
                "beta_1 mean {} should be positive for this data",
                beta_1_mean
            );
        }

        #[test]
        fn test_hierarchical_model() {
            // Hierarchical (random intercepts) model test
            // Data: 3 groups with different means
            // Group 0: y around 2.0
            // Group 1: y around 4.0
            // Group 2: y around 6.0
            // With a common slope beta = 1.0 for predictor x

            // Observations: (group_id, x, y)
            // Group 0: x=0 -> y=2, x=1 -> y=3
            // Group 1: x=0 -> y=4, x=1 -> y=5
            // Group 2: x=0 -> y=6, x=1 -> y=7

            let model_json = serde_json::json!({
                "groupIds": [0, 0, 1, 1, 2, 2],
                "predictors": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                "response": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "numGroups": 3,
                "muAlphaPriorScale": 10.0,
                "sigmaAlphaPriorScale": 5.0,
                "betaPriorScale": 10.0,
                "sigmaPriorScale": 1.0
            })
            .to_string();

            let config_json = r#"{
            "numSamples": 200,
            "numWarmup": 100,
            "numChains": 2,
            "targetAccept": 0.8,
            "seed": 555
        }"#;

            let result = run_hierarchical_model(&model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Should run without error
            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );
            assert!(parsed.get("samples").is_some());

            let samples = parsed.get("samples").unwrap();

            // Check we got the expected parameters:
            // mu_alpha (population mean intercept)
            // sigma_alpha (between-group variance)
            // alpha_0, alpha_1, alpha_2 (group intercepts)
            // beta_0 (slope coefficient)
            // sigma (noise)
            assert!(samples.get("mu_alpha").is_some(), "Missing mu_alpha");
            assert!(samples.get("sigma_alpha").is_some(), "Missing sigma_alpha");
            assert!(samples.get("alpha_0").is_some(), "Missing alpha_0");
            assert!(samples.get("alpha_1").is_some(), "Missing alpha_1");
            assert!(samples.get("alpha_2").is_some(), "Missing alpha_2");
            assert!(samples.get("beta_0").is_some(), "Missing beta_0");
            assert!(samples.get("sigma").is_some(), "Missing sigma");

            // Check group intercepts are in expected order (alpha_0 < alpha_1 < alpha_2)
            let alpha_0: Vec<f64> = samples
                .get("alpha_0")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let alpha_1: Vec<f64> = samples
                .get("alpha_1")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let alpha_2: Vec<f64> = samples
                .get("alpha_2")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();

            let alpha_0_mean: f64 = alpha_0.iter().sum::<f64>() / alpha_0.len() as f64;
            let alpha_1_mean: f64 = alpha_1.iter().sum::<f64>() / alpha_1.len() as f64;
            let alpha_2_mean: f64 = alpha_2.iter().sum::<f64>() / alpha_2.len() as f64;

            // Group means should be approximately 2, 4, 6 respectively
            assert!(
                alpha_0_mean < alpha_1_mean,
                "alpha_0 ({}) should be < alpha_1 ({})",
                alpha_0_mean,
                alpha_1_mean
            );
            assert!(
                alpha_1_mean < alpha_2_mean,
                "alpha_1 ({}) should be < alpha_2 ({})",
                alpha_1_mean,
                alpha_2_mean
            );

            // Slope should be around 1.0
            let beta_0: Vec<f64> = samples
                .get("beta_0")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let beta_0_mean: f64 = beta_0.iter().sum::<f64>() / beta_0.len() as f64;
            assert!(
                (beta_0_mean - 1.0).abs() < 1.5,
                "beta_0 mean {} too far from 1.0",
                beta_0_mean
            );
        }

        #[cfg(all(feature = "direct-gpu", feature = "sync-gpu"))]
        mod gpu_tests {
            use super::*;

            #[test]
            fn test_gpu_logp_and_grad() {
                // Use the global shared GPU context to avoid Metal resource conflicts
                let gpu_ctx_arc = match gpu::sync::GpuContextSync::global() {
                    Some(ctx) => ctx,
                    None => {
                        eprintln!("Skipping GPU test - no GPU");
                        return;
                    }
                };
                let gpu_ctx = gpu_ctx_arc.as_ref();

                // Create a simple Normal-Normal model with enough data to use GPU
                let model_json = r#"{
                "priors": [
                    {
                        "name": "mu",
                        "distribution": {
                            "type": "Normal",
                            "params": {"loc": 0, "scale": 10}
                        }
                    }
                ],
                "likelihood": {
                    "distribution": {
                        "type": "Normal",
                        "params": {"loc": "mu", "scale": 1}
                    },
                    "observed": [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.3, 2.7, 1.6, 2.4,
                                 1.1, 2.1, 1.4, 2.6, 1.9, 2.3, 1.2, 2.8, 1.7, 2.5]
                }
            }"#;

                let model_spec: ModelSpec = serde_json::from_str(model_json).unwrap();
                let device = backend::get_device_or_init();
                let model = DynamicModel::new(model_spec, device);

                // Test can_use_gpu (should be false for small data, need 256+)
                assert!(!model.can_use_gpu(), "Small data should not use GPU");

                // Create model with more data
                let large_data: Vec<f64> = (0..300).map(|i| 2.0 + (i as f64 * 0.001)).collect();
                let model_json_large = format!(
                    r#"{{
                "priors": [
                    {{
                        "name": "mu",
                        "distribution": {{
                            "type": "Normal",
                            "params": {{"loc": 0, "scale": 10}}
                        }}
                    }}
                ],
                "likelihood": {{
                    "distribution": {{
                        "type": "Normal",
                        "params": {{"loc": "mu", "scale": 1}}
                    }},
                    "observed": {:?}
                }}
            }}"#,
                    large_data
                );

                let model_spec_large: ModelSpec = serde_json::from_str(&model_json_large).unwrap();
                let model_large = DynamicModel::new(model_spec_large, device);

                assert!(model_large.can_use_gpu(), "Large data should use GPU");

                // Test logp_and_grad_gpu
                let params = [2.0f32]; // mu = 2.0
                let result = model_large.logp_and_grad_gpu(&params, &gpu_ctx);
                assert!(
                    result.is_ok(),
                    "GPU logp_and_grad should succeed: {:?}",
                    result.err()
                );

                let (logp, grad) = result.unwrap();
                // logp should be negative (it's a log probability)
                assert!(
                    logp < 0.0,
                    "Log probability should be negative, got {}",
                    logp
                );
                // Gradient should be approximately 0 when mu matches data mean (~2.0)
                // Since data mean is ~2.15, gradient should be small positive
                assert!(grad.len() == 1, "Should have one gradient");
                // The gradient d/dmu sum_i log N(y_i | mu, 1) = sum_i (y_i - mu)
                // For data centered around 2.15 with mu=2.0, this is positive
                println!("GPU logp_and_grad: logp={}, grad={:?}", logp, grad);
            }

            #[test]
            fn test_gpu_vs_cpu_consistency() {
                // Use the global shared GPU context to avoid Metal resource conflicts
                let gpu_ctx_arc = match gpu::sync::GpuContextSync::global() {
                    Some(ctx) => ctx,
                    None => {
                        eprintln!("Skipping GPU test - no GPU");
                        return;
                    }
                };
                let gpu_ctx = gpu_ctx_arc.as_ref();

                // Create model with data above GPU threshold
                let data: Vec<f64> = (0..300).map(|i| 2.0 + 0.1 * ((i % 10) as f64)).collect();
                let model_json = format!(
                    r#"{{
                "priors": [
                    {{
                        "name": "mu",
                        "distribution": {{
                            "type": "Normal",
                            "params": {{"loc": 0, "scale": 10}}
                        }}
                    }}
                ],
                "likelihood": {{
                    "distribution": {{
                        "type": "Normal",
                        "params": {{"loc": "mu", "scale": 1}}
                    }},
                    "observed": {:?}
                }}
            }}"#,
                    data
                );

                let model_spec: ModelSpec = serde_json::from_str(&model_json).unwrap();
                let device = backend::get_device_or_init();
                let model = DynamicModel::new(model_spec, device);

                // Compute using GPU
                let params = [2.5f32];
                let (gpu_logp, gpu_grad) = model.logp_and_grad_gpu(&params, &gpu_ctx).unwrap();

                // Compute using CPU (autodiff)
                let params_tensor =
                    burn::prelude::Tensor::<WasmBackend, 1>::from_floats([2.5f32], &device);
                let (cpu_logp, cpu_grad) =
                    bayesian_sampler::model::logp_and_grad(&model, params_tensor);

                // Compare results (should be very close)
                let logp_diff = (gpu_logp - cpu_logp).abs();
                let grad_diff = (gpu_grad[0] - cpu_grad[0]).abs();

                println!("GPU: logp={}, grad={:?}", gpu_logp, gpu_grad);
                println!("CPU: logp={}, grad={:?}", cpu_logp, cpu_grad);
                println!("Diff: logp={}, grad={}", logp_diff, grad_diff);

                // Allow some numerical tolerance due to f32 vs f64 and different accumulation order
                assert!(
                    logp_diff < 0.1,
                    "Log prob difference too large: {}",
                    logp_diff
                );
                assert!(
                    grad_diff < 0.1,
                    "Gradient difference too large: {}",
                    grad_diff
                );
            }

            #[test]
            fn test_nuts_uses_gpu_via_logp_and_grad_direct() {
                // Verify the full integration chain:
                // logp_and_grad() -> logp_and_grad_direct() -> GPU REDUCE kernels
                //
                // This proves that when NUTS calls logp_and_grad(), it transparently
                // uses GPU for the likelihood computation.
                let data: Vec<f64> = (0..300).map(|i| 3.0 + 0.01 * (i as f64)).collect();
                let model_json = format!(
                    r#"{{
                "priors": [
                    {{
                        "name": "mu",
                        "distribution": {{
                            "type": "Normal",
                            "params": {{"loc": 0, "scale": 10}}
                        }}
                    }}
                ],
                "likelihood": {{
                    "distribution": {{
                        "type": "Normal",
                        "params": {{"loc": "mu", "scale": 1}}
                    }},
                    "observed": {:?}
                }}
            }}"#,
                    data
                );

                let model_spec: ModelSpec = serde_json::from_str(&model_json).unwrap();
                let device = backend::get_device_or_init();
                let model = DynamicModel::new(model_spec, device);

                // 1. GPU context initialized
                assert!(model.gpu_ctx.is_some(), "GPU context should be initialized");

                // 2. GPU path active for large data
                assert!(
                    model.can_use_gpu(),
                    "Model should use GPU for 300 observations"
                );

                // 3. logp_and_grad_direct returns Some (GPU path taken)
                let test_params = [3.0f32];
                let direct_result = model.logp_and_grad_direct(&test_params);
                assert!(
                    direct_result.is_some(),
                    "logp_and_grad_direct should return Some when GPU is available"
                );

                let (gpu_logp, gpu_grad) = direct_result.unwrap();
                // Sanity check: log_prob should be negative
                assert!(
                    gpu_logp < 0.0,
                    "Log prob should be negative, got {}",
                    gpu_logp
                );
                assert_eq!(gpu_grad.len(), 1, "Should have 1 gradient element");
                assert!(gpu_grad[0].is_finite(), "Gradient should be finite");

                // 4. logp_and_grad() (the sampler entry point) dispatches to GPU
                let params_tensor =
                    burn::prelude::Tensor::<WasmBackend, 1>::from_floats([3.0f32], &device);
                let (sampler_logp, sampler_grad) =
                    bayesian_sampler::model::logp_and_grad(&model, params_tensor);

                // Should match GPU result exactly (same code path)
                assert!(
                    (sampler_logp - gpu_logp).abs() < 1e-6,
                    "logp_and_grad should dispatch to GPU: sampler={}, gpu={}",
                    sampler_logp,
                    gpu_logp
                );
                assert!(
                    (sampler_grad[0] - gpu_grad[0]).abs() < 1e-6,
                    "Gradient should match: sampler={}, gpu={}",
                    sampler_grad[0],
                    gpu_grad[0]
                );

                println!("GPU-NUTS integration verified:");
                println!("  logp_and_grad -> logp_and_grad_direct -> GPU REDUCE kernels");
                println!("  logp={:.4}, grad={:.4}", gpu_logp, gpu_grad[0]);
            }
        }

        // ========================================================================
        // Prediction Utilities Tests
        // ========================================================================

        #[test]
        fn test_posterior_mean() {
            // Create mock samples
            let samples_json = r#"{
            "samples": {
                "beta_0": [1.0, 2.0, 3.0, 4.0],
                "beta_1": [0.5, 0.5, 1.0, 1.0],
                "sigma": [0.1, 0.2, 0.1, 0.2]
            }
        }"#;

            let result = posterior_mean(samples_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Should have means for each parameter
            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );

            let beta_0_mean = parsed.get("beta_0").unwrap().as_f64().unwrap();
            let beta_1_mean = parsed.get("beta_1").unwrap().as_f64().unwrap();
            let sigma_mean = parsed.get("sigma").unwrap().as_f64().unwrap();

            assert!(
                (beta_0_mean - 2.5).abs() < 1e-10,
                "beta_0 mean should be 2.5, got {}",
                beta_0_mean
            );
            assert!(
                (beta_1_mean - 0.75).abs() < 1e-10,
                "beta_1 mean should be 0.75, got {}",
                beta_1_mean
            );
            assert!(
                (sigma_mean - 0.15).abs() < 1e-10,
                "sigma mean should be 0.15, got {}",
                sigma_mean
            );
        }

        #[test]
        fn test_credible_interval() {
            // Create mock samples (sorted for easy quantile verification)
            let samples_json = r#"{
            "samples": {
                "theta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        }"#;

            // 90% credible interval
            let prob_json = r#"{"prob": 0.90}"#;
            let result = credible_interval(samples_json, prob_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );

            let theta_ci = parsed.get("theta").unwrap().as_array().unwrap();
            let lower = theta_ci[0].as_f64().unwrap();
            let upper = theta_ci[1].as_f64().unwrap();

            // 5th and 95th percentiles of [0.1..1.0] should be around 0.145 and 0.955
            assert!(lower < 0.3, "Lower bound {} should be around 0.15", lower);
            assert!(upper > 0.7, "Upper bound {} should be around 0.95", upper);
        }

        #[test]
        fn test_credible_interval_default_95() {
            // Test default 95% interval
            let samples_json = r#"{
            "samples": {
                "mu": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            }
        }"#;

            // No prob specified - should default to 0.95
            let result = credible_interval(samples_json, "{}");
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );
            assert!(parsed.get("mu").is_some(), "Should have mu interval");
        }

        #[test]
        fn test_predict_linear_regression() {
            // Mock samples from a linear regression with known coefficients
            // y = beta_0 + beta_1 * x + noise
            // beta_0 ~ 2.0, beta_1 ~ 3.0, sigma ~ 0.1
            let samples_json = r#"{
            "samples": {
                "beta_0": [2.0, 2.0, 2.0, 2.0],
                "beta_1": [3.0, 3.0, 3.0, 3.0],
                "sigma": [0.1, 0.1, 0.1, 0.1]
            }
        }"#;

            // New data: x = [0, 1, 2] with intercept column
            let new_x_json = r#"{
            "predictors": [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
        }"#;

            let result = predict_linear_regression(samples_json, new_x_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );

            // Mean predictions: y = 2 + 3*x -> [2, 5, 8]
            let mean_preds = parsed.get("mean").unwrap().as_array().unwrap();
            assert_eq!(mean_preds.len(), 3);
            assert!((mean_preds[0].as_f64().unwrap() - 2.0).abs() < 1e-6);
            assert!((mean_preds[1].as_f64().unwrap() - 5.0).abs() < 1e-6);
            assert!((mean_preds[2].as_f64().unwrap() - 8.0).abs() < 1e-6);

            // Should also have intervals
            assert!(parsed.get("lower").is_some());
            assert!(parsed.get("upper").is_some());
        }

        #[test]
        fn test_predict_logistic_regression() {
            // Mock samples from logistic regression
            // p = sigmoid(beta_0 + beta_1 * x)
            // With beta_0 = 0, beta_1 = 2, at x=0: p = 0.5, at x=1: p = sigmoid(2) ~ 0.88
            let samples_json = r#"{
            "samples": {
                "beta_0": [0.0, 0.0, 0.0, 0.0],
                "beta_1": [2.0, 2.0, 2.0, 2.0]
            }
        }"#;

            // New data: x = [0, 1] with intercept column
            let new_x_json = r#"{
            "predictors": [[1.0, 0.0], [1.0, 1.0]]
        }"#;

            let result = predict_logistic_regression(samples_json, new_x_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );

            // Mean probability predictions
            let mean_probs = parsed.get("mean").unwrap().as_array().unwrap();
            assert_eq!(mean_probs.len(), 2);

            // At x=0: p = sigmoid(0) = 0.5
            assert!((mean_probs[0].as_f64().unwrap() - 0.5).abs() < 1e-6);

            // At x=1: p = sigmoid(2) ≈ 0.8808
            let sigmoid_2 = 1.0 / (1.0 + (-2.0_f64).exp());
            assert!((mean_probs[1].as_f64().unwrap() - sigmoid_2).abs() < 1e-6);
        }

        #[test]
        fn test_predict_hierarchical() {
            // Mock samples from hierarchical model
            // y = alpha_g + beta * x + noise
            // 2 groups: alpha_0 = 1.0, alpha_1 = 3.0, beta = 2.0
            let samples_json = r#"{
            "samples": {
                "mu_alpha": [2.0, 2.0],
                "sigma_alpha": [1.0, 1.0],
                "alpha_0": [1.0, 1.0],
                "alpha_1": [3.0, 3.0],
                "beta_0": [2.0, 2.0],
                "sigma": [0.1, 0.1]
            }
        }"#;

            // New data: 2 observations
            // Obs 0: group 0, x=1 -> y = 1 + 2*1 = 3
            // Obs 1: group 1, x=1 -> y = 3 + 2*1 = 5
            let new_x_json = r#"{
            "predictors": [[1.0], [1.0]],
            "groupIds": [0, 1]
        }"#;

            let result = predict_hierarchical(samples_json, new_x_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );

            let mean_preds = parsed.get("mean").unwrap().as_array().unwrap();
            assert_eq!(mean_preds.len(), 2);

            // y = alpha_g + beta * x
            assert!(
                (mean_preds[0].as_f64().unwrap() - 3.0).abs() < 1e-6,
                "Group 0 pred should be 3.0"
            );
            assert!(
                (mean_preds[1].as_f64().unwrap() - 5.0).abs() < 1e-6,
                "Group 1 pred should be 5.0"
            );
        }

        // ========================================================================
        // Survival Analysis Tests
        // ========================================================================

        #[test]
        fn test_exponential_survival() {
            // Exponential survival model with right-censored data
            // True rate λ = 0.5, so mean survival time = 2.0
            // Mix of events and censored observations
            let model_json = serde_json::json!({
                "times": [1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 0.5, 1.0],
                "events": [1, 1, 0, 1, 0, 0, 1, 1],  // 1 = event, 0 = censored
                "ratePriorScale": 1.0
            })
            .to_string();

            let config_json = r#"{
            "numSamples": 200,
            "numWarmup": 100,
            "numChains": 2,
            "targetAccept": 0.8,
            "seed": 42
        }"#;

            let result = run_exponential_survival(&model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Should run without error
            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );
            assert!(parsed.get("samples").is_some());

            let samples = parsed.get("samples").unwrap();

            // Should have rate parameter
            assert!(samples.get("rate").is_some(), "Missing rate parameter");

            // Rate should be positive and reasonable
            let rate: Vec<f64> = samples
                .get("rate")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let rate_mean: f64 = rate.iter().sum::<f64>() / rate.len() as f64;

            assert!(rate_mean > 0.0, "Rate should be positive");
            // With data centered around mean time ~2, rate should be around 0.5
            assert!(
                rate_mean > 0.1 && rate_mean < 2.0,
                "Rate mean {} should be in reasonable range",
                rate_mean
            );
        }

        #[test]
        fn test_weibull_survival() {
            // Weibull survival model
            // Times generated from Weibull with scale=2.0, shape=1.5
            let model_json = serde_json::json!({
                "times": [1.5, 2.0, 2.5, 1.0, 3.0, 2.2, 1.8, 2.8],
                "events": [1, 1, 1, 1, 0, 1, 1, 0],
                "scalePriorScale": 2.0,
                "shapePriorScale": 1.0
            })
            .to_string();

            let config_json = r#"{
            "numSamples": 200,
            "numWarmup": 100,
            "numChains": 2,
            "targetAccept": 0.8,
            "seed": 123
        }"#;

            let result = run_weibull_survival(&model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            // Should run without error
            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );
            assert!(parsed.get("samples").is_some());

            let samples = parsed.get("samples").unwrap();

            // Should have scale and shape parameters
            assert!(samples.get("scale").is_some(), "Missing scale parameter");
            assert!(samples.get("shape").is_some(), "Missing shape parameter");

            // Both should be positive
            let scale: Vec<f64> = samples
                .get("scale")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let shape: Vec<f64> = samples
                .get("shape")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();

            let scale_mean: f64 = scale.iter().sum::<f64>() / scale.len() as f64;
            let shape_mean: f64 = shape.iter().sum::<f64>() / shape.len() as f64;

            assert!(
                scale_mean > 0.0,
                "Scale should be positive, got {}",
                scale_mean
            );
            assert!(
                shape_mean > 0.0,
                "Shape should be positive, got {}",
                shape_mean
            );
        }

        #[test]
        fn test_exponential_survival_with_covariates() {
            // Exponential model with covariates: λ_i = λ * exp(X_i @ beta)
            let model_json = serde_json::json!({
                "times": [1.0, 0.5, 2.0, 1.5, 3.0, 0.8, 1.2, 2.5],
                "events": [1, 1, 1, 1, 0, 1, 1, 0],
                "predictors": [[1.0], [1.0], [-1.0], [-1.0], [-1.0], [1.0], [0.0], [0.0]],
                "ratePriorScale": 1.0,
                "betaPriorScale": 1.0
            })
            .to_string();

            let config_json = r#"{
            "numSamples": 200,
            "numWarmup": 100,
            "numChains": 2,
            "seed": 456
        }"#;

            let result = run_exponential_survival(&model_json, config_json);
            let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

            assert!(
                parsed.get("error").is_none(),
                "Got error: {:?}",
                parsed.get("error")
            );

            let samples = parsed.get("samples").unwrap();
            assert!(samples.get("rate").is_some(), "Missing rate");
            assert!(samples.get("beta_0").is_some(), "Missing beta_0");
        }
    }
} // mod wasm_impl

#[cfg(any(feature = "ndarray", feature = "wgpu"))]
pub use wasm_impl::*;
