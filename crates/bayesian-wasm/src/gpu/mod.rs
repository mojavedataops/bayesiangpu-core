//! Direct wgpu compute shaders for GPU-accelerated inference
//!
//! This module provides a Burn-free path to GPU compute using direct wgpu
//! with hand-written WGSL shaders. This works around browser WASM limitations
//! in Burn's Autodiff<Wgpu> backend.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │       GpuContext                │
//! │  (device, queue, pipelines)     │
//! ├─────────────────────────────────┤
//! │       WGSL Kernels              │
//! │  - normal_log_prob              │
//! │  - normal_grad_log_prob         │
//! │  - reduction_sum                │
//! └─────────────────────────────────┘
//! ```

mod context;
mod kernels;

#[cfg(feature = "sync-gpu")]
pub mod sync;

pub use context::GpuContext;
pub use context::PersistentGpuBuffers;
pub use kernels::{
    BernoulliReduceParams, BetaReduceParams, BinomialReduceParams, CategoricalReduceParams,
    CauchyReduceParams, ExponentialReduceParams, GammaReduceParams, GpuBatchResult,
    GpuGradReduceResult, GpuReduceResult, GpuResult, HalfNormalParams, HalfNormalReduceParams,
    InverseGammaReduceParams, LogNormalReduceParams, NegativeBinomialReduceParams,
    NormalBatchParams, NormalParams, PoissonReduceParams, StudentTReduceParams,
    UniformReduceParams,
};

use js_sys::{Float32Array, Promise};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

// Thread-local storage for GPU context (WASM is single-threaded)
thread_local! {
    static GPU_CONTEXT: RefCell<Option<GpuContext>> = const { RefCell::new(None) };
}

/// Initialize the GPU compute context
///
/// This must be called before any GPU operations. Returns a Promise
/// that resolves to "gpu_initialized" on success.
#[wasm_bindgen]
pub fn init_gpu_context() -> Promise {
    future_to_promise(async move {
        match GpuContext::new().await {
            Ok(ctx) => {
                GPU_CONTEXT.with(|cell| {
                    *cell.borrow_mut() = Some(ctx);
                });
                Ok(JsValue::from_str("gpu_initialized"))
            }
            Err(e) => Err(JsValue::from_str(&format!("GPU init failed: {}", e))),
        }
    })
}

/// Check if GPU context is initialized
#[wasm_bindgen]
pub fn is_gpu_initialized() -> bool {
    GPU_CONTEXT.with(|cell| cell.borrow().is_some())
}

/// Test the Normal distribution GPU kernel
///
/// Computes log_prob and gradient for Normal(mu, sigma) at point x.
/// Returns a Promise that resolves to JSON: { "log_prob": f32, "grad": f32 }
#[wasm_bindgen]
pub fn test_normal_kernel(x: f32, mu: f32, sigma: f32) -> Promise {
    web_sys::console::log_1(
        &format!(
            "[test_normal_kernel] Called with x={}, mu={}, sigma={}",
            x, mu, sigma
        )
        .into(),
    );

    // Get context reference - we need to work around RefCell borrowing
    let result = GPU_CONTEXT.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(ctx) => {
                web_sys::console::log_1(&"[test_normal_kernel] GPU context found".into());
                Some((
                    ctx.device_clone(),
                    ctx.queue_clone(),
                    ctx.normal_pipeline_clone(),
                    ctx.normal_bind_group_layout_clone(),
                ))
            }
            None => {
                web_sys::console::log_1(&"[test_normal_kernel] GPU context NOT found".into());
                None
            }
        }
    });

    future_to_promise(async move {
        web_sys::console::log_1(&"[test_normal_kernel] Inside future".into());

        let (device, queue, pipeline, layout) = result.ok_or_else(|| {
            JsValue::from_str("GPU not initialized. Call init_gpu_context() first.")
        })?;

        web_sys::console::log_1(&"[test_normal_kernel] Calling run_normal_kernel".into());

        match context::run_normal_kernel(&device, &queue, &pipeline, &layout, x, mu, sigma).await {
            Ok(result) => {
                web_sys::console::log_1(
                    &format!(
                        "[test_normal_kernel] Success: log_prob={}, grad={}",
                        result.log_prob, result.grad
                    )
                    .into(),
                );
                let json = serde_json::json!({
                    "log_prob": result.log_prob,
                    "grad": result.grad
                });
                Ok(JsValue::from_str(&json.to_string()))
            }
            Err(e) => {
                web_sys::console::log_1(&format!("[test_normal_kernel] Error: {}", e).into());
                Err(JsValue::from_str(&format!("Kernel failed: {}", e)))
            }
        }
    })
}

/// Test the HalfNormal distribution GPU kernel
///
/// Computes log_prob and gradient for HalfNormal(sigma) at point x >= 0.
/// Returns a Promise that resolves to JSON: { "log_prob": f32, "grad": f32 }
#[wasm_bindgen]
pub fn test_half_normal_kernel(x: f32, sigma: f32) -> Promise {
    let result = GPU_CONTEXT.with(|cell| {
        let borrow = cell.borrow();
        borrow.as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.half_normal_pipeline_clone(),
                ctx.half_normal_bind_group_layout_clone(),
            )
        })
    });

    future_to_promise(async move {
        let (device, queue, pipeline, layout) = result.ok_or_else(|| {
            JsValue::from_str("GPU not initialized. Call init_gpu_context() first.")
        })?;

        match context::run_half_normal_kernel(&device, &queue, &pipeline, &layout, x, sigma).await {
            Ok(result) => {
                let json = serde_json::json!({
                    "log_prob": result.log_prob,
                    "grad": result.grad
                });
                Ok(JsValue::from_str(&json.to_string()))
            }
            Err(e) => Err(JsValue::from_str(&format!("Kernel failed: {}", e))),
        }
    })
}

/// Run batched Normal distribution GPU kernel
///
/// Processes multiple x values in parallel with shared mu, sigma.
/// This is the efficient path for likelihood computation.
///
/// Args:
///   x_values: Float32Array of x values
///   mu: distribution mean
///   sigma: distribution standard deviation
///
/// Returns: Promise resolving to JSON { "log_probs": [...], "grads": [...], "count": N, "time_ms": T }
#[wasm_bindgen]
pub fn run_normal_batch(x_values: Float32Array, mu: f32, sigma: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();

    let result = GPU_CONTEXT.with(|cell| {
        let borrow = cell.borrow();
        borrow.as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.normal_batch_pipeline_clone(),
                ctx.normal_batch_bind_group_layout_clone(),
            )
        })
    });

    future_to_promise(async move {
        let (device, queue, pipeline, layout) = result.ok_or_else(|| {
            JsValue::from_str("GPU not initialized. Call init_gpu_context() first.")
        })?;

        match context::run_normal_batch_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, mu, sigma,
        )
        .await
        {
            Ok(result) => {
                let json = serde_json::json!({
                    "log_probs": result.log_probs,
                    "grads": result.grads,
                    "count": x_vec.len()
                });
                Ok(JsValue::from_str(&json.to_string()))
            }
            Err(e) => Err(JsValue::from_str(&format!("Batch kernel failed: {}", e))),
        }
    })
}

/// Run Normal distribution REDUCE kernel (realistic MCMC pattern)
///
/// Computes log_prob for all x values AND reduces to a single sum on GPU.
/// Only returns the scalar total - minimal data transfer.
///
/// Args:
///   x_values: Float32Array of x values
///   mu: distribution mean
///   sigma: distribution standard deviation
///
/// Returns: Promise resolving to JSON { "total_log_prob": f32, "count": N }
#[wasm_bindgen]
pub fn run_normal_reduce(x_values: Float32Array, mu: f32, sigma: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();

    let result = GPU_CONTEXT.with(|cell| {
        let borrow = cell.borrow();
        borrow.as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.normal_reduce_pipeline_clone(),
                ctx.normal_reduce_bind_group_layout_clone(),
            )
        })
    });

    future_to_promise(async move {
        let (device, queue, pipeline, layout) = result.ok_or_else(|| {
            JsValue::from_str("GPU not initialized. Call init_gpu_context() first.")
        })?;

        match context::run_normal_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, mu, sigma,
        )
        .await
        {
            Ok(result) => {
                let json = serde_json::json!({
                    "total_log_prob": result.total_log_prob,
                    "count": x_vec.len()
                });
                Ok(JsValue::from_str(&json.to_string()))
            }
            Err(e) => Err(JsValue::from_str(&format!("Reduce kernel failed: {}", e))),
        }
    })
}

/// Run HalfNormal distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_half_normal_reduce(x_values: Float32Array, sigma: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.half_normal_reduce_pipeline_clone(),
                ctx.half_normal_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_half_normal_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, sigma,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "HalfNormal reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Exponential distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_exponential_reduce(x_values: Float32Array, lambda: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.exponential_reduce_pipeline_clone(),
                ctx.exponential_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_exponential_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, lambda,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "Exponential reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Gamma distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_gamma_reduce(x_values: Float32Array, alpha: f32, beta: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.gamma_reduce_pipeline_clone(),
                ctx.gamma_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_gamma_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, alpha, beta,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!("Gamma reduce failed: {}", e))),
        }
    })
}

/// Run Beta distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_beta_reduce(x_values: Float32Array, alpha: f32, beta: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.beta_reduce_pipeline_clone(),
                ctx.beta_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_beta_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, alpha, beta,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!("Beta reduce failed: {}", e))),
        }
    })
}

/// Run InverseGamma distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_inverse_gamma_reduce(x_values: Float32Array, alpha: f32, beta: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.inverse_gamma_reduce_pipeline_clone(),
                ctx.inverse_gamma_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_inverse_gamma_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, alpha, beta,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "InverseGamma reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Uniform distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_uniform_reduce(x_values: Float32Array, low: f32, high: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.uniform_reduce_pipeline_clone(),
                ctx.uniform_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_uniform_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, low, high,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!("Uniform reduce failed: {}", e))),
        }
    })
}

/// Run Cauchy distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_cauchy_reduce(x_values: Float32Array, loc: f32, scale: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.cauchy_reduce_pipeline_clone(),
                ctx.cauchy_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_cauchy_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, loc, scale,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!("Cauchy reduce failed: {}", e))),
        }
    })
}

/// Run StudentT distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_student_t_reduce(x_values: Float32Array, loc: f32, scale: f32, nu: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.student_t_reduce_pipeline_clone(),
                ctx.student_t_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_student_t_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, loc, scale, nu,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!("StudentT reduce failed: {}", e))),
        }
    })
}

/// Run LogNormal distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_lognormal_reduce(x_values: Float32Array, mu: f32, sigma: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.lognormal_reduce_pipeline_clone(),
                ctx.lognormal_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_lognormal_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, mu, sigma,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "LogNormal reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Bernoulli distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_bernoulli_reduce(x_values: Float32Array, p: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.bernoulli_reduce_pipeline_clone(),
                ctx.bernoulli_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_bernoulli_reduce_kernel(&device, &queue, &pipeline, &layout, &x_vec, p)
            .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "Bernoulli reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Binomial distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_binomial_reduce(x_values: Float32Array, n: f32, p: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.binomial_reduce_pipeline_clone(),
                ctx.binomial_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_binomial_reduce_kernel(&device, &queue, &pipeline, &layout, &x_vec, n, p)
            .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!("Binomial reduce failed: {}", e))),
        }
    })
}

/// Run Poisson distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_poisson_reduce(x_values: Float32Array, lambda: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.poisson_reduce_pipeline_clone(),
                ctx.poisson_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_poisson_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, lambda,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!("Poisson reduce failed: {}", e))),
        }
    })
}

/// Run NegativeBinomial distribution REDUCE kernel
#[wasm_bindgen]
pub fn run_negative_binomial_reduce(x_values: Float32Array, r: f32, p: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.negative_binomial_reduce_pipeline_clone(),
                ctx.negative_binomial_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_negative_binomial_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, r, p,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "NegativeBinomial reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Categorical distribution REDUCE kernel
///
/// Computes log_prob for categorical observations and reduces to a single sum.
/// Takes x_values (category indices as f32) and probs (probability vector).
#[wasm_bindgen]
pub fn run_categorical_reduce(x_values: Float32Array, probs: Float32Array) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let probs_vec: Vec<f32> = probs.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.categorical_reduce_pipeline_clone(),
                ctx.categorical_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_categorical_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, &probs_vec,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_log_prob": result.total_log_prob, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "Categorical reduce failed: {}",
                e
            ))),
        }
    })
}

// =============================================================================
// GRADIENT REDUCE KERNELS
// =============================================================================

/// Run Normal distribution GRAD REDUCE kernel
///
/// Computes grad_log_prob for all x values and reduces to a single sum.
/// Returns the scalar sum of gradients.
#[wasm_bindgen]
pub fn run_normal_grad_reduce(x_values: Float32Array, mu: f32, sigma: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.normal_grad_reduce_pipeline_clone(),
                ctx.normal_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_normal_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, mu, sigma,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "Normal grad reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run HalfNormal distribution GRAD REDUCE kernel
#[wasm_bindgen]
pub fn run_half_normal_grad_reduce(x_values: Float32Array, sigma: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.half_normal_grad_reduce_pipeline_clone(),
                ctx.half_normal_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_half_normal_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, sigma,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "HalfNormal grad reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Exponential distribution GRAD REDUCE kernel
#[wasm_bindgen]
pub fn run_exponential_grad_reduce(x_values: Float32Array, lambda: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.exponential_grad_reduce_pipeline_clone(),
                ctx.exponential_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_exponential_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, lambda,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "Exponential grad reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Beta distribution GRAD REDUCE kernel
#[wasm_bindgen]
pub fn run_beta_grad_reduce(x_values: Float32Array, alpha: f32, beta: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.beta_grad_reduce_pipeline_clone(),
                ctx.beta_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_beta_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, alpha, beta,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "Beta grad reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Gamma distribution GRAD REDUCE kernel
#[wasm_bindgen]
pub fn run_gamma_grad_reduce(x_values: Float32Array, alpha: f32, beta: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.gamma_grad_reduce_pipeline_clone(),
                ctx.gamma_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_gamma_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, alpha, beta,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "Gamma grad reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run InverseGamma distribution GRAD REDUCE kernel
#[wasm_bindgen]
pub fn run_inverse_gamma_grad_reduce(x_values: Float32Array, alpha: f32, beta: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.inverse_gamma_grad_reduce_pipeline_clone(),
                ctx.inverse_gamma_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_inverse_gamma_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, alpha, beta,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "InverseGamma grad reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run StudentT distribution GRAD REDUCE kernel
#[wasm_bindgen]
pub fn run_student_t_grad_reduce(x_values: Float32Array, loc: f32, scale: f32, nu: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.student_t_grad_reduce_pipeline_clone(),
                ctx.student_t_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_student_t_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, loc, scale, nu,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "StudentT grad reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run Cauchy distribution GRAD REDUCE kernel
#[wasm_bindgen]
pub fn run_cauchy_grad_reduce(x_values: Float32Array, loc: f32, scale: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.cauchy_grad_reduce_pipeline_clone(),
                ctx.cauchy_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_cauchy_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, loc, scale,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "Cauchy grad reduce failed: {}",
                e
            ))),
        }
    })
}

/// Run LogNormal distribution GRAD REDUCE kernel
#[wasm_bindgen]
pub fn run_lognormal_grad_reduce(x_values: Float32Array, mu: f32, sigma: f32) -> Promise {
    let x_vec: Vec<f32> = x_values.to_vec();
    let result = GPU_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|ctx| {
            (
                ctx.device_clone(),
                ctx.queue_clone(),
                ctx.lognormal_grad_reduce_pipeline_clone(),
                ctx.lognormal_grad_reduce_bind_group_layout_clone(),
            )
        })
    });
    future_to_promise(async move {
        let (device, queue, pipeline, layout) =
            result.ok_or_else(|| JsValue::from_str("GPU not initialized"))?;
        match context::run_lognormal_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, &x_vec, mu, sigma,
        )
        .await
        {
            Ok(result) => Ok(JsValue::from_str(
                &serde_json::json!({"total_grad": result.total_grad, "count": x_vec.len()})
                    .to_string(),
            )),
            Err(e) => Err(JsValue::from_str(&format!(
                "LogNormal grad reduce failed: {}",
                e
            ))),
        }
    })
}
