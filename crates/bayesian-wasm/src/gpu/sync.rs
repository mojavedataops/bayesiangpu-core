//! Synchronous wrappers for GPU kernel execution
//!
//! This module provides synchronous (blocking) versions of the async GPU kernel runners.
//! These are intended for native Rust usage (not WASM) where blocking on GPU operations
//! is acceptable, such as in NUTS sampler integration.
//!
//! # Example
//!
//! ```ignore
//! use bayesian_wasm::gpu::sync::*;
//!
//! // Initialize GPU context synchronously
//! let ctx = GpuContextSync::new()?;
//!
//! // Run kernel synchronously
//! let result = ctx.run_normal_reduce(&data, mu, sigma)?;
//! println!("Total log prob: {}", result);
//! ```

use std::sync::{Arc, OnceLock};
use wgpu::{BindGroupLayout, ComputePipeline, Device, Queue};

use super::context::{self, GpuContext, PersistentGpuBuffers};
use super::kernels::{
    BernoulliReduceParams, BetaReduceParams, BinomialReduceParams, CauchyReduceParams,
    ExponentialReduceParams, FusedLogpGradResult, GammaReduceParams, GpuBatchResult,
    GpuGradReduceResult, GpuReduceResult, GpuResult, HalfNormalReduceParams,
    InverseGammaReduceParams, LogNormalReduceParams, NegativeBinomialReduceParams,
    NormalBatchParams, PoissonReduceParams, StudentTReduceParams, UniformReduceParams,
};

/// Block on an async future using a shared tokio runtime.
///
/// pollster::block_on hangs on macOS Metal because it parks the thread waiting
/// for a waker, but wgpu's Metal backend needs an active event loop to process
/// adapter requests. tokio's multi-threaded runtime provides this.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    let rt = RT.get_or_init(|| {
        tokio::runtime::Runtime::new().expect("Failed to create tokio runtime for GPU sync")
    });
    rt.block_on(f)
}

/// Synchronous GPU context wrapper
///
/// Wraps the async GpuContext and provides synchronous methods for all kernel operations.
/// Uses a shared tokio runtime to block on async operations.
pub struct GpuContextSync {
    inner: GpuContext,
}

impl GpuContextSync {
    /// Create a new synchronous GPU context
    ///
    /// Blocks until the GPU context is initialized.
    pub fn new() -> Result<Self, String> {
        let inner = block_on(GpuContext::new())?;
        Ok(Self { inner })
    }

    /// Get or create a shared global GPU context.
    ///
    /// Multiple wgpu `Device` instances on the same Metal adapter don't play well
    /// together under concurrent access (e.g. parallel test threads). This singleton
    /// ensures all GPU work goes through a single device/queue pair.
    pub fn global() -> Option<Arc<Self>> {
        static GLOBAL_CTX: OnceLock<Option<Arc<GpuContextSync>>> = OnceLock::new();
        GLOBAL_CTX
            .get_or_init(|| match GpuContextSync::new() {
                Ok(ctx) => Some(Arc::new(ctx)),
                Err(_) => None,
            })
            .clone()
    }

    /// Get a reference to the underlying device
    pub fn device(&self) -> Arc<Device> {
        self.inner.device_clone()
    }

    /// Get a reference to the underlying queue
    pub fn queue(&self) -> Arc<Queue> {
        self.inner.queue_clone()
    }

    // ==================== Persistent GPU Buffers ====================

    /// Create persistent GPU buffers for repeated kernel execution.
    ///
    /// Observation data is uploaded once. The returned buffers can be passed
    /// to the `_persistent` methods to avoid per-call buffer allocation.
    pub fn create_persistent_buffers(
        &self,
        x_values: &[f32],
        max_params_size: u64,
    ) -> PersistentGpuBuffers {
        context::create_persistent_buffers(
            self.inner.device_ref(),
            self.inner.queue_ref(),
            x_values,
            max_params_size,
        )
    }

    // ==================== Persistent Reduce kernels (log_prob sum) ====================

    /// Run Normal reduce using persistent buffers (fast path).
    pub fn run_normal_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_reduce_pipeline_clone();
        let layout = self.inner.normal_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            NormalBatchParams {
                mu,
                sigma,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run HalfNormal reduce using persistent buffers (fast path).
    pub fn run_half_normal_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.half_normal_reduce_pipeline_clone();
        let layout = self.inner.half_normal_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            HalfNormalReduceParams {
                sigma,
                count: buffers.count,
                _padding1: 0,
                _padding2: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Exponential reduce using persistent buffers (fast path).
    pub fn run_exponential_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        lambda: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.exponential_reduce_pipeline_clone();
        let layout = self.inner.exponential_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            ExponentialReduceParams {
                lambda,
                count: buffers.count,
                _padding1: 0,
                _padding2: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Gamma reduce using persistent buffers (fast path).
    pub fn run_gamma_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.gamma_reduce_pipeline_clone();
        let layout = self.inner.gamma_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            GammaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Beta reduce using persistent buffers (fast path).
    pub fn run_beta_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.beta_reduce_pipeline_clone();
        let layout = self.inner.beta_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            BetaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run InverseGamma reduce using persistent buffers (fast path).
    pub fn run_inverse_gamma_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.inverse_gamma_reduce_pipeline_clone();
        let layout = self.inner.inverse_gamma_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            InverseGammaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Uniform reduce using persistent buffers (fast path).
    pub fn run_uniform_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        low: f32,
        high: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.uniform_reduce_pipeline_clone();
        let layout = self.inner.uniform_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            UniformReduceParams {
                low,
                high,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Cauchy reduce using persistent buffers (fast path).
    pub fn run_cauchy_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.cauchy_reduce_pipeline_clone();
        let layout = self.inner.cauchy_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            CauchyReduceParams {
                loc,
                scale,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run StudentT reduce using persistent buffers (fast path).
    pub fn run_student_t_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
        nu: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.student_t_reduce_pipeline_clone();
        let layout = self.inner.student_t_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            StudentTReduceParams {
                loc,
                scale,
                nu,
                count: buffers.count,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run LogNormal reduce using persistent buffers (fast path).
    pub fn run_lognormal_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.lognormal_reduce_pipeline_clone();
        let layout = self.inner.lognormal_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            LogNormalReduceParams {
                mu,
                sigma,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Bernoulli reduce using persistent buffers (fast path).
    pub fn run_bernoulli_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        p: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.bernoulli_reduce_pipeline_clone();
        let layout = self.inner.bernoulli_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            BernoulliReduceParams {
                p,
                count: buffers.count,
                _padding1: 0,
                _padding2: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Binomial reduce using persistent buffers (fast path).
    pub fn run_binomial_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        n: f32,
        p: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.binomial_reduce_pipeline_clone();
        let layout = self.inner.binomial_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            BinomialReduceParams {
                n,
                p,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Poisson reduce using persistent buffers (fast path).
    pub fn run_poisson_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        lambda: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.poisson_reduce_pipeline_clone();
        let layout = self.inner.poisson_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            PoissonReduceParams {
                lambda,
                count: buffers.count,
                _padding1: 0,
                _padding2: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run NegativeBinomial reduce using persistent buffers (fast path).
    pub fn run_negative_binomial_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        r: f32,
        p: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.negative_binomial_reduce_pipeline_clone();
        let layout = self
            .inner
            .negative_binomial_reduce_bind_group_layout_clone();
        block_on(context::run_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            NegativeBinomialReduceParams {
                r,
                p,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_log_prob)
    }

    // ==================== Persistent Gradient Reduce kernels ====================

    /// Run Normal grad reduce using persistent buffers (fast path).
    pub fn run_normal_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_grad_reduce_pipeline_clone();
        let layout = self.inner.normal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            NormalBatchParams {
                mu,
                sigma,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_grad)
    }

    /// Run HalfNormal grad reduce using persistent buffers (fast path).
    pub fn run_half_normal_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.half_normal_grad_reduce_pipeline_clone();
        let layout = self.inner.half_normal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            HalfNormalReduceParams {
                sigma,
                count: buffers.count,
                _padding1: 0,
                _padding2: 0,
            },
        ))
        .map(|r| r.total_grad)
    }

    /// Run Exponential grad reduce using persistent buffers (fast path).
    pub fn run_exponential_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        lambda: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.exponential_grad_reduce_pipeline_clone();
        let layout = self.inner.exponential_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            ExponentialReduceParams {
                lambda,
                count: buffers.count,
                _padding1: 0,
                _padding2: 0,
            },
        ))
        .map(|r| r.total_grad)
    }

    /// Run Beta grad reduce using persistent buffers (fast path).
    pub fn run_beta_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.beta_grad_reduce_pipeline_clone();
        let layout = self.inner.beta_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            BetaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_grad)
    }

    /// Run Gamma grad reduce using persistent buffers (fast path).
    pub fn run_gamma_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.gamma_grad_reduce_pipeline_clone();
        let layout = self.inner.gamma_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            GammaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_grad)
    }

    /// Run InverseGamma grad reduce using persistent buffers (fast path).
    pub fn run_inverse_gamma_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.inverse_gamma_grad_reduce_pipeline_clone();
        let layout = self
            .inner
            .inverse_gamma_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            InverseGammaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_grad)
    }

    /// Run StudentT grad reduce using persistent buffers (fast path).
    pub fn run_student_t_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
        nu: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.student_t_grad_reduce_pipeline_clone();
        let layout = self.inner.student_t_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            StudentTReduceParams {
                loc,
                scale,
                nu,
                count: buffers.count,
            },
        ))
        .map(|r| r.total_grad)
    }

    /// Run Cauchy grad reduce using persistent buffers (fast path).
    pub fn run_cauchy_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.cauchy_grad_reduce_pipeline_clone();
        let layout = self.inner.cauchy_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            CauchyReduceParams {
                loc,
                scale,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_grad)
    }

    /// Run LogNormal grad reduce using persistent buffers (fast path).
    pub fn run_lognormal_grad_reduce_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.lognormal_grad_reduce_pipeline_clone();
        let layout = self.inner.lognormal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_grad_reduce_persistent(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            LogNormalReduceParams {
                mu,
                sigma,
                count: buffers.count,
                _padding: 0,
            },
        ))
        .map(|r| r.total_grad)
    }

    // ==================== Fused logp + grad persistent kernels ====================

    /// Run fused Normal logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_normal_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.normal_reduce_pipeline_clone();
        let logp_layout = self.inner.normal_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.normal_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.normal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            NormalBatchParams {
                mu,
                sigma,
                count: buffers.count,
                _padding: 0,
            },
        ))
    }

    /// Run fused HalfNormal logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_half_normal_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        sigma: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.half_normal_reduce_pipeline_clone();
        let logp_layout = self.inner.half_normal_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.half_normal_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.half_normal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            HalfNormalReduceParams {
                sigma,
                count: buffers.count,
                _padding1: 0,
                _padding2: 0,
            },
        ))
    }

    /// Run fused Exponential logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_exponential_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        lambda: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.exponential_reduce_pipeline_clone();
        let logp_layout = self.inner.exponential_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.exponential_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.exponential_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            ExponentialReduceParams {
                lambda,
                count: buffers.count,
                _padding1: 0,
                _padding2: 0,
            },
        ))
    }

    /// Run fused Beta logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_beta_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.beta_reduce_pipeline_clone();
        let logp_layout = self.inner.beta_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.beta_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.beta_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            BetaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
    }

    /// Run fused Gamma logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_gamma_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.gamma_reduce_pipeline_clone();
        let logp_layout = self.inner.gamma_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.gamma_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.gamma_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            GammaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
    }

    /// Run fused InverseGamma logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_inverse_gamma_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.inverse_gamma_reduce_pipeline_clone();
        let logp_layout = self.inner.inverse_gamma_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.inverse_gamma_grad_reduce_pipeline_clone();
        let grad_layout = self
            .inner
            .inverse_gamma_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            InverseGammaReduceParams {
                alpha,
                beta,
                count: buffers.count,
                _padding: 0,
            },
        ))
    }

    /// Run fused StudentT logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_student_t_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
        nu: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.student_t_reduce_pipeline_clone();
        let logp_layout = self.inner.student_t_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.student_t_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.student_t_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            StudentTReduceParams {
                loc,
                scale,
                nu,
                count: buffers.count,
            },
        ))
    }

    /// Run fused Cauchy logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_cauchy_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.cauchy_reduce_pipeline_clone();
        let logp_layout = self.inner.cauchy_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.cauchy_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.cauchy_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            CauchyReduceParams {
                loc,
                scale,
                count: buffers.count,
                _padding: 0,
            },
        ))
    }

    /// Run fused LogNormal logp + grad using persistent buffers (one GPU dispatch).
    pub fn run_lognormal_fused_persistent(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.lognormal_reduce_pipeline_clone();
        let logp_layout = self.inner.lognormal_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.lognormal_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.lognormal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_fused_logp_and_grad_persistent(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            LogNormalReduceParams {
                mu,
                sigma,
                count: buffers.count,
                _padding: 0,
            },
        ))
    }

    // ==================== Single-value kernels ====================

    /// Run Normal distribution kernel synchronously
    ///
    /// Computes log_prob and gradient for Normal(mu, sigma) at point x.
    pub fn run_normal(&self, x: f32, mu: f32, sigma: f32) -> Result<GpuResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_pipeline_clone();
        let layout = self.inner.normal_bind_group_layout_clone();

        block_on(context::run_normal_kernel(
            &device, &queue, &pipeline, &layout, x, mu, sigma,
        ))
    }

    /// Run HalfNormal distribution kernel synchronously
    ///
    /// Computes log_prob and gradient for HalfNormal(sigma) at point x >= 0.
    pub fn run_half_normal(&self, x: f32, sigma: f32) -> Result<GpuResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.half_normal_pipeline_clone();
        let layout = self.inner.half_normal_bind_group_layout_clone();

        block_on(context::run_half_normal_kernel(
            &device, &queue, &pipeline, &layout, x, sigma,
        ))
    }

    // ==================== Batch kernels ====================

    /// Run batched Normal distribution kernel synchronously
    ///
    /// Processes multiple x values in parallel with shared mu, sigma.
    pub fn run_normal_batch(
        &self,
        x_values: &[f32],
        mu: f32,
        sigma: f32,
    ) -> Result<GpuBatchResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_batch_pipeline_clone();
        let layout = self.inner.normal_batch_bind_group_layout_clone();

        block_on(context::run_normal_batch_kernel(
            &device, &queue, &pipeline, &layout, x_values, mu, sigma,
        ))
    }

    // ==================== Reduce kernels (log_prob sum) ====================

    /// Run Normal distribution REDUCE kernel synchronously
    ///
    /// Computes sum of log_prob for all x values. Returns scalar total.
    pub fn run_normal_reduce(&self, x_values: &[f32], mu: f32, sigma: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_reduce_pipeline_clone();
        let layout = self.inner.normal_reduce_bind_group_layout_clone();

        block_on(context::run_normal_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, mu, sigma,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run HalfNormal distribution REDUCE kernel synchronously
    pub fn run_half_normal_reduce(&self, x_values: &[f32], sigma: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.half_normal_reduce_pipeline_clone();
        let layout = self.inner.half_normal_reduce_bind_group_layout_clone();

        block_on(context::run_half_normal_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, sigma,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Exponential distribution REDUCE kernel synchronously
    pub fn run_exponential_reduce(&self, x_values: &[f32], lambda: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.exponential_reduce_pipeline_clone();
        let layout = self.inner.exponential_reduce_bind_group_layout_clone();

        block_on(context::run_exponential_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, lambda,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Gamma distribution REDUCE kernel synchronously
    pub fn run_gamma_reduce(&self, x_values: &[f32], alpha: f32, beta: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.gamma_reduce_pipeline_clone();
        let layout = self.inner.gamma_reduce_bind_group_layout_clone();

        block_on(context::run_gamma_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, alpha, beta,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Beta distribution REDUCE kernel synchronously
    pub fn run_beta_reduce(&self, x_values: &[f32], alpha: f32, beta: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.beta_reduce_pipeline_clone();
        let layout = self.inner.beta_reduce_bind_group_layout_clone();

        block_on(context::run_beta_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, alpha, beta,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run InverseGamma distribution REDUCE kernel synchronously
    pub fn run_inverse_gamma_reduce(
        &self,
        x_values: &[f32],
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.inverse_gamma_reduce_pipeline_clone();
        let layout = self.inner.inverse_gamma_reduce_bind_group_layout_clone();

        block_on(context::run_inverse_gamma_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, alpha, beta,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Uniform distribution REDUCE kernel synchronously
    pub fn run_uniform_reduce(&self, x_values: &[f32], low: f32, high: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.uniform_reduce_pipeline_clone();
        let layout = self.inner.uniform_reduce_bind_group_layout_clone();

        block_on(context::run_uniform_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, low, high,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Cauchy distribution REDUCE kernel synchronously
    pub fn run_cauchy_reduce(&self, x_values: &[f32], loc: f32, scale: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.cauchy_reduce_pipeline_clone();
        let layout = self.inner.cauchy_reduce_bind_group_layout_clone();

        block_on(context::run_cauchy_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, loc, scale,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run StudentT distribution REDUCE kernel synchronously
    pub fn run_student_t_reduce(
        &self,
        x_values: &[f32],
        loc: f32,
        scale: f32,
        nu: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.student_t_reduce_pipeline_clone();
        let layout = self.inner.student_t_reduce_bind_group_layout_clone();

        block_on(context::run_student_t_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, loc, scale, nu,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run LogNormal distribution REDUCE kernel synchronously
    pub fn run_lognormal_reduce(
        &self,
        x_values: &[f32],
        mu: f32,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.lognormal_reduce_pipeline_clone();
        let layout = self.inner.lognormal_reduce_bind_group_layout_clone();

        block_on(context::run_lognormal_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, mu, sigma,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Bernoulli distribution REDUCE kernel synchronously
    pub fn run_bernoulli_reduce(&self, x_values: &[f32], p: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.bernoulli_reduce_pipeline_clone();
        let layout = self.inner.bernoulli_reduce_bind_group_layout_clone();

        block_on(context::run_bernoulli_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, p,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Binomial distribution REDUCE kernel synchronously
    pub fn run_binomial_reduce(&self, x_values: &[f32], n: f32, p: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.binomial_reduce_pipeline_clone();
        let layout = self.inner.binomial_reduce_bind_group_layout_clone();

        block_on(context::run_binomial_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, n, p,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Poisson distribution REDUCE kernel synchronously
    pub fn run_poisson_reduce(&self, x_values: &[f32], lambda: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.poisson_reduce_pipeline_clone();
        let layout = self.inner.poisson_reduce_bind_group_layout_clone();

        block_on(context::run_poisson_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, lambda,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run NegativeBinomial distribution REDUCE kernel synchronously
    pub fn run_negative_binomial_reduce(
        &self,
        x_values: &[f32],
        r: f32,
        p: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.negative_binomial_reduce_pipeline_clone();
        let layout = self
            .inner
            .negative_binomial_reduce_bind_group_layout_clone();

        block_on(context::run_negative_binomial_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, r, p,
        ))
        .map(|r| r.total_log_prob)
    }

    /// Run Categorical distribution REDUCE kernel synchronously
    pub fn run_categorical_reduce(&self, x_values: &[f32], probs: &[f32]) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.categorical_reduce_pipeline_clone();
        let layout = self.inner.categorical_reduce_bind_group_layout_clone();

        block_on(context::run_categorical_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, probs,
        ))
        .map(|r| r.total_log_prob)
    }

    // ==================== Gradient Reduce kernels ====================

    /// Run Normal distribution GRAD REDUCE kernel synchronously
    ///
    /// Computes sum of grad_log_prob for all x values. Returns scalar total gradient.
    pub fn run_normal_grad_reduce(
        &self,
        x_values: &[f32],
        mu: f32,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_grad_reduce_pipeline_clone();
        let layout = self.inner.normal_grad_reduce_bind_group_layout_clone();

        block_on(context::run_normal_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, mu, sigma,
        ))
        .map(|r| r.total_grad)
    }

    /// Run HalfNormal distribution GRAD REDUCE kernel synchronously
    pub fn run_half_normal_grad_reduce(&self, x_values: &[f32], sigma: f32) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.half_normal_grad_reduce_pipeline_clone();
        let layout = self.inner.half_normal_grad_reduce_bind_group_layout_clone();

        block_on(context::run_half_normal_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, sigma,
        ))
        .map(|r| r.total_grad)
    }

    /// Run Exponential distribution GRAD REDUCE kernel synchronously
    pub fn run_exponential_grad_reduce(
        &self,
        x_values: &[f32],
        lambda: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.exponential_grad_reduce_pipeline_clone();
        let layout = self.inner.exponential_grad_reduce_bind_group_layout_clone();

        block_on(context::run_exponential_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, lambda,
        ))
        .map(|r| r.total_grad)
    }

    /// Run Beta distribution GRAD REDUCE kernel synchronously
    pub fn run_beta_grad_reduce(
        &self,
        x_values: &[f32],
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.beta_grad_reduce_pipeline_clone();
        let layout = self.inner.beta_grad_reduce_bind_group_layout_clone();

        block_on(context::run_beta_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, alpha, beta,
        ))
        .map(|r| r.total_grad)
    }

    /// Run Gamma distribution GRAD REDUCE kernel synchronously
    pub fn run_gamma_grad_reduce(
        &self,
        x_values: &[f32],
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.gamma_grad_reduce_pipeline_clone();
        let layout = self.inner.gamma_grad_reduce_bind_group_layout_clone();

        block_on(context::run_gamma_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, alpha, beta,
        ))
        .map(|r| r.total_grad)
    }

    /// Run InverseGamma distribution GRAD REDUCE kernel synchronously
    pub fn run_inverse_gamma_grad_reduce(
        &self,
        x_values: &[f32],
        alpha: f32,
        beta: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.inverse_gamma_grad_reduce_pipeline_clone();
        let layout = self
            .inner
            .inverse_gamma_grad_reduce_bind_group_layout_clone();

        block_on(context::run_inverse_gamma_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, alpha, beta,
        ))
        .map(|r| r.total_grad)
    }

    /// Run StudentT distribution GRAD REDUCE kernel synchronously
    pub fn run_student_t_grad_reduce(
        &self,
        x_values: &[f32],
        loc: f32,
        scale: f32,
        nu: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.student_t_grad_reduce_pipeline_clone();
        let layout = self.inner.student_t_grad_reduce_bind_group_layout_clone();

        block_on(context::run_student_t_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, loc, scale, nu,
        ))
        .map(|r| r.total_grad)
    }

    /// Run Cauchy distribution GRAD REDUCE kernel synchronously
    pub fn run_cauchy_grad_reduce(
        &self,
        x_values: &[f32],
        loc: f32,
        scale: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.cauchy_grad_reduce_pipeline_clone();
        let layout = self.inner.cauchy_grad_reduce_bind_group_layout_clone();

        block_on(context::run_cauchy_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, loc, scale,
        ))
        .map(|r| r.total_grad)
    }

    /// Run LogNormal distribution GRAD REDUCE kernel synchronously
    pub fn run_lognormal_grad_reduce(
        &self,
        x_values: &[f32],
        mu: f32,
        sigma: f32,
    ) -> Result<f32, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.lognormal_grad_reduce_pipeline_clone();
        let layout = self.inner.lognormal_grad_reduce_bind_group_layout_clone();

        block_on(context::run_lognormal_grad_reduce_kernel(
            &device, &queue, &pipeline, &layout, x_values, mu, sigma,
        ))
        .map(|r| r.total_grad)
    }
}

// ==================== Standalone sync functions ====================
//
// These are lower-level functions that take device/queue/pipeline directly,
// for use when the caller manages their own GPU resources.

/// Run Normal distribution kernel synchronously (standalone)
pub fn run_normal_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x: f32,
    mu: f32,
    sigma: f32,
) -> Result<GpuResult, String> {
    block_on(context::run_normal_kernel(
        device, queue, pipeline, layout, x, mu, sigma,
    ))
}

/// Run HalfNormal distribution kernel synchronously (standalone)
pub fn run_half_normal_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x: f32,
    sigma: f32,
) -> Result<GpuResult, String> {
    block_on(context::run_half_normal_kernel(
        device, queue, pipeline, layout, x, sigma,
    ))
}

/// Run batched Normal distribution kernel synchronously (standalone)
pub fn run_normal_batch_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuBatchResult, String> {
    block_on(context::run_normal_batch_kernel(
        device, queue, pipeline, layout, x_values, mu, sigma,
    ))
}

/// Run Normal distribution REDUCE kernel synchronously (standalone)
pub fn run_normal_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_normal_reduce_kernel(
        device, queue, pipeline, layout, x_values, mu, sigma,
    ))
}

/// Run HalfNormal distribution REDUCE kernel synchronously (standalone)
pub fn run_half_normal_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    sigma: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_half_normal_reduce_kernel(
        device, queue, pipeline, layout, x_values, sigma,
    ))
}

/// Run Exponential distribution REDUCE kernel synchronously (standalone)
pub fn run_exponential_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    lambda: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_exponential_reduce_kernel(
        device, queue, pipeline, layout, x_values, lambda,
    ))
}

/// Run Gamma distribution REDUCE kernel synchronously (standalone)
pub fn run_gamma_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_gamma_reduce_kernel(
        device, queue, pipeline, layout, x_values, alpha, beta,
    ))
}

/// Run Beta distribution REDUCE kernel synchronously (standalone)
pub fn run_beta_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_beta_reduce_kernel(
        device, queue, pipeline, layout, x_values, alpha, beta,
    ))
}

/// Run InverseGamma distribution REDUCE kernel synchronously (standalone)
pub fn run_inverse_gamma_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_inverse_gamma_reduce_kernel(
        device, queue, pipeline, layout, x_values, alpha, beta,
    ))
}

/// Run Uniform distribution REDUCE kernel synchronously (standalone)
pub fn run_uniform_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    low: f32,
    high: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_uniform_reduce_kernel(
        device, queue, pipeline, layout, x_values, low, high,
    ))
}

/// Run Cauchy distribution REDUCE kernel synchronously (standalone)
pub fn run_cauchy_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    loc: f32,
    scale: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_cauchy_reduce_kernel(
        device, queue, pipeline, layout, x_values, loc, scale,
    ))
}

/// Run StudentT distribution REDUCE kernel synchronously (standalone)
#[allow(clippy::too_many_arguments)]
pub fn run_student_t_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    loc: f32,
    scale: f32,
    nu: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_student_t_reduce_kernel(
        device, queue, pipeline, layout, x_values, loc, scale, nu,
    ))
}

/// Run LogNormal distribution REDUCE kernel synchronously (standalone)
pub fn run_lognormal_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_lognormal_reduce_kernel(
        device, queue, pipeline, layout, x_values, mu, sigma,
    ))
}

/// Run Bernoulli distribution REDUCE kernel synchronously (standalone)
pub fn run_bernoulli_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    p: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_bernoulli_reduce_kernel(
        device, queue, pipeline, layout, x_values, p,
    ))
}

/// Run Binomial distribution REDUCE kernel synchronously (standalone)
pub fn run_binomial_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    n: f32,
    p: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_binomial_reduce_kernel(
        device, queue, pipeline, layout, x_values, n, p,
    ))
}

/// Run Poisson distribution REDUCE kernel synchronously (standalone)
pub fn run_poisson_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    lambda: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_poisson_reduce_kernel(
        device, queue, pipeline, layout, x_values, lambda,
    ))
}

/// Run NegativeBinomial distribution REDUCE kernel synchronously (standalone)
pub fn run_negative_binomial_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    r: f32,
    p: f32,
) -> Result<GpuReduceResult, String> {
    block_on(context::run_negative_binomial_reduce_kernel(
        device, queue, pipeline, layout, x_values, r, p,
    ))
}

/// Run Categorical distribution REDUCE kernel synchronously (standalone)
pub fn run_categorical_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    probs: &[f32],
) -> Result<GpuReduceResult, String> {
    block_on(context::run_categorical_reduce_kernel(
        device, queue, pipeline, layout, x_values, probs,
    ))
}

// ==================== Gradient Reduce standalone sync functions ====================

/// Run Normal distribution GRAD REDUCE kernel synchronously (standalone)
pub fn run_normal_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_normal_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, mu, sigma,
    ))
}

/// Run HalfNormal distribution GRAD REDUCE kernel synchronously (standalone)
pub fn run_half_normal_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    sigma: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_half_normal_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, sigma,
    ))
}

/// Run Exponential distribution GRAD REDUCE kernel synchronously (standalone)
pub fn run_exponential_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    lambda: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_exponential_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, lambda,
    ))
}

/// Run Beta distribution GRAD REDUCE kernel synchronously (standalone)
pub fn run_beta_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_beta_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, alpha, beta,
    ))
}

/// Run Gamma distribution GRAD REDUCE kernel synchronously (standalone)
pub fn run_gamma_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_gamma_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, alpha, beta,
    ))
}

/// Run InverseGamma distribution GRAD REDUCE kernel synchronously (standalone)
pub fn run_inverse_gamma_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_inverse_gamma_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, alpha, beta,
    ))
}

/// Run StudentT distribution GRAD REDUCE kernel synchronously (standalone)
#[allow(clippy::too_many_arguments)]
pub fn run_student_t_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    loc: f32,
    scale: f32,
    nu: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_student_t_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, loc, scale, nu,
    ))
}

/// Run Cauchy distribution GRAD REDUCE kernel synchronously (standalone)
pub fn run_cauchy_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    loc: f32,
    scale: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_cauchy_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, loc, scale,
    ))
}

/// Run LogNormal distribution GRAD REDUCE kernel synchronously (standalone)
pub fn run_lognormal_grad_reduce_kernel_sync(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuGradReduceResult, String> {
    block_on(context::run_lognormal_grad_reduce_kernel(
        device, queue, pipeline, layout, x_values, mu, sigma,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_sync_creation() {
        // Use the global shared context to avoid Metal resource conflicts
        // when tests run in parallel
        match GpuContextSync::global() {
            Some(ctx) => {
                // Test that we can run a simple kernel
                let result = ctx.run_normal_reduce(&[1.0, 2.0, 3.0], 2.0, 1.0);
                assert!(result.is_ok(), "Normal reduce should succeed");
            }
            None => {
                eprintln!("Skipping GPU test - no GPU available");
            }
        }
    }
}
