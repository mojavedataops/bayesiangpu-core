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

use super::context::{self, GpuContext, LinpredGpuBuffers, PersistentGpuBuffers};
pub use super::context::{ChainBuffers, MultiChainGpuBuffers};
use super::kernels::{
    BernoulliReduceParams, BetaFusedParams, BetaReduceParams, BinomialReduceParams,
    CauchyReduceParams, ExponentialReduceParams, FusedLogpGradResult, FusedMultiGradResult,
    GammaFusedParams, GammaReduceParams, GpuBatchResult, GpuGradReduceResult, GpuReduceResult,
    GpuResult, HalfNormalReduceParams, InverseGammaFusedParams, InverseGammaReduceParams,
    LinpredGpuResult, LogNormalReduceParams, NegativeBinomialReduceParams, NormalBatchParams,
    PoissonReduceParams, StudentTFusedParams, StudentTReduceParams, UniformReduceParams,
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

    // ==================== Linear Predictor Buffers ====================

    /// Create linpred GPU buffers for y ~ Normal(X @ beta, sigma).
    pub fn create_linpred_buffers(
        &self,
        y_values: &[f32],
        x_matrix: &[f32],
        p: u32,
    ) -> LinpredGpuBuffers {
        context::create_linpred_buffers(self.inner.device_ref(), y_values, x_matrix, p)
    }

    /// Run Normal linear predictor fused kernel: y ~ Normal(X @ beta, sigma).
    pub fn run_normal_linpred_fused(
        &self,
        buffers: &LinpredGpuBuffers,
        beta_values: &[f32],
        sigma: f32,
    ) -> Result<LinpredGpuResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_linpred_fused_reduce_pipeline_clone();
        let layout = self
            .inner
            .normal_linpred_fused_reduce_bind_group_layout_clone();
        block_on(context::run_linpred_fused(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            beta_values,
            sigma,
        ))
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
                log_norm: context::gamma_log_norm(alpha, beta),
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
                log_norm: context::beta_log_norm(alpha, beta),
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
                log_norm: context::inverse_gamma_log_norm(alpha, beta),
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
                log_norm: context::student_t_log_norm(nu, scale),
                _padding1: 0,
                _padding2: 0,
                _padding3: 0,
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
                log_norm: context::lognormal_log_norm(sigma),
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
                log_norm: context::beta_log_norm(alpha, beta),
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
                log_norm: context::gamma_log_norm(alpha, beta),
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
                log_norm: context::inverse_gamma_log_norm(alpha, beta),
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
                log_norm: context::student_t_log_norm(nu, scale),
                _padding1: 0,
                _padding2: 0,
                _padding3: 0,
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
                log_norm: context::lognormal_log_norm(sigma),
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
                log_norm: context::beta_log_norm(alpha, beta),
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
                log_norm: context::gamma_log_norm(alpha, beta),
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
                log_norm: context::inverse_gamma_log_norm(alpha, beta),
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
                log_norm: context::student_t_log_norm(nu, scale),
                _padding1: 0,
                _padding2: 0,
                _padding3: 0,
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
                log_norm: context::lognormal_log_norm(sigma),
            },
        ))
    }

    // ==================== Single-pass fused logp + grad persistent kernels ====================
    // These delegate to the multi-grad fused path and extract logp + first gradient
    // for backward compatibility with code expecting FusedLogpGradResult.

    /// Run single-pass fused Normal logp + grad using persistent buffers.
    pub fn run_normal_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_normal_multi_grad_fused(buffers, mu, sigma)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    /// Run single-pass fused HalfNormal logp + grad using persistent buffers.
    pub fn run_half_normal_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        sigma: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_half_normal_multi_grad_fused(buffers, sigma)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    /// Run single-pass fused Exponential logp + grad using persistent buffers.
    pub fn run_exponential_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        lambda: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_exponential_multi_grad_fused(buffers, lambda)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    /// Run single-pass fused Gamma logp + grad using persistent buffers.
    pub fn run_gamma_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_gamma_multi_grad_fused(buffers, alpha, beta)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    /// Run single-pass fused Beta logp + grad using persistent buffers.
    pub fn run_beta_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_beta_multi_grad_fused(buffers, alpha, beta)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    /// Run single-pass fused InverseGamma logp + grad using persistent buffers.
    pub fn run_inverse_gamma_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_inverse_gamma_multi_grad_fused(buffers, alpha, beta)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    /// Run single-pass fused StudentT logp + grad using persistent buffers.
    pub fn run_student_t_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        nu: f32,
        loc: f32,
        scale: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_student_t_multi_grad_fused(buffers, loc, scale, nu)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    /// Run single-pass fused Cauchy logp + grad using persistent buffers.
    pub fn run_cauchy_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_cauchy_multi_grad_fused(buffers, loc, scale)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    /// Run single-pass fused LogNormal logp + grad using persistent buffers.
    pub fn run_lognormal_single_pass_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<FusedLogpGradResult, String> {
        let r = self.run_lognormal_multi_grad_fused(buffers, mu, sigma)?;
        Ok(FusedLogpGradResult {
            total_log_prob: r.total_log_prob,
            total_grad: r.total_grads[0],
        })
    }

    // ==================== Multi-gradient fused kernels ====================
    // These use the updated fused shaders that output gradients for ALL parameters.

    /// Run multi-grad fused Normal logp + grad_mu + grad_sigma.
    pub fn run_normal_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_fused_reduce_pipeline_clone();
        let layout = self.inner.normal_fused_reduce_bind_group_layout_clone();
        block_on(context::run_single_pass_fused_persistent_multi(
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
            3, // logp + grad_mu + grad_sigma
        ))
    }

    /// Run multi-grad fused Beta logp + grad_alpha + grad_beta.
    pub fn run_beta_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.beta_fused_reduce_pipeline_clone();
        let layout = self.inner.beta_fused_reduce_bind_group_layout_clone();
        let (psi_sum_minus_alpha, psi_sum_minus_beta) = context::beta_fused_psi_consts(alpha, beta);
        block_on(context::run_single_pass_fused_persistent_multi(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            BetaFusedParams {
                alpha,
                beta,
                count: buffers.count,
                log_norm: context::beta_log_norm(alpha, beta),
                psi_sum_minus_alpha,
                psi_sum_minus_beta,
                _padding1: 0,
                _padding2: 0,
            },
            3,
        ))
    }

    /// Run multi-grad fused Gamma logp + grad_alpha + grad_beta.
    pub fn run_gamma_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.gamma_fused_reduce_pipeline_clone();
        let layout = self.inner.gamma_fused_reduce_bind_group_layout_clone();
        block_on(context::run_single_pass_fused_persistent_multi(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            GammaFusedParams {
                alpha,
                beta,
                count: buffers.count,
                log_norm: context::gamma_log_norm(alpha, beta),
                neg_psi_alpha_plus_log_beta: context::gamma_fused_psi_const(alpha, beta),
                _padding1: 0,
                _padding2: 0,
                _padding3: 0,
            },
            3,
        ))
    }

    /// Run multi-grad fused InverseGamma logp + grad_alpha + grad_beta.
    pub fn run_inverse_gamma_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        alpha: f32,
        beta: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.inverse_gamma_fused_reduce_pipeline_clone();
        let layout = self
            .inner
            .inverse_gamma_fused_reduce_bind_group_layout_clone();
        block_on(context::run_single_pass_fused_persistent_multi(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            InverseGammaFusedParams {
                alpha,
                beta,
                count: buffers.count,
                log_norm: context::inverse_gamma_log_norm(alpha, beta),
                log_beta_minus_psi_alpha: context::inverse_gamma_fused_psi_const(alpha, beta),
                _padding1: 0,
                _padding2: 0,
                _padding3: 0,
            },
            3,
        ))
    }

    /// Run multi-grad fused StudentT logp + grad_loc + grad_scale + grad_nu.
    pub fn run_student_t_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
        nu: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.student_t_fused_reduce_pipeline_clone();
        let layout = self.inner.student_t_fused_reduce_bind_group_layout_clone();
        block_on(context::run_single_pass_fused_persistent_multi(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            StudentTFusedParams {
                loc,
                scale,
                nu,
                count: buffers.count,
                log_norm: context::student_t_log_norm(nu, scale),
                psi_const: context::student_t_fused_psi_const(nu),
                _padding1: 0,
                _padding2: 0,
            },
            4, // logp + grad_loc + grad_scale + grad_nu
        ))
    }

    /// Run multi-grad fused Cauchy logp + grad_loc + grad_scale.
    pub fn run_cauchy_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        loc: f32,
        scale: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.cauchy_fused_reduce_pipeline_clone();
        let layout = self.inner.cauchy_fused_reduce_bind_group_layout_clone();
        block_on(context::run_single_pass_fused_persistent_multi(
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
            3,
        ))
    }

    /// Run multi-grad fused LogNormal logp + grad_mu + grad_sigma.
    pub fn run_lognormal_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        mu: f32,
        sigma: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.lognormal_fused_reduce_pipeline_clone();
        let layout = self.inner.lognormal_fused_reduce_bind_group_layout_clone();
        block_on(context::run_single_pass_fused_persistent_multi(
            &device,
            &queue,
            &pipeline,
            &layout,
            buffers,
            LogNormalReduceParams {
                mu,
                sigma,
                count: buffers.count,
                log_norm: context::lognormal_log_norm(sigma),
            },
            3,
        ))
    }

    /// Run multi-grad fused HalfNormal logp + grad_sigma.
    pub fn run_half_normal_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        sigma: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.half_normal_fused_reduce_pipeline_clone();
        let layout = self
            .inner
            .half_normal_fused_reduce_bind_group_layout_clone();
        block_on(context::run_single_pass_fused_persistent_multi(
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
            2, // logp + grad_sigma
        ))
    }

    /// Run multi-grad fused Exponential logp + grad_lambda.
    pub fn run_exponential_multi_grad_fused(
        &self,
        buffers: &PersistentGpuBuffers,
        lambda: f32,
    ) -> Result<FusedMultiGradResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.exponential_fused_reduce_pipeline_clone();
        let layout = self
            .inner
            .exponential_fused_reduce_bind_group_layout_clone();
        block_on(context::run_single_pass_fused_persistent_multi(
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
            2, // logp + grad_lambda
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

impl GpuContextSync {
    // ==================== Indexed parameter methods (hierarchical models) ====================

    /// Run indexed Normal reduce for y[i] ~ Normal(theta[group[i]], sigma).
    ///
    /// Returns total logp, grad_sigma, and per-group grad_theta.
    #[allow(clippy::too_many_arguments)]
    pub fn run_normal_indexed_reduce(
        &self,
        y_sorted: &[f32],
        theta: &[f32],
        group_idx: &[u32],
        group_boundaries: &[usize],
        sigma: f32,
    ) -> Result<context::IndexedNormalResult, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let pipeline = self.inner.normal_indexed_reduce_pipeline_clone();
        let layout = self.inner.normal_indexed_reduce_bind_group_layout_clone();
        block_on(context::run_normal_indexed_reduce(
            &device,
            &queue,
            &pipeline,
            &layout,
            y_sorted,
            theta,
            group_idx,
            group_boundaries,
            sigma,
        ))
    }
}

impl GpuContextSync {
    // ==================== Multi-chain methods ====================

    /// Create multi-chain GPU buffers with shared observation data.
    ///
    /// Allocates one x_buffer shared across all chains, plus per-chain buffers
    /// for params, partial sums, and staging.
    pub fn create_multi_chain_buffers(
        &self,
        x_values: &[f32],
        max_params_size: u64,
        num_chains: u32,
    ) -> MultiChainGpuBuffers {
        context::create_multi_chain_buffers(
            self.inner.device_ref(),
            self.inner.queue_ref(),
            x_values,
            max_params_size,
            num_chains,
        )
    }

    /// Run fused Normal logp + grad for multiple chains in a single GPU dispatch.
    ///
    /// All chains share the same observation data but have different (mu, sigma).
    /// Returns one FusedLogpGradResult per chain.
    pub fn run_normal_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[NormalBatchParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.normal_reduce_pipeline_clone();
        let logp_layout = self.inner.normal_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.normal_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.normal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }

    /// Run fused Exponential logp + grad for multiple chains in a single GPU dispatch.
    pub fn run_exponential_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[ExponentialReduceParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.exponential_reduce_pipeline_clone();
        let logp_layout = self.inner.exponential_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.exponential_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.exponential_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }

    /// Run fused Beta logp + grad for multiple chains in a single GPU dispatch.
    pub fn run_beta_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[BetaReduceParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.beta_reduce_pipeline_clone();
        let logp_layout = self.inner.beta_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.beta_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.beta_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }

    /// Run fused Gamma logp + grad for multiple chains in a single GPU dispatch.
    pub fn run_gamma_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[GammaReduceParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.gamma_reduce_pipeline_clone();
        let logp_layout = self.inner.gamma_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.gamma_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.gamma_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }

    /// Run fused StudentT logp + grad for multiple chains in a single GPU dispatch.
    pub fn run_student_t_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[StudentTReduceParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.student_t_reduce_pipeline_clone();
        let logp_layout = self.inner.student_t_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.student_t_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.student_t_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }

    /// Run fused Cauchy logp + grad for multiple chains in a single GPU dispatch.
    pub fn run_cauchy_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[CauchyReduceParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.cauchy_reduce_pipeline_clone();
        let logp_layout = self.inner.cauchy_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.cauchy_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.cauchy_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }

    /// Run fused LogNormal logp + grad for multiple chains in a single GPU dispatch.
    pub fn run_lognormal_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[LogNormalReduceParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.lognormal_reduce_pipeline_clone();
        let logp_layout = self.inner.lognormal_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.lognormal_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.lognormal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }

    /// Run fused HalfNormal logp + grad for multiple chains in a single GPU dispatch.
    pub fn run_half_normal_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[HalfNormalReduceParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.half_normal_reduce_pipeline_clone();
        let logp_layout = self.inner.half_normal_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.half_normal_grad_reduce_pipeline_clone();
        let grad_layout = self.inner.half_normal_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }

    /// Run fused InverseGamma logp + grad for multiple chains in a single GPU dispatch.
    pub fn run_inverse_gamma_multi_chain_fused(
        &self,
        buffers: &MultiChainGpuBuffers,
        chain_params: &[InverseGammaReduceParams],
    ) -> Result<Vec<FusedLogpGradResult>, String> {
        let device = self.inner.device_clone();
        let queue = self.inner.queue_clone();
        let logp_pipeline = self.inner.inverse_gamma_reduce_pipeline_clone();
        let logp_layout = self.inner.inverse_gamma_reduce_bind_group_layout_clone();
        let grad_pipeline = self.inner.inverse_gamma_grad_reduce_pipeline_clone();
        let grad_layout = self
            .inner
            .inverse_gamma_grad_reduce_bind_group_layout_clone();
        block_on(context::run_multi_chain_fused(
            &device,
            &queue,
            &logp_pipeline,
            &logp_layout,
            &grad_pipeline,
            &grad_layout,
            buffers,
            chain_params,
        ))
    }
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

    #[test]
    fn test_multi_chain_normal_fused() {
        match GpuContextSync::global() {
            Some(ctx) => {
                let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
                let mc_buffers = ctx.create_multi_chain_buffers(&data, 16, 4);

                let chain_params: Vec<NormalBatchParams> = (0..4)
                    .map(|i| NormalBatchParams {
                        mu: 5.0 + i as f32,
                        sigma: 1.0,
                        count: data.len() as u32,
                        _padding: 0,
                    })
                    .collect();

                let results = ctx.run_normal_multi_chain_fused(&mc_buffers, &chain_params);
                assert!(results.is_ok(), "Multi-chain fused should succeed");
                let results = results.unwrap();
                assert_eq!(results.len(), 4, "Should have 4 chain results");
                // Each chain should have a different logp (different mu)
                assert_ne!(
                    results[0].total_log_prob, results[1].total_log_prob,
                    "Different mu should give different logp"
                );
            }
            None => {
                eprintln!("Skipping multi-chain GPU test - no GPU available");
            }
        }
    }

    #[test]
    fn test_normal_linpred_fused() {
        match GpuContextSync::global() {
            Some(ctx) => {
                // Simple regression: y = 2*x1 + 3*x2 + noise
                // N=100, P=2
                let n = 100usize;
                let p = 2u32;
                let true_beta = [2.0f32, 3.0];
                let sigma = 1.0f32;

                // Generate design matrix and response
                let mut x_matrix = vec![0.0f32; n * p as usize];
                let mut y_values = vec![0.0f32; n];
                for i in 0..n {
                    let x1 = (i as f32) * 0.1;
                    let x2 = (i as f32) * 0.05 + 1.0;
                    x_matrix[i * 2] = x1;
                    x_matrix[i * 2 + 1] = x2;
                    // y = 2*x1 + 3*x2 (no noise for deterministic test)
                    y_values[i] = true_beta[0] * x1 + true_beta[1] * x2;
                }

                let buffers = ctx.create_linpred_buffers(&y_values, &x_matrix, p);

                // At true beta, residuals are zero, logp should be N * (-0.5*log(2*pi) - log(sigma))
                let result = ctx
                    .run_normal_linpred_fused(&buffers, &true_beta, sigma)
                    .expect("Linpred kernel should succeed");

                let expected_logp_per_obs = -0.5 * std::f32::consts::TAU.ln() - sigma.ln();
                let expected_total_logp = expected_logp_per_obs * n as f32;
                let logp_err = (result.total_log_prob - expected_total_logp).abs();
                assert!(
                    logp_err < 0.5,
                    "Log-prob at true beta should be ~{}, got {} (err={})",
                    expected_total_logp,
                    result.total_log_prob,
                    logp_err
                );

                // At true beta, residuals are zero, so grad_beta should be ~0
                for (j, &gb) in result.grad_beta.iter().enumerate() {
                    assert!(
                        gb.abs() < 0.01,
                        "grad_beta[{}] at true beta should be ~0, got {}",
                        j,
                        gb
                    );
                }

                // Test with perturbed beta - gradients should point toward true beta
                let perturbed_beta = [1.5f32, 2.5]; // below true values
                let result2 = ctx
                    .run_normal_linpred_fused(&buffers, &perturbed_beta, sigma)
                    .expect("Linpred kernel should succeed");

                // Gradients should be positive (pushing beta upward toward true values)
                for (j, &gb) in result2.grad_beta.iter().enumerate() {
                    assert!(
                        gb > 0.0,
                        "grad_beta[{}] with beta below true should be positive, got {}",
                        j,
                        gb
                    );
                }

                // Log-prob at perturbed beta should be lower than at true beta
                assert!(
                    result2.total_log_prob < result.total_log_prob,
                    "Log-prob at perturbed beta ({}) should be less than at true beta ({})",
                    result2.total_log_prob,
                    result.total_log_prob
                );
            }
            None => {
                eprintln!("Skipping linpred GPU test - no GPU available");
            }
        }
    }

    #[test]
    fn test_normal_indexed_reduce() {
        // Test hierarchical model: y[i] ~ Normal(theta[group[i]], sigma)
        // 3 groups with known theta values, check logp and gradients
        match GpuContextSync::global() {
            Some(ctx) => {
                let sigma: f32 = 1.0;

                // 3 groups: theta = [0.0, 2.0, -1.0]
                let theta: Vec<f32> = vec![0.0, 2.0, -1.0];
                let num_groups = theta.len();

                // Observations: 4 per group, already sorted by group
                // Group 0: y = [0.1, -0.2, 0.3, -0.1]
                // Group 1: y = [1.8, 2.1, 2.3, 1.9]
                // Group 2: y = [-0.8, -1.2, -0.9, -1.1]
                let y_sorted: Vec<f32> = vec![
                    0.1, -0.2, 0.3, -0.1, // group 0
                    1.8, 2.1, 2.3, 1.9, // group 1
                    -0.8, -1.2, -0.9, -1.1, // group 2
                ];
                let group_idx: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
                let group_boundaries: Vec<usize> = vec![0, 4, 8, 12];

                let result = ctx
                    .run_normal_indexed_reduce(
                        &y_sorted,
                        &theta,
                        &group_idx,
                        &group_boundaries,
                        sigma,
                    )
                    .expect("Indexed reduce should succeed");

                // Verify logp: sum of Normal logp for each observation
                let log_2pi = (2.0 * std::f64::consts::PI).ln();
                let mut expected_logp = 0.0f64;
                for (i, &y) in y_sorted.iter().enumerate() {
                    let mu = theta[group_idx[i] as usize] as f64;
                    let z = (y as f64 - mu) / sigma as f64;
                    expected_logp += -0.5 * log_2pi - (sigma as f64).ln() - 0.5 * z * z;
                }
                assert!(
                    (result.total_log_prob - expected_logp).abs() < 0.1,
                    "logp mismatch: GPU={}, expected={}",
                    result.total_log_prob,
                    expected_logp,
                );

                // Verify grad_sigma
                let sigma_f64 = sigma as f64;
                let mut expected_grad_sigma = 0.0f64;
                for (i, &y) in y_sorted.iter().enumerate() {
                    let mu = theta[group_idx[i] as usize] as f64;
                    let z = (y as f64 - mu) / sigma_f64;
                    expected_grad_sigma += (-1.0 + z * z) / sigma_f64;
                }
                assert!(
                    (result.grad_sigma - expected_grad_sigma).abs() < 0.1,
                    "grad_sigma mismatch: GPU={}, expected={}",
                    result.grad_sigma,
                    expected_grad_sigma,
                );

                // Verify per-group theta gradients
                assert_eq!(result.grad_theta.len(), num_groups);
                for k in 0..num_groups {
                    let start = group_boundaries[k];
                    let end = group_boundaries[k + 1];
                    let mu_k = theta[k] as f64;
                    let expected_grad: f64 = y_sorted[start..end]
                        .iter()
                        .map(|&y| (y as f64 - mu_k) / (sigma_f64 * sigma_f64))
                        .sum();
                    assert!(
                        (result.grad_theta[k] - expected_grad).abs() < 1e-4,
                        "grad_theta[{}] mismatch: GPU={}, expected={}",
                        k,
                        result.grad_theta[k],
                        expected_grad,
                    );
                }
            }
            None => {
                eprintln!("Skipping indexed reduce GPU test - no GPU available");
            }
        }
    }
}
