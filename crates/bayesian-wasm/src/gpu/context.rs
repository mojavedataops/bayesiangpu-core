//! GPU context management for direct wgpu compute
//!
//! Handles wgpu device/queue initialization and compute pipeline creation.
//! Pipelines are lazily compiled on first use via OnceLock, avoiding the
//! upfront cost of compiling 30+ WGSL shaders during initialization.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::{util::DeviceExt, BindGroupLayout, BufferUsages, ComputePipeline, Device, Queue};

use super::kernels::{
    BernoulliReduceParams, BetaReduceParams, BinomialReduceParams, CategoricalReduceParams,
    CauchyReduceParams, ExponentialReduceParams, FusedLogpGradResult, FusedMultiGradResult,
    GammaReduceParams, GpuBatchResult, GpuGradReduceResult, GpuReduceResult, GpuResult,
    HalfNormalParams, HalfNormalReduceParams, InverseGammaReduceParams, LinpredGpuResult,
    LogNormalReduceParams, NegativeBinomialReduceParams, NormalBatchParams,
    NormalIndexedReduceParams, NormalLinpredParams, NormalParams, PoissonReduceParams,
    StudentTReduceParams, UniformReduceParams,
};

/// Compute lgamma(x) using the libm crate.
fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Compute log_norm for Gamma distribution: alpha * ln(beta) - lgamma(alpha)
pub(crate) fn gamma_log_norm(alpha: f32, beta: f32) -> f32 {
    let a = alpha as f64;
    let b = beta as f64;
    (a * b.ln() - lgamma(a)) as f32
}

/// Compute log_norm for Beta distribution: lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
pub(crate) fn beta_log_norm(alpha: f32, beta: f32) -> f32 {
    let a = alpha as f64;
    let b = beta as f64;
    (lgamma(a + b) - lgamma(a) - lgamma(b)) as f32
}

/// Compute log_norm for InverseGamma: alpha * ln(beta) - lgamma(alpha)
pub(crate) fn inverse_gamma_log_norm(alpha: f32, beta: f32) -> f32 {
    gamma_log_norm(alpha, beta) // Same formula
}

/// Compute log_norm for StudentT: lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*ln(nu*PI) - ln(scale)
pub(crate) fn student_t_log_norm(nu: f32, scale: f32) -> f32 {
    let n = nu as f64;
    let s = scale as f64;
    (lgamma((n + 1.0) / 2.0) - lgamma(n / 2.0) - 0.5 * (n * std::f64::consts::PI).ln() - s.ln())
        as f32
}

/// Compute log_norm for LogNormal: -0.5 * ln(2*PI) - ln(sigma)
pub(crate) fn lognormal_log_norm(sigma: f32) -> f32 {
    let s = sigma as f64;
    (-0.5 * (2.0 * std::f64::consts::PI).ln() - s.ln()) as f32
}

/// Compute the digamma (psi) function using the asymptotic series.
pub(crate) fn digamma(x: f64) -> f64 {
    // Shift x up until x >= 8 for good asymptotic convergence
    let mut result = 0.0;
    let mut x = x;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion for large x
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    result += x.ln()
        - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0 - inv_x2 / 240.0)));
    result
}

/// Pre-compute digamma constants for Beta fused kernel
pub(crate) fn beta_fused_psi_consts(alpha: f32, beta: f32) -> (f32, f32) {
    let a = alpha as f64;
    let b = beta as f64;
    let psi_sum = digamma(a + b);
    (
        (psi_sum - digamma(a)) as f32, // psi(alpha+beta) - psi(alpha)
        (psi_sum - digamma(b)) as f32, // psi(alpha+beta) - psi(beta)
    )
}

/// Pre-compute digamma constants for Gamma fused kernel
pub(crate) fn gamma_fused_psi_const(alpha: f32, beta: f32) -> f32 {
    let a = alpha as f64;
    let b = beta as f64;
    (-digamma(a) + b.ln()) as f32
}

/// Pre-compute digamma constants for InverseGamma fused kernel
pub(crate) fn inverse_gamma_fused_psi_const(alpha: f32, beta: f32) -> f32 {
    let a = alpha as f64;
    let b = beta as f64;
    (b.ln() - digamma(a)) as f32
}

/// Pre-compute digamma constants for StudentT fused kernel
pub(crate) fn student_t_fused_psi_const(nu: f32) -> f32 {
    let n = nu as f64;
    (0.5 * (digamma((n + 1.0) / 2.0) - digamma(n / 2.0) - 1.0 / n)) as f32
}

/// A lazily-initialized pipeline + bind group layout pair.
type LazyPipeline = OnceLock<(Arc<ComputePipeline>, Arc<BindGroupLayout>)>;

/// Parameters for the generic reduce_sum shader (second-pass reduction).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReduceSumParams {
    pub count: u32,
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
}

/// Number of elements processed per thread in coarsened reduce shaders.
const ELEMS_PER_THREAD: u32 = 4;
/// Effective elements per workgroup: 256 threads * 4 elements/thread = 1024
const ELEMS_PER_WORKGROUP: u32 = 256 * ELEMS_PER_THREAD;

/// Threshold: if workgroup_count exceeds this, use a second GPU pass instead of CPU sum.
const SECOND_PASS_THRESHOLD: u32 = 1024;

/// Maximum number of fused outputs per workgroup (logp + up to 3 gradients for StudentT).
const MAX_FUSED_OUTPUTS: u32 = 4;

/// GPU compute context holding wgpu resources
///
/// Resources are wrapped in Arc for cheap cloning when needed for async operations.
/// Pipelines are lazily compiled on first access to avoid long initialization times
/// when only a subset of distributions is used.
pub struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    // Basic kernels
    normal: LazyPipeline,
    half_normal: LazyPipeline,
    normal_batch: LazyPipeline,
    // Log-prob reduce kernels
    normal_reduce: LazyPipeline,
    half_normal_reduce: LazyPipeline,
    exponential_reduce: LazyPipeline,
    gamma_reduce: LazyPipeline,
    beta_reduce: LazyPipeline,
    inverse_gamma_reduce: LazyPipeline,
    uniform_reduce: LazyPipeline,
    cauchy_reduce: LazyPipeline,
    student_t_reduce: LazyPipeline,
    lognormal_reduce: LazyPipeline,
    bernoulli_reduce: LazyPipeline,
    binomial_reduce: LazyPipeline,
    poisson_reduce: LazyPipeline,
    negative_binomial_reduce: LazyPipeline,
    categorical_reduce: LazyPipeline,
    // Generic second-pass reduction
    reduce_sum: LazyPipeline,
    // Gradient reduce kernels
    normal_grad_reduce: LazyPipeline,
    half_normal_grad_reduce: LazyPipeline,
    exponential_grad_reduce: LazyPipeline,
    beta_grad_reduce: LazyPipeline,
    gamma_grad_reduce: LazyPipeline,
    inverse_gamma_grad_reduce: LazyPipeline,
    student_t_grad_reduce: LazyPipeline,
    cauchy_grad_reduce: LazyPipeline,
    lognormal_grad_reduce: LazyPipeline,
    // Fused single-pass logp+grad reduce kernels
    normal_fused_reduce: LazyPipeline,
    half_normal_fused_reduce: LazyPipeline,
    exponential_fused_reduce: LazyPipeline,
    gamma_fused_reduce: LazyPipeline,
    beta_fused_reduce: LazyPipeline,
    inverse_gamma_fused_reduce: LazyPipeline,
    student_t_fused_reduce: LazyPipeline,
    cauchy_fused_reduce: LazyPipeline,
    lognormal_fused_reduce: LazyPipeline,
    // Linear predictor kernel
    normal_linpred_fused_reduce: LazyPipeline,
    // Indexed parameter kernel (hierarchical models)
    normal_indexed_reduce: LazyPipeline,
}

// ============================================================================
// Helper functions
// ============================================================================

/// On native targets, poll the device to drive buffer mapping to completion.
/// On WASM, the browser event loop handles wgpu polling automatically.
#[cfg(not(target_arch = "wasm32"))]
fn poll_device(device: &Device) {
    let _ = device.poll(wgpu::PollType::Wait);
}

#[cfg(target_arch = "wasm32")]
fn poll_device(_device: &Device) {
    // Browser event loop handles wgpu polling on WASM
}

// ============================================================================
// Helper functions for pipeline creation (used by lazy initializers)
// ============================================================================

/// Create the standard 3-binding reduce layout (params, x_values, partial_sums)
fn create_reduce_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create a compute pipeline from shader source and bind group layout
fn create_pipeline(
    device: &Device,
    shader_source: &str,
    label: &str,
    layout: &BindGroupLayout,
) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{}_layout", label)),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Create the standard 2-binding basic kernel layout (params uniform, output storage)
fn create_basic_2_binding_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create the 4-binding batch layout (params, x_values, log_probs, grads)
fn create_batch_4_binding_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create the 4-binding categorical layout (params, x_values, probs, partial_sums)
fn create_categorical_reduce_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create the 5-binding indexed reduce layout (params, y_values, theta, group_idx, output)
fn create_indexed_reduce_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Helper: lazily create a standard 3-binding reduce pipeline
fn init_reduce_pipeline(
    device: &Device,
    shader_source: &str,
    label: &str,
) -> (Arc<ComputePipeline>, Arc<BindGroupLayout>) {
    let layout = create_reduce_layout(device, &format!("{}_bind_group_layout", label));
    let pipeline = create_pipeline(
        device,
        shader_source,
        &format!("{}_pipeline", label),
        &layout,
    );
    (Arc::new(pipeline), Arc::new(layout))
}

/// Helper: lazily create a 2-binding basic pipeline
fn init_basic_pipeline(
    device: &Device,
    shader_source: &str,
    label: &str,
) -> (Arc<ComputePipeline>, Arc<BindGroupLayout>) {
    let layout = create_basic_2_binding_layout(device, &format!("{}_bind_group_layout", label));
    let pipeline = create_pipeline(
        device,
        shader_source,
        &format!("{}_pipeline", label),
        &layout,
    );
    (Arc::new(pipeline), Arc::new(layout))
}

/// Helper: lazily create a 5-binding linpred pipeline (same layout as indexed: params, y, X, beta, output)
fn init_linpred_pipeline(
    device: &Device,
    shader_source: &str,
    label: &str,
) -> (Arc<ComputePipeline>, Arc<BindGroupLayout>) {
    let layout = create_indexed_reduce_layout(device, &format!("{}_bind_group_layout", label));
    let pipeline = create_pipeline(
        device,
        shader_source,
        &format!("{}_pipeline", label),
        &layout,
    );
    (Arc::new(pipeline), Arc::new(layout))
}

/// Pre-allocated GPU buffers for linear predictor kernel execution.
///
/// y is uploaded once. X matrix is uploaded once. Beta is updated via write_buffer per step.
/// Output buffer is sized for (P+2) * workgroup_count f32 values.
pub struct LinpredGpuBuffers {
    pub y_buffer: wgpu::Buffer,
    pub x_buffer: wgpu::Buffer,
    pub beta_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
    pub workgroup_count: u32,
    pub count: u32,
    pub p: u32,
}

/// Create linpred GPU buffers for y ~ Normal(X @ beta, sigma).
pub fn create_linpred_buffers(
    device: &Device,
    y_values: &[f32],
    x_matrix: &[f32],
    p: u32,
) -> LinpredGpuBuffers {
    let count = y_values.len() as u32;
    let workgroup_count = count.div_ceil(ELEMS_PER_WORKGROUP);
    let output_count = workgroup_count as u64 * (p as u64 + 2);
    let output_size = output_count * 4;

    let y_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("linpred_y"),
        contents: bytemuck::cast_slice(y_values),
        usage: BufferUsages::STORAGE,
    });

    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("linpred_x"),
        contents: bytemuck::cast_slice(x_matrix),
        usage: BufferUsages::STORAGE,
    });

    let beta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("linpred_beta"),
        size: (p as u64) * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("linpred_params"),
        size: std::mem::size_of::<NormalLinpredParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("linpred_output"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("linpred_staging"),
        size: output_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    LinpredGpuBuffers {
        y_buffer,
        x_buffer,
        beta_buffer,
        params_buffer,
        output_buffer,
        staging_buffer,
        workgroup_count,
        count,
        p,
    }
}

// =============================================================================
// PERSISTENT GPU BUFFERS
// =============================================================================

/// Pre-allocated GPU buffers for repeated kernel execution.
///
/// Observation data is uploaded once. The params buffer is updated in-place
/// via `queue.write_buffer`. Partial sums and staging buffers are reused.
pub struct PersistentGpuBuffers {
    pub x_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub partial_sums_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
    pub grad_partial_sums_buffer: wgpu::Buffer,
    pub grad_staging_buffer: wgpu::Buffer,
    pub workgroup_count: u32,
    pub count: u32,
    /// Max params struct size this buffer was allocated for
    pub params_capacity: u64,
    /// Cached bind groups keyed by pipeline pointer address.
    /// Since the underlying buffers never change, cached bind groups remain valid for the
    /// lifetime of the PersistentGpuBuffers. Uses partial_sums_buffer at binding 2.
    bind_group_cache: Mutex<HashMap<usize, wgpu::BindGroup>>,
    /// Cached bind groups for the grad path (uses grad_partial_sums_buffer at binding 2)
    grad_bind_group_cache: Mutex<HashMap<usize, wgpu::BindGroup>>,
    /// Small output buffer for second-pass GPU reduction (single f32)
    pub reduction_output_buffer: wgpu::Buffer,
    /// Staging buffer for reading back the single reduced value
    pub reduction_staging_buffer: wgpu::Buffer,
    /// Lazily-initialized reduce_sum pipeline for second-pass GPU reduction.
    /// Created on first use when workgroup_count > SECOND_PASS_THRESHOLD.
    reduce_sum_pipeline: OnceLock<(Arc<ComputePipeline>, Arc<BindGroupLayout>)>,
    /// Output buffer for single-pass fused shaders (interleaved logp+grad, 2 * workgroup_count f32s)
    pub fused_output_buffer: wgpu::Buffer,
    /// Staging buffer for reading back fused interleaved results
    pub fused_staging_buffer: wgpu::Buffer,
    /// Cached bind groups for fused single-pass path (uses fused_output_buffer at binding 2)
    fused_bind_group_cache: Mutex<HashMap<usize, wgpu::BindGroup>>,
}

/// Create persistent GPU buffers for repeated kernel execution.
///
/// Observation data (`x_values`) is uploaded once into a storage buffer.
/// The params buffer is sized to `max_params_size` and can be updated in-place
/// via `queue.write_buffer` on each call. Partial sums and staging buffers
/// are sized for the workgroup count and reused across dispatches.
pub fn create_persistent_buffers(
    device: &Device,
    queue: &Queue,
    x_values: &[f32],
    max_params_size: u64,
) -> PersistentGpuBuffers {
    let _ = queue; // queue not needed for buffer creation, but kept in API for future use
    let count = x_values.len() as u32;
    let workgroup_count = count.div_ceil(ELEMS_PER_WORKGROUP);
    let partial_sums_size = (workgroup_count as u64) * 4;

    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("persistent_x"),
        contents: bytemuck::cast_slice(x_values),
        usage: BufferUsages::STORAGE,
    });

    // Params buffer with COPY_DST so we can update in-place
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_params"),
        size: max_params_size,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let partial_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_partial_sums"),
        size: partial_sums_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_staging"),
        size: partial_sums_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let grad_partial_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_grad_partial_sums"),
        size: partial_sums_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let grad_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_grad_staging"),
        size: partial_sums_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let reduction_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_reduction_output"),
        size: 4, // single f32
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let reduction_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_reduction_staging"),
        size: 4, // single f32
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Fused single-pass output: supports up to MAX_FUSED_OUTPUTS values per workgroup.
    // StudentT needs 4 (logp + 3 grads), others need 2 or 3.
    let fused_output_size = (workgroup_count as u64) * MAX_FUSED_OUTPUTS as u64 * 4;
    let fused_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_fused_output"),
        size: fused_output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let fused_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("persistent_fused_staging"),
        size: fused_output_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    PersistentGpuBuffers {
        x_buffer,
        params_buffer,
        partial_sums_buffer,
        staging_buffer,
        grad_partial_sums_buffer,
        grad_staging_buffer,
        workgroup_count,
        count,
        params_capacity: max_params_size,
        bind_group_cache: Mutex::new(HashMap::new()),
        grad_bind_group_cache: Mutex::new(HashMap::new()),
        reduction_output_buffer,
        reduction_staging_buffer,
        reduce_sum_pipeline: OnceLock::new(),
        fused_output_buffer,
        fused_staging_buffer,
        fused_bind_group_cache: Mutex::new(HashMap::new()),
    }
}

impl PersistentGpuBuffers {
    /// Get or create a cached bind group for the reduce path (partial_sums_buffer at binding 2).
    /// The returned MutexGuard must be held while the bind group is used.
    fn cached_bind_group<'a>(
        &'a self,
        device: &Device,
        pipeline: &Arc<ComputePipeline>,
        layout: &Arc<BindGroupLayout>,
    ) -> std::sync::MutexGuard<'a, HashMap<usize, wgpu::BindGroup>> {
        let key = Arc::as_ptr(pipeline) as usize;
        let mut cache = self.bind_group_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("persistent_reduce_bg"),
                layout: layout.as_ref(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.x_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.partial_sums_buffer.as_entire_binding(),
                    },
                ],
            })
        });
        cache
    }

    /// Get or create a cached bind group for the fused grad path (grad_partial_sums_buffer at binding 2).
    fn cached_grad_bind_group<'a>(
        &'a self,
        device: &Device,
        pipeline: &Arc<ComputePipeline>,
        layout: &Arc<BindGroupLayout>,
    ) -> std::sync::MutexGuard<'a, HashMap<usize, wgpu::BindGroup>> {
        let key = Arc::as_ptr(pipeline) as usize;
        let mut cache = self.grad_bind_group_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("persistent_grad_bg"),
                layout: layout.as_ref(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.x_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.grad_partial_sums_buffer.as_entire_binding(),
                    },
                ],
            })
        });
        cache
    }

    /// Get or create a cached bind group for the fused single-pass path (fused_output_buffer at binding 2).
    fn cached_fused_bind_group<'a>(
        &'a self,
        device: &Device,
        pipeline: &Arc<ComputePipeline>,
        layout: &Arc<BindGroupLayout>,
    ) -> std::sync::MutexGuard<'a, HashMap<usize, wgpu::BindGroup>> {
        let key = Arc::as_ptr(pipeline) as usize;
        let mut cache = self.fused_bind_group_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("persistent_fused_bg"),
                layout: layout.as_ref(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.x_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.fused_output_buffer.as_entire_binding(),
                    },
                ],
            })
        });
        cache
    }

    /// Get the lazily-initialized reduce_sum pipeline for second-pass reduction.
    fn reduce_sum_pipeline(
        &self,
        device: &Device,
    ) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.reduce_sum_pipeline.get_or_init(|| {
            init_reduce_pipeline(
                device,
                include_str!("shaders/reduce_sum.wgsl"),
                "reduce_sum",
            )
        })
    }
}

/// Perform a second-pass GPU reduction on partial sums when the count exceeds SECOND_PASS_THRESHOLD.
///
/// Dispatches the generic `reduce_sum` shader to reduce the partial_sums_buffer (or
/// grad_partial_sums_buffer) down to `reduction_output_buffer` (single f32).
/// Then reads back and returns the value as f64.
#[allow(clippy::too_many_arguments)]
async fn reduce_partial_sums_gpu(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    reduce_pipeline: &Arc<ComputePipeline>,
    reduce_layout: &Arc<BindGroupLayout>,
    input_buffer: &wgpu::Buffer,
    input_count: u32,
    output_buffer: &wgpu::Buffer,
    staging_buffer: &wgpu::Buffer,
) -> Result<f64, String> {
    let params = ReduceSumParams {
        count: input_count,
        _padding1: 0,
        _padding2: 0,
        _padding3: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reduce_sum_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let second_workgroup_count = input_count.div_ceil(256);
    let output_size = (second_workgroup_count as u64) * 4;

    // When second pass produces multiple partial sums, we need temp buffers
    // larger than the pre-allocated single-f32 output/staging buffers.
    let temp_output = if second_workgroup_count > 1 {
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce_sum_temp_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    } else {
        None
    };
    let out_buf = temp_output.as_ref().unwrap_or(output_buffer);

    let temp_staging = if second_workgroup_count > 1 {
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce_sum_temp_staging"),
            size: output_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    } else {
        None
    };
    let staging = temp_staging.as_ref().unwrap_or(staging_buffer);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reduce_sum_bg"),
        layout: reduce_layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reduce_sum_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reduce_sum_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(reduce_pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(second_workgroup_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(out_buf, 0, staging, 0, output_size);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Reduce sum buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let results: &[f32] = bytemuck::cast_slice(&data);
    let total: f64 = results.iter().map(|&x| x as f64).sum();
    drop(data);
    staging.unmap();
    Ok(total)
}

/// Run a reduce kernel using persistent buffers (fast path for log_prob).
///
/// No buffer allocation occurs -- params are written in-place, the compute pass
/// is dispatched, partial sums are copied to the staging buffer, and the result
/// is read back and summed on the CPU.
pub async fn run_reduce_persistent<P: bytemuck::Pod>(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    buffers: &PersistentGpuBuffers,
    params: P,
) -> Result<GpuReduceResult, String> {
    if buffers.count == 0 {
        return Ok(GpuReduceResult {
            total_log_prob: 0.0,
        });
    }

    // Update params in-place (no allocation)
    queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

    let bg_key = Arc::as_ptr(pipeline) as usize;
    let use_second_pass = buffers.workgroup_count > SECOND_PASS_THRESHOLD;

    // Encode first-pass compute dispatch
    let encoder = {
        let cache = buffers.cached_bind_group(device, pipeline, layout);
        let bind_group = cache.get(&bg_key).unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("persistent_reduce_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("persistent_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.as_ref());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
        }
        if !use_second_pass {
            let partial_sums_size = (buffers.workgroup_count as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &buffers.partial_sums_buffer,
                0,
                &buffers.staging_buffer,
                0,
                partial_sums_size,
            );
        }
        encoder
    }; // cache lock released here

    queue.submit(std::iter::once(encoder.finish()));

    if use_second_pass {
        // Second-pass GPU reduction
        let (reduce_pipeline, reduce_layout) = buffers.reduce_sum_pipeline(device);
        let total = reduce_partial_sums_gpu(
            device,
            queue,
            reduce_pipeline,
            reduce_layout,
            &buffers.partial_sums_buffer,
            buffers.workgroup_count,
            &buffers.reduction_output_buffer,
            &buffers.reduction_staging_buffer,
        )
        .await?;
        Ok(GpuReduceResult {
            total_log_prob: total as f32,
        })
    } else {
        // CPU summation for small workgroup counts
        let buffer_slice = buffers.staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        poll_device(device);
        receiver
            .await
            .map_err(|_| "Channel cancelled")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let partial_sums: &[f32] = bytemuck::cast_slice(&data);
        let total_log_prob: f64 = partial_sums.iter().map(|&x| x as f64).sum();
        drop(data);
        buffers.staging_buffer.unmap();
        Ok(GpuReduceResult {
            total_log_prob: total_log_prob as f32,
        })
    }
}

/// Run a gradient reduce kernel using persistent buffers (fast path for gradients).
///
/// Identical dispatch pattern to `run_reduce_persistent` but returns a
/// `GpuGradReduceResult` (total_grad) instead of a `GpuReduceResult`.
pub async fn run_grad_reduce_persistent<P: bytemuck::Pod>(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    buffers: &PersistentGpuBuffers,
    params: P,
) -> Result<GpuGradReduceResult, String> {
    if buffers.count == 0 {
        return Ok(GpuGradReduceResult { total_grad: 0.0 });
    }

    // Update params in-place (no allocation)
    queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

    let bg_key = Arc::as_ptr(pipeline) as usize;
    let use_second_pass = buffers.workgroup_count > SECOND_PASS_THRESHOLD;

    // Encode compute pass while holding the bind group cache lock.
    // The grad standalone path uses partial_sums_buffer at binding 2 (same as logp path).
    let encoder = {
        let cache = buffers.cached_bind_group(device, pipeline, layout);
        let bind_group = cache.get(&bg_key).unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("persistent_grad_reduce_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("persistent_grad_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.as_ref());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
        }
        if !use_second_pass {
            let partial_sums_size = (buffers.workgroup_count as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &buffers.partial_sums_buffer,
                0,
                &buffers.staging_buffer,
                0,
                partial_sums_size,
            );
        }
        encoder
    }; // cache lock released here

    queue.submit(std::iter::once(encoder.finish()));

    if use_second_pass {
        let (reduce_pipeline, reduce_layout) = buffers.reduce_sum_pipeline(device);
        let total = reduce_partial_sums_gpu(
            device,
            queue,
            reduce_pipeline,
            reduce_layout,
            &buffers.partial_sums_buffer,
            buffers.workgroup_count,
            &buffers.reduction_output_buffer,
            &buffers.reduction_staging_buffer,
        )
        .await?;
        Ok(GpuGradReduceResult {
            total_grad: total as f32,
        })
    } else {
        let buffer_slice = buffers.staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        poll_device(device);
        receiver
            .await
            .map_err(|_| "Channel cancelled")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let partial_sums: &[f32] = bytemuck::cast_slice(&data);
        let total_grad: f64 = partial_sums.iter().map(|&x| x as f64).sum();
        drop(data);
        buffers.staging_buffer.unmap();
        Ok(GpuGradReduceResult {
            total_grad: total_grad as f32,
        })
    }
}

/// Run fused logp + grad reduce using persistent buffers (fast path).
///
/// The key optimization: ONE command encoder, ONE queue.submit, ONE poll_device
/// for both logp and grad kernels. This halves the synchronization overhead
/// compared to calling `run_reduce_persistent` and `run_grad_reduce_persistent`
/// separately.
#[allow(clippy::too_many_arguments)]
pub async fn run_fused_logp_and_grad_persistent<P: bytemuck::Pod>(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    logp_pipeline: &Arc<ComputePipeline>,
    logp_layout: &Arc<BindGroupLayout>,
    grad_pipeline: &Arc<ComputePipeline>,
    grad_layout: &Arc<BindGroupLayout>,
    buffers: &PersistentGpuBuffers,
    params: P,
) -> Result<FusedLogpGradResult, String> {
    if buffers.count == 0 {
        return Ok(FusedLogpGradResult {
            total_log_prob: 0.0,
            total_grad: 0.0,
        });
    }

    // Update params in-place once (shared by both kernels)
    queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

    // Use cached bind groups for both logp and grad paths
    let logp_bg_key = Arc::as_ptr(logp_pipeline) as usize;
    let grad_bg_key = Arc::as_ptr(grad_pipeline) as usize;
    let use_second_pass = buffers.workgroup_count > SECOND_PASS_THRESHOLD;

    // Encode both compute passes while holding cache locks
    let encoder = {
        let logp_cache = buffers.cached_bind_group(device, logp_pipeline, logp_layout);
        let grad_cache = buffers.cached_grad_bind_group(device, grad_pipeline, grad_layout);
        let logp_bind_group = logp_cache.get(&logp_bg_key).unwrap();
        let grad_bind_group = grad_cache.get(&grad_bg_key).unwrap();

        // ONE command encoder with TWO compute passes
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_logp_grad_enc"),
        });

        // Pass 1: logp reduce
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fused_logp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(logp_pipeline.as_ref());
            pass.set_bind_group(0, logp_bind_group, &[]);
            pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
        }

        // Pass 2: grad reduce
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fused_grad_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(grad_pipeline.as_ref());
            pass.set_bind_group(0, grad_bind_group, &[]);
            pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
        }

        if !use_second_pass {
            // TWO copy operations (only when doing CPU reduction)
            let partial_sums_size = (buffers.workgroup_count as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &buffers.partial_sums_buffer,
                0,
                &buffers.staging_buffer,
                0,
                partial_sums_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.grad_partial_sums_buffer,
                0,
                &buffers.grad_staging_buffer,
                0,
                partial_sums_size,
            );
        }
        encoder
    }; // cache locks released here

    // ONE submit for both first-pass dispatches
    queue.submit(std::iter::once(encoder.finish()));

    if use_second_pass {
        // Second-pass GPU reduction for both logp and grad
        let (reduce_pipeline, reduce_layout) = buffers.reduce_sum_pipeline(device);

        let total_log_prob = reduce_partial_sums_gpu(
            device,
            queue,
            reduce_pipeline,
            reduce_layout,
            &buffers.partial_sums_buffer,
            buffers.workgroup_count,
            &buffers.reduction_output_buffer,
            &buffers.reduction_staging_buffer,
        )
        .await?;

        let total_grad = reduce_partial_sums_gpu(
            device,
            queue,
            reduce_pipeline,
            reduce_layout,
            &buffers.grad_partial_sums_buffer,
            buffers.workgroup_count,
            &buffers.reduction_output_buffer,
            &buffers.reduction_staging_buffer,
        )
        .await?;

        Ok(FusedLogpGradResult {
            total_log_prob: total_log_prob as f32,
            total_grad: total_grad as f32,
        })
    } else {
        // CPU summation path
        // TWO map_async callbacks
        let logp_slice = buffers.staging_buffer.slice(..);
        let grad_slice = buffers.grad_staging_buffer.slice(..);

        let (logp_sender, logp_receiver) = futures_channel::oneshot::channel();
        let (grad_sender, grad_receiver) = futures_channel::oneshot::channel();

        logp_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = logp_sender.send(r);
        });
        grad_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = grad_sender.send(r);
        });

        // ONE poll drives both mappings
        poll_device(device);

        logp_receiver
            .await
            .map_err(|_| "Channel cancelled")?
            .map_err(|e| format!("Logp buffer mapping failed: {:?}", e))?;
        grad_receiver
            .await
            .map_err(|_| "Channel cancelled")?
            .map_err(|e| format!("Grad buffer mapping failed: {:?}", e))?;

        // Read both staging buffers and sum with f64 accumulation for precision
        let logp_data = logp_slice.get_mapped_range();
        let logp_partial_sums: &[f32] = bytemuck::cast_slice(&logp_data);
        let total_log_prob: f64 = logp_partial_sums.iter().map(|&x| x as f64).sum();
        drop(logp_data);
        buffers.staging_buffer.unmap();

        let grad_data = grad_slice.get_mapped_range();
        let grad_partial_sums: &[f32] = bytemuck::cast_slice(&grad_data);
        let total_grad: f64 = grad_partial_sums.iter().map(|&x| x as f64).sum();
        drop(grad_data);
        buffers.grad_staging_buffer.unmap();

        Ok(FusedLogpGradResult {
            total_log_prob: total_log_prob as f32,
            total_grad: total_grad as f32,
        })
    }
}

/// Run a single-pass fused logp+grad shader using persistent buffers.
///
/// Unlike `run_fused_logp_and_grad_persistent` which dispatches two separate shaders
/// (logp then grad) in one command encoder, this uses a single fused shader that computes
/// both logp and grad in one pass, sharing intermediate values to halve memory reads.
///
/// The fused shader writes interleaved output: [logp0, grad0, logp1, grad1, ...] where
/// each pair corresponds to one workgroup's partial sums.
#[allow(dead_code)]
pub async fn run_single_pass_fused_persistent<P: bytemuck::Pod>(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    fused_pipeline: &Arc<ComputePipeline>,
    fused_layout: &Arc<BindGroupLayout>,
    buffers: &PersistentGpuBuffers,
    params: P,
) -> Result<FusedLogpGradResult, String> {
    if buffers.count == 0 {
        return Ok(FusedLogpGradResult {
            total_log_prob: 0.0,
            total_grad: 0.0,
        });
    }

    // Update params in-place
    queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

    // Use cached bind group for fused path
    let fused_bg_key = Arc::as_ptr(fused_pipeline) as usize;

    let encoder = {
        let fused_cache = buffers.cached_fused_bind_group(device, fused_pipeline, fused_layout);
        let fused_bind_group = fused_cache.get(&fused_bg_key).unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("single_pass_fused_enc"),
        });

        // Single compute pass for both logp and grad
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("single_pass_fused"),
                timestamp_writes: None,
            });
            pass.set_pipeline(fused_pipeline.as_ref());
            pass.set_bind_group(0, fused_bind_group, &[]);
            pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
        }

        // Copy interleaved output to staging
        let fused_output_size = (buffers.workgroup_count as u64) * 2 * 4;
        encoder.copy_buffer_to_buffer(
            &buffers.fused_output_buffer,
            0,
            &buffers.fused_staging_buffer,
            0,
            fused_output_size,
        );
        encoder
    }; // cache lock released

    queue.submit(std::iter::once(encoder.finish()));

    // Read back interleaved results
    let fused_slice = buffers.fused_staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    fused_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Fused buffer mapping failed: {:?}", e))?;

    let data = fused_slice.get_mapped_range();
    let interleaved: &[f32] = bytemuck::cast_slice(&data);

    // De-interleave and sum with f64 accumulation
    let mut total_logp: f64 = 0.0;
    let mut total_grad: f64 = 0.0;
    for pair in interleaved.chunks(2) {
        total_logp += pair[0] as f64;
        total_grad += pair[1] as f64;
    }
    drop(data);
    buffers.fused_staging_buffer.unmap();

    Ok(FusedLogpGradResult {
        total_log_prob: total_logp as f32,
        total_grad: total_grad as f32,
    })
}

/// Run a single-pass fused shader that outputs N values per workgroup (logp + N-1 grads).
///
/// `num_outputs` specifies how many f32 values each workgroup writes:
/// - 2 for single-param distributions (Exponential, HalfNormal)
/// - 3 for two-param distributions (Normal, Beta, Gamma, InverseGamma, Cauchy, LogNormal)
/// - 4 for three-param distributions (StudentT)
pub async fn run_single_pass_fused_persistent_multi<P: bytemuck::Pod>(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    fused_pipeline: &Arc<ComputePipeline>,
    fused_layout: &Arc<BindGroupLayout>,
    buffers: &PersistentGpuBuffers,
    params: P,
    num_outputs: u32,
) -> Result<FusedMultiGradResult, String> {
    if buffers.count == 0 {
        return Ok(FusedMultiGradResult {
            total_log_prob: 0.0,
            total_grads: vec![0.0; (num_outputs - 1) as usize],
        });
    }

    queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

    let fused_bg_key = Arc::as_ptr(fused_pipeline) as usize;

    let encoder = {
        let fused_cache = buffers.cached_fused_bind_group(device, fused_pipeline, fused_layout);
        let fused_bind_group = fused_cache.get(&fused_bg_key).unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multi_grad_fused_enc"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("multi_grad_fused"),
                timestamp_writes: None,
            });
            pass.set_pipeline(fused_pipeline.as_ref());
            pass.set_bind_group(0, fused_bind_group, &[]);
            pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
        }

        let fused_output_size = (buffers.workgroup_count as u64) * (num_outputs as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &buffers.fused_output_buffer,
            0,
            &buffers.fused_staging_buffer,
            0,
            fused_output_size,
        );
        encoder
    };

    queue.submit(std::iter::once(encoder.finish()));

    let fused_output_size = (buffers.workgroup_count as u64) * (num_outputs as u64) * 4;
    let fused_slice = buffers.fused_staging_buffer.slice(..fused_output_size);
    let (sender, receiver) = futures_channel::oneshot::channel();
    fused_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Fused buffer mapping failed: {:?}", e))?;

    let data = fused_slice.get_mapped_range();
    let interleaved: &[f32] = bytemuck::cast_slice(&data);

    let n = num_outputs as usize;
    let mut total_logp: f64 = 0.0;
    let mut total_grads: Vec<f64> = vec![0.0; n - 1];
    for chunk in interleaved.chunks(n) {
        total_logp += chunk[0] as f64;
        for (i, grad) in total_grads.iter_mut().enumerate() {
            *grad += chunk[1 + i] as f64;
        }
    }
    drop(data);
    buffers.fused_staging_buffer.unmap();

    Ok(FusedMultiGradResult {
        total_log_prob: total_logp as f32,
        total_grads: total_grads.iter().map(|&g| g as f32).collect(),
    })
}

impl GpuContext {
    /// Create a new GPU context with async initialization
    ///
    /// Only initializes the wgpu adapter, device, and queue.
    /// Compute pipelines are lazily compiled on first use.
    pub async fn new() -> Result<Self, String> {
        // Use BROWSER_WEBGPU for WASM targets, platform-specific native backends otherwise.
        // NOTE: Backends::all() includes BROWSER_WEBGPU when the "webgpu" feature is enabled,
        // which causes request_adapter to hang on native macOS. Use PRIMARY to get only
        // Metal/Vulkan/DX12 based on the platform.
        #[cfg(target_arch = "wasm32")]
        let backends = wgpu::Backends::BROWSER_WEBGPU;
        #[cfg(not(target_arch = "wasm32"))]
        let backends = wgpu::Backends::PRIMARY;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to get GPU adapter: {:?}", e))?;

        // Use appropriate limits: WebGL2 defaults for WASM, device defaults for native
        #[cfg(target_arch = "wasm32")]
        let required_limits = wgpu::Limits::downlevel_webgl2_defaults();
        #[cfg(not(target_arch = "wasm32"))]
        let required_limits = wgpu::Limits::default();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bayesiangpu"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("Failed to get GPU device: {:?}", e))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            // All pipelines start uninitialized - created on first access
            normal: OnceLock::new(),
            half_normal: OnceLock::new(),
            normal_batch: OnceLock::new(),
            normal_reduce: OnceLock::new(),
            half_normal_reduce: OnceLock::new(),
            exponential_reduce: OnceLock::new(),
            gamma_reduce: OnceLock::new(),
            beta_reduce: OnceLock::new(),
            inverse_gamma_reduce: OnceLock::new(),
            uniform_reduce: OnceLock::new(),
            cauchy_reduce: OnceLock::new(),
            student_t_reduce: OnceLock::new(),
            lognormal_reduce: OnceLock::new(),
            bernoulli_reduce: OnceLock::new(),
            binomial_reduce: OnceLock::new(),
            poisson_reduce: OnceLock::new(),
            negative_binomial_reduce: OnceLock::new(),
            categorical_reduce: OnceLock::new(),
            reduce_sum: OnceLock::new(),
            normal_grad_reduce: OnceLock::new(),
            half_normal_grad_reduce: OnceLock::new(),
            exponential_grad_reduce: OnceLock::new(),
            beta_grad_reduce: OnceLock::new(),
            gamma_grad_reduce: OnceLock::new(),
            inverse_gamma_grad_reduce: OnceLock::new(),
            student_t_grad_reduce: OnceLock::new(),
            cauchy_grad_reduce: OnceLock::new(),
            lognormal_grad_reduce: OnceLock::new(),
            normal_fused_reduce: OnceLock::new(),
            half_normal_fused_reduce: OnceLock::new(),
            exponential_fused_reduce: OnceLock::new(),
            gamma_fused_reduce: OnceLock::new(),
            beta_fused_reduce: OnceLock::new(),
            inverse_gamma_fused_reduce: OnceLock::new(),
            student_t_fused_reduce: OnceLock::new(),
            cauchy_fused_reduce: OnceLock::new(),
            lognormal_fused_reduce: OnceLock::new(),
            normal_linpred_fused_reduce: OnceLock::new(),
            normal_indexed_reduce: OnceLock::new(),
        })
    }

    // =========================================================================
    // LAZY PIPELINE ACCESSORS
    // =========================================================================

    /// Get the Normal distribution pipeline + layout, compiling on first access
    fn normal_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.normal.get_or_init(|| {
            init_basic_pipeline(&self.device, include_str!("shaders/normal.wgsl"), "normal")
        })
    }

    /// Get the HalfNormal distribution pipeline + layout, compiling on first access
    fn half_normal_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.half_normal.get_or_init(|| {
            init_basic_pipeline(
                &self.device,
                include_str!("shaders/half_normal.wgsl"),
                "half_normal",
            )
        })
    }

    /// Get the Normal batch pipeline + layout, compiling on first access
    fn normal_batch_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.normal_batch.get_or_init(|| {
            let layout =
                create_batch_4_binding_layout(&self.device, "normal_batch_bind_group_layout");
            let pipeline = create_pipeline(
                &self.device,
                include_str!("shaders/normal_batch.wgsl"),
                "normal_batch_pipeline",
                &layout,
            );
            (Arc::new(pipeline), Arc::new(layout))
        })
    }

    // --- Log-prob reduce lazy accessors ---

    fn normal_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.normal_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/normal_reduce.wgsl"),
                "normal_reduce",
            )
        })
    }

    fn half_normal_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.half_normal_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/half_normal_reduce.wgsl"),
                "half_normal_reduce",
            )
        })
    }

    fn exponential_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.exponential_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/exponential_reduce.wgsl"),
                "exponential_reduce",
            )
        })
    }

    fn gamma_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.gamma_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/gamma_reduce.wgsl"),
                "gamma_reduce",
            )
        })
    }

    fn beta_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.beta_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/beta_reduce.wgsl"),
                "beta_reduce",
            )
        })
    }

    fn inverse_gamma_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.inverse_gamma_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/inverse_gamma_reduce.wgsl"),
                "inverse_gamma_reduce",
            )
        })
    }

    fn uniform_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.uniform_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/uniform_reduce.wgsl"),
                "uniform_reduce",
            )
        })
    }

    fn cauchy_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.cauchy_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/cauchy_reduce.wgsl"),
                "cauchy_reduce",
            )
        })
    }

    fn student_t_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.student_t_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/student_t_reduce.wgsl"),
                "student_t_reduce",
            )
        })
    }

    fn lognormal_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.lognormal_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/lognormal_reduce.wgsl"),
                "lognormal_reduce",
            )
        })
    }

    fn bernoulli_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.bernoulli_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/bernoulli_reduce.wgsl"),
                "bernoulli_reduce",
            )
        })
    }

    fn binomial_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.binomial_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/binomial_reduce.wgsl"),
                "binomial_reduce",
            )
        })
    }

    fn poisson_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.poisson_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/poisson_reduce.wgsl"),
                "poisson_reduce",
            )
        })
    }

    fn negative_binomial_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.negative_binomial_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/negative_binomial_reduce.wgsl"),
                "negative_binomial_reduce",
            )
        })
    }

    fn categorical_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.categorical_reduce.get_or_init(|| {
            let layout = create_categorical_reduce_layout(
                &self.device,
                "categorical_reduce_bind_group_layout",
            );
            let pipeline = create_pipeline(
                &self.device,
                include_str!("shaders/categorical_reduce.wgsl"),
                "categorical_reduce_pipeline",
                &layout,
            );
            (Arc::new(pipeline), Arc::new(layout))
        })
    }

    // --- Generic second-pass reduction ---

    fn reduce_sum_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.reduce_sum.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/reduce_sum.wgsl"),
                "reduce_sum",
            )
        })
    }

    /// Get the reduce_sum pipeline (public for second-pass reduction in persistent dispatch)
    pub fn reduce_sum_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.reduce_sum_lazy().0)
    }
    pub fn reduce_sum_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.reduce_sum_lazy().1)
    }

    // --- Gradient reduce lazy accessors ---

    fn normal_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.normal_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/normal_grad_reduce.wgsl"),
                "normal_grad_reduce",
            )
        })
    }

    fn half_normal_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.half_normal_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/half_normal_grad_reduce.wgsl"),
                "half_normal_grad_reduce",
            )
        })
    }

    fn exponential_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.exponential_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/exponential_grad_reduce.wgsl"),
                "exponential_grad_reduce",
            )
        })
    }

    fn beta_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.beta_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/beta_grad_reduce.wgsl"),
                "beta_grad_reduce",
            )
        })
    }

    fn gamma_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.gamma_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/gamma_grad_reduce.wgsl"),
                "gamma_grad_reduce",
            )
        })
    }

    fn inverse_gamma_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.inverse_gamma_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/inverse_gamma_grad_reduce.wgsl"),
                "inverse_gamma_grad_reduce",
            )
        })
    }

    fn student_t_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.student_t_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/student_t_grad_reduce.wgsl"),
                "student_t_grad_reduce",
            )
        })
    }

    fn cauchy_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.cauchy_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/cauchy_grad_reduce.wgsl"),
                "cauchy_grad_reduce",
            )
        })
    }

    fn lognormal_grad_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.lognormal_grad_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/lognormal_grad_reduce.wgsl"),
                "lognormal_grad_reduce",
            )
        })
    }

    // =========================================================================
    // FUSED SINGLE-PASS PIPELINE ACCESSORS
    // =========================================================================

    fn normal_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.normal_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/normal_fused_reduce.wgsl"),
                "normal_fused_reduce",
            )
        })
    }

    fn half_normal_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.half_normal_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/half_normal_fused_reduce.wgsl"),
                "half_normal_fused_reduce",
            )
        })
    }

    fn exponential_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.exponential_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/exponential_fused_reduce.wgsl"),
                "exponential_fused_reduce",
            )
        })
    }

    fn gamma_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.gamma_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/gamma_fused_reduce.wgsl"),
                "gamma_fused_reduce",
            )
        })
    }

    fn beta_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.beta_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/beta_fused_reduce.wgsl"),
                "beta_fused_reduce",
            )
        })
    }

    fn inverse_gamma_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.inverse_gamma_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/inverse_gamma_fused_reduce.wgsl"),
                "inverse_gamma_fused_reduce",
            )
        })
    }

    fn student_t_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.student_t_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/student_t_fused_reduce.wgsl"),
                "student_t_fused_reduce",
            )
        })
    }

    fn cauchy_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.cauchy_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/cauchy_fused_reduce.wgsl"),
                "cauchy_fused_reduce",
            )
        })
    }

    fn lognormal_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.lognormal_fused_reduce.get_or_init(|| {
            init_reduce_pipeline(
                &self.device,
                include_str!("shaders/lognormal_fused_reduce.wgsl"),
                "lognormal_fused_reduce",
            )
        })
    }

    // --- Indexed parameter kernel (hierarchical models) ---

    fn normal_indexed_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.normal_indexed_reduce.get_or_init(|| {
            let layout = create_indexed_reduce_layout(
                &self.device,
                "normal_indexed_reduce_bind_group_layout",
            );
            let pipeline = create_pipeline(
                &self.device,
                include_str!("shaders/normal_indexed_reduce.wgsl"),
                "normal_indexed_reduce_pipeline",
                &layout,
            );
            (Arc::new(pipeline), Arc::new(layout))
        })
    }

    // --- Linear predictor kernel ---

    fn normal_linpred_fused_reduce_lazy(&self) -> &(Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        self.normal_linpred_fused_reduce.get_or_init(|| {
            init_linpred_pipeline(
                &self.device,
                include_str!("shaders/normal_linpred_fused_reduce.wgsl"),
                "normal_linpred_fused_reduce",
            )
        })
    }

    // =========================================================================
    // CLONE METHODS - Basic
    // =========================================================================

    pub fn device_clone(&self) -> Arc<Device> {
        Arc::clone(&self.device)
    }
    pub fn queue_clone(&self) -> Arc<Queue> {
        Arc::clone(&self.queue)
    }
    /// Get a reference to the underlying device (avoids Arc clone for buffer creation)
    pub fn device_ref(&self) -> &Device {
        &self.device
    }
    /// Get a reference to the underlying queue (avoids Arc clone for buffer creation)
    pub fn queue_ref(&self) -> &Queue {
        &self.queue
    }
    pub fn normal_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_lazy().0)
    }
    pub fn normal_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_lazy().1)
    }
    pub fn half_normal_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.half_normal_lazy().0)
    }
    pub fn half_normal_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.half_normal_lazy().1)
    }
    pub fn normal_batch_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_batch_lazy().0)
    }
    pub fn normal_batch_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_batch_lazy().1)
    }

    // =========================================================================
    // CLONE METHODS - Log-prob reduce
    // =========================================================================

    pub fn normal_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_reduce_lazy().0)
    }
    pub fn normal_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_reduce_lazy().1)
    }
    pub fn half_normal_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.half_normal_reduce_lazy().0)
    }
    pub fn half_normal_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.half_normal_reduce_lazy().1)
    }
    pub fn exponential_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.exponential_reduce_lazy().0)
    }
    pub fn exponential_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.exponential_reduce_lazy().1)
    }
    pub fn gamma_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.gamma_reduce_lazy().0)
    }
    pub fn gamma_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.gamma_reduce_lazy().1)
    }
    pub fn beta_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.beta_reduce_lazy().0)
    }
    pub fn beta_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.beta_reduce_lazy().1)
    }
    pub fn inverse_gamma_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.inverse_gamma_reduce_lazy().0)
    }
    pub fn inverse_gamma_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.inverse_gamma_reduce_lazy().1)
    }
    pub fn uniform_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.uniform_reduce_lazy().0)
    }
    pub fn uniform_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.uniform_reduce_lazy().1)
    }
    pub fn cauchy_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.cauchy_reduce_lazy().0)
    }
    pub fn cauchy_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.cauchy_reduce_lazy().1)
    }
    pub fn student_t_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.student_t_reduce_lazy().0)
    }
    pub fn student_t_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.student_t_reduce_lazy().1)
    }
    pub fn lognormal_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.lognormal_reduce_lazy().0)
    }
    pub fn lognormal_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.lognormal_reduce_lazy().1)
    }
    pub fn bernoulli_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.bernoulli_reduce_lazy().0)
    }
    pub fn bernoulli_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.bernoulli_reduce_lazy().1)
    }
    pub fn binomial_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.binomial_reduce_lazy().0)
    }
    pub fn binomial_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.binomial_reduce_lazy().1)
    }
    pub fn poisson_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.poisson_reduce_lazy().0)
    }
    pub fn poisson_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.poisson_reduce_lazy().1)
    }
    pub fn negative_binomial_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.negative_binomial_reduce_lazy().0)
    }
    pub fn negative_binomial_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.negative_binomial_reduce_lazy().1)
    }
    pub fn categorical_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.categorical_reduce_lazy().0)
    }
    pub fn categorical_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.categorical_reduce_lazy().1)
    }

    // =========================================================================
    // CLONE METHODS - Gradient reduce
    // =========================================================================

    pub fn normal_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_grad_reduce_lazy().0)
    }
    pub fn normal_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_grad_reduce_lazy().1)
    }
    pub fn half_normal_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.half_normal_grad_reduce_lazy().0)
    }
    pub fn half_normal_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.half_normal_grad_reduce_lazy().1)
    }
    pub fn exponential_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.exponential_grad_reduce_lazy().0)
    }
    pub fn exponential_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.exponential_grad_reduce_lazy().1)
    }
    pub fn beta_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.beta_grad_reduce_lazy().0)
    }
    pub fn beta_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.beta_grad_reduce_lazy().1)
    }
    pub fn gamma_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.gamma_grad_reduce_lazy().0)
    }
    pub fn gamma_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.gamma_grad_reduce_lazy().1)
    }
    pub fn inverse_gamma_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.inverse_gamma_grad_reduce_lazy().0)
    }
    pub fn inverse_gamma_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.inverse_gamma_grad_reduce_lazy().1)
    }
    pub fn student_t_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.student_t_grad_reduce_lazy().0)
    }
    pub fn student_t_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.student_t_grad_reduce_lazy().1)
    }
    pub fn cauchy_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.cauchy_grad_reduce_lazy().0)
    }
    pub fn cauchy_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.cauchy_grad_reduce_lazy().1)
    }
    pub fn lognormal_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.lognormal_grad_reduce_lazy().0)
    }
    pub fn lognormal_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.lognormal_grad_reduce_lazy().1)
    }

    // =========================================================================
    // FUSED SINGLE-PASS PIPELINE CLONES
    // =========================================================================

    pub fn normal_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_fused_reduce_lazy().0)
    }
    pub fn normal_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_fused_reduce_lazy().1)
    }
    pub fn half_normal_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.half_normal_fused_reduce_lazy().0)
    }
    pub fn half_normal_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.half_normal_fused_reduce_lazy().1)
    }
    pub fn exponential_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.exponential_fused_reduce_lazy().0)
    }
    pub fn exponential_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.exponential_fused_reduce_lazy().1)
    }
    pub fn gamma_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.gamma_fused_reduce_lazy().0)
    }
    pub fn gamma_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.gamma_fused_reduce_lazy().1)
    }
    pub fn beta_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.beta_fused_reduce_lazy().0)
    }
    pub fn beta_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.beta_fused_reduce_lazy().1)
    }
    pub fn inverse_gamma_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.inverse_gamma_fused_reduce_lazy().0)
    }
    pub fn inverse_gamma_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.inverse_gamma_fused_reduce_lazy().1)
    }
    pub fn student_t_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.student_t_fused_reduce_lazy().0)
    }
    pub fn student_t_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.student_t_fused_reduce_lazy().1)
    }
    pub fn cauchy_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.cauchy_fused_reduce_lazy().0)
    }
    pub fn cauchy_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.cauchy_fused_reduce_lazy().1)
    }
    pub fn lognormal_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.lognormal_fused_reduce_lazy().0)
    }
    pub fn lognormal_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.lognormal_fused_reduce_lazy().1)
    }

    // =========================================================================
    // CLONE METHODS - Indexed parameter kernel
    // =========================================================================

    pub fn normal_indexed_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_indexed_reduce_lazy().0)
    }
    pub fn normal_indexed_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_indexed_reduce_lazy().1)
    }

    // =========================================================================
    // CLONE METHODS - Linear predictor kernel
    // =========================================================================

    pub fn normal_linpred_fused_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_linpred_fused_reduce_lazy().0)
    }
    pub fn normal_linpred_fused_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_linpred_fused_reduce_lazy().1)
    }
}

// =============================================================================
// KERNEL RUNNER FUNCTIONS
// =============================================================================

/// Run Normal distribution kernel (single point)
pub async fn run_normal_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x: f32,
    mu: f32,
    sigma: f32,
) -> Result<GpuResult, String> {
    let params = NormalParams {
        x,
        mu,
        sigma,
        _padding: 0.0,
    };
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normal_input"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normal_output"),
        size: 8,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: 8,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normal_bind_group"),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("normal_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normal_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 8);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let result: [f32; 2] = bytemuck::cast_slice(&data)[0..2].try_into().unwrap();
    drop(data);
    staging_buffer.unmap();
    Ok(GpuResult {
        log_prob: result[0],
        grad: result[1],
    })
}

/// Run HalfNormal distribution kernel (single point)
pub async fn run_half_normal_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x: f32,
    sigma: f32,
) -> Result<GpuResult, String> {
    let params = HalfNormalParams {
        x,
        sigma,
        _padding1: 0.0,
        _padding2: 0.0,
    };
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("half_normal_input"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("half_normal_output"),
        size: 8,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: 8,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("half_normal_bind_group"),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("half_normal_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("half_normal_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 8);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let result: [f32; 2] = bytemuck::cast_slice(&data)[0..2].try_into().unwrap();
    drop(data);
    staging_buffer.unmap();
    Ok(GpuResult {
        log_prob: result[0],
        grad: result[1],
    })
}

/// Run Normal batch kernel
pub async fn run_normal_batch_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuBatchResult, String> {
    let count = x_values.len() as u32;
    if count == 0 {
        return Ok(GpuBatchResult {
            log_probs: vec![],
            grads: vec![],
        });
    }

    let params = NormalBatchParams {
        mu,
        sigma,
        count,
        _padding: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normal_batch_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normal_batch_x"),
        contents: bytemuck::cast_slice(x_values),
        usage: BufferUsages::STORAGE,
    });
    let output_size = (count as u64) * 4;
    let log_probs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("log_probs"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let grads_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grads"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let log_probs_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_lp"),
        size: output_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let grads_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_g"),
        size: output_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normal_batch_bind_group"),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: log_probs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grads_buffer.as_entire_binding(),
            },
        ],
    });

    let workgroup_count = count.div_ceil(256);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("normal_batch_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normal_batch_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&log_probs_buffer, 0, &log_probs_staging, 0, output_size);
    encoder.copy_buffer_to_buffer(&grads_buffer, 0, &grads_staging, 0, output_size);
    queue.submit(std::iter::once(encoder.finish()));

    let lp_slice = log_probs_staging.slice(..);
    let g_slice = grads_staging.slice(..);
    let (lp_sender, lp_receiver) = futures_channel::oneshot::channel();
    let (g_sender, g_receiver) = futures_channel::oneshot::channel();
    lp_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = lp_sender.send(r);
    });
    g_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = g_sender.send(r);
    });
    poll_device(device);
    lp_receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;
    g_receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let lp_data = lp_slice.get_mapped_range();
    let log_probs: Vec<f32> = bytemuck::cast_slice(&lp_data).to_vec();
    drop(lp_data);
    log_probs_staging.unmap();
    let g_data = g_slice.get_mapped_range();
    let grads: Vec<f32> = bytemuck::cast_slice(&g_data).to_vec();
    drop(g_data);
    grads_staging.unmap();
    Ok(GpuBatchResult { log_probs, grads })
}

// =============================================================================
// LOG-PROB REDUCE KERNEL RUNNERS
// =============================================================================

/// Generic reduce kernel runner for log_prob
async fn run_reduce_kernel_impl<P: bytemuck::Pod>(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    params: P,
    x_values: &[f32],
    label: &str,
) -> Result<GpuReduceResult, String> {
    let count = x_values.len() as u32;
    if count == 0 {
        return Ok(GpuReduceResult {
            total_log_prob: 0.0,
        });
    }

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{}_params", label)),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{}_x", label)),
        contents: bytemuck::cast_slice(x_values),
        usage: BufferUsages::STORAGE,
    });

    let workgroup_count = count.div_ceil(ELEMS_PER_WORKGROUP);
    let partial_sums_size = (workgroup_count as u64) * 4;
    let partial_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{}_partial_sums", label)),
        size: partial_sums_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_partial_sums"),
        size: partial_sums_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{}_bind_group", label)),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: partial_sums_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("{}_encoder", label)),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{}_compute_pass", label)),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &partial_sums_buffer,
        0,
        &staging_buffer,
        0,
        partial_sums_size,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let partial_sums: &[f32] = bytemuck::cast_slice(&data);
    let total_log_prob: f64 = partial_sums.iter().map(|&x| x as f64).sum();
    drop(data);
    staging_buffer.unmap();
    Ok(GpuReduceResult {
        total_log_prob: total_log_prob as f32,
    })
}

pub async fn run_normal_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        NormalBatchParams {
            mu,
            sigma,
            count: x_values.len() as u32,
            _padding: 0,
        },
        x_values,
        "normal_reduce",
    )
    .await
}

pub async fn run_half_normal_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    sigma: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        HalfNormalReduceParams {
            sigma,
            count: x_values.len() as u32,
            _padding1: 0,
            _padding2: 0,
        },
        x_values,
        "half_normal_reduce",
    )
    .await
}

pub async fn run_exponential_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    lambda: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        ExponentialReduceParams {
            lambda,
            count: x_values.len() as u32,
            _padding1: 0,
            _padding2: 0,
        },
        x_values,
        "exponential_reduce",
    )
    .await
}

pub async fn run_gamma_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        GammaReduceParams {
            alpha,
            beta,
            count: x_values.len() as u32,
            log_norm: gamma_log_norm(alpha, beta),
        },
        x_values,
        "gamma_reduce",
    )
    .await
}

pub async fn run_beta_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        BetaReduceParams {
            alpha,
            beta,
            count: x_values.len() as u32,
            log_norm: beta_log_norm(alpha, beta),
        },
        x_values,
        "beta_reduce",
    )
    .await
}

pub async fn run_inverse_gamma_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        InverseGammaReduceParams {
            alpha,
            beta,
            count: x_values.len() as u32,
            log_norm: inverse_gamma_log_norm(alpha, beta),
        },
        x_values,
        "inverse_gamma_reduce",
    )
    .await
}

pub async fn run_uniform_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    low: f32,
    high: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        UniformReduceParams {
            low,
            high,
            count: x_values.len() as u32,
            _padding: 0,
        },
        x_values,
        "uniform_reduce",
    )
    .await
}

pub async fn run_cauchy_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    loc: f32,
    scale: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        CauchyReduceParams {
            loc,
            scale,
            count: x_values.len() as u32,
            _padding: 0,
        },
        x_values,
        "cauchy_reduce",
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn run_student_t_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    loc: f32,
    scale: f32,
    nu: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        StudentTReduceParams {
            loc,
            scale,
            nu,
            count: x_values.len() as u32,
            log_norm: student_t_log_norm(nu, scale),
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        },
        x_values,
        "student_t_reduce",
    )
    .await
}

pub async fn run_lognormal_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        LogNormalReduceParams {
            mu,
            sigma,
            count: x_values.len() as u32,
            log_norm: lognormal_log_norm(sigma),
        },
        x_values,
        "lognormal_reduce",
    )
    .await
}

pub async fn run_bernoulli_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    p: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        BernoulliReduceParams {
            p,
            count: x_values.len() as u32,
            _padding1: 0,
            _padding2: 0,
        },
        x_values,
        "bernoulli_reduce",
    )
    .await
}

pub async fn run_binomial_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    n: f32,
    p: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        BinomialReduceParams {
            n,
            p,
            count: x_values.len() as u32,
            _padding: 0,
        },
        x_values,
        "binomial_reduce",
    )
    .await
}

pub async fn run_poisson_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    lambda: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        PoissonReduceParams {
            lambda,
            count: x_values.len() as u32,
            _padding1: 0,
            _padding2: 0,
        },
        x_values,
        "poisson_reduce",
    )
    .await
}

pub async fn run_negative_binomial_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    r: f32,
    p: f32,
) -> Result<GpuReduceResult, String> {
    run_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        NegativeBinomialReduceParams {
            r,
            p,
            count: x_values.len() as u32,
            _padding: 0,
        },
        x_values,
        "negative_binomial_reduce",
    )
    .await
}

/// Categorical reduce kernel (4-binding: params, x_values, probs, partial_sums)
pub async fn run_categorical_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    probs: &[f32],
) -> Result<GpuReduceResult, String> {
    let count = x_values.len() as u32;
    let num_categories = probs.len() as u32;
    if count == 0 {
        return Ok(GpuReduceResult {
            total_log_prob: 0.0,
        });
    }

    let params = CategoricalReduceParams {
        num_categories,
        count,
        _padding1: 0,
        _padding2: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("categorical_reduce_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("categorical_reduce_x"),
        contents: bytemuck::cast_slice(x_values),
        usage: BufferUsages::STORAGE,
    });
    let probs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("categorical_reduce_probs"),
        contents: bytemuck::cast_slice(probs),
        usage: BufferUsages::STORAGE,
    });

    let workgroup_count = count.div_ceil(ELEMS_PER_WORKGROUP);
    let partial_sums_size = (workgroup_count as u64) * 4;
    let partial_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("categorical_reduce_partial_sums"),
        size: partial_sums_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_partial_sums"),
        size: partial_sums_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("categorical_reduce_bind_group"),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: probs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: partial_sums_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("categorical_reduce_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("categorical_reduce_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &partial_sums_buffer,
        0,
        &staging_buffer,
        0,
        partial_sums_size,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let partial_sums: &[f32] = bytemuck::cast_slice(&data);
    let total_log_prob: f64 = partial_sums.iter().map(|&x| x as f64).sum();
    drop(data);
    staging_buffer.unmap();
    Ok(GpuReduceResult {
        total_log_prob: total_log_prob as f32,
    })
}

// =============================================================================
// GRADIENT REDUCE KERNEL RUNNERS
// =============================================================================

/// Generic reduce kernel runner for gradients
async fn run_grad_reduce_kernel_impl<P: bytemuck::Pod>(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    params: P,
    x_values: &[f32],
    label: &str,
) -> Result<GpuGradReduceResult, String> {
    let count = x_values.len() as u32;
    if count == 0 {
        return Ok(GpuGradReduceResult { total_grad: 0.0 });
    }

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{}_params", label)),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{}_x", label)),
        contents: bytemuck::cast_slice(x_values),
        usage: BufferUsages::STORAGE,
    });

    let workgroup_count = count.div_ceil(ELEMS_PER_WORKGROUP);
    let partial_sums_size = (workgroup_count as u64) * 4;
    let partial_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{}_partial_sums", label)),
        size: partial_sums_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_partial_sums"),
        size: partial_sums_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{}_bind_group", label)),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: partial_sums_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("{}_encoder", label)),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{}_compute_pass", label)),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &partial_sums_buffer,
        0,
        &staging_buffer,
        0,
        partial_sums_size,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let partial_sums: &[f32] = bytemuck::cast_slice(&data);
    let total_grad: f64 = partial_sums.iter().map(|&x| x as f64).sum();
    drop(data);
    staging_buffer.unmap();
    Ok(GpuGradReduceResult {
        total_grad: total_grad as f32,
    })
}

pub async fn run_normal_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        NormalBatchParams {
            mu,
            sigma,
            count: x_values.len() as u32,
            _padding: 0,
        },
        x_values,
        "normal_grad_reduce",
    )
    .await
}

pub async fn run_half_normal_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    sigma: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        HalfNormalReduceParams {
            sigma,
            count: x_values.len() as u32,
            _padding1: 0,
            _padding2: 0,
        },
        x_values,
        "half_normal_grad_reduce",
    )
    .await
}

pub async fn run_exponential_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    lambda: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        ExponentialReduceParams {
            lambda,
            count: x_values.len() as u32,
            _padding1: 0,
            _padding2: 0,
        },
        x_values,
        "exponential_grad_reduce",
    )
    .await
}

pub async fn run_beta_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        BetaReduceParams {
            alpha,
            beta,
            count: x_values.len() as u32,
            log_norm: beta_log_norm(alpha, beta),
        },
        x_values,
        "beta_grad_reduce",
    )
    .await
}

pub async fn run_gamma_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        GammaReduceParams {
            alpha,
            beta,
            count: x_values.len() as u32,
            log_norm: gamma_log_norm(alpha, beta),
        },
        x_values,
        "gamma_grad_reduce",
    )
    .await
}

pub async fn run_inverse_gamma_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    alpha: f32,
    beta: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        InverseGammaReduceParams {
            alpha,
            beta,
            count: x_values.len() as u32,
            log_norm: inverse_gamma_log_norm(alpha, beta),
        },
        x_values,
        "inverse_gamma_grad_reduce",
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn run_student_t_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    loc: f32,
    scale: f32,
    nu: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        StudentTReduceParams {
            loc,
            scale,
            nu,
            count: x_values.len() as u32,
            log_norm: student_t_log_norm(nu, scale),
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        },
        x_values,
        "student_t_grad_reduce",
    )
    .await
}

pub async fn run_cauchy_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    loc: f32,
    scale: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        CauchyReduceParams {
            loc,
            scale,
            count: x_values.len() as u32,
            _padding: 0,
        },
        x_values,
        "cauchy_grad_reduce",
    )
    .await
}

pub async fn run_lognormal_grad_reduce_kernel(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    x_values: &[f32],
    mu: f32,
    sigma: f32,
) -> Result<GpuGradReduceResult, String> {
    run_grad_reduce_kernel_impl(
        device,
        queue,
        pipeline,
        layout,
        LogNormalReduceParams {
            mu,
            sigma,
            count: x_values.len() as u32,
            log_norm: lognormal_log_norm(sigma),
        },
        x_values,
        "lognormal_grad_reduce",
    )
    .await
}

// =============================================================================
// INDEXED PARAMETER KERNEL (HIERARCHICAL MODELS)
// =============================================================================

/// Result from the indexed Normal reduce kernel.
pub struct IndexedNormalResult {
    pub total_log_prob: f64,
    pub grad_sigma: f64,
    pub grad_theta: Vec<f64>,
}

/// Run the indexed Normal reduce kernel for y[i] ~ Normal(theta[group[i]], sigma).
///
/// Dispatches the indexed shader for total logp and grad_sigma,
/// then computes per-group theta gradients on CPU from pre-sorted observations.
#[allow(clippy::too_many_arguments, dead_code)]
pub async fn run_normal_indexed_reduce(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    y_sorted: &[f32],
    theta: &[f32],
    group_idx: &[u32],
    group_boundaries: &[usize],
    sigma: f32,
) -> Result<IndexedNormalResult, String> {
    let count = y_sorted.len() as u32;

    if count == 0 {
        return Ok(IndexedNormalResult {
            total_log_prob: 0.0,
            grad_sigma: 0.0,
            grad_theta: vec![0.0; theta.len()],
        });
    }

    let workgroup_count = count.div_ceil(ELEMS_PER_WORKGROUP);
    let num_outputs: u32 = 2; // logp + grad_sigma

    let params = NormalIndexedReduceParams {
        sigma,
        count,
        num_groups: theta.len() as u32,
        _padding: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indexed_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let y_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indexed_y"),
        contents: bytemuck::cast_slice(y_sorted),
        usage: BufferUsages::STORAGE,
    });

    let theta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indexed_theta"),
        contents: bytemuck::cast_slice(theta),
        usage: BufferUsages::STORAGE,
    });

    let group_idx_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indexed_group_idx"),
        contents: bytemuck::cast_slice(group_idx),
        usage: BufferUsages::STORAGE,
    });

    let output_size = (workgroup_count as u64) * (num_outputs as u64) * 4;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indexed_output"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indexed_staging"),
        size: output_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("indexed_bg"),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: theta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: group_idx_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("indexed_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("indexed_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..output_size);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Indexed buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let interleaved: &[f32] = bytemuck::cast_slice(&data);

    let n = num_outputs as usize;
    let mut total_logp: f64 = 0.0;
    let mut total_grad_sigma: f64 = 0.0;
    for chunk in interleaved.chunks(n) {
        total_logp += chunk[0] as f64;
        total_grad_sigma += chunk[1] as f64;
    }
    drop(data);
    staging_buffer.unmap();

    // Compute per-group theta gradients on CPU from pre-sorted data.
    // For group k: grad_theta[k] = sum_{i in group k} (y[i] - theta[k]) / sigma^2
    let sigma2 = (sigma as f64) * (sigma as f64);
    let num_groups_usize = theta.len();
    let mut grad_theta = vec![0.0f64; num_groups_usize];
    for k in 0..num_groups_usize {
        let start = group_boundaries[k];
        let end = group_boundaries[k + 1];
        let mu_k = theta[k] as f64;
        let mut sum = 0.0f64;
        for &y in &y_sorted[start..end] {
            sum += (y as f64) - mu_k;
        }
        grad_theta[k] = sum / sigma2;
    }

    Ok(IndexedNormalResult {
        total_log_prob: total_logp,
        grad_sigma: total_grad_sigma,
        grad_theta,
    })
}

// =============================================================================
// MULTI-CHAIN GPU BUFFERS
// =============================================================================

/// Per-chain GPU buffers for multi-chain batched dispatch.
pub struct ChainBuffers {
    pub params_buffer: wgpu::Buffer,
    pub partial_sums_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
    pub grad_partial_sums_buffer: wgpu::Buffer,
    pub grad_staging_buffer: wgpu::Buffer,
}

/// Multi-chain GPU buffers that share observation data across chains.
///
/// All chains observe the same data (x_values), but each chain has different
/// parameter values. One shared x_buffer + N independent buffer sets.
pub struct MultiChainGpuBuffers {
    pub x_buffer: wgpu::Buffer,
    pub num_chains: u32,
    pub count: u32,
    pub workgroup_count: u32,
    pub chain_buffers: Vec<ChainBuffers>,
    pub params_capacity: u64,
}

/// Create multi-chain GPU buffers with shared observation data.
pub fn create_multi_chain_buffers(
    device: &Device,
    queue: &Queue,
    x_values: &[f32],
    max_params_size: u64,
    num_chains: u32,
) -> MultiChainGpuBuffers {
    let _ = queue;
    let count = x_values.len() as u32;
    let workgroup_count = count.div_ceil(ELEMS_PER_WORKGROUP);
    let partial_sums_size = (workgroup_count as u64) * 4;

    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("multichain_x"),
        contents: bytemuck::cast_slice(x_values),
        usage: BufferUsages::STORAGE,
    });

    let mut chain_buffers = Vec::with_capacity(num_chains as usize);
    for chain_idx in 0..num_chains {
        let lbl = format!("chain{}", chain_idx);
        chain_buffers.push(ChainBuffers {
            params_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_params", lbl)),
                size: max_params_size,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            partial_sums_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_partial_sums", lbl)),
                size: partial_sums_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_staging", lbl)),
                size: partial_sums_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            grad_partial_sums_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_grad_partial_sums", lbl)),
                size: partial_sums_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            grad_staging_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_grad_staging", lbl)),
                size: partial_sums_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        });
    }

    MultiChainGpuBuffers {
        x_buffer,
        num_chains,
        count,
        workgroup_count,
        chain_buffers,
        params_capacity: max_params_size,
    }
}

/// Run fused logp + grad for multiple chains in a single GPU submission.
///
/// Instead of N sequential submit+poll cycles, we encode ALL chains' compute
/// passes into ONE command encoder, submit once, and poll once.
#[allow(clippy::too_many_arguments)]
pub async fn run_multi_chain_fused<P: bytemuck::Pod>(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    logp_pipeline: &Arc<ComputePipeline>,
    logp_layout: &Arc<BindGroupLayout>,
    grad_pipeline: &Arc<ComputePipeline>,
    grad_layout: &Arc<BindGroupLayout>,
    buffers: &MultiChainGpuBuffers,
    chain_params: &[P],
) -> Result<Vec<FusedLogpGradResult>, String> {
    let num_chains = buffers.num_chains as usize;
    if chain_params.len() != num_chains {
        return Err(format!(
            "Expected {} chain params, got {}",
            num_chains,
            chain_params.len()
        ));
    }

    if buffers.count == 0 {
        return Ok(vec![
            FusedLogpGradResult {
                total_log_prob: 0.0,
                total_grad: 0.0,
            };
            num_chains
        ]);
    }

    // Write all chain params to their respective buffers
    for (i, params) in chain_params.iter().enumerate() {
        queue.write_buffer(
            &buffers.chain_buffers[i].params_buffer,
            0,
            bytemuck::bytes_of(params),
        );
    }

    // ONE command encoder with 2*N compute passes (logp + grad per chain)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("multi_chain_fused_enc"),
    });

    for (i, chain_buf) in buffers.chain_buffers.iter().enumerate() {
        let logp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("mc_logp_bg_{}", i)),
            layout: logp_layout.as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chain_buf.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: chain_buf.partial_sums_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("mc_logp_pass_{}", i)),
                timestamp_writes: None,
            });
            pass.set_pipeline(logp_pipeline.as_ref());
            pass.set_bind_group(0, &logp_bg, &[]);
            pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
        }

        let grad_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("mc_grad_bg_{}", i)),
            layout: grad_layout.as_ref(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chain_buf.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: chain_buf.grad_partial_sums_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("mc_grad_pass_{}", i)),
                timestamp_writes: None,
            });
            pass.set_pipeline(grad_pipeline.as_ref());
            pass.set_bind_group(0, &grad_bg, &[]);
            pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
        }
    }

    // Copy all partial sums to staging
    let partial_sums_size = (buffers.workgroup_count as u64) * 4;
    for chain_buf in &buffers.chain_buffers {
        encoder.copy_buffer_to_buffer(
            &chain_buf.partial_sums_buffer,
            0,
            &chain_buf.staging_buffer,
            0,
            partial_sums_size,
        );
        encoder.copy_buffer_to_buffer(
            &chain_buf.grad_partial_sums_buffer,
            0,
            &chain_buf.grad_staging_buffer,
            0,
            partial_sums_size,
        );
    }

    // ONE submit
    queue.submit(std::iter::once(encoder.finish()));

    // Map all staging buffers
    let mut logp_receivers = Vec::with_capacity(num_chains);
    let mut grad_receivers = Vec::with_capacity(num_chains);
    for chain_buf in &buffers.chain_buffers {
        let logp_slice = chain_buf.staging_buffer.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        logp_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        logp_receivers.push(rx);

        let grad_slice = chain_buf.grad_staging_buffer.slice(..);
        let (tx2, rx2) = futures_channel::oneshot::channel();
        grad_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx2.send(r);
        });
        grad_receivers.push(rx2);
    }

    // ONE poll drives ALL mappings
    poll_device(device);

    // Read results
    let mut results = Vec::with_capacity(num_chains);
    for (i, (logp_rx, grad_rx)) in logp_receivers
        .into_iter()
        .zip(grad_receivers.into_iter())
        .enumerate()
    {
        logp_rx
            .await
            .map_err(|_| format!("Chain {} logp channel cancelled", i))?
            .map_err(|e| format!("Chain {} logp buffer mapping failed: {:?}", i, e))?;
        grad_rx
            .await
            .map_err(|_| format!("Chain {} grad channel cancelled", i))?
            .map_err(|e| format!("Chain {} grad buffer mapping failed: {:?}", i, e))?;

        let logp_data = buffers.chain_buffers[i]
            .staging_buffer
            .slice(..)
            .get_mapped_range();
        let logp_partial: &[f32] = bytemuck::cast_slice(&logp_data);
        let total_log_prob: f64 = logp_partial.iter().map(|&x| x as f64).sum();
        drop(logp_data);
        buffers.chain_buffers[i].staging_buffer.unmap();

        let grad_data = buffers.chain_buffers[i]
            .grad_staging_buffer
            .slice(..)
            .get_mapped_range();
        let grad_partial: &[f32] = bytemuck::cast_slice(&grad_data);
        let total_grad: f64 = grad_partial.iter().map(|&x| x as f64).sum();
        drop(grad_data);
        buffers.chain_buffers[i].grad_staging_buffer.unmap();

        results.push(FusedLogpGradResult {
            total_log_prob: total_log_prob as f32,
            total_grad: total_grad as f32,
        });
    }

    Ok(results)
}

// =============================================================================
// LINEAR PREDICTOR KERNEL RUNNER
// =============================================================================

/// Run the Normal linear predictor fused kernel: y ~ Normal(X @ beta, sigma).
///
/// Uploads beta and params, dispatches the linpred shader, reads back (P+2) values
/// per workgroup and reduces on CPU.
pub async fn run_linpred_fused(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    layout: &Arc<BindGroupLayout>,
    buffers: &LinpredGpuBuffers,
    beta_values: &[f32],
    sigma: f32,
) -> Result<LinpredGpuResult, String> {
    let p = buffers.p;
    let count = buffers.count;

    if count == 0 {
        return Ok(LinpredGpuResult {
            total_log_prob: 0.0,
            grad_sigma: 0.0,
            grad_beta: vec![0.0; p as usize],
        });
    }

    // Upload beta values
    queue.write_buffer(&buffers.beta_buffer, 0, bytemuck::cast_slice(beta_values));

    // Upload params
    let params = NormalLinpredParams {
        sigma,
        count,
        p,
        _padding: 0,
    };
    queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("linpred_bind_group"),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.beta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffers.output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("linpred_enc"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("linpred_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
    }

    let output_size = (buffers.workgroup_count as u64) * (p as u64 + 2) * 4;
    encoder.copy_buffer_to_buffer(
        &buffers.output_buffer,
        0,
        &buffers.staging_buffer,
        0,
        output_size,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let staging_slice = buffers.staging_buffer.slice(..output_size);
    let (sender, receiver) = futures_channel::oneshot::channel();
    staging_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    poll_device(device);
    receiver
        .await
        .map_err(|_| "Channel cancelled")?
        .map_err(|e| format!("Linpred buffer mapping failed: {:?}", e))?;

    let data = staging_slice.get_mapped_range();
    let raw: &[f32] = bytemuck::cast_slice(&data);

    // Each workgroup outputs (P+2) values: [logp, grad_sigma, grad_beta[0..P]]
    let stride = (p + 2) as usize;
    let mut total_logp: f64 = 0.0;
    let mut total_grad_sigma: f64 = 0.0;
    let mut total_grad_beta: Vec<f64> = vec![0.0; p as usize];

    for chunk in raw.chunks(stride) {
        total_logp += chunk[0] as f64;
        total_grad_sigma += chunk[1] as f64;
        for j in 0..p as usize {
            total_grad_beta[j] += chunk[2 + j] as f64;
        }
    }

    drop(data);
    buffers.staging_buffer.unmap();

    Ok(LinpredGpuResult {
        total_log_prob: total_logp as f32,
        grad_sigma: total_grad_sigma as f32,
        grad_beta: total_grad_beta.iter().map(|&g| g as f32).collect(),
    })
}
