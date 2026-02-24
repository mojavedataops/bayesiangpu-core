//! GPU context management for direct wgpu compute
//!
//! Handles wgpu device/queue initialization and compute pipeline creation.
//! Pipelines are lazily compiled on first use via OnceLock, avoiding the
//! upfront cost of compiling 30+ WGSL shaders during initialization.

use std::sync::{Arc, OnceLock};
use wgpu::{util::DeviceExt, BindGroupLayout, BufferUsages, ComputePipeline, Device, Queue};

use super::kernels::{
    BernoulliReduceParams, BetaReduceParams, BinomialReduceParams, CategoricalReduceParams,
    CauchyReduceParams, ExponentialReduceParams, FusedLogpGradResult, GammaReduceParams,
    GpuBatchResult, GpuGradReduceResult, GpuReduceResult, GpuResult, HalfNormalParams,
    HalfNormalReduceParams, InverseGammaReduceParams, LogNormalReduceParams,
    NegativeBinomialReduceParams, NormalBatchParams, NormalParams, PoissonReduceParams,
    StudentTReduceParams, UniformReduceParams,
};

/// A lazily-initialized pipeline + bind group layout pair.
type LazyPipeline = OnceLock<(Arc<ComputePipeline>, Arc<BindGroupLayout>)>;

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
    let workgroup_count = count.div_ceil(256);
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
    }
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("persistent_reduce_bg"),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.partial_sums_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("persistent_reduce_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("persistent_reduce_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
    }
    let partial_sums_size = (buffers.workgroup_count as u64) * 4;
    encoder.copy_buffer_to_buffer(
        &buffers.partial_sums_buffer,
        0,
        &buffers.staging_buffer,
        0,
        partial_sums_size,
    );
    queue.submit(std::iter::once(encoder.finish()));

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
    let total_log_prob: f32 = partial_sums.iter().sum();
    drop(data);
    buffers.staging_buffer.unmap();
    Ok(GpuReduceResult { total_log_prob })
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("persistent_grad_reduce_bg"),
        layout: layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.partial_sums_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("persistent_grad_reduce_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("persistent_grad_reduce_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
    }
    let partial_sums_size = (buffers.workgroup_count as u64) * 4;
    encoder.copy_buffer_to_buffer(
        &buffers.partial_sums_buffer,
        0,
        &buffers.staging_buffer,
        0,
        partial_sums_size,
    );
    queue.submit(std::iter::once(encoder.finish()));

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
    let total_grad: f32 = partial_sums.iter().sum();
    drop(data);
    buffers.staging_buffer.unmap();
    Ok(GpuGradReduceResult { total_grad })
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

    // Create logp bind group: params, x_values, partial_sums
    let logp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fused_logp_bg"),
        layout: logp_layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.partial_sums_buffer.as_entire_binding(),
            },
        ],
    });

    // Create grad bind group: params, x_values, grad_partial_sums
    let grad_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fused_grad_bg"),
        layout: grad_layout.as_ref(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.grad_partial_sums_buffer.as_entire_binding(),
            },
        ],
    });

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
        pass.set_bind_group(0, &logp_bind_group, &[]);
        pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
    }

    // Pass 2: grad reduce
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_grad_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(grad_pipeline.as_ref());
        pass.set_bind_group(0, &grad_bind_group, &[]);
        pass.dispatch_workgroups(buffers.workgroup_count, 1, 1);
    }

    // TWO copy operations
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

    // ONE submit
    queue.submit(std::iter::once(encoder.finish()));

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

    // Read both staging buffers and sum
    let logp_data = logp_slice.get_mapped_range();
    let logp_partial_sums: &[f32] = bytemuck::cast_slice(&logp_data);
    let total_log_prob: f32 = logp_partial_sums.iter().sum();
    drop(logp_data);
    buffers.staging_buffer.unmap();

    let grad_data = grad_slice.get_mapped_range();
    let grad_partial_sums: &[f32] = bytemuck::cast_slice(&grad_data);
    let total_grad: f32 = grad_partial_sums.iter().sum();
    drop(grad_data);
    buffers.grad_staging_buffer.unmap();

    Ok(FusedLogpGradResult {
        total_log_prob,
        total_grad,
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
            normal_grad_reduce: OnceLock::new(),
            half_normal_grad_reduce: OnceLock::new(),
            exponential_grad_reduce: OnceLock::new(),
            beta_grad_reduce: OnceLock::new(),
            gamma_grad_reduce: OnceLock::new(),
            inverse_gamma_grad_reduce: OnceLock::new(),
            student_t_grad_reduce: OnceLock::new(),
            cauchy_grad_reduce: OnceLock::new(),
            lognormal_grad_reduce: OnceLock::new(),
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

    let workgroup_count = count.div_ceil(256);
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
    let total_log_prob: f32 = partial_sums.iter().sum();
    drop(data);
    staging_buffer.unmap();
    Ok(GpuReduceResult { total_log_prob })
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
            _padding: 0,
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
            _padding: 0,
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
            _padding: 0,
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
            _padding: 0,
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

    let workgroup_count = count.div_ceil(256);
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
    let total_log_prob: f32 = partial_sums.iter().sum();
    drop(data);
    staging_buffer.unmap();
    Ok(GpuReduceResult { total_log_prob })
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

    let workgroup_count = count.div_ceil(256);
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
    let total_grad: f32 = partial_sums.iter().sum();
    drop(data);
    staging_buffer.unmap();
    Ok(GpuGradReduceResult { total_grad })
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
            _padding: 0,
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
            _padding: 0,
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
            _padding: 0,
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
            _padding: 0,
        },
        x_values,
        "lognormal_grad_reduce",
    )
    .await
}
