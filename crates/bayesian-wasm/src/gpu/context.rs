//! GPU context management for direct wgpu compute
//!
//! Handles wgpu device/queue initialization and compute pipeline creation.

use std::sync::Arc;
use wgpu::{util::DeviceExt, BindGroupLayout, BufferUsages, ComputePipeline, Device, Queue};

use super::kernels::{
    BernoulliReduceParams, BetaReduceParams, BinomialReduceParams, CategoricalReduceParams,
    CauchyReduceParams, ExponentialReduceParams, GammaReduceParams, GpuBatchResult,
    GpuGradReduceResult, GpuReduceResult, GpuResult, HalfNormalParams, HalfNormalReduceParams,
    InverseGammaReduceParams, LogNormalReduceParams, NegativeBinomialReduceParams,
    NormalBatchParams, NormalParams, PoissonReduceParams, StudentTReduceParams,
    UniformReduceParams,
};

/// GPU compute context holding wgpu resources
///
/// Resources are wrapped in Arc for cheap cloning when needed for async operations.
pub struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    // Basic kernels
    normal_pipeline: Arc<ComputePipeline>,
    normal_bind_group_layout: Arc<BindGroupLayout>,
    half_normal_pipeline: Arc<ComputePipeline>,
    half_normal_bind_group_layout: Arc<BindGroupLayout>,
    normal_batch_pipeline: Arc<ComputePipeline>,
    normal_batch_bind_group_layout: Arc<BindGroupLayout>,
    // Log-prob reduce kernels
    normal_reduce_pipeline: Arc<ComputePipeline>,
    normal_reduce_bind_group_layout: Arc<BindGroupLayout>,
    half_normal_reduce_pipeline: Arc<ComputePipeline>,
    half_normal_reduce_bind_group_layout: Arc<BindGroupLayout>,
    exponential_reduce_pipeline: Arc<ComputePipeline>,
    exponential_reduce_bind_group_layout: Arc<BindGroupLayout>,
    gamma_reduce_pipeline: Arc<ComputePipeline>,
    gamma_reduce_bind_group_layout: Arc<BindGroupLayout>,
    beta_reduce_pipeline: Arc<ComputePipeline>,
    beta_reduce_bind_group_layout: Arc<BindGroupLayout>,
    inverse_gamma_reduce_pipeline: Arc<ComputePipeline>,
    inverse_gamma_reduce_bind_group_layout: Arc<BindGroupLayout>,
    uniform_reduce_pipeline: Arc<ComputePipeline>,
    uniform_reduce_bind_group_layout: Arc<BindGroupLayout>,
    cauchy_reduce_pipeline: Arc<ComputePipeline>,
    cauchy_reduce_bind_group_layout: Arc<BindGroupLayout>,
    student_t_reduce_pipeline: Arc<ComputePipeline>,
    student_t_reduce_bind_group_layout: Arc<BindGroupLayout>,
    lognormal_reduce_pipeline: Arc<ComputePipeline>,
    lognormal_reduce_bind_group_layout: Arc<BindGroupLayout>,
    bernoulli_reduce_pipeline: Arc<ComputePipeline>,
    bernoulli_reduce_bind_group_layout: Arc<BindGroupLayout>,
    binomial_reduce_pipeline: Arc<ComputePipeline>,
    binomial_reduce_bind_group_layout: Arc<BindGroupLayout>,
    poisson_reduce_pipeline: Arc<ComputePipeline>,
    poisson_reduce_bind_group_layout: Arc<BindGroupLayout>,
    negative_binomial_reduce_pipeline: Arc<ComputePipeline>,
    negative_binomial_reduce_bind_group_layout: Arc<BindGroupLayout>,
    categorical_reduce_pipeline: Arc<ComputePipeline>,
    categorical_reduce_bind_group_layout: Arc<BindGroupLayout>,
    // Gradient reduce kernels
    normal_grad_reduce_pipeline: Arc<ComputePipeline>,
    normal_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
    half_normal_grad_reduce_pipeline: Arc<ComputePipeline>,
    half_normal_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
    exponential_grad_reduce_pipeline: Arc<ComputePipeline>,
    exponential_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
    beta_grad_reduce_pipeline: Arc<ComputePipeline>,
    beta_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
    gamma_grad_reduce_pipeline: Arc<ComputePipeline>,
    gamma_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
    inverse_gamma_grad_reduce_pipeline: Arc<ComputePipeline>,
    inverse_gamma_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
    student_t_grad_reduce_pipeline: Arc<ComputePipeline>,
    student_t_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
    cauchy_grad_reduce_pipeline: Arc<ComputePipeline>,
    cauchy_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
    lognormal_grad_reduce_pipeline: Arc<ComputePipeline>,
    lognormal_grad_reduce_bind_group_layout: Arc<BindGroupLayout>,
}

impl GpuContext {
    /// Create a new GPU context with async initialization
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bayesiangpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("Failed to get GPU device: {:?}", e))?;

        // Helper to create standard 3-binding reduce layout (params, x_values, partial_sums)
        let create_reduce_layout = |device: &Device, label: &str| -> BindGroupLayout {
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
        };

        // Helper to create pipeline from shader and layout
        let create_reduce_pipeline = |device: &Device,
                                      shader_source: &str,
                                      label: &str,
                                      layout: &BindGroupLayout|
         -> ComputePipeline {
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
        };

        // =====================================================================
        // BASIC KERNELS
        // =====================================================================

        // Normal distribution pipeline (2-binding: params, output)
        let normal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normal_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/normal.wgsl").into()),
        });
        let normal_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("normal_bind_group_layout"),
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
            });
        let normal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("normal_pipeline_layout"),
                bind_group_layouts: &[&normal_bind_group_layout],
                push_constant_ranges: &[],
            });
        let normal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("normal_pipeline"),
            layout: Some(&normal_pipeline_layout),
            module: &normal_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // HalfNormal distribution pipeline (2-binding)
        let half_normal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("half_normal_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/half_normal.wgsl").into()),
        });
        let half_normal_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("half_normal_bind_group_layout"),
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
            });
        let half_normal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("half_normal_pipeline_layout"),
                bind_group_layouts: &[&half_normal_bind_group_layout],
                push_constant_ranges: &[],
            });
        let half_normal_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("half_normal_pipeline"),
                layout: Some(&half_normal_pipeline_layout),
                module: &half_normal_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Normal BATCH pipeline (4-binding: params, x_values, log_probs, grads)
        let normal_batch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normal_batch_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/normal_batch.wgsl").into()),
        });
        let normal_batch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("normal_batch_bind_group_layout"),
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
            });
        let normal_batch_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("normal_batch_pipeline_layout"),
                bind_group_layouts: &[&normal_batch_bind_group_layout],
                push_constant_ranges: &[],
            });
        let normal_batch_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("normal_batch_pipeline"),
                layout: Some(&normal_batch_pipeline_layout),
                module: &normal_batch_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // =====================================================================
        // LOG-PROB REDUCE KERNELS (3-binding)
        // =====================================================================

        let normal_reduce_bind_group_layout =
            create_reduce_layout(&device, "normal_reduce_bind_group_layout");
        let normal_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/normal_reduce.wgsl"),
            "normal_reduce_pipeline",
            &normal_reduce_bind_group_layout,
        );

        let half_normal_reduce_bind_group_layout =
            create_reduce_layout(&device, "half_normal_reduce_bind_group_layout");
        let half_normal_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/half_normal_reduce.wgsl"),
            "half_normal_reduce_pipeline",
            &half_normal_reduce_bind_group_layout,
        );

        let exponential_reduce_bind_group_layout =
            create_reduce_layout(&device, "exponential_reduce_bind_group_layout");
        let exponential_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/exponential_reduce.wgsl"),
            "exponential_reduce_pipeline",
            &exponential_reduce_bind_group_layout,
        );

        let gamma_reduce_bind_group_layout =
            create_reduce_layout(&device, "gamma_reduce_bind_group_layout");
        let gamma_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/gamma_reduce.wgsl"),
            "gamma_reduce_pipeline",
            &gamma_reduce_bind_group_layout,
        );

        let beta_reduce_bind_group_layout =
            create_reduce_layout(&device, "beta_reduce_bind_group_layout");
        let beta_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/beta_reduce.wgsl"),
            "beta_reduce_pipeline",
            &beta_reduce_bind_group_layout,
        );

        let inverse_gamma_reduce_bind_group_layout =
            create_reduce_layout(&device, "inverse_gamma_reduce_bind_group_layout");
        let inverse_gamma_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/inverse_gamma_reduce.wgsl"),
            "inverse_gamma_reduce_pipeline",
            &inverse_gamma_reduce_bind_group_layout,
        );

        let uniform_reduce_bind_group_layout =
            create_reduce_layout(&device, "uniform_reduce_bind_group_layout");
        let uniform_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/uniform_reduce.wgsl"),
            "uniform_reduce_pipeline",
            &uniform_reduce_bind_group_layout,
        );

        let cauchy_reduce_bind_group_layout =
            create_reduce_layout(&device, "cauchy_reduce_bind_group_layout");
        let cauchy_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/cauchy_reduce.wgsl"),
            "cauchy_reduce_pipeline",
            &cauchy_reduce_bind_group_layout,
        );

        let student_t_reduce_bind_group_layout =
            create_reduce_layout(&device, "student_t_reduce_bind_group_layout");
        let student_t_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/student_t_reduce.wgsl"),
            "student_t_reduce_pipeline",
            &student_t_reduce_bind_group_layout,
        );

        let lognormal_reduce_bind_group_layout =
            create_reduce_layout(&device, "lognormal_reduce_bind_group_layout");
        let lognormal_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/lognormal_reduce.wgsl"),
            "lognormal_reduce_pipeline",
            &lognormal_reduce_bind_group_layout,
        );

        let bernoulli_reduce_bind_group_layout =
            create_reduce_layout(&device, "bernoulli_reduce_bind_group_layout");
        let bernoulli_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/bernoulli_reduce.wgsl"),
            "bernoulli_reduce_pipeline",
            &bernoulli_reduce_bind_group_layout,
        );

        let binomial_reduce_bind_group_layout =
            create_reduce_layout(&device, "binomial_reduce_bind_group_layout");
        let binomial_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/binomial_reduce.wgsl"),
            "binomial_reduce_pipeline",
            &binomial_reduce_bind_group_layout,
        );

        let poisson_reduce_bind_group_layout =
            create_reduce_layout(&device, "poisson_reduce_bind_group_layout");
        let poisson_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/poisson_reduce.wgsl"),
            "poisson_reduce_pipeline",
            &poisson_reduce_bind_group_layout,
        );

        let negative_binomial_reduce_bind_group_layout =
            create_reduce_layout(&device, "negative_binomial_reduce_bind_group_layout");
        let negative_binomial_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/negative_binomial_reduce.wgsl"),
            "negative_binomial_reduce_pipeline",
            &negative_binomial_reduce_bind_group_layout,
        );

        // Categorical has 4 bindings (params, x_values, probs, partial_sums)
        let categorical_reduce_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("categorical_reduce_bind_group_layout"),
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
            });
        let categorical_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/categorical_reduce.wgsl"),
            "categorical_reduce_pipeline",
            &categorical_reduce_bind_group_layout,
        );

        // =====================================================================
        // GRADIENT REDUCE KERNELS (3-binding)
        // =====================================================================

        let normal_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "normal_grad_reduce_bind_group_layout");
        let normal_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/normal_grad_reduce.wgsl"),
            "normal_grad_reduce_pipeline",
            &normal_grad_reduce_bind_group_layout,
        );

        let half_normal_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "half_normal_grad_reduce_bind_group_layout");
        let half_normal_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/half_normal_grad_reduce.wgsl"),
            "half_normal_grad_reduce_pipeline",
            &half_normal_grad_reduce_bind_group_layout,
        );

        let exponential_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "exponential_grad_reduce_bind_group_layout");
        let exponential_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/exponential_grad_reduce.wgsl"),
            "exponential_grad_reduce_pipeline",
            &exponential_grad_reduce_bind_group_layout,
        );

        let beta_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "beta_grad_reduce_bind_group_layout");
        let beta_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/beta_grad_reduce.wgsl"),
            "beta_grad_reduce_pipeline",
            &beta_grad_reduce_bind_group_layout,
        );

        let gamma_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "gamma_grad_reduce_bind_group_layout");
        let gamma_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/gamma_grad_reduce.wgsl"),
            "gamma_grad_reduce_pipeline",
            &gamma_grad_reduce_bind_group_layout,
        );

        let inverse_gamma_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "inverse_gamma_grad_reduce_bind_group_layout");
        let inverse_gamma_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/inverse_gamma_grad_reduce.wgsl"),
            "inverse_gamma_grad_reduce_pipeline",
            &inverse_gamma_grad_reduce_bind_group_layout,
        );

        let student_t_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "student_t_grad_reduce_bind_group_layout");
        let student_t_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/student_t_grad_reduce.wgsl"),
            "student_t_grad_reduce_pipeline",
            &student_t_grad_reduce_bind_group_layout,
        );

        let cauchy_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "cauchy_grad_reduce_bind_group_layout");
        let cauchy_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/cauchy_grad_reduce.wgsl"),
            "cauchy_grad_reduce_pipeline",
            &cauchy_grad_reduce_bind_group_layout,
        );

        let lognormal_grad_reduce_bind_group_layout =
            create_reduce_layout(&device, "lognormal_grad_reduce_bind_group_layout");
        let lognormal_grad_reduce_pipeline = create_reduce_pipeline(
            &device,
            include_str!("shaders/lognormal_grad_reduce.wgsl"),
            "lognormal_grad_reduce_pipeline",
            &lognormal_grad_reduce_bind_group_layout,
        );

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            normal_pipeline: Arc::new(normal_pipeline),
            normal_bind_group_layout: Arc::new(normal_bind_group_layout),
            half_normal_pipeline: Arc::new(half_normal_pipeline),
            half_normal_bind_group_layout: Arc::new(half_normal_bind_group_layout),
            normal_batch_pipeline: Arc::new(normal_batch_pipeline),
            normal_batch_bind_group_layout: Arc::new(normal_batch_bind_group_layout),
            normal_reduce_pipeline: Arc::new(normal_reduce_pipeline),
            normal_reduce_bind_group_layout: Arc::new(normal_reduce_bind_group_layout),
            half_normal_reduce_pipeline: Arc::new(half_normal_reduce_pipeline),
            half_normal_reduce_bind_group_layout: Arc::new(half_normal_reduce_bind_group_layout),
            exponential_reduce_pipeline: Arc::new(exponential_reduce_pipeline),
            exponential_reduce_bind_group_layout: Arc::new(exponential_reduce_bind_group_layout),
            gamma_reduce_pipeline: Arc::new(gamma_reduce_pipeline),
            gamma_reduce_bind_group_layout: Arc::new(gamma_reduce_bind_group_layout),
            beta_reduce_pipeline: Arc::new(beta_reduce_pipeline),
            beta_reduce_bind_group_layout: Arc::new(beta_reduce_bind_group_layout),
            inverse_gamma_reduce_pipeline: Arc::new(inverse_gamma_reduce_pipeline),
            inverse_gamma_reduce_bind_group_layout: Arc::new(
                inverse_gamma_reduce_bind_group_layout,
            ),
            uniform_reduce_pipeline: Arc::new(uniform_reduce_pipeline),
            uniform_reduce_bind_group_layout: Arc::new(uniform_reduce_bind_group_layout),
            cauchy_reduce_pipeline: Arc::new(cauchy_reduce_pipeline),
            cauchy_reduce_bind_group_layout: Arc::new(cauchy_reduce_bind_group_layout),
            student_t_reduce_pipeline: Arc::new(student_t_reduce_pipeline),
            student_t_reduce_bind_group_layout: Arc::new(student_t_reduce_bind_group_layout),
            lognormal_reduce_pipeline: Arc::new(lognormal_reduce_pipeline),
            lognormal_reduce_bind_group_layout: Arc::new(lognormal_reduce_bind_group_layout),
            bernoulli_reduce_pipeline: Arc::new(bernoulli_reduce_pipeline),
            bernoulli_reduce_bind_group_layout: Arc::new(bernoulli_reduce_bind_group_layout),
            binomial_reduce_pipeline: Arc::new(binomial_reduce_pipeline),
            binomial_reduce_bind_group_layout: Arc::new(binomial_reduce_bind_group_layout),
            poisson_reduce_pipeline: Arc::new(poisson_reduce_pipeline),
            poisson_reduce_bind_group_layout: Arc::new(poisson_reduce_bind_group_layout),
            negative_binomial_reduce_pipeline: Arc::new(negative_binomial_reduce_pipeline),
            negative_binomial_reduce_bind_group_layout: Arc::new(
                negative_binomial_reduce_bind_group_layout,
            ),
            categorical_reduce_pipeline: Arc::new(categorical_reduce_pipeline),
            categorical_reduce_bind_group_layout: Arc::new(categorical_reduce_bind_group_layout),
            normal_grad_reduce_pipeline: Arc::new(normal_grad_reduce_pipeline),
            normal_grad_reduce_bind_group_layout: Arc::new(normal_grad_reduce_bind_group_layout),
            half_normal_grad_reduce_pipeline: Arc::new(half_normal_grad_reduce_pipeline),
            half_normal_grad_reduce_bind_group_layout: Arc::new(
                half_normal_grad_reduce_bind_group_layout,
            ),
            exponential_grad_reduce_pipeline: Arc::new(exponential_grad_reduce_pipeline),
            exponential_grad_reduce_bind_group_layout: Arc::new(
                exponential_grad_reduce_bind_group_layout,
            ),
            beta_grad_reduce_pipeline: Arc::new(beta_grad_reduce_pipeline),
            beta_grad_reduce_bind_group_layout: Arc::new(beta_grad_reduce_bind_group_layout),
            gamma_grad_reduce_pipeline: Arc::new(gamma_grad_reduce_pipeline),
            gamma_grad_reduce_bind_group_layout: Arc::new(gamma_grad_reduce_bind_group_layout),
            inverse_gamma_grad_reduce_pipeline: Arc::new(inverse_gamma_grad_reduce_pipeline),
            inverse_gamma_grad_reduce_bind_group_layout: Arc::new(
                inverse_gamma_grad_reduce_bind_group_layout,
            ),
            student_t_grad_reduce_pipeline: Arc::new(student_t_grad_reduce_pipeline),
            student_t_grad_reduce_bind_group_layout: Arc::new(
                student_t_grad_reduce_bind_group_layout,
            ),
            cauchy_grad_reduce_pipeline: Arc::new(cauchy_grad_reduce_pipeline),
            cauchy_grad_reduce_bind_group_layout: Arc::new(cauchy_grad_reduce_bind_group_layout),
            lognormal_grad_reduce_pipeline: Arc::new(lognormal_grad_reduce_pipeline),
            lognormal_grad_reduce_bind_group_layout: Arc::new(
                lognormal_grad_reduce_bind_group_layout,
            ),
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
    pub fn normal_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_pipeline)
    }
    pub fn normal_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_bind_group_layout)
    }
    pub fn half_normal_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.half_normal_pipeline)
    }
    pub fn half_normal_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.half_normal_bind_group_layout)
    }
    pub fn normal_batch_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_batch_pipeline)
    }
    pub fn normal_batch_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_batch_bind_group_layout)
    }

    // =========================================================================
    // CLONE METHODS - Log-prob reduce
    // =========================================================================

    pub fn normal_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_reduce_pipeline)
    }
    pub fn normal_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_reduce_bind_group_layout)
    }
    pub fn half_normal_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.half_normal_reduce_pipeline)
    }
    pub fn half_normal_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.half_normal_reduce_bind_group_layout)
    }
    pub fn exponential_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.exponential_reduce_pipeline)
    }
    pub fn exponential_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.exponential_reduce_bind_group_layout)
    }
    pub fn gamma_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.gamma_reduce_pipeline)
    }
    pub fn gamma_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.gamma_reduce_bind_group_layout)
    }
    pub fn beta_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.beta_reduce_pipeline)
    }
    pub fn beta_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.beta_reduce_bind_group_layout)
    }
    pub fn inverse_gamma_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.inverse_gamma_reduce_pipeline)
    }
    pub fn inverse_gamma_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.inverse_gamma_reduce_bind_group_layout)
    }
    pub fn uniform_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.uniform_reduce_pipeline)
    }
    pub fn uniform_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.uniform_reduce_bind_group_layout)
    }
    pub fn cauchy_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.cauchy_reduce_pipeline)
    }
    pub fn cauchy_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.cauchy_reduce_bind_group_layout)
    }
    pub fn student_t_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.student_t_reduce_pipeline)
    }
    pub fn student_t_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.student_t_reduce_bind_group_layout)
    }
    pub fn lognormal_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.lognormal_reduce_pipeline)
    }
    pub fn lognormal_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.lognormal_reduce_bind_group_layout)
    }
    pub fn bernoulli_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.bernoulli_reduce_pipeline)
    }
    pub fn bernoulli_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.bernoulli_reduce_bind_group_layout)
    }
    pub fn binomial_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.binomial_reduce_pipeline)
    }
    pub fn binomial_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.binomial_reduce_bind_group_layout)
    }
    pub fn poisson_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.poisson_reduce_pipeline)
    }
    pub fn poisson_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.poisson_reduce_bind_group_layout)
    }
    pub fn negative_binomial_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.negative_binomial_reduce_pipeline)
    }
    pub fn negative_binomial_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.negative_binomial_reduce_bind_group_layout)
    }
    pub fn categorical_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.categorical_reduce_pipeline)
    }
    pub fn categorical_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.categorical_reduce_bind_group_layout)
    }

    // =========================================================================
    // CLONE METHODS - Gradient reduce
    // =========================================================================

    pub fn normal_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.normal_grad_reduce_pipeline)
    }
    pub fn normal_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.normal_grad_reduce_bind_group_layout)
    }
    pub fn half_normal_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.half_normal_grad_reduce_pipeline)
    }
    pub fn half_normal_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.half_normal_grad_reduce_bind_group_layout)
    }
    pub fn exponential_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.exponential_grad_reduce_pipeline)
    }
    pub fn exponential_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.exponential_grad_reduce_bind_group_layout)
    }
    pub fn beta_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.beta_grad_reduce_pipeline)
    }
    pub fn beta_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.beta_grad_reduce_bind_group_layout)
    }
    pub fn gamma_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.gamma_grad_reduce_pipeline)
    }
    pub fn gamma_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.gamma_grad_reduce_bind_group_layout)
    }
    pub fn inverse_gamma_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.inverse_gamma_grad_reduce_pipeline)
    }
    pub fn inverse_gamma_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.inverse_gamma_grad_reduce_bind_group_layout)
    }
    pub fn student_t_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.student_t_grad_reduce_pipeline)
    }
    pub fn student_t_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.student_t_grad_reduce_bind_group_layout)
    }
    pub fn cauchy_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.cauchy_grad_reduce_pipeline)
    }
    pub fn cauchy_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.cauchy_grad_reduce_bind_group_layout)
    }
    pub fn lognormal_grad_reduce_pipeline_clone(&self) -> Arc<ComputePipeline> {
        Arc::clone(&self.lognormal_grad_reduce_pipeline)
    }
    pub fn lognormal_grad_reduce_bind_group_layout_clone(&self) -> Arc<BindGroupLayout> {
        Arc::clone(&self.lognormal_grad_reduce_bind_group_layout)
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

    let workgroup_count = (count + 255) / 256;
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

    let workgroup_count = (count + 255) / 256;
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

    let workgroup_count = (count + 255) / 256;
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

    let workgroup_count = (count + 255) / 256;
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
