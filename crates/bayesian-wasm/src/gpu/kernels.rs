//! Kernel data structures for GPU compute

use serde::{Deserialize, Serialize};

/// Input parameters for Normal distribution kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NormalParams {
    pub x: f32,
    pub mu: f32,
    pub sigma: f32,
    pub _padding: f32, // Align to 16 bytes for GPU
}

/// Input parameters for HalfNormal distribution kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HalfNormalParams {
    pub x: f32,
    pub sigma: f32,
    pub _padding1: f32,
    pub _padding2: f32,
}

/// Input parameters for batched Normal distribution kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NormalBatchParams {
    pub mu: f32,
    pub sigma: f32,
    pub count: u32,
    pub _padding: u32,
}

/// Result from GPU kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuResult {
    pub log_prob: f32,
    pub grad: f32,
}

/// Result from batched GPU kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchResult {
    pub log_probs: Vec<f32>,
    pub grads: Vec<f32>,
}

/// Input parameters for HalfNormal REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HalfNormalReduceParams {
    pub sigma: f32,
    pub count: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// Input parameters for Exponential REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExponentialReduceParams {
    pub lambda: f32,
    pub count: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// Input parameters for Gamma REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GammaReduceParams {
    pub alpha: f32,
    pub beta: f32,
    pub count: u32,
    /// Pre-computed: alpha * ln(beta) - lgamma(alpha)
    pub log_norm: f32,
}

/// Input parameters for Beta REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BetaReduceParams {
    pub alpha: f32,
    pub beta: f32,
    pub count: u32,
    /// Pre-computed: lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
    pub log_norm: f32,
}

/// Input parameters for InverseGamma REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InverseGammaReduceParams {
    pub alpha: f32,
    pub beta: f32,
    pub count: u32,
    /// Pre-computed: alpha * ln(beta) - lgamma(alpha)
    pub log_norm: f32,
}

/// Input parameters for Uniform REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformReduceParams {
    pub low: f32,
    pub high: f32,
    pub count: u32,
    pub _padding: u32,
}

/// Input parameters for Cauchy REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CauchyReduceParams {
    pub loc: f32,
    pub scale: f32,
    pub count: u32,
    pub _padding: u32,
}

/// Input parameters for Bernoulli REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BernoulliReduceParams {
    pub p: f32,
    pub count: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// Input parameters for Binomial REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinomialReduceParams {
    pub n: f32, // trials (as f32 for GPU)
    pub p: f32,
    pub count: u32,
    pub _padding: u32,
}

/// Input parameters for Poisson REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PoissonReduceParams {
    pub lambda: f32,
    pub count: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// Input parameters for NegativeBinomial REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NegativeBinomialReduceParams {
    pub r: f32, // failures (as f32)
    pub p: f32,
    pub count: u32,
    pub _padding: u32,
}

/// Input parameters for StudentT REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StudentTReduceParams {
    pub loc: f32,
    pub scale: f32,
    pub nu: f32,
    pub count: u32,
    /// Pre-computed: lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*ln(nu*PI) - ln(scale)
    pub log_norm: f32,
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
}

/// Input parameters for LogNormal REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LogNormalReduceParams {
    pub mu: f32,
    pub sigma: f32,
    pub count: u32,
    /// Pre-computed: -0.5 * ln(2*PI) - ln(sigma)
    pub log_norm: f32,
}

/// Input parameters for Categorical REDUCE kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CategoricalReduceParams {
    pub num_categories: u32,
    pub count: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// Result from reduced GPU kernel (returns scalar sum)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuReduceResult {
    pub total_log_prob: f32,
}

/// Result from gradient reduced GPU kernel (returns scalar sum of gradients)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuGradReduceResult {
    pub total_grad: f32,
}

/// Result from fused logp + grad GPU kernel (returns both sums in one dispatch)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedLogpGradResult {
    pub total_log_prob: f32,
    pub total_grad: f32,
}
