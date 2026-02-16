//! GPU-accelerated Random Number Generation for BayesianGPU
//!
//! This crate provides GPU-accelerated random number generation using the
//! XorShift128 algorithm implemented in WGSL shaders. It is designed for
//! use with the Burn framework's wgpu backend.
//!
//! # Features
//!
//! - Per-thread RNG state for parallel sampling
//! - Uniform distribution sampling in [0, 1)
//! - Normal distribution sampling via Box-Muller transform
//! - PCG hash for high-quality seed initialization
//!
//! # Example
//!
//! ```ignore
//! use bayesian_rng::GpuRng;
//! use burn::backend::Wgpu;
//!
//! let device = Default::default();
//! let mut rng = GpuRng::<Wgpu>::new(42, 1024, &device);
//!
//! // Generate uniform samples
//! let uniform_samples = rng.uniform(&[1000]);
//!
//! // Generate normal samples
//! let normal_samples = rng.normal(&[1000]);
//! ```

pub mod pcg;

// Re-export main types
pub use pcg::GpuRng;

/// WGSL shader source for XorShift128 RNG
///
/// This shader implements a high-quality PRNG suitable for GPU-based
/// Monte Carlo sampling. Each thread maintains independent state.
pub const WGSL_RNG_SHADER: &str = include_str!("wgsl/rng.wgsl");
