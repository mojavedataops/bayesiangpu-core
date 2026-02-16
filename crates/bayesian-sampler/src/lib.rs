//! HMC and NUTS Samplers for BayesianGPU
//!
//! This crate provides Hamiltonian Monte Carlo (HMC) and No-U-Turn Sampler (NUTS)
//! implementations for GPU-accelerated Bayesian inference using the Burn framework.
//!
//! # Overview
//!
//! The sampler crate implements gradient-based MCMC methods that leverage automatic
//! differentiation for efficient posterior sampling. Key components include:
//!
//! - **Leapfrog integrator**: Symplectic integrator for Hamiltonian dynamics
//! - **HMC sampler**: Basic Hamiltonian Monte Carlo with fixed trajectory length
//! - **NUTS sampler**: No-U-Turn Sampler with adaptive trajectory length
//! - **Adaptation**: Dual averaging for step size, mass matrix estimation
//!
//! # Example
//!
//! ```ignore
//! use bayesian_sampler::{NutsSampler, NutsConfig};
//! use bayesian_core::BayesianModel;
//! use bayesian_rng::GpuRng;
//! use burn::prelude::*;
//!
//! // Define your model implementing BayesianModel trait
//! let model = MyModel::new(data);
//!
//! // Configure the NUTS sampler
//! let config = NutsConfig {
//!     num_samples: 1000,
//!     num_warmup: 1000,
//!     max_tree_depth: 10,
//!     target_accept: 0.8,
//!     init_step_size: 1.0,
//! };
//!
//! // Create sampler
//! let mut rng = GpuRng::new(42, model.dim(), &device);
//! let mut sampler = NutsSampler::new(model, config, rng);
//!
//! // Run sampling
//! let init = Tensor::zeros([model.dim()], &device);
//! let result = sampler.sample(init);
//!
//! println!("Divergences: {}", result.divergences);
//! println!("Final step size: {:.4}", result.final_step_size);
//! ```

pub mod adaptation;
pub mod advi;
pub mod chain;
pub mod hmc;
pub mod leapfrog;
pub mod model;
pub mod nuts;

// Re-export main types
pub use adaptation::{AdaptationPhase, AdaptationSchedule, DualAveraging, MassMatrixAdaptation};
pub use advi::{
    AdviConfig, ConvergenceDiagnostics, FullRankAdvi, FullRankResult, MeanFieldAdvi,
    MeanFieldResult,
};
pub use chain::{MultiChainConfig, MultiChainResult, MultiChainSampler};
pub use hmc::{HmcConfig, HmcResult, HmcSampler};
pub use leapfrog::{
    hamiltonian, kinetic_energy, leapfrog, leapfrog_step, leapfrog_step_with_mass,
    leapfrog_with_mass, LeapfrogResult,
};
pub use model::{
    batched_logp_and_grad, batched_logp_and_grad_tensor, log_prob_transformed, logp_and_grad,
    BayesianModel,
};
pub use nuts::{log_add_exp, NutsConfig, NutsResult, NutsSampler, TreeNode, TreeResult};
