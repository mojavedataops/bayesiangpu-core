//! Leapfrog integrator for Hamiltonian dynamics
//!
//! The leapfrog integrator is a symplectic integrator that preserves the
//! Hamiltonian structure of the dynamics, making it ideal for HMC/NUTS sampling.
//!
//! # Algorithm
//!
//! The leapfrog integration scheme for Hamiltonian dynamics with potential
//! energy U(q) and kinetic energy K(p) = 0.5 * p^T * M^-1 * p consists of:
//!
//! 1. Half step for momentum: p <- p + (eps/2) * grad_q(log_prob)
//! 2. Full step for position: q <- q + eps * p
//! 3. Half step for momentum: p <- p + (eps/2) * grad_q(log_prob)
//!
//! This scheme is:
//! - **Symplectic**: Preserves phase space volume
//! - **Time-reversible**: Running backward recovers the initial state
//! - **Second-order accurate**: Error is O(eps^3) per step

use crate::model::{logp_and_grad, BayesianModel};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// Result of a leapfrog integration
#[derive(Debug, Clone)]
pub struct LeapfrogResult<B: Backend> {
    /// Final position after integration
    pub position: Tensor<B, 1>,
    /// Final momentum after integration
    pub momentum: Tensor<B, 1>,
    /// Log probability at final position
    pub log_prob: f64,
    /// Gradient at final position
    pub gradient: Vec<f64>,
}

/// Perform a single leapfrog step
///
/// This function executes one complete leapfrog step consisting of
/// half-step momentum update, full-step position update, and
/// half-step momentum update.
///
/// # Arguments
///
/// * `model` - The Bayesian model to sample from
/// * `position` - Current position in parameter space
/// * `momentum` - Current momentum
/// * `step_size` - Integration step size (epsilon)
///
/// # Returns
///
/// A [`LeapfrogResult`] containing the new position, momentum, and log probability.
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::leapfrog::leapfrog_step;
///
/// let result = leapfrog_step(&model, position, momentum, 0.01);
/// println!("New position: {:?}", result.position);
/// println!("Log prob: {}", result.log_prob);
/// ```
pub fn leapfrog_step<B, M>(
    model: &M,
    position: Tensor<B, 1>,
    momentum: Tensor<B, 1>,
    step_size: f64,
) -> LeapfrogResult<B>
where
    B: AutodiffBackend,
    M: BayesianModel<B>,
{
    leapfrog_step_with_mass(model, position, momentum, step_size, None)
}

/// Perform a single leapfrog step with mass matrix preconditioning
///
/// With diagonal mass matrix M, the dynamics are:
/// - p <- p + (eps/2) * grad_log_prob
/// - q <- q + eps * M^{-1} * p
/// - p <- p + (eps/2) * grad_log_prob
///
/// # Arguments
///
/// * `model` - The Bayesian model to sample from
/// * `position` - Current position in parameter space
/// * `momentum` - Current momentum
/// * `step_size` - Integration step size (epsilon)
/// * `inv_mass_matrix` - Optional inverse mass matrix (diagonal elements)
///
/// # Returns
///
/// A [`LeapfrogResult`] containing the new position, momentum, and log probability.
pub fn leapfrog_step_with_mass<B, M>(
    model: &M,
    position: Tensor<B, 1>,
    momentum: Tensor<B, 1>,
    step_size: f64,
    inv_mass_matrix: Option<&[f64]>,
) -> LeapfrogResult<B>
where
    B: AutodiffBackend,
    M: BayesianModel<B>,
{
    let device = position.device();
    let half_step = step_size / 2.0;

    // Half step for momentum using gradient at current position
    let (_, grad) = logp_and_grad(model, position.clone());
    let grad_tensor = Tensor::<B, 1>::from_floats(
        grad.iter()
            .map(|&x| x as f32)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );

    // p <- p + (eps/2) * grad_log_prob
    let momentum = momentum + grad_tensor.mul_scalar(half_step as f32);

    // Full step for position: q <- q + eps * M^{-1} * p
    let position = match inv_mass_matrix {
        Some(inv_m) => {
            // q <- q + eps * diag(inv_m) * p
            let mom_data: Vec<f32> = momentum.clone().into_data().to_vec().unwrap();
            let pos_data: Vec<f32> = position.clone().into_data().to_vec().unwrap();
            let new_pos: Vec<f32> = pos_data
                .iter()
                .zip(mom_data.iter())
                .enumerate()
                .map(|(i, (&q, &p))| {
                    let inv_m_i = inv_m.get(i).copied().unwrap_or(1.0) as f32;
                    q + (step_size as f32) * inv_m_i * p
                })
                .collect();
            Tensor::<B, 1>::from_floats(new_pos.as_slice(), &device)
        }
        None => {
            // Standard: q <- q + eps * p
            position + momentum.clone().mul_scalar(step_size as f32)
        }
    };

    // Half step for momentum using gradient at new position
    let (logp_new, grad_new) = logp_and_grad(model, position.clone());
    let grad_tensor_new = Tensor::<B, 1>::from_floats(
        grad_new
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );

    // p <- p + (eps/2) * grad_log_prob
    let momentum = momentum + grad_tensor_new.mul_scalar(half_step as f32);

    LeapfrogResult {
        position,
        momentum,
        log_prob: logp_new,
        gradient: grad_new,
    }
}

/// Perform multiple leapfrog steps
///
/// This function chains multiple leapfrog steps to integrate the
/// Hamiltonian dynamics for a specified trajectory length.
///
/// # Arguments
///
/// * `model` - The Bayesian model to sample from
/// * `position` - Initial position in parameter space
/// * `momentum` - Initial momentum
/// * `step_size` - Integration step size (epsilon)
/// * `num_steps` - Number of leapfrog steps (L)
///
/// # Returns
///
/// A [`LeapfrogResult`] containing the final position, momentum, and log probability.
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::leapfrog::leapfrog;
///
/// // Integrate for 10 steps with step size 0.01
/// let result = leapfrog(&model, position, momentum, 0.01, 10);
/// ```
///
/// # Note
///
/// The total trajectory length is `step_size * num_steps`. For HMC,
/// this should be tuned to achieve good mixing while avoiding
/// U-turns in the trajectory.
pub fn leapfrog<B, M>(
    model: &M,
    position: Tensor<B, 1>,
    momentum: Tensor<B, 1>,
    step_size: f64,
    num_steps: usize,
) -> LeapfrogResult<B>
where
    B: AutodiffBackend,
    M: BayesianModel<B>,
{
    leapfrog_with_mass(model, position, momentum, step_size, num_steps, None)
}

/// Perform multiple leapfrog steps with mass matrix preconditioning
///
/// # Arguments
///
/// * `model` - The Bayesian model to sample from
/// * `position` - Initial position in parameter space
/// * `momentum` - Initial momentum
/// * `step_size` - Integration step size (epsilon)
/// * `num_steps` - Number of leapfrog steps (L)
/// * `inv_mass_matrix` - Optional inverse mass matrix (diagonal elements)
///
/// # Returns
///
/// A [`LeapfrogResult`] containing the final position, momentum, and log probability.
pub fn leapfrog_with_mass<B, M>(
    model: &M,
    position: Tensor<B, 1>,
    momentum: Tensor<B, 1>,
    step_size: f64,
    num_steps: usize,
    inv_mass_matrix: Option<&[f64]>,
) -> LeapfrogResult<B>
where
    B: AutodiffBackend,
    M: BayesianModel<B>,
{
    let mut pos = position;
    let mut mom = momentum;
    let mut logp = 0.0;
    let mut grad = vec![];

    for _ in 0..num_steps {
        let result = leapfrog_step_with_mass(model, pos, mom, step_size, inv_mass_matrix);
        pos = result.position;
        mom = result.momentum;
        logp = result.log_prob;
        grad = result.gradient;
    }

    LeapfrogResult {
        position: pos,
        momentum: mom,
        log_prob: logp,
        gradient: grad,
    }
}

/// Compute the kinetic energy for a given momentum
///
/// With mass matrix M: K(p) = 0.5 * p^T * M^{-1} * p
/// For diagonal M with elements m_i: K = sum_i 0.5 * p_i^2 / m_i
///
/// # Arguments
///
/// * `momentum` - Momentum tensor
/// * `inv_mass_matrix` - Optional inverse mass matrix (diagonal elements).
///   If None, uses identity (standard HMC).
///
/// # Returns
///
/// Scalar kinetic energy value
pub fn kinetic_energy<B: Backend>(momentum: &Tensor<B, 1>, inv_mass_matrix: Option<&[f64]>) -> f64 {
    let mom_data: Vec<f32> = momentum.clone().into_data().to_vec().unwrap();

    match inv_mass_matrix {
        Some(inv_m) => {
            // K = 0.5 * sum_i (p_i^2 / m_i) = 0.5 * sum_i (p_i^2 * inv_m_i)
            let mut ke = 0.0;
            for (i, &p) in mom_data.iter().enumerate() {
                let inv_m_i = inv_m.get(i).copied().unwrap_or(1.0);
                ke += (p as f64).powi(2) * inv_m_i;
            }
            ke / 2.0
        }
        None => {
            // Standard kinetic energy: K = 0.5 * p^T * p
            let sum: f64 = mom_data.iter().map(|&p| (p as f64).powi(2)).sum();
            sum / 2.0
        }
    }
}

/// Compute the Hamiltonian (total energy)
///
/// H(q, p) = -log_prob(q) + 0.5 * p^T * M^{-1} * p
///
/// Note: We use negative log probability as the potential energy.
///
/// # Arguments
///
/// * `log_prob` - Log probability at current position
/// * `momentum` - Current momentum
/// * `inv_mass_matrix` - Optional inverse mass matrix (diagonal elements)
///
/// # Returns
///
/// Scalar Hamiltonian value
pub fn hamiltonian<B: Backend>(
    log_prob: f64,
    momentum: &Tensor<B, 1>,
    inv_mass_matrix: Option<&[f64]>,
) -> f64 {
    let kinetic = kinetic_energy(momentum, inv_mass_matrix);
    -log_prob + kinetic
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray<f32>>;

    /// Simple quadratic model: log p(x) = -0.5 * x^2
    struct QuadraticModel {
        dim: usize,
    }

    impl BayesianModel<TestBackend> for QuadraticModel {
        fn dim(&self) -> usize {
            self.dim
        }

        fn log_prob(&self, params: &Tensor<TestBackend, 1>) -> Tensor<TestBackend, 1> {
            let squared = params.clone().powf_scalar(2.0);
            squared.mul_scalar(-0.5).sum().reshape([1])
        }

        fn param_names(&self) -> Vec<String> {
            (0..self.dim).map(|i| format!("x[{}]", i)).collect()
        }
    }

    #[test]
    fn test_kinetic_energy() {
        let device = NdArrayDevice::default();

        // p = [1, 2, 3], K = 0.5 * (1 + 4 + 9) = 7.0
        let momentum = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let ke = kinetic_energy(&momentum, None);

        assert!((ke - 7.0).abs() < 1e-5, "Expected 7.0, got {}", ke);
    }

    #[test]
    fn test_kinetic_energy_zero() {
        let device = NdArrayDevice::default();

        let momentum = Tensor::<TestBackend, 1>::zeros([3], &device);
        let ke = kinetic_energy(&momentum, None);

        assert!((ke - 0.0).abs() < 1e-10, "Expected 0.0, got {}", ke);
    }

    #[test]
    fn test_kinetic_energy_with_mass() {
        let device = NdArrayDevice::default();

        // p = [2, 2], inv_mass = [0.5, 2.0]
        // K = 0.5 * (4 * 0.5 + 4 * 2.0) = 0.5 * (2 + 8) = 5.0
        let momentum = Tensor::<TestBackend, 1>::from_floats([2.0, 2.0], &device);
        let inv_mass = vec![0.5, 2.0];
        let ke = kinetic_energy(&momentum, Some(&inv_mass));

        assert!((ke - 5.0).abs() < 1e-5, "Expected 5.0, got {}", ke);
    }

    #[test]
    fn test_hamiltonian() {
        let device = NdArrayDevice::default();

        // log_prob = -0.5, p = [1, 0, 0]
        // H = -(-0.5) + 0.5 * 1 = 0.5 + 0.5 = 1.0
        let momentum = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0, 0.0], &device);
        let h = hamiltonian(-0.5, &momentum, None);

        assert!((h - 1.0).abs() < 1e-5, "Expected 1.0, got {}", h);
    }

    #[test]
    fn test_leapfrog_step_preserves_dimensionality() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel { dim: 3 };

        let position = Tensor::<TestBackend, 1>::zeros([3], &device);
        let momentum = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0, 0.0], &device);

        let result = leapfrog_step(&model, position, momentum, 0.1);

        assert_eq!(result.position.dims()[0], 3);
        assert_eq!(result.momentum.dims()[0], 3);
        assert_eq!(result.gradient.len(), 3);
    }

    #[test]
    fn test_leapfrog_energy_conservation() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel { dim: 2 };

        // Start away from origin
        let position = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);
        let momentum = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0], &device);

        // Compute initial Hamiltonian
        let (logp_init, _) = logp_and_grad(&model, position.clone());
        let h_init = hamiltonian(logp_init, &momentum, None);

        // Take many small leapfrog steps
        let result = leapfrog(&model, position, momentum, 0.01, 100);

        // Compute final Hamiltonian
        let h_final = hamiltonian(result.log_prob, &result.momentum, None);

        // Energy should be approximately conserved (symplectic property)
        let energy_error = (h_final - h_init).abs();
        assert!(
            energy_error < 0.01,
            "Energy not conserved: initial={}, final={}, error={}",
            h_init,
            h_final,
            energy_error
        );
    }

    #[test]
    fn test_leapfrog_time_reversibility() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel { dim: 2 };

        let position = Tensor::<TestBackend, 1>::from_floats([1.0, 0.5], &device);
        let momentum = Tensor::<TestBackend, 1>::from_floats([0.5, -0.3], &device);

        // Forward integration
        let forward_result = leapfrog(&model, position.clone(), momentum.clone(), 0.05, 20);

        // Negate momentum and integrate backward
        let neg_momentum = forward_result.momentum.mul_scalar(-1.0);
        let backward_result = leapfrog(&model, forward_result.position, neg_momentum, 0.05, 20);

        // Should return approximately to starting position
        let pos_original: Vec<f32> = position.into_data().to_vec().unwrap();
        let pos_final: Vec<f32> = backward_result.position.into_data().to_vec().unwrap();

        for (orig, final_) in pos_original.iter().zip(pos_final.iter()) {
            assert!(
                (orig - final_).abs() < 0.01,
                "Time reversibility violated: {} vs {}",
                orig,
                final_
            );
        }
    }

    #[test]
    fn test_leapfrog_moves_position() {
        let device = NdArrayDevice::default();
        let model = QuadraticModel { dim: 2 };

        let position = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);
        let momentum = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);

        let result = leapfrog_step(&model, position.clone(), momentum, 0.1);

        // Position should have changed
        let pos_before: Vec<f32> = position.into_data().to_vec().unwrap();
        let pos_after: Vec<f32> = result.position.into_data().to_vec().unwrap();

        let moved = pos_before
            .iter()
            .zip(pos_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(moved, "Leapfrog step should move the position");
    }
}
