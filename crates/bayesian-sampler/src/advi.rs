//! Automatic Differentiation Variational Inference (ADVI)
//!
//! This module implements ADVI for approximate Bayesian inference using
//! variational methods. ADVI approximates the posterior distribution with
//! a Gaussian and optimizes the Evidence Lower Bound (ELBO) using gradient descent.
//!
//! # Algorithm
//!
//! ADVI transforms the target posterior to an unconstrained space and then
//! approximates it with a Gaussian variational distribution. The ELBO is:
//!
//! ```text
//! ELBO = E_q[log p(x,z)] - E_q[log q(z)]
//!      = E_q[log p(x,z)] + H[q]
//! ```
//!
//! Where:
//! - q(z) is the variational approximation (Gaussian)
//! - p(x,z) is the joint model (prior x likelihood)
//! - H[q] is the entropy of the variational distribution
//!
//! # Variants
//!
//! - **MeanFieldADVI**: Diagonal covariance (independent parameters), fast but less accurate
//! - **FullRankADVI**: Full covariance matrix, slower but captures correlations
//!
//! # References
//!
//! - Kucukelbir, A., et al. (2017). Automatic Differentiation Variational Inference.
//!   Journal of Machine Learning Research.
//! - Hoffman, M. D., et al. (2013). Stochastic Variational Inference. JMLR.

use crate::model::BayesianModel;
use bayesian_rng::GpuRng;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// Configuration for ADVI
#[derive(Debug, Clone)]
pub struct AdviConfig {
    /// Number of optimization iterations
    pub num_iterations: usize,
    /// Number of Monte Carlo samples for ELBO gradient estimation
    pub num_samples: usize,
    /// Learning rate for Adam optimizer
    pub learning_rate: f64,
    /// Adam beta1 (momentum decay)
    pub beta1: f64,
    /// Adam beta2 (velocity decay)
    pub beta2: f64,
    /// Adam epsilon (numerical stability)
    pub epsilon: f64,
    /// Convergence tolerance for relative ELBO change
    pub tol_rel_change: f64,
    /// Number of iterations to check for convergence
    pub convergence_window: usize,
}

impl Default for AdviConfig {
    fn default() -> Self {
        Self {
            num_iterations: 10000,
            num_samples: 1,
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            tol_rel_change: 1e-4,
            convergence_window: 100,
        }
    }
}

impl AdviConfig {
    /// Create a new ADVI configuration
    pub fn new(num_iterations: usize, num_samples: usize, learning_rate: f64) -> Self {
        Self {
            num_iterations,
            num_samples,
            learning_rate,
            ..Default::default()
        }
    }

    /// Set Adam optimizer parameters
    pub fn with_adam(mut self, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.epsilon = epsilon;
        self
    }

    /// Set convergence parameters
    pub fn with_convergence(mut self, tol_rel_change: f64, window: usize) -> Self {
        self.tol_rel_change = tol_rel_change;
        self.convergence_window = window;
        self
    }
}

/// Result of Mean-Field ADVI optimization
#[derive(Debug, Clone)]
pub struct MeanFieldResult {
    /// Variational mean (posterior mean estimate)
    pub mu: Vec<f64>,
    /// Variational log standard deviation (log scale for numerical stability)
    pub omega: Vec<f64>,
    /// Variational standard deviation (exp(omega))
    pub sigma: Vec<f64>,
    /// ELBO values during optimization
    pub elbo_history: Vec<f64>,
    /// Final ELBO value
    pub final_elbo: f64,
    /// Number of iterations run
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
}

impl MeanFieldResult {
    /// Sample from the variational posterior
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples to draw
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of samples, each sample is a Vec<f64> of parameters
    pub fn sample(&self, n_samples: usize, rng: &mut impl FnMut() -> f64) -> Vec<Vec<f64>> {
        let dim = self.mu.len();
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let mut sample = Vec::with_capacity(dim);
            for i in 0..dim {
                // z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
                let epsilon = rng();
                sample.push(self.mu[i] + self.sigma[i] * epsilon);
            }
            samples.push(sample);
        }

        samples
    }

    /// Get the posterior covariance (diagonal)
    pub fn variance(&self) -> Vec<f64> {
        self.sigma.iter().map(|s| s * s).collect()
    }
}

/// Result of Full-Rank ADVI optimization
#[derive(Debug, Clone)]
pub struct FullRankResult {
    /// Variational mean (posterior mean estimate)
    pub mu: Vec<f64>,
    /// Lower triangular Cholesky factor of covariance (flattened, row-major)
    pub scale_tril: Vec<f64>,
    /// Dimension of the parameter space
    pub dim: usize,
    /// ELBO values during optimization
    pub elbo_history: Vec<f64>,
    /// Final ELBO value
    pub final_elbo: f64,
    /// Number of iterations run
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
}

impl FullRankResult {
    /// Sample from the variational posterior
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples to draw
    /// * `rng` - Random number generator returning N(0,1) samples
    ///
    /// # Returns
    /// Vector of samples, each sample is a Vec<f64> of parameters
    pub fn sample(&self, n_samples: usize, rng: &mut impl FnMut() -> f64) -> Vec<Vec<f64>> {
        let dim = self.dim;
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // Sample epsilon ~ N(0, I)
            let epsilon: Vec<f64> = (0..dim).map(|_| rng()).collect();

            // Transform: z = mu + L @ epsilon
            let mut sample = self.mu.clone();
            #[allow(clippy::needless_range_loop)]
            for i in 0..dim {
                for j in 0..=i {
                    sample[i] += self.scale_tril[i * dim + j] * epsilon[j];
                }
            }
            samples.push(sample);
        }

        samples
    }

    /// Get the posterior covariance matrix (Sigma = L @ L^T)
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        let dim = self.dim;
        let mut cov = vec![vec![0.0; dim]; dim];

        #[allow(clippy::needless_range_loop)]
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..=i.min(j) {
                    cov[i][j] += self.scale_tril[i * dim + k] * self.scale_tril[j * dim + k];
                }
            }
        }

        cov
    }

    /// Get the posterior standard deviation (sqrt of diagonal of covariance)
    pub fn std(&self) -> Vec<f64> {
        let dim = self.dim;
        let mut stds = vec![0.0; dim];

        #[allow(clippy::needless_range_loop)]
        for i in 0..dim {
            for k in 0..=i {
                stds[i] += self.scale_tril[i * dim + k].powi(2);
            }
            stds[i] = stds[i].sqrt();
        }

        stds
    }
}

/// Adam optimizer state
struct AdamState {
    /// First moment estimate (momentum)
    m: Vec<f64>,
    /// Second moment estimate (velocity)
    v: Vec<f64>,
    /// Current iteration
    t: usize,
    /// Beta1 parameter
    beta1: f64,
    /// Beta2 parameter
    beta2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Learning rate
    lr: f64,
}

impl AdamState {
    fn new(dim: usize, lr: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            t: 0,
            beta1,
            beta2,
            epsilon,
            lr,
        }
    }

    fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        self.t += 1;
        let t = self.t as f64;

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);

        for i in 0..params.len() {
            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];

            // Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];

            // Compute bias-corrected estimates
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Update parameters (gradient ascent for ELBO maximization)
            params[i] += self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

/// Mean-Field ADVI sampler
///
/// Approximates the posterior with independent Gaussians for each parameter:
/// q(z) = prod_i Normal(mu_i, sigma_i^2)
///
/// This is fast but doesn't capture correlations between parameters.
///
/// # Type Parameters
///
/// * `B` - The autodiff-enabled backend
/// * `M` - The Bayesian model type
pub struct MeanFieldAdvi<B: AutodiffBackend, M: BayesianModel<B>> {
    /// The Bayesian model
    model: M,
    /// ADVI configuration
    config: AdviConfig,
    /// Random number generator
    rng: GpuRng<B>,
}

impl<B: AutodiffBackend, M: BayesianModel<B>> MeanFieldAdvi<B, M> {
    /// Create a new Mean-Field ADVI optimizer
    ///
    /// # Arguments
    /// * `model` - The Bayesian model to approximate
    /// * `config` - ADVI configuration
    /// * `rng` - Random number generator
    pub fn new(model: M, config: AdviConfig, rng: GpuRng<B>) -> Self {
        Self { model, config, rng }
    }

    /// Compute ELBO estimate using reparameterization trick
    ///
    /// ELBO = E_q[log p(z)] - E_q[log q(z)]
    ///      = E_q[log p(z)] + H[q]
    ///
    /// For mean-field Gaussian:
    /// H[q] = sum_i [0.5 * log(2*pi*e) + log(sigma_i)]
    ///      = 0.5 * D * log(2*pi*e) + sum_i log(sigma_i)
    ///      = 0.5 * D * (1 + log(2*pi)) + sum_i omega_i
    ///
    /// where omega_i = log(sigma_i)
    fn compute_elbo(&mut self, mu: &Tensor<B, 1>, omega: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = mu.device();
        let dim = self.model.dim();

        // sigma = exp(omega)
        let sigma = omega.clone().exp();

        // Sample epsilon ~ N(0, I)
        let mut elbo_sum = Tensor::<B, 1>::zeros([1], &device);

        for _ in 0..self.config.num_samples {
            let epsilon = self.rng.normal(&[dim]);

            // Reparameterization: z = mu + sigma * epsilon
            let z = mu.clone() + sigma.clone() * epsilon;

            // Compute log p(z) (model log probability)
            let log_p = self.model.log_prob(&z);

            elbo_sum = elbo_sum + log_p;
        }

        // Average over samples
        let avg_log_p = elbo_sum.div_scalar(self.config.num_samples as f32);

        // Entropy term: H[q] = 0.5 * D * (1 + log(2*pi)) + sum(omega)
        let half_d = 0.5 * dim as f32;
        let log_2pi = (2.0 * std::f64::consts::PI).ln() as f32;
        let entropy_const = half_d * (1.0 + log_2pi);
        let entropy = omega.clone().sum().reshape([1])
            + Tensor::<B, 1>::from_floats([entropy_const], &device);

        // ELBO = E[log p] + H[q]
        avg_log_p + entropy
    }

    /// Run ADVI optimization
    ///
    /// # Arguments
    /// * `init_mu` - Optional initial mean (default: zeros)
    /// * `init_omega` - Optional initial log-std (default: zeros, i.e., unit variance)
    ///
    /// # Returns
    /// A [`MeanFieldResult`] containing the optimized variational parameters
    pub fn fit(
        &mut self,
        init_mu: Option<Vec<f64>>,
        init_omega: Option<Vec<f64>>,
    ) -> MeanFieldResult {
        let dim = self.model.dim();
        let device = self.rng.device().clone();

        // Initialize variational parameters
        let mut mu: Vec<f64> = init_mu.unwrap_or_else(|| vec![0.0; dim]);
        let mut omega: Vec<f64> = init_omega.unwrap_or_else(|| vec![0.0; dim]);

        // Adam optimizer for both mu and omega
        let mut adam_mu = AdamState::new(
            dim,
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
        );
        let mut adam_omega = AdamState::new(
            dim,
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
        );

        let mut elbo_history = Vec::with_capacity(self.config.num_iterations);
        let mut converged = false;

        for iter in 0..self.config.num_iterations {
            // Create tensors with gradient tracking
            let mu_f32: Vec<f32> = mu.iter().map(|&x| x as f32).collect();
            let omega_f32: Vec<f32> = omega.iter().map(|&x| x as f32).collect();

            let mu_tensor = Tensor::<B, 1>::from_floats(mu_f32.as_slice(), &device).require_grad();
            let omega_tensor =
                Tensor::<B, 1>::from_floats(omega_f32.as_slice(), &device).require_grad();

            // Compute ELBO
            let elbo = self.compute_elbo(&mu_tensor, &omega_tensor);

            // Extract ELBO value before backward pass
            let elbo_val: f64 = {
                let data: Vec<f32> = elbo.clone().into_data().to_vec().unwrap();
                data[0] as f64
            };
            elbo_history.push(elbo_val);

            // Compute gradients
            let grads = elbo.backward();

            let grad_mu = mu_tensor
                .grad(&grads)
                .expect("Gradient for mu should exist");
            let grad_omega = omega_tensor
                .grad(&grads)
                .expect("Gradient for omega should exist");

            // Extract gradients
            let grad_mu_vec: Vec<f64> = {
                let data: Vec<f32> = grad_mu.into_data().to_vec().unwrap();
                data.iter().map(|&x| x as f64).collect()
            };
            let grad_omega_vec: Vec<f64> = {
                let data: Vec<f32> = grad_omega.into_data().to_vec().unwrap();
                data.iter().map(|&x| x as f64).collect()
            };

            // Adam update (gradient ascent for ELBO maximization)
            adam_mu.step(&mut mu, &grad_mu_vec);
            adam_omega.step(&mut omega, &grad_omega_vec);

            // Check convergence
            if iter >= self.config.convergence_window {
                let recent_start = iter - self.config.convergence_window;
                let old_elbo = elbo_history[recent_start];
                let new_elbo = elbo_val;

                if old_elbo.is_finite() && new_elbo.is_finite() {
                    let rel_change = ((new_elbo - old_elbo) / old_elbo.abs()).abs();
                    if rel_change < self.config.tol_rel_change {
                        converged = true;
                        break;
                    }
                }
            }
        }

        // Compute final sigma from omega
        let sigma: Vec<f64> = omega.iter().map(|&w| w.exp()).collect();

        let final_elbo = *elbo_history.last().unwrap_or(&f64::NEG_INFINITY);

        let iterations = elbo_history.len();

        MeanFieldResult {
            mu,
            omega,
            sigma,
            elbo_history,
            final_elbo,
            iterations,
            converged,
        }
    }

    /// Get reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get reference to the configuration
    pub fn config(&self) -> &AdviConfig {
        &self.config
    }

    /// Reset the RNG with a new seed
    pub fn reseed(&mut self, seed: u64) {
        self.rng.reseed(seed);
    }
}

/// Full-Rank ADVI sampler
///
/// Approximates the posterior with a multivariate Gaussian:
/// q(z) = MVN(mu, Sigma) where Sigma = L @ L^T
///
/// This is slower than mean-field but captures parameter correlations.
///
/// # Type Parameters
///
/// * `B` - The autodiff-enabled backend
/// * `M` - The Bayesian model type
pub struct FullRankAdvi<B: AutodiffBackend, M: BayesianModel<B>> {
    /// The Bayesian model
    model: M,
    /// ADVI configuration
    config: AdviConfig,
    /// Random number generator
    rng: GpuRng<B>,
}

impl<B: AutodiffBackend, M: BayesianModel<B>> FullRankAdvi<B, M> {
    /// Create a new Full-Rank ADVI optimizer
    ///
    /// # Arguments
    /// * `model` - The Bayesian model to approximate
    /// * `config` - ADVI configuration
    /// * `rng` - Random number generator
    pub fn new(model: M, config: AdviConfig, rng: GpuRng<B>) -> Self {
        Self { model, config, rng }
    }

    /// Compute ELBO for full-rank approximation
    ///
    /// For full-rank Gaussian with Cholesky factor L:
    /// H[q] = 0.5 * D * (1 + log(2*pi)) + sum_i log(L_ii)
    fn compute_elbo(&mut self, mu: &Tensor<B, 1>, scale_tril: &Tensor<B, 2>) -> Tensor<B, 1> {
        let device = mu.device();
        let dim = self.model.dim();

        let mut elbo_sum = Tensor::<B, 1>::zeros([1], &device);

        for _ in 0..self.config.num_samples {
            // Sample epsilon ~ N(0, I)
            let epsilon = self.rng.normal(&[dim]);

            // Reparameterization: z = mu + L @ epsilon
            // L is [D, D], epsilon is [D], result is [D]
            let epsilon_col = epsilon.reshape([dim, 1]);
            let z_offset = scale_tril.clone().matmul(epsilon_col).reshape([dim]);
            let z = mu.clone() + z_offset;

            // Compute log p(z)
            let log_p = self.model.log_prob(&z);

            elbo_sum = elbo_sum + log_p;
        }

        // Average over samples
        let avg_log_p = elbo_sum.div_scalar(self.config.num_samples as f32);

        // Entropy term: H[q] = 0.5 * D * (1 + log(2*pi)) + sum_i log(L_ii)
        // Extract diagonal elements of L and sum their logs
        let half_d = 0.5 * dim as f32;
        let log_2pi = (2.0 * std::f64::consts::PI).ln() as f32;
        let entropy_const = half_d * (1.0 + log_2pi);

        // Sum of log of diagonal elements
        // We need to extract diagonal and take log
        let l_data: Vec<f32> = scale_tril.clone().into_data().to_vec().unwrap();
        let mut log_diag_sum = 0.0f32;
        for i in 0..dim {
            log_diag_sum += l_data[i * dim + i].abs().ln();
        }

        let entropy = Tensor::<B, 1>::from_floats([entropy_const + log_diag_sum], &device);

        avg_log_p + entropy
    }

    /// Run Full-Rank ADVI optimization
    ///
    /// # Arguments
    /// * `init_mu` - Optional initial mean
    /// * `init_scale_tril` - Optional initial Cholesky factor (flattened, row-major)
    ///
    /// # Returns
    /// A [`FullRankResult`] containing the optimized variational parameters
    pub fn fit(
        &mut self,
        init_mu: Option<Vec<f64>>,
        init_scale_tril: Option<Vec<f64>>,
    ) -> FullRankResult {
        let dim = self.model.dim();
        let device = self.rng.device().clone();

        // Initialize variational parameters
        let mut mu: Vec<f64> = init_mu.unwrap_or_else(|| vec![0.0; dim]);

        // Initialize scale_tril as identity matrix (lower triangular)
        let mut scale_tril: Vec<f64> = init_scale_tril.unwrap_or_else(|| {
            let mut l = vec![0.0; dim * dim];
            for i in 0..dim {
                l[i * dim + i] = 1.0;
            }
            l
        });

        // Number of parameters: D (for mu) + D*(D+1)/2 (for lower triangular L)
        let num_tril = dim * (dim + 1) / 2;

        // Adam optimizer for mu
        let mut adam_mu = AdamState::new(
            dim,
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
        );

        // Adam optimizer for lower triangular elements
        // We optimize the log of diagonal elements for positivity
        let mut adam_tril = AdamState::new(
            num_tril,
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
        );

        // Convert scale_tril to unconstrained parameters
        // Diagonal elements: use log for positivity
        // Off-diagonal: use as-is
        let mut tril_params = Vec::with_capacity(num_tril);
        for i in 0..dim {
            for j in 0..=i {
                let idx = i * dim + j;
                if i == j {
                    tril_params.push(scale_tril[idx].ln());
                } else {
                    tril_params.push(scale_tril[idx]);
                }
            }
        }

        let mut elbo_history = Vec::with_capacity(self.config.num_iterations);
        let mut converged = false;

        for iter in 0..self.config.num_iterations {
            // Reconstruct scale_tril from unconstrained parameters
            let mut param_idx = 0;
            for i in 0..dim {
                for j in 0..=i {
                    let idx = i * dim + j;
                    if i == j {
                        scale_tril[idx] = tril_params[param_idx].exp();
                    } else {
                        scale_tril[idx] = tril_params[param_idx];
                    }
                    param_idx += 1;
                }
            }

            // Create tensors
            let mu_f32: Vec<f32> = mu.iter().map(|&x| x as f32).collect();
            let scale_tril_f32: Vec<f32> = scale_tril.iter().map(|&x| x as f32).collect();

            let mu_tensor = Tensor::<B, 1>::from_floats(mu_f32.as_slice(), &device).require_grad();
            let scale_tril_tensor = Tensor::<B, 1>::from_floats(scale_tril_f32.as_slice(), &device)
                .reshape([dim, dim])
                .require_grad();

            // Compute ELBO
            let elbo = self.compute_elbo(&mu_tensor, &scale_tril_tensor);

            let elbo_val: f64 = {
                let data: Vec<f32> = elbo.clone().into_data().to_vec().unwrap();
                data[0] as f64
            };
            elbo_history.push(elbo_val);

            // Compute gradients
            let grads = elbo.backward();

            let grad_mu = mu_tensor
                .grad(&grads)
                .expect("Gradient for mu should exist");
            let grad_scale_tril = scale_tril_tensor
                .grad(&grads)
                .expect("Gradient for scale_tril should exist");

            // Extract gradients
            let grad_mu_vec: Vec<f64> = {
                let data: Vec<f32> = grad_mu.into_data().to_vec().unwrap();
                data.iter().map(|&x| x as f64).collect()
            };
            let grad_scale_tril_full: Vec<f64> = {
                let data: Vec<f32> = grad_scale_tril.into_data().to_vec().unwrap();
                data.iter().map(|&x| x as f64).collect()
            };

            // Extract gradients for lower triangular elements
            // Apply chain rule for log-transformed diagonal elements
            let mut grad_tril_params = Vec::with_capacity(num_tril);
            for i in 0..dim {
                for j in 0..=i {
                    let idx = i * dim + j;
                    if i == j {
                        // d/d(log_l) = d/d(l) * l (chain rule for exp)
                        grad_tril_params.push(grad_scale_tril_full[idx] * scale_tril[idx]);
                    } else {
                        grad_tril_params.push(grad_scale_tril_full[idx]);
                    }
                }
            }

            // Adam updates
            adam_mu.step(&mut mu, &grad_mu_vec);
            adam_tril.step(&mut tril_params, &grad_tril_params);

            // Check convergence
            if iter >= self.config.convergence_window {
                let recent_start = iter - self.config.convergence_window;
                let old_elbo = elbo_history[recent_start];
                let new_elbo = elbo_val;

                if old_elbo.is_finite() && new_elbo.is_finite() {
                    let rel_change = ((new_elbo - old_elbo) / old_elbo.abs()).abs();
                    if rel_change < self.config.tol_rel_change {
                        converged = true;
                        break;
                    }
                }
            }
        }

        // Reconstruct final scale_tril
        {
            let mut param_idx = 0;
            for i in 0..dim {
                for j in 0..=i {
                    let idx = i * dim + j;
                    if i == j {
                        scale_tril[idx] = tril_params[param_idx].exp();
                    } else {
                        scale_tril[idx] = tril_params[param_idx];
                    }
                    param_idx += 1;
                }
            }
        }

        let final_elbo = *elbo_history.last().unwrap_or(&f64::NEG_INFINITY);
        let iterations = elbo_history.len();

        FullRankResult {
            mu,
            scale_tril,
            dim,
            elbo_history,
            final_elbo,
            iterations,
            converged,
        }
    }

    /// Get reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get reference to the configuration
    pub fn config(&self) -> &AdviConfig {
        &self.config
    }

    /// Reset the RNG with a new seed
    pub fn reseed(&mut self, seed: u64) {
        self.rng.reseed(seed);
    }
}

/// Convergence diagnostics for ADVI
#[derive(Debug, Clone)]
pub struct ConvergenceDiagnostics {
    /// Final ELBO value
    pub final_elbo: f64,
    /// ELBO standard deviation in final window
    pub elbo_std: f64,
    /// Relative change in ELBO
    pub rel_change: f64,
    /// Whether the optimization converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

impl ConvergenceDiagnostics {
    /// Compute convergence diagnostics from ELBO history
    pub fn from_history(history: &[f64], window: usize, converged: bool) -> Self {
        let n = history.len();
        let final_elbo = *history.last().unwrap_or(&f64::NEG_INFINITY);

        // Compute statistics on final window
        let window_start = n.saturating_sub(window);
        let window_values: Vec<f64> = history[window_start..].to_vec();

        let mean: f64 = window_values.iter().sum::<f64>() / window_values.len() as f64;
        let variance: f64 = window_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / window_values.len() as f64;
        let elbo_std = variance.sqrt();

        let rel_change = if n >= window && history[window_start].is_finite() {
            ((final_elbo - history[window_start]) / history[window_start].abs()).abs()
        } else {
            f64::INFINITY
        };

        Self {
            final_elbo,
            elbo_std,
            rel_change,
            converged,
            iterations: n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray<f32>>;

    /// Simple quadratic model: log p(x) = -0.5 * x^2
    /// True posterior is N(0, 1)
    struct StandardNormalModel {
        dim: usize,
    }

    impl StandardNormalModel {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    impl BayesianModel<TestBackend> for StandardNormalModel {
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
    fn test_advi_config_default() {
        let config = AdviConfig::default();

        assert_eq!(config.num_iterations, 10000);
        assert_eq!(config.num_samples, 1);
        assert!((config.learning_rate - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_mean_field_result_sample() {
        let result = MeanFieldResult {
            mu: vec![0.0, 1.0],
            omega: vec![0.0, 0.0],
            sigma: vec![1.0, 1.0],
            elbo_history: vec![-1.0],
            final_elbo: -1.0,
            iterations: 1,
            converged: true,
        };

        let mut counter = 0.0;
        let samples = result.sample(5, &mut || {
            counter += 0.1;
            counter
        });

        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0].len(), 2);
    }

    #[test]
    fn test_full_rank_result_covariance() {
        // Identity Cholesky factor
        let result = FullRankResult {
            mu: vec![0.0, 0.0],
            scale_tril: vec![1.0, 0.0, 0.0, 1.0], // Identity
            dim: 2,
            elbo_history: vec![-1.0],
            final_elbo: -1.0,
            iterations: 1,
            converged: true,
        };

        let cov = result.covariance();

        // Should be identity
        assert!((cov[0][0] - 1.0).abs() < 1e-10);
        assert!((cov[0][1] - 0.0).abs() < 1e-10);
        assert!((cov[1][0] - 0.0).abs() < 1e-10);
        assert!((cov[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_field_advi_standard_normal() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let config = AdviConfig {
            num_iterations: 500,
            num_samples: 5,
            learning_rate: 0.1,
            ..Default::default()
        };

        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut advi = MeanFieldAdvi::new(model, config, rng);

        let result = advi.fit(None, None);

        // Mean should be close to 0
        for (i, &m) in result.mu.iter().enumerate() {
            assert!(m.abs() < 1.0, "Mean[{}] = {} should be close to 0", i, m);
        }

        // Sigma should be close to 1
        for (i, &s) in result.sigma.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 0.5,
                "Sigma[{}] = {} should be close to 1",
                i,
                s
            );
        }

        // ELBO should have improved
        assert!(
            result.elbo_history.len() > 1,
            "Should have multiple ELBO values"
        );
        let first_elbo = result.elbo_history[0];
        let last_elbo = result.final_elbo;
        assert!(
            last_elbo >= first_elbo - 1.0, // Allow some noise
            "ELBO should improve or stay similar"
        );
    }

    #[test]
    fn test_full_rank_advi_standard_normal() {
        let device = NdArrayDevice::default();
        let model = StandardNormalModel::new(2);

        let config = AdviConfig {
            num_iterations: 200,
            num_samples: 3,
            learning_rate: 0.05,
            ..Default::default()
        };

        let rng = GpuRng::<TestBackend>::new(42, 64, &device);
        let mut advi = FullRankAdvi::new(model, config, rng);

        let result = advi.fit(None, None);

        // Mean should be close to 0
        for (i, &m) in result.mu.iter().enumerate() {
            assert!(m.abs() < 1.5, "Mean[{}] = {} should be close to 0", i, m);
        }

        // Standard deviation should be close to 1
        let stds = result.std();
        for (i, &s) in stds.iter().enumerate() {
            assert!(
                s > 0.1 && s < 5.0,
                "Std[{}] = {} should be reasonable",
                i,
                s
            );
        }
    }

    #[test]
    fn test_convergence_diagnostics() {
        let history: Vec<f64> = (0..100).map(|i| -(100 - i) as f64).collect();
        let diagnostics = ConvergenceDiagnostics::from_history(&history, 10, true);

        assert_eq!(diagnostics.final_elbo, -1.0);
        assert!(diagnostics.elbo_std < 10.0);
        assert!(diagnostics.iterations == 100);
        assert!(diagnostics.converged);
    }

    #[test]
    fn test_adam_state() {
        let mut adam = AdamState::new(2, 0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![0.0, 0.0];
        let grads = vec![1.0, -1.0];

        adam.step(&mut params, &grads);

        // Parameters should have moved in direction of gradient
        assert!(params[0] > 0.0);
        assert!(params[1] < 0.0);
    }

    #[test]
    fn test_mean_field_variance() {
        let result = MeanFieldResult {
            mu: vec![0.0, 0.0],
            omega: vec![0.0, (2.0_f64).ln()],
            sigma: vec![1.0, 2.0],
            elbo_history: vec![],
            final_elbo: 0.0,
            iterations: 0,
            converged: false,
        };

        let var = result.variance();
        assert!((var[0] - 1.0).abs() < 1e-10);
        assert!((var[1] - 4.0).abs() < 1e-10);
    }
}
