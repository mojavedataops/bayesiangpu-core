//! Adaptation algorithms for MCMC samplers
//!
//! This module implements adaptation strategies used during the warmup phase
//! of HMC and NUTS samplers to automatically tune sampler parameters.
//!
//! # Algorithms
//!
//! - **Dual Averaging**: Adapts step size to achieve target acceptance rate
//! - **Mass Matrix Adaptation**: Estimates the posterior covariance for preconditioning
//!
//! # References
//!
//! - Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems.
//!   Mathematical programming.
//! - Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler.

/// Dual averaging algorithm for step size adaptation
///
/// This algorithm adapts the step size to achieve a target acceptance rate.
/// It uses a Robbins-Monro stochastic approximation scheme with
/// polynomial-decay averaging.
///
/// # Algorithm
///
/// At each iteration m:
/// 1. Update running average: H_bar = (1 - 1/(m+t0)) * H_bar + (target - accept_prob) / (m+t0)
/// 2. Update log step size: log_eps = mu - sqrt(m) / gamma * H_bar
/// 3. Update averaged log step size with polynomial decay
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::adaptation::DualAveraging;
///
/// let mut adapter = DualAveraging::new(1.0, 0.8);
///
/// // During warmup iterations
/// for accept_prob in accept_probs {
///     adapter.update(accept_prob);
///     let step_size = adapter.step_size();
/// }
///
/// // After warmup, use averaged step size
/// let final_step_size = adapter.final_step_size();
/// ```
#[derive(Debug, Clone)]
pub struct DualAveraging {
    /// Target acceptance probability (typically 0.65-0.85)
    target_accept: f64,
    /// Free parameter controlling step size shrinkage (default: 0.05)
    gamma: f64,
    /// Stabilization parameter for early iterations (default: 10)
    t0: f64,
    /// Controls decay rate for averaging (default: 0.75)
    kappa: f64,
    /// log(10 * initial_step_size), controls asymptotic behavior
    mu: f64,

    // State variables
    /// Current log step size
    log_step_size: f64,
    /// Averaged log step size (final value used after warmup)
    log_step_size_bar: f64,
    /// Running average of acceptance statistic
    h_bar: f64,
    /// Iteration counter
    m: usize,
}

impl DualAveraging {
    /// Create a new dual averaging adapter
    ///
    /// # Arguments
    ///
    /// * `initial_step_size` - Starting step size (will be adapted)
    /// * `target_accept` - Target acceptance probability (typically 0.8 for NUTS)
    ///
    /// # Panics
    ///
    /// Panics if initial_step_size <= 0 or target_accept is not in (0, 1)
    pub fn new(initial_step_size: f64, target_accept: f64) -> Self {
        assert!(
            initial_step_size > 0.0,
            "Initial step size must be positive"
        );
        assert!(
            target_accept > 0.0 && target_accept < 1.0,
            "Target accept must be in (0, 1)"
        );

        Self {
            target_accept,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            mu: (10.0 * initial_step_size).ln(),

            log_step_size: initial_step_size.ln(),
            log_step_size_bar: 0.0,
            h_bar: 0.0,
            m: 0,
        }
    }

    /// Create a new dual averaging adapter with custom parameters
    ///
    /// # Arguments
    ///
    /// * `initial_step_size` - Starting step size
    /// * `target_accept` - Target acceptance probability
    /// * `gamma` - Step size shrinkage parameter (default: 0.05)
    /// * `t0` - Early iteration stabilization (default: 10)
    /// * `kappa` - Averaging decay rate (default: 0.75)
    pub fn with_params(
        initial_step_size: f64,
        target_accept: f64,
        gamma: f64,
        t0: f64,
        kappa: f64,
    ) -> Self {
        assert!(initial_step_size > 0.0);
        assert!(target_accept > 0.0 && target_accept < 1.0);
        assert!(gamma > 0.0);
        assert!(t0 >= 0.0);
        assert!(kappa > 0.0 && kappa <= 1.0);

        Self {
            target_accept,
            gamma,
            t0,
            kappa,
            mu: (10.0 * initial_step_size).ln(),

            log_step_size: initial_step_size.ln(),
            log_step_size_bar: 0.0,
            h_bar: 0.0,
            m: 0,
        }
    }

    /// Update the step size based on observed acceptance probability
    ///
    /// # Arguments
    ///
    /// * `accept_prob` - Acceptance probability from current iteration
    ///   (can be average acceptance across tree for NUTS)
    pub fn update(&mut self, accept_prob: f64) {
        self.m += 1;
        let m = self.m as f64;

        // Update running average of acceptance statistic
        // H_bar = (1 - 1/(m+t0)) * H_bar + (target - accept_prob) / (m+t0)
        let w = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target_accept - accept_prob);

        // Update step size: log_eps = mu - sqrt(m) / gamma * H_bar
        self.log_step_size = self.mu - (m.sqrt() / self.gamma) * self.h_bar;

        // Update averaged step size with polynomial decay
        // log_eps_bar = m^(-kappa) * log_eps + (1 - m^(-kappa)) * log_eps_bar
        let m_pow_neg_kappa = m.powf(-self.kappa);
        self.log_step_size_bar =
            m_pow_neg_kappa * self.log_step_size + (1.0 - m_pow_neg_kappa) * self.log_step_size_bar;
    }

    /// Get the current step size (for use during warmup)
    pub fn step_size(&self) -> f64 {
        self.log_step_size.exp()
    }

    /// Get the final averaged step size (for use after warmup)
    ///
    /// This is the recommended step size after adaptation completes.
    /// The averaging helps stabilize the final value.
    pub fn final_step_size(&self) -> f64 {
        self.log_step_size_bar.exp()
    }

    /// Get the current iteration count
    pub fn iteration(&self) -> usize {
        self.m
    }

    /// Get the running average of the acceptance statistic
    pub fn h_bar(&self) -> f64 {
        self.h_bar
    }

    /// Reset the adapter to initial state
    pub fn reset(&mut self, initial_step_size: f64) {
        self.mu = (10.0 * initial_step_size).ln();
        self.log_step_size = initial_step_size.ln();
        self.log_step_size_bar = 0.0;
        self.h_bar = 0.0;
        self.m = 0;
    }
}

/// Mass matrix adaptation using sample variance
///
/// This adapter collects samples during warmup and estimates the
/// posterior variance to create a preconditioning (mass) matrix.
/// Using the inverse variance as the mass matrix approximately
/// decorrelates the posterior, improving sampling efficiency.
///
/// # Strategy
///
/// During warmup:
/// 1. Collect samples in a sliding window
/// 2. Estimate variance from collected samples
/// 3. Use inverse variance as diagonal mass matrix
///
/// # Example
///
/// ```ignore
/// use bayesian_sampler::adaptation::MassMatrixAdaptation;
///
/// let mut adapter = MassMatrixAdaptation::new(10, 200);
///
/// // During warmup
/// for sample in warmup_samples {
///     adapter.add_sample(sample);
/// }
///
/// // Get diagonal mass matrix
/// let mass_matrix = adapter.diagonal_mass_matrix();
/// ```
#[derive(Debug, Clone)]
pub struct MassMatrixAdaptation {
    /// Number of parameters
    dim: usize,
    /// Collected samples
    samples: Vec<Vec<f64>>,
    /// Maximum number of samples to store
    window_size: usize,
    /// Regularization factor for variance estimation
    regularization: f64,
}

impl MassMatrixAdaptation {
    /// Create a new mass matrix adapter
    ///
    /// # Arguments
    ///
    /// * `dim` - Number of parameters
    /// * `window_size` - Number of samples to use for variance estimation
    pub fn new(dim: usize, window_size: usize) -> Self {
        Self {
            dim,
            samples: Vec::with_capacity(window_size),
            window_size,
            regularization: 1e-3,
        }
    }

    /// Create a new mass matrix adapter with custom regularization
    ///
    /// # Arguments
    ///
    /// * `dim` - Number of parameters
    /// * `window_size` - Number of samples to use
    /// * `regularization` - Regularization added to variance (prevents division by zero)
    pub fn with_regularization(dim: usize, window_size: usize, regularization: f64) -> Self {
        Self {
            dim,
            samples: Vec::with_capacity(window_size),
            window_size,
            regularization,
        }
    }

    /// Add a sample for variance estimation
    ///
    /// # Arguments
    ///
    /// * `sample` - Parameter values from current iteration
    pub fn add_sample(&mut self, sample: Vec<f64>) {
        assert_eq!(
            sample.len(),
            self.dim,
            "Sample dimension {} doesn't match adapter dimension {}",
            sample.len(),
            self.dim
        );

        self.samples.push(sample);

        // Keep only the most recent samples
        if self.samples.len() > self.window_size {
            self.samples.remove(0);
        }
    }

    /// Compute the diagonal mass matrix (inverse variance)
    ///
    /// Returns a vector of length `dim` containing the diagonal elements
    /// of the mass matrix. Each element is 1/variance for that parameter.
    ///
    /// If insufficient samples are available, returns a vector of ones
    /// (identity mass matrix).
    pub fn diagonal_mass_matrix(&self) -> Vec<f64> {
        if self.samples.len() < 2 {
            return vec![1.0; self.dim];
        }

        let n = self.samples.len() as f64;
        let mut mean = vec![0.0; self.dim];
        let mut var = vec![0.0; self.dim];

        // Compute mean
        for sample in &self.samples {
            for (i, &x) in sample.iter().enumerate() {
                mean[i] += x / n;
            }
        }

        // Compute variance (using Bessel's correction)
        for sample in &self.samples {
            for (i, &x) in sample.iter().enumerate() {
                var[i] += (x - mean[i]).powi(2) / (n - 1.0);
            }
        }

        // Return inverse variance (with regularization for numerical stability)
        var.iter()
            .map(|&v| 1.0 / (v + self.regularization))
            .collect()
    }

    /// Compute the variance for each parameter
    ///
    /// Returns a vector of variance estimates, one per parameter.
    pub fn variance(&self) -> Vec<f64> {
        if self.samples.len() < 2 {
            return vec![1.0; self.dim];
        }

        let n = self.samples.len() as f64;
        let mut mean = vec![0.0; self.dim];
        let mut var = vec![0.0; self.dim];

        // Compute mean
        for sample in &self.samples {
            for (i, &x) in sample.iter().enumerate() {
                mean[i] += x / n;
            }
        }

        // Compute variance
        for sample in &self.samples {
            for (i, &x) in sample.iter().enumerate() {
                var[i] += (x - mean[i]).powi(2) / (n - 1.0);
            }
        }

        var
    }

    /// Compute the mean for each parameter
    pub fn mean(&self) -> Vec<f64> {
        if self.samples.is_empty() {
            return vec![0.0; self.dim];
        }

        let n = self.samples.len() as f64;
        let mut mean = vec![0.0; self.dim];

        for sample in &self.samples {
            for (i, &x) in sample.iter().enumerate() {
                mean[i] += x / n;
            }
        }

        mean
    }

    /// Get the number of samples currently stored
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Check if the adapter has enough samples for reliable estimation
    pub fn is_ready(&self) -> bool {
        self.samples.len() >= 10
    }

    /// Clear all stored samples
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

/// Windowed adaptation schedule
///
/// Implements the standard Stan/PyMC adaptation windows:
/// - Initial window: Fast initial adaptation
/// - Middle windows: Slow adaptation with mass matrix updates
/// - Final window: Fixed parameters for final warmup
#[derive(Debug, Clone)]
pub struct AdaptationSchedule {
    /// Total warmup iterations
    num_warmup: usize,
    /// Initial window size (fast adaptation)
    init_window: usize,
    /// Terminal window size (no adaptation)
    term_window: usize,
    /// Base window size for middle phase
    base_window: usize,
}

impl AdaptationSchedule {
    /// Create a new adaptation schedule
    ///
    /// # Arguments
    ///
    /// * `num_warmup` - Total number of warmup iterations
    pub fn new(num_warmup: usize) -> Self {
        // Standard Stan defaults
        let init_window = 75.min(num_warmup / 4);
        let term_window = 50.min(num_warmup / 4);
        let base_window = 25;

        Self {
            num_warmup,
            init_window,
            term_window,
            base_window,
        }
    }

    /// Check if we should adapt step size at this iteration
    pub fn adapt_step_size(&self, iteration: usize) -> bool {
        iteration < self.num_warmup - self.term_window
    }

    /// Check if we should update the mass matrix at this iteration
    pub fn adapt_mass_matrix(&self, iteration: usize) -> bool {
        iteration >= self.init_window && iteration < self.num_warmup - self.term_window
    }

    /// Check if this iteration ends a mass matrix window
    pub fn is_window_end(&self, iteration: usize) -> bool {
        if iteration < self.init_window {
            return false;
        }
        if iteration >= self.num_warmup - self.term_window {
            return false;
        }

        // Window boundaries follow geometric progression
        let mut window_start = self.init_window;
        let mut window_size = self.base_window;

        while window_start < self.num_warmup - self.term_window {
            let window_end = (window_start + window_size).min(self.num_warmup - self.term_window);
            if iteration == window_end - 1 {
                return true;
            }
            window_start = window_end;
            window_size *= 2;
        }

        false
    }

    /// Get the current adaptation phase
    pub fn phase(&self, iteration: usize) -> AdaptationPhase {
        if iteration < self.init_window {
            AdaptationPhase::Initial
        } else if iteration < self.num_warmup - self.term_window {
            AdaptationPhase::Middle
        } else {
            AdaptationPhase::Terminal
        }
    }
}

/// Adaptation phase indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationPhase {
    /// Fast initial adaptation (step size only)
    Initial,
    /// Slow adaptation with mass matrix updates
    Middle,
    /// No adaptation (fixed parameters)
    Terminal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_averaging_new() {
        let adapter = DualAveraging::new(0.5, 0.8);

        assert!((adapter.step_size() - 0.5).abs() < 1e-10);
        assert_eq!(adapter.iteration(), 0);
    }

    #[test]
    fn test_dual_averaging_adapts_down() {
        // If acceptance rate is too high, step size should increase
        // If acceptance rate is too low, step size should decrease
        let mut adapter = DualAveraging::new(1.0, 0.8);

        // Simulate low acceptance rate (need smaller step size)
        for _ in 0..100 {
            adapter.update(0.3); // Much lower than target 0.8
        }

        // Step size should have decreased
        assert!(
            adapter.step_size() < 1.0,
            "Step size {} should be < 1.0 for low acceptance",
            adapter.step_size()
        );
    }

    #[test]
    fn test_dual_averaging_adapts_up() {
        let mut adapter = DualAveraging::new(0.1, 0.8);

        // Simulate high acceptance rate (could use larger step size)
        for _ in 0..100 {
            adapter.update(0.99); // Much higher than target 0.8
        }

        // Step size should have increased
        assert!(
            adapter.step_size() > 0.1,
            "Step size {} should be > 0.1 for high acceptance",
            adapter.step_size()
        );
    }

    #[test]
    fn test_dual_averaging_converges() {
        let mut adapter = DualAveraging::new(1.0, 0.8);

        // Simulate acceptance at target rate
        for _ in 0..500 {
            adapter.update(0.8);
        }

        // Step size should stabilize near initial value
        let step_size = adapter.step_size();
        assert!(
            step_size > 0.01 && step_size < 100.0,
            "Step size {} should be reasonable",
            step_size
        );
    }

    #[test]
    fn test_dual_averaging_final_step_size() {
        let mut adapter = DualAveraging::new(0.5, 0.8);

        for i in 0..200 {
            // Oscillating acceptance
            let accept = if i % 2 == 0 { 0.6 } else { 0.9 };
            adapter.update(accept);
        }

        // Final step size should be smoothed version
        let current = adapter.step_size();
        let final_val = adapter.final_step_size();

        // Both should be positive and reasonable
        assert!(
            current > 0.0 && final_val > 0.0,
            "Step sizes should be positive: current={}, final={}",
            current,
            final_val
        );
    }

    #[test]
    fn test_dual_averaging_reset() {
        let mut adapter = DualAveraging::new(1.0, 0.8);

        for _ in 0..100 {
            adapter.update(0.5);
        }

        adapter.reset(2.0);

        assert_eq!(adapter.iteration(), 0);
        assert!((adapter.step_size() - 2.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Initial step size must be positive")]
    fn test_dual_averaging_invalid_step_size() {
        let _ = DualAveraging::new(-1.0, 0.8);
    }

    #[test]
    #[should_panic(expected = "Target accept must be in (0, 1)")]
    fn test_dual_averaging_invalid_target() {
        let _ = DualAveraging::new(1.0, 1.5);
    }

    #[test]
    fn test_mass_matrix_adaptation_new() {
        let adapter = MassMatrixAdaptation::new(5, 100);

        assert_eq!(adapter.num_samples(), 0);
        assert!(!adapter.is_ready());
    }

    #[test]
    fn test_mass_matrix_identity_when_empty() {
        let adapter = MassMatrixAdaptation::new(3, 100);

        let mass = adapter.diagonal_mass_matrix();
        assert_eq!(mass.len(), 3);
        for &m in &mass {
            assert!((m - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mass_matrix_adaptation() {
        let mut adapter = MassMatrixAdaptation::new(2, 100);

        // Add samples with known variance
        // x[0] ~ N(0, 4) -> var = 4 -> mass = 0.25
        // x[1] ~ N(0, 1) -> var = 1 -> mass = 1.0
        let samples = vec![
            vec![2.0, 1.0],
            vec![-2.0, -1.0],
            vec![2.0, 1.0],
            vec![-2.0, -1.0],
            vec![0.0, 0.0],
        ];

        for sample in samples {
            adapter.add_sample(sample);
        }

        let mass = adapter.diagonal_mass_matrix();
        let var = adapter.variance();

        // Check variance estimates are reasonable
        // With these samples, var[0] ≈ 4, var[1] ≈ 1
        assert!(var[0] > var[1], "var[0] should be > var[1]");
        assert!(mass[0] < mass[1], "mass[0] should be < mass[1]");
    }

    #[test]
    fn test_mass_matrix_window() {
        let mut adapter = MassMatrixAdaptation::new(1, 3);

        adapter.add_sample(vec![1.0]);
        adapter.add_sample(vec![2.0]);
        adapter.add_sample(vec![3.0]);
        assert_eq!(adapter.num_samples(), 3);

        // Adding more should remove oldest
        adapter.add_sample(vec![4.0]);
        assert_eq!(adapter.num_samples(), 3);

        // Mean should be (2+3+4)/3 = 3
        let mean = adapter.mean();
        assert!((mean[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mass_matrix_reset() {
        let mut adapter = MassMatrixAdaptation::new(2, 100);

        adapter.add_sample(vec![1.0, 2.0]);
        adapter.add_sample(vec![3.0, 4.0]);
        assert_eq!(adapter.num_samples(), 2);

        adapter.reset();
        assert_eq!(adapter.num_samples(), 0);
    }

    #[test]
    fn test_mass_matrix_is_ready() {
        let mut adapter = MassMatrixAdaptation::new(1, 100);

        for i in 0..9 {
            adapter.add_sample(vec![i as f64]);
            assert!(!adapter.is_ready());
        }

        adapter.add_sample(vec![9.0]);
        assert!(adapter.is_ready());
    }

    #[test]
    fn test_adaptation_schedule_phases() {
        let schedule = AdaptationSchedule::new(1000);

        assert_eq!(schedule.phase(0), AdaptationPhase::Initial);
        assert_eq!(schedule.phase(50), AdaptationPhase::Initial);
        assert_eq!(schedule.phase(100), AdaptationPhase::Middle);
        assert_eq!(schedule.phase(500), AdaptationPhase::Middle);
        assert_eq!(schedule.phase(960), AdaptationPhase::Terminal);
    }

    #[test]
    fn test_adaptation_schedule_step_size() {
        let schedule = AdaptationSchedule::new(1000);

        // Should adapt step size except in terminal window
        assert!(schedule.adapt_step_size(0));
        assert!(schedule.adapt_step_size(500));
        assert!(schedule.adapt_step_size(949));
        assert!(!schedule.adapt_step_size(950));
        assert!(!schedule.adapt_step_size(999));
    }

    #[test]
    fn test_adaptation_schedule_mass_matrix() {
        let schedule = AdaptationSchedule::new(1000);

        // Should not adapt mass matrix in initial or terminal windows
        assert!(!schedule.adapt_mass_matrix(0));
        assert!(!schedule.adapt_mass_matrix(50));
        assert!(schedule.adapt_mass_matrix(100));
        assert!(schedule.adapt_mass_matrix(500));
        assert!(!schedule.adapt_mass_matrix(960));
    }
}
