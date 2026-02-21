//! Truncated Normal distribution
//!
//! The truncated normal distribution is a normal distribution bounded to [a, b].

use super::{Distribution, Support};
use burn::prelude::*;

/// Approximate the error function using Abramowitz and Stegun formula 7.1.26
///
/// erf(x) ~ 1 - (a1*t + a2*t^2 + a3*t^3) * exp(-x^2)
/// where t = 1 / (1 + 0.47047*|x|)
fn erf_approx(x: f64) -> f64 {
    let a1: f64 = 0.3480242;
    let a2: f64 = -0.0958798;
    let a3: f64 = 0.7478556;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.47047 * abs_x);

    let poly = a1 * t + a2 * t * t + a3 * t * t * t;
    let result = 1.0 - poly * (-abs_x * abs_x).exp();

    sign * result
}

/// Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Truncated Normal distribution
///
/// # Parameters
/// - `loc` (mu): Location parameter
/// - `scale` (sigma > 0): Scale parameter
/// - `low` (a): Lower bound
/// - `high` (b): Upper bound
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::truncated_normal::TruncatedNormal;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let loc = Tensor::<B, 1>::from_floats([0.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = TruncatedNormal::new(loc, scale, -1.0, 1.0);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 0.5, -0.5], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct TruncatedNormal<B: Backend> {
    /// Location parameter (mu)
    pub loc: Tensor<B, 1>,
    /// Scale parameter (sigma)
    pub scale: Tensor<B, 1>,
    /// Lower bound (a)
    pub low: f64,
    /// Upper bound (b)
    pub high: f64,
    /// Pre-computed: -0.5*ln(2*pi) - ln(sigma)
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: ln(Phi((b-mu)/sigma) - Phi((a-mu)/sigma))
    /// Stored as a tensor for broadcasting
    log_z: Tensor<B, 1>,
}

impl<B: Backend> TruncatedNormal<B> {
    /// Create a new Truncated Normal distribution
    ///
    /// # Arguments
    /// * `loc` - Location parameter (mu)
    /// * `scale` - Scale parameter (sigma > 0)
    /// * `low` - Lower bound (a)
    /// * `high` - Upper bound (b)
    ///
    /// # Panics
    /// Panics if low >= high (in debug mode)
    pub fn new(loc: Tensor<B, 1>, scale: Tensor<B, 1>, low: f64, high: f64) -> Self {
        debug_assert!(low < high, "low must be less than high");

        let device = loc.device();

        // Pre-compute the normal log normalizer: -0.5 * ln(2*pi) - ln(sigma)
        let log_norm_const = -0.5 * (2.0 * std::f64::consts::PI).ln();
        let log_normalizer = scale.clone().log().neg().add_scalar(log_norm_const);

        // Pre-compute log_z = ln(Phi((b-mu)/sigma) - Phi((a-mu)/sigma))
        // We compute this per element of loc/scale
        let loc_data: Vec<f32> = loc.clone().into_data().to_vec().unwrap();
        let scale_data: Vec<f32> = scale.clone().into_data().to_vec().unwrap();

        let log_z_data: Vec<f32> = loc_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&mu, &sigma)| {
                let mu = mu as f64;
                let sigma = sigma as f64;
                let alpha = (low - mu) / sigma;
                let beta = (high - mu) / sigma;
                let cdf_diff = normal_cdf(beta) - normal_cdf(alpha);
                cdf_diff.max(1e-12).ln() as f32
            })
            .collect();

        let log_z = Tensor::from_floats(log_z_data.as_slice(), &device);

        Self {
            loc,
            scale,
            low,
            high,
            log_normalizer,
            log_z,
        }
    }
}

impl<B: Backend> Distribution<B> for TruncatedNormal<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Truncated Normal distribution:
        // log p(x | mu, sigma, a, b) = -0.5*ln(2*pi) - ln(sigma) - 0.5*((x-mu)/sigma)^2
        //                              - ln(Phi((b-mu)/sigma) - Phi((a-mu)/sigma))
        //
        // Breaking it down:
        // 1. log_normalizer = -0.5*ln(2*pi) - ln(sigma)  [pre-computed]
        // 2. -0.5 * z^2 where z = (x-mu)/sigma
        // 3. -log_z  [pre-computed]
        //
        // We don't enforce bounds in log_prob since the sampler operates
        // in unconstrained space and transforms handle constraints.

        let z = (x.clone() - self.loc.clone()) / self.scale.clone();
        let quadratic = z.powf_scalar(2.0).mul_scalar(-0.5);

        self.log_normalizer.clone() + quadratic - self.log_z.clone()
    }

    fn support(&self) -> Support {
        Support::Real
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_normal_cdf_values() {
        // Test the CDF approximation at known values
        // Phi(0) = 0.5
        assert!(
            (normal_cdf(0.0) - 0.5).abs() < 1e-4,
            "Phi(0) should be 0.5, got {}",
            normal_cdf(0.0)
        );

        // Phi(inf) -> 1
        assert!(
            (normal_cdf(5.0) - 1.0).abs() < 1e-4,
            "Phi(5) should be ~1.0, got {}",
            normal_cdf(5.0)
        );

        // Phi(-inf) -> 0
        assert!(
            normal_cdf(-5.0).abs() < 1e-4,
            "Phi(-5) should be ~0.0, got {}",
            normal_cdf(-5.0)
        );

        // Phi(1) ~ 0.8413
        assert!(
            (normal_cdf(1.0) - 0.8413).abs() < 1e-3,
            "Phi(1) should be ~0.8413, got {}",
            normal_cdf(1.0)
        );
    }

    #[test]
    fn test_truncated_normal_at_mode() {
        let device = Default::default();

        // TruncatedNormal(0, 1, -2, 2) at x=0 (mode)
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = TruncatedNormal::new(loc, scale, -2.0, 2.0);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // log p(0 | 0, 1, -2, 2) = -0.5*ln(2*pi) - 0 - ln(Phi(2)-Phi(-2))
        // Phi(2) - Phi(-2) ~ 0.9545
        let z = normal_cdf(2.0) - normal_cdf(-2.0);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - z.ln();

        assert!(
            (log_prob as f64 - expected).abs() < 1e-3,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_truncated_normal_higher_than_normal() {
        let device = Default::default();

        // Truncated normal should have higher log_prob within bounds
        // because probability mass is concentrated
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let trunc = TruncatedNormal::new(loc.clone(), scale.clone(), -1.0, 1.0);

        // Normal log_prob at 0
        let normal_lp = -0.5 * (2.0 * std::f64::consts::PI).ln(); // ~ -0.9189

        let x = Tensor::from_floats([0.0f32], &device);
        let trunc_lp: f32 = trunc.log_prob(&x).into_scalar().elem();

        // Truncated normal should have higher density at mode
        assert!(
            (trunc_lp as f64) > normal_lp,
            "Truncated normal log_prob ({}) should exceed normal log_prob ({})",
            trunc_lp,
            normal_lp
        );
    }

    #[test]
    fn test_truncated_normal_vectorized() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = TruncatedNormal::new(loc, scale, -2.0, 2.0);

        let x = Tensor::from_floats([-1.0f32, 0.0, 0.5, 1.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 4);

        // Mode at 0 should have highest density
        assert!(log_probs[1] > log_probs[0], "log_prob(0) > log_prob(-1)");
        assert!(log_probs[1] > log_probs[2], "log_prob(0) > log_prob(0.5)");
    }

    #[test]
    fn test_truncated_normal_symmetry() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = TruncatedNormal::new(loc, scale, -3.0, 3.0);

        let x = Tensor::from_floats([-1.0f32, 1.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        // Symmetric bounds around symmetric loc -> symmetric log_prob
        assert!(
            (log_probs[0] - log_probs[1]).abs() < 1e-5,
            "log_prob(-1) should equal log_prob(1)"
        );
    }

    #[test]
    fn test_truncated_normal_wide_bounds_approaches_normal() {
        let device = Default::default();

        // With very wide bounds, should approach normal distribution
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = TruncatedNormal::new(loc, scale, -100.0, 100.0);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // Should be very close to standard normal log_prob at 0
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "With wide bounds, should approach normal: expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_truncated_normal_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([0.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = TruncatedNormal::new(loc, scale, -1.0, 1.0);
        assert_eq!(dist.support(), Support::Real);
    }
}
