//! Log-normal distribution
//!
//! The log-normal distribution is parameterized by the mean and standard deviation
//! of the underlying normal distribution.

use super::{Distribution, Support};
use burn::prelude::*;

/// Log-normal distribution
///
/// If X ~ Normal(loc, scale), then exp(X) ~ LogNormal(loc, scale).
///
/// # Parameters
/// - `loc`: Mean of the underlying normal distribution
/// - `scale`: Standard deviation of the underlying normal distribution (must be positive)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::log_normal::LogNormal;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let loc = Tensor::<B, 1>::from_floats([0.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = LogNormal::new(loc, scale);
///
/// let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct LogNormal<B: Backend> {
    /// Location (mean of underlying normal)
    pub loc: Tensor<B, 1>,
    /// Scale (std of underlying normal)
    pub scale: Tensor<B, 1>,
    /// Pre-computed: -0.5 * log(2 * pi) - log(sigma)
    log_normalizer: Tensor<B, 1>,
}

impl<B: Backend> LogNormal<B> {
    /// Create a new LogNormal distribution
    ///
    /// # Arguments
    /// * `loc` - Mean of the underlying normal distribution
    /// * `scale` - Standard deviation of the underlying normal distribution
    pub fn new(loc: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        let log_norm_const = -0.5 * (2.0 * std::f64::consts::PI).ln();
        let log_normalizer = scale.clone().log().neg().add_scalar(log_norm_const);

        Self {
            loc,
            scale,
            log_normalizer,
        }
    }

    /// Create a standard log-normal distribution (loc=0, scale=1)
    pub fn standard(device: &B::Device) -> Self {
        let loc = Tensor::zeros([1], device);
        let scale = Tensor::ones([1], device);
        Self::new(loc, scale)
    }
}

impl<B: Backend> Distribution<B> for LogNormal<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of LogNormal distribution:
        // log p(x | mu, sigma) = -log(x) - log(sigma) - 0.5*log(2*pi) - 0.5*((log(x) - mu)/sigma)^2
        //
        // Breaking it down:
        // 1. log_normalizer = -0.5 * log(2*pi) - log(sigma)  [pre-computed]
        // 2. log_x = log(x)
        // 3. z = (log_x - mu) / sigma
        // 4. log_prob = -log_x + log_normalizer - 0.5 * z^2

        let log_x = x.clone().log();

        // Standardized value in log space
        let z = (log_x.clone() - self.loc.clone()) / self.scale.clone();

        // Quadratic term
        let quadratic = z.powf_scalar(2.0).mul_scalar(-0.5);

        // Total: -log(x) + log_normalizer + quadratic
        log_x.neg() + self.log_normalizer.clone() + quadratic
    }

    fn support(&self) -> Support {
        Support::Positive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_log_normal_at_one() {
        let device = Default::default();
        let dist = LogNormal::<TestBackend>::standard(&device);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob = dist.log_prob(&x);

        // At x=1, log(x)=0, so it's like Normal(0,1) at 0: -0.5*log(2*pi)
        let result: f32 = log_prob.into_scalar().elem();
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_log_normal_at_e() {
        let device = Default::default();
        let dist = LogNormal::<TestBackend>::standard(&device);

        // At x=e, log(x)=1, so density is exp(-0.5)/(e*sqrt(2*pi))
        let e = std::f64::consts::E as f32;
        let x = Tensor::from_floats([e], &device);
        let log_prob = dist.log_prob(&x);

        let result: f32 = log_prob.into_scalar().elem();
        // log(exp(-0.5)/(e*sqrt(2*pi))) = -0.5 - 1 - 0.5*log(2*pi)
        let expected = -0.5 - 1.0 - 0.5 * (2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_log_normal_with_params() {
        let device = Default::default();

        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([1.0f32], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([0.5f32], &device);
        let dist = LogNormal::new(loc, scale);

        // At x=e (log(x)=1), with mu=1, it's at the mode
        let e = std::f64::consts::E as f32;
        let x = Tensor::from_floats([e], &device);
        let log_prob = dist.log_prob(&x);

        let result: f32 = log_prob.into_scalar().elem();
        // log(1/(e * 0.5 * sqrt(2*pi))) = -1 - log(0.5) - 0.5*log(2*pi)
        let expected = -1.0 - (0.5_f64).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();

        assert!(
            (result as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_log_normal_vectorized() {
        let device = Default::default();
        let dist = LogNormal::<TestBackend>::standard(&device);

        let x = Tensor::from_floats([0.5f32, 1.0, 2.0], &device);
        let log_prob = dist.log_prob(&x);

        let results: Vec<f32> = log_prob.into_data().to_vec().unwrap();

        // Mode of standard log-normal is at exp(-σ²) = exp(-1) ≈ 0.368
        // For standard log-normal, x=0.5 is closer to the mode than x=1
        // log_prob(0.5) > log_prob(1) > log_prob(2)
        assert!(
            results[0] > results[1],
            "log_prob(0.5) should be greater than log_prob(1), got {} vs {}",
            results[0],
            results[1]
        );
        assert!(
            results[1] > results[2],
            "log_prob(1) should be greater than log_prob(2), got {} vs {}",
            results[1],
            results[2]
        );
    }

    #[test]
    fn test_log_normal_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let dist = LogNormal::<TestBackend>::standard(&device);
        assert_eq!(dist.support(), Support::Positive);
    }
}
