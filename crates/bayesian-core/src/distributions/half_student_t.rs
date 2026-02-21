//! Half Student's t-distribution
//!
//! The half Student's t-distribution is a Student's t-distribution truncated
//! to positive values. It is useful as a prior for scale parameters when
//! heavier tails than HalfNormal are desired.

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Half Student's t-distribution
///
/// # Parameters
/// - `df`: Degrees of freedom (nu > 0)
/// - `scale`: Scale parameter (sigma > 0)
///
/// # Mathematical Definition
/// ```text
/// log f(x | nu, sigma) = ln(2) + ln_gamma((nu+1)/2) - ln_gamma(nu/2)
///                         - 0.5*ln(nu*pi) - ln(sigma)
///                         - ((nu+1)/2)*ln(1 + (x/sigma)^2/nu)    for x > 0
/// ```
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::half_student_t::HalfStudentT;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let df = Tensor::<B, 1>::from_floats([3.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = HalfStudentT::new(df, scale);
///
/// let x = Tensor::<B, 1>::from_floats([0.5, 1.0, 2.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct HalfStudentT<B: Backend> {
    /// Degrees of freedom (nu)
    pub df: Tensor<B, 1>,
    /// Scale parameter (sigma)
    pub scale: Tensor<B, 1>,
    /// Pre-computed: ln(2) + ln_gamma((nu+1)/2) - ln_gamma(nu/2) - 0.5*ln(nu*pi) - ln(sigma)
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: (nu+1)/2
    half_df_plus_one: Tensor<B, 1>,
}

impl<B: Backend> HalfStudentT<B> {
    /// Create a new Half Student's t-distribution
    ///
    /// # Arguments
    /// * `df` - Degrees of freedom (nu > 0)
    /// * `scale` - Scale parameter (sigma > 0)
    pub fn new(df: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        let df_data: Vec<f32> = df.clone().into_data().to_vec().unwrap();
        let device = df.device();

        let normalizers: Vec<f32> = df_data
            .iter()
            .map(|&nu| {
                let nu = nu as f64;
                let half_nu_plus_one = (nu + 1.0) / 2.0;
                let half_nu = nu / 2.0;
                (2.0_f64.ln() + ln_gamma(half_nu_plus_one)
                    - ln_gamma(half_nu)
                    - 0.5 * (nu * std::f64::consts::PI).ln()) as f32
            })
            .collect();

        let normalizer_tensor = Tensor::from_floats(normalizers.as_slice(), &device);
        let log_normalizer = normalizer_tensor - scale.clone().log();

        let half_df_plus_one = df.clone().add_scalar(1.0).div_scalar(2.0);

        Self {
            df,
            scale,
            log_normalizer,
            half_df_plus_one,
        }
    }
}

impl<B: Backend> Distribution<B> for HalfStudentT<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // log f(x | nu, sigma) = log_normalizer - ((nu+1)/2) * ln(1 + (x/sigma)^2 / nu)
        //
        // Breaking it down:
        // 1. log_normalizer [pre-computed]
        // 2. z = x / sigma
        // 3. inner = 1 + z^2 / nu
        // 4. log_kernel = ((nu+1)/2) * ln(inner)

        let z = x.clone() / self.scale.clone();
        let z_sq = z.powf_scalar(2.0);

        // 1 + z^2 / nu
        let inner = z_sq.clone() / self.df.clone() + Tensor::ones_like(&z_sq);

        // -((nu+1)/2) * ln(inner)
        let log_kernel = self.half_df_plus_one.clone() * inner.log();

        self.log_normalizer.clone() - log_kernel
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
    fn test_half_student_t_at_zero() {
        let device = Default::default();

        let df: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = HalfStudentT::new(df, scale);

        let x = Tensor::from_floats([0.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // At x=0: log_normalizer - 0 (since ln(1 + 0) = 0)
        // = ln(2) + ln_gamma(2) - ln_gamma(1.5) - 0.5*ln(3*pi) - ln(1)
        let expected =
            2.0_f64.ln() + ln_gamma(2.0) - ln_gamma(1.5) - 0.5 * (3.0 * std::f64::consts::PI).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_half_student_t_known_value() {
        let device = Default::default();

        let df: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = HalfStudentT::new(df, scale);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // Manually computed:
        // norm = ln(2) + ln_gamma(3) - ln_gamma(2.5) - 0.5*ln(5*pi)
        // kernel = -3 * ln(1 + 1/5) = -3 * ln(1.2)
        let norm =
            2.0_f64.ln() + ln_gamma(3.0) - ln_gamma(2.5) - 0.5 * (5.0 * std::f64::consts::PI).ln();
        let kernel = -3.0 * (1.2_f64).ln();
        let expected = norm + kernel;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_half_student_t_approaches_half_normal() {
        // As df -> infinity, HalfStudentT should approach HalfNormal
        let device = Default::default();

        let df_large: Tensor<TestBackend, 1> = Tensor::from_floats([10000.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist_t = HalfStudentT::new(df_large, scale.clone());

        // HalfNormal log_prob at x=1: log(sqrt(2/pi)) - 0.5
        let half_normal_expected = 0.5 * (2.0_f64.ln() - std::f64::consts::PI.ln()) - 0.5;

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob_t: f32 = dist_t.log_prob(&x).into_scalar().elem();

        // Should be close (within ~0.01 for df=10000)
        assert!(
            (log_prob_t as f64 - half_normal_expected).abs() < 0.02,
            "HalfStudentT(10000, 1) at x=1 should approximate HalfNormal(1). Expected ~{}, got {}",
            half_normal_expected,
            log_prob_t
        );
    }

    #[test]
    fn test_half_student_t_decreasing_from_zero() {
        let device = Default::default();

        let df: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = HalfStudentT::new(df, scale);

        // Density should be decreasing from x=0
        let x = Tensor::from_floats([0.0f32, 0.5, 1.0, 2.0, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        for i in 0..log_probs.len() - 1 {
            assert!(
                log_probs[i] > log_probs[i + 1],
                "HalfStudentT density should decrease from 0: log_prob[{}]={} > log_prob[{}]={}",
                i,
                log_probs[i],
                i + 1,
                log_probs[i + 1]
            );
        }
    }

    #[test]
    fn test_half_student_t_with_scale() {
        let device = Default::default();

        let df: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let dist = HalfStudentT::new(df, scale);

        let x = Tensor::from_floats([1.0f32], &device);
        let log_prob: f32 = dist.log_prob(&x).into_scalar().elem();

        // norm = ln(2) + ln_gamma(2) - ln_gamma(1.5) - 0.5*ln(3*pi) - ln(2)
        // kernel = -2 * ln(1 + (1/2)^2/3) = -2 * ln(1 + 1/12) = -2 * ln(13/12)
        let norm = 2.0_f64.ln() + ln_gamma(2.0)
            - ln_gamma(1.5)
            - 0.5 * (3.0 * std::f64::consts::PI).ln()
            - 2.0_f64.ln();
        let kernel = -2.0 * (1.0 + 1.0 / 12.0_f64).ln();
        let expected = norm + kernel;
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_half_student_t_vectorized() {
        let device = Default::default();

        let df: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = HalfStudentT::new(df, scale);

        let x = Tensor::from_floats([0.5f32, 1.0, 2.0, 5.0], &device);
        let log_probs: Vec<f32> = dist.log_prob(&x).into_data().to_vec().unwrap();

        assert_eq!(log_probs.len(), 4);
    }

    #[test]
    fn test_half_student_t_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let df: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([1.0], &device);
        let dist = HalfStudentT::new(df, scale);
        assert_eq!(dist.support(), Support::Positive);
    }
}
