//! Student's t-distribution
//!
//! The Student's t-distribution is a heavy-tailed distribution that approaches
//! the normal distribution as degrees of freedom increases.

use super::{Distribution, Support};
use crate::math::ln_gamma;
use burn::prelude::*;

/// Student's t-distribution
///
/// The t-distribution is commonly used for robust regression and as a prior
/// when outliers are expected. With df=1 it is the Cauchy distribution.
///
/// # Parameters
/// - `df`: Degrees of freedom (ν > 0)
/// - `loc`: Location parameter (default 0)
/// - `scale`: Scale parameter (σ > 0, default 1)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::student_t::StudentT;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let df = Tensor::<B, 1>::from_floats([3.0], &device);
/// let loc = Tensor::<B, 1>::from_floats([0.0], &device);
/// let scale = Tensor::<B, 1>::from_floats([1.0], &device);
/// let dist = StudentT::new(df, loc, scale);
///
/// let x = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0], &device);
/// let log_prob = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct StudentT<B: Backend> {
    /// Degrees of freedom (ν)
    pub df: Tensor<B, 1>,
    /// Location parameter
    pub loc: Tensor<B, 1>,
    /// Scale parameter
    pub scale: Tensor<B, 1>,
    /// Pre-computed: log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(νπ) - log(σ)
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed: (ν+1)/2
    half_df_plus_one: Tensor<B, 1>,
}

impl<B: Backend> StudentT<B> {
    /// Create a new Student's t-distribution.
    ///
    /// # Arguments
    /// * `df` - Degrees of freedom (ν > 0)
    /// * `loc` - Location parameter
    /// * `scale` - Scale parameter (σ > 0)
    pub fn new(df: Tensor<B, 1>, loc: Tensor<B, 1>, scale: Tensor<B, 1>) -> Self {
        // Pre-compute: log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(νπ) - log(σ)
        let df_data: Vec<f32> = df.clone().into_data().to_vec().unwrap();
        let device = df.device();

        let normalizers: Vec<f32> = df_data
            .iter()
            .map(|&nu| {
                let nu = nu as f64;
                let half_nu_plus_one = (nu + 1.0) / 2.0;
                let half_nu = nu / 2.0;
                (ln_gamma(half_nu_plus_one)
                    - ln_gamma(half_nu)
                    - 0.5 * (nu * std::f64::consts::PI).ln()) as f32
            })
            .collect();

        let normalizer_tensor = Tensor::from_floats(normalizers.as_slice(), &device);
        let log_normalizer = normalizer_tensor - scale.clone().log();

        let half_df_plus_one = df.clone().add_scalar(1.0).div_scalar(2.0);

        Self {
            df,
            loc,
            scale,
            log_normalizer,
            half_df_plus_one,
        }
    }

    /// Create a standard Student's t-distribution (loc=0, scale=1).
    pub fn standard(df: Tensor<B, 1>) -> Self {
        let device = df.device();
        let loc = Tensor::zeros([1], &device);
        let scale = Tensor::ones([1], &device);
        Self::new(df, loc, scale)
    }
}

impl<B: Backend> Distribution<B> for StudentT<B> {
    fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // Log probability of Student's t-distribution:
        // log p(x | ν, μ, σ) = log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(νπ) - log(σ)
        //                      - ((ν+1)/2) * log(1 + (1/ν)*((x-μ)/σ)²)
        //
        // Breaking it down:
        // 1. log_normalizer = log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(νπ) - log(σ)  [pre-computed]
        // 2. z = (x - μ) / σ
        // 3. log_kernel = ((ν+1)/2) * log(1 + z²/ν)

        // Compute standardized value
        let z = (x.clone() - self.loc.clone()) / self.scale.clone();
        let z_sq = z.powf_scalar(2.0);

        // Compute: 1 + z²/ν
        let inner = z_sq.clone() / self.df.clone() + Tensor::ones_like(&z_sq);

        // Compute: -((ν+1)/2) * log(1 + z²/ν)
        let log_kernel = self.half_df_plus_one.clone() * inner.log();

        self.log_normalizer.clone() - log_kernel
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
    fn test_student_t_at_zero() {
        let device = Default::default();

        // t(3) standard at x=0
        let df: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let t = StudentT::standard(df);

        let x = Tensor::from_floats([0.0], &device);
        let log_prob: f32 = t.log_prob(&x).into_scalar().elem();

        // At x=0: log(Γ(2)) - log(Γ(1.5)) - 0.5*log(3π) - 2*log(1)
        // = 0 - ln(Γ(1.5)) - 0.5*ln(3π)
        // Γ(1.5) = sqrt(π)/2 ≈ 0.8862
        // = -ln(0.8862) - 0.5*ln(9.4248)
        // ≈ 0.1207 - 1.1217 = -1.001 (approximately)
        let expected = ln_gamma(2.0) - ln_gamma(1.5) - 0.5 * (3.0 * std::f64::consts::PI).ln();
        assert!(
            (log_prob as f64 - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            log_prob
        );
    }

    #[test]
    fn test_student_t_symmetry() {
        let device = Default::default();

        let df: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let t = StudentT::standard(df);

        let x = Tensor::from_floats([-2.0, 0.0, 2.0], &device);
        let log_probs: Vec<f32> = t.log_prob(&x).into_data().to_vec().unwrap();

        // Symmetric around 0
        assert!(
            (log_probs[0] - log_probs[2]).abs() < 1e-5,
            "t-distribution should be symmetric"
        );

        // Maximum at 0
        assert!(log_probs[1] > log_probs[0], "log_prob(0) > log_prob(-2)");
    }

    #[test]
    fn test_student_t_heavier_tails_than_normal() {
        let device = Default::default();

        // t(3) should have heavier tails than t(30) (closer to normal)
        let df_low: Tensor<TestBackend, 1> = Tensor::from_floats([3.0], &device);
        let df_high: Tensor<TestBackend, 1> = Tensor::from_floats([30.0], &device);
        let t_low = StudentT::standard(df_low);
        let t_high = StudentT::standard(df_high);

        // At x=5 (far from center), low df should have higher density
        let x = Tensor::from_floats([5.0], &device);
        let log_prob_low: f32 = t_low.log_prob(&x).into_scalar().elem();
        let log_prob_high: f32 = t_high.log_prob(&x).into_scalar().elem();

        assert!(
            log_prob_low > log_prob_high,
            "t(3) should have heavier tails than t(30) at x=5"
        );
    }

    #[test]
    fn test_student_t_with_loc_scale() {
        let device = Default::default();

        // t(5, loc=2, scale=0.5)
        let df: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let loc: Tensor<TestBackend, 1> = Tensor::from_floats([2.0], &device);
        let scale: Tensor<TestBackend, 1> = Tensor::from_floats([0.5], &device);
        let t = StudentT::new(df, loc, scale);

        // At x=loc, standardized value is 0
        let x = Tensor::from_floats([2.0], &device);
        let log_prob: f32 = t.log_prob(&x).into_scalar().elem();

        // Compare with standard t(5) at 0, adjusted for scale
        let df2: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let t_std = StudentT::standard(df2);
        let x_std = Tensor::from_floats([0.0], &device);
        let log_prob_std: f32 = t_std.log_prob(&x_std).into_scalar().elem();

        // log_prob should differ by -log(scale) = -log(0.5) = ln(2)
        let expected_diff = 2.0_f32.ln();
        assert!(
            (log_prob - log_prob_std - expected_diff).abs() < 1e-4,
            "loc/scale should shift by -log(scale)"
        );
    }

    #[test]
    fn test_student_t_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let df: Tensor<TestBackend, 1> = Tensor::from_floats([5.0], &device);
        let t = StudentT::standard(df);
        assert_eq!(t.support(), Support::Real);
    }
}
