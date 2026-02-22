//! LKJ Correlation distribution
//!
//! The LKJ distribution is a prior for correlation matrices, parameterized by
//! a concentration parameter eta. It is defined over the space of D x D
//! positive-definite correlation matrices (symmetric, unit diagonal).
//!
//! This implementation works with the Cholesky factor of the correlation matrix.
//! The input to `log_prob` is a flattened lower-triangular vector of D*(D-1)/2
//! off-diagonal elements; diagonal elements are reconstructed from the
//! unit-norm row constraint.
//!
//! # Reference
//! Lewandowski, Kurowicka, and Joe (2009). "Generating random correlation
//! matrices based on vines and extended onion method."

use crate::math::ln_gamma;
use burn::prelude::*;

/// LKJ Correlation distribution over the Cholesky factor of a correlation matrix.
///
/// # Parameters
/// - `eta`: Concentration parameter (eta > 0).
///   - eta = 1 gives a uniform distribution over correlation matrices.
///   - eta > 1 concentrates toward the identity.
///   - 0 < eta < 1 concentrates toward singular matrices.
/// - `dim`: Matrix dimension D (must be >= 2).
///
/// # Input format for `log_prob`
/// A flat vector of D*(D-1)/2 values representing the strictly lower-triangular
/// elements of the Cholesky factor L, stored row-major:
///   [L[1,0], L[2,0], L[2,1], L[3,0], L[3,1], L[3,2], ...]
///
/// The diagonal elements L[i,i] are computed as:
///   L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2)
///
/// # Example
/// ```ignore
/// use bayesian_core::distributions::lkj_corr::LKJCorr;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// // 3x3 correlation matrix with eta=1 (uniform)
/// let dist = LKJCorr::<B>::new(3, 1.0, &device);
///
/// // Identity Cholesky factor: L = I, off-diag elements all zero
/// let x = Tensor::<B, 1>::from_floats([0.0, 0.0, 0.0], &device);
/// let lp = dist.log_prob(&x);
/// ```
#[derive(Debug, Clone)]
pub struct LKJCorr<B: Backend> {
    /// Concentration parameter (eta > 0)
    pub eta: f64,
    /// Matrix dimension D
    pub dim: usize,
    /// Number of free parameters: D*(D-1)/2
    num_params: usize,
    /// Pre-computed log normalizing constant (scalar tensor)
    log_normalizer: Tensor<B, 1>,
    /// Pre-computed exponents for each diagonal element L[i,i], i = 1..D-1
    /// exponent[i-1] = D - i - 1 + 2*(eta - 1)
    /// These are the powers that multiply log(L[i,i]) in the log density.
    diag_exponents: Vec<f64>,
}

impl<B: Backend> LKJCorr<B> {
    /// Create a new LKJ Correlation distribution.
    ///
    /// # Arguments
    /// * `dim` - Dimension D of the correlation matrix (must be >= 2)
    /// * `eta` - Concentration parameter (must be > 0)
    /// * `device` - Device for tensor allocation
    ///
    /// # Panics
    /// Panics if dim < 2 or eta <= 0.
    pub fn new(dim: usize, eta: f64, device: &B::Device) -> Self {
        assert!(dim >= 2, "LKJ dimension must be at least 2");
        assert!(eta > 0.0, "LKJ concentration eta must be positive");

        let num_params = dim * (dim - 1) / 2;

        // Compute the log normalizing constant for the Cholesky parameterization.
        //
        // The LKJ(eta) density over a correlation matrix C is:
        //   p(C | eta) proportional to det(C)^(eta - 1)
        //
        // When parameterized by the Cholesky factor L (C = L L^T), we have:
        //   det(C) = prod_{i=0}^{D-1} L[i,i]^2
        //   so log det(C) = 2 * sum_{i=0}^{D-1} log(L[i,i])
        //
        // The Jacobian of the transformation from C to L contributes:
        //   prod_{i=0}^{D-1} L[i,i]^(D - i - 1)
        //
        // Combined, the log density of the Cholesky factor is:
        //   log p(L | eta) = log_normalizer
        //                  + sum_{i=1}^{D-1} (D - i - 1 + 2*(eta - 1)) * log(L[i,i])
        //
        // Note: L[0,0] = 1 always for a correlation matrix, so it contributes 0.
        //
        // The normalizing constant integrates the density over the valid Cholesky
        // factors. Each row i (0-indexed, for i >= 1) contributes an integral
        // over the i-dimensional unit ball for its off-diagonal elements.
        //
        // Row i has i off-diagonal elements. Using hyperspherical coordinates
        // with radius r = sqrt(sum of squares of off-diag), and noting that
        // L[i,i] = sqrt(1 - r^2), the integral for row i is:
        //
        //   S_{i-1} * int_0^1 r^{i-1} * (1 - r^2)^{exponent/2} dr
        //
        // where S_{i-1} = 2*pi^{i/2} / Gamma(i/2) is the surface area of the
        // (i-1)-sphere. Using the substitution u = r^2:
        //
        //   = S_{i-1}/2 * int_0^1 u^{i/2 - 1} * (1-u)^{exponent/2} du
        //   = S_{i-1}/2 * B(i/2, exponent/2 + 1)
        //   = pi^{i/2} / Gamma(i/2) * B(i/2, exponent/2 + 1)
        //   = pi^{i/2} * Gamma(exponent/2 + 1) / Gamma(i/2 + exponent/2 + 1)
        //
        // This is valid for exponent > -2, which holds for all eta > 0.

        let mut log_norm = 0.0f64;
        let mut diag_exponents = Vec::with_capacity(dim - 1);

        for i in 1..dim {
            let k = i as f64; // number of off-diagonal elements in row i
            let exponent = (dim - i - 1) as f64 + 2.0 * (eta - 1.0);
            diag_exponents.push(exponent);

            // log integral for row i:
            //   (i/2)*log(pi) + ln_gamma(exponent/2 + 1) - ln_gamma(i/2 + exponent/2 + 1)
            let half_k = k / 2.0;
            let half_exp_plus_1 = exponent / 2.0 + 1.0;

            log_norm += half_k * std::f64::consts::PI.ln();
            log_norm += ln_gamma(half_exp_plus_1);
            log_norm -= ln_gamma(half_k + half_exp_plus_1);
        }

        let log_normalizer = Tensor::<B, 1>::from_floats([(-log_norm) as f32], device);

        Self {
            eta,
            dim,
            num_params,
            log_normalizer,
            diag_exponents,
        }
    }

    /// Get the number of free parameters: D*(D-1)/2.
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Get the matrix dimension D.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Compute the log probability of a Cholesky factor.
    ///
    /// # Arguments
    /// * `x` - Flat vector of D*(D-1)/2 off-diagonal elements of the Cholesky
    ///   factor, stored row-major:
    ///   [L[1,0], L[2,0], L[2,1], L[3,0], ...]
    ///
    /// # Returns
    /// Scalar tensor (shape [1]) containing log p(L | eta).
    ///
    /// The diagonal elements are reconstructed as:
    ///   L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2)
    ///
    /// The log density is:
    ///   log p(L | eta) = log_normalizer
    ///                  + sum_{i=1}^{D-1} exponent_i * log(L[i,i])
    ///
    /// where exponent_i = (D - i - 1) + 2*(eta - 1).
    pub fn log_prob(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let device = x.device();

        // For dim=2, there is exactly 1 off-diagonal element.
        // For dim=D, we have D*(D-1)/2 off-diagonal elements.
        //
        // We need to compute log(L[i,i]) for each row i = 1..D-1.
        // L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2)
        //
        // Row i has i off-diagonal elements, starting at index i*(i-1)/2.

        // Accumulate the weighted sum of log-diagonals.
        // We must do this with tensor operations for autodiff compatibility.

        let mut log_diag_sum = Tensor::<B, 1>::from_floats([0.0f32], &device);

        for i in 1..self.dim {
            let row_start = i * (i - 1) / 2;
            let row_end = row_start + i;

            // Extract the off-diagonal elements for row i
            #[allow(clippy::single_range_in_vec_init)]
            let row_offdiag = x.clone().slice([row_start..row_end]);

            // Compute sum of squares of off-diagonal elements
            let sq = row_offdiag.clone() * row_offdiag;
            let sum_sq = sq.sum().reshape([1]); // shape [1]

            // L[i,i]^2 = 1 - sum_sq
            // L[i,i] = sqrt(1 - sum_sq)
            // log(L[i,i]) = 0.5 * log(1 - sum_sq)
            let one = Tensor::<B, 1>::from_floats([1.0f32], &device);
            let one_minus_sumsq = one - sum_sq;

            // Clamp to avoid log(0) for degenerate matrices
            let one_minus_sumsq = one_minus_sumsq.clamp_min(1e-30);

            let log_diag_i = one_minus_sumsq.log().mul_scalar(0.5);

            // Multiply by the exponent for this row
            let exponent = self.diag_exponents[i - 1] as f32;
            let contribution = log_diag_i.mul_scalar(exponent);

            log_diag_sum = log_diag_sum + contribution;
        }

        self.log_normalizer.clone() + log_diag_sum
    }

    /// Reconstruct the full D x D Cholesky factor from the flat off-diagonal vector.
    ///
    /// # Arguments
    /// * `x` - Flat vector of D*(D-1)/2 off-diagonal elements
    ///
    /// # Returns
    /// A Vec<f32> of length D*D representing the row-major Cholesky factor.
    pub fn to_cholesky_matrix(&self, x: &Tensor<B, 1>) -> Vec<f32> {
        let d = self.dim;
        let x_data: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        let mut l = vec![0.0f32; d * d];

        // Row 0: L[0,0] = 1, all others 0
        l[0] = 1.0;

        for i in 1..d {
            let row_start = i * (i - 1) / 2;
            let mut sum_sq = 0.0f32;

            // Fill off-diagonal elements
            for j in 0..i {
                let val = x_data[row_start + j];
                l[i * d + j] = val;
                sum_sq += val * val;
            }

            // Diagonal element
            l[i * d + i] = (1.0 - sum_sq).max(0.0).sqrt();
        }

        l
    }

    /// Reconstruct the correlation matrix C = L L^T from the flat off-diagonal vector.
    ///
    /// # Arguments
    /// * `x` - Flat vector of D*(D-1)/2 off-diagonal elements
    ///
    /// # Returns
    /// A Vec<f32> of length D*D representing the row-major correlation matrix.
    pub fn to_correlation_matrix(&self, x: &Tensor<B, 1>) -> Vec<f32> {
        let d = self.dim;
        let l = self.to_cholesky_matrix(x);

        // C = L @ L^T
        let mut c = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = 0.0f32;
                for k in 0..=j.min(i) {
                    sum += l[i * d + k] * l[j * d + k];
                }
                c[i * d + j] = sum;
                c[j * d + i] = sum;
            }
        }

        c
    }
}

/// Support type for correlation Cholesky factors.
///
/// Correlation Cholesky factors live in a constrained space where each row
/// of off-diagonal elements must have squared norm < 1. This is handled
/// separately from the standard `Support` enum variants.
impl<B: Backend> LKJCorr<B> {
    /// Get the support description for this distribution.
    ///
    /// Returns `Support::CorrelationCholesky(dim)` indicating the constrained
    /// space of Cholesky factors of correlation matrices.
    pub fn support(&self) -> super::Support {
        super::Support::CorrelationCholesky(self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_lkj_corr_identity_eta1_dim2() {
        // For D=2, eta=1 (uniform), the Cholesky factor of the identity is L = I.
        // Off-diagonal vector: [0.0] (L[1,0] = 0)
        let device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(2, 1.0, &device);

        let x = Tensor::from_floats([0.0f32], &device);
        let lp: f32 = dist.log_prob(&x).into_scalar().elem();

        // For D=2, eta=1:
        // exponent = (2 - 1 - 1) + 2*(1-1) = 0
        // So the density is constant (uniform over valid Cholesky factors).
        // The log_prob should be the negative of the log normalizer.
        assert!(
            lp.is_finite(),
            "log_prob should be finite at identity, got {}",
            lp
        );
    }

    #[test]
    fn test_lkj_corr_identity_eta1_dim3() {
        // D=3, eta=1, identity
        let device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(3, 1.0, &device);

        // Off-diagonal elements: [L[1,0], L[2,0], L[2,1]] = [0, 0, 0]
        let x = Tensor::from_floats([0.0f32, 0.0, 0.0], &device);
        let lp: f32 = dist.log_prob(&x).into_scalar().elem();

        assert!(
            lp.is_finite(),
            "log_prob should be finite at identity, got {}",
            lp
        );
    }

    #[test]
    fn test_lkj_corr_num_params() {
        let device: <TestBackend as Backend>::Device = Default::default();

        let d2 = LKJCorr::<TestBackend>::new(2, 1.0, &device);
        assert_eq!(d2.num_params(), 1);

        let d3 = LKJCorr::<TestBackend>::new(3, 1.0, &device);
        assert_eq!(d3.num_params(), 3);

        let d4 = LKJCorr::<TestBackend>::new(4, 1.0, &device);
        assert_eq!(d4.num_params(), 6);

        let d5 = LKJCorr::<TestBackend>::new(5, 1.0, &device);
        assert_eq!(d5.num_params(), 10);
    }

    #[test]
    fn test_lkj_corr_eta_gt1_prefers_identity() {
        // With eta > 1, density should be higher at identity than at a
        // matrix with off-diagonal correlation.
        let device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(3, 5.0, &device);

        // Identity: all off-diag zeros
        let x_identity = Tensor::from_floats([0.0f32, 0.0, 0.0], &device);
        let lp_identity: f32 = dist.log_prob(&x_identity).into_scalar().elem();

        // Correlated: L[1,0] = 0.5 => L[1,1] = sqrt(1 - 0.25) = sqrt(0.75)
        let x_correlated = Tensor::from_floats([0.5f32, 0.3, 0.2], &device);
        let lp_correlated: f32 = dist.log_prob(&x_correlated).into_scalar().elem();

        assert!(
            lp_identity > lp_correlated,
            "eta > 1 should prefer identity. lp_identity={}, lp_correlated={}",
            lp_identity,
            lp_correlated
        );
    }

    #[test]
    fn test_lkj_corr_eta_lt1_prefers_singular() {
        // With eta < 1, density should be higher near singular matrices
        // (diagonal elements close to 0).
        let device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(3, 0.5, &device);

        // Identity
        let x_identity = Tensor::from_floats([0.0f32, 0.0, 0.0], &device);
        let lp_identity: f32 = dist.log_prob(&x_identity).into_scalar().elem();

        // Near-singular: large off-diagonal => small diagonal
        let x_singular = Tensor::from_floats([0.95f32, 0.0, 0.95], &device);
        let lp_singular: f32 = dist.log_prob(&x_singular).into_scalar().elem();

        assert!(
            lp_singular > lp_identity,
            "eta < 1 should prefer near-singular. lp_identity={}, lp_singular={}",
            lp_identity,
            lp_singular
        );
    }

    #[test]
    fn test_lkj_corr_dim2_uniform_constant() {
        // For D=2, eta=1, the exponent is 0, so log_prob should be constant
        // regardless of the off-diagonal value (within valid range).
        let device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(2, 1.0, &device);

        let x1 = Tensor::from_floats([0.0f32], &device);
        let x2 = Tensor::from_floats([0.5f32], &device);
        let x3 = Tensor::from_floats([-0.3f32], &device);

        let lp1: f32 = dist.log_prob(&x1).into_scalar().elem();
        let lp2: f32 = dist.log_prob(&x2).into_scalar().elem();
        let lp3: f32 = dist.log_prob(&x3).into_scalar().elem();

        assert!(
            (lp1 - lp2).abs() < 1e-5,
            "D=2, eta=1 should be constant. lp1={}, lp2={}",
            lp1,
            lp2
        );
        assert!(
            (lp1 - lp3).abs() < 1e-5,
            "D=2, eta=1 should be constant. lp1={}, lp3={}",
            lp1,
            lp3
        );
    }

    #[test]
    fn test_lkj_corr_to_cholesky_matrix_identity() {
        let device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(3, 1.0, &device);

        let x = Tensor::from_floats([0.0f32, 0.0, 0.0], &device);
        let l = dist.to_cholesky_matrix(&x);

        // Should be identity
        let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (i, (&got, &exp)) in l.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "Element {} mismatch: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_lkj_corr_to_correlation_matrix_symmetry() {
        let device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(3, 1.0, &device);

        let x = Tensor::from_floats([0.5f32, 0.3, 0.2], &device);
        let c = dist.to_correlation_matrix(&x);

        // Check symmetry
        let d = 3;
        for i in 0..d {
            for j in 0..d {
                assert!(
                    (c[i * d + j] - c[j * d + i]).abs() < 1e-6,
                    "Correlation matrix not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Check unit diagonal
        for i in 0..d {
            assert!(
                (c[i * d + i] - 1.0).abs() < 1e-5,
                "Diagonal element {} should be 1.0, got {}",
                i,
                c[i * d + i]
            );
        }
    }

    #[test]
    fn test_lkj_corr_support() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(4, 2.0, &device);
        assert_eq!(
            dist.support(),
            super::super::Support::CorrelationCholesky(4)
        );
    }

    #[test]
    #[should_panic(expected = "LKJ dimension must be at least 2")]
    fn test_lkj_corr_dim_too_small() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let _ = LKJCorr::<TestBackend>::new(1, 1.0, &device);
    }

    #[test]
    #[should_panic(expected = "LKJ concentration eta must be positive")]
    fn test_lkj_corr_eta_nonpositive() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let _ = LKJCorr::<TestBackend>::new(3, 0.0, &device);
    }

    #[test]
    fn test_lkj_corr_dim2_known_value() {
        // For D=2, eta=2:
        // There is one free parameter: L[1,0] = rho (the correlation).
        // L[1,1] = sqrt(1 - rho^2).
        // exponent = (2 - 1 - 1) + 2*(2 - 1) = 0 + 2 = 2
        // log p(L) = log_norm + 2 * log(sqrt(1 - rho^2))
        //          = log_norm + log(1 - rho^2)
        //
        // At rho=0: log p = log_norm + log(1) = log_norm
        // At rho=0.5: log p = log_norm + log(1 - 0.25) = log_norm + log(0.75)

        let device = Default::default();
        let dist = LKJCorr::<TestBackend>::new(2, 2.0, &device);

        let x0 = Tensor::from_floats([0.0f32], &device);
        let x05 = Tensor::from_floats([0.5f32], &device);

        let lp0: f32 = dist.log_prob(&x0).into_scalar().elem();
        let lp05: f32 = dist.log_prob(&x05).into_scalar().elem();

        let diff = lp0 - lp05;
        let expected_diff = -(0.75_f32).ln(); // -log(0.75) = log(4/3)

        assert!(
            (diff - expected_diff).abs() < 1e-5,
            "Difference should be -log(0.75). Got diff={}, expected={}",
            diff,
            expected_diff
        );
    }
}
