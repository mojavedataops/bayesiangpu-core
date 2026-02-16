//! Parameter transformations for constrained optimization
//!
//! This module provides bijective transformations between constrained and
//! unconstrained parameter spaces. These are essential for gradient-based
//! inference (HMC/NUTS) where parameters must be transformed to unconstrained
//! space for the sampler to work correctly.
//!
//! Each transform provides:
//! - `forward`: Transform from unconstrained to constrained space
//! - `inverse`: Transform from constrained to unconstrained space
//! - `log_det_jacobian`: Log absolute determinant of the Jacobian
//!
//! The log Jacobian is needed to correctly adjust probability densities when
//! changing variables.

use burn::prelude::*;

/// Transform from unconstrained R^(K-1) to the K-simplex.
///
/// Uses the stick-breaking (softmax) parameterization to map from
/// K-1 unconstrained real values to K non-negative values that sum to 1.
///
/// # Transform Details
///
/// Given unconstrained y ∈ R^(K-1), the simplex values x are computed as:
///
/// 1. z_k = sigmoid(y_k + log(K - k)) for k = 1, ..., K-1
/// 2. x_k = z_k × (1 - ∑_{j<k} x_j)
/// 3. x_K = 1 - ∑_{k<K} x_k
///
/// The offset log(K - k) helps center the prior near uniform.
///
/// # Jacobian
///
/// The log absolute determinant of the Jacobian is:
/// log |det(∂x/∂y)| = ∑_{k=1}^{K-1} [log(z_k) + log(1 - z_k) + log(1 - ∑_{j<k} x_j)]
///
/// # Example
/// ```ignore
/// use bayesian_core::transforms::SimplexTransform;
/// use burn::prelude::*;
/// use burn::backend::NdArray;
///
/// type B = NdArray<f32>;
/// let device = Default::default();
///
/// let transform = SimplexTransform::<B>::new(3, &device);
///
/// // Transform from unconstrained (K-1 = 2 values) to simplex (K = 3 values)
/// let y = Tensor::<B, 1>::from_floats([0.0, 0.0], &device);
/// let (x, log_jac) = transform.forward(&y);
/// // x is approximately [1/3, 1/3, 1/3] for y = [0, 0]
/// ```
#[derive(Debug, Clone)]
pub struct SimplexTransform<B: Backend> {
    /// Dimension of the simplex (K)
    pub dim: usize,
    /// Pre-computed offset: log(K - k) for k = 1, ..., K-1
    offsets: Tensor<B, 1>,
    /// Device for tensor operations
    device: B::Device,
}

impl<B: Backend> SimplexTransform<B> {
    /// Create a new simplex transform for a K-dimensional simplex.
    ///
    /// # Arguments
    /// * `dim` - Dimension K of the simplex (number of categories)
    /// * `device` - Device to create tensors on
    ///
    /// # Panics
    /// Panics if dim < 2.
    pub fn new(dim: usize, device: &B::Device) -> Self {
        assert!(dim >= 2, "Simplex dimension must be at least 2");

        // Pre-compute offsets: log(K - k) for k = 1, ..., K-1
        let offsets: Vec<f32> = (1..dim).map(|k| ((dim - k) as f32).ln()).collect();
        let offsets = Tensor::from_floats(offsets.as_slice(), device);

        Self {
            dim,
            offsets,
            device: device.clone(),
        }
    }

    /// Transform from unconstrained R^(K-1) to the K-simplex.
    ///
    /// # Arguments
    /// * `y` - Tensor of shape [K-1] with unconstrained real values
    ///
    /// # Returns
    /// Tuple of:
    /// - Tensor of shape [K] with simplex values (sum to 1, all positive)
    /// - Scalar tensor with log absolute Jacobian determinant
    pub fn forward(&self, y: &Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let [k_minus_1] = y.dims();
        assert_eq!(k_minus_1 + 1, self.dim, "y must have K-1 elements");

        // Compute z_k = sigmoid(y_k + offset_k)
        let adjusted = y.clone() + self.offsets.clone();
        let z = sigmoid(&adjusted);

        // Convert to Vec for stick-breaking computation
        let z_data: Vec<f32> = z.clone().into_data().to_vec().unwrap();

        // Stick-breaking: x_k = z_k × (1 - ∑_{j<k} x_j)
        let mut x = Vec::with_capacity(self.dim);
        let mut remaining = 1.0_f32;
        let mut log_jac = 0.0_f32;

        for &z_k in z_data.iter() {
            let x_k = z_k * remaining;
            x.push(x_k);

            // Accumulate log Jacobian:
            // ∂x_k/∂y_k involves z_k(1-z_k) from sigmoid derivative and remaining
            log_jac += z_k.ln() + (1.0 - z_k).ln() + remaining.ln();

            remaining -= x_k;
        }

        // Last component gets remaining probability
        x.push(remaining);

        let x_tensor = Tensor::from_floats(x.as_slice(), &self.device);
        let log_jac_tensor = Tensor::from_floats([log_jac], &self.device);

        (x_tensor, log_jac_tensor)
    }

    /// Transform from the K-simplex to unconstrained R^(K-1).
    ///
    /// # Arguments
    /// * `x` - Tensor of shape [K] with simplex values (must sum to 1)
    ///
    /// # Returns
    /// Tensor of shape [K-1] with unconstrained real values
    pub fn inverse(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let x_data: Vec<f32> = x.clone().into_data().to_vec().unwrap();

        // Inverse stick-breaking: z_k = x_k / (1 - ∑_{j<k} x_j)
        let mut z = Vec::with_capacity(self.dim - 1);
        let mut remaining = 1.0_f32;

        for x_k in x_data.iter().take(self.dim - 1) {
            let z_k = if remaining > 1e-10 {
                (x_k / remaining).clamp(1e-10, 1.0 - 1e-10)
            } else {
                0.5
            };
            z.push(z_k);
            remaining -= x_k;
        }

        // y_k = logit(z_k) - offset_k = log(z_k/(1-z_k)) - log(K-k)
        let offsets_data: Vec<f32> = self.offsets.clone().into_data().to_vec().unwrap();
        let y: Vec<f32> = z
            .iter()
            .zip(offsets_data.iter())
            .map(|(&z_k, &offset)| logit(z_k) - offset)
            .collect();

        Tensor::from_floats(y.as_slice(), &self.device)
    }

    /// Compute only the log absolute Jacobian determinant (for efficiency).
    ///
    /// # Arguments
    /// * `y` - Tensor of shape [K-1] with unconstrained real values
    ///
    /// # Returns
    /// Scalar tensor with log |det(∂x/∂y)|
    pub fn log_det_jacobian(&self, y: &Tensor<B, 1>) -> Tensor<B, 1> {
        let (_, log_jac) = self.forward(y);
        log_jac
    }

    /// Get the dimension of the simplex.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the dimension of the unconstrained space.
    pub fn unconstrained_dim(&self) -> usize {
        self.dim - 1
    }
}

/// Sigmoid function: σ(x) = 1 / (1 + exp(-x))
fn sigmoid<B: Backend>(x: &Tensor<B, 1>) -> Tensor<B, 1> {
    let neg_x = x.clone().neg();
    let exp_neg_x = neg_x.exp();
    let one = Tensor::ones_like(&exp_neg_x);
    one.clone() / (one + exp_neg_x)
}

/// Logit function (inverse sigmoid): logit(p) = log(p / (1-p))
fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

/// Transform from unconstrained R to positive R+.
///
/// Uses the exponential transformation: x = exp(y)
///
/// # Jacobian
/// log |∂x/∂y| = y (since ∂exp(y)/∂y = exp(y) = x, and log(x) = y)
#[derive(Debug, Clone)]
pub struct PositiveTransform;

impl PositiveTransform {
    /// Transform from unconstrained to positive.
    pub fn forward<B: Backend>(y: &Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let x = y.clone().exp();
        let log_jac = y.clone().sum().reshape([1]); // Sum of y values
        (x, log_jac)
    }

    /// Transform from positive to unconstrained.
    pub fn inverse<B: Backend>(x: &Tensor<B, 1>) -> Tensor<B, 1> {
        x.clone().log()
    }

    /// Compute the log Jacobian determinant.
    pub fn log_det_jacobian<B: Backend>(y: &Tensor<B, 1>) -> Tensor<B, 1> {
        y.clone().sum().reshape([1])
    }
}

/// Transform from unconstrained R to unit interval (0, 1).
///
/// Uses the sigmoid transformation: x = sigmoid(y) = 1 / (1 + exp(-y))
///
/// # Jacobian
/// log |∂x/∂y| = log(x) + log(1-x) = log(sigmoid(y)) + log(1-sigmoid(y))
#[derive(Debug, Clone)]
pub struct UnitIntervalTransform;

impl UnitIntervalTransform {
    /// Transform from unconstrained to unit interval.
    pub fn forward<B: Backend>(y: &Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let x = sigmoid(y);
        let log_jac = (x.clone().log() + (Tensor::ones_like(&x) - x.clone()).log())
            .sum()
            .reshape([1]);
        (x, log_jac)
    }

    /// Transform from unit interval to unconstrained.
    pub fn inverse<B: Backend>(x: &Tensor<B, 1>) -> Tensor<B, 1> {
        // logit(x) = log(x / (1-x)) = log(x) - log(1-x)
        let one = Tensor::ones_like(x);
        x.clone().log() - (one - x.clone()).log()
    }

    /// Compute the log Jacobian determinant.
    pub fn log_det_jacobian<B: Backend>(y: &Tensor<B, 1>) -> Tensor<B, 1> {
        let x = sigmoid(y);
        (x.clone().log() + (Tensor::ones_like(&x) - x.clone()).log())
            .sum()
            .reshape([1])
    }
}

/// Transform from unconstrained R to bounded interval (a, b).
///
/// Uses a scaled sigmoid: x = a + (b - a) × sigmoid(y)
///
/// # Jacobian
/// log |∂x/∂y| = log(b - a) + log(sigmoid(y)) + log(1 - sigmoid(y))
#[derive(Debug, Clone)]
pub struct BoundedTransform<B: Backend> {
    /// Lower bound
    pub lower: Tensor<B, 1>,
    /// Upper bound
    pub upper: Tensor<B, 1>,
    /// Pre-computed: upper - lower
    range: Tensor<B, 1>,
    /// Pre-computed: log(upper - lower)
    log_range: Tensor<B, 1>,
}

impl<B: Backend> BoundedTransform<B> {
    /// Create a new bounded transform.
    ///
    /// # Arguments
    /// * `lower` - Lower bound (scalar as 1D tensor)
    /// * `upper` - Upper bound (scalar as 1D tensor)
    pub fn new(lower: Tensor<B, 1>, upper: Tensor<B, 1>) -> Self {
        let range = upper.clone() - lower.clone();
        let log_range = range.clone().log();

        Self {
            lower,
            upper,
            range,
            log_range,
        }
    }

    /// Create a bounded transform with scalar bounds.
    pub fn from_scalars(lower: f32, upper: f32, device: &B::Device) -> Self {
        let lower_t = Tensor::from_floats([lower], device);
        let upper_t = Tensor::from_floats([upper], device);
        Self::new(lower_t, upper_t)
    }

    /// Transform from unconstrained to bounded interval.
    pub fn forward(&self, y: &Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let s = sigmoid(y);
        let x = self.lower.clone() + self.range.clone() * s.clone();

        // log_jac = log(range) + log(s) + log(1-s)
        let one = Tensor::ones_like(&s);
        let log_jac =
            self.log_range.clone() + (s.clone().log() + (one - s).log()).sum().reshape([1]);

        (x, log_jac)
    }

    /// Transform from bounded interval to unconstrained.
    pub fn inverse(&self, x: &Tensor<B, 1>) -> Tensor<B, 1> {
        let s = (x.clone() - self.lower.clone()) / self.range.clone();
        // logit(s) = log(s / (1-s))
        let one = Tensor::ones_like(&s);
        s.clone().log() - (one - s).log()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_simplex_forward_sums_to_one() {
        let device = Default::default();
        let transform = SimplexTransform::<TestBackend>::new(4, &device);

        let y = Tensor::from_floats([0.0, 0.5, -0.5], &device);
        let (x, _) = transform.forward(&y);

        let sum: f32 = x.clone().sum().into_scalar().elem();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Simplex values should sum to 1, got {}",
            sum
        );

        // All values should be positive
        let x_data: Vec<f32> = x.into_data().to_vec().unwrap();
        for &xi in &x_data {
            assert!(xi > 0.0, "Simplex values should be positive, got {}", xi);
        }
    }

    #[test]
    fn test_simplex_roundtrip() {
        let device = Default::default();
        let transform = SimplexTransform::<TestBackend>::new(3, &device);

        let y_original = Tensor::from_floats([1.0, -1.0], &device);
        let (x, _) = transform.forward(&y_original);
        let y_recovered = transform.inverse(&x);

        let y_orig_data: Vec<f32> = y_original.into_data().to_vec().unwrap();
        let y_rec_data: Vec<f32> = y_recovered.into_data().to_vec().unwrap();

        for (orig, rec) in y_orig_data.iter().zip(y_rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 1e-4,
                "Roundtrip should preserve y: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_simplex_center_reasonable() {
        let device = Default::default();
        let transform = SimplexTransform::<TestBackend>::new(3, &device);

        // y = [0, 0] gives some well-defined point on simplex
        let y = Tensor::from_floats([0.0, 0.0], &device);
        let (x, _) = transform.forward(&y);

        let x_data: Vec<f32> = x.into_data().to_vec().unwrap();

        // All values should be positive and sum to 1
        let sum: f32 = x_data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Simplex should sum to 1, got {}",
            sum
        );

        for &xi in &x_data {
            assert!(
                xi > 0.0 && xi < 1.0,
                "Each component should be in (0,1), got {}",
                xi
            );
        }
    }

    #[test]
    fn test_simplex_log_jacobian_finite() {
        let device = Default::default();
        let transform = SimplexTransform::<TestBackend>::new(5, &device);

        let y = Tensor::from_floats([0.5, -0.5, 1.0, -1.0], &device);
        let log_jac: f32 = transform.log_det_jacobian(&y).into_scalar().elem();

        assert!(
            log_jac.is_finite(),
            "Log Jacobian should be finite, got {}",
            log_jac
        );
    }

    #[test]
    fn test_positive_transform() {
        let device: <TestBackend as Backend>::Device = Default::default();

        let y = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0, -1.0], &device);
        let (x, _) = PositiveTransform::forward(&y);

        let x_data: Vec<f32> = x.into_data().to_vec().unwrap();

        // exp(0) = 1, exp(1) ≈ 2.718, exp(-1) ≈ 0.368
        assert!((x_data[0] - 1.0).abs() < 1e-5);
        assert!((x_data[1] - 1.0_f32.exp()).abs() < 1e-4);
        assert!((x_data[2] - (-1.0_f32).exp()).abs() < 1e-4);
    }

    #[test]
    fn test_positive_roundtrip() {
        let device: <TestBackend as Backend>::Device = Default::default();

        let y_original = Tensor::<TestBackend, 1>::from_floats([0.5, -0.5, 2.0], &device);
        let (x, _) = PositiveTransform::forward(&y_original);
        let y_recovered = PositiveTransform::inverse::<TestBackend>(&x);

        let y_orig_data: Vec<f32> = y_original.into_data().to_vec().unwrap();
        let y_rec_data: Vec<f32> = y_recovered.into_data().to_vec().unwrap();

        for (orig, rec) in y_orig_data.iter().zip(y_rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 1e-5,
                "Roundtrip should preserve y: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_unit_interval_transform() {
        let device: <TestBackend as Backend>::Device = Default::default();

        // y = 0 should give x = 0.5 (sigmoid center)
        let y = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let (x, _) = UnitIntervalTransform::forward(&y);

        let x_val: f32 = x.into_scalar().elem();
        assert!(
            (x_val - 0.5).abs() < 1e-5,
            "sigmoid(0) should be 0.5, got {}",
            x_val
        );
    }

    #[test]
    fn test_unit_interval_roundtrip() {
        let device: <TestBackend as Backend>::Device = Default::default();

        let y_original = Tensor::<TestBackend, 1>::from_floats([1.0, -2.0, 0.5], &device);
        let (x, _) = UnitIntervalTransform::forward(&y_original);
        let y_recovered = UnitIntervalTransform::inverse::<TestBackend>(&x);

        let y_orig_data: Vec<f32> = y_original.into_data().to_vec().unwrap();
        let y_rec_data: Vec<f32> = y_recovered.into_data().to_vec().unwrap();

        for (orig, rec) in y_orig_data.iter().zip(y_rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 1e-4,
                "Roundtrip should preserve y: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_bounded_transform() {
        let device: <TestBackend as Backend>::Device = Default::default();

        // Transform to interval [2, 8]
        let transform = BoundedTransform::<TestBackend>::from_scalars(2.0, 8.0, &device);

        // y = 0 should give midpoint = 5
        let y = Tensor::from_floats([0.0], &device);
        let (x, _) = transform.forward(&y);

        let x_val: f32 = x.into_scalar().elem();
        assert!(
            (x_val - 5.0).abs() < 1e-5,
            "Midpoint should be 5, got {}",
            x_val
        );
    }

    #[test]
    fn test_bounded_roundtrip() {
        let device: <TestBackend as Backend>::Device = Default::default();

        let transform = BoundedTransform::<TestBackend>::from_scalars(-1.0, 1.0, &device);

        let y_original = Tensor::from_floats([0.5, -0.5, 2.0], &device);
        let (x, _) = transform.forward(&y_original);
        let y_recovered = transform.inverse(&x);

        let y_orig_data: Vec<f32> = y_original.into_data().to_vec().unwrap();
        let y_rec_data: Vec<f32> = y_recovered.into_data().to_vec().unwrap();

        for (orig, rec) in y_orig_data.iter().zip(y_rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 1e-4,
                "Roundtrip should preserve y: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    #[should_panic(expected = "Simplex dimension must be at least 2")]
    fn test_simplex_invalid_dim() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let _ = SimplexTransform::<TestBackend>::new(1, &device);
    }
}
