//! PCG-based GPU RNG state management
//!
//! This module provides GPU-accelerated random number generation using PCG hash
//! for seed initialization and XorShift128 for the main PRNG. The implementation
//! is based on the PCG research (https://www.pcg-random.org/) and XorShift
//! algorithms described in Marsaglia's paper.
//!
//! # Design
//!
//! Each parallel thread maintains independent RNG state stored in a GPU tensor.
//! The state is a 4-element vector (vec4<u32>) per thread implementing XorShift128.
//! Seeds are initialized using PCG hash to ensure well-distributed initial states.

use burn::prelude::*;
use burn::tensor::Int;

/// GPU-accelerated Random Number Generator
///
/// This struct manages per-thread RNG state on the GPU for parallel sampling.
/// It uses XorShift128 as the underlying PRNG algorithm with PCG hash for
/// seed initialization.
///
/// # Type Parameters
///
/// * `B` - The Burn backend (should be `Wgpu` for GPU acceleration)
///
/// # Example
///
/// ```ignore
/// use bayesian_rng::GpuRng;
/// use burn::backend::Wgpu;
///
/// let device = Default::default();
/// let mut rng = GpuRng::<Wgpu>::new(42, 256, &device);
///
/// // Generate uniform samples in [0, 1)
/// let samples = rng.uniform(&[256]);
/// ```
#[derive(Debug)]
pub struct GpuRng<B: Backend> {
    /// Per-thread RNG state (4 u32s per thread for XorShift128)
    /// Shape: [num_threads, 4]
    state: Tensor<B, 2, Int>,
    /// Number of threads/parallel RNG streams
    num_threads: usize,
    /// Device for tensor operations
    device: B::Device,
}

impl<B: Backend> GpuRng<B> {
    /// Create a new GPU RNG with the specified number of threads
    ///
    /// Each thread gets its own RNG state, initialized from a unique seed
    /// derived from the base seed using PCG hash.
    ///
    /// # Arguments
    ///
    /// * `seed` - Base seed for the RNG
    /// * `num_threads` - Number of parallel RNG streams
    /// * `device` - Device for tensor operations
    ///
    /// # Example
    ///
    /// ```ignore
    /// let rng = GpuRng::<Wgpu>::new(42, 1024, &device);
    /// ```
    pub fn new(seed: u64, num_threads: usize, device: &B::Device) -> Self {
        // Initialize state with different seeds per thread using PCG hash
        // Each thread needs 4 u32 values for XorShift128 state
        let mut state_data: Vec<i64> = Vec::with_capacity(num_threads * 4);

        for thread_id in 0..num_threads {
            // Generate unique seed for each thread
            let thread_seed = seed.wrapping_add(thread_id as u64);

            // Use PCG hash to generate 4 initial state values
            let s0 = pcg_hash(thread_seed as u32);
            let s1 = pcg_hash(s0);
            let s2 = pcg_hash(s1);
            let s3 = pcg_hash(s2);

            // Ensure non-zero state (XorShift requires at least one non-zero)
            let (s0, s1, s2, s3) = if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
                (1, s1, s2, s3)
            } else {
                (s0, s1, s2, s3)
            };

            // Cast through i32 to preserve bit pattern (wrapping for values > i32::MAX)
            state_data.push((s0 as i32) as i64);
            state_data.push((s1 as i32) as i64);
            state_data.push((s2 as i32) as i64);
            state_data.push((s3 as i32) as i64);
        }

        let state =
            Tensor::<B, 1, Int>::from_ints(&state_data[..], device).reshape([num_threads, 4]);

        Self {
            state,
            num_threads,
            device: device.clone(),
        }
    }

    /// Generate uniform random samples in [0, 1)
    ///
    /// Uses the XorShift128 algorithm for state updates (CPU) and GPU tensor
    /// operations for the float conversion. This provides reproducible seeded
    /// generation while leveraging GPU for math operations.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    ///
    /// # Returns
    ///
    /// A tensor of uniform random values in [0, 1)
    pub fn uniform(&mut self, shape: &[usize]) -> Tensor<B, 1> {
        let total_samples: usize = shape.iter().product();

        // Extract current state to CPU for update
        let state_data: Vec<i64> = self.state.clone().into_data().to_vec().unwrap();

        // Mutable copy of state
        let mut state_vec: Vec<u32> = state_data.iter().map(|&x| x as u32).collect();

        // Generate u32 samples using round-robin across threads
        let mut u32_samples: Vec<i64> = Vec::with_capacity(total_samples);
        for i in 0..total_samples {
            let thread_idx = i % self.num_threads;
            let state_offset = thread_idx * 4;

            // Get current state for this thread
            let mut x = state_vec[state_offset];
            let mut y = state_vec[state_offset + 1];
            let mut z = state_vec[state_offset + 2];
            let mut w = state_vec[state_offset + 3];

            // XorShift128 step
            let t = x ^ (x << 11);
            x = y;
            y = z;
            z = w;
            w = w ^ (w >> 19) ^ t ^ (t >> 8);

            // Update state
            state_vec[state_offset] = x;
            state_vec[state_offset + 1] = y;
            state_vec[state_offset + 2] = z;
            state_vec[state_offset + 3] = w;

            // Store as i64 (preserving bits via i32 for tensor compatibility)
            u32_samples.push((w as i32) as i64);
        }

        // Update GPU state
        let new_state_data: Vec<i64> = state_vec.iter().map(|&x| (x as i32) as i64).collect();
        self.state = Tensor::<B, 1, Int>::from_ints(&new_state_data[..], &self.device)
            .reshape([self.num_threads, 4]);

        // Convert to GPU tensor and perform float conversion on GPU
        let u32_tensor = Tensor::<B, 1, Int>::from_ints(&u32_samples[..], &self.device);

        // Convert to float on GPU: cast to float then divide by 2^32
        // Use add_scalar to shift negative i32 values to positive range before division
        // i32 range is [-2^31, 2^31-1], we need [0, 2^32-1]
        u32_tensor
            .float()
            .add_scalar(2147483648.0)
            .div_scalar(4294967296.0)
    }

    /// Generate standard normal random samples using Box-Muller transform
    ///
    /// The Box-Muller transform converts pairs of uniform samples to
    /// independent standard normal samples. All math operations (sqrt, log,
    /// sin, cos) are performed on GPU using tensor operations.
    ///
    /// ```text
    /// z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
    /// z1 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    ///
    /// # Returns
    ///
    /// A tensor of standard normal random values (mean=0, std=1)
    pub fn normal(&mut self, shape: &[usize]) -> Tensor<B, 1> {
        let total_samples: usize = shape.iter().product();

        // Box-Muller produces pairs, so we need half as many pairs
        // Round up to handle odd numbers
        let num_pairs = total_samples.div_ceil(2);
        let uniform_shape = [num_pairs * 2];

        // Generate uniform samples on GPU
        let u = self.uniform(&uniform_shape);

        // Split into u1 (even indices) and u2 (odd indices)
        // We'll use slicing and reshaping for GPU-friendly operations
        let u_reshaped = u.reshape([num_pairs, 2]);
        let u1 = u_reshaped
            .clone()
            .slice([0..num_pairs, 0..1])
            .reshape([num_pairs]);
        let u2 = u_reshaped.slice([0..num_pairs, 1..2]).reshape([num_pairs]);

        // Clamp u1 away from 0 to avoid log(0) - GPU operation
        let u1_clamped = u1.clamp_min(1e-10);

        // Box-Muller on GPU:
        // r = sqrt(-2 * ln(u1))
        // theta = 2 * pi * u2
        // z0 = r * cos(theta)
        // z1 = r * sin(theta)
        let r = u1_clamped.log().mul_scalar(-2.0).sqrt();
        let theta = u2.mul_scalar(2.0 * std::f32::consts::PI);

        let z0 = r.clone() * theta.clone().cos();
        let z1 = r * theta.sin();

        // Interleave z0 and z1: [z0_0, z1_0, z0_1, z1_1, ...]
        let z0_expanded = z0.reshape([num_pairs, 1]);
        let z1_expanded = z1.reshape([num_pairs, 1]);

        // Stack along last dimension and flatten
        let interleaved = Tensor::cat(vec![z0_expanded, z1_expanded], 1).reshape([num_pairs * 2]);

        // Slice to exact size requested (handles odd total_samples)
        if total_samples < num_pairs * 2 {
            #[allow(clippy::single_range_in_vec_init)]
            interleaved.slice([0..total_samples])
        } else {
            interleaved
        }
    }

    /// Get the number of parallel RNG streams
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Get the device this RNG is running on
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Reset the RNG state with a new seed
    ///
    /// This reinitializes all thread states from the new seed.
    pub fn reseed(&mut self, seed: u64) {
        *self = Self::new(seed, self.num_threads, &self.device);
    }

    /// Generate Gamma distributed random samples using Marsaglia and Tsang's method
    ///
    /// For shape α ≥ 1, uses the transformation method.
    /// For shape α < 1, uses the identity: Gamma(α, β) = Gamma(α+1, β) * U^(1/α)
    ///
    /// Reference: Marsaglia, G. & Tsang, W.W. (2000). "A Simple Method for
    /// Generating Gamma Variables". ACM Transactions on Mathematical Software.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape parameter (α > 0), single value applied to all samples
    /// * `n` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// A tensor of Gamma(shape, 1) random values (rate=1)
    pub fn gamma(&mut self, shape: f32, n: usize) -> Tensor<B, 1> {
        assert!(shape > 0.0, "Gamma shape must be positive");

        if shape < 1.0 {
            // For α < 1: Gamma(α) = Gamma(α+1) * U^(1/α)
            let gamma_samples = self.gamma_ge_1(shape + 1.0, n);
            let uniform_samples = self.uniform(&[n]);
            let power = 1.0 / shape;
            gamma_samples * uniform_samples.powf_scalar(power)
        } else {
            self.gamma_ge_1(shape, n)
        }
    }

    /// Generate Gamma samples for shape ≥ 1 using Marsaglia-Tsang method
    ///
    /// This is the core algorithm for generating Gamma variates.
    fn gamma_ge_1(&mut self, shape: f32, n: usize) -> Tensor<B, 1> {
        // Marsaglia-Tsang: d = α - 1/3, c = 1/sqrt(9d)
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        let mut samples: Vec<f32> = Vec::with_capacity(n);

        // Extract state for CPU-side rejection sampling
        let state_data: Vec<i64> = self.state.clone().into_data().to_vec().unwrap();
        let mut state_vec: Vec<u32> = state_data.iter().map(|&x| x as u32).collect();

        let mut sample_idx = 0;
        let mut attempts = 0;
        const MAX_ATTEMPTS: usize = 1_000_000; // Safety limit

        while samples.len() < n && attempts < MAX_ATTEMPTS {
            let thread_idx = sample_idx % self.num_threads;
            let state_offset = thread_idx * 4;

            // Generate two uniforms for Box-Muller
            let (u1, u2, new_state) = self.xorshift_pair(&state_vec, state_offset);
            state_vec[state_offset..state_offset + 4].copy_from_slice(&new_state);

            // Box-Muller for standard normal
            let u1_clamped = u1.max(1e-10);
            let r = (-2.0 * u1_clamped.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let x = r * theta.cos();

            // Marsaglia-Tsang transformation
            let v = 1.0 + c * x;
            if v > 0.0 {
                let v3 = v * v * v;

                // Generate uniform for acceptance test
                let (u_accept, _, new_state2) = self.xorshift_pair(&state_vec, state_offset);
                state_vec[state_offset..state_offset + 4].copy_from_slice(&new_state2);

                // Squeeze test
                let x2 = x * x;
                if u_accept < 1.0 - 0.0331 * x2 * x2
                    || u_accept.ln() < 0.5 * x2 + d * (1.0 - v3 + v3.ln())
                {
                    samples.push(d * v3);
                }
            }

            sample_idx += 1;
            attempts += 1;
        }

        // Update GPU state
        let new_state_data: Vec<i64> = state_vec.iter().map(|&x| (x as i32) as i64).collect();
        self.state = Tensor::<B, 1, Int>::from_ints(&new_state_data[..], &self.device)
            .reshape([self.num_threads, 4]);

        // Pad with mean value if we didn't get enough samples (shouldn't happen normally)
        while samples.len() < n {
            samples.push(shape); // Use mean of Gamma(shape, 1) as fallback
        }

        Tensor::from_floats(samples.as_slice(), &self.device)
    }

    /// Helper to generate two uniform values from XorShift state
    fn xorshift_pair(&self, state_vec: &[u32], offset: usize) -> (f32, f32, [u32; 4]) {
        let mut x = state_vec[offset];
        let mut y = state_vec[offset + 1];
        let mut z = state_vec[offset + 2];
        let mut w = state_vec[offset + 3];

        // First XorShift step
        let t1 = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        w = w ^ (w >> 19) ^ t1 ^ (t1 >> 8);
        let u1 = (w as f64 / 4294967296.0) as f32;

        // Second XorShift step
        let t2 = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        w = w ^ (w >> 19) ^ t2 ^ (t2 >> 8);
        let u2 = (w as f64 / 4294967296.0) as f32;

        (u1, u2, [x, y, z, w])
    }

    /// Generate Dirichlet distributed random samples
    ///
    /// Uses the property that if X_k ~ Gamma(α_k, 1) independently,
    /// then (X_1/S, ..., X_K/S) ~ Dirichlet(α) where S = Σ X_k.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Concentration parameters (slice of K positive values)
    ///
    /// # Returns
    ///
    /// A tensor of shape [K] containing a single Dirichlet sample
    pub fn dirichlet(&mut self, alpha: &[f32]) -> Tensor<B, 1> {
        let k = alpha.len();
        assert!(k >= 2, "Dirichlet requires at least 2 categories");

        // Generate independent Gamma(α_k, 1) samples
        let mut gamma_samples: Vec<f32> = Vec::with_capacity(k);
        for &a in alpha {
            let g = self.gamma(a, 1);
            let val: f32 = g.into_scalar().elem();
            gamma_samples.push(val);
        }

        // Normalize to get Dirichlet sample
        let sum: f32 = gamma_samples.iter().sum();
        let dirichlet_sample: Vec<f32> = gamma_samples.iter().map(|&g| g / sum).collect();

        Tensor::from_floats(dirichlet_sample.as_slice(), &self.device)
    }

    /// Generate multiple Dirichlet samples
    ///
    /// # Arguments
    ///
    /// * `alpha` - Concentration parameters (slice of K positive values)
    /// * `n` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// A tensor of shape [n, K] containing n Dirichlet samples
    pub fn dirichlet_batch(&mut self, alpha: &[f32], n: usize) -> Tensor<B, 2> {
        let k = alpha.len();
        assert!(k >= 2, "Dirichlet requires at least 2 categories");

        let mut all_samples: Vec<f32> = Vec::with_capacity(n * k);

        for _ in 0..n {
            // Generate independent Gamma(α_k, 1) samples
            let mut gamma_samples: Vec<f32> = Vec::with_capacity(k);
            for &a in alpha {
                let g = self.gamma(a, 1);
                let val: f32 = g.into_scalar().elem();
                gamma_samples.push(val);
            }

            // Normalize and append
            let sum: f32 = gamma_samples.iter().sum();
            for g in gamma_samples {
                all_samples.push(g / sum);
            }
        }

        Tensor::<B, 1>::from_floats(all_samples.as_slice(), &self.device).reshape([n, k])
    }
}

impl<B: Backend> Clone for GpuRng<B> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            num_threads: self.num_threads,
            device: self.device.clone(),
        }
    }
}

/// PCG hash function for seed initialization
///
/// This is a simple but high-quality hash function from the PCG family.
/// It's used to derive independent seeds for each thread from a single
/// base seed.
///
/// Reference: https://www.pcg-random.org/
fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_pcg_hash_deterministic() {
        // PCG hash should be deterministic
        assert_eq!(pcg_hash(0), pcg_hash(0));
        assert_eq!(pcg_hash(42), pcg_hash(42));
    }

    #[test]
    fn test_pcg_hash_different_inputs() {
        // Different inputs should produce different outputs
        assert_ne!(pcg_hash(0), pcg_hash(1));
        assert_ne!(pcg_hash(42), pcg_hash(43));
    }

    #[test]
    fn test_gpu_rng_new() {
        let device = Default::default();
        let rng = GpuRng::<TestBackend>::new(42, 10, &device);

        assert_eq!(rng.num_threads(), 10);
    }

    #[test]
    fn test_uniform_range() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 16, &device);

        let samples = rng.uniform(&[1000]);
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();

        // All samples should be in [0, 1)
        for &sample in &data {
            assert!(
                sample >= 0.0 && sample < 1.0,
                "Sample {} out of range [0, 1)",
                sample
            );
        }
    }

    #[test]
    fn test_uniform_distribution() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        let samples = rng.uniform(&[10000]);
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();

        // Mean should be approximately 0.5
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            (mean - 0.5).abs() < 0.05,
            "Uniform mean {} should be close to 0.5",
            mean
        );

        // Variance should be approximately 1/12 = 0.0833
        let variance: f32 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let expected_var = 1.0 / 12.0;
        assert!(
            (variance - expected_var).abs() < 0.02,
            "Uniform variance {} should be close to {}",
            variance,
            expected_var
        );
    }

    #[test]
    fn test_normal_distribution() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        let samples = rng.normal(&[10000]);
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();

        // Mean should be approximately 0
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            mean.abs() < 0.1,
            "Normal mean {} should be close to 0",
            mean
        );

        // Standard deviation should be approximately 1
        let variance: f32 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        assert!(
            (std - 1.0).abs() < 0.1,
            "Normal std {} should be close to 1",
            std
        );
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let device = Default::default();

        let mut rng1 = GpuRng::<TestBackend>::new(42, 16, &device);
        let samples1 = rng1.uniform(&[100]);

        let mut rng2 = GpuRng::<TestBackend>::new(42, 16, &device);
        let samples2 = rng2.uniform(&[100]);

        let data1: Vec<f32> = samples1.into_data().to_vec().unwrap();
        let data2: Vec<f32> = samples2.into_data().to_vec().unwrap();

        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Same seed should produce same sequence"
            );
        }
    }

    #[test]
    fn test_different_with_different_seed() {
        let device = Default::default();

        let mut rng1 = GpuRng::<TestBackend>::new(42, 16, &device);
        let samples1 = rng1.uniform(&[100]);

        let mut rng2 = GpuRng::<TestBackend>::new(43, 16, &device);
        let samples2 = rng2.uniform(&[100]);

        let data1: Vec<f32> = samples1.into_data().to_vec().unwrap();
        let data2: Vec<f32> = samples2.into_data().to_vec().unwrap();

        // At least some samples should differ
        let different_count = data1
            .iter()
            .zip(data2.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-10)
            .count();

        assert!(
            different_count > 90,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_reseed() {
        let device = Default::default();

        let mut rng = GpuRng::<TestBackend>::new(42, 16, &device);
        let samples1 = rng.uniform(&[100]);

        rng.reseed(42);
        let samples2 = rng.uniform(&[100]);

        let data1: Vec<f32> = samples1.into_data().to_vec().unwrap();
        let data2: Vec<f32> = samples2.into_data().to_vec().unwrap();

        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!((a - b).abs() < 1e-10, "Reseed should produce same sequence");
        }
    }

    #[test]
    fn test_clone() {
        let device = Default::default();
        let mut rng1 = GpuRng::<TestBackend>::new(42, 16, &device);

        // Advance the RNG
        let _ = rng1.uniform(&[50]);

        // Clone after advancing
        let mut rng2 = rng1.clone();

        // Both should produce the same next values
        let samples1 = rng1.uniform(&[100]);
        let samples2 = rng2.uniform(&[100]);

        let data1: Vec<f32> = samples1.into_data().to_vec().unwrap();
        let data2: Vec<f32> = samples2.into_data().to_vec().unwrap();

        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Cloned RNG should produce same sequence"
            );
        }
    }

    #[test]
    fn test_gamma_shape_ge_1() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        // Gamma(2, 1) should have mean 2 and variance 2
        let shape = 2.0;
        let samples = rng.gamma(shape, 5000);
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();

        // All samples should be positive
        for &sample in &data {
            assert!(
                sample > 0.0,
                "Gamma samples must be positive, got {}",
                sample
            );
        }

        // Mean should be approximately α = 2
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            (mean - shape).abs() < 0.2,
            "Gamma({}) mean {} should be close to {}",
            shape,
            mean,
            shape
        );

        // Variance should be approximately α = 2
        let variance: f32 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        assert!(
            (variance - shape).abs() < 0.5,
            "Gamma({}) variance {} should be close to {}",
            shape,
            variance,
            shape
        );
    }

    #[test]
    fn test_gamma_shape_lt_1() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        // Gamma(0.5, 1) should have mean 0.5 and variance 0.5
        let shape = 0.5;
        let samples = rng.gamma(shape, 5000);
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();

        // All samples should be positive
        for &sample in &data {
            assert!(sample > 0.0, "Gamma samples must be positive");
        }

        // Mean should be approximately α = 0.5
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            (mean - shape).abs() < 0.1,
            "Gamma({}) mean {} should be close to {}",
            shape,
            mean,
            shape
        );
    }

    #[test]
    fn test_gamma_exponential_equivalence() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        // Gamma(1, 1) = Exponential(1), mean = 1, variance = 1
        let samples = rng.gamma(1.0, 5000);
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            (mean - 1.0).abs() < 0.1,
            "Gamma(1) should equal Exponential(1), mean {} should be ~1",
            mean
        );
    }

    #[test]
    fn test_dirichlet_simplex() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        let alpha = [1.0, 2.0, 3.0];
        let sample = rng.dirichlet(&alpha);
        let data: Vec<f32> = sample.into_data().to_vec().unwrap();

        // All components should be positive
        for &x in &data {
            assert!(x > 0.0, "Dirichlet components must be positive, got {}", x);
        }

        // Components should sum to 1
        let sum: f32 = data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Dirichlet samples should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_dirichlet_batch() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        let alpha = [2.0, 2.0, 2.0];
        let n = 100;
        let samples = rng.dirichlet_batch(&alpha, n);

        let [rows, cols] = samples.dims();
        assert_eq!(rows, n, "Should have {} samples", n);
        assert_eq!(cols, alpha.len(), "Should have {} categories", alpha.len());

        // Each row should sum to 1
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();
        for i in 0..n {
            let row_sum: f32 = data[i * cols..(i + 1) * cols].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "Dirichlet sample {} should sum to 1, got {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_dirichlet_mean() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        // For Dirichlet(α), mean_k = α_k / Σα
        let alpha = [1.0, 2.0, 3.0];
        let alpha_sum: f32 = alpha.iter().sum();
        let expected_means: Vec<f32> = alpha.iter().map(|&a| a / alpha_sum).collect();

        let n = 5000;
        let samples = rng.dirichlet_batch(&alpha, n);
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();

        // Compute sample means
        let k = alpha.len();
        for j in 0..k {
            let col_sum: f32 = (0..n).map(|i| data[i * k + j]).sum();
            let col_mean = col_sum / n as f32;
            assert!(
                (col_mean - expected_means[j]).abs() < 0.05,
                "Dirichlet mean[{}] {} should be close to {}",
                j,
                col_mean,
                expected_means[j]
            );
        }
    }

    #[test]
    fn test_dirichlet_symmetric() {
        let device = Default::default();
        let mut rng = GpuRng::<TestBackend>::new(42, 64, &device);

        // Symmetric Dirichlet should have equal means
        let alpha = [5.0, 5.0, 5.0, 5.0];
        let n = 2000;
        let samples = rng.dirichlet_batch(&alpha, n);
        let data: Vec<f32> = samples.into_data().to_vec().unwrap();

        let k = alpha.len();
        let expected_mean = 1.0 / k as f32;

        for j in 0..k {
            let col_sum: f32 = (0..n).map(|i| data[i * k + j]).sum();
            let col_mean = col_sum / n as f32;
            assert!(
                (col_mean - expected_mean).abs() < 0.05,
                "Symmetric Dirichlet mean[{}] {} should be ~{}",
                j,
                col_mean,
                expected_mean
            );
        }
    }
}
