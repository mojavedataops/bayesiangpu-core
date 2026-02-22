//! Benchmarks for MCMC samplers
//!
//! Run with: cargo bench -p bayesian-sampler

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use bayesian_rng::GpuRng;
use bayesian_sampler::{
    hmc::{HmcConfig, HmcSampler},
    leapfrog::leapfrog_step,
    model::BayesianModel,
};

type TestBackend = Autodiff<NdArray<f32>>;

/// Simple quadratic model for benchmarking
/// log p(x) = -0.5 * x^T * x (standard normal)
#[derive(Clone)]
struct QuadraticModel {
    dim: usize,
}

impl QuadraticModel {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl BayesianModel<TestBackend> for QuadraticModel {
    fn dim(&self) -> usize {
        self.dim
    }

    fn log_prob(&self, params: &Tensor<TestBackend, 1>) -> Tensor<TestBackend, 1> {
        // log p(x) = -0.5 * ||x||^2
        let squared = params.clone().powf_scalar(2.0);
        squared.mul_scalar(-0.5).sum().reshape([1])
    }

    fn param_names(&self) -> Vec<String> {
        (0..self.dim).map(|i| format!("x[{}]", i)).collect()
    }
}

fn bench_leapfrog_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("leapfrog_step");

    for dim in [1, 10, 100].iter() {
        let model = QuadraticModel::new(*dim);
        let device = NdArrayDevice::default();

        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let position = Tensor::<TestBackend, 1>::zeros([dim], &device);
            let momentum = Tensor::<TestBackend, 1>::ones([dim], &device);
            let step_size = 0.1;

            b.iter(|| {
                leapfrog_step(
                    black_box(&model),
                    black_box(position.clone()),
                    black_box(momentum.clone()),
                    black_box(step_size),
                )
            });
        });
    }

    group.finish();
}

fn bench_hmc_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmc_sample");
    group.sample_size(10); // Fewer samples due to long runtime

    for dim in [1, 10].iter() {
        let device = NdArrayDevice::default();

        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |b, &dim| {
            let model = QuadraticModel::new(dim);
            let config = HmcConfig {
                step_size: 0.1,
                num_leapfrog_steps: 10,
                num_samples: 100,
                num_warmup: 50,
            };
            let rng = GpuRng::<TestBackend>::new(42, dim, &device);
            let init_params = Tensor::<TestBackend, 1>::zeros([dim], &device);
            let mut sampler = HmcSampler::new(model, config, rng);

            b.iter(|| sampler.sample(black_box(init_params.clone())));
        });
    }

    group.finish();
}

fn bench_leapfrog_trajectory(c: &mut Criterion) {
    let mut group = c.benchmark_group("leapfrog_trajectory");

    for num_steps in [5, 10, 20, 50].iter() {
        let model = QuadraticModel::new(10);
        let device = NdArrayDevice::default();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_steps),
            num_steps,
            |b, &num_steps| {
                let position = Tensor::<TestBackend, 1>::zeros([10], &device);
                let momentum = Tensor::<TestBackend, 1>::ones([10], &device);
                let step_size = 0.1;

                b.iter(|| {
                    let mut pos = position.clone();
                    let mut mom = momentum.clone();

                    for _ in 0..num_steps {
                        let result = leapfrog_step(&model, pos, mom, step_size);
                        pos = result.position;
                        mom = result.momentum;
                    }

                    (pos, mom)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_leapfrog_step,
    bench_hmc_sampling,
    bench_leapfrog_trajectory
);
criterion_main!(benches);
