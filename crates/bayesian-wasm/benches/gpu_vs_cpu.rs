//! Benchmark: GPU REDUCE kernels vs CPU for log_prob + gradient computation
//!
//! Compares four paths across 9 distributions (Normal, HalfNormal, Exponential,
//! Beta, Gamma, InverseGamma, StudentT, Cauchy, LogNormal):
//!
//!   1. GPU (alloc)      -- creates GPU buffers per call (baseline)
//!   2. GPU (persistent) -- reuses pre-allocated buffers, writes params in-place
//!   3. GPU (fused)      -- single command encoder + submit for logp and grad
//!   4. CPU (SIMD)       -- auto-vectorized f32 loop on host
//!
//! Run with:
//!   cargo bench -p bayesian-wasm --features sync-gpu --bench gpu_vs_cpu

use std::sync::Arc;
use std::time::{Duration, Instant};

use bayesian_wasm::gpu::sync::GpuContextSync;
use bayesian_wasm::gpu::PersistentGpuBuffers;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DATA_SIZES: &[usize] = &[256, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];
const ITERS: u32 = 200;
const WARMUP: u32 = 10;

// ---------------------------------------------------------------------------
// Stirling lgamma approximation
// ---------------------------------------------------------------------------

fn lgammaf_approx(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * (2.0 * std::f64::consts::PI).ln() + (x - 0.5) * x.ln() - x) as f32
}

fn lnbeta_approx(a: f32, b: f32) -> f32 {
    lgammaf_approx(a) + lgammaf_approx(b) - lgammaf_approx(a + b)
}

// ---------------------------------------------------------------------------
// CPU reference implementations (logp_sum, grad_sum) per distribution
// ---------------------------------------------------------------------------

fn cpu_normal_logp_sum(x: &[f32], mu: f32, sigma: f32) -> f32 {
    let inv_s2 = 1.0 / (sigma * sigma);
    let log_norm = -0.5 * (2.0 * std::f32::consts::PI * sigma * sigma).ln();
    x.iter()
        .map(|&v| log_norm - 0.5 * (v - mu) * (v - mu) * inv_s2)
        .sum()
}

fn cpu_normal_grad_sum(x: &[f32], mu: f32, sigma: f32) -> f32 {
    let inv_s2 = 1.0 / (sigma * sigma);
    x.iter().map(|&v| (v - mu) * inv_s2).sum()
}

fn cpu_half_normal_logp_sum(x: &[f32], sigma: f32) -> f32 {
    let inv_s2 = 1.0 / (sigma * sigma);
    let log_norm = (2.0f32).ln() - 0.5 * (2.0 * std::f32::consts::PI * sigma * sigma).ln();
    x.iter().map(|&v| log_norm - 0.5 * v * v * inv_s2).sum()
}

fn cpu_half_normal_grad_sum(x: &[f32], sigma: f32) -> f32 {
    let inv_s2 = 1.0 / (sigma * sigma);
    x.iter().map(|&v| -v * inv_s2).sum()
}

fn cpu_exponential_logp_sum(x: &[f32], lambda: f32) -> f32 {
    let ln_lam = lambda.ln();
    x.iter().map(|&v| ln_lam - lambda * v).sum()
}

fn cpu_exponential_grad_sum(x: &[f32], lambda: f32) -> f32 {
    let inv_lam = 1.0 / lambda;
    x.iter().map(|&v| inv_lam - v).sum()
}

fn cpu_beta_logp_sum(x: &[f32], alpha: f32, beta: f32) -> f32 {
    let norm = lnbeta_approx(alpha, beta);
    x.iter()
        .map(|&v| (alpha - 1.0) * v.ln() + (beta - 1.0) * (1.0 - v).ln() - norm)
        .sum()
}

fn cpu_beta_grad_sum(x: &[f32], _alpha: f32, _beta: f32) -> f32 {
    // d/dalpha: sum(ln(x_i)) - n*(digamma(alpha) - digamma(alpha+beta))
    // Approximate: just sum(ln(x_i)) for benchmark purposes
    x.iter().map(|&v| v.ln()).sum()
}

fn cpu_gamma_logp_sum(x: &[f32], alpha: f32, beta: f32) -> f32 {
    let norm = alpha * beta.ln() - lgammaf_approx(alpha);
    x.iter()
        .map(|&v| (alpha - 1.0) * v.ln() - beta * v + norm)
        .sum()
}

fn cpu_gamma_grad_sum(x: &[f32], _alpha: f32, _beta: f32) -> f32 {
    // d/dalpha: sum(ln(x_i)) + n*ln(beta) - n*digamma(alpha), approximate as sum(ln(x_i))
    x.iter().map(|&v| v.ln()).sum()
}

fn cpu_inverse_gamma_logp_sum(x: &[f32], alpha: f32, beta: f32) -> f32 {
    let norm = alpha * beta.ln() - lgammaf_approx(alpha);
    x.iter()
        .map(|&v| -(alpha + 1.0) * v.ln() - beta / v + norm)
        .sum()
}

fn cpu_inverse_gamma_grad_sum(x: &[f32], _alpha: f32, _beta: f32) -> f32 {
    // d/dalpha approximation: sum(-ln(x_i)) + n*ln(beta) - n*digamma(alpha)
    x.iter().map(|&v| -(v.ln())).sum()
}

fn cpu_student_t_logp_sum(x: &[f32], loc: f32, scale: f32, nu: f32) -> f32 {
    let half_nu_plus1 = 0.5 * (nu + 1.0);
    let norm = lgammaf_approx(half_nu_plus1)
        - lgammaf_approx(0.5 * nu)
        - 0.5 * (nu * std::f32::consts::PI * scale * scale).ln();
    x.iter()
        .map(|&v| {
            let z = (v - loc) / scale;
            norm - half_nu_plus1 * (1.0 + z * z / nu).ln()
        })
        .sum()
}

fn cpu_student_t_grad_sum(x: &[f32], loc: f32, scale: f32, nu: f32) -> f32 {
    let s2 = scale * scale;
    x.iter()
        .map(|&v| {
            let d = v - loc;
            (nu + 1.0) * d / (nu * s2 + d * d)
        })
        .sum()
}

fn cpu_cauchy_logp_sum(x: &[f32], loc: f32, scale: f32) -> f32 {
    let log_norm = -(std::f32::consts::PI * scale).ln();
    let s2 = scale * scale;
    x.iter()
        .map(|&v| {
            let d = v - loc;
            log_norm - (1.0 + d * d / s2).ln()
        })
        .sum()
}

fn cpu_cauchy_grad_sum(x: &[f32], loc: f32, scale: f32) -> f32 {
    let s2 = scale * scale;
    x.iter()
        .map(|&v| {
            let d = v - loc;
            2.0 * d / (s2 + d * d)
        })
        .sum()
}

fn cpu_lognormal_logp_sum(x: &[f32], mu: f32, sigma: f32) -> f32 {
    let inv_s2 = 1.0 / (sigma * sigma);
    let log_norm = -0.5 * (2.0 * std::f32::consts::PI * sigma * sigma).ln();
    x.iter()
        .map(|&v| {
            let lnv = v.ln();
            -lnv + log_norm - 0.5 * (lnv - mu) * (lnv - mu) * inv_s2
        })
        .sum()
}

fn cpu_lognormal_grad_sum(x: &[f32], mu: f32, sigma: f32) -> f32 {
    let inv_s2 = 1.0 / (sigma * sigma);
    x.iter().map(|&v| (v.ln() - mu) * inv_s2).sum()
}

// ---------------------------------------------------------------------------
// Distribution benchmark descriptor
// ---------------------------------------------------------------------------

/// Holds all function pointers needed to benchmark one distribution across
/// the four paths (GPU alloc, GPU persistent, GPU fused, CPU).
struct DistBench {
    name: &'static str,
    header: &'static str,
    generate_data: fn(usize) -> Vec<f32>,
    gpu_alloc: fn(&GpuContextSync, &[f32]) -> (f32, f32),
    gpu_persistent: fn(&GpuContextSync, &PersistentGpuBuffers) -> (f32, f32),
    gpu_fused: fn(&GpuContextSync, &PersistentGpuBuffers) -> (f32, f32),
    cpu: fn(&[f32]) -> (f32, f32),
}

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

fn gen_normal(n: usize) -> Vec<f32> {
    (0..n).map(|i| 2.0 + (i as f32 * 0.001)).collect()
}

fn gen_half_normal(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 + (i as f32 * 0.001)).collect()
}

fn gen_exponential(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 + (i as f32 * 0.001)).collect()
}

fn gen_beta(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.01 + 0.98 * (i as f32) / (n as f32))
        .collect()
}

fn gen_gamma(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 + (i as f32 * 0.001)).collect()
}

fn gen_inverse_gamma(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + (i as f32 * 0.001)).collect()
}

fn gen_student_t(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.001) - 0.5 * n as f32 * 0.001)
        .collect()
}

fn gen_cauchy(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.001) - 0.5 * n as f32 * 0.001)
        .collect()
}

fn gen_lognormal(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + (i as f32 * 0.001)).collect()
}

// ---------------------------------------------------------------------------
// Build the table of 9 distributions
// ---------------------------------------------------------------------------

fn build_dist_benches() -> Vec<DistBench> {
    vec![
        // ----- Normal(mu=2.0, sigma=1.0) -----
        DistBench {
            name: "Normal",
            header: "Normal(mu=2.0, sigma=1.0)",
            generate_data: gen_normal,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_normal_reduce(x, 2.0, 1.0).unwrap();
                let gr = ctx.run_normal_grad_reduce(x, 2.0, 1.0).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx.run_normal_reduce_persistent(buf, 2.0, 1.0).unwrap();
                let gr = ctx
                    .run_normal_grad_reduce_persistent(buf, 2.0, 1.0)
                    .unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx.run_normal_fused_persistent(buf, 2.0, 1.0).unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_normal_logp_sum(x, 2.0, 1.0);
                let gr = cpu_normal_grad_sum(x, 2.0, 1.0);
                (lp, gr)
            },
        },
        // ----- HalfNormal(sigma=1.0) -----
        DistBench {
            name: "HalfNormal",
            header: "HalfNormal(sigma=1.0)",
            generate_data: gen_half_normal,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_half_normal_reduce(x, 1.0).unwrap();
                let gr = ctx.run_half_normal_grad_reduce(x, 1.0).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx.run_half_normal_reduce_persistent(buf, 1.0).unwrap();
                let gr = ctx
                    .run_half_normal_grad_reduce_persistent(buf, 1.0)
                    .unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx.run_half_normal_fused_persistent(buf, 1.0).unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_half_normal_logp_sum(x, 1.0);
                let gr = cpu_half_normal_grad_sum(x, 1.0);
                (lp, gr)
            },
        },
        // ----- Exponential(lambda=1.5) -----
        DistBench {
            name: "Exponential",
            header: "Exponential(lambda=1.5)",
            generate_data: gen_exponential,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_exponential_reduce(x, 1.5).unwrap();
                let gr = ctx.run_exponential_grad_reduce(x, 1.5).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx.run_exponential_reduce_persistent(buf, 1.5).unwrap();
                let gr = ctx
                    .run_exponential_grad_reduce_persistent(buf, 1.5)
                    .unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx.run_exponential_fused_persistent(buf, 1.5).unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_exponential_logp_sum(x, 1.5);
                let gr = cpu_exponential_grad_sum(x, 1.5);
                (lp, gr)
            },
        },
        // ----- Beta(alpha=2.0, beta=5.0) -----
        DistBench {
            name: "Beta",
            header: "Beta(alpha=2.0, beta=5.0)",
            generate_data: gen_beta,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_beta_reduce(x, 2.0, 5.0).unwrap();
                let gr = ctx.run_beta_grad_reduce(x, 2.0, 5.0).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx.run_beta_reduce_persistent(buf, 2.0, 5.0).unwrap();
                let gr = ctx.run_beta_grad_reduce_persistent(buf, 2.0, 5.0).unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx.run_beta_fused_persistent(buf, 2.0, 5.0).unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_beta_logp_sum(x, 2.0, 5.0);
                let gr = cpu_beta_grad_sum(x, 2.0, 5.0);
                (lp, gr)
            },
        },
        // ----- Gamma(alpha=2.0, beta=1.0) -----
        DistBench {
            name: "Gamma",
            header: "Gamma(alpha=2.0, beta=1.0)",
            generate_data: gen_gamma,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_gamma_reduce(x, 2.0, 1.0).unwrap();
                let gr = ctx.run_gamma_grad_reduce(x, 2.0, 1.0).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx.run_gamma_reduce_persistent(buf, 2.0, 1.0).unwrap();
                let gr = ctx.run_gamma_grad_reduce_persistent(buf, 2.0, 1.0).unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx.run_gamma_fused_persistent(buf, 2.0, 1.0).unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_gamma_logp_sum(x, 2.0, 1.0);
                let gr = cpu_gamma_grad_sum(x, 2.0, 1.0);
                (lp, gr)
            },
        },
        // ----- InverseGamma(alpha=3.0, beta=2.0) -----
        DistBench {
            name: "InverseGamma",
            header: "InverseGamma(alpha=3.0, beta=2.0)",
            generate_data: gen_inverse_gamma,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_inverse_gamma_reduce(x, 3.0, 2.0).unwrap();
                let gr = ctx.run_inverse_gamma_grad_reduce(x, 3.0, 2.0).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx
                    .run_inverse_gamma_reduce_persistent(buf, 3.0, 2.0)
                    .unwrap();
                let gr = ctx
                    .run_inverse_gamma_grad_reduce_persistent(buf, 3.0, 2.0)
                    .unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx
                    .run_inverse_gamma_fused_persistent(buf, 3.0, 2.0)
                    .unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_inverse_gamma_logp_sum(x, 3.0, 2.0);
                let gr = cpu_inverse_gamma_grad_sum(x, 3.0, 2.0);
                (lp, gr)
            },
        },
        // ----- StudentT(loc=0.0, scale=1.0, nu=4.0) -----
        DistBench {
            name: "StudentT",
            header: "StudentT(loc=0.0, scale=1.0, nu=4.0)",
            generate_data: gen_student_t,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_student_t_reduce(x, 0.0, 1.0, 4.0).unwrap();
                let gr = ctx.run_student_t_grad_reduce(x, 0.0, 1.0, 4.0).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx
                    .run_student_t_reduce_persistent(buf, 0.0, 1.0, 4.0)
                    .unwrap();
                let gr = ctx
                    .run_student_t_grad_reduce_persistent(buf, 0.0, 1.0, 4.0)
                    .unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx
                    .run_student_t_fused_persistent(buf, 0.0, 1.0, 4.0)
                    .unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_student_t_logp_sum(x, 0.0, 1.0, 4.0);
                let gr = cpu_student_t_grad_sum(x, 0.0, 1.0, 4.0);
                (lp, gr)
            },
        },
        // ----- Cauchy(loc=0.0, scale=1.0) -----
        DistBench {
            name: "Cauchy",
            header: "Cauchy(loc=0.0, scale=1.0)",
            generate_data: gen_cauchy,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_cauchy_reduce(x, 0.0, 1.0).unwrap();
                let gr = ctx.run_cauchy_grad_reduce(x, 0.0, 1.0).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx.run_cauchy_reduce_persistent(buf, 0.0, 1.0).unwrap();
                let gr = ctx
                    .run_cauchy_grad_reduce_persistent(buf, 0.0, 1.0)
                    .unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx.run_cauchy_fused_persistent(buf, 0.0, 1.0).unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_cauchy_logp_sum(x, 0.0, 1.0);
                let gr = cpu_cauchy_grad_sum(x, 0.0, 1.0);
                (lp, gr)
            },
        },
        // ----- LogNormal(mu=0.0, sigma=1.0) -----
        DistBench {
            name: "LogNormal",
            header: "LogNormal(mu=0.0, sigma=1.0)",
            generate_data: gen_lognormal,
            gpu_alloc: |ctx, x| {
                let lp = ctx.run_lognormal_reduce(x, 0.0, 1.0).unwrap();
                let gr = ctx.run_lognormal_grad_reduce(x, 0.0, 1.0).unwrap();
                (lp, gr)
            },
            gpu_persistent: |ctx, buf| {
                let lp = ctx.run_lognormal_reduce_persistent(buf, 0.0, 1.0).unwrap();
                let gr = ctx
                    .run_lognormal_grad_reduce_persistent(buf, 0.0, 1.0)
                    .unwrap();
                (lp, gr)
            },
            gpu_fused: |ctx, buf| {
                let r = ctx.run_lognormal_fused_persistent(buf, 0.0, 1.0).unwrap();
                (r.total_log_prob, r.total_grad)
            },
            cpu: |x| {
                let lp = cpu_lognormal_logp_sum(x, 0.0, 1.0);
                let gr = cpu_lognormal_grad_sum(x, 0.0, 1.0);
                (lp, gr)
            },
        },
    ]
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn bench_fn<F: FnMut() -> T, T>(mut f: F) -> (Duration, T) {
    for _ in 0..WARMUP {
        let _ = f();
    }
    let start = Instant::now();
    let mut result = f();
    for _ in 1..ITERS {
        result = f();
    }
    (start.elapsed(), result)
}

fn us_per_call(d: Duration) -> f64 {
    d.as_nanos() as f64 / ITERS as f64 / 1000.0
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let ctx = match GpuContextSync::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("GPU not available: {}. Cannot benchmark.", e);
            return;
        }
    };

    let dists = build_dist_benches();

    println!();
    println!("=== BayesianGPU: Multi-Distribution GPU vs CPU Benchmark ===");
    println!(
        "  {} iterations per measurement (+ {} warmup)",
        ITERS, WARMUP
    );
    println!(
        "  {} distributions, {} data sizes",
        dists.len(),
        DATA_SIZES.len()
    );
    println!();

    // Max params struct size across all distributions (StudentT is 16 bytes)
    let max_params_size: u64 = 16;

    for dist in &dists {
        println!();
        println!("=== {} ===", dist.header);
        println!(
            "{:<12} {:>16} {:>16} {:>16} {:>12} {:>11} {:>11}",
            "N_obs",
            "GPU alloc (us)",
            "GPU persist (us)",
            "GPU fused (us)",
            "CPU (us)",
            "Fused/Alloc",
            "Fused/CPU"
        );
        println!("{}", "-".repeat(98));

        for &n_obs in DATA_SIZES {
            let x_values = (dist.generate_data)(n_obs);

            // Create persistent buffers (one-time cost, not measured)
            let buffers = ctx.create_persistent_buffers(&x_values, max_params_size);

            let ctx_ref = &*ctx;
            let x_ref = &x_values;
            let buf_ref = &buffers;

            // 1. GPU allocating path
            let (gpu_alloc_time, _) = bench_fn(|| (dist.gpu_alloc)(ctx_ref, x_ref));
            let alloc_us = us_per_call(gpu_alloc_time);

            // 2. GPU persistent path
            let (gpu_persist_time, _) = bench_fn(|| (dist.gpu_persistent)(ctx_ref, buf_ref));
            let persist_us = us_per_call(gpu_persist_time);

            // 3. GPU fused path
            let (gpu_fused_time, _) = bench_fn(|| (dist.gpu_fused)(ctx_ref, buf_ref));
            let fused_us = us_per_call(gpu_fused_time);

            // 4. CPU SIMD path
            let (cpu_time, _) = bench_fn(|| (dist.cpu)(x_ref));
            let cpu_us = us_per_call(cpu_time);

            let fused_vs_alloc = if fused_us > 0.0 {
                alloc_us / fused_us
            } else {
                0.0
            };
            let fused_vs_cpu = if cpu_us > 0.0 {
                fused_us / cpu_us
            } else {
                f64::INFINITY
            };

            println!(
                "{:<12} {:>14.0}us {:>14.0}us {:>14.0}us {:>10.0}us {:>10.2}x {:>10.2}x",
                format_n(n_obs),
                alloc_us,
                persist_us,
                fused_us,
                cpu_us,
                fused_vs_alloc,
                fused_vs_cpu,
            );
        }
    }

    // -------------------------------------------------------------------
    // Numerical consistency check
    // -------------------------------------------------------------------
    println!();
    println!("=== Numerical Consistency (N=10,000) ===");
    println!();

    let consistency_n = 10_000;

    for dist in &dists {
        let x_values = (dist.generate_data)(consistency_n);
        let buffers = ctx.create_persistent_buffers(&x_values, max_params_size);

        let (alloc_lp, alloc_gr) = (dist.gpu_alloc)(&ctx, &x_values);
        let (persist_lp, persist_gr) = (dist.gpu_persistent)(&ctx, &buffers);
        let (fused_lp, fused_gr) = (dist.gpu_fused)(&ctx, &buffers);
        let (cpu_lp, cpu_gr) = (dist.cpu)(&x_values);

        println!("--- {} ---", dist.name);
        println!(
            "  logp:  alloc={:.4}  persist={:.4}  fused={:.4}  cpu={:.4}",
            alloc_lp, persist_lp, fused_lp, cpu_lp,
        );
        println!(
            "  grad:  alloc={:.4}  persist={:.4}  fused={:.4}  cpu={:.4}",
            alloc_gr, persist_gr, fused_gr, cpu_gr,
        );

        // Check GPU paths agree with each other (allow f32 tolerance)
        let lp_ok = (alloc_lp - persist_lp).abs() < 1.0 && (alloc_lp - fused_lp).abs() < 1.0;
        let gr_ok = (alloc_gr - persist_gr).abs() < 1.0 && (alloc_gr - fused_gr).abs() < 1.0;

        if lp_ok && gr_ok {
            println!("  [PASS] GPU paths agree within tolerance");
        } else {
            println!("  [WARN] GPU paths diverge beyond tolerance");
        }
        println!();
    }
}

/// Format large numbers with underscores for readability
fn format_n(n: usize) -> String {
    if n >= 10_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}
