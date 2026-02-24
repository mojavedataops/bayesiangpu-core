//! Benchmark: GPU REDUCE kernels vs CPU for log_prob computation
//!
//! Compares three paths:
//!   1. GPU (alloc)      — creates 4 GPU buffers per call (old path)
//!   2. GPU (persistent) — reuses pre-allocated buffers, only writes params (new path)
//!   3. CPU (SIMD)       — auto-vectorized loop on host
//!
//! Run with:
//!   cargo bench -p bayesian-wasm --features sync-gpu --bench gpu_vs_cpu

use std::sync::Arc;
use std::time::{Duration, Instant};

use bayesian_wasm::gpu::sync::GpuContextSync;

/// Number of timed iterations per benchmark
const ITERS: u32 = 200;
/// Warmup iterations (not timed)
const WARMUP: u32 = 10;

/// CPU reference: compute Normal log_prob sum
fn cpu_normal_logp_sum(x_values: &[f32], mu: f32, sigma: f32) -> f32 {
    let inv_sigma2 = 1.0 / (sigma * sigma);
    let log_norm = -0.5 * (2.0 * std::f32::consts::PI * sigma * sigma).ln();
    x_values
        .iter()
        .map(|&x| log_norm - 0.5 * (x - mu) * (x - mu) * inv_sigma2)
        .sum()
}

/// CPU reference: compute Normal grad_log_prob sum (d/d_mu)
fn cpu_normal_grad_sum(x_values: &[f32], mu: f32, sigma: f32) -> f32 {
    let inv_sigma2 = 1.0 / (sigma * sigma);
    x_values.iter().map(|&x| (x - mu) * inv_sigma2).sum()
}

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

fn ns_per_call(d: Duration) -> f64 {
    d.as_nanos() as f64 / ITERS as f64
}

fn main() {
    let ctx = match GpuContextSync::new() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("GPU not available: {}. Cannot benchmark.", e);
            return;
        }
    };

    println!();
    println!("=== BayesianGPU: GPU vs CPU Benchmark ===");
    println!(
        "  {} iterations per measurement (+ {} warmup)",
        ITERS, WARMUP
    );
    println!("  Distribution: Normal(mu=2.0, sigma=1.0)");
    println!();

    // --- Combined logp + grad (the actual NUTS hot path) ---
    println!("--- logp + grad per call (NUTS hot path) ---");
    println!(
        "{:<10} {:>16} {:>16} {:>12} {:>10} {:>10}",
        "N_obs", "GPU alloc (us)", "GPU persist (us)", "CPU (us)", "Persist/A", "Persist/C"
    );
    println!("{}", "-".repeat(78));

    for &n_obs in &[256, 500, 1_000, 5_000, 10_000, 50_000, 100_000] {
        let x_values: Vec<f32> = (0..n_obs).map(|i| 2.0 + (i as f32 * 0.001)).collect();
        let mu = 2.0f32;
        let sigma = 1.0f32;

        // Create persistent buffers (one-time cost, not measured)
        let buffers = ctx.create_persistent_buffers(&x_values, 16);

        let ctx_ref = &ctx;
        let x_ref = &x_values;
        let buf_ref = &buffers;

        // 1. GPU allocating path (old: creates 4 buffers per call)
        let (gpu_alloc_time, _) = bench_fn(|| {
            let lp = ctx_ref.run_normal_reduce(x_ref, mu, sigma).unwrap();
            let gr = ctx_ref.run_normal_grad_reduce(x_ref, mu, sigma).unwrap();
            (lp, gr)
        });
        let alloc_ns = ns_per_call(gpu_alloc_time);

        // 2. GPU persistent path (new: reuses buffers, only writes params)
        let (gpu_persist_time, _) = bench_fn(|| {
            let lp = ctx_ref
                .run_normal_reduce_persistent(buf_ref, mu, sigma)
                .unwrap();
            let gr = ctx_ref
                .run_normal_grad_reduce_persistent(buf_ref, mu, sigma)
                .unwrap();
            (lp, gr)
        });
        let persist_ns = ns_per_call(gpu_persist_time);

        // 3. CPU SIMD path
        let (cpu_time, _) = bench_fn(|| {
            let lp = cpu_normal_logp_sum(x_ref, mu, sigma);
            let gr = cpu_normal_grad_sum(x_ref, mu, sigma);
            (lp, gr)
        });
        let cpu_ns = ns_per_call(cpu_time);

        let speedup_vs_alloc = if persist_ns > 0.0 {
            alloc_ns / persist_ns
        } else {
            0.0
        };
        let speedup_vs_cpu = if cpu_ns > 0.0 {
            persist_ns / cpu_ns
        } else {
            f64::INFINITY
        };

        println!(
            "{:<10} {:>14.0}us {:>14.0}us {:>10.0}us {:>9.1}x {:>9.0}x",
            n_obs,
            alloc_ns / 1000.0,
            persist_ns / 1000.0,
            cpu_ns / 1000.0,
            speedup_vs_alloc,
            speedup_vs_cpu
        );
    }

    // --- Numerical consistency check ---
    println!();
    println!("--- Numerical consistency ---");
    let x_values: Vec<f32> = (0..10_000).map(|i| 2.0 + (i as f32 * 0.001)).collect();
    let buffers = ctx.create_persistent_buffers(&x_values, 16);
    let mu = 2.0f32;
    let sigma = 1.0f32;

    let gpu_alloc_lp = ctx.run_normal_reduce(&x_values, mu, sigma).unwrap();
    let gpu_persist_lp = ctx
        .run_normal_reduce_persistent(&buffers, mu, sigma)
        .unwrap();
    let cpu_lp = cpu_normal_logp_sum(&x_values, mu, sigma);

    let gpu_alloc_gr = ctx.run_normal_grad_reduce(&x_values, mu, sigma).unwrap();
    let gpu_persist_gr = ctx
        .run_normal_grad_reduce_persistent(&buffers, mu, sigma)
        .unwrap();
    let cpu_gr = cpu_normal_grad_sum(&x_values, mu, sigma);

    println!(
        "  logp:  alloc={:.4}  persist={:.4}  cpu={:.4}",
        gpu_alloc_lp, gpu_persist_lp, cpu_lp
    );
    println!(
        "  grad:  alloc={:.4}  persist={:.4}  cpu={:.4}",
        gpu_alloc_gr, gpu_persist_gr, cpu_gr
    );
    println!();
}
