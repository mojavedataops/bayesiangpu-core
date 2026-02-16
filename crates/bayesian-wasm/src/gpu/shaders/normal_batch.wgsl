// Normal distribution BATCHED log_prob and gradient kernel
//
// Processes N observations in parallel with shared mu, sigma.
// Each workgroup thread handles one element.
//
// For Normal(mu, sigma) at points x[i]:
// log_prob[i] = -0.5 * log(2π) - log(σ) - 0.5 * ((x[i] - μ) / σ)²
// grad[i]     = -(x[i] - μ) / σ²

struct Params {
    mu: f32,
    sigma: f32,
    count: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> log_probs: array<f32>;
@group(0) @binding(3) var<storage, read_write> grads: array<f32>;

// Constants
const LOG_2PI: f32 = 1.8378770664093453;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Bounds check
    if (idx >= params.count) {
        return;
    }

    let x = x_values[idx];
    let mu = params.mu;
    let sigma = params.sigma;

    // Compute z = (x - mu) / sigma
    let z = (x - mu) / sigma;

    // log_prob = -0.5 * log(2π) - log(σ) - 0.5 * z²
    let log_prob = -0.5 * LOG_2PI - log(sigma) - 0.5 * z * z;

    // grad = -(x - mu) / σ²
    let grad = -z / sigma;

    log_probs[idx] = log_prob;
    grads[idx] = grad;
}
