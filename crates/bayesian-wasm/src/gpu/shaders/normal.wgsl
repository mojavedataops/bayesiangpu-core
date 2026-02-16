// Normal distribution log_prob and gradient kernel
//
// For Normal(mu, sigma) at point x:
// log_prob = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²
// grad     = -(x - μ) / σ²

struct Params {
    x: f32,
    mu: f32,
    sigma: f32,
    _padding: f32,
}

struct Output {
    log_prob: f32,
    grad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: Output;

// Constants
const PI: f32 = 3.14159265358979323846;
const LOG_2PI: f32 = 1.8378770664093453; // log(2π)

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = params.x;
    let mu = params.mu;
    let sigma = params.sigma;

    // Compute z = (x - mu) / sigma
    let z = (x - mu) / sigma;

    // log_prob = -0.5 * log(2π) - log(σ) - 0.5 * z²
    let log_prob = -0.5 * LOG_2PI - log(sigma) - 0.5 * z * z;

    // grad = ∂log_prob/∂x = -z / sigma = -(x - mu) / sigma²
    let grad = -z / sigma;

    output.log_prob = log_prob;
    output.grad = grad;
}
