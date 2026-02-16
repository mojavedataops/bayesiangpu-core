// HalfNormal distribution log_prob and gradient kernel
//
// For HalfNormal(sigma) at point x >= 0:
// log_prob = log(sqrt(2/π)) - log(σ) - 0.5 * (x/σ)²
// grad     = -x / σ²

struct Params {
    x: f32,
    sigma: f32,
    _padding1: f32,
    _padding2: f32,
}

struct Output {
    log_prob: f32,
    grad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: Output;

// Constants
const PI: f32 = 3.14159265358979323846;
const LOG_SQRT_2_OVER_PI: f32 = -0.2257913526447274; // log(sqrt(2/π))

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = params.x;
    let sigma = params.sigma;

    // Compute z = x / sigma
    let z = x / sigma;

    // log_prob = log(sqrt(2/π)) - log(σ) - 0.5 * z²
    let log_prob = LOG_SQRT_2_OVER_PI - log(sigma) - 0.5 * z * z;

    // grad = ∂log_prob/∂x = -x / σ²
    let grad = -x / (sigma * sigma);

    output.log_prob = log_prob;
    output.grad = grad;
}
