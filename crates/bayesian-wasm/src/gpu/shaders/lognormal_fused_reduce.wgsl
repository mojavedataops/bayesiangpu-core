// LogNormal distribution FUSED logp + grad REDUCE kernel
//
// Computes BOTH log_prob and grad_log_prob in a single pass, sharing
// intermediate values (log(x), z) to halve global memory reads.
//
// For LogNormal(mu, sigma) at point x > 0:
// log_prob = log_norm - log(x) - 0.5 * z^2
// grad     = -(log(x) - mu + sigma^2) / (x * sigma^2)
// where z = (log(x) - mu) / sigma
//
// log_norm = -0.5 * ln(2*PI) - ln(sigma)
// (pre-computed on CPU and passed via params)

struct Params {
    mu: f32,
    sigma: f32,
    count: u32,
    log_norm: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad: array<f32, 256>;

const WORKGROUP_SIZE: u32 = 256u;
const ELEMS_PER_THREAD: u32 = 4u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let lid = local_id.x;

    var local_logp: f32 = 0.0;
    var local_grad: f32 = 0.0;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = x_values[data_idx];
            let mu = params.mu;
            let sigma = params.sigma;
            let sigma_sq = sigma * sigma;

            // Shared intermediates: log(x), z
            let log_x = log(x);
            let z = (log_x - mu) / sigma;

            local_logp = local_logp + (params.log_norm - log_x - 0.5 * z * z);
            local_grad = local_grad + (-(log_x - mu + sigma_sq) / (x * sigma_sq));
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad[lid] = local_grad;
    workgroupBarrier();

    // Tree reduce both arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad[lid] = shared_grad[lid] + shared_grad[lid + stride];
        }
        workgroupBarrier();
    }

    // Interleaved output: logp at even indices, grad at odd indices
    if (lid == 0u) {
        output[workgroup_id.x * 2u] = shared_logp[0];
        output[workgroup_id.x * 2u + 1u] = shared_grad[0];
    }
}
