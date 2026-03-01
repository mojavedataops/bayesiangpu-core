// LogNormal distribution FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for ALL parameters in a single pass.
//
// For LogNormal(mu, sigma) at point x > 0:
// log_prob    = log_norm - log(x) - 0.5 * z^2
// grad_mu     = (log(x) - mu) / sigma^2 = z / sigma
// grad_sigma  = -1/sigma + (log(x) - mu)^2 / sigma^3 = (-1 + z^2) / sigma
// where z = (log(x) - mu) / sigma
//
// Output layout: output[wid*3] = logp, output[wid*3+1] = grad_mu, output[wid*3+2] = grad_sigma

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
var<workgroup> shared_grad_mu: array<f32, 256>;
var<workgroup> shared_grad_sigma: array<f32, 256>;

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
    var local_grad_mu: f32 = 0.0;
    var local_grad_sigma: f32 = 0.0;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = x_values[data_idx];
            let mu = params.mu;
            let sigma = params.sigma;

            let log_x = log(x);
            let z = (log_x - mu) / sigma;
            let z_sq = z * z;

            local_logp = local_logp + (params.log_norm - log_x - 0.5 * z_sq);
            local_grad_mu = local_grad_mu + (z / sigma);
            local_grad_sigma = local_grad_sigma + ((-1.0 + z_sq) / sigma);
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_mu[lid] = local_grad_mu;
    shared_grad_sigma[lid] = local_grad_sigma;
    workgroupBarrier();

    // Tree reduce all arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_mu[lid] = shared_grad_mu[lid] + shared_grad_mu[lid + stride];
            shared_grad_sigma[lid] = shared_grad_sigma[lid] + shared_grad_sigma[lid + stride];
        }
        workgroupBarrier();
    }

    // Output: 3 values per workgroup
    if (lid == 0u) {
        output[workgroup_id.x * 3u] = shared_logp[0];
        output[workgroup_id.x * 3u + 1u] = shared_grad_mu[0];
        output[workgroup_id.x * 3u + 2u] = shared_grad_sigma[0];
    }
}
