// Normal distribution FUSED logp + grad REDUCE kernel
//
// Computes BOTH log_prob and grad_log_prob in a single pass, sharing
// intermediate values (z) to halve global memory reads.
//
// For Normal(mu, sigma) at point x:
// log_prob = -0.5 * log(2pi) - log(sigma) - 0.5 * z^2
// grad     = -z / sigma    (where z = (x - mu) / sigma)

struct Params {
    mu: f32,
    sigma: f32,
    count: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad: array<f32, 256>;

const LOG_2PI: f32 = 1.8378770664093453;
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

            // Shared intermediate: z = (x - mu) / sigma
            let z = (x - mu) / sigma;

            local_logp = local_logp + (-0.5 * LOG_2PI - log(sigma) - 0.5 * z * z);
            local_grad = local_grad + (-z / sigma);
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
