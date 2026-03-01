// Normal indexed REDUCE kernel for hierarchical models
//
// Computes logp and grad_sigma for y[i] ~ Normal(theta[group[i]], sigma)
// where theta is a vector of group-level parameters.
//
// Per-group gradients (grad_theta[k]) are computed via separate dispatches
// over pre-sorted group slices on the CPU side.
//
// Output layout: output[wid*2] = logp, output[wid*2+1] = grad_sigma
// (same as the 2-output fused pattern)

struct Params {
    sigma: f32,
    count: u32,
    num_groups: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> y_values: array<f32>;
@group(0) @binding(2) var<storage, read> theta: array<f32>;
@group(0) @binding(3) var<storage, read> group_idx: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad_sigma: array<f32, 256>;

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
    var local_grad_sigma: f32 = 0.0;

    let sigma = params.sigma;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let y = y_values[data_idx];
            let g = group_idx[data_idx];
            let mu_i = theta[g];

            let z = (y - mu_i) / sigma;
            let z_sq = z * z;

            local_logp = local_logp + (-0.5 * LOG_2PI - log(sigma) - 0.5 * z_sq);
            local_grad_sigma = local_grad_sigma + ((-1.0 + z_sq) / sigma);
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_sigma[lid] = local_grad_sigma;
    workgroupBarrier();

    // Tree reduce
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_sigma[lid] = shared_grad_sigma[lid] + shared_grad_sigma[lid + stride];
        }
        workgroupBarrier();
    }

    // Output: 2 values per workgroup
    if (lid == 0u) {
        output[workgroup_id.x * 2u] = shared_logp[0];
        output[workgroup_id.x * 2u + 1u] = shared_grad_sigma[0];
    }
}
