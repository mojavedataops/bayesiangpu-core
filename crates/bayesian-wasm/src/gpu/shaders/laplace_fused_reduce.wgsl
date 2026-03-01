// Laplace distribution FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for ALL parameters in a single pass.
//
// For Laplace(loc, scale) at point x:
// log_prob    = -ln(2*scale) - |x - loc| / scale
// grad_loc    = sign(x - loc) / scale
// grad_scale  = (-1.0 + |x - loc| / scale) / scale
//
// Output layout: output[wid*3] = logp, output[wid*3+1] = grad_loc, output[wid*3+2] = grad_scale

struct Params {
    loc: f32,
    scale: f32,
    count: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad_loc: array<f32, 256>;
var<workgroup> shared_grad_scale: array<f32, 256>;

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
    var local_grad_loc: f32 = 0.0;
    var local_grad_scale: f32 = 0.0;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = x_values[data_idx];
            let loc = params.loc;
            let scale = params.scale;

            let diff = x - loc;
            let abs_diff = abs(diff);

            local_logp = local_logp + (-log(2.0 * scale) - abs_diff / scale);
            local_grad_loc = local_grad_loc + (sign(diff) / scale);
            local_grad_scale = local_grad_scale + ((-1.0 + abs_diff / scale) / scale);
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_loc[lid] = local_grad_loc;
    shared_grad_scale[lid] = local_grad_scale;
    workgroupBarrier();

    // Tree reduce all arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_loc[lid] = shared_grad_loc[lid] + shared_grad_loc[lid + stride];
            shared_grad_scale[lid] = shared_grad_scale[lid] + shared_grad_scale[lid + stride];
        }
        workgroupBarrier();
    }

    // Output: 3 values per workgroup
    if (lid == 0u) {
        output[workgroup_id.x * 3u] = shared_logp[0];
        output[workgroup_id.x * 3u + 1u] = shared_grad_loc[0];
        output[workgroup_id.x * 3u + 2u] = shared_grad_scale[0];
    }
}
