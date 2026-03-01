// Logistic distribution FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for ALL parameters in a single pass.
//
// For Logistic(loc, scale) at point x:
// z = (x - loc) / scale
// log_prob    = -ln(scale) - |z| - 2*ln(1 + exp(-|z|))
// grad_loc    = (2*sigmoid(z) - 1) / scale
// grad_scale  = (-1 + z*(2*sigmoid(z) - 1)) / scale
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

fn stable_sigmoid(z: f32) -> f32 {
    if (z >= 0.0) {
        return 1.0 / (1.0 + exp(-z));
    } else {
        let ez = exp(z);
        return ez / (1.0 + ez);
    }
}

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

            let z = (x - loc) / scale;
            let abs_z = abs(z);

            // Numerically stable log-pdf
            local_logp = local_logp + (-log(scale) - abs_z - 2.0 * log(1.0 + exp(-abs_z)));

            // Gradients via stable sigmoid
            let sig = stable_sigmoid(z);
            let two_sig_minus_one = 2.0 * sig - 1.0;
            local_grad_loc = local_grad_loc + (two_sig_minus_one / scale);
            local_grad_scale = local_grad_scale + ((-1.0 + z * two_sig_minus_one) / scale);
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
