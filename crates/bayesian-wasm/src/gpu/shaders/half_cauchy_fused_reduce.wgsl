// Half-Cauchy distribution FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for ALL parameters in a single pass.
//
// For HalfCauchy(scale) at point x (x >= 0):
// log_prob    = ln(2/pi) - ln(scale) - ln(1 + (x/scale)^2)
// grad_scale  = (x^2 - scale^2) / (scale * (scale^2 + x^2))
//
// Output layout: output[wid*2] = logp, output[wid*2+1] = grad_scale

struct Params {
    scale: f32,
    count: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad_scale: array<f32, 256>;

const LOG_2_OVER_PI: f32 = -0.4515827052894549;
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
    var local_grad_scale: f32 = 0.0;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = x_values[data_idx];
            let s = params.scale;
            let x_sq = x * x;
            let s_sq = s * s;

            local_logp = local_logp + (LOG_2_OVER_PI - log(s) - log(1.0 + x_sq / s_sq));
            local_grad_scale = local_grad_scale + ((x_sq - s_sq) / (s * (s_sq + x_sq)));
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_scale[lid] = local_grad_scale;
    workgroupBarrier();

    // Tree reduce all arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_scale[lid] = shared_grad_scale[lid] + shared_grad_scale[lid + stride];
        }
        workgroupBarrier();
    }

    // Output: 2 values per workgroup
    if (lid == 0u) {
        output[workgroup_id.x * 2u] = shared_logp[0];
        output[workgroup_id.x * 2u + 1u] = shared_grad_scale[0];
    }
}
