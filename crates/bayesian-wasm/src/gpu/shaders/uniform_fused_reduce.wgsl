// Uniform distribution FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for ALL parameters in a single pass.
//
// For Uniform(low, high) at point x:
// log_prob   = -log(high - low)  if low <= x <= high, else -1e10
// grad_low   = 1.0 / (high - low)
// grad_high  = -1.0 / (high - low)
//
// Output layout: output[wid*3] = logp, output[wid*3+1] = grad_low, output[wid*3+2] = grad_high

struct Params {
    low: f32,
    high: f32,
    count: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad_low: array<f32, 256>;
var<workgroup> shared_grad_high: array<f32, 256>;

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
    var local_grad_low: f32 = 0.0;
    var local_grad_high: f32 = 0.0;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = x_values[data_idx];
            let low = params.low;
            let high = params.high;
            let range = high - low;

            if (x >= low && x <= high) {
                local_logp = local_logp + (-log(range));
                local_grad_low = local_grad_low + (1.0 / range);
                local_grad_high = local_grad_high + (-1.0 / range);
            } else {
                // Out-of-bounds penalty, zero gradients
                local_logp = local_logp + (-1e10);
            }
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_low[lid] = local_grad_low;
    shared_grad_high[lid] = local_grad_high;
    workgroupBarrier();

    // Tree reduce all arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_low[lid] = shared_grad_low[lid] + shared_grad_low[lid + stride];
            shared_grad_high[lid] = shared_grad_high[lid] + shared_grad_high[lid + stride];
        }
        workgroupBarrier();
    }

    // Output: 3 values per workgroup
    if (lid == 0u) {
        output[workgroup_id.x * 3u] = shared_logp[0];
        output[workgroup_id.x * 3u + 1u] = shared_grad_low[0];
        output[workgroup_id.x * 3u + 2u] = shared_grad_high[0];
    }
}
