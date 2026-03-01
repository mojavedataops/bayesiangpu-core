// Weibull distribution FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for ALL parameters in a single pass.
//
// For Weibull(shape=k, scale=lambda) at point x (x > 0):
// t = x / lambda
// log_prob    = ln(k) - k*ln(lambda) + (k-1)*ln(x) - t^k
// grad_shape  = 1/k + ln(t) * (1 - t^k)
// grad_scale  = k * (t^k - 1) / lambda
//
// Output layout: output[wid*3] = logp, output[wid*3+1] = grad_shape, output[wid*3+2] = grad_scale

struct Params {
    shape: f32,
    scale: f32,
    count: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad_shape: array<f32, 256>;
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
    var local_grad_shape: f32 = 0.0;
    var local_grad_scale: f32 = 0.0;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = max(x_values[data_idx], 1e-10);
            let k = params.shape;
            let lambda = params.scale;

            let t = x / lambda;
            let ln_t = log(t);
            let t_k = pow(t, k);

            local_logp = local_logp + (log(k) - k * log(lambda) + (k - 1.0) * log(x) - t_k);
            local_grad_shape = local_grad_shape + (1.0 / k + ln_t * (1.0 - t_k));
            local_grad_scale = local_grad_scale + (k * (t_k - 1.0) / lambda);
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_shape[lid] = local_grad_shape;
    shared_grad_scale[lid] = local_grad_scale;
    workgroupBarrier();

    // Tree reduce all arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_shape[lid] = shared_grad_shape[lid] + shared_grad_shape[lid + stride];
            shared_grad_scale[lid] = shared_grad_scale[lid] + shared_grad_scale[lid + stride];
        }
        workgroupBarrier();
    }

    // Output: 3 values per workgroup
    if (lid == 0u) {
        output[workgroup_id.x * 3u] = shared_logp[0];
        output[workgroup_id.x * 3u + 1u] = shared_grad_shape[0];
        output[workgroup_id.x * 3u + 2u] = shared_grad_scale[0];
    }
}
