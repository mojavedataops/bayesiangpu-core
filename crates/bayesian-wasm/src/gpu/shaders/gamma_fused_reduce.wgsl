// Gamma distribution FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for ALL parameters in a single pass.
//
// For Gamma(alpha, beta) at point x > 0 (shape, rate parameterization):
// log_prob    = log_norm + (alpha - 1) * log(x) - beta * x
// grad_alpha  = log(x) + neg_psi_alpha_plus_log_beta  (pre-computed: -psi(alpha) + ln(beta))
// grad_beta   = alpha / beta - x
//
// Output layout: output[wid*3] = logp, output[wid*3+1] = grad_alpha, output[wid*3+2] = grad_beta

struct Params {
    alpha: f32,
    beta: f32,
    count: u32,
    log_norm: f32,
    neg_psi_alpha_plus_log_beta: f32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad_alpha: array<f32, 256>;
var<workgroup> shared_grad_beta: array<f32, 256>;

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
    var local_grad_alpha: f32 = 0.0;
    var local_grad_beta: f32 = 0.0;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = x_values[data_idx];
            let alpha = params.alpha;
            let beta = params.beta;

            let log_x = log(x);

            local_logp = local_logp + (params.log_norm + (alpha - 1.0) * log_x - beta * x);
            local_grad_alpha = local_grad_alpha + (log_x + params.neg_psi_alpha_plus_log_beta);
            local_grad_beta = local_grad_beta + (alpha / beta - x);
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_alpha[lid] = local_grad_alpha;
    shared_grad_beta[lid] = local_grad_beta;
    workgroupBarrier();

    // Tree reduce all arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_alpha[lid] = shared_grad_alpha[lid] + shared_grad_alpha[lid + stride];
            shared_grad_beta[lid] = shared_grad_beta[lid] + shared_grad_beta[lid + stride];
        }
        workgroupBarrier();
    }

    // Output: 3 values per workgroup
    if (lid == 0u) {
        output[workgroup_id.x * 3u] = shared_logp[0];
        output[workgroup_id.x * 3u + 1u] = shared_grad_alpha[0];
        output[workgroup_id.x * 3u + 2u] = shared_grad_beta[0];
    }
}
