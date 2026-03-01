// StudentT distribution FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for ALL parameters in a single pass.
//
// For StudentT(nu, loc, scale) at point x:
// log_prob    = log_norm - ((nu + 1) / 2) * log(1 + z^2 / nu)
// grad_loc    = (nu + 1) * z / (scale * (nu + z^2))
// grad_scale  = (-1 + (nu + 1) * z^2 / (nu + z^2)) / scale
// grad_nu     = psi_const - 0.5 * log(1 + z^2/nu) + 0.5 * (nu+1) * z^2 / (nu * (nu + z^2))
//   where psi_const = 0.5 * (psi((nu+1)/2) - psi(nu/2) - 1/nu) (pre-computed on CPU)
//
// Output layout: output[wid*4] = logp, +1 = grad_loc, +2 = grad_scale, +3 = grad_nu

struct Params {
    loc: f32,
    scale: f32,
    nu: f32,
    count: u32,
    log_norm: f32,
    psi_const: f32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad_loc: array<f32, 256>;
var<workgroup> shared_grad_scale: array<f32, 256>;
var<workgroup> shared_grad_nu: array<f32, 256>;

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
    var local_grad_nu: f32 = 0.0;

    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = x_values[data_idx];
            let loc = params.loc;
            let scale = params.scale;
            let nu = params.nu;

            let z = (x - loc) / scale;
            let z_sq = z * z;
            let nu_plus_zsq = nu + z_sq;
            let ratio = z_sq / nu_plus_zsq;

            local_logp = local_logp + (params.log_norm - ((nu + 1.0) / 2.0) * log(1.0 + z_sq / nu));
            local_grad_loc = local_grad_loc + ((nu + 1.0) * z / (scale * nu_plus_zsq));
            local_grad_scale = local_grad_scale + ((-1.0 + (nu + 1.0) * ratio) / scale);
            local_grad_nu = local_grad_nu + (params.psi_const - 0.5 * log(1.0 + z_sq / nu) + 0.5 * (nu + 1.0) * z_sq / (nu * nu_plus_zsq));
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_loc[lid] = local_grad_loc;
    shared_grad_scale[lid] = local_grad_scale;
    shared_grad_nu[lid] = local_grad_nu;
    workgroupBarrier();

    // Tree reduce all arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_loc[lid] = shared_grad_loc[lid] + shared_grad_loc[lid + stride];
            shared_grad_scale[lid] = shared_grad_scale[lid] + shared_grad_scale[lid + stride];
            shared_grad_nu[lid] = shared_grad_nu[lid] + shared_grad_nu[lid + stride];
        }
        workgroupBarrier();
    }

    // Output: 4 values per workgroup
    if (lid == 0u) {
        output[workgroup_id.x * 4u] = shared_logp[0];
        output[workgroup_id.x * 4u + 1u] = shared_grad_loc[0];
        output[workgroup_id.x * 4u + 2u] = shared_grad_scale[0];
        output[workgroup_id.x * 4u + 3u] = shared_grad_nu[0];
    }
}
