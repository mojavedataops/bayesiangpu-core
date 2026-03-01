// Normal linear predictor FUSED logp + multi-grad REDUCE kernel
//
// Computes log_prob AND gradients for y ~ Normal(X @ beta, sigma)
//
// For observation i:
//   mu_i = dot(X[i,:], beta)   (linear predictor)
//   r_i  = y[i] - mu_i
//   z_i  = r_i / sigma
//   logp += -0.5 * log(2pi) - log(sigma) - 0.5 * z^2
//   grad_sigma += (-1 + z^2) / sigma
//   grad_beta[j] += r_i * X[i,j] / sigma^2    for j in 0..P
//
// Output layout per workgroup: [logp, grad_sigma, grad_beta[0], ..., grad_beta[P-1]]
// Total: (P+2) values per workgroup

struct Params {
    sigma: f32,
    count: u32,
    p: u32,       // number of predictors
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> y_values: array<f32>;
@group(0) @binding(2) var<storage, read> x_matrix: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// MAX_P=29 keeps shared memory within 32KB limit:
// (29*256 + 256 + 256) * 4 = 31744 bytes < 32768
const MAX_P: u32 = 29u;
const WORKGROUP_SIZE: u32 = 256u;
const ELEMS_PER_THREAD: u32 = 4u;
const LOG_2PI: f32 = 1.8378770664093453;

// Shared memory for tree reduction
// logp: 256 f32, grad_sigma: 256 f32, grad_beta: MAX_P * 256 f32
var<workgroup> shared_logp: array<f32, 256>;
var<workgroup> shared_grad_sigma: array<f32, 256>;
var<workgroup> shared_grad_beta: array<f32, 7424>; // MAX_P(29) * 256

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let lid = local_id.x;
    let p = params.p;
    let sigma = params.sigma;
    let inv_sigma_sq = 1.0 / (sigma * sigma);

    var local_logp: f32 = 0.0;
    var local_grad_sigma: f32 = 0.0;

    // Zero out local grad_beta accumulators in shared memory
    // (we accumulate directly into shared for the reduction)
    for (var j: u32 = 0u; j < p; j = j + 1u) {
        shared_grad_beta[j * WORKGROUP_SIZE + lid] = 0.0;
    }

    let base = workgroup_id.x * (WORKGROUP_SIZE * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * WORKGROUP_SIZE;
        if (data_idx < params.count) {
            // Compute mu_i = dot(X[data_idx, :], beta)
            var mu_i: f32 = 0.0;
            let row_offset = data_idx * p;
            for (var j: u32 = 0u; j < p; j = j + 1u) {
                mu_i = mu_i + x_matrix[row_offset + j] * beta[j];
            }

            let r_i = y_values[data_idx] - mu_i;
            let z = r_i / sigma;
            let z_sq = z * z;

            local_logp = local_logp + (-0.5 * LOG_2PI - log(sigma) - 0.5 * z_sq);
            local_grad_sigma = local_grad_sigma + ((-1.0 + z_sq) / sigma);

            // grad_beta[j] += r_i * X[i,j] / sigma^2
            for (var j: u32 = 0u; j < p; j = j + 1u) {
                shared_grad_beta[j * WORKGROUP_SIZE + lid] = shared_grad_beta[j * WORKGROUP_SIZE + lid] + r_i * x_matrix[row_offset + j] * inv_sigma_sq;
            }
        }
    }

    shared_logp[lid] = local_logp;
    shared_grad_sigma[lid] = local_grad_sigma;
    workgroupBarrier();

    // Tree reduce all arrays
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_logp[lid] = shared_logp[lid] + shared_logp[lid + stride];
            shared_grad_sigma[lid] = shared_grad_sigma[lid] + shared_grad_sigma[lid + stride];
            for (var j: u32 = 0u; j < p; j = j + 1u) {
                let idx = j * WORKGROUP_SIZE + lid;
                shared_grad_beta[idx] = shared_grad_beta[idx] + shared_grad_beta[idx + stride];
            }
        }
        workgroupBarrier();
    }

    // Output: (P+2) values per workgroup
    if (lid == 0u) {
        let out_offset = workgroup_id.x * (p + 2u);
        output[out_offset] = shared_logp[0];
        output[out_offset + 1u] = shared_grad_sigma[0];
        for (var j: u32 = 0u; j < p; j = j + 1u) {
            output[out_offset + 2u + j] = shared_grad_beta[j * WORKGROUP_SIZE];
        }
    }
}
