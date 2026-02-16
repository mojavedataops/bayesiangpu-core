// HalfNormal distribution BATCHED log_prob with REDUCTION
//
// Computes log_prob for N observations (x >= 0), then reduces to partial sums.
// Each workgroup outputs one partial sum. Final reduction done on CPU.
//
// For HalfNormal(sigma) at point x >= 0:
// log_prob = log(sqrt(2/π)) - log(σ) - 0.5 * (x/σ)²

struct Params {
    sigma: f32,
    count: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f32>;

// Shared memory for workgroup reduction
var<workgroup> shared_data: array<f32, 256>;

// log(sqrt(2/π)) = 0.5 * log(2/π) = 0.5 * (log(2) - log(π))
const LOG_SQRT_2_OVER_PI: f32 = -0.2257913526447274;
const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let idx = global_id.x;
    let lid = local_id.x;

    // Compute log_prob for this element (or 0 if out of bounds)
    var log_prob: f32 = 0.0;
    if (idx < params.count) {
        let x = x_values[idx];
        let sigma = params.sigma;
        let z = x / sigma;
        log_prob = LOG_SQRT_2_OVER_PI - log(sigma) - 0.5 * z * z;
    }

    // Store in shared memory
    shared_data[lid] = log_prob;
    workgroupBarrier();

    // Parallel reduction within workgroup
    // Tree-based reduction: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_data[lid] = shared_data[lid] + shared_data[lid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the workgroup's partial sum
    if (lid == 0u) {
        partial_sums[workgroup_id.x] = shared_data[0];
    }
}
