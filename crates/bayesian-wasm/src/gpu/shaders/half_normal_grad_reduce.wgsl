// HalfNormal distribution gradient REDUCE kernel
//
// Computes grad_log_prob for N observations (x >= 0), then reduces to partial sums.
// Each workgroup outputs one partial sum. Final reduction done on CPU.
//
// For HalfNormal(sigma) at point x >= 0:
// grad_log_prob = -x / sigma^2

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

const WORKGROUP_SIZE: u32 = 256u;
const ELEMS_PER_THREAD: u32 = 4u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let lid = local_id.x;

    // Each thread accumulates ELEMS_PER_THREAD elements
    var local_sum: f32 = 0.0;
    let base = workgroup_id.x * (256u * ELEMS_PER_THREAD) + lid;
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
        let data_idx = base + i * 256u;
        if (data_idx < params.count) {
            let x = x_values[data_idx];
            let sigma = params.sigma;
            let sigma_sq = sigma * sigma;
            // grad = -x / sigma^2
            local_sum = local_sum + (-x / sigma_sq);
        }
    }

    // Store in shared memory
    shared_data[lid] = local_sum;
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
