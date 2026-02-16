// Exponential distribution gradient REDUCE kernel
//
// Computes grad_log_prob for N observations (x >= 0), then reduces to partial sums.
// Each workgroup outputs one partial sum. Final reduction done on CPU.
//
// For Exponential(lambda) at point x >= 0:
// grad_log_prob = -lambda (constant for all x)
//
// Note: The sum of gradients = -lambda * N (where N is the number of observations)
// This kernel computes this via reduction for consistency with other distributions.

struct Params {
    lambda: f32,
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

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let idx = global_id.x;
    let lid = local_id.x;

    // Compute grad_log_prob for this element (or 0 if out of bounds)
    var grad: f32 = 0.0;
    if (idx < params.count) {
        // grad = -lambda (constant, independent of x)
        grad = -params.lambda;
    }

    // Store in shared memory
    shared_data[lid] = grad;
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
