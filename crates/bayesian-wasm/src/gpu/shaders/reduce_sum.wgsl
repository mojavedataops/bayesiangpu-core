// Generic array summation reduction
//
// Reduces an array of f32 partial sums to fewer partial sums (or a single value).
// Used as a second-pass reduction when the first-pass produces > 1024 partial sums.
//
// Input: array of f32 values to sum
// Output: array of partial sums (one per workgroup)

struct Params {
    count: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let lid = local_id.x;
    let gid = workgroup_id.x * WORKGROUP_SIZE + lid;

    // Load from input (0 if out of bounds)
    if (gid < params.count) {
        shared_data[lid] = input[gid];
    } else {
        shared_data[lid] = 0.0;
    }
    workgroupBarrier();

    // Tree-based parallel reduction
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            shared_data[lid] = shared_data[lid] + shared_data[lid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the workgroup's partial sum
    if (lid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
