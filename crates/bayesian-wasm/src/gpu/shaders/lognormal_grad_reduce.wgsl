// LogNormal distribution gradient REDUCE kernel
//
// Computes grad_log_prob for N observations (x > 0), then reduces to partial sums.
// Each workgroup outputs one partial sum. Final reduction done on CPU.
//
// For LogNormal(mu, sigma) at point x > 0:
// grad_log_prob = -(ln(x) - mu + sigma^2) / (x * sigma^2)
//
// Derivation:
// log_prob = -log(x) - log(sigma) - 0.5*log(2*pi) - 0.5*((log(x) - mu) / sigma)^2
// d/dx[log_prob] = -1/x - (log(x) - mu) / (x * sigma^2)
//                = (-sigma^2 - log(x) + mu) / (x * sigma^2)
//                = -(log(x) - mu + sigma^2) / (x * sigma^2)

struct Params {
    mu: f32,
    sigma: f32,
    count: u32,
    _padding: u32,
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
        let x = x_values[idx];
        let mu = params.mu;
        let sigma = params.sigma;
        let sigma_sq = sigma * sigma;

        let log_x = log(x);

        // grad = -(log(x) - mu + sigma^2) / (x * sigma^2)
        grad = -(log_x - mu + sigma_sq) / (x * sigma_sq);
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
