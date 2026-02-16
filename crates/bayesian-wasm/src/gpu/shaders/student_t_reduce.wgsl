// StudentT distribution BATCHED log_prob with REDUCTION
//
// Computes log_prob for N observations, then reduces to partial sums.
// Each workgroup outputs one partial sum. Final reduction done on CPU.
//
// For StudentT(nu, loc, scale) at point x:
// log_prob = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * π) - log(scale) - ((nu + 1) / 2) * log(1 + z² / nu)
// where z = (x - loc) / scale

struct Params {
    loc: f32,
    scale: f32,
    nu: f32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f32>;

// Shared memory for workgroup reduction
var<workgroup> shared_data: array<f32, 256>;

const PI: f32 = 3.141592653589793;
const WORKGROUP_SIZE: u32 = 256u;

// Lanczos approximation for lgamma (accurate for all x > 0)
fn lgamma(x: f32) -> f32 {
    let g: f32 = 7.0;
    let c0: f32 = 0.99999999999980993;
    let c1: f32 = 676.5203681218851;
    let c2: f32 = -1259.1392167224028;
    let c3: f32 = 771.32342877765313;
    let c4: f32 = -176.61502916214059;
    let c5: f32 = 12.507343278686905;
    let c6: f32 = -0.13857109526572012;
    let c7: f32 = 9.9843695780195716e-6;
    let c8: f32 = 1.5056327351493116e-7;

    let x1 = x - 1.0;
    var sum = c0;
    sum += c1 / (x1 + 1.0);
    sum += c2 / (x1 + 2.0);
    sum += c3 / (x1 + 3.0);
    sum += c4 / (x1 + 4.0);
    sum += c5 / (x1 + 5.0);
    sum += c6 / (x1 + 6.0);
    sum += c7 / (x1 + 7.0);
    sum += c8 / (x1 + 8.0);

    let t = x1 + g + 0.5;
    return 0.5 * log(2.0 * 3.14159265358979323846) + (x1 + 0.5) * log(t) - t + log(sum);
}

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
        let loc = params.loc;
        let scale = params.scale;
        let nu = params.nu;

        let z = (x - loc) / scale;
        let z_sq = z * z;

        // log_prob = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * π) - log(scale) - ((nu + 1) / 2) * log(1 + z² / nu)
        log_prob = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(nu * PI) - log(scale) - ((nu + 1.0) / 2.0) * log(1.0 + z_sq / nu);
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
