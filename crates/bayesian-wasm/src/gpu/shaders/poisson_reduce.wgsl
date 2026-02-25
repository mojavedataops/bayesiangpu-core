// Poisson distribution BATCHED log_prob with REDUCTION
//
// Computes log_prob for N observations, then reduces to partial sums.
// Each workgroup outputs one partial sum. Final reduction done on CPU.
//
// log_prob = k * log(lambda) - lambda - log(k!)
//          = k * log(lambda) - lambda - lgamma(k + 1)

struct Params {
    lambda: f32,  // rate parameter
    count: u32,   // number of observations
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f32>;

// Shared memory for workgroup reduction
var<workgroup> shared_data: array<f32, 256>;

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

// log(n!) = lgamma(n + 1)
fn log_factorial(n: f32) -> f32 {
    return lgamma(n + 1.0);
}

// Precomputed log(n!) for n=0..20 for exact values on small integers
fn log_factorial_exact(n: u32) -> f32 {
    switch(n) {
        case 0u, 1u: { return 0.0; }
        case 2u: { return 0.6931471805599453; }
        case 3u: { return 1.7917594692280550; }
        case 4u: { return 3.1780538303479458; }
        case 5u: { return 4.7874917427820458; }
        case 6u: { return 6.5792512120101012; }
        case 7u: { return 8.5251613610654147; }
        case 8u: { return 10.6046029027452509; }
        case 9u: { return 12.8018274800814705; }
        case 10u: { return 15.1044125730755159; }
        case 11u: { return 17.5023078458738865; }
        case 12u: { return 19.9872144956618882; }
        case 13u: { return 22.5521638531234237; }
        case 14u: { return 25.1912211827386828; }
        case 15u: { return 27.8992713838408929; }
        case 16u: { return 30.6718601060806738; }
        case 17u: { return 33.5050734501368895; }
        case 18u: { return 36.3954452080330546; }
        case 19u: { return 39.3398841871994946; }
        case 20u: { return 42.3356164607534888; }
        default: { return lgamma(f32(n) + 1.0); }
    }
}

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
            let k = x_values[data_idx];      // observed count
            let lambda = params.lambda;  // rate

            // Poisson log probability
            // log_prob = k * log(lambda) - lambda - log(k!)
            // Use exact lookup for k (observed count, typically small)
            local_sum = local_sum + (k * log(lambda) - lambda - log_factorial_exact(u32(k)));
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
