// WGSL RNG Kernel for BayesianGPU
//
// This shader implements GPU-accelerated random number generation using
// XorShift128 algorithm. It is designed for parallel Monte Carlo sampling
// where each thread maintains independent RNG state.
//
// References:
// - XorShift: Marsaglia, G. (2003). "Xorshift RNGs"
// - PCG: O'Neill, M. E. (2014). "PCG: A Family of Simple Fast Space-Efficient
//   Statistically Good Algorithms for Random Number Generation"
// - WGSL impl: https://gist.github.com/mattdesl/e72d39a1d80b3c1faca81d7425903715

// RNG state storage - each thread has a vec4<u32> state
struct RngState {
    // Array of states, one vec4<u32> per thread
    // state[i] = (x, y, z, w) for thread i
    state: array<vec4<u32>>,
}

// Output buffer for generated random numbers
@group(0) @binding(0)
var<storage, read_write> rng_state: RngState;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// Uniform parameters
struct Params {
    // Number of samples to generate per thread
    num_samples: u32,
    // Total number of threads
    num_threads: u32,
}

@group(0) @binding(2)
var<uniform> params: Params;

// ============================================================================
// PCG Hash Functions
// ============================================================================

/// PCG hash function for seed initialization
/// Takes a 32-bit input and produces a high-quality 32-bit hash
fn pcg_hash(v: u32) -> u32 {
    var state = v * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Initialize RNG state from a seed
/// Generates all 4 components of XorShift128 state using PCG hash chain
fn init_state(seed: u32) -> vec4<u32> {
    let s0 = pcg_hash(seed);
    let s1 = pcg_hash(s0);
    let s2 = pcg_hash(s1);
    let s3 = pcg_hash(s2);

    // Ensure state is never all zeros (XorShift requirement)
    if (s0 == 0u && s1 == 0u && s2 == 0u && s3 == 0u) {
        return vec4<u32>(1u, s1, s2, s3);
    }

    return vec4<u32>(s0, s1, s2, s3);
}

// ============================================================================
// XorShift128 Random Number Generator
// ============================================================================

/// XorShift128 step function
/// Updates state in-place and returns the next random u32
///
/// XorShift128 has period 2^128 - 1 and passes most statistical tests.
/// The constants (11, 8, 19) are from Marsaglia's original paper.
fn xorshift128(state: ptr<function, vec4<u32>>) -> u32 {
    var t = (*state).x;
    let s = (*state).w;

    // Shift left
    (*state).x = (*state).y;
    (*state).y = (*state).z;
    (*state).z = (*state).w;

    // XOR operations
    t = t ^ (t << 11u);
    t = t ^ (t >> 8u);

    // Final state update
    (*state).w = t ^ s ^ (s >> 19u);

    return (*state).w;
}

// ============================================================================
// Distribution Functions
// ============================================================================

/// Convert u32 to float in [0, 1)
/// Uses division by 2^32 for uniform distribution
fn to_float(x: u32) -> f32 {
    // 4294967296.0 = 2^32
    return f32(x) / 4294967296.0;
}

/// Generate standard normal sample using Box-Muller transform
/// Returns one of two normal samples (the other could be cached)
fn box_muller_z0(state: ptr<function, vec4<u32>>) -> f32 {
    let u1 = max(to_float(xorshift128(state)), 0.0000001);  // Avoid log(0)
    let u2 = to_float(xorshift128(state));

    let r = sqrt(-2.0 * log(u1));
    let theta = 6.283185307179586 * u2;  // 2 * PI

    return r * cos(theta);
}

/// Generate two standard normal samples using Box-Muller transform
fn box_muller(state: ptr<function, vec4<u32>>) -> vec2<f32> {
    let u1 = max(to_float(xorshift128(state)), 0.0000001);  // Avoid log(0)
    let u2 = to_float(xorshift128(state));

    let r = sqrt(-2.0 * log(u1));
    let theta = 6.283185307179586 * u2;  // 2 * PI

    return vec2<f32>(r * cos(theta), r * sin(theta));
}

// ============================================================================
// Compute Shaders
// ============================================================================

/// Generate uniform random samples in [0, 1)
///
/// Each thread generates one sample and stores it to the output buffer.
/// Thread ID determines which state to use and where to write.
@compute @workgroup_size(256)
fn generate_uniform(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // Bounds check
    if (idx >= arrayLength(&output)) {
        return;
    }

    // Determine which thread state to use (round-robin)
    let thread_idx = idx % params.num_threads;

    // Load state
    var state = rng_state.state[thread_idx];

    // Generate random number
    let rand = xorshift128(&state);

    // Store updated state
    rng_state.state[thread_idx] = state;

    // Convert to float and store
    output[idx] = to_float(rand);
}

/// Generate standard normal random samples
///
/// Uses Box-Muller transform to convert uniform samples to normal.
/// Each thread generates one normal sample.
@compute @workgroup_size(256)
fn generate_normal(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // Bounds check
    if (idx >= arrayLength(&output)) {
        return;
    }

    // Determine which thread state to use
    let thread_idx = idx % params.num_threads;

    // Load state
    var state = rng_state.state[thread_idx];

    // Generate normal sample using Box-Muller
    let normal = box_muller_z0(&state);

    // Store updated state
    rng_state.state[thread_idx] = state;

    // Store result
    output[idx] = normal;
}

/// Generate multiple uniform samples per thread
///
/// More efficient for large batches - each thread generates multiple samples
/// in sequence, reducing memory bandwidth for state access.
@compute @workgroup_size(256)
fn generate_uniform_batch(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let thread_id = global_id.x;

    // Bounds check
    if (thread_id >= params.num_threads) {
        return;
    }

    // Load state for this thread
    var state = rng_state.state[thread_id];

    // Generate multiple samples
    let samples_per_thread = params.num_samples / params.num_threads;
    let base_idx = thread_id * samples_per_thread;

    for (var i: u32 = 0u; i < samples_per_thread; i = i + 1u) {
        let output_idx = base_idx + i;
        if (output_idx < arrayLength(&output)) {
            let rand = xorshift128(&state);
            output[output_idx] = to_float(rand);
        }
    }

    // Store updated state
    rng_state.state[thread_id] = state;
}

/// Generate multiple normal samples per thread
///
/// Uses Box-Muller to generate pairs of normal samples efficiently.
@compute @workgroup_size(256)
fn generate_normal_batch(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let thread_id = global_id.x;

    // Bounds check
    if (thread_id >= params.num_threads) {
        return;
    }

    // Load state for this thread
    var state = rng_state.state[thread_id];

    // Generate multiple samples
    let samples_per_thread = params.num_samples / params.num_threads;
    let base_idx = thread_id * samples_per_thread;

    // Generate pairs using Box-Muller
    var i: u32 = 0u;
    while (i < samples_per_thread) {
        let normals = box_muller(&state);

        let idx0 = base_idx + i;
        if (idx0 < arrayLength(&output)) {
            output[idx0] = normals.x;
        }

        let idx1 = base_idx + i + 1u;
        if (i + 1u < samples_per_thread && idx1 < arrayLength(&output)) {
            output[idx1] = normals.y;
        }

        i = i + 2u;
    }

    // Store updated state
    rng_state.state[thread_id] = state;
}

// ============================================================================
// Utility Kernels
// ============================================================================

/// Initialize RNG states from seeds
///
/// Each thread initializes its own state from a base seed + thread offset.
@compute @workgroup_size(256)
fn init_rng_states(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let thread_id = global_id.x;

    // Bounds check
    if (thread_id >= params.num_threads) {
        return;
    }

    // Initialize state from base seed (stored in output[0]) + thread offset
    // This assumes output[0] contains the base seed as a reinterpreted float
    let base_seed = bitcast<u32>(output[0]);
    let thread_seed = base_seed + thread_id * 0x9E3779B9u;  // Golden ratio

    rng_state.state[thread_id] = init_state(thread_seed);
}
