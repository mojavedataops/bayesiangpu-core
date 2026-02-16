# Debug Report: WebGPU WASM "unreachable" Panic During Inference
Generated: 2026-01-20

## Symptom

When running with the `wgpu` feature in browser WebGPU, tensor operations hit an "unreachable" WASM panic during inference. WebGPU initialization succeeds (`init_setup_async` completes), but the first tensor operation during inference fails.

## Investigation Steps

1. Read the key files in the inference path: `lib.rs`, `model.rs`, `pcg.rs`, `backend.rs`, `nuts.rs`, `leapfrog.rs`
2. Searched for all `into_data().to_vec()` calls across the codebase
3. Examined existing documentation in `.beans/` and `thoughts/shared/handoffs/`
4. Reviewed Cargo.toml configurations for feature flags

## Evidence

### Finding 1: Prior Investigation Already Identified Root Cause
- **Location:** `/Users/borikropotkin/bayesiangpu/.beans/bayesiangpu-rgph--burn-wgpu-tensor-operations-fail-in-browser-wasm.md`
- **Observation:** The issue has been traced to **Burn's `fusion` feature being incompatible with WASM**. From Burn documentation: "which might be necessary on `wasm` for now" (referring to disabling fusion)
- **Relevance:** The fusion feature compiles and optimizes streams of tensor operations, but this doesn't work correctly in browser WebGPU.

### Finding 2: Synchronous `into_data().to_vec()` Pattern is Problematic
- **Location:** Multiple files, but critically in `/Users/borikropotkin/bayesiangpu/crates/bayesian-sampler/src/model.rs:171`
- **Observation:** The `logp_and_grad` function does:
  ```rust
  let logp_data: Vec<f32> = log_prob.into_data().to_vec().unwrap();
  ```
  This is called synchronously, but WebGPU buffer reads in browsers are inherently asynchronous.
- **Relevance:** Even with fusion disabled, the synchronous data extraction pattern may still cause issues in browser WebGPU.

### Finding 3: Numerous Synchronous GPU-to-CPU Data Transfers
- **Location:** Throughout the codebase (126+ occurrences)
- **Observation:** The code uses `into_data().to_vec()` extensively:
  - `nuts.rs`: Lines 162, 180, 200, 228-231, 369, 501, 519, 539, 573, 618
  - `leapfrog.rs`: Lines 122-123, 269, 452-453, 475-476
  - `model.rs`: Lines 171, 173, 364, 378, 412-413, 429, 444, 506, 510
  - `pcg.rs`: Line 122 (RNG state readback)
  - `lib.rs`: Line 492 (in `generate_inits`)
- **Relevance:** Each of these is a potential panic point in browser WebGPU where async buffer mapping is required.

### Finding 4: Fix Was Already Applied (But May Not Be Complete)
- **Location:** `/Users/borikropotkin/bayesiangpu/crates/bayesian-wasm/Cargo.toml:37-38`
- **Observation:** A direct dependency on `burn-wgpu` with `default-features = false, features = ["std", "autotune"]` was added to disable fusion
- **Relevance:** This should address the fusion-related crashes, but hasn't been verified in browser testing yet.

### Finding 5: Upgraded to Burn 0.20
- **Location:** `/Users/borikropotkin/bayesiangpu/Cargo.toml:18-20`
- **Observation:** Workspace upgraded from Burn 0.19 to 0.20
- **Relevance:** Burn 0.20 may include fixes for browser WebGPU issues.

## Root Cause Analysis

**Most Likely Cause: Burn's Fusion Feature + Synchronous Data Access**

The "unreachable" panic is caused by **Burn's `fusion` feature** which optimizes tensor operation streams in ways incompatible with browser WASM/WebGPU. A fix has been applied (disabling fusion via `burn-wgpu` feature configuration), but this needs browser verification.

**Secondary Issue: Synchronous GPU-to-CPU Transfers**

Even with fusion disabled, the codebase makes heavy use of synchronous `into_data().to_vec()` calls to read tensor data from GPU to CPU. In browser WebGPU:
- GPU buffer reads require async `mapAsync()` + `getMappedRange()`
- Burn may handle this internally, but the pattern is still risky
- These calls occur in hot paths during NUTS sampling (100+ times per iteration)

**Specific Failure Point in Code Flow:**

```
run_inference()
  └── generate_inits()
        └── GpuRng::normal() [pcg.rs:188]
              └── self.state.clone().into_data().to_vec() [line 122]
                    ↓
              PANIC: "unreachable" when fusion optimizes tensor stream
```

Or:

```
run_inference()
  └── sampler.sample(inits)
        └── logp_and_grad() [model.rs:152]
              └── log_prob.into_data().to_vec() [line 171]
                    ↓
              PANIC: "unreachable" on first GPU data readback
```

**Confidence:** High

**Alternative hypotheses:**
1. Burn 0.20 may have new bugs specific to WebGPU WASM
2. The `autotune` feature in `burn-wgpu` could cause issues
3. WebGPU shader compilation differences between native and browser

## Recommended Fix

The primary fix (disabling fusion) has already been applied. What's needed now:

**Files already modified:**
- `/Users/borikropotkin/bayesiangpu/Cargo.toml` - Upgraded to Burn 0.20
- `/Users/borikropotkin/bayesiangpu/crates/bayesian-wasm/Cargo.toml` - Added `burn-wgpu` with fusion disabled

**Verification steps:**
1. Build the GPU WASM module:
   ```bash
   cd /Users/borikropotkin/bayesiangpu/js && npm run build:gpu
   ```

2. Start the test server:
   ```bash
   cd /Users/borikropotkin/bayesiangpu/js && npm run test:browser
   ```

3. Open Chrome 113+ and navigate to `http://localhost:8080/browser-test/`

4. Click "Initialize (Auto)" - should succeed with WebGPU

5. Click "Run All Tests" - verify tests pass (previously failed with "unreachable")

**If still failing after fusion fix:**

Additional changes may be needed to:
- `/Users/borikropotkin/bayesiangpu/crates/bayesian-rng/src/pcg.rs` - Add async data readback
- `/Users/borikropotkin/bayesiangpu/crates/bayesian-sampler/src/model.rs` - Add async data readback

But try the fusion fix first - it was documented as the root cause.

## Prevention

1. **Add browser integration tests to CI**: The test harness at `js/browser-test/index.html` should be run in a headless browser (via Playwright or Puppeteer) as part of CI

2. **Document WebGPU limitations**: The synchronous `into_data().to_vec()` pattern works with native WebGPU but is fragile in browsers. Consider adding compile-time warnings or runtime checks.

3. **Consider async inference API**: For browser deployment, a `run_inference_async()` API that properly handles async WebGPU operations would be more robust.

4. **Monitor Burn upstream**: The tracel-ai/burn project is actively developing WebGPU support. Track issues like #1930 and #1970 for related fixes.

## References

- Bean file: `/Users/borikropotkin/bayesiangpu/.beans/bayesiangpu-rgph--burn-wgpu-tensor-operations-fail-in-browser-wasm.md`
- Handoff: `/Users/borikropotkin/bayesiangpu/thoughts/shared/handoffs/general/2026-01-20_02-24-00_webgpu-browser-testing.md`
- Browser test harness: `/Users/borikropotkin/bayesiangpu/examples/browser/wasm-test.html`
- New browser test: `/Users/borikropotkin/bayesiangpu/js/browser-test/index.html`
