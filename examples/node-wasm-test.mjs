/**
 * Node.js WASM test script
 * Tests the bayesian-wasm module in Node.js environment
 */

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load the WASM module manually for Node.js
async function loadWasm() {
  const jsPath = join(__dirname, '../js/pkg/bayesian_wasm.js');
  const wasmPath = join(__dirname, '../js/pkg/bayesian_wasm_bg.wasm');

  // Read JS module as text and evaluate
  const jsSource = await readFile(jsPath, 'utf-8');

  // Read WASM binary
  const wasmBytes = await readFile(wasmPath);

  // Create a minimal DOM shim for wasm-bindgen
  globalThis.TextEncoder = TextEncoder;
  globalThis.TextDecoder = TextDecoder;

  // Compile WASM
  const wasmModule = await WebAssembly.compile(wasmBytes);
  const wasmInstance = await WebAssembly.instantiate(wasmModule, {
    // wasm-bindgen imports are handled by the JS wrapper
  });

  return wasmInstance;
}

async function runTests() {
  console.log('='.repeat(50));
  console.log('BayesianGPU WASM Node.js Test');
  console.log('='.repeat(50));
  console.log();

  let passed = 0;
  let failed = 0;

  // Test 1: WASM compilation
  try {
    const wasmPath = join(__dirname, '../js/pkg/bayesian_wasm_bg.wasm');
    const wasmBytes = await readFile(wasmPath);
    const wasmModule = await WebAssembly.compile(wasmBytes);

    console.log('[PASS] WASM compilation');
    console.log(`       Size: ${wasmBytes.length} bytes`);
    console.log(`       Exports: ${WebAssembly.Module.exports(wasmModule).map(e => e.name).slice(0, 5).join(', ')}...`);
    passed++;
  } catch (e) {
    console.log(`[FAIL] WASM compilation: ${e.message}`);
    failed++;
  }

  // Test 2: Import the JS module dynamically
  try {
    // Dynamic import of the wasm-pack generated module
    const wasm = await import('../js/pkg/bayesian_wasm.js');

    // Initialize with the WASM file
    const wasmPath = join(__dirname, '../js/pkg/bayesian_wasm_bg.wasm');
    const wasmBytes = await readFile(wasmPath);

    await wasm.default(wasmBytes);
    wasm.init();

    console.log('[PASS] Module initialization');
    passed++;

    // Test 3: Version
    try {
      const version = wasm.version();
      console.log(`[PASS] Version: ${version}`);
      passed++;
    } catch (e) {
      console.log(`[FAIL] Version: ${e.message}`);
      failed++;
    }

    // Test 4: Backend type
    try {
      const backend = wasm.get_backend_type();
      console.log(`[PASS] Backend type: ${backend}`);
      passed++;
    } catch (e) {
      console.log(`[FAIL] Backend type: ${e.message}`);
      failed++;
    }

    // Test 5: WebGPU check (will be false in Node)
    try {
      const webgpu = wasm.is_webgpu_available();
      console.log(`[PASS] WebGPU available: ${webgpu} (expected false in Node)`);
      passed++;
    } catch (e) {
      console.log(`[FAIL] WebGPU check: ${e.message}`);
      failed++;
    }

    // Test 6: Backend init
    try {
      const initResult = wasm.init_backend_sync();
      console.log(`[PASS] Backend init: ${initResult}`);
      passed++;
    } catch (e) {
      console.log(`[FAIL] Backend init: ${e.message}`);
      failed++;
    }

    // Test 7: Run inference (Beta-Binomial)
    try {
      // Correct format per lib.rs: priors, likelihood with distribution
      const model = {
        priors: [
          {
            name: "theta",
            distribution: {
              type: "Beta",
              params: { alpha: 2, beta: 2 }
            }
          }
        ],
        likelihood: {
          distribution: {
            type: "Binomial",
            params: { n: 10, p: "theta" }
          },
          observed: [7]
        }
      };

      const config = {
        numSamples: 100,
        numWarmup: 50,
        numChains: 2,
        targetAccept: 0.8,
        seed: 42
      };

      const resultJson = wasm.run_inference(JSON.stringify(model), JSON.stringify(config));
      const result = JSON.parse(resultJson);

      if (result.error) {
        throw new Error(result.error);
      }

      const samples = result.samples?.theta || [];
      const mean = samples.length > 0 ? samples.reduce((a, b) => a + b, 0) / samples.length : NaN;

      console.log(`[PASS] Beta-Binomial inference`);
      console.log(`       Samples: ${samples.length}`);
      console.log(`       Posterior mean: ${mean.toFixed(3)}`);
      console.log(`       Expected: ~0.67 (Beta(9,5) posterior)`);
      console.log(`       R-hat: ${result.diagnostics?.rhat?.theta?.toFixed(3) || 'N/A'}`);
      console.log(`       ESS: ${result.diagnostics?.ess?.theta?.toFixed(0) || 'N/A'}`);
      passed++;
    } catch (e) {
      console.log(`[FAIL] Inference: ${e.message}`);
      failed++;
    }

  } catch (e) {
    console.log(`[FAIL] Module initialization: ${e.message}`);
    console.log(`       Stack: ${e.stack}`);
    failed++;
  }

  console.log();
  console.log('='.repeat(50));
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log('='.repeat(50));

  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(e => {
  console.error('Test runner error:', e);
  process.exit(1);
});
