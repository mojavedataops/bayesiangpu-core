# BayesianGPU Browser Test Harness

This directory contains a browser-based test harness for testing the WebGPU backend.

## Quick Start

1. Build the WebGPU WASM module:
   ```bash
   cd js
   npm run build:gpu
   ```

2. Start the test server:
   ```bash
   npm run test:browser
   ```

3. Open http://localhost:8080/browser-test/ in your browser

## Browser Requirements

### Chrome (Recommended)
- Chrome 113+ has WebGPU enabled by default
- Older versions: enable via `chrome://flags/#enable-unsafe-webgpu`

### Edge
- Edge 113+ has WebGPU enabled by default

### Firefox
- WebGPU is behind a flag
- Enable via `about:config` -> `dom.webgpu.enabled` = true
- Requires Firefox Nightly or Developer Edition for best support

### Safari
- Safari 17+ (macOS Sonoma) has WebGPU support
- Enable via Develop menu -> Feature Flags -> WebGPU

## Test Cases

The test harness includes:

1. **Simple Beta-Binomial**: Basic conjugate model test
2. **Multi-parameter (Gamma-Poisson)**: Tests multi-dimensional inference
3. **Performance Benchmark**: Tests larger sampling runs for performance comparison

## What Gets Tested

- WebGPU API availability detection
- GPU adapter and device creation
- WASM module loading and initialization
- Backend selection (WebGPU vs CPU fallback)
- Inference correctness
- Diagnostic computation (R-hat, ESS)

## Interpreting Results

- **Green badges**: Test passed
- **Red badges**: Test failed
- Check the Diagnostic Log for detailed information

### Expected Behavior

- If WebGPU is available, the backend should initialize to "webgpu"
- If WebGPU is not available, it falls back to "cpu"
- All inference tests should pass regardless of backend
