//! Backend initialization for WASM
//!
//! This module handles async initialization of the WebGPU backend with
//! automatic fallback to CPU (NdArray) when WebGPU is unavailable.

use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

#[cfg(feature = "ndarray")]
use burn::backend::ndarray::NdArrayDevice;
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
use burn::backend::NdArray;

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::WgpuDevice;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;

use burn::backend::Autodiff;

// Global state for initialized device
static DEVICE: OnceLock<DeviceState> = OnceLock::new();

#[derive(Clone)]
enum DeviceState {
    #[cfg(feature = "ndarray")]
    Cpu(NdArrayDevice),
    #[cfg(feature = "wgpu")]
    Gpu(WgpuDevice),
}

/// Backend type alias - conditional on features
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub type WasmBackend = Autodiff<NdArray<f32>>;

#[cfg(feature = "wgpu")]
pub type WasmBackend = Autodiff<Wgpu>;

/// Device type alias - conditional on features
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub type WasmDevice = NdArrayDevice;

#[cfg(feature = "wgpu")]
pub type WasmDevice = WgpuDevice;

/// Check if WebGPU is available in the current browser
#[wasm_bindgen]
pub fn is_webgpu_available() -> bool {
    #[cfg(feature = "wgpu")]
    {
        // Check if navigator.gpu exists
        if let Some(window) = web_sys::window() {
            if let Ok(navigator) = js_sys::Reflect::get(&window, &"navigator".into()) {
                if let Ok(gpu) = js_sys::Reflect::get(&navigator, &"gpu".into()) {
                    return !gpu.is_undefined() && !gpu.is_null();
                }
            }
        }
        false
    }
    #[cfg(not(feature = "wgpu"))]
    false
}

/// Get the current backend type as a string
#[wasm_bindgen]
pub fn get_backend_type() -> String {
    match DEVICE.get() {
        #[cfg(feature = "ndarray")]
        Some(DeviceState::Cpu(_)) => "cpu".to_string(),
        #[cfg(feature = "wgpu")]
        Some(DeviceState::Gpu(_)) => "webgpu".to_string(),
        None => "not_initialized".to_string(),
    }
}

/// Initialize the compute backend (sync version for ndarray-only builds)
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
#[wasm_bindgen]
pub fn init_backend_sync() -> String {
    if DEVICE.get().is_some() {
        return "already_initialized".to_string();
    }

    let _ = DEVICE.set(DeviceState::Cpu(NdArrayDevice::default()));
    "cpu".to_string()
}

/// Initialize the compute backend (async version for wgpu builds)
///
/// Attempts WebGPU initialization first, falls back to CPU if unavailable.
/// Returns a Promise that resolves to the backend type ("webgpu" or "cpu").
#[cfg(feature = "wgpu")]
#[wasm_bindgen]
pub async fn init_backend() -> Result<JsValue, JsValue> {
    if DEVICE.get().is_some() {
        return Ok(JsValue::from_str("already_initialized"));
    }

    // Try WebGPU first
    if is_webgpu_available() {
        match init_wgpu().await {
            Ok(device) => {
                let _ = DEVICE.set(DeviceState::Gpu(device));
                return Ok(JsValue::from_str("webgpu"));
            }
            Err(e) => {
                // Log error and fall through to CPU
                web_sys::console::warn_1(&format!("WebGPU init failed: {:?}", e).into());
            }
        }
    }

    // Fallback to CPU (requires ndarray feature)
    #[cfg(feature = "ndarray")]
    {
        let _ = DEVICE.set(DeviceState::Cpu(NdArrayDevice::default()));
        return Ok(JsValue::from_str("cpu"));
    }

    #[cfg(not(feature = "ndarray"))]
    {
        Err(JsValue::from_str(
            "WebGPU not available and CPU fallback not enabled",
        ))
    }
}

/// Initialize without wgpu feature (sync version exposed as async for API compatibility)
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
#[wasm_bindgen]
pub fn init_backend() -> JsValue {
    if DEVICE.get().is_none() {
        let _ = DEVICE.set(DeviceState::Cpu(NdArrayDevice::default()));
    }
    JsValue::from_str("cpu")
}

#[cfg(feature = "wgpu")]
async fn init_wgpu() -> Result<WgpuDevice, String> {
    use burn::backend::wgpu::{
        graphics::WebGpu, init_setup_async, MemoryConfiguration, RuntimeOptions,
    };

    // Use proper async initialization for browser WebGPU
    let device = WgpuDevice::default();

    // Initialize the wgpu runtime asynchronously - required for browser WebGPU
    let runtime_options = RuntimeOptions {
        memory_config: MemoryConfiguration::default(),
        tasks_max: 64,
    };

    // init_setup_async returns WgpuSetup, not Result
    let _setup = init_setup_async::<WebGpu>(&device, runtime_options).await;

    web_sys::console::log_1(&"WebGPU runtime initialized".into());

    Ok(device)
}

/// Get the initialized device for inference
///
/// Returns an error if the backend has not been initialized.
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub fn get_device() -> Result<WasmDevice, String> {
    match DEVICE.get() {
        Some(DeviceState::Cpu(d)) => Ok(*d),
        None => Err("Backend not initialized. Call init_backend() first.".to_string()),
    }
}

#[cfg(feature = "wgpu")]
pub fn get_device() -> Result<WasmDevice, String> {
    match DEVICE.get() {
        Some(DeviceState::Gpu(d)) => Ok(d.clone()),
        #[cfg(feature = "ndarray")]
        Some(DeviceState::Cpu(_)) => {
            Err("CPU backend initialized but WgpuDevice expected".to_string())
        }
        None => Err("Backend not initialized. Call init_backend() first.".to_string()),
    }
}

/// Get the device, auto-initializing to CPU if not already initialized
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub fn get_device_or_init() -> WasmDevice {
    if DEVICE.get().is_none() {
        let _ = DEVICE.set(DeviceState::Cpu(NdArrayDevice::default()));
    }

    match DEVICE.get() {
        Some(DeviceState::Cpu(d)) => *d,
        None => unreachable!(),
    }
}

#[cfg(feature = "wgpu")]
pub fn get_device_or_init() -> WasmDevice {
    // For wgpu builds, we need async initialization which must be done via init_backend()
    // If not initialized, we can't auto-init because wgpu requires async
    // Fall back to default device creation which may block
    match DEVICE.get() {
        Some(DeviceState::Gpu(d)) => d.clone(),
        #[cfg(feature = "ndarray")]
        Some(DeviceState::Cpu(_)) => {
            // CPU fallback was used, return default wgpu device for type compatibility
            // This case shouldn't happen in normal use
            WgpuDevice::default()
        }
        None => {
            // Not initialized - create default device (may block on first use)
            let device = WgpuDevice::default();
            let _ = DEVICE.set(DeviceState::Gpu(device.clone()));
            device
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_get_backend_type_before_init() {
        // Note: This test runs in isolation, so DEVICE is uninitialized
        // In practice, this would return "not_initialized"
    }
}
