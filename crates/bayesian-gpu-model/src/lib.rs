mod model;
pub use model::*;
// Re-export GPU types so consumers don't need bayesian-wasm directly
pub use bayesian_wasm::gpu::sync::GpuContextSync;
pub use bayesian_wasm::gpu::PersistentGpuBuffers;
