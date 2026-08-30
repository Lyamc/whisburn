#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;
#[cfg(target_arch = "wasm32")]
pub use web::*;

#[derive(Debug, Clone)]
pub struct AudioFile {
    pub name: String,
    pub bytes: Vec<u8>,
    #[cfg(not(target_arch = "wasm32"))]
    pub path: std::path::PathBuf,
}