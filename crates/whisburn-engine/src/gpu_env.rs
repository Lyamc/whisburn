//! Make `libvulkan.so.1` visible before the first wgpu `dlopen`.
//!
//! On NixOS the Vulkan loader lives in a store path that `ldconfig` does not
//! know. A cargo binary then starts wgpu with `active_backends` empty and
//! cubecl panics in `request_adapter`. glibc reads `LD_LIBRARY_PATH` once, at
//! process start, so setting it here does not change a later `dlopen` of the
//! soname. Loading the absolute `libvulkan.so.1` with `RTLD_GLOBAL` puts that
//! soname in the global namespace, and wgpu's own `dlopen("libvulkan.so.1")`
//! returns the same object. The NixOS loader then searches
//! `/run/opengl-driver/share` for ICDs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock};

static ADAPTER_MISSING: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuRuntimePrep {
    /// `libvulkan.so.1` can be loaded. `added` is true when this process had to
    /// prepend a directory to `LD_LIBRARY_PATH`.
    Ready {
        loader: Option<PathBuf>,
        added: bool,
    },
    /// No loader. Callers must stay on Flex and must not construct a wgpu device.
    Missing,
}

pub fn prepare_gpu_runtime() -> &'static GpuRuntimePrep {
    static STATUS: OnceLock<GpuRuntimePrep> = OnceLock::new();
    STATUS.get_or_init(|| {
        install_adapter_panic_hook();
        #[cfg(target_os = "linux")]
        {
            return linux::prepare();
        }
        #[cfg(not(target_os = "linux"))]
        GpuRuntimePrep::Ready {
            loader: None,
            added: false,
        }
    })
}

pub fn log_gpu_runtime() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| match prepare_gpu_runtime() {
        GpuRuntimePrep::Ready {
            loader: Some(loader),
            ..
        } => {
            tracing::info!(loader = %loader.display(), "preloaded libvulkan.so.1");
        }
        GpuRuntimePrep::Ready { loader: None, .. } => {
            tracing::debug!("Vulkan loader already visible");
        }
        GpuRuntimePrep::Missing => {
            tracing::error!(
                "libvulkan.so.1 was not found; GPU inference stays on Flex. Set WHISBURN_VULKAN_LIBDIR to the directory that contains it."
            );
        }
    });
}

/// Set from the panic hook when cubecl reports that wgpu created no adapter.
pub fn adapter_missing() -> bool {
    ADAPTER_MISSING.load(Ordering::SeqCst)
}

pub fn loader_ready() -> bool {
    matches!(prepare_gpu_runtime(), GpuRuntimePrep::Ready { .. })
}

fn install_adapter_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let rendered = info.to_string();
        if missing_adapter_message(&rendered) {
            ADAPTER_MISSING.store(true, Ordering::SeqCst);
        }
        prev(info);
    }));
}

pub(crate) fn missing_adapter_message(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("no possible adapter") || m.contains("no adapter available")
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{prepend_path_env, vulkan_loader_dirs_in_text, GpuRuntimePrep};
    use std::path::{Path, PathBuf};

    const ICD_SHARE: &str = "/run/opengl-driver/share";
    const DRIVER_LIB: &str = "/run/opengl-driver/lib";
    const WRAPPERS: &[&str] = &[
        "/run/current-system/sw/bin/whisburn",
        "/run/current-system/sw/bin/firefox",
        "/run/current-system/sw/bin/servoshell",
    ];

    pub fn prepare() -> GpuRuntimePrep {
        expose_icd_share();
        prepend_path_env("LD_LIBRARY_PATH", DRIVER_LIB);

        if let Some(dir) = std::env::var_os("WHISBURN_VULKAN_LIBDIR") {
            if let Some(prep) = activate(PathBuf::from(dir)) {
                return prep;
            }
        }

        if let Some(dir) = loader_on_library_path() {
            if let Some(prep) = activate(dir) {
                return prep;
            }
        }

        if let Some(dir) = loader_from_wrappers() {
            if let Some(prep) = activate(dir) {
                return prep;
            }
        }

        if ldconfig_has_vulkan() {
            return GpuRuntimePrep::Ready {
                loader: None,
                added: false,
            };
        }

        GpuRuntimePrep::Missing
    }

    fn expose_icd_share() {
        if std::env::var_os("VK_DRIVER_FILES").is_some()
            || std::env::var_os("VK_ICD_FILENAMES").is_some()
        {
            return;
        }
        let icd = Path::new(ICD_SHARE).join("vulkan/icd.d");
        let has_json = std::fs::read_dir(&icd)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .any(|ent| ent.path().extension().and_then(|s| s.to_str()) == Some("json"));
        if has_json {
            prepend_path_env("XDG_DATA_DIRS", ICD_SHARE);
        }
    }

    fn activate(dir: PathBuf) -> Option<GpuRuntimePrep> {
        if !loader_in(&dir) || !preload_vulkan(&dir) {
            return None;
        }
        let added = prepend_path_env("LD_LIBRARY_PATH", &dir.to_string_lossy());
        Some(GpuRuntimePrep::Ready {
            loader: Some(dir),
            added,
        })
    }

    fn loader_in(dir: &Path) -> bool {
        dir.join("libvulkan.so.1").is_file()
    }

    /// Map `libvulkan.so.1` into the global namespace so a later soname
    /// `dlopen` from wgpu finds it. glibc will not notice `LD_LIBRARY_PATH`
    /// changes made after this process started.
    fn preload_vulkan(dir: &Path) -> bool {
        extern "C" {
            fn dlopen(filename: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
        }
        const RTLD_NOW: i32 = 2;
        const RTLD_GLOBAL: i32 = 0x00100;

        let lib = dir.join("libvulkan.so.1");
        let Some(text) = lib.to_str() else {
            return false;
        };
        let Ok(path) = std::ffi::CString::new(text) else {
            return false;
        };
        // Safety: `path` is a NUL-terminated absolute path. The handle is kept
        // for the process lifetime; `dlclose` would unmap the loader wgpu uses.
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        !handle.is_null()
    }

    fn loader_on_library_path() -> Option<PathBuf> {
        let path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        path.split(':')
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from)
            .find(|dir| loader_in(dir))
    }

    fn loader_from_wrappers() -> Option<PathBuf> {
        for path in WRAPPERS {
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            for dir in vulkan_loader_dirs_in_text(&text) {
                let dir = PathBuf::from(dir);
                if loader_in(&dir) {
                    return Some(dir);
                }
            }
        }
        None
    }

    fn ldconfig_has_vulkan() -> bool {
        let Ok(output) = std::process::Command::new("ldconfig").arg("-p").output() else {
            return false;
        };
        if !output.status.success() {
            return false;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        text.lines().any(|line| line.contains("libvulkan.so.1"))
    }
}

/// Directories mentioned next to a `vulkan-loader` store path.
pub(crate) fn vulkan_loader_dirs_in_text(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(idx) = rest.find("vulkan-loader") {
        let head = &rest[..idx];
        let start = head.rfind("/nix/store/").unwrap_or(0);
        let tail = &rest[idx..];
        let end_rel = tail
            .find(|c: char| !is_store_path_char(c))
            .unwrap_or(tail.len());
        let raw = &rest[start..idx + end_rel];
        for suffix in ["/lib64", "/lib"] {
            if let Some(at) = raw.find(suffix) {
                let dir = raw[..at + suffix.len()].to_string();
                if dir.contains("vulkan-loader") && !out.iter().any(|seen: &String| seen == &dir) {
                    out.push(dir);
                }
            }
        }
        rest = &rest[idx + "vulkan-loader".len()..];
    }
    out
}

fn is_store_path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '+' | '-')
}

/// Prepend `dir` to a `:`-separated environment variable. Returns true when the
/// value changed.
fn prepend_path_env(key: &str, dir: &str) -> bool {
    if dir.is_empty() || !Path::new(dir).is_dir() {
        return false;
    }
    let current = std::env::var(key).unwrap_or_default();
    if current.split(':').any(|part| part == dir) {
        return false;
    }
    let next = if current.is_empty() {
        dir.to_string()
    } else {
        format!("{dir}:{current}")
    };
    // Safety: one-time startup write, before wgpu threads call `dlopen`.
    // `set_var` is unsafe because a concurrent `getenv` is a data race.
    unsafe { std::env::set_var(key, next) };
    true
}

#[cfg(test)]
mod tests {
    use super::vulkan_loader_dirs_in_text;

    #[test]
    fn parses_nix_wrapper_loader_dirs() {
        let text = "\
LD_LIBRARY_PATH='/nix/store/vjdqsn0bz2llabfqa746zfgk4kmswba5-vulkan-loader-1.4.357.0/lib64'$LD_LIBRARY_PATH
LD_LIBRARY_PATH='/nix/store/vjdqsn0bz2llabfqa746zfgk4kmswba5-vulkan-loader-1.4.357.0/lib'$LD_LIBRARY_PATH
";
        let dirs = vulkan_loader_dirs_in_text(text);
        assert!(dirs
            .iter()
            .any(|d| d.ends_with("vulkan-loader-1.4.357.0/lib")));
        assert!(dirs
            .iter()
            .any(|d| d.ends_with("vulkan-loader-1.4.357.0/lib64")));
    }

    #[test]
    fn missing_adapter_message_matches_cubecl_panic() {
        let msg = "No possible adapter available for backend. Falling back to first available.: NotFound { active_backends: Backends(0x0) }";
        assert!(super::missing_adapter_message(msg));
        assert!(!super::missing_adapter_message("RecvError"));
    }
}
