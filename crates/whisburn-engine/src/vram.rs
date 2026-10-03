//! GPU VRAM budget so inference cannot starve the display compositor.
//!
//! Remaining VRAM is compared to `max(reserve_mb, total * reserve_percent / 100)`.
//! When the model itself would breach that floor, the runtime offloads to the
//! ndarray CPU backend. When the model fits but the clip is long enough to push
//! activations into the danger zone, the runtime keeps the GPU and splits audio.

use std::process::Command;

/// Default floor: 1 GiB remaining, or 15% of the card, whichever is larger.
pub const DEFAULT_RESERVE_MB: u32 = 1024;
pub const DEFAULT_RESERVE_PERCENT: u32 = 15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramPolicy {
    /// Keep at least this many MiB free on the GPU.
    pub reserve_mb: u32,
    /// Keep at least this percentage of total VRAM free (0–100).
    pub reserve_percent: u32,
}

impl Default for VramPolicy {
    fn default() -> Self {
        Self {
            reserve_mb: DEFAULT_RESERVE_MB,
            reserve_percent: DEFAULT_RESERVE_PERCENT,
        }
    }
}

impl VramPolicy {
    pub fn from_env() -> Self {
        Self {
            reserve_mb: env_u32("WHISBURN_VRAM_RESERVE_MB").unwrap_or(DEFAULT_RESERVE_MB),
            reserve_percent: env_u32("WHISBURN_VRAM_RESERVE_PERCENT")
                .unwrap_or(DEFAULT_RESERVE_PERCENT)
                .min(100),
        }
    }

    pub fn disabled() -> Self {
        Self {
            reserve_mb: 0,
            reserve_percent: 0,
        }
    }

    /// MiB that must stay free for this card size.
    pub fn reserve_mb_for(&self, total_mb: u32) -> u32 {
        let from_pct = (u64::from(total_mb) * u64::from(self.reserve_percent) / 100) as u32;
        self.reserve_mb.max(from_pct)
    }

    /// True when loading `extra_needed_mb` more would leave too little VRAM.
    pub fn should_offload(&self, snap: &VramSnapshot, extra_needed_mb: u32) -> bool {
        if self.reserve_mb == 0 && self.reserve_percent == 0 {
            return false;
        }
        let Some(total) = snap.total_mb.filter(|t| *t > 0) else {
            return false;
        };
        let reserve = self.reserve_mb_for(total);
        let needed = reserve.saturating_add(extra_needed_mb);
        if let Some(free) = snap.free_mb {
            return free < needed;
        }
        if let Some(used) = snap.used_mb {
            return used.saturating_add(extra_needed_mb) + reserve > total;
        }
        extra_needed_mb + reserve > total
    }
}

/// Shortest GPU slice worth running (matches Parakeet’s ~15 s encoder window).
pub const MIN_GPU_CHUNK_SEC: f64 = 15.0;
/// Longest GPU slice before the cubecl pool is reset, even on a large card.
pub const MAX_GPU_CHUNK_SEC: f64 = 120.0;
const ACTIVATION_MB_PER_SEC: f64 = 12.0;

/// Workspace for one encoder window, on top of loaded weights.
pub fn window_workspace_mb(model_mb: u32) -> u32 {
    (model_mb / 4).max(128)
}

/// Extra VRAM (MiB) on top of weights for one pass of `audio_seconds`.
/// CubeCL/wgpu pools retain workspace from each window until cleanup.
pub fn activation_mb(audio_seconds: f64, model_mb: u32) -> u32 {
    window_workspace_mb(model_mb)
        .saturating_add((audio_seconds.max(0.0) * ACTIVATION_MB_PER_SEC).ceil() as u32)
}

/// How to run this clip without crossing the VRAM reserve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuAudioPlan {
    /// Model (plus one window) would breach the reserve — use CPU.
    Cpu,
    /// Whole clip fits in the budget.
    SinglePass,
    /// Keep the GPU; process `chunk_sec` at a time.
    Split { chunk_sec: f64 },
}

/// Decide GPU / split / CPU from current VRAM, model size, and audio length.
pub fn plan_gpu_audio(
    policy: &VramPolicy,
    snap: &VramSnapshot,
    model_mb: u32,
    audio_seconds: f64,
    model_already_on_gpu: bool,
) -> GpuAudioPlan {
    if policy.reserve_mb == 0 && policy.reserve_percent == 0 {
        return GpuAudioPlan::SinglePass;
    }
    let extra_model = if model_already_on_gpu { 0 } else { model_mb };
    let window = window_workspace_mb(model_mb);
    if policy.should_offload(snap, extra_model.saturating_add(window)) {
        return GpuAudioPlan::Cpu;
    }
    let extra_full = extra_model.saturating_add(activation_mb(audio_seconds, model_mb));
    if !policy.should_offload(snap, extra_full) {
        if audio_seconds > MAX_GPU_CHUNK_SEC {
            return GpuAudioPlan::Split {
                chunk_sec: MAX_GPU_CHUNK_SEC,
            };
        }
        return GpuAudioPlan::SinglePass;
    }
    let chunk_sec = max_safe_chunk_seconds(policy, snap, extra_model, model_mb)
        .clamp(MIN_GPU_CHUNK_SEC, MAX_GPU_CHUNK_SEC);
    if audio_seconds <= chunk_sec {
        GpuAudioPlan::SinglePass
    } else {
        GpuAudioPlan::Split { chunk_sec }
    }
}

fn max_safe_chunk_seconds(
    policy: &VramPolicy,
    snap: &VramSnapshot,
    extra_model: u32,
    model_mb: u32,
) -> f64 {
    let Some(total) = snap.total_mb.filter(|t| *t > 0) else {
        return MAX_GPU_CHUNK_SEC;
    };
    let reserve = policy.reserve_mb_for(total);
    let free = snap.free_mb.or_else(|| snap.used_mb.map(|u| total.saturating_sub(u)));
    let Some(free) = free else {
        return MAX_GPU_CHUNK_SEC;
    };
    let budget = free.saturating_sub(reserve).saturating_sub(extra_model);
    let window = window_workspace_mb(model_mb);
    if budget <= window {
        return MIN_GPU_CHUNK_SEC;
    }
    (budget - window) as f64 / ACTIVATION_MB_PER_SEC
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VramSnapshot {
    pub name: Option<String>,
    pub total_mb: Option<u32>,
    pub used_mb: Option<u32>,
    pub free_mb: Option<u32>,
    pub source: &'static str,
}

pub fn probe() -> VramSnapshot {
    if let Some(snap) = probe_env() {
        return snap;
    }
    if let Some(snap) = probe_nvidia_smi() {
        return snap;
    }
    if let Some(snap) = probe_sysfs_vram() {
        return snap;
    }
    #[cfg(windows)]
    if let Some(snap) = probe_windows_wmi() {
        return snap;
    }
    VramSnapshot {
        source: "unknown",
        ..VramSnapshot::default()
    }
}

pub fn looks_like_gpu_oom(msg: &str) -> bool {
    looks_like_gpu_failure(msg)
}

/// OOM, dead wgpu adapter, or a cubecl worker that already panicked.
pub fn looks_like_gpu_failure(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("out of memory")
        || m.contains("failed to allocate")
        || m.contains("nvkms")
        || m.contains("memory page")
        || m.contains("egl_bad_alloc")
        || m.contains("wgpu error")
        || m.contains("no possible adapter")
        || m.contains("recverror")
        || m.contains("device lost")
        || m.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|w| w == "oom")
}

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn probe_env() -> Option<VramSnapshot> {
    let total_mb = env_u32("WHISBURN_GPU_VRAM_TOTAL_MB").or_else(|| env_u32("WHISBURN_GPU_VRAM_MB"));
    let free_mb = env_u32("WHISBURN_GPU_VRAM_FREE_MB");
    let used_mb = env_u32("WHISBURN_GPU_VRAM_USED_MB").or_else(|| match (total_mb, free_mb) {
        (Some(t), Some(f)) => Some(t.saturating_sub(f)),
        _ => None,
    });
    if total_mb.is_none() && free_mb.is_none() && used_mb.is_none() {
        return None;
    }
    Some(VramSnapshot {
        name: std::env::var("WHISBURN_GPU_NAME").ok(),
        total_mb,
        used_mb,
        free_mb: free_mb.or_else(|| match (total_mb, used_mb) {
            (Some(t), Some(u)) => Some(t.saturating_sub(u)),
            _ => None,
        }),
        source: "env",
    })
}

fn probe_nvidia_smi() -> Option<VramSnapshot> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut best: Option<VramSnapshot> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let total: u32 = parts[1].parse().ok()?;
        let used: u32 = parts[2].parse().ok()?;
        let free: u32 = parts[3].parse().ok()?;
        let snap = VramSnapshot {
            name: Some(parts[0].to_string()),
            total_mb: Some(total),
            used_mb: Some(used),
            free_mb: Some(free),
            source: "nvidia-smi",
        };
        let better = match &best {
            None => true,
            Some(prev) => prev.total_mb.unwrap_or(0) < total,
        };
        if better {
            best = Some(snap);
        }
    }
    best
}

/// MiB this process still has allocated, from `nvidia-smi` compute-apps.
/// `None` when the tool is missing or this pid is not listed.
pub fn process_vram_mb() -> Option<u32> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_process_vram_mb(&String::from_utf8_lossy(&out.stdout), std::process::id())
}

pub fn parse_process_vram_mb(csv: &str, pid: u32) -> Option<u32> {
    let mut total = 0u32;
    let mut found = false;
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',').map(str::trim);
        let row_pid: u32 = parts.next()?.parse().ok()?;
        let mb: u32 = parts.next()?.split_whitespace().next()?.parse().ok()?;
        if row_pid == pid {
            found = true;
            total = total.saturating_add(mb);
        }
    }
    found.then_some(total)
}

fn probe_sysfs_vram() -> Option<VramSnapshot> {
    let drm = std::fs::read_dir("/sys/class/drm").ok()?;
    let mut best: Option<VramSnapshot> = None;
    for ent in drm.flatten() {
        let name = ent.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = ent.path().join("device");
        let total = read_sysfs_mb(&dev.join("mem_info_vram_total"))?;
        let used = read_sysfs_mb(&dev.join("mem_info_vram_used"));
        let snap = VramSnapshot {
            name: Some(name.into_owned()),
            total_mb: Some(total),
            used_mb: used,
            free_mb: used.map(|u| total.saturating_sub(u)),
            source: "sysfs",
        };
        let better = match &best {
            None => true,
            Some(prev) => prev.total_mb.unwrap_or(0) < total,
        };
        if better {
            best = Some(snap);
        }
    }
    best
}

fn read_sysfs_mb(path: &std::path::Path) -> Option<u32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let bytes: u64 = raw.trim().parse().ok()?;
    Some((bytes / (1024 * 1024)) as u32)
}

#[cfg(windows)]
fn probe_windows_wmi() -> Option<VramSnapshot> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    let script = r#"
$gpu = Get-CimInstance Win32_VideoController |
  Where-Object { $_.AdapterRAM -and $_.AdapterRAM -gt 0 } |
  Sort-Object AdapterRAM -Descending |
  Select-Object -First 1
if ($null -eq $gpu) { exit 1 }
Write-Output ($gpu.Name + '|' + [int]($gpu.AdapterRAM / 1MB))
"#;
    let out = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let (name, mb) = text.rsplit_once('|')?;
    let total_mb: u32 = mb.trim().parse().ok()?;
    if total_mb == 0 {
        return None;
    }
    Some(VramSnapshot {
        name: Some(name.trim().to_string()),
        total_mb: Some(total_mb),
        used_mb: None,
        free_mb: None,
        source: "wmi",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_uses_the_larger_of_mb_and_percent() {
        let p = VramPolicy {
            reserve_mb: 1024,
            reserve_percent: 15,
        };
        assert_eq!(p.reserve_mb_for(8192), 1228); // 15% of 8192 = 1228.8 → 1228
        assert_eq!(p.reserve_mb_for(4096), 1024); // 15% = 614, mb wins
        assert_eq!(p.reserve_mb_for(24576), 3686); // 15% of 24 GiB
    }

    #[test]
    fn offload_when_free_below_reserve() {
        let p = VramPolicy {
            reserve_mb: 1024,
            reserve_percent: 15,
        };
        let snap = VramSnapshot {
            total_mb: Some(8192),
            free_mb: Some(510),
            used_mb: Some(7682),
            source: "test",
            name: None,
        };
        assert!(p.should_offload(&snap, 0));
        assert!(p.should_offload(&snap, 1800));
    }

    #[test]
    fn stay_on_gpu_when_enough_headroom() {
        let p = VramPolicy::default();
        let snap = VramSnapshot {
            total_mb: Some(8192),
            free_mb: Some(4000),
            used_mb: Some(4192),
            source: "test",
            name: None,
        };
        assert!(!p.should_offload(&snap, 0));
        assert!(!p.should_offload(&snap, 1800));
        assert!(p.should_offload(&snap, 3500));
    }

    #[test]
    fn disabled_policy_never_offloads() {
        let p = VramPolicy::disabled();
        let snap = VramSnapshot {
            total_mb: Some(8192),
            free_mb: Some(0),
            used_mb: Some(8192),
            source: "test",
            name: None,
        };
        assert!(!p.should_offload(&snap, 9999));
    }

    #[test]
    fn oom_detector_matches_wgpu_and_nvkms() {
        assert!(looks_like_gpu_oom("wgpu error: Out of Memory"));
        assert!(looks_like_gpu_oom(
            "Failed to allocate NVKMS memory for GEM object"
        ));
        assert!(looks_like_gpu_oom("couldn't find resource: Memory page 0"));
        assert!(looks_like_gpu_failure("No possible adapter available for backend"));
        assert!(looks_like_gpu_oom("cuda OOM while allocating buffer"));
        assert!(!looks_like_gpu_oom("unknown model"));
        assert!(!looks_like_gpu_oom("failed to open audio file"));
    }

    fn snap_8g(free_mb: u32) -> VramSnapshot {
        VramSnapshot {
            total_mb: Some(8192),
            free_mb: Some(free_mb),
            used_mb: Some(8192 - free_mb),
            source: "test",
            name: None,
        }
    }

    #[test]
    fn long_audio_is_split_instead_of_cpu() {
        let p = VramPolicy::default();
        let snap = snap_8g(4000);
        assert_eq!(
            plan_gpu_audio(&p, &snap, 1200, 30.0, false),
            GpuAudioPlan::SinglePass
        );
        match plan_gpu_audio(&p, &snap, 1200, 3600.0, false) {
            GpuAudioPlan::Split { chunk_sec } => {
                assert!(chunk_sec >= MIN_GPU_CHUNK_SEC);
                assert!(chunk_sec <= MAX_GPU_CHUNK_SEC);
            }
            other => panic!("expected split, got {other:?}"),
        }
    }

    #[test]
    fn exhausted_card_offloads_even_short_audio() {
        let p = VramPolicy::default();
        assert_eq!(
            plan_gpu_audio(&p, &snap_8g(510), 1200, 10.0, false),
            GpuAudioPlan::Cpu
        );
    }

    #[test]
    fn process_vram_parser_sums_this_pid() {
        let csv = "1147, 43\n316764, 6917\n316764, 10\n";
        assert_eq!(parse_process_vram_mb(csv, 316764), Some(6927));
        assert_eq!(parse_process_vram_mb(csv, 1), None);
    }

    #[test]
    fn activation_grows_with_duration() {
        assert!(activation_mb(60.0, 1200) > activation_mb(15.0, 1200));
        assert!(activation_mb(0.0, 1200) >= 128);
    }
}
