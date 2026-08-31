use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: Option<u32>,
    pub source: String,
}

pub fn probe_gpu() -> GpuInfo {
    if let Ok(raw) = std::env::var("WHISBURN_GPU_VRAM_MB") {
        if let Ok(mb) = raw.parse::<u32>() {
            return GpuInfo {
                name: std::env::var("WHISBURN_GPU_NAME")
                    .unwrap_or_else(|_| "configured".into()),
                vram_mb: Some(mb),
                source: "env".into(),
            };
        }
    }

    #[cfg(windows)]
    {
        if let Some((name, vram_mb)) = probe_windows_gpu() {
            return GpuInfo {
                name,
                vram_mb: Some(vram_mb),
                source: "wmi".into(),
            };
        }
    }

    if let Some((name, vram_mb)) = probe_wgpu_adapter() {
        return GpuInfo {
            name,
            vram_mb,
            source: "wgpu".into(),
        };
    }

    GpuInfo {
        name: "unknown".into(),
        vram_mb: None,
        source: "unknown".into(),
    }
}

fn probe_wgpu_adapter() -> Option<(String, Option<u32>)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let info = adapter.get_info();
    let name = info.name.clone();
    let vram_mb = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => Some(8192),
        wgpu::DeviceType::IntegratedGpu => Some(2048),
        wgpu::DeviceType::VirtualGpu => Some(4096),
        wgpu::DeviceType::Cpu => Some(0),
        _ => None,
    };
    Some((name, vram_mb))
}

#[cfg(windows)]
fn probe_windows_gpu() -> Option<(String, u32)> {
    use std::os::windows::process::CommandExt;
    use std::process::Command;

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
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let (name, mb) = text.rsplit_once('|')?;
    let vram_mb: u32 = mb.trim().parse().ok()?;
    if vram_mb == 0 {
        return None;
    }
    Some((name.trim().to_string(), vram_mb))
}

#[cfg(not(windows))]
fn probe_windows_gpu() -> Option<(String, u32)> {
    None
}