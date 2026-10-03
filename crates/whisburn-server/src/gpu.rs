use serde::Serialize;
use whisburn_engine::vram::{probe, VramSnapshot};

#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_used_mb: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_free_mb: Option<u32>,
    pub source: String,
}

pub fn probe_gpu() -> GpuInfo {
    if let Ok(raw) = std::env::var("WHISBURN_GPU_VRAM_MB") {
        if let Ok(mb) = raw.parse::<u32>() {
            let snap = probe();
            return GpuInfo {
                name: std::env::var("WHISBURN_GPU_NAME")
                    .unwrap_or_else(|_| "configured".into()),
                vram_mb: Some(mb),
                vram_used_mb: snap.used_mb,
                vram_free_mb: snap.free_mb.or_else(|| {
                    snap.used_mb.map(|u| mb.saturating_sub(u))
                }),
                source: "env".into(),
            };
        }
    }

    let snap = probe();
    if snap.total_mb.is_some() || snap.free_mb.is_some() {
        return GpuInfo::from_snapshot(snap);
    }

    if let Some((name, vram_mb)) = probe_wgpu_adapter() {
        return GpuInfo {
            name,
            vram_mb,
            vram_used_mb: None,
            vram_free_mb: None,
            source: "wgpu".into(),
        };
    }

    GpuInfo {
        name: "unknown".into(),
        vram_mb: None,
        vram_used_mb: None,
        vram_free_mb: None,
        source: "unknown".into(),
    }
}

impl GpuInfo {
    fn from_snapshot(snap: VramSnapshot) -> Self {
        Self {
            name: snap.name.unwrap_or_else(|| "unknown".into()),
            vram_mb: snap.total_mb,
            vram_used_mb: snap.used_mb,
            vram_free_mb: snap.free_mb,
            source: snap.source.to_string(),
        }
    }
}

fn probe_wgpu_adapter() -> Option<(String, Option<u32>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
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