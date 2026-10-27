//! Time-based unload of cached speech models when the server is idle.
//!
//! Default policy: keep the last-loaded model warm for
//! `clamp(5 × weight_load_duration, 30s, 15m)`. If no load has been timed yet,
//! fall back to 5 minutes. Override with `WHISBURN_IDLE_UNLOAD_SECS`
//! (`0`/`off` disables; a positive integer forces a fixed timeout).

use std::time::Duration;

/// Multiply measured weight-load time by this for the auto timeout.
pub const AUTO_MULTIPLIER: u32 = 5;
/// Shortest auto idle window (avoids thrashing on tiny models).
pub const AUTO_FLOOR: Duration = Duration::from_secs(30);
/// Longest auto idle window.
pub const AUTO_CEILING: Duration = Duration::from_secs(15 * 60);
/// Used when no weight-load duration has been recorded yet.
pub const UNKNOWN_LOAD_TIMEOUT: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleUnloadPolicy {
    /// `timeout = clamp(multiplier × load, floor, ceiling)`, else [`UNKNOWN_LOAD_TIMEOUT`].
    Auto {
        multiplier: u32,
        floor: Duration,
        ceiling: Duration,
    },
    /// Always unload after this much idle time.
    Fixed(Duration),
    /// Keep models resident until something else releases them.
    Disabled,
}

impl Default for IdleUnloadPolicy {
    fn default() -> Self {
        Self::Auto {
            multiplier: AUTO_MULTIPLIER,
            floor: AUTO_FLOOR,
            ceiling: AUTO_CEILING,
        }
    }
}

impl IdleUnloadPolicy {
    pub fn from_env() -> Self {
        match std::env::var("WHISBURN_IDLE_UNLOAD_SECS") {
            Err(_) => Self::default(),
            Ok(raw) => Self::parse_override(&raw).unwrap_or_else(Self::default),
        }
    }

    /// Parse an override string: empty/auto → Auto; `0`/`off`/`false`/`none` → Disabled;
    /// a non-negative integer → Fixed seconds.
    pub fn parse_override(raw: &str) -> Option<Self> {
        let v = raw.trim();
        if v.is_empty() || v.eq_ignore_ascii_case("auto") {
            return Some(Self::default());
        }
        if matches!(
            v.to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "none" | "disabled"
        ) {
            return Some(Self::Disabled);
        }
        let secs: u64 = v.parse().ok()?;
        if secs == 0 {
            Some(Self::Disabled)
        } else {
            Some(Self::Fixed(Duration::from_secs(secs)))
        }
    }

    pub fn from_secs_override(secs: Option<u32>) -> Self {
        match secs {
            None => Self::default(),
            Some(0) => Self::Disabled,
            Some(n) => Self::Fixed(Duration::from_secs(u64::from(n))),
        }
    }

    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }

    pub fn mode_label(&self) -> &'static str {
        match self {
            Self::Auto { .. } => "auto",
            Self::Fixed(_) => "fixed",
            Self::Disabled => "off",
        }
    }

    /// How long to stay idle before unloading, given the last weight-load time.
    pub fn timeout_for(&self, last_load: Option<Duration>) -> Option<Duration> {
        match *self {
            Self::Disabled => None,
            Self::Fixed(d) => Some(d),
            Self::Auto {
                multiplier,
                floor,
                ceiling,
            } => Some(match last_load {
                None => UNKNOWN_LOAD_TIMEOUT,
                Some(load) => {
                    let scaled = load.saturating_mul(multiplier.max(1) as u32);
                    clamp_duration(scaled, floor, ceiling)
                }
            }),
        }
    }

    /// Fixed override seconds for config APIs (`None` means auto).
    pub fn override_secs(&self) -> Option<u32> {
        match self {
            Self::Auto { .. } => None,
            Self::Fixed(d) => Some(d.as_secs().min(u64::from(u32::MAX)) as u32),
            Self::Disabled => Some(0),
        }
    }
}

fn clamp_duration(value: Duration, floor: Duration, ceiling: Duration) -> Duration {
    if value < floor {
        floor
    } else if value > ceiling {
        ceiling
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_scales_and_clamps() {
        let p = IdleUnloadPolicy::default();
        assert_eq!(
            p.timeout_for(Some(Duration::from_secs(10))),
            Some(Duration::from_secs(50))
        );
        assert_eq!(
            p.timeout_for(Some(Duration::from_secs(2))),
            Some(AUTO_FLOOR)
        );
        assert_eq!(
            p.timeout_for(Some(Duration::from_secs(4 * 60))),
            Some(AUTO_CEILING)
        );
        assert_eq!(p.timeout_for(None), Some(UNKNOWN_LOAD_TIMEOUT));
    }

    #[test]
    fn disabled_has_no_timeout() {
        assert_eq!(IdleUnloadPolicy::Disabled.timeout_for(None), None);
        assert_eq!(
            IdleUnloadPolicy::Disabled.timeout_for(Some(Duration::from_secs(3))),
            None
        );
    }

    #[test]
    fn parse_override_forms() {
        assert_eq!(
            IdleUnloadPolicy::parse_override("auto"),
            Some(IdleUnloadPolicy::default())
        );
        assert_eq!(
            IdleUnloadPolicy::parse_override("off"),
            Some(IdleUnloadPolicy::Disabled)
        );
        assert_eq!(
            IdleUnloadPolicy::parse_override("0"),
            Some(IdleUnloadPolicy::Disabled)
        );
        assert_eq!(
            IdleUnloadPolicy::parse_override("600"),
            Some(IdleUnloadPolicy::Fixed(Duration::from_secs(600)))
        );
        assert!(IdleUnloadPolicy::parse_override("nope").is_none());
    }

    #[test]
    fn from_secs_override() {
        assert!(matches!(
            IdleUnloadPolicy::from_secs_override(None),
            IdleUnloadPolicy::Auto { .. }
        ));
        assert_eq!(
            IdleUnloadPolicy::from_secs_override(Some(0)),
            IdleUnloadPolicy::Disabled
        );
        assert_eq!(
            IdleUnloadPolicy::from_secs_override(Some(120)),
            IdleUnloadPolicy::Fixed(Duration::from_secs(120))
        );
    }
}
