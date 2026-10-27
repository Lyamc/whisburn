//! Job admission and model placement across VRAM, RAM, and disk.
//!
//! These decisions are pure: the runtime applies them by dropping one device's
//! copy and loading another. Disk means no in-memory copy; weights stay the
//! on-disk model files. At most one model is active on VRAM.

use burn::tensor::{Device, DeviceKind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Where a model's weights live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Active inference device.
    Vram,
    /// Warm, not executing.
    Ram,
    /// Only the on-disk checkpoint remains.
    Disk,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentModel {
    pub name: String,
    pub tier: Tier,
    pub weight_mb: u32,
    /// True while a job is inside inference on this copy. An in-flight model
    /// is not idle: admission must not demote it.
    pub active: bool,
}

/// Free memory the admission rules can see. `vram_known == false` skips the
/// footprint check (the probe failed) but still keeps a single VRAM resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostBudget {
    pub vram_known: bool,
    pub vram_free_mb: u32,
    pub vram_reserve_mb: u32,
    pub ram_free_mb: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierMove {
    pub name: String,
    pub to: Tier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmitPlan {
    /// Device the incoming job should execute on.
    pub run_on: Tier,
    /// Demotions first, then the incoming model's placement.
    pub moves: Vec<TierMove>,
}

/// `0` and `1` are serial. Any larger value is the maximum number of inference
/// jobs allowed to allocate at once.
pub fn parallelism_limit(configured: u32) -> usize {
    match configured {
        0 | 1 => 1,
        n => n as usize,
    }
}

/// Parse `WHISBURN_MAX_PARALLEL_JOBS`. `off` / `disabled` / `0` / `1` are serial.
pub fn parse_parallelism(raw: &str) -> u32 {
    let v = raw.trim();
    if matches!(
        v.to_ascii_lowercase().as_str(),
        "" | "off" | "false" | "disabled" | "serial" | "none"
    ) {
        return 1;
    }
    let n: u32 = v.parse().unwrap_or(1);
    parallelism_limit(n) as u32
}

/// In-process admission cap. `try_acquire` fails while `held == cap`, so a
/// second job cannot overlap the first when the limit is 1.
#[derive(Debug)]
pub struct InferenceSlots {
    cap: usize,
    held: AtomicUsize,
}

impl InferenceSlots {
    pub fn from_setting(configured: u32) -> Self {
        Self {
            cap: parallelism_limit(configured),
            held: AtomicUsize::new(0),
        }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn held(&self) -> usize {
        self.held.load(Ordering::SeqCst)
    }

    pub fn try_acquire(&self) -> bool {
        self.held
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |held| {
                if held >= self.cap {
                    None
                } else {
                    Some(held + 1)
                }
            })
            .is_ok()
    }

    pub fn release(&self) {
        let _ = self.held.try_update(Ordering::SeqCst, Ordering::SeqCst, |held| {
            Some(held.saturating_sub(1))
        });
    }
}

/// Admit `incoming` for inference.
///
/// Idle VRAM residents are demoted before the incoming model is placed on
/// VRAM. An [`ResidentModel::active`] resident is in another job's forward
/// pass, so it is left on the device. Its weight is not treated as free.
/// If `weight + activation` still exceeds free VRAM minus the reserve beside
/// that live model, this job runs from RAM.
pub fn admit_inference(
    budget: &HostBudget,
    residents: &[ResidentModel],
    incoming: &str,
    weight_mb: u32,
    activation_mb: u32,
) -> AdmitPlan {
    let already_on_vram = residents
        .iter()
        .any(|m| m.name == incoming && m.tier == Tier::Vram);
    let incoming_active = residents
        .iter()
        .any(|m| m.name == incoming && m.tier == Tier::Vram && m.active);
    let footprint = if already_on_vram {
        activation_mb
    } else {
        weight_mb.saturating_add(activation_mb)
    };
    let available = budget
        .vram_free_mb
        .saturating_sub(budget.vram_reserve_mb);
    let live_other = residents
        .iter()
        .any(|m| m.tier == Tier::Vram && m.name != incoming && m.active);
    // Dropping any model tears down the shared device. While a forward pass
    // is in flight, nothing on that device can be demoted.
    let can_demote = !live_other && !incoming_active;

    let mut freed = 0u32;
    if can_demote {
        for other in residents
            .iter()
            .filter(|m| m.tier == Tier::Vram && m.name != incoming && !m.active)
        {
            freed = freed.saturating_add(other.weight_mb);
        }
    }

    let fits = !budget.vram_known || footprint <= available.saturating_add(freed);
    let run_on = if fits { Tier::Vram } else { Tier::Ram };
    // When the incoming job itself must sit in RAM, that weight is already
    // claimed. Inactive models only stay warm if the rest of RAM can hold them.
    let mut ram_left = if fits {
        budget.ram_free_mb
    } else {
        budget.ram_free_mb.saturating_sub(weight_mb)
    };
    let mut moves = Vec::new();
    if can_demote {
        for other in residents
            .iter()
            .filter(|m| m.tier == Tier::Vram && m.name != incoming && !m.active)
        {
            let to = demote_target(ram_left, other.weight_mb);
            if to == Tier::Ram {
                ram_left = ram_left.saturating_sub(other.weight_mb);
            }
            moves.push(TierMove {
                name: other.name.clone(),
                to,
            });
        }
    }
    // Leave an in-flight copy where it is. This job can still run from RAM.
    if !(incoming_active && run_on == Tier::Ram) {
        moves.push(TierMove {
            name: incoming.to_string(),
            to: run_on,
        });
    }
    AdmitPlan { run_on, moves }
}

fn demote_target(ram_free_mb: u32, weight_mb: u32) -> Tier {
    if ram_free_mb >= weight_mb {
        Tier::Ram
    } else {
        Tier::Disk
    }
}

/// Summarize handoff.
///
/// The summarizer is staged in RAM first while an idle speech model may still
/// be on VRAM. The summarizer is promoted only when `weight + activation`
/// fits in free VRAM minus the reserve after that idle model is demoted.
/// A footprint that still does not fit runs the summary from RAM and does not
/// launch on the GPU. An in-flight speech model is not demoted; the summary
/// then has to fit beside it or stay on RAM.
pub fn plan_summarize_handoff(
    budget: &HostBudget,
    speech: Option<&ResidentModel>,
    summarizer: &str,
    summarizer_weight_mb: u32,
    summarizer_activation_mb: u32,
) -> Vec<TierMove> {
    let speech_weight = speech.map(|m| m.weight_mb).unwrap_or(0);
    let both = speech_weight.saturating_add(summarizer_weight_mb);
    let ram_holds_summarizer = budget.ram_free_mb >= summarizer_weight_mb;
    let ram_holds_both = budget.ram_free_mb >= both;
    let speech_live = speech.is_some_and(|m| m.tier == Tier::Vram && m.active);
    let speech_idle_vram = speech.is_some_and(|m| m.tier == Tier::Vram && !m.active);
    let freed = if speech_idle_vram { speech_weight } else { 0 };
    let available = budget
        .vram_free_mb
        .saturating_sub(budget.vram_reserve_mb);
    let footprint = summarizer_weight_mb.saturating_add(summarizer_activation_mb);
    let fits = !budget.vram_known || footprint <= available.saturating_add(freed);
    let summarizer_to = if fits { Tier::Vram } else { Tier::Ram };

    let mut moves = Vec::new();
    if summarizer_to == Tier::Vram && ram_holds_summarizer && summarizer_weight_mb > 0 {
        moves.push(TierMove {
            name: summarizer.to_string(),
            to: Tier::Ram,
        });
    }
    if let Some(speech) = speech {
        if !speech_live {
            let speech_to = if ram_holds_both {
                Tier::Ram
            } else {
                Tier::Disk
            };
            if speech.tier != speech_to {
                moves.push(TierMove {
                    name: speech.name.clone(),
                    to: speech_to,
                });
            }
        }
    }
    moves.push(TierMove {
        name: summarizer.to_string(),
        to: summarizer_to,
    });
    moves
}

/// Apply moves in order. A name that is not yet resident is inserted.
pub fn apply_moves(
    mut models: Vec<ResidentModel>,
    moves: &[TierMove],
    weight_of: impl Fn(&str) -> u32,
) -> Vec<ResidentModel> {
    for mv in moves {
        if let Some(existing) = models.iter_mut().find(|m| m.name == mv.name) {
            existing.tier = mv.to;
        } else {
            models.push(ResidentModel {
                name: mv.name.clone(),
                tier: mv.to,
                weight_mb: weight_of(&mv.name),
                active: false,
            });
        }
    }
    models
}

pub fn tier_of(models: &[ResidentModel], name: &str) -> Option<Tier> {
    models.iter().find(|m| m.name == name).map(|m| m.tier)
}

/// Host RAM available to stage a warm model, in MiB.
pub fn ram_available_mb() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("MemAvailable:") else {
                continue;
            };
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some((kb / 1024).min(u64::from(u32::MAX)) as u32);
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Block until a slot is free, then hold it until the guard drops.
/// Used by tests to show a second caller does not enter while the first holds
/// the only slot. The server sizes its semaphore with [`parallelism_limit`].
pub fn acquire_slot_blocking(slots: &Arc<InferenceSlots>) {
    while !slots.try_acquire() {
        thread::sleep(Duration::from_millis(2));
    }
}

pub struct SlotGuard {
    slots: Arc<InferenceSlots>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.slots.release();
    }
}

pub fn enter_slot(slots: &Arc<InferenceSlots>) -> SlotGuard {
    acquire_slot_blocking(slots);
    SlotGuard {
        slots: Arc::clone(slots),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn budget(vram_free: u32, reserve: u32, ram_free: u32) -> HostBudget {
        HostBudget {
            vram_known: true,
            vram_free_mb: vram_free,
            vram_reserve_mb: reserve,
            ram_free_mb: ram_free,
        }
    }

    fn weights(name: &str) -> u32 {
        match name {
            "speech" => 2_000,
            "other" => 2_500,
            "summarizer" => 1_800,
            _ => 0,
        }
    }

    #[test]
    fn serial_queue_does_not_let_a_second_job_overlap() {
        let slots = Arc::new(InferenceSlots::from_setting(1));
        let first = enter_slot(&slots);
        assert_eq!(slots.held(), 1);
        let slots2 = Arc::clone(&slots);
        let entered = Arc::new(AtomicBool::new(false));
        let entered2 = Arc::clone(&entered);
        let handle = thread::spawn(move || {
            let _second = enter_slot(&slots2);
            entered2.store(true, Ordering::SeqCst);
            slots2.held()
        });
        thread::sleep(Duration::from_millis(40));
        assert!(
            !entered.load(Ordering::SeqCst),
            "second job overlapped the first while parallelism is 1"
        );
        drop(first);
        let held_when_second_entered = handle.join().unwrap();
        assert!(entered.load(Ordering::SeqCst));
        assert_eq!(held_when_second_entered, 1);
    }

    #[test]
    fn parallelism_n_allows_n_and_blocks_the_next() {
        let n = 3u32;
        let slots = InferenceSlots::from_setting(n);
        assert_eq!(slots.cap(), parallelism_limit(n));
        for _ in 0..n {
            assert!(slots.try_acquire());
        }
        assert!(
            !slots.try_acquire(),
            "job N+1 must wait instead of allocating"
        );
        assert_eq!(slots.held(), n as usize);
        slots.release();
        assert!(slots.try_acquire());
    }

    #[test]
    fn disabled_parallelism_is_serial() {
        assert_eq!(parallelism_limit(0), 1);
        assert_eq!(parse_parallelism("off"), 1);
        assert_eq!(parse_parallelism("disabled"), 1);
        assert_eq!(parse_parallelism("4"), 4);
    }

    #[test]
    fn oversized_footprint_is_not_on_vram_until_the_other_model_is_demoted() {
        let residents = vec![ResidentModel {
            name: "other".into(),
            tier: Tier::Vram,
            weight_mb: 2_500,
            active: false,
        }];
        // free 4000 − reserve 1024 = 2976. Footprint 2000+1500 = 3500 does not fit
        // until `other` (2500) is demoted, after which 2976+2500 covers 3500.
        let plan = admit_inference(&budget(4_000, 1_024, 16_000), &residents, "incoming", 2_000, 1_500);
        assert_eq!(plan.run_on, Tier::Vram);
        assert_eq!(plan.moves[0].name, "other");
        assert_ne!(plan.moves[0].to, Tier::Vram);
        let incoming_at = plan
            .moves
            .iter()
            .position(|m| m.name == "incoming" && m.to == Tier::Vram)
            .expect("incoming reaches VRAM only as a later move");
        assert!(incoming_at > 0);

        let mut cursor = residents.clone();
        for mv in &plan.moves {
            if mv.name == "incoming" && mv.to == Tier::Vram {
                assert!(
                    cursor.iter().all(|m| m.name == "incoming" || m.tier != Tier::Vram),
                    "incoming must not be on VRAM while another model is still resident"
                );
            }
            cursor = apply_moves(cursor, std::slice::from_ref(mv), weights);
        }
        assert_eq!(tier_of(&cursor, "incoming"), Some(Tier::Vram));
        assert_ne!(tier_of(&cursor, "other"), Some(Tier::Vram));
    }

    #[test]
    fn footprint_that_still_does_not_fit_after_demotion_runs_from_ram() {
        let residents = vec![ResidentModel {
            name: "other".into(),
            tier: Tier::Vram,
            weight_mb: 500,
            active: false,
        }];
        let plan = admit_inference(&budget(2_000, 1_024, 8_000), &residents, "incoming", 3_000, 2_000);
        assert_eq!(plan.run_on, Tier::Ram);
        let end = apply_moves(residents, &plan.moves, |_| 3_000);
        assert_eq!(tier_of(&end, "incoming"), Some(Tier::Ram));
        assert_ne!(tier_of(&end, "other"), Some(Tier::Vram));
    }

    #[test]
    fn summarize_handoff_parks_speech_in_ram_when_ram_holds_both() {
        let speech = ResidentModel {
            name: "speech".into(),
            tier: Tier::Vram,
            weight_mb: 2_000,
            active: false,
        };
        let moves = plan_summarize_handoff(
            &budget(6_000, 1_024, 8_000),
            Some(&speech),
            "summarizer",
            1_800,
            0,
        );
        assert_eq!(moves[0].to, Tier::Ram, "summarizer is prepared in RAM first");
        assert_eq!(moves[0].name, "summarizer");
        let end = apply_moves(vec![speech], &moves, weights);
        assert_eq!(tier_of(&end, "summarizer"), Some(Tier::Vram));
        assert_eq!(tier_of(&end, "speech"), Some(Tier::Ram));
    }

    #[test]
    fn summarize_handoff_spills_speech_to_disk_when_ram_cannot_hold_both() {
        let speech = ResidentModel {
            name: "speech".into(),
            tier: Tier::Vram,
            weight_mb: 2_000,
            active: false,
        };
        // 1_800 fits the summarizer alone, not speech+summarizer (3_800).
        let moves = plan_summarize_handoff(
            &budget(6_000, 1_024, 2_000),
            Some(&speech),
            "summarizer",
            1_800,
            0,
        );
        let speech_move = moves.iter().find(|m| m.name == "speech").unwrap();
        assert_eq!(speech_move.to, Tier::Disk);
        let speech_at = moves.iter().position(|m| m.name == "speech").unwrap();
        let promote_at = moves
            .iter()
            .rposition(|m| m.name == "summarizer" && m.to == Tier::Vram)
            .unwrap();
        assert!(speech_at < promote_at, "inactive model hits disk before promotion");
        let end = apply_moves(vec![speech], &moves, weights);
        assert_eq!(tier_of(&end, "summarizer"), Some(Tier::Vram));
        assert_eq!(tier_of(&end, "speech"), Some(Tier::Disk));
    }

    #[test]
    fn summarize_handoff_stays_on_ram_when_footprint_exceeds_reserve() {
        let speech = ResidentModel {
            name: "speech".into(),
            tier: Tier::Vram,
            weight_mb: 400,
            active: false,
        };
        // free 1200 − reserve 1024 = 176. Demoting speech frees 400 → 576.
        // Weight 7000 + activation 512 does not fit, so the summary stays on RAM.
        let moves = plan_summarize_handoff(
            &budget(1_200, 1_024, 32_000),
            Some(&speech),
            "summarizer",
            7_000,
            512,
        );
        assert!(
            moves
                .iter()
                .all(|m| !(m.name == "summarizer" && m.to == Tier::Vram)),
            "summarizer must not be launched on the GPU"
        );
        let end = apply_moves(vec![speech], &moves, |name| match name {
            "summarizer" => 7_000,
            "speech" => 400,
            _ => 0,
        });
        assert_eq!(tier_of(&end, "summarizer"), Some(Tier::Ram));
        assert_ne!(tier_of(&end, "summarizer"), Some(Tier::Vram));
    }

    #[test]
    fn in_flight_vram_resident_is_not_demoted_and_overflow_runs_from_ram() {
        let residents = vec![ResidentModel {
            name: "other".into(),
            tier: Tier::Vram,
            weight_mb: 2_500,
            active: true,
        }];
        // free 4000 − reserve 1024 = 2976. Footprint 3500 does not fit beside
        // the live model, and that model must stay on VRAM.
        let plan = admit_inference(
            &budget(4_000, 1_024, 16_000),
            &residents,
            "incoming",
            2_000,
            1_500,
        );
        assert_eq!(plan.run_on, Tier::Ram);
        assert!(
            plan.moves.iter().all(|m| m.name != "other"),
            "in-flight resident must not be demoted"
        );
        let end = apply_moves(residents, &plan.moves, |_| 2_000);
        assert_eq!(tier_of(&end, "other"), Some(Tier::Vram));
        assert_eq!(tier_of(&end, "incoming"), Some(Tier::Ram));
    }
}
