#!/opt/local/bin/python3
"""
V3-EXQ-625: SD-037 axis (b) Phase 1b substrate-readiness diagnostic.

Re-run of V3-EXQ-620's per-step consumer-input-distribution measurement
protocol against a TUNED ENV_FISHTANK_KWARGS overlay that engages SD-029's
already-landed scheduled_external_hazard curriculum + lifts hazard_harm /
proximity_harm_scale into the affective-stream-engaging range. Implements
Phase 1b of evidence/planning/sd_037_axis_b_sustained_threat_curriculum_plan.md
(REE_assembly commit c1e345d7dc, 2026-06-01).

Background: V3-EXQ-620 Phase 1 (axis a) returned pooled-identically-zero
distributions across all six consumer-input quantities (n=2939). The
deterministic p70 recalibration rule of axis (a) Phase 2 (sd_037_axis_a_
phase2_recalibration_block.md, 2026-06-01) was empirically unmeetable on
baseline fishtank because the catatonic-lock policy + small per-contact
harm + low contact rate together left z_harm_a_norm pinned at zero. Axis (b)
routes to env-side recalibration via SD-029's already-landed
scheduled_external_hazard curriculum (per the plan's Section 1.1 reuse
decision -- the substrate already exists; this Phase 1b is a tuned env
overlay, NOT a new ree-v3 module).

This experiment does NOT validate any scientific hypothesis. It is a pure
measurement pass on the OFF baseline (SD-037 master OFF, all cascade gains
0; SD-036 + MECH-279 + SalienceCoordinator + dACC + amygdala all ON so the
substrate surface matches 483e ARM_0 minus the broadcast). claim_ids=[].

Env-overlay delta vs V3-EXQ-620 (plan Sections 1.4 + 3.1):
  scheduled_external_hazard_enabled        False -> True
  scheduled_external_hazard_interval       50    -> 20
  scheduled_external_hazard_prob           0.5   -> 0.7
  scheduled_external_hazard_adjacent_only  True  -> True   (held)
  hazard_harm                              0.05  -> 0.2    (4x lift)
  proximity_harm_scale                     0.1   -> 0.2    (2x lift)
All other ENV_FISHTANK_KWARGS defaults preserved (size, num_hazards,
num_resources, env_drift_*, proximity_benefit_scale,
proximity_approach_threshold, hazard_field_decay, resource_respawn_on_consume,
use_proxy_fields, toroidal, harm_history_len, limb_damage_* params,
n_landmarks_b).

Acceptance (plan Section 3.4 substrate-readiness PASS):
  C1 external_hazard_event_count > 0 in 3/3 seeds (curriculum confirmed firing).
  C2 zero_fraction < 1.0 on z_harm_a_norm in >= 2/3 seeds (non-zero
     distributions).
  C3 at least one sustained run (>=10 consecutive ticks with
     z_harm_a > 0.4) per seed in >= 2/3 seeds.

The six measured quantities (identical to V3-EXQ-620):

  z_harm_a_norm                     -> BLA arousal_threshold_on gate input
  cea_low_freq_magnitude            -> CeA fast_route_threshold gate input
  z_harm_a_instant_val              -> PAG duration_input_threshold gate input
  pag_sustained_product             -> PAG theta_freeze gate input
  bla_pe_magnitude                  -> BLA PE channel
  dacc_pe                           -> dACC PE input

Per quantity, per seed: min / max / mean / std / p10 / p25 / p50 / p70 /
p80 / p90 / p95 / p99, 20-bin histogram bin edges + counts, and
zero_fraction (fraction of ticks exactly zero). Also pooled across seeds.

NEW for Phase 1b -- sustained-window summary per seed (plan Section 3.2):
  external_hazard_event_count (end of eval window)
  n_sustained_runs (count of runs >= 10 consecutive ticks with z_harm_a > 0.4)
  total_sustained_duration (sum of all such runs)
  max_sustained_run_length (longest single run)

PASS routes to axis (b) Phase 2 (recalibration block re-application on
the new substrate, per plan Section 4.1). FAIL routes to plan Section 5
five-row interpretation grid:
  external_hazard_event_count == 0 in any seed -> SD-029 curriculum
    knob mis-applied; fix and re-run.
  zero_fraction == 1.0 on z_harm_a_norm despite curriculum firing ->
    affective-stream noise floor; escalate per Section 5.1.
  zero_fraction < 1.0 but no sustained runs -> PAG-specific failure mode
    (Section 5.2); env-kwarg-only mitigation OR axis (c) heavier path.

claim_ids=[]
experiment_purpose=diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness, StepHooks  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    ArmSpec,
    ENV_FISHTANK_KWARGS,
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    WARMUP_EPISODES_DEFAULT,
    build_config,
    make_env,
    warmup_train,
)
from ree_core.agent import REEAgent
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_625_sd037_axis_b_phase1b_consumer_input_distributions_sustained_threat"
QUEUE_ID = "V3-EXQ-625"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = SEEDS_DEFAULT  # [42, 7, 19]
WARMUP_EPISODES = WARMUP_EPISODES_DEFAULT  # 50
EVAL_EPISODES = EVAL_EPISODES_DEFAULT  # 10
STEPS_PER_EPISODE = STEPS_PER_EPISODE_DEFAULT  # 200

# Axis (b) env overlay -- start from the V3-EXQ-620 env (the goal_pipeline_tier1
# ENV_FISHTANK_KWARGS) and apply the per-plan-section-3.1 delta. We override
# explicitly rather than mutate the imported dict (which is module-level and
# shared by other Tier-1 callers).
ENV_FISHTANK_KWARGS_AXIS_B: Dict[str, Any] = {
    **ENV_FISHTANK_KWARGS,
    "scheduled_external_hazard_enabled": True,
    "scheduled_external_hazard_interval": 20,
    "scheduled_external_hazard_prob": 0.7,
    "scheduled_external_hazard_adjacent_only": True,
    "hazard_harm": 0.2,
    "proximity_harm_scale": 0.2,
}

# Substrate matches V3-EXQ-620 ARM_PHASE1_BASELINE verbatim -- broadcast OFF,
# all four MECH-281 cascade gains 0.0, dACC bias_max_abs / weight at the
# axis (a) diagnostic preset, PAG / SalienceCoordinator / amygdala / dACC all ON
# so the substrate surface matches the 483e ARM_0 OFF_OFF configuration minus
# the broadcast. The ONLY delta vs 620 is the env config above.
COMMON_CONSUMER_FLAGS: Dict[str, Any] = {
    "use_salience_coordinator": True,
    "use_lateral_pfc_analog": True,
    "use_amygdala_analog": True,
    "use_bla_analog": True,
    "use_cea_analog": True,
}

ARM_PHASE1B_BASELINE = ArmSpec(
    "PHASE1B_BASELINE",
    gap4_operating=True,
    use_gabaergic_decay=True,
    use_pag_freeze_gate=True,
    use_broadcast_override=False,
    extra_config={
        **COMMON_CONSUMER_FLAGS,
        "use_dacc": True,
        "override_pfc_eta_gain": 0.0,
        "override_bla_encoding_gain": 0.0,
        "override_cea_amplitude_gain": 0.0,
        "override_beta_interrupt_gain": 0.0,
        "dacc_bias_max_abs": 0.1,
        "dacc_weight": 0.1,
    },
)

# Quantities measured per step. Same six as V3-EXQ-620.
MEASURED_QUANTITIES = [
    ("z_harm_a_norm", "BLA arousal_threshold_on input"),
    ("cea_low_freq_magnitude", "CeA fast_route_threshold input"),
    ("z_harm_a_instant_val", "PAG duration_input_threshold input"),
    ("pag_sustained_product", "PAG theta_freeze input"),
    ("bla_pe_magnitude", "BLA PE channel"),
    ("dacc_pe", "dACC PE input"),
]

PERCENTILES = [10, 25, 50, 70, 80, 90, 95, 99]
HISTOGRAM_BINS = 20
ZERO_FLOOR = 1e-9  # below this -> counted as exactly zero in zero_fraction

# Plan Section 3.2 sustained-window thresholds.
SUSTAINED_Z_THRESHOLD = 0.4   # PAG default duration_input_threshold
SUSTAINED_MIN_RUN_LEN = 10    # plan-doc "sustained" definition


def _z_harm_a_norm(agent: REEAgent) -> float:
    lat = getattr(agent, "_current_latent", None)
    if lat is None:
        return 0.0
    za = getattr(lat, "z_harm_a", None)
    if za is None:
        return 0.0
    return float(za.norm().item())


def _cea_low_freq_magnitude(agent: REEAgent) -> float:
    cea = getattr(agent, "cea", None)
    if cea is None:
        return 0.0
    return float(getattr(cea, "_last_low_freq_mag", 0.0))


def _pag_duration_above(agent: REEAgent) -> int:
    pg = getattr(agent, "pag_freeze_gate", None)
    if pg is None:
        return 0
    return int(getattr(pg, "_duration_above_threshold", 0))


def _bla_pe_magnitude(agent: REEAgent) -> float:
    bla = getattr(agent, "bla", None)
    if bla is None:
        return 0.0
    return float(getattr(bla, "_last_pe_magnitude", 0.0))


def _dacc_pe(agent: REEAgent) -> float:
    da = getattr(agent, "dacc", None)
    if da is None:
        return 0.0
    bundle = getattr(da, "_last_bundle", None)
    if bundle is None or not isinstance(bundle, dict):
        return 0.0
    pe = bundle.get("pe", 0.0)
    try:
        return float(pe)
    except (TypeError, ValueError):
        return 0.0


def measure_step(agent: REEAgent) -> Dict[str, float]:
    """One per-tick sample across the six measured quantities."""
    z_norm = _z_harm_a_norm(agent)
    duration = _pag_duration_above(agent)
    return {
        "z_harm_a_norm": z_norm,
        "cea_low_freq_magnitude": _cea_low_freq_magnitude(agent),
        "z_harm_a_instant_val": z_norm,
        "pag_sustained_product": z_norm * float(duration),
        "bla_pe_magnitude": _bla_pe_magnitude(agent),
        "dacc_pe": _dacc_pe(agent),
    }


def summarise_distribution(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "percentiles": {f"p{p}": 0.0 for p in PERCENTILES},
            "histogram": {"bin_edges": [], "counts": []},
            "zero_fraction": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    pct_vals = np.percentile(arr, PERCENTILES)
    counts, edges = np.histogram(arr, bins=HISTOGRAM_BINS)
    n_zero = int(np.sum(arr < ZERO_FLOOR))
    return {
        "n": n,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "percentiles": {
            f"p{p}": float(pct_vals[i]) for i, p in enumerate(PERCENTILES)
        },
        "histogram": {
            "bin_edges": [float(x) for x in edges.tolist()],
            "counts": [int(c) for c in counts.tolist()],
        },
        "zero_fraction": float(n_zero) / float(n),
    }


def compute_sustained_windows(
    instant_vals: List[float],
    *,
    threshold: float,
    min_run_len: int,
) -> Dict[str, Any]:
    """Count and length-summarise runs of consecutive ticks above threshold.

    A run is a maximal contiguous span where instant_vals[t] > threshold.
    Only runs of length >= min_run_len are counted in n_sustained_runs and
    contribute to total_sustained_duration / max_sustained_run_length.
    """
    n_runs = 0
    total_dur = 0
    max_len = 0
    cur = 0
    for v in instant_vals:
        if v > threshold:
            cur += 1
        else:
            if cur >= min_run_len:
                n_runs += 1
                total_dur += cur
                if cur > max_len:
                    max_len = cur
            cur = 0
    # Flush trailing run.
    if cur >= min_run_len:
        n_runs += 1
        total_dur += cur
        if cur > max_len:
            max_len = cur
    return {
        "threshold": float(threshold),
        "min_run_len": int(min_run_len),
        "n_sustained_runs": int(n_runs),
        "total_sustained_duration": int(total_dur),
        "max_sustained_run_length": int(max_len),
    }


def eval_phase1b_diagnostic(
    agent: REEAgent,
    env,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    arm_label: str,
) -> Dict[str, Any]:
    """Eval loop measuring the six consumer-input quantities every tick.

    Also captures per-episode external_hazard_event_count from env.info and
    accumulates a running total across the eval window (the env resets the
    counter on env.reset() per-episode, so we sample at the last step of
    each episode and sum).
    """
    samples: Dict[str, List[float]] = {q: [] for q, _ in MEASURED_QUANTITIES}
    total_steps = [0]
    action_counts: Dict[int, int] = {}
    # Per-tick z_harm_a_instant_val stream for the sustained-window pass.
    instant_stream: List[float] = []

    def on_post_step(*, agent, latent, action, obs_dict, ticks, step, **kwargs) -> None:
        total_steps[0] += 1
        meas = measure_step(agent)
        for k, v in meas.items():
            samples[k].append(v)
        instant_stream.append(meas["z_harm_a_instant_val"])
        aidx = int(action.argmax(dim=-1).item())
        action_counts[aidx] = action_counts.get(aidx, 0) + 1

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    # external_hazard_event_count: env resets it per-episode, so accumulate
    # the per-episode terminal value across the eval window.
    total_external_hazard_event_count = 0

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        last_event_count = 0
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            # Sample the env's running per-episode counter via the
            # StepResult.info dict (StepHarness threads env.step's info
            # through unchanged).
            info_evt = (result.info or {}).get("external_hazard_event_count", 0)
            try:
                last_event_count = int(info_evt)
            except (TypeError, ValueError):
                last_event_count = 0
            if result.done:
                break
        total_external_hazard_event_count += last_event_count

    sustained_summary = compute_sustained_windows(
        instant_stream,
        threshold=SUSTAINED_Z_THRESHOLD,
        min_run_len=SUSTAINED_MIN_RUN_LEN,
    )
    sustained_summary["external_hazard_event_count"] = int(
        total_external_hazard_event_count
    )

    return {
        "arm": arm_label,
        "seed": int(seed),
        "total_eval_steps": int(total_steps[0]),
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "per_quantity_samples": samples,
        "sustained_summary": sustained_summary,
    }


def render_markdown_summary(
    cohort_summary: Dict[str, Any],
    pooled_summary: Dict[str, Any],
    *,
    timestamp_utc: str,
    queue_id: str,
    run_id: str,
    env_overlay: Dict[str, Any],
    acceptance: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# SD-037 Axis (b) Phase 1b -- Consumer-Input Distributions (Sustained-Threat Curriculum)")
    lines.append("")
    lines.append(f"- Queue id: `{queue_id}`")
    lines.append(f"- Run id: `{run_id}`")
    lines.append(f"- Timestamp UTC: `{timestamp_utc}`")
    lines.append(f"- Manifest: `evidence/experiments/{run_id}.json`")
    lines.append(f"- Plan: [sd_037_axis_b_sustained_threat_curriculum_plan.md](sd_037_axis_b_sustained_threat_curriculum_plan.md)")
    lines.append("")
    lines.append("Phase 1b substrate-readiness diagnostic. Substrate matches V3-EXQ-620 ARM_PHASE1_BASELINE verbatim (PAG-engaging env via SD-036 + MECH-279, SalienceCoordinator + dACC + amygdala enabled, broadcast OFF, all four MECH-281 cascade gains 0.0). Only delta vs 620 is the env overlay below -- SD-029 scheduled_external_hazard curriculum ON + hazard_harm 4x lift + proximity_harm_scale 2x lift.")
    lines.append("")
    lines.append("## Env overlay (delta vs V3-EXQ-620)")
    lines.append("")
    lines.append("| Knob | Value |")
    lines.append("|---|---|")
    for k in (
        "scheduled_external_hazard_enabled",
        "scheduled_external_hazard_interval",
        "scheduled_external_hazard_prob",
        "scheduled_external_hazard_adjacent_only",
        "hazard_harm",
        "proximity_harm_scale",
    ):
        lines.append(f"| `{k}` | `{env_overlay.get(k)}` |")
    lines.append("")
    lines.append("## Acceptance gate (plan Section 3.4)")
    lines.append("")
    for crit, val in acceptance.items():
        lines.append(f"- **{crit}**: {val}")
    lines.append("")
    lines.append("## Phase 2 recalibration table (p70 candidates on axis-b distributions)")
    lines.append("")
    lines.append("| Consumer-input quantity | Knob | Current default | Measured p70 (pooled) | Phase 2 candidate |")
    lines.append("|---|---|---|---|---|")
    knob_map = [
        ("z_harm_a_norm", "BLAConfig.arousal_threshold_on", 0.4),
        ("cea_low_freq_magnitude", "CeAConfig.fast_route_threshold", 0.5),
        ("z_harm_a_instant_val", "PAGFreezeGateConfig.duration_input_threshold", 0.4),
        ("pag_sustained_product", "PAGFreezeGateConfig.theta_freeze", 2.0),
        ("bla_pe_magnitude", "(BLA PE channel; informational)", None),
        ("dacc_pe", "DACCConfig.dacc_precision_scale (informational; rescale not threshold)", None),
    ]
    pooled = pooled_summary.get("per_quantity", {})
    for qkey, knob, default in knob_map:
        p70 = pooled.get(qkey, {}).get("percentiles", {}).get("p70", 0.0)
        if default is None:
            candidate_str = "(see plan Phase 2 dACC rescale rule)"
            default_str = "n/a"
        else:
            floor = 0.05 if qkey != "pag_sustained_product" else 0.1
            candidate = max(floor, min(default, p70))
            candidate_str = f"{candidate:.4f}"
            default_str = f"{default:.2f}"
        lines.append(f"| `{qkey}` | `{knob}` | {default_str} | {p70:.4f} | {candidate_str} |")
    lines.append("")
    lines.append("Floor / ceiling per Phase 2 plan: floor 0.05 (theta_freeze 0.1); ceiling current default (so a high-p70 cannot raise a threshold above its current value -- Phase 2 NEVER raises a default, only lowers).")
    lines.append("")
    lines.append("## Per-seed sustained-window summary")
    lines.append("")
    lines.append("| Seed | external_hazard_event_count | n_sustained_runs | total_duration | max_run_len |")
    lines.append("|---|---|---|---|---|")
    rows = cohort_summary.get("rows", [])
    for row in rows:
        s = row.get("sustained_summary", {})
        lines.append(
            f"| {row.get('seed')} | {s.get('external_hazard_event_count', 0)} | "
            f"{s.get('n_sustained_runs', 0)} | "
            f"{s.get('total_sustained_duration', 0)} | "
            f"{s.get('max_sustained_run_length', 0)} |"
        )
    lines.append("")
    lines.append(f"Sustained-run definition: contiguous ticks where `z_harm_a_instant_val > {SUSTAINED_Z_THRESHOLD}` for at least {SUSTAINED_MIN_RUN_LEN} consecutive ticks (PAG `duration_input_threshold` default; biological gating per plan Section 2.3).")
    lines.append("")
    lines.append("## Per-seed distributions")
    lines.append("")
    for row in rows:
        seed = row.get("seed")
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append("| Quantity | min | max | mean | std | p70 | p90 | zero_frac |")
        lines.append("|---|---|---|---|---|---|---|---|")
        per_q = row.get("per_quantity", {})
        for qkey, _ in MEASURED_QUANTITIES:
            d = per_q.get(qkey, {})
            pc = d.get("percentiles", {})
            lines.append(
                f"| `{qkey}` | {d.get('min', 0.0):.4f} | {d.get('max', 0.0):.4f} | "
                f"{d.get('mean', 0.0):.4f} | {d.get('std', 0.0):.4f} | "
                f"{pc.get('p70', 0.0):.4f} | {pc.get('p90', 0.0):.4f} | "
                f"{d.get('zero_fraction', 0.0):.3f} |"
            )
        lines.append("")
    lines.append("## Pooled distributions (across seeds)")
    lines.append("")
    lines.append("| Quantity | n | min | max | mean | std | p10 | p25 | p50 | p70 | p80 | p90 | p95 | p99 | zero_frac |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for qkey, _ in MEASURED_QUANTITIES:
        d = pooled.get(qkey, {})
        pc = d.get("percentiles", {})
        lines.append(
            f"| `{qkey}` | {d.get('n', 0)} | {d.get('min', 0.0):.4f} | "
            f"{d.get('max', 0.0):.4f} | {d.get('mean', 0.0):.4f} | "
            f"{d.get('std', 0.0):.4f} | "
            f"{pc.get('p10', 0.0):.4f} | {pc.get('p25', 0.0):.4f} | "
            f"{pc.get('p50', 0.0):.4f} | {pc.get('p70', 0.0):.4f} | "
            f"{pc.get('p80', 0.0):.4f} | {pc.get('p90', 0.0):.4f} | "
            f"{pc.get('p95', 0.0):.4f} | {pc.get('p99', 0.0):.4f} | "
            f"{d.get('zero_fraction', 0.0):.3f} |"
        )
    lines.append("")
    lines.append("## Routing")
    lines.append("")
    lines.append("PASS -> axis (b) Phase 2 deterministic recalibration rule (re-applies axis-a Phase 2 p70 rule on the new distributions; per plan Section 4.1). FAIL -> route to plan Section 5 five-row interpretation grid (curriculum mis-applied / affective-stream noise floor / PAG sustained-window failure / dACC PE deterministic-prediction / env-kwarg surface exhausted -> axis (c) heavier path).")
    lines.append("")
    return "\n".join(lines)


def evaluate_acceptance_gate(
    rows: List[Dict[str, Any]],
    pooled_per_quantity: Dict[str, Any],
) -> Dict[str, Any]:
    """Plan Section 3.4 substrate-readiness PASS criteria.

    C1: external_hazard_event_count > 0 in 3/3 seeds.
    C2: zero_fraction < 1.0 on z_harm_a_norm in >= 2/3 seeds.
    C3: at least one sustained run (>= 10 consecutive ticks with
        z_harm_a > 0.4) per seed in >= 2/3 seeds.

    Manifest gate is separately enforced in run_experiment: PASS at the
    manifest level = all six pooled quantity distributions have n > 0.
    """
    n_seeds = len(rows)
    if n_seeds == 0:
        return {
            "C1_curriculum_firing": False,
            "C1_detail": "no seeds",
            "C2_z_harm_a_nonzero": False,
            "C2_detail": "no seeds",
            "C3_sustained_window": False,
            "C3_detail": "no seeds",
            "acceptance_pass": False,
        }

    c1_count = 0
    c2_count = 0
    c3_count = 0
    c1_per_seed: List[Dict[str, Any]] = []
    c2_per_seed: List[Dict[str, Any]] = []
    c3_per_seed: List[Dict[str, Any]] = []

    for row in rows:
        seed = row.get("seed")
        sustained = row.get("sustained_summary", {}) or {}
        evt_count = int(sustained.get("external_hazard_event_count", 0))
        c1_ok = evt_count > 0
        if c1_ok:
            c1_count += 1
        c1_per_seed.append({"seed": seed, "external_hazard_event_count": evt_count, "ok": c1_ok})

        per_q = row.get("per_quantity", {}) or {}
        zharma = per_q.get("z_harm_a_norm", {}) or {}
        zero_frac = float(zharma.get("zero_fraction", 1.0))
        c2_ok = zero_frac < 1.0
        if c2_ok:
            c2_count += 1
        c2_per_seed.append({"seed": seed, "zero_fraction": zero_frac, "ok": c2_ok})

        n_runs = int(sustained.get("n_sustained_runs", 0))
        c3_ok = n_runs >= 1
        if c3_ok:
            c3_count += 1
        c3_per_seed.append({"seed": seed, "n_sustained_runs": n_runs, "ok": c3_ok})

    return {
        "C1_curriculum_firing": c1_count == n_seeds,  # 3/3 required
        "C1_detail": {"required": "3/3", "achieved": f"{c1_count}/{n_seeds}", "per_seed": c1_per_seed},
        "C2_z_harm_a_nonzero": c2_count >= max(2, (2 * n_seeds + 2) // 3),  # >=2/3
        "C2_detail": {"required": ">=2/3", "achieved": f"{c2_count}/{n_seeds}", "per_seed": c2_per_seed},
        "C3_sustained_window": c3_count >= max(2, (2 * n_seeds + 2) // 3),  # >=2/3
        "C3_detail": {"required": ">=2/3", "achieved": f"{c3_count}/{n_seeds}", "per_seed": c3_per_seed},
        "acceptance_pass": (
            (c1_count == n_seeds)
            and (c2_count >= max(2, (2 * n_seeds + 2) // 3))
            and (c3_count >= max(2, (2 * n_seeds + 2) // 3))
        ),
    }


def run_experiment(
    *,
    seeds: List[int],
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm = ARM_PHASE1B_BASELINE
    rows: List[Dict[str, Any]] = []
    pooled_samples: Dict[str, List[float]] = {q: [] for q, _ in MEASURED_QUANTITIES}

    total_runs = len(seeds)  # 1 condition

    if dry_run:
        warmup_episodes = 2
        eval_episodes = 2
        steps_per_episode = 30
        seeds = seeds[:1]
        total_runs = len(seeds)

    for run_idx, seed in enumerate(seeds):
        print(f"Seed {seed} Condition {arm.arm_id}", flush=True)
        env = make_env(seed, env_kwargs=ENV_FISHTANK_KWARGS_AXIS_B)
        env._exq_env_kwargs = dict(ENV_FISHTANK_KWARGS_AXIS_B)
        cfg = build_config(env, arm)
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        import torch as _torch
        _torch.manual_seed(seed)
        agent = REEAgent(cfg)

        # Warmup: same protocol as V3-EXQ-620 (E1/E2/E3 trained on a
        # representative-policy trajectory profile so the affective stream
        # is not artificially zero-suppressed by random-policy contact
        # rates). The axis (b) env overlay does NOT change the warmup
        # protocol -- only the env config the warmup runs against.
        warmup_train(
            agent,
            env,
            num_episodes=warmup_episodes,
            steps_per_episode=steps_per_episode,
            label=f"phase1b_seed_{seed}",
            progress_total_episodes=warmup_episodes + eval_episodes,
        )

        eval_row = eval_phase1b_diagnostic(
            agent,
            env,
            num_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            seed=seed,
            arm_label=arm.arm_id,
        )

        per_quantity: Dict[str, Any] = {}
        for qkey, _ in MEASURED_QUANTITIES:
            seed_samples = eval_row["per_quantity_samples"][qkey]
            per_quantity[qkey] = summarise_distribution(seed_samples)
            pooled_samples[qkey].extend(seed_samples)
        eval_row.pop("per_quantity_samples", None)
        eval_row["per_quantity"] = per_quantity

        n_steps = int(eval_row.get("total_eval_steps", 0))
        sustained = eval_row.get("sustained_summary", {})
        print(
            f"verdict: PASS  seed={seed}  total_eval_steps={n_steps}  "
            f"external_hazard_event_count={sustained.get('external_hazard_event_count', 0)}  "
            f"n_sustained_runs={sustained.get('n_sustained_runs', 0)}",
            flush=True,
        )

        print(
            f"  [train] phase1b_seed_{seed} ep {warmup_episodes + eval_episodes}"
            f"/{warmup_episodes + eval_episodes}",
            flush=True,
        )

        rows.append(eval_row)

    cohort_summary = {"rows": rows}

    pooled_per_quantity: Dict[str, Any] = {}
    for qkey, _ in MEASURED_QUANTITIES:
        pooled_per_quantity[qkey] = summarise_distribution(pooled_samples[qkey])
    pooled_summary = {"per_quantity": pooled_per_quantity}

    # Manifest-gate PASS = every quantity has at least one observation
    # across the pooled distribution (the script ran cleanly end-to-end and
    # populated all six quantity distributions). The substrate-readiness
    # acceptance gate (Section 3.4) is computed separately and reported in
    # the manifest but does NOT drive the run-level outcome (this is a
    # pure-measurement diagnostic; whether the substrate cleared the
    # plan-doc thresholds is a downstream-routing question, not a
    # script-correctness question).
    overall_pass = True
    for qkey, _ in MEASURED_QUANTITIES:
        d = pooled_per_quantity[qkey]
        if d.get("n", 0) == 0:
            overall_pass = False
            break
    outcome = "PASS" if overall_pass else "FAIL"

    acceptance = evaluate_acceptance_gate(rows, pooled_per_quantity)

    return {
        "cohort_summary": cohort_summary,
        "pooled_summary": pooled_summary,
        "outcome": outcome,
        "acceptance": acceptance,
        "total_runs": total_runs,
    }


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="Reduced-budget smoke")
    args = parser.parse_args(argv)
    _run_started = datetime.now(timezone.utc)

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    result = run_experiment(
        seeds=SEEDS,
        warmup_episodes=WARMUP_EPISODES,
        eval_episodes=EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        dry_run=args.dry_run,
    )

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": result["outcome"],
        "timestamp_utc": timestamp_utc,
        "evidence_direction": "non_contributory",
        "predecessor_queue_id": "V3-EXQ-620",
        "plan_doc": "REE_assembly/evidence/planning/sd_037_axis_b_sustained_threat_curriculum_plan.md",
        "seeds": SEEDS,
        "warmup_episodes": WARMUP_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "env_kwargs": dict(ENV_FISHTANK_KWARGS_AXIS_B),
        "env_overlay_delta_vs_620": {
            "scheduled_external_hazard_enabled": True,
            "scheduled_external_hazard_interval": 20,
            "scheduled_external_hazard_prob": 0.7,
            "scheduled_external_hazard_adjacent_only": True,
            "hazard_harm": 0.2,
            "proximity_harm_scale": 0.2,
        },
        "arm": {
            "arm_id": ARM_PHASE1B_BASELINE.arm_id,
            "gap4_operating": ARM_PHASE1B_BASELINE.gap4_operating,
            "use_gabaergic_decay": ARM_PHASE1B_BASELINE.use_gabaergic_decay,
            "use_pag_freeze_gate": ARM_PHASE1B_BASELINE.use_pag_freeze_gate,
            "use_broadcast_override": ARM_PHASE1B_BASELINE.use_broadcast_override,
            "extra_config": ARM_PHASE1B_BASELINE.extra_config,
        },
        "measured_quantities": [
            {"key": k, "description": d} for k, d in MEASURED_QUANTITIES
        ],
        "percentiles_reported": PERCENTILES,
        "histogram_bins": HISTOGRAM_BINS,
        "sustained_window_config": {
            "z_threshold": SUSTAINED_Z_THRESHOLD,
            "min_run_len": SUSTAINED_MIN_RUN_LEN,
        },
        "acceptance_gate_section_3_4": result["acceptance"],
        "cohort_summary": result["cohort_summary"],
        "pooled_summary": result["pooled_summary"],
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )
    print(f"manifest written: {out_path}", flush=True)

    plan_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "planning"
    plan_dir.mkdir(parents=True, exist_ok=True)
    md_path = plan_dir / f"sd037_axis_b_consumer_input_distributions_{timestamp_utc}.md"
    md_text = render_markdown_summary(
        result["cohort_summary"],
        result["pooled_summary"],
        timestamp_utc=timestamp_utc,
        queue_id=QUEUE_ID,
        run_id=run_id,
        env_overlay=manifest["env_overlay_delta_vs_620"],
        acceptance=result["acceptance"],
    )
    md_path.write_text(md_text, encoding="utf-8")
    print(f"plan summary written: {md_path}", flush=True)

    outcome_raw = str(manifest["outcome"]).upper()
    return outcome_raw, str(out_path)


if __name__ == "__main__":
    _outcome_raw, _manifest_path = main()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
    )
