#!/opt/local/bin/python3
"""
V3-EXQ-625c: SD-037 axis (b) Phase 1b -- C3 dynamic-crossings redesign + MECH-341
(SUPERSEDES V3-EXQ-625b -- sharper C3 acceptance criterion + on-policy E3 diversity).

WHAT CHANGED vs V3-EXQ-625b (two coupled changes):

  (1) C3 ACCEPTANCE CRITERION REDESIGN (the headline change).
      V3-EXQ-625b's C3 was "at least one sustained run of >=10 consecutive ticks
      with z_harm_a > 0.4 per seed, in >=2/3 seeds". The failure_autopsy_V3-EXQ-625b
      (2026-06-02, REE_assembly/evidence/planning/) section 4.2 + 7.2 found this
      criterion is too crude: it cannot distinguish PAG dynamic engagement (the
      biological intent -- z_harm_a rises across the 0.4 duration_input_threshold,
      the duration integrator accumulates, behaviour transitions, z_harm_a falls
      back, the cycle repeats) from a CATATONIC-LOCK literal-pass (seed 7 froze on
      action 0 for 1409/1413 steps with z_harm_a pinned continuously above 0.4 ->
      one "sustained run" of length 1413 that counts toward C3 but is biologically
      degenerate). Seeds 42/19 conversely locked monomorphically BELOW 0.4 (0 runs).
      Three seeds, three monomorphic action attractors, no dynamic threat crossings.

      New C3 (per user directive 2026-06-02 + autopsy section 8.3): require genuine
      DYNAMIC CROSSINGS of the z_harm_a duration_input_threshold:
        per seed: >=1 above->below transition AND >=1 below->above transition.
      A continuously-frozen-above policy (seed-7-style) has 0 above->below
      transitions -> FAILS. A continuously-frozen-below policy (seed-42/19-style)
      has 0 crossings of either type -> FAILS. Only a policy whose behaviour
      transitions enough to drive z_harm_a both up across and back down across the
      threshold passes. C3 overall = dynamic-crossings pass on >=2/3 seeds.
      The legacy sustained-window summary is STILL COMPUTED and reported under
      `legacy_sustained_window` for continuity / comparison, but it NO LONGER
      gates acceptance.

  (2) ON-POLICY E3 DIVERSITY SUBSTRATE ENABLED (MECH-341).
      The autopsy diagnosed the 625b C3 FAIL as monostrategy collapse -- the
      behavioural-diversity gap. CRITICAL FINDING confirmed by code inspection
      2026-06-02: the 625b lineage measures the gap4-baseline base policy via
      _lib.goal_pipeline_tier1.warmup_train; it does NOT run the
      ScaffoldedSD054OnboardingScheduler (whose update_z_goal wiring amend landed
      2026-06-02 commit deb24cc), so that amend does NOT feed this measurement
      path. The on-policy diversity lever that DOES apply to this warmup_train
      base policy is MECH-341 (E3 score-layer diversity preservation; validated
      via V3-EXQ-611b), which 625b left OFF. Per user decision (AskUserQuestion
      2026-06-02), 625c enables MECH-341 in the measurement arm:
        use_e3_score_diversity=True (master)
        use_e3_diversity_stratified_select=True (one representative per
          first-action class, softmax-sampled -- the strongest on-policy lever)
        use_e3_diversity_entropy_bonus=True
      SP-CEM main-path (ARC-065) is already the default; SD-056 E2-contrastive is
      deliberately left OFF so a PASS/FAIL is not confounded across two diversity
      levers (only MECH-341 is the on-policy variable vs 625b).
      This departs from 625b's "pure OFF baseline" framing -- intentional: the
      experiment now tests whether a validated diversity substrate produces the
      dynamic z_harm_a crossings the sharper C3 measures.

INTERPRETATION GRID (claim_ids=[], experiment_purpose=diagnostic):
  - C3 dynamic-crossings PASS (>=2/3 seeds show both transition directions):
      MECH-341 delivers dynamic behavioural transitions under sustained threat;
      PAG duration integrator sees genuine crossings. Routes axis-b to Phase 2
      (recalibration block) per the plan-of-record. SD-037 substrate-ceiling
      diagnosis at the consumer-input-threshold layer can be re-examined.
  - C3 FAIL with monostrategy still locked (n_total_transitions ~ 0 across seeds):
      MECH-341 insufficient on this env/policy to break the lock; the
      behavioural-diversity ceiling persists one layer downstream. Routes to
      /failure-autopsy with the corroborating record alongside V3-EXQ-625b /
      V3-EXQ-603d on the behavioural-diversity cluster. NOT a re-test of the
      scaffolded_sd054_onboarding amend (which is disjoint from this path).
  - C1 FAIL (external_hazard_event_count==0 any seed): SD-029 curriculum knob
      mis-applied; fix env overlay and re-run.
  - C2 FAIL (z_harm_a_norm zero_fraction==1.0): affective-stream config defect
      (the 625-vs-625b artifact); verify enable_affective_harm_stream path.

Re-run of V3-EXQ-620's per-step consumer-input-distribution measurement protocol
against the SAME tuned ENV_FISHTANK_KWARGS axis-b overlay as 625b (SD-029
scheduled_external_hazard curriculum + hazard_harm 4x / proximity_harm_scale 2x
lift), with the affective harm stream enabled (the 2026-06-01 /diagnose-errors
fix) and MECH-341 on. Implements the autopsy section 8.3 companion follow-on to
the SD-037 axis-(b) sustained-threat curriculum plan
(REE_assembly/evidence/planning/sd_037_axis_b_sustained_threat_curriculum_plan.md).

The six measured quantities (identical to V3-EXQ-620 / 625b):
  z_harm_a_norm                     -> BLA arousal_threshold_on gate input
  cea_low_freq_magnitude            -> CeA fast_route_threshold gate input
  z_harm_a_instant_val              -> PAG duration_input_threshold gate input
  pag_sustained_product             -> PAG theta_freeze gate input
  bla_pe_magnitude                  -> BLA PE channel
  dacc_pe                           -> dACC PE input

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

EXPERIMENT_TYPE = "v3_exq_625c_sd037_axis_b_phase1b_dynamic_crossings_mech341"
QUEUE_ID = "V3-EXQ-625c"
SUPERSEDES = "V3-EXQ-625b"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = SEEDS_DEFAULT  # [42, 7, 19]
WARMUP_EPISODES = WARMUP_EPISODES_DEFAULT  # 50
EVAL_EPISODES = EVAL_EPISODES_DEFAULT  # 10
STEPS_PER_EPISODE = STEPS_PER_EPISODE_DEFAULT  # 200

# Axis (b) env overlay -- IDENTICAL to V3-EXQ-625b (start from the V3-EXQ-620 env
# and apply the per-plan-section-3.1 delta). Override explicitly rather than
# mutate the imported module-level dict.
ENV_FISHTANK_KWARGS_AXIS_B: Dict[str, Any] = {
    **ENV_FISHTANK_KWARGS,
    "scheduled_external_hazard_enabled": True,
    "scheduled_external_hazard_interval": 20,
    "scheduled_external_hazard_prob": 0.7,
    "scheduled_external_hazard_adjacent_only": True,
    "hazard_harm": 0.2,
    "proximity_harm_scale": 0.2,
}

# Substrate matches V3-EXQ-625b ARM_PHASE1B_BASELINE, PLUS MECH-341 on-policy E3
# score-diversity (the only delta vs 625b's policy). See module docstring (2).
COMMON_CONSUMER_FLAGS: Dict[str, Any] = {
    "use_salience_coordinator": True,
    "use_lateral_pfc_analog": True,
    "use_amygdala_analog": True,
    "use_bla_analog": True,
    "use_cea_analog": True,
}

ARM_PHASE1B_MECH341 = ArmSpec(
    "PHASE1B_MECH341_DYNCROSS",
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
        # MECH-341 on-policy E3 score-diversity (the 625c delta vs 625b).
        # build_config applies these via setattr when hasattr(cfg, key); all
        # three are REEConfig dataclass fields (config.py:1814-1828). Sub-flavour
        # sub-knobs (entropy_lambda / bias_scale / stratified_temperature /
        # min_classes) keep their validated post-611b defaults -- not overridden
        # here (NOT inventing magnitudes; the validated defaults are the
        # calibrated values).
        "use_e3_score_diversity": True,
        "use_e3_diversity_stratified_select": True,
        "use_e3_diversity_entropy_bonus": True,
    },
)

# Quantities measured per step. Same six as V3-EXQ-620 / 625b.
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

# PAG duration_input_threshold default; the z_harm_a level the C3 crossing logic
# and the legacy sustained-window logic both reference.
CROSSING_Z_THRESHOLD = 0.4
SUSTAINED_Z_THRESHOLD = 0.4   # legacy diagnostic (no longer gating)
SUSTAINED_MIN_RUN_LEN = 10    # legacy "sustained" definition (no longer gating)


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


def compute_crossings(
    instant_vals: List[float],
    *,
    threshold: float,
) -> Dict[str, Any]:
    """Count directional threshold crossings of the z_harm_a instant stream.

    The C3 redesign (V3-EXQ-625c). A tick is "above" when v > threshold (strict;
    a tick exactly == threshold counts as below, matching the > convention used
    by the legacy sustained-window logic). A crossing is recorded whenever the
    above/below state flips between consecutive ticks:
      above->below : prev above, curr not above (signal fell back across threshold)
      below->above : prev not above, curr above (signal rose across threshold)

    dynamic_crossings_pass requires BOTH directions present in this seed
    (>=1 above->below AND >=1 below->above) -- i.e. the signal genuinely rose
    across and fell back across the PAG duration_input_threshold at least once
    each. This excludes the seed-7-style continuously-frozen-above literal pass
    (0 above->below) and the seed-42/19-style continuously-frozen-below lock
    (0 crossings of either type).
    """
    n_above_to_below = 0
    n_below_to_above = 0
    n_total_transitions = 0
    prev_above: Optional[bool] = None
    for v in instant_vals:
        above = v > threshold
        if prev_above is not None and above != prev_above:
            n_total_transitions += 1
            if prev_above and not above:
                n_above_to_below += 1
            else:  # (not prev_above) and above
                n_below_to_above += 1
        prev_above = above
    return {
        "threshold": float(threshold),
        "n_ticks": int(len(instant_vals)),
        "n_above_to_below": int(n_above_to_below),
        "n_below_to_above": int(n_below_to_above),
        "n_total_transitions": int(n_total_transitions),
        "dynamic_crossings_pass": bool(
            n_above_to_below >= 1 and n_below_to_above >= 1
        ),
    }


def compute_sustained_windows(
    instant_vals: List[float],
    *,
    threshold: float,
    min_run_len: int,
) -> Dict[str, Any]:
    """Legacy V3-EXQ-625b sustained-window summary (REPORTED ONLY, not gating).

    A run is a maximal contiguous span where instant_vals[t] > threshold. Only
    runs of length >= min_run_len contribute. Retained for continuity with the
    625b manifest and to expose the catatonic-lock literal-pass signature (a
    single run spanning ~the whole eval) that the new C3 correctly excludes.
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

    Also captures per-episode external_hazard_event_count from env.info (summed
    across the eval window), the per-tick z_harm_a_instant_val stream for both
    the new dynamic-crossings C3 and the legacy sustained-window diagnostic, and
    per-action-class counts for the monostrategy diagnosis.
    """
    samples: Dict[str, List[float]] = {q: [] for q, _ in MEASURED_QUANTITIES}
    total_steps = [0]
    action_counts: Dict[int, int] = {}
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

    total_external_hazard_event_count = 0

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        last_event_count = 0
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            info_evt = (result.info or {}).get("external_hazard_event_count", 0)
            try:
                last_event_count = int(info_evt)
            except (TypeError, ValueError):
                last_event_count = 0
            if result.done:
                break
        total_external_hazard_event_count += last_event_count

    crossings_summary = compute_crossings(
        instant_stream,
        threshold=CROSSING_Z_THRESHOLD,
    )
    crossings_summary["external_hazard_event_count"] = int(
        total_external_hazard_event_count
    )

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
        "n_distinct_actions": int(len(action_counts)),
        "per_quantity_samples": samples,
        "crossings_summary": crossings_summary,
        "legacy_sustained_window": sustained_summary,
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
    lines.append("# SD-037 Axis (b) Phase 1b -- C3 Dynamic-Crossings Redesign + MECH-341")
    lines.append("")
    lines.append(f"- Queue id: `{queue_id}`")
    lines.append(f"- Run id: `{run_id}`")
    lines.append(f"- Timestamp UTC: `{timestamp_utc}`")
    lines.append(f"- Supersedes: `{SUPERSEDES}`")
    lines.append(f"- Manifest: `evidence/experiments/{run_id}.json`")
    lines.append(f"- Plan: [sd_037_axis_b_sustained_threat_curriculum_plan.md](sd_037_axis_b_sustained_threat_curriculum_plan.md)")
    lines.append(f"- Autopsy: [failure_autopsy_V3-EXQ-625b_2026-06-02.md](failure_autopsy_V3-EXQ-625b_2026-06-02.md)")
    lines.append("")
    lines.append("Sharper C3 (dynamic-crossings: >=1 above->below AND >=1 below->above per seed) on the SD-037 axis-b env overlay, with MECH-341 on-policy E3 score-diversity enabled. The scaffolded_sd054_onboarding amend does NOT feed this warmup_train measurement path; MECH-341 is the applicable on-policy diversity lever (validated V3-EXQ-611b).")
    lines.append("")
    lines.append("## Env overlay (delta vs V3-EXQ-620; identical to 625b)")
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
    lines.append("## Acceptance gate (C1 AND C2 AND C3-dynamic-crossings)")
    lines.append("")
    for crit, val in acceptance.items():
        lines.append(f"- **{crit}**: {val}")
    lines.append("")
    lines.append("## Per-seed dynamic-crossings (NEW gating C3) + legacy sustained-window (diagnostic only)")
    lines.append("")
    lines.append("| Seed | n_distinct_actions | ext_hazard_events | above->below | below->above | crossings_pass | legacy n_runs | legacy max_run |")
    lines.append("|---|---|---|---|---|---|---|---|")
    rows = cohort_summary.get("rows", [])
    for row in rows:
        cr = row.get("crossings_summary", {})
        sw = row.get("legacy_sustained_window", {})
        lines.append(
            f"| {row.get('seed')} | {row.get('n_distinct_actions', 0)} | "
            f"{cr.get('external_hazard_event_count', 0)} | "
            f"{cr.get('n_above_to_below', 0)} | {cr.get('n_below_to_above', 0)} | "
            f"{cr.get('dynamic_crossings_pass', False)} | "
            f"{sw.get('n_sustained_runs', 0)} | {sw.get('max_sustained_run_length', 0)} |"
        )
    lines.append("")
    lines.append(f"C3 (gating): per seed, n_above_to_below >= 1 AND n_below_to_above >= 1 at z_harm_a > {CROSSING_Z_THRESHOLD}; PASS on >= 2/3 seeds. Legacy sustained-window (>= {SUSTAINED_MIN_RUN_LEN} consecutive ticks above {SUSTAINED_Z_THRESHOLD}) is reported for comparison only -- a single near-full-eval run is the catatonic-lock literal-pass signature the new C3 correctly excludes.")
    lines.append("")
    lines.append("## Pooled distributions (across seeds)")
    lines.append("")
    lines.append("| Quantity | n | min | max | mean | std | p70 | p90 | zero_frac |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    pooled = pooled_summary.get("per_quantity", {})
    for qkey, _ in MEASURED_QUANTITIES:
        d = pooled.get(qkey, {})
        pc = d.get("percentiles", {})
        lines.append(
            f"| `{qkey}` | {d.get('n', 0)} | {d.get('min', 0.0):.4f} | "
            f"{d.get('max', 0.0):.4f} | {d.get('mean', 0.0):.4f} | "
            f"{d.get('std', 0.0):.4f} | {pc.get('p70', 0.0):.4f} | "
            f"{pc.get('p90', 0.0):.4f} | {d.get('zero_fraction', 0.0):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_acceptance_gate(
    rows: List[Dict[str, Any]],
    pooled_per_quantity: Dict[str, Any],
) -> Dict[str, Any]:
    """C1 + C2 unchanged from 625b; C3 redesigned to dynamic-crossings.

    C1: external_hazard_event_count > 0 in 3/3 seeds (curriculum firing).
    C2: zero_fraction < 1.0 on z_harm_a_norm in >= 2/3 seeds (non-zero stream).
    C3: dynamic-crossings (>=1 above->below AND >=1 below->above) in >= 2/3 seeds.
        Legacy sustained-window pass is reported but NOT gating.
    """
    n_seeds = len(rows)
    if n_seeds == 0:
        return {
            "C1_curriculum_firing": False,
            "C1_detail": "no seeds",
            "C2_z_harm_a_nonzero": False,
            "C2_detail": "no seeds",
            "C3_dynamic_crossings": False,
            "C3_detail": "no seeds",
            "legacy_sustained_window_pass": False,
            "legacy_sustained_detail": "no seeds",
            "acceptance_pass": False,
        }

    threshold_2of3 = max(2, (2 * n_seeds + 2) // 3)

    c1_count = 0
    c2_count = 0
    c3_count = 0
    legacy_count = 0
    c1_per_seed: List[Dict[str, Any]] = []
    c2_per_seed: List[Dict[str, Any]] = []
    c3_per_seed: List[Dict[str, Any]] = []
    legacy_per_seed: List[Dict[str, Any]] = []

    for row in rows:
        seed = row.get("seed")
        crossings = row.get("crossings_summary", {}) or {}
        evt_count = int(crossings.get("external_hazard_event_count", 0))
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

        n_ab = int(crossings.get("n_above_to_below", 0))
        n_ba = int(crossings.get("n_below_to_above", 0))
        c3_ok = (n_ab >= 1) and (n_ba >= 1)
        if c3_ok:
            c3_count += 1
        c3_per_seed.append(
            {
                "seed": seed,
                "n_above_to_below": n_ab,
                "n_below_to_above": n_ba,
                "n_total_transitions": int(crossings.get("n_total_transitions", 0)),
                "ok": c3_ok,
            }
        )

        sustained = row.get("legacy_sustained_window", {}) or {}
        n_runs = int(sustained.get("n_sustained_runs", 0))
        legacy_ok = n_runs >= 1
        if legacy_ok:
            legacy_count += 1
        legacy_per_seed.append(
            {
                "seed": seed,
                "n_sustained_runs": n_runs,
                "max_sustained_run_length": int(sustained.get("max_sustained_run_length", 0)),
                "ok": legacy_ok,
            }
        )

    c1_pass = c1_count == n_seeds
    c2_pass = c2_count >= threshold_2of3
    c3_pass = c3_count >= threshold_2of3

    return {
        "C1_curriculum_firing": c1_pass,
        "C1_detail": {"required": "3/3", "achieved": f"{c1_count}/{n_seeds}", "per_seed": c1_per_seed},
        "C2_z_harm_a_nonzero": c2_pass,
        "C2_detail": {"required": ">=2/3", "achieved": f"{c2_count}/{n_seeds}", "per_seed": c2_per_seed},
        "C3_dynamic_crossings": c3_pass,
        "C3_detail": {
            "required": ">=2/3 seeds with >=1 above->below AND >=1 below->above",
            "achieved": f"{c3_count}/{n_seeds}",
            "per_seed": c3_per_seed,
        },
        "legacy_sustained_window_pass": legacy_count >= threshold_2of3,
        "legacy_sustained_detail": {
            "note": "reported for comparison only -- NOT part of acceptance_pass",
            "achieved": f"{legacy_count}/{n_seeds}",
            "per_seed": legacy_per_seed,
        },
        "acceptance_pass": bool(c1_pass and c2_pass and c3_pass),
    }


def run_experiment(
    *,
    seeds: List[int],
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm = ARM_PHASE1B_MECH341
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
        # enable_affective_harm_stream=True is the V3-EXQ-625 root-cause fix:
        # the gap4_operating path otherwise leaves the SD-011 affective harm
        # stream off, so z_harm_a is None and every measured quantity reads 0.0.
        cfg = build_config(env, arm, enable_affective_harm_stream=True)
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        import torch as _torch
        _torch.manual_seed(seed)
        agent = REEAgent(cfg)

        # Sanity: confirm MECH-341 instantiated (the 625c delta). Loud-not-silent
        # so a config regression that drops the diversity substrate is visible in
        # the log rather than silently reverting to the 625b base policy.
        if getattr(agent, "score_diversity", None) is None:
            print(
                "  [warn] MECH-341 score_diversity is None -- use_e3_score_diversity "
                "did not take; this run is a 625b-equivalent base policy.",
                flush=True,
            )
        else:
            print("  [info] MECH-341 E3 score_diversity active (stratified_select on).", flush=True)

        warmup_train(
            agent,
            env,
            num_episodes=warmup_episodes,
            steps_per_episode=steps_per_episode,
            label=f"phase1c_seed_{seed}",
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
        cr = eval_row.get("crossings_summary", {})
        print(
            f"verdict: PASS  seed={seed}  total_eval_steps={n_steps}  "
            f"n_distinct_actions={eval_row.get('n_distinct_actions', 0)}  "
            f"above_to_below={cr.get('n_above_to_below', 0)}  "
            f"below_to_above={cr.get('n_below_to_above', 0)}  "
            f"dynamic_crossings_pass={cr.get('dynamic_crossings_pass', False)}",
            flush=True,
        )

        print(
            f"  [train] phase1c_seed_{seed} ep {warmup_episodes + eval_episodes}"
            f"/{warmup_episodes + eval_episodes}",
            flush=True,
        )

        rows.append(eval_row)

    cohort_summary = {"rows": rows}

    pooled_per_quantity: Dict[str, Any] = {}
    for qkey, _ in MEASURED_QUANTITIES:
        pooled_per_quantity[qkey] = summarise_distribution(pooled_samples[qkey])
    pooled_summary = {"per_quantity": pooled_per_quantity}

    acceptance = evaluate_acceptance_gate(rows, pooled_per_quantity)

    data_populated = all(
        pooled_per_quantity[qkey].get("n", 0) > 0 for qkey, _ in MEASURED_QUANTITIES
    )
    outcome = "PASS" if (data_populated and acceptance["acceptance_pass"]) else "FAIL"

    return {
        "cohort_summary": cohort_summary,
        "pooled_summary": pooled_summary,
        "outcome": outcome,
        "outcome_basis": (
            "acceptance_gate (C1 3/3 AND C2 >=2/3 AND C3 dynamic-crossings >=2/3) "
            "AND all six pooled distributions populated"
        ),
        "data_populated": data_populated,
        "acceptance": acceptance,
        "total_runs": total_runs,
    }


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="Reduced-budget smoke")
    args = parser.parse_args(argv)

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
        "outcome_basis": result.get("outcome_basis", ""),
        "timestamp_utc": timestamp_utc,
        "evidence_direction": "non_contributory",
        "supersedes": SUPERSEDES,
        "supersedes_run_type": "v3_exq_625b_sd037_axis_b_phase1b_consumer_input_distributions_sustained_threat",
        "predecessor_queue_id": "V3-EXQ-625b",
        "affective_harm_stream_enabled": True,
        "mech341_enabled": True,
        "mech341_flags": {
            "use_e3_score_diversity": True,
            "use_e3_diversity_stratified_select": True,
            "use_e3_diversity_entropy_bonus": True,
        },
        "c3_redesign_note": (
            "C3 redesigned from sustained-window (625b) to dynamic-crossings: per "
            "seed >=1 above->below AND >=1 below->above transition of z_harm_a across "
            "0.4; PASS on >=2/3 seeds. Excludes seed-7-style continuously-frozen-above "
            "literal pass and seed-42/19-style frozen-below lock. Legacy sustained-"
            "window reported under acceptance.legacy_sustained_window_pass (not gating)."
        ),
        "measurement_path_note": (
            "Measures the gap4-baseline base policy via _lib.goal_pipeline_tier1."
            "warmup_train; does NOT run the ScaffoldedSD054OnboardingScheduler, so the "
            "2026-06-02 update_z_goal amend (commit deb24cc) does NOT feed this path. "
            "MECH-341 is the applicable on-policy diversity lever (validated 611b) and "
            "is the only policy delta vs 625b (SD-056 deliberately left OFF to avoid "
            "confounding two diversity levers)."
        ),
        "plan_doc": "REE_assembly/evidence/planning/sd_037_axis_b_sustained_threat_curriculum_plan.md",
        "autopsy_ref": "REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-625b_2026-06-02.md",
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
            "arm_id": ARM_PHASE1B_MECH341.arm_id,
            "gap4_operating": ARM_PHASE1B_MECH341.gap4_operating,
            "use_gabaergic_decay": ARM_PHASE1B_MECH341.use_gabaergic_decay,
            "use_pag_freeze_gate": ARM_PHASE1B_MECH341.use_pag_freeze_gate,
            "use_broadcast_override": ARM_PHASE1B_MECH341.use_broadcast_override,
            "extra_config": ARM_PHASE1B_MECH341.extra_config,
        },
        "measured_quantities": [
            {"key": k, "description": d} for k, d in MEASURED_QUANTITIES
        ],
        "percentiles_reported": PERCENTILES,
        "histogram_bins": HISTOGRAM_BINS,
        "crossings_config": {
            "z_threshold": CROSSING_Z_THRESHOLD,
            "require_both_directions": True,
            "above_definition": "v > threshold (strict)",
        },
        "legacy_sustained_window_config": {
            "z_threshold": SUSTAINED_Z_THRESHOLD,
            "min_run_len": SUSTAINED_MIN_RUN_LEN,
            "gating": False,
        },
        "acceptance_gate": result["acceptance"],
        "cohort_summary": result["cohort_summary"],
        "pooled_summary": result["pooled_summary"],
    }

    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {out_path}", flush=True)

    plan_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "planning"
    plan_dir.mkdir(parents=True, exist_ok=True)
    md_path = plan_dir / f"sd037_axis_b_dynamic_crossings_{timestamp_utc}.md"
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
