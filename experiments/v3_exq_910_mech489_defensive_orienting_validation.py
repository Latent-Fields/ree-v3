"""
V3-EXQ-910 -- MECH-489 (SD-099) Defensive-Orienting Response: ablation validation.

SLEEP DRIVER: K=10 multi-fire (SleepLoopManager, fires every 10 episodes) -- inherited
unchanged from the reused 906b substrate/curriculum; sleep is not the mechanism under
test here.

PURPOSE. Validates the defensive-orienting chain built this session (SD-099 / MECH-489
-- see ree_core/pag/defensive_orienting.py and
REE_assembly/docs/architecture/sd_099_defensive_orienting_response.md) against the exact
pre-registered pass criteria from
REE_assembly/evidence/planning/observational_review_V3-EXQ-906b_2026-08-09.md
Section 13-A / 12h / 12b:

  (a) Does the trigger fire reliably on the ground-truth injected events
      (limb_damage_injected, external_hazard_injected, world_rule_shift_occurred) --
      the specific event set Section 12h found the naive single-channel
      residue_surprise > p90(0.040) design under-fires on (limb_damage_injected mean
      response BELOW the global baseline, unreachable by any threshold on that channel
      alone)?
  (b) Do the surprise-onset->freeze and post-identification dread->withdraw /
      excite->approach couplings move well past the pre-build incidental baselines
      (12b, single-seed, all-ON, no purpose-built orienting mechanism:
      P(moved@t+1|spike)=44.3% vs 24.0% unconditional;
      P(mode-change@t+1|spike)=15.4% vs 11.1%)?

DESIGN. Two-arm ablation on the IDENTICAL 906b full-stack substrate/curriculum/env --
ARM orienting_off (baseline, use_defensive_orienting defaults to False, unchanged) vs
ARM orienting_on (use_defensive_orienting=True at its shipped SD-099 config defaults --
NOT re-tuned here; re-calibration, if this run shows it is needed, is a follow-on, not
something this validation quietly does for itself). Same seeds, same event-injection
schedule, same everything else. This is a single-variable ablation, not a redesign.

Reuses experiments/v3_exq_906b_full_stack_observational_fishtank.py's `_make_config` /
`run_seed` via direct import (GOV-REUSE-1 spirit -- see Step 2.4 note below) rather than
copy-pasting the ~300-line curriculum-training + observational-eval pipeline: this is the
SAME env/agent-training substrate 12b/12g/12h were computed against, so the OFF arm here
reproduces that baseline on fresh seeds and the ON arm is a controlled, single-variable
ablation on top of it. `_make_config` is monkeypatched (module-global function-pointer
swap between arms, restored via try/finally) rather than modified, so the reused module's
own behaviour for its own callers (906b itself, any other future importer) is untouched.

Step 2.4 (GOV-REUSE-1) note: the decisive readouts here (does orienting_trigger_fired
align with the three injected-event fields; do the post-trigger movement/mode-change
couplings clear the 12b baselines) do not exist in any prior manifest -- the mechanism
under test did not exist before this session. Not recoverable by reanalysis; a run is
required.

EXPERIMENT_PURPOSE = "diagnostic": first test of a freshly-built mechanism, substrate-
readiness validation for MECH-489/SD-099, not a governance evidence run.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from experiments.v3_exq_906b_full_stack_observational_fishtank import (
    _make_config,
    run_seed,
)
import experiments.v3_exq_906b_full_stack_observational_fishtank as _fishtank906b

EXPERIMENT_TYPE = "v3_exq_910_mech489_defensive_orienting_validation"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = ["MECH-489"]
SUPERSEDES = None

SEEDS = [0, 1]
ARMS = ["orienting_off", "orienting_on"]

# Pre-registered baselines, observational_review_V3-EXQ-906b_2026-08-09.md Section 12b
# (single-seed, all-ON, incidental correlation with NO purpose-built orienting
# mechanism). MECH-489 must move well past these, not merely reproduce them.
BASELINE_P_MOVED_GIVEN_SPIKE = 0.443
BASELINE_P_MOVED_UNCONDITIONAL = 0.240
BASELINE_P_MODECHANGE_GIVEN_SPIKE = 0.154
BASELINE_P_MODECHANGE_UNCONDITIONAL = 0.111
# 12b's own naive-design threshold, reproduced here on fresh seeds for the OFF arm as a
# same-run sanity replication (not itself a pass criterion for MECH-489).
LEGACY_SURPRISE_SPIKE_THRESHOLD = 0.040

GROUND_TRUTH_EVENT_FIELDS = [
    "limb_damage_injected", "external_hazard_injected", "world_rule_shift_occurred",
]
# Pass criterion (a): fraction of ground-truth events with a trigger within this
# many steps (t..t+window-1) must clear ALIGNMENT_RATE_FLOOR for EVERY event field.
ALIGNMENT_WINDOW = 3
ALIGNMENT_RATE_FLOOR = 0.5
# Pass criterion (b): ON-arm couplings must clear the 12b baseline by at least this
# many percentage points (absolute), not merely exceed it.
COUPLING_MARGIN_FLOOR = 0.10
# Readiness precondition: need at least this many pooled ground-truth events per field
# across all ON-arm seeds for the alignment_rate to be a meaningful measurement at all
# (906b's own single-seed run saw 15-31 per field over 3909 steps; two seeds of the same
# eval length are comfortably expected to clear this floor barring bad luck).
MIN_POOLED_EVENTS_FOR_READINESS = 3


def _make_config_orienting_on(env):
    """ON arm: the OFF arm's exact config plus use_defensive_orienting=True at its
    shipped SD-099 defaults (ree_core/utils/config.py). Deliberately NOT re-tuned here
    -- this validation tests the mechanism as built, not a hand-picked-for-this-run
    calibration."""
    cfg = _make_config(env)
    cfg.use_defensive_orienting = True
    return cfg


_ZG = ZGoalStreamAccumulator()


def _run_arm_seed(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    orig_make_config = _fishtank906b._make_config
    try:
        if arm == "orienting_on":
            _fishtank906b._make_config = _make_config_orienting_on
        result = run_seed(seed, dry_run=dry_run)
    finally:
        _fishtank906b._make_config = orig_make_config
    agent = result.pop("agent", None)
    if agent is not None:
        _ZG.observe(agent)
    return result


def _flatten_steps(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for ep in episodes for s in ep.get("steps", [])]


def _event_trigger_alignment(all_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass criterion (a): for each ground-truth event field, does
    orienting_trigger_fired occur within [t, t+ALIGNMENT_WINDOW) of the event at t?"""
    per_event: Dict[str, Any] = {}
    for field in GROUND_TRUTH_EVENT_FIELDS:
        n_events = 0
        n_aligned = 0
        for i, s in enumerate(all_steps):
            if not s.get(field):
                continue
            n_events += 1
            window = all_steps[i:i + ALIGNMENT_WINDOW]
            if any(w.get("orienting_trigger_fired") for w in window):
                n_aligned += 1
        per_event[field] = {
            "n_events": n_events,
            "n_aligned": n_aligned,
            "alignment_rate": (n_aligned / n_events) if n_events > 0 else None,
        }
    return per_event


def _coupling_stats(episodes: List[Dict[str, Any]], spike_key_fn) -> Dict[str, Any]:
    """P(moved@t+1 | spike@t) and P(mode-change@t+1 | spike@t), within-episode only
    (no cross-segment-boundary lag), matching 12b's exact methodology. spike_key_fn(step)
    -> bool selects the trigger signal per arm (orienting_trigger_fired for ON; a
    residue_surprise > LEGACY_SURPRISE_SPIKE_THRESHOLD proxy for OFF, reproducing 12b)."""
    n_spike = n_spike_moved = n_spike_modechange = 0
    n_nospike = n_nospike_moved = n_nospike_modechange = 0
    for ep in episodes:
        steps = ep.get("steps", [])
        for i in range(len(steps) - 1):
            spiked = bool(spike_key_fn(steps[i]))
            nxt = steps[i + 1]
            moved = nxt.get("pos") != steps[i].get("pos")
            mode_changed = nxt.get("mode") != steps[i].get("mode")
            if spiked:
                n_spike += 1
                n_spike_moved += int(moved)
                n_spike_modechange += int(mode_changed)
            else:
                n_nospike += 1
                n_nospike_moved += int(moved)
                n_nospike_modechange += int(mode_changed)
    return {
        "n_spike": n_spike,
        "p_moved_given_spike": (n_spike_moved / n_spike) if n_spike else None,
        "p_modechange_given_spike": (n_spike_modechange / n_spike) if n_spike else None,
        "n_nospike": n_nospike,
        "p_moved_unconditional": (n_nospike_moved / n_nospike) if n_nospike else None,
        "p_modechange_unconditional": (n_nospike_modechange / n_nospike) if n_nospike else None,
    }


def _decision_alignment(all_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive: how often each resolved decision (approach/withdraw/resume) fired
    on override, for context alongside the coupling numbers. Not itself a pass
    criterion (the chip's pre-registered criteria are the alignment + coupling checks
    above) -- recorded because it is cheap and directly shows whether the action-
    decision component (11b step 5) is exercised at all under real training."""
    counts: Dict[str, int] = {"approach": 0, "withdraw": 0, "resume": 0}
    n_overrides = 0
    for s in all_steps:
        if s.get("orienting_override_fired"):
            n_overrides += 1
        d = s.get("orienting_decision")
        if d in counts:
            counts[d] += 1
    return {"n_overrides": n_overrides, "decision_counts": counts}


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict[str, Any]:
    if seeds is None:
        seeds = SEEDS if not dry_run else [0]
    arms = ARMS

    arm_results: List[Dict[str, Any]] = []
    per_arm_pooled_steps: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    per_arm_episodes: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}

    total_runs = len(arms) * len(seeds)
    run_idx = 0
    for arm in arms:
        for seed in seeds:
            run_idx += 1
            print(f"Seed {seed} Condition {arm}", flush=True)
            with arm_cell(
                seed,
                config_slice={"arm": arm, "use_defensive_orienting": arm == "orienting_on"},
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                seed_result = _run_arm_seed(arm, seed, dry_run=dry_run)
                episodes = seed_result.get("episodes", [])
                cell.stamp(seed_result)
            per_arm_episodes[arm].extend(episodes)
            per_arm_pooled_steps[arm].extend(_flatten_steps(episodes))
            arm_results.append({
                "arm": arm, "seed": seed,
                "chan_std": seed_result.get("chan_std", {}),
                "harm_trained": seed_result.get("harm_trained"),
                "limb_damage_events": seed_result.get("limb_damage_events"),
                "external_hazard_events": seed_result.get("external_hazard_events"),
                "world_rule_shift_events": seed_result.get("world_rule_shift_events"),
                "arm_fingerprint": seed_result.get("arm_fingerprint"),
            })
            print(f"verdict: PASS seed={seed} arm={arm} "
                  f"(cell ran to completion -- see aggregate criteria below)", flush=True)
            print(f"  [progress] run {run_idx}/{total_runs} complete", flush=True)

    # --- Pass criterion (a): ground-truth event -> trigger alignment (ON arm only) ---
    on_steps = per_arm_pooled_steps["orienting_on"]
    off_steps = per_arm_pooled_steps["orienting_off"]
    alignment = _event_trigger_alignment(on_steps)

    # --- Readiness precondition: enough pooled ground-truth events to measure at all ---
    preconditions = []
    for field in GROUND_TRUTH_EVENT_FIELDS:
        n_events = alignment[field]["n_events"]
        preconditions.append({
            "name": f"pooled_{field}_count_sufficient",
            "description": (
                f"At least {MIN_POOLED_EVENTS_FOR_READINESS} pooled {field} events across "
                "all ON-arm seeds are needed for its alignment_rate to be a meaningful "
                "measurement (906b's own single-seed run saw 15-31 per field over 3909 "
                "steps -- this is a not-structurally-impossible check, not a power analysis)."
            ),
            "measured": n_events,
            "threshold": MIN_POOLED_EVENTS_FOR_READINESS,
            "met": n_events >= MIN_POOLED_EVENTS_FOR_READINESS,
        })

    all_preconditions_met = all(p["met"] for p in preconditions)

    if not all_preconditions_met:
        interpretation = {
            "label": "substrate_not_ready_requeue",
            "preconditions": preconditions,
            "criteria_non_degenerate": {},
        }
        outcome = "FAIL"
        criteria = []
    else:
        # --- Pass criterion (a) evaluation ---
        criteria = []
        for field in GROUND_TRUTH_EVENT_FIELDS:
            rate = alignment[field]["alignment_rate"]
            passed = (rate is not None) and (rate >= ALIGNMENT_RATE_FLOOR)
            criteria.append({
                "name": f"C_alignment_{field}",
                "load_bearing": True,
                "passed": bool(passed),
                "measured_alignment_rate": rate,
                "floor": ALIGNMENT_RATE_FLOOR,
            })

        # --- Pass criterion (b): coupling deltas, ON (real trigger) vs OFF (legacy proxy,
        # same-run replication of 12b's methodology) vs the frozen 12b baseline constants ---
        on_coupling = _coupling_stats(
            per_arm_episodes["orienting_on"],
            lambda s: bool(s.get("orienting_trigger_fired")),
        )
        off_coupling_legacy_proxy = _coupling_stats(
            per_arm_episodes["orienting_off"],
            lambda s: (s.get("residue_surprise") or 0.0) > LEGACY_SURPRISE_SPIKE_THRESHOLD,
        )

        def _clears(measured, baseline):
            return (measured is not None) and (measured >= baseline + COUPLING_MARGIN_FLOOR)

        c_moved = _clears(on_coupling["p_moved_given_spike"], BASELINE_P_MOVED_GIVEN_SPIKE)
        c_mode = _clears(on_coupling["p_modechange_given_spike"], BASELINE_P_MODECHANGE_GIVEN_SPIKE)
        criteria.append({
            "name": "C_coupling_moved_beats_12b_baseline",
            "load_bearing": True,
            "passed": bool(c_moved),
            "measured": on_coupling["p_moved_given_spike"],
            "baseline_12b": BASELINE_P_MOVED_GIVEN_SPIKE,
            "margin_floor": COUPLING_MARGIN_FLOOR,
        })
        criteria.append({
            "name": "C_coupling_modechange_beats_12b_baseline",
            "load_bearing": True,
            "passed": bool(c_mode),
            "measured": on_coupling["p_modechange_given_spike"],
            "baseline_12b": BASELINE_P_MODECHANGE_GIVEN_SPIKE,
            "margin_floor": COUPLING_MARGIN_FLOOR,
        })

        decision_alignment = _decision_alignment(on_steps)

        criteria_non_degenerate = {
            c["name"]: (
                (alignment[c["name"].replace("C_alignment_", "")]["n_events"] > 0)
                if c["name"].startswith("C_alignment_")
                else (on_coupling["n_spike"] > 0)
            )
            for c in criteria
        }
        vacuous = any(
            c["passed"] and not criteria_non_degenerate.get(c["name"], True)
            for c in criteria
        )

        all_criteria_pass = all(c["passed"] for c in criteria)
        outcome = "PASS" if (all_criteria_pass and not vacuous) else "FAIL"
        label = (
            "defensive_orienting_validated"
            if outcome == "PASS"
            else "defensive_orienting_partial_or_unmet"
        )
        interpretation = {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "event_trigger_alignment": alignment,
            "on_arm_coupling": on_coupling,
            "off_arm_coupling_legacy_proxy": off_coupling_legacy_proxy,
            "baseline_12b": {
                "p_moved_given_spike": BASELINE_P_MOVED_GIVEN_SPIKE,
                "p_moved_unconditional": BASELINE_P_MOVED_UNCONDITIONAL,
                "p_modechange_given_spike": BASELINE_P_MODECHANGE_GIVEN_SPIKE,
                "p_modechange_unconditional": BASELINE_P_MODECHANGE_UNCONDITIONAL,
            },
            "decision_alignment": decision_alignment,
        }

    metrics: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "arms": arms,
        "event_trigger_alignment": alignment,
    }
    for field in GROUND_TRUTH_EVENT_FIELDS:
        metrics[f"alignment_rate_{field}"] = alignment[field]["alignment_rate"]
        metrics[f"n_events_{field}"] = alignment[field]["n_events"]

    summary_markdown = (
        f"# V3-EXQ-910 -- MECH-489 (SD-099) Defensive-Orienting Validation\n\n"
        f"Seeds: {seeds}. Arms: {arms}.\n\n"
        f"## Pass criterion (a): ground-truth event -> trigger alignment (ON arm)\n"
        + "\n".join(
            f"- {field}: n_events={alignment[field]['n_events']} "
            f"alignment_rate={alignment[field]['alignment_rate']}"
            for field in GROUND_TRUTH_EVENT_FIELDS
        )
        + f"\n\n## Pass criterion (b): behaviour-coupling vs 12b baseline\n"
        f"See interpretation.on_arm_coupling / interpretation.baseline_12b in the manifest.\n\n"
        f"## Outcome: {outcome}\n"
    )

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "arm_results": arm_results, "supersedes": SUPERSEDES,
        "seeds": seeds,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config={"note": "see arm_results[].arm_fingerprint for per-cell config_slice"},
        seeds=result.get("seeds") or (args.seeds or SEEDS),
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
