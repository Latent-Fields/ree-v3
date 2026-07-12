"""V3-EXQ-467: MECH-266 mode stickiness / hold decay parametric sweep
(EXP-0163 -- "mode stickiness / hold decay" row of the SD-033 governance
test battery, ocd4 thought file).

Purpose: diagnostic. Substrate-landing parametric dose-response for the
MECH-266 asymmetric per-mode hysteresis rails on SD-032a
SalienceCoordinator. Verifies that the coordinator produces the
predicted U-shaped behavioural signature across a five-arm sweep of
the over-binding <-> under-binding axis on a fixed synthetic
competing-goals signal sequence. Does NOT run full behavioural
episodes -- those depend on the CausalGridWorldV2 dual-cue env
extension that EXP-0160 / EXP-0163 also need. When that env lands, an
EXQ-467b behavioural successor can measure goal-completion rate and
residue accumulation directly.

Five-arm sweep (ratio parameter r):
  A1 r=0.10  over-binding severe  (OCD axis)
  A2 r=0.50  over-binding moderate
  A3 r=1.00  symmetric baseline   (legacy MECH-259 behaviour)
  A4 r=1.50  under-binding mild
  A5 r=2.00  under-binding severe (depression / aggressive-switching axis)

Ratio encoding (see configure_arm below):
  r <= 1.0: enter_threshold = switch_threshold (1.0); exit_threshold
    per mode = r. Over-binding: tight exit rail; current mode must
    decay to near-zero probability before release. No change to entry.
  r > 1.0:  enter_threshold per mode = 1.0 / r (lower, easier to cross);
    exit_threshold per mode = 1.0 (legacy no-op, trivially satisfied
    for any proper softmax output). Under-binding: salience clears the
    entry rail more easily; argmax flips propagate through.

Predicted signature (substrate-side U-shape):
  n_switches              monotone increasing in r
  longest_run_in_mode     monotone decreasing in r
  max_mode_share          A1 >> A3 ~ A5 (A1 stuck; A3/A5 dispersed)
  ideal_switch_distance   U-shaped around A3: |n_switches - ideal|
                          minimised at r=1.0, larger at both extremes
                          (the behavioural-fitness analogue tested in
                          a future EXQ-467b env run).

Five sub-tests:

  UC1 monotonic switch rate: n_switches(A1) <= n_switches(A2) <=
      n_switches(A3) <= n_switches(A4) <= n_switches(A5). Pairs
      permitted to be equal but never inverted.

  UC2 monotonic longest-run: longest_run_in_mode(A1) >=
      longest_run_in_mode(A2) >= ... >= longest_run_in_mode(A5).

  UC3 A1 stuck signature: n_switches <= 1 AND max_mode_share >= 0.8
      (OCD over-binding axis: current_mode pinned to external_task).

  UC4 A3 baseline match: A3 (ratio=1.0 via set_hysteresis_ratio)
      reproduces the legacy empty-dicts trigger count on the same
      signal sequence.

  UC5 U-shape around ideal: the synthetic signal has an
      "ideal" number of mode changes (regime count - 1 in the
      constructed sequence). |n_switches - ideal| is minimised at
      A3 and strictly larger at BOTH A1 and A5. This is the
      substrate-side U-shape; a behavioural-fitness U-shape (goal
      completion, residue accumulation) requires a dual-cue env.

Substrate-level acceptance only. Full behavioural dose-response
(goal-completion curve, residue-rate curve) deferred to EXQ-467b
when the env lands. Shared synthetic signal sequence and ratio
encoding are designed to transfer straight into the behavioural
successor.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_467_mech266_mode_stickiness.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.cingulate.salience_coordinator import (
    SalienceCoordinator,
    SalienceCoordinatorConfig,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


MODE_NAMES = ["external_task", "internal_planning", "internal_replay", "offline_consolidation"]

# Sweep ratios (A1..A5).
RATIOS = [0.10, 0.50, 1.00, 1.50, 2.00]
ARM_NAMES = ["A1_over_binding_severe", "A2_over_binding_moderate",
             "A3_symmetric_baseline", "A4_under_binding_mild",
             "A5_under_binding_severe"]


def _competing_goals_sequence():
    """Fixed synthetic signal sequence with 3 argmax regimes + intra-regime noise.

    Each tick is (dacc_bundle, drive_level, is_offline, extra_signals).

    Regimes:
      R1 (ticks 0-4):   external_task bias -- low pe, high drive.
      R2 (ticks 5-9):   internal_planning bias -- high pe, low drive. Within
                        R2 a single "doubt tick" at index 7 has moderate pe
                        below A3's enter threshold but above A4/A5's lowered
                        enter rails -- catches lenient Schmitt arms only.
      R3 (ticks 10-14): external_task return -- low pe, high drive, plus
                        an aic_salience injection that raises
                        salience_aggregate above 1.0 so A3 can re-trigger
                        into external_task. Without this, the low-pe R3
                        signal would leave A3 trapped in planning and
                        obscure the U-shape.

    Ideal switch count (regime boundaries): 2.
    """
    seq = []
    # Regime 1 (ticks 0-4): external_task bias.
    for _ in range(5):
        seq.append((
            {"pe": 0.3, "foraging_value": 0.0, "choice_difficulty": 0.0},
            0.9, False, None,
        ))
    # Regime 2 (ticks 5-9): planning bias. Tick 7 is the "doubt" tick --
    # argmax briefly flips back to external_task at sub-A3 salience.
    for i in range(5):
        if i == 2:  # tick 7: doubt
            seq.append((
                {"pe": 0.7, "foraging_value": 0.0, "choice_difficulty": 0.0},
                0.9, False, None,
            ))
        else:
            seq.append((
                {"pe": 1.3, "foraging_value": 0.6, "choice_difficulty": 0.1},
                0.3, False, None,
            ))
    # Regime 3 (ticks 10-14): external_task return with aic_salience.
    # aic_salience contributes to salience_aggregate (weight 1.0) but also
    # adds +1.0 to internal_planning logit -- so drive=0.9 still keeps
    # argmax at external_task.
    for _ in range(5):
        seq.append((
            {"pe": 0.3, "foraging_value": 0.0, "choice_difficulty": 0.0},
            0.9, False, {"aic_salience": 0.8},
        ))
    return seq


IDEAL_SWITCHES = 2  # one switch per regime transition, two transitions.


def configure_arm(coord: SalienceCoordinator, ratio: float) -> dict:
    """Apply ratio to the coordinator's enter/exit thresholds.

    Returns a dict describing the applied thresholds for the manifest.
    """
    if ratio <= 1.0:
        # Over-binding / symmetric: tighten exit, leave enter alone.
        coord.set_hysteresis_ratio(ratio)
        enters = {m: coord.config.switch_threshold for m in coord.mode_names}
        exits = {m: ratio for m in coord.mode_names}
    else:
        # Under-binding: loosen enter (easier to cross), exit stays
        # at 1.0 (legacy no-op).
        for m in coord.mode_names:
            coord.set_enter_threshold(m, 1.0 / ratio)
            coord.set_exit_threshold(m, 1.0)
        enters = {m: 1.0 / ratio for m in coord.mode_names}
        exits = {m: 1.0 for m in coord.mode_names}
    return {"enter_thresholds": enters, "exit_thresholds": exits}


def _run_sequence(coord: SalienceCoordinator):
    """Run the fixed synthetic sequence through a coordinator; return trace."""
    trace = []
    for bundle, drive, offline, extra in _competing_goals_sequence():
        out = coord.tick(
            dacc_bundle=bundle, drive_level=drive,
            is_offline=offline, extra_signals=extra,
        )
        trace.append({
            "current_mode": out["current_mode"],
            "trigger": out["mode_switch_trigger"],
            "salience": round(float(out["salience_aggregate"]), 3),
            "enter_thr": round(float(out["enter_threshold"]), 3),
            "exit_thr": round(float(out["exit_threshold"]), 3),
            "current_mode_prob": round(float(out["current_mode_prob"]), 3),
        })
    return trace


def _n_switches(trace):
    return sum(1 for t in trace if t["trigger"])


def _longest_run_in_mode(trace):
    if not trace:
        return 0
    best = 1
    run = 1
    for i in range(1, len(trace)):
        if trace[i]["current_mode"] == trace[i - 1]["current_mode"]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best


def _mode_shares(trace):
    if not trace:
        return {}
    counts = {m: 0 for m in MODE_NAMES}
    for t in trace:
        counts[t["current_mode"]] = counts.get(t["current_mode"], 0) + 1
    total = len(trace)
    return {m: counts.get(m, 0) / total for m in MODE_NAMES}


def _max_mode_share(trace):
    shares = _mode_shares(trace)
    if not shares:
        return 0.0
    return max(shares.values())


def run_arm(ratio: float) -> dict:
    cfg = SalienceCoordinatorConfig()
    coord = SalienceCoordinator(cfg)
    applied = configure_arm(coord, ratio)
    trace = _run_sequence(coord)
    return {
        "ratio": ratio,
        "applied_thresholds": applied,
        "n_switches": _n_switches(trace),
        "longest_run_in_mode": _longest_run_in_mode(trace),
        "max_mode_share": round(_max_mode_share(trace), 3),
        "mode_shares": {k: round(v, 3) for k, v in _mode_shares(trace).items()},
        "final_mode": trace[-1]["current_mode"] if trace else None,
        "ideal_switch_distance": abs(_n_switches(trace) - IDEAL_SWITCHES),
    }


def run_uc1_monotonic_switches(arm_results) -> dict:
    """UC1: n_switches non-decreasing across A1..A5."""
    switches = [r["n_switches"] for r in arm_results]
    monotone = all(switches[i] <= switches[i + 1] for i in range(len(switches) - 1))
    return {
        "switches_per_arm": switches,
        "monotone_nondecreasing": monotone,
        "pass": monotone,
    }


def run_uc2_monotonic_longest_run(arm_results) -> dict:
    """UC2: longest_run_in_mode non-increasing across A1..A5."""
    runs = [r["longest_run_in_mode"] for r in arm_results]
    monotone = all(runs[i] >= runs[i + 1] for i in range(len(runs) - 1))
    return {
        "longest_run_per_arm": runs,
        "monotone_nonincreasing": monotone,
        "pass": monotone,
    }


def run_uc3_a1_stuck(arm_results) -> dict:
    """UC3: A1 (r=0.1) stuck signature."""
    a1 = arm_results[0]
    return {
        "a1_n_switches": a1["n_switches"],
        "a1_max_mode_share": a1["max_mode_share"],
        "pass": a1["n_switches"] <= 1 and a1["max_mode_share"] >= 0.8,
    }


def run_uc4_a3_matches_legacy(arm_results) -> dict:
    """UC4: A3 (r=1.0 set via hysteresis ratio) matches legacy empty-dict trigger count."""
    a3 = arm_results[2]
    cfg_legacy = SalienceCoordinatorConfig()
    coord_legacy = SalienceCoordinator(cfg_legacy)
    trace_legacy = _run_sequence(coord_legacy)
    legacy_switches = _n_switches(trace_legacy)
    return {
        "a3_switches": a3["n_switches"],
        "legacy_switches": legacy_switches,
        "pass": a3["n_switches"] == legacy_switches,
    }


def run_uc5_ushape_around_ideal(arm_results) -> dict:
    """UC5: U-shape on ideal-switch distance, minimised at A3.

    Substrate-side shape test. A3 should be at the bottom of the U
    (distance from IDEAL_SWITCHES is smallest) and BOTH A1 and A5
    should be strictly worse. A2 / A4 are unconstrained -- they sit
    on the curve without acceptance requirements.
    """
    a1_d = arm_results[0]["ideal_switch_distance"]
    a3_d = arm_results[2]["ideal_switch_distance"]
    a5_d = arm_results[4]["ideal_switch_distance"]
    return {
        "a1_distance": a1_d,
        "a3_distance": a3_d,
        "a5_distance": a5_d,
        "ideal_switches": IDEAL_SWITCHES,
        "pass": a3_d < a1_d and a3_d < a5_d,
    }


def main() -> None:
    t0 = time.time()

    arm_results = [run_arm(r) for r in RATIOS]

    uc1 = run_uc1_monotonic_switches(arm_results)
    uc2 = run_uc2_monotonic_longest_run(arm_results)
    uc3 = run_uc3_a1_stuck(arm_results)
    uc4 = run_uc4_a3_matches_legacy(arm_results)
    uc5 = run_uc5_ushape_around_ideal(arm_results)

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_467_mech266_mode_stickiness_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_467_mech266_mode_stickiness",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-266", "SD-032a"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-266": "supports" if all_pass else "weakens",
            "SD-032a": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "arm_results": [
                {"arm": ARM_NAMES[i], **arm_results[i]}
                for i in range(len(RATIOS))
            ],
            "UC1_monotonic_switches": uc1,
            "UC2_monotonic_longest_run": uc2,
            "UC3_a1_stuck": uc3,
            "UC4_a3_matches_legacy": uc4,
            "UC5_ushape_around_ideal": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-266 parametric dose-response landing diagnostic for "
            "EXP-0163 (mode stickiness / hold decay). Sweeps the "
            "MECH-266 exit/enter ratio from 0.1 (over-binding / OCD) "
            "through 1.0 (symmetric baseline) to 2.0 (under-binding / "
            "aggressive switching) on a fixed synthetic three-regime "
            "competing-goals signal sequence. Measures n_switches, "
            "longest_run_in_mode, mode_shares, and distance from the "
            "ideal switch count as substrate proxies for the full "
            "behavioural metrics (goal-completion rate, residue "
            "accumulation) that require the CausalGridWorldV2 dual-cue "
            "env extension. The behavioural U-shape test is deferred "
            "to EXQ-467b when the env lands. Anchor plan: "
            "REE_assembly/evidence/planning/sd033_governance_plan.md."
        ),
    }

    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

    print(f"result: {manifest['result']}")
    for i, arm in enumerate(arm_results):
        print(f"  {ARM_NAMES[i]} r={arm['ratio']}: "
              f"switches={arm['n_switches']} "
              f"longest={arm['longest_run_in_mode']} "
              f"max_share={arm['max_mode_share']}")
    for k in ["UC1_monotonic_switches", "UC2_monotonic_longest_run",
              "UC3_a1_stuck", "UC4_a3_matches_legacy",
              "UC5_ushape_around_ideal"]:
        print(f"  {k}: pass={manifest['metrics'][k]['pass']}")
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
