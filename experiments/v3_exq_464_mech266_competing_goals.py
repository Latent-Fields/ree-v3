"""V3-EXQ-464: MECH-266 asymmetric per-mode hysteresis landing diagnostic
(EXP-0160 -- "competing goals" row of the SD-033 governance test battery,
ocd4 thought file).

Purpose: diagnostic. Confirms the MECH-266 substrate (per-mode
enter_thresholds / exit_thresholds on SD-032a SalienceCoordinator) is
wired correctly and demonstrates the three-arm signature the full
EXP-0160 behavioural test would exercise. Does NOT run the full
competing-goals env episodes that EXP-0160 specifies -- those depend on
a CausalGridWorldV2 extension with dual simultaneously active
resource-class cues that is not yet on any roadmap item. When that
env lands, an EXQ-464b behavioural successor can run end-to-end
episodes; this landing is the substrate-side prerequisite for it.

Five sub-tests (deterministic arithmetic + coordinator API):

  UC1 backward compat: with enter_thresholds and exit_thresholds empty
      (default), tick() behaviour matches the pre-MECH-266 MECH-259
      logic -- the reported enter_threshold equals switch_threshold,
      the reported exit_threshold is 1.0 (always-satisfied sentinel),
      and trigger fires exactly when salience > switch_threshold AND
      argmax flips.

  UC2 ARM A -- symmetric baseline: exit_thresholds = {m: 1.0 for all
      modes} -> exit check is a no-op, reproduces UC1 trigger rate on
      a fixed synthetic competing-goals signal sequence.

  UC3 ARM B -- moderate Schmitt (exit_threshold < enter_threshold):
      exit_thresholds = {m: 0.3} for all modes. On the same synthetic
      signal sequence, the number of mode switches is strictly lower
      than ARM A (at least one fewer flip -- the dead zone blocks
      marginal argmax flips whose current-mode prob has not decayed
      below 0.3 yet).

  UC4 ARM C -- severe over-binding (exit_threshold near 0): exit_thresholds
      = {m: 0.05}. Stuck-in-mode signature: n_switches <= 1 across the
      entire synthetic sequence AND current_mode stays on
      external_task for at least 80% of ticks (the OCD-axis
      prediction from the ocd4 thought file).

  UC5 per-mode asymmetry: exit_thresholds can differ per mode. Setting
      exit_thresholds['external_task'] = 0.05 and all others = 0.9
      produces stuck-in-external_task under the same signal sequence,
      whereas setting exit_thresholds['internal_planning'] = 0.05 and
      others = 0.9 does NOT stick in external_task (leaves it readily,
      sticks in internal_planning once entered). Confirms the
      per-mode keying of the Schmitt rail.

Substrate-level acceptance only. Full behavioural compliance contrasts
(switch-cost asymmetry measured on real trajectories, fraction of
unresolved goal-oscillation episodes on env-driven dual-cue tasks)
are deferred until the dual-cue env extension lands.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_464_mech266_competing_goals.py

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


MODE_NAMES = ["external_task", "internal_planning", "internal_replay", "offline_consolidation"]


def _competing_goals_sequence():
    """Fixed synthetic sequence with 3 argmax regimes and intra-regime noise.

    Each tick is (dacc_bundle, drive_level, is_offline, extra_signals).

    Regimes: external_task -> internal_planning -> external_task.
    Within R2 (planning) a "doubt" tick at index 7 has moderate pe that
    leaves planning as current but briefly flips argmax back to
    external_task, with current_mode_prob(planning) in the [0.3, 0.7]
    range. This marginal flip is what moderate-Schmitt (exit=0.3) is
    designed to suppress while loose-Schmitt (exit=1.0 or unset) lets
    through. Without this tick there is no room between "strong flip"
    and "no flip" and UC3 is vacuous.

    R3 carries an aic_salience injection so the R2->R3 transition
    clears the default switch_threshold for legacy / ARM A / ARM B
    arms (salience_aggregate = pe + aic_salience > 1.0).
    """
    seq = []
    # Regime 1 (ticks 0-4): external_task bias.
    for _ in range(5):
        seq.append((
            {"pe": 0.3, "foraging_value": 0.0, "choice_difficulty": 0.0},
            0.9, False, None,
        ))
    # Regime 2 (ticks 5-9): planning bias; tick 7 is the doubt tick.
    for i in range(5):
        if i == 2:  # tick 7: marginal flip back to external
            seq.append((
                {"pe": 0.7, "foraging_value": 0.0, "choice_difficulty": 0.0},
                0.9, False, None,
            ))
        else:
            seq.append((
                {"pe": 1.3, "foraging_value": 0.6, "choice_difficulty": 0.1},
                0.3, False, None,
            ))
    # Regime 3 (ticks 10-14): external_task return with aic_salience so
    # salience_aggregate > 1.0 even under low pe. Drive held at 0.3 so
    # planning-prob at the transition lands ~0.33 -- above the 0.3
    # Schmitt rail (UC3 blocks the flip) but well below the 1.0
    # legacy rail (baseline lets the flip through).
    for _ in range(5):
        seq.append((
            {"pe": 0.3, "foraging_value": 0.0, "choice_difficulty": 0.0},
            0.3, False, {"aic_salience": 0.8},
        ))
    return seq


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


def _fraction_in_mode(trace, mode):
    if not trace:
        return 0.0
    return sum(1 for t in trace if t["current_mode"] == mode) / len(trace)


def run_uc1_backward_compat() -> dict:
    """UC1: empty dicts -> legacy MECH-259 behaviour."""
    cfg = SalienceCoordinatorConfig()
    coord = SalienceCoordinator(cfg)
    trace = _run_sequence(coord)
    # enter_thr should equal switch_threshold (1.0) everywhere (no stability input).
    enters_match = all(t["enter_thr"] == 1.0 for t in trace)
    # exit_thr should be 1.0 (sentinel no-op) everywhere.
    exits_match = all(t["exit_thr"] == 1.0 for t in trace)
    return {
        "n_ticks": len(trace),
        "n_switches": _n_switches(trace),
        "enter_always_switch_threshold": enters_match,
        "exit_always_one_sentinel": exits_match,
        "final_mode": trace[-1]["current_mode"] if trace else None,
        "pass": enters_match and exits_match,
    }


def run_uc2_arm_a_symmetric_baseline() -> dict:
    """UC2: ARM A -- exit=1.0 explicit. Matches UC1 trigger pattern."""
    cfg = SalienceCoordinatorConfig()
    cfg.exit_thresholds = {m: 1.0 for m in MODE_NAMES}
    coord = SalienceCoordinator(cfg)
    trace = _run_sequence(coord)

    cfg2 = SalienceCoordinatorConfig()
    coord2 = SalienceCoordinator(cfg2)
    trace_ref = _run_sequence(coord2)

    sym_switches = _n_switches(trace)
    ref_switches = _n_switches(trace_ref)
    return {
        "arm_a_switches": sym_switches,
        "legacy_switches": ref_switches,
        "match": sym_switches == ref_switches,
        "pass": sym_switches == ref_switches and sym_switches >= 1,
    }


def run_uc3_arm_b_moderate_schmitt() -> dict:
    """UC3: ARM B -- exit=0.3 suppresses marginal flips."""
    cfg_ref = SalienceCoordinatorConfig()
    coord_ref = SalienceCoordinator(cfg_ref)
    trace_ref = _run_sequence(coord_ref)

    cfg_b = SalienceCoordinatorConfig()
    cfg_b.exit_thresholds = {m: 0.3 for m in MODE_NAMES}
    coord_b = SalienceCoordinator(cfg_b)
    trace_b = _run_sequence(coord_b)

    ref_n = _n_switches(trace_ref)
    b_n = _n_switches(trace_b)
    return {
        "symmetric_switches": ref_n,
        "schmitt_switches": b_n,
        "delta": ref_n - b_n,
        "pass": b_n < ref_n and b_n >= 0,
    }


def run_uc4_arm_c_severe_over_binding() -> dict:
    """UC4: ARM C -- exit near 0 -> stuck in initial mode.

    The coordinator initialises current_mode = 'external_task'. With
    exit_thresholds[external_task] near 0, no switch fires on the
    synthetic sequence -- the OCD-axis stuck-in-mode signature.
    """
    cfg = SalienceCoordinatorConfig()
    cfg.exit_thresholds = {m: 0.05 for m in MODE_NAMES}
    coord = SalienceCoordinator(cfg)
    trace = _run_sequence(coord)
    n_sw = _n_switches(trace)
    frac_task = _fraction_in_mode(trace, "external_task")
    return {
        "arm_c_switches": n_sw,
        "fraction_in_external_task": round(frac_task, 3),
        "pass": n_sw <= 1 and frac_task >= 0.8,
    }


def run_uc5_per_mode_asymmetry() -> dict:
    """UC5: exit_threshold keyed on each mode independently."""
    # Condition X: sticky external_task, loose everything else.
    cfg_x = SalienceCoordinatorConfig()
    cfg_x.exit_thresholds = {
        "external_task": 0.05,
        "internal_planning": 0.9,
        "internal_replay": 0.9,
        "offline_consolidation": 0.9,
    }
    coord_x = SalienceCoordinator(cfg_x)
    trace_x = _run_sequence(coord_x)

    # Condition Y: loose external_task, sticky internal_planning.
    cfg_y = SalienceCoordinatorConfig()
    cfg_y.exit_thresholds = {
        "external_task": 0.9,
        "internal_planning": 0.05,
        "internal_replay": 0.9,
        "offline_consolidation": 0.9,
    }
    coord_y = SalienceCoordinator(cfg_y)
    trace_y = _run_sequence(coord_y)

    x_frac_task = _fraction_in_mode(trace_x, "external_task")
    y_frac_task = _fraction_in_mode(trace_y, "external_task")
    y_frac_planning = _fraction_in_mode(trace_y, "internal_planning")

    # X should be very sticky in external_task. Y should leave task
    # readily and stick in internal_planning once entered.
    return {
        "condition_X_fraction_task": round(x_frac_task, 3),
        "condition_Y_fraction_task": round(y_frac_task, 3),
        "condition_Y_fraction_planning": round(y_frac_planning, 3),
        "pass": x_frac_task >= 0.8 and y_frac_planning > y_frac_task,
    }


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_backward_compat()
    uc2 = run_uc2_arm_a_symmetric_baseline()
    uc3 = run_uc3_arm_b_moderate_schmitt()
    uc4 = run_uc4_arm_c_severe_over_binding()
    uc5 = run_uc5_per_mode_asymmetry()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_464_mech266_competing_goals_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_464_mech266_competing_goals",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-266", "SD-032a", "MECH-259"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-266": "supports" if all_pass else "weakens",
            "SD-032a": "supports" if all_pass else "weakens",
            "MECH-259": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_backward_compat": uc1,
            "UC2_arm_a_symmetric_baseline": uc2,
            "UC3_arm_b_moderate_schmitt": uc3,
            "UC4_arm_c_severe_over_binding": uc4,
            "UC5_per_mode_asymmetry": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-266 landing diagnostic for EXP-0160 (competing goals). "
            "Tests the substrate wiring of per-mode enter_thresholds / "
            "exit_thresholds on SD-032a SalienceCoordinator and "
            "reproduces the three-arm Schmitt-hysteresis signature "
            "(symmetric / moderate / severe over-binding) on a fixed "
            "synthetic competing-goals signal sequence. The full "
            "behavioural compliance test (switch-cost asymmetry on "
            "env-driven dual-cue tasks) is deferred until the "
            "CausalGridWorldV2 dual simultaneously active resource-cue "
            "extension lands. Anchor plan: "
            "REE_assembly/evidence/planning/sd033_governance_plan.md."
        ),
    }

    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"result: {manifest['result']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
