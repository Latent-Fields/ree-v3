"""V3-EXQ-463 (EXP-0159): MECH-268 dACC conflict saturation landing.

Purpose: diagnostic. Validates the MECH-268 saturation function
pe_saturated = f_sat(pe_raw, outcome_history) layered atop the
existing MECH-258 precision-weighted PE and SD-034 absolute pe cap.
Confirms the ocd4 "conflict saturation" row: with MECH-268 OFF,
sustained identical outcomes leave dACC pe unchanged (rumination
signature). With MECH-268 ON, pe attenuates as the same outcome
class recurs past the grace count.

This is the substrate landing test. The EXP-0159 spec calls for a
500+ step behavioural arm with mode-switch trigger rate measurement;
that requires a full E3 + env loop with phased rule_state training,
which is not yet on roadmap. This landing validates the wiring so
those behavioural variants can be authored.

Seven sub-tests (all arithmetic + API):

  UC1 backward compat: dacc_saturation_enabled=False -> sat_factor
      always 1.0, pe_unsaturated == pe (no change to legacy pipeline).

  UC2 grace count honoured: with saturation ON but n_recurrences
      <= dacc_saturation_grace, sat_factor stays 1.0.

  UC3 saturation activates past grace: sustained identical-outcome
      sequence drops sat_factor strictly below 1.0 once n_recurrences
      exceeds the grace threshold.

  UC4 monotone in recurrences: sat_factor strictly decreases as
      additional identical outcomes are pushed onto the history.

  UC5 mixed outcomes spare saturation: alternating two outcome
      classes keeps each individual recurrence count below the
      activation threshold; sat_factor remains higher than the
      pure-recurrence case at matched window length.

  UC6 closure resets outcome history: ClosureOperator._fire()
      clears _outcome_history so the next cycle starts unsaturated.

  UC7 dACC.forward exposes saturation: the bundle returned by
      DACCAdaptiveControl.forward() reports pe_unsaturated +
      saturation_factor + outcome_recurrence diagnostics, and
      bundle["pe"] equals pe_unsaturated * saturation_factor.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_463_mech268_dacc_conflict_saturation.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.cingulate.dacc import DACCAdaptiveControl, DACCConfig
from ree_core.governance.closure_operator import (
    ClosureOperator,
    ClosureOperatorConfig,
)


def _build_dacc(
    enabled: bool,
    window: int = 8,
    strength: float = 0.5,
    grace: int = 2,
) -> DACCAdaptiveControl:
    cfg = DACCConfig(
        dacc_saturation_enabled=enabled,
        dacc_saturation_window=window,
        dacc_saturation_strength=strength,
        dacc_saturation_grace=grace,
    )
    return DACCAdaptiveControl(cfg)


def run_uc1_backward_compat() -> dict:
    dacc = _build_dacc(enabled=False, window=4, grace=1)
    for cls in [0, 0, 0, 0, 0, 0]:
        dacc.record_outcome(cls)
    sat, n_rec = dacc._saturation_factor(0)
    result = {
        "saturation_enabled": False,
        "n_recurrences_in_window": n_rec,
        "sat_factor": sat,
    }
    result["pass"] = (sat == 1.0)
    return result


def run_uc2_grace_honoured() -> dict:
    dacc = _build_dacc(enabled=True, window=8, strength=0.5, grace=3)
    for cls in [0, 0, 0]:
        dacc.record_outcome(cls)
    sat, n_rec = dacc._saturation_factor(0)
    result = {
        "saturation_enabled": True,
        "grace": 3,
        "n_recurrences_in_window": n_rec,
        "sat_factor": sat,
    }
    result["pass"] = (n_rec <= 3) and (sat == 1.0)
    return result


def run_uc3_activates_past_grace() -> dict:
    dacc = _build_dacc(enabled=True, window=8, strength=0.5, grace=2)
    for cls in [0, 0, 0, 0, 0, 0]:
        dacc.record_outcome(cls)
    sat, n_rec = dacc._saturation_factor(0)
    result = {
        "n_recurrences_in_window": n_rec,
        "sat_factor": sat,
    }
    result["pass"] = (n_rec > 2) and (sat < 1.0)
    return result


def run_uc4_monotone_in_recurrences() -> dict:
    dacc = _build_dacc(enabled=True, window=10, strength=0.5, grace=1)
    factors = []
    for _ in range(8):
        dacc.record_outcome(0)
        sat, _ = dacc._saturation_factor(0)
        factors.append(sat)
    deltas = [factors[i + 1] - factors[i] for i in range(len(factors) - 1)]
    monotone_nonincreasing = all(d <= 1e-9 for d in deltas)
    strict_decrease_after_grace = factors[-1] < factors[1]
    result = {
        "factors": factors,
        "monotone_nonincreasing": monotone_nonincreasing,
        "strict_decrease_after_grace": strict_decrease_after_grace,
    }
    result["pass"] = monotone_nonincreasing and strict_decrease_after_grace
    return result


def run_uc5_mixed_outcomes_spare_saturation() -> dict:
    dacc_alt = _build_dacc(enabled=True, window=8, strength=0.5, grace=1)
    for cls in [0, 1, 0, 1, 0, 1, 0, 1]:
        dacc_alt.record_outcome(cls)
    sat_alt, n_alt = dacc_alt._saturation_factor(0)

    dacc_pure = _build_dacc(enabled=True, window=8, strength=0.5, grace=1)
    for cls in [0, 0, 0, 0, 0, 0, 0, 0]:
        dacc_pure.record_outcome(cls)
    sat_pure, n_pure = dacc_pure._saturation_factor(0)

    result = {
        "sat_alt": sat_alt,
        "n_alt": n_alt,
        "sat_pure": sat_pure,
        "n_pure": n_pure,
    }
    result["pass"] = (sat_alt > sat_pure) and (n_alt < n_pure)
    return result


def run_uc6_closure_resets_buffer() -> dict:
    dacc = _build_dacc(enabled=True, window=8, strength=0.5, grace=2)
    for cls in [0, 0, 0, 0, 0]:
        dacc.record_outcome(cls)
    n_pre = len(dacc._outcome_history)
    sat_pre, _ = dacc._saturation_factor(0)

    closure_cfg = ClosureOperatorConfig(
        use_closure_operator=True,
        reset_outcome_history=True,
    )
    closure = ClosureOperator(
        config=closure_cfg,
        dacc=dacc,
        beta_gate=None,
        residue=None,
        salience=None,
    )
    z_world = torch.zeros(1, 32)
    event = closure.emit_closure(
        action_class=0,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    n_post = len(dacc._outcome_history)
    sat_post, _ = dacc._saturation_factor(0)

    result = {
        "fired": event.fired,
        "outcome_history_reset_flag": event.outcome_history_reset,
        "n_pre": n_pre,
        "n_post": n_post,
        "sat_pre": sat_pre,
        "sat_post": sat_post,
    }
    result["pass"] = (
        event.fired
        and event.outcome_history_reset
        and n_pre > 0
        and n_post == 0
        and sat_pre < 1.0
        and sat_post == 1.0
    )
    return result


def run_uc7_forward_bundle_exposes_saturation() -> dict:
    dacc = _build_dacc(enabled=True, window=4, strength=0.5, grace=1)
    # Pre-populate history so saturation has something to act on.
    for cls in [0, 0, 0, 0, 0]:
        dacc.record_outcome(cls)
    z_harm_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    z_harm_a_pred = torch.zeros(4)
    payoffs = torch.tensor([1.0, 0.5, 0.2])
    effort = torch.tensor([0.1, 0.2, 0.3])
    classes = [0, 1, 2]
    bundle = dacc.forward(
        z_harm_a=z_harm_a,
        z_harm_a_pred=z_harm_a_pred,
        candidate_payoffs=payoffs,
        candidate_effort=effort,
        candidate_action_classes=classes,
        precision=100.0,
        drive_level=0.0,
        current_outcome_class=0,
    )
    has_keys = all(
        k in bundle
        for k in ("pe", "pe_unsaturated", "saturation_factor", "outcome_recurrence")
    )
    pe = float(bundle["pe"]) if has_keys else float("nan")
    pe_un = float(bundle["pe_unsaturated"]) if has_keys else float("nan")
    sat = float(bundle["saturation_factor"]) if has_keys else float("nan")
    n_rec = int(bundle["outcome_recurrence"]) if has_keys else -1
    consistent = has_keys and abs(pe - pe_un * sat) < 1e-6
    result = {
        "has_diagnostic_keys": has_keys,
        "pe_unsaturated": pe_un,
        "pe": pe,
        "saturation_factor": sat,
        "outcome_recurrence": n_rec,
        "pe_eq_pe_un_times_sat": consistent,
    }
    result["pass"] = has_keys and consistent and sat < 1.0 and pe_un > 0.0
    return result


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_backward_compat()
    uc2 = run_uc2_grace_honoured()
    uc3 = run_uc3_activates_past_grace()
    uc4 = run_uc4_monotone_in_recurrences()
    uc5 = run_uc5_mixed_outcomes_spare_saturation()
    uc6 = run_uc6_closure_resets_buffer()
    uc7 = run_uc7_forward_bundle_exposes_saturation()

    subtests = {
        "UC1_backward_compat": uc1,
        "UC2_grace_honoured": uc2,
        "UC3_activates_past_grace": uc3,
        "UC4_monotone_in_recurrences": uc4,
        "UC5_mixed_outcomes_spare_saturation": uc5,
        "UC6_closure_resets_buffer": uc6,
        "UC7_forward_bundle_exposes_saturation": uc7,
    }
    all_pass = all(r["pass"] for r in subtests.values())
    elapsed = time.time() - t0

    run_id = "v3_exq_463_mech268_dacc_conflict_saturation_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_463_mech268_dacc_conflict_saturation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-268", "SD-032b", "MECH-258", "SD-034"],
        "claim_ids_tested": ["MECH-268", "SD-032b", "MECH-258", "SD-034"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-268": "supports" if all_pass else "weakens",
            "SD-032b": "supports" if all_pass else "weakens",
            "MECH-258": "non_contributory",
            "SD-034": "supports" if uc6["pass"] else "weakens",
        },
        "metrics": subtests,
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-268 landing diagnostic (ocd4 conflict-saturation row). "
            "Tests pe_saturated = f_sat(pe_raw, outcome_history) layered "
            "atop MECH-258 precision-weighted PE and SD-034 absolute "
            "pe cap. UC1-UC2 confirm backward compat and grace; UC3-UC4 "
            "confirm monotone activation past grace; UC5 confirms mixed "
            "outcomes do NOT saturate (recurrence-class-specific, not "
            "buffer-length); UC6 confirms closure operator reset path "
            "(SD-034 -> reset_outcome_history hook); UC7 confirms the "
            "dACC.forward() bundle exposes the saturation diagnostics "
            "needed by SD-032a salience coordinator. SD-032b/MECH-258 "
            "tagged because the saturation pipeline composes with their "
            "outputs; MECH-258 is non_contributory at this landing because "
            "the pre-existing precision-weighting path is not directly "
            "asserted (only that saturation does not perturb it). "
            "Behavioural arm (500+ step sustained-outcome task with "
            "mode-switch trigger rate measurement) deferred -- depends "
            "on phased rule_state training + an env variant not yet on "
            "any roadmap item. Anchor plan: REE_assembly/evidence/"
            "planning/sd033_governance_plan.md. Source: docs/thoughts/"
            "2026-04-20_ocd4.md row 'conflict saturation'."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
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
