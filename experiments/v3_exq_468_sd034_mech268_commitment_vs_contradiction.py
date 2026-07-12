"""V3-EXQ-468 (EXP-0164): SD-034 x MECH-268 commitment vs contradiction.

Purpose: diagnostic. Validates the four-arm separability of the
SD-034 closure-on-completion path and the MECH-268 pe-saturation-on-
sustained-counter-evidence path. The ocd4 commitment-vs-contradiction
row asks: how does the system adjudicate between "completion achieved,
release via closure" and "counter-evidence mounting, release via pe
habituation"?

This is the substrate landing test. The EXP-0164 spec calls for a
behavioural arm with counter-evidence injection during commitment
and release-type distribution measurement; that requires an env
extension (counter-evidence injection hook) and a full E3 task loop
that is not yet on roadmap. This landing validates the wiring so the
behavioural variants can be authored.

Six sub-tests (all arithmetic + API):

  UC1 backward compat (Arm D, both OFF): no closure firing, no pe
      saturation; pipeline reduces to baseline MECH-258 path. Outcome
      history pushed but sat_factor stays 1.0; emit_closure returns
      fired=False with reason "skipped:disabled".

  UC2 Arm A (both ON): with beta elevated and pe attenuated by
      sustained outcomes, emit_closure releases beta, injects No-Go,
      resets pe EMA, AND resets outcome history -- so sat_factor
      returns to 1.0 on the post-closure cycle. Confirms the two
      mechanisms compose without conflict.

  UC3 Arm B (SD-034 ON, MECH-268 OFF): closure fires + releases beta
      + injects No-Go; sat_factor is 1.0 throughout (saturation never
      activates without dacc_saturation_enabled). Pe grows monotone in
      precision-weighted PE without an upper habituation limit other
      than the SD-034 absolute pe_cap. Proves the two paths are
      mechanistically separable: closure on completion does NOT
      require saturation.

  UC4 Arm C (SD-034 OFF, MECH-268 ON): closure cannot fire (the
      operator returns "skipped:disabled"); beta stays elevated.
      Saturation still attenuates pe as identical outcomes accumulate.
      Proves saturation does NOT require closure -- the habituation
      release pathway exists independently.

  UC5 mode-conditioning blocks closure but not saturation: with
      both ON, force the salience layer to report a disallowed mode
      (e.g. internal_replay) so closure refuses to fire; outcome
      history continues to accumulate and sat_factor falls below 1.0.
      Confirms the falsifiability predicate: SD-034 mode-gating does
      not gate MECH-268.

  UC6 closure-after-saturation interaction: with both ON, drive
      sat_factor below 0.5 via sustained identical outcomes, then
      emit_closure with bypass. Verify (a) the cycle that fires uses
      the still-attenuated pe (so closure carries the historical
      signal) and (b) the next cycle starts with sat_factor=1.0
      because the buffer was reset. This is the "release-type
      distribution" landing observation: closure's effect on pe is
      forward-looking (resets baseline), not retroactive.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_468_sd034_mech268_commitment_vs_contradiction.py

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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


class _StubBetaGate:
    """Minimal beta gate stub: tracks elevation; supports release()."""

    def __init__(self) -> None:
        self.is_elevated = False

    def elevate(self) -> None:
        self.is_elevated = True

    def release(self) -> None:
        self.is_elevated = False


def _build(
    sd034_on: bool,
    mech268_on: bool,
    saturation_window: int = 6,
    saturation_strength: float = 0.5,
    saturation_grace: int = 2,
):
    dacc_cfg = DACCConfig(
        dacc_saturation_enabled=mech268_on,
        dacc_saturation_window=saturation_window,
        dacc_saturation_strength=saturation_strength,
        dacc_saturation_grace=saturation_grace,
    )
    dacc = DACCAdaptiveControl(dacc_cfg)
    beta = _StubBetaGate()
    closure_cfg = ClosureOperatorConfig(
        use_closure_operator=sd034_on,
        reset_pe_ema=True,
        reset_outcome_history=True,
    )
    closure = ClosureOperator(
        config=closure_cfg,
        dacc=dacc,
        beta_gate=beta,
        residue=None,
        salience=None,
    )
    return dacc, beta, closure


def _push_outcomes(dacc: DACCAdaptiveControl, classes) -> None:
    for c in classes:
        dacc.record_outcome(c)


def run_uc1_arm_d_both_off() -> dict:
    dacc, beta, closure = _build(sd034_on=False, mech268_on=False)
    _push_outcomes(dacc, [0] * 6)
    sat, n_rec = dacc._saturation_factor(0)
    z_world = torch.zeros(1, 32)
    event = closure.emit_closure(
        action_class=0,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    result = {
        "sat_factor": sat,
        "n_recurrences": n_rec,
        "closure_fired": event.fired,
        "closure_reason": event.reason,
    }
    result["pass"] = (
        sat == 1.0
        and not event.fired
        and "disabled" in (event.reason or "")
    )
    return result


def run_uc2_arm_a_both_on() -> dict:
    dacc, beta, closure = _build(sd034_on=True, mech268_on=True)
    beta.elevate()
    dacc._pe_ema = 0.7
    _push_outcomes(dacc, [0] * 6)
    sat_pre, _ = dacc._saturation_factor(0)
    z_world = torch.zeros(1, 32)
    event = closure.emit_closure(
        action_class=0,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    sat_post, n_post = dacc._saturation_factor(0)
    result = {
        "sat_pre": sat_pre,
        "sat_post": sat_post,
        "n_post": n_post,
        "fired": event.fired,
        "beta_released": event.beta_released,
        "outcome_history_reset": event.outcome_history_reset,
        "pe_ema_reset": event.pe_ema_reset,
        "nogo_pushed": event.nogo_pushed,
        "post_pe_ema": dacc._pe_ema,
    }
    result["pass"] = (
        event.fired
        and sat_pre < 1.0
        and sat_post == 1.0
        and event.beta_released
        and event.outcome_history_reset
        and event.pe_ema_reset
        and dacc._pe_ema is None
        and event.nogo_pushed > 0
        and not beta.is_elevated
    )
    return result


def run_uc3_arm_b_sd034_only() -> dict:
    dacc, beta, closure = _build(sd034_on=True, mech268_on=False)
    beta.elevate()
    _push_outcomes(dacc, [0] * 6)
    sat_pre, _ = dacc._saturation_factor(0)
    # Confirm forward bundle leaves pe unsaturated.
    z_h = torch.tensor([1.0, 0.0, 0.0, 0.0])
    z_h_pred = torch.zeros(4)
    bundle = dacc.forward(
        z_harm_a=z_h,
        z_harm_a_pred=z_h_pred,
        candidate_payoffs=torch.tensor([0.5, 0.5]),
        candidate_effort=torch.tensor([0.1, 0.1]),
        candidate_action_classes=[0, 1],
        precision=100.0,
        drive_level=0.0,
        current_outcome_class=0,
    )
    z_world = torch.zeros(1, 32)
    event = closure.emit_closure(
        action_class=0,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    result = {
        "sat_pre": sat_pre,
        "bundle_pe": float(bundle["pe"]),
        "bundle_pe_unsaturated": float(bundle["pe_unsaturated"]),
        "bundle_saturation_factor": float(bundle["saturation_factor"]),
        "fired": event.fired,
        "beta_released": event.beta_released,
        "nogo_pushed": event.nogo_pushed,
    }
    result["pass"] = (
        sat_pre == 1.0
        and abs(bundle["pe"] - bundle["pe_unsaturated"]) < 1e-9
        and bundle["saturation_factor"] == 1.0
        and event.fired
        and event.beta_released
        and event.nogo_pushed > 0
    )
    return result


def run_uc4_arm_c_mech268_only() -> dict:
    dacc, beta, closure = _build(sd034_on=False, mech268_on=True)
    beta.elevate()
    _push_outcomes(dacc, [0] * 6)
    sat_after_history, n_rec = dacc._saturation_factor(0)
    z_world = torch.zeros(1, 32)
    event = closure.emit_closure(
        action_class=0,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    result = {
        "sat_after_history": sat_after_history,
        "n_recurrences": n_rec,
        "closure_fired": event.fired,
        "closure_reason": event.reason,
        "beta_still_elevated": beta.is_elevated,
    }
    result["pass"] = (
        sat_after_history < 1.0
        and not event.fired
        and "disabled" in (event.reason or "")
        and beta.is_elevated
    )
    return result


def run_uc5_mode_conditioning_blocks_closure_only() -> dict:
    dacc, beta, closure = _build(sd034_on=True, mech268_on=True)
    beta.elevate()
    _push_outcomes(dacc, [0] * 5)
    sat_before, _ = dacc._saturation_factor(0)
    z_world = torch.zeros(1, 32)
    # Disallowed mode + do NOT bypass.
    event = closure.emit_closure(
        action_class=0,
        z_world=z_world,
        current_mode="offline_consolidation",
        sd033a_gate=1.0,
        bypass_mode_conditioning=False,
    )
    # Push another outcome -- saturation should keep activating.
    dacc.record_outcome(0)
    sat_after, n_after = dacc._saturation_factor(0)
    result = {
        "sat_before": sat_before,
        "sat_after": sat_after,
        "n_after": n_after,
        "fired": event.fired,
        "reason": event.reason,
        "beta_still_elevated": beta.is_elevated,
    }
    result["pass"] = (
        not event.fired
        and "mode_disallowed" in (event.reason or "")
        and beta.is_elevated
        and sat_after < 1.0
        and sat_after <= sat_before
    )
    return result


def run_uc6_closure_after_saturation_interaction() -> dict:
    dacc, beta, closure = _build(sd034_on=True, mech268_on=True)
    beta.elevate()
    _push_outcomes(dacc, [0] * 8)
    sat_pre, _ = dacc._saturation_factor(0)
    # Compute attenuated pe_during_closure_cycle BEFORE firing.
    z_h = torch.tensor([1.0, 0.0, 0.0, 0.0])
    z_h_pred = torch.zeros(4)
    bundle_pre = dacc.forward(
        z_harm_a=z_h,
        z_harm_a_pred=z_h_pred,
        candidate_payoffs=torch.tensor([0.5, 0.5]),
        candidate_effort=torch.tensor([0.1, 0.1]),
        candidate_action_classes=[0, 1],
        precision=100.0,
        drive_level=0.0,
        current_outcome_class=0,
    )
    pe_during_closure = float(bundle_pre["pe"])
    sat_during_closure = float(bundle_pre["saturation_factor"])

    z_world = torch.zeros(1, 32)
    event = closure.emit_closure(
        action_class=0,
        z_world=z_world,
        bypass_mode_conditioning=True,
    )
    sat_post, _ = dacc._saturation_factor(0)
    # Forward bundle on the next cycle should now show sat_factor=1.0.
    bundle_post = dacc.forward(
        z_harm_a=z_h,
        z_harm_a_pred=z_h_pred,
        candidate_payoffs=torch.tensor([0.5, 0.5]),
        candidate_effort=torch.tensor([0.1, 0.1]),
        candidate_action_classes=[0, 1],
        precision=100.0,
        drive_level=0.0,
        current_outcome_class=None,
    )
    result = {
        "sat_pre": sat_pre,
        "pe_during_closure": pe_during_closure,
        "sat_during_closure": sat_during_closure,
        "fired": event.fired,
        "outcome_history_reset": event.outcome_history_reset,
        "sat_post": sat_post,
        "bundle_post_saturation_factor": float(bundle_post["saturation_factor"]),
        "bundle_post_pe": float(bundle_post["pe"]),
        "bundle_post_pe_unsaturated": float(bundle_post["pe_unsaturated"]),
    }
    result["pass"] = (
        sat_during_closure < 0.5
        and event.fired
        and event.outcome_history_reset
        and sat_post == 1.0
        and bundle_post["saturation_factor"] == 1.0
        and abs(bundle_post["pe"] - bundle_post["pe_unsaturated"]) < 1e-9
    )
    return result


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_arm_d_both_off()
    uc2 = run_uc2_arm_a_both_on()
    uc3 = run_uc3_arm_b_sd034_only()
    uc4 = run_uc4_arm_c_mech268_only()
    uc5 = run_uc5_mode_conditioning_blocks_closure_only()
    uc6 = run_uc6_closure_after_saturation_interaction()

    subtests = {
        "UC1_arm_d_both_off": uc1,
        "UC2_arm_a_both_on": uc2,
        "UC3_arm_b_sd034_only": uc3,
        "UC4_arm_c_mech268_only": uc4,
        "UC5_mode_conditioning_blocks_closure_only": uc5,
        "UC6_closure_after_saturation_interaction": uc6,
    }
    all_pass = all(r["pass"] for r in subtests.values())
    elapsed = time.time() - t0

    run_id = "v3_exq_468_sd034_mech268_commitment_vs_contradiction_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_468_sd034_mech268_commitment_vs_contradiction",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["SD-034", "MECH-268", "MECH-090"],
        "claim_ids_tested": ["SD-034", "MECH-268", "MECH-090"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "SD-034": "supports" if (uc2["pass"] and uc3["pass"] and uc5["pass"] and uc6["pass"]) else "weakens",
            "MECH-268": "supports" if (uc2["pass"] and uc4["pass"] and uc5["pass"] and uc6["pass"]) else "weakens",
            "MECH-090": "supports" if (uc2["pass"] and uc3["pass"] and uc4["pass"]) else "weakens",
        },
        "metrics": subtests,
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-034 x MECH-268 four-arm landing diagnostic (ocd4 "
            "commitment-vs-contradiction row). Validates that the "
            "closure-on-completion (SD-034) and pe-saturation-on-"
            "sustained-counter-evidence (MECH-268) release pathways are "
            "mechanistically separable but jointly load-bearing: Arm A "
            "(both ON) shows clean closure + pe rebaselining; Arm B "
            "(SD-034 only) confirms closure fires without needing "
            "saturation; Arm C (MECH-268 only) confirms habituation "
            "fires without needing closure; Arm D (both OFF) is "
            "baseline. UC5 confirms the falsifiability predicate -- "
            "mode-conditioning gates closure but does NOT gate "
            "saturation. UC6 confirms the closure cycle uses the "
            "still-attenuated pe (closure carries the historical "
            "signal) but the buffer reset means the next cycle "
            "starts unsaturated -- forward-looking, not retroactive. "
            "Behavioural arm (env counter-evidence injection + release-"
            "type distribution / time-to-release / post-release residue "
            "patterning across all four arms) deferred -- depends on "
            "env extension that is not yet on roadmap. Anchor plan: "
            "REE_assembly/evidence/planning/sd033_governance_plan.md. "
            "Source: docs/thoughts/2026-04-20_ocd4.md row 'commitment "
            "vs contradiction'."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

    print(f"result: {manifest['result']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
