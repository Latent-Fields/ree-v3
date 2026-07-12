"""V3-EXQ-465: MECH-267 x MECH-094 intrusive-simulation filtering
landing diagnostic (EXP-0161 -- "intrusive simulation filtering" row of
the SD-033 governance test battery, ocd4 thought file).

Purpose: diagnostic. Confirms the substrate-level interaction between
MECH-267 (mode-conditioned hippocampal proposal distribution) and
MECH-094 (hypothesis_tag write-gate on residue) under a simulated
internal_replay intrusion. Does NOT yet run the full behavioural 3-arm
test that EXP-0161 specifies (MECH-094 ON + MECH-267 ON vs MECH-094 OFF
vs MECH-267 OFF with forced replay injection and rollout-persistence
scoring) -- those require a forced-replay injection hook into the E3 /
hippocampal tick that is not yet on any roadmap item. This landing is
the substrate-side prerequisite for it.

Five sub-tests (deterministic module API + residue-write semantics):

  UC1 signature accepts operating_mode under replay semantics: as in
      EXP-0158 UC1, but the operating_mode pinned is internal_replay
      (the mode in which OCD-like intrusions would occur).

  UC2 disabled-by-default backward compat: mode_conditioning_enabled=
      False -> operating_mode={"internal_replay": 1.0} does not modulate
      CEM ao_std (last_mode_noise_scale is None). Safety net.

  UC3 internal_replay < external_task noise scale: with
      mode_conditioning_enabled=True, pinning operating_mode to
      {"internal_replay": 1.0} yields noise scale 0.5 (from
      mode_noise_scale default); pinning to {"external_task": 1.0}
      yields 1.0. Ratio replay/task <= 0.6 (replay proposes from a
      tightened, higher-precision distribution -- consistent with
      experience-derived rehearsal rather than broad task exploration).

  UC4 MECH-094 residue-write gate under simulated intrusion: invoking
      ResidueField.accumulate with hypothesis_tag=True (the replay /
      simulation tag per MECH-094) must NOT produce a change in the
      RBF weights; invoking it with hypothesis_tag=False from the same
      starting state MUST produce a positive weight delta. Asymmetric
      gate is the first-line filter that prevents intrusive simulated
      content from contaminating the consolidation substrate.

  UC5 ARM structural contrast (the EXP-0161 design intent):
      ARM A (MECH-094 ON + MECH-267 ON): internal_replay operating_mode
        yields reduced ao_std multiplier AND hypothesis_tag=True blocks
        the residue write. Both filters compose.
      ARM B (MECH-094 OFF): hypothesis_tag is ignored, simulated
        intrusion writes to residue. MECH-267 mode conditioning still
        differentiates noise scales but the residue is contaminated.
      ARM C (MECH-267 OFF): mode conditioning absent (noise scale None
        in both internal_replay and external_task). MECH-094 still
        blocks the write. Residue protected but proposal distribution
        is mode-insensitive -- intrusive content is proposed with the
        same CEM spread as task content.
      Structural assertion: ARM A is the only configuration in which
      both the proposal distribution is mode-appropriate AND the
      residue is write-protected.

Substrate-level acceptance only. Behavioural intrusion-filtering
contrasts (per EXP-0161 design) are deferred until the forced-replay
injection hook is implemented.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_465_mech267_intrusive_simulation_filtering.py

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

from ree_core.hippocampal.module import HippocampalModule
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.residue.field import ResidueField
from ree_core.utils.config import (
    E2Config,
    HippocampalConfig,
    ResidueConfig,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


def _make_hippocampal(mode_conditioning_enabled: bool = False) -> HippocampalModule:
    e2_cfg = E2Config(
        self_dim=16, world_dim=32, action_dim=4,
        action_object_dim=16, hidden_dim=64,
    )
    e2 = E2FastPredictor(e2_cfg)
    res_cfg = ResidueConfig(world_dim=32, hidden_dim=32, num_basis_functions=8)
    res = ResidueField(res_cfg)
    hip_cfg = HippocampalConfig(
        world_dim=32, action_dim=4, action_object_dim=16,
        hidden_dim=32, horizon=4, num_candidates=4, num_cem_iterations=1,
        mode_conditioning_enabled=mode_conditioning_enabled,
    )
    return HippocampalModule(hip_cfg, e2, res)


def _make_residue() -> ResidueField:
    res_cfg = ResidueConfig(world_dim=32, hidden_dim=32, num_basis_functions=8)
    return ResidueField(res_cfg)


def run_uc1_signature_accepts_operating_mode_replay() -> dict:
    """UC1: signature accepts operating_mode in replay-like shape."""
    torch.manual_seed(0)
    hip = _make_hippocampal(mode_conditioning_enabled=False)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)
    op_mode = {"internal_replay": 1.0}
    error = None
    try:
        trajs_propose = hip.propose_trajectories(
            z_world, z_self, num_candidates=4, operating_mode=op_mode,
        )
        trajs_forward = hip.forward(
            z_world, z_self, num_candidates=4, operating_mode=op_mode,
        )
        ok = isinstance(trajs_propose, list) and isinstance(trajs_forward, list)
    except Exception as exc:
        ok = False
        error = repr(exc)
    return {
        "accepts_via_propose": ok,
        "accepts_via_forward": ok,
        "error": error,
        "pass": ok,
    }


def run_uc2_disabled_by_default_backward_compat() -> dict:
    """UC2: mode_conditioning_enabled=False -> replay mode ignored."""
    torch.manual_seed(0)
    hip = _make_hippocampal(mode_conditioning_enabled=False)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)
    hip.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_replay": 1.0},
    )
    last_scale = hip._last_mode_noise_scale
    last_mode  = hip._last_operating_mode
    return {
        "last_mode_recorded": last_mode is not None,
        "last_noise_scale_is_none": last_scale is None,
        "pass": last_mode is not None and last_scale is None,
    }


def run_uc3_replay_vs_task_noise_scale() -> dict:
    """UC3: internal_replay noise scale is strictly less than external_task."""
    torch.manual_seed(0)
    hip = _make_hippocampal(mode_conditioning_enabled=True)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)

    hip.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_replay": 1.0},
    )
    scale_replay = hip._last_mode_noise_scale

    hip.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"external_task": 1.0},
    )
    scale_task = hip._last_mode_noise_scale

    ratio = (scale_replay or 0.0) / max(scale_task or 1e-12, 1e-12)
    return {
        "scale_internal_replay": scale_replay,
        "scale_external_task": scale_task,
        "ratio_replay_over_task": ratio,
        "pass": (
            scale_replay is not None
            and scale_task is not None
            and ratio <= 0.6
        ),
    }


def run_uc4_mech094_residue_gate_under_intrusion() -> dict:
    """UC4: hypothesis_tag=True blocks residue write; False permits it."""
    torch.manual_seed(0)
    z_world = torch.randn(1, 32)
    harm_magnitude = 1.0

    # Baseline: tagged write (simulation / replay content).
    res_tagged = _make_residue()
    w_before_tagged = res_tagged.rbf_field.weights.detach().clone()
    res_tagged.accumulate(z_world, harm_magnitude, hypothesis_tag=True)
    w_after_tagged = res_tagged.rbf_field.weights.detach().clone()
    delta_tagged = (w_after_tagged - w_before_tagged).abs().sum().item()

    # Contrast: untagged write (waking, real observation).
    res_untagged = _make_residue()
    w_before_untagged = res_untagged.rbf_field.weights.detach().clone()
    res_untagged.accumulate(z_world, harm_magnitude, hypothesis_tag=False)
    w_after_untagged = res_untagged.rbf_field.weights.detach().clone()
    delta_untagged = (w_after_untagged - w_before_untagged).abs().sum().item()

    return {
        "tagged_weight_delta": delta_tagged,
        "untagged_weight_delta": delta_untagged,
        "pass": delta_tagged == 0.0 and delta_untagged > 0.0,
    }


def run_uc5_arm_structural_contrast() -> dict:
    """UC5: ARM A (both filters composed) is the only complete configuration.

    EXP-0161 design intent at the substrate level:
      ARM A: MECH-094 ON + MECH-267 ON -- tagged intrusion blocked AND
             replay-mode proposal distribution is tightened.
      ARM B: MECH-094 OFF -- residue contaminated by tagged intrusion
             (hypothesis_tag is ignored). MECH-267 still tightens proposal.
      ARM C: MECH-267 OFF -- proposal distribution is mode-insensitive;
             MECH-094 still protects residue.
    """
    torch.manual_seed(0)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)
    harm_magnitude = 1.0

    # ARM A: both ON.
    hip_a = _make_hippocampal(mode_conditioning_enabled=True)
    res_a = _make_residue()
    hip_a.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_replay": 1.0},
    )
    a_scale = hip_a._last_mode_noise_scale
    w_before_a = res_a.rbf_field.weights.detach().clone()
    res_a.accumulate(z_world, harm_magnitude, hypothesis_tag=True)
    a_residue_delta = (
        res_a.rbf_field.weights - w_before_a
    ).abs().sum().item()
    arm_a_pass = (
        a_scale is not None
        and a_scale < 1.0
        and a_residue_delta == 0.0
    )

    # ARM B: MECH-094 OFF (simulated by not passing the tag -- i.e., an
    # experiment that forgets to stamp simulation content). MECH-267 ON.
    hip_b = _make_hippocampal(mode_conditioning_enabled=True)
    res_b = _make_residue()
    hip_b.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_replay": 1.0},
    )
    b_scale = hip_b._last_mode_noise_scale
    w_before_b = res_b.rbf_field.weights.detach().clone()
    res_b.accumulate(z_world, harm_magnitude, hypothesis_tag=False)
    b_residue_delta = (
        res_b.rbf_field.weights - w_before_b
    ).abs().sum().item()
    arm_b_contaminated = b_residue_delta > 0.0
    arm_b_pass = (
        b_scale is not None
        and b_scale < 1.0
        and arm_b_contaminated
    )  # ARM B SHOULD contaminate -- that is the falsification signature.

    # ARM C: MECH-267 OFF, MECH-094 ON.
    hip_c = _make_hippocampal(mode_conditioning_enabled=False)
    res_c = _make_residue()
    hip_c.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_replay": 1.0},
    )
    c_scale_replay = hip_c._last_mode_noise_scale
    hip_c.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"external_task": 1.0},
    )
    c_scale_task = hip_c._last_mode_noise_scale
    w_before_c = res_c.rbf_field.weights.detach().clone()
    res_c.accumulate(z_world, harm_magnitude, hypothesis_tag=True)
    c_residue_delta = (
        res_c.rbf_field.weights - w_before_c
    ).abs().sum().item()
    arm_c_pass = (
        c_scale_replay is None
        and c_scale_task is None
        and c_residue_delta == 0.0
    )

    return {
        "arm_a_noise_scale": a_scale,
        "arm_a_residue_delta": a_residue_delta,
        "arm_a_pass": arm_a_pass,
        "arm_b_noise_scale": b_scale,
        "arm_b_residue_delta": b_residue_delta,
        "arm_b_contaminated_as_expected": arm_b_contaminated,
        "arm_b_pass": arm_b_pass,
        "arm_c_noise_scale_replay": c_scale_replay,
        "arm_c_noise_scale_task": c_scale_task,
        "arm_c_residue_delta": c_residue_delta,
        "arm_c_pass": arm_c_pass,
        "pass": arm_a_pass and arm_b_pass and arm_c_pass,
    }


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_signature_accepts_operating_mode_replay()
    uc2 = run_uc2_disabled_by_default_backward_compat()
    uc3 = run_uc3_replay_vs_task_noise_scale()
    uc4 = run_uc4_mech094_residue_gate_under_intrusion()
    uc5 = run_uc5_arm_structural_contrast()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_465_mech267_intrusive_simulation_filtering_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_465_mech267_intrusive_simulation_filtering",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-267", "MECH-094", "MECH-261"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-267": "supports" if all_pass else "weakens",
            "MECH-094": "supports" if all_pass else "weakens",
            "MECH-261": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_signature_accepts_operating_mode_replay": uc1,
            "UC2_disabled_by_default_backward_compat": uc2,
            "UC3_replay_vs_task_noise_scale": uc3,
            "UC4_mech094_residue_gate_under_intrusion": uc4,
            "UC5_arm_structural_contrast": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-267 x MECH-094 landing diagnostic for EXP-0161 (intrusive "
            "simulation filtering). Tests substrate wiring of operating_mode "
            "through HippocampalModule.propose_trajectories under "
            "internal_replay semantics, and confirms the MECH-094 "
            "hypothesis_tag write-gate still asymmetrically protects "
            "ResidueField.accumulate. Behavioural intrusion-filtering "
            "contrasts (the full ARM-A/B/C design from EXP-0161) are "
            "deferred to a behavioural successor pending implementation "
            "of a forced-replay injection hook into the E3 / hippocampal "
            "tick. UC5 establishes the structural ARM-A vs ARM-B vs ARM-C "
            "contrast at the substrate level: ARM A is the only "
            "configuration where both the proposal distribution is "
            "mode-appropriate AND the residue is write-protected."
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
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
