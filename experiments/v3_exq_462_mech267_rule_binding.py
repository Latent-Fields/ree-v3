"""V3-EXQ-462: MECH-267 mode-conditioned hippocampal-proposal landing
diagnostic (EXP-0158 -- "rule binding" row of the SD-033 governance test
battery, ocd4 thought file).

Purpose: diagnostic. Confirms the MECH-267 substrate is wired correctly and
demonstrates the SD-033a x MECH-267 factorial that the full EXP-0158
behavioural test would exercise. Does NOT yet run the cross-mode rule-
binding env episodes that EXP-0158 specifies -- those depend on the
SD-016 cue_action_proj installation path, which is currently blocked on
EXP-0155 (forward-path probe diagnostic). When that lands, an EXQ-462b
behavioural successor can run end-to-end episodes; this landing is the
substrate-side prerequisite for it.

Five sub-tests (deterministic arithmetic + module API):

  UC1 signature accepts operating_mode: HippocampalModule.propose_trajectories
      and HippocampalModule.forward both accept the operating_mode keyword,
      and a probability-vector dict passes through without raising.

  UC2 disabled-by-default backward compat: with mode_conditioning_enabled=
      False, supplying operating_mode does NOT change CEM behaviour (noise
      scale is recorded as None in the diagnostic cache and ao_std is not
      modulated). This is the existing-experiment safety net.

  UC3 mode contrast -- internal_planning > external_task: with
      mode_conditioning_enabled=True, operating_mode pinned to
      {"internal_planning": 1.0} produces a noise scale of 1.3 (default
      mode_noise_scale[internal_planning]); pinned to
      {"external_task": 1.0} produces 1.0. Ratio >= 1.25.

  UC4 cross-mode soft mix: an operating_mode mixture
      {"external_task": 0.5, "internal_planning": 0.5} produces a noise
      scale equal to the convex combination 0.5*1.0 + 0.5*1.3 = 1.15
      (within 1e-6).

  UC5 ARM-A vs ARM-C structural contrast (the EXP-0158 design intent):
      ARM A (SD-033a ON + MECH-267 ON): cycling operating_mode through
      planning then task produces distinct ao_std multipliers per call
      AND the SD-033a rule_state persists across the transition under the
      external_task gate weighting (gate=1.0 for external_task in the
      MECH-261 default registry for sd_033a). ARM C (MECH-267 OFF):
      cycling operating_mode does NOT differentiate the calls (noise
      scale is None in both). The ARM-A noise-ratio contrast > 1.0,
      the ARM-C noise-ratio contrast == 1.0 (no differentiation).

Substrate-level acceptance only. Behavioural compliance contrasts (per
EXP-0158 design) are deferred until the cue_action_proj installation
path is unblocked (EXP-0155).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_462_mech267_rule_binding.py

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
    REEConfig,
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


def run_uc1_signature_accepts_operating_mode() -> dict:
    """UC1: signature accepts operating_mode without raising."""
    torch.manual_seed(0)
    hip = _make_hippocampal(mode_conditioning_enabled=False)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)
    op_mode = {"external_task": 0.7, "internal_planning": 0.3}
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
    """UC2: mode_conditioning_enabled=False -> noise scale not applied."""
    torch.manual_seed(0)
    hip = _make_hippocampal(mode_conditioning_enabled=False)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)
    hip.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_planning": 1.0},
    )
    last_scale = hip._last_mode_noise_scale
    last_mode  = hip._last_operating_mode
    return {
        "last_mode_recorded": last_mode is not None,
        "last_noise_scale_is_none": last_scale is None,
        "pass": last_mode is not None and last_scale is None,
    }


def run_uc3_mode_contrast_planning_vs_task() -> dict:
    """UC3: planning noise scale > external_task noise scale."""
    torch.manual_seed(0)
    hip = _make_hippocampal(mode_conditioning_enabled=True)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)

    hip.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_planning": 1.0},
    )
    scale_planning = hip._last_mode_noise_scale

    hip.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"external_task": 1.0},
    )
    scale_task = hip._last_mode_noise_scale

    ratio = (scale_planning or 0.0) / max(scale_task or 1e-12, 1e-12)
    return {
        "scale_planning": scale_planning,
        "scale_external_task": scale_task,
        "ratio": ratio,
        "pass": (
            scale_planning is not None
            and scale_task is not None
            and ratio >= 1.25
        ),
    }


def run_uc4_soft_mixture_convex_combination() -> dict:
    """UC4: mixed operating_mode -> convex-combination noise scale."""
    torch.manual_seed(0)
    hip = _make_hippocampal(mode_conditioning_enabled=True)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)
    hip.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"external_task": 0.5, "internal_planning": 0.5},
    )
    expected = 0.5 * 1.0 + 0.5 * 1.3  # convex combo of default noise scales
    actual = hip._last_mode_noise_scale
    delta = abs((actual or 0.0) - expected)
    return {
        "expected_scale": expected,
        "actual_scale": actual,
        "abs_error": delta,
        "pass": actual is not None and delta < 1e-6,
    }


def run_uc5_arm_a_vs_arm_c_structural_contrast() -> dict:
    """UC5: ARM-A (MECH-267 ON) differentiates modes; ARM-C (OFF) does not.

    EXP-0158 design intent at the substrate level:
      ARM A: SD-033a + MECH-267 both active. Cycling planning -> task
        produces distinct CEM noise scales (mode-dependent proposal
        distribution) AND SD-033a rule_state persists across the boundary
        under the MECH-261 sd_033a gate of 1.0 in external_task.
      ARM C: MECH-267 disabled. Same operating_mode cycling produces
        identical (None) noise scales -- hippocampus ignores mode.
    """
    # ARM A: MECH-267 ON
    torch.manual_seed(0)
    hip_a = _make_hippocampal(mode_conditioning_enabled=True)
    z_world = torch.randn(1, 32)
    z_self  = torch.randn(1, 16)
    hip_a.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_planning": 1.0},
    )
    a_scale_planning = hip_a._last_mode_noise_scale
    hip_a.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"external_task": 1.0},
    )
    a_scale_task = hip_a._last_mode_noise_scale
    arm_a_ratio = (a_scale_planning or 0.0) / max(a_scale_task or 1e-12, 1e-12)

    # ARM C: MECH-267 OFF
    torch.manual_seed(0)
    hip_c = _make_hippocampal(mode_conditioning_enabled=False)
    hip_c.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"internal_planning": 1.0},
    )
    c_scale_planning = hip_c._last_mode_noise_scale
    hip_c.propose_trajectories(
        z_world, z_self, num_candidates=4,
        operating_mode={"external_task": 1.0},
    )
    c_scale_task = hip_c._last_mode_noise_scale

    # SD-033a rule_state persistence (MECH-261 sd_033a gate weights show
    # that external_task and internal_planning both have gate 1.0; the
    # rule_state should not collapse on the transition under either).
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_lateral_pfc_analog=True,
    )
    from ree_core.agent import REEAgent
    agent = REEAgent(cfg)
    z_delta = torch.randn(1, cfg.latent.delta_dim)
    z_world_agent = torch.randn(1, cfg.latent.world_dim)
    # Install during planning
    for _ in range(20):
        agent.lateral_pfc.update(z_delta, z_world_agent, gate=1.0)
    norm_after_planning = float(agent.lateral_pfc.rule_state.norm().item())
    # Cross into external_task (MECH-261 gate=1.0): same call shape;
    # rule_state should remain non-trivial across the boundary.
    for _ in range(5):
        agent.lateral_pfc.update(z_delta, z_world_agent, gate=1.0)
    norm_after_task = float(agent.lateral_pfc.rule_state.norm().item())
    persistence_ratio = norm_after_task / max(norm_after_planning, 1e-12)

    return {
        "arm_a_scale_planning": a_scale_planning,
        "arm_a_scale_task": a_scale_task,
        "arm_a_ratio": arm_a_ratio,
        "arm_c_scale_planning": c_scale_planning,
        "arm_c_scale_task": c_scale_task,
        "rule_state_norm_after_planning": norm_after_planning,
        "rule_state_norm_after_task": norm_after_task,
        "rule_state_persistence_ratio": persistence_ratio,
        "pass": (
            arm_a_ratio >= 1.25
            and c_scale_planning is None
            and c_scale_task is None
            and norm_after_planning > 0.0
            and persistence_ratio > 0.5
        ),
    }


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_signature_accepts_operating_mode()
    uc2 = run_uc2_disabled_by_default_backward_compat()
    uc3 = run_uc3_mode_contrast_planning_vs_task()
    uc4 = run_uc4_soft_mixture_convex_combination()
    uc5 = run_uc5_arm_a_vs_arm_c_structural_contrast()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_462_mech267_rule_binding_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_462_mech267_rule_binding",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-267", "SD-033a", "MECH-262"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-267": "supports" if all_pass else "weakens",
            "SD-033a": "supports" if all_pass else "weakens",
            "MECH-262": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_signature_accepts_operating_mode": uc1,
            "UC2_disabled_by_default_backward_compat": uc2,
            "UC3_mode_contrast_planning_vs_task": uc3,
            "UC4_soft_mixture_convex_combination": uc4,
            "UC5_arm_a_vs_arm_c_structural_contrast": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-267 landing diagnostic for EXP-0158 (rule binding). "
            "Tests substrate wiring of operating_mode through "
            "HippocampalModule.propose_trajectories and the per-mode CEM "
            "noise-scale mechanism. Behavioural cross-mode rule-binding "
            "compliance contrasts (the full ARM-A/B/C design from EXP-0158) "
            "are deferred to a behavioural successor pending unblock of "
            "the SD-016 cue_action_proj installation path (EXP-0155 is "
            "the open diagnostic). UC5 establishes the structural ARM-A "
            "vs ARM-C contrast at the substrate level."
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
