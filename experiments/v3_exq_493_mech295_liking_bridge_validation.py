"""V3-EXQ-493: MECH-295 drive -> liking-stream -> approach_cue bridge validation.

Purpose: diagnostic. Confirms the MECH-295 bridge substrate is wired correctly
and produces the falsifiable signature for the WEAK-NECESSITY reading:

    weak reading: baseline liking-stream activation is sufficient, but the
    bridge wiring must be intact. If the cue side is severed (the link is
    broken), drive amplification produces no approach regardless of drive
    magnitude.

The clean direct test of the weak reading is a 2x2 factorial across the two
bridge gain knobs:

    write side x cue side
    -------------------------------------------
    mech295_drive_to_liking_gain in {0.0, 1.0}      (a) anticipatory write
    mech295_liking_to_approach_cue_gain in {0.0, 0.5} (b) approach cue

Four arms:

    A0 OFF_OFF   master OFF                     -- baseline / control
    A1 ON_OFF    master ON, both gains 0        -- bridge wired but silent
    A2 ON_WRITEONLY  master ON, write 1.0, cue 0 -- severed cue (dominant FAIL)
    A3 ON_FULL   master ON, write 1.0, cue 0.5  -- intact bridge

Six sub-tests (UC1-UC6):

  UC1 module_importable: regulators.mech295_liking_bridge importable;
      MECH295LikingBridge / Config / Output classes accessible. Pure module
      contract.

  UC2 master_off_no_op: REEAgent built with default config has
      agent.mech295_bridge is None (bit-identical OFF guarantee).

  UC3 write_side_fires: with master ON + drive_to_liking_gain=1.0, after a
      run of 30 ticks with forced drive=0.8 and benefit_exposure=0.4 (so
      z_goal seeds), bridge.get_diagnostics()['n_write_fires'] > 0.

  UC4 cue_side_negative_bias: with master ON + liking_to_approach_cue_gain=0.5,
      compute_approach_cue_score_bias(drive=0.6, prox=tensor([0.1, 0.5, 0.9]))
      returns negative tensor with bias[2] < bias[0] (more proximity ->
      more negative bias) AND bias_max_abs > 0.

  UC5 severed_bridge_collapse: with master ON + drive_to_liking_gain=1.0 +
      liking_to_approach_cue_gain=0.0, compute_approach_cue_score_bias on
      same inputs returns exactly-zero tensor. The write side STILL fires
      (n_write_fires > 0) confirming the asymmetry: the cue side is what
      reaches action selection.

  UC6 mech094_simulation_gate: with master ON + non-zero gains, calls with
      simulation_mode=True return zero on both sides AND counters do not
      advance.

PASS criteria: all 6 sub-tests PASS. The weak-necessity falsifiable
signature is captured by UC4 + UC5: drive elevated alone does NOT produce
approach (UC5 cue=0 produces zero bias regardless of drive); drive + cue
gain together DO produce approach pull (UC4 negative bias).

Behavioural validation (EXQ-483-style 4-arm with the orexin substrate ON
plus this bridge in arms 2-3, measuring approach_commit recovery) is
deferred to a successor EXQ once V3-EXQ-490 (MECH-269b) lands and the
combined substrate stack is tested end-to-end.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_493_mech295_liking_bridge_validation.py

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

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


def run_uc1_module_importable() -> dict:
    """UC1: module / dataclass importable without side effects."""
    try:
        from ree_core.regulators import (
            MECH295LikingBridge,
            MECH295LikingBridgeConfig,
            MECH295LikingBridgeOutput,
        )
        cfg = MECH295LikingBridgeConfig()
        bridge = MECH295LikingBridge(cfg)
        out = bridge.get_last_output()
        ok = (
            cfg.drive_to_liking_gain == 1.0
            and cfg.liking_to_approach_cue_gain == 0.5
            and out is None
        )
        return {"pass": bool(ok)}
    except Exception as exc:  # pragma: no cover - smoke
        return {"pass": False, "error": repr(exc)}


def run_uc2_master_off_no_op() -> dict:
    """UC2: master OFF -> agent.mech295_bridge is None."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
    )
    agent = REEAgent(cfg)
    result = {"bridge_is_none": agent.mech295_bridge is None}
    result["pass"] = bool(result["bridge_is_none"])
    return result


def run_uc3_write_side_fires() -> dict:
    """UC3: drive elevated + z_goal seeded -> write side fires."""
    torch.manual_seed(42)
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_mech295_liking_bridge=True,
        mech295_drive_to_liking_gain=1.0,
        mech295_liking_to_approach_cue_gain=0.5,
        mech295_min_z_goal_norm_to_fire=0.001,  # easier to clear in 30 ticks
        drive_weight=2.0,
    )
    cfg.goal.z_goal_enabled = True
    agent = REEAgent(cfg)
    env = CausalGridWorldV2()
    flat_obs, obs_dict = env.reset()
    agent.reset()
    n_ticks = 30
    for _ in range(n_ticks):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick", False)
            else torch.zeros(1, cfg.latent.world_dim)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        # Force drive elevation + benefit so z_goal seeds.
        agent.update_z_goal(benefit_exposure=0.4, drive_level=0.8)
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = torch.zeros(1, 4)
            action[0, 0] = 1.0
            agent._last_action = action
        flat_obs, harm_signal, done, info, obs_dict = env.step(action)
        if done:
            agent.reset()
            flat_obs, obs_dict = env.reset()
    diag = agent.mech295_bridge.get_diagnostics()
    final_goal_norm = (
        agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
    )
    result = {
        "n_write_fires": int(diag["n_write_fires"]),
        "n_cue_fires": int(diag["n_cue_fires"]),
        "final_goal_norm": float(final_goal_norm),
    }
    result["pass"] = result["n_write_fires"] > 0
    return result


def run_uc4_cue_side_negative_bias() -> dict:
    """UC4: cue side produces monotone-negative bias under elevated drive."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    cfg = MECH295LikingBridgeConfig(liking_to_approach_cue_gain=0.5)
    bridge = MECH295LikingBridge(cfg)
    prox = torch.tensor([0.1, 0.5, 0.9])
    bias = bridge.compute_approach_cue_score_bias(
        drive_level=0.6, candidate_proximities=prox
    )
    result = {
        "bias": bias.tolist(),
        "bias_max_abs": float(bias.abs().max().item()),
        "bias_at_low_prox": float(bias[0].item()),
        "bias_at_high_prox": float(bias[2].item()),
    }
    result["pass"] = bool(
        result["bias_max_abs"] > 0.0
        and result["bias_at_high_prox"] < result["bias_at_low_prox"]
        and torch.all(bias <= 0.0).item()
    )
    return result


def run_uc5_severed_bridge_collapse() -> dict:
    """UC5: cue gain=0 produces zero bias even at elevated drive (severed)."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    cfg = MECH295LikingBridgeConfig(
        drive_to_liking_gain=1.0,           # write side intact
        liking_to_approach_cue_gain=0.0,    # cue side severed
    )
    bridge = MECH295LikingBridge(cfg)
    prox = torch.tensor([0.1, 0.5, 0.9])
    bias = bridge.compute_approach_cue_score_bias(
        drive_level=0.9, candidate_proximities=prox
    )
    write = bridge.compute_anticipatory_liking_write(
        drive_level=0.9, z_goal_norm=0.5
    )
    result = {
        "bias_max_abs": float(bias.abs().max().item()),
        "write_value": float(write),
    }
    # Severed cue: bias is exactly zero. Write side STILL fires.
    result["pass"] = bool(
        result["bias_max_abs"] == 0.0
        and result["write_value"] > 0.0
    )
    return result


def run_uc6_mech094_simulation_gate() -> dict:
    """UC6: simulation_mode=True -> zero on both sides, counters do not advance."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    bridge = MECH295LikingBridge(MECH295LikingBridgeConfig())
    write = bridge.compute_anticipatory_liking_write(
        drive_level=0.9, z_goal_norm=0.5, simulation_mode=True,
    )
    bias = bridge.compute_approach_cue_score_bias(
        drive_level=0.9,
        candidate_proximities=torch.tensor([0.5, 0.9]),
        simulation_mode=True,
    )
    diag = bridge.get_diagnostics()
    result = {
        "write_under_simulation": float(write),
        "bias_max_abs_under_simulation": float(bias.abs().max().item()),
        "n_write_fires": int(diag["n_write_fires"]),
        "n_cue_fires": int(diag["n_cue_fires"]),
    }
    result["pass"] = bool(
        result["write_under_simulation"] == 0.0
        and result["bias_max_abs_under_simulation"] == 0.0
        and result["n_write_fires"] == 0
        and result["n_cue_fires"] == 0
    )
    return result


def main(dry_run: bool = False) -> None:
    t0 = time.time()
    uc1 = run_uc1_module_importable()
    uc2 = run_uc2_master_off_no_op()
    uc3 = run_uc3_write_side_fires()
    uc4 = run_uc4_cue_side_negative_bias()
    uc5 = run_uc5_severed_bridge_collapse()
    uc6 = run_uc6_mech094_simulation_gate()

    metrics = {
        "UC1_module_importable": uc1,
        "UC2_master_off_no_op": uc2,
        "UC3_write_side_fires": uc3,
        "UC4_cue_side_negative_bias": uc4,
        "UC5_severed_bridge_collapse": uc5,
        "UC6_mech094_simulation_gate": uc6,
    }
    all_pass = all(m["pass"] for m in metrics.values())
    elapsed = time.time() - t0

    experiment_type = "v3_exq_493_mech295_liking_bridge_validation"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{experiment_type}_{ts}_v3",
        "experiment_type": experiment_type,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-295", "SD-012", "SD-014", "SD-015", "MECH-117"],
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-295": "supports" if all_pass else "weakens",
            "SD-012":   "non_contributory",
            "SD-014":   "non_contributory",
            "SD-015":   "non_contributory",
            "MECH-117": "non_contributory",
        },
        "result": "PASS" if all_pass else "FAIL",
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-295 weak-reading bridge validation diagnostic. Tests "
            "substrate wiring (UC1-UC3) and the falsifiable cue-side "
            "necessity signature (UC4 cue active produces approach pull; "
            "UC5 cue severed produces zero bias even with write side "
            "intact). UC6 confirms the MECH-094 simulation gate is honoured. "
            "MECH-295 listed as primary claim; SD-012 / SD-014 / SD-015 / "
            "MECH-117 are tagged as architectural prerequisites only "
            "(non_contributory direction -- this experiment does not test "
            "their independent claims). Behavioural EXQ-483-style "
            "approach_commit recovery is deferred to a successor experiment "
            "after V3-EXQ-490 (MECH-269b) and this bridge land together."
        ),
    }

    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=None,
            script_path=Path(__file__),
        )
        print(f"result: {manifest['result']}")
        for k, v in metrics.items():
            print(f"  {k}: pass={v['pass']}")
        print(f"Result written to: {out_path}", flush=True)
    else:
        print(f"DRY RUN result: {manifest['result']}")
        for k, v in metrics.items():
            print(f"  {k}: pass={v['pass']}")
        print("DRY RUN complete (no manifest written).", flush=True)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
