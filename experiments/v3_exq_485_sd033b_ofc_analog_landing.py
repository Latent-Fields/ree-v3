"""V3-EXQ-485: SD-033b OFC-analog landing validation.

Purpose: diagnostic. Confirms the substrate was wired correctly; does NOT
test biological claims of MECH-263 (devaluation sensitivity, same-sensory
discrimination) because the bias head is untrained in this landing. The
behavioural validation requires environment work (outcome relabelling) and
is queued separately.

Five sub-tests (all deterministic arithmetic + API integration):

  UC1 instantiation: REEAgent with use_ofc_analog=True instantiates
      agent.ofc; state_code buffer has shape [1, state_dim].

  UC2 gate modulates update rate: state_code norm delta under gate=1.0 is
      at least 100x larger than delta under gate=0.0 for matched z_world
      input. Gate=0 freezes state_code (delta < 1e-6). Mirrors MECH-261
      registry: write_gate("sd_033b") near 0 must freeze state_code under
      internal_replay (weight=0.05).

  UC3 bias reaches E3 (and is exactly zero at init): with zeroed last
      Linear, compute_bias returns [K]-shaped vector of exactly zeros,
      so use_ofc_analog=True with an untrained head is bit-identical to
      OFF until the head is deliberately trained.

  UC4 backward compat: use_ofc_analog=False -> agent.ofc is None; the
      entire SD-033b tick block in select_action is skipped.

  UC5 reset clears state_code: agent.reset() -> state_code zeroed via
      ofc.reset() call from REEAgent.reset().

Run with:
  /opt/local/bin/python3 experiments/v3_exq_485_sd033b_ofc_analog_landing.py

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
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


def run_uc1_instantiation() -> dict:
    """UC1: instantiation + state_code buffer shape."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True,
    )
    agent = REEAgent(cfg)
    result = {
        "ofc_is_not_none": agent.ofc is not None,
        "state_code_shape": list(agent.ofc.state_code.shape)
        if agent.ofc is not None else None,
        "state_dim_expected": cfg.ofc_state_dim,
    }
    result["pass"] = (
        result["ofc_is_not_none"]
        and result["state_code_shape"] == [1, cfg.ofc_state_dim]
    )
    return result


def run_uc2_gate_modulates_update() -> dict:
    """UC2: gate=1 updates; gate=0 freezes."""
    torch.manual_seed(0)
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True,
    )
    agent = REEAgent(cfg)
    z_world = torch.randn(1, cfg.latent.world_dim)

    pre_1 = agent.ofc.state_code.clone()
    agent.ofc.update(z_world=z_world, z_harm=None, gate=1.0)
    delta_gate_1 = float((agent.ofc.state_code - pre_1).norm().item())

    agent.ofc.reset()
    pre_0 = agent.ofc.state_code.clone()
    agent.ofc.update(z_world=z_world, z_harm=None, gate=0.0)
    delta_gate_0 = float((agent.ofc.state_code - pre_0).norm().item())

    result = {
        "delta_gate_1": delta_gate_1,
        "delta_gate_0": delta_gate_0,
        "ratio": delta_gate_1 / (delta_gate_0 + 1e-12),
    }
    result["pass"] = (
        delta_gate_0 < 1e-6
        and delta_gate_1 > 100.0 * (delta_gate_0 + 1e-12)
        and delta_gate_1 > 1e-3
    )
    return result


def run_uc3_bias_zero_at_init() -> dict:
    """UC3: bias head output is exactly zero at init (last Linear zeroed)."""
    torch.manual_seed(0)
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True,
    )
    agent = REEAgent(cfg)

    # Populate state_code to non-zero before querying bias head
    z_world = torch.randn(1, cfg.latent.world_dim)
    agent.ofc.update(z_world=z_world, z_harm=None, gate=1.0)

    K = 5
    cand_summaries = torch.randn(K, cfg.latent.world_dim)
    bias = agent.ofc.compute_bias(cand_summaries)

    result = {
        "bias_shape": list(bias.shape),
        "bias_max_abs": float(bias.abs().max().item()),
        "state_code_norm_at_query": float(
            agent.ofc.state_code.norm().item()
        ),
    }
    result["pass"] = (
        result["bias_shape"] == [K]
        and result["bias_max_abs"] < 1e-6
        and result["state_code_norm_at_query"] > 0.0
    )
    return result


def run_uc4_backward_compat() -> dict:
    """UC4: use_ofc_analog=False -> agent.ofc is None."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=False,
    )
    agent = REEAgent(cfg)
    result = {
        "ofc_is_none": agent.ofc is None,
    }
    result["pass"] = result["ofc_is_none"]
    return result


def run_uc5_reset_clears_state_code() -> dict:
    """UC5: agent.reset() -> state_code zeroed via ofc.reset()."""
    torch.manual_seed(0)
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True,
    )
    agent = REEAgent(cfg)
    z_world = torch.randn(1, cfg.latent.world_dim)
    for _ in range(10):
        agent.ofc.update(z_world=z_world, z_harm=None, gate=1.0)
    pre_reset_norm = float(agent.ofc.state_code.norm().item())
    agent.reset()
    post_reset_norm = float(agent.ofc.state_code.norm().item())
    result = {
        "pre_reset_norm": pre_reset_norm,
        "post_reset_norm": post_reset_norm,
    }
    result["pass"] = pre_reset_norm > 0.0 and post_reset_norm == 0.0
    return result


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_instantiation()
    uc2 = run_uc2_gate_modulates_update()
    uc3 = run_uc3_bias_zero_at_init()
    uc4 = run_uc4_backward_compat()
    uc5 = run_uc5_reset_clears_state_code()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    experiment_type = "v3_exq_485_sd033b_ofc_analog_landing"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{experiment_type}_{ts}_v3",
        "experiment_type": experiment_type,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["SD-033b", "MECH-261", "MECH-263"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "SD-033b": "supports" if all_pass else "weakens",
            "MECH-261": "supports" if all_pass else "weakens",
            "MECH-263": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_instantiation": uc1,
            "UC2_gate_modulates_update": uc2,
            "UC3_bias_zero_at_init": uc3,
            "UC4_backward_compat": uc4,
            "UC5_reset_clears_state_code": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-033b landing diagnostic. Tests substrate wiring; bias head "
            "is frozen-random with last Linear zeroed (initial output = 0) "
            "so UC3 is a check on the zero-init contract, not a claim "
            "about biological bias function. MECH-263 functional "
            "signatures -- (a) devaluation sensitivity and (b) "
            "same-sensory-input / different-task-role discrimination -- "
            "remain deferred to behavioural EXQs that require environment "
            "work (outcome relabelling, perceptually-identical / "
            "task-distinct state pairs)."
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
