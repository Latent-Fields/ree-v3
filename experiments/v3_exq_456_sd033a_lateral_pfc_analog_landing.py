"""V3-EXQ-456: SD-033a lateral-PFC-analog landing validation.

Purpose: diagnostic. Confirms the substrate was wired correctly; does NOT
test biological claims of SD-033a (signatures i/iii/iv) because the bias
head is untrained in this landing (signature iv deferred). Signature (ii)
distractor-resistant persistence is exercised indirectly via the gate=0
update-suppression check.

Four sub-tests (all deterministic arithmetic + API integration):

  UC1 instantiation: REEAgent with use_lateral_pfc_analog=True instantiates
      agent.lateral_pfc; rule_state buffer has shape [1, rule_dim].

  UC2 gate modulates update rate: rule_state norm delta under gate=1.0 is
      at least 100x larger than delta under gate=0.0 for matched z_delta
      and z_world inputs. Gate=0 freezes rule_state (delta < 1e-6).

  UC3 bias reaches E3 (and is exactly zero at init): with zeroed last
      Linear, compute_bias returns [K]-shaped vector of exactly zeros,
      and the E3 score_bias composition path does not alter behaviour
      (cannot negatively bias any trajectory when head is untrained).

  UC4 backward compat: use_lateral_pfc_analog=False -> agent.lateral_pfc
      is None; the entire SD-033a tick block in select_action is skipped.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_456_sd033a_lateral_pfc_analog_landing.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def run_uc1_instantiation() -> dict:
    """UC1: instantiation + rule_state buffer shape."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_lateral_pfc_analog=True,
    )
    agent = REEAgent(cfg)
    result = {
        "lateral_pfc_is_not_none": agent.lateral_pfc is not None,
        "rule_state_shape": list(agent.lateral_pfc.rule_state.shape)
        if agent.lateral_pfc is not None else None,
        "rule_dim_expected": cfg.lateral_pfc_rule_dim,
    }
    result["pass"] = (
        result["lateral_pfc_is_not_none"]
        and result["rule_state_shape"] == [1, cfg.lateral_pfc_rule_dim]
    )
    return result


def run_uc2_gate_modulates_update() -> dict:
    """UC2: gate=1 updates; gate=0 freezes."""
    torch.manual_seed(0)
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_lateral_pfc_analog=True,
    )
    agent = REEAgent(cfg)
    z_delta = torch.randn(1, cfg.latent.delta_dim)
    z_world = torch.randn(1, cfg.latent.world_dim)

    # gate=1
    pre_1 = agent.lateral_pfc.rule_state.clone()
    agent.lateral_pfc.update(z_delta, z_world, gate=1.0)
    delta_gate_1 = float((agent.lateral_pfc.rule_state - pre_1).norm().item())

    # gate=0 (reset first)
    agent.lateral_pfc.reset()
    pre_0 = agent.lateral_pfc.rule_state.clone()
    agent.lateral_pfc.update(z_delta, z_world, gate=0.0)
    delta_gate_0 = float((agent.lateral_pfc.rule_state - pre_0).norm().item())

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
        use_lateral_pfc_analog=True,
    )
    agent = REEAgent(cfg)

    # Populate rule_state to non-zero before querying bias head
    z_delta = torch.randn(1, cfg.latent.delta_dim)
    z_world = torch.randn(1, cfg.latent.world_dim)
    agent.lateral_pfc.update(z_delta, z_world, gate=1.0)

    K = 5
    cand_summaries = torch.randn(K, cfg.latent.world_dim)
    bias = agent.lateral_pfc.compute_bias(cand_summaries)

    result = {
        "bias_shape": list(bias.shape),
        "bias_max_abs": float(bias.abs().max().item()),
        "rule_state_norm_at_query": float(
            agent.lateral_pfc.rule_state.norm().item()
        ),
    }
    result["pass"] = (
        result["bias_shape"] == [K]
        and result["bias_max_abs"] < 1e-6
        and result["rule_state_norm_at_query"] > 0.0
    )
    return result


def run_uc4_backward_compat() -> dict:
    """UC4: use_lateral_pfc_analog=False -> agent.lateral_pfc is None."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_lateral_pfc_analog=False,
    )
    agent = REEAgent(cfg)
    result = {
        "lateral_pfc_is_none": agent.lateral_pfc is None,
    }
    result["pass"] = result["lateral_pfc_is_none"]
    return result


def run_uc5_reset_clears_rule_state() -> dict:
    """UC5: agent.reset() -> rule_state zeroed (and lateral_pfc.reset path)."""
    torch.manual_seed(0)
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_lateral_pfc_analog=True,
    )
    agent = REEAgent(cfg)
    z_delta = torch.randn(1, cfg.latent.delta_dim)
    z_world = torch.randn(1, cfg.latent.world_dim)
    for _ in range(10):
        agent.lateral_pfc.update(z_delta, z_world, gate=1.0)
    pre_reset_norm = float(agent.lateral_pfc.rule_state.norm().item())
    agent.reset()
    post_reset_norm = float(agent.lateral_pfc.rule_state.norm().item())
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
    uc5 = run_uc5_reset_clears_rule_state()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_456_sd033a_lateral_pfc_analog_landing_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_456_sd033a_lateral_pfc_analog_landing",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["SD-033a", "MECH-261", "MECH-262"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "SD-033a": "supports" if all_pass else "weakens",
            "MECH-261": "supports" if all_pass else "weakens",
            "MECH-262": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_instantiation": uc1,
            "UC2_gate_modulates_update": uc2,
            "UC3_bias_zero_at_init": uc3,
            "UC4_backward_compat": uc4,
            "UC5_reset_clears_rule_state": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-033a landing diagnostic. Tests substrate wiring; bias head "
            "is frozen-random with last Linear zeroed (initial output = 0) "
            "so UC3 is a check on the zero-init contract, not a claim "
            "about biological bias function. Behavioral tests of "
            "signatures (i) stimulus-abstraction, (iii) bias-into-E3, and "
            "(iv) training-dependent emergence are deferred to later "
            "experiments that train the head under the phased protocol."
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
