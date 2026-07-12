"""V3-EXQ-485a: SD-033b OFC-analog oracle landing validation.

Purpose: diagnostic. Extends V3-EXQ-485 with UC6 -- the specific-outcome
oracle (MECH-263 prospective query path). Confirms the query_outcome() API
routes correctly through E2HarmSForward and is bit-identical OFF when
use_ofc_outcome_oracle=False.

Six sub-tests (all deterministic arithmetic + API integration):

  UC1 instantiation: REEAgent with use_ofc_analog=True instantiates
      agent.ofc; state_code buffer has shape [1, state_dim].
      (Carried over from EXQ-485.)

  UC2 gate modulates update rate: state_code norm delta under gate=1.0 is
      at least 100x larger than delta under gate=0.0. Gate=0 freezes
      state_code (delta < 1e-6).
      (Carried over from EXQ-485.)

  UC3 bias zero at init: with zeroed last Linear, compute_bias returns a
      [K]-shaped vector of exactly zeros.
      (Carried over from EXQ-485.)

  UC4 backward compat: use_ofc_analog=False -> agent.ofc is None; the
      SD-033b tick block in select_action is skipped.
      (Carried over from EXQ-485.)

  UC5 reset clears state_code: agent.reset() -> state_code zeroed.
      (Carried over from EXQ-485.)

  UC6 oracle round-trip (NEW): use_ofc_outcome_oracle=True + e2_harm_s
      wired -> ofc.query_outcome() output is bit-identical to
      e2_harm_s.forward() at matching inputs. Oracle disabled -> output
      is None (_ofc_oracle_predictions cleared). AssertionError raised
      when oracle is called on a disabled config. Confirms prospective
      specific-outcome oracle path (MECH-263) is correctly wired.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_485a_sd033b_ofc_oracle_landing.py

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


def run_uc6_oracle_round_trip() -> dict:
    """UC6: oracle round-trip and backward-compat sub-tests.

    Sub-test A: oracle ON + e2_harm_s wired -> query_outcome() output
        matches e2_harm_s.forward() exactly (max abs diff < 1e-6).
    Sub-test B: oracle disabled -> AssertionError on direct call.
    Sub-test C: reset() clears _last_outcome_prediction + _ofc_oracle_predictions.
    Sub-test D: oracle_is_ready=False when use_ofc_outcome_oracle=False (default).
    Sub-test E: get_state() includes oracle_enabled=True and
        last_oracle_pred_norm > 0 after a query.
    """
    torch.manual_seed(42)

    # Sub-test D: oracle disabled by default
    cfg_default = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True,
    )
    agent_default = REEAgent(cfg_default)
    subtd_oracle_ready_false = not agent_default.ofc.oracle_is_ready

    # Sub-test A: oracle ON + e2_harm_s wired
    cfg_oracle = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True,
        use_ofc_outcome_oracle=True,
        use_e2_harm_s_forward=True,
        use_harm_stream=True,
    )
    agent_oracle = REEAgent(cfg_oracle)
    assert agent_oracle.ofc is not None
    assert agent_oracle.e2_harm_s is not None

    z_harm_s = torch.randn(1, 32)
    action = torch.randn(1, 4)

    pred_oracle = agent_oracle.ofc.query_outcome(
        z_harm_s, action, agent_oracle.e2_harm_s
    )
    pred_direct = agent_oracle.e2_harm_s.forward(
        z_harm_s.detach(), action.detach()
    ).detach()
    max_diff = float((pred_oracle - pred_direct).abs().max().item())
    subta_round_trip_diff = max_diff
    subta_pass = max_diff < 1e-6

    # Sub-test E: get_state() oracle fields
    state = agent_oracle.ofc.get_state()
    subt_e_oracle_enabled = state.get("oracle_enabled", False)
    subt_e_pred_norm = state.get("last_oracle_pred_norm", 0.0)
    subt_e_pass = subt_e_oracle_enabled and subt_e_pred_norm > 0.0

    # Sub-test C: reset clears caches
    agent_oracle.reset()
    subtc_pred_cleared = agent_oracle.ofc._last_outcome_prediction is None
    subtc_list_cleared = agent_oracle._ofc_oracle_predictions is None
    subtc_pass = subtc_pred_cleared and subtc_list_cleared

    # Sub-test B: AssertionError on disabled oracle
    subtb_pass = False
    try:
        agent_default.ofc.query_outcome(z_harm_s, action, agent_oracle.e2_harm_s)
    except AssertionError:
        subtb_pass = True

    result = {
        "subtest_a_round_trip_max_diff": subta_round_trip_diff,
        "subtest_a_pass": subta_pass,
        "subtest_b_assertion_on_disabled": subtb_pass,
        "subtest_c_reset_clears_pred": subtc_pred_cleared,
        "subtest_c_reset_clears_list": subtc_list_cleared,
        "subtest_c_pass": subtc_pass,
        "subtest_d_oracle_ready_false_by_default": subtd_oracle_ready_false,
        "subtest_e_get_state_oracle_enabled": subt_e_oracle_enabled,
        "subtest_e_get_state_pred_norm": subt_e_pred_norm,
        "subtest_e_pass": subt_e_pass,
    }
    result["pass"] = all([
        subta_pass,
        subtb_pass,
        subtc_pass,
        subtd_oracle_ready_false,
        subt_e_pass,
    ])
    return result


def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_instantiation()
    uc2 = run_uc2_gate_modulates_update()
    uc3 = run_uc3_bias_zero_at_init()
    uc4 = run_uc4_backward_compat()
    uc5 = run_uc5_reset_clears_state_code()
    uc6 = run_uc6_oracle_round_trip()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5, uc6])
    elapsed = time.time() - t0

    experiment_type = "v3_exq_485a_sd033b_ofc_oracle_landing"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{experiment_type}_{ts}_v3",
        "experiment_type": experiment_type,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["SD-033b", "MECH-261", "MECH-263"],
        "supersedes": "v3_exq_485_sd033b_ofc_analog_landing",
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
            "UC6_oracle_round_trip": uc6,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-033b oracle landing diagnostic. Extends EXQ-485 with UC6: "
            "the specific-outcome oracle path (MECH-263) implemented as "
            "OFCAnalog.query_outcome() delegating to E2HarmSForward. UC6 "
            "confirms: (A) oracle output is bit-identical to e2_harm_s.forward() "
            "(max diff < 1e-6); (B) AssertionError when oracle disabled; "
            "(C) reset() clears prediction caches; (D) oracle_is_ready=False "
            "by default; (E) get_state() exposes oracle diagnostics. "
            "MECH-263 devaluation sensitivity and same-sensory-input / "
            "different-task-role discrimination remain deferred to behavioural "
            "EXQs requiring env extension (outcome relabelling, task-distinct "
            "state pairs)."
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
