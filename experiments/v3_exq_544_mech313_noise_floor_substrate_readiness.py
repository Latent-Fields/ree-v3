"""V3-EXQ-544: MECH-313 (ARC-065) stochastic_noise_floor substrate-readiness diagnostic.

Purpose: diagnostic. Confirms the MECH-313 substrate (LC-NE tonic / SAC
max-entropy analog) was wired correctly. Does NOT test the biological
claim that the noise floor IS load-bearing for behavioural diversity --
that is Q-045's job (4-arm ablation MECH-313 OFF / 313 only / 260 only /
both ON), queued AFTER this substrate lands.

Substrate-readiness verifies:
  - The module instantiates and the agent wires it under the master flag.
  - The master-OFF flag produces bit-identical agent.noise_floor=None
    behaviour (backward compatibility).
  - effective_temperature lift arithmetic is correct under default and
    custom config (alpha + min_temperature interaction).
  - The select_action call site passes the lifted temperature to E3 (the
    integration contract).
  - MECH-094 simulation gate returns the baseline unchanged and does not
    advance waking-call counters.

Five sub-tests (deterministic arithmetic + API integration):

  UC1 forward-pass instantiation: REEAgent with use_noise_floor=True
      instantiates agent.noise_floor; the regulator's
      compute_effective_temperature returns a finite scalar > baseline
      under default config.

  UC2 master-OFF backward-compat: with use_noise_floor=False,
      agent.noise_floor is None and a one-tick sense() runs without
      error.

  UC3 effective_temperature lift correctness: across a sweep of
      baseline temperatures and (alpha, min_temperature) settings, the
      regulator returns max(baseline + alpha, min_temperature) exactly.

  UC4 select_action wiring contract: with use_noise_floor=True,
      noise_floor.compute_effective_temperature is invoked at
      select_action time (n_waking_calls > 0 after one tick) and the
      diagnostic snapshot reflects the lifted value.

  UC5 MECH-094 simulation gate: simulation_mode=True returns the
      baseline temperature unchanged, increments only the simulation-
      skip counter, and a subsequent waking call advances the waking-
      call counter without retroactively re-incrementing skip.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_544_mech313_noise_floor_substrate_readiness.py

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
from ree_core.policy import NoiseFloor, NoiseFloorConfig
from ree_core.utils.config import REEConfig


def _build_obs_kwargs(env, obs_dict, cfg):
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    kwargs = {"obs_body": body, "obs_world": world}
    obs_harm = obs_dict.get("harm_obs")
    if obs_harm is not None and getattr(cfg.latent, "use_harm_stream", False):
        if obs_harm.dim() == 1:
            obs_harm = obs_harm.unsqueeze(0)
        kwargs["obs_harm"] = obs_harm
    obs_harm_a = obs_dict.get("harm_obs_a")
    if obs_harm_a is not None and getattr(cfg.latent, "use_affective_harm_stream", False):
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
        kwargs["obs_harm_a"] = obs_harm_a
    return kwargs


def _build_agent_one_tick(use_noise_floor: bool, seed: int = 7, **extra_flags):
    """Helper: build REEAgent + run one sense() + one select_action tick."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1,
        use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16, world_dim=16,
        use_noise_floor=use_noise_floor,
        **extra_flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    kwargs = _build_obs_kwargs(env, obs_dict, cfg)
    with torch.no_grad():
        latent = agent.sense(**kwargs)
        # Exercise the full V3 act loop so select_action runs end-to-end
        # (generate_trajectories -> select_action -> e3.select with the
        # noise-floored temperature). act_with_split_obs is the canonical
        # entry point for split-obs experiments.
        try:
            obs_body = kwargs["obs_body"]
            obs_world = kwargs["obs_world"]
            _action = agent.act_with_split_obs(
                obs_body, obs_world, temperature=1.0,
            )
        except Exception as e:  # noqa: BLE001
            # Diagnostic only -- UC4 below catches this path.
            agent._select_action_error = str(e)  # type: ignore[attr-defined]
    return agent, latent


# ----------------------------------------------------------------------
# UC1
# ----------------------------------------------------------------------
def run_uc1_forward_pass_instantiation() -> dict:
    """UC1: NoiseFloor instantiates; compute_effective_temperature returns lifted scalar."""
    nf = NoiseFloor(NoiseFloorConfig(use_noise_floor=True))
    eff = nf.compute_effective_temperature(1.0, simulation_mode=False)
    state = nf.get_state()
    result = {
        "noise_floor_is_module": isinstance(nf, NoiseFloor),
        "effective_temperature": float(eff),
        "lifted_above_baseline": eff > 1.0,
        "finite": (eff == eff and eff != float("inf") and eff != float("-inf")),
        "state_keys": sorted(state.keys()),
        "n_waking_calls_after_one_call": state["n_waking_calls"],
    }
    result["pass"] = (
        result["noise_floor_is_module"]
        and result["lifted_above_baseline"]
        and result["finite"]
        and result["n_waking_calls_after_one_call"] == 1
    )
    return result


# ----------------------------------------------------------------------
# UC2 master-OFF backward-compat
# ----------------------------------------------------------------------
def run_uc2_master_off_no_op() -> dict:
    """UC2: master-OFF preserves backward-compat (noise_floor is None)."""
    agent_off, latent_off = _build_agent_one_tick(False)
    agent_on, latent_on = _build_agent_one_tick(True)

    off_is_none = (agent_off.noise_floor is None)
    on_is_module = (agent_on.noise_floor is not None)
    z_world_off_finite = bool(torch.isfinite(latent_off.z_world).all().item())
    z_world_on_finite = bool(torch.isfinite(latent_on.z_world).all().item())

    result = {
        "off_noise_floor_is_none": off_is_none,
        "on_noise_floor_is_module": on_is_module,
        "z_world_off_finite": z_world_off_finite,
        "z_world_on_finite": z_world_on_finite,
    }
    result["pass"] = (
        off_is_none
        and on_is_module
        and z_world_off_finite
        and z_world_on_finite
    )
    return result


# ----------------------------------------------------------------------
# UC3 effective_temperature lift correctness
# ----------------------------------------------------------------------
def run_uc3_effective_temperature_lift_correctness() -> dict:
    """UC3: max(baseline + alpha, min_temperature) arithmetic exact on a sweep."""
    cases = [
        # (alpha, min_T, baseline, expected_effective)
        (0.1, 1.0, 1.0, 1.1),     # alpha lift only
        (0.1, 1.0, 0.5, 1.0),     # min_temperature binds
        (0.5, 2.0, 1.0, 2.0),     # min_temperature binds (custom)
        (0.5, 2.0, 3.0, 3.5),     # alpha lift dominates
        (0.0, 1.5, 1.0, 1.5),     # alpha=0; floor binds
        (0.0, 0.5, 1.0, 1.0),     # alpha=0; floor below; baseline through
        (1.0, 1.0, 0.001, 1.001), # tiny baseline + alpha clears floor
    ]
    per_case = []
    all_ok = True
    for alpha, min_T, baseline, expected in cases:
        nf = NoiseFloor(NoiseFloorConfig(
            use_noise_floor=True,
            noise_floor_alpha=alpha,
            noise_floor_min_temperature=min_T,
        ))
        actual = nf.compute_effective_temperature(baseline)
        ok = abs(actual - expected) < 1e-9
        if not ok:
            all_ok = False
        per_case.append({
            "alpha": alpha,
            "min_T": min_T,
            "baseline": baseline,
            "expected": expected,
            "actual": float(actual),
            "ok": ok,
        })
    return {"per_case": per_case, "n_cases": len(cases), "pass": all_ok}


# ----------------------------------------------------------------------
# UC4 select_action wiring contract
# ----------------------------------------------------------------------
def run_uc4_select_action_wiring_contract() -> dict:
    """UC4: select_action invokes noise_floor; n_waking_calls advances after one tick."""
    agent, _ = _build_agent_one_tick(True)
    nf_state = agent.noise_floor.get_state()
    select_action_error = getattr(agent, "_select_action_error", None)

    # The wiring contract: at least one waking call recorded by the
    # noise_floor diagnostic snapshot AFTER the select_action tick AND
    # last_baseline_temperature set to the temperature passed in (1.0).
    # The lifted temperature must clear the floor (default min_T=1.0)
    # and reflect alpha=0.1 (default).
    n_waking = nf_state["n_waking_calls"]
    last_baseline = nf_state["last_baseline_temperature"]
    last_effective = nf_state["last_effective_temperature"]
    result = {
        "select_action_error": select_action_error,
        "n_waking_calls": n_waking,
        "last_baseline_temperature": last_baseline,
        "last_effective_temperature": last_effective,
        "expected_lift": 1.0 + 0.1,  # default alpha
    }
    result["pass"] = (
        select_action_error is None
        and n_waking >= 1
        and abs(last_baseline - 1.0) < 1e-9
        and abs(last_effective - 1.1) < 1e-9
    )
    return result


# ----------------------------------------------------------------------
# UC5 MECH-094 simulation gate
# ----------------------------------------------------------------------
def run_uc5_mech094_simulation_gate() -> dict:
    """UC5: simulation_mode=True returns baseline unchanged + increments skip only."""
    nf = NoiseFloor(NoiseFloorConfig(
        use_noise_floor=True,
        noise_floor_alpha=0.5,
        noise_floor_min_temperature=2.0,
    ))
    pre_skip = nf._last_n_simulation_skips
    pre_waking = nf._n_waking_calls

    sim_T = nf.compute_effective_temperature(1.0, simulation_mode=True)
    sim_skip_advanced = (nf._last_n_simulation_skips == pre_skip + 1)
    waking_unchanged = (nf._n_waking_calls == pre_waking)

    pre_skip_2 = nf._last_n_simulation_skips
    waking_T = nf.compute_effective_temperature(1.0, simulation_mode=False)
    waking_advanced = (nf._n_waking_calls == pre_waking + 1)
    skip_did_not_re_increment = (nf._last_n_simulation_skips == pre_skip_2)

    result = {
        "sim_returned_baseline": abs(sim_T - 1.0) < 1e-9,
        "sim_skip_advanced": sim_skip_advanced,
        "waking_counter_unchanged_in_sim": waking_unchanged,
        "waking_T_lifted": abs(waking_T - 2.0) < 1e-9,  # min_temperature=2.0 binds
        "waking_counter_advanced": waking_advanced,
        "skip_counter_not_re_incremented": skip_did_not_re_increment,
    }
    result["pass"] = (
        result["sim_returned_baseline"]
        and result["sim_skip_advanced"]
        and result["waking_counter_unchanged_in_sim"]
        and result["waking_T_lifted"]
        and result["waking_counter_advanced"]
        and result["skip_counter_not_re_incremented"]
    )
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_forward_pass_instantiation()
    uc2 = run_uc2_master_off_no_op()
    uc3 = run_uc3_effective_temperature_lift_correctness()
    uc4 = run_uc4_select_action_wiring_contract()
    uc5 = run_uc5_mech094_simulation_gate()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_544_mech313_noise_floor_substrate_readiness_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_544_mech313_noise_floor_substrate_readiness",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-313", "ARC-065"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-313": "supports" if all_pass else "weakens",
            "ARC-065": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_forward_pass_instantiation": uc1,
            "UC2_master_off_no_op": uc2,
            "UC3_effective_temperature_lift_correctness": uc3,
            "UC4_select_action_wiring_contract": uc4,
            "UC5_mech094_simulation_gate": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-313 (ARC-065) stochastic_noise_floor substrate-readiness "
            "diagnostic. Tests substrate wiring + Pull 1 SYNTHESIS R4 "
            "verdict (continuous, every tick) at the API level. Does NOT "
            "test the load-bearing claim of MECH-313 itself or its "
            "distinction from MECH-260 -- both require behavioural "
            "validation via Q-045 4-arm ablation (MECH-313 OFF / "
            "313 only / 260 only / both ON), queued AFTER substrate "
            "landing. Pull 1 SYNTHESIS verdicts (R1 BOTH-CHANNELS-NEEDED, "
            "R2 LC-NE tonic load-bearing, R4 continuous) are "
            "architectural commitments, not signals tested here."
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

    print(f"verdict: {manifest['result']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)

    from experiment_protocol import emit_outcome
    emit_outcome(outcome=manifest["result"], manifest_path=str(out_path))


if __name__ == "__main__":
    main()
