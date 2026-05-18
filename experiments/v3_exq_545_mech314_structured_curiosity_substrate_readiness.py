"""V3-EXQ-545: MECH-314 (ARC-065) structured_curiosity_bonus substrate-readiness diagnostic.

Purpose: diagnostic. Confirms the MECH-314 substrate (frontopolar
exploration / EFE analog) was wired correctly with master + 3
independently-togglable sub-flavour switches. Does NOT test the
biological claim that any sub-flavour IS load-bearing for behavioural
diversity -- that is Q-044's job (three-arm ablation 314a-OFF / 314b-
OFF / 314c-OFF on V3-EXQ-543b/c successors), queued AFTER this
substrate lands AND the MECH-318 / MECH-319 absorption-check sessions
complete.

Substrate-readiness verifies:
  - The module instantiates and the agent wires it under the master
    flag.
  - The master-OFF flag produces bit-identical agent.curiosity=None
    behaviour (backward compatibility).
  - Each sub-flavour fires under flag-set isolation (314a-only,
    314b-only, 314c-only) -- this is the architectural prerequisite
    that makes Q-044's three-arm ablation a flag-set decision.
  - The select_action call site invokes compute_score_bias and the
    output lands additively in the dACC / lateral_pfc / ofc / mech295
    score_bias chain.
  - MECH-094 simulation gate returns zeros[K] and does not advance
    waking counters; update_prediction_error is a no-op on the LP
    buffer under simulation_mode.

Five sub-tests (deterministic):

  UC1 forward-pass instantiation: REEAgent with use_structured_curiosity
      =True instantiates agent.curiosity; the regulator's
      compute_score_bias returns a finite [K] tensor under default
      config.

  UC2 master-OFF backward-compat: with use_structured_curiosity=False,
      agent.curiosity is None and a one-tick sense() + act runs
      without error.

  UC3 sub-flavour flag-set isolation: master ON with each individual
      sub-flavour switch ON in turn (other two OFF). Each combination
      produces non-zero bias when its signal source is non-trivial,
      and the active sub-flavour count reads as 1 in each case. The
      all-three-off-master-on case produces zeros[K]. This is the
      "Q-044 three-arm ablation is a flag-set decision" contract.

  UC4 select_action wiring contract: with use_structured_curiosity=
      True, curiosity.compute_score_bias is invoked at select_action
      time (n_waking_calls > 0 after one tick) and the diagnostic
      snapshot reflects non-trivial signals from at least one sub-
      flavour.

  UC5 MECH-094 simulation gate: simulation_mode=True returns zeros[K],
      increments only the simulation-skip counter; subsequent waking
      call advances the waking-call counter without retroactively
      re-incrementing skip; update_prediction_error(simulation_mode=
      True) is a no-op on the LP buffer.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_545_mech314_structured_curiosity_substrate_readiness.py

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
from ree_core.policy import StructuredCuriosity, StructuredCuriosityConfig
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


def _build_agent_one_tick(seed: int = 7, **flags):
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
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    kwargs = _build_obs_kwargs(env, obs_dict, cfg)
    with torch.no_grad():
        latent = agent.sense(**kwargs)
        try:
            obs_body = kwargs["obs_body"]
            obs_world = kwargs["obs_world"]
            _action = agent.act_with_split_obs(
                obs_body, obs_world, temperature=1.0,
            )
        except Exception as e:  # noqa: BLE001
            agent._select_action_error = str(e)  # type: ignore[attr-defined]
    return agent, latent


# ----------------------------------------------------------------------
# UC1
# ----------------------------------------------------------------------
def run_uc1_forward_pass_instantiation() -> dict:
    """UC1: StructuredCuriosity instantiates; compute_score_bias finite."""
    cfg = StructuredCuriosityConfig(use_structured_curiosity=True)
    mod = StructuredCuriosity(cfg)
    K = 5
    summaries = torch.randn(K, 16)
    bias = mod.compute_score_bias(
        summaries, residue_field=None, e3=None, simulation_mode=False,
    )
    state = mod.get_state()
    finite = bool(torch.isfinite(bias).all().item())
    result = {
        "curiosity_is_module": isinstance(mod, StructuredCuriosity),
        "bias_shape": list(bias.shape),
        "bias_dtype": str(bias.dtype),
        "finite": finite,
        "n_waking_calls_after_one_call": state["n_waking_calls"],
        "subflavours_fired": state["last_n_subflavours_fired"],
    }
    result["pass"] = (
        result["curiosity_is_module"]
        and result["bias_shape"] == [K]
        and finite
        and result["n_waking_calls_after_one_call"] == 1
    )
    return result


# ----------------------------------------------------------------------
# UC2 master-OFF backward-compat
# ----------------------------------------------------------------------
def run_uc2_master_off_no_op() -> dict:
    """UC2: master-OFF preserves backward-compat (curiosity is None)."""
    agent_off, latent_off = _build_agent_one_tick()
    agent_on, latent_on = _build_agent_one_tick(use_structured_curiosity=True)

    off_is_none = (agent_off.curiosity is None)
    on_is_module = (agent_on.curiosity is not None)
    z_world_off_finite = bool(torch.isfinite(latent_off.z_world).all().item())
    z_world_on_finite = bool(torch.isfinite(latent_on.z_world).all().item())
    no_select_action_error_off = getattr(agent_off, "_select_action_error", None) is None
    no_select_action_error_on = getattr(agent_on, "_select_action_error", None) is None

    result = {
        "off_curiosity_is_none": off_is_none,
        "on_curiosity_is_module": on_is_module,
        "z_world_off_finite": z_world_off_finite,
        "z_world_on_finite": z_world_on_finite,
        "no_select_action_error_off": no_select_action_error_off,
        "no_select_action_error_on": no_select_action_error_on,
    }
    result["pass"] = all(result.values())
    return result


# ----------------------------------------------------------------------
# UC3 sub-flavour flag-set isolation (Q-044 three-arm flag-set contract)
# ----------------------------------------------------------------------
class _MockE3:
    def __init__(self, running_variance: float = 1.0):
        self._running_variance = running_variance


class _MockResidue:
    def __init__(self, world_dim: int = 16):
        class _RBF:
            pass
        rbf = _RBF()
        rbf.centers = torch.zeros(4, world_dim)
        rbf.active_mask = torch.tensor([True, False, False, False])
        self.rbf_field = rbf


def _build_module(**overrides):
    cfg = StructuredCuriosityConfig(use_structured_curiosity=True, **overrides)
    return StructuredCuriosity(cfg)


def _seed_lp(mod, k: int = 3):
    pe_seq = [0.5, 0.4, 0.3, 0.2]
    for pe in pe_seq[: k + 1]:
        mod.update_prediction_error(pe)


def run_uc3_subflavour_flag_set_isolation() -> dict:
    """UC3: each sub-flavour fires independently under flag-set isolation."""
    K = 4
    world_dim = 16
    summaries = torch.randn(K, world_dim) * 0.5
    e3 = _MockE3(running_variance=1.0)
    res = _MockResidue(world_dim=world_dim)

    arm_a = _build_module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
    )
    bias_a = arm_a.compute_score_bias(summaries, residue_field=res, e3=e3)
    a_fired = arm_a._last_n_subflavours_fired
    a_nonzero = bool((bias_a != 0).any().item())

    arm_b = _build_module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=False,
    )
    bias_b = arm_b.compute_score_bias(summaries, residue_field=res, e3=e3)
    b_fired = arm_b._last_n_subflavours_fired
    b_nonzero = bool((bias_b != 0).any().item())

    arm_c = _build_module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=True,
        curiosity_lp_window_k=3,
    )
    _seed_lp(arm_c, k=3)
    bias_c = arm_c.compute_score_bias(summaries, residue_field=res, e3=e3)
    c_fired = arm_c._last_n_subflavours_fired
    c_nonzero = bool((bias_c != 0).any().item())

    arm_off = _build_module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
    )
    bias_off = arm_off.compute_score_bias(summaries, residue_field=res, e3=e3)
    off_fired = arm_off._last_n_subflavours_fired
    off_zero = bool((bias_off == 0).all().item())

    result = {
        "arm_a_n_fired": a_fired,
        "arm_a_nonzero_bias": a_nonzero,
        "arm_b_n_fired": b_fired,
        "arm_b_nonzero_bias": b_nonzero,
        "arm_c_n_fired": c_fired,
        "arm_c_nonzero_bias": c_nonzero,
        "arm_off_n_fired": off_fired,
        "arm_off_zero_bias": off_zero,
    }
    result["pass"] = (
        a_fired == 1 and a_nonzero
        and b_fired == 1 and b_nonzero
        and c_fired == 1 and c_nonzero
        and off_fired == 0 and off_zero
    )
    return result


# ----------------------------------------------------------------------
# UC4 select_action wiring contract
# ----------------------------------------------------------------------
def run_uc4_select_action_wiring_contract() -> dict:
    """UC4: select_action invokes curiosity.compute_score_bias once per tick."""
    agent, _ = _build_agent_one_tick(use_structured_curiosity=True)
    state = agent.curiosity.get_state()
    select_action_error = getattr(agent, "_select_action_error", None)

    n_waking = state["n_waking_calls"]
    n_subflavours_fired = state["last_n_subflavours_fired"]
    bias_max_abs = state["last_bias_max_abs"]
    unc_signal = state["last_uncertainty_signal"]
    result = {
        "select_action_error": select_action_error,
        "n_waking_calls": n_waking,
        "n_subflavours_fired_at_first_tick": n_subflavours_fired,
        "bias_max_abs": bias_max_abs,
        "uncertainty_signal_nonzero": unc_signal != 0.0,
    }
    # Pass: at least one waking call recorded and at least one sub-flavour
    # fired at the first tick. The default config has all three sub-
    # flavours ON; 314a contributes 0 (no active residue centers on
    # tick 1) and 314c contributes 0 (LP not yet seeded), but 314b
    # always contributes from e3._running_variance.
    result["pass"] = (
        select_action_error is None
        and n_waking >= 1
        and n_subflavours_fired >= 1
        and bias_max_abs > 0.0
        and unc_signal != 0.0
    )
    return result


# ----------------------------------------------------------------------
# UC5 MECH-094 simulation gate
# ----------------------------------------------------------------------
def run_uc5_mech094_simulation_gate() -> dict:
    """UC5: simulation_mode=True returns zeros + skip-only; LP no-op."""
    K = 5
    world_dim = 16
    summaries = torch.randn(K, world_dim)
    mod = _build_module(curiosity_lp_window_k=3)
    e3 = _MockE3(running_variance=1.0)
    res = _MockResidue(world_dim=world_dim)

    # Pre-seed LP so 314c would fire in waking conditions.
    _seed_lp(mod, k=3)
    pre_lp = mod._lp_ema

    pre_skip = mod._last_n_simulation_skips
    pre_waking = mod._n_waking_calls

    sim_bias = mod.compute_score_bias(
        summaries, residue_field=res, e3=e3, simulation_mode=True,
    )
    sim_returned_zeros = bool((sim_bias == 0).all().item())
    sim_skip_advanced = (mod._last_n_simulation_skips == pre_skip + 1)
    waking_unchanged = (mod._n_waking_calls == pre_waking)

    # update_prediction_error in simulation mode: no LP advance.
    pre_ring = list(mod._pe_ring)
    mod.update_prediction_error(99.0, simulation_mode=True)
    lp_unchanged_sim = (mod._pe_ring == pre_ring) and (
        abs(mod._lp_ema - pre_lp) < 1e-12
    )

    pre_skip_2 = mod._last_n_simulation_skips
    waking_bias = mod.compute_score_bias(summaries, residue_field=res, e3=e3)
    waking_advanced = (mod._n_waking_calls == pre_waking + 1)
    skip_did_not_re_increment = (mod._last_n_simulation_skips == pre_skip_2)
    waking_nonzero = bool((waking_bias != 0).any().item())

    result = {
        "sim_returned_zeros": sim_returned_zeros,
        "sim_skip_advanced": sim_skip_advanced,
        "waking_counter_unchanged_in_sim": waking_unchanged,
        "lp_unchanged_in_sim": lp_unchanged_sim,
        "waking_counter_advanced": waking_advanced,
        "skip_counter_not_re_incremented": skip_did_not_re_increment,
        "waking_bias_nonzero": waking_nonzero,
    }
    result["pass"] = all(result.values())
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_forward_pass_instantiation()
    uc2 = run_uc2_master_off_no_op()
    uc3 = run_uc3_subflavour_flag_set_isolation()
    uc4 = run_uc4_select_action_wiring_contract()
    uc5 = run_uc5_mech094_simulation_gate()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_545_mech314_structured_curiosity_substrate_readiness_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_545_mech314_structured_curiosity_substrate_readiness",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-314", "MECH-314a", "MECH-314b", "MECH-314c", "ARC-065"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-314": "supports" if all_pass else "weakens",
            "MECH-314a": "supports" if all_pass else "weakens",
            "MECH-314b": "supports" if all_pass else "weakens",
            "MECH-314c": "supports" if all_pass else "weakens",
            "ARC-065": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_forward_pass_instantiation": uc1,
            "UC2_master_off_no_op": uc2,
            "UC3_subflavour_flag_set_isolation": uc3,
            "UC4_select_action_wiring_contract": uc4,
            "UC5_mech094_simulation_gate": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-314 (ARC-065) structured_curiosity_bonus substrate-"
            "readiness diagnostic. Tests substrate wiring + master + 3 "
            "sub-flavour switches at the API level. UC3 is the Q-044 "
            "three-arm-ablation flag-set-decision contract. Does NOT "
            "test the load-bearing claim of MECH-314 / its sub-flavours "
            "or distinct failure signatures -- both require behavioural "
            "validation via Q-044 three-arm ablation on V3-EXQ-543b/c "
            "successors AFTER substrate landing AND the MECH-318 / "
            "MECH-319 absorption-check sessions complete. Pull 1 "
            "SYNTHESIS verdicts (R1 BOTH-CHANNELS-NEEDED, R3 sub-flavour "
            "split, R4 continuous in computation) are architectural "
            "commitments, not signals tested here. 314b and 314c are "
            "broadcast scalars in Phase 1; per-candidate refinement is "
            "a Phase 2 follow-on."
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
