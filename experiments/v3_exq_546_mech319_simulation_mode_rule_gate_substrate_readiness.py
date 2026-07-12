"""V3-EXQ-546: MECH-319 simulation_mode_rule_write_gate substrate-readiness diagnostic.

Purpose: diagnostic. Confirms the MECH-319 substrate (categorical
simulation-mode rule-write gate; substrate-level instantiation of MECH-094
at the rule-arbitration layer) was wired correctly. Does NOT test the
biological claim that the gate's normal-mode (admit_writes=False)
behaviour preserves arbitration robustness against simulation content,
or the falsifier prediction that admit_writes=True (V3-EXQ-543c)
produces monomodal-collapse re-emergence -- both require behavioural
validation in V3-EXQ-543c-successor experiments AFTER the
MECH-313 / MECH-314 / MECH-318 sibling substrates have landed.

Substrate-readiness verifies:
  - The module instantiates and the agent wires it under the master flag.
  - The master-OFF flag produces bit-identical
    agent.simulation_mode_rule_gate=None behaviour.
  - The truth-table semantics hold across all valid
    (master, admit_writes, caller_sim) combinations.
  - The select_action call sites (gated_policy + lateral_pfc) consult
    the gate at integration time when master is on.
  - MECH-094 invariance: the gate's behaviour matches pre-MECH-319
    simulation_mode argument semantics in master-OFF and master-ON-
    waking-only regimes (the only regime currently exercised in
    select_action).

Five sub-tests (deterministic arithmetic + API integration):

  UC1 forward-pass instantiation: SimulationModeRuleGate constructs
      under the master flag; agent.simulation_mode_rule_gate is the
      module instance; get_state() returns the canonical diagnostic
      keys with the expected initial counter values (all zero).

  UC2 master-OFF backward-compat: with use_simulation_mode_rule_gate=
      False, agent.simulation_mode_rule_gate is None and a one-tick
      sense() + select_action runs without error.

  UC3 truth-table coverage: across the six valid combinations of
      (master, admit_writes, caller_sim), the gate returns the
      expected output and the per-cell counter increments. The
      master-OFF + admit_writes=True combination raises ValueError
      (loud-not-silent guard) -- tested separately in UC3b.

  UC4 select_action wiring contract: with master ON + waking caller
      (the regime select_action exercises), the gate's diagnostic
      counters reflect at least one waking call from each of the two
      wired sites (gated_policy + lateral_pfc) after one
      act_with_split_obs tick. n_simulation_blocked and
      n_simulation_admitted remain zero (no replay caller in the
      waking action selection path).

  UC5 MECH-094 invariance: paired runs at master OFF vs master
      ON-with-waking-caller produce identical behaviour at the
      gate-output level (output False at every wired call site).
      The asymmetry surfaces only when caller_sim=True -- which
      V3-EXQ-543c-successor experiments will introduce. This sub-
      test confirms the bit-identical OFF guarantee at the wiring-
      contract level.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_546_mech319_simulation_mode_rule_gate_substrate_readiness.py

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
from ree_core.regulators import (
    SimulationModeRuleGate,
    SimulationModeRuleGateConfig,
    SITE_GATED_POLICY,
    SITE_LATERAL_PFC,
)
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


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


def _build_agent_one_tick(
    use_simulation_mode_rule_gate: bool,
    admit_writes: bool = False,
    seed: int = 7,
    **extra_flags,
):
    """Helper: build REEAgent + run one sense() + one select_action tick.

    The wired call sites (gated_policy + lateral_pfc) only fire under
    use_gated_policy=True / use_lateral_pfc_analog=True. Default extras
    enable both so UC4 can exercise the integration contract.
    """
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1,
        use_proxy_fields=True,
    )
    flags = {
        "use_simulation_mode_rule_gate": use_simulation_mode_rule_gate,
        "simulation_mode_rule_gate_admit_writes": admit_writes,
        # Enable both wired arbitration consumers so the gate has
        # something to be consulted by at select_action time.
        "use_gated_policy": True,
        "use_lateral_pfc_analog": True,
    }
    flags.update(extra_flags)
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
    sa_error = None
    with torch.no_grad():
        latent = agent.sense(**kwargs)
        try:
            obs_body = kwargs["obs_body"]
            obs_world = kwargs["obs_world"]
            _action = agent.act_with_split_obs(
                obs_body, obs_world, temperature=1.0,
            )
        except Exception as e:  # noqa: BLE001
            sa_error = str(e)
    return agent, latent, sa_error


# ----------------------------------------------------------------------
# UC1
# ----------------------------------------------------------------------
def run_uc1_forward_pass_instantiation() -> dict:
    """UC1: SimulationModeRuleGate instantiates; get_state has canonical keys."""
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(use_simulation_mode_rule_gate=True)
    )
    state = gate.get_state()
    expected_keys = {
        "use_simulation_mode_rule_gate",
        "admit_writes",
        "n_calls_total",
        "n_waking_admitted",
        "n_simulation_blocked",
        "n_simulation_admitted",
        "per_site_calls",
        "per_site_waking_admitted",
        "per_site_simulation_blocked",
        "per_site_simulation_admitted",
    }
    keys_ok = expected_keys.issubset(set(state.keys()))
    counters_zero = all(
        state[k] == 0 for k in (
            "n_calls_total", "n_waking_admitted",
            "n_simulation_blocked", "n_simulation_admitted",
        )
    )
    per_site_dicts_empty = all(
        state[k] == {} for k in (
            "per_site_calls", "per_site_waking_admitted",
            "per_site_simulation_blocked", "per_site_simulation_admitted",
        )
    )
    result = {
        "is_module": isinstance(gate, SimulationModeRuleGate),
        "state_keys_ok": keys_ok,
        "initial_counters_zero": counters_zero,
        "initial_per_site_dicts_empty": per_site_dicts_empty,
        "config_master_on": state["use_simulation_mode_rule_gate"] is True,
        "config_admit_writes_off": state["admit_writes"] is False,
    }
    result["pass"] = all([
        result["is_module"], result["state_keys_ok"],
        result["initial_counters_zero"],
        result["initial_per_site_dicts_empty"],
        result["config_master_on"], result["config_admit_writes_off"],
    ])
    return result


# ----------------------------------------------------------------------
# UC2 master-OFF backward-compat
# ----------------------------------------------------------------------
def run_uc2_master_off_no_op() -> dict:
    """UC2: master-OFF preserves backward-compat (gate is None)."""
    agent_off, latent_off, err_off = _build_agent_one_tick(False)
    agent_on, latent_on, err_on = _build_agent_one_tick(True)

    off_is_none = (agent_off.simulation_mode_rule_gate is None)
    on_is_module = (agent_on.simulation_mode_rule_gate is not None)
    z_world_off_finite = bool(torch.isfinite(latent_off.z_world).all().item())
    z_world_on_finite = bool(torch.isfinite(latent_on.z_world).all().item())

    result = {
        "off_gate_is_none": off_is_none,
        "on_gate_is_module": on_is_module,
        "z_world_off_finite": z_world_off_finite,
        "z_world_on_finite": z_world_on_finite,
        "select_action_error_off": err_off,
        "select_action_error_on": err_on,
    }
    result["pass"] = (
        off_is_none and on_is_module
        and z_world_off_finite and z_world_on_finite
        and err_off is None and err_on is None
    )
    return result


# ----------------------------------------------------------------------
# UC3 truth-table coverage
# ----------------------------------------------------------------------
def run_uc3_truth_table_coverage() -> dict:
    """UC3: exhaustive sweep of (master, admit_writes, caller_sim) -> output."""
    cases = [
        # (master, admit_writes, caller_sim, expected_output, description)
        (False, False, False, False, "OFF_waking_passthrough"),
        (False, False, True,  True,  "OFF_simulation_passthrough"),
        # admit_writes=True with master OFF -> ValueError -- UC3b separate
        (True,  False, False, False, "ON_default_waking_admit"),
        (True,  False, True,  True,  "ON_default_simulation_block"),
        (True,  True,  False, False, "ON_falsifier_waking_admit"),
        (True,  True,  True,  False, "ON_falsifier_simulation_admit"),
    ]
    per_case = []
    all_ok = True
    for master, admit, caller, expected, desc in cases:
        gate = SimulationModeRuleGate(
            SimulationModeRuleGateConfig(
                use_simulation_mode_rule_gate=master,
                admit_writes=admit,
            )
        )
        actual = gate.effective_simulation_mode(simulation_mode=caller, site="t")
        ok = (actual is expected)
        if not ok:
            all_ok = False
        per_case.append({
            "master": master,
            "admit_writes": admit,
            "caller_sim": caller,
            "expected": expected,
            "actual": actual,
            "description": desc,
            "ok": ok,
        })
    return {"per_case": per_case, "n_cases": len(cases), "pass": all_ok}


def run_uc3b_precondition_raises() -> dict:
    """UC3b: admit_writes=True without master ON raises ValueError."""
    raised = False
    msg = None
    try:
        SimulationModeRuleGate(
            SimulationModeRuleGateConfig(
                use_simulation_mode_rule_gate=False,
                admit_writes=True,
            )
        )
    except ValueError as e:
        raised = True
        msg = str(e)
    return {
        "raised_value_error": raised,
        "error_message": msg,
        "pass": raised and msg is not None and "MECH-319" in msg,
    }


# ----------------------------------------------------------------------
# UC4 select_action wiring contract
# ----------------------------------------------------------------------
def run_uc4_select_action_wiring_contract() -> dict:
    """UC4: gate sees waking calls from both gated_policy and lateral_pfc sites."""
    agent, _, sa_error = _build_agent_one_tick(True)
    state = agent.simulation_mode_rule_gate.get_state()

    n_total = state["n_calls_total"]
    n_waking = state["n_waking_admitted"]
    n_sim_block = state["n_simulation_blocked"]
    n_sim_admit = state["n_simulation_admitted"]
    per_site = state["per_site_calls"]
    gp_calls = per_site.get(SITE_GATED_POLICY, 0)
    lp_calls = per_site.get(SITE_LATERAL_PFC, 0)

    result = {
        "select_action_error": sa_error,
        "n_calls_total": n_total,
        "n_waking_admitted": n_waking,
        "n_simulation_blocked": n_sim_block,
        "n_simulation_admitted": n_sim_admit,
        "gated_policy_calls": gp_calls,
        "lateral_pfc_calls": lp_calls,
        "per_site_calls": per_site,
    }
    # Wiring contract: at least one call from each wired site after one
    # act_with_split_obs tick. n_simulation_* must remain zero (waking
    # path only).
    result["pass"] = (
        sa_error is None
        and n_total >= 2
        and gp_calls >= 1
        and lp_calls >= 1
        and n_sim_block == 0
        and n_sim_admit == 0
        and n_waking == n_total
    )
    return result


# ----------------------------------------------------------------------
# UC5 MECH-094 invariance
# ----------------------------------------------------------------------
def run_uc5_mech094_invariance() -> dict:
    """UC5: master OFF vs ON-waking-only produce identical wiring behaviour.

    Both regimes pass simulation_mode=False to gated_policy.forward and skip
    no lateral_pfc.update calls. The gate's diagnostic counters in master-ON
    confirm zero simulation-mode firings on the waking path.
    """
    # Master ON, default admit_writes=False, waking caller
    gate_normal = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(use_simulation_mode_rule_gate=True)
    )
    out_normal_gp = gate_normal.effective_simulation_mode(False, site=SITE_GATED_POLICY)
    out_normal_lp = gate_normal.effective_simulation_mode(False, site=SITE_LATERAL_PFC)

    # Master ON, admit_writes=True (falsifier), waking caller
    gate_falsifier = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(
            use_simulation_mode_rule_gate=True,
            admit_writes=True,
        )
    )
    out_falsifier_gp = gate_falsifier.effective_simulation_mode(False, site=SITE_GATED_POLICY)
    out_falsifier_lp = gate_falsifier.effective_simulation_mode(False, site=SITE_LATERAL_PFC)

    # Master OFF, identity passthrough
    gate_off = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(use_simulation_mode_rule_gate=False)
    )
    out_off_gp = gate_off.effective_simulation_mode(False, site=SITE_GATED_POLICY)
    out_off_lp = gate_off.effective_simulation_mode(False, site=SITE_LATERAL_PFC)

    waking_invariance = (
        out_normal_gp is False and out_normal_lp is False
        and out_falsifier_gp is False and out_falsifier_lp is False
        and out_off_gp is False and out_off_lp is False
    )

    # Asymmetry surfaces ONLY when caller_sim=True. Quick confirmation
    # so the test demonstrates the gate IS active (not just always
    # returning False) -- normal blocks, falsifier admits.
    sim_normal = gate_normal.effective_simulation_mode(True, site="probe")
    sim_falsifier = gate_falsifier.effective_simulation_mode(True, site="probe")
    asymmetry_observed = (sim_normal is True and sim_falsifier is False)

    result = {
        "waking_outputs_invariant_across_modes": waking_invariance,
        "out_normal_gp": out_normal_gp,
        "out_normal_lp": out_normal_lp,
        "out_falsifier_gp": out_falsifier_gp,
        "out_falsifier_lp": out_falsifier_lp,
        "out_off_gp": out_off_gp,
        "out_off_lp": out_off_lp,
        "sim_normal_blocked": sim_normal,
        "sim_falsifier_admitted_as_false": sim_falsifier,
        "asymmetry_observed_at_simulation_mode_true": asymmetry_observed,
    }
    result["pass"] = waking_invariance and asymmetry_observed
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_forward_pass_instantiation()
    uc2 = run_uc2_master_off_no_op()
    uc3 = run_uc3_truth_table_coverage()
    uc3b = run_uc3b_precondition_raises()
    uc4 = run_uc4_select_action_wiring_contract()
    uc5 = run_uc5_mech094_invariance()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc3b, uc4, uc5])
    elapsed = time.time() - t0

    run_id = "v3_exq_546_mech319_simulation_mode_rule_gate_substrate_readiness_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_546_mech319_simulation_mode_rule_gate_substrate_readiness",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-319"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-319": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_forward_pass_instantiation": uc1,
            "UC2_master_off_no_op": uc2,
            "UC3_truth_table_coverage": uc3,
            "UC3b_precondition_raises": uc3b,
            "UC4_select_action_wiring_contract": uc4,
            "UC5_mech094_invariance": uc5,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-319 simulation_mode_rule_write_gate substrate-readiness "
            "diagnostic. Tests substrate wiring + truth-table semantics + "
            "MECH-094 invariance + falsifier-flag behaviour at the API "
            "level. Does NOT test the V3-EXQ-543c falsifier prediction "
            "(monomodal-collapse re-emergence under admit_writes=True with "
            "a replay-driven invocation path) -- that experiment is "
            "downstream of this substrate AND the MECH-313 / MECH-314 / "
            "MECH-318 sibling substrates. arc_062_rule_apprehension_plan.md "
            "GAP-K. Pull 3 SYNTHESIS R1 GENUINE-NOVELTY-CONFIRMED + Pull 4 "
            "R3 KEEP-AS-IS verdicts are architectural commitments, not "
            "signals tested here."
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

    print(f"verdict: {manifest['result']}")
    for k, v in manifest["metrics"].items():
        print(f"  {k}: pass={v['pass']}")
    print(f"Result written to: {out_path}", flush=True)

    from experiment_protocol import emit_outcome
    emit_outcome(outcome=manifest["result"], manifest_path=str(out_path))


if __name__ == "__main__":
    main()
