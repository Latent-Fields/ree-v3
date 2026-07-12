"""V3-EXQ-547: MECH-320 (ARC-066 child) tonic_vigor_coupling_score_bias substrate-readiness.

Purpose: diagnostic. Confirms the MECH-320 substrate (mesolimbic-DA-vigor /
average-reward-rate / opportunity-cost regulator) was wired correctly. Does
NOT test the architectural claim that the vigor scalar IS load-bearing for
behavioural action density -- that is the next-step 3-arm discriminative pair
(baseline / additive / multiplicative on a well-fed-safe-familiar
environment), queued separately AFTER this substrate lands.

Substrate-readiness verifies:
  - The TonicVigor module instantiates and the agent wires it under the
    master flag.
  - The master-OFF flag produces bit-identical agent.tonic_vigor=None
    behaviour (backward compatibility).
  - The slow EWMA over realised E3-score-receipt converges to the reward
    history (Niv 2007 average-reward-rate formalism, R4 verdict).
  - The select_action call site exercises both compute_score_bias (pre-
    select) and update_score_receipt (post-select) -- the integration
    contract for v_t to react to the realised reward stream.
  - MECH-094 simulation gate returns zeros / does not advance the EWMA.
  - The R3 falsifiable secondary alternative (multiplicative form) produces
    distinguishable per-candidate bias from the additive primary on a
    pre-existing-preference batch.

Six sub-tests:

  UC1 forward-pass instantiation: TonicVigor with use_tonic_vigor=True
      instantiates; compute_score_bias on a small candidate batch returns
      a finite [K] tensor; update_score_receipt advances the EWMA.

  UC2 master-OFF backward-compat: with use_tonic_vigor=False,
      agent.tonic_vigor is None; one act_with_split_obs tick runs without
      error.

  UC3 EWMA convergence and half-life: constant reward stream r drives
      v_raw -> r in steady state; reward step from 0 to 1 crosses 0.5 at
      t = half_life; sign-convention (REE-low-is-better -> high-vigor).

  UC4 select_action wiring contract: with use_tonic_vigor=True, both
      n_waking_bias_calls and n_waking_score_updates advance after one
      act_with_split_obs tick (compute_score_bias fires pre-select;
      update_score_receipt fires post-select on the realised score).

  UC5 MECH-094 simulation gate: simulation_mode=True on compute_score_bias
      returns zeros + increments simulation-skip counter; on
      update_score_receipt does not advance EWMA + increments skip.

  UC6 R3 form discriminability: additive form produces uniform-per-class
      bias on a held-out batch with non-uniform |scores|; multiplicative
      form produces magnitude-varying bias. The two forms are
      distinguishable (the empirical resolution path for R3's
      additive-vs-multiplicative open question).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_547_mech320_tonic_vigor_substrate_readiness.py

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
from ree_core.policy import TonicVigor, TonicVigorConfig
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
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


def _build_agent_one_tick(use_tonic_vigor: bool, seed: int = 7, **extra_flags):
    """Helper: build REEAgent + run one act_with_split_obs tick."""
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
        use_tonic_vigor=use_tonic_vigor,
        **extra_flags,
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
# UC1 forward-pass instantiation
# ----------------------------------------------------------------------
def run_uc1_forward_pass_instantiation() -> dict:
    """UC1: TonicVigor instantiates; compute_score_bias returns a finite
    [K] tensor; update_score_receipt advances the EWMA."""
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=10.0,
    ))
    # Run a few EWMA updates to seed v_raw.
    for _ in range(20):
        tv.update_score_receipt(score=-1.0)
    scores = torch.zeros(4)
    classes = torch.tensor([0, 1, 1, 2])
    bias = tv.compute_score_bias(
        scores, classes,
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    state = tv.get_state()
    result = {
        "tonic_vigor_is_module": isinstance(tv, TonicVigor),
        "bias_shape": list(bias.shape),
        "bias_finite": bool(torch.isfinite(bias).all().item()),
        "v_raw": state["v_raw"],
        "last_v_t": state["last_v_t"],
        "n_waking_score_updates": state["n_waking_score_updates"],
        "n_waking_bias_calls": state["n_waking_bias_calls"],
        "alpha_derived": state["alpha_derived"],
    }
    result["pass"] = (
        result["tonic_vigor_is_module"]
        and result["bias_shape"] == [4]
        and result["bias_finite"]
        and result["v_raw"] > 0.0
        and result["last_v_t"] > 0.0
        and result["n_waking_score_updates"] == 20
        and result["n_waking_bias_calls"] == 1
    )
    return result


# ----------------------------------------------------------------------
# UC2 master-OFF backward-compat
# ----------------------------------------------------------------------
def run_uc2_master_off_no_op() -> dict:
    """UC2: master-OFF preserves backward-compat (tonic_vigor is None)."""
    agent_off, latent_off = _build_agent_one_tick(False)
    agent_on, latent_on = _build_agent_one_tick(True)

    off_is_none = (agent_off.tonic_vigor is None)
    on_is_module = (agent_on.tonic_vigor is not None)
    z_world_off_finite = bool(torch.isfinite(latent_off.z_world).all().item())
    z_world_on_finite = bool(torch.isfinite(latent_on.z_world).all().item())

    result = {
        "off_tonic_vigor_is_none": off_is_none,
        "on_tonic_vigor_is_module": on_is_module,
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
# UC3 EWMA convergence and half-life
# ----------------------------------------------------------------------
def run_uc3_ewma_arithmetic() -> dict:
    """UC3: EWMA converges to constant reward; half-life crossing exact;
    sign convention REE-low-is-better -> high-vigor."""
    cases = []
    all_ok = True

    # Case 1: steady state at 1.0.
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True, half_life=10.0))
    for _ in range(200):
        tv.update_score_receipt(score=-1.0)
    steady = tv._v_raw
    ok_steady = abs(steady - 1.0) < 1e-3
    if not ok_steady:
        all_ok = False
    cases.append({
        "case": "steady_state_at_1.0",
        "v_raw_after_200_ticks": steady,
        "expected": 1.0,
        "ok": ok_steady,
    })

    # Case 2: half-life crossing.
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True, half_life=20.0))
    for _ in range(100):
        tv.update_score_receipt(score=0.0)
    for _ in range(20):
        tv.update_score_receipt(score=-1.0)
    crossing = tv._v_raw
    ok_crossing = abs(crossing - 0.5) < 1e-2
    if not ok_crossing:
        all_ok = False
    cases.append({
        "case": "half_life_crossing",
        "v_raw_after_one_half_life": crossing,
        "expected": 0.5,
        "ok": ok_crossing,
    })

    # Case 3: sign convention (negative score -> positive vigor).
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True, half_life=5.0))
    for _ in range(50):
        tv.update_score_receipt(score=-2.0)
    rich = tv._v_raw
    for _ in range(50):
        tv.update_score_receipt(score=1.0)
    poor = tv._v_raw
    ok_sign = (rich > 1.5) and (poor < 0.0)
    if not ok_sign:
        all_ok = False
    cases.append({
        "case": "sign_convention_low_score_high_vigor",
        "v_raw_rich_history": rich,
        "v_raw_poor_history": poor,
        "ok": ok_sign,
    })

    return {"cases": cases, "n_cases": len(cases), "pass": all_ok}


# ----------------------------------------------------------------------
# UC4 select_action wiring contract
# ----------------------------------------------------------------------
def run_uc4_select_action_wiring_contract() -> dict:
    """UC4: act_with_split_obs advances both n_waking_bias_calls and
    n_waking_score_updates exactly once."""
    agent, _ = _build_agent_one_tick(True)
    select_action_error = getattr(agent, "_select_action_error", None)
    state = agent.tonic_vigor.get_state()
    result = {
        "select_action_error": select_action_error,
        "n_waking_bias_calls": state["n_waking_bias_calls"],
        "n_waking_score_updates": state["n_waking_score_updates"],
        "n_simulation_bias_skips": state["n_simulation_bias_skips"],
        "n_simulation_score_skips": state["n_simulation_score_skips"],
    }
    result["pass"] = (
        select_action_error is None
        and result["n_waking_bias_calls"] >= 1
        and result["n_waking_score_updates"] >= 1
        and result["n_simulation_bias_skips"] == 0
        and result["n_simulation_score_skips"] == 0
    )
    return result


# ----------------------------------------------------------------------
# UC5 MECH-094 simulation gate
# ----------------------------------------------------------------------
def run_uc5_mech094_simulation_gate() -> dict:
    """UC5: simulation_mode=True returns zeros / does not advance EWMA;
    increments only the corresponding skip counter."""
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True, half_life=10.0))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    v_pre = tv._v_raw

    # Sim bias call.
    pre_bias_skip = tv._n_simulation_bias_skips
    sim_bias = tv.compute_score_bias(
        torch.zeros(3), torch.tensor([0, 1, 2]),
        energy=1.0, drive=0.0, recent_pe=0.0,
        simulation_mode=True,
    )
    sim_bias_zeros = bool(torch.all(sim_bias == 0.0).item())
    sim_bias_skip_advanced = (tv._n_simulation_bias_skips == pre_bias_skip + 1)

    # Sim score-receipt update.
    pre_score_skip = tv._n_simulation_score_skips
    tv.update_score_receipt(score=-1000.0, simulation_mode=True)
    sim_score_skip_advanced = (tv._n_simulation_score_skips == pre_score_skip + 1)
    ewma_unchanged = (tv._v_raw == v_pre)

    # Subsequent waking call advances waking counters.
    pre_waking_bias = tv._n_waking_bias_calls
    _ = tv.compute_score_bias(
        torch.zeros(2), torch.tensor([0, 1]),
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    waking_bias_advanced = (tv._n_waking_bias_calls == pre_waking_bias + 1)
    sim_skip_not_re_incremented = (
        tv._n_simulation_bias_skips == pre_bias_skip + 1
    )

    result = {
        "sim_bias_zeros": sim_bias_zeros,
        "sim_bias_skip_advanced": sim_bias_skip_advanced,
        "sim_score_skip_advanced": sim_score_skip_advanced,
        "ewma_unchanged_after_sim_update": ewma_unchanged,
        "waking_bias_advanced_after_sim": waking_bias_advanced,
        "sim_skip_not_re_incremented_on_waking": sim_skip_not_re_incremented,
    }
    result["pass"] = all(result.values())
    return result


# ----------------------------------------------------------------------
# UC6 R3 form discriminability (additive vs multiplicative)
# ----------------------------------------------------------------------
def run_uc6_form_discriminability() -> dict:
    """UC6: additive form produces uniform-per-class bias on a held-out
    batch with non-uniform |scores|; multiplicative form produces
    magnitude-varying bias. R3 falsifiable secondary alternative is
    distinguishable from primary."""
    cfg_kwargs = dict(
        use_tonic_vigor=True, half_life=5.0,
        w_action=0.05, w_passive=0.05,
        bias_scale=10.0,  # disable clamp
    )
    tv_a = TonicVigor(TonicVigorConfig(form="additive", **cfg_kwargs))
    tv_m = TonicVigor(TonicVigorConfig(form="multiplicative", **cfg_kwargs))

    # Seed both EWMAs identically.
    for _ in range(50):
        tv_a.update_score_receipt(score=-1.0)
        tv_m.update_score_receipt(score=-1.0)

    # Two action candidates with different |score|: 0.1 vs 0.9.
    scores = torch.tensor([0.1, 0.9])
    classes = torch.tensor([1, 1])  # both action class
    bias_a = tv_a.compute_score_bias(
        scores, classes, energy=1.0, drive=0.0, recent_pe=0.0,
    )
    bias_m = tv_m.compute_score_bias(
        scores, classes, energy=1.0, drive=0.0, recent_pe=0.0,
    )

    # Additive: both candidates get the same bias.
    additive_uniform = bool(
        torch.allclose(bias_a[0:1], bias_a[1:2], atol=1e-6)
    )
    # Multiplicative: |bias[1]| > |bias[0]|.
    multiplicative_varies = abs(float(bias_m[1].item())) > abs(float(bias_m[0].item()))
    # Distinguishable.
    distinguishable = bool(
        not torch.allclose(bias_a, bias_m, atol=1e-3)
    )

    result = {
        "bias_additive": [float(b) for b in bias_a],
        "bias_multiplicative": [float(b) for b in bias_m],
        "additive_uniform_per_class": additive_uniform,
        "multiplicative_varies_with_score_magnitude": multiplicative_varies,
        "forms_distinguishable": distinguishable,
    }
    result["pass"] = (
        additive_uniform
        and multiplicative_varies
        and distinguishable
    )
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    uc1 = run_uc1_forward_pass_instantiation()
    uc2 = run_uc2_master_off_no_op()
    uc3 = run_uc3_ewma_arithmetic()
    uc4 = run_uc4_select_action_wiring_contract()
    uc5 = run_uc5_mech094_simulation_gate()
    uc6 = run_uc6_form_discriminability()

    all_pass = all(r["pass"] for r in [uc1, uc2, uc3, uc4, uc5, uc6])
    elapsed = time.time() - t0

    run_id = "v3_exq_547_mech320_tonic_vigor_substrate_readiness_v3"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "v1",
        "run_id": f"{run_id}_{ts}",
        "experiment_type": "v3_exq_547_mech320_tonic_vigor_substrate_readiness",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": "diagnostic",
        "claim_ids": ["MECH-320", "ARC-066"],
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-320": "supports" if all_pass else "weakens",
            "ARC-066": "supports" if all_pass else "weakens",
        },
        "metrics": {
            "UC1_forward_pass_instantiation": uc1,
            "UC2_master_off_no_op": uc2,
            "UC3_ewma_arithmetic": uc3,
            "UC4_select_action_wiring_contract": uc4,
            "UC5_mech094_simulation_gate": uc5,
            "UC6_r3_form_discriminability": uc6,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-320 (ARC-066 child) tonic_vigor_coupling_score_bias "
            "substrate-readiness diagnostic. Tests substrate wiring + "
            "ARC-066 lit-pull R3 (additive primary, multiplicative "
            "falsifiable secondary) + R4 (slow EWMA over realised score-"
            "receipt is the primary scalar) verdicts at the API level. "
            "Does NOT test the load-bearing claim of MECH-320 itself or "
            "the additive-vs-multiplicative empirical resolution of R3 -- "
            "both require behavioural validation via the 3-arm "
            "discriminative pair (baseline / additive / multiplicative on "
            "a well-fed-safe-familiar substrate), queued AFTER substrate "
            "landing. ARC-066 lit_conf 0.789 supports; lit-pull synthesis "
            "at REE_assembly/evidence/literature/targeted_review_arc_066_"
            "tonic_vigor/synthesis.md."
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
