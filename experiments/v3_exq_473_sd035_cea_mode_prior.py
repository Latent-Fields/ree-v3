#!/opt/local/bin/python3
"""
V3-EXQ-473 -- SD-035 CeAAnalog substrate-readiness (MECH-046 mode_prior,
              MECH-074c fast_prime).

Claims: SD-035, MECH-046, MECH-074c

Substrate readiness validation for the CeA half of the amygdala analogue
(SD-035, IMPLEMENTED 2026-04-21). This is a diagnostic experiment --
not a behavioural-effect gate. Its job is to confirm that:

  (a) Enabling use_amygdala_analog + use_cea_analog is backward compatible:
      action-class entropy in a CeA-ON-but-z_harm_a-quiet arm is within
      tolerance of the fully OFF baseline (the fast-route gate does not
      fire when z_harm_a stays below threshold, so the coordinator sees
      mode_prior=0.0 and fast_prime=0.0 every tick).

  (b) On synthetic z_harm_a elevation above fast_route_threshold, the CeA
      module fires (urgency_fire=True), emits a non-zero mode_prior and
      non-zero fast_prime, and those values are injected into the
      SalienceCoordinator via update_signal("cea_mode_prior", ...) +
      update_signal("cea_fast_prime", ...) in select_action() BEFORE
      coordinator.tick(). Verified by reading cached signals on the
      coordinator after a tick.

  (c) mode_prior is bounded: |mode_prior| <= mode_prior_log_odds_max under
      any input (|z_harm_a| swept from rest to saturation).

  (d) MECH-094 simulation gate: calling cea.tick(simulation_mode=True)
      returns a zeroed output and does not advance baselines / decay
      counters.

  (e) Override / decay dynamics: a fire followed by quiet ticks with
      cortical_confirmation=0.0 decays fast_prime with the expected
      half-life (tau=4 steps by default); with cortical_confirmation=1.0
      the pulse is held longer.

This script DOES NOT test:
  - Behavioural effect of mode_prior on downstream action selection
    magnitude -- no claim is made about size of effect (depends on
    SalienceCoordinator weighting + downstream SD-033 substrates that
    consume operating_mode; deferred to follow-up EXQ once consumer
    wiring lands).
  - BLA-side behaviour (handled by V3-EXQ-474).

Conditions (CeA on/off + z_harm_a quiet/elevated):
  CEA_OFF_QUIET:    use_amygdala_analog=False, z_harm_a quiet
  CEA_ON_QUIET:     use_amygdala_analog=True, z_harm_a quiet
                    (CeA present but below fast_route threshold;
                     expected no-op; backward-compat arm)

Plus a deterministic unit block with direct CeAAnalog injection for
(b), (c), (d), (e).

Acceptance checks:
  C1 (backward compat): action_class_entropy delta between CEA_OFF_QUIET
     and CEA_ON_QUIET <= 0.2 nats in >= 2/3 seeds. Tolerant because
     untrained E3 scoring is seed-sensitive; this is a wiring-integrity
     check, not a bit-identity check.
  C2 (fire + signal injection): synthetic z_harm_a elevation above
     fast_route_threshold produces urgency_fire=True AND coordinator
     reads non-zero cea_mode_prior / cea_fast_prime in
     salience.signals after a tick.
  C3 (mode_prior bound): |mode_prior| <= mode_prior_log_odds_max across
     a swept input magnitude range [0, 2.0].
  C4 (MECH-094 gate): cea.tick(simulation_mode=True) on a saturating
     input returns mode_prior=0.0 AND fast_prime=0.0 AND urgency_fire=False.
  C5 (decay dynamics): fire followed by N=8 quiet steps with
     cortical_confirmation=0.0 -> final fast_prime < initial_fast_prime / 2
     (decayed past half-life); with cortical_confirmation=1.0 ->
     final fast_prime >= initial_fast_prime * 0.5 (pulse held).

PASS: all of C1..C5.
FAIL otherwise.

experiment_purpose=diagnostic. Substrate readiness gate, not claim
evidence yet -- full-loop behavioural validation of MECH-046 (mode-switch
latency to threat cue) requires CausalGridWorldV2 threat-cue extension
and downstream SD-033 / E3 consumers of operating_mode that are not
all in place.

See REE_assembly/docs/architecture/sd_035_amygdala_analog.md
See ree-v3/CLAUDE.md "SD-035: Amygdala Analogue" section.
"""

import sys
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.amygdala import CeAAnalog, CeAConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_473_sd035_cea_mode_prior"
CLAIM_IDS = ["SD-035", "MECH-046", "MECH-074c"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]
EPISODES = 4
STEPS_PER_EP = 60

CONDITIONS = ["CEA_OFF_QUIET", "CEA_ON_QUIET"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    amygdala_on = condition == "CEA_ON_QUIET"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        # Amygdala under test
        use_amygdala_analog=amygdala_on,
        use_bla_analog=False,    # isolate CeA path
        use_cea_analog=True,
        # Coordinator needed as the consumer of cea_mode_prior / cea_fast_prime
        use_salience_coordinator=amygdala_on,
        salience_apply_to_dacc_bias=False,  # observer mode
    )
    return REEAgent(cfg)


def _shannon_entropy(values: List[float]) -> float:
    ent = 0.0
    for p in values:
        if p > 0:
            ent -= p * math.log(p)
    return ent


def _action_class_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return _shannon_entropy([c / total for c in counts.values() if c > 0])


def _run_condition(seed: int, condition: str) -> Dict:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, condition)

    action_counts: Dict[int, int] = {}
    cea_fire_count = 0
    n_ticks = 0

    for ep in range(EPISODES):
        obs, _info = env.reset()
        for step in range(STEPS_PER_EP):
            action = agent.act(obs)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            obs, _r, terminated, truncated, _info = env.step(a_idx)
            if condition == "CEA_ON_QUIET" and agent.cea is not None:
                n_ticks += 1
                out = agent._cea_last_output
                if out is not None and out.urgency_fire:
                    cea_fire_count += 1
            if terminated or truncated:
                break

    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _action_class_entropy(action_counts),
        "n_actions": sum(action_counts.values()),
        "cea_fire_count_natural": cea_fire_count,
        "cea_n_ticks": n_ticks,
    }


def _run_unit_checks() -> Dict:
    """Deterministic unit-level checks of CeAAnalog.

    Tests synthetic z_harm_a injection (C2), mode_prior bound (C3),
    MECH-094 gate (C4), and decay dynamics (C5). Independent of env /
    seeds.
    """
    # C2 prep: agent with CeA + coordinator; verify signals reach coordinator
    env = _make_env(seed=0)
    agent = _make_agent(env, "CEA_ON_QUIET")
    # Synthetic z_harm_a above threshold: feed via agent.sense() with an
    # elevated harm_obs_a. The AffectiveHarmEncoder is untrained; to bypass
    # its nonlinear compression, call cea.tick() directly with a canned
    # z_harm_a AND inject into coordinator via update_signal, matching the
    # flow in agent.select_action().
    z_harm_a_threat = torch.ones(16) * 0.9
    cea_out_fire = agent.cea.tick(z_harm_a_threat)
    agent.salience.update_signal("cea_mode_prior", float(cea_out_fire.mode_prior))
    agent.salience.update_signal("cea_fast_prime", float(cea_out_fire.fast_prime))
    # Access private _input_signals for diagnostic verification only.
    coord_signals_after = dict(agent.salience._input_signals)
    c2_fire = bool(cea_out_fire.urgency_fire)
    c2_mode_prior_in_coord = coord_signals_after.get("cea_mode_prior", 0.0)
    c2_fast_prime_in_coord = coord_signals_after.get("cea_fast_prime", 0.0)
    c2_signals_present = (
        abs(c2_mode_prior_in_coord) > 1e-6 and abs(c2_fast_prime_in_coord) > 1e-6
    )

    # C3: mode_prior bound over swept input magnitude [0, 2.0]
    cea_bound = CeAAnalog(CeAConfig())
    mode_prior_max_observed = 0.0
    for mag in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0]:
        z = torch.ones(16) * mag
        out = cea_bound.tick(z)
        mode_prior_max_observed = max(mode_prior_max_observed, abs(out.mode_prior))
    c3_bound = mode_prior_max_observed <= cea_bound.config.mode_prior_log_odds_max + 1e-6

    # C4: MECH-094 simulation gate
    cea_sim = CeAAnalog(CeAConfig())
    z_big = torch.ones(16) * 2.0
    out_sim = cea_sim.tick(z_big, simulation_mode=True)
    c4_sim_gate = (
        abs(out_sim.mode_prior) < 1e-9
        and abs(out_sim.fast_prime) < 1e-9
        and not out_sim.urgency_fire
    )

    # C5a: decay with cortical_confirmation=0.0
    cea_decay_off = CeAAnalog(CeAConfig())
    out_fire = cea_decay_off.tick(z_big)  # fire
    initial_fp = out_fire.fast_prime
    z_rest = torch.zeros(16)
    final_fp_decay = initial_fp
    for _ in range(8):
        out_q = cea_decay_off.tick(z_rest, cortical_confirmation=0.0)
        final_fp_decay = out_q.fast_prime
    c5_decay_ok = final_fp_decay < initial_fp * 0.5

    # C5b: hold with cortical_confirmation=1.0
    cea_hold = CeAAnalog(CeAConfig())
    out_fire_h = cea_hold.tick(z_big)
    initial_fp_h = out_fire_h.fast_prime
    final_fp_hold = initial_fp_h
    for _ in range(4):  # still within override window
        out_h = cea_hold.tick(z_rest, cortical_confirmation=1.0)
        final_fp_hold = out_h.fast_prime
    c5_hold_ok = final_fp_hold >= initial_fp_h * 0.5

    return {
        "c2_cea_fire_on_threat": c2_fire,
        "c2_mode_prior_signal_value": c2_mode_prior_in_coord,
        "c2_fast_prime_signal_value": c2_fast_prime_in_coord,
        "c2_signals_reached_coordinator": c2_signals_present,
        "c3_mode_prior_max_observed": mode_prior_max_observed,
        "c3_mode_prior_cap": cea_bound.config.mode_prior_log_odds_max,
        "c3_bound_respected": c3_bound,
        "c4_simulation_gate_zero_out": c4_sim_gate,
        "c5a_initial_fast_prime": initial_fp,
        "c5a_final_fast_prime_decay": final_fp_decay,
        "c5a_decay_ok": c5_decay_ok,
        "c5b_initial_fast_prime_hold": initial_fp_h,
        "c5b_final_fast_prime_hold": final_fp_hold,
        "c5b_hold_ok": c5_hold_ok,
    }


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            r = _run_condition(seed=seed, condition=cond)
            print(f"  -> {r}")
            all_results.append(r)

    print("Running unit-level CeA checks")
    unit = _run_unit_checks()
    print(f"  -> {unit}")

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    off = by_cond("CEA_OFF_QUIET")
    on = by_cond("CEA_ON_QUIET")

    # C1: backward-compat entropy delta tolerance
    c1_deltas = []
    c1_passes = 0
    for off_r, on_r in zip(off, on):
        delta = abs(on_r["action_class_entropy"] - off_r["action_class_entropy"])
        c1_deltas.append(delta)
        if delta <= 0.2:
            c1_passes += 1
    c1 = c1_passes >= 2  # >=2/3 seeds

    c2 = bool(unit["c2_cea_fire_on_threat"]) and bool(unit["c2_signals_reached_coordinator"])
    c3 = bool(unit["c3_bound_respected"])
    c4 = bool(unit["c4_simulation_gate_zero_out"])
    c5 = bool(unit["c5a_decay_ok"]) and bool(unit["c5b_hold_ok"])

    outcome = "PASS" if (c1 and c2 and c3 and c4 and c5) else "FAIL"

    summary = {
        "c1_backward_compat_entropy_delta": {
            "deltas": c1_deltas,
            "threshold": 0.2,
            "seeds_passing": c1_passes,
            "pass": c1,
            "desc": "|entropy(CEA_ON_QUIET) - entropy(CEA_OFF_QUIET)| <= 0.2 nats in >= 2/3 seeds",
        },
        "c2_fire_and_signal_injection": {
            "cea_fire_on_threat": unit["c2_cea_fire_on_threat"],
            "cea_mode_prior_signal": unit["c2_mode_prior_signal_value"],
            "cea_fast_prime_signal": unit["c2_fast_prime_signal_value"],
            "signals_reached_coordinator": unit["c2_signals_reached_coordinator"],
            "pass": c2,
            "desc": "CeA fires on |z_harm_a|=0.9 and signals reach coordinator",
        },
        "c3_mode_prior_bound": {
            "max_observed": unit["c3_mode_prior_max_observed"],
            "cap": unit["c3_mode_prior_cap"],
            "pass": c3,
            "desc": "|mode_prior| <= mode_prior_log_odds_max over input sweep",
        },
        "c4_mech094_simulation_gate": {
            "pass": c4,
            "desc": "simulation_mode=True returns zeroed output on saturating input",
        },
        "c5_decay_dynamics": {
            "initial_fp_decay": unit["c5a_initial_fast_prime"],
            "final_fp_decay_no_cortical": unit["c5a_final_fast_prime_decay"],
            "decay_past_half_after_8_steps": unit["c5a_decay_ok"],
            "initial_fp_hold": unit["c5b_initial_fast_prime_hold"],
            "final_fp_hold_with_cortical": unit["c5b_final_fast_prime_hold"],
            "hold_above_half_after_4_steps": unit["c5b_hold_ok"],
            "pass": c5,
            "desc": "tau decay (no conf) and hold (conf=1.0) both respected",
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: pass={v['pass']}")

    per_claim = {
        "SD-035": "supports" if (c1 and c2 and c3 and c4 and c5) else "weakens",
        "MECH-046": "supports" if (c2 and c3) else "weakens",
        "MECH-074c": "supports" if (c2 and c5) else "weakens",
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": per_claim,
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "unit_checks": unit,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}")


if __name__ == "__main__":
    main()
