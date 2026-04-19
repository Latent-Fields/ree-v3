#!/opt/local/bin/python3
"""
V3-EXQ-446 -- SD-032a Salience-Network Coordinator validation.

Claims: SD-032a, MECH-259, MECH-261

Substrate readiness validation for SD-032a (IMPLEMENTED 2026-04-19). Confirms:
  (a) The coordinator instantiates under use_salience_coordinator=True and
      ticks on every action selection without disturbing the existing dACC /
      E3 path.
  (b) operating_mode is non-trivially populated -- entropy across the four
      modes is > 0 across the rollout (i.e., the soft vector reflects actual
      input variation, not a frozen one-hot baseline).
  (c) MECH-259 mode_switch_trigger fires when a synthetic high-PE bundle is
      injected (salience aggregate exceeds the configured threshold AND the
      soft argmax flips off external_task).
  (d) MECH-261 write-gate registry returns values in [0, 1] for all 8
      default targets. A V4-style new mode + new target can be registered
      via the public API without coordinator schema changes.
  (e) Backward compatibility -- coordinator-OFF arm produces identical
      action-class distribution to the dACC-only baseline (within seed
      tolerance), since coordinator is purely additive when
      salience_apply_to_dacc_bias=False.

Conditions (2 seeds each):
  COORD_OFF:  use_dacc=True, use_salience_coordinator=False
  COORD_ON:   use_dacc=True, use_salience_coordinator=True,
              salience_apply_to_dacc_bias=False (no behavioural change path)

Plus a non-stochastic "unit" block that tests the coordinator API directly
(synthetic injection + V4 extension), independent of seeds.

Acceptance checks:
  C1 (operating_mode populates): in COORD_ON, mean entropy of operating_mode
     across the rollout > 0.05 nats (some mode mixing). Threshold low: any
     non-frozen vector passes.
  C2 (MECH-259 trigger fires): synthetic injection of a high-pe bundle
     (pe=10.0) into the coordinator produces mode_switch_trigger=True at
     least once. Deterministic, run once.
  C3 (MECH-261 gate range): all 8 default targets return write_gate values
     in [0, 1] under both initial state and post-injection state.
  C4 (V4 extensibility): coordinator accepts a new mode key via
     register_target without exception, and write_gate on the new target
     returns a value in [0, 1] when the new mode is added to the affinity
     map.
  C5 (backward compat): COORD_OFF and COORD_ON action-class entropies are
     within 0.2 nats of each other (coordinator is a pure observer when
     salience_apply_to_dacc_bias=False).

PASS: all of C1..C5.
FAIL otherwise.

experiment_purpose=diagnostic. Substrate readiness gate, not governance
evidence yet -- behavioural validation of MECH-261 (forward-propagation
bias, EXP-0148) requires SD-033 substrates to consume operating_mode
natively, which is deferred until those land.

See REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
See ree-v3/CLAUDE.md "SD-032a / MECH-259 / MECH-261 ..." section.
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
from ree_core.cingulate import SalienceCoordinator, SalienceCoordinatorConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_446_sd032a_salience_coordinator"
CLAIM_IDS = ["SD-032a", "MECH-259", "MECH-261"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
EPISODES = 6
STEPS_PER_EP = 80

CONDITIONS = ["COORD_OFF", "COORD_ON"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    use_coord = condition == "COORD_ON"
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
        # SD-032b dACC active in both arms so coordinator has a real bundle to read.
        use_dacc=True,
        dacc_weight=1.0,
        dacc_interaction_weight=0.3,
        dacc_foraging_weight=0.2,
        dacc_suppression_weight=0.5,
        dacc_suppression_memory=8,
        dacc_precision_scale=500.0,
        dacc_effort_cost=0.1,
        # SD-032a coordinator under test
        use_salience_coordinator=use_coord,
        salience_switch_threshold=1.0,
        salience_dacc_pe_weight=1.0,
        salience_dacc_foraging_weight=0.5,
        salience_apply_to_dacc_bias=False,  # observer mode -- no behavioural change
    )
    return REEAgent(cfg)


def _shannon_entropy(values: List[float]) -> float:
    """Shannon entropy of a probability distribution (in nats)."""
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
    mode_entropies: List[float] = []
    trigger_count = 0
    n_ticks = 0

    for ep in range(EPISODES):
        obs, _info = env.reset()
        for step in range(STEPS_PER_EP):
            action = agent.act(obs)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            obs, _r, terminated, truncated, _info = env.step(a_idx)
            if condition == "COORD_ON" and agent.salience is not None:
                tick = agent._salience_last_tick
                if tick is not None:
                    n_ticks += 1
                    mode_entropies.append(
                        _shannon_entropy(list(tick["operating_mode"].values()))
                    )
                    if tick["mode_switch_trigger"]:
                        trigger_count += 1
            if terminated or truncated:
                break

    diag = (
        agent.salience.diagnostics if condition == "COORD_ON" and agent.salience is not None
        else {"n_ticks": 0, "n_switches": 0}
    )
    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _action_class_entropy(action_counts),
        "n_actions": sum(action_counts.values()),
        "mean_mode_entropy": (sum(mode_entropies) / len(mode_entropies)) if mode_entropies else 0.0,
        "trigger_count_natural": trigger_count,
        "coord_n_ticks": diag["n_ticks"],
        "coord_n_switches": diag["n_switches"],
    }


def _run_unit_checks() -> Dict:
    """Deterministic unit-level checks of the coordinator API.

    Independent of env / seeds. Tests synthetic high-PE injection (C2),
    write-gate range (C3), and V4 extensibility (C4).
    """
    coord = SalienceCoordinator(SalienceCoordinatorConfig(switch_threshold=1.0))

    # Initial gate values (operating_mode = pure external_task).
    initial_gates = coord.write_gates()
    initial_gate_in_range = all(0.0 <= v <= 1.0 for v in initial_gates.values())
    initial_gate_count = len(initial_gates)

    # Synthetic injection: high pe bundle -> trigger should fire.
    bundle = {"pe": 10.0, "foraging_value": 1.0, "choice_difficulty": 0.5}
    out = coord.tick(dacc_bundle=bundle, drive_level=0.0, is_offline=False)
    triggered = bool(out["mode_switch_trigger"])
    soft_argmax_after = max(out["operating_mode"].items(), key=lambda kv: kv[1])[0]
    salience_aggregate = out["salience_aggregate"]
    effective_threshold = out["effective_threshold"]

    # Gates after tick.
    post_gates = coord.write_gates()
    post_gate_in_range = all(0.0 <= v <= 1.0 for v in post_gates.values())

    # V4 extensibility: register a new mode + new target.
    new_mode = "parallel_goal_deliberation"
    coord.mode_names.append(new_mode)
    # Repeat tick to avoid the new mode being absent from operating_mode dict.
    coord.tick(dacc_bundle=bundle, drive_level=0.0, is_offline=False)
    coord.register_target("sd_033e", {
        "external_task": 0.0,
        "internal_planning": 0.5,
        "internal_replay": 0.0,
        "offline_consolidation": 0.0,
        new_mode: 1.0,
    })
    sd_033e_gate = coord.write_gate("sd_033e")
    v4_ok = isinstance(sd_033e_gate, float) and 0.0 <= sd_033e_gate <= 1.0

    return {
        "trigger_fired_on_injection": triggered,
        "soft_argmax_after_injection": soft_argmax_after,
        "salience_aggregate": salience_aggregate,
        "effective_threshold": effective_threshold,
        "initial_gate_count": initial_gate_count,
        "initial_gates_in_range": initial_gate_in_range,
        "post_gates_in_range": post_gate_in_range,
        "v4_register_target_ok": v4_ok,
        "v4_sd_033e_gate_value": sd_033e_gate,
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

    print("Running unit-level coordinator checks")
    unit = _run_unit_checks()
    print(f"  -> {unit}")

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    off = by_cond("COORD_OFF")
    on = by_cond("COORD_ON")

    # C1: operating_mode populates non-trivially in COORD_ON.
    c1 = all(r["mean_mode_entropy"] > 0.05 for r in on)

    # C2: synthetic injection fires the trigger.
    c2 = unit["trigger_fired_on_injection"]

    # C3: gate values in [0, 1] both initially and after a tick.
    c3 = unit["initial_gates_in_range"] and unit["post_gates_in_range"] and unit["initial_gate_count"] >= 8

    # C4: V4 extensibility works.
    c4 = unit["v4_register_target_ok"]

    # C5: backward compatibility -- action-class entropy similar across arms.
    backwards_compat_deltas = []
    for off_r, on_r in zip(off, on):
        backwards_compat_deltas.append(
            abs(on_r["action_class_entropy"] - off_r["action_class_entropy"])
        )
    c5 = all(d <= 0.2 for d in backwards_compat_deltas)

    outcome = "PASS" if (c1 and c2 and c3 and c4 and c5) else "FAIL"

    summary = {
        "c1_operating_mode_populates": {
            "min_mean_mode_entropy": min(r["mean_mode_entropy"] for r in on) if on else 0.0,
            "threshold": 0.05,
            "pass": c1,
            "desc": "mean operating_mode entropy > 0.05 nats per seed in COORD_ON",
        },
        "c2_mech259_trigger_fires_on_injection": {
            "salience_aggregate": unit["salience_aggregate"],
            "effective_threshold": unit["effective_threshold"],
            "soft_argmax_after_injection": unit["soft_argmax_after_injection"],
            "pass": c2,
            "desc": "synthetic pe=10.0 bundle produces mode_switch_trigger=True",
        },
        "c3_mech261_gate_range": {
            "initial_gate_count": unit["initial_gate_count"],
            "initial_gates_in_range": unit["initial_gates_in_range"],
            "post_gates_in_range": unit["post_gates_in_range"],
            "pass": c3,
            "desc": "all 8 default write-gate values in [0, 1]",
        },
        "c4_v4_extensibility": {
            "sd_033e_gate_value": unit["v4_sd_033e_gate_value"],
            "pass": c4,
            "desc": "register_target accepts new mode + target without schema change",
        },
        "c5_backward_compat": {
            "max_entropy_delta": max(backwards_compat_deltas) if backwards_compat_deltas else 0.0,
            "threshold": 0.2,
            "pass": c5,
            "desc": "COORD_OFF and COORD_ON action-class entropies within 0.2 nats",
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    per_claim = {
        "SD-032a": "supports" if (c1 and c5) else "weakens",
        "MECH-259": "supports" if c2 else "weakens",
        "MECH-261": "supports" if (c3 and c4) else "weakens",
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
