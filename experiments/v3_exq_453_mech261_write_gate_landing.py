#!/opt/local/bin/python3
"""
V3-EXQ-453 -- MECH-261 per-subdivision write-gate landing validation.

Claims: MECH-261, SD-032a, SD-032e

Extends V3-EXQ-446 (SD-032a coordinator general arithmetic, PASS) to MECH-261's
specific claim: the SalienceCoordinator's dict-keyed write-gate registry hosts
per-subdivision gates that (a) all exist under default configuration, (b) fall
in [0, 1] for all operating-mode vectors, (c) differ meaningfully across modes
(not degenerate to a single constant), (d) support post-hoc registration of new
targets without schema changes (V4 extensibility), and (e) actually LAND on
their named substrate -- i.e., changing the gate value changes the downstream
effect on that substrate. The pACC (SD-032e) autonomic write-back provides the
landing site for (e) -- a substrate that genuinely consumes the "autonomic"
gate via its EMA accumulator.

Conditions (2 seeds each):
  COORD_OFF:  use_salience_coordinator=False (coordinator/gates not exercised).
  COORD_ON:   use_salience_coordinator=True (core test path; registry live).

Unit checks (deterministic, no seed dependence, run once at start):
  UC1 default_targets_exist
       SalienceCoordinator's default registry keys include all 8 of
       {sd_033a, sd_033b, sd_033c, sd_033d, hc_viability, sensory_buffer,
        autonomic, e3_policy}.
  UC2 gate_values_in_bounds
       For each of 5 synthetic operating_mode one-hot vectors, every default
       gate returns a float in [0, 1].
  UC3 per_subdivision_gate_specificity
       5-by-8 matrix of gate values (5 one-hot modes x 8 targets). Per-target
       std across modes, averaged across targets, must be >= 0.1 (gates are
       meaningfully mode-specific, not all ~0.5 constants).
  UC4 v4_extensibility
       register_target("parallel_goal_deliberation", custom_weights) with
       arbitrary per-mode weights; the new gate is evaluable and returns
       values in [0, 1] for all 5 synthetic mode vectors.
  UC5 pACC_write_landing
       With pACC enabled, running 50 synthetic ticks at sustained
       z_harm_a_norm=0.5, the autonomic write_gate actually modulates the
       drive_bias update. Operationalised as two 50-tick sweeps with gate
       forced to 1.0 and 0.1 respectively; require
       |drive_bias_high - drive_bias_low| >= 0.02.

Per-seed / per-condition rollout checks (agent-integration live dynamics):
  c6 coord_on_tick_live
       In COORD_ON, the operating_mode vector is non-uniform across the
       episode -- argmax occurs on at least 2 distinct modes (coordinator
       actually switches modes in response to env signals).
  c7 coord_off_no_op
       In COORD_OFF, agent.salience is None, no errors raised, episodes run
       to completion normally.

Acceptance:
  C1 (unit_checks): UC1 AND UC2 AND UC3 AND UC4 AND UC5 all PASS.
  C2 (integration): c6 AND c7 PASS in >=1/2 seeds AND >=2/2 conditions.
  Overall PASS if C1 AND C2.

experiment_purpose = diagnostic (substrate validation).
"""

import argparse
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.cingulate import SalienceCoordinator, SalienceCoordinatorConfig
from ree_core.cingulate.pacc_analog import PACCAnalog, PACCConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_453_mech261_write_gate_landing"
CLAIM_IDS = ["MECH-261", "SD-032a", "SD-032e"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
EPISODES = 6
STEPS_PER_EP = 80
CONDITIONS = ["COORD_OFF", "COORD_ON"]

# The 8 MECH-261 default targets per DEFAULT_GATE_WEIGHTS in
# ree_core/cingulate/salience_coordinator.py.
REQUIRED_TARGETS = [
    "sd_033a",
    "sd_033b",
    "sd_033c",
    "sd_033d",
    "hc_viability",
    "sensory_buffer",
    "autonomic",
    "e3_policy",
]

# 5 synthetic mode vectors: 4 one-hot default modes + external_task baseline
# repeated in first position (makes the "5 one-hot" phrasing exact when
# parallel_goal_deliberation is not registered yet).
DEFAULT_MODES_FOR_MATRIX = [
    "external_task",
    "internal_planning",
    "internal_replay",
    "offline_consolidation",
]


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
        use_dacc=True,
        dacc_weight=1.0,
        dacc_interaction_weight=0.3,
        dacc_foraging_weight=0.2,
        dacc_suppression_weight=0.5,
        dacc_suppression_memory=8,
        dacc_precision_scale=500.0,
        dacc_effort_cost=0.1,
        use_salience_coordinator=use_coord,
        salience_switch_threshold=1.0,
        salience_dacc_pe_weight=1.0,
        salience_dacc_foraging_weight=0.5,
        salience_apply_to_dacc_bias=False,
    )
    return REEAgent(cfg)


def _force_operating_mode(coord: SalienceCoordinator, mode_name: str) -> None:
    """Force the coordinator's cached operating_mode to a one-hot vector.

    Used by UC2/UC3/UC4 to evaluate write_gate against a known mode vector
    without plumbing real inputs. Safe because write_gate() reads
    self._operating_mode directly.
    """
    coord._operating_mode = {
        m: (1.0 if m == mode_name else 0.0) for m in coord.mode_names
    }


def _gate_matrix(coord: SalienceCoordinator, modes: List[str],
                 targets: List[str]) -> List[List[float]]:
    """Return matrix[len(modes)][len(targets)] of write_gate values."""
    out = []
    for m in modes:
        _force_operating_mode(coord, m)
        row = [coord.write_gate(t) for t in targets]
        out.append(row)
    return out


def _std(values: List[float]) -> float:
    """Population std (no torch dependency, keeps unit tests ASCII-pure)."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) * (v - mean) for v in values) / len(values)
    return var ** 0.5


def _run_unit_checks() -> Dict:
    """Deterministic MECH-261 unit checks (no seeds, no env)."""
    unit: Dict = {}

    # --- UC1: default targets exist --------------------------------------
    coord = SalienceCoordinator(SalienceCoordinatorConfig(switch_threshold=1.0))
    default_keys = list(coord.gate_targets)
    missing = [t for t in REQUIRED_TARGETS if t not in default_keys]
    uc1_pass = len(missing) == 0
    unit["UC1"] = {
        "name": "default_targets_exist",
        "required_targets": REQUIRED_TARGETS,
        "default_keys": default_keys,
        "missing": missing,
        "pass": uc1_pass,
        "desc": "all 8 required MECH-261 default targets present in registry",
    }
    print(f"UC1 default_targets_exist verdict: {'PASS' if uc1_pass else 'FAIL'}")

    # --- UC2: gate values in bounds for 5 mode vectors -------------------
    # Register a dummy fifth mode so a genuinely 5-entry matrix can be built,
    # matching the spec's "5 synthetic one-hot mode vectors".
    coord_uc2 = SalienceCoordinator(SalienceCoordinatorConfig(switch_threshold=1.0))
    fifth_mode = "parallel_goal_deliberation"
    coord_uc2.mode_names.append(fifth_mode)
    modes_for_matrix = DEFAULT_MODES_FOR_MATRIX + [fifth_mode]
    matrix = _gate_matrix(coord_uc2, modes_for_matrix, REQUIRED_TARGETS)
    flat_values = [v for row in matrix for v in row]
    in_bounds = all((isinstance(v, float) and 0.0 <= v <= 1.0) for v in flat_values)
    n_out_of_bounds = sum(1 for v in flat_values if not (0.0 <= v <= 1.0))
    uc2_pass = in_bounds
    unit["UC2"] = {
        "name": "gate_values_in_bounds",
        "modes_for_matrix": modes_for_matrix,
        "targets": REQUIRED_TARGETS,
        "n_values": len(flat_values),
        "n_out_of_bounds": n_out_of_bounds,
        "min_value": min(flat_values) if flat_values else None,
        "max_value": max(flat_values) if flat_values else None,
        "pass": uc2_pass,
        "desc": "all 40 (5 modes x 8 targets) gate values in [0, 1]",
    }
    print(f"UC2 gate_values_in_bounds verdict: {'PASS' if uc2_pass else 'FAIL'}")

    # --- UC3: per-subdivision gate specificity ---------------------------
    # Use only the 4 default modes (fifth mode has no weights -> would force
    # artificial zeros and inflate std; UC4 handles the fifth-mode case).
    coord_uc3 = SalienceCoordinator(SalienceCoordinatorConfig(switch_threshold=1.0))
    matrix_4 = _gate_matrix(coord_uc3, DEFAULT_MODES_FOR_MATRIX, REQUIRED_TARGETS)
    # Per-target std across the 4 one-hot modes.
    per_target_std = []
    for t_idx, target in enumerate(REQUIRED_TARGETS):
        col = [matrix_4[m_idx][t_idx] for m_idx in range(len(DEFAULT_MODES_FOR_MATRIX))]
        per_target_std.append((target, _std(col)))
    mean_std = sum(s for _, s in per_target_std) / len(per_target_std)
    uc3_pass = mean_std >= 0.1
    unit["UC3"] = {
        "name": "per_subdivision_gate_specificity",
        "modes": DEFAULT_MODES_FOR_MATRIX,
        "targets": REQUIRED_TARGETS,
        "per_target_std": {t: round(s, 6) for t, s in per_target_std},
        "mean_std_across_targets": round(mean_std, 6),
        "threshold": 0.1,
        "pass": uc3_pass,
        "desc": "mean per-target std across default modes >= 0.1 (mode-specific gates)",
    }
    print(f"UC3 per_subdivision_gate_specificity verdict: {'PASS' if uc3_pass else 'FAIL'}")

    # --- UC4: V4 extensibility -------------------------------------------
    coord_uc4 = SalienceCoordinator(SalienceCoordinatorConfig(switch_threshold=1.0))
    new_mode = "parallel_goal_deliberation"
    coord_uc4.mode_names.append(new_mode)
    # Arbitrary-but-sensible weights: strong write under the new mode, minor
    # read elsewhere. Covers all 5 modes so gate is well-defined.
    custom_weights = {
        "external_task": 0.2,
        "internal_planning": 0.6,
        "internal_replay": 0.1,
        "offline_consolidation": 0.3,
        new_mode: 1.0,
    }
    uc4_pass = True
    register_error = None
    try:
        coord_uc4.register_target(new_mode, custom_weights)
    except Exception as e:
        register_error = repr(e)
        uc4_pass = False

    uc4_gate_values = {}
    if uc4_pass:
        modes_uc4 = DEFAULT_MODES_FOR_MATRIX + [new_mode]
        for m in modes_uc4:
            _force_operating_mode(coord_uc4, m)
            g = coord_uc4.write_gate(new_mode)
            uc4_gate_values[m] = g
            if not (isinstance(g, float) and 0.0 <= g <= 1.0):
                uc4_pass = False

    unit["UC4"] = {
        "name": "v4_extensibility",
        "new_target": new_mode,
        "custom_weights": custom_weights,
        "gate_values_per_mode": uc4_gate_values,
        "register_error": register_error,
        "pass": uc4_pass,
        "desc": "register_target + evaluable gate in [0, 1] across 5 synthetic modes",
    }
    print(f"UC4 v4_extensibility verdict: {'PASS' if uc4_pass else 'FAIL'}")

    # --- UC5: pACC write landing -----------------------------------------
    # Standalone arithmetic test on PACCAnalog. Two 50-tick sweeps at
    # z_harm_a_norm=0.5 sustained, gate forced to 1.0 vs 0.1. Difference in
    # final drive_bias must be >= 0.02 (gate actually modulates accumulation).
    pacc_high = PACCAnalog(PACCConfig(
        drive_alpha=0.01,
        drive_scale=1.0,
        drive_bias_cap=0.5,
        z_harm_a_min=0.0,
        offline_decay=0.0,
    ))
    pacc_low = PACCAnalog(PACCConfig(
        drive_alpha=0.01,
        drive_scale=1.0,
        drive_bias_cap=0.5,
        z_harm_a_min=0.0,
        offline_decay=0.0,
    ))
    N_TICKS = 50
    Z_H_A = 0.5
    for _ in range(N_TICKS):
        pacc_high.tick(z_harm_a_norm=Z_H_A, write_gate=1.0, hypothesis_tag=False)
        pacc_low.tick(z_harm_a_norm=Z_H_A, write_gate=0.1, hypothesis_tag=False)

    drive_bias_high = pacc_high.drive_bias
    drive_bias_low = pacc_low.drive_bias
    delta = abs(drive_bias_high - drive_bias_low)
    uc5_threshold = 0.02
    uc5_pass = delta >= uc5_threshold
    unit["UC5"] = {
        "name": "pACC_write_landing",
        "n_ticks": N_TICKS,
        "z_harm_a_norm": Z_H_A,
        "drive_alpha": 0.01,
        "drive_scale": 1.0,
        "drive_bias_high_gate_1.0": round(drive_bias_high, 6),
        "drive_bias_low_gate_0.1": round(drive_bias_low, 6),
        "abs_delta": round(delta, 6),
        "threshold": uc5_threshold,
        "pass": uc5_pass,
        "desc": "autonomic write_gate actually lands -- drive_bias delta >= 0.02",
    }
    print(f"UC5 pACC_write_landing verdict: {'PASS' if uc5_pass else 'FAIL'}")

    unit["all_unit_checks_pass"] = bool(
        uc1_pass and uc2_pass and uc3_pass and uc4_pass and uc5_pass
    )
    return unit


def _run_condition(seed: int, condition: str, episodes: int, steps_per_ep: int) -> Dict:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, condition)

    argmax_modes_seen = set()
    error_str = None
    n_steps = 0
    try:
        for ep in range(episodes):
            obs, _info = env.reset()
            for _step in range(steps_per_ep):
                action = agent.act(obs)
                a_idx = int(action[0].argmax().item())
                obs, _r, terminated, truncated, _info = env.step(a_idx)
                n_steps += 1
                if condition == "COORD_ON" and agent.salience is not None:
                    tick = agent._salience_last_tick
                    if tick is not None:
                        soft_argmax = max(
                            tick["operating_mode"].items(),
                            key=lambda kv: kv[1],
                        )[0]
                        argmax_modes_seen.add(soft_argmax)
                if terminated or truncated:
                    break
            if (ep + 1) % 2 == 0 or ep == 0:
                print(f"[train] seed={seed} cond={condition} ep {ep+1}/{episodes}")
    except Exception as e:
        error_str = repr(e)

    # c6: in COORD_ON, at least 2 distinct argmax modes observed.
    if condition == "COORD_ON":
        c6_pass = len(argmax_modes_seen) >= 2
        c7_pass = True  # not applicable in ON arm
    else:
        c6_pass = True  # not applicable in OFF arm
        # c7: agent.salience is None and no exception raised.
        c7_pass = (agent.salience is None) and (error_str is None)

    verdict = "PASS" if (c6_pass and c7_pass and error_str is None) else "FAIL"
    print(f"[train] seed={seed} cond={condition} verdict: {verdict}")

    return {
        "seed": seed,
        "condition": condition,
        "n_steps_total": n_steps,
        "argmax_modes_seen": sorted(argmax_modes_seen),
        "agent_salience_is_none": agent.salience is None,
        "error": error_str,
        "c6_coord_on_tick_live_pass": bool(c6_pass),
        "c7_coord_off_no_op_pass": bool(c7_pass),
        "verdict": verdict,
    }


def _evaluate_integration(per_seed_results: List[Dict]) -> Tuple[bool, Dict]:
    """C2 acceptance: c6 AND c7 pass in >=1/2 seeds AND >=2/2 conditions."""
    # Per condition: require pass in >= 1 seed AND all seeds across conditions
    # report non-None results.
    by_cond: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}
    for r in per_seed_results:
        by_cond[r["condition"]].append(r)

    per_cond_pass = {}
    cond_pass_count = 0
    for cond, results in by_cond.items():
        # Check the relevant criterion per condition.
        if cond == "COORD_ON":
            seeds_ok = sum(1 for r in results if r["c6_coord_on_tick_live_pass"] and r["error"] is None)
        else:  # COORD_OFF
            seeds_ok = sum(1 for r in results if r["c7_coord_off_no_op_pass"] and r["error"] is None)
        cond_ok = seeds_ok >= 1
        per_cond_pass[cond] = {"seeds_ok": seeds_ok, "cond_pass": cond_ok}
        if cond_ok:
            cond_pass_count += 1

    c2_pass = cond_pass_count >= 2
    return c2_pass, {
        "per_condition_pass": per_cond_pass,
        "conditions_passed": cond_pass_count,
        "required_conditions": 2,
        "pass": c2_pass,
        "desc": "c6/c7 pass in >=1/2 seeds in each of 2/2 conditions",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Unit checks only (no rollouts). Used for smoke test.",
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        seeds = [SEEDS[0]]
        conditions = [CONDITIONS[0]]
        episodes = 1
        steps_per_ep = 20
        print("DRY-RUN MODE: unit checks + 1 seed x 1 condition x 1 ep")
    else:
        seeds = SEEDS
        conditions = CONDITIONS
        episodes = EPISODES
        steps_per_ep = STEPS_PER_EP

    # Unit checks run once up front.
    print("Running MECH-261 unit checks (UC1-UC5)")
    unit = _run_unit_checks()
    print(f"  unit checks all_pass={unit['all_unit_checks_pass']}")

    # Integration rollouts.
    all_results: List[Dict] = []
    total_runs = len(seeds) * len(conditions)
    run_idx = 0
    for seed in seeds:
        for cond in conditions:
            run_idx += 1
            print(f"Seed {seed} Condition {cond} ({run_idx}/{total_runs})")
            r = _run_condition(
                seed=seed,
                condition=cond,
                episodes=episodes,
                steps_per_ep=steps_per_ep,
            )
            print(f"  -> {r}")
            all_results.append(r)

    # Acceptance.
    c1_pass = bool(unit["all_unit_checks_pass"])
    if args.dry_run:
        # Dry-run: integration skipped from a scientific standpoint. Mark c2
        # as not-evaluated; overall outcome tracks c1 only.
        c2_pass = True
        c2_summary = {
            "per_condition_pass": {},
            "conditions_passed": len(conditions),
            "required_conditions": len(conditions),
            "pass": True,
            "desc": "dry-run: integration eval skipped; c2 set to True",
        }
    else:
        c2_pass, c2_summary = _evaluate_integration(all_results)

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    unit_checks_summary = {
        uc_key: {
            "pass": unit[uc_key]["pass"],
            "desc": unit[uc_key]["desc"],
        }
        for uc_key in ["UC1", "UC2", "UC3", "UC4", "UC5"]
    }

    summary = {
        "c1_unit_checks_all_pass": {
            "unit_checks": unit_checks_summary,
            "all_pass": c1_pass,
            "pass": c1_pass,
            "desc": "UC1 AND UC2 AND UC3 AND UC4 AND UC5 all PASS",
        },
        "c2_integration": c2_summary,
    }

    print(f"\nOutcome: {outcome}")
    print(f"  c1_unit_checks_all_pass: {c1_pass}")
    print(f"  c2_integration: {c2_pass}")
    for uc_key in ["UC1", "UC2", "UC3", "UC4", "UC5"]:
        verdict = "PASS" if unit[uc_key]["pass"] else "FAIL"
        print(f"  {uc_key} ({unit[uc_key]['name']}): {verdict}")

    if outcome == "PASS":
        per_claim = {
            "MECH-261": "supports",
            "SD-032a": "supports",
            "SD-032e": "supports",
        }
    else:
        per_claim = {
            "MECH-261": "weakens",
            "SD-032a": "mixed",
            "SD-032e": "mixed",
        }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "diagnostic",
        "evidence_direction_per_claim": per_claim,
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "unit_checks": unit,
        "config": {
            "conditions": conditions,
            "seeds": seeds,
            "episodes": episodes,
            "steps_per_ep": steps_per_ep,
            "dry_run": bool(args.dry_run),
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Output written to: {out_file}")


if __name__ == "__main__":
    main()
