#!/opt/local/bin/python3
"""
V3-EXQ-730 -- Q-080.a: does effort->harm coupling add protective disengagement?

Sub-question (Q-080.a)
----------------------
When chronic physical effort is routed into the affective harm stream (z_harm_a)
via the env's effort_harm_coupling, does the agent gain (1) PROTECTIVE
DISENGAGEMENT -- it abandons a costly-but-committed path before energy depletes --
and (2) IRREVERSIBILITY-AWARE caution -- its disengagement tracks the IRREVERSIBLE
exertion component (exertion_permanent) more than the recoverable load
(exertion_load), which a value+allostatic baseline (dACC effort-cost + pACC/AIC
suffering accumulator) alone does NOT provide?

Design
------
Two arms differ ONLY in an ENV knob:
  OFF: effort_harm_coupling_enabled=False   (effort never enters z_harm_a)
  ON:  effort_harm_coupling_enabled=True    (effort injected into harm_obs_a)
BOTH arms share the SAME agent config with the full value+allostatic machinery
ON (use_dacc, use_pacc_analog, use_aic_analog, use_harm_suffering_accumulator,
harm_suffering_redirect_aic). No experiment-level selection override: the agent
selects with its FULL machinery, so MECH-091's urgency interrupt fires on z_harm_a
and coupling-ON routes chronic effort into that same path. The dACC/pACC/AIC/
MECH-219 readouts are non-trainable arithmetic over latent norms and run
forward-only, identical across arms -- so the contrast is clean without any
phased encoder training.

Commitment probe: each episode teleports the agent to row 2 of the HIGH-effort
corridor (just inside it, committed to the costly path). We then run
STEPS_PER_EP steps of normal agent selection and watch whether the agent
DISENGAGES (turns down / exits the high corridor) and whether it does so BEFORE
energy floors.

Evidence directions (BOTH stated per repo policy)
-------------------------------------------------
SUPPORTS a harm-coupling mechanism (Q-080.a): the ON arm disengages more, and
  earlier (before depletion), than OFF, AND its per-step disengagement tracks
  exertion_permanent (irreversible) more than exertion_load (recoverable) --
  i.e. coupling adds behaviour the value+allostatic baseline lacks.
REFUTES / weakens: OFF already disengages just as much (the dACC effort-cost +
  pACC/AIC suffering machinery already yields protective disengagement); then
  effort is adequately captured as FOREGONE VALUE and no separate harm-coupling
  mechanism need be minted.

Lit anchors: Hogan 2020 (effort as an aversive interoceptive signal);
McMorris 2018 (central-fatigue / effort accumulation and disengagement);
Pizzolla 2026 (irreversible physiological cost and protective withdrawal).
Env spec: REE_assembly/docs/architecture/effort_dissociation_env.md.

experiment_purpose: "diagnostic" -- this arbitrates WHETHER a harm-coupling
mechanism is warranted vs. effort-as-foregone-value; it does not press a
promotion.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import p0_readiness_gate, P0NotReady  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_730_q080a_effort_harm_coupling"
CLAIM_IDS = ["Q-080"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 200
EPISODES = 8

GRID_SIZE = 11
# Lower the irreversible-ratchet mark from the 0.5 default so exertion_permanent
# actually accrues within a STEPS_PER_EP=200 chronic episode -- otherwise the
# C2 irreversibility-tracking slope has no permanent-component variance to read.
# Both arms share this env config, so the ON-vs-OFF contrast stays clean.
RATCHET_MARK = 0.1

# Criterion thresholds.
C1_DISENGAGE_MARGIN = 0.10      # ON disengagement_rate must exceed OFF by this
C1_SEED_MAJORITY = 2            # of 3 seeds
C2_SEED_MAJORITY = 2
ENERGY_FLOOR = 0.3              # "before depletion" boundary for disengage-before
OFF_DISENGAGE_HIGH = 0.30       # OFF already disengages if mean rate exceeds this
ON_ADDS_NEGLIGIBLE = 0.05       # ON adds < this over OFF -> baseline suffices

# Readiness (non-degeneracy) floors.
READY_COST_DIFFERENTIAL = 0.005  # mean(high effort cost) - mean(low) must exceed
READY_EXERTION_ACCRUED = 0.05    # max exertion_load in probe must exceed


def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
    harm_hist = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
    return body, world, harm, harm_a, harm_hist


def _make_env(seed: int, coupling: bool) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=0,
        num_resources=1,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        energy_decay=0.005,
        effort_dissociation_enabled=True,
        effort_exertion_ratchet_mark=RATCHET_MARK,
        effort_harm_coupling_enabled=coupling,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
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
        z_harm_a_dim=16,
        # value + allostatic machinery ON in BOTH arms
        use_dacc=True,
        dacc_weight=2.0,
        dacc_effort_cost=0.1,
        use_pacc_analog=True,
        pacc_drive_alpha=0.05,
        use_aic_analog=True,
        use_harm_suffering_accumulator=True,
        harm_suffering_redirect_aic=True,
    )
    # MECH-219 suffering accumulator requires the SD-019a unpleasantness channel
    # (z_harm_un) as its drive input; use_harm_un is a LatentStackConfig field,
    # not a from_dims kwarg, so set it after construction.
    cfg.latent.use_harm_un = True
    return REEAgent(cfg)


def _teleport_into_high_corridor(env: CausalGridWorldV2) -> None:
    """Place the agent at row 2 of the HIGH-effort corridor (committed to it)."""
    env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["empty"]
    env.agent_x, env.agent_y = 2, env._effort_high_col
    env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["agent"]


def _slope(d: List[float], x: List[float]) -> float:
    """Simple OLS slope of d on x; 0.0 when x has no variance."""
    if len(d) < 2:
        return 0.0
    d_arr = np.asarray(d, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    var_x = float(np.var(x_arr))
    if var_x <= 1e-12:
        return 0.0
    cov = float(np.mean((d_arr - d_arr.mean()) * (x_arr - x_arr.mean())))
    return cov / var_x


def _run_cell(seed: int, arm: str, coupling: bool, episodes: int,
              steps_per_ep: int) -> Dict[str, Any]:
    """One (seed x arm) cell. arm in {"OFF", "ON"}."""
    print(f"Seed {seed} Condition {arm}", flush=True)

    config_slice = {
        "arm": arm,
        "effort_harm_coupling_enabled": coupling,
        "size": GRID_SIZE,
        "num_hazards": 0,
        "num_resources": 1,
        "energy_decay": 0.005,
        "effort_dissociation_enabled": True,
        "effort_exertion_ratchet_mark": RATCHET_MARK,
        "use_dacc": True,
        "dacc_weight": 2.0,
        "dacc_effort_cost": 0.1,
        "use_pacc_analog": True,
        "pacc_drive_alpha": 0.05,
        "use_aic_analog": True,
        "use_harm_suffering_accumulator": True,
        "harm_suffering_redirect_aic": True,
        "steps_per_ep": steps_per_ep,
        "episodes": episodes,
    }

    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        env = _make_env(seed, coupling)
        agent = _make_agent(env)

        # Per-step observations (only over steps where exit from HIGH is possible).
        disengage_flags: List[float] = []
        exertion_load_at: List[float] = []
        exertion_perm_at: List[float] = []

        high_steps = 0
        total_steps = 0
        exertion_permanent_peak = 0.0
        exertion_load_peak = 0.0
        min_energy = 1.0
        disengage_before_depletion = False

        for ep in range(episodes):
            agent.reset()
            _obs, obs_dict = env.reset()
            _teleport_into_high_corridor(env)
            # Rebuild obs_dict for the new position without stepping dynamics.
            obs_dict = env._get_observation_dict()

            if ep % 2 == 0 or ep == episodes - 1:
                print(f"  [train] seed={seed} arm={arm} ep {ep+1}/{episodes}",
                      flush=True)

            for step in range(steps_per_ep):
                body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
                latent = agent.sense(
                    obs_body=body,
                    obs_world=world,
                    obs_harm=harm,
                    obs_harm_a=harm_a,
                    obs_harm_history=harm_hist,
                )
                ticks = agent.clock.advance()
                world_dim_local = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, world_dim_local, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                a_idx = int(action[0].argmax().item())

                pre_corridor = int(obs_dict["effort_corridor"]) if "effort_corridor" in obs_dict \
                    else int(env._effort_corridor_at(env.agent_x, env.agent_y))
                load_pre = float(env._exertion_load)
                perm_pre = float(env._exertion_permanent)
                energy_pre = float(env.agent_energy)

                _obs, harm_signal, done, info, obs_dict = env.step(action)

                post_corridor = int(info["effort_corridor"])
                # action 1 == +x == "down"/away from the top resource == exit move.
                exited = (post_corridor != 2)
                is_disengage = (a_idx == 1) or exited

                total_steps += 1
                if post_corridor == 2 or pre_corridor == 2:
                    if post_corridor == 2:
                        high_steps += 1
                # Only steps where the agent is IN the high corridor and can exit.
                if pre_corridor == 2:
                    disengage_flags.append(1.0 if is_disengage else 0.0)
                    exertion_load_at.append(load_pre)
                    exertion_perm_at.append(perm_pre)
                    if exited and energy_pre > ENERGY_FLOOR:
                        disengage_before_depletion = True

                exertion_permanent_peak = max(exertion_permanent_peak,
                                              float(info["exertion_permanent"]))
                exertion_load_peak = max(exertion_load_peak,
                                         float(info["exertion_load"]))
                min_energy = min(min_energy, float(info["energy"]))

                if done:
                    break

        n_high_pre = len(disengage_flags)
        disengagement_rate = (sum(disengage_flags) / n_high_pre) if n_high_pre else 0.0
        mean_high_corridor_occupancy = (high_steps / total_steps) if total_steps else 0.0
        slope_vs_permanent = _slope(disengage_flags, exertion_perm_at)
        slope_vs_load = _slope(disengage_flags, exertion_load_at)

        row = {
            "seed": seed,
            "arm": arm,
            "effort_harm_coupling_enabled": coupling,
            "disengagement_rate": float(disengagement_rate),
            "mean_high_corridor_occupancy": float(mean_high_corridor_occupancy),
            "disengage_before_depletion": bool(disengage_before_depletion),
            "exertion_permanent_peak": float(exertion_permanent_peak),
            "exertion_load_peak": float(exertion_load_peak),
            "min_energy": float(min_energy),
            "slope_disengage_vs_permanent": float(slope_vs_permanent),
            "slope_disengage_vs_load": float(slope_vs_load),
            "n_high_corridor_steps": int(n_high_pre),
            "total_steps": int(total_steps),
            # NOTE: MECH-091 urgency-interrupt fire-count is not cleanly exposed
            # as a per-tick counter; we use corridor-exit events as the
            # disengagement proxy (see disengagement_rate), which is the
            # behavioural quantity the criterion cares about.
            "urgency_interrupt_proxy": "corridor_exit_events",
        }
        cell.stamp(row)

    verdict = "PASS" if (disengagement_rate > 0.0 or slope_vs_permanent != 0.0) else "FAIL"
    print(
        f"  [seed={seed} {arm}] disengage_rate={disengagement_rate:.3f} "
        f"occ_high={mean_high_corridor_occupancy:.3f} "
        f"before_depletion={disengage_before_depletion} "
        f"slope_perm={slope_vs_permanent:.4f} slope_load={slope_vs_load:.4f} "
        f"verdict: {verdict}",
        flush=True,
    )
    return row


def _readiness_probe(steps: int = 6) -> Dict[str, float]:
    """Walk a fresh env down each corridor; measure cost differential + accrual.

    Mirrors the effort-env smoke: teleport to the top of each corridor, step
    'down' (action 1), and record per-step effort_cost_this_step + exertion_load.
    """
    high_costs: List[float] = []
    low_costs: List[float] = []
    max_load = 0.0
    for col_kind in ("low", "high"):
        env = _make_env(seed=1, coupling=False)
        env.reset()
        col = env._effort_low_col if col_kind == "low" else env._effort_high_col
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["empty"]
        env.agent_x, env.agent_y = 2, col
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["agent"]
        env._reset_effort_state()
        env.agent_energy = 1.0
        for _ in range(steps):
            _obs, _hs, _done, info, _od = env.step(torch.tensor(1))
            if col_kind == "high":
                high_costs.append(float(info["effort_cost_this_step"]))
            else:
                low_costs.append(float(info["effort_cost_this_step"]))
            max_load = max(max_load, float(info["exertion_load"]))
    mean_high = float(np.mean(high_costs)) if high_costs else 0.0
    mean_low = float(np.mean(low_costs)) if low_costs else 0.0
    return {
        "energy_cost_differential": mean_high - mean_low,
        "exertion_accrued": max_load,
    }


def run_experiment(dry_run: bool) -> Dict[str, Any]:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    seeds = [42] if dry_run else SEEDS
    episodes = 2 if dry_run else EPISODES
    steps_per_ep = 20 if dry_run else STEPS_PER_EP
    arms = [("ON", True)] if dry_run else [("OFF", False), ("ON", True)]

    # -- P0 readiness (non-degeneracy) --------------------------------------
    probe = _readiness_probe(steps=(3 if dry_run else 6))
    checks = [
        {"name": "energy_cost_differential",
         "measured": probe["energy_cost_differential"],
         "threshold": READY_COST_DIFFERENTIAL, "direction": "lower"},
        {"name": "exertion_accrued",
         "measured": probe["exertion_accrued"],
         "threshold": READY_EXERTION_ACCRUED, "direction": "lower"},
    ]
    ready = True
    try:
        preconditions = p0_readiness_gate(checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False

    if not ready and not dry_run:
        return {
            "run_id": run_id,
            "experiment_type": EXPERIMENT_TYPE,
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "timestamp_utc": ts,
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": preconditions,
                "criteria_non_degenerate": {
                    "C1_arms_differ": False,
                    "C2_exertion_varies": False,
                },
            },
            "criteria": [
                {"name": "C1_protective_disengagement_added",
                 "load_bearing": True, "passed": False},
                {"name": "C2_irreversibility_aware", "load_bearing": False,
                 "passed": False},
            ],
            "readiness_probe": probe,
            "arm_results": [],
        }

    # -- Arms ---------------------------------------------------------------
    arm_results = []
    for arm, coupling in arms:
        for seed in seeds:
            row = _run_cell(seed, arm, coupling, episodes, steps_per_ep)
            arm_results.append(row)

    if dry_run:
        print("Smoke test PASSED", flush=True)
        return {
            "run_id": run_id,
            "experiment_type": EXPERIMENT_TYPE,
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "timestamp_utc": ts,
            "outcome": "FAIL",
            "evidence_direction": "unknown",
            "interpretation": {
                "label": "inconclusive",
                "preconditions": preconditions,
                "criteria_non_degenerate": {
                    "C1_arms_differ": False,
                    "C2_exertion_varies": any(
                        r["exertion_permanent_peak"] > 0 for r in arm_results),
                },
            },
            "criteria": [
                {"name": "C1_protective_disengagement_added",
                 "load_bearing": True, "passed": False},
                {"name": "C2_irreversibility_aware", "load_bearing": False,
                 "passed": False},
            ],
            "readiness_probe": probe,
            "arm_results": arm_results,
            "dry_run": True,
        }

    # -- Adjudication (real run) --------------------------------------------
    off = {r["seed"]: r for r in arm_results if r["arm"] == "OFF"}
    on = {r["seed"]: r for r in arm_results if r["arm"] == "ON"}

    # C1: ON disengages more than OFF by margin AND ON disengages-before-depletion
    #     where OFF does not -- in a seed majority.
    c1_seed_hits = 0
    for s in seeds:
        o, n = off.get(s), on.get(s)
        if o is None or n is None:
            continue
        margin_ok = (n["disengagement_rate"] - o["disengagement_rate"]) >= C1_DISENGAGE_MARGIN
        before_ok = n["disengage_before_depletion"] and (not o["disengage_before_depletion"])
        if margin_ok and before_ok:
            c1_seed_hits += 1
    c1 = c1_seed_hits >= C1_SEED_MAJORITY

    # C2: ON slope(disengage vs permanent) > slope(disengage vs load), seed majority.
    c2_seed_hits = 0
    for s in seeds:
        n = on.get(s)
        if n is None:
            continue
        if n["slope_disengage_vs_permanent"] > n["slope_disengage_vs_load"]:
            c2_seed_hits += 1
    c2 = c2_seed_hits >= C2_SEED_MAJORITY

    mean_off_rate = float(np.mean([off[s]["disengagement_rate"] for s in seeds if s in off])) if off else 0.0
    mean_on_rate = float(np.mean([on[s]["disengagement_rate"] for s in seeds if s in on])) if on else 0.0
    off_already_disengages = (mean_off_rate > OFF_DISENGAGE_HIGH) and \
        ((mean_on_rate - mean_off_rate) < ON_ADDS_NEGLIGIBLE)

    # Non-degeneracy: arms must actually differ on occupancy, exertion must vary.
    mean_off_occ = float(np.mean([off[s]["mean_high_corridor_occupancy"] for s in seeds if s in off])) if off else 0.0
    mean_on_occ = float(np.mean([on[s]["mean_high_corridor_occupancy"] for s in seeds if s in on])) if on else 0.0
    c1_arms_differ = abs(mean_off_occ - mean_on_occ) > 1e-9
    max_perm_peak = max((r["exertion_permanent_peak"] for r in arm_results), default=0.0)
    c2_exertion_varies = max_perm_peak > 0.0

    if c1 and c2:
        label = "harm_coupling_adds_protective_disengagement"
        evidence_direction = "supports"
        outcome = "PASS"
    elif off_already_disengages:
        label = "value_allostatic_suffices_effort_is_foregone_value"
        evidence_direction = "weakens"
        outcome = "PASS"
    else:
        label = "inconclusive"
        evidence_direction = "unknown"
        outcome = "FAIL"

    print(
        f"\nFINAL: {outcome}  label={label} "
        f"C1={c1} (hits={c1_seed_hits}) C2={c2} (hits={c2_seed_hits}) "
        f"off_rate={mean_off_rate:.3f} on_rate={mean_on_rate:.3f}",
        flush=True,
    )

    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"Q-080": evidence_direction},
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1_arms_differ": bool(c1_arms_differ),
                "C2_exertion_varies": bool(c2_exertion_varies),
            },
        },
        "criteria": [
            {"name": "C1_protective_disengagement_added", "load_bearing": True,
             "passed": bool(c1), "seed_hits": c1_seed_hits,
             "desc": "ON disengagement_rate > OFF by >=0.1 AND ON disengages before "
                     "depletion where OFF does not, in >=2/3 seeds"},
            {"name": "C2_irreversibility_aware", "load_bearing": False,
             "passed": bool(c2), "seed_hits": c2_seed_hits,
             "desc": "ON slope(disengage vs exertion_permanent) > slope vs "
                     "exertion_load in >=2/3 seeds"},
        ],
        "metrics": {
            "mean_off_disengagement_rate": mean_off_rate,
            "mean_on_disengagement_rate": mean_on_rate,
            "mean_off_high_occupancy": mean_off_occ,
            "mean_on_high_occupancy": mean_on_occ,
            "max_exertion_permanent_peak": float(max_perm_peak),
            "off_already_disengages": bool(off_already_disengages),
        },
        "readiness_probe": probe,
        "config": {
            "seeds": SEEDS, "episodes": EPISODES, "steps_per_ep": STEPS_PER_EP,
            "grid_size": GRID_SIZE, "ratchet_mark": RATCHET_MARK,
            "c1_disengage_margin": C1_DISENGAGE_MARGIN,
            "energy_floor": ENERGY_FLOOR,
        },
        "arm_results": arm_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== {EXPERIMENT_TYPE} ===", flush=True)
    print(f"Claim: {CLAIM_IDS}  purpose: {EXPERIMENT_PURPOSE}", flush=True)
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL RUN'}", flush=True)

    result = run_experiment(dry_run=args.dry_run)

    script_dir = Path(__file__).resolve().parents[1]
    out_dir = (script_dir.parent / "REE_assembly" / "evidence"
               / "experiments" / EXPERIMENT_TYPE)
    out_file = write_flat_manifest(
        result,
        out_dir,
        dry_run=False,
        config=result.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Output written to: {out_file}", flush=True)
    print(f"Outcome: {result['outcome']}  "
          f"evidence_direction: {result['evidence_direction']}", flush=True)

    return out_file, result["outcome"], args.dry_run


if __name__ == "__main__":
    _out_file, _outcome, _dry = main()
    # Reached on every manifest-writing path (readiness-abort, dry-run, and full
    # adjudication all return through main()).
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_file,
        dry_run=_dry,
    )
