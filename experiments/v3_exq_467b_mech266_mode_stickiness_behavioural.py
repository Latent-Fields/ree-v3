"""V3-EXQ-467b (EXP-0163 behavioural): MECH-266 mode stickiness / hold decay.

Purpose: evidence. Full-agent-loop behavioural dose-response measuring MODE-DWELL
TIME across the MECH-266 asymmetric hysteresis sweep (r in [0.10, 0.50, 1.00, 1.50,
2.00]) in a live competing-goals loop on CausalGridWorldV2 with the GAP-3 dual-cue
primitive ON (SD-049 multi-resource heterogeneity required). Uses the
committed_mode_curriculum (P0 warmup -> P1 consolidation) to establish a trained
agent, then runs five instrumented frozen-policy P2 evals -- one per ratio arm --
measuring how long the SalienceCoordinator stays in the dominant mode before
switching.

Prediction: mode-dwell time is MONOTONE in stickiness -- low r (over-binding, OCD
axis) -> long dwell / stuck-in-mode; high r (under-binding, depression/ADHD axis)
-> short dwell / readily flips. This dose-response distinguishes the two ends of
the MECH-266 hysteresis axis in an ecologically-valid agent loop.

Five-arm sweep:
  r=0.10  over-binding severe  (OCD axis)
  r=0.50  over-binding moderate
  r=1.00  symmetric baseline
  r=1.50  under-binding mild
  r=2.00  under-binding severe (depression/ADHD axis)

Pre-registered acceptance (PASS = majority 2/3 seeds with all criteria):
  C1  mode-dwell time is monotone non-increasing across r sweep within seed
  C2  dwell(r=0.10) >= C2_MIN_DWELL_RATIO * dwell(r=2.00) -- material effect size
  C3  coordinator actually switches modes in r=2.00 arm (n_switches >= C3_MIN)

Interpretation grid:
  All C1-C3 PASS ......... MECH-266 dose-response confirmed in live loop.
                           -> governance: behavioural support MECH-266 / SD-032a.
  No switching at any r .. diagnose dual-cue wiring / dACC -> coordinator signal;
                           coordinator may not receive sufficient salience from dACC.
                           -> /diagnose-errors before re-queue.
  Flat dwell across r .... MECH-266 hysteresis not load-bearing in live agent loop;
                           other signals dominate mode selection. -> governance review.

Run:
  /opt/local/bin/python3 experiments/v3_exq_467b_mech266_mode_stickiness_behavioural.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_467b_mech266_mode_stickiness_behavioural.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.committed_mode_curriculum import (  # noqa: E402
    clone_trained_agent,
    run_p0_warmup,
    run_p1_consolidation,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_467b_mech266_mode_stickiness_behavioural"
QUEUE_ID = "V3-EXQ-467b"
CLAIM_IDS = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]

# Hysteresis ratio sweep (5 arms, matches V3-EXQ-467 substrate diagnostic).
RATIOS = [0.10, 0.50, 1.00, 1.50, 2.00]

# Pre-registered thresholds (constants, not derived from the run).
# C2: lowest-r arm dwell must be >= this multiple of highest-r arm dwell.
C2_MIN_DWELL_RATIO = 2.0
# C3: high-r arm (r=2.00) must have at least this many mode switches across eval.
C3_MIN_SWITCHES = 1
# Majority rule: 2 of 3 seeds must PASS all three criteria.
PASS_FRACTION_REQUIRED = 2.0 / 3.0


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_easy_env(size: int) -> CausalGridWorldV2:
    """P0 warmup env: SD-049 enabled (obs_dim match), no dual_cue, few hazards."""
    return CausalGridWorldV2(
        size=size,
        num_hazards=2,
        num_resources=4,
        num_waypoints=2,
        resource_respawn_on_consume=True,
        # SD-049 multi-resource heterogeneity -- required for obs_dim parity with target env.
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=3,
        resource_type_names=("food", "water", "novelty"),
        resource_type_drive_axes=("hunger", "thirst", "curiosity"),
        resource_type_benefit_curves=(
            "sigmoidal_saturating",
            "sharp_saturation",
            "novelty_decay",
        ),
        per_axis_drive_enabled=False,
        # dual_cue OFF for P0 warmup (simpler env; obs_dim stays the same).
        dual_cue_enabled=False,
    )


def _build_target_env(size: int) -> CausalGridWorldV2:
    """P1 consolidation + P2 eval env: SD-049 with GAP-3 dual-cue ON."""
    return CausalGridWorldV2(
        size=size,
        num_hazards=3,
        num_resources=5,
        num_waypoints=2,
        resource_respawn_on_consume=True,
        # SD-049 multi-resource heterogeneity (required; dual_cue raises ValueError
        # if this is False -- Q-3a fail-fast).
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=3,
        resource_type_names=("food", "water", "novelty"),
        resource_type_drive_axes=("hunger", "thirst", "curiosity"),
        resource_type_benefit_curves=(
            "sigmoidal_saturating",
            "sharp_saturation",
            "novelty_decay",
        ),
        per_axis_drive_enabled=False,
        # GAP-3 dual-cue primitive ON -- two resource types simultaneously active.
        dual_cue_enabled=True,
        dual_cue_min_active_ticks=10,
        dual_cue_replace_on_early_consume=False,
        dual_cue_type_tags=(1, 2),
    )


def _build_agent(world_obs_dim: int, body_obs_dim: int) -> REEAgent:
    """Build agent with salience coordinator + dACC + lateral PFC."""
    cfg = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
    )
    cfg.heartbeat.beta_gate_bistable = True
    return REEAgent(cfg)


def _eval_mode_dwell(
    agent: REEAgent,
    env: CausalGridWorldV2,
    ratio: float,
    device: torch.device,
    n_eps: int,
    steps_per_ep: int,
) -> dict:
    """Frozen-policy eval instrumented for MECH-266 mode-dwell measurement.

    Applies set_hysteresis_ratio(ratio) to the coordinator before running.
    Tracks consecutive ticks in the dominant mode (runs) and counts switches.
    Returns per-condition metrics: mean_dwell, n_switches, n_runs, total_steps.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience  # attribute is agent.salience per REEAgent.__init__

    # Apply MECH-266 hysteresis ratio for this condition arm.
    coord.set_hysteresis_ratio(ratio)

    all_run_lengths = []
    total_switches = 0
    total_steps = 0

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()

            prev_mode = coord.current_mode
            current_run = 1

            for _ in range(steps_per_ep):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                new_mode = coord.current_mode
                if new_mode != prev_mode:
                    all_run_lengths.append(current_run)
                    total_switches += 1
                    current_run = 1
                    prev_mode = new_mode
                else:
                    current_run += 1

                total_steps += 1
                _, _, done, _, obs_dict = env.step(action_idx)
                if done:
                    all_run_lengths.append(current_run)
                    current_run = 0
                    break

            # Append final run if episode ended naturally (no done).
            if current_run > 0:
                all_run_lengths.append(current_run)

    mean_dwell = (
        float(sum(all_run_lengths)) / len(all_run_lengths)
        if all_run_lengths else float(steps_per_ep)
    )
    return {
        "ratio": ratio,
        "mean_dwell": round(mean_dwell, 3),
        "n_switches": total_switches,
        "n_runs": len(all_run_lengths),
        "total_steps": total_steps,
        "n_episodes": n_eps,
    }


def run_seed(seed: int, device: torch.device, smoke: bool) -> dict:
    torch.manual_seed(seed)

    size = 8 if smoke else 10
    p0_budget = 3 if smoke else 150
    p1_budget = 3 if smoke else 100
    steps_per_ep = 20 if smoke else 150
    eval_eps = 2 if smoke else 20

    easy_env = _build_easy_env(size)
    target_env = _build_target_env(size)

    # Derive obs dims from target env (both envs have same world_obs_dim via SD-049).
    world_obs_dim = target_env.world_obs_dim
    # body_obs_dim: 12 (default; limb_damage disabled).
    body_obs_dim = 12

    agent = _build_agent(world_obs_dim, body_obs_dim).to(device)

    # -- P0 warmup (easy env, no dual_cue) --
    print(f"Seed {seed} Condition warmup P0", flush=True)
    p0 = run_p0_warmup(
        agent, easy_env, device,
        budget=p0_budget, steps_per_episode=steps_per_ep,
    )
    print(
        f"  [train] ep {p0.n_episodes}/{p0_budget}"
        f" converged={p0.converged} aborted={p0.aborted} rv={p0.final_rv:.5f}",
        flush=True,
    )
    if p0.aborted:
        print("verdict: FAIL", flush=True)
        return {
            "seed": seed,
            "outcome": "commitment_not_elicited",
            "p0_aborted": True,
            "p0_abort_reason": p0.abort_reason,
            "pass": False,
        }

    # -- P1 consolidation (target env with dual_cue) --
    print(f"Seed {seed} Condition consolidation P1", flush=True)
    p1 = run_p1_consolidation(
        agent, target_env, device,
        budget=p1_budget, steps_per_episode=steps_per_ep,
    )
    print(
        f"  [train] ep {p1.n_episodes}/{p1_budget}"
        f" emerged={p1.commitment_emerged}"
        f" committed/ep={p1.final_committed_steps_per_ep:.1f}",
        flush=True,
    )

    # -- Per-condition evaluation: sweep hysteresis ratios --
    # Clone trained agent for each condition to prevent cross-condition state leakage.
    condition_results = []
    for r in RATIOS:
        print(f"Seed {seed} Condition r={r}", flush=True)
        agent_cond = clone_trained_agent(agent, bistable=True, device=device)
        cond_result = _eval_mode_dwell(
            agent_cond, target_env, r, device, eval_eps, steps_per_ep,
        )
        print(
            f"verdict: dwell={cond_result['mean_dwell']}"
            f" n_switches={cond_result['n_switches']}",
            flush=True,
        )
        condition_results.append(cond_result)

    # -- Criteria evaluation --
    dwells = [c["mean_dwell"] for c in condition_results]

    # C1: mean_dwell is monotone non-increasing across r sweep.
    c1 = all(dwells[i] >= dwells[i + 1] for i in range(len(dwells) - 1))

    # C2: lowest-r dwell >= C2_MIN_DWELL_RATIO * highest-r dwell.
    dwell_low = condition_results[0]["mean_dwell"]   # r=0.10
    dwell_high = condition_results[-1]["mean_dwell"]  # r=2.00
    c2 = dwell_low >= C2_MIN_DWELL_RATIO * max(dwell_high, 1.0)

    # C3: coordinator actually switches in the high-r arm.
    c3 = condition_results[-1]["n_switches"] >= C3_MIN_SWITCHES

    seed_pass = bool(c1 and c2 and c3)
    print(
        f"verdict: {'PASS' if seed_pass else 'FAIL'}"
        f" C1={c1} C2={c2} C3={c3}"
        f" dwell_low={dwell_low} dwell_high={dwell_high}",
        flush=True,
    )

    return {
        "seed": seed,
        "conditions": condition_results,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "dwell_per_ratio": {
            str(c["ratio"]): c["mean_dwell"] for c in condition_results
        },
        "p0_final_rv": p0.final_rv,
        "p1_commitment_emerged": p1.commitment_emerged,
        "p1_committed_per_ep": round(p1.final_committed_steps_per_ep, 2),
        "pass": seed_pass,
    }


def build_manifest(seed_results: list, smoke: bool) -> dict:
    n_pass = sum(1 for r in seed_results if r.get("pass"))
    n_seeds = len(seed_results)
    overall_pass = (n_pass / max(1, n_seeds)) >= PASS_FRACTION_REQUIRED
    outcome = "PASS" if overall_pass else "FAIL"
    direction = "supports" if overall_pass else "weakens"
    run_id = f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3"
    return {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_iso(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {cid: direction for cid in CLAIM_IDS},
        "thresholds": {
            "C2_min_dwell_ratio": C2_MIN_DWELL_RATIO,
            "C3_min_switches": C3_MIN_SWITCHES,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
            "ratios_sweep": RATIOS,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": n_pass,
        "smoke": smoke,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "V3-EXQ-467b behavioural arm: MECH-266 mode stickiness / hold decay "
            "dose-response in a live competing-goals loop on CausalGridWorldV2 with "
            "GAP-3 dual-cue primitive ON (SD-049 multi-resource heterogeneity "
            "required). Sweeps MECH-266 hysteresis ratio r across five arms "
            "[0.10, 0.50, 1.00, 1.50, 2.00] and measures coordinator MODE-DWELL "
            "TIME after committed_mode_curriculum P0 warmup + P1 consolidation. "
            "Prediction: monotone non-increasing dwell with r: low r = OCD-axis "
            "over-binding = long dwell (stuck-in-mode); high r = depression/ADHD-"
            "axis under-binding = short dwell (readily flips). Dose-response "
            "distinguishes the two ends of the MECH-266 hysteresis axis in an "
            "ecologically-valid agent loop. Behavioral successor to V3-EXQ-467 "
            "(substrate-only parametric sweep on synthetic signal sequences). "
            "Plan: commitment_closure_plan.md GAP-4 Phase 4/5 OCD cohort."
        ),
    }


def main(smoke: bool):
    device = torch.device("cpu")
    seeds = SEEDS[:1] if smoke else SEEDS
    seed_results = [run_seed(s, device, smoke) for s in seeds]
    manifest = build_manifest(seed_results, smoke)

    print(f"=== {QUEUE_ID} {EXPERIMENT_TYPE} ===", flush=True)
    print(
        f"outcome: {manifest['outcome']}"
        f" ({manifest['n_seeds_pass']}/{manifest['n_seeds']} seeds pass)",
        flush=True,
    )

    if smoke:
        return None

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Smoke run (tiny budgets, no manifest)."
    )
    args = parser.parse_args()
    result = main(smoke=args.dry_run)
    if args.dry_run or result is None:
        sys.exit(0)
    _outcome, _out_path, _run_id = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        queue_id=QUEUE_ID,
        exit_reason="ok" if _outcome == "PASS" else "fail",
    )
    sys.exit(0)
