"""V3-EXQ-464b (EXP-0160 behavioural): MECH-266 competing goals / switch-cost asymmetry.

Purpose: evidence. Full-agent-loop behavioural successor to the substrate-readiness
diagnostic V3-EXQ-464 (which exercised the per-mode enter/exit Schmitt rails on a
fixed synthetic competing-goals signal sequence). This arm exercises the MECH-266
asymmetric per-mode hysteresis inside the real committed_mode_curriculum
(P0 warmup -> P1 consolidation -> P2 eval) on a CausalGridWorldV2 with the GAP-3
DUAL-CUE primitive ON (SD-049 multi-resource heterogeneity required), measuring the
ocd4 "competing goals" switch-cost asymmetry in a live loop.

The MECH-266 prediction: a per-mode exit rail that makes external_task STICKY
(exit_threshold near 0) raises occupancy of external_task and suppresses mode
switches relative to the symmetric MECH-259 baseline (empty rails) -- it is
harder to switch OUT of the sticky mode. The symmetric baseline IS the legacy
MECH-259 configuration (no per-mode rails) and switches modes freely.

Arms (one P0->P1 training run, three instrumented P2 evals on cloned agents):

  ARM_SYMMETRIC        -- legacy MECH-259 (no per-mode enter/exit rails). The
                          baseline the asymmetric arm is measured against.
                          Coordinator switches modes freely. Reference arm.
  ARM_ASYM_STICKY_TASK -- per-mode exit rails: external_task exit=0.05 (sticky),
                          all other modes exit=0.90 (loose). OCD-axis over-binding
                          on the task mode. Positive arm.
  ARM_FORCED_RV_ON     -- O-2 mandatory contrast (committed_mode_curriculum):
                          clone_trained_agent(bistable=True) with running_variance
                          forced to 0.001, asymmetric rails applied. Isolates
                          whether the switch-cost asymmetry needs EMERGENT
                          commitment or merely the committed state.

Pre-registered acceptance (PASS = majority 2/3 seeds with all criteria):
  C1  asymmetry detectable: ARM_ASYM_STICKY_TASK fraction_in_external_task
      >= ARM_SYMMETRIC fraction_in_external_task + C1_OCCUPANCY_MARGIN (0.10)
      (the sticky exit rail raises task-mode occupancy).
  C2  switch-cost asymmetry: ARM_ASYM_STICKY_TASK n_switches
      <= ARM_SYMMETRIC n_switches (over-binding suppresses switches out of
      the sticky mode).
  C3  symmetric baseline non-vacuous (legacy MECH-259 actually switches):
      ARM_SYMMETRIC n_switches >= C3_MIN_SWITCHES (1) -- so the C1/C2 contrast
      is meaningful, not measured against a degenerate no-switch baseline.

Interpretation grid (one row per plausible outcome -> next action):
  C1-C3 all PASS .......... MECH-266 switch-cost asymmetry confirmed in a live
                            competing-goals loop. -> governance: behavioural
                            support for MECH-266 / SD-032a.
  C3 FAIL (sym never switches) the live loop does not drive the coordinator out
                            of external_task at all -> diagnose dual-cue ->
                            dACC -> coordinator salience path (mirror the
                            V3-EXQ-467b no-switching grid row); /diagnose-errors,
                            NOT a re-run under 464b.
  C3 PASS, C1 FAIL ........ the sticky exit rail does not raise task occupancy ->
                            MECH-266 per-mode keying not load-bearing in the live
                            loop; other signals dominate mode selection ->
                            governance review.
  C1 PASS, C2 FAIL ........ occupancy rises but switch count does not fall (sticky
                            mode entered more often without being harder to leave)
                            -> inspect whether the rail is acting on exit vs entry
                            -> /diagnose-errors.

Run:
  /opt/local/bin/python3 experiments/v3_exq_464b_mech266_competing_goals_behavioural.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_464b_mech266_competing_goals_behavioural.py --dry-run
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


EXPERIMENT_TYPE = "v3_exq_464b_mech266_competing_goals_behavioural"
QUEUE_ID = "V3-EXQ-464b"
CLAIM_IDS = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]

# Coordinator mode names (SD-032a DEFAULT_MODE_NAMES order).
MODE_NAMES = [
    "external_task",
    "internal_planning",
    "internal_replay",
    "offline_consolidation",
]
STICKY_MODE = "external_task"
STICKY_EXIT = 0.05   # sticky exit rail on the task mode (OCD-axis over-binding)
LOOSE_EXIT = 0.90    # loose exit rail on all other modes

# Pre-registered thresholds (constants, not derived from the run).
C1_OCCUPANCY_MARGIN = 0.10   # asym task-occupancy must exceed sym by this much
C3_MIN_SWITCHES = 1          # symmetric baseline must switch at least this many times
PASS_FRACTION_REQUIRED = 2.0 / 3.0  # majority of seeds


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_easy_env(size: int) -> CausalGridWorldV2:
    """P0 warmup env: SD-049 enabled (obs_dim parity), no dual_cue, few hazards."""
    return CausalGridWorldV2(
        size=size,
        num_hazards=2,
        num_resources=4,
        num_waypoints=2,
        resource_respawn_on_consume=True,
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
        dual_cue_enabled=False,
    )


def _build_target_env(size: int) -> CausalGridWorldV2:
    """P1 consolidation + P2 eval env: SD-049 with GAP-3 dual-cue ON.

    Two resource types are simultaneously active -- the competing-goals
    substrate. dual_cue raises ValueError if SD-049 is not enabled (Q-3a
    fail-fast).
    """
    return CausalGridWorldV2(
        size=size,
        num_hazards=3,
        num_resources=5,
        num_waypoints=2,
        resource_respawn_on_consume=True,
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


def _apply_symmetric(coord) -> None:
    """ARM_SYMMETRIC: legacy MECH-259 -- ensure no per-mode rails are active."""
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}


def _apply_asymmetric_sticky_task(coord) -> None:
    """ARM_ASYM_STICKY_TASK: sticky exit rail on external_task, loose elsewhere."""
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}
    for mode in MODE_NAMES:
        coord.set_exit_threshold(mode, STICKY_EXIT if mode == STICKY_MODE else LOOSE_EXIT)


def _eval_competing_goals(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_ep: int,
) -> dict:
    """Frozen-policy eval instrumented for MECH-266 competing-goals measurement.

    Tracks coordinator mode occupancy + mode switches across the eval.
    The per-mode rails must already be applied to agent.salience by the caller.
    Returns: fraction_in_external_task, n_switches, mean_dwell, total_steps.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience  # attribute is agent.salience per REEAgent.__init__

    mode_step_counts = {m: 0 for m in MODE_NAMES}
    other_mode_steps = 0
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

                cur_mode = coord.current_mode
                if cur_mode in mode_step_counts:
                    mode_step_counts[cur_mode] += 1
                else:
                    other_mode_steps += 1

                if cur_mode != prev_mode:
                    all_run_lengths.append(current_run)
                    total_switches += 1
                    current_run = 1
                    prev_mode = cur_mode
                else:
                    current_run += 1

                total_steps += 1
                _, _, done, _, obs_dict = env.step(action_idx)
                if done:
                    all_run_lengths.append(current_run)
                    current_run = 0
                    break

            if current_run > 0:
                all_run_lengths.append(current_run)

    frac_task = (
        mode_step_counts[STICKY_MODE] / total_steps if total_steps else 0.0
    )
    mean_dwell = (
        float(sum(all_run_lengths)) / len(all_run_lengths)
        if all_run_lengths else float(steps_per_ep)
    )
    return {
        "fraction_in_external_task": round(frac_task, 4),
        "n_switches": total_switches,
        "mean_dwell": round(mean_dwell, 3),
        "n_runs": len(all_run_lengths),
        "mode_step_counts": mode_step_counts,
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

    world_obs_dim = target_env.world_obs_dim
    body_obs_dim = 12  # default; limb_damage disabled.

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

    # -- ARM_SYMMETRIC (legacy MECH-259) --
    print(f"Seed {seed} Condition ARM_SYMMETRIC", flush=True)
    agent_sym = clone_trained_agent(agent, bistable=True, device=device)
    _apply_symmetric(agent_sym.salience)
    arm_sym = _eval_competing_goals(
        agent_sym, target_env, device, eval_eps, steps_per_ep
    )
    print(
        f"verdict: frac_task={arm_sym['fraction_in_external_task']}"
        f" n_switches={arm_sym['n_switches']}",
        flush=True,
    )

    # -- ARM_ASYM_STICKY_TASK --
    print(f"Seed {seed} Condition ARM_ASYM_STICKY_TASK", flush=True)
    agent_asym = clone_trained_agent(agent, bistable=True, device=device)
    _apply_asymmetric_sticky_task(agent_asym.salience)
    arm_asym = _eval_competing_goals(
        agent_asym, target_env, device, eval_eps, steps_per_ep
    )
    print(
        f"verdict: frac_task={arm_asym['fraction_in_external_task']}"
        f" n_switches={arm_asym['n_switches']}",
        flush=True,
    )

    # -- ARM_FORCED_RV_ON (O-2 mandatory contrast) --
    print(f"Seed {seed} Condition ARM_FORCED_RV_ON", flush=True)
    agent_forced = clone_trained_agent(agent, bistable=True, device=device)
    agent_forced.e3._running_variance = 0.001
    _apply_asymmetric_sticky_task(agent_forced.salience)
    arm_forced = _eval_competing_goals(
        agent_forced, target_env, device, eval_eps, steps_per_ep
    )
    print(
        f"verdict: frac_task={arm_forced['fraction_in_external_task']}"
        f" n_switches={arm_forced['n_switches']}",
        flush=True,
    )

    # -- Criteria evaluation --
    c1 = (
        arm_asym["fraction_in_external_task"]
        >= arm_sym["fraction_in_external_task"] + C1_OCCUPANCY_MARGIN
    )
    c2 = arm_asym["n_switches"] <= arm_sym["n_switches"]
    c3 = arm_sym["n_switches"] >= C3_MIN_SWITCHES
    seed_pass = bool(c1 and c2 and c3)

    print(
        f"verdict: {'PASS' if seed_pass else 'FAIL'} C1={c1} C2={c2} C3={c3}",
        flush=True,
    )
    return {
        "seed": seed,
        "ARM_SYMMETRIC": arm_sym,
        "ARM_ASYM_STICKY_TASK": arm_asym,
        "ARM_FORCED_RV_ON": arm_forced,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
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
            "C1_occupancy_margin": C1_OCCUPANCY_MARGIN,
            "C3_min_switches": C3_MIN_SWITCHES,
            "sticky_exit": STICKY_EXIT,
            "loose_exit": LOOSE_EXIT,
            "sticky_mode": STICKY_MODE,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": n_pass,
        "smoke": smoke,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "V3-EXQ-464b behavioural successor to the V3-EXQ-464 substrate-"
            "readiness diagnostic (which is NOT superseded -- its UC1-UC5 "
            "Schmitt-rail arithmetic evidence on synthetic signal sequences "
            "stands). This arm validates the ocd4 competing-goals switch-cost "
            "asymmetry in a live committed_mode_curriculum loop on "
            "CausalGridWorldV2 with the GAP-3 dual-cue primitive ON (SD-049 "
            "multi-resource heterogeneity required): a per-mode sticky exit "
            "rail on external_task raises task-mode occupancy and suppresses "
            "mode switches relative to the symmetric MECH-259 baseline (empty "
            "rails). O-2 forced-rv contrast included per the GAP-11 "
            "committed_mode_curriculum mandatory-contrast rule. Plan: "
            "commitment_closure_plan.md GAP-4 Phase 4/5 cohort."
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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
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
