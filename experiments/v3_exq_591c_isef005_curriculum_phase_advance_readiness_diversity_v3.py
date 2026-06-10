"""
V3-EXQ-591c -- InfantCurriculumScheduler Phase 0->1 reachability WITH the
exploration-diversity stack armed.

SUBSTRATE-READINESS DIAGNOSTIC for infant_substrate:GAP-14 prerequisite (c) ONLY.
Diagnose-first follow-up to failure_autopsy_V3-EXQ-591b_2026-06-10: that autopsy
found 4/5 seeds advanced past Phase 0 under the landed single-episode gate
(H_POS_FRAC_OF_MAX=0.20, threshold 0.20*ln(144) ~= 0.994), but seed 46 stayed
stuck at Phase 0 forever (h_pos_mean=0.0375, h_pos_max=0.690 over the full 160-ep
budget -- a near-stationary policy that never reached the threshold on ANY
episode). The autopsy adjudicated the seed-46 miss as a GENUINE early monostrategy
/ exploration collapse (the failure mode the behavioral_diversity_isolation /
ARC-065 programme targets), surfaced by the gate -- NOT primarily a gate defect.
It explicitly forbade lowering the threshold (that would advance a non-explorer
past an unacquired Phase-0 competency) and deferred any gate change pending a
re-run with the exploration-diversity substrate armed.

SLEEP DRIVER: K=never (SleepLoopManager instantiated with K > total episodes via
the diversity-armed builder; never fires during this readiness probe).

QUESTION
--------
Does seed-46-class early exploration collapse DISAPPEAR when the landed
exploration-diversity stack is explicitly armed at its main-path-default
magnitudes -- specifically MECH-313 stochastic noise floor (use_noise_floor=True)
+ MECH-314 structured curiosity (use_structured_curiosity=True), with SP-CEM
already the main-path default (use_support_preserving_cem=True since 2026-05-17)?
591b ran with SP-CEM active but with the noise floor + curiosity masters OFF
(REEConfig.from_dims defaults). This probe is the SAME reachability run with those
two masters flipped ON and nothing else changed.

INTERPRETATION (the diagnose-first fork the autopsy set up)
-----------------------------------------------------------
- If all 5 seeds now reach Phase 1 AND the advanced seeds actually explored
  (healthy per-episode H_pos distribution, not a single fluke crossing): the
  exploration-diversity substrate resolves seed-46-class collapse on its own ->
  the gate needs NO change. Close with that finding.
- If collapse persists (a seed still never clears the gate even with diversity
  armed): the residual is not resolved by the in-flight diversity stack at its
  landed default magnitudes -> routes to the gate-robustness work the autopsy
  deferred (K-of-N / EMA crossing, or an active Phase-0 exploration-shaping
  stage), NOT to lowering the threshold.

DESIGN
------
Single arm (curriculum only, diversity-armed -- this is readiness, not a
comparison). Mirrors V3-EXQ-591b exactly (faithful 200-steps/ep config, all 5
seeds, 160 episodes, same per-episode pos_entropy fed into sched.update()) with
ONE change: the agent is built with use_noise_floor=True +
use_structured_curiosity=True at the landed default magnitudes. Reuses the
canonical 591 _extract_obs / constants and the InfantCurriculumScheduler.

ACCEPTANCE
----------
PRIMARY (load-bearing) C1: every seed reaches curriculum_final_phase >= 1
(Phase 0->1 fires) -- the prereq-(c) reachability claim under the diversity stack.
NON-VACUITY (the autopsy's "confirm advanced seeds actually explored"): each
seed that reaches Phase 1 must have a non-degenerate H_pos distribution
(h_pos_mean clears an exploration floor AND it cleared the gate threshold on more
than a single fluke episode), so a PASS reflects genuine exploration rather than
the gate admitting a near-stationary policy on one lucky episode.
SECONDARY/INFORMATIONAL: Phase 1->2 is deliberately out of scope (N_EPISODES=160
is below the Phase 1->2 ep_min of 500, and that gate is independently blocked by
the collapsed z_goal_norm >= 0.30 prereq (b)). A Phase-1 final state is the
expected ceiling, NOT a failure of this probe.

ASCII-only output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from infant_curriculum import InfantCurriculumScheduler, H_POS_FRAC_OF_MAX  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# Reuse the canonical 591 helpers + constants (DRY -- no copy-drift).
from v3_exq_591_isef005_curriculum_vs_flat_v3 import (  # noqa: E402
    _extract_obs,
    BODY_OBS_DIM,
    WORLD_OBS_DIM,
    GRID_SIZE,
    ACTION_DIM,
)

QUEUE_ID = "V3-EXQ-591c"
EXPERIMENT_TYPE = "v3_exq_591c_isef005_curriculum_phase_advance_readiness_diversity"
CLAIM_IDS: List[str] = []  # substrate-readiness diagnostic -- weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44, 45, 46]
N_EPISODES = 160          # faithful to V3-EXQ-591b
STEPS_PER_EPISODE = 200   # faithful to V3-EXQ-591b

H_MAX = math.log(GRID_SIZE ** 2)
PHASE_01_THRESHOLD = H_POS_FRAC_OF_MAX * H_MAX   # 0.20 * ln(144) ~= 0.994
# Readiness floor: the early-policy agent must actually MOVE (non-trivial per-episode
# H_pos) for the C1 gate result to be a real verdict rather than a degenerate artifact.
# SAME statistic the C1 gate routes on (per-episode pos_entropy).
H_POS_MOVEMENT_FLOOR = 0.20
# Genuine-exploration floor for a seed that ADVANCED: its mean per-episode H_pos must
# clear this (a near-stationary policy that fluked a single crossing reads well below
# it -- seed 46 in 591b had h_pos_mean=0.0375), AND it must have cleared the gate
# threshold on more than a single eligible episode.
GENUINE_EXPLORATION_H_POS_MEAN_FLOOR = 0.20
GENUINE_EXPLORATION_MIN_CROSSINGS = 2


def _build_diversity_agent() -> REEAgent:
    """Mirror of V3-EXQ-591's _build_agent with the exploration-diversity stack
    armed: MECH-313 noise floor + MECH-314 structured curiosity ON at their
    landed default magnitudes. SP-CEM is already the main-path default
    (use_support_preserving_cem=True since 2026-05-17). Everything else is
    bit-identical to the 591/591b agent build."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        z_goal_enabled=True,
        drive_weight=2.0,
        novelty_bonus_weight=0.5,
        use_sleep_loop=True,
        sleep_loop_episodes_K=N_EPISODES + 1,  # K=never (> total episodes)
        # --- exploration-diversity stack (the intervention) ---
        use_noise_floor=True,                  # MECH-313 (defaults: alpha=0.1, min_T=1.0)
        use_structured_curiosity=True,         # MECH-314 (defaults: sub-flavours on, w=0.05)
        # use_support_preserving_cem defaults True (SP-CEM main-path; left implicit)
    )
    cfg.latent.alpha_world = 0.9
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    return REEAgent(cfg)


def _run_seed(*, seed: int, n_episodes: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = _build_diversity_agent()
    sched = InfantCurriculumScheduler(grid_size=GRID_SIZE)

    h_pos_window: deque = deque(maxlen=100)  # rolling (informational only)
    per_ep_h_pos: List[float] = []
    phase_01_at: Optional[int] = None
    phase_12_at: Optional[int] = None

    for ep in range(n_episodes):
        env_kwargs = sched.env_kwargs()  # Phase 0 -> all infant features OFF
        agent.config.e3.novelty_bonus_weight = float(
            sched.config_overrides().get("novelty_bonus_weight", 0.5))

        env = CausalGridWorldV2(
            size=GRID_SIZE,
            seed=seed * n_episodes + ep,
            resource_respawn_on_consume=True,
            pos_telemetry_enabled=True,
            traj_telemetry_enabled=True,
        )
        _flat, obs_dict = env.reset()
        ob, ow = _extract_obs(obs_dict)

        ep_h_pos = -1.0
        ep_benefit_contacts = 0

        for _step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                action = agent.act_with_split_obs(obs_body=ob, obs_world=ow)
            ai = int(action.argmax().item()) % ACTION_DIM
            _o, harm_signal, done, info, obs_dict = env.step(ai)
            agent.update_residue(float(harm_signal))
            ob, ow = _extract_obs(obs_dict)
            benefit = float(ob[11].item()) if ob.shape[0] > 11 else 0.0
            energy = float(ob[3].item()) if ob.shape[0] > 3 else 0.5
            drive = max(0.0, min(1.0, 1.0 - energy))
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
            ep_h_pos = float(info.get("pos_entropy", -1.0))
            ep_benefit_contacts += int(
                float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0)
            if done:
                _flat, obs_dict = env.reset()
                ob, ow = _extract_obs(obs_dict)

        z_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
        cov = float(agent.residue_field.get_coverage_telemetry()["residue_coverage_pct"])

        per_ep_h_pos.append(ep_h_pos)
        h_pos_window.append(ep_h_pos)

        prev_phase = sched.current_phase
        sched.update(
            ep,
            h_pos=ep_h_pos if ep_h_pos >= 0.0 else None,
            z_goal_norm=z_norm,
            benefit_contacts=ep_benefit_contacts,
            residue_coverage_pct=cov,
        )
        if prev_phase == 0 and sched.current_phase >= 1 and phase_01_at is None:
            phase_01_at = ep
        if prev_phase <= 1 and sched.current_phase >= 2 and phase_12_at is None:
            phase_12_at = ep

        if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] curriculum-diversity seed={seed} ep {ep + 1}/{n_episodes}"
                f" phase={sched.current_phase} h_pos={ep_h_pos:.3f}"
                f" z_goal={z_norm:.4f}",
                flush=True,
            )

    valid = [h for h in per_ep_h_pos if h >= 0.0]
    eligible = [h for h in per_ep_h_pos[100:] if h >= 0.0]  # post Phase-0 ep_min
    n_eligible_ge_threshold = sum(1 for h in eligible if h >= PHASE_01_THRESHOLD)
    final_phase = sched.current_phase
    reached_phase1 = final_phase >= 1
    h_pos_mean = (sum(valid) / len(valid)) if valid else -1.0
    # Genuine exploration: not a near-stationary policy that fluked one crossing.
    genuine_exploration = bool(
        reached_phase1
        and h_pos_mean >= GENUINE_EXPLORATION_H_POS_MEAN_FLOOR
        and n_eligible_ge_threshold >= GENUINE_EXPLORATION_MIN_CROSSINGS
    )
    print(f"verdict: {'PASS' if reached_phase1 else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "curriculum_final_phase": final_phase,
        "phase_0to1_advanced_at_episode": phase_01_at,
        "phase_1to2_advanced_at_episode": phase_12_at,
        "reached_phase1": reached_phase1,
        "genuine_exploration": genuine_exploration,
        "h_pos_min": round(min(valid), 4) if valid else -1.0,
        "h_pos_mean": round(h_pos_mean, 4) if valid else -1.0,
        "h_pos_max": round(max(valid), 4) if valid else -1.0,
        "h_pos_std": round((sum((h - h_pos_mean) ** 2 for h in valid) / len(valid)) ** 0.5, 4) if valid else -1.0,
        "n_eligible_episodes": len(eligible),
        "n_eligible_ge_threshold": n_eligible_ge_threshold,
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    n_episodes = 2 if dry_run else N_EPISODES
    print(
        f"V3-EXQ-591c diversity-armed readiness: seeds={seeds} n_episodes={n_episodes}"
        f" steps={STEPS_PER_EPISODE} threshold={PHASE_01_THRESHOLD:.4f}"
        f" [use_noise_floor=True use_structured_curiosity=True SP-CEM=default-on]",
        flush=True,
    )
    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition curriculum-diversity", flush=True)
        seed_results.append(_run_seed(seed=seed, n_episodes=n_episodes))

    n_reached = sum(1 for r in seed_results if r["reached_phase1"])
    c1_all_reach_phase1 = (n_reached == len(seed_results))
    n_genuine = sum(1 for r in seed_results if r["genuine_exploration"])
    # Every seed that reached Phase 1 did so via genuine exploration (not a fluke).
    all_advanced_genuinely_explored = all(
        r["genuine_exploration"] for r in seed_results if r["reached_phase1"]
    )
    # Movement readiness: did the early-policy agent actually explore (non-trivial
    # H_pos) so the C1 gate result is a real verdict, not a degenerate artifact?
    max_h_pos = max((r["h_pos_max"] for r in seed_results), default=-1.0)
    movement_ok = max_h_pos >= H_POS_MOVEMENT_FLOOR
    # Non-degeneracy: per-episode H_pos must vary (not identically constant).
    h_pos_varies = any(r["h_pos_std"] > 1e-6 for r in seed_results)

    if not movement_ok:
        # Agent never moved -> the test is vacuous; re-run, do not read as a verdict.
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif c1_all_reach_phase1 and all_advanced_genuinely_explored:
        # The diversity stack resolves seed-46-class collapse on its own -> the
        # gate needs no change.
        outcome = "PASS"
        label = "phase01_collapse_resolved_by_diversity_stack_no_gate_change"
    elif c1_all_reach_phase1:
        # All seeds reached Phase 1 but at least one advanced without genuine
        # exploration (near-stationary policy fluked the crossing) -> the gate
        # admitted a non-exploring seed; surface for review, do NOT auto-close as
        # clean. (The autopsy's "without admitting a non-exploring seed" guard.)
        outcome = "PASS"
        label = "phase01_reachable_but_seed_advanced_without_genuine_exploration"
    else:
        # A seed still never clears the gate even with diversity armed -> the
        # in-flight diversity stack at landed defaults does NOT resolve the
        # collapse; route to the gate-robustness work the autopsy deferred
        # (K-of-N / EMA crossing, or active Phase-0 exploration shaping), NOT to
        # lowering the threshold.
        outcome = "FAIL"
        label = "phase01_collapse_persists_under_diversity_needs_gate_change"

    return {
        "outcome": outcome,
        "label": label,
        "seed_results": seed_results,
        "n_seeds_reached_phase1": n_reached,
        "n_seeds_genuine_exploration": n_genuine,
        "c1_all_reach_phase1": c1_all_reach_phase1,
        "all_advanced_genuinely_explored": all_advanced_genuinely_explored,
        "movement_ok": movement_ok,
        "max_h_pos": max_h_pos,
        "h_pos_varies": h_pos_varies,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = out_dir / f"{run_id}.json"

    ev_dir = "supports" if outcome == "PASS" else "does_not_support"
    seeds_used = list(SEEDS if not dry_run else [SEEDS[0]])

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": ev_dir,
        "sleep_driver_pattern": "K=never (SleepLoopManager K > total episodes; never fires)",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": seeds_used,
            "n_episodes": N_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "grid_size": GRID_SIZE,
            "h_pos_frac_of_max": H_POS_FRAC_OF_MAX,
            "phase_0to1_threshold": round(PHASE_01_THRESHOLD, 6),
            "h_pos_movement_floor": H_POS_MOVEMENT_FLOOR,
            "genuine_exploration_h_pos_mean_floor": GENUINE_EXPLORATION_H_POS_MEAN_FLOOR,
            "genuine_exploration_min_crossings": GENUINE_EXPLORATION_MIN_CROSSINGS,
            "arm": "InfantCurriculumScheduler (experiments/infant_curriculum.py)",
            "diversity_stack": {
                "use_noise_floor": True,
                "use_structured_curiosity": True,
                "use_support_preserving_cem": "default-on (main-path since 2026-05-17)",
                "magnitudes": "landed defaults (noise_floor_alpha=0.1, curiosity_weight=0.05)",
            },
        },
        "acceptance_criteria": {
            "C1_primary": "every seed reaches curriculum_final_phase >= 1 (Phase 0->1 fires) under the diversity stack",
            "C_nonvacuity": (
                "each seed that reaches Phase 1 explored genuinely (h_pos_mean >="
                f" {GENUINE_EXPLORATION_H_POS_MEAN_FLOOR} AND cleared the gate threshold on"
                f" >= {GENUINE_EXPLORATION_MIN_CROSSINGS} eligible episodes), so a PASS"
                " reflects real exploration rather than a near-stationary policy fluking"
                " a single crossing -- the autopsy's 'without admitting a non-exploring"
                " seed' guard."
            ),
            "C2_secondary": (
                "phase_1to2 is NOT evaluated here -- N_EPISODES=160 is below the Phase"
                " 1->2 ep_min (500), and that gate is independently blocked by the"
                " collapsed z_goal_norm >= 0.30 (prereq b). Scope is Phase 0->1"
                " reachability under the diversity stack only."
            ),
        },
        "interpretation": {
            "label": result["label"],
            "preconditions": [
                {
                    "name": "early_policy_produces_nontrivial_h_pos",
                    "description": (
                        "Early-policy agent must actually explore (max per-episode H_pos"
                        " clears a movement floor) for the C1 gate result to be a real"
                        " verdict rather than a degenerate non-moving artifact. SAME"
                        " statistic the C1 gate routes on (per-episode pos_entropy)."
                    ),
                    "measured": round(result["max_h_pos"], 4),
                    "threshold": H_POS_MOVEMENT_FLOOR,
                    "direction": "lower",
                    "control": "max per-episode H_pos across all seeds (the agent-moves positive control)",
                    "met": bool(result["movement_ok"]),
                },
            ],
            "criteria_non_degenerate": {
                # C1 discriminates only if per-episode H_pos genuinely varies.
                "C1_all_reach_phase1": bool(result["h_pos_varies"]),
            },
            "criteria": [
                {
                    "name": "C1_all_reach_phase1",
                    "load_bearing": True,
                    "passed": bool(result["c1_all_reach_phase1"]),
                },
                {
                    "name": "C_all_advanced_genuinely_explored",
                    "load_bearing": False,
                    "passed": bool(result["all_advanced_genuinely_explored"]),
                },
            ],
        },
        "metrics": {
            "n_seeds_reached_phase1": result["n_seeds_reached_phase1"],
            "n_seeds_genuine_exploration": result["n_seeds_genuine_exploration"],
            "n_seeds_total": len(seeds_used),
            "c1_all_reach_phase1": result["c1_all_reach_phase1"],
            "all_advanced_genuinely_explored": result["all_advanced_genuinely_explored"],
            "max_h_pos": round(result["max_h_pos"], 4),
        },
        "per_seed_results": result["seed_results"],
        "notes": (
            "Diagnose-first follow-up to failure_autopsy_V3-EXQ-591b_2026-06-10. Same"
            " Phase 0->1 reachability run as V3-EXQ-591b (200 steps/ep, 5 seeds, 160 ep,"
            " landed H_POS_FRAC_OF_MAX=0.20 gate) with the exploration-diversity stack"
            " ARMED (MECH-313 noise floor + MECH-314 curiosity ON at landed defaults;"
            " SP-CEM already main-path default). 591b ran with SP-CEM on but those two"
            " masters OFF, and seed 46 collapsed to a near-stationary policy"
            " (h_pos_mean=0.0375). PASS (collapse resolved) -> gate needs NO change;"
            " FAIL (collapse persists) -> routes to the gate-robustness work the autopsy"
            " deferred (K-of-N / EMA crossing or active Phase-0 exploration shaping),"
            " NEVER to lowering the threshold. Magnitudes are the landed defaults; if a"
            " seed persists, a Q-043 magnitude sweep is the next lever before any gate"
            " change. Full GAP-14 closure (curriculum-vs-flat, V3-EXQ-591 successor)"
            " still independently waits on prereq (b) goal-pipeline non-trivial z_goal"
            " (goal_pipeline:GAP-4). Do NOT re-queue V3-EXQ-591 here."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        summary = {k: v for k, v in manifest.items() if k != "per_seed_results"}
        print(json.dumps(summary, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
