"""
V3-EXQ-591b -- InfantCurriculumScheduler Phase 0->1 advancement-gate reachability.

SUBSTRATE-READINESS DIAGNOSTIC for infant_substrate:GAP-14 prerequisite (c) ONLY.
This is NOT the full curriculum-vs-flat governance comparison (that is V3-EXQ-591's
job and still waits on prereq (b): the goal-pipeline z_goal collapse owned by
goal_pipeline:GAP-4 / behavioral_diversity_isolation:GAP-C).

SLEEP DRIVER: K=never (SleepLoopManager instantiated with K > total episodes via the
reused 591 _build_agent; never fires during this readiness probe).

QUESTION
--------
Under the landed InfantCurriculumScheduler Phase 0->1 advancement gate
(H_POS_FRAC_OF_MAX = 0.20, recalibrated from the structurally-unreachable 0.70 on
2026-05-31), does a real early-policy (untrained) REEAgent actually advance past
Phase 0 -- i.e. does the curriculum WALK?

Background: V3-EXQ-591 (FAIL 2026-05-26) found the curriculum stuck at Phase 0 in
all 5 seeds under the OLD 0.70 gate (threshold 0.70*ln(144) ~= 3.48 vs observed
per-episode H_pos max ~2). A live early-policy smoke (2026-06-09, 112 ep, 130
steps/ep) under the new 0.20 gate (threshold 0.20*ln(144) ~= 0.994) advanced 2/3
seeds (42 at ep107, 44 at ep101) but left seed 43 stuck inside the short window.
This probe runs the faithful 200-steps/ep config over more eligible episodes and
all 5 seeds to confirm reachability.

DESIGN
------
Single arm (curriculum only -- no flat baselines; this is readiness, not a
comparison). Reuses the canonical _build_agent / _extract_obs from the 591 module
and the InfantCurriculumScheduler, feeding per-episode pos_entropy into
sched.update() exactly as the 591 curriculum arm does. Per seed it records the
curriculum final phase, the episode at which Phase 0->1 (and, if reached, 1->2)
fired, and the per-episode H_pos distribution.

ACCEPTANCE
----------
PRIMARY (load-bearing) C1: every seed reaches curriculum_final_phase >= 1 (Phase
0->1 fires). This is the prereq-(c) reachability claim.
SECONDARY/INFORMATIONAL C2: Phase 1->2 is deliberately out of scope here -- the run
stops at 160 episodes (below the Phase 1->2 ep_min of 500) AND that gate is
independently blocked by the collapsed z_goal_norm >= 0.30 (prereq b). A Phase-1
final state is the expected ceiling, NOT a failure of this probe.

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
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402

# Reuse the canonical 591 curriculum-arm helpers (DRY -- no copy-drift).
from v3_exq_591_isef005_curriculum_vs_flat_v3 import (  # noqa: E402
    _build_agent,
    _extract_obs,
    GRID_SIZE,
    ACTION_DIM,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-591b"
EXPERIMENT_TYPE = "v3_exq_591b_isef005_curriculum_phase_advance_readiness"
CLAIM_IDS: List[str] = []  # substrate-readiness diagnostic -- weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44, 45, 46]
# Scoped to the Phase 0->1 reachability question (prereq c). 60 eligible episodes
# past the Phase 0->1 ep_min (100) -- 5x the stuck-seed window the 2026-06-09 smoke
# had. NOT run past the Phase 1->2 ep_min (500): the full hippocampal-CEM agent
# costs ~20 s/episode (~14 h at 520 ep x 5 seeds), and Phase 1->2 cannot fire
# regardless because its z_goal_norm >= 0.30 gate is collapsed pending prereq (b),
# so episodes 160..500 would burn compute on a foregone z_goal stall.
N_EPISODES = 160
STEPS_PER_EPISODE = 200   # faithful to V3-EXQ-591 (higher H_pos than the 130-step smoke)

H_MAX = math.log(GRID_SIZE ** 2)
PHASE_01_THRESHOLD = H_POS_FRAC_OF_MAX * H_MAX   # 0.20 * ln(144) ~= 0.994
# Readiness floor: the early-policy agent must actually MOVE (non-trivial H_pos)
# for the C1 gate result to be a real verdict rather than a degenerate artifact.
H_POS_MOVEMENT_FLOOR = 0.20


def _run_seed(*, seed: int, n_episodes: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = _build_agent(novelty_bonus_weight=0.5)
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
                f"  [train] curriculum seed={seed} ep {ep + 1}/{n_episodes}"
                f" phase={sched.current_phase} h_pos={ep_h_pos:.3f}"
                f" z_goal={z_norm:.4f}",
                flush=True,
            )

    valid = [h for h in per_ep_h_pos if h >= 0.0]
    eligible = [h for h in per_ep_h_pos[100:] if h >= 0.0]  # post Phase-0 ep_min
    final_phase = sched.current_phase
    reached_phase1 = final_phase >= 1
    print(f"verdict: {'PASS' if reached_phase1 else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "curriculum_final_phase": final_phase,
        "phase_0to1_advanced_at_episode": phase_01_at,
        "phase_1to2_advanced_at_episode": phase_12_at,
        "reached_phase1": reached_phase1,
        "h_pos_min": round(min(valid), 4) if valid else -1.0,
        "h_pos_mean": round(sum(valid) / len(valid), 4) if valid else -1.0,
        "h_pos_max": round(max(valid), 4) if valid else -1.0,
        "h_pos_std": round((sum((h - sum(valid) / len(valid)) ** 2 for h in valid) / len(valid)) ** 0.5, 4) if valid else -1.0,
        "n_eligible_episodes": len(eligible),
        "n_eligible_ge_threshold": sum(1 for h in eligible if h >= PHASE_01_THRESHOLD),
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    n_episodes = 2 if dry_run else N_EPISODES
    print(
        f"V3-EXQ-591b readiness: seeds={seeds} n_episodes={n_episodes}"
        f" steps={STEPS_PER_EPISODE} threshold={PHASE_01_THRESHOLD:.4f}",
        flush=True,
    )
    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition curriculum", flush=True)
        seed_results.append(_run_seed(seed=seed, n_episodes=n_episodes))

    n_reached = sum(1 for r in seed_results if r["reached_phase1"])
    c1_all_reach_phase1 = (n_reached == len(seed_results))
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
    elif c1_all_reach_phase1:
        outcome = "PASS"
        label = "phase01_gate_reachable_under_landed_threshold"
    else:
        # Agent explores but some seed never clears the gate -> the landed 0.20
        # threshold is not reliably early-policy-reachable; route to gate
        # strengthening (e.g. hard episode-count fallback), NOT a claim verdict.
        outcome = "FAIL"
        label = "phase01_gate_unreliable_needs_strengthening"

    return {
        "outcome": outcome,
        "label": label,
        "seed_results": seed_results,
        "n_seeds_reached_phase1": n_reached,
        "c1_all_reach_phase1": c1_all_reach_phase1,
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
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_episodes": N_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "grid_size": GRID_SIZE,
            "h_pos_frac_of_max": H_POS_FRAC_OF_MAX,
            "phase_0to1_threshold": round(PHASE_01_THRESHOLD, 6),
            "h_pos_movement_floor": H_POS_MOVEMENT_FLOOR,
            "arm": "InfantCurriculumScheduler (experiments/infant_curriculum.py)",
        },
        "acceptance_criteria": {
            "C1_primary": "every seed reaches curriculum_final_phase >= 1 (Phase 0->1 fires)",
            "C2_secondary": (
                "phase_1to2 is NOT evaluated here -- N_EPISODES=160 is below the"
                " Phase 1->2 ep_min (500). It is also independently blocked by the"
                " collapsed z_goal_norm >= 0.30 gate (prereq b). Scope is Phase 0->1"
                " reachability only; phase_1to2_advanced_at_episode will be null."
            ),
        },
        "interpretation": {
            "label": result["label"],
            "preconditions": [
                {
                    "name": "early_policy_produces_nontrivial_h_pos",
                    "description": (
                        "Early-policy agent must actually explore (per-episode H_pos"
                        " clears a movement floor) for the C1 gate result to be a real"
                        " verdict rather than a degenerate non-moving artifact. Same"
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
            ],
        },
        "metrics": {
            "n_seeds_reached_phase1": result["n_seeds_reached_phase1"],
            "n_seeds_total": len(SEEDS if not dry_run else [SEEDS[0]]),
            "c1_all_reach_phase1": result["c1_all_reach_phase1"],
            "max_h_pos": round(result["max_h_pos"], 4),
        },
        "per_seed_results": result["seed_results"],
        "notes": (
            "Substrate-readiness for GAP-14 prereq (c) ONLY (Phase 0->1 gate"
            " reachability under the landed H_POS_FRAC_OF_MAX=0.20, recalibrated"
            " from the unreachable 0.70 on 2026-05-31). Full GAP-14 closure"
            " (curriculum-vs-flat, V3-EXQ-591 successor) still waits on prereq (b)"
            " goal-pipeline non-trivial z_goal, owned by goal_pipeline:GAP-4 /"
            " behavioral_diversity_isolation:GAP-C. Do NOT re-queue V3-EXQ-591"
            " until prereq (b) clears. Phase 1->2 stall here is the documented"
            " prereq-(b) z_goal ceiling, not a failure of this probe."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
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
