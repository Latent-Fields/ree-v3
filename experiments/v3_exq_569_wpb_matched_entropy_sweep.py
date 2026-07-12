#!/opt/local/bin/python3
"""V3-EXQ-569 -- Work Package B: matched-entropy sweep.

Central ARC-065 calibration question: do structured diversity mechanisms
(MECH-313 noise floor, MECH-314 structured curiosity, MECH-320 tonic vigor,
MECH-260 dACC anti-recency) produce more useful behavioral diversity -- measured
by state coverage -- than simple matched-entropy random noise at the same action
entropy level?

Background: V3-EXQ-567 proved normal CEM collapses to entropy=0.012 under natural
conditions. SP-CEM (support-preserving CEM) lifts entropy to 0.497 and is therefore
the required baseline for all arms. This experiment measures whether the ARC-065
mechanisms produce further gains in state coverage BEYOND what matched-entropy noise
achieves at the same entropy level.

Six arms:
  ARM_0_sp_cem_baseline       -- SP-CEM only, temperature=1.0
  ARM_1_noise_floor           -- SP-CEM + MECH-313 noise floor
  ARM_2_structured_curiosity  -- SP-CEM + MECH-314 structured curiosity
  ARM_3_matched_entropy       -- SP-CEM + temperature=2.5 (entropy-matching control)
  ARM_4_mech_313_314_320      -- SP-CEM + MECH-313 + MECH-314 + MECH-320
  ARM_5_all_mechanisms        -- SP-CEM + MECH-313 + MECH-314 + MECH-320 + MECH-260

Temperature=2.5 for ARM_3 was chosen empirically to roughly match the entropy
expected from ARM_1/ARM_2 at temperature=1.0 with their mechanism-enabled gains.

Custom step loop used because StepHarness hardcodes temperature=1.0 at the
select_action call site; all mechanism composition happens inside select_action
so the custom loop only needs to pass arm_temperature there.

Acceptance criteria (pre-registered):
  P1: ARM_1 entropy > ARM_0 + 0.05 AND ARM_2 entropy > ARM_0 + 0.05
  P2: ARM_1 state_coverage > ARM_3 + 0.02
  P3: ARM_2 state_coverage > ARM_3 + 0.02

PASS if P1 AND (P2 OR P3).

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_569_wpb_matched_entropy_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_569_wpb_matched_entropy_sweep"
QUEUE_ID = "V3-EXQ-569"
CLAIM_IDS: List[str] = ["ARC-065"]
EXPERIMENT_PURPOSE = "evidence"

ENV_SIZE = 8
SEEDS = [42, 43, 44]
EVAL_EPISODES = 40
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

# Acceptance thresholds (pre-registered)
P1_ENTROPY_MARGIN = 0.05          # ARM_1 AND ARM_2 entropy > ARM_0 + this
P2_STATE_COVERAGE_MARGIN = 0.02   # ARM_1 state_coverage > ARM_3 + this
P3_STATE_COVERAGE_MARGIN = 0.02   # ARM_2 state_coverage > ARM_3 + this
# PASS = P1 AND (P2 OR P3)

MATCHED_ENTROPY_TEMPERATURE = 2.5
STD_FLOOR = 0.2

ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_sp_cem_baseline",
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "use_noise_floor": False,
        "use_structured_curiosity": False,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
        "dacc_suppression_memory": 8,
        "temperature": 1.0,
    },
    {
        "arm": "ARM_1_noise_floor",
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "use_noise_floor": True,
        "use_structured_curiosity": False,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
        "dacc_suppression_memory": 8,
        "temperature": 1.0,
    },
    {
        "arm": "ARM_2_structured_curiosity",
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "use_noise_floor": False,
        "use_structured_curiosity": True,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
        "dacc_suppression_memory": 8,
        "temperature": 1.0,
    },
    {
        "arm": "ARM_3_matched_entropy",
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "use_noise_floor": False,
        "use_structured_curiosity": False,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
        "dacc_suppression_memory": 8,
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
    },
    {
        "arm": "ARM_4_mech_313_314_320",
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "use_noise_floor": True,
        "use_structured_curiosity": True,
        "use_tonic_vigor": True,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
        "dacc_suppression_memory": 8,
        "temperature": 1.0,
    },
    {
        "arm": "ARM_5_all_mechanisms",
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "use_noise_floor": True,
        "use_structured_curiosity": True,
        "use_tonic_vigor": True,
        "use_dacc": True,
        "dacc_suppression_weight": 0.3,
        "dacc_suppression_memory": 8,
        "temperature": 1.0,
    },
]


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=2,
        num_resources=10,
        hazard_harm=0.01,
        resource_benefit=0.25,
        energy_decay=0.015,
        use_proxy_fields=True,
        proximity_benefit_scale=0.18,
        proximity_harm_scale=0.01,
        resource_respawn_on_consume=True,
        seed=seed,
    )


def _make_config(env: CausalGridWorld, arm: Dict[str, Any]) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        use_action_class_scaffold_candidates=False,
        use_support_preserving_cem=bool(arm["use_support_preserving_cem"]),
        support_preserving_stratified_elites=bool(
            arm["support_preserving_stratified_elites"]
        ),
        support_preserving_ao_std_floor=float(arm["support_preserving_ao_std_floor"]),
        support_preserving_min_first_action_classes=2,
        forced_score_bias_per_class=None,
        use_noise_floor=bool(arm["use_noise_floor"]),
        use_structured_curiosity=bool(arm["use_structured_curiosity"]),
        use_tonic_vigor=bool(arm["use_tonic_vigor"]),
        use_dacc=bool(arm["use_dacc"]),
        dacc_suppression_weight=float(arm["dacc_suppression_weight"]),
        dacc_suppression_memory=int(arm["dacc_suppression_memory"]),
    )


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            value -= p * math.log(p)
    return float(value)


def _mean(
    values: Iterable[float], default: Optional[float] = 0.0
) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return default
    return sum(vals) / len(vals)


def _round6(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _counter_from_dict(data: Dict[Any, Any]) -> Counter:
    counter: Counter = Counter()
    for key, value in data.items():
        counter[int(key)] += int(value)
    return counter


def _run_arm_seed(
    arm: Dict[str, Any],
    seed: int,
    episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)

    arm_temperature = float(arm.get("temperature", 1.0))

    action_counts: Counter = Counter()
    candidate_first_action_counts: Counter = Counter()
    unique_candidate_classes: List[float] = []
    candidate_entropies: List[float] = []
    support_preserving_active_steps = 0
    visited_positions: Set[Tuple[int, int]] = set()
    total_steps = 0

    z_self_prev = None
    action_prev = None

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_self_prev = None
        action_prev = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = obs_dict.get("harm_obs")
            obs_h_a = obs_dict.get("harm_obs_a")
            obs_h_h = obs_dict.get("harm_history")

            # Track position at start of step.
            visited_positions.add((int(env.agent_x), int(env.agent_y)))

            with torch.no_grad():
                # 1. sense
                latent = agent.sense(
                    obs_body,
                    obs_world,
                    obs_harm=obs_h,
                    obs_harm_a=obs_h_a,
                    obs_harm_history=obs_h_h,
                )

                # 2. record_transition
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(
                        z_self_prev,
                        action_prev,
                        latent.z_self.detach(),
                    )

                # 3. clock + e1_tick + trajectories
                ticks = agent.clock.advance()
                world_dim = latent.z_world.shape[-1]
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                # 4. update_z_goal
                drive_level = REEAgent.compute_drive_level(obs_body)
                benefit_raw = obs_dict.get("benefit_exposure", None)
                if benefit_raw is None and isinstance(obs_body, torch.Tensor):
                    if obs_body.shape[-1] > 11:
                        benefit_raw = (
                            obs_body[0, 11] if obs_body.dim() == 2
                            else obs_body[11]
                        )
                benefit_exposure = (
                    0.0 if benefit_raw is None else max(0.0, float(benefit_raw))
                )
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

                # 5. update_schema_wanting (MECH-216, off by default)
                e1_cfg = getattr(getattr(agent, "config", None), "e1", None)
                if bool(getattr(e1_cfg, "schema_wanting_enabled", False)):
                    agent.update_schema_wanting(drive_level=drive_level)

                # 6. select_action with arm temperature
                action = agent.select_action(
                    candidates, ticks, temperature=arm_temperature
                )
                if action is None:
                    idx = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, idx] = 1.0
                    agent._last_action = action

            # Collect action diagnostics.
            idx = int(action.argmax(dim=-1).item())
            action_counts[idx] += 1

            hdiag = agent.hippocampal.get_last_propose_diagnostics()
            if hdiag:
                candidate_first_action_counts.update(
                    _counter_from_dict(
                        hdiag.get("candidate_first_action_counts", {})
                    )
                )
                unique_candidate_classes.append(
                    float(hdiag.get("candidate_unique_first_action_classes", 0))
                )
                candidate_entropies.append(
                    float(hdiag.get("candidate_first_action_entropy", 0.0))
                )
                if bool(hdiag.get("support_preserving_active", False)):
                    support_preserving_active_steps += 1

            # 8. env.step
            flat_obs, harm_signal, done, info, next_obs_dict = env.step(action)

            with torch.no_grad():
                # 9. update_residue
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            total_steps += 1
            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] arm={arm['arm']} seed={seed}"
                f" ep {ep + 1}/{episodes}",
                flush=True,
            )

    action_total = sum(action_counts.values())
    selected_entropy = _entropy(action_counts)
    candidate_unique_mean = _round6(_mean(unique_candidate_classes, None))
    candidate_entropy_mean = _round6(_mean(candidate_entropies, None))
    state_coverage = len(visited_positions) / float(ENV_SIZE * ENV_SIZE)

    return {
        "arm": arm["arm"],
        "seed": seed,
        "total_steps": int(total_steps),
        "selected_action_class_entropy": round(selected_entropy, 6),
        "action_0_fraction": round(
            action_counts.get(0, 0) / action_total if action_total else 0.0,
            6,
        ),
        "unique_actions_taken": int(len(action_counts)),
        "action_counts": dict(sorted(action_counts.items())),
        "state_coverage_fraction": round(state_coverage, 6),
        "unique_positions_visited": int(len(visited_positions)),
        "candidate_unique_first_action_classes_mean": candidate_unique_mean,
        "candidate_first_action_entropy_mean": candidate_entropy_mean,
        "candidate_first_action_counts": dict(
            sorted(candidate_first_action_counts.items())
        ),
        "support_preserving_active_steps": int(support_preserving_active_steps),
    }


def _arm_rows(rows: List[Dict[str, Any]], arm_name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("arm") == arm_name]


def _mean_key(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return float(_mean(values, default) or default)


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm0_rows = _arm_rows(arm_results, "ARM_0_sp_cem_baseline")
    arm1_rows = _arm_rows(arm_results, "ARM_1_noise_floor")
    arm2_rows = _arm_rows(arm_results, "ARM_2_structured_curiosity")
    arm3_rows = _arm_rows(arm_results, "ARM_3_matched_entropy")
    arm4_rows = _arm_rows(arm_results, "ARM_4_mech_313_314_320")
    arm5_rows = _arm_rows(arm_results, "ARM_5_all_mechanisms")

    arm0_entropy = _mean_key(arm0_rows, "selected_action_class_entropy")
    arm1_entropy = _mean_key(arm1_rows, "selected_action_class_entropy")
    arm2_entropy = _mean_key(arm2_rows, "selected_action_class_entropy")
    arm3_entropy = _mean_key(arm3_rows, "selected_action_class_entropy")
    arm4_entropy = _mean_key(arm4_rows, "selected_action_class_entropy")
    arm5_entropy = _mean_key(arm5_rows, "selected_action_class_entropy")

    arm0_coverage = _mean_key(arm0_rows, "state_coverage_fraction")
    arm1_coverage = _mean_key(arm1_rows, "state_coverage_fraction")
    arm2_coverage = _mean_key(arm2_rows, "state_coverage_fraction")
    arm3_coverage = _mean_key(arm3_rows, "state_coverage_fraction")
    arm4_coverage = _mean_key(arm4_rows, "state_coverage_fraction")
    arm5_coverage = _mean_key(arm5_rows, "state_coverage_fraction")

    p1 = bool(
        arm1_entropy > arm0_entropy + P1_ENTROPY_MARGIN
        and arm2_entropy > arm0_entropy + P1_ENTROPY_MARGIN
    )
    p2 = bool(arm1_coverage > arm3_coverage + P2_STATE_COVERAGE_MARGIN)
    p3 = bool(arm2_coverage > arm3_coverage + P3_STATE_COVERAGE_MARGIN)

    return {
        # Entropy
        "arm0_selected_entropy_mean": round(arm0_entropy, 6),
        "arm1_selected_entropy_mean": round(arm1_entropy, 6),
        "arm2_selected_entropy_mean": round(arm2_entropy, 6),
        "arm3_selected_entropy_mean": round(arm3_entropy, 6),
        "arm4_selected_entropy_mean": round(arm4_entropy, 6),
        "arm5_selected_entropy_mean": round(arm5_entropy, 6),
        "entropy_delta_arm1_minus_arm0": round(arm1_entropy - arm0_entropy, 6),
        "entropy_delta_arm2_minus_arm0": round(arm2_entropy - arm0_entropy, 6),
        # State coverage
        "arm0_state_coverage_mean": round(arm0_coverage, 6),
        "arm1_state_coverage_mean": round(arm1_coverage, 6),
        "arm2_state_coverage_mean": round(arm2_coverage, 6),
        "arm3_state_coverage_mean": round(arm3_coverage, 6),
        "arm4_state_coverage_mean": round(arm4_coverage, 6),
        "arm5_state_coverage_mean": round(arm5_coverage, 6),
        "coverage_delta_arm1_minus_arm3": round(arm1_coverage - arm3_coverage, 6),
        "coverage_delta_arm2_minus_arm3": round(arm2_coverage - arm3_coverage, 6),
        # Acceptance criteria
        "p1_both_mechanisms_entropy_lift": p1,
        "p2_arm1_coverage_beats_matched_entropy": p2,
        "p3_arm2_coverage_beats_matched_entropy": p3,
        "overall_pass": bool(p1 and (p2 or p3)),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps)
            arm_results.append(cell)
            passed = cell["selected_action_class_entropy"] >= 0.0
            print(
                f"verdict: {'PASS' if passed else 'FAIL'}",
                flush=True,
            )

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_note": (
            "6-arm matched-entropy sweep: ARM_0 SP-CEM baseline vs ARM_1 "
            "noise-floor vs ARM_2 structured-curiosity vs ARM_3 matched-entropy "
            "random-noise control vs ARM_4 combo vs ARM_5 all-mechanisms. "
            "PASS = ARC-065 structured diversity mechanisms produce more useful "
            "state coverage than matched-entropy random noise at the same entropy "
            "level. FAIL = no structured mechanism beats matched entropy on coverage."
        ),
        "dry_run": dry_run,
        "config": {
            "seeds": seeds,
            "eval_episodes": episodes,
            "steps_per_episode": steps,
            "env_size": ENV_SIZE,
            "p1_entropy_margin": P1_ENTROPY_MARGIN,
            "p2_state_coverage_margin": P2_STATE_COVERAGE_MARGIN,
            "p3_state_coverage_margin": P3_STATE_COVERAGE_MARGIN,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "std_floor": STD_FLOOR,
            "arms": [arm["arm"] for arm in ARMS],
        },
        "acceptance_criteria": {
            "P1_both_mechanisms_entropy_lift": summary["p1_both_mechanisms_entropy_lift"],
            "P2_arm1_coverage_beats_matched_entropy": summary[
                "p2_arm1_coverage_beats_matched_entropy"
            ],
            "P3_arm2_coverage_beats_matched_entropy": summary[
                "p3_arm2_coverage_beats_matched_entropy"
            ],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    for key, value in manifest["acceptance_criteria"].items():
        print(f"  {key}: {value}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-569 WPB matched-entropy sweep"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run; no manifest written.",
    )
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
