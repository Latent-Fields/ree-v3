#!/opt/local/bin/python3
"""V3-EXQ-562 -- WPC Rung 1: forced z_goal score-margin seam diagnostic.

Claims: [] (diagnostic only; no governance weighting)

Purpose
-------
WPC Rung 1 bypasses the monostrategy problem and verifies the E3 goal-scoring
seam directly by forcibly injecting a known z_goal vector before each
select_action call. This decouples whether E3 CAN produce a score margin
from whether the upstream goal-stream can PRODUCE a live z_goal.

EXQ-559a: goal stream signals LIVE internally.
EXQ-560:  score/bias rank-constant because candidates all share 1 action class.
EXQ-561:  MECH-313 + MECH-314 + MECH-320 diversity modules cannot break
          monostrategy (all arm entropy = 0.0).

Design
------
ARM_0_baseline:
    goal_weight=0.0, z_goal natural (no forcing). Baseline: no goal signal.

ARM_1_natural:
    goal_weight=1.0, z_goal natural (may be near-zero without contact).
    Tests whether organic seeding produces any score margin.

ARM_2_forced:
    goal_weight=1.0, z_goal FORCED to normalized z_world at every eval step.
    Tests: does a non-zero, meaningful z_goal produce a score margin?

ARM_3_forced_high:
    goal_weight=2.0, z_goal FORCED.
    Tests: does a doubled goal_weight increase the margin further?

PASS criteria (all must hold):
  P1: ARM_2 candidate_score_std_mean > ARM_0 candidate_score_std_mean * 2
  P2: ARM_3 candidate_goal_component_range_mean > ARM_2 candidate_goal_component_range_mean
  P3: z_goal_norm_mean > 0.1 in ARM_2 and ARM_3

Interpretation grid (pre-registered):
  PASS all       -> E3 goal-scoring seam is live; monostrategy is the primary
                   blocker. Next step: break monostrategy (SD-054 reef / WPC-2).
  P1 fails       -> Goal component too small to move scores; goal_weight or
                   z_goal injection strength inadequate. Check norm print-outs.
  P2 fails       -> goal_weight doubling has no differential effect; suspect
                   score swamping by base / harm costs (increase goal_weight).
  P3 fails       -> Forced injection did not stick; inspect _z_goal after force.
  All fail       -> Deeper instrumentation needed; check score_trajectory path
                   (is_active gate, config.goal_weight > 0 guard).

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_562_wpc1_goal_score_margin.py --dry-run
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_562_wpc1_goal_score_margin"
QUEUE_ID = "V3-EXQ-562"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
TRAIN_EPISODES = 40
EVAL_EPISODES = 15
STEPS_PER_EPISODE = 120

DRY_RUN_SEEDS = [42]
DRY_RUN_TRAIN_EPISODES = 3
DRY_RUN_EVAL_EPISODES = 2
DRY_RUN_STEPS = 50

EPS = 1e-9

ARMS = [
    {
        "arm": "ARM_0_baseline",
        "goal_weight": 0.0,
        "force_z_goal": False,
    },
    {
        "arm": "ARM_1_natural",
        "goal_weight": 1.0,
        "force_z_goal": False,
    },
    {
        "arm": "ARM_2_forced",
        "goal_weight": 1.0,
        "force_z_goal": True,
    },
    {
        "arm": "ARM_3_forced_high",
        "goal_weight": 2.0,
        "force_z_goal": True,
    },
]


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=8,
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


def _make_config(env: CausalGridWorld, arm: Dict) -> REEConfig:
    """Build config for this arm.

    goal_weight=0 for baseline; >= 1 for goal arms.
    z_goal_enabled=True so GoalState is instantiated and can be written to.
    drive_weight=2.0 for organic seeding in ARM_1; has no effect when forcing.
    """
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        z_goal_enabled=True,
        goal_weight=arm["goal_weight"],
        benefit_threshold=0.05,
        drive_weight=2.0,
    )


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _install_score_probe(
    agent: REEAgent,
    arm: Dict,
    score_records: List[Dict],
    z_goal_norms: List[float],
) -> None:
    """Monkey-patch e3.select to capture per-candidate scores and z_goal norm."""
    real_select = agent.e3.select
    force_z_goal = arm["force_z_goal"]

    def select_spy(candidates, temperature=1.0, *args, **kwargs):
        # --- Force z_goal injection before scoring ---
        goal_state = kwargs.get("goal_state")
        if force_z_goal and goal_state is not None:
            latent = getattr(agent, "_current_latent", None)
            if latent is not None and latent.z_world is not None:
                with torch.no_grad():
                    forced = F.normalize(
                        latent.z_world.detach(), dim=-1
                    )
                    goal_state._z_goal = forced
                    # Ensure is_active() sees a non-zero vector
                    norm = float(goal_state._z_goal.norm().item())
                    goal_state._goal_norm_peak = max(
                        float(getattr(goal_state, "_goal_norm_peak", 0.0)),
                        norm,
                    )

        # Capture z_goal norm before calling real select
        if goal_state is not None:
            z_goal_norms.append(float(goal_state._z_goal.norm().item()))
        else:
            z_goal_norms.append(0.0)

        result = real_select(candidates, temperature, *args, **kwargs)

        # Capture per-candidate scores from result
        scores_tensor = result.scores.detach().float()
        scores_list = scores_tensor.tolist()
        n = len(scores_list)
        if n > 1:
            score_std = float(scores_tensor.std(unbiased=False).item())
            score_range = float((scores_tensor.max() - scores_tensor.min()).item())
        elif n == 1:
            score_std = 0.0
            score_range = 0.0
        else:
            score_std = 0.0
            score_range = 0.0

        # Compute goal component range for arm with goal scoring
        goal_component_range = 0.0
        if (goal_state is not None
                and goal_state.is_active()
                and agent.e3.config.goal_weight > 0.0):
            try:
                goal_scores = torch.stack([
                    agent.e3.compute_goal_score(c, goal_state).mean()
                    for c in candidates
                ]).detach().float()
                g_vals = (-agent.e3.config.goal_weight) * goal_scores
                if g_vals.numel() > 1:
                    goal_component_range = float(
                        (g_vals.max() - g_vals.min()).item()
                    )
            except Exception:
                pass

        # Compute action-class argmax for each candidate
        action_classes = []
        for c in candidates:
            try:
                ac = int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
            except Exception:
                ac = -1
            action_classes.append(ac)

        score_records.append({
            "score_std": score_std,
            "score_range": score_range,
            "goal_component_range": goal_component_range,
            "n_candidates": n,
            "action_classes": action_classes,
            "selected_index": int(result.selected_index),
        })
        return result

    agent.e3.select = select_spy


def _run_arm_seed(
    arm: Dict,
    seed: int,
    train_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)

    # --- P0: warm-up training (natural, goal stream naturally updating) ---
    train_harness = StepHarness(agent, env, train_mode=True, seed=seed)
    for ep in range(train_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        train_harness.reset()
        for _ in range(steps_per_episode):
            result = train_harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (ep + 1) % 10 == 0 or (ep + 1) == train_episodes:
            print(
                f"  [train] seed={seed} arm={arm['arm']} ep {ep + 1}/{train_episodes}",
                flush=True,
            )

    # --- P1: eval phase with probe installed ---
    score_records: List[Dict] = []
    z_goal_norms: List[float] = []
    action_counts: Counter = Counter()

    # Install probe AFTER training (eval-only injection)
    _install_score_probe(agent, arm, score_records, z_goal_norms)

    eval_harness = StepHarness(agent, env, train_mode=False, seed=seed)

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        eval_harness.reset()
        for _ in range(steps_per_episode):
            result = eval_harness.step(obs_dict)
            action_idx = int(result.action.argmax(dim=-1).item())
            action_counts[action_idx] += 1
            obs_dict = result.next_obs_dict
            if result.done:
                break

    # --- Summarize metrics ---
    n_score = len(score_records)
    z_goal_norm_mean = float(np.mean(z_goal_norms)) if z_goal_norms else 0.0
    candidate_score_std_mean = (
        float(np.mean([r["score_std"] for r in score_records]))
        if score_records else 0.0
    )
    candidate_score_range_mean = (
        float(np.mean([r["score_range"] for r in score_records]))
        if score_records else 0.0
    )
    candidate_goal_component_range_mean = (
        float(np.mean([r["goal_component_range"] for r in score_records]))
        if score_records else 0.0
    )

    # Action entropy across all eval steps
    action_entropy = _entropy(action_counts)

    # Unique action classes across candidates (avg per step)
    if score_records:
        unique_action_classes_per_step = [
            float(len(set(r["action_classes"])))
            for r in score_records
        ]
        candidate_unique_action_classes_mean = float(
            np.mean(unique_action_classes_per_step)
        )
    else:
        candidate_unique_action_classes_mean = 0.0

    # Final z_goal norm (after all eval)
    goal_norm_final = (
        float(agent.goal_state.goal_norm())
        if agent.goal_state is not None else 0.0
    )

    return {
        "seed": int(seed),
        "arm": arm["arm"],
        "goal_weight": float(arm["goal_weight"]),
        "force_z_goal": bool(arm["force_z_goal"]),
        "n_score_records": int(n_score),
        "z_goal_norm_mean": z_goal_norm_mean,
        "goal_norm_final": goal_norm_final,
        "candidate_score_std_mean": candidate_score_std_mean,
        "candidate_score_range_mean": candidate_score_range_mean,
        "candidate_goal_component_range_mean": candidate_goal_component_range_mean,
        "candidate_unique_action_classes_mean": candidate_unique_action_classes_mean,
        "action_class_entropy": action_entropy,
        "action_counts": {str(k): int(v) for k, v in sorted(action_counts.items())},
    }


def _aggregate(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for arm in ARMS:
        arm_name = arm["arm"]
        subset = [r for r in rows if r["arm"] == arm_name]
        if not subset:
            continue

        def mean_key(key: str) -> float:
            vals = [float(r[key]) for r in subset]
            return float(np.mean(vals)) if vals else 0.0

        out[arm_name] = {
            "arm": arm_name,
            "n_seeds": len(subset),
            "goal_weight": float(arm["goal_weight"]),
            "force_z_goal": bool(arm["force_z_goal"]),
            "z_goal_norm_mean": mean_key("z_goal_norm_mean"),
            "candidate_score_std_mean": mean_key("candidate_score_std_mean"),
            "candidate_score_range_mean": mean_key("candidate_score_range_mean"),
            "candidate_goal_component_range_mean": mean_key(
                "candidate_goal_component_range_mean"
            ),
            "candidate_unique_action_classes_mean": mean_key(
                "candidate_unique_action_classes_mean"
            ),
            "action_class_entropy_mean": mean_key("action_class_entropy"),
            "n_score_records_mean": mean_key("n_score_records"),
        }
    return out


def _evaluate(rows: List[Dict], agg: Dict[str, Dict]) -> Dict:
    finite = True
    for r in rows:
        for key in (
            "z_goal_norm_mean", "candidate_score_std_mean",
            "candidate_score_range_mean", "candidate_goal_component_range_mean",
        ):
            v = r.get(key, 0.0)
            if not math.isfinite(float(v)):
                finite = False

    arm0 = agg.get("ARM_0_baseline", {})
    arm2 = agg.get("ARM_2_forced", {})
    arm3 = agg.get("ARM_3_forced_high", {})

    arm0_score_std = float(arm0.get("candidate_score_std_mean", 0.0))
    arm2_score_std = float(arm2.get("candidate_score_std_mean", 0.0))
    arm2_goal_rng = float(arm2.get("candidate_goal_component_range_mean", 0.0))
    arm3_goal_rng = float(arm3.get("candidate_goal_component_range_mean", 0.0))
    arm2_z_goal_norm = float(arm2.get("z_goal_norm_mean", 0.0))
    arm3_z_goal_norm = float(arm3.get("z_goal_norm_mean", 0.0))

    p1 = bool(arm2_score_std > arm0_score_std * 2.0 + EPS)
    p2 = bool(arm3_goal_rng > arm2_goal_rng + EPS)
    p3 = bool(arm2_z_goal_norm > 0.1 and arm3_z_goal_norm > 0.1)

    all_pass = bool(finite and p1 and p2 and p3)

    return {
        "all_pass": all_pass,
        "finite_metrics": bool(finite),
        "P1_arm2_score_std_gt_2x_arm0": p1,
        "P2_arm3_goal_rng_gt_arm2": p2,
        "P3_z_goal_norm_above_01_in_forced_arms": p3,
        "arm0_score_std": arm0_score_std,
        "arm2_score_std": arm2_score_std,
        "arm2_goal_component_range": arm2_goal_rng,
        "arm3_goal_component_range": arm3_goal_rng,
        "arm2_z_goal_norm_mean": arm2_z_goal_norm,
        "arm3_z_goal_norm_mean": arm3_z_goal_norm,
    }


def main(dry_run: bool = False):
    print(f"[{EXPERIMENT_TYPE}] starting dry_run={dry_run}", flush=True)
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    train_eps = DRY_RUN_TRAIN_EPISODES if dry_run else TRAIN_EPISODES
    eval_eps = DRY_RUN_EVAL_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    rows: List[Dict] = []
    t0 = __import__("time").time()

    for seed in seeds:
        for arm in ARMS:
            row = _run_arm_seed(arm, seed, train_eps, eval_eps, steps)
            rows.append(row)
            verdict = "PASS" if (
                row["z_goal_norm_mean"] > 0.1 or not arm["force_z_goal"]
            ) else "check"
            print(
                f"  result seed={seed} arm={arm['arm']}"
                f" z_goal_norm={row['z_goal_norm_mean']:.4f}"
                f" score_std={row['candidate_score_std_mean']:.6f}"
                f" score_rng={row['candidate_score_range_mean']:.6f}"
                f" goal_rng={row['candidate_goal_component_range_mean']:.6f}"
                f" uniq_ac={row['candidate_unique_action_classes_mean']:.2f}"
                f" act_ent={row['action_class_entropy']:.3f}"
                f" verdict: {verdict}",
                flush=True,
            )

    agg = _aggregate(rows)
    acceptance = _evaluate(rows, agg)
    elapsed = __import__("time").time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] aggregate summary:", flush=True)
    for arm_name, arm_agg in agg.items():
        print(
            f"  {arm_name}:"
            f" z_goal_norm={arm_agg['z_goal_norm_mean']:.4f}"
            f" score_std={arm_agg['candidate_score_std_mean']:.6f}"
            f" score_rng={arm_agg['candidate_score_range_mean']:.6f}"
            f" goal_rng={arm_agg['candidate_goal_component_range_mean']:.6f}"
            f" act_ent={arm_agg['action_class_entropy_mean']:.3f}",
            flush=True,
        )
    print(f"[{EXPERIMENT_TYPE}] acceptance={acceptance}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(
            f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest written.",
            flush=True,
        )
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "WPC Rung 1: forced z_goal score-margin diagnostic. "
            "Non-contributory by design. "
            "Interpretation grid: "
            "PASS-all -> E3 goal-scoring seam live; monostrategy is primary blocker; "
            "proceed to WPC Rung 2 (break monostrategy). "
            "P1-fail -> goal component too small to move scores even with forced z_goal; "
            "goal_weight or z_goal injection strength inadequate. "
            "P2-fail -> goal_weight doubling has no differential effect; "
            "suspect score swamping by base/harm costs. "
            "P3-fail -> forced injection did not produce z_goal.norm > 0.1; "
            "check is_active() gate and _z_goal assignment. "
            "All-fail -> deeper instrumentation needed."
        ),
        "elapsed_seconds": elapsed,
        "seeds": seeds,
        "train_episodes": train_eps,
        "eval_episodes": eval_eps,
        "steps_per_episode": steps,
        "arms": list(agg.values()),
        "per_seed_per_arm": rows,
        "acceptance": acceptance,
        "wpc_context": {
            "wpc_rung": 1,
            "prior_exq_559a": "goal stream signals live",
            "prior_exq_560": "score rank-constant due to monostrategy",
            "prior_exq_561": "diversity modules cannot break monostrategy",
            "design": (
                "Force z_goal = normalize(z_world) at every eval step "
                "to bypass organic seeding and test E3 scoring seam directly."
            ),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if dry_run := args.dry_run:
        sys.exit(0 if result == 0 else 1)
    _outcome, _out_path = result
    emit_outcome(outcome=_outcome, manifest_path=_out_path)
    sys.exit(0)
