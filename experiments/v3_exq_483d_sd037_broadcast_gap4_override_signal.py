#!/opt/local/bin/python3
"""
V3-EXQ-483d: SD-037 broadcast override GAP-4 tier-1 retest with PAG/override_signal C2.

Supersedes V3-EXQ-483c. Root cause of 483c failure: C2 measured agent.dacc which was
None in all arms (use_dacc omitted from arm configs); C3 hit approach_commit_rate=1.0
ceiling in OFF_OFF baseline.

C2 fix: override_signal_nonzero_steps -- count steps where
  agent.broadcast_override.override_signal > 1e-3 in the ON_ON arm.
  OFF arms have broadcast_override=None so signal is 0 by construction.
C3 fix: goal_norm_peak delta vs baseline (ON_ON vs OFF_OFF). Has headroom because
  SD-037 override_goal_seeding_gain=2.0 doubles seeding amplification when
  override_signal > 0, which is structurally 0 in OFF_OFF (no BroadcastOverrideRegulator).
Cluster fix: use_dacc=True added explicitly to all arm extra_config (library
  build_config already sets it for gap4_operating=True arms; explicit for clarity).

claim_ids: [SD-037, MECH-280, MECH-281]
experiment_purpose: evidence
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np
import torch

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness, StepHooks  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    ArmSpec,
    ENV_FISHTANK_KWARGS,
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    TIER1_APPROACH_COMMIT_MIN,
    TIER1_CUE_FIRES_MIN,
    TIER1_GOAL_ACTIVE_FRAC_MIN,
    TIER1_SEEDS_PASS_MIN,
    WARMUP_EPISODES_DEFAULT,
    _approach_commit,
    _dacc_bias_norm,
    build_config,
    make_env,
    warmup_train,
)
from ree_core.agent import REEAgent

EXPERIMENT_TYPE = "v3_exq_483d_sd037_broadcast_gap4_override_signal"
QUEUE_ID = "V3-EXQ-483d"
CLAIM_IDS = ["SD-037", "MECH-280", "MECH-281"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds
TIER1_OVERRIDE_SIGNAL_MIN = 10    # steps in ON_ON where override_signal > 1e-3
TIER1_GOAL_NORM_PEAK_DELTA = 0.01  # ON_ON goal_norm_peak must exceed OFF_OFF by this

ARMS = [
    ArmSpec("OFF_OFF", gap4_operating=True, use_gabaergic_decay=False,
            use_broadcast_override=False, extra_config={"use_dacc": True}),
    ArmSpec("ON_OFF",  gap4_operating=True, use_gabaergic_decay=True,
            use_pag_freeze_gate=True, use_broadcast_override=False,
            extra_config={"use_dacc": True}),
    ArmSpec("OFF_ON",  gap4_operating=True, use_gabaergic_decay=False,
            use_broadcast_override=True, extra_config={"use_dacc": True}),
    ArmSpec("ON_ON",   gap4_operating=True, use_gabaergic_decay=True,
            use_pag_freeze_gate=True, use_broadcast_override=True,
            extra_config={"use_dacc": True}),
]
GAP4_ARM = "ON_ON"
BASE_ARM = "OFF_OFF"

SEEDS = SEEDS_DEFAULT             # [42, 7, 19]
WARMUP_EPISODES = WARMUP_EPISODES_DEFAULT   # 50
EVAL_EPISODES = EVAL_EPISODES_DEFAULT       # 10
STEPS_PER_EPISODE = STEPS_PER_EPISODE_DEFAULT  # 200


def _override_signal_value(agent: REEAgent) -> float:
    bo = getattr(agent, "broadcast_override", None)
    if bo is None:
        return 0.0
    return float(getattr(bo, "override_signal", 0.0))


def eval_tier1_483d(
    agent: REEAgent,
    env,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    arm_label: str,
) -> Dict[str, Any]:
    """Standard tier-1 eval extended with override_signal_nonzero_steps for C2."""
    metrics: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "approach_commit_steps": 0,
        "total_eval_steps": 0,
        "dacc_bias_nonzero_steps": 0,
        "override_signal_nonzero_steps": 0,
        "bridge_cue_fires": 0,
        "bridge_write_fires": 0,
        "goal_active_steps": 0,
        "resource_contacts": 0,
        "action_counts": {},
    }

    def on_post_step(*, agent, latent, action, obs_dict, ticks, step, **kwargs) -> None:
        metrics["total_eval_steps"] += 1
        if _approach_commit(agent):
            metrics["approach_commit_steps"] += 1
        if _dacc_bias_norm(agent) > 1e-6:
            metrics["dacc_bias_nonzero_steps"] += 1
        if _override_signal_value(agent) > 1e-3:
            metrics["override_signal_nonzero_steps"] += 1
        if agent.goal_state is not None and agent.goal_state.is_active():
            metrics["goal_active_steps"] += 1
        br = getattr(agent, "mech295_bridge", None)
        if br is not None:
            metrics["bridge_cue_fires"] = int(getattr(br, "_n_cue_fires", 0))
            metrics["bridge_write_fires"] = int(getattr(br, "_n_write_fires", 0))

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        if getattr(agent, "mech295_bridge", None) is not None:
            agent.mech295_bridge._n_cue_fires = 0
            agent.mech295_bridge._n_write_fires = 0

        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            ttype = result.info.get("transition_type", "none")
            if ttype == "resource":
                metrics["resource_contacts"] += 1
            aidx = int(result.action.argmax(dim=-1).item())
            ac = metrics["action_counts"]
            ac[aidx] = ac.get(aidx, 0) + 1
            obs_dict = result.next_obs_dict
            if result.done:
                break

    total = max(1, int(metrics["total_eval_steps"]))
    metrics["approach_commit_rate"] = float(metrics["approach_commit_steps"]) / total
    metrics["goal_active_fraction"] = float(metrics["goal_active_steps"]) / total
    metrics["action_counts"] = {str(k): int(v) for k, v in metrics["action_counts"].items()}
    if agent.goal_state is not None:
        metrics["goal_norm_peak"] = float(getattr(agent.goal_state, "_goal_norm_peak", 0.0))
    else:
        metrics["goal_norm_peak"] = 0.0
    return metrics


def tier1_seed_pass_483d(metrics: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "C1_cue_fires": int(metrics.get("bridge_cue_fires", 0)) >= TIER1_CUE_FIRES_MIN,
        "C2_override_signal": int(metrics.get("override_signal_nonzero_steps", 0)) >= TIER1_OVERRIDE_SIGNAL_MIN,
        "C3_approach_commit": int(metrics.get("approach_commit_steps", 0)) >= TIER1_APPROACH_COMMIT_MIN,
        "C4_goal_active": float(metrics.get("goal_active_fraction", 0.0)) >= TIER1_GOAL_ACTIVE_FRAC_MIN,
    }


def evaluate_483d_cohort(
    rows: List[Dict[str, Any]],
    *,
    gap4_arm_id: str,
    baseline_arm_id: str,
) -> Dict[str, Any]:
    """PASS when ON_ON arm clears C1-C4 in >= 2 seeds and beats OFF_OFF on goal_norm_peak."""
    gap4_rows = [r for r in rows if r.get("arm") == gap4_arm_id]
    base_rows = [r for r in rows if r.get("arm") == baseline_arm_id]

    per_seed = [tier1_seed_pass_483d(r) for r in gap4_rows]
    c1 = sum(1 for p in per_seed if p["C1_cue_fires"]) >= TIER1_SEEDS_PASS_MIN
    c2 = sum(1 for p in per_seed if p["C2_override_signal"]) >= TIER1_SEEDS_PASS_MIN
    c3_direct = sum(1 for p in per_seed if p["C3_approach_commit"]) >= TIER1_SEEDS_PASS_MIN
    c4 = sum(1 for p in per_seed if p["C4_goal_active"]) >= TIER1_SEEDS_PASS_MIN

    # C3_lift: ON_ON goal_norm_peak > OFF_OFF goal_norm_peak + TIER1_GOAL_NORM_PEAK_DELTA
    c3_lift = False
    lifts = 0
    if base_rows:
        for g in gap4_rows:
            seed = g.get("seed")
            b = next((x for x in base_rows if x.get("seed") == seed), None)
            if b is None:
                continue
            if float(g.get("goal_norm_peak", 0.0)) > float(b.get("goal_norm_peak", 0.0)) + TIER1_GOAL_NORM_PEAK_DELTA:
                lifts += 1
        c3_lift = lifts >= TIER1_SEEDS_PASS_MIN

    passed = bool(c1 and c2 and c3_direct and c4 and c3_lift)
    return {
        "pass": passed,
        "C1_cue_fires": c1,
        "C2_override_signal": c2,
        "C3_approach_commit": c3_direct,
        "C3_lift_vs_baseline": c3_lift,
        "C3_lift_count": lifts,
        "C4_goal_active": c4,
        "gap4_arm_id": gap4_arm_id,
        "baseline_arm_id": baseline_arm_id,
    }


def run_seed_arm_483d(
    seed: int,
    arm: ArmSpec,
    *,
    warmup_episodes: int = WARMUP_EPISODES,
    eval_episodes: int = EVAL_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(seed, ENV_FISHTANK_KWARGS)
    env._exq_env_kwargs = dict(ENV_FISHTANK_KWARGS)
    cfg = build_config(env, arm)
    agent = REEAgent(cfg)
    label = f"seed={seed} arm={arm.arm_id}"
    print(f"Seed {seed} Condition {arm.arm_id}", flush=True)
    total_episodes = warmup_episodes + eval_episodes
    warmup_train(
        agent,
        env,
        num_episodes=warmup_episodes,
        steps_per_episode=steps_per_episode,
        label=label,
        progress_total_episodes=total_episodes,
    )
    for ep in range(eval_episodes):
        if (ep + 1) == eval_episodes:
            print(
                f"  [train] {label} ep {warmup_episodes + ep + 1}/{total_episodes}",
                flush=True,
            )
    metrics = eval_tier1_483d(
        agent,
        env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        seed=seed,
        arm_label=arm.arm_id,
    )
    checks = tier1_seed_pass_483d(metrics)
    passed = all(checks.values())
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    metrics["tier1_checks"] = checks
    metrics["seed_pass"] = passed
    return metrics


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> "Tuple[str, Path] | int":
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warmup = 6 if dry_run else WARMUP_EPISODES
    eval_eps = 2 if dry_run else EVAL_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        for arm in ARMS:
            rows.append(
                run_seed_arm_483d(
                    seed,
                    arm,
                    warmup_episodes=warmup,
                    eval_episodes=eval_eps,
                    steps_per_episode=steps,
                )
            )

    acceptance = evaluate_483d_cohort(rows, gap4_arm_id=GAP4_ARM, baseline_arm_id=BASE_ARM)
    outcome = "PASS" if acceptance["pass"] else "FAIL"
    per_claim = {
        "SD-037": "supports" if outcome == "PASS" else "weakens",
        "MECH-280": "unknown",
        "MECH-281": "unknown",
    }
    elapsed = time.time() - t0

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run outcome={outcome}", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
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
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": "mixed",
        "evidence_direction_per_claim": per_claim,
        "supersedes": "V3-EXQ-483c",
        "acceptance": acceptance,
        "thresholds": {
            "TIER1_OVERRIDE_SIGNAL_MIN": TIER1_OVERRIDE_SIGNAL_MIN,
            "TIER1_GOAL_NORM_PEAK_DELTA": TIER1_GOAL_NORM_PEAK_DELTA,
        },
        "per_run": rows,
        "elapsed_seconds": elapsed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path)
    sys.exit(0)
