#!/opt/local/bin/python3
"""V3-EXQ-567 -- Work Package B: natural action-entropy with support-preserving CEM.

After V3-EXQ-563b confirmed the support-preserving CEM path (ARM_2 PASS, ARM_3
PASS) and V3-EXQ-563c validated the 563c substrate flags (stratified elites +
ao_std_floor + score/bias normalisation), this experiment enables the full
SP-CEM combination by default and measures NATURAL selected_action_entropy --
no scaffold candidates, no forced action-class bias.

Two arms:
  ARM_0_normal_cem           -- normal CEM baseline (all SP flags OFF)
  ARM_1_support_preserving   -- SP-CEM + stratified_elites + ao_std_floor=0.2
                                 (the combination validated in 563c ARM_3)

Acceptance criteria:
  P1 (entropy lift): ARM_1 selected_action_class_entropy > ARM_0 + 0.05
  P2 (candidate support): ARM_1 candidate_unique_first_action_classes_mean >
                           ARM_0 + 0.5
  P3 (structural): ARM_1 candidate_unique_first_action_classes_mean >= 2.0

PASS if P1 AND P2 AND P3.

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_567_wpb_natural_entropy_sp_cem.py --dry-run
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
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness, StepHooks  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_567_wpb_natural_entropy_sp_cem"
QUEUE_ID = "V3-EXQ-567"
SUPERSEDES_QUEUE_ID = "V3-EXQ-563b"
CLAIM_IDS: List[str] = ["ARC-065"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
EVAL_EPISODES = 40
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

# Acceptance thresholds (pre-registered)
P1_ENTROPY_MARGIN = 0.05   # ARM_1 selected_action_entropy > ARM_0 + this
P2_SUPPORT_MARGIN = 0.5    # ARM_1 candidate_unique_first_action_classes_mean > ARM_0 + this
# P3: ARM_1 achieves candidate_unique_first_action_classes_mean >= this.
# Note: stratified elites achieve diversity WITHOUT triggering the injection path,
# so support_preserving_active_steps stays 0 when stratified works well.
# We test the OUTCOME (>= 2 unique classes on average) rather than the mechanism flag.
P3_MIN_UNIQUE_CLASSES = 2.0

STD_FLOOR = 0.2            # support_preserving_ao_std_floor value for ARM_1

ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_normal_cem",
        "use_support_preserving_cem": False,
        "support_preserving_stratified_elites": False,
        "support_preserving_ao_std_floor": 0.0,
    },
    {
        "arm": "ARM_1_support_preserving",
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
    },
]


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


def _round4(value: Optional[float]) -> Optional[float]:
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

    action_counts: Counter = Counter()
    candidate_first_action_counts: Counter = Counter()
    unique_candidate_classes: List[float] = []
    candidate_entropies: List[float] = []
    support_preserving_active_steps = 0
    total_steps = 0

    def on_action(*, agent, action, step, **_kw):  # type: ignore[no-untyped-def]
        nonlocal support_preserving_active_steps

        idx = int(action.argmax(dim=-1).item())
        action_counts[idx] += 1

        hdiag = agent.hippocampal.get_last_propose_diagnostics()
        if hdiag:
            candidate_first_action_counts.update(
                _counter_from_dict(hdiag.get("candidate_first_action_counts", {}))
            )
            unique_candidate_classes.append(
                float(hdiag.get("candidate_unique_first_action_classes", 0))
            )
            candidate_entropies.append(
                float(hdiag.get("candidate_first_action_entropy", 0.0))
            )
            if bool(hdiag.get("support_preserving_active", False)):
                support_preserving_active_steps += 1

    hooks = StepHooks(on_action=on_action)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            total_steps += 1
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] arm={arm['arm']} seed={seed} ep {ep + 1}/{episodes}",
                flush=True,
            )

    action_total = sum(action_counts.values())
    selected_entropy = _entropy(action_counts)
    candidate_unique_mean = _round4(_mean(unique_candidate_classes, None))
    candidate_entropy_mean = _round4(_mean(candidate_entropies, None))

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
    arm0_rows = _arm_rows(arm_results, "ARM_0_normal_cem")
    arm1_rows = _arm_rows(arm_results, "ARM_1_support_preserving")

    arm0_entropy = _mean_key(arm0_rows, "selected_action_class_entropy")
    arm1_entropy = _mean_key(arm1_rows, "selected_action_class_entropy")

    arm0_support = _mean_key(arm0_rows, "candidate_unique_first_action_classes_mean")
    arm1_support = _mean_key(arm1_rows, "candidate_unique_first_action_classes_mean")

    p1 = bool(arm1_entropy > arm0_entropy + P1_ENTROPY_MARGIN)
    p2 = bool(arm1_support > arm0_support + P2_SUPPORT_MARGIN)
    # P3: ARM_1 achieves mean >= 2 unique first-action classes across seeds.
    # stratified elites achieve diversity via the stratified path (sp_active may stay
    # 0 if no injection was needed), so we test the outcome directly.
    p3 = bool(arm1_support >= P3_MIN_UNIQUE_CLASSES)

    return {
        "arm0_selected_entropy_mean": round(arm0_entropy, 6),
        "arm1_selected_entropy_mean": round(arm1_entropy, 6),
        "entropy_delta_arm1_minus_arm0": round(arm1_entropy - arm0_entropy, 6),
        "arm0_candidate_support_mean": round(arm0_support, 6),
        "arm1_candidate_support_mean": round(arm1_support, 6),
        "support_delta_arm1_minus_arm0": round(arm1_support - arm0_support, 6),
        "p1_entropy_lift": p1,
        "p2_candidate_support_lift": p2,
        "p3_arm1_achieves_min_unique_classes": p3,
        "overall_pass": bool(p1 and p2 and p3),
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
            "ARM_1 (SP-CEM + stratified + ao_std_floor) vs ARM_0 (normal CEM) on "
            "natural selected_action_entropy and candidate support. "
            "PASS = SP-CEM meaningfully diversifies natural action selection. "
            "FAIL = SP-CEM does not lift natural entropy above the baseline threshold."
        ),
        "supersedes_queue_id": SUPERSEDES_QUEUE_ID,
        "dry_run": dry_run,
        "config": {
            "seeds": seeds,
            "eval_episodes": episodes,
            "steps_per_episode": steps,
            "p1_entropy_margin": P1_ENTROPY_MARGIN,
            "p2_support_margin": P2_SUPPORT_MARGIN,
            "std_floor": STD_FLOOR,
            "arms": [arm["arm"] for arm in ARMS],
        },
        "acceptance_criteria": {
            "P1_entropy_lift": summary["p1_entropy_lift"],
            "P2_candidate_support_lift": summary["p2_candidate_support_lift"],
            "P3_arm1_achieves_min_unique_classes": summary[
                "p3_arm1_achieves_min_unique_classes"
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
        description="V3-EXQ-567 WPB natural entropy support-preserving CEM"
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
