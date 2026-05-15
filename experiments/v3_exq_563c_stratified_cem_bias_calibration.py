#!/opt/local/bin/python3
"""V3-EXQ-563c -- stratified CEM and E3 score-bias calibration diagnostic.

Successor to V3-EXQ-563b. Tests three new support-preserving CEM flags
(support_preserving_stratified_elites, support_preserving_ao_std_floor,
support_preserving_per_class_quota) and the new E3 score-bias scale
diagnostics (normalize_score_bias_to_e3_range, last_score_diagnostics).

Seven arms:
  ARM_0 -- normal CEM baseline (no support mechanism)
  ARM_1 -- current support-preserving CEM (pre-563c flags)
  ARM_2 -- stratified SP-CEM (new stratified_elites flag)
  ARM_3 -- stratified + ao_std_floor=0.2
  ARM_4 -- ARM_3 + forced action-class-0 bias
  ARM_5 -- ARM_3 + normalize_score_bias_to_e3_range + forced bias
  ARM_6 -- scaffold reference (use_action_class_scaffold_candidates)

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_563c_stratified_cem_bias_calibration.py --dry-run
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


EXPERIMENT_TYPE = "v3_exq_563c_stratified_cem_bias_calibration"
QUEUE_ID = "V3-EXQ-563c"
SUPERSEDES_RUN_ID = "v3_exq_563b_candidate_support_repair"
SUPERSEDES_QUEUE_ID = "V3-EXQ-563b"
CLAIM_IDS: List[str] = ["ARC-065", "MECH-320"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43]
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 1
DRY_RUN_STEPS = 50

SUPPORT_ENTROPY_FLOOR = 1e-6
WEAK_FORCED_BIAS = -10.0
STD_FLOOR = 0.2


def _bias_vector(action_dim: int, action0_bias: float) -> List[float]:
    values = [0.0 for _ in range(action_dim)]
    values[0] = float(action0_bias)
    return values


ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_normal_cem",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": False,
        "support_preserving_stratified_elites": False,
        "support_preserving_ao_std_floor": 0.0,
        "normalize_score_bias_to_e3_range": False,
        "forced_action0_bias": None,
    },
    {
        "arm": "ARM_1_current_support_preserving_cem",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": False,
        "support_preserving_ao_std_floor": 0.0,
        "normalize_score_bias_to_e3_range": False,
        "forced_action0_bias": None,
    },
    {
        "arm": "ARM_2_stratified_support_preserving_cem",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": 0.0,
        "normalize_score_bias_to_e3_range": False,
        "forced_action0_bias": None,
    },
    {
        "arm": "ARM_3_stratified_plus_std_floor",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "normalize_score_bias_to_e3_range": False,
        "forced_action0_bias": None,
    },
    {
        "arm": "ARM_4_stratified_std_floor_forced_bias",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "normalize_score_bias_to_e3_range": False,
        "forced_action0_bias": WEAK_FORCED_BIAS,
    },
    {
        "arm": "ARM_5_stratified_std_floor_bias_normalisation",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": True,
        "support_preserving_stratified_elites": True,
        "support_preserving_ao_std_floor": STD_FLOOR,
        "normalize_score_bias_to_e3_range": True,
        "forced_action0_bias": WEAK_FORCED_BIAS,
    },
    {
        "arm": "ARM_6_scaffold_reference",
        "use_action_class_scaffold_candidates": True,
        "use_support_preserving_cem": False,
        "support_preserving_stratified_elites": False,
        "support_preserving_ao_std_floor": 0.0,
        "normalize_score_bias_to_e3_range": False,
        "forced_action0_bias": None,
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
    forced: Optional[List[float]] = None
    if arm.get("forced_action0_bias") is not None:
        forced = _bias_vector(env.action_dim, float(arm["forced_action0_bias"]))
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        use_action_class_scaffold_candidates=bool(
            arm.get("use_action_class_scaffold_candidates", False)
        ),
        use_support_preserving_cem=bool(
            arm.get("use_support_preserving_cem", False)
        ),
        support_preserving_stratified_elites=bool(
            arm.get("support_preserving_stratified_elites", False)
        ),
        support_preserving_ao_std_floor=float(
            arm.get("support_preserving_ao_std_floor", 0.0)
        ),
        normalize_score_bias_to_e3_range=bool(
            arm.get("normalize_score_bias_to_e3_range", False)
        ),
        support_preserving_min_first_action_classes=2,
        forced_score_bias_per_class=forced,
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


def _round_or_none(value: Optional[float]) -> Optional[float]:
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
    scaffold_added_total = 0
    total_steps = 0

    # E3 score-bias diagnostics
    e3_raw_score_range_values: List[float] = []
    e3_raw_score_std_values: List[float] = []
    score_bias_abs_mean_values: List[float] = []
    score_bias_range_values: List[float] = []
    score_bias_to_raw_range_ratio_values: List[float] = []
    normalize_score_bias_active_count = 0
    e3_diag_steps = 0
    selected_rank_before_bias: List[float] = []
    selected_rank_after_bias: List[float] = []

    def on_action(*, agent, action, step, **_kw):  # type: ignore[no-untyped-def]
        nonlocal support_preserving_active_steps
        nonlocal scaffold_added_total
        nonlocal normalize_score_bias_active_count
        nonlocal e3_diag_steps

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
            scaffold_added_total += int(
                hdiag.get("action_class_scaffold_candidates_added", 0)
            )

        # E3 score diagnostics from last_score_diagnostics
        e3_selector = getattr(agent, "e3", None)
        if e3_selector is not None:
            e3_diag = getattr(e3_selector, "last_score_diagnostics", None)
            if e3_diag:
                e3_diag_steps += 1
                e3_raw_score_range_values.append(
                    float(e3_diag.get("e3_raw_score_range_mean", 0.0))
                )
                e3_raw_score_std_values.append(
                    float(e3_diag.get("e3_raw_score_std_mean", 0.0))
                )
                score_bias_abs_mean_values.append(
                    float(e3_diag.get("score_bias_abs_mean", 0.0))
                )
                score_bias_range_values.append(
                    float(e3_diag.get("score_bias_range_mean", 0.0))
                )
                score_bias_to_raw_range_ratio_values.append(
                    float(e3_diag.get("score_bias_to_raw_range_ratio", 0.0))
                )
                if bool(e3_diag.get("normalize_score_bias_active", False)):
                    normalize_score_bias_active_count += 1
                rank_before = e3_diag.get("selected_candidate_rank_before_bias", -1)
                rank_after = e3_diag.get("selected_candidate_rank_after_bias", -1)
                if rank_before is not None and int(rank_before) >= 0:
                    selected_rank_before_bias.append(float(rank_before))
                if rank_after is not None and int(rank_after) >= 0:
                    selected_rank_after_bias.append(float(rank_after))

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
        print(
            f"  [eval] arm={arm['arm']} seed={seed} ep {ep + 1}/{episodes}",
            flush=True,
        )

    action_total = sum(action_counts.values())
    candidate_support_ok = bool(
        unique_candidate_classes
        and min(unique_candidate_classes) >= 2.0
        and _mean(candidate_entropies, 0.0) > SUPPORT_ENTROPY_FLOOR
    )
    normalize_score_bias_active_frac = (
        normalize_score_bias_active_count / max(e3_diag_steps, 1)
    )
    forced_active = (
        arm.get("forced_action0_bias") is not None
        and abs(float(arm.get("forced_action0_bias", 0.0))) > 0.0
    )
    bias_ab_mean = _mean(score_bias_abs_mean_values, 0.0) or 0.0
    bias_active = bool(forced_active and bias_ab_mean > 0.0)

    return {
        "arm": arm["arm"],
        "seed": seed,
        "total_steps": int(total_steps),
        "candidate_support_ok": candidate_support_ok,
        "selected_action_class_entropy": round(_entropy(action_counts), 6),
        "action_0_fraction": round(
            action_counts.get(0, 0) / action_total if action_total else 0.0,
            6,
        ),
        "unique_actions_taken": int(len(action_counts)),
        "action_counts": dict(sorted(action_counts.items())),
        "candidate_unique_first_action_classes_mean": _round_or_none(
            _mean(unique_candidate_classes, None)
        ),
        "candidate_unique_first_action_classes_min": _round_or_none(
            min(unique_candidate_classes) if unique_candidate_classes else None
        ),
        "candidate_first_action_entropy_mean": _round_or_none(
            _mean(candidate_entropies, None)
        ),
        "candidate_first_action_entropy_min": _round_or_none(
            min(candidate_entropies) if candidate_entropies else None
        ),
        "candidate_first_action_counts": dict(
            sorted(candidate_first_action_counts.items())
        ),
        "support_preserving_active_steps": int(support_preserving_active_steps),
        "scaffold_candidates_added_total": int(scaffold_added_total),
        # E3 score diagnostics
        "e3_diag_steps": int(e3_diag_steps),
        "e3_raw_score_range_mean": _round_or_none(_mean(e3_raw_score_range_values, None)),
        "e3_raw_score_std_mean": _round_or_none(_mean(e3_raw_score_std_values, None)),
        "score_bias_abs_mean": _round_or_none(_mean(score_bias_abs_mean_values, None)),
        "score_bias_range_mean": _round_or_none(_mean(score_bias_range_values, None)),
        "score_bias_to_raw_range_ratio_mean": _round_or_none(
            _mean(score_bias_to_raw_range_ratio_values, None)
        ),
        "normalize_score_bias_active_fraction": round(
            normalize_score_bias_active_frac, 6
        ),
        "normalize_score_bias_active_count": int(normalize_score_bias_active_count),
        "selected_rank_before_bias_mean": _round_or_none(
            _mean(selected_rank_before_bias, None)
        ),
        "selected_rank_after_bias_mean": _round_or_none(
            _mean(selected_rank_after_bias, None)
        ),
        "forced_bias_active": bias_active,
    }


def _arm_rows(rows: List[Dict[str, Any]], arm_name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("arm") == arm_name]


def _mean_key(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return float(_mean(values, default) or default)


def _summarize(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm_names = [arm["arm"] for arm in ARMS]
    arm_means: Dict[str, Dict[str, Any]] = {}
    for name in arm_names:
        rows = _arm_rows(arm_results, name)
        arm_means[name] = {
            "action_0_fraction": round(_mean_key(rows, "action_0_fraction"), 6),
            "candidate_unique_first_action_classes_mean": round(
                _mean_key(rows, "candidate_unique_first_action_classes_mean"), 6
            ),
            "candidate_unique_first_action_classes_min": round(
                min(
                    row.get("candidate_unique_first_action_classes_min") or 0.0
                    for row in rows
                ) if rows else 0.0,
                6,
            ),
            "candidate_first_action_entropy_mean": round(
                _mean_key(rows, "candidate_first_action_entropy_mean"), 6
            ),
            "support_preserving_active_steps": int(
                sum(row.get("support_preserving_active_steps", 0) for row in rows)
            ),
            "scaffold_candidates_added_total": int(
                sum(row.get("scaffold_candidates_added_total", 0) for row in rows)
            ),
            "e3_raw_score_range_mean": round(
                _mean_key(rows, "e3_raw_score_range_mean"), 6
            ),
            "score_bias_abs_mean": round(_mean_key(rows, "score_bias_abs_mean"), 6),
            "score_bias_to_raw_range_ratio_mean": round(
                _mean_key(rows, "score_bias_to_raw_range_ratio_mean"), 6
            ),
            "normalize_score_bias_active_fraction": round(
                _mean_key(rows, "normalize_score_bias_active_fraction"), 6
            ),
            "selected_rank_before_bias_mean": round(
                _mean_key(rows, "selected_rank_before_bias_mean"), 6
            ),
            "selected_rank_after_bias_mean": round(
                _mean_key(rows, "selected_rank_after_bias_mean"), 6
            ),
        }

    arm0 = arm_means["ARM_0_normal_cem"]
    arm1 = arm_means["ARM_1_current_support_preserving_cem"]
    arm2 = arm_means["ARM_2_stratified_support_preserving_cem"]
    arm3 = arm_means["ARM_3_stratified_plus_std_floor"]
    arm4 = arm_means["ARM_4_stratified_std_floor_forced_bias"]
    arm5 = arm_means["ARM_5_stratified_std_floor_bias_normalisation"]
    arm6 = arm_means["ARM_6_scaffold_reference"]

    # P1: stratified CEM (ARM_2) achieves >= ARM_1 candidate support
    p1 = bool(
        arm2["candidate_unique_first_action_classes_mean"]
        >= arm1["candidate_unique_first_action_classes_mean"] - 0.1
        and arm2["candidate_unique_first_action_classes_min"] >= 2
    )

    # P2: std_floor arm (ARM_3) maintains diversity min >= 2
    p2 = bool(arm3["candidate_unique_first_action_classes_min"] >= 2)

    # P3: forced bias arm (ARM_4) registers non-zero bias in E3 score diagnostics
    p3 = bool(arm4["score_bias_abs_mean"] > 0.0)

    # P4: normalisation arm (ARM_5) shows normalize_score_bias_active fires
    p4 = bool(arm5["normalize_score_bias_active_fraction"] > 0.0)

    # P5: scaffold (ARM_6) achieves scaffold_candidates_added_total > 0
    p5 = bool(arm6["scaffold_candidates_added_total"] > 0)

    # P6: last_score_diagnostics populated in all arms (e3_raw_score_range_mean != None)
    p6 = all(
        arm_means[name]["e3_raw_score_range_mean"] is not None
        for name in arm_names
        if _arm_rows(arm_results, name)
    )

    return {
        "arm_means": arm_means,
        "p1_stratified_cem_maintains_support": p1,
        "p2_std_floor_maintains_diversity_min": p2,
        "p3_forced_bias_registers_in_e3_diagnostics": p3,
        "p4_bias_normalisation_activates": p4,
        "p5_scaffold_adds_candidates": p5,
        "p6_last_score_diagnostics_populated_all_arms": p6,
        "interpretation_note": (
            "Diagnostic calibration: tests 563c new CEM flags and E3 score-bias "
            "diagnostics. P1-P6 are wiring checks, not behavioural evidence."
        ),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Arm {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps)
            arm_results.append(cell)

    summary = _summarize(arm_results)

    all_finite = all(
        math.isfinite(float(row.get("selected_action_class_entropy", 0.0)))
        for row in arm_results
    )
    outcome = "PASS" if all_finite else "FAIL"

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
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Diagnostic calibration run; tests 563c CEM flags and E3 bias "
            "diagnostics. Do not promote behavioural claims from this result alone."
        ),
        "evidence_direction_per_claim": {
            "ARC-065": "non_contributory",
            "MECH-320": "non_contributory",
        },
        "supersedes": SUPERSEDES_RUN_ID,
        "supersedes_queue_id": SUPERSEDES_QUEUE_ID,
        "dry_run": dry_run,
        "config": {
            "seeds": seeds,
            "eval_episodes": episodes,
            "steps_per_episode": steps,
            "weak_forced_bias": WEAK_FORCED_BIAS,
            "support_entropy_floor": SUPPORT_ENTROPY_FLOOR,
            "std_floor": STD_FLOOR,
            "arms": [arm["arm"] for arm in ARMS],
        },
        "acceptance_criteria": {
            "P1_stratified_cem_maintains_support": summary[
                "p1_stratified_cem_maintains_support"
            ],
            "P2_std_floor_maintains_diversity_min": summary[
                "p2_std_floor_maintains_diversity_min"
            ],
            "P3_forced_bias_registers_in_e3_diagnostics": summary[
                "p3_forced_bias_registers_in_e3_diagnostics"
            ],
            "P4_bias_normalisation_activates": summary[
                "p4_bias_normalisation_activates"
            ],
            "P5_scaffold_adds_candidates": summary["p5_scaffold_adds_candidates"],
            "P6_last_score_diagnostics_populated_all_arms": summary[
                "p6_last_score_diagnostics_populated_all_arms"
            ],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
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
        description="V3-EXQ-563c stratified CEM and E3 score-bias calibration"
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
