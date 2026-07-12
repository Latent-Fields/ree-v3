#!/opt/local/bin/python3
"""V3-EXQ-563b -- candidate-support repair diagnostic.

Successor to V3-EXQ-563 / 563a. This script does not add a new high-level
drive. It compares normal CEM, diagnostic scaffold support, and a default-off
support-preserving CEM repair at the candidate-support boundary.

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_563b_candidate_support_repair.py --dry-run
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


EXPERIMENT_TYPE = "v3_exq_563b_candidate_support_repair"
QUEUE_ID = "V3-EXQ-563b"
SUPERSEDES_RUN_ID = "v3_exq_563a_action_bias_scaffold_actuator_test_20260514T194658Z_v3"
SUPERSEDES_QUEUE_ID = "V3-EXQ-563a"
CLAIM_IDS: List[str] = ["ARC-065", "MECH-320"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43]
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200
DOSE_EPISODES = 10

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 1
DRY_RUN_STEPS = 50
DRY_RUN_DOSE_EPISODES = 1

SUPPORT_ENTROPY_FLOOR = 1e-6
ACTION0_MARGIN = 0.10
WEAK_FORCED_BIAS = -10.0
DOSE_BIASES = [-100.0, -30.0, -10.0, -3.0, -1.0, -0.3, -0.1, 0.0]


def _bias_vector(action_dim: int, action0_bias: float) -> List[float]:
    values = [0.0 for _ in range(action_dim)]
    values[0] = float(action0_bias)
    return values


ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_normal_cem",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": False,
        "forced_action0_bias": None,
    },
    {
        "arm": "ARM_1_scaffold_candidates",
        "use_action_class_scaffold_candidates": True,
        "use_support_preserving_cem": False,
        "forced_action0_bias": None,
    },
    {
        "arm": "ARM_2_support_preserving_cem",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": True,
        "forced_action0_bias": None,
    },
    {
        "arm": "ARM_3_support_preserving_cem_plus_weak_forced_bias",
        "use_action_class_scaffold_candidates": False,
        "use_support_preserving_cem": True,
        "forced_action0_bias": WEAK_FORCED_BIAS,
    },
    {
        "arm": "ARM_4_scaffold_plus_weak_forced_bias",
        "use_action_class_scaffold_candidates": True,
        "use_support_preserving_cem": False,
        "forced_action0_bias": WEAK_FORCED_BIAS,
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
    forced = None
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


def _mean(values: Iterable[float], default: Optional[float] = 0.0) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return default
    return sum(vals) / len(vals)


def _range(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float((values.max() - values.min()).detach().cpu().item())


def _rank_of_index(values: torch.Tensor, index: int) -> int:
    ranked = sorted(
        range(int(values.numel())),
        key=lambda i: (float(values[i].detach().cpu().item()), i),
    )
    try:
        return int(ranked.index(int(index)))
    except ValueError:
        return -1


def _counter_from_dict(data: Dict[Any, Any]) -> Counter:
    counter: Counter = Counter()
    for key, value in data.items():
        counter[int(key)] += int(value)
    return counter


def _round_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


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
    selected_index_counts: Counter = Counter()
    preflight_status_counts: Counter = Counter()

    unique_candidate_classes: List[float] = []
    candidate_entropies: List[float] = []
    forced_bias_abs_values: List[float] = []
    e3_raw_score_ranges: List[float] = []
    e3_biased_score_ranges: List[float] = []
    score_bias_ranges: List[float] = []
    score_bias_abs_means: List[float] = []
    selected_rank_before_bias: List[float] = []
    selected_rank_after_bias: List[float] = []
    support_preserving_active_steps = 0
    scaffold_added_total = 0
    total_steps = 0

    def on_action(*, agent, action, step, **_kw):  # type: ignore[no-untyped-def]
        nonlocal support_preserving_active_steps
        nonlocal scaffold_added_total
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

        preflight = getattr(agent, "_last_candidate_support_preflight", {}) or {}
        preflight_status_counts[str(preflight.get("preflight_status", "UNKNOWN"))] += 1
        fmean = preflight.get("forced_bias_abs_mean", None)
        if fmean is not None:
            forced_bias_abs_values.append(float(fmean))

        result = getattr(agent, "_last_e3_selection_result", None)
        if result is not None:
            selected_index_counts[int(result.selected_index)] += 1
            biased_scores = result.scores.detach().float().cpu()
            bias = getattr(agent, "_last_e3_score_bias", None)
            if bias is not None and bias.numel() == biased_scores.numel():
                bias_cpu = bias.detach().float().cpu()
                raw_scores = biased_scores - bias_cpu
                score_bias_ranges.append(_range(bias_cpu))
                score_bias_abs_means.append(
                    float(bias_cpu.abs().mean().item())
                )
            else:
                raw_scores = biased_scores
                score_bias_ranges.append(0.0)
                score_bias_abs_means.append(0.0)
            e3_raw_score_ranges.append(_range(raw_scores))
            e3_biased_score_ranges.append(_range(biased_scores))
            selected_rank_before_bias.append(
                float(_rank_of_index(raw_scores, int(result.selected_index)))
            )
            selected_rank_after_bias.append(
                float(_rank_of_index(biased_scores, int(result.selected_index)))
            )

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
    candidate_total = sum(candidate_first_action_counts.values())
    candidate_support_ok = bool(
        unique_candidate_classes
        and min(unique_candidate_classes) >= 2.0
        and _mean(candidate_entropies, 0.0) > SUPPORT_ENTROPY_FLOOR
    )
    forced_active = (
        arm.get("forced_action0_bias") is not None
        and abs(float(arm.get("forced_action0_bias", 0.0))) > 0.0
    )
    forced_bias_abs_mean = _mean(forced_bias_abs_values, None)
    forced_ok = bool(
        not forced_active
        or (forced_bias_abs_mean is not None and forced_bias_abs_mean > 0.0)
    )
    interpretation = (
        "contributory_diagnostic"
        if candidate_support_ok and forced_ok
        else "NOT_RUN: candidate_support_collapse"
    )

    return {
        "arm": arm["arm"],
        "seed": seed,
        "total_steps": int(total_steps),
        "interpretation": interpretation,
        "candidate_support_ok": candidate_support_ok,
        "forced_bias_ok": forced_ok,
        "selected_action_class_entropy": round(_entropy(action_counts), 6),
        "action_0_fraction": round(
            action_counts.get(0, 0) / action_total if action_total else 0.0,
            6,
        ),
        "unique_actions_taken": int(len(action_counts)),
        "action_counts": dict(sorted(action_counts.items())),
        "candidate_first_action_entropy_mean": _round_or_none(
            _mean(candidate_entropies, None)
        ),
        "candidate_first_action_entropy_min": _round_or_none(
            min(candidate_entropies) if candidate_entropies else None
        ),
        "candidate_unique_first_action_classes_mean": _round_or_none(
            _mean(unique_candidate_classes, None)
        ),
        "candidate_unique_first_action_classes_min": _round_or_none(
            min(unique_candidate_classes) if unique_candidate_classes else None
        ),
        "candidate_first_action_counts": dict(
            sorted(candidate_first_action_counts.items())
        ),
        "candidate_samples_collected": int(candidate_total),
        "selected_index_distribution": dict(sorted(selected_index_counts.items())),
        "forced_bias_abs_mean": _round_or_none(forced_bias_abs_mean),
        "e3_raw_score_range_mean": _round_or_none(_mean(e3_raw_score_ranges, None)),
        "e3_biased_score_range_mean": _round_or_none(
            _mean(e3_biased_score_ranges, None)
        ),
        "score_bias_range_mean": _round_or_none(_mean(score_bias_ranges, None)),
        "score_bias_abs_mean": _round_or_none(_mean(score_bias_abs_means, None)),
        "selected_candidate_rank_before_bias_mean": _round_or_none(
            _mean(selected_rank_before_bias, None)
        ),
        "selected_candidate_rank_after_bias_mean": _round_or_none(
            _mean(selected_rank_after_bias, None)
        ),
        "support_preserving_active_steps": int(support_preserving_active_steps),
        "scaffold_candidates_added_total": int(scaffold_added_total),
        "preflight_status_counts": dict(sorted(preflight_status_counts.items())),
    }


def _arm_rows(rows: List[Dict[str, Any]], arm_name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("arm") == arm_name]


def _mean_key(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return float(_mean(values, default) or default)


def _run_main_arms(
    *,
    seeds: List[int],
    episodes: int,
    steps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Arm {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps)
            rows.append(cell)
            print(f"interpretation: {cell['interpretation']}", flush=True)
    return rows


def _run_dose_response(
    *,
    seeds: List[int],
    episodes: int,
    steps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bias in DOSE_BIASES:
        arm = {
            "arm": f"DOSE_scaffold_action0_bias_{bias:g}",
            "use_action_class_scaffold_candidates": True,
            "use_support_preserving_cem": False,
            "forced_action0_bias": bias,
        }
        for seed in seeds:
            print(f"Seed {seed} Dose action0_bias={bias:g}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps)
            cell["dose_action0_bias"] = float(bias)
            rows.append(cell)
            print(f"dose interpretation: {cell['interpretation']}", flush=True)
    return rows


def _summarize(
    arm_results: List[Dict[str, Any]],
    dose_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    arm_names = [arm["arm"] for arm in ARMS]
    arm_means: Dict[str, Dict[str, Any]] = {}
    for name in arm_names:
        rows = _arm_rows(arm_results, name)
        arm_means[name] = {
            "action_0_fraction": round(_mean_key(rows, "action_0_fraction"), 6),
            "selected_action_class_entropy": round(
                _mean_key(rows, "selected_action_class_entropy"),
                6,
            ),
            "candidate_unique_first_action_classes_mean": round(
                _mean_key(rows, "candidate_unique_first_action_classes_mean"),
                6,
            ),
            "candidate_unique_first_action_classes_min": round(
                min(
                    row.get("candidate_unique_first_action_classes_min", 0.0)
                    for row in rows
                ) if rows else 0.0,
                6,
            ),
            "candidate_first_action_entropy_mean": round(
                _mean_key(rows, "candidate_first_action_entropy_mean"),
                6,
            ),
            "forced_bias_abs_mean": _round_or_none(
                _mean(
                    [
                        row.get("forced_bias_abs_mean")
                        for row in rows
                        if row.get("forced_bias_abs_mean") is not None
                    ],
                    None,
                )
            ),
            "support_preserving_active_steps": int(
                sum(row.get("support_preserving_active_steps", 0) for row in rows)
            ),
        }

    arm0 = arm_means["ARM_0_normal_cem"]
    arm1 = arm_means["ARM_1_scaffold_candidates"]
    arm2 = arm_means["ARM_2_support_preserving_cem"]
    arm3 = arm_means["ARM_3_support_preserving_cem_plus_weak_forced_bias"]
    arm4 = arm_means["ARM_4_scaffold_plus_weak_forced_bias"]

    p1 = bool(
        arm1["candidate_unique_first_action_classes_min"] >= 2
        and arm4["candidate_unique_first_action_classes_min"] >= 2
        and arm4["action_0_fraction"] > arm1["action_0_fraction"] + ACTION0_MARGIN
        and (arm4["forced_bias_abs_mean"] or 0.0) > 0.0
    )
    p2 = bool(
        arm2["candidate_unique_first_action_classes_mean"]
        > arm0["candidate_unique_first_action_classes_mean"]
        and arm2["candidate_unique_first_action_classes_min"] >= 2
    )
    p3 = bool(
        arm3["action_0_fraction"] > arm2["action_0_fraction"] + ACTION0_MARGIN
        and (arm3["forced_bias_abs_mean"] or 0.0) > 0.0
    )
    p4 = bool(p2 and not p3)
    p5 = bool(not p2)

    zero_rows = [
        row for row in dose_results
        if float(row.get("dose_action0_bias", 0.0)) == 0.0
    ]
    zero_a0 = _mean_key(zero_rows, "action_0_fraction")
    dose_summary: List[Dict[str, Any]] = []
    first_moving_bias: Optional[float] = None
    for bias in DOSE_BIASES:
        rows = [
            row for row in dose_results
            if float(row.get("dose_action0_bias", 999.0)) == float(bias)
        ]
        a0 = _mean_key(rows, "action_0_fraction")
        entry = {
            "action0_bias": float(bias),
            "action_0_fraction": round(a0, 6),
            "selected_index_distribution": {},
            "candidate_first_action_entropy_mean": round(
                _mean_key(rows, "candidate_first_action_entropy_mean"),
                6,
            ),
            "e3_raw_score_range_mean": round(
                _mean_key(rows, "e3_raw_score_range_mean"),
                6,
            ),
            "score_bias_abs_mean": round(
                _mean_key(rows, "score_bias_abs_mean"),
                6,
            ),
        }
        merged: Counter = Counter()
        for row in rows:
            merged.update(
                _counter_from_dict(row.get("selected_index_distribution", {}))
            )
        entry["selected_index_distribution"] = dict(sorted(merged.items()))
        dose_summary.append(entry)
        if (
            first_moving_bias is None
            and bias < 0.0
            and a0 > zero_a0 + ACTION0_MARGIN
        ):
            first_moving_bias = float(bias)

    if p5:
        support_status = "not_touching_live_proposal_path"
    elif p3:
        support_status = "support_and_weak_bias_move_selection"
    elif p4:
        support_status = "support_present_bias_magnitude_or_score_range_blocked"
    else:
        support_status = "support_improved_interpret_with_caution"

    return {
        "arm_means": arm_means,
        "dose_response": dose_summary,
        "dose_zero_action_0_fraction": round(zero_a0, 6),
        "dose_first_bias_moving_action0_by_margin": first_moving_bias,
        "p1_scaffold_support_and_forced_bias_action_movement": p1,
        "p2_support_preserving_cem_increases_support": p2,
        "p3_weak_forced_bias_changes_support_preserving_selection": p3,
        "p4_support_present_but_bias_not_sufficient": p4,
        "p5_support_preserving_failed_live_path": p5,
        "support_preserving_live_path_status": support_status,
        "interpretation_note": (
            "E3 score-bias seam remains confirmed by 563a; this diagnostic "
            "tests proposal support and forced-bias magnitude only."
        ),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    dose_episodes = DRY_RUN_DOSE_EPISODES if dry_run else DOSE_EPISODES

    arm_results = _run_main_arms(seeds=seeds, episodes=episodes, steps=steps)
    dose_results = _run_dose_response(
        seeds=seeds,
        episodes=dose_episodes,
        steps=steps,
    )
    summary = _summarize(arm_results, dose_results)

    all_finite = all(
        math.isfinite(float(row.get("selected_action_class_entropy", 0.0)))
        for row in arm_results + dose_results
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
            "Diagnostic calibration run; do not promote behavioural agency "
            "claims from this result alone."
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
            "dose_episodes": dose_episodes,
            "weak_forced_bias_vector": _bias_vector(5, WEAK_FORCED_BIAS),
            "dose_biases_to_action_0": list(DOSE_BIASES),
            "support_entropy_floor": SUPPORT_ENTROPY_FLOOR,
            "action0_margin": ACTION0_MARGIN,
        },
        "acceptance_criteria": {
            "P1_scaffold_support_and_forced_bias_action_movement": summary[
                "p1_scaffold_support_and_forced_bias_action_movement"
            ],
            "P2_support_preserving_cem_increases_support": summary[
                "p2_support_preserving_cem_increases_support"
            ],
            "P3_weak_forced_bias_changes_support_preserving_selection": summary[
                "p3_weak_forced_bias_changes_support_preserving_selection"
            ],
            "P4_support_present_but_bias_not_sufficient": summary[
                "p4_support_present_but_bias_not_sufficient"
            ],
            "P5_support_preserving_failed_live_path": summary[
                "p5_support_preserving_failed_live_path"
            ],
        },
        "summary": summary,
        "arm_results": arm_results,
        "dose_response_results": dose_results,
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
        print(f"{key}: {value}", flush=True)
    print(
        "Support preserving status: "
        f"{summary['support_preserving_live_path_status']}",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-563b candidate-support repair diagnostic"
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
