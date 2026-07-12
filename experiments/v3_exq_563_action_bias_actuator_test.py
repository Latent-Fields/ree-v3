#!/opt/local/bin/python3
"""V3-EXQ-563 -- Forced action-bias actuator test.

Calibration debt Work Package B follow-up. Diagnostic only -- no governance weighting.

Purpose
-------
V3-EXQ-561 produced entropy=0.0 across ALL 6 arms including ARM_3 (entropy ceiling,
noise_floor_alpha=5.0). This confirms two distinct calibration failure layers:

  Layer 1 (primary): CEM candidate collapse -- all K CEM candidates share the same
    first-action argmax, so no post-CEM scoring or temperature change can diversify.
    ARM_3 entropy=0.0 in EXQ-561 is definitive proof (raw noise cannot help if every
    candidate already starts with the same first step).

  Layer 2 (confirmed): MECH-320 v_raw sign/scale failure (Rung 2) -- update_score_receipt
    applies reward_signal = -float(score). E3 scores are large positive (~70). After
    updates: v_raw ~ -70. v_t = max(0, -70) = 0. Bias always zeros.

This 6-arm actuator test:
  1. Confirms Layer 2 (P2: ARM_1 v_t=0.0 with natural MECH-320).
  2. Confirms the v_t_floor bypass fix (P3: ARM_2 v_t>0.0 with floor=1.0).
  3. Tests forced_score_bias_per_class wiring (P1: ARM_4 action_0 fraction > ARM_0 + 10pp).
     If P1 FAILs AND ARM_0 candidate_first_action_entropy ~ 0, CEM collapse is confirmed
     as the primary bottleneck requiring a Rung 7 fix (CEM candidate diversification).

Arms
----
ARM_0_baseline:
    No modules. Action entropy floor.

ARM_1_tonic_vigor_natural:
    MECH-320 tonic vigor, natural (no v_t_floor). Expected: v_t=0. Confirms Layer 2.

ARM_2_tonic_vigor_floor:
    MECH-320 tonic vigor with v_t_floor=1.0. Expected: v_t>0. Confirms bypass fix.

ARM_3_forced_zero:
    forced_score_bias_per_class=[0.0, 0.0, 0.0, 0.0]. Sanity check: should equal ARM_0.

ARM_4_forced_nonzero:
    forced_score_bias_per_class=[-2.0, 0.0, 0.0, 0.0]. Tests bias wiring. If CEM is not
    collapsed, action_0 fraction increases vs ARM_0. CEM collapse confirmed if P1 FAILs
    and ARM_0 candidate_first_action_entropy is near zero.

ARM_5_combined:
    MECH-320 + v_t_floor=1.0 + forced_score_bias=[-2.0, 0.0, 0.0, 0.0]. Full actuator.

PASS: all arms produced finite action entropy (diagnostic ran cleanly).
Evidence direction: non_contributory for all claims (diagnostic run).

Acceptance criteria (reported even when overall outcome is PASS):
  P1: ARM_4 mean action_0 fraction > ARM_0 mean action_0 fraction + 0.10
  P2: ARM_1 mech320_last_v_t == 0.0 for ALL seeds
  P3: ARM_2 mech320_last_v_t > 0.0 for ANY seed

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_563_action_bias_actuator_test.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


EXPERIMENT_TYPE = "v3_exq_563_action_bias_actuator_test"
QUEUE_ID = "V3-EXQ-563"
CLAIM_IDS: List[str] = ["ARC-065", "MECH-320"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43]
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 1
DRY_RUN_STEPS = 50

P1_ACTION0_MARGIN = 0.10

ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_baseline",
        "use_tonic_vigor": False,
        "tonic_vigor_v_t_floor": None,
        "forced_score_bias_per_class": None,
    },
    {
        "arm": "ARM_1_tonic_vigor_natural",
        "use_tonic_vigor": True,
        "tonic_vigor_v_t_floor": None,
        "forced_score_bias_per_class": None,
    },
    {
        "arm": "ARM_2_tonic_vigor_floor",
        "use_tonic_vigor": True,
        "tonic_vigor_v_t_floor": 1.0,
        "forced_score_bias_per_class": None,
    },
    {
        "arm": "ARM_3_forced_zero",
        "use_tonic_vigor": False,
        "tonic_vigor_v_t_floor": None,
        "forced_score_bias_per_class": [0.0, 0.0, 0.0, 0.0],
    },
    {
        "arm": "ARM_4_forced_nonzero",
        "use_tonic_vigor": False,
        "tonic_vigor_v_t_floor": None,
        "forced_score_bias_per_class": [-2.0, 0.0, 0.0, 0.0],
    },
    {
        "arm": "ARM_5_combined",
        "use_tonic_vigor": True,
        "tonic_vigor_v_t_floor": 1.0,
        "forced_score_bias_per_class": [-2.0, 0.0, 0.0, 0.0],
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
    kwargs: Dict[str, Any] = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
    )
    if arm["use_tonic_vigor"]:
        kwargs["use_tonic_vigor"] = True
    if arm["tonic_vigor_v_t_floor"] is not None:
        kwargs["tonic_vigor_v_t_floor"] = arm["tonic_vigor_v_t_floor"]
    if arm["forced_score_bias_per_class"] is not None:
        kwargs["forced_score_bias_per_class"] = arm["forced_score_bias_per_class"]
    return REEConfig.from_dims(**kwargs)


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
    cand_first_action_counts: Counter = Counter()
    forced_bias_abs_vals: List[float] = []

    fsb: Optional[List[float]] = arm["forced_score_bias_per_class"]

    def on_action(*, agent, latent, action, obs_dict, ticks, step, **_kw):  # type: ignore[no-untyped-def]
        idx = int(action.argmax(dim=-1).item())
        action_counts[idx] += 1

        cands = getattr(agent, "_committed_candidates", None)
        if cands is not None and len(cands) > 0:
            try:
                first_cls = [
                    int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                    for c in cands
                ]
                cand_first_action_counts.update(first_cls)
                if fsb is not None:
                    applied = [
                        abs(fsb[ac]) if ac < len(fsb) else 0.0 for ac in first_cls
                    ]
                    forced_bias_abs_vals.append(max(applied) if applied else 0.0)
            except Exception:
                pass

    hooks = StepHooks(on_action=on_action)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)

    total_steps = 0
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
            f"  [train] arm={arm['arm']} seed={seed} ep {ep+1}/{episodes}",
            flush=True,
        )

    action_entropy = _entropy(action_counts)
    cand_entropy = _entropy(cand_first_action_counts)
    action_total = sum(action_counts.values())
    action_0_fraction = (
        action_counts.get(0, 0) / action_total if action_total > 0 else 0.0
    )
    cand_total = sum(cand_first_action_counts.values())

    metrics: Dict[str, Any] = {
        "arm": arm["arm"],
        "seed": seed,
        "total_steps": total_steps,
        "action_class_entropy": round(action_entropy, 6),
        "action_0_fraction": round(action_0_fraction, 6),
        "unique_actions_taken": len(action_counts),
        "action_counts": dict(action_counts),
        "candidate_first_action_entropy": round(cand_entropy, 6),
        "candidate_first_action_counts": dict(cand_first_action_counts),
        "candidate_samples_collected": cand_total,
        "forced_bias_abs_mean": (
            round(sum(forced_bias_abs_vals) / len(forced_bias_abs_vals), 6)
            if forced_bias_abs_vals
            else None
        ),
    }

    if arm["use_tonic_vigor"] and agent.tonic_vigor is not None:
        tv_state = agent.tonic_vigor.get_state()
        metrics["mech320_v_raw_final"] = round(float(tv_state.get("v_raw", 0.0)), 6)
        metrics["mech320_last_v_t"] = round(float(tv_state.get("last_v_t", 0.0)), 6)
        metrics["mech320_n_score_updates"] = int(
            tv_state.get("n_waking_score_updates", 0)
        )
        metrics["mech320_n_bias_calls"] = int(
            tv_state.get("n_waking_bias_calls", 0)
        )

    return metrics


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
            finite = math.isfinite(cell["action_class_entropy"])
            print(f"verdict: {'PASS' if finite else 'FAIL'}", flush=True)

    def _arm_rows(name: str) -> List[Dict[str, Any]]:
        return [r for r in arm_results if r["arm"] == name]

    def _mean_val(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else default

    arm0_rows = _arm_rows("ARM_0_baseline")
    arm1_rows = _arm_rows("ARM_1_tonic_vigor_natural")
    arm2_rows = _arm_rows("ARM_2_tonic_vigor_floor")
    arm4_rows = _arm_rows("ARM_4_forced_nonzero")

    arm0_a0f = _mean_val(arm0_rows, "action_0_fraction")
    arm4_a0f = _mean_val(arm4_rows, "action_0_fraction")

    p1_pass = bool(arm4_a0f > arm0_a0f + P1_ACTION0_MARGIN)
    p2_pass = bool(
        arm1_rows
        and all(r.get("mech320_last_v_t", -1.0) == 0.0 for r in arm1_rows)
    )
    p3_pass = bool(
        arm2_rows
        and any(r.get("mech320_last_v_t", 0.0) > 0.0 for r in arm2_rows)
    )

    arm_entropy_means: Dict[str, float] = {}
    arm_a0f_means: Dict[str, float] = {}
    arm_cand_entropy_means: Dict[str, float] = {}
    for arm in ARMS:
        rows = _arm_rows(arm["arm"])
        if rows:
            arm_entropy_means[arm["arm"]] = round(
                _mean_val(rows, "action_class_entropy"), 6
            )
            arm_a0f_means[arm["arm"]] = round(
                _mean_val(rows, "action_0_fraction"), 6
            )
            arm_cand_entropy_means[arm["arm"]] = round(
                _mean_val(rows, "candidate_first_action_entropy"), 6
            )

    arm0_cand_entropy = arm_cand_entropy_means.get("ARM_0_baseline", 0.0)
    arm0_cand_total = sum(r.get("candidate_samples_collected", 0) for r in arm0_rows)
    cem_data_collected = arm0_cand_total > 0
    cem_collapse_confirmed = bool(cem_data_collected and arm0_cand_entropy < 0.01)

    summary: Dict[str, Any] = {
        "arm_entropy_means": arm_entropy_means,
        "arm_action_0_fraction_means": arm_a0f_means,
        "arm_candidate_first_action_entropy_means": arm_cand_entropy_means,
        "p1_forced_bias_changes_action_dist": p1_pass,
        "p1_arm0_action_0_fraction": round(arm0_a0f, 6),
        "p1_arm4_action_0_fraction": round(arm4_a0f, 6),
        "p1_margin": round(arm4_a0f - arm0_a0f, 6),
        "p2_tonic_vigor_natural_v_t_zero": p2_pass,
        "p3_tonic_vigor_floor_v_t_nonzero": p3_pass,
        "cem_data_collected": cem_data_collected,
        "cem_collapse_confirmed": cem_collapse_confirmed,
        "arm0_candidate_first_action_entropy": round(arm0_cand_entropy, 6),
    }

    for arm_name in (
        "ARM_1_tonic_vigor_natural",
        "ARM_2_tonic_vigor_floor",
        "ARM_5_combined",
    ):
        rows = _arm_rows(arm_name)
        if rows and "mech320_last_v_t" in rows[0]:
            summary[f"{arm_name}_v_t_mean"] = round(
                _mean_val(rows, "mech320_last_v_t"), 6
            )
            summary[f"{arm_name}_v_raw_mean"] = round(
                _mean_val(rows, "mech320_v_raw_final"), 6
            )

    all_finite = all(math.isfinite(r["action_class_entropy"]) for r in arm_results)
    outcome = "PASS" if all_finite else "FAIL"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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
        "evidence_direction_per_claim": {
            "ARC-065": "non_contributory",
            "MECH-320": "non_contributory",
        },
        "dry_run": dry_run,
        "config": {
            "seeds": seeds,
            "eval_episodes": episodes,
            "steps_per_episode": steps,
            "n_arms": len(ARMS),
            "p1_action0_margin_threshold": P1_ACTION0_MARGIN,
        },
        "acceptance_criteria": {
            "P1_forced_bias_changes_action_dist": p1_pass,
            "P2_tonic_vigor_natural_v_t_zero": p2_pass,
            "P3_tonic_vigor_floor_v_t_nonzero": p3_pass,
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
        print("Dry run -- manifest not written.", flush=True)
        out_path = Path("/dev/null")

    print(f"Outcome: {outcome}", flush=True)
    print(f"P1 forced bias changes action dist: {p1_pass}", flush=True)
    print(f"  ARM_0 action_0 frac: {arm0_a0f:.4f}", flush=True)
    print(f"  ARM_4 action_0 frac: {arm4_a0f:.4f}", flush=True)
    print(f"P2 tonic_vigor_natural v_t=0: {p2_pass}", flush=True)
    print(f"P3 tonic_vigor_floor v_t>0:   {p3_pass}", flush=True)
    print(f"CEM data collected: {cem_data_collected}", flush=True)
    print(f"CEM collapse confirmed: {cem_collapse_confirmed}", flush=True)
    print(
        f"ARM_0 candidate_first_action_entropy: {arm0_cand_entropy:.6f}",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-563 forced action-bias actuator test"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run (1 seed, 1 episode, 50 steps); no manifest written.",
    )
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    _manifest_path = result.get("manifest_path", str(Path("/dev/null")))

    emit_outcome(
        outcome=_outcome,
        manifest_path=_manifest_path,
    )
    sys.exit(0)
