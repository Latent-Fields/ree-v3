#!/opt/local/bin/python3
"""V3-EXQ-530c: ARC-016 precision-to-commitment circuit, StepHarness retest.

Tier-1 Slam-Dunk retest under StepHarness. Supersedes V3-EXQ-530b which was
reclassified non_contributory 2026-05-07T23:12Z (rv pinned at precision_init=0.5
because the inline loop never called agent.update_residue(), so
e3.post_action_update() and update_running_variance() never fired).

The Q-042 regression contract (tests/contracts/test_running_variance_contract.py
landed 2026-05-08T01:39Z) locks the failure mode: rv only moves when
update_residue is called once per env step. StepHarness encodes this invariant
in its canonical sequence (step 9 of _harness.py), so this retest puts ARC-016
back on a substrate where rv is actually live.

Tests the claim: E3-derived precision (1/running_variance) modulates commit
threshold. With rv now live, the dACC score_bias path can read true precision
each tick instead of a constant 1/0.5 = 1.999996 from the pinned variable.

Two arms, both reef-enabled (SD-054 substrate; behavioural diversity attractor):
  ARM_0: use_dacc=False -- no precision-commit circuit, baseline.
  ARM_1: use_dacc=True  -- dACC reads e3.current_precision in score_bias
                           (dacc_precision_scale=500.0).

Protocol
--------
Per (arm, seed):
  Training: random policy via StepHarness; compute_prediction_loss after each
            harness.step(); Adam lr=5e-4. StepHarness internally drives sense
            -> _e1_tick -> generate_trajectories -> update_z_goal -> select_action
            -> env.step -> update_residue exactly once per env step.
  Eval:     same StepHarness loop in eval (no_grad) mode; record
            (precision, beta_elevated) at each tick.

Metrics (eval phase, per arm per seed):
  commit_rate                : fraction of steps where beta_elevated=True
  high_precision_commit_rate : commit_rate when precision > median
  low_precision_commit_rate  : commit_rate when precision <= median
  precision_commit_ratio     : high / (low + 1e-8)
  mean_precision             : average e3.current_precision
  rv_final                   : agent.e3._running_variance at end of run
                               (Q-042 sanity probe -- must NOT equal precision_init=0.5
                               for this experiment to be interpretable)

Acceptance criteria (pre-registered)
------------------------------------
  C0: rv_final differs from precision_init by > 1e-6 in BOTH arms (Q-042 contract).
      If C0 FAILs the substrate is not actually live and the experiment is
      non_contributory regardless of C1/C2/C3.
  C1: ARM_1 precision_commit_ratio > ARM_0 precision_commit_ratio.
  C2: ARM_1 precision_commit_ratio > 1.0.
  C3: ARM_1 commit_rate > 0.01 (non-degenerate).

Overall PASS = C0 AND C1 AND C2 AND C3.

claim_ids: ["ARC-016"]
experiment_purpose: evidence
supersedes: V3-EXQ-530b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402

QUEUE_ID = "V3-EXQ-530c"
EXPERIMENT_TYPE = "v3_exq_530c_arc016_precision_commit_stepharness"
CLAIM_IDS = ["ARC-016"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-530b"

N_TRAIN_EPS = 30
N_EVAL_EPS = 20
N_SEEDS = 3
GRID_SIZE = 12
STEPS_PER_EP = 200
LR = 5e-4

# Q-042 sanity probe: rv must move away from this value if the substrate is live.
PRECISION_INIT_BASELINE = 0.5
RV_DIFF_FLOOR = 1e-6

REEF_KWARGS = dict(
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
)

ARMS = [
    {"arm_id": 0, "use_dacc": False, "label": "ARM_0_no_dacc"},
    {"arm_id": 1, "use_dacc": True,  "label": "ARM_1_dacc"},
]


def _make_env(seed: int, dry_run: bool = False) -> CausalGridWorldV2:
    size = GRID_SIZE if not dry_run else 8
    return CausalGridWorldV2(
        seed=seed,
        size=size,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        **REEF_KWARGS,
    )


def _make_agent(env: CausalGridWorldV2, use_dacc: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_dacc=use_dacc,
        dacc_weight=0.5 if use_dacc else 0.0,
        dacc_precision_scale=500.0,
    )
    return REEAgent(cfg)


def _run_arm_seed(
    arm_cfg: Dict,
    seed: int,
    n_train_eps: int,
    n_eval_eps: int,
    steps_per_ep: int,
    dry_run: bool = False,
) -> Dict:
    """Run one arm for one seed via StepHarness. Returns dict of metrics."""
    use_dacc = arm_cfg["use_dacc"]
    arm_id = arm_cfg["arm_id"]
    arm_label = arm_cfg["label"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = _make_env(seed, dry_run=dry_run)
    agent = _make_agent(env, use_dacc)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    # Boundary line for runner progress display.
    print(f"Seed {seed} Condition {arm_label}", flush=True)

    # ------------------------------------------------------------------
    # Training phase
    # ------------------------------------------------------------------
    train_harness = StepHarness(agent, env, train_mode=True, seed=seed)

    for ep in range(n_train_eps):
        agent.reset()
        _, obs_dict = env.reset()
        train_harness.reset()

        for _step in range(steps_per_ep):
            result = train_harness.step(obs_dict)

            optimizer.zero_grad()
            loss = agent.compute_prediction_loss()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            obs_dict = result.next_obs_dict
            if result.done:
                break

        if (ep + 1) % 5 == 0 or ep + 1 == n_train_eps:
            print(
                f"  [train] arm={arm_label} seed={seed} ep {ep + 1}/{n_train_eps} "
                f"rv={float(agent.e3._running_variance):.6f} "
                f"prec={float(agent.e3.current_precision):.4f}",
                flush=True,
            )

    rv_after_training = float(agent.e3._running_variance)

    # ------------------------------------------------------------------
    # Eval phase
    # ------------------------------------------------------------------
    eval_harness = StepHarness(agent, env, train_mode=False, seed=seed + 10000)

    precision_vals: List[float] = []
    beta_vals: List[bool] = []

    for _ep in range(n_eval_eps):
        agent.reset()
        _, obs_dict = env.reset()
        eval_harness.reset()

        for _step in range(steps_per_ep):
            result = eval_harness.step(obs_dict)
            precision_vals.append(float(agent.e3.current_precision))
            beta_vals.append(bool(agent.beta_gate.is_elevated))
            obs_dict = result.next_obs_dict
            if result.done:
                break

    rv_final = float(agent.e3._running_variance)

    n_steps = len(precision_vals)
    if n_steps == 0:
        return {
            "arm_id": arm_id, "arm_label": arm_label, "seed": seed,
            "commit_rate": 0.0,
            "high_precision_commit_rate": 0.0,
            "low_precision_commit_rate": 0.0,
            "precision_commit_ratio": 1.0,
            "mean_precision": 0.0,
            "rv_after_training": rv_after_training,
            "rv_final": rv_final,
            "n_steps": 0,
        }

    commit_rate = sum(1 for b in beta_vals if b) / n_steps
    mean_precision = sum(precision_vals) / n_steps

    sorted_prec = sorted(precision_vals)
    median_prec = sorted_prec[n_steps // 2]

    high_steps = [(p, b) for p, b in zip(precision_vals, beta_vals) if p > median_prec]
    low_steps = [(p, b) for p, b in zip(precision_vals, beta_vals) if p <= median_prec]

    high_commit = (sum(1 for _, b in high_steps if b) / len(high_steps)) if high_steps else 0.0
    low_commit = (sum(1 for _, b in low_steps if b) / len(low_steps)) if low_steps else 0.0

    precision_commit_ratio = high_commit / (low_commit + 1e-8)

    return {
        "arm_id": arm_id,
        "arm_label": arm_label,
        "seed": seed,
        "commit_rate": float(commit_rate),
        "high_precision_commit_rate": float(high_commit),
        "low_precision_commit_rate": float(low_commit),
        "precision_commit_ratio": float(precision_commit_ratio),
        "mean_precision": float(mean_precision),
        "median_precision": float(median_prec),
        "rv_after_training": rv_after_training,
        "rv_final": rv_final,
        "n_steps": n_steps,
        "n_high_steps": len(high_steps),
        "n_low_steps": len(low_steps),
    }


def run_experiment(dry_run: bool = False) -> Dict:
    if dry_run:
        n_train, n_eval, n_seeds, steps = 1, 1, 1, 20
    else:
        n_train, n_eval, n_seeds, steps = N_TRAIN_EPS, N_EVAL_EPS, N_SEEDS, STEPS_PER_EP

    rng = np.random.default_rng(42)
    seeds = [int(rng.integers(1000, 9999)) for _ in range(n_seeds)]

    all_results: List[Dict] = []
    t_start = time.time()

    for arm_cfg in ARMS:
        for seed in seeds:
            res = _run_arm_seed(arm_cfg, seed, n_train, n_eval, steps, dry_run=dry_run)
            all_results.append(res)
            passed = (
                res["rv_final"] != PRECISION_INIT_BASELINE
                and res["commit_rate"] > 0.01
            )
            print(
                f"  arm={res['arm_label']} seed={seed} "
                f"commit_rate={res['commit_rate']:.3f} "
                f"ratio={res['precision_commit_ratio']:.3f} "
                f"rv_final={res['rv_final']:.6f}",
                flush=True,
            )
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    elapsed = time.time() - t_start

    arm0 = [r for r in all_results if r["arm_id"] == 0]
    arm1 = [r for r in all_results if r["arm_id"] == 1]

    def _mean(rs: List[Dict], key: str) -> float:
        vals = [r[key] for r in rs]
        return float(sum(vals) / len(vals)) if vals else 0.0

    arm0_ratio = _mean(arm0, "precision_commit_ratio")
    arm1_ratio = _mean(arm1, "precision_commit_ratio")
    arm0_commit = _mean(arm0, "commit_rate")
    arm1_commit = _mean(arm1, "commit_rate")

    rv_diffs_arm0 = [abs(r["rv_final"] - PRECISION_INIT_BASELINE) for r in arm0]
    rv_diffs_arm1 = [abs(r["rv_final"] - PRECISION_INIT_BASELINE) for r in arm1]
    c0 = (
        all(d > RV_DIFF_FLOOR for d in rv_diffs_arm0)
        and all(d > RV_DIFF_FLOOR for d in rv_diffs_arm1)
    )

    c1 = arm1_ratio > arm0_ratio
    c2 = arm1_ratio > 1.0
    c3 = arm1_commit > 0.01

    overall_pass = c0 and c1 and c2 and c3
    outcome = "PASS" if overall_pass else "FAIL"

    print("\nResults:", flush=True)
    print(f"  ARM_0 precision_commit_ratio: {arm0_ratio:.4f}", flush=True)
    print(f"  ARM_1 precision_commit_ratio: {arm1_ratio:.4f}", flush=True)
    print(f"  ARM_0 commit_rate: {arm0_commit:.4f}", flush=True)
    print(f"  ARM_1 commit_rate: {arm1_commit:.4f}", flush=True)
    print(f"  C0 (rv moved both arms): {c0}", flush=True)
    print(f"  C1 (arm1_ratio > arm0_ratio): {c1}", flush=True)
    print(f"  C2 (arm1_ratio > 1.0): {c2}", flush=True)
    print(f"  C3 (arm1_commit > 0.01): {c3}", flush=True)
    print(f"  Outcome: {outcome}", flush=True)

    metrics = {
        "arm0_mean_precision_commit_ratio": arm0_ratio,
        "arm1_mean_precision_commit_ratio": arm1_ratio,
        "arm0_mean_commit_rate": arm0_commit,
        "arm1_mean_commit_rate": arm1_commit,
        "arm0_mean_precision": _mean(arm0, "mean_precision"),
        "arm1_mean_precision": _mean(arm1, "mean_precision"),
        "arm0_rv_final_per_seed": [r["rv_final"] for r in arm0],
        "arm1_rv_final_per_seed": [r["rv_final"] for r in arm1],
        "arm0_rv_after_training_per_seed": [r["rv_after_training"] for r in arm0],
        "arm1_rv_after_training_per_seed": [r["rv_after_training"] for r in arm1],
        "c0_rv_moved_both_arms": c0,
        "c1_arm1_ratio_gt_arm0": c1,
        "c2_arm1_ratio_gt_1": c2,
        "c3_arm1_commit_positive": c3,
        "per_seed_results": all_results,
        "elapsed_seconds": float(elapsed),
        "n_seeds": n_seeds,
        "n_train_eps": n_train,
        "n_eval_eps": n_eval,
        "steps_per_ep": steps,
    }

    return {"metrics": metrics, "overall_pass": overall_pass, "outcome": outcome}


def write_result(result: Dict, run_id: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outcome = result["outcome"]
    metrics = result["metrics"]

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "metrics": metrics,
        "config": {
            "n_train_eps": metrics["n_train_eps"],
            "n_eval_eps": metrics["n_eval_eps"],
            "n_seeds": metrics["n_seeds"],
            "grid_size": GRID_SIZE,
            "steps_per_ep": metrics["steps_per_ep"],
            "lr": LR,
            "arms": ARMS,
            "reef_kwargs": REEF_KWARGS,
        },
        "supersedes_note": (
            "V3-EXQ-530b reclassified non_contributory 2026-05-07T23:12Z because "
            "its inline loop never called agent.update_residue(); rv pinned at "
            "precision_init=0.5 -> e3.current_precision constant 1.999996 -> "
            "ARC-016 precision-to-commit pathway untestable. This retest routes "
            "the agent loop through StepHarness which calls update_residue once "
            "per env step, ensuring rv is live. Q-042 contract test C0 added as "
            "an explicit gate."
        ),
    }

    out_dir = Path(__file__).resolve().parent.parent.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test with minimal episodes")
    args = parser.parse_args()

    dry_run = args.dry_run
    if dry_run:
        print("Dry run mode: n_train=1, n_eval=1, n_seeds=1, steps=20", flush=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    result = run_experiment(dry_run=dry_run)
    out_path = None
    if not dry_run:
        out_path = write_result(result, run_id)
    else:
        print("Dry run complete. Outcome:", result["outcome"], flush=True)
        print("DRY_RUN_COMPLETE", flush=True)
    return result, run_id, out_path


if __name__ == "__main__":
    _result, _run_id, _out_path = main()
    _outcome_emit = _result["outcome"] if _result["outcome"] in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome_emit,
        manifest_path=_out_path,
        run_id=_run_id,
        exit_reason="ok" if _outcome_emit == "PASS" else "fail",
    )
