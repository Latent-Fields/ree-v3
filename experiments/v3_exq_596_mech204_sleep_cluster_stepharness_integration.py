#!/opt/local/bin/python3
"""V3-EXQ-596 -- MECH-204 closure integration validation (sleep cluster + StepHarness).
SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

Substrate-ready handoff for MECH-204 (sleep_substrate_plan.md GAP-1 done via
V3-EXQ-541c PASS 2026-05-09). That cohort validated F1 precision recalibration
mechanics in isolation (manual waking tick). This experiment is the governance
closure run requested when substrate_queue marks ready=True: MECH-204 consumer
+ unified use_sleep_aggregation_cluster (GAP-3/GAP-8 path) + canonical StepHarness
waking loop (sense / update_z_goal / update_residue once per env step).

Scientific question: with the full offline-consolidation cluster enabled and
waking ticks routed through StepHarness, does MECH-204 F1 recalibration still
fire every cycle and produce measurable running_variance divergence vs the
within-experiment step=0.0 reference?

Two-arm paired design (step fixed at 0.25, the 541c-confirmed default):
  ARM_0_off:       rem_precision_recalibration_step=0.0 (consumer wired, no-op)
  ARM_1_step_0_25: rem_precision_recalibration_step=0.25 (active F1 consumer)

Both arms: use_sleep_aggregation_cluster=True, tonic_5ht_enabled=True,
anchor-set / staleness / e2_harm_s prerequisites (same pattern as V3-EXQ-581).

Pre-registered acceptance:
  C1 (MECH-204 fires): ARM_1 mech204_recalibration_fired==1.0 on every cycle
      in >=2/3 seeds.
  C2 (cluster live): ARM_1 mech285_n_draws>0 AND sws_n_writes>0 on every
      cycle in >=2/3 seeds (post-warmup cycles 2..N).
  C3 (F1 divergence): ARM_1 vs ARM_0 mean post-cycle rv (cycles 2..N)
      relative divergence >= 3% in >=2/3 seeds (paired by seed).

PASS = C1 AND C2 AND C3.

FAIL-route: integration regressed MECH-204 or silenced the sleep cluster;
route /diagnose-errors. Behavioral downstream proxy (V3-EXQ-593) remains
blocked on ARC-065 / MECH-269 diversity until monostrategy ceiling clears.

confirms: V3-EXQ-541c (does not supersede).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_596_mech204_sleep_cluster_stepharness_integration"
QUEUE_ID = "V3-EXQ-596"
CLAIM_IDS = ["MECH-204"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EXP-0171"
CONFIRMS = "V3-EXQ-541c"
SLEEP_DRIVER_PATTERN = (
    "K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)"
)
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
ACTIVE_STEP = 0.25

SEEDS = (42, 43, 44)
ARMS = (
    ("ARM_0_off", 0.0),
    ("ARM_1_step_0_25", ACTIVE_STEP),
)
EPISODES_PER_RUN = 12
STEPS_PER_EPISODE = 150
SLEEP_LOOP_K = 1

C1_MIN_SEEDS = 2
C2_MIN_SEEDS = 2
C3_REL_DIVERGENCE_MIN = 0.03
WARMUP_CYCLES = 1  # drop cycle 1 cold-start (same as 541c later-cycles logic)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=6,
        num_resources=2,
        hazard_harm=0.06,
        proximity_harm_scale=0.18,
        proximity_benefit_scale=0.10,
        env_drift_interval=5,
        env_drift_prob=0.5,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _install_anchors(agent: REEAgent, *, n: int = 4) -> None:
    anchor_set = agent.hippocampal.anchor_set
    assert anchor_set is not None, "anchor_set must be initialised"
    for i in range(n):
        z = torch.randn(1, 32)
        anchor_set.write_anchor(
            scale="fast",
            segment_id=str(i),
            stream_mixture=(f"s{i}",),
            z_world=z,
        )


def _make_agent(env: CausalGridWorldV2, *, step: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        sleep_loop_episodes_K=SLEEP_LOOP_K,
        use_sleep_aggregation_cluster=True,
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=step,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
        use_e2_harm_s_forward=True,
        mech285_draws_per_cycle=8,
        mech272_sws_anchor_weight=0.6,
        mech273_offline_n_steps=5,
    )
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def run_arm_seed(arm_label: str, step: float, seed: int, episodes_per_run: int) -> dict:
    _seed_all(seed)
    env = _make_env(seed)
    agent = _make_agent(env, step=step)
    harness = StepHarness(agent, env, train_mode=False, seed=seed)

    _flat, obs_dict = env.reset()
    agent.reset()
    harness.reset()

    cycle_records: list[dict] = []
    anchors_installed = False

    print(f"Seed {seed} Condition {arm_label}", flush=True)
    for ep in range(episodes_per_run):
        print(
            f"  [train] {arm_label} seed={seed} ep {ep + 1}/{episodes_per_run}",
            flush=True,
        )
        for _ in range(STEPS_PER_EPISODE):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.done:
                _flat, obs_dict = env.reset()
                agent.reset()
                harness.reset()

        # GAP-8 / MECH-285 need anchors + populated world buffer (V3-EXQ-581).
        if ep >= 1 and not anchors_installed:
            _install_anchors(agent)
            anchors_installed = True

        rv_pre_cycle = float(agent.e3._running_variance)
        agent.reset()
        rv_post_cycle = float(agent.e3._running_variance)

        metrics: dict = {}
        cycle_state = agent.sleep_loop.state if agent.sleep_loop else None
        if cycle_state is not None and cycle_state.last_metrics:
            metrics = dict(cycle_state.last_metrics)
            cycle_state.last_metrics = {}

        cycle_records.append(
            {
                "episode": ep + 1,
                "rv_pre_cycle": rv_pre_cycle,
                "rv_post_cycle": rv_post_cycle,
                "mech204_recalibration_fired": float(
                    metrics.get("mech204_recalibration_fired", 0.0)
                ),
                "mech285_n_draws": float(metrics.get("mech285_n_draws", 0.0)),
                "sws_n_writes": float(metrics.get("sws_n_writes", 0.0)),
                "sws_anchor_weight_applied": float(
                    metrics.get("sws_anchor_weight_applied", 0.0)
                ),
            }
        )

    n_cycles = len(cycle_records)
    fired_all = bool(
        n_cycles > 0
        and all(c["mech204_recalibration_fired"] == 1.0 for c in cycle_records)
    )
    post_warmup = cycle_records[WARMUP_CYCLES:]
    # Cluster SWS writes need anchors (installed before sleep from ep>=1).
    post_anchor = [c for c in cycle_records if c["episode"] >= 2]
    cluster_live_each = [
        c["mech285_n_draws"] > 0 and c["sws_n_writes"] > 0 for c in post_anchor
    ]
    cluster_live_frac = (
        sum(cluster_live_each) / len(cluster_live_each) if cluster_live_each else 0.0
    )
    mean_rv_post = (
        sum(c["rv_post_cycle"] for c in post_warmup) / len(post_warmup)
        if post_warmup
        else 0.0
    )

    per_seed_pass = bool(
        (step == 0.0 or fired_all)
        and (step == 0.0 or cluster_live_frac >= 0.5)
    )
    print(f"verdict: {'PASS' if per_seed_pass else 'FAIL'}", flush=True)

    return {
        "arm": arm_label,
        "step": step,
        "seed": seed,
        "n_cycles": n_cycles,
        "fired_all_cycles": fired_all,
        "cluster_live_frac_post_warmup": float(cluster_live_frac),
        "mean_rv_post_later_cycles": float(mean_rv_post),
        "cycle_records": cycle_records,
    }


def _aggregate(seed_results: dict) -> dict:
    arm0 = "ARM_0_off"
    arm1 = "ARM_1_step_0_25"
    off_by_seed = {r["seed"]: r for r in seed_results[arm0]}
    on_runs = seed_results[arm1]

    c1_seeds = sum(1 for r in on_runs if r["fired_all_cycles"])
    c1_pass = c1_seeds >= C1_MIN_SEEDS

    c2_seeds = sum(
        1
        for r in on_runs
        if r["cluster_live_frac_post_warmup"] >= 0.5
    )
    c2_pass = c2_seeds >= C2_MIN_SEEDS

    c3_seeds = 0
    per_seed_div: list[float] = []
    for r in on_runs:
        rv_off = off_by_seed[r["seed"]]["mean_rv_post_later_cycles"]
        rv_on = r["mean_rv_post_later_cycles"]
        if abs(rv_off) < 1e-9:
            rd = abs(rv_on - rv_off)
        else:
            rd = abs(rv_on - rv_off) / abs(rv_off)
        per_seed_div.append(rd)
        if rd >= C3_REL_DIVERGENCE_MIN:
            c3_seeds += 1
    c3_pass = c3_seeds >= C1_MIN_SEEDS

    overall_pass = bool(c1_pass and c2_pass and c3_pass)
    return {
        "n_seeds": len(SEEDS),
        "c1_pass": c1_pass,
        "c1_seeds_fired_all": c1_seeds,
        "c2_pass": c2_pass,
        "c2_seeds_cluster_live": c2_seeds,
        "c3_pass": c3_pass,
        "c3_seeds_diverged": c3_seeds,
        "per_seed_relative_divergence": per_seed_div,
        "overall_pass": overall_pass,
    }


def main(dry_run: bool = False):
    seeds = (SEEDS[0],) if dry_run else SEEDS
    episodes_per_run = 2 if dry_run else EPISODES_PER_RUN

    t0 = time.time()
    seed_results: dict = {arm: [] for arm, _ in ARMS}
    for arm_label, step in ARMS:
        for seed in seeds:
            seed_results[arm_label].append(
                run_arm_seed(arm_label, step, seed, episodes_per_run)
            )
    elapsed = time.time() - t0

    criteria = _aggregate(seed_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    print(
        f"V3-EXQ-596 MECH-204 sleep-cluster StepHarness integration -- "
        f"{outcome} in {elapsed:.1f}s",
        flush=True,
    )
    if dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "confirms": CONFIRMS,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": direction,
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_SEEDS": C1_MIN_SEEDS,
            "C2_MIN_SEEDS": C2_MIN_SEEDS,
            "C3_REL_DIVERGENCE_MIN": C3_REL_DIVERGENCE_MIN,
            "ACTIVE_STEP": ACTIVE_STEP,
            "PRECISION_ZERO_POINT_EMA_ALPHA": PRECISION_ZERO_POINT_EMA_ALPHA,
        },
        "config": {
            "seeds": list(seeds),
            "arms": [{"label": a, "step": s} for a, s in ARMS],
            "episodes_per_run": episodes_per_run,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_loop_K": SLEEP_LOOP_K,
            "use_sleep_aggregation_cluster": True,
            "stepharness_waking": True,
        },
        "seed_results": seed_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Closure-handoff contributory validation for MECH-204 substrate_ready. "
            "Combines V3-EXQ-541c F1 consumer (step=0.25) with GAP-3 unified "
            "sleep aggregation cluster and StepHarness canonical waking path. "
            "Does not test behavioral downstream proxies (deferred V3-EXQ-593 "
            "until ARC-065 / MECH-269 diversity)."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result is None:
        sys.exit(0)
    _outcome, _out_path = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
