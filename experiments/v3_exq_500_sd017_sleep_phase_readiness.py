#!/opt/local/bin/python3
"""V3-EXQ-500 -- EXP-0171 SD-017 sleep-phase substrate readiness.
SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a dedicated N_CYCLES wake-sleep-test loop)

This is a diagnostic readiness probe, not the later sleep-vs-no-sleep
discriminative ablation. It verifies that the existing SD-017 substrate can
enter a repeated four-stage cycle with measurable wake feedstock, replay
preparation, replay, and post-sleep test signatures.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_500_sd017_sleep_phase_readiness"
CLAIM_IDS = ["SD-017"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = (42, 43, 44)
CONDITION = "SLEEP_PHASE_READINESS"
CYCLES_PER_RUN = 3
WAKE_STEPS_PER_CYCLE = 160
POST_SLEEP_TEST_STEPS = 24
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6

MIN_PHASE_ENTRIES = 3
MAX_PHASE_DURATION_CV = 0.20
REPLAY_QUALITY_BASELINE = 8.0


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.16,
        env_drift_interval=7,
        env_drift_prob=0.35,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        scheduled_external_hazard_enabled=True,
        scheduled_external_hazard_interval=40,
        scheduled_external_hazard_prob=0.7,
    )


def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        sws_schema_weight=0.1,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
    )
    return REEAgent(cfg)


def _one_hot_action(action_idx: int, action_dim: int = 5) -> torch.Tensor:
    action = torch.zeros(1, action_dim)
    action[0, int(action_idx)] = 1.0
    return action


def _tick_wake(agent: REEAgent, env: CausalGridWorldV2, obs_dict: dict, rng: random.Random) -> dict:
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    obs_harm = obs_dict.get("harm_obs")
    obs_harm_a = obs_dict.get("harm_obs_a")
    obs_harm_history = obs_dict.get("harm_history")

    latent = agent.sense(
        obs_body,
        obs_world,
        obs_harm=obs_harm,
        obs_harm_a=obs_harm_a,
        obs_harm_history=obs_harm_history,
    )
    ticks = agent.clock.advance()
    if ticks.get("e1_tick", False):
        agent._e1_tick(latent)

    action_idx = rng.randrange(env.action_dim)
    action = _one_hot_action(action_idx, env.action_dim)
    _flat, _harm, done, _info, next_obs = env.step(action)
    if done:
        _flat, next_obs = env.reset()
    return next_obs


def _slot_diversity(agent: REEAgent) -> float:
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        if mem.shape[0] < 2:
            return 0.0
        normed = torch.nn.functional.normalize(mem, dim=-1)
        sim = normed @ normed.t()
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return float((1.0 - sim[mask]).mean().item())


def _cv(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) < 1e-9:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return float(math.sqrt(var) / abs(mean))


def run_seed(seed: int, cycles_to_run: int, progress_denominator: int) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = _make_env(seed)
    agent = _make_agent(env, seed)
    _flat, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    phase_counts = {"wake": 0, "replay_prep": 0, "replay": 0, "test": 0}
    phase_durations = {k: [] for k in phase_counts}
    cycle_metrics = []

    print(f"Seed {seed} Condition {CONDITION}", flush=True)
    for cycle in range(cycles_to_run):
        print(
            f"  [train] {CONDITION} seed={seed} ep {cycle + 1}/{progress_denominator}",
            flush=True,
        )

        phase_counts["wake"] += 1
        wake_steps = 0
        for _ in range(WAKE_STEPS_PER_CYCLE):
            obs_dict = _tick_wake(agent, env, obs_dict, rng)
            wake_steps += 1
        phase_durations["wake"].append(float(wake_steps))

        phase_counts["replay_prep"] += 1
        prep_ready = (
            len(agent._world_experience_buffer) >= 2
            and len(agent._self_experience_buffer) >= 2
            and agent.theta_buffer.recent is not None
        )
        phase_durations["replay_prep"].append(1.0)

        phase_counts["replay"] += 1
        sleep_metrics = agent.run_sleep_cycle()
        replay_ops = (
            float(sleep_metrics.get("sws_n_writes", 0.0))
            + float(sleep_metrics.get("rem_n_rollouts", 0.0))
        )
        phase_durations["replay"].append(replay_ops)

        phase_counts["test"] += 1
        before_div = _slot_diversity(agent)
        test_steps = 0
        for _ in range(POST_SLEEP_TEST_STEPS):
            obs_dict = _tick_wake(agent, env, obs_dict, rng)
            test_steps += 1
        after_div = _slot_diversity(agent)
        phase_durations["test"].append(float(test_steps))

        rem_rollouts = float(sleep_metrics.get("rem_n_rollouts", 0.0))
        sws_writes = float(sleep_metrics.get("sws_n_writes", 0.0))
        sws_diversity = float(sleep_metrics.get("sws_slot_diversity", 0.0))
        replay_quality = rem_rollouts + 0.25 * sws_writes + 0.1 * sws_diversity
        cycle_metrics.append(
            {
                "cycle": cycle + 1,
                "prep_ready": bool(prep_ready),
                "sws_n_writes": sws_writes,
                "rem_n_rollouts": rem_rollouts,
                "sws_slot_diversity": sws_diversity,
                "slot_diversity_before_test": before_div,
                "slot_diversity_after_test": after_div,
                "replay_quality": replay_quality,
            }
        )

    duration_cv = {k: _cv(v) for k, v in phase_durations.items()}
    mean_replay_quality = sum(c["replay_quality"] for c in cycle_metrics) / max(1, len(cycle_metrics))
    min_phase_count = min(phase_counts.values())
    c1 = min_phase_count >= cycles_to_run and all(c["prep_ready"] for c in cycle_metrics)
    c2 = all(v <= MAX_PHASE_DURATION_CV for v in duration_cv.values())
    c3 = mean_replay_quality >= REPLAY_QUALITY_BASELINE
    seed_pass = bool(c1 and c2 and c3)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "condition": CONDITION,
        "phase_entry_count": phase_counts,
        "phase_duration_cv": duration_cv,
        "cycle_metrics": cycle_metrics,
        "mean_replay_quality": mean_replay_quality,
        "c1_phase_entries_and_prep": c1,
        "c2_duration_cv_within_20pct": c2,
        "c3_replay_quality_ge_baseline": c3,
        "passed": seed_pass,
    }


def _aggregate(seed_results: list[dict], cycles_required: int) -> dict:
    c1_all = all(r["c1_phase_entries_and_prep"] for r in seed_results)
    c2_all = all(r["c2_duration_cv_within_20pct"] for r in seed_results)
    c3_all = all(r["c3_replay_quality_ge_baseline"] for r in seed_results)
    return {
        "n_seeds": len(seed_results),
        "cycles_required": cycles_required,
        "c1_phase_entry_count_ge_required_all_seeds": c1_all,
        "c2_phase_duration_variance_within_20pct_all_seeds": c2_all,
        "c3_replay_quality_ge_exq265_baseline_all_seeds": c3_all,
        "overall_pass": bool(c1_all and c2_all and c3_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = (SEEDS[0],) if args.dry_run else SEEDS
    cycles_to_run = 1 if args.dry_run else CYCLES_PER_RUN
    t0 = time.time()
    seed_results = [
        run_seed(seed, cycles_to_run=cycles_to_run, progress_denominator=CYCLES_PER_RUN)
        for seed in seeds
    ]
    elapsed = time.time() - t0
    criteria = _aggregate(seed_results, cycles_required=cycles_to_run)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    print(f"V3-EXQ-500 SD-017 sleep-phase readiness -- {outcome} in {elapsed:.1f}s", flush=True)
    if args.dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"SD-017": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "MIN_PHASE_ENTRIES": MIN_PHASE_ENTRIES,
            "MAX_PHASE_DURATION_CV": MAX_PHASE_DURATION_CV,
            "REPLAY_QUALITY_BASELINE": REPLAY_QUALITY_BASELINE,
        },
        "config": {
            "seeds": list(seeds),
            "condition": CONDITION,
            "cycles_per_run": CYCLES_PER_RUN,
            "wake_steps_per_cycle": WAKE_STEPS_PER_CYCLE,
            "post_sleep_test_steps": POST_SLEEP_TEST_STEPS,
            "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
            "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        },
        "seed_results": seed_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Diagnostic readiness probe for EXP-0171. It verifies repeated "
            "wake, replay-prep, replay, and test phase entries over the "
            "existing SD-017 REEAgent sleep methods before any sleep ablation "
            "is queued."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
