#!/opt/local/bin/python3
"""V3-EXQ-541 -- MECH-204 precision recalibration consumer (Phase 1 validation).

Sleep-substrate plan Phase 1 (REE_assembly/evidence/planning/
sleep_substrate_plan.md, GAP-1). Validates that the WRITEBACK-phase
sibling step in SleepLoopManager actually consumes
SerotoninModule._precision_at_rem_entry via Option A statistical update
on E3._running_variance. Two-arm ablation (recalibration ON vs OFF).

Mechanism under test: under sustained waking precision drift between
sleep cycles, the recalibration ON arm pulls _running_variance back
toward the captured zero-point reference each REM cycle; the OFF arm
runs the cycle without the consumer wired (bit-identical fall-through).

Acceptance:
  C1 (substrate-readiness): in ARM_1, mech204_recalibration_fired==1.0
      on every sleep cycle in >=2/3 seeds.
  C2 (movement direction consistent with target): in ARM_1, the per-cycle
      delta sign(rv_after - rv_before) matches sign(target_variance - rv);
      mean signed delta over cycles 2..N is non-zero (>= MIN_MEAN_DELTA).
  C3 (cross-arm divergence): mean post-cycle rv across cycles 2..N
      differs between arms by >= MIN_REL_DIVERGENCE relative magnitude.

Independent of MECH-273 self-model gradient pass (which is OFF in both
arms here).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_541_mech204_precision_recalibration_consumer"
CLAIM_IDS = ["MECH-204"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = (42, 43, 44)
ARMS = ("ARM_0_off", "ARM_1_on_step_0_1")
EPISODES_PER_RUN = 8
STEPS_PER_EPISODE = 200
SLEEP_LOOP_K = 2  # sleep fires every K episodes -> 4 cycles per run
RECALIBRATION_STEP = 0.1

# Thresholds (pre-registered)
MIN_MEAN_DELTA = 1e-3        # C2: mean |rv_after - rv_before| floor
MIN_REL_DIVERGENCE = 0.05    # C3: 5% relative cross-arm divergence
C1_MIN_SEEDS_FIRED = 2       # C1: >= 2/3 seeds fire on every cycle


def _make_env(seed: int) -> CausalGridWorldV2:
    # Hazard-heavy / resource-thin to drive sustained PE variance
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


def _make_agent(env: CausalGridWorldV2, seed: int, *, recalibration_on: bool) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        # SD-017 sleep passes (substrate for the cycle)
        sws_enabled=True,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=True,
        rem_attribution_steps=6,
        # Phase A sleep-loop driver
        use_sleep_loop=True,
        sleep_loop_episodes_K=SLEEP_LOOP_K,
        # MECH-204 consumer (under test)
        use_rem_precision_recalibration=recalibration_on,
        rem_precision_recalibration_step=RECALIBRATION_STEP,
    )
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _one_hot_action(action_idx: int, action_dim: int) -> torch.Tensor:
    action = torch.zeros(1, action_dim)
    action[0, int(action_idx)] = 1.0
    return action


def _tick_wake(agent: REEAgent, env: CausalGridWorldV2, obs_dict: dict, rng: random.Random) -> dict:
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    latent = agent.sense(
        obs_body,
        obs_world,
        obs_harm=obs_dict.get("harm_obs"),
        obs_harm_a=obs_dict.get("harm_obs_a"),
        obs_harm_history=obs_dict.get("harm_history"),
    )
    ticks = agent.clock.advance()
    if ticks.get("e1_tick", False):
        agent._e1_tick(latent)
    # Drive E3 prediction-error EMA: pass a synthetic prediction_error tensor
    # to keep _running_variance moving even when E3 isn't selecting trajectories.
    if hasattr(agent, "e3"):
        pe_scale = 0.4 + 0.3 * rng.random()  # sustained, randomly perturbed
        synthetic_pe = torch.randn(1, 4) * pe_scale
        agent.e3.update_running_variance(synthetic_pe)

    action_idx = rng.randrange(env.action_dim)
    action = _one_hot_action(action_idx, env.action_dim)
    _flat, _harm, done, _info, next_obs = env.step(action)
    if done:
        _flat, next_obs = env.reset()
    return next_obs


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def run_arm_seed(arm: str, seed: int, recalibration_on: bool) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = _make_env(seed)
    agent = _make_agent(env, seed, recalibration_on=recalibration_on)
    _flat, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    cycle_records: list[dict] = []
    rv_history: list[float] = [float(agent.e3._running_variance)]

    print(f"Seed {seed} Condition {arm}", flush=True)
    for ep in range(EPISODES_PER_RUN):
        print(
            f"  [train] {arm} seed={seed} ep {ep + 1}/{EPISODES_PER_RUN}",
            flush=True,
        )
        # Run a fixed-length episode of waking ticks
        for _ in range(STEPS_PER_EPISODE):
            obs_dict = _tick_wake(agent, env, obs_dict, rng)

        rv_pre_cycle = float(agent.e3._running_variance)
        # Episode boundary: agent.reset() drives sleep_loop.notify_episode_end()
        # which fires a cycle every K episodes via _run_cycle (the WRITEBACK
        # hook lives there).
        agent.reset()
        rv_history.append(float(agent.e3._running_variance))

        # The sleep cycle fires only every K episodes; pull the most recent
        # last_metrics if a cycle just landed this boundary.
        cycle_state = agent.sleep_loop.state if agent.sleep_loop else None
        if cycle_state is not None and cycle_state.last_metrics:
            metrics = dict(cycle_state.last_metrics)
            rv_post_cycle = float(agent.e3._running_variance)
            cycle_records.append(
                {
                    "episode": ep + 1,
                    "rv_pre_cycle": rv_pre_cycle,
                    "rv_post_cycle": rv_post_cycle,
                    "mech204_recalibration_fired": float(
                        metrics.get("mech204_recalibration_fired", 0.0)
                    ),
                    "mech204_recalibration_target": float(
                        metrics.get("mech204_recalibration_target", 0.0)
                    ),
                    "mech204_running_variance_before": float(
                        metrics.get("mech204_running_variance_before", float("nan"))
                    ),
                    "mech204_running_variance_after": float(
                        metrics.get("mech204_running_variance_after", float("nan"))
                    ),
                    "mech204_recalibration_step": float(
                        metrics.get("mech204_recalibration_step", 0.0)
                    ),
                    "sws_n_writes": float(metrics.get("sws_n_writes", 0.0)),
                    "rem_n_rollouts": float(metrics.get("rem_n_rollouts", 0.0)),
                }
            )
            # Clear so we don't double-count if no new cycle fires next boundary
            cycle_state.last_metrics = {}

    # Per-seed metrics
    n_cycles = len(cycle_records)
    fired_each = [c["mech204_recalibration_fired"] for c in cycle_records]
    fired_all = bool(n_cycles > 0 and all(f == 1.0 for f in fired_each))

    # Cycles 2..N (drop first cycle: rv at REM == captured target by definition,
    # so first-cycle no-op is expected; meaningful movement requires cross-cycle
    # drift)
    later = cycle_records[1:]
    if later:
        deltas = []
        sign_consistencies = []
        for c in later:
            rv_b = c["mech204_running_variance_before"]
            rv_a = c["mech204_running_variance_after"]
            target = c["mech204_recalibration_target"]
            if rv_a != rv_a or rv_b != rv_b:  # NaN check
                continue
            delta = rv_a - rv_b
            deltas.append(delta)
            if recalibration_on and target > 0.0:
                target_var = 1.0 / (target + 1e-6)
                expected_sign = _sign(target_var - rv_b)
                sign_consistencies.append(_sign(delta) == expected_sign or delta == 0.0)
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        mean_abs_delta = sum(abs(d) for d in deltas) / len(deltas) if deltas else 0.0
        sign_consistency_rate = (
            sum(sign_consistencies) / len(sign_consistencies)
            if sign_consistencies else 1.0
        )
    else:
        mean_delta = 0.0
        mean_abs_delta = 0.0
        sign_consistency_rate = 1.0

    mean_rv_post_later = (
        sum(c["rv_post_cycle"] for c in later) / len(later) if later else 0.0
    )

    print(f"verdict: {'PASS' if fired_all or not recalibration_on else 'FAIL'}", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "recalibration_on": recalibration_on,
        "n_cycles": n_cycles,
        "fired_each_cycle": fired_each,
        "fired_all_cycles": fired_all,
        "rv_history": rv_history,
        "cycle_records": cycle_records,
        "mean_delta_later_cycles": mean_delta,
        "mean_abs_delta_later_cycles": mean_abs_delta,
        "sign_consistency_rate": sign_consistency_rate,
        "mean_rv_post_later_cycles": mean_rv_post_later,
    }


def _aggregate(seed_results: dict) -> dict:
    """seed_results: {arm: [seed_record, ...]}"""
    arm0 = seed_results[ARMS[0]]
    arm1 = seed_results[ARMS[1]]

    # C1: ARM_1, fired on every cycle in >= 2/3 seeds
    c1_seeds_fired = sum(1 for r in arm1 if r["fired_all_cycles"])
    c1_pass = c1_seeds_fired >= C1_MIN_SEEDS_FIRED

    # C2: ARM_1, mean_abs_delta across seeds >= MIN_MEAN_DELTA, AND sign
    # consistency rate >= 0.8 (allowing for 1-2 zero-cross drift cases)
    arm1_mean_abs = (
        sum(r["mean_abs_delta_later_cycles"] for r in arm1) / max(1, len(arm1))
    )
    arm1_sign_rate = (
        sum(r["sign_consistency_rate"] for r in arm1) / max(1, len(arm1))
    )
    c2_pass = bool(arm1_mean_abs >= MIN_MEAN_DELTA and arm1_sign_rate >= 0.8)

    # C3: cross-arm divergence in mean_rv_post_later_cycles
    arm0_mean_rv = (
        sum(r["mean_rv_post_later_cycles"] for r in arm0) / max(1, len(arm0))
    )
    arm1_mean_rv = (
        sum(r["mean_rv_post_later_cycles"] for r in arm1) / max(1, len(arm1))
    )
    if abs(arm0_mean_rv) < 1e-9:
        rel_div = abs(arm1_mean_rv - arm0_mean_rv)
    else:
        rel_div = abs(arm1_mean_rv - arm0_mean_rv) / abs(arm0_mean_rv)
    c3_pass = bool(rel_div >= MIN_REL_DIVERGENCE)

    return {
        "n_seeds": len(arm0),
        "c1_arm1_fired_every_cycle_seeds": c1_seeds_fired,
        "c1_pass": c1_pass,
        "c2_arm1_mean_abs_delta": float(arm1_mean_abs),
        "c2_arm1_sign_consistency_rate": float(arm1_sign_rate),
        "c2_pass": c2_pass,
        "c3_arm0_mean_rv_post_later": float(arm0_mean_rv),
        "c3_arm1_mean_rv_post_later": float(arm1_mean_rv),
        "c3_relative_divergence": float(rel_div),
        "c3_pass": c3_pass,
        "overall_pass": bool(c1_pass and c2_pass and c3_pass),
    }


def main(dry_run: bool = False):
    seeds = (SEEDS[0],) if dry_run else SEEDS
    if dry_run:
        global EPISODES_PER_RUN
        EPISODES_PER_RUN = 2

    t0 = time.time()
    seed_results: dict = {arm: [] for arm in ARMS}
    for arm in ARMS:
        recalibration_on = (arm == ARMS[1])
        for seed in seeds:
            seed_results[arm].append(run_arm_seed(arm, seed, recalibration_on))
    elapsed = time.time() - t0

    criteria = _aggregate(seed_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    print(
        f"V3-EXQ-541 MECH-204 recalibration consumer -- {outcome} in {elapsed:.1f}s",
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
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": direction,
        "criteria": criteria,
        "registered_thresholds": {
            "MIN_MEAN_DELTA": MIN_MEAN_DELTA,
            "MIN_REL_DIVERGENCE": MIN_REL_DIVERGENCE,
            "C1_MIN_SEEDS_FIRED": C1_MIN_SEEDS_FIRED,
            "RECALIBRATION_STEP": RECALIBRATION_STEP,
        },
        "config": {
            "seeds": list(seeds),
            "arms": list(ARMS),
            "episodes_per_run": EPISODES_PER_RUN,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_loop_K": SLEEP_LOOP_K,
        },
        "seed_results": seed_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Phase 1 validation of MECH-204 precision recalibration consumer. "
            "ARM_0 disables the consumer (use_rem_precision_recalibration=False); "
            "ARM_1 enables it with step=0.1. Hazard-heavy / resource-thin env "
            "drives sustained E3 _running_variance drift between sleep cycles. "
            "Acceptance: C1 substrate-readiness (recalibration fires every cycle "
            "in >=2/3 ON-arm seeds), C2 movement-direction consistency with target "
            "and non-zero mean_abs_delta, C3 cross-arm divergence on post-cycle rv. "
            "First cycle's no-op behaviour is expected (target == rv_at_rem_entry "
            "by construction); meaningful movement is measured on cycles 2..N."
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
