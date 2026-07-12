#!/opt/local/bin/python3
"""V3-EXQ-541b -- MECH-204 precision recalibration step-size sweep
SLEEP DRIVER: K=2 multi-fire (SleepLoopManager, sleep_loop_episodes_K=2, fires every 2 episodes)
(EXP-0171 instantiation; gated on V3-EXQ-541a F1 result).

V3-EXQ-541a (F1 substrate) PASSed C1 (substrate-readiness, 3/3) and C2
(mean_abs_delta=3.62e-3 vs 1e-3 threshold; sign-consistency=1.0) but
FAILed C3 (cross-arm divergence=5.64e-3 vs 5e-2 threshold). Per-cycle
recalibration is doing its job, but the per-cycle effect (~5e-3) is
largely re-absorbed by waking drift over the ~400 steps between sleep
cycles, so cumulative cross-arm divergence stays small.

This sweep tests whether step-size tuning alone (within the same F1
substrate) can clear C3, OR whether F1+step-tuning is intrinsically
limited and Phase 7 / Option B (broadcast read at action selection;
deferred-conditional per Q-042 verdict, gated on the REM-precision
lit-pull queued in parallel) becomes the load-bearing path.

Five-arm sweep on rem_precision_recalibration_step:
  ARM_0_off       step=0.0  (within-experiment no-op reference; consumer
                              wired but step=0 makes recalibrate a no-op
                              per contract test C6).
  ARM_1_step_0_05 step=0.05 (low-end of biologically defensible band).
  ARM_2_step_0_10 step=0.10 (V3-EXQ-541a baseline; mid-band).
  ARM_3_step_0_25 step=0.25 (high-end of biologically defensible band per
                              Q-042 Option A verdict).
  ARM_4_step_0_50 step=0.50 (above-band stress test; risks overshoot).

precision_zero_point_ema_alpha=0.1 held constant across arms (the F1
substrate fix that landed 2026-05-09; this sweep is on step only).

Pre-registered acceptance per EXP-0171:
  C1 (substrate-readiness): in arms with step > 0,
      mech204_recalibration_fired==1.0 on every cycle in >=2/3 seeds.
  C2 (tracking_quality): per arm, tracking_quality =
      1 - mean(|rv_after - target_variance| / max(target_variance, 1e-6))
      over cycles 2..N. At least one step in {0.05, 0.10, 0.25} achieves
      tracking_quality >= 0.7 in >=2/3 seeds.
  C3 (overshoot_rate): per arm, overshoot_rate = fraction of cycles
      where sign(rv_after - target_variance) != sign(rv_before - target_variance).
      At least one step in {0.05, 0.10, 0.25} keeps overshoot_rate <= 0.1
      in >=2/3 seeds.
  C4 (cross-arm divergence): for each step > 0 arm, compute mean
      post-cycle rv on cycles 2..N and compare to ARM_0_off. At least
      one step in {0.05, 0.10, 0.25, 0.5} produces relative divergence
      >= 0.05 (5%) in >=2/3 seeds.
PASS = C1 AND (some step satisfies C2 AND C3) AND (some step satisfies C4).
The same step does not need to satisfy all three; the report identifies
the regime each metric peaks in.

FAIL-route: if no step in {0.05, 0.10, 0.25} satisfies C4, OR if 0.5
also fails C4, F1+step-tuning alone is insufficient and the diagnosis
points at Phase 7 / Option B as the next architectural lever (gated on
the REM-precision lit-pull verdict).
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_541b_mech204_step_size_sweep"
CLAIM_IDS = ["MECH-204"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EXP-0171"
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1

SEEDS = (42, 43, 44)
# (arm_label, step_value)
ARMS = (
    ("ARM_0_off", 0.0),
    ("ARM_1_step_0_05", 0.05),
    ("ARM_2_step_0_10", 0.10),
    ("ARM_3_step_0_25", 0.25),
    ("ARM_4_step_0_50", 0.50),
)
EPISODES_PER_RUN = 8
STEPS_PER_EPISODE = 200
SLEEP_LOOP_K = 2  # sleep fires every K episodes -> 4 cycles per run

# Pre-registered thresholds (per EXP-0171)
C1_MIN_SEEDS_FIRED = 2          # >=2/3 seeds with every-cycle fire (step>0 arms)
C2_TRACKING_QUALITY_MIN = 0.7   # tracking_quality >= 0.7 in >=2/3 seeds
C3_OVERSHOOT_RATE_MAX = 0.1     # overshoot_rate <= 0.1 in >=2/3 seeds
C4_REL_DIVERGENCE_MIN = 0.05    # relative divergence vs ARM_0 >= 5% in >=2/3 seeds
DEFENSIBLE_STEPS = (0.05, 0.10, 0.25)  # band per Q-042 Option A verdict


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


def _make_agent(env: CausalGridWorldV2, seed: int, *, step: float) -> REEAgent:
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
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=True,
        rem_attribution_steps=6,
        use_sleep_loop=True,
        sleep_loop_episodes_K=SLEEP_LOOP_K,
        # MECH-204 F1 consumer (under test). All arms have the consumer
        # wired; ARM_0 uses step=0.0 (no-op per contract C6) as the
        # within-experiment OFF reference.
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=step,
    )
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _one_hot_action(action_idx: int, action_dim: int) -> torch.Tensor:
    action = torch.zeros(1, action_dim)
    action[0, int(action_idx)] = 1.0
    return action


def _tick_wake(agent: REEAgent, env: CausalGridWorldV2,
               obs_dict: dict, rng: random.Random) -> dict:
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
    # Drive E3 prediction-error EMA: synthetic PE keeps _running_variance
    # moving across waking ticks so the recalibration consumer has
    # something to act on at REM entry / WRITEBACK.
    if hasattr(agent, "e3"):
        pe_scale = 0.4 + 0.3 * rng.random()
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


def run_arm_seed(arm_label: str, step: float, seed: int,
                 episodes_per_run: int) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = _make_env(seed)
    agent = _make_agent(env, seed, step=step)
    _flat, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    cycle_records: list[dict] = []
    rv_history: list[float] = [float(agent.e3._running_variance)]

    print(f"Seed {seed} Condition {arm_label}", flush=True)
    for ep in range(episodes_per_run):
        print(
            f"  [train] {arm_label} seed={seed} ep {ep + 1}/{episodes_per_run}",
            flush=True,
        )
        for _ in range(STEPS_PER_EPISODE):
            obs_dict = _tick_wake(agent, env, obs_dict, rng)

        rv_pre_cycle = float(agent.e3._running_variance)
        agent.reset()
        rv_history.append(float(agent.e3._running_variance))

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
                }
            )
            cycle_state.last_metrics = {}

    n_cycles = len(cycle_records)
    fired_each = [c["mech204_recalibration_fired"] for c in cycle_records]
    fired_all = bool(n_cycles > 0 and all(f == 1.0 for f in fired_each))

    # Drop cycle 1 (cold-start: persistent = first capture = rv at REM
    # entry by construction). Meaningful movement measured on cycles 2..N.
    later = cycle_records[1:]
    if later:
        tracking_errs = []
        overshoots = 0
        deltas = []
        for c in later:
            rv_b = c["mech204_running_variance_before"]
            rv_a = c["mech204_running_variance_after"]
            target = c["mech204_recalibration_target"]
            if rv_a != rv_a or rv_b != rv_b or target <= 0.0:
                continue
            target_var = 1.0 / (target + 1e-6)
            tracking_errs.append(abs(rv_a - target_var) / max(target_var, 1e-6))
            # overshoot: rv crossed target in the wrong direction
            sign_pre = _sign(target_var - rv_b)
            sign_post = _sign(target_var - rv_a)
            if sign_pre != 0 and sign_post != 0 and sign_pre != sign_post:
                overshoots += 1
            deltas.append(rv_a - rv_b)
        n_later = len(later)
        tracking_quality = (
            1.0 - (sum(tracking_errs) / len(tracking_errs))
            if tracking_errs else 0.0
        )
        overshoot_rate = overshoots / n_later if n_later else 0.0
        mean_abs_delta = (
            sum(abs(d) for d in deltas) / len(deltas) if deltas else 0.0
        )
    else:
        tracking_quality = 0.0
        overshoot_rate = 0.0
        mean_abs_delta = 0.0

    mean_rv_post_later = (
        sum(c["rv_post_cycle"] for c in later) / len(later) if later else 0.0
    )

    # Per-seed verdict: arms with step > 0 must fire on every cycle.
    # ARM_0 (step=0.0) fires but produces no movement -- still verdict PASS
    # at the per-seed level (substrate is wired; the no-op is intentional).
    per_seed_pass = bool(step == 0.0 or fired_all)
    print(f"verdict: {'PASS' if per_seed_pass else 'FAIL'}", flush=True)

    return {
        "arm": arm_label,
        "step": step,
        "seed": seed,
        "n_cycles": n_cycles,
        "fired_each_cycle": fired_each,
        "fired_all_cycles": fired_all,
        "rv_history": rv_history,
        "cycle_records": cycle_records,
        "tracking_quality": float(tracking_quality),
        "overshoot_rate": float(overshoot_rate),
        "mean_abs_delta_later_cycles": float(mean_abs_delta),
        "mean_rv_post_later_cycles": float(mean_rv_post_later),
    }


def _aggregate(seed_results: dict) -> dict:
    """seed_results: {arm_label: [seed_record, ...]}"""
    arm_labels = [a[0] for a in ARMS]
    arm_steps = {a[0]: a[1] for a in ARMS}

    per_arm = {}
    for arm_label in arm_labels:
        runs = seed_results[arm_label]
        step = arm_steps[arm_label]
        seeds_fired_all = sum(1 for r in runs if r["fired_all_cycles"])
        seeds_track_ok = sum(
            1 for r in runs if r["tracking_quality"] >= C2_TRACKING_QUALITY_MIN
        )
        seeds_overshoot_ok = sum(
            1 for r in runs if r["overshoot_rate"] <= C3_OVERSHOOT_RATE_MAX
        )
        mean_track = sum(r["tracking_quality"] for r in runs) / max(1, len(runs))
        mean_overshoot = sum(r["overshoot_rate"] for r in runs) / max(1, len(runs))
        mean_rv_post = sum(r["mean_rv_post_later_cycles"] for r in runs) / max(1, len(runs))
        per_arm[arm_label] = {
            "step": step,
            "seeds_fired_all": seeds_fired_all,
            "seeds_track_ok": seeds_track_ok,
            "seeds_overshoot_ok": seeds_overshoot_ok,
            "mean_tracking_quality": float(mean_track),
            "mean_overshoot_rate": float(mean_overshoot),
            "mean_rv_post_later_cycles": float(mean_rv_post),
        }

    # C1: every step>0 arm fires every cycle in >=2/3 seeds.
    c1_per_arm = {
        label: per_arm[label]["seeds_fired_all"] >= C1_MIN_SEEDS_FIRED
        for label in arm_labels if per_arm[label]["step"] > 0
    }
    c1_pass = bool(c1_per_arm) and all(c1_per_arm.values())

    # C2: at least one step in DEFENSIBLE_STEPS achieves tracking_quality
    # >= 0.7 in >=2/3 seeds.
    c2_qualifying = []
    for label in arm_labels:
        if per_arm[label]["step"] in DEFENSIBLE_STEPS:
            if per_arm[label]["seeds_track_ok"] >= C1_MIN_SEEDS_FIRED:
                c2_qualifying.append(label)
    c2_pass = bool(c2_qualifying)

    # C3: at least one step in DEFENSIBLE_STEPS keeps overshoot_rate
    # <= 0.1 in >=2/3 seeds.
    c3_qualifying = []
    for label in arm_labels:
        if per_arm[label]["step"] in DEFENSIBLE_STEPS:
            if per_arm[label]["seeds_overshoot_ok"] >= C1_MIN_SEEDS_FIRED:
                c3_qualifying.append(label)
    c3_pass = bool(c3_qualifying)

    # C4: at least one step>0 arm produces relative divergence >= 5%
    # vs ARM_0 in >=2/3 seeds. (Per-seed because env seeds are matched
    # across arms.)
    arm0_seeds = {r["seed"]: r["mean_rv_post_later_cycles"]
                  for r in seed_results[arm_labels[0]]}
    c4_per_arm = {}
    for label in arm_labels:
        if per_arm[label]["step"] == 0.0:
            continue
        seeds_diverged = 0
        per_seed_div = []
        for r in seed_results[label]:
            arm0_rv = arm0_seeds.get(r["seed"], 0.0)
            arm1_rv = r["mean_rv_post_later_cycles"]
            if abs(arm0_rv) < 1e-9:
                rd = abs(arm1_rv - arm0_rv)
            else:
                rd = abs(arm1_rv - arm0_rv) / abs(arm0_rv)
            per_seed_div.append(rd)
            if rd >= C4_REL_DIVERGENCE_MIN:
                seeds_diverged += 1
        c4_per_arm[label] = {
            "seeds_diverged": seeds_diverged,
            "per_seed_relative_divergence": per_seed_div,
            "mean_relative_divergence": (
                sum(per_seed_div) / len(per_seed_div) if per_seed_div else 0.0
            ),
        }
    c4_qualifying = [
        label for label, d in c4_per_arm.items()
        if d["seeds_diverged"] >= C1_MIN_SEEDS_FIRED
    ]
    c4_pass = bool(c4_qualifying)

    overall_pass = bool(c1_pass and c2_pass and c3_pass and c4_pass)

    return {
        "n_seeds": len(SEEDS),
        "per_arm": per_arm,
        "c1_pass": c1_pass,
        "c1_per_arm": c1_per_arm,
        "c2_pass": c2_pass,
        "c2_qualifying_arms": c2_qualifying,
        "c3_pass": c3_pass,
        "c3_qualifying_arms": c3_qualifying,
        "c4_pass": c4_pass,
        "c4_qualifying_arms": c4_qualifying,
        "c4_per_arm": c4_per_arm,
        "overall_pass": overall_pass,
    }


def main(dry_run: bool = False):
    seeds = (SEEDS[0],) if dry_run else SEEDS
    episodes_per_run = 2 if dry_run else EPISODES_PER_RUN

    t0 = time.time()
    seed_results: dict = {arm_label: [] for arm_label, _ in ARMS}
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
        f"V3-EXQ-541b MECH-204 step-size sweep -- {outcome} in {elapsed:.1f}s",
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
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": direction,
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_SEEDS_FIRED": C1_MIN_SEEDS_FIRED,
            "C2_TRACKING_QUALITY_MIN": C2_TRACKING_QUALITY_MIN,
            "C3_OVERSHOOT_RATE_MAX": C3_OVERSHOOT_RATE_MAX,
            "C4_REL_DIVERGENCE_MIN": C4_REL_DIVERGENCE_MIN,
            "DEFENSIBLE_STEPS": list(DEFENSIBLE_STEPS),
            "PRECISION_ZERO_POINT_EMA_ALPHA": PRECISION_ZERO_POINT_EMA_ALPHA,
        },
        "config": {
            "seeds": list(seeds),
            "arms": [{"label": a, "step": s} for a, s in ARMS],
            "episodes_per_run": episodes_per_run,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_loop_K": SLEEP_LOOP_K,
            "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        },
        "seed_results": seed_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "EXP-0171 step-size sweep on MECH-204 F1 substrate. "
            "Five arms vary rem_precision_recalibration_step in "
            "{0.0_off, 0.05, 0.10, 0.25, 0.50} with "
            "precision_zero_point_ema_alpha=0.1 held constant. ARM_0 "
            "uses step=0.0 as the within-experiment no-op reference "
            "(consumer wired but step=0 makes recalibrate a no-op per "
            "contract C6). Acceptance: C1 substrate-readiness across "
            "step>0 arms; C2 tracking_quality >= 0.7 in some defensible "
            "step; C3 overshoot_rate <= 0.1 in some defensible step; "
            "C4 cross-arm divergence >= 5% in some step>0 arm. PASS = "
            "all four. The same step does not need to satisfy all "
            "three behavioural criteria; the report identifies the "
            "regime each metric peaks in. FAIL on C4 across all "
            "defensible steps points at Phase 7 / Option B (broadcast "
            "read at action selection) as the next architectural lever, "
            "gated on the parallel REM-precision lit-pull verdict."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
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
