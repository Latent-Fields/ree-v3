#!/opt/local/bin/python3
"""V3-EXQ-593 -- MECH-204 downstream behavioral proxy effects at >20 sleep cycles
SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

V3-EXQ-541c (16 cycles, PASS 2026-05-09) confirmed MECH-204 precision
recalibration fires reliably at step=0.25 and produces monotone dose-response
cross-arm divergence in running_variance. Failure autopsy (2026-05-17)
reclassified 541a/541b evidence as inconclusive_timescale: the C1-C4 mechanics
criteria confirm the mechanism fires, but do not test whether recalibration
produces measurable downstream behavioral effects within the reachable cycle
count.

This experiment shifts the primary criterion from REM-mechanics (tracking
quality, rv divergence) to GAP-8 behavioral proxy metrics:
  replay_diversity_index   -- distinct SWS replay regions / total draws per cycle
  post_sleep_z_goal_before -- z_goal norm at sleep entry (advisory)

Scientific question: after >20 sleep cycles at step=0.25, does MECH-204
precision recalibration produce measurable divergence in downstream behavioral
proxy metrics compared to the step=0.0 no-op reference?

Mechanism path:
  Cycle N:  REM recalibration adjusts running_variance toward zero-point EMA
  Waking N+1: calibrated precision weighting -> different E3 exploration profile
  Cycle N+1: MECH-285 draws from accumulated hippocampal anchors; if precision
            calibration broadens explored regions, replay_diversity_index rises

Two-arm design (step fixed at 0.25, the 541c-confirmed optimal):
  ARM_0_off:       step=0.0  (consumer wired; recalibrate is no-op per C6)
  ARM_1_step_0_25: step=0.25 (active recalibration per Q-042 Option A verdict)

Full sleep aggregation cluster (use_sleep_aggregation_cluster=True) provides
MECH-285 sampler + MECH-272 routing + routing consumer (GAP-8) so that
replay_diversity_index is non-sentinel. z_goal_enabled=True to capture
post_sleep_z_goal_before when benefit contacts seed z_goal during waking.

Pre-registered acceptance (primary criterion: GAP-8 proxy metrics):
  C1 (substrate-readiness): in ARM_1, mech204_recalibration_fired==1.0 on
      every cycle in >=2/3 seeds.
  C2 (proxy non-sentinel): ARM_1 replay_diversity_index > SENTINEL_THRESHOLD
      in >=50% of post-warmup cycles in >=2/3 seeds. Confirms MECH-285 is
      drawing anchors in the sleep cluster setup.
  C3 (divergence direction, PRIMARY): ARM_1 mean replay_diversity_index
      (post-warmup cycles only, cycles 5..24) > ARM_0 mean + C3_MIN_DIVERGENCE
      in >=2/3 seeds. This is the primary PASS gate.
  C4_advisory (z_goal proxy): if >=1/3 seeds have non-sentinel
      post_sleep_z_goal_before in ARM_1, report direction (ARM_1 > ARM_0).
      Advisory only -- does not block PASS.

PASS = C1 AND C2 AND C3.

FAIL-route: if C3 fails (replay_diversity divergence < 0.02 after 24 cycles),
MECH-204 recalibration at the F1 level does not produce detectable downstream
proxy effects within the V3 episode budget. Route to:
  (a) longer-timescale experiment (>100 cycles), or
  (b) direct task-performance behavioral criteria, or
  (c) Phase 7 / Option B if REM-precision lit-pull SYNTHESIS dispatch case #3.
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

EXPERIMENT_TYPE = "v3_exq_593_mech204_gap8_proxy_behavioral_effects"
CLAIM_IDS = ["MECH-204"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EXP-0171"
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1

SEEDS = (42, 43, 44)
ARMS = (
    ("ARM_0_off", 0.0),
    ("ARM_1_step_0_25", 0.25),
)
EPISODES_PER_RUN = 24   # 24 cycles per run at K=1 (>20 as required)
STEPS_PER_EPISODE = 200
SLEEP_LOOP_K = 1        # fires every episode -> 24 cycles per run

# Pre-registered thresholds
C1_MIN_SEEDS_FIRED = 2          # >=2/3 seeds with every-cycle fire (ARM_1)
SENTINEL_THRESHOLD = -0.5       # replay_diversity_index > this = non-sentinel
C2_MIN_NON_SENTINEL_FRAC = 0.5  # >=50% of post-warmup cycles non-sentinel
C3_MIN_DIVERGENCE = 0.02        # ARM_1 mean_rdi > ARM_0 mean_rdi + 0.02
WARMUP_CYCLES = 4               # skip first 4 cycles (cold-start transient)
Z_GOAL_SENTINEL = -0.5          # post_sleep_z_goal_before > this = active


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
        # Full sleep aggregation cluster: enables MECH-285 sampler (Phase B),
        # MECH-272 routing gate (Phase C), routing consumer (GAP-8 telemetry),
        # MECH-275 Bayesian aggregator (Phase D), MECH-273 self-model writeback
        # (Phase E). Gives non-sentinel replay_diversity_index via MECH-285 draws.
        use_sleep_aggregation_cluster=True,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_attribution_steps=6,
        sleep_loop_episodes_K=SLEEP_LOOP_K,
        # z_goal enabled for post_sleep_z_goal_before telemetry.
        z_goal_enabled=True,
        # MECH-204 F1 consumer (mechanism under test). Both arms have the
        # consumer wired; ARM_0 uses step=0.0 (no-op per C6 contract).
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=step,
    )
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _one_hot_action(action_idx: int, action_dim: int) -> torch.Tensor:
    a = torch.zeros(1, action_dim)
    a[0, int(action_idx)] = 1.0
    return a


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
    # Drive E3 prediction-error EMA so running_variance evolves across waking
    # ticks; recalibration consumer needs a moving target.
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

    print(f"Seed {seed} Condition {arm_label}", flush=True)
    for ep in range(episodes_per_run):
        print(
            f"  [train] {arm_label} seed={seed} ep {ep + 1}/{episodes_per_run}",
            flush=True,
        )
        for _ in range(STEPS_PER_EPISODE):
            obs_dict = _tick_wake(agent, env, obs_dict, rng)

        # agent.reset() triggers SleepLoopManager.on_episode_end() -> sleep cycle.
        agent.reset()

        cycle_state = agent.sleep_loop.state if agent.sleep_loop else None
        if cycle_state is not None and cycle_state.last_metrics:
            m = dict(cycle_state.last_metrics)
            cycle_records.append({
                "episode": ep + 1,
                "mech204_recalibration_fired": float(
                    m.get("mech204_recalibration_fired", 0.0)
                ),
                "replay_diversity_index": float(
                    m.get("replay_diversity_index", -1.0)
                ),
                "post_sleep_z_goal_before": float(
                    m.get("post_sleep_z_goal_before", -1.0)
                ),
                "post_sleep_z_goal_retention": float(
                    m.get("post_sleep_z_goal_retention", -1.0)
                ),
                "mech285_n_draws": float(m.get("mech285_n_draws", 0.0)),
                "mech272_n_routed": float(m.get("mech272_n_routed", 0.0)),
            })
            cycle_state.last_metrics = {}

    n_cycles = len(cycle_records)
    fired_each = [c["mech204_recalibration_fired"] for c in cycle_records]
    fired_all = bool(n_cycles > 0 and all(f == 1.0 for f in fired_each))

    # Post-warmup: skip first WARMUP_CYCLES cycles (cold-start transient).
    post_warmup = cycle_records[WARMUP_CYCLES:]

    # C2: replay_diversity non-sentinel in post-warmup cycles.
    n_non_sentinel = sum(
        1 for c in post_warmup
        if c["replay_diversity_index"] > SENTINEL_THRESHOLD
    )
    non_sentinel_frac = n_non_sentinel / max(1, len(post_warmup))

    # C3: mean replay_diversity_index over non-sentinel post-warmup cycles.
    rdi_vals = [
        c["replay_diversity_index"]
        for c in post_warmup
        if c["replay_diversity_index"] > SENTINEL_THRESHOLD
    ]
    mean_rdi = sum(rdi_vals) / len(rdi_vals) if rdi_vals else -1.0

    # Advisory: z_goal proxy (post-warmup, non-sentinel only).
    zgb_vals = [
        c["post_sleep_z_goal_before"]
        for c in post_warmup
        if c["post_sleep_z_goal_before"] > Z_GOAL_SENTINEL
    ]
    mean_z_goal_before = sum(zgb_vals) / len(zgb_vals) if zgb_vals else -1.0

    # Per-seed verdict: ARM_1 must fire on every cycle; ARM_0 is reference.
    per_seed_pass = bool(step == 0.0 or fired_all)
    print(f"verdict: {'PASS' if per_seed_pass else 'FAIL'}", flush=True)

    return {
        "arm": arm_label,
        "step": step,
        "seed": seed,
        "n_cycles": n_cycles,
        "fired_each_cycle": fired_each,
        "fired_all_cycles": fired_all,
        "non_sentinel_frac": float(non_sentinel_frac),
        "mean_rdi_post_warmup": float(mean_rdi),
        "mean_z_goal_before_post_warmup": float(mean_z_goal_before),
        "cycle_records": cycle_records,
    }


def _aggregate(seed_results: dict) -> dict:
    arm_labels = [a[0] for a in ARMS]
    arm_steps = {a[0]: a[1] for a in ARMS}

    per_arm: dict = {}
    for arm_label in arm_labels:
        runs = seed_results[arm_label]
        seeds_fired_all = sum(1 for r in runs if r["fired_all_cycles"])
        seeds_c2_ok = sum(
            1 for r in runs
            if r["non_sentinel_frac"] >= C2_MIN_NON_SENTINEL_FRAC
        )
        mean_rdi = (
            sum(r["mean_rdi_post_warmup"] for r in runs) / len(runs)
            if runs else -1.0
        )
        mean_zgb = (
            sum(r["mean_z_goal_before_post_warmup"] for r in runs) / len(runs)
            if runs else -1.0
        )
        per_arm[arm_label] = {
            "step": arm_steps[arm_label],
            "seeds_fired_all": seeds_fired_all,
            "seeds_c2_ok": seeds_c2_ok,
            "mean_rdi_post_warmup": float(mean_rdi),
            "mean_z_goal_before_post_warmup": float(mean_zgb),
        }

    arm0_label = arm_labels[0]
    arm1_label = arm_labels[1]

    # C1: ARM_1 fires every cycle in >=2/3 seeds.
    c1_pass = per_arm[arm1_label]["seeds_fired_all"] >= C1_MIN_SEEDS_FIRED

    # C2: ARM_1 non-sentinel frac >= 50% in >=2/3 seeds.
    c2_pass = per_arm[arm1_label]["seeds_c2_ok"] >= C1_MIN_SEEDS_FIRED

    # C3 (PRIMARY): per-seed, ARM_1 mean_rdi > ARM_0 mean_rdi + C3_MIN_DIVERGENCE.
    arm0_rdi_by_seed = {
        r["seed"]: r["mean_rdi_post_warmup"]
        for r in seed_results[arm0_label]
    }
    per_seed_c3: list[dict] = []
    seeds_c3_ok = 0
    for r in seed_results[arm1_label]:
        arm0_rdi = arm0_rdi_by_seed.get(r["seed"], -1.0)
        arm1_rdi = r["mean_rdi_post_warmup"]
        # Only compare when both arms have non-sentinel rdi.
        if arm0_rdi > SENTINEL_THRESHOLD and arm1_rdi > SENTINEL_THRESHOLD:
            divergence = arm1_rdi - arm0_rdi
        else:
            divergence = -1.0
        per_seed_c3.append({
            "seed": r["seed"],
            "arm1_rdi": arm1_rdi,
            "arm0_rdi": arm0_rdi,
            "divergence": divergence,
        })
        if divergence >= C3_MIN_DIVERGENCE:
            seeds_c3_ok += 1
    c3_pass = seeds_c3_ok >= C1_MIN_SEEDS_FIRED

    # C4_advisory: z_goal proxy direction.
    arm0_zgb_by_seed = {
        r["seed"]: r["mean_z_goal_before_post_warmup"]
        for r in seed_results[arm0_label]
    }
    seeds_zgb_active_arm1 = sum(
        1 for r in seed_results[arm1_label]
        if r["mean_z_goal_before_post_warmup"] > Z_GOAL_SENTINEL
    )
    zgb_advisory_pass = False
    if seeds_zgb_active_arm1 >= 1:
        seeds_zgb_direction_ok = 0
        for r in seed_results[arm1_label]:
            arm0_zgb = arm0_zgb_by_seed.get(r["seed"], -1.0)
            arm1_zgb = r["mean_z_goal_before_post_warmup"]
            if (arm0_zgb > Z_GOAL_SENTINEL
                    and arm1_zgb > Z_GOAL_SENTINEL
                    and arm1_zgb > arm0_zgb):
                seeds_zgb_direction_ok += 1
        zgb_advisory_pass = seeds_zgb_direction_ok >= C1_MIN_SEEDS_FIRED

    overall_pass = bool(c1_pass and c2_pass and c3_pass)

    return {
        "n_seeds": len(SEEDS),
        "per_arm": per_arm,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c3_seeds_ok": seeds_c3_ok,
        "per_seed_c3": per_seed_c3,
        "c4_advisory_z_goal": zgb_advisory_pass,
        "c4_seeds_zgb_active_arm1": seeds_zgb_active_arm1,
        "overall_pass": overall_pass,
    }


def main(dry_run: bool = False) -> "tuple[str, Path] | None":
    seeds = (SEEDS[0],) if dry_run else SEEDS
    episodes_per_run = 3 if dry_run else EPISODES_PER_RUN

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
        f"V3-EXQ-593 MECH-204 GAP-8 proxy behavioral effects -- {outcome} in {elapsed:.1f}s",
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
        "sleep_driver_pattern": "K=1 single-fire (SleepLoopManager, fires every episode)",
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": direction,
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_SEEDS_FIRED": C1_MIN_SEEDS_FIRED,
            "SENTINEL_THRESHOLD": SENTINEL_THRESHOLD,
            "C2_MIN_NON_SENTINEL_FRAC": C2_MIN_NON_SENTINEL_FRAC,
            "C3_MIN_DIVERGENCE": C3_MIN_DIVERGENCE,
            "WARMUP_CYCLES": WARMUP_CYCLES,
            "PRECISION_ZERO_POINT_EMA_ALPHA": PRECISION_ZERO_POINT_EMA_ALPHA,
        },
        "config": {
            "seeds": list(seeds),
            "arms": [{"label": a, "step": s} for a, s in ARMS],
            "episodes_per_run": episodes_per_run,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_loop_K": SLEEP_LOOP_K,
            "use_sleep_aggregation_cluster": True,
            "z_goal_enabled": True,
            "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        },
        "seed_results": seed_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-204 GAP-8 proxy behavioral-effects experiment. Shifts primary "
            "criterion from REM-mechanics (541a/541b/541c) to downstream proxy "
            "metrics. Full sleep aggregation cluster (MECH-285 + MECH-272 + "
            "routing consumer + MECH-275 + MECH-273) enabled for non-sentinel "
            "replay_diversity_index. Two-arm design: ARM_0 step=0.0 (no-op ref) "
            "vs ARM_1 step=0.25 (optimal per 541c). 24 cycles (K=1 x 24 ep). "
            "PASS = C1 (mechanism fires) AND C2 (non-sentinel rdi) AND C3 "
            "(rdi ARM_1 > ARM_0 + 0.02 in >=2/3 seeds, post-warmup cycles 5..24). "
            "z_goal proxy advisory only."
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke run (3 ep/arm, no manifest).")
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
