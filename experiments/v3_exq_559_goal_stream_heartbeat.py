#!/opt/local/bin/python3
"""V3-EXQ-559 -- Goal-stream heartbeat diagnostic.

Claims: [] (diagnostic only; no governance weighting)

Purpose
-------
After the 2026-05-13 goal-stream seam fixes:

* REEConfig.from_dims(goal_weight=...) now routes the weight into
  E3Config.goal_weight, the value actually consumed by E3 scoring.
* StepHarness now sources benefit_exposure from body_state[11] when the env
  does not expose a top-level obs_dict["benefit_exposure"] key.

This diagnostic asks whether the existing goal-stream pieces can produce any
live signal under a compact heartbeat run. It is not a success claim for
navigation. It classifies each arm as:

  zero_stream
  partial_stream_write_only
  live_read_side
  live_action_side
  contact_without_goal_seeding

Arms
----
ARM_0_baseline:
    Goal stream off.

ARM_1_zgoal_e3:
    z_goal enabled, E3 goal_weight enabled, contact-gated seeding only.

ARM_2_schema_wanting:
    ARM_1 plus MECH-216 schema wanting and hippocampal VALENCE_WANTING read.

ARM_3_full_mech307:
    ARM_2 plus MECH-295 bridge and MECH-307 write/read-side conjunction flags.

PASS means the full arm produces any non-zero goal-stream activity. FAIL means
the stream remains completely inert even under the deliberately permissive
heartbeat condition. Either outcome is non-contributory governance evidence;
the value is the failure localization metrics in the manifest.

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_559_goal_stream_heartbeat.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness, StepHooks  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.residue.field import (  # noqa: E402
    VALENCE_LIKING,
    VALENCE_NEGATIVE_SURPRISE,
    VALENCE_POSITIVE_SURPRISE,
    VALENCE_SURPRISE,
    VALENCE_WANTING,
)
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_559_goal_stream_heartbeat"
QUEUE_ID = "V3-EXQ-559"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [7, 42]
EVAL_EPISODES = 6
STEPS_PER_EPISODE = 120

DRY_RUN_SEEDS = [7]
DRY_RUN_EPISODES = 1
DRY_RUN_STEPS = 50


ARMS = [
    {
        "arm": "ARM_0_baseline",
        "z_goal_enabled": False,
        "goal_weight": 0.0,
        "schema_wanting": False,
        "wanting_weight": 0.0,
        "bridge": False,
        "mech307": False,
        "resource_encoder": False,
    },
    {
        "arm": "ARM_1_zgoal_e3",
        "z_goal_enabled": True,
        "goal_weight": 0.5,
        "schema_wanting": False,
        "wanting_weight": 0.0,
        "bridge": False,
        "mech307": False,
        "resource_encoder": False,
    },
    {
        "arm": "ARM_2_schema_wanting",
        "z_goal_enabled": True,
        "goal_weight": 0.5,
        "schema_wanting": True,
        "wanting_weight": 0.5,
        "bridge": False,
        "mech307": False,
        "resource_encoder": False,
    },
    {
        "arm": "ARM_3_full_mech307",
        "z_goal_enabled": True,
        "goal_weight": 0.5,
        "schema_wanting": True,
        "wanting_weight": 0.5,
        "bridge": True,
        "mech307": True,
        "resource_encoder": True,
    },
]


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _make_config(env: CausalGridWorld, arm: Dict) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        benefit_threshold=0.05,
        z_goal_enabled=bool(arm["z_goal_enabled"]),
        goal_weight=float(arm["goal_weight"]),
        use_resource_proximity_head=bool(arm["resource_encoder"]),
    )
    cfg.residue.valence_enabled = True
    cfg.hippocampal.wanting_weight = float(arm["wanting_weight"])

    cfg.e1.schema_wanting_enabled = bool(arm["schema_wanting"])
    cfg.schema_wanting_threshold = 0.10
    cfg.schema_wanting_gain = 0.60

    cfg.latent.use_resource_encoder = bool(arm["resource_encoder"])
    cfg.latent.z_resource_dim = cfg.goal.goal_dim

    cfg.use_mech295_liking_bridge = bool(arm["bridge"])
    cfg.mech295_drive_to_liking_gain = 1.0
    cfg.mech295_liking_to_approach_cue_gain = 0.5
    cfg.mech295_min_drive_to_fire = 0.01
    cfg.mech295_min_z_goal_norm_to_fire = 0.03

    cfg.surprise_gated_replay = bool(arm["mech307"])
    cfg.use_mech307_split_surprise = bool(arm["mech307"])
    cfg.use_mech307_schema_multichannel = bool(arm["mech307"])
    cfg.use_mech307_predicted_location_write = bool(arm["mech307"])
    cfg.use_mech307_consumer_conjunction_read = bool(arm["mech307"])
    cfg.mech307_conjunction_gain = 1.0
    cfg.mech307_conjunction_wanting_threshold = 0.10
    cfg.mech307_conjunction_liking_threshold = 0.05
    cfg.mech307_conjunction_z_beta_threshold = 0.10
    return cfg


def _entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _valence_abs_sum(agent: REEAgent, component: int) -> float:
    field = getattr(agent.residue_field, "rbf_field", None)
    vecs = getattr(field, "valence_vecs", None)
    if vecs is None or vecs.numel() == 0 or component >= vecs.shape[1]:
        return 0.0
    return float(vecs[:, component].abs().sum().item())


def _valence_active_centers(agent: REEAgent, component: int) -> int:
    field = getattr(agent.residue_field, "rbf_field", None)
    vecs = getattr(field, "valence_vecs", None)
    if vecs is None or vecs.numel() == 0 or component >= vecs.shape[1]:
        return 0
    return int((vecs[:, component].abs() > 1e-8).sum().item())


def _metric_template(seed: int, arm: Dict) -> Dict:
    return {
        "seed": int(seed),
        "arm": arm["arm"],
        "total_steps": 0,
        "resource_contact_count": 0,
        "benefit_approach_count": 0,
        "benefit_exposure_count": 0,
        "effective_benefit_threshold_crossings": 0,
        "schema_salience_count": 0,
        "schema_salience_sum": 0.0,
        "schema_salience_max": 0.0,
        "schema_salience_threshold_crossings": 0,
        "valence_wanting_write_count": 0,
        "valence_liking_write_count": 0,
        "z_beta_pulse_count": 0,
        "predicted_location_write_count": 0,
        "goal_active_steps": 0,
        "goal_norm_sum": 0.0,
        "goal_norm_peak_observed": 0.0,
        "beta_elevated_steps": 0,
        "action_counts": {},
    }


def _classify(row: Dict) -> str:
    stream_writes = (
        row["valence_wanting_write_count"]
        + row["valence_liking_write_count"]
        + row["effective_benefit_threshold_crossings"]
    )
    read_side = row["bridge_cue_fires"] + row["bridge_conjunction_fires"]
    action_side = row["beta_elevated_steps"]
    if row["resource_contact_count"] > 0 and row["goal_norm_peak"] <= 1e-8:
        return "contact_without_goal_seeding"
    if stream_writes <= 0 and row["goal_norm_peak"] <= 1e-8:
        return "zero_stream"
    if read_side > 0 and action_side > 0:
        return "live_action_side"
    if read_side > 0:
        return "live_read_side"
    return "partial_stream_write_only"


def _run_cell(seed: int, arm: Dict, episodes: int, steps_per_episode: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)
    metrics = _metric_template(seed, arm)

    def on_sense(
        *,
        agent,
        latent,
        obs_dict,
        ticks,
        drive_level,
        benefit_exposure,
        step,
    ) -> None:
        if benefit_exposure > 1e-9:
            metrics["benefit_exposure_count"] += 1
        if agent.goal_state is not None:
            effective_benefit = benefit_exposure * cfg.goal.z_goal_seeding_gain * (
                1.0 + cfg.goal.drive_weight * drive_level
            )
            if effective_benefit > cfg.goal.benefit_threshold:
                metrics["effective_benefit_threshold_crossings"] += 1

        sal = getattr(agent, "_schema_salience", None)
        sal_val = float(sal.squeeze().item()) if sal is not None else 0.0
        metrics["schema_salience_count"] += 1
        metrics["schema_salience_sum"] += sal_val
        metrics["schema_salience_max"] = max(metrics["schema_salience_max"], sal_val)
        if sal_val >= cfg.schema_wanting_threshold:
            metrics["schema_salience_threshold_crossings"] += 1

        wanting_pre = _valence_abs_sum(agent, VALENCE_WANTING)
        liking_pre = _valence_abs_sum(agent, VALENCE_LIKING)
        z_beta_pre = (
            float(latent.z_beta[..., 0].abs().mean().item())
            if latent.z_beta is not None and latent.z_beta.numel() > 0
            else 0.0
        )
        had_cached_prior = getattr(agent, "_cached_e1_prior", None) is not None

        agent.update_schema_wanting(drive_level=drive_level)

        wanting_post = _valence_abs_sum(agent, VALENCE_WANTING)
        liking_post = _valence_abs_sum(agent, VALENCE_LIKING)
        z_beta_post = (
            float(latent.z_beta[..., 0].abs().mean().item())
            if latent.z_beta is not None and latent.z_beta.numel() > 0
            else 0.0
        )
        if wanting_post > wanting_pre + 1e-9:
            metrics["valence_wanting_write_count"] += 1
            if cfg.use_mech307_predicted_location_write and had_cached_prior:
                metrics["predicted_location_write_count"] += 1
        if liking_post > liking_pre + 1e-9:
            metrics["valence_liking_write_count"] += 1
        if z_beta_post > z_beta_pre + 1e-9:
            metrics["z_beta_pulse_count"] += 1

    def on_action(*, agent, latent, action, obs_dict, ticks, step) -> None:
        action_idx = int(action.argmax(dim=-1).item())
        counts = metrics["action_counts"]
        counts[action_idx] = counts.get(action_idx, 0) + 1
        if agent.goal_state is not None and agent.goal_state.is_active():
            metrics["goal_active_steps"] += 1
            g = agent.goal_state.goal_norm()
            metrics["goal_norm_sum"] += g
            metrics["goal_norm_peak_observed"] = max(
                metrics["goal_norm_peak_observed"], g
            )
        if bool(getattr(agent.beta_gate, "is_elevated", False)):
            metrics["beta_elevated_steps"] += 1

    hooks = StepHooks(on_sense=on_sense, on_action=on_action)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            metrics["total_steps"] += 1
            ttype = result.info.get("transition_type", "none")
            if ttype == "resource":
                metrics["resource_contact_count"] += 1
            elif ttype == "benefit_approach":
                metrics["benefit_approach_count"] += 1
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (ep + 1) == episodes:
            print(
                f"  seed={seed} arm={arm['arm']} ep={ep + 1}/{episodes}",
                flush=True,
            )

    bridge = agent.mech295_bridge
    metrics["bridge_write_fires"] = int(getattr(bridge, "_n_write_fires", 0) or 0)
    metrics["bridge_cue_fires"] = int(getattr(bridge, "_n_cue_fires", 0) or 0)
    metrics["bridge_conjunction_reads"] = int(
        getattr(bridge, "_n_conjunction_reads", 0) or 0
    )
    metrics["bridge_conjunction_fires"] = int(
        getattr(bridge, "_n_conjunction_fires", 0) or 0
    )
    metrics["goal_norm_peak"] = (
        float(agent.goal_state._goal_norm_peak)
        if agent.goal_state is not None else 0.0
    )
    metrics["goal_norm_final"] = (
        agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
    )
    metrics["goal_active_fraction"] = (
        metrics["goal_active_steps"] / max(1, metrics["total_steps"])
    )
    metrics["beta_elevated_fraction"] = (
        metrics["beta_elevated_steps"] / max(1, metrics["total_steps"])
    )
    metrics["schema_salience_mean"] = (
        metrics["schema_salience_sum"] / max(1, metrics["schema_salience_count"])
    )
    metrics["action_entropy"] = _entropy(metrics["action_counts"])
    metrics["action_counts"] = {
        str(k): int(v) for k, v in sorted(metrics["action_counts"].items())
    }
    metrics["total_wanting_field_strength"] = _valence_abs_sum(
        agent, VALENCE_WANTING
    )
    metrics["total_liking_field_strength"] = _valence_abs_sum(
        agent, VALENCE_LIKING
    )
    metrics["total_surprise_field_strength"] = _valence_abs_sum(
        agent, VALENCE_SURPRISE
    )
    metrics["positive_surprise_centers"] = _valence_active_centers(
        agent, VALENCE_POSITIVE_SURPRISE
    )
    metrics["negative_surprise_centers"] = _valence_active_centers(
        agent, VALENCE_NEGATIVE_SURPRISE
    )
    metrics["e3_goal_weight"] = float(cfg.e3.goal_weight)
    metrics["goal_config_weight"] = float(cfg.goal.goal_weight)
    metrics["hippocampal_wanting_weight"] = float(cfg.hippocampal.wanting_weight)
    metrics["classification"] = _classify(metrics)
    return metrics


def _aggregate(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for arm in ARMS:
        arm_name = arm["arm"]
        subset = [r for r in rows if r["arm"] == arm_name]
        if not subset:
            continue
        keys = [
            "resource_contact_count",
            "benefit_approach_count",
            "benefit_exposure_count",
            "effective_benefit_threshold_crossings",
            "schema_salience_threshold_crossings",
            "valence_wanting_write_count",
            "valence_liking_write_count",
            "z_beta_pulse_count",
            "predicted_location_write_count",
            "bridge_write_fires",
            "bridge_cue_fires",
            "bridge_conjunction_reads",
            "bridge_conjunction_fires",
            "goal_norm_peak",
            "goal_active_fraction",
            "beta_elevated_fraction",
            "schema_salience_mean",
            "schema_salience_max",
            "action_entropy",
            "total_wanting_field_strength",
            "total_liking_field_strength",
            "positive_surprise_centers",
            "negative_surprise_centers",
        ]
        agg = {"arm": arm_name, "n_seeds": len(subset)}
        for key in keys:
            vals = [float(r.get(key, 0.0)) for r in subset]
            agg[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
            agg[f"{key}_max"] = float(np.max(vals)) if vals else 0.0
        classifications: Dict[str, int] = {}
        for r in subset:
            cls = str(r["classification"])
            classifications[cls] = classifications.get(cls, 0) + 1
        agg["classifications"] = classifications
        out[arm_name] = agg
    return out


def _evaluate(agg: Dict[str, Dict]) -> Dict:
    full = agg.get("ARM_3_full_mech307", {})
    full_stream_nonzero = bool(
        full.get("valence_wanting_write_count_mean", 0.0) > 0.0
        or full.get("valence_liking_write_count_mean", 0.0) > 0.0
        or full.get("goal_norm_peak_max", 0.0) > 0.0
        or full.get("bridge_cue_fires_mean", 0.0) > 0.0
        or full.get("bridge_conjunction_fires_mean", 0.0) > 0.0
    )
    schema_arm = agg.get("ARM_2_schema_wanting", {})
    schema_write_nonzero = bool(
        schema_arm.get("valence_wanting_write_count_mean", 0.0) > 0.0
    )
    contact_seeded = bool(full.get("goal_norm_peak_max", 0.0) > 0.0)
    return {
        "full_arm_stream_nonzero": full_stream_nonzero,
        "schema_arm_wanting_write_nonzero": schema_write_nonzero,
        "contact_seeded_z_goal": contact_seeded,
        "all_pass": full_stream_nonzero,
    }


def main(dry_run: bool = False):
    print(f"[{EXPERIMENT_TYPE}] starting dry_run={dry_run}", flush=True)
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    rows: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        for arm in ARMS:
            row = _run_cell(seed, arm, episodes, steps)
            rows.append(row)
            print(
                f"  result seed={seed} arm={arm['arm']} "
                f"class={row['classification']} "
                f"schema_cross={row['schema_salience_threshold_crossings']} "
                f"want_writes={row['valence_wanting_write_count']} "
                f"like_writes={row['valence_liking_write_count']} "
                f"goal_peak={row['goal_norm_peak']:.4f} "
                f"cue_fires={row['bridge_cue_fires']} "
                f"conj_fires={row['bridge_conjunction_fires']} "
                f"contacts={row['resource_contact_count']} "
                f"entropy={row['action_entropy']:.3f}",
                flush=True,
            )

    agg = _aggregate(rows)
    acceptance = _evaluate(agg)
    elapsed = time.time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] aggregate summary", flush=True)
    for arm_name, arm_agg in agg.items():
        print(
            f"  {arm_name}: class={arm_agg['classifications']} "
            f"want_mean={arm_agg['valence_wanting_write_count_mean']:.2f} "
            f"like_mean={arm_agg['valence_liking_write_count_mean']:.2f} "
            f"goal_peak_max={arm_agg['goal_norm_peak_max']:.4f} "
            f"cue_mean={arm_agg['bridge_cue_fires_mean']:.2f} "
            f"conj_mean={arm_agg['bridge_conjunction_fires_mean']:.2f}",
            flush=True,
        )
    print(f"[{EXPERIMENT_TYPE}] acceptance={acceptance}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest written.", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_quality_note": (
            "Goal-stream heartbeat diagnostic after 2026-05-13 config and "
            "StepHarness benefit-exposure seam fixes. Non-contributory by "
            "design; use classifications and stream counters to route the "
            "next implementation step."
        ),
        "elapsed_seconds": elapsed,
        "seeds": seeds,
        "episodes": episodes,
        "steps_per_episode": steps,
        "arms": list(agg.values()),
        "per_seed_per_arm": rows,
        "acceptance": acceptance,
        "recent_substrate_fixes": {
            "from_dims_routes_goal_weight_to_e3": True,
            "step_harness_sources_benefit_from_body_state_11": True,
        },
    }
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result == 0:
        sys.exit(0)
    _outcome, _out_path = result
    emit_outcome(outcome=_outcome, manifest_path=_out_path)
    sys.exit(0)
