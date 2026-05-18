#!/opt/local/bin/python3
"""V3-EXQ-560 -- goal-stream selectivity and score decomposition.

Claims: [] (diagnostic only; no governance weighting)

Purpose
-------
V3-EXQ-559a proved the canonical goal-stream path is live, but action entropy
remained 0.0. This diagnostic localizes why:

  1. Is MECH-216 schema salience selective for future resource contact, or
     does it fire everywhere?
  2. Do candidate trajectories differ enough for any goal signal to choose
     among them?
  3. Do z_goal, MECH-295, and MECH-307 components move the E3 argmin, or are
     they flat/swamped by the base score?
  4. If clean goal attractors are fed from real resource contact, does pure
     z_goal scoring or the full goal-stream bundle gain action diversity?

Arms
----
ARM_0_canonical_full:
    The landed V3-EXQ-559a goal_stream preset.

ARM_1_contact_memory_full:
    Same as ARM_0, but the first tick after real resource contact force-seeds
    z_goal to the current z_world latent. This tests the user's "feed goals"
    hypothesis under the full bridge/conjunction path.

ARM_2_contact_memory_zgoal_only:
    z_goal + E3 goal scoring only, with the same contact-memory force seed.
    This isolates whether a clean attractor can affect E3 without schema
    wanting, MECH-295, or MECH-307.

PASS means the diagnostic ran and produced finite decomposition metrics. It
does not claim behavioural goal competence.

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_560_goal_stream_selectivity_score_decomp.py --dry-run
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
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness, StepHooks  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.goal import GoalState  # noqa: E402
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_560_goal_stream_selectivity_score_decomp"
QUEUE_ID = "V3-EXQ-560"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [7, 42]
EVAL_EPISODES = 6
STEPS_PER_EPISODE = 120

DRY_RUN_SEEDS = [7]
DRY_RUN_EPISODES = 1
DRY_RUN_STEPS = 50

RESOURCE_LOOKAHEAD_STEPS = 10
EPS = 1e-9


ARMS = [
    {
        "arm": "ARM_0_canonical_full",
        "mode": "full",
        "contact_memory": False,
    },
    {
        "arm": "ARM_1_contact_memory_full",
        "mode": "full",
        "contact_memory": True,
    },
    {
        "arm": "ARM_2_contact_memory_zgoal_only",
        "mode": "zgoal_only",
        "contact_memory": True,
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
    if arm["mode"] == "full":
        return REEConfig.goal_stream(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            alpha_world=0.9,
            world_dim=32,
            use_resource_proximity_head=True,
        )

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        z_goal_enabled=True,
        goal_weight=0.5,
        benefit_threshold=0.05,
        drive_weight=2.0,
        schema_wanting_enabled=False,
        wanting_weight=0.0,
    )
    cfg.residue.valence_enabled = True
    return cfg


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _entropy_from_values(values: List[int]) -> float:
    counts: Dict[int, int] = {}
    for v in values:
        counts[int(v)] = counts.get(int(v), 0) + 1
    return _entropy_from_counts(counts)


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    if x is None or x.numel() == 0:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
            "std": 0.0,
        }
    xd = x.detach().float().flatten()
    return {
        "mean": float(xd.mean().item()),
        "min": float(xd.min().item()),
        "max": float(xd.max().item()),
        "range": float((xd.max() - xd.min()).item()),
        "std": float(xd.std(unbiased=False).item()) if xd.numel() > 1 else 0.0,
    }


def _pairwise_mean(x: torch.Tensor) -> float:
    if x is None or x.shape[0] < 2:
        return 0.0
    try:
        vals = torch.pdist(x.detach().float(), p=2)
    except RuntimeError:
        return 0.0
    if vals.numel() == 0:
        return 0.0
    return float(vals.mean().item())


def _candidate_world_summaries(
    candidates: List[Trajectory],
    fallback: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    first: List[torch.Tensor] = []
    final: List[torch.Tensor] = []
    for c in candidates:
        ws = c.get_world_state_sequence()
        if ws is not None:
            first.append(ws[0, 0, :].detach())
            final.append(ws[0, -1, :].detach())
        elif fallback is not None:
            first.append(fallback[0].detach())
            final.append(fallback[0].detach())
    if not first:
        if fallback is None:
            raise ValueError("No candidate world states and no fallback latent")
        first.append(fallback[0].detach())
        final.append(fallback[0].detach())
    return torch.stack(first, dim=0), torch.stack(final, dim=0)


def _candidate_action_classes(candidates: List[Trajectory]) -> List[int]:
    out: List[int] = []
    for c in candidates:
        out.append(int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item()))
    return out


def _pure_mech295_bias(
    agent: REEAgent,
    goal_state: Optional[GoalState],
    first_world: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros(first_world.shape[0], dtype=first_world.dtype, device=first_world.device)
    bridge = getattr(agent, "mech295_bridge", None)
    if bridge is None or goal_state is None or not goal_state.is_active():
        return zero
    cfg = bridge.config
    if float(cfg.liking_to_approach_cue_gain) == 0.0:
        return zero
    drive = float(getattr(goal_state, "_last_drive_level", 0.0))
    if drive < float(cfg.min_drive_to_fire):
        return zero
    prox = goal_state.goal_proximity(first_world).detach().clamp(min=0.0, max=1.0)
    return -float(cfg.liking_to_approach_cue_gain) * drive * prox


def _pure_mech307_bias(agent: REEAgent, first_world: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(first_world.shape[0], dtype=first_world.dtype, device=first_world.device)
    bridge = getattr(agent, "mech295_bridge", None)
    if bridge is None:
        return zero
    cfg = bridge.config
    if not bool(getattr(cfg, "use_mech307_conjunction_read", False)):
        return zero
    if float(getattr(cfg, "mech307_conjunction_gain", 0.0)) == 0.0:
        return zero
    residue = getattr(agent, "residue_field", None)
    if residue is None or not hasattr(residue, "evaluate_valence"):
        return zero
    drive = 0.0
    if agent.goal_state is not None:
        drive = float(getattr(agent.goal_state, "_last_drive_level", 0.0))
    if drive < float(cfg.min_drive_to_fire):
        return zero
    try:
        v = residue.evaluate_valence(first_world)
    except Exception:
        return zero
    if v is None or v.shape[0] != first_world.shape[0] or v.shape[1] < 4:
        return zero
    z_beta = 0.0
    latent = getattr(agent, "_current_latent", None)
    if latent is not None and latent.z_beta is not None and latent.z_beta.numel() > 0:
        z_beta = float(latent.z_beta[..., 0].abs().mean().item())
    cond = (
        (v[:, 0] > float(cfg.mech307_conjunction_wanting_threshold))
        & (v[:, 1] > float(cfg.mech307_conjunction_liking_threshold))
        & (v[:, 3] > 0.0)
        & (torch.full_like(v[:, 0], z_beta) > float(cfg.mech307_conjunction_z_beta_threshold))
    )
    return -float(cfg.mech307_conjunction_gain) * drive * cond.to(dtype=first_world.dtype)


def _score_components(
    agent: REEAgent,
    candidates: List[Trajectory],
    *,
    goal_state: Optional[GoalState],
    terrain_weight,
    z_harm_a,
    score_bias,
) -> Dict:
    e3 = agent.e3
    base = torch.stack([
        e3.score_trajectory(
            c,
            goal_state=None,
            terrain_weight=terrain_weight,
            z_harm_a=z_harm_a,
        ).mean()
        for c in candidates
    ]).detach()

    if goal_state is not None and goal_state.is_active() and e3.config.goal_weight > 0.0:
        goal_score = torch.stack([
            e3.compute_goal_score(c, goal_state).mean()
            for c in candidates
        ]).detach()
        goal_component = -float(e3.config.goal_weight) * goal_score
    else:
        goal_score = torch.zeros_like(base)
        goal_component = torch.zeros_like(base)

    fallback = None
    latent = getattr(agent, "_current_latent", None)
    if latent is not None:
        fallback = latent.z_world
    first_world, final_world = _candidate_world_summaries(candidates, fallback)
    action_classes = _candidate_action_classes(candidates)

    m295_bias = _pure_mech295_bias(agent, goal_state, first_world).detach()
    m307_bias = _pure_mech307_bias(agent, first_world).detach()
    actual_bias = (
        torch.zeros_like(base)
        if score_bias is None
        else score_bias.detach().to(dtype=base.dtype, device=base.device)
    )
    bias_residual = actual_bias - m295_bias.to(base.device) - m307_bias.to(base.device)
    total_reconstructed = base + goal_component.to(base.device) + actual_bias
    base_plus_goal = base + goal_component.to(base.device)

    return {
        "n_candidates": int(len(candidates)),
        "candidate_action_entropy": _entropy_from_values(action_classes),
        "candidate_unique_action_classes": int(len(set(action_classes))),
        "candidate_first_world_pairwise_mean": _pairwise_mean(first_world),
        "candidate_final_world_pairwise_mean": _pairwise_mean(final_world),
        "base": base,
        "goal_score": goal_score,
        "goal_component": goal_component,
        "m295_bias": m295_bias,
        "m307_bias": m307_bias,
        "actual_bias": actual_bias,
        "bias_residual": bias_residual,
        "base_plus_goal": base_plus_goal,
        "total_reconstructed": total_reconstructed,
        "base_argmin": int(base.argmin().item()),
        "base_plus_goal_argmin": int(base_plus_goal.argmin().item()),
        "reconstructed_argmin": int(total_reconstructed.argmin().item()),
        "base_argmin_action": int(action_classes[int(base.argmin().item())]),
        "base_plus_goal_argmin_action": int(action_classes[int(base_plus_goal.argmin().item())]),
        "reconstructed_argmin_action": int(action_classes[int(total_reconstructed.argmin().item())]),
    }


def _metric_template(seed: int, arm: Dict) -> Dict:
    return {
        "seed": int(seed),
        "arm": arm["arm"],
        "mode": arm["mode"],
        "contact_memory": bool(arm["contact_memory"]),
        "total_steps": 0,
        "resource_contact_count": 0,
        "benefit_exposure_count": 0,
        "contact_memory_writes": 0,
        "schema_salience_records": [],
        "score_records": [],
        "action_counts": {},
    }


def _record_score(metrics: Dict, pre: Dict, result) -> None:
    final_scores = result.scores.detach()
    selected_idx = int(result.selected_index)
    selected_action = int(result.selected_action.argmax(dim=-1).flatten()[0].item())

    rec = {
        "selected_index": selected_idx,
        "selected_action": selected_action,
        "committed": bool(result.committed),
        "precision": float(result.precision),
        "candidate_action_entropy": float(pre["candidate_action_entropy"]),
        "candidate_unique_action_classes": int(pre["candidate_unique_action_classes"]),
        "candidate_first_world_pairwise_mean": float(pre["candidate_first_world_pairwise_mean"]),
        "candidate_final_world_pairwise_mean": float(pre["candidate_final_world_pairwise_mean"]),
        "base_argmin": int(pre["base_argmin"]),
        "base_plus_goal_argmin": int(pre["base_plus_goal_argmin"]),
        "reconstructed_argmin": int(pre["reconstructed_argmin"]),
        "final_argmin": int(final_scores.argmin().item()),
        "base_argmin_action": int(pre["base_argmin_action"]),
        "base_plus_goal_argmin_action": int(pre["base_plus_goal_argmin_action"]),
        "reconstructed_argmin_action": int(pre["reconstructed_argmin_action"]),
        "base_stats": _tensor_stats(pre["base"]),
        "goal_component_stats": _tensor_stats(pre["goal_component"]),
        "goal_score_stats": _tensor_stats(pre["goal_score"]),
        "m295_bias_stats": _tensor_stats(pre["m295_bias"]),
        "m307_bias_stats": _tensor_stats(pre["m307_bias"]),
        "actual_bias_stats": _tensor_stats(pre["actual_bias"]),
        "bias_residual_stats": _tensor_stats(pre["bias_residual"]),
        "final_score_stats": _tensor_stats(final_scores),
    }
    rec["argmin_changed_by_goal"] = bool(rec["base_argmin"] != rec["base_plus_goal_argmin"])
    rec["argmin_changed_by_goal_or_bias"] = bool(rec["base_argmin"] != rec["final_argmin"])
    rec["selected_matches_final_argmin"] = bool(selected_idx == rec["final_argmin"])
    metrics["score_records"].append(rec)


def _install_e3_select_probe(agent: REEAgent, metrics: Dict) -> None:
    real_select = agent.e3.select

    def select_spy(candidates, temperature=1.0, *args, **kwargs):
        pre = _score_components(
            agent,
            candidates,
            goal_state=kwargs.get("goal_state"),
            terrain_weight=kwargs.get("terrain_weight"),
            z_harm_a=kwargs.get("z_harm_a"),
            score_bias=kwargs.get("score_bias"),
        )
        result = real_select(candidates, temperature, *args, **kwargs)
        _record_score(metrics, pre, result)
        return result

    agent.e3.select = select_spy


def _summarize_schema(records: List[Dict]) -> Dict:
    if not records:
        return {
            "n": 0,
            "future_resource_n": 0,
            "non_future_n": 0,
            "salience_mean": 0.0,
            "salience_future_resource_mean": 0.0,
            "salience_non_future_mean": 0.0,
            "salience_selectivity_delta": 0.0,
            "threshold_cross_fraction": 0.0,
            "threshold_cross_future_resource_fraction": 0.0,
            "threshold_cross_non_future_fraction": 0.0,
        }
    all_sal = [float(r["schema_salience"]) for r in records]
    future = [float(r["schema_salience"]) for r in records if r["future_resource"]]
    non = [float(r["schema_salience"]) for r in records if not r["future_resource"]]
    all_cross = [bool(r["schema_crossed"]) for r in records]
    future_cross = [bool(r["schema_crossed"]) for r in records if r["future_resource"]]
    non_cross = [bool(r["schema_crossed"]) for r in records if not r["future_resource"]]

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def frac(xs: List[bool]) -> float:
        return float(np.mean([1.0 if x else 0.0 for x in xs])) if xs else 0.0

    return {
        "n": len(records),
        "future_resource_n": len(future),
        "non_future_n": len(non),
        "salience_mean": mean(all_sal),
        "salience_future_resource_mean": mean(future),
        "salience_non_future_mean": mean(non),
        "salience_selectivity_delta": mean(future) - mean(non),
        "threshold_cross_fraction": frac(all_cross),
        "threshold_cross_future_resource_fraction": frac(future_cross),
        "threshold_cross_non_future_fraction": frac(non_cross),
    }


def _summarize_scores(records: List[Dict]) -> Dict:
    if not records:
        return {"n": 0}

    def mean_key(key: str) -> float:
        return float(np.mean([float(r[key]) for r in records]))

    def mean_stat(stat_group: str, stat_name: str) -> float:
        return float(np.mean([float(r[stat_group][stat_name]) for r in records]))

    selected_counts: Dict[int, int] = {}
    base_counts: Dict[int, int] = {}
    total_counts: Dict[int, int] = {}
    for r in records:
        selected_counts[int(r["selected_action"])] = selected_counts.get(int(r["selected_action"]), 0) + 1
        base_counts[int(r["base_argmin_action"])] = base_counts.get(int(r["base_argmin_action"]), 0) + 1
        total_counts[int(r["final_argmin"])] = total_counts.get(int(r["final_argmin"]), 0) + 1

    return {
        "n": len(records),
        "candidate_action_entropy_mean": mean_key("candidate_action_entropy"),
        "candidate_unique_action_classes_mean": mean_key("candidate_unique_action_classes"),
        "candidate_first_world_pairwise_mean": mean_key("candidate_first_world_pairwise_mean"),
        "candidate_final_world_pairwise_mean": mean_key("candidate_final_world_pairwise_mean"),
        "selected_action_entropy_on_e3_ticks": _entropy_from_counts(selected_counts),
        "base_argmin_action_entropy": _entropy_from_counts(base_counts),
        "base_score_range_mean": mean_stat("base_stats", "range"),
        "goal_component_range_mean": mean_stat("goal_component_stats", "range"),
        "m295_bias_range_mean": mean_stat("m295_bias_stats", "range"),
        "m307_bias_range_mean": mean_stat("m307_bias_stats", "range"),
        "actual_bias_range_mean": mean_stat("actual_bias_stats", "range"),
        "bias_residual_range_mean": mean_stat("bias_residual_stats", "range"),
        "final_score_range_mean": mean_stat("final_score_stats", "range"),
        "goal_to_base_range_ratio": (
            mean_stat("goal_component_stats", "range")
            / max(EPS, mean_stat("base_stats", "range"))
        ),
        "bias_to_base_range_ratio": (
            mean_stat("actual_bias_stats", "range")
            / max(EPS, mean_stat("base_stats", "range"))
        ),
        "argmin_changed_by_goal_fraction": float(np.mean([
            1.0 if r["argmin_changed_by_goal"] else 0.0 for r in records
        ])),
        "argmin_changed_by_goal_or_bias_fraction": float(np.mean([
            1.0 if r["argmin_changed_by_goal_or_bias"] else 0.0 for r in records
        ])),
        "selected_matches_final_argmin_fraction": float(np.mean([
            1.0 if r["selected_matches_final_argmin"] else 0.0 for r in records
        ])),
        "selected_action_counts_on_e3_ticks": {
            str(k): int(v) for k, v in sorted(selected_counts.items())
        },
    }


def _classify(row: Dict) -> str:
    schema = row["schema_summary"]
    scores = row["score_summary"]
    nonselective = (
        schema.get("threshold_cross_non_future_fraction", 0.0) >= 0.90
        and abs(schema.get("salience_selectivity_delta", 0.0)) < 0.05
    )
    candidate_collapse = (
        scores.get("candidate_action_entropy_mean", 0.0) < 0.10
        or scores.get("candidate_unique_action_classes_mean", 0.0) <= 1.25
    )
    goal_flat = scores.get("goal_component_range_mean", 0.0) < 1e-4
    goal_moves_argmin = scores.get("argmin_changed_by_goal_fraction", 0.0) > 0.10
    full_moves_argmin = scores.get("argmin_changed_by_goal_or_bias_fraction", 0.0) > 0.10
    selected_entropy = row.get("action_entropy", 0.0)
    if candidate_collapse:
        return "candidate_or_policy_collapse"
    if nonselective and not goal_moves_argmin:
        return "nonselective_schema_no_score_effect"
    if goal_flat:
        return "goal_component_flat"
    if full_moves_argmin and selected_entropy <= 0.05:
        return "score_moves_argmin_but_policy_locked"
    if goal_moves_argmin and selected_entropy > 0.05:
        return "fed_goal_changes_behavior"
    return "mixed_unresolved"


def _run_cell(seed: int, arm: Dict, episodes: int, steps_per_episode: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)
    metrics = _metric_template(seed, arm)
    _install_e3_select_probe(agent, metrics)
    bootstrap_pending = {"value": False}

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
        if arm["contact_memory"] and bootstrap_pending["value"]:
            if agent.goal_state is not None and latent.z_world is not None:
                agent.goal_state._z_goal = latent.z_world.detach().clone()
                norm = agent.goal_state.goal_norm()
                agent.goal_state._goal_norm_peak = max(
                    float(agent.goal_state._goal_norm_peak),
                    float(norm),
                )
                metrics["contact_memory_writes"] += 1
            bootstrap_pending["value"] = False

        sal = getattr(agent, "_schema_salience", None)
        sal_val = float(sal.squeeze().item()) if sal is not None else 0.0
        crossed = sal_val >= float(getattr(cfg, "schema_wanting_threshold", 0.3))
        metrics["schema_salience_records"].append({
            "schema_salience": sal_val,
            "schema_crossed": bool(crossed),
            "future_resource": False,
            "benefit_exposure": float(benefit_exposure),
        })

    def on_action(*, agent, latent, action, obs_dict, ticks, step) -> None:
        action_idx = int(action.argmax(dim=-1).item())
        counts = metrics["action_counts"]
        counts[action_idx] = counts.get(action_idx, 0) + 1

    hooks = StepHooks(on_sense=on_sense, on_action=on_action)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        bootstrap_pending["value"] = False
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            metrics["total_steps"] += 1
            ttype = result.info.get("transition_type", "none")
            if ttype == "resource":
                metrics["resource_contact_count"] += 1
                bootstrap_pending["value"] = True
                for r in metrics["schema_salience_records"][-RESOURCE_LOOKAHEAD_STEPS:]:
                    r["future_resource"] = True
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (ep + 1) == episodes:
            print(
                f"  seed={seed} arm={arm['arm']} ep={ep + 1}/{episodes}",
                flush=True,
            )

    metrics["action_counts"] = {
        str(k): int(v) for k, v in sorted(metrics["action_counts"].items())
    }
    metrics["action_entropy"] = _entropy_from_counts({
        int(k): int(v) for k, v in metrics["action_counts"].items()
    })
    metrics["schema_summary"] = _summarize_schema(metrics["schema_salience_records"])
    metrics["score_summary"] = _summarize_scores(metrics["score_records"])
    metrics["goal_norm_peak"] = (
        float(agent.goal_state._goal_norm_peak)
        if agent.goal_state is not None else 0.0
    )
    metrics["goal_norm_final"] = (
        agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
    )
    metrics["bridge_cue_fires"] = int(
        getattr(getattr(agent, "mech295_bridge", None), "_n_cue_fires", 0) or 0
    )
    metrics["bridge_conjunction_fires"] = int(
        getattr(getattr(agent, "mech295_bridge", None), "_n_conjunction_fires", 0) or 0
    )
    metrics["classification"] = _classify(metrics)
    metrics["schema_salience_records"] = metrics["schema_salience_records"][:25]
    metrics["score_records"] = metrics["score_records"][:25]
    return metrics


def _aggregate(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for arm in ARMS:
        arm_name = arm["arm"]
        subset = [r for r in rows if r["arm"] == arm_name]
        if not subset:
            continue

        def mean_path(path: List[str]) -> float:
            vals: List[float] = []
            for r in subset:
                cur = r
                for p in path:
                    cur = cur.get(p, {}) if isinstance(cur, dict) else {}
                vals.append(float(cur or 0.0))
            return float(np.mean(vals)) if vals else 0.0

        classifications: Dict[str, int] = {}
        for r in subset:
            classifications[r["classification"]] = classifications.get(r["classification"], 0) + 1
        out[arm_name] = {
            "arm": arm_name,
            "n_seeds": len(subset),
            "classifications": classifications,
            "action_entropy_mean": float(np.mean([r["action_entropy"] for r in subset])),
            "resource_contact_mean": float(np.mean([r["resource_contact_count"] for r in subset])),
            "contact_memory_writes_mean": float(np.mean([r["contact_memory_writes"] for r in subset])),
            "goal_norm_peak_max": float(np.max([r["goal_norm_peak"] for r in subset])),
            "schema_threshold_cross_non_future_fraction_mean": mean_path([
                "schema_summary", "threshold_cross_non_future_fraction",
            ]),
            "schema_salience_selectivity_delta_mean": mean_path([
                "schema_summary", "salience_selectivity_delta",
            ]),
            "candidate_action_entropy_mean": mean_path([
                "score_summary", "candidate_action_entropy_mean",
            ]),
            "candidate_unique_action_classes_mean": mean_path([
                "score_summary", "candidate_unique_action_classes_mean",
            ]),
            "candidate_final_world_pairwise_mean": mean_path([
                "score_summary", "candidate_final_world_pairwise_mean",
            ]),
            "base_score_range_mean": mean_path([
                "score_summary", "base_score_range_mean",
            ]),
            "goal_component_range_mean": mean_path([
                "score_summary", "goal_component_range_mean",
            ]),
            "actual_bias_range_mean": mean_path([
                "score_summary", "actual_bias_range_mean",
            ]),
            "goal_to_base_range_ratio_mean": mean_path([
                "score_summary", "goal_to_base_range_ratio",
            ]),
            "bias_to_base_range_ratio_mean": mean_path([
                "score_summary", "bias_to_base_range_ratio",
            ]),
            "argmin_changed_by_goal_fraction_mean": mean_path([
                "score_summary", "argmin_changed_by_goal_fraction",
            ]),
            "argmin_changed_by_goal_or_bias_fraction_mean": mean_path([
                "score_summary", "argmin_changed_by_goal_or_bias_fraction",
            ]),
        }
    return out


def _evaluate(rows: List[Dict], agg: Dict[str, Dict]) -> Dict:
    finite = True
    has_score_samples = True
    for row in rows:
        if row["score_summary"].get("n", 0) <= 0:
            has_score_samples = False
        for key in ("action_entropy", "goal_norm_peak"):
            finite = finite and math.isfinite(float(row.get(key, 0.0)))
        for summary_name in ("schema_summary", "score_summary"):
            for val in row[summary_name].values():
                if isinstance(val, (int, float)):
                    finite = finite and math.isfinite(float(val))

    canonical = agg.get("ARM_0_canonical_full", {})
    memory_full = agg.get("ARM_1_contact_memory_full", {})
    zgoal_only = agg.get("ARM_2_contact_memory_zgoal_only", {})
    return {
        "all_pass": bool(finite and has_score_samples),
        "finite_metrics": bool(finite),
        "has_score_samples": bool(has_score_samples),
        "canonical_schema_nonselective": bool(
            canonical.get("schema_threshold_cross_non_future_fraction_mean", 0.0) >= 0.90
            and abs(canonical.get("schema_salience_selectivity_delta_mean", 0.0)) < 0.05
        ),
        "canonical_candidate_action_entropy_mean": float(
            canonical.get("candidate_action_entropy_mean", 0.0)
        ),
        "contact_memory_full_action_entropy_delta": float(
            memory_full.get("action_entropy_mean", 0.0)
            - canonical.get("action_entropy_mean", 0.0)
        ),
        "contact_memory_zgoal_only_goal_argmin_change_fraction": float(
            zgoal_only.get("argmin_changed_by_goal_fraction_mean", 0.0)
        ),
        "contact_memory_full_argmin_change_fraction": float(
            memory_full.get("argmin_changed_by_goal_or_bias_fraction_mean", 0.0)
        ),
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
                f"schema_non_future={row['schema_summary']['threshold_cross_non_future_fraction']:.3f} "
                f"schema_delta={row['schema_summary']['salience_selectivity_delta']:.4f} "
                f"cand_ent={row['score_summary'].get('candidate_action_entropy_mean', 0.0):.3f} "
                f"goal_rng={row['score_summary'].get('goal_component_range_mean', 0.0):.6f} "
                f"arg_goal={row['score_summary'].get('argmin_changed_by_goal_fraction', 0.0):.3f} "
                f"arg_total={row['score_summary'].get('argmin_changed_by_goal_or_bias_fraction', 0.0):.3f} "
                f"act_ent={row['action_entropy']:.3f}",
                flush=True,
            )

    agg = _aggregate(rows)
    acceptance = _evaluate(rows, agg)
    elapsed = time.time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] aggregate summary", flush=True)
    for arm_name, arm_agg in agg.items():
        print(
            f"  {arm_name}: class={arm_agg['classifications']} "
            f"act_ent={arm_agg['action_entropy_mean']:.3f} "
            f"schema_non_future={arm_agg['schema_threshold_cross_non_future_fraction_mean']:.3f} "
            f"cand_ent={arm_agg['candidate_action_entropy_mean']:.3f} "
            f"goal_rng={arm_agg['goal_component_range_mean']:.6f} "
            f"bias_rng={arm_agg['actual_bias_range_mean']:.6f} "
            f"arg_goal={arm_agg['argmin_changed_by_goal_fraction_mean']:.3f} "
            f"arg_total={arm_agg['argmin_changed_by_goal_or_bias_fraction_mean']:.3f}",
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
            "Goal-stream selectivity and score-decomposition diagnostic. "
            "Non-contributory by design; localizes whether zero action "
            "entropy after V3-EXQ-559a is caused by nonselective schema "
            "salience, flat goal components, score swamping, candidate "
            "collapse, or absence of clean goal attractor seeding."
        ),
        "elapsed_seconds": elapsed,
        "seeds": seeds,
        "episodes": episodes,
        "steps_per_episode": steps,
        "resource_lookahead_steps": RESOURCE_LOOKAHEAD_STEPS,
        "arms": list(agg.values()),
        "per_seed_per_arm": rows,
        "acceptance": acceptance,
        "recent_context": {
            "supersedes_diagnostic": "v3_exq_559a_goal_stream_canonical",
            "goal_stream_canonical_path_landed": True,
            "action_entropy_remained_zero_in_559a": True,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
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
