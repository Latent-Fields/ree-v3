#!/opt/local/bin/python3
"""
V3-EXQ-558 -- Agent seed-pair readout/rank diagnostic.

Claims: [] (monostrategy-investigation diagnostic; no substrate claim under test)

Purpose
-------
V3-EXQ-555 showed that, at fixed env_seed=42, agent_seed=7 escapes the
monostrategy basin while agent_seed=42 collapses. V3-EXQ-557 then showed
that this is not a one-off: most agent initializations collapse, with a few
rare high-entropy escapes. V3-EXQ-556 tried module-level init swapping, but
its seed-7 baseline failed to replicate, so the monkey-patch path is not a
reliable next inference.

This experiment compares the clean seed pair directly, with no monkey-patch:

    ARM_ESCAPER:   env_seed=42, agent_seed=7
    ARM_COLLAPSED: env_seed=42, agent_seed=42

It asks where the two agents diverge after identical training depth:

  * candidate/readout diversity: do the candidates already differ?
  * rank/scoring diversity: do score argmins collapse onto one action class?
  * selection/propagation diversity: do selected candidates differ, but executed
    actions still collapse through commitment or held-action propagation?

Design
------
* Reuses V3-EXQ-555's factored seed helper and P0/P1 config.
* P0 = 40 training episodes, P1 = 60 eval episodes, 200 steps/episode.
* env_seed fixed at 42 in both cells.
* agent_seed is the variable: 7 vs 42.
* P1 is instrumented at fresh E3 proposal ticks. All executed action classes
  are counted across the whole P1 window.
* No ree_core edits and no experiment-side monkey-patching.

Pre-registered interpretation grid
----------------------------------
R0_replication_failure:
    seed 7 does not clear executed entropy >= 0.30, or seed 42 does not stay
    below executed entropy < 0.10. Do not use localization rows.

R1_candidate_readout_diff:
    seed 42 is already low-diversity at candidate/readout level while seed 7
    is not. The root is upstream of ranking: proposer, action-object decoder,
    or candidate formation.

R2_rank_score_diff:
    candidate/readout diversity is comparable, but score argmin action-class
    entropy collapses in seed 42 and not in seed 7. The root is score/rank
    geometry: terrain/E3 scoring, score margins, or candidate ordering.

R3_selection_propagation_diff:
    candidate/readout and argmin diversity are comparable, but selected or
    executed action entropy collapses only in seed 42. The root is selection,
    commitment, softmax/argmax, or held-action propagation.

R4_mixed_or_unresolved:
    more than one layer differs, or aggregate summaries do not isolate a clean
    first divergence. Use the manifest's per-layer metrics for the next probe.

PASS = both cells complete with finite metrics. The interpretation row, not the
verdict, drives the follow-up.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.optim as optim

from experiment_protocol import emit_outcome

from experiments.v3_exq_555_seed7_env_factorization import (
    ENV_KWARGS,
    LR_E1,
    LR_E2_WF,
    LR_E3_HARM,
    LR_ENC_AUX,
    P0_TRAIN_EPISODES,
    P1_EVAL_EPISODES,
    STEPS_PER_EPISODE,
    WF_BUF_MAX,
    HARM_EVAL_BUF_MAX,
    _action_to_onehot,
    _make_agent_and_env,
    _obs_accum,
    _obs_harm,
    _obs_harm_a,
    _obs_harm_history,
    _obs_resource_prox,
    _run_one_phase,
    _shannon_entropy,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_558_agent_seed_readout_rank_diagnostic"
QUEUE_ID = "V3-EXQ-558"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

ENV_SEED_FIXED = 42
CELLS: List[Tuple[str, int]] = [
    ("ARM_ESCAPER_seed7", 7),
    ("ARM_COLLAPSED_seed42", 42),
]

TEMPERATURE = 1.0
ACTION_ENTROPY_DIVERSE = 0.30
ACTION_ENTROPY_COLLAPSED = 0.10
LAYER_DIFF_DELTA = 0.25


def _make_optimizers(agent) -> Dict:
    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
    )
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)
    return {
        "e1_optimizer": e1_optimizer,
        "e2_wf_optimizer": e2_wf_optimizer,
        "harm_eval_optimizer": harm_eval_optimizer,
        "aux_optimizer": aux_optimizer,
        "aux_params": aux_params,
        "wf_buf": [],
        "harm_eval_buf": [],
    }


def _pairwise_stats(tensors: List[torch.Tensor]) -> Dict[str, float]:
    if len(tensors) < 2:
        if not tensors:
            return {
                "pairwise_l2_mean": 0.0,
                "pairwise_l2_min": 0.0,
                "elementwise_var_mean": 0.0,
            }
        flat = tensors[0].detach().reshape(1, -1).float()
        return {
            "pairwise_l2_mean": 0.0,
            "pairwise_l2_min": 0.0,
            "elementwise_var_mean": float(flat.var(unbiased=False).item()),
        }
    flat = torch.stack([t.detach().reshape(-1).float() for t in tensors], dim=0)
    diffs = flat.unsqueeze(0) - flat.unsqueeze(1)
    l2 = diffs.norm(dim=-1)
    idx = torch.triu_indices(flat.shape[0], flat.shape[0], offset=1)
    pairs = l2[idx[0], idx[1]]
    return {
        "pairwise_l2_mean": float(pairs.mean().item()),
        "pairwise_l2_min": float(pairs.min().item()),
        "elementwise_var_mean": float(flat.var(dim=0, unbiased=False).mean().item()),
    }


def _top2_margin(vec: torch.Tensor) -> float:
    flat = vec.detach().reshape(-1).float()
    if flat.numel() < 2:
        return 0.0
    vals = torch.topk(flat, k=2).values
    return float((vals[0] - vals[1]).item())


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _bump_count(counts: Dict[int, int], key: int) -> None:
    counts[key] = counts.get(key, 0) + 1


def _counts_as_str_keys(counts: Dict[int, int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counts.items())}


def _candidate_metrics(candidates) -> Dict:
    first_actions: List[torch.Tensor] = []
    first_action_objects: List[torch.Tensor] = []
    final_worlds: List[torch.Tensor] = []
    final_selfs: List[torch.Tensor] = []
    action_classes: List[int] = []
    action_top2_margins: List[float] = []

    for traj in candidates:
        first_action = traj.actions[:, 0, :].detach().squeeze(0)
        first_actions.append(first_action)
        action_classes.append(int(first_action.argmax().item()))
        action_top2_margins.append(_top2_margin(first_action))

        ao_seq = traj.get_action_object_sequence()
        if ao_seq is not None:
            first_action_objects.append(ao_seq[:, 0, :].detach().squeeze(0))

        if traj.world_states is not None and traj.world_states:
            final_worlds.append(traj.world_states[-1].detach().squeeze(0))
        if traj.states:
            final_selfs.append(traj.states[-1].detach().squeeze(0))

    candidate_counts: Dict[int, int] = {}
    for cls in action_classes:
        _bump_count(candidate_counts, cls)

    return {
        "k_candidates": len(candidates),
        "candidate_action_classes": action_classes,
        "candidate_action_class_counts": _counts_as_str_keys(candidate_counts),
        "candidate_action_class_entropy": _shannon_entropy(candidate_counts),
        "candidate_action_class_cardinality": len(candidate_counts),
        "first_action_pairwise": _pairwise_stats(first_actions),
        "first_action_object_pairwise": _pairwise_stats(first_action_objects),
        "final_world_pairwise": _pairwise_stats(final_worlds),
        "final_self_pairwise": _pairwise_stats(final_selfs),
        "candidate_action_top2_margin_mean": _mean(action_top2_margins),
        "candidate_action_top2_margin_min": min(action_top2_margins)
        if action_top2_margins else 0.0,
    }


def _score_metrics(scores: torch.Tensor, candidate_classes: List[int]) -> Dict:
    scores_cpu = scores.detach().reshape(-1).float().cpu()
    if scores_cpu.numel() == 0:
        return {
            "score_var": 0.0,
            "score_range": 0.0,
            "best_second_margin": 0.0,
            "argmin_index": None,
            "argmin_action_class": None,
            "softmax_prob_entropy": 0.0,
            "softmax_top_prob": 0.0,
        }
    sorted_vals, sorted_idx = torch.sort(scores_cpu)
    argmin_index = int(sorted_idx[0].item())
    margin = (
        float((sorted_vals[1] - sorted_vals[0]).item())
        if sorted_vals.numel() >= 2 else 0.0
    )
    probs = torch.softmax(-scores_cpu / TEMPERATURE, dim=0)
    prob_entropy = 0.0
    for p in probs.tolist():
        if p > 0.0:
            prob_entropy -= float(p) * math.log(float(p))
    argmin_class = (
        int(candidate_classes[argmin_index])
        if 0 <= argmin_index < len(candidate_classes) else None
    )
    return {
        "score_var": float(scores_cpu.var(unbiased=False).item()),
        "score_range": float((scores_cpu.max() - scores_cpu.min()).item()),
        "best_second_margin": margin,
        "argmin_index": argmin_index,
        "argmin_action_class": argmin_class,
        "softmax_prob_entropy": prob_entropy,
        "softmax_top_prob": float(probs.max().item()),
    }


def _selected_index(agent, candidates) -> Optional[int]:
    selected = getattr(agent.e3, "_last_selected_trajectory", None)
    if selected is None:
        return None
    for idx, cand in enumerate(candidates):
        if cand is selected:
            return idx
    return None


def _sample_record(records: List[Dict], fresh_idx: int, record: Dict) -> None:
    if len(records) < 100 or fresh_idx % 25 == 0:
        records.append(record)


def _run_eval_phase_instrumented(agent, env, rng_module) -> Dict:
    device = agent.device
    action_dim = env.action_dim
    agent.eval()

    executed_counts: Dict[int, int] = {}
    fresh_selected_counts: Dict[int, int] = {}
    fresh_argmin_counts: Dict[int, int] = {}

    candidate_entropy_values: List[float] = []
    candidate_cardinality_values: List[float] = []
    first_action_l2_values: List[float] = []
    first_action_object_l2_values: List[float] = []
    final_world_l2_values: List[float] = []
    final_self_l2_values: List[float] = []
    score_var_values: List[float] = []
    score_range_values: List[float] = []
    score_margin_values: List[float] = []
    softmax_entropy_values: List[float] = []
    softmax_top_prob_values: List[float] = []
    selected_matches_argmin_values: List[float] = []
    selected_action_margin_values: List[float] = []
    beta_elevated_after_values: List[float] = []
    running_variance_values: List[float] = []

    fresh_records_sample: List[Dict] = []
    n_total_actions = 0
    n_fresh_ticks = 0
    n_nonfinite_actions = 0
    prev_candidates_id: Optional[int] = None

    for ep in range(P1_EVAL_EPISODES):
        _flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        prev_candidates_id = None

        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, latent.z_world.shape[-1], device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level = agent.compute_drive_level(obs_body)
            benefit_exposure = max(
                0.0, float(obs_dict.get("benefit_exposure", 0.0)),
            )
            agent.update_z_goal(
                benefit_exposure=benefit_exposure,
                drive_level=drive_level,
            )

            cur_candidates_id = id(candidates)
            fresh = bool(candidates) and cur_candidates_id != prev_candidates_id
            prev_candidates_id = cur_candidates_id
            cand_metrics: Optional[Dict] = None
            if fresh:
                cand_metrics = _candidate_metrics(candidates)

            action = agent.select_action(candidates, ticks, temperature=TEMPERATURE)
            if action is None:
                action = _action_to_onehot(
                    rng_module.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = action

            if not torch.isfinite(action).all():
                n_nonfinite_actions += 1
                action = _action_to_onehot(0, action_dim, device)
                agent._last_action = action

            action_idx = int(action[0].argmax().item())
            _bump_count(executed_counts, action_idx)
            n_total_actions += 1

            if fresh and cand_metrics is not None:
                n_fresh_ticks += 1
                candidate_classes = [
                    int(v) for v in cand_metrics["candidate_action_classes"]
                ]
                scores = getattr(agent.e3, "last_scores", None)
                score_metrics = (
                    _score_metrics(scores, candidate_classes)
                    if scores is not None else _score_metrics(torch.tensor([]), [])
                )
                sel_idx = _selected_index(agent, candidates)
                selected_class = (
                    int(candidate_classes[sel_idx])
                    if sel_idx is not None and 0 <= sel_idx < len(candidate_classes)
                    else None
                )
                if selected_class is not None:
                    _bump_count(fresh_selected_counts, selected_class)
                argmin_class = score_metrics["argmin_action_class"]
                if argmin_class is not None:
                    _bump_count(fresh_argmin_counts, int(argmin_class))

                selected_matches_argmin = (
                    sel_idx is not None
                    and score_metrics["argmin_index"] is not None
                    and int(sel_idx) == int(score_metrics["argmin_index"])
                )
                selected_matches_argmin_values.append(
                    1.0 if selected_matches_argmin else 0.0
                )

                selected_margin = 0.0
                if sel_idx is not None and 0 <= sel_idx < len(candidates):
                    selected_margin = _top2_margin(candidates[sel_idx].actions[:, 0, :])
                selected_action_margin_values.append(selected_margin)

                candidate_entropy_values.append(
                    float(cand_metrics["candidate_action_class_entropy"])
                )
                candidate_cardinality_values.append(
                    float(cand_metrics["candidate_action_class_cardinality"])
                )
                first_action_l2_values.append(
                    cand_metrics["first_action_pairwise"]["pairwise_l2_mean"]
                )
                first_action_object_l2_values.append(
                    cand_metrics["first_action_object_pairwise"]["pairwise_l2_mean"]
                )
                final_world_l2_values.append(
                    cand_metrics["final_world_pairwise"]["pairwise_l2_mean"]
                )
                final_self_l2_values.append(
                    cand_metrics["final_self_pairwise"]["pairwise_l2_mean"]
                )
                score_var_values.append(float(score_metrics["score_var"]))
                score_range_values.append(float(score_metrics["score_range"]))
                score_margin_values.append(float(score_metrics["best_second_margin"]))
                softmax_entropy_values.append(
                    float(score_metrics["softmax_prob_entropy"])
                )
                softmax_top_prob_values.append(float(score_metrics["softmax_top_prob"]))
                beta_elevated_after_values.append(
                    1.0 if agent.beta_gate.is_elevated else 0.0
                )
                running_variance_values.append(
                    float(getattr(agent.e3, "_running_variance", 0.0))
                )

                record = {
                    "episode": ep,
                    "step": step,
                    "fresh_tick_index": n_fresh_ticks,
                    "candidate_action_class_entropy": cand_metrics[
                        "candidate_action_class_entropy"
                    ],
                    "candidate_action_class_counts": cand_metrics[
                        "candidate_action_class_counts"
                    ],
                    "candidate_action_class_cardinality": cand_metrics[
                        "candidate_action_class_cardinality"
                    ],
                    "first_action_pairwise_l2_mean": cand_metrics[
                        "first_action_pairwise"
                    ]["pairwise_l2_mean"],
                    "first_action_object_pairwise_l2_mean": cand_metrics[
                        "first_action_object_pairwise"
                    ]["pairwise_l2_mean"],
                    "final_world_pairwise_l2_mean": cand_metrics[
                        "final_world_pairwise"
                    ]["pairwise_l2_mean"],
                    "score_var": score_metrics["score_var"],
                    "score_range": score_metrics["score_range"],
                    "best_second_margin": score_metrics["best_second_margin"],
                    "softmax_prob_entropy": score_metrics["softmax_prob_entropy"],
                    "softmax_top_prob": score_metrics["softmax_top_prob"],
                    "argmin_action_class": score_metrics["argmin_action_class"],
                    "selected_index": sel_idx,
                    "selected_action_class": selected_class,
                    "executed_action_class": action_idx,
                    "selected_matches_argmin": selected_matches_argmin,
                    "beta_elevated_after_select": bool(agent.beta_gate.is_elevated),
                    "running_variance": float(
                        getattr(agent.e3, "_running_variance", 0.0)
                    ),
                }
                _sample_record(fresh_records_sample, n_fresh_ticks, record)

            _flat_obs, harm_signal, done, _info, obs_dict = env.step(action)

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

    selected_entropy = _shannon_entropy(fresh_selected_counts)
    argmin_entropy = _shannon_entropy(fresh_argmin_counts)
    executed_entropy = _shannon_entropy(executed_counts)

    return {
        "p1_n_total_actions": n_total_actions,
        "p1_n_fresh_ticks": n_fresh_ticks,
        "p1_n_nonfinite_actions": n_nonfinite_actions,
        "p1_executed_action_class_counts": _counts_as_str_keys(executed_counts),
        "p1_executed_action_class_entropy": executed_entropy,
        "fresh_selected_action_class_counts": _counts_as_str_keys(
            fresh_selected_counts
        ),
        "fresh_selected_action_class_entropy": selected_entropy,
        "fresh_argmin_action_class_counts": _counts_as_str_keys(fresh_argmin_counts),
        "fresh_argmin_action_class_entropy": argmin_entropy,
        "candidate_action_class_entropy": _summary(candidate_entropy_values),
        "candidate_action_class_cardinality": _summary(candidate_cardinality_values),
        "first_action_pairwise_l2": _summary(first_action_l2_values),
        "first_action_object_pairwise_l2": _summary(first_action_object_l2_values),
        "final_world_pairwise_l2": _summary(final_world_l2_values),
        "final_self_pairwise_l2": _summary(final_self_l2_values),
        "score_var": _summary(score_var_values),
        "score_range": _summary(score_range_values),
        "score_best_second_margin": _summary(score_margin_values),
        "softmax_prob_entropy": _summary(softmax_entropy_values),
        "softmax_top_prob": _summary(softmax_top_prob_values),
        "selected_matches_argmin_rate": _mean(selected_matches_argmin_values),
        "selected_action_top2_margin": _summary(selected_action_margin_values),
        "beta_elevated_after_select_rate": _mean(beta_elevated_after_values),
        "running_variance_at_fresh_ticks": _summary(running_variance_values),
        "fresh_tick_records_sample": fresh_records_sample,
    }


def _run_cell(cell_label: str, agent_seed: int) -> Dict:
    print(
        f"Cell {cell_label}: env_seed={ENV_SEED_FIXED} agent_seed={agent_seed}",
        flush=True,
    )
    agent, env = _make_agent_and_env(
        env_seed=ENV_SEED_FIXED, agent_seed=agent_seed,
    )
    optimizers_and_params = _make_optimizers(agent)
    rng_module = random.Random(ENV_SEED_FIXED)

    p0_diag = _run_one_phase(
        agent=agent,
        env=env,
        phase_label="P0",
        num_episodes=P0_TRAIN_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        train=True,
        optimizers_and_params=optimizers_and_params,
        rng_module=rng_module,
        action_count_window=None,
    )
    print(
        f"  P0 complete: n_actions={p0_diag['n_total_actions']}",
        flush=True,
    )

    eval_diag = _run_eval_phase_instrumented(agent, env, rng_module)
    print(
        f"  P1 complete: executed_entropy="
        f"{eval_diag['p1_executed_action_class_entropy']:.4f} "
        f"fresh_selected_entropy="
        f"{eval_diag['fresh_selected_action_class_entropy']:.4f} "
        f"fresh_argmin_entropy="
        f"{eval_diag['fresh_argmin_action_class_entropy']:.4f}",
        flush=True,
    )
    print("verdict: PASS", flush=True)

    return {
        "cell": cell_label,
        "env_seed": ENV_SEED_FIXED,
        "agent_seed": agent_seed,
        "p0_n_total_actions": p0_diag["n_total_actions"],
        **eval_diag,
    }


def _classify(cells: Dict[str, Dict]) -> Tuple[str, str]:
    esc = cells.get("ARM_ESCAPER_seed7")
    col = cells.get("ARM_COLLAPSED_seed42")
    if esc is None or col is None:
        return ("R0_replication_failure", "Missing one or both seed-pair cells.")

    esc_exec = float(esc["p1_executed_action_class_entropy"])
    col_exec = float(col["p1_executed_action_class_entropy"])
    if esc_exec < ACTION_ENTROPY_DIVERSE or col_exec >= ACTION_ENTROPY_COLLAPSED:
        return (
            "R0_replication_failure",
            (
                f"Seed-pair anchor did not replicate: seed7 executed entropy "
                f"{esc_exec:.4f} (need >= {ACTION_ENTROPY_DIVERSE:.2f}); "
                f"seed42 executed entropy {col_exec:.4f} "
                f"(need < {ACTION_ENTROPY_COLLAPSED:.2f}). Do not use "
                f"localization rows."
            ),
        )

    esc_cand = float(esc["candidate_action_class_entropy"]["mean"])
    col_cand = float(col["candidate_action_class_entropy"]["mean"])
    esc_argmin = float(esc["fresh_argmin_action_class_entropy"])
    col_argmin = float(col["fresh_argmin_action_class_entropy"])
    esc_selected = float(esc["fresh_selected_action_class_entropy"])
    col_selected = float(col["fresh_selected_action_class_entropy"])

    candidate_diff = (
        esc_cand - col_cand >= LAYER_DIFF_DELTA
        or col_cand < ACTION_ENTROPY_COLLAPSED
    )
    rank_diff = (
        esc_argmin - col_argmin >= LAYER_DIFF_DELTA
        or col_argmin < ACTION_ENTROPY_COLLAPSED
    )
    selection_diff = (
        esc_selected - col_selected >= LAYER_DIFF_DELTA
        or col_selected < ACTION_ENTROPY_COLLAPSED
    )

    if candidate_diff:
        return (
            "R1_candidate_readout_diff",
            (
                f"First divergence is visible at candidate/readout level: "
                f"seed7 candidate entropy mean {esc_cand:.4f}, seed42 "
                f"{col_cand:.4f}. Route to proposer/action-object decoder/"
                f"candidate-formation diagnostics."
            ),
        )
    if rank_diff:
        return (
            "R2_rank_score_diff",
            (
                f"Candidate/readout entropy is comparable enough, but score "
                f"argmin entropy differs: seed7 {esc_argmin:.4f}, seed42 "
                f"{col_argmin:.4f}. Route to scoring/rank geometry: score "
                f"margins, terrain/E3 scoring, and candidate ordering."
            ),
        )
    if selection_diff:
        return (
            "R3_selection_propagation_diff",
            (
                f"Candidate and argmin summaries do not isolate the cliff, "
                f"but selected action entropy differs: seed7 {esc_selected:.4f}, "
                f"seed42 {col_selected:.4f}. Route to selection temperature, "
                f"commitment, or held-action propagation."
            ),
        )
    return (
        "R4_mixed_or_unresolved",
        (
            "Seed-pair anchors replicated, but aggregate layer summaries do "
            "not isolate a single first divergence. Inspect the sampled fresh "
            "tick records and per-layer summary metrics before queuing the "
            "next probe."
        ),
    )


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- clean seed-pair readout/rank diagnostic")
    print(f"queue_id={QUEUE_ID}")
    print(f"env_seed fixed: {ENV_SEED_FIXED}")
    print(f"cells: {CELLS}")
    print(
        f"P0 train: {P0_TRAIN_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print(
        f"P1 eval: {P1_EVAL_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print("Instrumentation: fresh E3 proposal ticks during P1 eval.", flush=True)
    print("No monkey-patching; no ree_core edits.", flush=True)


def _run_smoke() -> None:
    print(
        "SMOKE MODE: both cells, P0=1 ep x 20 steps, P1=1 ep x 20 steps; "
        "wiring/non-crash only.",
        flush=True,
    )
    original_p0 = globals()["P0_TRAIN_EPISODES"]
    original_p1 = globals()["P1_EVAL_EPISODES"]
    original_steps = globals()["STEPS_PER_EPISODE"]
    globals()["P0_TRAIN_EPISODES"] = 1
    globals()["P1_EVAL_EPISODES"] = 1
    globals()["STEPS_PER_EPISODE"] = 20
    try:
        results = [_run_cell(label, seed) for label, seed in CELLS]
        for r in results:
            print(
                f"  smoke {r['cell']} executed_counts="
                f"{r['p1_executed_action_class_counts']} "
                f"entropy={r['p1_executed_action_class_entropy']:.4f}",
                flush=True,
            )
    finally:
        globals()["P0_TRAIN_EPISODES"] = original_p0
        globals()["P1_EVAL_EPISODES"] = original_p1
        globals()["STEPS_PER_EPISODE"] = original_steps
    print("verdict: PASS", flush=True)
    print("SMOKE OK", flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return (None, None)
    if args.smoke:
        _run_smoke()
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell_results = [_run_cell(label, seed) for label, seed in CELLS]
    by_cell = {r["cell"]: r for r in per_cell_results}
    interpretation_row, interpretation_description = _classify(by_cell)

    all_finite = True
    for r in per_cell_results:
        checks = [
            float(r["p1_executed_action_class_entropy"]),
            float(r["fresh_selected_action_class_entropy"]),
            float(r["fresh_argmin_action_class_entropy"]),
            float(r["candidate_action_class_entropy"]["mean"]),
            float(r["score_var"]["mean"]),
        ]
        all_finite = all_finite and all(math.isfinite(x) for x in checks)

    outcome = "PASS" if all_finite else "FAIL"
    evidence_direction = "non_contributory" if outcome == "PASS" else "inconclusive"

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Interpretation row: {interpretation_row}", flush=True)
    print(f"  {interpretation_description}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": (
            "Monostrategy-investigation diagnostic. Clean seed-pair "
            "comparison of agent_seed=7 versus agent_seed=42 at fixed "
            "env_seed=42, with no monkey-patching and no ree_core edits. "
            "Reuses V3-EXQ-555 factored-seed helper and full run depth "
            "(P0=40, P1=60, 200 steps/episode). P1 instruments fresh E3 "
            "proposal ticks for candidate action-class entropy, first-action "
            "and action-object diversity, world/self rollout diversity, score "
            "variance/range/margins, softmax probability entropy, argmin "
            "action-class entropy, selected action-class entropy, and executed "
            "action-class entropy. Pre-registered interpretation rows: "
            "R0_replication_failure, R1_candidate_readout_diff, "
            "R2_rank_score_diff, R3_selection_propagation_diff, "
            "R4_mixed_or_unresolved. experiment_purpose=diagnostic; "
            "evidence_direction set to non_contributory on PASS so governance "
            "does not weight any claim from this run."
        ),
        "pass_criteria_summary": {
            "outcome_rule": (
                "PASS if both cells complete with finite aggregate metrics; "
                "interpretation_row drives follow-up."
            ),
            "interpretation_row": interpretation_row,
            "interpretation_description": interpretation_description,
            "anchor_thresholds": {
                "seed7_executed_entropy_min": ACTION_ENTROPY_DIVERSE,
                "seed42_executed_entropy_max_exclusive": ACTION_ENTROPY_COLLAPSED,
            },
        },
        "per_cell_results": per_cell_results,
        "config": {
            "queue_id": QUEUE_ID,
            "env_seed_fixed": ENV_SEED_FIXED,
            "cells": [{"cell": label, "agent_seed": seed} for label, seed in CELLS],
            "p0_train_episodes": P0_TRAIN_EPISODES,
            "p1_eval_episodes": P1_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "temperature": TEMPERATURE,
            "env_kwargs": ENV_KWARGS,
            "factored_seed_helper_source": (
                "experiments.v3_exq_555_seed7_env_factorization."
                "_make_agent_and_env"
            ),
            "no_monkey_patch": True,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_file}", flush=True)
    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
