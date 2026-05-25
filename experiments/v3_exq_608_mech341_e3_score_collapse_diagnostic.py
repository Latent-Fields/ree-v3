#!/opt/local/bin/python3
"""
V3-EXQ-608 -- MECH-341 E3 score-collapse diagnostic (Phase P2 of
behavioral_diversity_isolation_plan.md).

Claims: [MECH-341] (diagnostic; experiment_purpose=diagnostic; no claim
        weighting -- runs are excluded from confidence/conflict scoring per
        REE_assembly Phase-3 governance rules).

Purpose
-------
Apply isolation-plan decision rules R2.a / R2.b to determine whether MECH-341
substrate work (E3 scoring diversity preservation) is justified.

The 2026-05-25 4-theory analysis identified E3 score aggregation as the
highest-residual-leverage candidate for diversity collapse downstream of
ARC-065 SP-CEM. ARC-065's distributed pathway commits the architecture to
four substrate slots (LC-NE tonic noise, frontopolar curiosity, striatal
novelty, hippocampal trajectory sampling), but the post-CEM scoring step
at E3 has no corresponding diversity-preservation mechanism. The
acceptance-criteria doc names this as Rung-1 FP "diversity in candidates
that collapses to one class after E3 scoring", but until 2026-05-25
no positive MECH asserted that E3 must preserve.

This experiment is purely instrumentation -- no substrate change, no new
config flag exercised. It runs the current main-path stack (SP-CEM ON,
MECH-313 noise floor ON, V_s ON, SD-054 reef ON, z_goal ON, drive_weight
2.0 -- matching V3-EXQ-543k / V3-EXQ-605 wiring) and logs, per tick:

  - pre_e3_class_count:   number of distinct first-action classes among
                          the CEM-supplied candidate pool
  - pre_e3_classes:       set of first-action class ids in the pool
  - per_class_e3_stats:   {class_id -> {n, score_mean, score_std}}
                          where scores are read from agent.e3.last_scores
  - post_e3_selected_class: first-action class of the candidate whose E3
                          score was argmax (E3's own pick, BEFORE any
                          downstream noise-floor / commitment logic
                          rewrites the action)
  - score_top2_class_gap: per_class_mean[top_class] - per_class_mean[2nd]
                          (positive; quantifies how far the dominant
                          class wins by)
  - committed_class:      first-action class of the action that actually
                          reached env.step (post-noise-floor)

These per-tick records are aggregated to four decision-rule metrics:

  1. frac_ticks_pre_ge2:      fraction of measurement ticks with
                              pre_e3_class_count >= 2 (the precondition
                              for R2 to fire at all -- if this is low,
                              theory 1 / ARC-065 SP-CEM is the bottleneck,
                              not E3 scoring).

  2. frac_ticks_e3_collapse:  fraction of (pre_ge2) ticks where post_e3_
                              selected_class is the same across all
                              candidates in the top-K E3 set (K=1 by
                              construction -- E3 always selects one
                              trajectory; the relevant statistic is
                              whether E3's score distribution makes the
                              selection deterministic regardless of
                              proposal diversity). Operationalised as
                              "score_top2_class_gap > epsilon for
                              pre_ge2 ticks". epsilon = 0.05 of
                              score-range default.

  3. mean_top2_class_gap:     mean of score_top2_class_gap over pre_ge2
                              ticks. Distinguishes near-tie (mean << 1,
                              option 3 jitter sufficient) from large-gap
                              (mean ~ score-range, option 1 entropy
                              bonus or option 2 stratified argmax
                              required).

  4. frac_near_tie:           fraction of pre_ge2 ticks where
                              score_top2_class_gap < epsilon
                              (1 - frac_ticks_e3_collapse but reported
                              separately for the substrate-design choice
                              between MECH-341 implementation options 1
                              / 2 / 3 enumerated in the isolation plan).

Pre-registered interpretation grid
----------------------------------
Apply at end of P1 measurement window across all seeds:

  R2.a fires if frac_ticks_pre_ge2 >= 0.5 AND frac_ticks_e3_collapse >= 0.8:
    => theory 2 confirmed as real collapse site.
       MECH-341 substrate work priority-promoted.
       Substrate option selection by frac_near_tie:
         < 0.2 -> options 1/2 (entropy bonus or stratified argmax) required
         >= 0.2 -> option 3 (jittered tie-breaking) sufficient
                  (NB: combinations are not mutually exclusive).

  R2.b fires if frac_ticks_pre_ge2 >= 0.5 AND frac_ticks_e3_collapse < 0.5:
    => theory 2 not load-bearing. E3 is preserving the diversity it
       receives. Layer-B collapse is not the bottleneck; attention
       returns to layers A/C/D. MECH-341 retains as architectural
       commitment but no substrate work triggered.

  Mixed (frac_pre_ge2 >= 0.5, 0.5 <= frac_ticks_e3_collapse < 0.8):
    inconclusive at current sample size; re-run with heavier scope
    (5 seeds, 200 ep) before MECH-341 commitment.

  Low precondition (frac_ticks_pre_ge2 < 0.5):
    SP-CEM is not producing >=2 distinct classes >50%% of the time --
    diagnostic is non-informative for theory 2. Re-evaluate theory 1
    (CEM elite-pool collapse) via the V3-EXQ-569 matched-entropy
    control before re-running this diagnostic.

experiment_purpose=diagnostic. PASS = (a) instrumentation present
across all P1 ticks (no measurement gap), (b) >= 75%% of seeds run to
P1 completion without ERROR, (c) at least one of R2.a / R2.b /
inconclusive interpretation classes fires unambiguously. The
interpretation drives the next experiment, NOT a substrate change.

Phases
------
P0 (60 ep, instrumentation OFF): warmup. Online agent updates
(record_transition, update_z_goal) accumulate residue field / EWMA /
V_s anchor sets. Per autopsy V3-EXQ-603b finding: untrained inference on
SD-054 reef terminates 2/3 seeds at ~12-16 step/ep before substrate
gates fire. P0 budget set conservatively above the autopsy threshold.

P1 (40 ep, instrumentation ON): measurement window. Same step loop,
adds per-tick pre/post-E3 class accounting + per-class score stats.

Total: 100 ep per seed x 3 seeds = 300 ep x 200 steps = 60000 steps
maximum. Estimated runtime: ~3.5h on a 1050 Ti (matches autopsy 603b
seed-43 timing); ~1.5h on faster GPUs.

Implementation notes
--------------------
- The CEM proposer + E3 selector run on every step (verified via
  V3-EXQ-551a stage-3 instrumentation 2026-05-11). Per-tick measurement
  is well-defined.
- Pre-E3 classes derive from candidates[i].actions[:, 0, :].argmax(-1)
  -- same convention as hippocampal._trajectory_first_action_class.
- Per-class scores derive from agent.e3.last_scores indexed by
  candidate position. E3.select caches last_scores at every select()
  call (verified 2026-05-11, V3-EXQ-551a stage-3 path).
- "Committed class" is the action.argmax that reaches env.step --
  this can differ from post_e3_selected when noise_floor (MECH-313)
  resamples post-E3. Both are logged so the proposal->E3->commit
  collapse chain is fully traced.
- No agent module is monkey-patched. No ree_core modification.

See REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
and the sibling acceptance-criteria doc.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_608_mech341_e3_score_collapse_diagnostic"
QUEUE_ID = "V3-EXQ-608"
CLAIM_IDS: List[str] = ["MECH-341"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60
P1_MEASUREMENT_EPISODES = 40
TOTAL_TRAIN_EPISODES = P0_WARMUP_EPISODES + P1_MEASUREMENT_EPISODES
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Decision-rule thresholds (pre-registered, never derived from this run).
PRE_GE2_FRAC_GATE = 0.5
E3_COLLAPSE_FRAC_GATE = 0.8
E3_COLLAPSE_FRAC_FLOOR = 0.5  # below this -> R2.b fires
NEAR_TIE_FRAC_GATE = 0.2
# Score-gap epsilon is a fraction of the seed's observed score range.
# Computed per-seed from P1 measurements as 5%% of (max - min) over
# per-class mean scores. Pre-registered as a multiplier, not an
# absolute threshold.
SCORE_GAP_EPSILON_RANGE_FRAC = 0.05

# Acceptance gate: at least this many seeds must reach P1 completion.
MIN_SEEDS_FOR_PASS = 2  # of 3

# Mirror V3-EXQ-543k / V3-EXQ-605 reef-enabled SD-054 wiring.
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Main-path SP-CEM + MECH-313 + V_s + SD-054 stack. No substrate change."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        # Harm streams (matches main-path).
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        # Goal pipeline (matches 543k / 605 main-path).
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # Resource/benefit heads (matches main-path; needed for
        # update_z_goal to receive meaningful drive signal).
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        # ARC-065 SP-CEM child (Layer A; landed as default 2026-05-17).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # MECH-313 noise floor (Layer C). Default alpha 0.1.
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        # MECH-269 V_s regional verisimilitude (Layer D).
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers
# ---------------------------------------------------------------------------


def _trajectory_first_action_class(traj) -> int:
    """Mirror hippocampal.module._trajectory_first_action_class (static)."""
    return int(
        traj.actions[:, 0, :]
        .argmax(dim=-1)
        .detach()
        .reshape(-1)[0]
        .item()
    )


def _per_class_score_stats(
    candidates: List, last_scores: Optional[torch.Tensor]
) -> Tuple[Dict[int, Dict[str, float]], Optional[int], Optional[float], bool]:
    """Compute per-class E3 score stats and identify E3-selected class.

    Returns:
      per_class: {class_id -> {n, score_mean, score_std}}
      selected_class: first-action class of the candidate at argmax(last_scores),
                      or None if measurement was not possible.
      top2_gap: per_class_mean[top] - per_class_mean[2nd] (positive),
                or None if fewer than 2 distinct classes were present.
      logged: True iff at least 2 candidates AND last_scores aligns with len.

    By REE-v3 E3.select convention LOWER score is BETTER (residue cost),
    so "top class" = class with MINIMUM per-class mean score, and
    top2_gap = per_class_mean[2nd] - per_class_mean[top] (still positive).
    """
    if (
        not candidates
        or len(candidates) < 2
        or last_scores is None
        or last_scores.numel() != len(candidates)
    ):
        return {}, None, None, False

    scores_t = last_scores.detach().reshape(-1).float()
    # Aggregate per class.
    per_class_scores: Dict[int, List[float]] = {}
    classes_per_cand: List[int] = []
    for i, traj in enumerate(candidates):
        cls = _trajectory_first_action_class(traj)
        classes_per_cand.append(cls)
        per_class_scores.setdefault(cls, []).append(float(scores_t[i].item()))

    per_class: Dict[int, Dict[str, float]] = {}
    for cls, vals in per_class_scores.items():
        n = len(vals)
        mean_v = sum(vals) / n
        if n == 1:
            std_v = 0.0
        else:
            var_v = sum((v - mean_v) ** 2 for v in vals) / n
            std_v = math.sqrt(var_v)
        per_class[cls] = {
            "n": int(n),
            "score_mean": float(mean_v),
            "score_std": float(std_v),
        }

    # Selected class = class of candidate at argmin(score) (lower is better).
    sel_idx = int(scores_t.argmin().item())
    selected_class = int(classes_per_cand[sel_idx])

    # top2 gap on per-class means: 2nd-lowest minus lowest (positive).
    sorted_means = sorted(m["score_mean"] for m in per_class.values())
    if len(sorted_means) >= 2:
        top2_gap = float(sorted_means[1] - sorted_means[0])
    else:
        top2_gap = None

    return per_class, selected_class, top2_gap, True


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _run_seed(
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Run one seed: P0 warmup (no instrumentation) then P1 measurement.

    Returns per-seed aggregate metrics + per-tick log for the P1 window.
    """
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env)
    agent.eval()  # online updates only via record_transition + update_z_goal

    total_train_eps = p0_episodes + p1_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p1_logged = 0
    n_p1_pre_ge2 = 0
    n_p1_pre_eq1 = 0
    n_p1_collapse_above_eps = 0  # score_top2_gap > epsilon AND pre_ge2

    per_tick_log: List[Dict[str, Any]] = []
    top2_gaps_pre_ge2: List[float] = []
    pre_e3_class_count_log: List[int] = []
    selected_classes_p1: Dict[int, int] = {}  # class -> count
    committed_classes_p1: Dict[int, int] = {}

    # Run all eps in one loop; instrument only when ep >= p0_episodes.
    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs_harm(obs_dict),
                obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )

            # Record transition for online E2 update.
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach()
                )

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            # === Pre-E3 instrumentation (P1 only) ===
            pre_e3_classes: List[int] = []
            if is_p1 and candidates:
                pre_e3_classes = sorted({
                    _trajectory_first_action_class(t) for t in candidates
                })

            action = agent.select_action(candidates, ticks)
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at seed={seed} phase={phase_label} "
                        f"ep={ep} step={_step}"
                    )
                break

            # === Post-E3 instrumentation (P1 only) ===
            if is_p1:
                last_scores = getattr(agent.e3, "last_scores", None)
                per_class, sel_class, top2_gap, logged = _per_class_score_stats(
                    candidates, last_scores
                )
                committed_class = int(action[0].argmax().item())

                pre_count = len(pre_e3_classes)
                pre_e3_class_count_log.append(pre_count)
                if pre_count >= 2:
                    n_p1_pre_ge2 += 1
                elif pre_count == 1:
                    n_p1_pre_eq1 += 1

                if logged:
                    n_p1_logged += 1
                    if sel_class is not None:
                        selected_classes_p1[sel_class] = (
                            selected_classes_p1.get(sel_class, 0) + 1
                        )
                    if pre_count >= 2 and top2_gap is not None:
                        top2_gaps_pre_ge2.append(top2_gap)

                committed_classes_p1[committed_class] = (
                    committed_classes_p1.get(committed_class, 0) + 1
                )

                # Compact per-tick record (only every 25 steps to keep
                # manifest size bounded; the aggregates are computed from
                # the full P1 stream above, the log is for spot-check).
                if (n_p1_ticks % 25) == 0:
                    per_tick_log.append({
                        "p1_tick": n_p1_ticks,
                        "ep": ep,
                        "step": _step,
                        "pre_e3_class_count": pre_count,
                        "pre_e3_classes": pre_e3_classes,
                        "per_class": per_class,
                        "selected_class": sel_class,
                        "committed_class": committed_class,
                        "top2_class_gap": top2_gap,
                        "logged": logged,
                    })
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            # Step env.
            _, _harm_signal, done, info, obs_dict = env.step(action)

            # Online z_goal update (mirrors V3-EXQ-543k / 550).
            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()

            if done:
                break

        # Progress / boundary print every 10 ep (runner-instrumentation rule).
        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    # === Decision-rule aggregates (computed only over P1 measurement) ===
    if n_p1_ticks > 0:
        frac_pre_ge2 = float(n_p1_pre_ge2 / n_p1_ticks)
    else:
        frac_pre_ge2 = 0.0

    if top2_gaps_pre_ge2:
        score_range = max(top2_gaps_pre_ge2) - min(top2_gaps_pre_ge2)
        # Use observed range floor when range is degenerate.
        epsilon = max(score_range * SCORE_GAP_EPSILON_RANGE_FRAC, 1e-6)
        n_above_eps = sum(1 for g in top2_gaps_pre_ge2 if g > epsilon)
        n_below_eps = sum(1 for g in top2_gaps_pre_ge2 if g <= epsilon)
        mean_top2_gap = float(sum(top2_gaps_pre_ge2) / len(top2_gaps_pre_ge2))
        frac_e3_collapse = float(n_above_eps / len(top2_gaps_pre_ge2))
        frac_near_tie = float(n_below_eps / len(top2_gaps_pre_ge2))
    else:
        epsilon = None
        mean_top2_gap = None
        frac_e3_collapse = None
        frac_near_tie = None

    # Interpretation label (pre-registered grid; see module docstring).
    if frac_pre_ge2 < PRE_GE2_FRAC_GATE:
        interp = "low_precondition_re_evaluate_theory1"
    elif frac_e3_collapse is None:
        interp = "no_pre_ge2_measurements"
    elif frac_e3_collapse >= E3_COLLAPSE_FRAC_GATE:
        if frac_near_tie is not None and frac_near_tie >= NEAR_TIE_FRAC_GATE:
            interp = "R2a_e3_collapse_confirmed_near_tie_or_mixed"
        else:
            interp = "R2a_e3_collapse_confirmed_large_gap"
    elif frac_e3_collapse < E3_COLLAPSE_FRAC_FLOOR:
        interp = "R2b_e3_preserves_diversity"
    else:
        interp = "inconclusive_resample_heavier"

    return {
        "seed": int(seed),
        "p0_episodes_run": int(min(p0_episodes, total_train_eps)),
        "p1_episodes_run": int(max(0, total_train_eps - p0_episodes)),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p1_logged": int(n_p1_logged),
        "n_p1_pre_ge2": int(n_p1_pre_ge2),
        "n_p1_pre_eq1": int(n_p1_pre_eq1),
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "mean_top2_class_gap": (
            None if mean_top2_gap is None else round(mean_top2_gap, 6)
        ),
        "score_gap_epsilon": (
            None if epsilon is None else round(epsilon, 6)
        ),
        "frac_e3_collapse_above_eps": (
            None if frac_e3_collapse is None else round(frac_e3_collapse, 6)
        ),
        "frac_near_tie_below_eps": (
            None if frac_near_tie is None else round(frac_near_tie, 6)
        ),
        "selected_classes_p1_counts": {
            str(k): int(v) for k, v in sorted(selected_classes_p1.items())
        },
        "committed_classes_p1_counts": {
            str(k): int(v) for k, v in sorted(committed_classes_p1.items())
        },
        "interpretation_label": interp,
        "per_tick_log": per_tick_log,
        "error_note": error_note,
    }


def _interpret_run(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-seed aggregation + final interpretation."""
    labels = [r["interpretation_label"] for r in seed_rows]
    label_counts: Dict[str, int] = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1

    # Majority interpretation across seeds.
    majority = max(label_counts.items(), key=lambda kv: kv[1])
    return {
        "per_seed_labels": labels,
        "label_counts": label_counts,
        "majority_label": majority[0],
        "majority_n_of_total": [int(majority[1]), int(len(labels))],
        "unanimous": bool(len(label_counts) == 1),
    }


def _all_finite(*vals: Any) -> bool:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return False
    return True


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    print(
        f"Seed {seeds[0]} Condition main_path  "
        f"(P0={p0_episodes} ep, P1={p1_episodes} ep, "
        f"steps_per_episode={steps_per_episode}, dry_run={dry_run})",
        flush=True,
    )

    seed_rows: List[Dict[str, Any]] = []
    n_seeds_completed = 0
    for s in seeds:
        if s != seeds[0]:
            print(f"Seed {s} Condition main_path", flush=True)
        row = _run_seed(s, p0_episodes, p1_episodes, steps_per_episode)
        seed_rows.append(row)
        if row["error_note"] is None:
            n_seeds_completed += 1
        verdict = "PASS" if row["error_note"] is None else "FAIL"
        print(f"verdict: {verdict}", flush=True)

    cross = _interpret_run(seed_rows)

    passed = (
        n_seeds_completed >= MIN_SEEDS_FOR_PASS
        and cross["majority_label"] != "no_pre_ge2_measurements"
    )

    return {
        "outcome": "PASS" if passed else "FAIL",
        "seeds": seeds,
        "n_seeds_completed": int(n_seeds_completed),
        "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "pre_ge2_frac_gate": float(PRE_GE2_FRAC_GATE),
            "e3_collapse_frac_gate": float(E3_COLLAPSE_FRAC_GATE),
            "e3_collapse_frac_floor": float(E3_COLLAPSE_FRAC_FLOOR),
            "near_tie_frac_gate": float(NEAR_TIE_FRAC_GATE),
            "score_gap_epsilon_range_frac": float(SCORE_GAP_EPSILON_RANGE_FRAC),
        },
        "per_seed_results": seed_rows,
        "cross_seed_interpretation": cross,
    }


def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",  # diagnostic
        "evidence_direction_note": (
            "experiment_purpose=diagnostic; not weighted in confidence or "
            "conflict scoring. Interpretation drives next-experiment routing "
            "per behavioral_diversity_isolation_plan.md R2.a / R2.b decision "
            "rules. cross_seed_interpretation.majority_label = "
            f"{result.get('cross_seed_interpretation', {}).get('majority_label')}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "use_support_preserving_cem": True,
            "support_preserving_stratified_elites": True,
            "support_preserving_ao_std_floor": 0.2,
            "support_preserving_min_first_action_classes": 2,
            "use_noise_floor": True,
            "noise_floor_alpha": 0.1,
            "use_per_stream_vs": True,
            "use_per_region_vs": True,
            "use_event_segmenter": True,
            "use_invalidation_trigger": True,
            "use_anchor_sets": True,
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Short smoke-test run (1 seed, 2+2 ep, 30 steps).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override output dir (default: REE_assembly evidence/experiments).",
    )
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    # Output path.
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"

    if args.dry_run:
        # Don't write manifest to evidence dir on dry-run.
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"interp(majority)={result['cross_seed_interpretation']['majority_label']} "
        f"seeds_completed={result['n_seeds_completed']}/{len(seeds)}",
        flush=True,
    )

    if args.dry_run:
        # Scrub dry-run manifest before sentinel emission so we don't
        # pollute evidence dir.
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
