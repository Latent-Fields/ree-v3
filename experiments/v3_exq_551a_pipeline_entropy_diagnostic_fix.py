#!/opt/local/bin/python3
"""
V3-EXQ-551a -- Pipeline entropy diagnostic FIX (supersedes V3-EXQ-551).

Claims: none (diagnostic; experiment_purpose=diagnostic; no claim weighting)
Supersedes: V3-EXQ-551 (instrumentation bug: cascading if/elif interpretation
fired e2_rollout_collapse on the first sub-threshold stage and never inspected
the remaining stages independently; manifest stage 2/3 values were finite but
the interpretation label collapsed them to the default branch).

Purpose
-------
Same scientific question as V3-EXQ-551: WHERE in the action-selection
pipeline does the EXQ-550 monostrategy cliff originate? Same four-stage
decomposition (CEM proposer / E2 rollout / E3 evaluator / final action
class entropy). Same seeds / ticks / env wiring as 551.

What was wrong in V3-EXQ-551
----------------------------
1. Cascading if/elif interpretation: as soon as Stage 1 OR Stage 2 OR
   Stage 3 fell below the 1e-6 flag threshold, the label was assigned
   and the rest of the stages were never independently classified.
   Result: a run with stage 1 ~3e-4 (proposer near-collapse), stage 2
   ~9e-7 (just barely sub-threshold), and stage 3 ~9e-4 (well above
   threshold; E3 IS discriminating) was labelled "e2_rollout_collapse"
   and the proposer-collapse + finite-evaluator signal was lost.
2. Empty per-stage lists silently mean-aggregated to 0.0 (via the
   _mean() helper returning 0.0 on empty input), making "actually zero"
   indistinguishable from "didn't measure".
3. The per-tick measurement-eligibility booleans were implicit
   (n_ticks_with_new_candidates etc. as bulk counters) rather than
   per-tick logged flags, so a single-stage instrumentation gap could
   not be diagnosed from the manifest.

What this fix changes
---------------------
- Per-tick measurement booleans are emitted alongside per-stage values
  in per_tick_log: stage1_logged / stage2_logged / stage3_logged.
- Stage aggregates carry both _mean and _n_measured. Sentinel 0.0 mean
  with n_measured=0 is the "didn't measure" signal; sentinel 0.0 mean
  with n_measured>0 is the "actually zero" signal.
- Independent per-stage zero-variance flags (stage{1_pairwise,1_elem,2,3}
  _is_zero) are computed once per seed against the ZERO_VARIANCE_FLAG
  threshold and reported in the manifest as a bitmask the interpretation
  consumes.
- Interpretation grid replaced with a multi-cliff classifier that
  reports the conjunction of per-stage zero/non-zero status. The label
  is one of:
    proposer_only_collapse           (s1 zero; s2 and s3 finite)
    rollout_only_collapse            (s2 zero; s1 and s3 finite)
    evaluator_only_flatness          (s3 zero; s1 and s2 finite)
    proposer_and_rollout_collapse    (s1 and s2 zero; s3 finite)
    proposer_and_evaluator_collapse  (s1 and s3 zero; s2 finite)
    rollout_and_evaluator_collapse   (s2 and s3 zero; s1 finite)
    full_pipeline_collapse           (s1 and s2 and s3 zero)
    selection_or_policy_bug          (s1, s2, s3 all finite; s4 zero)
    all_stages_diverse               (none zero; s4 finite)
- Pre-registered interpretation grid pinned to the user-supplied
  5-row mapping (see evidence_direction_note + module docstring).

Pre-registered interpretation grid
----------------------------------
Stage 1 already known to be collapsed at ~2-3e-4 (EXQ-551 + ARC-062
finding). The new signal is whether stage 2 OR stage 3 ALSO show
zero-variance independently:

  Stage 2 zero AND Stage 3 zero -> downstream cliffs are independent
    of proposer; fixing proposer alone won't lift entropy.
    Next move = fix proposer + downstream simultaneously.

  Stage 2 finite AND Stage 3 zero -> rollout carries diversity,
    evaluator flattens it (evaluator-fix path, idea #5).
    Next move = V3-EXQ-552 forced-exploration warmup is correct
    framing; queue a focused evaluator-conditioning diagnostic.

  Stage 2 zero AND Stage 3 finite -> rollout collapses before
    evaluator (E2 dynamics issue).
    Next move = E2 conditioning diagnostic on z_world rollout.

  Stage 2 finite AND Stage 3 finite -> proposer collapse is the only
    cliff; structural CEM seeding alone (V3-EXQ-553 candidate)
    should lift entropy.

  All three zero -> something pathological in trajectory generation;
    deep root-cause investigation needed (gradient / init / env reset).

experiment_purpose=diagnostic. PASS is per-stage metrics present (or
explicit sentinel with n_measured=0) and no agent ERROR. Interpretation
drives the next experiment.

Implementation notes
--------------------
- Stage 1 pairwise + elementwise variance and Stage 2 final world_states
  variance are read from agent.generate_trajectories() output.
- Stage 3 reads agent.e3.last_scores immediately after select_action().
- Per-tick measurement booleans gate aggregation AND are logged so the
  manifest distinguishes instrumentation gaps from real zero variance.
- No monkey-patch. No ree_core modification.

See REE_assembly/evidence/planning/goal_pipeline_plan.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_551a_pipeline_entropy_diagnostic_fix"
QUEUE_ID = "V3-EXQ-551a"
SUPERSEDES_QUEUE_ID = "V3-EXQ-551"
SUPERSEDES_EXPERIMENT_TYPE = "v3_exq_551_pipeline_entropy_diagnostic"
CLAIM_IDS: List[str] = []  # purely diagnostic; no claim under test
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 17]
TICKS_PER_SEED = 400  # matches EXQ-551 per parent spec
EPISODE_STEPS = 200   # matches EXQ-550/551 per-episode budget

# Acceptance thresholds for "essentially zero" variance.
# These are FLAG thresholds (not pass-gate). PASS gates only that per-stage
# metrics are present + finite. The flags drive the multi-cliff
# interpretation downstream.
ZERO_VARIANCE_FLAG = 1e-6

# Sentinel for "not measured this tick / not measured this seed."
NOT_MEASURED_SENTINEL = 0.0


def _make_env(seed: int) -> CausalGridWorldV2:
    """Mirror V3-EXQ-550/551 env config exactly (SD-029 scheduled hazards ON)."""
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
        scheduled_external_hazard_enabled=True,
        scheduled_external_hazard_interval=50,
        scheduled_external_hazard_prob=0.5,
        scheduled_external_hazard_adjacent_only=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Mirror V3-EXQ-550/551 ARM_OFF agent wiring."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        z_goal_enabled=False,
        drive_weight=0.0,
        e1_goal_conditioned=False,
        goal_weight=0.0,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
    )
    return REEAgent(cfg)


def _shannon_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _stage1_action_variance(candidates) -> Tuple[float, float, int, bool]:
    """Stage 1: variance of FIRST-STEP actions across K candidates.

    Returns (mean_pairwise_l2, elementwise_variance, n_candidates, logged).
    logged=True means the metric was actually computed (K >= 2 candidates).
    logged=False means measurement was not eligible this tick.
    """
    if not candidates or len(candidates) < 2:
        return (NOT_MEASURED_SENTINEL, NOT_MEASURED_SENTINEL,
                len(candidates) if candidates else 0, False)
    firsts = []
    for t in candidates:
        a = t.actions[:, 0, :]  # [1, action_dim]
        firsts.append(a.squeeze(0).detach())
    stacked = torch.stack(firsts, dim=0)  # [K, action_dim]
    K = stacked.shape[0]
    elem_var = float(stacked.var(dim=0, unbiased=False).mean().item())
    diffs = stacked.unsqueeze(0) - stacked.unsqueeze(1)
    l2 = diffs.norm(dim=-1)
    iu = torch.triu_indices(K, K, offset=1)
    pairwise = float(l2[iu[0], iu[1]].mean().item())
    return (pairwise, elem_var, K, True)


def _stage2_rollout_variance(candidates) -> Tuple[float, int, bool]:
    """Stage 2: variance of FINAL z_world states across K candidates.

    Returns (mean_elementwise_variance, n_candidates_with_world_states,
    logged). logged=True means at least 2 candidates produced final
    world_states. logged=False means the rollouts produced fewer than 2
    valid final states (instrumentation gap distinct from "rollouts all
    converged to identical state" which would log with value ~0).
    """
    finals = []
    for t in candidates:
        if t.world_states is None or len(t.world_states) == 0:
            continue
        final = t.world_states[-1]  # [1, world_dim]
        finals.append(final.squeeze(0).detach())
    if len(finals) < 2:
        return (NOT_MEASURED_SENTINEL, len(finals), False)
    stacked = torch.stack(finals, dim=0)
    elem_var = float(stacked.var(dim=0, unbiased=False).mean().item())
    return (elem_var, stacked.shape[0], True)


def _stage3_score_variance(agent) -> Tuple[float, float, int, bool]:
    """Stage 3: variance of E3 per-candidate scores.

    Reads agent.e3.last_scores AFTER select_action(). The E3Selector
    caches last_scores on every select() at
    ree_core/predictors/e3_selector.py:689 (verified 2026-05-11).
    Returns (variance, score_range, K, logged). logged=False means the
    cache was None / empty / had fewer than 2 entries (instrumentation
    gap distinct from "all scores identical" which logs as ~0).
    """
    e3 = agent.e3
    last_scores = getattr(e3, "last_scores", None)
    if last_scores is None or last_scores.numel() < 2:
        K = 0 if last_scores is None else int(last_scores.numel())
        return (NOT_MEASURED_SENTINEL, NOT_MEASURED_SENTINEL, K, False)
    s = last_scores.detach()
    var = float(s.var(unbiased=False).item())
    rng = float((s.max() - s.min()).item())
    return (var, rng, int(s.numel()), True)


def _run_seed(seed: int) -> Dict:
    """Run one seed. Per-tick measurement booleans are recorded so the
    manifest distinguishes "didn't measure" from "measured zero".
    """
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env)

    stage1_pairwise: List[float] = []
    stage1_elem_var: List[float] = []
    stage2_world_var: List[float] = []
    stage3_score_var: List[float] = []
    stage3_score_range: List[float] = []

    # Per-tick logged booleans (one row per tick we attempted measurement).
    per_tick_log: List[Dict] = []

    action_counts: Dict[int, int] = {}
    n_ticks = 0
    n_ticks_with_new_candidates = 0
    n_ticks_with_world_states = 0
    n_ticks_with_scores = 0
    n_nans = 0
    error_note: Optional[str] = None

    n_episodes = max(1, (TICKS_PER_SEED + EPISODE_STEPS - 1) // EPISODE_STEPS)
    prev_candidates_id: Optional[int] = None

    for ep in range(n_episodes):
        if n_ticks >= TICKS_PER_SEED:
            break
        _, obs_dict = env.reset()
        agent.reset()
        prev_candidates_id = None
        for _step in range(EPISODE_STEPS):
            if n_ticks >= TICKS_PER_SEED:
                break
            body = obs_dict["body_state"].float().unsqueeze(0)
            world = obs_dict["world_state"].float().unsqueeze(0)
            harm = obs_dict.get("harm_obs")
            if harm is not None:
                harm = harm.float().unsqueeze(0)
            harm_a = obs_dict.get("harm_obs_a")
            if harm_a is not None:
                harm_a = harm_a.float().unsqueeze(0)
            harm_hist = obs_dict.get("harm_history")
            if harm_hist is not None:
                harm_hist = harm_hist.float().unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=harm, obs_harm_a=harm_a, obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            cur_id = id(candidates)
            fresh_candidates = (cur_id != prev_candidates_id)
            prev_candidates_id = cur_id

            tick_record: Dict = {
                "tick": n_ticks,
                "ep": ep,
                "fresh_candidates": fresh_candidates,
                "stage1_logged": False,
                "stage2_logged": False,
                "stage3_logged": False,
            }

            if fresh_candidates and candidates:
                n_ticks_with_new_candidates += 1
                pw, ev, K1, s1_logged = _stage1_action_variance(candidates)
                if s1_logged:
                    stage1_pairwise.append(pw)
                    stage1_elem_var.append(ev)
                    tick_record["stage1_logged"] = True
                    tick_record["stage1_K"] = K1

                wv, K2, s2_logged = _stage2_rollout_variance(candidates)
                if s2_logged:
                    n_ticks_with_world_states += 1
                    stage2_world_var.append(wv)
                    tick_record["stage2_logged"] = True
                    tick_record["stage2_K"] = K2

            action = agent.select_action(candidates, ticks)

            if not torch.isfinite(action).all():
                n_nans += 1
                if error_note is None:
                    error_note = (
                        f"non-finite action at seed={seed} ep={ep} "
                        f"step={_step}"
                    )
                per_tick_log.append(tick_record)
                break

            if fresh_candidates:
                sv, sr, K3, s3_logged = _stage3_score_variance(agent)
                if s3_logged:
                    n_ticks_with_scores += 1
                    stage3_score_var.append(sv)
                    stage3_score_range.append(sr)
                    tick_record["stage3_logged"] = True
                    tick_record["stage3_K"] = K3

            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            _, _harm_signal, done, info, obs_dict = env.step(action)
            n_ticks += 1
            per_tick_log.append(tick_record)

            if done:
                break

        if error_note is not None:
            break

    def _mean_or_sentinel(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else NOT_MEASURED_SENTINEL

    def _median_or_sentinel(xs: List[float]) -> float:
        if not xs:
            return NOT_MEASURED_SENTINEL
        ss = sorted(xs)
        m = len(ss)
        return float(ss[m // 2]) if m % 2 == 1 else float(0.5 * (ss[m // 2 - 1] + ss[m // 2]))

    return {
        "seed": seed,
        "n_ticks": n_ticks,
        "n_ticks_with_new_candidates": n_ticks_with_new_candidates,
        "n_ticks_with_world_states": n_ticks_with_world_states,
        "n_ticks_with_scores": n_ticks_with_scores,
        "n_nans": n_nans,
        "error_note": error_note,
        # Stage 1
        "stage1_pairwise_l2_mean": _mean_or_sentinel(stage1_pairwise),
        "stage1_pairwise_l2_median": _median_or_sentinel(stage1_pairwise),
        "stage1_pairwise_l2_n_measured": len(stage1_pairwise),
        "stage1_elem_var_mean": _mean_or_sentinel(stage1_elem_var),
        "stage1_elem_var_n_measured": len(stage1_elem_var),
        # Stage 2
        "stage2_world_var_mean": _mean_or_sentinel(stage2_world_var),
        "stage2_world_var_median": _median_or_sentinel(stage2_world_var),
        "stage2_world_var_n_measured": len(stage2_world_var),
        # Stage 3
        "stage3_score_var_mean": _mean_or_sentinel(stage3_score_var),
        "stage3_score_var_median": _median_or_sentinel(stage3_score_var),
        "stage3_score_range_mean": _mean_or_sentinel(stage3_score_range),
        "stage3_score_var_n_measured": len(stage3_score_var),
        # Stage 4
        "stage4_action_class_entropy": _shannon_entropy(action_counts),
        "stage4_action_class_counts": action_counts,
        "stage4_n_actions": sum(action_counts.values()),
    }


def _classify_seed(r: Dict) -> Dict:
    """Multi-cliff classifier. Independent per-stage zero-flag inspection
    rather than cascading if/elif. Returns dict of independent flags +
    canonical label.
    """
    z = ZERO_VARIANCE_FLAG
    # A stage is "zero" only if it was measured AND below threshold.
    # A stage with n_measured=0 is "not measured" -- distinct from "zero".
    s1_pw_measured = r["stage1_pairwise_l2_n_measured"] > 0
    s1_pw_zero = s1_pw_measured and r["stage1_pairwise_l2_mean"] < z
    s2_measured = r["stage2_world_var_n_measured"] > 0
    s2_zero = s2_measured and r["stage2_world_var_mean"] < z
    s3_measured = r["stage3_score_var_n_measured"] > 0
    s3_zero = s3_measured and r["stage3_score_var_mean"] < z
    s4_zero = r["stage4_action_class_entropy"] < z

    # Compose label. Reports independent collapse status per stage. Uses
    # the user-supplied 5-row interpretation grid given that stage 1 is
    # expected to collapse at ~2-3e-4 per ARC-062 / EXQ-551 finding.
    if not (s1_pw_measured and s2_measured and s3_measured):
        # Instrumentation gap. Should never happen at 400 ticks but the
        # sentinel makes it observable rather than silent.
        label = "instrumentation_gap"
    elif s1_pw_zero and s2_zero and s3_zero:
        label = "full_pipeline_collapse"
    elif s1_pw_zero and s2_zero and not s3_zero:
        label = "proposer_and_rollout_collapse"
    elif s1_pw_zero and not s2_zero and s3_zero:
        label = "proposer_and_evaluator_collapse"
    elif not s1_pw_zero and s2_zero and s3_zero:
        label = "rollout_and_evaluator_collapse"
    elif s1_pw_zero and not s2_zero and not s3_zero:
        label = "proposer_only_collapse"
    elif not s1_pw_zero and s2_zero and not s3_zero:
        label = "rollout_only_collapse"
    elif not s1_pw_zero and not s2_zero and s3_zero:
        label = "evaluator_only_flatness"
    elif (not s1_pw_zero) and (not s2_zero) and (not s3_zero) and s4_zero:
        label = "selection_or_policy_bug"
    else:
        label = "all_stages_diverse"

    return {
        "stage1_measured": s1_pw_measured,
        "stage1_is_zero": s1_pw_zero,
        "stage2_measured": s2_measured,
        "stage2_is_zero": s2_zero,
        "stage3_measured": s3_measured,
        "stage3_is_zero": s3_zero,
        "stage4_is_zero": s4_zero,
        "label": label,
    }


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- pipeline entropy localization (fix)",
          flush=True)
    print(f"Supersedes: {SUPERSEDES_QUEUE_ID}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Ticks per seed: {TICKS_PER_SEED} "
          f"(episode_steps={EPISODE_STEPS})", flush=True)
    print(f"Env: CausalGridWorldV2 + SD-029 scheduled hazards ON",
          flush=True)
    print("Single arm. No training. V_s circuit ON. No goal arm.",
          flush=True)
    print("Metrics: stage1 (CEM action var), stage2 (E2 rollout var), "
          "stage3 (E3 score var), stage4 (action class entropy)",
          flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)
    print(f"Zero-variance flag threshold: {ZERO_VARIANCE_FLAG}",
          flush=True)
    print("Fix: per-tick measurement booleans + sentinels + multi-cliff "
          "classifier (no cascading if/elif).", flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} pipeline entropy diagnostic fix"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit 0; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="1 seed x ~100 ticks smoke test (no manifest written).",
    )
    parser.add_argument(
        "--smoke-ticks", type=int, default=100,
        help="Smoke tick count (default 100 -- exercises the 400-tick "
             "code paths with enough new-candidate ticks to populate all "
             "stage aggregates).",
    )
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return (None, None)

    if args.smoke:
        global SEEDS, TICKS_PER_SEED, EPISODE_STEPS
        SEEDS = [42]
        TICKS_PER_SEED = max(40, int(args.smoke_ticks))
        EPISODE_STEPS = min(EPISODE_STEPS, TICKS_PER_SEED)
        print(f"SMOKE MODE: 1 seed x {TICKS_PER_SEED} ticks; no manifest "
              f"write", flush=True)
        print(f"Seed {SEEDS[0]} Condition single_arm", flush=True)
        r = _run_seed(SEEDS[0])
        cls = _classify_seed(r)
        print(
            f"  [train] label seed={SEEDS[0]} ep 1/1 "
            f"n_ticks={r['n_ticks']} "
            f"s1_pw={r['stage1_pairwise_l2_mean']:.4e} "
            f"(n={r['stage1_pairwise_l2_n_measured']}) "
            f"s2_wv={r['stage2_world_var_mean']:.4e} "
            f"(n={r['stage2_world_var_n_measured']}) "
            f"s3_sv={r['stage3_score_var_mean']:.4e} "
            f"(n={r['stage3_score_var_n_measured']}) "
            f"s4_entropy={r['stage4_action_class_entropy']:.4f}",
            flush=True,
        )
        print(f"  per-stage zero flags: s1={cls['stage1_is_zero']} "
              f"s2={cls['stage2_is_zero']} s3={cls['stage3_is_zero']} "
              f"s4={cls['stage4_is_zero']}", flush=True)
        print(f"  interpretation label: {cls['label']}", flush=True)
        # Smoke must verify all four stages actually populated.
        smoke_ok = (
            r["stage1_pairwise_l2_n_measured"] > 0
            and r["stage2_world_var_n_measured"] > 0
            and r["stage3_score_var_n_measured"] > 0
        )
        verdict = "FAIL" if (r["error_note"] is not None or not smoke_ok) else "PASS"
        print(f"verdict: {verdict}", flush=True)
        if not smoke_ok:
            print(f"  SMOKE INSTRUMENTATION GAP: "
                  f"s1_n={r['stage1_pairwise_l2_n_measured']} "
                  f"s2_n={r['stage2_world_var_n_measured']} "
                  f"s3_n={r['stage3_score_var_n_measured']}", flush=True)
        print("SMOKE OK" if verdict == "PASS" else "SMOKE FAIL",
              flush=True)
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    any_error = False
    for seed in SEEDS:
        print(f"Seed {seed} Condition single_arm", flush=True)
        r = _run_seed(seed)
        cls = _classify_seed(r)
        print(
            f"  [train] label seed={seed} ep 1/1 "
            f"n_ticks={r['n_ticks']} "
            f"new_cand_ticks={r['n_ticks_with_new_candidates']} "
            f"s1_pw_mean={r['stage1_pairwise_l2_mean']:.4e} "
            f"(n={r['stage1_pairwise_l2_n_measured']}) "
            f"s1_elem_var={r['stage1_elem_var_mean']:.4e} "
            f"s2_world_var={r['stage2_world_var_mean']:.4e} "
            f"(n={r['stage2_world_var_n_measured']}) "
            f"s3_score_var={r['stage3_score_var_mean']:.4e} "
            f"(n={r['stage3_score_var_n_measured']}) "
            f"s3_score_range={r['stage3_score_range_mean']:.4e} "
            f"s4_entropy={r['stage4_action_class_entropy']:.4f}",
            flush=True,
        )
        print(f"  per-stage zero flags: s1={cls['stage1_is_zero']} "
              f"s2={cls['stage2_is_zero']} s3={cls['stage3_is_zero']} "
              f"s4={cls['stage4_is_zero']}", flush=True)
        print(f"  interpretation label: {cls['label']}", flush=True)
        if r["error_note"] is not None:
            any_error = True
            print(f"  ERROR: {r['error_note']}", flush=True)
            print("verdict: FAIL", flush=True)
        else:
            print("verdict: PASS", flush=True)
        r["interpretation"] = cls
        all_results.append(r)

    def _finite(x: float) -> bool:
        return math.isfinite(x)

    all_finite = all(
        _finite(r["stage1_pairwise_l2_mean"])
        and _finite(r["stage2_world_var_mean"])
        and _finite(r["stage3_score_var_mean"])
        and _finite(r["stage4_action_class_entropy"])
        for r in all_results
    )

    # Diagnostic gate: every stage must have been MEASURED at least once
    # per seed (instrumentation-gap detection). If any seed produced
    # n_measured=0 for any stage, the diagnostic itself is broken; emit
    # FAIL even though metrics are finite-sentinel.
    all_measured = all(
        r["stage1_pairwise_l2_n_measured"] > 0
        and r["stage2_world_var_n_measured"] > 0
        and r["stage3_score_var_n_measured"] > 0
        for r in all_results
    )
    outcome = "FAIL" if (any_error or not all_finite or not all_measured) else "PASS"

    label_counts: Dict[str, int] = {}
    for r in all_results:
        label_counts[r["interpretation"]["label"]] = label_counts.get(
            r["interpretation"]["label"], 0) + 1

    summary = {
        "gate_rule": (
            "Diagnostic PASS = all per-stage metrics finite AND every "
            "stage measured at least once per seed AND no agent ERROR. "
            "Interpretation drives the next experiment, not the verdict."
        ),
        "label_distribution": label_counts,
        "n_seeds": len(SEEDS),
        "zero_variance_flag_threshold": ZERO_VARIANCE_FLAG,
        "any_error_seed": any_error,
        "all_metrics_finite": all_finite,
        "all_stages_measured_per_seed": all_measured,
    }

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Label distribution across seeds: {label_counts}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "unknown",
        "evidence_direction_note": (
            "Pipeline entropy localization fix (supersedes V3-EXQ-551). "
            "V3-EXQ-551 stage-2/3 values were finite in the manifest "
            "but the cascading if/elif interpretation grid fired on "
            "the first sub-threshold stage and never independently "
            "classified the remaining stages. V3-EXQ-551 stage-2/3 "
            "None readings reported in the supersession task were an "
            "instrumentation gap, NOT substrate signal; this fix "
            "distinguishes 'didn't measure' (n_measured=0 + sentinel) "
            "from 'measured zero' (n_measured>0 + value<flag). "
            "Pre-registered multi-cliff interpretation grid: "
            "(a) Stage 2 zero AND Stage 3 zero -> downstream cliffs "
            "independent of proposer; fixing proposer alone won't lift "
            "entropy. "
            "(b) Stage 2 finite AND Stage 3 zero -> evaluator-fix path "
            "(idea #5); V3-EXQ-552 forced-exploration warmup framing "
            "correct. "
            "(c) Stage 2 zero AND Stage 3 finite -> E2 dynamics issue; "
            "E2 conditioning diagnostic. "
            "(d) Stage 2 finite AND Stage 3 finite -> proposer collapse "
            "is the only cliff; structural CEM seeding (V3-EXQ-553) "
            "alone should lift entropy. "
            "(e) All three zero -> pathological trajectory generation; "
            "deep root-cause investigation needed. "
            "experiment_purpose=diagnostic: PASS gates only that all "
            "stages were measured and finite; the multi-cliff label "
            "drives the next experiment, not a claim-direction call."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "supersedes": SUPERSEDES_QUEUE_ID,
        "supersedes_experiment_type": SUPERSEDES_EXPERIMENT_TYPE,
        "config": {
            "seeds": SEEDS,
            "ticks_per_seed": TICKS_PER_SEED,
            "episode_steps": EPISODE_STEPS,
            "zero_variance_flag_threshold": ZERO_VARIANCE_FLAG,
            "single_arm": True,
            "z_goal_enabled": False,
            "v_s_circuit_on": True,
            "sd029_scheduled_hazards": True,
            "no_training": True,
            "supersedes": SUPERSEDES_QUEUE_ID,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )
    print(f"Output written to: {out_file}", flush=True)

    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
