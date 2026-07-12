#!/opt/local/bin/python3
"""
V3-EXQ-490k: MECH-295 modulatory-SUFFICIENCY probe (goal_pipeline:GAP-4
successor to the TERMINAL V3-EXQ-490j severed-bridge baseline).

PURPOSE (diagnostic)
====================
V3-EXQ-490j (2026-05-31) established, with a clean severed-bridge sentinel,
that MECH-295 is NOT behaviourally NECESSARY for approach: ARM_0 (bridge
severed) + drive amplification still produced approach_commit_rate=1.0 in
3/3 seeds via architecturally first-class PARALLEL drive->approach pathways
(MECH-216 schema wanting, MECH-290 backward credit, MECH-307 anticipatory
liking, tonic_5ht benefit_salience). The substrate-firing layer (490j
C6/C7/C9 PASS) supports the narrowed MODULATORY reading of MECH-295, but
490j could NOT show the modulatory contribution is behaviourally
CONSEQUENTIAL -- approach_commit_rate ceiling-saturates at 1.0 in BOTH arms
because the parallel pathways dominate, so it cannot discriminate whether
MECH-295 ever changes a decision.

490k asks the modulatory-sufficiency question with a metric that CANNOT
ceiling-saturate: the per-candidate MECH-295 approach-cue score_bias is a
NEGATIVE [K] vector added to E3's candidate scores before the committed
argmin. The probe measures, per tick where the bridge fires, whether
SUBTRACTING the MECH-295 contribution from the final E3 scores CHANGES the
argmin (i.e. whether the modulation ever flips WHICH action wins). This is
the autopsy's routing_secondary option (c) -- the cleanest mechanistic
isolation of the modulatory contribution.

  MUST NOT use aggregate approach_commit_rate as the discriminating metric:
  it saturates at 1.0 in both arms via the parallel pathways (this is exactly
  the 490j contamination). The argmin-change probe replaces it.

DESIGN -- 3 arms x 3 seeds, on the GAP-4 operating substrate
============================================================
gap4 substrate (from goal_pipeline_tier1.build_config gap4_operating=True):
  drive_floor=0.9, drive_ema_alpha=1.0, REEConfig.goal_stream bundle
  (MECH-307 + MECH-295 + schema wanting), post-540e relaxed MECH-295
  activation floors, use_dacc=True.

  ARM_1_full_gap4   -- full GAP-4 operating config (MECH-295 cue ON, all
      parallel writers ON). This is the PRIMARY arm for option (c): the
      within-arm argmin-change probe. Also the reference for option (a).
  ARM_0_cue_severed -- gap4 substrate with mech295_liking_to_approach_cue_gain
      = 0.0 (the MECH-295 approach-cue score_bias is exactly zero; write
      side still fires). The no-MECH-295-behavioural-contribution
      counterfactual for the option (a) action-class divergence comparison.
  ARM_2_mech295_only -- gap4 substrate with all OTHER VALENCE_WANTING /
      approach writers OFF (schema_wanting / tonic_5ht / backward_credit /
      MECH-307 consumer-conjunction-read), MECH-295 cue ON. SENSITIVITY
      ablation (option b): approach reflects only the MECH-295 contribution.
      NOTE: this is a sensitivity analysis, NOT a necessity test. Even if
      approach collapses here it only shows MECH-295 is necessary GIVEN the
      others are off, not necessary in the full multi-pathway substrate.

PRIMARY METRIC (option c, ARM_1)
================================
Per tick where the MECH-295 cue bias fires non-trivially:
  scores_with    = e3.last_scores                (final scores, MECH-295 in)
  mech295_bias   = the raw [K] cue (+conjunction) bias for that tick
  scores_without = scores_with - mech295_bias    (counterfactual)
  argmin_flip    = argmin(scores_with) != argmin(scores_without)
  mech295_argmin_flip_fraction = (# flip ticks) / (# mech295-fired ticks)
The committed-only variant restricts to ticks where beta is elevated.

SECONDARY METRIC (option a, ARM_1 vs ARM_0)
===========================================
Per seed: total-variation distance between the ARM_1 and ARM_0 first-step
action-class distributions + whether the first-committed-tick action class
differs. Behavioural sanity check that the modulation changes behaviour.

SENSITIVITY METRIC (option b, ARM_2)
====================================
ARM_2 mech295_argmin_flip_fraction + approach_commit_rate with only the
MECH-295 contribution active.

PRE-REGISTERED OUTCOME INTERPRETATION GRID
==========================================
(per-claim evidence_direction for MECH-295, and the governance action it
implies; experiment_purpose=diagnostic so this does NOT auto-weight
governance -- the GAP-4 disposition is an interactive governance decision.)

ROW 1 -- ARM_1 mech295_argmin_flip_fraction > 0 on >= 2/3 seeds:
  -> The MECH-295 modulation DOES flip the committed argmin at some ticks:
     the bias is behaviourally CONSEQUENTIAL, not merely substrate-firing.
  -> evidence_direction_per_claim[MECH-295] = supports (modulatory reading).
  -> Governance: register the narrowed MODULATORY MECH-295 claim with this
     behavioural support; close goal_pipeline:GAP-4.

ROW 2 -- ARM_1 mech295 fires (>=1 tick on >=2/3 seeds) but
         argmin_flip_fraction == 0 on EVERY seed:
  -> The bridge fires but never changes which action wins -- the modulation
     is always sub-threshold relative to the parallel pathways + primary
     harm/goal scores. This WEAKENS even the modulatory reading.
  -> evidence_direction_per_claim[MECH-295] = weakens.
  -> Governance: MECH-295 modulatory contribution is not behaviourally
     consequential under this substrate; consider modulatory-bias-selection-
     authority (gap-relative scaling) before any further MECH-295 behavioural
     claim, OR retain MECH-295 as substrate-validated-only (493 + 490j) with
     the behavioural-sufficiency falsifier left open.

ROW 3 -- ARM_1 mech295 NEVER fires across all 3 seeds (mech295_fired_ticks
         == 0 everywhere):
  -> Instrumentation / substrate-activation failure (the cue side never
     reached its drive / z_goal floors), NOT a claim result. The probe could
     not run.
  -> evidence_direction_per_claim[MECH-295] = unknown (test could not run).
  -> Routing: /diagnose-errors (drive + z_goal_norm histograms vs the
     post-540e floors), mirroring the 490j ROW 8 path. outcome = FAIL.

ROW 4 -- ARM_2 (mech295-only) argmin_flip_fraction or approach_commit_rate
         markedly EXCEEDS ARM_1 while ARM_1 flip_fraction is ~0:
  -> Corroborates that in the full substrate the parallel pathways drown the
     MECH-295 contribution (sensitivity confirms a real-but-subdominant
     modulation). Reading is the ROW-2 weakens at the behavioural layer with
     a documented "consequential only when others are silenced" note.
  -> Feeds the governance note, does not by itself flip the per-claim
     direction.

claim_ids: [MECH-295]
experiment_purpose: diagnostic
supersedes: (none -- V3-EXQ-490j is TERMINAL/reviewed; 490k is the
            modulatory-sufficiency successor, not a correction of 490j)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np
import torch

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness, StepHooks  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    ArmSpec,
    ENV_FISHTANK_KWARGS,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    WARMUP_EPISODES_DEFAULT,
    build_config,
    make_env,
    warmup_train,
    _approach_commit,
    _entropy,
)
from ree_core.agent import REEAgent  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_490k_mech295_modulatory_sufficiency"
QUEUE_ID = "V3-EXQ-490k"
CLAIM_IDS = ["MECH-295"]
EXPERIMENT_PURPOSE = "diagnostic"

# Fixed total-step eval budget per (seed, arm). Matches 490j so cross-run
# magnitudes are calibrated.
TOTAL_EVAL_STEP_BUDGET = 900

SEEDS_PASS_MIN = 2

# Sentinels consumed by build_config_490k (stripped before the shared
# build_config sees them so its hasattr+setattr loop is not polluted).
SENTINEL_CUE_SEVER = "__490k_sever_mech295_cue__"
SENTINEL_MECH295_ONLY = "__490k_mech295_only__"

ARMS: List[ArmSpec] = [
    ArmSpec("ARM_1_full_gap4", gap4_operating=True),
    ArmSpec(
        "ARM_0_cue_severed",
        gap4_operating=True,
        extra_config={SENTINEL_CUE_SEVER: True},
    ),
    ArmSpec(
        "ARM_2_mech295_only",
        gap4_operating=True,
        extra_config={SENTINEL_MECH295_ONLY: True},
    ),
]
FULL_ARM = "ARM_1_full_gap4"
SEVERED_ARM = "ARM_0_cue_severed"
ONLY_ARM = "ARM_2_mech295_only"


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_config_490k(env, arm: ArmSpec):
    """Wrap the shared build_config to apply 490k arm-specific overrides.

    Two sentinels are consumed here and stripped before the shared
    build_config sees them:
      SENTINEL_CUE_SEVER     -> mech295_liking_to_approach_cue_gain = 0.0
                                (ARM_0: MECH-295 approach-cue bias is exactly
                                 zero; write side still fires).
      SENTINEL_MECH295_ONLY  -> turn OFF the other VALENCE_WANTING / approach
                                writers so approach reflects only MECH-295
                                (ARM_2 sensitivity ablation). These live at
                                nested config levels the shared build_config
                                top-level setattr loop cannot reach.
    """
    sever_cue = bool(arm.extra_config.pop(SENTINEL_CUE_SEVER, False))
    mech295_only = bool(arm.extra_config.pop(SENTINEL_MECH295_ONLY, False))
    cfg = build_config(env, arm)

    if sever_cue:
        # ARM_0: bridge cue contributes exactly zero score_bias (weak-reading
        # severed-cue arm). Pure MECH-295 behavioural contribution removed.
        cfg.mech295_liking_to_approach_cue_gain = 0.0

    if mech295_only:
        # ARM_2: silence the parallel VALENCE_WANTING / approach writers so the
        # MECH-295 cue is the only approach contributor (sensitivity arm).
        # MECH-216 schema wanting (E1Config).
        if hasattr(cfg, "e1") and hasattr(cfg.e1, "schema_wanting_enabled"):
            cfg.e1.schema_wanting_enabled = False
        # tonic_5ht benefit_salience (SerotoninConfig).
        if hasattr(cfg, "serotonin") and hasattr(cfg.serotonin, "tonic_5ht_enabled"):
            cfg.serotonin.tonic_5ht_enabled = False
        # MECH-290 backward credit sweep (HippocampalConfig).
        if hasattr(cfg, "hippocampal") and hasattr(
            cfg.hippocampal, "use_backward_credit_sweep"
        ):
            cfg.hippocampal.use_backward_credit_sweep = False
        # MECH-307 consumer-side conjunction read (adds a separate approach bias
        # on top of the pure MECH-295 cue). Off so only the pure cue remains.
        if hasattr(cfg, "use_mech307_consumer_conjunction_read"):
            cfg.use_mech307_consumer_conjunction_read = False

    return cfg


def _install_mech295_bias_recorder(agent: REEAgent) -> Dict[str, Any]:
    """Wrap the live MECH-295 bridge's cue + conjunction compute methods to
    record the raw per-candidate [K] bias each tick, WITHOUT touching the
    substrate (same no-substrate-edit pattern as 490j).

    The MECH-295 contribution composed into E3 score_bias is
    compute_approach_cue_score_bias(...) [+ compute_conjunction_score_bias(...)
    when the MECH-307 consumer-read is active]. We sum both to get the full
    per-tick MECH-295 contribution vector. Returns a mutable scratch dict the
    caller reads (and resets) once per tick.
    """
    scratch: Dict[str, Any] = {"cue": None, "conj": None}
    bridge = getattr(agent, "mech295_bridge", None)
    if bridge is None:
        return scratch

    orig_cue = bridge.compute_approach_cue_score_bias

    def wrapped_cue(*a, **k):
        out = orig_cue(*a, **k)
        try:
            scratch["cue"] = out.detach().clone()
        except Exception:
            scratch["cue"] = None
        return out

    bridge.compute_approach_cue_score_bias = wrapped_cue

    orig_conj = getattr(bridge, "compute_conjunction_score_bias", None)
    if orig_conj is not None:
        def wrapped_conj(*a, **k):
            out = orig_conj(*a, **k)
            try:
                scratch["conj"] = out.detach().clone()
            except Exception:
                scratch["conj"] = None
            return out

        bridge.compute_conjunction_score_bias = wrapped_conj

    return scratch


def _tick_mech295_vector(scratch: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Combine the cue + conjunction recordings into one [K] vector (or None)."""
    cue = scratch.get("cue")
    conj = scratch.get("conj")
    if cue is None and conj is None:
        return None
    if cue is None:
        return conj
    if conj is None:
        return cue
    try:
        return cue + conj.to(dtype=cue.dtype, device=cue.device)
    except Exception:
        return cue


def eval_490k(
    agent: REEAgent,
    env,
    *,
    total_step_budget: int,
    steps_per_episode: int,
    seed: int,
    arm_label: str,
) -> Dict[str, Any]:
    """Fixed-budget eval loop with the MECH-295 argmin-change probe."""
    scratch = _install_mech295_bias_recorder(agent)
    # Richer manifest: enable the built-in score-bias decomposition so we also
    # record mech295_liking_bias_range_mean as a corroborating substrate-side
    # signal. The probe itself does NOT depend on it.
    try:
        agent.e3.e3_score_decomp_enabled = True
    except Exception:
        pass

    metrics: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "total_eval_steps": 0,
        "approach_commit_steps": 0,
        "goal_active_steps": 0,
        # option (c) probe counters
        "mech295_fired_ticks": 0,
        "argmin_flip_ticks": 0,
        "mech295_fired_commit_ticks": 0,
        "argmin_flip_commit_ticks": 0,
        "mech295_bias_range_sum": 0.0,
        # option (a) action distribution + first-commit action
        "action_counts": {},
        "first_commit_action": -1,
        "bridge_cue_fires": 0,
        "bridge_write_fires": 0,
        "n_episodes_started": 0,
        "n_episodes_ended_done": 0,
    }

    bridge = getattr(agent, "mech295_bridge", None)

    def on_post_step(*, agent, latent, action, obs_dict, ticks, step, **kwargs) -> None:
        metrics["total_eval_steps"] += 1
        committed = bool(getattr(agent.beta_gate, "is_elevated", False))
        if agent.goal_state is not None and agent.goal_state.is_active():
            metrics["goal_active_steps"] += 1
        if _approach_commit(agent):
            metrics["approach_commit_steps"] += 1
            if metrics["first_commit_action"] < 0:
                a_idx = int(action.argmax(dim=-1).flatten()[0].item())
                metrics["first_commit_action"] = a_idx
        if bridge is not None:
            metrics["bridge_cue_fires"] = int(getattr(bridge, "_n_cue_fires", 0))
            metrics["bridge_write_fires"] = int(getattr(bridge, "_n_write_fires", 0))

        # --- option (c) argmin-change probe -------------------------------
        scores = getattr(agent.e3, "last_scores", None)
        m_vec = _tick_mech295_vector(scratch)
        if scores is not None and m_vec is not None:
            try:
                s = scores.detach().reshape(-1)
                m = m_vec.reshape(-1)
                if s.shape[0] == m.shape[0] and s.shape[0] >= 2:
                    nonzero = bool(torch.any(m != 0).item())
                    if nonzero:
                        metrics["mech295_fired_ticks"] += 1
                        metrics["mech295_bias_range_sum"] += float(
                            (m.max() - m.min()).item()
                        )
                        argmin_with = int(s.argmin().item())
                        argmin_without = int((s - m).argmin().item())
                        flipped = argmin_with != argmin_without
                        if flipped:
                            metrics["argmin_flip_ticks"] += 1
                        if committed:
                            metrics["mech295_fired_commit_ticks"] += 1
                            if flipped:
                                metrics["argmin_flip_commit_ticks"] += 1
            except Exception:
                pass

        # reset per-tick scratch
        scratch["cue"] = None
        scratch["conj"] = None

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    if bridge is not None:
        bridge._n_cue_fires = 0
        bridge._n_write_fires = 0

    _, obs_dict = env.reset()
    agent.reset()
    harness.reset()
    metrics["n_episodes_started"] += 1
    steps_in_episode = 0

    while metrics["total_eval_steps"] < total_step_budget:
        result = harness.step(obs_dict)
        steps_in_episode += 1
        aidx = int(result.action.argmax(dim=-1).item())
        ac = metrics["action_counts"]
        ac[aidx] = ac.get(aidx, 0) + 1
        obs_dict = result.next_obs_dict
        if result.done or steps_in_episode >= steps_per_episode:
            if result.done:
                metrics["n_episodes_ended_done"] += 1
            if metrics["total_eval_steps"] >= total_step_budget:
                break
            _, obs_dict = env.reset()
            agent.reset()
            harness.reset()
            metrics["n_episodes_started"] += 1
            steps_in_episode = 0

    total = max(1, int(metrics["total_eval_steps"]))
    fired = int(metrics["mech295_fired_ticks"])
    fired_c = int(metrics["mech295_fired_commit_ticks"])
    metrics["approach_commit_rate"] = float(metrics["approach_commit_steps"]) / total
    metrics["goal_active_fraction"] = float(metrics["goal_active_steps"]) / total
    metrics["mech295_argmin_flip_fraction"] = (
        float(metrics["argmin_flip_ticks"]) / fired if fired > 0 else 0.0
    )
    metrics["mech295_argmin_flip_fraction_commit"] = (
        float(metrics["argmin_flip_commit_ticks"]) / fired_c if fired_c > 0 else 0.0
    )
    metrics["mech295_bias_range_mean"] = (
        float(metrics["mech295_bias_range_sum"]) / fired if fired > 0 else 0.0
    )
    metrics["action_entropy"] = _entropy(metrics["action_counts"])
    metrics["action_counts"] = {str(k): int(v) for k, v in metrics["action_counts"].items()}
    if agent.goal_state is not None:
        metrics["goal_norm_peak"] = float(getattr(agent.goal_state, "_goal_norm_peak", 0.0))
    else:
        metrics["goal_norm_peak"] = 0.0
    return metrics


def run_seed_arm_490k(
    seed: int,
    arm: ArmSpec,
    *,
    env_kwargs: Optional[Dict[str, Any]] = None,
    warmup_episodes: int,
    total_eval_step_budget: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(seed, env_kwargs)
    env._exq_env_kwargs = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    cfg = build_config_490k(env, arm)
    agent = REEAgent(cfg)
    label = f"seed={seed} arm={arm.arm_id}"
    print(f"Seed {seed} Condition {arm.arm_id}", flush=True)
    warmup_train(
        agent,
        env,
        num_episodes=warmup_episodes,
        steps_per_episode=steps_per_episode,
        label=label,
        progress_total_episodes=warmup_episodes,
    )
    metrics = eval_490k(
        agent,
        env,
        total_step_budget=total_eval_step_budget,
        steps_per_episode=steps_per_episode,
        seed=seed,
        arm_label=arm.arm_id,
    )
    print(
        "  [eval] {} steps={} mech295_fired={} argmin_flip_frac={:.3f} "
        "approach_rate={:.3f}".format(
            label,
            metrics["total_eval_steps"],
            metrics["mech295_fired_ticks"],
            metrics["mech295_argmin_flip_fraction"],
            metrics["approach_commit_rate"],
        ),
        flush=True,
    )
    # Per-run verdict line (progress bookkeeping only; the scientific outcome
    # is computed at the cohort level). PASS = the probe was exercised this run.
    run_ok = metrics["mech295_fired_ticks"] > 0
    print(f"verdict: {'PASS' if run_ok else 'FAIL'}", flush=True)
    return metrics


def _tv_distance(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """Total-variation distance between two action-class count distributions."""
    keys = set(counts_a) | set(counts_b)
    tot_a = max(1, sum(counts_a.values()))
    tot_b = max(1, sum(counts_b.values()))
    s = 0.0
    for k in keys:
        pa = counts_a.get(k, 0) / tot_a
        pb = counts_b.get(k, 0) / tot_b
        s += abs(pa - pb)
    return 0.5 * s


def evaluate_490k_cohort(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    full_rows = [r for r in rows if r.get("arm") == FULL_ARM]
    severed_rows = [r for r in rows if r.get("arm") == SEVERED_ARM]
    only_rows = [r for r in rows if r.get("arm") == ONLY_ARM]

    def n_seeds_with(rows_: List[Dict[str, Any]], pred) -> int:
        return sum(1 for r in rows_ if pred(r))

    # option (c) -- ARM_1 primary
    full_fired_seeds = n_seeds_with(full_rows, lambda r: int(r.get("mech295_fired_ticks", 0)) > 0)
    full_flip_seeds = n_seeds_with(
        full_rows, lambda r: float(r.get("mech295_argmin_flip_fraction", 0.0)) > 0.0
    )

    # option (a) -- ARM_1 vs ARM_0 per-seed action-class divergence
    per_seed_divergence: List[Dict[str, Any]] = []
    for f in full_rows:
        seed = f.get("seed")
        s = next((x for x in severed_rows if x.get("seed") == seed), None)
        if s is None:
            continue
        tv = _tv_distance(f.get("action_counts", {}), s.get("action_counts", {}))
        first_diff = int(f.get("first_commit_action", -1)) != int(
            s.get("first_commit_action", -1)
        )
        per_seed_divergence.append(
            {"seed": seed, "action_tv_distance": tv, "first_commit_action_differs": first_diff}
        )

    # option (b) -- ARM_2 sensitivity
    only_fired_seeds = n_seeds_with(only_rows, lambda r: int(r.get("mech295_fired_ticks", 0)) > 0)
    only_flip_seeds = n_seeds_with(
        only_rows, lambda r: float(r.get("mech295_argmin_flip_fraction", 0.0)) > 0.0
    )

    # --- pre-registered grid -> per-claim direction --------------------------
    probe_ran = full_fired_seeds >= SEEDS_PASS_MIN
    flips_consequential = full_flip_seeds >= SEEDS_PASS_MIN

    if not probe_ran and full_fired_seeds == 0:
        # ROW 3: probe could not run at all.
        per_claim_direction = "unknown"
        grid_row = "ROW_3_probe_never_fired"
        outcome = "FAIL"
    elif flips_consequential:
        # ROW 1: modulation flips the argmin on >=2/3 seeds.
        per_claim_direction = "supports"
        grid_row = "ROW_1_argmin_flip_consequential"
        outcome = "PASS"
    elif probe_ran:
        # ROW 2: bridge fires but never flips argmin.
        per_claim_direction = "weakens"
        grid_row = "ROW_2_fires_but_never_flips"
        outcome = "PASS"
    else:
        # mixed: fired on <2 seeds; treat as inconclusive but ran.
        per_claim_direction = "unknown"
        grid_row = "MIXED_partial_fire"
        outcome = "PASS"

    return {
        "outcome": outcome,
        "grid_row": grid_row,
        "per_claim_direction_mech295": per_claim_direction,
        "probe_ran": probe_ran,
        "full_arm_fired_seeds": int(full_fired_seeds),
        "full_arm_argmin_flip_seeds": int(full_flip_seeds),
        "only_arm_fired_seeds": int(only_fired_seeds),
        "only_arm_argmin_flip_seeds": int(only_flip_seeds),
        "per_seed_action_divergence": per_seed_divergence,
        "seeds_pass_min": SEEDS_PASS_MIN,
        "full_arm_id": FULL_ARM,
        "severed_arm_id": SEVERED_ARM,
        "only_arm_id": ONLY_ARM,
        "total_eval_step_budget": TOTAL_EVAL_STEP_BUDGET,
    }


def main(dry_run: bool = False) -> "Tuple[str, Path] | int":
    seeds = [SEEDS_DEFAULT[0]] if dry_run else SEEDS_DEFAULT
    warmup = 6 if dry_run else WARMUP_EPISODES_DEFAULT
    budget = 80 if dry_run else TOTAL_EVAL_STEP_BUDGET
    steps_per_ep = 40 if dry_run else STEPS_PER_EPISODE_DEFAULT

    print(f"[{EXPERIMENT_TYPE}] dry_run={dry_run}", flush=True)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        for arm in ARMS:
            cloned = ArmSpec(
                arm_id=arm.arm_id,
                gap4_operating=arm.gap4_operating,
                use_gabaergic_decay=arm.use_gabaergic_decay,
                use_pag_freeze_gate=arm.use_pag_freeze_gate,
                use_broadcast_override=arm.use_broadcast_override,
                extra_config=dict(arm.extra_config),
            )
            rows.append(
                run_seed_arm_490k(
                    seed,
                    cloned,
                    env_kwargs=ENV_FISHTANK_KWARGS,
                    warmup_episodes=warmup,
                    total_eval_step_budget=budget,
                    steps_per_episode=steps_per_ep,
                )
            )

    acceptance = evaluate_490k_cohort(rows)
    outcome = str(acceptance["outcome"])
    elapsed = time.time() - t0
    print(f"[{EXPERIMENT_TYPE}] acceptance={acceptance}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest.", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
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
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": acceptance["per_claim_direction_mech295"],
        "evidence_direction_per_claim": {
            "MECH-295": acceptance["per_claim_direction_mech295"],
        },
        "elapsed_seconds": elapsed,
        "design_notes": {
            "primary_metric": (
                "option (c) argmin-change probe in ARM_1: per tick where the "
                "MECH-295 cue bias fires, whether subtracting the MECH-295 "
                "contribution from e3.last_scores changes the argmin. Cannot "
                "ceiling-saturate (unlike approach_commit_rate, the 490j "
                "contamination)."
            ),
            "sanity_metric": (
                "option (a) ARM_1 vs ARM_0 (cue severed) per-seed action-class "
                "total-variation distance + first-committed-action difference."
            ),
            "sensitivity_metric": (
                "option (b) ARM_2 mech295-only (schema_wanting / tonic_5ht / "
                "backward_credit / MECH-307 consumer-conjunction-read all OFF): "
                "approach + argmin-flip with only the MECH-295 contribution "
                "active. Sensitivity analysis, NOT a necessity test."
            ),
            "predecessor": (
                "V3-EXQ-490j TERMINAL 2026-05-31 established MECH-295 is not "
                "behaviourally necessary (severed-bridge ARM_0 still reached "
                "approach_commit_rate=1.0 via parallel pathways) and the "
                "substrate-firing layer supports the modulatory reading. 490k "
                "tests whether the modulatory contribution is behaviourally "
                "CONSEQUENTIAL."
            ),
            "total_eval_step_budget": TOTAL_EVAL_STEP_BUDGET,
        },
        "gap4_operating": {
            "drive_floor": 0.9,
            "drive_ema_alpha": 1.0,
            "goal_stream": True,
            "use_dacc": True,
        },
        "acceptance": acceptance,
        "per_run": rows,
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path)
    sys.exit(0)
