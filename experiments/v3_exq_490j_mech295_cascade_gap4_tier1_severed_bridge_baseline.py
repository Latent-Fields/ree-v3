#!/opt/local/bin/python3
"""
V3-EXQ-490j: MECH-295 drive->liking->approach cascade Tier-1 retest with a TRUE
severed-bridge baseline arm and a DIRECT bridge-magnitude probe replacing the
contaminated goal_norm_peak delta metric used by V3-EXQ-490i.

Supersedes V3-EXQ-490i. The autopsy at
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-490i_2026-05-30.{md,json}
(REE_assembly bcdf9e2a0c, 2026-05-30) concluded:

  - The MECH-295 behavioural sign-test (approach_commit_rate ARM_1 vs ARM_0)
    PASSED in 3/3 seeds: ARM_1 = 1.0, ARM_0 = 0.0. This is the falsifiable
    test registered in the claims.yaml MECH-295 entry as the primary V3
    factorial.
  - The manifest's evidence_direction=weakens rolled up TWO failures that are
    NOT MECH-295 falsifications:
      (i)  C3_lift_vs_baseline FAIL = metric-design contamination. 490i used
           goal_norm_peak ARM_1 - ARM_0 as the substrate-side proxy. ARM_0
           kept z_goal_enabled=True (only drive_floor=0 + no goal_stream +
           no bridge stack), so ARM_0 still accumulated a goal_norm_peak
           baseline the metric assumed was zero. Seed-7 ARM_0 spiked to
           goal_norm_peak=12.4888 -- about 50x other ARM_0 seeds -- and
           dominated cross-seed comparison.
      (ii) C2 dacc_bias_nonzero_steps=0 across all 3 ARM_1 seeds despite
           cfg.use_dacc=True. This is a SEPARATE SD-032b consumer-pathway
           wiring gap (dACC bundle is constructed but _last_bundle /
           E3 score_bias adapter dormant). Going to a SEPARATE
           /diagnose-errors session, NOT 490j.

490j corrections (user-confirmed routing 2026-05-30T19:43Z):

  (a) TRUE severed-bridge baseline arm. ARM_0_severed_bridge sets
      cfg.goal.z_goal_enabled=False. GoalState.is_active() returns False
      structurally; agent.update_z_goal() short-circuits before the
      MECH-295 bridge is ever called; the bridge's anticipatory-liking
      write site is unreachable. Clean baseline.
  (b) Direct bridge-magnitude probe REPLACING the contaminated
      goal_norm_peak delta. The substrate-side proxy now measures the
      bridge's actual output via the load-bearing arithmetic from the
      MECH-295 implementation entry (claims.yaml MECH-295):
          - mech295_anticipatory_liking_write_sum  (per-tick
              compute_anticipatory_liking_write return value summed over
              the eval window)
          - mech295_anticipatory_liking_write_peak (peak value)
          - mech295_approach_cue_score_bias_sum    (per-tick
              compute_approach_cue_score_bias.abs().max() summed)
          - mech295_approach_cue_score_bias_peak   (peak value)
          - bridge_cue_fires                       (existing counter)
          - bridge_write_fires                     (existing counter)
      The behavioural sign-test (approach_commit_rate ARM_1 vs ARM_0) is
      RETAINED as a parallel acceptance criterion. It is the claim entry's
      registered primary falsifiable test and 490i passed it.
  (c) Consistent total-step budget across seeds. 490i had wild variance:
      ARM_1 seeds ran 1379 / 59 / 793 total eval steps (seed-7 ARM_1 ran
      only 59 steps total). 490j caps each (seed, arm) at a FIXED
      total_eval_step_budget = 900 ticks. The env is auto-reset on
      result.done; the eval marches until budget is exhausted; per-seed
      magnitude comparisons are calibrated. Choice of 900: above the 490i
      worst-case ARM_0 budget (911 at seed 19) and well above the
      mean (~744-1117), so cross-seed comparison floors at a uniform
      window.
  (d) Pre-registered interpretation grid in this docstring and in the
      queue entry description. Covers composite-FAIL cells (e.g. both
      C3_lift=False AND C6=False simultaneously), not just single-
      criterion rows. Each outcome row declares the per-claim evidence_
      direction it implies AND the routing step it forces.
  (e) experiment_purpose=evidence. supersedes=V3-EXQ-490i.
      claim_ids=[MECH-295] only (per claim_ids accuracy rule -- the
      cascade test does not test MECH-269b / ARC-030 / MECH-117 / Q-040
      at this metric resolution). architecture_epoch=ree_hybrid_guardrails_v1.
      run_id ends _v3.

C2 dacc_bias is recorded as a DIAGNOSTIC-ONLY metric in 490j (so the C2
dACC wiring gap remains visible for the parallel /diagnose-errors session
to consume), NOT a gating criterion. The autopsy's "do not gate on C2"
rule is encoded here.

Arms (2 x 3 seeds):
  ARM_0_severed_bridge:    gap4_operating substrate WITH the post-build
      override cfg.goal.z_goal_enabled=False. GoalState.is_active() is
      structurally False; bridge cannot fire; approach_commit at the
      VALENCE_WANTING-threshold gate cannot trip. This is the
      MECH-295-severed counterfactual.
  ARM_1_gap4_operating:    full GAP-4 operating config (drive_floor=0.9 +
      goal_stream + MECH-295 bridge + use_dacc=True via the rebuilt
      library default). Identical to 490i ARM_1.

Pre-registered Tier-1 acceptance on ARM_1_gap4_operating (>=2/3 seeds for
each row except where noted):
  C1 bridge_cue_fires             >= 1   (existing)
  C2 dacc_bias_nonzero_steps      DIAGNOSTIC-ONLY (NOT gating; see autopsy)
  C3 approach_commit_steps        >= 1   (existing; behavioural sign-test
                                          local to ARM_1)
  C4 goal_active_fraction         >= 0.05
  C5 bridge_write_fires           >= 1
  C6 mech295_anticipatory_liking_write_peak  > 0  (direct write-side)
  C7 mech295_approach_cue_score_bias_peak    > 0  (direct cue-side)
  C8 approach_commit_rate_lift    ARM_1 - ARM_0 >= 0.5 per seed
                                  (replaces 490i's contaminated
                                   goal_norm_peak delta; behavioural)
  C9 direct_bridge_magnitude_lift ARM_1 mech295_anticipatory_liking_
                                  write_sum > ARM_0 (+ a very small floor)
                                  AND ARM_1 mech295_approach_cue_score_
                                  bias_sum > ARM_0 (+ small floor) per
                                  seed (substrate-side; the load-bearing
                                  arithmetic from the claim entry).

Pre-registered Outcome Interpretation Grid
==========================================
Each row declares (per-claim evidence_direction, routing step). The grid
covers composite cells where MULTIPLE criteria fail jointly, which the
490i grid did not anticipate.

ROW 1 -- ALL PASS (C1, C3, C4, C5, C6, C7, C8, C9 all clear; C2 ignored):
  -> MECH-295 substrate validated on direct bridge magnitudes + behavioural
     sign-test simultaneously. The claim entry's primary falsifiable test
     plus the substrate-side proxy both fire.
  -> evidence_direction_per_claim["MECH-295"] = supports.
  -> Routing: governance (promote MECH-295 toward provisional, lift
     v3_pending after one stable cycle; clear evidence_quality_note
     entries 471/475/483/483a/483b/490/490b/490c/490e/490f/524 from the
     contamination block).

ROW 2 -- ALL CRITERIA PASS EXCEPT C2 (the autopsied dACC wiring gap):
  -> Identical scientific reading to ROW 1 for MECH-295. C2 is recorded
     for the parallel /diagnose-errors session.
  -> evidence_direction_per_claim["MECH-295"] = supports.
  -> Routing: governance MECH-295 -> supports; /diagnose-errors continues
     to own SD-032b dACC wiring independently. This is the EXPECTED row
     if the autopsy's diagnosis is correct.

ROW 3 -- C8 FAIL alone (behavioural lift collapses but direct bridge
         magnitudes C6+C7 PASS in ARM_1 AND C9 lifts):
  -> Substrate-side bridge fires cleanly; downstream linkage to the
     committed-state VALENCE_WANTING threshold gate is broken. The bridge
     produces a per-candidate negative score_bias that is too small to
     trip approach commitment.
  -> evidence_direction_per_claim["MECH-295"] = narrow_supports (substrate
     validated, behavioural-test ambiguity).
  -> Routing: /queue-experiment (parametric sweep on
     mech295_liking_to_approach_cue_gain and APPROACH_WANTING_THRESH).

ROW 4 -- C9 FAIL alone (substrate-side magnitude lift fails but ARM_1
         direct C6+C7 PASS and ARM_1 behavioural C3 + C8 PASS):
  -> The direct-magnitude SUM proxy is dominated by per-tick variance
     across ARM_0 (low-signal) and ARM_1 (low-signal-but-firing) -- the
     bridge fires in ARM_1 but not at sufficient cumulative magnitude
     to clear the small floor over the 900-tick window. Borderline.
  -> evidence_direction_per_claim["MECH-295"] = narrow_supports.
  -> Routing: review floor calibration; if behavioural C8 carried the
     PASS, MECH-295 reading stands.

ROW 5 -- C3 FAIL in ARM_1 alone (ARM_1 approach_commit_steps == 0)
         AND C8 FAIL (the autopsied 490i FAIL would have been here if
         the bridge had broken across the rebuild):
  -> Bridge cue side does not reach approach commitment in ARM_1. EITHER
     the rebuild regressed the bridge OR the env episode-length budget
     is masking it.
  -> evidence_direction_per_claim["MECH-295"] = weakens (narrow).
  -> Routing: /failure-autopsy on 490j; check bridge fire counts in
     ARM_1, episode-length distribution.

ROW 6 -- COMPOSITE: C3 FAIL AND C8 FAIL AND C9 FAIL (ARM_1 approach
         collapses AND substrate-side fails):
  -> Substrate failure path. Either the rebuild regressed the bridge or
     the env has changed in a way that disables the cascade.
  -> evidence_direction_per_claim["MECH-295"] = weakens.
  -> Routing: /failure-autopsy on the rebuilt library + bridge wiring
     audit.

ROW 7 -- COMPOSITE: C6 OR C7 FAIL in ARM_1 (direct bridge magnitudes
         zero across all 3 seeds) -- the bridge structurally did not
         fire even at the load-bearing arithmetic site:
  -> The bridge instance is not being constructed or its compute methods
     are not invoked at the integration sites in agent.py.
  -> evidence_direction_per_claim["MECH-295"] = non_contributory
     (test could not run as intended).
  -> Routing: /diagnose-errors (substrate-wiring failure, not a claim
     falsification).

ROW 8 -- COMPOSITE: C1 FAIL AND C5 FAIL (the bridge fire counters never
         increment in ARM_1):
  -> Bridge cue side never fires -- the agent's compute_anticipatory_
     liking_write / compute_approach_cue_score_bias calls return 0.0 on
     every tick. Most likely cause: drive_level never clears
     min_drive_to_fire under the env config, or z_goal_norm never clears
     min_z_goal_norm_to_fire.
  -> evidence_direction_per_claim["MECH-295"] = non_contributory.
  -> Routing: /diagnose-errors with the 490c-style probe (drive +
     z_goal_norm histograms vs floors).

ROW 9 -- ARM_0 contamination (ARM_0 bridge_cue_fires + bridge_write_fires
         + mech295_*_sum NON-ZERO -- the supposedly severed-bridge arm is
         not actually severed):
  -> Sentinel failure: cfg.goal.z_goal_enabled=False did not actually
     prevent the bridge from firing. Architecture-level issue (e.g.,
     GoalState reseeding from a different source).
  -> evidence_direction_per_claim["MECH-295"] = inconclusive
     (test invalid).
  -> Routing: /failure-autopsy on the severed-bridge contract.

C2_dacc_bias rows: regardless of C2 outcome on ARM_1, the autopsy already
routed the dACC wiring gap to a separate /diagnose-errors. The 490j review
should record the C2 number for that downstream session but NOT use it to
gate the per-claim direction for MECH-295.

claim_ids: [MECH-295]
experiment_purpose: evidence
supersedes: V3-EXQ-490i
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np
import torch

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness, StepHooks  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    APPROACH_WANTING_THRESH,
    ArmSpec,
    ENV_FISHTANK_KWARGS,
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    WARMUP_EPISODES_DEFAULT,
    build_config,
    make_env,
    warmup_train,
    _approach_commit,
    _dacc_bias_norm,
    _entropy,
)
from ree_core.agent import REEAgent  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_490j_mech295_cascade_gap4_tier1_severed_bridge_baseline"
QUEUE_ID = "V3-EXQ-490j"
CLAIM_IDS = ["MECH-295"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-490i"

# Fixed total-step eval budget per (seed, arm). See docstring section (c).
# Above the 490i ARM_0 worst-case (911 at seed 19) and well above mean.
TOTAL_EVAL_STEP_BUDGET = 900

# Acceptance criterion floors.
TIER1_CUE_FIRES_MIN = 1
TIER1_WRITE_FIRES_MIN = 1
TIER1_APPROACH_COMMIT_MIN = 1
TIER1_GOAL_ACTIVE_FRAC_MIN = 0.05
TIER1_DIRECT_MAGNITUDE_PEAK_MIN = 1e-6
APPROACH_COMMIT_LIFT_MIN = 0.5
DIRECT_BRIDGE_SUM_LIFT_FLOOR = 1e-3
SEEDS_PASS_MIN = 2

# 490j-specific severed-bridge ARM is gap4-substrate-on with a post-build
# override that flips cfg.goal.z_goal_enabled = False. ArmSpec does not
# carry a nested override channel and extending it touches the shared lib
# (used by 483c/524a/etc.). Keep changes script-scoped: pass a script-local
# sentinel via extra_config and intercept it in build_config_490j below.
SEVERED_BRIDGE_SENTINEL = "__490j_sever_z_goal_enabled__"

ARMS: List[ArmSpec] = [
    ArmSpec(
        "ARM_0_severed_bridge",
        gap4_operating=True,
        extra_config={SEVERED_BRIDGE_SENTINEL: True},
    ),
    ArmSpec("ARM_1_gap4_operating", gap4_operating=True),
]
GAP4_ARM = "ARM_1_gap4_operating"
BASE_ARM = "ARM_0_severed_bridge"


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_config_490j(env, arm: ArmSpec):
    """Wrap shared build_config to handle the severed-bridge override.

    The sentinel in arm.extra_config is consumed here and stripped before the
    shared build_config sees it (so the shared lib's hasattr+setattr loop
    does not write a bogus attribute onto cfg).
    """
    sever = bool(arm.extra_config.pop(SEVERED_BRIDGE_SENTINEL, False))
    cfg = build_config(env, arm)
    if sever:
        # Sever the goal stream structurally. GoalState.is_active() returns
        # False, so agent.update_z_goal short-circuits before the MECH-295
        # bridge is invoked. The bridge object is still constructed (master
        # flag use_mech295_liking_bridge is on under goal_stream factory),
        # but neither compute_anticipatory_liking_write nor compute_approach_
        # cue_score_bias is reached on the waking path. This is the clean
        # severed-bridge counterfactual per the 490i autopsy section 7(a).
        cfg.goal.z_goal_enabled = False
    return cfg


def _bridge_magnitude_recorder(bridge) -> Dict[str, float]:
    """Monkey-patch the bridge instance's two compute methods to record per-tick
    peak/sum of (a) anticipatory liking write magnitude and (b) approach-cue
    score_bias magnitude (max-abs across K candidates).

    The MECH-295 bridge module exposes _n_write_fires and _n_cue_fires as
    cumulative counters already, but does NOT expose per-tick magnitudes
    when the agent calls compute_anticipatory_liking_write /
    compute_approach_cue_score_bias separately (agent.py does not invoke
    tick(), so bridge._last_output stays None on the waking path). The
    direct-bridge-magnitude metric is the load-bearing arithmetic from the
    claim entry, so we record it here without touching the substrate.

    Returns a mutable stats dict the caller reads at end of eval. Safe to
    call once per (seed, arm) -- resets on each call.
    """
    stats: Dict[str, float] = {
        "anticipatory_liking_write_sum": 0.0,
        "anticipatory_liking_write_peak": 0.0,
        "anticipatory_liking_write_calls": 0,
        "approach_cue_score_bias_sum": 0.0,
        "approach_cue_score_bias_peak": 0.0,
        "approach_cue_score_bias_calls": 0,
    }
    if bridge is None:
        return stats
    orig_write = bridge.compute_anticipatory_liking_write
    orig_cue = bridge.compute_approach_cue_score_bias

    def wrapped_write(*args, **kwargs) -> float:
        v = orig_write(*args, **kwargs)
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        stats["anticipatory_liking_write_calls"] += 1
        if fv > 0.0:
            stats["anticipatory_liking_write_sum"] += fv
            if fv > stats["anticipatory_liking_write_peak"]:
                stats["anticipatory_liking_write_peak"] = fv
        return v

    def wrapped_cue(*args, **kwargs):
        out = orig_cue(*args, **kwargs)
        try:
            mag = float(out.abs().max().item())
        except Exception:
            mag = 0.0
        stats["approach_cue_score_bias_calls"] += 1
        if mag > 0.0:
            stats["approach_cue_score_bias_sum"] += mag
            if mag > stats["approach_cue_score_bias_peak"]:
                stats["approach_cue_score_bias_peak"] = mag
        return out

    bridge.compute_anticipatory_liking_write = wrapped_write
    bridge.compute_approach_cue_score_bias = wrapped_cue
    return stats


def eval_tier1_fixed_budget(
    agent: REEAgent,
    env,
    *,
    total_step_budget: int,
    steps_per_episode: int,
    seed: int,
    arm_label: str,
) -> Dict[str, Any]:
    """Custom eval loop with a FIXED total_step_budget per (seed, arm).

    Differs from goal_pipeline_tier1.eval_tier1 in three ways:
      1. Total-step budget cap (autopsy item c) -- env auto-resets on
         result.done; eval marches until budget is exhausted; cross-seed
         magnitude comparison is calibrated.
      2. Direct bridge magnitude metrics (autopsy item b) via the
         _bridge_magnitude_recorder monkey-patch on the bridge instance.
      3. dacc_bias recorded as DIAGNOSTIC-only (autopsy item: do not
         gate). The metric stays in the manifest for the parallel
         /diagnose-errors session to consume.
    """
    bridge = getattr(agent, "mech295_bridge", None)
    bridge_stats = _bridge_magnitude_recorder(bridge)

    metrics: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "approach_commit_steps": 0,
        "total_eval_steps": 0,
        "dacc_bias_nonzero_steps": 0,
        "bridge_cue_fires": 0,
        "bridge_write_fires": 0,
        "goal_active_steps": 0,
        "resource_contacts": 0,
        "action_counts": {},
        "n_episodes_started": 0,
        "n_episodes_ended_done": 0,
    }

    def on_post_step(*, agent, latent, action, obs_dict, ticks, step, **kwargs) -> None:
        metrics["total_eval_steps"] += 1
        if _approach_commit(agent):
            metrics["approach_commit_steps"] += 1
        if _dacc_bias_norm(agent) > 1e-6:
            metrics["dacc_bias_nonzero_steps"] += 1
        if agent.goal_state is not None and agent.goal_state.is_active():
            metrics["goal_active_steps"] += 1
        if bridge is not None:
            metrics["bridge_cue_fires"] = int(getattr(bridge, "_n_cue_fires", 0))
            metrics["bridge_write_fires"] = int(getattr(bridge, "_n_write_fires", 0))

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    # Per-eval reset of bridge counters (cumulative across episodes within
    # this eval, but not across seeds/arms which use a fresh bridge).
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
        ttype = result.info.get("transition_type", "none")
        if ttype == "resource":
            metrics["resource_contacts"] += 1
        aidx = int(result.action.argmax(dim=-1).item())
        ac = metrics["action_counts"]
        ac[aidx] = ac.get(aidx, 0) + 1
        obs_dict = result.next_obs_dict
        # Auto-reset on env done OR steps_per_episode cap (defensive --
        # do not let one episode hog the entire budget).
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
    metrics["approach_commit_rate"] = float(metrics["approach_commit_steps"]) / total
    metrics["goal_active_fraction"] = float(metrics["goal_active_steps"]) / total
    metrics["action_entropy"] = _entropy(metrics["action_counts"])
    metrics["action_counts"] = {str(k): int(v) for k, v in metrics["action_counts"].items()}
    if agent.goal_state is not None:
        metrics["goal_norm_peak"] = float(getattr(agent.goal_state, "_goal_norm_peak", 0.0))
    else:
        metrics["goal_norm_peak"] = 0.0
    # Direct bridge magnitude metrics (the autopsy fix).
    metrics["mech295_anticipatory_liking_write_sum"] = float(
        bridge_stats["anticipatory_liking_write_sum"]
    )
    metrics["mech295_anticipatory_liking_write_peak"] = float(
        bridge_stats["anticipatory_liking_write_peak"]
    )
    metrics["mech295_anticipatory_liking_write_calls"] = int(
        bridge_stats["anticipatory_liking_write_calls"]
    )
    metrics["mech295_approach_cue_score_bias_sum"] = float(
        bridge_stats["approach_cue_score_bias_sum"]
    )
    metrics["mech295_approach_cue_score_bias_peak"] = float(
        bridge_stats["approach_cue_score_bias_peak"]
    )
    metrics["mech295_approach_cue_score_bias_calls"] = int(
        bridge_stats["approach_cue_score_bias_calls"]
    )
    return metrics


def run_seed_arm_490j(
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
    cfg = build_config_490j(env, arm)
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
    metrics = eval_tier1_fixed_budget(
        agent,
        env,
        total_step_budget=total_eval_step_budget,
        steps_per_episode=steps_per_episode,
        seed=seed,
        arm_label=arm.arm_id,
    )
    print(
        "  [eval] {} steps={} cue_fires={} write_fires={} approach_rate={:.3f}".format(
            label,
            metrics["total_eval_steps"],
            metrics["bridge_cue_fires"],
            metrics["bridge_write_fires"],
            metrics["approach_commit_rate"],
        ),
        flush=True,
    )
    return metrics


def evaluate_490j_cohort(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-seed pair evaluation against the pre-registered acceptance grid.

    See module docstring for the full grid and routing rules. This function
    only computes the boolean acceptance vector; routing is applied at
    review time by the human reviewer from the per-claim grid.
    """
    gap4_rows = [r for r in rows if r.get("arm") == GAP4_ARM]
    base_rows = [r for r in rows if r.get("arm") == BASE_ARM]

    per_seed_arm1: List[Dict[str, bool]] = []
    for g in gap4_rows:
        seed = g.get("seed")
        b = next((x for x in base_rows if x.get("seed") == seed), None)
        if b is None:
            continue
        checks = {
            "C1_cue_fires": int(g.get("bridge_cue_fires", 0)) >= TIER1_CUE_FIRES_MIN,
            "C3_approach_commit": int(g.get("approach_commit_steps", 0))
            >= TIER1_APPROACH_COMMIT_MIN,
            "C4_goal_active": float(g.get("goal_active_fraction", 0.0))
            >= TIER1_GOAL_ACTIVE_FRAC_MIN,
            "C5_write_fires": int(g.get("bridge_write_fires", 0))
            >= TIER1_WRITE_FIRES_MIN,
            "C6_anticipatory_write_peak": float(
                g.get("mech295_anticipatory_liking_write_peak", 0.0)
            )
            > TIER1_DIRECT_MAGNITUDE_PEAK_MIN,
            "C7_approach_cue_bias_peak": float(
                g.get("mech295_approach_cue_score_bias_peak", 0.0)
            )
            > TIER1_DIRECT_MAGNITUDE_PEAK_MIN,
            "C8_approach_commit_lift": float(g.get("approach_commit_rate", 0.0))
            - float(b.get("approach_commit_rate", 0.0))
            >= APPROACH_COMMIT_LIFT_MIN,
            "C9_direct_magnitude_lift": (
                float(g.get("mech295_anticipatory_liking_write_sum", 0.0))
                > float(b.get("mech295_anticipatory_liking_write_sum", 0.0))
                + DIRECT_BRIDGE_SUM_LIFT_FLOOR
                and float(g.get("mech295_approach_cue_score_bias_sum", 0.0))
                > float(b.get("mech295_approach_cue_score_bias_sum", 0.0))
                + DIRECT_BRIDGE_SUM_LIFT_FLOOR
            ),
        }
        per_seed_arm1.append(checks)

    def n_pass(key: str) -> int:
        return sum(1 for p in per_seed_arm1 if p.get(key, False))

    # ARM_0 severed-bridge sentinel: bridge fire counters MUST be zero in
    # the severed arm. Non-zero indicates the override failed (row 9 of
    # the interpretation grid).
    severed_bridge_clean = all(
        int(r.get("bridge_cue_fires", 0)) == 0
        and int(r.get("bridge_write_fires", 0)) == 0
        for r in base_rows
    )

    c1 = n_pass("C1_cue_fires") >= SEEDS_PASS_MIN
    c3 = n_pass("C3_approach_commit") >= SEEDS_PASS_MIN
    c4 = n_pass("C4_goal_active") >= SEEDS_PASS_MIN
    c5 = n_pass("C5_write_fires") >= SEEDS_PASS_MIN
    c6 = n_pass("C6_anticipatory_write_peak") >= SEEDS_PASS_MIN
    c7 = n_pass("C7_approach_cue_bias_peak") >= SEEDS_PASS_MIN
    c8 = n_pass("C8_approach_commit_lift") >= SEEDS_PASS_MIN
    c9 = n_pass("C9_direct_magnitude_lift") >= SEEDS_PASS_MIN

    # C2 is DIAGNOSTIC-only per the 490i autopsy. Recorded for transparency.
    c2_diagnostic = sum(
        1
        for g in gap4_rows
        if int(g.get("dacc_bias_nonzero_steps", 0)) >= 1
    )

    passed = bool(c1 and c3 and c4 and c5 and c6 and c7 and c8 and c9 and severed_bridge_clean)
    return {
        "pass": passed,
        "C1_cue_fires": c1,
        "C2_dacc_bias_diagnostic_seeds_nonzero": int(c2_diagnostic),
        "C3_approach_commit": c3,
        "C4_goal_active": c4,
        "C5_write_fires": c5,
        "C6_anticipatory_write_peak": c6,
        "C7_approach_cue_bias_peak": c7,
        "C8_approach_commit_lift": c8,
        "C9_direct_magnitude_lift": c9,
        "severed_bridge_sentinel_clean": severed_bridge_clean,
        "per_seed_arm1_checks": per_seed_arm1,
        "gap4_arm_id": GAP4_ARM,
        "baseline_arm_id": BASE_ARM,
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
            # extra_config is mutated by build_config_490j (sentinel pop),
            # so clone it per-seed to keep ARMS module-level state stable.
            cloned = ArmSpec(
                arm_id=arm.arm_id,
                gap4_operating=arm.gap4_operating,
                use_gabaergic_decay=arm.use_gabaergic_decay,
                use_pag_freeze_gate=arm.use_pag_freeze_gate,
                use_broadcast_override=arm.use_broadcast_override,
                extra_config=dict(arm.extra_config),
            )
            rows.append(
                run_seed_arm_490j(
                    seed,
                    cloned,
                    env_kwargs=ENV_FISHTANK_KWARGS,
                    warmup_episodes=warmup,
                    total_eval_step_budget=budget,
                    steps_per_episode=steps_per_ep,
                )
            )

    acceptance = evaluate_490j_cohort(rows)
    outcome = "PASS" if acceptance["pass"] else "FAIL"
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
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "MECH-295": "supports" if outcome == "PASS" else "weakens",
        },
        "elapsed_seconds": elapsed,
        "design_notes": {
            "severed_bridge_arm": (
                "ARM_0_severed_bridge sets cfg.goal.z_goal_enabled=False after "
                "build_config; GoalState.is_active() returns False; the MECH-295 "
                "bridge cannot fire on the waking path. Replaces the 490i "
                "ARM_0_legacy_collapsed arm whose drive_floor=0 + goal_stream off + "
                "z_goal_enabled=True left a goal_norm_peak baseline that contaminated "
                "the substrate-side delta metric."
            ),
            "direct_bridge_magnitude_probe": (
                "Substrate-side proxy is the per-tick anticipatory liking write "
                "value + per-tick approach-cue score_bias magnitude, recorded by "
                "monkey-patching the bridge instance's compute methods at eval time. "
                "Replaces the contaminated 490i goal_norm_peak delta. The "
                "behavioural sign-test (approach_commit_rate ARM_1 vs ARM_0; "
                "claims.yaml MECH-295 primary falsifiable test) is retained as a "
                "parallel acceptance criterion."
            ),
            "total_eval_step_budget": TOTAL_EVAL_STEP_BUDGET,
            "total_eval_step_budget_justification": (
                "490i ARM_1 total_eval_steps ranged 59 / 793 / 1379 across seeds "
                "42 / 7 / 19; ARM_0 ranged 441 / 911 / 2000. Cross-seed magnitude "
                "comparisons were uncalibrated. 490j caps every (seed, arm) at a "
                "fixed budget above the 490i ARM_0 mean. Env auto-resets on done; "
                "eval marches until budget exhausted."
            ),
            "c2_dacc_bias_is_diagnostic_only": (
                "Per the 490i autopsy, C2 dacc_bias=0 across all ARM_1 seeds was a "
                "separate SD-032b consumer-pathway wiring gap, routed to /diagnose-"
                "errors. 490j records dacc_bias_nonzero_steps for the parallel "
                "diagnostic session to consume but does NOT gate the acceptance on it."
            ),
        },
        "gap4_operating": {
            "drive_floor": 0.9,
            "drive_ema_alpha": 1.0,
            "goal_stream": True,
            "use_dacc": True,
            "library_rebuild": "2026-05-29 Fork A (V3-EXQ-490g-cohort autopsy)",
        },
        "acceptance": acceptance,
        "per_run": rows,
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
    if result == 0:
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path)
    sys.exit(0)
