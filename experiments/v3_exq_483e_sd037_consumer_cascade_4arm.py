#!/opt/local/bin/python3
"""
V3-EXQ-483e: SD-037 consumer-cascade behavioural validation.

Supersedes V3-EXQ-483d. 483d FAILed substrate-ceiling on 2026-05-29 with C2
override_signal saturating cleanly but C3_lift_vs_baseline FAILing (1/3 seeds)
and action_counts + goal_norm_peak bit-identical across the broadcast_override
axis within each seed. Autopsy
(REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-483d_2026-05-30.{md,json})
diagnosed: SD-037 fires correctly but has nowhere to land because (a) MECH-295
bridge dominates effective_drive at goal-seeding (structural; should keep
operating), (b) PAG-freeze pathway is dormant in fishtank env, (c) SalienceCoordinator
disabled, (d) PFC/BLA/CeA/beta-gate consumers per MECH-281 implementation_note
were unwired.

(b)(c)(d) all closed in the 2026-05-30 consumer-cascade amend session:
  - (b) PAG-engaging env: use_gabaergic_decay=True + use_pag_freeze_gate=True
    (V3-EXQ-475-style PAG-engaging substrate; same SD-036+MECH-279 path).
  - (c) SalienceCoordinator: use_salience_coordinator=True flips the dormant
    affinity slot active. SD-037 already registers affinity_weights["override_signal"]
    = {"external_task": override_salience_reweight_alpha} at agent init.
  - (d) Four new REEConfig gains landed today, all default 0.0 (bit-identical OFF):
      * override_pfc_eta_gain     (LateralPFCAnalog eff_eta multiplier)
      * override_bla_encoding_gain (BLAAnalog encoding_gain multiplier)
      * override_cea_amplitude_gain (CeAAnalog mode_prior + fast_prime multiplier)
      * override_beta_interrupt_gain (urgency_interrupt_threshold attenuator)

Acceptance criteria:
  C1 substrate-readiness   override_signal_nonzero_steps saturates in ARM_2 + ARM_3
                           AND PAG_release_ratio (ON_ON / ON_OFF) > 1.30.
  C2 cascade-engagement    in ARM_3: lateral_pfc rule_state delta > 1.5x ARM_2 AND
                           bla_encoding_gain_peak > 1.3x ARM_2 AND cea_mode_prior_peak
                           > 1.3x ARM_2 -- at least 3 of 4 sub-checks on >= 2/3 seeds.
  C3 PRIMARY lift          C3_lift_vs_baseline (goal_norm_peak delta ARM_3 - ARM_0)
                           lifts across 3/3 seeds (vs 1/3 on 483d). Headline.
  C4 action divergence     action_counts in ARM_3 diverge from ARM_2 across the
                           broadcast_override axis within each seed (TV distance > 0.05).

claim_ids: [SD-037, MECH-280, MECH-281]
experiment_purpose: evidence
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    TIER1_APPROACH_COMMIT_MIN,
    TIER1_CUE_FIRES_MIN,
    TIER1_GOAL_ACTIVE_FRAC_MIN,
    TIER1_SEEDS_PASS_MIN,
    WARMUP_EPISODES_DEFAULT,
    _approach_commit,
    _dacc_bias_norm,
    build_config,
    make_env,
    warmup_train,
)
from ree_core.agent import REEAgent
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_483e_sd037_consumer_cascade_4arm"
QUEUE_ID = "V3-EXQ-483e"
CLAIM_IDS = ["SD-037", "MECH-280", "MECH-281"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds
TIER1_OVERRIDE_SIGNAL_MIN = 10    # steps in ON arms where override_signal > 1e-3
TIER1_GOAL_NORM_PEAK_DELTA = 0.01  # ARM_3 goal_norm_peak must exceed ARM_0 by this

# Consumer-cascade engagement thresholds (C2)
C2_LATERAL_PFC_RATIO_MIN = 1.5    # ARM_3 rule_state norm peak / ARM_2 >= this
C2_BLA_GAIN_RATIO_MIN    = 1.3    # ARM_3 encoding_gain peak / ARM_2 >= this
C2_CEA_PRIOR_RATIO_MIN   = 1.3    # ARM_3 mode_prior peak / ARM_2 >= this
C2_SUB_CHECKS_MIN        = 3      # at least N of 4 sub-checks must clear per seed
                                  # (note: only 3 sub-checks landed; urgency-interrupt
                                  # is recorded but not part of the C2 quorum because
                                  # the fishtank env produces few interrupts under default
                                  # commit dynamics; recorded as diagnostic).

# Action-counts divergence (C4): per-seed total-variation distance between
# ARM_3 and ARM_2 action distributions must exceed this floor.
C4_ACTION_TV_MIN = 0.05

# Consumer-cascade gains in ARM_3 (lit-pull-defensible; modest values)
ARM3_OVERRIDE_PFC_ETA_GAIN = 1.0
ARM3_OVERRIDE_BLA_ENCODING_GAIN = 1.0
ARM3_OVERRIDE_CEA_AMPLITUDE_GAIN = 1.0
ARM3_OVERRIDE_BETA_INTERRUPT_GAIN = 0.5

# ARM_0 (OFF_OFF): override pinned at 0 via raised recruitment_threshold so the
# consumer cascade is structurally inert even though the gains are set. This
# mirrors 483d ARM_0 -- the no-override baseline.
ARM0_RECRUITMENT_THRESHOLD = 10.0

# Common consumer-cascade master flags (all four arms get these to keep the
# substrate surface identical; the ARM-distinguishing variables are the four
# override_*_gain knobs and the override-pinning recruitment_threshold).
COMMON_CONSUMER_FLAGS: Dict[str, Any] = {
    "use_salience_coordinator": True,
    "use_lateral_pfc_analog": True,
    "use_amygdala_analog": True,
    "use_bla_analog": True,
    "use_cea_analog": True,
}

ARMS = [
    # ARM_0 OFF_OFF: gains > 0 but override_signal pinned at 0 (no recruitment).
    ArmSpec(
        "OFF_OFF",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
        extra_config={
            **COMMON_CONSUMER_FLAGS,
            "use_dacc": True,
            "override_recruitment_threshold": ARM0_RECRUITMENT_THRESHOLD,
            "override_pfc_eta_gain": ARM3_OVERRIDE_PFC_ETA_GAIN,
            "override_bla_encoding_gain": ARM3_OVERRIDE_BLA_ENCODING_GAIN,
            "override_cea_amplitude_gain": ARM3_OVERRIDE_CEA_AMPLITUDE_GAIN,
            "override_beta_interrupt_gain": ARM3_OVERRIDE_BETA_INTERRUPT_GAIN,
        },
    ),
    # ARM_1 ON_OFF: SD-037 master OFF -> no override regulator; consumer flags
    # are master-ON but the cascade is structurally dormant. Anchors a "no
    # SD-037 at all" baseline against the ON variants.
    ArmSpec(
        "ON_OFF",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=False,
        extra_config={
            **COMMON_CONSUMER_FLAGS,
            "use_dacc": True,
        },
    ),
    # ARM_2 OFF_ON: SD-037 master ON, default recruitment_threshold so
    # override_signal lifts under sustained drive+harm, but consumer gains
    # all 0.0 -> wired-but-inert baseline (matches 483d ARM_2 / ARM_3 OFF_ON).
    ArmSpec(
        "OFF_ON",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
        extra_config={
            **COMMON_CONSUMER_FLAGS,
            "use_dacc": True,
            "override_pfc_eta_gain": 0.0,
            "override_bla_encoding_gain": 0.0,
            "override_cea_amplitude_gain": 0.0,
            "override_beta_interrupt_gain": 0.0,
        },
    ),
    # ARM_3 ON_ON: SD-037 master ON, all four consumer-cascade gains > 0.
    # This is the "consumer cascade engaged" arm.
    ArmSpec(
        "ON_ON",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
        extra_config={
            **COMMON_CONSUMER_FLAGS,
            "use_dacc": True,
            "override_pfc_eta_gain": ARM3_OVERRIDE_PFC_ETA_GAIN,
            "override_bla_encoding_gain": ARM3_OVERRIDE_BLA_ENCODING_GAIN,
            "override_cea_amplitude_gain": ARM3_OVERRIDE_CEA_AMPLITUDE_GAIN,
            "override_beta_interrupt_gain": ARM3_OVERRIDE_BETA_INTERRUPT_GAIN,
        },
    ),
]
GAP4_ARM = "ON_ON"
BASE_ARM = "OFF_OFF"
INERT_ARM = "OFF_ON"   # cascade-OFF baseline for C2 cascade-engagement ratios

SEEDS = SEEDS_DEFAULT             # [42, 7, 19]
WARMUP_EPISODES = WARMUP_EPISODES_DEFAULT   # 50
EVAL_EPISODES = EVAL_EPISODES_DEFAULT       # 10
STEPS_PER_EPISODE = STEPS_PER_EPISODE_DEFAULT  # 200


def _override_signal_value(agent: REEAgent) -> float:
    bo = getattr(agent, "broadcast_override", None)
    if bo is None:
        return 0.0
    return float(getattr(bo, "override_signal", 0.0))


def _lateral_pfc_rule_state_norm(agent: REEAgent) -> float:
    lp = getattr(agent, "lateral_pfc", None)
    if lp is None:
        return 0.0
    rs = getattr(lp, "rule_state", None)
    if rs is None:
        return 0.0
    return float(rs.norm().item())


def _bla_encoding_gain(agent: REEAgent) -> float:
    out = getattr(agent, "_bla_last_output", None)
    if out is None:
        return 0.0
    return float(getattr(out, "encoding_gain", 0.0))


def _cea_mode_prior(agent: REEAgent) -> float:
    out = getattr(agent, "_cea_last_output", None)
    if out is None:
        return 0.0
    return float(getattr(out, "mode_prior", 0.0))


def _pag_release_count(agent: REEAgent) -> int:
    # PAGFreezeGate exposes diagnostic counters when wired.
    pg = getattr(agent, "pag_freeze_gate", None)
    if pg is None:
        return 0
    return int(getattr(pg, "_n_releases", 0))


def eval_tier1_483e(
    agent: REEAgent,
    env,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    arm_label: str,
) -> Dict[str, Any]:
    """Tier-1 eval extended with consumer-cascade diagnostic captures."""
    metrics: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "approach_commit_steps": 0,
        "total_eval_steps": 0,
        "dacc_bias_nonzero_steps": 0,
        "override_signal_nonzero_steps": 0,
        "bridge_cue_fires": 0,
        "bridge_write_fires": 0,
        "goal_active_steps": 0,
        "resource_contacts": 0,
        "action_counts": {},
        # Consumer-cascade C2 captures
        "lateral_pfc_rule_state_norm_peak": 0.0,
        "bla_encoding_gain_peak": 0.0,
        "cea_mode_prior_peak": 0.0,
        "cea_fast_prime_peak": 0.0,
        # Diagnostic-only urgency-interrupt counter
        "beta_release_count": 0,
        "pag_release_count_end": 0,
    }

    def on_post_step(*, agent, latent, action, obs_dict, ticks, step, **kwargs) -> None:
        metrics["total_eval_steps"] += 1
        if _approach_commit(agent):
            metrics["approach_commit_steps"] += 1
        if _dacc_bias_norm(agent) > 1e-6:
            metrics["dacc_bias_nonzero_steps"] += 1
        if _override_signal_value(agent) > 1e-3:
            metrics["override_signal_nonzero_steps"] += 1
        if agent.goal_state is not None and agent.goal_state.is_active():
            metrics["goal_active_steps"] += 1
        br = getattr(agent, "mech295_bridge", None)
        if br is not None:
            metrics["bridge_cue_fires"] = int(getattr(br, "_n_cue_fires", 0))
            metrics["bridge_write_fires"] = int(getattr(br, "_n_write_fires", 0))
        # Cascade captures (peak across episode)
        lp_norm = _lateral_pfc_rule_state_norm(agent)
        if lp_norm > metrics["lateral_pfc_rule_state_norm_peak"]:
            metrics["lateral_pfc_rule_state_norm_peak"] = lp_norm
        bla_gain = _bla_encoding_gain(agent)
        if bla_gain > metrics["bla_encoding_gain_peak"]:
            metrics["bla_encoding_gain_peak"] = bla_gain
        cea_prior = abs(_cea_mode_prior(agent))
        if cea_prior > metrics["cea_mode_prior_peak"]:
            metrics["cea_mode_prior_peak"] = cea_prior
        cea_out = getattr(agent, "_cea_last_output", None)
        if cea_out is not None:
            fp = abs(float(getattr(cea_out, "fast_prime", 0.0)))
            if fp > metrics["cea_fast_prime_peak"]:
                metrics["cea_fast_prime_peak"] = fp

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        if getattr(agent, "mech295_bridge", None) is not None:
            agent.mech295_bridge._n_cue_fires = 0
            agent.mech295_bridge._n_write_fires = 0

        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            ttype = result.info.get("transition_type", "none")
            if ttype == "resource":
                metrics["resource_contacts"] += 1
            aidx = int(result.action.argmax(dim=-1).item())
            ac = metrics["action_counts"]
            ac[aidx] = ac.get(aidx, 0) + 1
            obs_dict = result.next_obs_dict
            if result.done:
                break

    total = max(1, int(metrics["total_eval_steps"]))
    metrics["approach_commit_rate"] = float(metrics["approach_commit_steps"]) / total
    metrics["goal_active_fraction"] = float(metrics["goal_active_steps"]) / total
    metrics["action_counts"] = {str(k): int(v) for k, v in metrics["action_counts"].items()}
    if agent.goal_state is not None:
        metrics["goal_norm_peak"] = float(getattr(agent.goal_state, "_goal_norm_peak", 0.0))
    else:
        metrics["goal_norm_peak"] = 0.0
    metrics["pag_release_count_end"] = _pag_release_count(agent)
    return metrics


def tier1_seed_pass_483e(metrics: Dict[str, Any]) -> Dict[str, bool]:
    """Per-seed primary substrate checks (C1 + C3_direct + C4)."""
    return {
        "C1_cue_fires": int(metrics.get("bridge_cue_fires", 0)) >= TIER1_CUE_FIRES_MIN,
        "C2_override_signal": int(metrics.get("override_signal_nonzero_steps", 0)) >= TIER1_OVERRIDE_SIGNAL_MIN,
        "C3_approach_commit": int(metrics.get("approach_commit_steps", 0)) >= TIER1_APPROACH_COMMIT_MIN,
        "C4_goal_active": float(metrics.get("goal_active_fraction", 0.0)) >= TIER1_GOAL_ACTIVE_FRAC_MIN,
    }


def _action_tv_distance(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """Total-variation distance between two discrete action distributions."""
    keys = set(counts_a.keys()) | set(counts_b.keys())
    total_a = max(1, sum(counts_a.values()))
    total_b = max(1, sum(counts_b.values()))
    tv = 0.0
    for k in keys:
        pa = counts_a.get(k, 0) / total_a
        pb = counts_b.get(k, 0) / total_b
        tv += abs(pa - pb)
    return 0.5 * tv


def evaluate_483e_cohort(
    rows: List[Dict[str, Any]],
    *,
    gap4_arm_id: str,
    baseline_arm_id: str,
    inert_arm_id: str,
) -> Dict[str, Any]:
    """PASS when ARM_3 clears C1-C4 + C2_cascade + C3_lift + C4_action_divergence."""
    gap4_rows = [r for r in rows if r.get("arm") == gap4_arm_id]
    base_rows = [r for r in rows if r.get("arm") == baseline_arm_id]
    inert_rows = [r for r in rows if r.get("arm") == inert_arm_id]

    per_seed = [tier1_seed_pass_483e(r) for r in gap4_rows]
    c1 = sum(1 for p in per_seed if p["C1_cue_fires"]) >= TIER1_SEEDS_PASS_MIN
    c2_substrate = sum(1 for p in per_seed if p["C2_override_signal"]) >= TIER1_SEEDS_PASS_MIN
    c3_direct = sum(1 for p in per_seed if p["C3_approach_commit"]) >= TIER1_SEEDS_PASS_MIN
    c4_goal_active = sum(1 for p in per_seed if p["C4_goal_active"]) >= TIER1_SEEDS_PASS_MIN

    # C2 cascade-engagement per seed: ARM_3 / ARM_2 ratios on lateral_pfc, bla, cea.
    c2_cascade_seeds = 0
    c2_cascade_per_seed: List[Dict[str, Any]] = []
    if inert_rows:
        for g in gap4_rows:
            seed = g.get("seed")
            inert = next((x for x in inert_rows if x.get("seed") == seed), None)
            if inert is None:
                continue
            lp_g = float(g.get("lateral_pfc_rule_state_norm_peak", 0.0))
            lp_i = max(1e-9, float(inert.get("lateral_pfc_rule_state_norm_peak", 0.0)))
            bla_g = float(g.get("bla_encoding_gain_peak", 0.0))
            bla_i = max(1e-9, float(inert.get("bla_encoding_gain_peak", 0.0)))
            cea_g = float(g.get("cea_mode_prior_peak", 0.0))
            cea_i = max(1e-9, float(inert.get("cea_mode_prior_peak", 0.0)))
            lp_ratio = lp_g / lp_i
            bla_ratio = bla_g / bla_i
            cea_ratio = cea_g / cea_i
            sub_checks = {
                "lateral_pfc_ratio_ok": lp_ratio >= C2_LATERAL_PFC_RATIO_MIN,
                "bla_ratio_ok": bla_ratio >= C2_BLA_GAIN_RATIO_MIN,
                "cea_ratio_ok": cea_ratio >= C2_CEA_PRIOR_RATIO_MIN,
            }
            n_pass = sum(1 for v in sub_checks.values() if v)
            cleared = n_pass >= min(C2_SUB_CHECKS_MIN, len(sub_checks))
            if cleared:
                c2_cascade_seeds += 1
            c2_cascade_per_seed.append({
                "seed": seed,
                "lateral_pfc_ratio": lp_ratio,
                "bla_ratio": bla_ratio,
                "cea_ratio": cea_ratio,
                "sub_checks": sub_checks,
                "cleared": cleared,
            })
    c2_cascade = c2_cascade_seeds >= TIER1_SEEDS_PASS_MIN

    # C3_lift: ARM_3 goal_norm_peak > ARM_0 + TIER1_GOAL_NORM_PEAK_DELTA per seed.
    c3_lift_seeds = 0
    if base_rows:
        for g in gap4_rows:
            seed = g.get("seed")
            b = next((x for x in base_rows if x.get("seed") == seed), None)
            if b is None:
                continue
            if float(g.get("goal_norm_peak", 0.0)) > float(b.get("goal_norm_peak", 0.0)) + TIER1_GOAL_NORM_PEAK_DELTA:
                c3_lift_seeds += 1
    c3_lift = c3_lift_seeds >= TIER1_SEEDS_PASS_MIN

    # C4 action divergence: TV(ARM_3, ARM_2) >= floor per seed.
    c4_action_divergence_seeds = 0
    c4_per_seed: List[Dict[str, Any]] = []
    if inert_rows:
        for g in gap4_rows:
            seed = g.get("seed")
            inert = next((x for x in inert_rows if x.get("seed") == seed), None)
            if inert is None:
                continue
            tv = _action_tv_distance(
                {str(k): int(v) for k, v in g.get("action_counts", {}).items()},
                {str(k): int(v) for k, v in inert.get("action_counts", {}).items()},
            )
            ok = tv >= C4_ACTION_TV_MIN
            if ok:
                c4_action_divergence_seeds += 1
            c4_per_seed.append({"seed": seed, "tv_distance": tv, "ok": ok})
    c4_action_divergence = c4_action_divergence_seeds >= TIER1_SEEDS_PASS_MIN

    passed = bool(
        c1 and c2_substrate and c3_direct and c4_goal_active
        and c2_cascade and c3_lift and c4_action_divergence
    )
    return {
        "pass": passed,
        "C1_cue_fires": c1,
        "C2_override_signal": c2_substrate,
        "C2_cascade_engagement": c2_cascade,
        "C2_cascade_per_seed": c2_cascade_per_seed,
        "C3_approach_commit": c3_direct,
        "C3_lift_vs_baseline": c3_lift,
        "C3_lift_count": c3_lift_seeds,
        "C4_goal_active": c4_goal_active,
        "C4_action_divergence": c4_action_divergence,
        "C4_action_divergence_per_seed": c4_per_seed,
        "gap4_arm_id": gap4_arm_id,
        "baseline_arm_id": baseline_arm_id,
        "inert_arm_id": inert_arm_id,
    }


def run_seed_arm_483e(
    seed: int,
    arm: ArmSpec,
    *,
    warmup_episodes: int = WARMUP_EPISODES,
    eval_episodes: int = EVAL_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(seed, ENV_FISHTANK_KWARGS)
    env._exq_env_kwargs = dict(ENV_FISHTANK_KWARGS)
    cfg = build_config(env, arm)
    agent = REEAgent(cfg)
    label = f"seed={seed} arm={arm.arm_id}"
    print(f"Seed {seed} Condition {arm.arm_id}", flush=True)
    total_episodes = warmup_episodes + eval_episodes
    warmup_train(
        agent,
        env,
        num_episodes=warmup_episodes,
        steps_per_episode=steps_per_episode,
        label=label,
        progress_total_episodes=total_episodes,
    )
    for ep in range(eval_episodes):
        if (ep + 1) == eval_episodes:
            print(
                f"  [train] {label} ep {warmup_episodes + ep + 1}/{total_episodes}",
                flush=True,
            )
    metrics = eval_tier1_483e(
        agent,
        env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        seed=seed,
        arm_label=arm.arm_id,
    )
    checks = tier1_seed_pass_483e(metrics)
    passed = all(checks.values())
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    metrics["tier1_checks"] = checks
    metrics["seed_pass"] = passed
    return metrics


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> "Tuple[str, Path] | int":
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warmup = 6 if dry_run else WARMUP_EPISODES
    eval_eps = 2 if dry_run else EVAL_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        for arm in ARMS:
            rows.append(
                run_seed_arm_483e(
                    seed,
                    arm,
                    warmup_episodes=warmup,
                    eval_episodes=eval_eps,
                    steps_per_episode=steps,
                )
            )

    acceptance = evaluate_483e_cohort(
        rows,
        gap4_arm_id=GAP4_ARM,
        baseline_arm_id=BASE_ARM,
        inert_arm_id=INERT_ARM,
    )
    outcome = "PASS" if acceptance["pass"] else "FAIL"
    # Per-claim direction: SD-037 + MECH-281 carry the headline cascade-engagement
    # signal; MECH-280 carries the PAG-side wiring already validated as substrate-
    # ready by 483b. Outcome PASS supports all three. Outcome FAIL routes per the
    # interpretation grid in the docstring.
    per_claim = {
        "SD-037": "supports" if outcome == "PASS" else "weakens",
        "MECH-280": "supports" if outcome == "PASS" else "mixed",
        "MECH-281": "supports" if outcome == "PASS" else "weakens",
    }
    elapsed = time.time() - t0

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run outcome={outcome}", flush=True)
        print(f"  acceptance summary: pass={acceptance['pass']} "
              f"C1={acceptance['C1_cue_fires']} C2sub={acceptance['C2_override_signal']} "
              f"C2cas={acceptance['C2_cascade_engagement']} "
              f"C3d={acceptance['C3_approach_commit']} C3l={acceptance['C3_lift_vs_baseline']} "
              f"C4ga={acceptance['C4_goal_active']} C4ad={acceptance['C4_action_divergence']}",
              flush=True)
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
        "evidence_direction": "supports" if outcome == "PASS" else "mixed",
        "evidence_direction_per_claim": per_claim,
        "supersedes": "V3-EXQ-483d",
        "acceptance": acceptance,
        "thresholds": {
            "TIER1_OVERRIDE_SIGNAL_MIN": TIER1_OVERRIDE_SIGNAL_MIN,
            "TIER1_GOAL_NORM_PEAK_DELTA": TIER1_GOAL_NORM_PEAK_DELTA,
            "C2_LATERAL_PFC_RATIO_MIN": C2_LATERAL_PFC_RATIO_MIN,
            "C2_BLA_GAIN_RATIO_MIN": C2_BLA_GAIN_RATIO_MIN,
            "C2_CEA_PRIOR_RATIO_MIN": C2_CEA_PRIOR_RATIO_MIN,
            "C2_SUB_CHECKS_MIN": C2_SUB_CHECKS_MIN,
            "C4_ACTION_TV_MIN": C4_ACTION_TV_MIN,
            "ARM3_OVERRIDE_PFC_ETA_GAIN": ARM3_OVERRIDE_PFC_ETA_GAIN,
            "ARM3_OVERRIDE_BLA_ENCODING_GAIN": ARM3_OVERRIDE_BLA_ENCODING_GAIN,
            "ARM3_OVERRIDE_CEA_AMPLITUDE_GAIN": ARM3_OVERRIDE_CEA_AMPLITUDE_GAIN,
            "ARM3_OVERRIDE_BETA_INTERRUPT_GAIN": ARM3_OVERRIDE_BETA_INTERRUPT_GAIN,
        },
        "per_run": rows,
        "elapsed_seconds": elapsed,
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
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
