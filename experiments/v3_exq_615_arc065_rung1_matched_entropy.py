#!/opt/local/bin/python3
"""
V3-EXQ-615 -- ARC-065 Rung-1 validation with matched-entropy control.

Retest of ARC-065 (distributed diversity-generation pathway) after MECH-341
(E3 score diversity preservation, Layer B) substrate landed 2026-05-27. This
experiment adds the FP-2 matched-entropy control (ARM_1_MATCHED_NOISE) to
resolve the confound: does the full 4-substrate stack produce STRUCTURED
diversity above what entropy-matched noise alone would yield?

V3-EXQ-614a completed 2026-05-30 and showed MECH-341 is "load-bearing in
stack only" -- it works with other substrates (ARM_2_ALL_ON PASS) but not
in isolation (ARM_0_B_only FAIL). That result routes to THIS design: test
whether the full stack (A=SP-CEM, B=MECH-341, C=MECH-313, D=MECH-269) is
necessary for behavioral diversity AND produces structured diversity above
entropy-matched noise.

3-Arm Design
------------
ARM_0 (BASE_OFF): All diversity substrates OFF (SP-CEM off, MECH-341 off,
  noise_floor off, V_s minimal). Expect: monomodal collapse.
ARM_1 (MATCHED_NOISE): Entropy bonus only (via softmax temperature or noise
  injection), NO structured diversity substrates. This is the FP-2
  matched-entropy control.
ARM_2 (ALL_ON): Full 4-substrate stack (SP-CEM + MECH-341 both options +
  MECH-313 noise + MECH-269 V_s). Expect: Rung-1 PASS with structured
  diversity.

Environment: CausalGridWorldV2 bipartite reef (same as V3-EXQ-614a for
comparability):
  size=12, num_hazards=4, num_resources=5, reef_enabled=true,
  reef_bipartite_layout=true, n_reef_patches=3

Acceptance Criteria (pre-registered)
------------------------------------
C1 (Rung-1): ARM_2 (ALL_ON) achieves selected_action_entropy >0.3 AND
  n_unique_classes >=2 in >=2/3 seeds.
C2 (FP-2 control): ARM_2 entropy - ARM_1 entropy >= 0.15 (structured
  diversity > matched noise).
C3 (architectural necessity): ARM_0 (BASE_OFF) fails Rung-1 in >=2/3 seeds
  (diversity pathway necessary).

Overall outcome:
  PASS = all 3 criteria fire.
  FAIL otherwise.

Routing:
  PASS (all 3 criteria) -> ARC-065 supports, clear
    pending_retest_after_substrate flag, route to governance for
    provisional promotion consideration.
  FAIL C1 only -> ARC-065 weakens, route to diagnose-errors on full-stack
    integration.
  FAIL C2 only -> ARC-065 mixed (pathway works but not above
    entropy-matched control), route to Q-054 calibration.
  FAIL C3 only -> BASE_OFF unexpectedly passes (undermines necessity
    claim), route to failure-autopsy.

claim_ids: ["ARC-065"]
experiment_purpose: "evidence"

Phases
------
P0 (30 ep, instrumentation OFF): encoder warmup.
P1 (60 ep, instrumentation ON): Phase 2 eval. Measurement window long
  enough to distinguish structured diversity from noise floor.

Budget: 3 arms x 3 seeds x 90 ep x 200 steps = 162k steps total.
Estimated ~50-60 min on Mac (DLAPTOP-4.local @ ~14 steps/sec).

Implementation notes
--------------------
- env_kwargs IDENTICAL to V3-EXQ-614a -- same SD-054 reef-bipartite +
  hazard_food_attraction + harm_history substrate. Direct manifest
  comparability with the 614a cluster.
- Each arm is an independent REEAgent build (no in-process flag toggling
  mid-run); _make_agent reads the per-arm substrate-state dict to
  construct REEConfig.from_dims with the right per-axis kwargs.
- ARM_1 (MATCHED_NOISE) uses noise_floor=True with an elevated
  noise_floor_alpha to match the observed ARM_2 entropy level. This is a
  parametric control: we tune noise_floor_alpha in ARM_1 to approximate
  the entropy ARM_2 produces, then compare whether ARM_2's entropy is
  structured (higher n_unique_classes, lower collapse) vs ARM_1's
  unstructured noise.
- Per-tick measurement reuses the V3-EXQ-614a helpers verbatim
  (_per_class_score_stats, _entropy_from_counts) so the manifest's
  metric semantics are identical to the existing P2/P3 lineage.

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md,
REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md,
REE_assembly/docs/architecture/sd_054_reef_enrichment_substrate.md,
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md.
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_615_arc065_rung1_matched_entropy"
QUEUE_ID = "V3-EXQ-615"
CLAIM_IDS: List[str] = ["ARC-065"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 30
P1_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 1
DRY_RUN_P1 = 1
DRY_RUN_STEPS = 10

# Pre-registered behavioral thresholds (Rung-1 + FP-2).
RUNG1_ENTROPY_THRESHOLD = 0.3
RUNG1_MIN_CLASSES = 2
FP2_ENTROPY_DELTA = 0.15  # ARM_2 - ARM_1 must exceed this
MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3

# Pre-registered measurement gates (preserved from V3-EXQ-614a lineage so
# the P1 per-tick semantics remain comparable across the cluster).
PRE_GE2_FRAC_GATE = 0.5
SCORE_GAP_EPSILON_RANGE_FRAC = 0.05

# MECH-341 sub-flavour scale used in the entropy-ON arms. 2.0 was the
# upper-range value selected for the V3-EXQ-611b retune sweep target.
MECH341_ENTROPY_BIAS_SCALE = 2.0

# V_s (D) thresholds (minimal stack). 0.5 / 0.4 match the
# V3-EXQ-601 MECH-269b substrate-readiness PASS defaults.
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

# ARM_1 (MATCHED_NOISE) noise_floor_alpha calibration. This is a
# pre-registered parameter: we set noise_floor_alpha high enough to
# produce entropy ~= ARM_2's observed entropy, then test whether ARM_2's
# entropy is structured (higher n_unique_classes) vs ARM_1's unstructured
# noise. Start with 0.3 (3x the MECH-313 default 0.1); may need tuning.
MATCHED_NOISE_ALPHA = 0.3


# IDENTICAL to V3-EXQ-614a for direct manifest comparability across the
# cluster.
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


# ---------------------------------------------------------------------------
# Arm definitions: each arm is a full specification of the 4-substrate
# isolation axes + noise control.
# ---------------------------------------------------------------------------


ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_BASE_OFF",
        "label": "base_off_all_substrates_disabled",
        "substrate_axes": {
            "A_sp_cem": False,   # SP-CEM OFF (legacy collapsing CEM)
            "B_mech341": False,  # MECH-341 OFF
            "C_noise_floor": False,   # MECH-313 OFF
            "D_vs": False,       # V_s OFF
        },
    },
    {
        "arm_id": "ARM_1_MATCHED_NOISE",
        "label": "matched_noise_entropy_only",
        "substrate_axes": {
            "A_sp_cem": False,   # SP-CEM OFF
            "B_mech341": False,  # MECH-341 OFF
            "C_noise_floor": True,    # MECH-313 ON with elevated alpha
            "D_vs": False,       # V_s OFF
        },
    },
    {
        "arm_id": "ARM_2_ALL_ON",
        "label": "all_on_a_b_c_d",
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, axes: Dict[str, bool], arm_id: str) -> REEAgent:
    """Build a REEAgent with the 4-substrate state specified by ``axes``.

    Axis -> config flags mapping (read off ree-v3/CLAUDE.md SD Design
    Decisions Implemented sections + ree-v3/ree_core/utils/config.py):

      A_sp_cem (SP-CEM main-path):
        use_support_preserving_cem
        support_preserving_stratified_elites
        support_preserving_ao_std_floor (0.2 when ON, 0.0 when OFF)
      B_mech341 (E3 score diversity preservation):
        use_e3_score_diversity (master)
        use_e3_diversity_entropy_bonus + use_e3_diversity_stratified_select
          (both ON when B ON; both irrelevant when master OFF)
        e3_diversity_entropy_bias_scale = MECH341_ENTROPY_BIAS_SCALE
      C_noise_floor (MECH-313 tonic noise floor):
        use_noise_floor + noise_floor_alpha (default 0.1; ARM_1 uses
          MATCHED_NOISE_ALPHA)
      D_vs (minimal V_s pathology stack per user-confirmed scope):
        use_per_stream_vs + use_vs_rollout_gating
        vs_gate_snapshot_refresh_threshold (0.5) + vs_gate_e1_threshold (0.4)
        NO anchor_sets, NO staleness_accumulator, NO event_segmenter,
        NO invalidation_trigger. The gate hold-substitutes a fresh
        snapshot at the E1 rollout call site when V_s falls below
        the e1 threshold (MECH-269b minimal behavioural mechanism).
    """
    a_on = bool(axes["A_sp_cem"])
    b_on = bool(axes["B_mech341"])
    c_on = bool(axes["C_noise_floor"])
    d_on = bool(axes["D_vs"])

    # ARM_1 (MATCHED_NOISE) uses elevated noise_floor_alpha to match
    # entropy; other arms use default 0.1.
    noise_alpha = MATCHED_NOISE_ALPHA if arm_id == "ARM_1_MATCHED_NOISE" else 0.1

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        # A (SP-CEM)
        use_support_preserving_cem=a_on,
        support_preserving_stratified_elites=a_on,
        support_preserving_ao_std_floor=(0.2 if a_on else 0.0),
        support_preserving_min_first_action_classes=2,
        # B (MECH-341)
        use_e3_score_diversity=b_on,
        use_e3_diversity_entropy_bonus=True,    # consulted only when master ON
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        # C (MECH-313)
        use_noise_floor=c_on,
        noise_floor_alpha=noise_alpha,
        # D (V_s minimal stack)
        use_per_stream_vs=d_on,
        use_vs_rollout_gating=d_on,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # NO anchor sets, NO staleness accumulator, NO event segmenter
        use_anchor_sets=False,
        use_staleness_accumulator=False,
        use_event_segmenter=False,
    )
    agent = REEAgent(cfg)
    agent.to(torch.device("cpu"))
    return agent


def _per_class_score_stats(
    scores: torch.Tensor, selected_action: torch.Tensor, action_dim: int
) -> Dict[str, float]:
    """Per-class score statistics (mean/std/min/max) over the selected
    action's class. Returns 4 scalars.
    """
    sel_idx = int(selected_action.item())
    class_scores = scores[sel_idx]
    return {
        "mean": float(class_scores.mean().item()),
        "std": float(class_scores.std().item()),
        "min": float(class_scores.min().item()),
        "max": float(class_scores.max().item()),
    }


def _entropy_from_counts(counts: torch.Tensor, total: int) -> float:
    """Shannon entropy in nats from class counts."""
    if total == 0:
        return 0.0
    probs = counts.float() / float(total)
    probs = probs[probs > 0]
    return float(-(probs * torch.log(probs)).sum().item())


def run_arm(
    arm: Dict[str, Any],
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one arm (all seeds) and return aggregate metrics."""
    arm_id = arm["arm_id"]
    axes = arm["substrate_axes"]
    label = arm["label"]

    seed_results = []
    for seed in seeds:
        print(f"Seed {seed} Condition {arm_id}")
        env = _make_env(seed)
        agent = _make_agent(env, axes, arm_id)

        torch.manual_seed(seed)

        # P0: encoder warmup (no instrumentation)
        for ep in range(p0_episodes):
            if (ep + 1) % 10 == 0 or ep == p0_episodes - 1:
                print(f"  [P0 warmup] seed={seed} arm={arm_id} ep {ep+1}/{p0_episodes}")
            _, obs_dict = env.reset()
            agent.reset()
            for step in range(steps_per_episode):
                body = obs_dict["body_state"].float()
                world = obs_dict["world_state"].float()
                if body.dim() == 1:
                    body = body.unsqueeze(0)
                if world.dim() == 1:
                    world = world.unsqueeze(0)

                # Minimal helper to get harm obs
                obs_harm = obs_dict.get("harm_obs")
                obs_harm = obs_harm.float().unsqueeze(0) if obs_harm is not None else None
                obs_harm_a = obs_dict.get("harm_obs_a")
                obs_harm_a = obs_harm_a.float().unsqueeze(0) if obs_harm_a is not None else None
                obs_harm_history = obs_dict.get("harm_history")
                obs_harm_history = obs_harm_history.float().unsqueeze(0) if obs_harm_history is not None else None

                latent = agent.sense(obs_body=body, obs_world=world,
                                   obs_harm=obs_harm, obs_harm_a=obs_harm_a,
                                   obs_harm_history=obs_harm_history)
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                           else torch.zeros(1, wdim, device=agent.device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

                _, harm_signal, done, info, obs_dict = env.step(action)
                if done:
                    break

        # P1: measurement window
        selected_classes = torch.zeros(env.action_dim, dtype=torch.long)
        pre_ge2_ticks = 0
        total_ticks = 0

        for ep in range(p1_episodes):
            if (ep + 1) % 10 == 0 or ep == p1_episodes - 1:
                print(f"  [P1 eval] seed={seed} arm={arm_id} ep {ep+1}/{p1_episodes}")
            _, obs_dict = env.reset()
            agent.reset()
            for step in range(steps_per_episode):
                body = obs_dict["body_state"].float()
                world = obs_dict["world_state"].float()
                if body.dim() == 1:
                    body = body.unsqueeze(0)
                if world.dim() == 1:
                    world = world.unsqueeze(0)

                obs_harm = obs_dict.get("harm_obs")
                obs_harm = obs_harm.float().unsqueeze(0) if obs_harm is not None else None
                obs_harm_a = obs_dict.get("harm_obs_a")
                obs_harm_a = obs_harm_a.float().unsqueeze(0) if obs_harm_a is not None else None
                obs_harm_history = obs_dict.get("harm_history")
                obs_harm_history = obs_harm_history.float().unsqueeze(0) if obs_harm_history is not None else None

                latent = agent.sense(obs_body=body, obs_world=world,
                                   obs_harm=obs_harm, obs_harm_a=obs_harm_a,
                                   obs_harm_history=obs_harm_history)
                ticks = agent.clock.advance()
                wdim = latent.z_world.shape[-1]
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                           else torch.zeros(1, wdim, device=agent.device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                # Instrument: capture E3 scores before selection (P1 only)
                if candidates:
                    last_scores = getattr(agent.e3, "last_scores", None)
                    if last_scores is not None:
                        n_pre = int((last_scores >= 0.0).sum().item())
                        if n_pre >= 2:
                            pre_ge2_ticks += 1

                        score_range = float(last_scores.max().item() - last_scores.min().item())
                        epsilon = SCORE_GAP_EPSILON_RANGE_FRAC * score_range
                        if score_range > epsilon:
                            total_ticks += 1

                action = agent.select_action(candidates, ticks)
                selected_classes[int(action.argmax().item())] += 1

                _, harm_signal, done, info, obs_dict = env.step(action)
                if done:
                    break

        # Per-seed metrics
        n_unique = int((selected_classes > 0).sum().item())
        total_actions = int(selected_classes.sum().item())
        entropy_nats = _entropy_from_counts(selected_classes, total_actions)
        frac_pre_ge2 = float(pre_ge2_ticks) / max(1, total_ticks)

        seed_results.append({
            "seed": seed,
            "n_unique_classes": n_unique,
            "selected_class_entropy_nats": entropy_nats,
            "frac_pre_ge2": frac_pre_ge2,
            "total_actions": total_actions,
        })

        verdict = "PASS" if (n_unique >= RUNG1_MIN_CLASSES and entropy_nats > RUNG1_ENTROPY_THRESHOLD) else "FAIL"
        print(f"verdict: {verdict}")

    # Aggregate across seeds
    mean_entropy = sum(r["selected_class_entropy_nats"] for r in seed_results) / len(seed_results)
    mean_unique = sum(r["n_unique_classes"] for r in seed_results) / len(seed_results)
    n_seeds_pass_rung1 = sum(
        1 for r in seed_results
        if r["n_unique_classes"] >= RUNG1_MIN_CLASSES and r["selected_class_entropy_nats"] > RUNG1_ENTROPY_THRESHOLD
    )

    return {
        "arm_id": arm_id,
        "label": label,
        "substrate_axes": axes,
        "seed_results": seed_results,
        "mean_selected_class_entropy_nats": mean_entropy,
        "mean_n_unique_classes": mean_unique,
        "n_seeds_pass_rung1": n_seeds_pass_rung1,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    """Run the full 3-arm experiment."""
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results = []
    for arm in ARMS:
        result = run_arm(arm, seeds, p0, p1, steps, dry_run)
        arm_results.append(result)

    # Extract per-arm means for criteria
    arm0 = arm_results[0]  # BASE_OFF
    arm1 = arm_results[1]  # MATCHED_NOISE
    arm2 = arm_results[2]  # ALL_ON

    # C1: ARM_2 (ALL_ON) Rung-1 PASS in >= 2/3 seeds
    c1_pass = arm2["n_seeds_pass_rung1"] >= MIN_SEEDS_PER_ARM_FOR_PASS

    # C2: FP-2 control: ARM_2 entropy - ARM_1 entropy >= 0.15
    entropy_delta = arm2["mean_selected_class_entropy_nats"] - arm1["mean_selected_class_entropy_nats"]
    c2_pass = entropy_delta >= FP2_ENTROPY_DELTA

    # C3: ARM_0 (BASE_OFF) FAIL Rung-1 in >= 2/3 seeds (necessity)
    c3_pass = arm0["n_seeds_pass_rung1"] < MIN_SEEDS_PER_ARM_FOR_PASS

    overall_pass = c1_pass and c2_pass and c3_pass
    outcome = "PASS" if overall_pass else "FAIL"

    # Routing
    if overall_pass:
        routing = "ARC-065 supports, clear pending_retest_after_substrate flag, route to governance for provisional promotion consideration"
        evidence_direction = "supports"
    elif not c1_pass:
        routing = "ARC-065 weakens, route to diagnose-errors on full-stack integration"
        evidence_direction = "weakens"
    elif not c2_pass:
        routing = "ARC-065 mixed (pathway works but not above entropy-matched control), route to Q-054 calibration"
        evidence_direction = "mixed"
    elif not c3_pass:
        routing = "BASE_OFF unexpectedly passes (undermines necessity claim), route to failure-autopsy"
        evidence_direction = "mixed"
    else:
        routing = "Unknown failure mode"
        evidence_direction = "unknown"

    return {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "arm_results": arm_results,
        "criteria": {
            "c1_arm2_rung1_pass": c1_pass,
            "c2_fp2_entropy_delta": c2_pass,
            "c3_arm0_necessity_fail": c3_pass,
        },
        "metrics": {
            "arm0_base_off_entropy": arm0["mean_selected_class_entropy_nats"],
            "arm1_matched_noise_entropy": arm1["mean_selected_class_entropy_nats"],
            "arm2_all_on_entropy": arm2["mean_selected_class_entropy_nats"],
            "entropy_delta_arm2_vs_arm1": entropy_delta,
            "arm0_n_seeds_pass_rung1": arm0["n_seeds_pass_rung1"],
            "arm1_n_seeds_pass_rung1": arm1["n_seeds_pass_rung1"],
            "arm2_n_seeds_pass_rung1": arm2["n_seeds_pass_rung1"],
        },
        "routing": routing,
        "seeds": seeds,
        "p0_warmup_episodes": p0,
        "p1_measurement_episodes": p1,
        "steps_per_episode": steps,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal episodes/steps")
    args = parser.parse_args()

    print(f"Starting {EXPERIMENT_TYPE}")
    print(f"Queue ID: {QUEUE_ID}")
    print(f"Claim IDs: {CLAIM_IDS}")
    print(f"Dry run: {args.dry_run}")

    result = run_experiment(dry_run=args.dry_run)

    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=False,
        config=result.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"\nResults written to: {out_path}")
    print(f"Outcome: {result['outcome']}")
    print(f"Evidence direction: {result['evidence_direction']}")
    print(f"Routing: {result['routing']}")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
    )
