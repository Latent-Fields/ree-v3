#!/opt/local/bin/python3
"""
V3-EXQ-616 -- Q-054 entropy_bias_scale sweep on MECH-341 B_only isolation arm.

ROUTING ANCHOR
--------------
Triggered by V3-EXQ-614a (2026-05-30 PASS, interpretation_label
PASS_C2_C3_only_mech341_load_bearing_in_stack_only). 614a interpretation
grid PASS_via_C2+C3 row routes to:

    "MECH-341 contributes to the full stack but not sufficient in
     isolation. ARC-065 supports (distributed pathway necessary).
     Route to /governance with MECH-341 supports + Q-054
     entropy_bias_scale calibration sweep before Phase P4."

614a ARM_0_B_only failed C1 (R2.c isolation) at entropy_bias_scale=2.0
(the 614a default). The 2026-05-31T08:45Z governance walk applied
evidence_direction_per_claim[MECH-341]=supports + [ARC-065]=supports
and routed Q-054 entropy_bias_scale sweep here per the
behavioral_diversity_isolation_plan.md R2.c rule.

Q-054 ANCHOR (claims.yaml registered_utc 2026-05-25)
----------------------------------------------------
"What is the minimum trajectory-class diversity floor (Rung 1
first_action_entropy threshold) required for the ARC-062 context
discriminator to learn a reliable discriminative cut?"

Per the governance routing, the Q-054 question is operationalised here
as: "At what entropy_bias_scale value (the MECH-341 sub-flavour scale
knob) does B_only isolation produce Rung-1 selected-action-class
diversity sufficient to clear R2.c (>=2 classes AND entropy>0.3 nats
on >=2/3 seeds)?". The lowest passing scale identifies the load-bearing
range of MECH-341 in isolation; FAIL across all scales surfaces a
substrate-ceiling result that routes to alternative substrate paths
(stratified_temperature, A-vs-B redundancy, V_s contribution).

SLEEP DRIVER: K=never (SleepLoopManager disabled during training; sleep
NOT called -- this script does NOT use sleep, listed here for the
GAP-7 standardisation grep). use_sleep_loop is OFF.

Design (single-axis sweep, B_only fixed)
----------------------------------------
4 arms x 3 seeds; only entropy_bias_scale varies; all other substrate
flags + env config IDENTICAL to V3-EXQ-614a ARM_0_B_only:

  ARM_0_S1      e3_diversity_entropy_bias_scale=1.0
  ARM_1_S2      e3_diversity_entropy_bias_scale=2.0   (614a baseline)
  ARM_2_S4      e3_diversity_entropy_bias_scale=4.0
  ARM_3_S8      e3_diversity_entropy_bias_scale=8.0

All arms: SP-CEM OFF, MECH-341 ON (both sub-flavours), MECH-313 OFF,
V_s OFF -- direct comparability to 614a ARM_0_B_only.

Per-arm acceptance (mirrors 614a C1 R2.c rule)
----------------------------------------------
A seed passes Rung-1 when:
  n_unique_selected_classes >= 2
  AND selected_class_entropy_nats > 0.3
  AND frac_pre_ge2 >= 0.5

An arm passes when >=2 of 3 seeds clear Rung-1.

Cross-arm outcome (Q-054 answer)
--------------------------------
The PASSING scale floor is the lowest scale value where the arm passes.
The interpretation grid maps the floor to a routing decision:

  PASS_at_floor_1.0:
    MECH-341 strong at baseline scale; 614a C1 FAIL is a seed-variance
    or measurement artifact, not substrate insufficiency. Route to
    /failure-autopsy on V3-EXQ-614a ARM_0_B_only to discriminate.

  PASS_at_floor_2.0:
    Replicates 614a baseline scale; the C1 FAIL on 614a was a single
    -draw seed-variance event. Route to /governance with MECH-341 in-
    isolation sufficient at default scale; note 614a discrepancy.

  PASS_at_floor_4.0_or_8.0:
    Scale lever works; in-isolation floor is above the 614a baseline.
    Q-054 supports (specific load-bearing floor identified). Route to
    /governance with MECH-341 supports + recommend raising default
    e3_diversity_entropy_bias_scale to >=4.0 (config-default change is
    a separate /implement-substrate decision, NOT applied here).

  FAIL_no_floor_under_8.0:
    Scale lever insufficient in isolation. MECH-341 contributes to the
    full stack (614a C2+C3 PASS) but cannot become sufficient in
    isolation under scale lever alone. Q-054 weakens for the scale
    axis specifically; route to substrate revisit (Option-2
    stratified_temperature default + A-vs-B partial-redundancy probe).

Per-claim direction
-------------------
  Q-054 (the question being answered):
    PASS at any floor -> supports (floor identified, question answered)
    FAIL              -> mixed (scale lever ruled out; further sweeps
                          required to answer the broader Q-054)
  MECH-341 (the substrate under calibration):
    PASS_at_floor in {1.0, 2.0} -> supports (in-isolation sufficient
                                    at baseline-or-below scale)
    PASS_at_floor in {4.0, 8.0} -> mixed (sufficient only above default;
                                    default config too low)
    FAIL                         -> mixed (in-isolation sufficiency not
                                    achievable via scale lever; full
                                    stack contribution preserved from
                                    614a C2+C3 PASS)

Phased training
---------------
P0 (30 ep, instrumentation OFF): encoder warmup. Matches 614a budget.
P1 (60 ep, instrumentation ON): behavioural measurement window.

Budget: 4 arms x 3 seeds x 90 ep x 200 steps = 216k steps.
Estimated ~80-90 min on Mac (DLAPTOP-4.local @ ~45 steps/sec from 614a
calibration of 162k steps in ~60 min).

Implementation notes
--------------------
- env_kwargs IDENTICAL to V3-EXQ-611/611b/611c/614a -- SD-054
  bipartite reef + hazard_food_attraction + harm_history substrate.
- B_only substrate config IDENTICAL to 614a ARM_0_B_only except for
  the entropy_bias_scale sweep axis.
- Per-tick measurement helpers verbatim from 614a so manifest metric
  semantics are identical across the cluster.

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md,
REE_assembly/docs/architecture/sd_054_reef_enrichment_substrate.md,
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md,
REE_assembly/docs/claims/claims.yaml (Q-054, MECH-341, ARC-065).
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


EXPERIMENT_TYPE = "v3_exq_616_q054_mech341_entropy_bias_scale_sweep"
QUEUE_ID = "V3-EXQ-616"
CLAIM_IDS: List[str] = ["Q-054", "MECH-341"]
EXPERIMENT_PURPOSE = "evidence"
SLEEP_DRIVER_PATTERN = "K=never (SleepLoopManager disabled during training; sleep NOT called)"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 30
P1_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Pre-registered behavioural thresholds (R2.c + Rung 1; identical to 614a).
RUNG1_ENTROPY_THRESHOLD = 0.3
RUNG1_MIN_CLASSES = 2
MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3
PRE_GE2_FRAC_GATE = 0.5


# Sweep axis: entropy_bias_scale values. ARM_1_S2 replicates the 614a
# ARM_0_B_only baseline scale so the cluster manifest is directly
# comparable; ARM_0/2/3 extend the sweep below and above.
SWEEP_SCALES: List[float] = [1.0, 2.0, 4.0, 8.0]


# IDENTICAL to V3-EXQ-611 / 611b / 611c / 614a for direct manifest
# comparability across the cluster.
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


def _arm_label(scale: float) -> str:
    # Render scale like S1 / S2 / S4 / S8 / S1_5 for the human label.
    if float(scale).is_integer():
        return f"S{int(scale)}"
    return "S" + str(scale).replace(".", "_")


def _build_arms(scales: List[float]) -> List[Dict[str, Any]]:
    return [
        {
            "arm_id": f"ARM_{i}_B_only_{_arm_label(s)}",
            "label": f"B_only_entropy_bias_scale_{s}",
            "entropy_bias_scale": float(s),
        }
        for i, s in enumerate(scales)
    ]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, entropy_bias_scale: float) -> REEAgent:
    """Build a REEAgent in B_only configuration with the swept
    entropy_bias_scale.

    Substrate-axis assignment identical to V3-EXQ-614a ARM_0_B_only:
      A_sp_cem OFF, B_mech341 ON (both sub-flavours), C_noise_floor OFF,
      D_vs OFF. Only the entropy_bias_scale knob varies across arms.
    """
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
        # A (SP-CEM) OFF for B_only isolation
        use_support_preserving_cem=False,
        support_preserving_stratified_elites=False,
        support_preserving_ao_std_floor=0.0,
        support_preserving_min_first_action_classes=2,
        # B (MECH-341) ON -- swept knob
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=float(entropy_bias_scale),
        # C (MECH-313) OFF
        use_noise_floor=False,
        # D (V_s) OFF
        use_per_stream_vs=False,
        use_vs_rollout_gating=False,
    )
    return REEAgent(cfg)


def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _per_class_score_stats(
    candidates: List, last_scores: Optional[torch.Tensor]
) -> Tuple[Dict[int, Dict[str, float]], Optional[int], Optional[float], bool]:
    if (
        not candidates
        or len(candidates) < 2
        or last_scores is None
        or last_scores.numel() != len(candidates)
    ):
        return {}, None, None, False

    scores_t = last_scores.detach().reshape(-1).float()
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

    sel_idx = int(scores_t.argmin().item())
    selected_class = int(classes_per_cand[sel_idx])

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


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, arm["entropy_bias_scale"])
    agent.eval()

    total_train_eps = p0_episodes + p1_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p1_logged = 0
    n_p1_pre_ge2 = 0
    n_p1_pre_eq1 = 0
    selected_classes_p1: Dict[int, int] = {}
    committed_classes_p1: Dict[int, int] = {}
    top2_gaps_pre_ge2: List[float] = []

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

            pre_e3_classes: List[int] = []
            if is_p1 and candidates:
                pre_e3_classes = sorted({
                    _trajectory_first_action_class(t) for t in candidates
                })

            action = agent.select_action(candidates, ticks)
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            if is_p1:
                last_scores = getattr(agent.e3, "last_scores", None)
                per_class, sel_class, top2_gap, logged = _per_class_score_stats(
                    candidates, last_scores
                )
                committed_class = int(action[0].argmax().item())

                pre_count = len(pre_e3_classes)
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
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            _, _harm_signal, done, info, obs_dict = env.step(action)

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

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    if n_p1_ticks > 0:
        frac_pre_ge2 = float(n_p1_pre_ge2 / n_p1_ticks)
    else:
        frac_pre_ge2 = 0.0

    if top2_gaps_pre_ge2:
        mean_top2_gap = float(sum(top2_gaps_pre_ge2) / len(top2_gaps_pre_ge2))
    else:
        mean_top2_gap = None

    n_selected_classes = len(selected_classes_p1)
    selected_class_entropy = _entropy_from_counts(selected_classes_p1)
    committed_class_entropy = _entropy_from_counts(committed_classes_p1)

    seed_passes_rung1 = bool(
        n_selected_classes >= RUNG1_MIN_CLASSES
        and selected_class_entropy > RUNG1_ENTROPY_THRESHOLD
        and frac_pre_ge2 >= PRE_GE2_FRAC_GATE
    )

    return {
        "arm_id": arm["arm_id"],
        "entropy_bias_scale": float(arm["entropy_bias_scale"]),
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
        "selected_classes_p1_counts": {
            str(k): int(v) for k, v in sorted(selected_classes_p1.items())
        },
        "committed_classes_p1_counts": {
            str(k): int(v) for k, v in sorted(committed_classes_p1.items())
        },
        "n_unique_selected_classes": int(n_selected_classes),
        "selected_class_entropy_nats": round(selected_class_entropy, 6),
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "seed_passes_rung1": seed_passes_rung1,
        "error_note": error_note,
    }


def _interpret_arm(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_seeds_completed = sum(1 for r in seed_rows if r["error_note"] is None)
    n_seeds_rung1_pass = sum(
        1 for r in seed_rows if r["error_note"] is None and r["seed_passes_rung1"]
    )
    selected_entropies = [
        r["selected_class_entropy_nats"]
        for r in seed_rows if r["error_note"] is None
    ]
    mean_selected_entropy = (
        sum(selected_entropies) / len(selected_entropies)
        if selected_entropies else 0.0
    )
    n_unique_class_per_seed = [
        r["n_unique_selected_classes"]
        for r in seed_rows if r["error_note"] is None
    ]
    mean_n_unique = (
        sum(n_unique_class_per_seed) / len(n_unique_class_per_seed)
        if n_unique_class_per_seed else 0.0
    )

    return {
        "n_seeds_completed": int(n_seeds_completed),
        "n_seeds_rung1_pass": int(n_seeds_rung1_pass),
        "majority_rung1_pass": bool(
            n_seeds_rung1_pass >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "mean_selected_class_entropy_nats": round(mean_selected_entropy, 6),
        "mean_n_unique_selected_classes": round(mean_n_unique, 6),
    }


def _classify_outcome(
    arms_out: List[Dict[str, Any]]
) -> Tuple[str, str, str, str, str, Optional[float]]:
    """Identify the lowest passing scale (if any) and map to outcome,
    overall direction, per-claim directions, interpretation label, and
    the floor scale value."""
    passing_arms = [
        a for a in arms_out
        if a["cross_seed_interpretation"]["majority_rung1_pass"]
    ]
    if not passing_arms:
        return (
            "FAIL", "weakens", "mixed", "mixed",
            "FAIL_no_floor_under_max_swept_scale",
            None,
        )

    # Pick the LOWEST scale among passing arms -> floor of the load-bearing range
    passing_arms.sort(key=lambda a: a["entropy_bias_scale"])
    floor_scale = float(passing_arms[0]["entropy_bias_scale"])

    if floor_scale <= 1.0 + 1e-9:
        return (
            "PASS", "supports", "supports", "supports",
            "PASS_at_floor_1p0_baseline_or_below",
            floor_scale,
        )
    if floor_scale <= 2.0 + 1e-9:
        return (
            "PASS", "supports", "supports", "supports",
            "PASS_at_floor_2p0_replicates_614a_baseline_scale",
            floor_scale,
        )
    # 4.0 or 8.0
    return (
        "PASS", "mixed", "supports", "mixed",
        f"PASS_at_floor_{floor_scale}_above_default_scale",
        floor_scale,
    )


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
    sweep_scales: List[float],
) -> Dict[str, Any]:
    arms = _build_arms(sweep_scales)
    arms_out: List[Dict[str, Any]] = []
    for arm in arms:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) "
            f"(scale={arm['entropy_bias_scale']}, P0={p0_episodes} ep, "
            f"P1={p1_episodes} ep, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        seed_rows: List[Dict[str, Any]] = []
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(arm, s, p0_episodes, p1_episodes, steps_per_episode)
            seed_rows.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)
        cross = _interpret_arm(seed_rows)
        arms_out.append({
            "arm_id": arm["arm_id"],
            "label": arm["label"],
            "entropy_bias_scale": float(arm["entropy_bias_scale"]),
            "per_seed_results": seed_rows,
            "cross_seed_interpretation": cross,
        })

    (
        outcome_label, overall_direction,
        q054_direction, mech341_direction,
        interpretation_label, floor_scale,
    ) = _classify_outcome(arms_out)

    total_seeds = len(arms) * len(seeds)
    total_completed = sum(
        a["cross_seed_interpretation"]["n_seeds_completed"] for a in arms_out
    )

    return {
        "outcome": outcome_label,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": {
            "Q-054": q054_direction,
            "MECH-341": mech341_direction,
        },
        "interpretation_label": interpretation_label,
        "load_bearing_floor_scale": floor_scale,
        "seeds": seeds,
        "sweep_scales": sweep_scales,
        "n_arms": len(arms_out),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "rung1_entropy_threshold": float(RUNG1_ENTROPY_THRESHOLD),
            "rung1_min_classes": int(RUNG1_MIN_CLASSES),
            "min_seeds_per_arm_for_pass": int(MIN_SEEDS_PER_ARM_FOR_PASS),
            "pre_ge2_frac_gate": float(PRE_GE2_FRAC_GATE),
        },
        "acceptance_grid": {
            "sweep_scales": sweep_scales,
            "per_arm_majority_pass": {
                a["arm_id"]: bool(
                    a["cross_seed_interpretation"]["majority_rung1_pass"]
                )
                for a in arms_out
            },
            "load_bearing_floor_scale": floor_scale,
        },
        "interpretation_grid": {
            "PASS_at_floor_1p0": (
                "MECH-341 strong at baseline scale; 614a ARM_0_B_only C1 "
                "FAIL is a seed-variance artifact. Route to /failure-autopsy "
                "on V3-EXQ-614a ARM_0_B_only."
            ),
            "PASS_at_floor_2p0": (
                "Replicates 614a baseline scale; 614a C1 FAIL was a single "
                "draw seed-variance event. Route to /governance with "
                "MECH-341 in-isolation sufficient at default scale."
            ),
            "PASS_at_floor_4p0_or_8p0": (
                "Scale lever works; in-isolation floor is above the 614a "
                "baseline. Q-054 supports (specific load-bearing floor "
                "identified). Route to /governance + recommend raising "
                "e3_diversity_entropy_bias_scale default. Default-change is "
                "a separate /implement-substrate decision."
            ),
            "FAIL_no_floor_under_max": (
                "Scale lever insufficient in isolation. MECH-341 in-stack "
                "contribution preserved per 614a C2+C3 PASS. Q-054 mixed "
                "for the scale axis; route to substrate revisit "
                "(stratified_temperature default, A-vs-B redundancy "
                "probe)."
            ),
        },
        "arms": arms_out,
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
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "evidence_direction_note": (
            f"experiment_purpose=evidence; Q-054 entropy_bias_scale "
            f"sweep on MECH-341 B_only isolation arm "
            f"(routed from V3-EXQ-614a PASS_via_C2_C3). "
            f"interpretation_label={result['interpretation_label']}. "
            f"load_bearing_floor_scale={result['load_bearing_floor_scale']}. "
            f"Per-arm acceptance grid in result.acceptance_grid."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "isolation_axis": "B_only (MECH-341 only; SP-CEM + MECH-313 + V_s OFF)",
            "swept_knob": "e3_diversity_entropy_bias_scale",
            "sweep_scales": result["sweep_scales"],
            "mech341_sub_flavours": "both (entropy_bonus + stratified_select)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
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
        sweep_scales=list(SWEEP_SCALES),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

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
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"floor_scale={result['load_bearing_floor_scale']} "
        f"label={result['interpretation_label']}",
        flush=True,
    )

    if args.dry_run:
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
