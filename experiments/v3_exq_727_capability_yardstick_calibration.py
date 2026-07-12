#!/opt/local/bin/python3
"""
V3-EXQ-727 -- CAPABILITY YARDSTICK CALIBRATION (WS-3 capability-eval suite reference run).

WHY THIS EXISTS (a BASELINE reference run; experiment_purpose="baseline", claim_ids=[];
PROMOTES / DEMOTES NOTHING; EXCLUDED from governance scoring). WS-3 of
REE_assembly/evidence/planning/ree_ai_design_critique_plan.md asks for a minimal
capability-eval suite that is INDEPENDENT of any REE mechanism claim, so the
conversion-ceiling campaign can separate "the structural claim is wrong" from "the
substrate is too coarse to carry the signal." The integrated all-ON agent forages
0.065/0.0/0.455 resources/episode (below the 1.0 floor, 0/3 seeds;
failure_autopsy_V3-EXQ-719a_2026-07-08), so structural claims are being measured on an
agent that cannot act.

This run does two things:
  1. Exercises the reusable, claim-agnostic yardstick module
     experiments/_lib/capability_eval.py end-to-end on the real V3 substrate (proving it
     attaches to REEAgent's forward-eval path), and
  2. CALIBRATES the four capability metrics' scale by measuring three reference policies
     under the identical env/seed/protocol:
       - random_walk        (FLOOR anchor; uniform-random actions)
       - ree_p0warmup_allon (the all-ON REE stack after the standard P0 world-model warmup,
                             NO P1 competence-training -- an honest "substrate as-built,
                             pre-competence-training" point measured on the yardstick)
       - greedy_oracle      (CEILING / achievability anchor; nearest-resource forager,
                             reused verbatim from V3-EXQ-724's positive control)

The four claim-agnostic metrics (defined + measured entirely in _lib/capability_eval.py):
  foraging_competence  -- mean resources/episode (env transition_type == "resource")
  survival_horizon     -- mean ticks survived/episode (+ death_rate on agent_health<=0)
  goal_reach_rate      -- fraction of episodes collecting >= 1 resource
  planning_depth       -- mean longest strictly-decreasing-nearest-resource-distance run

DENOMINATOR ROLE. The report normalizes each policy to [random_floor, oracle_ceiling] per
metric, so any future all-ON experiment that imports the module can state "structure X moved
capability metric Y by Z on a substrate already above the competence floor." The TRAINED
all-ON foraging denominator itself is produced by the in-flight V3-EXQ-724 (P1=90 A0 arm);
this run does NOT re-train that (it would duplicate 724's expensive competence measurement).
Its job is the reusable yardstick + the full-four-metric floor/ceiling scale + REEAgent
integration. A follow-on should import capability_eval into the next TRAINED all-ON run so
the trained point lands on all four metrics against this calibrated scale.

SELF-ROUTE (BASELINE; a HYPOTHESIS, not a verdict):
  * READINESS: the greedy oracle must clear COMPETENCE_RESOURCE_FLOOR (=1.0 resource/ep) on
    the ceiling anchor -- proving the floor is ACHIEVABLE in this env. If it does not, the
    yardstick is not calibratable here -> label `substrate_not_ready_requeue` (draw NO
    conclusion). Additionally the oracle ceiling must exceed the random floor on foraging
    (yardstick_discriminates) for the scale to be non-degenerate.
  * PASS: readiness holds AND the yardstick discriminates -> label
    `capability_yardstick_calibrated`. The suite is validated + the scale is established.

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic/baseline-description requirement):
  * EVIDENCE the yardstick is usable: oracle clears the floor AND oracle > random on
    foraging -> the four metrics have a meaningful, discriminating scale on this substrate.
  * EVIDENCE AGAINST (substrate_not_ready_requeue): the oracle itself cannot clear the floor
    (env too sparse/lethal for the floor to be achievable) OR the anchors do not separate.
  This run tags NO claim; the label is a hypothesis for adjudication, never a governance act.

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Sourced config (all-ON matched stack) from V3-EXQ-714 ARM_ON via V3-EXQ-719a / 724:
  experiments/v3_exq_724_competence_localization_diagnostic.py (all-ON config + oracle),
  ree_core/environment/causal_grid_world.py, ree_core/agent.py, ree_core/utils/config.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    METRIC_KEYS,
    OraclePolicy,
    RandomPolicy,
    REEForwardPolicy,
    build_report,
    evaluate_seed,
    summarize_arm,
)
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_727_capability_yardstick_calibration"
QUEUE_ID = "V3-EXQ-727"
CLAIM_IDS: List[str] = []                 # tags NO claim -- pure capability yardstick
EXPERIMENT_PURPOSE = "baseline"

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 120          # world-model (encoder/e2) warmup for the REE arm
EVAL_EPISODES = 20                # capability-eval episodes per (arm, seed) cell
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42, 43]
DRY_RUN_P0 = 3
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 30

# ---------------------------------------------------------------------------
# SD-056 online e2 world-forward contrastive (mirror V3-EXQ-724 P0 warmup)
# ---------------------------------------------------------------------------
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
SD056_MULTISTEP_CONTRASTIVE = True
SD056_CONTRASTIVE_HORIZON = 5
SD056_OUTPUT_NORM_CLAMP = True
SD056_OUTPUT_NORM_CLAMP_RATIO = 2.0

# All-ON matched-stack constants (sourced from V3-EXQ-714 ARM_ON via 724).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30
F_ELIGIBILITY_DN_SIGMA = 0.0
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0
GNG_PERSEVERATION_FLOOR = 0.5
GNG_SAFETY_FLOOR = 0.5
GNG_PROTECT_MIN_ELIGIBLE = 1
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4
OFC_STATE_DIM = 16
OFC_HARM_DIM = 32
OFC_BIAS_SCALE = 0.5
OFC_DEVAL_BIAS_SCALE = 2.0
GNG_VIABILITY_FLOOR = 0.1


# Identical env to V3-EXQ-714 / 719a / 724 (SD-054 reef + hazard_food_attraction + bipartite).
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


# ---------------------------------------------------------------------------
# All-ON agent config (719a ARM_ON via 724).
# ---------------------------------------------------------------------------
def _base_config_kwargs(env: CausalGridWorldV2) -> Dict[str, Any]:
    return dict(
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
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        candidate_summary_source="e2_world_forward",
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
    )


def _all_on_extra_kwargs() -> Dict[str, Any]:
    """The gating / valuation / modulation superstructure (V3-EXQ-714 ARM_ON)."""
    return dict(
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
        modulatory_channel_route_weight=MODULATORY_CHANNEL_ROUTE_WEIGHT,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        use_f_eligibility_demotion=True,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_f_eligibility_adaptive_floor=True,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        use_dacc=True,
        use_go_nogo_constitution=True,
        gng_perseveration_floor=GNG_PERSEVERATION_FLOOR,
        gng_safety_floor=GNG_SAFETY_FLOOR,
        gng_protect_min_eligible=GNG_PROTECT_MIN_ELIGIBLE,
        gng_viability_floor=GNG_VIABILITY_FLOOR,
        use_ofc_analog=True,
        ofc_state_dim=OFC_STATE_DIM,
        ofc_harm_dim=OFC_HARM_DIM,
        ofc_bias_scale=OFC_BIAS_SCALE,
        use_ofc_devaluation_head=True,
        ofc_devaluation_bias_scale=OFC_DEVAL_BIAS_SCALE,
        ofc_train_devaluation_head=True,
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        e3_diversity_stratified_within_class_temperature=None,
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
        use_candidate_rule_field=True,
    )


def _make_all_on_agent(env: CausalGridWorldV2) -> REEAgent:
    kwargs = _base_config_kwargs(env)
    kwargs.update(_all_on_extra_kwargs())
    cfg = REEConfig.from_dims(**kwargs)
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 e2 contrastive step (mirror V3-EXQ-724)
# ---------------------------------------------------------------------------
def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
    return samples


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K,
        actions=actions_K,
        z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _warmup_all_on_agent(
    agent: REEAgent,
    seed: int,
    p0_episodes: int,
    steps_per_episode: int,
) -> int:
    """P0 world-model warmup: agent acts, transitions captured, e2 contrastive trains.

    Mirrors the V3-EXQ-724 P0 phase (no downstream head training -- bias heads stay
    untrained; this is the standard warmup that precedes competence-training, NOT P1).
    Returns the number of e2 training steps taken (for provenance).
    """
    env = _make_env(seed)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    n_train_steps = 0

    for ep in range(p0_episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

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

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                break

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                if _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng) is not None:
                    n_train_steps += 1

            _flat, harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(harm_signal)

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=harm_signal, world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=max(0.0, 1.0 - energy),
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        cur = ep + 1
        if cur % 25 == 0 or cur == p0_episodes:
            print(f"  [train] warmup seed={seed} ep {cur}/{p0_episodes}", flush=True)

    return n_train_steps


# ---------------------------------------------------------------------------
# Arm table
# ---------------------------------------------------------------------------
ARMS = ("random_walk", "ree_p0warmup_allon", "greedy_oracle")


def _run_cell(
    arm_id: str,
    seed: int,
    p0_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """One (arm, seed) capability-eval cell. Returns the seed-level metric row."""
    n_train_steps = 0
    if arm_id == "random_walk":
        policy = RandomPolicy(seed)
    elif arm_id == "greedy_oracle":
        policy = OraclePolicy()
    elif arm_id == "ree_p0warmup_allon":
        warm_env = _make_env(seed)
        agent = _make_all_on_agent(warm_env)
        n_train_steps = _warmup_all_on_agent(agent, seed, p0_episodes, steps_per_episode)
        policy = REEForwardPolicy(agent, name="ree_p0warmup_allon")
    else:
        raise ValueError(f"unknown arm {arm_id!r}")

    eval_env = _make_env(seed)
    row = evaluate_seed(policy, eval_env, eval_episodes, steps_per_episode)
    row["arm_id"] = arm_id
    row["seed"] = int(seed)
    row["n_e2_train_steps"] = int(n_train_steps)
    return row


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    print(
        f"Capability yardstick calibration ({len(ARMS)} arms x {len(seeds)} seeds; "
        f"P0_warmup={p0_episodes}, eval_eps={eval_episodes}, steps={steps_per_episode}, "
        f"dry_run={dry_run})",
        flush=True,
    )

    cells: List[Dict[str, Any]] = []
    for arm_id in ARMS:
        for s in seeds:
            print(f"Seed {s} Condition {arm_id}", flush=True)
            slice_cfg = {
                "arm_id": arm_id,
                "p0_episodes": int(p0_episodes) if arm_id == "ree_p0warmup_allon" else 0,
                "eval_episodes": int(eval_episodes),
                "steps_per_episode": int(steps_per_episode),
                "env_kwargs": dict(ENV_KWARGS),
            }
            with arm_cell(
                s,
                config_slice=slice_cfg,
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = _run_cell(arm_id, s, p0_episodes, eval_episodes, steps_per_episode)
                cell.stamp(row)
            cells.append(row)
            print(
                f"verdict: PASS (arm={arm_id} seed={s} "
                f"forage/ep={row['foraging_competence']} "
                f"survival={row['survival_horizon']} "
                f"goal_reach={row['goal_reach_rate']} "
                f"plan_depth={row['planning_depth']})",
                flush=True,
            )

    # ----- Per-arm aggregation -----
    arm_summaries: Dict[str, Dict[str, Any]] = {}
    for arm_id in ARMS:
        rows = [c for c in cells if c["arm_id"] == arm_id]
        arm_summaries[arm_id] = summarize_arm(rows)

    report = build_report(arm_summaries, floor="random_walk", ceiling="greedy_oracle")
    readiness = report["readiness"]
    oracle_clears_floor = bool(readiness["oracle_clears_floor"])
    yardstick_discriminates = bool(readiness["yardstick_discriminates"])
    readiness_met = bool(oracle_clears_floor)

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif yardstick_discriminates:
        outcome = "PASS"
        label = "capability_yardstick_calibrated"
    else:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    direction = "non_contributory"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "oracle_clears_competence_floor",
                "kind": "readiness",
                "description": (
                    "The greedy nearest-resource ORACLE (ceiling anchor, no agent) must clear "
                    "COMPETENCE_RESOURCE_FLOOR resources/episode, proving the floor is "
                    "ACHIEVABLE in this exact env so the yardstick's scale is calibratable. "
                    "Below-floor => the floor is not achievable here (env too sparse/lethal) => "
                    "substrate_not_ready_requeue, NEVER a capability verdict."
                ),
                "control": "greedy nearest-resource oracle forager, same ENV_KWARGS/seed, no agent",
                "measured": float(readiness["oracle_foraging_competence"]),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(oracle_clears_floor),
            },
        ],
        "criteria": [
            {
                "name": "yardstick_discriminates_ceiling_above_floor",
                "load_bearing": True,
                "passed": bool(yardstick_discriminates),
            },
        ],
        "criteria_non_degenerate": {
            "oracle_clears_floor": bool(oracle_clears_floor),
            "oracle_foraging_above_random": bool(yardstick_discriminates),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": list(seeds),
        "p0_warmup_episodes": int(p0_episodes),
        "eval_episodes": int(eval_episodes),
        "steps_per_episode": int(steps_per_episode),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "capability_report": report,
        "per_arm": arm_summaries,
        "arm_results": cells,
        "readiness_gates": {
            "oracle_clears_floor": oracle_clears_floor,
            "yardstick_discriminates": yardstick_discriminates,
            "readiness_met": readiness_met,
        },
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    rep = result["capability_report"]
    rg = result["readiness_gates"]
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "sleep_driver_pattern": "none",
        "evidence_direction_note": (
            f"V3-EXQ-727 CAPABILITY YARDSTICK CALIBRATION (experiment_purpose=baseline, "
            f"claim_ids=[], non_contributory -- EXCLUDED from governance scoring; PROMOTES / "
            f"DEMOTES NOTHING). WS-3 of ree_ai_design_critique_plan.md: a claim-agnostic "
            f"capability-eval suite (foraging_competence, survival_horizon, goal_reach_rate, "
            f"planning_depth) implemented as the reusable harness block "
            f"experiments/_lib/capability_eval.py, calibrated here against random_walk (floor) "
            f"and greedy_oracle (ceiling) anchors, plus an ree_p0warmup_allon forward-eval point "
            f"that proves the block attaches to REEAgent. Readiness: oracle clears the "
            f"{COMPETENCE_RESOURCE_FLOOR} floor "
            f"(oracle_forage/ep={rep['readiness']['oracle_foraging_competence']}, "
            f"clears_floor={rg['oracle_clears_floor']}); yardstick_discriminates="
            f"{rg['yardstick_discriminates']}. Self-route (HYPOTHESIS, not a verdict): "
            f"readiness_met={rg['readiness_met']} -> label="
            f"{result['interpretation_label']}. The TRAINED all-ON denominator is produced by "
            f"the in-flight V3-EXQ-724 (this run does not re-train it); import capability_eval "
            f"into the next trained all-ON run to land all four metrics on this scale."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "capability yardstick calibration; 3 policy arms x 3 seeds",
            "arms": {
                "random_walk": "uniform-random action policy -- FLOOR anchor",
                "ree_p0warmup_allon": (
                    "all-ON REE stack (714 ARM_ON) after P0 world-model warmup, NO P1 "
                    "competence-training (bias heads untrained) -- REEAgent integration + "
                    "substrate-as-built point"
                ),
                "greedy_oracle": "nearest-resource greedy forager -- CEILING/achievability anchor",
            },
            "metrics": list(METRIC_KEYS),
            "reusable_block": "experiments/_lib/capability_eval.py",
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-724",
            "alpha_world": 0.9,
            "reef_bipartite_layout": True,
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-727 capability yardstick calibration (baseline; claim_ids=[])"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        eval_eps = DRY_RUN_EVAL
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        eval_eps = EVAL_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    rep = result["capability_report"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness_met={result['readiness_gates']['readiness_met']} "
        f"oracle_forage/ep={rep['readiness']['oracle_foraging_competence']} "
        f"random_forage/ep={rep['readiness']['floor_foraging_competence']}",
        flush=True,
    )
    for arm_id, st in result["per_arm"].items():
        print(
            f"  ARM {arm_id}: forage/ep={st['foraging_competence_mean']} "
            f"survival={st['survival_horizon_mean']} "
            f"goal_reach={st['goal_reach_rate_mean']} "
            f"plan_depth={st['planning_depth_mean']}",
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
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
