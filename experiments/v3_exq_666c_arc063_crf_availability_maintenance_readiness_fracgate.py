"""V3-EXQ-666c: ARC-063 CRF availability-maintenance readiness -- clean fraction-gated re-run.

Purpose: diagnostic (claim_ids=[]). Supersedes V3-EXQ-666b. The CLEAN re-run of the
666b ISOLATING-STATISTIC (crf_frac_maintained) fraction-gate design, fixing the ONE
defect that starved 666b: the e2ctx full-pool differentiation NON-VACUITY precondition
was measured POST the context-absent silence gap, where ARM_1 (no-maintenance) has
ERODED its pool to empty BY DESIGN -- so its end-state crf_max_pairwise_rule_dist=0.0
dragged the min-over-ARM_1+ARM_2 precondition to 0.0 and self-routed
substrate_not_ready_requeue even though ARM_2 held 1.71 and BOTH load-bearing criteria
(ARM_2 clears the frac gate, ARM_1 below the ceiling) PASSED. Does NOT weight governance
and does NOT validate/weaken MECH-309/ARC-062/ARC-063 -- those stay candidate /
substrate_ceiling / v3_pending / pending_retest_after_substrate. The ARC-062 GAP-B
BEHAVIOURAL re-run (654c, committed-class entropy falsifier on the field-populated
maintained substrate) is a SEPARATE later /queue-experiment session GATED on this
readiness PASS.

CONTEXT (the 666 lineage; all claim-free diagnostics, all FAIL/non_contributory):
  V3-EXQ-666  (2026-06-11): mature_dynamics_insufficient -- differentiation (e2ctx dist
              1.71) and persistence (crf_frac_active) in tension. Routed -> built the
              crf-availability-maintenance mechanism (ARC-063 amend).
  V3-EXQ-666a (2026-06-11): the maintenance mechanism WORKS and strictly dominates on the
              isolating crf_frac_maintained statistic (ARM_2 1.0/0.625/0.938 vs ARM_1
              0.188/0.125/0.438 vs ARM_0 0.0, clean monotone 3/3-seed separation), but the
              pre-registered gate used a COUNT floor (crf_n_maintained_reactivatable>=2)
              that differentiation-alone also clears -> measurement/test-design defect,
              NOT substrate failure. Routed -> 666b re-gated on the FRACTION statistic.
  V3-EXQ-666b (2026-06-12): the fraction-gate logic DISCRIMINATED correctly
              (frac_maintained_gate_discriminates_vs_arm1=true; ARM_2 0.8125 mean cleared,
              ARM_1 below ceiling -- both load-bearing criteria PASSED) BUT the non-vacuity
              precondition e2ctx_full_pool_differentiated_supra_floor returned 0.0 because
              it read crf_max_pairwise_rule_dist POST-gap, where ARM_1 (maintenance OFF)
              erodes to an empty pool by design -> self-route substrate_not_ready_requeue
              (a starved test, NOT a fix verdict). NET STATE: maintenance CONFIRMED
              FUNCTIONAL; no single run cleared the fraction gate AND all its non-vacuity
              preconditions simultaneously.

WHAT 666c CHANGES vs 666b (two coupled fixes; the task-directed re-gate):
  (1) PRE-GAP DIFFERENTIATION READOUT (the load-bearing fix). The e2ctx full-pool
      differentiation precondition is now measured PRE-gap (crf_max_pairwise_rule_dist
      snapshotted right after behavioural-runtime training, BEFORE the context-absent
      silence gap). This precondition asks "CAN the e2_world_forward context differentiate
      the full pool?" -- a property of the differentiation SOURCE, which exists at end-of-
      training -- NOT "did the pool survive the silence gap" (ARM_1's expected erosion, the
      maintenance story the load-bearing gate already tests post-gap). 666b conflated the
      two by reading the precondition post-gap.
  (2) ENLARGED MATURATION WINDOW. N_EPISODES 100 -> 200 so the e2ctx context differentiates
      a >=2-rule pool PRE-gap on EVERY seed (666b ARM_1 seed-43 held only 1 rule pre-gap;
      a larger maturation budget makes pre-gap full-pool differentiation > 0.1 robust
      across all 3 seeds, so the now-correctly-timed precondition is reliably met).

WHAT IS UNCHANGED (the 666b discrimination design, all measured POST-gap -- the maintenance
test the silence gap exists to deliver):
  - LOAD-BEARING gate: ARM_2 (maintenance) crf_frac_maintained >= FRAC_MAINTAINED_FLOOR
    (0.625) on >=2/3 seeds (maintenance HOLDS the differentiated pool through silence).
  - DISCRIMINATION: ARM_1 (mature+e2ctx, no maintenance) crf_frac_maintained <
    ARM1_FRAC_CEILING (0.5) on >=2/3 seeds (differentiation-alone does NOT hold -- isolates
    the maintenance contribution; the 666a count floor could not do this).
  - ARM_2 maintained-pool non-vacuity: crf_n_maintained_reactivatable >= MAINT_N_FLOOR (2)
    AND crf_maintained_pairwise_dist > MAINT_DIST_FLOOR (0.1) (count + differentiation
    readiness for the count-fraction gate; the counted maintained pool is a genuine >=2-rule
    differentiated pool, not a vacuous clone-count).

SAME-STATISTIC NON-VACUITY DISCIPLINE (V3-EXQ-643 lesson; the gate statistic itself is
asserted as a readiness precondition so a starved test self-routes
substrate_not_ready_requeue, never a false verdict):
  - minted >= 2 on every cell (the field engaged at all).
  - e2ctx arms DIFFERENTIATE PRE-GAP (crf_max_pairwise_rule_dist > DIFF_FLOOR on the full
    pool, >=2/3) -- the differentiated pool the maintenance is supposed to HOLD must first
    exist at end-of-training. [666c FIX: measured PRE-gap, not post-gap.]
  - the MAINTAINED pool the FRACTION gate counts is itself non-degenerate on the ARM_2
    positive control (post-gap n_maintained >= 2 AND crf_maintained_pairwise_dist > floor).
  Any unmet precondition -> substrate_not_ready_requeue (NOT a fix verdict).

DESIGN (3 arms x 3 seeds). ALL arms run the SAME 654b/666/666a/666b matched stack
(use_candidate_rule_field=True + use_lateral_pfc_analog=True UN-ZEROED trainable head +
crf_persist_rules_across_episode_reset=True + SD-056 e2 contrastive trained ONLINE so
e2.world_forward is action-divergent). The swept variables are the three CRF amend flags;
ARM_2 (mature + e2ctx + activity-silent maintenance) is the adopted mature-regime default:
  ARM_0_OFF                      : legacy CRF (no mature / no e2ctx / no maintenance)
                                   -- positive control reproducing the 654b/666 churn.
  ARM_1_MATURE_E2CTX             : mature + e2ctx, maintenance OFF
                                   -- DIFFERENTIATES (pre-gap) but the pool does NOT hold
                                      through silence (the 666 persistence collapse).
  ARM_2_MATURE_E2CTX_MAINTENANCE : mature + e2ctx + maintenance (floor 0.45, decay 0.0)
                                   -- the differentiated pool is held activity-silently.

INTERPRETATION GRID (the self-route is a hypothesis -- adjudicate before it drives any
governance action):
  - non_vacuity unmet (a cell minted <2, OR the e2ctx arms did not differentiate PRE-gap,
    OR the ARM_2 post-gap maintained pool is degenerate: n<2 / undifferentiated) ->
    substrate_not_ready_requeue (the fraction-gate test was starved; re-queue at a still-
    larger P0; NOT a verdict).
  - ARM_2 clears the frac_maintained gate on >=2/3 AND ARM_1 does NOT clear the frac ceiling
    on >=2/3 -> PASS: activity-silent maintenance HOLDS a larger fraction of the
    differentiated pool than differentiation-alone (isolated on the fraction statistic).
  - ARM_2 does NOT clear the frac gate on >=2/3 (with differentiation real) -> FAIL:
    maintenance insufficient at behavioural runtime -> /failure-autopsy (NOT a requeue).
  - ARM_1 ALSO clears the frac ceiling -> FAIL: even the isolating fraction statistic does
    not separate maintenance from differentiation-alone -> /failure-autopsy (would
    contradict the 666a/666b manifest data; revisit the gap regime).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_666c_arc063_crf_availability_maintenance_readiness_fracgate.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_666b_arc063_crf_availability_maintenance_readiness_fracgate"

SEEDS = [42, 43, 44]
# ENLARGED maturation window (666c fix #2): 100 -> 200 so the e2ctx context differentiates
# a >=2-rule full pool PRE-gap on EVERY seed (666b ARM_1 seed-43 held only 1 rule pre-gap).
N_EPISODES = 200
STEPS_PER_EPISODE = 40
# Context-absent silence gap (the longer-gap regime lever, unchanged from 666b): pure decay
# ticks AFTER behavioural-runtime training so differentiation-alone (ARM_1) erodes its
# maintained pool below the floor while activity-silent maintenance (ARM_2) HOLDS it.
GAP_TICKS = 2000
DRY_N_EPISODES = 3
DRY_STEPS_PER_EPISODE = 10
DRY_GAP_TICKS = 50

# CRF-readiness gate floors -- LOAD-BEARING, on the ISOLATING statistic crf_frac_maintained
# (the maintained-reactivatable fraction = n_maintained / n_slots), measured POST-gap.
FRAC_MAINTAINED_FLOOR = 0.625   # ARM_2 crf_frac_maintained must clear this (maintenance holds
#                                 the pool); 666a ARM_2 = 1.0/0.625/0.938.
ARM1_FRAC_CEILING = 0.5         # ARM_1 crf_frac_maintained must stay BELOW this
#                                 (differentiation-alone does NOT hold); 666a ARM_1 =
#                                 0.188/0.125/0.438.

# Same-statistic non-vacuity floors.
MAINT_N_FLOOR = 2          # crf_n_maintained_reactivatable on ARM_2 POST-gap (count-readiness
#                            for a count-fraction gate).
MAINT_DIST_FLOOR = 0.1     # crf_maintained_pairwise_dist on ARM_2 POST-gap -- the COUNTED pool
#                            is differentiated, not a vacuous clone-count.
DIFF_FLOOR = 0.1           # crf_max_pairwise_rule_dist on the FULL pool, measured PRE-GAP (666c
#                            fix #1): the differentiated pool the maintenance is supposed to
#                            HOLD exists at end-of-training (the differentiation SOURCE works).
MIN_MINTED_FLOOR = 2       # every cell must mint >=2 rules (field engaged).
SEED_MAJORITY = 2          # >=2/3 seeds.

# SD-056 online e2 training (mirror V3-EXQ-666b / 666a / 666 / 654b / 649 harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Maintenance knobs (the amend defaults; set explicitly when maintenance ON).
MAINTENANCE_FLOOR = 0.45
MAINTENANCE_DECAY = 0.0

# IDENTICAL env to V3-EXQ-666b / 666a / 666 / 654b / 614e so the readiness regime matches the
# GAP-B falsifier.
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

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_OFF",
        "label": "legacy_crf_no_mature_no_e2ctx_no_maintenance",
        "crf_mature_pool_dynamics": False,
        "crf_context_from_e2_world_forward": False,
        "crf_availability_maintenance": False,
    },
    {
        "arm_id": "ARM_1_MATURE_E2CTX",
        "label": "mature_e2ctx_no_maintenance_differentiates_pre_gap_but_pool_erodes_under_silence",
        "crf_mature_pool_dynamics": True,
        "crf_context_from_e2_world_forward": True,
        "crf_availability_maintenance": False,
    },
    {
        "arm_id": "ARM_2_MATURE_E2CTX_MAINTENANCE",
        "label": "mature_e2ctx_activity_silent_maintenance_adopted_mature_default",
        "crf_mature_pool_dynamics": True,
        "crf_context_from_e2_world_forward": True,
        "crf_availability_maintenance": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """654b/666/666a/666b matched stack; the ONLY swept variables are the three CRF amend flags."""
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
        # Layer A: SP-CEM (candidate-pool first-action-class diversity).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # GAP-A (V3-EXQ-649): shared per-candidate signal from e2.world_forward.
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (V3-EXQ-643a float32 fix).
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
        # MECH-341 (stratified across-class; within-class temperature default).
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=1.0,
        e3_diversity_stratified_within_class_temperature=None,
        # MECH-313 noise floor.
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        # V_s minimal stack.
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=0.5,
        vs_gate_e1_threshold=0.4,
        # ARC-062 GatedPolicy + SD-033a LateralPFCAnalog (un-zeroed + trainable head).
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        # SD-056 (e2 action-conditional divergence; trained online below).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=True,
        e2_action_contrastive_horizon=5,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # The 654a maturity fix (matched across arms).
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        # --- The ONLY swept variables: the mature + e2ctx + maintenance flags ---
        crf_mature_pool_dynamics=bool(arm["crf_mature_pool_dynamics"]),
        crf_context_from_e2_world_forward=bool(
            arm["crf_context_from_e2_world_forward"]
        ),
        crf_availability_maintenance=bool(arm["crf_availability_maintenance"]),
        crf_maintenance_floor=MAINTENANCE_FLOOR,
        crf_maintenance_decay=MAINTENANCE_DECAY,
    )
    return REEAgent(cfg)


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-666b)
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
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K,
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


# ---------------------------------------------------------------------------
# One (arm, seed) cell: drive the agent, snapshot the PRE-gap full-pool
# differentiation, run a context-absent silence gap, then read the maintained pool.
# ---------------------------------------------------------------------------
def _run_cell(arm: Dict[str, Any], seed: int, n_episodes: int,
              steps_per_episode: int, gap_ticks: int) -> Dict[str, Any]:
    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    sample_rng = random.Random(seed * 7919 + 13)
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        deque(maxlen=TRANSITION_BUFFER_MAX)
    )
    n_contrastive_steps = 0
    error_note: Optional[str] = None

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()  # crf_persist=True -> the field's reset() is a no-op
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

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

            # SD-056 transition capture (z0, a) this tick -> z1 next tick.
            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                        and torch.isfinite(z1_obs).all()):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

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
                error_note = (
                    f"non-finite action arm={arm['arm_id']} seed={seed} "
                    f"ep={ep} step={_step}"
                )
                break

            # Capture (z0, a) for the next-tick SD-056 transition.
            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 online contrastive training (every tick).
            loss_val = _e2_contrastive_step(
                agent=agent, buffer=transition_buffer,
                optimiser=e2_opt, rng=sample_rng,
            )
            if loss_val is not None and math.isfinite(loss_val):
                n_contrastive_steps += 1

            _, _harm_signal, done, info, obs_dict = env.step(action)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(_harm_signal), world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=max(0.0, 1.0 - energy),
                )
            if done:
                break

        if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] cell {arm['arm_id']} seed={seed} "
                f"ep {ep + 1}/{n_episodes}",
                flush=True,
            )

    crf = agent.candidate_rule_field

    # --- PRE-GAP SNAPSHOT (the 666c fix): read full-pool differentiation at end-of-training,
    # BEFORE the silence gap erodes ARM_1. This is the differentiation-SOURCE readiness
    # ("CAN the e2ctx context differentiate the full pool?"), which the e2ctx non-vacuity
    # precondition routes on -- NOT the post-gap survival (the maintenance story the
    # load-bearing frac gate already tests). 666b conflated the two by reading post-gap.
    pre_gap_state = crf.get_state() if crf is not None else {}
    pre_gap_max_dist = float(pre_gap_state.get("crf_max_pairwise_rule_dist", 0.0))
    pre_gap_n_maintained = (
        len(crf.maintained_reactivatable_rules()) if crf is not None else 0
    )

    # --- CONTEXT-ABSENT SILENCE GAP (unchanged from 666b) ---
    # Iterate the field's per-tick credit/decay path with NO matching context: the
    # maintenance-vs-decay branch (CandidateRuleField.credit) runs over pure silence.
    # Differentiation-alone (ARM_1) erodes its maintained pool below the floor;
    # activity-silent maintenance (ARM_2, maintenance_decay=0.0) HOLDS it. Waking-stream
    # silence (NOT replay) -- no minting, no matching, no z_goal write.
    if crf is not None and gap_ticks > 0:
        for _ in range(gap_ticks):
            crf.credit(outcome_signal=0.0)
    if gap_ticks > 0:
        print(
            f"  [gap] cell {arm['arm_id']} seed={seed} silence_ticks={gap_ticks} "
            f"pre_gap_dist={pre_gap_max_dist:.3f} n_maintained "
            f"{pre_gap_n_maintained}->"
            f"{len(crf.maintained_reactivatable_rules()) if crf is not None else 0}",
            flush=True,
        )

    st = crf.get_state() if crf is not None else {}
    # PRIMARY (load-bearing) -- the ISOLATING fraction statistic + its substrate, POST-gap.
    frac_maintained = float(st.get("crf_frac_maintained", 0.0))
    maint_n = int(st.get("crf_n_maintained_reactivatable", 0))
    maint_dist = float(st.get("crf_maintained_pairwise_dist", 0.0))
    # SECONDARY / diagnostic readouts.
    frac_active = float(st.get("crf_frac_active", 0.0))
    max_dist_post = float(st.get("crf_max_pairwise_rule_dist", 0.0))
    n_minted = int(st.get("crf_n_minted_total", 0))
    n_slots = int(st.get("crf_n_slots_minted", 0))

    # Per-cell verdict.
    frac_gate_cleared = frac_maintained >= FRAC_MAINTAINED_FLOOR
    frac_below_ceiling = frac_maintained < ARM1_FRAC_CEILING
    # 666c FIX: the differentiation flag routes on the PRE-gap full-pool distance (the
    # differentiation source), NOT the post-gap eroded pool.
    differentiated = pre_gap_max_dist > DIFF_FLOOR
    maintained_pool_nondeg = (maint_n >= MAINT_N_FLOOR) and (maint_dist > MAINT_DIST_FLOOR)
    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": seed,
        # primary (maintained-pool fraction) readout -- POST-gap
        "crf_frac_maintained": frac_maintained,
        "crf_n_maintained_reactivatable": maint_n,
        "crf_maintained_pairwise_dist": maint_dist,
        "crf_maintained_reactivation_threshold": float(
            st.get("crf_maintained_reactivation_threshold", 0.0)
        ),
        # PRE-gap differentiation source readout (the 666c fix)
        "crf_max_pairwise_rule_dist_pre_gap": pre_gap_max_dist,
        "n_maintained_pre_gap": pre_gap_n_maintained,
        # secondary / diagnostic
        "crf_frac_active": frac_active,
        "crf_max_pairwise_rule_dist_post_gap": max_dist_post,
        "crf_n_minted_total": n_minted,
        "crf_n_slots_minted": n_slots,
        "crf_n_retired_total": int(st.get("crf_n_retired_total", 0)),
        "n_contrastive_steps": n_contrastive_steps,
        "gap_ticks": gap_ticks,
        # gates
        "frac_gate_cleared": bool(frac_gate_cleared),
        "frac_below_ceiling": bool(frac_below_ceiling),
        "differentiated": bool(differentiated),
        "maintained_pool_nondegenerate": bool(maintained_pool_nondeg),
        "minted_ge2": bool(n_minted >= MIN_MINTED_FLOOR),
        "error_note": error_note,
    }


def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["arm_id"] == arm_id]


def main(dry_run: bool = False):
    t0 = time.time()
    n_episodes = DRY_N_EPISODES if dry_run else N_EPISODES
    steps_per_episode = DRY_STEPS_PER_EPISODE if dry_run else STEPS_PER_EPISODE
    gap_ticks = DRY_GAP_TICKS if dry_run else GAP_TICKS

    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            config_slice = {
                "arm_id": arm["arm_id"],
                "crf_mature_pool_dynamics": arm["crf_mature_pool_dynamics"],
                "crf_context_from_e2_world_forward":
                    arm["crf_context_from_e2_world_forward"],
                "crf_availability_maintenance": arm["crf_availability_maintenance"],
                "maintenance_floor": MAINTENANCE_FLOOR,
                "maintenance_decay": MAINTENANCE_DECAY,
                "gap_ticks": gap_ticks,
                "env_kwargs": ENV_KWARGS,
                "n_episodes": n_episodes,
                "steps_per_episode": steps_per_episode,
            }
            with arm_cell(seed, config_slice=config_slice,
                          script_path=Path(__file__)) as cell:
                row = _run_cell(arm, seed, n_episodes, steps_per_episode, gap_ticks)
                cell.stamp(row)
            rows.append(row)
            print(
                f"verdict: {'PASS' if row['frac_gate_cleared'] else 'FAIL'}",
                flush=True,
            )

    # --- Aggregate per arm ---
    arm_summary: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        ar = _arm_rows(rows, arm["arm_id"])
        arm_summary[arm["arm_id"]] = {
            "n_seeds": len(ar),
            "n_frac_gate_cleared": sum(1 for r in ar if r["frac_gate_cleared"]),
            "n_frac_below_ceiling": sum(1 for r in ar if r["frac_below_ceiling"]),
            "n_differentiated": sum(1 for r in ar if r["differentiated"]),
            "n_maintained_pool_nondegenerate": sum(
                1 for r in ar if r["maintained_pool_nondegenerate"]
            ),
            "n_minted_ge2": sum(1 for r in ar if r["minted_ge2"]),
            "min_n_minted": min((r["crf_n_minted_total"] for r in ar), default=0),
            "mean_frac_maintained": (
                sum(r["crf_frac_maintained"] for r in ar) / len(ar) if ar else 0.0
            ),
            "seed_frac_maintained": [r["crf_frac_maintained"] for r in ar],
            "mean_n_maintained": (
                sum(r["crf_n_maintained_reactivatable"] for r in ar) / len(ar)
                if ar else 0.0
            ),
            "mean_maintained_pairwise_dist": (
                sum(r["crf_maintained_pairwise_dist"] for r in ar) / len(ar)
                if ar else 0.0
            ),
            "mean_frac_active": (
                sum(r["crf_frac_active"] for r in ar) / len(ar) if ar else 0.0
            ),
            "mean_max_pairwise_rule_dist_pre_gap": (
                sum(r["crf_max_pairwise_rule_dist_pre_gap"] for r in ar) / len(ar)
                if ar else 0.0
            ),
            "seed_max_pairwise_rule_dist_pre_gap": [
                r["crf_max_pairwise_rule_dist_pre_gap"] for r in ar
            ],
            "mean_max_pairwise_rule_dist_post_gap": (
                sum(r["crf_max_pairwise_rule_dist_post_gap"] for r in ar) / len(ar)
                if ar else 0.0
            ),
        }

    # --- Gates ---
    # NON-VACUITY (same-statistic discipline): (a) every cell mints >=2; (b) the e2ctx arms
    # DIFFERENTIATE on the FULL pool PRE-GAP (crf_max_pairwise_rule_dist_pre_gap > DIFF_FLOOR,
    # >=2/3) -- the 666c fix: the differentiation source exists at end-of-training, read
    # BEFORE the silence gap erodes ARM_1; (c) the ARM_2 POST-gap MAINTAINED pool the FRACTION
    # gate counts is itself a genuine >=2-rule DIFFERENTIATED pool. Any unmet ->
    # substrate_not_ready_requeue.
    min_minted_all = min(r["crf_n_minted_total"] for r in rows)
    min_diff_e2ctx_pre_gap = min(
        r["crf_max_pairwise_rule_dist_pre_gap"] for r in rows
        if r["arm_id"] in ("ARM_1_MATURE_E2CTX", "ARM_2_MATURE_E2CTX_MAINTENANCE")
    )
    arm2_rows = _arm_rows(rows, "ARM_2_MATURE_E2CTX_MAINTENANCE")
    min_arm2_maint_n = min((r["crf_n_maintained_reactivatable"] for r in arm2_rows),
                           default=0)
    min_arm2_maint_dist = min((r["crf_maintained_pairwise_dist"] for r in arm2_rows),
                              default=0.0)

    # NON-VACUITY preconditions are ALL-CELLS (min over relevant cells > floor) so the
    # precondition's `measured` (= the min) and its `met` agree with the indexer's recompute
    # (build_experiment_indexes recomputes met from measured+threshold and ignores the
    # author's met when both are present -- a >=2/3 majority here would disagree with a
    # min-based measured and false-flag precondition_unmet). This also matches the task
    # gate "e2ctx full-pool differentiation > 0.1 on all 3 seeds".
    minted_met = min_minted_all >= MIN_MINTED_FLOOR
    differentiation_met = min_diff_e2ctx_pre_gap > DIFF_FLOOR  # all e2ctx cells, PRE-gap
    maintained_pool_nondeg_met = (
        (min_arm2_maint_n >= MAINT_N_FLOOR) and (min_arm2_maint_dist > MAINT_DIST_FLOOR)
    )  # all ARM_2 cells, POST-gap
    non_vacuity_met = minted_met and differentiation_met and maintained_pool_nondeg_met

    # LOAD-BEARING (the isolating fraction statistic): ARM_2 clears the
    # crf_frac_maintained floor on >=2/3 seeds.
    readiness_met = (
        arm_summary["ARM_2_MATURE_E2CTX_MAINTENANCE"]["n_frac_gate_cleared"]
        >= SEED_MAJORITY
    )
    # DISCRIMINATION (same statistic): ARM_1 (no maintenance) stays BELOW the fraction
    # ceiling on >=2/3 seeds.
    discrimination_met = (
        arm_summary["ARM_1_MATURE_E2CTX"]["n_frac_below_ceiling"] >= SEED_MAJORITY
    )
    any_error = any(r["error_note"] for r in rows)

    if not non_vacuity_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route = (
            "re-queue at a still-larger P0 (a cell minted <2, OR the e2ctx arms did not "
            "differentiate PRE-gap, OR the ARM_2 post-gap maintained pool the fraction "
            "gate counts is degenerate -- the isolating-fraction test was starved, NOT a "
            "fix verdict)"
        )
    elif readiness_met and discrimination_met:
        outcome = "PASS"
        label = "crf_availability_maintenance_isolated_on_frac_maintained"
        route = (
            "PASS -> governance flips substrate_queue crf-availability-maintenance "
            "ready=True (maintenance isolated on the crf_frac_maintained fraction); "
            "queue the 654c GAP-B behavioural re-run (committed-class entropy "
            "falsifier) on the field-populated maintained substrate"
        )
    elif not readiness_met:
        outcome = "FAIL"
        label = "maintenance_insufficient_at_behavioural_runtime"
        route = (
            "/failure-autopsy (ARM_2 maintenance did not hold a >=0.625 maintained "
            "fraction through the silence gap; NOT a requeue)"
        )
    else:  # readiness_met but not discrimination_met
        outcome = "FAIL"
        label = "differentiation_alone_clears_frac_maintained_ceiling"
        route = (
            "/failure-autopsy (ARM_1 without maintenance ALSO cleared the "
            "crf_frac_maintained ceiling -> even the isolating fraction statistic does "
            "not separate maintenance from differentiation-alone; contradicts the "
            "666a/666b manifest data, revisit the gap regime)"
        )
    if any_error:
        outcome = "FAIL"

    exp_type = "v3_exq_666c_arc063_crf_availability_maintenance_readiness_fracgate"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{exp_type}_{ts}_v3"  # run_id must END in _v3 (V3 tagging rule)
    elapsed = time.time() - t0

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": exp_type,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": [],
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "supersedes": SUPERSEDES,
        "metrics": {
            "arm_summary": arm_summary,
            "readiness_met_arm2_frac_maintained": readiness_met,
            "discrimination_met_arm1_below_frac_ceiling": discrimination_met,
            "non_vacuity_met": non_vacuity_met,
            "minted_met_all_cells_ge2": minted_met,
            "differentiation_met_e2ctx_arms_pre_gap": differentiation_met,
            "maintained_pool_nondegenerate_met_arm2": maintained_pool_nondeg_met,
            "min_n_minted_all_cells": min_minted_all,
            "min_max_pairwise_dist_e2ctx_arms_pre_gap": min_diff_e2ctx_pre_gap,
            "min_arm2_n_maintained": min_arm2_maint_n,
            "min_arm2_maintained_pairwise_dist": min_arm2_maint_dist,
            "frac_maintained_floor": FRAC_MAINTAINED_FLOOR,
            "arm1_frac_ceiling": ARM1_FRAC_CEILING,
            "maint_n_floor": MAINT_N_FLOOR,
            "maint_dist_floor": MAINT_DIST_FLOOR,
            "diff_floor": DIFF_FLOOR,
            "min_minted_floor": MIN_MINTED_FLOOR,
            "seed_majority": SEED_MAJORITY,
            "gap_ticks": gap_ticks,
            "n_episodes": n_episodes,
        },
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "route": route,
            "preconditions": [
                {
                    "name": "field_minted_ge2_all_cells",
                    "kind": "readiness",
                    "description": (
                        "every cell must mint >=2 rules so a below-gate reading is "
                        "maintenance-insufficiency, not field-not-engaged. Below "
                        "floor -> substrate_not_ready_requeue."
                    ),
                    "control": "all (arm x seed) cells driven over the 654b hazard env",
                    "measured": min_minted_all,
                    "threshold": MIN_MINTED_FLOOR,
                    "direction": "lower",
                    "met": bool(minted_met),
                },
                {
                    "name": "e2ctx_full_pool_differentiated_supra_floor_pre_gap",
                    "kind": "readiness",
                    "description": (
                        "the differentiated pool the maintenance is supposed to HOLD "
                        "must first exist at end-of-training: crf_max_pairwise_rule_dist "
                        "> floor on the FULL pool of the e2ctx arms, measured PRE-GAP "
                        "(666c fix -- the differentiation SOURCE readiness, before the "
                        "silence gap erodes ARM_1; 666b read this post-gap and ARM_1's "
                        "by-design erosion starved it). Below floor (e2ctx did not "
                        "differentiate at end-of-training) -> substrate_not_ready_requeue, "
                        "NOT a maintenance verdict (same-statistic V3-EXQ-643 lesson)."
                    ),
                    "control": (
                        "min crf_max_pairwise_rule_dist_pre_gap over ARM_1+ARM_2 cells "
                        "(e2ctx differentiation source ON, SD-056-trained e2)"
                    ),
                    "measured": min_diff_e2ctx_pre_gap,
                    "threshold": DIFF_FLOOR,
                    "direction": "lower",
                    "met": bool(differentiation_met),
                },
                {
                    "name": "arm2_maintained_pool_count_nondegenerate",
                    "kind": "readiness",
                    "description": (
                        "the load-bearing criterion gates on crf_frac_maintained (a "
                        "COUNT-fraction = n_maintained / n_slots). Count-readiness for "
                        "a count-fraction gate: the ARM_2 positive control must hold "
                        ">=2 maintained-reactivatable rules post-gap, else the fraction "
                        "the gate reads is degenerate -> substrate_not_ready_requeue."
                    ),
                    "control": (
                        "min crf_n_maintained_reactivatable over ARM_2 "
                        "(maintenance-ON positive control) after the silence gap"
                    ),
                    "measured": min_arm2_maint_n,
                    "threshold": MAINT_N_FLOOR,
                    "direction": "lower",
                    "met": bool(min_arm2_maint_n >= MAINT_N_FLOOR),
                },
                {
                    "name": "arm2_maintained_pool_differentiated_nondegenerate",
                    "kind": "readiness",
                    "description": (
                        "a high crf_frac_maintained must not be a vacuous count of "
                        "undifferentiated clones: the COUNTED maintained subset on the "
                        "ARM_2 positive control must itself be differentiated "
                        "(crf_maintained_pairwise_dist > floor). Below floor -> "
                        "substrate_not_ready_requeue (the fraction is degenerate)."
                    ),
                    "control": (
                        "min crf_maintained_pairwise_dist over ARM_2 cells after the "
                        "silence gap (the differentiation of the silently-held pool)"
                    ),
                    "measured": min_arm2_maint_dist,
                    "threshold": MAINT_DIST_FLOOR,
                    "direction": "lower",
                    "met": bool(min_arm2_maint_dist > MAINT_DIST_FLOOR),
                },
            ],
            "criteria_non_degenerate": {
                "frac_maintained_gate_discriminates_vs_arm1_no_maintenance": bool(
                    discrimination_met
                ),
                "arm2_maintained_pool_differentiated": bool(
                    maintained_pool_nondeg_met
                ),
                "e2ctx_full_pool_differentiated_pre_gap": bool(differentiation_met),
                "field_minted_ge2_all_cells": bool(minted_met),
            },
            "criteria": [
                {
                    "name": "ARM_2_MAINTENANCE_clears_frac_maintained_gate",
                    "load_bearing": True,
                    "passed": bool(readiness_met),
                },
                {
                    "name": "ARM_1_NO_MAINTENANCE_below_frac_maintained_ceiling",
                    "load_bearing": False,
                    "passed": bool(discrimination_met),
                },
            ],
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "ARC-063 crf-availability-maintenance substrate-readiness diagnostic "
            "(claim_ids=[], non_contributory -- does NOT weight governance). Supersedes "
            "V3-EXQ-666b. CLEAN re-run of the 666b crf_frac_maintained fraction-gate design, "
            "fixing the ONE defect that starved 666b: the e2ctx full-pool differentiation "
            "NON-VACUITY precondition is now measured PRE-gap (crf_max_pairwise_rule_dist "
            "snapshotted at end-of-training, before the context-absent silence gap erodes "
            "ARM_1's pool to empty by design) and the maturation window is enlarged "
            "(N_EPISODES 100 -> 200) so the e2ctx context differentiates a >=2-rule full pool "
            "PRE-gap on every seed. The LOAD-BEARING discrimination is unchanged and stays "
            "POST-gap: ARM_2 (maintenance) crf_frac_maintained >= 0.625 where ARM_1 "
            "(mature+e2ctx, no maintenance) stays below 0.5 -- the silence gap sharpens ARM_1's "
            "erosion while activity-silent maintenance (maintenance_decay=0.0) HOLDS the ARM_2 "
            "pool. Same-statistic non-vacuity discipline: minted>=2; e2ctx full-pool "
            "differentiation PRE-gap; AND the ARM_2 post-gap maintained pool the fraction gate "
            "counts is itself a >=2-rule differentiated pool (count + differentiation readiness "
            "on the positive control); any unmet -> substrate_not_ready_requeue, never a false "
            "verdict. PASS clears the substrate_queue ready gate before the 654c GAP-B "
            "behavioural re-run (MECH-309 / ARC-062) is scored; that falsifier is the "
            "governance-weighting successor, queued separately. crf_frac_active is the "
            "averaged-activity artefact the lit verdict identified -- kept as a secondary "
            "readout, NOT the persistence criterion. MECH-309/ARC-062/ARC-063 stay candidate / "
            "substrate_ceiling / v3_pending / pending_retest_after_substrate."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"verdict: {manifest['outcome']}")
    print(f"  label: {label}")
    for aid, summ in arm_summary.items():
        print(
            f"  {aid}: frac_cleared={summ['n_frac_gate_cleared']}/{summ['n_seeds']} "
            f"below_ceiling={summ['n_frac_below_ceiling']}/{summ['n_seeds']} "
            f"diff_pre_gap={summ['n_differentiated']}/{summ['n_seeds']} "
            f"mean_frac_maint={summ['mean_frac_maintained']:.3f} "
            f"mean_n_maint={summ['mean_n_maintained']:.2f} "
            f"mean_maint_dist={summ['mean_maintained_pairwise_dist']:.3f} "
            f"mean_dist_pre_gap={summ['mean_max_pairwise_rule_dist_pre_gap']:.3f} "
            f"mean_dist_post_gap={summ['mean_max_pairwise_rule_dist_post_gap']:.3f}",
            flush=True,
        )
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], str(out_path)


if __name__ == "__main__":
    _dry = "--dry-run" in sys.argv
    _outcome, _path = main(dry_run=_dry)
    emit_outcome(outcome=_outcome, manifest_path=_path)
