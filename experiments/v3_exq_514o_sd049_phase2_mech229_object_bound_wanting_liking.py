"""
V3-EXQ-514o -- SD-049 Phase-2 MECH-229 wanting != liking, measured as the SD-057
OBJECT-BOUND wanting-target != liking-target dissociation on a curriculum-built agent.
Successor to V3-EXQ-514n (NOT a supersede). Closes goal_pipeline:GAP-2 (and thereby
goal_pipeline:GAP-7, gated only on this L9 dissociation) on PASS.

ROUTING: V3-EXQ-514n FAILed (non_contributory, route_reason=contact_guard_unmet): the
per-seed contact guard (P2 contact_rate>0 AND z_goal_norm_at_contact_peak>0.4) cleared only
1/3 seeds (seed42 0.529; seeds43/44 0.391/0.377, just under 0.4). It ALSO scored
n_scored_wl_steps=0 on the one guard-passing seed because it sourced the liking target from
_contacted_resource_type(obs_dict), which at eval read-time reads the post-consumption-CLEARED
resource_type_at_agent cell and returns None -- so the WL DV never scored a step even when the
bank held >= 2 tokens. The SD-057 channel-write precursor V3-EXQ-681 PASSED (forced-contact
WL channel-write validated: C1 instrument-separates, C2 n_scored_wl_steps=40 / wl_dissoc 0.5 /
distinct_tokens 2 / drive_spread 1.0, C3 null-parity, C4 co-timing-structural) and its C4 leg
established that the consumed (liking) tag lives in info["sd049_consumed_type_tag_this_tick"],
NOT in the obs_dict at eval read-time (consumed_tag_in_obs_dict=false, consumed_tag_in_info=true).
This run applies that fix so the now-proven channel-write fires during REAL foraging.

WHAT THIS FIXES vs 514n:
  (1) LIKING-TARGET SOURCING (the 681-C4 fix; load-bearing). The liking target (last-consumed
      type) AND the L2-bind resource_type are now sourced from info["sd049_consumed_type_tag_this_tick"]
      returned by env.step(), the authoritative consumption signal, instead of the
      post-consumption-cleared obs_dict cell. A step is a genuine consumption event when that
      tag > 0. This is the only thing that lets the WL DV score steps during ecological foraging.
  (2) L9 ACCEPTANCE THRESHOLD. WL_FRACTION raised 0.3 -> 0.6, the original SD-049 verdict ARM_2
      acceptance ("wanting_target != liking_target trajectory fraction >= 0.6").
  (3) DRIVE-DEPENDENCE CONTROL (within-eval, free). Alongside the drive-coupled most-wanted
      (per_axis_drive set), the eval also reads the drive-uniform most-wanted (per_axis_drive=None),
      which collapses the depletion coupling -> argmax base_value. A genuine MECH-229
      drive-coupled dissociation has WL_drive >> WL_nodrive (reported diagnostic).

WHAT THIS MEASURES (the biologically + forward-faithful test the SD-049 verdict pre-registered):
the SD-057 OBJECT-BOUND dissociation between
  WANTING target = the MECH-346 most-wanted z_goal pointer
                   (IncentiveTokenBank.most_wanted k* = argmax_k base_value[k] *
                    (1 + kappa * per_axis_drive[k-1]); the object that SEEDS z_goal), and
  LIKING  target = the last-consumed object type (the consummatory tag at contact, from info).
SD-057 (landed 2026-06-04) makes these structurally able to DIFFER: having just eaten food
(liking=food), the per-axis drive for food is restored, so the most-wanted object shifts to a
still-depleted type (e.g. water) -- so z_goal points at water while liking just fired on food
(Berridge & Robinson wanting!=liking, object-bound). use_incentive_token_bank is already on in
the SD-057 substrate; the L2 bind + L4 pointer are exercised by agent.update_z_goal(resource_type=...)
at each contact (mirrors the substrate's own seed path).

SUBSTRATE: built through the FULL scaffolded_sd054_onboarding curriculum at the 603n config
(Stage-0 -> Stage-0b -> P0 -> Stage-H -> P1 -> P2; harm-pathway training ON), the substrate whose
readiness V3-EXQ-603n PASSED (scaffolded_sd054_onboarding.ready flipped true 2026-06-11). The full
603n lever stack is ported VERBATIM (see _make_scaffold_cfg). Then a frozen-policy SD-049 Phase-2
behavioural eval collects the object-bound WL DV at genuine consumption events.

CONTACT GUARD (retained from 514n / 603n G2+G3, the foraging non-vacuity floor):
  per-seed guard = (P2 contact_rate > 0) AND (P2 z_goal_norm_at_contact_peak > 0.4). A seed failing
  the guard is excluded from DV aggregation; < 2/3 seeds passing -> substrate_not_ready_requeue
  (FAIL; MECH-229 non_contributory). NOTE: 603n itself cleared this leg only 2/3 (seed44 0.374);
  the ecological z_goal-at-contact magnitude sits at ~0.4 +- noise on this substrate AND the
  per-episode env layout is OS-entropy random (default_rng(None)), so a >= 2/3 pass is NOT
  robustly reproducible by any already-available lever. If this run again self-routes
  contact_guard_unmet, that is a SUBSTRATE-CEILING finding on the foraging-competence axis (route
  to /implement-substrate), NOT a weakens of MECH-229.

SAME-STATISTIC WL NON-VACUITY READINESS GATE (the 514m gap fix, retained):
  Before scoring C_WL, assert -- on the SAME cross-target-inequality statistic C_WL routes on --
  that the wanting- and liking-targets are actually defined/written and CAN separate:
    leg 1 POSITIVE CONTROL (instrument-can-fire): a constructed 2-token IncentiveTokenBank
      (built from this run's GoalConfig) with drive-favored most-wanted DIFFERS from a designated
      last-consumed type -- must equal PC_SEPARATION_FLOOR (1.0).
    leg 2 IN-RUN POPULATION (channels-written-THIS-run): per guard-passing seed, the behavioural
      run's bank reached >= 2 distinct object tokens AND scored >= MIN_SCORED_STEPS consumption
      events with BOTH targets defined AND a per-axis-drive spread > DRIVE_SPREAD_FLOOR. Needs
      >= 2/3 seeds. Catches the exact 514m/514n channels-never-scored gap.
  readiness_met = (pc_separation_frac >= PC_SEPARATION_FLOOR) AND (run_populated_frac >= MIN_FRACTION).
  readiness NOT met -> substrate_not_ready_requeue (FAIL; MECH-229 non_contributory), NEVER a false
  weakens. This is the V3-EXQ-643 same-statistic readiness lesson applied to the WL DV.

PRE-REGISTERED GATES (in order; each self-routes substrate_not_ready_requeue if unmet, NEVER a
false MECH-229 weakens):
  (a) contact non-vacuity: guard >= 2/3 seeds.
  (b) WL readiness: leg-1 separation == 1.0 AND leg-2 in-run bank >= 2 distinct drive-differentiated
      tokens with n_scored_wl_steps >= MIN_SCORED_STEPS on >= 2/3 guard-passing seeds.
  (c) L9 ACCEPTANCE DV: mean object-bound wanting!=liking drive-coupled dissociation fraction
      >= WL_FRACTION (0.6) over guard-passing seeds.
  EXPERIMENT PASS = (a) AND (b) AND (c).

PER-CLAIM DIRECTION (single claim -> evidence_direction maps to MECH-229 directly):
  (a) NOT met -> non_contributory (substrate_not_ready_requeue; substrate-ceiling foraging axis).
  (b) NOT met -> non_contributory (substrate_not_ready_requeue).
  (a)+(b) met: MECH-229 = supports if (c) else weakens (a GENUINE weakens: the agent forages,
      the bank is populated, the instrument separates on the positive control, yet the object-bound
      drive-coupled wanting-target does not dissociate from the consumed type in behaviour).

FALSIFIER OFF-RAMP (pre-registered; per substrate_queue SD-049-PHASE-2 ready_blocked_by row-6
Woo/Spelke branch): if (a)+(b) are met but (c) FAILs (a genuine weakens), the disambiguating
follow-on is the OFF/control arm (SD-049 identity / bank disabled -> wanting structurally ==
liking -> WL ~ 0; a clean negative anchor) AND the OVERSHOOT arm (n_resource_types=5). JOINT
failure across the SD-049-on AND overshoot arms routes MECH-229 to substrate_conditional with a
V4-1 multi-agent-ecology dependency -- a clean exit, not endless iteration. This run is the
SD-049-on arm; the off/control + overshoot are queued only if THIS run produces a genuine weakens
(avoids burning ~2x curriculum builds while gate (a) is substrate-marginal and likely to
self-route). The routing is recorded in interpretation.falsifier_offramp.

claim_ids: MECH-229 (ONLY -- MECH-230 is NOT re-tagged; its structured-z_goal support is already
  recorded by 514m, and this run does not test that claim).
experiment_purpose: evidence
predecessor: V3-EXQ-514n (NOT a supersede -- 514n is non_contributory, not invalid-by-bug in any
  arm this run re-tests).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking goal-pipeline onboarding scheduler).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.goal import IncentiveTokenBank  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _sense_with_optional_harm,
    stage_plan,
)

EXPERIMENT_TYPE = "v3_exq_514o_sd049_phase2_mech229_object_bound_wanting_liking"
QUEUE_ID = "V3-EXQ-514o"
CLAIM_IDS: List[str] = ["MECH-229"]
EXPERIMENT_PURPOSE = "evidence"
PREDECESSOR = "V3-EXQ-514n (successor, NOT supersede)"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_SD049_PHASE2_OBJECT_BOUND_WL"

# --- Goal-pipeline / encoder dims (mirror 603n / 514n exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- Curriculum budgets (mirror 603n / 514n exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15            # the 603n-canonical contact-guard measurement (run_p2)
BEHAV_EVAL_EPISODES = 15  # the SD-049 Phase-2 object-bound WL measurement (this script)
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 603n / 514n) ---
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (mirror 603n / 514n) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- Pre-registered acceptance thresholds (NOT derived from the run) ---
P2_ZGOAL_GATE = 0.4          # per-seed contact-guard: z_goal_norm_at_contact_peak floor (603n G3)
CONTACT_GATE = 0.0           # per-seed contact-guard: P2 contact_rate floor (603n G2)
MIN_FRACTION = 2.0 / 3.0     # >= 2/3 seeds for non-vacuity + any aggregate gate
WL_FRACTION = 0.6            # C_WL: L9 acceptance object-bound wanting!=liking dissociation floor

# WL same-statistic non-vacuity readiness gate
PC_SEPARATION_FLOOR = 1.0    # leg 1: the constructed positive control MUST separate (deterministic)
MIN_SCORED_STEPS = 5         # leg 2: min consumption events with both targets defined per seed
DRIVE_SPREAD_FLOOR = 1e-3    # leg 2: min per-axis-drive spread for a genuine drive-differentiated test


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    """The FULL V3-EXQ-603n lever stack, ported verbatim (the substrate-readiness config that
    flipped scaffolded_sd054_onboarding ready=true 2026-06-11)."""
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        # developmental-window / consolidation amend (2026-06-03b)
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        # 634c seeding calibration (2026-06-03c)
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        # foraging-competence residual amend (2026-06-05)
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        # SD-057 cue-recall bridge (wean-to-wild contact lever; enables SD-049 in envs)
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # curriculum-decomposition amend (2026-06-07): isolated Stage-H
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver (mirror 603n)
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE: feed the env harm stream so z_harm / z_harm_a populate
        scaffold_feed_harm_stream=True,
        # harm-pathway training (2026-06-09 amend; ON, validated by 603k/603n)
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        # SD-057 object-bound incentive-salience layer (the substrate under test)
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
    )
    cfg.latent.use_resource_encoder = True   # SD-015 (z_resource -> bank L2 bind requires it)
    return cfg


def _ben_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    b = obs_body.reshape(-1)
    benefit = float(b[11].item()) if b.shape[0] > 11 else 0.0
    energy = float(b[3].item()) if b.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _per_axis_drive_from_obs(obs_dict: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    pad = obs_dict.get("per_axis_drive", None)
    if pad is None:
        return None
    if hasattr(pad, "to"):
        return pad.to(device)
    return torch.as_tensor(np.asarray(pad, dtype=np.float32), device=device)


def _consumed_type_tag_from_info(info: Dict[str, Any]) -> Optional[int]:
    """The 681-C4 fix: the authoritative consumed (liking) tag is in the info dict returned by
    env.step(), NOT the obs_dict (which carries the post-consumption-cleared cell at eval
    read-time). > 0 = a genuine consumption of that type this tick; None otherwise."""
    if not isinstance(info, dict):
        return None
    raw = info.get("sd049_consumed_type_tag_this_tick", 0)
    try:
        tag = int(raw[0] if hasattr(raw, "__len__") else raw)
    except (TypeError, ValueError):
        return None
    return tag if tag > 0 else None


def _positive_control_separation(agent, device: torch.device) -> float:
    """SAME-STATISTIC readiness leg 1: build a 2-token IncentiveTokenBank from THIS run's
    GoalConfig and verify the cross-target inequality the C_WL criterion routes on CAN fire.
    Two deterministic probes: drive-favored most-wanted must differ from a designated
    last-consumed type. Returns the cross-target-inequality fraction (1.0 if the instrument
    works; < 1.0 means the wanting!=liking instrument is structurally degenerate)."""
    goal_cfg = agent.config.goal
    d = int(getattr(goal_cfg, "goal_dim", WORLD_DIM))
    pc_bank = IncentiveTokenBank(goal_cfg, device)
    z_a = torch.zeros(1, d, device=device); z_a[0, 0] = 1.0   # type 1 identity embedding
    z_b = torch.zeros(1, d, device=device); z_b[0, 1] = 1.0   # type 2 identity embedding
    pc_bank.update(1, 1.0, z_a)   # bind food token (tag 1)
    pc_bank.update(2, 1.0, z_b)   # bind water token (tag 2)
    # probe 1: drive favours type 2 -> most-wanted should be 2; designated last-consumed = 1.
    pad1 = torch.zeros(1, N_RESOURCE_TYPES, device=device); pad1[0, 1] = 1.0
    mw1 = pc_bank.most_wanted(per_axis_drive=pad1, scalar_drive=1.0)
    sep1 = 1.0 if (mw1 is not None and int(mw1[0]) != 1) else 0.0
    # probe 2: drive favours type 1 -> most-wanted should be 1; designated last-consumed = 2.
    pad2 = torch.zeros(1, N_RESOURCE_TYPES, device=device); pad2[0, 0] = 1.0
    mw2 = pc_bank.most_wanted(per_axis_drive=pad2, scalar_drive=1.0)
    sep2 = 1.0 if (mw2 is not None and int(mw2[0]) != 2) else 0.0
    return float((sep1 + sep2) / 2.0)


def _run_object_bound_wl_eval(agent, scaffold_cfg, device: torch.device, n_eps: int) -> Dict[str, Any]:
    """SD-049 Phase-2 OBJECT-BOUND wanting!=liking measurement. Frozen policy (no optimizer
    steps). At each genuine consumption event (info consumed tag > 0): bind the per-object token
    + seed z_goal via agent.update_z_goal (the substrate's own L2/L4 path), then read the
    MECH-346 most-wanted pointer (wanting target) and compare its type to the just-consumed type
    (liking target, from info). A step is SCORED only when both targets are defined and the bank
    holds >= 2 distinct object tokens. WL_nodrive (drive-uniform most-wanted) is the
    drive-dependence diagnostic control."""
    env = _build_env(scaffold_cfg, "p2")
    env.reset()
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode

    bank = getattr(agent.goal_state, "incentive_bank", None)

    cosine_sims: List[float] = []
    scored_steps = 0
    wl_dissoc_steps = 0          # drive-coupled wanting != liking
    wl_nodrive_dissoc_steps = 0  # drive-uniform wanting != liking (control)
    contact_steps = 0
    total_steps = 0
    distinct_tokens_max = 0
    drive_spread_max = 0.0

    for _ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps_per_ep):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device,
                    scaffold_cfg.scaffold_feed_harm_stream,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            # goal_resource_r diagnostic (cosine z_goal vs z_resource at goal-active steps)
            goal_state = getattr(agent, "goal_state", None)
            if goal_state is not None and goal_state.is_active():
                with torch.no_grad():
                    z_g = goal_state.z_goal
                    if z_g.ndim == 1:
                        z_g = z_g.unsqueeze(0)
                    z_ref = latent.z_resource if latent.z_resource is not None else latent.z_world
                    if z_ref.ndim == 1:
                        z_ref = z_ref.unsqueeze(0)
                    if z_ref.shape[-1] == z_g.shape[-1]:
                        cosine_sims.append(
                            float(F.cosine_similarity(z_g, z_ref, dim=-1).mean().item())
                        )

            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            total_steps += 1

            # --- post-step consumption handling: the 681-C4 liking source is the INFO tag ---
            consumed_tag = _consumed_type_tag_from_info(info)   # liking target (last-consumed type)
            benefit, drive = _ben_drive(obs_dict["body_state"].to(device))
            if consumed_tag is not None:
                contact_steps += 1
                pad = _per_axis_drive_from_obs(obs_dict, device)
                # set the cached per-axis drive the substrate reads (mirrors scheduler/cue path)
                if pad is not None:
                    agent._per_axis_drive = pad.reshape(-1)
                    flat = pad.reshape(-1)
                    if flat.numel() >= 2:
                        spread = float(flat.max().item() - flat.min().item())
                        drive_spread_max = max(drive_spread_max, spread)
                with torch.no_grad():
                    # L2 bind + L4 seed (the substrate's own object-bound path), bound to the
                    # AUTHORITATIVE consumed type so the right token revalues.
                    try:
                        agent.update_z_goal(float(benefit), drive_level=float(drive),
                                            resource_type=consumed_tag)
                    except TypeError:
                        agent.update_z_goal(float(benefit), drive_level=float(drive))
                    # WANTING target = MECH-346 most-wanted pointer (the object that seeds z_goal)
                    if bank is not None and not bank.is_empty():
                        pad_drive = getattr(agent, "_per_axis_drive", None)
                        n_distinct = len(bank.wanting(per_axis_drive=pad_drive, scalar_drive=float(drive)))
                        distinct_tokens_max = max(distinct_tokens_max, n_distinct)
                        mw = bank.most_wanted(per_axis_drive=pad_drive, scalar_drive=float(drive))
                        mw_nd = bank.most_wanted(per_axis_drive=None, scalar_drive=float(drive))
                        # SCORE only when wanting + liking defined AND bank holds >= 2 distinct tokens
                        if mw is not None and n_distinct >= 2:
                            scored_steps += 1
                            if int(mw[0]) != int(consumed_tag):
                                wl_dissoc_steps += 1
                            if mw_nd is not None and int(mw_nd[0]) != int(consumed_tag):
                                wl_nodrive_dissoc_steps += 1

            if done:
                break

    wl_fraction = (float(wl_dissoc_steps) / float(scored_steps)) if scored_steps > 0 else 0.0
    wl_nodrive_fraction = (float(wl_nodrive_dissoc_steps) / float(scored_steps)) if scored_steps > 0 else 0.0
    goal_resource_r = float(np.mean(cosine_sims)) if cosine_sims else 0.0
    behav_contact_rate = (float(contact_steps) / float(total_steps)) if total_steps > 0 else 0.0
    run_populated = bool(
        distinct_tokens_max >= 2
        and scored_steps >= MIN_SCORED_STEPS
        and drive_spread_max > DRIVE_SPREAD_FLOOR
    )

    return {
        "object_bound_wl_dissoc_fraction": wl_fraction,
        "wl_nodrive_dissoc_fraction": wl_nodrive_fraction,
        "n_scored_wl_steps": scored_steps,
        "n_wl_dissoc_steps": wl_dissoc_steps,
        "n_wl_nodrive_dissoc_steps": wl_nodrive_dissoc_steps,
        "distinct_tokens_max": distinct_tokens_max,
        "drive_spread_max": drive_spread_max,
        "run_bank_populated": run_populated,
        "goal_resource_r": goal_resource_r,
        "n_cosine_samples": len(cosine_sims),
        "behav_contact_rate": behav_contact_rate,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "object_bound_wl_dissoc_fraction": 0.0, "wl_nodrive_dissoc_fraction": 0.0,
        "n_scored_wl_steps": 0, "n_wl_dissoc_steps": 0, "n_wl_nodrive_dissoc_steps": 0,
        "distinct_tokens_max": 0, "drive_spread_max": 0.0,
        "run_bank_populated": False,
        "goal_resource_r": 0.0, "behav_contact_rate": 0.0,
        "pc_separation_frac": 0.0,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")

    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    # --- Curriculum build: Stage-0 -> Stage-0b -> P0 -> Stage-H -> P1 (full 603n stack) ---
    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}"
          f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" median_last={hz.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    # --- 603n-canonical contact guard via run_p2 (consumption-event-gated readout) ---
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    # --- SD-057 object-bound WL DV (always measured; gated at aggregation) ---
    behav = _run_object_bound_wl_eval(agent, scaffold_cfg, device, BEHAV_EVAL_EPISODES)
    done += BEHAV_EVAL_EPISODES
    print(f"  [train] wl_eval seed={seed} ep {done}/{total_eps}"
          f" wl_frac={behav['object_bound_wl_dissoc_fraction']:.3f}"
          f" wl_nodrive={behav['wl_nodrive_dissoc_fraction']:.3f}"
          f" scored={behav['n_scored_wl_steps']}"
          f" distinct_tokens={behav['distinct_tokens_max']}"
          f" run_populated={behav['run_bank_populated']}"
          f" goal_r={behav['goal_resource_r']:.3f}", flush=True)

    print(f"verdict: {'PASS' if guard_pass else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass}"
          f" (contact_rate={p2.contact_rate:.4f} z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})",
          flush=True)

    rec: Dict[str, Any] = {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
    }
    rec.update(behav)
    # SAME-STATISTIC readiness leg 1: positive control built from THIS agent's GoalConfig.
    rec["pc_separation_frac"] = _positive_control_separation(agent, device)
    return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + BEHAV_EVAL_EPISODES
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + BEHAV_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # --- WL same-statistic non-vacuity readiness gate ---
    pc_vals = [r.get("pc_separation_frac", 0.0) for r in guard_passing]
    pc_separation_frac = float(np.mean(pc_vals)) if pc_vals else 0.0
    pc_ok = bool(pc_separation_frac >= PC_SEPARATION_FLOOR)
    run_populated_flags = [bool(r.get("run_bank_populated", False)) for r in guard_passing]
    run_populated_frac = _frac(run_populated_flags)
    run_populated_ok = bool(run_populated_frac >= MIN_FRACTION)
    wl_readiness_met = bool(pc_ok and run_populated_ok)

    # --- Aggregate the WL DV over guard-passing seeds ONLY ---
    def _mean(key: str) -> float:
        vals = [r[key] for r in guard_passing]
        return float(np.mean(vals)) if vals else 0.0

    mean_wl = _mean("object_bound_wl_dissoc_fraction")
    mean_wl_nodrive = _mean("wl_nodrive_dissoc_fraction")
    mean_goal_r = _mean("goal_resource_r")
    n_scored_total = int(sum(r["n_scored_wl_steps"] for r in guard_passing))

    c_wl = bool(mean_wl >= WL_FRACTION)

    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        route_reason = "contact_guard_unmet"
    elif not wl_readiness_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        route_reason = ("wl_positive_control_degenerate" if not pc_ok
                        else "wl_channels_not_written_this_run")
    else:
        outcome = "PASS" if c_wl else "FAIL"
        readiness_route = "mech229_object_bound_dissociation" if c_wl else "residual_wl_open_falsifier"
        evidence_direction = "supports" if c_wl else "weakens"
        route_reason = "c_wl_met_l9_closed" if c_wl else "c_wl_unmet_genuine_weakens_run_offarm_overshoot"

    # Non-degeneracy scoring net (2026-06-11 helper convention): the WL DV got a fair, non-vacuous
    # test only when the same-statistic readiness gate is met. A self-routed run is already
    # non_contributory (scoring-excluded), but flag non_degenerate=False as belt-and-suspenders so
    # a future "weakens" can never score off a structurally-pinned/unwritten channel (the 514m class).
    gates_ab_met = bool(contact_non_vacuity_met and wl_readiness_met)
    non_degenerate = bool(gates_ab_met)
    degeneracy_reason = ("" if non_degenerate
                         else f"WL DV not non-vacuously testable: {route_reason} "
                              f"(contact_non_vacuity={contact_non_vacuity_met}, wl_readiness={wl_readiness_met})")

    print(f"[{EXPERIMENT_TYPE}] contact_non_vacuity={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) wl_readiness={wl_readiness_met}"
          f" (pc={pc_separation_frac:.3f} run_pop={run_populated_frac:.3f})"
          f" C_WL={c_wl} (mean_wl={mean_wl:.3f} nodrive={mean_wl_nodrive:.3f})"
          f" -> outcome={outcome} route={readiness_route}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] per_claim MECH-229={evidence_direction}"
          f" non_degenerate={non_degenerate}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "wl_readiness_met": wl_readiness_met,
        "pc_separation_frac": pc_separation_frac,
        "run_bank_populated_frac": run_populated_frac,
        "C_WL_object_bound_wanting_neq_liking": c_wl,
        "mean_object_bound_wl_dissoc_fraction": mean_wl,
        "mean_wl_nodrive_dissoc_fraction": mean_wl_nodrive,
        "wl_drive_minus_nodrive": float(mean_wl - mean_wl_nodrive),
        "n_scored_wl_steps_total": n_scored_total,
        "mean_goal_resource_r": mean_goal_r,
        "overall_pass": bool(contact_non_vacuity_met and wl_readiness_met and c_wl),
        "per_seed_guard_pass": guard_flags,
        "route_reason": route_reason,
    }

    c_wl_non_degenerate = bool(wl_readiness_met and n_scored_total >= MIN_SCORED_STEPS)

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "preconditions": [
                {
                    "name": "wl_instrument_positive_control_separates",
                    "description": "leg 1: a constructed 2-token IncentiveTokenBank (this run's "
                                   "GoalConfig) yields a drive-favored most-wanted that DIFFERS "
                                   "from a designated last-consumed type -- the SAME cross-target "
                                   "inequality statistic C_WL routes on, on a positive control.",
                    "control": "2 deterministic probes: drive-favored most-wanted vs designated "
                               "last-consumed (instrument-can-fire).",
                    "measured": pc_separation_frac,
                    "threshold": PC_SEPARATION_FLOOR,
                    "met": pc_ok,
                },
                {
                    "name": "run_bank_populated_two_tokens_differing_drive",
                    "description": "leg 2: per guard-passing seed the behavioural run's bank reached "
                                   ">= 2 distinct object tokens AND scored >= MIN_SCORED_STEPS "
                                   "consumption events (info consumed tag > 0) with both targets "
                                   "defined AND per-axis-drive spread > floor (channels-written-this-"
                                   "run -- the exact 514n n_scored=0 gap).",
                    "control": "fraction of guard-passing seeds with a populated, drive-differentiated "
                               "bank at consumption events.",
                    "measured": run_populated_frac,
                    "threshold": MIN_FRACTION,
                    "met": run_populated_ok,
                },
            ],
            "criteria": [
                {
                    "name": "C_WL_object_bound_wanting_neq_liking",
                    "load_bearing": True,
                    "passed": c_wl,
                },
            ],
            "criteria_non_degenerate": {
                "C_WL": c_wl_non_degenerate,
            },
            "contact_guard": {
                "definition": "per-seed: P2 contact_rate > 0 AND z_goal_norm_at_contact_peak > 0.4 "
                              "(603n G2 + G3). A seed failing the guard is excluded from DV "
                              "aggregation; < 2/3 seeds passing -> substrate_not_ready_requeue, "
                              "never a false weakens. 603n itself cleared G3 only 2/3; the leg is "
                              "substrate-marginal (~0.4 +- noise, env layout default_rng(None)).",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
            "wl_readiness_gate": {
                "definition": "SAME-STATISTIC non-vacuity for the WL DV (the 514m/514n gap fix): leg 1 "
                              "positive-control cross-target separation >= 1.0 AND leg 2 in-run bank "
                              "populated >= 2 distinct tokens at differing per-axis drives on >= 2/3 "
                              "guard-passing seeds. Below floor -> substrate_not_ready_requeue "
                              "(non_contributory), NEVER a false MECH-229 weakens.",
                "pc_separation_floor": PC_SEPARATION_FLOOR,
                "min_scored_steps": MIN_SCORED_STEPS,
                "drive_spread_floor": DRIVE_SPREAD_FLOOR,
                "min_fraction": MIN_FRACTION,
            },
            "drive_dependence_control": {
                "definition": "WL_nodrive (drive-uniform most-wanted, per_axis_drive=None) is the "
                              "within-eval drive-dependence control: a genuine MECH-229 drive-coupled "
                              "dissociation has WL_drive >> WL_nodrive. Reported as the diagnostic "
                              "wl_drive_minus_nodrive; not a load-bearing gate this run.",
                "mean_wl_drive": mean_wl,
                "mean_wl_nodrive": mean_wl_nodrive,
            },
            "falsifier_offramp": {
                "definition": "Pre-registered Woo/Spelke off-ramp (substrate_queue SD-049-PHASE-2 "
                              "row-6). If (a)+(b) met but (c) FAILs (genuine weakens), the "
                              "disambiguating follow-on is an OFF/control arm (SD-049 identity / bank "
                              "disabled -> wanting structurally == liking -> WL ~ 0) AND an OVERSHOOT "
                              "arm (n_resource_types=5). JOINT failure across the SD-049-on AND "
                              "overshoot arms routes MECH-229 -> substrate_conditional with a V4-1 "
                              "multi-agent-ecology dependency (clean exit). This run is the SD-049-on "
                              "arm; off/control + overshoot are queued only on a genuine weakens here.",
                "triggers_on": "route_reason == c_wl_unmet_genuine_weakens_run_offarm_overshoot",
                "joint_failure_routes_to": "substrate_conditional (MECH-229, dependency V4-1)",
            },
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (FULL 603n curriculum: Stage-0 -> Stage-0b -> P0 -> "
                     "Stage-H -> P1 -> P2; harm-pathway training ON; ready=true 2026-06-11) + SD-049 "
                     "Phase-2 hybrid encoder (SD-015 z_resource) + SD-057 object-bound incentive-"
                     "salience layer (use_incentive_token_bank; MECH-344/345/346).",
        "condition": CONDITION_LABEL,
        "predecessor": PREDECESSOR,
        "method_note": "MECH-229 wanting!=liking measured as the SD-057 OBJECT-BOUND wanting-target "
                       "(MECH-346 most-wanted z_goal pointer = IncentiveTokenBank.most_wanted k*) != "
                       "liking-target (last-consumed type) dissociation. THE 681-C4 FIX: the liking "
                       "target AND the L2-bind resource_type are sourced from "
                       "info['sd049_consumed_type_tag_this_tick'] returned by env.step() (the "
                       "authoritative consumption signal), NOT _contacted_resource_type(obs_dict), "
                       "which at eval read-time reads the post-consumption-cleared cell and returns "
                       "None (the 514n n_scored_wl_steps=0 cause; V3-EXQ-681 C4 established "
                       "consumed_tag_in_obs_dict=false / consumed_tag_in_info=true). At each "
                       "consumption event the substrate's own L2 bind + L4 seed "
                       "(agent.update_z_goal(resource_type=...)) runs, then the most-wanted pointer "
                       "(wanting target) is compared to the consumed type (liking target). z_goal "
                       "forms ECOLOGICALLY via the full 603n curriculum.",
        "readiness_note": "Retains the same-statistic WL non-vacuity readiness gate: leg 1 "
                          "positive-control cross-target separation (instrument-can-fire) + leg 2 "
                          "in-run bank populated >= 2 distinct tokens at differing per-axis drives "
                          "(channels-written-this-run). Below floor -> substrate_not_ready_requeue "
                          "(non_contributory), NEVER a false MECH-229 weakens. The positive control "
                          "reads the SAME cross-target-inequality statistic C_WL routes on "
                          "(V3-EXQ-643 lesson).",
        "contact_guard_note": "The full 603n lever stack is ported VERBATIM (_make_scaffold_cfg == "
                              "603n). The contact guard (z_goal@contact > 0.4) is substrate-marginal: "
                              "603n itself cleared G3 only 2/3 (seed44 0.374) and 514n got seed43 "
                              "0.391; the ecological magnitude sits at ~0.4 +- noise AND env layout "
                              "is default_rng(None) (OS-entropy), so a >= 2/3 pass is not robustly "
                              "reproducible by any already-available lever. A contact_guard_unmet "
                              "self-route is a SUBSTRATE-CEILING finding on the foraging-competence "
                              "axis (route /implement-substrate), NOT a MECH-229 weakens.",
        "claim_tag_note": "claim_ids=[MECH-229] ONLY. MECH-230 is NOT re-tagged: its structured-z_goal "
                          "support is already recorded by 514m and this run does not test that claim.",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "wl_fraction": WL_FRACTION,
            "pc_separation_floor": PC_SEPARATION_FLOOR,
            "min_scored_steps": MIN_SCORED_STEPS,
            "drive_spread_floor": DRIVE_SPREAD_FLOOR,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "behav_eval_episodes": BEHAV_EVAL_EPISODES,
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "use_incentive_token_bank": True,
            "scaffold_train_harm_pathway": True,
            "config_basis": "V3-EXQ-603n (full lever stack ported verbatim; the substrate-readiness "
                            "run that flipped scaffolded_sd054_onboarding ready=true)",
        },
        "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
        )
