#!/opt/local/bin/python3
"""
V3-EXQ-228c -- ARC-032: Theta-Rate Pathway DIRECT-READOUT test, measured at
E3-tick granularity on the scaffolded_sd054_onboarding curriculum
(measurement-redesign continuation of the V3-EXQ-228/228a/228b lineage).

Claims: ARC-032
EXPERIMENT_PURPOSE = "evidence"
Supersedes: V3-EXQ-228b

=== WHY THIS REDESIGN (228b -> 228c) ===

V3-EXQ-228b (2026-08-09) was the FIRST run in the whole 076/228/247/228a/228b
lineage to clear ARC-032's non-degeneracy precondition (goal_norm mean 0.469,
1.00 frac seeds) -- the theta-patch was confirmed genuinely wired into
trajectory candidate generation and non-inert at both smoke and full scale.
But its resource-collection-lift readout FAILED 0/3 seeds with a small,
consistently NEGATIVE effect (lift_mean=-0.0021). The failure_autopsy
(failure_autopsy_V3-EXQ-228b_2026-08-09) did NOT accept this as falsifying,
for two measurement-adequacy reasons this redesign fixes:

  (a) E3-TICK CACHE-GATING. generate_trajectories() only regenerates candidates
      from a fresh theta summary when e3_tick fires (MECH-089/EXQ-052b:
      e3_tick_ratio ~= 0.109, ~1 in 9 steps). The remaining ~89% of a 200-step
      episode REUSE cached candidates unaffected by the theta ablation
      (agent.py generate_trajectories: `if not ticks["e3_tick"] and
      self._committed_candidates is not None: return self._committed_candidates`).
      A full-episode aggregate readout (228b's resource_visit_rate over all
      steps) therefore dilutes any theta-specific behavioral signal ~9x.
  (b) WRONG DV. claims.yaml's own CONFIRMING criterion names z_goal PERSISTENCE
      and goal-PROXIMITY-SCORE NOISE in E3's trajectory scoring as the decisive
      readouts. 228b measured neither directly -- resource-collection lift is a
      downstream behavioral proxy, not the pre-registered DV.

228c keeps 228b's PROVEN substrate + training path byte-identical (the only
lineage member to clear the precondition) and changes ONLY the measurement
protocol, to score the claim's OWN pre-registered readouts directly, restricted
to the E3-tick steps where the ablation actually acts.

=== ABLATION (unchanged from 228a/228b -- pre-registered) ===

claims.yaml what_would_answer sanctions exactly this ablation: "a genuine
theta-bypass ablation condition (E1 output -> E3 direct, bypassing MECH-089
ThetaBuffer -- as designed in V3-EXQ-228a: THETA_ACTIVE vs THETA_ZEROED)".
THETA_ZEROED patches theta_buffer.summary() to always return zeros, severing
the E1 -> theta_buffer -> E3 pathway that feeds z_world_for_e3 to _e3_tick's
proposer + goal appraisals.

=== ARCHITECTURE (unchanged from 228b -- shared training, deepcopy split) ===

ONE agent trained per seed through the full curriculum (Stage0 -> Stage0b -> P0
-> Stage-H -> P1), using scaffolded_sd054_onboarding's own FULL agent config
verbatim (the config that clears the z_goal precondition). Immediately after P1,
before any measurement, the trained agent is deep-copied into two clones with
byte-identical weights + goal_state (agent.reset() does NOT clear z_goal). One
clone's theta_buffer.summary() is patched to zeros (THETA_ZEROED); the other is
untouched (THETA_ACTIVE). RNG is reset once at training start, and again (same
seed) before EACH condition's P2 measurement, so both conditions see the same
env resets and action-sampling draws -- the theta patch is the only source of
divergence in the measurement rollout.

Per the arm-fingerprint contract's escape hatch: both cells share the SAME
trained agent object via deepcopy, so both are stamped reuse_eligible=False.

=== MEASUREMENT PROTOCOL (REDESIGNED -- the 228b->228c change) ===

Frozen-policy (agent.eval(), no gradients) rollout on a P2-phase env, action
selection at temperature=0.5, update_z_goal() every step. At EVERY step we
record whether ticks["e3_tick"] fired. The theta-sensitive readouts below are
computed ONLY on E3-tick steps -- the ~11% of steps where the ablation is not
diluted by cached-candidate reuse (autopsy finding (a)).

PRIMARY readouts (claims.yaml's pre-registered CONFIRMING DVs -- autopsy (b)):

  C1  z_goal PERSISTENCE. On each E3-tick step we snapshot agent.goal_state.
      z_goal. persistence_cos = the mean lag-1 cosine similarity of z_goal
      between consecutive E3 ticks within an episode, averaged over episodes.
      Higher = the goal representation is held more stably across the E3 cadence
      (~10 env steps apart). ARC-032 predicts THETA_ACTIVE > THETA_ZEROED (theta
      packaging "provides smoothing that stabilises z_goal influence"). Also
      recorded: zgoal_norm_e3 (mean z_goal norm on E3 ticks) as a secondary
      persistence facet.

  C2  goal-PROXIMITY-SCORE NOISE. On each E3-tick step we compute the
      per-candidate goal_proximity of the freshly-generated trajectory
      candidates (GoalState.goal_proximity over each candidate's first-step
      z_world summary; the same [K, world_dim] -> [K] read the substrate itself
      uses at agent.py:7091). prox_noise_temporal = the std, across E3 ticks, of
      the LEAD candidate's goal_proximity (candidates[0], the proposer's lead --
      the score E3 acts on). Higher = the winning goal-proximity score jitters
      more over time. ARC-032 predicts THETA_ACTIVE < THETA_ZEROED (theta
      smoothing -> less noisy scores). We choose the TEMPORAL statistic as
      primary because theta's function IS temporal smoothing (a flat mean over a
      fixed 10-step window); the CROSS-CANDIDATE dispersion (prox_noise_xcand,
      mean over E3 ticks of the per-tick std across candidates) is recorded as a
      secondary facet and, deliberately, DISCRIMINATES the monostrategy-collapse
      alternative: if THETA_ZEROED collapses candidate diversity, xcand noise
      would FALL under bypass (opposite of the temporal prediction), which the
      raw per-condition values make visible rather than hiding.

INFORMATIONAL readout (228b's behavioral proxy, kept for continuity, NOT
gating): resource_visit_rate / harm_rate, recorded both full-episode
(228b-comparable) AND restricted to E3-tick steps.

Precondition (unchanged): after P1, before the deepcopy split, the single
trained agent's goal_norm (compute_goal_maintenance_diagnostic()['goal_norm'])
must clear GOAL_NORM_THRESH (0.05). One check per seed covers both clones.

=== PRE-REGISTERED CRITERIA ===

Precondition: goal_norm >= GOAL_NORM_THRESH (0.05) in >= 2/3 seeds.
C1 (primary, persistence): persistence_cos_active - persistence_cos_zeroed
    >= PERSIST_MARGIN (0.02) in >= 2/3 seeds.
C2 (primary, noise): prox_noise_temporal_zeroed - prox_noise_temporal_active
    >= NOISE_MARGIN (0.005) in >= 2/3 seeds  (theta ACTIVE is the quieter arm).
C3 (informational): E3-tick resource lift (active - zeroed), reported, not gating.

PASS = precondition met AND C1 AND C2, majority >= 2/3 seeds each.

NOTE ON THRESHOLDS: PERSIST_MARGIN / NOISE_MARGIN are exploratory-magnitude
absolute floors chosen ahead of the FIRST direct measurement of these DVs
(no prior run recorded them; the readouts are not recoverable by reanalysis --
GOV-REUSE-1 checked). The primary scientific value is the DIRECTION on the
claim's own named readouts plus the direct-readout record itself. Per the
autopsy's explicit instruction, a CONSISTENT REVERSED direction (theta ACTIVE
worse -- less persistent and/or noisier) is registered as a finding to
investigate, not dismissed: it would converge with MECH-089's own already-
confirmed EXQ-066/EXQ-122 result that static/uniform theta-averaging measurably
HURTS E3's fine-grained discrimination -- implicating the flat-mean "packaging"
implementation rather than the frontal-hippocampal-synchrony hypothesis itself.

Evidence interpretation:
  precondition FAIL             -> non_contributory (substrate_limitation)
  precond met, C1 & C2 pass     -> supports (theta packaging stabilises z_goal
                                   maintenance AND quiets goal-proximity scores)
  precond met, exactly one pass -> mixed (partial support on one named readout)
  precond met, neither pass     -> does_not_support (reversed-trend flag set if
                                   theta ACTIVE is consistently the worse arm)

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

KNOWN SUBSTRATE LIMITATION AT RUN TIME: substrate_queue SD-E3-SCORER-COMPLETION
(severity: degrading) is open against ree_core/predictors/e3_selector.py, which
this driver's select_action path exercises. Recorded here per the
substrate-path-overlap gate so a later autopsy sees the run happened under a
known (degrading, non-corrupting) limitation.
"""

import argparse
import copy
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import reset_all_rng, compute_arm_fingerprint
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator

from experiments.scaffolded_sd054_onboarding import (
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _benefit_and_drive,
    _sense_with_optional_harm,
)

# --------------------------------------------------------------------- #
# Experiment metadata
# --------------------------------------------------------------------- #
EXPERIMENT_TYPE = "v3_exq_228c_arc032_theta_bypass_readout"
QUEUE_ID = "V3-EXQ-228c"
SUPERSEDES = "V3-EXQ-228b"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["ARC-032"]
EXPERIMENT_PURPOSE = "evidence"

# Seeds: NOT 44 -- confirmed recurring per-seed instability (early episode
# death ~step 40) on reef-config envs (EXQ-539-540, V3-EXQ-538a autopsies).
SEEDS = [42, 43, 45]
CONDITIONS = ["THETA_ACTIVE", "THETA_ZEROED"]

# --- Curriculum budgets (866a's validated Stage0/0b/P0/Stage-H/P1 budgets,
#     reused verbatim from 228b -- the exact recipe proven to clear
#     z_goal_norm >= 0.05 on this substrate) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 30
STEPS_PER_EP = 200
P0_NUM_HAZARDS = 1
TOTAL_TRAIN_EPS = (
    STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET + P1_BUDGET
)  # 220 -- shared training, once per seed
TOTAL_EPS_PER_CONDITION = TOTAL_TRAIN_EPS + P2_BUDGET  # 250 -- progress denominator M

# Dry-run (smoke) budgets.
DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_P1, DRY_P2, DRY_STEPS = 2, 2, 5, 5, 5, 8, 30
DRY_TOTAL_TRAIN_EPS = DRY_STAGE0 + DRY_STAGE0B + DRY_P0 + DRY_HAZARD + DRY_P1
DRY_TOTAL_EPS_PER_CONDITION = DRY_TOTAL_TRAIN_EPS + DRY_P2

# --- Stage-H regime (866a's validated anchor, same as 603q) ----------------
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- Seeding calibration + cue-recall bridge (mirror 866a / 603q) ----------
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal ---------------------------
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# --- Harm-pathway training (603k) + stabilization (603q amend) --------------
HARM_PATHWAY_LR = 1e-3
HARM_PATHWAY_ENCODER_LR = 3e-4
HARM_PATHWAY_WARMUP_STEPS = 250

# --- P2 measurement guard -- see 228b "MEASUREMENT PROTOCOL". --------------
P2_HFA_GUARD = 0.3

# --- Encoder / latent dims (mirror 866a) ------------------------------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
ALPHA_WORLD = 0.9
SELF_DIM = 32

# --- Pre-registered thresholds ---------------------------------------------
GOAL_NORM_THRESH = 0.05    # Precondition: z_goal genuinely seeded (unchanged)
PERSIST_MARGIN = 0.02      # C1: persistence_cos_active - _zeroed (cosine units)
NOISE_MARGIN = 0.005       # C2: prox_noise_temporal_zeroed - _active (theta quieter)
MIN_SEEDS_PASS = 2         # of 3 -- ">= 2/3 seeds"


# --------------------------------------------------------------------- #
# Scaffold + agent config builders (verbatim from 228b)
# --------------------------------------------------------------------- #

def build_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_P1, DRY_P2, DRY_STEPS)
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, STEPS_PER_EP)

    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_strict_goal_isolation=False,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
    )
    if steps < 75:
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def build_agent_config(env) -> REEConfig:
    """866a's FULL-arm config verbatim (the empirically validated config that
    clears the z_goal precondition on this curriculum). ARC-032 needs the
    goal/wanting channel live; the manipulation is entirely post-training in
    the theta patch, not in REEConfig."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
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
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_escape_affordance_bridge=False,
        use_escape_relief_credit=False,
        use_escape_safety_credit=False,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
        e2_action_contrastive_enabled=True,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        e1_goal_conditioned=True,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


# --------------------------------------------------------------------- #
# Theta ablation patch (228a/228b's patch, applied post-training)
# --------------------------------------------------------------------- #

def _patch_theta_zeroed(agent: REEAgent) -> None:
    """Patch theta_buffer.summary() to always return zeros, severing the
    E1 -> theta_buffer -> E3 pathway for trajectory proposals."""
    device = agent.device
    world_dim = agent.config.latent.world_dim

    def _zeroed_summary():
        return torch.zeros(1, world_dim, device=device)

    agent.theta_buffer.summary = _zeroed_summary


# --------------------------------------------------------------------- #
# Direct-readout helpers (the 228c redesign)
# --------------------------------------------------------------------- #

def _candidate_goal_proximities(
    agent: REEAgent, candidates: List, latent, device: torch.device
) -> Optional[torch.Tensor]:
    """Per-candidate goal_proximity in [0,1], shape [K]. Mirrors the substrate's
    own per-candidate read (agent.py:7075-7093): prefer the shared
    _candidate_world_summaries (e2.world_forward source), else build first-step
    z_world summaries from each candidate's rolled-out world_states, falling
    back to the current observed z_world when a candidate tracks none.
    goal_proximity expands z_goal over the [K, world_dim] input -> [K]."""
    if agent.goal_state is None or not candidates:
        return None
    summ = None
    try:
        summ = agent._candidate_world_summaries(candidates)
    except Exception:
        summ = None
    if summ is None:
        rows: List[torch.Tensor] = []
        for c in candidates:
            ws = c.get_world_state_sequence()  # [batch, horizon+1, world_dim] or None
            if ws is not None:
                rows.append(ws[0, 0, :])
            else:
                rows.append(latent.z_world[0].detach())
        if not rows:
            return None
        summ = torch.stack(rows, dim=0)
    with torch.no_grad():
        prox = agent.goal_state.goal_proximity(summ).detach().reshape(-1)
    return prox


def _mean_lag1_cosine(vecs_by_ep: List[List[torch.Tensor]]) -> Optional[float]:
    """Mean lag-1 cosine similarity of consecutive z_goal snapshots WITHIN each
    episode, averaged over all consecutive pairs across episodes. Higher = the
    goal representation is held more stably across the E3 cadence. Returns None
    when there are no consecutive pairs (fewer than 2 E3 ticks in every ep)."""
    cos_vals: List[float] = []
    for ep_vecs in vecs_by_ep:
        for a, b in zip(ep_vecs[:-1], ep_vecs[1:]):
            na = float(a.norm().item())
            nb = float(b.norm().item())
            if na < 1e-12 or nb < 1e-12:
                continue
            cos_vals.append(float((a @ b).item()) / (na * nb))
    if not cos_vals:
        return None
    return sum(cos_vals) / len(cos_vals)


def _pstd(xs: List[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


# --------------------------------------------------------------------- #
# P2 frozen-policy measurement -- REDESIGNED: E3-tick-restricted direct readouts
# --------------------------------------------------------------------- #

def _measure_p2(
    agent: REEAgent,
    env,
    device: torch.device,
    cfg: ScaffoldedSD054OnboardingConfig,
    n_episodes: int,
    steps_per_ep: int,
) -> Dict[str, Any]:
    agent.eval()

    # Full-episode behavioral (informational; 228b-comparable) -----------------
    resource_visits = 0
    harm_events = 0
    total_steps = 0
    zgoal_norms_full: List[float] = []

    # E3-tick-restricted captures (the redesign) ------------------------------
    e3_zgoal_by_ep: List[List[torch.Tensor]] = []   # z_goal snapshots per episode
    e3_lead_prox: List[float] = []                  # lead-candidate goal_proximity per E3 tick
    e3_xcand_prox_std: List[float] = []             # cross-candidate proximity std per E3 tick
    e3_zgoal_norms: List[float] = []                # z_goal norm on E3 ticks
    e3_resource_visits = 0
    e3_harm_events = 0
    n_e3_ticks = 0

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_e3_zgoal: List[torch.Tensor] = []

        for _step in range(steps_per_ep):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device,
                    cfg.scaffold_feed_harm_stream,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", True)
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=0.5)
            action_idx = int(action.argmax(dim=-1).item())

            benefit, drive = _benefit_and_drive(obs_dict["body_state"].to(device))
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
            if agent.goal_state is not None:
                zgoal_norms_full.append(float(agent.goal_state.z_goal.norm().item()))

            is_e3 = bool(ticks.get("e3_tick", False))
            if is_e3 and agent.goal_state is not None:
                n_e3_ticks += 1
                # z_goal snapshot for persistence (C1)
                zg = agent.goal_state.z_goal.detach().reshape(-1).clone()
                ep_e3_zgoal.append(zg)
                e3_zgoal_norms.append(float(zg.norm().item()))
                # per-candidate goal-proximity scores for noise (C2)
                prox = _candidate_goal_proximities(agent, candidates, latent, device)
                if prox is not None and prox.numel() >= 1:
                    e3_lead_prox.append(float(prox[0].item()))  # candidates[0] == proposer lead
                    if prox.numel() >= 2:
                        e3_xcand_prox_std.append(float(prox.std(unbiased=False).item()))

            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            if ttype in ("benefit_approach", "resource"):
                resource_visits += 1
                if is_e3:
                    e3_resource_visits += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1
                if is_e3:
                    e3_harm_events += 1
            total_steps += 1
            if done:
                break

        if ep_e3_zgoal:
            e3_zgoal_by_ep.append(ep_e3_zgoal)

        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            print(f"  [p2] ep {ep + 1}/{n_episodes}", flush=True)

    persistence_cos = _mean_lag1_cosine(e3_zgoal_by_ep)  # None if <2 ticks/ep everywhere
    prox_noise_temporal = _pstd(e3_lead_prox)            # std of lead prox across E3 ticks
    prox_noise_xcand = _mean(e3_xcand_prox_std)          # mean per-tick cross-candidate std

    return {
        # PRIMARY (E3-tick, direct readouts) --------------------------------
        "persistence_cos": persistence_cos if persistence_cos is not None else 0.0,
        "persistence_cos_measured": persistence_cos is not None,
        "prox_noise_temporal": prox_noise_temporal,
        "prox_noise_xcand": prox_noise_xcand,
        "zgoal_norm_e3_mean": _mean(e3_zgoal_norms),
        "lead_prox_e3_mean": _mean(e3_lead_prox),
        # E3-tick behavioral (informational) --------------------------------
        "e3_resource_visit_rate": e3_resource_visits / max(1, n_e3_ticks),
        "e3_harm_rate": e3_harm_events / max(1, n_e3_ticks),
        # Full-episode behavioral (informational; 228b-comparable) ----------
        "resource_visit_rate": resource_visits / max(1, total_steps),
        "harm_rate": harm_events / max(1, total_steps),
        "zgoal_norm_mean": _mean(zgoal_norms_full),
        # Auditability ------------------------------------------------------
        "n_e3_ticks": n_e3_ticks,
        "total_steps": total_steps,
        "e3_tick_ratio": n_e3_ticks / max(1, total_steps),
        "n_lead_prox_obs": len(e3_lead_prox),
        "n_resource_events": resource_visits,
        "n_harm_events": harm_events,
    }


# --------------------------------------------------------------------- #
# Per-seed run: shared training, deep-copy split, dual measurement
# --------------------------------------------------------------------- #

def _cell_config_slice(condition: str, seed: int, dry_run: bool) -> Dict:
    return {
        "condition": condition,
        "seed": seed,
        "dry_run": dry_run,
        "shared_training": True,
        "measurement": "e3_tick_restricted_direct_readouts",
        "stage0_budget": STAGE0_BUDGET,
        "stage0b_budget": STAGE0B_BUDGET,
        "p0_budget": P0_BUDGET,
        "hazard_budget": HAZARD_STAGE_BUDGET,
        "p1_budget": P1_BUDGET,
        "p2_budget": P2_BUDGET,
        "steps_per_ep": STEPS_PER_EP,
        "goal_norm_thresh": GOAL_NORM_THRESH,
        "persist_margin": PERSIST_MARGIN,
        "noise_margin": NOISE_MARGIN,
        "p2_hfa_guard": P2_HFA_GUARD,
    }


def _run_seed(seed: int, zg: ZGoalStreamAccumulator, dry_run: bool) -> Dict[str, Any]:
    device = torch.device("cpu")
    sched_cfg = build_scaffold_cfg(dry_run)

    total_train = DRY_TOTAL_TRAIN_EPS if dry_run else TOTAL_TRAIN_EPS
    total_per_cond = DRY_TOTAL_EPS_PER_CONDITION if dry_run else TOTAL_EPS_PER_CONDITION
    p2_eps = DRY_P2 if dry_run else P2_BUDGET
    steps_per = DRY_STEPS if dry_run else STEPS_PER_EP

    # --- Shared training: one agent, once per seed (verbatim from 228b). ---
    reset_all_rng(seed)
    scheduler = ScaffoldedSD054OnboardingScheduler(sched_cfg)
    probe_env = _build_env(sched_cfg, phase="p2", seed=None)
    torch.manual_seed(seed)
    agent_cfg = build_agent_config(probe_env)
    agent = REEAgent(agent_cfg).to(device)

    ep_so_far = 0

    def _progress(phase_name: str, n_this_phase: int):
        nonlocal ep_so_far
        ep_so_far += n_this_phase
        print(
            f"  [train] seed={seed} shared ep {ep_so_far}/{total_train} phase={phase_name}",
            flush=True,
        )

    s0 = scheduler.run_stage0_nursery(agent, device)
    _progress("stage0", s0.n_episodes if not s0.aborted else 0)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    _progress("stage0b", s0b.n_episodes if not s0b.aborted else 0)

    p0 = scheduler.run_p0(agent, device)
    _progress("p0", p0.n_episodes)

    hz = scheduler.run_hazard_avoidance(agent, device)
    _progress("stage_h", hz.n_episodes)

    p1 = scheduler.run_p1(agent, device)
    _progress("p1", p1.n_episodes)

    diag = agent.compute_goal_maintenance_diagnostic()
    goal_norm_post_p1 = float(diag["goal_norm"])
    print(f"  [train done] seed={seed} goal_norm_post_p1={goal_norm_post_p1:.4f}", flush=True)

    zg.observe(agent)

    # --- Flush cached graph-attached tensors before deepcopy (verbatim 228b). ---
    agent.eval()
    with torch.no_grad():
        _, _flush_obs = probe_env.reset()
        agent.reset()
        _flush_body = _flush_obs["body_state"].to(device)
        _flush_world = _flush_obs["world_state"].to(device)
        _flush_latent = _sense_with_optional_harm(
            agent, _flush_body, _flush_world, _flush_obs, device,
            sched_cfg.scaffold_feed_harm_stream,
        )
        _flush_ticks = agent.clock.advance()
        _flush_prior = (
            agent._e1_tick(_flush_latent)
            if _flush_ticks.get("e1_tick", True)
            else torch.zeros(1, agent.config.latent.world_dim, device=device)
        )
        _flush_cand = agent.generate_trajectories(_flush_latent, _flush_prior, _flush_ticks)
        _ = agent.select_action(_flush_cand, _flush_ticks, temperature=0.5)

    # hippocampal._rng defaults to the `random` MODULE (unpicklable) -- same
    # fix as V3-EXQ-838. Installs a picklable per-seed random.Random instance.
    agent.hippocampal.seed_replay_rng(seed)

    # --- Deep-copy split. Both clones start byte-identical. ---
    agent_active = agent
    agent_zeroed = copy.deepcopy(agent)
    _patch_theta_zeroed(agent_zeroed)

    train_summary = {
        "stage0_z_goal_norm_peak": s0.z_goal_norm_peak,
        "stage0_aborted": s0.aborted,
        "stage0b_retention_ratio": s0b.retention_ratio,
        "stage0b_aborted": s0b.aborted,
        "p0_mean_episode_length": p0.mean_episode_length,
        "hazard_median_last_window": hz.median_last_window_episode_length,
        "hazard_survival_gate_passed": hz.survival_gate_passed,
        "p1_median_last_window": p1.median_last_window_episode_length,
        "p1_survival_gate_passed": p1.survival_gate_passed,
        "goal_norm_post_p1": goal_norm_post_p1,
    }

    rows: Dict[str, Dict[str, Any]] = {}
    for condition, agent_cell in [
        ("THETA_ACTIVE", agent_active), ("THETA_ZEROED", agent_zeroed),
    ]:
        print(f"Seed {seed} Condition {condition}", flush=True)

        # RNG reset again, same seed, before EACH condition's measurement --
        # both conditions see the same env resets + action-sampling draws.
        reset_all_rng(seed)
        p2_env = _build_env(sched_cfg, phase="p2", seed=None)
        metrics = _measure_p2(agent_cell, p2_env, device, sched_cfg, p2_eps, steps_per)

        print(
            f"  [train] seed={seed} {condition} ep {total_per_cond}/{total_per_cond} phase=p2",
            flush=True,
        )
        print(
            f"  [eval] {condition} seed={seed}"
            f" persistence_cos={metrics['persistence_cos']:.4f}"
            f" prox_noise_temporal={metrics['prox_noise_temporal']:.4f}"
            f" prox_noise_xcand={metrics['prox_noise_xcand']:.4f}"
            f" e3_ticks={metrics['n_e3_ticks']} e3_tick_ratio={metrics['e3_tick_ratio']:.3f}"
            f" e3_resource_visit_rate={metrics['e3_resource_visit_rate']:.4f}",
            flush=True,
        )
        print("verdict: PASS", flush=True)  # cell ran to completion; scientific verdict is aggregate-level

        row: Dict[str, Any] = {"seed": seed, "condition": condition}
        row.update(train_summary)
        row.update(metrics)
        row["goal_norm_active"] = goal_norm_post_p1  # shared precondition value, both rows
        row["arm_fingerprint"] = compute_arm_fingerprint(
            config_slice=_cell_config_slice(condition, seed, dry_run),
            seed=seed,
            script_path=Path(__file__),
            rng_fully_reset=True,
            # Both cells share the same trained agent object via deepcopy --
            # not independently retrainable from config+seed alone.
            extra_ineligible_reasons=["shared_trained_agent_deepcopy_split_across_theta_conditions"],
        )
        rows[condition] = row

    return rows


# --------------------------------------------------------------------- #
# Aggregate + acceptance criteria
# --------------------------------------------------------------------- #

def _frac_seeds(flags: List[bool]) -> float:
    return sum(1 for f in flags if f) / max(1, len(flags))


def run(dry_run: bool = False) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    print(
        f"\n[{QUEUE_ID}] ARC-032 Theta-Rate Pathway DIRECT-READOUT test"
        f" (E3-tick-restricted persistence + proximity-noise, supersedes {SUPERSEDES})"
        f" dry_run={dry_run}",
        flush=True,
    )

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict] = []
    per_seed: Dict[int, Dict[str, Dict]] = {}

    for seed in SEEDS:
        per_seed[seed] = _run_seed(seed, zg, dry_run=dry_run)
        arm_results.append(per_seed[seed]["THETA_ACTIVE"])
        arm_results.append(per_seed[seed]["THETA_ZEROED"])

    gn_flags, c1_flags, c2_flags = [], [], []
    persist_deltas, noise_deltas, behav_deltas = [], [], []
    reversed_persist, reversed_noise = [], []
    for seed in SEEDS:
        active = per_seed[seed]["THETA_ACTIVE"]
        zeroed = per_seed[seed]["THETA_ZEROED"]

        # C1 persistence: theta ACTIVE should be MORE persistent (higher cosine)
        persist_delta = active["persistence_cos"] - zeroed["persistence_cos"]
        # C2 noise: theta ACTIVE should be QUIETER (lower temporal noise), so
        # delta = zeroed_noise - active_noise should be >= margin
        noise_delta = zeroed["prox_noise_temporal"] - active["prox_noise_temporal"]
        # Informational: E3-tick behavioral lift
        behav_delta = active["e3_resource_visit_rate"] - zeroed["e3_resource_visit_rate"]

        persist_deltas.append(persist_delta)
        noise_deltas.append(noise_delta)
        behav_deltas.append(behav_delta)
        reversed_persist.append(persist_delta < 0.0)   # theta ACTIVE less persistent
        reversed_noise.append(noise_delta < 0.0)       # theta ACTIVE noisier

        gn_flags.append(active["goal_norm_active"] >= GOAL_NORM_THRESH)
        c1_flags.append(persist_delta >= PERSIST_MARGIN)
        c2_flags.append(noise_delta >= NOISE_MARGIN)

    threshold = MIN_SEEDS_PASS / len(SEEDS)
    gn_frac, c1_frac, c2_frac = _frac_seeds(gn_flags), _frac_seeds(c1_flags), _frac_seeds(c2_flags)
    precond_pass = gn_frac >= threshold
    c1_pass = c1_frac >= threshold
    c2_pass = c2_frac >= threshold

    # Standardized paired effects (mean delta / std delta) -- reported for
    # governance; 3 seeds so these are directional, not powered.
    def _std_effect(deltas: List[float]) -> float:
        sd = _pstd(deltas)
        return (sum(deltas) / len(deltas)) / sd if sd > 1e-12 else 0.0

    non_degenerate = True
    degeneracy_reason = None
    reversed_trend = False

    if not precond_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        decision = "substrate_limitation"
        non_degenerate = False
        degeneracy_reason = (
            f"Precondition (goal_norm >= {GOAL_NORM_THRESH}) FAILED on "
            f"{gn_frac:.2f} of seeds -- unexpected regression from 228b, which "
            "cleared it 1.00. The theta pathway could not be tested: no live "
            "goal-context signal to measure persistence/noise against."
        )
        interpretation = (
            "ARC-032 NON-CONTRIBUTORY: the non-degeneracy precondition "
            f"(z_goal_norm >= {GOAL_NORM_THRESH}) failed on {gn_frac:.2f} of "
            "seeds, so the theta pathway was not exercised under a live goal-"
            "context signal. (228b cleared this precondition 1.00; a failure "
            "here indicates seed/substrate drift, not an ARC-032 result.)"
        )
    elif c1_pass and c2_pass:
        status = "PASS"
        evidence_direction = "supports"
        decision = "retain_ree"
        interpretation = (
            "ARC-032 SUPPORTED: with the precondition met "
            f"(goal_norm >= {GOAL_NORM_THRESH} on {gn_frac:.2f} of seeds), the "
            "theta-packaged (ACTIVE) condition maintains z_goal MORE persistently "
            f"(C1, {c1_frac:.2f} of seeds) AND scores goal-proximity LESS noisily "
            f"(C2, {c2_frac:.2f} of seeds) than the theta-bypassed (ZEROED) "
            "condition, measured directly at E3-tick granularity. The theta-rate "
            "packaging of E1 output stabilises z_goal influence on E3's "
            "trajectory scoring, as ARC-032 predicts."
        )
    elif c1_pass or c2_pass:
        status = "FAIL"
        evidence_direction = "mixed"
        decision = "inconclusive"
        which = "persistence (C1)" if c1_pass else "proximity-noise (C2)"
        interpretation = (
            "ARC-032 MIXED: with the precondition met, theta packaging shows the "
            f"predicted effect on {which} but not on the other pre-registered "
            f"readout (C1 {c1_frac:.2f}, C2 {c2_frac:.2f} of seeds; both need "
            f">= {threshold:.2f}). Partial support on one of two named CONFIRMING "
            "DVs -- theta influences one facet of goal maintenance but not both."
        )
    else:
        status = "FAIL"
        evidence_direction = "does_not_support"
        decision = "inconclusive"
        rev_persist_frac = _frac_seeds(reversed_persist)
        rev_noise_frac = _frac_seeds(reversed_noise)
        reversed_trend = (rev_persist_frac >= threshold) or (rev_noise_frac >= threshold)
        base = (
            "ARC-032 DOES NOT SUPPORT: with the precondition met "
            f"(goal_norm >= {GOAL_NORM_THRESH} on {gn_frac:.2f} of seeds), neither "
            f"the persistence readout (C1, {c1_frac:.2f} of seeds) nor the "
            f"proximity-noise readout (C2, {c2_frac:.2f} of seeds) showed the "
            "ARC-032-predicted effect at E3-tick granularity, measuring the "
            "claim's own pre-registered CONFIRMING DVs directly."
        )
        if reversed_trend:
            interpretation = base + (
                " REVERSED-TREND FINDING (registered, not dismissed): theta ACTIVE "
                f"was the WORSE arm on >= {threshold:.2f} of seeds (less persistent "
                f"on {rev_persist_frac:.2f}, noisier on {rev_noise_frac:.2f}). This "
                "converges with MECH-089's own already-confirmed EXQ-066/EXQ-122 "
                "result that static/uniform theta-averaging measurably HURTS E3's "
                "fine-grained discrimination -- implicating the flat-mean ThetaBuffer "
                "'packaging' implementation, not the frontal-hippocampal-synchrony "
                "hypothesis itself. Route: investigate a phase/sequence-order-aware "
                "theta summary (Dragoi 2006 / Colgin 2016) rather than demoting ARC-032."
            )
        else:
            interpretation = base + (
                " No consistent reversed trend; the theta channel appears neither "
                "necessary for goal-context persistence nor a source of proximity-"
                "score smoothing on this substrate at E3-tick granularity."
            )

    metrics: Dict[str, float] = {
        "precond_goal_frac_seeds": gn_frac,
        "c1_persistence_frac_seeds": c1_frac,
        "c2_noise_frac_seeds": c2_frac,
        "precond_goal_pass": 1.0 if precond_pass else 0.0,
        "c1_persistence_pass": 1.0 if c1_pass else 0.0,
        "c2_noise_pass": 1.0 if c2_pass else 0.0,
        "persist_delta_mean": _mean(persist_deltas),
        "noise_delta_mean": _mean(noise_deltas),
        "behav_e3_lift_mean": _mean(behav_deltas),
        "persist_delta_std_effect": _std_effect(persist_deltas),
        "noise_delta_std_effect": _std_effect(noise_deltas),
        "reversed_persist_frac_seeds": _frac_seeds(reversed_persist),
        "reversed_noise_frac_seeds": _frac_seeds(reversed_noise),
        "reversed_trend": 1.0 if reversed_trend else 0.0,
        "goal_norm_active_mean": sum(
            per_seed[s]["THETA_ACTIVE"]["goal_norm_active"] for s in SEEDS
        ) / len(SEEDS),
    }
    for cond in CONDITIONS:
        for key in (
            "persistence_cos", "prox_noise_temporal", "prox_noise_xcand",
            "zgoal_norm_e3_mean", "lead_prox_e3_mean",
            "e3_resource_visit_rate", "e3_harm_rate",
            "resource_visit_rate", "harm_rate", "zgoal_norm_mean",
            "n_e3_ticks", "e3_tick_ratio",
        ):
            vals = [per_seed[s][cond][key] for s in SEEDS]
            metrics[f"{key}_mean_{cond}"] = sum(vals) / len(vals)

    summary_markdown = (
        f"# {QUEUE_ID} -- ARC-032 Theta-Rate Pathway DIRECT-READOUT test\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}  "
        f"**Decision:** {decision}\n"
        f"**Supersedes:** {SUPERSEDES}\n"
        f"**Claims:** ARC-032\n\n"
        f"## Gates\n\n"
        f"| Gate | Frac seeds | Pass |\n|---|---|---|\n"
        f"| Precondition (goal_norm >= {GOAL_NORM_THRESH}) | {gn_frac:.2f} | {precond_pass} |\n"
        f"| C1 persistence (active-zeroed cos >= {PERSIST_MARGIN}) | {c1_frac:.2f} | {c1_pass} |\n"
        f"| C2 prox-noise (zeroed-active >= {NOISE_MARGIN}) | {c2_frac:.2f} | {c2_pass} |\n\n"
        f"## Direct-readout deltas (mean across seeds)\n\n"
        f"- persistence_cos (active - zeroed): {metrics['persist_delta_mean']:+.4f} "
        f"(std_effect {metrics['persist_delta_std_effect']:+.2f})\n"
        f"- prox_noise_temporal (zeroed - active): {metrics['noise_delta_mean']:+.4f} "
        f"(std_effect {metrics['noise_delta_std_effect']:+.2f})\n"
        f"- E3-tick resource lift (active - zeroed, info): {metrics['behav_e3_lift_mean']:+.4f}\n"
        f"- e3_tick_ratio (ACTIVE): {metrics.get('e3_tick_ratio_mean_THETA_ACTIVE', 0.0):.3f}\n\n"
        f"## Interpretation\n\n{interpretation}\n"
    )

    result: Dict[str, Any] = {
        "status": status,
        "outcome": status,
        "decision": decision,
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "per_seed_results": per_seed,
        "fatal_error_count": 0,
    }
    if not non_degenerate:
        result["non_degenerate"] = False
        result["degeneracy_reason"] = degeneracy_reason

    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "stage0_budget": STAGE0_BUDGET,
        "stage0b_budget": STAGE0B_BUDGET,
        "p0_budget": P0_BUDGET,
        "hazard_budget": HAZARD_STAGE_BUDGET,
        "p1_budget": P1_BUDGET,
        "p2_budget": P2_BUDGET,
        "steps_per_ep": STEPS_PER_EP,
        "p0_num_hazards": P0_NUM_HAZARDS,
        "hazard_stage_regime": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                                 HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR, N_RESOURCE_TYPES],
        "drive_weight": DRIVE_WEIGHT,
        "p2_hfa_guard": P2_HFA_GUARD,
        "goal_norm_thresh": GOAL_NORM_THRESH,
        "persist_margin": PERSIST_MARGIN,
        "noise_margin": NOISE_MARGIN,
        "measurement": "e3_tick_restricted_direct_readouts",
        "primary_readouts": ["persistence_cos", "prox_noise_temporal"],
        "design": "shared_training_deepcopy_split_at_p2",
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
