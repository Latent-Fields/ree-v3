#!/opt/local/bin/python3
"""
V3-EXQ-728 -- TRAINED ALL-ON CAPABILITY POINT (WS-3 "reported alongside every all-ON run").

WHY THIS EXISTS (a BASELINE reference run; experiment_purpose="baseline", claim_ids=[];
PROMOTES / DEMOTES NOTHING; EXCLUDED from governance scoring). WS-3 of
REE_assembly/evidence/planning/ree_ai_design_critique_plan.md built the reusable,
claim-agnostic capability yardstick experiments/_lib/capability_eval.py and calibrated its
scale with V3-EXQ-727 (random_walk floor / greedy_oracle ceiling / ree_p0warmup_allon
integration point). V3-EXQ-727 DELIBERATELY did NOT re-train the all-ON stack. This run
lands the OWED next step: the TRAINED all-ON capability point on ALL FOUR metrics against
that calibrated scale, closing WS-3's "reported alongside every all-ON run" clause -- i.e.
the trained denominator on the full metric set.

WHAT IT DOES. Three policy arms measured under identical env/seed/protocol (mirrors 727 but
replaces 727's ree_p0warmup_allon arm with a TRAINED all-ON arm):
  - random_walk       (FLOOR anchor; uniform-random actions)
  - ree_trained_allon (the all-ON REE stack -- 714 ARM_ON config -- trained with the
                       V3-EXQ-724 A0 budget: P0=200 world-model warmup THEN P1=90 two-head
                       REINFORCE (lateral-PFC bias head + OFC devaluation head) with the
                       SD-056 e2 world-forward encoder FROZEN through P1. This is the
                       trained-all-ON point; the SAME training recipe as 724's
                       A0_baseline_allon_p1short_frozen arm.)
  - greedy_oracle     (CEILING / achievability anchor; nearest-resource forager, reused
                       verbatim from V3-EXQ-724's positive control)

The four claim-agnostic metrics (defined + measured entirely in _lib/capability_eval.py):
  foraging_competence  -- mean resources/episode (env transition_type == "resource")
  survival_horizon     -- mean ticks survived/episode (+ death_rate on agent_health<=0)
  goal_reach_rate      -- fraction of episodes collecting >= 1 resource
  planning_depth       -- mean longest strictly-decreasing-nearest-resource-distance run

EVAL PATH. After training, the trained agent is evaluated via the yardstick's STANDARD
forward-eval policy capability_eval.REEForwardPolicy (sense -> e1 tick ->
generate_trajectories -> select_action, plus deployed z_goal/residue updates). This is the
IDENTICAL eval path 727 used for its ree_p0warmup_allon arm, so the 727 (P0-warmup only) vs
728 (P0+P1 trained) comparison isolates the effect of P1 REINFORCE competence-training on the
yardstick, with NO eval-protocol confound. NOTE: this eval does NOT inject the trained-OFC
viability signal into Go/No-Go the way 724's P2 competence phase does -- the yardstick eval is
mechanism-agnostic by construction. So the trained-all-ON foraging number here is measured on
the reusable yardstick's terms and is comparable to 727's p0warmup point; it is NOT required
to reproduce 724's A0 P2 competence DV byte-for-byte.

DENOMINATOR ROLE. build_report normalizes each policy to [random_floor, oracle_ceiling] per
metric, so the trained all-ON point lands with an explicit normalized position on every one of
the four metrics. Any future all-ON experiment can now state "structure X moved capability
metric Y by Z relative to the trained-all-ON denominator on a scale whose floor/ceiling are
fixed."

SELF-ROUTE (BASELINE; a HYPOTHESIS, not a verdict; tags NO claim):
  * READINESS: the greedy oracle must clear COMPETENCE_RESOURCE_FLOOR (=1.0 resource/ep) on
    the ceiling anchor -- proving the floor is ACHIEVABLE in this env -- AND the oracle ceiling
    must exceed the random floor on foraging (yardstick_discriminates) for the scale to be
    non-degenerate. If either fails the scale is not usable here -> label
    `substrate_not_ready_requeue` (draw NO conclusion; NEVER a capability verdict).
  * PASS: readiness holds AND the yardstick discriminates -> label
    `trained_allon_capability_point_landed`. The trained all-ON point is measured on all four
    metrics against the calibrated scale. Whether the trained point CLEARS the competence floor
    is reported as context (trained_allon_supra_floor / normalized_position), NOT a governance
    act -- this run is excluded from scoring.

EVIDENCE-FOR / EVIDENCE-AGAINST (baseline-description requirement):
  * EVIDENCE the trained point is measurable on the scale: oracle clears the floor AND
    oracle > random on foraging -> the four metrics have a meaningful, discriminating scale, so
    the trained-all-ON normalized positions are interpretable.
  * EVIDENCE AGAINST (substrate_not_ready_requeue): the oracle itself cannot clear the floor
    (env too sparse/lethal for the floor to be achievable) OR the anchors do not separate --
    the scale is degenerate and no normalized trained point is licensed.
  This run tags NO claim; the label is a hypothesis for adjudication, never a governance act.

UNTRAINED-WORLD-ENCODER GUARD (wired 2026-07-19; DETECTION ONLY).
  experiments/_lib/zworld_encoder_guard.py is wired into the `ree_trained_allon` arm only.
  NONE of the optimizer groups inside _train_all_on_agent (e2, lateral-PFC bias head, OFC
  devaluation head) covers ANY latent_stack parameter, so split_encoder.world_encoder
  receives no gradient in P0 or P1 and z_world stays a FROZEN RANDOM PROJECTION at
  initialisation (measured: 0/61 latent_stack tensors changed, 0/4 world_encoder tensors
  changed, max|delta| = 0.000e+00).

  POLICY = STRICT for this script, and the reason is specific to it: the script is NAMED
  "trained_allon" and its P0 phase is documented as a "world-model (encoder/e2) warmup",
  so the `ree_trained_allon` arm's premise REQUIRES a learned world representation. A
  frozen random projection does not merely weaken that arm, it voids its premise, so the
  arm is REFUSED rather than annotated.

  ARM-SCOPED, NEVER RUN-SCOPED. The guard refuses the ARM, never the RUN. `random_walk`
  and `greedy_oracle` run no P0 warmup at all and their premise does not involve z_world,
  so the guard precondition is explicitly scoped OUT of them and their yardstick anchors
  remain fully valid and scored. Per the multi-arm regime-conditioning rule
  (.claude/skills/queue-experiment/SKILL.md, the V3-EXQ-785 defect), no arm's gate result
  may vacate another arm's: `non_degenerate` is ANY-arm-green, not all-arms-green.

  SCOPE: DETECTION ONLY. Nothing here attempts to make the encoder train; that fix is
  downstream of the V3-EXQ-783 adjudication and belongs to governance.
  Diagnosis: REE_assembly/evidence/planning/
             zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Sourced config (all-ON matched stack + training harness) from V3-EXQ-714 ARM_ON via
V3-EXQ-724 / 727:
  experiments/v3_exq_724_competence_localization_diagnostic.py (A0 all-ON config + P0/P1
    training harness + oracle positive control),
  experiments/v3_exq_727_capability_yardstick_calibration.py (yardstick arm structure),
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
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.zworld_encoder_guard import (
    ZWORLD_PRECONDITION_NAME,
    ZWorldEncoderUntrainedError,
    assert_world_encoder_trained,
    latent_stack_snapshot,
    latent_stack_weight_delta,
    zworld_precondition,
)
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_728_trained_allon_capability_point"
QUEUE_ID = "V3-EXQ-728"
CLAIM_IDS: List[str] = []                 # tags NO claim -- trained-all-ON capability point
EXPERIMENT_PURPOSE = "baseline"

# ---------------------------------------------------------------------------
# Budget (mirrors V3-EXQ-724 A0: P0=200 warmup + P1=90 REINFORCE, frozen encoder in P1)
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
ZWORLD_P0_EPISODES = 60           # P0a SD-070 z_world ENCODER warmup, run AHEAD of the e2
                                  # warmup below. 60 is SD-070's validated operating point
                                  # (exq783_zworld_granularity.OFF_P0_ENCODER_EPISODES).
                                  # Set 0 to restore the pre-fix frozen-encoder behaviour.
P0_WARMUP_EPISODES = 200          # SD-056 e2 forward-model warmup (mirrors 724 / 719a / 714 P0).
                                  # This stage trains e2 ONLY -- no optimizer here covers a
                                  # latent_stack parameter. It is P0a above, NOT this stage,
                                  # that trains split_encoder.world_encoder; with
                                  # ZWORLD_P0_EPISODES=0 z_world stays a frozen random
                                  # projection, which zworld_encoder_guard detects.
P1_REINFORCE_EPISODES = 90        # two-head REINFORCE (mirrors 724 A0 P1_SHORT)
EVAL_EPISODES = 20                # capability-eval episodes per (arm, seed) cell
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42, 43]
DRY_RUN_ZWORLD_P0 = 2             # P0a: enough to exercise the real SD-070 training code
DRY_RUN_P0 = 3
DRY_RUN_P1 = 3
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

# P1 bias-head REINFORCE training (mirror V3-EXQ-724 / 714).
LR_LPFC_BIAS = 5e-4
LR_OFC_DEVAL = 2e-3
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9
# The all-ON encoder is FROZEN through P1 (A0 recipe: e2_train_in_p1 = False).
E2_TRAIN_IN_P1 = False

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


# Identical env to V3-EXQ-714 / 719a / 724 / 727 (SD-054 reef + hazard_food_attraction + bipartite).
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
# All-ON agent config (719a ARM_ON via 724 / 727).
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


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        return summ.detach()
    rows: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            rows.append(c.get_world_state_sequence()[0, 0, :].detach())
        elif agent._current_latent is not None:
            rows.append(agent._current_latent.z_world[0].detach())
        else:
            return None
    return torch.stack(rows, dim=0) if rows else None


# ---------------------------------------------------------------------------
# P1 two-head REINFORCE (mirror V3-EXQ-724).
# ---------------------------------------------------------------------------
def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    if getattr(agent, "lateral_pfc", None) is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = agent.lateral_pfc.compute_bias(cand_features.to(device))
        log_p = torch.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _ofc_deval_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    ofc = getattr(agent, "ofc", None)
    if ofc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = ofc.compute_devaluation_bias(cand_features.to(device))
        if not bias.requires_grad or bias.shape[0] < 2:
            continue
        log_p = torch.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# Train the all-ON agent: P0a SD-070 z_world encoder warmup (OPT-IN) THEN P0b e2 forward-model
# warmup THEN P1 two-head REINFORCE.
#
# THE V3-EXQ-780 DEFECT AND ITS REMEDY. The three optimizer groups below cover NO latent_stack
# parameter, so split_encoder.world_encoder is never stepped and z_world stays a frozen random
# projection -- measured here as 0 of 4 world_encoder / 0 of 61 latent_stack tensors changed at
# p0_episodes=200, 3 of 3 seeds (V3-EXQ-728, the SECOND independent strike after V3-EXQ-737a).
# This is a SEPARATE COPY of x734's function, which is why the defect was per-copy rather than
# confined to the shared _lib path -- so it needs its own fix, not just x734's.
# `zworld_p0_episodes > 0` adds the SD-070 recipe as a P0a stage AHEAD of the e2 warmup, per the
# V3-EXQ-783 adjudication. See experiments/_lib/zworld_p0_warmup.py.
#
# ORDERING IS NOT ARBITRARY: e2 regresses on z_world, so the encoder must be trained BEFORE the
# e2 warmup -- training it afterwards would leave e2 fitted to the random projection.
#
# DEFAULT zworld_p0_episodes=0 IS EXACTLY THE PRIOR BEHAVIOUR, bit-identical: no extra tensor,
# no extra optimizer group, no extra env construction, and no RNG draw.
#
# Mirrors the V3-EXQ-724 A0 recipe (e2 encoder FROZEN through P1). No P2 phase --
# competence is measured downstream by the capability yardstick's REEForwardPolicy eval.
# ---------------------------------------------------------------------------
def _train_all_on_agent(
    agent: REEAgent,
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    zworld_p0_episodes: int = 0,
    zworld_p0_dry_run: bool = False,
) -> Dict[str, Any]:
    env = _make_env(seed)

    # -- P0a: SD-070 z_world encoder warmup (opt-in; see the header note) -------------------
    # A DEDICATED env instance: the warmup rollout consumes env RNG, so running it on `env`
    # would shift the layout sequence P0b/P1 then see. Built only when the stage is enabled,
    # so the OFF path constructs nothing and draws nothing.
    zworld_p0_stats: Dict[str, Any] = {"p0a_recipe": "sd070", "p0a_ran": False}
    if zworld_p0_episodes > 0:
        zworld_p0_stats = run_zworld_p0(
            agent, _make_env(seed), seed, zworld_p0_episodes, steps_per_episode,
            policy=RandomPolicy(seed), label="ree_allon_728",
            dry_run=zworld_p0_dry_run,
        )
    has_ofc = getattr(agent, "ofc", None) is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = (
        torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        if has_lpfc else None
    )
    ofc_deval_opt = (
        torch.optim.Adam(list(agent.ofc.devaluation_bias_head_parameters()), lr=LR_OFC_DEVAL)
        if has_ofc else None
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes
    p1_start = p0_episodes
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_e2_train_steps = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    for ep in range(total_train_eps):
        is_p1 = (ep >= p1_start)
        is_p0 = not is_p1
        phase_label = "P1" if is_p1 else "P0"

        _flat, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

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

            # P1 REINFORCE snapshot of candidate summaries (all-ON only; heads present).
            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and has_lpfc and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                break

            committed_class = int(action[0].argmax().item())

            if is_p1 and p1_snap_summaries is not None:
                sel = 0
                for ci, c in enumerate(candidates):
                    if (
                        getattr(c, "actions", None) is not None
                        and c.actions.shape[1] >= 1
                        and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                        == committed_class
                    ):
                        sel = min(ci, p1_snap_summaries.shape[0] - 1)
                        break
                ep_buf.append((p1_snap_summaries, sel))

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 always; P1 only when the recipe unfreezes the encoder
            # (A0: frozen, so E2_TRAIN_IN_P1 = False -> P0-only).
            train_e2_now = is_p0 or (is_p1 and E2_TRAIN_IN_P1)
            if train_e2_now and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                if _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng) is not None:
                    n_e2_train_steps += 1

            _flat, _harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(_harm_signal)
            if is_p1:
                ep_reward += harm_signal

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

        # P1 end-of-episode: TWO-head REINFORCE (lateral-PFC bias + OFC devaluation).
        if is_p1 and (has_lpfc or has_ofc):
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            if has_lpfc and bias_opt is not None:
                l_loss = _lpfc_reinforce_loss(agent, outcome_buf, reinforce_baseline, agent.device)
                if l_loss.requires_grad:
                    bias_opt.zero_grad()
                    l_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.lateral_pfc.bias_head_parameters(), 1.0)
                    bias_opt.step()
            if has_ofc and ofc_deval_opt is not None:
                ofc_loss = _ofc_deval_reinforce_loss(agent, outcome_buf, reinforce_baseline, agent.device)
                if ofc_loss.requires_grad:
                    ofc_deval_opt.zero_grad()
                    ofc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.ofc.devaluation_bias_head_parameters(), 1.0)
                    ofc_deval_opt.step()

        cur = ep + 1
        if cur % 25 == 0 or cur == total_train_eps or phase_label == "P1":
            print(
                f"  [train] trained_allon seed={seed} phase={phase_label} "
                f"ep {cur}/{total_train_eps}",
                flush=True,
            )

    return {
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_e2_train_steps": int(n_e2_train_steps),
        "zworld_p0": zworld_p0_stats,
    }


# ---------------------------------------------------------------------------
# Arm table
# ---------------------------------------------------------------------------
ARMS = ("random_walk", "ree_trained_allon", "greedy_oracle")

# The ONLY arm the z_world untrained-encoder guard applies to: the only arm that runs the
# P0 warmup and the only arm whose premise requires a learned world representation.
GUARDED_ARM = "ree_trained_allon"


def _run_cell(
    arm_id: str,
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    zworld_p0_episodes: int = 0,
    zworld_p0_dry_run: bool = False,
) -> Dict[str, Any]:
    """One (arm, seed) capability-eval cell. Returns the seed-level metric row."""
    train_stats: Dict[str, Any] = {}
    guard_report: Optional[Dict[str, Any]] = None
    guard_ok: Optional[bool] = None
    guard_message: Optional[str] = None
    if arm_id == "random_walk":
        policy = RandomPolicy(seed)
    elif arm_id == "greedy_oracle":
        policy = OraclePolicy()
    elif arm_id == "ree_trained_allon":
        train_env = _make_env(seed)
        agent = _make_all_on_agent(train_env)
        # z_world untrained-encoder guard -- DETECTION ONLY, and scoped to THIS arm only.
        # random_walk / greedy_oracle run no warmup, so they are never snapshotted.
        before = latent_stack_snapshot(agent)
        train_stats = _train_all_on_agent(
            agent, seed, p0_episodes, p1_episodes, steps_per_episode,
            zworld_p0_episodes=zworld_p0_episodes,
            zworld_p0_dry_run=zworld_p0_dry_run,
        )
        guard_report = latent_stack_weight_delta(agent, before)
        guard_report["p0_episodes"] = int(p0_episodes)
        guard_report["guard_checked"] = bool(p0_episodes > 0 and before)
        # Exercise the guard's real raising contract path, but catch it here: a bare raise
        # would abort the process, produce a runner ERROR with NO manifest, and destroy the
        # random_walk / greedy_oracle yardstick anchors, which are perfectly valid.
        try:
            assert_world_encoder_trained(
                agent, before, p0=p0_episodes, strict=True,
                context="V3-EXQ-728a ree_trained_allon",
            )
            guard_ok = True
        except ZWorldEncoderUntrainedError as exc:
            guard_ok = False
            guard_message = str(exc)
            print(f"[GUARD-REFUSAL] arm=ree_trained_allon seed={seed}: {guard_message}",
                  file=sys.stderr, flush=True)
            print(f"[GUARD-REFUSAL] arm=ree_trained_allon seed={seed}: {guard_message}",
                  flush=True)
        policy = REEForwardPolicy(agent, name="ree_trained_allon")
    else:
        raise ValueError(f"unknown arm {arm_id!r}")

    eval_env = _make_env(seed)
    row = evaluate_seed(policy, eval_env, eval_episodes, steps_per_episode)
    row["arm_id"] = arm_id
    row["seed"] = int(seed)
    row["n_p0_ticks"] = int(train_stats.get("n_p0_ticks", 0))
    row["n_p1_ticks"] = int(train_stats.get("n_p1_ticks", 0))
    row["n_e2_train_steps"] = int(train_stats.get("n_e2_train_steps", 0))
    # None (not a fabricated report) for the arms that run no warmup.
    row["zworld_guard"] = guard_report
    row["zworld_guard_ok"] = guard_ok
    if guard_message is not None:
        row["zworld_guard_message"] = guard_message
    return row


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
    zworld_p0_episodes: int = 0,
) -> Dict[str, Any]:
    print(
        f"Trained all-ON capability point ({len(ARMS)} arms x {len(seeds)} seeds; "
        f"P0a={zworld_p0_episodes}, P0={p0_episodes}, P1={p1_episodes}, "
        f"eval_eps={eval_episodes}, "
        f"steps={steps_per_episode}, dry_run={dry_run})",
        flush=True,
    )

    cells: List[Dict[str, Any]] = []
    for arm_id in ARMS:
        for s in seeds:
            print(f"Seed {s} Condition {arm_id}", flush=True)
            is_trained = (arm_id == "ree_trained_allon")
            slice_cfg = {
                "arm_id": arm_id,
                # P0a MUST appear in the fingerprinted slice: an arm warmed with the SD-070
                # encoder recipe is a DIFFERENT arm from the frozen-random-projection arm of
                # every prior run, so a pre-fix banked arm must not cache-HIT here.
                "zworld_p0_episodes": int(zworld_p0_episodes) if is_trained else 0,
                "p0_episodes": int(p0_episodes) if is_trained else 0,
                "p1_episodes": int(p1_episodes) if is_trained else 0,
                "e2_train_in_p1": bool(E2_TRAIN_IN_P1) if is_trained else False,
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
                row = _run_cell(
                    arm_id, s, p0_episodes, p1_episodes, eval_episodes, steps_per_episode,
                    zworld_p0_episodes=zworld_p0_episodes,
                    zworld_p0_dry_run=dry_run,
                )
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

    # ----- z_world untrained-encoder guard: ARM-SCOPED aggregation -----
    # The guard precondition applies ONLY to ree_trained_allon. Scoping it out of the two
    # anchor arms is explicit and reasoned, not an omission: no arm's gate result may
    # vacate another arm's (queue-experiment multi-arm regime-conditioning rule).
    guard_rows = [
        c for c in cells
        if c["arm_id"] == GUARDED_ARM and c.get("zworld_guard") is not None
    ]
    guard_reports = {
        int(c["seed"]): c["zworld_guard"] for c in guard_rows
    }
    checked_rows = [c for c in guard_rows if c["zworld_guard"].get("guard_checked")]
    n_guard_failed = sum(1 for c in checked_rows if c.get("zworld_guard_ok") is False)
    guard_ok_arm = bool(checked_rows) and n_guard_failed == 0
    # No seed was actually checked (p0 <= 0) => the guard cannot speak; do not refuse on it.
    guard_applicable = bool(checked_rows)
    guard_arm_green = (not guard_applicable) or guard_ok_arm
    guard_messages = sorted({
        str(c["zworld_guard_message"]) for c in guard_rows if c.get("zworld_guard_message")
    })
    # Representative report for the preconditions[] entry: the first FAILING seed when any
    # seed failed (so the entry reports the refusing measurement), else the first checked.
    if checked_rows:
        failing = [c for c in checked_rows if c.get("zworld_guard_ok") is False]
        rep_report = (failing or checked_rows)[0]["zworld_guard"]
    else:
        rep_report = guard_rows[0]["zworld_guard"] if guard_rows else {}

    # Per-arm gate. Anchors carry ONLY their own (scale-readiness) precondition.
    anchor_green = bool(readiness_met)
    arm_green: Dict[str, bool] = {
        "random_walk": anchor_green,
        "greedy_oracle": anchor_green,
        GUARDED_ARM: bool(readiness_met and guard_arm_green),
    }
    failed_preconditions_by_arm: Dict[str, List[str]] = {}
    for a_id in ARMS:
        failed: List[str] = []
        if not readiness_met:
            failed.append("oracle_clears_competence_floor")
        if a_id == GUARDED_ARM and not guard_arm_green:
            failed.append(ZWORLD_PRECONDITION_NAME)
        failed_preconditions_by_arm[a_id] = failed
    green_arms = [a for a in ARMS if arm_green.get(a)]
    red_arms = [a for a in ARMS if not arm_green.get(a)]
    per_arm_gate = {
        "green_arms": green_arms,
        "red_arms": red_arms,
        "failed_preconditions_by_arm": failed_preconditions_by_arm,
        "scoped_out": {
            "random_walk": {
                ZWORLD_PRECONDITION_NAME: (
                    "arm runs no P0 warmup and its premise does not involve z_world"
                ),
            },
            "greedy_oracle": {
                ZWORLD_PRECONDITION_NAME: (
                    "arm runs no P0 warmup and its premise does not involve z_world"
                ),
            },
        },
        "policy": (
            "STRICT for ree_trained_allon: the script is named 'trained_allon' and its P0 "
            "phase is documented as a world-model warmup, so a frozen random projection "
            "voids that arm's premise and the ARM is refused. The RUN is never refused -- "
            "the two anchor arms are scored independently."
        ),
    }
    # ANY arm green, NOT all -- a refused ree arm must not vacate the valid anchors.
    non_degenerate = bool(green_arms)
    degeneracy_reason: Optional[str] = None
    if not non_degenerate:
        degeneracy_reason = (
            "every arm is red: "
            + ", ".join(
                f"{a} ({'; '.join(failed_preconditions_by_arm[a]) or 'unspecified'})"
                for a in red_arms
            )
            + " -- no arm remains scored."
        )

    trained_arm_green = bool(arm_green[GUARDED_ARM])

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif not trained_arm_green:
        # The PASS label is carried entirely by the ree_trained_allon capability number, so
        # a refused ree arm must never be reported as PASS. NEVER a substrate verdict label
        # (substrate_ceiling / substrate_conditional / does_not_support / *_nondiscriminative
        # / *_unmeetable): a below-floor guard reading means "no gradient reached the
        # encoder", not "the criterion was falsified".
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif yardstick_discriminates:
        outcome = "PASS"
        label = "trained_allon_capability_point_landed"
    else:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    direction = "non_contributory"

    # Reported context (NOT a governance act -- this run is excluded from scoring):
    # where the trained all-ON point lands on the calibrated scale.
    trained = arm_summaries.get("ree_trained_allon", {})
    trained_supra_floor = bool(trained.get("majority_supra_floor", False))
    trained_norm_positions = {
        key: report["metrics"][key].get("normalized_position", {}).get("ree_trained_allon")
        for key in METRIC_KEYS
    }

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
                "applies_to_arms": list(ARMS),
            },
            dict(
                zworld_precondition(
                    rep_report,
                    arm=GUARDED_ARM,
                    context="V3-EXQ-728a ree_trained_allon",
                ),
                applies_to_arms=[GUARDED_ARM],
                scoped_out_arms={
                    "random_walk": (
                        "arm runs no P0 warmup and its premise does not involve z_world"
                    ),
                    "greedy_oracle": (
                        "arm runs no P0 warmup and its premise does not involve z_world"
                    ),
                },
                policy="strict",
                n_seeds_checked=len(checked_rows),
                n_seeds_failed=int(n_guard_failed),
                guard_messages=guard_messages,
            ),
        ],
        "criteria": [
            {
                "name": "yardstick_discriminates_ceiling_above_floor",
                "load_bearing": True,
                "passed": bool(yardstick_discriminates),
            },
        ],
        # Each criterion is keyed to its OWNING arm's gate. The two anchor criteria read only
        # greedy_oracle / random_walk, so a refused ree arm leaves them True; the criteria
        # that read the ree_trained_allon capability number go False when that arm is red.
        "criteria_non_degenerate": {
            "oracle_clears_floor": bool(anchor_green),
            "oracle_foraging_above_random": bool(anchor_green and yardstick_discriminates),
            "trained_allon_capability_point_measurable": bool(trained_arm_green),
            "trained_allon_normalized_position_interpretable": bool(trained_arm_green),
        },
        "criteria_non_degenerate_owning_arm": {
            "oracle_clears_floor": "greedy_oracle",
            "oracle_foraging_above_random": "greedy_oracle+random_walk",
            "trained_allon_capability_point_measurable": GUARDED_ARM,
            "trained_allon_normalized_position_interpretable": GUARDED_ARM,
        },
        "per_arm_gate": per_arm_gate,
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": degeneracy_reason,
        "reported_context_not_a_verdict": {
            "trained_allon_majority_supra_floor": trained_supra_floor,
            "trained_allon_normalized_position": trained_norm_positions,
            "trained_allon_capability_valid": bool(trained_arm_green),
            "note": (
                "trained-all-ON supra-floor status + normalized positions are REPORTED CONTEXT "
                "on an excluded-from-scoring baseline; they are NOT a governance verdict on "
                "all-ON competence."
                + (
                    ""
                    if trained_arm_green else
                    " REFUSED: the z_world untrained-encoder guard tripped on this arm, so "
                    "these trained-all-ON numbers were produced on a FROZEN RANDOM "
                    "PROJECTION and are NOT valid as a trained-all-ON capability point. The "
                    "random_walk and greedy_oracle anchors are unaffected and remain scored."
                )
            ),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": list(seeds),
        "zworld_p0_episodes": int(zworld_p0_episodes),
        "p0_warmup_episodes": int(p0_episodes),
        "p1_reinforce_episodes": int(p1_episodes),
        "e2_train_in_p1": bool(E2_TRAIN_IN_P1),
        "eval_episodes": int(eval_episodes),
        "steps_per_episode": int(steps_per_episode),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "capability_report": report,
        "trained_allon_supra_floor": trained_supra_floor,
        "trained_allon_normalized_position": trained_norm_positions,
        "trained_allon_capability_valid": bool(trained_arm_green),
        "per_arm_gate": per_arm_gate,
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": degeneracy_reason,
        "zworld_encoder_guard": {
            "policy": "strict",
            "guarded_arm": GUARDED_ARM,
            "applicable": bool(guard_applicable),
            "arm_green": bool(guard_arm_green),
            "n_seeds_checked": len(checked_rows),
            "n_seeds_failed": int(n_guard_failed),
            "messages": guard_messages,
            "per_seed_reports": guard_reports,
        },
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
        "per_arm_gate": result["per_arm_gate"],
        "non_degenerate": bool(result["non_degenerate"]),
        "degeneracy_reason": result["degeneracy_reason"],
        "diagnostics": {
            "zworld_encoder_guard": result["zworld_encoder_guard"],
        },
        "sleep_driver_pattern": "none",
        "evidence_direction_note": (
            f"V3-EXQ-728 TRAINED ALL-ON CAPABILITY POINT (experiment_purpose=baseline, "
            f"claim_ids=[], non_contributory -- EXCLUDED from governance scoring; PROMOTES / "
            f"DEMOTES NOTHING). WS-3 of ree_ai_design_critique_plan.md: lands the TRAINED all-ON "
            f"stack on ALL FOUR claim-agnostic capability metrics (foraging_competence, "
            f"survival_horizon, goal_reach_rate, planning_depth) against the V3-EXQ-727 "
            f"random-floor / greedy-oracle scale, closing WS-3's 'reported alongside every "
            f"all-ON run' clause. The all-ON stack (714 ARM_ON) is trained with the V3-EXQ-724 "
            f"A0 recipe: P0={result['p0_warmup_episodes']} world-model warmup THEN "
            f"P1={result['p1_reinforce_episodes']} two-head REINFORCE (lateral-PFC bias + OFC "
            f"devaluation), SD-056 e2 encoder FROZEN through P1 (e2_train_in_p1="
            f"{result['e2_train_in_p1']}), then evaluated via the yardstick's REEForwardPolicy. "
            f"Readiness: oracle clears the {COMPETENCE_RESOURCE_FLOOR} floor "
            f"(oracle_forage/ep={rep['readiness']['oracle_foraging_competence']}, "
            f"clears_floor={rg['oracle_clears_floor']}); yardstick_discriminates="
            f"{rg['yardstick_discriminates']}. Self-route (HYPOTHESIS, not a verdict): "
            f"readiness_met={rg['readiness_met']} -> label={result['interpretation_label']}. "
            f"REPORTED CONTEXT (not a governance verdict): trained-all-ON majority_supra_floor="
            f"{result['trained_allon_supra_floor']}, normalized_position="
            f"{result['trained_allon_normalized_position']}. Eval uses the reusable yardstick's "
            f"mechanism-agnostic REEForwardPolicy (no P2 OFC-viability injection), so the number "
            f"is comparable to 727's p0warmup point and is NOT required to reproduce 724's A0 P2 "
            f"competence DV byte-for-byte. "
            f"Z_WORLD UNTRAINED-ENCODER GUARD (STRICT, arm-scoped to ree_trained_allon; "
            f"DETECTION ONLY): arm_green={result['zworld_encoder_guard']['arm_green']}, "
            f"seeds_checked={result['zworld_encoder_guard']['n_seeds_checked']}, "
            f"seeds_failed={result['zworld_encoder_guard']['n_seeds_failed']}; "
            f"per_arm_gate green={result['per_arm_gate']['green_arms']} "
            f"red={result['per_arm_gate']['red_arms']}; non_degenerate="
            f"{result['non_degenerate']} (ANY arm green -- a refused ree_trained_allon arm "
            f"does NOT vacate the random_walk / greedy_oracle anchors, which run no P0 "
            f"warmup and whose premise does not involve z_world). "
            f"trained_allon_capability_valid={result['trained_allon_capability_valid']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "trained all-ON capability point; 3 policy arms x 3 seeds",
            "arms": {
                "random_walk": "uniform-random action policy -- FLOOR anchor",
                "ree_trained_allon": (
                    "all-ON REE stack (714 ARM_ON) trained with the 724 A0 recipe "
                    "(P0 world-model warmup + P1 two-head REINFORCE, e2 encoder frozen in P1), "
                    "evaluated via the yardstick REEForwardPolicy -- the trained-all-ON point"
                ),
                "greedy_oracle": "nearest-resource greedy forager -- CEILING/achievability anchor",
            },
            "metrics": list(METRIC_KEYS),
            "reusable_block": "experiments/_lib/capability_eval.py",
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-724",
            "training_recipe_sourced_from": "V3-EXQ-724 A0_baseline_allon_p1short_frozen",
            "zworld_p0_episodes": int(result["zworld_p0_episodes"]),
            "p0_warmup_episodes": int(result["p0_warmup_episodes"]),
            "p1_reinforce_episodes": int(result["p1_reinforce_episodes"]),
            "e2_train_in_p1": bool(result["e2_train_in_p1"]),
            "alpha_world": 0.9,
            "reef_bipartite_layout": True,
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-728 trained all-ON capability point (baseline; claim_ids=[])"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        zp0 = DRY_RUN_ZWORLD_P0
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        eval_eps = DRY_RUN_EVAL
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        zp0 = ZWORLD_P0_EPISODES
        p0 = P0_WARMUP_EPISODES
        p1 = P1_REINFORCE_EPISODES
        eval_eps = EVAL_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
        zworld_p0_episodes=zp0,
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
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
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
    _g = result["zworld_encoder_guard"]
    print(
        f"  zworld_encoder_guard: policy=strict arm={_g['guarded_arm']} "
        f"arm_green={_g['arm_green']} seeds_checked={_g['n_seeds_checked']} "
        f"seeds_failed={_g['n_seeds_failed']}",
        flush=True,
    )
    print(
        f"  per_arm_gate: green={result['per_arm_gate']['green_arms']} "
        f"red={result['per_arm_gate']['red_arms']} "
        f"non_degenerate={result['non_degenerate']}",
        flush=True,
    )
    if result["trained_allon_capability_valid"]:
        print(
            f"  trained-all-ON supra_floor={result['trained_allon_supra_floor']} "
            f"normalized_position={result['trained_allon_normalized_position']}",
            flush=True,
        )
    else:
        print(
            "  trained-all-ON capability point REFUSED (zworld_encoder_guard): measured on a "
            "FROZEN RANDOM PROJECTION, NOT a trained encoder -- do NOT read the numbers below "
            "as a trained-all-ON capability point.",
            flush=True,
        )
        print(
            f"  [refused] trained-all-ON supra_floor={result['trained_allon_supra_floor']} "
            f"normalized_position={result['trained_allon_normalized_position']}",
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
