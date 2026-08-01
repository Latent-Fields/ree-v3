#!/opt/local/bin/python3
"""
V3-EXQ-863 -- ARC-062/MECH-309 FULL-BUDGET REPLICATION of V3-EXQ-859's
lateral_pfc-vs-none route_source ablation: does routing
modulatory_channel_route_source="lateral_pfc" cause the MECH-448/449 collapse
observed in V3-EXQ-851, and is that causal effect training-duration-dependent?

WHY THIS EXISTS (failure_autopsy_V3-EXQ-859_2026-08-01, user-confirmed routing).
V3-EXQ-851 (the full ~7.7-hour, P0=200/P1=90/P2=60 falsifier design, seeds
42/43/44) found MECH-448 (F-eligibility demotion) completely dead (0.0) under
modulatory_channel_route_source='lateral_pfc', vs robustly live (17.76) under
'cand_world_summary' in the matched-stack template V3-EXQ-654j (same seeds).
V3-EXQ-859 (a cheap ~45-minute diagnostic: P0=10/P2=10 episodes, NO P1) compared
'lateral_pfc' vs 'none' at the SAME seeds to isolate whether route_source itself
is causal. Result: MECH-448 was ALIVE in BOTH arms (3/3 seeds each) -- it did NOT
reproduce 851's collapse under either route_source value. MECH-449 (Go/No-Go) was
DEAD in BOTH arms, matching its deadness in 851, suggesting MECH-449's suppression
is unrelated to route_source. `criteria_non_degenerate` on that run: neither
mechanism discriminated by route_source at all.

TWO READINGS LEFT UNDISTINGUISHED BY 859 (this script's whole purpose is to tell
them apart):
  (a) route_source has NO causal effect on MECH-448 at all -- 851's own diagnosis
      (route_source -> MECH-448/449 suppression) needs a different explanation.
  (b) route_source's suppressive effect on MECH-448 is TRAINING-DURATION-
      DEPENDENT -- it accumulates or emerges only over a longer run, and 859's
      ~45-minute probe (P0=10 episodes -- 20x shorter than 851's P0=200) simply
      cannot detect it by construction, per 859's own "single biggest risk this
      script accepts" caveat in its own docstring.

A cheap short-budget scale-down is not automatically a safe cost-reduction for
reproducing a training-dynamics-dependent effect -- 859's own routing
(interpretation_label=mixed_partial_result_needs_full_replication) named this
directly and its autopsy's Section 7 routing is exactly this script.

THE ABLATION (UNCHANGED from V3-EXQ-859; see that script's own "THE ABLATION"
section for the full mechanism trace through ree_core/agent.py's
REEAgent.select_action elif chain, ~line 7454-7480). ONLY
modulatory_channel_route_source varies between arms ('lateral_pfc' identity-
routes _bdc_lpfc into channel_route_bias; 'none' matches no branch, leaving
channel_route_bias=None -- "bit-identical" to routing being off per the code's
own comment). use_modulatory_channel_routing STAYS True on BOTH arms;
use_candidate_rule_field STAYS True (fixed) on BOTH arms -- CRF must be building
a rule_state for the ablation to be meaningful. Both arms enable
use_lateral_pfc_analog=True + lateral_pfc_train_rule_bias_head=True, so
agent.lateral_pfc (and its trainable bias head) EXISTS and is exercised on BOTH
arms regardless of whether the channel is ROUTED -- route_source controls only
whether the analog's computed bias reaches the modulatory accumulator, not
whether it is computed or trained.

WHAT CHANGED vs V3-EXQ-859 (the ONLY changes -- everything else, including the
agent config, env, and MECH-448/449 readout instrumentation, is copied
VERBATIM from that script):
  1. FULL EPISODE BUDGET, matching V3-EXQ-851's own schedule exactly:
     P0_WARMUP_EPISODES = 200 (was 10 in 859 -- 20x), P1_BIAS_TRAIN_EPISODES = 90
     (859 had NO P1 phase at all), P2_MEASUREMENT_EPISODES = 60 (was 10 in 859 --
     6x). Total 350 episodes/cell x 200 steps x 3 seeds x 2 arms = 420000 env
     ticks -- IDENTICAL to 851's own total, which measured elapsed_seconds=27623
     (~7.67h) on the cloud fleet. This design targets the SAME wall-clock budget
     for the SAME reason: the open question is explicitly about training
     DURATION, so matching 851's own duration is the only way to answer it
     apples-to-apples.
  2. P1 (bias-head REINFORCE training phase) is ADDED BACK, reusing 851's own
     _lpfc_reinforce_loss / outcome-buffer / EMA-baseline machinery VERBATIM
     (851 lines ~707-738, ~1116-1135). WHY this is safe to add despite 859's own
     "P1 REINFORCE bias-head training phase ... is out of scope entirely"
     reasoning: 859's reasoning was about SCOPE (P1 trains
     agent.lateral_pfc's bias head via REINFORCE, and by inspection neither
     MECH-448's nor MECH-449's own code path references
     agent.candidate_rule_field or agent.lateral_pfc -- so P1's training target
     is mechanistically downstream of, not upstream of, the F-eligibility/Go-
     No-Go gates), NOT a claim that P1 episodes are somehow unsafe to run. Since
     agent.lateral_pfc exists and is exercised on BOTH arms (item above), running
     P1 here does not privilege one arm, does not touch MECH-448/449's own
     decision logic, and its only purpose is to reproduce 851's EXACT training
     schedule so this run's ~7.7h duration is not just numerically similar to
     851's but STRUCTURALLY identical to it (same three phases, same per-phase
     episode counts, same per-phase machinery) -- the strongest form of
     apples-to-apples this diagnostic can offer without rebuilding 851's full
     C1-C7 falsifier apparatus (deliberately NOT done here; see next item).
  3. READOUT UNCHANGED FROM 859 (deliberately, per the autopsy's own routing:
     "a full-training-budget version of THIS SAME ablation ... is the only
     design that can actually discriminate reading (a) from (b)" -- NOT a
     rebuild of 851's full falsifier). This script does NOT compute GAP-A
     consumed-summary divergence, CRF differentiation/maturity readouts,
     committed-class-axis exercisability, propagation non-vacuity, committed-
     class entropy, C1(a-g) readiness gates, or a DV-symmetry / declared-null
     three-branch interpretation grid. The ONLY readouts are the raw quantities
     859 measured: MECH-448 (f_eligibility_demotion_active_frac +
     f_eligibility_excluded_count_mean) and MECH-449 (go_nogo_active_frac +
     go_nogo_suppressed_per_tick_mean), read with the SAME attribute names off
     agent.e3.last_score_diagnostics, latch-cleared on every tick (859's 791a-
     derived pattern) so every P2 reading is a genuine fresh E3.select() this
     tick, never a held/replayed snapshot. P1 ticks are NOT read for MECH-448/449
     (per 859's own framing, that question is structural-engagement-at-
     measurement-time, not a P1 behavioural question) -- P1 exists solely to
     replicate 851's training schedule; only P2 is measured.

RELEVANCE (unchanged from 859; restated because this run is the one that settles
it). This diagnostic's result determines whether the currently-SUSPENDED
V3-EXQ-858 (a 1200-minute, 4-rung f_weight ladder run that reuses the IDENTICAL
lateral_pfc-routed + MECH-448/449-active matched-stack config as V3-EXQ-851 --
GOV-FANOUT-1 Leg P-B) can safely resume as-is, needs a redesign, or should be
held for an /implement-substrate fix. Per failure_autopsy_V3-EXQ-859_2026-08-01
Section 5: "V3-EXQ-858 should remain suspended rather than being unblocked on an
ambiguous result" -- this run is what resolves that ambiguity.

COMPUTE COST -- FLAGGED EXPLICITLY (user-confirmed judgment call, 2026-08-01,
chip-20260801-859-full-replication). This is a real ~7.7-hour compute
commitment, deliberately accepted by the user given V3-EXQ-858 is ALREADY
sitting suspended pending exactly this question -- the cost of running this once
is smaller than the cost of leaving a 1200-minute cloud allocation suspended
indefinitely on an ambiguous short-probe result. See the queue entry `note` for
the same flag in queue-visible form.

See REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-859_2026-08-01.md (+
its .json sibling) for the full autopsy this replication answers,
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-851_2026-08-01.md for the
original full-budget finding this isolates,
experiments/v3_exq_859_arc062_lateral_pfc_route_mech448_449_ablation.py for the
short-budget diagnostic this extends (agent config + MECH-448/449 readout copied
verbatim from that script),
experiments/v3_exq_851_arc062_pa_lateral_pfc_route_source_gapfanout.py for the
P0/P1/P2 phase-structure + REINFORCE bias-head training machinery this reuses,
ree_core/agent.py (~line 7454) for the route-source dispatch this ablates.

claim_ids = [] (matches V3-EXQ-859's own convention -- this is a routing
diagnostic, not new MECH-309/ARC-062 evidence). experiment_purpose =
"diagnostic". supersedes = None -- this does NOT supersede V3-EXQ-859; 859's
short-budget result stands on its own terms (it correctly self-routed
"needs_full_replication" rather than overclaiming), and this is a duration-
extended follow-up, not a bug-fix correction of it.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator


EXPERIMENT_TYPE = "v3_exq_863_arc062_lateral_pfc_route_mech448_449_full_replication"
QUEUE_ID = "V3-EXQ-863"
SUPERSEDES = None
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

# Same shape as V3-EXQ-859/858's own exemption: this is a live-measurement COUNT
# aggregate (min n_p2_fresh_select across all 6 cells vs a cadence-derived
# floor), not a known-degenerate-reference anchor.
ANCHOR_REACHABILITY_EXEMPT = (
    "p2_fresh_select_sample_adequate_both_arms is a live-measurement count aggregate, "
    "not a known-degenerate-reference anchor; see V3-EXQ-858/859's identical exemption "
    "reasoning for the same precondition shape."
)

SEEDS = [42, 43, 44]

# ----- Budget: IDENTICAL to V3-EXQ-851's own schedule (see module docstring
# "WHAT CHANGED" item 1). This is the load-bearing difference from V3-EXQ-859. -----
P0_WARMUP_EPISODES = 200
P1_BIAS_TRAIN_EPISODES = 90
P2_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# ----- MECH-448/449 readiness thresholds -- REUSED VERBATIM from V3-EXQ-859/851's
# C1(e)/C1(f) preconditions. Used here only as a live/dead READ per arm x seed,
# not as a PASS/FAIL gate on a scientific criterion (mechanism-isolation
# ablation, not a claim test).
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8
EXCLUDED_COUNT_FLOOR = 0.0
NOGO_ACTIVE_FRAC_FLOOR = 0.8
NOGO_SUPPRESSED_FLOOR = 0.0

# Majority-of-seeds convention (851/654j/859 precedent).
MIN_SEEDS_FOR_LIVE = 2  # of 3

# Sample-adequacy readiness precondition (the ONE precondition this diagnostic
# gates on -- unchanged from 859). Cadence-derived worst-case floor, same
# formula/constant as 851's C1g / 859's own floor (nominal P2 window ticks /
# beta_rate_max_steps). At P2=60 this recovers 851's own 600-tick floor.
BETA_RATE_MAX_STEPS = 20  # ree_core/heartbeat/clock.py MECH-093 slowest E3-reselection cadence
FRESH_SELECT_FLOOR = (P2_MEASUREMENT_EPISODES * STEPS_PER_EPISODE) // BETA_RATE_MAX_STEPS  # 600

# ----- Matched-stack constants -- IDENTICAL to V3-EXQ-859/851/654j (env,
# MECH-448, MECH-449, modulatory selection authority, channel routing, CRF
# maturity levers, SD-056 online e2 training). ONLY modulatory_channel_route_source
# varies (the swept variable), and use_candidate_rule_field is FIXED True on both
# arms. Copied verbatim from V3-EXQ-859. -----
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

USE_MODULATORY_SHORTLIST_THEN_MODULATE = True
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3

USE_F_ELIGIBILITY_DEMOTION = True
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30
F_ELIGIBILITY_DN_SIGMA = 0.0
USE_F_ELIGIBILITY_ADAPTIVE_FLOOR = True
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0

USE_GO_NOGO_CONSTITUTION = True
USE_DACC = True
GNG_PERSEVERATION_FLOOR = 0.5
GNG_SAFETY_FLOOR = 0.5
GNG_PROTECT_MIN_ELIGIBLE = 1

MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

# CRF maturity/maintenance levers (851/654-lineage; matched constants).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# SD-056 online e2 training (mirror 851/859/649). P0-ONLY, matching 851 exactly
# (e2 frozen in P1/P2 for stable measurement -- see 851's own comment at its P0
# training-step gate).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
SD056_MULTISTEP_CONTRASTIVE = True
SD056_CONTRASTIVE_HORIZON = 5
SD056_OUTPUT_NORM_CLAMP = True
SD056_OUTPUT_NORM_CLAMP_RATIO = 2.0
MAX_GRAD_NORM = 1.0

# P1 bias-head REINFORCE training (mirror V3-EXQ-851 / V3-EXQ-598b). NEW vs 859
# (859 had no P1); reused VERBATIM from 851's own constants.
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

# IDENTICAL env to V3-EXQ-859/851/654 (SD-054 reef + hazard_food_attraction +
# bipartite layout) -- the behavioural falsifier substrate.
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
        "arm_id": "ARM_LPFC",
        "label": "modulatory_channel_route_source_lateral_pfc",
        "modulatory_channel_route_source": "lateral_pfc",
    },
    {
        "arm_id": "ARM_NONE",
        "label": "modulatory_channel_route_source_none_ablated",
        "modulatory_channel_route_source": "none",
    },
]

_ZG = ZGoalStreamAccumulator()


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, route_source: str) -> REEAgent:
    """Matched-stack agent identical to V3-EXQ-859/851/654j; the ONLY varied flag
    is modulatory_channel_route_source. use_candidate_rule_field is FIXED True
    (both arms build a real rule_state). use_lateral_pfc_analog +
    lateral_pfc_train_rule_bias_head are True on BOTH arms so the P1 REINFORCE
    phase has a real bias head to train regardless of route_source.
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
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        candidate_summary_source="e2_world_forward",
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=USE_MODULATORY_CHANNEL_ROUTING,
        modulatory_channel_route_source=str(route_source),
        modulatory_channel_route_weight=MODULATORY_CHANNEL_ROUTE_WEIGHT,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_shortlist_then_modulate=USE_MODULATORY_SHORTLIST_THEN_MODULATE,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        use_f_eligibility_demotion=USE_F_ELIGIBILITY_DEMOTION,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_f_eligibility_adaptive_floor=USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        use_dacc=USE_DACC,
        use_go_nogo_constitution=USE_GO_NOGO_CONSTITUTION,
        gng_perseveration_floor=GNG_PERSEVERATION_FLOOR,
        gng_safety_floor=GNG_SAFETY_FLOOR,
        gng_protect_min_eligible=GNG_PROTECT_MIN_ELIGIBLE,
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
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
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
    return REEAgent(cfg)


def _obs_harm(obs_dict):
    return obs_dict.get("harm_state")


def _obs_harm_a(obs_dict):
    return obs_dict.get("harm_affective_state")


def _obs_harm_history(obs_dict):
    return obs_dict.get("harm_history")


def _sample_class_diverse_batch(buffer, k, rng):
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    idxs = rng.sample(range(len(buffer)), min(k, len(buffer)))
    return [buffer[i] for i in idxs]


def _e2_contrastive_step(agent, buffer, optimiser, rng):
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K, simulation_mode=False,
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


def _lpfc_reinforce_loss(agent, outcome_buf, baseline, device):
    """REINFORCE on the SD-033a bias head over stored (candidate_features, sel,
    return). Copied verbatim from V3-EXQ-851's own _lpfc_reinforce_loss (mirrors
    v3_exq_598b._lpfc_reinforce_loss)."""
    if agent.lateral_pfc is None or len(outcome_buf) < 2:
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
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _consumed_summaries(agent, candidates):
    """Per-candidate cand_world_summaries the P1 bias head consumes (GAP-A
    e2.world_forward source). Copied from V3-EXQ-851's own helper -- used ONLY
    to snapshot P1 REINFORCE features, not for any MECH-448/449 readout."""
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


def _traj_first_action_class(traj) -> int:
    return int(traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    import random
    from collections import deque

    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env, str(arm["modulatory_channel_route_source"]))
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(
        list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS
    )
    transition_buffer = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes + p2_episodes
    p1_start = p0_episodes
    p2_start = p0_episodes + p1_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p2_ticks = 0
    n_p0_contrastive_steps = 0
    n_p1_bias_updates = 0

    # P1 REINFORCE state (851 pattern -- runs on BOTH arms since agent.lateral_pfc
    # exists on both; see module docstring "WHAT CHANGED" item 2).
    reinforce_baseline = 0.0
    outcome_buf: List[Any] = []

    # ----- MECH-448/449 latch-cleared P2 readouts (859 pattern, unchanged) -----
    n_p2_fresh_select = 0
    n_p2_latched_ticks = 0
    demotion_active_ticks = 0
    demotion_excluded_counts: List[float] = []
    nogo_active_ticks = 0
    nogo_suppressed_per_tick: List[int] = []
    # Context-only (not gating): the routed channel's own range/activity.
    route_ranges: List[float] = []
    route_active_ticks = 0

    for ep in range(total_train_eps):
        is_p1 = p1_start <= ep < p2_start
        is_p2 = ep >= p2_start
        phase_label = "P2" if is_p2 else ("P1" if is_p1 else "P0")

        _, obs_dict = env.reset()
        agent.reset()
        z_self_prev = None
        action_prev = None
        pending_capture = None
        tick_in_ep = 0

        # P1 per-episode REINFORCE buffers.
        ep_reward = 0.0
        ep_buf: List[Any] = []

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

            # Capture candidate summaries BEFORE select_action for P1 REINFORCE
            # snap (851 pattern).
            p1_snap_summaries = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            # V3-EXQ-859 latch-clearing (851/791a pattern -- the ~9x
            # pseudo-replication defect). EVERY read here is latch-cleared: we
            # only ever record a diagnostic from a genuine fresh E3.select()
            # this tick.
            agent.e3.last_score_diagnostics = None
            action = agent.select_action(candidates, ticks)
            fresh_diag = agent.e3.last_score_diagnostics
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            committed_class = int(action[0].argmax().item())

            # P1: record (candidate_features, selected-candidate-index) snap.
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

            if is_p2:
                n_p2_ticks += 1
                if fresh_diag is None:
                    n_p2_latched_ticks += 1
                else:
                    n_p2_fresh_select += 1
                    rr = fresh_diag.get("modulatory_channel_route_range")
                    if rr is not None and math.isfinite(float(rr)):
                        route_ranges.append(float(rr))
                    if bool(fresh_diag.get("modulatory_channel_route_active", False)):
                        route_active_ticks += 1

                    if bool(fresh_diag.get("f_eligibility_demotion_active", False)):
                        demotion_active_ticks += 1
                        excl = float(fresh_diag.get("f_eligibility_excluded_count", -1))
                        if math.isfinite(excl) and excl >= 0:
                            demotion_excluded_counts.append(excl)

                    if bool(fresh_diag.get("go_nogo_constitution_active", False)):
                        nogo_active_ticks += 1
                        n_safety = int(fresh_diag.get("go_nogo_n_safety_nogo", 0) or 0)
                        n_soft = int(fresh_diag.get("go_nogo_n_soft_applied", 0) or 0)
                        nogo_suppressed_per_tick.append(n_safety + n_soft)
            elif is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 ONLY (851 pattern: e2 frozen in P1/P2).
            if (not is_p1) and (not is_p2) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                loss_val = _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)
                if loss_val is not None and math.isfinite(loss_val):
                    n_p0_contrastive_steps += 1

            _, harm_signal, done, info, obs_dict = env.step(action)
            if is_p1:
                ep_reward += float(harm_signal)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal), world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        # P1 end-of-episode: REINFORCE update on the SD-033a bias head (851
        # pattern, verbatim).
        if is_p1:
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            l_loss = _lpfc_reinforce_loss(agent, outcome_buf, reinforce_baseline, agent.device)
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.lateral_pfc.bias_head_parameters(), 1.0)
                bias_opt.step()
                n_p1_bias_updates += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    _ZG.observe(agent)

    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_p2_fresh_select)
        if n_p2_fresh_select > 0 else 0.0
    )
    demotion_excluded_count_mean = (
        float(sum(demotion_excluded_counts) / len(demotion_excluded_counts))
        if demotion_excluded_counts else 0.0
    )
    nogo_active_frac = (
        float(nogo_active_ticks) / float(n_p2_fresh_select)
        if n_p2_fresh_select > 0 else 0.0
    )
    nogo_suppressed_mean = (
        float(sum(nogo_suppressed_per_tick) / len(nogo_suppressed_per_tick))
        if nogo_suppressed_per_tick else 0.0
    )
    route_range_mean = float(sum(route_ranges) / len(route_ranges)) if route_ranges else 0.0
    route_active_frac = (
        float(route_active_ticks) / float(n_p2_fresh_select)
        if n_p2_fresh_select > 0 else 0.0
    )

    mech448_live = bool(
        demotion_active_frac >= DEMOTION_ACTIVE_FRAC_FLOOR
        and demotion_excluded_count_mean > EXCLUDED_COUNT_FLOOR
    )
    mech449_live = bool(
        nogo_active_frac >= NOGO_ACTIVE_FRAC_FLOOR
        and nogo_suppressed_mean > NOGO_SUPPRESSED_FLOOR
    )

    passed = "PASS" if error_note is None else "FAIL"
    print(f"verdict: {passed}", flush=True)

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "modulatory_channel_route_source": str(arm["modulatory_channel_route_source"]),
        "error_note": error_note,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p2_fresh_select": int(n_p2_fresh_select),
        "n_p2_latched_ticks": int(n_p2_latched_ticks),
        # ----- MECH-448 (F-eligibility demotion) -----
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        "f_eligibility_excluded_count_mean": round(demotion_excluded_count_mean, 6),
        "mech448_live": mech448_live,
        # ----- MECH-449 (Go/No-Go constitution) -----
        "go_nogo_active_frac": round(nogo_active_frac, 6),
        "go_nogo_suppressed_per_tick_mean": round(nogo_suppressed_mean, 6),
        "mech449_live": mech449_live,
        # ----- context-only: routed-channel range (851's independent C1g finding) -----
        "modulatory_channel_route_range_mean": round(route_range_mean, 6),
        "modulatory_channel_route_active_frac": round(route_active_frac, 6),
    }


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []

    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} (route_source={arm['modulatory_channel_route_source']}) "
            f"(P0={p0_episodes} ep, P1={p1_episodes} ep, P2={p2_episodes} ep, "
            f"steps_per_episode={steps_per_episode}, dry_run={dry_run})",
            flush=True,
        )
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode)
            row["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm_id": arm["arm_id"],
                    "modulatory_channel_route_source": str(arm["modulatory_channel_route_source"]),
                    "use_candidate_rule_field": True,
                    "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
                    "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
                    "use_dacc": bool(USE_DACC),
                    "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
                    "use_modulatory_channel_routing": bool(USE_MODULATORY_CHANNEL_ROUTING),
                    "p0_episodes": int(p0_episodes),
                    "p1_episodes": int(p1_episodes),
                    "p2_episodes": int(p2_episodes),
                },
                seed=s,
                script_path=Path(__file__),
                rng_fully_reset=True,
                config_slice_declared=True,
            )
            arm_results.append(row)

    lpfc_rows = [r for r in arm_results if r["arm_id"] == "ARM_LPFC"]
    none_rows = [r for r in arm_results if r["arm_id"] == "ARM_NONE"]

    def _n_live(rows: List[Dict[str, Any]], key: str) -> int:
        return sum(1 for r in rows if r.get(key))

    n_lpfc_448_live = _n_live(lpfc_rows, "mech448_live")
    n_lpfc_449_live = _n_live(lpfc_rows, "mech449_live")
    n_none_448_live = _n_live(none_rows, "mech448_live")
    n_none_449_live = _n_live(none_rows, "mech449_live")

    arm_lpfc_448_live_majority = bool(n_lpfc_448_live >= MIN_SEEDS_FOR_LIVE)
    arm_lpfc_449_live_majority = bool(n_lpfc_449_live >= MIN_SEEDS_FOR_LIVE)
    arm_none_448_live_majority = bool(n_none_448_live >= MIN_SEEDS_FOR_LIVE)
    arm_none_449_live_majority = bool(n_none_449_live >= MIN_SEEDS_FOR_LIVE)

    arm_lpfc_engaged = arm_lpfc_448_live_majority and arm_lpfc_449_live_majority
    arm_none_engaged = arm_none_448_live_majority and arm_none_449_live_majority

    # ----- Sample-adequacy readiness precondition (859 pattern, unchanged). -----
    min_fresh_select = min([r["n_p2_fresh_select"] for r in arm_results] or [0])
    sample_adequate = bool(min_fresh_select >= FRESH_SELECT_FLOOR)

    if not sample_adequate:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif (not arm_lpfc_engaged) and arm_none_engaged:
        label = "route_source_confirmed_causal_channel_route_bias_drives_collapse"
        outcome = "PASS"
    elif (not arm_lpfc_engaged) and (not arm_none_engaged):
        label = "route_source_ruled_out_collapse_persists_without_routing"
        outcome = "PASS"
    elif arm_lpfc_engaged and arm_none_engaged:
        label = "mech448_449_live_under_lateral_pfc_route_851_not_replicated_at_full_budget"
        outcome = "PASS"
    elif arm_lpfc_engaged and (not arm_none_engaged):
        label = "unexpected_ablation_reverses_engagement_needs_followup"
        outcome = "PASS"
    else:
        label = "mixed_partial_result_needs_expert_review"
        outcome = "PASS"

    mech448_449_agree_lpfc = arm_lpfc_448_live_majority == arm_lpfc_449_live_majority
    mech448_449_agree_none = arm_none_448_live_majority == arm_none_449_live_majority
    if not (mech448_449_agree_lpfc and mech448_449_agree_none) and sample_adequate:
        label = "mixed_partial_result_needs_expert_review"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "p2_fresh_select_sample_adequate_both_arms",
                "kind": "readiness",
                "description": (
                    "The latch-cleared genuine-fresh-E3.select() sample collected "
                    "in P2 (n_p2_fresh_select) clears the cadence-derived worst-case "
                    "floor (P2_MEASUREMENT_EPISODES*STEPS_PER_EPISODE / "
                    "BETA_RATE_MAX_STEPS) on the WORST cell across both arms x all "
                    "seeds. Below-floor means the live/dead read for that cell is "
                    "not trustworthy -> substrate_not_ready_requeue, never a false "
                    "discrimination."
                ),
                "control": "min(n_p2_fresh_select) across all 6 cells",
                "measured": float(min_fresh_select),
                "threshold": float(FRESH_SELECT_FLOOR),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(sample_adequate),
            },
        ],
        "criteria": [
            {"name": "sample_adequate", "load_bearing": True, "passed": bool(sample_adequate)},
        ],
        "criteria_non_degenerate": {
            "mech448_ablation_discriminates": bool(
                arm_lpfc_448_live_majority != arm_none_448_live_majority
            ),
            "mech449_ablation_discriminates": bool(
                arm_lpfc_449_live_majority != arm_none_449_live_majority
            ),
        },
        "arm_lpfc_mech448_live_seeds": int(n_lpfc_448_live),
        "arm_lpfc_mech449_live_seeds": int(n_lpfc_449_live),
        "arm_none_mech448_live_seeds": int(n_none_448_live),
        "arm_none_mech449_live_seeds": int(n_none_449_live),
        "arm_lpfc_mech448_live_majority": arm_lpfc_448_live_majority,
        "arm_lpfc_mech449_live_majority": arm_lpfc_449_live_majority,
        "arm_none_mech448_live_majority": arm_none_448_live_majority,
        "arm_none_mech449_live_majority": arm_none_449_live_majority,
        "summary": {
            "route_source_confirmed_causal_channel_route_bias_drives_collapse": (
                "At FULL training budget, ARM_LPFC (route_source='lateral_pfc') "
                "reproduces V3-EXQ-851's dead MECH-448/449 on a majority of seeds; "
                "ARM_NONE (route_source='none') restores live MECH-448/449 on a "
                "majority of seeds. The routed lateral_pfc signal ITSELF is "
                "causally responsible, and reading (b) (duration-dependence) is "
                "CONFIRMED -- 859's short probe simply ran too short to see it. "
                "V3-EXQ-858 should NOT resume with lateral_pfc routing as-is; hold "
                "for a substrate fix or redesign the routing."
            ),
            "route_source_ruled_out_collapse_persists_without_routing": (
                "At FULL training budget, BOTH arms show dead MECH-448/449 on a "
                "majority of seeds, including ARM_NONE. The routed lateral_pfc "
                "signal is NOT the cause -- reading (a) is CONFIRMED, and "
                "something else in the matched stack (independent of route_source) "
                "is responsible. V3-EXQ-858 should NOT resume on the assumption "
                "that ablating the route fixes it; the real cause is still open "
                "and needs /implement-substrate or a full matched-stack re-audit."
            ),
            "mech448_449_live_under_lateral_pfc_route_851_not_replicated_at_full_budget": (
                "Both arms show LIVE MECH-448/449 on a majority of seeds even at "
                "FULL training budget matching V3-EXQ-851's own schedule exactly. "
                "This means neither reading (a) nor (b) as originally framed holds "
                "cleanly: duration alone does not reproduce 851's collapse, so "
                "851's own finding likely depended on something this replication "
                "did not hold constant (a seed-order effect, a substrate drift "
                "between when 851 and this run executed, or an interaction this "
                "diagnostic's narrower readout cannot see). Flag for expert "
                "review / /failure-autopsy rather than self-routing further -- "
                "V3-EXQ-858 should NOT be assumed safe on this basis alone."
            ),
            "mixed_partial_result_needs_expert_review": (
                "MECH-448 and MECH-449 disagree with each other within an arm, or "
                "the ablation's effect direction was not clean across both "
                "mechanisms, even at full training budget. Report the raw "
                "per-seed/per-arm numbers to the user for expert adjudication; do "
                "not treat this as a clean discrimination either way."
            ),
            "unexpected_ablation_reverses_engagement_needs_followup": (
                "ARM_LPFC shows LIVE MECH-448/449 while ARM_NONE shows DEAD -- the "
                "opposite of the hypothesis this diagnostic was built to test. "
                "Worth a closer look before drawing any conclusion; report to the "
                "user rather than self-routing further."
            ),
            "substrate_not_ready_requeue": (
                "The genuine-fresh-selection sample was too small on at least one "
                "cell to trust the live/dead read, even at full training budget "
                "(should not happen at P2=60 matching 851's own calibrated floor -- "
                "if this fires, investigate the E3 reselection cadence for a "
                "regression before drawing any conclusion about route_source "
                "causality)."
            ),
        }.get(label, ""),
    }

    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "interpretation_label": label,
        "interpretation": interpretation,
        "arm_results": arm_results,
        "seeds": seeds,
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "thresholds": {
            "demotion_active_frac_floor": float(DEMOTION_ACTIVE_FRAC_FLOOR),
            "excluded_count_floor": float(EXCLUDED_COUNT_FLOOR),
            "nogo_active_frac_floor": float(NOGO_ACTIVE_FRAC_FLOOR),
            "nogo_suppressed_floor": float(NOGO_SUPPRESSED_FLOOR),
            "min_seeds_for_live": int(MIN_SEEDS_FOR_LIVE),
            "fresh_select_floor": int(FRESH_SELECT_FLOOR),
            "beta_rate_max_steps": int(BETA_RATE_MAX_STEPS),
        },
        "arm_lpfc_engaged_majority": arm_lpfc_engaged,
        "arm_none_engaged_majority": arm_none_engaged,
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "evidence_direction_note": (
            f"V3-EXQ-863: FULL-TRAINING-BUDGET replication of V3-EXQ-859's "
            f"modulatory_channel_route_source ('lateral_pfc' vs 'none') ablation, "
            f"at the SAME seeds (42/43/44) as V3-EXQ-851/859, matching V3-EXQ-851's "
            f"own P0=200/P1=90/P2=60 episode schedule exactly (859 used P0=10/P2=10, "
            f"no P1). Tracks ONLY MECH-448 (f_eligibility_demotion_active_frac + "
            f"mean excluded_count) and MECH-449 (go_nogo_active_frac + mean "
            f"safety+soft suppressions) engagement, latch-cleared on every P2 "
            f"tick -- the SAME readout as V3-EXQ-859, not a rebuild of V3-EXQ-851's "
            f"full C1(a-g)/C2 falsifier apparatus. claim_ids=[] -- this run cannot "
            f"support/weaken MECH-309/ARC-062; it exists to resolve whether "
            f"V3-EXQ-851's MECH-448 collapse under lateral_pfc routing is "
            f"training-duration-dependent (V3-EXQ-859's own open question) and to "
            f"determine whether the suspended V3-EXQ-858 allocation can be "
            f"unblocked. interpretation_label={result['interpretation_label']}. "
            f"See failure_autopsy_V3-EXQ-859_2026-08-01.md for the full context "
            f"this replication answers."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "arms": "ARM_LPFC (route_source='lateral_pfc') vs ARM_NONE (route_source='none')",
            "swept_variable": "modulatory_channel_route_source",
            "use_candidate_rule_field": "True on BOTH arms (matched constant)",
            "use_f_eligibility_demotion": USE_F_ELIGIBILITY_DEMOTION,
            "use_go_nogo_constitution": USE_GO_NOGO_CONSTITUTION,
            "use_dacc": USE_DACC,
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "use_modulatory_channel_routing": USE_MODULATORY_CHANNEL_ROUTING,
            "p0_warmup_episodes": P0_WARMUP_EPISODES,
            "p1_bias_train_episodes": P1_BIAS_TRAIN_EPISODES,
            "p2_measurement_episodes": P2_MEASUREMENT_EPISODES,
            "budget_note": (
                "IDENTICAL to V3-EXQ-851's own P0=200/P1=90/P2=60 schedule (350 "
                "episodes/cell x 200 steps x 3 seeds x 2 arms = 420000 ticks, "
                "~7.67h on the cloud fleet per 851's own measured elapsed_seconds "
                "=27623) -- 20x longer P0 and 6x longer P2 than V3-EXQ-859's "
                "short-budget probe, with a P1 REINFORCE bias-head training phase "
                "859 did not run at all. See module docstring 'WHAT CHANGED'."
            ),
        },
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="V3-EXQ-863 full-budget replication of V3-EXQ-859: lateral_pfc route_source vs none ablation, MECH-448/449 engagement"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_BIAS_TRAIN_EPISODES
        p2 = P2_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds, p0_episodes=p0, p1_episodes=p1, p2_episodes=p2,
        steps_per_episode=steps, dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

    elapsed_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=elapsed_seconds,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"manifest: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"lpfc_engaged={result['arm_lpfc_engaged_majority']} "
        f"none_engaged={result['arm_none_engaged_majority']}",
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
