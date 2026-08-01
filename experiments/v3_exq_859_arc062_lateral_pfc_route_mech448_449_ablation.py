#!/opt/local/bin/python3
"""
V3-EXQ-859 -- ARC-062/MECH-309 TARGETED DIAGNOSTIC: does routing
modulatory_channel_route_source="lateral_pfc" itself cause the MECH-448/449
collapse observed in V3-EXQ-851, or does the collapse persist even with the
channel unrouted ("none")?

WHY THIS EXISTS (failure_autopsy_V3-EXQ-851_2026-08-01). V3-EXQ-851 (a ~7.7-hour,
350-episode-per-run, 3-seed full falsifier) found that switching
modulatory_channel_route_source from 'cand_world_summary' (its template
v3_exq_654j's value, where MECH-448 measured 17.76 and MECH-449 measured 1.549 --
both robustly live) to 'lateral_pfc' caused BOTH MECH-448 (F-eligibility demotion)
and MECH-449 (active Go/No-Go) to measure EXACTLY 0.0 (completely dead) -- under
IDENTICAL seeds (42/43/44) and identical USE_F_ELIGIBILITY_DEMOTION=True /
USE_GO_NOGO_CONSTITUTION=True flags. Same seeds rules out random variance; this is
a deterministic same-seed collapse. The autopsy's user-confirmed routing: "a
TARGETED DIAGNOSTIC isolating whether channel_route_bias itself (a numerically
near-zero-range signal per the failing C1g check) is causally responsible -- e.g.
via a controlled ablation comparing modulatory_channel_route_source='lateral_pfc'
vs 'none' at the SAME seeds, tracking mech448/449 engagement -- BEFORE spending
another ~7.7 hours re-running the same full design blind. Do not requeue the same
design as-is."

THE ABLATION. ree_core/agent.py's REEAgent.select_action elif chain
(modulatory_channel_route_source dispatch, ~line 7454-7480): "lateral_pfc" routes
_bdc_lpfc (the SD-033a LateralPFCAnalog rule-apprehension bias) identity-wise into
channel_route_bias. The literal string "none" (or any unrecognised value) matches
NO branch in the elif chain -- _route_repr stays None, so channel_route_bias stays
None, which the code's own comment states is "bit-identical" to routing being off.
use_modulatory_channel_routing itself STAYS True on BOTH arms here (only the source
string differs) -- this is the clean, minimal ablation: it isolates whether the
routed lateral_pfc signal ITSELF is the causal factor, changing nothing else in the
matched stack.

WHAT THIS SCRIPT DOES NOT DO (deliberately, per the autopsy's own "not the full
falsifier design" routing, and confirmed with the user). This is NOT a rebuild of
V3-EXQ-851's C1(a-g) readiness-gate + C2 committed-class-entropy falsifier
apparatus. It does not test use_candidate_rule_field (fixed True on BOTH arms
here -- CRF/lateral_pfc must be BUILDING a rule_state for the ablation to be
meaningful), does not compute GAP-A consumed-summary divergence, CRF
differentiation/maturity readouts, committed-class-axis exercisability, propagation
non-vacuity, committed-class entropy, or a DV-symmetry / declared-null three-branch
interpretation grid. The ONLY readouts are the raw quantities V3-EXQ-851's own
C1(e)/C1(f) preconditions measured (mech448_demotion_lever, mech449_active_nogo),
read with the SAME attribute names/paths off agent.e3.last_score_diagnostics,
latch-cleared on every tick (see below) so every reading is a genuine fresh
E3.select() this tick, never a held/replayed snapshot.

BUDGET -- WHY THIS IS SHORT (deliberately, and the risk this accepts). V3-EXQ-851 /
654j used P0=200 (e2_world_forward SD-056 online-trained + CRF field
matured+maintained) + P1=90 (bias-head REINFORCE) + P2=60 (frozen measurement) = 350
episodes x 200 steps x 3 seeds x 2 arms = 420000 total env ticks, elapsed_seconds =
27623 (~7.67h), i.e. ~0.066 s/tick blended across phases on the cloud fleet. An
empirical timing probe of the IDENTICAL matched-stack agent+env config on THIS
machine (Mac, darwin-arm64, torch 2.12.0; ~800 steady-state ticks, JIT/warmup
excluded) measured ~0.084-0.105 s/tick with NO training overhead -- i.e. even a
per-cell budget matching 851's own P0 alone (200 episodes = 40000 ticks) would cost
several HOURS for this diagnostic, which directly contradicts the "cheap diagnostic,
not another blind multi-hour run" routing this script exists to satisfy. So P0/P2
here are cut far below 851/654j's calibrated values:
  P0_WARMUP_EPISODES = 10 (2000 ticks/cell) -- NOT the 654j-validated 200 (or even
    the pre-654c 150). This is the single biggest risk this script accepts: MECH-448
    (F-eligibility demotion) and MECH-449 (Go/No-Go) are NOT gated on CRF/rule-field
    maturity per se -- e3_selector.py's f_eligibility/go_nogo code paths read the
    e2_world_forward-derived candidate merit distribution (F) and the dACC
    recency-share vector, independent of candidate_rule_field/lateral_pfc's own
    output (confirmed by inspection: neither mechanism's code path references
    agent.candidate_rule_field or agent.lateral_pfc). Both quantities can show
    non-trivial cross-candidate STRUCTURE well before either component is
    "trained" in the accuracy sense -- F-eligibility gates on SPREAD/differentiation
    (excluded_count>0), not on predictive accuracy, and dACC recency-share is a
    non-parametric running statistic over committed-action history, not a trained
    quantity, so it only needs enough STEPS (not gradient updates) to become
    non-uniform. This is why a much-shorter-than-654j P0 is judged safe enough to
    TRY here -- but it is a real, accepted risk, not a proof. If BOTH arms read dead
    on both mechanisms, this script's own interpretation label
    (inconclusive_undertrained_cannot_discriminate) says so explicitly rather than
    reading that outcome as evidence for either hypothesis (see the interpretation
    grid below) -- this is the failure mode that keeps a too-short P0 from silently
    producing a confidently wrong verdict.
  P2_MEASUREMENT_EPISODES = 10 (2000 ticks/cell) -- "a few thousand P2 ticks" per
    the routing note, NOT 851's 60 (12000 ticks). At the MECH-093-modulated E3
    reselection cadence (nominal e3_steps_per_tick=10, worst case
    beta_rate_max_steps=20; ree_core/heartbeat/clock.py), 2000 P2 ticks yields
    roughly 100-200 GENUINE fresh E3.select() calls per cell -- a modest but
    workable sample for a coarse live/dead fraction read (851 required >=600 for
    its own formal C1g gate; this diagnostic reports the achieved sample size
    plainly rather than gating on a formal floor, since the question here is
    qualitative discrimination, not a publishable effect size).
  NO P1 phase at all -- the question is whether MECH-448/449 STRUCTURALLY ENGAGE
    (active_frac, excluded/suppressed counts), not whether a TRAINED bias head
    produces a measurable behavioural effect, so the P1 REINFORCE bias-head
    training phase (854j/851's GAP-D) is out of scope entirely.
Total: 20 episodes/cell x 200 steps x 3 seeds x 2 arms = 24000 ticks. At the
measured ~0.084-0.105 s/tick this Mac probe found (no training overhead priced
in -- P0 will run a bit slower than this due to the SD-056 e2 contrastive step;
P2 has no training and should track the probe closely), the design targets
roughly 35-45 minutes wall clock -- "minutes to low tens of minutes", not hours,
per the routing instruction; estimated_minutes is set generously (45) to allow for
machine variance since machine_affinity is "any". Timing probe script (not
committed; throwaway, run during authoring):
  n_ticks=800 (100 warmup excluded), elapsed=58.921s, per_tick=84.173ms.

GOV-REUSE-1 (existing-evidence check). Decisive readout: MECH-448
(demotion_active_frac + mean f_eligibility_excluded_count) and MECH-449
(nogo_active_frac + mean go_nogo_n_safety_nogo+go_nogo_n_soft_applied) under
modulatory_channel_route_source='lateral_pfc' vs 'none', SAME seeds (42/43/44),
CRF ON on both arms. Checked: V3-EXQ-851 tested 'lateral_pfc' alone with NO 'none'
comparator arm in the same run (route source was a matched CONSTANT there, not the
swept variable -- the swept variable was use_candidate_rule_field). V3-EXQ-654j
tested a DIFFERENT route_source value entirely ('cand_world_summary'), not the
'none' ablation this diagnostic needs. No manifest anywhere sweeps
modulatory_channel_route_source across {'lateral_pfc', 'none'} at matched seeds.
Not recoverable by reanalysis -> proceed to author.

Re-derive brake (Step 2.5b): claim_ids=[]. Per the autopsy's own re_derive_brake
field: fired=false, prior_substrate_ceiling_autopsies=[] (0 -- this autopsy's own
category is measurement_test_design_defect, not substrate_ceiling). Brake does not
apply.

RELEVANCE. This diagnostic's result determines whether the currently-SUSPENDED
V3-EXQ-858 (a 1200-minute, 4-rung f_weight ladder run that reuses the IDENTICAL
lateral_pfc-routed + MECH-448/449-active matched-stack config as V3-EXQ-851 --
GOV-FANOUT-1 Leg P-B) can safely resume as-is (if 'none' also collapses -- the
matched stack itself is broken, independent of routing, and 858 would need a
different fix), needs a redesign (if 'lateral_pfc' alone is the cause -- 858
inherits the causal bug and should not resume until routed), or should be held for
an /implement-substrate fix (if the collapse traces to something in agent.py/
e3_selector.py rather than the route-source dispatch itself). Its priority (42) is
set above the portfolio's ordinary diagnostic priority (P-D's 40) because it gates
a suspended 20-hour cloud allocation.

See REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-851_2026-08-01.md (+ its
.json sibling) for the full autopsy this diagnostic answers,
experiments/v3_exq_851_arc062_pa_lateral_pfc_route_source_gapfanout.py for the
matched-stack config + latch-clearing pattern this reuses, ree_core/agent.py
(~line 7454) for the route-source dispatch this ablates.

claim_ids = [] (diagnostic; discriminates a causal mechanism, does not test a claim
hypothesis). experiment_purpose = "diagnostic". This run cannot support or weaken
MECH-309 / ARC-062 by itself -- it only routes GOV-FANOUT-1 Leg P-A follow-on work
and the suspended V3-EXQ-858 allocation.
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

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator


EXPERIMENT_TYPE = "v3_exq_859_arc062_lateral_pfc_route_mech448_449_ablation"
QUEUE_ID = "V3-EXQ-859"
SUPERSEDES = None
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

# The one precondition this script declares (p2_fresh_select_sample_adequate_both_arms)
# is a live-measurement COUNT aggregate (min n_p2_fresh_select across all 6 cells vs a
# cadence-derived floor) -- not a known-degenerate-reference anchor asserting a control
# signature. There is no predicate-vs-frozen-reference mismatch for the reachability
# check to guard against here (the same reasoning V3-EXQ-858 used for its own
# structurally identical exemption).
ANCHOR_REACHABILITY_EXEMPT = (
    "p2_fresh_select_sample_adequate_both_arms is a live-measurement count aggregate, "
    "not a known-degenerate-reference anchor; see V3-EXQ-858's identical exemption "
    "reasoning for the same precondition shape."
)

SEEDS = [42, 43, 44]

# ----- Budget (see module docstring "BUDGET" section for the full derivation) -----
P0_WARMUP_EPISODES = 10
P2_MEASUREMENT_EPISODES = 10
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# ----- MECH-448/449 readiness thresholds -- REUSED VERBATIM from V3-EXQ-851's
# C1(e)/C1(f) preconditions (same names, same values; see that script's own
# DEMOTION_ACTIVE_FRAC_FLOOR / NOGO_ACTIVE_FRAC_FLOOR block for the calibration
# history). Used here only as a live/dead READ per arm x seed, not as a
# PASS/FAIL gate on a scientific criterion (this is a mechanism-isolation
# ablation, not a claim test).
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8
EXCLUDED_COUNT_FLOOR = 0.0
NOGO_ACTIVE_FRAC_FLOOR = 0.8
NOGO_SUPPRESSED_FLOOR = 0.0

# Majority-of-seeds convention (851/654j precedent).
MIN_SEEDS_FOR_LIVE = 2  # of 3

# Sample-adequacy readiness precondition (the ONE precondition this diagnostic
# gates on -- see the diagnostic-adjudication note in the module docstring).
# Cadence-derived worst-case floor, same formula/constant as 851's C1g
# (nominal P2 window ticks / beta_rate_max_steps).
BETA_RATE_MAX_STEPS = 20  # ree_core/heartbeat/clock.py MECH-093 slowest E3-reselection cadence
FRESH_SELECT_FLOOR = (P2_MEASUREMENT_EPISODES * STEPS_PER_EPISODE) // BETA_RATE_MAX_STEPS  # 100

# ----- Matched-stack constants -- IDENTICAL to V3-EXQ-851 / 654j (env, MECH-448,
# MECH-449, modulatory selection authority, channel routing, CRF maturity levers,
# SD-056 online e2 training). ONLY modulatory_channel_route_source varies (the
# swept variable), and use_candidate_rule_field is FIXED True on both arms
# (851/654j's ARM_ON; CRF must build a rule_state for the ablation to test
# anything).
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

# CRF maturity/maintenance levers (851/654-lineage; matched constants, inert if
# CRF were off, but CRF is ON here on both arms).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# SD-056 online e2 training (mirror 851/649).
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

# IDENTICAL env to V3-EXQ-851/654 (SD-054 reef + hazard_food_attraction + bipartite
# layout) -- the behavioural falsifier substrate.
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
    """Matched-stack agent identical to V3-EXQ-851/654j; the ONLY varied flag is
    modulatory_channel_route_source. use_candidate_rule_field is FIXED True (both
    arms build a real rule_state) -- unlike 851, where CRF on/off was itself the
    swept variable.
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


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    import random
    from collections import deque

    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env, str(arm["modulatory_channel_route_source"]))
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    transition_buffer = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p2_episodes
    p2_start = p0_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p2_ticks = 0
    n_p0_contrastive_steps = 0

    # ----- MECH-448/449 latch-cleared P2 readouts -----
    n_p2_fresh_select = 0
    n_p2_latched_ticks = 0
    demotion_active_ticks = 0
    demotion_excluded_counts: List[float] = []
    nogo_active_ticks = 0
    nogo_suppressed_per_tick: List[int] = []
    # Context-only (not gating): the routed channel's own range/activity, the
    # same statistic 851's C1g precondition measured -- reported here purely as
    # supporting context for the interpretation, since 851's C1g independently
    # failed (route range sub-floor) alongside MECH-448/449.
    route_ranges: List[float] = []
    route_active_ticks = 0

    for ep in range(total_train_eps):
        is_p2 = ep >= p2_start
        phase_label = "P2" if is_p2 else "P0"

        _, obs_dict = env.reset()
        agent.reset()
        z_self_prev = None
        action_prev = None
        pending_capture = None
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

            # V3-EXQ-859 latch-clearing (851/791a pattern -- the ~9x
            # pseudo-replication defect). Unlike 851's own C1e/C1f (which
            # deliberately stay bit-identical to 654j's held/latched calibration),
            # EVERY read here is latch-cleared: we only ever record a diagnostic
            # from a genuine fresh E3.select() this tick.
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
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            if (not is_p2) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                loss_val = _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)
                if loss_val is not None and math.isfinite(loss_val):
                    n_p0_contrastive_steps += 1

            _, harm_signal, done, info, obs_dict = env.step(action)
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

        if (ep + 1) % 5 == 0 or (ep + 1) == total_train_eps:
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

    passed = f"PASS" if error_note is None else "FAIL"
    print(f"verdict: {passed}", flush=True)

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "modulatory_channel_route_source": str(arm["modulatory_channel_route_source"]),
        "error_note": error_note,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
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
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []

    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} (route_source={arm['modulatory_channel_route_source']}) "
            f"(P0={p0_episodes} ep, P2={p2_episodes} ep, "
            f"steps_per_episode={steps_per_episode}, dry_run={dry_run})",
            flush=True,
        )
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(arm, s, p0_episodes, p2_episodes, steps_per_episode)
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

    # ----- Sample-adequacy readiness precondition (the ONE gate this diagnostic
    # applies -- see the diagnostic-adjudication note in the module docstring). -----
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
        label = "mech448_449_live_under_lateral_pfc_route_851_not_replicated_at_this_dose"
        outcome = "PASS"
    elif arm_lpfc_engaged and (not arm_none_engaged):
        label = "unexpected_ablation_reverses_engagement_needs_followup"
        outcome = "PASS"
    else:
        label = "mixed_partial_result_needs_full_replication"
        outcome = "PASS"

    # If either arm's two mechanisms disagree with each other (448 live but 449
    # dead, or vice versa) at the majority-of-seeds level, flag it rather than
    # silently folding it into the combined "engaged" boolean above.
    mech448_449_agree_lpfc = arm_lpfc_448_live_majority == arm_lpfc_449_live_majority
    mech448_449_agree_none = arm_none_448_live_majority == arm_none_449_live_majority
    if not (mech448_449_agree_lpfc and mech448_449_agree_none) and sample_adequate:
        label = "mixed_partial_result_needs_full_replication"

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
                    "not trustworthy (too few genuine selections observed) -> "
                    "substrate_not_ready_requeue, never a false discrimination."
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
            # Did the ablation actually discriminate anything, or did both arms
            # read identically (uninformative in a different way than a failed
            # sample-adequacy gate -- this is a DEGENERATE ablation if arms never
            # differ across the whole readout, which is itself worth flagging).
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
                "ARM_LPFC (route_source='lateral_pfc') reproduces V3-EXQ-851's dead "
                "MECH-448/449 on a majority of seeds; ARM_NONE (route_source='none', "
                "channel_route_bias=None) restores live MECH-448/449 on a majority "
                "of seeds. The routed lateral_pfc signal ITSELF is causally "
                "responsible -- V3-EXQ-858 should NOT resume with lateral_pfc "
                "routing as-is; hold for a substrate fix or redesign the routing."
            ),
            "route_source_ruled_out_collapse_persists_without_routing": (
                "BOTH arms show dead MECH-448/449 on a majority of seeds, including "
                "ARM_NONE where channel_route_bias is None (bit-identical to routing "
                "off per the code comment). The routed lateral_pfc signal is NOT the "
                "cause -- something else in the matched stack (or under-maturation "
                "at this diagnostic's short P0 -- see the module docstring's "
                "explicit risk note) is responsible. V3-EXQ-858 should NOT resume "
                "on the assumption that ablating the route fixes it; the real cause "
                "is still open and needs /implement-substrate or a full "
                "matched-stack re-audit, not a routing tweak."
            ),
            "mech448_449_live_under_lateral_pfc_route_851_not_replicated_at_this_dose": (
                "Both arms show LIVE MECH-448/449 on a majority of seeds -- V3-EXQ-851's "
                "collapse under lateral_pfc routing did NOT reproduce at this "
                "diagnostic's much shorter budget. This does not confirm 851's "
                "finding was wrong; it may mean the collapse is dose-dependent "
                "(emerges only under 851's fuller P0/P1 maturation) rather than "
                "structural. V3-EXQ-858 should NOT be assumed safe on this basis "
                "alone -- flag for a closer look at what differs between this "
                "diagnostic's short schedule and 851's full one."
            ),
            "mixed_partial_result_needs_full_replication": (
                "MECH-448 and MECH-449 disagree with each other within an arm, or "
                "the ablation's effect direction was not clean across both "
                "mechanisms. Report the raw per-seed/per-arm numbers to the user; "
                "do not treat this as a clean discrimination either way."
            ),
            "unexpected_ablation_reverses_engagement_needs_followup": (
                "ARM_LPFC shows LIVE MECH-448/449 while ARM_NONE shows DEAD -- the "
                "opposite of the hypothesis this diagnostic was built to test. "
                "Worth a closer look before drawing any conclusion; report to the "
                "user rather than self-routing further."
            ),
            "substrate_not_ready_requeue": (
                "The genuine-fresh-selection sample was too small on at least one "
                "cell to trust the live/dead read (this diagnostic's short P2 "
                "budget may simply be too short at the observed E3 reselection "
                "cadence). Re-run with a longer P2 (and/or investigate the cadence) "
                "before drawing any conclusion about route_source causality."
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
            f"V3-EXQ-859: targeted diagnostic ablating "
            f"modulatory_channel_route_source ('lateral_pfc' vs 'none') at the SAME "
            f"seeds (42/43/44) as V3-EXQ-851, tracking ONLY MECH-448 "
            f"(f_eligibility_demotion_active_frac + mean excluded_count) and "
            f"MECH-449 (go_nogo_active_frac + mean safety+soft suppressions) "
            f"engagement, latch-cleared on every tick. claim_ids=[] -- this run "
            f"cannot support/weaken MECH-309/ARC-062; it exists to route "
            f"GOV-FANOUT-1 Leg P-A follow-on work and the suspended V3-EXQ-858 "
            f"allocation. interpretation_label={result['interpretation_label']}. "
            f"See failure_autopsy_V3-EXQ-851_2026-08-01.md for the full context "
            f"this diagnostic answers."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "arms": "ARM_LPFC (route_source='lateral_pfc') vs ARM_NONE (route_source='none')",
            "swept_variable": "modulatory_channel_route_source",
            "use_candidate_rule_field": "True on BOTH arms (matched constant, unlike 851)",
            "use_f_eligibility_demotion": USE_F_ELIGIBILITY_DEMOTION,
            "use_go_nogo_constitution": USE_GO_NOGO_CONSTITUTION,
            "use_dacc": USE_DACC,
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "use_modulatory_channel_routing": USE_MODULATORY_CHANNEL_ROUTING,
            "p0_warmup_episodes": P0_WARMUP_EPISODES,
            "p2_measurement_episodes": P2_MEASUREMENT_EPISODES,
            "p1_phase": "SKIPPED (structural engagement question, not a trained-bias behavioural question)",
            "budget_note": (
                "Much shorter than V3-EXQ-851/654j's P0=200/P1=90/P2=60 -- see the "
                "module docstring BUDGET section for the timing-probe-derived "
                "justification and the explicit false-null risk this accepts."
            ),
        },
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="V3-EXQ-859 ARC-062/MECH-309 diagnostic: lateral_pfc route_source vs none ablation, MECH-448/449 engagement"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p2 = P2_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds, p0_episodes=p0, p2_episodes=p2,
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
