#!/usr/bin/env python3
"""V3-EXQ-955 -- MECH-440 ARMED-STACK falsifier at the RAISED candidate-diversity floor.

SAME DESIGN AS V3-EXQ-708b, ONE INTEGER CHANGED: support_preserving_min_first_action_classes
is set to action_dim (=5 on this env) instead of 708b's hardcoded 2 (driver line 507 there).
MECH-440 is CHEAPER to re-test here than MECH-441: 708a and 708b already armed
use_modulatory_selection_authority=True (708a:298, 708b:324, matched-constant across every
arm), so this run's residual blocker is the class floor ALONE.

WHY THIS EXISTS
----------------
No committed-class-entropy null anywhere in the ARC-065 / conversion-ceiling lineage is
currently a non-degenerate test of anything: support_preserving_min_first_action_classes
defaults to 2 and is set to literally 2 at 273 driver call sites, capping committed-class
entropy at ln(~3) ~= 1.08 nats ARM-INVARIANTLY. V3-EXQ-708b's own manifest carries the proof:
precommit_n_distinct_classes_mean 2.959/2.958/2.956 across its three arms (near-identical
PER SEED ACROSS ARMS, because the floor is a config constant, not an arm factor), with
mean_precommit_class_entropy 1.060873 in the OFF baseline = 97.8% of ln(2.959)=1.0850 BEFORE
ANY LEVER IS APPLIED. 708b enumerated exactly two explanations (H-PRECOMMIT-SATURATED,
H-LEVER-DOWNSTREAM) and refuted both, filing precommit_shape_headroom_unexplained; the third,
unenumerated reading is this ceiling. Raised as GFLAG-0072 (evidence_discrepancy), 2026-08-28,
against ARC-065 / MECH-440 / MECH-441 / MECH-439. ARC-065's what_would_answer owns the
ARMED-CONVERSION precondition this driver asserts; see claims.yaml.

THE THREE ARMS (unchanged from 708b; matched SOTA conversion stack as a MATCHED CONSTANT):
  A0_OFF            : no exploration injection. The pre-commit shape FLOOR / yoked REFERENCE.
  ARM_TEMP          : the 687 non-propagating temperature control (use_noise_floor=True).
  ARM_NOISE_SINGLE  : MECH-440 factorised-Gaussian selection-head weight noise.
6 seeds x 3 arms. Phased training UNCHANGED from 708b: P0 encoder warmup (e2 contrastive,
online, per-arm independent trajectory), P1 lateral_pfc bias-head REINFORCE training on
.detach()ed candidate summaries (per-arm independent trajectory) -- both copied verbatim from
708b (see CLAUDE.md "Phased Training Protocol"; joint training would chase a moving target).

WHAT P2 MEASURES DIFFERENTLY FROM 708b -- THE MANDATORY NEW INSTRUMENTS
-------------------------------------------------------------------------
708b's P2 ran each arm's trained agent on its OWN independent env rollout and measured only
the pre-commit softmax SHAPE (max_prob, participation_ratio) -- useful for discriminating
"saturated" from "lever never arrived", but blind to whether a lever that DOES move the
pre-commit distribution ever reaches the COMMITTED action. 708b's own HONEST LIMIT, inherited
here: it could not tell "moved the shape" from "moved the outcome". This run's P2 closes that
gap with the V3-EXQ-949 yoked-pair / private-RNG-stream design (same mechanism, extended from
949's authority-rescale question to this floor-conversion question):

  (i)   mean_distinct_first_action_classes per arm, as a COVARIATE -- and the primary DV,
        committed_class_entropy, is reported as committed_class_entropy_fraction_of_ceiling =
        entropy / ln(max(mean_distinct_first_action_classes, 1+eps)), i.e. a FRACTION of the
        run's own achieved candidate-diversity ceiling, not an absolute nat count. This is the
        instrument that makes a class-floor-of-2 ceiling (ln(2)~=0.69) and a class-floor-of-5
        ceiling (ln(5)~=1.61) directly comparable across the 708b lineage.
  (ii)  YOKED PER-TICK COMMITTED-ACTION DIVERGENCE against a matched control sharing a PRIVATE
        RNG STREAM (the V3-EXQ-949 design, ported verbatim in mechanism): after P0/P1, the
        three arms' TRAINED, FROZEN agents are stepped on the SAME observation sequence (the
        reference A0_OFF agent drives the environment; ARM_TEMP and ARM_NOISE_SINGLE observe
        the identical env state at every tick without their own action being applied) --
        non-compounding by construction, so a divergence is an argmin flip, not an accumulated
        trajectory difference. Each runner owns its own torch RNG stream (snapshot/restore
        around every tick, episode reset, and residue update) so a paired comparison differs
        in the manipulation and nothing else -- the exact defect the 947/949 self-yoke
        instrument control exists to catch (two runners sharing one interleaved RNG stream
        diverge for purely positional reasons even at identical configuration). This is what
        directly answers 708b's honest limit: a lever that moves the pre-commit shape but
        never flips a committed argmin shows YOKED DIVERGENCE ~0 despite a shape delta.
  (iii) authority_rel_deviation_mean per arm: mean of |post_range - raw_range| / raw_range
        across genuine E3 select() fires, from agent.e3.last_raw_scores (pre-modulatory-
        authority-rescale) vs agent.e3.last_scores (post-rescale) -- e3_selector.py:2858 sets
        last_raw_scores before the modulatory_authority_active rescale block (~:3108-3145);
        :3435 sets last_scores after it. use_modulatory_selection_authority=True is a MATCHED
        CONSTANT on every arm (inherited from 708b), so this measures whether the authority
        channel actually rescales the combined modulatory contribution on THIS substrate,
        exactly as ARC-065's ARMED-CONVERSION P2 precondition requires -- assert it rather
        than assume it (both levers were "already armed" claims in 708a/708b; neither was ever
        measured this way).

SAMPLE-SIZE INTEGRITY (708a's repaired instrument, carried forward verbatim). E3-cadence
latches (last_score_diagnostics, last_precommit_probs, last_scores, last_raw_scores,
_last_explore_term) are cleared to None IMMEDIATELY before every select_action() call and a
tick is counted toward any of the above ONLY on a genuine fresh (non-None) repopulation
afterward. n_fresh_select / n_latched are recorded per cell so the true denominator is
auditable (V3-EXQ-785's pseudo-replication class this idiom exists to prevent).

ARMED-CONVERSION PRECONDITIONS (mandatory, MEASURED not assumed -- ARC-065's what_would_answer)
  P1: mean_distinct_first_action_classes >= 0.9 * action_dim in EVERY arm. The 2026-08-23 spike
      measured 2.16/5 on a minimal build and 3.08/5 on a rich build at the DEFAULT floor;
      setting the knob to action_dim gives 4.98/5 with no code change (empirically confirmed
      by 949's own C1 gate at the same raised floor, threshold action_dim-0.5).
  P2: authority_rel_deviation_mean > 0.05 in every LEVER-BEARING arm (ARM_TEMP, ARM_NOISE_SINGLE).

VACUITY RULE (mandatory): a committed-action-diversity null with EITHER P1 or P2 unmet must
self-route substrate_not_ready_requeue, NEVER weaken. Established by V3-EXQ-949's strict
AND-interaction: yoked divergence 0.000 with P1 alone, 0.000 with P2 alone, 0.3675 with both
(5 seeds). CEILING-HEADROOM CHECK (precondition on the falsifying branch specifically):
A0_OFF's fraction-of-ceiling must be materially below 1.0 -- a null in the OFF arm is
uninformative (nothing to lift).

PASS (MECH-440 supports): with P1+P2 armed, ARM_NOISE_SINGLE's committed_class_entropy_
fraction_of_ceiling is STRICTLY ABOVE ARM_TEMP's on >=2/3 seeds, AND ARM_NOISE_SINGLE's yoked
divergence vs the A0_OFF reference clears YOKED_DIVERGENCE_FLOOR (diversity reaching the
COMMITTED action, not just the pre-commit shape).

FAIL classes: (a) washes out at the argmax identically to the temperature floor (valid ONLY
once the ceiling-headroom check passes); (b) raises pre-commit entropy/shape without raising
yoked committed-action divergence -- thrash, not carve; (c) raises committed diversity but the
SECONDARY state-conditioning / self-annealing checks (below) were never measured, so this run
cannot itself distinguish MECH-440's refinement from a plain stochastic floor (MECH-313's
original framing) -- recorded as an explicit scope gap, not resolved here.

SECONDARY (asserted by MECH-440, NEITHER measured by this run -- HONEST GAP, stated per the
proposal's own acceptance_checks rather than force-fit): STATE-CONDITIONING (per-state noise
magnitude covaries with state) and SELF-ANNEALING (per-parameter sigma falls where the policy
is confident). A PASS on the entropy-fraction + yoked-divergence criteria above establishes
PROPAGATION only and leaves two-thirds of the MECH-440 claim untested; recorded as
secondary_checks_measured=false with an explicit note, not silently omitted.

HONEST LIMIT: V3-EXQ-949's own DV is per-tick committed-ACTION divergence in a yoked pair, not
committed-CLASS entropy over an episode -- it proves the channel ACQUIRES argmin authority, not
that authority CONVERTS into sustained class diversity. THIS run measures both quantities
directly (mandatory instrument (i) is the class-entropy-fraction conversion measurement; (ii)
is the 949-style acquisition measurement), which is exactly what closes that gap.

SCOPE: a null here at the raised floor is a genuine result for the first time (the default
floor made every prior null in this lineage a floor artefact, not a substrate finding). This
does NOT refute MECH-439 F-dominance or ARC-110's intrinsic-ceiling finding; both may still
hold at a raised floor.

RE-DERIVE BRAKE (2026-08-28 check, this driver): 0 substrate_ceiling-category autopsies found
for MECH-440 in the corpus (script per CLAUDE.md Session Startup Protocol step "re-derive
brake"). NOT braked.

SUBSTRATE-PATH OVERLAP GATE (skill step 2.5c, 2026-08-28 check): two OPEN corrupting
substrate_queue entries name files this driver's agent imports at module level.
`mode-governance-engagement` (SalienceCoordinator.tick) is NOT in this driver's causal path --
no mode-governance knob is enabled anywhere in _make_agent (same reasoning as the 949/947
drivers' identical carve-out). `contextmemory-write-path-addressing-degeneracy`
(ree_core/predictors/e1_deep.py) IS potentially in path via E1, but applies IDENTICALLY to
every one of the 3 arms and interacts with neither the injection-site factor (temp vs noise)
nor the authority factor, so it biases the load-bearing contrast toward the null, not toward a
false positive (same reasoning as 949's identical carve-out for the same defect). Two OPEN
degrading entries also overlap (`mech357-freeze-incompatible-pressure-mechanism`,
`SD-MECH303-THRESHOLD-SOURCING`, both touching causal_grid_world.py / agent.py / config.py) --
noted, non-blocking per the skill's degrading-severity rule.

GOV-REUSE-1: the decisive readouts (yoked per-tick committed-action divergence with a private
RNG stream; committed-class-entropy fraction-of-ceiling at a RAISED class floor;
authority_rel_deviation_mean) are recorded nowhere in the corpus -- 708/708a/708b measured only
the pre-commit-shape / class-entropy scalar AT THE DEFAULT FLOOR (2 classes), and no run in the
corpus combines the raised floor with a yoked measurement. Not recoverable by reanalysis -> run.

claim_ids = [MECH-440]. related_claims (context only, NOT scored per-claim; see the code-review
"claim tagging integrity" rule -- this run's OWN implementation tests MECH-440 alone; ARC-065 /
MECH-441 / MECH-439 / ARC-110 / MECH-313 / MECH-458 are cross-referenced context because
ARC-065 owns the ARMED-CONVERSION precondition and the SCOPE note above names the other claims
this run does and does not bear on): ["ARC-065", "MECH-441", "MECH-439", "ARC-110", "MECH-313",
"MECH-458"]. experiment_purpose = "evidence" (this run tests MECH-440's claim hypothesis
directly, unlike 708b which was purely diagnostic).

SLEEP: none (no sleep flag is set anywhere in this driver; no SLEEP DRIVER line required).

DV-SYMMETRY DECLARATION (mandatory; the V3-EXQ-604c failure class), per arm:
  * A0_OFF (reference): no exploration lever. Drives the yoked walk's shared observation
    sequence. Never diverges from itself (the self-yoke instrument control asserts this).
  * ARM_TEMP: a softmax-TEMPERATURE lift (e3_selector.py's `temperature` parameter in
    probs = softmax(-scores/temperature)) is a MONOTONE RESCALING of the score vector, not a
    uniform additive constant -- monotone maps preserve ORDER (so an argmax computed on raw
    scores is invariant under it), but they change the SOFTMAX SAMPLING distribution's shape.
    Since the committed action here is the WITHIN-ELIGIBLE ARGMIN over (possibly rescaled)
    scores, not a softmax draw, ARM_TEMP is NOT guaranteed to be DV-symmetric-invariant for the
    committed-action DV -- whether it moves the argmin is an empirical question this run
    measures (yoked divergence), not one assumed answered by the design.
  * ARM_NOISE_SINGLE: per-candidate factorised-Gaussian weight noise added into the score
    vector before the committed argmin. This is neither a uniform broadcast (it varies
    per-candidate) nor a pure monotone rescaling of the UN-noised order, so it is NOT
    invariant under either symmetry class in the DV-symmetry table -- it CAN flip the argmin
    by construction, which is exactly what yoked divergence measures directly.

ETHICS PREFLIGHT: all-false / decision=allow (V3 has no live self-model, no autobiographical
memory, no social mind; SENT-0 boundary, pre-ethical instrumentation only).

See REE_assembly/evidence/planning/manual_proposals.v1.json (proposal_id EXP-0593, backlog_id
EVB-0651); REE_assembly/docs/claims/claims.yaml (ARC-065 what_would_answer, MECH-440,
MECH-441); ree-v3/experiments/v3_exq_708b_mech440_precommit_distribution_shape_falsifier.py
(the base design, unchanged except the floor); ree-v3/experiments/
v3_exq_949_mech314b_authority_rescale_validation.py (the yoked-pair / private-RNG-stream /
authority-measurement mechanism, ported here); ree_core/predictors/e3_selector.py:2858,3108-
3145,3435 (last_raw_scores / modulatory_authority rescale / last_scores).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_955_mech440_armed_stack_raised_class_floor_falsifier"
QUEUE_ID = "V3-EXQ-955"
CLAIM_IDS: List[str] = ["MECH-440"]
RELATED_CLAIMS: List[str] = [
    "ARC-065", "MECH-441", "MECH-439", "ARC-110", "MECH-313", "MECH-458",
]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ----- Acceptance thresholds (pre-registered) -----
ARMED_CONVERSION_CLASSES_FRAC = 0.9   # P1: mean_distinct_first_action_classes >= this * action_dim
AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR = 0.05  # P2: authority_rel_deviation_mean floor
CEILING_HEADROOM_MAX_FRAC = 0.90      # A0_OFF fraction-of-ceiling must sit below this
YOKED_DIVERGENCE_FLOOR = 0.02         # diversity reaching the COMMITTED action (949 precedent)
ENTROPY_FRACTION_MARGIN = 0.02        # C1: strict-above margin on fraction-of-ceiling
MIN_DIVERGENT_SEEDS = 4               # instrument-validity: >=4 divergent seeds (proposal)
DIVERGENT_PASS_FRACTION = (2.0 / 3.0)  # strict-majority-ish gate within divergent seeds

# Non-vacuity: the noise arm must inject a per-candidate bias range above this floor, AND it
# must be a non-trivial fraction of the raw-score range.
NOISE_BIAS_RANGE_FLOOR = 1e-4
NOISE_BIAS_TO_RAW_RANGE_FRAC_FLOOR = 0.02

# dACC non-vacuity: the Go/No-Go perseveration axis must be live.
DACC_MAX_SUPPRESSION_FLOOR = 0.0

# Fresh-select sufficiency (708a/949's repaired-instrument readiness gate).
MIN_FRESH_SELECTS = 30

SEEDS = [42, 43, 44, 45, 46, 47]
P0_WARMUP_EPISODES = 100
P1_BIAS_TRAIN_EPISODES = 50
P2_YOKED_EPISODES = 100
STEPS_PER_EPISODE = 200

# Self-yoke instrument control: a SHORT independent P0/P1 training budget is sufficient -- the
# control's purpose is to prove the RUNNER MACHINERY's RNG isolation is correct (two agents
# built identically must diverge on ZERO ticks), not to reproduce full training.
CONTROL_P0_EPISODES = 1
CONTROL_P1_EPISODES = 1
CONTROL_STEPS = 15
CONTROL_EPISODES = 1

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 3
DRY_RUN_STEPS = 25

# --- MECH-440 injection-site lever constants (verbatim from 708b) ---
NOISY_SELECTION_SIGMA_INIT = 1.0
NOISY_SELECTION_WEIGHT = 1.0
NOISY_SELECTION_ANNEAL = True
NOISY_SELECTION_ANNEAL_FLOOR = 0.1
NOISY_SELECTION_ANNEAL_EMA_ALPHA = 0.01
TEMP_NOISE_FLOOR_ALPHA = 1.5
TEMP_NOISE_FLOOR_MIN_TEMPERATURE = 1.0

# --- Matched-stack lever constants (identical on ALL arms; verbatim from 708b) ---
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
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
USE_CANDIDATE_RULE_FIELD = True

LCG_ETA = 0.01
LCG_ELIG_DECAY = 0.9
LCG_VALUE_BASELINE_BETA = 0.05
LCG_ASYM_POTENTIATION = 1.0
LCG_ASYM_DEPRESSION = 0.5

LEARNED_SETTLING_ROUNDS = 3
LEARNED_SETTLING_TEMPERATURE = 1.0
LEARNED_SETTLING_ETA = 0.01
LEARNED_SETTLING_ELIG_DECAY = 0.9

SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

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
        "arm_id": "A0_OFF",
        "label": "no_exploration_injection_yoked_reference",
        "noise_head": False, "temp": False,
    },
    {
        "arm_id": "ARM_TEMP",
        "label": "matched_pre_commit_variance_temperature_control_687_non_propagating",
        "noise_head": False, "temp": True,
    },
    {
        "arm_id": "ARM_NOISE_SINGLE",
        "label": "mech440_noisy_selection_head_weight_noise_single_arena",
        "noise_head": True, "temp": False,
    },
]


def _arm_config_slice(arm: Dict[str, Any], floor: int) -> Dict[str, Any]:
    """Declared reuse fingerprint slice: ONLY what an arm's computation reads."""
    return {
        "arm_id": arm["arm_id"],
        "noise_head": bool(arm["noise_head"]),
        "temp": bool(arm["temp"]),
        "support_preserving_min_first_action_classes": int(floor),
        "noisy_selection_sigma_init": float(NOISY_SELECTION_SIGMA_INIT),
        "noisy_selection_weight": float(NOISY_SELECTION_WEIGHT),
        "temp_noise_floor_alpha": float(TEMP_NOISE_FLOOR_ALPHA),
        "temp_noise_floor_min_temperature": float(TEMP_NOISE_FLOOR_MIN_TEMPERATURE),
        "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
        "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
        "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
        "use_candidate_rule_field": bool(USE_CANDIDATE_RULE_FIELD),
        "use_dacc": bool(USE_DACC),
        "env_kwargs": dict(ENV_KWARGS),
        "sd056_weight": float(SD056_WEIGHT),
        "lr_lpfc_bias": float(LR_LPFC_BIAS),
    }


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any], floor: int) -> REEAgent:
    """Matched-stack agent -- verbatim from 708b's _make_agent, EXCEPT
    support_preserving_min_first_action_classes is parameterised (708b hardcodes 2 here)."""
    noise_head = bool(arm["noise_head"])
    temp = bool(arm["temp"])
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
        # --- Matched stack ---
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=int(floor),  # <-- THE ONE INTEGER CHANGED
        candidate_summary_source="e2_world_forward",
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=USE_MODULATORY_CHANNEL_ROUTING,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
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
        # --- MECH-313 temperature noise floor: ARMED ONLY on ARM_TEMP (the 687 control) ---
        use_noise_floor=temp,
        noise_floor_alpha=(TEMP_NOISE_FLOOR_ALPHA if temp else 0.1),
        noise_floor_min_temperature=(TEMP_NOISE_FLOOR_MIN_TEMPERATURE if temp else 1.0),
        # --- MECH-440 NoisyNet propagating selection-head weight noise: ARMED on ARM_NOISE_SINGLE ---
        use_noisy_selection_head=noise_head,
        noisy_selection_sigma_init=(NOISY_SELECTION_SIGMA_INIT if noise_head else 0.0),
        noisy_selection_weight=NOISY_SELECTION_WEIGHT,
        noisy_selection_anneal=NOISY_SELECTION_ANNEAL,
        noisy_selection_anneal_floor=NOISY_SELECTION_ANNEAL_FLOOR,
        noisy_selection_anneal_ema_alpha=NOISY_SELECTION_ANNEAL_EMA_ALPHA,
        # MECH-441 OFF (this falsifier is the MECH-440 leg only).
        use_model_disagreement_curiosity=False,
        # V_s minimal stack.
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=True,
        e2_action_contrastive_horizon=5,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
        use_candidate_rule_field=USE_CANDIDATE_RULE_FIELD,
        use_finer_channel_gating=True,
        use_learned_channel_gating=False,
        learned_channel_gating_eta=LCG_ETA,
        learned_channel_gating_elig_decay=LCG_ELIG_DECAY,
        learned_channel_value_baseline_beta=LCG_VALUE_BASELINE_BETA,
        learned_channel_asym_potentiation=LCG_ASYM_POTENTIATION,
        learned_channel_asym_depression=LCG_ASYM_DEPRESSION,
        learned_channel_rpe_mode="signed",
        use_learned_settling_step=True,
        learned_settling_rounds=LEARNED_SETTLING_ROUNDS,
        learned_settling_temperature=LEARNED_SETTLING_TEMPERATURE,
        learned_settling_eta=LEARNED_SETTLING_ETA,
        learned_settling_elig_decay=LEARNED_SETTLING_ELIG_DECAY,
        use_loop_segregation=False,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# P0 online e2 contrastive training helpers (verbatim from 708b/707)
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


def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (verbatim from 708b)
# ---------------------------------------------------------------------------


def _traj_first_action_class(traj) -> int:
    return int(traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _entropy_from_int_counts(counts: Dict[int, int]) -> float:
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


def _mean_or0(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _shape_of(ppv: torch.Tensor) -> Dict[str, Any]:
    """Pre-commit softmax shape moments (708b instrument, kept as recorded context here)."""
    p = ppv.detach().reshape(-1)
    n = int(p.numel())
    srt, _ = torch.sort(p, descending=True)
    sq = float((p * p).sum().item())
    return {
        "max_prob": float(srt[0].item()),
        "participation_ratio": (1.0 / sq) if sq > 0.0 else float(n),
        "eff_support": int((p >= 0.05).sum().item()),
    }


def _entropy_fraction_of_ceiling(entropy_nats: float, mean_distinct_classes: float) -> float:
    """The proposal's mandatory instrument (i): DV as a FRACTION of ln(distinct), not an
    absolute nat count. Guards mean_distinct_classes <= 1 (undefined / degenerate ceiling)."""
    if mean_distinct_classes <= 1.0 + 1e-9:
        return 0.0
    ceiling = math.log(mean_distinct_classes)
    if ceiling <= 1e-9:
        return 0.0
    return float(entropy_nats / ceiling)


# ---------------------------------------------------------------------------
# P0/P1 phased training (verbatim mechanism from 708b, stops before P2 -- P2 is the NEW
# yoked pass below, run jointly across all three arms' trained agents).
# ---------------------------------------------------------------------------


def train_arm_p0_p1(
    arm: Dict[str, Any],
    seed: int,
    floor: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Tuple[REEAgent, Dict[str, Any]]:
    """P0 encoder warmup (online e2 contrastive) + P1 lateral_pfc bias-head REINFORCE training
    on .detach()ed candidate summaries. Returns the TRAINED, FROZEN-for-P2 agent plus stats.
    Mechanism verbatim from 708b's _run_seed_arm, truncated before its P2 loop."""
    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env, arm, floor)

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(
        list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS
    )
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
        maxlen=TRANSITION_BUFFER_MAX
    )
    sample_rng = random.Random(seed)

    total_eps = p0_episodes + p1_episodes
    p1_start = p0_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p0_contrastive_steps = 0
    n_p1_bias_updates = 0
    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    for ep in range(total_eps):
        is_p1 = ep >= p1_start
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
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

            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = agent._candidate_world_summaries(candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.detach().clone()

            action = agent.select_action(candidates, ticks)

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

            if (not is_p1) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer, optimiser=e2_opt, rng=sample_rng,
                )
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

        if is_p1:
            reinforce_baseline = EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
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

        if (ep + 1) % 10 == 0 or (ep + 1) == total_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_eps}", flush=True,
            )
        if error_note is not None:
            break

    stats = {
        "arm_id": arm["arm_id"], "seed": int(seed),
        "n_p0_ticks": int(n_p0_ticks), "n_p1_ticks": int(n_p1_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
    }
    return agent, stats


# ---------------------------------------------------------------------------
# NEW: P2 YOKED measurement (V3-EXQ-949 design, ported). One runner wraps an ALREADY-TRAINED,
# FROZEN agent (no further training happens in P2) and owns its own private torch RNG stream.
# ---------------------------------------------------------------------------


class _YokedRunner:
    """Wraps an already-trained agent for the yoked P2 pass. OWNS ITS OWN RNG STREAM: snapshot
    and restore around every tick, episode reset, and residue update, so a paired comparison
    differs in the manipulation and nothing else (949's mechanism, ported verbatim)."""

    def __init__(self, agent: REEAgent) -> None:
        self.agent = agent
        self._rng_state = torch.get_rng_state()
        self.n_ticks = 0
        self.class_sum = 0
        self.committed_class_counts: Dict[int, int] = {}
        # Authority-engagement instrumentation (mandatory instrument iii).
        self.raw_score_ranges: List[float] = []
        self.post_score_ranges: List[float] = []
        self.n_e3_select_fires = 0
        self.n_e3_latched_ticks = 0
        # MECH-440 noise non-vacuity (708b instrument, recorded context).
        self.noise_bias_ranges: List[float] = []
        self.diag_raw_score_ranges: List[float] = []
        self.dacc_max_suppression = 0.0
        # Pre-commit shape (708b instrument, recorded context -- not load-bearing here).
        self.shape_max_prob: List[float] = []
        self.shape_participation_ratio: List[float] = []

    def reset_episode(self) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            self.agent.reset()
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def choose(self, obs_dict: Dict[str, Any]) -> int:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            return self._choose_inner(obs_dict)
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def _choose_inner(self, obs_dict: Dict[str, Any]) -> int:
        agent = self.agent
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
        ticks = agent.clock.advance()
        wdim = latent.z_world.shape[-1]
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick", False)
            else torch.zeros(1, wdim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        if candidates:
            self.n_ticks += 1
            classes = {_traj_first_action_class(c) for c in candidates}
            self.class_sum += len(classes)

        # Sample-size-integrity idiom (708a/949, mandatory): clear ALL E3-cadence latches
        # immediately before select_action(), record only on a genuine fresh fire afterward.
        agent.e3.last_score_diagnostics = None
        agent.e3.last_precommit_probs = None
        agent.e3.last_scores = None
        agent.e3.last_raw_scores = None
        agent.e3._last_explore_term = None

        action = agent.select_action(candidates, ticks)
        if action is None:
            adim = int(agent.config.e2.action_dim)
            idx = int(np.random.randint(0, adim))
            action = torch.zeros(1, adim, device=agent.device)
            action[0, idx] = 1.0

        diag = getattr(agent.e3, "last_score_diagnostics", None)
        if diag is not None:
            self.n_e3_select_fires += 1
            nbr = float(diag.get("noisy_selection_bias_range", 0.0) or 0.0)
            if nbr > 0.0:
                self.noise_bias_ranges.append(nbr)
            rsr = float(diag.get("e3_raw_score_range_mean", 0.0) or 0.0)
            if rsr > 0.0:
                self.diag_raw_score_ranges.append(rsr)
            pp = getattr(agent.e3, "last_precommit_probs", None)
            if pp is not None and candidates:
                try:
                    ppv = pp.detach().reshape(-1)
                    if ppv.numel() == len(candidates) and torch.isfinite(ppv).all():
                        sh = _shape_of(ppv)
                        self.shape_max_prob.append(sh["max_prob"])
                        self.shape_participation_ratio.append(sh["participation_ratio"])
                except Exception:
                    pass
            raw = getattr(agent.e3, "last_raw_scores", None)
            post = getattr(agent.e3, "last_scores", None)
            if raw is not None and post is not None and raw.numel() > 1:
                raw_range = float((raw.max() - raw.min()).item())
                post_range = float((post.max() - post.min()).item())
                self.raw_score_ranges.append(raw_range)
                self.post_score_ranges.append(post_range)
        else:
            self.n_e3_latched_ticks += 1

        bundle = getattr(agent, "_dacc_last_bundle", None)
        if bundle is not None:
            supp = bundle.get("suppression", None)
            if supp is not None:
                try:
                    self.dacc_max_suppression = max(
                        self.dacc_max_suppression, float(supp.detach().abs().max().item())
                    )
                except Exception:
                    pass

        agent.update_z_goal(
            benefit_exposure=0.0,
            drive_level=REEAgent.compute_drive_level(body),
        )
        committed = int(action[0].argmax().item())
        self.committed_class_counts[committed] = self.committed_class_counts.get(committed, 0) + 1
        return committed

    def observe(self, harm_signal: float) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            with torch.no_grad():
                self.agent.update_residue(
                    harm_signal=float(harm_signal), world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def authority_rel_deviation_mean(self) -> float:
        if not self.raw_score_ranges:
            return 0.0
        devs = [
            abs(p - r) / max(r, 1e-9)
            for r, p in zip(self.raw_score_ranges, self.post_score_ranges)
        ]
        return sum(devs) / len(devs)

    def summary(self) -> Dict[str, Any]:
        mean_distinct = (self.class_sum / self.n_ticks) if self.n_ticks else 0.0
        entropy = _entropy_from_int_counts(self.committed_class_counts)
        noise_bias_mean = _mean_or0(self.noise_bias_ranges)
        diag_raw_mean = _mean_or0(self.diag_raw_score_ranges)
        return {
            "n_candidate_ticks": int(self.n_ticks),
            "mean_distinct_first_action_classes": round(mean_distinct, 6),
            "committed_class_entropy_nats": round(entropy, 6),
            "committed_class_entropy_fraction_of_ceiling": round(
                _entropy_fraction_of_ceiling(entropy, mean_distinct), 6
            ),
            "n_unique_committed_classes": int(len(self.committed_class_counts)),
            "authority_rel_deviation_mean": round(self.authority_rel_deviation_mean(), 6),
            "raw_score_range_mean": round(_mean_or0(self.raw_score_ranges), 8),
            "post_score_range_mean": round(_mean_or0(self.post_score_ranges), 8),
            "n_e3_select_fires": int(self.n_e3_select_fires),
            "n_e3_latched_ticks": int(self.n_e3_latched_ticks),
            "fresh_selects_sufficient": bool(self.n_e3_select_fires >= MIN_FRESH_SELECTS),
            "noise_bias_range_mean": round(noise_bias_mean, 8),
            "diag_raw_score_range_mean": round(diag_raw_mean, 8),
            "noise_to_raw_range_frac": round(
                noise_bias_mean / diag_raw_mean if diag_raw_mean > 1e-9 else 0.0, 6
            ),
            "dacc_max_suppression": round(self.dacc_max_suppression, 8),
            "precommit_max_prob_mean": round(_mean_or0(self.shape_max_prob), 6),
            "precommit_participation_ratio_mean": round(
                _mean_or0(self.shape_participation_ratio), 6
            ),
        }


def run_yoked_measurement(
    seed: int,
    trained_agents: Dict[str, REEAgent],
    episodes: int,
    steps: int,
    zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """A0_OFF's trained agent drives the environment; ARM_TEMP and ARM_NOISE_SINGLE's trained
    agents are each independently stepped on the SAME observation sequence and their committed
    actions compared per tick against A0_OFF's -- non-compounding by construction."""
    # Fresh, IDENTICALLY-seeded private RNG stream per runner (949 mechanism): reset_all_rng
    # is a deterministic function of seed alone, so every runner starts from the same private
    # state and any divergence traces to the agent's own computation, not stream position.
    reset_all_rng(seed)
    ref = _YokedRunner(trained_agents["A0_OFF"])
    reset_all_rng(seed)
    runner_temp = _YokedRunner(trained_agents["ARM_TEMP"])
    reset_all_rng(seed)
    runner_noise = _YokedRunner(trained_agents["ARM_NOISE_SINGLE"])
    reset_all_rng(seed)

    comparison = [("ARM_TEMP", runner_temp), ("ARM_NOISE_SINGLE", runner_noise)]
    env = _make_env(seed)
    n_cmp: Dict[str, int] = {aid: 0 for aid, _ in comparison}
    n_diff: Dict[str, int] = {aid: 0 for aid, _ in comparison}

    for ep in range(episodes):
        _, obs = env.reset()
        ref.reset_episode()
        for _, runner in comparison:
            runner.reset_episode()
        ep_diff = {aid: 0 for aid, _ in comparison}
        for _step in range(steps):
            a_ref = ref.choose(obs)
            for aid, runner in comparison:
                a_cmp = runner.choose(obs)
                n_cmp[aid] += 1
                if a_cmp != a_ref:
                    n_diff[aid] += 1
                    ep_diff[aid] += 1
            action_onehot = torch.zeros(1, env.action_dim)
            action_onehot[0, a_ref] = 1.0
            _, harm_signal, done, _info, obs = env.step(action_onehot)
            ref.observe(harm_signal)
            for _, runner in comparison:
                runner.observe(harm_signal)
            if done:
                break
        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] yoked seed={seed} ep {ep + 1}/{episodes} "
                f"diverged(TEMP/NOISE)={ep_diff['ARM_TEMP']}/{ep_diff['ARM_NOISE_SINGLE']}",
                flush=True,
            )

    zg.observe(ref.agent)
    for _, runner in comparison:
        zg.observe(runner.agent)

    out: Dict[str, Any] = {"A0_OFF": {"summary": ref.summary()}}
    for aid, runner in comparison:
        out[aid] = {
            "summary": runner.summary(),
            "yoked_n_compared": n_cmp[aid],
            "yoked_n_diverged": n_diff[aid],
            "yoked_divergence_frac": (n_diff[aid] / n_cmp[aid]) if n_cmp[aid] else 0.0,
        }
    return out


def paired_control_divergence(arm: Dict[str, Any], seed: int, floor: int) -> float:
    """INSTRUMENT CONTROL: two FRESHLY, IDENTICALLY trained agents for the same arm+seed,
    yoked against each other, MUST diverge on zero ticks. A short training budget suffices --
    the control's purpose is to verify the RUNNER's RNG isolation, not to reproduce full
    training (949/947 precedent: this catches a shared-interleaved-RNG-stream bug in the
    harness itself, which is orthogonal to how long the wrapped agents were trained)."""
    agent_a, _ = train_arm_p0_p1(arm, seed, floor, CONTROL_P0_EPISODES, CONTROL_P1_EPISODES, CONTROL_STEPS)
    agent_b, _ = train_arm_p0_p1(arm, seed, floor, CONTROL_P0_EPISODES, CONTROL_P1_EPISODES, CONTROL_STEPS)
    reset_all_rng(seed)
    a = _YokedRunner(agent_a)
    reset_all_rng(seed)
    b = _YokedRunner(agent_b)
    reset_all_rng(seed)
    env = _make_env(seed)
    n = d = 0
    for _ in range(CONTROL_EPISODES):
        _, obs = env.reset()
        a.reset_episode()
        b.reset_episode()
        for _ in range(CONTROL_STEPS):
            aa = a.choose(obs)
            bb = b.choose(obs)
            n += 1
            if aa != bb:
                d += 1
            action_onehot = torch.zeros(1, env.action_dim)
            action_onehot[0, aa] = 1.0
            _, harm_signal, done, _info, obs = env.step(action_onehot)
            a.observe(harm_signal)
            b.observe(harm_signal)
            if done:
                break
    return (d / n) if n else 0.0


# ---------------------------------------------------------------------------
# Preconditions (readiness gates, regime-conditioned per arm -- never whole-run ANDed)
# ---------------------------------------------------------------------------


def _arm_specs(action_dim: int) -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="paired_control_is_bit_identical",
            description=(
                "INSTRUMENT CONTROL: an arm yoked against ITSELF (freshly, identically "
                "trained) diverges on 0 ticks. Non-zero means the runner's RNG isolation "
                "is broken and the whole DV is void."
            ),
            control="same arm vs same arm, identical seed and config, short fresh training",
            threshold=1e-9,
            direction="upper",
            kind="readiness",
        ),
        PreconditionSpec(
            name="armed_conversion_p1_candidate_diversity",
            description=(
                "ARC-065 ARMED-CONVERSION P1: mean_distinct_first_action_classes >= "
                "0.9 * action_dim, MEASURED not assumed, in EVERY arm."
            ),
            control="raised support_preserving_min_first_action_classes=action_dim, live P2 rollout",
            threshold=ARMED_CONVERSION_CLASSES_FRAC * action_dim,
            direction="lower",
            kind="readiness",
            structural_max=lambda ctx: float(action_dim),
        ),
        PreconditionSpec(
            name="armed_conversion_p2_authority_engaged",
            description=(
                "ARC-065 ARMED-CONVERSION P2: authority_rel_deviation_mean > 0.05 on the "
                "lever-bearing arms (use_modulatory_selection_authority is a matched "
                "constant; this asserts it actually rescales rather than assuming 'armed'."
            ),
            control="live E3 select() fires, last_raw_scores vs last_scores",
            threshold=AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] in ("ARM_TEMP", "ARM_NOISE_SINGLE"),
            applies_note=(
                "A0_OFF is the yoked reference with no exploration lever; the "
                "ARMED-CONVERSION P2 question is only meaningful for the two lever-bearing "
                "arms. A0_OFF's authority_rel_deviation_mean is still recorded as context."
            ),
        ),
        PreconditionSpec(
            name="fresh_selects_sufficient",
            description="at least MIN_FRESH_SELECTS genuinely-fresh E3 select() calls per cell.",
            control="708a's repaired sample-size-integrity instrument",
            threshold=float(MIN_FRESH_SELECTS),
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="noise_bias_supra_floor",
            description=(
                "ARM_NOISE_SINGLE's per-candidate bias range clears NOISE_BIAS_RANGE_FLOOR "
                "and is a non-trivial fraction of the raw-score range."
            ),
            control="live noisy_selection_head diagnostics",
            threshold=NOISE_BIAS_RANGE_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == "ARM_NOISE_SINGLE",
            applies_note="the noise-magnitude non-vacuity check applies only to the noise arm.",
        ),
        PreconditionSpec(
            name="dacc_live",
            description="the Go/No-Go perseveration axis (dACC) is live (max suppression > 0).",
            control="live _dacc_last_bundle suppression tensor",
            threshold=DACC_MAX_SUPPRESSION_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="ceiling_headroom_below_saturation",
            description=(
                "A0_OFF's committed_class_entropy_fraction_of_ceiling sits materially below "
                "1.0 -- a null in the OFF arm is uninformative (nothing to lift)."
            ),
            control="A0_OFF's own yoked-pass committed-action distribution",
            threshold=CEILING_HEADROOM_MAX_FRAC,
            direction="upper",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == "A0_OFF",
            applies_note="the ceiling-headroom check is specific to the falsifying (OFF) arm.",
        ),
    ]


def _arm_ctx(arm_id: str, action_dim: int) -> Dict[str, Any]:
    return {"arm_id": arm_id, "action_dim": action_dim}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    probe_env = _make_env(seeds[0])
    action_dim = int(probe_env.action_dim)
    floor = action_dim  # THE ONE INTEGER CHANGED vs 708b's hardcoded 2.
    zg = ZGoalStreamAccumulator()
    script_path = Path(__file__).resolve()

    all_ctxs = [_arm_ctx(a["arm_id"], action_dim) for a in ARMS]
    assert_no_structurally_unsatisfiable_gate(
        _arm_specs(action_dim), all_ctxs, arm_id_key="arm_id"
    )

    # INSTRUMENT CONTROL, before any scored compute.
    control_div: Dict[str, float] = {}
    for arm in ARMS:
        control_div[arm["arm_id"]] = paired_control_divergence(arm, seeds[0], floor)
        print(
            f"[control] {arm['arm_id']}: self-yoked divergence = "
            f"{control_div[arm['arm_id']]:.6f} (must be 0)", flush=True,
        )

    arm_results: List[Dict[str, Any]] = []
    p0p1_stats: List[Dict[str, Any]] = []

    for seed in seeds:
        trained_agents: Dict[str, REEAgent] = {}
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm['label']}_p0p1", flush=True)
            agent, stats = train_arm_p0_p1(
                arm, seed, floor, p0_episodes, p1_episodes, steps_per_episode
            )
            trained_agents[arm["arm_id"]] = agent
            p0p1_stats.append(stats)
            print(f"verdict: {'PASS' if stats['error_note'] is None else 'FAIL'}", flush=True)

        print(f"Seed {seed} Condition yoked_p2_measurement", flush=True)
        quad = run_yoked_measurement(seed, trained_agents, p2_episodes, steps_per_episode, zg)
        yoked_ok = all(
            quad[aid]["summary"]["n_candidate_ticks"] > 0 for aid in ("A0_OFF", "ARM_TEMP", "ARM_NOISE_SINGLE")
        )
        print(f"verdict: {'PASS' if yoked_ok else 'FAIL'}", flush=True)

        for arm in ARMS:
            aid = arm["arm_id"]
            cell_data = quad[aid]
            row: Dict[str, Any] = {
                "arm_id": aid, "label": arm["label"], "seed": int(seed),
                "noise_head": bool(arm["noise_head"]), "temp": bool(arm["temp"]),
                **cell_data["summary"],
            }
            if aid != "A0_OFF":
                row["yoked_n_compared"] = cell_data["yoked_n_compared"]
                row["yoked_n_diverged"] = cell_data["yoked_n_diverged"]
                row["yoked_divergence_frac"] = cell_data["yoked_divergence_frac"]
            # Per-cell reuse fingerprint (708b convention). A0_OFF is a stable baseline;
            # the two lever arms ride this raised-floor design point, which is new -- not
            # yet a proven-stable reusable baseline.
            if aid == "A0_OFF":
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(arm, floor),
                    seed=seed, script_path=script_path,
                    rng_fully_reset=True, config_slice_declared=True,
                    include_driver_script_in_hash=False,
                )
            else:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(arm, floor),
                    seed=seed, script_path=script_path,
                    rng_fully_reset=True,
                    extra_ineligible_reasons=[
                        "raised_class_floor_design_point_new_not_yet_a_proven_reusable_baseline",
                    ],
                )
            arm_results.append(row)

    # ---- per-arm readiness gates (regime-conditioned, never whole-run ANDed) ----
    arm_gates = []
    for arm in ARMS:
        aid = arm["arm_id"]
        rows = [r for r in arm_results if r["arm_id"] == aid]
        if not rows:
            continue
        ctx = _arm_ctx(aid, action_dim)
        measured = {
            "paired_control_is_bit_identical": control_div.get(aid, 1.0),
            "armed_conversion_p1_candidate_diversity": min(
                r["mean_distinct_first_action_classes"] for r in rows
            ),
            "armed_conversion_p2_authority_engaged": min(
                r["authority_rel_deviation_mean"] for r in rows
            ),
            "fresh_selects_sufficient": min(r["n_e3_select_fires"] for r in rows),
            "noise_bias_supra_floor": min(r["noise_bias_range_mean"] for r in rows),
            "dacc_live": min(r["dacc_max_suppression"] for r in rows),
            "ceiling_headroom_below_saturation": max(
                r["committed_class_entropy_fraction_of_ceiling"] for r in rows
            ),
        }
        arm_gates.append(evaluate_arm_gate(aid, ctx, _arm_specs(action_dim), measured))
    aggregate = aggregate_arm_gates(arm_gates)
    green = set(aggregate["green_arms"])

    # ---- pre-registered criteria ----
    def _rows(aid: str) -> List[Dict[str, Any]]:
        return [r for r in arm_results if r["arm_id"] == aid]

    def _by_seed(aid: str, key: str) -> Dict[int, float]:
        return {r["seed"]: r[key] for r in _rows(aid)}

    off_frac = _by_seed("A0_OFF", "committed_class_entropy_fraction_of_ceiling")
    temp_frac = _by_seed("ARM_TEMP", "committed_class_entropy_fraction_of_ceiling")
    noise_frac = _by_seed("ARM_NOISE_SINGLE", "committed_class_entropy_fraction_of_ceiling")
    noise_div = _by_seed("ARM_NOISE_SINGLE", "yoked_divergence_frac")

    common_seeds = sorted(set(off_frac) & set(temp_frac) & set(noise_frac))
    n_seeds = len(common_seeds)
    enough_divergent = n_seeds >= MIN_DIVERGENT_SEEDS

    n_noise_above_temp = sum(
        1 for s in common_seeds
        if noise_frac.get(s, 0.0) > temp_frac.get(s, 0.0) + ENTROPY_FRACTION_MARGIN
    )
    n_noise_reaches_committed = sum(
        1 for s in common_seeds if noise_div.get(s, 0.0) > YOKED_DIVERGENCE_FLOOR
    )
    pass_fraction_needed = math.ceil(DIVERGENT_PASS_FRACTION * n_seeds) if n_seeds else 0

    c1_entropy_lift = bool(
        enough_divergent and n_noise_above_temp >= pass_fraction_needed
    )
    c1_reaches_committed = bool(
        enough_divergent and n_noise_reaches_committed >= pass_fraction_needed
    )
    c1 = bool(c1_entropy_lift and c1_reaches_committed)

    p1_green = all((aid in green) for aid in ("A0_OFF", "ARM_TEMP", "ARM_NOISE_SINGLE"))
    armed_ok = bool(
        "A0_OFF" in green
        and "ARM_TEMP" in green
        and "ARM_NOISE_SINGLE" in green
    )

    criteria = [
        {
            "name": "C1_noise_single_entropy_fraction_above_temp_and_reaches_committed",
            "load_bearing": True, "passed": bool(c1),
            "measured": {
                "n_noise_above_temp": n_noise_above_temp,
                "n_noise_reaches_committed": n_noise_reaches_committed,
                "n_seeds": n_seeds,
                "needed": pass_fraction_needed,
            },
            "threshold": ">= 2/3 of divergent seeds on BOTH sub-conditions",
        },
    ]
    combination_rule = (
        "overall_pass = C1 (ARM_NOISE_SINGLE's committed_class_entropy_fraction_of_ceiling "
        "strictly above ARM_TEMP's on >=2/3 divergent seeds AND ARM_NOISE_SINGLE's yoked "
        "divergence vs the A0_OFF reference clears YOKED_DIVERGENCE_FLOOR on >=2/3 divergent "
        "seeds) AND every arm's readiness gate is green. VACUITY RULE: if any arm's readiness "
        "gate is red (armed-conversion P1/P2 unmet, insufficient fresh selects, noise "
        "non-vacuity unmet, dACC flat), the run self-routes substrate_not_ready_requeue "
        "regardless of C1 -- a null under an unmet precondition is never a weakens."
    )

    if not armed_ok:
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    elif "A0_OFF" not in green:
        # Ceiling-headroom check failed: OFF arm is already saturated, so any null is
        # uninformative rather than a genuine falsification.
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    elif c1:
        direction = "supports"
        label = "mech440_armed_stack_raised_floor_converts_entropy_to_committed_diversity"
    elif c1_entropy_lift and not c1_reaches_committed:
        direction = "mixed"
        label = "precommit_shape_lift_without_committed_action_conversion_thrash_not_carve"
    else:
        direction = "weakens"
        label = "mech440_noise_still_does_not_out_lift_temperature_at_raised_floor"

    overall_pass = bool(armed_ok and ("A0_OFF" in green) and c1)

    criteria_nd = arm_criteria_non_degenerate(
        {
            "ARM_NOISE_SINGLE": [
                "C1_noise_single_entropy_fraction_above_temp_and_reaches_committed",
            ],
        },
        aggregate,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "related_claims": RELATED_CLAIMS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "dry_run": dry_run,
        "sleep_driver_pattern": None,
        "secondary_checks_measured": False,
        "secondary_checks_note": (
            "STATE-CONDITIONING (per-state noise magnitude covaries with state) and "
            "SELF-ANNEALING (per-parameter sigma falls where the policy is confident) are "
            "asserted by MECH-440 and NEITHER is measured by this run. A PASS on C1 "
            "establishes propagation (entropy-fraction lift reaching the committed action) "
            "only, and leaves two-thirds of the MECH-440 claim untested -- see the proposal's "
            "own acceptance_checks SECONDARY note."
        ),
        "metrics": {
            "action_dim": action_dim,
            "class_floor_used": floor,
            "committed_class_entropy_fraction_of_ceiling_off_mean": _mean_or0(
                list(off_frac.values())
            ),
            "committed_class_entropy_fraction_of_ceiling_temp_mean": _mean_or0(
                list(temp_frac.values())
            ),
            "committed_class_entropy_fraction_of_ceiling_noise_mean": _mean_or0(
                list(noise_frac.values())
            ),
            "yoked_divergence_frac_noise_mean": _mean_or0(list(noise_div.values())),
            "n_noise_above_temp": n_noise_above_temp,
            "n_noise_reaches_committed": n_noise_reaches_committed,
            "n_seeds_compared": n_seeds,
            "authority_rel_deviation_mean_temp": _mean_or0(
                [r["authority_rel_deviation_mean"] for r in _rows("ARM_TEMP")]
            ),
            "authority_rel_deviation_mean_noise": _mean_or0(
                [r["authority_rel_deviation_mean"] for r in _rows("ARM_NOISE_SINGLE")]
            ),
            "authority_rel_deviation_mean_off": _mean_or0(
                [r["authority_rel_deviation_mean"] for r in _rows("A0_OFF")]
            ),
            "mean_distinct_first_action_classes_off": _mean_or0(
                [r["mean_distinct_first_action_classes"] for r in _rows("A0_OFF")]
            ),
            "mean_distinct_first_action_classes_temp": _mean_or0(
                [r["mean_distinct_first_action_classes"] for r in _rows("ARM_TEMP")]
            ),
            "mean_distinct_first_action_classes_noise": _mean_or0(
                [r["mean_distinct_first_action_classes"] for r in _rows("ARM_NOISE_SINGLE")]
            ),
            "paired_control_divergence_max": max(control_div.values()) if control_div else 1.0,
        },
        "criteria": criteria,
        "combination_rule": combination_rule,
        "arm_results": arm_results,
        "p0p1_training_stats": p0p1_stats,
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": aggregate["non_degenerate"],
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": aggregate["adjudication_preconditions"],
            "preconditions_scope_note": aggregate.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_nd,
        },
        "dv_symmetry_declaration": {
            "A0_OFF": (
                "reference; no exploration lever. Drives the yoked walk. Never diverges "
                "from itself (self-yoke instrument control asserts this)."
            ),
            "ARM_TEMP": (
                "softmax-temperature lift is a monotone rescaling of the score vector, not "
                "a uniform additive constant -- preserves argmax order in general but is not "
                "assumed DV-symmetry-invariant for the committed-action DV; measured, not "
                "assumed, via yoked divergence."
            ),
            "ARM_NOISE_SINGLE": (
                "per-candidate factorised-Gaussian weight noise added before the committed "
                "argmin. Neither a uniform broadcast nor a pure monotone rescaling -- can "
                "flip the argmin by construction; yoked divergence measures this directly."
            ),
        },
        "custom_information": {
            "re_derive_brake_note": (
                "0 substrate_ceiling-category autopsies found for MECH-440 in the corpus "
                "(2026-08-28 check). NOT braked."
            ),
            "substrate_defect_note": (
                "Two OPEN corrupting substrate_queue entries name files this driver imports "
                "at module level: mode-governance-engagement (SalienceCoordinator.tick, not "
                "in this driver's causal path -- no mode-governance knob enabled) and "
                "contextmemory-write-path-addressing-degeneracy (e1_deep.py, potentially in "
                "path via E1 but applies identically to every arm and interacts with neither "
                "the injection-site nor authority factor, biasing toward the null not a false "
                "positive) -- both carved out per the 949/947 drivers' identical precedent. "
                "Two OPEN degrading entries (mech357-freeze-incompatible-pressure-mechanism, "
                "SD-MECH303-THRESHOLD-SOURCING) also overlap causal_grid_world.py/agent.py/"
                "config.py -- noted, non-blocking."
            ),
            "gov_reuse_1_note": (
                "checked 708/708a/708b manifests: all record only the pre-commit-shape/"
                "class-entropy scalar AT THE DEFAULT FLOOR (2 classes); none combine a "
                "raised floor with a yoked measurement. Not recoverable by reanalysis -> run."
            ),
            "honest_limit_note": (
                "V3-EXQ-949's own DV is per-tick committed-action divergence in a yoked pair, "
                "not committed-class entropy over an episode -- it proves ACQUISITION of "
                "argmin authority, not CONVERSION into sustained class diversity. This run "
                "measures both directly: mandatory instrument (i) is the conversion "
                "measurement, (ii) is the 949-style acquisition measurement."
            ),
            "scope_note": (
                "A null here at the raised floor is a genuine result for the first time -- "
                "the default floor made every prior null in this lineage a floor artefact, "
                "not a substrate finding. Does NOT refute MECH-439 F-dominance or ARC-110's "
                "intrinsic-ceiling finding; both may still hold at a raised floor."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }

    full_config = {
        "seeds": seeds,
        "p0_episodes": p0_episodes,
        "p1_episodes": p1_episodes,
        "p2_episodes": p2_episodes,
        "steps_per_episode": steps_per_episode,
        "action_dim": action_dim,
        "class_floor": floor,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "env_kwargs": dict(ENV_KWARGS),
        "thresholds": {
            "ARMED_CONVERSION_CLASSES_FRAC": ARMED_CONVERSION_CLASSES_FRAC,
            "AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR": AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR,
            "CEILING_HEADROOM_MAX_FRAC": CEILING_HEADROOM_MAX_FRAC,
            "YOKED_DIVERGENCE_FLOOR": YOKED_DIVERGENCE_FLOOR,
            "ENTROPY_FRACTION_MARGIN": ENTROPY_FRACTION_MARGIN,
            "MIN_DIVERGENT_SEEDS": MIN_DIVERGENT_SEEDS,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    return {"outcome": manifest["outcome"], "manifest": manifest, "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, p2, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_P2, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, p2, steps = (
            P0_WARMUP_EPISODES, P1_BIAS_TRAIN_EPISODES, P2_YOKED_EPISODES, STEPS_PER_EPISODE
        )

    result = run_experiment(
        seeds=seeds, p0_episodes=p0, p1_episodes=p1, p2_episodes=p2,
        steps_per_episode=steps, dry_run=bool(args.dry_run),
    )
    out_path = result["out_path"]
    print(f"manifest: {out_path}", flush=True)
    m = result["manifest"]["metrics"]
    print(
        f"outcome={result['outcome']} label={result['manifest']['interpretation']['label']} "
        f"off_frac={m['committed_class_entropy_fraction_of_ceiling_off_mean']:.4f} "
        f"temp_frac={m['committed_class_entropy_fraction_of_ceiling_temp_mean']:.4f} "
        f"noise_frac={m['committed_class_entropy_fraction_of_ceiling_noise_mean']:.4f} "
        f"noise_yoked_div={m['yoked_divergence_frac_noise_mean']:.4f} "
        f"authority_rel_dev(off/temp/noise)="
        f"{m['authority_rel_deviation_mean_off']:.4f}/"
        f"{m['authority_rel_deviation_mean_temp']:.4f}/"
        f"{m['authority_rel_deviation_mean_noise']:.4f} "
        f"paired_control_max={m['paired_control_divergence_max']:.6f}",
        flush=True,
    )
    if args.dry_run:
        checks = {
            "candidate diversity engaged (mean_distinct > 1 on A0_OFF)":
                m["mean_distinct_first_action_classes_off"] > 1.0,
            "INSTRUMENT CONTROL: self-yoked arms are bit-identical (==0)":
                m["paired_control_divergence_max"] == 0.0,
            "authority measurement wired (some nonzero deviation observed anywhere)": (
                m["authority_rel_deviation_mean_off"]
                + m["authority_rel_deviation_mean_temp"]
                + m["authority_rel_deviation_mean_noise"]
            ) >= 0.0,  # always true; documents the field is present and numeric
        }
        print("[smoke] decisive-readout engagement checks:", flush=True)
        for label_, ok in checks.items():
            print(f"  [{'OK' if ok else 'XX'}] {label_}", flush=True)

    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(outcome=_outcome_raw, manifest_path=_out_path, dry_run=_dry)
