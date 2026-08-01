#!/opt/local/bin/python3
"""
V3-EXQ-858 -- ARC-062 conversion-fanout GOV-FANOUT-1 Leg P-B (H2: F-dominance,
downstream of selection). F-weight attenuation ladder, committed-class entropy.

DESIGN OF RECORD: REE_assembly/evidence/planning/arc_062_conversion_fanout_2026-07-29.md
Section 2 "Leg P-B", the "P-B buildability resolution (2026-07-31)" section, and the
"Erratum (2026-07-31)" section (the corrected `lateral_pfc` route source this leg's
matched stack uses). Structural template: v3_exq_851 (Leg P-A -- the matched-stack
base: SP-CEM, GAP-A e2_world_forward candidate summaries, MECH-448 rank-preserving
F->eligibility demotion, MECH-449 active Go/No-Go opponency, the CRF/LateralPFCAnalog
rule-apprehension stack with `lateral_pfc`-routed channel, C1(a-g) readiness gates,
latch-clearing pattern for E3 diagnostics reads, arm_fingerprint/stamp_recording_core/
emit_outcome/z_goal-stream wiring). This leg reuses ALL of that structural machinery
UNCHANGED except:
  1. The swept variable is `agent.e3.config.f_weight` (SD-085's new no-op-default
     coefficient on F's contribution to E3TrajectorySelector.score_trajectory), across
     a 4-rung attenuation ladder {1.0, 0.5, 0.25, 0.0}, REPLACING 851's two-arm
     `use_candidate_rule_field` ON/OFF sweep. `use_candidate_rule_field=True` is now a
     MATCHED CONSTANT on every rung (851's ARM_ON config), per this leg's own design
     spec: "hold the rule-apprehension channel ON and lateral_pfc-routed... as a
     matched constant."
  2. C1(a) through C1(g) are reused VERBATIM (same names, same thresholds/floors) but
     re-scoped from "both arms" (851's ON/OFF split) to "all four rungs" (since every
     rung now carries the SAME CRF-ON/lateral_pfc-routed config that was previously
     ARM_ON only) -- C1_holds requires each sub-gate to hold at majority-of-seeds
     (>=2/3) on EVERY rung, not just a majority of rungs. C1(d) (propagation
     non-vacuity) is necessarily re-expressed as a DIRECT magnitude floor check
     (mean_lateral_pfc_bias_abs > PROP_NONVAC_FLOOR) rather than a paired ON-vs-OFF
     diff, because there is no OFF arm in this design -- CRF is ON everywhere. This is
     the substrate-readiness-independent-of-f_weight machinery this leg's own spec
     names, so applying it identically to every rung is the correct generalisation of
     851's "both arms" convention, not a design change.
  3. NEW C1(h) -- F_WEIGHT_KNOB_LIVE: confirms `f_weight` genuinely rescales F through
     the FULL matched-stack pipeline (CRF-ON, lateral_pfc-routed, MECH-448/449-active)
     at EVERY rung, not just at substrate defaults (V3-EXQ-852 tested defaults only).
     See the F_WEIGHT_KNOB_LIVE section below.
  4. PRIMARY DV unchanged from 851/654-lineage: paired-by-seed COMMITTED-CLASS
     entropy. C2 (load-bearing) is the F000-vs-F100 lift (loosening F fully vs the
     f_weight=1.0 baseline rung, which reproduces the current substrate default).
  5. `claim_ids=[]`, `experiment_purpose="diagnostic"` (brake-exempt per the fanout
     doc Section 4 -- this leg tests a different mechanism (F-dominance) from Leg
     P-A's coupling-gap question, and is explicitly claim-neutral by design).
  6. The 851 within-class-representative-entropy SECONDARY negative control is
     DROPPED (deviation from 851, noted here for the record): that control's framing
     ("the rule bias is class-keyed, so it cannot move within-class selection")
     is specific to sweeping `use_candidate_rule_field`, which is now a matched
     constant here, not the swept variable. Nothing this leg's own design spec asks
     for depends on it. The C1(a-h) readiness machinery, the primary C2 DV, and the
     always-core recording are all unaffected.

SD-085 (REE_assembly/docs/architecture/sd_085_e3_reality_cost_weight.md, ree-v3 main,
landed 2026-07-31) added `E3Config.f_weight: float = 1.0` as a multiplicative
coefficient on F in `score_trajectory` (ree_core/predictors/e3_selector.py:1190):
`score = self.config.f_weight * f + lambda_eff * m + rho_residue * phi [+ optional
terms]`. Not wired through `REEConfig.from_dims()` (matches the `lambda_ethical`/
`rho_residue` sibling-field convention this coefficient is modelled on) -- set
directly post-construction: `agent.e3.config.f_weight = X`. When
`agent.e3.e3_score_decomp_enabled = True`, `agent.e3._last_traj_components` gains
keys "f" (raw, unweighted `compute_reality_cost` output) and "f_weighted" (=
`f_weight * f`), populated inside the SAME `score_trajectory` call that
`agent.e3.select()` makes -- so both are fresh exactly when
`agent.e3.last_score_diagnostics` transitions from the pre-clear `None` to non-None
this tick (the SAME latch-clearing discipline this script already needs for C1g's
route-range read covers both reads at zero extra cost).

CORRECTED ROUTE SOURCE (P-A erratum, 2026-07-31): the design doc's Leg P-B text
inherits Leg P-A's original (WRONG) `modulatory_channel_route_source="gated_policy"`
premise by cross-reference ("the P-A ON-arm config"). Per the erratum
(design doc, "Erratum (2026-07-31)" section) and V3-EXQ-851's own build,
`"gated_policy"` does NOT route the rule-apprehension/CRF channel -- it routes the
unrelated ARC-062 Phase-1 GatedPolicy module's output. The CORRECT route source for
the CRF/`use_candidate_rule_field` channel is `"lateral_pfc"` (the new branch V3-EXQ-851
added to `REEAgent.select_action`'s elif chain, identity-routing `_bdc_lpfc`). This
script's matched-stack config uses `modulatory_channel_route_source="lateral_pfc"`,
matching V3-EXQ-851's corrected value, NOT the design doc's original literal
`"gated_policy"` text.

C1(h) F_WEIGHT_KNOB_LIVE (new readiness gate; any fail on any rung ->
substrate_not_ready_requeue, NEVER a verdict). At each rung/seed, on genuine
fresh-select P2 ticks (latch-cleared), collect `agent.e3._last_traj_components["f"]`
(raw) and `["f_weighted"]`. Per seed/rung: mean_f_raw / mean_f_weighted over those
ticks; expected = rung_f_weight * mean_f_raw; tolerance = F_WEIGHT_SCALING_TOL *
max(1e-6, abs(expected)) (2%); met = |mean_f_weighted - expected| <= tolerance --
a TWO-SIDED closeness check, expressed at the per-seed/rung row level as an INTERVAL
(threshold_low = expected - tolerance, threshold_high = expected + tolerance,
direction="interval") per the skill's two-sided-precondition convention. The
AGGREGATE gating entry in interpretation.preconditions[] follows the SAME
COUNT-shaped convention as C1(a/b/e/f/g) (measured = min count of seeds-meeting
across rungs, threshold = MIN_SEEDS_FOR_PASS, comparator ">=", direction "lower") --
this is what the indexer's `_precondition_unmet` recompute can verify exactly (a
count over a per-row interval-decided boolean is exactly reproducible; a single
interval bound spanning four different rung_f_weight values is not), while the
underlying interval math that DECIDES `met` for each row is recorded per-row in
`arm_results[i]["f_weight_knob_live"]` for full auditability. This confirms the
f_weight knob is genuinely taking effect through the FULL matched-stack pipeline at
EVERY rung (V3-EXQ-852 only checked tick-0 default-vs-one-attenuated value under
substrate DEFAULTS; this reconfirms across the whole ladder under the CRF-ON +
lateral_pfc-routed + MECH-448/449-active matched-stack config, a materially different
substrate configuration from 852's).
`C1_holds = c1a and c1b and c1c and c1d and c1e and c1f and c1g and c1h` (AND across
all eight sub-gates, each itself "holds on all 4 rungs at majority-of-seeds-per-rung").

PRIMARY LOAD-BEARING CRITERION (C2): paired-by-seed committed-class entropy at
ARM_F000 (f_weight=0.0) minus ARM_F100 (f_weight=1.0, the current substrate default
baseline rung) >= C2_LIFT_MARGIN_NATS (0.05, reused from 851) on >= C2_MIN_LIFT_SEEDS
(2 of 3, reused from 851). SECONDARY/DIAGNOSTIC (non-gating, always reported):
committed-class entropy at EVERY rung (F100/F050/F025/F000), per seed AND the
cross-seed mean per rung, so the ladder shape (monotonic vs threshold-like vs flat)
is visible regardless of C2's pass/fail.

DV-SYMMETRY INVARIANCE DECLARATION (MANDATORY, per the 604c net). Committed-class
entropy is a function of the argmax action-class sequence. `f_weight` rescales ONE
per-candidate-VARYING term (F, the reality cost -- NOT a broadcast scalar; F differs
across candidates at every tick) relative to the other additive terms (M, Phi) in the
score sum. This is neither a uniform additive shift across candidates (which argmax
would cancel) nor a monotone rescaling of the WHOLE score (only one component's
relative weight changes, so relative candidate ordering under the OTHER terms is not
preserved) -- so it is NOT invariant under the argmax symmetry of the committed-class
DV, and legitimately CAN change which candidate wins the committed argmin. This holds
identically at every rung (the manipulation is the SAME kind of rescaling at every
attenuation level, only the coefficient differs) -- stated once here rather than
per-rung in the manifest, since it does not vary by rung.

DECLARED NULL / INTERPRETATION GRID (three pre-registered branches; NO weakens --
claim_ids=[] means nothing is weakened by any branch):
  1. Any C1(a-h) fails on any rung -> label "substrate_not_ready_requeue".
     evidence_direction = "unknown". Never a verdict.
  2. C1 holds on all 4 rungs AND C2 holds (F000-vs-F100 lift clears the margin on
     >=2 seeds) -> label "f_dominance_confirmed_h2_operative_ceiling". H2
     (F-dominance) is CONFIRMED as the operative ceiling -- the fix is an
     F-rebalance substrate (an f_weight-like lever made a genuine, non-eligibility-
     face attenuation lever), and ARC-062's behavioural retest should be gated
     behind that substrate landing. evidence_direction = "non_contributory"
     (claim_ids=[] diagnostic; neither branch weights any claim) but this is
     nonetheless a load-bearing GOVERNANCE ROUTING signal, flagged prominently in
     `interpretation.note` even though claim-neutral.
  3. C1 holds on all 4 rungs AND C2 fails -> label
     "f_dominance_refuted_as_sole_cause". H2-as-sole-cause is REFUTED as the
     ceiling. Per the fanout doc's own P-C resolution section, H3
     (competence/action-learning) is ALREADY resolved --
     hypothesis_space_registry.v1.json's H-policy-learning state is "eliminated",
     and the surviving live root is H-observation-interface (representation/
     observation-encoding, MECH-457-tagged, owned by the standing competence_floor
     re-posing thread, registry delta D12). So this branch does NOT reopen H3 as a
     live discriminator target -- it corroborates that the conversion ceiling, if
     not F-dominance, sits at the already-tracked H-observation-interface root, not
     at a still-open competence question. evidence_direction = "non_contributory".

GOV-REUSE-1 (Step 2.4): the decisive readout (paired-by-seed committed-class entropy
across an f_weight attenuation ladder, under the lateral_pfc-routed + CRF-ON +
MECH-448/449-active matched-stack config) does not exist in any recorded manifest.
`f_weight` did not exist before V3-EXQ-852 (SD-085 landed 2026-07-31), and no prior
run (852 included -- 852 used substrate DEFAULTS, not the CRF-ON matched stack) swept
it under this configuration. Not recoverable by reanalysis -> proceed to author.

RE-DERIVE BRAKE (Step 2.5b): `claim_ids=[]`, brake-EXEMPT per the fanout doc Section 4
("P-B/P-C/P-D are brake-exempt... P-B tests a different mechanism").

See REE_assembly/evidence/planning/arc_062_conversion_fanout_2026-07-29.md (Section 2
Leg P-B, the P-B buildability resolution, and the Erratum sections),
REE_assembly/docs/architecture/sd_085_e3_reality_cost_weight.md,
experiments/v3_exq_851_arc062_pa_lateral_pfc_route_source_gapfanout.py (the
matched-stack + C1(a-g) + latch-clearing template this merges from),
experiments/v3_exq_852_sd085_f_weight_substrate_readiness.py (the f_weight call
idiom + F_WEIGHT decomp readout worked example),
ree_core/predictors/e3_selector.py (score_trajectory, f_weight, _last_traj_components).
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_858_arc062_pb_fweight_attenuation_ladder_committed_class_entropy"
QUEUE_ID = "V3-EXQ-858"
SUPERSEDES = None
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

# All C1(a-h) readiness preconditions below are COUNT-shaped aggregates -- "does a
# majority of seeds meet a live per-seed MEASURED criterion, on every rung" -- not
# known-degenerate-REFERENCE anchors in the sense validate_experiments.py's
# anchor_reachability_lint exists for (a frozen fixture replayed through a
# hand-written predicate that might be narrower than the degeneracy it targets). Each
# `met` value here is computed DIRECTLY from the same live run's measured statistic
# against its threshold, at run time -- there is no separate "shipped predicate vs
# replayed reference" distinction to guard (assert_anchor_reachable's rules 1/2 do not
# apply: there is no frozen reference_cells fixture, and no risk of a predicate/
# reference mismatch, since the predicate and the reachability question are the same
# computation). This exactly mirrors V3-EXQ-851's identically-SHAPED C1(a-g)
# preconditions (same "majority of seeds in a group clears a live measured floor"
# aggregate pattern, same `control` provenance-documentation usage), which carry no
# such warning under experiment_purpose="evidence" -- the precondition SHAPE this leg
# reuses verbatim from 851 is unchanged; only this leg's required
# experiment_purpose="diagnostic" (per the design of record) newly subjects the
# identical shape to a lint whose true concern (a replayed known-degenerate reference
# scored by a narrower hand-written predicate) does not apply here. The `control`
# fields document PROVENANCE (which channel/statistic is measured), matching
# validate_experiments.py's own documented over-fire case ("a `control` key documents
# provenance on a precondition that anchors nothing reproducible"), not a frozen
# reference cell to replay.
ANCHOR_REACHABILITY_EXEMPT = (
    "COUNT-shaped readiness aggregates over live per-seed/per-rung measured "
    "statistics (majority-of-seeds-per-rung, all rungs) -- not known-degenerate-"
    "reference anchors; met is computed directly from the live measured value at run "
    "time, so there is no predicate-vs-reference mismatch to guard against. Mirrors "
    "V3-EXQ-851's identically-shaped C1(a-g) preconditions, which are unguarded "
    "because that script's experiment_purpose='evidence' does not trigger this lint "
    "-- the shape is unchanged; only this leg's required "
    "experiment_purpose='diagnostic' newly exposes it to a lint whose concern (a "
    "replayed reference scored by a narrower hand-written predicate) is orthogonal "
    "to a live-measurement count aggregate."
)

_ZG = ZGoalStreamAccumulator()

# CRF-gate calibration amend levers (matched-stack constant; identical to 851/654j).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# C2 (PRIMARY): paired-by-seed committed-class entropy lift of ARM_F000 over ARM_F100.
C2_LIFT_MARGIN_NATS = 0.05
C2_MIN_LIFT_SEEDS = 2  # of 3

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# C1(b) readiness: GAP-A consumed-summary divergence.
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# C1(c) readiness: CRF minted distinct rules AND fired a non-zero differentiated
# rule_state on a meaningful fraction of MATURED P2 ticks. CRF is ON on every rung.
CRF_MIN_MINTED = 2
CRF_N_ACTIVE_FLOOR = 1
CRF_FRAC_ACTIVE_FLOOR = 0.30
CRF_DIST_FLOOR = 1e-3

# C1(d) PROPAGATION non-vacuity: re-expressed as a DIRECT magnitude floor (no OFF arm
# in this design -- CRF is a matched constant ON everywhere).
PROP_NONVAC_FLOOR = 1e-3

MIN_SEEDS_FOR_PASS = 2  # of 3

# C1(e) MECH-448 DEMOTION non-vacuity (matched conversion constant; identical to 851).
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8
EXCLUDED_COUNT_FLOOR = 0.0

# C1(f) MECH-449 ACTIVE NO-GO non-vacuity (matched opponency constant; identical to 851).
NOGO_ACTIVE_FRAC_FLOOR = 0.8
NOGO_SUPPRESSED_FLOOR = 0.0

MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 200
P1_BIAS_TRAIN_EPISODES = 90
P2_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# Matched-stack lever constants (IDENTICAL on EVERY rung -- the P-A ON-arm config,
# held constant per this leg's own design spec).
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
# V3-EXQ-851's CORRECTED route source (see module docstring "CORRECTED ROUTE SOURCE"
# section) -- NOT the design doc's original literal "gated_policy" text.
MODULATORY_CHANNEL_ROUTE_SOURCE = "lateral_pfc"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

# C1(g) ROUTING READINESS (reused verbatim from 851; matched constant on every rung).
ROUTE_RANGE_SEED_FLOOR = 0.01
BETA_RATE_MAX_STEPS = 20  # ree_core/heartbeat/clock.py MECH-093 slowest E3-reselection cadence
FRESH_SELECT_FLOOR = (P2_MEASUREMENT_EPISODES * STEPS_PER_EPISODE) // BETA_RATE_MAX_STEPS  # 600
FRESH_SELECT_YIELD_FLOOR = 1.0 / BETA_RATE_MAX_STEPS  # 0.05

# C1(h) NEW -- F_WEIGHT_KNOB_LIVE tolerance (2%, per this leg's design spec).
F_WEIGHT_SCALING_TOL = 0.02

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

# The candidate-rule-field channel is a MATCHED CONSTANT here (P-A's ON-arm value),
# not the swept variable -- see module docstring point 1.
USE_CANDIDATE_RULE_FIELD = True

# SD-056 online e2 training (mirror V3-EXQ-649 / 851 harness).
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

# P1 bias-head REINFORCE training (mirror V3-EXQ-598b / 851).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# IDENTICAL env to V3-EXQ-654 / 851 (SD-054 reef + hazard_food_attraction + bipartite
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


# The 4-rung f_weight attenuation ladder -- the ONLY swept variable.
RUNGS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_F100", "label": "f_weight_1p00_baseline_full_strength", "f_weight": 1.0},
    {"arm_id": "ARM_F050", "label": "f_weight_0p50_half_attenuated", "f_weight": 0.5},
    {"arm_id": "ARM_F025", "label": "f_weight_0p25_quarter_attenuated", "f_weight": 0.25},
    {"arm_id": "ARM_F000", "label": "f_weight_0p00_fully_removed", "f_weight": 0.0},
]
RUNG_IDS: List[str] = [r["arm_id"] for r in RUNGS]
RUNG_F_WEIGHT: Dict[str, float] = {r["arm_id"]: float(r["f_weight"]) for r in RUNGS}


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Matched-stack agent; IDENTICAL config on every rung (851's ARM_ON config).

    f_weight is NOT a REEConfig/from_dims field (SD-085: set directly post-
    construction, matching the lambda_ethical/rho_residue sweep idiom) -- the caller
    sets ``agent.e3.config.f_weight`` and ``agent.e3.e3_score_decomp_enabled = True``
    after this returns.
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
        # --- Matched stack (identical on EVERY rung) ---
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
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
        # --- Matched constant (NOT swept here -- see module docstring point 1) ---
        use_candidate_rule_field=USE_CANDIDATE_RULE_FIELD,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-649 / 851)
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


# ---------------------------------------------------------------------------
# Per-tick measurement helpers
# ---------------------------------------------------------------------------


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    """Per-candidate cand_world_summaries the bias channels consume (GAP-A
    e2.world_forward source; matched on every rung)."""
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


def _mean_pairwise_l2(summ: torch.Tensor) -> float:
    summ = summ.detach()
    k = summ.shape[0]
    if k < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += float(torch.linalg.vector_norm(summ[i] - summ[j]))
            n += 1
    return total / max(n, 1)


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


# ---------------------------------------------------------------------------
# P1 bias-head REINFORCE training (mirror V3-EXQ-598b / 851)
# ---------------------------------------------------------------------------


def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    """REINFORCE on the SD-033a bias head over stored (candidate_features, sel, return).

    Re-runs compute_bias (differentiable w.r.t. rule_bias_head weights) with the
    CURRENT rule_state on stored candidate summaries, REINFORCE-weighted by the
    episode-return advantage. Mirrors v3_exq_598b/851's _lpfc_reinforce_loss.
    """
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


def _propagation_counterfactual_delta(
    agent: REEAgent, summaries: torch.Tensor
) -> Optional[float]:
    """Within-cell isolation: mean |bias(field rule_state) - bias(zeroed rule_state)|.

    Supporting diagnostic alongside the load-bearing C1(d) magnitude precondition.
    Best-effort.
    """
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None or summaries is None:
        return None
    try:
        with torch.no_grad():
            bias_field = lpfc.compute_bias(summaries).detach().clone()
            saved = lpfc.rule_state.detach().clone()
            lpfc.rule_state.zero_()
            bias_zero = lpfc.compute_bias(summaries).detach().clone()
            lpfc.rule_state.copy_(saved)
        return float((bias_field - bias_zero).abs().mean().item())
    except Exception:
        return None


def _run_seed_rung(
    rung: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)
    f_weight = float(rung["f_weight"])
    env = _make_env(seed)
    agent = _make_agent(env)
    agent.e3.config.f_weight = f_weight  # SD-085 lever: set directly (matches the
                                          # lambda_ethical/rho_residue sweep idiom)
    agent.e3.e3_score_decomp_enabled = True  # populates _last_traj_components (C1h)

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(
        list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
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

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # PRIMARY DV + readiness accumulators (P2).
    committed_class_counts: Dict[int, int] = {}
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # C1(g) routing readiness (latch-aware; see the select_action call).
    n_p2_fresh_select = 0
    n_p2_latched_ticks = 0
    route_ranges: List[float] = []
    route_range_max = 0.0
    route_active_ticks = 0

    # C1(h) NEW -- F_WEIGHT_KNOB_LIVE readouts (latch-aware; same fresh-select ticks
    # as C1g, since _last_traj_components is populated in the SAME score_trajectory
    # call inside select() that populates last_score_diagnostics).
    f_raw_samples: List[float] = []
    f_weighted_samples: List[float] = []

    # CRF differentiation + bias diagnostics (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_matched_per_tick: List[int] = []
    crf_max_pairwise_rule_dist_max = 0.0
    crf_n_minted_total_last = 0
    lateral_pfc_bias_abs_vals: List[float] = []
    prop_counterfactual_deltas: List[float] = []

    # MECH-448 f_eligibility-demotion non-vacuity readouts (matched conversion
    # constant; read LIVE from e3.last_score_diagnostics at the P2 select tick).
    demotion_active_ticks = 0
    demotion_envelope_sizes: List[float] = []
    demotion_excluded_counts: List[float] = []

    # MECH-449 Go/No-Go-constitution non-vacuity readouts (matched active-No-Go
    # opponency constant).
    nogo_active_ticks = 0
    nogo_suppressed_per_tick: List[int] = []
    nogo_envelope_sizes: List[float] = []

    for ep in range(total_train_eps):
        is_p1 = (p1_start <= ep < p2_start)
        is_p2 = (ep >= p2_start)
        phase_label = "P2" if is_p2 else ("P1" if is_p1 else "P0")

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
            if is_p2 and candidates:
                pre_e3_classes = sorted({
                    _traj_first_action_class(t) for t in candidates
                })

            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            # Latch-clearing (791a/851 pattern; the ~9x pseudo-replication defect).
            # Both the C1g route-range read AND the NEW C1h f/f_weighted read are
            # latch-aware off the SAME fresh-select check -- _last_traj_components is
            # populated inside the SAME score_trajectory call select() makes.
            _prev_diag_snapshot = agent.e3.last_score_diagnostics
            agent.e3.last_score_diagnostics = None
            action = agent.select_action(candidates, ticks)
            _fresh_diag = agent.e3.last_score_diagnostics
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at rung={rung['arm_id']} seed={seed} "
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

            if is_p2:
                n_p2_ticks += 1
                committed_class_counts[committed_class] = (
                    committed_class_counts.get(committed_class, 0) + 1
                )
                if len(pre_e3_classes) >= 2:
                    n_p2_pre_ge2 += 1

                if candidates and len(candidates) >= 2:
                    consumed = _consumed_summaries(agent, candidates)
                    if consumed is not None and torch.isfinite(consumed).all():
                        d = _mean_pairwise_l2(consumed)
                        if math.isfinite(d):
                            consumed_dists.append(d)
                            consumed_dist_max = max(consumed_dist_max, d)

                lpfc = getattr(agent, "lateral_pfc", None)
                if lpfc is not None:
                    lb_mean = getattr(lpfc, "_last_bias_abs_mean", None)
                    if isinstance(lb_mean, (int, float)):
                        lateral_pfc_bias_abs_vals.append(float(lb_mean))

                diag = _fresh_diag if _fresh_diag is not None else (_prev_diag_snapshot or {})

                if _fresh_diag is None:
                    n_p2_latched_ticks += 1
                else:
                    n_p2_fresh_select += 1
                    rr = _fresh_diag.get("modulatory_channel_route_range")
                    if rr is not None and math.isfinite(float(rr)):
                        route_ranges.append(float(rr))
                        route_range_max = max(route_range_max, float(rr))
                    if bool(_fresh_diag.get("modulatory_channel_route_active", False)):
                        route_active_ticks += 1

                    # C1(h) NEW -- read the SAME-tick score decomposition.
                    decomp = getattr(agent.e3, "_last_traj_components", None)
                    if isinstance(decomp, dict):
                        f_raw = decomp.get("f")
                        f_wt = decomp.get("f_weighted")
                        if (
                            f_raw is not None and f_wt is not None
                            and math.isfinite(float(f_raw))
                            and math.isfinite(float(f_wt))
                        ):
                            f_raw_samples.append(float(f_raw))
                            f_weighted_samples.append(float(f_wt))

                if bool(diag.get("f_eligibility_demotion_active", False)):
                    demotion_active_ticks += 1
                    env_size = float(diag.get("f_eligibility_envelope_size", -1))
                    if math.isfinite(env_size) and env_size >= 0:
                        demotion_envelope_sizes.append(env_size)
                    excl = float(diag.get("f_eligibility_excluded_count", -1))
                    if math.isfinite(excl) and excl >= 0:
                        demotion_excluded_counts.append(excl)

                if bool(diag.get("go_nogo_constitution_active", False)):
                    nogo_active_ticks += 1
                    n_safety = int(diag.get("go_nogo_n_safety_nogo", 0) or 0)
                    n_soft = int(diag.get("go_nogo_n_soft_applied", 0) or 0)
                    nogo_suppressed_per_tick.append(n_safety + n_soft)
                    gng_env = float(diag.get("go_nogo_envelope_size", -1))
                    if math.isfinite(gng_env) and gng_env >= 0:
                        nogo_envelope_sizes.append(gng_env)

                crf = getattr(agent, "candidate_rule_field", None)
                if crf is not None:
                    st = crf.get_state()
                    n_active = int(st.get("crf_n_active_last", 0))
                    crf_n_active_per_tick.append(n_active)
                    crf_n_matched_per_tick.append(int(st.get("crf_n_matched_last", 0)))
                    crf_max_pairwise_rule_dist_max = max(
                        crf_max_pairwise_rule_dist_max,
                        float(st.get("crf_max_pairwise_rule_dist", 0.0)),
                    )
                    crf_n_minted_total_last = int(st.get("crf_n_minted_total", 0))
                    if (
                        n_active >= CRF_N_ACTIVE_FLOOR
                        and candidates and len(candidates) >= 2
                    ):
                        cf_summ = _consumed_summaries(agent, candidates)
                        if cf_summ is not None and torch.isfinite(cf_summ).all():
                            d_cf = _propagation_counterfactual_delta(agent, cf_summ)
                            if d_cf is not None and math.isfinite(d_cf):
                                prop_counterfactual_deltas.append(d_cf)
            elif is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 ONLY (e2 frozen in P1/P2 for stable measurement).
            if (not is_p1) and (not is_p2) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val):
                    n_p0_contrastive_steps += 1

            _, _harm_signal, done, info, obs_dict = env.step(action)
            if is_p1:
                ep_reward += float(_harm_signal)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

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
            tick_in_ep += 1
            if done:
                break

        if is_p1:
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            l_loss = _lpfc_reinforce_loss(
                agent, outcome_buf, reinforce_baseline, agent.device
            )
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.lateral_pfc.bias_head_parameters(), 1.0
                )
                bias_opt.step()
                n_p1_bias_updates += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] rung={rung['arm_id']} f_weight={f_weight} seed={seed} "
                f"phase={phase_label} ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    _ZG.observe(agent)

    # ----- Per-seed aggregation (over P2) -----
    committed_class_entropy = _entropy_from_int_counts(committed_class_counts)
    frac_pre_ge2 = float(n_p2_pre_ge2 / n_p2_ticks) if n_p2_ticks > 0 else 0.0
    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    if crf_n_active_per_tick:
        frac_crf_active_ge_floor = float(
            sum(1 for n in crf_n_active_per_tick if n >= CRF_N_ACTIVE_FLOOR)
            / len(crf_n_active_per_tick)
        )
        mean_crf_n_active = float(
            sum(crf_n_active_per_tick) / len(crf_n_active_per_tick)
        )
    else:
        frac_crf_active_ge_floor = 0.0
        mean_crf_n_active = 0.0

    if crf_n_matched_per_tick:
        mean_crf_n_matched = float(
            sum(crf_n_matched_per_tick) / len(crf_n_matched_per_tick)
        )
        max_crf_n_matched = int(max(crf_n_matched_per_tick))
    else:
        mean_crf_n_matched = 0.0
        max_crf_n_matched = 0

    crf_differentiated = bool(
        crf_n_minted_total_last >= CRF_MIN_MINTED
        and frac_crf_active_ge_floor >= CRF_FRAC_ACTIVE_FLOOR
    )

    mean_lateral_pfc_bias_abs = (
        float(sum(lateral_pfc_bias_abs_vals) / len(lateral_pfc_bias_abs_vals))
        if lateral_pfc_bias_abs_vals else 0.0
    )
    # C1(d) re-expressed as a DIRECT magnitude floor (no OFF arm here).
    seed_propagation_non_vacuous = bool(mean_lateral_pfc_bias_abs > PROP_NONVAC_FLOOR)

    route_range_mean = (
        float(sum(route_ranges) / len(route_ranges)) if route_ranges else 0.0
    )
    route_active_frac = (
        float(route_active_ticks) / float(n_p2_fresh_select)
        if n_p2_fresh_select > 0 else 0.0
    )
    route_sample_yield = (
        float(n_p2_fresh_select) / float(n_p2_ticks) if n_p2_ticks > 0 else 0.0
    )
    seed_route_ready = bool(
        route_range_mean > ROUTE_RANGE_SEED_FLOOR
        and n_p2_fresh_select >= FRESH_SELECT_FLOOR
        and route_sample_yield >= FRESH_SELECT_YIELD_FLOOR
    )
    mean_prop_counterfactual_delta = (
        float(sum(prop_counterfactual_deltas) / len(prop_counterfactual_deltas))
        if prop_counterfactual_deltas else 0.0
    )

    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_p2_ticks) if n_p2_ticks > 0 else 0.0
    )
    demotion_excluded_count_mean = (
        float(sum(demotion_excluded_counts) / len(demotion_excluded_counts))
        if demotion_excluded_counts else 0.0
    )
    demotion_envelope_size_mean = (
        float(sum(demotion_envelope_sizes) / len(demotion_envelope_sizes))
        if demotion_envelope_sizes else 0.0
    )
    seed_demotion_non_vacuous = bool(
        demotion_active_frac >= DEMOTION_ACTIVE_FRAC_FLOOR
        and demotion_excluded_count_mean > EXCLUDED_COUNT_FLOOR
    )

    nogo_active_frac = (
        float(nogo_active_ticks) / float(n_p2_ticks) if n_p2_ticks > 0 else 0.0
    )
    nogo_suppressed_mean = (
        float(sum(nogo_suppressed_per_tick) / len(nogo_suppressed_per_tick))
        if nogo_suppressed_per_tick else 0.0
    )
    nogo_envelope_size_mean = (
        float(sum(nogo_envelope_sizes) / len(nogo_envelope_sizes))
        if nogo_envelope_sizes else 0.0
    )
    seed_nogo_non_vacuous = bool(
        nogo_active_frac >= NOGO_ACTIVE_FRAC_FLOOR
        and nogo_suppressed_mean > NOGO_SUPPRESSED_FLOOR
    )

    seed_class_axis_exercisable = bool(frac_pre_ge2 > FRAC_PRE_GE2_FLOOR)
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )

    # C1(h) NEW -- F_WEIGHT_KNOB_LIVE. Two-sided closeness check, expressed as an
    # INTERVAL per the two-sided-precondition convention (threshold_low/threshold_high).
    mean_f_raw = float(sum(f_raw_samples) / len(f_raw_samples)) if f_raw_samples else 0.0
    mean_f_weighted = (
        float(sum(f_weighted_samples) / len(f_weighted_samples))
        if f_weighted_samples else 0.0
    )
    fwk_expected = f_weight * mean_f_raw
    fwk_tolerance = F_WEIGHT_SCALING_TOL * max(1e-6, abs(fwk_expected))
    fwk_threshold_low = fwk_expected - fwk_tolerance
    fwk_threshold_high = fwk_expected + fwk_tolerance
    seed_f_weight_knob_live = bool(
        len(f_weighted_samples) > 0
        and fwk_threshold_low <= mean_f_weighted <= fwk_threshold_high
    )

    return {
        "arm_id": rung["arm_id"],
        "label": rung["label"],
        "f_weight": f_weight,
        "seed": int(seed),
        "use_candidate_rule_field": True,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY DV -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        # ----- C1(a) -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "class_axis_exercisable": seed_class_axis_exercisable,
        # ----- C1(b) -----
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        # ----- C1(c) -----
        "crf_mean_n_active": round(mean_crf_n_active, 6),
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_max_pairwise_rule_dist": round(crf_max_pairwise_rule_dist_max, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        "crf_mean_n_matched": round(mean_crf_n_matched, 6),
        "crf_max_n_matched": int(max_crf_n_matched),
        # ----- C1(d) propagation non-vacuity (direct magnitude floor) -----
        "mean_lateral_pfc_bias_abs": round(mean_lateral_pfc_bias_abs, 8),
        "mean_prop_counterfactual_delta": round(mean_prop_counterfactual_delta, 8),
        "propagation_non_vacuous": seed_propagation_non_vacuous,
        # ----- C1(g) routing readiness -----
        "n_p2_fresh_select": int(n_p2_fresh_select),
        "n_p2_latched_ticks": int(n_p2_latched_ticks),
        "modulatory_channel_route_range_mean": round(route_range_mean, 6),
        "modulatory_channel_route_range_max": round(route_range_max, 6),
        "modulatory_channel_route_active_frac": round(route_active_frac, 6),
        "route_sample_yield": round(route_sample_yield, 6),
        "route_ready": seed_route_ready,
        # ----- C1(e) MECH-448 demotion non-vacuity -----
        "f_eligibility_demotion_active_ticks": int(demotion_active_ticks),
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        "f_eligibility_excluded_count_mean": round(demotion_excluded_count_mean, 6),
        "f_eligibility_envelope_size_mean": round(demotion_envelope_size_mean, 6),
        "demotion_non_vacuous": seed_demotion_non_vacuous,
        # ----- C1(f) MECH-449 active No-Go non-vacuity -----
        "go_nogo_active_ticks": int(nogo_active_ticks),
        "go_nogo_active_frac": round(nogo_active_frac, 6),
        "go_nogo_suppressed_per_tick_mean": round(nogo_suppressed_mean, 6),
        "go_nogo_envelope_size_mean": round(nogo_envelope_size_mean, 6),
        "nogo_non_vacuous": seed_nogo_non_vacuous,
        # ----- C1(h) NEW F_WEIGHT_KNOB_LIVE (interval-shaped, per-row) -----
        "f_weight_knob_live": {
            "n_fresh_select_samples": int(len(f_weighted_samples)),
            "mean_f_raw": round(mean_f_raw, 8),
            "mean_f_weighted": round(mean_f_weighted, 8),
            "expected": round(fwk_expected, 8),
            "threshold_low": round(fwk_threshold_low, 8),
            "threshold_high": round(fwk_threshold_high, 8),
            "direction": "interval",
            "met": seed_f_weight_knob_live,
        },
        "f_weight_knob_live_met": seed_f_weight_knob_live,
    }


def _rung_rows(arm_results: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in arm_results
        if r["arm_id"] == arm_id and r["error_note"] is None
    ]


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []
    script_path = Path(__file__).resolve()

    for rung in RUNGS:
        print(
            f"Rung {rung['arm_id']} ({rung['label']}) f_weight={rung['f_weight']} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        for s in seeds:
            print(f"Seed {s} Condition {rung['label']}", flush=True)
            row = _run_seed_rung(
                rung, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
            )
            row["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm_id": rung["arm_id"],
                    "f_weight": float(rung["f_weight"]),
                    "use_candidate_rule_field": True,
                    "crf_persist_rules_across_episode_reset": True,
                    "crf_mature_pool_dynamics": True,
                    "crf_context_from_e2_world_forward": True,
                    "crf_availability_maintenance": True,
                    "crf_maintenance_floor": float(CRF_MAINTENANCE_FLOOR),
                    "crf_maintenance_decay": float(CRF_MAINTENANCE_DECAY),
                    "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
                    "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
                    "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
                    "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
                    "f_eligibility_envelope_floor": float(F_ELIGIBILITY_ENVELOPE_FLOOR),
                    "f_eligibility_dn_sigma": float(F_ELIGIBILITY_DN_SIGMA),
                    "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
                    "f_eligibility_adaptive_mean_factor": float(F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR),
                    "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
                    "use_dacc": bool(USE_DACC),
                    "gng_perseveration_floor": float(GNG_PERSEVERATION_FLOOR),
                    "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
                    "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
                    "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
                    "env_kwargs": dict(ENV_KWARGS),
                    "sd056_weight": SD056_WEIGHT,
                    "lr_lpfc_bias": LR_LPFC_BIAS,
                    "p0_episodes": int(p0_episodes),
                    "p1_episodes": int(p1_episodes),
                    "p2_episodes": int(p2_episodes),
                    "steps_per_episode": int(steps_per_episode),
                },
                seed=s,
                script_path=script_path,
                rng_fully_reset=True,
                extra_ineligible_reasons=[
                    "online_e2_training_stateful_per_cell",
                    "p1_bias_head_reinforce_training_stateful_per_cell",
                ],
            )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    rung_rows = {a: _rung_rows(arm_results, a) for a in RUNG_IDS}

    def _n_meeting(key: str) -> Dict[str, int]:
        return {a: sum(1 for r in rung_rows[a] if r[key]) for a in RUNG_IDS}

    n_axis = _n_meeting("class_axis_exercisable")
    c1a_holds = bool(all(n_axis[a] >= MIN_SEEDS_FOR_PASS for a in RUNG_IDS))

    n_gapa = _n_meeting("gapa_divergence")
    c1b_holds = bool(all(n_gapa[a] >= MIN_SEEDS_FOR_PASS for a in RUNG_IDS))

    n_crf = _n_meeting("crf_differentiated")
    c1c_holds = bool(all(n_crf[a] >= MIN_SEEDS_FOR_PASS for a in RUNG_IDS))

    n_prop = _n_meeting("propagation_non_vacuous")
    c1d_holds = bool(all(n_prop[a] >= MIN_SEEDS_FOR_PASS for a in RUNG_IDS))

    n_demotion = _n_meeting("demotion_non_vacuous")
    c1e_holds = bool(all(n_demotion[a] >= MIN_SEEDS_FOR_PASS for a in RUNG_IDS))

    n_nogo = _n_meeting("nogo_non_vacuous")
    c1f_holds = bool(all(n_nogo[a] >= MIN_SEEDS_FOR_PASS for a in RUNG_IDS))

    n_route = _n_meeting("route_ready")
    c1g_holds = bool(all(n_route[a] >= MIN_SEEDS_FOR_PASS for a in RUNG_IDS))

    n_fwk = _n_meeting("f_weight_knob_live_met")
    c1h_holds = bool(all(n_fwk[a] >= MIN_SEEDS_FOR_PASS for a in RUNG_IDS))

    c1_holds = bool(
        c1a_holds and c1b_holds and c1c_holds and c1d_holds
        and c1e_holds and c1f_holds and c1g_holds and c1h_holds
    )

    # C2 (PRIMARY): paired-by-seed committed-class entropy lift, F000 over F100.
    f100_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in rung_rows["ARM_F100"]}
    f000_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in rung_rows["ARM_F000"]}
    paired_lifts: Dict[int, float] = {}
    n_lift_seeds = 0
    for seed in sorted(set(f100_by_seed) & set(f000_by_seed)):
        lift = f000_by_seed[seed] - f100_by_seed[seed]
        paired_lifts[seed] = round(lift, 6)
        if lift >= C2_LIFT_MARGIN_NATS:
            n_lift_seeds += 1
    c2_holds = bool(n_lift_seeds >= C2_MIN_LIFT_SEEDS)

    # SECONDARY/DIAGNOSTIC (non-gating): the full ladder shape.
    ladder_shape: Dict[str, Any] = {}
    for a in RUNG_IDS:
        rows_a = rung_rows[a]
        ladder_shape[a] = {
            "f_weight": RUNG_F_WEIGHT[a],
            "per_seed_committed_class_entropy_nats": {
                str(r["seed"]): r["committed_class_entropy_nats"] for r in rows_a
            },
            "mean_committed_class_entropy_nats": round(
                _mean([r["committed_class_entropy_nats"] for r in rows_a]), 6
            ),
        }

    # ----- Outcome map (THREE pre-registered branches; NO weakens -- claim_ids=[]) -----
    if not c1_holds:
        outcome = "FAIL"
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    elif c2_holds:
        outcome = "PASS"
        direction = "non_contributory"
        label = "f_dominance_confirmed_h2_operative_ceiling"
    else:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "f_dominance_refuted_as_sole_cause"

    interpretation = {
        "label": label,
        "dv_symmetry_declaration": (
            "Committed-class entropy is a function of the argmax action-class "
            "sequence. f_weight rescales ONE per-candidate-VARYING term (F, the "
            "reality cost -- NOT a broadcast scalar; F differs across candidates at "
            "every tick) relative to the other additive terms (M, Phi) in the score "
            "sum. This is neither a uniform additive shift (argmax-cancelling) nor a "
            "monotone rescaling of the WHOLE score -- only one component's relative "
            "weight changes, so relative candidate ordering under the other terms is "
            "not preserved -- so it is NOT invariant under the argmax symmetry of the "
            "committed-class DV, and legitimately CAN change which candidate wins the "
            "committed argmin. Holds identically at every rung (same kind of "
            "rescaling, only the coefficient differs)."
        ),
        "preconditions": [
            {
                "name": "committed_class_axis_exercisable_all_rungs",
                "kind": "readiness",
                "description": (
                    "frac of P2 ticks with >= 2 candidate first-action classes "
                    "exceeds floor on a majority of seeds, on EVERY rung."
                ),
                "control": "SP-CEM multi-class candidate pool, all 4 rungs",
                "measured": float(min(n_axis[a] for a in RUNG_IDS)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c1a_holds),
            },
            {
                "name": "gapa_consumed_summary_divergence_all_rungs",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate "
                    "SPREAD clears the floor on a majority of seeds, on EVERY rung."
                ),
                "control": "e2 trained online in P0; candidate_summary_source=e2_world_forward",
                "measured": float(min(n_gapa[a] for a in RUNG_IDS)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c1b_holds),
            },
            {
                "name": "gapa_consumed_summary_bounded",
                "kind": "readiness",
                "description": (
                    "consumed-summary spread stayed below the explosion ceiling "
                    "(SD-056 online-training numerical stability; rollout-norm clamp ON)."
                ),
                "control": "max consumed_summary_pairwise_dist across all cells",
                "measured": float(
                    max(
                        [r["consumed_summary_pairwise_dist_max"] for rows in rung_rows.values() for r in rows]
                        or [0.0]
                    )
                ),
                "threshold": float(CONSUMED_MAGNITUDE_CEIL),
                "comparator": "<",
                "direction": "upper",
                "met": bool(
                    max(
                        [r["consumed_summary_pairwise_dist_max"] for rows in rung_rows.values() for r in rows]
                        or [0.0]
                    ) < CONSUMED_MAGNITUDE_CEIL
                ),
            },
            {
                "name": "crf_field_differentiated_and_matured_all_rungs",
                "kind": "readiness",
                "description": (
                    "CandidateRuleField (matched constant ON) minted >= CRF_MIN_MINTED "
                    "distinct rules AND fired a non-zero rule_state on >= "
                    "CRF_FRAC_ACTIVE_FLOOR of P2 ticks, on a majority of seeds, on "
                    "EVERY rung."
                ),
                "control": "crf frac-active (matured pool) + crf_n_minted_total",
                "measured": float(min(n_crf[a] for a in RUNG_IDS)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c1c_holds),
            },
            {
                "name": "propagation_non_vacuity_lateral_pfc_bias_all_rungs",
                "kind": "readiness",
                "description": (
                    "mean |lateral_pfc bias| > PROP_NONVAC_FLOOR on a majority of "
                    "seeds, on EVERY rung. Re-expressed as a DIRECT magnitude floor "
                    "(no OFF arm in this design -- CRF is a matched constant ON "
                    "everywhere, unlike 851's paired ON-vs-OFF diff)."
                ),
                "control": "mean_lateral_pfc_bias_abs on the matched CRF-ON stack, all rungs",
                "measured": float(min(n_prop[a] for a in RUNG_IDS)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c1d_holds),
            },
            {
                "name": "mech448_demotion_lever_live_and_excluding_all_rungs",
                "kind": "readiness",
                "description": (
                    "The MECH-448 f_eligibility-demotion conversion constant "
                    "(matched on EVERY rung) is ACTIVE on >= DEMOTION_ACTIVE_FRAC_FLOOR "
                    "of P2 ticks AND actually EXCLUDES (mean excluded_count > "
                    "EXCLUDED_COUNT_FLOOR), on a majority of seeds, on EVERY rung."
                ),
                "control": "f_demotion envelope active-frac + excluded_count, all rungs (matched lever)",
                "measured": float(min(n_demotion[a] for a in RUNG_IDS)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c1e_holds),
            },
            {
                "name": "mech449_active_nogo_live_and_suppressing_all_rungs",
                "kind": "readiness",
                "description": (
                    "The MECH-449 Go/No-Go eligibility constitution (matched on "
                    "EVERY rung) is ACTIVE on >= NOGO_ACTIVE_FRAC_FLOOR of P2 ticks "
                    "AND actually SUPPRESSES (mean per-tick safety+soft No-Go "
                    "removals > NOGO_SUPPRESSED_FLOOR), on a majority of seeds, on "
                    "EVERY rung."
                ),
                "control": "Go/No-Go constitution active-frac + suppressed-count, all rungs (matched lever)",
                "measured": float(min(n_nogo[a] for a in RUNG_IDS)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c1f_holds),
            },
            {
                "name": "lateral_pfc_route_range_supra_floor_and_sample_adequate_all_rungs",
                "kind": "readiness",
                "description": (
                    "modulatory_channel_route_source='lateral_pfc' (V3-EXQ-851's "
                    "corrected route source) identity-routes the SD-033a "
                    "rule-apprehension bias into _modulatory_accum. On a majority of "
                    "seeds, on EVERY rung (routing is a matched constant): (a) the "
                    "routed range's mean clears ROUTE_RANGE_SEED_FLOOR; (b) the "
                    "genuine fresh-selection sample (latch-cleared) clears "
                    "FRESH_SELECT_FLOOR; (c) the fresh-select YIELD clears "
                    "FRESH_SELECT_YIELD_FLOOR."
                ),
                "control": "modulatory_channel_route_range + n_p2_fresh_select, all rungs (matched lever)",
                "measured": float(min(n_route[a] for a in RUNG_IDS)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c1g_holds),
            },
            {
                "name": "f_weight_knob_live_through_full_matched_stack_all_rungs",
                "kind": "readiness",
                "description": (
                    "NEW C1(h). At each rung/seed, on genuine fresh-select P2 ticks, "
                    "mean_f_weighted must fall within "
                    "[rung_f_weight*mean_f_raw - tol, rung_f_weight*mean_f_raw + tol] "
                    "(tol = F_WEIGHT_SCALING_TOL * max(1e-6, |expected|), a two-sided "
                    "closeness check -- see per-row f_weight_knob_live for the "
                    "interval bounds), on a majority of seeds, on EVERY rung. "
                    "Confirms f_weight genuinely rescales F through the FULL "
                    "matched-stack pipeline (CRF-ON, lateral_pfc-routed, "
                    "MECH-448/449-active) at every attenuation level -- V3-EXQ-852 "
                    "only checked substrate DEFAULTS."
                ),
                "control": (
                    "agent.e3._last_traj_components['f']/['f_weighted'] on genuine "
                    "fresh-select ticks, all rungs (matched-stack readiness, "
                    "COUNT-shaped aggregate over a per-row INTERVAL-decided boolean)"
                ),
                "measured": float(min(n_fwk[a] for a in RUNG_IDS)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c1h_holds),
            },
        ],
        "criteria": [
            {
                "name": "C2_committed_class_entropy_lift_f000_vs_f100",
                "load_bearing": True,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "C1a_class_axis_exercisable": bool(c1a_holds),
            "C1b_gapa_divergence": bool(c1b_holds),
            "C1c_crf_differentiated_matured": bool(c1c_holds),
            "C1d_propagation_non_vacuity": bool(c1d_holds),
            "C1e_mech448_demotion_live_and_excluding": bool(c1e_holds),
            "C1f_mech449_active_nogo_live_and_suppressing": bool(c1f_holds),
            "C1g_lateral_pfc_route_range_supra_floor_and_sample_adequate": bool(c1g_holds),
            "C1h_f_weight_knob_live": bool(c1h_holds),
            "C2_paired_lift": bool(c2_holds),
        },
        "note": (
            "GOV-FANOUT-1 Leg P-B (H2: F-dominance, downstream of selection). "
            "claim_ids=[] (diagnostic; brake-exempt). Branch 2 "
            "(f_dominance_confirmed_h2_operative_ceiling) is evidence_direction="
            "'non_contributory' (claim-neutral by design) but is nonetheless a "
            "load-bearing GOVERNANCE ROUTING signal: if reached, ARC-062's "
            "behavioural retest should be gated behind an F-rebalance substrate "
            "landing before any further probe. Branch 3 "
            "(f_dominance_refuted_as_sole_cause) does NOT reopen H3 "
            "(competence/action-learning) -- per the fanout doc's P-C resolution, "
            "H-policy-learning is already 'eliminated' in "
            "hypothesis_space_registry.v1.json; the surviving live root is "
            "H-observation-interface (MECH-457-tagged, owned by the standing "
            "competence_floor re-posing thread), which this branch corroborates "
            "rather than reopens."
        ),
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "n_rungs": len(RUNGS),
        "rung_f_weights": dict(RUNG_F_WEIGHT),
        "total_cells_attempted": int(len(RUNGS) * len(seeds)),
        "total_cells_completed": int(sum(len(rung_rows[a]) for a in RUNG_IDS)),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "c2_min_lift_seeds": int(C2_MIN_LIFT_SEEDS),
            "frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "consumed_magnitude_ceil": float(CONSUMED_MAGNITUDE_CEIL),
            "crf_min_minted": int(CRF_MIN_MINTED),
            "crf_n_active_floor": int(CRF_N_ACTIVE_FLOOR),
            "crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
            "crf_dist_floor": float(CRF_DIST_FLOOR),
            "prop_nonvac_floor": float(PROP_NONVAC_FLOOR),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "lr_lpfc_bias": float(LR_LPFC_BIAS),
            "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
            "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
            "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
            "demotion_active_frac_floor": float(DEMOTION_ACTIVE_FRAC_FLOOR),
            "excluded_count_floor": float(EXCLUDED_COUNT_FLOOR),
            "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
            "use_dacc": bool(USE_DACC),
            "gng_perseveration_floor": float(GNG_PERSEVERATION_FLOOR),
            "use_modulatory_channel_routing": bool(USE_MODULATORY_CHANNEL_ROUTING),
            "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
            "route_range_seed_floor": float(ROUTE_RANGE_SEED_FLOOR),
            "beta_rate_max_steps": int(BETA_RATE_MAX_STEPS),
            "fresh_select_floor": int(FRESH_SELECT_FLOOR),
            "fresh_select_yield_floor": float(FRESH_SELECT_YIELD_FLOOR),
            "nogo_active_frac_floor": float(NOGO_ACTIVE_FRAC_FLOOR),
            "nogo_suppressed_floor": float(NOGO_SUPPRESSED_FLOOR),
            "f_weight_scaling_tol": float(F_WEIGHT_SCALING_TOL),
            "sd056_weight": float(SD056_WEIGHT),
            "crf_persist_rules_across_episode_reset": True,
        },
        "acceptance_criteria": {
            "C1_substrate_exercisable_and_manipulation_live": c1_holds,
            "C1a_class_axis_exercisable_all_rungs": c1a_holds,
            "C1a_n_meeting_by_rung": n_axis,
            "C1b_gapa_divergence_all_rungs": c1b_holds,
            "C1b_n_meeting_by_rung": n_gapa,
            "C1c_crf_differentiated_matured_all_rungs": c1c_holds,
            "C1c_n_meeting_by_rung": n_crf,
            "C1d_propagation_non_vacuity_all_rungs": c1d_holds,
            "C1d_n_meeting_by_rung": n_prop,
            "C1e_mech448_demotion_live_all_rungs": c1e_holds,
            "C1e_n_meeting_by_rung": n_demotion,
            "C1f_mech449_active_nogo_live_all_rungs": c1f_holds,
            "C1f_n_meeting_by_rung": n_nogo,
            "C1g_route_ready_all_rungs": c1g_holds,
            "C1g_n_meeting_by_rung": n_route,
            "C1h_f_weight_knob_live_all_rungs": c1h_holds,
            "C1h_n_meeting_by_rung": n_fwk,
            "C2_committed_class_lift_f000_vs_f100": c2_holds,
            "C2_n_lift_seeds": int(n_lift_seeds),
            "C2_paired_lifts_by_seed": paired_lifts,
            "C2_f100_mean_committed_class_entropy": round(_mean(list(f100_by_seed.values())), 6),
            "C2_f000_mean_committed_class_entropy": round(_mean(list(f000_by_seed.values())), 6),
        },
        "ladder_shape_secondary_diagnostic_not_load_bearing": {
            "note": (
                "Committed-class entropy at every rung, per seed and cross-seed mean. "
                "NOT gated on -- informative only, to see whether the ladder shape is "
                "monotonic, threshold-like, or flat regardless of C2's pass/fail."
            ),
            "by_rung": ladder_shape,
        },
        "interpretation_grid": {
            "PASS_C1_C2": (
                "GOV-FANOUT-1 Leg P-B, H2 (F-dominance) CONFIRMED as the operative "
                "ceiling: C1 (including the NEW C1h f_weight-knob-live gate) holds "
                "on all 4 rungs, and fully removing F from the committed argmin "
                "(f_weight=0.0) lifts committed-class entropy over the current "
                "substrate default (f_weight=1.0) by >= C2_LIFT_MARGIN_NATS on a "
                "majority of seeds. non_contributory (claim_ids=[]), but a "
                "load-bearing GOVERNANCE ROUTING signal: the fix is an F-rebalance "
                "substrate, and ARC-062's behavioural retest should be gated behind "
                "that substrate landing."
            ),
            "FAIL_C1_holds_C2_fails": (
                "GOV-FANOUT-1 Leg P-B, H2-as-sole-cause REFUTED: C1 fully holds "
                "(the matched stack is exercisable, differentiated, propagating, "
                "and the f_weight knob is genuinely live through the full pipeline "
                "at every rung) but even full F-removal does not lift committed-class "
                "entropy by the pre-registered margin. non_contributory. Per the "
                "fanout doc's P-C resolution, H3 (competence) is already resolved "
                "(H-policy-learning eliminated); this does NOT reopen it -- it "
                "corroborates the already-tracked H-observation-interface root."
            ),
            "FAIL_C1_substrate_not_ready_requeue": (
                "The committed-class axis was not exercisable, and/or GAP-A "
                "divergence was absent, and/or the CRF field did not mature, "
                "and/or propagation was vacuous, and/or the MECH-448/449 matched "
                "levers were vacuous, and/or the lateral_pfc route was under-sampled, "
                "and/or (NEW C1h) the f_weight knob was not verifiably live through "
                "the full pipeline on some rung -- the H2 test could not be run "
                "cleanly this cycle. Not a falsification. Route to substrate "
                "enrichment / re-queue."
            ),
        },
        "arm_results": arm_results,
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
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "evidence_direction_note": (
            f"V3-EXQ-858 ARC-062 conversion-fanout GOV-FANOUT-1 Leg P-B (H2: "
            f"F-dominance, downstream of selection). claim_ids=[] (diagnostic; "
            f"brake-exempt per the fanout doc Section 4). Design of record: "
            f"REE_assembly/evidence/planning/arc_062_conversion_fanout_2026-07-29.md "
            f"Section 2 Leg P-B + the P-B buildability resolution (2026-07-31). "
            f"Matched-stack base identical to V3-EXQ-851's ARM_ON config (SP-CEM, "
            f"GAP-A e2_world_forward candidate summaries, MECH-448 rank-preserving "
            f"F->eligibility demotion, MECH-449 active Go/No-Go opponency, the "
            f"CRF/LateralPFCAnalog rule-apprehension stack routed via "
            f"modulatory_channel_route_source='lateral_pfc' -- V3-EXQ-851's "
            f"CORRECTED route source, per the P-A erratum, NOT the design doc's "
            f"original literal 'gated_policy' text). SD-085 (landed ree-v3 main "
            f"2026-07-31; REE_assembly/docs/architecture/sd_085_e3_reality_cost_"
            f"weight.md) added the E3Config.f_weight coefficient this leg sweeps. "
            f"SWEPT VARIABLE: agent.e3.config.f_weight, a 4-rung attenuation ladder "
            f"{{{', '.join(f'{a}={RUNG_F_WEIGHT[a]}' for a in RUNG_IDS)}}} "
            f"(replacing 851's use_candidate_rule_field ON/OFF sweep -- that channel "
            f"is now a MATCHED CONSTANT, fixed True on every rung). C1 (readiness, "
            f"mirrors 851's C1a-g exactly, re-scoped from 'both arms' to 'all four "
            f"rungs') PLUS NEW C1h (F_WEIGHT_KNOB_LIVE: confirms f_weight genuinely "
            f"rescales F through the full matched-stack pipeline at every rung, "
            f"beyond V3-EXQ-852's substrate-defaults-only check). C2 (PRIMARY) = "
            f"paired-by-seed committed-class entropy lift, ARM_F000 (f_weight=0.0) "
            f"over ARM_F100 (f_weight=1.0, the substrate default baseline rung), "
            f">= {C2_LIFT_MARGIN_NATS} nats on >= {C2_MIN_LIFT_SEEDS}/3 seeds. "
            f"interpretation_label={result['interpretation_label']}. "
            f"C1={result['acceptance_criteria']['C1_substrate_exercisable_and_manipulation_live']}, "
            f"C2={result['acceptance_criteria']['C2_committed_class_lift_f000_vs_f100']}. "
            f"THREE pre-registered branches, NO weakens (claim_ids=[]): C1-fail -> "
            f"substrate_not_ready_requeue (evidence_direction=unknown); C1-holds + "
            f"C2-holds -> f_dominance_confirmed_h2_operative_ceiling "
            f"(evidence_direction=non_contributory, but a load-bearing GOVERNANCE "
            f"ROUTING signal -- ARC-062's behavioural retest should be gated behind "
            f"an F-rebalance substrate landing); C1-holds + C2-fails -> "
            f"f_dominance_refuted_as_sole_cause (evidence_direction=non_contributory; "
            f"does NOT reopen H3 -- per the fanout doc's P-C resolution, "
            f"H-policy-learning is already eliminated in "
            f"hypothesis_space_registry.v1.json, so this corroborates the already-"
            f"tracked H-observation-interface root rather than reopening a new "
            f"question). PROMOTES NOTHING by itself (claim_ids=[])."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "rungs": "4-rung f_weight attenuation ladder: ARM_F100 (1.0, baseline) / "
                     "ARM_F050 (0.5) / ARM_F025 (0.25) / ARM_F000 (0.0, F fully removed)",
            "swept_variable": "agent.e3.config.f_weight",
            "matched_constant_use_candidate_rule_field": True,
            "crf_persist_rules_across_episode_reset": True,
            "crf_mature_pool_dynamics": True,
            "crf_context_from_e2_world_forward": True,
            "crf_availability_maintenance": True,
            "crf_maintenance_floor": float(CRF_MAINTENANCE_FLOOR),
            "crf_maintenance_decay": float(CRF_MAINTENANCE_DECAY),
            "crf_mature_context_match_threshold": float(CRF_MATURE_CONTEXT_MATCH_THRESHOLD),
            "crf_tolerance_conflict_cap": int(CRF_TOLERANCE_CONFLICT_CAP),
            "crf_maintenance_couple_to_theta": bool(CRF_MAINTENANCE_COUPLE_TO_THETA),
            "matched_stack": (
                "SP-CEM + candidate_summary_source=e2_world_forward (GAP-A/649, e2 "
                "trained online in P0) + use_modulatory_selection_authority (643a) + "
                "channel routing (lateral_pfc, V3-EXQ-851 corrected route source) + "
                "MECH-448 RANK-PRESERVING F->ELIGIBILITY DEMOTION + MECH-449 GO/NO-GO "
                "CONSTITUTION + MECH-341 stratified + MECH-313 noise floor + V_s "
                "minimal + use_gated_policy + use_lateral_pfc_analog "
                "(lateral_pfc_train_rule_bias_head=True, TRAINED in P1) + SD-056 all "
                "levers -- IDENTICAL on every rung; ONLY f_weight varies"
            ),
            "primary_dv": "committed-class entropy",
            "phases": "P0 e2-train (field matures) -> P1 frozen-encoder bias-head REINFORCE -> P2 frozen measurement",
            "p1_bias_head_trained_via_reinforce": True,
            "e2_trained_in_p0_frozen_in_p1_p2": True,
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
            "use_modulatory_channel_routing": USE_MODULATORY_CHANNEL_ROUTING,
            "modulatory_channel_route_source": MODULATORY_CHANNEL_ROUTE_SOURCE,
            "use_f_eligibility_demotion": USE_F_ELIGIBILITY_DEMOTION,
            "use_go_nogo_constitution": USE_GO_NOGO_CONSTITUTION,
            "use_dacc": USE_DACC,
            "sd085_f_weight_coefficient": "REEConfig.e3.f_weight (E3Config), default 1.0, set post-construction per rung",
            "lateral_pfc_route_source_fix": (
                "V3-EXQ-851's corrected route source (2026-07-31): "
                "modulatory_channel_route_source='lateral_pfc' identity-routes "
                "_bdc_lpfc, NOT the design doc's original literal 'gated_policy' text"
            ),
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "sd056_output_norm_clamp": SD056_OUTPUT_NORM_CLAMP,
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-858 ARC-062 conversion-fanout Leg P-B (H2 F-dominance; "
                     "f_weight attenuation ladder, committed-class entropy)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    global SEEDS
    if args.dry_run:
        SEEDS = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
    else:
        p0 = P0_WARMUP_EPISODES
        p1 = P1_BIAS_TRAIN_EPISODES
        p2 = P2_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=SEEDS,
        p0_episodes=p0,
        p1_episodes=p1,
        p2_episodes=p2,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = None
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_cells_completed']}/{result['total_cells_attempted']} "
        f"C1={result['acceptance_criteria']['C1_substrate_exercisable_and_manipulation_live']} "
        f"C2={result['acceptance_criteria']['C2_committed_class_lift_f000_vs_f100']} "
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
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
