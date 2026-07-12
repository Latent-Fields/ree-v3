#!/opt/local/bin/python3
"""
V3-EXQ-614c -- GAP-B Phase P3 MECH-341 stratified_within_class_temperature
4-arm sweep on the SD-056-amended baseline.

Successor to V3-EXQ-614b (FAIL_no_criterion 2026-05-31, manifest
v3_exq_614b_mech341_p3_behavioural_falsifier_3arm_sd056_amended_20260531T182040Z_v3.json).
Routed by failure_autopsy_V3-EXQ-616_2026-05-31 Sections 7 + 10
(contingent-on-614b-FAIL-C1 path) and failure_autopsy_V3-EXQ-614b_2026-05-31:
the 614b 3-arm falsifier failed C1 R2.c (B_only ARM_0 frac_pre_ge2=0.0 -- structural
CEM proposer collapse without SP-CEM Layer A) AND C2 necessity_delta 0.087 just
below pre-amend 0.1 threshold; C3 ALL_ON Rung-1 PASSed at 0.800 nats (highest
of any 614-lineage run -- positive substrate-readiness for SD-056 amend at the
behavioural-runtime horizon). The 616 autopsy named the contingent action
"stratified_temperature default + A-vs-B partial-redundancy probe"; this
experiment validates Part (a) of the resulting MECH-341 amend landed
2026-06-01 via /implement-substrate.

What's different from V3-EXQ-614b
---------------------------------
- QUEUE_ID = V3-EXQ-614c
- EXPERIMENT_TYPE = v3_exq_614c_mech341_stratified_within_class_temperature_sweep
- SUPERSEDES = V3-EXQ-614b
- 4 arms (vs 3), all on the SD-056-amended baseline (ARM_2 ALL_ON config
  from 614b: SP-CEM ON + MECH-341 ON + MECH-313 ON + V_s minimal ON + the
  5 SD-056 amend flags). The ONLY axis swept is the new
  `stratified_within_class_temperature` lever introduced by the 2026-06-01
  MECH-341 amend.
- claim_ids = [MECH-341] (ARC-065 dropped vs 614b; this sweep only varies
  the score-layer within-class temperature, which is a Layer-B isolation
  probe; ARC-065's distributed-pathway commitment is not directly varied).
  Per REE_assembly/CLAUDE.md claim_ids accuracy rule.
- experiment_purpose = "evidence" (within-class temperature lift will move
  governance state if C2 fires).

Arms (4, all on the same SD-056-amended baseline as V3-EXQ-614b ARM_2)
----------------------------------------------------------------------
  ARM_0_LEGACY:   stratified_within_class_temperature = None   (legacy argmin)
  ARM_1_T_0_5:    stratified_within_class_temperature = 0.5    (sharpened)
  ARM_2_T_1_0:    stratified_within_class_temperature = 1.0    (mid-T)
  ARM_3_T_2_0:    stratified_within_class_temperature = 2.0    (flatter)

All other levers held at the V3-EXQ-614b ARM_2 ALL_ON config:
  Layer A: SP-CEM ON (use_support_preserving_cem=True)
  Layer B: MECH-341 ON (use_e3_score_diversity=True; both sub-flavours ON;
           entropy_bias_scale=2.0)
  Layer C: MECH-313 ON (use_noise_floor=True; noise_floor_alpha=0.1)
  Layer D: V_s minimal stack ON (use_per_stream_vs=True + use_vs_rollout_gating=True)
  SD-056 amend: all 5 lever flags ON (multi-step contrastive h=5 +
                per-step output norm clamp ratio=2.0 + t=1 contrastive)
  use_differentiable_cem: NOT FLIPPED (default False; SD-055 safety note
                          per substrate_queue.json preserved)

Pre-registered acceptance criteria
----------------------------------
  C1 (regression guard): ARM_0_LEGACY mean_selected_class_entropy_nats
     within 10% of the V3-EXQ-614b ARM_2 ALL_ON value (0.800 nats).
     Specifically, ARM_0_LEGACY mean_selected_class_entropy_nats in
     [0.720, 0.880] on majority (>= 2/3) of seeds. This is the
     bit-identical-OFF check: amend default None must reproduce 614b
     ARM_2 ALL_ON behaviour. Direct test that the new lever has not
     broken the legacy path.
  C2 (within-class lift): at least one of {ARM_1, ARM_2, ARM_3}
     produces mean_selected_class_entropy_nats >= 0.800 (the 614b
     ARM_2 ALL_ON value) on majority (>= 2/3) of seeds. Direct test
     that within-class proportional sampling adds diversity over the
     legacy argmin-within-class baseline.
  C3 (substrate-readiness): all 4 arms produce frac_pre_ge2 > 0.3 on
     majority (>= 2/3) of seeds. Substrate-readiness check at the
     SP-CEM-ON layer; should be high since SP-CEM is active across
     all arms (frac_pre_ge2 = 1.0 in 614b ARM_2).

Overall outcome:
  PASS = C1 fires (regression guard holds) AND (C2 fires OR C3 fires).
  FAIL = C1 fails (amend broke legacy bit-identical OFF, or upstream
         substrate drifted) OR neither C2 nor C3 fires.

Interpretation grid (FOUR rows):
  PASS via C1 + C2:
    Within-class proportional sampling lever validated. MECH-341
    within-class sub-axis is load-bearing under SD-056-amended
    substrate. Route to /governance for MECH-341 supports + v3_pending
    clearance candidate. Within-class temperature winner identified
    for downstream wiring decisions.

  PASS via C1 + C3 only (no within-class lift):
    Amend is functional but not load-bearing under SD-056-amended
    substrate. Layer B may be redundant with Layer A under SP-CEM-ON.
    Route to /governance with MECH-341 mixed (substrate is firing but
    within-class lever produces no marginal improvement over legacy
    argmin) + propagate to Q-054 collapse-with-ARC-065 question.

  FAIL via C1 fail (regression vs 614b ARM_2 ALL_ON > 10%):
    Amend broke the legacy bit-identical OFF guarantee, OR upstream
    substrate drifted between 614b run and 614c run (SD-056 amend
    config drift, scaffolded_sd054_onboarding default change, etc.).
    Route to /failure-autopsy on the amend itself (the only path the
    contract suite cannot catch -- contracts verify per-call output
    determinism but not end-to-end environment-level behaviour).

  FAIL via neither C2 nor C3 (C1 holds):
    Within-class lever is functional but produces no measurable lift
    over legacy AND substrate is not surfacing diversity. Substrate-
    redesign signal. Route to /failure-autopsy on the combined
    MECH-341 + SP-CEM stack -- the score-layer preserver may need a
    different structural mechanism than within-class proportional
    sampling can deliver.

Claims: [MECH-341].
  MECH-341 is the only claim under direct test in this sweep. ARC-065
  is downstream beneficiary if C2 fires (within-class lever contributes
  to the distributed-pathway commitment) but is not varied here.

Cross-plan rationale
--------------------
Same amend transitively unblocks arc_062_rule_apprehension:GAP-B
(V3-EXQ-543l successor cohort) under shared SD-056-amended substrate.
Layer B within-class diversity contributes to ARC-062 head-input
candidate-pool diversity; if 614c surfaces a within-class lift, the
ARC-062 cohort inherits the same substrate.

experiment_purpose=evidence (weighted in confidence/conflict scoring per
Phase-3 governance rules).

Multi-claim direction (single-claim):
  evidence_direction reads off the interpretation grid row that fires:
    PASS via C1 + C2          -> supports.
    PASS via C1 + C3 only     -> mixed.
    FAIL via C1 fail          -> weakens (regression on legacy).
    FAIL via neither C2 nor C3 -> weakens (within-class lever non-load-bearing).

Phases
------
P0 (30 ep, instrumentation OFF): encoder warmup.
P1 (60 ep, instrumentation ON): behavioural measurement window. Matches
the V3-EXQ-614b P1 budget for direct manifest comparability.

Budget: 4 arms x 3 seeds x 90 ep x 200 steps = 216k steps total.
Estimated ~70-90 min on Mac (DLAPTOP-4.local @ ~14 steps/sec). Slightly
larger than 614b (162k) because 4 arms instead of 3.

Implementation notes
--------------------
- env_kwargs IDENTICAL to V3-EXQ-614b -- same SD-054 reef-bipartite +
  hazard_food_attraction + harm_history substrate. Direct manifest
  comparability with the 614b ARM_2 ALL_ON manifest.
- Each arm is an independent REEAgent build; _make_agent reads the
  per-arm within-class temperature from the ARMs spec and passes through
  REEConfig.from_dims(e3_diversity_stratified_within_class_temperature=...).
- All four substrate isolation axes (A/B/C/D) held constant ON across
  all arms; only the within-class temperature varies. This is a Layer-B
  sub-axis isolation, not a Layer-axis isolation (614b already covered
  that).
- 4-row interpretation grid wired into _classify_outcome (different
  from 614b's grid because the axes are different).

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
("Amend 2026-06-01" section), REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
GAP-B governance_2026_06_01, REE_assembly/evidence/planning/substrate_queue.json
MECH-341 amend_history, REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-616_2026-05-31.md
Sections 7 + 10, REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-614b_2026-05-31.md.

LEGACY 614b CHARTER NOTES (preserved below for cluster context)
================================================================

Successor to V3-EXQ-614a (PASS 2026-05-30T19:32Z, interpretation_label
PASS_C2_C3_only_mech341_load_bearing_in_stack_only). User-confirmed re-run on
the substrate where the SD-056 multi-step rollout stability amend has been
landed (ree-v3 main d327b89, 2026-05-31T11:25Z), with V3-EXQ-617 substrate-
readiness PASS at 11:31Z. The amend introduces two togglable levers
(multi-step contrastive loss h=5 and per-step output norm clamp ratio=2.0)
that bound iterated E2 forward rollouts at the behavioural-runtime horizon.

Successor to V3-EXQ-614a (PASS 2026-05-30T19:32Z, interpretation_label
PASS_C2_C3_only_mech341_load_bearing_in_stack_only). User-confirmed re-run on
the substrate where the SD-056 multi-step rollout stability amend has been
landed (ree-v3 main d327b89, 2026-05-31T11:25Z), with V3-EXQ-617 substrate-
readiness PASS at 11:31Z. The amend introduces two togglable levers
(multi-step contrastive loss h=5 and per-step output norm clamp ratio=2.0)
that bound iterated E2 forward rollouts at the behavioural-runtime horizon.

Why a re-run, not a new hypothesis. V3-EXQ-569e Pathway A vs B mechanism
probe was autopsied verdict=INSTRUMENTATION_FAILURE 2026-05-31: SD-056
contrastive training at t=1 was clean (569d t=1 measurements stable) but
iterated multi-step rollouts blew up to 1e16+ magnitudes on most ON-arm
seeds at P1=200-step episodes. The amend (ree-v3/CLAUDE.md "SD-056 multi-
step rollout stability amend (2026-05-31)" section) fixes the multi-step
instability via Lever (a) multi-step InfoNCE objective at h=5 and Lever
(b) per-step output norm clamp at ratio=2.0. V3-EXQ-614a does not directly
read iterated E2 rollouts at large horizon, but the SD-056 substrate it
runs on is the same substrate that was unstable for 569e. The re-run
verifies that the 614a behavioural reading (PASS via C2+C3) holds on the
post-amend substrate AND tests whether the stabilised substrate now lets
C1 (R2.c MECH-341 isolation) fire.

What's different from V3-EXQ-614a body
--------------------------------------
- QUEUE_ID = V3-EXQ-614b
- EXPERIMENT_TYPE = v3_exq_614b_mech341_p3_behavioural_falsifier_3arm_sd056_amended
- SUPERSEDES = V3-EXQ-614a
- _make_agent applies four new SD-056 amend lever flags uniformly across
  ALL three arms (B_only / ablate_B / ALL_ON; the amend is held constant
  across the diversity-axis comparison so R2.c interpretation is not
  confounded):
    e2_action_contrastive_multistep_enabled=True
    e2_action_contrastive_horizon=5
    e2_rollout_output_norm_clamp_enabled=True
    e2_rollout_output_norm_clamp_ratio=2.0
- _make_agent also sets the t=1 SD-056 contrastive ON (per user-approved
  recommended-but-optional knob; defaults in the substrate are OFF, so a
  behavioural arm that wants SD-056 active needs to enable it explicitly):
    e2_action_contrastive_enabled=True
    e2_action_contrastive_weight=0.01
- claim_ids UNCHANGED (MECH-341 + ARC-065) per REE_assembly/CLAUDE.md
  claim_ids accuracy rule -- the science is identical to 614a; only the
  substrate has been amended underneath.
- env_kwargs UNCHANGED (same SD-054 reef-bipartite + hazard_food_attraction
  + harm_history substrate).
- acceptance criteria UNCHANGED (C1 R2.c MECH-341 isolation, C2 ablate_B
  necessity delta, C3 Rung-1 ALL_ON; PASS = C1 fires OR (C2 AND C3) fires).
- 4-row interpretation grid UNCHANGED. Header note for V3-EXQ-614b:
  PASS via C1 is now the load-bearing target (614a already established
  PASS via C2+C3; this re-run tests whether the amended substrate lifts
  the R2.c isolation reading). If 614b lands PASS via C2+C3 only again
  (i.e. amend did not change the C1 outcome), the result confirms the
  substrate-confirmed-but-isolation-still-not-clearing reading under the
  same R2.c-isolation-deferred routing as 614a.

Cross-plan rationale: the same Layer-B substrate stabilisation unblocks
arc_062_rule_apprehension:GAP-B (V3-EXQ-543l successor cohort) under the
shared SD-056-amended substrate. The 614b PASS verifies the MECH-341
behavioural reading holds under amend; subsequent ARC-062 GAP-B work on
the same substrate then runs against a known-stable Layer-B baseline.

Original V3-EXQ-614 / 614a charter (unchanged below)
----------------------------------------------------

Behavioural test of MECH-341 under the R2.c decision rule from
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md.

V3-EXQ-608 P2 PASS 2026-05-26 confirmed R2.a (E3 scoring collapses
candidate-pool diversity at large score gaps), routing the plan to
MECH-341 substrate work. The substrate landed 2026-05-27 + the call-site
retune landed 2026-05-28; the retune-readiness diagnostic V3-EXQ-611c is
gated as the substrate-side validation. With the substrate readiness path
in flight, the plan's NEXT step (2) is THIS behavioural falsifier:
isolate each layer of the 4-substrate stack and test whether MECH-341 in
isolation drives observable action-class diversity.

Plan-of-record (behavioral_diversity_isolation_plan.md sections 4 + 5 +
6 Phase P3): test whether MECH-341 alone (B_only arm) produces Rung-1
behavioural diversity AND whether dropping MECH-341 from the full stack
(ablate_B arm) collapses it.

Arms (3, all on the same SD-054 bipartite reef env as V3-EXQ-611/611c):
  ARM_0_B_only:     SP-CEM OFF, MECH-341 ON, MECH-313 OFF, V_s OFF
  ARM_1_ablate_B:   SP-CEM ON,  MECH-341 OFF, MECH-313 ON,  V_s ON
  ARM_2_ALL_ON:     SP-CEM ON,  MECH-341 ON,  MECH-313 ON,  V_s ON

V_s scope per user-confirmed 2026-05-29 design choice: minimal stack =
use_per_stream_vs + use_vs_rollout_gating only. NO anchor sets, NO
staleness accumulator, NO event segmenter, NO invalidation trigger. The
gate substitutes a snapshot at the E1 rollout call site when V_s drops
below threshold; that is the minimal behaviorally consequential D-stack
(per MECH-269b CLAUDE.md section; gate raises ValueError without
use_per_stream_vs at agent build time).

MECH-341 sub-flavours: BOTH (entropy_bonus + stratified_select) ON at
entropy_bias_scale=2.0 (the scale value selected for the V3-EXQ-611b
retune sweep target).

Pre-registered acceptance criteria (per R2.c + Rung 1)
------------------------------------------------------
  C1 (R2.c MECH-341 in isolation): ARM_0_B_only produces
     n_unique_selected_classes >= 2 AND selected_class_entropy_nats > 0.3
     on majority (>= 2/3) of seeds. Direct test of MECH-341 provisional
     promotion candidate.
  C2 (B necessity in the stack): ARM_2_ALL_ON mean_selected_class_entropy
     - ARM_1_ablate_B mean_selected_class_entropy >= 0.1 (entropy drops
     when B is removed from the full stack).
  C3 (Rung-1 ALL_ON works): ARM_2_ALL_ON produces n_unique_selected_classes
     >= 2 AND selected_class_entropy_nats > 0.3 on majority (>= 2/3) of
     seeds.

Overall outcome:
  PASS = C1 fires (R2.c provisional promotion candidate) OR (C2 fires AND
         C3 fires) (B is load-bearing in the stack at Rung 1).
  FAIL = neither path.

Interpretation grid (FOUR rows):
  PASS via C1 only:
    MECH-341 R2.c provisional promotion candidate. ARC-065 mixed: B can
    drive diversity in isolation without A's SP-CEM proposal lift -- the
    distributed-pathway commitment is partially decoupled from the
    proposer-layer commitment. Route to /governance for MECH-341
    provisional promotion + R_X.b A-vs-B partial-redundancy follow-up.

  PASS via C1 + C2 + C3:
    MECH-341 supports + load-bearing. ARC-065 supports (B is a substrate
    of the ARC-065 pathway). Route to /governance for MECH-341 promotion
    + Phase P4 full 11-arm matrix on downstream env.

  PASS via C2 + C3 only (C1 false):
    MECH-341 contributes to the full stack but is not sufficient in
    isolation. ARC-065 supports (distributed pathway necessary -- the
    architectural commitment to multi-substrate diversity generation
    holds). Route to /governance with MECH-341 supports (load-bearing)
    + propagate the "B alone insufficient" finding to Q-054 calibration
    (entropy_bias_scale sweep) before Phase P4.

  FAIL (no criterion fires):
    MECH-341 weakens at the behavioural layer. ARC-065 weakens at the B
    layer specifically (B does not contribute to ARC-065's distributed
    pathway). Route to /diagnose-errors on the e3_selector + MECH-341
    integration; if substrate is firing per V3-EXQ-611c PASS, the
    candidate-pool diversity may be too imbalanced at scale to surface
    selected-class diversity. Pre-register Q-054 entropy_bias_scale
    sweep + revisit the stratified-temperature default.

Claims: [MECH-341, ARC-065].
  MECH-341 is the load-bearing claim per R2.c.
  ARC-065 is the parent distributed-diversity-generation architecture --
  this experiment tests whether B alone (without A=SP-CEM) reproduces the
  ARC-065-style observable diversity (the architectural-necessity check).

experiment_purpose=evidence (weighted in confidence/conflict scoring per
Phase-3 governance rules).

Multi-claim direction (MANDATORY per skill rule):
  evidence_direction (overall summary) reads off the interpretation grid
  row that fires:
    PASS via C1 only      -> overall=mixed; MECH-341 supports; ARC-065 mixed.
    PASS via C1+C2+C3     -> overall=supports; MECH-341 supports; ARC-065 supports.
    PASS via C2+C3 only   -> overall=supports; MECH-341 supports; ARC-065 supports.
    FAIL                  -> overall=weakens; MECH-341 weakens; ARC-065 weakens.

Phases
------
P0 (30 ep, instrumentation OFF): encoder warmup.
P1 (60 ep, instrumentation ON): behavioural measurement window. Longer
than the V3-EXQ-611/611c P1=20 budget because this is a behavioural
falsifier; selected-action-class entropy needs sufficient samples to
distinguish from noise floor.

Budget: 3 arms x 3 seeds x 90 ep x 200 steps = 162k steps total.
Estimated ~50-60 min on Mac (DLAPTOP-4.local @ ~14 steps/sec).

Implementation notes
--------------------
- env_kwargs IDENTICAL to V3-EXQ-611/611b/611c -- same SD-054 reef-
  bipartite + hazard_food_attraction + harm_history substrate. Direct
  manifest comparability with the 611 ARM_0_ALL_OFF baseline and the
  611c retune-readiness manifest.
- Each arm is an independent REEAgent build (no in-process flag toggling
  mid-run); _make_agent reads the per-arm substrate-state dict to
  construct REEConfig.from_dims with the right per-axis kwargs.
- Per-tick measurement reuses the V3-EXQ-611 helpers verbatim
  (_per_class_score_stats, _entropy_from_counts) so the manifest's
  metric semantics are identical to the existing P2/P3 lineage.

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md,
REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md,
REE_assembly/docs/architecture/sd_054_reef_enrichment_substrate.md,
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md,
arc_062_rule_apprehension_plan.md (cross-link GAP-B).
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_614c_mech341_stratified_within_class_temperature_sweep"
QUEUE_ID = "V3-EXQ-614c"
SUPERSEDES = "V3-EXQ-614b"
CLAIM_IDS: List[str] = ["MECH-341"]
EXPERIMENT_PURPOSE = "evidence"

# V3-EXQ-614c sweep axis: within-class temperature in {None, 0.5, 1.0, 2.0}.
# None = legacy argmin within-class (bit-identical OFF; regression guard).
# Set on E3ScoreDiversityConfig.stratified_within_class_temperature via
# REEConfig.e3_diversity_stratified_within_class_temperature. See MECH-341
# amend 2026-06-01 (ree-v3/CLAUDE.md + design doc + substrate_queue
# amend_history).
WITHIN_CLASS_T_BY_ARM: Dict[str, Optional[float]] = {
    "ARM_0_LEGACY": None,
    "ARM_1_T_0_5": 0.5,
    "ARM_2_T_1_0": 1.0,
    "ARM_3_T_2_0": 2.0,
}

# V3-EXQ-614b ARM_2 ALL_ON reference (regression guard target for C1).
EXQ_614B_ALL_ON_ENTROPY_NATS = 0.800
C1_REGRESSION_FRACTION = 0.10   # +/- 10% of the reference value
C1_REGRESSION_LOWER = EXQ_614B_ALL_ON_ENTROPY_NATS * (1.0 - C1_REGRESSION_FRACTION)
C1_REGRESSION_UPPER = EXQ_614B_ALL_ON_ENTROPY_NATS * (1.0 + C1_REGRESSION_FRACTION)

# C2 within-class lift: at least one of T in {0.5, 1.0, 2.0} produces
# mean_selected_class_entropy_nats >= 614b ARM_2 ALL_ON value (0.800 nats)
# on majority of seeds.
C2_LIFT_TARGET_NATS = EXQ_614B_ALL_ON_ENTROPY_NATS

# SD-056 amend lever defaults applied uniformly across all 3 arms
# (multi-step contrastive horizon + per-step output norm clamp ratio).
# See ree-v3/CLAUDE.md "SD-056 multi-step rollout stability amend (2026-05-31)".
SD056_MULTISTEP_CONTRASTIVE = True
SD056_CONTRASTIVE_HORIZON = 5
SD056_OUTPUT_NORM_CLAMP = True
SD056_OUTPUT_NORM_CLAMP_RATIO = 2.0
SD056_T1_CONTRASTIVE_ENABLED = True
SD056_T1_CONTRASTIVE_WEIGHT = 0.01

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 30
P1_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Pre-registered behavioural thresholds (R2.c + Rung 1).
RUNG1_ENTROPY_THRESHOLD = 0.3
RUNG1_MIN_CLASSES = 2
NECESSITY_ENTROPY_DELTA = 0.1
MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3

# Pre-registered measurement gates (preserved from V3-EXQ-611 lineage so
# the P1 per-tick semantics remain comparable across the cluster).
PRE_GE2_FRAC_GATE = 0.5
SCORE_GAP_EPSILON_RANGE_FRAC = 0.05

# MECH-341 sub-flavour scale used in the entropy-ON arms. 2.0 was the
# upper-range value selected for the V3-EXQ-611b retune sweep target.
MECH341_ENTROPY_BIAS_SCALE = 2.0

# V_s (D) thresholds (minimal stack). 0.5 / 0.4 match the
# V3-EXQ-601 MECH-269b substrate-readiness PASS defaults.
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4


# IDENTICAL to V3-EXQ-611 / 611b / 611c for direct manifest comparability
# across the cluster.
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


# ---------------------------------------------------------------------------
# Arm definitions: each arm is a full specification of the 4-substrate
# isolation axes. _make_agent reads these directly into REEConfig.from_dims
# kwargs (substrate flags do not all live at the top-level config field; SP-CEM
# and V_s live under config.hippocampal, so passing through from_dims is the
# right route).
# ---------------------------------------------------------------------------


ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_LEGACY",
        "label": "within_class_legacy_argmin",
        "within_class_temperature": None,
        # All four substrate axes ON for the full SD-056-amended baseline.
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
    {
        "arm_id": "ARM_1_T_0_5",
        "label": "within_class_T_0_5_sharpened",
        "within_class_temperature": 0.5,
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
    {
        "arm_id": "ARM_2_T_1_0",
        "label": "within_class_T_1_0_mid",
        "within_class_temperature": 1.0,
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
    {
        "arm_id": "ARM_3_T_2_0",
        "label": "within_class_T_2_0_flatter",
        "within_class_temperature": 2.0,
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(
    env: CausalGridWorldV2,
    axes: Dict[str, bool],
    within_class_temperature: Optional[float] = None,
) -> REEAgent:
    """Build a REEAgent with the 4-substrate state specified by ``axes``.

    Axis -> config flags mapping (read off ree-v3/CLAUDE.md SD Design
    Decisions Implemented sections + ree-v3/ree_core/utils/config.py):

      A_sp_cem (SP-CEM main-path):
        use_support_preserving_cem
        support_preserving_stratified_elites
        support_preserving_ao_std_floor (0.2 when ON, 0.0 when OFF)
      B_mech341 (E3 score diversity preservation):
        use_e3_score_diversity (master)
        use_e3_diversity_entropy_bonus + use_e3_diversity_stratified_select
          (both ON when B ON; both irrelevant when master OFF)
        e3_diversity_entropy_bias_scale = MECH341_ENTROPY_BIAS_SCALE
      C_noise_floor (MECH-313 tonic noise floor):
        use_noise_floor + noise_floor_alpha (default 0.1)
      D_vs (minimal V_s pathology stack per user-confirmed scope):
        use_per_stream_vs + use_vs_rollout_gating
        vs_gate_snapshot_refresh_threshold (0.5) + vs_gate_e1_threshold (0.4)
        NO anchor_sets, NO staleness_accumulator, NO event_segmenter,
        NO invalidation_trigger. The gate hold-substitutes a fresh
        snapshot at the E1 rollout call site when V_s falls below
        the e1 threshold (MECH-269b minimal behavioural mechanism).

    SD-056 amend (NEW IN 614b vs 614a): applied uniformly across ALL arms
    so the amend is held constant on the diversity-axis comparison (R2.c
    interpretation is not confounded by amend on/off being correlated with
    A/B/C/D axis state). Five SD-056 flags:
      e2_action_contrastive_multistep_enabled=True  (Lever (a))
      e2_action_contrastive_horizon=5               (Dreamer default; calibratable)
      e2_rollout_output_norm_clamp_enabled=True     (Lever (b) defensive)
      e2_rollout_output_norm_clamp_ratio=2.0        (per autopsy acceptance bound)
      e2_action_contrastive_enabled=True            (t=1 contrastive ON)
      e2_action_contrastive_weight=0.01             (substrate default)
    """
    a_on = bool(axes["A_sp_cem"])
    b_on = bool(axes["B_mech341"])
    c_on = bool(axes["C_noise_floor"])
    d_on = bool(axes["D_vs"])

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
        # A (SP-CEM)
        use_support_preserving_cem=a_on,
        support_preserving_stratified_elites=a_on,
        support_preserving_ao_std_floor=(0.2 if a_on else 0.0),
        support_preserving_min_first_action_classes=2,
        # B (MECH-341)
        use_e3_score_diversity=b_on,
        use_e3_diversity_entropy_bonus=True,    # consulted only when master ON
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        # MECH-341 amend 2026-06-01 sweep axis: within-class proportional
        # sampling temperature. None = legacy argmin within-class
        # (bit-identical to pre-amend, V3-EXQ-614c ARM_0_LEGACY regression
        # guard). Positive float = sample within each first-action class via
        # softmax(-class_scores / T) before the existing across-class
        # softmax step.
        e3_diversity_stratified_within_class_temperature=within_class_temperature,
        # C (MECH-313)
        use_noise_floor=c_on,
        noise_floor_alpha=0.1,
        # D (V_s minimal)
        use_per_stream_vs=d_on,
        use_vs_rollout_gating=d_on,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # SD-056 amend (uniform across ALL arms; not an isolation axis).
        # Lever (a) multi-step contrastive + Lever (b) per-step output
        # norm clamp + t=1 contrastive ON (substrate defaults are OFF).
        e2_action_contrastive_enabled=SD056_T1_CONTRASTIVE_ENABLED,
        e2_action_contrastive_weight=SD056_T1_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (mirror EXQ-608 / EXQ-611 conventions)
# ---------------------------------------------------------------------------


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
    agent = _make_agent(
        env,
        arm["substrate_axes"],
        within_class_temperature=arm.get("within_class_temperature"),
    )
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

    # Per-seed pre-registered acceptance:
    seed_passes_rung1 = bool(
        n_selected_classes >= RUNG1_MIN_CLASSES
        and selected_class_entropy > RUNG1_ENTROPY_THRESHOLD
        and frac_pre_ge2 >= PRE_GE2_FRAC_GATE
    )

    return {
        "arm_id": arm["arm_id"],
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
    c1: bool, c2: bool, c3: bool
) -> Tuple[str, str, str]:
    """V3-EXQ-614c interpretation grid.

    Map (c1, c2, c3) -> (outcome, mech341_direction, interpretation_label).

    C1: regression guard. ARM_0_LEGACY mean entropy in
        [0.720, 0.880] (614b ARM_2 ALL_ON 0.800 +/- 10%) on majority of seeds.
    C2: within-class lift. At least one of ARM_1/2/3 has mean entropy
        >= 0.800 nats on majority of seeds.
    C3: substrate-readiness. All 4 arms have frac_pre_ge2 > 0.3 on
        majority of seeds.

    PASS = C1 AND (C2 OR C3).
    """
    if c1 and c2:
        return (
            "PASS", "supports",
            "PASS_C1_C2_within_class_lever_validated",
        )
    if c1 and (not c2) and c3:
        return (
            "PASS", "mixed",
            "PASS_C1_C3_only_lever_functional_no_marginal_lift",
        )
    if not c1:
        return (
            "FAIL", "weakens",
            "FAIL_C1_regression_against_614b_routes_to_amend_autopsy",
        )
    # c1=True, c2=False, c3=False
    return (
        "FAIL", "weakens",
        "FAIL_neither_c2_nor_c3_within_class_lever_non_load_bearing",
    )


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arms_out: List[Dict[str, Any]] = []
    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) "
            f"(P0={p0_episodes} ep, P1={p1_episodes} ep, "
            f"steps_per_episode={steps_per_episode}, dry_run={dry_run})",
            flush=True,
        )
        seed_rows: List[Dict[str, Any]] = []
        for s in seeds:
            print(
                f"Seed {s} Condition {arm['label']}", flush=True,
            )
            row = _run_seed_arm(arm, s, p0_episodes, p1_episodes, steps_per_episode)
            seed_rows.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)
        cross = _interpret_arm(seed_rows)
        arms_out.append({
            "arm_id": arm["arm_id"],
            "label": arm["label"],
            "substrate_axes": arm["substrate_axes"],
            "per_seed_results": seed_rows,
            "cross_seed_interpretation": cross,
        })

    # ----- Cross-arm acceptance criteria (V3-EXQ-614c within-class sweep) -----
    by_id = {a["arm_id"]: a for a in arms_out}
    arm_legacy = by_id["ARM_0_LEGACY"]
    swept_arms = [by_id["ARM_1_T_0_5"], by_id["ARM_2_T_1_0"], by_id["ARM_3_T_2_0"]]

    # C1 (regression guard): ARM_0_LEGACY mean_selected_class_entropy_nats
    # within +/- 10% of V3-EXQ-614b ARM_2 ALL_ON 0.800 nats reference,
    # checked on majority (>= 2/3) of seeds.
    legacy_seed_entropies = [
        r["selected_class_entropy_nats"]
        for r in arm_legacy["per_seed_results"]
        if r["error_note"] is None
    ]
    legacy_seeds_in_band = sum(
        1 for e in legacy_seed_entropies
        if C1_REGRESSION_LOWER <= e <= C1_REGRESSION_UPPER
    )
    c1_holds = bool(legacy_seeds_in_band >= MIN_SEEDS_PER_ARM_FOR_PASS)

    # C2 (within-class lift): at least one of ARM_1 / ARM_2 / ARM_3
    # produces mean entropy >= 614b ARM_2 ALL_ON value (0.800 nats) on
    # majority of seeds.
    def _arm_seed_lift_count(arm):
        return sum(
            1 for r in arm["per_seed_results"]
            if r["error_note"] is None
            and r["selected_class_entropy_nats"] >= C2_LIFT_TARGET_NATS
        )
    c2_per_arm_lift = {a["arm_id"]: _arm_seed_lift_count(a) for a in swept_arms}
    c2_holds = any(
        n >= MIN_SEEDS_PER_ARM_FOR_PASS for n in c2_per_arm_lift.values()
    )

    # C3 (substrate-readiness): all 4 arms have frac_pre_ge2 > 0.3 on
    # majority of seeds.
    def _arm_seed_pre_ge2_count(arm):
        return sum(
            1 for r in arm["per_seed_results"]
            if r["error_note"] is None and r["frac_pre_ge2"] > 0.3
        )
    c3_per_arm_pre_ge2 = {a["arm_id"]: _arm_seed_pre_ge2_count(a) for a in arms_out}
    c3_holds = all(
        n >= MIN_SEEDS_PER_ARM_FOR_PASS for n in c3_per_arm_pre_ge2.values()
    )

    (
        outcome_label, mech341_direction, interpretation_label,
    ) = _classify_outcome(c1_holds, c2_holds, c3_holds)
    overall_direction = mech341_direction
    arc065_direction = "non_contributory"   # not under direct test in 614c

    total_seeds = len(ARMS) * len(seeds)
    total_completed = sum(
        a["cross_seed_interpretation"]["n_seeds_completed"] for a in arms_out
    )

    return {
        "outcome": outcome_label,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": {
            "MECH-341": mech341_direction,
        },
        "interpretation_label": interpretation_label,
        "seeds": seeds,
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
            "mech341_entropy_bias_scale": float(MECH341_ENTROPY_BIAS_SCALE),
            "vs_snapshot_refresh_threshold": float(VS_SNAPSHOT_REFRESH_THRESHOLD),
            "vs_e1_threshold": float(VS_E1_THRESHOLD),
            "exq_614b_all_on_reference_nats": float(EXQ_614B_ALL_ON_ENTROPY_NATS),
            "c1_regression_band_lower": float(C1_REGRESSION_LOWER),
            "c1_regression_band_upper": float(C1_REGRESSION_UPPER),
            "c2_lift_target_nats": float(C2_LIFT_TARGET_NATS),
        },
        "acceptance_criteria": {
            "C1_legacy_regression_band": c1_holds,
            "C1_legacy_seeds_in_band": int(legacy_seeds_in_band),
            "C1_legacy_per_seed_entropies": [
                round(e, 6) for e in legacy_seed_entropies
            ],
            "C2_within_class_lift": c2_holds,
            "C2_per_arm_lift_seed_counts": {k: int(v) for k, v in c2_per_arm_lift.items()},
            "C3_substrate_readiness": c3_holds,
            "C3_per_arm_pre_ge2_seed_counts": {k: int(v) for k, v in c3_per_arm_pre_ge2.items()},
        },
        "interpretation_grid_header_note_614c": (
            "V3-EXQ-614c context: 614b FAILed C1 R2.c (B_only structural "
            "CEM-proposer collapse without SP-CEM Layer A) AND C2 necessity "
            "0.087 below pre-amend 0.1 threshold; C3 ALL_ON Rung-1 PASSed "
            "at 0.800 nats (highest of any 614-lineage run; positive "
            "substrate-readiness for SD-056 amend at behavioural-runtime "
            "horizon). 616 autopsy Sections 7 + 10 named the contingent "
            "amend: stratified_temperature default + A-vs-B partial-"
            "redundancy probe. This sweep validates Part (a) of the resulting "
            "MECH-341 amend by varying ONLY the within-class temperature "
            "lever {None, 0.5, 1.0, 2.0} on the full SD-056-amended baseline "
            "(all four substrate axes ON; same config as 614b ARM_2 ALL_ON). "
            "C1 is a regression guard against the 614b ARM_2 value; C2 "
            "tests whether within-class proportional sampling adds marginal "
            "diversity over legacy argmin within-class; C3 confirms upstream "
            "substrate is firing across all 4 arms."
        ),
        "interpretation_grid": {
            "PASS_C1_C2": (
                "Within-class proportional sampling lever validated. MECH-341 "
                "within-class sub-axis is load-bearing under SD-056-amended "
                "substrate. Route to /governance for MECH-341 supports + "
                "v3_pending clearance candidate. Within-class temperature "
                "winner identified for downstream wiring decisions; flag "
                "for review whether default should be flipped from None "
                "(governance ratification gate; do NOT auto-flip)."
            ),
            "PASS_C1_C3_only": (
                "Amend is functional but not load-bearing under SD-056-"
                "amended substrate. Layer B may be redundant with Layer A "
                "under SP-CEM-ON. Route to /governance with MECH-341 mixed "
                "(substrate firing but within-class lever produces no "
                "marginal improvement over legacy argmin) + propagate to "
                "Q-054 collapse-with-ARC-065 question. Default stays None."
            ),
            "FAIL_C1_regression": (
                "Amend broke the legacy bit-identical OFF guarantee OR "
                "upstream substrate drifted between 614b run and 614c run "
                "(SD-056 amend config drift, scaffolded_sd054_onboarding "
                "default change, etc.). Route to /failure-autopsy on the "
                "amend itself -- the only path the contract suite cannot "
                "catch (contracts verify per-call output determinism but "
                "not end-to-end environment-level behaviour)."
            ),
            "FAIL_neither_c2_nor_c3": (
                "Within-class lever functional but produces no measurable "
                "lift over legacy AND substrate is not surfacing diversity. "
                "Substrate-redesign signal. Route to /failure-autopsy on "
                "the combined MECH-341 + SP-CEM stack -- the score-layer "
                "preserver may need a different structural mechanism than "
                "within-class proportional sampling can deliver."
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
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "evidence_direction_note": (
            f"experiment_purpose=evidence; V3-EXQ-614c MECH-341 amend "
            f"within-class temperature sweep on SD-056-amended baseline; "
            f"routed by failure_autopsy_V3-EXQ-616_2026-05-31 Sections 7 + 10 "
            f"contingent-on-614b-FAIL-C1 path. "
            f"interpretation_label={result['interpretation_label']}. "
            f"C1 (legacy regression band)={result['acceptance_criteria']['C1_legacy_regression_band']}, "
            f"C2 (within-class lift)={result['acceptance_criteria']['C2_within_class_lift']}, "
            f"C3 (substrate-readiness)={result['acceptance_criteria']['C3_substrate_readiness']}. "
            f"Per-claim direction follows the interpretation grid; see "
            f"manifest.result.interpretation_grid for the full mapping."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "vs_stack": "minimal (use_per_stream_vs + use_vs_rollout_gating)",
            "mech341_sub_flavours": "both (entropy_bonus + stratified_select)",
            "mech341_entropy_bias_scale": MECH341_ENTROPY_BIAS_SCALE,
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "sd056_t1_contrastive_enabled": SD056_T1_CONTRASTIVE_ENABLED,
            "sd056_t1_contrastive_weight": SD056_T1_CONTRASTIVE_WEIGHT,
            "sd056_multistep_contrastive": SD056_MULTISTEP_CONTRASTIVE,
            "sd056_contrastive_horizon": SD056_CONTRASTIVE_HORIZON,
            "sd056_output_norm_clamp": SD056_OUTPUT_NORM_CLAMP,
            "sd056_output_norm_clamp_ratio": SD056_OUTPUT_NORM_CLAMP_RATIO,
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
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"C1_legacy_regression_band={result['acceptance_criteria']['C1_legacy_regression_band']} "
        f"C2_within_class_lift={result['acceptance_criteria']['C2_within_class_lift']} "
        f"C3_substrate_readiness={result['acceptance_criteria']['C3_substrate_readiness']} "
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
