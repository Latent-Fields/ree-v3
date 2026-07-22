"""V3-EXQ-791a: channel-routing route_range MAGNITUDE cross-machine-class
replication (metrology diagnostic) -- SAME-QUESTION RE-RUN OF V3-EXQ-790,
GATE FIX ONLY.

SUPERSEDES
  v3_exq_790_channel_routing_cross_class_magnitude_replication_20260722T021558Z_v3

WHY THIS RE-RUN EXISTS (failure_autopsy_V3-EXQ-790_2026-07-22.md, confirmed +
user-adjudicated 2026-07-22: "Gate defect -- uphold the science"). The 790 run
PASSED BOTH SCIENCE CRITERIA -- routing active in 10/10 ARM_1_ROUTE_ON seeds and
inactive in 10/10 ARM_0_NO_ROUTE seeds (C1); committed-class TV above floor on
8/10 seeds against a bar of 7 (C2); routed bias range supra-floor at 0.334 vs a
0.01 floor and bounded at 1.644 vs a 1e6 ceiling -- and was nonetheless recorded
FAIL / substrate_not_ready_requeue / precondition_unmet on ONE failed
precondition: adequate_fresh_selection_sample, worst cell 53 vs a floor of 200,
at ARM_0_NO_ROUTE::seed49. That is one seed of the CONTROL arm, whose
route_range is structurally 0.0 by construction and from which no C1/C2
statistic is estimated. NOTHING about the substrate or the design changes here.
The DESIGN, ENV, ARMS, SEEDS, SCHEDULE and SCIENCE THRESHOLDS are byte-for-byte
790's. Only the readiness GATE changes, in the two ways the autopsy names.

DEFECT 1 -- THE GATE WAS ARM-BLIND WHERE IT MUST BE ARM-SCOPED.
790 computed `all(n_fresh_select >= FLOOR for r in arm_results)` across BOTH
arms and flattened every cell into interpretation.preconditions. The REE_assembly
indexer's _compute_adjudication reads that flat list ARM-BLIND and returns
whole-run precondition_unmet on the FIRST unmet entry, so one starved control
cell vacated a 10/10 result in the arm that was actually measured. The driver is
the ONLY place this can be fixed. Fixed here by adopting the pattern V3-EXQ-794
(per_arm_gate green/red + preconditions_scope_note) and V3-EXQ-737
(non-gating guards moved to recorded_preconditions) already use, via the shared
experiments/_lib/precondition_gate.py:
  - the sample gates apply ONLY to ARM_1_ROUTE_ON, the arm whose statistics are
    estimated from the selection sample (PreconditionSpec.applies_to);
  - ARM_0_NO_ROUTE keeps one gate that IS meaningful for it -- routing
    inactivity must be OBSERVABLE, i.e. every control cell recorded at least one
    genuine selection -- because that, and only that, is what C1's off-inactive
    leg reads from the control arm;
  - the control arm's full fresh-selection counts are carried as NON-GATING
    `recorded_preconditions` (the 737 pattern) so nothing is hidden;
  - interpretation.preconditions carries green-arm entries only, so a scored-out
    cell can never re-vacate a green arm at adjudication time.
A red arm never vacates a green one (failure_autopsy_V3-EXQ-785 sections 2a/8).

DEFECT 2 -- THE 200-SELECTION FLOOR WAS MIS-DERIVED.
790 derived 200 from the NOMINAL default cadence: 3600 window ticks at
heartbeat.e3_steps_per_tick=10 gives ~360 selections. But the LIVE cadence is
MECH-093-modulated over ree_core/heartbeat/clock.py's
beta_rate_min_steps=5 .. beta_rate_max_steps=20, so the same window yields
180..720 selections depending on arousal. A floor of 200 sits ABOVE the
substrate's own worst case and will fire on healthy runs indefinitely. Two
replacements, both derived from the modulated range rather than the default knob:
  (a) FRESH_SELECT_FLOOR = nominal_window_ticks / beta_rate_max_steps
      = 3600 / 20 = 180  -- the modulated range's WORST case, exactly as the
      autopsy prescribes, replacing the 200 keyed to e3_steps_per_tick=10.
  (b) FRESH_SELECT_YIELD_FLOOR = 1 / beta_rate_max_steps = 0.05 -- the same
      bound expressed as a FRACTION of the cell's OBSERVED window ticks, which
      is truncation-invariant. This is the load-bearing half: 790's per-cell
      windows ranged 330..3600 ticks because episodes terminate early on `done`,
      so an ABSOLUTE count conflates "the selection loop under-fired" (a real
      readiness failure) with "this cell simply had a shorter window" (an env
      fact, not a substrate fact). Yield separates them. On 790's numbers every
      cell yielded 0.100..0.181, i.e. an effective cadence of 5.5..10 steps per
      selection -- consistently at or FASTER than nominal and never near the
      slow end, so the autopsy's "sat at the slow end" reading is superseded by
      the measured cause: WINDOW TRUNCATION, which (b) is immune to.
  QUANTIFIER ALIGNMENT (a): `met` for the absolute floor is a COUNT quantifier
  matching the criterion it guards -- the load-bearing readiness criterion needs
  >= MIN_SEEDS_FOR_PASS (7) of 10 ARM_1 seeds above the route floor, so the
  sample gate is starved only when FEWER THAN 7 ARM_1 seeds have an adequate
  sample. `measured` is therefore the MIN_SEEDS_FOR_PASS-th largest per-seed
  n_fresh_select (an order statistic, recomputable against the count), the
  identical shape arm1_routed_bias_range_supra_floor already used in 790 and
  which the sample gate simply failed to adopt. This is quantifier alignment,
  NOT a lowered bar: a 7-of-10 criterion is not starved by 3 short seeds.

WHAT IS *NOT* WEAKENED. The pseudo-replication protection the gate exists to
provide is UNCHANGED and remains load-bearing. The corrected denominator --
agent.e3.last_score_diagnostics cleared immediately before every
select_action(), so a latched tick contributes NO row -- is byte-identical to
790's and closes the ~9x defect the shared 662/663 driver carried. Both
n_fresh_select and n_latched_ticks are still emitted per cell so the true
denominator stays auditable, and the yield gate (b) is a STRICTLY STRONGER
latch-pathology detector than the absolute count it supplements, because a
latch-inflated or dead selection loop is visible in the ratio at any window
length. No threshold protecting against pseudo-replication is lowered.

ONE SCRIPT, ONE QUEUE ENTRY. 790/791 shared this script across two machine-class
pins; 791a re-confirms the readiness finding under a correctly-scoped gate and
runs on a cloud worker (790 ran darwin-arm64). QUEUE_ID is still read from the
runner's REE_QUEUE_ID env var so the run self-labels.

DV-SYMMETRY DECLARATION (skill Step 3, MANDATORY, per arm). The DV is
modulatory_channel_route_range, a max-minus-min CROSS-CANDIDATE RANGE. Its
symmetry group is (i) a uniform additive constant broadcast across candidates
and (ii) any permutation of the candidate set.
  ARM_1_ROUTE_ON  -- the manipulation is use_modulatory_channel_routing=True,
    which projects the PER-CANDIDATE cand_world_summary through
    project_channel_range into a per-candidate bias. It is per-candidate, not a
    broadcast scalar, so it is NOT invariant under (i); and it is a function of
    each candidate's own summary rather than of the pooled set, so it is NOT
    invariant under (ii). The delta is a measurement.
  ARM_0_NO_ROUTE  -- the manipulation is the ABSENCE of routing, which yields
    route_range identically 0.0 by construction. This arm is a structural
    control, not a measured contrast, which is precisely why its selection
    sample gates only the OBSERVABILITY of that zero (>= 1 genuine selection per
    cell) and not the sample size for an estimate that is not being made.
The C2 DV (committed_class_counts) is a selected-action histogram and IS
rank-based, so it would be invariant under a broadcast constant -- but the
routed bias is per-candidate as argued above, so C2 is a live readout too. C2
is secondary and not load-bearing either way.

THE QUESTION (metrology, NOT a claim test -- claim_ids=[]): the 2026-07-19
machine-class divergence root-cause (torch.multinomial returns different
categories on linux-x86_64/torch 2.5.1 vs darwin-arm64/torch 2.10.0 from a
bit-identical probability tensor at the same seed) concluded that no completed
experiment's CONCLUSION is at risk, because machine_class is inside the
arm-fingerprint hash and every queue item runs all its arms on one machine --
so the class offset is common-mode WITHIN each contrast and cancels.

That common-mode-cancellation is an ASSUMPTION, and V3-EXQ-662 (ree-cloud-3)
vs V3-EXQ-663 (DLAPTOP-4) is the one natural cross-class replication available
to test it. Both PASSed, route_active_frac 1.0 vs 0.0 in both, OFF arms
bit-identical at exactly 0.0 -- the CONCLUSION replicated. But ARM_1
route_range_mean was HIGHER ON THE MAC IN 3/3 SEEDS:
    seed 42  cloud 0.486815 | mac 0.545524
    seed 43  cloud 0.581449 | mac 0.726480
    seed 44  cloud 0.267550 | mac 0.333261
At n=3 a 3/3 direction is not distinguishable from chance (sign test p=0.125),
so the MAGNITUDE could not be asserted common-mode-cancelling either way.

TWO CONFOUNDS IN THAT 3-SEED PICTURE, both fixed here:
 (1) PSEUDO-REPLICATION. The shared 662/663 driver read
     agent.e3.last_score_diagnostics once per ENV STEP without clearing the
     latch. E3 populates those diagnostics only inside select(), which runs on
     ~1 tick in heartbeat.e3_steps_per_tick (default 10), so every held tick
     re-recorded the PREVIOUS selection as a fresh independent observation.
     route_range_mean was a mean over ~9x-latched repeats: the point estimate
     survives (uniform repetition is ~weight-preserving; measured +0.01% on a
     matched replay) but the effective n -- and therefore ANY variance or
     significance statement -- was inflated ~9-fold. Fixed here and in the 663
     driver via clear-before-select; n_fresh_select / n_latched_ticks make the
     true denominator auditable.
 (2) UNEVEN TRUNCATION. The 662/663 cells did not all reach the nominal 3600
     window ticks (662 ARM_1 seed 44: 1771; 663 ARM_1 seed 44: 2922), so the
     two classes averaged over different amounts of training. Per-cell
     n_fresh_select is recorded here so the analysis can condition on it.

DESIGN: identical to 663 (the 662 script was a byte-identical pre-rename
duplicate -- verified against git 522662d -- so the 3-seed comparison is
code-clean and only the class differed), with SEEDS widened 3 -> 10. The
original seeds 42/43/44 are NESTED at the head of the seed list so the prior
observation is directly recoverable as a subset. Same 2-arm ablation, same
env, same pre-registered route thresholds.
  ARM_0_NO_ROUTE   use_modulatory_channel_routing=False
  ARM_1_ROUTE_ON   use_modulatory_channel_routing=True, source="cand_world_summary"

WHY route_range IS THE RIGHT DV FOR A CROSS-CLASS TEST: it is SIGN-INVARIANT
(a cross-candidate range), so the known torch.linalg.svd singular-vector sign
flip across LAPACK backends cannot confound it. It is also read upstream of
the discrete quantizer, per the standing rule never to assert exact committed
ACTION streams across machine classes.

ACCEPTANCE (per-run; the CROSS-CLASS comparison is a post-hoc analysis over
the two manifests, NOT a criterion either run can evaluate alone). Science
thresholds are 790's verbatim; only the sample gate's scoping and derivation
change:
  READINESS (load-bearing, RANGE statistic, ARM_1-scoped): ARM_1
    route_range_mean > C0_ROUTE_FLOOR on >= MIN_SEEDS_FOR_PASS seeds, finite
    and below the 643a explosion ceiling.
  SAMPLE ADEQUACY (load-bearing, ARM_1-scoped -- the arm whose statistics are
    estimated from the selection sample). Two legs, both derived from the
    MECH-093-modulated cadence range, NOT from e3_steps_per_tick=10:
      absolute: >= MIN_SEEDS_FOR_PASS ARM_1 seeds cleared FRESH_SELECT_FLOOR
        (= 3600 / beta_rate_max_steps = 180) genuine E3 selections. Measured as
        the MIN_SEEDS_FOR_PASS-th largest per-seed count (order statistic
        matching the count quantifier).
      yield: EVERY ARM_1 cell's n_fresh_select / n_p1_ticks_past_window cleared
        FRESH_SELECT_YIELD_FLOOR (= 1 / beta_rate_max_steps = 0.05). Measured as
        the WORST cell (all(...) quantifier), with the offending cell named.
        Truncation-invariant, so it isolates a genuinely under-firing selection
        loop from a merely short window.
  OBSERVABILITY (ARM_0-scoped, the only sample question the control arm poses):
    every ARM_0 cell recorded >= 1 genuine selection, so "routing is inactive"
    is an OBSERVED zero rather than an absent one. The control arm's full
    fresh-selection counts are additionally carried as NON-GATING
    `recorded_preconditions`.
  C1: ARM_1 route active on >= MIN_SEEDS_FOR_PASS seeds AND ARM_0 range ~0.
  C2 (secondary, not load-bearing): committed-class TV ARM_1 vs ARM_0. NOTE its
    readout denominator is n_p1_ticks_past_window (every env step in the window,
    since the HELD action is the committed behaviour across latched ticks), NOT
    n_fresh_select -- so the fresh-selection floor does not bear on C2 at all.
  PASS = ARM_1 GATE GREEN AND C1. A red ARM_0 does not vacate ARM_1; if the
    control arm were genuinely unobservable, C1's own off-inactive leg fails on
    its own and the run self-routes route_range_inert.

WHAT THE CROSS-CLASS ANALYSIS DECIDES (both outcomes are informative):
  Mac-higher survives at 10 paired seeds  -> a SYSTEMATIC machine-class effect
    on effect SIZE. Must be recorded as a standing caveat on every cross-class
    MAGNITUDE comparison; common-mode cancellation holds for the direction of
    a within-machine contrast but NOT for its magnitude.
  Direction does not survive               -> the common-mode-cancellation
    assumption becomes empirically SUPPORTED rather than assumed, which is
    what the blast-radius assessment currently rests on.
  Either way this does NOT retroactively correct 662/663: a completed run's
    reported measurements are not rewritten. The fix is prospective.

SLEEP DRIVER: K=never (no sleep; waking action-selection diagnostic).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_791a_channel_routing_cross_class_magnitude_replication.py --dry-run
"""

import argparse
import os
import json
import math
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_791a_channel_routing_cross_class_magnitude_replication"
# The runner always exports REE_QUEUE_ID, so the run self-labels with the entry
# that claimed it.
QUEUE_ID = os.environ.get("REE_QUEUE_ID") or "V3-EXQ-791a"
# Same-question re-run of 790 with a GATE FIX ONLY (see module docstring); the
# superseded run's science stands, its readiness ADJUDICATION does not.
SUPERSEDES_RUN_ID = (
    "v3_exq_790_channel_routing_cross_class_magnitude_replication_20260722T021558Z_v3"
)
CLAIM_IDS: List[str] = []  # substrate-readiness diagnostic (gates per-claim behavioural retests)
EXPERIMENT_PURPOSE = "diagnostic"

# All three readiness anchors are reachable by construction, with the load-bearing
# one proven reachable by COMPLETED runs rather than argued:
#   arm1_routed_bias_range_supra_floor -- floor 0.01; V3-EXQ-662 (ree-cloud-3)
#     recorded ARM_1 route_range_mean 0.486815 / 0.581449 / 0.267550 and V3-EXQ-663
#     (DLAPTOP-4) 0.545524 / 0.726480 / 0.333261 on this identical design. Worst
#     observed cell clears the floor by ~27x, on BOTH machine classes.
#   adequate_fresh_selection_sample -- 180 = 3600 nominal window ticks /
#     beta_rate_max_steps=20, i.e. the MECH-093-modulated cadence range's WORST
#     case, against a MIN_SEEDS_FOR_PASS-of-10 count quantifier. Proven reachable
#     by the COMPLETED V3-EXQ-790 run of this identical design: its 7th-largest
#     ARM_1 per-seed n_fresh_select was 262. (790's 200 was keyed to the NOMINAL
#     e3_steps_per_tick=10 and sat ABOVE the substrate's own worst case -- the
#     mis-derivation this re-run corrects.)
#   fresh_selection_yield_supra_cadence_floor -- 0.05 = 1/beta_rate_max_steps,
#     the same bound as a truncation-invariant FRACTION of observed window ticks.
#     790 measured 0.100..0.181 in every one of its 20 cells, ~2x-3.6x headroom.
#   routed_range_bounded -- 1e6 explosion ceiling; the 643a stability guard.
# None is a hand-written scoring predicate that could be narrower than the state
# it anchors to, which is the failure mode the reachability check exists to catch.
ANCHOR_REACHABILITY_EXEMPT = (
    "All anchors reachable by construction, and now by MEASUREMENT: the "
    "load-bearing route-range floor (0.01) is proven reachable on BOTH machine "
    "classes by the completed V3-EXQ-662/663 runs of this identical design "
    "(worst cell 0.267550, ~27x headroom) and again by V3-EXQ-790 (10/10 ARM_1 "
    "seeds above floor). The two re-derived sample anchors are proven reachable "
    "by 790's own per-cell counts (7th-largest ARM_1 n_fresh_select 262 vs a "
    "180 floor; worst-cell yield 0.100 vs a 0.05 floor). No hand-written scoring "
    "predicate is involved."
)

# Originals 42/43/44 nested at the head so the 662/663 observation is a subset.
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (V3-EXQ-649/648a proven budget)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200
MEASURE_AFTER_TICK = 20

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_AFTER_TICK = 2

# Acceptance thresholds (pre-registered).
C0_ROUTE_FLOOR = 0.01             # readiness: ARM_1 routed-bias cross-candidate RANGE floor
C0_MAGNITUDE_CEIL = 1.0e6         # readiness: routed range bounded (643a explosion guard)
C1_OFF_INACTIVE_CEIL = 1e-9       # C1: ARM_0 routed range ~0 (routing off -> diagnostic stays 0.0)
C2_TV_FLOOR = 0.02               # C2 (secondary): committed-class distribution TV ARM_1 vs ARM_0
MIN_SEEDS_FOR_PASS = 7            # of 10

# --- Sample-adequacy gate on the CORRECTED denominator (RE-DERIVED for 791a) ---
# 790 keyed its floor to the NOMINAL default cadence: P1*(steps-measure_after)
# = 3600 window ticks / heartbeat.e3_steps_per_tick=10 -> ~360 selections, floor
# 200. But e3_steps_per_tick is MECH-093-modulated at RUN TIME over
# ree_core/heartbeat/clock.py's [beta_rate_min_steps, beta_rate_max_steps] =
# [5, 20], so the same window yields 180..720 selections depending on arousal.
# A floor of 200 therefore sat ABOVE the substrate's own worst case and fires on
# healthy runs indefinitely (failure_autopsy_V3-EXQ-790_2026-07-22 defect 2).
# Both replacements below are derived from the MODULATED range, not the knob.
BETA_RATE_MAX_STEPS = 20          # ree_core/heartbeat/clock.py default (MECH-093)
NOMINAL_WINDOW_TICKS = P1_MEASUREMENT_EPISODES * (STEPS_PER_EPISODE - MEASURE_AFTER_TICK)

# (a) ABSOLUTE floor = worst-case cadence over the nominal window (3600/20 = 180).
#     `met` is a COUNT quantifier matching the criterion it guards (the readiness
#     criterion needs >= MIN_SEEDS_FOR_PASS of 10 ARM_1 seeds), so `measured` is
#     the MIN_SEEDS_FOR_PASS-th largest per-seed count -- an order statistic that
#     recomputes exactly against that count, the same shape
#     arm1_routed_bias_range_supra_floor already uses.
FRESH_SELECT_FLOOR = NOMINAL_WINDOW_TICKS // BETA_RATE_MAX_STEPS   # 180

# (b) YIELD floor = the SAME cadence bound as a fraction of the cell's OBSERVED
#     window ticks (1/20 = 0.05), so it is invariant to window TRUNCATION.
#     Load-bearing half: 790's per-cell windows ran 330..3600 ticks because
#     episodes terminate early on `done`, so an absolute count conflates an
#     under-firing selection loop (a readiness failure) with a short window (an
#     env fact). `met` is an all(...) quantifier, so `measured` is the WORST cell.
FRESH_SELECT_YIELD_FLOOR = 1.0 / float(BETA_RATE_MAX_STEPS)        # 0.05

# (c) The control arm poses only an OBSERVABILITY question -- its route_range is
#     structurally 0.0, so "routing inactive" must be an OBSERVED zero, which
#     needs >= 1 genuine selection per cell and nothing more. Worst cell; floor 0
#     with the gate's strict `>` comparator.
ARM0_OBSERVABILITY_FLOOR = 0.0

# SD-056 online contrastive training (mirror V3-EXQ-649 harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# HARM-FREE env: SP-CEM + resources give action-divergent candidates for SD-056 to
# train z_world divergence on; no hazards needed for the routing-readiness test.
ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=0,
    num_resources=5,
    hazard_harm=0.0,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.0,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
)

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_NO_ROUTE",
        "label": "channel_routing_off_569f_washout_baseline",
        "use_modulatory_channel_routing": False,
    },
    {
        "arm_id": "ARM_1_ROUTE_ON",
        "label": "channel_routing_on_cand_world_summary",
        "use_modulatory_channel_routing": True,
    },
]


# ---------------------------------------------------------------------------
# Regime-conditioned precondition gate (the 791a FIX -- see module docstring).
#
# 790 ANDed one flat sample gate across BOTH arms, so ARM_0_NO_ROUTE::seed49
# (53 genuine selections, in the CONTROL arm, whose route_range is structurally
# 0.0) vacated a 10/10 result in ARM_1. Every precondition below declares the
# arms it is MEANINGFUL for via applies_to; a scoped-out precondition is never
# failed by that arm and never enters the flat adjudication list the REE_assembly
# indexer reads arm-blind.
# ---------------------------------------------------------------------------

_SAMPLE_SCOPE_NOTE = (
    "ARM_1_ROUTE_ON only. The sample floor guards a STATISTIC ESTIMATED FROM THE "
    "SELECTION SAMPLE, and the only such statistic is ARM_1's route_range_mean. "
    "ARM_0_NO_ROUTE is the structural control: its route_range is identically 0.0 "
    "by construction (routing off), so no C1 or C2 quantity is estimated from its "
    "sample beyond 'routing is inactive' -- which is gated separately and far more "
    "cheaply by arm0_route_inactivity_observable. Asserting an ESTIMATION sample "
    "floor on a structurally-zero control arm is what produced the V3-EXQ-790 "
    "false refusal. The control arm's full per-cell counts are carried NON-GATING "
    "in recorded_preconditions (the V3-EXQ-737 pattern), so nothing is hidden."
)

PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="adequate_fresh_selection_sample",
        description=(
            "At least MIN_SEEDS_FOR_PASS ARM_1 seeds recorded FRESH_SELECT_FLOOR "
            "genuine E3 select() calls on the CORRECTED denominator (diagnostics "
            "cleared immediately before every select_action, so a latched tick "
            "contributes NO row). This is the gate on the ~9x pseudo-replication "
            "defect the shared 662/663 driver carried, and it is UNCHANGED in "
            "strength: the corrected denominator is byte-identical to 790's. Two "
            "things changed. (1) SCOPE: applied only to the arm whose statistics "
            "are estimated from the sample. (2) DERIVATION: the floor is now "
            "3600 nominal window ticks / beta_rate_max_steps=20 = 180, the "
            "MECH-093-modulated cadence range's WORST case, replacing 790's 200 "
            "keyed to the nominal heartbeat.e3_steps_per_tick=10 (which the "
            "substrate does not run). measured is the MIN_SEEDS_FOR_PASS-th "
            "largest per-seed n_fresh_select -- an ORDER STATISTIC, not a mean -- "
            "because met is a count quantifier matching the >= "
            "MIN_SEEDS_FOR_PASS-of-10 readiness criterion it guards. Below floor "
            "=> too few seeds carry an adequate sample => "
            "substrate_not_ready_requeue, never a substrate verdict."
        ),
        control=(
            "nominal window P1*(steps-measure_after)=3600 ticks at the "
            "MECH-093-modulated worst-case cadence of beta_rate_max_steps=20 "
            "steps/selection yields 180 selections; V3-EXQ-790 measured a "
            "7th-largest ARM_1 count of 262 on this identical design"
        ),
        threshold=float(FRESH_SELECT_FLOOR),
        direction="lower",
        applies_to=lambda ctx: bool(ctx["routing"]),
        applies_note=_SAMPLE_SCOPE_NOTE,
    ),
    PreconditionSpec(
        name="fresh_selection_yield_supra_cadence_floor",
        description=(
            "EVERY ARM_1 cell's n_fresh_select / n_p1_ticks_past_window cleared "
            "1/beta_rate_max_steps = 0.05. The same cadence bound as the absolute "
            "floor above, expressed as a FRACTION of the cell's OBSERVED window "
            "ticks and therefore invariant to window TRUNCATION -- which is the "
            "measured cause of 790's starved cells (per-cell windows ran "
            "330..3600 ticks because episodes terminate early on `done`; every "
            "cell's yield was 0.100..0.181, i.e. an effective cadence of 5.5..10 "
            "steps per selection, at or FASTER than nominal and never near the "
            "slow end). This is a STRICTLY STRONGER latch-pathology detector than "
            "an absolute count: a dead or latch-inflated selection loop is "
            "visible in the ratio at any window length. measured is the WORST "
            "cell (met is an all(...) quantifier), with the offending cell named "
            "in recorded_preconditions."
        ),
        control=(
            "MECH-093 caps the E3 cadence at beta_rate_max_steps=20 steps per "
            "selection, so a healthy cell yields >= 1/20 of its observed window "
            "ticks as genuine selections regardless of how short that window is"
        ),
        threshold=FRESH_SELECT_YIELD_FLOOR,
        direction="lower",
        applies_to=lambda ctx: bool(ctx["routing"]),
        applies_note=_SAMPLE_SCOPE_NOTE,
    ),
    PreconditionSpec(
        name="arm1_routed_bias_range_supra_floor",
        description=(
            "ARM_1 (routing ON, cand_world_summary source) routed-bias "
            "cross-candidate RANGE (modulatory_channel_route_range -- the RAW "
            "range the P0 gate keys on, pre-normalise/pre-rescale) clears the "
            "floor. This is a RANGE statistic, the SAME one C1 routes on, NOT a "
            "magnitude. measured is the MIN_SEEDS_FOR_PASS-th largest per-seed "
            "route_range_mean (order statistic, not a mean), which clears the "
            "floor iff at least that many seeds do -- exactly what met asserts. "
            "Below floor => under-trained e2 / collapsed candidate pool / amend "
            "not wired => substrate_not_ready_requeue, never a substrate verdict."
        ),
        control=(
            "ARM_1: SD-056 contrastive trained online; SP-CEM multi-class "
            "candidates; candidate_summary_source=e2_world_forward; routing ON"
        ),
        threshold=C0_ROUTE_FLOOR,
        direction="lower",
        applies_to=lambda ctx: bool(ctx["routing"]),
        applies_note=(
            "ARM_0_NO_ROUTE has routing OFF, so its routed range is 0.0 by "
            "construction. Asserting a supra-floor range there would be "
            "structurally unsatisfiable and would vacate the control arm."
        ),
    ),
    PreconditionSpec(
        name="arm0_route_inactivity_observable",
        description=(
            "Every ARM_0 cell recorded at least one genuine E3 selection, so C1's "
            "off-inactive leg reads an OBSERVED zero rather than an absent one. "
            "This is the ONLY sample question the structural control arm poses: "
            "it makes no estimate, so it needs observability, not statistical "
            "power. measured is the worst cell (met is an all(...) quantifier)."
        ),
        control=(
            "routing OFF makes modulatory_channel_route_range return 0.0 on every "
            "genuine selection; one observation per cell suffices to record that"
        ),
        threshold=ARM0_OBSERVABILITY_FLOOR,
        direction="lower",
        applies_to=lambda ctx: not bool(ctx["routing"]),
        applies_note=(
            "ARM_1_ROUTE_ON is gated on the far stronger estimation-sample "
            "preconditions above; a bare observability check would be redundant."
        ),
    ),
    PreconditionSpec(
        name="routed_range_bounded",
        description=(
            "Routed-bias range stayed finite and below the 643a explosion ceiling "
            "(SD-056 online training numerical stability; rollout-norm clamp ON). "
            "Applies to BOTH arms -- a blown-up range is a stability failure "
            "wherever it appears."
        ),
        control="max route_range across this arm's cells",
        threshold=C0_MAGNITUDE_CEIL,
        direction="upper",
    ),
]

ARM_CONTEXTS: List[Dict[str, Any]] = [
    {"id": a["arm_id"], "arm_id": a["arm_id"],
     "routing": bool(a["use_modulatory_channel_routing"])}
    for a in ARMS
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM stack with the SHARED E3-side bias channels (lateral_pfc +
    mech295) ON, the modulatory selection authority ON (gain=0.5),
    candidate_summary_source=e2_world_forward, and SD-056 contrastive trained online
    (the e2.world_forward divergence the routed world-summary channel range depends
    on; rollout-norm clamp ON per the 643a stability lesson). The ONLY swept axis is
    use_modulatory_channel_routing."""
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
        # ARC-065 SP-CEM (Layer A) -- main-path default (action-divergent pool)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # SHARED E3-side bias channels (consume cand_world_summaries) ON in BOTH arms
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators OFF (channel routing is the swept axis)
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        # SD-056 substrate trained online on every arm (e2.world_forward divergence)
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # ARC-065 GAP-A shared channel source (the world-summary representation routed)
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (the gate the routed range reaches)
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
        # --- route-range AMEND: the swept axis ---
        use_modulatory_channel_routing=bool(arm["use_modulatory_channel_routing"]),
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_min_range_floor=1e-6,
        modulatory_channel_route_weight=1.0,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _tv_distance(counts_a: Dict[int, int], counts_b: Dict[int, int]) -> float:
    """Total-variation distance between two committed-class histograms (0.5 * L1 of
    the normalised distributions over the union of classes)."""
    ta = sum(counts_a.values())
    tb = sum(counts_b.values())
    if ta <= 0 or tb <= 0:
        return 0.0
    classes = set(counts_a) | set(counts_b)
    tv = 0.0
    for cls in classes:
        pa = counts_a.get(cls, 0) / ta
        pb = counts_b.get(cls, 0) / tb
        tv += abs(pa - pb)
    return 0.5 * tv


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
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    measure_after_tick: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes

    route_ranges: List[float] = []
    route_range_max = 0.0
    route_active_ticks = 0
    committed_class_counts: Dict[int, int] = {}
    n_p1_ticks_past_window = 0
    # Route-row denominators. E3 select() runs on ~1 tick in
    # heartbeat.e3_steps_per_tick (default 10), so these are NOT
    # n_p1_ticks_past_window -- see the readout block below.
    n_fresh_select = 0
    n_latched_ticks = 0
    n_contrastive_steps = 0
    error_note: Optional[str] = None

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
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
                obs_harm=_obs(obs_dict, "harm_obs"),
                obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs(obs_dict, "harm_history"),
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

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            # Freshness marker: E3 populates last_score_diagnostics ONLY inside
            # select(), and it LATCHES -- on a tick where select() did not run it
            # still holds the previous selection's values. Clearing here makes a
            # populated dict below proof that THIS tick ran a genuine selection.
            agent.e3.last_score_diagnostics = None

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

            # --- route-range readout (P0 gate) + committed-action readout ---
            past_window = is_p1 and tick_in_ep >= measure_after_tick
            if past_window and candidates and len(candidates) >= 2:
                # Route rows come from FRESH selections only. Reading the latch
                # unconditionally re-recorded the previous selection as a new
                # independent observation on every held tick, inflating the route
                # sample ~e3_steps_per_tick-fold (the V3-EXQ-785 defect: 600 rows
                # behind 67 genuine selections). n_latched_ticks makes the true
                # denominator auditable.
                diag = agent.e3.last_score_diagnostics
                if diag is None:
                    n_latched_ticks += 1
                else:
                    n_fresh_select += 1
                    rr = diag.get("modulatory_channel_route_range")
                    if rr is not None and math.isfinite(float(rr)):
                        route_ranges.append(float(rr))
                        route_range_max = max(route_range_max, float(rr))
                    if bool(diag.get("modulatory_channel_route_active", False)):
                        route_active_ticks += 1
                # The committed-action readout stays on EVERY env step in the
                # window: the held action IS the agent's committed behaviour across
                # latched ticks, so its class distribution is a behavioural readout,
                # not a per-selection one. Both denominators are now emitted.
                cls = int(action.argmax(dim=-1).item())
                committed_class_counts[cls] = committed_class_counts.get(cls, 0) + 1
                n_p1_ticks_past_window += 1

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val) and is_p1:
                    n_contrastive_steps += 1

            _, harm_signal, done, info, next_obs_dict = env.step(action)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    # Denominator is FRESH selections, not env steps: both numerator and
    # denominator are now counted on the same (genuine-selection) event.
    route_active_frac = (
        float(route_active_ticks) / float(n_fresh_select)
        if n_fresh_select > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "use_modulatory_channel_routing": bool(arm["use_modulatory_channel_routing"]),
        "n_p1_ticks_past_window": int(n_p1_ticks_past_window),
        # True route-row denominator + the latched-tick count it was inflated by.
        "n_fresh_select": int(n_fresh_select),
        "n_latched_ticks": int(n_latched_ticks),
        "fresh_select_yield": (
            round(float(n_fresh_select) / float(n_p1_ticks_past_window), 4)
            if n_p1_ticks_past_window > 0 else 0.0
        ),
        "n_route_rows": int(len(route_ranges)),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # P0 gate: RAW cross-candidate range of the routed channel bias (pre-rescale).
        "route_range_mean": round(_mean(route_ranges), 6),
        "route_range_max": round(route_range_max, 6),
        "route_active_frac": round(route_active_frac, 4),
        # Committed-action readout (the 569f selected-action axis).
        "committed_class_counts": {str(k): int(v) for k, v in committed_class_counts.items()},
        "committed_class_entropy": round(_entropy_from_counts(committed_class_counts), 6),
        "n_committed_classes": int(len(committed_class_counts)),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds(rows: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for r in rows if predicate(r))


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _counts(r: Dict[str, Any]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in (r.get("committed_class_counts") or {}).items()}


def _cell_id(r: Dict[str, Any]) -> str:
    return f"{r.get('arm_id')}::seed{r.get('seed')}"


def _yield_of(r: Dict[str, Any]) -> float:
    """Fresh-selection yield recomputed from RAW counts (not the rounded field)."""
    ticks = int(r.get("n_p1_ticks_past_window", 0))
    if ticks <= 0:
        return 0.0
    return float(r.get("n_fresh_select", 0)) / float(ticks)


def _worst_cell(rows: List[Dict[str, Any]], value_fn) -> Tuple[float, Optional[str]]:
    """Minimum of `value_fn` over `rows`, plus the offending cell id.

    `met` for these preconditions is an all(...) quantifier, so `measured` must be
    the WORST cell -- a mean would let one starved cell recompute as met and
    silently pass exactly the condition the gate exists to catch (skill Step 3).
    """
    if not rows:
        return 0.0, None
    worst_row = min(rows, key=value_fn)
    return float(value_fn(worst_row)), _cell_id(worst_row)


def _kth_largest(values: List[float], k: int) -> float:
    """k-th largest value, or 0.0 when fewer than k exist.

    Order statistic for a COUNT quantifier: it clears a floor iff at least k
    values do, which is exactly what the corresponding `met` asserts. A mean is
    NOT recomputable against a count quantifier.
    """
    ordered = sorted(values, reverse=True)
    return float(ordered[k - 1]) if len(ordered) >= k else 0.0


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm0 = _arm_rows(arm_results, "ARM_0_NO_ROUTE")
    arm1 = _arm_rows(arm_results, "ARM_1_ROUTE_ON")
    arm0_by_seed = {r["seed"]: r for r in arm0}

    # READINESS (load-bearing non-vacuity, RANGE statistic): ARM_1 routed range > floor.
    readiness_seeds_ok = _n_seeds(
        arm1, lambda r: float(r.get("route_range_mean", 0.0)) > C0_ROUTE_FLOOR
    )
    arm1_route_mean = _mean_key(arm1, "route_range_mean")
    # Order statistic for the readiness precondition: `met` is a COUNT quantifier
    # ("at least MIN_SEEDS_FOR_PASS seeds above floor"), so a mean is not
    # recomputable against it -- the indexer recomputes met from
    # (measured, threshold) and would disagree with our own flag. The
    # MIN_SEEDS_FOR_PASS-th largest per-seed value clears the floor if and only
    # if at least that many seeds do.
    arm1_kth_route = _kth_largest(
        [float(r.get("route_range_mean", 0.0)) for r in arm1], MIN_SEEDS_FOR_PASS
    )
    max_route = max(
        [float(r.get("route_range_max", 0.0)) for r in arm_results] or [0.0]
    )
    magnitude_ok = bool(math.isfinite(max_route) and max_route < C0_MAGNITUDE_CEIL)

    # ---- SAMPLE ADEQUACY, ARM-SCOPED (the 791a fix) -------------------------
    # 790 ANDed a single flat floor across BOTH arms and flattened every cell into
    # interpretation.preconditions, so one starved CONTROL cell vacated a 10/10
    # result in the measured arm at adjudication time. Here each precondition
    # declares the arms it is meaningful for (PreconditionSpec.applies_to), and
    # aggregate_arm_gates keeps the flat adjudication list free of scored-out
    # cells so the indexer's arm-blind _compute_adjudication cannot re-vacate a
    # green arm.
    arm1_kth_fresh = _kth_largest(
        [float(r.get("n_fresh_select", 0)) for r in arm1], MIN_SEEDS_FOR_PASS
    )
    arm1_worst_yield, arm1_worst_yield_cell = _worst_cell(arm1, _yield_of)
    arm1_worst_fresh, arm1_worst_fresh_cell = _worst_cell(
        arm1, lambda r: float(r.get("n_fresh_select", 0))
    )
    arm0_worst_fresh, arm0_worst_fresh_cell = _worst_cell(
        arm0, lambda r: float(r.get("n_fresh_select", 0))
    )
    arm0_worst_yield, arm0_worst_yield_cell = _worst_cell(arm0, _yield_of)

    def _arm_max_route(rows: List[Dict[str, Any]]) -> float:
        return max([float(r.get("route_range_max", 0.0)) for r in rows] or [0.0])

    measured_by_arm: Dict[str, Dict[str, float]] = {
        "ARM_0_NO_ROUTE": {
            "arm0_route_inactivity_observable": arm0_worst_fresh,
            "routed_range_bounded": _arm_max_route(arm0),
        },
        "ARM_1_ROUTE_ON": {
            "adequate_fresh_selection_sample": arm1_kth_fresh,
            "fresh_selection_yield_supra_cadence_floor": arm1_worst_yield,
            "arm1_routed_bias_range_supra_floor": arm1_kth_route,
            "routed_range_bounded": _arm_max_route(arm1),
        },
    }
    arm_gates = [
        evaluate_arm_gate(ctx["id"], ctx, PRECONDITION_SPECS,
                          measured_by_arm[ctx["id"]])
        for ctx in ARM_CONTEXTS
    ]
    gate = aggregate_arm_gates(arm_gates)
    gate_green_by_arm = {g["arm"]: bool(g["gate_green"]) for g in arm_gates}
    arm1_green = bool(gate_green_by_arm.get("ARM_1_ROUTE_ON", False))
    arm0_green = bool(gate_green_by_arm.get("ARM_0_NO_ROUTE", False))

    # Sample adequacy is now an ARM_1 property. Kept as named booleans so the
    # summary block stays readable and comparable against 790's.
    sample_ok = bool(
        arm1_kth_fresh > float(FRESH_SELECT_FLOOR)
        and arm1_worst_yield > FRESH_SELECT_YIELD_FLOOR
    )
    readiness_ok = bool(arm1_green)

    # C1 PRIMARY (load-bearing, SAME range statistic): ARM_1 routing active on >=N seeds
    # AND ARM_0 routed range ~0 (routing off). The routed range reaches the accumulator.
    c1_on_active_seeds = _n_seeds(arm1, lambda r: float(r.get("route_active_frac", 0.0)) > 0.5)
    c1_off_inactive_seeds = _n_seeds(
        arm0, lambda r: float(r.get("route_range_mean", 0.0)) <= C1_OFF_INACTIVE_CEIL
    )
    c1_pass = bool(
        c1_on_active_seeds >= MIN_SEEDS_FOR_PASS
        and c1_off_inactive_seeds >= MIN_SEEDS_FOR_PASS
    )
    c1_non_degenerate = bool(arm1_route_mean > C0_ROUTE_FLOOR)  # real routed range exists

    # C2 SECONDARY (behavioural reach, NOT load-bearing): committed-class TV per seed.
    def _c2(r1: Dict[str, Any]) -> bool:
        r0 = arm0_by_seed.get(r1["seed"])
        if r0 is None:
            return False
        return _tv_distance(_counts(r1), _counts(r0)) > C2_TV_FLOOR
    c2_seeds_ok = _n_seeds(arm1, _c2)
    c2_pass = bool(c2_seeds_ok >= MIN_SEEDS_FOR_PASS)
    # TV per seed for the manifest.
    tv_per_seed = {}
    for r1 in arm1:
        r0 = arm0_by_seed.get(r1["seed"])
        if r0 is not None:
            tv_per_seed[str(r1["seed"])] = round(_tv_distance(_counts(r1), _counts(r0)), 4)
    # Non-degeneracy now requires BOTH denominators non-zero: env steps in the
    # window (the committed-class readout) AND genuine E3 selections (the route
    # readout). A window with steps but zero fresh selections yields no route rows
    # at all, which the old check could not see.
    c2_non_degenerate = bool(
        len(arm1) > 0
        and all(int(r.get("n_p1_ticks_past_window", 0)) > 0 for r in arm1)
        and all(int(r.get("n_fresh_select", 0)) > 0 for r in arm1)
        and len(arm0) > 0
        and all(int(r.get("n_p1_ticks_past_window", 0)) > 0 for r in arm0)
        and all(int(r.get("n_fresh_select", 0)) > 0 for r in arm0)
    )

    criteria_pass = {"C1": c1_pass, "C2": c2_pass}
    # PASS gated on the ARM_1 gate (readiness + arm-scoped sample adequacy) + C1.
    # C2 corroborates. A red ARM_0 does NOT gate the run: it makes no estimate, and
    # if its inactivity were genuinely unobservable C1's own off-inactive leg fails
    # on its own and the run self-routes route_range_inert -- a wiring verdict, not
    # a readiness one. This is the whole point of the arm-scoping fix.
    if not readiness_ok:
        # Denominator starvation / absent routed range is a SUBSTRATE-READINESS
        # failure, never a verdict.
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif c1_pass:
        label = "route_range_substrate_ready"
        overall_pass = True
    else:
        label = "route_range_inert"
        overall_pass = False

    return {
        "readiness": {
            "c0_route_floor": C0_ROUTE_FLOOR,
            "arm1_route_range_mean": round(arm1_route_mean, 6),
            "arm1_seeds_above_floor": int(readiness_seeds_ok),
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "max_route_range_observed": round(max_route, 6),
            "magnitude_ceil": C0_MAGNITUDE_CEIL,
            "magnitude_ok": magnitude_ok,
            "readiness_ok": readiness_ok,
            "sample_ok": sample_ok,
            # Sample adequacy is ARM_1-scoped in 791a (790 ANDed it across arms).
            "fresh_select_floor": FRESH_SELECT_FLOOR,
            "fresh_select_floor_derivation": (
                f"{NOMINAL_WINDOW_TICKS} nominal window ticks / "
                f"beta_rate_max_steps={BETA_RATE_MAX_STEPS} (MECH-093 modulated "
                f"worst-case cadence); replaces 790's 200 keyed to the nominal "
                f"heartbeat.e3_steps_per_tick=10"
            ),
            "fresh_select_yield_floor": round(FRESH_SELECT_YIELD_FLOOR, 6),
            "arm1_kth_n_fresh_select": int(arm1_kth_fresh),
            "arm1_kth_k": MIN_SEEDS_FOR_PASS,
            "arm1_worst_n_fresh_select": int(arm1_worst_fresh),
            "arm1_worst_fresh_select_cell": arm1_worst_fresh_cell,
            "arm1_worst_fresh_select_yield": round(arm1_worst_yield, 6),
            "arm1_worst_fresh_select_yield_cell": arm1_worst_yield_cell,
            "arm1_gate_green": arm1_green,
            "arm0_gate_green": arm0_green,
        },
        "criteria_pass": criteria_pass,
        "c1_arm1_seeds_route_active": int(c1_on_active_seeds),
        "c1_arm0_seeds_route_inactive": int(c1_off_inactive_seeds),
        "c2_arm1_seeds_committed_tv_above_floor": int(c2_seeds_ok),
        "c2_committed_tv_per_seed": tv_per_seed,
        "route_range_per_arm_mean": {
            "ARM_0_NO_ROUTE": round(_mean_key(arm0, "route_range_mean"), 6),
            "ARM_1_ROUTE_ON": round(_mean_key(arm1, "route_range_mean"), 6),
        },
        "route_active_frac_per_arm_mean": {
            "ARM_0_NO_ROUTE": round(_mean_key(arm0, "route_active_frac"), 4),
            "ARM_1_ROUTE_ON": round(_mean_key(arm1, "route_active_frac"), 4),
        },
        "committed_class_entropy_per_arm_mean": {
            "ARM_0_NO_ROUTE": round(_mean_key(arm0, "committed_class_entropy"), 6),
            "ARM_1_ROUTE_ON": round(_mean_key(arm1, "committed_class_entropy"), 6),
        },
        "label": label,
        "overall_pass": overall_pass,
        # ---- Diagnostic adjudication structures (skill Step 3.5) -----------
        # ARM-SCOPED. `adjudication_preconditions` carries GREEN arms only, so a
        # scored-out control cell can never re-vacate a green arm through the
        # REE_assembly indexer's flat, arm-blind _compute_adjudication. The full
        # per-arm picture (green AND red, applied AND scoped_out) is carried at
        # manifest top level under per_arm_gate -- nothing is hidden.
        "preconditions": gate["adjudication_preconditions"],
        "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
        "per_arm_gate": gate["per_arm_gate"],
        "non_degenerate": gate["non_degenerate"],
        "degeneracy_reason": gate["degeneracy_reason"],
        # NON-GATING record of the control arm's selection sample (the V3-EXQ-737
        # pattern). ARM_0_NO_ROUTE makes no estimate from this sample, so these
        # numbers must never gate the run -- but they must remain visible, because
        # they are exactly the numbers V3-EXQ-790 refused on.
        "recorded_preconditions": [
            {
                "name": "arm0_control_fresh_selection_sample",
                "kind": "recorded",
                "gating": False,
                "description": (
                    "ARM_0_NO_ROUTE per-cell genuine E3 selection counts, RECORDED "
                    "AND NOT GATED. The control arm's route_range is structurally "
                    "0.0 (routing off) and no C1/C2 statistic is estimated from its "
                    "sample beyond 'routing is inactive', which is gated separately "
                    "by arm0_route_inactivity_observable. V3-EXQ-790 applied the "
                    "ARM_1 estimation-sample floor to these cells and refused the "
                    "whole run on ARM_0_NO_ROUTE::seed49 (53 vs 200) while both "
                    "science criteria passed 10/10 and 8/10."
                ),
                "worst_n_fresh_select": int(arm0_worst_fresh),
                "worst_cell": arm0_worst_fresh_cell,
                "worst_fresh_select_yield": round(arm0_worst_yield, 6),
                "worst_fresh_select_yield_cell": arm0_worst_yield_cell,
                "reference_floor_not_applied": FRESH_SELECT_FLOOR,
                "per_cell": [
                    {"cell": _cell_id(r),
                     "n_fresh_select": int(r.get("n_fresh_select", 0)),
                     "n_latched_ticks": int(r.get("n_latched_ticks", 0)),
                     "n_p1_ticks_past_window": int(r.get("n_p1_ticks_past_window", 0)),
                     "fresh_select_yield": round(_yield_of(r), 6)}
                    for r in arm0
                ],
            },
            {
                "name": "arm1_per_cell_fresh_selection_sample",
                "kind": "recorded",
                "gating": False,
                "description": (
                    "ARM_1_ROUTE_ON per-cell selection sample, recorded in full "
                    "alongside the two gating legs so the count quantifier "
                    "(adequate_fresh_selection_sample, k-th largest) and the "
                    "all(...) quantifier (fresh_selection_yield_supra_cadence_floor, "
                    "worst cell) are both independently checkable by a reader."
                ),
                "per_cell": [
                    {"cell": _cell_id(r),
                     "n_fresh_select": int(r.get("n_fresh_select", 0)),
                     "n_latched_ticks": int(r.get("n_latched_ticks", 0)),
                     "n_p1_ticks_past_window": int(r.get("n_p1_ticks_past_window", 0)),
                     "fresh_select_yield": round(_yield_of(r), 6)}
                    for r in arm1
                ],
            },
        ],
        "criteria": [
            {"name": "C1_routed_range_reaches_accumulator_on_active_off_inactive",
             "load_bearing": True, "passed": c1_pass},
            {"name": "C2_committed_class_distribution_moves_on_vs_off",
             "load_bearing": False, "passed": c2_pass},
        ],
        # Per-criterion non-degeneracy keyed to the OWNING arm's gate -- the channel
        # build_experiment_indexes.py honours, and what makes a green arm's result
        # separable at adjudication time without an autopsy. Both criteria are owned
        # by ARM_1_ROUTE_ON (the arm carrying the measured contrast); C1 additionally
        # requires the control arm's inactivity to be OBSERVABLE, since its
        # off-inactive leg is read there.
        "criteria_non_degenerate": arm_criteria_non_degenerate(
            {"ARM_1_ROUTE_ON": [
                "C1_routed_range_reaches_accumulator_on_active_off_inactive",
                "C2_committed_class_distribution_moves_on_vs_off",
            ]},
            gate,
            extra={
                "C1_routed_range_reaches_accumulator_on_active_off_inactive":
                    bool(c1_non_degenerate and arm0_green),
                "C2_committed_class_distribution_moves_on_vs_off":
                    bool(c2_non_degenerate),
            },
        ),
        # Short-key mirror of the above, preserving 790's C1/C2 keys for any reader
        # (including the explorer) that indexes by the bare criterion label.
        "criteria_non_degenerate_short": {
            "C1": bool(arm1_green and c1_non_degenerate and arm0_green),
            "C2": bool(arm1_green and c2_non_degenerate),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Design-time proof, BEFORE any compute is spent: no arm carries a precondition
    # it provably cannot satisfy from its PRE-REGISTERED config. This is the check
    # that would have caught V3-EXQ-785 for free at queue time, and it is what
    # distinguishes the 791a re-scoping (disposition (a): the sample floor is NOT
    # MEANINGFUL for a structurally-zero control arm) from silently lowering a
    # threshold to make an unsatisfiable gate pass.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, ARM_CONTEXTS)

    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    measure_after = DRY_RUN_MEASURE_AFTER_TICK if dry_run else MEASURE_AFTER_TICK

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps, measure_after)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {k: arm[k] for k in ("arm_id", "use_modulatory_channel_routing")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "p0_episodes": p0, "p1_episodes": p1, "steps_per_episode": steps,
                },
                seed=seed,
                script_path=Path(__file__),
                rng_fully_reset=True,
                extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
            )
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES_RUN_ID,
        # Arm-scoped gate, at TOP LEVEL where the indexer and pending_review can see
        # it. V3-EXQ-785's manifest carried per-regime gates buried inside
        # regime_analyses and nothing downstream ever read them.
        "per_arm_gate": summary["per_arm_gate"],
        "non_degenerate": summary["non_degenerate"],
        "degeneracy_reason": summary["degeneracy_reason"],
        "recorded_preconditions": summary["recorded_preconditions"],
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "modulatory-bias-selection-authority route-range AMEND substrate-readiness "
            "diagnostic (E3Config.use_modulatory_channel_routing + project_channel_range; "
            "landed 2026-06-10). Routed by failure_autopsy_569f-661-654a_2026-06-10. "
            "claim_ids=[] (does NOT weight claim confidence). PASS "
            "(label=route_range_substrate_ready) confirms the P0 routed-range gate: the "
            "channel-under-test's cross-candidate range is now routed into the modulatory "
            "bias the authority rescales (the 569f/661/654a gap). It UNBLOCKS the per-claim "
            "behavioural retests of ARC-065 / MECH-294 / ARC-062 / MECH-309 / MECH-341 "
            "(each a SEPARATE /queue-experiment session). Readiness-below-floor self-routes "
            "substrate_not_ready_requeue, NOT a substrate verdict. Those claims stay "
            "candidate / v3_pending / pending_retest_after_substrate; not weakened. "
            "791a SUPERSEDES the V3-EXQ-790 run (GATE FIX ONLY -- identical design, "
            "env, arms, seeds, schedule and science thresholds). 790's science stood "
            "(C1 10/10 ARM_1 active + 10/10 ARM_0 inactive; C2 8/10 vs a bar of 7; "
            "routed range 0.334 supra-floor and 1.644 bounded) and its "
            "substrate_not_ready_requeue self-route was WITHDRAWN as a gate defect by "
            "failure_autopsy_V3-EXQ-790_2026-07-22 (user-adjudicated). Fixed here: "
            "(1) the sample gate is ARM-SCOPED to the arm whose statistics are "
            "estimated from the selection sample, with the control arm's counts "
            "carried non-gating in recorded_preconditions; (2) the floor is "
            "re-derived from the MECH-093-modulated cadence's worst case (3600/20 = "
            "180) plus a truncation-invariant yield floor (1/20 = 0.05), replacing "
            "790's 200 keyed to the nominal e3_steps_per_tick=10. The "
            "pseudo-replication protection is UNCHANGED and remains load-bearing."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "preconditions_scope_note": summary["preconditions_scope_note"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "route_range_substrate_ready": "PASS -> /queue-experiment per-claim behavioural retests (ARC-065/MECH-294/ARC-062/MECH-309/MECH-341), each separate",
                "substrate_not_ready_requeue": "ARM_1 gate red -> re-queue at higher P0 budget (or fix e2/SD-056 wiring); do NOT weaken. NOTE a red ARM_0_NO_ROUTE alone never routes here -- that was the V3-EXQ-790 defect.",
                "route_range_inert": "FAIL -> /failure-autopsy on the routing wiring",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "measure_after_tick": measure_after,
            "env_kwargs": ENV_KWARGS,
            "arms": [{k: a[k] for k in ("arm_id", "label", "use_modulatory_channel_routing")} for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "modulatory_channel_route_source": "cand_world_summary",
            "modulatory_authority_gain": 0.5,
            "thresholds": {
                "c0_route_floor": C0_ROUTE_FLOOR,
                "c0_magnitude_ceil": C0_MAGNITUDE_CEIL,
                "c1_off_inactive_ceil": C1_OFF_INACTIVE_CEIL,
                "c2_tv_floor": C2_TV_FLOOR,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
                # Re-derived in 791a from the MECH-093-modulated cadence range
                # (clock.beta_rate_min_steps=5 .. beta_rate_max_steps=20), NOT
                # from the nominal heartbeat.e3_steps_per_tick=10 that produced
                # 790's unreachable 200.
                "fresh_select_floor": FRESH_SELECT_FLOOR,
                "fresh_select_yield_floor": round(FRESH_SELECT_YIELD_FLOOR, 6),
                "beta_rate_max_steps": BETA_RATE_MAX_STEPS,
                "nominal_window_ticks": NOMINAL_WINDOW_TICKS,
                "arm0_observability_floor": ARM0_OBSERVABILITY_FLOOR,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "C1_routed_range_reaches_accumulator": summary["criteria_pass"]["C1"],
            "C2_committed_distribution_moves": summary["criteria_pass"]["C2"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
            started_at=t0,
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)
    pag = summary["per_arm_gate"]
    print(
        f"  per_arm_gate: green={pag['green_arms']} red={pag['red_arms']}",
        flush=True,
    )
    rd = summary["readiness"]
    print(
        f"  sample (ARM_1 only): kth_n_fresh_select={rd['arm1_kth_n_fresh_select']} "
        f"(k={rd['arm1_kth_k']}, floor {rd['fresh_select_floor']}); "
        f"worst_yield={rd['arm1_worst_fresh_select_yield']} "
        f"(floor {rd['fresh_select_yield_floor']}) at "
        f"{rd['arm1_worst_fresh_select_yield_cell']}",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-791a modulatory-bias-selection-authority route-range "
            "substrate-readiness diagnostic (same-question re-run of V3-EXQ-790, "
            "arm-scoped gate + cadence-derived fresh-select floor)"
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
