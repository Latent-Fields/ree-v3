"""V3-EXQ-571c: E3 CROSS-CANDIDATE variance-MONOPOLY PRESENCE audit in the
936-family REGIME, clamp armed, with the temporal partition recorded alongside and
a residue-protocol / warm-up wiring-check + open-contrast grid.

WHY THIS RUN EXISTS. failure_autopsy_V3-EXQ-936a_2026-08-30 gated every further
936-family MECH-439 falsifier on "a clamped 571b-shape presence audit". V3-EXQ-571b
ran that audit and returned monopoly-PRESENT (harm_weighted 0.937-0.995 of committed
variance, clamp armed) -- but in 571's OWN regime (8x8, 1 hazard, static, no residue
feeding, 20 ep x 100 steps), while V3-EXQ-936a had already measured the same
premise ABSENT in the 936 regime (F committed share 6.33e-06; residue_weighted
0.999988-0.999998). failure_autopsy_V3-EXQ-571b_2026-09-01 (confirmed, user-ratified
2026-09-01) therefore kept 936a's gate SHUT for the 936 regime and routed THIS run:
a presence audit matched to the 936 falsifier family on ALL THREE dimensions the
two runs differ on --

  1. ENVIRONMENT      12x12, 4 hazards, 5 resources, reef bipartite, env drift
  2. RESIDUE PROTOCOL agent.update_residue(harm_signal=..., owned=True) on EVERY
                      env step (936a driver:760); 571b never called it
  3. SCHEDULE         P0 60 episodes x 200 steps, then a P1 measurement phase

-- reporting which channel holds the monopoly and whether the 0.85 bar clears.

THE ROUTED DV IS THE PER-TICK CROSS-CANDIDATE PARTITION, NOT THE TEMPORAL ONE
(redesign after the Step 4.5 red-team, BLOCKING, model opus, 2026-09-02). Every
predecessor in this lineage (571, 936, 936a, 571b) routed on a TEMPORAL variance
share: the variance ACROSS SELECTIONS of a component's value (at the pool mean or at
the committed candidate). MECH-439's premise is about which channel drives WHICH
candidate is committed -- a WITHIN-TICK, CROSS-CANDIDATE quantity -- and the two can
diverge sharply: a channel that is near-uniform across the K candidates at each tick
but drifts across the phase monopolises the temporal share while moving no argmin,
which is exactly the divergence failure_autopsy_V3-EXQ-925 learning point 1 (retained
by 925a) already records, and which this run's own first smoke exhibited
(dispersion_to_range_ratio 229-256 in the fed arms against 571b's band of 0.44-3.18).
So, at every genuine (latch-gated) P1 selection, for every component, this driver
computes the WITHIN-TICK variance of that component ACROSS THE CANDIDATE SET, averages
it over the P1 selections, and forms the sum-to-one partition of those means:
`xcand_share_<comp>`. C1 and C2 route on THAT partition. The temporal partitions --
571b's committed-candidate one and 571's / 936a's pool one, on both the corrected
f_weighted set and the legacy raw-f set -- are still recorded per cell as
`temporal_*` for comparability with the four predecessors, and the DISAGREEMENT
between the temporal and cross-candidate partitions is emitted as a first-class
readout (`partition_disagreement`: total-variation distance between the two share
vectors, per-component pairs, and whether the top channel differs), because that
disagreement is itself the diagnostic the 925 autopsy asked for.

THE LOAD-BEARING CELL IS 936a's OWN ARM_OFF, MEASURED BY 571b's INSTRUMENT. The B1
arm is constructed from the 936 lineage's canonical baseline module
(experiments/_lib/baselines/mech439_f_variance_share.py: make_env / make_agent_kwargs
/ OFF_ARM_FLAGS / P0 schedule / SD-056 online contrastive training / the rollout
clamp at ratio 2.0), stepped exactly as the 936a driver steps it (per-step
update_residue included), and read out with 571b's instrument (latch-gated fresh
selections, the corrected f_weighted decomposition alongside 571's legacy set,
clamp-binding scale anchors) plus the cross-candidate partition above.

THE PARTITION'S TRUE DIMENSIONALITY IS RECORDED, NOT ASSUMED. Three of the six
declared score components are identically zero under this config for structural
reasons the red-team traced to source: `novelty_weighted` is the hardcoded constant
`_dc_novelty_w = 0.0` (e3_selector.py:1295; the MECH-111 broadcast branch was deleted
2026-05-25 as argmin-invariant); `benefit_weighted` is gated on
`record_benefit_sample()`, which has NO callers anywhere in ree_core or the drivers;
`goal_weighted` is gated on `goal_state.is_active()`, and this driver seeds z_goal
with `benefit_exposure=0.0` (936a parity), so it never activates. Each cell records
`n_live_channels` (components whose mean cross-candidate variance clears
MIN_LIVE_CHANNEL_VARIANCE), the `live_channels` list and
`structurally_zero_components` with the reason for each, and a readiness predicate
`n_live_channels >= 2` self-routes a one-live-channel partition to
`substrate_not_ready_requeue` instead of letting it read as a monopoly.

ARMS (2x2 inside the 936 env; matched seeds 42/43/45/46 from the lineage module,
seed 44 deliberately absent -- reef per-seed instability; every arm shares the
layout at a seed because make_env(seed) is arm-independent):

  B1_936_regime_fed_warmup     fed + P0 warm-up   -- LOAD-BEARING (the 936 regime)
  B2_936_env_starved_warmup    STARVED + warm-up  -- residue-protocol WIRING CHECK
  B3_936_env_fed_no_warmup     fed + NO warm-up   -- the one OPEN contrast (schedule)
  B4_936_env_starved_no_warmup STARVED + no P0    -- 571b's protocol in the 936 env;
                                                    wiring check + 571b replication

WHAT THE CONTRASTS CAN AND CANNOT ESTABLISH (red-team finding 4, accepted). "Starved"
means exactly what 571b did: no update_residue call at all -- and update_residue is
the ONLY write path into the residue RBF field (accumulate is commitment-gated inside
post_action_update, which update_residue calls). Removing the only source of a
channel's variance removes that channel's variance by definition, so an occupant
flip between B1 and B2 (C3) or B1 and B4 (C5) is OVERDETERMINED by the manipulation:
those two criteria are WIRING CHECKS that the protocol dimension landed as declared,
not attribution of the 936a-vs-571b gap. They are kept because a wiring check that
FAILS is informative. The only contrast whose outcome is open is C4 (B3 vs B1: fed
vs fed, warm-up removed). No `gap_carrier` verdict is emitted; the run does not claim
to apportion the gap between environment, protocol and schedule.

ARMS ARE MATCHED ON DECOMPOSITION SAMPLE COUNT, NOT ON EXPOSURE (red-team finding 7,
accepted). P1 runs until N_FRESH_SELECT_TARGET genuine selections, which equalises
the DV's sample size by construction -- but the starved arms hold ~3x longer (fresh
yield ~0.45 fed vs ~0.13 starved, measured in the probe and the smoke), so at the full
target they run ~3x more P1 env ticks and receive ~3x more SD-056 online E2 updates
(every 4 ticks, all phases), and E2 shapes the candidate summaries the modulatory
channels read (candidate_summary_source="e2_world_forward"). `n_env_ticks_total`,
`n_p1_ticks` and `n_contrastive_steps_p1` are recorded per cell and per arm, and the
interpretation block carries an explicit `exposure_matching_note`. Quantitative
fed-vs-starved comparisons are confounded by exposure; the occupant identity is not.

STEP 2.5a PROBE (2026-09-02, seed 42, 120 env steps, no warm-up): clamp live on
agent.e2.config (True, ratio 2.0); per-candidate decomp exposes f, f_weighted,
harm_weighted, residue_weighted, lambda_eff; committed rollout pinned at the
ceiling (||z_w_last|| / (2 ||z_w_0||) = 1.0000 both arms). FED: TEMPORAL
residue_weighted var 190.1 vs harm_weighted var 1.3e-04, 32/32 RBF centres active.
STARVED: residue var 2.9e-07 vs harm var 2.1e-04, 0/32 centres. Both channels are
live on the temporal axis; the cross-candidate axis is what this run measures.

INTERPRETATION GRID (C1 alone routes; C2 refines; C3/C5 wiring; C4 open):
  B1 gate red                    -> substrate_not_ready_requeue (FAIL; no verdict)
  B1 seeds disagree on occupant  -> seeds_disagree_on_occupant_936_regime (PASS; no
                                    single channel monopolises ACROSS seeds; C1 false)
  C1 and C2                      -> f_monopoly_present_936_regime (PASS): the 936a
                                    gate CLEARS for the 936 regime; the outcome_note
                                    names WHICH F channel (f_weighted / harm_weighted)
  C1 and not C2                  -> monopoly_present_non_f_occupant_936_regime (PASS):
                                    a cross-candidate monopoly exists but F does not
                                    hold it; the 936-family falsifier must be re-posed
                                    against the named occupant, not against F
  not C1                         -> no_monopoly_936_regime (PASS): no channel >= 0.85
                                    of cross-candidate variance
C1 = the B1 seeds are UNANIMOUS on the cross-candidate top channel AND the mean over
seeds of THAT channel's share >= 0.85 (never the mean of per-seed maxima, which can
pass while seeds disagree). The modal channel is picked deterministically
(sorted by (-count, name)).

RE-DERIVE BRAKE (Step 2.5b): MECH-439 carries 14 counted autopsies under the
reconciled predicate (571b's own is counted by the predicate because its direction
is non_contributory and it owes an SD-056 amend, although its artifact states it
adds no ceiling hit). Not braked: this is a DIAGNOSTIC on the MEASUREMENT axis
asking a different question (regime-dependence of the premise), not another letter
of the braked f-dominance design, exactly as 571b's autopsy re_derive_brake note
states. No ceiling hit is added by this run.

KNOWN OPEN SUBSTRATE LIMITATIONS THIS RUN EXECUTES UNDER (Step 2.5c, recorded not
blocking). Three open `corrupting` entries overlap paths this driver exercises:
`SD-082` (ree_core/pfc/lateral_pfc_analog.py::compute_bias -- the 936 regime sets
use_lateral_pfc_analog=True), `contextmemory-write-path-addressing-degeneracy`
(e1_deep.py::ContextMemory.write) and `SD-e1-rollout-consistency-training`
(e1_deep.py::forward / predict_long_horizon). All three are fix-landed,
validation-pending. None can reach the decomposed channels DIRECTLY: the E3 score is
`f_weight * f + lambda_eff * m + rho_residue * phi` (e3_selector.py:1310) and the
component dict (:1375-1383) exposes exactly those three weighted terms; the lPFC bias
enters through the `score_bias` kwarg composed in REEAgent.select_action, and the E1
paths bear on rollout/representation QUALITY. They can move this DV only by changing
the candidate set or which candidate is committed. Disabling them would break the
936-regime parity that is the entire point of the audit. The other open corrupting
entries do not apply: use_salience_coordinator and use_blocked_agency are
config-default False and the 936 CONFIG_FLAGS do not enable them; probe_warmup.py is
not imported.

DV-SYMMETRY DECLARATION (one line per arm). The ROUTED DV is a set of SHARES of
mean within-tick cross-candidate variance. Its symmetry group: a per-tick offset
that is UNIFORM across candidates (a broadcast scalar -- correctly invisible, since
it is argmin-invariant), permutation of the candidate index, permutation of the
selection index, and a COMMON positive rescaling of every channel (a share is
invariant in the common factor).
  B1: the reference cell -- no manipulation; it is the 936 regime as the 936 family
      runs it, measured. It anchors the contrasts and carries the presence verdict on
      its own absolute share.
  B2 vs B1 (residue protocol): removing update_residue removes the ONLY write path
      into the residue RBF field, so phi's profile ACROSS CANDIDATES (which depends on
      where each candidate's rollout lands in the field) collapses toward a constant --
      a per-channel change of cross-candidate spread, not a broadcast offset and not a
      common rescaling. It reaches the DV; that is why C3 is a wiring check.
  B3 vs B1 (warm-up): 60 episodes of SD-056 online E2 training, residue accumulation
      and env drift before measurement change the E2 weights (hence every candidate's
      rollout) and the RBF field state, which enter f/m/phi through different maps --
      not a common rescaling and not a broadcast offset. Outcome genuinely open.
  B4 vs B1 (both): composition of the two above.

SCALE ANCHORS ARE ANCHORS, NOT DVs. `rollout_at_ceiling_frac_mean` reads ~0.9667 in
every arm of the smoke and `rollout_max_over_ceiling_mean` reads 1.0000: the clamp is
ON in every arm (it is a 936-regime constant, not a manipulated dimension) and it
binds on this config, so identical pin statistics across arms are the EXPECTED reading
-- they certify that every cell was measured under the same binding clamp, as the 936a
autopsy's prescription (b) asked. No criterion routes on them. `score_range_mean` (the
within-tick cross-candidate range of the FINAL score) and `dispersion_to_range_ratio`
(sqrt of the TEMPORAL committed variance over it) are recorded so the temporal
partition can be read against the regime's own cross-candidate score scale
(failure_autopsy_V3-EXQ-571b section 2b); a ratio far above 571b's 0.44-3.18 band is
the signature of a drifting broadcast offset dominating the temporal share.

RESIDUE RECRUITMENT IS RECORDED, NOT HEADLINED (red-team finding 8, accepted).
`residue_rbf_active_centers` saturates at 32/32 within the first fed episode, so at
full scale it restates "was update_residue called" and apportions nothing. It stays
in the manifest (after P0, per episode, at end) because the autopsy asked for it; the
headline residue readouts are the per-episode time-to-saturation
(`residue_rbf_first_saturation_episode`) and the residue channel's cross-candidate
share, which do not saturate by construction.

WHICH STAGE OF SELECTION THE PARTITION CHARACTERISES (red-team pass 2, N1 --
BLOCKING on a measurement, resolved by recording it). The 936 lineage's CONFIG_FLAGS
set use_modulatory_shortlist_then_modulate=True, so selection is two-stage: the
PRIMARY score (the routed partition's channels) builds an eligibility shortlist
(raw_scores <= best + 0.25 * range, e3_selector.py:3599-3684) and the final commit
is the argmin of the MODULATORY accumulator within it (e3_selector.py:3766-3794)
unless the shortlist is inactive or a singleton. The partition therefore
characterises the primary stage. Every fresh selection records
modulatory_shortlist_active / modulatory_shortlist_size (written by select() into
last_score_diagnostics, e3_selector.py:3835-3837; previously never persisted), and
C6 (recorded, not routing) summarises B1: final_commit_by_primary_frac >= 0.9 in every
seed means the occupant C1/C2 name drives the commit; otherwise it drives
eligibility and the within-shortlist stage is unmeasured by this partition. Both
monopoly labels' notes say to read them with C6. Also recorded per cell (N2):
partition_coverage = sum of component cross-candidate variances / cross-candidate
variance of the FINAL score (< 1 = an omitted additive channel; > 1 = cancellation);
(N3): max_component_abs_mean, and liveness now needs a relative share >= 1e-3 as
well as the absolute floor; (N4): C4's own exposure and commitment-holding
asymmetries are stated in exposure_matching_note with the recorded n_latched means.

SLEEP DRIVER: none (no sleep flags set; 936-regime parity).
red-team (opus): pass 1 BLOCKING -> routed DV redesigned to the cross-candidate
partition; pass 2 BLOCKING on one measurement (N1, shortlist stage never persisted)
-> recorded per cell + C6; N2/N3/N4 CONTESTED -> recorded (coverage, magnitude,
relative liveness floor, C4 note). Verdicts and model in the queue entry note.
EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence scoring.
ASCII-only output (repo rule).
"""

from __future__ import annotations

import argparse
import datetime
import math
import random
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.fresh_select import (  # noqa: E402
    FreshSelectCounter,
    FreshSelectProbe,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.baselines.mech439_f_variance_share import (  # noqa: E402
    CONTRASTIVE_BATCH_K,
    E2_CONTRASTIVE_LR,
    E2_TRAIN_EVERY_K_TICKS,
    ENV_KWARGS,
    MAX_GRAD_NORM,
    OFF_ARM_FLAGS,
    P0_WARMUP_EPISODES,
    SD056_WEIGHT,
    SEEDS,
    STEPS_PER_EPISODE,
    TRANSITION_BUFFER_MAX,
    make_agent_kwargs,
    make_env,
    off_path_config_slice,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_571c_e3_variance_monopoly_presence_936_regime"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-439"]

# The clamp IS set: e2_rollout_output_norm_clamp_enabled=True lives in the 936
# lineage's CONFIG_FLAGS, applied via kwargs.update(CONFIG_FLAGS) inside
# make_agent_kwargs() -- the **kwargs-splat indirection the lint documents as its
# own blind spot. The Step 2.5a probe read the flag back True (ratio 2.0) off
# agent.e2.config, and every cell re-asserts it (clamp_config_landed).
SD056_ROLLOUT_CLAMP_EXEMPT = (
    "clamp set via CONFIG_FLAGS dict-splat in "
    "experiments/_lib/baselines/mech439_f_variance_share.py::make_agent_kwargs "
    "(936-regime parity); landing re-asserted per cell as clamp_config_landed"
)

# Each readiness predicate measures EXACTLY the quantity that must be non-degenerate
# for its criterion to mean anything: the ROUTED partition's own denominator and
# its dimensionality (n_live_channels -- the red-team finding 5 gap: a single live
# channel satisfies a sum floor and makes any share criterion vacuous), the sample
# count the variances are taken over, and two config-landing identities.
# Reachability was demonstrated by the Step 2.5a probe (55 fresh selections in 120
# env steps fed, 17 starved) and by 936a (201-212 fresh selections per cell against
# the same 60 floor).
ANCHOR_REACHABILITY_EXEMPT = (
    "each predicate IS the degeneracy definition for the criterion it gates "
    "(routed-partition denominator / live-channel count / sample count / "
    "config-landing identity); reachability demonstrated by the Step 2.5a probe on "
    "this exact config and by 936a's 201-212 fresh selections per cell"
)

# --- Pre-registered thresholds (constants, never derived from this run) ------
F_MONOPOLY_THRESHOLD = 0.85   # 936a's own monopoly bar; 571b left it here too
MIN_FRESH_SELECTIONS = 60     # 936a/571b decomp-sample floor
MIN_TOTAL_VARIANCE = 1e-9     # 936a/571b non-degeneracy floor (TEMPORAL partition)
# Cross-candidate spreads live at the score_range scale (~1e-3..1e-2 measured in
# the probe and the smoke), whose square is ~1e-6..1e-4; 1e-12 is six orders below
# that. NUMERICAL FLOOR, corrected after red-team pass 2 (N3): the components are
# float32 (torch default) read via .item(), so quantization-induced cross-candidate
# variance is ~ULP^2/12 -- 7.6e-14 at magnitude ~15 (harm_weighted) and 3.0e-13 at
# ~22 (residue_weighted, after 2 P0 episodes) -- i.e. the absolute floor holds by
# ~3x today and shrinks as magnitude^2 (false-live above magnitude ~64). Liveness is
# therefore ALSO gated on a RELATIVE share (MIN_LIVE_CHANNEL_SHARE), and each cell
# records max_component_abs_mean so a later reader can recompute the quantization
# floor for the magnitudes that actually occurred.
MIN_XCAND_TOTAL_VARIANCE = 1e-12   # routed-partition denominator floor
MIN_LIVE_CHANNEL_VARIANCE = 1e-12  # absolute liveness floor (cross-candidate variance)
MIN_LIVE_CHANNEL_SHARE = 1e-3      # relative liveness floor: >= 0.1% of the routed partition
MIN_LIVE_CHANNELS = 2              # a share needs a contest to mean anything
FINAL_COMMIT_PRIMARY_FLOOR = 0.9   # C6 (recorded): fraction of fresh selections whose final
                                   # commit was decided at the primary (routed) stage
PIN_EPS = 1e-6                     # relative tolerance for "at the clamp ceiling"

# 936a's P1 measurement budget: equalises the fresh-selection sample across arms BY
# CONSTRUCTION (the starved arms hold longer, so they need more env steps -- and
# therefore get more exposure; see the docstring).
N_FRESH_SELECT_TARGET = 200
P1_EPISODE_CAP = 40

# The TRUE additive terms of the E3 score (e3_selector.py:1310, 1375-1383).
SCORE_COMPONENTS = (
    "f_weighted", "harm_weighted", "residue_weighted",
    "benefit_weighted", "novelty_weighted", "goal_weighted",
)
# EXACTLY V3-EXQ-571's / 936a's list (raw unweighted `f`), kept comparable.
LEGACY_571_COMPONENTS = (
    "f", "harm_weighted", "residue_weighted",
    "benefit_weighted", "novelty_weighted", "goal_weighted",
)
F_COMPONENTS = ("f_weighted", "harm_weighted")
LEGACY_F_COMPONENTS = ("f", "harm_weighted")

# Components identically zero under this config, with the source-traced reason
# (red-team finding 1). A cell records whichever of these it actually measures as
# zero on BOTH axes; any OTHER zero component is recorded as unexplained.
STRUCTURAL_ZERO_REASONS = {
    "novelty_weighted": (
        "hardcoded constant _dc_novelty_w = 0.0 (e3_selector.py:1295); the MECH-111 "
        "broadcast branch was deleted 2026-05-25 as argmin-invariant"
    ),
    "benefit_weighted": (
        "benefit branch gated on _benefit_samples_seen >= warmup, fed only by "
        "record_benefit_sample(), which has no callers in ree_core or any driver"
    ),
    "goal_weighted": (
        "gated on goal_state.is_active(); this driver seeds z_goal with "
        "benefit_exposure=0.0 (936a parity) so z_goal never leaves zero-init"
    ),
}

ARMS: List[Dict[str, Any]] = [
    {"id": "B1_936_regime_fed_warmup",     "feed_residue": True,  "warmup": True,  "load_bearing": True},
    {"id": "B2_936_env_starved_warmup",    "feed_residue": False, "warmup": True,  "load_bearing": False},
    {"id": "B3_936_env_fed_no_warmup",     "feed_residue": True,  "warmup": False, "load_bearing": False},
    {"id": "B4_936_env_starved_no_warmup", "feed_residue": False, "warmup": False, "load_bearing": False},
]
LOAD_BEARING_ARM = "B1_936_regime_fed_warmup"
STARVED_ARM = "B2_936_env_starved_warmup"
NO_WARMUP_ARM = "B3_936_env_fed_no_warmup"
BOTH_ARM = "B4_936_env_starved_no_warmup"

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1_CAP = 2
DRY_RUN_STEPS = 60
DRY_RUN_FRESH_TARGET = 6

_FRESH_SELECT = FreshSelectProbe("exq571c")
_ZG = ZGoalStreamAccumulator()
# The LAST cell's agent, kept alive only so the manifest stamper can record
# `enabled_default_off_flags` off a live `.config` (every arm shares one config, so
# one agent is representative). The z_goal liveness block itself comes from _ZG,
# which accumulates across all cells and takes precedence in the stamper.
_LAST_AGENT: Dict[str, Any] = {"agent": None}


# --------------------------------------------------------------------------- #
# Readiness preconditions (regime-conditioned, per arm)                        #
# --------------------------------------------------------------------------- #

PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="decomp_samples_sufficient",
        description=(
            "genuine fresh E3 selections per cell (latch-gated, never per-env-step) "
            "-- the true denominator behind every variance share"
        ),
        control="worst cell across this arm's seeds; FreshSelectProbe-gated reads",
        threshold=float(MIN_FRESH_SELECTIONS),
        direction="lower",
    ),
    PreconditionSpec(
        name="xcand_total_variance_nondegenerate",
        description=(
            "sum over components of the mean within-tick cross-candidate variance -- "
            "literally the denominator the ROUTED share criterion divides by"
        ),
        control="worst cell across this arm's seeds",
        threshold=float(MIN_XCAND_TOTAL_VARIANCE),
        direction="lower",
    ),
    PreconditionSpec(
        name="n_live_channels",
        description=(
            "number of components whose mean cross-candidate variance clears "
            "MIN_LIVE_CHANNEL_VARIANCE -- a share needs >= 2 live channels to be a "
            "contest rather than a restatement of the partition's dimensionality"
        ),
        control="worst cell across this arm's seeds",
        threshold=float(MIN_LIVE_CHANNELS),
        direction="lower",
    ),
    PreconditionSpec(
        name="temporal_committed_total_variance_nondegenerate",
        description=(
            "sum of per-component TEMPORAL variances at the committed candidate -- the "
            "denominator of the recorded 571b-comparable partition (not routed here)"
        ),
        control="worst cell across this arm's seeds",
        threshold=float(MIN_TOTAL_VARIANCE),
        direction="lower",
    ),
    PreconditionSpec(
        name="clamp_config_landed",
        description=(
            "the rollout clamp flag actually reached agent.e2.config in every cell of "
            "this arm (REEConfig.from_dims swallows unknown kwargs silently, so this "
            "is the only proof the 936-regime clamp exists in the cell)"
        ),
        control="all cells of the arm; 1.0 = every cell's live clamp is True at ratio 2.0",
        threshold=1.0,
        direction="lower",
    ),
    PreconditionSpec(
        name="residue_protocol_landed",
        description=(
            "the arm's residue-feeding protocol was executed as declared: fed arms "
            "called update_residue on EVERY env step, starved arms on NONE"
        ),
        control="all cells of the arm; 1.0 = n_update_residue_calls matches the declaration",
        threshold=1.0,
        direction="lower",
    ),
]
# Floors that are met AT the threshold (>=), as the indexer recomputes them.
GEQ_PRECONDITIONS = {
    "decomp_samples_sufficient", "n_live_channels", "clamp_config_landed",
    "residue_protocol_landed",
}


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

def config_slice_for(
    arm: Dict[str, Any], p0: int, p1_cap: int, steps: int, fresh_target: int
) -> Dict[str, Any]:
    """Every readout-affecting constant, declared.

    Built ON the 936 lineage's canonical OFF slice (env, SD-056 training scheme,
    CONFIG_FLAGS incl. the clamp, OFF arm flags) so the regime is declared by the
    same module that constructs it. The protocol dimensions and the cell readout
    constants are added because this fingerprint is cross-driver reusable
    (include_driver_script_in_hash=False) and a consumer with different values
    must MISS rather than silently read readouts computed under another scheme.
    """
    base = off_path_config_slice()
    return {
        "env_kwargs": base["env_kwargs"],
        "sd056_training": base["sd056_training"],
        "config_flags": base["config_flags"],
        "off_arm_flags": base["off_arm_flags"],
        "schedule": {
            "p0_warmup_episodes": int(p0 if arm["warmup"] else 0),
            "p1_episode_cap": int(p1_cap),
            "steps_per_episode": int(steps),
            "fresh_select_target": int(fresh_target),
        },
        "protocol": {
            "feed_residue_per_step": bool(arm["feed_residue"]),
            "p0_warmup": bool(arm["warmup"]),
            "update_z_goal_per_step": True,   # 936a parity (benefit_exposure=0.0)
        },
        "cell_readout_constants": {
            "f_monopoly_threshold": F_MONOPOLY_THRESHOLD,
            "min_fresh_selections": MIN_FRESH_SELECTIONS,
            "min_total_variance": MIN_TOTAL_VARIANCE,
            "min_xcand_total_variance": MIN_XCAND_TOTAL_VARIANCE,
            "min_live_channel_variance": MIN_LIVE_CHANNEL_VARIANCE,
            "min_live_channel_share": MIN_LIVE_CHANNEL_SHARE,
            "final_commit_primary_floor": FINAL_COMMIT_PRIMARY_FLOOR,
            "min_live_channels": MIN_LIVE_CHANNELS,
            "pin_eps": PIN_EPS,
            "score_components": list(SCORE_COMPONENTS),
            "legacy571_components": list(LEGACY_571_COMPONENTS),
        },
    }


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    # Every arm is 936a's ARM_OFF agent; arms differ ONLY in the stepping protocol.
    cfg = REEConfig.from_dims(**make_agent_kwargs(env, OFF_ARM_FLAGS))
    return REEAgent(cfg)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None or not torch.is_tensor(v):
        return None
    return v.float().unsqueeze(0) if v.dim() == 1 else v.float()


def _mean(xs: Sequence[float], default: float = 0.0) -> float:
    return float(statistics.fmean(xs)) if xs else default


def _pool_variance_share(
    series: Dict[str, List[float]], components: Sequence[str], f_components: Sequence[str]
) -> Tuple[float, Dict[str, float], float]:
    """V3-EXQ-571's TEMPORAL variance fraction, reproduced exactly (covariance
    retained in the denominator -- can exceed 1.0; recorded, never routed)."""
    n = 0
    for comp in components:
        n = max(n, len(series.get(comp, [])))
    if n < 2:
        return 0.0, {c: 0.0 for c in components}, 0.0
    arrays: Dict[str, List[float]] = {}
    for comp in components:
        s = list(series.get(comp, []))
        if len(s) < n:
            s = s + [0.0] * (n - len(s))
        arrays[comp] = s[:n]
    total_per_step = [float(sum(arrays[c][i] for c in components)) for i in range(n)]
    total_var = float(statistics.pvariance(total_per_step)) + 1e-12
    fractions: Dict[str, float] = {}
    for comp in components:
        v = float(statistics.pvariance(arrays[comp]))
        fractions[comp] = (v / total_var) if math.isfinite(v) else 0.0
    f_share = float(sum(fractions.get(c, 0.0) for c in f_components))
    return f_share, fractions, total_var


def _temporal_committed_share(
    series: Dict[str, List[float]], components: Sequence[str], f_components: Sequence[str]
) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """571b's routed statistic, RECORDED here: the TEMPORAL variance of each
    component's value AT THE SELECTED candidate, sum-of-variances denominator (a
    genuine partition bounded [0,1]). Returns (f_share, fractions, total, variances)."""
    variances: Dict[str, float] = {}
    for comp in components:
        s = series.get(comp, [])
        v = float(statistics.pvariance(s)) if len(s) > 1 else 0.0
        variances[comp] = v if math.isfinite(v) else 0.0
    total = float(sum(variances.values()))
    if total <= 0.0:
        return 0.0, {c: 0.0 for c in components}, 0.0, variances
    fractions = {k: (v / total) for k, v in variances.items()}
    f_share = float(sum(fractions.get(c, 0.0) for c in f_components))
    return f_share, fractions, total, variances


def _xcand_partition(
    var_series: Dict[str, List[float]], components: Sequence[str], f_components: Sequence[str]
) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """THE ROUTED statistic: for each component, the mean over P1 selections of its
    WITHIN-TICK variance ACROSS THE CANDIDATE SET; shares are those means divided by
    their sum (a genuine partition bounded [0,1], summing to 1.0).
    Returns (f_share, shares, total, per-component means)."""
    means: Dict[str, float] = {}
    for comp in components:
        s = [v for v in var_series.get(comp, []) if math.isfinite(v)]
        means[comp] = _mean(s)
    total = float(sum(means.values()))
    if total <= 0.0:
        return 0.0, {c: 0.0 for c in components}, 0.0, means
    shares = {k: (v / total) for k, v in means.items()}
    f_share = float(sum(shares.get(c, 0.0) for c in f_components))
    return f_share, shares, total, means


def _top_channel(fractions: Dict[str, float]) -> Tuple[str, float]:
    """Deterministic: ties broken by sorted component name."""
    if not fractions:
        return "none", 0.0
    top = sorted(fractions.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[0]
    return str(top[0]), float(top[1])


def _modal(tops: Sequence[str]) -> Tuple[str, bool]:
    """Deterministic mode (sorted by (-count, name)) plus unanimity."""
    if not tops:
        return "none", False
    counts = Counter(tops)
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return str(best), len(counts) == 1


def _partition_disagreement(
    temporal_fr: Dict[str, float], xcand_fr: Dict[str, float], components: Sequence[str]
) -> Dict[str, Any]:
    """The 925-autopsy diagnostic: how far the temporal and cross-candidate
    partitions disagree. Total-variation distance in [0, 1] (0 = identical)."""
    tv = 0.5 * float(sum(abs(temporal_fr.get(c, 0.0) - xcand_fr.get(c, 0.0)) for c in components))
    t_top, t_share = _top_channel(temporal_fr)
    x_top, x_share = _top_channel(xcand_fr)
    return {
        "total_variation_distance": tv,
        "temporal_top_channel": t_top,
        "temporal_top_share": t_share,
        "xcand_top_channel": x_top,
        "xcand_top_share": x_share,
        "top_channel_differs": bool(t_top != x_top),
        "per_component": {
            c: {"temporal": float(temporal_fr.get(c, 0.0)), "xcand": float(xcand_fr.get(c, 0.0))}
            for c in components
        },
    }


def _pin_stats(traj: Any, clamp_ratio: float) -> Optional[Dict[str, float]]:
    """Read-only clamp-binding statistics off the COMMITTED rollout (571b)."""
    ws = getattr(traj, "world_states", None) if traj is not None else None
    if not ws or len(ws) < 2:
        return None
    try:
        n0 = float(ws[0].detach().norm(dim=-1).reshape(-1)[0].item())
    except Exception:
        return None
    if not math.isfinite(n0) or n0 <= 0.0:
        return None
    ceiling = clamp_ratio * n0
    norms: List[float] = []
    for t in range(1, len(ws)):
        try:
            norms.append(float(ws[t].detach().norm(dim=-1).reshape(-1)[0].item()))
        except Exception:
            continue
    if not norms:
        return None
    tol = ceiling * PIN_EPS
    n_at = sum(1 for v in norms if math.isfinite(v) and abs(v - ceiling) <= tol)
    n_above = sum(1 for v in norms if math.isfinite(v) and v > ceiling + tol)
    finite = [v for v in norms if math.isfinite(v)]
    return {
        "zw0_norm": n0,
        "zw_max_norm": float(max(finite)) if finite else 0.0,
        "at_ceiling_frac": float(n_at) / float(len(norms)),
        "above_ceiling_frac": float(n_above) / float(len(norms)),
        "max_over_ceiling": (float(max(finite)) / ceiling) if finite else 0.0,
    }


def _residue_rbf(agent: REEAgent) -> Tuple[int, int]:
    """(active centres, total centres) of the residue RBF field; (-1, -1) if unreadable."""
    try:
        rf = getattr(agent, "residue_field", None)
        rbf = getattr(rf, "rbf_field", None) if rf is not None else None
        mask = getattr(rbf, "active_mask", None) if rbf is not None else None
        if mask is None:
            return -1, -1
        return int(mask.sum().item()), int(mask.numel())
    except Exception:
        return -1, -1


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    """SD-056 online contrastive step, verbatim from the 936a driver (regime parity)."""
    if len(buffer) < CONTRASTIVE_BATCH_K:
        return None
    batch = rng.sample(list(buffer), CONTRASTIVE_BATCH_K)
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K, actions=actions_K,
        z_world_1_targets=z1_K, simulation_mode=False,
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


# --------------------------------------------------------------------------- #
# Per-cell run                                                                 #
# --------------------------------------------------------------------------- #

def run_cell(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episode_cap: int,
    steps_per_episode: int,
    fresh_target: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    arm_id = str(arm["id"])
    feed = bool(arm["feed_residue"])
    p0 = int(p0_episodes) if arm["warmup"] else 0
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    slice_for_cell = config_slice_for(arm, p0_episodes, p1_episode_cap, steps_per_episode, fresh_target)

    with arm_cell(
        seed,
        config_slice=slice_for_cell,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        # arm_cell's entry already performed the complete RNG reset.
        env = make_env(seed)          # arm-INDEPENDENT: arms share a layout per seed
        agent = _make_agent(env)
        agent.e3.e3_score_decomp_enabled = True   # diagnostics-only; selection bit-identical
        e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

        clamp_live = bool(getattr(agent.e2.config, "e2_rollout_output_norm_clamp_enabled", False))
        ratio_live = float(getattr(agent.e2.config, "e2_rollout_output_norm_clamp_ratio", 2.0))

        transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=TRANSITION_BUFFER_MAX
        )
        sample_rng = random.Random(seed)
        total_train_eps = p0 + int(p1_episode_cap)

        all_components = tuple(sorted(set(SCORE_COMPONENTS) | set(LEGACY_571_COMPONENTS)))
        pool_series: Dict[str, List[float]] = {}        # temporal, pool mean (571 method)
        sel_series: Dict[str, List[float]] = {}         # temporal, committed candidate (571b)
        xcand_var_series: Dict[str, List[float]] = {}   # ROUTED: within-tick cross-candidate var
        score_xcand_vars: List[float] = []                # N2: cross-candidate variance of the FINAL score
        shortlist_active_flags: List[bool] = []           # N1: modulatory shortlist engaged this selection?
        shortlist_sizes: List[int] = []                   # N1: shortlist size when engaged (0 otherwise)
        final_commit_by_primary: List[bool] = []          # N1: commit decided by the primary (routed) stage?
        xcand_range_series: Dict[str, List[float]] = {}
        n_candidates_seen: List[int] = []
        score_ranges: List[float] = []
        at_ceiling_vals: List[float] = []
        above_ceiling_vals: List[float] = []
        max_over_ceiling_vals: List[float] = []
        zw0_vals: List[float] = []
        zw_max_vals: List[float] = []
        lambda_eff_vals: List[float] = []
        selected_class_counts: Counter = Counter()

        fs = FreshSelectCounter()
        n_p1_ticks = 0
        n_ticks_total = 0
        n_p1_fresh = 0
        n_update_residue_calls = 0
        n_harm_steps_total = 0
        harm_p1_abs_sum = 0.0
        harm_p1_ticks = 0
        n_contrastive_steps = 0
        n_contrastive_steps_total = 0
        p1_episodes_run = 0
        target_met = False
        residue_active_after_p0, residue_num_centers = _residue_rbf(agent)  # fresh-init
        residue_active_trace: List[int] = []

        for ep in range(total_train_eps):
            is_p1 = ep >= p0
            phase_label = "P1" if is_p1 else "P0"
            if is_p1:
                p1_episodes_run += 1
                if ep == p0:
                    residue_active_after_p0, _ = _residue_rbf(agent)

            _, obs_dict = env.reset()
            agent.reset()   # does NOT reset the residue field (agent.py: invariant)

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
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

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
                    agent.update_z_goal(benefit_exposure=0.0, drive_level=max(0.0, 1.0 - energy))

                # --- FRESHNESS-GATED SELECTION (E3 last_* attributes LATCH) ----
                with _FRESH_SELECT.watch(agent) as _sel:
                    action = agent.select_action(candidates, ticks)
                fresh_select = _sel.fresh
                n_ticks_total += 1
                if is_p1:
                    n_p1_ticks += 1
                    fs.record(fresh_select)

                if is_p1 and fresh_select:
                    decomp = agent.e3.last_score_decomp or {}
                    per_cand = decomp.get("per_candidate")
                    sel_idx = decomp.get("selected_idx")
                    # N1 (red-team pass 2): under use_modulatory_shortlist_then_modulate the
                    # final commit is argmin over the MODULATORY accumulator among a shortlist
                    # built by the primary score (e3_selector.py:3599-3684, 3766-3794); the
                    # routed partition characterises that PRIMARY stage. Record per selection
                    # whether the shortlist engaged and its size, so a later reader knows at
                    # which stage the commit was decided. Read-only; the diagnostics dict is
                    # rewritten by select() itself (defaults at e3_selector.py:3265-3266).
                    _diag = getattr(agent.e3, "last_score_diagnostics", None) or {}
                    _sl_active = bool(_diag.get("modulatory_shortlist_active", False))
                    try:
                        _sl_size = int(_diag.get("modulatory_shortlist_size", 0) or 0)
                    except (TypeError, ValueError):
                        _sl_size = 0
                    shortlist_active_flags.append(_sl_active)
                    shortlist_sizes.append(_sl_size if _sl_active else 0)
                    final_commit_by_primary.append((not _sl_active) or _sl_size <= 1)
                    if isinstance(per_cand, list) and per_cand:
                        dict_cands = [c for c in per_cand if isinstance(c, dict)]
                        if dict_cands:
                            n_p1_fresh += 1
                            n_candidates_seen.append(len(dict_cands))
                            for comp in all_components:
                                vals = []
                                for c in dict_cands:
                                    try:
                                        v = float(c.get(comp, 0.0))
                                    except (TypeError, ValueError):
                                        continue
                                    if math.isfinite(v):
                                        vals.append(v)
                                # temporal (571): per-selection mean across candidates
                                pool_series.setdefault(comp, []).append(_mean(vals))
                                # ROUTED: within-tick spread ACROSS candidates
                                if len(vals) > 1:
                                    xv = float(statistics.pvariance(vals))
                                    xr = float(max(vals) - min(vals))
                                else:
                                    xv, xr = 0.0, 0.0
                                xcand_var_series.setdefault(comp, []).append(xv if math.isfinite(xv) else 0.0)
                                xcand_range_series.setdefault(comp, []).append(xr if math.isfinite(xr) else 0.0)
                            if (
                                isinstance(sel_idx, int)
                                and 0 <= sel_idx < len(per_cand)
                                and isinstance(per_cand[sel_idx], dict)
                            ):
                                chosen = per_cand[sel_idx]
                                for comp in all_components:
                                    try:
                                        v = float(chosen.get(comp, 0.0))
                                    except (TypeError, ValueError):
                                        v = 0.0
                                    sel_series.setdefault(comp, []).append(v if math.isfinite(v) else 0.0)
                                try:
                                    le = float(chosen.get("lambda_eff", 0.0))
                                    if math.isfinite(le):
                                        lambda_eff_vals.append(le)
                                except (TypeError, ValueError):
                                    pass

                    ls = getattr(agent.e3, "last_scores", None)
                    if ls is not None and getattr(ls, "numel", lambda: 0)() > 1:
                        try:
                            r = float((ls.max() - ls.min()).item())
                            if math.isfinite(r):
                                score_ranges.append(r)
                            # N2: cross-candidate variance of the FINAL score, so the routed
                            # partition's coverage (sum of component variances / Var(score))
                            # is auditable: < 1 = an omitted additive channel, > 1 = net
                            # cancellation among the declared ones.
                            _vals_ls = [float(v) for v in ls.detach().reshape(-1).tolist()]
                            if len(_vals_ls) > 1:
                                _sv = float(statistics.pvariance(_vals_ls))
                                if math.isfinite(_sv):
                                    score_xcand_vars.append(_sv)
                        except Exception:
                            pass

                    ps = _pin_stats(getattr(agent.e3, "_last_selected_trajectory", None), ratio_live)
                    if ps is not None:
                        at_ceiling_vals.append(ps["at_ceiling_frac"])
                        above_ceiling_vals.append(ps["above_ceiling_frac"])
                        max_over_ceiling_vals.append(ps["max_over_ceiling"])
                        zw0_vals.append(ps["zw0_norm"])
                        zw_max_vals.append(ps["zw_max_norm"])

                    try:
                        cls = int(action.argmax(dim=-1).reshape(-1)[0].item())
                        selected_class_counts[cls] += 1
                    except Exception:
                        pass

                if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                    pending_capture = (
                        latent.z_world.detach().reshape(-1).clone(),
                        action.detach().reshape(-1).clone(),
                    )

                # SD-056 online contrastive training on EVERY arm (936-regime parity).
                if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                    loss_val = _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)
                    if loss_val is not None and math.isfinite(loss_val):
                        n_contrastive_steps_total += 1
                        if is_p1:
                            n_contrastive_steps += 1

                _, harm_signal, done, info, next_obs_dict = env.step(action)
                hv = float(harm_signal)
                if math.isfinite(hv) and hv < 0.0:
                    n_harm_steps_total += 1
                if is_p1 and math.isfinite(hv):
                    harm_p1_abs_sum += abs(hv)
                    harm_p1_ticks += 1

                # --- THE RESIDUE-PROTOCOL DIMENSION --------------------------
                # Fed arms: the 936a post-action path, verbatim (driver:760).
                # Starved arms: nothing, exactly as 571b (0 calls in that driver).
                if feed:
                    with torch.no_grad():
                        agent.update_residue(
                            harm_signal=hv, world_delta=None,
                            hypothesis_tag=False, owned=True,
                        )
                    n_update_residue_calls += 1

                z_self_prev = latent.z_self.detach()
                action_prev = action
                obs_dict = next_obs_dict
                tick_in_ep += 1
                if done:
                    break

            fs.flush()
            residue_active_trace.append(_residue_rbf(agent)[0])

            # P1 is short (936a: 7-16 episodes to 200 fresh selections) and ends on
            # a break, so print every P1 episode to keep the runner's progress live.
            if ep == 0 or is_p1 or (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
                print(
                    f"  [train] arm={arm_id} seed={seed} phase={phase_label} "
                    f"ep {ep + 1}/{total_train_eps} fresh={n_p1_fresh}/{fresh_target} "
                    f"latched={fs.n_latched} rbf_active={residue_active_trace[-1]}",
                    flush=True,
                )

            if is_p1 and n_p1_fresh >= fresh_target:
                target_met = True
                print(
                    f"  [p1-done] arm={arm_id} seed={seed} fresh={n_p1_fresh} "
                    f"after {p1_episodes_run} P1 episode(s)",
                    flush=True,
                )
                break

        _ZG.observe(agent)
        _LAST_AGENT["agent"] = agent
        residue_active_end, residue_num_centers = _residue_rbf(agent)

        # --- ROUTED partition: within-tick cross-candidate variance ---------------
        x_f, x_fr, x_total, x_means = _xcand_partition(xcand_var_series, SCORE_COMPONENTS, F_COMPONENTS)
        x_top, x_top_share = _top_channel(x_fr)
        lx_f, lx_fr, _, _ = _xcand_partition(xcand_var_series, LEGACY_571_COMPONENTS, LEGACY_F_COMPONENTS)
        x_range_means = {c: _mean(xcand_range_series.get(c, [])) for c in all_components}

        # --- RECORDED temporal partitions (571b committed; 571/936a pool; both sets) --
        t_f, t_fr, t_total, t_vars = _temporal_committed_share(sel_series, SCORE_COMPONENTS, F_COMPONENTS)
        t_top, t_top_share = _top_channel(t_fr)
        lt_f, lt_fr, _, _ = _temporal_committed_share(sel_series, LEGACY_571_COMPONENTS, LEGACY_F_COMPONENTS)
        pool_f, pool_fr, pool_total_var = _pool_variance_share(pool_series, SCORE_COMPONENTS, F_COMPONENTS)
        pool_top, pool_top_share = _top_channel(pool_fr)
        lpool_f, lpool_fr, lpool_total_var = _pool_variance_share(
            pool_series, LEGACY_571_COMPONENTS, LEGACY_F_COMPONENTS
        )
        lpool_top, lpool_top_share = _top_channel(lpool_fr)

        disagreement = _partition_disagreement(t_fr, x_fr, SCORE_COMPONENTS)

        # --- partition dimensionality (red-team findings 1 + 5) -----------------------
        live_channels = [
            c for c in SCORE_COMPONENTS
            if x_means.get(c, 0.0) > MIN_LIVE_CHANNEL_VARIANCE and x_fr.get(c, 0.0) >= MIN_LIVE_CHANNEL_SHARE
        ]
        score_xcand_var_mean = _mean(score_xcand_vars)
        partition_coverage = (x_total / score_xcand_var_mean) if score_xcand_var_mean > 0 else 0.0
        max_component_abs_mean = max(
            [abs(_mean(pool_series.get(c, []))) for c in SCORE_COMPONENTS] or [0.0]
        )
        n_sl = len(final_commit_by_primary)
        shortlist_active_frac = (sum(1 for a in shortlist_active_flags if a) / n_sl) if n_sl else 0.0
        shortlist_size_mean = _mean([float(v) for v in shortlist_sizes]) if n_sl else 0.0
        final_commit_by_primary_frac = (sum(1 for b in final_commit_by_primary if b) / n_sl) if n_sl else 0.0
        structurally_zero = {
            c: STRUCTURAL_ZERO_REASONS.get(
                c, "measured zero on BOTH the cross-candidate and temporal axes; cause not pre-identified"
            )
            for c in SCORE_COMPONENTS
            if x_means.get(c, 0.0) == 0.0 and t_vars.get(c, 0.0) == 0.0
        }
        below_live_floor = [
            c for c in SCORE_COMPONENTS
            if c not in live_channels and c not in structurally_zero
        ]

        temporal_channel_means = {c: _mean(sel_series.get(c, [])) for c in all_components}
        e3cfg = getattr(agent.e3, "config", None)
        score_weights_live = {
            "f_weight": float(getattr(e3cfg, "f_weight", float("nan"))),
            "rho_residue": float(getattr(e3cfg, "rho_residue", float("nan"))),
            "lambda_eff_mean": _mean(lambda_eff_vals),
        }

        n_ent = sum(selected_class_counts.values())
        committed_entropy = 0.0
        if n_ent > 0:
            for cnt in selected_class_counts.values():
                p = cnt / n_ent
                if p > 0:
                    committed_entropy -= p * math.log(p)

        first_sat = -1
        if residue_num_centers > 0:
            for i, v in enumerate(residue_active_trace):
                if v >= residue_num_centers:
                    first_sat = i
                    break

        expected_calls = n_ticks_total if feed else 0
        row: Dict[str, Any] = {
            "arm": arm_id,
            "seed": int(seed),
            "load_bearing_arm": bool(arm["load_bearing"]),
            "feed_residue_per_step": feed,
            "p0_warmup": bool(arm["warmup"]),
            "p0_episodes": int(p0),
            "p1_episodes_run": int(p1_episodes_run),
            "fresh_target_met": bool(target_met),
            "clamp_live_on_e2": clamp_live,
            "clamp_ratio_live": ratio_live,
            "clamp_config_landed": bool(clamp_live and abs(ratio_live - 2.0) < 1e-12),

            # exposure (NOT matched across arms -- see the docstring)
            "n_env_ticks_total": int(n_ticks_total),
            "n_p1_ticks": int(n_p1_ticks),
            "n_p1_decomp_samples": int(n_p1_fresh),
            "n_candidates_mean": _mean(n_candidates_seen),
            "n_update_residue_calls": int(n_update_residue_calls),
            "residue_protocol_landed": bool(n_update_residue_calls == expected_calls),
            "n_harm_steps_total": int(n_harm_steps_total),
            "p1_mean_abs_harm": (harm_p1_abs_sum / harm_p1_ticks) if harm_p1_ticks else 0.0,
            "n_contrastive_steps_p1": int(n_contrastive_steps),
            "n_contrastive_steps_total": int(n_contrastive_steps_total),

            # ===== ROUTED: within-tick CROSS-CANDIDATE partition =====
            "xcand_f_share": float(x_f),
            "xcand_share": {k: float(v) for k, v in x_fr.items()},
            "xcand_total_variance": float(x_total),
            "xcand_var_mean": {k: float(v) for k, v in x_means.items()},
            "xcand_range_mean": {k: float(v) for k, v in x_range_means.items()},
            "xcand_top_channel": x_top,
            "xcand_top_share": float(x_top_share),
            "xcand_legacy571_f_share": float(lx_f),
            "xcand_legacy571_share": {k: float(v) for k, v in lx_fr.items()},
            "n_live_channels": int(len(live_channels)),
            "live_channels": list(live_channels),
            "structurally_zero_components": structurally_zero,
            "below_live_floor_components": below_live_floor,
            # N2: partition validity audit; N3: quantization-floor recompute input
            "partition_coverage": float(partition_coverage),
            "score_xcand_var_mean": float(score_xcand_var_mean),
            "max_component_abs_mean": float(max_component_abs_mean),
            # N1: at which stage was the commit decided?
            "modulatory_shortlist_active_frac": float(shortlist_active_frac),
            "modulatory_shortlist_size_mean": float(shortlist_size_mean),
            "final_commit_by_primary_frac": float(final_commit_by_primary_frac),
            "n_shortlist_reads": int(n_sl),

            # ===== RECORDED: temporal partitions (571b committed; 571/936a pool) =====
            "temporal_committed_f_share": float(t_f),
            "temporal_committed_fractions": {k: float(v) for k, v in t_fr.items()},
            "temporal_committed_variances": {k: float(v) for k, v in t_vars.items()},
            "temporal_committed_total_variance": float(t_total),
            "temporal_committed_top_channel": t_top,
            "temporal_committed_top_share": float(t_top_share),
            "temporal_committed_channel_means": temporal_channel_means,
            "temporal_legacy571_committed_f_share": float(lt_f),
            "temporal_legacy571_committed_fractions": {k: float(v) for k, v in lt_fr.items()},
            "temporal_pool_f_share": float(pool_f),
            "temporal_pool_fractions": {k: float(v) for k, v in pool_fr.items()},
            "temporal_pool_total_variance": float(pool_total_var),
            "temporal_pool_top_channel": pool_top,
            "temporal_pool_top_share": float(pool_top_share),
            "temporal_legacy571_pool_f_share": float(lpool_f),
            "temporal_legacy571_pool_fractions": {k: float(v) for k, v in lpool_fr.items()},
            "temporal_legacy571_pool_total_variance": float(lpool_total_var),
            "temporal_legacy571_pool_top_channel": lpool_top,
            "temporal_legacy571_pool_top_share": float(lpool_top_share),
            "temporal_pool_share_exceeds_unity": bool(pool_top_share > 1.0),

            # ===== the 925 diagnostic: temporal vs cross-candidate disagreement =====
            "partition_disagreement": disagreement,

            # scale anchors (936a prescription (b), as implemented by 571b)
            "score_range_mean": _mean(score_ranges),
            "score_range_max": float(max(score_ranges)) if score_ranges else 0.0,
            "dispersion_to_range_ratio": (
                (math.sqrt(t_total) / _mean(score_ranges)) if score_ranges and _mean(score_ranges) > 0 else 0.0
            ),
            "zw0_norm_mean": _mean(zw0_vals),
            "zw_max_norm_mean": _mean(zw_max_vals),
            "rollout_at_ceiling_frac_mean": _mean(at_ceiling_vals),
            "rollout_above_ceiling_frac_mean": _mean(above_ceiling_vals),
            "rollout_max_over_ceiling_mean": _mean(max_over_ceiling_vals),
            "rollout_max_over_ceiling_max": float(max(max_over_ceiling_vals)) if max_over_ceiling_vals else 0.0,
            "n_pin_samples": int(len(at_ceiling_vals)),

            # residue recruitment (recorded; saturates -- see the docstring)
            "residue_rbf_num_centers": int(residue_num_centers),
            "residue_rbf_active_centers": int(residue_active_end),
            "residue_rbf_active_centers_after_p0": int(residue_active_after_p0),
            "residue_rbf_active_centers_per_episode": [int(x) for x in residue_active_trace],
            "residue_rbf_first_saturation_episode": int(first_sat),

            "score_weights_live": score_weights_live,
            "committed_action_class_entropy": float(committed_entropy),
            "n_committed_classes": int(len(selected_class_counts)),
            "selected_class_counts": {str(k): int(v) for k, v in selected_class_counts.items()},
        }
        row.update(fs.as_dict(n_p1_ticks))

        monopoly_present = bool(row["xcand_top_share"] >= F_MONOPOLY_THRESHOLD)
        row["monopoly_present"] = monopoly_present
        row["monopoly_occupant_is_f"] = bool(monopoly_present and row["xcand_top_channel"] in F_COMPONENTS)
        row["cell_decidable"] = bool(
            row["n_fresh_select"] >= MIN_FRESH_SELECTIONS
            and row["xcand_total_variance"] > MIN_XCAND_TOTAL_VARIANCE
            and row["n_live_channels"] >= MIN_LIVE_CHANNELS
            and row["temporal_committed_total_variance"] > MIN_TOTAL_VARIANCE
        )

        cell.stamp(row)

    if dry_run:
        print(
            f"  [smoke] arm={arm_id} seed={seed} clamp_live={row['clamp_live_on_e2']} "
            f"fed={feed} calls={row['n_update_residue_calls']}/{row['n_env_ticks_total']} "
            f"n_fresh={row['n_fresh_select']} n_latched={row['n_latched']} "
            f"K={row['n_candidates_mean']:.1f} "
            f"ROUTED xcand_top={row['xcand_top_channel']}({row['xcand_top_share']:.6g}) "
            f"xcand_f={row['xcand_f_share']:.6g} live={row['n_live_channels']}{row['live_channels']} "
            f"xcand_total={row['xcand_total_variance']:.4g} | "
            f"temporal_top={row['temporal_committed_top_channel']}"
            f"({row['temporal_committed_top_share']:.6g}) "
            f"tv_dist={row['partition_disagreement']['total_variation_distance']:.4f} "
            f"top_differs={row['partition_disagreement']['top_channel_differs']} | "
            f"score_range={row['score_range_mean']:.4g} "
            f"disp/range={row['dispersion_to_range_ratio']:.3g} "
            f"at_ceil={row['rollout_at_ceiling_frac_mean']:.4f} "
            f"rbf_sat_ep={row['residue_rbf_first_saturation_episode']} "
            f"sl_active={row['modulatory_shortlist_active_frac']:.2f} "
            f"sl_size={row['modulatory_shortlist_size_mean']:.2f} "
            f"primary_stage={row['final_commit_by_primary_frac']:.2f} "
            f"coverage={row['partition_coverage']:.3f} "
            f"max_abs={row['max_component_abs_mean']:.3g}",
            flush=True,
        )

    print(f"verdict: {'PASS' if row['cell_decidable'] else 'FAIL'}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# Aggregation + interpretation                                                 #
# --------------------------------------------------------------------------- #

def _worst_cell(rows: List[Dict[str, Any]], key: str) -> Tuple[float, str]:
    """Minimum plus the offending cell id, so `measured` recomputes exactly."""
    if not rows:
        return 0.0, "none"
    best = min(rows, key=lambda r: float(r.get(key, 0.0)))
    return float(best.get(key, 0.0)), f"{best.get('arm')}/seed{best.get('seed')}"


def _arm_gate(arm: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fresh_w, fresh_c = _worst_cell(rows, "n_fresh_select")
    xvar_w, xvar_c = _worst_cell(rows, "xcand_total_variance")
    live_w, live_c = _worst_cell(rows, "n_live_channels")
    tvar_w, tvar_c = _worst_cell(rows, "temporal_committed_total_variance")
    clamp_ok = 1.0 if rows and all(bool(r["clamp_config_landed"]) for r in rows) else 0.0
    proto_ok = 1.0 if rows and all(bool(r["residue_protocol_landed"]) for r in rows) else 0.0
    measured = {
        "decomp_samples_sufficient": fresh_w,
        "xcand_total_variance_nondegenerate": xvar_w,
        "n_live_channels": live_w,
        "temporal_committed_total_variance_nondegenerate": tvar_w,
        "clamp_config_landed": clamp_ok,
        "residue_protocol_landed": proto_ok,
    }
    overrides = {
        spec.name: bool(measured[spec.name] >= spec.threshold)
        for spec in PRECONDITION_SPECS if spec.name in GEQ_PRECONDITIONS
    }
    ctx = {"id": arm["id"], "feed_residue": arm["feed_residue"], "warmup": arm["warmup"]}
    gate = evaluate_arm_gate(arm["id"], ctx, PRECONDITION_SPECS, measured, met_overrides=overrides)
    offenders = {
        "decomp_samples_sufficient": fresh_c,
        "xcand_total_variance_nondegenerate": xvar_c,
        "n_live_channels": live_c,
        "temporal_committed_total_variance_nondegenerate": tvar_c,
    }
    for p in gate["preconditions"]:
        p["kind"] = "readiness"
        if p["precondition"] in offenders:
            p["offending_cell"] = offenders[p["precondition"]]
    return gate


def _flip_count(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]]) -> Tuple[int, int, List[int]]:
    """Matched-seed occupant flips (ROUTED partition) between two arms, over seeds
    decidable in both."""
    a_by = {int(r["seed"]): r for r in a_rows}
    b_by = {int(r["seed"]): r for r in b_rows}
    n_pairs, n_flip, flipped = 0, 0, []
    for s in sorted(set(a_by) & set(b_by)):
        if not (a_by[s]["cell_decidable"] and b_by[s]["cell_decidable"]):
            continue
        n_pairs += 1
        if a_by[s]["xcand_top_channel"] != b_by[s]["xcand_top_channel"]:
            n_flip += 1
            flipped.append(s)
    return n_pairs, n_flip, flipped


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = list(DRY_RUN_SEEDS if dry_run else SEEDS)
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1_cap = DRY_RUN_P1_CAP if dry_run else P1_EPISODE_CAP
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    fresh_target = DRY_RUN_FRESH_TARGET if dry_run else N_FRESH_SELECT_TARGET

    # Design audit: no arm carries a structurally unsatisfiable gate (785 rule).
    arm_ctxs = [{"id": a["id"], "feed_residue": a["feed_residue"], "warmup": a["warmup"]} for a in ARMS]
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(run_cell(arm, seed, p0, p1_cap, steps, fresh_target, dry_run=dry_run))

    by_arm: Dict[str, List[Dict[str, Any]]] = {a["id"]: [r for r in rows if r["arm"] == a["id"]] for a in ARMS}
    lb_rows = by_arm[LOAD_BEARING_ARM]

    # --- per-arm readiness gates; a red arm never vacates a green one -----------
    arm_gates = [_arm_gate(a, by_arm[a["id"]]) for a in ARMS]
    gate = aggregate_arm_gates(arm_gates)
    lb_gate = next(g for g in arm_gates if g["arm"] == LOAD_BEARING_ARM)
    green = set(gate["green_arms"])

    def _arm_summary(arm_id: str) -> Dict[str, Any]:
        rs = by_arm[arm_id]
        x_tops = [r["xcand_top_channel"] for r in rs]
        x_modal, x_unanimous = _modal(x_tops)
        t_tops = [r["temporal_committed_top_channel"] for r in rs]
        t_modal, t_unanimous = _modal(t_tops)
        return {
            "arm": arm_id,
            "gate_green": arm_id in green,
            "n_cells_decidable": int(sum(1 for r in rs if r["cell_decidable"])),
            # ROUTED
            "xcand_top_channel_modal": x_modal,
            "xcand_top_channel_unanimous": bool(x_unanimous),
            "xcand_top_channel_per_seed": {str(r["seed"]): r["xcand_top_channel"] for r in rs},
            "xcand_modal_channel_share_mean": _mean([r["xcand_share"].get(x_modal, 0.0) for r in rs]),
            "xcand_modal_channel_share_min": (
                float(min([r["xcand_share"].get(x_modal, 0.0) for r in rs])) if rs else 0.0
            ),
            "xcand_modal_channel_share_per_seed": {str(r["seed"]): r["xcand_share"].get(x_modal, 0.0) for r in rs},
            "xcand_top_share_mean": _mean([r["xcand_top_share"] for r in rs]),
            "xcand_f_share_mean": _mean([r["xcand_f_share"] for r in rs]),
            "xcand_f_share_per_seed": {str(r["seed"]): r["xcand_f_share"] for r in rs},
            "xcand_residue_share_mean": _mean([r["xcand_share"].get("residue_weighted", 0.0) for r in rs]),
            "xcand_share_mean_by_component": {
                c: _mean([r["xcand_share"].get(c, 0.0) for r in rs]) for c in SCORE_COMPONENTS
            },
            "n_live_channels_min": int(min([r["n_live_channels"] for r in rs])) if rs else 0,
            "live_channels_union": sorted({c for r in rs for c in r["live_channels"]}),
            "structurally_zero_components_union": sorted({c for r in rs for c in r["structurally_zero_components"]}),
            # N1 / N2 / N3 / N4 readouts
            "partition_coverage_mean": _mean([r["partition_coverage"] for r in rs]),
            "modulatory_shortlist_active_frac_mean": _mean([r["modulatory_shortlist_active_frac"] for r in rs]),
            "modulatory_shortlist_size_mean": _mean([r["modulatory_shortlist_size_mean"] for r in rs]),
            "final_commit_by_primary_frac_mean": _mean([r["final_commit_by_primary_frac"] for r in rs]),
            "final_commit_by_primary_frac_min": (
                float(min([r["final_commit_by_primary_frac"] for r in rs])) if rs else 0.0
            ),
            "n_latched_mean": _mean([r["n_latched"] for r in rs]),
            "max_component_abs_mean_max": (float(max([r["max_component_abs_mean"] for r in rs])) if rs else 0.0),
            # RECORDED temporal
            "temporal_top_channel_modal": t_modal,
            "temporal_top_channel_unanimous": bool(t_unanimous),
            "temporal_committed_top_share_mean": _mean([r["temporal_committed_top_share"] for r in rs]),
            "temporal_committed_f_share_mean": _mean([r["temporal_committed_f_share"] for r in rs]),
            "temporal_committed_residue_share_mean": _mean(
                [r["temporal_committed_fractions"].get("residue_weighted", 0.0) for r in rs]
            ),
            "temporal_legacy571_pool_f_share_mean": _mean([r["temporal_legacy571_pool_f_share"] for r in rs]),
            # disagreement
            "partition_tv_distance_mean": _mean(
                [r["partition_disagreement"]["total_variation_distance"] for r in rs]
            ),
            "partition_top_channel_differs_count": int(
                sum(1 for r in rs if r["partition_disagreement"]["top_channel_differs"])
            ),
            # exposure (NOT matched across arms)
            "n_fresh_select_mean": _mean([r["n_fresh_select"] for r in rs]),
            "fresh_select_yield_mean": _mean([r["fresh_select_yield"] for r in rs]),
            "n_env_ticks_total_mean": _mean([r["n_env_ticks_total"] for r in rs]),
            "n_p1_ticks_mean": _mean([r["n_p1_ticks"] for r in rs]),
            "n_contrastive_steps_p1_mean": _mean([r["n_contrastive_steps_p1"] for r in rs]),
            "n_contrastive_steps_total_mean": _mean([r["n_contrastive_steps_total"] for r in rs]),
            # anchors + residue recruitment
            "rollout_at_ceiling_frac_mean": _mean([r["rollout_at_ceiling_frac_mean"] for r in rs]),
            "rollout_max_over_ceiling_mean": _mean([r["rollout_max_over_ceiling_mean"] for r in rs]),
            "score_range_mean": _mean([r["score_range_mean"] for r in rs]),
            "dispersion_to_range_ratio_mean": _mean([r["dispersion_to_range_ratio"] for r in rs]),
            "residue_rbf_first_saturation_episode_per_seed": {
                str(r["seed"]): r["residue_rbf_first_saturation_episode"] for r in rs
            },
            "residue_rbf_active_centers_mean": _mean([r["residue_rbf_active_centers"] for r in rs]),
        }

    arm_summaries = {a["id"]: _arm_summary(a["id"]) for a in ARMS}
    lb = arm_summaries[LOAD_BEARING_ARM]

    # --- criteria -----------------------------------------------------------------
    lb_unanimous = bool(lb["xcand_top_channel_unanimous"]) and bool(lb_rows)
    c1_monopoly_present = bool(lb_unanimous and lb["xcand_modal_channel_share_mean"] >= F_MONOPOLY_THRESHOLD)
    c2_occupant_is_f = bool(c1_monopoly_present and lb["xcand_top_channel_modal"] in F_COMPONENTS)

    def _flip_criterion(name: str, other_arm: str, dimension: str, role: str) -> Dict[str, Any]:
        n_pairs, n_flip, flipped = _flip_count(lb_rows, by_arm[other_arm])
        passed = bool(n_pairs > 0 and n_flip >= max(1, (n_pairs + 1) // 2))
        return {
            "name": name,
            "load_bearing": False,
            "role": role,
            "passed": passed,
            "dimension": dimension,
            "contrast": f"{LOAD_BEARING_ARM} vs {other_arm} (matched seeds, routed partition)",
            "n_matched_decidable_pairs": int(n_pairs),
            "n_occupant_flips": int(n_flip),
            "flipped_seeds": flipped,
            "detail": (
                f"xcand occupant B1={lb['xcand_top_channel_modal']} vs "
                f"{other_arm}={arm_summaries[other_arm]['xcand_top_channel_modal']}; "
                f"{n_flip}/{n_pairs} matched seeds flip; xcand F share "
                f"{lb['xcand_f_share_mean']:.6g} -> {arm_summaries[other_arm]['xcand_f_share_mean']:.6g}; "
                f"xcand residue share {lb['xcand_residue_share_mean']:.6g} -> "
                f"{arm_summaries[other_arm]['xcand_residue_share_mean']:.6g}"
            ),
        }

    WIRING = (
        "WIRING CHECK -- overdetermined by the manipulation's definition (removing the "
        "only write path into the residue field removes residue variance by construction); "
        "a PASS confirms the protocol dimension landed, a FAIL would be informative; NOT "
        "attribution of the 936a-vs-571b gap"
    )
    OPEN = "OPEN CONTRAST -- fed vs fed, warm-up removed; outcome not derivable from the design"
    c3 = _flip_criterion("C3_residue_protocol_wiring_check", STARVED_ARM, "residue_protocol", WIRING)
    c4 = _flip_criterion("C4_warmup_flips_occupant", NO_WARMUP_ARM, "warmup_schedule", OPEN)
    c5 = _flip_criterion("C5_both_dimensions_wiring_check", BOTH_ARM, "residue_protocol+warmup_schedule", WIRING)
    c6 = {
        "name": "C6_final_commit_decided_at_primary_stage",
        "load_bearing": False,
        "role": (
            "RECORDED stage check (red-team pass 2, N1) -- not routing. Under "
            "use_modulatory_shortlist_then_modulate the primary score (the routed "
            "partition's channels) builds a shortlist and the modulatory accumulator "
            "picks within it; a PASS means the final commit was decided at the primary "
            "stage (shortlist inactive or singleton) on >= FINAL_COMMIT_PRIMARY_FLOOR of "
            "B1's fresh selections in EVERY seed, so the named occupant drives the commit; "
            "a FAIL means it drives ELIGIBILITY and the within-shortlist stage is unmeasured "
            "by this partition"
        ),
        "passed": bool(lb_rows and lb["final_commit_by_primary_frac_min"] >= FINAL_COMMIT_PRIMARY_FLOOR),
        "measured": float(lb["final_commit_by_primary_frac_min"]),
        "threshold": float(FINAL_COMMIT_PRIMARY_FLOOR),
        "detail": (
            f"B1 shortlist active frac {lb['modulatory_shortlist_active_frac_mean']:.3f}, mean size "
            f"{lb['modulatory_shortlist_size_mean']:.2f}, final commit at primary stage "
            f"{lb['final_commit_by_primary_frac_mean']:.3f} (min {lb['final_commit_by_primary_frac_min']:.3f})"
        ),
    }

    criteria = [
        {
            "name": "C1_monopoly_present_936_regime",
            "load_bearing": True,
            "role": "verdict (routed partition: within-tick cross-candidate variance shares)",
            "passed": c1_monopoly_present,
            "detail": (
                f"B1 seeds unanimous on xcand top channel = {lb_unanimous} "
                f"(per seed {lb['xcand_top_channel_per_seed']}); modal channel "
                f"{lb['xcand_top_channel_modal']} mean share {lb['xcand_modal_channel_share_mean']:.6g} "
                f"(min {lb['xcand_modal_channel_share_min']:.6g}) vs bar {F_MONOPOLY_THRESHOLD}"
            ),
        },
        {
            "name": "C2_monopoly_occupant_is_f",
            "load_bearing": False,
            "role": "refines C1",
            "passed": c2_occupant_is_f,
            "detail": f"modal xcand top channel in the 936 regime = {lb['xcand_top_channel_modal']}",
        },
        c3, c4, c5, c6,
    ]

    def _pair_green(other: str) -> bool:
        return LOAD_BEARING_ARM in green and other in green

    criteria_non_degenerate = {
        "C1_monopoly_present_936_regime": bool(
            LOAD_BEARING_ARM in green and all(r["cell_decidable"] for r in lb_rows)
        ),
        "C2_monopoly_occupant_is_f": bool(LOAD_BEARING_ARM in green and c1_monopoly_present),
        "C3_residue_protocol_wiring_check": bool(_pair_green(STARVED_ARM) and c3["n_matched_decidable_pairs"] > 0),
        "C4_warmup_flips_occupant": bool(_pair_green(NO_WARMUP_ARM) and c4["n_matched_decidable_pairs"] > 0),
        "C5_both_dimensions_wiring_check": bool(_pair_green(BOTH_ARM) and c5["n_matched_decidable_pairs"] > 0),
        "C6_final_commit_decided_at_primary_stage": bool(
            LOAD_BEARING_ARM in green and lb_rows and all(r["n_shortlist_reads"] > 0 for r in lb_rows)
        ),
    }

    # --- route ----------------------------------------------------------------
    if not lb_gate["gate_green"]:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif not lb_unanimous:
        label = "seeds_disagree_on_occupant_936_regime"
        outcome = "PASS"
    elif c1_monopoly_present and c2_occupant_is_f:
        label = "f_monopoly_present_936_regime"
        outcome = "PASS"
    elif c1_monopoly_present:
        label = "monopoly_present_non_f_occupant_936_regime"
        outcome = "PASS"
    else:
        label = "no_monopoly_936_regime"
        outcome = "PASS"

    note_map = {
        "f_monopoly_present_936_regime": (
            "An F-channel monopoly of within-tick CROSS-CANDIDATE variance IS present in "
            "the 936 regime with the clamp armed, so 936a's gate CLEARS for the 936 regime: "
            "a 936-family falsifier may be re-posed there, naming the F channel recorded "
            "here as the occupant and carrying a monopoly-presence precondition. READ WITH "
            "C6: the routed partition characterises the PRIMARY-score stage (shortlist "
            "membership); if C6 fails, the final commit was decided within the shortlist by "
            "the modulatory accumulator and the occupant named here drives eligibility, not "
            "the final choice."
        ),
        "monopoly_present_non_f_occupant_936_regime": (
            "A cross-candidate monopoly exists in the 936 regime (one channel >= 0.85 of "
            "mean within-tick cross-candidate variance) but F does not hold it. 936a's gate "
            "stays SHUT for a falsifier posed against F; the 936-family design must be "
            "re-posed against the named occupant, and MECH-439's F-specific premise does "
            "not obtain in this regime on the steering axis. READ WITH C6: the routed "
            "partition characterises the PRIMARY-score stage (shortlist membership); if C6 "
            "fails, the final commit was decided within the shortlist by the modulatory "
            "accumulator and the occupant named here drives eligibility, not the final choice."
        ),
        "no_monopoly_936_regime": (
            "No channel monopolises within-tick cross-candidate variance in the 936 regime "
            "with the clamp armed. MECH-439's quantitative premise does not obtain in this "
            "regime on the steering axis; 936a's gate stays SHUT and the falsifier family "
            "needs a different premise, not a relative bar."
        ),
        "seeds_disagree_on_occupant_936_regime": (
            "The B1 seeds do not agree on which channel tops the cross-candidate partition, "
            "so no single channel monopolises ACROSS seeds; C1 is false by construction and "
            "the per-seed occupants are recorded. 936a's gate stays SHUT; a re-pose must "
            "handle occupant seed-dependence explicitly."
        ),
        "substrate_not_ready_requeue": (
            "The load-bearing arm's readiness gate is red -- the routed partition is not "
            "measurable (sample floor, denominator floor, fewer than two live channels, or "
            "a config-landing identity) in at least one of its cells. Re-queue at an "
            "adequate n; this run routes to no verdict on the gate (contrast arms that are "
            "green are still recorded under per_arm_gate and summary.arms)."
        ),
    }

    exposure_note = (
        "Arms are matched on DECOMPOSITION SAMPLE COUNT (P1 runs to N_FRESH_SELECT_TARGET "
        "genuine selections per cell) but NOT on env exposure or SD-056 online E2 training: "
        "starved arms hold longer, so they accrue more P1 env ticks and more E2 updates. "
        "Per-arm means -- "
        + "; ".join(
            f"{a['id']}: env_ticks {arm_summaries[a['id']]['n_env_ticks_total_mean']:.0f}, "
            f"p1_ticks {arm_summaries[a['id']]['n_p1_ticks_mean']:.0f}, "
            f"e2_steps_p1 {arm_summaries[a['id']]['n_contrastive_steps_p1_mean']:.0f}, "
            f"e2_steps_total {arm_summaries[a['id']]['n_contrastive_steps_total_mean']:.0f}"
            for a in ARMS
        )
        + ". Occupant identity comparisons across arms are robust to this; quantitative "
        "share comparisons across fed/starved arms are confounded by it. C4 (B1 vs B3, the "
        "one open contrast) carries its OWN asymmetries (red-team pass 2, N4): B3 has no P0 "
        "and a different P1 tick count, and the two arms differ in commitment holding "
        "(n_latched_mean B1 "
        + f"{arm_summaries[LOAD_BEARING_ARM]['n_latched_mean']:.1f} vs B3 "
        + f"{arm_summaries[NO_WARMUP_ARM]['n_latched_mean']:.1f}), which switches the "
        "within-shortlist rule between committed argmin and uncommitted softmax "
        "(e3_selector.py:3766-3794); a C4 occupant flip is therefore attributable to the "
        "warm-up dimension only jointly with those recorded differences, not to it alone."
    )

    selection_stage_note = (
        "The routed partition is over the PRIMARY score's additive channels "
        "(f_weighted, harm_weighted, residue_weighted, ...). Under the 936 lineage's "
        "use_modulatory_shortlist_then_modulate=True those channels build the eligibility "
        "shortlist (raw_scores <= best + 0.25 * range) and the final commit is the argmin of "
        "the MODULATORY accumulator (lPFC bias, MECH-295 liking bridge, channel routing) "
        "within it, unless the shortlist is a singleton or inactive. modulatory_shortlist_"
        "active_frac / modulatory_shortlist_size_mean / final_commit_by_primary_frac are "
        "recorded per cell and C6 summarises B1: the occupant named by C1/C2 drives the "
        "final commit only when C6 passes; otherwise it drives eligibility. partition_"
        "coverage (sum of component cross-candidate variances / cross-candidate variance of "
        "the final score) is recorded per cell: < 1 = an omitted additive channel, > 1 = net "
        "cancellation among the declared ones (red-team pass 2, N1 and N2)."
    )

    outcome_note = (
        f"{label}: ROUTED (cross-candidate) 936-regime occupant {lb['xcand_top_channel_modal']} "
        f"(unanimous={lb_unanimous}, per seed {lb['xcand_top_channel_per_seed']}) at modal share "
        f"{lb['xcand_modal_channel_share_mean']:.6g}; xcand F share {lb['xcand_f_share_mean']:.6g}, "
        f"xcand residue share {lb['xcand_residue_share_mean']:.6g}; live channels "
        f"{lb['live_channels_union']} (min per cell {lb['n_live_channels_min']}), structurally zero "
        f"{lb['structurally_zero_components_union']}. RECORDED temporal (571b-comparable) occupant "
        f"{lb['temporal_top_channel_modal']} at {lb['temporal_committed_top_share_mean']:.6g} "
        f"(temporal F share {lb['temporal_committed_f_share_mean']:.6g}); temporal-vs-xcand "
        f"total-variation distance {lb['partition_tv_distance_mean']:.4f}, top channel differs in "
        f"{lb['partition_top_channel_differs_count']}/{len(lb_rows)} seeds; dispersion/range "
        f"{lb['dispersion_to_range_ratio_mean']:.3g} (571b band 0.44-3.18). Residue recruitment: "
        f"first-saturation episode per seed {lb['residue_rbf_first_saturation_episode_per_seed']}. "
        f"Selection stage (C6 {'PASS' if c6['passed'] else 'FAIL'}): modulatory shortlist active on "
        f"{lb['modulatory_shortlist_active_frac_mean']:.3f} of B1 fresh selections, mean size "
        f"{lb['modulatory_shortlist_size_mean']:.2f}, final commit decided at the primary (routed) stage on "
        f"{lb['final_commit_by_primary_frac_mean']:.3f} (min {lb['final_commit_by_primary_frac_min']:.3f}); "
        f"partition coverage {lb['partition_coverage_mean']:.3f}; max component |mean| "
        f"{lb['max_component_abs_mean_max']:.3g}. "
        f"Contrasts (routed occupant): starved+warmup "
        f"{arm_summaries[STARVED_ARM]['xcand_top_channel_modal']} "
        f"(xcand F {arm_summaries[STARVED_ARM]['xcand_f_share_mean']:.6g}) [C3 wiring check "
        f"{'PASS' if c3['passed'] else 'FAIL'}]; fed+no-warmup "
        f"{arm_summaries[NO_WARMUP_ARM]['xcand_top_channel_modal']} "
        f"(xcand F {arm_summaries[NO_WARMUP_ARM]['xcand_f_share_mean']:.6g}) [C4 OPEN contrast: "
        f"{c4['n_occupant_flips']}/{c4['n_matched_decidable_pairs']} flips]; starved+no-warmup "
        f"{arm_summaries[BOTH_ARM]['xcand_top_channel_modal']} "
        f"(xcand F {arm_summaries[BOTH_ARM]['xcand_f_share_mean']:.6g}) [C5 wiring check "
        f"{'PASS' if c5['passed'] else 'FAIL'}; also the 571b-protocol replication in the 936 env, "
        f"vs 571b A1 temporal harm_weighted 0.937-0.995]. No gap apportionment is claimed. "
        f"Clamp binding in B1: max||z_w||/ceiling {lb['rollout_max_over_ceiling_mean']:.4f}, "
        f"at-ceiling fraction {lb['rollout_at_ceiling_frac_mean']:.4f}, score range "
        f"{lb['score_range_mean']:.4g}. {note_map[label]}"
    )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "arm_results": rows,
        "per_arm_gate": gate["per_arm_gate"],
        "non_degenerate": bool(gate["non_degenerate"]),
        "degeneracy_reason": gate["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": gate["adjudication_preconditions"],
            "criteria_non_degenerate": criteria_non_degenerate,
            "combination_rule": (
                "C1 alone routes the verdict (load-bearing; the B1 arm's gate must be "
                "green or the run self-routes substrate_not_ready_requeue; the B1 seeds "
                "must be unanimous on the routed top channel or the run routes "
                "seeds_disagree_on_occupant_936_regime). C1 = mean over B1 seeds of the "
                "MODAL channel's cross-candidate share >= 0.85. C2 refines WHICH channel "
                "holds a monopoly C1 found. C3 and C5 are WIRING CHECKS (overdetermined by "
                "the residue-protocol manipulation); C4 is the one OPEN contrast; none of "
                "C3/C4/C5 override C1 and no gap apportionment is derived from them."
            ),
            "routed_statistic": (
                "xcand_share: per component, the mean over P1 genuine selections of the "
                "within-tick variance of that component across the candidate set, divided "
                "by the sum over components (bounded [0,1], sums to 1). The temporal "
                "partitions routed by 571/571b/936/936a are recorded as temporal_* and "
                "their disagreement with the routed one as partition_disagreement."
            ),
            "exposure_matching_note": exposure_note,
            "selection_stage_note": selection_stage_note,
            "liveness_note": (
                "A channel is LIVE when its mean cross-candidate variance exceeds "
                f"{MIN_LIVE_CHANNEL_VARIANCE:g} AND its routed share is >= {MIN_LIVE_CHANNEL_SHARE:g}; "
                "the absolute floor is ~3x above the float32 quantization floor at the "
                "magnitudes measured in the smoke (max_component_abs_mean recorded per cell, "
                "red-team pass 2, N3)."
            ),
            "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
        },
        "criteria": criteria,
        "summary": {
            "label": label,
            "arms": arm_summaries,
            "load_bearing_arm": LOAD_BEARING_ARM,
            "f_monopoly_threshold": F_MONOPOLY_THRESHOLD,
            "apportioning_scope_note": (
                "C3/C5 (fed vs starved) are wiring checks, not attribution; C4 (warm-up) is "
                "the only open contrast; the environment dimension is not varied inside this "
                "run. The run does NOT apportion the 936a-vs-571b gap."
            ),
            "reference_936a_off_temporal_committed_f_share_range": [1.83e-06, 1.24e-05],
            "reference_571b_a1_temporal_committed_top_share_range": [0.9368, 0.9953],
            "n_temporal_pool_shares_exceeding_unity": int(
                sum(1 for r in rows if r.get("temporal_pool_share_exceeds_unity"))
            ),
        },
        "diagnostics": {
            "share_method_note": (
                "The ROUTED statistic is the within-tick cross-candidate variance partition "
                "(xcand_share) on the corrected component set (f_weighted). 571b's committed "
                "TEMPORAL partition, 571's / 936a's covariance-retained temporal pool method "
                "and the legacy raw-f set are recorded per cell for comparability and never "
                "routed. The 0.85 bar is left where 936a set it."
            ),
            "dispersion_to_range_note": (
                "dispersion_to_range_ratio = sqrt(temporal_committed_total_variance) / "
                "score_range_mean -- the scale check failure_autopsy_V3-EXQ-571b section 2b "
                "used (0.44-3.18 there). A value far above that band marks a TEMPORAL share "
                "dominated by drift the selector cannot act on within a tick. Recorded, not gated."
            ),
            "per_arm_dispersion_to_range_ratio": {
                a["id"]: arm_summaries[a["id"]]["dispersion_to_range_ratio_mean"] for a in ARMS
            },
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-571c: clamped E3 cross-candidate variance-monopoly presence audit in the 936 regime"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    seeds_used = list(DRY_RUN_SEEDS if args.dry_run else SEEDS)
    p0_used = DRY_RUN_P0 if args.dry_run else P0_WARMUP_EPISODES
    p1_cap_used = DRY_RUN_P1_CAP if args.dry_run else P1_EPISODE_CAP
    steps_used = DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE
    fresh_used = DRY_RUN_FRESH_TARGET if args.dry_run else N_FRESH_SELECT_TARGET
    full_config = {
        "seeds": seeds_used,
        "env_kwargs": dict(ENV_KWARGS),
        "schedule": {
            "p0_warmup_episodes": p0_used,
            "p1_episode_cap": p1_cap_used,
            "steps_per_episode": steps_used,
            "fresh_select_target": fresh_used,
        },
        "pre_registered_thresholds": {
            "F_MONOPOLY_THRESHOLD": F_MONOPOLY_THRESHOLD,
            "MIN_FRESH_SELECTIONS": MIN_FRESH_SELECTIONS,
            "MIN_TOTAL_VARIANCE": MIN_TOTAL_VARIANCE,
            "MIN_XCAND_TOTAL_VARIANCE": MIN_XCAND_TOTAL_VARIANCE,
            "MIN_LIVE_CHANNEL_VARIANCE": MIN_LIVE_CHANNEL_VARIANCE,
            "MIN_LIVE_CHANNEL_SHARE": MIN_LIVE_CHANNEL_SHARE,
            "FINAL_COMMIT_PRIMARY_FLOOR": FINAL_COMMIT_PRIMARY_FLOOR,
            "MIN_LIVE_CHANNELS": MIN_LIVE_CHANNELS,
            "PIN_EPS": PIN_EPS,
            "F_COMPONENTS": list(F_COMPONENTS),
            "SCORE_COMPONENTS": list(SCORE_COMPONENTS),
            "LEGACY_571_COMPONENTS": list(LEGACY_571_COMPONENTS),
        },
        "structural_zero_reasons": dict(STRUCTURAL_ZERO_REASONS),
        "arms": [dict(a) for a in ARMS],
        "arm_config_slices": {
            a["id"]: config_slice_for(a, p0_used, p1_cap_used, steps_used, fresh_used) for a in ARMS
        },
        "dry_run": bool(args.dry_run),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "diagnostic",
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "timestamp_utc": timestamp,
        "non_degenerate": result["non_degenerate"],
        "per_arm_gate": result["per_arm_gate"],
        "arm_results": result["arm_results"],
        "per_seed_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "criteria": result["criteria"],
        "summary": result["summary"],
        "diagnostics": result["diagnostics"],
        "custom_information": {
            "audits": "MECH-439 monopoly premise, in the 936-family regime (clamp armed), on the cross-candidate steering axis",
            "routed_by": "failure_autopsy_V3-EXQ-571b_2026-09-01 sections 3 and 7 (user-confirmed)",
            "gates": "failure_autopsy_V3-EXQ-936a_2026-08-30 -- 936-family falsifiers held until a monopoly-present regime is established",
            "reference_runs": {
                "v3_exq_936a": "v3_exq_936a_mech439_f_variance_share_rollout_clamp_fix_20260829T071510Z_v3",
                "v3_exq_571b": "v3_exq_571b_e3_variance_monopoly_presence_clamped_20260901T061141Z_v3",
                "temporal_vs_cross_candidate_precedent": "failure_autopsy_V3-EXQ-925_2026-08-12 learning point 1 (retained by 925a)",
            },
            "closure_node": "behavioral_diversity_isolation:GAP-I",
            "load_bearing_cell_identity": (
                "B1 is 936a's ARM_OFF (canonical baseline module; per-step update_residue; "
                "P0 60x200; SD-056 online training; clamp ratio 2.0) measured by 571b's instrument "
                "plus the within-tick cross-candidate partition"
            ),
            "red_team_redesign": (
                "Step 4.5 pass 1 (opus) BLOCKING: the temporal committed partition could not "
                "fail C1 (3 of 6 components structurally zero, remaining two 700-84000x apart) and "
                "is insensitive to cross-candidate steering. Routed DV moved to the within-tick "
                "cross-candidate partition; n_live_channels readiness added; C1 routes on the "
                "unanimous modal channel's share; C3/C5 demoted to wiring checks; gap_carrier "
                "removed; exposure mismatch stated; residue count demoted from the headline."
            ),
            "gov_reuse_1_check": (
                "Decisive readouts: xcand_top_share / xcand_top_channel (within-tick cross-"
                "candidate partition) in the 936 regime. No MECH-439 manifest carries a cross-"
                "candidate component partition (571/571b/936/936a all record temporal shares "
                "only; the per-candidate spread was computed and discarded at read time). "
                "residue_rbf_active_centers carried by no MECH-439 manifest; the temporal "
                "committed partition for the full 936 regime IS in 936a (residue ~1.0) and is "
                "carried here as the reference point. Not recoverable -> run."
            ),
            "step_2_5c_open_corrupting_overlaps_recorded_not_blocking": [
                "SD-082 (ree_core/pfc/lateral_pfc_analog.py::compute_bias; 936 CONFIG_FLAGS use_lateral_pfc_analog=True)",
                "contextmemory-write-path-addressing-degeneracy (e1_deep.py::ContextMemory.write)",
                "SD-e1-rollout-consistency-training (e1_deep.py::forward / predict_long_horizon)",
            ],
            "brake_count_note": (
                "MECH-439 counts 14 autopsies under the reconciled predicate (571b's own is "
                "counted by direction/owed-amend although its artifact adds no ceiling hit). "
                "Not braked: diagnostic on the measurement axis, different question. No "
                "ceiling hit added by this run."
            ),
            "dv_symmetry_statement": "see module docstring, DV-SYMMETRY DECLARATION (one line per arm)",
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
    if result["degeneracy_reason"]:
        manifest["degeneracy_reason"] = result["degeneracy_reason"]

    stamp_recording_core(
        manifest, config=full_config, seeds=seeds_used,
        script_path=Path(__file__), started_at=t0,
        agent=_LAST_AGENT["agent"], z_goal_stream_stats=_ZG.stats(),
    )

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(args.dry_run),
        config=full_config, seeds=seeds_used,
        script_path=Path(__file__), started_at=t0,
        agent=_LAST_AGENT["agent"], z_goal_stream_stats=_ZG.stats(),
        json_default=str,
    )
    print(f"Manifest written: {out_path}", flush=True)

    s = result["summary"]
    lbs = s["arms"][LOAD_BEARING_ARM]
    print(
        f"SUMMARY: 936-regime ROUTED xcand top={lbs['xcand_top_channel_modal']}"
        f"(modal share {lbs['xcand_modal_channel_share_mean']:.6g}, unanimous={lbs['xcand_top_channel_unanimous']}) "
        f"xcand_F={lbs['xcand_f_share_mean']:.6g} xcand_residue={lbs['xcand_residue_share_mean']:.6g} "
        f"live_min={lbs['n_live_channels_min']} | temporal top={lbs['temporal_top_channel_modal']}"
        f"({lbs['temporal_committed_top_share_mean']:.6g}) tv_dist={lbs['partition_tv_distance_mean']:.4f} | "
        f"starved xcand top={s['arms'][STARVED_ARM]['xcand_top_channel_modal']} "
        f"F={s['arms'][STARVED_ARM]['xcand_f_share_mean']:.6g} | "
        f"no-warmup xcand top={s['arms'][NO_WARMUP_ARM]['xcand_top_channel_modal']} "
        f"F={s['arms'][NO_WARMUP_ARM]['xcand_f_share_mean']:.6g} | "
        f"both xcand top={s['arms'][BOTH_ARM]['xcand_top_channel_modal']} "
        f"F={s['arms'][BOTH_ARM]['xcand_f_share_mean']:.6g}",
        flush=True,
    )
    print(f"LABEL: {result['interpretation']['label']}", flush=True)
    for c in result["criteria"]:
        print(f"  {c['name']}: {c['passed']}", flush=True)
    print(
        f"  per_arm_gate: green={result['per_arm_gate']['green_arms']} "
        f"red={result['per_arm_gate']['red_arms']}",
        flush=True,
    )

    if args.dry_run:
        # Engagement assertions before the full 16-cell grid is committed to:
        # (1) the residue-protocol manipulation reaches the recorded TEMPORAL partition
        #     (a wiring check -- it must, by construction);
        # (2) REPORT (not assert) the ROUTED cross-candidate readouts per cell so a
        #     reader can see whether C1 can fail in this regime (red-team H).
        fed = [r for r in result["arm_results"] if r["arm"] == LOAD_BEARING_ARM]
        starved = [r for r in result["arm_results"] if r["arm"] == STARVED_ARM]
        if fed and starved:
            d_t = abs(
                fed[0]["temporal_committed_fractions"].get("residue_weighted", 0.0)
                - starved[0]["temporal_committed_fractions"].get("residue_weighted", 0.0)
            )
            d_x = abs(
                fed[0]["xcand_share"].get("residue_weighted", 0.0)
                - starved[0]["xcand_share"].get("residue_weighted", 0.0)
            )
            print(
                f"  [smoke] residue share delta fed-vs-starved: temporal={d_t:.6g} xcand={d_x:.6g}",
                flush=True,
            )
            assert d_t > 1e-6, "SMOKE FAIL: residue protocol did not reach the temporal residue share"
        for r in result["arm_results"]:
            print(
                f"  [smoke-routed] {r['arm']}/seed{r['seed']}: xcand_top={r['xcand_top_channel']} "
                f"xcand_top_share={r['xcand_top_share']:.6g} xcand_share="
                + "{"
                + ", ".join(f"{k}:{v:.3g}" for k, v in r["xcand_share"].items())
                + "}"
                + f" live={r['n_live_channels']} zero={sorted(r['structurally_zero_components'])}",
                flush=True,
            )
        print("DRY RUN complete.", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
