"""V3-EXQ-1095 -- MECH-439 operator-ON conversion falsifier at the eligibility stage.

Does committed-action-class entropy LIFT over the collapsed-proposer and matched-noise
controls once eligibility is made commensurate by the E3 channel-commensurability operator
(use_e3_channel_commensurability=True)?

AUTHORITY. governance 2026-09-24 (substrate_queue f_dominance_conversion_ceiling
`governance_2026_09_24`) ratified the GFLAG-0297 eligibility-stage target as rung 3's
acceptance condition, recorded rung 3 VALIDATED at the eligibility stage (V3-EXQ-1012c PASS
8/8), and RELEASED -- SCOPED -- the 936-family / 654h-class conversion-falsifier refusal:
"only for falsifiers run in this same regime with the operator ON and the 3-channel scope
stated". Its named, not-built follow-on is "a redesigned conversion falsifier with the
operator ON (new EXQ; restate MECH-439's confirming branch first)". Confirmed
failure_autopsy_V3-EXQ-1012c_2026-09-24 sec 7b names the same run. Chip:
chip-20260924-mech439-operator-on-falsifier. Claim-text restatement: GFLAG-0471 (open;
supersedes GFLAG-0469, same session).

RED-TEAM (Step 4.5): see the RED-TEAM line at the end of this docstring.

WHY THE CONFIRMING BRANCH HAD TO BE RESTATED FIRST (GFLAG-0471)
---------------------------------------------------------------
MECH-439's registered CONFIRMING branch is "committed-action-class entropy stays flat ...
WHILE F's cross-candidate variance share stays above 0.85". With the operator ON that
conjunct is UNREACHABLE BY CONSTRUCTION: the operator divides each channel's per-candidate
term by an EMA of that channel's own cross-candidate SD, so shares tend to 1/k (GFLAG-0234
arithmetic identity; 1012c autopsy sec 2, "unreachable by construction"). 571c additionally
found the fed-regime monopolist is residue_weighted, not F.

The direction map used here is NOT invented for this run. It is V3-EXQ-936a's already
registered `combination_rule` with the operator substituted for its C2 share-reduction:

  936a: "C2 true -> supports (conversion required rebalancing F's share, as MECH-439
         predicts) ... C1 false otherwise ... the falsifier was NOT evaluated and the run
         is scored non_contributory / non_degenerate."

  here: C1 true (entropy lifts above BOTH controls) with the eligibility gate MET
         -> SUPPORTS.  C1 false with the gate MET -> NON_CONTRIBUTORY, routing to the
         RESERVED final-commit-stage replay (1012c autopsy sec 7b), never a weakens.

SCOPE LIMIT THE READER MUST CARRY: AN OPERATOR-ON RUN CANNOT FALSIFY MECH-439. The
registered falsifier requires a lift WITHOUT a reduction in F's cross-candidate variance
share; the operator reduces that share by construction, so the falsifying antecedent has no
instance in this regime. This is a SUPPORT-OR-ROUTE instrument: it can deliver the positive
discrimination on a richer substrate that MECH-439's own ceiling_routing_note names as its
promotion re-gate, or it can make the reserved final-commit locus the measured next step.
It cannot weaken the claim. `ready` on f_dominance_conversion_ceiling STAYS false (SD-018
field-head validation and SD-e1 var-bar re-registration are untouched); this run promotes
nothing on its own.

THE ARM CONTRAST (matched seeds, residue-FED regime only, 4 arms x 4 seeds = 16 cells)
--------------------------------------------------------------------------------------
Arms are 689i's four-arm contrast re-seated in the 936/571c/1012a/1012c regime (the
`_lib/baselines/mech439_f_variance_share.py` canonical OFF path, margin shortlist mode),
with the treatment lever swapped from the MECH-448 demotion to the commensurability
operator, which is what the governance release scopes to.

  ARM_PROPOSER_CTRL   candidate_summary_source=proposer, operator OFF
                      -- collapsed-proposer control (the no-conversion-reaches floor).
  ARM_MATCHED_NOISE   candidate_summary_source=proposer, operator OFF, Factor B gap-scaled
                      stochastic commit (alpha 1.0) -- the noise-as-diversity negative
                      control, reachable on the COMMITTED path (689i defect-2 repair: a
                      proposer-temperature control is washed out by the argmin).
  ARM_OFF             e2wf source, operator OFF -- the paired reference cell.
  ARM_ON (PRIMARY)    e2wf source, operator ON -- the lever under test; the only arm that
                      carries the eligibility-stage knockout instrument.

Residue regime: FED only -- `agent.update_residue` is called every env step. The starved
half of MECH-439's registered fed-vs-starved contrast is OUT by user decision; see "THE
STARVED REGIME IS OUT" below for why it cannot carry this DV and what that costs.

ACCEPTANCE (claim_ids=[MECH-439]). Verdict chain, in order -- instrument and control
validity precede everything, so a broken instrument can never reach a claim verdict:
  C_INSTRUMENT_CLEAN     (hard)      -> instrument_defect
  C_CONTROL_DISTINCT     (hard)      -> control_arms_not_distinct_invalid
  C_NOISE_LIFTS          (gating)    -> matched_noise_control_unmeetable
  C_READINESS (per arm)  (gating)    -> substrate_not_ready_requeue
  C2_ELIGIBILITY_COMMENSURATE        -> eligibility_not_commensurate_requeue
  C1_CONVERSION          (verdict)   -> conversion_ceiling_persists_under_commensurate_eligibility
  C1b_OPERATOR_ATTRIBUTION (verdict) -> conversion_not_attributable_to_operator
  else (C1 AND C1b)                  -> commensurate_eligibility_converts (supports)

THE GFLAG-0072 CLASS-ENTROPY CEILING -- PREMISE RE-MEASURED, NOT ASSUMED
------------------------------------------------------------------------
GFLAG-0072 (resolved on registration) records that
`support_preserving_min_first_action_classes` defaults to 2, caps committed-class entropy
near ln(~3) ARM-INVARIANTLY, and that "ANY future ARC-065/MECH-439/440/441/ARC-108/ARC-110
null must set P1 and arm P2 at design time". P2 (modulatory authority) is already armed by
the lineage baseline. P1 is NOT raised here, and that is a measured decision, not an
oversight:

  * Raising it would break byte-identity with 571c/1012a/1012c, which is exactly the regime
    the governance release is SCOPED to ("in this same regime"). 1012c made the same call
    for the same reason.
  * The ceiling is EMPIRICALLY NOT BINDING in this regime. GFLAG-0072's evidence is
    V3-EXQ-708b's PRE-COMMIT statistics (precommit_n_distinct_classes_mean 2.959,
    mean_precommit_class_entropy 1.0609) in a different regime. V3-EXQ-936a's landed
    manifest, on the very baseline module and env this run uses, measures COMMITTED-class
    entropy 0.479 / 0.639 / 0.958 / 1.189 (ARM_OFF) and 0.693 / 0.966 / 1.210 / 1.420
    (ARM_DEMOTION) over 2-5 distinct committed classes, against ln(action_dim=5) = 1.6094 --
    it varies by ARM at every seed and exceeds ln(3) = 1.0986 in three cells, so it is
    neither arm-invariant nor pinned.
  * The concern is nonetheless converted from an assumption into a GATE rather than
    dismissed: `control_entropy_headroom` (below) is a readiness precondition on the SAME
    statistic C1 routes on, measured on the CONTROL arms C1 must exceed, and a cell whose
    controls sit within CONTROL_HEADROOM_FLOOR of ln(5) self-routes
    substrate_not_ready_requeue -- never a claim verdict. `per_arm_headroom` is emitted for
    every arm on PASS runs too.

DV-SYMMETRY INVARIANCE (one line per arm, per /queue-experiment Step 3)
-----------------------------------------------------------------------
DV: committed-action-class entropy (Miller-Madow corrected; see RED-TEAM FIX 2) -- a
set-aggregate over the sequence of first-action classes of the SELECTED candidate, one
sample per genuine fresh selection, banked to a FIXED n per cell.
Its symmetry group is (a) any transform of the score vector that preserves the argmin, (b)
permutation of the observation sequence, (c) relabelling of classes.
  * ARM_ON: the operator applies a PER-CHANNEL, non-uniform divisor (each channel's own
    cross-candidate SD EMA) to an additive score decomposition. That is not a common
    monotone transform and it demonstrably moves the argmin -- V3-EXQ-1012a measured
    primary_argmin_flip_rate 0.82 fed / 0.72 starved in this exact regime. NOT invariant.
  * ARM_PROPOSER_CTRL: replaces the candidate summary the modulatory channels route on
    (e2_world_forward -> proposer), changing which candidates are score-distinguishable at
    all rather than rescaling a fixed score vector. NOT invariant.
  * ARM_MATCHED_NOISE: Factor B routes the committed pick through a real torch.multinomial
    sample (_gap_scaled_commit_pick) instead of the deterministic argmin, so it changes the
    committed-class DISTRIBUTION, which IS the DV. NOT invariant. (CLAUDE.md's
    cross-machine torch.multinomial caveat applies to EXACT committed actions; this DV is a
    distributional aggregate over >= N_FRESH_SELECT_TARGET draws per cell and machine_class
    is recorded.)
  * ARM_OFF: reference cell, no manipulation -- invariance is not in question.

SUBSTRATE-PATH GATE (Step 2.5c). Open corrupting entries touching imported modules, with
each one's execution condition measured on this config at authoring time:
  * MECH-320 (policy/tonic_vigor.py) -- use_tonic_vigor=False. Absent.
  * sd_blocked_agency_mismatch_floor_calibration (affect/blocked_agency.py) --
    use_blocked_agency=False. Absent.
  * sd105_frozen_shared_entropy_floor_multiplier (regulators/selection_entropy_floor.py) --
    use_selection_entropy_floor=False. Absent.
  * sd_zself_training_path (latent/self_recurrence.py) -- latent_stack.self_recurrence is
    None on this config. Absent.
  * SD-PP-B5-z-world-per-step-displacement-range -- its ::function entry
    (e2_fast.py::compute_world_interventional_loss) is NOT called by this driver, which
    trains e2 through world_forward_contrastive_loss (SD-056). Its SUBSTANTIVE finding
    (compressed z_world per-step displacement bounding the world-forward head's dynamic
    range) does bear on the e2wf-sourced arms' candidate spread, so it is guarded rather
    than dismissed: `candidate_first_action_classes` and `control_entropy_headroom` are
    gating readiness preconditions, and a compressed pool self-routes
    substrate_not_ready_requeue.
  * contextmemory-write-path-addressing-degeneracy (e1_deep.py ContextMemory.write) -- LIVE
    here (e1.context_memory is instantiated). Carried as an INHERITED, REGIME-WIDE scope
    bound exactly as V3-EXQ-1012c carried it: it is upstream of the stage under test and
    fires identically in every arm of a paired-by-seed within-regime design, so it cannot
    manufacture or mask a between-arm difference; it bounds external validity only.

E3 LATCH / PSEUDO-REPLICATION. last_score_* latch across non-E3 ticks (cadence 10), so
every read is gated on the shared FreshSelectProbe and n_fresh_select / n_latched are
emitted per cell.

SEED 44 is deliberately absent (reef-config per-seed instability; baseline module docstring).

Z_GOAL: update_z_goal is called with benefit_exposure=0.0 exactly as in 1012c; z_goal is
deliberately inert in this lineage and reads 0 content ticks.

ASCII-only output (repo rule).

RED-TEAM (Step 4.5), Fable 5.1, cross-model: BLOCKING on the first pass -> user decision ->
re-scoped and re-reviewed. See "THE STARVED REGIME" and "RED-TEAM FIXES" below, and the
queue entry note for the final verdict.

THE STARVED REGIME IS OUT -- USER DECISION A, 2026-09-24 (rec-20260924-5b0f9daa)
--------------------------------------------------------------------------------
This driver originally ran {residue-fed, residue-starved} x 4 arms x 4 seeds = 32 cells.
The Step 4.5 red-team returned BLOCKING on the starved half, and the finding was confirmed
at source and then numerically before anything was acted on:

  * `update_running_variance` has exactly ONE caller in all of `ree_core`:
    `e3_selector.py:4686`, inside `post_action_update`.
  * `post_action_update` has exactly ONE caller: `agent.py:11070`, inside
    `REEAgent.update_residue` (def `agent.py:11041`; next method def `record_transition` at
    `agent.py:11240`).
  * The STARVED protocol IS "never call `update_residue`" -- that is the manipulation, and
    571c / 1012a / 1012c all inherit it.
  * So `_running_variance` stays pinned at `precision_init` 0.5 (`config.py:1137`) against
    `commitment_threshold` 0.40 (`config.py:1135`). MEASURED on a freshly constructed
    lineage agent: rv 0.5, threshold 0.4, `rv < threshold` -> False. No starved tick is
    ever COMMITTED.
  * `e3_selector.py:956` documents exactly this shape in the substrate's own words ("a
    driver that never calls post_action_update leaves rv pinned at precision_init forever
    (v3_exq_925a documents exactly this)") -- the fact was already recorded in ree-v3 and
    had simply never been joined up with MECH-439's control text.
  * Consequence: MECH-439's registered matched-noise control is the Factor B gap-scaled
    stochastic commit, which is reachable ONLY on the committed branch. In a starved arm it
    is inert, so the matched-noise and collapsed-proposer controls differ by RNG stream
    alone and `C_NOISE_LIFTS` there is a coin flip -- which, AND-ed across regimes, would
    have vacated all 32 cells including a clean fed result. This driver's own smoke had
    corroborated it independently: all four STARVED arms returned an identical committed-
    class entropy of 0.6931 over 2 classes.

SCOPE LIMIT THIS BUYS, STATED RATHER THAN HIDDEN: MECH-439's `what_would_answer` names "a
residue-fed vs residue-starved protocol contrast" as part of its CONTROL set, because 571c
showed WHICH channel monopolises depends on the residue-feeding protocol. This run drops
that contrast. So a result here is NOT shown to be protocol-independent, and the
residue-STARVED half of MECH-439's registered control set remains unanswered for any
committed-action DV. Re-open it only if the RESERVED final-commit-stage replay (confirmed
failure_autopsy_V3-EXQ-1012c_2026-09-24 sec 7b) is ever built -- that replay reads the
commit stage directly and would need its own answer to the same problem. The finding is
registered for governance as GFLAG-0480 (MECH-439's registered fed/starved control contrast
is DV-CONDITIONAL: sound for eligibility-stage and variance-share DVs, so V3-EXQ-1012c's
PASS is unaffected; not sound for the claim's own committed-action-class DV).

RED-TEAM FIXES APPLIED AS INSTRUMENT CHANGES (same decision; neither alters the hypothesis,
an arm's meaning, or which claim the evidence attaches to)
--------------------------------------------------------------------------------------
FIX 1 (F2) -- C1 alone could not attribute a lift to the OPERATOR. ARM_ON differs from both
controls in `candidate_summary_source` (e2wf vs proposer) AS WELL AS the operator, and the
e2wf source is ARC-065 GAP-A's own designed conversion mechanism, so "the summary source
converted" is a live alternative, not a hypothetical one. Added `C1b_operator_attribution`:
ARM_ON strictly above its own operator-OFF twin ARM_OFF, paired by seed, on >= 3 of 4
seeds. ARM_OFF is byte-identical to ARM_ON except for the operator, so this is the
single-variable contrast. PASS now requires C1 AND C1b; C1 true with C1b false self-routes
`conversion_not_attributable_to_operator` / non_contributory, which is a real outcome rather
than a silently weaker claim. The CONTROLS' summary source is deliberately NOT equalised --
being proposer-sourced is precisely what makes ARM_PROPOSER_CTRL the collapsed-channel
floor, so equalising it would destroy the control rather than fix the confound.

FIX 2 (F5) -- C1 was a zero-margin strict `>` on PLUG-IN entropies at unequal realised n.
Plug-in Shannon entropy is biased by about -(K-1)/(2n): at K=5 that is 0.033 nats at n=60
against 0.010 at n=200, an artefact larger than the zero margin. Both halves of 689i/699c's
split are adopted: (i) FIXED N BY CONSTRUCTION -- each cell records exactly
`N_FRESH_SELECT_TARGET` committed samples and stops, and the readiness gate refuses a cell
that could not reach it, so the differential bias between compared cells is ZERO rather
than corrected; (ii) Miller-Madow closes the residual K_obs term. C1/C1b route on
`committed_action_class_entropy_mm`; the plug-in value, the per-cell correction, and the
counterfactual verdict under the uncorrected estimator are all recorded
(`diagnostics.miller_madow_audit.estimator_changes_verdict`).

RED-TEAM FINDINGS RECORDED BUT NOT ACTED ON, and why: F3 (the matched-noise control is
uniform-over-E rather than gap-scaled on the proposer summary source) and F4 (a red CONTROL
arm lands on the "conversion_ceiling_persists" label rather than substrate_not_ready) were
raised against the two-regime design; F6 (`C_CONTROL_DISTINCT` is whole-run, so a starved
degeneracy silences a valid fed contrast) is moot now that the starved regime is out. F4's
shape survives in reduced form for a red CONTROL arm and is left standing deliberately: the
label is reported alongside `per_arm_gate`, which names every red arm, and `c1_passed`
already requires `controls_green`, so a reader cannot take the label without seeing the red
arm beside it.
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.entropy_headroom import per_arm_headroom  # noqa: E402
from experiments._lib.fresh_select import FreshSelectCounter, FreshSelectProbe  # noqa: E402
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
    OFF_ARM_FLAGS,
    P0_WARMUP_EPISODES,
    SEEDS,
    STEPS_PER_EPISODE,
    TRANSITION_BUFFER_MAX,
    make_agent_kwargs,
    make_env,
    off_path_config_slice,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_1095_mech439_operator_on_conversion_falsifier"
QUEUE_ID = "V3-EXQ-1095"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-439"]

SD056_ROLLOUT_CLAMP_EXEMPT = (
    "clamp set via CONFIG_FLAGS dict-splat in "
    "experiments/_lib/baselines/mech439_f_variance_share.py::make_agent_kwargs "
    "(936-regime parity with 571c/1012a/1012c); re-asserted per cell as clamp_config_landed"
)

DISJUNCTIVE_CRITERIA_LOAD_BEARING_EXEMPT = (
    "the combination_rule is a strict CONJUNCTION -- instrument_clean AND C_CONTROL_DISTINCT "
    "AND C_NOISE_LIFTS in BOTH regimes AND C2 in BOTH regimes AND C1 in BOTH regimes -- so "
    "every tagged member genuinely must hold for the PASS branch. The lint matches the single "
    "'>=1 arm gate green' conjunct (the precondition_gate aggregate, which is deliberately "
    "any-arm-green so one red arm never vacates another's finding, per V3-EXQ-785); that "
    "conjunct is not itself a criterion and no per-criterion family is disjunctive here."
)

CRITERIA_THRESHOLD_EXEMPT = (
    "every load-bearing criterion records a numeric measured + threshold pair "
    "(C1/C2 per regime: n seeds at bar vs SEEDS_REQUIRED, plus per_seed_measured vs "
    "per_seed_threshold; the two hard gates: failure count vs 0) -- built in a loop in "
    "run_experiment(), which the AST scan does not resolve"
)

# --- Pre-registered thresholds (constants, never derived from this run) ------
# FIXED-N: a scorable cell must reach the FULL measurement budget, not a lower floor, so
# every compared cell carries the same n (red-team fix 2, load-bearing half; 689i's
# MIN_FRESH_SELECT_PER_CELL = N_FRESH_SELECT_TARGET). The .5 makes the gate's `>` and the
# arm-gate override's `>=` agree on an integer count of exactly N_FRESH_SELECT_TARGET.
MIN_FRESH_SELECTIONS = 199.5       # i.e. "at least 200"; was a 60-sample floor pre-fix
N_FRESH_SELECT_TARGET = 200        # 936a's P1 measurement budget
P1_EPISODE_CAP = 40
R_BAR = 0.25                       # the ratified eligibility-stage bar (governance 2026-09-24)
SEEDS_REQUIRED = 3                 # of 4 seeds, per regime (C1 and C2 alike)
CONTENT_FRAC = 0.1                 # channel has content on a tick iff sd_c(t) >= 0.1 * s_hat_c
MIN_CONTENT_TICKS = 30             # a channel enters R only with >= this many content ticks
RESIDUAL_REL_TOL = 1e-5            # I1 / I1c relative tolerance
SELF_CHECK_TOLERANCE = 1e-6        # I1b relative tolerance (1012a)
I2_MAX_MISMATCH_RATE = 0.01        # I2: float32 eligible-set size mismatch rate

# action_dim is fixed by ENV_KWARGS; asserted live in _make_agent so a substrate change
# cannot silently move the ceiling this run's headroom gate is denominated on.
EXPECTED_ACTION_DIM = 5
MAX_COMMITTED_CLASS_ENTROPY = math.log(EXPECTED_ACTION_DIM)   # 1.6094 nats
# C1 needs the arms it must EXCEED to be off the ceiling. Worst control cell must sit at
# least this far below ln(action_dim). 0.10 nats ~ 6% of the range; below that a strict-above
# comparison is arithmetic noise rather than a measurement.
CONTROL_HEADROOM_FLOOR = 0.10
# The pool must be able to express more than a binary choice for the DV to move at all.
MIN_POOL_FIRST_ACTION_CLASSES = 3.0

NOISE_ARM_COMMIT_ENTROPY_ALPHA = 1.0   # Factor B, noise control only (substrate default)

# Channel -> sign in score_trajectory (score = f + harm + residue - benefit - goal).
CHANNEL_SIGN: Dict[str, float] = {
    "f_weighted": 1.0,
    "harm_weighted": 1.0,
    "residue_weighted": 1.0,
    "benefit_weighted": -1.0,
    "goal_weighted": -1.0,
}
CHANNELS: Tuple[str, ...] = tuple(CHANNEL_SIGN.keys())
SCORINGS: Tuple[str, ...] = ("ON", "OFF", "ORACLE")

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1_CAP = 2
DRY_RUN_STEPS = 40
DRY_RUN_FRESH_TARGET = 6
DRY_RUN_MIN_CONTENT_TICKS = 2   # smoke only: lets the smoke exercise R/J on few ticks

_ZG = ZGoalStreamAccumulator()
_LAST_AGENT: Dict[str, Any] = {"agent": None}
_FRESH_SELECT = FreshSelectProbe("exq1095")

REGIMES: Tuple[str, ...] = ("fed",)   # USER DECISION A 2026-09-24 -- see docstring
LEVERS: List[Dict[str, Any]] = [
    {"lever": "PROPOSER_CTRL", "label": "proposer_collapsed_channel_baseline_control",
     "summary_source": "proposer", "operator": False, "factor_b": False, "role": "control"},
    {"lever": "MATCHED_NOISE", "label": "proposer_gap_scaled_stochastic_commit_noise_control",
     "summary_source": "proposer", "operator": False, "factor_b": True, "role": "control"},
    {"lever": "OFF", "label": "e2wf_operator_off_reference",
     "summary_source": "e2_world_forward", "operator": False, "factor_b": False, "role": "reference"},
    {"lever": "ON", "label": "e2wf_operator_on_commensurate_eligibility",
     "summary_source": "e2_world_forward", "operator": True, "factor_b": False, "role": "treatment"},
]


def _arm_id(regime: str, lever: str) -> str:
    return "ARM_%s_%s" % (regime.upper(), lever)


ARMS: List[Dict[str, Any]] = [
    {
        "id": _arm_id(regime, lv["lever"]),
        "regime": regime,
        "lever": lv["lever"],
        "label": lv["label"],
        "summary_source": lv["summary_source"],
        "operator": lv["operator"],
        "factor_b": lv["factor_b"],
        "role": lv["role"],
        "feed_residue": regime == "fed",
        "warmup": True,
        "instrumented": lv["operator"],       # only ON arms carry the knockout replay
        "load_bearing": True,
    }
    for regime in REGIMES
    for lv in LEVERS
]
CONTROL_LEVERS: Tuple[str, ...] = ("PROPOSER_CTRL", "MATCHED_NOISE")
# The three non-treatment arms, for the pairwise C_CONTROL_DISTINCT assertion (689i).
DISTINCTNESS_LEVERS: Tuple[str, ...] = ("PROPOSER_CTRL", "MATCHED_NOISE", "OFF")


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def _is_instrumented(ctx: Dict[str, Any]) -> bool:
    return bool(ctx.get("instrumented"))


def _is_control(ctx: Dict[str, Any]) -> bool:
    return str(ctx.get("lever")) in CONTROL_LEVERS


PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="fresh_selections_sufficient",
        description="genuine (fresh, latch-gated) P1 selections -- the committed-class-entropy denominator",
        control="worst seed of this arm",
        threshold=float(MIN_FRESH_SELECTIONS),
        direction="lower",
    ),
    PreconditionSpec(
        name="clamp_config_landed",
        description="the SD-056 E2 rollout output-norm clamp reached agent.e2.config in every cell",
        control="all cells of the arm",
        threshold=1.0,
        direction="lower",
    ),
    PreconditionSpec(
        name="residue_protocol_landed",
        description="the arm's residue-feeding protocol executed as declared (fed: every step; starved: never)",
        control="all cells of the arm",
        threshold=1.0,
        direction="lower",
    ),
    PreconditionSpec(
        name="candidate_first_action_classes",
        description=(
            "mean distinct first-action classes in the candidate pool per fresh selection -- "
            "the DV cannot move beyond a binary choice below this. Guards SD-PP-B5's "
            "compressed-candidate-summary bound on the e2wf-sourced arms."
        ),
        control="worst seed of this arm",
        threshold=MIN_POOL_FIRST_ACTION_CLASSES,
        direction="lower",
        structural_max=lambda ctx: float(EXPECTED_ACTION_DIM),
    ),
    PreconditionSpec(
        name="control_entropy_headroom",
        description=(
            "ln(action_dim) minus this control arm's committed-class entropy, worst cell. C1 is "
            "a strict-above comparison against the controls, so a control sitting on the ceiling "
            "starves it. Same statistic C1 routes on (GFLAG-0072 ceiling, measured not assumed)."
        ),
        control="worst seed of this arm; controls only",
        threshold=CONTROL_HEADROOM_FLOOR,
        direction="lower",
        structural_max=lambda ctx: float(MAX_COMMITTED_CLASS_ENTROPY),
        applies_to=_is_control,
        applies_note="only the arms C1 must exceed can starve C1's strict-above comparison",
    ),
    PreconditionSpec(
        name="operator_engaged",
        description=(
            "channel_scale_estimates.engaged at cell end -- an ON arm that never left the "
            "operator's warmup carries no manipulation"
        ),
        control="all cells of the arm",
        threshold=1.0,
        direction="lower",
        applies_to=_is_instrumented,
        applies_note="operator-OFF arms have no operator to engage",
    ),
    PreconditionSpec(
        name="n_content_channels",
        description=(
            "channels live by EMA with >= MIN_CONTENT_TICKS content ticks -- R needs >= 2 "
            "(571c's n_live_channels gate, eligibility-stage operationalisation)"
        ),
        control="worst seed of this arm",
        threshold=2.0,
        direction="lower",
        structural_max=lambda ctx: float(len(CHANNELS)),
        applies_to=_is_instrumented,
        applies_note="R is only computed on the instrumented ON arms",
    ),
]
GEQ_PRECONDITIONS = {s.name for s in PRECONDITION_SPECS}


def config_slice_for(arm: Dict[str, Any], p0_episodes: int, p1_cap: int, steps: int,
                     fresh_target: int) -> Dict[str, Any]:
    base = off_path_config_slice()
    return {
        "env_kwargs": base["env_kwargs"],
        "sd056_training": base["sd056_training"],
        "config_flags": base["config_flags"],
        "off_arm_flags": base["off_arm_flags"],
        "arm_flags": {
            "candidate_summary_source": str(arm["summary_source"]),
            "use_e3_channel_commensurability": bool(arm["operator"]),
            "use_gap_scaled_commit_temperature": bool(arm["factor_b"]),
            "gap_scaled_commit_entropy_alpha": NOISE_ARM_COMMIT_ENTROPY_ALPHA,
        },
        "schedule": {
            "p0_warmup_episodes": int(p0_episodes if arm["warmup"] else 0),
            "p1_episode_cap": int(p1_cap),
            "steps_per_episode": int(steps),
            "fresh_select_target": int(fresh_target),
        },
        "protocol": {
            "feed_residue_per_step": bool(arm["feed_residue"]),
            "p0_warmup": bool(arm["warmup"]),
        },
        "cell_readout_constants": {
            "r_bar": R_BAR,
            "content_frac": CONTENT_FRAC,
            "min_content_ticks": MIN_CONTENT_TICKS,
            "residual_rel_tol": RESIDUAL_REL_TOL,
            "self_check_tolerance": SELF_CHECK_TOLERANCE,
            "control_headroom_floor": CONTROL_HEADROOM_FLOOR,
            "min_pool_first_action_classes": MIN_POOL_FIRST_ACTION_CLASSES,
        },
    }


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Lineage baseline agent with this arm's flags applied.

    candidate_summary_source and the Factor B pair go through from_dims (they are real
    from_dims kwargs, verified at authoring time). use_e3_channel_commensurability is set on
    cfg.e3 AFTER from_dims and verified live, never through from_dims, which swallows unknown
    kwargs (1012c's idiom; memory reference-reeconfig-from-dims-silent-kwargs).
    """
    kwargs = dict(make_agent_kwargs(env, OFF_ARM_FLAGS))
    kwargs.update(
        candidate_summary_source=str(arm["summary_source"]),
        use_gap_scaled_commit_temperature=bool(arm["factor_b"]),
        gap_scaled_commit_entropy_alpha=NOISE_ARM_COMMIT_ENTROPY_ALPHA,
    )
    cfg = REEConfig.from_dims(**kwargs)
    cfg.e3.use_e3_channel_commensurability = bool(arm["operator"])
    agent = REEAgent(cfg)

    if int(env.action_dim) != EXPECTED_ACTION_DIM:
        raise RuntimeError(
            "action_dim is %d, not the pre-registered %d: the committed-class-entropy "
            "headroom gate is denominated on ln(%d)"
            % (int(env.action_dim), EXPECTED_ACTION_DIM, EXPECTED_ACTION_DIM)
        )
    if bool(getattr(agent.e3.config, "use_e3_channel_commensurability", False)) != bool(arm["operator"]):
        raise RuntimeError("e3 channel-commensurability knob did not reach the agent")
    if bool(getattr(agent.e3.config, "use_gap_scaled_commit_temperature", False)) != bool(arm["factor_b"]):
        raise RuntimeError("Factor B gap-scaled commit knob did not reach the agent")
    if str(getattr(cfg, "candidate_summary_source", "")) != str(arm["summary_source"]):
        raise RuntimeError("candidate_summary_source did not reach the config")
    for flag in ("use_dualsystem_arbitration", "use_pe_confidence_weighting"):
        if bool(getattr(agent.e3.config, flag, False)):
            raise RuntimeError(
                "%s is ON: the channel-sum reconstruction assumes it is off "
                "(it adds a non-channel term between the channel sum and raw_scores)" % flag
            )
    if str(getattr(agent.e3.config, "modulatory_shortlist_mode", "margin")) != "margin":
        raise RuntimeError("eligibility reconstruction assumes the margin shortlist mode")
    for flag in ("use_f_eligibility_demotion", "use_go_nogo_constitution"):
        if bool(getattr(agent.e3.config, flag, False)):
            raise RuntimeError("%s is ON: eligibility would not be the margin rule" % flag)
    return agent


# ---------------------------------------------------------------------------
# Measurement helpers (1012c's instrument, unchanged where reused)
# ---------------------------------------------------------------------------

def _scale_estimates(agent: Any) -> Dict[str, Any]:
    raw = getattr(getattr(agent, "e3", None), "last_channel_scale_estimates", None)
    if not isinstance(raw, dict):
        return {"present": False, "engaged": False, "n_updates": None, "scales": {}}
    out: Dict[str, Any] = {"present": True}
    for k in ("engaged", "n_updates", "warmup_ticks", "floor", "ema_alpha"):
        if k in raw:
            out[k] = raw[k]
    sc = raw.get("scales")
    out["scales"] = ({str(k): float(v) for k, v in sc.items()} if isinstance(sc, dict) else {})
    out["engaged"] = bool(raw.get("engaged", False))
    return out


def _mean(xs: Sequence[float], default: float = 0.0) -> float:
    return float(statistics.fmean(xs)) if xs else default


def _quantiles(xs: Sequence[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0, "p10": None, "p50": None, "p90": None, "max": None}
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        return float(s[min(n - 1, max(0, int(round(p * (n - 1)))))])

    return {"n": n, "p10": q(0.10), "p50": q(0.50), "p90": q(0.90), "max": float(s[-1])}


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None or not torch.is_tensor(v):
        return None
    return v.float().unsqueeze(0) if v.dim() == 1 else v.float()


def _rel_close(a: float, b: float, scale: float, tol: float) -> bool:
    return bool(abs(a - b) <= tol * max(1.0, abs(scale)))


def _entropy_from_counts(counts: Counter) -> float:
    """PLUG-IN Shannon entropy (nats). Byte-identical to 936a's helper -- kept so the
    recorded value stays comparable with the 936/936a lineage. NOT what C1/C1b route on."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        h -= p * math.log(p)
    return float(h)


def _entropy_miller_madow(counts: Counter) -> float:
    """Miller-Madow bias-corrected Shannon entropy: H_MM = H_plugin + (K_obs - 1) / (2N).

    RED-TEAM FIX 2 (F5, applied under the user's 2026-09-24 decision). The plug-in estimator
    is biased by about -(K-1)/(2n), so at K=5 it costs 0.033 nats at n=60 against 0.010 at
    n=200 -- an artefact larger than the zero margin C1's strict `>` allows, whenever two
    compared cells realise different n. Two fixes, and the FIRST is the load-bearing one
    (689i / 699c's split, adopted verbatim):
      (i) FIXED N BY CONSTRUCTION -- every cell banks exactly N_FRESH_SELECT_TARGET committed
          samples and stops recording (see run_cell), and the readiness gate refuses a cell
          that could not reach it, so the differential bias between compared cells is ZERO
          rather than corrected;
      (ii) Miller-Madow closes the residual K_obs-driven term. At N=200 it is (K_obs-1)/400,
           i.e. 0.0025 nats at K_obs=2 up to 0.0100 at K_obs=5.
    The plug-in value is emitted alongside as `committed_action_class_entropy`, with
    `miller_madow_correction_nats` beside it and a run-level `miller_madow_audit` recording
    the counterfactual verdict under the uncorrected estimator.
    """
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    k_obs = sum(1 for v in counts.values() if v > 0)
    return float(_entropy_from_counts(counts) + (k_obs - 1) / (2.0 * n))


def _first_action_class(action: torch.Tensor) -> Optional[int]:
    try:
        return int(action.argmax(dim=-1).reshape(-1)[0].item())
    except Exception:
        return None


def _pool_first_action_classes(candidates: Any) -> Optional[int]:
    """Distinct FIRST-ACTION classes across the candidate pool, this tick.

    `traj.actions[:, 0, :]` is 689i's `_first_actions_K` extraction -- the FIRST action of
    each candidate trajectory -- and its argmax is the same class the committed DV counts
    (`action.argmax(-1)[0]`). Getting this wrong is not cosmetic: a first draft flattened the
    whole trajectory tensor and reported ~28 "classes" out of 32 candidates against an
    action_dim of 5, i.e. a readiness gate that could never fail. Bounded by action_dim, so
    the structural_max on this precondition is a real bound.
    """
    try:
        classes = set()
        for traj in candidates:
            acts = getattr(traj, "actions", None)
            if acts is None or not torch.is_tensor(acts):
                return None
            first = acts[:, 0, :].detach().reshape(-1)
            classes.add(int(first.argmax().item()))
        return len(classes) if classes else None
    except Exception:
        return None


class _ScoreCallCapture:
    """1012c's capture harness, reused verbatim (see that driver for the full rationale).

    Wraps agent.e3.score_trajectory for ONE select_action() call; live behaviour and return
    value unchanged. __enter__ snapshots _chan_scale_ema AND _chan_scale_n BEFORE the live
    call, because select() folds this tick's own spread into the EMA after scoring.
    """

    def __init__(self, agent: REEAgent):
        self._agent = agent
        self._orig = None
        self._ema_snapshot: Dict[str, float] = {}
        self._n_snapshot: int = 0
        self.calls: List[Tuple[tuple, dict, float]] = []
        self.raw_terms: List[Dict[str, float]] = []

    def __enter__(self) -> "_ScoreCallCapture":
        self.calls = []
        self.raw_terms = []
        sel = self._agent.e3
        self._ema_snapshot = dict(getattr(sel, "_chan_scale_ema", {}) or {})
        self._n_snapshot = int(getattr(sel, "_chan_scale_n", 0) or 0)
        self._orig = sel.score_trajectory

        def _capturing(*args: Any, **kwargs: Any) -> torch.Tensor:
            s = self._orig(*args, **kwargs)
            self.calls.append((args, kwargs, float(s.detach().reshape(-1).mean().item())))
            raw = getattr(sel, "_last_commensurability_raw", None) or {}
            self.raw_terms.append({c: float(raw.get(c, 0.0)) for c in CHANNELS})
            return s

        sel.score_trajectory = _capturing
        return self

    def __exit__(self, *exc: Any) -> bool:
        sel = self._agent.e3
        try:
            del sel.__dict__["score_trajectory"]
        except KeyError:
            sel.score_trajectory = self._orig
        return False

    def divisors(self) -> Dict[str, float]:
        """_commensurability_scale's exact semantics, on the PRE-tick snapshot."""
        cfg = self._agent.e3.config
        warm = int(cfg.e3_commensurability_warmup_ticks)
        floor = float(cfg.e3_commensurability_floor)
        out: Dict[str, float] = {}
        for c in CHANNELS:
            if self._n_snapshot < warm:
                out[c] = 1.0
                continue
            s = float(self._ema_snapshot.get(c, 0.0))
            out[c] = s if s > floor else 1.0
        return out

    def replay(self, commensurability: bool) -> List[float]:
        sel = self._agent.e3
        prev_comm = bool(getattr(sel.config, "use_e3_channel_commensurability", False))
        prev_ema = dict(getattr(sel, "_chan_scale_ema", {}) or {})
        prev_n = int(getattr(sel, "_chan_scale_n", 0) or 0)
        prev_raw = dict(getattr(sel, "_last_commensurability_raw", {}) or {})
        sel._chan_scale_n = self._n_snapshot
        sel.config.use_e3_channel_commensurability = bool(commensurability)
        sel._chan_scale_ema = dict(self._ema_snapshot)
        try:
            out = []
            for args, kwargs, _live in self.calls:
                s = self._orig(*args, **kwargs)
                out.append(float(s.detach().reshape(-1).mean().item()))
            return out
        finally:
            sel.config.use_e3_channel_commensurability = prev_comm
            sel._chan_scale_ema = prev_ema
            sel._chan_scale_n = prev_n
            sel._last_commensurability_raw = prev_raw


def _eligible(scores: Sequence[float], margin: float) -> frozenset:
    lo = min(scores)
    rg = max(scores) - lo
    cut = lo + margin * rg
    return frozenset(i for i, s in enumerate(scores) if s <= cut)


def _eligible_live_float32(raw: torch.Tensor, margin: float) -> Tuple[int, frozenset]:
    raw = raw.detach().reshape(-1)
    raw_score_range = float((raw.max() - raw.min()).item())
    best_raw = float(raw.min().item())
    cutoff = best_raw + margin * raw_score_range
    idx = torch.nonzero(raw <= cutoff, as_tuple=False).flatten()
    return int(idx.numel()), frozenset(int(i) for i in idx.tolist())


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return 0.0 if u == 0 else 1.0 - len(a & b) / float(u)


def _sd(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) * (x - mu) for x in xs) / n)


def _r_from_j(jbar: Dict[str, float]) -> Optional[float]:
    if len(jbar) < 2:
        return None
    mx = max(jbar.values())
    if mx <= 0.0:
        return 0.0
    return float(min(jbar.values()) / mx)


def _e2_contrastive_step(agent: REEAgent,
                         buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                         optimiser: torch.optim.Optimizer,
                         rng: random.Random) -> Optional[float]:
    """SD-056 online contrastive step -- fires on EVERY arm (baseline module contract).

    Byte-equivalent to 1012c's helper; the OFF arm's computation is part of the lineage's
    fingerprinted baseline, so it must not drift between drivers.
    """
    if len(buffer) < CONTRASTIVE_BATCH_K:
        return None
    batch = rng.sample(list(buffer), CONTRASTIVE_BATCH_K)
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
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), 1.0)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

def run_cell(arm: Dict[str, Any], seed: int, p0_episodes: int, p1_episode_cap: int,
             steps_per_episode: int, fresh_target: int, dry_run: bool = False) -> Dict[str, Any]:
    min_content = DRY_RUN_MIN_CONTENT_TICKS if dry_run else MIN_CONTENT_TICKS
    arm_id = str(arm["id"])
    feed = bool(arm["feed_residue"])
    instrumented = bool(arm["instrumented"])
    p0 = int(p0_episodes) if arm["warmup"] else 0
    print("Seed %d Condition %s" % (seed, arm_id), flush=True)

    slice_for_cell = config_slice_for(arm, p0_episodes, p1_episode_cap, steps_per_episode, fresh_target)

    with arm_cell(seed, config_slice=slice_for_cell, script_path=Path(__file__),
                  config_slice_declared=True) as cell:
        env = make_env(seed)
        agent = _make_agent(env, arm)
        agent.e3.e3_score_decomp_enabled = True
        e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
        margin = float(getattr(agent.e3.config, "modulatory_shortlist_margin", 0.25))
        floor = float(agent.e3.config.e3_commensurability_floor)

        clamp_live = bool(getattr(agent.e2.config, "e2_rollout_output_norm_clamp_enabled", False))
        ratio_live = float(getattr(agent.e2.config, "e2_rollout_output_norm_clamp_ratio", 2.0))

        transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=TRANSITION_BUFFER_MAX
        )
        sample_rng = random.Random(seed)
        total_train_eps = p0 + int(p1_episode_cap)

        fs = FreshSelectCounter()
        n_ticks_total = 0
        n_update_residue_calls = 0
        n_contrastive_steps_total = 0
        p1_episodes_run = 0
        target_met = False

        # --- the conversion DV ---
        selected_class_counts: Counter = Counter()
        pool_classes: List[float] = []
        harm_total = 0.0
        n_env_steps = 0

        # --- instrument counters (ON arms only) ---
        n_scored = 0
        n_residual_fail = 0
        max_residual_rel = 0.0
        n_selfcheck_fail = 0
        n_offcheck_fail = 0
        n_shortlist_active = 0
        n_i2_mismatch = 0
        n_selected_outside_e = 0
        j_ticks: Dict[str, Dict[str, List[float]]] = {x: {c: [] for c in CHANNELS} for x in SCORINGS}
        e_sizes: Dict[str, List[int]] = {x: [] for x in SCORINGS}
        e_live_sizes: List[int] = []
        k_series: List[int] = []
        r_series: Dict[str, List[float]] = {c: [] for c in CHANNELS}
        live_by_ema_ticks: Dict[str, int] = {c: 0 for c in CHANNELS}

        for ep in range(total_train_eps):
            is_p1 = ep >= p0
            phase_label = "P1" if is_p1 else "P0"
            if is_p1:
                p1_episodes_run += 1

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
                    if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                            and torch.isfinite(z1_obs).all()):
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

                if instrumented:
                    with _ScoreCallCapture(agent) as _cap, _FRESH_SELECT.watch(agent) as _sel:
                        action = agent.select_action(candidates, ticks)
                    cap: Optional[_ScoreCallCapture] = _cap
                else:
                    with _FRESH_SELECT.watch(agent) as _sel:
                        action = agent.select_action(candidates, ticks)
                    cap = None
                fresh_select = _sel.fresh
                n_ticks_total += 1
                if is_p1:
                    fs.record(fresh_select)

                # --- THE CONVERSION DV: one sample per GENUINE fresh selection ---
                # FIXED-N BY CONSTRUCTION (red-team fix 2, load-bearing half): stop
                # recording at exactly `fresh_target` committed samples, so every scorable
                # cell carries the SAME n and the plug-in estimator's differential bias
                # between compared cells is zero rather than corrected. Stepping continues;
                # only the DV accumulator is capped.
                if is_p1 and fresh_select and sum(selected_class_counts.values()) < fresh_target:
                    cls = _first_action_class(action)
                    if cls is not None:
                        selected_class_counts[cls] += 1
                    npc = _pool_first_action_classes(candidates)
                    if npc is not None:
                        pool_classes.append(float(npc))

                # --- the eligibility-stage instrument (ON arms only) ---
                if instrumented and is_p1 and fresh_select and cap is not None and len(cap.calls) >= 2:
                    n_scored += 1
                    live = [c[2] for c in cap.calls]
                    terms = cap.raw_terms
                    k = len(live)
                    k_series.append(k)
                    m = cap.divisors()

                    s_on: List[float] = []
                    s_off: List[float] = []
                    tick_resid_fail = False
                    for i in range(k):
                        parts = [CHANNEL_SIGN[c] * terms[i][c] / m[c] for c in CHANNELS]
                        s = sum(parts)
                        s_on.append(s)
                        s_off.append(sum(CHANNEL_SIGN[c] * terms[i][c] for c in CHANNELS))
                        scale = max([abs(p) for p in parts] + [abs(live[i])])
                        rel = abs(live[i] - s) / max(1.0, scale)
                        max_residual_rel = max(max_residual_rel, rel)
                        if rel > RESIDUAL_REL_TOL:
                            tick_resid_fail = True
                    if tick_resid_fail:
                        n_residual_fail += 1

                    sc_scores = cap.replay(True)
                    if any(not _rel_close(a, b, max(abs(a), abs(b)), SELF_CHECK_TOLERANCE)
                           for a, b in zip(sc_scores, live)):
                        n_selfcheck_fail += 1

                    off_replay = cap.replay(False)
                    if any(not _rel_close(a, b, max([abs(a), abs(b)] + [abs(terms[i][c]) for c in CHANNELS]),
                                          RESIDUAL_REL_TOL)
                           for i, (a, b) in enumerate(zip(off_replay, s_off))):
                        n_offcheck_fail += 1

                    diag = getattr(agent.e3, "last_score_diagnostics", None) or {}
                    sl_active = bool(diag.get("modulatory_shortlist_active", False))
                    sl_size = int(diag.get("modulatory_shortlist_size", 0) or 0)
                    raw_t = getattr(agent.e3, "last_raw_scores", None)
                    if sl_active and torch.is_tensor(raw_t) and int(raw_t.numel()) == k:
                        n_shortlist_active += 1
                        n_live_e, e_live = _eligible_live_float32(raw_t, margin)
                        e_live_sizes.append(n_live_e)
                        if n_live_e != sl_size:
                            n_i2_mismatch += 1
                        sel_idx = getattr(agent.e3, "last_selected_idx", None)
                        if sel_idx is not None and int(sel_idx) not in e_live:
                            n_selected_outside_e += 1

                    sd_t: Dict[str, float] = {}
                    for c in CHANNELS:
                        col = [terms[i][c] for i in range(k)]
                        sd_t[c] = _sd(col)
                        if m[c] != 1.0:
                            live_by_ema_ticks[c] += 1
                            r_series[c].append(sd_t[c] / m[c])
                    oracle_div = {c: (sd_t[c] if sd_t[c] > floor else 1.0) for c in CHANNELS}
                    scaled = {
                        "ON": {c: [CHANNEL_SIGN[c] * terms[i][c] / m[c] for i in range(k)] for c in CHANNELS},
                        "OFF": {c: [CHANNEL_SIGN[c] * terms[i][c] for i in range(k)] for c in CHANNELS},
                        "ORACLE": {c: [CHANNEL_SIGN[c] * terms[i][c] / oracle_div[c] for i in range(k)]
                                   for c in CHANNELS},
                    }
                    for x in SCORINGS:
                        s_x = [sum(scaled[x][c][i] for c in CHANNELS) for i in range(k)]
                        e_x = _eligible(s_x, margin)
                        e_sizes[x].append(len(e_x))
                        for c in CHANNELS:
                            if m[c] == 1.0 or sd_t[c] < CONTENT_FRAC * m[c]:
                                continue
                            s_ko = [s_x[i] - scaled[x][c][i] for i in range(k)]
                            j_ticks[x][c].append(_jaccard_distance(e_x, _eligible(s_ko, margin)))

                if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                    pending_capture = (
                        latent.z_world.detach().reshape(-1).clone(),
                        action.detach().reshape(-1).clone(),
                    )

                if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                    loss_val = _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)
                    if loss_val is not None and math.isfinite(loss_val):
                        n_contrastive_steps_total += 1

                _, harm_signal, done, info, next_obs_dict = env.step(action)
                hv = float(harm_signal)
                # C_SAFETY reads REALIZED per-env-step harm: the env step is the correct
                # sampling unit for a realized-harm rate, so this one readout is deliberately
                # NOT fresh-gated (689i's note).
                harm_total += hv
                n_env_steps += 1

                if feed:
                    with torch.no_grad():
                        agent.update_residue(harm_signal=hv, world_delta=None,
                                             hypothesis_tag=False, owned=True)
                    n_update_residue_calls += 1

                z_self_prev = latent.z_self.detach()
                action_prev = action
                obs_dict = next_obs_dict
                tick_in_ep += 1
                if done:
                    break

            fs.flush()

            if ep == 0 or is_p1 or (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
                print(
                    "  [train] arm=%s seed=%d phase=%s ep %d/%d fresh=%d/%d scored=%d"
                    % (arm_id, seed, phase_label, ep + 1, total_train_eps,
                       fs.n_fresh_select, fresh_target, n_scored),
                    flush=True,
                )

            if is_p1 and fs.n_fresh_select >= fresh_target:
                target_met = True
                print("  [p1-done] arm=%s seed=%d fresh=%d after %d P1 episode(s)"
                      % (arm_id, seed, fs.n_fresh_select, p1_episodes_run), flush=True)
                break

        _ZG.observe(agent)
        _LAST_AGENT["agent"] = agent

        scale_est = _scale_estimates(agent)
        content_channels = sorted(c for c in CHANNELS if len(j_ticks["ON"][c]) >= min_content)
        jbar = {x: {c: _mean(j_ticks[x][c]) for c in content_channels} for x in SCORINGS}
        r_vals = {x: _r_from_j(jbar[x]) for x in SCORINGS}

        committed_entropy_plugin = _entropy_from_counts(selected_class_counts)
        committed_entropy = _entropy_miller_madow(selected_class_counts)   # ROUTED
        expected_calls = n_ticks_total if feed else 0
        row: Dict[str, Any] = {
            "arm": arm_id,
            "arm_id": arm_id,
            "regime": str(arm["regime"]),
            "lever": str(arm["lever"]),
            "role": str(arm["role"]),
            "label": str(arm["label"]),
            "seed": int(seed),
            "load_bearing_arm": bool(arm["load_bearing"]),
            "instrumented": instrumented,
            "feed_residue_per_step": feed,
            "candidate_summary_source": str(arm["summary_source"]),
            "operator_on": bool(arm["operator"]),
            "factor_b_on": bool(arm["factor_b"]),
            "p0_episodes": int(p0),
            "p1_episodes_run": int(p1_episodes_run),
            "fresh_target_met": bool(target_met),
            "clamp_live_on_e2": clamp_live,
            "clamp_ratio_live": ratio_live,
            "clamp_config_landed": bool(clamp_live and abs(ratio_live - 2.0) < 1e-12),
            "n_env_ticks_total": int(n_ticks_total),
            "n_update_residue_calls": int(n_update_residue_calls),
            "residue_protocol_landed": bool(n_update_residue_calls == expected_calls),
            "n_contrastive_steps_total": int(n_contrastive_steps_total),
            "n_fresh_select": int(fs.n_fresh_select),
            "n_latched": int(fs.n_latched),

            # ===== THE ROUTED DV (Miller-Madow; plug-in emitted alongside) =====
            "committed_action_class_entropy_mm": committed_entropy,
            "committed_action_class_entropy": committed_entropy_plugin,
            "miller_madow_correction_nats": float(committed_entropy - committed_entropy_plugin),
            "dv_sample_cap": int(fresh_target),
            "dv_sample_cap_met": bool(sum(selected_class_counts.values()) >= fresh_target),
            "n_committed_classes": len(selected_class_counts),
            "selected_class_counts": {str(k_): int(v) for k_, v in selected_class_counts.items()},
            "n_committed_samples": int(sum(selected_class_counts.values())),
            "committed_entropy_headroom": float(MAX_COMMITTED_CLASS_ENTROPY - committed_entropy),
            "max_committed_class_entropy": float(MAX_COMMITTED_CLASS_ENTROPY),

            # ===== readiness readouts =====
            "pool_first_action_classes_mean": _mean(pool_classes),
            "pool_first_action_classes_quantiles": _quantiles(pool_classes),
            "realized_harm_per_step": float(harm_total / n_env_steps) if n_env_steps else 0.0,

            # ===== eligibility-stage instrument (ON arms only; None elsewhere) =====
            "n_ticks_scored": int(n_scored),
            "n_residual_failures": int(n_residual_fail),
            "max_residual_rel": float(max_residual_rel),
            "n_self_check_failures": int(n_selfcheck_fail),
            "n_off_crosscheck_failures": int(n_offcheck_fail),
            "n_shortlist_active_scored": int(n_shortlist_active),
            "n_i2_size_mismatch": int(n_i2_mismatch),
            "i2_mismatch_rate": float(n_i2_mismatch / n_shortlist_active) if n_shortlist_active else 0.0,
            "n_selected_outside_live_e": int(n_selected_outside_e),
            "content_channels": content_channels,
            "n_content_channels": int(len(content_channels)),
            "jbar": jbar,
            "n_content_ticks": {c: len(j_ticks["ON"][c]) for c in CHANNELS},
            "R_ON": r_vals["ON"],
            "R_OFF": r_vals["OFF"],
            "R_ORACLE": r_vals["ORACLE"],
            "r_realised_quantiles": {c: _quantiles(r_series[c]) for c in CHANNELS if r_series[c]},
            "eligible_size_mean": {x: _mean([float(v) for v in e_sizes[x]]) for x in SCORINGS},
            "eligible_size_live_float32_mean": _mean([float(v) for v in e_live_sizes]),
            "k_quantiles": _quantiles([float(v) for v in k_series]),
            "channel_scale_estimates": scale_est,
            "operator_engaged": bool(scale_est.get("engaged", False)),
        }

        cell.stamp(row)

    cell_ready = bool(
        row["n_fresh_select"] >= (MIN_FRESH_SELECTIONS if not dry_run else 1)
        and row["clamp_config_landed"] and row["residue_protocol_landed"]
    )
    # One `verdict:` line per seed x condition, for the runner's run counter.
    print("verdict: %s" % ("PASS" if cell_ready else "FAIL"), flush=True)
    if dry_run:
        print("  [smoke] arm=%s seed=%d H=%.4f classes=%d pool=%.2f fresh=%d scored=%d "
              "R_ON=%s R_OFF=%s engaged=%s resid_fail=%d selfcheck_fail=%d offcheck_fail=%d"
              % (arm_id, seed, row["committed_action_class_entropy_mm"], row["n_committed_classes"],
                 row["pool_first_action_classes_mean"], row["n_fresh_select"], n_scored,
                 ("%.4f" % r_vals["ON"]) if r_vals["ON"] is not None else "None",
                 ("%.4f" % r_vals["OFF"]) if r_vals["OFF"] is not None else "None",
                 row["operator_engaged"], n_residual_fail, n_selfcheck_fail, n_offcheck_fail),
              flush=True)
    return row


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _arm_gate(arm: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _worst(key: str) -> float:
        vals = [float(r[key]) for r in rows]
        return min(vals) if vals else 0.0

    measured: Dict[str, float] = {
        "fresh_selections_sufficient": _worst("n_fresh_select"),
        "clamp_config_landed": 1.0 if rows and all(bool(r["clamp_config_landed"]) for r in rows) else 0.0,
        "residue_protocol_landed": 1.0 if rows and all(bool(r["residue_protocol_landed"]) for r in rows) else 0.0,
        "candidate_first_action_classes": _worst("pool_first_action_classes_mean"),
    }
    ctx = {
        "id": arm["id"], "regime": arm["regime"], "lever": arm["lever"], "role": arm["role"],
        "feed_residue": arm["feed_residue"], "warmup": arm["warmup"],
        "instrumented": arm["instrumented"],
    }
    if _is_control(ctx):
        # WORST CELL, not the mean: `met` is an all-cells claim (skill's worst-cell rule).
        measured["control_entropy_headroom"] = _worst("committed_entropy_headroom")
    if _is_instrumented(ctx):
        measured["operator_engaged"] = (
            1.0 if rows and all(bool(r["operator_engaged"]) for r in rows) else 0.0
        )
        measured["n_content_channels"] = _worst("n_content_channels")

    overrides = {spec.name: bool(measured[spec.name] >= spec.threshold)
                 for spec in PRECONDITION_SPECS
                 if spec.name in GEQ_PRECONDITIONS and spec.applies(ctx)}
    gate = evaluate_arm_gate(arm["id"], ctx, PRECONDITION_SPECS, measured, met_overrides=overrides)
    for p in gate["preconditions"]:
        p["kind"] = "readiness"
        if p.get("name", "").endswith("::control_entropy_headroom"):
            worst = min(rows, key=lambda r: float(r["committed_entropy_headroom"])) if rows else None
            if worst is not None:
                p["offending_cell"] = "%s/seed%s" % (worst["arm"], worst["seed"])
        if p.get("name", "").endswith("::candidate_first_action_classes"):
            worst = min(rows, key=lambda r: float(r["pool_first_action_classes_mean"])) if rows else None
            if worst is not None:
                p["offending_cell"] = "%s/seed%s" % (worst["arm"], worst["seed"])
    return gate


def _by(rows: List[Dict[str, Any]], regime: str, lever: str) -> Dict[int, Dict[str, Any]]:
    return {int(r["seed"]): r for r in rows if r["regime"] == regime and r["lever"] == lever}


def _identical_control_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """689i's C_CONTROL_DISTINCT: any (regime, seed) pair of non-treatment arms whose
    committed-class COUNT VECTOR is identical. In 689d ARM_MATCHED_NOISE and
    ARM_PROPOSER_CTRL were bit-identical on every metric on every seed, which made the
    negative control unmeetable by construction."""
    out: List[Dict[str, Any]] = []
    for regime in REGIMES:
        tables = {lv: _by(rows, regime, lv) for lv in DISTINCTNESS_LEVERS}
        seeds = sorted(set().union(*[set(t.keys()) for t in tables.values()]) if tables else set())
        for seed in seeds:
            for i in range(len(DISTINCTNESS_LEVERS)):
                for j in range(i + 1, len(DISTINCTNESS_LEVERS)):
                    a_lv, b_lv = DISTINCTNESS_LEVERS[i], DISTINCTNESS_LEVERS[j]
                    a, b = tables[a_lv].get(seed), tables[b_lv].get(seed)
                    if a is None or b is None:
                        continue
                    if a["selected_class_counts"] == b["selected_class_counts"]:
                        out.append({"regime": regime, "seed": seed, "arm_a": a["arm"],
                                    "arm_b": b["arm"], "counts": a["selected_class_counts"]})
    return out


def _regime_reading(rows: List[Dict[str, Any]], regime: str, green: set) -> Dict[str, Any]:
    on = _by(rows, regime, "ON")
    ctrls = {lv: _by(rows, regime, lv) for lv in CONTROL_LEVERS}
    off = _by(rows, regime, "OFF")
    seeds = sorted(on.keys())

    # --- C2: the ratified eligibility-stage gate, on the ON arm -----------------
    r_on = [on[s]["R_ON"] for s in seeds]
    r_off = [on[s]["R_OFF"] for s in seeds]
    r_or = [on[s]["R_ORACLE"] for s in seeds]
    n_on_at_bar = sum(1 for v in r_on if v is not None and float(v) >= R_BAR)
    on_arm_green = _arm_id(regime, "ON") in green
    c2_passed = bool(n_on_at_bar >= SEEDS_REQUIRED and on_arm_green)

    # --- C1: strict-above BOTH controls at the SAME seed ------------------------
    per_seed: List[Dict[str, Any]] = []
    n_converting = 0
    for s in seeds:
        h_on = float(on[s]["committed_action_class_entropy_mm"])
        h_ctrl = {lv: (float(ctrls[lv][s]["committed_action_class_entropy_mm"])
                       if s in ctrls[lv] else None) for lv in CONTROL_LEVERS}
        h_off = float(off[s]["committed_action_class_entropy_mm"]) if s in off else None
        above = all(v is not None and h_on > v for v in h_ctrl.values())
        if above:
            n_converting += 1
        per_seed.append({
            "seed": s, "H_on": h_on, "H_controls": h_ctrl, "H_off": h_off,
            "strict_above_both_controls": bool(above),
            "margin_vs_worst_control": (h_on - max(v for v in h_ctrl.values())
                                        if all(v is not None for v in h_ctrl.values()) else None),
        })
    controls_green = all(_arm_id(regime, lv) in green for lv in CONTROL_LEVERS)
    c1_passed = bool(n_converting >= SEEDS_REQUIRED and on_arm_green and controls_green)

    # --- C1b: OPERATOR ATTRIBUTION -- ARM_ON vs ARM_OFF, paired by seed ---------
    # RED-TEAM FIX 1 (F2, applied under the user's 2026-09-24 decision). ARM_ON differs
    # from BOTH controls in `candidate_summary_source` (e2wf vs proposer) AS WELL AS the
    # operator, so C1 alone cannot separate "the operator converted" from "the e2wf summary
    # source converted" -- and the e2wf source is ARC-065 GAP-A's own designed conversion
    # mechanism, so that alternative is live, not hypothetical. ARM_OFF is byte-identical to
    # ARM_ON except for the operator, so this paired contrast is the single-variable test.
    # The controls' summary source is deliberately NOT changed: being proposer-sourced IS
    # what makes ARM_PROPOSER_CTRL the collapsed-channel floor, so equalising it would
    # destroy the control's meaning rather than fix the confound (689i's arm geometry).
    off_green = _arm_id(regime, "OFF") in green
    n_on_above_off = 0
    attribution_per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        h_on = float(on[s]["committed_action_class_entropy_mm"])
        h_off_s = float(off[s]["committed_action_class_entropy_mm"]) if s in off else None
        above_off = h_off_s is not None and h_on > h_off_s
        if above_off:
            n_on_above_off += 1
        attribution_per_seed.append({
            "seed": s, "H_on": h_on, "H_off": h_off_s,
            "strict_above_off": bool(above_off),
            "margin_vs_off": (h_on - h_off_s) if h_off_s is not None else None,
        })
    c1b_passed = bool(n_on_above_off >= SEEDS_REQUIRED and on_arm_green and off_green)

    # --- C_NOISE_LIFTS: the matched-noise control must verifiably lift -----------
    n_noise_lifts = 0
    noise_per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        a = ctrls["MATCHED_NOISE"].get(s)
        b = ctrls["PROPOSER_CTRL"].get(s)
        if a is None or b is None:
            noise_per_seed.append({"seed": s, "H_noise": None, "H_proposer": None, "lifts": False})
            continue
        ha = float(a["committed_action_class_entropy_mm"])
        hb = float(b["committed_action_class_entropy_mm"])
        lifts = ha > hb
        if lifts:
            n_noise_lifts += 1
        noise_per_seed.append({"seed": s, "H_noise": ha, "H_proposer": hb, "lifts": bool(lifts)})
    noise_lifts = bool(n_noise_lifts >= SEEDS_REQUIRED)

    # --- C_SAFETY (recorded; routes to a SUBSTRATE self-route, never a claim direction) ---
    harm_on = _mean([float(on[s]["realized_harm_per_step"]) for s in seeds])
    harm_off = _mean([float(off[s]["realized_harm_per_step"]) for s in seeds if s in off])

    return {
        "regime": regime,
        "seeds": seeds,
        "on_arm_green": on_arm_green,
        "controls_green": controls_green,
        "c2_passed": c2_passed,
        "n_seeds_R_ON_at_bar": int(n_on_at_bar),
        "R_ON_per_seed": r_on,
        "R_OFF_per_seed": r_off,
        "R_ORACLE_per_seed": r_or,
        "R_ON_median": (float(statistics.median([v for v in r_on if v is not None]))
                        if any(v is not None for v in r_on) else None),
        "c1_passed": c1_passed,
        "n_seeds_converting": int(n_converting),
        "conversion_per_seed": per_seed,
        "c1b_passed": c1b_passed,
        "n_seeds_on_above_off": int(n_on_above_off),
        "attribution_per_seed": attribution_per_seed,
        "off_arm_green": bool(off_green),
        "H_on_mean": _mean([float(on[s]["committed_action_class_entropy_mm"]) for s in seeds]),
        "H_off_mean": _mean([float(off[s]["committed_action_class_entropy_mm"]) for s in seeds if s in off]),
        "H_control_means": {lv: _mean([float(ctrls[lv][s]["committed_action_class_entropy_mm"])
                                       for s in seeds if s in ctrls[lv]]) for lv in CONTROL_LEVERS},
        "noise_control_lifts": noise_lifts,
        "n_seeds_noise_lifts": int(n_noise_lifts),
        "noise_control_per_seed": noise_per_seed,
        "realized_harm_per_step_on": harm_on,
        "realized_harm_per_step_off": harm_off,
        "seeds_required": SEEDS_REQUIRED,
        "r_bar": R_BAR,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = list(DRY_RUN_SEEDS if dry_run else SEEDS)
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1_cap = DRY_RUN_P1_CAP if dry_run else P1_EPISODE_CAP
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    fresh_target = DRY_RUN_FRESH_TARGET if dry_run else N_FRESH_SELECT_TARGET

    arm_ctxs = [
        {"id": a["id"], "regime": a["regime"], "lever": a["lever"], "role": a["role"],
         "feed_residue": a["feed_residue"], "warmup": a["warmup"], "instrumented": a["instrumented"]}
        for a in ARMS
    ]
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(run_cell(arm, seed, p0, p1_cap, steps, fresh_target, dry_run=dry_run))

    by_arm = {a["id"]: [r for r in rows if r["arm"] == a["id"]] for a in ARMS}
    arm_gates = [_arm_gate(a, by_arm[a["id"]]) for a in ARMS]
    gate = aggregate_arm_gates(arm_gates)
    green = set(gate["green_arms"])

    inst_rows = [r for r in rows if r["instrumented"]]
    n_resid = sum(r["n_residual_failures"] for r in inst_rows)
    n_self = sum(r["n_self_check_failures"] for r in inst_rows)
    n_offc = sum(r["n_off_crosscheck_failures"] for r in inst_rows)
    worst_i2 = max((r["i2_mismatch_rate"] for r in inst_rows), default=0.0)
    on_off_separated = any(
        r["R_ON"] is not None and r["R_OFF"] is not None and abs(r["R_ON"] - r["R_OFF"]) > 1e-9
        for r in inst_rows
    )
    instrument_ok = bool(
        n_resid == 0 and n_self == 0 and n_offc == 0
        and worst_i2 <= I2_MAX_MISMATCH_RATE and on_off_separated
    )

    identical_pairs = _identical_control_pairs(rows)
    control_distinct = bool(len(identical_pairs) == 0)

    readings = {regime: _regime_reading(rows, regime, green) for regime in REGIMES}
    noise_lifts_all = all(readings[r]["noise_control_lifts"] for r in REGIMES)
    any_arm_green = bool(green)
    c2_all = all(readings[r]["c2_passed"] for r in REGIMES)
    c1_all = all(readings[r]["c1_passed"] for r in REGIMES)
    c1_any = any(readings[r]["c1_passed"] for r in REGIMES)
    c1b_all = all(readings[r]["c1b_passed"] for r in REGIMES)

    # --- Verdict chain: instrument and control validity precede any claim verdict ---
    if not instrument_ok:
        outcome, label, direction = "FAIL", "instrument_defect", "non_contributory"
    elif not control_distinct:
        outcome, label, direction = "FAIL", "control_arms_not_distinct_invalid", "non_contributory"
    elif not noise_lifts_all:
        outcome, label, direction = "FAIL", "matched_noise_control_unmeetable", "non_contributory"
    elif not any_arm_green:
        outcome, label, direction = "FAIL", "substrate_not_ready_requeue", "non_contributory"
    elif not c2_all:
        outcome, label, direction = "FAIL", "eligibility_not_commensurate_requeue", "non_contributory"
    elif c1_all and c1b_all:
        outcome, label, direction = "PASS", "commensurate_eligibility_converts", "supports"
    elif c1_all and not c1b_all:
        # C1 cleared the controls but ARM_ON did not beat its own operator-OFF twin: the
        # lift is not attributable to the operator (most likely the e2wf summary source,
        # which ARM_ON shares with ARM_OFF and the controls do not). NOT a claim verdict.
        outcome, label, direction = ("FAIL",
                                     "conversion_not_attributable_to_operator",
                                     "non_contributory")
    elif c1_any:
        outcome, label, direction = "FAIL", "mixed_by_regime", "non_contributory"
    else:
        outcome = "FAIL"
        label = "conversion_ceiling_persists_under_commensurate_eligibility"
        direction = "non_contributory"

    non_degenerate = bool(
        instrument_ok and control_distinct and noise_lifts_all and any_arm_green and c2_all
    )
    degeneracy_reason = None
    if not non_degenerate:
        degeneracy_reason = (
            "The conversion contrast was not validly evaluated: instrument_ok=%s "
            "control_distinct=%s noise_control_lifts=%s any_arm_green=%s "
            "eligibility_commensurate=%s. Per V3-EXQ-936a's registered combination_rule a "
            "conversion criterion that was not validly evaluated is scored non_contributory "
            "/ non_degenerate, never a direction." % (
                instrument_ok, control_distinct, noise_lifts_all, any_arm_green, c2_all)
        )

    criteria: List[Dict[str, Any]] = [
        {
            "name": "instrument_clean",
            "load_bearing": True,
            "role": "instrument correctness gate",
            "passed": bool(instrument_ok),
            "measured": float(n_resid + n_self + n_offc),
            "threshold": 0.0,
            "direction": "upper",
            "detail": ("I1 residual / I1b replay self-check / I1c OFF cross-check failures, "
                       "plus worst I2 mismatch %.4f <= %.4f and ON/OFF reconstruction separated=%s"
                       % (worst_i2, I2_MAX_MISMATCH_RATE, on_off_separated)),
        },
        {
            "name": "C_CONTROL_DISTINCT",
            "load_bearing": True,
            "role": "control validity gate (689i defect-2 repair)",
            "passed": bool(control_distinct),
            "measured": float(len(identical_pairs)),
            "threshold": 0.0,
            "direction": "upper",
            "detail": "(regime, seed) pairs of non-treatment arms with identical committed-class count vectors",
        },
    ]
    for regime in REGIMES:
        rd = readings[regime]
        criteria.append({
            "name": "C_NOISE_LIFTS_%s" % regime,
            "load_bearing": True,
            "role": "negative-control validity gate",
            "passed": bool(rd["noise_control_lifts"]),
            "measured": float(rd["n_seeds_noise_lifts"]),
            "threshold": float(SEEDS_REQUIRED),
            "detail": ("matched-noise committed-class entropy strictly above the collapsed-proposer "
                       "control on >= %d of %d seeds" % (SEEDS_REQUIRED, len(rd["seeds"]))),
        })
        criteria.append({
            "name": "C2_eligibility_commensurate_%s" % regime,
            "load_bearing": True,
            "role": "manipulation check -- the ratified rung-3 gate (governance 2026-09-24)",
            "passed": bool(rd["c2_passed"]),
            "measured": float(rd["n_seeds_R_ON_at_bar"]),
            "threshold": float(SEEDS_REQUIRED),
            "per_seed_measured": rd["R_ON_per_seed"],
            "per_seed_threshold": R_BAR,
            "detail": "R_ON >= %s in >= %d of %d seeds, with same-tick R_OFF / R_ORACLE anchors"
                      % (R_BAR, SEEDS_REQUIRED, len(rd["seeds"])),
        })
        criteria.append({
            "name": "C1b_operator_attribution_%s" % regime,
            "load_bearing": True,
            "role": "attribution -- isolates the operator from the candidate_summary_source",
            "passed": bool(rd["c1b_passed"]),
            "measured": float(rd["n_seeds_on_above_off"]),
            "threshold": float(SEEDS_REQUIRED),
            "per_seed_measured": [p["margin_vs_off"] for p in rd["attribution_per_seed"]],
            "per_seed_threshold": 0.0,
            "detail": ("ARM_ON committed-class entropy (Miller-Madow) strictly above its own "
                       "operator-OFF twin ARM_OFF at the same seed, on >= %d of %d seeds. "
                       "ARM_OFF differs from ARM_ON only in the operator, so this is the "
                       "single-variable contrast the two controls cannot supply."
                       % (SEEDS_REQUIRED, len(rd["seeds"]))),
        })
        criteria.append({
            "name": "C1_conversion_%s" % regime,
            "load_bearing": True,
            "role": "verdict",
            "passed": bool(rd["c1_passed"]),
            "measured": float(rd["n_seeds_converting"]),
            "threshold": float(SEEDS_REQUIRED),
            "per_seed_measured": [p["margin_vs_worst_control"] for p in rd["conversion_per_seed"]],
            "per_seed_threshold": 0.0,
            "detail": ("ARM_ON committed-action-class entropy strictly above BOTH the "
                       "collapsed-proposer and matched-noise controls at the same seed, on "
                       ">= %d of %d seeds" % (SEEDS_REQUIRED, len(rd["seeds"]))),
        })

    criteria_non_degenerate = {c["name"]: bool(non_degenerate) for c in criteria}
    criteria_non_degenerate["instrument_clean"] = True
    criteria_non_degenerate["C_CONTROL_DISTINCT"] = True
    for regime in REGIMES:
        criteria_non_degenerate["C_NOISE_LIFTS_%s" % regime] = bool(instrument_ok and control_distinct)
        criteria_non_degenerate["C2_eligibility_commensurate_%s" % regime] = bool(
            instrument_ok and on_off_separated and _arm_id(regime, "ON") in green)
        criteria_non_degenerate["C1_conversion_%s" % regime] = bool(
            instrument_ok and control_distinct and readings[regime]["noise_control_lifts"]
            and readings[regime]["c2_passed"] and readings[regime]["controls_green"])
        criteria_non_degenerate["C1b_operator_attribution_%s" % regime] = bool(
            instrument_ok and readings[regime]["c2_passed"]
            and readings[regime]["off_arm_green"] and readings[regime]["on_arm_green"])

    combination_rule = (
        "Verdict chain, in order: instrument_clean AND C_CONTROL_DISTINCT AND C_NOISE_LIFTS "
        "AND (>=1 arm gate green) AND C2 must ALL hold before any claim direction is "
        "assigned; any failure -> non_contributory with the corresponding self-route label. "
        "Then BOTH conversion criteria must hold for 'supports': C1 (ARM_ON above BOTH "
        "controls -- MECH-439's own registered bar) AND C1b (ARM_ON above its operator-OFF "
        "twin ARM_OFF -- the attribution contrast, since ARM_ON differs from the controls in "
        "candidate_summary_source as well as the operator). C1 true with C1b FALSE -> "
        "conversion_not_attributable_to_operator / non_contributory, NOT a claim verdict. "
        "Once they hold: C1 true in BOTH regimes -> PASS / "
        "supports (conversion required rebalancing the score-scale monopoly, which is "
        "MECH-439's own conditional -- inherited verbatim from V3-EXQ-936a's registered "
        "combination_rule 'C2 true -> supports'). C1 false in both -> FAIL / "
        "non_contributory, labelled conversion_ceiling_persists_under_commensurate_eligibility: "
        "MECH-439's confirming conjunct (F share > 0.85) is unreachable under the operator, so "
        "936a's 'C1 false otherwise ... the falsifier was NOT evaluated' applies and the cell's "
        "value is ROUTING (to the reserved final-commit-stage replay, 1012c autopsy sec 7b), "
        "not direction. C1 true in exactly one regime -> mixed_by_regime / non_contributory. "
        "NOTE: this design cannot produce 'weakens'. MECH-439's registered falsifier requires a "
        "lift WITHOUT a reduction in F's cross-candidate variance share, and the operator "
        "reduces that share by construction, so the falsifying antecedent has no instance in "
        "this regime (GFLAG-0471)."
    )

    preconditions = list(gate.get("adjudication_preconditions") or [])

    interpretation = {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": combination_rule,
        "scope_limits": [
            "Eligibility stage only: the operator acts on the margin-eligible set E; the final "
            "committed argmin WITHIN E -- MECH-439's own Consumer/boundary -- is untouched, so a "
            "flat result does not separate 'no cap' from 'cap at the final-commit stage'.",
            "3 content channels only (f, harm, residue); benefit and goal read 0 content ticks in "
            "this regime (measured 1012c, 8/8 cells).",
            "Cannot falsify MECH-439 -- support-or-route only (see combination_rule).",
            "ready on f_dominance_conversion_ceiling STAYS false: SD-018 field-head validation and "
            "SD-e1 var-bar re-registration are untouched by this run.",
            "Inherited regime-wide scope bound: open corrupting entry "
            "contextmemory-write-path-addressing-degeneracy is live and fires identically in every "
            "arm of this paired-by-seed within-regime design; it bounds external validity only.",
        ],
        "claim_text_restatement": "GFLAG-0471 (open; supersedes GFLAG-0469)",
    }

    # Miller-Madow audit: would the uncorrected plug-in estimator have changed the verdict?
    def _counterfactual(key: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for regime in REGIMES:
            on_r = _by(rows, regime, "ON")
            off_r = _by(rows, regime, "OFF")
            ctl_r = {lv: _by(rows, regime, lv) for lv in CONTROL_LEVERS}
            seeds_r = sorted(on_r.keys())
            n_c = sum(1 for sd in seeds_r
                      if all(sd in ctl_r[lv] and float(on_r[sd][key]) > float(ctl_r[lv][sd][key])
                             for lv in CONTROL_LEVERS))
            n_a = sum(1 for sd in seeds_r
                      if sd in off_r and float(on_r[sd][key]) > float(off_r[sd][key]))
            out[regime] = {"n_seeds_converting": n_c, "n_seeds_on_above_off": n_a}
        return out

    cf_plugin = _counterfactual("committed_action_class_entropy")
    cf_routed = _counterfactual("committed_action_class_entropy_mm")
    estimator_changes_verdict = bool(cf_plugin != cf_routed)

    diagnostics = {
        "identical_control_pairs": identical_pairs,
        "miller_madow_audit": {
            "routed_estimator": "committed_action_class_entropy_mm (Miller-Madow)",
            "plugin_estimator": "committed_action_class_entropy",
            "per_cell_correction_nats": {
                "%s/seed%s" % (r["arm"], r["seed"]): r["miller_madow_correction_nats"]
                for r in rows},
            "n_committed_samples_per_cell": {
                "%s/seed%s" % (r["arm"], r["seed"]): r["n_committed_samples"] for r in rows},
            "fixed_n_target": N_FRESH_SELECT_TARGET,
            "all_cells_at_fixed_n": bool(rows and all(r["dv_sample_cap_met"] for r in rows)),
            "counterfactual_under_plugin": cf_plugin,
            "counterfactual_under_routed": cf_routed,
            "estimator_changes_verdict": estimator_changes_verdict,
        },
        "entropy_headroom_per_arm": per_arm_headroom(
            rows, value_key="committed_action_class_entropy_mm",
            low=0.0, high=MAX_COMMITTED_CLASS_ENTROPY, arm_key="arm"),
        "per_regime": readings,
        "gflag0072_remeasurement": {
            "note": ("GFLAG-0072's ceiling is denominated on PRE-COMMIT class entropy in "
                     "V3-EXQ-708b's regime. Measured here on the COMMITTED DV, per arm and seed; "
                     "the control_entropy_headroom precondition gates on it rather than assuming it."),
            "max_committed_class_entropy_nats": MAX_COMMITTED_CLASS_ENTROPY,
            "control_headroom_floor_nats": CONTROL_HEADROOM_FLOOR,
            "prior_936a_committed_entropy_off_arm": [0.4788, 0.6387, 0.9581, 1.1887],
            "prior_936a_committed_entropy_demotion_arm": [0.6930, 0.9657, 1.2102, 1.4203],
        },
    }

    summary = {
        "per_regime": readings,
        "instrument_ok": instrument_ok,
        "on_off_separated": on_off_separated,
        "n_residual_failures_total": int(n_resid),
        "n_self_check_failures_total": int(n_self),
        "n_off_crosscheck_failures_total": int(n_offc),
        "worst_i2_mismatch_rate": float(worst_i2),
        "control_distinct": control_distinct,
        "n_identical_control_pairs": int(len(identical_pairs)),
        "c1_all_regimes": c1_all,
        "c1b_all_regimes": c1b_all,
        "c2_all_regimes": c2_all,
        "estimator_changes_verdict": estimator_changes_verdict,
        "all_cells_at_fixed_n": bool(rows and all(r["dv_sample_cap_met"] for r in rows)),
        "green_arms": sorted(green),
        "red_arms": sorted(gate.get("red_arms") or []),
    }

    outcome_note = (
        "%s: per regime, C1 seeds converting %s (bar %d), C2 seeds at R_ON>=%s %s; "
        "H_on mean %s vs controls %s; noise-control lifts %s; instrument residual=%d "
        "self_check=%d off_crosscheck=%d worst_i2=%.4f; identical control pairs=%d; "
        "gate green=%s. Direction map per GFLAG-0471 / V3-EXQ-936a combination_rule; this "
        "design cannot produce 'weakens'."
        % (label,
           {r: readings[r]["n_seeds_converting"] for r in REGIMES}, SEEDS_REQUIRED, R_BAR,
           {r: readings[r]["n_seeds_R_ON_at_bar"] for r in REGIMES},
           {r: round(readings[r]["H_on_mean"], 4) for r in REGIMES},
           {r: {k: round(v, 4) for k, v in readings[r]["H_control_means"].items()} for r in REGIMES},
           {r: readings[r]["noise_control_lifts"] for r in REGIMES},
           n_resid, n_self, n_offc, worst_i2, len(identical_pairs), sorted(green))
    )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "evidence_direction": direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_arm_gate": gate,
        "arm_results": rows,
        "interpretation": interpretation,
        "criteria": criteria,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def _flat(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-1095: MECH-439 operator-ON conversion falsifier at the eligibility stage"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="smoke: 1 seed, short P0/P1, relaxed content-tick floor")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, timestamp)

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
            "MIN_FRESH_SELECTIONS": MIN_FRESH_SELECTIONS,
            "N_FRESH_SELECT_TARGET": N_FRESH_SELECT_TARGET,
            "R_BAR": R_BAR,
            "SEEDS_REQUIRED": SEEDS_REQUIRED,
            "CONTENT_FRAC": CONTENT_FRAC,
            "MIN_CONTENT_TICKS": MIN_CONTENT_TICKS,
            "RESIDUAL_REL_TOL": RESIDUAL_REL_TOL,
            "SELF_CHECK_TOLERANCE": SELF_CHECK_TOLERANCE,
            "I2_MAX_MISMATCH_RATE": I2_MAX_MISMATCH_RATE,
            "CONTROL_HEADROOM_FLOOR": CONTROL_HEADROOM_FLOOR,
            "MIN_POOL_FIRST_ACTION_CLASSES": MIN_POOL_FIRST_ACTION_CLASSES,
            "MAX_COMMITTED_CLASS_ENTROPY": MAX_COMMITTED_CLASS_ENTROPY,
            "NOISE_ARM_COMMIT_ENTROPY_ALPHA": NOISE_ARM_COMMIT_ENTROPY_ALPHA,
        },
        "arms": [dict(a) for a in ARMS],
        "arm_config_slices": {
            a["id"]: config_slice_for(a, p0_used, p1_cap_used, steps_used, fresh_used) for a in ARMS
        },
        "dry_run": bool(args.dry_run),
    }

    s = result["summary"]
    readout: Dict[str, float] = {}
    for regime in REGIMES:
        rd = s["per_regime"][regime]
        for key in ("n_seeds_converting", "n_seeds_on_above_off", "n_seeds_R_ON_at_bar",
                    "n_seeds_noise_lifts", "H_on_mean", "H_off_mean", "R_ON_median",
                    "c1_passed", "c1b_passed", "c2_passed", "noise_control_lifts",
                    "realized_harm_per_step_on", "realized_harm_per_step_off"):
            v = _flat(rd.get(key))
            if v is not None:
                readout["%s_%s" % (regime, key)] = v
        for lv, hv in rd["H_control_means"].items():
            v = _flat(hv)
            if v is not None:
                readout["%s_H_control_%s" % (regime, lv)] = v
    for key in ("instrument_ok", "on_off_separated", "control_distinct",
                "n_identical_control_pairs", "n_residual_failures_total",
                "n_self_check_failures_total", "n_off_crosscheck_failures_total",
                "worst_i2_mismatch_rate", "c1_all_regimes", "c1b_all_regimes",
                "c2_all_regimes", "estimator_changes_verdict", "all_cells_at_fixed_n"):
        v = _flat(s.get(key))
        if v is not None:
            readout[key] = v

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_note": (
            "Direction map inherited from V3-EXQ-936a's registered combination_rule with the "
            "commensurability operator substituted for its C2 share-reduction, and restated for "
            "the operator-ON regime in GFLAG-0471 (open at queue time). 'supports' is emitted "
            "ONLY when every instrument and control gate holds, the ratified eligibility-stage "
            "gate is met in both regimes, and C1 fires in both. This design CANNOT emit "
            "'weakens': MECH-439's registered falsifier requires a lift WITHOUT a reduction in "
            "F's cross-candidate variance share, and the operator reduces that share by "
            "construction. A flat result is non_contributory and routes to the reserved "
            "final-commit-stage replay (failure_autopsy_V3-EXQ-1012c_2026-09-24 sec 7b). "
            "V3-EXQ-1012c is NOT read as MECH-439 evidence anywhere in this run."
        ),
        "evidence_direction_per_claim": {"MECH-439": result["evidence_direction"]},
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
        "readout": readout,
        "custom_information": {
            "governance_authority": (
                "substrate_queue f_dominance_conversion_ceiling governance_2026_09_24 "
                "(refusal RELEASED, scoped to operator ON in this regime with the 3-channel "
                "scope stated); failure_autopsy_V3-EXQ-1012c_2026-09-24 sec 7b"
            ),
            "claim_text_restatement_flag": "GFLAG-0471 (open; supersedes GFLAG-0469)",
            "chip_ref": "chip-20260924-mech439-operator-on-falsifier",
            "predecessors": (
                "V3-EXQ-936a (same regime, operator OFF, MECH-448 demotion lever, committed-class "
                "entropy 0.479-1.420); V3-EXQ-689i (the four-arm control contrast, top_k regime); "
                "V3-EXQ-1012c (the ratified eligibility-stage gate this run re-measures in situ)"
            ),
            "gov_reuse_1_check": (
                "Decisive readout: committed-action-class entropy under the commensurability "
                "operator, paired against a collapsed-proposer and a matched-noise control. "
                "reanalysis_query.py query --readout committed_action_class_entropy --claim "
                "MECH-439 (2026-09-24): 12 manifests, 0 carry the readout on any substrate_hash; "
                "the only operator-ON runs (1012a, 1012c) record commit-flip rate and "
                "eligibility-knockout R, carry no control arms, and 1012c is explicitly not to be "
                "read as MECH-439 evidence. The manipulation (operator ON) is present in no "
                "recorded run with this DV. Not recoverable -> run."
            ),
            "re_derive_brake_note": (
                "MECH-439 brake count 15 (threshold 2), re-derived 2026-09-24. RELEASED on both "
                "routes: (1) the named upstream substrate SD-E3-CHANNEL-COMMENSURABILITY reads "
                "IMPLEMENTED (2026-09-07) and VALIDATED at the eligibility stage in ree-v3 "
                "CLAUDE.md; (2) the confirmed 1012c autopsy sec 7c states the permitted shape "
                "explicitly -- 'The permitted shape is the redesigned, operator-ON falsifier in "
                "7b, and that is a user release decision' -- and governance 2026-09-24 made that "
                "release. NOT a lettered same-granularity re-test: new EXQ number, a lever that "
                "has never been tested against this DV, and the four-arm control contrast the "
                "936 family never carried."
            ),
            "step_2_5c_known_limitations": (
                "Open corrupting entries whose execution condition is ABSENT on this config "
                "(measured at authoring time): MECH-320 (use_tonic_vigor=False), "
                "sd_blocked_agency (use_blocked_agency=False), sd105 "
                "(use_selection_entropy_floor=False), sd_zself_training_path "
                "(latent_stack.self_recurrence is None), SD-PP-B5 (its ::function "
                "compute_world_interventional_loss is not called; its substantive "
                "candidate-compression finding is GATED by candidate_first_action_classes and "
                "control_entropy_headroom rather than dismissed). LIVE and carried as an "
                "inherited regime-wide scope bound, as in 1012c: "
                "contextmemory-write-path-addressing-degeneracy."
            ),
            "gflag0072_note": (
                "support_preserving_min_first_action_classes stays at the lineage value 2 -- "
                "raising it would break the regime identity the governance release is scoped to. "
                "GFLAG-0072's arm-invariant ceiling is denominated on PRE-COMMIT class entropy in "
                "V3-EXQ-708b's regime; V3-EXQ-936a's landed manifest on THIS baseline measures "
                "committed-class entropy varying by arm at every seed (0.479-1.420 over 2-5 "
                "classes, ln(5)=1.6094). The concern is converted into the gating "
                "control_entropy_headroom precondition rather than assumed either way."
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
    if result["degeneracy_reason"]:
        manifest["degeneracy_reason"] = result["degeneracy_reason"]

    stamp_recording_core(
        manifest, config=full_config, seeds=seeds_used, script_path=Path(__file__), started_at=t0,
        agent=_LAST_AGENT["agent"], z_goal_stream_stats=_ZG.stats(),
    )

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=bool(args.dry_run),
        config=full_config, seeds=seeds_used, script_path=Path(__file__), started_at=t0,
        agent=_LAST_AGENT["agent"], z_goal_stream_stats=_ZG.stats(), json_default=str,
    )
    print("Manifest written: %s" % out_path, flush=True)
    print("LABEL: %s" % result["interpretation"]["label"], flush=True)
    for c in result["criteria"]:
        print("  %s: %s (measured %s, threshold %s)"
              % (c["name"], c["passed"], c["measured"], c["threshold"]), flush=True)
    print("  per_arm_gate: green=%s red=%s"
          % (result["per_arm_gate"]["green_arms"], result["per_arm_gate"]["red_arms"]), flush=True)

    if args.dry_run:
        assert s["n_residual_failures_total"] == 0, "SMOKE FAIL: I1 residual -- capture/reconstruction bug"
        assert s["n_self_check_failures_total"] == 0, "SMOKE FAIL: I1b replay self-check"
        assert s["n_off_crosscheck_failures_total"] == 0, "SMOKE FAIL: I1c OFF cross-check"
        # The DV must be non-trivially engaged BEFORE the full 32-cell grid is committed to
        # (skill Step 3.5: a structural zero on an evidence run's decisive readout costs the
        # same as on a diagnostic).
        n_samples = [int(r["n_committed_samples"]) for r in result["arm_results"]]
        assert min(n_samples) > 0, "SMOKE FAIL: a cell recorded zero committed-class samples"
        n_classes = [int(r["n_committed_classes"]) for r in result["arm_results"]]
        assert max(n_classes) >= 2, "SMOKE FAIL: no cell committed more than one action class"
        # The manipulation must reach the DV: at least two arms must differ at the same seed.
        by_seed: Dict[int, List[float]] = {}
        for r in result["arm_results"]:
            by_seed.setdefault(int(r["seed"]), []).append(float(r["committed_action_class_entropy_mm"]))
        assert any(len(set(v)) > 1 for v in by_seed.values()), \
            "SMOKE FAIL: every arm produced an identical committed-class entropy at every seed"
        print("DRY RUN complete.", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
