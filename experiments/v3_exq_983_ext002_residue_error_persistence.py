#!/opt/local/bin/python3
"""
V3-EXQ-983 -- EXT-002: does the residue field leave a persistent error record?

Claim: EXT-002 (external_failure_mode, subject llm.hallucination) -- "Hallucination:
no persistent error residue accumulates to shape future outputs." Proposal: EXP-0526
(backlog_id EVB-1247). REE mechanism: ARC-005, INV-006, INV-008.
ree_failure_mode_analog: moral_amnesia.

Claim notes (verbatim intent): "Transformer LLMs generate each token from the current
context window without any mechanism for prediction errors to leave a persistent trace
that penalises future outputs. A false statement incurs no enduring cost. REE's residue
field phi(z) functions as a persistent error record: E1 prediction errors accumulate
into the latent terrain and actively modulate trajectory selection on subsequent steps,
creating structural pressure against repeat errors."

WHAT IS ACTUALLY UNDER TEST, AND WHY "ERROR" = HARM HERE (not literal E1 loss)
-------------------------------------------------------------------------------
The claim's prose uses "E1 prediction error" as the general concept of "an unexpected,
costly outcome that should leave a lasting trace." Reading `ree_core/agent.py`'s
`update_residue()` (the substrate's actual accumulation call site,
`ree_core/agent.py:10641-10649`) shows the residue field accumulates on
`harm_signal < 0` -- i.e. on HARM, not on a raw E1 world-model loss value. Both
`agent.update_residue()` (production path) and `V3-EXQ-800`'s manual
`agent.residue_field.accumulate(z_world, harm_magnitude=...)` (this script's direct
precedent) gate accumulation the same way. So "error" is operationalised here as a
harm event: the substrate's actual, wired implementation of "a costly outcome the
agent's world-model did not want", and residue accumulation is the substrate's actual
implementation of "a persistent trace of that error". This keeps the DV causally wired
to the manipulated mechanism (see DV-symmetry declaration below) rather than to a
quantity (raw E1 MSE loss) the residue field does not read.

The literal E1/E3 "prediction_error" signal (`e3_selector.py:4247`,
`actual_z_world - predicted_world` from the E3-selected candidate's own rollout) is
used instead as the MANDATORY POSITIVE CONTROL (contract item 13): see
`_positive_control_e1_prediction_error()` below. It confirms the substrate's
world-model genuinely registers a non-degenerate error signal on a real hazard
collision -- linking the harm-triggered accumulation this script measures back to the
literal "prediction error" language in the claim notes -- WITHOUT gating the scored
arms' DV on it (that would decouple the DV from the mechanism actually manipulated;
see "why the P0 probe is a separate throwaway agent" below).

DESIGN -- two arms, residue read identically, accumulation the manipulation
-----------------------------------------------------------------------------
  A0_INTACT           residue accumulates from every harm event and is consulted at
                      action-selection time (E3 argmin over residue-scored candidates,
                      same selection rule as V3-EXQ-800's A0/A2/A3).
  A1_RESIDUE_FROZEN   residue field frozen at initialisation -- never accumulates --
                      but is READ via the identical E3 argmin path. This isolates
                      "does the persistent trace exist and shape selection" without a
                      scramble-correspondence leg (that is ARC-007's question, tested
                      by V3-EXQ-800, not this one).

Both arms are otherwise IDENTICAL: same env, same E1/E2 training, same CEM candidate
budget, same action-selection code path. The only difference is whether harm events
ever get written into `agent.residue_field`.

OPERATIONAL DEFINITION OF "THE SAME ERROR RECURRING" (the design's crux)
--------------------------------------------------------------------------
A "matched (state-region, action) pair" is operationalised EXACTLY, using the grid
world's own discrete coordinates (the same convention V3-EXQ-800 uses for
`visited_cells`):

    key = (pre_action_grid_x, pre_action_grid_y, executed_action_class)

captured BEFORE `env.step()` (the agent's actual decision point) and paired with the
REALISED action class (from `agent.select_action`'s E3 output, never the collapsed
`action_object_decoder` round trip -- see `_select_action` below).

  first_harm_at[key] := the (chronological, whole-training-phase) step index of the
                         FIRST time a step at `key` produced `harm_signal < 0`.
  A REVISIT of `key` (any later step landing on the identical key, harm or not) is
    scored as a REPEAT if `harm_signal < 0` again, else NOT a repeat.
  repeat_error_rate(period) := mean(repeat indicator) over all revisits whose step
    index falls in `period` (EARLY = first half of the planned training-step budget,
    LATE = second half; a fixed, pre-registered split point, not data-dependent).

This is P(harm | a revisit of a key that has already produced harm once), computed
separately for the early and late halves of training. REE's prediction: A0_INTACT's
repeat_error_rate DECLINES from early to late (residue-driven structural pressure
against repeating a known-costly action-region pairing); A1_RESIDUE_FROZEN's stays
flat (no persistent cost -- the hallucination-analog failure mode the claim names).
`decline := repeat_error_rate(early) - repeat_error_rate(late)`; positive = pressure
observed.

DV-SYMMETRY DECLARATION (mandatory; one line per arm)
-----------------------------------------------------
DV = `decline`, a function of the harm-event sequence at matched (cell, action) keys,
which is itself a function of the executed action stream, which is produced by an
E3 ARGMIN over per-candidate residue trajectory scores (same selection rule and same
symmetry group as V3-EXQ-800: (i) permutation of candidate INDEX order, (ii) any
uniform additive constant applied to all candidate scores, (iii) any monotone
rescaling of all candidate scores -- none of which can move an argmin).
  A0  reference arm: residue accumulates from real harm events (magnitude
      `abs(harm_signal) * accumulation_rate`, `ree_core/residue/field.py:665`) at a
      NEW RBF center each event, so later candidate-trajectory scores differ from
      earlier ones by an amount that depends on which regions were actually harmed --
      neither a broadcast constant (per-candidate trajectories pass through different
      z_world regions) nor a monotone map of the earlier score vector. Not invariant.
  A1  freezes the field at init, so EVERY step's candidate scores are computed
      against the SAME (untrained, harm-history-blind) field regardless of calendar
      time -- late-training selection is computed from literally the same source as
      early-training selection. This removes the experience-dependent differentiation
      the claim's mechanism requires; the remaining per-candidate spread from the
      untrained `neural_field` term (nonzero because different candidates pass
      through different z_world points even pre-training) is not a broadcast constant
      or monotone rescaling either, so A1's argmin is not literally a constant
      function -- but it carries no memory of PAST harm, which is exactly the
      manipulation this experiment tests. Not invariant, and not the mechanism
      the claim asserts.

NON-DEGENERACY PRECONDITIONS (breach -> substrate_not_ready_requeue, NOT a verdict)
------------------------------------------------------------------------------------
  P1 residue_structure_live      A0 only: cross-center weight variance of the trained
                                 residue field > floor. A flat field means no trace was
                                 ever written and a null decline result would be
                                 meaningless. Scoped OUT of A1 (frozen BY DESIGN --
                                 asserting structure there is structurally
                                 unsatisfiable, not a substrate fact).
  P2 candidate_score_range       Both arms: cross-candidate residue-score RANGE on a
                                 post-training positive-control probe -- the SAME
                                 statistic the E3 argmin routes on (V3-EXQ-643 rule).
                                 Both arms read the field via this path.
  P3 e1_prediction_error_floor   Global (contract item 13): E3.post_action_update's
                                 world-model prediction-error metric on a genuine
                                 hazard collision, measured on a THROWAWAY agent
                                 isolated from both scored arms. Confirms the
                                 substrate's error signal is non-degenerate before a
                                 later null is trusted as meaningful.
  P4 revisit_denominator         Both arms: minimum number of (cell,action) revisit
                                 events observed in the EARLY training half only
                                 (DECOUPLED from C4 -- see below). Below this the
                                 early-half repeat rate itself is undefined or
                                 dominated by single-sample noise.
  P5 harm_rate_not_saturated     Both arms: training harm_rate must stay below a
                                 ceiling. A rate near 1.0 is the mechanical
                                 signature of a harm source independent of true
                                 hazards (e.g. contamination-spread self-harm) and
                                 pins repeat_rate_early/late near 1.0 in BOTH arms
                                 regardless of any true residue effect -- a general
                                 degeneracy guard, not tied to one named confound.

Per-arm gates are aggregated with experiments/_lib/precondition_gate.py. UNLIKE
V3-EXQ-800 (whose four-arm design lets one manipulated arm's result stand alone),
THIS design's primary statistic (`decline_gap = decline_A0 - decline_A1`) is a PAIRED
comparison and is only interpretable when BOTH arms are green -- so this script uses
`gate["all_green"]`, not the module's default `any_green`, as its own readiness
verdict (see `run()`; `per_arm_gate` is still recorded verbatim for audit).

PRE-REGISTERED THRESHOLDS (constants below; never inferred post-hoc)
----------------------------------------------------------------------
  C1 (LOAD-BEARING) decline_gap = decline_A0 - decline_A1 >= 0.15 (15 percentage
     points). [RE-DERIVED post red-team fix -- contamination_spread=0.0 removes
     the self-manufactured, hazard-independent harm source. Remaining harm comes
     from real hazard contact (hazard_harm=0.02 + proximity_harm_scale=0.05 at
     contact) and proxy-field "hazard_approach" (proximity_harm_scale * h_field,
     up to 0.05, on any non-contact step where the hazard field is above
     proximity_approach_threshold=0.15 -- MECH-203: this wins by default over
     "benefit_approach" whenever it is active, `causal_grid_world.py:2495-2506`).
     With num_hazards=4 on a 6x6 grid this proxy channel is expected to be the
     dominant harm source, not contact -- see dry-run smoke output for the
     measured harm_rate_train under the fixed config; if it is still saturated
     near 1.0 the new P5 gate (below) routes the run to
     substrate_not_ready_requeue rather than a false verdict.] Arithmetic check:
     accumulation_rate=0.1 -> ~0.002-0.01 residue mass per harm event, hundreds
     of events over 300x200-step warmup -> ample field structure (same config
     V3-EXQ-800 validated against the identical 1e-6 weight-variance floor). If
     A0's repeat rate falls from a naive pre-learning baseline toward a
     materially lower learned-avoidance rate while A1 stays within noise of its
     early value, the realised gap is comfortably >0.15; a null effect realises
     ~0.0 and correctly fails C1. THRESHOLD ITSELF UNCHANGED at 0.15 -- the env
     config was wrong (2a-i), not the threshold; see the smoke-test numbers
     recorded in the queueing session's report for the post-fix magnitude check.
  C2 effect >= 0.80 SD of the cross-seed paired delta (decline_A0 - decline_A1).
  C3 direction consistent on >= 2 of 3 seeds.
  C4 data quality (VERDICT-level, unchanged): min revisit count over BOTH halves,
     both arms >= 20 -- with 3 seeds x 200 warmup episodes x 200 steps = 40000
     steps over a 6x6 grid x n_actions keys, hazard-adjacent keys are visited
     repeatedly; 20 is a conservative floor well below the expected count. C4 is
     deliberately STRICTER than, and decoupled from, P4 (readiness) -- see P4
     above and Family-4 in the red-team-verdict note.

SLEEP: not used (no sleep flags set) -- no SLEEP DRIVER line required.

RED-TEAM (fable, Step 4.5, 2026-09-02): BLOCKING -> fixed. Findings and fixes:
  - 2a-i (CONFIRMED): `_make_env` never set `contamination_spread`, defaulting to
    0.5 (`causal_grid_world.py:135`). The env's own docstring (lines 71-77) names
    this exact trap: revisited cells cross `contamination_threshold` after a
    handful of contacts and deal `contaminated_harm=0.4` regardless of hazards,
    independent of residue -- the mechanical cause of the dry-run's harm_rate~1.0.
    Fixed: `contamination_spread=0.0` passed explicitly (V3-EXQ-513 precedent, the
    env docstring's own recommended fix).
  - 2a-ii (CONFIRMED): `half_point` was computed from the PLANNED step budget
    (`n_warmup * n_steps`), but `agent_health <= 0` can end an episode early, so
    the cumulative `step_counter` can undershoot `half_point` and leave
    `revisit_late` permanently empty (decline = nan for every cell). Fixed:
    (cell,action) revisit events are now buffered with their step index and the
    early/late split is computed AFTER training from the REALIZED total step
    count (`step_counter`'s final value), never the planned one.
  - Family 1 (CONTESTED -> confirmed live risk, non-blocking mitigation added):
    P2 measures `residue_field.evaluate_trajectory` in isolation, not the
    composite E3 score `f_weight*F + lambda_eff*M + rho_residue*phi` the argmin
    actually routes on (`e3_selector.py` `score_trajectory`, ~line 1386;
    `rho_residue` defaults to 0.5 against `f_weight`/`lambda_ethical` = 1.0,
    `ree_core/utils/config.py:911-925`). So P2 green does not guarantee the
    freeze manipulation reaches behaviour. Not upgraded to a hard gate (no
    calibrated threshold exists for "how much divergence is enough"); instead a
    same-seed A0-vs-A1 executed-action-stream divergence is now RECORDED
    (`family1_action_stream_divergence` in the manifest) so a null C1 result can
    be told apart from "manipulation never reached behaviour" (near-zero
    divergence) vs. a genuine null.
  - Family 4 (CONFIRMED): `FLOOR_REVISIT_DENOMINATOR` (P4, readiness) was set
    literally equal to `THRESH_C4_MIN_REVISITS` (C4, verdict) -- tautological,
    and a run where residue's suppression WORKS (few late revisits, precisely
    because errors are being avoided) drove P4 red and self-routed to
    `substrate_not_ready_requeue`, discarding the very outcome that would
    confirm the claim. Fixed: decoupled. P4 now measures only the EARLY half's
    minimum revisit count (a low floor -- "enough to trust the baseline rate at
    all"); C4 is unchanged and still requires the stricter floor on BOTH halves
    at verdict time.
  - Family 3: not upgraded to a separate P5-vs-existing-routing change beyond
    the P5 precondition below -- once harm_rate is no longer pinned near
    ceiling (2a-i fix), the FAIL->"weakens" routing is a legitimate null
    reading, not an instrumentation artifact. Left as-is.
  - New P5 (harm_rate_not_saturated, general degeneracy guard, not tied to any
    single confound): both arms' training `harm_rate_train` must stay below a
    ceiling. Catches a saturated harm source -- this one or a future one -- by
    its mechanical SIGNATURE (repeat_rate pinned near 1.0 in both arms) rather
    than by name, and routes to `substrate_not_ready_requeue`, never a verdict.
See "C1 arithmetic" below for the corrected satisfiability estimate under the
fixed env config.
"""

import argparse
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import HippocampalConfig, REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.manifest_core import stamp_recording_core
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

EXPERIMENT_TYPE = "v3_exq_983_ext002_residue_error_persistence"
CLAIM_IDS = ["EXT-002"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EVB-1247"

# Action-object-selection gate (`validate_experiments.action_object_selection_lint`)
# does not apply: this script never calls `action_object_decoder` at all. Every
# executed action comes from `agent.select_action(candidates, ticks)` (E3's J(zeta)),
# per `_select_action` below -- the substrate path V3-EXQ-800 established as the only
# one that lets a residue manipulation reach behaviour on this substrate.

# Hold-weighted-E3-readout gate (`validate_experiments.e3_hold_weighted_readout_lint`).
# Fires on `executed.append(int(torch.argmax(action, dim=-1).item()))` inside
# `_probe_candidate_score_range` below -- copied unchanged from V3-EXQ-800's helper of
# the same name, which carries the identical construct and the identical exemption.
# TRIAGED SAFE for the same exact reason: that list's ONLY reader is
# `float(len(set(executed)))` (`executed_action_classes`, fed into P2's readiness
# gate). Set CARDINALITY is exactly invariant under hold-duration replication --
# replicating a held class can neither add a class nor remove one -- so hold
# weighting cannot move this statistic. No magnitude or distribution-shape quantity
# (entropy, variance, histogram mass) is derived from it, and it feeds no C1-C4
# criterion.
#
# This exemption does NOT cover `run_cell`'s (cell,action) repeat-error tracking --
# that accumulation is independently gated on `ticks["e3_tick"]` (fresh E3 selections
# only) precisely so a single commitment held across a hazard-adjacent wall cannot be
# double-counted as many identical revisits; see the `is_fresh_select` gate there.
E3_HOLD_WEIGHTED_READOUT_EXEMPT = (
    "_probe_candidate_score_range's executed-class list is consumed ONLY by "
    "len(set(...)) (executed_action_classes, P2 readiness diagnostic); set "
    "cardinality is exactly invariant under hold-duration replication, and no "
    "magnitude or distribution-shape statistic is derived from it. run_cell's "
    "repeat-error tracking is separately protected by the ticks['e3_tick'] fresh-"
    "selection gate, not by this exemption."
)

ARM_INTACT = "A0_INTACT"
ARM_FROZEN = "A1_RESIDUE_FROZEN"
ARMS = [ARM_INTACT, ARM_FROZEN]

# --- pre-registered thresholds -------------------------------------------
THRESH_C1_DECLINE_GAP = 0.15     # percentage points (decline_A0 - decline_A1)
THRESH_C2_EFFECT_SD = 0.80       # effect in SD of the cross-seed paired delta
THRESH_C3_MIN_SEEDS = 2          # of 3
THRESH_C4_MIN_REVISITS = 20      # per period, worst case across arm/seed

# --- non-degeneracy floors ------------------------------------------------
FLOOR_RESIDUE_WEIGHT_VAR = 1e-6      # P1: field must not be uniform-pinned
                                      # (same floor V3-EXQ-800 validated on this
                                      # identical env/residue config)
FLOOR_CANDIDATE_SCORE_RANGE = 1e-6   # P2: cross-candidate score spread
FLOOR_E1_PREDICTION_ERROR = 1e-4     # P3: world-model error on a known hazard hit

# P4 (readiness) is DELIBERATELY DECOUPLED from C4 (verdict) -- red-team Family 4
# finding. P4 was previously set literally equal to THRESH_C4_MIN_REVISITS, which
# is tautological (P4 can never independently fail once passed) and, worse, a run
# where residue's suppression genuinely WORKS (few late revisits, because errors
# are being avoided) drove P4 red and self-routed to substrate_not_ready_requeue,
# discarding the very outcome that would confirm the claim. P4 now checks only
# that the EARLY half (which residue has not yet had a chance to shape) has
# enough revisits to trust as a baseline rate at all -- a low floor, "a handful".
# C4 stays the stricter, both-halves floor for trusting the decline COMPARISON at
# verdict time, entirely independently of P4.
FLOOR_REVISIT_DENOMINATOR_P4 = 5.0   # P4: early-half-only readiness floor

# P5 (new, red-team Family 1/3 mitigation): both arms' training harm_rate must
# stay below this ceiling. A rate near 1.0 is the mechanical signature of a harm
# source independent of true hazards (contamination-spread self-harm was one
# instance; this guard is not tied to that one confound specifically) and pins
# repeat_rate_early/late near 1.0 in BOTH arms regardless of any true residue
# effect, making C1 unreachable under any true effect. General insurance, kept
# even after the contamination_spread fix.
CEILING_HARM_RATE_P5 = 0.90

# CEM candidate budget. MUST keep num_elite = int(NUM_CANDIDATES * elite_fraction)
# at >= 2 -- see V3-EXQ-800's identical design-time audit below for why (a
# single-elite CEM refit NaNs every candidate rollout via std() of one element).
NUM_CANDIDATES = 32
MIN_REQUIRED_ELITES = 2
CONTROL_PROBE_STEPS = 25  # positive-control probe for P2, run after training

# P3 positive-control probe budget (throwaway agent, isolated from scored arms).
PROBE_HAZARD_WARMUP_EPISODES = 10
PROBE_HAZARD_MAX_STEPS = 300
PROBE_SEED = 999983


# =========================================================================
# precondition specs
# =========================================================================
PRECONDITION_SPECS = [
    PreconditionSpec(
        name="residue_structure_live",
        description=(
            "cross-center variance of active residue RBF weights after training -- "
            "a flat field means no persistent trace was ever written, and any "
            "decline-rate difference from A1 would be a comparison against nothing"
        ),
        control="active centers accumulated over the full training phase",
        threshold=FLOOR_RESIDUE_WEIGHT_VAR,
        direction="lower",
        applies_to=lambda ctx: ctx["id"] == ARM_INTACT,
        applies_note=(
            "A1 is frozen-at-init BY DESIGN (asserting structure there is "
            "unsatisfiable, not a substrate fact)"
        ),
    ),
    PreconditionSpec(
        name="candidate_score_range",
        description=(
            "cross-candidate RANGE of residue trajectory scores on a post-training "
            "positive-control probe -- the SAME statistic the E3 argmin routes on "
            "(V3-EXQ-643 rule)"
        ),
        control="CEM candidates that genuinely differ, probed after training",
        threshold=FLOOR_CANDIDATE_SCORE_RANGE,
        direction="lower",
        applies_to=lambda ctx: True,
        applies_note="both arms read the residue field via the E3 argmin selection path",
    ),
    PreconditionSpec(
        name="e1_prediction_error_floor",
        description=(
            "E3.post_action_update's world-model prediction-error metric on a "
            "genuine hazard collision, measured on a throwaway agent isolated from "
            "both scored arms -- confirms the substrate's error signal is "
            "non-degenerate before a later null repeat-rate result is trusted"
        ),
        control="throwaway agent forced through hazard collisions",
        threshold=FLOOR_E1_PREDICTION_ERROR,
        direction="lower",
        applies_to=lambda ctx: True,
        applies_note=(
            "a substrate-wide readiness check, not an arm-specific property -- "
            "measured once and shared by both arm gates"
        ),
    ),
    PreconditionSpec(
        name="revisit_denominator",
        description=(
            "minimum number of (cell,action) revisit events observed in the "
            "EARLY training half only (DECOUPLED from C4's stricter both-halves "
            "verdict floor -- red-team Family 4 fix: the old floor equalled C4's, "
            "which is tautological and self-defeatingly routed a WORKING residue "
            "effect, few late revisits by construction, to "
            "substrate_not_ready_requeue) -- the early-half repeat rate itself is "
            "undefined/noisy below this"
        ),
        control="revisits of a previously-erred (cell,action) key during the early half",
        threshold=FLOOR_REVISIT_DENOMINATOR_P4,
        direction="lower",
        applies_to=lambda ctx: True,
        applies_note="both arms must accumulate enough EARLY-half revisit events to trust a baseline rate",
    ),
    PreconditionSpec(
        name="harm_rate_not_saturated",
        description=(
            "per-arm training harm_rate must stay below a ceiling -- a rate near "
            "1.0 is the mechanical signature of a harm source independent of "
            "true hazards (e.g. the contamination-spread self-harm trap fixed by "
            "this script's contamination_spread=0.0) and pins "
            "repeat_rate_early/late near 1.0 in BOTH arms regardless of any true "
            "residue effect, making C1 unreachable under any true effect -- a "
            "general degeneracy guard, not tied to one named confound"
        ),
        control="hazard_harm=0.02 + proximity_harm_scale=0.05, contamination_spread=0.0",
        threshold=CEILING_HARM_RATE_P5,
        direction="upper",
        applies_to=lambda ctx: True,
        applies_note="both arms -- a saturated harm source is substrate-level, not arm-specific",
    ),
]


def arm_contexts() -> List[Dict[str, Any]]:
    return [
        {"id": ARM_INTACT, "freeze_residue": False},
        {"id": ARM_FROZEN, "freeze_residue": True},
    ]


# =========================================================================
# substrate helpers (selection path + readiness probes mirror V3-EXQ-800)
# =========================================================================
def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_env(seed: int, full_config: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=full_config["hazard_harm"],
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=full_config["proximity_harm_scale"],
        proximity_benefit_scale=full_config["proximity_harm_scale"] * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
        # RED-TEAM 2a-i FIX: contamination_spread defaults to 0.5, which the
        # env's OWN docstring (causal_grid_world.py:71-77) names as a trap --
        # every cell the agent revisits crosses contamination_threshold after a
        # handful of contacts and starts dealing contaminated_harm=0.4 per
        # contact regardless of hazards, independent of residue. That silent
        # default was the mechanical cause of the pre-fix dry-run's
        # harm_rate_train~1.0. The docstring's own named precedent
        # (V3-EXQ-513) is to pass contamination_spread=0.0 explicitly.
        contamination_spread=0.0,
    )


def _make_agent(env: CausalGridWorldV2, full_config: Dict[str, Any]) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=full_config["self_dim"],
        world_dim=full_config["world_dim"],
        alpha_world=full_config["alpha_world"],
        alpha_self=full_config["alpha_self"],
        reafference_action_dim=0,  # SD-007 off -- isolate the residue mechanism
    )
    config.latent.unified_latent_mode = False  # SD-005 split latents
    return REEAgent(config)


def _active_weight_stats(residue_field) -> Dict[str, float]:
    """Variance / count over ACTIVE residue RBF weights (P1 measurement)."""
    rbf = residue_field.rbf_field
    with torch.no_grad():
        mask = rbf.active_mask
        n_active = int(mask.sum().item())
        if n_active < 2:
            return {"weight_var": 0.0, "n_active": n_active}
        w = rbf.weights[mask].detach()
        return {"weight_var": float(w.var(unbiased=False).item()), "n_active": n_active}


def _candidate_scores(agent, candidates) -> List[Tuple[int, float]]:
    """FINITE residue trajectory scores as (candidate_index, score), lower = better.

    Non-finite scores are DROPPED rather than ranked (same rationale as
    V3-EXQ-800's identical helper: an early-training E2 rollout can diverge and
    yield inf/nan trajectory residue, and ranking that would silently make the
    argmin arbitrary).
    """
    out: List[Tuple[int, float]] = []
    for i, traj in enumerate(candidates):
        world_seq = traj.get_world_state_sequence()
        if world_seq is None:
            continue
        val = float(agent.residue_field.evaluate_trajectory(world_seq).sum().item())
        if math.isfinite(val):
            out.append((i, val))
    return out


def _select_action(
    agent, latent, ticks, n_actions: int, rng: random.Random,
) -> Tuple[torch.Tensor, Optional[float], bool]:
    """Select via the CANONICAL E3 path: e3 ranks the hippocampal candidates.

    Reused verbatim (selection rule and rationale) from V3-EXQ-800's `_select_action`:
    `agent.select_action(candidates, ticks)` routes through E3's J(zeta), which reads
    the residue field via Phi_R and returns the action DIRECTLY -- never the
    `action_object_decoder` round trip, which V3-EXQ-800 measured collapses to a
    single constant class on this substrate (see that script's docstring for the
    measurement). Using the collapsed decoder here would make the residue-freeze
    manipulation an arithmetically forced no-op.

    Returns (action_tensor, cross_candidate_residue_score_range, used_e3).
    """
    with torch.no_grad():
        candidates = agent.hippocampal.propose_trajectories(
            latent.z_world.detach(),
            z_self=latent.z_self.detach(),
            num_candidates=NUM_CANDIDATES,
        )
        if not candidates:
            return (
                _action_to_onehot(rng.randint(0, n_actions - 1), n_actions, agent.device),
                None,
                False,
            )
        scored = _candidate_scores(agent, candidates)
        vals = [v for _i, v in scored]
        score_range = float(max(vals) - min(vals)) if len(vals) >= 2 else None
        action = agent.select_action(candidates, ticks)
        return action, score_range, True


def _probe_candidate_score_range(agent, env, n_actions, rng, n_steps) -> Dict[str, float]:
    """P2 post-training positive control: worst-case (minimum) cross-candidate range.

    Reports the MINIMUM observed range (not the mean), matching V3-EXQ-800's
    convention: P2's `met` is the claim that residue COULD discriminate candidates
    at every probed tick, and a mean can hide ticks where it could not.
    """
    ranges: List[float] = []
    executed: List[int] = []
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            ticks = agent.clock.advance()
            action, span, _used = _select_action(agent, latent, ticks, n_actions, rng)
        if span is not None and math.isfinite(span):
            ranges.append(span)
        executed.append(int(torch.argmax(action, dim=-1).item()))
        _, _harm, done, _info, obs_dict = env.step(action)
        if done:
            break
    return {
        "score_range_min": float(min(ranges)) if ranges else 0.0,
        "executed_action_classes": float(len(set(executed))),
    }


def _positive_control_e1_prediction_error(
    full_config: Dict[str, Any], dry_run: bool,
) -> Dict[str, float]:
    """P3 (mandatory contract item 13): confirm the world-model error signal named
    by the claim ("E1 prediction errors") clears a non-degenerate floor on a KNOWN-
    erroneous action, before trusting a later null on the scored arms as meaningful.

    Uses `agent.e3.post_action_update`'s `prediction_error` metric
    (`actual_z_world - predicted_world`, `e3_selector.py:4243-4247`, where
    `predicted_world` is the E3-selected candidate's own E1/E2-rollout prediction)
    -- the closest available reading of "the world-model's error on the action just
    taken" on this substrate at inference time.

    Runs on a FRESH, THROWAWAY agent+env, entirely separate from A0/A1's residue
    fields -- `post_action_update` itself would otherwise accumulate residue via its
    own commitment-gated write (`e3_selector.py:4253-4256`), which must NEVER touch
    either scored arm's field. Hunts for real hazard collisions within a bounded step
    budget and returns the MINIMUM (worst-case) reading observed across them, or 0.0
    if none occurred (which correctly fails the floor and routes the run to
    substrate_not_ready_requeue rather than a false verdict on the claim).
    """
    reset_all_rng(PROBE_SEED)
    env = _make_env(PROBE_SEED, full_config)
    n_actions = env.action_dim
    agent = _make_agent(env, full_config)
    rng = random.Random(PROBE_SEED)

    n_warmup = min(2, PROBE_HAZARD_WARMUP_EPISODES) if dry_run else PROBE_HAZARD_WARMUP_EPISODES
    max_steps = min(30, PROBE_HAZARD_MAX_STEPS) if dry_run else PROBE_HAZARD_MAX_STEPS

    readings: List[float] = []
    steps_done = 0
    agent.train()
    for _ep in range(n_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        while steps_done < max_steps:
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            ticks = agent.clock.advance()
            action, _span, _used = _select_action(agent, latent, ticks, n_actions, rng)
            _, harm_signal, done, _info, obs_dict = env.step(action)
            steps_done += 1
            if float(harm_signal) < 0:
                with torch.no_grad():
                    next_latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                    e3_metrics = agent.e3.post_action_update(
                        actual_z_world=next_latent.z_world, harm_occurred=True,
                    )
                pe = e3_metrics.get("prediction_error")
                if pe is not None:
                    readings.append(float(pe.detach().item()))
            if done:
                break
        if steps_done >= max_steps:
            break

    return {
        "e1_prediction_error_min": float(min(readings)) if readings else 0.0,
        "e1_prediction_error_mean": (
            float(statistics.fmean(readings)) if readings else 0.0
        ),
        "n_harm_events_observed": len(readings),
        "steps_run": steps_done,
    }


# =========================================================================
# one (seed, arm) cell
# =========================================================================
def run_cell(
    arm_ctx: Dict[str, Any],
    seed: int,
    full_config: Dict[str, Any],
    warmup_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Tuple[Dict[str, Any], REEAgent, List[List[int]]]:
    arm_id = arm_ctx["id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    with arm_cell(
        seed,
        config_slice=full_config,
        script_path=Path(__file__),
        include_driver_script_in_hash=False,
    ) as cell:
        rng = random.Random(seed)
        env = _make_env(seed, full_config)
        n_actions = env.action_dim
        agent = _make_agent(env, full_config)
        optimizer = optim.Adam(list(agent.parameters()), lr=full_config["lr"])

        n_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
        n_steps = min(20, steps_per_episode) if dry_run else steps_per_episode

        planned_total_steps = max(1, n_warmup * n_steps)

        # ---------------- TRAIN (encoder + E1/E2; residue accumulates per-arm) ---
        agent.train()
        harm_train = 0
        steps_train = 0
        hippo_fallbacks = 0

        first_harm_at: Dict[Tuple[int, int, int], int] = {}
        # RED-TEAM 2a-ii FIX: revisit events are buffered with their step index
        # rather than classified into early/late DURING the loop. The old code
        # classified against `half_point` computed from the PLANNED budget
        # (n_warmup * n_steps); but `agent_health <= 0` can end an episode well
        # short of `n_steps`, so the REALIZED `step_counter` can undershoot the
        # planned `half_point` and leave `revisit_late` permanently empty
        # (decline = nan for every row, regardless of any residue effect). The
        # early/late split below is instead computed AFTER training from the
        # REALIZED total step count.
        revisit_events: List[Tuple[int, float]] = []  # (step_index, is_repeat_harm)
        step_counter = 0
        n_fresh_select = 0
        n_latched = 0

        # Family-1 mitigation (recorded, not gated): full per-episode executed-
        # action stream, compared against the other arm's (same seed) after both
        # arms have run -- see `_action_stream_divergence` / run()'s
        # `family1_action_stream_divergence`.
        episode_actions: List[List[int]] = []

        for ep in range(n_warmup):
            _, obs_dict = env.reset()
            agent.reset()
            current_episode_actions: List[int] = []
            for _ in range(n_steps):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                ticks = agent.clock.advance()
                z_world_curr = latent.z_world.detach()
                pre_x, pre_y = int(env.agent_x), int(env.agent_y)

                action, _span, used = _select_action(agent, latent, ticks, n_actions, rng)
                if not used:
                    hippo_fallbacks += 1
                action_idx = int(torch.argmax(action, dim=-1).item())
                current_episode_actions.append(action_idx)

                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_train += 1
                is_harm = float(harm_signal) < 0

                # Repeat-error bookkeeping is gated on a FRESH E3 selection tick
                # (`ticks["e3_tick"]`), never on a HELD tick (`agent.py:7138` returns
                # the committed action unchanged while `not ticks["e3_tick"]`). Without
                # this gate a single commitment held across a hazard-adjacent wall
                # (agent pinned, cell unchanged for the whole hold) would be
                # double-counted as many identical (cell,action) revisits from ONE
                # underlying decision -- exactly the hold-weighted-readout construct
                # defect `validate_experiments.e3_hold_weighted_readout_lint` exists
                # to catch (see the module-level EXEMPT comment for why the OTHER
                # site in this file, `_probe_candidate_score_range`, does not need
                # this same gate: its statistic is a cardinality, invariant under
                # duplication -- this one is a RATE, which is not).
                is_fresh_select = bool(ticks.get("e3_tick", True))
                if is_fresh_select:
                    n_fresh_select += 1
                    key = (pre_x, pre_y, action_idx)
                    if key in first_harm_at:
                        revisit_events.append((step_counter, 1.0 if is_harm else 0.0))
                    elif is_harm:
                        first_harm_at[key] = step_counter
                else:
                    n_latched += 1

                if is_harm:
                    harm_train += 1
                    # A1 freezes the terrain at initialisation: never accumulate.
                    if not arm_ctx["freeze_residue"]:
                        agent.residue_field.accumulate(
                            z_world_curr, harm_magnitude=abs(float(harm_signal)),
                        )

                loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

                step_counter += 1
                if done:
                    break

            episode_actions.append(current_episode_actions)

            if (ep + 1) % 50 == 0 or ep == n_warmup - 1:
                print(
                    f"  [train] seed={seed} arm={arm_id} ep {ep+1}/{n_warmup}"
                    f" harm_events={harm_train}"
                    f" harm_rate={harm_train / max(1, steps_train):.4f}"
                    f" n_revisit_events={len(revisit_events)}"
                    f" step_counter={step_counter}/{planned_total_steps}",
                    flush=True,
                )

        # ---------------- RED-TEAM 2a-ii FIX: split by REALIZED steps, not the
        # planned budget -- see the comment at revisit_events above. -------------
        realized_total_steps = step_counter
        half_point_realized = max(1, realized_total_steps // 2)
        revisit_early = [v for (i, v) in revisit_events if i < half_point_realized]
        revisit_late = [v for (i, v) in revisit_events if i >= half_point_realized]

        # ---------------- readiness measurements (post-training) -----------------
        agent.eval()
        w_stats = _active_weight_stats(agent.residue_field)
        probe = _probe_candidate_score_range(
            agent, env, n_actions, rng,
            min(5, CONTROL_PROBE_STEPS) if dry_run else CONTROL_PROBE_STEPS,
        )

        repeat_rate_early = (
            float(statistics.fmean(revisit_early)) if revisit_early else float("nan")
        )
        repeat_rate_late = (
            float(statistics.fmean(revisit_late)) if revisit_late else float("nan")
        )
        decline = (
            float(repeat_rate_early - repeat_rate_late)
            if (revisit_early and revisit_late) else float("nan")
        )

        row: Dict[str, Any] = {
            "arm_id": arm_id,
            "seed": int(seed),
            "harm_rate_train": float(harm_train / max(1, steps_train)),
            "harm_events_train": int(harm_train),
            "total_steps_train": int(steps_train),
            "n_distinct_erred_keys": int(len(first_harm_at)),
            "n_revisits_early": int(len(revisit_early)),
            "n_revisits_late": int(len(revisit_late)),
            "n_fresh_select": int(n_fresh_select),
            "n_latched": int(n_latched),
            "fresh_select_yield": float(
                n_fresh_select / max(1, n_fresh_select + n_latched)
            ),
            "repeat_rate_early": repeat_rate_early,
            "repeat_rate_late": repeat_rate_late,
            "decline": decline,
            "residue_weight_var": float(w_stats["weight_var"]),
            "residue_active_centers": int(w_stats["n_active"]),
            "candidate_score_range_min": float(probe["score_range_min"]),
            "executed_action_classes": float(probe["executed_action_classes"]),
            "hippo_fallback_steps": int(hippo_fallbacks),
            "residue_total": float(agent.residue_field.total_residue.item()),
            "residue_harm_events": int(agent.residue_field.num_harm_events.item()),
            "residue_coverage": agent.residue_field.get_coverage_telemetry(),
            # 2a-ii diagnostics: how much episode-death truncation actually
            # occurred, distinguishing "nan due to tiny dry-run scale" from "nan
            # due to the underlying defect" at read time.
            "planned_total_steps": int(planned_total_steps),
            "realized_total_steps": int(realized_total_steps),
            "steps_realized_frac": float(realized_total_steps / planned_total_steps),
        }
        cell.stamp(row)

    print(
        f"  [eval] seed={seed} arm={arm_id}"
        f" repeat_early={row['repeat_rate_early']:.4f}"
        f" repeat_late={row['repeat_rate_late']:.4f}"
        f" decline={row['decline']:.4f}"
        f" n_revisits=({row['n_revisits_early']},{row['n_revisits_late']})"
        f" steps_realized_frac={row['steps_realized_frac']:.4f}",
        flush=True,
    )
    # Cell-level sanity check, NOT the scientific verdict (that is computed at the
    # run level in analyse()): did this cell produce any usable harm/revisit data.
    cell_ok = row["harm_events_train"] > 0 and (
        row["n_revisits_early"] + row["n_revisits_late"] > 0
    )
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row, agent, episode_actions


# =========================================================================
# analysis
# =========================================================================
def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else float("nan")


def analyse(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    def _seed_row(arm: str, s: int) -> Optional[Dict[str, Any]]:
        return next((r for r in by_arm[arm] if r["seed"] == s), None)

    per_seed_diff: List[float] = []
    declines: Dict[str, List[float]] = {ARM_INTACT: [], ARM_FROZEN: []}
    for s in seeds:
        r0 = _seed_row(ARM_INTACT, s)
        r1 = _seed_row(ARM_FROZEN, s)
        if r0 and r1 and math.isfinite(r0["decline"]) and math.isfinite(r1["decline"]):
            per_seed_diff.append(r0["decline"] - r1["decline"])
            declines[ARM_INTACT].append(r0["decline"])
            declines[ARM_FROZEN].append(r1["decline"])

    mean_decline_a0 = _mean(declines[ARM_INTACT])
    mean_decline_a1 = _mean(declines[ARM_FROZEN])
    decline_gap = (
        float(mean_decline_a0 - mean_decline_a1)
        if math.isfinite(mean_decline_a0) and math.isfinite(mean_decline_a1)
        else float("nan")
    )

    delta_mean = _mean(per_seed_diff)
    delta_sd = (
        float(statistics.pstdev(per_seed_diff)) if len(per_seed_diff) > 1 else 0.0
    )
    effect_sd = (
        float(delta_mean / delta_sd) if delta_sd > 1e-12
        else (float("inf") if math.isfinite(delta_mean) and delta_mean > 0 else 0.0)
    )
    seeds_consistent = sum(1 for d in per_seed_diff if d > 0)
    min_revisits = min(
        (min(r["n_revisits_early"], r["n_revisits_late"]) for r in rows), default=0,
    )

    c1 = bool(math.isfinite(decline_gap) and decline_gap >= THRESH_C1_DECLINE_GAP)
    c2 = bool(math.isfinite(effect_sd) and effect_sd >= THRESH_C2_EFFECT_SD)
    c3 = bool(seeds_consistent >= THRESH_C3_MIN_SEEDS)
    c4 = bool(min_revisits >= THRESH_C4_MIN_REVISITS)

    return {
        "mean_decline_A0_intact": mean_decline_a0,
        "mean_decline_A1_frozen": mean_decline_a1,
        "decline_gap": decline_gap,
        "per_seed_diff": per_seed_diff,
        "delta_mean": float(delta_mean) if math.isfinite(delta_mean) else float("nan"),
        "delta_sd": delta_sd,
        "effect_sd": effect_sd,
        "seeds_consistent": int(seeds_consistent),
        "min_revisits": int(min_revisits),
        "c1_decline_gap_pass": c1,
        "c2_effect_sd_pass": c2,
        "c3_seed_consistency_pass": c3,
        "c4_data_quality_pass": c4,
        "all_pass": bool(c1 and c2 and c3 and c4),
    }


def _action_stream_divergence(
    episodes_a: List[List[int]], episodes_b: List[List[int]],
) -> Dict[str, float]:
    """Family-1 mitigation (red-team, contested -> confirmed live risk, non-blocking).

    P2 measures `residue_field.evaluate_trajectory` in ISOLATION, not the composite
    E3 score `f_weight*F + lambda_eff*M + rho_residue*phi` the argmin actually
    routes on (`e3_selector.py` `score_trajectory`, ~line 1386; `rho_residue`
    defaults to 0.5 against `f_weight`/`lambda_ethical` = 1.0,
    `ree_core/utils/config.py:911-925`). So a green P2 does not guarantee the
    residue-freeze manipulation (A0 vs A1) actually reaches BEHAVIOUR -- if F/M
    dominate the composite, A0 and A1 could select identical actions throughout
    despite a non-trivial isolated residue-score range.

    Not upgraded to a hard gate -- no calibrated threshold exists for "how much
    divergence is enough" -- but RECORDED so a null C1 result can be told apart
    from "the manipulation never reached behaviour" (near-zero divergence here)
    vs. a genuine null (non-trivial divergence, still no persistence effect).

    Compares same-seed A0/A1 executed-action streams EPISODE-BY-EPISODE (not as
    one flat concatenation) because `agent_health` death can end an episode at a
    different real length per arm once behaviour diverges -- comparing flat
    streams past that point would compare unrelated ticks.
    """
    n_compared = 0
    n_diverged = 0
    for ep_a, ep_b in zip(episodes_a, episodes_b):
        m = min(len(ep_a), len(ep_b))
        for i in range(m):
            n_compared += 1
            if ep_a[i] != ep_b[i]:
                n_diverged += 1
    frac = float(n_diverged / n_compared) if n_compared > 0 else float("nan")
    return {
        "n_episodes_compared": int(min(len(episodes_a), len(episodes_b))),
        "n_steps_compared": int(n_compared),
        "n_steps_diverged": int(n_diverged),
        "diverged_fraction": frac,
    }


def build_gate(rows: List[Dict[str, Any]], e1_pe_min: float) -> Dict[str, Any]:
    """Per-arm precondition gate. A red arm never vacates a green one (module
    default) -- but see run() for why THIS script additionally requires
    `all_green`, not `any_green`, as its own readiness verdict."""
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    gates = []
    for ctx in arm_contexts():
        arm_rows = by_arm[ctx["id"]]
        measured: Dict[str, float] = {
            "candidate_score_range": min(
                r["candidate_score_range_min"] for r in arm_rows
            ),
            "e1_prediction_error_floor": float(e1_pe_min),
            # P4 (readiness): EARLY half only -- see FLOOR_REVISIT_DENOMINATOR_P4
            # and the PreconditionSpec note for why this is decoupled from C4.
            "revisit_denominator": float(
                min(r["n_revisits_early"] for r in arm_rows)
            ),
            # P5: worst-case (highest) observed harm_rate across this arm's seeds.
            "harm_rate_not_saturated": float(
                max(r["harm_rate_train"] for r in arm_rows)
            ),
        }
        if ctx["id"] == ARM_INTACT:
            measured["residue_structure_live"] = min(
                r["residue_weight_var"] for r in arm_rows
            )
        gates.append(
            evaluate_arm_gate(ctx["id"], ctx, PRECONDITION_SPECS, measured=measured)
        )
    return aggregate_arm_gates(gates)


# =========================================================================
# main
# =========================================================================
def run(
    seeds: Tuple[int, ...] = (42, 123, 456),
    warmup_episodes: int = 300,
    steps_per_episode: int = 200,
    dry_run: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    full_config: Dict[str, Any] = {
        "env": "CausalGridWorldV2",
        "size": 6,
        "num_hazards": 4,
        "num_resources": 3,
        "hazard_harm": 0.02,
        "proximity_harm_scale": 0.05,
        "use_proxy_fields": True,
        "self_dim": 32,
        "world_dim": 32,
        "alpha_world": 0.9,   # SD-008: z_world fidelity needed for terrain reads
        "alpha_self": 0.3,
        "lr": 1e-3,
        "num_candidates": NUM_CANDIDATES,
        "warmup_episodes": warmup_episodes,
        "steps_per_episode": steps_per_episode,
        "unified_latent_mode": False,
        "reafference_action_dim": 0,
        "arms": ARMS,
        "selection_rule": "argmin_over_candidate_residue_scores",
        "repeat_error_key": "(pre_action_grid_x, pre_action_grid_y, executed_action_class)",
        "control_probe_steps": CONTROL_PROBE_STEPS,
        "thresholds": {
            "C1_decline_gap": THRESH_C1_DECLINE_GAP,
            "C2_effect_sd": THRESH_C2_EFFECT_SD,
            "C3_min_seeds": THRESH_C3_MIN_SEEDS,
            "C4_min_revisits": THRESH_C4_MIN_REVISITS,
        },
    }

    # Design-time audit 1: refuse a run carrying a structurally unsatisfiable gate
    # BEFORE any compute is spent (the V3-EXQ-785 free catch).
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_contexts())

    # Design-time audit 2 (identical to V3-EXQ-800): a single-elite CEM refit makes
    # ao_std NaN and silently NaNs every candidate rollout, zeroing the cross-
    # candidate score range and turning the residue-freeze manipulation into a
    # guaranteed no-op.
    _elite_fraction = float(HippocampalConfig().elite_fraction)
    _num_elite = max(1, int(NUM_CANDIDATES * _elite_fraction))
    if _num_elite < MIN_REQUIRED_ELITES:
        raise ValueError(
            f"CEM refit would use num_elite={_num_elite} from "
            f"num_candidates={NUM_CANDIDATES} x elite_fraction={_elite_fraction}. "
            f"std() over a single elite is NaN and poisons ao_std, NaN-ing every "
            f"candidate rollout and identically zeroing the cross-candidate residue "
            f"score range -- the A1 freeze manipulation would become a guaranteed "
            f"no-op. Raise NUM_CANDIDATES so num_elite >= {MIN_REQUIRED_ELITES}."
        )
    print(
        f"[V3-EXQ-983] design-audit OK: gate satisfiable, "
        f"num_elite={_num_elite} (>= {MIN_REQUIRED_ELITES})",
        flush=True,
    )

    # P3 positive control -- global, isolated from both scored arms.
    pe_probe = _positive_control_e1_prediction_error(full_config, dry_run)
    print(
        f"[V3-EXQ-983] P3 positive control: e1_prediction_error_min="
        f"{pe_probe['e1_prediction_error_min']:.6g} "
        f"(n_harm_events={pe_probe['n_harm_events_observed']}, "
        f"steps_run={pe_probe['steps_run']})",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    agents: List[REEAgent] = []
    family1_divergence: Dict[str, Dict[str, float]] = {}
    for seed in seeds:
        seed_episode_actions: Dict[str, List[List[int]]] = {}
        for ctx in arm_contexts():
            row, agent, episode_actions = run_cell(
                arm_ctx=ctx,
                seed=seed,
                full_config=full_config,
                warmup_episodes=warmup_episodes,
                steps_per_episode=steps_per_episode,
                dry_run=dry_run,
            )
            rows.append(row)
            agents.append(agent)
            seed_episode_actions[ctx["id"]] = episode_actions

        # Family-1 mitigation: same-seed A0-vs-A1 executed-action divergence,
        # recorded (not gated) -- see _action_stream_divergence.
        div = _action_stream_divergence(
            seed_episode_actions[ARM_INTACT], seed_episode_actions[ARM_FROZEN],
        )
        family1_divergence[str(seed)] = div
        print(
            f"[V3-EXQ-983] Family-1 action-stream divergence seed={seed}: "
            f"diverged_fraction={div['diverged_fraction']:.4f} "
            f"({div['n_steps_diverged']}/{div['n_steps_compared']} steps, "
            f"{div['n_episodes_compared']} episodes compared)",
            flush=True,
        )

    gate = build_gate(rows, pe_probe["e1_prediction_error_min"])
    analysis = analyse(rows, list(seeds))

    # This design's primary statistic is a PAIRED comparison (decline_A0 -
    # decline_A1) and is only interpretable when BOTH arms cleared their gate --
    # unlike V3-EXQ-800's four-arm design where a manipulated arm's result stands
    # alone against a common reference. So `all_green`, not the module default
    # `any_green`, is this script's readiness verdict (see module docstring).
    non_degenerate = bool(gate["all_green"])
    if not non_degenerate:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        interpretation_text = (
            "SUBSTRATE NOT READY -- not a verdict on EXT-002. " + gate["degeneracy_reason"]
        )
    else:
        outcome = "PASS" if analysis["all_pass"] else "FAIL"
        if analysis["all_pass"]:
            label = "ext002_residue_persistent_error_record_supported"
            direction = "supports"
            interpretation_text = (
                "EXT-002 SUPPORTED: with the residue field intact and accumulating, "
                "the repeat-error rate at matched (cell,action) pairs declined by "
                f"{analysis['decline_gap']*100:.1f} percentage points more than under "
                "the residue-frozen control across training. Residue accumulation "
                "measurably suppresses repeating a known-costly action, structural "
                "pressure the hallucination analog (no persistent error cost) lacks."
            )
        else:
            label = "ext002_residue_persistent_error_record_not_supported"
            direction = "weakens"
            interpretation_text = (
                "EXT-002 WEAKENED: with the residue field intact, the repeat-error "
                "rate did not decline materially more than under the residue-frozen "
                f"control (decline_gap {analysis['decline_gap']*100:.1f}pp < "
                f"{THRESH_C1_DECLINE_GAP*100:.0f}pp, or effect/consistency/data-"
                "quality criteria unmet). Residue accumulation is live on this "
                "substrate but is not shown here to create the structural pressure "
                "against repeat errors ARC-005/INV-006/INV-008 assert."
            )

    criteria = [
        {"name": "C1_decline_gap", "load_bearing": True,
         "passed": analysis["c1_decline_gap_pass"]},
        {"name": "C2_effect_sd", "load_bearing": False,
         "passed": analysis["c2_effect_sd_pass"]},
        {"name": "C3_seed_consistency", "load_bearing": False,
         "passed": analysis["c3_seed_consistency_pass"]},
        {"name": "C4_data_quality", "load_bearing": False,
         "passed": analysis["c4_data_quality_pass"]},
    ]

    # All four criteria are a PAIRED property of (A0, A1) together -- see the
    # `non_degenerate` note above -- so their non-degeneracy is the paired-green
    # flag directly, rather than `precondition_gate.arm_criteria_non_degenerate`
    # (which attributes a criterion to a single owning arm; not the right shape
    # here, since neither arm's gate alone makes the paired decline_gap meaningful).
    paired_green = bool(
        ARM_INTACT in gate["green_arms"] and ARM_FROZEN in gate["green_arms"]
    )
    criteria_non_degenerate = {c["name"]: paired_green for c in criteria}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": gate["degeneracy_reason"],
        "per_arm_gate": gate["per_arm_gate"],
        "positive_control_e1_prediction_error": pe_probe,
        "interpretation": {
            "label": label,
            "text": interpretation_text,
            "preconditions": gate["adjudication_preconditions"],
            "preconditions_scope_note": gate.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "criteria": criteria,
        "analysis": analysis,
        "arm_results": rows,
        "per_seed_decline": {
            a: [r["decline"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "registered_thresholds": full_config["thresholds"],
        # Family-1 mitigation (recorded, non-gating) -- see
        # _action_stream_divergence for what this does and does not prove.
        "family1_action_stream_divergence": family1_divergence,
    }

    # stamp AFTER arm_results so substrate_hash HOISTS from the per-cell
    # fingerprints; agent= wires the z_goal_stream liveness block (this driver
    # steps agents; z_goal is not the mechanism under test but the block must be
    # recorded, not silently left "NOT RECORDED" -- see /queue-experiment Step 4).
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0,
        agent=agents,
    )

    print("\n[V3-EXQ-983] Results", flush=True)
    print(
        f"  mean_decline_A0={analysis['mean_decline_A0_intact']:.4f}"
        f"  mean_decline_A1={analysis['mean_decline_A1_frozen']:.4f}"
        f"  decline_gap={analysis['decline_gap']:.4f}",
        flush=True,
    )
    print(f"  non_degenerate={non_degenerate}  outcome={outcome}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    manifest = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        steps_per_episode=args.steps,
        dry_run=args.dry_run,
    )

    out_path = write_flat_manifest(
        manifest,
        None,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=list(args.seeds),
        script_path=Path(__file__),
        started_at=_t_start,
    )
    print(f"\nResult written to: {out_path}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
