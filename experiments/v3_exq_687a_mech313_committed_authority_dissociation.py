"""V3-EXQ-687a -- MECH-313 committed-path authority dissociation (DIAGNOSTIC).

SLEEP DRIVER: K=never (no sleep loop in this probe; waking action selection only)

PURPOSE
=======
Successor to V3-EXQ-687 (Q-045 four-arm tonic-noise ablation, FAIL /
non_contributory / substrate_ceiling). 687's autopsy
(REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-687_2026-06-18.md)
routed a successor that ARMS the 569i-validated GAP-A conversion stack and adds
a MECH-260-operativity precondition. This is that successor -- but re-scoped
from `evidence` to `diagnostic`, because a static line-pin performed at
authoring time (2026-07-29) showed the Q-045 four-arm EVIDENCE design is not
answerable on a committed-action DV. See THE INVARIANCE below.

THIS PROBE DOES NOT ASK "are MECH-313 and MECH-260 jointly load-bearing".
It asks the prior question that 687 could not separate: DOES THE MECH-313
TONIC CHANNEL REACH THE COMMITTED SELECTION AT ALL? It answers by emitting
BOTH readouts the substrate already exposes and measuring the committed
fraction, so the dissociation becomes a MEASURED FACT rather than an
inference from a null.

THE INVARIANCE (line-pinned 2026-07-29; the reason this is a diagnostic)
=======================================================================
MECH-313 is a UNIFORM, state-independent softmax-temperature lift
(`ree_core/agent.py:7261-7272`). Temperature enters E3 at exactly ONE site,
`ree_core/predictors/e3_selector.py:3101`:

    probs = F.softmax(-scores / temperature, dim=0)

whose own comment (:3102-3106) reads: "Pure diagnostic ... a temperature lift
raises THIS but not the committed-class entropy. No behaviour change."

The committed selection branch is `e3_selector.py:3358-3385`:

    if n_eligible == 1:   local = 0                                   # T invisible
    elif committed:
        if use_gap_scaled_commit_temperature: _gap_scaled_commit_pick()  # T live
        else:                                 mod_eligible.argmin()       # T INVISIBLE  <- DEFAULT
    else:                 multinomial(softmax(-mod_eligible / T))         # T live

A uniform temperature is a MONOTONE RESCALING, so it cannot move an argmin.
MECH-313 therefore has selection authority only on UNCOMMITTED ticks. Arming
the GAP-A conversion stack does NOT change this -- that stack feeds
`_modulatory_accum`, and the committed pick over it is the hard argmin above.

Consequence for Q-045 as an evidence design: on a committed-action DV,
ARM_1 == ARM_0 and ARM_3 == ARM_2 ARITHMETICALLY, so the interaction cell --
which is the whole of "jointly load-bearing" -- is unmeasurable. That is the
`/queue-experiment` DV-symmetry disposition (b). Rather than lower a gate or
enable `use_gap_scaled_commit_temperature=True` (which would test "MECH-313
GIVEN MECH-439", manufacturing the authority the claim asserts), this probe
MEASURES the invariance and routes the substrate question onward.

use_gap_scaled_commit_temperature is DELIBERATELY LEFT FALSE (the default).
Flipping it is the substrate question, not this probe's manipulation.

DV-SYMMETRY DECLARATION (per arm; required by /queue-experiment)
===============================================================
  ARM_1 / ARM_3 (MECH-313 tonic lift):
    * `committed_action_entropy` -- the manipulation IS invariant under this DV's
      symmetry group (monotone rescaling; argmax/argmin order-preservation) on
      COMMITTED ticks. This is stated as the HYPOTHESIS UNDER TEST, not assumed:
      a diagnostic measuring an invariance is legitimate exactly where an
      evidence run RESTING on it is not.
    * `precommit_entropy` -- NOT invariant. probs = softmax(-scores/T) is a
      temperature-scaled distribution; T changes its shape by construction.
      This is the positive control that the lever fires at all.
  ARM_2 / ARM_3 (MECH-260 dACC anti-recency):
    * dACC suppression is a PER-CANDIDATE additive penalty
      (`dacc.py:271-282`, count(c in history)/len), NOT a broadcast scalar, so
      it is NOT invariant under argmin. Measurable on the committed DV.

FIVE INSTRUMENT REPAIRS carried from the 603e autopsy
=====================================================
(failure_autopsy_V3-EXQ-603e_hold_weighted_entropy_dv_2026-07-29.md, section 7)
 1. Accumulation is FRESH-SELECTION GATED via `_lib/fresh_select.FreshSelectProbe`
    (sentinel key, NOT the `= None` idiom the autopsy names -- nulling
    `last_score_diagnostics` is NOT inert, `agent.py:9660` reads it on non-E3
    ticks; see the module docstring of `_lib/fresh_select.py`). Emits
    n_fresh_select / n_latched / fresh_select_yield / replication_factor.
 2. BOTH readouts kept DISTINCT: `committed_action_entropy` (selection
    diversity) and `hold_weighted_action_entropy` (occupancy -- a legitimate
    measure, just not a selection-diversity one).
 3. Every cross-arm criterion carries a PRE-REGISTERED MARGIN plus a
    cross-seed-SD gate. 603e's C3 landed `true` on a 0.001235-nat margin.
 4. Behavioural coverage is a CONJUNCT of the diversity reading, not a separate
    criterion: 603e's ARM_2/3 carried the highest action entropy (~ln 2, a
    near-exact 2-cycle) while position_entropy was EXACTLY 0.0 -- the agent sat
    in ONE grid cell. A stuck oscillation must not score as diverse.
 5. The FIFO warmup is expressed in FRESH SELECTIONS, not env steps.
    `dacc.record_action` (`agent.py:8138`) sits AFTER the held-action early
    return (`agent.py:5917`), so dACC memory is in fresh selections while
    687/603e compared it against an env-step warmup -- incommensurable units
    that certified a ring that never filled once.

PRE-REGISTERED GRID
===================
  PRECONDITIONS (regime-conditioned via `_lib/precondition_gate`; a precondition
  that is not meaningful for an arm is SCOPED OUT of it, never failed by it):
    P1 conversion_stack_routes_range  (ALL arms)   modulatory_channel_route_range > 1e-6
    P2 tonic_lever_fires              (noise arms) effective_T - baseline_T > 0.01
    P3 fresh_select_sufficiency       (ALL arms)   n_fresh_select >= 100
    P4 dacc_fifo_full                 (dACC arms)  dacc_history_len_max >= 8

  CRITERIA:
    C1 precommit_entropy_responds_to_tonic   -- does the lever move ANYTHING?
    C2 committed_entropy_responds_to_tonic   -- LOAD-BEARING: does it reach selection?
    C3 dacc_operative_with_stack_armed       -- 687's residual diagnose-first item
    C4 behavioural_coverage_nondegenerate    -- the requirement-4 conjunct

  ROUTES:
    P2 unmet                 -> substrate_not_ready_requeue (lever inert; NOT a verdict)
    C1 false                 -> substrate_not_ready_requeue (precommit channel inert)
    C1 true  AND C2 false    -> tonic_channel_lacks_committed_authority
                                (the predicted finding; escalates MECH-313 to
                                 /implement-substrate)
    C1 true  AND C2 true     -> tonic_channel_propagates_committed
                                (687's null was NOT the tonic route; Q-045
                                 becomes re-testable as an evidence design)

Diagnostic purpose => excluded from governance confidence/conflict scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.fresh_select import (  # noqa: E402
    FreshSelectCounter,
    FreshSelectProbe,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _benefit_and_drive,
    _build_env,
    _sense_with_optional_harm,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_687a_mech313_committed_authority_dissociation"
QUEUE_ID = "V3-EXQ-687a"
SUPERSEDES = "v3_exq_687_q045_mech313_mech260_4arm_tonic_noise_ablation"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["Q-045", "MECH-313", "MECH-260"]

FRESH_SELECT_NAMESPACE = "exq687a"

SEEDS = [42, 43, 44]

# ----- substrate dims (identical to 687) ------------------------------------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# ----- scaffold curriculum budgets (identical to 687) -----------------------------
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
TRAIN_STEPS = 200
P0_NUM_HAZARDS = 1
P1_HOLD_FRACTION = 0.3
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

ESCAPE_THREAT_FLOOR = 0.1
ESCAPE_THREAT_REF = 0.35
ESCAPE_APPROACH_GAIN = 0.1
ESCAPE_BIAS_SCALE = 0.1
ESCAPE_SAFETY_SIGNAL_THRESHOLD = 0.5

HARM_PATHWAY_LR = 1e-3
HARM_PATHWAY_ENCODER_LR = 3e-4
HARM_PATHWAY_WARMUP_STEPS = 250

# ----- the two levers under observation -------------------------------------------
NOISE_FLOOR_ALPHA = 0.5
DACC_SUPPRESSION_WEIGHT = 0.5
DACC_SUPPRESSION_MEMORY = 8

# ----- GAP-A conversion stack (CONSTANT on every arm; 569i ARM_1 config) ----------
# 569i (v3_exq_569i_gapa_conversion_topk_shortlist_falsifier) PASSed / supports on
# ARC-065 with exactly this stack. It is SCAFFOLDING here, not the axis: it is what
# lets the routed cross-candidate range reach the committed selection at all, which
# 687 lacked (it ran the plain SP-CEM main path and inherited the un-converted
# GAP-A monostrategy ceiling by construction).
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3
CANDIDATE_SUMMARY_SOURCE = "e2_world_forward"
# DELIBERATE DEVIATION from the 687 autopsy's prose stack list, which also names
# `use_e3_score_diversity`. 569i -- the run that VALIDATED this stack -- set it
# False, and it is a score-DIVERSITY regulator: switching it on constant across
# arms would inject diversity directly into the DV this probe measures. Kept
# False to match the validated configuration and keep the readout clean.
USE_E3_SCORE_DIVERSITY = False
# DELIBERATELY FALSE -- see THE INVARIANCE in the module docstring. Flipping this
# is the substrate question this probe exists to route, not its manipulation.
USE_GAP_SCALED_COMMIT_TEMPERATURE = False

# ----- P2 measurement --------------------------------------------------------------
EVAL_EPISODES = 30
P2_STEPS_PER_EPISODE = 200
BASELINE_TEMPERATURE = 1.0  # the temperature passed to e3.select() on the main path

# Requirement 5: warmup in FRESH SELECTIONS, not env steps. Filling an 8-deep FIFO
# twice needs 2 * DACC_SUPPRESSION_MEMORY genuine selections; at cadence 10-20 that
# is ~160-320 env steps, which is why the 603e/687 env-step warmups (75 / 30) never
# filled the ring once.
FIFO_WARMUP_FRESH_SELECTS = 2 * DACC_SUPPRESSION_MEMORY

# ----- pre-registered thresholds ---------------------------------------------------
ENTROPY_MARGIN_NATS = 0.05      # requirement 3: absolute floor on any cross-arm delta
SD_MARGIN_MULTIPLIER = 2.0      # ... AND the delta must clear 2x its cross-seed SD
POSITION_ENTROPY_FLOOR = 0.20   # requirement 4: coverage conjunct (603e ARM_2/3 = 0.0)
TONIC_TEMP_LIFT_FLOOR = 0.01    # P2: the tonic lever demonstrably fires
FRESH_SELECT_FLOOR = 100        # P3: minimum genuine selections for a cell to count
COMMITTED_FLOOR = 50            # P5: minimum COMMITTED selections -- C2's denominator
ROUTE_RANGE_FLOOR = MODULATORY_ROUTE_MIN_RANGE_FLOOR  # P1: SAME statistic the argmin eats
MIN_FRACTION = 2.0 / 3.0

TOTAL_RUNS = 0

# z_goal liveness accumulator. Agents are built per cell inside _run_seed_arm, so
# the accumulator reads counters at observe() time rather than retaining every
# arm x seed agent (nets + optimiser state) until the last cell finishes.
_ZG = ZGoalStreamAccumulator()

ARMS: List[Dict[str, Any]] = [
    {"label": "ARM_0_both_off", "use_noise_floor": False, "use_dacc": False},
    {"label": "ARM_1_mech313_only", "use_noise_floor": True, "use_dacc": False},
    {"label": "ARM_2_mech260_only", "use_noise_floor": False, "use_dacc": True},
    {"label": "ARM_3_both_on", "use_noise_floor": True, "use_dacc": True},
]
DACC_ARM_LABELS = {"ARM_2_mech260_only", "ARM_3_both_on"}
NOISE_ARM_LABELS = {"ARM_1_mech313_only", "ARM_3_both_on"}


# ---------------------------------------------------------------------------
# Regime-conditioned precondition specs
# ---------------------------------------------------------------------------
# `applies_to` is load-bearing: P2 is not MEANINGFUL for a noise-floor-OFF arm
# (there is no tonic lift to fire) and P4 is not meaningful for a dACC-OFF arm
# (there is no FIFO). Asserting either whole-run would make those arms
# structurally un-passable and would vacate the arms that ARE clean -- the
# V3-EXQ-785 defect this module exists to prevent.
PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="conversion_stack_routes_range",
        description=(
            "GAP-A conversion stack live: the routed cross-candidate range "
            "(modulatory_channel_route_range) clears its floor. This is the SAME "
            "statistic the within-shortlist argmin consumes -- not a magnitude "
            "proxy for it (the V3-EXQ-643 same-statistic rule)."
        ),
        threshold=ROUTE_RANGE_FLOOR,
        control="frozen-policy P2 ticks with >=2 candidates and the stack armed",
        kind="readiness",
    ),
    PreconditionSpec(
        name="tonic_lever_fires",
        description=(
            "MECH-313 positive control: the noise floor lifts the effective softmax "
            "temperature above baseline. Measured by calling "
            "noise_floor.compute_effective_temperature() directly, so it does not "
            "depend on diagnostic plumbing. Below floor => the lever never fired and "
            "NO reading about its authority is possible."
        ),
        threshold=TONIC_TEMP_LIFT_FLOOR,
        control="direct compute_effective_temperature(baseline_temperature=1.0)",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["use_noise_floor"]),
    ),
    PreconditionSpec(
        name="fresh_select_sufficiency",
        description=(
            "Enough GENUINE selections for an entropy to have an n. The env-step "
            "count is not the denominator -- E3 cadence 10 (driven to [5,20] by "
            "MECH-093) means ~90-95% of env steps replicate a prior commitment."
        ),
        threshold=float(FRESH_SELECT_FLOOR),
        control="fresh-selection counter over the frozen-policy P2 phase",
        kind="readiness",
    ),
    PreconditionSpec(
        name="committed_sufficiency",
        description=(
            "NON-VACUITY GUARD ON THE LOAD-BEARING CRITERION. C2 compares "
            "committed_action_entropy across arms, so its DENOMINATOR is the count "
            "of COMMITTED fresh selections -- not env steps and not fresh selections. "
            "An agent that never commits yields a C2 computed over an empty "
            "histogram, which would read as 'tonic has no committed authority' for "
            "the wrong reason (nothing was committed) rather than the right one "
            "(the argmin is temperature-invariant). Below floor => "
            "substrate_not_ready_requeue, NOT an authority verdict."
        ),
        threshold=float(COMMITTED_FLOOR),
        control="count of fresh selections with last_score_diagnostics['committed']",
        kind="readiness",
    ),
    PreconditionSpec(
        name="dacc_fifo_full",
        description=(
            "MECH-260 non-vacuity: the dACC anti-recency ring actually filled, "
            "measured in FRESH SELECTIONS (requirement 5). An unfilled ring cannot "
            "produce suppression, so a zero-suppression reading would be "
            "uninterpretable."
        ),
        threshold=float(DACC_SUPPRESSION_MEMORY),
        control="dacc_history_len_max over the measured phase",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["use_dacc"]),
    ),
]


def _arm_context(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm_id": arm["label"],
        "use_noise_floor": bool(arm["use_noise_floor"]),
        "use_dacc": bool(arm["use_dacc"]),
    }


# ---------------------------------------------------------------------------
# Config builders (survival stack + conversion stack constant; only levers vary)
# ---------------------------------------------------------------------------

def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, steps = 2, 2, 5, 5, 5, 30
    else:
        stage0, stage0b, p0, hazard, p1, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=1,  # scheduler P2 unused; this script owns P2
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env, arm: Dict[str, Any]) -> REEConfig:
    """Survival stack AND the 569i GAP-A conversion stack held CONSTANT; ONLY
    use_noise_floor (MECH-313) and the dACC knobs (MECH-260) vary per arm."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        # Survival stack (constant across arms).
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_escape_affordance_bridge=True,
        use_escape_relief_credit=True,
        use_escape_safety_credit=True,
        escape_threat_floor=ESCAPE_THREAT_FLOOR,
        escape_threat_ref=ESCAPE_THREAT_REF,
        escape_approach_gain=ESCAPE_APPROACH_GAIN,
        escape_bias_scale=ESCAPE_BIAS_SCALE,
        escape_use_trained_safety_signal=True,
        escape_safety_signal_threshold=ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
        # ===== 569i GAP-A CONVERSION STACK -- CONSTANT ON EVERY ARM =====
        # This is what 687 lacked. It gives the routed cross-candidate range a
        # path to the committed selection (top-k shortlist + within-set argmin).
        candidate_summary_source=CANDIDATE_SUMMARY_SOURCE,
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        use_e3_score_diversity=USE_E3_SCORE_DIVERSITY,
        use_gap_scaled_commit_temperature=USE_GAP_SCALED_COMMIT_TEMPERATURE,
        # ===== MECH-313 lever (per arm) =====
        use_noise_floor=bool(arm["use_noise_floor"]),
        noise_floor_alpha=(NOISE_FLOOR_ALPHA if arm["use_noise_floor"] else 0.1),
        # ===== MECH-260 lever (per arm) =====
        use_dacc=bool(arm["use_dacc"]),
        dacc_weight=(1.0 if arm["use_dacc"] else 0.0),
        dacc_suppression_weight=(DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0),
        dacc_suppression_memory=DACC_SUPPRESSION_MEMORY,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _config_slice(arm: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """Content-addressed config slice for the per-cell arm fingerprint."""
    return {
        "arm": arm["label"],
        "use_noise_floor": bool(arm["use_noise_floor"]),
        "noise_floor_alpha": (NOISE_FLOOR_ALPHA if arm["use_noise_floor"] else 0.1),
        "use_dacc": bool(arm["use_dacc"]),
        "dacc_suppression_weight": (
            DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0),
        "dacc_suppression_memory": DACC_SUPPRESSION_MEMORY,
        # GAP-A conversion stack identity (constant, but part of the substrate id):
        "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
        "modulatory_stack": [
            MODULATORY_AUTHORITY_GAIN, MODULATORY_AUTHORITY_NORMALIZE_BASIS,
            MODULATORY_SHORTLIST_MODE, MODULATORY_SHORTLIST_K,
            MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        ],
        "use_e3_score_diversity": USE_E3_SCORE_DIVERSITY,
        "use_gap_scaled_commit_temperature": USE_GAP_SCALED_COMMIT_TEMPERATURE,
        # Constant survival-stack identity:
        "use_gated_policy": True,
        "use_pag_freeze_gate": True,
        "use_instrumental_avoidance": True,
        "use_escape_affordance_bridge": True,
        "scaffold_train_harm_pathway": True,
        "harm_pathway_lr": HARM_PATHWAY_LR,
        "harm_pathway_encoder_lr": HARM_PATHWAY_ENCODER_LR,
        "harm_pathway_warmup_steps": HARM_PATHWAY_WARMUP_STEPS,
        "feed_harm_stream": True,
        "world_dim": WORLD_DIM,
        "drive_weight": DRIVE_WEIGHT,
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
                    P1_BUDGET, TRAIN_STEPS],
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR],
        "p2": [EVAL_EPISODES, P2_STEPS_PER_EPISODE, FIFO_WARMUP_FRESH_SELECTS,
               BASELINE_TEMPERATURE],
        "dry_run": bool(dry_run),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _probs_entropy(probs: Optional[torch.Tensor]) -> Optional[float]:
    """Shannon entropy (nats) of the pre-commit softmax distribution."""
    if probs is None:
        return None
    p = probs.detach().reshape(-1).clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


def _z_goal_norm(agent: REEAgent) -> float:
    gs = getattr(agent, "goal_state", None)
    z = getattr(gs, "z_goal", None) if gs is not None else None
    if z is None:
        return 0.0
    try:
        return float(z.norm(dim=-1).mean().item())
    except Exception:
        return 0.0


def _dacc_diag(agent: REEAgent) -> Dict[str, Any]:
    """MECH-260 operativity readout.

    INSTRUMENT REPAIR (found in the 687a code-review pass, 2026-07-29). V3-EXQ-687
    read the suppression bundle as `dacc._last_bundle` -- an attribute that DOES
    NOT EXIST on the dACC module. `getattr` therefore returned None on every tick
    and `dacc_max_suppression` was pinned to 0.0 BY CONSTRUCTION, independently of
    substrate behaviour. The bundle lives on the AGENT (`agent._dacc_last_bundle`,
    set at `agent.py:6148`; canonical read at `agent.py:10340`) with a per-candidate
    [K] `suppression` tensor (`ree_core/cingulate/dacc.py:430`).

    Consequence: 687's PRE_MECH260 failure (`dacc_max_suppression=0.0` with a full
    FIFO) was an INSTRUMENT ARTIFACT, not a measurement, and the 687 autopsy's
    section 6.2 record/score-decoupling root cause rests on it. The FIFO-depth leg
    (`_action_history`) used the correct path and IS real.
    """
    d = getattr(agent, "dacc", None)
    if d is None:
        return {"dacc_forward_calls": 0, "dacc_history_len": 0,
                "dacc_max_suppression": 0.0}
    hist = getattr(d, "_action_history", None)
    max_sup = 0.0
    bundle = getattr(agent, "_dacc_last_bundle", None)
    if isinstance(bundle, dict):
        sup = bundle.get("suppression")
        if isinstance(sup, torch.Tensor) and sup.numel() > 0:
            max_sup = float(sup.max().item())
    return {
        "dacc_forward_calls": int(getattr(d, "_n_forward_calls", 0) or 0),
        "dacc_history_len": int(len(hist)) if hist is not None else 0,
        "dacc_max_suppression": max_sup,
    }


def _tonic_temp_lift(agent: REEAgent) -> float:
    """Positive control for MECH-313: the tonic lift over baseline temperature.

    Calls the regulator directly rather than reading a diagnostic key, so the
    measurement cannot be confounded by diagnostic plumbing being disabled.
    Returns 0.0 when the regulator is absent (a noise-floor-OFF arm).
    """
    nf = getattr(agent, "noise_floor", None)
    if nf is None:
        return 0.0
    try:
        eff = float(nf.compute_effective_temperature(
            baseline_temperature=BASELINE_TEMPERATURE, simulation_mode=False))
    except Exception:
        return 0.0
    return eff - BASELINE_TEMPERATURE


def _frac(flags: List[bool]) -> float:
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def _mean(vals: List[float]) -> float:
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else 0.0


def _sd(vals: List[float]) -> float:
    vals = [v for v in vals if v is not None]
    return float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Frozen-policy P2 dissociation measurement
# ---------------------------------------------------------------------------

def _run_p2_dissociation(agent: REEAgent,
                         scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                         arm: Dict[str, Any], seed: int, device: torch.device,
                         episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    """Measure the tonic-authority dissociation on the frozen trained policy.

    Emits FOUR distinct readouts (requirements 1, 2 and 4):
      * precommit_entropy_mean       -- softmax(-scores/T); temperature-SENSITIVE
      * committed_action_entropy     -- fresh-select gated, COMMITTED ticks only
      * fresh_action_entropy         -- fresh-select gated, all fresh ticks
      * hold_weighted_action_entropy -- the 603e/687 readout, kept for comparison
                                        and explicitly labelled an OCCUPANCY measure
    """
    env = _build_env(scaffold_cfg, "p2")
    world_dim = agent.config.latent.world_dim

    probe = FreshSelectProbe(FRESH_SELECT_NAMESPACE)
    counter = FreshSelectCounter()

    committed_counts: Counter = Counter()
    fresh_counts: Counter = Counter()
    hold_counts: Counter = Counter()
    position_counts: Counter = Counter()

    precommit_entropies: List[float] = []
    route_ranges: List[float] = []
    authority_active = 0
    shortlist_active = 0
    n_committed = 0
    n_measured_fresh = 0

    total_steps = 0
    z_goal_norm_peak = 0.0
    max_dacc_forward = 0
    max_dacc_history = 0
    max_dacc_suppression = 0.0
    tonic_lift = _tonic_temp_lift(agent)

    with torch.no_grad():
        for ep in range(episodes):
            _, obs_dict = env.reset()
            agent.reset()
            for _step in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device,
                    scaffold_cfg.scaffold_feed_harm_stream,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                # Requirement 1: sentinel-key freshness. The `= None` idiom is NOT
                # used -- see _lib/fresh_select.py, agent.py:9660 reads the dict on
                # non-E3 ticks and nulling it is not inert.
                with probe.watch(agent) as fresh:
                    action = agent.select_action(candidates, ticks)
                counter.record(bool(fresh))

                action_idx = int(action.argmax(dim=-1).item())
                pos = (int(env.agent_x), int(env.agent_y))

                # The hold-weighted readout: one count per ENV STEP. Retained
                # deliberately (requirement 2) as an OCCUPANCY measure so the
                # distortion is auditable, NOT as a selection-diversity readout.
                hold_counts[action_idx] += 1
                position_counts[pos] += 1

                if bool(fresh):
                    diag = probe.diagnostics(agent, True)
                    # Warmup measured in FRESH SELECTIONS (requirement 5).
                    if counter.n_fresh_select > FIFO_WARMUP_FRESH_SELECTS:
                        n_measured_fresh += 1
                        fresh_counts[action_idx] += 1

                        pe = _probs_entropy(
                            getattr(agent.e3, "last_precommit_probs", None))
                        if pe is not None:
                            precommit_entropies.append(pe)

                        rr = diag.get("modulatory_channel_route_range")
                        if rr is not None:
                            route_ranges.append(float(rr))
                        if float(diag.get("modulatory_authority_active", 0.0) or 0.0) > 0.0:
                            authority_active += 1
                        if bool(diag.get("modulatory_shortlist_active", False)):
                            shortlist_active += 1

                        if bool(diag.get("committed", False)):
                            n_committed += 1
                            committed_counts[action_idx] += 1

                if arm["use_dacc"]:
                    d = _dacc_diag(agent)
                    max_dacc_forward = max(max_dacc_forward, d["dacc_forward_calls"])
                    max_dacc_history = max(max_dacc_history, d["dacc_history_len"])
                    max_dacc_suppression = max(
                        max_dacc_suppression, d["dacc_max_suppression"])

                zg = _z_goal_norm(agent)
                if zg > z_goal_norm_peak:
                    z_goal_norm_peak = zg

                _, _harm, done, _, obs_dict = env.step(action_idx)
                total_steps += 1

                benefit, drive = _benefit_and_drive(obs_dict["body_state"].to(device))
                agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
                zg_post = _z_goal_norm(agent)
                if zg_post > z_goal_norm_peak:
                    z_goal_norm_peak = zg_post

                if done:
                    break
            # agent.reset() clears the commitment latch, so a hold cannot span
            # episodes -- flush at every boundary.
            counter.flush()

            if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
                # Deliberately NOT the 'ep N/M' token -- that is reserved for the
                # training-loop denominator the runner reads.
                print(f"  [P2 eval] arm={arm['label']} seed={seed} eval_ep {ep + 1}"
                      f" of {episodes} fresh={counter.n_fresh_select}"
                      f" committed={n_committed}", flush=True)
    counter.flush()

    row: Dict[str, Any] = {
        "arm": arm["label"],
        "seed": seed,
        "reached_p2": True,
        # --- the dissociation readouts (requirement 2: kept DISTINCT) ---
        "precommit_entropy_mean": round(_mean(precommit_entropies), 6),
        "precommit_entropy_n": int(len(precommit_entropies)),
        "committed_action_entropy": round(_entropy(committed_counts), 6),
        "fresh_action_entropy": round(_entropy(fresh_counts), 6),
        "hold_weighted_action_entropy": round(_entropy(hold_counts), 6),
        "hold_weighted_is_occupancy_not_diversity": True,
        "position_entropy": round(_entropy(position_counts), 6),
        # --- unique-action counts at each weighting ---
        "unique_actions_committed": len(committed_counts),
        "unique_actions_fresh": len(fresh_counts),
        "unique_actions_hold_weighted": len(hold_counts),
        # --- commitment structure ---
        "n_committed": int(n_committed),
        "n_uncommitted": int(n_measured_fresh - n_committed),
        "committed_frac": round(
            (n_committed / n_measured_fresh) if n_measured_fresh else 0.0, 6),
        "n_measured_fresh": int(n_measured_fresh),
        "fifo_warmup_fresh_selects": int(FIFO_WARMUP_FRESH_SELECTS),
        # --- conversion stack liveness (P1 statistic) ---
        "modulatory_route_range_mean": round(_mean(route_ranges), 9),
        "modulatory_route_range_n": int(len(route_ranges)),
        "modulatory_authority_active_frac": round(
            (authority_active / n_measured_fresh) if n_measured_fresh else 0.0, 6),
        "modulatory_shortlist_active_frac": round(
            (shortlist_active / n_measured_fresh) if n_measured_fresh else 0.0, 6),
        # --- MECH-313 positive control (P2 statistic) ---
        "tonic_temp_lift": round(tonic_lift, 6),
        "baseline_temperature": BASELINE_TEMPERATURE,
        # --- substrate engagement ---
        "z_goal_norm_peak": round(z_goal_norm_peak, 6),
        "total_steps": int(total_steps),
        "use_noise_floor": bool(arm["use_noise_floor"]),
        "use_dacc": bool(arm["use_dacc"]),
    }
    row.update(counter.as_dict(total_steps))
    if arm["use_dacc"]:
        row.update({
            "dacc_forward_calls_max": int(max_dacc_forward),
            "dacc_history_len_max": int(max_dacc_history),
            "dacc_max_suppression": round(max_dacc_suppression, 6),
            "mech260_operative": bool(
                max_dacc_forward > 0 and max_dacc_history > 0
                and max_dacc_suppression > 0.0),
        })
    return row


def _empty_row(arm: Dict[str, Any], seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "arm": arm["label"], "seed": seed, "reached_p2": False,
        "aborted_at": stage, "abort_reason": reason,
        "precommit_entropy_mean": 0.0, "precommit_entropy_n": 0,
        "committed_action_entropy": 0.0, "fresh_action_entropy": 0.0,
        "hold_weighted_action_entropy": 0.0, "position_entropy": 0.0,
        "unique_actions_committed": 0, "unique_actions_fresh": 0,
        "unique_actions_hold_weighted": 0,
        "n_committed": 0, "committed_frac": 0.0, "n_measured_fresh": 0,
        "n_fresh_select": 0, "n_latched": 0, "fresh_select_yield": 0.0,
        "modulatory_route_range_mean": 0.0, "modulatory_route_range_n": 0,
        "modulatory_authority_active_frac": 0.0,
        "modulatory_shortlist_active_frac": 0.0,
        "tonic_temp_lift": 0.0, "z_goal_norm_peak": 0.0, "total_steps": 0,
        "use_noise_floor": bool(arm["use_noise_floor"]),
        "use_dacc": bool(arm["use_dacc"]),
    }


def _run_seed_arm(arm: Dict[str, Any], seed: int, dry_run: bool,
                  total_eps: int) -> Dict[str, Any]:
    with arm_cell(seed, config_slice=_config_slice(arm, dry_run),
                  script_path=Path(__file__)) as cell:
        scaffold_cfg = _make_scaffold_cfg(dry_run)
        device = torch.device("cpu")
        probe_env = _build_env(scaffold_cfg, "p2")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env, arm)).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
        print(f"Seed {seed} Condition {arm['label']}", flush=True)

        done = 0
        s0 = scheduler.run_stage0_nursery(agent, device)
        done += s0.n_episodes
        print(f"  [train] stage0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0",
                  flush=True)
            row = _empty_row(arm, seed, "stage0", s0.abort_reason)
            cell.stamp(row)
            return row

        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0b",
                  flush=True)
            row = _empty_row(arm, seed, "stage0b", s0b.abort_reason)
            cell.stamp(row)
            return row

        p0 = scheduler.run_p0(agent, device)
        done += p0.n_episodes
        print(f"  [train] p0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f}", flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=p0",
                  flush=True)
            row = _empty_row(arm, seed, "p0", p0.abort_reason)
            cell.stamp(row)
            return row

        hz = scheduler.run_hazard_avoidance(agent, device)
        done += hz.n_episodes
        print(f"  [train] hazard {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={hz.mean_episode_length:.1f}"
              f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}",
              flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=hazard",
                  flush=True)
            row = _empty_row(arm, seed, "hazard", hz.abort_reason)
            cell.stamp(row)
            return row

        p1 = scheduler.run_p1(agent, device)
        done += p1.n_episodes
        print(f"  [train] p1 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={p1.median_last_window_episode_length:.1f}"
              f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}",
              flush=True)

        p2_eps = 1 if dry_run else EVAL_EPISODES
        p2_steps = 30 if dry_run else P2_STEPS_PER_EPISODE
        row = _run_p2_dissociation(agent, scaffold_cfg, arm, seed, device,
                                   episodes=p2_eps, steps_per_episode=p2_steps)
        _ZG.observe(agent)  # AFTER stepping -- reads the counters at call time
        print(f"verdict: PASS seed={seed} arm={arm['label']}"
              f" precommit_H={row['precommit_entropy_mean']:.4f}"
              f" committed_H={row['committed_action_entropy']:.4f}"
              f" pos_H={row['position_entropy']:.4f}"
              f" fresh={row['n_fresh_select']}"
              f" tonic_lift={row['tonic_temp_lift']:.4f}"
              f" dacc_supp={row.get('dacc_max_suppression', 'n/a')}", flush=True)
        cell.stamp(row)
        return row


# ---------------------------------------------------------------------------
# Preconditions (regime-conditioned) + dissociation grid
# ---------------------------------------------------------------------------

def _measured_for(spec_name: str, rows: List[Dict[str, Any]]) -> float:
    """Worst-cell (not mean) reduction, so `measured` recomputes exactly and the
    quantifier in `met` matches the reported statistic."""
    if not rows:
        return 0.0
    if spec_name == "conversion_stack_routes_range":
        return min(float(r.get("modulatory_route_range_mean", 0.0)) for r in rows)
    if spec_name == "tonic_lever_fires":
        return min(float(r.get("tonic_temp_lift", 0.0)) for r in rows)
    if spec_name == "fresh_select_sufficiency":
        return float(min(int(r.get("n_fresh_select", 0)) for r in rows))
    if spec_name == "committed_sufficiency":
        return float(min(int(r.get("n_committed", 0)) for r in rows))
    if spec_name == "dacc_fifo_full":
        return float(min(int(r.get("dacc_history_len_max", 0)) for r in rows))
    return 0.0


def _evaluate_arm_gates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm_gates = []
    for arm in ARMS:
        arm_rows = [r for r in rows if r.get("arm") == arm["label"]
                    and r.get("reached_p2")]
        ctx = _arm_context(arm)
        measured = {s.name: _measured_for(s.name, arm_rows)
                    for s in PRECONDITION_SPECS}
        arm_gates.append(evaluate_arm_gate(
            arm_id=arm["label"], specs=PRECONDITION_SPECS,
            arm_ctx=ctx, measured=measured))
    return aggregate_arm_gates(arm_gates)


def _arm_stat(rows: List[Dict[str, Any]], label: str, key: str) -> List[float]:
    return [float(r.get(key, 0.0)) for r in rows
            if r.get("arm") == label and r.get("reached_p2")]


def _delta_with_margin(rows: List[Dict[str, Any]], on_label: str,
                       off_label: str, key: str) -> Dict[str, Any]:
    """Requirement 3: a cross-arm delta must clear BOTH a pre-registered absolute
    floor AND `SD_MARGIN_MULTIPLIER` x its own cross-seed SD. 603e's C3 landed
    `true` on 0.001235 nats under a bare `>`."""
    on_vals = _arm_stat(rows, on_label, key)
    off_vals = _arm_stat(rows, off_label, key)
    on_m, off_m = _mean(on_vals), _mean(off_vals)
    delta = on_m - off_m
    pooled_sd = max(_sd(on_vals), _sd(off_vals))
    sd_gate = SD_MARGIN_MULTIPLIER * pooled_sd
    passed = bool(delta >= ENTROPY_MARGIN_NATS and delta >= sd_gate)
    return {
        "on_arm": on_label, "off_arm": off_label, "metric": key,
        "on_mean": round(on_m, 6), "off_mean": round(off_m, 6),
        "delta": round(delta, 6), "pooled_sd": round(pooled_sd, 6),
        "absolute_margin": ENTROPY_MARGIN_NATS,
        "sd_gate": round(sd_gate, 6),
        "n_on": len(on_vals), "n_off": len(off_vals),
        "passed": passed,
    }


def _evaluate_grid(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # C1 / C2: the dissociation pair, both measured ARM_1 vs ARM_0.
    c1 = _delta_with_margin(rows, "ARM_1_mech313_only", "ARM_0_both_off",
                            "precommit_entropy_mean")
    c2 = _delta_with_margin(rows, "ARM_1_mech313_only", "ARM_0_both_off",
                            "committed_action_entropy")
    # Same pair on the both-on/260-only contrast, as a replication of the same
    # invariance question in the presence of dACC.
    c1b = _delta_with_margin(rows, "ARM_3_both_on", "ARM_2_mech260_only",
                             "precommit_entropy_mean")
    c2b = _delta_with_margin(rows, "ARM_3_both_on", "ARM_2_mech260_only",
                             "committed_action_entropy")

    # C3: 687's residual diagnose-first item -- is dACC operative with the
    # conversion stack armed and the FIFO warmed in FRESH selections?
    dacc_flags = [bool(r.get("mech260_operative", False)) for r in rows
                  if r.get("arm") in DACC_ARM_LABELS and r.get("reached_p2")]
    c3_frac = _frac(dacc_flags)
    c3 = bool(c3_frac >= MIN_FRACTION)

    # C4: requirement-4 coverage conjunct -- an in-place oscillation must not
    # read as diverse.
    pos_flags = [float(r.get("position_entropy", 0.0)) > POSITION_ENTROPY_FLOOR
                 for r in rows if r.get("reached_p2")]
    c4_frac = _frac(pos_flags)
    c4 = bool(c4_frac >= MIN_FRACTION)

    return {
        "c1_precommit_responds_to_tonic": c1,
        "c1b_precommit_responds_to_tonic_with_dacc": c1b,
        "c2_committed_responds_to_tonic": c2,
        "c2b_committed_responds_to_tonic_with_dacc": c2b,
        "c3_dacc_operative_with_stack_armed": {
            "passed": c3, "fraction": round(c3_frac, 6),
            "min_fraction": round(MIN_FRACTION, 6), "per_cell": dacc_flags,
        },
        "c4_behavioural_coverage_nondegenerate": {
            "passed": c4, "fraction": round(c4_frac, 6),
            "floor": POSITION_ENTROPY_FLOOR,
            "min_fraction": round(MIN_FRACTION, 6),
        },
    }


def _interpret(gate: Dict[str, Any], grid: Dict[str, Any],
               rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    c1 = bool(grid["c1_precommit_responds_to_tonic"]["passed"])
    c2 = bool(grid["c2_committed_responds_to_tonic"]["passed"])
    c3 = bool(grid["c3_dacc_operative_with_stack_armed"]["passed"])

    noise_arms_green = [a for a in gate.get("green", [])
                        if a in NOISE_ARM_LABELS]

    if not gate.get("non_degenerate", False) or not noise_arms_green:
        label = "substrate_not_ready_requeue"
        note = ("No noise-floor arm cleared its precondition gate, so no reading "
                "about tonic authority is possible. NOT a substrate verdict.")
    elif not c1:
        label = "substrate_not_ready_requeue"
        note = ("The tonic lift did not move the PRE-COMMIT distribution, which it "
                "must by construction (probs = softmax(-scores/T)). The lever did "
                "not effectively fire; no authority reading is possible.")
    elif c1 and not c2:
        label = "tonic_channel_lacks_committed_authority"
        note = ("PREDICTED FINDING. The tonic lift moves the pre-commit softmax but "
                "NOT the committed-class entropy -- the monotone-rescaling "
                "invariance of the committed argmin (e3_selector.py:3380) measured "
                "rather than inferred. Q-045 is NOT answerable as a four-arm "
                "committed-action evidence design while this holds. Routes MECH-313 "
                "to /implement-substrate (committed-path authority), or to a claim "
                "amendment scoping MECH-313 to uncommitted exploration.")
    else:
        label = "tonic_channel_propagates_committed"
        note = ("The tonic lift reaches the committed selection. 687's null was NOT "
                "the tonic route, and Q-045 becomes re-testable as a four-arm "
                "evidence design on this substrate.")

    preconditions = gate.get("adjudication_preconditions",
                             gate.get("preconditions", []))
    return {
        "label": label,
        "note": note,
        "preconditions": preconditions,
        "criteria_non_degenerate": arm_criteria_non_degenerate(
            criteria_by_arm={
                "ARM_0_both_off": ["C1", "C2", "C4"],
                "ARM_1_mech313_only": ["C1", "C2", "C4"],
                "ARM_2_mech260_only": ["C3", "C4"],
                "ARM_3_both_on": ["C1b", "C2b", "C3", "C4"],
            },
            aggregate=gate,
        ),
        "criteria": [
            {"name": "C1_precommit_responds_to_tonic", "load_bearing": False,
             "passed": c1},
            {"name": "C2_committed_responds_to_tonic", "load_bearing": True,
             "passed": c2},
            {"name": "C3_dacc_operative_with_stack_armed", "load_bearing": False,
             "passed": c3},
            {"name": "C4_behavioural_coverage_nondegenerate", "load_bearing": False,
             "passed": bool(grid["c4_behavioural_coverage_nondegenerate"]["passed"])},
        ],
        "per_arm_gate": gate.get("per_arm_gate"),
        "dv_symmetry_declaration": {
            "ARM_1_mech313_only": (
                "committed_action_entropy: manipulation IS invariant (uniform "
                "temperature = monotone rescaling; argmin order-preserving) -- this "
                "is the HYPOTHESIS UNDER TEST, measured not assumed. "
                "precommit_entropy: NOT invariant (softmax is temperature-scaled by "
                "construction) -- the positive control."),
            "ARM_3_both_on": "As ARM_1, replicated in the presence of dACC.",
            "ARM_2_mech260_only": (
                "dACC suppression is a PER-CANDIDATE additive penalty "
                "(count(c in history)/len), not a broadcast scalar, so it is NOT "
                "invariant under the committed argmin. Measurable."),
            "ARM_0_both_off": "No manipulation; the reference arm.",
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global TOTAL_RUNS
    t0 = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else SEEDS
    TOTAL_RUNS = len(seeds) * len(ARMS)

    # Design-time refusal: a precondition that no arm could satisfy from its own
    # PRE-REGISTERED config is a proof, not a substrate fact. Runs before compute.
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS, [_arm_context(a) for a in ARMS],
        arm_id_key="arm_id")

    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5
    else:
        total_eps = (STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET
                     + HAZARD_STAGE_BUDGET + P1_BUDGET)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            rows.append(_run_seed_arm(arm, seed, dry_run, total_eps))

    gate = _evaluate_arm_gates(rows)
    grid = _evaluate_grid(rows)
    interpretation = _interpret(gate, grid, rows)

    label = interpretation["label"]
    # A diagnostic PASSes when it produced an interpretable reading; it FAILs when
    # it could not (substrate_not_ready_requeue).
    outcome = "FAIL" if label == "substrate_not_ready_requeue" else "PASS"

    per_claim = {
        "Q-045": "non_contributory",
        "MECH-313": "non_contributory",
        "MECH-260": "non_contributory",
    }

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        # Diagnostic: excluded from confidence/conflict scoring. Every claim reads
        # non_contributory -- this probe measures whether Q-045 is ANSWERABLE, it
        # does not answer it.
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": per_claim,
        "sleep_driver_pattern": "K=never (no sleep loop; waking action selection only)",
        "substrate": (
            "scaffolded_sd054_onboarding (full curriculum, harm-pathway ON) + "
            "569i GAP-A conversion stack ARMED constant on every arm"),
        "non_degenerate": bool(gate.get("non_degenerate", False)),
        "degeneracy_reason": gate.get("degeneracy_reason"),
        "pre_registered_thresholds": {
            "entropy_margin_nats": ENTROPY_MARGIN_NATS,
            "sd_margin_multiplier": SD_MARGIN_MULTIPLIER,
            "position_entropy_floor": POSITION_ENTROPY_FLOOR,
            "tonic_temp_lift_floor": TONIC_TEMP_LIFT_FLOOR,
            "fresh_select_floor": FRESH_SELECT_FLOOR,
            "committed_floor": COMMITTED_FLOOR,
            "route_range_floor": ROUTE_RANGE_FLOOR,
            "fifo_warmup_fresh_selects": FIFO_WARMUP_FRESH_SELECTS,
            "min_fraction": round(MIN_FRACTION, 6),
        },
        "conversion_stack": {
            "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
            "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
            "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
            "use_e3_score_diversity": USE_E3_SCORE_DIVERSITY,
            "use_gap_scaled_commit_temperature": USE_GAP_SCALED_COMMIT_TEMPERATURE,
            "validated_by": "v3_exq_569i_gapa_conversion_topk_shortlist_falsifier (PASS)",
        },
        "dissociation_grid": grid,
        "per_arm_gate": gate.get("per_arm_gate"),
        "interpretation": interpretation,
        "arm_results": rows,
        "per_seed_rows": rows,
        "eval_episodes": EVAL_EPISODES,
        "p2_steps_per_episode": P2_STEPS_PER_EPISODE,
        "noise_floor_alpha": NOISE_FLOOR_ALPHA,
        "dacc_suppression_weight": DACC_SUPPRESSION_WEIGHT,
        "dacc_suppression_memory": DACC_SUPPRESSION_MEMORY,
        "dry_run": bool(dry_run),
    }

    full_config = {
        "arms": ARMS,
        "scaffold": _config_slice(ARMS[0], dry_run),
        "seeds": seeds,
        "eval_episodes": EVAL_EPISODES if not dry_run else 1,
        "p2_steps_per_episode": P2_STEPS_PER_EPISODE if not dry_run else 30,
    }
    stamp_recording_core(
        manifest, config=full_config, seeds=seeds,
        script_path=Path(__file__), started_at=t0,
    )
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V3-EXQ-687a dissociation probe")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny-budget smoke test (1 seed, toy episodes)")
    args = ap.parse_args()

    manifest = run_experiment(dry_run=args.dry_run)
    # NB: the second positional of write_flat_manifest is `out_dir`, NOT the
    # experiment type -- passing EXPERIMENT_TYPE there would redirect the manifest
    # into a directory named after the experiment.
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=manifest.get("seeds"),
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
    )

    print("")
    print(f"outcome: {manifest['outcome']}")
    print(f"label: {manifest['interpretation']['label']}")
    print(f"non_degenerate: {manifest['non_degenerate']}")
    print(f"manifest: {out_path}")

    _o = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
