"""
V3-EXQ-1000 -- SD-e1-rollout-consistency-training ITEM 3 validation: does the
rollout-ENDPOINT InfoNCE contrastive objective (E1DeepPredictor.
rollout_sequence_divergence_loss, e1_rollout_sequence_divergence_*) lift
per-action rollout divergence at the E1 output -- cr_ratio(h=1) -- over three
incumbents (the depth-0 single-step baseline, V3-EXQ-968's stateful
output_proj_residual anchor, and V3-EXQ-976's stronger rc_decay candidate),
on the ITEM-1-ON substrate?

Claims: [] (diagnostic; validates the SUBSTRATE, not MECH-135/INV-088. Both
keep pending_retest_after_substrate: true regardless of outcome here.)
DIAGNOSTIC. unblocks_claims (per substrate_queue entry, NOT claim_ids on this
run): MECH-135, INV-088.

SLEEP DRIVER: none (no sleep flags set).

WHY THIS RUN EXISTS
---------------------------------------------------------------------------
ITEM 3 (E1DeepPredictor.rollout_sequence_divergence_loss) landed on ree-v3
origin/main 2026-09-03 (df551f38fe) -- a rollout-endpoint InfoNCE contrastive
over K sibling candidate action SEQUENCES sharing an initial state, licensed
by the confirmed V3-EXQ-976 autopsy (user decision Q1, 2026-09-02): ITEM 2
(candidate 1, trajectory-accuracy rollout_consistency_loss) made no absolute
progress on the evaluator bars and DAMPS per-action divergence growth at
depth on 8/8 ON cells -- an accuracy objective trained against observed
intermediate states works AGAINST the per-action divergence the evaluator
needs. ITEM 3 is purely a function of the rollout ENDPOINT (no per-step MSE
against intermediate states at all), so it cannot be minimised by the same
collapse-toward-a-common-trajectory move; a collapsed configuration is
exactly what it penalises hardest (verified by ITEM 3's own contract suite:
the gradient's dot product with the direction toward each candidate's own
target is positive, and toward every other candidate's target is negative).

This run is ITEM 3's owed validation experiment (substrate_queue entry
SD-e1-rollout-consistency-training, implementation_log item 3
validation_status "BUILT, not yet validated -- validation experiment owed").

ARMS (all ITEM-1-ON: action_conditioned_transition=True,
action_cond_unzero_self_slot=True (its 965 null is NOT re-ablated here),
output_proj_residual only on ARM_ANCHOR (the 968-spent branch's own form)):
  ARM_OFF      ROUTED incumbent: depth-0-symmetrised single-step teacher-
               forced MSE at horizon 1, trained from a zero hidden state
               (976's ARM_single_step, verbatim symmetrisation).
  ARM_ANCHOR   V3-EXQ-968's stateful anchor: stateful forward() call (hidden
               state persists and accumulates within an episode), WITH
               output_proj_residual=True -- the strongest prior incumbent
               per the SD-e1 substrate_queue entry's own record ("carried
               forward as the strongest prior incumbent"). NOTE this is a
               DIFFERENT symmetrisation regime from ARM_OFF (stateful vs
               depth-0) -- carried deliberately, as the entry's own text
               names it as the incumbent to beat, not as a routing-neutral
               anchor the way 976 used its stateful arm.
  ARM_RC_DECAY ITEM 2, rollout_consistency_loss, horizon 5, decay=0.5 (the
               stronger of 976's two ON arms on 3/4 seeds).
  ARM_RSD      ITEM 3, rollout_sequence_divergence_loss, defaults
               (weight=1.0, horizon=5 (RSD_HORIZON), temperature=0.1,
               min_batch_classes=2). The candidate under test.

ARM_OFF is the ROUTED baseline; ARM_ANCHOR, ARM_RC_DECAY, ARM_RSD are the
three ON arms, each scored by relative lift of cr_ratio(h=1) over ARM_OFF
against the FIXED pre-registered LIFT_BAR (3.0) -- the identical convention
V3-EXQ-976 used for its two ON arms, extended to three. 6 seeds (n>=6, per
this entry's own user_decision_2026_09_02 Q1 -- up from 976's 4 seeds).

UNIFORM TRAINING-WINDOW TRIGGER (load-bearing for cross-arm step-count
parity -- reused verbatim from 976). ALL FOUR arms share one trailing-window
trigger: `if len(totals) >= H + 1` with H = TRAIN_WINDOW_H = 5, EVEN THOUGH
ARM_OFF's and ARM_ANCHOR's own training objective is single-step (H=1).
This is 976's own device (see 976's docstring "MATCHED TRAINING ACROSS
ARMS"): every arm takes exactly one E1 optimiser step per env step once a
trailing window of TRAIN_WINDOW_H observed latents exists, so the
e1_grad_steps_matched readiness precondition is TRUE BY CONSTRUCTION rather
than by post-hoc measurement. ARM_OFF and ARM_ANCHOR use only the window's
first step (total_i, a_i) -> total_{i+1}; ARM_RC_DECAY uses the full H-step
window (976's mechanism, unchanged); ARM_RSD BUFFERS the full H-step window
(initial, H-step action sequence, observed endpoint) into a replay buffer of
REAL executed windows and, once the buffer holds >= RSD_MIN_BATCH_CLASSES
windows, samples K = RSD_BATCH_K of them (without replacement) each trigger
tick to build one rollout_sequence_divergence_loss call.

RSD TRAINING IS BUFFER-BASED, NOT LITERALLY BRANCHING FROM ONE SHARED
STATE -- A DELIBERATE DESIGN CHOICE, NOT A DEVIATION FROM THE LOSS'S OWN
CONTRACT. rollout_sequence_divergence_loss's docstring frames its intended
call shape as "K sibling candidate action SEQUENCES sharing one initial
state" (a CEM-proposer-style branching call), but its own signature accepts
initial_state as EITHER a single vector broadcast to K OR a genuinely
per-sample [K, total_dim] tensor -- and this driver uses the latter. This
mirrors the ESTABLISHED precedent for exactly the same shape of objective on
E2: SD-056's world_forward_contrastive_loss is likewise documented as "K
sibling CEM candidates sharing z_world_0" at its comparator call site inside
the hippocampal proposer, but is ALSO called, unmodified, against a REPLAY
BUFFER of K independently-timestamped real (z0, a, z1) transitions at
several training call sites in this repo (e.g. V3-EXQ-701c). The InfoNCE
mechanism does not require a literal shared origin to be informative: it
only needs K real, independently-observed (window, endpoint) pairs to serve
as mutual negatives. Branching K candidate sequences from one literal env
snapshot via copy.deepcopy(env) (the pattern used by V3-EXQ-817/817a) was
considered and rejected for the TRAINING loop specifically: it would require
either K-fold more env-stepping per training tick (an env is stepped H times
per candidate, K candidates, every trigger tick -- a >K-fold slowdown over
the single-trajectory arms) or discarding the single continuous online
random-policy trajectory the other three arms train on, breaking the
"identical random policy, env, seeds" cross-arm control 976 established.
The buffer-based form keeps that control intact and is licensed by the
existing SD-056 precedent. Degeneracy is guarded exactly as the loss itself
guards it (distinct-FULL-SEQUENCE floor, not first-action-only) via the
rsd_objective_engaged readiness precondition below, computed by replicating
the loss's own argmax + unique(dim=0) check BEFORE each training call so a
degenerate call is counted, not silently absorbed into the zero-loss branch.

READINESS PRECONDITIONS (any unmet -> substrate_not_ready_requeue, FAIL, no
verdict). 968/976's five verbatim (encoder_trained; real_zworld_nondegenerate_h1;
no_missing_action_calls; direct_action_supply_fraction >= 0.999;
cr_ratio_h1_finite) plus THREE this design needs:
  e1_grad_steps_matched   every arm took the same number of E1 optimiser
                          steps at a seed. TRUE BY CONSTRUCTION here (see
                          "UNIFORM TRAINING-WINDOW TRIGGER" above) -- kept as
                          a measured precondition rather than assumed, per
                          976's own discipline.
  rsd_objective_engaged   ARM_RSD's training calls were non-degenerate
                          (n_distinct_full_sequences >= RSD_MIN_BATCH_CLASSES)
                          on a nonzero fraction of ticks, on every seed. If
                          this fails, ARM_RSD's own loss NEVER meaningfully
                          engaged (every call fell through to the
                          grad-connected zero-loss branch) -- the run
                          self-routes to substrate_not_ready_requeue rather
                          than reporting a "null" verdict for an arm that was
                          never actually tested. Per the SD entry's own
                          instruction: "an arm whose loss never engages must
                          self-route to substrate_not_ready_requeue, never a
                          claim verdict." NOTE (red-team finding #1/#2,
                          2026-09-03): this precondition is a property of the
                          SAMPLED BATCH (distinct action-class diversity), not
                          of what the model DOES with it -- it cannot by
                          itself rule out an action-blind "identity shortcut"
                          (see rsd_action_sensitive immediately below).
  rsd_action_sensitive    ARM_RSD's loss actually depends on the sampled
                          action sequences, not merely on the sampled inits.
                          Under the replay-buffer substitution (K windows
                          drawn from different times along ONE continuous
                          trajectory, per "RSD TRAINING IS BUFFER-BASED"
                          below, rather than literal siblings of one shared
                          state) a slow-drift environment opens an identity
                          shortcut: E1 can minimise the K-way contrastive
                          loss by predicting endpoint ~= init, since
                          temporally-close windows are spatially close
                          regardless of which actions separate them -- with
                          ZERO use of the action sequence, and
                          rsd_objective_engaged would still read True (the
                          batch is genuinely action-diverse; the MODEL just
                          ignores that diversity). Measured by an
                          action-sensitivity control: every RSD training tick,
                          recompute the identical loss with the K action
                          sequences ROW-PERMUTED across samples (init/endpoint
                          pairing unchanged) under no_grad via an isolated RNG
                          stream (rsd_shuffle_rng, never touching
                          rsd_sample_rng or the shared `random` module used
                          for action selection). If E1 is action-blind,
                          permuting which action sequence is attached to which
                          init/endpoint pair changes nothing about
                          predict_long_horizon's output for each row, so
                          shuffled loss ~= real loss (ratio ~= 1.0, the null
                          value). If E1 is genuinely action-sensitive,
                          permuting degrades the K-way classification (each
                          row's prediction no longer tracks the true
                          generating action sequence for its assigned
                          target), so shuffled loss is materially larger.
                          RSD_ACTION_SENSITIVITY_RATIO_FLOOR = 1.05 (a
                          conservative 5% margin above the exact null of
                          1.0) must be cleared on every seed, else
                          self-route to substrate_not_ready_requeue -- an arm
                          whose loss engaged only on an action-blind basis is
                          exactly as untested, for this experiment's purpose,
                          as an arm whose loss never engaged at all.

SCOPE REDUCTION FROM 976 (recorded, not hidden). This driver does NOT
replicate 976's bespoke gradient-cosine (grad_cos) non-vacuity apparatus for
ARM_ANCHOR / ARM_RC_DECAY -- those two arms reuse already-landed, already-
autopsy-confirmed code paths (V3-EXQ-968 validated the stateful+residual
combination as a null vs the plain stateful form; V3-EXQ-976 validated
rc_decay's non-vacuity via exactly that apparatus), so re-deriving their
own non-vacuity here would duplicate settled work. rsd_objective_engaged is
this run's ONE new non-vacuity gate, scoped to the ONE new objective under
test.

DVs AND DECISION RULE (pre-registered)
---------------------------------------------------------------------------
Primary: cr_ratio(h=1) = CR_rollout(h=1) / CR_real(h=1) per (arm, seed),
Phase 4/4b (identical statistic and computation to 968/976).
Absolute evaluator bars (per this SD entry's own text, RECORDED and
headlined, ROUTING per the label rule below): cr_ratio(h=1) >= 0.1
(CR_ROLLOUT_COLLAPSE_RATIO) AND e1coe_score_var(h=1) >= 0.002
(C3_VAR_THRESHOLD).
Relative lift (976's convention, extended to 3 ON arms): per seed,
relative_lift = cr_ratio_h1(ON arm) / cr_ratio_h1(ARM_OFF). LIFT_BAR = 3.0,
fixed and pre-registered (976's B3 red-team finding: a noise-inflated bar is
NOT applied here either -- recorded as a diagnostic only). Per ON arm:
LIFTS if relative lift >= LIFT_BAR on >= MAJORITY_SEEDS (n_seeds//2+1, i.e.
4 of 6); DEGRADES if 1/lift >= LIFT_BAR on >= MAJORITY_SEEDS; NULL if
neither direction fires on ANY seed; MIXED otherwise.
SECONDARY at h=RSD_HORIZON=5 (976's trained-horizon override, applied to
ARM_RC_DECAY and ARM_RSD, both of which have a well-defined trained
horizon; ARM_ANCHOR has none and is exempt): a null/mixed/degrades at h=1
with a LIFT at h=5 becomes "..._lift_at_trained_horizon_only", which does
NOT license a strong claim -- it says h=1 is the wrong readout.

LABEL COMPOSITION (this run's own routing, extending 976's two-ON-arm
scheme to three):
  rollout_endpoint_contrastive_clears_evaluator_bars
    ARM_RSD clears BOTH absolute evaluator bars on >= MAJORITY_SEEDS seeds
    at h=1 -- supports the entry's remaining hypothesis (Q1's "ITEM 3
    clears an absolute bar" branch).
  rollout_endpoint_contrastive_lifts_cr_ratio_h1
    ARM_RSD's relative-lift verdict is "lifts" (but does not clear the
    absolute bars) -- progress on the SD's target, not its closure.
  rollout_endpoint_contrastive_null_others_lift
    ARM_RSD is "null" while at least one of ARM_ANCHOR / ARM_RC_DECAY
    "lifts" -- the contrastive candidate is specifically a null on this
    entry, but the crush is not confirmed everywhere (Q1's "ITEM 3 no
    better than rc_decay/anchor" branch).
  all_arms_null_residual_crush_locus_elsewhere
    ARM_ANCHOR, ARM_RC_DECAY, AND ARM_RSD are all "null" -- contributes a
    3rd leg to the sd_e1_residual_crush_locus ledger (Q1's "all arms
    damped" branch, per user_decision_2026_09_02 Q3).
  rollout_endpoint_contrastive_degrades_cr_ratio_h1
    ARM_RSD's verdict is "degrades" and no other arm lifts.
  rollout_endpoint_contrastive_lift_at_trained_horizon_only
    ARM_RSD null/mixed/degrades at h=1 but lifts at h=RSD_HORIZON=5 (976's
    trained-horizon override, applied identically here).
  mixed_across_arms
    anything else.
  substrate_not_ready_requeue
    any readiness precondition unmet (including rsd_objective_engaged).

GOV-REUSE-1 (Step 2.4): decisive readout = cr_ratio(h=1) on an E1 trained
with rollout_sequence_divergence_loss. No manifest carries it -- ITEM 3
landed 2026-09-03 and this is its first validation run. Checked
evidence/experiments/ for any prior v3_exq_1000* / *rollout_sequence_
divergence* manifest: none exists. Not recoverable -> run.

STEP 2.5 / 2.5a: ITEM 3 substrate confirmed IMPLEMENTED in ree-v3/CLAUDE.md
("SD-e1-rollout-consistency-training ITEM 3" entry, 2026-09-03) and
empirically probed against origin/main df551f38fe (method signature,
E1Config fields, and REEConfig.from_dims() wiring for all five
e1_rollout_sequence_divergence_* knobs confirmed present at all three sites
-- dataclass field, from_dims signature, from_dims assignment).

STEP 2.5b (re-derive brake): checked failure_autopsy_V3-EXQ-108b, -965, -976
for MECH-135 / INV-088. All three are "substrate_ceiling"-adjacent but each
routed a DIFFERENT, now-landed substrate fix (108b -> action-conditioning;
965 -> confirmed ITEM 1; 976 -> confirmed ITEM 2 null, licensing ITEM 3)
rather than repeating the same probe -- this is exactly the
substrate-genuinely-enriched-between-lettered-iterations case the brake
does not apply to (this is also a NEW EXQ number, not a lettered iteration
of any of the three).

STEP 2.5c: no open substrate_queue entry with severity=corrupting overlaps
ree_core/predictors/e1_deep.py at the time of writing (SD-e1-rollout-
consistency-training's own severity is not corrupting; it is the entry
under test).

CROSS-MACHINE-CLASS SAFETY: no torch.multinomial-dependent assertions
anywhere in this driver; every DV is a continuous rollout statistic
(cr_ratio, e1coe_score_var, spread/centroid_norm). Priority 60
(front-critical, per CURRENT_FRONT.md / insights_report.md Recommendation
1). machine_affinity any (cloud-preferred).
EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence
scoring.
red-team (fable): BLOCKING on first pass (2 findings: #1/#2 -- the replay-buffer
substitution could let RSD's loss be minimised via an action-blind "identity
shortcut" that rsd_objective_engaged cannot detect), FIXED via the
rsd_action_sensitive precondition + action-sensitivity control above; plus
finding #4a (trained-horizon override could relabel a genuine h=1 degradation
as partial support) FIXED by excluding "degrades" from the override-eligible
label set. See report for the full disposition of every finding, including
the CONTESTED/lower-severity ones left as recorded limitations rather than
fixed (statistical power at n=6/majority-4, the h=5-vs-h=1 loss/DV mismatch,
and the trained-vs-eval hidden-state mismatch on ARM_ANCHOR).
ASCII-only output (repo rule).
"""

import itertools
import sys
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalConfig, GoalState
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_encoder_guard import (
    latent_stack_snapshot,
    assert_world_encoder_trained,
)
from experiments._lib.arm_fingerprint import arm_cell


EXPERIMENT_TYPE = "v3_exq_1000_sd_e1_item3_rollout_endpoint_contrastive_validation"
CLAIM_IDS: List[str] = []
UNBLOCKS_CLAIMS: List[str] = ["MECH-135", "INV-088"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None

# agent_construction_before_seed lint exemption -- same shape and same
# justification as V3-EXQ-954/965/976: every scored quantity in this script
# is read off the LITERAL SAME agent object within a (seed, arm) cell.
# Cross-arm comparisons ARE across independently-constructed agents by
# design (that is the whole point of the A/B) and are seed-matched via
# arm_cell()'s RNG reset.
ANCHOR_REACHABILITY_EXEMPT = (
    "e1_grad_steps_matched IS the degeneracy definition -- it measures the "
    "cross-arm step-count gap under the uniform TRAIN_WINDOW_H trigger that this "
    "script constructs deliberately so the gap is exactly 0 (see module docstring "
    "'UNIFORM TRAINING-WINDOW TRIGGER'); it is not a hand-narrowed predicate on an "
    "external signal. rsd_objective_engaged IS ALSO the degeneracy definition for "
    "ARM_RSD's contrastive loss -- it replicates rollout_sequence_divergence_loss's "
    "own internal argmax+unique(dim=0) non-degeneracy check (see 'RSD TRAINING IS "
    "BUFFER-BASED' in the module docstring) so a control that clears it is by "
    "construction the same event the loss itself would count as engaged, not a "
    "narrower proxy for it."
)

AGENT_SEED_ORDER_EXEMPT = (
    "Every within-cell comparison (action-vs-action, horizon-vs-horizon) is "
    "scored off the literal same agent object; cross-arm comparisons are the "
    "A/B itself and are seed-matched via arm_cell()."
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
CR_REAL_FLOOR = 1e-4               # unchanged from V3-EXQ-954/965/968/976
CR_ROLLOUT_COLLAPSE_RATIO = 0.1    # ITEM 2/3's evaluator bar -- headlined
C3_VAR_THRESHOLD = 0.002           # ITEM 2/3's evaluator bar -- headlined
ZWORLD_P0_EPISODES = 60            # SD-070 encoder warmup -- matches lineage
N_REAL_SAMPLES = 40                # per-checkpoint target sample count
MIN_REAL_SAMPLES_PER_HORIZON = 10  # readiness floor: surviving real samples
HORIZON_CHECKPOINTS_FULL = [1, 2, 3, 5, 10, 20, 30]

LIFT_BAR = 3.0                     # FIXED pre-registered relative-lift bar

# ITEM 2 objective parameters (ARM_RC_DECAY).
RC_HORIZON = 5
RC_DECAY = 0.5
RC_WEIGHT = 1.0

# ITEM 3 objective parameters (ARM_RSD) -- all E1Config defaults, per the
# entry's own "defaults otherwise" instruction.
RSD_HORIZON = 5
RSD_TEMPERATURE = 0.1
RSD_MIN_BATCH_CLASSES = 2
RSD_WEIGHT = 1.0
RSD_BATCH_K = 8                    # this driver's own training-batch size
RSD_BUFFER_MAX = 256               # FIFO cap on the real-window replay buffer

# Action-sensitivity control floor (red-team finding #1/#2, 2026-09-03): the
# minimum tolerated ratio of (mean loss under a row-permuted action
# assignment) / (mean loss under the real assignment), across every RSD
# training tick in a cell. 1.0 = shuffling the actions made no difference to
# the loss (the "identity shortcut" failure mode -- action-blind). Set
# modestly above 1.0 (not exactly 1.0) so ordinary float/minibatch noise on a
# ratio that is genuinely ~1.0 does not spuriously pass; NOT tuned to make
# this run pass -- it is a floor on a ratio whose null value is exactly 1.0,
# and 1.05 is a conservative 5% margin above that null.
RSD_ACTION_SENSITIVITY_RATIO_FLOOR = 1.05

# Uniform trailing-window trigger shared by ALL FOUR arms (see docstring
# "UNIFORM TRAINING-WINDOW TRIGGER"). Deliberately equal to RC_HORIZON /
# RSD_HORIZON so ARM_RC_DECAY's and ARM_RSD's full-window objectives use
# exactly the window this trigger already produces.
TRAIN_WINDOW_H = 5

OFF_ARM = "ARM_OFF"
ON_ARMS = ["ARM_ANCHOR", "ARM_RC_DECAY", "ARM_RSD"]
ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ARM_OFF": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": False,
        "e1_rollout_sequence_divergence_enabled": False,
        "e1_loss": "single_step_depth0",
    },
    "ARM_ANCHOR": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": True,
        "e1_rollout_consistency_enabled": False,
        "e1_rollout_sequence_divergence_enabled": False,
        "e1_loss": "single_step_stateful",
    },
    "ARM_RC_DECAY": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": True,
        "e1_rollout_sequence_divergence_enabled": False,
        "e1_loss": "rollout_consistency",
    },
    "ARM_RSD": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": False,
        "e1_rollout_sequence_divergence_enabled": True,
        "e1_loss": "rollout_sequence_divergence",
    },
}
ARM_ORDER = [OFF_ARM] + ON_ARMS

SEEDS_DEFAULT = [42, 123, 7, 2024, 17, 31]   # n=6 per user_decision_2026_09_02 Q1;
                                              # 42/123/7/2024 = the 976 lineage
                                              # seeds; 17/31 disjoint from any
                                              # authoring pilot seed; reef config
                                              # not live here, so seed 44/45's
                                              # instability convention does not
                                              # apply, but both are still avoided.


# ---------------------------------------------------------------------------
# Helpers (unchanged from V3-EXQ-954/965/976 unless noted)
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
    """Env config, unchanged from V3-EXQ-954/965/976."""
    return dict(
        size=10, num_hazards=2, num_resources=4,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.05,
        proximity_harm_scale=0.03, proximity_benefit_scale=0.04,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
        resource_respawn_on_consume=True,
    )


def _build_agent(
    seed: int, world_dim: int, self_dim: int, arm: str,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    arm_cfg = ARM_CONFIGS[arm]
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        action_conditioned_transition=arm_cfg["action_conditioned_transition"],
        action_cond_unzero_self_slot=arm_cfg["action_cond_unzero_self_slot"],
        output_proj_residual=arm_cfg["output_proj_residual"],
        e1_rollout_consistency_enabled=arm_cfg["e1_rollout_consistency_enabled"],
        e1_rollout_consistency_weight=RC_WEIGHT,
        e1_rollout_consistency_horizon=RC_HORIZON,
        e1_rollout_consistency_horizon_weights_decay=RC_DECAY,
        e1_rollout_sequence_divergence_enabled=arm_cfg["e1_rollout_sequence_divergence_enabled"],
        e1_rollout_sequence_divergence_weight=RSD_WEIGHT,
        e1_rollout_sequence_divergence_horizon=RSD_HORIZON,
        e1_rollout_sequence_divergence_temperature=RSD_TEMPERATURE,
        e1_rollout_sequence_divergence_min_batch_classes=RSD_MIN_BATCH_CLASSES,
    )
    config.latent.unified_latent_mode = False
    # from_dims swallows unknown kwargs silently (reference-reeconfig-from-dims-
    # silent-kwargs); assert every knob this design depends on actually landed.
    assert bool(config.e1.action_conditioned_transition) is True, arm
    assert bool(config.e1.output_proj_residual) == bool(arm_cfg["output_proj_residual"]), arm
    assert bool(config.e1.e1_rollout_consistency_enabled) == bool(arm_cfg["e1_rollout_consistency_enabled"]), arm
    assert bool(config.e1.e1_rollout_sequence_divergence_enabled) == bool(
        arm_cfg["e1_rollout_sequence_divergence_enabled"]
    ), arm
    if arm_cfg["e1_rollout_sequence_divergence_enabled"]:
        assert int(config.e1.e1_rollout_sequence_divergence_horizon) == RSD_HORIZON, arm
        assert int(config.e1.e1_rollout_sequence_divergence_min_batch_classes) == RSD_MIN_BATCH_CLASSES, arm
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0a: SD-070 sanctioned z_world encoder warmup (unchanged from lineage)
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_1000 P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context="v3_exq_1000_sd_e1_item3_rollout_endpoint_contrastive_validation",
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 training, action-conditioned, uniform trailing
# window (H=TRAIN_WINDOW_H) shared by all four arms.
# ---------------------------------------------------------------------------

def _single_step_loss_state_preserving(
    agent: REEAgent, initial: torch.Tensor, action: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Single-step teacher-forced loss from a ZERO hidden state, with E1's
    hidden state saved and restored around the call (976's B1 symmetrisation,
    verbatim). ARM_OFF only."""
    saved = agent.e1._hidden_state
    agent.e1.reset_hidden_state()
    try:
        e1_pred, _ = agent.e1(initial, horizon=1, actions=action)
        return F.mse_loss(e1_pred[:, 0, :], target)
    finally:
        agent.e1._hidden_state = saved


def _flat_grad(loss: torch.Tensor, params: List[torch.nn.Parameter]) -> Optional[torch.Tensor]:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    parts = [g.reshape(-1) for g in grads if g is not None]
    if not parts:
        return None
    return torch.cat(parts)


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    e1_call_counter: Dict[str, int],
    arm: str,
) -> Dict[str, Any]:
    """Phase 0b: E1/E2 training on a random policy (StepHarness invariant #1:
    agent.sense() exactly ONCE per env step, under torch.no_grad(), so Phase
    0a's trained encoder is never disturbed).

    UNIFORM TRIGGER for all four arms: exactly one E1 optimiser step per env
    step once a trailing window of TRAIN_WINDOW_H observed latents exists
    (window start i = t - TRAIN_WINDOW_H). See module docstring "UNIFORM
    TRAINING-WINDOW TRIGGER" for why this makes e1_grad_steps_matched true
    by construction across all four arms.

    single_step_depth0    F.mse_loss(e1(total_i, h=1, a_i)[:,0], total_{i+1}),
                          zero hidden state (976 B1 symmetrisation). ARM_OFF.
    single_step_stateful  968's incumbent verbatim: stateful forward() call,
                          hidden state persists/accumulates across the
                          episode. ARM_ANCHOR (output_proj_residual=True on
                          the model this loss trains).
    rollout_consistency   agent.e1.rollout_consistency_loss(total_i,
                          stack(total_{i+1..i+H}), actions=stack(a_i..a_{i+H-1}),
                          horizon=H, horizon_weights_decay=RC_DECAY) *
                          RC_WEIGHT. ARM_RC_DECAY. 976's mechanism, unchanged.
    rollout_sequence_divergence
                          buffers the full H-step window (initial, action
                          sequence, observed endpoint) into a REAL-window
                          replay buffer; once the buffer holds
                          >= RSD_MIN_BATCH_CLASSES windows, samples K =
                          min(RSD_BATCH_K, len(buffer)) windows WITHOUT
                          replacement and calls
                          agent.e1.rollout_sequence_divergence_loss(init,
                          action_sequences, endpoint_targets,
                          horizon=RSD_HORIZON) * RSD_WEIGHT. ARM_RSD. See
                          module docstring "RSD TRAINING IS BUFFER-BASED".
    """
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    # Dedicated RNG stream for RSD buffer sampling (random.sample), NEVER the
    # shared `random` module -- drawing from `random` here would consume
    # entries from the SAME stream that drives action_idx = random.randint(...)
    # below, perturbing the action trajectory ARM_RSD walks relative to the
    # other three arms (which never call random.sample). That breaks BOTH the
    # "identical random policy, env, seeds" cross-arm control this driver's
    # docstring promises AND the uniform e1_grad_steps_matched precondition
    # (a perturbed trajectory can end an episode via `done` at a different
    # step, changing the per-arm window-trigger count). Confirmed empirically
    # during --dry-run smoke: without this isolation, e1_grad_step_gap_per_seed
    # was 27/8 instead of 0/0.
    rsd_sample_rng = random.Random(seed + 9000)
    # Second, independently-isolated RNG stream for the action-sensitivity
    # control's row-permutation draw (red-team finding #1/#2, see the control
    # block below) -- kept separate from rsd_sample_rng so the CONTROL draw
    # can never perturb which K windows the REAL training call samples.
    rsd_shuffle_rng = random.Random(seed + 9500)

    arm_cfg = ARM_CONFIGS[arm]
    loss_kind = str(arm_cfg["e1_loss"])
    H = int(TRAIN_WINDOW_H)  # window length is the SAME in every arm
    e1_params = [q for q in agent.e1.parameters() if q.requires_grad]

    opt_e1 = optim.Adam(agent.e1.parameters(), lr=1e-3)
    opt_e2 = optim.Adam(agent.e2.parameters(), lr=1e-3)

    rsd_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    rsd_n_engaged = 0
    rsd_n_degenerate = 0
    rsd_n_buffer_starved = 0
    rsd_real_loss_sum = 0.0
    rsd_shuffled_loss_sum = 0.0
    rsd_n_sensitivity_checks = 0

    stats: Dict[str, Any] = {
        "e1_loss_kind": loss_kind,
        "train_window_h": H,
        "n_e1_grad_steps": 0,
        "n_e2_grad_steps": 0,
        "n_windows": 0,
        "trained_loss_mean": 0.0,
        "per_episode_trained_loss": [],
        "n_nonfinite_losses": 0,
        "rsd_n_engaged": 0,
        "rsd_n_degenerate": 0,
        "rsd_n_buffer_starved": 0,
        "rsd_engaged_frac": 0.0,
        "rsd_action_sensitivity_ratio": float("nan"),
    }
    trained_sum = 0.0

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_loss_e1 = 0.0
        ep_loss_e2 = 0.0
        n_steps = 0

        totals: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        latent_prev: Optional[object] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent_curr = agent.sense(obs_body, obs_world)

            action_idx = random.randint(0, env.action_dim - 1)
            action_curr = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent.record_executed_action(action_curr)

            total_curr = torch.cat([latent_curr.z_self, latent_curr.z_world], dim=-1).detach()
            totals.append(total_curr)
            actions.append(action_curr)

            # E2 (not under test): identical in every arm.
            if latent_prev is not None:
                opt_e2.zero_grad()
                z_self_pred = agent.e2.predict_next_self(latent_prev.z_self.detach(), action_prev)
                e2_loss = F.mse_loss(z_self_pred, latent_curr.z_self.detach())
                e2_loss.backward()
                opt_e2.step()
                ep_loss_e2 += e2_loss.item()
                stats["n_e2_grad_steps"] += 1

            # E1: one optimiser step per env step on the trailing window.
            if len(totals) >= H + 1:
                i = len(totals) - 1 - H
                initial = totals[i]
                window_targets = torch.stack(totals[i + 1:i + 1 + H], dim=1)  # [1,H,total_dim]
                window_acts = torch.stack(actions[i:i + H], dim=1)            # [1,H,action_dim]

                if loss_kind == "single_step_depth0":
                    trained = _single_step_loss_state_preserving(agent, initial, actions[i], totals[i + 1])
                    e1_call_counter["n_e1_calls"] += 1
                    e1_call_counter["n_e1_calls_nonzero_action"] += 1
                elif loss_kind == "single_step_stateful":
                    # 968's incumbent VERBATIM: stateful forward() call.
                    e1_pred, _ = agent.e1(initial, horizon=1, actions=actions[i])
                    trained = F.mse_loss(e1_pred[:, 0, :], totals[i + 1])
                    e1_call_counter["n_e1_calls"] += 1
                    e1_call_counter["n_e1_calls_nonzero_action"] += 1
                elif loss_kind == "rollout_consistency":
                    trained = agent.e1.rollout_consistency_loss(
                        initial, window_targets, actions=window_acts,
                        horizon=H, horizon_weights_decay=RC_DECAY,
                    )
                    e1_call_counter["n_e1_calls"] += 1
                    e1_call_counter["n_e1_calls_nonzero_action"] += 1
                elif loss_kind == "rollout_sequence_divergence":
                    # buffer the real window (initial, H-step action seq,
                    # observed endpoint), squeezed to unbatched shape.
                    rsd_buffer.append((
                        initial.squeeze(0).detach().clone(),          # [total_dim]
                        window_acts.squeeze(0).detach().clone(),      # [H, action_dim]
                        totals[i + H].squeeze(0).detach().clone(),    # [total_dim]
                    ))
                    if len(rsd_buffer) > RSD_BUFFER_MAX:
                        rsd_buffer.pop(0)
                    if len(rsd_buffer) < RSD_MIN_BATCH_CLASSES:
                        rsd_n_buffer_starved += 1
                        opt_e1.zero_grad()
                        stats["n_e1_grad_steps"] += 1
                        stats["n_windows"] += 1
                        n_steps += 1
                        _, _, done, _, obs_dict = env.step(action_curr)
                        latent_prev = latent_curr
                        action_prev = action_curr
                        if done:
                            break
                        continue
                    k = min(RSD_BATCH_K, len(rsd_buffer))
                    sampled_idx = rsd_sample_rng.sample(range(len(rsd_buffer)), k)
                    sampled = [rsd_buffer[j] for j in sampled_idx]
                    batch_init = torch.stack([w[0] for w in sampled], dim=0)      # [K,total_dim]
                    batch_acts = torch.stack([w[1] for w in sampled], dim=0)      # [K,H,action_dim]
                    batch_endpoints = torch.stack([w[2] for w in sampled], dim=0)  # [K,total_dim]
                    # replicate rollout_sequence_divergence_loss's own
                    # degeneracy check for instrumentation (distinct FULL
                    # sequences, not first-action-only).
                    seq_classes = batch_acts[:, :RSD_HORIZON, :].argmax(dim=-1)  # [K,h]
                    n_distinct = int(torch.unique(seq_classes, dim=0).shape[0])
                    if n_distinct >= RSD_MIN_BATCH_CLASSES:
                        rsd_n_engaged += 1
                    else:
                        rsd_n_degenerate += 1
                    trained = agent.e1.rollout_sequence_divergence_loss(
                        batch_init, batch_acts, batch_endpoints, horizon=RSD_HORIZON,
                    ) * RSD_WEIGHT
                    e1_call_counter["n_e1_calls"] += 1
                    e1_call_counter["n_e1_calls_nonzero_action"] += 1

                    # ACTION-SENSITIVITY CONTROL (red-team finding #1/#2,
                    # 2026-09-03): the replay-buffer substitution samples K
                    # windows from DIFFERENT times along ONE continuous
                    # random-policy trajectory rather than K siblings sharing
                    # one initial state. In a slow-drift env this opens an
                    # "identity shortcut" -- the loss can be minimised by
                    # predicting endpoint ~= init (temporally-close windows
                    # are spatially close regardless of action), with ZERO
                    # actual use of the action sequence. rsd_objective_engaged
                    # (n_distinct full sequences) cannot detect this: it is a
                    # property of the SAMPLED BATCH, not of what the model
                    # DOES with it, so it reads engaged=True in exactly this
                    # failure mode. This control recomputes the identical loss
                    # with batch_acts ROW-PERMUTED across the K samples (init/
                    # endpoint pairing unchanged) under no_grad, via a
                    # dedicated RNG isolated from both rsd_sample_rng and the
                    # shared `random` module (same isolation rationale as
                    # rsd_sample_rng itself -- must not perturb the env
                    # trajectory or the real training draw). If the model is
                    # action-BLIND (endpoint~=init shortcut), permuting which
                    # action sequence is attached to which init/endpoint pair
                    # changes nothing about predict_long_horizon's output for
                    # each row, so shuffled loss ~= real loss (ratio ~= 1.0).
                    # If the model is genuinely action-sensitive, permuting
                    # the action sequences degrades the K-way classification
                    # (each row's prediction no longer tracks its assigned
                    # target's true generating action sequence), so shuffled
                    # loss > real loss. Accumulated per-arm-seed into
                    # rsd_action_sensitivity_ratio; a ratio near 1.0 marks the
                    # run substrate_not_ready_requeue via the
                    # rsd_action_sensitive readiness precondition below,
                    # rather than reporting a null/lifts verdict for an arm
                    # whose loss engaged on a degenerate (init-driven) basis.
                    with torch.no_grad():
                        shuffle_perm = torch.tensor(
                            rsd_shuffle_rng.sample(range(k), k), dtype=torch.long,
                        )
                        batch_acts_shuffled = batch_acts[shuffle_perm]
                        shuffled_loss = agent.e1.rollout_sequence_divergence_loss(
                            batch_init, batch_acts_shuffled, batch_endpoints,
                            horizon=RSD_HORIZON,
                        )
                        rsd_real_loss_sum += float(trained.item()) / max(RSD_WEIGHT, 1e-12)
                        rsd_shuffled_loss_sum += float(shuffled_loss.item())
                        rsd_n_sensitivity_checks += 1
                else:
                    raise ValueError(f"unknown e1_loss kind: {loss_kind}")

                val = float(trained.item())
                if not (val == val):
                    stats["n_nonfinite_losses"] += 1
                weighted = trained if loss_kind != "rollout_sequence_divergence" else trained
                opt_e1.zero_grad()
                weighted.backward()
                opt_e1.step()
                stats["n_e1_grad_steps"] += 1
                stats["n_windows"] += 1
                trained_sum += val
                ep_loss_e1 += val
                n_steps += 1

            _, _, done, _, obs_dict = env.step(action_curr)
            latent_prev = latent_curr
            action_prev = action_curr
            if done:
                break

        stats["per_episode_trained_loss"].append(ep_loss_e1 / max(n_steps, 1))
        if (ep + 1) % 20 == 0:
            print(
                f"  [Train] label {arm} seed={seed} ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    n_w = max(int(stats["n_windows"]), 1)
    stats["trained_loss_mean"] = trained_sum / n_w
    stats["rsd_n_engaged"] = rsd_n_engaged
    stats["rsd_n_degenerate"] = rsd_n_degenerate
    stats["rsd_n_buffer_starved"] = rsd_n_buffer_starved
    denom = rsd_n_engaged + rsd_n_degenerate
    stats["rsd_engaged_frac"] = (rsd_n_engaged / denom) if denom > 0 else 0.0
    # Action-sensitivity control ratio (red-team finding #1/#2): mean shuffled
    # loss / mean real loss over every RSD training tick this cell ran. ~1.0
    # means permuting which action sequence is attached to which init/endpoint
    # pair did not change the loss -- the model is not reading the action
    # sequence at all (the "identity shortcut" failure mode); the batch could
    # satisfy rsd_objective_engaged (>=2 distinct sequences sampled) while the
    # model uses none of that distinctness. >1 means shuffling genuinely hurt
    # the K-way classification, i.e. the model's endpoint prediction actually
    # depends on the assigned action sequence.
    if rsd_n_sensitivity_checks > 0 and rsd_real_loss_sum > 1e-12:
        stats["rsd_action_sensitivity_ratio"] = rsd_shuffled_loss_sum / rsd_real_loss_sum
    stats["rsd_n_sensitivity_checks"] = rsd_n_sensitivity_checks

    agent.eval()
    print(
        f"  [Train] Done. {n_episodes} episodes; e1_grad_steps={stats['n_e1_grad_steps']} "
        f"trained_loss_mean={stats['trained_loss_mean']:.6e} "
        f"rsd_engaged={rsd_n_engaged} rsd_degenerate={rsd_n_degenerate} "
        f"rsd_buffer_starved={rsd_n_buffer_starved}",
        flush=True,
    )
    return stats


# ---------------------------------------------------------------------------
# Phase 1: goal template (unchanged from lineage)
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, max_steps: int,
) -> Tuple[torch.Tensor, str]:
    torch.manual_seed(seed)
    random.seed(seed)
    _, obs_dict = env.reset()
    agent.reset()

    for _ in range(max_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        agent.record_executed_action(action)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            print(
                f"  [Phase1] Resource contact, z_world_norm={latent.z_world.norm().item():.3f}",
                flush=True,
            )
            return latent.z_world.detach(), "resource_contact"
        if done:
            _, obs_dict = env.reset()
            agent.reset()

    print("  [Phase1] WARNING: no resource contact -- using fallback unit vector", flush=True)
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    z_goal = F.normalize(z_goal, dim=-1)
    return z_goal, "fallback_unit_vector"


# ---------------------------------------------------------------------------
# Phase 2: warmup state (unchanged from lineage)
# ---------------------------------------------------------------------------

def _get_warmup_state(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None
    warmup_actions: List[int] = []

    for _ in range(n_warmup_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        warmup_actions.append(action_idx)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        agent.record_executed_action(action)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            latent = None
            warmup_actions = []

    if latent is None:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)

    return latent.z_self.detach(), latent.z_world.detach(), warmup_actions


# ---------------------------------------------------------------------------
# Phase 3: candidate sequences (unchanged from lineage)
# ---------------------------------------------------------------------------

def _generate_candidate_sequences(
    n_sequences: int, horizon: int, n_actions: int, seed: int,
) -> List[List[int]]:
    torch.manual_seed(seed + 500)
    random.seed(seed + 500)
    seqs = []
    for _ in range(n_sequences):
        seq = [random.randint(0, n_actions - 1) for _ in range(horizon)]
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Phase 4 (hybrid E1/E2 readout, NON-GATING beyond h=1): score sequences at
# every horizon checkpoint. Unchanged from V3-EXQ-965/976.
# ---------------------------------------------------------------------------

def _score_sequence_hybrid_multi_horizon(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
    checkpoints: List[int],
    e1_call_counter: Dict[str, int],
) -> Dict[int, Tuple[float, torch.Tensor]]:
    device = agent.device
    n_actions = agent.config.e2.action_dim
    checkpoint_set = set(checkpoints)

    agent.e1.reset_hidden_state()

    z_self_curr = z_self_start.clone()
    z_world_curr = z_world_start.clone()
    out: Dict[int, Tuple[float, torch.Tensor]] = {}

    for step_idx, a_idx in enumerate(action_sequence, start=1):
        action = _action_to_onehot(a_idx, n_actions, device)
        total_curr = torch.cat([z_self_curr, z_world_curr], dim=-1)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1, actions=action)
        e1_call_counter["n_e1_calls"] += 1
        e1_call_counter["n_e1_calls_nonzero_action"] += (
            1 if float(action.abs().sum()) > 0.0 else 0
        )
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        with torch.no_grad():
            z_self_next = agent.e2.predict_next_self(z_self_curr, action)
        z_self_curr = z_self_next
        z_world_curr = z_world_next

        if step_idx in checkpoint_set:
            score = float(goal_state.goal_proximity(z_world_curr).item())
            out[step_idx] = (score, z_world_curr.detach().clone())

    return out


# ---------------------------------------------------------------------------
# Phase 4-e1-alone (V3-EXQ-980 sibling readout, NEVER GATING): E1-alone
# rollout readout at every checkpoint, including h=30. Reused verbatim from
# 980's design per this entry's own instructions ("keep 980's E1-alone
# readout beside the hybrid at h=30... already written, ~zero cost").
# ---------------------------------------------------------------------------

def _score_sequence_e1_alone_multi_horizon(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
    checkpoints: List[int],
    e1_call_counter: Dict[str, int],
) -> Dict[int, Tuple[float, torch.Tensor]]:
    device = agent.device
    n_actions = agent.config.e2.action_dim
    max_h = max(checkpoints)
    checkpoint_set = set(checkpoints)

    agent.e1.reset_hidden_state()
    total_0 = torch.cat([z_self_start, z_world_start], dim=-1)
    action_tensors = [
        _action_to_onehot(a_idx, n_actions, device) for a_idx in action_sequence[:max_h]
    ]
    action_seq_tensor = torch.stack(action_tensors, dim=1)  # [1, max_h, action_dim]

    with torch.no_grad():
        preds = agent.e1.predict_long_horizon(total_0, horizon=max_h, actions=action_seq_tensor)
    e1_call_counter["n_e1_calls"] += 1
    e1_call_counter["n_e1_calls_nonzero_action"] += 1

    out: Dict[int, Tuple[float, torch.Tensor]] = {}
    for step_idx in range(1, max_h + 1):
        if step_idx not in checkpoint_set:
            continue
        z_world_h = preds[0, step_idx - 1, self_dim:].unsqueeze(0)
        score = float(goal_state.goal_proximity(z_world_h).item())
        out[step_idx] = (score, z_world_h.detach().clone())

    return out


# ---------------------------------------------------------------------------
# Phase 4b: real z_world sample at every horizon checkpoint. Unchanged from
# lineage.
# ---------------------------------------------------------------------------

def _collect_real_zworld_sample_multi_horizon(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int,
    checkpoints: List[int],
) -> Dict[int, List[torch.Tensor]]:
    torch.manual_seed(seed + 3000)
    random.seed(seed + 3000)
    max_h = max(checkpoints)
    checkpoint_set = set(checkpoints)
    samples_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for _ in range(n_samples):
        _, obs_dict = env.reset()
        agent.reset()
        for step_idx in range(1, max_h + 1):
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent.record_executed_action(action)
            _, _, done, _, obs_dict = env.step(action)
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            if step_idx in checkpoint_set:
                samples_by_h[step_idx].append(latent.z_world.detach())
            if done:
                break

    return samples_by_h


def _contrast_ratio(vectors: List[torch.Tensor]) -> Dict[str, float]:
    """CR = spread / ||centroid||. Unchanged from lineage."""
    stacked = torch.cat(vectors, dim=0)
    centroid = stacked.mean(dim=0, keepdim=True)
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr, "n": len(vectors)}


# ---------------------------------------------------------------------------
# Phase 4c (positive-control cross-reference): one-step per-action
# divergence probe, DIRECT E1 channel. Unchanged from lineage.
# ---------------------------------------------------------------------------

def _one_step_action_divergence(
    agent: REEAgent,
    z_self_0: torch.Tensor,
    z_world_0: torch.Tensor,
    self_dim: int,
    e1_call_counter: Dict[str, int],
) -> Dict[str, Any]:
    device = agent.device
    n_actions = agent.config.e2.action_dim
    total_curr = torch.cat([z_self_0, z_world_0], dim=-1)

    predictions: List[torch.Tensor] = []
    n_direct_supply_ok = 0
    for a_idx in range(n_actions):
        agent.e1.reset_hidden_state()
        action = _action_to_onehot(a_idx, n_actions, device)
        is_direct_supply_ok = (
            action is not None
            and float(action.abs().sum().item()) == 1.0
            and int((action != 0).sum().item()) == 1
        )
        n_direct_supply_ok += 1 if is_direct_supply_ok else 0
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1, actions=action)
        e1_call_counter["n_e1_calls"] += 1
        e1_call_counter["n_e1_calls_nonzero_action"] += 1 if is_direct_supply_ok else 0
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        predictions.append(z_world_next.detach())

    pairwise_dists = [
        float((predictions[i] - predictions[j]).norm().item())
        for i, j in itertools.combinations(range(len(predictions)), 2)
    ]
    cr = _contrast_ratio(predictions)

    return {
        "n_actions": n_actions,
        "pairwise_dists": pairwise_dists,
        "pairwise_dist_mean": float(sum(pairwise_dists) / len(pairwise_dists)) if pairwise_dists else 0.0,
        "pairwise_dist_min": float(min(pairwise_dists)) if pairwise_dists else 0.0,
        "pairwise_dist_max": float(max(pairwise_dists)) if pairwise_dists else 0.0,
        "contrast_ratio": cr,
        "n_direct_supply_ok": n_direct_supply_ok,
        "direct_action_supply_fraction": (n_direct_supply_ok / n_actions) if n_actions else 0.0,
    }


# ---------------------------------------------------------------------------
# Single-cell (seed, arm) runner
# ---------------------------------------------------------------------------

def run_cell(
    seed: int,
    arm: str,
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    checkpoints: List[int],
    dry_run: bool = False,
) -> Dict[str, Any]:
    print(f"\n[EXQ-1000] seed={seed} arm={arm}", flush=True)
    print(f"Seed {seed} Condition {arm}", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim, arm)
    e1_call_counter = {"n_e1_calls": 0, "n_e1_calls_nonzero_action": 0}

    print(f"[EXQ-1000] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    print(f"[EXQ-1000] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    train_stats = _train_agent(agent, env, seed, n_train_episodes, steps_per_episode, e1_call_counter, arm)

    print("[EXQ-1000] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    print("[EXQ-1000] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    print(f"[EXQ-1000] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    print(f"[EXQ-1000] Phase 4: scoring sequences (hybrid) at horizons {checkpoints}...", flush=True)
    scores_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}
    scores_e1alone_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_e1alone_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for i, seq in enumerate(seqs):
        per_h = _score_sequence_hybrid_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints,
            e1_call_counter,
        )
        for h, (score, endpoint) in per_h.items():
            scores_by_h[h].append(score)
            endpoints_by_h[h].append(endpoint)

        per_h_e1alone = _score_sequence_e1_alone_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints,
            e1_call_counter,
        )
        for h, (score, endpoint) in per_h_e1alone.items():
            scores_e1alone_by_h[h].append(score)
            endpoints_e1alone_by_h[h].append(endpoint)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    e1coe_score_var_by_h: Dict[int, float] = {}
    cr_rollout_by_h: Dict[int, Dict[str, float]] = {}
    cr_rollout_e1alone_by_h: Dict[int, Dict[str, float]] = {}
    for h in checkpoints:
        scores_t = torch.tensor(scores_by_h[h])
        e1coe_score_var_by_h[h] = float(scores_t.var().item()) if len(scores_by_h[h]) > 1 else 0.0
        cr_rollout_by_h[h] = _contrast_ratio(endpoints_by_h[h])
        cr_rollout_e1alone_by_h[h] = _contrast_ratio(endpoints_e1alone_by_h[h])
        print(
            f"  h={h:>2d}: e1coe_score_var={e1coe_score_var_by_h[h]:.6e} "
            f"CR_rollout(hybrid)={cr_rollout_by_h[h]['contrast_ratio']:.6e} "
            f"CR_rollout(e1_alone)={cr_rollout_e1alone_by_h[h]['contrast_ratio']:.6e}",
            flush=True,
        )

    print(f"[EXQ-1000] Phase 4b: sampling {n_real_samples} real trajectories at horizons {checkpoints}...", flush=True)
    real_samples_by_h = _collect_real_zworld_sample_multi_horizon(
        agent, env, seed, n_real_samples, checkpoints,
    )
    cr_real_by_h: Dict[int, Dict[str, float]] = {}
    cr_ratio_by_h: Dict[int, float] = {}
    cr_ratio_e1alone_by_h: Dict[int, float] = {}
    for h in checkpoints:
        samples = real_samples_by_h[h]
        if len(samples) >= 2:
            cr_real_by_h[h] = _contrast_ratio(samples)
        else:
            cr_real_by_h[h] = {"spread": 0.0, "centroid_norm": 0.0, "contrast_ratio": float("nan"), "n": len(samples)}
        cr_real = cr_real_by_h[h]["contrast_ratio"]
        cr_roll = cr_rollout_by_h[h]["contrast_ratio"]
        cr_roll_e1alone = cr_rollout_e1alone_by_h[h]["contrast_ratio"]
        cr_ratio_by_h[h] = (cr_roll / cr_real) if (cr_real == cr_real and cr_real > 0) else float("nan")
        cr_ratio_e1alone_by_h[h] = (cr_roll_e1alone / cr_real) if (cr_real == cr_real and cr_real > 0) else float("nan")
        print(
            f"  h={h:>2d}: CR_real={cr_real:.6e} (n={cr_real_by_h[h]['n']}) "
            f"ratio(hybrid)={cr_ratio_by_h[h]:.6e} ratio(e1_alone)={cr_ratio_e1alone_by_h[h]:.6e}",
            flush=True,
        )

    print("[EXQ-1000] Phase 4c: direct-channel one-step per-action divergence probe...", flush=True)
    action_probe = _one_step_action_divergence(agent, z_self_0, z_world_0, self_dim, e1_call_counter)
    cr_real_h1 = cr_real_by_h.get(1, {}).get("contrast_ratio", float("nan"))
    action_cr = action_probe["contrast_ratio"]["contrast_ratio"]
    ratio_action_vs_real_h1 = (
        (action_cr / cr_real_h1) if (cr_real_h1 == cr_real_h1 and cr_real_h1 > 0) else float("nan")
    )
    print(
        f"  K={action_probe['n_actions']} pairwise_dist mean={action_probe['pairwise_dist_mean']:.6e} "
        f"cr_action_h1={action_cr:.6e} ratio_vs_CR_real(h=1)={ratio_action_vs_real_h1:.6e}",
        flush=True,
    )

    missing_action_calls = float(
        getattr(agent.e1, "_action_cond_missing_calls", 0)
    )
    buffer_stats = agent.e1_action_buffer_stats()
    direct_supply_fraction = (
        (e1_call_counter["n_e1_calls_nonzero_action"] / e1_call_counter["n_e1_calls"])
        if e1_call_counter["n_e1_calls"] else 0.0
    )
    print(
        f"  [vacuity] missing_action_calls={missing_action_calls:.0f} "
        f"direct_action_supply_fraction={direct_supply_fraction:.4f} "
        f"(internal_buffer_nonzero_fraction={buffer_stats.get('nonzero_fraction', 0.0):.4f})",
        flush=True,
    )

    verdict = "PASS" if readiness_report.get("zworld_encoder_trained") else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "arm": arm,
        "readiness": readiness_report,
        "goal_template_source": goal_template_source,
        "z_goal_norm": goal_state.goal_norm(),
        "base_prox": base_prox,
        "checkpoints": checkpoints,
        "e1coe_score_var_by_h": e1coe_score_var_by_h,
        "cr_rollout_by_h": cr_rollout_by_h,
        "cr_rollout_e1alone_by_h": cr_rollout_e1alone_by_h,
        "cr_real_by_h": cr_real_by_h,
        "cr_ratio_by_h": cr_ratio_by_h,
        "cr_ratio_e1alone_by_h": cr_ratio_e1alone_by_h,
        "action_probe": action_probe,
        "ratio_action_vs_real_h1": ratio_action_vs_real_h1,
        "action_cr": action_cr,
        "cr_real_h1": cr_real_h1,
        "missing_action_calls": missing_action_calls,
        "e1_action_buffer_stats": buffer_stats,
        "direct_action_supply_fraction": direct_supply_fraction,
        "n_e1_calls_total": e1_call_counter["n_e1_calls"],
        "train_stats": train_stats,
        "n_e1_grad_steps": int(train_stats["n_e1_grad_steps"]),
        "rsd_n_engaged": int(train_stats["rsd_n_engaged"]),
        "rsd_n_degenerate": int(train_stats["rsd_n_degenerate"]),
        "rsd_n_buffer_starved": int(train_stats["rsd_n_buffer_starved"]),
        "rsd_engaged_frac": float(train_stats["rsd_engaged_frac"]),
        "rsd_action_sensitivity_ratio": float(train_stats["rsd_action_sensitivity_ratio"]),
        "rsd_n_sensitivity_checks": int(train_stats["rsd_n_sensitivity_checks"]),
    }, agent


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    seeds: List[int],
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    checkpoints = sorted(set(
        [h for h in HORIZON_CHECKPOINTS_FULL if h <= rollout_horizon] + [rollout_horizon]
    ))

    cell_config_slice = {
        "world_dim": world_dim, "self_dim": self_dim,
        "n_train_episodes": n_train_episodes, "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences, "rollout_horizon": rollout_horizon,
        "n_warmup_steps": n_warmup_steps, "goal_max_steps": goal_max_steps,
        "zworld_p0_episodes": zworld_p0_episodes, "n_real_samples": n_real_samples,
        "env_kwargs": _env_kwargs(),
        "rc_horizon": RC_HORIZON, "rc_decay": RC_DECAY, "rc_weight": RC_WEIGHT,
        "rsd_horizon": RSD_HORIZON, "rsd_temperature": RSD_TEMPERATURE,
        "rsd_min_batch_classes": RSD_MIN_BATCH_CLASSES, "rsd_weight": RSD_WEIGHT,
        "rsd_batch_k": RSD_BATCH_K, "rsd_buffer_max": RSD_BUFFER_MAX,
        "train_window_h": TRAIN_WINDOW_H,
        "cr_real_floor": CR_REAL_FLOOR, "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "c3_var_threshold": C3_VAR_THRESHOLD, "zworld_p0_episodes_default": ZWORLD_P0_EPISODES,
        "n_real_samples_default": N_REAL_SAMPLES,
        "min_real_samples_per_horizon": MIN_REAL_SAMPLES_PER_HORIZON,
        "horizon_checkpoints_full": list(HORIZON_CHECKPOINTS_FULL),
        "lift_bar": LIFT_BAR,
        "alpha_world": 0.9, "alpha_self": 0.3, "unified_latent_mode": False,
        "train_lr_e1": 1e-3, "train_lr_e2": 1e-3,
    }

    arm_results: List[Dict[str, Any]] = []
    agents_for_manifest = []
    for arm in ARM_ORDER:
        for seed in seeds:
            with arm_cell(
                seed,
                config_slice={**cell_config_slice, "arm": arm, **ARM_CONFIGS[arm]},
                script_path=Path(__file__),
                config_slice_declared=True,
                # MINT AS YOU GO: ARM_OFF is the lineage's OFF baseline; emit
                # it cross-driver reusable so a later letter can cite it.
                include_driver_script_in_hash=(arm != OFF_ARM),
            ) as cell:
                row, agent = run_cell(
                    seed=seed, arm=arm,
                    world_dim=world_dim, self_dim=self_dim,
                    n_train_episodes=n_train_episodes, steps_per_episode=steps_per_episode,
                    n_sequences=n_sequences, rollout_horizon=rollout_horizon,
                    n_warmup_steps=n_warmup_steps, goal_max_steps=goal_max_steps,
                    zworld_p0_episodes=zworld_p0_episodes, n_real_samples=n_real_samples,
                    checkpoints=checkpoints, dry_run=dry_run,
                )
                cell.stamp(row)
            arm_results.append(row)
            agents_for_manifest.append(agent)

    by_arm_seed: Dict[Tuple[str, int], Dict[str, Any]] = {
        (r["arm"], r["seed"]): r for r in arm_results
    }

    # ---- Readiness (P0) ----
    encoder_trained_per_cell = [bool(r["readiness"].get("zworld_encoder_trained")) for r in arm_results]
    p_encoder_trained_met = all(encoder_trained_per_cell)
    min_encoder_delta = min(
        r["readiness"].get("world_encoder_max_abs_delta", 0.0) for r in arm_results
    )

    p_real_nondegenerate_met = True
    min_real_samples_h1 = min(
        r["cr_real_by_h"].get(1, {}).get("n", 0) for r in arm_results
    )
    for r in arm_results:
        cr1 = r["cr_real_by_h"].get(1, {})
        ok = (
            cr1.get("contrast_ratio", float("nan")) == cr1.get("contrast_ratio", float("nan"))
            and cr1.get("contrast_ratio", 0.0) > CR_REAL_FLOOR
            and cr1.get("n", 0) >= MIN_REAL_SAMPLES_PER_HORIZON
        )
        p_real_nondegenerate_met = p_real_nondegenerate_met and ok

    max_missing_action_calls = max((r["missing_action_calls"] for r in arm_results), default=0.0)
    p_no_missing_action_calls = max_missing_action_calls == 0.0

    min_direct_supply_fraction = min(
        (r["direct_action_supply_fraction"] for r in arm_results), default=0.0
    )
    p_direct_action_supply = min_direct_supply_fraction >= 0.999

    cr_ratio_h1_values = [
        by_arm_seed[(arm, seed)]["cr_ratio_by_h"].get(1, float("nan"))
        for arm in ARM_ORDER for seed in seeds
    ]
    p_cr_ratio_h1_finite = all(v == v and v > 0 for v in cr_ratio_h1_values)
    min_cr_ratio_h1 = min((v for v in cr_ratio_h1_values if v == v), default=float("nan"))

    # (a) matched E1 gradient-step counts across arms, per seed (true by
    # construction under the uniform trailing-window trigger)
    grad_step_gap_per_seed: Dict[str, int] = {}
    for seed in seeds:
        counts = [int(by_arm_seed[(arm, seed)]["n_e1_grad_steps"]) for arm in ARM_ORDER]
        grad_step_gap_per_seed[f"seed{seed}"] = int(max(counts) - min(counts))
    max_grad_step_gap = max(grad_step_gap_per_seed.values()) if grad_step_gap_per_seed else 0
    p_grad_steps_matched = max_grad_step_gap == 0

    # (b) ARM_RSD's own non-vacuity: engaged on a nonzero fraction of ticks,
    # on every seed. Per the entry's own instruction: below-floor here means
    # the RSD arm was never genuinely tested -> substrate_not_ready_requeue.
    rsd_engaged_frac_per_seed = {
        f"seed{seed}": float(by_arm_seed[("ARM_RSD", seed)]["rsd_engaged_frac"])
        for seed in seeds
    }
    min_rsd_engaged_frac = min(rsd_engaged_frac_per_seed.values()) if rsd_engaged_frac_per_seed else 0.0
    p_rsd_objective_engaged = min_rsd_engaged_frac > 0.0

    # rsd_action_sensitive (red-team finding #1/#2, 2026-09-03): rsd_objective_
    # engaged only asserts the SAMPLED BATCH was non-degenerate (>=2 distinct
    # full action sequences among the K windows); it says nothing about
    # whether E1's prediction actually DEPENDS on those sequences. Under the
    # replay-buffer substitution (K windows from different times along one
    # continuous trajectory rather than literal siblings of one state), the
    # loss can be minimised by an "identity shortcut" -- predict endpoint ~=
    # init -- with zero action sensitivity, in a slow-drift env where
    # temporally-close windows are spatially close regardless of action. That
    # failure mode satisfies rsd_objective_engaged while never exercising the
    # per-action divergence this whole experiment measures. This precondition
    # reads the action-sensitivity control ratio (mean loss under a row-
    # permuted action assignment / mean loss under the real assignment,
    # accumulated over every RSD training tick this cell ran): a ratio near
    # 1.0 means shuffling which action sequence is attached to which init/
    # endpoint pair did not change the loss, i.e. the model is not reading the
    # action sequence at all.
    rsd_sensitivity_ratio_per_seed = {
        f"seed{seed}": float(by_arm_seed[("ARM_RSD", seed)]["rsd_action_sensitivity_ratio"])
        for seed in seeds
    }
    _finite_sensitivity_ratios = [v for v in rsd_sensitivity_ratio_per_seed.values() if v == v]
    min_rsd_sensitivity_ratio = (
        min(_finite_sensitivity_ratios) if _finite_sensitivity_ratios else float("nan")
    )
    p_rsd_action_sensitive = (
        len(_finite_sensitivity_ratios) == len(seeds)
        and min_rsd_sensitivity_ratio == min_rsd_sensitivity_ratio  # not NaN
        and min_rsd_sensitivity_ratio >= RSD_ACTION_SENSITIVITY_RATIO_FLOOR
    )

    preconditions = [
        {
            "name": "encoder_trained",
            "kind": "readiness",
            "description": (
                "At least one split_encoder.world_encoder tensor moved during "
                "the Phase 0a SD-070 warmup, per every (seed, arm) cell."
            ),
            "measured": min_encoder_delta,
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "met": p_encoder_trained_met,
        },
        {
            "name": "real_zworld_nondegenerate_h1",
            "kind": "readiness",
            "description": (
                "CR_real(h=1) is finite, positive, and backed by at least "
                f"{MIN_REAL_SAMPLES_PER_HORIZON} surviving real samples, every cell."
            ),
            "measured": float(min_real_samples_h1),
            "threshold": float(MIN_REAL_SAMPLES_PER_HORIZON),
            "direction": "lower",
            "met": p_real_nondegenerate_met,
        },
        {
            "name": "no_missing_action_calls",
            "kind": "readiness",
            "description": (
                "E1DeepPredictor._action_cond_missing_calls is 0 on every cell -- "
                "every actions= call this script made (incl. the ARM_RC_DECAY / "
                "ARM_RSD training calls) supplied a real action sequence, never a "
                "silent zero-fallback."
            ),
            "measured": max_missing_action_calls,
            "threshold": 0.0,
            "direction": "upper",
            "comparator": "<=",
            "met": p_no_missing_action_calls,
        },
        {
            "name": "direct_action_supply_fraction",
            "kind": "readiness",
            "description": (
                "The fraction of E1 calls that received a genuine non-None "
                "one-hot actions= argument, minimum across cells."
            ),
            "measured": min_direct_supply_fraction,
            "threshold": 0.999,
            "direction": "lower",
            "met": p_direct_action_supply,
        },
        {
            "name": "cr_ratio_h1_finite",
            "kind": "readiness",
            "description": (
                "cr_ratio(h=1) is finite and positive on every (arm, seed) cell -- "
                "the exact statistic the decision rule routes on."
            ),
            "measured": min_cr_ratio_h1,
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "met": p_cr_ratio_h1_finite,
        },
        {
            "name": "e1_grad_steps_matched",
            "kind": "readiness",
            "description": (
                "Every arm took the same number of E1 optimiser steps at a seed. "
                "TRUE BY CONSTRUCTION under the uniform trailing-window trigger "
                "(see module docstring); measured, not assumed."
            ),
            "measured": float(max_grad_step_gap),
            "threshold": 0.0,
            "direction": "upper",
            "comparator": "<=",
            "control": "identical random policy, env, seeds and trailing-window schedule in every arm",
            "met": p_grad_steps_matched,
        },
        {
            "name": "rsd_objective_engaged",
            "kind": "readiness",
            "description": (
                "ARM_RSD's rollout_sequence_divergence_loss training calls were "
                "non-degenerate (n_distinct_full_sequences >= "
                f"{RSD_MIN_BATCH_CLASSES}) on a nonzero fraction of ticks, on "
                "every seed. Below floor means ARM_RSD's objective NEVER "
                "meaningfully engaged -- every call fell through to the "
                "grad-connected zero-loss branch. This gates the whole run: "
                "an arm whose loss never engages self-routes to "
                "substrate_not_ready_requeue, never a claim verdict."
            ),
            "measured": min_rsd_engaged_frac,
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "control": "K sampled real windows per tick, distinct-full-sequence floor replicated pre-call",
            "per_seed": rsd_engaged_frac_per_seed,
            "met": p_rsd_objective_engaged,
        },
        {
            "name": "rsd_action_sensitive",
            "kind": "readiness",
            "description": (
                "ARM_RSD's rollout_sequence_divergence_loss actually DEPENDS on "
                "the sampled action sequences (mean loss under a row-permuted "
                "action assignment / mean loss under the real assignment >= "
                f"{RSD_ACTION_SENSITIVITY_RATIO_FLOOR}, on every seed). "
                "rsd_objective_engaged alone can read True on a degenerate "
                "'identity shortcut' where the replay-buffer's per-sample "
                "distinct initial states let E1 minimise the loss via endpoint "
                "~= init (temporally-close windows are spatially close in a "
                "slow-drift env regardless of action) without ever reading the "
                "action sequence -- red-team finding #1/#2, 2026-09-03. Below "
                "floor means the loss engaged on a basis unrelated to "
                "per-action divergence: self-routes substrate_not_ready_"
                "requeue, never a claim verdict."
            ),
            "measured": min_rsd_sensitivity_ratio,
            "threshold": RSD_ACTION_SENSITIVITY_RATIO_FLOOR,
            "direction": "lower",
            "comparator": ">=",
            "control": (
                "row-permuted action-sequence assignment (init/endpoint pairing "
                "unchanged), isolated RNG stream, computed under no_grad every "
                "RSD training tick"
            ),
            "per_seed": rsd_sensitivity_ratio_per_seed,
            "met": p_rsd_action_sensitive,
        },
    ]

    non_degenerate = bool(
        p_encoder_trained_met and p_real_nondegenerate_met
        and p_no_missing_action_calls and p_direct_action_supply
        and p_cr_ratio_h1_finite and p_grad_steps_matched and p_rsd_objective_engaged
        and p_rsd_action_sensitive
    )

    majority_seeds = len(seeds) // 2 + 1
    evaluator_bar_reached_cells: List[str] = []
    for arm in ARM_ORDER:
        for seed in seeds:
            r = by_arm_seed[(arm, seed)]
            cr1 = r["cr_ratio_by_h"].get(1, float("nan"))
            var1 = r["e1coe_score_var_by_h"].get(1, 0.0)
            if cr1 == cr1 and cr1 >= CR_ROLLOUT_COLLAPSE_RATIO and var1 >= C3_VAR_THRESHOLD:
                evaluator_bar_reached_cells.append(f"{arm}_seed{seed}")

    def _arm_verdict_at(
        on_arm: str, h: int, bar: float, off_arm: str = OFF_ARM,
    ) -> Tuple[str, Dict[str, Any], int, int, int]:
        per_seed: Dict[str, Any] = {}
        n_exceeds = 0
        n_below = 0
        n_positive = 0
        for seed in seeds:
            off_row = by_arm_seed[(off_arm, seed)]
            on_row = by_arm_seed[(on_arm, seed)]
            off_cr = off_row["cr_ratio_by_h"].get(h, float("nan"))
            on_cr = on_row["cr_ratio_by_h"].get(h, float("nan"))
            rel = (on_cr / off_cr) if (off_cr == off_cr and off_cr > 0) else float("nan")
            exceeds = (rel == rel) and (rel >= bar)
            below = (rel == rel) and (rel > 0) and ((1.0 / rel) >= bar)
            positive = (rel == rel) and (rel > 1.0)
            n_exceeds += 1 if exceeds else 0
            n_below += 1 if below else 0
            n_positive += 1 if positive else 0
            var_off = off_row["e1coe_score_var_by_h"].get(h, 0.0)
            var_on = on_row["e1coe_score_var_by_h"].get(h, 0.0)
            per_seed[f"seed{seed}"] = {
                "seed": seed, "h": h,
                "cr_ratio_off_arm": off_cr, "cr_ratio_on_arm": on_cr,
                "relative_lift_on_over_off": rel,
                "e1coe_score_var_off_arm": var_off, "e1coe_score_var_on_arm": var_on,
                "on_materially_exceeds": exceeds, "on_materially_below": below,
                "on_direction_positive": positive,
            }
        if n_exceeds >= majority_seeds:
            verdict = "lifts"
        elif n_below >= majority_seeds:
            verdict = "degrades"
        elif n_exceeds == 0 and n_below == 0:
            verdict = "null"
        else:
            verdict = "mixed"
        return verdict, per_seed, n_exceeds, n_below, n_positive

    def _clears_absolute_bars(arm: str) -> Tuple[bool, int]:
        n = 0
        for seed in seeds:
            r = by_arm_seed[(arm, seed)]
            cr1 = r["cr_ratio_by_h"].get(1, float("nan"))
            var1 = r["e1coe_score_var_by_h"].get(1, 0.0)
            if cr1 == cr1 and cr1 >= CR_ROLLOUT_COLLAPSE_RATIO and var1 >= C3_VAR_THRESHOLD:
                n += 1
        return (n >= majority_seeds), n

    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        unmet_names = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = "P0 readiness unmet: " + ", ".join(unmet_names)
        status = "FAIL"
        evidence_direction = "non_contributory"
        per_seed_lift: Dict[str, Dict[str, Any]] = {}
        per_seed_lift_trained_horizon: Dict[str, Dict[str, Any]] = {}
        arm_verdicts: Dict[str, Dict[str, Any]] = {}
        arm_verdicts_trained_horizon: Dict[str, Dict[str, Any]] = {}
        criteria = []
        criteria_non_degenerate = {"C1_rsd_relative_lift": False, "C2_rsd_absolute_bars": False}
        rsd_clears_bars = False
        rsd_clears_bars_n = 0
    else:
        degeneracy_reason = None
        per_seed_lift = {}
        per_seed_lift_trained_horizon = {}
        arm_verdicts = {}
        arm_verdicts_trained_horizon = {}

        for on_arm in ON_ARMS:
            v1, ps1, ne1, nb1, npos1 = _arm_verdict_at(on_arm, 1, LIFT_BAR)
            per_seed_lift[on_arm] = ps1
            arm_verdicts[on_arm] = {
                "verdict": v1, "h": 1, "bar": LIFT_BAR,
                "n_seeds_exceeds": ne1, "n_seeds_below": nb1,
                "n_seeds_direction_positive": npos1,
                "sign_test_p_all_positive": (0.5 ** len(seeds)) if npos1 == len(seeds) else None,
                "n_seeds": len(seeds), "majority_seeds": majority_seeds,
            }
            if on_arm in ("ARM_RC_DECAY", "ARM_RSD"):
                trained_h = RC_HORIZON if on_arm == "ARM_RC_DECAY" else RSD_HORIZON
                vH, psH, neH, nbH, nposH = _arm_verdict_at(on_arm, trained_h, LIFT_BAR)
                per_seed_lift_trained_horizon[on_arm] = psH
                arm_verdicts_trained_horizon[on_arm] = {
                    "verdict": vH, "h": trained_h, "bar": LIFT_BAR,
                    "n_seeds_exceeds": neH, "n_seeds_below": nbH,
                    "n_seeds_direction_positive": nposH,
                    "sign_test_p_all_positive": (0.5 ** len(seeds)) if nposH == len(seeds) else None,
                    "n_seeds": len(seeds), "majority_seeds": majority_seeds,
                }

        v_anchor = arm_verdicts["ARM_ANCHOR"]["verdict"]
        v_rc_decay = arm_verdicts["ARM_RC_DECAY"]["verdict"]
        v_rsd = arm_verdicts["ARM_RSD"]["verdict"]

        rsd_clears_bars, rsd_clears_bars_n = _clears_absolute_bars("ARM_RSD")

        if rsd_clears_bars:
            label = "rollout_endpoint_contrastive_clears_evaluator_bars"
        elif v_rsd == "lifts":
            label = "rollout_endpoint_contrastive_lifts_cr_ratio_h1"
        elif v_anchor == "null" and v_rc_decay == "null" and v_rsd == "null":
            label = "all_arms_null_residual_crush_locus_elsewhere"
        elif v_rsd == "null" and (v_anchor == "lifts" or v_rc_decay == "lifts"):
            label = "rollout_endpoint_contrastive_null_others_lift"
        elif v_rsd == "degrades":
            label = "rollout_endpoint_contrastive_degrades_cr_ratio_h1"
        else:
            label = "mixed_across_arms"

        h1_label_before_override = label
        rsd_trained_h_verdict = arm_verdicts_trained_horizon.get("ARM_RSD", {}).get("verdict")
        rc_trained_h_verdict = arm_verdicts_trained_horizon.get("ARM_RC_DECAY", {}).get("verdict")
        any_lift_at_trained_h = (rsd_trained_h_verdict == "lifts") or (rc_trained_h_verdict == "lifts")
        # Red-team finding #4a (2026-09-03): "degrades" is DELIBERATELY
        # EXCLUDED from the override-eligible label set below. A confirmed
        # degradation on the primary DV (h=1) is a stronger, more specific
        # finding than "h=1 is the wrong readout" -- relabelling it via the
        # trained-horizon override would let a genuine negative result at h=1
        # be reported as partial support merely because RSD also lifts at its
        # own trained horizon (h=5), which is a materially different claim.
        # A degrades verdict at h=1 is reported as degrades regardless of what
        # happens at h=5; the h=5 reading is still fully visible in
        # arm_verdicts_trained_horizon for anyone reading the manifest.
        if label in (
            "all_arms_null_residual_crush_locus_elsewhere",
            "rollout_endpoint_contrastive_null_others_lift",
            "mixed_across_arms",
        ) and rsd_trained_h_verdict == "lifts":
            label = "rollout_endpoint_contrastive_lift_at_trained_horizon_only"

        status = "PASS"  # diagnostic discrimination -- informative in every direction
        evidence_direction = "non_contributory"
        criteria = [
            {
                "name": "C1_rsd_relative_lift",
                "load_bearing": True,
                "passed": True,  # classifies, does not gate PASS/FAIL
                "arm_verdict": v_rsd,
                "arm_verdict_at_trained_horizon": rsd_trained_h_verdict,
                "measured": arm_verdicts["ARM_RSD"]["n_seeds_exceeds"],
                "threshold": LIFT_BAR,
                "statement": (
                    "cr_ratio(h=1) relative lift of ARM_RSD over ARM_OFF, per seed, "
                    f"against the FIXED pre-registered bar {LIFT_BAR}; verdict on >= "
                    f"{majority_seeds}/{len(seeds)} seeds."
                ),
            },
            {
                "name": "C2_rsd_absolute_bars",
                "load_bearing": True,
                "passed": bool(rsd_clears_bars),
                "measured": rsd_clears_bars_n,
                "threshold": majority_seeds,
                "statement": (
                    f"ARM_RSD clears BOTH absolute evaluator bars (cr_ratio(h=1) >= "
                    f"{CR_ROLLOUT_COLLAPSE_RATIO} AND e1coe_score_var(h=1) >= "
                    f"{C3_VAR_THRESHOLD}) on >= {majority_seeds}/{len(seeds)} seeds."
                ),
            },
            {
                "name": "C3_evaluator_bar_reached_any_arm",
                "load_bearing": False,
                "passed": bool(evaluator_bar_reached_cells),
                "measured": len(evaluator_bar_reached_cells),
                "threshold": 1,
                "statement": (
                    "Any (arm, seed) cell reaches both evaluator bars. RECORDED, not routing."
                ),
            },
        ]
        criteria_non_degenerate = {
            "C1_rsd_relative_lift": non_degenerate,
            "C2_rsd_absolute_bars": non_degenerate,
            "C3_evaluator_bar_reached_any_arm": non_degenerate,
        }

    print(f"\n[EXQ-1000] Label: {label}", flush=True)
    print(f"[EXQ-1000] Status: {status}", flush=True)

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "unblocks_claims": UNBLOCKS_CLAIMS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "evidence_class": "diagnostic_disambiguation",
        "evidence_direction": evidence_direction,
        "seeds": seeds,
        "arms": ARM_ORDER,
        "off_arm": OFF_ARM,
        "on_arms": ON_ARMS,
        "arm_configs": ARM_CONFIGS,
        "world_dim": world_dim,
        "self_dim": self_dim,
        "n_train_episodes": n_train_episodes,
        "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences,
        "rollout_horizon": rollout_horizon,
        "horizon_checkpoints": checkpoints,
        "n_warmup_steps": n_warmup_steps,
        "zworld_p0_episodes": zworld_p0_episodes,
        "n_real_samples": n_real_samples,
        "rc_horizon": RC_HORIZON, "rc_decay": RC_DECAY, "rc_weight": RC_WEIGHT,
        "rsd_horizon": RSD_HORIZON, "rsd_temperature": RSD_TEMPERATURE,
        "rsd_min_batch_classes": RSD_MIN_BATCH_CLASSES, "rsd_weight": RSD_WEIGHT,
        "rsd_batch_k": RSD_BATCH_K,
        "registered_cr_real_floor": CR_REAL_FLOOR,
        "registered_lift_bar": LIFT_BAR,
        "registered_majority_seeds": majority_seeds,
        "registered_cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "registered_c3_var_threshold": C3_VAR_THRESHOLD,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_seed_lift": per_seed_lift,
        "per_seed_lift_trained_horizon": per_seed_lift_trained_horizon,
        "arm_verdicts": arm_verdicts,
        "arm_verdicts_trained_horizon": arm_verdicts_trained_horizon,
        "evaluator_bar_reached_cells": evaluator_bar_reached_cells,
        "e1_grad_step_gap_per_seed": grad_step_gap_per_seed,
        "rsd_engaged_frac_per_seed": rsd_engaged_frac_per_seed,
        "rsd_sensitivity_ratio_per_seed": rsd_sensitivity_ratio_per_seed,
        "status": status,
        "outcome": status,
        "verdict": status,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "combination_rule": (
                "ARM_RSD is compared against ARM_OFF by relative lift of cr_ratio(h=1) "
                f"against the FIXED pre-registered LIFT_BAR ({LIFT_BAR}); arm verdict "
                f"lifts/degrades needs >= MAJORITY_SEEDS ({majority_seeds}/{len(seeds)}) "
                "seeds in that direction, null = neither direction on any seed, mixed "
                "otherwise. Separately, ARM_RSD is checked against the entry's own "
                "ABSOLUTE evaluator bars (cr_ratio(h=1)>=0.1 AND e1coe_score_var>=0.002) "
                "on >= MAJORITY_SEEDS seeds -- clearing these routes the label "
                "regardless of the relative-lift verdict (Q1's strongest branch). "
                "See module docstring LABEL COMPOSITION for the full routing table. "
                "A null or mixed/degrades at h=1 with a LIFT at the trained horizon "
                "(RC_HORIZON=5 or RSD_HORIZON=5) becomes "
                "rollout_endpoint_contrastive_lift_at_trained_horizon_only."
            ),
            "dv_symmetry_note": (
                "cr_ratio(h) = cr_rollout(h)/cr_real(h); cr_real(h) is arm-invariant per "
                "seed, so the contrast reduces to cr_rollout(h) = spread/centroid_norm "
                "over the 40 candidate rollout endpoints. A changed training objective "
                "moves E1's learned weights and hence every endpoint non-uniformly "
                "(neither a common rescaling nor a permutation of the candidate index), "
                "so the manipulation reaches the DV and the direction is genuinely open."
            ),
            "e1_alone_readout_note": (
                "cr_ratio_e1alone_by_h (and cr_rollout_e1alone_by_h) are the V3-EXQ-980 "
                "sibling readout at every checkpoint including h=30 -- an E1-alone "
                "predict_long_horizon rollout with no E2 self-slot re-priming. RECORDED "
                "for the MECH-135 30-step consumer question, EXPLICITLY NOT A GATE "
                "(user_decision_2026_09_02 Q2); the label above routes only on the "
                "hybrid readout's cr_ratio_by_h."
            ),
            "what_all_null_licenses": (
                "all_arms_null_residual_crush_locus_elsewhere: ARM_ANCHOR (968's "
                "strongest prior incumbent), ARM_RC_DECAY (ITEM 2), and ARM_RSD (ITEM 3) "
                "all null against ARM_OFF -- contributes a third leg to the "
                "sd_e1_residual_crush_locus ledger per user_decision_2026_09_02 Q3. "
                "Does NOT by itself say where the crush is; that is governance's read."
            ),
        },
        "source_substrate_entry": "SD-e1-rollout-consistency-training (ITEM 3, landed 2026-09-03)",
        "source_substrate_commit_item3": "ree-v3 df551f38fe",
        "source_design_doc": "sd_e1_rollout_consistency_training.md",
        "reference_runs": {
            "v3_exq_965": "v3_exq_965_sd_e1_item1_action_conditioning_validation_20260830T145908Z_v3",
            "v3_exq_968": "v3_exq_968_sd_e1_output_proj_residual_ab_20260901T162647Z_v3",
            "v3_exq_976": "v3_exq_976_sd_e1_item2_rollout_consistency_validation_20260902T114700Z_v3",
            "v3_exq_980": "v3_exq_980_sd_e1_h1c_readout_regime_e1_alone_20260902T212300Z_v3",
        },
        "hypothesis_space_qid": "inv088_evaluator_degeneracy_cause",
    }

    for r in arm_results:
        key = f"{r['arm']}_seed{r['seed']}"
        for k, v in r.items():
            if k not in ("seed", "arm"):
                result[f"cell_{key}_{k}"] = v

    result["arm_results"] = arm_results
    result["_agents_for_manifest"] = agents_for_manifest
    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-1000: SD-e1-rollout-consistency-training ITEM 3 validation -- "
            "rollout-endpoint contrastive vs three incumbents (diagnostic)"
        )
    )
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in SEEDS_DEFAULT))
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--self-dim", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--n-sequences", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--goal-max-steps", type=int, default=2000)
    parser.add_argument("--zworld-p0-episodes", type=int, default=ZWORLD_P0_EPISODES)
    parser.add_argument("--n-real-samples", type=int, default=N_REAL_SAMPLES)
    parser.add_argument("--dry-run", "--smoke-test", dest="dry_run", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.dry_run:
        n_train = 2
        steps_ep = 50
        n_sequences = 5
        horizon = 10
        warmup = 5
        goal_max = 300
        zworld_p0 = 3
        n_real = 15
        seeds = seeds[:2]
        print("[V3-EXQ-1000] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_sequences = args.n_sequences
        horizon = args.rollout_horizon
        warmup = args.warmup_steps
        goal_max = args.goal_max_steps
        zworld_p0 = args.zworld_p0_episodes
        n_real = args.n_real_samples

    t0 = time.perf_counter()
    result = run(
        seeds=seeds,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        n_train_episodes=n_train,
        steps_per_episode=steps_ep,
        n_sequences=n_sequences,
        rollout_horizon=horizon,
        n_warmup_steps=warmup,
        goal_max_steps=goal_max,
        zworld_p0_episodes=zworld_p0,
        n_real_samples=n_real,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    agents_for_manifest = result.pop("_agents_for_manifest", [])

    full_config = {
        "seeds": seeds,
        "arms": ARM_ORDER,
        "arm_configs": ARM_CONFIGS,
        "world_dim": args.world_dim,
        "self_dim": args.self_dim,
        "n_train_episodes": n_train,
        "steps_per_episode": steps_ep,
        "n_sequences": n_sequences,
        "rollout_horizon": horizon,
        "n_warmup_steps": warmup,
        "goal_max_steps": goal_max,
        "zworld_p0_episodes": zworld_p0,
        "n_real_samples": n_real,
        "rc_horizon": RC_HORIZON, "rc_decay": RC_DECAY, "rc_weight": RC_WEIGHT,
        "rsd_horizon": RSD_HORIZON, "rsd_temperature": RSD_TEMPERATURE,
        "rsd_min_batch_classes": RSD_MIN_BATCH_CLASSES, "rsd_weight": RSD_WEIGHT,
        "rsd_batch_k": RSD_BATCH_K, "rsd_buffer_max": RSD_BUFFER_MAX,
        "train_window_h": TRAIN_WINDOW_H,
        "cr_real_floor": CR_REAL_FLOOR,
        "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "c3_var_threshold": C3_VAR_THRESHOLD,
        "lift_bar": LIFT_BAR,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "alpha_world": 0.9,
        "alpha_self": 0.3,
        "unified_latent_mode": False,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        agent=agents_for_manifest,
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Label: {result['interpretation']['label']}", flush=True)

    if args.dry_run:
        print("[V3-EXQ-1000] SMOKE TEST COMPLETE", flush=True)
        for k in ["status", "non_degenerate", "degeneracy_reason"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
        print(f"  label: {result['interpretation']['label']}", flush=True)
        print(f"  arm_verdicts: {result.get('arm_verdicts')}", flush=True)
        print(f"  arm_verdicts_trained_horizon: {result.get('arm_verdicts_trained_horizon')}", flush=True)
        print(f"  e1_grad_step_gap_per_seed: {result.get('e1_grad_step_gap_per_seed')}", flush=True)
        print(f"  rsd_engaged_frac_per_seed: {result.get('rsd_engaged_frac_per_seed')}", flush=True)
        print(f"  rsd_sensitivity_ratio_per_seed: {result.get('rsd_sensitivity_ratio_per_seed')}", flush=True)
        for on_arm, d in (result.get("per_seed_lift") or {}).items():
            for k, v in d.items():
                print(
                    f"  [smoke] {on_arm}/{k}: cr_ratio_h1 off={v['cr_ratio_off_arm']:.4e} "
                    f"on={v['cr_ratio_on_arm']:.4e} lift={v['relative_lift_on_over_off']:.3f}",
                    flush=True,
                )

    # --- runner-conformance sentinel ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
