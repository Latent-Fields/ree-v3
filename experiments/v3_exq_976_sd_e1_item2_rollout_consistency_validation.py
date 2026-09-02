"""
V3-EXQ-976 -- SD-e1-rollout-consistency-training ITEM 2 validation: does
TRAINING E1 with the multi-step rollout-consistency objective (candidate 1,
E1DeepPredictor.rollout_consistency_loss) lift per-action rollout divergence at
the E1 output -- cr_ratio(h=1) -- over the incumbent single-step teacher-forced
objective, on the ITEM-1-ON substrate?

Claims: [] (diagnostic; validates the SUBSTRATE, not MECH-135/INV-088. Both keep
pending_retest_after_substrate: true regardless of outcome here -- their retest is
a SEPARATE, governance-routed successor.)
DIAGNOSTIC. See EXPERIMENT_PURPOSE below.

WHY THIS RUN EXISTS
---------------------------------------------------------------------------
substrate_queue entry SD-e1-rollout-consistency-training, ITEM 2 (candidate 1)
landed 2026-09-01 as E1DeepPredictor.rollout_consistency_loss plus four
default-off E1Config knobs (e1_rollout_consistency_{enabled,weight,horizon,
horizon_weights_decay}); contract tests/contracts/test_e1_rollout_consistency_loss.py
(24 tests). It rolls E1 out autoregressively under an action sequence and
penalises per-step deviation from the OBSERVED latent trajectory, step t weighted
decay**t; at decay=1.0 it is exactly F.mse_loss over the rollout window (the flat
form), so the decay is the only axis it adds over a flat multi-step loss.

THE TRAP THIS SCRIPT IS BUILT AROUND: ENABLING THE FLAG CHANGES NOTHING ON ITS
OWN, BY DESIGN. The knob gates a training OBJECTIVE, not the forward path; the
contract pins the rollout bit-identical with the flag on. Every driver in this
lineage (V3-EXQ-954:312, 965:409, 968:431) trains E1 SINGLE-STEP teacher-forced,
`F.mse_loss(e1_pred[:, 0, :], total_curr.detach())`, so the multi-step objective
has never been exercised in the lineage that motivated the SD. An arm that merely
sets the flag and trains the old way is a VACUOUS ON arm. This driver therefore
CALLS agent.e1.rollout_consistency_loss(...) in its training loop on the ON arms
and, on EVERY training window of EVERY arm, also computes the OTHER arm's loss
under no_grad, so the manifest records per cell that the two objectives actually
took different values on the same data (`rc_objective_non_vacuous`, a readiness
precondition -- not a design assumption).

WHAT WAS ALREADY KNOWN (commensurable numbers, same statistics)
---------------------------------------------------------------------------
V3-EXQ-965 (2026-08-30, confirmed autopsy): ITEM 1 (action-conditioned
transition) produces genuine per-action structure at the E1 output, and
cr_ratio(h=1) rose 6455x-9775x to 2.67e-03..3.96e-03 -- still 25-37x short of the
0.1 evaluator bar (CR_ROLLOUT_COLLAPSE_RATIO), e1coe_score_var 5-7 orders below
0.002 (C3_VAR_THRESHOLD). V3-EXQ-968 (2026-09-01): the design doc's pre-registered
absolute-vs-residual output_proj branch is SPENT -- `residual_no_material_difference`
(seed42 2.21x, seed123 0.34x, neither near the 3.0 floor; NOT to be read as residual
being better or worse). So the ~675x LSTM+output_proj crush is not a
parameterisation artefact, and the training objective is the remaining route.
This run keeps 968's methodology (Phases 0a/1/2/3/4/4b/4c verbatim, the same
cr_ratio(h) / e1coe_score_var(h) sweep over the same checkpoints, the same
readiness set) and changes TWO things in Phase 0b: the E1 objective (the
manipulation) and, for every ROUTED arm, the hidden-state regime of the trained call
(the B1 symmetrisation below). Because the second change makes the routed OFF arm a
DIFFERENT incumbent from 965/968's, 968's stateful incumbent is kept VERBATIM as a
fourth, non-routing arm (ARM_single_step_stateful) so the run stays commensurable
(red-team pass 2, N1).

ARMS (all ITEM-1-ON: action_conditioned_transition=True,
action_cond_unzero_self_slot=True (its 965 null is NOT re-ablated here),
output_proj_residual=False (the 968-spent branch's default form)):
  ARM_single_step  ROUTED incumbent: single-step teacher-forced MSE at horizon 1,
                   trained from a zero hidden state (968's ARM_absolute config,
                   symmetrised call -- see below)
  ARM_single_step_stateful  968's incumbent VERBATIM (stateful forward() call);
                   NON-ROUTING commensurability anchor to 965/968's numbers
  ARM_rc_flat      rollout_consistency_loss, horizon RC_HORIZON=5, decay=1.0 --
                   the FLAT-FORM control: exactly F.mse_loss over the 5-step window,
                   so it isolates multi-step-vs-single-step with no discount confound
  ARM_rc_decay     rollout_consistency_loss, horizon 5, decay=RC_DECAY=0.5 --
                   isolates the discount itself (TD-MPC's actual form)
Arms (ii) and (iii) make this a 2-factor read (multi-step; discount) rather than one
blurred arm. 4 seeds (SEEDS_DEFAULT) because 968's 2 seeds disagreed in DIRECTION.
Objective non-vacuity is gated PER ON ARM (gradient cosine with the incumbent's
objective on the same windows, <= 0.95 on every seed); a vacuous arm is excluded
from the label and does not void the other arm; both vacuous -> not ready
(red-team pass 2, N2 -- as a whole-run conjunct the gate would have misfiled a real
"decay collapses toward the incumbent" finding as substrate_not_ready_requeue).

MATCHED TRAINING ACROSS ARMS -- the only difference is the objective. Phase 0b
runs the same random policy, the same env, the same seeds, and takes exactly ONE
E1 optimiser step per env step once a trailing window of RC_HORIZON observed
latents exists (window start i = t - RC_HORIZON): the single-step arm trains on
(total_i, a_i) -> total_{i+1}; the rollout arms train on total_i under
(a_i..a_{i+H-1}) -> (total_{i+1}..total_{i+H}). Same data stream, same number of
optimiser steps (recorded per cell; `e1_grad_steps_matched` certifies the schedule
landed -- supervision VOLUME per step differs by construction, H targets vs 1, and
that asymmetry IS the manipulation; red-team C2), E2 trained identically in every
arm (E2 is not under test).
HIDDEN-STATE SYMMETRISATION (red-team B1, BLOCKING, accepted). 968's incumbent
trained call is the STATEFUL forward(), whose LSTM hidden state persists and
accumulates within an episode (e1_deep.py:981), while rollout_consistency_loss
resets and restores the hidden state so it always trains at depth 0 -- and Phase 4
evaluates at depth 0 (reset_hidden_state before every candidate sequence). An OFF
arm trained 968's way would therefore differ from the ON arms in train/eval regime
as well as in objective, and a lift would be unattributable. This driver DELIBERATELY
departs from 968's verbatim incumbent: the single-step arm's trained call goes
through the same save/reset/restore lifecycle, so every trained E1 call in every
arm starts from a zero hidden state; `trained_calls_at_depth0` (min over ALL cells
of the fraction of trained calls that started at depth 0, must be 1.0) is a readiness
precondition, so the symmetrisation is measured, not assumed. Consequence of the
trailing window: every arm's first E1 update in an episode lands RC_HORIZON-1 env
steps later than 968's did; the data distribution is unchanged and the shift is
identical across arms.

DVs AND DECISION RULE (pre-registered; identical statistics to 965/968)
---------------------------------------------------------------------------
Primary: cr_ratio(h=1) = CR_rollout(h=1) / CR_real(h=1) per (arm, seed), Phase 4/4b.
Per seed, relative lift = cr_ratio_h1(ON arm) / cr_ratio_h1(ARM_single_step).
THE BAR IS FIXED: LIFT_BAR = 3.0, pre-registered. 968's rule (max(3.0, 2 x the
single-step arm's cross-seed max/min noise ratio)) is RECORDED as a diagnostic and
NOT applied (red-team B3, BLOCKING, accepted): that ratio measures cross-seed LEVEL
variation, a nuisance the within-seed paired ratio already cancels (cr_real(h=1) and
z_goal_norm differ across seeds by 1.5x while being bit-identical across arms at a
seed), it is monotone non-decreasing in the seed count (so adding seeds for power
RAISES the bar), and in the first smoke it reached 7.92 against lifts of 1.1-1.6 --
a near-determined null that would have licensed retiring candidate 1. Per ON arm:
LIFTS if relative lift >= LIFT_BAR on at least MAJORITY_SEEDS (= n_seeds//2 + 1,
i.e. 3 of 4); DEGRADES if 1/lift >= LIFT_BAR on at least MAJORITY_SEEDS; NULL if
neither direction fires on ANY seed; MIXED otherwise. A paired SIGN TEST across
seeds (direction only; P(all n positive | null) = 0.5**n) is recorded per arm.
SECONDARY at the TRAINED horizon (red-team C1): the identical rule at h=RC_HORIZON=5
(data already collected); a null or mixed at h=1 with a LIFT at h=5 becomes
rollout_consistency_lift_at_trained_horizon_only, which says h=1 is the wrong
readout for this objective and does NOT license the withheld contrastive. Labels:
  rollout_consistency_lifts_cr_ratio_h1        both ON arms LIFT
  rollout_consistency_lift_flat_only           flat LIFTS, decay does not
  rollout_consistency_lift_decay_only          decay LIFTS, flat does not
  rollout_consistency_degrades_cr_ratio_h1     an ON arm DEGRADES and none LIFTS
  rollout_consistency_null                     both ON arms NULL -- THIS is what
                                               licenses the deliberately-withheld
                                               rollout-endpoint contrastive
                                               (e1_rollout_sequence_divergence_*,
                                               substrate_queue item-2 log,
                                               why_not_contrastive)
  rollout_consistency_mixed_across_seeds       anything else
  rollout_consistency_lift_at_trained_horizon_only  null/mixed/DEGRADES at h=1 but
                                               LIFT at h=5 (does NOT license the
                                               contrastive; red-team pass 2 N3 widened
                                               this to cover degrades)
  rollout_consistency_one_arm_vacuous_other_not_lifting  one ON arm collinear with the
                                               incumbent (per-arm gate), the other
                                               neither lifts (licenses nothing)
  substrate_not_ready_requeue                  any readiness precondition unmet
Secondary, RECORDED and headlined but NOT routing the label: whether any ON cell
reaches the evaluator bars themselves (cr_ratio(h=1) >= 0.1 AND e1coe_score_var(h=1)
>= 0.002 -- `evaluator_bar_reached_cells`). A LIFT that still sits below the bar
is progress on the SD's target, not its closure. Also recorded: 968's caveat that
cr_ratio(h=1) = spread/centroid_norm and an objective change can move the
centroid_norm denominator mechanically -- per_seed_lift carries
cr_rollout_spread_h1 for both arms so a lift can be checked against the numerator
alone before being credited.

WHAT A NULL MEANS / DOES NOT MEAN. Null = training with the multi-step objective
at H=5 (flat or discounted) does not materially move E1's per-action rollout
divergence at h=1 on this substrate and budget; it licenses the withheld
contrastive. It does NOT say the objective is wrong at other horizons or budgets,
and it says nothing about MECH-135 / INV-088.

READINESS PRECONDITIONS (any unmet -> substrate_not_ready_requeue, FAIL, no verdict):
968's five verbatim (encoder_trained; real_zworld_nondegenerate_h1;
no_missing_action_calls; direct_action_supply_fraction >= 0.999;
cr_ratio_h1_finite) plus three this design needs:
  e1_grad_steps_matched     every arm took the same number of E1 optimiser steps
                            at a seed (max abs difference across arms == 0)
  rc_objective_non_vacuous  on every ON cell the mean cosine similarity between the
                            E1 GRADIENT of the trained rollout objective and the E1
                            gradient of the incumbent single-step objective on the
                            same window (sampled every 10 windows) is <= 0.95: the
                            objective trained E1 in a measurably different direction.
                            Red-team B2 (BLOCKING, accepted): the earlier gate --
                            |rc_loss - single_loss| / single_loss >= 0.01 -- was a
                            tautology of autoregressive rollout that the OFF arm
                            scored HIGHER on (1.5-1.8) than the ON arms (0.2-0.5);
                            it is kept as a recorded diagnostic only.
  trained_calls_at_depth0   every trained E1 call in every arm started from a zero
                            hidden state (min over cells == 1.0; red-team B1)
The same-statistic discipline holds: cr_ratio_h1_finite guards the exact quantity
the decision rule routes on.

DV-SYMMETRY DECLARATION (one line per arm). The DV is cr_ratio(h=1) = spread /
||centroid|| over 40 candidate rollout endpoints, divided by the arm-invariant
CR_real(h=1). Its symmetry group: a common positive rescaling of all endpoints
(cancels in spread/centroid), and permutation of the candidate index.
  ARM_single_step: reference; no manipulation.
  ARM_rc_flat vs single_step: a different training objective changes E1's learned
    weights, hence every endpoint's position non-uniformly -- neither a common
    rescaling nor a permutation. Reaches the DV; direction genuinely open.
  ARM_rc_decay vs single_step: same argument; additionally differs from rc_flat
    only in the per-step weights, which reweight the gradient across horizon steps.
The 968 caveat (centroid_norm can move mechanically) applies to every arm and is
why cr_rollout_spread_h1 is recorded per cell alongside the ratio.

GOV-REUSE-1 (Step 2.4): decisive readout = cr_ratio(h=1) on an E1 TRAINED with
rollout_consistency_loss. No manifest carries it: 954/965/968 all train single-step
(their own docstrings say so), and the substrate did not exist before 2026-09-01.
Not recoverable -> run. 965/968 are carried as the reference points for
ARM_single_step_stateful (their own stateful incumbent; same statistics, same
seeds 42/123 among the four) -- NOT for the symmetrised routed OFF arm.

STEP 2.5c (recorded, not blocking): the driver exercises e1_deep.py::forward /
predict_long_horizon (SD-e1-rollout-consistency-training -- this run IS its owed
validation) and ContextMemory.write (contextmemory-write-path-addressing-degeneracy,
corrupting, implemented_pending_validation): E1's context read/write is part of the
forward path in every arm identically, exactly as in 965/968, so it can bias the
absolute level of cr_ratio but not the between-arm contrast this run routes on.

SLEEP DRIVER: none (no sleep flags set).
red-team (opus): pass 1 BLOCKING (B1 hidden-state regime asymmetry; B2 tautological
non-vacuity gate; B3 noise-inflated lift bar) -> all three fixed in this file, plus
C1 (secondary verdict at the trained horizon), C2 (honest e1_grad_steps_matched),
C3 (e1coe_score_var_rel_change recorded), C4 (convergence proxy recorded).
Pass 2 BLOCKING on N1 (commensurability: stateful 968 incumbent added as a
non-routing anchor arm; docstring claims retracted), N2 (per-arm vacuity gate),
N3 (trained-horizon override covers degrades; degrades licensing stated); N5
(h=1 structural favour) recorded as a limitation. No third pass -- dispositions
in the queue entry note and scratchpad redteam.md.
EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence scoring.
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
from experiments._metrics import p0_readiness_gate, P0NotReady


EXPERIMENT_TYPE = "v3_exq_976_sd_e1_item2_rollout_consistency_validation"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None

# agent_construction_before_seed lint exemption -- same shape and same
# justification as V3-EXQ-954/965: every scored quantity in this script is
# read off the LITERAL SAME agent object within a (seed, arm) cell. Cross-arm
# comparisons ARE across independently-constructed agents by design (that is
# the whole point of the A/B) and are seed-matched via arm_cell()'s RNG reset.
AGENT_SEED_ORDER_EXEMPT = (
    "Every within-cell comparison (action-vs-action, horizon-vs-horizon) is "
    "scored off the literal same agent object; cross-arm comparisons are the "
    "A/B itself and are seed-matched via arm_cell()."
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
CR_REAL_FLOOR = 1e-4               # unchanged from V3-EXQ-954/965/968 -- P_REAL_ZWORLD floor
CR_ROLLOUT_COLLAPSE_RATIO = 0.1    # ITEM 2's evaluator bar -- RECORDED (evaluator_bar_reached), not routing
C3_VAR_THRESHOLD = 0.002           # ITEM 2's evaluator bar -- RECORDED, not routing
ZWORLD_P0_EPISODES = 60            # SD-070 encoder warmup -- matches V3-EXQ-954/965/968
N_REAL_SAMPLES = 40                # per-checkpoint target sample count for CR_real(h)
MIN_REAL_SAMPLES_PER_HORIZON = 10  # readiness floor: surviving real samples per checkpoint
HORIZON_CHECKPOINTS_FULL = [1, 2, 3, 5, 10, 20, 30]

# This run's OWN load-bearing thresholds (Phase 4/4b cr_ratio(h=1)), 968's rule:
# LIFT_FACTOR is derived per-run from the single-step arm's own measured
# cross-seed noise ratio in run() -- never a fixed inherited constant.
LIFT_FACTOR_ABS_FLOOR = 3.0        # == LIFT_BAR; kept under 968's name for the recorded diagnostic
LIFT_FACTOR_NOISE_MULTIPLE = 2.0   # 968's rule, RECORDED as noise_inflated_bar_diagnostic, NOT applied (B3)

# ITEM 2 objective parameters (the manipulation).
RC_HORIZON = 5                     # E1Config.e1_rollout_consistency_horizon default
RC_DECAY = 0.5                     # the discounted arm; 1.0 = flat form (exactly F.mse_loss over the window)
RC_WEIGHT = 1.0                    # E1Config.e1_rollout_consistency_weight default; caller composes
RC_NON_VACUITY_FLOOR = 0.01        # RECORDED diagnostic only (red-team B2): value difference is a tautology
GRAD_COS_VACUITY_CEILING = 0.95    # readiness (B2): ON-arm mean gradient cosine between the two objectives
GRAD_COS_SAMPLE_EVERY = 10         # gradient-cosine measured on every 10th training window
LIFT_BAR = 3.0                     # FIXED pre-registered relative-lift bar (B3): never max()'d with a noise ratio

OFF_ARM = "ARM_single_step"
STATEFUL_ARM = "ARM_single_step_stateful"   # 968's incumbent VERBATIM (stateful call): commensurability anchor, non-routing
ON_ARMS = ["ARM_rc_flat", "ARM_rc_decay"]
ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ARM_single_step": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": False,
        "e1_loss": "single_step",
        "rc_horizon": 1,
        "rc_decay": 1.0,
    },
    "ARM_single_step_stateful": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": False,
        "e1_loss": "single_step_stateful",
        "rc_horizon": 1,
        "rc_decay": 1.0,
    },
    "ARM_rc_flat": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": True,
        "e1_loss": "rollout_consistency",
        "rc_horizon": RC_HORIZON,
        "rc_decay": 1.0,
    },
    "ARM_rc_decay": {
        "action_conditioned_transition": True,
        "action_cond_unzero_self_slot": True,
        "output_proj_residual": False,
        "e1_rollout_consistency_enabled": True,
        "e1_loss": "rollout_consistency",
        "rc_horizon": RC_HORIZON,
        "rc_decay": RC_DECAY,
    },
}
ARM_ORDER = [OFF_ARM, STATEFUL_ARM] + ON_ARMS
DEPTH0_ARMS = [OFF_ARM] + ON_ARMS            # the arms the depth-0 symmetrisation applies to

# The two anchor-kind readiness predicates this design adds ARE the definitions of
# the quantities they gate (a gradient-step count difference; a per-window loss
# difference on identical data), measured on every cell, not on a separate control
# run. Reachability was demonstrated by the --dry-run smoke (2026-09-02, 2 seeds x 3
# arms): e1_grad_step gap 0 on every seed; rc_vs_single_rel_diff 0.17-0.48 on every
# ON cell against the 0.01 floor.
ANCHOR_REACHABILITY_EXEMPT = (
    "e1_grad_steps_matched and rc_objective_non_vacuous are measured on every cell "
    "from the training loop itself (no separate positive-control run); reachability "
    "demonstrated by the dry-run smoke (gap 0; rel diff 0.17-0.48 vs floor 0.01)"
)
SEEDS_DEFAULT = [42, 123, 7, 2024]   # 42/123 = 954/965/968 lineage seeds; +2 because 968's two disagreed in direction


# ---------------------------------------------------------------------------
# Helpers (unchanged from V3-EXQ-954/965 unless noted)
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
    """Env config, unchanged from V3-EXQ-954/965."""
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
        e1_rollout_consistency_horizon=int(arm_cfg["rc_horizon"]),
        e1_rollout_consistency_horizon_weights_decay=float(arm_cfg["rc_decay"]),
    )
    config.latent.unified_latent_mode = False
    # from_dims swallows unknown kwargs silently (reference-reeconfig-from-dims-silent-kwargs);
    # assert the four knobs actually landed on the built config.
    assert bool(config.e1.e1_rollout_consistency_enabled) == bool(arm_cfg["e1_rollout_consistency_enabled"]), arm
    assert int(config.e1.e1_rollout_consistency_horizon) == int(arm_cfg["rc_horizon"]), arm
    assert abs(float(config.e1.e1_rollout_consistency_horizon_weights_decay) - float(arm_cfg["rc_decay"])) < 1e-12, arm
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0a: SD-070 sanctioned z_world encoder warmup (unchanged from 954/965)
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_976 P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context="v3_exq_976_sd_e1_item2_rollout_consistency_validation",
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 single-step training. Unchanged from V3-EXQ-965
# (threads actions=action_prev through the E1 call; calls
# agent.record_executed_action(action_curr) after every chosen action).
# ---------------------------------------------------------------------------

def _single_step_loss_state_preserving(
    agent: REEAgent, initial: torch.Tensor, action: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Single-step teacher-forced loss evaluated from a ZERO hidden state, with E1's
    hidden state saved and restored around the call -- the same lifecycle
    rollout_consistency_loss uses (e1_deep.py: saved_hidden / reset_hidden_state /
    restore), so every training call in every arm starts at hidden depth 0, which is
    also the Phase 4 evaluation condition (reset_hidden_state before every sequence).
    Red-team B1: 968's incumbent used the STATEFUL forward() call, whose hidden state
    persists and accumulates across the episode (e1_deep.py:981), so an OFF arm trained
    that way and evaluated at depth 0 would differ from the ON arms in train/eval regime
    as well as objective. Symmetrised deliberately; recorded as depth0_trained_call_frac."""
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
    agent.sense() exactly ONCE per env step, under torch.no_grad(), so Phase 0a's
    trained encoder is never disturbed).

    THE ONLY THING THAT DIFFERS BETWEEN ARMS IS E1'S OBJECTIVE. Every arm takes
    exactly one E1 optimiser step per env step once a trailing window of RC_HORIZON
    observed latents exists (window start i = t - RC_HORIZON), and EVERY trained E1
    call starts from a zero hidden state (red-team B1 symmetrisation -- see
    _single_step_loss_state_preserving):
      single_step         : F.mse_loss(e1(total_i, h=1, a_i)[:, 0], total_{i+1})
      rollout_consistency : agent.e1.rollout_consistency_loss(total_i,
                            stack(total_{i+1..i+H}), actions=stack(a_i..a_{i+H-1}),
                            horizon=H, horizon_weights_decay=decay) * RC_WEIGHT
    On every GRAD_COS_SAMPLE_EVERY-th window the OTHER objective is also built with
    gradients and the cosine similarity between the two objectives' E1 gradients is
    recorded (red-team B2: the value difference between the two losses is a tautology
    of autoregressive rollout; gradient direction is what training actually consumes).
    On every window the other objective's VALUE is recorded under no_grad as a
    diagnostic. Per-episode trained-loss traces are recorded (red-team C4). E2 trains
    identically in every arm on the latest (prev, curr) pair."""
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    arm_cfg = ARM_CONFIGS[arm]
    loss_kind = str(arm_cfg["e1_loss"])
    H = int(RC_HORIZON)  # window length is the SAME in every arm (data parity)
    decay = float(arm_cfg["rc_decay"])
    e1_params = [q for q in agent.e1.parameters() if q.requires_grad]

    opt_e1 = optim.Adam(agent.e1.parameters(), lr=1e-3)
    opt_e2 = optim.Adam(agent.e2.parameters(), lr=1e-3)

    stats: Dict[str, Any] = {
        "e1_loss_kind": loss_kind,
        "rc_horizon": H if loss_kind == "rollout_consistency" else 1,
        "rc_decay": decay,
        "rc_weight": RC_WEIGHT if loss_kind == "rollout_consistency" else 0.0,
        "supervision_targets_per_window": H if loss_kind == "rollout_consistency" else 1,
        "n_e1_grad_steps": 0,
        "n_e2_grad_steps": 0,
        "n_windows": 0,
        "n_depth0_trained_calls": 0,
        "depth0_trained_call_frac": 0.0,
        "single_step_loss_mean": 0.0,
        "rc_loss_mean": 0.0,
        "trained_loss_mean": 0.0,
        "rc_vs_single_rel_diff_mean": 0.0,
        "grad_cos_samples": 0,
        "grad_cos_mean": float("nan"),
        "grad_cos_min": float("nan"),
        "grad_cos_max": float("nan"),
        "grad_norm_trained_mean": 0.0,
        "grad_norm_other_mean": 0.0,
        "per_episode_trained_loss": [],
        "n_nonfinite_losses": 0,
    }
    single_sum = 0.0
    rc_sum = 0.0
    trained_sum = 0.0
    rel_diff_sum = 0.0
    cos_vals: List[float] = []
    gn_trained: List[float] = []
    gn_other: List[float] = []

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
                targets = torch.stack(totals[i + 1:i + 1 + H], dim=1)      # [1, H, total_dim]
                acts = torch.stack(actions[i:i + H], dim=1)                 # [1, H, action_dim]
                sample_grad = (stats["n_windows"] % GRAD_COS_SAMPLE_EVERY == 0)
                opt_e1.zero_grad()
                # both trained calls start from a zero hidden state (B1)
                if agent.e1._hidden_state is None:
                    stats["n_depth0_trained_calls"] += 1  # counted BEFORE the call, for the OFF arm's own path
                if loss_kind in ("single_step", "single_step_stateful"):
                    if loss_kind == "single_step":
                        trained = _single_step_loss_state_preserving(agent, initial, actions[i], totals[i + 1])
                    else:
                        # 968's incumbent VERBATIM: stateful forward(), hidden state persists
                        # and accumulates across the episode (e1_deep.py:981). Commensurability
                        # anchor to 965/968; NOT routed (red-team pass 2, N1).
                        e1_pred, _ = agent.e1(initial, horizon=1, actions=actions[i])
                        trained = F.mse_loss(e1_pred[:, 0, :], totals[i + 1])
                    e1_call_counter["n_e1_calls"] += 1
                    e1_call_counter["n_e1_calls_nonzero_action"] += 1
                    single_val = float(trained.item())
                    if sample_grad:
                        other = agent.e1.rollout_consistency_loss(
                            initial, targets, actions=acts, horizon=H, horizon_weights_decay=decay,
                        )
                    else:
                        with torch.no_grad():
                            other = agent.e1.rollout_consistency_loss(
                                initial, targets, actions=acts, horizon=H, horizon_weights_decay=decay,
                            )
                    rc_val = float(other.item())
                    weighted = trained
                else:
                    trained = agent.e1.rollout_consistency_loss(
                        initial, targets, actions=acts, horizon=H, horizon_weights_decay=decay,
                    )
                    e1_call_counter["n_e1_calls"] += 1
                    e1_call_counter["n_e1_calls_nonzero_action"] += 1
                    rc_val = float(trained.item())
                    if sample_grad:
                        other = _single_step_loss_state_preserving(agent, initial, actions[i], totals[i + 1])
                    else:
                        with torch.no_grad():
                            other = _single_step_loss_state_preserving(agent, initial, actions[i], totals[i + 1])
                    single_val = float(other.item())
                    weighted = RC_WEIGHT * trained
                if not (single_val == single_val and rc_val == rc_val):
                    stats["n_nonfinite_losses"] += 1
                if sample_grad and trained.requires_grad and other.requires_grad:
                    g_t = _flat_grad(trained, e1_params)
                    g_o = _flat_grad(other, e1_params)
                    if g_t is not None and g_o is not None and g_t.numel() == g_o.numel():
                        nt = float(g_t.norm().item())
                        no = float(g_o.norm().item())
                        if nt > 0 and no > 0:
                            cos_vals.append(float(torch.dot(g_t, g_o).item() / (nt * no)))
                            gn_trained.append(nt)
                            gn_other.append(no)
                weighted.backward()
                opt_e1.step()
                stats["n_e1_grad_steps"] += 1
                stats["n_windows"] += 1
                single_sum += single_val
                rc_sum += rc_val
                trained_sum += float(weighted.item())
                rel_diff_sum += (abs(rc_val - single_val) / single_val) if single_val > 0 else 0.0
                ep_loss_e1 += float(weighted.item())
                n_steps += 1

            _, _, done, _, obs_dict = env.step(action_curr)

            latent_prev = latent_curr
            action_prev = action_curr

            if done:
                break

        stats["per_episode_trained_loss"].append(ep_loss_e1 / max(n_steps, 1))
        if (ep + 1) % 20 == 0:
            print(
                f"  [Train] ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    n_w = max(int(stats["n_windows"]), 1)
    stats["single_step_loss_mean"] = single_sum / n_w
    stats["rc_loss_mean"] = rc_sum / n_w
    stats["trained_loss_mean"] = trained_sum / n_w
    stats["rc_vs_single_rel_diff_mean"] = rel_diff_sum / n_w
    stats["depth0_trained_call_frac"] = stats["n_depth0_trained_calls"] / n_w
    if cos_vals:
        stats["grad_cos_samples"] = len(cos_vals)
        stats["grad_cos_mean"] = float(sum(cos_vals) / len(cos_vals))
        stats["grad_cos_min"] = float(min(cos_vals))
        stats["grad_cos_max"] = float(max(cos_vals))
        stats["grad_norm_trained_mean"] = float(sum(gn_trained) / len(gn_trained))
        stats["grad_norm_other_mean"] = float(sum(gn_other) / len(gn_other))
    tr = stats["per_episode_trained_loss"]
    k = max(1, len(tr) // 5)
    stats["trained_loss_first_fifth_mean"] = float(sum(tr[:k]) / k) if tr else 0.0
    stats["trained_loss_last_fifth_mean"] = float(sum(tr[-k:]) / k) if tr else 0.0
    stats["trained_loss_last_over_first_fifth"] = (
        (stats["trained_loss_last_fifth_mean"] / stats["trained_loss_first_fifth_mean"])
        if stats["trained_loss_first_fifth_mean"] > 0 else float("nan")
    )

    agent.eval()
    print(
        f"  [Train] Done. {n_episodes} episodes; e1_grad_steps={stats['n_e1_grad_steps']} "
        f"depth0_frac={stats['depth0_trained_call_frac']:.3f} "
        f"single_step_loss_mean={stats['single_step_loss_mean']:.6e} "
        f"rc_loss_mean={stats['rc_loss_mean']:.6e} "
        f"grad_cos_mean={stats['grad_cos_mean']:.4f} (n={stats['grad_cos_samples']}) "
        f"loss_last/first_fifth={stats['trained_loss_last_over_first_fifth']:.3f}",
        flush=True,
    )
    return stats


# ---------------------------------------------------------------------------
# Phase 1: goal template (unchanged from 954/965)
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
# Phase 2: warmup state (unchanged from 954/965)
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
# Phase 3: generate candidate sequences (unchanged from 954/965)
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
# Phase 4 (NON-GATING; recorded so ITEM 2 has its baseline): rollout scoring
# at every horizon checkpoint. Unchanged from V3-EXQ-965.
# ---------------------------------------------------------------------------

def _score_sequence_e1coe_multi_horizon(
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
# Phase 4b (NON-GATING beyond the h=1 readiness floor; recorded so ITEM 2 has
# its baseline): real z_world sample at every horizon checkpoint. Unchanged
# from V3-EXQ-965.
# ---------------------------------------------------------------------------

def _collect_real_zworld_sample_multi_horizon(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int,
    checkpoints: List[int],
) -> Dict[int, List[torch.Tensor]]:
    """n_samples independent random-policy rollouts from reset. Uses its own
    seed offset (+3000), mirroring V3-EXQ-954/965's Phase 4b, so it does not
    disturb the deterministic warmup-state RNG stream."""
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
    """CR = spread / ||centroid||, per zworld_near_static_characterisation_2026-07-18
    sec 2's offset-invariant statistic. Unchanged from V3-EXQ-954/965."""
    stacked = torch.cat(vectors, dim=0)  # [N, dim]
    centroid = stacked.mean(dim=0, keepdim=True)  # [1, dim]
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr, "n": len(vectors)}


# ---------------------------------------------------------------------------
# Phase 4c (secondary discrimination cross-reference -- see DV-SYMMETRY note
# in the module docstring for why this is a genuine, non-confounded readout
# for this trained-per-arm design, not merely a sanity check): one-step
# per-action divergence probe, DIRECT E1 channel. Unchanged from V3-EXQ-965's
# redesign.
# ---------------------------------------------------------------------------

def _one_step_action_divergence(
    agent: REEAgent,
    z_self_0: torch.Tensor,
    z_world_0: torch.Tensor,
    self_dim: int,
    e1_call_counter: Dict[str, int],
) -> Dict[str, Any]:
    """From the SAME Phase-2 warmup state, one deterministic single-step E1
    forward call per action (every action tested exactly once, agent.e1's
    hidden state reset before each so the K calls are independent). (z_self_0,
    z_world_0) is held BITWISE FIXED across all K calls; the ONLY varying
    input is the actions= one-hot."""
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
    print(f"\n[EXQ-976] seed={seed} arm={arm}", flush=True)
    print(f"Seed {seed} Condition {arm}", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim, arm)
    e1_call_counter = {"n_e1_calls": 0, "n_e1_calls_nonzero_action": 0}

    # Phase 0a: SD-070 sanctioned encoder warmup (unchanged from 954/965)
    print(f"[EXQ-976] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    # Phase 0b: bespoke E1/E2 single-step training, action-conditioned
    print(f"[EXQ-976] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    train_stats = _train_agent(agent, env, seed, n_train_episodes, steps_per_episode, e1_call_counter, arm)

    # Phase 1: goal template (unchanged from 954/965)
    print("[EXQ-976] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    # Phase 2: warmup state (unchanged from 954/965)
    print("[EXQ-976] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    # Phase 3: candidate sequences (unchanged from 954/965)
    print(f"[EXQ-976] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 4 (NON-GATING): score sequences at every horizon checkpoint
    print(f"[EXQ-976] Phase 4: scoring sequences at horizons {checkpoints} (non-gating)...", flush=True)
    scores_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for i, seq in enumerate(seqs):
        per_h = _score_sequence_e1coe_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints,
            e1_call_counter,
        )
        for h, (score, endpoint) in per_h.items():
            scores_by_h[h].append(score)
            endpoints_by_h[h].append(endpoint)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    e1coe_score_var_by_h: Dict[int, float] = {}
    cr_rollout_by_h: Dict[int, Dict[str, float]] = {}
    for h in checkpoints:
        scores_t = torch.tensor(scores_by_h[h])
        e1coe_score_var_by_h[h] = float(scores_t.var().item()) if len(scores_by_h[h]) > 1 else 0.0
        cr_rollout_by_h[h] = _contrast_ratio(endpoints_by_h[h])
        print(
            f"  h={h:>2d}: e1coe_score_var={e1coe_score_var_by_h[h]:.6e} "
            f"CR_rollout={cr_rollout_by_h[h]['contrast_ratio']:.6e}",
            flush=True,
        )

    # Phase 4b: real z_world sample at every horizon checkpoint (NON-GATING
    # beyond the h=1 readiness floor)
    print(f"[EXQ-976] Phase 4b: sampling {n_real_samples} real trajectories at horizons {checkpoints}...", flush=True)
    real_samples_by_h = _collect_real_zworld_sample_multi_horizon(
        agent, env, seed, n_real_samples, checkpoints,
    )
    cr_real_by_h: Dict[int, Dict[str, float]] = {}
    cr_ratio_by_h: Dict[int, float] = {}
    for h in checkpoints:
        samples = real_samples_by_h[h]
        if len(samples) >= 2:
            cr_real_by_h[h] = _contrast_ratio(samples)
        else:
            cr_real_by_h[h] = {"spread": 0.0, "centroid_norm": 0.0, "contrast_ratio": float("nan"), "n": len(samples)}
        cr_real = cr_real_by_h[h]["contrast_ratio"]
        cr_roll = cr_rollout_by_h[h]["contrast_ratio"]
        cr_ratio_by_h[h] = (cr_roll / cr_real) if (cr_real == cr_real and cr_real > 0) else float("nan")
        print(
            f"  h={h:>2d}: CR_real={cr_real:.6e} (n={cr_real_by_h[h]['n']}) "
            f"ratio={cr_ratio_by_h[h]:.6e}",
            flush=True,
        )

    # Phase 4c (sanity/positive-control -- see DV-SYMMETRY note in module docstring)
    print("[EXQ-976] Phase 4c: direct-channel one-step per-action divergence probe...", flush=True)
    action_probe = _one_step_action_divergence(agent, z_self_0, z_world_0, self_dim, e1_call_counter)
    cr_real_h1 = cr_real_by_h.get(1, {}).get("contrast_ratio", float("nan"))
    action_cr = action_probe["contrast_ratio"]["contrast_ratio"]
    ratio_action_vs_real_h1 = (
        (action_cr / cr_real_h1) if (cr_real_h1 == cr_real_h1 and cr_real_h1 > 0) else float("nan")
    )
    print(
        f"  K={action_probe['n_actions']} pairwise_dist mean={action_probe['pairwise_dist_mean']:.6e} "
        f"min={action_probe['pairwise_dist_min']:.6e} max={action_probe['pairwise_dist_max']:.6e} "
        f"spread={action_probe['contrast_ratio']['spread']:.6e} "
        f"centroid_norm={action_probe['contrast_ratio']['centroid_norm']:.6e} "
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
        f"(internal_buffer_nonzero_fraction={buffer_stats.get('nonzero_fraction', 0.0):.4f}, "
        "recorded for cross-reference only, expected 0.0 -- _e1_tick() is never invoked)",
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
        "cr_real_by_h": cr_real_by_h,
        "cr_ratio_by_h": cr_ratio_by_h,
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
        "rc_vs_single_rel_diff_mean": float(train_stats["rc_vs_single_rel_diff_mean"]),
        "grad_cos_mean": float(train_stats["grad_cos_mean"]),
        "grad_cos_samples": int(train_stats["grad_cos_samples"]),
        "depth0_trained_call_frac": float(train_stats["depth0_trained_call_frac"]),
        "trained_loss_last_over_first_fifth": float(train_stats["trained_loss_last_over_first_fifth"]),
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
        # every module constant a cell's readouts depend on, declared so the OFF arm's
        # cross-driver-reusable fingerprint (include_driver_script_in_hash=False) is honest
        "rc_horizon": RC_HORIZON, "rc_decay": RC_DECAY, "rc_weight": RC_WEIGHT,
        "rc_non_vacuity_floor": RC_NON_VACUITY_FLOOR,
        "cr_real_floor": CR_REAL_FLOOR, "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "c3_var_threshold": C3_VAR_THRESHOLD, "zworld_p0_episodes_default": ZWORLD_P0_EPISODES,
        "n_real_samples_default": N_REAL_SAMPLES,
        "min_real_samples_per_horizon": MIN_REAL_SAMPLES_PER_HORIZON,
        "horizon_checkpoints_full": list(HORIZON_CHECKPOINTS_FULL),
        "lift_factor_abs_floor": LIFT_FACTOR_ABS_FLOOR,
        "lift_factor_noise_multiple": LIFT_FACTOR_NOISE_MULTIPLE,
        "lift_bar": LIFT_BAR,
        "grad_cos_vacuity_ceiling": GRAD_COS_VACUITY_CEILING,
        "grad_cos_sample_every": GRAD_COS_SAMPLE_EVERY,
        "stateful_arm": STATEFUL_ARM,
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
                # MINT AS YOU GO: the incumbent single-step arm is the lineage's OFF
                # baseline; emit it cross-driver reusable so a later letter can cite it.
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

    # Every arm here is ITEM-1-ON -- every row carries the missing-action-calls /
    # direct-supply vacuity checks.
    max_missing_action_calls = max((r["missing_action_calls"] for r in arm_results), default=0.0)
    p_no_missing_action_calls = max_missing_action_calls == 0.0

    min_direct_supply_fraction = min(
        (r["direct_action_supply_fraction"] for r in arm_results), default=0.0
    )
    p_direct_action_supply = min_direct_supply_fraction >= 0.999

    # ---- cr_ratio(h=1) finite on every cell (968's guard, verbatim) ----
    cr_ratio_h1_values = [
        by_arm_seed[(arm, seed)]["cr_ratio_by_h"].get(1, float("nan"))
        for arm in ARM_ORDER for seed in seeds
    ]
    p_cr_ratio_h1_finite = all(v == v and v > 0 for v in cr_ratio_h1_values)
    min_cr_ratio_h1 = min((v for v in cr_ratio_h1_values if v == v), default=float("nan"))

    # ---- THIS design's two readiness predicates ----
    # (a) matched E1 gradient-step counts across arms, per seed
    grad_step_gap_per_seed: Dict[str, int] = {}
    for seed in seeds:
        counts = [int(by_arm_seed[(arm, seed)]["n_e1_grad_steps"]) for arm in ARM_ORDER]
        grad_step_gap_per_seed[f"seed{seed}"] = int(max(counts) - min(counts))
    max_grad_step_gap = max(grad_step_gap_per_seed.values()) if grad_step_gap_per_seed else 0
    p_grad_steps_matched = max_grad_step_gap == 0
    # (b) the ON arms' objective actually TRAINED E1 differently from the incumbent:
    #     gradient cosine between the two objectives on sampled windows (red-team B2 --
    #     the loss-VALUE difference is a tautology of autoregressive rollout and the OFF
    #     arm scores highest on it; it is kept as a recorded diagnostic only).
    rc_rel_diff_on_cells = {
        f"{arm}_seed{seed}": float(by_arm_seed[(arm, seed)]["rc_vs_single_rel_diff_mean"])
        for arm in ON_ARMS for seed in seeds
    }
    rc_rel_diff_off_cells = {
        f"{OFF_ARM}_seed{seed}": float(by_arm_seed[(OFF_ARM, seed)]["rc_vs_single_rel_diff_mean"])
        for seed in seeds
    }
    grad_cos_on_cells = {
        f"{arm}_seed{seed}": float(by_arm_seed[(arm, seed)]["grad_cos_mean"])
        for arm in ON_ARMS for seed in seeds
    }
    grad_cos_off_cells = {
        f"{OFF_ARM}_seed{seed}": float(by_arm_seed[(OFF_ARM, seed)]["grad_cos_mean"])
        for seed in seeds
    }
    grad_cos_samples_min = min(
        (int(by_arm_seed[(arm, seed)]["grad_cos_samples"]) for arm in ON_ARMS for seed in seeds), default=0
    )
    finite_cos = [v for v in grad_cos_on_cells.values() if v == v]
    max_grad_cos_on = max(finite_cos) if finite_cos else float("nan")
    worst_cos_cell = (
        max((k for k in grad_cos_on_cells if grad_cos_on_cells[k] == grad_cos_on_cells[k]),
            key=lambda k: grad_cos_on_cells[k], default="none")
    )
    p_rc_non_vacuous_all = bool(
        grad_cos_samples_min > 0 and len(finite_cos) == len(grad_cos_on_cells)
        and max_grad_cos_on <= GRAD_COS_VACUITY_CEILING
    )
    # (c) every trained E1 call in every arm started from a zero hidden state (red-team B1)
    depth0_per_cell = {
        f"{arm}_seed{seed}": float(by_arm_seed[(arm, seed)]["depth0_trained_call_frac"])
        for arm in ARM_ORDER for seed in seeds
    }
    min_depth0 = min(
        (depth0_per_cell[f"{arm}_seed{seed}"] for arm in DEPTH0_ARMS for seed in seeds), default=0.0
    )
    p_depth0 = min_depth0 >= 1.0
    stateful_depth0_per_seed = {
        f"seed{seed}": depth0_per_cell[f"{STATEFUL_ARM}_seed{seed}"] for seed in seeds
    }
    # per-ARM objective non-vacuity (red-team pass 2, N2): an ON arm whose gradient direction
    # is collinear with the incumbent's on ANY seed is marked vacuous and excluded from the
    # label -- it does NOT void the other arm's result. Both vacuous -> not ready.
    on_arm_vacuous: Dict[str, bool] = {}
    on_arm_grad_cos_max: Dict[str, float] = {}
    for on_arm in ON_ARMS:
        vals = [float(by_arm_seed[(on_arm, seed)]["grad_cos_mean"]) for seed in seeds]
        ns = [int(by_arm_seed[(on_arm, seed)]["grad_cos_samples"]) for seed in seeds]
        finite = [v for v in vals if v == v]
        on_arm_grad_cos_max[on_arm] = max(finite) if finite else float("nan")
        on_arm_vacuous[on_arm] = bool(
            min(ns, default=0) == 0 or len(finite) != len(vals) or max(finite) > GRAD_COS_VACUITY_CEILING
        )
    p_some_on_arm_non_vacuous = not all(on_arm_vacuous.values())
    convergence_proxy_per_cell = {
        f"{arm}_seed{seed}": float(by_arm_seed[(arm, seed)]["trained_loss_last_over_first_fifth"])
        for arm in ARM_ORDER for seed in seeds
    }

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
                f"{MIN_REAL_SAMPLES_PER_HORIZON} surviving real samples, every "
                "cell -- the same statistic cr_ratio(h=1)'s denominator uses."
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
                "E1DeepPredictor._action_cond_missing_calls is 0 on every cell "
                "(every arm is ITEM-1-ON) -- every actions= call this script made, "
                "including the rollout_consistency_loss rollouts, supplied a real "
                "action sequence, never a silent zero-fallback."
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
                "The fraction of E1 calls that received a genuine non-None one-hot "
                "actions= argument (or explicit per-step sequence), minimum across cells."
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
                "cr_ratio(h=1) = CR_rollout(h=1)/CR_real(h=1) is finite and positive on "
                "every (arm, seed) cell -- the exact statistic the decision rule routes on "
                "(same-statistic discipline; 968's guard against a NaN silently reading "
                "as no-difference)."
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
                "Every arm took the same number of E1 OPTIMISER STEPS at a seed (max over "
                "seeds of max-minus-min across arms). True by construction of the shared "
                "trailing-window schedule (red-team C2) -- it certifies the schedule landed, "
                "not that supervision VOLUME matched: a rollout window carries RC_HORIZON "
                "targets per step against the incumbent's one (supervision_targets_per_window "
                "is recorded per cell). That asymmetry IS the manipulation."
            ),
            "measured": float(max_grad_step_gap),
            "threshold": 0.0,
            "direction": "upper",
            "comparator": "<=",
            "control": "identical random policy, env, seeds and trailing-window schedule in every arm",
            "met": p_grad_steps_matched,
        },
        {
            "name": "rc_objective_non_vacuous",
            "kind": "readiness",
            "description": (
                "On every ON cell, the mean cosine similarity between the E1 GRADIENT of the "
                "trained rollout-consistency objective and the E1 gradient of the incumbent "
                "single-step objective, both built on the SAME window (sampled every "
                f"{GRAD_COS_SAMPLE_EVERY} windows), is <= {GRAD_COS_VACUITY_CEILING}: the "
                "multi-step objective trained E1 in a measurably different direction, so the "
                "ON arm is not a vacuous flag-only arm (red-team B2: a loss-VALUE difference "
                "is a tautology of autoregressive rollout and the OFF arm scores highest on "
                "it; kept as rc_vs_single_rel_diff, a diagnostic). Worst (highest) cell recorded."
            ),
            "measured": float(max_grad_cos_on),
            "threshold": float(GRAD_COS_VACUITY_CEILING),
            "direction": "upper",
            "comparator": "<=",
            "control": "both objectives' gradients taken on identical windows in every arm; ON arms only gate",
            "offending_cell": worst_cos_cell,
            "met": p_rc_non_vacuous_all,
            "kind_note": (
                "INFORMATIONAL for adjudication of the WHOLE run: the gate is applied PER ON ARM "
                "(a vacuous arm is excluded from the label, it does not void the other arm -- "
                "red-team pass 2, N2). The whole-run readiness conjunct is at_least_one_on_arm_non_vacuous."
            ),
            "per_arm_max_grad_cos": on_arm_grad_cos_max,
            "per_arm_vacuous": on_arm_vacuous,
        },
        {
            "name": "at_least_one_on_arm_non_vacuous",
            "kind": "readiness",
            "description": (
                "At least one ON arm clears the per-arm gradient-cosine vacuity gate on every "
                "seed; if BOTH ON arms are collinear with the incumbent the run has no "
                "non-vacuous manipulation to route on."
            ),
            "measured": float(sum(1 for v in on_arm_vacuous.values() if not v)),
            "threshold": 1.0,
            "direction": "lower",
            "met": p_some_on_arm_non_vacuous,
        },
        {
            "name": "trained_calls_at_depth0",
            "kind": "readiness",
            "description": (
                "Fraction of trained E1 calls that started from a zero LSTM hidden state, "
                "minimum over ALL cells -- must be 1.0: the OFF arm's incumbent (968) call is "
                "stateful and accumulates hidden state within an episode while "
                "rollout_consistency_loss resets and restores, and Phase 4 evaluates at depth 0, "
                "so without symmetrisation a lift would be jointly caused by the objective and "
                "a train/eval regime match (red-team B1). Symmetrised deliberately in this driver."
            ),
            "measured": float(min_depth0),
            "threshold": 1.0,
            "direction": "lower",
            "control": (
                "every ROUTED arm's trained call routed through the same save/reset/restore "
                "lifecycle; ARM_single_step_stateful is EXCLUDED by design (it is the stateful "
                "968 incumbent kept as the commensurability anchor) and its own depth-0 fraction "
                "is recorded as stateful_anchor_depth0_frac_per_seed"
            ),
            "stateful_anchor_depth0_frac_per_seed": stateful_depth0_per_seed,
            "met": p_depth0,
        },
    ]

    non_degenerate = bool(
        p_encoder_trained_met and p_real_nondegenerate_met
        and p_no_missing_action_calls and p_direct_action_supply
        and p_cr_ratio_h1_finite and p_grad_steps_matched and p_some_on_arm_non_vacuous
        and p_depth0
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
        """Per-seed relative lift of on_arm over off_arm in cr_ratio(h) against a FIXED bar."""
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
            cr_roll_off = off_row["cr_rollout_by_h"].get(h, {})
            cr_roll_on = on_row["cr_rollout_by_h"].get(h, {})
            var_off = off_row["e1coe_score_var_by_h"].get(h, 0.0)
            var_on = on_row["e1coe_score_var_by_h"].get(h, 0.0)
            per_seed[f"seed{seed}"] = {
                "seed": seed,
                "h": h,
                "cr_ratio_single_step": off_cr,
                "cr_ratio_on_arm": on_cr,
                "relative_lift_on_over_single_step": rel,
                "cr_rollout_spread_single_step": cr_roll_off.get("spread"),
                "cr_rollout_spread_on_arm": cr_roll_on.get("spread"),
                "cr_rollout_centroid_norm_single_step": cr_roll_off.get("centroid_norm"),
                "cr_rollout_centroid_norm_on_arm": cr_roll_on.get("centroid_norm"),
                "e1coe_score_var_single_step": var_off,
                "e1coe_score_var_on_arm": var_on,
                "e1coe_score_var_rel_change": ((var_on / var_off) if var_off > 0 else float("nan")),
                "on_materially_exceeds": exceeds,
                "on_materially_below": below,
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

    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        unmet_names = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = "P0 readiness unmet: " + ", ".join(unmet_names)
        status = "FAIL"
        evidence_direction = "non_contributory"
        per_seed_lift: Dict[str, Dict[str, Any]] = {}
        per_seed_lift_trained_horizon: Dict[str, Dict[str, Any]] = {}
        per_seed_lift_vs_stateful_anchor: Dict[str, Dict[str, Any]] = {}
        arm_verdicts: Dict[str, Dict[str, Any]] = {}
        arm_verdicts_trained_horizon: Dict[str, Dict[str, Any]] = {}
        arm_verdicts_vs_stateful_anchor: Dict[str, Dict[str, Any]] = {}
        h1_label_before_override = label
        lift_bar_used = LIFT_BAR
        noise_ratio_measured = None
        noise_inflated_bar_diagnostic = None
        criteria = []
        criteria_non_degenerate = {
            "C1_cr_ratio_h1_relative_lift_rc_flat": False,
            "C1_cr_ratio_h1_relative_lift_rc_decay": False,
        }
    else:
        degeneracy_reason = None
        lift_bar_used = LIFT_BAR

        # 968's noise-inflated bar, RECORDED as a diagnostic and NOT applied (red-team B3:
        # max/min of the OFF arm's cross-seed level is a nuisance the paired per-seed
        # ratio cancels, and it is monotone in the seed count).
        off_h1_vals = sorted(
            by_arm_seed[(OFF_ARM, seed)]["cr_ratio_by_h"].get(1, float("nan")) for seed in seeds
        )
        off_h1_vals = [v for v in off_h1_vals if v == v and v > 0]
        noise_ratio_measured = (
            (off_h1_vals[-1] / off_h1_vals[0]) if len(off_h1_vals) >= 2 else 1.0
        )
        noise_inflated_bar_diagnostic = max(LIFT_FACTOR_ABS_FLOOR, LIFT_FACTOR_NOISE_MULTIPLE * noise_ratio_measured)
        print(
            f"\n[EXQ-976] LIFT_BAR (fixed, pre-registered) = {LIFT_BAR}; MAJORITY_SEEDS = "
            f"{majority_seeds} of {len(seeds)}; diagnostic only: {OFF_ARM} cross-seed noise ratio "
            f"{noise_ratio_measured:.4f} -> 968-style inflated bar would have been "
            f"{noise_inflated_bar_diagnostic:.4f}",
            flush=True,
        )

        per_seed_lift = {}
        per_seed_lift_trained_horizon = {}
        arm_verdicts = {}
        arm_verdicts_trained_horizon = {}
        # commensurability anchor (red-team pass 2, N1): the same lift rule against 968's
        # stateful incumbent, RECORDED, never routed
        per_seed_lift_vs_stateful_anchor = {}
        arm_verdicts_vs_stateful_anchor = {}
        for on_arm in ON_ARMS:
            v1, ps1, ne1, nb1, npos1 = _arm_verdict_at(on_arm, 1, LIFT_BAR)
            if on_arm_vacuous[on_arm]:
                v1 = "vacuous_objective"
            per_seed_lift[on_arm] = ps1
            # paired sign test across seeds (direction only, no bar): under the null of
            # no effect each seed is a fair coin, P(all n positive) = 0.5 ** n
            arm_verdicts[on_arm] = {
                "verdict": v1,
                "h": 1,
                "bar": LIFT_BAR,
                "n_seeds_exceeds": ne1,
                "n_seeds_below": nb1,
                "n_seeds_direction_positive": npos1,
                "sign_test_p_all_positive": (0.5 ** len(seeds)) if npos1 == len(seeds) else None,
                "n_seeds": len(seeds),
                "majority_seeds": majority_seeds,
            }
            vH, psH, neH, nbH, nposH = _arm_verdict_at(on_arm, RC_HORIZON, LIFT_BAR)
            if on_arm_vacuous[on_arm]:
                vH = "vacuous_objective"
            per_seed_lift_trained_horizon[on_arm] = psH
            vA, psA, neA, nbA, nposA = _arm_verdict_at(on_arm, 1, LIFT_BAR, off_arm=STATEFUL_ARM)
            per_seed_lift_vs_stateful_anchor[on_arm] = psA
            arm_verdicts_vs_stateful_anchor[on_arm] = {
                "verdict_vs_stateful_968_incumbent": vA, "h": 1, "bar": LIFT_BAR,
                "n_seeds_exceeds": neA, "n_seeds_below": nbA, "n_seeds_direction_positive": nposA,
                "n_seeds": len(seeds), "majority_seeds": majority_seeds,
            }
            arm_verdicts_trained_horizon[on_arm] = {
                "verdict": vH,
                "h": RC_HORIZON,
                "bar": LIFT_BAR,
                "n_seeds_exceeds": neH,
                "n_seeds_below": nbH,
                "n_seeds_direction_positive": nposH,
                "sign_test_p_all_positive": (0.5 ** len(seeds)) if nposH == len(seeds) else None,
                "n_seeds": len(seeds),
                "majority_seeds": majority_seeds,
            }

        v_flat = arm_verdicts["ARM_rc_flat"]["verdict"]
        v_decay = arm_verdicts["ARM_rc_decay"]["verdict"]
        n_vac = sum(1 for v in on_arm_vacuous.values() if v)
        if v_flat == "lifts" and v_decay == "lifts":
            label = "rollout_consistency_lifts_cr_ratio_h1"
        elif v_flat == "lifts":
            label = "rollout_consistency_lift_flat_only"
        elif v_decay == "lifts":
            label = "rollout_consistency_lift_decay_only"
        elif n_vac == 1:
            # one arm vacuous (collinear with the incumbent), the other neither lifts: the
            # null-licensing needs BOTH arms genuinely tested, so this is its own label
            label = "rollout_consistency_one_arm_vacuous_other_not_lifting"
        elif "degrades" in (v_flat, v_decay):
            label = "rollout_consistency_degrades_cr_ratio_h1"
        elif v_flat == "null" and v_decay == "null":
            label = "rollout_consistency_null"
        else:
            label = "rollout_consistency_mixed_across_seeds"
        h1_label_before_override = label
        # Secondary (red-team C1, widened per pass 2 N3): a null / mixed / DEGRADES at h=1 with
        # a LIFT at the TRAINED horizon is NOT a null (or a degradation) for candidate 1 -- it
        # says h=1 is the wrong readout for this objective.
        any_lift_at_H = any(v["verdict"] == "lifts" for v in arm_verdicts_trained_horizon.values())
        if label in (
            "rollout_consistency_null", "rollout_consistency_mixed_across_seeds",
            "rollout_consistency_degrades_cr_ratio_h1",
        ) and any_lift_at_H:
            label = "rollout_consistency_lift_at_trained_horizon_only"

        status = "PASS"  # diagnostic discrimination -- informative in every direction
        evidence_direction = "non_contributory"  # diagnostic, claim-free -- see docstring
        criteria = []
        for on_arm in ON_ARMS:
            lifts = [
                v["relative_lift_on_over_single_step"] for v in per_seed_lift[on_arm].values()
                if v["relative_lift_on_over_single_step"] == v["relative_lift_on_over_single_step"]
            ]
            criteria.append({
                "name": f"C1_cr_ratio_h1_relative_lift_{on_arm.replace('ARM_', '')}",
                "load_bearing": True,
                "passed": True,  # this criterion CLASSIFIES (lifts/degrades/null/mixed), it does not gate PASS/FAIL
                "arm_verdict": arm_verdicts[on_arm]["verdict"],
                "arm_verdict_at_trained_horizon": arm_verdicts_trained_horizon[on_arm]["verdict"],
                "measured": max(lifts, default=float("nan")),
                "measured_min": min(lifts, default=float("nan")),
                "threshold": LIFT_BAR,
                "statement": (
                    f"cr_ratio(h=1) relative lift of {on_arm} over {OFF_ARM}, per seed, against the "
                    f"FIXED pre-registered bar {LIFT_BAR}; verdict on at least {majority_seeds}/{len(seeds)} "
                    "seeds; paired sign test recorded. The same rule at h=RC_HORIZON is the recorded "
                    "secondary. See interpretation.dv_symmetry_note: check cr_rollout_spread before "
                    "crediting either direction (centroid-norm caveat)."
                ),
            })
        criteria.append({
            "name": "C2_evaluator_bar_reached",
            "load_bearing": False,
            "passed": bool(evaluator_bar_reached_cells),
            "measured": len(evaluator_bar_reached_cells),
            "threshold": 1,
            "statement": (
                f"Any ON cell reaches BOTH evaluator bars (cr_ratio(h=1) >= {CR_ROLLOUT_COLLAPSE_RATIO} "
                f"and e1coe_score_var(h=1) >= {C3_VAR_THRESHOLD}). RECORDED, not routing -- a lift "
                "below the bar is progress on the SD's target, not its closure. NOTE (red-team C3): "
                "the two evaluator quantities can move in opposite directions; e1coe_score_var_rel_change "
                "is recorded per seed so the reader sees both."
            ),
        })
        criteria_non_degenerate = {
            "C1_cr_ratio_h1_relative_lift_rc_flat": non_degenerate,
            "C1_cr_ratio_h1_relative_lift_rc_decay": non_degenerate,
            "C2_evaluator_bar_reached": non_degenerate,
        }

    print(f"\n[EXQ-976] Label: {label}", flush=True)
    print(f"[EXQ-976] Status: {status}", flush=True)

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
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
        "rc_horizon": RC_HORIZON,
        "rc_decay": RC_DECAY,
        "rc_weight": RC_WEIGHT,
        "registered_cr_real_floor": CR_REAL_FLOOR,
        "registered_lift_bar": LIFT_BAR,
        "registered_lift_factor_abs_floor": LIFT_FACTOR_ABS_FLOOR,
        "registered_lift_factor_noise_multiple": LIFT_FACTOR_NOISE_MULTIPLE,
        "registered_majority_seeds": majority_seeds,
        "registered_rc_non_vacuity_floor": RC_NON_VACUITY_FLOOR,
        "registered_grad_cos_vacuity_ceiling": GRAD_COS_VACUITY_CEILING,
        "registered_grad_cos_sample_every": GRAD_COS_SAMPLE_EVERY,
        "registered_cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "registered_c3_var_threshold": C3_VAR_THRESHOLD,
        "lift_bar_used": lift_bar_used,
        "measured_single_step_arm_noise_ratio_diagnostic": noise_ratio_measured,
        "noise_inflated_bar_diagnostic_not_applied": noise_inflated_bar_diagnostic,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_seed_lift": per_seed_lift,
        "per_seed_lift_trained_horizon": per_seed_lift_trained_horizon,
        "per_seed_lift_vs_stateful_968_incumbent": per_seed_lift_vs_stateful_anchor,
        "arm_verdicts": arm_verdicts,
        "arm_verdicts_trained_horizon": arm_verdicts_trained_horizon,
        "arm_verdicts_vs_stateful_968_incumbent": arm_verdicts_vs_stateful_anchor,
        "h1_label_before_trained_horizon_override": h1_label_before_override,
        "on_arm_vacuous": on_arm_vacuous,
        "on_arm_grad_cos_max": on_arm_grad_cos_max,
        "stateful_arm": STATEFUL_ARM,
        "depth0_arms": DEPTH0_ARMS,
        "evaluator_bar_reached_cells": evaluator_bar_reached_cells,
        "e1_grad_step_gap_per_seed": grad_step_gap_per_seed,
        "rc_vs_single_rel_diff_on_cells": rc_rel_diff_on_cells,
        "rc_vs_single_rel_diff_off_cells_diagnostic": rc_rel_diff_off_cells,
        "grad_cos_on_cells": grad_cos_on_cells,
        "grad_cos_off_cells_diagnostic": grad_cos_off_cells,
        "depth0_trained_call_frac_per_cell": depth0_per_cell,
        "trained_loss_last_over_first_fifth_per_cell": convergence_proxy_per_cell,
        "status": status,
        "outcome": status,
        "verdict": status,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "combination_rule": (
                "Per ON arm, per seed: relative lift of cr_ratio(h=1) over ARM_single_step "
                "vs the FIXED pre-registered LIFT_BAR (3.0); arm verdict lifts/degrades needs >= "
                "MAJORITY_SEEDS seeds in that direction, null = neither direction on any seed, "
                "mixed otherwise. Label composes the two arm verdicts (see docstring); a null or "
                "mixed at h=1 with a LIFT at h=RC_HORIZON (same rule) becomes "
                "rollout_consistency_lift_at_trained_horizon_only, which does NOT license the "
                "withheld contrastive. A paired sign test across seeds is recorded per arm. No "
                "criterion gates PASS/FAIL once readiness is met -- the run is a classification. "
                "C2 is recorded only."
            ),
            "convergence_note": (
                "trained_loss_last_over_first_fifth per cell (ratio of the last-fifth to the "
                "first-fifth per-episode trained loss) is recorded so a null can be read against "
                "how far training had progressed (red-team C4); it is not gated, because the "
                "budget (100 x 200) is the lineage's and matching 965/968 is what makes the "
                "numbers commensurable."
            ),
            "dv_symmetry_note": (
                "cr_ratio(h=1) = cr_rollout(h=1)/cr_real(h=1); cr_real(h=1) is arm-invariant "
                "per seed, so the contrast reduces to cr_rollout(h=1) = spread/centroid_norm "
                "over the 40 candidate rollout endpoints. A changed training objective moves "
                "E1's learned weights and hence every endpoint non-uniformly (neither a common "
                "rescaling nor a permutation of the candidate index), so the manipulation "
                "reaches the DV and the direction is genuinely open for both ON arms. 968's "
                "caveat stands: the centroid_norm denominator can shift mechanically under a "
                "different objective, so per_seed_lift carries cr_rollout_spread_h1 for both "
                "arms -- check the numerator alone before crediting a lift or a degradation "
                "to a real change in per-action divergence."
            ),
            "commensurability_note": (
                "ARM_single_step (the ROUTED OFF arm) trains from a zero hidden state -- a "
                "deliberate departure from 965/968's stateful incumbent (red-team B1). So a "
                "'degrades' means 'worse than a depth-0-trained single-step incumbent', NOT "
                "'worse than the lineage's incumbent'. ARM_single_step_stateful is 968's "
                "incumbent verbatim, kept as a NON-ROUTING commensurability anchor: its "
                "cr_ratio(h=1) is directly comparable to 965/968's recorded numbers, and "
                "per_seed_lift_vs_stateful_968_incumbent / arm_verdicts_vs_stateful_968_incumbent "
                "apply the identical lift rule against it (recorded, never routed). Red-team "
                "pass 2, N1. Its measured depth-0 fraction is recorded per seed."
            ),
            "h1_structural_favour_note": (
                "The routed OFF arm's training step is, after symmetrisation, the same operation "
                "as the h=1 evaluation (one depth-0 single-step prediction), while the ON arms "
                "spread their gradient over RC_HORIZON steps of which the first is that same "
                "operation. A 'degrades' at h=1 is therefore structurally favoured, not "
                "evidence against multi-step consistency per se (red-team pass 2, N5); the "
                "trained-horizon secondary is the counterweight, and it overrides degrades as "
                "well as null/mixed. A lift at h=5 is credible; a null at h=5 is weaker evidence "
                "than a null at h=1, because the h=5 candidate rollouts are re-primed per step "
                "with an E2-predicted self slot whereas training rolls E1 out alone."
            ),
            "what_degrades_licenses": (
                "rollout_consistency_degrades_cr_ratio_h1 (an ON arm worse than the depth-0 "
                "incumbent by >= LIFT_BAR on >= MAJORITY_SEEDS seeds at h=1, with no lift at h=5) "
                "licenses NOTHING downstream on its own: it is read against the stateful anchor "
                "(is it also below 968's incumbent?), the numerator spread, and the convergence "
                "proxy before any substrate decision; it does not license the contrastive."
            ),
            "what_a_null_licenses": (
                "rollout_consistency_null -- a null at h=1 AND no lift at h=RC_HORIZON, against the "
                "fixed bar 3.0 on >= MAJORITY_SEEDS seeds with the sign test recorded -- licenses "
                "the deliberately-withheld rollout-endpoint contrastive over candidate action "
                "sequences (e1_rollout_sequence_divergence_*; substrate_queue "
                "SD-e1-rollout-consistency-training item-2 log, why_not_contrastive). "
                "rollout_consistency_lift_at_trained_horizon_only does NOT license it: it says "
                "h=1 is the wrong readout for this objective. Neither says anything about "
                "MECH-135 / INV-088 or about other budgets. cr_real(h) is arm-invariant per seed "
                "(measured from the environment), so the h=1 contrast is exactly cr_rollout(h=1); "
                "the ratio is a per-seed constant rescale, not a confound remover."
            ),
        },
        "source_substrate_entry": "SD-e1-rollout-consistency-training (ITEM 2, candidate 1, landed 2026-09-01)",
        "source_substrate_commit_item1": "ree-v3 26557a3758",
        "source_chip": "chip-20260901-sde1-item2-rollout-consistency-validation-exq",
        "source_design_doc": "sd_e1_rollout_consistency_training.md",
        "reference_runs": {
            "v3_exq_965": "v3_exq_965_sd_e1_item1_action_conditioning_validation_20260830T145908Z_v3",
            "v3_exq_968": "v3_exq_968_sd_e1_output_proj_residual_ab_20260901T162647Z_v3",
        },
        "hypothesis_space_qid": "inv088_evaluator_degeneracy_cause",
    }

    # Flatten per-cell metrics (dict-of-dicts kept as-is -- JSON-serialisable)
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
            "V3-EXQ-976: SD-e1-rollout-consistency-training ITEM 2 validation -- "
            "single-step vs rollout-consistency (flat / discounted) E1 objective on the "
            "ITEM 1 ON arm (diagnostic)"
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
        n_real = 15  # > MIN_REAL_SAMPLES_PER_HORIZON so the smoke exercises the label branch too
        seeds = seeds[:2]  # two seeds so the majority rule / noise ratio are exercised
        print("[V3-EXQ-976] SMOKE TEST MODE", flush=True)
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
        "rc_horizon": RC_HORIZON,
        "rc_decay": RC_DECAY,
        "rc_weight": RC_WEIGHT,
        "rc_non_vacuity_floor": RC_NON_VACUITY_FLOOR,
        "cr_real_floor": CR_REAL_FLOOR,
        "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "c3_var_threshold": C3_VAR_THRESHOLD,
        "lift_factor_abs_floor": LIFT_FACTOR_ABS_FLOOR,
        "lift_factor_noise_multiple": LIFT_FACTOR_NOISE_MULTIPLE,
        "lift_bar": LIFT_BAR,
        "grad_cos_vacuity_ceiling": GRAD_COS_VACUITY_CEILING,
        "grad_cos_sample_every": GRAD_COS_SAMPLE_EVERY,
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
        print("[V3-EXQ-976] SMOKE TEST COMPLETE", flush=True)
        for k in ["status", "non_degenerate", "degeneracy_reason"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
        print(f"  label: {result['interpretation']['label']}", flush=True)
        print(f"  arm_verdicts: {result.get('arm_verdicts')}", flush=True)
        print(f"  arm_verdicts_trained_horizon: {result.get('arm_verdicts_trained_horizon')}", flush=True)
        print(f"  arm_verdicts_vs_stateful_968_incumbent: {result.get('arm_verdicts_vs_stateful_968_incumbent')}", flush=True)
        print(f"  on_arm_vacuous: {result.get('on_arm_vacuous')}  h1_label_before_override: {result.get('h1_label_before_trained_horizon_override')}", flush=True)
        print(f"  grad_cos_on_cells: {result.get('grad_cos_on_cells')}", flush=True)
        print(f"  grad_cos_off_cells_diagnostic: {result.get('grad_cos_off_cells_diagnostic')}", flush=True)
        print(f"  depth0_trained_call_frac_per_cell: {result.get('depth0_trained_call_frac_per_cell')}", flush=True)
        print(f"  rc_vs_single_rel_diff_on_cells: {result.get('rc_vs_single_rel_diff_on_cells')}", flush=True)
        print(f"  e1_grad_step_gap_per_seed: {result.get('e1_grad_step_gap_per_seed')}", flush=True)
        print(f"  trained_loss_last_over_first_fifth_per_cell: {result.get('trained_loss_last_over_first_fifth_per_cell')}", flush=True)
        print(f"  noise_inflated_bar_diagnostic_not_applied: {result.get('noise_inflated_bar_diagnostic_not_applied')}", flush=True)
        for on_arm, d in (result.get("per_seed_lift") or {}).items():
            for k, v in d.items():
                print(
                    f"  [smoke] {on_arm}/{k}: cr_ratio_h1 single={v['cr_ratio_single_step']:.4e} "
                    f"on={v['cr_ratio_on_arm']:.4e} lift={v['relative_lift_on_over_single_step']:.3f} "
                    f"spread single={v['cr_rollout_spread_single_step']:.4e} "
                    f"on={v['cr_rollout_spread_on_arm']:.4e} "
                    f"var_rel_change={v['e1coe_score_var_rel_change']:.3f}",
                    flush=True,
                )

    # --- runner-conformance sentinel ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
