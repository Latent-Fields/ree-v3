"""
V3-EXQ-980 -- SD-e1-rollout-consistency-training fan-out sibling H-c
(readout-regime, eval-dynamics): does the Phase-4 evaluator's HYBRID E1/E2
rollout regime suppress the per-action divergence a rollout-consistency-trained
E1 objective produces at depth, relative to an E1-ALONE readout of the SAME
trained cells?

Claims: [] (deliberate; diagnostic; non_contributory by design -- read-across
only, per failure_autopsy_V3-EXQ-976_2026-09-02.json targets[0].fanout_recommendation.
Bears on MECH-135 / INV-088 but retests neither.)
DIAGNOSTIC. See EXPERIMENT_PURPOSE below.

WHY THIS RUN EXISTS
---------------------------------------------------------------------------
V3-EXQ-976 (confirmed autopsy, 2026-09-02) trained E1 with the multi-step
rollout_consistency_loss objective (candidate 1, flat and decay-0.5 forms) on
the SD-e1 ITEM-1-ON substrate and measured a weak h=1 positive with DEPTH
DAMPING at h>=2: the OFF (routed depth-0 single-step) arm's cr_ratio(h) grew
x2.6-3.8 from h=1 to h=5 while the ON arms grew only x1.1-1.75 (flat) / x0.5-1.7
(decay) -- 8/8 ON cells (2 ON arms x 4 seeds x {h=5,h=30}) grew LESS than the
OFF arm. That damping was read off 976's Phase-4 evaluator, which is a HYBRID
E1/E2 rollout: at every step past h=1 it RE-PRIMES z_self from
E2.predict_next_self and RE-COMPUTES E1's context/prior from the hybrid state
(driver v3_exq_976.py Phase 4, e1_deep.py forward()) -- a DIFFERENT map from
the one rollout_consistency_loss actually trains, which is E1's OWN
autoregressive predict_long_horizon: prior/context computed ONCE from the
initial state, E1's own predicted total fed back at every step, no E2 input
anywhere in the rollout. The two coincide EXACTLY at h=1 (a single depth-0
step from a sensed total) and DIVERGE from h=2 onward. 976 only ever measured
the hybrid readout, so whether the damping it found is a property of E1's
trained weights or an artifact of the consumer's own regime is UNMEASURED --
and the same hybrid scorer is the MECH-135 30-step endpoint consumer (108b's
C3, v3_exq_108b:451-463), so that consumer question stays open regardless of
how the withheld rollout-endpoint contrastive (ITEM 3 / H-f) is adjudicated.

THIS IS A SIBLING, NEVER A GATE. Per the autopsy's Step 7c red-team-accepted
revision: H-c does not gate the ITEM-3 contrastive build (H-f) -- the SD's
evaluator bars are read at h=1, where the hybrid and trained maps are the same
operation, so H-c cannot speak to whether the contrastive is licensed. H-c
answers a narrower, downstream question: can the MECH-135 30-step endpoint
consumer, AS BUILT, see what an E1 objective does to per-action divergence at
depth? User-confirmed 2026-09-02 (Q2 = yes, run alongside).

DESIGN: a RETRAIN of 976's exact shape (976's agents were never saved) -- same
4 arms, same 40 candidate sequences, same 4 seeds, same readiness gates, same
B1 depth-0 symmetrisation, same non-routing stateful anchor (ARM_single_step_stateful,
968's incumbent verbatim). The ONLY addition is a SECOND Phase-4 readout per
cell: E1DeepPredictor.predict_long_horizon(total_0, horizon=30, actions=seq) on
the identical 40 sequences at the identical checkpoints, reported BESIDE (not
instead of) the existing hybrid readout. Everything upstream of Phase 4
(Phases 0a/0b/1/2/3: encoder warmup, E1/E2 training, goal template, warmup
state, candidate-sequence generation) is byte-identical in code and RNG
consumption to v3_exq_976_sd_e1_item2_rollout_consistency_validation.py, so
this run's hybrid-readout numbers are a genuine reproduction check on 976's,
not merely a retrain under the same label.

ARMS (identical to 976; see that driver's docstring for the full B1/N1/N2
rationale -- not restated here):
  ARM_single_step           ROUTED incumbent, depth-0 symmetrised (B1)
  ARM_single_step_stateful  968's incumbent VERBATIM; NON-ROUTING anchor
  ARM_rc_flat                rollout_consistency_loss, H=5, decay=1.0
  ARM_rc_decay                rollout_consistency_loss, H=5, decay=0.5

DVs AND DECISION RULE (pre-registered; the discrimination this run adds)
---------------------------------------------------------------------------
For each ON arm, at each of TWO depth checkpoints DEPTH_HORIZONS_FOR_DAMPING =
{RC_HORIZON=5, rollout_horizon=30}, and under EACH readout (hybrid, e1_alone)
independently: "damped" = growth_on(h) < growth_off(h), where growth_X(h) =
cr_ratio_X(h) / cr_ratio_X(1) (X = the ON arm or the routed OFF arm, SAME
readout for both sides of the ratio -- never mixed across readouts). Per
(arm, readout): a MAJORITY of the {seed x horizon} cells (>= 5 of 8) must read
damped for the arm-readout cell to verdict "damped"; else "not_damped".
Per ON arm, compose the two readouts' verdicts:
  readout_regime_replicates_damping                  both readouts: damped
  readout_regime_discrepant_consumer_suppresses_divergence
                                                       hybrid damped, e1-alone
                                                       NOT damped -- the
                                                       consumer's own regime,
                                                       not E1's objective, is
                                                       what suppresses the
                                                       divergence at depth
  readout_regime_discrepant_hybrid_suppresses_divergence
                                                       e1-alone damped, hybrid
                                                       NOT damped (unexpected;
                                                       recorded, not predicted)
  readout_regime_neither_damped                       neither readout shows
                                                       damping (would diverge
                                                       from 976's own hybrid
                                                       finding; flagged, not
                                                       predicted)
Run-level label composes the two ON arms' per-arm discrimination:
  readout_regime_consistent_damping_replicates         BOTH arms replicate --
                                                        THIS is the declared
                                                        null: the readouts
                                                        agree, so 976's damping
                                                        finding is a property
                                                        of E1's trained
                                                        weights, not an
                                                        artifact of the hybrid
                                                        consumer's regime.
                                                        MECH-135's endpoint
                                                        retest may use either
                                                        readout.
  readout_regime_discrepant_consumer_suppresses_divergence
                                                        >= 1 arm shows the
                                                        consumer-suppresses
                                                        pattern and none show
                                                        the opposite --
                                                        DISCRIMINATING: the
                                                        MECH-135 endpoint
                                                        retest MUST use an
                                                        E1-alone readout, not
                                                        the hybrid scorer, or
                                                        it will under-read a
                                                        divergence-preserving
                                                        objective's effect.
  readout_regime_discrepant_hybrid_suppresses_divergence
                                                        the opposite pattern
                                                        (recorded; unexpected)
  readout_regime_neither_damped_unexpected             BOTH arms read
                                                        neither_damped under
                                                        both readouts (would
                                                        diverge from 976's own
                                                        hybrid finding;
                                                        flagged, not predicted)
  readout_regime_mixed_across_arms                     any other combination,
                                                        INCLUDING a
                                                        contradictory pair (one
                                                        arm each direction) --
                                                        never composes to
                                                        either directional
                                                        label
  readout_regime_undetermined                          a decision-horizon cell
                                                        was non-finite despite
                                                        readiness passing
                                                        (defense-in-depth;
                                                        should be unreachable --
                                                        see
                                                        cr_ratio_finite_at_decision_horizons)
  substrate_not_ready_requeue                          any readiness
                                                        precondition unmet

WHAT THIS DOES NOT LICENSE. Neither direction bears on Q1 (whether 976
licenses the withheld rollout-endpoint contrastive, H-f) -- the SD's bars are
read at h=1, where hybrid and trained maps coincide regardless of this run's
finding. This run only bears on how the MECH-135 30-step endpoint consumer
should be measured in any successor.

READINESS PRECONDITIONS: 976's nine verbatim (encoder_trained;
real_zworld_nondegenerate_h1; no_missing_action_calls;
direct_action_supply_fraction; cr_ratio_h1_finite [hybrid];
e1_grad_steps_matched; rc_objective_non_vacuous; at_least_one_on_arm_non_vacuous;
trained_calls_at_depth0) plus TWO this design needs:
  cr_ratio_e1alone_h1_finite  cr_ratio(h=1) under the E1-ALONE readout is
                              finite and positive on every cell -- same-statistic
                              discipline extended to the new readout (at h=1
                              the two readouts are the same operation, so this
                              should track cr_ratio_h1_finite closely; recorded
                              independently in case the two computations
                              diverge for a non-scientific reason, e.g. a
                              hidden-state lifecycle bug).
  cr_ratio_finite_at_decision_horizons  cr_ratio(h) under BOTH readouts is
                              finite and positive on every cell at every
                              horizon in DEPTH_HORIZONS_FOR_DAMPING (h=5, h=30
                              by default) -- red-team (fable) Finding B: the
                              two h=1-only gates above do not cover the
                              horizons the damping VERDICT actually reads, so
                              a NaN cell there (e.g. a real-sample dropout, or
                              an untrained-at-depth autoregressive rollout)
                              could pass every other gate while silently
                              starving the verdict's sample size.

DV-SYMMETRY / READOUT NOTE (one line per readout). cr_ratio(h) = spread /
||centroid|| over 40 candidate rollout endpoints, divided by the arm-invariant
CR_real(h) (identical across both readouts and both training and Phase 4b are
unaffected by which Phase-4 readout is used). The HYBRID readout and the
E1-ALONE readout are two DIFFERENT deterministic functions of the SAME trained
E1/E2 weights and the SAME 40 action sequences -- neither a common rescaling
nor a permutation of either the candidate index or the readout choice, so a
discrepancy between them is not a DV-symmetry artifact; it is a genuine
measurement of which map's rollout the objective's effect is visible on.

GOV-REUSE-1 (Step 2.4): decisive readout = the AGREEMENT (or disagreement)
between two cr_ratio(h) growth-damping verdicts, one under a readout that has
NEVER been computed in this lineage (E1-alone predict_long_horizon at h>1;
every prior driver in the 954/965/968/976 lineage uses only the hybrid
scorer). Not recoverable from any existing manifest -> run.

STEP 2.5c (recorded, not blocking): exercises e1_deep.py::predict_long_horizon
directly (SD-e1-rollout-consistency-training -- this run is diagnostic
read-across on that same SD, not a new dependency) and ContextMemory.write
(contextmemory-write-path-addressing-degeneracy, corrupting,
implemented_pending_validation) identically to 976 -- see that driver's
identical note; the between-readout contrast this run routes on is unaffected
by ContextMemory's absolute-level bias since both readouts share it.

SLEEP DRIVER: none (no sleep flags set).
red-team (fable): CONTESTED, both findings accepted and fixed before queueing.
F-A (Family 3, verdict-grid): the run-level label composition was missing an
"and none show the opposite" conjunct (docstring line ~108 already stated this
requirement; the code did not implement it) -- a contradictory arm pair (one
consumer-suppresses, one hybrid-suppresses) would have emitted the
consumer-suppresses headline label instead of readout_regime_mixed_across_arms.
Fixed with explicit exclusion conjuncts in both directional branches.
F-B (Family 4, self-certifying gate): the h=1-only finiteness preconditions did
not cover DEPTH_HORIZONS_FOR_DAMPING (h=5, h=30), so a NaN cell at a decision
horizon could pass every readiness gate while _damping_verdict's n_total
silently shrank -- at n_total=0, producing a definite "not_damped" from zero
evidence, up to manufacturing the discriminating label from a broken E1-alone
rollout. Fixed with a new precondition (cr_ratio_finite_at_decision_horizons)
gating the SAME horizons/readouts the verdict reads, plus a defense-in-depth
n_total floor in _damping_verdict that verdicts "undetermined" (composing to
run-level readout_regime_undetermined) rather than a false "not_damped" if
ever reached. Families 1 (manipulation->DV) and 2 (criterion construction)
cleared with no finding.
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


EXPERIMENT_TYPE = "v3_exq_980_sd_e1_h1c_readout_regime_e1_alone"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None

# agent_construction_before_seed lint exemption -- identical shape and
# justification to V3-EXQ-976/965/954.
AGENT_SEED_ORDER_EXEMPT = (
    "Every within-cell comparison (action-vs-action, horizon-vs-horizon, "
    "readout-vs-readout) is scored off the literal same agent object; "
    "cross-arm comparisons are the A/B itself and are seed-matched via "
    "arm_cell()'s RNG reset."
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds -- identical to 976 except where noted.
# ---------------------------------------------------------------------------
CR_REAL_FLOOR = 1e-4
ZWORLD_P0_EPISODES = 60
N_REAL_SAMPLES = 40
MIN_REAL_SAMPLES_PER_HORIZON = 10
HORIZON_CHECKPOINTS_FULL = [1, 2, 3, 5, 10, 20, 30]

RC_HORIZON = 5
RC_DECAY = 0.5
RC_WEIGHT = 1.0
RC_NON_VACUITY_FLOOR = 0.01        # RECORDED diagnostic only, not applied (inherited from 976)
GRAD_COS_VACUITY_CEILING = 0.95
GRAD_COS_SAMPLE_EVERY = 10

OFF_ARM = "ARM_single_step"
STATEFUL_ARM = "ARM_single_step_stateful"
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
DEPTH0_ARMS = [OFF_ARM] + ON_ARMS

ANCHOR_REACHABILITY_EXEMPT = (
    "e1_grad_steps_matched and rc_objective_non_vacuous are measured on every cell "
    "from the training loop itself (no separate positive-control run); reachability "
    "inherited from V3-EXQ-976, whose --dry-run smoke (2026-09-02) demonstrated it "
    "on byte-identical Phase 0a/0b code."
)
SEEDS_DEFAULT = [42, 123, 7, 2024]


# ---------------------------------------------------------------------------
# Helpers -- byte-identical to V3-EXQ-976 Phases 0a/0b/1/2/3.
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
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
    assert bool(config.e1.e1_rollout_consistency_enabled) == bool(arm_cfg["e1_rollout_consistency_enabled"]), arm
    assert int(config.e1.e1_rollout_consistency_horizon) == int(arm_cfg["rc_horizon"]), arm
    assert abs(float(config.e1.e1_rollout_consistency_horizon_weights_decay) - float(arm_cfg["rc_decay"])) < 1e-12, arm
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0a: SD-070 sanctioned z_world encoder warmup (byte-identical to 976).
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_980 P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context="v3_exq_980_sd_e1_h1c_readout_regime_e1_alone",
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 training. Byte-identical to 976 (same RNG streams,
# same objective/schedule per arm) so this run's hybrid-readout numbers are a
# genuine reproduction check on 976's.
# ---------------------------------------------------------------------------

def _single_step_loss_state_preserving(
    agent: REEAgent, initial: torch.Tensor, action: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Single-step teacher-forced loss from a ZERO hidden state, save/restore
    lifecycle -- identical to V3-EXQ-976 (red-team B1 symmetrisation)."""
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
    """Byte-identical to V3-EXQ-976's _train_agent -- see that driver's
    docstring for the full per-arm objective description. Reproduced here
    (not imported) so this file's own arm_fingerprint substrate_hash covers
    the exact code that trained the cell, per the arm-fingerprint convention."""
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    arm_cfg = ARM_CONFIGS[arm]
    loss_kind = str(arm_cfg["e1_loss"])
    H = int(RC_HORIZON)
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

            if latent_prev is not None:
                opt_e2.zero_grad()
                z_self_pred = agent.e2.predict_next_self(latent_prev.z_self.detach(), action_prev)
                e2_loss = F.mse_loss(z_self_pred, latent_curr.z_self.detach())
                e2_loss.backward()
                opt_e2.step()
                ep_loss_e2 += e2_loss.item()
                stats["n_e2_grad_steps"] += 1

            if len(totals) >= H + 1:
                i = len(totals) - 1 - H
                initial = totals[i]
                targets = torch.stack(totals[i + 1:i + 1 + H], dim=1)
                acts = torch.stack(actions[i:i + H], dim=1)
                sample_grad = (stats["n_windows"] % GRAD_COS_SAMPLE_EVERY == 0)
                opt_e1.zero_grad()
                if agent.e1._hidden_state is None:
                    stats["n_depth0_trained_calls"] += 1
                if loss_kind in ("single_step", "single_step_stateful"):
                    if loss_kind == "single_step":
                        trained = _single_step_loss_state_preserving(agent, initial, actions[i], totals[i + 1])
                    else:
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
# Phase 1/2/3: goal template, warmup state, candidate sequences. Byte-identical
# to 976 (same RNG offsets: seed, seed+1000, seed+500).
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
# Phase 4 (NON-GATING; recorded): TWO readouts per sequence.
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
    """The HYBRID Phase-4 evaluator readout -- byte-identical to
    V3-EXQ-976's _score_sequence_e1coe_multi_horizon: per step, z_world comes
    from a single-step E1 call and z_self is RE-PRIMED from
    E2.predict_next_self. This is the SAME consumer as the MECH-135 30-step
    endpoint (108b's C3, v3_exq_108b:451-463)."""
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
    """H-c sibling probe (V3-EXQ-980): the E1-ALONE rollout readout -- E1's
    OWN autoregressive predict_long_horizon, the map rollout_consistency_loss
    actually trains (e1_deep.py:880-990): prior/context computed ONCE from the
    initial state via reset_hidden_state(), then E1's own predicted total fed
    back at every step. NO E2 call anywhere in this function -- that is
    exactly the difference from _score_sequence_hybrid_multi_horizon. One
    forward call covers the whole horizon (cheaper than the hybrid's
    per-step loop), so this is added cost, not a bottleneck."""
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


def _collect_real_zworld_sample_multi_horizon(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int,
    checkpoints: List[int],
) -> Dict[int, List[torch.Tensor]]:
    """Byte-identical to 976 (own seed offset +3000)."""
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
    stacked = torch.cat(vectors, dim=0)
    centroid = stacked.mean(dim=0, keepdim=True)
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr, "n": len(vectors)}


def _one_step_action_divergence(
    agent: REEAgent,
    z_self_0: torch.Tensor,
    z_world_0: torch.Tensor,
    self_dim: int,
    e1_call_counter: Dict[str, int],
) -> Dict[str, Any]:
    """Byte-identical to 976's Phase 4c (positive-control cross-reference)."""
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
    print(f"\n[EXQ-980] seed={seed} arm={arm}", flush=True)
    print(f"Seed {seed} Condition {arm}", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim, arm)
    e1_call_counter = {"n_e1_calls": 0, "n_e1_calls_nonzero_action": 0}

    print(f"[EXQ-980] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    print(f"[EXQ-980] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    train_stats = _train_agent(agent, env, seed, n_train_episodes, steps_per_episode, e1_call_counter, arm)

    print("[EXQ-980] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    print("[EXQ-980] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    print(f"[EXQ-980] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 4 (NON-GATING): score sequences under BOTH readouts at every
    # horizon checkpoint. The hybrid readout loop runs first (per-step),
    # matching 976's exact call order and RNG-irrelevant control flow; the
    # e1-alone readout runs second, per sequence, immediately after.
    print(f"[EXQ-980] Phase 4: scoring sequences at horizons {checkpoints} (hybrid + e1-alone, non-gating)...", flush=True)
    scores_hybrid_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_hybrid_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}
    scores_e1alone_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_e1alone_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for i, seq in enumerate(seqs):
        per_h_hybrid = _score_sequence_hybrid_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints,
            e1_call_counter,
        )
        for h, (score, endpoint) in per_h_hybrid.items():
            scores_hybrid_by_h[h].append(score)
            endpoints_hybrid_by_h[h].append(endpoint)

        per_h_e1alone = _score_sequence_e1_alone_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints,
            e1_call_counter,
        )
        for h, (score, endpoint) in per_h_e1alone.items():
            scores_e1alone_by_h[h].append(score)
            endpoints_e1alone_by_h[h].append(endpoint)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    e1coe_score_var_hybrid_by_h: Dict[int, float] = {}
    cr_rollout_hybrid_by_h: Dict[int, Dict[str, float]] = {}
    e1coe_score_var_e1alone_by_h: Dict[int, float] = {}
    cr_rollout_e1alone_by_h: Dict[int, Dict[str, float]] = {}
    for h in checkpoints:
        st = torch.tensor(scores_hybrid_by_h[h])
        e1coe_score_var_hybrid_by_h[h] = float(st.var().item()) if len(scores_hybrid_by_h[h]) > 1 else 0.0
        cr_rollout_hybrid_by_h[h] = _contrast_ratio(endpoints_hybrid_by_h[h])

        se = torch.tensor(scores_e1alone_by_h[h])
        e1coe_score_var_e1alone_by_h[h] = float(se.var().item()) if len(scores_e1alone_by_h[h]) > 1 else 0.0
        cr_rollout_e1alone_by_h[h] = _contrast_ratio(endpoints_e1alone_by_h[h])

        print(
            f"  h={h:>2d}: HYBRID var={e1coe_score_var_hybrid_by_h[h]:.6e} "
            f"CR={cr_rollout_hybrid_by_h[h]['contrast_ratio']:.6e} | "
            f"E1ALONE var={e1coe_score_var_e1alone_by_h[h]:.6e} "
            f"CR={cr_rollout_e1alone_by_h[h]['contrast_ratio']:.6e}",
            flush=True,
        )

    print(f"[EXQ-980] Phase 4b: sampling {n_real_samples} real trajectories at horizons {checkpoints}...", flush=True)
    real_samples_by_h = _collect_real_zworld_sample_multi_horizon(
        agent, env, seed, n_real_samples, checkpoints,
    )
    cr_real_by_h: Dict[int, Dict[str, float]] = {}
    cr_ratio_hybrid_by_h: Dict[int, float] = {}
    cr_ratio_e1alone_by_h: Dict[int, float] = {}
    for h in checkpoints:
        samples = real_samples_by_h[h]
        if len(samples) >= 2:
            cr_real_by_h[h] = _contrast_ratio(samples)
        else:
            cr_real_by_h[h] = {"spread": 0.0, "centroid_norm": 0.0, "contrast_ratio": float("nan"), "n": len(samples)}
        cr_real = cr_real_by_h[h]["contrast_ratio"]
        cr_roll_hybrid = cr_rollout_hybrid_by_h[h]["contrast_ratio"]
        cr_roll_e1alone = cr_rollout_e1alone_by_h[h]["contrast_ratio"]
        cr_ratio_hybrid_by_h[h] = (cr_roll_hybrid / cr_real) if (cr_real == cr_real and cr_real > 0) else float("nan")
        cr_ratio_e1alone_by_h[h] = (cr_roll_e1alone / cr_real) if (cr_real == cr_real and cr_real > 0) else float("nan")
        print(
            f"  h={h:>2d}: CR_real={cr_real:.6e} (n={cr_real_by_h[h]['n']}) "
            f"ratio_hybrid={cr_ratio_hybrid_by_h[h]:.6e} ratio_e1alone={cr_ratio_e1alone_by_h[h]:.6e}",
            flush=True,
        )

    print("[EXQ-980] Phase 4c: direct-channel one-step per-action divergence probe...", flush=True)
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
        "e1coe_score_var_hybrid_by_h": e1coe_score_var_hybrid_by_h,
        "cr_rollout_hybrid_by_h": cr_rollout_hybrid_by_h,
        "e1coe_score_var_e1alone_by_h": e1coe_score_var_e1alone_by_h,
        "cr_rollout_e1alone_by_h": cr_rollout_e1alone_by_h,
        "cr_real_by_h": cr_real_by_h,
        "cr_ratio_hybrid_by_h": cr_ratio_hybrid_by_h,
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
    depth_horizons_for_damping = sorted(set(h for h in [RC_HORIZON, rollout_horizon] if h in checkpoints))

    cell_config_slice = {
        "world_dim": world_dim, "self_dim": self_dim,
        "n_train_episodes": n_train_episodes, "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences, "rollout_horizon": rollout_horizon,
        "n_warmup_steps": n_warmup_steps, "goal_max_steps": goal_max_steps,
        "zworld_p0_episodes": zworld_p0_episodes, "n_real_samples": n_real_samples,
        "env_kwargs": _env_kwargs(),
        "rc_horizon": RC_HORIZON, "rc_decay": RC_DECAY, "rc_weight": RC_WEIGHT,
        "rc_non_vacuity_floor": RC_NON_VACUITY_FLOOR,
        "cr_real_floor": CR_REAL_FLOOR, "zworld_p0_episodes_default": ZWORLD_P0_EPISODES,
        "n_real_samples_default": N_REAL_SAMPLES,
        "min_real_samples_per_horizon": MIN_REAL_SAMPLES_PER_HORIZON,
        "horizon_checkpoints_full": list(HORIZON_CHECKPOINTS_FULL),
        "grad_cos_vacuity_ceiling": GRAD_COS_VACUITY_CEILING,
        "grad_cos_sample_every": GRAD_COS_SAMPLE_EVERY,
        "stateful_arm": STATEFUL_ARM,
        "alpha_world": 0.9, "alpha_self": 0.3, "unified_latent_mode": False,
        "train_lr_e1": 1e-3, "train_lr_e2": 1e-3,
        # this driver adds a second Phase-4 readout with no new trained hyperparameter,
        # but declaring it keeps the config_slice honest about what a cell's readouts depend on
        "adds_e1_alone_readout": True,
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

    # ---- Readiness (P0) -- identical to 976, plus one new precondition ----
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

    cr_ratio_hybrid_h1_values = [
        by_arm_seed[(arm, seed)]["cr_ratio_hybrid_by_h"].get(1, float("nan"))
        for arm in ARM_ORDER for seed in seeds
    ]
    p_cr_ratio_hybrid_h1_finite = all(v == v and v > 0 for v in cr_ratio_hybrid_h1_values)
    min_cr_ratio_hybrid_h1 = min((v for v in cr_ratio_hybrid_h1_values if v == v), default=float("nan"))

    cr_ratio_e1alone_h1_values = [
        by_arm_seed[(arm, seed)]["cr_ratio_e1alone_by_h"].get(1, float("nan"))
        for arm in ARM_ORDER for seed in seeds
    ]
    p_cr_ratio_e1alone_h1_finite = all(v == v and v > 0 for v in cr_ratio_e1alone_h1_values)
    min_cr_ratio_e1alone_h1 = min((v for v in cr_ratio_e1alone_h1_values if v == v), default=float("nan"))

    # red-team (fable) Finding B: cr_ratio_hybrid_h1_finite / cr_ratio_e1alone_h1_finite
    # above only gate h=1 -- but the damping VERDICT is read exclusively at
    # depth_horizons_for_damping (h=5, h=30 by default). A NaN/non-positive cell
    # at a DECISION horizon (e.g. a real-sample dropout from an episode ending
    # early, or an untrained-at-depth autoregressive rollout) would otherwise pass
    # every readiness gate while silently starving _damping_verdict's n_total --
    # at the limit, producing a confident "not_damped" from zero evidence. Gate
    # finiteness at the SAME horizons and readouts the verdict actually reads,
    # for every arm (OFF included, since every damping cell reads the OFF arm's
    # value too) and both readouts.
    cr_ratio_at_decision_horizons_values = [
        by_arm_seed[(arm, seed)][f"cr_ratio_{readout}_by_h"].get(h, float("nan"))
        for arm in ARM_ORDER for seed in seeds
        for h in depth_horizons_for_damping for readout in ("hybrid", "e1alone")
    ]
    p_cr_ratio_finite_at_decision_horizons = all(
        v == v and v > 0 for v in cr_ratio_at_decision_horizons_values
    )
    min_cr_ratio_at_decision_horizons = min(
        (v for v in cr_ratio_at_decision_horizons_values if v == v), default=float("nan")
    )

    grad_step_gap_per_seed: Dict[str, int] = {}
    for seed in seeds:
        counts = [int(by_arm_seed[(arm, seed)]["n_e1_grad_steps"]) for arm in ARM_ORDER]
        grad_step_gap_per_seed[f"seed{seed}"] = int(max(counts) - min(counts))
    max_grad_step_gap = max(grad_step_gap_per_seed.values()) if grad_step_gap_per_seed else 0
    p_grad_steps_matched = max_grad_step_gap == 0

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
            "description": "At least one split_encoder.world_encoder tensor moved during Phase 0a, every cell.",
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
                f"CR_real(h=1) is finite, positive, and backed by at least {MIN_REAL_SAMPLES_PER_HORIZON} "
                "surviving real samples, every cell."
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
                "E1DeepPredictor._action_cond_missing_calls is 0 on every cell, including this "
                "driver's new E1-alone predict_long_horizon calls."
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
            "description": "Fraction of E1 calls (both readouts + training) with a genuine one-hot actions=.",
            "measured": min_direct_supply_fraction,
            "threshold": 0.999,
            "direction": "lower",
            "met": p_direct_action_supply,
        },
        {
            "name": "cr_ratio_hybrid_h1_finite",
            "kind": "readiness",
            "description": "cr_ratio(h=1) under the HYBRID readout is finite and positive on every cell.",
            "measured": min_cr_ratio_hybrid_h1,
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "met": p_cr_ratio_hybrid_h1_finite,
        },
        {
            "name": "cr_ratio_e1alone_h1_finite",
            "kind": "readiness",
            "description": (
                "cr_ratio(h=1) under the NEW E1-ALONE readout is finite and positive on every cell "
                "-- same-statistic discipline extended to this driver's own readout (V3-EXQ-980)."
            ),
            "measured": min_cr_ratio_e1alone_h1,
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "met": p_cr_ratio_e1alone_h1_finite,
        },
        {
            "name": "cr_ratio_finite_at_decision_horizons",
            "kind": "readiness",
            "description": (
                "cr_ratio(h) under BOTH readouts is finite and positive on every (arm, seed) cell "
                "at every horizon in depth_horizons_for_damping (the horizons the damping VERDICT "
                "actually reads, not just h=1) -- red-team (fable) Finding B: a NaN/non-positive "
                "cell here would silently starve _damping_verdict's n_total while every other "
                "readiness gate stayed green, up to producing a definite 'not_damped' from zero "
                "evidence at n_total=0."
            ),
            "measured": min_cr_ratio_at_decision_horizons,
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "depth_horizons_checked": depth_horizons_for_damping,
            "met": p_cr_ratio_finite_at_decision_horizons,
        },
        {
            "name": "e1_grad_steps_matched",
            "kind": "readiness",
            "description": "Every arm took the same number of E1 optimiser steps at a seed.",
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
                "On every ON cell, the mean cosine similarity between the E1 GRADIENT of the trained "
                f"rollout-consistency objective and the incumbent single-step objective is <= {GRAD_COS_VACUITY_CEILING}."
            ),
            "measured": float(max_grad_cos_on),
            "threshold": float(GRAD_COS_VACUITY_CEILING),
            "direction": "upper",
            "comparator": "<=",
            "control": "both objectives' gradients taken on identical windows in every arm; ON arms only gate",
            "offending_cell": worst_cos_cell,
            "met": p_rc_non_vacuous_all,
            "kind_note": (
                "INFORMATIONAL for adjudication of the WHOLE run; applied PER ON ARM. The whole-run "
                "readiness conjunct is at_least_one_on_arm_non_vacuous."
            ),
            "per_arm_max_grad_cos": on_arm_grad_cos_max,
            "per_arm_vacuous": on_arm_vacuous,
        },
        {
            "name": "at_least_one_on_arm_non_vacuous",
            "kind": "readiness",
            "description": "At least one ON arm clears the per-arm gradient-cosine vacuity gate on every seed.",
            "measured": float(sum(1 for v in on_arm_vacuous.values() if not v)),
            "threshold": 1.0,
            "direction": "lower",
            "met": p_some_on_arm_non_vacuous,
        },
        {
            "name": "trained_calls_at_depth0",
            "kind": "readiness",
            "description": (
                "Fraction of trained E1 calls that started from a zero LSTM hidden state, minimum over "
                "ALL cells -- must be 1.0 (B1 depth-0 symmetrisation, inherited from 976)."
            ),
            "measured": float(min_depth0),
            "threshold": 1.0,
            "direction": "lower",
            "control": (
                "every ROUTED arm's trained call routed through the same save/reset/restore lifecycle; "
                "ARM_single_step_stateful is EXCLUDED by design (the non-routing anchor)"
            ),
            "stateful_anchor_depth0_frac_per_seed": stateful_depth0_per_seed,
            "met": p_depth0,
        },
    ]

    non_degenerate = bool(
        p_encoder_trained_met and p_real_nondegenerate_met
        and p_no_missing_action_calls and p_direct_action_supply
        and p_cr_ratio_hybrid_h1_finite and p_cr_ratio_e1alone_h1_finite
        and p_cr_ratio_finite_at_decision_horizons
        and p_grad_steps_matched and p_some_on_arm_non_vacuous and p_depth0
    )

    majority_seeds = len(seeds) // 2 + 1

    def _damping_verdict(readout_key: str) -> Dict[str, Any]:
        """Per ON arm: majority-of-{seed x horizon}-cells rule for whether the
        arm's cr_ratio(h) growth from h=1 is LESS than the routed OFF arm's,
        under the given readout ('hybrid' or 'e1alone'). This is the
        arm-readout-level building block for the run's discrimination."""
        field = f"cr_ratio_{readout_key}_by_h"
        per_arm: Dict[str, Any] = {}
        for on_arm in ON_ARMS:
            cells: Dict[str, Any] = {}
            n_damped = 0
            n_total = 0
            for seed in seeds:
                on_h1 = by_arm_seed[(on_arm, seed)][field].get(1, float("nan"))
                off_h1 = by_arm_seed[(OFF_ARM, seed)][field].get(1, float("nan"))
                for h in depth_horizons_for_damping:
                    on_h = by_arm_seed[(on_arm, seed)][field].get(h, float("nan"))
                    off_h = by_arm_seed[(OFF_ARM, seed)][field].get(h, float("nan"))
                    vals = (on_h1, off_h1, on_h, off_h)
                    if not all(v == v and v > 0 for v in vals):
                        continue
                    growth_on = on_h / on_h1
                    growth_off = off_h / off_h1
                    damped = growth_on < growth_off
                    n_total += 1
                    n_damped += int(damped)
                    cells[f"seed{seed}_h{h}"] = {
                        "growth_on": growth_on, "growth_off": growth_off, "damped": damped,
                    }
            expected_total = len(seeds) * len(depth_horizons_for_damping)
            if n_total < expected_total:
                # red-team (fable) Finding B: a NaN/non-positive cell at a
                # DECISION horizon silently shrunk n_total and (at n_total==0)
                # produced a definite "not_damped" from zero evidence. The
                # cr_ratio_finite_at_decision_horizons precondition below
                # should make this unreachable when non_degenerate is True;
                # this is the defense-in-depth floor for that guarantee.
                verdict = "undetermined"
                majority = None
            else:
                majority = (n_total // 2 + 1)
                verdict = "damped" if n_damped >= majority else "not_damped"
            per_arm[on_arm] = {
                "verdict": verdict, "n_damped": n_damped, "n_total": n_total,
                "expected_total": expected_total, "majority_needed": majority, "cells": cells,
            }
        return per_arm

    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        unmet_names = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = "P0 readiness unmet: " + ", ".join(unmet_names)
        status = "FAIL"
        evidence_direction = "non_contributory"
        damping_hybrid: Dict[str, Any] = {}
        damping_e1alone: Dict[str, Any] = {}
        arm_readout_discrimination: Dict[str, str] = {}
        criteria = []
        criteria_non_degenerate = {
            f"C1_readout_regime_damping_replication_{arm.replace('ARM_', '')}": False for arm in ON_ARMS
        }
    else:
        degeneracy_reason = None
        damping_hybrid = _damping_verdict("hybrid")
        damping_e1alone = _damping_verdict("e1alone")

        arm_readout_discrimination = {}
        for on_arm in ON_ARMS:
            vh = damping_hybrid[on_arm]["verdict"]
            ve = damping_e1alone[on_arm]["verdict"]
            if vh == "undetermined" or ve == "undetermined":
                disc = "readout_regime_undetermined"
            elif vh == "damped" and ve == "damped":
                disc = "readout_regime_replicates_damping"
            elif vh == "damped" and ve == "not_damped":
                disc = "readout_regime_discrepant_consumer_suppresses_divergence"
            elif vh == "not_damped" and ve == "damped":
                disc = "readout_regime_discrepant_hybrid_suppresses_divergence"
            else:
                disc = "readout_regime_neither_damped"
            arm_readout_discrimination[on_arm] = disc

        discs = set(arm_readout_discrimination.values())
        if "readout_regime_undetermined" in discs:
            # defense-in-depth: should be unreachable once non_degenerate is True,
            # because the cr_ratio_finite_at_decision_horizons precondition below
            # gates the whole run first -- see _damping_verdict's n_total floor.
            label = "readout_regime_undetermined"
        elif discs == {"readout_regime_replicates_damping"}:
            label = "readout_regime_consistent_damping_replicates"
        elif (
            "readout_regime_discrepant_consumer_suppresses_divergence" in discs
            and "readout_regime_discrepant_hybrid_suppresses_divergence" not in discs
        ):
            # red-team (fable) Finding A: a contradictory pair (one arm
            # consumer-suppresses, the other hybrid-suppresses) must NOT
            # compose to either directional label -- the docstring's own
            # run-level definition requires "and none show the opposite".
            label = "readout_regime_discrepant_consumer_suppresses_divergence"
        elif (
            "readout_regime_discrepant_hybrid_suppresses_divergence" in discs
            and "readout_regime_discrepant_consumer_suppresses_divergence" not in discs
        ):
            label = "readout_regime_discrepant_hybrid_suppresses_divergence"
        elif discs == {"readout_regime_neither_damped"}:
            label = "readout_regime_neither_damped_unexpected"
        else:
            label = "readout_regime_mixed_across_arms"

        status = "PASS"  # diagnostic discrimination -- informative in every direction
        evidence_direction = "non_contributory"

        criteria = []
        for on_arm in ON_ARMS:
            dh = damping_hybrid[on_arm]
            de = damping_e1alone[on_arm]
            criteria.append({
                "name": f"C1_readout_regime_damping_replication_{on_arm.replace('ARM_', '')}",
                "load_bearing": True,
                "passed": True,  # this criterion CLASSIFIES, it does not gate PASS/FAIL
                "hybrid_verdict": dh["verdict"],
                "e1alone_verdict": de["verdict"],
                "discrimination": arm_readout_discrimination[on_arm],
                "measured_hybrid_damped_fraction": (dh["n_damped"] / dh["n_total"]) if dh["n_total"] else float("nan"),
                "measured_e1alone_damped_fraction": (de["n_damped"] / de["n_total"]) if de["n_total"] else float("nan"),
                "threshold": 0.5,
                "statement": (
                    f"Depth-growth damping of {on_arm} relative to {OFF_ARM} in cr_ratio(h) at "
                    f"h in {depth_horizons_for_damping}, majority-of-cells rule, compared under the "
                    "HYBRID Phase-4 evaluator readout vs the E1-ALONE readout. Declared null: both "
                    "readouts agree. Discriminating: hybrid damped but e1-alone NOT damped -> the "
                    "consumer's regime, not E1's objective, suppresses divergence at depth."
                ),
            })
        criteria_non_degenerate = {
            f"C1_readout_regime_damping_replication_{arm.replace('ARM_', '')}": non_degenerate for arm in ON_ARMS
        }

    print(f"\n[EXQ-980] Label: {label}", flush=True)
    print(f"[EXQ-980] Status: {status}", flush=True)

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
        "depth_horizons_for_damping": depth_horizons_for_damping,
        "n_warmup_steps": n_warmup_steps,
        "zworld_p0_episodes": zworld_p0_episodes,
        "n_real_samples": n_real_samples,
        "rc_horizon": RC_HORIZON,
        "rc_decay": RC_DECAY,
        "rc_weight": RC_WEIGHT,
        "registered_cr_real_floor": CR_REAL_FLOOR,
        "registered_majority_seeds": majority_seeds,
        "registered_rc_non_vacuity_floor": RC_NON_VACUITY_FLOOR,
        "registered_grad_cos_vacuity_ceiling": GRAD_COS_VACUITY_CEILING,
        "registered_grad_cos_sample_every": GRAD_COS_SAMPLE_EVERY,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "damping_hybrid": damping_hybrid,
        "damping_e1alone": damping_e1alone,
        "arm_readout_discrimination": arm_readout_discrimination,
        "on_arm_vacuous": on_arm_vacuous,
        "on_arm_grad_cos_max": on_arm_grad_cos_max,
        "stateful_arm": STATEFUL_ARM,
        "depth0_arms": DEPTH0_ARMS,
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
                "Per ON arm, per readout (hybrid, e1alone): 4 seeds x 2 depth horizons = 8 "
                "{seed x horizon} cells; a majority (>= 5 of 8) must show growth_on(h) < "
                "growth_off(h) (same readout on both sides) for that arm-readout to verdict "
                "'damped'. A cell with a non-finite/non-positive input is excluded from n_total "
                "(never counted as non-damped); if fewer than all 8 cells survive, the arm-readout "
                "verdicts 'undetermined' rather than a false 'not_damped' -- gated in practice by "
                "the cr_ratio_finite_at_decision_horizons precondition. Per ON arm, compose the "
                "hybrid and e1alone verdicts into a discrimination (replicates / consumer-suppresses "
                "/ hybrid-suppresses / neither / undetermined). Run-level label composes the two ON "
                "arms' discriminations (see module docstring for the full composition; a "
                "contradictory pair -- one arm each direction -- composes to "
                "readout_regime_mixed_across_arms, not to a directional label). No criterion "
                "gates PASS/FAIL once readiness is met -- the run is a classification (every "
                "per-arm criterion below is load_bearing because EACH one independently reports "
                "a genuine discrimination outcome that must be read, not because the run-level "
                "label accepts on any single one clearing a bar)."
            ),
            "readout_symmetry_note": (
                "cr_ratio(h) = cr_rollout(h)/cr_real(h); cr_real(h) is identical across both "
                "readouts (Phase 4b, environment-derived, unaffected by which Phase-4 scorer runs). "
                "The hybrid and e1-alone readouts are two different deterministic functions of the "
                "SAME trained E1/E2 weights and the SAME 40 action sequences -- neither a common "
                "rescaling nor a permutation of the readout choice, so a discrepancy between them is "
                "a genuine measurement of which map the objective's depth effect is visible on, not "
                "a DV-symmetry artifact."
            ),
            "what_replication_means": (
                "readout_regime_consistent_damping_replicates: 976's damping finding is a property "
                "of E1's trained weights, visible on E1's own rollout map, not an artifact of the "
                "hybrid consumer's regime. Does NOT license or withhold the ITEM-3 rollout-endpoint "
                "contrastive (H-f) -- that licence is read at h=1 from 976, unaffected by this run."
            ),
            "what_discrepancy_means": (
                "readout_regime_discrepant_consumer_suppresses_divergence: the hybrid Phase-4 "
                "evaluator (and by extension the MECH-135 108b:451-463 30-step endpoint consumer, "
                "the SAME scorer) suppresses per-action divergence that IS present in E1's own "
                "rollout map at depth. Any successor measuring the MECH-135 endpoint question must "
                "use an E1-alone readout, or it will under-read a divergence-preserving objective's "
                "true effect."
            ),
        },
        "source_substrate_entry": "SD-e1-rollout-consistency-training (fan-out sibling H-c, per confirmed autopsy)",
        "source_autopsy": "failure_autopsy_V3-EXQ-976_2026-09-02",
        "fanout_hypothesis": "H-c",
        "sibling_not_gate_of": "H-f (ITEM 3 rollout-endpoint contrastive build)",
        "source_chip": "chip-20260902-sde1-hc-e1alone-rollout-readout-probe",
        "reference_runs": {
            "v3_exq_976": "V3-EXQ-976 (v3_exq_976_sd_e1_item2_rollout_consistency_validation_20260902T114700Z_v3)",
            "v3_exq_965": "v3_exq_965_sd_e1_item1_action_conditioning_validation_20260830T145908Z_v3",
            "v3_exq_968": "v3_exq_968_sd_e1_output_proj_residual_ab_20260901T162647Z_v3",
        },
        "hypothesis_space_qid": "sd_e1_residual_crush_locus",
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
            "V3-EXQ-980: H-c readout-regime sibling probe -- does the hybrid E1/E2 Phase-4 "
            "evaluator suppress the depth damping a rollout-consistency-trained E1 objective "
            "produces, relative to an E1-alone readout of the same trained cells? (diagnostic)"
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
        print("[V3-EXQ-980] SMOKE TEST MODE", flush=True)
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
        print("[V3-EXQ-980] SMOKE TEST COMPLETE", flush=True)
        for k in ["status", "non_degenerate", "degeneracy_reason"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
        print(f"  label: {result['interpretation']['label']}", flush=True)
        print(f"  damping_hybrid: {result.get('damping_hybrid')}", flush=True)
        print(f"  damping_e1alone: {result.get('damping_e1alone')}", flush=True)
        print(f"  arm_readout_discrimination: {result.get('arm_readout_discrimination')}", flush=True)
        print(f"  on_arm_vacuous: {result.get('on_arm_vacuous')}", flush=True)
        print(f"  grad_cos_on_cells: {result.get('grad_cos_on_cells')}", flush=True)
        print(f"  depth0_trained_call_frac_per_cell: {result.get('depth0_trained_call_frac_per_cell')}", flush=True)
        print(f"  e1_grad_step_gap_per_seed: {result.get('e1_grad_step_gap_per_seed')}", flush=True)
        print(f"  trained_loss_last_over_first_fifth_per_cell: {result.get('trained_loss_last_over_first_fifth_per_cell')}", flush=True)

    # --- runner-conformance sentinel ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
