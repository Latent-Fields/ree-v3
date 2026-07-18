"""V3-EXQ-783 -- z_world under-differentiation: separate (a1) untrained encoder from
(a2) the world_dim=32 discriminative-granularity ceiling.

DIAGNOSTIC. Discriminates WHY z_world is under-differentiated. Not a claim falsifier.

THE OPEN PIECE. The 2026-07-18 characterisation
(`REE_assembly/evidence/experiments/zworld_near_static_characterisation_2026-07-18.md`,
landed REE_assembly master 735e067e4d) discriminated the three candidate causes of the
near-static z_world manifold recorded by convergence probe DREAMER-V3-P-008. Verdict:
cause (a) ENCODER. Cause (b) RUNG refuted (contrast flat across the full four-rung x734
ladder, and HIGHEST at the most impoverished D3_hazard_free). Cause (c) COMPETENCE refuted
for manifold spread (contrast ratio ~0.094 invariant across a 782x foraging range and a
17x survival range), though it does own the separate one-step-movement statistic.

Its section 3d records that cause (a) has TWO components which that design cannot separate:

  (a1) the world encoder is never trained in the measured configuration. The weight-delta
       check showed latent_stack 0/61 tensors change across the x734 P0 warmup, because the
       P0 loop buffers latent.z_world.detach() and optimises only Adam(agent.e2.parameters())
       (x724._e2_contrastive_step), so SplitEncoder.world_encoder receives no gradient.
  (a2) the world_dim=32 discriminative-granularity ceiling, already adjudicated
       epistemic_category `substrate_ceiling` / `pending_retest_after_substrate` by
       `REE_assembly/evidence/planning/failure_autopsy_zworld-integration-cluster_2026-06-06.md`.

Every cell measured on 2026-07-18 was dim=32 AND untrained, so (a1) and (a2) are perfectly
confounded in that data.

NOT A FRESH DESIGN. This runs the 2026-06-06 cluster autopsy's USER-CONFIRMED retest spec
(that doc, section 7): world_dim=128 AND a behaviourally-balanced / exploratory policy
(ARC-065 diversity active), re-measuring event selectivity + counterfactual attribution +
fine-vs-coarse residue accuracy TOGETHER in one harness. It extends that harness to also
emit the four DREAMER-V3-P-008 manifold statistics plus the offset-invariant contrast ratio
CR = spread / ||centroid|| and the participation ratio, on BOTH z_world and the raw
world_state channel, so the numbers are directly comparable against the 2026-07-18 table.
Crossing {dim 32, 128} x {encoder untrained, trained} is what separates (a1) from (a2).

DESIGN -- the 2x2 crossing (4 arms x 3 seeds = 12 cells):

  D32_UNTRAINED   world_dim=32,  encoder untrained  -- CONTROL: reproduces the measured
                                                       2026-07-18 / DREAMER-V3-P-008 config
  D32_TRAINED     world_dim=32,  encoder trained    -- isolates (a1) at the old dim
  D128_UNTRAINED  world_dim=128, encoder untrained  -- isolates (a2) with training held off
  D128_TRAINED    world_dim=128, encoder trained    -- the retest-spec cell

ENV EXPOSURE IS HELD IDENTICAL ACROSS ARMS. Every arm runs the SAME P0 rollouts and computes
the SAME auxiliary losses; the untrained arms simply do not step the encoder optimiser. So
the arms differ in weight updates only, never in what they saw -- the state-distribution
confound the 2026-07-18 grid controlled by holding the encoder fixed within a cell.

BEHAVIOURALLY-BALANCED POLICY (the retest spec's second requirement). Training and
measurement are driven by an explicitly exploratory epsilon-mixed policy (LocalViewGreedy
with epsilon-random), NOT the E3/SP-CEM planner. This satisfies the "behaviourally-balanced
/ exploratory policy" requirement by construction and controllably, rather than depending on
emergent ARC-065 diversity, and it keeps the state distribution matched across arms. The
achieved balance is recorded in `label_balance` for the SD-009 event labels, separately for
the TRAINING and the EVAL partitions, so a saturated training label cannot silently
invalidate the run.

ENCODER TRAINING PATH. The TRAINED arms backprop two auxiliary losses into latent_stack via
the reusable agent methods, which are documented to carry gradient through the world encoder:
  SD-009  agent.compute_event_contrastive_loss(prev_transition_type, latent_state)   [CE]
  SD-018  agent.compute_resource_proximity_loss(resource_prox_target, latent_state)  [MSE]
optimised by Adam(agent.latent_stack.parameters()). This is exactly the P0 the substrate
records as the precondition: `substrate_queue.json:971` (E2WorldForward) states "P0 must
train the z_world encoder (SD-009 + SD-018) before the forward model -- a random encoder
gives a vacuous zero comparator (MECH-353/V3-EXQ-642 lesson)", and `e2_world.py` lines
42-54 specify the same phased P0/P1/P2.

PHASED TRAINING (mandatory protocol; also the SD-031 design-doc phasing):
  P0  encoder warmup    -- SD-009 + SD-018 into latent_stack. No downstream loss.
  P1  E2WorldForward    -- trains on FROZEN z_world (target = z_world_next.detach()).
                           Encoder optimiser is NOT stepped in P1.
  P2  measurement       -- manifold / selectivity / residue / attribution.

ATTRIBUTION LEG IS DIM-GATED BY THE SUBSTRATE, NOT BY CHOICE. E2WorldForward hard-asserts
world_dim >= 128 (MIN_DISCRIMINATIVE_WORLD_DIM) as a carry-forward guard from the 2026-06-06
autopsy. At dim=32 it is constructed with allow_subthreshold_dim=True, which by design
reports attribution_ready=False and returns a ZEROED comparator residual as a
"do-not-interpret" sentinel rather than a misleadingly-zero attribution gap. So the
attribution statistic yields a 1-D trained-vs-untrained contrast at dim=128 only; it is
recorded as None (never 0.0) at dim=32. The (a1)/(a2) separation rests on the three
statistics that DO cross the full 2x2: manifold contrast ratio, event selectivity, and
fine-vs-coarse residue accuracy.

LOAD-BEARING DV: z_world contrast ratio CR = spread / ||centroid||, the offset-invariant
statistic the 2026-07-18 characterisation established (raw magnitude spread is not comparable
across channels because a biased MLP adds a constant offset that inflates ||z|| without
adding variance). Baseline to beat: CR ~ 0.094, whose observed dynamic range across all 24
cells of the 2026-07-18 grid was only 1.52x (0.0764 - 0.1159) against a 782x foraging range.

PRE-REGISTERED SELF-ROUTE (a HYPOTHESIS, not a verdict):
  * READINESS fails -> `substrate_not_ready_requeue`. Either the raw world_state channel's
    own contrast is degenerate (no upstream variation to encode, so the measurement is
    starved), or a TRAINED arm's encoder did not actually move (weight-delta 0 -- the very
    failure this experiment exists to fix). NEVER a substrate verdict in that case.
  * `a1_untrained_encoder_dominates`  -- training lifts CR at BOTH dims and dim adds little.
  * `a2_dim32_granularity_ceiling_dominates` -- dim lifts CR and training adds little.
  * `a1_a2_conjunctive` -- CR lifts only in the D128_TRAINED cell (both necessary).
  * `neither_axis_lifts_contrast` -- CR flat across the whole crossing. A genuinely new
    finding: the cause is upstream of both axes.

SCOPE -- what this does and does not bear on (carried forward verbatim from the 2026-07-18
characterisation's section 6, at user instruction):
  * Bears on INV-088's ANTECEDENT only (that z_world differentiation is genuinely low).
    NOT tagged in claim_ids: the coupling leg is untouched here and V3-EXQ-744a's WEAKENS
    reading stands. This must not be read as rehabilitating the strong bound.
  * NO bearing on MECH-459 / return-scale / normaliser.
  * NO bearing on either leg of the live GOV-FANOUT-1 discrimination (V3-EXQ-780 H-bc-prior
    vs V3-EXQ-781 H-approach-primitive). Both remain `claimed`; the discrimination remains
    open and `mech457_competence_bootstrap_explorer` stays `blocked_pending_discrimination`.

GOV-REUSE-1: the decisive readout is the z_world contrast ratio under a TRAINED encoder.
`reanalysis_query.py query --readout zworld_contrast_ratio` and `--readout zworld_eff_rank`
returned NO substrate-compatible MATCH -- every candidate is UNVERIFIABLE (pre-2026-07-12
manifests carry no recoverable substrate_hash), and no recorded run measures contrast under
a trained encoder at either dim. Not recoverable -> run.

RE-DERIVE BRAKE (MOVE-3): Q-002 count 1, SD-031 count 0 -- neither braked. INV-088 count 2
(failure_autopsy_MECH-457-fanout-751-750_2026-07-14,
failure_autopsy_MECH-457-fanout-752-753-754_2026-07-15) would trip the threshold, but it is
NOT tagged here, those autopsies are a different lineage and granularity (MECH-457 fanout,
not this crossing), the named upstream substrate SD-031 is now IMPLEMENTED (2026-06-06,
which releases the brake), and this is a `diagnostic` whose purpose is to discriminate WHY
the ceiling holds -- explicitly outside the re-derive loop.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_protocol import emit_outcome

from experiments.pack_writer import write_flat_manifest

from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.capability_eval import LocalViewGreedyPolicy, RandomPolicy
from experiments._lib.baselines import exq783_zworld_granularity as base

from ree_core.agent import REEAgent
from ree_core.predictors.e2_world import (
    MIN_DISCRIMINATIVE_WORLD_DIM,
    E2WorldConfig,
    E2WorldForward,
)
from ree_core.residue.field import ResidueConfig, ResidueField
from ree_core.utils.config import REEConfig

import experiments.v3_exq_724_competence_localization_diagnostic as x724

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_783_zworld_granularity_training_crossing"
QUEUE_ID = "V3-EXQ-783"
CLAIM_IDS: List[str] = ["Q-002", "SD-031"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"

# ---------------------------------------------------------------------------
# Crossing
# ---------------------------------------------------------------------------
# 8 seeds. The 2026-07-18 characterisation ran 2 seeds per cell and flagged that as
# "adequate for effects this large (782x vs 1.5x), thin for the dimensionality-tracking
# claim" (its scope limit 4). The fixed-policy design here costs ~10 s/cell (no CEM planner
# is invoked), so the power is essentially free -- and the CR_LIFT_SD_MULTIPLE gate is a
# cross-seed SD of the paired delta, which is badly estimated at n=2-3.
SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49]

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "D32_UNTRAINED", "world_dim": 32, "encoder_training": False, "is_off_arm": True},
    {"arm_id": "D32_TRAINED", "world_dim": 32, "encoder_training": True, "is_off_arm": False},
    {"arm_id": "D128_UNTRAINED", "world_dim": 128, "encoder_training": False, "is_off_arm": False},
    {"arm_id": "D128_TRAINED", "world_dim": 128, "encoder_training": True, "is_off_arm": False},
]

# ---------------------------------------------------------------------------
# Schedule. P0 and P1 episode counts are IDENTICAL across arms so env exposure is matched;
# only the encoder optimiser step is gated. TOTAL_TRAIN_EPISODES is the progress denominator.
# ---------------------------------------------------------------------------
# Sourced from the canonical baseline module so the driver can never disagree with the
# config slice its fingerprints declare (see that module's schedule block).
P0_ENCODER_EPISODES = base.OFF_P0_ENCODER_EPISODES
P1_E2_EPISODES = base.OFF_P1_E2_EPISODES
TOTAL_TRAIN_EPISODES = P0_ENCODER_EPISODES + P1_E2_EPISODES
STEPS_PER_EPISODE = base.OFF_STEPS_PER_EPISODE
MEASURE_EPISODES = base.OFF_MEASURE_EPISODES

ENCODER_LR = 1e-4
E2_WORLD_LR = 3e-4
SD009_WEIGHT = 1.0
SD018_WEIGHT = 0.5
MAX_GRAD_NORM = 1.0

# Exploratory epsilon for the behaviourally-balanced policy (retest-spec requirement).
EXPLORE_EPSILON = 0.35

# Residue resolution pair (Q-002 fine-vs-coarse leg, mirroring V3-EXQ-215).
RESIDUE_COARSE_CENTERS = 8
RESIDUE_FINE_CENTERS = 64
RESIDUE_BANDWIDTH = 1.0

# ---------------------------------------------------------------------------
# Pre-registered thresholds (constants, never derived from the run's own statistics)
# ---------------------------------------------------------------------------
# Absolute floor on a CR lift. The 2026-07-18 grid put CR ~0.094 with a total observed
# dynamic range of only 1.52x (0.0764-0.1159) across 24 cells spanning 782x foraging. A lift
# must clear that entire observed band to count, so the floor is set at ~0.05 absolute.
CR_LIFT_FLOOR = 0.05
# Scale-aware companion: the lift must also clear 2x the cross-seed SD of the delta itself.
CR_LIFT_SD_MULTIPLE = 2.0
# "Adds little" bound for the non-dominant axis in the single-axis readings.
CR_MINOR_AXIS_CEILING = 0.02
# Readiness: the raw world_state channel must itself be non-degenerate on contrast, else
# there is no upstream variation for the encoder to transmit and the test is starved.
RAW_CR_READINESS_FLOOR = 0.20
# Readiness: a TRAINED arm must actually move world-path encoder weights.
MIN_CHANGED_WORLD_TENSORS = 1
# Readiness (ANTI-COLLAPSE): a TRAINED arm must not have DESTROYED the representation.
# Authoring-time measurement on this substrate (2026-07-18, world_dim=128, seed 42, 40 eval
# episodes) found the prescribed P0 collapses z_world to ~1 effective dimension:
#     untrained                       PR = 9.67   CR = 0.1538
#     SD-009 + SD-018 (lr 1e-4)       PR = 1.07   CR = 0.0824
#     SD-009 only (w_018 = 0)         PR = 1.14   CR = 0.0930
# i.e. the collapse is NOT merely the scalar SD-018 regression -- the SD-009 CE is saturated
# at ~95% class-0 in this env, and predicting a near-constant class is trivially achieved by
# collapsing z_world onto one axis. A PR ~1 representation has NO discriminative geometry, so
# its (necessarily low) contrast ratio says "the training recipe destroyed the manifold", NOT
# "the dim=32 ceiling holds". Reading a collapsed arm as an (a1)/(a2) verdict would be exactly
# the trivial-prediction signature the P0 readiness-assert rule exists to catch. Below-floor
# therefore routes to substrate_not_ready_requeue, never to a substrate verdict.
MIN_TRAINED_PR_ABSOLUTE = 2.0
MIN_TRAINED_PR_FRACTION_OF_UNTRAINED = 0.5
# Sample floor for the manifold statistics.
MIN_STATES_FOR_MANIFOLD = 200
# Both SD-009 event label classes must have >=10% coverage for the selectivity leg to mean
# anything (proxy-label calibration rule).
MIN_LABEL_CLASS_FRAC = 0.10


# ---------------------------------------------------------------------------
# Behaviourally-balanced exploratory policy
# ---------------------------------------------------------------------------
class ExploratoryBalancedPolicy:
    """LocalViewGreedy with epsilon-random -- the retest spec's behaviourally-balanced
    exploratory policy, applied controllably rather than via emergent ARC-065 diversity.

    Greedy alone under-visits hazard states (the 6.6:1 event imbalance the 2026-06-06
    autopsy recorded for V3-EXQ-177, which made event selectivity near-unmeasurable);
    pure random under-visits resources. The epsilon mixture exposes both classes. Achieved
    balance is recorded in label_balance rather than assumed.
    """

    name = "exploratory_balanced"

    def __init__(self, seed: int, epsilon: float = EXPLORE_EPSILON) -> None:
        self._greedy = LocalViewGreedyPolicy()
        self._random = RandomPolicy(seed=seed)
        self._rng = np.random.RandomState(int(seed) + 9973)
        self._epsilon = float(epsilon)

    def reset(self, env: Any) -> None:
        self._greedy.reset(env)
        self._random.reset(env)

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        if self._rng.rand() < self._epsilon:
            return self._random.act(env, obs_dict)
        return self._greedy.act(env, obs_dict)

    def post_step(self, env: Any, info: Dict[str, Any], obs_dict: Dict[str, Any]) -> None:
        return None


def _make_measure_policies(seed: int) -> List[Any]:
    """Fixed measurement policies, matching the 2026-07-18 grid's design choice: the encoder
    is held FIXED within a cell and only the measurement policy varies, so state-distribution
    effects are separated from the representation itself."""
    return [RandomPolicy(seed=seed), LocalViewGreedyPolicy()]


# ---------------------------------------------------------------------------
# Manifold statistics -- the four DREAMER-V3-P-008 quantities plus CR and PR
# ---------------------------------------------------------------------------
def _manifold_stats(states: np.ndarray) -> Dict[str, Any]:
    """Compute the offset-invariant manifold statistics for one channel.

    states: [n_states, dim], in visitation order (so consecutive rows are consecutive steps
    within an episode; episode boundaries are excluded by the caller).

    Returns the four DREAMER-V3-P-008 statistics, plus:
      contrast_ratio      CR = spread / ||centroid||  (offset-invariant; the load-bearing DV)
      participation_ratio PR of the centred covariance spectrum (also offset-invariant)
    """
    n = int(states.shape[0])
    if n < 2:
        return {"n_states": n, "insufficient": True}

    norms = np.linalg.norm(states, axis=1)
    centroid = states.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    dist_to_centroid = np.linalg.norm(states - centroid[None, :], axis=1)

    # Whole-dataset inter-state spread: RMS pairwise distance, computed as sqrt(2)x the RMS
    # distance-to-centroid (exact identity, avoids the O(n^2) pairwise matrix).
    spread = float(np.sqrt(2.0) * np.sqrt(float((dist_to_centroid ** 2).mean())))

    # Mean one-step movement (consecutive rows within an episode).
    step_moves = np.linalg.norm(np.diff(states, axis=0), axis=1)
    mean_one_step_movement = float(step_moves.mean()) if step_moves.size else 0.0

    contrast_ratio = float(spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")

    # Participation ratio of the CENTRED covariance spectrum: (sum l_i)^2 / sum(l_i^2).
    centred = states - centroid[None, :]
    cov = (centred.T @ centred) / float(max(n - 1, 1))
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    s1 = float(eig.sum())
    s2 = float((eig ** 2).sum())
    participation_ratio = float((s1 * s1) / s2) if s2 > 1e-24 else 0.0

    return {
        "n_states": n,
        "mean_norm": float(norms.mean()),
        "mean_one_step_movement": mean_one_step_movement,
        "mean_dist_to_centroid": float(dist_to_centroid.mean()),
        "inter_state_spread": spread,
        "centroid_norm": centroid_norm,
        "contrast_ratio": contrast_ratio,
        "participation_ratio": participation_ratio,
        "insufficient": False,
    }


# ---------------------------------------------------------------------------
# Agent / env construction
# ---------------------------------------------------------------------------
def _make_agent(env, world_dim: int, encoder_training: bool) -> REEAgent:
    cfg = REEConfig.from_dims(**base.agent_config_kwargs(env, world_dim, encoder_training))
    return REEAgent(cfg)


def _world_path_param_names(agent: REEAgent) -> List[str]:
    """The latent_stack tensors on the z_world path -- the exact set the 2026-07-18
    weight-delta check reported as 'world-path CHANGED: NONE'."""
    names = []
    for name, _p in agent.latent_stack.named_parameters():
        if ("world_encoder" in name or "world_topdown" in name
                or "world_precision_logit" in name or "world_predictor" in name
                or "event_classifier" in name or "resource_proximity_head" in name):
            names.append(name)
    return names


def _snapshot_world_path(agent: REEAgent) -> Dict[str, torch.Tensor]:
    keep = set(_world_path_param_names(agent))
    return {n: p.detach().clone() for n, p in agent.latent_stack.named_parameters() if n in keep}


def _count_changed(agent: REEAgent, snapshot: Dict[str, torch.Tensor]) -> Tuple[int, int, float]:
    changed = 0
    max_delta = 0.0
    for name, p in agent.latent_stack.named_parameters():
        if name not in snapshot:
            continue
        d = float((p.detach() - snapshot[name]).abs().max().item())
        max_delta = max(max_delta, d)
        if d > 0.0:
            changed += 1
    return changed, len(snapshot), max_delta


def _resource_prox_target(obs_dict: Dict[str, Any]) -> Optional[float]:
    rfv = obs_dict.get("resource_field_view")
    if rfv is None:
        return None
    try:
        return float(torch.as_tensor(rfv).max().item())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# P0 -- encoder warmup (SD-009 event-contrastive + SD-018 resource proximity)
# ---------------------------------------------------------------------------
def _train_encoder_phase(
    agent: REEAgent,
    env,
    seed: int,
    encoder_training: bool,
    arm_id: str,
    episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Run P0. Rollouts and losses are computed for EVERY arm (matched env exposure);
    the optimiser is stepped only when encoder_training is True."""
    policy = ExploratoryBalancedPolicy(seed=seed)
    opt = torch.optim.Adam(agent.latent_stack.parameters(), lr=ENCODER_LR)

    label_counts: Dict[str, int] = {}
    losses: List[float] = []

    for ep in range(episodes):
        _flat0, obs_dict = env.reset()
        policy.reset(env)
        prev_ttype: Optional[str] = None

        for _step in range(steps_per_episode):
            latent = x724_sense(agent, obs_dict)

            loss = torch.zeros((), dtype=torch.float32)
            if prev_ttype is not None:
                loss = loss + SD009_WEIGHT * agent.compute_event_contrastive_loss(
                    prev_ttype, latent
                )
            prox_target = _resource_prox_target(obs_dict)
            if prox_target is not None:
                loss = loss + SD018_WEIGHT * agent.compute_resource_proximity_loss(
                    prox_target, latent
                )

            if encoder_training and loss.requires_grad:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.latent_stack.parameters(), max_norm=MAX_GRAD_NORM
                )
                opt.step()
            if torch.is_tensor(loss):
                lv = float(loss.detach().item())
                if math.isfinite(lv):
                    losses.append(lv)

            action = policy.act(env, obs_dict)
            with torch.no_grad():
                _flat, _harm, done, info, obs_dict = env.step(action)
            if not isinstance(info, dict):
                info = {}
            ttype = str(info.get("transition_type", "none"))
            label_counts[ttype] = label_counts.get(ttype, 0) + 1
            prev_ttype = ttype
            if done:
                break

        if ep == 0 or (ep + 1) % 20 == 0:
            print(
                "  [train] %s seed=%d ep %d/%d (P0 encoder)"
                % (arm_id, seed, ep + 1, TOTAL_TRAIN_EPISODES),
                flush=True,
            )

    return {
        "p0_mean_loss": float(np.mean(losses)) if losses else None,
        "p0_label_counts": dict(label_counts),
        "p0_n_labelled_steps": int(sum(label_counts.values())),
    }


def x724_sense(agent: REEAgent, obs_dict: Dict[str, Any]):
    """Encode one observation WITHOUT no_grad -- the encoder gradient must be able to reach
    latent_stack. Mirrors v3_exq_742._sense."""
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent.sense(
        obs_body=body,
        obs_world=world,
        obs_harm=x724._obs_harm(obs_dict),
        obs_harm_a=x724._obs_harm_a(obs_dict),
        obs_harm_history=x724._obs_harm_history(obs_dict),
    )


# ---------------------------------------------------------------------------
# P1 -- E2WorldForward on FROZEN z_world (stop-gradient targets; encoder NOT stepped)
# ---------------------------------------------------------------------------
def _train_e2_world_phase(
    agent: REEAgent,
    env,
    seed: int,
    world_dim: int,
    arm_id: str,
    episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    e2cfg = E2WorldConfig(
        use_e2_world_forward=True,
        z_world_dim=int(world_dim),
        action_dim=int(env.action_dim),
        learning_rate=E2_WORLD_LR,
        # Below the discriminative threshold the substrate REQUIRES an explicit opt-in and
        # then reports attribution_ready=False with a zeroed sentinel residual.
        allow_subthreshold_dim=(int(world_dim) < MIN_DISCRIMINATIVE_WORLD_DIM),
    )
    world_fwd = E2WorldForward(e2cfg)
    opt = torch.optim.Adam(world_fwd.parameters(), lr=E2_WORLD_LR)
    policy = ExploratoryBalancedPolicy(seed=seed + 101)

    losses: List[float] = []
    enc_snapshot = _snapshot_world_path(agent)

    for ep in range(episodes):
        _flat0, obs_dict = env.reset()
        policy.reset(env)
        with torch.no_grad():
            z_prev = x724_sense(agent, obs_dict).z_world.detach()

        for _step in range(steps_per_episode):
            action = policy.act(env, obs_dict)
            a_onehot = torch.zeros(1, int(env.action_dim))
            a_onehot[0, int(action)] = 1.0
            _flat, _harm, done, info, obs_dict = env.step(action)
            with torch.no_grad():
                z_next = x724_sense(agent, obs_dict).z_world.detach()

            z_pred = world_fwd(z_prev, a_onehot)
            loss = world_fwd.compute_loss(z_pred, z_next)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lv = float(loss.detach().item())
            if math.isfinite(lv):
                losses.append(lv)

            z_prev = z_next
            if done:
                break

        if ep == 0 or (ep + 1) % 20 == 0:
            print(
                "  [train] %s seed=%d ep %d/%d (P1 e2_world)"
                % (arm_id, seed, P0_ENCODER_EPISODES + ep + 1, TOTAL_TRAIN_EPISODES),
                flush=True,
            )

    # P1 must NOT move the encoder (stop-gradient discipline). Verify it.
    changed_in_p1, _tot, _mx = _count_changed(agent, enc_snapshot)

    return {
        "world_fwd": world_fwd,
        "p1_mean_loss": float(np.mean(losses)) if losses else None,
        "p1_encoder_tensors_changed": int(changed_in_p1),
        "attribution_ready": bool(world_fwd.attribution_ready),
    }


# ---------------------------------------------------------------------------
# P2 -- measurement
# ---------------------------------------------------------------------------
def _collect_states(
    agent: REEAgent, env, policy, episodes: int, steps_per_episode: int
) -> Dict[str, Any]:
    """Roll out under a FIXED measurement policy, collecting z_world and raw world_state.

    Episode boundaries are recorded so the one-step-movement statistic never straddles a
    reset (which would contaminate it with a teleport).
    """
    z_rows: List[np.ndarray] = []
    raw_rows: List[np.ndarray] = []
    ep_bounds: List[Tuple[int, int]] = []
    ev_labels: Dict[str, int] = {}
    harm_events: List[Tuple[np.ndarray, float]] = []

    for _ep in range(episodes):
        _flat0, obs_dict = env.reset()
        policy.reset(env)
        start = len(z_rows)
        prev_ttype: Optional[str] = None

        for _step in range(steps_per_episode):
            with torch.no_grad():
                latent = x724_sense(agent, obs_dict)
                zw = latent.z_world.detach().squeeze(0).cpu().numpy().astype(np.float64)
            raw = obs_dict["world_state"].float().detach().squeeze().cpu().numpy().astype(np.float64)
            z_rows.append(zw)
            raw_rows.append(raw)

            action = policy.act(env, obs_dict)
            _flat, harm_signal, done, info, obs_dict = env.step(action)
            if not isinstance(info, dict):
                info = {}
            ttype = str(info.get("transition_type", "none"))
            ev_labels[ttype] = ev_labels.get(ttype, 0) + 1
            hv = float(harm_signal)
            if hv < 0.0:
                harm_events.append((zw, abs(hv)))
            prev_ttype = ttype
            if done:
                break

        ep_bounds.append((start, len(z_rows)))

    return {
        "z": np.asarray(z_rows, dtype=np.float64) if z_rows else np.zeros((0, 1)),
        "raw": np.asarray(raw_rows, dtype=np.float64) if raw_rows else np.zeros((0, 1)),
        "episode_bounds": ep_bounds,
        "event_labels": ev_labels,
        "harm_events": harm_events,
    }


def _stats_over_episodes(states: np.ndarray, bounds: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Whole-dataset statistics, but with one-step movement averaged WITHIN episodes only."""
    overall = _manifold_stats(states)
    if overall.get("insufficient"):
        return overall
    moves: List[float] = []
    for (a, b) in bounds:
        if b - a >= 2:
            seg = states[a:b]
            moves.extend(np.linalg.norm(np.diff(seg, axis=0), axis=1).tolist())
    overall["mean_one_step_movement"] = float(np.mean(moves)) if moves else 0.0
    overall["n_episodes"] = len(bounds)
    return overall


def _event_selectivity(agent: REEAgent, collected: Dict[str, Any]) -> Optional[float]:
    """event_selectivity_margin = 1 - cos_sim between the mean z_world of hazard-event states
    and the mean z_world of open (no-event) states. 0 = identical, 1 = orthogonal.
    Mirrors the V3-EXQ-177 definition (that script, lines 415-423)."""
    z = collected["z"]
    if z.shape[0] < MIN_STATES_FOR_MANIFOLD:
        return None
    # Recompute per-state labels is not possible post-hoc; use the harm-event z rows as the
    # event class and the remainder as the open class.
    harm = collected["harm_events"]
    if len(harm) < 10:
        return None
    harm_z = np.asarray([h[0] for h in harm], dtype=np.float64)
    harm_set = {tuple(np.round(r, 10)) for r in harm_z}
    open_rows = np.asarray(
        [r for r in z if tuple(np.round(r, 10)) not in harm_set], dtype=np.float64
    )
    if open_rows.shape[0] < 10:
        return None
    m_h = harm_z.mean(axis=0)
    m_o = open_rows.mean(axis=0)
    denom = float(np.linalg.norm(m_h) * np.linalg.norm(m_o))
    if denom < 1e-12:
        return None
    cos_sim = float(np.dot(m_h, m_o) / denom)
    return float(1.0 - cos_sim)


def _residue_fine_vs_coarse(world_dim: int, collected: Dict[str, Any]) -> Dict[str, Any]:
    """Q-002 fine-vs-coarse residue accuracy leg, mirroring V3-EXQ-215's construction
    (ResidueField with num_basis_functions as the resolution parameter, bandwidth 1.0).

    residue_accuracy = Pearson r between the field's evaluated residue at a state and the
    harm magnitude actually observed there. Held-out split: the field is accumulated on the
    first half of harm events and scored on the second half.
    """
    harm = collected["harm_events"]
    out: Dict[str, Any] = {"n_harm_events": len(harm)}
    if len(harm) < 20:
        out["insufficient"] = True
        return out
    out["insufficient"] = False
    split = len(harm) // 2
    fit, test = harm[:split], harm[split:]

    for label, n_centers in (("coarse", RESIDUE_COARSE_CENTERS), ("fine", RESIDUE_FINE_CENTERS)):
        field = ResidueField(
            ResidueConfig(
                world_dim=int(world_dim),
                hidden_dim=32,
                accumulation_rate=0.2,
                num_basis_functions=int(n_centers),
                kernel_bandwidth=RESIDUE_BANDWIDTH,
            )
        )
        with torch.no_grad():
            for zw, mag in fit:
                field.accumulate(
                    torch.as_tensor(zw, dtype=torch.float32).unsqueeze(0),
                    harm_magnitude=float(mag),
                )
            preds, actuals = [], []
            for zw, mag in test:
                v = field.evaluate(torch.as_tensor(zw, dtype=torch.float32).unsqueeze(0))
                preds.append(float(torch.as_tensor(v).squeeze().item()))
                actuals.append(float(mag))
        if len(preds) >= 3 and float(np.std(preds)) > 1e-12 and float(np.std(actuals)) > 1e-12:
            r = float(np.corrcoef(preds, actuals)[0, 1])
        else:
            r = None
        out["%s_residue_accuracy" % label] = r
        out["%s_num_centers" % label] = int(n_centers)

    fine_r, coarse_r = out.get("fine_residue_accuracy"), out.get("coarse_residue_accuracy")
    out["fine_minus_coarse"] = (
        float(fine_r - coarse_r) if (fine_r is not None and coarse_r is not None) else None
    )
    return out


def _attribution_gap(
    agent: REEAgent, world_fwd: E2WorldForward, env, seed: int, world_dim: int
) -> Dict[str, Any]:
    """SD-031 single-pass MECH-256 comparator attribution gap.

    DIM-GATED BY THE SUBSTRATE. Below MIN_DISCRIMINATIVE_WORLD_DIM the comparator reports
    attribution_ready=False and returns a ZEROED residual as a do-not-interpret sentinel.
    We record None (never 0.0) in that case -- a zeroed residual from a not-ready model is
    not a genuine self-caused zero gap.
    """
    if not world_fwd.attribution_ready:
        return {
            "attribution_ready": False,
            "attribution_gap": None,
            "attribution_note": (
                "world_dim=%d < MIN_DISCRIMINATIVE_WORLD_DIM=%d; comparator returns a zeroed "
                "sentinel by design. Recorded as None, NOT 0.0." % (world_dim, MIN_DISCRIMINATIVE_WORLD_DIM)
            ),
        }

    policy = ExploratoryBalancedPolicy(seed=seed + 202)
    self_caused: List[float] = []
    ext_caused: List[float] = []

    for _ep in range(max(4, MEASURE_EPISODES // 4)):
        _flat0, obs_dict = env.reset()
        policy.reset(env)
        with torch.no_grad():
            z_prev = x724_sense(agent, obs_dict).z_world.detach()
        for _step in range(STEPS_PER_EPISODE):
            action = policy.act(env, obs_dict)
            a_onehot = torch.zeros(1, int(env.action_dim))
            a_onehot[0, int(action)] = 1.0
            _flat, _harm, done, info, obs_dict = env.step(action)
            if not isinstance(info, dict):
                info = {}
            with torch.no_grad():
                z_obs = x724_sense(agent, obs_dict).z_world.detach()
                resid = world_fwd.comparator_residual(z_obs, z_prev, a_onehot)
                mag = float(torch.as_tensor(resid).norm(dim=-1).mean().item())
            ttype = str(info.get("transition_type", "none"))
            if ttype.startswith("agent_caused"):
                self_caused.append(mag)
            elif ttype.startswith("env_caused"):
                ext_caused.append(mag)
            z_prev = z_obs
            if done:
                break

    if len(self_caused) < 5 or len(ext_caused) < 5:
        return {
            "attribution_ready": True,
            "attribution_gap": None,
            "n_self_caused": len(self_caused),
            "n_ext_caused": len(ext_caused),
            "attribution_note": "insufficient events of one or both causal classes",
        }
    gap = float(np.mean(ext_caused) - np.mean(self_caused))
    return {
        "attribution_ready": True,
        "attribution_gap": gap,
        "mean_self_caused_residual": float(np.mean(self_caused)),
        "mean_ext_caused_residual": float(np.mean(ext_caused)),
        "n_self_caused": len(self_caused),
        "n_ext_caused": len(ext_caused),
    }


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def _config_slice_for(arm: Dict[str, Any]) -> Dict[str, Any]:
    if arm["is_off_arm"]:
        # The OFF cell is anchored on the canonical baseline module so a future sibling with
        # a DIFFERENT driver matches this fingerprint by construction.
        return base.off_path_config_slice()
    return {
        "lineage": "exq783_zworld_granularity",
        "env_kwargs": base.env_kwargs(),
        "world_dim": arm["world_dim"],
        "encoder_training": arm["encoder_training"],
        "use_event_classifier": arm["encoder_training"],
        "alpha_world": 0.9,
        "schedule": {
            "p0_encoder_episodes": P0_ENCODER_EPISODES,
            "p1_e2_episodes": P1_E2_EPISODES,
            "measure_episodes": MEASURE_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
    }


def _run_cell(arm: Dict[str, Any], seed: int, dry_run: bool) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    world_dim = int(arm["world_dim"])
    encoder_training = bool(arm["encoder_training"])

    p0_eps = 2 if dry_run else P0_ENCODER_EPISODES
    p1_eps = 2 if dry_run else P1_E2_EPISODES
    meas_eps = 2 if dry_run else MEASURE_EPISODES
    steps = 25 if dry_run else STEPS_PER_EPISODE

    print("Seed %d Condition %s" % (seed, arm_id), flush=True)

    row: Dict[str, Any] = {
        "arm_id": arm_id,
        "seed": seed,
        "world_dim": world_dim,
        "encoder_training": encoder_training,
    }

    with arm_cell(
        seed,
        config_slice=_config_slice_for(arm),
        script_path=Path(__file__),
        include_driver_script_in_hash=False,
    ) as cell:
        env = base.make_env(seed)
        agent = _make_agent(env, world_dim, encoder_training)

        pre = _snapshot_world_path(agent)
        p0 = _train_encoder_phase(
            agent, env, seed, encoder_training, arm_id, p0_eps, steps
        )
        changed, total, max_delta = _count_changed(agent, pre)
        row.update(p0)
        row["world_path_tensors_changed"] = int(changed)
        row["world_path_tensors_total"] = int(total)
        row["world_path_max_abs_delta"] = float(max_delta)

        p1 = _train_e2_world_phase(agent, env, seed, world_dim, arm_id, p1_eps, steps)
        world_fwd = p1.pop("world_fwd")
        row.update(p1)

        # --- measurement, per fixed policy ---
        per_policy: Dict[str, Any] = {}
        eval_labels: Dict[str, int] = {}
        primary: Optional[Dict[str, Any]] = None
        for pol in _make_measure_policies(seed):
            coll = _collect_states(agent, env, pol, meas_eps, steps)
            zs = _stats_over_episodes(coll["z"], coll["episode_bounds"])
            rs = _stats_over_episodes(coll["raw"], coll["episode_bounds"])
            entry = {
                "z_world": zs,
                "raw_world_state": rs,
                "attenuation_factor": (
                    float(rs["contrast_ratio"] / zs["contrast_ratio"])
                    if (not zs.get("insufficient") and not rs.get("insufficient")
                        and zs.get("contrast_ratio") and zs["contrast_ratio"] > 1e-12)
                    else None
                ),
                "event_selectivity_margin": _event_selectivity(agent, coll),
                "residue": _residue_fine_vs_coarse(world_dim, coll),
            }
            per_policy[pol.name] = entry
            for k, v in coll["event_labels"].items():
                eval_labels[k] = eval_labels.get(k, 0) + v
            # local_view_greedy is the primary measurement policy (the local-view-achievable
            # regime the learner itself sees); random_walk is the floor anchor.
            if pol.name == "local_view_greedy":
                primary = entry
        if primary is None:
            primary = next(iter(per_policy.values()))

        row["per_policy"] = per_policy
        row["eval_label_counts"] = eval_labels
        row["attribution"] = _attribution_gap(agent, world_fwd, env, seed, world_dim)

        # Primary (load-bearing) readouts, hoisted for convenience.
        row["zworld_contrast_ratio"] = primary["z_world"].get("contrast_ratio")
        row["raw_contrast_ratio"] = primary["raw_world_state"].get("contrast_ratio")
        row["zworld_participation_ratio"] = primary["z_world"].get("participation_ratio")
        row["raw_participation_ratio"] = primary["raw_world_state"].get("participation_ratio")
        row["zworld_mean_norm"] = primary["z_world"].get("mean_norm")
        row["zworld_mean_one_step_movement"] = primary["z_world"].get("mean_one_step_movement")
        row["zworld_mean_dist_to_centroid"] = primary["z_world"].get("mean_dist_to_centroid")
        row["zworld_inter_state_spread"] = primary["z_world"].get("inter_state_spread")
        row["event_selectivity_margin"] = primary.get("event_selectivity_margin")
        row["fine_residue_accuracy"] = primary["residue"].get("fine_residue_accuracy")
        row["coarse_residue_accuracy"] = primary["residue"].get("coarse_residue_accuracy")

        cell.stamp(row)

    ok = row.get("zworld_contrast_ratio") is not None
    print("verdict: %s" % ("PASS" if ok else "FAIL"), flush=True)
    return row


# ---------------------------------------------------------------------------
# Aggregation + self-route
# ---------------------------------------------------------------------------
# The SD-009 CE head has exactly three classes; agent.compute_event_contrastive_loss maps
# every unrecognised transition_type to class 0. Recording the RAW transition_type balance
# is therefore NOT enough -- 'hazard_approach' and 'resource' both collapse into class 0, so
# a raw distribution that looks varied can be a saturated CE label. This is the 047m lesson
# (a saturated TRAINING label silently invalidating a run), so the mapped balance is recorded
# separately and drives the C2 non-degeneracy flag.
SD009_LABEL_MAP = {"none": 0, "env_caused_hazard": 1, "agent_caused_hazard": 2}


def _label_balance(counts: Dict[str, int]) -> Dict[str, float]:
    tot = float(sum(counts.values())) or 1.0
    return {k: float(v) / tot for k, v in sorted(counts.items())}


def _sd009_class_balance(counts: Dict[str, int]) -> Dict[str, Any]:
    """Balance over the THREE CE classes the SD-009 head actually sees."""
    mapped: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    for k, v in counts.items():
        mapped[SD009_LABEL_MAP.get(k, 0)] += int(v)
    tot = float(sum(mapped.values())) or 1.0
    fracs = {str(k): float(v) / tot for k, v in sorted(mapped.items())}
    return {
        "class_counts": {str(k): int(v) for k, v in sorted(mapped.items())},
        "class_fracs": fracs,
        "min_class_frac": float(min(fracs.values())) if fracs else 0.0,
        "saturated": bool(max(fracs.values()) > (1.0 - MIN_LABEL_CLASS_FRAC)) if fracs else True,
    }


def _cr_by_arm(rows: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {a["arm_id"]: [] for a in ARMS}
    for r in rows:
        v = r.get("zworld_contrast_ratio")
        if v is not None and math.isfinite(float(v)):
            out[r["arm_id"]].append(float(v))
    return out


def _paired_delta(rows: List[Dict[str, Any]], arm_hi: str, arm_lo: str) -> Dict[str, Any]:
    """Per-seed paired delta in CR between two arms, plus the cross-seed SD of the delta."""
    hi = {r["seed"]: r.get("zworld_contrast_ratio") for r in rows if r["arm_id"] == arm_hi}
    lo = {r["seed"]: r.get("zworld_contrast_ratio") for r in rows if r["arm_id"] == arm_lo}
    deltas = [
        float(hi[s]) - float(lo[s])
        for s in sorted(set(hi) & set(lo))
        if hi[s] is not None and lo[s] is not None
        and math.isfinite(float(hi[s])) and math.isfinite(float(lo[s]))
    ]
    if not deltas:
        return {"mean_delta": None, "sd_delta": None, "n": 0, "clears": False}
    mean_d = float(np.mean(deltas))
    sd_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    clears = bool(
        mean_d >= CR_LIFT_FLOOR and mean_d >= CR_LIFT_SD_MULTIPLE * sd_d
    )
    return {
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "per_seed_deltas": deltas,
        "n": len(deltas),
        "clears": clears,
    }


def _self_route(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-registered interpretation grid. A HYPOTHESIS, not a verdict."""
    # --- readiness ---
    preconditions: List[Dict[str, Any]] = []

    # Readiness 1: the raw channel's own contrast must be non-degenerate. This asserts the
    # SAME statistic the load-bearing criterion routes on (a contrast RATIO, i.e. a
    # spread-relative quantity) on a positive control (the raw observation the encoder
    # consumes), per the V3-EXQ-643 same-statistic rule -- NOT a magnitude proxy.
    raw_crs = [
        float(r["raw_contrast_ratio"]) for r in rows
        if r.get("raw_contrast_ratio") is not None and math.isfinite(float(r["raw_contrast_ratio"]))
    ]
    raw_cr_min = float(min(raw_crs)) if raw_crs else 0.0
    preconditions.append({
        "name": "raw_channel_contrast_ratio_non_degenerate",
        "description": (
            "The raw world_state channel's own contrast ratio (the SAME spread-relative "
            "statistic the load-bearing CR criterion routes on) must clear the floor on the "
            "positive control, else there is no upstream variation for the encoder to "
            "transmit and the whole measurement is starved rather than informative."
        ),
        "control": "raw world_state observations the encoder consumes, min across all cells",
        "measured": raw_cr_min,
        "threshold": RAW_CR_READINESS_FLOOR,
        "met": bool(raw_cr_min >= RAW_CR_READINESS_FLOOR),
    })

    # Readiness 2: every TRAINED arm must have actually moved world-path encoder weights.
    trained_changed = [
        int(r.get("world_path_tensors_changed", 0)) for r in rows if r.get("encoder_training")
    ]
    min_changed = int(min(trained_changed)) if trained_changed else 0
    preconditions.append({
        "name": "trained_arm_encoder_weights_actually_moved",
        "description": (
            "Every encoder_training arm must show >=1 changed world-path tensor across P0. "
            "This is the exact check that produced 'world-path CHANGED: NONE' (0/61) on the "
            "x734 configuration; if it fails here the manipulation did not take and no "
            "(a1)/(a2) conclusion is licensed."
        ),
        "control": "world-path latent_stack tensors, min across all TRAINED cells",
        "measured": min_changed,
        "threshold": MIN_CHANGED_WORLD_TENSORS,
        "met": bool(min_changed >= MIN_CHANGED_WORLD_TENSORS),
    })

    # Readiness 3: P1 must NOT have moved the encoder (stop-gradient discipline held).
    p1_moved = max([int(r.get("p1_encoder_tensors_changed", 0)) for r in rows] or [0])
    preconditions.append({
        "name": "p1_encoder_frozen",
        "description": (
            "P1 trains E2WorldForward on stop-gradient z_world targets and must not move the "
            "encoder. A non-zero count means the phase separation leaked."
        ),
        "control": "world-path tensors changed during P1, max across all cells",
        "measured": p1_moved,
        "threshold": 0,
        "direction": "upper",
        "met": bool(p1_moved == 0),
    })

    # Readiness 4 (ANTI-COLLAPSE): each TRAINED arm must retain effective dimensionality
    # relative to its dim-matched UNTRAINED arm. This asserts a SPREAD/dimensionality
    # statistic -- the same family the load-bearing CR criterion routes on -- rather than a
    # magnitude proxy (the V3-EXQ-643 same-statistic rule).
    def _mean_pr(arm_id: str) -> Optional[float]:
        vals = [
            float(r["zworld_participation_ratio"]) for r in rows
            if r["arm_id"] == arm_id and r.get("zworld_participation_ratio") is not None
            and math.isfinite(float(r["zworld_participation_ratio"]))
        ]
        return float(np.mean(vals)) if vals else None

    pr_ratios: List[float] = []
    pr_detail: Dict[str, Any] = {}
    for hi, lo in (("D32_TRAINED", "D32_UNTRAINED"), ("D128_TRAINED", "D128_UNTRAINED")):
        pr_hi, pr_lo = _mean_pr(hi), _mean_pr(lo)
        if pr_hi is None or pr_lo is None or pr_lo <= 1e-9:
            continue
        ratio = float(pr_hi / pr_lo)
        pr_ratios.append(ratio)
        pr_detail[hi] = {
            "trained_pr": pr_hi, "untrained_pr": pr_lo, "retained_fraction": ratio,
            "absolute_ok": bool(pr_hi >= MIN_TRAINED_PR_ABSOLUTE),
        }
    min_pr_ratio = float(min(pr_ratios)) if pr_ratios else 0.0
    min_trained_pr = float(min([v["trained_pr"] for v in pr_detail.values()])) if pr_detail else 0.0
    preconditions.append({
        "name": "trained_arm_representation_not_collapsed",
        "description": (
            "Each TRAINED arm must retain >= %.0f%% of its dim-matched UNTRAINED arm's z_world "
            "participation ratio AND clear an absolute PR floor of %.1f. The prescribed P0 "
            "(SD-009 + SD-018) was measured at authoring time to collapse z_world to PR ~1.1 "
            "from an untrained PR ~9.7, driven by a ~95%% class-0-saturated SD-009 CE label. A "
            "collapsed manifold has no discriminative geometry, so its low contrast ratio "
            "reflects the training recipe destroying the representation -- NOT the dim=32 "
            "granularity ceiling. Below floor: substrate_not_ready_requeue, never a verdict."
            % (100.0 * MIN_TRAINED_PR_FRACTION_OF_UNTRAINED, MIN_TRAINED_PR_ABSOLUTE)
        ),
        "control": "dim-matched UNTRAINED arm's mean z_world participation ratio",
        "measured": min_pr_ratio,
        "threshold": MIN_TRAINED_PR_FRACTION_OF_UNTRAINED,
        "met": bool(
            pr_detail
            and min_pr_ratio >= MIN_TRAINED_PR_FRACTION_OF_UNTRAINED
            and min_trained_pr >= MIN_TRAINED_PR_ABSOLUTE
        ),
        "detail": pr_detail,
        "min_trained_pr_absolute": min_trained_pr,
        "absolute_threshold": MIN_TRAINED_PR_ABSOLUTE,
    })

    readiness_ok = all(bool(p["met"]) for p in preconditions)

    # --- axis deltas ---
    train_at_32 = _paired_delta(rows, "D32_TRAINED", "D32_UNTRAINED")
    train_at_128 = _paired_delta(rows, "D128_TRAINED", "D128_UNTRAINED")
    dim_untrained = _paired_delta(rows, "D128_UNTRAINED", "D32_UNTRAINED")
    dim_trained = _paired_delta(rows, "D128_TRAINED", "D32_TRAINED")
    conjunctive = _paired_delta(rows, "D128_TRAINED", "D32_UNTRAINED")

    def _small(d: Dict[str, Any]) -> bool:
        m = d.get("mean_delta")
        return m is not None and abs(float(m)) < CR_MINOR_AXIS_CEILING

    training_lifts_both = bool(train_at_32["clears"] and train_at_128["clears"])
    dim_lifts_both = bool(dim_untrained["clears"] and dim_trained["clears"])
    training_minor = _small(train_at_32) and _small(train_at_128)
    dim_minor = _small(dim_untrained) and _small(dim_trained)

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
    elif training_lifts_both and dim_minor:
        label = "a1_untrained_encoder_dominates"
    elif dim_lifts_both and training_minor:
        label = "a2_dim32_granularity_ceiling_dominates"
    elif conjunctive["clears"] and not training_lifts_both and not dim_lifts_both:
        label = "a1_a2_conjunctive"
    elif not conjunctive["clears"] and training_minor and dim_minor:
        label = "neither_axis_lifts_contrast"
    else:
        label = "mixed_partial_separation"

    # --- non-degeneracy of each criterion ---
    cr_by_arm = _cr_by_arm(rows)
    all_cr = [v for vals in cr_by_arm.values() for v in vals]
    cr_spread_across_arms = (
        float(np.std(all_cr, ddof=1)) if len(all_cr) > 1 else 0.0
    )
    sel_vals = [
        r["event_selectivity_margin"] for r in rows
        if r.get("event_selectivity_margin") is not None
    ]
    res_vals = [
        r["fine_residue_accuracy"] for r in rows if r.get("fine_residue_accuracy") is not None
    ]
    attr_ready_cells = [
        r for r in rows if (r.get("attribution") or {}).get("attribution_ready")
    ]

    # C2 is degenerate when the SD-009 CE label is saturated: with class-0 dominating, the
    # event-contrastive head has almost no signal to shape z_world with, so a near-zero
    # selectivity margin says "the label was starved", NOT "selectivity is absent". C2 is
    # NOT load-bearing, so this is a recorded caveat rather than a blocker on the run.
    train_counts_all: Dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("p0_label_counts") or {}).items():
            train_counts_all[k] = train_counts_all.get(k, 0) + v
    sd009_train = _sd009_class_balance(train_counts_all)

    criteria_non_degenerate = {
        "C1_cr_crossing": bool(len(all_cr) >= len(ARMS) and cr_spread_across_arms > 1e-9),
        "C2_event_selectivity": bool(
            len(sel_vals) >= 2
            and float(np.std(sel_vals)) > 1e-12
            and not sd009_train["saturated"]
        ),
        "C3_residue_fine_coarse": bool(len(res_vals) >= 2),
        "C4_attribution_dim128_only": bool(len(attr_ready_cells) >= 2),
    }

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": [
            {"name": "C1_cr_crossing", "load_bearing": True,
             "passed": bool(readiness_ok and label not in (
                 "substrate_not_ready_requeue", "neither_axis_lifts_contrast"))},
            {"name": "C2_event_selectivity", "load_bearing": False,
             "passed": bool(len(sel_vals) >= 2)},
            {"name": "C3_residue_fine_coarse", "load_bearing": False,
             "passed": bool(len(res_vals) >= 2)},
            {"name": "C4_attribution_dim128_only", "load_bearing": False,
             "passed": bool(len(attr_ready_cells) >= 2)},
        ],
        "axis_deltas": {
            "training_at_dim32": train_at_32,
            "training_at_dim128": train_at_128,
            "dim_at_untrained": dim_untrained,
            "dim_at_trained": dim_trained,
            "conjunctive_D128T_vs_D32U": conjunctive,
        },
        "sd009_ce_label_balance_train": sd009_train,
        "thresholds": {
            "cr_lift_floor": CR_LIFT_FLOOR,
            "cr_lift_sd_multiple": CR_LIFT_SD_MULTIPLE,
            "cr_minor_axis_ceiling": CR_MINOR_AXIS_CEILING,
            "raw_cr_readiness_floor": RAW_CR_READINESS_FLOOR,
        },
        "readiness_ok": readiness_ok,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm, seed, dry_run))

    interp = _self_route(rows)
    outcome = "PASS" if interp["readiness_ok"] and interp["label"] != "substrate_not_ready_requeue" else "FAIL"

    train_counts: Dict[str, int] = {}
    eval_counts: Dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("p0_label_counts") or {}).items():
            train_counts[k] = train_counts.get(k, 0) + v
        for k, v in (r.get("eval_label_counts") or {}).items():
            eval_counts[k] = eval_counts.get(k, 0) + v

    return {
        "outcome": outcome,
        "arm_results": rows,
        "interpretation": interp,
        "label_balance": {
            "transition_type": {
                "train_fracs": _label_balance(train_counts),
                "eval_fracs": _label_balance(eval_counts),
                "train_counts": train_counts,
                "eval_counts": eval_counts,
            },
            # The mapped CE-class balance is the one that matters for SD-009 (see
            # SD009_LABEL_MAP): raw transition_type variety can mask a saturated CE label.
            "sd009_ce_class_train": _sd009_class_balance(train_counts),
            "sd009_ce_class_eval": _sd009_class_balance(eval_counts),
        },
        "seeds_used": seeds,
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str,
                    dry_run: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, timestamp_utc)
    interp = result["interpretation"]

    full_config = {
        "env_kwargs": base.env_kwargs(),
        "arms": ARMS,
        "seeds": result["seeds_used"],
        "p0_encoder_episodes": P0_ENCODER_EPISODES,
        "p1_e2_episodes": P1_E2_EPISODES,
        "total_train_episodes": TOTAL_TRAIN_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "measure_episodes": MEASURE_EPISODES,
        "encoder_lr": ENCODER_LR,
        "e2_world_lr": E2_WORLD_LR,
        "sd009_weight": SD009_WEIGHT,
        "sd018_weight": SD018_WEIGHT,
        "explore_epsilon": EXPLORE_EPSILON,
        "residue_coarse_centers": RESIDUE_COARSE_CENTERS,
        "residue_fine_centers": RESIDUE_FINE_CENTERS,
        "residue_bandwidth": RESIDUE_BANDWIDTH,
        "min_discriminative_world_dim": MIN_DISCRIMINATIVE_WORLD_DIM,
        "thresholds": interp["thresholds"],
        "dry_run": bool(dry_run),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {
            # A diagnostic is scoring-excluded; it discriminates WHY the ceiling holds and
            # does not by itself move either claim's confidence.
            "Q-002": "unknown",
            "SD-031": "unknown",
        },
        "interpretation": interp,
        "arm_results": result["arm_results"],
        "label_balance": result["label_balance"],
        "custom_information": {
            "bears_on": (
                "INV-088 ANTECEDENT ONLY (that z_world differentiation is genuinely low). "
                "Does NOT bear on the INV-088 coupling leg -- V3-EXQ-744a's WEAKENS reading "
                "stands. Does NOT bear on MECH-459 / return-scale. Does NOT bear on either "
                "leg of the GOV-FANOUT-1 discrimination (V3-EXQ-780 vs V3-EXQ-781); both "
                "remain claimed and that discrimination remains open."
            ),
            "retest_spec_source": (
                "REE_assembly/evidence/planning/"
                "failure_autopsy_zworld-integration-cluster_2026-06-06.md section 7 "
                "'Retest spec (user-confirmed)'"
            ),
            "comparison_table_source": (
                "REE_assembly/evidence/experiments/"
                "zworld_near_static_characterisation_2026-07-18.md section 3a"
            ),
            "reference_2026_07_18_zworld_cr": 0.094,
            "reference_2026_07_18_raw_cr": 0.63,
            "reference_2026_07_18_attenuation_factor": 6.7,
            "gov_reuse_1_check": (
                "reanalysis_query.py query --readout zworld_contrast_ratio / zworld_eff_rank "
                "-> no substrate-compatible MATCH (all UNVERIFIABLE, pre-2026-07-12 manifests "
                "carry no recoverable substrate_hash); no recorded run measures contrast under "
                "a trained encoder at either dim. Not recoverable -> run."
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

    # NOTE: the always-record core (recording_schema / substrate_hash / machine /
    # machine_class / elapsed_seconds / config / seeds) is stamped by the sanctioned single
    # writer pack_writer.write_flat_manifest in main(), which calls stamp_recording_core
    # AFTER arm_results is assembled so substrate_hash HOISTS from the per-cell fingerprints
    # rather than being recomputed (and mismatching).
    return manifest, full_config


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started_at = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)
    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest, full_config = _build_manifest(result, timestamp_utc, args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        EVIDENCE_DIR,
        dry_run=args.dry_run,
        config=full_config,
        seeds=result["seeds_used"],
        script_path=Path(__file__),
        started_at=started_at,
        json_default=str,
    )

    print("", flush=True)
    print("=" * 72, flush=True)
    print("%s -- outcome=%s" % (QUEUE_ID, manifest["outcome"]), flush=True)
    print("self-route label: %s" % manifest["interpretation"]["label"], flush=True)
    for a in ARMS:
        crs = [
            r["zworld_contrast_ratio"] for r in result["arm_results"]
            if r["arm_id"] == a["arm_id"] and r.get("zworld_contrast_ratio") is not None
        ]
        if crs:
            print("  %-16s z_world CR mean=%.4f" % (a["arm_id"], float(np.mean(crs))), flush=True)
    print("manifest: %s" % out_path, flush=True)
    print("=" * 72, flush=True)

    return str(out_path), manifest["outcome"], args.dry_run


if __name__ == "__main__":
    _out_path, _outcome, _dry = main()
    _outcome_raw = str(_outcome).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry,
    )
