"""ARC-021 H1 (drive axis) -- encoder-unfrozen merged-channel instrument. NOT QUEUED.

STATUS: **NOT QUEUED. No EXQ number consumed, no queue entry, no coordinator row.** This
file is the INSTRUMENT that produced the refusal, kept under `_scratch/` exactly as the H2
leg's reproducer is (`_scratch/arc021_h2_merged_optimizer_runnability_probe.py`), so the
successor does not have to rebuild it. Record:
`REE_assembly/evidence/planning/arc021_h1_leg_refused_readout_dies_under_unfreeze_20260911.md`.

WHY IT WAS REFUSED, IN ONE LINE: unfreezing the encoder destroys the readout's signal in
BOTH arms, so the pre-registered criterion cannot fire in either direction.

=== WHAT THIS WOULD HAVE BEEN ===

The registry leg `H-encoder-level-merge-degrades` of question
`arc021_channel_separation_necessity`, on the DRIVE axis of the GOV-FANOUT-1 portfolio in
`failure_autopsy_V3-EXQ-993a_2026-09-05` section 6: "Same two arms with the encoder UNFROZEN
and jointly optimised in P1, so a collapsed objective can corrupt the latent itself.
Null: both mean paired diffs still > -0.15, no sign consistency."

Built from V3-EXQ-1011 (not 993a): its paired-CI verdict, its two falsifier DVs
(`calibration_gap` AND `attribution_auc`), its `--self-test` grid and its honest dv_bounds
handling are all strictly better than 993a's threshold-plus-sign-consistency rule. Changes:
P1 no longer freezes the encoder; each arm owns its encoder(s) inside its own optimizer(s);
a third DIAGNOSTIC arm (`SHARED_ENC_SEP_HEADS`) merges the ENCODER while leaving the heads
disjoint, so a positive result could be attributed to the encoder level -- the registry
leg's own name -- rather than to the trunk level 993a/1011 already measured as null.

=== THE MEASUREMENT THAT REFUSED IT (2026-09-11, four probes, all through THIS driver's own
`_run_cell` -- never a probe subclass; V3-EXQ-993a red-team F3) ===

PROBE 1, runnability (`_scratch/arc021_h1_runnability_probe.py`). All three arms run for 6
steps and the encoder DEMONSTRABLY RECEIVES GRADIENT in each. The H2 leg's ContextMemory
crash does not reproduce, as the 2026-09-10 pre-flight predicted: this driver imports only
`ree_core.environment.causal_grid_world`. **The leg is NOT substrate-blocked.** It also
confirms the unfreeze is not INERT -- deleting 993a's `requires_grad_(False)` loop alone
would have changed nothing, because the encoder calls sat inside a separate `no_grad` block.

PROBE 3, calibration at the production schedule (P0=80, P1=80, 120 steps; 32 cells, 8 seeds
DISJOINT from any scored seed set, both conditions, both criterion arms):

  condition  control mean gap   control max gap   control AUC   above chance
  DENSE            +0.02144           +0.06268        0.5322         0.032
  SPARSE           +0.01421           +0.04400        0.5409         0.041

against a FROZEN-encoder reference (V3-EXQ-1011, n=96) of 0.4147 / 0.4071 on the gap and
~0.69 on the AUC. **0 of 16 control cells reach SEPARATED_SIGNAL_FLOOR = 0.20**; the largest
is 0.0627, 3.2x below it.

PROBE 4, the decisive control (`_scratch/arc021_h1_frozen_control_probe.py`). The
alternative hypothesis -- that the collapse is a plumbing bug introduced while unfreezing --
is ruled out by running THIS driver with the encoder RE-FROZEN at `_encode`, leaving every
other line (heads, optimizers, clip, probes, verdict) on the shipped path:

  regime    arm          mean gap    mean AUC
  FROZEN    SEPARATED     +0.21685      0.652     <- reproduces the reference regime
  FROZEN    MERGED        +0.33038      0.725
  UNFROZEN  SEPARATED     +0.00242      0.513     <- 90x collapse (ratio 0.011)
  UNFROZEN  MERGED        +0.02604      0.533

So the plumbing is sound and the unfreeze is the cause.

PROBE 5, is it repairable (`_scratch/arc021_h1_anchor_probe.py`)? Control arm, DENSE, full
schedule, 3 seeds:

  mode                                      mean gap   mean AUC
  unfrozen (the leg as specified)            +0.00178    0.499    BELOW FLOOR
  anchored (P0 recon continued through P1)   +0.02456    0.563    BELOW FLOOR
  short    (unfrozen for the LAST 10 of 80)  +0.21158    0.675    CLEARS FLOOR

Anchoring the encoder with a maintained reconstruction term does NOT rescue it. What does is
shortening the unfreeze -- i.e. the signal loss is CUMULATIVE DRIFT over ~9,600 encoder
updates, not an instantaneous property of unfreezing.

PROBE 2, the detach decision (`_scratch/arc021_h1_detach_probe.py`), kept because it
settles a design question the successor will face again. With the encoder unfrozen both MSE
terms admit the trivial optimum "encoder emits a constant". Measured: undetached targets
collapse ARM_MERGED's latent dispersion 0.327 -> 0.048 (6.9x) while leaving ARM_SEPARATED
untouched (ratio 0.999) -- an ASYMMETRIC collapse in the treatment arm alone, which would
have manufactured a spurious result for a reason unrelated to channel incommensurability.
Hence `DETACH_MSE_TARGETS = True`: sensory target `sg(z)`, forward target produced under
`no_grad`, harm BCE fully differentiable into the encoder. Detaching is the best available
option and the readout still dies, so the refusal does not rest on this choice.

=== WHY THAT IS A REFUSAL AND NOT A BAR-CALIBRATION PROBLEM ===

Three of the five preconditions fail deterministically, on every seed, in both conditions,
BEFORE a single treatment cell is trained -- computed exactly as this driver computes them:

  dv_headroom_margin_room_below_control     range          0.09262  vs 0.30000   3.2x short
  dv_headroom_control_signal_floor_reachable max_abs       0.06268  vs 0.40000   6.4x short
  dv_headroom_auc_room_below_control        floor_headroom -0.01857 vs 0.10500   NEGATIVE
  non-degeneracy DENSE   control mean gap +0.02144 vs floor 0.20                 9x short
  non-degeneracy SPARSE  control mean gap +0.01421 vs floor 0.20                14x short

The run would self-route `substrate_not_ready_requeue` / `non_contributory` every time. The
pre-registered MARGIN of 0.15 is **1.6x the ENTIRE observed control range** -- a bar wider
than the instrument's whole dynamic range, which is the V3-EXQ-936a failure shape (a bar
~7,900x above the maximum attainable effect, logged clean for three governance cycles).

Lowering the floor or shrinking MARGIN to fit is refused on the standing rule: a
pre-registered value that provably fails a gate is a DESIGN-TIME PROOF, never a threshold to
lower. And the deeper point is that it is not a threshold problem at all -- BOTH arms read
within noise of zero, so their contrast measures the death of the readout, not channel
separation. (For the record, the sign is the same "wrong direction" 993a and 1011 both saw:
MERGED > SEPARATED by +0.037 DENSE / +0.125 SPARSE. At this scale that is uninterpretable.)

ALSO RECORDED, for the successor: the union grad-clip binds ASYMMETRICALLY once the encoder
is unfrozen -- 18.6% of steps in ARM_SEPARATED vs 44.2% in ARM_MERGED (mean pre-clip norms
1.71 vs 1.84). Under 993a the encoders were outside the clip entirely, so this asymmetry is
new and is a second, independent reason the leg as specified is not clean.

=== WHAT IS ACTUALLY OWED ===

`complex (probe-gated)`, not `complicated (buildable)`: is there an unfreeze DOSE at which
the encoder receives materially channel-shaped gradient AND the readout survives -- and at
that dose, is a merge-vs-separate contrast still a test of ARC-021 rather than a test of the
dose? Probe 5 shows the two ends (10/80 alive, 80/80 dead) and nothing in between. The
answer may well be NO, in which case H1 is unanswerable in this surrogate and the honest
route to ARC-021's necessity half is H2 -- the leg ARC-021's `what_would_answer` literally
names, currently substrate-blocked on the ContextMemory in-place write (GFLAG-0229). That
would be a real prioritisation finding, not a null.

Governance flag raised: `evidence_discrepancy`, ARC-021 / MECH-069.

=== RUNNING THIS FILE ===

`--self-test` walks the verdict cube and the criterion-can-fire arithmetic; `--dry-run` is a
smoke. A SCORED run is REFUSED (see main()): this driver is not queued, its bars were never
finalised against a live regime, and a manifest from it must not enter the evidence record.

SLEEP DRIVER: not applicable -- no sleep loop used in this probe.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# parents[2] (not [1]) -- this file lives under experiments/_scratch/, one level
# deeper than a queued driver, so the repo root is two hops up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import (  # noqa: E402
    P0NotReady,
    dv_headroom_check,
    p0_readiness_gate,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "arc021_h1_encoder_unfrozen_drive_axis_instrument"
QUEUE_ID = None   # NOT QUEUED -- no EXQ number was consumed. See the docstring.
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["ARC-021", "MECH-069"]
# NOT a supersession. V3-EXQ-993a (surrogate trunk merge, frozen encoder) and V3-EXQ-1011
# (the same topology at n=96 with a sub-margin CI) both stand exactly as recorded. This leg
# asks the DIFFERENT question the 993a autopsy's GOV-FANOUT-1 table puts on the DRIVE axis:
# with the encoder UNFROZEN, so a collapsed objective can corrupt the LATENT rather than only
# a downstream trunk, does merging degrade harm calibration?
SUPERSEDES = None
HYPOTHESIS_QID = "arc021_channel_separation_necessity"
HYPOTHESIS_LEG = "H-encoder-level-merge-degrades"

# ---- SEEDS: sized from THIS regime's MEASURED paired-diff SD, not inherited ------------
# failure_autopsy_V3-EXQ-993a_2026-09-05.md section 8 repair 5 is binding and explicit:
# "the successor must pre-register an effect its seed count can actually DETECT, not inherit
# 993a's -0.15/n=4". 993a's own power at its own pre-registered effect was ~12% JOINTLY.
#
# The registry leg H-encoder-level-merge-degrades pre-registers 0.15, so 0.15 is the effect
# this leg must be able to resolve. Sizing inputs, in order of authority:
#   (a) V3-EXQ-1011, n=96, FROZEN encoder, paired-diff SD: DENSE 0.1331 / SPARSE 0.1917.
#   (b) THIS driver's own pre-build probe, UNFROZEN encoder, run through `_run_cell`
#       (never a probe subclass -- 993a red-team F3): see PRE-BUILD MEASUREMENT in the
#       module docstring. That SD is the one the table below is computed on.
# The binding requirement is that a 95% CI on the paired mean must sit clear of -0.15 when
# the truth is no degradation, with room for the SD to be larger than measured.
SEEDS = [101 + 7 * i for i in range(48)]
CONDITIONS = ["DENSE", "SPARSE"]
# THREE arms. SEPARATED is the control and MERGED the treatment -- those two, and only those
# two, carry every pre-registered criterion and every precondition. SHARED_ENC_SEP_HEADS is a
# DIAGNOSTIC arm with no criterion of its own (see CRITERION_ARMS): it merges the ENCODER
# while leaving the three heads disjoint, which is the registry leg's own name
# (H-ENCODER-level-merge-degrades) in isolation. Without it a degradation could not be
# attributed to the encoder level rather than to the trunk level that 993a/1011 already
# measured as null -- so it is what makes a positive result interpretable, at 50% more
# compute and zero additional criterion surface.
ARMS = ["SEPARATED", "MERGED", "SHARED_ENC_SEP_HEADS"]
CRITERION_ARMS = ("SEPARATED", "MERGED")   # control, treatment. Read by EVERY criterion,
                                           # EVERY precondition and the verdict assembly.
DIAGNOSTIC_ARMS = ("SHARED_ENC_SEP_HEADS",)
CONTROL_ARM, TREATMENT_ARM = CRITERION_ARMS
NUM_HAZARDS = {"DENSE": 15, "SPARSE": 4}
GRID_SIZE = 12
OBS_RADIUS = 1
OBS_VIEW_SIZE = 2 * OBS_RADIUS + 1  # 3x3

WORLD_DIM = 32
HIDDEN_DIM = 64
P0_EPISODES = 80
P1_EPISODES = 80
STEPS_PER_EPISODE = 120
TOTAL_EPISODES_PER_CELL = P0_EPISODES + P1_EPISODES
PROBE_RESETS = 15
STAY_ACTION_IDX = 4  # env.ACTIONS[4] == (0, 0)

LR_HEAD = 1e-3
LR_ENCODER = 1e-3
GRAD_CLIP_NORM = 1.0

# ---- THE MANIPULATION, AND THE ONE DESIGN DECISION IT FORCES ---------------------------
# P1 no longer freezes the encoder. `DETACH_MSE_TARGETS` records, as a config field rather
# than as an implicit code fact, that the two MSE losses take DETACHED targets. See the
# module docstring section "WHY THE MSE TARGETS ARE DETACHED" -- undetached, both MSE terms
# admit the trivial global optimum "encoder emits a constant", which collapses the latent in
# EVERY arm and makes the run non-contributory by construction. Measured, not argued.
DETACH_MSE_TARGETS = True

CI_CONFIDENCE = 0.95
# BOTH of ARC-021's falsifier DVs. 993a read only the first and therefore could not have
# falsified the claim on the claim's own terms (its autopsy section 13 point 2).
DVS = ["calibration_gap", "attribution_auc"]

# ---- THE PRE-REGISTERED EFFECT, PER DV -------------------------------------------------
# MARGIN = 0.15 is the registry leg's own stated effect ("MERGED degrades harm calibration by
# >= 0.15 in both conditions"), on calibration_gap.
#
# The two DVs do NOT share a threshold -- V3-EXQ-1011 red-team F4: 0.15 of a logit difference
# and 0.15 of an AUC are not the same effect, and applying one number to both is assignment by
# assertion. 1011 set its AUC threshold as the SAME FRACTION OF THE ABOVE-CHANCE CONTROL
# SIGNAL that its gap threshold represented (1/8 of ~0.28 -> 0.035). This leg inherits that
# METHOD, not that number, and re-derives the fraction under the unfrozen regime from its own
# pre-build probe (module docstring, PRE-BUILD MEASUREMENT). Chance is 0.5, so "above-chance
# signal" is (auc - 0.5).
MARGIN = 0.15
MARGIN_AUC = 0.105                  # re-derived at authoring time; see the docstring
MARGIN_BY_DV = {"calibration_gap": MARGIN, "attribution_auc": MARGIN_AUC}

# ---- THE SD THE SEED COUNT IS SIZED AGAINST -------------------------------------------
# A CI criterion is only as good as the SD it was sized against, and V3-EXQ-1011's red-team
# F3 is the standing lesson: n=48 there was sized to the EDGE of an SD estimated from four
# paired diffs, and had to be raised to 96. These two constants are the PESSIMISTIC SDs this
# leg sizes to -- deliberately ABOVE both the measured frozen-encoder SDs (1011, n=96:
# gap 0.1331 DENSE / 0.1917 SPARSE; auc 0.0606 / 0.0725) and this driver's own unfrozen
# pre-build probe (module docstring, PRE-BUILD MEASUREMENT) -- so that the design still
# resolves the pre-registered effect if the unfrozen regime is noisier than measured.
# `--self-test` asserts the resulting CI half-width is strictly inside MARGIN at these SDs,
# so a later edit to SEEDS or to a MARGIN cannot silently un-power the run.
SD_PESSIMISTIC_GAP = 0.30
SD_PESSIMISTIC_AUC = 0.12

SEPARATED_SIGNAL_FLOOR = 0.20       # non-degeneracy per condition, on the CONTROL arm's MEAN
                                    # calibration_gap. Carried from 993a/1011 unchanged and
                                    # deliberately > nothing it gates: it is INDEPENDENT of
                                    # the criterion (993a red-team F10), so a control arm that
                                    # merely fails to produce signal cannot pass by default.
HARM_ACTION_SENSITIVITY_FLOOR = 0.05  # precondition 1, on the CONTROL arm only (993a F2).
                                    # NOTE the standing thin-margin warning: V3-EXQ-1011
                                    # cleared this at 0.0505 vs 0.05, a 1.01x margin on one
                                    # worst cell, under the FROZEN encoder. Under this leg the
                                    # harm encoder is trained by the harm BCE alone, which
                                    # should raise it -- but that is a prediction, and the gate
                                    # is what checks it.
HARM_EVENTS_FLOOR_PER_CONDITION = 10  # precondition 2, MIN across the two conditions (993 F7)
MIN_PROBE_COVERAGE = 5              # per-cell minimum n_near / n_safe to count (993 F5)

# SMOKE POSITIVE CONTROL, dry-run only. At --dry-run scale the readiness gate ALWAYS fires
# (2 episodes produce no harm events and a near-zero DV), so a plain dry-run exercises only
# the gate-blocked branch and leaves the whole treatment path -- the unfrozen train_step, the
# three arm topologies, the paired-CI verdict -- unexecuted. That is exactly the V3-EXQ-591g
# short-circuit-hides-a-crash shape, on the very code this letter exists to add.
# `--smoke-force-gate` forces BOTH staged gates green so the smoke covers the full path as a
# POSITIVE control (V3-EXQ-1011 red-team F2: forcing only stage A left the CI code unrun).
# It is refused outside --dry-run, and the runner never passes it.
_SMOKE_FORCE_GATE = False


class SmallViewEnv(CausalGridWorld):
    """CausalGridWorld with a 3x3 local observation (radius 1) instead of
    the 5x5 default. Ported from V3-EXQ-010's `SmallViewEnv` (the one
    config with positive-direction SD-003 calibration_gap signal), with
    `num_hazards` taken from the constructor kwargs rather than a module
    constant, so it can serve both the DENSE and SPARSE conditions.
    Experiment-local -- does not modify the base class.
    """

    def __init__(self, **kwargs):
        view = OBS_VIEW_SIZE
        self._body_obs_dim = 10  # same as base
        self._world_obs_dim = view * view * 7 + view * view  # placeholder pre-super()
        super().__init__(**kwargs)
        self._world_obs_dim = view * view * self.NUM_ENTITY_TYPES + view * view

    @property
    def world_obs_dim(self) -> int:
        return self._world_obs_dim

    @property
    def body_obs_dim(self) -> int:
        return self._body_obs_dim

    def _get_observation_dict(self) -> Dict[str, torch.Tensor]:
        ax, ay = self.agent_x, self.agent_y
        r = OBS_RADIUS
        view = OBS_VIEW_SIZE

        body = torch.zeros(self._body_obs_dim)
        body[0] = ax / self.size
        body[1] = ay / self.size
        body[2] = self.agent_health
        body[3] = self.agent_energy
        max_vis = max(1, self.footprint_grid.max())
        body[4] = float(self.footprint_grid[ax, ay]) / max_vis
        action_enc = self._last_action if self._last_action < 4 else 0
        body[5 + action_enc] = 1.0
        body[9] = min(1.0, self.steps / 500.0)

        local_view = torch.zeros(view, view, self.NUM_ENTITY_TYPES)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = ax + di, ay + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    etype = self.grid[ni, nj]
                else:
                    etype = self.ENTITY_TYPES["wall"]
                local_view[di + r, dj + r, etype] = 1.0
        local_view_flat = local_view.reshape(-1)

        cont_view = torch.zeros(view, view)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = ax + di, ay + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    cont_view[di + r, dj + r] = float(self.contamination_grid[ni, nj])
        cont_view_flat = (cont_view / (self.contamination_threshold + 1e-6)).reshape(-1)

        world_state = torch.cat([local_view_flat, cont_view_flat])

        return {
            "body_state": body.float(),
            "world_state": world_state.float(),
            "contamination_view": cont_view_flat.float(),
        }


def _action_onehot(action_idx: int, num_actions: int) -> torch.Tensor:
    v = torch.zeros(1, num_actions)
    v[0, action_idx] = 1.0
    return v


def _random_cf_action(actual_idx: int, num_actions: int) -> int:
    """A counterfactual MOVEMENT action, never STAY (red-team F8).

    993 drew the counterfactual from all actions including STAY (index 4) while
    the ACTUAL action was always a movement -- near-hazard probes step into the
    hazard, safe probes draw randint(0, action_dim-2). With p=1/4 the contrast
    was therefore P(harm | move) - P(harm | STAY) rather than a
    movement-versus-movement counterfactual, and STAY on a cell whose
    contamination has crossed threshold is itself a harm transition
    (causal_grid_world.py:2421 retypes the vacated cell, :2497 then fires).
    Measured consequence in 993a's own design probe: in SPARSE the near-hazard
    mean causal_sig was itself <= 0 on 3 of 4 seeds, so the positive gap was
    being carried by the safe population's STAY counterfactual rather than by
    hazard-approach discrimination. Restricting the counterfactual to movement
    matches the actual-action distribution and makes the contrast the one the
    claim is about."""
    choices = [a for a in range(num_actions) if a != actual_idx and a != STAY_ACTION_IDX]
    return random.choice(choices) if choices else 0


def _make_world_codec(world_obs_dim: int) -> Tuple[nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Linear(world_obs_dim, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, WORLD_DIM),
    )
    decoder = nn.Sequential(
        nn.Linear(WORLD_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, world_obs_dim),
    )
    return encoder, decoder


def _sensory_head() -> nn.Module:
    return nn.Sequential(
        nn.Linear(WORLD_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, WORLD_DIM),
    )


def _forward_head(action_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(WORLD_DIM + action_dim, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, WORLD_DIM),
    )


def _harm_head(action_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(WORLD_DIM + action_dim, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, 1),
    )


def _clone_encoder(src: nn.Module) -> nn.Module:
    """An exact structural + parameter copy of the P0-warmed encoder.

    `copy.deepcopy` of an `nn.Sequential` consumes NO RNG (it copies tensors; it does not
    re-initialise), which is what preserves the bit-identical-P0 property across arms: at a
    given (condition, seed) every arm enters P1 from the identical warmed representation and
    with the identical torch RNG state, so the only degree of freedom is P1's topology.
    """
    dst = copy.deepcopy(src)
    for p in dst.parameters():
        p.requires_grad_(True)
    dst.train()
    return dst


class _ChannelsBase:
    """Common P1 step for all three arms.

    THE ONLY THING THAT DIFFERS BETWEEN ARMS IS WHICH PARAMETERS ARE SHARED.

    Every arm computes the same three losses from the same data, sums them, takes ONE
    backward over the sum, applies ONE union `clip_grad_norm_` at GRAD_CLIP_NORM, and steps
    its own optimizers. A single `total.backward()` is EXACTLY EQUIVALENT to V3-EXQ-993a's
    three separate backward passes wherever the parameter sets are disjoint -- autograd
    accumulates into each leaf only from the loss terms whose graph reaches it -- so for
    ARM_SEPARATED no gradient from any loss touches another channel's parameters, which is
    the separate-channels premise, exactly as before. Using one backward uniformly removes an
    arm-dependent difference in graph retention that would otherwise be a silent asymmetry,
    and it is a simplification of the mechanics, never of the science.

    Adam has no cross-parameter coupling, so N Adam instances over N disjoint parameter sets
    are numerically identical to one Adam over their union. The optimizer COUNT is therefore
    not itself a manipulation; what is manipulated is which parameters are shared, and hence
    which parameters receive gradient from more than one loss term. That is precisely
    ARC-021's content.

    `harm_encoder` is the encoder on the HARM READOUT PATH -- the one `_eval_probes` and
    `_harm_action_sensitivity` must use. Reading it off the channels object rather than
    passing an encoder in is what stops an arm being probed through an encoder no criterion
    ever trained.
    """

    arm: str = ""
    harm_encoder: nn.Module

    # -- subclass contract -------------------------------------------------------------
    def _encode(self, world_obs_t: torch.Tensor, world_obs_t1: torch.Tensor):
        """Return (z_sensory, z_forward, z_forward_next_DETACHED, z_harm)."""
        raise NotImplementedError

    def _optimizers(self) -> List[optim.Optimizer]:
        raise NotImplementedError

    def _all_params(self) -> List[torch.nn.Parameter]:
        raise NotImplementedError

    def predict_sensory_from_z(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_forward_from_z(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def harm_logit(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def modules(self) -> List[nn.Module]:
        raise NotImplementedError

    # -- shared machinery --------------------------------------------------------------
    def eval_mode(self) -> None:
        for m in self.modules():
            m.eval()

    def train_mode(self) -> None:
        for m in self.modules():
            m.train()

    def train_step(
        self, world_obs_t: torch.Tensor, action_onehot: torch.Tensor,
        world_obs_t1: torch.Tensor, harm_label: torch.Tensor,
    ) -> Dict[str, float]:
        z_s, z_f, z_f1, z_h = self._encode(world_obs_t, world_obs_t1)

        # SENSORY: self-reconstruction in latent space, z -> z (993a red-team fix F1 -- a
        # z_t1 target would make an action-free head a degenerate proxy for the separate
        # forward task). The TARGET is detached: see the module docstring, "WHY THE MSE
        # TARGETS ARE DETACHED". Undetached, (encoder, head) -> (constant, identity) drives
        # this to exactly zero and the latent collapses.
        sensory_target = z_s.detach() if DETACH_MSE_TARGETS else z_s
        sensory_loss = F.mse_loss(self.predict_sensory_from_z(z_s), sensory_target)

        # FORWARD: z_t1 is produced under no_grad by `_encode`, which IS the detach. Without
        # it the encoder minimises this by making the future trivially predictable -- the
        # standard self-predictive collapse a stop-gradient exists to prevent.
        forward_loss = F.mse_loss(self.predict_forward_from_z(z_f, action_onehot), z_f1)

        # HARM: fully differentiable end-to-end into the encoder. This is the ONE gradient
        # that is supposed to shape the latent toward harm discrimination, and it is
        # deliberately NOT detached anywhere -- detaching it would remove the manipulation.
        # harm_label is ALWAYS 0.0 or 1.0, never skipped (993a red-team fix F2).
        harm_loss = F.binary_cross_entropy_with_logits(
            self.harm_logit(z_h, action_onehot), harm_label)

        total = sensory_loss + forward_loss + harm_loss
        opts = self._optimizers()
        for o in opts:
            o.zero_grad()
        total.backward()
        # ONE union clip per arm, at the same norm in every arm (993a red-team fix F5: an
        # arm clipped per-component is under a strictly weaker constraint than an arm
        # clipped as a union, and that difference is not ARC-021's content). Under an
        # unfrozen encoder the union is larger for ARM_SEPARATED (three encoders) than for
        # ARM_MERGED (one), so the clip could in principle bind asymmetrically -- which is a
        # measurable fact, not an argument, and `clip_active_frac` / `mean_pre_clip_norm`
        # are recorded per cell so a reader can check whether it ever bound at all.
        pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(self._all_params(), GRAD_CLIP_NORM))
        for o in opts:
            o.step()
        return {
            "sensory_loss": float(sensory_loss.item()),
            "forward_loss": float(forward_loss.item()),
            "harm_loss": float(harm_loss.item()),
            "pre_clip_norm": pre_clip_norm,
            "clipped": 1.0 if pre_clip_norm > GRAD_CLIP_NORM else 0.0,
        }


class SeparatedChannels(_ChannelsBase):
    """ARM_SEPARATED (CONTROL). Three channels, each with its OWN encoder, its own head,
    its own loss and its own optimizer. No parameter is shared between channels.

    This is V3-EXQ-993a's SEPARATED arm with the encoder unfrozen, and the encoder folded
    into each channel's already-disjoint parameter set -- which is the faithful extension,
    not an extra change. Under 993a the single encoder was FROZEN, and a frozen shared
    encoder is functionally indistinguishable from three frozen identical copies; unfreezing
    is what makes encoder topology meaningful at all, and the topology then has to follow the
    arm's own premise. Giving this arm ONE jointly-updated encoder instead would ADD a shared
    parameter it did not have in 993a -- a larger deviation, and a partial merge inside the
    control.

    NO CAPACITY CONFOUND ON THE READOUT PATH. The harm readout traverses exactly
    encoder(obs -> 64 -> 32) + harm_head(32+A -> 64 -> 64 -> 1) here, and exactly
    encoder(obs -> 64 -> 32) + trunk(32+A -> 64 -> 64) + harm_out(64 -> 1) in ARM_MERGED:
    the SAME depth and the SAME parameter count. The two extra encoders in this arm are not
    on the harm readout path at all. What differs is only which gradients shaped those
    parameters.
    """

    arm = "SEPARATED"

    def __init__(self, action_dim: int, base_encoder: nn.Module):
        self.action_dim = action_dim
        # Head construction order is IDENTICAL to SharedEncoderSeparateHeads', so at a given
        # seed those two arms receive bit-identical head inits and differ ONLY in encoder
        # topology. That makes the SEPARATED-vs-SHARED_ENC diagnostic contrast exactly
        # controlled at init, which is the whole point of carrying it.
        self.sensory_head = _sensory_head()
        self.forward_head = _forward_head(action_dim)
        self.harm_head = _harm_head(action_dim)
        self.enc_sensory = _clone_encoder(base_encoder)
        self.enc_forward = _clone_encoder(base_encoder)
        self.enc_harm = _clone_encoder(base_encoder)
        self.harm_encoder = self.enc_harm
        self.opt_sensory = optim.Adam(
            [{"params": list(self.enc_sensory.parameters()), "lr": LR_ENCODER},
             {"params": list(self.sensory_head.parameters()), "lr": LR_HEAD}])
        self.opt_forward = optim.Adam(
            [{"params": list(self.enc_forward.parameters()), "lr": LR_ENCODER},
             {"params": list(self.forward_head.parameters()), "lr": LR_HEAD}])
        self.opt_harm = optim.Adam(
            [{"params": list(self.enc_harm.parameters()), "lr": LR_ENCODER},
             {"params": list(self.harm_head.parameters()), "lr": LR_HEAD}])

    def modules(self) -> List[nn.Module]:
        return [self.enc_sensory, self.enc_forward, self.enc_harm,
                self.sensory_head, self.forward_head, self.harm_head]

    def _optimizers(self) -> List[optim.Optimizer]:
        return [self.opt_sensory, self.opt_forward, self.opt_harm]

    def _all_params(self) -> List[torch.nn.Parameter]:
        out: List[torch.nn.Parameter] = []
        for m in self.modules():
            out.extend(m.parameters())
        return out

    def _encode(self, world_obs_t: torch.Tensor, world_obs_t1: torch.Tensor):
        z_s = self.enc_sensory(world_obs_t)
        z_f = self.enc_forward(world_obs_t)
        with torch.no_grad():
            z_f1 = self.enc_forward(world_obs_t1)
        z_h = self.enc_harm(world_obs_t)
        return z_s, z_f, z_f1, z_h

    def predict_sensory_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.sensory_head(z)

    def predict_forward_from_z(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.forward_head(torch.cat([z, action_onehot], dim=-1))

    def harm_logit(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.harm_head(torch.cat([z, action_onehot], dim=-1))


class MergedChannels(_ChannelsBase):
    """ARM_MERGED (TREATMENT). ONE encoder and ONE shared trunk carrying all three losses,
    under ONE optimizer, with the three losses summed.

    This is V3-EXQ-993a's MERGED arm with the encoder unfrozen and folded into the single
    optimizer -- a pure unfreeze, nothing else moved. It is where the drive-axis manipulation
    bites: the harm readout's representation is now shaped by the sensory and forward
    gradients all the way down into the LATENT, not merely in a downstream 2-layer trunk.
    That confinement to a downstream trunk is the exact limitation the 993a autopsy names as
    the reason its null may be a translation artefact (its section 13 point 3).
    """

    arm = "MERGED"

    def __init__(self, action_dim: int, base_encoder: nn.Module):
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(WORLD_DIM + action_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )
        self.sensory_out = nn.Linear(HIDDEN_DIM, WORLD_DIM)
        self.forward_out = nn.Linear(HIDDEN_DIM, WORLD_DIM)
        self.harm_out = nn.Linear(HIDDEN_DIM, 1)
        self.encoder = _clone_encoder(base_encoder)
        self.harm_encoder = self.encoder
        head_params = (list(self.trunk.parameters()) + list(self.sensory_out.parameters())
                       + list(self.forward_out.parameters()) + list(self.harm_out.parameters()))
        self.opt = optim.Adam(
            [{"params": list(self.encoder.parameters()), "lr": LR_ENCODER},
             {"params": head_params, "lr": LR_HEAD}])
        self._stay = _action_onehot(STAY_ACTION_IDX, action_dim)

    def modules(self) -> List[nn.Module]:
        return [self.encoder, self.trunk, self.sensory_out, self.forward_out, self.harm_out]

    def _optimizers(self) -> List[optim.Optimizer]:
        return [self.opt]

    def _all_params(self) -> List[torch.nn.Parameter]:
        out: List[torch.nn.Parameter] = []
        for m in self.modules():
            out.extend(m.parameters())
        return out

    def _stay_onehot(self, batch_size: int) -> torch.Tensor:
        return self._stay.expand(batch_size, -1)

    def _encode(self, world_obs_t: torch.Tensor, world_obs_t1: torch.Tensor):
        z = self.encoder(world_obs_t)
        with torch.no_grad():
            z1 = self.encoder(world_obs_t1)
        return z, z, z1, z

    def predict_sensory_from_z(self, z: torch.Tensor) -> torch.Tensor:
        # STAY is the "evaluate this state, no action" convention, matching the sensory
        # head's action-free role in ARM_SEPARATED (993a red-team fix F1).
        repr_ = self.trunk(torch.cat([z, self._stay_onehot(z.shape[0])], dim=-1))
        return self.sensory_out(repr_)

    def predict_forward_from_z(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        repr_ = self.trunk(torch.cat([z, action_onehot], dim=-1))
        return self.forward_out(repr_)

    def harm_logit(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        repr_ = self.trunk(torch.cat([z, action_onehot], dim=-1))
        return self.harm_out(repr_)


class SharedEncoderSeparateHeads(_ChannelsBase):
    """ARM_SHARED_ENC_SEP_HEADS (DIAGNOSTIC ONLY -- carries NO pre-registered criterion and
    is excluded from every precondition; see CRITERION_ARMS).

    ONE encoder, shared by all three channels and receiving the SUMMED gradient, with three
    otherwise-disjoint heads and three head-local optimizers. It merges the ENCODER and
    nothing else.

    WHY IT IS HERE. The registry leg is named `H-encoder-level-merge-degrades`, and
    V3-EXQ-993a/1011 already measured the TRUNK-level merge (frozen encoder) as null at
    n=4 and n=96. Without this arm, a `confirmed` reading on SEPARATED-vs-MERGED could not
    be attributed to the encoder level rather than to some interaction with the trunk; with
    it, the decomposition is read directly off the two contrasts. It costs 50% more cells and
    adds no criterion surface at all.
    """

    arm = "SHARED_ENC_SEP_HEADS"

    def __init__(self, action_dim: int, base_encoder: nn.Module):
        self.action_dim = action_dim
        # SAME construction order and shapes as SeparatedChannels -> bit-identical head init
        # at a given seed, so this contrast isolates encoder topology exactly.
        self.sensory_head = _sensory_head()
        self.forward_head = _forward_head(action_dim)
        self.harm_head = _harm_head(action_dim)
        self.encoder = _clone_encoder(base_encoder)
        self.harm_encoder = self.encoder
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=LR_ENCODER)
        self.opt_sensory = optim.Adam(self.sensory_head.parameters(), lr=LR_HEAD)
        self.opt_forward = optim.Adam(self.forward_head.parameters(), lr=LR_HEAD)
        self.opt_harm = optim.Adam(self.harm_head.parameters(), lr=LR_HEAD)

    def modules(self) -> List[nn.Module]:
        return [self.encoder, self.sensory_head, self.forward_head, self.harm_head]

    def _optimizers(self) -> List[optim.Optimizer]:
        return [self.opt_encoder, self.opt_sensory, self.opt_forward, self.opt_harm]

    def _all_params(self) -> List[torch.nn.Parameter]:
        out: List[torch.nn.Parameter] = []
        for m in self.modules():
            out.extend(m.parameters())
        return out

    def _encode(self, world_obs_t: torch.Tensor, world_obs_t1: torch.Tensor):
        z = self.encoder(world_obs_t)
        with torch.no_grad():
            z1 = self.encoder(world_obs_t1)
        return z, z, z1, z

    def predict_sensory_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.sensory_head(z)

    def predict_forward_from_z(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.forward_head(torch.cat([z, action_onehot], dim=-1))

    def harm_logit(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.harm_head(torch.cat([z, action_onehot], dim=-1))


_ARM_CLASSES = {
    "SEPARATED": SeparatedChannels,
    "MERGED": MergedChannels,
    "SHARED_ENC_SEP_HEADS": SharedEncoderSeparateHeads,
}


def _make_channels(arm: str, action_dim: int, base_encoder: nn.Module) -> _ChannelsBase:
    try:
        cls = _ARM_CLASSES[arm]
    except KeyError:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(_ARM_CLASSES)}") from None
    return cls(action_dim, base_encoder)


def _collect_random_episode(env: SmallViewEnv) -> List[Tuple[torch.Tensor, int, torch.Tensor, float]]:
    """One episode of uniform-random-policy rollout. Returns a list of
    (world_obs_t, action_idx_t, world_obs_t1, harm_signal_t) tuples."""
    _, obs_dict = env.reset()
    transitions = []
    for _step in range(STEPS_PER_EPISODE):
        world_obs_t = obs_dict["world_state"]
        action_idx = random.randint(0, env.action_dim - 1)
        _, harm_signal, done, _info, obs_dict = env.step(action_idx)
        world_obs_t1 = obs_dict["world_state"]
        transitions.append((world_obs_t, action_idx, world_obs_t1, float(harm_signal)))
        if done:
            break
    return transitions


def _run_p0(env: SmallViewEnv, encoder: nn.Module, decoder: nn.Module, opt: optim.Optimizer) -> Dict[str, float]:
    encoder.train()
    decoder.train()
    total_recon = 0.0
    n = 0
    for ep in range(P0_EPISODES):
        transitions = _collect_random_episode(env)
        for world_obs_t, _a, _world_obs_t1, _h in transitions:
            w = world_obs_t.unsqueeze(0)
            z = encoder(w)
            recon = decoder(z)
            loss = F.mse_loss(recon, w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()
            total_recon += float(loss.item())
            n += 1
        if (ep + 1) % 20 == 0 or ep == P0_EPISODES - 1:
            print(
                f"  [train] p0 ep {ep + 1}/{TOTAL_EPISODES_PER_CELL} "
                f"mean_recon={total_recon / max(1, n):.5f}",
                flush=True,
            )
    return {"mean_recon_loss": total_recon / max(1, n)}


def _run_p1(env: SmallViewEnv, channels: "_ChannelsBase", action_dim: int) -> Dict[str, Any]:
    """P1 with the encoder UNFROZEN -- the whole manipulation of this leg.

    V3-EXQ-993a/1011 did three things here that this driver deliberately does NOT do, and
    removing any ONE of them alone would have been inert (the 2026-09-10 pre-flight verdict
    recorded exactly this trap):
      (1) `encoder.eval()` + `p.requires_grad_(False)` over the encoder parameters;
      (2) computing z_t / z_t1 inside `with torch.no_grad():`, so the encoder outputs carried
          no grad_fn at all -- deleting only (1) would have changed NOTHING;
      (3) holding no encoder parameter in any optimizer.
    All three are gone: each arm's channels object owns its own encoder(s), holds them in its
    own optimizer(s), and encodes inside the autograd graph.

    The FORWARD target z_t1 is still produced under no_grad inside each arm's `_encode` --
    that is the deliberate stop-gradient on the TARGET, not a residue of the freeze. See the
    module docstring, "WHY THE MSE TARGETS ARE DETACHED".
    """
    channels.train_mode()
    total_harm_events = 0
    n_steps = 0
    n_clipped = 0.0
    sum_pre_clip_norm = 0.0
    sum_harm_loss = 0.0
    for ep in range(P1_EPISODES):
        transitions = _collect_random_episode(env)
        for world_obs_t, action_idx, world_obs_t1, harm_signal in transitions:
            action_onehot = _action_onehot(action_idx, action_dim)
            is_harm = harm_signal < 0
            if is_harm:
                total_harm_events += 1
            harm_label = torch.ones(1, 1) if is_harm else torch.zeros(1, 1)
            stats = channels.train_step(
                world_obs_t.unsqueeze(0), action_onehot, world_obs_t1.unsqueeze(0), harm_label)
            n_steps += 1
            n_clipped += stats["clipped"]
            sum_pre_clip_norm += stats["pre_clip_norm"]
            sum_harm_loss += stats["harm_loss"]
        if (ep + 1) % 20 == 0 or ep == P1_EPISODES - 1:
            print(
                f"  [train] p1 ep {P0_EPISODES + ep + 1}/{TOTAL_EPISODES_PER_CELL} "
                f"harm_events={total_harm_events}",
                flush=True,
            )
    # LATENT-COLLAPSE DIAGNOSTIC, measured on the HARM readout path, which is the only path
    # any criterion reads. A collapsed latent is the failure mode the detached MSE targets
    # exist to prevent, so the run must RECORD whether it happened rather than assume it did
    # not. `zharm_dispersion` is the mean per-dimension standard deviation of z_harm over a
    # fixed 64-observation probe batch; it goes to ~0 exactly when the encoder has collapsed
    # to a constant. It is a DIAGNOSTIC, not a gate: the load-bearing readiness gate is
    # `harm_head_action_sensitivity_present`, which is downstream of collapse and is the
    # statistic the criteria actually route on (the V3-EXQ-643 same-statistic rule).
    channels.eval_mode()
    with torch.no_grad():
        probe = torch.cat([env.reset()[1]["world_state"].unsqueeze(0) for _ in range(64)], dim=0)
        z = channels.harm_encoder(probe)
        zharm_dispersion = float(z.std(dim=0).mean().item())
        zharm_abs_mean = float(z.abs().mean().item())
    return {
        "p1_harm_events": total_harm_events,
        "p1_steps": n_steps,
        "clip_active_frac": (n_clipped / n_steps) if n_steps else None,
        "mean_pre_clip_norm": (sum_pre_clip_norm / n_steps) if n_steps else None,
        "mean_harm_loss": (sum_harm_loss / n_steps) if n_steps else None,
        "zharm_dispersion": zharm_dispersion,
        "zharm_abs_mean": zharm_abs_mean,
    }


def _harm_action_sensitivity(encoder: nn.Module, channels, env: SmallViewEnv, action_dim: int) -> float:
    """Positive control on the READOUT ITSELF: mean
    |harm_logit(z,a1) - harm_logit(z,a2)| over a fixed probe batch and two
    distinct movement actions, measured immediately after P1.

    This asserts the SAME statistic the load-bearing criteria consume (a
    difference of two harm logits at one state under two actions), not a
    magnitude proxy on a different component. 993 asserted the FORWARD head's
    action sensitivity, which was 17.9x over its floor while the readout the
    criteria actually read was flat -- the precondition certified a component
    the verdict never touched. An action-blind harm head makes causal_sig
    structurally zero in BOTH arms, so a below-floor reading means the
    substrate is not ready, never that ARC-021 was falsified."""
    with torch.no_grad():
        probe_worlds = []
        for _ in range(16):
            _, obs_dict = env.reset()
            probe_worlds.append(obs_dict["world_state"].unsqueeze(0))
        z = encoder(torch.cat(probe_worlds, dim=0))
        a1 = _action_onehot(0, action_dim).expand(z.shape[0], -1)
        a2 = _action_onehot(1, action_dim).expand(z.shape[0], -1)
        diff = channels.harm_logit(z, a1) - channels.harm_logit(z, a2)
        return float(diff.abs().mean().item())


def _auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """P(a random `pos` scores above a random `neg`), ties at half. Chance = 0.5.

    The rank form of the same evidence `calibration_gap` reads as a difference of means, so it
    is invariant to any monotone rescaling of the signature -- which is exactly the failure mode
    a mean-difference DV cannot see.
    """
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


H1_READINGS = ("eliminated", "confirmed", "undetermined")


def _h1_verdict(readings: Sequence[str]) -> Tuple[str, str, str]:
    """(outcome, label, evidence_direction) from the per-(DV, condition) H1 readings.

    PURE, and extracted so it can be self-tested. V3-EXQ-1011's own red-team caught a SIGN
    INVERSION when V3-EXQ-993a's verdict assembly was reused verbatim -- there
    `criterion_passed` meant "MERGED degraded" and here it means "the pre-registered
    degradation was ELIMINATED", the opposite claim -- which would have recorded a clean null
    as `supports`. This function plus the `--self-test` grid is what stops that recurring.

    Priority order, and each branch's reasoning:
      any CONFIRMED   -> the 95% CI on the paired mean lies entirely BELOW -MARGIN for some
                         (DV, condition): an encoder-level merge degrades harm calibration by
                         at least the registry leg's pre-registered effect. SUPPORTS ARC-021's
                         necessity half, and localises the effect to representation learning,
                         which is where the biology puts it.
      all ELIMINATED  -> every CI lies entirely ABOVE -MARGIN: a degradation of the
                         pre-registered size is RULED OUT on both DVs in both conditions,
                         even with the merge extended all the way into the latent. This is the
                         registry leg's own stated null, and it WEAKENS ARC-021.
      otherwise       -> at least one interval STRADDLED -MARGIN. Under-powered for the effect
                         it was sized against, which is the ABSENCE of a measurement, not
                         evidence in either direction. `non_contributory` -- never `weakens`,
                         which is exactly how an under-powered run gets logged as a null.
    """
    if any(r == "confirmed" for r in readings):
        return ("PASS", "encoder_level_merge_degrades_confirmed", "supports")
    if readings and all(r == "eliminated" for r in readings):
        return ("PASS", "encoder_level_merge_degradation_ruled_out", "weakens")
    return ("FAIL", "underpowered_ci_straddles_margin", "non_contributory")


def _self_test_h1() -> int:
    """Prove BOTH directions are reachable and that under-powered never reads as evidence."""
    fails = 0

    def chk(name: str, ok: bool, detail: Any = "") -> None:
        nonlocal fails
        if not ok:
            fails += 1
        print(f"  [self-test] {'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)

    E, C, U = "eliminated", "confirmed", "undetermined"
    chk("all eliminated -> weakens/PASS", _h1_verdict([E, E, E, E])
        == ("PASS", "encoder_level_merge_degradation_ruled_out", "weakens"))
    chk("any confirmed -> supports/PASS", _h1_verdict([E, E, E, C])
        == ("PASS", "encoder_level_merge_degrades_confirmed", "supports"))
    chk("confirmed dominates undetermined", _h1_verdict([U, U, U, C])
        == ("PASS", "encoder_level_merge_degrades_confirmed", "supports"))
    chk("any undetermined (no confirmed) -> non_contributory/FAIL",
        _h1_verdict([E, E, E, U])
        == ("FAIL", "underpowered_ci_straddles_margin", "non_contributory"))
    chk("all undetermined -> non_contributory, NEVER weakens", _h1_verdict([U, U, U, U])
        == ("FAIL", "underpowered_ci_straddles_margin", "non_contributory"))
    import itertools
    bad = []
    for combo in itertools.product(H1_READINGS, repeat=4):
        _o, _l, d = _h1_verdict(list(combo))
        if d == "supports" and "confirmed" not in combo:
            bad.append(combo)
        if d == "weakens" and set(combo) != {"eliminated"}:
            bad.append(combo)
        if "undetermined" in combo and "confirmed" not in combo and d != "non_contributory":
            bad.append(combo)
    chk("cube (3^4): supports iff some confirmed; weakens iff all eliminated; "
        "undetermined-without-confirmed is always non_contributory", not bad, bad[:4])

    # ---- THE CRITERION CAN FIRE IN BOTH DIRECTIONS, ARITHMETICALLY ---------------------
    # The question the 2026-09-08 batch failed on is "could this criterion have failed?".
    # Assert it on numbers rather than on prose: build the exact CI the driver builds, from
    # synthetic paired diffs, and check that a no-degradation sample reads `eliminated` and a
    # large-degradation sample reads `confirmed` at THIS design's n and a pessimistic SD.
    def _reading(diffs: Sequence[float], dv: str) -> str:
        ci = _paired_ci(list(diffs))
        lo, hi, thr = ci["ci_low"], ci["ci_high"], -MARGIN_BY_DV[dv]
        if lo is None or hi is None:
            return "undetermined"
        return "eliminated" if lo > thr else ("confirmed" if hi < thr else "undetermined")

    n = len(SEEDS)
    # A null sample: alternating +/- SD_PESSIMISTIC around zero -> mean 0, sd ~ SD_PESSIMISTIC.
    null_gap = [SD_PESSIMISTIC_GAP * (1 if i % 2 else -1) for i in range(n)]
    big_gap = [-4.0 * MARGIN + SD_PESSIMISTIC_GAP * (1 if i % 2 else -1) for i in range(n)]
    chk("null sample at the pessimistic SD reads ELIMINATED (the criterion CAN pass)",
        _reading(null_gap, "calibration_gap") == "eliminated",
        {"n": n, "sd": SD_PESSIMISTIC_GAP, "reading": _reading(null_gap, "calibration_gap")})
    chk("a 4x-MARGIN degradation reads CONFIRMED (the criterion CAN fail the null)",
        _reading(big_gap, "calibration_gap") == "confirmed",
        {"reading": _reading(big_gap, "calibration_gap")})
    null_auc = [SD_PESSIMISTIC_AUC * (1 if i % 2 else -1) for i in range(n)]
    big_auc = [-4.0 * MARGIN_AUC + SD_PESSIMISTIC_AUC * (1 if i % 2 else -1) for i in range(n)]
    chk("AUC: null reads ELIMINATED", _reading(null_auc, "attribution_auc") == "eliminated",
        {"sd": SD_PESSIMISTIC_AUC})
    chk("AUC: a 4x-MARGIN_AUC degradation reads CONFIRMED",
        _reading(big_auc, "attribution_auc") == "confirmed")
    # ...and that the two readings are mutually exclusive by construction.
    chk("eliminated and confirmed are mutually exclusive (thr is a single point)",
        all(not (r == "eliminated" and r == "confirmed") for r in ("eliminated", "confirmed")))

    # Per-DV thresholds are distinct and on each DV's own scale (1011 red-team F4).
    chk("per-DV thresholds are distinct",
        MARGIN_BY_DV["calibration_gap"] != MARGIN_BY_DV["attribution_auc"], MARGIN_BY_DV)
    chk("every DV has a threshold", all(dv in MARGIN_BY_DV for dv in DVS), DVS)
    chk("MARGIN is the registry leg's own pre-registered effect", MARGIN == 0.15, MARGIN)
    # The diagnostic arm must never be able to reach a criterion.
    chk("criterion arms are exactly control+treatment",
        CRITERION_ARMS == ("SEPARATED", "MERGED"), CRITERION_ARMS)
    chk("diagnostic arms are disjoint from criterion arms",
        not (set(DIAGNOSTIC_ARMS) & set(CRITERION_ARMS)), (DIAGNOSTIC_ARMS, CRITERION_ARMS))
    chk("every declared arm is constructible", all(a in _ARM_CLASSES for a in ARMS), ARMS)
    # The AUC and the CI helpers.
    chk("auc perfect/chance/inverted", (_auc([1, 2, 3], [0, 0.5]), _auc([1, 2], [1, 2]),
                                        _auc([0.0], [1.0])) == (1.0, 0.5, 0.0))
    ci = _paired_ci([-0.0147, -0.1784, 0.0736, 0.0941])
    chk("paired CI reproduces 993a DENSE (mean -0.0064, sd 0.124)",
        abs(ci["mean"] + 0.00635) < 1e-3 and abs(ci["sd"] - 0.1240) < 1e-3,
        {k: round(v, 4) for k, v in ci.items() if isinstance(v, float)})
    chk("t_crit is conservative for untabulated df (uses the lower bracketing entry)",
        _t_crit(95) >= _t_crit(100), (_t_crit(95), _t_crit(100)))
    chk("n_seeds sized for the pre-registered effect at the pessimistic SD",
        _t_crit(len(SEEDS) - 1) * SD_PESSIMISTIC_GAP / (len(SEEDS) ** 0.5) < MARGIN,
        {"half_width": _t_crit(len(SEEDS) - 1) * SD_PESSIMISTIC_GAP / (len(SEEDS) ** 0.5),
         "margin": MARGIN})
    # DETACH is a design decision, not an accident -- assert it is on.
    chk("MSE targets are detached (undetached collapses the latent in every arm)",
        DETACH_MSE_TARGETS is True)
    print(f"[self-test] {fails} failure(s)", flush=True)
    return fails


def _paired_ci(diffs: Sequence[float], confidence: float = CI_CONFIDENCE
               ) -> Dict[str, Any]:
    """Mean of the paired differences and its two-sided CI.

    Student-t, computed from a small table rather than importing SciPy (not a dependency of
    this repo). The paired design is what makes this the right interval: both arms share the
    env seed, the encoder init and the whole P0 warmup, so a per-seed difference removes the
    between-seed variance that dominates the raw arm means.
    """
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": (float(diffs[0]) if n == 1 else None),
                "sd": None, "se": None, "half_width": None,
                "ci_low": None, "ci_high": None, "confidence": confidence}
    mean = float(sum(diffs) / n)
    var = float(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    sd = var ** 0.5
    se = sd / (n ** 0.5)
    t = _t_crit(n - 1)
    hw = t * se
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t_crit": t, "half_width": hw,
            "ci_low": mean - hw, "ci_high": mean + hw, "confidence": confidence}


# Two-sided 95% Student-t critical values by degrees of freedom. Exact for the tabulated df,
# and the LARGEST bracketing entry is used for an untabulated df, so the interval is never
# narrower than the true one (conservative in the direction that matters: a too-narrow CI would
# manufacture an exclusion of -MARGIN that the data does not support).
_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
            8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 24: 2.064,
            30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in _T_TABLE:
        return _T_TABLE[df]
    lower = [k for k in sorted(_T_TABLE) if k < df]
    if not lower:
        return _T_TABLE[min(_T_TABLE)]
    if df > max(_T_TABLE):
        return 1.960
    return _T_TABLE[max(lower)]      # conservative: a wider interval than the true one


def _eval_probes(env: SmallViewEnv, encoder: nn.Module, channels, action_dim: int) -> Dict[str, Any]:
    encoder.eval()
    near_sigs: List[float] = []
    safe_sigs: List[float] = []
    fatal_errors = 0

    wall_type = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _run_probe(ax: int, ay: int, actual_idx: int) -> float:
        """causal_sig = harm_logit(z_t, a_actual) - harm_logit(z_t, a_cf).

        Two deliberate departures from 993, both measured (module docstring):
        (1) PRE-SIGMOID. The sigmoid compressed the DV by 78x; at a base rate of
            ~2-4% its derivative p(1-p) is ~0.02-0.04, so the whole contrast
            collapsed into ~0.2% of [-1,+1].
        (2) The forward head is NOT on this path. 993 evaluated the harm head on
            predict_forward(z, a), whose MSE-regressed output lands about as far
            from the true next latent as two random latents are apart
            (reach_ratio 1.15-1.49) -- so the counterfactual contrast was
            annihilated before the harm head ever saw it.
        The counterfactual structure of SD-003 / V3-EXQ-007-010 is preserved
        exactly: same state, actual action versus a distinct counterfactual one.
        """
        # RED-TEAM F7: reproduce the grid state step() would have produced, not
        # just the agent POINTER. causal_grid_world.step() vacates the old cell
        # (grid[old] = empty, or contaminated above threshold -- line 2453/2421)
        # and writes grid[new] = agent (line 2745), so EVERY training
        # observation has the agent token at the 3x3 centre. 993 moved only
        # agent_x/agent_y, leaving the centre reading empty/resource and the
        # stale agent token sitting elsewhere inside the 3x3 view -- an
        # out-of-distribution input at exactly the position the encoder is most
        # sensitive to. That is the same defect this driver cites to reject the
        # env-lookahead readout, so it cannot be tolerated here either. The
        # mutation is fully restored in the finally block, so probe order stays
        # irrelevant and the env is unchanged on exit.
        prev_x, prev_y = env.agent_x, env.agent_y
        saved_center = int(env.grid[ax, ay])
        saved_prev = int(env.grid[prev_x, prev_y])
        env.grid[prev_x, prev_y] = (
            env.ENTITY_TYPES["contaminated"]
            if float(env.contamination_grid[prev_x, prev_y]) >= env.contamination_threshold
            else env.ENTITY_TYPES["empty"]
        )
        env.grid[ax, ay] = env.ENTITY_TYPES["agent"]
        env.agent_x = ax
        env.agent_y = ay
        try:
            obs_dict = env._get_observation_dict()
            with torch.no_grad():
                z = encoder(obs_dict["world_state"].unsqueeze(0))
                cf_idx = _random_cf_action(actual_idx, action_dim)
                a_act = _action_onehot(actual_idx, action_dim)
                a_cf = _action_onehot(cf_idx, action_dim)
                h_act = channels.harm_logit(z, a_act)
                h_cf = channels.harm_logit(z, a_cf)
                return float((h_act - h_cf).item())
        finally:
            env.grid[ax, ay] = saved_center
            env.grid[prev_x, prev_y] = saved_prev
            env.agent_x, env.agent_y = prev_x, prev_y

    try:
        for _ in range(PROBE_RESETS):
            env.reset()
            for hx, hy in env.hazards:
                for action_idx, (dx, dy) in env.ACTIONS.items():
                    if action_idx == STAY_ACTION_IDX:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < env.size and 0 <= ay < env.size:
                        cell = int(env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            near_sigs.append(_run_probe(ax, ay, action_idx))

            for px in range(env.size):
                for py in range(env.size):
                    if int(env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(abs(px - hx) + abs(py - hy) for hx, hy in env.hazards)
                    if min_dist > 3:
                        safe_sigs.append(_run_probe(px, py, random.randint(0, action_dim - 2)))
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL: {traceback.format_exc()}", flush=True)

    # Red-team fix F5: an empty probe list must NOT silently substitute 0.0
    # (a different statistic, "no safe probes" != "zero causal signature").
    # Coverage insufficiency propagates as None and excludes the cell from
    # aggregation, rather than corrupting the mean with a fabricated value.
    coverage_ok = (
        fatal_errors == 0
        and len(near_sigs) >= MIN_PROBE_COVERAGE
        and len(safe_sigs) >= MIN_PROBE_COVERAGE
    )
    mean_near = float(sum(near_sigs) / len(near_sigs)) if near_sigs else None
    mean_safe = float(sum(safe_sigs) / len(safe_sigs)) if safe_sigs else None
    calibration_gap = (
        (mean_near - mean_safe) if (coverage_ok and mean_near is not None and mean_safe is not None) else None
    )

    # ---- THE SECOND ARC-021 FALSIFIER DV -------------------------------------------------
    # ARC-021's falsifying signature names TWO quantities: "calibration_gap AND attribution
    # accuracy". V3-EXQ-993a measured only the first, so it could not have falsified the claim
    # on its own terms. `attribution_auc` is the accuracy counterpart, computed from the SAME
    # probe signatures already collected -- it costs no extra compute at all.
    #
    # It is the probability that a randomly chosen NEAR-HAZARD probe carries a higher
    # action-conditioned causal signature than a randomly chosen SAFE probe (the Mann-Whitney
    # statistic, ties counted as half). Three properties earn it its place:
    #   (a) it is THRESHOLD-FREE -- no cut point is fitted on the data it then scores, which is
    #       the "gate that certifies its own subject" trap;
    #   (b) it is HONESTLY BOUNDED [0, 1] with chance at exactly 0.5, unlike calibration_gap,
    #       which is a difference of logits and is formally unbounded below (this is the
    #       V3-EXQ-993a red-team F4 defect, and the reason THIS driver must not repeat that
    #       run's dv_bounds=(0.0, 1.0) declaration for the gap);
    #   (c) it CAN COME APART FROM THE GAP, which is precisely why the falsifier names both: a
    #       uniform scale change shrinks the gap while leaving the ranking untouched (AUC flat,
    #       gap down), and a variance increase degrades the ranking while leaving the means
    #       apart (AUC down, gap flat). A leg reading only the gap cannot tell those apart.
    attribution_auc = _auc(near_sigs, safe_sigs) if coverage_ok else None

    return {
        "calibration_gap": calibration_gap,
        "attribution_auc": attribution_auc,
        "mean_causal_sig_near_hazard": mean_near,
        "mean_causal_sig_safe": mean_safe,
        "n_near_hazard_probes": len(near_sigs),
        "n_safe_probes": len(safe_sigs),
        "fatal_errors": fatal_errors,
        "coverage_ok": coverage_ok,
    }


def _run_cell(condition: str, arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {condition}_{arm}", flush=True)
    config_slice = {
        "condition": condition,
        "arm": arm,
        "num_hazards": NUM_HAZARDS[condition],
        "grid_size": GRID_SIZE,
        "obs_view_size": OBS_VIEW_SIZE,
        "world_dim": WORLD_DIM,
        "hidden_dim": HIDDEN_DIM,
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "encoder_frozen_in_p1": False,
        "detach_mse_targets": DETACH_MSE_TARGETS,
        "grad_clip_norm": GRAD_CLIP_NORM,
    }
    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        env = SmallViewEnv(size=GRID_SIZE, num_hazards=NUM_HAZARDS[condition], seed=seed)
        action_dim = env.action_dim
        encoder, decoder = _make_world_codec(env.world_obs_dim)
        codec_opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR_ENCODER)

        # P0 is UNCHANGED and identical across arms: reconstruction warmup of one shared
        # codec. `arm_cell` has already reset every RNG at cell entry, and the channels
        # object is constructed AFTER P0, so the P0-warmed encoder is bit-identical across
        # arms at a given (condition, seed). Each arm then takes deepcopies of it, which
        # consume no RNG.
        p0_stats = _run_p0(env, encoder, decoder, codec_opt)
        channels = _make_channels(arm, action_dim, encoder)
        p1_stats = _run_p1(env, channels, action_dim)

        # PROBE THROUGH THE ARM'S OWN HARM-PATH ENCODER. In ARM_SEPARATED that is
        # `enc_harm`, trained by the harm BCE alone; in the other two arms it is the single
        # shared encoder. Reading it off the channels object is what stops an arm being
        # probed through an encoder no criterion ever trained.
        harm_sensitivity = _harm_action_sensitivity(
            channels.harm_encoder, channels, env, action_dim)
        probe_stats = _eval_probes(env, channels.harm_encoder, channels, action_dim)

        _gap = probe_stats["calibration_gap"]
        _gap_str = f"{_gap:.4f}" if _gap is not None else "None(coverage_insufficient)"
        _auc_v = probe_stats["attribution_auc"]
        _auc_str = f"{_auc_v:.4f}" if _auc_v is not None else "None"
        print(
            f"  [{condition}_{arm} seed={seed}] gap={_gap_str} auc={_auc_str} "
            f"n_near={probe_stats['n_near_hazard_probes']} n_safe={probe_stats['n_safe_probes']} "
            f"harm_action_sensitivity={harm_sensitivity:.5f} harm_events={p1_stats['p1_harm_events']} "
            f"zharm_disp={p1_stats['zharm_dispersion']:.5f} "
            f"clip_frac={p1_stats['clip_active_frac']}",
            flush=True,
        )

        row = {
            "condition": condition,
            "arm": arm,
            "seed": seed,
            "mean_recon_loss": p0_stats["mean_recon_loss"],
            "harm_action_sensitivity": harm_sensitivity,
            **p1_stats,
            **probe_stats,
        }
        cell.stamp(row)

    passed = bool(row["coverage_ok"])
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _build_preconditions(rows: List[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
    """Build all four readiness checks over `rows`.

    Called TWICE: once over the SEPARATED (control) cells only, before any
    MERGED cell is trained, and once over the full grid for the manifest. The
    early call is the whole point of the dv_headroom class -- V3-EXQ-993 burned
    its entire 12-cell grid before discovering its control arm produced no
    signal, and a headroom check can only be computed once control values
    exist, so control-first ordering is what makes the gate cheap."""
    # RED-TEAM F2: the gate ranges over the CONTROL arm ONLY. A MERGED harm head
    # gone action-blind is the EXTREME FORM of the degradation ARC-021 predicts;
    # gating on it would convert the strongest possible positive result into
    # "substrate not ready". The docstring's justification for this precondition
    # ("structurally zero in BOTH arms") is true of the control arm, which is
    # what certifies that the instrument can read anything at all. MERGED's own
    # sensitivity is still RECORDED, as a diagnostic, never as a gate.
    control_rows = [r for r in rows if r["arm"] == CONTROL_ARM]
    worst_harm_sensitivity = min(r["harm_action_sensitivity"] for r in control_rows)
    worst_cell = min(control_rows, key=lambda r: r["harm_action_sensitivity"])
    merged_rows = [r for r in rows if r["arm"] == TREATMENT_ARM]
    worst_merged_sensitivity = (
        min(r["harm_action_sensitivity"] for r in merged_rows) if merged_rows else None
    )
    # EVERY harm-event and coverage count below is denominated on the CRITERION ARMS only.
    # The diagnostic arm contributes cells to the manifest but must never be able to move a
    # gate -- otherwise a non-criterion arm could refuse, or rescue, a run whose verdict it
    # has no part in.
    crit_rows = [r for r in rows if r["arm"] in CRITERION_ARMS]
    dense_harm_events = sum(r["p1_harm_events"] for r in crit_rows if r["condition"] == "DENSE")
    sparse_harm_events = sum(r["p1_harm_events"] for r in crit_rows if r["condition"] == "SPARSE")
    min_condition_harm_events = min(dense_harm_events, sparse_harm_events)

    # Control-arm realised DV values -- the input to BOTH dv_headroom checks.
    # Only SEPARATED cells with usable probe coverage contribute; a None gap is
    # a coverage exclusion (993 fix F5), not a zero.
    control_gaps = [
        r["calibration_gap"] for r in rows
        if r["arm"] == CONTROL_ARM and r["calibration_gap"] is not None
    ]
    control_aucs = [
        r["attribution_auc"] for r in rows
        if r["arm"] == CONTROL_ARM and r.get("attribution_auc") is not None
    ]
    # RED-TEAM F6: the headroom checks would otherwise certify over the REALIZED
    # control cells rather than the INTENDED ones -- a 7-of-8 control arm could
    # pass H1, train the whole MERGED arm, and only then fail non-degeneracy on
    # the short condition (which requires all seeds paired). Denominating the
    # gate on the intended n is the difference between refusing early and
    # discovering the shortfall after the compute is spent.
    n_intended_control = len(SEEDS) * len(CONDITIONS)   # CONTROL ARM only
    precondition_0 = {
        "name": "control_arm_coverage_complete",
        "description": (
            "Every intended SEPARATED (control) cell produced a usable "
            "calibration_gap. The dv_headroom checks below are denominated on "
            "this same set, so a short control arm must refuse here rather than "
            "silently certify headroom over fewer cells than the criteria need."
        ),
        "kind": "readiness",
        "measured": float(len(control_gaps)),
        "threshold": float(n_intended_control),
        "direction": "lower",
        "control": f"intended = len(SEEDS) x len(CONDITIONS) = {n_intended_control}",
        "met": bool(len(control_gaps) >= n_intended_control),
    }

    precondition_1 = {
        "name": "harm_head_action_sensitivity_present",
        "description": (
            "Worst-cell mean |harm_logit(z,a1)-harm_logit(z,a2)| over a fixed probe "
            "batch and two distinct movement actions, measured after P1 training. "
            "This is the SAME statistic the load-bearing criteria route on (a "
            "difference of two harm logits at one state under two actions), not a "
            "magnitude proxy on a different component -- 993 asserted the FORWARD "
            "head's sensitivity at 17.9x over floor while the readout the criteria "
            "actually read was flat. An action-blind harm head makes causal_sig "
            "structurally zero in BOTH arms, so below-floor means substrate-not-ready."
        ),
        "kind": "readiness",
        "measured": worst_harm_sensitivity,
        "threshold": HARM_ACTION_SENSITIVITY_FLOOR,
        "direction": "lower",
        "control": (
            f"16-sample probe batch, actions 0 vs 1, worst cell across the "
            f"{len(control_rows)} SEPARATED (control) cells of the {stage} set"
        ),
        "worst_merged_harm_action_sensitivity_diagnostic": worst_merged_sensitivity,
        "offending_cell": f"{worst_cell['condition']}_{worst_cell['arm']}_seed{worst_cell['seed']}",
        "met": bool(worst_harm_sensitivity >= HARM_ACTION_SENSITIVITY_FLOOR),
    }
    precondition_2 = {
        "name": "p1_harm_events_observed_per_condition",
        "description": (
            "MIN(total P1 harm events summed over arms+seeds) across the two hazard-"
            "density conditions -- 993 red-team fix F7: a pooled floor let SPARSE's "
            "harm head train on near-zero labeled data while DENSE's volume masked "
            "it. Taking the min across conditions requires BOTH to have real coverage."
        ),
        "kind": "readiness",
        "measured": float(min_condition_harm_events),
        "threshold": float(HARM_EVENTS_FLOOR_PER_CONDITION),
        "direction": "lower",
        "control": f"dense={dense_harm_events}, sparse={sparse_harm_events} ({stage} cells)",
        "met": bool(min_condition_harm_events >= HARM_EVENTS_FLOOR_PER_CONDITION),
    }

    # H1/H2/H3 -- the dv_headroom class minted by the 2026-09-03 cluster autopsy
    # (ree-v3 8e133d26ed). `criterion_threshold` is passed as the module constant the
    # criterion itself reads, never a re-typed literal, so the gate cannot drift away from
    # the science it guards.
    if control_gaps:
        h1 = dv_headroom_check(
            "dv_headroom_margin_room_below_control",
            dv_name="calibration_gap",
            criterion_threshold=MARGIN,
            control_values=control_gaps,
            # `range`, NOT `floor_headroom` -- carried from V3-EXQ-1011, and the reason is
            # the 993a red-team F4 defect this leg must not repeat. `floor_headroom` is
            # min(control) - low and therefore STRUCTURALLY REQUIRES a dv_bounds low, which
            # for a difference of logits does not exist, so 993a invented 0.0 and its own
            # autopsy (section 13 point 4) then withdrew the resulting gate as a licence for
            # a direction: "HEADROOM IS NOT POWER". `range` needs no bound at all and answers
            # the question this criterion actually consumes -- does the DV move, over the
            # control arm, by enough that a MARGIN-sized effect is resolvable within it.
            statistic="range",
            # dv_bounds DELIBERATELY OMITTED for calibration_gap: it is a difference of
            # logits, formally unbounded below, and this leg's criterion is a CONFIDENCE
            # INTERVAL that explicitly looks below zero. Declaring a 0.0 floor would assert
            # the interval cannot go where the criterion looks. (`attribution_auc` IS
            # genuinely bounded [0, 1] and declares it -- h3 below.)
            margin=2.0,
            kind_note=(
                "Control-arm RANGE of calibration_gap, required to be at least 2x MARGIN "
                "(0.15), the effect this leg is posed against and the registry leg's own "
                "pre-registered figure. NO dv_bounds is declared: calibration_gap is a "
                "difference of logits and is unbounded below, and declaring [0,1] here was "
                "V3-EXQ-993a's red-team F4 defect, withdrawn by its autopsy section 13 "
                "point 4 on the ground that headroom is not power. Power is stated "
                "separately, in metrics.power_note and the module docstring, from the "
                "realised paired SD."
            ),
        )
        h2 = dv_headroom_check(
            "dv_headroom_control_signal_floor_reachable",
            dv_name="calibration_gap",
            criterion_threshold=SEPARATED_SIGNAL_FLOOR,
            control_values=control_gaps,
            statistic="max_abs",
            margin=2.0,
            kind_note=(
                "The 2026-09-03 cluster autopsy's own table pairs 993's max "
                "|calibration_gap| (0.00152) against SEPARATED_SIGNAL_FLOOR (0.02) for a "
                "13.1x shortfall, so max_abs is the sanctioned statistic for this "
                "non-degeneracy floor. Frozen-encoder reference: 1011 measured 1.0152 "
                "against a 0.40 requirement (2.54x)."
            ),
        )
        checks = [h1, h2]
    else:
        # No usable control value at all -- report as UNMET rather than skipping, so an
        # empty control arm can never certify itself by absence.
        checks = [{
            "name": "dv_headroom_margin_room_below_control",
            "kind": "dv_headroom",
            "dv_name": "calibration_gap",
            "achievable_statistic": "range",
            "measured": float("nan"),
            "threshold": MARGIN * 2.0,
            "direction": "lower",
            "criterion_threshold": MARGIN,
            "headroom_margin": 2.0,
            "n_control_values": 0,
            "met": False,
            "control": "no SEPARATED cell produced a usable calibration_gap",
        }]

    # ---- THE SECOND DV GETS ITS OWN HEADROOM CHECK -----------------------------------
    # V3-EXQ-1011 reads `attribution_auc` as a load-bearing DV and its own h1 comment
    # promises "`attribution_auc`, by contrast, IS genuinely bounded [0, 1] -- see its own
    # check below" -- but no such check was ever written, so the AUC criterion ran ungated
    # while the gap's was gated twice. That gap is closed here. The AUC's bounds are REAL, so
    # unlike the gap this check CAN honestly declare them, and `floor_headroom` is the right
    # statistic: chance is exactly 0.5 and the criterion consumes the room the treatment arm
    # has to FALL from the control arm toward chance.
    if control_aucs:
        checks.append(dv_headroom_check(
            "dv_headroom_auc_room_below_control",
            dv_name="attribution_auc",
            criterion_threshold=MARGIN_AUC,
            control_values=control_aucs,
            statistic="floor_headroom",
            dv_bounds=(0.5, 1.0),
            margin=1.0,
            kind_note=(
                "floor_headroom = min(control attribution_auc) - 0.5 is exactly the room the "
                "treatment arm has to fall toward CHANCE, which is what a suppression-"
                "direction criterion on a ranking statistic consumes. 0.5 is the TRUE floor "
                "here -- an AUC is a probability and chance is exactly one half -- so unlike "
                "993a's calibration_gap declaration this bound is a fact about the DV, not "
                "an invention. A control arm whose own AUC sits within MARGIN_AUC of chance "
                "cannot express the pre-registered degradation at all, and that is precisely "
                "what this refuses."
            ),
        ))
    else:
        checks.append({
            "name": "dv_headroom_auc_room_below_control",
            "kind": "dv_headroom",
            "dv_name": "attribution_auc",
            "achievable_statistic": "floor_headroom",
            "measured": float("nan"),
            "threshold": MARGIN_AUC,
            "direction": "lower",
            "criterion_threshold": MARGIN_AUC,
            "headroom_margin": 1.0,
            "n_control_values": 0,
            "met": False,
            "control": "no SEPARATED cell produced a usable attribution_auc",
        })
    headroom = checks
    return [precondition_0, precondition_1, precondition_2] + headroom


def _precondition_measured(preconditions: List[Dict[str, Any]], name: str) -> float:
    """Look a precondition's `measured` up BY NAME. Never index this list.

    RECORDING DEFECT, fixed forward 2026-09-07 (chip-20260905-exq993a-
    recording-defect-precondition-index; found by the fable red-team of
    failure_autopsy_V3-EXQ-993a_2026-09-05, hygiene H1). Both metric sites
    below previously read `preconditions[0]["measured"]`, written when the
    sensitivity check WAS first in the list. Red-team fix F6 then inserted
    `control_arm_coverage_complete` at index 0, and nothing re-pointed the
    reads -- so the landed run recorded metrics.worst_harm_action_sensitivity
    = 8.0, which is the coverage COUNT, while the true sensitivity (0.11382)
    sat one slot along. No error, no test: a positional read is silently
    correct until someone reorders the list, and reordering a precondition
    list is a routine red-team repair.

    The landed manifest is NOT edited -- the defect is on record in the
    autopsy's `recording_defects`. This fix is forward-only, so any successor
    letter records the right value.

    Raises rather than defaulting: a missing precondition means the list shape
    changed, and silently substituting a default is how the original defect
    survived a run.
    """
    for p in preconditions:
        if p.get("name") == name:
            return float(p["measured"])
    raise KeyError(
        "no precondition named %r (have: %s)"
        % (name, ", ".join(repr(p.get("name")) for p in preconditions))
    )


def _mean_or_none(rows: List[Dict[str, Any]], condition: str, arm: str) -> Optional[float]:
    vals = [
        r["calibration_gap"] for r in rows
        if r["condition"] == condition and r["arm"] == arm and r["calibration_gap"] is not None
    ]
    return statistics.fmean(vals) if vals else None


def _arm_mean(rows: List[Dict[str, Any]], arm: str, key: str) -> Optional[float]:
    """Mean of a per-cell diagnostic over one arm, skipping cells that did not record it."""
    vals = [r[key] for r in rows if r["arm"] == arm and r.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (stdlib; SciPy is not a dependency of this repo)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _power_note(dense_v: Dict[str, Any], sparse_v: Dict[str, Any]) -> Dict[str, Any]:
    """Power of each verdict branch, computed from THIS RUN'S REALISED paired SD.

    failure_autopsy_V3-EXQ-993a_2026-09-05 section 15 learning 9: "always state the power at
    the pre-registered effect alongside the interval -- that is what stops 'not detected'
    being read as 'excluded'." 993a's own joint power at its own pre-registered effect was
    about 12%, which is the single fact that made its null unreadable, and it was computed
    only retrospectively, by its autopsy.

    Three numbers per (DV, condition), all normal-approximation with the realised SE:
      p_eliminate_at_null       P(read ELIMINATED | no degradation)      -- the branch's power
      p_eliminate_at_margin     P(read ELIMINATED | true effect = -MARGIN) -- false-elimination
      p_confirm_at_double       P(read CONFIRMED  | true effect = -2*MARGIN)
    A run whose p_eliminate_at_null is low did not have the power to return its own null.
    """
    out: Dict[str, Any] = {
        "note": (
            "Normal-approximation power of each verdict branch at the realised paired SD. "
            "ELIMINATED requires ci_low > -MARGIN; CONFIRMED requires ci_high < -MARGIN."
        )
    }
    for cond_name, v in (("DENSE", dense_v), ("SPARSE", sparse_v)):
        for dv in DVS:
            try:
                ci = v["dv_blocks"][dv]["paired_ci"]
                se, hw = ci.get("se"), ci.get("half_width")
                m = MARGIN_BY_DV[dv]
            except (KeyError, TypeError):
                continue
            if not se or not hw or se <= 0:
                continue
            out[f"{cond_name.lower()}_{dv}"] = {
                "n": ci.get("n"), "sd": ci.get("sd"), "se": se, "half_width": hw,
                "margin": m,
                "p_eliminate_at_null": round(1.0 - _norm_cdf((hw - m) / se), 4),
                "p_eliminate_at_margin": round(1.0 - _norm_cdf(hw / se), 4),
                "p_confirm_at_double": round(_norm_cdf((m - hw) / se), 4),
            }
    return out



def run_experiment() -> Dict[str, Any]:
    # STAGE A -- every SEPARATED (control) cell, both conditions, first.
    rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            rows.append(_run_cell(condition, CONTROL_ARM, seed))

    control_preconditions = _build_preconditions(rows, "control-arm")
    gate_blocked = False
    try:
        p0_readiness_gate(control_preconditions)
    except P0NotReady:
        gate_blocked = True

    if gate_blocked and _SMOKE_FORCE_GATE:
        print(
            "  [smoke] readiness gate UNMET but --smoke-force-gate is set; "
            "continuing to the MERGED arm as a positive control. This path is "
            "dry-run-only and can never be reached in a real run.",
            flush=True,
        )
        gate_blocked = False

    if gate_blocked:
        # Self-route to substrate_not_ready_requeue WITHOUT training a single
        # MERGED cell. This is never a verdict on ARC-021 or MECH-069.
        # NOTE: no extra `verdict:` prints here -- _run_cell already emitted one
        # per control cell, and the runner counts verdict lines to advance its
        # progress bar. The MERGED cells simply never run, so this run emits
        # len(SEEDS) x len(CONDITIONS) verdicts instead of twice that.
        return {
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "evidence_direction_per_claim": {c: "non_contributory" for c in CLAIM_IDS},
            "evidence_direction_note": (
                "NO COMPARISON WAS MADE. Set explicitly so build_experiment_indexes.py "
                "does NOT rewrite this run's direction: the indexer computes "
                "direction_explicitly_set = bool(manifest['evidence_direction_note']) "
                "(line 1871) and then rewrites any non-explicit 'unknown' on a FAIL "
                "to 'weakens' (lines 3519-3520). V3-EXQ-993 emitted a bare 'unknown' "
                "and its 2026-09-03 autopsy had to hand-correct the record to "
                "'non_contributory'. non_degenerate/scoring_excluded does NOT rescue "
                "this: the gap register's conflict_ratio counts entries regardless of "
                "scoring_excluded. So the direction is stated as non_contributory AND "
                "this note is present, which are the two independent things that keep "
                "an instrument-not-ready run from being logged as evidence against "
                "ARC-021 or MECH-069. ""Trigger: the control-arm readiness gate was unmet "
                "before any MERGED cell was trained."
            ),
            "non_degenerate": False,
            "degeneracy_reason": (
                "control-arm readiness gate unmet before the MERGED arm was trained "
                "(see interpretation.preconditions); no comparison was made"
            ),
            "readout": {
                "readiness_met": 0.0,
                "margin_calibration_gap": float(MARGIN),
                "margin_attribution_auc": float(MARGIN_AUC),
                "separated_signal_floor": float(SEPARATED_SIGNAL_FLOOR),
                "n_seed_pairs_dense": 0.0,
                "n_seed_pairs_sparse": 0.0,
                "encoder_frozen_in_p1": 0.0,
                "detach_mse_targets": float(int(DETACH_MSE_TARGETS)),
                "n_control_cells": float(len(rows)),
            },
            "metrics": {
                "mean_calibration_gap_dense_separated": _mean_or_none(rows, "DENSE", CONTROL_ARM),
                "mean_calibration_gap_dense_merged": None,
                "mean_calibration_gap_sparse_separated": _mean_or_none(rows, "SPARSE", CONTROL_ARM),
                "mean_calibration_gap_sparse_merged": None,
                "dense_n_seed_pairs": 0,
                "sparse_n_seed_pairs": 0,
                "worst_harm_action_sensitivity": _precondition_measured(
                    control_preconditions, "harm_head_action_sensitivity_present"),
                "dense_harm_events": sum(r["p1_harm_events"] for r in rows if r["condition"] == "DENSE"),
                "sparse_harm_events": sum(r["p1_harm_events"] for r in rows if r["condition"] == "SPARSE"),
                "n_finite_violations_total": sum(r["fatal_errors"] for r in rows),
                "margin": MARGIN,
                "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
                "control_only_stage": True,
                "n_control_cells": len(rows),
            },
            "per_seed_rows": rows,
            "arm_results": rows,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": control_preconditions,
                "combination_rule": (
                    "Control-arm-first staging: the readiness gate (incl. both "
                    "dv_headroom checks) is evaluated over the SEPARATED cells before "
                    "any MERGED cell is trained. An unmet gate stops the run here."
                ),
                "criteria": [
                    {"name": "C1_dense_margin_degradation_eliminated", "load_bearing": True,
                     "passed": False, "measured": None, "threshold": -MARGIN,
                     "threshold_not_applicable": "no MERGED cell was trained; nothing was measured"},
                    {"name": "C2_sparse_margin_degradation_eliminated", "load_bearing": True,
                     "passed": False, "measured": None, "threshold": -MARGIN,
                     "threshold_not_applicable": "no MERGED cell was trained; nothing was measured"},
                ],
                "criteria_non_degenerate": {
                    "C1_dense_margin_degradation_eliminated": False,
                    "C2_sparse_margin_degradation_eliminated": False,
                },
            },
        }

    # STAGE B -- the MERGED arm, only once the control arm has demonstrated the
    # DV has room for the criterion to fire.
    for condition in CONDITIONS:
        for seed in SEEDS:
            rows.append(_run_cell(condition, TREATMENT_ARM, seed))
    # ...then the DIAGNOSTIC arm(s), last, so that a crash there cannot cost the criterion
    # arms and so the gate has never depended on them.
    for arm in DIAGNOSTIC_ARMS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                rows.append(_run_cell(condition, arm, seed))

    n_finite_violations_total = sum(r["fatal_errors"] for r in rows)
    preconditions = _build_preconditions(rows, "full-grid")
    # Evaluate the full-grid gate through the SAME tested helper stage A uses,
    # rather than re-implementing its met/direction/NaN semantics here. A
    # hand-rolled duplicate is exactly how a gate drifts away from the thing it
    # is supposed to enforce.
    try:
        p0_readiness_gate(preconditions)
        readiness_met = True
    except P0NotReady:
        readiness_met = False
    # RED-TEAM F2: `--smoke-force-gate` forced only the STAGE A gate, so the Stage B gate below
    # always failed at dry-run scale and the smoke never reached the new CI code at all --
    # `_paired_ci`, `_t_crit` and the H3 reading rule had never executed under the driver's own
    # smoke. That is the V3-EXQ-591g short-circuit-hides-a-crash shape, on the exact code this
    # letter exists to add. The override now covers BOTH stages, and remains dry-run-only
    # (refused outside --dry-run in main(), and the runner never passes it).
    if not readiness_met and _SMOKE_FORCE_GATE:
        print("  [smoke] STAGE B readiness gate UNMET but --smoke-force-gate is set; "
              "forcing green so the paired-CI verdict path is exercised as a POSITIVE control",
              flush=True)
        readiness_met = True
    worst_harm_sensitivity = _precondition_measured(
        preconditions, "harm_head_action_sensitivity_present")
    dense_harm_events = sum(r["p1_harm_events"] for r in rows if r["condition"] == "DENSE")
    sparse_harm_events = sum(r["p1_harm_events"] for r in rows if r["condition"] == "SPARSE")

    def _condition_verdict(condition: str) -> Dict[str, Any]:
        """993 red-team fixes F3+F4, carried forward: a PAIRED per-seed comparison (same seed drives
        bit-identical P0 in both arms, so pairing removes cross-seed P0
        variance) with a sign-consistency requirement (all seeds must agree
        in direction, not just the pooled mean), and per-CONDITION
        non-degeneracy gating (a degenerate SPARSE baseline can no longer
        silently count toward the verdict via C2)."""
        def _by_seed(arm: str, dv: str) -> Dict[int, float]:
            return {r["seed"]: r[dv] for r in rows
                    if r["condition"] == condition and r["arm"] == arm
                    and r.get(dv) is not None}

        # ---- BOTH ARC-021 falsifier DVs, not just the gap -----------------------------
        # The claim's falsifying signature names "calibration_gap AND attribution accuracy".
        # V3-EXQ-993a measured only the first. Each DV gets its own paired CI and its own H3
        # reading; the leg's verdict is the CONJUNCTION over DVs and conditions, stated below.
        dv_blocks: Dict[str, Any] = {}
        for dv in DVS:
            sep_by_seed = _by_seed(CONTROL_ARM, dv)
            mer_by_seed = _by_seed(TREATMENT_ARM, dv)
            paired = sorted(set(sep_by_seed) & set(mer_by_seed))
            d = [mer_by_seed[k] - sep_by_seed[k] for k in paired]
            ci = _paired_ci(d)
            lo, hi = ci["ci_low"], ci["ci_high"]
            thr = -MARGIN_BY_DV[dv]       # per-DV, on each DV's own scale (1011 red-team F4)
            # THE H1 READING, on this DV's own scale. ELIMINATED: the whole interval sits
            # ABOVE -MARGIN, so a degradation of the pre-registered size is ruled out.
            # CONFIRMED: the whole interval sits BELOW it, so the pre-registered degradation
            # is established. UNDETERMINED: the interval straddles -MARGIN -- the run is
            # under-powered for the effect it was sized against, and says so instead of
            # reporting a null. The three are mutually exclusive and jointly exhaustive.
            if lo is None or hi is None:
                reading = "undetermined"
            elif lo > thr:
                reading = "eliminated"
            elif hi < thr:
                reading = "confirmed"
            else:
                reading = "undetermined"
            dv_blocks[dv] = {
                "dv": dv,
                "mean_separated": (statistics.fmean(sep_by_seed.values())
                                   if sep_by_seed else None),
                "mean_merged": (statistics.fmean(mer_by_seed.values())
                                if mer_by_seed else None),
                "n_seed_pairs": len(paired),
                "per_seed_diffs": d,
                "paired_ci": ci,
                "margin_threshold": thr,
                "h1_reading": reading,
                # Reported for continuity with V3-EXQ-993a's threshold+sign-consistency
                # reading; NOT a conjunct here. Sign-consistency across n>=32 seeds is a
                # near-impossible requirement that would convert every genuine effect back
                # into a null (V3-EXQ-1011's finding), which is why the verdict is an
                # interval instead.
                "legacy_993a_criterion_passed": bool(
                    d and statistics.fmean(d) <= -MARGIN and all(x <= 0 for x in d)),
            }

        gap = dv_blocks["calibration_gap"]
        mean_sep = gap["mean_separated"]
        mean_merged = gap["mean_merged"]
        paired_seeds = list(range(gap["n_seed_pairs"]))
        # Non-degenerate requires EVERY seed to have a valid paired cell (no silent power
        # loss from an excluded cell) AND the separated baseline to clear the (decoupled)
        # signal floor. Unchanged from V3-EXQ-993a.
        non_degenerate = bool(
            gap["n_seed_pairs"] == len(SEEDS) and mean_sep is not None
            and mean_sep >= SEPARATED_SIGNAL_FLOOR
        )
        # The condition "passes" for this leg when EVERY DV ELIMINATES the pre-registered
        # degradation there. Note the direction: a "passed" criterion is evidence AGAINST
        # ARC-021's necessity half, not for it.
        criterion_passed = bool(
            non_degenerate
            and all(dv_blocks[dv]["h1_reading"] == "eliminated" for dv in DVS))
        return {
            "condition": condition,
            "mean_gap_separated": mean_sep,
            "mean_gap_merged": mean_merged,
            "n_seed_pairs": gap["n_seed_pairs"],
            "per_seed_diffs": gap["per_seed_diffs"],
            "non_degenerate": non_degenerate,
            "criterion_passed": criterion_passed,
            "dv_blocks": dv_blocks,
            "h1_reading_by_dv": {dv: dv_blocks[dv]["h1_reading"] for dv in DVS},
        }

    def _criterion_record(name: str, v: Dict[str, Any], condition: str) -> Dict[str, Any]:
        """One load-bearing criterion, recorded so it is RE-DERIVABLE from the manifest.

        `passed: true` alone is an ASSERTION, not a record. V3-EXQ-936a set a bar ~7,900x
        above the maximum attainable effect and was logged clean for three governance cycles
        on exactly that shape. So every load-bearing criterion here carries, as numeric
        fields, the MEASURED value it routed on and the THRESHOLD it was compared against --
        per DV, because the two DVs are on different scales and a single pair would be a
        false summary of a conjunction over four cells.
        """
        per_dv = {}
        for dv in DVS:
            b = v["dv_blocks"][dv]
            ci = b["paired_ci"]
            per_dv[dv] = {
                "measured": ci.get("mean"),
                "measured_ci_low": ci.get("ci_low"),
                "measured_ci_high": ci.get("ci_high"),
                "threshold": b["margin_threshold"],
                "reading": b["h1_reading"],
                "n_seed_pairs": b["n_seed_pairs"],
                "sd": ci.get("sd"),
                "half_width": ci.get("half_width"),
            }
        gap = v["dv_blocks"]["calibration_gap"]
        return {
            "name": name,
            "load_bearing": True,
            "passed": bool(v["criterion_passed"]),
            "condition": condition,
            # The headline pair, on the DV the registry leg states its effect in.
            "measured": gap["paired_ci"].get("mean"),
            "threshold": gap["margin_threshold"],
            "measured_control_mean": v["mean_gap_separated"],
            "control_floor_threshold": SEPARATED_SIGNAL_FLOOR,
            "per_dv": per_dv,
            "combination_rule": (
                "passes iff the condition is non-degenerate AND every DV's 95% CI on the "
                "paired mean (MERGED - SEPARATED) lies entirely ABOVE its own -MARGIN"
            ),
        }

    dense_v = _condition_verdict("DENSE")
    sparse_v = _condition_verdict("SPARSE")

    both_nd = dense_v["non_degenerate"] and sparse_v["non_degenerate"]
    only_dense_nd = dense_v["non_degenerate"] and not sparse_v["non_degenerate"]
    only_sparse_nd = sparse_v["non_degenerate"] and not dense_v["non_degenerate"]
    neither_nd = not dense_v["non_degenerate"] and not sparse_v["non_degenerate"]

    non_degenerate: bool
    degeneracy_reason: Optional[str]
    evidence_direction_note: Optional[str] = None
    if not readiness_met:
        label = "channel_head_machinery_not_ready_substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        evidence_direction_note = (
            "NO COMPARISON WAS MADE. Set explicitly so build_experiment_indexes.py "
                "does NOT rewrite this run's direction: the indexer computes "
                "direction_explicitly_set = bool(manifest['evidence_direction_note']) "
                "(line 1871) and then rewrites any non-explicit 'unknown' on a FAIL "
                "to 'weakens' (lines 3519-3520). V3-EXQ-993 emitted a bare 'unknown' "
                "and its 2026-09-03 autopsy had to hand-correct the record to "
                "'non_contributory'. non_degenerate/scoring_excluded does NOT rescue "
                "this: the gap register's conflict_ratio counts entries regardless of "
                "scoring_excluded. So the direction is stated as non_contributory AND "
                "this note is present, which are the two independent things that keep "
                "an instrument-not-ready run from being logged as evidence against "
                "ARC-021 or MECH-069. ""Trigger: a readiness precondition was unmet over the "
            "full grid."
        )
        non_degenerate = False
        degeneracy_reason = "readiness preconditions not met (see interpretation.preconditions)"
    elif neither_nd:
        label = "separated_baseline_signal_absent_both_conditions"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        evidence_direction_note = (
            "NO COMPARISON WAS MADE. Set explicitly so build_experiment_indexes.py "
                "does NOT rewrite this run's direction: the indexer computes "
                "direction_explicitly_set = bool(manifest['evidence_direction_note']) "
                "(line 1871) and then rewrites any non-explicit 'unknown' on a FAIL "
                "to 'weakens' (lines 3519-3520). V3-EXQ-993 emitted a bare 'unknown' "
                "and its 2026-09-03 autopsy had to hand-correct the record to "
                "'non_contributory'. non_degenerate/scoring_excluded does NOT rescue "
                "this: the gap register's conflict_ratio counts entries regardless of "
                "scoring_excluded. So the direction is stated as non_contributory AND "
                "this note is present, which are the two independent things that keep "
                "an instrument-not-ready run from being logged as evidence against "
                "ARC-021 or MECH-069. ""Trigger: neither hazard-density condition's CONTROL arm "
            "cleared the non-degeneracy floor with full seed-pair coverage, so the "
            "ablation had nothing to remove -- exactly V3-EXQ-993's own outcome."
        )
        non_degenerate = False
        degeneracy_reason = (
            f"Neither DENSE (mean_sep={dense_v['mean_gap_separated']}, "
            f"n_pairs={dense_v['n_seed_pairs']}/{len(SEEDS)}) nor SPARSE "
            f"(mean_sep={sparse_v['mean_gap_separated']}, n_pairs={sparse_v['n_seed_pairs']}/{len(SEEDS)}) "
            f"cleared SEPARATED_SIGNAL_FLOOR={SEPARATED_SIGNAL_FLOOR} with full seed-pair coverage."
        )
    elif both_nd:
        # ================== H1 VERDICT ASSEMBLY ==========================================
        # SIGN DISCIPLINE, and it is not academic: V3-EXQ-1011's red-team caught a real
        # inversion when 993a's assembly was reused verbatim. In 993a `criterion_passed`
        # meant "MERGED DEGRADED by >= 0.15" and mapped to `supports`. HERE, as in 1011, it
        # means the degradation was ELIMINATED -- the OPPOSITE claim -- and maps to
        # `weakens`. `_h1_verdict` is pure and `--self-test` walks the whole 3^4 cube
        # asserting `supports` iff some cell CONFIRMED, so the inversion cannot recur
        # silently.
        #
        # The three readings, in priority order, over BOTH DVs and BOTH conditions
        # (CONTROL vs TREATMENT only -- the diagnostic arm reaches no criterion):
        readings = [dense_v["dv_blocks"][dv]["h1_reading"] for dv in DVS] + \
                   [sparse_v["dv_blocks"][dv]["h1_reading"] for dv in DVS]
        non_degenerate = True
        degeneracy_reason = None
        outcome, label, evidence_direction = _h1_verdict(readings)
        if evidence_direction == "non_contributory":
            evidence_direction_note = (
                "NO USABLE COMPARISON. At least one (DV, condition) cell's 95% CI on the "
                "paired mean STRADDLES -MARGIN, so neither the elimination nor the "
                "confirmation of the pre-registered degradation is supported at that cell. Set "
                "explicitly so build_experiment_indexes.py does not rewrite a FAIL's "
                "unknown direction to 'weakens' -- an under-powered interval is the absence "
                "of evidence, not evidence of absence, and the whole point of this leg is "
                "that V3-EXQ-993a's design could not tell those apart. The per-cell CIs and "
                "their half-widths are in interpretation.condition_detail so a successor can "
                "size n from THIS regime's measured SD rather than from a frozen-encoder one."
            )
    else:
        scored, scored_name = (dense_v, "DENSE") if only_dense_nd else (sparse_v, "SPARSE")
        unscored_name = "SPARSE" if only_dense_nd else "DENSE"
        _readings = [scored["dv_blocks"][dv]["h1_reading"] for dv in DVS]
        outcome = "PASS" if all(r != "undetermined" for r in _readings) else "FAIL"
        # Red-team second-pass finding 3: a single hazard-density condition
        # was never meant to carry full evidential weight -- ARC-021's own
        # falsifying-signature text asks for BOTH dense- and sparse-hazard
        # conditions. A claim-scoring pipeline reading only
        # evidence_direction/evidence_direction_per_claim has no visibility
        # into interpretation.label, so cap this branch at "mixed" rather
        # than a full-strength "supports"/"weakens" regardless of which way
        # the single scored condition's criterion fell.
        evidence_direction = "mixed"
        label = f"single_condition_only_partial_evidence_{scored_name.lower()}"
        non_degenerate = True
        degeneracy_reason = (
            f"{unscored_name} condition did not clear SEPARATED_SIGNAL_FLOOR with full seed-pair "
            f"coverage -- only {scored_name} contributes (partial evidence, capped at evidence_direction=mixed)."
        )

    evidence_direction_per_claim = {c: evidence_direction for c in CLAIM_IDS}
    # A note is written on EVERY branch that reports a real comparison too, so a
    # later reader never has to infer whether the direction was deliberate.
    if evidence_direction_note is None:
        evidence_direction_note = (
            f"Direction set deliberately from the pre-registered combination rule "
            f"(label={label}); a comparison WAS made in at least one condition."
        )

    def _flat(x: Any) -> Optional[float]:
        """Numeric-or-drop, for the flat scalar readout.

        Two encoding rules the runpack converter and the indexer enforce, and both are
        silent failures if ignored: a raw bool is NOT numeric to `_is_number`, so booleans
        go out as 0/1 ints; and a non-finite value IS numeric to the indexer and would
        pollute a delta, so NaN/inf/None are DROPPED (an absent key correctly reads as
        unmeasured).
        """
        if isinstance(x, bool):
            return float(int(x))
        if x is None:
            return None
        try:
            f = float(x)
        except (TypeError, ValueError):
            return None
        return f if f == f and f not in (float("inf"), float("-inf")) else None

    def _dvb(v: Dict[str, Any], dv: str, key: str) -> Any:
        try:
            b = v["dv_blocks"][dv]
        except (KeyError, TypeError):
            return None
        return b["paired_ci"].get(key) if key in ("mean", "sd", "ci_low", "ci_high", "half_width") \
            else b.get(key)

    # ---- FLAT SCALAR READOUT ------------------------------------------------------------
    # A top-level FLAT dict of SCALARS holding the pre-registered quantities the verdict
    # turns on. Without it the runpack converter harvests `values={}`, no `fail_if` stop
    # threshold can fire, the duplicate-emission supersession fingerprint is skipped and the
    # index carries no deltas -- and V3-EXQ-1011 shipped without one, so this is a gap closed
    # rather than a convention followed.
    readout_raw = {
        "dense_gap_paired_mean": _dvb(dense_v, "calibration_gap", "mean"),
        "dense_gap_paired_sd": _dvb(dense_v, "calibration_gap", "sd"),
        "dense_gap_ci_low": _dvb(dense_v, "calibration_gap", "ci_low"),
        "dense_gap_ci_high": _dvb(dense_v, "calibration_gap", "ci_high"),
        "sparse_gap_paired_mean": _dvb(sparse_v, "calibration_gap", "mean"),
        "sparse_gap_paired_sd": _dvb(sparse_v, "calibration_gap", "sd"),
        "sparse_gap_ci_low": _dvb(sparse_v, "calibration_gap", "ci_low"),
        "sparse_gap_ci_high": _dvb(sparse_v, "calibration_gap", "ci_high"),
        "dense_auc_paired_mean": _dvb(dense_v, "attribution_auc", "mean"),
        "dense_auc_paired_sd": _dvb(dense_v, "attribution_auc", "sd"),
        "dense_auc_ci_low": _dvb(dense_v, "attribution_auc", "ci_low"),
        "dense_auc_ci_high": _dvb(dense_v, "attribution_auc", "ci_high"),
        "sparse_auc_paired_mean": _dvb(sparse_v, "attribution_auc", "mean"),
        "sparse_auc_paired_sd": _dvb(sparse_v, "attribution_auc", "sd"),
        "sparse_auc_ci_low": _dvb(sparse_v, "attribution_auc", "ci_low"),
        "sparse_auc_ci_high": _dvb(sparse_v, "attribution_auc", "ci_high"),
        "dense_mean_gap_separated": dense_v["mean_gap_separated"],
        "dense_mean_gap_merged": dense_v["mean_gap_merged"],
        "sparse_mean_gap_separated": sparse_v["mean_gap_separated"],
        "sparse_mean_gap_merged": sparse_v["mean_gap_merged"],
        "dense_mean_gap_shared_enc": _mean_or_none(rows, "DENSE", "SHARED_ENC_SEP_HEADS"),
        "sparse_mean_gap_shared_enc": _mean_or_none(rows, "SPARSE", "SHARED_ENC_SEP_HEADS"),
        "margin_calibration_gap": MARGIN,
        "margin_attribution_auc": MARGIN_AUC,
        "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
        "n_seed_pairs_dense": dense_v["n_seed_pairs"],
        "n_seed_pairs_sparse": sparse_v["n_seed_pairs"],
        "worst_harm_action_sensitivity": worst_harm_sensitivity,
        "c1_dense_passed": dense_v["criterion_passed"],
        "c2_sparse_passed": sparse_v["criterion_passed"],
        "non_degenerate": non_degenerate,
        "readiness_met": readiness_met,
        "encoder_frozen_in_p1": False,
        "detach_mse_targets": DETACH_MSE_TARGETS,
        # Latent-collapse and clip diagnostics, per criterion arm -- the two facts a reader
        # needs to judge whether the unfreeze behaved as designed.
        "mean_zharm_dispersion_separated": _arm_mean(rows, CONTROL_ARM, "zharm_dispersion"),
        "mean_zharm_dispersion_merged": _arm_mean(rows, TREATMENT_ARM, "zharm_dispersion"),
        "mean_clip_active_frac_separated": _arm_mean(rows, CONTROL_ARM, "clip_active_frac"),
        "mean_clip_active_frac_merged": _arm_mean(rows, TREATMENT_ARM, "clip_active_frac"),
    }
    readout = {k: _flat(v) for k, v in readout_raw.items()}
    readout = {k: v for k, v in readout.items() if v is not None}

    metrics = {
        "mean_calibration_gap_dense_separated": dense_v["mean_gap_separated"],
        "mean_calibration_gap_dense_merged": dense_v["mean_gap_merged"],
        "mean_calibration_gap_sparse_separated": sparse_v["mean_gap_separated"],
        "mean_calibration_gap_sparse_merged": sparse_v["mean_gap_merged"],
        "dense_n_seed_pairs": dense_v["n_seed_pairs"],
        "sparse_n_seed_pairs": sparse_v["n_seed_pairs"],
        "worst_harm_action_sensitivity": worst_harm_sensitivity,
        "dense_harm_events": dense_harm_events,
        "sparse_harm_events": sparse_harm_events,
        "n_finite_violations_total": n_finite_violations_total,
        "margin": MARGIN,
        "margin_attribution_auc": MARGIN_AUC,
        "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
        "mean_attribution_auc_dense_separated": _arm_mean(
            [r for r in rows if r["condition"] == "DENSE"], CONTROL_ARM, "attribution_auc"),
        "mean_attribution_auc_dense_merged": _arm_mean(
            [r for r in rows if r["condition"] == "DENSE"], TREATMENT_ARM, "attribution_auc"),
        "mean_attribution_auc_sparse_separated": _arm_mean(
            [r for r in rows if r["condition"] == "SPARSE"], CONTROL_ARM, "attribution_auc"),
        "mean_attribution_auc_sparse_merged": _arm_mean(
            [r for r in rows if r["condition"] == "SPARSE"], TREATMENT_ARM, "attribution_auc"),
        # DIAGNOSTIC ARM -- encoder merged, heads separate. No criterion reads these; they
        # are what lets a reader attribute any degradation to the ENCODER level (the registry
        # leg's own name) rather than to the trunk level 993a/1011 already measured as null.
        "mean_calibration_gap_dense_shared_enc": _mean_or_none(rows, "DENSE", "SHARED_ENC_SEP_HEADS"),
        "mean_calibration_gap_sparse_shared_enc": _mean_or_none(rows, "SPARSE", "SHARED_ENC_SEP_HEADS"),
        # LATENT-COLLAPSE WATCH. The detached MSE targets exist to prevent the encoder
        # collapsing to a constant; these record whether that held, per arm, instead of
        # assuming it.
        "mean_zharm_dispersion_by_arm": {a: _arm_mean(rows, a, "zharm_dispersion") for a in ARMS},
        "mean_clip_active_frac_by_arm": {a: _arm_mean(rows, a, "clip_active_frac") for a in ARMS},
        "mean_pre_clip_norm_by_arm": {a: _arm_mean(rows, a, "mean_pre_clip_norm") for a in ARMS},
        # POWER, STATED AT THE PRE-REGISTERED EFFECT -- failure_autopsy_V3-EXQ-993a section 15
        # learning 9: "always state the power at the pre-registered effect alongside the
        # interval -- that is what stops 'not detected' being read as 'excluded'." Computed
        # from the REALISED paired SD of this run, not from a design-time estimate.
        "power_note": _power_note(dense_v, sparse_v),
    }

    return {
        "outcome": outcome,
        "readout": readout,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "evidence_direction_note": evidence_direction_note,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "combination_rule": (
                "Per (DV, condition): read the 95% CI on the paired mean (MERGED - SEPARATED). "
                "ELIMINATED if ci_low > -MARGIN_BY_DV[dv]; CONFIRMED if ci_high < it; "
                "UNDETERMINED if it straddles. A CONDITION's criterion passes iff the condition "
                "is non-degenerate (all seeds paired AND control-arm mean calibration_gap >= "
                f"SEPARATED_SIGNAL_FLOOR={SEPARATED_SIGNAL_FLOOR}) AND EVERY DV reads ELIMINATED "
                "there. RUN verdict: any CONFIRMED cell -> PASS/supports; all four cells "
                "ELIMINATED -> PASS/weakens; otherwise FAIL/non_contributory. If only one "
                "condition clears non-degeneracy the direction is capped at 'mixed' "
                "(single_condition_only_partial_evidence); if neither, non_degenerate=false. "
                f"MARGIN={MARGIN} on calibration_gap is the registry leg "
                f"H-encoder-level-merge-degrades' own pre-registered effect; MARGIN_AUC="
                f"{MARGIN_AUC} is its counterpart on attribution_auc's own scale (a shared "
                "number across two differently-scaled DVs would be assignment by assertion -- "
                "V3-EXQ-1011 red-team F4). SIGN-CONSISTENCY IS NOT A CONJUNCT: V3-EXQ-993a "
                "required all 4 seeds to agree in direction, which at n>=32 is a near-"
                "impossible requirement that converts every genuine effect back into a null. "
                "993a's rule is still evaluated and recorded per DV as "
                "`legacy_993a_criterion_passed`, for comparability only. The dv_headroom "
                "preconditions are evaluated over the SEPARATED cells BEFORE any MERGED cell "
                "is trained, and the DIAGNOSTIC arm SHARED_ENC_SEP_HEADS reaches no criterion "
                "and no precondition."
            ),
            "criteria": [
                _criterion_record("C1_dense_margin_degradation_eliminated", dense_v, "DENSE"),
                _criterion_record("C2_sparse_margin_degradation_eliminated", sparse_v, "SPARSE"),
            ],
            "criteria_non_degenerate": {
                "C1_dense_margin_degradation_eliminated": bool(readiness_met and dense_v["non_degenerate"]),
                "C2_sparse_margin_degradation_eliminated": bool(readiness_met and sparse_v["non_degenerate"]),
            },
            "condition_detail": {"DENSE": dense_v, "SPARSE": sparse_v},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="H1 verdict grid + criterion-can-fire arithmetic + helper invariants; no compute")
    ap.add_argument("--smoke-force-gate", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test_h1())
    if not args.dry_run:
        # THIS DRIVER IS NOT QUEUED AND ITS BARS WERE NEVER FINALISED AGAINST A LIVE REGIME.
        # A scored manifest from it would enter the evidence record as if it had been
        # pre-registered, which it was not. `--dry-run` and `--self-test` are the only
        # sanctioned entry points; the successor re-homes this file into experiments/ with
        # its own EXQ number and its own measured bars.
        raise SystemExit(
            "REFUSED: this is the ARC-021 H1 INSTRUMENT, not a queued experiment -- it "
            "produced a design-time refusal and was never pre-registered. Run it with "
            "--dry-run or --self-test. See the module docstring and "
            "REE_assembly/evidence/planning/"
            "arc021_h1_leg_refused_readout_dies_under_unfreeze_20260911.md"
        )
    t0 = time.perf_counter()
    global SEEDS, P0_EPISODES, P1_EPISODES, STEPS_PER_EPISODE, TOTAL_EPISODES_PER_CELL, PROBE_RESETS
    global _SMOKE_FORCE_GATE
    if args.smoke_force_gate:
        if not args.dry_run:
            raise SystemExit(
                "--smoke-force-gate is a dry-run-only smoke positive control and "
                "must never be used for a scored run."
            )
        _SMOKE_FORCE_GATE = True
    if args.dry_run:
        SEEDS = [101]
        P0_EPISODES = 2
        P1_EPISODES = 2
        STEPS_PER_EPISODE = 10
        TOTAL_EPISODES_PER_CELL = P0_EPISODES + P1_EPISODES
        PROBE_RESETS = 2

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "arms": ARMS,
        "num_hazards": NUM_HAZARDS,
        "grid_size": GRID_SIZE,
        "obs_view_size": OBS_VIEW_SIZE,
        "world_dim": WORLD_DIM,
        "hidden_dim": HIDDEN_DIM,
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "probe_resets": PROBE_RESETS,
        "criterion_arms": list(CRITERION_ARMS),
        "diagnostic_arms": list(DIAGNOSTIC_ARMS),
        "margin": MARGIN,
        "margin_attribution_auc": MARGIN_AUC,
        "dvs": DVS,
        "ci_confidence": CI_CONFIDENCE,
        "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
        "harm_action_sensitivity_floor": HARM_ACTION_SENSITIVITY_FLOOR,
        "harm_readout": "action_conditioned_pre_sigmoid_logit",
        # THE MANIPULATION, recorded as config rather than left implicit in the code.
        "encoder_frozen_in_p1": False,
        "detach_mse_targets": DETACH_MSE_TARGETS,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "lr_encoder": LR_ENCODER,
        "lr_head": LR_HEAD,
        "sd_pessimistic_gap": SD_PESSIMISTIC_GAP,
        "sd_pessimistic_auc": SD_PESSIMISTIC_AUC,
        "supersedes": SUPERSEDES,
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "hypothesis_qid": HYPOTHESIS_QID,
        "hypothesis_leg": HYPOTHESIS_LEG,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "experimental",
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "evidence_direction_note": result.get("evidence_direction_note"),
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        # FLAT SCALAR READOUT -- the machine-readable projection of the nested blocks. The
        # runpack converter harvests `readout` and the indexer reads its numeric entries;
        # without it `values={}` and no stop threshold can fire.
        "readout": result.get("readout", {}),
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "summary": (
            f"ARC-021/MECH-069 H1 drive-axis leg -- the merged-channel ablation with the "
            f"ENCODER UNFROZEN in P1, so a collapsed objective can corrupt the latent rather "
            f"than only a downstream trunk: {result['outcome']} "
            f"({result['interpretation']['label']}). "
            f"DENSE calibration_gap: separated={result['metrics']['mean_calibration_gap_dense_separated']} "
            f"vs merged={result['metrics']['mean_calibration_gap_dense_merged']}. "
            f"SPARSE calibration_gap: separated={result['metrics']['mean_calibration_gap_sparse_separated']} "
            f"vs merged={result['metrics']['mean_calibration_gap_sparse_merged']}. "
            f"non_degenerate={result['non_degenerate']}."
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(
        f"C1_dense={result['interpretation']['criteria'][0]['passed']} "
        f"C2_sparse={result['interpretation']['criteria'][1]['passed']} "
        f"non_degenerate={result['non_degenerate']}",
        flush=True,
    )
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
