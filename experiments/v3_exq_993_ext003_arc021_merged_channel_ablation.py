"""V3-EXQ-993 -- EXT-003/ARC-021/MECH-069 merged-channel ablation.

Claims: EXT-003 (external_failure_mode -- "Reward hacking: scalar reward
conflates incommensurable error signals"), ARC-021 (architectural_commitment
-- "Three BG-like cortico-striatal loops require distinct learning
channels"), MECH-069 (mechanism_hypothesis -- "Sensory prediction error,
motor-sensory error, and harm/goal error are incommensurable and cannot be
collapsed"). Proposal EXP-0528 / EVB-1248 (experiment_proposals.v1.json),
dispatch_mode=targeted_probe. (Dispatch brief named EXP-0522; that id does
not exist in the current, regenerated proposals file -- EXP-0528 is the
`(backlog_id=EVB-1248, proposal_type=experimental)` row for claim_id
EXT-003, confirmed by direct lookup 2026-09-02; its sibling literature_review
row is LIT-0529, not touched here.)

RED-TEAM VERDICT (Step 4.5, model fable, 2026-09-02): BLOCKING on the first
pass -- F1: the MERGED arm's sensory-head target was z_t1 (reached via a
random ACTUAL action) fed through a trunk(z, STAY) pass shared with the
harm readout, a designed-in target contradiction that would degrade the
shared trunk on every training step regardless of whether the three error
signals are genuinely incommensurable (confirmed against the dry-run
manifest: MERGED's forward-sensitivity was already 3.2x lower than
SEPARATED's with the harm channel completely untrained). Also confirmed:
F2 (harm head trained positive-label-only, never a negative example), F3
(MARGIN==SEPARATED_SIGNAL_FLOOR coupling forces near-total signal collapse
to fire a criterion at floor-scale baselines, with no seed-variance check
anywhere), F4 (SPARSE's baseline was never degeneracy-gated even though
C2_sparse counted toward the verdict), F5 (per-cell probe-coverage failures
and fatal errors were computed but never gated, and an empty probe list
silently substituted 0.0 rather than propagating "no data"), F6
(FORWARD_SENSITIVITY_FLOOR=1e-4 is vacuous -- any nonzero-weight network
clears it), F7 (the harm-event floor was pooled across both conditions,
letting DENSE's volume mask a near-zero SPARSE count). ALL SEVEN FIXED
before queuing (see inline comments at each site: SensoryHead's target is
now self-reconstruction z_t->z_t, not z_t1; harm_label is always 0.0/1.0,
never skipped; SEPARATED_SIGNAL_FLOOR raised to 0.02 and decoupled from
MARGIN=0.01; criteria are now per-condition-gated PAIRED per-seed
comparisons requiring sign-consistency across all 3 seeds, not a pooled
mean; per-cell coverage below MIN_PROBE_COVERAGE=5 excludes that cell from
aggregation instead of contributing a fabricated 0.0; the harm-event floor
is now MIN(dense_total, sparse_total) >= 10, not a pooled sum;
FORWARD_SENSITIVITY_FLOOR raised to 0.005). Each fix is cited to its exact
source line via inline comments so a later reader can re-verify the
disposition against the diff.

SECOND PASS (model fable, same day, per the "re-spawn exactly once when a
BLOCKING finding changed the manipulation/criterion" rule): NOT BLOCKING.
All seven first-pass fixes verified present and effective in source (the
self-reconstruction fix genuinely removes the STAY-pass contradiction
without introducing a new one; the train/eval distribution mismatch on
harm_logit is structurally symmetric across arms, not a MERGED-specific
confound; the paired sign-consistency bar is meaningfully strict, not a
naive 1/8). Three further findings, none BLOCKING: N1 (CONTESTED, real bug
-- `f"{None:.4f}"` on an excluded cell's calibration_gap would crash the
whole run, silently making the F5 exclusion path dead code -- FIXED, one
guarded format, changes no manipulation/criterion); N2 (docstring drift,
a stale 0.01 reference to SEPARATED_SIGNAL_FLOOR -- FIXED, this section
rewritten); finding 3 (CONTESTED -- the single-condition partial-evidence
branch emitted a full-strength "supports"/"weakens" that a claim-scoring
consumer reading only evidence_direction/evidence_direction_per_claim
could not distinguish from a full both-conditions result -- FIXED, that
branch now emits evidence_direction="mixed" unconditionally, see
LOAD-BEARING CRITERIA below). N3 (informational, not fixed -- the 0.02
SEPARATED_SIGNAL_FLOOR sits near the top of the historical signal range,
so a real chance exists that one or both conditions land in the
degenerate/"unknown" regime at full scale; this is the intended honest
behavior of the non-degeneracy gate, not something to paper over by
lowering the floor).

THE LOAD-BEARING TEST. ARC-021's own `what_would_answer` field (claims.yaml)
states the decisive test has never been run: collapse E1 (sensory
prediction) / E2 (motor-sensory action-outcome) / E3 (harm/goal) into a
shared trunk / single optimizer / one combined loss, and check whether the
SD-003 causal-signature calibration_gap = mean_causal_sig(near_hazard) -
mean_causal_sig(safe) is measurably DEGRADED relative to the
separated-channel baseline. Existing baseline evidence (V3-EXQ-007/008/009/
010) is weak: all four are formally FAIL against their own pre-registered
>0.05 bar; only EXQ-010 (dense hazards, 3x3 view) was even positive-direction
(+0.0267). A prior auto-tagged probe (2026-08-06, indexed on ARC-021, conf
0.5) is on record as "not the decisive merged-channel ablation ARC-021
needs" (failure_autopsy_grandfathered-misc2-ninethread-cluster_2026-08-08.
json). This script is the first attempt at the actual ablation.

GOV-REUSE-1 (Step 2.4): decisive readout is calibration_gap under a
merged-trunk/single-optimizer condition that has never existed in this
codebase (confirmed via substrate audit 2026-09-02: `ree_core/` has no
E1+E2+E3-spanning shared-trunk/merged-optimizer mechanism anywhere -- the
one precedent, ARC-058's HarmForwardTrunk, spans only the two harm streams
E2_harm_s/E2_harm_a, not E1/E2/E3). No manifest can carry this readout
because the manipulation itself has never been instantiated. Not
recoverable -> run.

SUBSTRATE READINESS (Step 2.5/2.5c). This driver deliberately does NOT
import `ree_core.agent`/`REEAgent` or any of the E1/E2/E3 predictor modules
(`e1_deep.py`, `e2_world.py`, `e2_fast.py`, `e2_harm_s.py`, `e2_harm_a.py`,
`e3_selector.py`) -- two OPEN corrupting-severity substrate_queue entries
(`mode-governance-engagement`, `SD-082`) whole-file-list `ree_core/agent.py`,
which a module-level-match gate treats as blocking regardless of which
agent.py method is actually called. The manipulation under test (shared
trunk / single optimizer / combined loss vs separate networks / separate
optimizers / separate losses) is about OPTIMIZER AND PARAMETER-SHARING
TOPOLOGY, not about any specific existing E1/E2/E3 implementation detail --
so it is legitimately instantiated as EXPERIMENT-LOCAL predictor heads
(mirroring each loop's I/O role: sensory next-state prediction, action-
conditioned forward prediction, harm/goal evaluation) reading off a
self-contained world_encoder, exactly as `SmallViewEnv` in V3-EXQ-010 is
experiment-local env code, not substrate code. This keeps the ablation's
substrate footprint to `ree_core/environment/causal_grid_world.py` only.
That file has one OPEN entry, `SD-MECH303-THRESHOLD-SOURCING` (severity:
degrading, contextual_safety_harm_threshold decoupling) -- noted per Step
2.5c, not blocking; this driver never reads `contextual_safety_harm_threshold`
or any MECH-303 field, so the open amendment is inert for this run.
`ree_core/latent/stack.py` (SplitEncoder) is likewise NOT imported -- the
world_encoder/world_decoder pair below is a self-contained 2-layer MLP
architecturally identical to `SplitEncoder.world_encoder`, so the ablation
does not depend on it either.

Re-derive brake (Step 2.5b): the one ARC-021-tagged autopsy on record
(2026-08-06 auto-tag, conf 0.5) is a single preliminary/weak-power probe,
not a substrate_ceiling-category autopsy, and is explicitly documented as
NOT the decisive test -- count is 0 toward the >=2 threshold. Brake does not
fire.

DESIGN. Two hazard-density conditions (DENSE: 12x12 grid, 15 hazards, 3x3
local view -- the one SD-003 config with positive-direction signal, ported
from V3-EXQ-010's `SmallViewEnv`; SPARSE: identical 12x12/3x3 view, 4
hazards -- isolates hazard density as ARC-021's own falsifying-signature
text calls for "both dense- and sparse-hazard conditions"). Two arms,
identical in every respect except optimizer/parameter topology:

  ARM_SEPARATED: three independent heads (SensoryHead, ForwardHead,
    HarmHead), three independent torch.optim.Adam instances, three
    independent losses -- no shared parameters, no combined backward pass.
    This operationalizes ARC-021's premise as literally as an
    experiment-local ablation can: E1/E2/E3-analog roles that cannot
    mutually satisfy a single exploit because their gradients never mix.

  ARM_MERGED: one shared MLP trunk consumes (z_world, action_onehot) and
    all three heads (sensory/forward/harm) read off its hidden
    representation; ONE torch.optim.Adam over trunk+heads; ONE combined
    loss = sensory_loss + forward_loss + harm_loss, backpropagated jointly
    every step. This operationalizes "collapse into a single scalar" --
    the three normally-incommensurable error signals are forced through a
    shared representation and a single gradient signal.

Both arms share: the SAME world_encoder/world_decoder architecture, warmed
up identically in P0 (phased training, MANDATORY -- P0 encoder warmup ->
P1 frozen-encoder head training on .detach()ed z_world -> P2 eval probes);
the SAME env config per (condition, seed); a full `reset_all_rng(seed)` at
cell entry (via `arm_cell`), so P0 is bit-identical across arms at a given
(condition, seed) and the only degree of freedom between arms is P1's head/
optimizer topology. The env, encoder, decoder, and channel heads are all
independent per-cell objects (no state shared BETWEEN cells), so cells are
genuinely order-independent and correctly `reuse_eligible` under
`arm_fingerprint` bookkeeping (no `extra_ineligible_reasons` needed).

The trunk's "state-only" pass (used for sensory prediction and harm
evaluation) feeds action_onehot(STAY) rather than a zero vector, reusing
the env's own no-op action (index 4, dx=dy=0) as the "evaluate this state"
convention rather than an out-of-distribution zero input.

CAUSAL SIGNATURE (both arms, same computation shape, differing only in
which network instantiates predict_forward/harm_logit):
    z_pred_actual = predict_forward(z_world_at_probe, a_actual)
    z_pred_cf     = predict_forward(z_world_at_probe, a_cf)
    causal_sig    = sigmoid(harm_logit(z_pred_actual)) - sigmoid(harm_logit(z_pred_cf))
    calibration_gap = mean(causal_sig | near_hazard) - mean(causal_sig | safe)
identical in shape to V3-EXQ-007-010's `E2(z_t,a_actual) - E2(z_t,a_cf)`
methodology, substituting this experiment's own forward/harm heads for
`agent.e2.world_forward`/`agent.e3.harm_eval`.

DV-SYMMETRY (Step 3 mandatory declaration). calibration_gap is a continuous
mean-difference of a sigmoid-bounded scalar (causal_sig), not an argmax/
rank-derived statistic and not a set-aggregate over interchangeable units
(near-hazard and safe probes are drawn from geometrically distinct
populations, never pooled then split). The SEPARATED-vs-MERGED manipulation
changes which parameters receive gradient from which loss term during
training -- it is not a uniform additive constant broadcast across
candidates (nothing here computes an argmax/softmax-sample over
candidates), not a monotone rescaling of an existing ranking (the DV is
never ranked or thresholded before being averaged), and not a permutation of
interchangeable seeds/units (each cell's forward/harm heads are trained from
scratch per cell -- MERGED's shared trunk is not a permutation-symmetric
function of anything SEPARATED also computes). The manipulation is
therefore not invariant under any of the three symmetry classes that have
previously produced an unfalsifiable design in this codebase (V3-EXQ-604c);
it can genuinely move calibration_gap in either direction.

PRECONDITIONS (positive control, Step 2.5a / P0 readiness-assert). Both are
readiness-kind checks on the SAME predictive machinery the load-bearing
criteria route through (forward_head action-sensitivity and harm-labeled
training data), not a proxy on an unrelated statistic:
  1. forward_head_action_sensitivity_present: worst-cell (across all 12
     cells) mean |predict_forward(z,a1) - predict_forward(z,a2)| over a
     fixed probe batch of z_world samples and two distinct movement
     actions, measured immediately after P1 training. A below-floor
     (0.005, post-red-team fix F6) reading means the forward head collapsed
     to a constant regardless of action, which would make ANY causal-
     signature reading (either arm) structurally uninformative -- self-
     routes to substrate_not_ready_requeue.
  2. p1_harm_events_observed_per_condition: MIN(total P1 harm events summed
     over 2 arms x 3 seeds) across DENSE and SPARSE (post-red-team fix F7 --
     a pooled sum let DENSE's volume mask a starved SPARSE count). Below
     10 means too little harm-labeled data existed in at least one
     condition -- self-routes to substrate_not_ready_requeue.

LOAD-BEARING CRITERIA, computed PER CONDITION by `_condition_verdict()`
(post-red-team fixes F3/F4 -- see RED-TEAM VERDICT above for what this
replaced): for each of DENSE and SPARSE, a PAIRED per-seed comparison
diff_i = calibration_gap(MERGED, seed_i) - calibration_gap(SEPARATED, seed_i)
(pairing removes cross-seed P0 variance, since P0 is bit-identical across
arms at a given seed). A condition's criterion (C1_dense / C2_sparse) fires
only if the condition is non-degenerate (below) AND mean(diffs) <= -MARGIN
(0.01) AND ALL 3 per-seed diffs are <= 0 (sign-consistency -- a pooled-mean
threshold alone cannot distinguish a genuine effect from one seed's outlier
driving the mean, which a MARGIN sized near the historical noise floor
would not otherwise guard against).

Non-degeneracy is evaluated PER CONDITION, independently (fix F4 -- a
pooled/DENSE-only gate previously let a degenerate SPARSE comparison count
toward the verdict via C2 with no check at all): a condition is
non-degenerate iff ALL 3 seeds produced a valid paired cell (no cell
excluded by the coverage gate below) AND SEPARATED's own mean
calibration_gap for that condition clears SEPARATED_SIGNAL_FLOOR = 0.02
(deliberately > MARGIN, fix F3 -- decoupled so a scored comparison never
requires near-total signal collapse to fire; calibrated against the
0.0067-0.0267 historical range, near the top of it, since these are
simpler custom heads than the original E1/E2/E3 -- expect a real chance of
landing in the degenerate/"unknown" regime for one or both conditions;
this is an honest design tradeoff, not a defect, per red-team second-pass
finding N3).

A per-cell coverage gate (fix F5) additionally excludes any cell from
aggregation whose near-hazard or safe probe count falls below
MIN_PROBE_COVERAGE=5, or which hit a fatal probe-execution error --
`calibration_gap` is `None` for that cell rather than a fabricated 0.0, and
a condition with fewer than 3 valid paired cells cannot be non-degenerate.

Three-way outcome branch: if BOTH conditions are non-degenerate, PASS iff
C1_dense AND C2_sparse (evidence_direction "supports" if both hold,
"weakens" if neither, "mixed" if exactly one). If NEITHER condition is
non-degenerate, the whole run is `non_degenerate: false`, evidence_direction
"unknown" (no informative comparison was made). If exactly ONE condition is
non-degenerate, the outcome is that condition's criterion alone, labeled
`single_condition_only_partial_evidence_<condition>`, and
evidence_direction is deliberately "mixed" rather than a full-strength
"supports"/"weakens" (red-team second-pass finding 3 -- a single hazard-
density condition was never meant to carry the same evidential weight as
ARC-021's own "both dense- and sparse-hazard conditions" falsifying-
signature text asks for; a claim-scoring pipeline reading only
`evidence_direction`/`evidence_direction_per_claim` has no visibility into
`interpretation.label`, so the field itself must reflect the partiality).

SLEEP DRIVER: not applicable -- no sleep loop used in this probe.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_993_ext003_arc021_merged_channel_ablation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["ARC-021", "EXT-003", "MECH-069"]

SEEDS = [101, 202, 303]
CONDITIONS = ["DENSE", "SPARSE"]
ARMS = ["SEPARATED", "MERGED"]
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

# Pre-registered thresholds (defined here, never inferred post-hoc).
# Calibrated post-red-team (2026-09-02, model fable, see RED-TEAM VERDICT in the
# module docstring): SEPARATED_SIGNAL_FLOOR is deliberately > MARGIN (decoupled,
# fix F3) so a scored comparison never REQUIRES near-total signal collapse to
# fire; FORWARD_SENSITIVITY_FLOOR is raised off the near-vacuous 1e-4 (fix F6).
MARGIN = 0.01                       # C1/C2: MERGED must be at least this much LOWER
SEPARATED_SIGNAL_FLOOR = 0.02       # non-degeneracy per condition: SEPARATED must clear this
FORWARD_SENSITIVITY_FLOOR = 0.005   # precondition 1
HARM_EVENTS_FLOOR_PER_CONDITION = 10  # precondition 2, MIN across the two conditions (fix F7)
MIN_PROBE_COVERAGE = 5              # per-cell minimum n_near / n_safe to count (fix F5)


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
    choices = [a for a in range(num_actions) if a != actual_idx]
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


class SeparatedChannels:
    """ARM_SEPARATED: three independent heads, three independent
    optimizers, three independent losses. No shared parameters."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.sensory_head = nn.Sequential(
            nn.Linear(WORLD_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, WORLD_DIM)
        )
        self.forward_head = nn.Sequential(
            nn.Linear(WORLD_DIM + action_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, WORLD_DIM)
        )
        self.harm_head = nn.Sequential(
            nn.Linear(WORLD_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 1)
        )
        self.opt_sensory = optim.Adam(self.sensory_head.parameters(), lr=LR_HEAD)
        self.opt_forward = optim.Adam(self.forward_head.parameters(), lr=LR_HEAD)
        self.opt_harm = optim.Adam(self.harm_head.parameters(), lr=LR_HEAD)

    def predict_forward(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.forward_head(torch.cat([z, action_onehot], dim=-1))

    def predict_sensory(self, z: torch.Tensor) -> torch.Tensor:
        return self.sensory_head(z)

    def harm_logit(self, z: torch.Tensor) -> torch.Tensor:
        return self.harm_head(z)

    def train_step(
        self, z_t: torch.Tensor, action_onehot: torch.Tensor, z_t1: torch.Tensor,
        harm_label: torch.Tensor,
    ) -> Dict[str, float]:
        # Sensory target is self-reconstruction (z_t -> z_t), NOT z_t1: z_t1 was
        # reached via a random ACTUAL action, so training an action-free head
        # against it would silently make "sensory prediction" a degenerate
        # proxy for the (already-separate) forward-prediction task. A stable
        # persistence/consistency objective is the coherent action-free
        # analogue (red-team fix F1; see module docstring DESIGN section).
        sensory_loss = F.mse_loss(self.sensory_head(z_t), z_t)
        self.opt_sensory.zero_grad()
        sensory_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sensory_head.parameters(), 1.0)
        self.opt_sensory.step()

        forward_loss = F.mse_loss(self.predict_forward(z_t, action_onehot), z_t1)
        self.opt_forward.zero_grad()
        forward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.forward_head.parameters(), 1.0)
        self.opt_forward.step()

        # harm_label is ALWAYS 0.0 or 1.0 (never skipped) -- red-team fix F2:
        # training on positive-only labels leaves harm_head undiscriminating
        # in any cell with sparse harm events (worse, at pure random init).
        harm_loss = F.binary_cross_entropy_with_logits(self.harm_head(z_t1), harm_label)
        self.opt_harm.zero_grad()
        harm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.harm_head.parameters(), 1.0)
        self.opt_harm.step()

        return {
            "sensory_loss": float(sensory_loss.item()),
            "forward_loss": float(forward_loss.item()),
            "harm_loss": float(harm_loss.item()),
        }


class MergedChannels:
    """ARM_MERGED: one shared trunk, one optimizer over trunk+heads, one
    combined loss backpropagated jointly. The trunk consumes
    (z_world, action_onehot); STAY (index 4) is used as the "evaluate this
    state" action for the sensory and harm heads (state-only passes),
    while the actually-taken/counterfactual action drives the forward
    head's action-conditioned pass."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(WORLD_DIM + action_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )
        self.sensory_out = nn.Linear(HIDDEN_DIM, WORLD_DIM)
        self.forward_out = nn.Linear(HIDDEN_DIM, WORLD_DIM)
        self.harm_out = nn.Linear(HIDDEN_DIM, 1)
        params = (
            list(self.trunk.parameters())
            + list(self.sensory_out.parameters())
            + list(self.forward_out.parameters())
            + list(self.harm_out.parameters())
        )
        self.opt = optim.Adam(params, lr=LR_HEAD)
        self._stay = _action_onehot(STAY_ACTION_IDX, action_dim)

    def _stay_onehot(self, batch_size: int) -> torch.Tensor:
        return self._stay.expand(batch_size, -1)

    def predict_forward(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        repr_ = self.trunk(torch.cat([z, action_onehot], dim=-1))
        return self.forward_out(repr_)

    def predict_sensory(self, z: torch.Tensor) -> torch.Tensor:
        repr_ = self.trunk(torch.cat([z, self._stay_onehot(z.shape[0])], dim=-1))
        return self.sensory_out(repr_)

    def harm_logit(self, z: torch.Tensor) -> torch.Tensor:
        repr_ = self.trunk(torch.cat([z, self._stay_onehot(z.shape[0])], dim=-1))
        return self.harm_out(repr_)

    def train_step(
        self, z_t: torch.Tensor, action_onehot: torch.Tensor, z_t1: torch.Tensor,
        harm_label: torch.Tensor,
    ) -> Dict[str, float]:
        # Sensory target is self-reconstruction (z_t -> z_t) through the SAME
        # trunk(z, STAY) pass the harm head reads -- both are now genuinely
        # "evaluate/represent this state" objectives, coherent with each
        # other and with STAY as the "no action" convention (red-team fix
        # F1: the prior z_t1 target was reached via a random ACTUAL action,
        # contradicting the STAY-conditioned trunk pass on every step).
        sensory_loss = F.mse_loss(self.predict_sensory(z_t), z_t)
        forward_loss = F.mse_loss(self.predict_forward(z_t, action_onehot), z_t1)
        # harm_label is ALWAYS 0.0 or 1.0 (never skipped) -- red-team fix F2.
        harm_loss = F.binary_cross_entropy_with_logits(self.harm_logit(z_t1), harm_label)
        total = sensory_loss + forward_loss + harm_loss
        self.opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.trunk.parameters()) + list(self.sensory_out.parameters())
            + list(self.forward_out.parameters()) + list(self.harm_out.parameters()),
            1.0,
        )
        self.opt.step()
        return {
            "sensory_loss": float(sensory_loss.item()),
            "forward_loss": float(forward_loss.item()),
            "harm_loss": float(harm_loss.item()),
        }


def _make_channels(arm: str, action_dim: int):
    return SeparatedChannels(action_dim) if arm == "SEPARATED" else MergedChannels(action_dim)


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


def _run_p1(
    env: SmallViewEnv, encoder: nn.Module, channels, action_dim: int,
) -> Dict[str, Any]:
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    total_harm_events = 0
    for ep in range(P1_EPISODES):
        transitions = _collect_random_episode(env)
        for world_obs_t, action_idx, world_obs_t1, harm_signal in transitions:
            with torch.no_grad():
                z_t = encoder(world_obs_t.unsqueeze(0))
                z_t1 = encoder(world_obs_t1.unsqueeze(0))
            action_onehot = _action_onehot(action_idx, action_dim)
            # harm_label is ALWAYS 0.0 or 1.0 -- red-team fix F2 (positive-only
            # labels leave harm_head undiscriminating, worse in sparse-harm cells).
            is_harm = harm_signal < 0
            if is_harm:
                total_harm_events += 1
            harm_label = torch.ones(1, 1) if is_harm else torch.zeros(1, 1)
            channels.train_step(z_t, action_onehot, z_t1, harm_label)
        if (ep + 1) % 20 == 0 or ep == P1_EPISODES - 1:
            print(
                f"  [train] p1 ep {P0_EPISODES + ep + 1}/{TOTAL_EPISODES_PER_CELL} "
                f"harm_events={total_harm_events}",
                flush=True,
            )
    return {"p1_harm_events": total_harm_events}


def _forward_action_sensitivity(encoder: nn.Module, channels, env: SmallViewEnv, action_dim: int) -> float:
    """Positive control: mean |predict_forward(z,a1) - predict_forward(z,a2)|
    over a fixed probe batch and two distinct movement actions."""
    with torch.no_grad():
        probe_worlds = []
        for _ in range(16):
            _, obs_dict = env.reset()
            probe_worlds.append(obs_dict["world_state"].unsqueeze(0))
        z = encoder(torch.cat(probe_worlds, dim=0))
        a1 = _action_onehot(0, action_dim).expand(z.shape[0], -1)
        a2 = _action_onehot(1, action_dim).expand(z.shape[0], -1)
        diff = channels.predict_forward(z, a1) - channels.predict_forward(z, a2)
        return float(diff.abs().mean().item())


def _eval_probes(env: SmallViewEnv, encoder: nn.Module, channels, action_dim: int) -> Dict[str, Any]:
    encoder.eval()
    near_sigs: List[float] = []
    safe_sigs: List[float] = []
    fatal_errors = 0

    wall_type = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _run_probe(ax: int, ay: int, actual_idx: int) -> float:
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            z = encoder(obs_dict["world_state"].unsqueeze(0))
            cf_idx = _random_cf_action(actual_idx, action_dim)
            a_act = _action_onehot(actual_idx, action_dim)
            a_cf = _action_onehot(cf_idx, action_dim)
            z_pred_act = channels.predict_forward(z, a_act)
            z_pred_cf = channels.predict_forward(z, a_cf)
            h_act = torch.sigmoid(channels.harm_logit(z_pred_act))
            h_cf = torch.sigmoid(channels.harm_logit(z_pred_cf))
            return float((h_act - h_cf).item())

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

    return {
        "calibration_gap": calibration_gap,
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
    }
    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        env = SmallViewEnv(size=GRID_SIZE, num_hazards=NUM_HAZARDS[condition], seed=seed)
        action_dim = env.action_dim
        encoder, decoder = _make_world_codec(env.world_obs_dim)
        codec_opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR_ENCODER)

        p0_stats = _run_p0(env, encoder, decoder, codec_opt)
        channels = _make_channels(arm, action_dim)
        p1_stats = _run_p1(env, encoder, channels, action_dim)

        forward_sensitivity = _forward_action_sensitivity(encoder, channels, env, action_dim)
        probe_stats = _eval_probes(env, encoder, channels, action_dim)

        # Red-team fix N1 (2nd pass): calibration_gap is None when coverage_ok
        # is False -- f"{None:.4f}" raises TypeError, which would crash the
        # whole run and make the F5 exclusion path unreachable dead code.
        _gap = probe_stats["calibration_gap"]
        _gap_str = f"{_gap:.4f}" if _gap is not None else "None(coverage_insufficient)"
        print(
            f"  [{condition}_{arm} seed={seed}] gap={_gap_str} "
            f"n_near={probe_stats['n_near_hazard_probes']} n_safe={probe_stats['n_safe_probes']} "
            f"fwd_sensitivity={forward_sensitivity:.5f} harm_events={p1_stats['p1_harm_events']}",
            flush=True,
        )

        row = {
            "condition": condition,
            "arm": arm,
            "seed": seed,
            "mean_recon_loss": p0_stats["mean_recon_loss"],
            "p1_harm_events": p1_stats["p1_harm_events"],
            "forward_action_sensitivity": forward_sensitivity,
            **probe_stats,
        }
        cell.stamp(row)

    passed = bool(row["coverage_ok"])
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        for arm in ARMS:
            for seed in SEEDS:
                rows.append(_run_cell(condition, arm, seed))

    n_finite_violations_total = sum(r["fatal_errors"] for r in rows)

    worst_forward_sensitivity = min(r["forward_action_sensitivity"] for r in rows)
    dense_harm_events = sum(r["p1_harm_events"] for r in rows if r["condition"] == "DENSE")
    sparse_harm_events = sum(r["p1_harm_events"] for r in rows if r["condition"] == "SPARSE")
    min_condition_harm_events = min(dense_harm_events, sparse_harm_events)

    precondition_1 = {
        "name": "forward_head_action_sensitivity_present",
        "description": (
            "Worst-cell mean |predict_forward(z,a1)-predict_forward(z,a2)| over a "
            "fixed probe batch and two distinct movement actions, measured after P1 "
            "training. A below-floor reading means the forward head is action-blind "
            "in at least one cell, which would make its causal-signature reading "
            "structurally uninformative regardless of arm."
        ),
        "measured": worst_forward_sensitivity,
        "threshold": FORWARD_SENSITIVITY_FLOOR,
        "direction": "lower",
        "control": "16-sample probe batch, actions 0 vs 1, worst cell across all 12 cells",
        "met": bool(worst_forward_sensitivity >= FORWARD_SENSITIVITY_FLOOR),
    }
    precondition_2 = {
        "name": "p1_harm_events_observed_per_condition",
        "description": (
            "MIN(total P1 harm events summed over arms+seeds) across the two hazard-"
            "density conditions -- red-team fix F7: a pooled floor let SPARSE's harm "
            "head train on near-zero labeled data while DENSE's volume masked it. "
            "Taking the min across conditions requires BOTH to have real coverage."
        ),
        "measured": float(min_condition_harm_events),
        "threshold": float(HARM_EVENTS_FLOOR_PER_CONDITION),
        "direction": "lower",
        "control": f"dense={dense_harm_events}, sparse={sparse_harm_events} (each summed over 2 arms x 3 seeds)",
        "met": bool(min_condition_harm_events >= HARM_EVENTS_FLOOR_PER_CONDITION),
    }
    preconditions = [precondition_1, precondition_2]
    readiness_met = precondition_1["met"] and precondition_2["met"]

    def _condition_verdict(condition: str) -> Dict[str, Any]:
        """Red-team fixes F3+F4: a PAIRED per-seed comparison (same seed drives
        bit-identical P0 in both arms, so pairing removes cross-seed P0
        variance) with a sign-consistency requirement (all seeds must agree
        in direction, not just the pooled mean), and per-CONDITION
        non-degeneracy gating (a degenerate SPARSE baseline can no longer
        silently count toward the verdict via C2)."""
        sep_by_seed = {
            r["seed"]: r["calibration_gap"] for r in rows
            if r["condition"] == condition and r["arm"] == "SEPARATED" and r["calibration_gap"] is not None
        }
        mer_by_seed = {
            r["seed"]: r["calibration_gap"] for r in rows
            if r["condition"] == condition and r["arm"] == "MERGED" and r["calibration_gap"] is not None
        }
        paired_seeds = sorted(set(sep_by_seed) & set(mer_by_seed))
        diffs = [mer_by_seed[s] - sep_by_seed[s] for s in paired_seeds]
        mean_sep = statistics.fmean(sep_by_seed.values()) if sep_by_seed else None
        mean_merged = statistics.fmean(mer_by_seed.values()) if mer_by_seed else None
        # Non-degenerate requires EVERY seed to have a valid paired cell (no
        # silent power loss from an excluded cell) AND the separated baseline
        # to clear the (decoupled) signal floor.
        non_degenerate = bool(
            len(paired_seeds) == len(SEEDS) and mean_sep is not None and mean_sep >= SEPARATED_SIGNAL_FLOOR
        )
        criterion_passed = bool(
            non_degenerate and diffs
            and statistics.fmean(diffs) <= -MARGIN
            and all(d <= 0 for d in diffs)
        )
        return {
            "condition": condition,
            "mean_gap_separated": mean_sep,
            "mean_gap_merged": mean_merged,
            "n_seed_pairs": len(paired_seeds),
            "per_seed_diffs": diffs,
            "non_degenerate": non_degenerate,
            "criterion_passed": criterion_passed,
        }

    dense_v = _condition_verdict("DENSE")
    sparse_v = _condition_verdict("SPARSE")

    both_nd = dense_v["non_degenerate"] and sparse_v["non_degenerate"]
    only_dense_nd = dense_v["non_degenerate"] and not sparse_v["non_degenerate"]
    only_sparse_nd = sparse_v["non_degenerate"] and not dense_v["non_degenerate"]
    neither_nd = not dense_v["non_degenerate"] and not sparse_v["non_degenerate"]

    non_degenerate: bool
    degeneracy_reason: Optional[str]
    if not readiness_met:
        label = "channel_head_machinery_not_ready_substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        degeneracy_reason = "readiness preconditions not met (see interpretation.preconditions)"
    elif neither_nd:
        label = "separated_baseline_signal_absent_both_conditions"
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        degeneracy_reason = (
            f"Neither DENSE (mean_sep={dense_v['mean_gap_separated']}, "
            f"n_pairs={dense_v['n_seed_pairs']}/{len(SEEDS)}) nor SPARSE "
            f"(mean_sep={sparse_v['mean_gap_separated']}, n_pairs={sparse_v['n_seed_pairs']}/{len(SEEDS)}) "
            f"cleared SEPARATED_SIGNAL_FLOOR={SEPARATED_SIGNAL_FLOOR} with full seed-pair coverage."
        )
    elif both_nd:
        overall = dense_v["criterion_passed"] and sparse_v["criterion_passed"]
        outcome = "PASS" if overall else "FAIL"
        non_degenerate = True
        degeneracy_reason = None
        if overall:
            label = "merged_channel_degrades_calibration_both_conditions"
            evidence_direction = "supports"
        elif dense_v["criterion_passed"] or sparse_v["criterion_passed"]:
            label = "merged_channel_degrades_calibration_one_condition_only"
            evidence_direction = "mixed"
        else:
            label = "merged_channel_hypothesis_not_supported"
            evidence_direction = "weakens"
    else:
        scored, scored_name = (dense_v, "DENSE") if only_dense_nd else (sparse_v, "SPARSE")
        unscored_name = "SPARSE" if only_dense_nd else "DENSE"
        outcome = "PASS" if scored["criterion_passed"] else "FAIL"
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

    metrics = {
        "mean_calibration_gap_dense_separated": dense_v["mean_gap_separated"],
        "mean_calibration_gap_dense_merged": dense_v["mean_gap_merged"],
        "mean_calibration_gap_sparse_separated": sparse_v["mean_gap_separated"],
        "mean_calibration_gap_sparse_merged": sparse_v["mean_gap_merged"],
        "dense_n_seed_pairs": dense_v["n_seed_pairs"],
        "sparse_n_seed_pairs": sparse_v["n_seed_pairs"],
        "worst_forward_action_sensitivity": worst_forward_sensitivity,
        "dense_harm_events": dense_harm_events,
        "sparse_harm_events": sparse_harm_events,
        "n_finite_violations_total": n_finite_violations_total,
        "margin": MARGIN,
        "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
    }

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "combination_rule": (
                "PASS iff C1_dense AND C2_sparse when BOTH conditions clear non-degeneracy; "
                "if only one condition clears it, outcome is that condition's criterion alone "
                "(label prefixed single_condition_only_partial_evidence); if neither clears it, "
                "unknown/non_degenerate=false. Each criterion additionally requires ALL 3 seeds "
                "to show MERGED<=SEPARATED (sign-consistency), not just a pooled mean margin."
            ),
            "criteria": [
                {"name": "C1_dense_merged_degrades_vs_separated", "load_bearing": True, "passed": dense_v["criterion_passed"]},
                {"name": "C2_sparse_merged_degrades_vs_separated", "load_bearing": True, "passed": sparse_v["criterion_passed"]},
            ],
            "criteria_non_degenerate": {
                "C1_dense_merged_degrades_vs_separated": bool(readiness_met and dense_v["non_degenerate"]),
                "C2_sparse_merged_degrades_vs_separated": bool(readiness_met and sparse_v["non_degenerate"]),
            },
            "condition_detail": {"DENSE": dense_v, "SPARSE": sparse_v},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, P0_EPISODES, P1_EPISODES, STEPS_PER_EPISODE, TOTAL_EPISODES_PER_CELL, PROBE_RESETS
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
        "margin": MARGIN,
        "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": "V3-EXQ-993",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "experimental",
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "summary": (
            f"EXT-003/ARC-021/MECH-069 merged-channel ablation: {result['outcome']} "
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
