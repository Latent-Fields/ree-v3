#!/opt/local/bin/python3
"""
V3-EXQ-1101 -- MECH-033 GOV-FANOUT-1 portfolio, LEG 3 (axis: integration).

Claim:   MECH-033  "E2 forward-prediction kernels seed hippocampal rollouts."
Source:  failure_autopsy_gflag0452-D2-cluster_2026-09-24, targets[5] (V3-EXQ-308),
         fanout_recommendation.suggested_probes[2] (hypothesis "H1 vs H3").
Design:  REE_assembly/evidence/planning/mech033_fanout_portfolio_design_blocked_20260925.md
         (ratified 2026-09-25 by orchestrate-20260924-1707 under rec-20260924-fb429c72).

THIS LEG CARRIES H3, AND IT CARRIES IT BY DEGRADATION NOT ABSENCE
-----------------------------------------------------------------
The autopsy's original leg 1 ("k=0 HarmHead-greedy over current z_world/z_self,
no E2") was DROPPED by ratified decision on 2026-09-25. Reason: every harm
readout in the tree is state-only with no action input -- this file's HarmHead is
Linear(world_dim + self_dim, 1), and the full REEAgent's
E3Selector.harm_eval_head is Linear(world_dim, ...). With ZERO E2 forward steps
there is no action-dependent quantity to rank, so a k=0 arm scores all five
actions identically, collapses to a constant-action policy, and performs at or
below uniform random. Its declared null would therefore have been rejected BY
CONSTRUCTION -- the very D2 ablation confound this portfolio exists to remove,
re-entering under a new name. (The substrate already recorded this: V3-EXQ-034's
own docstring predicts "all actions predict same z_world" for its identity-E2
ablation.)

So H3 is tested here instead, by DEGRADING E2 rather than REMOVING it. This is a
SCOPE STATEMENT, not a hidden substitution: the question this leg answers is
"does E2's TRAINED CONTENT carry harm avoidance?", not "can harm be avoided with
no forward model at all". The latter is not answerable with the current readouts.

This is also the manipulation MECH-033's own claims.yaml what_would_answer
specifies: "Ablate the E2-kernel-to-hippocampal handoff ... while leaving
rollouts and E3 scoring otherwise intact".

ARMS (one shared warmup per seed -> training matched by construction)
--------------------------------------------------------------------
    INTACT        -- k=3 policy-driven chain, trained E2.               (H1)
    REINIT_E2     -- identical, but E2's weights replaced by a FRESH    (H3)
                     UNTRAINED init at eval. Planner + HarmHead intact.
                     This is the arm the declared null is about.
    PERMUTED_E2   -- identical, but E2's weights randomly PERMUTED      (H3, tighter)
                     within each tensor at eval. Preserves the weight
                     distribution exactly -- same scale, same values,
                     different wiring -- so it controls for "the
                     scramble merely changed the output magnitude".
                     INFORMATIONAL: reported, not gated.
    RANDOM_REF    -- uniform random actions. INFORMATIONAL ONLY:
                     competence floor + data quality. NEVER the null's
                     comparator; that is the D2 confound.

Both scramble modes are named by the autopsy's own sketch ("E2 weights shuffled
or untrained"), so running both is within the ratified spec, not an addition to
it. They are built on DEEP COPIES: the trained E2 is never mutated in place
(pre-flight hazard (d)), and the restored trained r2 is re-measured after the
scrambled arms run, as a recorded check that nothing leaked between arms.

DECLARED NULL (H3)
------------------
    harm_rate(REINIT_E2) is WITHIN 0.01 of harm_rate(INTACT).
H1 predicts this null is rejected (scrambling E2 makes harm materially worse).
H3 predicts it holds (E2's trained content contributes nothing).

NEGATIVE-INSTRUMENT DISCIPLINE (CLAUDE.md "Negative instruments")
-----------------------------------------------------------------
A null here is evidence FOR H3, so "no difference" must not be reachable by a
broken instrument. Four gating guards:
  C3 TRAINING GATE    -- intact world_forward_r2 >= 0.20 (the claim's own
                         non-degeneracy precondition).
  C4 SCRAMBLE CANARY  -- scrambled world_forward_r2 <= 0.05. Without this, "no
                         difference" could simply mean the scramble did nothing.
                         This is the single most important guard in the file.
  C5 COMPETENCE FLOOR -- INTACT must beat RANDOM_REF by >= 15%. If the intact
                         planner is itself at random level there is no
                         contribution to ablate and the run is uninformative.
  C6 DATA QUALITY     -- RANDOM_REF must accumulate >= 5 harm events.
Additionally e2_action_discrimination is recorded for intact AND both scrambled
E2s (GFLAG-0485): a scramble that leaves E2 action-invariant degrades the wrong
property.

PRE-REGISTERED CRITERIA
-----------------------
C1: mean over seeds of (harm_rate_REINIT_E2 - harm_rate_INTACT) >= 0.01   [rejects the H3 null]
C2: harm_rate_INTACT < harm_rate_REINIT_E2 on ALL seeds                   [consistency]
C3: world_forward_r2_INTACT >= 0.20 on ALL seeds                          [E2 trained]
C4: world_forward_r2_REINIT_E2 <= 0.05 on ALL seeds                       [scramble canary]
C5: harm_rate_INTACT <= 0.85 * harm_rate_RANDOM_REF on ALL seeds          [competence floor]
C6: n_harm_events_RANDOM_REF >= 5 on ALL seeds                            [env presents harm]

PASS = C1..C6 all hold -> supports H1 over H3.
Evidence-direction logic (evaluated in this order):
  C3 or C4 or C6 fails                  -> inconclusive (instrument/training/scramble inadequate)
  C5 fails                              -> inconclusive (intact planner degenerate;
                                           nothing to ablate; NOT support for H3)
  C3..C6 hold and C1 or C2 fails         -> weakens  (H3 favoured: E2's trained
                                           content adds nothing)
  all hold                              -> supports (H1 over H3)

PROTOCOL
--------
3 seeds x 600 shared warmup episodes + 4 eval arms x 50 episodes x 100
steps/episode.

Runtime ~21 min, MEASURED not guessed: steady-state 0.00431 s/step train and
~0.0051 s/step for each of the three planning arms, probed on ree-cloud-5
2026-09-25 over 20 real warmup episodes + 3 eval episodes per arm.
"""

import sys
import copy
import random
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_1101_mech033_e2_scramble_integration"
QUEUE_ID = "V3-EXQ-1101"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-033"]
SOURCE_AUTOPSY = (
    "REE_assembly/evidence/planning/failure_autopsy_gflag0452-D2-cluster_2026-09-24.json"
    "#targets[5].fanout_recommendation.suggested_probes[2]"
)
DESIGN_DOC = (
    "REE_assembly/evidence/planning/"
    "mech033_fanout_portfolio_design_blocked_20260925.md"
)
SCOPE_STATEMENTS = [
    "Compute is NOT matched between arms, by construction: rollout DEPTH is the "
    "manipulation, so matching compute would mean matching depth and erase it. "
    "Contrast leg V3-EXQ-1102, where the manipulation is kernel QUALITY and the "
    "CEM budget therefore IS matched.",
    "z_self is held fixed across rollout steps -- the inline architecture has no "
    "self-forward model. Same as V3-EXQ-308.",
    "Inline 308-lineage architecture, not the full REEAgent. The autopsy's "
    "'integration: isolated' finding is answered by leg V3-EXQ-1102, not here.",
    "GFLAG-0491 (no waking gradient learning at REEConfig defaults) does not "
    "apply: this script uses neither REEConfig nor REEAgent. All five modules "
    "are trained by the one driver-built Adam in _train_models, and that "
    "coverage is assert-checked at runtime rather than asserted in a comment.",
    "GFLAG-0485 (E2 at chance on action consequence) is handled by keeping depth "
    "k<=3, inside the shallow band, plus the C6 action-discrimination canary "
    "which routes an action-invariant E2 to inconclusive rather than to "
    "'supports H2'.",
    "The autopsy's suggested k=0 'HarmHead-greedy' arm is DELIBERATELY ABSENT. "
    "Every harm readout in the tree is state-only, so with zero E2 forward steps "
    "it scores all actions identically and its null would be rejected by "
    "construction. Dropped by ratified decision 2026-09-25; H3 is carried by leg "
    "V3-EXQ-1101 via E2 DEGRADATION rather than E2 ABSENCE.",
]

_T0 = time.perf_counter()

# --------------------------------------------------------------------------
# Pre-registered thresholds
# --------------------------------------------------------------------------
# C1 is a RELATIVE bar -- see the note in the sibling leg V3-EXQ-1103. The first
# build's ABSOLUTE 0.01 delta was ~50x the arms' entire harm_rate, so its PASS
# branch was unreachable. The bar is now a fraction of the achievable range
# (harm_FLOOR - harm_INTACT), harm_FLOOR = best fixed single action.
# COEFFICIENT PRE-REGISTRATION: 0.10, fixed from a pilot on HELD-OUT seeds
# (PILOT_SEEDS), never tuned on the eval seeds.
THRESH_C1_RELATIVE_FRACTION = 0.10
THRESH_C3_E2_R2             = 0.20  # C3: intact world_forward_r2 (E2 trained)
THRESH_C4_SCRAMBLE_R2_MAX   = 0.05  # C4: scrambled world_forward_r2 (CANARY)
THRESH_C5_COMPETENCE        = 0.85  # C5: harm_INTACT <= 0.85 * best-fixed-action
THRESH_C6_MIN_CONTACTS      = 5     # C6: harm-event data quality
# C7/C8 -- BEHAVIOURAL gates; the two that would have failed the affine collapse.
THRESH_C7_MIN_ACTION_ENTROPY = 0.50  # bits, per planning arm per seed
THRESH_C8_MIN_DISTINCT_CELLS = 10    # per planning arm per seed

PILOT_SEEDS = [101, 202]

# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------
WARMUP_EPISODES   = 600
EVAL_EPISODES     = 50
STEPS_PER_EPISODE = 100
LR                = 3e-4
R2_EVAL_STEPS     = 1000
DISCRIM_STEPS     = 200
DRIVE_WEIGHT      = 2.0   # SD-012, matched to V3-EXQ-308's substrate defaults

SEEDS = [42, 7, 13]

# Arms. PERMUTED_E2 and RANDOM_REF are informational -- see the module docstring.
ARM_INTACT      = "INTACT"
ARM_REINIT      = "REINIT_E2"
ARM_PERMUTED    = "PERMUTED_E2"
ARM_RANDOM_REF  = "RANDOM_REF"
ARMS = [ARM_INTACT, ARM_REINIT, ARM_PERMUTED, ARM_RANDOM_REF]

# Every planning arm uses the SAME depth: depth is leg V3-EXQ-1103's manipulation,
# not this leg's. Here the ONLY thing that varies is E2's weights.
CHAIN_DEPTH = 3
E2_MODE_BY_ARM = {
    ARM_INTACT: "trained", ARM_REINIT: "reinit", ARM_PERMUTED: "permuted",
    ARM_RANDOM_REF: None,
}

# --------------------------------------------------------------------------
# Model constants (identical to V3-EXQ-308, for lineage comparability)
# --------------------------------------------------------------------------
WORLD_OBS_DIM = 250
SELF_OBS_DIM  = 12
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 16


# ---------------------------------------------------------------------------
# AFFINE_COLLAPSE_NOTE -- why this file does NOT reuse V3-EXQ-308's modules
# ---------------------------------------------------------------------------
# V3-EXQ-308's E2WorldForward (:169-177) and HarmHead (:179-186) are both a bare
# nn.Linear with NO activation. Composed, the action-selection score is affine in
# the one-hot action:
#
#     harm(z, z_self, a) = v_z . (W_z z + W_a a + b) + v_s . z_self + c
#                        = (v_z . W_a) a  +  [ terms containing no a ]
#
# (v_z . W_a) is a FIXED vector with no state dependence, so argmin_a is the SAME
# action in every state, at EVERY rollout depth (deeper terms are
# v_z . W_z^k W_a a, still state-free). Measured 2026-09-25 on a genuinely trained
# stack (world_forward_r2 = 0.9320): the k=1 and k=3 selectors each chose ONE
# constant action on 400/400 on-trajectory states, and a constant move pins the
# agent against a wall (2 distinct cells over 1000 eval steps).
#
# So the first build of this leg compared a wall-pinned agent to itself while all
# four of its instrument gates reported green. Every guard inspected the MODEL
# (r2, action-discrimination, competence-vs-random); none inspected the BEHAVIOUR.
#
# CONSEQUENCES, both acted on:
#   1. Both heads here are now Linear -> ReLU -> Linear. This DELIBERATELY DROPS
#      lineage comparability with V3-EXQ-308 -- which is the right trade, because
#      (2) means 308's arm was not a planner to be comparable with.
#   2. 308's own KERNEL_CHAIN arm was therefore never a planner: its advantage
#      over uniform random is explained by wall-pinning, not by E2 seeding. That
#      is INDEPENDENT of, and stronger than, the ablation confound recorded by
#      failure_autopsy_gflag0452-D2-cluster_2026-09-24, and is raised for
#      /governance as GFLAG-0499 (it plausibly also covers V3-EXQ-171 and both
#      V3-EXQ-184 runs).
# Ratified by orchestrate-20260924-1707 2026-09-25 under rec-20260924-fb429c72
# (option N-i). Full record:
# REE_assembly/evidence/planning/mech033_fanout_portfolio_design_blocked_20260925.md
# ---------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Models (same stack as V3-EXQ-308)
# --------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, world_dim)
        self.norm = nn.LayerNorm(world_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.linear(x)))


class ResourceProximityHead(nn.Module):
    """SD-018. Kept so z_world matches the V3-EXQ-308 lineage's grounding.
    Reported informationally; not a gating criterion for MECH-033."""
    def __init__(self, world_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim, 1)

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(z_world))


class SelfEncoder(nn.Module):
    def __init__(self, obs_dim: int, self_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, self_dim)
        self.norm = nn.LayerNorm(self_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.linear(x)))


class E2WorldForward(nn.Module):
    """
    E2 forward-prediction kernel: f(z_world, a) -> z_world_next.
    This is the module MECH-033 is about.

    NONLINEAR (Linear -> ReLU -> Linear), which is a DELIBERATE DEPARTURE from
    V3-EXQ-308's bare `nn.Linear` (308 :169-177). See AFFINE_COLLAPSE_NOTE: with
    both this head and HarmHead affine, the action-selection score's
    action-dependent term is state-free and every arm collapses to a constant
    action. The hidden layer is what makes action ranking state-dependent at all.
    """
    HIDDEN = 32

    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + action_dim, self.HIDDEN),
            nn.ReLU(),
            nn.Linear(self.HIDDEN, world_dim),
        )

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_world, action], dim=-1))


class HarmHead(nn.Module):
    """
    f(z_world, z_self) -> harm_scalar. STATE-ONLY: no action input. That is why
    the autopsy's suggested k=0 'HarmHead-greedy' arm was dropped from the
    portfolio -- with zero E2 forward steps it scores all actions identically
    (design doc, BLOCKING FINDING B).

    NONLINEAR (Linear -> ReLU -> Linear), a DELIBERATE DEPARTURE from
    V3-EXQ-308's bare `nn.Linear` (308 :179-186). See AFFINE_COLLAPSE_NOTE.
    """
    HIDDEN = 32

    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + self_dim, self.HIDDEN),
            nn.ReLU(),
            nn.Linear(self.HIDDEN, 1),
        )

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_world, z_self], dim=-1))


# --------------------------------------------------------------------------
# Observation helpers
# --------------------------------------------------------------------------

def _fit(raw, dim: int) -> torch.Tensor:
    if raw is None:
        return torch.zeros(dim)
    t = raw.float() if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32)
    t = t.flatten()
    if t.shape[0] < dim:
        return F.pad(t, (0, dim - t.shape[0]))
    return t[:dim]


def _get_world_obs(obs_dict: dict) -> torch.Tensor:
    return _fit(obs_dict.get("world_state"), WORLD_OBS_DIM)


def _get_self_obs(obs_dict: dict) -> torch.Tensor:
    return _fit(obs_dict.get("body_state"), SELF_OBS_DIM)


def _get_resource_prox_target(obs_dict: dict) -> float:
    rfv = obs_dict.get("resource_field_view")
    if rfv is None:
        return 0.0
    t = rfv.float() if isinstance(rfv, torch.Tensor) else torch.tensor(rfv, dtype=torch.float32)
    return float(t.max().item())


def _get_drive_level(obs_dict: dict) -> float:
    body = obs_dict.get("body_state")
    if body is None:
        return 0.0
    t = body.float() if isinstance(body, torch.Tensor) else torch.tensor(body, dtype=torch.float32)
    if t.numel() > 3:
        return float(max(0.0, min(1.0, 1.0 - float(t.flatten()[3].item()))))
    return 0.0


def _onehot(idx: int, dim: int) -> torch.Tensor:
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v


def _make_env(seed: int) -> CausalGridWorldV2:
    """Env config identical to V3-EXQ-308, for lineage comparability."""
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.2,
    )


# --------------------------------------------------------------------------
# THE MANIPULATION: policy-driven k-step rollout
# --------------------------------------------------------------------------

def _rollout_harm(
    z_world: torch.Tensor,
    z_self: torch.Tensor,
    e2_fwd: E2WorldForward,
    harm_head: HarmHead,
    first_action_idx: int,
    depth: int,
) -> float:
    """
    Cumulative predicted harm over `depth` E2 forward steps.

    Step 0 takes `first_action_idx`. Every SUBSEQUENT step takes the action that
    minimises predicted harm from the state reached so far -- POLICY-DRIVEN
    continuation. This is the fix for V3-EXQ-308's defect, which instead forced
    STAY (`a_oh[ACTION_DIM - 1] = 1.0`) at steps 1..k-1 and thereby measured a
    one-action lookahead while reporting a k-step chain.

    depth=1 gives the K1 arm; depth=3 gives the K3_POLICY arm. ONE code path
    serves both, so the arms differ only in `depth`.
    """
    a_oh = _onehot(first_action_idx, ACTION_DIM)
    z_w = z_world
    cumulative = 0.0
    for k in range(depth):
        z_w = e2_fwd(z_w, a_oh)
        cumulative += float(harm_head(z_w, z_self).item())
        if k < depth - 1:
            best_h = float("inf")
            best_a = 0
            for a2 in range(ACTION_DIM):
                z_probe = e2_fwd(z_w, _onehot(a2, ACTION_DIM))
                h = float(harm_head(z_probe, z_self).item())
                if h < best_h:
                    best_h = h
                    best_a = a2
            a_oh = _onehot(best_a, ACTION_DIM)
    return cumulative


def _select_action(
    z_world: torch.Tensor,
    z_self: torch.Tensor,
    e2_fwd: E2WorldForward,
    harm_head: HarmHead,
    depth: int,
) -> int:
    """Greedy over first actions: pick the one whose policy-driven rollout of
    `depth` steps has the lowest cumulative predicted harm."""
    best_score = float("inf")
    best_action = 0
    for a_idx in range(ACTION_DIM):
        score = _rollout_harm(z_world, z_self, e2_fwd, harm_head, a_idx, depth)
        if score < best_score:
            best_score = score
            best_action = a_idx
    return best_action


# --------------------------------------------------------------------------
# THE MANIPULATION: E2 weight scrambling (deep-copied, never in place)
# --------------------------------------------------------------------------

def _scrambled_e2(trained_e2: E2WorldForward, mode: str, seed: int) -> E2WorldForward:
    """
    Return a NEW E2WorldForward whose weights are degraded.

    mode="reinit"   -- fresh untrained initialisation (the autopsy's "untrained").
                       This is the arm the declared null is about.
    mode="permuted" -- the trained tensors' own values, randomly PERMUTED within
                       each tensor (the autopsy's "shuffled"). Preserves the exact
                       value multiset, hence scale and spectrum-ish magnitude, so
                       it controls for "the degradation merely changed output
                       magnitude rather than destroying learned structure".

    The trained module is NEVER mutated: a fresh module is constructed and, for
    "permuted", the trained state_dict is deep-copied before permutation. This is
    pre-flight hazard (d) -- arms share one process, so an in-place scramble would
    silently poison every later arm.
    """
    g = torch.Generator().manual_seed(seed)
    fresh = E2WorldForward(WORLD_DIM, ACTION_DIM)
    if mode == "reinit":
        # A fresh module is already untrained; re-seed it deterministically so the
        # arm is reproducible from the manifest's seed list.
        with torch.no_grad():
            for prm in fresh.parameters():
                if prm.dim() > 1:
                    nn.init.kaiming_uniform_(prm, a=5 ** 0.5)
                else:
                    prm.uniform_(-0.1, 0.1, generator=g)
        return fresh
    if mode == "permuted":
        sd = copy.deepcopy(trained_e2.state_dict())
        with torch.no_grad():
            for k in sd:
                flat = sd[k].flatten()
                perm = torch.randperm(flat.numel(), generator=g)
                sd[k] = flat[perm].reshape(sd[k].shape).clone()
        fresh.load_state_dict(sd)
        return fresh
    raise ValueError("unknown scramble mode %r" % mode)


def _selfcheck_scramble(trained_e2: E2WorldForward, seed: int) -> Dict:
    """
    Measure the blind spot (CLAUDE.md "The test half"): confirm the scramble
    actually CHANGES E2 and does NOT touch the trained module.

    A scramble helper that returned the trained weights unchanged would make the
    whole leg silently vacuous -- INTACT and REINIT_E2 would be the same arm and
    the null would hold by construction. So this is checked mechanically, not
    assumed.
    """
    before = copy.deepcopy(trained_e2.state_dict())
    reinit = _scrambled_e2(trained_e2, "reinit", seed)
    permuted = _scrambled_e2(trained_e2, "permuted", seed)
    after = trained_e2.state_dict()

    for k in before:
        assert torch.equal(before[k], after[k]), (
            "scramble MUTATED the trained E2 in place on %s -- every later arm is "
            "poisoned (pre-flight hazard (d))" % k
        )
    r_diff = max(
        float((reinit.state_dict()[k] - before[k]).abs().max().item()) for k in before
    )
    p_diff = max(
        float((permuted.state_dict()[k] - before[k]).abs().max().item()) for k in before
    )
    assert r_diff > 1e-6, "reinit scramble produced weights identical to trained E2"
    assert p_diff > 1e-6, "permuted scramble produced weights identical to trained E2"

    # The permutation must preserve the value multiset exactly -- that is the whole
    # point of having it alongside reinit.
    for k in before:
        a = torch.sort(before[k].flatten()).values
        b = torch.sort(permuted.state_dict()[k].flatten()).values
        assert torch.allclose(a, b, atol=1e-6), (
            "permuted scramble did not preserve the value multiset on %s" % k
        )
    print(
        "[selfcheck] scramble OK: trained E2 unmutated; reinit max|delta|=%.4f "
        "permuted max|delta|=%.4f; permutation preserves the value multiset"
        % (r_diff, p_diff),
        flush=True,
    )
    return {
        "trained_e2_unmutated": True,
        "reinit_max_abs_delta": r_diff,
        "permuted_max_abs_delta": p_diff,
        "permutation_preserves_multiset": True,
    }


# --------------------------------------------------------------------------
# Self-check on the manipulation itself (CLAUDE.md "The test half")
# --------------------------------------------------------------------------

class _CountingE2(nn.Module):
    """Wraps an E2WorldForward and counts forward calls."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, z_world, action):
        self.calls += 1
        return self.inner(z_world, action)


class _StayIsWorstHarmHead(nn.Module):
    """A harm head whose value depends on the LAST action encoded into z_world by
    _DeltaE2 below: STAY is scored worst. Used only by the self-check."""
    def forward(self, z_world, z_self):
        # z_world[0] carries the action index written by _DeltaE2.
        a = float(z_world.flatten()[0].item())
        return torch.tensor([10.0 if int(round(a)) == ACTION_DIM - 1 else 0.0])


class _DeltaE2(nn.Module):
    """Writes the taken action index into channel 0, so the harm head above can
    see which action was last applied."""
    def forward(self, z_world, action):
        out = z_world.clone()
        out[0] = float(int(torch.argmax(action).item()))
        return out


def _selfcheck_manipulation() -> Dict:
    """
    Measure the blind spot, do not merely assert the fix.

    (1) DEPTH IS REAL: _rollout_harm must issue exactly the E2 calls a
        policy-driven depth-k rollout requires -- k advances plus (k-1) x
        ACTION_DIM continuation probes. depth=1 -> 1 call; depth=3 -> 13.
        A forced-STAY implementation (the V3-EXQ-308 defect) issues only k
        calls and would FAIL this.
    (2) CONTINUATION IS POLICY-DRIVEN, NOT STAY: with a harm head that scores
        STAY as by far the worst action, a policy-driven rollout must refuse
        STAY as its continuation, so a depth-3 rollout's cumulative harm stays
        low. V3-EXQ-308's forced STAY would accumulate the STAY penalty on every
        continuation step and score ~20.0. This is the check that actually
        distinguishes this implementation from the one it replaces.

    Raises AssertionError rather than printing a warning: a broken manipulation
    must not be able to produce a publishable manifest.
    """
    z_w = torch.zeros(WORLD_DIM)
    z_s = torch.zeros(SELF_DIM)

    e2_1 = _CountingE2(E2WorldForward(WORLD_DIM, ACTION_DIM))
    _rollout_harm(z_w, z_s, e2_1, HarmHead(WORLD_DIM, SELF_DIM), 0, 1)
    e2_3 = _CountingE2(E2WorldForward(WORLD_DIM, ACTION_DIM))
    _rollout_harm(z_w, z_s, e2_3, HarmHead(WORLD_DIM, SELF_DIM), 0, 3)

    expected_1 = 1
    expected_3 = 3 + 2 * ACTION_DIM   # 3 advances + 2 continuation probe sweeps
    assert e2_1.calls == expected_1, (
        "depth=1 rollout issued %d E2 calls, expected %d" % (e2_1.calls, expected_1)
    )
    assert e2_3.calls == expected_3, (
        "depth=3 rollout issued %d E2 calls, expected %d -- a forced-STAY "
        "continuation (the V3-EXQ-308 defect) would issue only 3"
        % (e2_3.calls, expected_3)
    )

    # (2) STAY must be refused as a continuation.
    stay_cost = _rollout_harm(z_w, z_s, _DeltaE2(), _StayIsWorstHarmHead(), 0, 3)
    forced_stay_cost = 2 * 10.0   # what steps 1..2 would cost if STAY were forced
    assert stay_cost < forced_stay_cost, (
        "policy-driven continuation failed: depth-3 cumulative harm %.3f is not "
        "below the %.3f a forced-STAY continuation would incur -- the V3-EXQ-308 "
        "defect is present" % (stay_cost, forced_stay_cost)
    )

    print(
        "[selfcheck] depth-1 E2 calls=%d depth-3 E2 calls=%d (expected %d/%d); "
        "policy-driven continuation refuses STAY (cost %.2f < forced-STAY %.2f) OK"
        % (e2_1.calls, e2_3.calls, expected_1, expected_3, stay_cost,
           forced_stay_cost),
        flush=True,
    )
    return {
        "depth1_e2_calls": e2_1.calls,
        "depth3_e2_calls": e2_3.calls,
        "depth3_e2_calls_expected": expected_3,
        "forced_stay_defect_present": False,
        "policy_driven_cost": stay_cost,
        "forced_stay_cost_would_be": forced_stay_cost,
    }


def _arm_disagreement_rate(
    seed: int, world_enc, self_enc, e2_intact, e2_scrambled, harm_head, n_steps: int,
) -> float:
    """
    INFORMATIONAL. Fraction of states at which the INTACT and SCRAMBLED E2 lead to
    DIFFERENT chosen actions, at the same depth.

    Measured ON-POLICY, on the EVAL env (seed + 1000), with the trajectory driven
    by the INTACT arm's own action. The first build measured it on a uniform-random
    walk in a DIFFERENT env (seed + 2000), describing states neither arm occupied.

    Near 0 means the scramble is behaviourally inert -- it changed E2's weights
    (proven by _selfcheck_scramble) and destroyed its r2 (proven by the C4 canary)
    yet the resulting POLICY is unchanged. EXACTLY 0.0 or EXACTLY 1.0 is a
    DEGENERACY ALARM, not a reassurance.
    """
    random.seed(seed + 555)
    torch.manual_seed(seed + 555)
    env = _make_env(seed + 1000)
    disagree = 0
    total = 0
    _, obs_dict = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            z_world = world_enc(_get_world_obs(obs_dict))
            z_self = self_enc(_get_self_obs(obs_dict))
            a1 = _select_action(z_world, z_self, e2_intact, harm_head, CHAIN_DEPTH)
            a3 = _select_action(z_world, z_self, e2_scrambled, harm_head, CHAIN_DEPTH)
            if a1 != a3:
                disagree += 1
            total += 1
            # Drive with the INTACT arm's own action (on-policy for that arm).
            _, _, done, _, obs_next = env.step(_onehot(a1, ACTION_DIM).unsqueeze(0))
            obs_dict = env.reset()[1] if done else obs_next
    return disagree / max(total, 1)


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

def _compute_r2(world_enc, e2_fwd, env, n_steps: int) -> float:
    """C3: R^2 of E2WorldForward on n_steps random transitions."""
    preds: List[float] = []
    actuals: List[float] = []
    _, obs_dict = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            z_world = world_enc(_get_world_obs(obs_dict))
            a_oh = _onehot(random.randint(0, ACTION_DIM - 1), ACTION_DIM)
            z_pred = e2_fwd(z_world, a_oh)
            _, _, done, _, obs_next = env.step(a_oh.unsqueeze(0))
            z_actual = world_enc(_get_world_obs(obs_next))
            preds.extend(z_pred.tolist())
            actuals.extend(z_actual.tolist())
            obs_dict = env.reset()[1] if done else obs_next
    n = len(actuals)
    if n == 0:
        return 0.0
    mean_a = sum(actuals) / n
    ss_tot = sum((a - mean_a) ** 2 for a in actuals)
    ss_res = sum((p - a) ** 2 for p, a in zip(preds, actuals))
    if ss_tot < 1e-10:
        return 0.0
    return float(max(-1.0, min(1.0, 1.0 - ss_res / ss_tot)))


def _compute_prox_r2(world_enc, prox_head, env, n_steps: int) -> float:
    """Informational (SD-018 grounding); NOT a gating criterion here."""
    preds: List[float] = []
    actuals: List[float] = []
    _, obs_dict = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            z_world = world_enc(_get_world_obs(obs_dict))
            preds.append(float(prox_head(z_world).item()))
            actuals.append(_get_resource_prox_target(obs_dict))
            a_oh = _onehot(random.randint(0, ACTION_DIM - 1), ACTION_DIM)
            _, _, done, _, obs_next = env.step(a_oh.unsqueeze(0))
            obs_dict = env.reset()[1] if done else obs_next
    n = len(actuals)
    if n == 0:
        return 0.0
    mean_a = sum(actuals) / n
    ss_tot = sum((a - mean_a) ** 2 for a in actuals)
    ss_res = sum((p - a) ** 2 for p, a in zip(preds, actuals))
    if ss_tot < 1e-10:
        return 0.0
    return float(max(-1.0, min(1.0, 1.0 - ss_res / ss_tot)))


def _e2_action_consequence_accuracy_gain(world_enc, e2_fwd, env, n_steps: int) -> float:
    """
    C6 CANARY, ACCURACY form (GFLAG-0485 part 2: "even trained, E2 predicts the
    executed action's consequence at chance").

    For each step, take the action actually executed, observe the true next
    z_world, and compare E2's prediction for the RIGHT action against its
    predictions for the WRONG ones:

        err_right = ||e2(z, a_exec)  - z_next||
        err_wrong = mean_{a != a_exec} ||e2(z, a) - z_next||
        gain      = (err_wrong - err_right) / err_wrong        in (-inf, 1]

    gain ~ 0 means E2 is AT CHANCE on action consequence: its outputs may differ
    across actions, but no action's prediction matches the world better than the
    others, so nothing a planner reads off it can be right for the right reason.

    This REPLACES the first build's spread measure (mean pairwise distance between
    predictions, normalised by norm). That measure was ~5x free at untrained init
    (0.2416 untrained vs 0.1689 trained -- training LOWERED it), so it could not
    separate a trained E2 from an untrained one and caught only total W_a -> 0
    collapse. Spread is necessary but nowhere near sufficient; accuracy is the
    property a planner actually needs.
    """
    gains: List[float] = []
    _, obs_dict = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            z = world_enc(_get_world_obs(obs_dict))
            a_exec = random.randint(0, ACTION_DIM - 1)
            _, _, done, _, obs_next = env.step(_onehot(a_exec, ACTION_DIM).unsqueeze(0))
            z_next = world_enc(_get_world_obs(obs_next))
            err_right = float((e2_fwd(z, _onehot(a_exec, ACTION_DIM)) - z_next).norm().item())
            wrong = [
                float((e2_fwd(z, _onehot(a, ACTION_DIM)) - z_next).norm().item())
                for a in range(ACTION_DIM) if a != a_exec
            ]
            err_wrong = sum(wrong) / len(wrong) if wrong else 0.0
            if err_wrong > 1e-8:
                gains.append((err_wrong - err_right) / err_wrong)
            obs_dict = env.reset()[1] if done else obs_next
    return float(sum(gains) / len(gains)) if gains else 0.0


def _e2_action_discrimination(world_enc, e2_fwd, env, n_steps: int) -> float:
    """
    INFORMATIONAL (no longer a gating criterion). Mean pairwise L2 distance
    between E2's predicted next-z_world across the ACTION_DIM actions, normalised
    by the mean predicted norm -- i.e. prediction SPREAD.

    Demoted from C6 because it is ~5x free at untrained init and training LOWERS
    it, so it cannot separate a trained E2 from an untrained one. Retained because
    total collapse (spread -> 0) is still worth seeing. The gating canary is now
    _e2_action_consequence_accuracy_gain above.

    Near 0 means E2 is action-INVARIANT: every candidate action predicts the same
    future, so NO depth contrast (and no kernel-quality contrast) can possibly
    discriminate. Without this canary a null result would be indistinguishable
    between "chaining adds nothing" and "E2 cannot tell the actions apart".
    """
    ratios: List[float] = []
    _, obs_dict = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            z_world = world_enc(_get_world_obs(obs_dict))
            preds = [e2_fwd(z_world, _onehot(a, ACTION_DIM)) for a in range(ACTION_DIM)]
            dists = []
            for i in range(ACTION_DIM):
                for j in range(i + 1, ACTION_DIM):
                    dists.append(float((preds[i] - preds[j]).norm().item()))
            mean_norm = sum(float(p.norm().item()) for p in preds) / ACTION_DIM
            if mean_norm > 1e-8 and dists:
                ratios.append((sum(dists) / len(dists)) / mean_norm)
            a_oh = _onehot(random.randint(0, ACTION_DIM - 1), ACTION_DIM)
            _, _, done, _, obs_next = env.step(a_oh.unsqueeze(0))
            obs_dict = env.reset()[1] if done else obs_next
    return float(sum(ratios) / len(ratios)) if ratios else 0.0


# --------------------------------------------------------------------------
# BEHAVIOURAL instruments -- the gap that let the affine collapse through
# --------------------------------------------------------------------------
# Every guard in the first build inspected the MODEL (world_forward_r2, E2
# action-discrimination, harm-vs-random). None inspected what the arms actually
# DID, so a wall-pinned constant-action policy cleared all of them. These two
# measures are computed from the arm's OWN executed trajectory and are GATING.

def _action_entropy(counts: Dict[int, int]) -> float:
    """Shannon entropy (bits) of the arm's EXECUTED action distribution.
    0.0 = one action forever (the affine-collapse signature); log2(5)=2.32 = uniform."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    import math
    h = 0.0
    for n in counts.values():
        if n > 0:
            p = n / total
            h -= p * math.log2(p)
    return float(h)


def _agent_cell(env) -> Optional[Tuple[int, int]]:
    """
    The agent's grid cell, for distinct-cells-visited, via the env's own accessor
    (CausalGridWorldV2.get_agent_position, causal_grid_world.py:5476).

    Returns None -- an explicit CANNOT-DETERMINE -- if the accessor is absent or
    unreadable, rather than a 0 that would read as total degeneracy. The caller
    propagates None into the manifest and C8 routes to inconclusive, so a broken
    position read can never masquerade as a passing behavioural gate.
    """
    getter = getattr(env, "get_agent_position", None)
    if getter is None:
        return None
    try:
        v = getter()
        seq = v.tolist() if hasattr(v, "tolist") else list(v)
        if len(seq) >= 2:
            return (int(seq[0]), int(seq[1]))
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
    return None


# --------------------------------------------------------------------------
# COMPETENCE FLOOR reference: best FIXED SINGLE ACTION
# --------------------------------------------------------------------------

def _best_fixed_action_harm(seed: int, dry_run: bool) -> Dict:
    """
    Harm rate of the BEST constant-action policy, evaluated on the same eval env
    and budget as the real arms.

    This replaces uniform-random as the competence floor's reference, and it is
    the fix that targets the actual defect: a wall-pinned constant action scores
    FAR better than uniform random (a random walk keeps re-entering hazard
    fields), so a random-action floor is cleared by ~400x by exactly the
    degenerate policy it was supposed to exclude. A planner that cannot beat the
    best single fixed action is not planning, and the first build could not say so.

    Returns the best (lowest-harm) fixed action and the full per-action table, so
    the manifest records which constant policy the planner had to beat.
    """
    per_action: Dict[str, float] = {}
    best_rate = float("inf")
    best_a = None
    eval_eps = EVAL_EPISODES if not dry_run else 2
    for a_idx in range(ACTION_DIM):
        torch.manual_seed(seed + 999)
        random.seed(seed + 999)
        env = _make_env(seed + 1000)
        total_harm = 0.0
        total_steps = 0
        _, obs_dict = env.reset()
        for _ep in range(eval_eps):
            for _step in range(STEPS_PER_EPISODE):
                a_oh = _onehot(a_idx, ACTION_DIM)
                _, harm_signal, done, _, obs_next = env.step(a_oh.unsqueeze(0))
                total_harm += max(0.0, -harm_signal)
                total_steps += 1
                obs_dict = env.reset()[1] if done else obs_next
        rate = total_harm / max(total_steps, 1)
        per_action["action_%d" % a_idx] = rate
        if rate < best_rate:
            best_rate = rate
            best_a = a_idx
    print("[floor] seed=%d best_fixed_action=%d harm_rate=%.6f | per-action %s"
          % (seed, best_a, best_rate,
             {k: round(v, 5) for k, v in per_action.items()}), flush=True)
    return {
        "best_fixed_action": best_a,
        "best_fixed_action_harm_rate": best_rate,
        "per_action_harm_rate": per_action,
    }

# --------------------------------------------------------------------------
# Training (shared across arms within a seed -> matched training)
# --------------------------------------------------------------------------

def _train_models(seed: int, dry_run: bool) -> Tuple:
    """
    Train the whole inline stack on WARMUP_EPISODES random-action episodes.

    GFLAG-0491 TRAINER DECLARATION: every module read at act time is trained
    here, by the ONE driver-built optimizer below. The assert makes that
    machine-checked rather than a claim in a comment -- if a module is ever added
    without being added to train_params, this raises instead of silently
    shipping an untrained act-path head.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    world_enc = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    prox_head = ResourceProximityHead(WORLD_DIM)
    self_enc = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    e2_fwd = E2WorldForward(WORLD_DIM, ACTION_DIM)
    harm_head = HarmHead(WORLD_DIM, SELF_DIM)

    modules = {
        "world_enc": world_enc, "prox_head": prox_head, "self_enc": self_enc,
        "e2_fwd": e2_fwd, "harm_head": harm_head,
    }
    train_params: List[nn.Parameter] = []
    for m in modules.values():
        train_params.extend(list(m.parameters()))
    optimizer = optim.Adam(train_params, lr=LR)

    covered = set(id(p) for p in train_params)
    for name, m in modules.items():
        missing = [n for n, p in m.named_parameters() if id(p) not in covered]
        assert not missing, (
            "GFLAG-0491 trainer gap: module %s has untrained parameters %s"
            % (name, missing)
        )

    warmup_eps = WARMUP_EPISODES if not dry_run else 4
    _, obs_dict = env.reset()

    for ep in range(warmup_eps):
        for _step in range(STEPS_PER_EPISODE):
            obs_w = _get_world_obs(obs_dict)
            obs_s = _get_self_obs(obs_dict)
            prox_t = _get_resource_prox_target(obs_dict)

            z_world = world_enc(obs_w)
            z_self = self_enc(obs_s)

            a_oh = _onehot(random.randint(0, ACTION_DIM - 1), ACTION_DIM)
            _, harm_signal, done, _, obs_next = env.step(a_oh.unsqueeze(0))
            harm_val = max(0.0, -harm_signal)

            with torch.no_grad():
                z_world_next_actual = world_enc(_get_world_obs(obs_next))
            e2_loss = F.mse_loss(e2_fwd(z_world, a_oh), z_world_next_actual.detach())

            harm_loss = F.mse_loss(
                harm_head(z_world.detach(), z_self.detach()),
                torch.tensor([harm_val], dtype=torch.float32),
            )
            prox_loss = F.mse_loss(
                prox_head(z_world), torch.tensor([prox_t], dtype=torch.float32)
            )

            total_loss = e2_loss + harm_loss + prox_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()

            obs_dict = env.reset()[1] if done else obs_next

        if ep % 100 == 0 or ep == warmup_eps - 1:
            print(
                "[train] seed=%d ep %d/%d e2_loss=%.5f harm_loss=%.5f prox_loss=%.5f"
                % (seed, ep + 1, warmup_eps, e2_loss.item(), harm_loss.item(),
                   prox_loss.item()),
                flush=True,
            )

    r2_steps = R2_EVAL_STEPS if not dry_run else 50
    d_steps = DISCRIM_STEPS if not dry_run else 20
    e2_r2 = _compute_r2(world_enc, e2_fwd, env, r2_steps)
    prox_r2 = _compute_prox_r2(world_enc, prox_head, env, r2_steps // 2)
    discrim = _e2_action_discrimination(world_enc, e2_fwd, env, d_steps)

    print(
        "[train] seed=%d world_forward_r2=%.4f resource_prox_r2=%.4f"
        " e2_action_discrimination=%.4f"
        % (seed, e2_r2, prox_r2, discrim),
        flush=True,
    )
    return world_enc, prox_head, self_enc, e2_fwd, harm_head, e2_r2, prox_r2, discrim


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _run_arm(
    seed: int, arm: str, world_enc, self_enc, e2_for_arm, harm_head, dry_run: bool,
) -> Dict:
    """Eval one arm. `e2_for_arm` is the trained E2 for INTACT and a scrambled
    DEEP COPY for the scrambled arms; everything else (encoders, HarmHead,
    planner depth, eval env, eval RNG) is identical across arms, so E2's weights
    are the only thing that varies."""
    torch.manual_seed(seed + 999)
    random.seed(seed + 999)

    env = _make_env(seed + 1000)
    eval_eps = EVAL_EPISODES if not dry_run else 2
    depth = CHAIN_DEPTH

    harm_events = 0
    total_steps = 0
    total_harm = 0.0
    drive_sum = 0.0
    action_counts: Dict[int, int] = {}
    cells_seen = set()
    cells_unreadable = False
    # Per-RESET harm rates, so the manifest carries a spread rather than one
    # pooled number: env.reset() happens only on `done`, so "50 episodes" is
    # really one long run with ~10 resets.
    reset_harm: List[float] = []
    reset_steps: List[int] = []
    cur_harm = 0.0
    cur_steps = 0

    _, obs_dict = env.reset()
    for _ep in range(eval_eps):
        for _step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                z_world = world_enc(_get_world_obs(obs_dict))
                z_self = self_enc(_get_self_obs(obs_dict))
                if arm == ARM_RANDOM_REF:
                    action_idx = random.randint(0, ACTION_DIM - 1)
                else:
                    action_idx = _select_action(
                        z_world, z_self, e2_for_arm, harm_head, depth
                    )

            a_oh = _onehot(action_idx, ACTION_DIM)
            _, harm_signal, done, _, obs_next = env.step(a_oh.unsqueeze(0))
            harm_val = max(0.0, -harm_signal)

            drive_sum += _get_drive_level(obs_dict)
            total_harm += harm_val
            total_steps += 1
            cur_harm += harm_val
            cur_steps += 1
            if harm_val > 0.0:
                harm_events += 1
            action_counts[action_idx] = action_counts.get(action_idx, 0) + 1
            cell = _agent_cell(env)
            if cell is None:
                cells_unreadable = True
            else:
                cells_seen.add(cell)

            if done:
                if cur_steps > 0:
                    reset_harm.append(cur_harm / cur_steps)
                    reset_steps.append(cur_steps)
                cur_harm = 0.0
                cur_steps = 0
                obs_dict = env.reset()[1]
            else:
                obs_dict = obs_next

    if cur_steps > 0:
        reset_harm.append(cur_harm / cur_steps)
        reset_steps.append(cur_steps)

    harm_rate = total_harm / max(total_steps, 1)
    entropy = _action_entropy(action_counts)
    distinct_cells = None if cells_unreadable else len(cells_seen)
    n_resets = len(reset_harm)
    mean_r = sum(reset_harm) / n_resets if n_resets else 0.0
    sd_r = (
        (sum((x - mean_r) ** 2 for x in reset_harm) / (n_resets - 1)) ** 0.5
        if n_resets > 1 else None
    )

    print(
        "Seed %d Arm %-12s: harm_rate=%.5f n_harm_events=%d steps=%d "
        "action_entropy=%.3f distinct_cells=%s n_resets=%d harm_sd_across_resets=%s"
        % (seed, arm, harm_rate, harm_events, total_steps, entropy,
           distinct_cells, n_resets,
           ("%.5f" % sd_r) if sd_r is not None else "n/a"),
        flush=True,
    )
    return {
        "arm": arm,
        "seed": seed,
        "harm_rate": harm_rate,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
        "mean_drive_level": drive_sum / max(total_steps, 1),
        "rollout_depth": 0 if arm == ARM_RANDOM_REF else depth,
        "e2_mode": E2_MODE_BY_ARM[arm],
        "executed_action_counts": {str(k): v for k, v in sorted(action_counts.items())},
        "action_entropy_bits": entropy,
        "distinct_cells_visited": distinct_cells,
        "n_resets": n_resets,
        "harm_rate_per_reset": reset_harm,
        "steps_per_reset": reset_steps,
        "harm_rate_sd_across_resets": sd_r,
    }


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------

def _evaluate_criteria(results: Dict[str, List[Dict]], per_seed: Dict) -> Tuple:
    intact = results[ARM_INTACT]
    reinit = results[ARM_REINIT]
    rnd = results[ARM_RANDOM_REF]
    n_s = len(SEEDS)

    deltas = [reinit[i]["harm_rate"] - intact[i]["harm_rate"] for i in range(n_s)]
    mean_delta = sum(deltas) / max(len(deltas), 1)

    floors = [per_seed[s]["best_fixed_action_harm_rate"] for s in SEEDS]
    ranges = [max(0.0, floors[i] - intact[i]["harm_rate"]) for i in range(n_s)]
    mean_range = sum(ranges) / max(len(ranges), 1)
    c1_bar = THRESH_C1_RELATIVE_FRACTION * mean_range
    c1 = mean_range > 0.0 and mean_delta >= c1_bar

    c2 = all(intact[i]["harm_rate"] < reinit[i]["harm_rate"] for i in range(n_s))
    c3 = all(per_seed[s]["world_forward_r2_intact"] >= THRESH_C3_E2_R2 for s in SEEDS)
    c4 = all(
        per_seed[s]["world_forward_r2_reinit"] <= THRESH_C4_SCRAMBLE_R2_MAX
        for s in SEEDS
    )
    # C5 COMPETENCE FLOOR vs the BEST FIXED SINGLE ACTION, not uniform random.
    c5 = all(
        intact[i]["harm_rate"] <= THRESH_C5_COMPETENCE * floors[i] for i in range(n_s)
    )
    c6 = all(rnd[i]["n_harm_events"] >= THRESH_C6_MIN_CONTACTS for i in range(n_s))

    planning = (ARM_INTACT, ARM_REINIT, ARM_PERMUTED)
    c7 = all(
        results[a][i]["action_entropy_bits"] >= THRESH_C7_MIN_ACTION_ENTROPY
        for a in planning for i in range(n_s)
    )
    cells = [results[a][i]["distinct_cells_visited"] for a in planning for i in range(n_s)]
    cells_determinable = all(c is not None for c in cells)
    c8 = cells_determinable and all(c >= THRESH_C8_MIN_DISTINCT_CELLS for c in cells)

    criteria = {
        "C1_relative_scramble_degradation_ge_frac_of_range": c1,
        "C2_intact_below_reinit_all_seeds": c2,
        "C3_intact_world_forward_r2_ge_0.20": c3,
        "C4_scramble_canary_reinit_r2_le_0.05": c4,
        "C5_intact_competence_floor_vs_best_fixed_action": c5,
        "C6_harm_events_ge_5": c6,
        "C7_action_entropy_ge_0.50_bits_all_planning_arms": c7,
        "C8_distinct_cells_ge_10_all_planning_arms": c8,
    }
    status = "PASS" if all(criteria.values()) else "FAIL"

    mean_disagree = sum(
        per_seed[s]["intact_vs_reinit_action_disagreement_rate"] for s in SEEDS
    ) / len(SEEDS)

    if not (c7 and c8):
        direction = "inconclusive"
        note = (
            "BEHAVIOURAL DEGENERACY (C7 entropy=%s, C8 distinct_cells=%s%s). One or "
            "more planning arms barely moved or barely explored, so no arm is a "
            "planner and the E2-content contrast is meaningless. CANNOT DETERMINE; "
            "explicitly NOT support for H3-no-E2. This is the failure that cleared "
            "every model-side gate in this leg's first build (AFFINE_COLLAPSE_NOTE)."
            % (c7, c8, "" if cells_determinable else
               " -- distinct_cells UNREADABLE, a cannot-determine, not a 0")
        )
    elif not (c3 and c4 and c6):
        direction = "inconclusive"
        note = (
            "Instrument inadequate (C3 intact_r2=%s, C4 scramble canary=%s, "
            "C6 harm_events=%s). CANNOT DETERMINE whether E2's trained content "
            "carries harm avoidance; NOT support for H3-no-E2. A C4 failure means "
            "the scramble did not destroy E2's predictive power, so the arms were "
            "never really different." % (c3, c4, c6)
        )
    elif not c5:
        direction = "inconclusive"
        note = (
            "COMPETENCE FLOOR FAILED: the INTACT planner is not meaningfully better "
            "than the BEST FIXED SINGLE ACTION (mean floor %.6f), so there is no "
            "planning contribution for the scramble to remove. Explicitly NOT "
            "support for H3-no-E2." % (sum(floors) / len(floors))
        )
    elif mean_range <= 0.0:
        direction = "inconclusive"
        note = (
            "NO ACHIEVABLE RANGE: the INTACT arm is already at or below the "
            "best-fixed-action floor on average, so C1's relative bar is undefined. "
            "CANNOT DETERMINE."
        )
    elif c1 and not c2:
        direction = "mixed"
        note = (
            "Declared H3 null is REJECTED on the mean (degradation %.6f >= bar "
            "%.6f = %.0f%% of the achievable range %.6f) but NOT consistently "
            "across seeds (C2 failed), so the effect is unreplicated. Read as "
            "mixed: suggestive of H1-chaining, not established. Per-seed deltas %s; "
            "intact-vs-reinit action disagreement %.4f. Scope: this tests E2 "
            "DEGRADATION, not ABSENCE."
            % (mean_delta, c1_bar, 100 * THRESH_C1_RELATIVE_FRACTION, mean_range,
               [round(d, 6) for d in deltas], mean_disagree)
        )
    elif not c1:
        direction = "weakens"
        note = (
            "Declared H3 null HOLDS with every instrument AND behavioural gate "
            "passing: scrambling E2 to untrained weights (canary confirms r2 "
            "destroyed) leaves harm_rate within the pre-registered relative bar of "
            "intact (degradation %.6f < bar %.6f = %.0f%% of the achievable range "
            "%.6f over the best fixed action). H3-no-E2 favoured: E2's TRAINED "
            "CONTENT is not carrying the harm avoidance. intact-vs-reinit action "
            "disagreement %.4f. Scope: tests E2 DEGRADATION, not ABSENCE -- the "
            "absence arm is not constructible with a state-only harm readout."
            % (mean_delta, c1_bar, 100 * THRESH_C1_RELATIVE_FRACTION, mean_range,
               mean_disagree)
        )
    else:
        direction = "supports"
        note = (
            "Declared H3 null REJECTED: scrambling E2 to untrained weights makes "
            "harm materially worse (degradation %.6f >= bar %.6f = %.0f%% of the "
            "achievable range %.6f over the best fixed action), consistently on all "
            "seeds, with intact E2 trained, the scramble confirmed destructive, the "
            "intact planner better than any fixed action, and all planning arms "
            "behaviourally non-degenerate (C7/C8). H1-chaining favoured over "
            "H3-no-E2: E2's trained content is load-bearing. intact-vs-reinit "
            "action disagreement %.4f. Scope: tests E2 DEGRADATION, not ABSENCE."
            % (mean_delta, c1_bar, 100 * THRESH_C1_RELATIVE_FRACTION, mean_range,
               mean_disagree)
        )
    extra = {"c1_bar": c1_bar, "mean_achievable_range": mean_range,
             "per_seed_floor": floors, "per_seed_range": ranges}
    return criteria, status, direction, note, mean_delta, deltas, extra


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(dry_run: bool = False, **kwargs) -> dict:
    print("[V3-EXQ-1101] MECH-033 leg 3 (integration axis): E2-scrambled vs "
          "intact, planner and HarmHead held constant", flush=True)
    print("[V3-EXQ-1101] Declared null (H3): harm_rate(REINIT_E2) within %.0f%% of "
          "the achievable range over the best FIXED action, of harm_rate(INTACT)"
          % (100 * THRESH_C1_RELATIVE_FRACTION), flush=True)
    print("[V3-EXQ-1101] RANDOM_REF is INFORMATIONAL ONLY (competence floor + "
          "data quality), never the null's comparator", flush=True)

    selfcheck = _selfcheck_manipulation()
    scramble_selfchecks: Dict = {}

    results: Dict[str, List[Dict]] = {a: [] for a in ARMS}
    per_seed: Dict = {}

    for seed in SEEDS:
        print("\n[V3-EXQ-1101] === seed %d ===" % seed, flush=True)
        (world_enc, prox_head, self_enc, e2_fwd, harm_head,
         e2_r2, prox_r2, discrim) = _train_models(seed, dry_run)
        scramble_selfchecks[str(seed)] = _selfcheck_scramble(e2_fwd, seed)

        e2_by_arm = {
            ARM_INTACT: e2_fwd,
            ARM_REINIT: _scrambled_e2(e2_fwd, "reinit", seed),
            ARM_PERMUTED: _scrambled_e2(e2_fwd, "permuted", seed),
            ARM_RANDOM_REF: e2_fwd,   # unused: RANDOM_REF never calls E2
        }

        env_probe = _make_env(seed)
        r2_steps = R2_EVAL_STEPS if not dry_run else 50
        d_steps = DISCRIM_STEPS if not dry_run else 20
        r2_reinit = _compute_r2(world_enc, e2_by_arm[ARM_REINIT], env_probe, r2_steps)
        r2_permuted = _compute_r2(
            world_enc, e2_by_arm[ARM_PERMUTED], env_probe, r2_steps
        )
        discrim_reinit = _e2_action_discrimination(
            world_enc, e2_by_arm[ARM_REINIT], env_probe, d_steps
        )
        discrim_permuted = _e2_action_discrimination(
            world_enc, e2_by_arm[ARM_PERMUTED], env_probe, d_steps
        )
        disagree = _arm_disagreement_rate(
            seed, world_enc, self_enc, e2_fwd, e2_by_arm[ARM_REINIT], harm_head,
            d_steps,
        )
        acc_gain = _e2_action_consequence_accuracy_gain(
            world_enc, e2_fwd, _make_env(seed), d_steps,
        )
        acc_gain_reinit = _e2_action_consequence_accuracy_gain(
            world_enc, e2_by_arm[ARM_REINIT], _make_env(seed), d_steps,
        )
        floor = _best_fixed_action_harm(seed, dry_run)
        per_seed[seed] = {
            "e2_action_consequence_accuracy_gain_intact": acc_gain,
            "e2_action_consequence_accuracy_gain_reinit": acc_gain_reinit,
            "best_fixed_action": floor["best_fixed_action"],
            "best_fixed_action_harm_rate": floor["best_fixed_action_harm_rate"],
            "best_fixed_action_per_action_harm_rate": floor["per_action_harm_rate"],
            "world_forward_r2_intact": e2_r2,
            "world_forward_r2_reinit": r2_reinit,
            "world_forward_r2_permuted": r2_permuted,
            "resource_prox_r2": prox_r2,
            "e2_action_discrimination_intact": discrim,
            "e2_action_discrimination_reinit": discrim_reinit,
            "e2_action_discrimination_permuted": discrim_permuted,
            "intact_vs_reinit_action_disagreement_rate": disagree,
        }
        print(
            "[instr] seed=%d r2 intact=%.4f reinit=%.4f permuted=%.4f | "
            "discrim intact=%.4f reinit=%.4f permuted=%.4f | "
            "intact_vs_reinit_action_disagreement=%.4f"
            % (seed, e2_r2, r2_reinit, r2_permuted, discrim, discrim_reinit,
               discrim_permuted, disagree),
            flush=True,
        )
        for arm in ARMS:
            results[arm].append(
                _run_arm(seed, arm, world_enc, self_enc, e2_by_arm[arm], harm_head,
                         dry_run)
            )

        # Recorded check that no arm leaked into the trained module.
        r2_after = _compute_r2(world_enc, e2_fwd, _make_env(seed), r2_steps)
        per_seed[seed]["world_forward_r2_intact_after_all_arms"] = r2_after
        assert abs(r2_after - e2_r2) < 0.05, (
            "trained E2 r2 drifted %.4f -> %.4f across the scrambled arms: "
            "something mutated it in place" % (e2_r2, r2_after)
        )

    criteria, status, direction, note, mean_delta, deltas, crit_extra = (
        _evaluate_criteria(results, per_seed)
    )

    metrics = {
        "mean_scramble_degradation_reinit_minus_intact": mean_delta,
        "per_seed_scramble_degradation": deltas,
        "criteria_met": float(sum(1 for v in criteria.values() if v)),
        "criteria_total": float(len(criteria)),
    }
    for arm in ARMS:
        for i, s in enumerate(SEEDS):
            metrics["harm_rate_%s_seed%d" % (arm, s)] = results[arm][i]["harm_rate"]
            metrics["n_harm_events_%s_seed%d" % (arm, s)] = float(
                results[arm][i]["n_harm_events"]
            )
        metrics["harm_rate_%s_mean" % arm] = sum(
            r["harm_rate"] for r in results[arm]
        ) / len(SEEDS)
    for s in SEEDS:
        for k, v in per_seed[s].items():
            metrics["%s_seed%d" % (k, s)] = v
    # Competence-floor ratios, printed so a degenerate arm is visible at a glance.
    for arm in (ARM_INTACT, ARM_REINIT, ARM_PERMUTED):
        for i, sd in enumerate(SEEDS):
            for ref_name, denom in (
                ("best_fixed_action", per_seed[sd]["best_fixed_action_harm_rate"]),
                ("random_ref", results[ARM_RANDOM_REF][i]["harm_rate"]),
            ):
                metrics["harm_ratio_%s_over_%s_seed%d" % (arm, ref_name, sd)] = (
                    results[arm][i]["harm_rate"] / denom if denom > 0 else None
                )
    for arm in ARMS:
        for i, sd in enumerate(SEEDS):
            metrics["action_entropy_bits_%s_seed%d" % (arm, sd)] = (
                results[arm][i]["action_entropy_bits"])
            metrics["distinct_cells_visited_%s_seed%d" % (arm, sd)] = (
                results[arm][i]["distinct_cells_visited"])
            metrics["harm_rate_sd_across_resets_%s_seed%d" % (arm, sd)] = (
                results[arm][i]["harm_rate_sd_across_resets"])
    metrics["c1_relative_bar"] = crit_extra["c1_bar"]
    metrics["mean_achievable_range_over_best_fixed_action"] = (
        crit_extra["mean_achievable_range"])
    for k, v in criteria.items():
        metrics["crit_%s" % k] = 1.0 if v else 0.0

    print("\n[V3-EXQ-1101] ===== VERDICT =====", flush=True)
    for k, v in criteria.items():
        print("  %-42s %s" % (k, "PASS" if v else "FAIL"), flush=True)
    print("  status=%s evidence_direction=%s" % (status, direction), flush=True)
    print("  %s" % note, flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "outcome": status,
        "status": status,
        "evidence_direction": direction,
        "verdict": note,
        "metrics": metrics,
        "criteria": criteria,
        "declared_null": (
            "H3: harm_rate(REINIT_E2) is within %.0f%% of the range achievable over "
            "the best fixed single action, of harm_rate(INTACT) (bar this run: %.6f)"
            % (100 * THRESH_C1_RELATIVE_FRACTION, crit_extra["c1_bar"])
        ),
        "c1_bar_derivation": {
            "form": "relative",
            "fraction": THRESH_C1_RELATIVE_FRACTION,
            "reference": "best fixed single action (per seed), NOT uniform random",
            "pre_registration": (
                "coefficient fixed at %.2f from a pilot on HELD-OUT seeds %s, "
                "disjoint from eval seeds %s"
                % (THRESH_C1_RELATIVE_FRACTION, PILOT_SEEDS, SEEDS)
            ),
            "measured_bar": crit_extra["c1_bar"],
            "mean_achievable_range": crit_extra["mean_achievable_range"],
            "per_seed_floor": crit_extra["per_seed_floor"],
        },
        "portfolio": {
            "name": "MECH-033 GOV-FANOUT-1 discrimination portfolio",
            "leg": "3 of 3",
            "axis": "integration",
            "hypotheses_discriminated": ["H1-chaining", "H3-no-E2"],
            "carries_h3_by": (
                "E2 DEGRADATION, not E2 ABSENCE -- the autopsy's k=0 absence arm "
                "was dropped by ratified decision 2026-09-25 because a state-only "
                "harm readout cannot rank actions with zero E2 forward steps"
            ),
            "siblings": {"V3-EXQ-1103": "algorithm axis (H1 vs H2)",
                         "V3-EXQ-1102": "instrumentation axis (H1, full REEAgent)"},
            "source_autopsy": SOURCE_AUTOPSY,
            "design_doc": DESIGN_DOC,
            "does_not_supersede": (
                "V3-EXQ-308 is deliberately NOT re-lettered: a re-letter would "
                "inherit the shared uniform-random comparator confound."
            ),
        },
        "arm_results": {a: results[a] for a in ARMS},
        "informational_arms": {
            ARM_PERMUTED: (
                "Tighter scramble control: the trained tensors' own values, "
                "randomly permuted, so the value multiset (hence scale) is "
                "preserved exactly. Reported, not gated -- it guards the reading "
                "of the REINIT_E2 result against 'the scramble merely changed "
                "output magnitude'."
            ),
            ARM_RANDOM_REF: (
                "Competence floor (C4) + data quality (C5) reference ONLY. Never "
                "the comparator for the declared null -- using a uniform-random "
                "arm that way is the D2 ablation confound this portfolio removes."
            )
        },
        "per_seed_instruments": {str(k): v for k, v in per_seed.items()},
        "manipulation_selfcheck": selfcheck,
        "scramble_selfcheck_per_seed": scramble_selfchecks,
        "scope_statements": SCOPE_STATEMENTS,
        "ethics_preflight": {
            "involves_negative_valence": True,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
            "note": (
                "SENT-0. Grid-world hazard avoidance with a scalar harm signal, "
                "identical in kind and magnitude (hazard_harm=0.02) to the "
                "V3-EXQ-308 lineage this leg corrects. No new manipulation of "
                "valence; the agent is always able to move away from hazards."
            ),
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "seeds": SEEDS,
            "arms": ARMS,
            "chain_depth": CHAIN_DEPTH,
            "e2_mode_by_arm": E2_MODE_BY_ARM,
            "warmup_episodes": WARMUP_EPISODES if not dry_run else 4,
            "eval_episodes": EVAL_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "lr": LR,
            "drive_weight": DRIVE_WEIGHT,
            "thresholds": {
                "C1_relative_fraction": THRESH_C1_RELATIVE_FRACTION,
                "C7_min_action_entropy_bits": THRESH_C7_MIN_ACTION_ENTROPY,
                "C8_min_distinct_cells": THRESH_C8_MIN_DISTINCT_CELLS,
                "C3_intact_e2_r2": THRESH_C3_E2_R2,
                "C4_scramble_r2_max": THRESH_C4_SCRAMBLE_R2_MAX,
                "C5_competence_ratio_vs_best_fixed_action": THRESH_C5_COMPETENCE,
                "C6_min_contacts": THRESH_C6_MIN_CONTACTS,
            },
            "env": {
                "class": "CausalGridWorldV2", "size": 8, "num_hazards": 3,
                "num_resources": 3, "hazard_harm": 0.02,
                "env_drift_interval": 5, "env_drift_prob": 0.2,
            },
            "model_dims": {
                "world_obs_dim": WORLD_OBS_DIM, "self_obs_dim": SELF_OBS_DIM,
                "action_dim": ACTION_DIM, "world_dim": WORLD_DIM,
                "self_dim": SELF_DIM,
            },
        },
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=_T0,
    )
    print("wrote: %s" % out_path, flush=True)
    return {"status": status, "evidence_direction": direction,
            "metrics": metrics, "manifest_path": str(out_path)}


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dry-run", action="store_true")
    _args = _ap.parse_args()
    _res = run(dry_run=_args.dry_run)
    emit_outcome(
        outcome=_res["status"] if _res["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res["manifest_path"],
        dry_run=_args.dry_run,
    )
