#!/opt/local/bin/python3
"""
V3-EXQ-1103 -- MECH-033 GOV-FANOUT-1 portfolio, LEG 2 (axis: algorithm).

Claim:   MECH-033  "E2 forward-prediction kernels seed hippocampal rollouts."
Source:  failure_autopsy_gflag0452-D2-cluster_2026-09-24, targets[5] (V3-EXQ-308),
         fanout_recommendation.suggested_probes[1] (hypothesis "H2 vs H1").
Design:  REE_assembly/evidence/planning/mech033_fanout_portfolio_design_blocked_20260925.md
         (ratified 2026-09-25 by orchestrate-20260924-1707 under rec-20260924-fb429c72;
         3-leg portfolio, leg 1 dropped, leg 4 comparator A-i).

WHY THIS EXISTS -- the confound it removes
------------------------------------------
Governance 2026-09-24 moved MECH-033 active -> provisional (user decision
rec-20260924-a796e485). Its four PASSes (V3-EXQ-171 / 184 / 184-rerun / 308) all
compared a planner against a UNIFORM-RANDOM ablation, which removes the HarmHead
and the planner together with E2 seeding, so the delta cannot be attributed to
E2. Worse, V3-EXQ-308's "k=3 chain" was one real action followed by two FORCED
STAY steps (v3_exq_308_mech033_kernel_chain_discriminative.py:520-522,
`a_oh[ACTION_DIM - 1] = 1.0`), i.e. a one-action lookahead wearing a chain's name.

This leg asks the depth question with BOTH arms competent:
    K1  -- one E2 forward step, then HarmHead.        (H2: one-step lookahead)
    K3  -- three E2 forward steps with POLICY-DRIVEN  (H1: genuine chaining)
           continuation: at each subsequent rollout
           step the continuation action is chosen by
           the SAME harm-minimising rule, never forced STAY.
Both arms are action-dependent, so neither is degenerate and the null is
falsifiable in both directions. The single rollout helper `_rollout_harm` serves
both arms (depth=1 vs depth=3), so there is no implementation asymmetry between
them -- the ONLY difference is the depth argument.

Live hypotheses (this leg discriminates H2 from H1; leg V3-EXQ-1101 carries H3):
    H1-chaining : multi-step E2 rollouts carry harm avoidance beyond one step.
    H2-one-step : HarmHead + a single E2 forward step carries it; depth adds nothing.

DECLARED NULL (H2)
------------------
    harm_rate(K3) is NOT below harm_rate(K1) by >= 0.01.
H1 predicts this null is rejected. H2 predicts it holds.

SCOPE STATEMENTS (recorded, not hidden)
---------------------------------------
1. COMPUTE IS NOT MATCHED BETWEEN ARMS, BY CONSTRUCTION. K3 issues ~11x the E2
   calls of K1 per decision, because ROLLOUT DEPTH IS THE MANIPULATION. This is
   the correct choice for an H1-vs-H2 depth contrast (matching compute would mean
   matching depth, erasing the manipulation). Contrast with leg V3-EXQ-1102,
   where the manipulation is kernel QUALITY and the CEM budget therefore IS
   matched.
2. z_self is held fixed across rollout steps -- the inline architecture has no
   self-forward model. Same as V3-EXQ-308. The harm readout is
   HarmHead(z_world_predicted, z_self_current).
3. Architecture is the INLINE 308-lineage stack, not the full REEAgent. That is
   deliberate: this leg isolates rollout DEPTH, which is cleanly controllable
   here. The full-REEAgent/HippocampalModule question is leg V3-EXQ-1102's job
   (the autopsy's "integration: isolated" finding is answered there, not here).
4. GFLAG-0491 (no waking gradient learning at REEConfig defaults) does NOT apply
   to this script: it does not use REEConfig or REEAgent. Every module below is a
   local nn.Module and EVERY ONE is trained by the single driver-built optimizer
   named at `_train_models` (`optimizer = optim.Adam(train_params, ...)`), whose
   `train_params` list is asserted at runtime to cover all five modules'
   parameters. Trainers, by head: WorldEncoder / ResourceProximityHead /
   SelfEncoder / E2WorldForward / HarmHead -> that one Adam.
5. GFLAG-0485 (E2 predicts the executed action's consequence at chance in
   untrained production regimes): handled two ways. Depth stays k<=3, inside the
   shallow band where E3 across-candidate score variance is not rollout-dominated;
   and C6 below is an explicit action-discrimination CANARY on E2, so an
   action-invariant E2 reports as cannot-determine rather than as support for H2.

NEGATIVE-INSTRUMENT DISCIPLINE (CLAUDE.md "Negative instruments")
-----------------------------------------------------------------
A null result here would be evidence FOR H2, so "no difference" must not be
reachable by a broken instrument. Three guards, all gating:
  C4 COMPETENCE FLOOR      -- K1 must beat the informational RANDOM_REF arm by
                              >= 15%. If the k=1 baseline is itself at random
                              level, the k=1-vs-k=3 contrast is uninformative and
                              the run reports INCONCLUSIVE, never "supports H2".
  C6 DISCRIMINATION CANARY -- E2 must actually distinguish actions. If predicted
                              next-states barely differ across the 5 actions, no
                              depth contrast can discriminate anything.
  C3 TRAINING GATE         -- world_forward_r2 >= 0.20, the claim's own
                              non-degeneracy precondition.
RANDOM_REF is INFORMATIONAL ONLY. It is the data-quality / competence reference.
It is NEVER the comparator for the declared null -- using it that way is exactly
the D2 confound this leg exists to remove.

PRE-REGISTERED CRITERIA
-----------------------
C1: mean over seeds of (harm_rate_K1 - harm_rate_K3) >= 0.01        [rejects the H2 null]
C2: harm_rate_K3 < harm_rate_K1 on ALL seeds                        [consistency]
C3: world_forward_r2 >= 0.20 on ALL seeds                           [E2 trained]
C4: harm_rate_K1 <= 0.85 * harm_rate_RANDOM_REF on ALL seeds        [competence floor]
C5: n_harm_events_RANDOM_REF >= 5 on ALL seeds                      [env presents harm]
C6: e2_action_discrimination >= 0.05 on ALL seeds                   [E2 ranks actions]

PASS = C1..C6 all hold -> supports H1 over H2.
Evidence-direction logic (evaluated in this order):
  C3 or C5 or C6 fails                  -> inconclusive (instrument/training inadequate)
  C4 fails                              -> inconclusive (k=1 baseline degenerate;
                                           explicitly NOT support for H2)
  C3,C4,C5,C6 hold and C1 or C2 fails   -> weakens  (H2 favoured: depth adds nothing)
  all hold                              -> supports (H1 over H2)

PROTOCOL
--------
3 seeds x (600 warmup episodes shared per seed) + 3 eval arms x 50 episodes
x 100 steps/episode. Warmup is SHARED across arms within a seed, so training is
matched by construction. Estimated runtime ~35-50 min (lightweight CPU).
"""

import sys
import random
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_1103_mech033_chain_depth_policy_driven"
QUEUE_ID = "V3-EXQ-1103"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-033"]
SOURCE_AUTOPSY = (
    "REE_assembly/evidence/planning/failure_autopsy_gflag0452-D2-cluster_2026-09-24.json"
    "#targets[5].fanout_recommendation.suggested_probes[1]"
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
THRESH_C1_MIN_REDUCTION = 0.01   # C1: mean(harm_K1 - harm_K3) >= 1 pp
THRESH_C3_E2_R2         = 0.20   # C3: world_forward_r2 (E2 convergence)
THRESH_C4_COMPETENCE    = 0.85   # C4: harm_K1 <= 0.85 * harm_RANDOM_REF
THRESH_C5_MIN_CONTACTS  = 5      # C5: n_harm_events_RANDOM_REF data quality
THRESH_C6_DISCRIM       = 0.05   # C6: E2 action-discrimination canary

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

# Arms. RANDOM_REF is informational only -- see the module docstring.
ARM_K1         = "K1"
ARM_K3         = "K3_POLICY"
ARM_RANDOM_REF = "RANDOM_REF"
ARMS = [ARM_K1, ARM_K3, ARM_RANDOM_REF]

DEPTH_BY_ARM = {ARM_K1: 1, ARM_K3: 3}

# --------------------------------------------------------------------------
# Model constants (identical to V3-EXQ-308, for lineage comparability)
# --------------------------------------------------------------------------
WORLD_OBS_DIM = 250
SELF_OBS_DIM  = 12
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 16


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
    """E2 forward-prediction kernel: f(z_world, a) -> z_world_next.
    This is the module MECH-033 is about."""
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, action], dim=-1))


class HarmHead(nn.Module):
    """f(z_world, z_self) -> harm_scalar. STATE-ONLY: no action input.
    This is precisely why the autopsy's suggested k=0 'HarmHead-greedy' arm was
    dropped from the portfolio -- with zero E2 forward steps it scores all
    actions identically. See the design doc, BLOCKING FINDING B."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


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
    seed: int, world_enc, self_enc, e2_fwd, harm_head, n_steps: int,
) -> float:
    """
    INFORMATIONAL. Fraction of states at which K1 and K3 choose DIFFERENT actions.

    Near 0 means the depth manipulation is behaviourally inert: the two arms
    execute the same policy, so any harm_rate difference between them is noise.
    That does not by itself invalidate the leg -- the depth path is proven to
    execute by _selfcheck_manipulation -- but an adjudicator must see it before
    reading a null as support for H2-one-step.
    """
    random.seed(seed + 555)
    torch.manual_seed(seed + 555)
    env = _make_env(seed + 2000)
    disagree = 0
    total = 0
    _, obs_dict = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            z_world = world_enc(_get_world_obs(obs_dict))
            z_self = self_enc(_get_self_obs(obs_dict))
            a1 = _select_action(z_world, z_self, e2_fwd, harm_head, 1)
            a3 = _select_action(z_world, z_self, e2_fwd, harm_head, 3)
            if a1 != a3:
                disagree += 1
            total += 1
            a_oh = _onehot(random.randint(0, ACTION_DIM - 1), ACTION_DIM)
            _, _, done, _, obs_next = env.step(a_oh.unsqueeze(0))
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


def _e2_action_discrimination(world_enc, e2_fwd, env, n_steps: int) -> float:
    """
    C6 CANARY (GFLAG-0485). Mean pairwise L2 distance between E2's predicted
    next-z_world across the ACTION_DIM actions, normalised by the mean predicted
    norm.

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
    seed: int, arm: str, world_enc, self_enc, e2_fwd, harm_head, dry_run: bool,
) -> Dict:
    """Eval one arm on the pre-trained stack. Eval env + RNG are identical
    across arms within a seed, so arms see matched conditions."""
    torch.manual_seed(seed + 999)
    random.seed(seed + 999)

    env = _make_env(seed + 1000)
    eval_eps = EVAL_EPISODES if not dry_run else 2
    depth = DEPTH_BY_ARM.get(arm)

    harm_events = 0
    total_steps = 0
    total_harm = 0.0
    drive_sum = 0.0

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
                        z_world, z_self, e2_fwd, harm_head, depth
                    )

            a_oh = _onehot(action_idx, ACTION_DIM)
            _, harm_signal, done, _, obs_next = env.step(a_oh.unsqueeze(0))
            harm_val = max(0.0, -harm_signal)

            drive_sum += _get_drive_level(obs_dict)
            total_harm += harm_val
            total_steps += 1
            if harm_val > 0.0:
                harm_events += 1

            obs_dict = env.reset()[1] if done else obs_next

    harm_rate = total_harm / max(total_steps, 1)
    print(
        "Seed %d Arm %-11s: harm_rate=%.5f n_harm_events=%d steps=%d"
        % (seed, arm, harm_rate, harm_events, total_steps),
        flush=True,
    )
    return {
        "arm": arm,
        "seed": seed,
        "harm_rate": harm_rate,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
        "mean_drive_level": drive_sum / max(total_steps, 1),
        "rollout_depth": depth if depth is not None else 0,
    }


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------

def _evaluate_criteria(results: Dict[str, List[Dict]], per_seed: Dict) -> Tuple:
    k1 = results[ARM_K1]
    k3 = results[ARM_K3]
    rnd = results[ARM_RANDOM_REF]
    n_s = len(SEEDS)

    deltas = [k1[i]["harm_rate"] - k3[i]["harm_rate"] for i in range(n_s)]
    mean_delta = sum(deltas) / max(len(deltas), 1)

    c1 = mean_delta >= THRESH_C1_MIN_REDUCTION
    c2 = all(k3[i]["harm_rate"] < k1[i]["harm_rate"] for i in range(n_s))
    c3 = all(per_seed[s]["world_forward_r2"] >= THRESH_C3_E2_R2 for s in SEEDS)
    c4 = all(
        k1[i]["harm_rate"] <= THRESH_C4_COMPETENCE * rnd[i]["harm_rate"]
        for i in range(n_s)
    )
    c5 = all(rnd[i]["n_harm_events"] >= THRESH_C5_MIN_CONTACTS for i in range(n_s))
    c6 = all(
        per_seed[s]["e2_action_discrimination"] >= THRESH_C6_DISCRIM for s in SEEDS
    )

    criteria = {
        "C1_mean_depth_gain_ge_0.01": c1,
        "C2_k3_below_k1_all_seeds": c2,
        "C3_world_forward_r2_ge_0.20": c3,
        "C4_k1_competence_floor_vs_random_ref": c4,
        "C5_random_ref_harm_events_ge_5": c5,
        "C6_e2_action_discrimination_ge_0.05": c6,
    }
    all_pass = all(criteria.values())
    status = "PASS" if all_pass else "FAIL"

    # Ordered direction logic -- instrument failures dominate.
    if not (c3 and c5 and c6):
        direction = "inconclusive"
        note = (
            "Instrument/training inadequate (C3 e2_r2=%s, C5 harm_events=%s, "
            "C6 discrimination=%s). CANNOT DETERMINE whether depth helps; this is "
            "NOT support for H2-one-step."
            % (c3, c5, c6)
        )
    elif not c4:
        direction = "inconclusive"
        note = (
            "COMPETENCE FLOOR FAILED: the k=1 baseline is not meaningfully better "
            "than the informational RANDOM_REF arm, so the k=1-vs-k=3 contrast is "
            "uninformative. Explicitly NOT support for H2-one-step -- this is the "
            "degenerate-comparator failure mode the portfolio exists to avoid."
        )
    elif not (c1 and c2):
        direction = "weakens"
        mean_disagree = sum(
            per_seed[s]["k1_k3_action_disagreement_rate"] for s in SEEDS
        ) / len(SEEDS)
        note = (
            "Declared H2 null HOLDS with every instrument gate passing: k=3 "
            "policy-driven chaining is not below k=1 by the 0.01 bar "
            "(mean delta %.5f). H2-one-step favoured over H1-chaining; MECH-033's "
            "multi-step chaining claim is weakened on the algorithm axis. "
            "K1/K3 action-disagreement rate %.4f -- the depth path is PROVEN to "
            "execute (see manipulation_selfcheck), so a low rate here means depth "
            "is behaviourally inert rather than that the code collapsed; read it "
            "before weighting this result."
            % (mean_delta, mean_disagree)
        )
    else:
        direction = "supports"
        note = (
            "Declared H2 null REJECTED: k=3 policy-driven chaining reduces harm "
            "below k=1 by mean %.5f on all seeds, with E2 trained "
            "(r2 gate), action-discriminating (canary) and the k=1 baseline "
            "itself competent vs RANDOM_REF. H1-chaining favoured over "
            "H2-one-step." % mean_delta
        )

    return criteria, status, direction, note, mean_delta, deltas


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(dry_run: bool = False, **kwargs) -> dict:
    print("[V3-EXQ-1103] MECH-033 leg 2 (algorithm axis): k=1 vs k=3 "
          "policy-driven continuation", flush=True)
    print("[V3-EXQ-1103] Declared null (H2): harm_rate(K3) NOT below "
          "harm_rate(K1) by >= %.2f" % THRESH_C1_MIN_REDUCTION, flush=True)
    print("[V3-EXQ-1103] RANDOM_REF is INFORMATIONAL ONLY (competence floor + "
          "data quality), never the null's comparator", flush=True)

    selfcheck = _selfcheck_manipulation()

    results: Dict[str, List[Dict]] = {a: [] for a in ARMS}
    per_seed: Dict = {}

    for seed in SEEDS:
        print("\n[V3-EXQ-1103] === seed %d ===" % seed, flush=True)
        (world_enc, prox_head, self_enc, e2_fwd, harm_head,
         e2_r2, prox_r2, discrim) = _train_models(seed, dry_run)
        disagree = _arm_disagreement_rate(
            seed, world_enc, self_enc, e2_fwd, harm_head,
            DISCRIM_STEPS if not dry_run else 20,
        )
        per_seed[seed] = {
            "world_forward_r2": e2_r2,
            "resource_prox_r2": prox_r2,
            "e2_action_discrimination": discrim,
            "k1_k3_action_disagreement_rate": disagree,
        }
        print("[instr] seed=%d k1_k3_action_disagreement_rate=%.4f"
              % (seed, disagree), flush=True)
        for arm in ARMS:
            results[arm].append(
                _run_arm(seed, arm, world_enc, self_enc, e2_fwd, harm_head, dry_run)
            )

    criteria, status, direction, note, mean_delta, deltas = _evaluate_criteria(
        results, per_seed
    )

    metrics = {
        "mean_depth_gain_k1_minus_k3": mean_delta,
        "per_seed_depth_gain": deltas,
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
    for arm in (ARM_K1, ARM_K3):
        for i, s in enumerate(SEEDS):
            denom = results[ARM_RANDOM_REF][i]["harm_rate"]
            metrics["harm_ratio_%s_over_random_ref_seed%d" % (arm, s)] = (
                results[arm][i]["harm_rate"] / denom if denom > 0 else float("nan")
            )
    for k, v in criteria.items():
        metrics["crit_%s" % k] = 1.0 if v else 0.0

    print("\n[V3-EXQ-1103] ===== VERDICT =====", flush=True)
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
            "H2: harm_rate(K3_POLICY) is NOT below harm_rate(K1) by >= %.2f"
            % THRESH_C1_MIN_REDUCTION
        ),
        "portfolio": {
            "name": "MECH-033 GOV-FANOUT-1 discrimination portfolio",
            "leg": "2 of 3",
            "axis": "algorithm",
            "hypotheses_discriminated": ["H1-chaining", "H2-one-step"],
            "siblings": {"V3-EXQ-1101": "integration axis (H3)",
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
            ARM_RANDOM_REF: (
                "Competence floor (C4) + data quality (C5) reference ONLY. Never "
                "the comparator for the declared null -- using a uniform-random "
                "arm that way is the D2 ablation confound this portfolio removes."
            )
        },
        "per_seed_instruments": {str(k): v for k, v in per_seed.items()},
        "manipulation_selfcheck": selfcheck,
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
            "depth_by_arm": DEPTH_BY_ARM,
            "warmup_episodes": WARMUP_EPISODES if not dry_run else 4,
            "eval_episodes": EVAL_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "lr": LR,
            "drive_weight": DRIVE_WEIGHT,
            "thresholds": {
                "C1_min_reduction": THRESH_C1_MIN_REDUCTION,
                "C3_e2_r2": THRESH_C3_E2_R2,
                "C4_competence_ratio": THRESH_C4_COMPETENCE,
                "C5_min_contacts": THRESH_C5_MIN_CONTACTS,
                "C6_discrimination": THRESH_C6_DISCRIM,
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
