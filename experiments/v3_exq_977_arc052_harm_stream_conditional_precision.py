#!/opt/local/bin/python3
"""
V3-EXQ-977 -- ARC-052: HARM-STREAM CONDITIONAL PRECISION (context-dependence antecedent).

EXPERIMENT_PURPOSE = evidence. claim_ids = ["ARC-052"].

red-team (fable): CONTESTED -- one primary finding, fixed. The verdict originally
read C1/C2 off the raw per-cell metrics while gating only on the AGGREGATE
"any arm green", so a red load-bearing quantile arm could still record `weakens`
on ARC-052 -- the exact false null R1 exists to catch. Confirmed by the reviewer in
simulation against the real precondition_gate module, and independently reproduced
at full scale here (z_harm_a arms RED on the R0 floor while C2 still scored).
Fixed: every reading is now gated on the gate of the arm it is read off, and a red
clause arm is UNSCORED rather than a refutation. Two minor findings also fixed: the
R0 precondition description now names the statistic it actually stamps (each arm's
own residual, not the point arm's), and the floor comparisons below are stated as
strict ">" to match precondition_gate's `met_for`. Reviewer-verified CLEAN: the
volatility IV alignment across the train/test split, the C2 sign algebra, and the
point arm's structural zero not being used as the statistical null.

THE CLAIM AND WHAT THIS RUN TESTS
---------------------------------
ARC-052 (harm_precision_weighting) has two separable halves:

  (A) An ARCHITECTURAL prescription: each harm encoder emits a precision estimate
      alongside its latent (z, log_sigma), and E3 weights harm-stream influence by
      that precision (z_harm_s -> SD-003 attribution, z_harm_a -> commit gating).
      NOT BUILT: HarmEncoder / AffectiveHarmEncoder (ree_core/latent/stack.py:119,
      175) emit a bare latent, and E3's only precision signal is the GLOBAL,
      state-blind running-variance EMA (e3_selector.py current_precision). Half (A)
      is `complicated (buildable)` and belongs to /implement-substrate, NOT here.

  (B) An EMPIRICAL antecedent -- the reason (A) would be worth building. ARC-052
      asserts precision is CONTEXT-DEPENDENT, in two signed clauses:
        (1) z_harm_s precision increases with forward-model accuracy ("when
            E2_harm_s predictions are good, the PE is more informative");
        (2) z_harm_a precision increases with accumulation STABILITY ("high
            volatility reduces confidence in the accumulated state").
      Both are claims about measurable relationships in the harm streams. Neither
      needs a precision head wired into an encoder, and neither needs the E3
      weighting. THIS RUN TESTS (B), AND ONLY (B).

If (B) fails, (A) is not worth building: a state-conditional precision that carries
no more information than the state-blind EMA cannot improve a gate that already has
the EMA. So this is a genuine, decision-grade PASS-or-FAIL on the current substrate,
and it is commitment-free -- no behavioural DV, no action-commitment layer, no
exposure to the known conversion / F-dominance ceiling.

METHOD -- SD-063 GENERALISED FROM z_world TO THE HARM STREAMS
------------------------------------------------------------
SD-063 already did exactly this experiment for the WORLD stream. V3-EXQ-712 compared
forward-head formulations on (z_world_t, a_t) -> z_world_{t+1} and found the
distribution-free quantile head carried a genuine per-point error signal
(precision_error_corr 0.379) that the state-blind EMA null (0.0 by construction)
structurally cannot. V3-EXQ-716a confirmed it and SD-063 was built
(ree_core/predictors/e2_world_uncertainty.py, E2WorldUncertaintyHead).

ARC-052 is that same mechanism asserted for the HARM streams. So this run reuses the
SHIPPED SD-063 module unchanged -- it is dimension-parameterised, and was confirmed
2026-09-02 to instantiate and produce per-input predictive_variance at both harm dims
(z_harm_s D=32, z_harm_a D=16) as well as z_world (D=32). Reusing the real substrate
module (rather than a local re-implementation) is deliberate: a PASS here transfers
directly into an ARC-052 build as "extend SD-063 to the harm streams", and a FAIL
falsifies that build with no ambiguity about whether a bespoke head was at fault.

  3 STREAMS x 2 HEADS, all six cells trained on the SAME rollouts per seed:
    stream: z_world (POSITIVE CONTROL -- the 712/716a result must reproduce)
            z_harm_s (ARC-052 clause 1)
            z_harm_a (ARC-052 clause 2)
    head:   point     -- MLP -> mean, MSE loss; homoscedastic sigma fitted on train
                         residuals. Constant per-input variance => the STATE-BLIND
                         null, and the direct analog of E3's running-variance EMA.
            quantile  -- the shipped E2WorldUncertaintyHead (9 levels 0.1..0.9,
                         pinball loss), predictive_variance from the rearranged IQR.

  Collecting all three streams in ONE rollout per seed is a design property, not an
  optimisation: the three streams see byte-identical trajectories, so a cross-stream
  difference cannot be a trajectory-sampling artifact.

ENCODERS ARE FIXED FEATURE MAPS (the phased-training discipline, in its strongest form)
--------------------------------------------------------------------------------------
No encoder is trained anywhere in this script, and every head trains on `.detach()`ed
latents. This is the P0->P1->P2 stop-gradient requirement satisfied maximally rather
than minimally: with a never-trained encoder there is no moving latent target at all,
so the EXQ-166b/c/d joint-training collapse mode is structurally unreachable. It also
matches V3-EXQ-712's method exactly, which is what makes the z_world POSITIVE CONTROL
a legitimate reproduction of the 0.379 reference rather than a loose analogy.

The cost is stated plainly: this measures ARC-052's context-dependence antecedent at
the level of the harm OBSERVATION stream carried through a fixed projection. Clause (2)
survives that intact -- harm_obs_a is the ENV's own EMA-accumulated harm vector, so its
accumulation dynamics are present in the input independently of the encoder, and the
volatility IV below is measured on the env-level `accumulated_harm` scalar, never on a
latent. A trained-encoder replication is the natural successor and is NOT claimed here.

PRE-REGISTERED READINESS + CRITERIA (all constants; none derived from the run)
-----------------------------------------------------------------------------
Readiness is PER-STREAM and regime-conditioned -- never AND'd whole-run (the
V3-EXQ-785 rule). One stream failing readiness scopes that stream out of scoring; it
does NOT vacate the others.

  Floors are STRICT (`measured > threshold`), matching precondition_gate's `met_for`.

  R0 (z_world + z_harm_s arms only): the arm's own normalised held-out residual
     (rmse / target_std) > R0_NORM_RESIDUAL_FLOOR, and n_test >
     R0_MIN_TEST_TRANSITIONS. There must be realized prediction-error spread for a
     precision estimate to track, else "no conditional precision" is UNMEASURABLE
     rather than a finding. SCOPED OUT of z_harm_a: this is an error-MAGNITUDE gate
     and it is the right gate only where the routed criterion CONSUMES realized error
     (R2 and C1 both route on precision_error_corr). C2 routes on an external
     volatility IV and never reads realized error, so R3 is its gate instead. That is
     same-statistic scoping, NOT a threshold relaxation -- the threshold is unchanged.
  R3 (z_harm_a arms only): held-out volatility-IV relative spread (std/mean) >
     R3_VOLATILITY_IV_REL_SPREAD_FLOOR. A structurally constant regressor would make
     C2 unmeasurable rather than false. Degeneracy guard on C2's own IV.
  R1 (per stream, quantile cell): the trained quantile head's pvar RELATIVE spread
     (range/mean over the eval batch) >= R1_PVAR_RELATIVE_SPREAD_FLOOR. Per
     e2_world_uncertainty.py's own readiness note, RELATIVE spread -- not absolute
     range -- is what separates a trained head from a random-init one (untrained
     heads have LARGER absolute range). Guards a false null caused by a head that
     simply did not train.
  R2 (POSITIVE CONTROL, whole-run): z_world quantile precision_error_corr >=
     R2_ZWORLD_CONTROL_CORR_FLOOR (0.15, well under the 712-measured 0.379). This is
     the SAME statistic C1 routes on, measured on the stream where the answer is
     already known. If R2 fails, the instrument is not reproducing an established
     result -> substrate_not_ready_requeue, NEVER an ARC-052 verdict.

  C1 (LOAD-BEARING; ARC-052 clause 1): z_harm_s quantile precision_error_corr >=
     C1_CORR_FLOOR AND strictly above its own PERMUTATION-NULL 95th percentile, on a
     strict majority of seeds.
  C2 (ARC-052 clause 2): z_harm_a precision_volatility_corr <= C2_PRECISION_VOL_CORR_MAX
     (i.e. predicted PRECISION falls as accumulation volatility rises -- ARC-052's
     signed prediction) AND strictly beyond its permutation-null 5th percentile, on a
     strict majority of seeds.
  C3 (secondary, reported, NOT gating): per-stream CRPS improvement, quantile vs point.

  EVERY reading is gated on the gate of the ARM IT IS READ OFF, never on the
  aggregate: aggregate["non_degenerate"] is ANY-arm-green, and the point arms face
  fewer preconditions, so the aggregate can be green while the load-bearing quantile
  arm is red. A red clause arm is UNSCORED, not a refutation; if exactly one clause
  arm is scorable the run records `mixed` and says which clause was unmeasurable,
  rather than letting a half-test stand as a verdict in either direction.

WHY A PERMUTATION NULL RATHER THAN THE POINT ARM'S ZERO
------------------------------------------------------
The point head's per-input variance is CONSTANT, so its precision_error_corr is 0.0
BY CONSTRUCTION, not by measurement. Scoring C1/C2 as a delta against that number
would compare a measurement to a definition. So each correlation is additionally
tested against a within-cell permutation null (predictions shuffled against targets,
N_PERM draws), which is an actual sampling distribution. The point arm is still run
and reported -- it is the honest state-blind EMA analog for CRPS/calibration (C3) --
but it is NOT the statistical null for C1/C2.

DV-SYMMETRY INVARIANCE (per-arm declaration, mandatory)
------------------------------------------------------
DVs are precision_error_corr / precision_volatility_corr (Pearson correlations) and
crps_mean.
  - Pearson correlation is invariant under positive affine rescaling of either
    variable. The manipulation (point -> quantile head) is NOT such a transform: it
    changes the per-input variance from a CONSTANT to a state-varying quantity, and a
    constant has no correlation to rescale. Not invariant -> the DV can move.
  - Correlation is also invariant under a permutation of the sample index. The
    manipulation is not a permutation -- it changes the per-point values themselves.
    (The permutation null above deliberately EXPLOITS this symmetry as its null.)
  - CRPS is a proper scoring rule over the predictive distribution; a head-family
    change alters the distribution's shape, which CRPS is by construction sensitive
    to. Not invariant.
  - Cross-stream arms differ in D (32/32/16) but every DV is meaned over dims before
    the correlation, so no DV is a function of dimensionality alone.
None of the six arms is invariant under its DV's symmetry group.

MECH-094: N/A -- waking observation stream; no memory write, no simulation/replay.

Substrate-path gate (Step 2.5c): the two open `corrupting` substrate_queue entries
naming ree_core/agent.py (`mode-governance-engagement`, `SD-082`) carry their defects
in ree_core/cingulate/salience_coordinator.py and
ree_core/pfc/lateral_pfc_analog.py::compute_bias. A call trace over this driver's exact
path (agent.sense + env.step, no select_action, both features default-off) executed
NEITHER module. Recorded in the queue-entry note.

See ree_core/predictors/e2_world_uncertainty.py (the SD-063 head reused here),
    ree_core/latent/stack.py (HarmEncoder / AffectiveHarmEncoder -- the encoders
      ARC-052 half (A) would give precision heads),
    ree_core/predictors/e3_selector.py (current_precision EMA -- the state-blind null),
    experiments/v3_exq_712_distributional_world_forward_heads.py (the z_world
      predecessor whose method this generalises).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.manifest_core import stamp_recording_core
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e2_world_uncertainty import (
    E2WorldUncertaintyConfig,
    E2WorldUncertaintyHead,
    QUANTILE_LEVELS,
)
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_977_arc052_harm_stream_conditional_precision"
QUEUE_ID = "V3-EXQ-977"
SUPERSEDES = None
BACKLOG_ID = "EVB-1197"
CLAIM_IDS: List[str] = ["ARC-052"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ----- Substrate dims / config -----
WORLD_DIM = 32
SELF_DIM = 32
Z_HARM_DIM = 32          # HarmEncoder default (SD-010 sensory-discriminative)
Z_HARM_A_DIM = 16        # AffectiveHarmEncoder default (SD-011 affective)
ALPHA_WORLD = 0.9        # SD-008: default 0.3 is too low for z_world fidelity
ALPHA_SELF = 0.3
HARM_HISTORY_LEN = 10

HEAD_HIDDEN = 128
NOMINAL_COVERAGE = 0.80
CRPS_SAMPLES = 100

# ----- Run size -----
SEEDS = [42, 43, 45, 46, 47]     # seed 44 excluded: known reef-config instability
COLLECT_EPISODES = 40            # runner denominator M for the [train] prints
STEPS_PER_EPISODE = 120
HEAD_TRAIN_EPOCHS = 40
HEAD_BATCH = 256
HEAD_LR = 1e-3
TRAIN_FRAC = 0.8
VOLATILITY_WINDOW = 10           # matches HARM_HISTORY_LEN
N_PERM = 400                     # permutation-null draws per correlation

DRY_RUN_SEEDS = [42]
DRY_RUN_COLLECT = 3
DRY_RUN_STEPS = 25
DRY_RUN_EPOCHS = 3
DRY_RUN_PERM = 40

# ----- PRE-REGISTERED thresholds (constants; never derived from this run) -----
R0_NORM_RESIDUAL_FLOOR = 0.05        # point rmse / target std -- conditional spread exists
R0_MIN_TEST_TRANSITIONS = 200
R1_PVAR_RELATIVE_SPREAD_FLOOR = 0.05  # trained-head guard (relative, not absolute)
R2_ZWORLD_CONTROL_CORR_FLOOR = 0.15   # 712 measured 0.379 on this statistic
C1_CORR_FLOOR = 0.10                  # z_harm_s precision_error_corr
C2_PRECISION_VOL_CORR_MAX = -0.10     # z_harm_a precision vs volatility (signed NEGATIVE)
C3_CRPS_IMPROVE_FRAC = 0.02           # reported only, not gating
# Degeneracy guard on C2's own regressor: the volatility IV must actually vary.
# A pure non-degeneracy floor, far below the observed relative spread (~0.68).
R3_VOLATILITY_IV_REL_SPREAD_FLOOR = 0.05

# z_goal liveness counters. Agents are built INSIDE _collect (one per seed), so an
# accumulator is used rather than holding every agent alive to manifest-write time.
_ZG = ZGoalStreamAccumulator()

STREAMS = ["z_world", "z_harm_s", "z_harm_a"]
STREAM_DIM = {"z_world": WORLD_DIM, "z_harm_s": Z_HARM_DIM, "z_harm_a": Z_HARM_A_DIM}
HEADS = ["point", "quantile"]


def _majority(n: int) -> int:
    return n // 2 + 1


ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=HARM_HISTORY_LEN,
)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class PointHead(nn.Module):
    """MLP -> mean; MSE loss. Constant (homoscedastic) predictive variance fitted on
    train residuals -- the STATE-BLIND analog of E3's running-variance EMA."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_quantile_head(z_dim: int, action_dim: int) -> E2WorldUncertaintyHead:
    """The SHIPPED SD-063 module, instantiated at a harm-stream dimensionality.

    `z_world_dim` is the module's name for the latent width; it carries no
    world-stream semantics (confirmed 2026-09-02: the head instantiates and yields
    per-input predictive_variance at D=32 and D=16 alike)."""
    return E2WorldUncertaintyHead(
        E2WorldUncertaintyConfig(
            use_e2_world_uncertainty=True,
            z_world_dim=int(z_dim),
            action_dim=int(action_dim),
            hidden_dim=HEAD_HIDDEN,
        )
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = float(np.corrcoef(a, b)[0, 1])
    return c if math.isfinite(c) else 0.0


def _perm_null(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator
               ) -> Tuple[float, float]:
    """Permutation null for corr(a, b): returns (p05, p95) of the shuffled statistic."""
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return (0.0, 0.0)
    draws = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        draws[i] = _pearson(a, rng.permutation(b))
    return (float(np.percentile(draws, 5.0)), float(np.percentile(draws, 95.0)))


def _crps_from_samples(samples: torch.Tensor, target: torch.Tensor) -> float:
    """Sample-based CRPS: E|X-y| - 0.5 E|X-X'|. samples [S,B,D], target [B,D]."""
    s = samples
    y = target.unsqueeze(0)
    term1 = (s - y).abs().mean(dim=0)
    perm = torch.randperm(s.shape[0])
    term2 = (s - s[perm]).abs().mean(dim=0)
    return float((term1 - 0.5 * term2).mean().item())


def _coverage_from_samples(samples: torch.Tensor, target: torch.Tensor,
                           nominal: float) -> float:
    lo_q = (1.0 - nominal) / 2.0
    hi_q = 1.0 - lo_q
    lo = torch.quantile(samples, lo_q, dim=0)
    hi = torch.quantile(samples, hi_q, dim=0)
    inside = ((target >= lo) & (target <= hi)).float()
    return float(inside.mean().item())


# ---------------------------------------------------------------------------
# Transition collection -- all three streams from ONE rollout per seed
# ---------------------------------------------------------------------------


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        use_harm_stream=True,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=Z_HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
    )
    return REEAgent(cfg)


def _b(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x.unsqueeze(0) if x.dim() == 1 else x


def _collect(seed: int, collect_episodes: int, steps_per_episode: int
             ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], np.ndarray, int, Dict[str, Any]]:
    """Roll out random-action episodes once; harvest (z_t, a_t, z_{t+1}) for ALL
    three streams from the SAME transitions, plus the env-level accumulated-harm
    volatility IV aligned to each transition.

    Returns (per-stream {Z0, A, Z1}, volatility [N], action_dim, collection diag).
    """
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    agent = _make_agent(env)
    action_dim = int(env.action_dim)

    z0: Dict[str, List[torch.Tensor]] = {s: [] for s in STREAMS}
    z1: Dict[str, List[torch.Tensor]] = {s: [] for s in STREAMS}
    acts: List[torch.Tensor] = []
    vol: List[float] = []
    n_nonfinite = 0

    for ep in range(collect_episodes):
        _, obs = env.reset()
        agent.reset()
        pending: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = None
        accum_hist: List[float] = []

        for _step in range(steps_per_episode):
            latent = agent.sense(
                obs_body=_b(obs["body_state"]),
                obs_world=_b(obs["world_state"]),
                obs_harm=_b(obs["harm_obs"]),
                obs_harm_a=_b(obs["harm_obs_a"]),
                obs_harm_history=_b(obs["harm_history"]),
            )
            cur = {
                "z_world": latent.z_world.detach().reshape(-1).clone(),
                "z_harm_s": latent.z_harm.detach().reshape(-1).clone(),
                "z_harm_a": latent.z_harm_a.detach().reshape(-1).clone(),
            }
            accum_hist.append(float(obs["accumulated_harm"]))

            if pending is not None:
                prev, a_prev = pending
                ok = all(torch.isfinite(prev[s]).all() and torch.isfinite(cur[s]).all()
                         for s in STREAMS) and torch.isfinite(a_prev).all()
                if ok:
                    for s in STREAMS:
                        z0[s].append(prev[s])
                        z1[s].append(cur[s].clone())
                    acts.append(a_prev)
                    # Volatility IV: rolling std of the ENV-level accumulated_harm over
                    # the window ENDING at the transition's source step. Measured on the
                    # env scalar, never on a latent, so it is encoder-independent.
                    win = accum_hist[max(0, len(accum_hist) - 1 - VOLATILITY_WINDOW):
                                     len(accum_hist) - 1]
                    vol.append(float(np.std(win)) if len(win) >= 2 else 0.0)
                else:
                    n_nonfinite += 1
                pending = None

            a_idx = int(np.random.randint(0, action_dim))
            action = torch.zeros(1, action_dim)
            action[0, a_idx] = 1.0
            pending = ({s: cur[s].clone() for s in STREAMS}, action.reshape(-1).clone())
            # Run the full step budget rather than stopping at `done` (the established
            # collection pattern, cf. v3_exq_712 / v3_exq_711): terminating at the first
            # harm event ~step 12 would starve the dataset.
            _, _harm, _done, _info, obs = env.step(action)

        if (ep + 1) % 10 == 0 or (ep + 1) == collect_episodes:
            print(f"  [train] collect seed={seed} ep {ep + 1}/{collect_episodes} "
                  f"n_trans={len(acts)}", flush=True)

    # Read the z_goal counters AFTER stepping (observe() samples at call time).
    _ZG.observe(agent)

    diag = {"n_transitions": len(acts), "n_nonfinite_skipped": n_nonfinite}
    if not acts:
        empty = {s: {"Z0": torch.zeros(0, STREAM_DIM[s]), "A": torch.zeros(0, action_dim),
                     "Z1": torch.zeros(0, STREAM_DIM[s])} for s in STREAMS}
        return empty, np.zeros(0), action_dim, diag

    A = torch.stack(acts)
    out = {s: {"Z0": torch.stack(z0[s]), "A": A, "Z1": torch.stack(z1[s])} for s in STREAMS}
    return out, np.asarray(vol, dtype=float), action_dim, diag


# ---------------------------------------------------------------------------
# Train + evaluate one (stream, head) cell
# ---------------------------------------------------------------------------


def _train_cell(head_type: str, z_dim: int, action_dim: int,
                Z0_tr: torch.Tensor, A_tr: torch.Tensor, Z1_tr: torch.Tensor,
                epochs: int) -> Tuple[nn.Module, Optional[torch.Tensor]]:
    """P1: train the head on FROZEN (already-detached) latents. The encoder is never
    touched -- inputs and targets both come from a fixed feature map."""
    if head_type == "point":
        head: nn.Module = PointHead(z_dim + action_dim, z_dim)
    else:
        head = _build_quantile_head(z_dim, action_dim)
    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    n = Z0_tr.shape[0]
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, HEAD_BATCH):
            idx = perm[i:i + HEAD_BATCH]
            z0b, ab, z1b = Z0_tr[idx], A_tr[idx], Z1_tr[idx]
            opt.zero_grad(set_to_none=True)
            if head_type == "point":
                loss = F.mse_loss(head(torch.cat([z0b, ab], dim=-1)), z1b)
            else:
                loss = head.compute_loss(head(z0b, ab), z1b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

    sigma_global: Optional[torch.Tensor] = None
    if head_type == "point":
        with torch.no_grad():
            resid = Z1_tr - head(torch.cat([Z0_tr, A_tr], dim=-1))
            sigma_global = torch.sqrt(resid.var(dim=0, unbiased=True).clamp(min=1e-8))
    return head, sigma_global


def _eval_cell(head_type: str, head: nn.Module, z_dim: int,
               Z0_te: torch.Tensor, A_te: torch.Tensor, Z1_te: torch.Tensor,
               vol_te: np.ndarray, sigma_global: Optional[torch.Tensor],
               n_perm: int, rng: np.random.Generator) -> Dict[str, Any]:
    levels = torch.tensor(QUANTILE_LEVELS)
    with torch.no_grad():
        if head_type == "point":
            mean_pred = head(torch.cat([Z0_te, A_te], dim=-1))
            sig = sigma_global.view(1, -1).expand_as(mean_pred)
            samples = mean_pred.unsqueeze(0) + sig.unsqueeze(0) * torch.randn(
                CRPS_SAMPLES, *mean_pred.shape)
            # Constant per-input variance -- the state-blind EMA analog.
            pvar = (sig ** 2).mean(dim=-1)
            pvar_rel_spread = 0.0
        else:
            q = head(Z0_te, A_te)                                # [B, D, Q]
            qs, _ = torch.sort(q, dim=-1)
            mid = qs.shape[-1] // 2
            mean_pred = qs[..., mid]                             # median quantile
            u = torch.rand(CRPS_SAMPLES, qs.shape[0], qs.shape[1])
            idx = (u * qs.shape[-1]).long().clamp(0, qs.shape[-1] - 1)
            samples = torch.gather(
                qs.unsqueeze(0).expand(CRPS_SAMPLES, -1, -1, -1), -1, idx.unsqueeze(-1)
            ).squeeze(-1)
            pvar = head.predictive_variance(Z0_te, A_te)         # [B]
            pv = pvar.cpu().numpy()
            m = float(np.mean(pv))
            pvar_rel_spread = float((np.max(pv) - np.min(pv)) / m) if m > 1e-12 else 0.0

        rmse = float(torch.sqrt(((mean_pred - Z1_te) ** 2).mean()).item())
        target_std = float(Z1_te.std().item())
        crps = _crps_from_samples(samples, Z1_te)
        coverage = _coverage_from_samples(samples, Z1_te, NOMINAL_COVERAGE)
        sq_err = ((mean_pred - Z1_te) ** 2).mean(dim=-1)

    pvar_np = np.asarray(pvar.cpu().numpy(), dtype=float).reshape(-1)
    err_np = np.asarray(sq_err.cpu().numpy(), dtype=float).reshape(-1)

    # ARC-052 clause (1): predicted spread tracks realized error.
    prec_err_corr = _pearson(pvar_np, err_np)
    pe_p05, pe_p95 = _perm_null(pvar_np, err_np, n_perm, rng)

    # ARC-052 clause (2): predicted PRECISION falls as accumulation volatility rises.
    # corr(variance, volatility) > 0  <=>  corr(precision, volatility) < 0.
    v = np.asarray(vol_te, dtype=float).reshape(-1)
    if v.size == pvar_np.size:
        var_vol_corr = _pearson(pvar_np, v)
        vv_p05, vv_p95 = _perm_null(pvar_np, v, n_perm, rng)
    else:
        var_vol_corr, vv_p05, vv_p95 = 0.0, 0.0, 0.0

    return {
        "point_rmse": rmse,
        "target_std": target_std,
        "norm_residual": (rmse / target_std) if target_std > 1e-12 else 0.0,
        "crps_mean": crps,
        "coverage_80": coverage,
        "coverage_err_80": abs(coverage - NOMINAL_COVERAGE),
        "precision_error_corr": prec_err_corr,
        "precision_error_corr_perm_p95": pe_p95,
        "precision_error_corr_beats_null": bool(prec_err_corr > pe_p95),
        "variance_volatility_corr": var_vol_corr,
        # ARC-052 states the relationship in PRECISION terms; report it that way too.
        "precision_volatility_corr": -var_vol_corr,
        "precision_volatility_corr_perm_p05": -vv_p95,
        "precision_volatility_beats_null": bool((-var_vol_corr) < (-vv_p95)),
        "pvar_relative_spread": pvar_rel_spread,
        "volatility_iv_relative_spread": (
            float(np.std(v) / np.mean(v)) if v.size and float(np.mean(v)) > 1e-12 else 0.0
        ),
        "n_test": int(Z0_te.shape[0]),
    }


def _cell_config_slice(stream: str, head: str, seed: int, collect_episodes: int,
                       steps_per_episode: int, epochs: int) -> Dict[str, Any]:
    return {
        "stream": stream,
        "head": head,
        "z_dim": STREAM_DIM[stream],
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "z_harm_dim": Z_HARM_DIM,
        "z_harm_a_dim": Z_HARM_A_DIM,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "harm_history_len": HARM_HISTORY_LEN,
        "head_hidden": HEAD_HIDDEN,
        "quantile_levels": list(QUANTILE_LEVELS),
        "crps_samples": CRPS_SAMPLES,
        # Readout-affecting: sets the nominal interval for coverage_80 /
        # coverage_err_80, so a consumer using a different value must MISS.
        "nominal_coverage": NOMINAL_COVERAGE,
        "head_lr": HEAD_LR,
        "head_batch": HEAD_BATCH,
        "epochs": int(epochs),
        "train_frac": TRAIN_FRAC,
        "collect_episodes": int(collect_episodes),
        "steps_per_episode": int(steps_per_episode),
        "volatility_window": VOLATILITY_WINDOW,
        "env_kwargs": dict(ENV_KWARGS),
    }


# ---------------------------------------------------------------------------
# Precondition specs (per-stream, regime-conditioned)
# ---------------------------------------------------------------------------


def _precondition_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="norm_residual_supra_floor",
            description=("THIS ARM's own normalised held-out residual "
                         "(rmse/target_std; the arm's own mean prediction, so the "
                         "point and quantile cells each certify themselves): the "
                         "stream carries realized prediction-error spread for a "
                         "precision estimate to track."),
            control="this arm's own head on held-out transitions of this stream",
            threshold=R0_NORM_RESIDUAL_FLOOR,
            direction="lower",
            kind="readiness",
            # SAME-STATISTIC scoping, not a threshold relaxation. This is an
            # error-MAGNITUDE gate, and it is the right gate only where the routed
            # criterion CONSUMES realized error: R2 and C1 both route on
            # precision_error_corr = corr(predictive variance, realized squared error),
            # which is noise without error spread. z_harm_a's routed criterion (C2)
            # routes on corr(predictive variance, an EXTERNAL accumulation-volatility
            # IV) and never reads realized error at all, so an error-magnitude floor
            # would vacate C2 while measuring something C2 does not consume. C2's
            # same-statistic gate is volatility_iv_relative_spread_supra_floor below.
            applies_to=lambda ctx: ctx.get("stream") in ("z_world", "z_harm_s"),
            applies_note=("C2 (z_harm_a) routes on an external volatility IV, not on "
                          "realized error; its IV-spread gate applies instead"),
        ),
        PreconditionSpec(
            name="volatility_iv_relative_spread_supra_floor",
            description=("Held-out accumulation-volatility IV relative spread "
                         "(std/mean). C2 correlates predicted precision AGAINST this "
                         "IV, so a structurally constant IV would make C2 unmeasurable "
                         "rather than false. Degeneracy guard on C2's own regressor."),
            control="env-level accumulated_harm rolling std over held-out transitions",
            threshold=R3_VOLATILITY_IV_REL_SPREAD_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx.get("stream") == "z_harm_a",
            applies_note="only C2's stream routes on the volatility IV",
        ),
        PreconditionSpec(
            name="n_test_transitions_supra_floor",
            description="Held-out transition count for this stream.",
            control="collection budget for this seed",
            threshold=float(R0_MIN_TEST_TRANSITIONS),
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="pvar_relative_spread_supra_floor",
            description=("Trained quantile head's per-input predictive-variance RELATIVE "
                         "spread (range/mean). Per e2_world_uncertainty.py, RELATIVE "
                         "spread -- not absolute range -- separates a trained head from a "
                         "random-init one. Guards a false null from an untrained head."),
            control="the quantile cell's own trained head on held-out transitions",
            threshold=R1_PVAR_RELATIVE_SPREAD_FLOOR,
            direction="lower",
            kind="readiness",
            # Not meaningful for the point head: its variance is constant BY DESIGN
            # (that is what makes it the state-blind null), so a spread floor would
            # fail it structurally rather than diagnose it.
            applies_to=lambda ctx: ctx.get("head") == "quantile",
            applies_note="point head is homoscedastic by construction; spread floor N/A",
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    collect_eps = DRY_RUN_COLLECT if dry_run else COLLECT_EPISODES
    steps_ep = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    epochs = DRY_RUN_EPOCHS if dry_run else HEAD_TRAIN_EPOCHS
    n_perm = DRY_RUN_PERM if dry_run else N_PERM

    specs = _precondition_specs()
    arm_contexts = [{"id": f"{s}__{h}", "stream": s, "head": h}
                    for s in STREAMS for h in HEADS]
    # Refuse before compute if any gate is structurally unsatisfiable for an arm.
    assert_no_structurally_unsatisfiable_gate(specs, arm_contexts, arm_id_key="id")

    arm_results: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    collection_diag: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"Seed {seed} Condition arc052_harm_precision", flush=True)
        rng = np.random.default_rng(seed)

        # Collection is seeded once per seed and SHARED by all six cells, so the
        # cross-stream / cross-head comparison sees byte-identical trajectories.
        torch.manual_seed(seed)
        np.random.seed(seed)
        data, vol, action_dim, cdiag = _collect(seed, collect_eps, steps_ep)
        cdiag["seed"] = seed
        collection_diag.append(cdiag)

        n = int(data[STREAMS[0]]["Z0"].shape[0])
        n_tr = int(n * TRAIN_FRAC)
        seed_cells: Dict[str, Dict[str, Any]] = {}

        for stream in STREAMS:
            Z0, A, Z1 = data[stream]["Z0"], data[stream]["A"], data[stream]["Z1"]
            for head_type in HEADS:
                arm_id = f"{stream}__{head_type}"
                with arm_cell(
                    seed,
                    config_slice=_cell_config_slice(stream, head_type, seed,
                                                    collect_eps, steps_ep, epochs),
                    script_path=Path(__file__),
                    config_slice_declared=True,
                    include_driver_script_in_hash=False,
                    extra_ineligible_reasons=["shared_rollout_across_cells"],
                ) as cell:
                    if n < 8:
                        metrics = {"point_rmse": 0.0, "target_std": 0.0,
                                   "norm_residual": 0.0, "crps_mean": float("nan"),
                                   "coverage_80": 0.0, "coverage_err_80": 1.0,
                                   "precision_error_corr": 0.0,
                                   "precision_error_corr_perm_p95": 0.0,
                                   "precision_error_corr_beats_null": False,
                                   "variance_volatility_corr": 0.0,
                                   "precision_volatility_corr": 0.0,
                                   "precision_volatility_corr_perm_p05": 0.0,
                                   "precision_volatility_beats_null": False,
                                   "pvar_relative_spread": 0.0, "n_test": 0}
                    else:
                        head, sigma_g = _train_cell(
                            head_type, STREAM_DIM[stream], action_dim,
                            Z0[:n_tr], A[:n_tr], Z1[:n_tr], epochs)
                        metrics = _eval_cell(
                            head_type, head, STREAM_DIM[stream],
                            Z0[n_tr:], A[n_tr:], Z1[n_tr:], vol[n_tr:],
                            sigma_g, n_perm, rng)

                    row: Dict[str, Any] = {
                        "arm_id": arm_id, "stream": stream, "head": head_type,
                        "seed": int(seed), **metrics,
                    }
                    cell.stamp(row)
                arm_results.append(row)
                seed_cells[arm_id] = row

        # ---- per-seed criterion readings -------------------------------------
        zw_q = seed_cells["z_world__quantile"]
        hs_q = seed_cells["z_harm_s__quantile"]
        ha_q = seed_cells["z_harm_a__quantile"]

        r2_met = bool(zw_q["precision_error_corr"] >= R2_ZWORLD_CONTROL_CORR_FLOOR)
        c1_met = bool(hs_q["precision_error_corr"] >= C1_CORR_FLOOR
                      and hs_q["precision_error_corr_beats_null"])
        c2_met = bool(ha_q["precision_volatility_corr"] <= C2_PRECISION_VOL_CORR_MAX
                      and ha_q["precision_volatility_beats_null"])

        c3 = {}
        for stream in STREAMS:
            p = seed_cells[f"{stream}__point"]["crps_mean"]
            q = seed_cells[f"{stream}__quantile"]["crps_mean"]
            c3[stream] = bool(math.isfinite(p) and math.isfinite(q)
                              and q < p * (1.0 - C3_CRPS_IMPROVE_FRAC))

        seed_pass = bool(r2_met and c1_met and c2_met)
        per_seed.append({
            "seed": int(seed), "n_transitions": n,
            "R2_zworld_control_met": r2_met,
            "C1_harm_s_precision_error_met": c1_met,
            "C2_harm_a_precision_volatility_met": c2_met,
            "C3_crps_improved_by_stream": c3,
            "zworld_precision_error_corr": zw_q["precision_error_corr"],
            "harm_s_precision_error_corr": hs_q["precision_error_corr"],
            "harm_a_precision_volatility_corr": ha_q["precision_volatility_corr"],
            "seed_pass": seed_pass,
        })
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    # ---- aggregate across seeds ---------------------------------------------
    n_seeds = len(per_seed)
    maj = _majority(n_seeds)
    n_r2 = sum(1 for s in per_seed if s["R2_zworld_control_met"])
    n_c1 = sum(1 for s in per_seed if s["C1_harm_s_precision_error_met"])
    n_c2 = sum(1 for s in per_seed if s["C2_harm_a_precision_volatility_met"])

    R2_MET = bool(n_r2 >= maj)
    C1 = bool(n_c1 >= maj)
    C2 = bool(n_c2 >= maj)

    # ---- per-arm precondition gates (regime-conditioned, never AND'd whole-run) --
    arm_gates = []
    for ctx in arm_contexts:
        arm_id = ctx["id"]
        rows = [r for r in arm_results if r["arm_id"] == arm_id]
        if not rows:
            continue
        # Worst cell across seeds: the gate is a worst-case claim, so report the
        # extremum, not the mean (the indexer recomputes `met` from this number).
        worst_norm = min(float(r["norm_residual"]) for r in rows)
        worst_ntest = min(float(r["n_test"]) for r in rows)
        worst_spread = min(float(r["pvar_relative_spread"]) for r in rows)
        worst_vol_spread = min(float(r["volatility_iv_relative_spread"]) for r in rows)
        arm_gates.append(evaluate_arm_gate(
            arm_id, ctx, specs,
            measured={
                "norm_residual_supra_floor": worst_norm,
                "n_test_transitions_supra_floor": worst_ntest,
                "pvar_relative_spread_supra_floor": worst_spread,
                "volatility_iv_relative_spread_supra_floor": worst_vol_spread,
            },
        ))
    aggregate = aggregate_arm_gates(arm_gates)

    # ---- verdict ------------------------------------------------------------
    # TWO independent instrument gates precede any ARC-052 reading, and BOTH route
    # to substrate_not_ready_requeue rather than to a claim verdict:
    #   (a) no arm cleared its own readiness preconditions (aggregate.non_degenerate
    #       is False) -- e.g. too few held-out transitions, or a quantile head whose
    #       relative pvar spread shows it never trained. Without this the criteria
    #       would be read off cells that are not entitled to answer.
    #   (b) the z_world POSITIVE CONTROL failed to reproduce the established
    #       SD-063/712 precision_error_corr floor -- the instrument is not working.
    # Each reading is gated on the gate of the ARM IT IS READ OFF -- NOT on the
    # aggregate. aggregate["non_degenerate"] is ANY-arm-green (precondition_gate.py),
    # and the two point arms face fewer preconditions (the R1 spread floor is scoped
    # out of them), so a point arm alone would hold the aggregate green while the
    # load-bearing quantile arm that actually produces C1/C2 was red. Reading a
    # criterion off a red arm is exactly the false null R1 was pre-registered to
    # catch, and it would record as `weakens` on ARC-052. A red arm is UNSCORED, not
    # a refutation (precondition_gate.py's own doctrine).
    green = set(aggregate.get("green_arms") or [])
    control_arm_green = "z_world__quantile" in green
    scorable_c1 = "z_harm_s__quantile" in green
    scorable_c2 = "z_harm_a__quantile" in green
    control_ok = bool(control_arm_green and R2_MET)

    if not control_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        non_degenerate = False
        if not control_arm_green:
            summary = ("The z_world POSITIVE-CONTROL arm did not clear its own readiness "
                       f"preconditions ({aggregate.get('degeneracy_reason') or 'arm red'}); "
                       "the instrument is unvalidated, so ARC-052 was not adjudicated.")
        else:
            summary = ("z_world positive control did not reproduce the SD-063/712 "
                       "precision_error_corr floor; ARC-052 not adjudicated.")
    elif not scorable_c1 and not scorable_c2:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        non_degenerate = False
        summary = ("The control reproduced, but NEITHER harm-stream clause arm cleared "
                   "its readiness preconditions, so neither ARC-052 clause was measured. "
                   "This is an unscored run, NOT a refutation.")
    elif not (scorable_c1 and scorable_c2):
        # Exactly one clause is measurable. Report the measured one; do NOT let a
        # half-test stand as a verdict on the whole claim in either direction.
        outcome = "FAIL"
        which = "clause 2 (z_harm_a)" if scorable_c1 else "clause 1 (z_harm_s)"
        got = C1 if scorable_c1 else C2
        held = "clause 1 (z_harm_s)" if scorable_c1 else "clause 2 (z_harm_a)"
        label = "partial_adjudication_one_clause_unmeasurable"
        evidence_direction = "mixed"
        non_degenerate = True
        summary = (f"Only one ARC-052 clause was measurable: {which} did not clear its "
                   f"readiness preconditions and is UNSCORED (not refuted). The "
                   f"measurable clause, {held}, "
                   f"{'HELD' if got else 'did NOT hold'}. ARC-052's antecedent is "
                   "therefore neither established nor refuted by this run.")
    elif C1 and C2:
        outcome = "PASS"
        label = "harm_stream_precision_is_context_dependent"
        evidence_direction = "supports"
        non_degenerate = True
        summary = ("Both ARC-052 context-dependence clauses held on the harm streams: "
                   "conditional precision tracked forward-model error (clause 1) and "
                   "fell with accumulation volatility (clause 2).")
    elif (not C1) and (not C2):
        outcome = "FAIL"
        label = "harm_stream_precision_not_context_dependent"
        evidence_direction = "weakens"
        non_degenerate = True
        summary = ("Neither ARC-052 clause held while the z_world control reproduced, "
                   "so a state-conditional harm precision carries no signal the "
                   "state-blind EMA lacks; ARC-052's premise for precision heads fails.")
    else:
        outcome = "FAIL"
        label = "harm_stream_precision_partially_context_dependent"
        evidence_direction = "mixed"
        non_degenerate = True
        held = "clause 1 (z_harm_s)" if C1 else "clause 2 (z_harm_a)"
        summary = (f"Exactly one ARC-052 clause held ({held}); the harm streams are not "
                   "uniformly precision-context-dependent.")

    criteria = [
        {"name": "R2_zworld_positive_control", "load_bearing": False, "passed": R2_MET,
         "seeds_met": n_r2, "of": n_seeds},
        {"name": "C1_harm_s_precision_error_corr", "load_bearing": True, "passed": C1,
         "seeds_met": n_c1, "of": n_seeds},
        {"name": "C2_harm_a_precision_volatility_corr", "load_bearing": True, "passed": C2,
         "seeds_met": n_c2, "of": n_seeds},
    ]

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "backlog_id": BACKLOG_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "claim_ids_tested": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "exp:v3_substrate",
        "evidence_direction": evidence_direction,
        "outcome": outcome,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "non_degenerate": non_degenerate,
        "summary": summary,
        "arm_results": arm_results,
        "per_seed_results": per_seed,
        "collection_diagnostics": collection_diag,
        "per_arm_gate": aggregate,
        "criteria": criteria,
        # C2 SPECIFICITY (reported, NOT gating -- added after observing the
        # one-seed scale check, so deliberately not a post-hoc criterion). ARC-052
        # clause (2) is asserted about the ACCUMULATION stream specifically. If the
        # same precision-vs-volatility coupling appears just as strongly on z_world,
        # the effect is generic (e.g. hazard proximity degrading every stream at
        # once) rather than a property of accumulated threat state. Read C2 against
        # this block before treating it as stream-specific evidence.
        "c2_specificity": {
            "precision_volatility_corr_by_stream": {
                st: [r["precision_volatility_corr"] for r in arm_results
                     if r["arm_id"] == f"{st}__quantile"] for st in STREAMS
            },
            "note": ("clause (2) is stream-specific only if z_harm_a is materially "
                     "more negative than z_world and z_harm_s"),
        },
        "clause_scorability": {
            "control_arm_green": control_arm_green,
            "clause1_arm_scorable": scorable_c1,
            "clause2_arm_scorable": scorable_c2,
            "green_arms": sorted(green),
        },
        "combination_rule": ("PASS iff R2 (z_world positive control) AND C1 AND C2, each "
                             "on a strict majority of seeds. R2 unmet short-circuits to "
                             "substrate_not_ready_requeue / non_contributory rather than "
                             "to any ARC-052 verdict."),
        "registered_thresholds": {
            "R0_NORM_RESIDUAL_FLOOR": R0_NORM_RESIDUAL_FLOOR,
            "R0_MIN_TEST_TRANSITIONS": R0_MIN_TEST_TRANSITIONS,
            "R1_PVAR_RELATIVE_SPREAD_FLOOR": R1_PVAR_RELATIVE_SPREAD_FLOOR,
            "R2_ZWORLD_CONTROL_CORR_FLOOR": R2_ZWORLD_CONTROL_CORR_FLOOR,
            "C1_CORR_FLOOR": C1_CORR_FLOOR,
            "C2_PRECISION_VOL_CORR_MAX": C2_PRECISION_VOL_CORR_MAX,
            "C3_CRPS_IMPROVE_FRAC": C3_CRPS_IMPROVE_FRAC,
            "N_PERM": n_perm,
            "majority_of_seeds": maj,
        },
        "interpretation": {
            "label": label,
            "preconditions": aggregate.get("adjudication_preconditions", []),
            # Each criterion's non-degeneracy is keyed to the gate of the ARM it is
            # actually read off, so one stream failing readiness cannot mark another
            # stream's criterion degenerate (and vice versa).
            "criteria_non_degenerate": arm_criteria_non_degenerate(
                {
                    "z_world__quantile": ["R2_zworld_positive_control"],
                    "z_harm_s__quantile": ["C1_harm_s_precision_error_corr"],
                    "z_harm_a__quantile": ["C2_harm_a_precision_volatility_corr"],
                },
                aggregate,
            ),
            "scope": ("Tests ARC-052 half (B), the context-dependence antecedent, only. "
                      "Half (A) -- precision heads inside the encoders and E3 "
                      "precision-weighting of harm-stream influence -- is NOT built and "
                      "is NOT tested here; it is /implement-substrate work."),
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
            "note": ("SENT-0: V3 harm streams are pre-ethical instrumentation. This run is "
                     "measurement-only over an existing escapable hazard env; no suffering "
                     "accumulator (MECH-219 off), no replay over harm, no self-model."),
        },
        "config": {
            "streams": list(STREAMS), "heads": list(HEADS),
            "stream_dims": dict(STREAM_DIM),
            "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
            "z_harm_dim": Z_HARM_DIM, "z_harm_a_dim": Z_HARM_A_DIM,
            "alpha_world": ALPHA_WORLD, "alpha_self": ALPHA_SELF,
            "harm_history_len": HARM_HISTORY_LEN,
            "head_hidden": HEAD_HIDDEN, "head_lr": HEAD_LR, "head_batch": HEAD_BATCH,
            "epochs": epochs, "train_frac": TRAIN_FRAC,
            "collect_episodes": collect_eps, "steps_per_episode": steps_ep,
            "quantile_levels": list(QUANTILE_LEVELS), "crps_samples": CRPS_SAMPLES,
            "volatility_window": VOLATILITY_WINDOW, "n_perm": n_perm,
            "encoders_trained": False,
            "env_kwargs": dict(ENV_KWARGS),
        },
        "seeds": list(seeds),
        "dry_run": bool(dry_run),
    }

    stamp_recording_core(manifest, config=manifest["config"], seeds=list(seeds),
                         script_path=Path(__file__), started_at=t0,
                         z_goal_stream_stats=_ZG.stats())
    return manifest


def _write_manifest(manifest: Dict[str, Any], *, dry_run: bool = False) -> Path:
    evidence_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    return write_flat_manifest(
        manifest, evidence_dir, dry_run=dry_run,
        config=manifest.get("config"), seeds=manifest.get("seeds"),
        script_path=Path(__file__),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = run_experiment(dry_run=args.dry_run)
    out_path = _write_manifest(manifest, dry_run=bool(args.dry_run))

    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"direction: {manifest['evidence_direction']}", flush=True)
    print(f"saved: {out_path}", flush=True)

    _outcome = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=manifest["run_id"],
        queue_id=QUEUE_ID,
        dry_run=args.dry_run,
    )
