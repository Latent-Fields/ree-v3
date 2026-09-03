"""
V3-EXQ-999 -- MECH-161 (LC-NE inverted-U vigilance): does ready-vigilance
detection performance peak at an INTERIOR heartbeat-frequency level, or does
it keep improving monotonically as MECH-093's E3 clock is sped up (the
MECH-026 "set precision/sensitivity high" alternative)?

RED-TEAM: model "fable" (2026-09-03). Verdict BLOCKING on first pass, self-
verified against source (agent.py:5866, clock.py:201-221, and this script's
own _run_eval_level/_agent_tick), FIXED, and re-verified by direct probe
(see "MANIPULATION-REACH FIX" below) -- now CLEAR on the manipulation-reach
axis. Family 2 (bin-coverage) and Family 4 (torch.multinomial non-determinism
on the uncommitted E3 path) were also raised; both are handled by this
script's existing MIN_BIN_COVERAGE_STEPS coverage gate (Step 3.5 self-review)
and by machine-affinity-pinning + reporting committed-fraction (see
"MACHINE-CLASS NOTE" below) rather than by further code changes.

Claims: MECH-161 ("Ready vigilance (MECH-026) requires an arousal regulator
that maintains an optimal sensitivity level on the LC-NE inverted-U curve,
implemented via MECH-093 heartbeat frequency modulation rather than a binary
high/low precision switch.")

EXPERIMENT_PURPOSE = "evidence" (direct falsifier test of MECH-161's SHAPE
claim: interior-optimum vs monotone-improving detection performance as a
function of E3 heartbeat rate).

SUBSTRATE READINESS (Step 2.5/2.5a, confirmed by source read + this script's
own P0 gate):
  - MECH-093 (ree_core/heartbeat/clock.py MultiRateClock.update_e3_rate_from_
    beta / _current_e3_steps) is IMPLEMENTED and WIRED: e3_steps_per_tick is a
    live, continuous knob (HeartbeatConfig.beta_rate_min_steps=5 ..
    beta_rate_max_steps=20), not a binary switch. Confirmed by direct source
    read 2026-09-03 (ree-v3 CLAUDE.md line 17 cross-referenced against the
    actual clock.py implementation).
  - MANIPULATION-REACH FIX (found by Step 4.5 red-team, self-verified against
    source, then fixed here -- read before touching _agent_tick again). The
    ORIGINAL claim in this section was WRONG: the custom _agent_tick harness
    DOES still call agent._e1_tick(latent) every step (e1_steps_per_tick=1,
    so ticks["e1_tick"] is true on virtually every step -- agent.py:373-377),
    and _e1_tick unconditionally calls
    self.clock.update_e3_rate_from_beta(latent_state.z_beta) at agent.py:5866
    with NO gating flag. update_e3_rate_from_beta overwrites
    self._current_e3_steps (clock.py:221) based on live z_beta magnitude.
    Net effect, TRACED THROUGH clock.advance()'s actual read order
    (clock.py:127-160): _run_eval_level forces _current_e3_steps=P only at
    level/episode ENTRY (before step 1's advance() call, so step 1 alone
    reads the forced P correctly) -- but step 1's _e1_tick then overwrites
    _current_e3_steps for step 2's advance() call, and every step thereafter
    for the rest of the episode reads whatever MECH-093's live z_beta dynamics
    computed, NOT the forced P. Verified directly against source (not merely
    trusted from the red-team) 2026-09-03: agent.py:5866 has no `if` guard at
    all, and _run_eval_level (then at the equivalent of the current
    _agent_tick call site) forced _current_e3_steps only once per
    level/episode, never per-step. So for 27 of the 28 planned eval steps per
    episode-that-uses-P, the swept variable P was silently inert and E3 ran
    at whatever period MECH-093 chose -- a manipulation-cannot-reach-DV defect
    that would have made every downstream hit_rate/false_alarm_rate/
    sensitivity(P) reading a measurement of MECH-093's own arousal dynamics,
    not of P.
    THE FIX: _agent_tick now takes an optional force_heartbeat: Optional[int]
    parameter; when set, it re-asserts
    agent.clock._current_e3_steps = int(force_heartbeat) immediately AFTER
    the _e1_tick() call (i.e. after MECH-093 has run and overwritten it, but
    BEFORE the next step's advance() call reads it) so the forced period
    survives from one advance() call to the next. _run_eval_level passes
    force_heartbeat=heartbeat_steps; _train_warmup passes force_heartbeat=None
    (training deliberately runs under MECH-093's live, uncontrolled dynamics
    -- a fixed rate is only the eval-time experimental manipulation, not a
    training precondition). MECH-093 (update_e3_rate_from_beta) still RUNS on
    every eval step under this fix -- nothing bypasses or disables it -- its
    output for that tick is simply overwritten before the next advance() call
    consumes it, which is the intended "operating-point sweep, trained agent
    held fixed" semantics (WORKSPACE_STATE 2026-08-22 "operating-point sweep"
    entries), now actually achieved rather than merely intended. Re-verified
    by a standalone probe (agent.clock._current_e3_steps read at every step
    of a 15-step eval-level rollout, seed 42, P=5 and P=20, matching this
    script's own env config): with force_heartbeat=None (the pre-fix
    behaviour) both P=5 and P=20 traces collapse onto the SAME
    z_beta-governed natural-dynamics trajectory by step 2
    ([5,16,14,13,12,...] vs [20,16,14,12,11,...] -- identical at step 2,
    nearly identical thereafter, small residual divergence from step 4
    onward is action-dependent z_beta feedback, not the forced P); with
    force_heartbeat=P (the fix), both traces stay pinned at the forced value
    on every one of the 15 steps ([5,5,5,...] and [20,20,20,...]).
  - MECH-026 / ARC-016 / ARC-044 (MECH-161's other depends_on) are conceptual/
    architectural claims with no separate substrate gate of their own beyond
    MECH-093's clock and the existing E3 selection + hazard-avoidance
    machinery already exercised by V3-EXQ-981 (claims.yaml: neither
    implementation_phase nor v3_pending set on any of the three).

GOV-REUSE-1 (Step 2.4): decisive readout is
  "interior-vs-extreme heartbeat-level detection sensitivity margin"
  (see DECISIVE READOUT below). claims.yaml MECH-161 carries evidence: []
  (zero prior runs). Searched REE_assembly/scripts/reanalysis_query.py for
  any manifest carrying a hazard-detection-sensitivity-by-heartbeat-rate
  readout on a compatible substrate_hash: none found (no prior script sweeps
  e3_steps_per_tick against a signal-detection DV; V3-EXQ-097/097b/116/505
  probe MECH-093's mechanics directly, not detection performance; V3-EXQ-981
  probes MECH-027's hypervigilance signature via precision/horizon/replay
  forcing, never heartbeat rate). Not recoverable -> run.

RE-DERIVE BRAKE (Step 2.5b): zero prior autopsies exist for MECH-161 (zero
runs total) -- count is 0, below the >=2 threshold. Not braked.

SUBSTRATE-PATH OVERLAP (Step 2.5c): substrate_queue.json has no open
`corrupting`-severity entry whose substrate_paths overlap
ree_core/heartbeat/clock.py, ree_core/agent.py's clock-tick path, or
ree_core/hippocampal/module.py's propose_trajectories (checked 2026-09-03;
grep for corrupting-severity entries found none touching these paths).

WHY THIS DESIGN (avoiding a tautological inverted-U). A naive design that
scores "hit rate" purely from whether a PERIODIC E3 tick lands inside a
fixed-width window after a scheduled hazard injection is degenerate: a
periodic sampler's window-coverage probability is a property of interval
arithmetic (period P vs window width W), identical for "signal" and "noise"
windows unless something ties tick TIMING to the hazard's true onset -- and
MECH-091's phase_reset (a DIFFERENT mechanism) would trivially saturate hit
rate to 1.0 at every P if invoked, confounding MECH-093 with MECH-091 and
making the manipulation reach a phase_reset-dominated DV rather than a
heartbeat-rate-dominated one. This script therefore:
  (a) NEVER calls clock.phase_reset() -- only the periodic/forced-rate path
      is exercised, isolating MECH-093 from MECH-091.
  (b) Does NOT operationalise "detection" as tick-timing-vs-injection-timing
      at all. It reuses the SAME hazard_field_view-based signal-detection
      apparatus already validated in V3-EXQ-981 (_hazard_cells /
      _avoidant_action / _hazard_bin): whether the AGENT'S OWN chosen action
      (from a genuinely TRAINED E3 selection, frozen at eval) matches the
      geometrically-correct "move away from the hazard" action, binned by
      the hazard field's live proximity value at the agent's cell (SAFE <
      0.15, AMBIGUOUS, HIGH >= 0.50 -- thresholds reused verbatim from
      V3-EXQ-981's calibration against this identical env config). This is a
      genuine behavioural readout of a trained policy, not clock-phase
      arithmetic: whether the agent takes the correct evasive action depends
      on how STALE its held action selection is (how many env steps have
      elapsed since E3 last looked), which is a real function of the
      manipulated period P.
  (c) hit_rate(P) = avoidant-action rate in the HIGH hazard bin (correct
      evasion under genuine threat). false_alarm_rate(P) = avoidant-action
      rate in the SAFE hazard bin (evasive action with no real threat
      nearby -- taking flight from nothing).
      sensitivity(P) = hit_rate(P) - false_alarm_rate(P).
      This is NOT guaranteed to be inverted-U by construction: hit_rate could
      plausibly be flat-then-saturating in P (favouring MECH-026's monotone
      story) if staleness never actually degrades correct evasion within the
      swept P range, or false_alarm_rate could stay flat at every P (no
      hypervigilance cost ever appears), either of which would produce a
      monotone or flat sensitivity(P) curve -- a genuine, falsifiable
      alternative outcome, not a guaranteed result.

DV-SYMMETRY INVARIANCE (Step 3 MANDATORY declaration). The manipulated
variable is P = agent.clock._current_e3_steps (an integer period, forced at
each eval level's entry). sensitivity(P) is computed from realised per-step
argmax action selections taken by a frozen, already-trained policy under
env/agent state that depends on P-controlled staleness of the held action.
This is NOT a uniform additive constant (P does not add a constant to E3's
candidate scores; it changes WHEN a fresh score vector is computed at all,
so it directly gates the ACTION SEQUENCE the agent executes, not merely a
score offset that argmax would cancel), NOT a monotone rescaling of an
order-based statistic (hit_rate/false_alarm_rate are behavioural EVENT RATES
over realised trajectories, not a re-ranking of a fixed candidate set), and
NOT a permutation of interchangeable units (env steps within an episode are
not interchangeable -- hazard position and agent position are causally
ordered in time). The manipulation genuinely reaches the DV.

DECISIVE READOUT: sensitivity(P) = hit_rate(P) - false_alarm_rate(P), at each
of 5 pre-registered heartbeat levels P in HEARTBEAT_LEVELS, averaged across
3 seeds. Pre-registered non-monotonicity test (see NON_MONOTONICITY test
below): the level with maximum mean sensitivity must be INTERIOR (neither
the fastest nor the slowest swept level) AND must beat BOTH extreme levels'
mean sensitivity by a margin >= max(MARGIN_ABS_FLOOR, MARGIN_SD_MULT *
pooled_cross_seed_SD). Positive control (DV moves at all): the range of mean
sensitivity across the 5 levels must exceed DV_MOVEMENT_FLOOR, else the run
self-reports non_degenerate=False rather than a false "weakens".

Routing:
  - P0 gate fails (baseline-level avoidance not trained above
    POSITIVE_CONTROL_MARGIN) -> FAIL, interpretation.label
    "substrate_not_ready_requeue", evidence_direction "non_contributory".
  - P0 gate passes but sensitivity(P) is flat across levels (positive control
    fails) -> FAIL, non_degenerate=False, evidence_direction
    "non_contributory".
  - P0 + positive control pass, interior level wins by the pre-registered
    margin -> PASS, evidence_direction "supports" (inverted-U; MECH-161).
  - P0 + positive control pass, best level is an extreme (or interior wins
    but not by the margin) -> FAIL, evidence_direction "weakens" (monotone or
    indeterminate; MECH-026's "higher is always better" story not
    contradicted).

No sleep loop is used (use_sleep_loop left at REEConfig.from_dims default,
i.e. off) -- MECH-161 makes no sleep-dependent claim, so no SLEEP DRIVER
line is needed (Step 3 sleep-driver-pattern rule N/A).

PROGRESS-INSTRUMENTATION NOTE (deliberate, documented deviation): the
decisive readout is a CROSS-SEED aggregate (mean sensitivity per heartbeat
level, averaged over all 3 seeds), so there is exactly ONE authoritative
verdict for the whole run, printed once at the very end of run_experiment
(on every path, including the P0-not-ready early-abort). The queue entry
therefore declares seeds=1, conditions=1 (matching the single verdict line),
NOT seeds=3 -- episodes_per_run is WARMUP_EPISODES (the per-seed training
loop's own denominator). Each seed's training phase still prints its own
"Seed S Condition C" boundary line and "[train] ep N/M" progress (3 boundary
resets before the single final verdict), so the runner's live progress bar
will under/over-count slightly against a literal 1-run expectation -- a
known, accepted UX imprecision, not a correctness issue (the manifest's
outcome/metrics/criteria are unaffected).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import (  # noqa: E402
    check_degeneracy,
    p0_readiness_gate,
    P0NotReady,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_999_mech161_vigilance_inverted_u_heartbeat"
QUEUE_ID = "V3-EXQ-999"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-161"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Seed 44 excluded (documented reef-config early-death instability, CLAUDE.md).
SEEDS = [11, 23, 37]

# ---- Env (identical to V3-EXQ-981's calibrated config -- same hazard-field
# distribution, so the HAZARD_SAFE_MAX/HIGH_MIN thresholds below carry over
# without re-deriving them) --------------------------------------------------
ENV_SIZE = 10
NUM_HAZARDS = 1
NUM_RESOURCES = 2
HAZARD_HARM = 0.05
HAZARD_FIELD_DECAY = 0.5

# ---- Agent dims --------------------------------------------------------------
WORLD_DIM = 32
SELF_DIM = 32

# ---- Training (single warmup per seed, at the HeartbeatConfig canonical
# default rate -- the sweep below only changes EVAL-time heartbeat rate on
# this one frozen trained policy, so "different P" never confounds with
# "different training") -------------------------------------------------------
WARMUP_EPISODES = 150
STEPS_PER_EPISODE = 100
NAV_BIAS = 0.25          # mild bias toward the hazard during warmup so the
                         # agent experiences enough near-hazard states to
                         # learn a meaningful harm-avoidance signal
                         # (identical rationale/value to V3-EXQ-981).

# ---- Heartbeat sweep (Step 3 MANDATORY pre-registration) -------------------
# e3_steps_per_tick values, pre-registered BEFORE any run. Spans
# HeartbeatConfig's documented arousal envelope (beta_rate_min_steps=5 ..
# beta_rate_max_steps=20). TRAIN_HEARTBEAT_STEPS is also HEARTBEAT_LEVELS[2]
# (the middle/interior level) -- the canonical HeartbeatConfig.e3_steps_per_
# tick default (10), so training happens at a level that is also evaluated,
# with no post-hoc cherry-picking.
HEARTBEAT_LEVELS = [5, 8, 10, 15, 20]
TRAIN_HEARTBEAT_STEPS = 10
assert TRAIN_HEARTBEAT_STEPS in HEARTBEAT_LEVELS

# ---- Eval --------------------------------------------------------------------
EVAL_EPISODES_PER_LEVEL = 20
EVAL_STEPS_PER_EPISODE = 100

# ---- Hazard-signal bins (pre-registered; reused verbatim from V3-EXQ-981's
# calibration against this identical env config) -----------------------------
HAZARD_SAFE_MAX = 0.15
HAZARD_HIGH_MIN = 0.50
HAZARD_BIN_SAFE = "SAFE"
HAZARD_BIN_AMBIGUOUS = "AMBIGUOUS"
HAZARD_BIN_HIGH = "HIGH"
MIN_BIN_COVERAGE_STEPS = 5

# ---- Pre-registered thresholds (Step 3 MANDATORY -- fixed before any run) --
POSITIVE_CONTROL_MARGIN = 0.05   # P0 gate: baseline-level (hit - false_alarm)
                                  # must exceed this for avoidance to count as
                                  # "trained at all".
DV_MOVEMENT_FLOOR = 0.03         # positive control: range of mean
                                  # sensitivity across the 5 levels must
                                  # exceed this, else the DV is flat/degenerate.
MARGIN_ABS_FLOOR = 0.03          # non-monotonicity test: absolute floor.
MARGIN_SD_MULT = 1.0             # non-monotonicity test: SD multiplier.


def build_config(env: CausalGridWorldV2) -> REEConfig:
    """ONE config shared by every seed/level -- only agent.clock._current_
    e3_steps is forced at eval-level entry; nothing about agent structure
    changes across levels."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
        use_event_classifier=True,
    )
    cfg.heartbeat.e3_steps_per_tick = TRAIN_HEARTBEAT_STEPS
    return cfg


def config_slice() -> Dict[str, Any]:
    """Exactly what each seed's computation reads -- no acceptance thresholds."""
    return {
        "env": "CausalGridWorldV2",
        "env_size": ENV_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "hazard_harm": HAZARD_HARM,
        "hazard_field_decay": HAZARD_FIELD_DECAY,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "warmup_episodes": WARMUP_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "train_heartbeat_steps": TRAIN_HEARTBEAT_STEPS,
        "heartbeat_levels": HEARTBEAT_LEVELS,
        "eval_episodes_per_level": EVAL_EPISODES_PER_LEVEL,
        "hazard_safe_max": HAZARD_SAFE_MAX,
        "hazard_high_min": HAZARD_HIGH_MIN,
    }


def _random_onehot(action_dim: int, device) -> torch.Tensor:
    v = torch.zeros(1, action_dim, device=device)
    v[0, random.randint(0, action_dim - 1)] = 1.0
    return v


def _hazard_cells(env: CausalGridWorldV2) -> List[Tuple[int, int]]:
    hz = np.argwhere(env.grid == env.ENTITY_TYPES["hazard"])
    return [(int(x), int(y)) for x, y in hz]


def _avoidant_action(env: CausalGridWorldV2, hazard_cells: List[Tuple[int, int]]) -> Optional[int]:
    """The single grid move maximising the dot product with the
    away-from-nearest-hazard vector. Reused verbatim from V3-EXQ-981."""
    if not hazard_cells:
        return None
    ax, ay = int(env.agent_x), int(env.agent_y)
    hx, hy = min(hazard_cells, key=lambda h: abs(h[0] - ax) + abs(h[1] - ay))
    away_dx, away_dy = ax - hx, ay - hy
    best_a, best_score = None, -1e18
    for a, (dx, dy) in env._action_map.items():
        score = dx * away_dx + dy * away_dy
        if score > best_score:
            best_score = score
            best_a = a
    return best_a


def _hazard_bin(value: float) -> str:
    if value < HAZARD_SAFE_MAX:
        return HAZARD_BIN_SAFE
    if value >= HAZARD_HIGH_MIN:
        return HAZARD_BIN_HIGH
    return HAZARD_BIN_AMBIGUOUS


class _TickState:
    def __init__(self) -> None:
        self.last_action: Optional[torch.Tensor] = None
        self.z_self_prev: Optional[torch.Tensor] = None
        self.action_prev: Optional[torch.Tensor] = None
        # MACHINE-CLASS NOTE (red-team Family 4): E3.select() samples via
        # torch.multinomial on the UNCOMMITTED path (result.committed=False),
        # which is not bit-reproducible across machine classes (darwin-arm64
        # vs linux-x86_64/torch versions -- see umbrella CLAUDE.md "Running
        # the test suite" cross-machine-class note). We do not attempt to
        # force committed=True (that would be a different, artificial
        # manipulation); instead we RECORD the committed fraction per level
        # so a cross-machine replication can check whether a result depends
        # on the uncommitted-path RNG stream rather than on the assert-
        # ability of pinning e3.select() to the committed branch only.
        self.n_e3_ticks: int = 0
        self.n_committed_ticks: int = 0


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _agent_tick(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_dict: Dict[str, Any],
    state: _TickState,
    world_dim: int,
    device,
    train: bool,
    force_heartbeat: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One environment tick. Custom harness (mirrors V3-EXQ-981's _agent_tick)
    that calls agent.clock.advance() directly rather than going through
    agent.step()'s normal path -- but note this does NOT keep
    self.clock.update_e3_rate_from_beta from running: agent._e1_tick() is
    still called below (e1_steps_per_tick=1, so this fires on virtually every
    step) and _e1_tick unconditionally calls update_e3_rate_from_beta
    (agent.py:5866, no gating flag), which overwrites
    agent.clock._current_e3_steps based on live z_beta every time it runs.
    See the module docstring "MANIPULATION-REACH FIX" section for the full
    trace and the red-team finding this fixes.

    force_heartbeat: when not None, re-asserts
    agent.clock._current_e3_steps = int(force_heartbeat) immediately AFTER
    the _e1_tick() call each step, so the forced period survives into the
    NEXT step's agent.clock.advance() call regardless of what MECH-093 just
    computed from z_beta. Pass None (the default) to let MECH-093 run
    uncontrolled (the training-time behaviour); pass the swept P value during
    eval (see _run_eval_level).

    Returns (chosen action [1,action_dim], z_world [1,world_dim] detached).
    """
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    ctx = torch.no_grad() if not train else _nullcontext()
    with ctx:
        latent = agent.sense(obs_body, obs_world)
        if state.z_self_prev is not None and state.action_prev is not None:
            agent.record_transition(state.z_self_prev, state.action_prev, latent.z_self.detach())
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, world_dim, device=device)
        )
        if force_heartbeat is not None:
            # Re-assert AFTER _e1_tick (which just overwrote this via
            # MECH-093/update_e3_rate_from_beta) so the forced period is what
            # the NEXT advance() call reads. MECH-093 still ran this tick --
            # nothing is bypassed -- only its output is not allowed to
            # persist past this tick.
            agent.clock._current_e3_steps = int(force_heartbeat)
        if ticks.get("e3_tick", False):
            candidates = agent.hippocampal.propose_trajectories(
                latent.z_world, latent.z_self, e1_prior=e1_prior,
            )
            if candidates:
                result = agent.e3.select(candidates, temperature=1.0)
                state.last_action = result.selected_action.detach()
                state.n_e3_ticks += 1
                if bool(getattr(result, "committed", False)):
                    state.n_committed_ticks += 1
        action = state.last_action
        if action is None:
            action = _random_onehot(env.action_dim, device)
            state.last_action = action

        drive_level = REEAgent.compute_drive_level(obs_body)
        benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
        agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

    state.z_self_prev = latent.z_self.detach()
    state.action_prev = action.detach()
    return action, latent.z_world.detach()


def _train_warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    device,
) -> Dict[str, Any]:
    """Phased P0/P1 warmup: encoder + world-forward + harm-eval head trained
    on .detach()ed latents (no joint E3-through-encoder gradient flow).
    Structure mirrors V3-EXQ-981's _train_warmup (validated pattern)."""
    agent.train()
    state = _TickState()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        state = _TickState()
        z_world_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            action, z_world_curr_pre = _agent_tick(
                agent, env, obs_dict, state, world_dim=world_dim, device=device, train=True,
            )

            # nav_bias: with probability NAV_BIAS, override toward the
            # nearest hazard so training sees enough near-hazard states.
            if random.random() < NAV_BIAS:
                hz = _hazard_cells(env)
                if hz:
                    ax, ay = int(env.agent_x), int(env.agent_y)
                    hx, hy = min(hz, key=lambda h: abs(h[0] - ax) + abs(h[1] - ay))
                    dx, dy = hx - ax, hy - ay
                    best_a, best_score = None, -1e18
                    for a, (adx, ady) in env._action_map.items():
                        score = adx * dx + ady * dy
                        if score > best_score:
                            best_score = score
                            best_a = a
                    if best_a is not None:
                        action = _random_onehot(env.action_dim, device) * 0.0
                        action[0, best_a] = 1.0
                        state.last_action = action
                        state.action_prev = action.detach()

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(
                harm_signal=float(harm_signal), world_delta=None,
                hypothesis_tag=False, owned=True,
            )

            theta_z = agent.theta_buffer.summary()
            if z_world_prev is not None:
                wf_buf.append((z_world_prev.cpu(), state.action_prev.cpu(), z_world_curr_pre.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]
            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat([harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0)
                target = torch.cat([
                    torch.ones(k_p, 1, device=device), torch.zeros(k_n, 1, device=device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr_pre
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}", flush=True)

    return {"total_harm": total_harm, "wf_buf": wf_buf}


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw = torch.cat([wf_buf[i][0] for i in idxs])
        a = torch.cat([wf_buf[i][1] for i in idxs])
        zw1 = torch.cat([wf_buf[i][2] for i in idxs])
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


def _run_eval_level(
    agent: REEAgent,
    env: CausalGridWorldV2,
    heartbeat_steps: int,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    device,
    zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """Run one eval level (fixed E3 heartbeat period) on the already-trained,
    frozen agent. Forces agent.clock._current_e3_steps at level ENTRY and
    resets the phase counter so the new period takes effect immediately
    rather than waiting out whatever period was in force before."""
    agent.clock._current_e3_steps = int(heartbeat_steps)
    agent.clock._e3_phase_step = 0

    agent.eval()
    state = _TickState()
    step_rows: List[Dict[str, Any]] = []
    fatal = 0
    n_e3_ticks_total = 0
    n_committed_ticks_total = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        agent.clock._current_e3_steps = int(heartbeat_steps)
        state = _TickState()

        for step_idx in range(steps_per_episode):
            hazard_cells = _hazard_cells(env)
            hv = obs_dict.get("hazard_field_view", None)
            hazard_value = float(hv[12]) if hv is not None else 0.0
            hbin = _hazard_bin(hazard_value)
            avoidant = _avoidant_action(env, hazard_cells)

            try:
                action, _zw = _agent_tick(
                    agent, env, obs_dict, state, world_dim=world_dim, device=device, train=False,
                    force_heartbeat=heartbeat_steps,
                )
            except Exception:
                fatal += 1
                action = _random_onehot(env.action_dim, device)
                state.last_action = action

            chosen_idx = int(action.argmax(dim=-1).item())
            took_avoidant = bool(avoidant is not None and chosen_idx == avoidant)
            step_rows.append({"hazard_bin": hbin, "took_avoidant": took_avoidant})

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(
                harm_signal=float(harm_signal), world_delta=None,
                hypothesis_tag=False, owned=True,
            )
            if done:
                break

        n_e3_ticks_total += state.n_e3_ticks
        n_committed_ticks_total += state.n_committed_ticks

        print(
            f"  [eval] heartbeat_steps={heartbeat_steps} ep {ep+1}/{num_episodes} "
            f"steps_logged={len(step_rows)}", flush=True,
        )

    zg.observe(agent)

    bins: Dict[str, Dict[str, int]] = {
        HAZARD_BIN_SAFE: {"n": 0, "avoidant": 0},
        HAZARD_BIN_AMBIGUOUS: {"n": 0, "avoidant": 0},
        HAZARD_BIN_HIGH: {"n": 0, "avoidant": 0},
    }
    for row in step_rows:
        b = bins[row["hazard_bin"]]
        b["n"] += 1
        if row["took_avoidant"]:
            b["avoidant"] += 1

    rates = {k: (v["avoidant"] / v["n"] if v["n"] > 0 else 0.0) for k, v in bins.items()}
    coverage_ok = all(v["n"] >= MIN_BIN_COVERAGE_STEPS for v in bins.values())
    hit_rate = rates[HAZARD_BIN_HIGH]
    false_alarm_rate = rates[HAZARD_BIN_SAFE]

    committed_fraction = (
        n_committed_ticks_total / n_e3_ticks_total if n_e3_ticks_total > 0 else None
    )

    return {
        "heartbeat_steps": heartbeat_steps,
        "n_steps": len(step_rows),
        "bins": bins,
        "rates": rates,
        "coverage_ok": coverage_ok,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "sensitivity": hit_rate - false_alarm_rate,
        "fatal_errors": fatal,
        # Red-team Family 4 (machine-class torch.multinomial non-determinism
        # on the uncommitted E3 path): reported, not gated on. A low
        # committed_fraction means more of this level's action selections
        # went through the non-bit-reproducible-across-machine-classes
        # multinomial sampling branch; a cross-machine replication should
        # compare this alongside sensitivity(P).
        "n_e3_ticks": n_e3_ticks_total,
        "n_committed_ticks": n_committed_ticks_total,
        "committed_fraction": committed_fraction,
    }


def run_seed(seed: int, dry_run: bool) -> Dict[str, Any]:
    device = torch.device("cpu")
    reset_all_rng(seed)

    print(f"\nSeed {seed} Condition vigilance_inverted_u_heartbeat_sweep", flush=True)

    env = CausalGridWorldV2(
        seed=seed, size=ENV_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM, hazard_field_decay=HAZARD_FIELD_DECAY,
    )
    cfg = build_config(env)
    agent = REEAgent(cfg).to(device)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) + list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)

    warmup_eps = 5 if dry_run else WARMUP_EPISODES
    warmup_steps = 15 if dry_run else STEPS_PER_EPISODE
    eval_eps = 3 if dry_run else EVAL_EPISODES_PER_LEVEL
    eval_steps = 15 if dry_run else EVAL_STEPS_PER_EPISODE

    train_out = _train_warmup(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_eps, warmup_steps, WORLD_DIM, device,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2={world_forward_r2:.4f}", flush=True)

    zg = ZGoalStreamAccumulator()
    level_results: Dict[int, Dict[str, Any]] = {}
    # Baseline (== TRAIN_HEARTBEAT_STEPS) evaluated FIRST so the P0 gate can
    # abort before the remaining (more expensive) levels run.
    ordered_levels = [TRAIN_HEARTBEAT_STEPS] + [p for p in HEARTBEAT_LEVELS if p != TRAIN_HEARTBEAT_STEPS]
    for p in ordered_levels:
        level_results[p] = _run_eval_level(
            agent, env, p, eval_eps, eval_steps, WORLD_DIM, device, zg,
        )
        if p == TRAIN_HEARTBEAT_STEPS:
            baseline = level_results[p]
            preconditions = p0_readiness_gate([{
                "name": "baseline_avoidance_discrimination_margin",
                "measured": baseline["sensitivity"],
                "threshold": POSITIVE_CONTROL_MARGIN,
                "direction": "lower",
                "control": (
                    f"trained agent evaluated at the training heartbeat rate "
                    f"(P={TRAIN_HEARTBEAT_STEPS}); positive control that "
                    f"hazard-avoidance behaviour was trained at all"
                ),
            }])

    return {
        "seed": seed,
        "world_forward_r2": world_forward_r2,
        "level_results": {p: level_results[p] for p in HEARTBEAT_LEVELS},
        "p0_preconditions": preconditions,
        "agent": agent,
        "zg": zg,
    }


def run_experiment(seeds: List[int], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    arm_rows: List[Dict[str, Any]] = []
    agents: List[REEAgent] = []
    zg_stats_list: List[Dict[str, Any]] = []
    slice_ = config_slice()
    p0_not_ready: Optional[Dict[str, Any]] = None

    for seed in seeds:
        with arm_cell(seed, config_slice=slice_, script_path=Path(__file__),
                      config_slice_declared=True) as cell:
            try:
                result = run_seed(seed, dry_run)
            except P0NotReady as exc:
                p0_not_ready = {"seed": seed, "preconditions": exc.preconditions, "reason": exc.reason}
                row = {"seed": seed, "p0_not_ready": True}
                cell.stamp(row)
                arm_rows.append(row)
                break

            row = {
                "seed": seed,
                "world_forward_r2": result["world_forward_r2"],
                "p0_preconditions": result["p0_preconditions"],
                "level_rates": {p: result["level_results"][p]["rates"] for p in HEARTBEAT_LEVELS},
                "level_hit_rate": {p: result["level_results"][p]["hit_rate"] for p in HEARTBEAT_LEVELS},
                "level_false_alarm_rate": {
                    p: result["level_results"][p]["false_alarm_rate"] for p in HEARTBEAT_LEVELS
                },
                "level_sensitivity": {p: result["level_results"][p]["sensitivity"] for p in HEARTBEAT_LEVELS},
                "level_coverage_ok": {p: result["level_results"][p]["coverage_ok"] for p in HEARTBEAT_LEVELS},
                "level_bins": {p: result["level_results"][p]["bins"] for p in HEARTBEAT_LEVELS},
                "level_fatal_errors": {p: result["level_results"][p]["fatal_errors"] for p in HEARTBEAT_LEVELS},
                # Red-team Family 4 diagnostic (see _run_eval_level docstring):
                # fraction of E3 selections at each level that went through
                # the committed (bit-reproducible) vs uncommitted
                # (torch.multinomial, machine-class-dependent) path.
                "level_committed_fraction": {
                    p: result["level_results"][p]["committed_fraction"] for p in HEARTBEAT_LEVELS
                },
            }
            cell.stamp(row)
            arm_rows.append(row)
            agents.append(result["agent"])
            zg_stats_list.append(result["zg"].stats())

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": QUEUE_ID,
        "arm_results": arm_rows,
        "heartbeat_levels": HEARTBEAT_LEVELS,
        "train_heartbeat_steps": TRAIN_HEARTBEAT_STEPS,
        "positive_control_margin": POSITIVE_CONTROL_MARGIN,
        "dv_movement_floor": DV_MOVEMENT_FLOOR,
        "margin_abs_floor": MARGIN_ABS_FLOOR,
        "margin_sd_mult": MARGIN_SD_MULT,
    }

    if p0_not_ready is not None:
        manifest["outcome"] = "FAIL"
        manifest["evidence_class"] = "experimental"
        manifest["evidence_direction"] = "non_contributory"
        manifest["interpretation"] = {
            "label": "substrate_not_ready_requeue",
            "preconditions": p0_not_ready["preconditions"],
            "criteria_non_degenerate": {"C1_interior_max_margin": False},
        }
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            f"P0 readiness unmet at seed {p0_not_ready['seed']}: {p0_not_ready['reason']}"
        )
        print(
            f"verdict: FAIL  substrate_not_ready_requeue  seed={p0_not_ready['seed']}"
            f"  reason={p0_not_ready['reason']}", flush=True,
        )
        out_path = write_flat_manifest(
            manifest, dry_run=dry_run, config=slice_, seeds=seeds,
            script_path=Path(__file__), started_at=t0, agent=agents,
        )
        return {"outcome": manifest["outcome"], "manifest": manifest, "out_path": out_path}

    # ---- Cross-seed aggregation per level -----------------------------------
    per_level_mean: Dict[int, float] = {}
    per_level_sd: Dict[int, float] = {}
    per_level_values: Dict[int, List[float]] = {}
    for p in HEARTBEAT_LEVELS:
        vals = [row["level_sensitivity"][p] for row in arm_rows if "level_sensitivity" in row]
        per_level_values[p] = vals
        per_level_mean[p] = float(np.mean(vals)) if vals else 0.0
        # ddof=1 (sample SD, Bessel-corrected), not the numpy default ddof=0
        # (population SD) -- with only len(SEEDS)==3 samples, ddof=0
        # understates the SD by sqrt(3/2)~=1.22x, which would understate
        # required_margin below and make the pre-registered non-monotonicity
        # test slightly too easy to pass (red-team Family 3 finding).
        per_level_sd[p] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # ---- Positive control: DV must actually move across levels --------------
    degeneracy = check_degeneracy({
        "level_sensitivity_range": {
            "values": [per_level_mean[p] for p in HEARTBEAT_LEVELS],
            "floor": None,
        },
    })
    sensitivity_range = max(per_level_mean.values()) - min(per_level_mean.values())
    dv_moved = sensitivity_range >= DV_MOVEMENT_FLOOR

    # ---- Coverage adequacy: every level's SAFE/AMBIGUOUS/HIGH bins must clear
    # MIN_BIN_COVERAGE_STEPS on every seed, else hit_rate/false_alarm_rate at that
    # level are noisy artifacts of near-zero-sample bins (0.0-by-default), not a
    # real measurement. A single under-covered (level, seed) cell voids the run.
    coverage_ok_all = all(
        row["level_coverage_ok"][p]
        for row in arm_rows if "level_coverage_ok" in row
        for p in HEARTBEAT_LEVELS
    )

    non_degenerate = bool(degeneracy["non_degenerate"]) and dv_moved and coverage_ok_all
    degeneracy_reason = degeneracy["degeneracy_reason"]
    if not dv_moved:
        degeneracy_reason = (
            f"sensitivity range across levels ({sensitivity_range:.4f}) below "
            f"DV_MOVEMENT_FLOOR ({DV_MOVEMENT_FLOOR})"
            + (f"; {degeneracy_reason}" if degeneracy_reason else "")
        )
    if not coverage_ok_all:
        degeneracy_reason = (
            f"at least one (seed, heartbeat_level) cell has a hazard bin below "
            f"MIN_BIN_COVERAGE_STEPS ({MIN_BIN_COVERAGE_STEPS}) -- hit_rate/"
            f"false_alarm_rate at that cell is not a reliable measurement"
            + (f"; {degeneracy_reason}" if degeneracy_reason else "")
        )

    # ---- Non-monotonicity test (decisive) ------------------------------------
    best_level = max(HEARTBEAT_LEVELS, key=lambda p: per_level_mean[p])
    extremes = [HEARTBEAT_LEVELS[0], HEARTBEAT_LEVELS[-1]]
    is_interior = best_level not in extremes
    pooled_sd = float(np.mean([per_level_sd[best_level]] + [per_level_sd[e] for e in extremes]))
    required_margin = max(MARGIN_ABS_FLOOR, MARGIN_SD_MULT * pooled_sd)
    margins = {e: per_level_mean[best_level] - per_level_mean[e] for e in extremes}
    beats_both_extremes = all(m >= required_margin for m in margins.values())
    c1_pass = bool(non_degenerate and is_interior and beats_both_extremes)

    if not non_degenerate:
        label = "sensitivity_flat_non_degenerate"
        evidence_direction = "non_contributory"
        verdict = "FAIL"
    elif c1_pass:
        label = "interior_optimum_confirmed"
        evidence_direction = "supports"
        verdict = "PASS"
    else:
        label = "monotone_or_indeterminate_no_interior_margin"
        evidence_direction = "weakens"
        verdict = "FAIL"

    manifest["outcome"] = verdict
    manifest["evidence_class"] = "experimental"
    manifest["evidence_direction"] = evidence_direction
    manifest["non_degenerate"] = non_degenerate
    manifest["degeneracy_reason"] = degeneracy_reason
    manifest["metrics"] = {
        "per_level_mean_sensitivity": per_level_mean,
        "per_level_sd_sensitivity": per_level_sd,
        "best_level": best_level,
        "is_interior": is_interior,
        "required_margin": required_margin,
        "margins_vs_extremes": margins,
        "sensitivity_range": sensitivity_range,
    }
    manifest["interpretation"] = {
        "label": label,
        "preconditions": [
            {
                "name": "baseline_avoidance_discrimination_margin",
                "measured": float(np.mean([
                    row["level_sensitivity"][TRAIN_HEARTBEAT_STEPS] for row in arm_rows
                    if "level_sensitivity" in row
                ])) if any("level_sensitivity" in row for row in arm_rows) else float("nan"),
                "threshold": POSITIVE_CONTROL_MARGIN,
                "direction": "lower",
                "met": True,
                "kind": "readiness",
                "control": (
                    "trained agent evaluated at the training heartbeat rate; "
                    "positive control that hazard-avoidance behaviour was trained at all"
                ),
            },
            {
                "name": "sensitivity_dv_movement",
                "measured": sensitivity_range,
                "threshold": DV_MOVEMENT_FLOOR,
                "direction": "lower",
                "met": dv_moved,
                "kind": "readiness",
                "control": "range of mean sensitivity across the 5 pre-registered heartbeat levels",
            },
        ],
        "criteria_non_degenerate": {"C1_interior_max_margin": non_degenerate},
    }
    manifest["criteria"] = [
        {
            "name": "C1_interior_max_margin",
            "load_bearing": True,
            "passed": c1_pass,
            "description": (
                "The heartbeat level with maximum mean sensitivity is INTERIOR "
                "(neither the fastest nor slowest swept level) and beats both "
                "extreme levels' mean sensitivity by >= max(MARGIN_ABS_FLOOR, "
                "MARGIN_SD_MULT * pooled_cross_seed_SD)."
            ),
        },
    ]

    print(
        f"verdict: {verdict}  label={label}  best_level={best_level}"
        f"  is_interior={is_interior}  required_margin={required_margin:.4f}"
        f"  margins_vs_extremes={margins}", flush=True,
    )

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=slice_, seeds=seeds,
        script_path=Path(__file__), started_at=t0, agent=agents,
    )
    if dry_run:
        print(
            "[smoke] per_level_mean_sensitivity: "
            + ", ".join(f"P={p}:{per_level_mean[p]:.4f}" for p in HEARTBEAT_LEVELS),
            flush=True,
        )
    return {"outcome": manifest["outcome"], "manifest": manifest, "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (
        [SEEDS[0]] if args.dry_run else SEEDS
    )

    result = run_experiment(seeds, args.dry_run)
    out_path = result["out_path"]
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['outcome']}", flush=True)
    for k, v in result["manifest"].get("metrics", {}).items():
        print(f"  {k}: {v}", flush=True)

    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(outcome=_outcome_raw, manifest_path=_out_path, dry_run=_dry)
