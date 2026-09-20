#!/opt/local/bin/python3
"""
V3-EXQ-1069 -- INV-063 P1 manipulation check, re-measured under V3-EXQ-798a's EXACT P0.

SLEEP DRIVER: none (no sleep pass is configured or fired in any arm; this run measures
              a WAKING quantity only -- see "NO SLEEP" below)

WHY THIS RUN, AND WHY IT IS THE ONLY THING IT DOES (user decision, 2026-09-19T23:48:33Z)
------------------------------------------------------------------------------------------
INV-063's P1 non-degeneracy precondition requires, verbatim: "mean waking MEL strictly
monotone across the four intake arms with relative spread > 0.25 (the V3-EXQ-701c relative
criterion)". Everything else in INV-063's falsifier is downstream of it, because P1 gates the
INTAKE MANIPULATION -- the independent variable. With P1 unmet there is nothing for either
leg's DV to be monotone in, whichever readout leg B uses.

The corpus currently disagrees with itself about whether P1 holds:

  V3-EXQ-798a (LANDED, 2026-07-30)   relative spread 0.483 / 0.659 / 0.685, monotone 3/3
  session probe (2026-09-19, converged base)  0.1369 / 0.1802 / 0.1082, monotone 0/3
  session probe (2026-09-19, V3-EXQ-1060's unconverged base)  0.097 / 0.105 / 0.104

Same knob, same arm settings, same frozen-window definition of MEL. The session probes used a
CHEAPER P0 than 798a, and the disagreement was recorded as unexplained
(REE_assembly/evidence/planning/inv063_p1_gate_and_infonce_tau_20260919.md sections 1 and 1a).
This run settles it by rebuilding 798a's P0 exactly, and does nothing else. It does NOT run
the four-arm ladder, it does NOT touch leg B, and it routes NO verdict on INV-063.

WHAT "798a's EXACT P0" TURNED OUT TO MEAN -- and it is not just the step count
------------------------------------------------------------------------------
Reading 798a's `_make_agent` and `ENV_BASE` at authoring time found the session probes
differed from it in far more than P0 LENGTH, and one of those differences is a documented
root cause for exactly the quantity under test:

  | | 798a (transcribed here) | session probes |
  |---|---|---|
  | `alpha_world` | **0.9** | 0.3 (the REEConfig default) |
  | `self_dim` / `world_dim` | 32 / 32 | 16 / 16 |
  | grid / hazards / resources | 12 / 4 / 5 | 5 / 1 / 1 |
  | harm streams | `use_harm_stream` + `use_affective_harm_stream` | off |
  | z_goal, resource-proximity, benefit heads | on | off |
  | SD-056 contrastive auxiliary | `e2_action_contrastive_enabled` | off |
  | P0 policy | full E3 selection path | random action |
  | P0 length | 5400 steps | 3600 steps |

`alpha_world = 0.3` is the shipped default AND the documented SD-008 root cause for degraded
z_world fidelity; the skill's own review checklist says "experiments depending on z_world
fidelity need >= 0.9". MEL here is `e3_prediction_error`, a world-prediction quantity, so a
low-fidelity z_world plausibly floods it with representation noise and compresses the
intake-driven spread. That is a concrete, testable candidate explanation for the 5x MEL-level
gap (798a 1.5e-5..3.8e-5 against the probes' 8.4e-5..1.7e-4) and for the spread collapse.

This run does not assume that explanation. It transcribes 798a's configuration VERBATIM and
ASSERTS the load-bearing fields back off the live agent (P4 below), so if P1 clears here the
difference is attributable to the configuration, and if it does not, the session probes'
result stands and 798a's becomes the thing needing explanation.

THE ONE DEVIATION FROM 798a, stated up front: P0 IS COMPUTED ONCE PER SEED, NOT PER CELL
------------------------------------------------------------------------------------------
798a calls `_train_p0_and_probe(seed, ...)` inside its per-cell loop. That function takes NO
arm parameter -- its env is `_make_probe_env(seed, **STABLE_DRIFT)` and its training is the
recon-only world-forward pass -- so for a given seed it recomputes a BIT-IDENTICAL agent once
per arm. This run computes it ONCE per seed and hands each of that seed's four arms a
a fresh agent carrying a detached state_dict snapshot of it (NOT
copy.deepcopy -- after P0 the agent holds non-leaf tensors and torch refuses; the
authoring smoke caught that).

This is not an approximation: it is the same computation, performed once instead of four
times. It also makes the four arms EXACTLY seed-matched rather than merely
identically-constructed. The cost argument is why it matters -- 798a's measured 0.931 s/step
puts the per-cell form at ~20.2 h for this grid against ~7.6 h shared, and the user's decision
budgeted ~3 h. The honest revision is reported in the queue entry rather than absorbed by
quietly shrinking the design.

CONSEQUENCE, declared: the cells are NOT independent (they share a P0 agent), so every cell is
emitted `reuse_ineligible` via `extra_ineligible_reasons`. No later run may cite these cells
as a baseline mint.

NO SLEEP. MEL is a WAKING quantity -- mean `e3_prediction_error` over a frozen measurement
window (798a:629-634, :915-921). No sleep flag is set in any arm and `run_sleep_cycle` is
never called, so there is no `SLEEP DRIVER:` pattern to declare beyond the "none" above.

FOUR ARMS, 798a's registered ladder, unchanged
-----------------------------------------------
`world_rule_shift_interval` in {0 (never), 60 (long), 25 (medium), 10 (short)} at
`world_rule_shift_depth = 2` -- 798a's ARM_0_NONE / ARM_1_LOW / ARM_2_MED / ARM_3_HIGH, which
is also INV-063's "{0 / long / medium / short}". Seeds 42 / 123 / 456 (798a's set; seed 44 is
deliberately absent -- recurring reef-config early-death instability, EXQ-539/540,
V3-EXQ-538a). No observation-noise arm: that is P1's matched-PE control, which belongs to the
ladder this run is not running.

TWO PHASES PER CELL, both transcribed from 798a
------------------------------------------------
  P0 (shared per seed)  5400 steps = CONV_EPISODES(60) x STEPS_PER_EPISODE(90), train=True,
                        on the STABLE no-shift env, recon-only world-forward MSE on buffered
                        one-step transitions. The SD-056 contrastive auxiliary is a CONFIRMED
                        P0 destabiliser (V3-EXQ-701b ablation) and is NOT added to the P0 loss
                        -- 798a:606-609, transcribed.
  P1_MEL (per cell)     900 steps = MEAS_STEPS, train=False (FROZEN model), in the ARM'S OWN
                        env at that arm's interval. MEL = mean `e3_prediction_error`.
                        A fixed STEP budget, not an episode count: shift rate shortens
                        episodes, so a fixed-episode window would give each arm a different
                        measurement length AND a different episode-phase mix (798a:731-739).

LOAD-BEARING OUTPUT -- exactly what the user scoped, and nothing else
----------------------------------------------------------------------
  C1_mel_monotone_in_intake   mean MEL non-decreasing across the intake-sorted arms, with
                              798a's MONO_TOL = 0.02 slack relative to the floor arm, on
                              >= SEED_PASS_FRAC (2/3) of seeds.
  C2_mel_relative_spread      (max MEL - min MEL) / max MEL > MIN_REL_MEL_SPREAD (0.25), the
                              V3-EXQ-701c relative criterion INV-063's P1 names, on >= 2/3
                              seeds.
PASS iff BOTH (plain AND). Both are P1 AS REGISTERED; neither threshold is new and neither was
weakened. The absolute ABS_MEL_FLOOR = 1e-4 of 701c is deliberately NOT reused -- INV-063's own
text calls it "structurally unreachable on a converged base and must NOT be reused".

PRE-REGISTERED GO/NO-GO, committed here before the run so it cannot be re-read afterwards
-------------------------------------------------------------------------------------------
  GREEN (PASS) -> INV-063's intake ladder IS gradeable, and leg B then runs on the SD-056
    InfoNCE readout AT A PINNED TEMPERATURE tau = 1e-3. That tau is not free: this session
    measured the readout's headroom below ln(K) on a converged base at 21.6% / 23.9% / 23.45%
    per seed at tau = 1e-3, against 0.5-0.8% at the shipped tau = 0.1, where V3-EXQ-1063's
    pre-registered readability condition failed by 9.66x. Numbers and the distance-matrix
    evidence: GFLAG-0367, and the staged doc sections 3 and 3a. The re-point itself is
    USER-RATIFIED (2026-09-19T21:42:30Z) and PENDING governance application of GFLAG-0364 --
    exactly the posture V3-EXQ-1039a used for its A2 narrowing.
  RED (FAIL) -> INV-063 converts to `substrate_conditional` via its own escape hatch, which
    that claim already calls "a legitimate, useful outcome". No further ladder work is queued.

WHAT THIS RUN CANNOT SETTLE
-----------------------------
- It says nothing about leg B, about either frozen-battery readout, or about C1/C2 of the
  falsifier. It measures P1 and stops.
- A RED result does not distinguish "the ladder cannot grade MEL on this substrate" from "it
  cannot grade it at 798a's configuration either, so 798a's landed 3/3 needs its own
  explanation". Both readings are live on a RED and the manifest says so rather than routing
  one of them.
- P4 asserts the CONFIG is 798a's. It cannot assert that every unlisted default has not
  drifted in `ree_core` since 2026-07-30; `substrate_hash` is what a later reader compares.

DV-SYMMETRY / per-arm declaration (mandatory)
-----------------------------------------------
All four arms share one DV: mean `e3_prediction_error` over a fixed 900-step frozen window --
a genuine per-run measurement, not an argmax/rank statistic (so the monotone-rescaling class
cannot apply) and not a set-aggregate over interchangeable units (so the permutation class
cannot apply). The manipulation is the RATE of action-map re-permutation, which changes the
CONTENT the frozen model is wrong about; a mean is not invariant under it. The criteria read a
RANGE across arms (C2) and an ORDER across arms (C1), and a broadcast constant common to all
arms would cancel in C2's numerator -- which is the correct behaviour here, since a uniform
shift in MEL is exactly what "no intake grading" means and must NOT read as spread.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1069_inv063_p1_gate_798a_p0.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_1069_inv063_p1_gate_798a_p0.py
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments.pack_writer import write_flat_manifest

EXPERIMENT_TYPE = "v3_exq_1069_inv063_p1_gate_798a_p0"
QUEUE_ID = "V3-EXQ-1069"
CLAIM_IDS: List[str] = ["INV-063"]
EXPERIMENT_PURPOSE = "diagnostic"

# Inherited VERBATIM from V3-EXQ-798a along with the rest of its _make_agent block,
# and exempt for the same reason 798a is: the dead stream was adjudicated
# non_contributory 2026-07-27 (the E3 goal term and E1 conditioning are both gated on
# goal_state.is_active(); update_residue, which produces this run's DV, has no goal
# reference; the knobs are identical in every arm, so the inertness is arm-symmetric).
# The lint's own text says wiring it live "populates benefit_rbf_field and un-zeroes
# the SD-025 curiosity bonus -- a behaviour change, not a free wiring fix, and the run
# is then not comparable to its predecessors". Comparability with 798a is the entire
# purpose of this run, so that is disqualifying here rather than merely undesirable.
DEAD_Z_GOAL_STREAM_EXEMPT = (
    "inherited verbatim from V3-EXQ-798a for comparability, which is this run's whole "
    "purpose; the dead stream was adjudicated non_contributory 2026-07-27 and the "
    "knobs are arm-symmetric. Wiring it live would change behaviour relative to the "
    "run whose numbers this exists to reproduce."
)

# R2's predicate and threshold are inherited VERBATIM from V3-EXQ-701c via 798a, and
# have been MEASURED clearing on this exact instrument and recon-only base three
# separate times: 798a's conv_seed_fraction 1.0 (per-seed 0.986 / 0.982 / 0.989)
# against MIN_REL_CONV_DROP = 0.10, V3-EXQ-1063's C1 at 0.9967, and this session's own
# converged-base probes at 0.9980 / 0.9967 / 0.9992. P4 is a config EQUALITY check and
# mel_window_populated is a count >= 1 -- both reachable by construction. R1 is scoped
# OUT of the gate entirely (it certifies the frozen-probe battery, which this run's DV
# never reads) and routes nothing. So no anchor here is narrower than the state it
# anchors to, which is the failure mode the lint exists to catch.
ANCHOR_REACHABILITY_EXEMPT = (
    "R2's predicate/threshold are inherited verbatim from V3-EXQ-701c via 798a and "
    "measured clearing three times on this exact instrument and base (798a 1.0 at "
    "per-seed 0.986/0.982/0.989; V3-EXQ-1063 0.9967; this session 0.9980/0.9967/"
    "0.9992) against a 0.10 floor. P4 is a config equality check whose reachability "
    "was NOT free and is therefore VERIFIED rather than asserted: its first version "
    "read flat attribute names that from_dims routes into nested sub-configs, "
    "reported cfg_ok=False on every run, and was caught by the authoring smoke -- the "
    "shipped version uses dotted paths checked against a live REEConfig and observed "
    "passing. mel_window_populated is a count >= 1 and the restore control is a "
    "torch.equal identity, both reachable by construction. R1 is scoped out of the "
    "gate and routes nothing."
)

# --- 798a ENV_BASE, transcribed verbatim (798a :410-423) --------------------
ENV_BASE: Dict[str, Any] = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    use_proxy_fields=True,
)
# 798a :427-428
STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)
SHOCK_DRIFT = dict(env_drift_interval=1, env_drift_prob=0.8)

# --- 798a design parameters, transcribed (798a :349-360) --------------------
SEEDS: Tuple[int, ...] = (42, 123, 456)
CONV_EPISODES = 60
STEPS_PER_EPISODE = 90
P0_STEPS = CONV_EPISODES * STEPS_PER_EPISODE     # 5400
MEAS_STEPS = 900
PROBE_STEPS = 100
PROBE_BATTERY_SIZE = 64
MEAS_EPISODE_EQUIV = MEAS_STEPS // STEPS_PER_EPISODE
EPISODES_PER_RUN = CONV_EPISODES + MEAS_EPISODE_EQUIV     # 70

# --- 798a E2 world-forward online training (798a :364-372) ------------------
SD056_WEIGHT = 0.05          # module parity only; never added to the P0 loss
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# --- pre-registered thresholds (798a :375-379), NOT derived from run stats --
MIN_REL_PE_RESPONSE = 0.25   # R1 (RECORDED, scoped out of the gate -- see below)
MIN_REL_CONV_DROP = 0.10     # R2 (GATING)
SEED_PASS_FRAC = 2.0 / 3.0
MIN_REL_MEL_SPREAD = 0.25    # C2, the V3-EXQ-701c relative criterion INV-063 P1 names
MONO_TOL = 0.02              # C1 monotonicity slack, relative to the floor arm
EPS = 1e-12

ARMS: Tuple[Tuple[str, int, int], ...] = (
    ("ARM_0_NONE", 0, 0),
    ("ARM_1_LOW", 60, 2),
    ("ARM_2_MED", 25, 2),
    ("ARM_3_HIGH", 10, 2),
)

# P4: the config fields whose drift would make this run NOT 798a's. Asserted back
# off the LIVE agent, not trusted from the builder.
#
# DOTTED PATHS, and that is the point. These are NOT top-level REEConfig attributes
# -- from_dims routes them into nested sub-configs, so a naive
# getattr(agent.config, "alpha_world") returns None. The authoring smoke caught
# exactly that: the first version of this gate read the flat names, got None on all
# three, and reported cfg_ok=False, which would have routed EVERY run to
# substrate_not_ready_requeue forever -- a gate unmeetable by construction, i.e. the
# precise failure mode the readiness-anchor lint exists to catch. Verified at
# authoring time against a live REEConfig.from_dims(...) that these paths resolve and
# carry the 798a values.
REQUIRED_AGENT_CONFIG: Dict[str, Any] = {
    "latent.alpha_world": 0.9,
    "latent.alpha_self": 0.3,
    "latent.self_dim": 32,
    "latent.world_dim": 32,
    "e2.world_dim": 32,
}

ETHICS_PREFLIGHT = {
    "involves_negative_valence": False,
    "involves_suffering_like_state": False,
    "involves_self_model": False,
    "involves_inescapability_or_helplessness": False,
    "involves_offline_replay_over_harm": False,
    "involves_social_mind_or_language": False,
    "involves_human_data_or_clinical_context": False,
    "decision": "allow",
}


# ---------------------------------------------------------------------------
def _finite_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if _finite_or_none(x) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _make_env(seed: int, interval: int, depth: int) -> CausalGridWorldV2:
    """798a :501-510, verbatim."""
    kw = dict(ENV_BASE)
    kw.update(STABLE_DRIFT)
    kw.update(world_rule_shift_enabled=(interval > 0),
              world_rule_shift_interval=interval,
              world_rule_shift_depth=depth)
    return CausalGridWorldV2(seed=seed, **kw)


def _make_probe_env(seed: int, **drift) -> CausalGridWorldV2:
    """798a :512-515, verbatim."""
    kw = dict(ENV_BASE)
    kw.update(drift)
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """798a :518-556, transcribed VERBATIM. The encoder is frozen (only agent.e2
    trains), which is what makes the frozen-probe battery valid: the captured z0/z1
    live in a latent space CONSTANT across P0 checkpoints."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
    )
    return REEAgent(cfg)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None:
        return None
    return v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32)


def _sense_latent(agent: REEAgent, obs_dict: Dict[str, Any]):
    """798a :581-594, verbatim."""
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=_obs(obs_dict, "harm_obs"),
        obs_harm_a=_obs(obs_dict, "harm_obs_a"),
        obs_harm_history=_obs(obs_dict, "harm_history"),
    )


def _sample_class_diverse_batch(buffer: Deque, k: int, rng: random.Random):
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def _e2_train_step(agent: REEAgent, buffer: Deque,
                   optimiser: torch.optim.Optimizer,
                   rng: random.Random) -> Optional[float]:
    """798a :604-626, transcribed. RECON-ONLY: reconstruction MSE on buffered one-step
    transitions. The SD-056 contrastive auxiliary is omitted -- it is a CONFIRMED P0
    destabiliser (V3-EXQ-701b ablation)."""
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    z1_pred = agent.e2.world_forward(z0_K, actions_K)
    recon = F.mse_loss(z1_pred, z1_K)
    recon_val = float(recon.detach().item())
    if not math.isfinite(recon_val):
        return recon_val
    recon.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return recon_val


def _pe_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    """798a :629-634, verbatim. THIS IS MEL."""
    pe = metrics.get("e3_prediction_error")
    if pe is None:
        return None
    val = float(pe.detach().item()) if torch.is_tensor(pe) else float(pe)
    return val if math.isfinite(val) else None


def _step_cycle(agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
                train: bool, buffer: Optional[Deque],
                e2_opt: Optional[torch.optim.Optimizer],
                sample_rng: Optional[random.Random],
                pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
                ) -> Tuple[Optional[float], Dict[str, Any], bool]:
    """798a :637-691, transcribed. One waking step through the FULL E3 selection path
    -- generate_trajectories -> select_action -- which is what makes
    e3_prediction_error exist at all."""
    latent = _sense_latent(agent, obs_dict)

    if train and buffer is not None:
        pend = pending_capture_ref[0]
        if pend is not None:
            z0_prev, a_prev = pend
            z1_obs = latent.z_world.detach().reshape(-1).clone()
            if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()):
                buffer.append((z0_prev, a_prev, z1_obs))
            pending_capture_ref[0] = None

    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device))
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)

    if action is None:
        idx = int(np.random.randint(0, env.action_dim))
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        agent._last_action = action
    if not torch.isfinite(action).all():
        return None, obs_dict, True

    if train and buffer is not None and torch.isfinite(latent.z_world).all():
        pending_capture_ref[0] = (
            latent.z_world.detach().reshape(-1).clone(),
            action.detach().reshape(-1).clone(),
        )
        if e2_opt is not None and sample_rng is not None:
            _e2_train_step(agent, buffer, e2_opt, sample_rng)

    _, harm_signal, done, info, next_obs_dict = env.step(action)
    with torch.no_grad():
        metrics = agent.update_residue(
            harm_signal=float(harm_signal), world_delta=None,
            hypothesis_tag=False, owned=True,
        )
    pe = _pe_from_metrics(metrics)
    return pe, next_obs_dict, bool(done)


def _run_step_budget(agent: REEAgent, env: CausalGridWorldV2, budget_steps: int,
                     steps_per_episode: int, train: bool,
                     buffer: Optional[Deque],
                     e2_opt: Optional[torch.optim.Optimizer],
                     sample_rng: Optional[random.Random],
                     arm_id: str, seed: int, phase: str, ep_offset: int,
                     ) -> Dict[str, Any]:
    """798a :724-793, transcribed. A fixed STEP budget, not a fixed episode count:
    shift rate shortens episodes, so a fixed-episode window would give each arm a
    different total measurement length AND a different episode-phase mix."""
    all_pe: List[float] = []
    ep_lens: List[int] = []
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    used = 0
    while used < budget_steps:
        glob_ep = ep_offset + min(
            used // max(1, steps_per_episode),
            max(0, (budget_steps // max(1, steps_per_episode)) - 1))
        print(f"  [train] {arm_id} seed={seed} {phase} ep {glob_ep+1}/"
              f"{EPISODES_PER_RUN}", flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        pending_capture_ref[0] = None
        n_steps = 0
        for _s in range(steps_per_episode):
            if used >= budget_steps:
                break
            pe, obs_dict, done = _step_cycle(
                agent, env, obs_dict, train, buffer, e2_opt, sample_rng,
                pending_capture_ref)
            used += 1
            n_steps += 1
            if pe is not None:
                all_pe.append(pe)
            if done:
                break
        ep_lens.append(n_steps)
    return {
        "all_pe": all_pe,
        "mean_episode_length": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "n_episodes": len(ep_lens),
        "n_steps_used": used,
        "shift_count": int(getattr(env, "_world_rule_shift_count", 0)),
    }


def _sample_probe_battery(agent: REEAgent, seed: int, size: int, steps: int):
    """798a :798-825, transcribed. FIXED held-out one-step transitions captured
    BEFORE training with the agent's own (frozen) encoder."""
    env = _make_probe_env(seed + 9973, **STABLE_DRIFT)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    rng = np.random.default_rng(seed + 4242)
    for _t in range(steps):
        if len(battery) >= size:
            break
        latent = _sense_latent(agent, obs_dict)
        z_now = latent.z_world.detach().reshape(-1).clone()
        if prev is not None:
            z0_prev, a_prev = prev
            battery.append((z0_prev, a_prev, z_now))
        a_idx = int(rng.integers(0, env.action_dim))
        a_vec = torch.zeros(env.action_dim, dtype=torch.float32)
        a_vec[a_idx] = 1.0
        prev = (z_now, a_vec)
        _, _r, done, _info, obs_dict = env.step(a_idx)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
            prev = None
    return battery


def _frozen_probe_pe(agent: REEAgent, battery) -> float:
    """798a :829-839, verbatim."""
    if not battery:
        return 0.0
    with torch.no_grad():
        z0 = torch.stack([b[0] for b in battery]).to(agent.device)
        a = torch.stack([b[1] for b in battery]).to(agent.device)
        z1 = torch.stack([b[2] for b in battery]).to(agent.device)
        pred = agent.e2.world_forward(z0, a)
        return float(F.mse_loss(pred, z1).item())


def _assert_798a_config(agent: REEAgent) -> Dict[str, Any]:
    """P4: read the load-bearing fields back off the LIVE agent. alpha_world in
    particular is the shipped default 0.3 AND the documented SD-008 root cause for
    degraded z_world fidelity; MEL is a world-prediction quantity, so a silent
    revert to the default would make this run NOT the thing it claims to be."""
    out: Dict[str, Any] = {}
    for path, want in REQUIRED_AGENT_CONFIG.items():
        node: Any = agent.config
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        got = node
        out[path] = {"want": want,
                     "got": (float(got) if isinstance(got, (int, float)) else got),
                     "ok": bool(isinstance(got, (int, float))
                                and float(got) == float(want))}
    return out


# ---------------------------------------------------------------------------
def train_p0(seed: int, p0_steps: int, probe_steps: int, probe_size: int
             ) -> Dict[str, Any]:
    """798a's `_train_p0_and_probe`, which takes NO arm parameter -- see the
    docstring's deviation note. Computed ONCE per seed; each arm gets a deepcopy."""
    env_stable = _make_probe_env(seed, **STABLE_DRIFT)
    agent = _make_agent(env_stable)
    cfg_check = _assert_798a_config(agent)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    battery = _sample_probe_battery(agent, seed, probe_size, probe_steps)
    pe_probe_init = _frozen_probe_pe(agent, battery)

    _run_step_budget(agent, env_stable, p0_steps, STEPS_PER_EPISODE, train=True,
                     buffer=buffer, e2_opt=e2_opt, sample_rng=sample_rng,
                     arm_id="P0_SHARED", seed=seed, phase="P0", ep_offset=0)
    pe_probe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((pe_probe_init - pe_probe_final) / pe_probe_init)
                     if pe_probe_init > EPS else 0.0)

    # R1, RECORDED but SCOPED OUT of this run's gate. 798a treated it as
    # load-bearing because ITS DV routed through the frozen-probe battery; THIS
    # run's DV is MEL (e3_prediction_error) and never reads the battery, so
    # asserting the battery's novelty-responsiveness here would gate on an
    # instrument the run does not use. Recorded because 798a FAILED it at 0.223
    # and a repeat would say the P0 form is not the explanation for that either.
    stable_pe = _run_step_budget(
        agent, _make_probe_env(seed, **STABLE_DRIFT), probe_steps, probe_steps,
        train=False, buffer=None, e2_opt=None, sample_rng=None,
        arm_id="P0_SHARED", seed=seed, phase="PROBE_STABLE",
        ep_offset=CONV_EPISODES)["all_pe"]
    shock_pe = _run_step_budget(
        agent, _make_probe_env(seed, **SHOCK_DRIFT), probe_steps, probe_steps,
        train=False, buffer=None, e2_opt=None, sample_rng=None,
        arm_id="P0_SHARED", seed=seed, phase="PROBE_SHOCK",
        ep_offset=CONV_EPISODES)["all_pe"]
    pe_stable, pe_shock = _mean(stable_pe), _mean(shock_pe)
    pe_response_rel = ((pe_shock / pe_stable) - 1.0) if pe_stable > EPS else 0.0

    print(f"  [P0 seed={seed}] conv_rel_drop={conv_rel_drop:.4f} "
          f"(probe PE {pe_probe_init:.6g} -> {pe_probe_final:.6g})  "
          f"R1 pe_response_rel={pe_response_rel:.4f} "
          f"(stable {pe_stable:.6g} shock {pe_shock:.6g})  cfg_ok="
          f"{all(v['ok'] for v in cfg_check.values())}", flush=True)

    # Snapshot detached so nothing graph-attached rides along (see the deepcopy
    # note at run_cell). state_dict covers parameters AND registered buffers.
    p0_state = {k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in agent.state_dict().items()}

    return {"agent": agent, "p0_state": p0_state, "conv_rel_drop": float(conv_rel_drop),
            "pe_probe_init": pe_probe_init, "pe_probe_final": pe_probe_final,
            "pe_stable": pe_stable, "pe_shock": pe_shock,
            "pe_response_rel": float(pe_response_rel),
            "n_probe": len(battery), "config_check": cfg_check}


def run_cell(arm_id: str, interval: int, depth: int, seed: int,
             p0: Dict[str, Any], meas_steps: int) -> Dict[str, Any]:
    """One (arm, seed) cell: the P1_MEL frozen measurement window, on a fresh agent
    carrying this seed's shared P0 weights.

    NOT copy.deepcopy: after P0 the agent holds non-leaf tensors and torch refuses
    ("Only Tensors created explicitly by the user (graph leaves) support the deepcopy
    protocol"), which the authoring smoke caught. REEAgent is an nn.Module, so the
    sound idiom is a detached state_dict snapshot restored onto a freshly constructed
    agent. Rebuilding rather than restoring in place additionally drops every
    Python-level attribute the previous arm's window mutated (_pe_ema,
    _surprise_write_count, _current_latent, ...), so the four arms are exactly
    symmetric rather than order-dependent. RNG was reset at arm_cell entry, so this
    construction is bit-identical to the one P0 trained.
    """
    print(f"Seed {seed} Condition {arm_id}", flush=True)
    env = _make_env(seed, interval, depth)
    agent = _make_agent(env)
    agent.load_state_dict(p0["p0_state"])
    # Prove the restore landed rather than trusting it: a world-forward parameter
    # must now equal the snapshot bit-for-bit. A silent no-op here would make every
    # arm read an UNTRAINED model and the whole comparison vacuous.
    restore_ok = True
    for _k, _v in agent.state_dict().items():
        if not torch.is_tensor(_v):
            continue
        if not bool(torch.equal(_v, p0["p0_state"][_k])):
            restore_ok = False
            break

    meas = _run_step_budget(
        agent, env, meas_steps, STEPS_PER_EPISODE,
        train=False, buffer=None, e2_opt=None, sample_rng=None,
        arm_id=arm_id, seed=seed, phase="P1_MEL", ep_offset=CONV_EPISODES)

    mel = _mean(meas["all_pe"])
    ok = bool(_finite_or_none(mel) is not None and len(meas["all_pe"]) > 0
              and restore_ok)
    print(f"  {arm_id} seed={seed} interval={interval} shifts={meas['shift_count']} "
          f"n_pe={len(meas['all_pe'])} mean_ep_len={meas['mean_episode_length']:.1f} "
          f"n_episodes={meas['n_episodes']} restore_ok={int(restore_ok)} "
          f"MEL={mel:.6g}", flush=True)
    print(f"verdict: {'PASS' if ok else 'FAIL'}", flush=True)

    return {
        "arm_id": arm_id, "seed": seed,
        "world_rule_shift_interval": interval,
        "world_rule_shift_depth": depth,
        "mel_mean_pe": mel,
        "n_meas_pe": len(meas["all_pe"]),
        "meas_mean_episode_length": meas["mean_episode_length"],
        "meas_n_episodes": meas["n_episodes"],
        "meas_n_steps_used": meas["n_steps_used"],
        "meas_shift_count": meas["shift_count"],
        "conv_rel_drop": p0["conv_rel_drop"],
        "pe_response_rel": p0["pe_response_rel"],
        "p0_state_restore_ok": bool(restore_ok),
        "cell_ok": ok,
    }


# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> Tuple[str, Optional[str]]:
    seeds = list(SEEDS[:1]) if dry_run else list(SEEDS)
    # Dry-run budgets are small ON PURPOSE: this env is ~0.93 s/step (798a's own
    # measured cost), so a smoke at the real budgets would take hours. Every code
    # path is still exercised -- P0 trains, both probes run, all four arms measure.
    p0_steps = 30 if dry_run else P0_STEPS
    meas_steps = 30 if dry_run else MEAS_STEPS
    probe_steps = 10 if dry_run else PROBE_STEPS
    probe_size = 4 if dry_run else PROBE_BATTERY_SIZE

    t0 = time.time()
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    arm_results: List[Dict[str, Any]] = []
    agents_seen: List[REEAgent] = []
    p0_by_seed: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        # P0 is ARM-INDEPENDENT (798a's own _train_p0_and_probe takes no arm), so it
        # is computed once here and restored into a fresh agent per arm. RNG is reset inside each
        # arm_cell below; this P0 is seeded by its own torch/random seeding.
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        p0_by_seed[seed] = train_p0(seed, p0_steps, probe_steps, probe_size)
        agents_seen.append(p0_by_seed[seed]["agent"])

        for arm_id, interval, depth in ARMS:
            cfg_slice = {
                "experiment": EXPERIMENT_TYPE,
                "arm": arm_id,
                "world_rule_shift_interval": interval,
                "world_rule_shift_depth": depth,
                "p0_steps": p0_steps,
                "meas_steps": meas_steps,
                "steps_per_episode": STEPS_PER_EPISODE,
                "e2_lr": E2_LR,
                "env_base": dict(ENV_BASE),
                "stable_drift": dict(STABLE_DRIFT),
                "agent_config": "798a _make_agent verbatim (alpha_world=0.9, dims 32)",
            }
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                          extra_ineligible_reasons=[
                              "shared_p0_agent_across_arms_within_seed"]) as cell:
                row = run_cell(arm_id, interval, depth, seed, p0_by_seed[seed],
                               meas_steps)
                cell.stamp(row)
            rows[(arm_id, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    order = [a for a, _i, _d in ARMS]

    # ---- per-seed P1 evaluation -------------------------------------------
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        mels = [rows[(a, seed)]["mel_mean_pe"] for a in order]
        finite = all(_finite_or_none(m) is not None for m in mels)
        floor = mels[0] if finite else float("nan")
        # C1: non-decreasing across the intake-sorted arms, with 798a's MONO_TOL
        # slack expressed relative to the FLOOR arm (798a :379).
        tol = MONO_TOL * abs(floor) if finite else 0.0
        monotone = bool(finite and all(
            mels[i + 1] >= mels[i] - tol for i in range(len(order) - 1)))
        spread = (((max(mels) - min(mels)) / max(mels))
                  if finite and max(mels) > EPS else float("nan"))
        per_seed.append({
            "seed": seed,
            "mel_per_arm": {a: rows[(a, seed)]["mel_mean_pe"] for a in order},
            "mel_sorted_by_intake": mels,
            "monotone": monotone,
            "mono_tol_abs": tol,
            "relative_spread": spread,
            "spread_clears_floor": bool(_finite_or_none(spread) is not None
                                        and spread > MIN_REL_MEL_SPREAD),
            "conv_rel_drop": rows[(order[0], seed)]["conv_rel_drop"],
            "pe_response_rel": rows[(order[0], seed)]["pe_response_rel"],
            "mean_episode_length_per_arm": {
                a: rows[(a, seed)]["meas_mean_episode_length"] for a in order},
        })

    n_seeds = len(seeds)
    need = math.ceil(SEED_PASS_FRAC * n_seeds)
    n_mono = sum(1 for p in per_seed if p["monotone"])
    n_spread = sum(1 for p in per_seed if p["spread_clears_floor"])
    n_conv = sum(1 for p in per_seed if p["conv_rel_drop"] > MIN_REL_CONV_DROP)
    n_r1 = sum(1 for p in per_seed if p["pe_response_rel"] > MIN_REL_PE_RESPONSE)

    # ---- readiness ---------------------------------------------------------
    # R2 GATES: an unconverged base means P0 did not do its job, and MEL on an
    # untrained world model is not the quantity P1 names.
    # R1 is RECORDED and SCOPED OUT -- see train_p0's comment.
    r2_ok = bool(n_conv >= need)
    all_cells_ok = all(r["cell_ok"] for r in arm_results)
    cfg_ok = all(v["ok"] for p in p0_by_seed.values()
                 for v in p["config_check"].values())

    preconditions = [
        {"name": "R2_p0_converged", "kind": "readiness",
         "description": "per-seed frozen-probe PE drop across P0, 798a R2",
         "measured": float(n_conv), "threshold": float(need), "direction": "lower",
         "comparator": ">=",
         "control": "recon-only P0 on the STABLE no-shift env, 798a's own form",
         "met": r2_ok},
        {"name": "P4_agent_config_is_798a", "kind": "readiness",
         "description": ("alpha_world / self_dim / world_dim read back off the LIVE "
                         "agent; alpha_world's shipped default 0.3 is the documented "
                         "SD-008 root cause for degraded z_world fidelity and MEL is a "
                         "world-prediction quantity"),
         "measured": float(1.0 if cfg_ok else 0.0), "threshold": 1.0,
         "direction": "lower", "comparator": ">=",
         "control": "asserted from the constructed agent, not from the builder",
         "met": bool(cfg_ok)},
        {"name": "p0_weights_restored_into_every_cell", "kind": "readiness",
         "description": ("each cell's agent state_dict equals the shared P0 snapshot "
                         "bit-for-bit; a silent no-op would make every arm read an "
                         "UNTRAINED model and the comparison vacuous"),
         "measured": float(sum(1 for r in arm_results if r["p0_state_restore_ok"])),
         "threshold": float(len(arm_results)), "direction": "lower",
         "comparator": ">=",
         "control": "torch.equal against the snapshot, per tensor, per cell",
         "met": all(r["p0_state_restore_ok"] for r in arm_results)},
        {"name": "mel_window_populated", "kind": "readiness",
         "description": "every cell produced a finite MEL over a non-empty window",
         "measured": float(min(r["n_meas_pe"] for r in arm_results)),
         "threshold": 1.0, "direction": "lower", "comparator": ">=",
         "control": "798a's fixed-step measurement budget", "met": all_cells_ok},
        {"name": "R1_frozen_probe_responds_to_shock_RECORDED_NOT_GATING",
         "kind": "diagnostic",
         "description": ("798a's R1. SCOPED OUT of this run's gate: it certifies the "
                         "frozen-probe BATTERY, and this run's DV is MEL "
                         "(e3_prediction_error), which never reads the battery. "
                         "Recorded because 798a FAILED it at 0.223."),
         "measured": float(n_r1), "threshold": float(need), "direction": "lower",
         "comparator": ">=",
         "control": "SHOCK_DRIFT vs STABLE_DRIFT on the frozen model",
         "met": True,
         "applies": False,
         "scoped_out_reason": ("not meaningful for a MEL DV; asserting it would gate "
                               "on an instrument this run does not use")},
    ]

    ready = bool(r2_ok and cfg_ok and all_cells_ok)

    run_config = {
        "arms": [{"arm_id": a, "interval": i, "depth": d} for a, i, d in ARMS],
        "seeds": list(seeds),
        "p0_steps": p0_steps,
        "conv_episodes": CONV_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "meas_steps": meas_steps,
        "probe_steps": probe_steps,
        "probe_battery_size": probe_size,
        "episodes_per_run": EPISODES_PER_RUN,
        "e2_lr": E2_LR,
        "contrastive_batch_k": CONTRASTIVE_BATCH_K,
        "max_grad_norm": MAX_GRAD_NORM,
        "transition_buffer_max": TRANSITION_BUFFER_MAX,
        "env_base": dict(ENV_BASE),
        "stable_drift": dict(STABLE_DRIFT),
        "shock_drift": dict(SHOCK_DRIFT),
        "min_rel_mel_spread": MIN_REL_MEL_SPREAD,
        "mono_tol": MONO_TOL,
        "seed_pass_frac": SEED_PASS_FRAC,
        "min_rel_conv_drop": MIN_REL_CONV_DROP,
        "min_rel_pe_response": MIN_REL_PE_RESPONSE,
        "p0_shared_per_seed": True,
        "sleep": "NONE -- no sleep flag set, run_sleep_cycle never called",
    }

    base_manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "sleep_driver_pattern": "none (no sleep pass is configured or fired)",
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "arm_results": arm_results,
        "per_seed_p1": per_seed,
        "ethics_preflight": dict(ETHICS_PREFLIGHT),
        "p0_config_check": {str(s): p0_by_seed[s]["config_check"] for s in seeds},
        "scope_note": (
            "Measures INV-063's P1 manipulation check and NOTHING else. It does not "
            "run the four-arm falsifier, does not touch leg B or either "
            "frozen-battery readout, and routes NO verdict on INV-063. "
            "experiment_purpose=diagnostic; evidence_direction=non_contributory."),
        "deviation_from_798a": (
            "P0 is computed ONCE PER SEED and restored into a freshly built agent "
            "per arm via a detached state_dict snapshot, where 798a "
            "recomputes it per cell. 798a's _train_p0_and_probe takes NO arm "
            "parameter -- its env is _make_probe_env(seed, **STABLE_DRIFT) and its "
            "training is arm-independent -- so per-cell recomputation yields a "
            "bit-identical agent. This is the same computation performed once, not "
            "an approximation, and it makes the four arms EXACTLY seed-matched. "
            "Cost is why it matters: 798a's measured 0.931 s/step puts the per-cell "
            "form at ~20.2h for this grid against ~7.6h shared. CONSEQUENCE: the "
            "cells are NOT independent, so every cell is emitted reuse_ineligible "
            "and none may be cited as a baseline mint."),
    }

    def _write(manifest: Dict[str, Any]) -> Optional[str]:
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_path = write_flat_manifest(
            manifest, dry_run=False, config=manifest.get("config"),
            seeds=list(seeds), script_path=Path(__file__), agent=agents_seen)
        print(f"Result written to: {out_path}")
        return str(out_path)

    if not ready:
        unmet = [p["name"] for p in preconditions
                 if p.get("applies", True) and not p["met"]]
        reason = f"readiness unmet: {unmet}"
        print(f"\n[{EXPERIMENT_TYPE}] -> substrate_not_ready_requeue: {reason}",
              flush=True)
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL", "result": "FAIL",
            "evidence_direction": "non_contributory",
            "evidence_direction_note": (
                "Diagnostic readiness unmet; NO P1 verdict is emitted and the "
                "go/no-go below is NOT triggered in either direction. " + reason),
            "non_degenerate": False,
            "degeneracy_reason": "substrate_not_ready: " + reason,
            "interpretation": {"label": "substrate_not_ready_requeue",
                               "preconditions": preconditions,
                               "criteria_non_degenerate": {}},
            "readout": {"substrate_ready": 0, "overall_pass": 0},
            "config": run_config, "elapsed_seconds": elapsed,
        })
        return "FAIL", _write(manifest)

    # ---- the two load-bearing criteria, P1 AS REGISTERED -------------------
    c1 = bool(n_mono >= need)
    c2 = bool(n_spread >= need)
    criteria = [
        {"name": "C1_mel_monotone_in_intake", "load_bearing": True, "passed": c1,
         "measured": float(n_mono), "threshold": float(need), "comparator": ">=",
         "seeds_required": need, "n_seeds": n_seeds, "mono_tol": MONO_TOL},
        {"name": "C2_mel_relative_spread_clears_floor", "load_bearing": True,
         "passed": c2, "measured": float(n_spread), "threshold": float(need),
         "comparator": ">=", "seeds_required": need, "n_seeds": n_seeds,
         "spread_floor": MIN_REL_MEL_SPREAD,
         "per_seed_spread": [p["relative_spread"] for p in per_seed]},
    ]
    p1_pass = bool(c1 and c2)
    outcome = "PASS" if p1_pass else "FAIL"
    label = ("inv063_p1_intake_ladder_gradeable" if p1_pass
             else "inv063_p1_intake_ladder_not_gradeable")

    # Non-degeneracy: a spread/monotonicity verdict is only meaningful if the arms
    # actually differed in what they did -- i.e. the shift counts are graded.
    shifts = {a: [rows[(a, s)]["meas_shift_count"] for s in seeds] for a in order}
    shifts_graded = all(
        _mean(shifts[order[i]]) <= _mean(shifts[order[i + 1]])
        for i in range(len(order) - 1)) and _mean(shifts[order[-1]]) > 0.0
    criteria_non_degenerate = {
        "C1_mel_monotone_in_intake": bool(shifts_graded),
        "C2_mel_relative_spread_clears_floor": bool(shifts_graded),
    }

    go_no_go = {
        "green_means": (
            "INV-063's intake ladder IS gradeable. Leg B then runs on the SD-056 "
            "InfoNCE readout at a PINNED temperature tau = 1e-3 -- measured headroom "
            "below ln(K) 23.88% / 21.61% / 23.45% per seed at that tau on a converged "
            "base, against 0.5-0.8% at the shipped tau = 0.1 where V3-EXQ-1063's "
            "pre-registered readability condition failed by 9.66x (GFLAG-0367; staged "
            "doc sections 3 and 3a). The re-point is USER-RATIFIED 2026-09-19T21:42:30Z "
            "and PENDING governance application of GFLAG-0364."),
        "red_means": (
            "INV-063 converts to substrate_conditional via its own escape hatch, which "
            "that claim already calls 'a legitimate, useful outcome'. No further ladder "
            "work is queued."),
        "red_does_not_distinguish": (
            "A RED does NOT separate 'the ladder cannot grade MEL on this substrate' "
            "from 'it cannot grade it at 798a's configuration either, so 798a's landed "
            "3/3 at spread 0.483/0.659/0.685 needs its own explanation'. Both readings "
            "stay live and this manifest routes neither."),
        "triggered": ("green" if p1_pass else "red"),
    }

    note = (
        f"P1 AS REGISTERED, under V3-EXQ-798a's transcribed configuration and P0. "
        f"Monotone on {n_mono}/{n_seeds} seeds and relative spread clears the "
        f"{MIN_REL_MEL_SPREAD} floor on {n_spread}/{n_seeds} (per-seed spread "
        f"{[round(p['relative_spread'], 4) for p in per_seed]}), against a "
        f"requirement of {need}/{n_seeds} on each. P0 converged on {n_conv}/{n_seeds} "
        f"(798a R2). For comparison: 798a LANDED 0.483/0.659/0.685 monotone 3/3, and "
        f"this session's cheaper-P0 probes measured 0.1369/0.1802/0.1082 monotone 0/3 "
        f"on a converged base. EXPERIMENT_PURPOSE=diagnostic; this routes no verdict "
        f"on INV-063 and measures P1 only."
    )

    print(f"\n[{EXPERIMENT_TYPE}] P1 (the only load-bearing output):", flush=True)
    for p in per_seed:
        print(f"  seed {p['seed']}: MEL {['%.5g' % v for v in p['mel_sorted_by_intake']]} "
              f"monotone={p['monotone']} spread={p['relative_spread']:.4f} "
              f"(floor {MIN_REL_MEL_SPREAD}) conv_rel_drop={p['conv_rel_drop']:.4f}",
              flush=True)
    for c in criteria:
        print(f"  {c['name']}: passed={c['passed']} {c['measured']:.0f}/{n_seeds} "
              f"(need {c['threshold']:.0f})", flush=True)
    print(f"  -> {label} ({outcome}); go/no-go triggered: {go_no_go['triggered']}; "
          f"elapsed={elapsed:.1f}s", flush=True)

    flat: Dict[str, float] = {
        "c1_mel_monotone_seeds": float(n_mono),
        "c2_mel_spread_seeds": float(n_spread),
        "seeds_required": float(need),
        "n_seeds": float(n_seeds),
        "overall_pass": int(p1_pass),
        "r2_p0_converged_seeds": float(n_conv),
        "r1_probe_responds_seeds_RECORDED": float(n_r1),
        "min_conv_rel_drop": float(min(p["conv_rel_drop"] for p in per_seed)),
        "mel_spread_min": float(min(p["relative_spread"] for p in per_seed)),
        "mel_spread_max": float(max(p["relative_spread"] for p in per_seed)),
        "mel_spread_floor": float(MIN_REL_MEL_SPREAD),
        "shifts_graded_across_arms": int(shifts_graded),
    }
    for a in order:
        v = _finite_or_none(_mean([rows[(a, s)]["mel_mean_pe"] for s in seeds]))
        if v is not None:
            flat[f"mel_mean_{a.lower()}"] = v

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": outcome, "result": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "NON_CONTRIBUTORY BY DESIGN, not by failure. experiment_purpose is "
            "diagnostic and this run routes no verdict on INV-063: it measures that "
            "claim's P1 manipulation check under V3-EXQ-798a's transcribed "
            "configuration, so the corpus's disagreement about whether the intake "
            "ladder grades MEL at all can be settled before any falsifier is queued. "
        ) + note,
        "non_degenerate": bool(all(criteria_non_degenerate.values())),
        "degeneracy_reason": (
            None if all(criteria_non_degenerate.values())
            else ("degenerate: the arms' realised shift counts are not graded, so a "
                  "spread/monotonicity verdict says nothing about intake")),
        "interpretation": {
            "label": label,
            "criteria": criteria,
            "combination_rule": (
                "PASS iff BOTH C1 and C2 pass (plain AND). Both are INV-063's P1 as "
                "registered -- C1 monotone with 798a's MONO_TOL=0.02 slack, C2 the "
                "V3-EXQ-701c relative spread > 0.25 -- each on >= 2/3 seeds. Neither "
                "threshold is new and neither was weakened. Readiness R2 and P4 GATE "
                "(unmet -> substrate_not_ready_requeue, no P1 verdict); 798a's R1 is "
                "RECORDED and SCOPED OUT because it certifies the frozen-probe "
                "battery, which this run's DV never reads."),
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions": preconditions,
        },
        "readout": flat,
        "p1_go_no_go": go_no_go,
        "comparison_corpus": {
            "v3_exq_798a_landed": {
                "relative_spread_per_seed": [0.483, 0.659, 0.685],
                "monotone_seeds": 3,
                "note": ("LANDED 2026-07-30 at these identical arm settings; the run "
                         "FAILED its own R1 readiness (0.223 vs 0.25) and was recorded "
                         "substrate_not_ready_requeue, but its C1_mel_graded_in_shift_"
                         "rate passed 3/3."),
            },
            "session_probe_converged_base_20260919": {
                "relative_spread_per_seed": [0.1369, 0.1802, 0.1082],
                "monotone_seeds": 0,
                "note": ("cheaper P0: 3600 random-action steps, 5x5 grid, dims 16, "
                         "alpha_world at the 0.3 default. See deviation table in the "
                         "module docstring."),
            },
            "session_probe_unconverged_base_20260919": {
                "relative_spread_per_seed": [0.097, 0.105, 0.104],
                "monotone_seeds": 1,
            },
        },
        "config": run_config,
        "elapsed_seconds": elapsed,
    })
    return outcome, _write(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke run: 1 seed, short P0/window, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_clean, manifest_path=_manifest_path, dry_run=args.dry_run)
    sys.exit(0)
