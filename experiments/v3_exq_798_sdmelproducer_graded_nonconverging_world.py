"""V3-EXQ-798 -- SD-MEL-PRODUCER validation: does a non-converging world produce a
GRADED, ABOVE-REFERENCE, and genuinely LEARNABLE waking prediction-error load?

PURPOSE. This validates the ENVIRONMENT TEST-BED landed as SD-MEL-PRODUCER
(2026-07-21), not a claim. claim_ids is deliberately EMPTY.

BACKGROUND -- the producer gap. MECH-180 / INV-050 split into:
  link (i)  real graded novelty -> graded above-reference waking MEL
  link (ii) MEL -> graded offline-phase duration
Link (ii) is BUILT + PROVEN (SD-MEL-CONSUMER; V3-EXQ-718a's injection positive
control: injected MEL [0.6..2.5] -> offline duration [9,13,18,24,30,38], all seeds).
Link (i) has NEVER been demonstrated, and three runs establish that it cannot be
demonstrated with the env_drift knob:
  V3-EXQ-677   C1 manipulation check: high- vs low-novelty mean E1 prediction error
               differed by 8.8e-07 against a 0.01 threshold. NO novelty gradient.
  V3-EXQ-718   C1 novelty-label monotonicity 0/3 seeds.
  V3-EXQ-718a  ecological MEL ~1e-5 -- noise-level, and SCRAMBLED vs novelty level;
               conv_rel_drop ~0.98.
Both autopsies classify this measurement_gap (environment / test-bed producer gap),
explicitly NOT a substrate ceiling and NOT a falsification.

ROOT CAUSE. env_drift_interval fires _drift_hazards(), which only MOVES hazards. The
optimal prediction of a random walk is its mean, so the world-forward model learns
that fast and PE floors at the irreducible noise level. Drift adds sampling NOISE,
not learning LOAD.

MECHANISM UNDER TEST. SD-MEL-PRODUCER periodically re-permutes the action ->
displacement map. E2.world_forward(z_world, a) takes the action as an INPUT, so a
permutation makes the learned forward model systematically wrong until re-learned.
Between shifts the world is deterministic and therefore LEARNABLE; each shift
invalidates learned structure. Load is graded by shift RATE.

WHY THE NOISE ARM IS THE POINT OF THIS EXPERIMENT. Grading observation noise would
ALSO produce a monotone MEL ladder -- but by construction, on any substrate, whether
or not MECH-180 is true. That is the DV-symmetry artifact class
(failure_autopsy_V3-EXQ-604c; the defect that held V3-EXQ-683 on 2026-07-21).
Elevated PE is only learning LOAD if it is REDUCIBLE. So this experiment does not
merely measure whether MEL is graded -- it measures whether the gradient is made of
re-learnable structure, by comparing against an arm whose PE elevation is definitionally
un-learnable.

  ARM_NOISE runs the SAME shift SCHEDULE at world_rule_shift_depth=0 (the schedule
  fires; the rule never changes) plus additive observation noise. That keeps
  steps_since_world_rule_shift defined and the binning directly comparable, while
  guaranteeing there is nothing to re-learn.

DV-SYMMETRY DECLARATION (one line per arm, per the Step 3.5 requirement):
  ARM_NONE / LOW / MED / HIGH -- DV is mean per-step e3 prediction_error, an L2
    residual between E2's ACTION-CONDITIONED prediction and the observed next
    z_world. The manipulation re-permutes the action->displacement map, which
    changes that conditional mapping. The DV is NOT invariant under it: this is a
    conditional-prediction argument, not a distributional one -- even a uniform-random
    policy in a symmetric world leaves the trajectory DISTRIBUTION unchanged while
    breaking the learned per-action mapping the residual is computed against.
  ARM_NOISE -- DV is invariant in the trivial sense: additive observation noise
    raises an L2 residual BY ARITHMETIC. That is deliberate and is precisely its
    role. It is a NEGATIVE CONTROL calibrating what an artifact looks like, and it
    is NOT evidence for anything. It is excluded from the C1 grading ladder.
  Learnability DV (PE binned by steps_since_world_rule_shift, under a TRAINING
    model) -- NOT an arithmetic identity for either arm family: noise yields flat
    bins, re-learnable structure yields decaying bins. This is the discriminating DV.

WHY C4 NEEDS A TRAINING PHASE. A FROZEN model cannot re-learn, so PE after a shift
would never decay regardless of whether the structure is learnable. The MEL
measurement (P1) stays frozen for comparability with 701c/718a; the learnability
probe (P2) enables online world-forward training so decay is meaningful.

MEASUREMENT NOTE. Shift rate SHORTENS episodes (measured 69.0 -> 13.5 steps during
implementation), which would confound a fixed-episode-count window by giving each arm
a different episode-phase mix. So both measurement phases use a fixed STEP budget and
mean_episode_length is reported per arm.

ARMS (5 cells x SEEDS):
  ARM_0_NONE   interval 0   -- no shifts (stable-base reference)
  ARM_1_LOW    interval 60
  ARM_2_MED    interval 25
  ARM_3_HIGH   interval 10
  ARM_4_NOISE  interval 10, depth 0, + observation noise (NEGATIVE CONTROL)

CRITERIA (pre-registered):
  C1 (LOAD-BEARING) grading: MEL monotone NONE < LOW < MED < HIGH and relative
     spread (HIGH/NONE - 1) >= MIN_REL_MEL_SPREAD, on >= SEED_PASS_FRAC of seeds.
  C2 above-reference: every graded arm's MEL exceeds the NONE reference by at least
     MIN_ABS_MEL_SPREAD. This is the exact thing 718a failed (ecological HIGH DV
     72/74 BELOW the OFF baseline 90). The floor is 1e-6, matching the consumer's
     recalibrated mel_relative_floor -- NOT the 701c ABS_MEL_FLOOR=1e-4, which is
     ~5x the entire converged-base signal and structurally unreachable.
  C3 sustained non-convergence: the HIGH arm's PE stays ABOVE the stable-base
     reference (ARM_0_NONE's mean MEL) even in its LONG-AFTER-shift bin -- i.e. the
     world does not settle back within the measurement window. Read against NONE's
     OVERALL mean, not NONE's late bin: NONE has no shifts, so its late bin is
     empty and that comparison would be vacuous.
  C4 learnability (ANTI-ARTIFACT): under training, graded arms show post-shift PE
     DECAY (bin[0-2] -> bin[13+] relative drop >= MIN_LEARN_DECAY) while ARM_NOISE
     does not. Without this, a graded ladder is indistinguishable from graded noise.

INTERPRETATION GRID:
  C1+C2+C3+C4 pass                  -> producer_validated_graded_learnable
  C1+C2 pass, C4 fail               -> producer_graded_but_not_learnable (ARTIFACT --
                                       the ladder is noise-like; do NOT use as a
                                       novelty producer)
  C1 fail, C2 pass                  -> producer_elevates_but_not_graded
  C1+C2 fail                        -> producer_ineffective
  readiness unmet                   -> substrate_not_ready_requeue

PROMOTES / DEMOTES NOTHING. claim_ids=[]; experiment_purpose=diagnostic. A PASS
licenses the SEPARATE, still-gated ecological end-to-end MECH-180 run; it is not
itself evidence for MECH-180.

SLEEP DRIVER: not applicable (no sleep; the consumer is deliberately absent -- 718a's
learning_extracted[1] records that the consumer's DV is a deterministic function of
MEL, so involving it cannot validate a producer).
"""

import sys
import math
import time
import random
import statistics
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from _metrics import check_degeneracy
from experiments._lib.arm_fingerprint import arm_cell
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_798_sdmelproducer_graded_nonconverging_world"
QUEUE_ID = "V3-EXQ-798"
SUPERSEDES = None
CLAIM_IDS: List[str] = []          # validates a TEST-BED, tags no claim. See docstring.
EXPERIMENT_PURPOSE = "diagnostic"

# The R1/R2 readiness predicates and their thresholds are inherited VERBATIM from
# V3-EXQ-701c, which MEASURED both clearing comfortably on this exact instrument and
# this exact recon-only base: conv_seed_fraction 1.0 (3/3 seeds) against
# MIN_REL_CONV_DROP=0.10, and pe_response 1.15 against MIN_REL_PE_RESPONSE=0.25.
# So reachability is established by demonstration on a landed run rather than by a
# synthetic in-script assertion -- these are not hand-written predicates narrower
# than the state they anchor to (the V3-EXQ-778d defect this check exists to catch).
# The other two preconditions are 0/1 structural indicators, reachable by
# construction. Re-derive this if the base recipe or ENV_BASE ever changes.
ANCHOR_REACHABILITY_EXEMPT = (
    "R1/R2 predicates + thresholds inherited verbatim from V3-EXQ-701c, which "
    "measured conv_frac 1.0 (vs 0.10 floor) and pe_response 1.15 (vs 0.25 floor) "
    "on the same instrument and same recon-only base; remaining two preconditions "
    "are 0/1 structural indicators reachable by construction."
)

# -- Design parameters -------------------------------------------------------
SEEDS = [42, 123, 456]
CONV_EPISODES = 60          # P0 world-model convergence on the STABLE base env
STEPS_PER_EPISODE = 90
MEAS_STEPS = 900            # P1 FROZEN MEL window -- a fixed STEP budget (see note)
LEARN_STEPS = 900           # P2 TRAINING learnability window -- also step-budgeted
PROBE_STEPS = 100
PROBE_BATTERY_SIZE = 64
# Progress denominator M. P1/P2 are step-budgeted, so their episode-equivalents are
# what the runner's ETA can meaningfully count.
MEAS_EPISODE_EQUIV = MEAS_STEPS // STEPS_PER_EPISODE
LEARN_EPISODE_EQUIV = LEARN_STEPS // STEPS_PER_EPISODE
EPISODES_PER_RUN = CONV_EPISODES + MEAS_EPISODE_EQUIV + LEARN_EPISODE_EQUIV

# E2 world-forward online training. PRIMARY = reconstruction MSE. RECON-ONLY: the
# SD-056 contrastive auxiliary is a CONFIRMED P0 destabiliser (V3-EXQ-701b ablation),
# so the loss term is omitted. SD056_WEIGHT is retained only for module parity with
# the proven-converging recon-only arm; it is never added to the loss here.
SD056_WEIGHT = 0.05
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats) -------
MIN_REL_PE_RESPONSE = 0.25   # R1: pe_shock at least 25% above pe_stable
MIN_REL_CONV_DROP = 0.10     # R2: per-seed frozen-probe PE drops at least 10%
SEED_PASS_FRAC = 2.0 / 3.0
MIN_REL_MEL_SPREAD = 0.25    # C1: mel[HIGH] at least 25% above mel[NONE]
MONO_TOL = 0.02              # C1: monotonicity slack (relative to mel[NONE])
# C2 floor: 1e-6, matching SD-MEL-CONSUMER's recalibrated mel_relative_floor.
# NOT the 701c ABS_MEL_FLOOR=1e-4, which is ~5x the whole converged-base signal.
MIN_ABS_MEL_SPREAD = 1e-6
MIN_LEARN_DECAY = 0.15       # C4: post-shift PE must drop >=15% from bin[0-2]->bin[13+]
MAX_NOISE_DECAY = 0.10       # C4: the noise control must show < 10% decay

# Observation noise for the NEGATIVE CONTROL arm. Pre-registered; the achieved MEL
# ratio vs ARM_3_HIGH is REPORTED (noise_at_least_as_elevated) rather than tuned, so
# the C4 comparison is fair-or-conservative and no threshold is fitted post-hoc.
NOISE_SIGMA = 0.05

# Bins over steps_since_world_rule_shift. Bin 0 = immediately post-shift (model is
# maximally wrong), bin 3 = long after (model has had time to re-learn IF the
# structure is learnable at all).
SSL_BIN_EDGES = (2, 5, 12)   # -> bins [0-2], [3-5], [6-12], [13+]
N_SSL_BINS = 4

# -- Environment base (functional config; identical to 701c) -----------------
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

# The stable base carries NO hazard drift, so the only non-stationarity in the graded
# arms is the rule shift itself.
STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)
SHOCK_DRIFT = dict(env_drift_interval=1, env_drift_prob=0.8)

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE",  "level": 0, "interval": 0,  "depth": 2, "noise": 0.0,
     "graded": True},
    {"arm_id": "ARM_1_LOW",   "level": 1, "interval": 60, "depth": 2, "noise": 0.0,
     "graded": True},
    {"arm_id": "ARM_2_MED",   "level": 2, "interval": 25, "depth": 2, "noise": 0.0,
     "graded": True},
    {"arm_id": "ARM_3_HIGH",  "level": 3, "interval": 10, "depth": 2, "noise": 0.0,
     "graded": True},
    # NEGATIVE CONTROL: same schedule cadence as HIGH, but depth=0 so the rule never
    # changes (nothing to re-learn), plus observation noise to elevate PE.
    {"arm_id": "ARM_4_NOISE", "level": 4, "interval": 10, "depth": 0,
     "noise": NOISE_SIGMA, "graded": False},
]
GRADED_ORDER = ["ARM_0_NONE", "ARM_1_LOW", "ARM_2_MED", "ARM_3_HIGH"]


def _make_env(seed: int, interval: int, depth: int) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw.update(STABLE_DRIFT)
    kw.update(
        world_rule_shift_enabled=(interval > 0),
        world_rule_shift_interval=interval,
        world_rule_shift_depth=depth,
    )
    return CausalGridWorldV2(seed=seed, **kw)


def _make_probe_env(seed: int, **drift) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw.update(drift)
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Functional waking agent with a trainable world-forward model. NO sleep.

    Config is byte-identical to V3-EXQ-701c so the world_forward MODULE matches the
    proven-converging recon-only ablation arm exactly. The encoder is frozen (only
    agent.e2 trains), which is what makes the frozen-probe battery valid: the
    captured z0/z1 live in a latent space that is CONSTANT across P0 checkpoints.
    """
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


def _apply_obs_noise(obs_dict: Dict[str, Any], sigma: float,
                     gen: Optional[torch.Generator]) -> Dict[str, Any]:
    """NEGATIVE-CONTROL perturbation: additive Gaussian noise on the exteroceptive
    channel. Deliberately UN-LEARNABLE -- there is no structure here to re-learn, so
    a model cannot reduce the PE it induces no matter how long it trains."""
    if sigma <= 0.0:
        return obs_dict
    out = dict(obs_dict)
    ws = _obs(obs_dict, "world_state")
    if ws is not None:
        noise = torch.randn(ws.shape, generator=gen, dtype=ws.dtype) * sigma
        out["world_state"] = ws + noise
    return out


def _sense_latent(agent: REEAgent, obs_dict: Dict[str, Any]):
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
    """One world-forward training step. RECON-ONLY: reconstruction MSE on buffered
    one-step (z0, a, z1) transitions. The SD-056 contrastive auxiliary is omitted
    (confirmed P0 destabiliser, V3-EXQ-701b)."""
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
                noise_sigma: float = 0.0,
                noise_gen: Optional[torch.Generator] = None,
                ) -> Tuple[Optional[float], Dict[str, Any], bool]:
    """One waking step. Returns (per_step_pe, next_obs_dict, done)."""
    # Observation noise (ARM_4_NOISE only) perturbs what the agent PERCEIVES, so it
    # enters at sense time. sigma=0.0 is a pass-through for every other arm.
    latent = _sense_latent(agent, _apply_obs_noise(obs_dict, noise_sigma, noise_gen))

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


def _ssl_bin(steps_since: int) -> int:
    lo, mid, hi = SSL_BIN_EDGES
    if steps_since <= lo:
        return 0
    if steps_since <= mid:
        return 1
    if steps_since <= hi:
        return 2
    return 3


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _run_step_budget(agent: REEAgent, env: CausalGridWorldV2, budget_steps: int,
                     steps_per_episode: int, train: bool,
                     buffer: Optional[Deque],
                     e2_opt: Optional[torch.optim.Optimizer],
                     sample_rng: Optional[random.Random],
                     arm_id: str, seed: int, phase: str, ep_offset: int,
                     noise_sigma: float = 0.0,
                     noise_gen: Optional[torch.Generator] = None,
                     ) -> Dict[str, Any]:
    """Run until budget_steps env steps are consumed, across as many episodes as
    that takes.

    A fixed STEP budget (not a fixed episode count) is REQUIRED here: shift rate
    shortens episodes, so a fixed-episode window would give each arm a different
    total measurement length AND a different episode-phase mix. Returns per-step PE,
    PE binned by steps_since_world_rule_shift, and the episode lengths (reported so
    the confound stays visible)."""
    all_pe: List[float] = []
    bins: Dict[int, List[float]] = {i: [] for i in range(N_SSL_BINS)}
    ep_lens: List[int] = []
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    used = 0
    ep = 0
    while used < budget_steps:
        # Progress index is derived from STEPS consumed, not from the episode
        # counter. Episodes end early under a high shift rate, so an
        # episode-counted index would OVERSHOOT EPISODES_PER_RUN on exactly the
        # arms that matter most. Steps-derived, it tracks the phase budget
        # monotonically and cannot exceed it.
        glob_ep = ep_offset + min(used // max(1, steps_per_episode),
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
                pending_capture_ref, noise_sigma=noise_sigma, noise_gen=noise_gen,
            )
            used += 1
            n_steps += 1
            if pe is not None:
                all_pe.append(pe)
                bins[_ssl_bin(int(env._steps_since_world_rule_shift))].append(pe)
            if done:
                break
        ep_lens.append(n_steps)
        ep += 1
    return {
        "all_pe": all_pe,
        "bin_means": {i: _mean(bins[i]) for i in range(N_SSL_BINS)},
        "bin_counts": {i: len(bins[i]) for i in range(N_SSL_BINS)},
        "mean_episode_length": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "n_episodes": len(ep_lens),
        "n_steps_used": used,
        "shift_count": int(env._world_rule_shift_count),
    }


def _sample_probe_battery(agent: REEAgent, seed: int, size: int,
                          steps: int) -> List[Tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor]]:
    """FIXED held-out one-step transitions captured BEFORE training with the agent's
    own (frozen) encoder, so z0/z1 are CONSTANT across P0 checkpoints."""
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
    """Same battery + frozen encoder -> reads ONLY world_forward improvement, i.e.
    the same one-step world mismatch the e3 prediction_error reads."""
    if not battery:
        return 0.0
    with torch.no_grad():
        z0 = torch.stack([b[0] for b in battery]).to(agent.device)
        a = torch.stack([b[1] for b in battery]).to(agent.device)
        z1 = torch.stack([b[2] for b in battery]).to(agent.device)
        pred = agent.e2.world_forward(z0, a)
        return float(F.mse_loss(pred, z1).item())


def _train_p0_and_probe(seed: int, conv_eps: int, steps: int, probe_size: int,
                        label: str) -> Dict[str, Any]:
    env_stable = _make_probe_env(seed, **STABLE_DRIFT)
    agent = _make_agent(env_stable)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    battery = _sample_probe_battery(agent, seed, probe_size, steps)
    pe_probe_init = _frozen_probe_pe(agent, battery)

    _run_step_budget(agent, env_stable, conv_eps * steps, steps, train=True,
                     buffer=buffer, e2_opt=e2_opt, sample_rng=sample_rng,
                     arm_id=label, seed=seed, phase="P0", ep_offset=0)
    pe_probe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((pe_probe_init - pe_probe_final) / pe_probe_init)
                     if pe_probe_init > 1e-12 else 0.0)
    return {
        "agent": agent, "battery": battery, "e2_opt": e2_opt, "buffer": buffer,
        "sample_rng": sample_rng,
        "pe_probe_init": pe_probe_init, "pe_probe_final": pe_probe_final,
        "conv_rel_drop": conv_rel_drop, "n_probe": len(battery),
    }


def _decay_frac(bin_means: Dict[int, float], bin_counts: Dict[int, int]) -> Optional[float]:
    """Relative PE drop from the immediately-post-shift bin to the long-after bin.
    Positive => the model re-learned => the load was REDUCIBLE (genuine learning
    load). ~0 => nothing to re-learn (noise). None => bins too sparse to read."""
    if bin_counts.get(0, 0) < 5 or bin_counts.get(N_SSL_BINS - 1, 0) < 5:
        return None
    early = bin_means.get(0, 0.0)
    late = bin_means.get(N_SSL_BINS - 1, 0.0)
    if early <= 1e-12:
        return None
    return float((early - late) / early)


def run_cell(arm: Dict[str, Any], seed: int, conv_eps: int, meas_steps: int,
             learn_steps: int, steps: int, probe_steps: int,
             probe_size: int) -> Dict[str, Any]:
    """One (arm, seed) cell: recon-only P0 on the STABLE base -> positive-control
    probes -> P1 FROZEN MEL measurement -> P2 TRAINING learnability probe."""
    arm_id = arm["arm_id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    p0 = _train_p0_and_probe(seed, conv_eps, steps, probe_size, label=arm_id)
    agent = p0["agent"]

    # Positive-control probes on the FROZEN model (readiness R1).
    stable_pe = _run_step_budget(
        agent, _make_probe_env(seed, **STABLE_DRIFT), probe_steps, probe_steps,
        train=False, buffer=None, e2_opt=None, sample_rng=None, arm_id=arm_id,
        seed=seed, phase="PROBE_STABLE", ep_offset=conv_eps)["all_pe"]
    shock_pe = _run_step_budget(
        agent, _make_probe_env(seed, **SHOCK_DRIFT), probe_steps, probe_steps,
        train=False, buffer=None, e2_opt=None, sample_rng=None, arm_id=arm_id,
        seed=seed, phase="PROBE_SHOCK", ep_offset=conv_eps)["all_pe"]
    pe_stable = _mean(stable_pe)
    pe_shock = _mean(shock_pe)
    pe_response_rel = ((pe_shock / pe_stable) - 1.0) if pe_stable > 1e-12 else 0.0

    noise_gen = torch.Generator().manual_seed(seed + 31337)

    # P1 -- FROZEN MEL measurement (comparable to 701c / 718a).
    meas = _run_step_budget(
        agent, _make_env(seed, arm["interval"], arm["depth"]), meas_steps, steps,
        train=False, buffer=None, e2_opt=None, sample_rng=None, arm_id=arm_id,
        seed=seed, phase="P1_MEL", ep_offset=conv_eps + 1,
        noise_sigma=arm["noise"], noise_gen=noise_gen)

    # P2 -- TRAINING learnability probe. A frozen model cannot re-learn, so decay is
    # only meaningful with the world-forward optimiser live.
    learn = _run_step_budget(
        agent, _make_env(seed + 1, arm["interval"], arm["depth"]), learn_steps, steps,
        train=True, buffer=p0["buffer"], e2_opt=p0["e2_opt"],
        sample_rng=p0["sample_rng"], arm_id=arm_id, seed=seed, phase="P2_LEARN",
        ep_offset=conv_eps + 1 + MEAS_EPISODE_EQUIV,
        noise_sigma=arm["noise"], noise_gen=noise_gen)

    print("verdict: PASS", flush=True)
    return {
        "arm_id": arm_id,
        "level": arm["level"],
        "graded": bool(arm["graded"]),
        "seed": seed,
        "world_rule_shift_interval": arm["interval"],
        "world_rule_shift_depth": arm["depth"],
        "obs_noise_sigma": arm["noise"],
        "mel_mean_pe": _mean(meas["all_pe"]),
        "n_meas_pe": len(meas["all_pe"]),
        "meas_mean_episode_length": meas["mean_episode_length"],
        "meas_n_episodes": meas["n_episodes"],
        "meas_shift_count": meas["shift_count"],
        "meas_pe_by_ssl_bin": meas["bin_means"],
        "meas_ssl_bin_counts": meas["bin_counts"],
        "learn_mean_pe": _mean(learn["all_pe"]),
        "learn_mean_episode_length": learn["mean_episode_length"],
        "learn_shift_count": learn["shift_count"],
        "learn_pe_by_ssl_bin": learn["bin_means"],
        "learn_ssl_bin_counts": learn["bin_counts"],
        "learn_decay_frac": _decay_frac(learn["bin_means"], learn["bin_counts"]),
        "pe_stable": pe_stable,
        "pe_shock": pe_shock,
        "pe_response_rel": pe_response_rel,
        "conv_metric": "frozen_probe_battery",
        "p0_recipe": "recon_only",
        "conv_pe_init": p0["pe_probe_init"],
        "conv_pe_final": p0["pe_probe_final"],
        "conv_rel_drop": p0["conv_rel_drop"],
        "n_probe": p0["n_probe"],
    }


def evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    seed_list = sorted({r["seed"] for r in rows})
    n_seeds = len(seed_list)

    def cell(arm_id: str, seed: int) -> Optional[Dict[str, Any]]:
        for r in rows:
            if r["arm_id"] == arm_id and r["seed"] == seed:
                return r
        return None

    # -- Readiness (R1 / R2) on the recon-only converged base -----------------
    per_seed_conv = {s: _mean([r["conv_rel_drop"] for r in rows if r["seed"] == s])
                     for s in seed_list}
    per_seed_resp = {s: _mean([r["pe_response_rel"] for r in rows if r["seed"] == s])
                     for s in seed_list}
    conv_ready = [s for s in seed_list if per_seed_conv[s] >= MIN_REL_CONV_DROP]
    conv_seed_fraction = (len(conv_ready) / n_seeds) if n_seeds else 0.0
    r2_ok = conv_seed_fraction >= SEED_PASS_FRAC
    resp_ready = _mean([per_seed_resp[s] for s in conv_ready]) if conv_ready else 0.0
    r1_ok = resp_ready >= MIN_REL_PE_RESPONSE
    readiness_ok = bool(r2_ok and r1_ok)

    # -- C1 grading (graded arms only; the noise control is EXCLUDED) ---------
    c1_seed_pass: List[bool] = []
    per_seed_mel: Dict[int, Dict[str, float]] = {}
    for s in seed_list:
        vals = []
        rec = {}
        for a in GRADED_ORDER:
            c = cell(a, s)
            v = float(c["mel_mean_pe"]) if c else 0.0
            vals.append(v)
            rec[a] = v
        per_seed_mel[s] = rec
        base = vals[0]
        tol = MONO_TOL * base
        mono = all(vals[i] <= vals[i + 1] + tol for i in range(len(vals) - 1))
        spread = ((vals[-1] - base) / base) if base > 1e-12 else 0.0
        c1_seed_pass.append(bool(mono and spread >= MIN_REL_MEL_SPREAD))
    c1_frac = (sum(c1_seed_pass) / n_seeds) if n_seeds else 0.0
    c1_pass = c1_frac >= SEED_PASS_FRAC

    # -- C2 above-reference ---------------------------------------------------
    c2_seed_pass: List[bool] = []
    for s in seed_list:
        base = per_seed_mel[s][GRADED_ORDER[0]]
        c2_seed_pass.append(all(
            (per_seed_mel[s][a] - base) >= MIN_ABS_MEL_SPREAD
            for a in GRADED_ORDER[1:]))
    c2_frac = (sum(c2_seed_pass) / n_seeds) if n_seeds else 0.0
    c2_pass = c2_frac >= SEED_PASS_FRAC

    # -- C3 sustained non-convergence on HIGH --------------------------------
    # NOTE ON THE COMPARISON. ARM_0_NONE has NO shifts, so its
    # steps_since_world_rule_shift never advances and ALL of its PE lands in bin 0 --
    # its late bin is empty (0.0). Comparing HIGH's late bin against NONE's late bin
    # would therefore be VACUOUS (any HIGH late-bin PE beats an empty 0.0). The
    # meaningful reference is NONE's OVERALL mean, which IS the stable-base level.
    # C3 asks: does HIGH remain above the stable base even LONG AFTER its last shift,
    # i.e. does the world fail to settle back within the window?
    c3_seed_pass = []
    c3_readable = []
    for s in seed_list:
        none_c, high_c = cell("ARM_0_NONE", s), cell("ARM_3_HIGH", s)
        if none_c is None or high_c is None:
            c3_seed_pass.append(False)
            c3_readable.append(False)
            continue
        stable_ref = none_c["mel_mean_pe"]
        late_high = high_c["meas_pe_by_ssl_bin"].get(N_SSL_BINS - 1, 0.0)
        late_n = high_c["meas_ssl_bin_counts"].get(N_SSL_BINS - 1, 0)
        # Fall back to HIGH's overall mean when its late bin is too sparse to read
        # (at interval 10 the long-after bin can legitimately be thin).
        if late_n >= 5:
            c3_readable.append(True)
            c3_seed_pass.append(bool(late_high > stable_ref))
        else:
            c3_readable.append(False)
            c3_seed_pass.append(bool(high_c["mel_mean_pe"] > stable_ref))
    c3_frac = (sum(c3_seed_pass) / n_seeds) if n_seeds else 0.0
    c3_pass = c3_frac >= SEED_PASS_FRAC

    # -- C4 learnability (THE anti-artifact criterion) -----------------------
    graded_decays: List[float] = []
    noise_decays: List[float] = []
    noise_elevated: List[bool] = []
    for s in seed_list:
        hi = cell("ARM_3_HIGH", s)
        no = cell("ARM_4_NOISE", s)
        if hi is not None and hi["learn_decay_frac"] is not None:
            graded_decays.append(hi["learn_decay_frac"])
        if no is not None and no["learn_decay_frac"] is not None:
            noise_decays.append(no["learn_decay_frac"])
        if hi is not None and no is not None:
            noise_elevated.append(bool(no["mel_mean_pe"] >= hi["mel_mean_pe"]))
    graded_decay_mean = _mean(graded_decays) if graded_decays else None
    noise_decay_mean = _mean(noise_decays) if noise_decays else None
    c4_readable = graded_decay_mean is not None and noise_decay_mean is not None
    c4_pass = bool(
        c4_readable
        and graded_decay_mean >= MIN_LEARN_DECAY
        and noise_decay_mean <= MAX_NOISE_DECAY
    )
    noise_at_least_as_elevated = (all(noise_elevated) if noise_elevated else False)

    # -- Preconditions --------------------------------------------------------
    preconditions = [
        {"name": "world_model_converged_frozen_probe",
         "description": "recon-only P0 base converges on the FIXED frozen probe "
                        "battery (readiness R2)",
         "measured": float(min(per_seed_conv.values()) if per_seed_conv else 0.0),
         "threshold": MIN_REL_CONV_DROP, "direction": "lower",
         "control": "worst seed's stable-base P0 conv_rel_drop",
         "met": bool(r2_ok)},
        {"name": "frozen_pe_responds_to_shock",
         "description": "on the converged frozen base, PE rises under a max-novelty "
                        "shock (readiness R1) -- the instrument can move at all",
         "measured": float(resp_ready),
         "threshold": MIN_REL_PE_RESPONSE, "direction": "lower",
         "control": "SHOCK_DRIFT positive control vs STABLE_DRIFT, ready seeds",
         "met": bool(r1_ok)},
        {"name": "learnability_bins_populated",
         "description": "the post-shift and long-after bins both carry enough "
                        "samples for the C4 decay comparison to be readable",
         "measured": float(1.0 if c4_readable else 0.0),
         "threshold": 1.0, "direction": "lower",
         "control": "min 5 samples in bin[0-2] and bin[13+] on ARM_3_HIGH and "
                    "ARM_4_NOISE",
         "met": bool(c4_readable)},
        {"name": "noise_control_at_least_as_elevated",
         "description": "the negative control's MEL is at least the HIGH arm's, so "
                        "the C4 decay comparison is fair-or-conservative rather "
                        "than won by the graded arm simply having more PE",
         "measured": float(1.0 if noise_at_least_as_elevated else 0.0),
         "threshold": 1.0, "direction": "lower",
         "control": "ARM_4_NOISE mel vs ARM_3_HIGH mel, all seeds",
         "met": bool(noise_at_least_as_elevated)},
    ]

    # -- Routing --------------------------------------------------------------
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        note = ("Readiness unmet on the recon-only base (R2 conv_seed_fraction "
                f"{conv_seed_fraction:.2f}, R1 pe_response {resp_ready:.3f}). The "
                "graded-MEL readings are UNINTERPRETABLE without a converged, "
                "novelty-responsive base -- this is a re-queue at an adequate P0, "
                "NOT a verdict on the test-bed.")
    elif c1_pass and c2_pass and c3_pass and c4_pass:
        label = "producer_validated_graded_learnable"
        outcome = "PASS"
        note = ("SD-MEL-PRODUCER produces a graded, above-reference waking MEL load "
                "that is REDUCIBLE (post-shift PE decays under training) while the "
                "matched noise control does not. The test-bed is validated as a "
                "novelty producer. This licenses the SEPARATE, still-gated MECH-180 "
                "ecological end-to-end run; it is NOT itself MECH-180 evidence.")
    elif c1_pass and c2_pass and not c4_pass:
        label = "producer_graded_but_not_learnable"
        outcome = "FAIL"
        note = ("ARTIFACT WARNING: MEL is graded and above reference, but the "
                "elevation does NOT decay under training and is not separable from "
                "the matched noise control. A graded-but-unlearnable ladder is a "
                "DV-symmetry artifact, not a novelty producer -- do NOT use this "
                "test-bed for the ecological MECH-180 run on the strength of C1/C2.")
    elif (not c1_pass) and c2_pass:
        label = "producer_elevates_but_not_graded"
        outcome = "FAIL"
        note = ("The rule shift lifts MEL above the stable-base reference (C2) but "
                "not monotonically in shift rate (C1). The knob is a novelty source "
                "but not yet a GRADED one; re-centre the interval ladder before the "
                "ecological run.")
    else:
        label = "producer_ineffective"
        outcome = "FAIL"
        note = ("The rule shift did not lift MEL above the stable-base reference. "
                "Same outcome class as the env_drift knob it was built to replace.")

    criteria = [
        {"name": "C1_mel_graded_in_shift_rate", "load_bearing": True,
         "passed": bool(c1_pass), "seed_fraction": c1_frac},
        {"name": "C2_mel_above_stable_reference", "load_bearing": False,
         "passed": bool(c2_pass), "seed_fraction": c2_frac},
        {"name": "C3_sustained_non_convergence", "load_bearing": False,
         "passed": bool(c3_pass), "seed_fraction": c3_frac,
         "late_bin_readable_seeds": int(sum(c3_readable))},
        {"name": "C4_load_is_learnable_not_noise", "load_bearing": False,
         "passed": bool(c4_pass),
         "graded_decay_mean": graded_decay_mean,
         "noise_decay_mean": noise_decay_mean},
    ]

    # Non-degeneracy: a criterion is degenerate if its inputs carry no spread.
    mel_all = [r["mel_mean_pe"] for r in rows]
    criteria_non_degenerate = {
        "C1": bool(len(set(round(v, 12) for v in mel_all)) > 1),
        "C2": bool(len(set(round(v, 12) for v in mel_all)) > 1),
        # C3 is non-degenerate only when HIGH's late bin was actually readable on
        # some seed; otherwise it fell back to the coarser overall-mean comparison.
        "C3": bool(any(c3_readable)),
        "C4": bool(c4_readable),
    }

    degeneracy = check_degeneracy({
        "mel_mean_pe": mel_all,
        "learn_decay_frac": [r["learn_decay_frac"] for r in rows
                             if r["learn_decay_frac"] is not None],
    })

    return {
        "outcome": outcome, "label": label, "note": note,
        "evidence_direction": "non_contributory",
        "preconditions": preconditions,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "degeneracy": degeneracy,
        "per_seed_mel": per_seed_mel,
        "per_seed_conv_rel_drop": per_seed_conv,
        "per_seed_pe_response_rel": per_seed_resp,
        "conv_seed_fraction": conv_seed_fraction,
        "readiness_ok": readiness_ok, "r1_ok": r1_ok, "r2_ok": r2_ok,
        "c1_pass": c1_pass, "c1_frac": c1_frac,
        "c2_pass": c2_pass, "c2_frac": c2_frac,
        "c3_pass": c3_pass, "c3_frac": c3_frac,
        "c4_pass": c4_pass,
        "graded_decay_mean": graded_decay_mean,
        "noise_decay_mean": noise_decay_mean,
        "noise_at_least_as_elevated": noise_at_least_as_elevated,
        "mel_per_arm": {a: _mean([r["mel_mean_pe"] for r in rows
                                  if r["arm_id"] == a])
                        for a in [x["arm_id"] for x in ARMS]},
        "mean_episode_length_per_arm": {
            a: _mean([r["meas_mean_episode_length"] for r in rows
                      if r["arm_id"] == a])
            for a in [x["arm_id"] for x in ARMS]},
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [42]
        conv_eps, meas_steps, learn_steps = 2, 40, 40
        steps, probe_steps, probe_size = 12, 12, 8
    else:
        seeds = SEEDS
        conv_eps, meas_steps, learn_steps = CONV_EPISODES, MEAS_STEPS, LEARN_STEPS
        steps, probe_steps, probe_size = (STEPS_PER_EPISODE, PROBE_STEPS,
                                          PROBE_BATTERY_SIZE)

    cell_config = {
        "env_base": ENV_BASE, "conv_eps": conv_eps, "meas_steps": meas_steps,
        "learn_steps": learn_steps, "steps": steps, "probe_steps": probe_steps,
        "probe_size": probe_size, "sd056_weight": SD056_WEIGHT, "e2_lr": E2_LR,
        "p0_loss": "recon_only", "conv_metric": "frozen_probe_battery",
    }

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            slice_cfg = dict(cell_config)
            slice_cfg.update({
                "arm_id": arm["arm_id"],
                "world_rule_shift_interval": arm["interval"],
                "world_rule_shift_depth": arm["depth"],
                "obs_noise_sigma": arm["noise"],
            })
            with arm_cell(seed, config_slice=slice_cfg,
                          script_path=Path(__file__),
                          include_driver_script_in_hash=False) as cell:
                row = run_cell(arm, seed, conv_eps, meas_steps, learn_steps, steps,
                               probe_steps, probe_size)
                cell.stamp(row)
            rows.append(row)

    ev = evaluate(rows)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": ev["outcome"],
        "result": ev["outcome"],
        "evidence_direction": ev["evidence_direction"],
        "evidence_direction_note": ev["note"],
        "sleep_driver_pattern": "not_applicable_no_sleep",
        "interpretation": {
            "label": ev["label"],
            "preconditions": ev["preconditions"],
            "criteria_non_degenerate": ev["criteria_non_degenerate"],
        },
        "criteria": ev["criteria"],
        "summary": {
            "mel_mean_pe_per_arm": ev["mel_per_arm"],
            "mean_episode_length_per_arm": ev["mean_episode_length_per_arm"],
            "per_seed_mel": ev["per_seed_mel"],
            "per_seed_conv_rel_drop": ev["per_seed_conv_rel_drop"],
            "per_seed_pe_response_rel": ev["per_seed_pe_response_rel"],
            "conv_seed_fraction": ev["conv_seed_fraction"],
            "readiness_ok": ev["readiness_ok"],
            "r1_ok": ev["r1_ok"], "r2_ok": ev["r2_ok"],
            "c1_pass": ev["c1_pass"], "c1_frac": ev["c1_frac"],
            "c2_pass": ev["c2_pass"], "c2_frac": ev["c2_frac"],
            "c3_pass": ev["c3_pass"], "c3_frac": ev["c3_frac"],
            "c4_pass": ev["c4_pass"],
            "graded_decay_mean": ev["graded_decay_mean"],
            "noise_decay_mean": ev["noise_decay_mean"],
            "noise_at_least_as_elevated": ev["noise_at_least_as_elevated"],
            "conv_metric": "frozen_probe_battery",
            "p0_recipe": "recon_only",
            "thresholds": {
                "MIN_REL_PE_RESPONSE": MIN_REL_PE_RESPONSE,
                "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
                "SEED_PASS_FRAC": SEED_PASS_FRAC,
                "MIN_REL_MEL_SPREAD": MIN_REL_MEL_SPREAD,
                "MIN_ABS_MEL_SPREAD": MIN_ABS_MEL_SPREAD,
                "MIN_LEARN_DECAY": MIN_LEARN_DECAY,
                "MAX_NOISE_DECAY": MAX_NOISE_DECAY,
                "MONO_TOL": MONO_TOL,
                "NOISE_SIGMA": NOISE_SIGMA,
            },
        },
        "config": {
            "seeds": seeds, "conv_episodes": conv_eps, "meas_steps": meas_steps,
            "learn_steps": learn_steps, "steps_per_episode": steps,
            "probe_steps": probe_steps, "probe_battery_size": probe_size,
            "episodes_per_run": EPISODES_PER_RUN, "arms": ARMS,
            "env_base": ENV_BASE, "stable_drift": STABLE_DRIFT,
            "shock_drift": SHOCK_DRIFT, "p0_loss": "recon_only",
            "sd056_weight": SD056_WEIGHT, "e2_lr": E2_LR,
            "conv_metric": "frozen_probe_battery", "sleep_used": False,
            "substrate": "SD-MEL-PRODUCER",
        },
        "arm_results": rows,
    }
    manifest.update(ev["degeneracy"])

    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=False, config=manifest.get("config"),
        seeds=seeds, script_path=Path(__file__),
    )

    print(f"\nResults written to {out_path}")
    print(f"Outcome: {ev['outcome']}  Label: {ev['label']}")
    print(f"MEL per arm: {ev['mel_per_arm']}")
    print(f"Mean episode length per arm: {ev['mean_episode_length_per_arm']}")
    print(f"C1 {ev['c1_pass']} C2 {ev['c2_pass']} C3 {ev['c3_pass']} "
          f"C4 {ev['c4_pass']}")
    print(f"Learnability decay -- graded: {ev['graded_decay_mean']}  "
          f"noise control: {ev['noise_decay_mean']}")
    return {"outcome": ev["outcome"], "out_path": str(out_path)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Minimal smoke test (1 seed, tiny budgets)")
    args = parser.parse_args()

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE}")
    if args.dry_run:
        print("[DRY RUN] smoke test")

    result = run_experiment(dry_run=args.dry_run)
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    _oc = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_oc if _oc in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["out_path"],
        dry_run=args.dry_run,
    )
