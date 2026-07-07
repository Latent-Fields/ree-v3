"""
V3-EXQ-718a -- SD-MEL-CONSUMER re-validation: does accumulated waking MEL drive
adaptive sleep cadence, read against MEASURED MEL? (INV-050 third / learning-demand
drive; MECH-180). Test-bed redesign of V3-EXQ-718.

SLEEP DRIVER: manual-cycle-loop (agent.sleep_loop.force_cycle() called once per
cycle in a dedicated MEAS_CYCLES wake-sleep loop). The MEL consumer engages ONLY
through the SleepLoopManager path (force_cycle); a driver that calls
agent.run_sleep_cycle() directly (as V3-EXQ-677 did) BYPASSES the consumer. MEL
accumulates because this driver calls agent.update_residue() on every waking step.

PURPOSE (diagnostic; PROMOTES NOTHING): routed follow-on from
failure_autopsy_V3-EXQ-718_2026-07-07. V3-EXQ-718 established the SD-MEL-CONSUMER is
LIVE + FUNCTIONAL (offline-phase duration factor tracks waking MEL NONE 0.885 / LOW
0.822 / MED 1.263 / HIGH 1.266; C2 PASS 3/3, ON-HIGH DV > OFF-HIGH DV every seed,
OFF pinned cnt_var=0). The FAIL was NOT the consumer: 718's load-bearing C1 tested DV
monotonicity vs the graded NOVELTY LABEL, but the CausalGridWorldV2 drift schedule
(NONE 999/0.00, LOW 15/0.15, MED 6/0.30, HIGH 2/0.50) produced NON-monotone waking
MEL (factor ordering LOW < NONE < MED ~ HIGH). novelty->MEL is the broken link, NOT
MEL->cadence. So 718 accidentally tested the environment's novelty->MEL map rather
than the consumer's MEL->cadence response.

REDESIGN (test-bed, same scientific question; base = v3_exq_718_...):
  PRIMARY FIX -- drop the novelty-LABEL monotonicity DV. The LOAD-BEARING C1 now
  tests that offline-phase duration (DV = cumulative_sws_writes + rem_rollouts) is
  monotone in the MEASURED per-arm mean MEL (arms sorted by measured mean_mel), with
  a measured-MEL-gradient non-degeneracy guard, on >= 2/3 seeds. This reads the
  actual mediating variable (measured MEL), not the environment's novelty label.

  SAME-STATISTIC READINESS via a MEL-INJECTION POSITIVE CONTROL (capability-not-
  ecological). C1 routes on "DV monotone + spread response to graded MEL"; the skill
  (proposal_trivial_prediction_readiness_gate) requires the readiness precondition to
  assert the SAME statistic on a positive control where the gradient is non-degenerate
  BY CONSTRUCTION. ARM_5_INJECT_PC does exactly that: it warms the agent's experience
  buffers with stable-base foraging (so DV is not buffer-capped -- run_sws_schema_pass
  writes min(scaled_sws_steps, n_buf)), then OVERRIDES the waking-MEL accumulator to a
  graded per-cycle target (INJECT_LEVELS x base_ref) right before force_cycle,
  decoupling the MEL from foraging novelty. Per-cycle DV must then be monotone +
  spread in the injected level. Below-floor => the consumer cannot respond to cleanly-
  graded MEL => substrate_not_ready_requeue (NEVER a substrate verdict). This is also
  a clean link-(ii) capability check.

  C2 (control non-degeneracy) kept from 718: the matched-novelty consumer-OFF arm is
  pinned (zero per-cycle count variance) AND ON-HIGH DV > OFF-HIGH DV.

RE-DERIVE BRAKE: RELEASED. INV-050 has >= 2 non_contributory autopsies
(701/701a/701b/701c), but the named upstream substrate SD-MEL-CONSUMER is BUILT +
LANDED (ree-v3 main 909292c; ree_core/sleep/mel_consumer.py; IMPLEMENTED 2026-07-07)
and DEMONSTRABLY FUNCTIONAL (V3-EXQ-718: C2 PASS 3/3, duration factor tracks MEL).
718a is the CORRECT NEXT VALIDATION of a built + functional substrate -- a test-bed
redesign reading MEASURED MEL, NOT a same-selector substrate_ceiling re-derive. The
718 autopsy sets re_derive_brake.fired=false, route_to=queue-experiment. Cites:
SD-MEL-CONSUMER (ree-v3/CLAUDE.md, IMPLEMENTED 2026-07-07);
failure_autopsy_V3-EXQ-718_2026-07-07; failure_autopsy_V3-EXQ-701c_2026-06-30.

DESIGN (template = v3_exq_718_sdmelconsumer_adaptive_cadence_validation.py):
  Base: recon-only P0 world-forward convergence on the STABLE base env (frozen
    encoder; only agent.e2 trains) so per-step e3 PE lives at converged scale. The
    converged frozen-probe PE is the world-model convergence readiness metric;
    conv_rel_drop is a readiness precondition (below-floor -> substrate_not_ready).
  Reference set-point: base_ref = the cell's stable-base measurement-modality MEL,
    calibrated through the SAME path (agent.update_residue -> e3 PE -> consumer
    accumulator) as the measurement MEL. Passed FIXED to the consumer, so the NONE
    arm (stable env == base) yields factor ~ 1.0 and higher-MEL arms scale above it.
  Arms (per seed): 4 graded-novelty ecological arms consumer-ON + 1 matched-novelty
    consumer-OFF control + 1 MEL-injection positive-control:
      ARM_0_NONE_ON  (drift 999/0.00, mel_on=True)
      ARM_1_LOW_ON   (drift 15/0.15,  mel_on=True)
      ARM_2_MED_ON   (drift 6/0.30,   mel_on=True)
      ARM_3_HIGH_ON  (drift 2/0.50,   mel_on=True)
      ARM_4_HIGH_OFF (drift 2/0.50,   mel_on=False)   <- reproduces 677 pinning (C2)
      ARM_5_INJECT_PC(drift 999/0.00, mel_on=True, inject_pc)  <- graded MEL by construction
  Measurement per cell: MEAS_CYCLES wake-sleep cycles. Each cycle runs
    WAKE_EPISODES_PER_CYCLE episodes on the arm's env (agent.update_residue every
    waking step so MEL accumulates + buffers warm), then fires ONE sleep cycle via
    force_cycle. For ARM_5_INJECT_PC, the accumulator is reset + injected to
    INJECT_LEVELS[cyc]*base_ref right before force_cycle. Capture per-cycle
    sws_n_writes + rem_n_rollouts + mel_duration_factor + mel_mean.

DV = cumulative_sws_writes + cumulative_rem_rollouts (the V3-EXQ-677 pinned DV),
     per arm, per seed; plus per-arm MEASURED mean_mel + mel_duration_factor.

Pre-registered thresholds (constants, NOT derived from run stats):
  READINESS R (per seed): (a) frozen-probe conv_rel_drop >= MIN_REL_CONV_DROP AND
    (b) the MEL-injection positive control shows per-cycle DV monotone + spread in the
    injected level (the SAME statistic C1 routes on: graded MEL -> DV). Below-floor
    on >= 2/3 of seeds -> substrate_not_ready_requeue.
  C1 (LOAD-BEARING) MEL drives cadence, monotone in MEASURED MEL:
    the 4 ON arms, sorted ascending by MEASURED mean_mel, have a non-degenerate
    measured-MEL gradient (mel[max] >= mel[min] * (1 + MIN_MEL_SPREAD)) AND
    DV monotone non-decreasing along that order (MONO_TOL relative slack) AND
    DV[max-MEL] >= DV[min-MEL] * (1 + MIN_REL_DV_SPREAD), on >= 2/3 seeds.
  C2 (control non-degeneracy): per seed, ARM_4_HIGH_OFF has ZERO per-cycle count
    variance (pinned) AND DV_on[HIGH] > DV_off[HIGH] (matched novelty), on >= 2/3.

Self-route (NO-weakens map -- a functional substrate is never falsified by a
test-bed gap):
  readiness unmet                       -> substrate_not_ready_requeue (re-queue at an
                                           adequate P0 / reference; NOT a verdict)
  readiness met + C1 PASS + C2 PASS     -> mel_drives_adaptive_sleep_cadence (supports;
                                           /governance clears INV-050 retest +
                                           MECH-180 v3_pending)
  readiness met + C2 PASS + C1 FAIL     -> mel_consumer_capable_ecological_novelty_not_graded
                                           (non_contributory; the injection PC proves the
                                           consumer responds to graded MEL, so a C1 fail
                                           means the ECOLOGICAL foraging did not produce
                                           graded MEL -- a measurement/environment gap, NOT
                                           a substrate ceiling, NOT a falsification. Route:
                                           re-grade the novelty drift schedule.)
  readiness met + C2 FAIL               -> mel_control_degenerate (non_contributory;
                                           OFF control not pinned / not separated)

claim_ids: ["INV-050", "MECH-180"]
experiment_purpose: "diagnostic"
"""

import sys
import json
import math
import argparse
import random
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
from experiments._lib.arm_fingerprint import arm_cell

EXPERIMENT_TYPE = "v3_exq_718a_sdmelconsumer_measured_mel_cadence_validation"
QUEUE_ID = "V3-EXQ-718a"
CLAIM_IDS = ["INV-050", "MECH-180"]
EXPERIMENT_PURPOSE = "diagnostic"

# -- Design parameters -------------------------------------------------------
SEEDS = [42, 123, 456]
CONV_EPISODES = 60            # P0 world-model convergence on the STABLE base env
STEPS_PER_EPISODE = 90
PROBE_BATTERY_SIZE = 64       # FIXED held-out probe battery (frozen-encoder)
CALIB_EPISODES = 3            # stable-base MEL reference-calibration wake pass
MEAS_CYCLES = 6               # wake-sleep cycles per arm
WAKE_EPISODES_PER_CYCLE = 2   # wake episodes per cycle (populate buffers + MEL)
# Progress denominator M (per cell): P0 + calibration + measurement wake episodes.
EPISODES_PER_RUN = (CONV_EPISODES + CALIB_EPISODES
                    + MEAS_CYCLES * WAKE_EPISODES_PER_CYCLE)

# Sleep pass base durations (the scheduler-pinned counts V3-EXQ-677 measured).
SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 10

# MEL consumer config (validation test-bed).
MEL_GAIN = 1.0
FACTOR_MIN = 0.5
FACTOR_MAX = 3.0
MEL_RELATIVE_FLOOR = 1e-6     # relative floor only guards mel/ref against ref ~ 0

# MEL-injection positive control: graded per-cycle MEL as a multiple of base_ref.
# Monotone increasing by construction; all give factor in [FACTOR_MIN, FACTOR_MAX]
# at gain=1.0 (factor == level), so a functional consumer yields monotone per-cycle DV.
INJECT_LEVELS = [0.6, 0.9, 1.2, 1.6, 2.0, 2.5]
INJECT_N_STEPS = 50           # accumulate calls at the target PE (mean == target)

# E2 world-forward online training (recon-only; SD-056 auxiliary OFF at train time).
SD056_WEIGHT = 0.05
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats) -------
MIN_REL_CONV_DROP = 0.10      # R: per-seed frozen-probe PE drops at least 10%
SEED_PASS_FRAC = 2.0 / 3.0    # R / C1 / C2: at least 2/3 of seeds
MIN_MEL_SPREAD = 0.15         # C1: measured mean_mel[max] at least 15% above [min]
MIN_REL_DV_SPREAD = 0.15      # C1 / injection-PC: DV[max] at least 15% above DV[min]
MONO_TOL = 0.05               # C1 / injection-PC: monotonicity slack (relative to DV[min])

# -- Environment base (mirrors V3-EXQ-701c / V3-EXQ-718) ---------------------
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

STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)

# arm_id, novelty level, measurement-env drift, consumer on/off, injection flag.
ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE_ON",   "level": 0, "env_drift_interval": 999, "env_drift_prob": 0.00, "mel_on": True,  "inject_pc": False},
    {"arm_id": "ARM_1_LOW_ON",    "level": 1, "env_drift_interval": 15,  "env_drift_prob": 0.15, "mel_on": True,  "inject_pc": False},
    {"arm_id": "ARM_2_MED_ON",    "level": 2, "env_drift_interval": 6,   "env_drift_prob": 0.30, "mel_on": True,  "inject_pc": False},
    {"arm_id": "ARM_3_HIGH_ON",   "level": 3, "env_drift_interval": 2,   "env_drift_prob": 0.50, "mel_on": True,  "inject_pc": False},
    {"arm_id": "ARM_4_HIGH_OFF",  "level": 3, "env_drift_interval": 2,   "env_drift_prob": 0.50, "mel_on": False, "inject_pc": False},
    {"arm_id": "ARM_5_INJECT_PC", "level": -1, "env_drift_interval": 999, "env_drift_prob": 0.00, "mel_on": True,  "inject_pc": True},
]
# The 4 ecological ON arms (C1 is scored over these, sorted by MEASURED mean_mel).
ON_ECO_ARMS = ["ARM_0_NONE_ON", "ARM_1_LOW_ON", "ARM_2_MED_ON", "ARM_3_HIGH_ON"]


def _make_env(seed: int, env_drift_interval: int, env_drift_prob: float) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw["env_drift_interval"] = env_drift_interval
    kw["env_drift_prob"] = env_drift_prob
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2, mel_on: bool, mel_reference: float) -> REEAgent:
    """Converged-base agent (recon-only e2 training; encoder frozen) + SD-017
    SWS/REM passes + the SleepLoopManager. When mel_on, the SD-MEL-CONSUMER is
    enabled with a FIXED reference set-point. Base config is byte-identical to the
    V3-EXQ-718 recipe."""
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
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        surprise_gated_replay=True,
        # SD-017 sleep passes + SleepLoopManager (no aggregation cluster needed).
        use_sleep_loop=True,
        sleep_loop_episodes_K=10**9,   # never auto-fire; we drive via force_cycle
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        # SD-MEL-CONSUMER (GAP-5b) -- fixed reference set-point.
        use_mel_consumer=bool(mel_on),
        mel_gain=MEL_GAIN,
        mel_reference=float(mel_reference),
        mel_reference_mode="fixed",
        mel_duration_factor_min=FACTOR_MIN,
        mel_duration_factor_max=FACTOR_MAX,
        mel_relative_floor=MEL_RELATIVE_FLOOR,
        mel_scale_sws=True,
        mel_scale_rem=True,
        use_mel_entry=False,
    )
    return REEAgent(cfg)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


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


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int, rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen:
            seen[cls] = tup
        if len(seen) >= k:
            break
    if len(seen) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen.values())
    picked = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked:
            continue
        samples.append(tup)
        picked.add(id(tup))
    return samples


def _e2_train_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer, rng: random.Random,
) -> Optional[float]:
    """One recon-only P0 world-forward training step (reconstruction MSE)."""
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


def _waking_step(
    agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
    train: bool, buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer], sample_rng: Optional[random.Random],
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
) -> Tuple[Dict[str, Any], bool]:
    """One waking step. Always calls agent.update_residue() (hypothesis_tag=False)
    so the MEL consumer accumulates per-step e3 prediction error. Returns
    (next_obs_dict, done)."""
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
        return obs_dict, True

    if train and buffer is not None and torch.isfinite(latent.z_world).all():
        pending_capture_ref[0] = (
            latent.z_world.detach().reshape(-1).clone(),
            action.detach().reshape(-1).clone(),
        )
        if e2_opt is not None and sample_rng is not None:
            _e2_train_step(agent, buffer, e2_opt, sample_rng)

    _, harm_signal, done, info, next_obs_dict = env.step(action)
    with torch.no_grad():
        agent.update_residue(
            harm_signal=float(harm_signal), world_delta=None,
            hypothesis_tag=False, owned=True,
        )
    return next_obs_dict, bool(done)


def _run_wake_window(
    agent: REEAgent, env: CausalGridWorldV2, n_episodes: int, steps: int,
    train: bool, buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer], sample_rng: Optional[random.Random],
    ep_offset: int, arm_id: str, seed: int,
) -> None:
    """Run n_episodes of waking on env. During P0 (train=True) trains e2 recon-only.
    During measurement (train=False) just drives the agent + accumulates MEL + warms
    the agent's experience buffers."""
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    for ep in range(n_episodes):
        glob_ep = ep_offset + ep
        if (glob_ep % 10 == 0) or (glob_ep == EPISODES_PER_RUN - 1):
            print(f"  [train] {arm_id} seed={seed} ep {glob_ep+1}/{EPISODES_PER_RUN}",
                  flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        pending_capture_ref[0] = None
        for _step in range(steps):
            obs_dict, done = _waking_step(
                agent, env, obs_dict, train, buffer, e2_opt, sample_rng,
                pending_capture_ref,
            )
            if done:
                break


# -- FROZEN held-out probe battery (the V3-EXQ-701b/c convergence instrument) --
def _sample_probe_battery(
    agent: REEAgent, seed: int, n_transitions: int, steps: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    env = _make_env(seed, **STABLE_DRIFT)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    act_rng = random.Random(seed + 9973)
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    guard = 0
    max_guard = max(steps, 1) * 8
    while len(battery) < n_transitions and guard < max_guard:
        guard += 1
        latent = _sense_latent(agent, obs_dict)
        if not torch.isfinite(latent.z_world).all():
            break
        z_now = latent.z_world.detach().reshape(1, -1).clone()
        if prev is not None:
            z0, a = prev
            battery.append((z0, a, z_now))
        idx = act_rng.randrange(env.action_dim)
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        _, _, done, _, obs_dict = env.step(action)
        prev = (z_now, action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
            prev = None
    return battery


def _frozen_probe_pe(
    agent: REEAgent, battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """Mean one-step world_forward reconstruction error over the FIXED battery.
    The world-model convergence metric (readiness only)."""
    if not battery:
        return 0.0
    errs: List[float] = []
    with torch.no_grad():
        for z0, a, z1 in battery:
            pred = agent.e2.world_forward(z0.to(agent.device), a.to(agent.device))
            err = float((pred - z1.to(agent.device)).pow(2).mean().item())
            if math.isfinite(err):
                errs.append(err)
    return float(np.mean(errs)) if errs else 0.0


def _inject_mel(agent: REEAgent, target_pe: float) -> None:
    """MEL-injection positive control: overwrite the waking-MEL accumulator so
    current_mel() == target_pe by construction (decoupled from foraging novelty).
    Called AFTER the buffer-warming wake window and BEFORE force_cycle, so
    duration_factor() reads the injected value."""
    acc = agent.mel_consumer.accumulator
    acc.reset()
    for _ in range(INJECT_N_STEPS):
        acc.accumulate(float(target_pe))


def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int) -> Dict[str, Any]:
    """One (seed, arm) cell: build agent, converge P0 recon-only on the stable
    base, calibrate the MEL reference to the stable-base measurement-modality MEL,
    then run the wake-sleep measurement cycles on the arm's env. For the injection
    positive-control arm, the accumulated MEL is overridden to a graded per-cycle
    target right before each force_cycle."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm_id = arm["arm_id"]
    mel_on = bool(arm["mel_on"])
    is_inject = bool(arm.get("inject_pc"))
    # Progress boundary: resets episodes_in_run in the runner for this cell.
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # ONE agent per cell. The encoder is FROZEN (only agent.e2 trains). Build with
    # mel_reference=0.0, then fix it to the calibrated stable-base MEL AFTER P0
    # (mode="fixed", so the >0 reference is used and never auto-relocked).
    stable_env = _make_env(seed, **STABLE_DRIFT)
    agent = _make_agent(stable_env, mel_on=mel_on, mel_reference=0.0)
    battery = _sample_probe_battery(agent, seed, PROBE_BATTERY_SIZE, steps)
    probe_pe_init = _frozen_probe_pe(agent, battery)

    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    sample_rng = random.Random(seed + 4242)

    _run_wake_window(
        agent, stable_env, conv_eps, steps, train=True, buffer=buffer,
        e2_opt=e2_opt, sample_rng=sample_rng, ep_offset=0, arm_id=arm_id, seed=seed,
    )

    probe_pe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((probe_pe_init - probe_pe_final) / probe_pe_init)
                     if probe_pe_init > 1e-12 else 0.0)

    # Reference calibration: a stable-base wake pass measured through the SAME path
    # as the measurement MEL (agent.update_residue -> e3 PE -> consumer accumulator).
    # NONE arm (stable env) then yields factor ~ 1.0; higher-MEL arms scale above it.
    if agent.mel_consumer is not None:
        agent.mel_consumer.reset()   # clear P0 accumulation
    _run_wake_window(
        agent, stable_env, CALIB_EPISODES, steps, train=False, buffer=None,
        e2_opt=None, sample_rng=None, ep_offset=conv_eps, arm_id=arm_id, seed=seed,
    )
    if mel_on and agent.mel_consumer is not None:
        base_ref = float(agent.mel_consumer.current_mel())
        if not (base_ref > 0.0):     # degenerate fallback (no PE accumulated)
            base_ref = float(probe_pe_final)
        agent.config.mel_reference = base_ref
        agent.mel_consumer.config.mel_reference = base_ref
        agent.mel_consumer.reset()   # clean slate for the first measurement cycle
    else:
        base_ref = float(probe_pe_final)   # OFF arm: reference unused

    # Measurement: MEAS_CYCLES wake-sleep cycles on the arm's env. For the injection
    # arm the env is the stable base (novelty irrelevant -- MEL is injected).
    meas_env = _make_env(seed, arm["env_drift_interval"], arm["env_drift_prob"])
    cum_sws = 0.0
    cum_rem = 0.0
    per_cycle_sws: List[float] = []
    per_cycle_rem: List[float] = []
    factors: List[float] = []
    mels: List[float] = []
    inject_levels_used: List[float] = []
    ep_off = conv_eps + CALIB_EPISODES
    for _cyc in range(meas_cycles):
        _run_wake_window(
            agent, meas_env, WAKE_EPISODES_PER_CYCLE, steps, train=False,
            buffer=None, e2_opt=None, sample_rng=None, ep_offset=ep_off,
            arm_id=arm_id, seed=seed,
        )
        ep_off += WAKE_EPISODES_PER_CYCLE
        if is_inject and agent.mel_consumer is not None:
            lvl = INJECT_LEVELS[_cyc] if _cyc < len(INJECT_LEVELS) else INJECT_LEVELS[-1]
            _inject_mel(agent, float(lvl) * float(base_ref))
            inject_levels_used.append(float(lvl))
        m = agent.sleep_loop.force_cycle(agent)
        sws = float(m.get("sws_n_writes", 0.0))
        rem = float(m.get("rem_n_rollouts", 0.0))
        cum_sws += sws
        cum_rem += rem
        per_cycle_sws.append(sws)
        per_cycle_rem.append(rem)
        if mel_on:
            factors.append(float(m.get("mel_duration_factor", 1.0)))
            mels.append(float(m.get("mel_mean", 0.0)))

    dv = cum_sws + cum_rem
    count_var = float(np.var(per_cycle_sws)) + float(np.var(per_cycle_rem))
    mean_factor = float(np.mean(factors)) if factors else 1.0
    mean_mel = float(np.mean(mels)) if mels else 0.0

    print(f"    {arm_id} seed={seed}: conv_drop={conv_rel_drop:.3f} "
          f"ref={base_ref:.3e} mel={mean_mel:.3e} factor={mean_factor:.3f} "
          f"cum_sws={cum_sws:.0f} cum_rem={cum_rem:.0f} DV={dv:.0f} "
          f"cnt_var={count_var:.3f}", flush=True)
    print(f"verdict: {'PASS' if conv_rel_drop >= MIN_REL_CONV_DROP else 'FAIL'}",
          flush=True)

    return {
        "arm_id": arm_id,
        "level": arm["level"],
        "mel_on": mel_on,
        "inject_pc": is_inject,
        "seed": seed,
        "conv_rel_drop": conv_rel_drop,
        "probe_pe_init": probe_pe_init,
        "probe_pe_final": probe_pe_final,
        "mel_reference": base_ref,
        "mean_mel": mean_mel,
        "mean_duration_factor": mean_factor,
        "cumulative_sws_writes": cum_sws,
        "cumulative_rem_rollouts": cum_rem,
        "dv_total_offline_updates": dv,
        "per_cycle_sws": per_cycle_sws,
        "per_cycle_rem": per_cycle_rem,
        "per_cycle_mel": mels,
        "per_cycle_factor": factors,
        "inject_levels_used": inject_levels_used,
        "per_cycle_count_variance": count_var,
        "meas_cycles": meas_cycles,
    }


def _seed_c1_pass(on_eco_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """C1 per seed on the 4 ecological ON arms, sorted ascending by MEASURED
    mean_mel: (a) non-degenerate measured-MEL gradient, (b) DV monotone
    non-decreasing along that order, (c) DV[max-MEL] relative-spread above
    DV[min-MEL]. Returns the component flags + the ordered mel/DV vectors."""
    arms_sorted = sorted(on_eco_by_arm.values(), key=lambda r: r["mean_mel"])
    mels = [r["mean_mel"] for r in arms_sorted]
    dvs = [r["dv_total_offline_updates"] for r in arms_sorted]
    arm_order = [r["arm_id"] for r in arms_sorted]
    mel_gradient_ok = mels[0] > 0 and mels[-1] >= mels[0] * (1 + MIN_MEL_SPREAD)
    dv_lo = dvs[0]
    tol = MONO_TOL * max(dv_lo, 1.0)
    monotone_ok = all(dvs[i] <= dvs[i + 1] + tol for i in range(len(dvs) - 1))
    spread_ok = dv_lo > 0 and dvs[-1] >= dv_lo * (1 + MIN_REL_DV_SPREAD)
    c1 = bool(mel_gradient_ok and monotone_ok and spread_ok)
    return {
        "mel_gradient_ok": bool(mel_gradient_ok),
        "monotone_ok": bool(monotone_ok),
        "spread_ok": bool(spread_ok),
        "c1_pass": c1,
        "arm_order_by_measured_mel": arm_order,
        "mel_by_measured_order": mels,
        "dv_by_measured_order": dvs,
    }


def _seed_c2_pass(on_eco_by_arm: Dict[str, Dict[str, Any]],
                  off_cell: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """C2 per seed: OFF control pinned (zero per-cycle count variance) AND ON-HIGH
    DV > OFF-HIGH DV (matched novelty). Returns (pinned_ok, on_gt_off, c2_pass)."""
    pinned_ok = off_cell["per_cycle_count_variance"] <= 1e-9
    on_gt_off = (on_eco_by_arm["ARM_3_HIGH_ON"]["dv_total_offline_updates"]
                 > off_cell["dv_total_offline_updates"])
    return pinned_ok, on_gt_off, bool(pinned_ok and on_gt_off)


def _seed_inject_pc_pass(inject_cell: Dict[str, Any]) -> Dict[str, Any]:
    """Injection positive control per seed: with graded MEL injected by construction
    (monotone-increasing INJECT_LEVELS), per-cycle DV is monotone non-decreasing +
    spread. This is the SAME statistic C1 routes on (graded MEL -> DV), on a control
    where the MEL gradient is non-degenerate by construction."""
    dv_cyc = [s + r for s, r in zip(inject_cell["per_cycle_sws"],
                                    inject_cell["per_cycle_rem"])]
    if len(dv_cyc) < 2:
        return {"monotone_ok": False, "spread_ok": False, "inject_pc_pass": False,
                "dv_by_cycle": dv_cyc}
    dv_lo = dv_cyc[0]
    tol = MONO_TOL * max(dv_lo, 1.0)
    monotone_ok = all(dv_cyc[i] <= dv_cyc[i + 1] + tol for i in range(len(dv_cyc) - 1))
    spread_ok = dv_lo > 0 and dv_cyc[-1] >= dv_lo * (1 + MIN_REL_DV_SPREAD)
    return {
        "monotone_ok": bool(monotone_ok),
        "spread_ok": bool(spread_ok),
        "inject_pc_pass": bool(monotone_ok and spread_ok),
        "dv_by_cycle": dv_cyc,
        "inject_levels_used": inject_cell.get("inject_levels_used", []),
        "factor_by_cycle": inject_cell.get("per_cycle_factor", []),
    }


def run_experiment(steps: int, conv_eps: int, meas_cycles: int,
                   seeds: List[int], arms: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    arms = arms if arms is not None else ARMS
    arm_results: List[Dict[str, Any]] = []
    # Per-cell grid with arm fingerprint (independent cells: fresh agent per cell).
    for seed in seeds:
        for arm in arms:
            full_config = {
                "env_base": ENV_BASE,
                "arm": arm,
                "conv_episodes": conv_eps,
                "meas_cycles": meas_cycles,
                "steps_per_episode": steps,
                "sws_steps": SWS_CONSOLIDATION_STEPS,
                "rem_steps": REM_ATTRIBUTION_STEPS,
                "mel_gain": MEL_GAIN,
                "factor_min": FACTOR_MIN,
                "factor_max": FACTOR_MAX,
                "mel_relative_floor": MEL_RELATIVE_FLOOR,
                "inject_levels": INJECT_LEVELS,
            }
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__)) as cell:
                row = _run_cell(seed, arm, steps, conv_eps, meas_cycles)
                cell.stamp(row)
            arm_results.append(row)

    # -- Readiness (R): per seed = frozen-probe convergence AND injection-PC response.
    seed_ready: Dict[int, bool] = {}
    seed_conv_ok: Dict[int, bool] = {}
    seed_inject_ok: Dict[int, bool] = {}
    inject_detail: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        on_eco_cells = [r for r in arm_results
                        if r["seed"] == seed and r["mel_on"] and not r["inject_pc"]]
        conv_ok = all(r["conv_rel_drop"] >= MIN_REL_CONV_DROP for r in on_eco_cells) \
            if on_eco_cells else False
        inject_cell = next((r for r in arm_results
                            if r["seed"] == seed and r["inject_pc"]), None)
        inj = (_seed_inject_pc_pass(inject_cell) if inject_cell is not None
               else {"inject_pc_pass": False})
        inject_detail[seed] = inj
        seed_conv_ok[seed] = bool(conv_ok)
        seed_inject_ok[seed] = bool(inj.get("inject_pc_pass", False))
        seed_ready[seed] = bool(conv_ok and seed_inject_ok[seed])
    conv_frac = sum(seed_conv_ok.values()) / max(1, len(seeds))
    inject_frac = sum(seed_inject_ok.values()) / max(1, len(seeds))
    readiness_frac = sum(seed_ready.values()) / max(1, len(seeds))
    conv_ok_overall = conv_frac >= SEED_PASS_FRAC
    inject_ok_overall = inject_frac >= SEED_PASS_FRAC
    readiness_ok = readiness_frac >= SEED_PASS_FRAC

    # -- C1 (ecological, load-bearing) / C2 (control) per seed, on ready seeds only.
    c1_seed_pass = 0
    c2_seed_pass = 0
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        on_eco = {r["arm_id"]: r for r in arm_results
                  if r["seed"] == seed and r["mel_on"] and not r["inject_pc"]}
        off_cell = next((r for r in arm_results
                         if r["seed"] == seed and r["arm_id"] == "ARM_4_HIGH_OFF"), None)
        rec: Dict[str, Any] = {
            "seed": seed, "ready": seed_ready[seed],
            "conv_ok": seed_conv_ok[seed], "inject_pc_ok": seed_inject_ok[seed],
            "inject_pc_detail": inject_detail.get(seed, {}),
        }
        if seed_ready[seed] and len(on_eco) == len(ON_ECO_ARMS) and off_cell:
            c1 = _seed_c1_pass(on_eco)
            pinned_ok, on_gt_off, c2 = _seed_c2_pass(on_eco, off_cell)
            rec.update({
                "c1_mel_gradient_ok": c1["mel_gradient_ok"],
                "c1_monotone_ok": c1["monotone_ok"],
                "c1_spread_ok": c1["spread_ok"],
                "c1_pass": c1["c1_pass"],
                "arm_order_by_measured_mel": c1["arm_order_by_measured_mel"],
                "mel_by_measured_order": c1["mel_by_measured_order"],
                "dv_by_measured_order": c1["dv_by_measured_order"],
                "c2_pinned_ok": pinned_ok, "c2_on_gt_off": on_gt_off, "c2_pass": c2,
                "dv_off_high": off_cell["dv_total_offline_updates"],
                "factor_by_arm": {a: on_eco[a]["mean_duration_factor"] for a in ON_ECO_ARMS},
                "mel_by_arm": {a: on_eco[a]["mean_mel"] for a in ON_ECO_ARMS},
            })
            if c1["c1_pass"]:
                c1_seed_pass += 1
            if c2:
                c2_seed_pass += 1
        else:
            rec.update({"c1_pass": False, "c2_pass": False,
                        "skipped": "not_ready_or_missing_arms"})
        per_seed.append(rec)

    c1_frac = c1_seed_pass / max(1, len(seeds))
    c2_frac = c2_seed_pass / max(1, len(seeds))
    c1_pass = c1_frac >= SEED_PASS_FRAC
    c2_pass = c2_frac >= SEED_PASS_FRAC

    # -- Self-route (NO-weakens map).
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
    elif c1_pass and c2_pass:
        label = "mel_drives_adaptive_sleep_cadence"
        outcome = "PASS"
        direction = "supports"
    elif c2_pass and not c1_pass:
        label = "mel_consumer_capable_ecological_novelty_not_graded"
        outcome = "FAIL"
        direction = "non_contributory"
    else:
        label = "mel_control_degenerate"
        outcome = "FAIL"
        direction = "non_contributory"

    # criteria_non_degenerate: did each criterion discriminate (not pass/fail vacuously)?
    ready_seeds = [s for s in seeds if seed_ready[s]]
    # C1 measured-MEL gradient present on ready seeds (the load-bearing non-degeneracy).
    c1_gradient_present = any(
        rec.get("c1_mel_gradient_ok", False) for rec in per_seed
        if rec["seed"] in ready_seeds
    ) if ready_seeds else False
    ready_on_dvs = [r["dv_total_offline_updates"] for r in arm_results
                    if r["mel_on"] and not r["inject_pc"] and seed_ready.get(r["seed"], False)]
    c1_dv_spread_nonzero = (len(set(round(v, 3) for v in ready_on_dvs)) > 1) if ready_on_dvs else False
    off_cells = [r for r in arm_results if not r["mel_on"]]
    c2_off_present = len(off_cells) > 0
    # Injection PC produced distinct per-cycle DVs on >=1 ready seed.
    inject_levels_distinct = any(
        len(set(round(v, 3) for v in inject_detail.get(s, {}).get("dv_by_cycle", []))) > 1
        for s in ready_seeds
    ) if ready_seeds else False

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "world_forward_converged_frozen_probe",
                "description": "recon-only P0 converges on the fixed frozen-probe battery "
                               "(conv_rel_drop >= MIN_REL_CONV_DROP) on the ecological ON "
                               "arms, on >= 2/3 seeds; the world model must be trained for "
                               "MEL to live at a meaningful scale",
                "measured": conv_frac,
                "threshold": SEED_PASS_FRAC,
                "met": bool(conv_ok_overall),
            },
            {
                "name": "mel_injection_positive_control_dv_monotone_response",
                "description": "with graded waking MEL injected BY CONSTRUCTION (monotone "
                               "INJECT_LEVELS x base_ref, decoupled from foraging), per-cycle "
                               "DV is monotone non-decreasing + spread -- the SAME statistic "
                               "C1 routes on (graded MEL -> DV) on a positive control where the "
                               "MEL gradient is non-degenerate by construction. Below-floor => "
                               "the consumer cannot respond to cleanly-graded MEL (not-ready / "
                               "mis-configured), NOT a refuted mechanism; on >= 2/3 seeds",
                "measured": inject_frac,
                "threshold": SEED_PASS_FRAC,
                "met": bool(inject_ok_overall),
            },
        ],
        "criteria_non_degenerate": {
            "C1_measured_mel_gradient_present": bool(c1_gradient_present),
            "C1_dv_spread_nonzero": bool(c1_dv_spread_nonzero),
            "C2_off_control_present": bool(c2_off_present),
            "C3_injection_levels_produced_distinct_dv": bool(inject_levels_distinct),
        },
    }
    criteria = [
        {"name": "C1_mel_drives_cadence_monotone_in_MEASURED_mel", "load_bearing": True, "passed": bool(c1_pass)},
        {"name": "C2_consumer_not_env_causes_variation", "load_bearing": False, "passed": bool(c2_pass)},
        {"name": "C3_injection_positive_control_capability", "load_bearing": False, "passed": bool(inject_ok_overall)},
    ]

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"INV-050": direction, "MECH-180": direction},
        "interpretation": interpretation,
        "criteria": criteria,
        "readiness_ok": readiness_ok,
        "readiness_frac": readiness_frac,
        "conv_frac": conv_frac,
        "inject_pc_frac": inject_frac,
        "c1_pass": c1_pass, "c1_frac": c1_frac,
        "c2_pass": c2_pass, "c2_frac": c2_frac,
        "per_seed": per_seed,
        "arm_results": arm_results,
        "thresholds": {
            "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
            "SEED_PASS_FRAC": SEED_PASS_FRAC,
            "MIN_MEL_SPREAD": MIN_MEL_SPREAD,
            "MIN_REL_DV_SPREAD": MIN_REL_DV_SPREAD,
            "MONO_TOL": MONO_TOL,
            "MEL_GAIN": MEL_GAIN,
            "FACTOR_MIN": FACTOR_MIN,
            "FACTOR_MAX": FACTOR_MAX,
            "MEL_RELATIVE_FLOOR": MEL_RELATIVE_FLOOR,
            "INJECT_LEVELS": INJECT_LEVELS,
        },
    }


def write_manifest(result: Dict[str, Any]) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": "v3_exq_718_sdmelconsumer_adaptive_cadence_validation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "manual-cycle-loop (force_cycle() once per cycle in a "
                                "MEAS_CYCLES wake-sleep loop; MEL consumer engages via "
                                "SleepLoopManager)",
        "timestamp_utc": ts,
        "seeds": SEEDS,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation": result["interpretation"],
        "criteria": result["criteria"],
        "readiness_ok": result["readiness_ok"],
        "readiness_frac": result["readiness_frac"],
        "conv_frac": result["conv_frac"],
        "inject_pc_frac": result["inject_pc_frac"],
        "c1_pass": result["c1_pass"], "c1_frac": result["c1_frac"],
        "c2_pass": result["c2_pass"], "c2_frac": result["c2_frac"],
        "per_seed": result["per_seed"],
        "arm_results": result["arm_results"],
        "thresholds": result["thresholds"],
    }
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="1 seed, tiny convergence + measurement (smoke)")
    args = ap.parse_args()

    if args.dry_run:
        steps = 12
        conv_eps = 4
        meas_cycles = 3
        seeds = [42]
        # Smoke subset: exercise every distinct code path (ecological ON +
        # calibration, OFF pinning, injection override) with 3 agent builds
        # instead of 6. The full 6-arm grid runs at real scale.
        smoke_ids = {"ARM_0_NONE_ON", "ARM_4_HIGH_OFF", "ARM_5_INJECT_PC"}
        arms = [a for a in ARMS if a["arm_id"] in smoke_ids]
    else:
        steps = STEPS_PER_EPISODE
        conv_eps = CONV_EPISODES
        meas_cycles = MEAS_CYCLES
        seeds = SEEDS
        arms = ARMS

    result = run_experiment(steps, conv_eps, meas_cycles, seeds, arms)
    out_path = write_manifest(result)
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"readiness_frac={result['readiness_frac']:.2f} "
          f"(conv={result['conv_frac']:.2f} inject_pc={result['inject_pc_frac']:.2f}) "
          f"c1_frac={result['c1_frac']:.2f} c2_frac={result['c2_frac']:.2f}", flush=True)
    print(f"manifest: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry_run,
    )
