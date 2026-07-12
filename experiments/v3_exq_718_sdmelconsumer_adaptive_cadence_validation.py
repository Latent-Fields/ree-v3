"""
V3-EXQ-718 -- SD-MEL-CONSUMER validation: does accumulated waking MEL drive
adaptive sleep cadence? (INV-050 third / learning-demand drive; MECH-180)

SLEEP DRIVER: manual-cycle-loop (agent.sleep_loop.force_cycle() called once per
cycle in a dedicated MEAS_CYCLES wake-sleep loop). The MEL consumer engages ONLY
through the SleepLoopManager path (force_cycle / notify_episode_end); a driver
that calls agent.run_sleep_cycle() directly (as V3-EXQ-677 did) BYPASSES the
consumer. MEL accumulates only because this driver calls agent.update_residue()
on every waking step.

PURPOSE (diagnostic; PROMOTES NOTHING): substrate-readiness validation of the
SD-MEL-CONSUMER landing (ree-v3 main 909292c; ree_core/sleep/mel_consumer.py).
The consumer reads accumulated waking Model Error Load (MEL = mean per-step e3
prediction error over the wake window -- the SAME signal V3-EXQ-701c demonstrated
is measurable + MONOTONE in graded novelty on a converged recon-only base) and
scales the offline-phase DURATION: sws_consolidation_steps -> sws_n_writes and
rem_attribution_steps -> rem_n_rollouts, by
    factor = clamp(1 + mel_gain*(mel/ref - 1), factor_min, factor_max).
This is the consumer that un-pins the exact V3-EXQ-677 DV (cumulative_sws_writes /
cumulative_rem_rollouts were scheduler-pinned SWS=80/REM=60, zero cross-arm
variance, so MEL could not modulate cadence -> MECH-180 held v3_pending).

RE-DERIVE BRAKE: RELEASED. INV-050 has >=2 non_contributory autopsies
(701/701a/701b/701c), but the named upstream substrate (SD-MEL-CONSUMER) LANDED
2026-07-07 (failure_autopsy_V3-EXQ-701c_2026-06-30 recommended /implement-substrate
on exactly this consumer). This is not another measurability letter -- it tests a
DIFFERENT question (functional sufficiency of the now-built consumer) on
newly-built substrate. Cites: SD-MEL-CONSUMER; failure_autopsy_V3-EXQ-701c;
failure_autopsy_V3-EXQ-677.

DESIGN (template = v3_exq_701c_inv050_mel_measurability_recononly.py):
  Base: recon-only P0 world-forward convergence on the STABLE base env (frozen
    encoder; only agent.e2 trains) so per-step e3 PE lives at converged scale
    (~2e-5, per 701c). Convergence measured on the SAME fixed frozen-probe battery
    701c used; conv_rel_drop is a READINESS precondition (below-floor ->
    substrate_not_ready_requeue, never a substrate verdict).
  Reference set-point: mel_reference = the cell's converged frozen-probe PE (the
    converged base per-step PE). Passed FIXED to the consumer, so the NONE arm
    (stable measurement env == base) yields factor ~ 1.0 and higher-novelty arms
    scale above it. Shared-by-construction across arms of a seed (same base+seed
    -> same converged PE).
  Arms (per seed): 4 graded-novelty arms with the consumer ON + 1 matched-novelty
    control with the consumer OFF:
      ARM_0_NONE_ON  (drift 999/0.00, use_mel_consumer=True)
      ARM_1_LOW_ON   (drift 15/0.15,  use_mel_consumer=True)
      ARM_2_MED_ON   (drift 6/0.30,   use_mel_consumer=True)
      ARM_3_HIGH_ON  (drift 2/0.50,   use_mel_consumer=True)
      ARM_4_HIGH_OFF (drift 2/0.50,   use_mel_consumer=False)  <- reproduces 677 pinning
  Measurement per cell: MEAS_CYCLES wake-sleep cycles. Each cycle runs
    WAKE_EPISODES_PER_CYCLE episodes on the arm's novelty env (calling
    agent.update_residue() every waking step so MEL accumulates), then fires ONE
    sleep cycle via agent.sleep_loop.force_cycle(agent). Sum sws_n_writes +
    rem_n_rollouts across cycles; capture mel_duration_factor + mel_mean.

DV = cumulative_sws_writes + cumulative_rem_rollouts (the V3-EXQ-677 pinned DV),
     per arm, per seed; plus mel_duration_factor + mel_mean.

Pre-registered thresholds (constants, NOT derived from run stats):
  READINESS R (per seed, on the ON arms' base): frozen-probe conv_rel_drop >=
    MIN_REL_CONV_DROP on >= SEED_PASS_FRAC of seeds AND the NONE-arm factor has
    headroom (none_factor < FACTOR_MAX - FACTOR_HEADROOM_EPS) so cross-arm range
    CAN be non-zero -- the SAME statistic (duration factor -> DV) C1 routes on.
    Below-floor -> substrate_not_ready_requeue.
  C1 (LOAD-BEARING) MEL drives cadence, monotone in novelty:
    DV_on monotone non-decreasing NONE<=LOW<=MED<=HIGH (MONO_TOL relative slack)
    AND DV_on[HIGH] >= DV_on[NONE] * (1 + MIN_REL_DV_SPREAD)
    on >= SEED_PASS_FRAC of seeds.
  C2 (control non-degeneracy) the consumer, not the env, causes the variation:
    per seed, ARM_4_HIGH_OFF has ZERO per-cycle count variance (pinned) AND
    DV_on[HIGH] > DV_off[HIGH] (matched novelty, consumer ON scales above OFF)
    on >= SEED_PASS_FRAC of seeds.

Self-route (3-branch NO-weakens map):
  readiness unmet                 -> substrate_not_ready_requeue (re-queue at an
                                     adequate P0 / reference; NOT a verdict)
  readiness met + C1 PASS + C2    -> mel_drives_adaptive_sleep_cadence (supports;
                                     substrate ready; /governance clears INV-050
                                     retest + MECH-180 v3_pending)
  readiness met + C1 FAIL         -> mel_does_not_modulate_cadence (non_contributory;
                                     MEL measurable but does not drive cadence on
                                     this substrate)

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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_718_sdmelconsumer_adaptive_cadence_validation"
QUEUE_ID = "V3-EXQ-718"
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
MEL_RELATIVE_FLOOR = 1e-6     # recalibrated DOWN from 701c ABS_MEL_FLOOR=1e-4
FACTOR_HEADROOM_EPS = 0.1     # NONE factor must be < FACTOR_MAX - eps (headroom)

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
MIN_REL_DV_SPREAD = 0.15      # C1: DV[HIGH] at least 15% above DV[NONE]
MONO_TOL = 0.05               # C1: monotonicity slack (relative to DV[NONE])

# -- Environment base (mirrors V3-EXQ-701c) ----------------------------------
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

# arm_id, novelty level, measurement-env drift, consumer on/off.
ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE_ON",  "level": 0, "env_drift_interval": 999, "env_drift_prob": 0.00, "mel_on": True},
    {"arm_id": "ARM_1_LOW_ON",   "level": 1, "env_drift_interval": 15,  "env_drift_prob": 0.15, "mel_on": True},
    {"arm_id": "ARM_2_MED_ON",   "level": 2, "env_drift_interval": 6,   "env_drift_prob": 0.30, "mel_on": True},
    {"arm_id": "ARM_3_HIGH_ON",  "level": 3, "env_drift_interval": 2,   "env_drift_prob": 0.50, "mel_on": True},
    {"arm_id": "ARM_4_HIGH_OFF", "level": 3, "env_drift_interval": 2,   "env_drift_prob": 0.50, "mel_on": False},
]
ON_ARM_ORDER = ["ARM_0_NONE_ON", "ARM_1_LOW_ON", "ARM_2_MED_ON", "ARM_3_HIGH_ON"]


def _make_env(seed: int, env_drift_interval: int, env_drift_prob: float) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw["env_drift_interval"] = env_drift_interval
    kw["env_drift_prob"] = env_drift_prob
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2, mel_on: bool, mel_reference: float) -> REEAgent:
    """Converged-base agent (recon-only e2 training; encoder frozen) + SD-017
    SWS/REM passes + the SleepLoopManager. When mel_on, the SD-MEL-CONSUMER is
    enabled with a FIXED reference set-point (the converged base PE). Base config
    is otherwise byte-identical to the 701c converged-base recipe."""
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
    During measurement (train=False) just drives the agent + accumulates MEL."""
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
    This is the converged base per-step PE ~ what e3_prediction_error reads; it is
    both the convergence metric AND the MEL reference set-point."""
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


def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int) -> Dict[str, Any]:
    """One (seed, arm) cell: build agent, converge P0 recon-only on the stable
    base, set the MEL reference to the converged frozen-probe PE, then run the
    wake-sleep measurement cycles on the arm's novelty env."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm_id = arm["arm_id"]
    mel_on = bool(arm["mel_on"])
    # Progress boundary: resets episodes_in_run in the runner for this cell.
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # ONE agent per cell. The encoder is FROZEN (only agent.e2 trains), so the
    # frozen-probe battery -- captured with THIS agent's encoder BEFORE training --
    # stays valid across P0 checkpoints. Reference set-point unknown at construction
    # -> build with mel_reference=0.0, then fix it to the converged frozen-probe PE
    # AFTER P0 (mode="fixed", so the >0 reference is used and never auto-relocked).
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

    # Converged base PE on the fixed battery (world_forward convergence metric ONLY;
    # NOT the MEL reference -- the frozen probe uses random actions, so its PE is a
    # different scale (~7x) from the live committed-trajectory e3 MEL. Using it as
    # the reference would force factor to clamp at MIN. Reference is calibrated below
    # at the SAME modality as the measurement MEL.)
    probe_pe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((probe_pe_init - probe_pe_final) / probe_pe_init)
                     if probe_pe_init > 1e-12 else 0.0)

    # 2b. Reference calibration: a stable-base wake pass measured through the SAME
    #     path as the measurement MEL (agent.update_residue -> e3 prediction error,
    #     read via the consumer accumulator). NONE arm (stable measurement env) then
    #     yields factor ~ 1.0; higher-novelty arms scale above it. Runs for ALL arms
    #     (symmetric warmup); the reference is read + fixed for ON arms only.
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

    # 3. Measurement: MEAS_CYCLES wake-sleep cycles on the arm's novelty env.
    meas_env = _make_env(seed, arm["env_drift_interval"], arm["env_drift_prob"])
    cum_sws = 0.0
    cum_rem = 0.0
    per_cycle_sws: List[float] = []
    per_cycle_rem: List[float] = []
    factors: List[float] = []
    mels: List[float] = []
    ep_off = conv_eps + CALIB_EPISODES
    for _cyc in range(meas_cycles):
        _run_wake_window(
            agent, meas_env, WAKE_EPISODES_PER_CYCLE, steps, train=False,
            buffer=None, e2_opt=None, sample_rng=None, ep_offset=ep_off,
            arm_id=arm_id, seed=seed,
        )
        ep_off += WAKE_EPISODES_PER_CYCLE
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
    # Per-cycle count variance (0.0 for a pinned OFF arm).
    all_counts = per_cycle_sws + per_cycle_rem
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
        "per_cycle_count_variance": count_var,
        "meas_cycles": meas_cycles,
    }


def _seed_c1_pass(on_by_arm: Dict[str, Dict[str, Any]]) -> Tuple[bool, bool, bool]:
    """C1 per seed on the 4 ON arms: monotone non-decreasing DV in novelty + HIGH
    relative-spread above NONE. Returns (monotone_ok, spread_ok, c1_pass)."""
    dvs = [on_by_arm[a]["dv_total_offline_updates"] for a in ON_ARM_ORDER]
    dv_none = dvs[0]
    tol = MONO_TOL * max(dv_none, 1.0)
    monotone_ok = all(dvs[i] <= dvs[i + 1] + tol for i in range(len(dvs) - 1))
    spread_ok = dv_none > 0 and dvs[-1] >= dv_none * (1 + MIN_REL_DV_SPREAD)
    return monotone_ok, spread_ok, (monotone_ok and spread_ok)


def _seed_c2_pass(on_by_arm: Dict[str, Dict[str, Any]],
                  off_cell: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """C2 per seed: OFF control pinned (zero per-cycle count variance) AND ON-HIGH
    DV > OFF-HIGH DV (matched novelty). Returns (pinned_ok, on_gt_off, c2_pass)."""
    pinned_ok = off_cell["per_cycle_count_variance"] <= 1e-9
    on_gt_off = (on_by_arm["ARM_3_HIGH_ON"]["dv_total_offline_updates"]
                 > off_cell["dv_total_offline_updates"])
    return pinned_ok, on_gt_off, (pinned_ok and on_gt_off)


def run_experiment(steps: int, conv_eps: int, meas_cycles: int,
                   seeds: List[int]) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []
    # Per-cell grid with arm fingerprint (independent cells: fresh agent per cell).
    for seed in seeds:
        for arm in ARMS:
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
            }
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__)) as cell:
                row = _run_cell(seed, arm, steps, conv_eps, meas_cycles)
                cell.stamp(row)
            arm_results.append(row)

    # -- Readiness (R): per-seed frozen-probe convergence + NONE-arm factor headroom.
    seeds_converged = 0
    seed_ready: Dict[int, bool] = {}
    none_factor_headroom: Dict[int, bool] = {}
    for seed in seeds:
        on_cells = [r for r in arm_results if r["seed"] == seed and r["mel_on"]]
        conv_ok = all(r["conv_rel_drop"] >= MIN_REL_CONV_DROP for r in on_cells) \
            if on_cells else False
        none_cell = next((r for r in arm_results
                          if r["seed"] == seed and r["arm_id"] == "ARM_0_NONE_ON"), None)
        headroom = (none_cell is not None
                    and none_cell["mean_duration_factor"] < FACTOR_MAX - FACTOR_HEADROOM_EPS)
        seed_ready[seed] = bool(conv_ok and headroom)
        none_factor_headroom[seed] = bool(headroom)
        if seed_ready[seed]:
            seeds_converged += 1
    readiness_frac = seeds_converged / max(1, len(seeds))
    readiness_ok = readiness_frac >= SEED_PASS_FRAC

    # -- C1 / C2 per seed (scored only on ready seeds).
    c1_seed_pass = 0
    c2_seed_pass = 0
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        on_by_arm = {r["arm_id"]: r for r in arm_results
                     if r["seed"] == seed and r["mel_on"]}
        off_cell = next((r for r in arm_results
                         if r["seed"] == seed and r["arm_id"] == "ARM_4_HIGH_OFF"), None)
        rec: Dict[str, Any] = {"seed": seed, "ready": seed_ready[seed],
                               "none_factor_headroom": none_factor_headroom[seed]}
        if seed_ready[seed] and len(on_by_arm) == len(ON_ARM_ORDER) and off_cell:
            mono_ok, spread_ok, c1 = _seed_c1_pass(on_by_arm)
            pinned_ok, on_gt_off, c2 = _seed_c2_pass(on_by_arm, off_cell)
            rec.update({
                "c1_monotone_ok": mono_ok, "c1_spread_ok": spread_ok, "c1_pass": c1,
                "c2_pinned_ok": pinned_ok, "c2_on_gt_off": on_gt_off, "c2_pass": c2,
                "dv_on_by_level": [on_by_arm[a]["dv_total_offline_updates"] for a in ON_ARM_ORDER],
                "dv_off_high": off_cell["dv_total_offline_updates"],
                "factor_by_level": [on_by_arm[a]["mean_duration_factor"] for a in ON_ARM_ORDER],
                "mel_by_level": [on_by_arm[a]["mean_mel"] for a in ON_ARM_ORDER],
            })
            if c1:
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

    # -- Self-route (3-branch NO-weakens map).
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
    elif c1_pass and c2_pass:
        label = "mel_drives_adaptive_sleep_cadence"
        outcome = "PASS"
        direction = "supports"
    else:
        label = "mel_does_not_modulate_cadence"
        outcome = "FAIL"
        direction = "non_contributory"

    # criteria_non_degenerate: did each criterion discriminate (not pass/fail vacuously)?
    # ON-arm DV spread across levels non-zero on ready seeds -> C1 non-degenerate.
    ready_on_dvs = [r["dv_total_offline_updates"] for r in arm_results
                    if r["mel_on"] and seed_ready.get(r["seed"], False)]
    c1_non_degenerate = (len(set(round(v, 3) for v in ready_on_dvs)) > 1) if ready_on_dvs else False
    off_cells = [r for r in arm_results if not r["mel_on"]]
    c2_non_degenerate = len(off_cells) > 0

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "world_forward_converged_frozen_probe",
                "description": "recon-only P0 converges on the fixed frozen-probe battery "
                               "(conv_rel_drop >= floor) on >= 2/3 seeds; the converged base "
                               "PE is the MEL reference set-point",
                "measured": readiness_frac,
                "threshold": SEED_PASS_FRAC,
                "met": bool(readiness_ok),
            },
            {
                "name": "none_arm_duration_factor_headroom",
                "description": "NONE-arm mean MEL duration factor has headroom below FACTOR_MAX "
                               "(same statistic C1 routes on: factor -> DV), so cross-arm DV range "
                               "CAN be non-zero; a saturated NONE factor means base PE >> reference "
                               "(base not converged) not a refuted mechanism",
                "measured": float(np.mean([
                    next((r["mean_duration_factor"] for r in arm_results
                          if r["seed"] == s and r["arm_id"] == "ARM_0_NONE_ON"), FACTOR_MAX)
                    for s in seeds])),
                "threshold": FACTOR_MAX - FACTOR_HEADROOM_EPS,
                "direction": "upper",
                "met": bool(all(none_factor_headroom.values())),
            },
        ],
        "criteria_non_degenerate": {
            "C1_dv_spread_nonzero": bool(c1_non_degenerate),
            "C2_off_control_present": bool(c2_non_degenerate),
        },
    }
    criteria = [
        {"name": "C1_mel_drives_monotone_cadence", "load_bearing": True, "passed": bool(c1_pass)},
        {"name": "C2_consumer_not_env_causes_variation", "load_bearing": False, "passed": bool(c2_pass)},
    ]

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"INV-050": direction, "MECH-180": direction},
        "interpretation": interpretation,
        "criteria": criteria,
        "readiness_ok": readiness_ok,
        "readiness_frac": readiness_frac,
        "c1_pass": c1_pass, "c1_frac": c1_frac,
        "c2_pass": c2_pass, "c2_frac": c2_frac,
        "per_seed": per_seed,
        "arm_results": arm_results,
        "thresholds": {
            "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
            "SEED_PASS_FRAC": SEED_PASS_FRAC,
            "MIN_REL_DV_SPREAD": MIN_REL_DV_SPREAD,
            "MONO_TOL": MONO_TOL,
            "MEL_GAIN": MEL_GAIN,
            "FACTOR_MIN": FACTOR_MIN,
            "FACTOR_MAX": FACTOR_MAX,
            "MEL_RELATIVE_FLOOR": MEL_RELATIVE_FLOOR,
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
        "c1_pass": result["c1_pass"], "c1_frac": result["c1_frac"],
        "c2_pass": result["c2_pass"], "c2_frac": result["c2_frac"],
        "per_seed": result["per_seed"],
        "arm_results": result["arm_results"],
        "thresholds": result["thresholds"],
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="1 seed, tiny convergence + measurement (smoke)")
    args = ap.parse_args()

    if args.dry_run:
        steps = 12
        conv_eps = 4
        meas_cycles = 2
        seeds = [42]
    else:
        steps = STEPS_PER_EPISODE
        conv_eps = CONV_EPISODES
        meas_cycles = MEAS_CYCLES
        seeds = SEEDS

    result = run_experiment(steps, conv_eps, meas_cycles, seeds)
    out_path = write_manifest(result)
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"readiness_frac={result['readiness_frac']:.2f} "
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
