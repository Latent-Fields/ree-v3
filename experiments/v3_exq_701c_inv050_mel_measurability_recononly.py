#!/opt/local/bin/python3
"""
V3-EXQ-701c -- INV-050 MEL-measurability diagnostic, RECON-ONLY P0 base +
FROZEN-PROBE MEL battery (commitment-free). SUPERSEDES V3-EXQ-701b.

CLAIM UNDER TEST: INV-050
  "Sleep phase architecture is regulated by three drives -- circadian, homeostatic,
  and a learning/model-update demand drive proportional to daily prediction error
  accumulation (Model Error Load, MEL)." The novel, falsifiable content over the
  standard two-process model is the THIRD (MEL) drive. This run tests ONLY the
  directly-testable IV precondition: is accumulated waking prediction-error a
  measurable, graded, manipulable quantity?

WHY THIS RE-ISSUE (read first):
  V3-EXQ-701b FAILed non_contributory (confirmed failure_autopsy_V3-EXQ-701b_2026-06-29)
  on its R2 readiness gate (frozen-probe conv_frac 0.333, 1/3 seeds converged). The
  701b autopsy showed this was NOT an INV-050 ceiling and NOT a metric artifact: a
  MATCHED 2-arm ablation (identical init + the SAME frozen probe battery; only the
  SD-056 contrastive auxiliary term differs in the P0 loss) isolated the destabiliser:
    - recon+contrastive (701b's deployed P0 recipe): converged 1/3 seeds (frac 0.333)
    - recon-only                                    : converged 3/3 seeds (frac 1.000,
      conv_rel_drop ~0.97-0.98, PE ~1.4e-4)
  destabiliser_verdict = sd056_contrastive_is_destabiliser. The SD-056 contrastive
  auxiliary destabilises the reconstruction-primary world-forward P0; recon-only
  supplies a clean, demonstrably-converging base.

THE FIX (this experiment):
  Build the MEL battery on the RECON-ONLY P0 warmup -- the VALID converged base the
  701b ablation identified -- and re-run the SAME frozen fixed-probe MEL instrument
  701b used (which removed the 701a episode-rollout exploration-drift confound).
  Concretely: the main (arm, seed) cells now train P0 with use_contrastive=False
  (the SD-056 contrastive auxiliary is OMITTED from the P0 loss; L_E2 = L_recon only).
  Everything else is unchanged from 701b:
    - The agent config is BYTE-IDENTICAL to 701b's _make_agent (the contrastive head
      stays e2_action_contrastive_enabled=True so the world_forward MODULE matches the
      proven-converging recon-only ablation arm exactly; recon-only means the
      contrastive LOSS TERM is omitted in training, precisely what the 701b ablation
      did to converge 3/3 -- NOT a structural reconfiguration of the model).
    - The FROZEN held-out probe-state convergence metric (701b's fix): conv_rel_drop
      measured on a FIXED, pre-sampled battery of one-step (z0, action, z1)
      transitions evaluated on the frozen-at-checkpoint world-forward model, removing
      the exploration/coverage-drift confound by construction (encoder is frozen:
      alpha_world=0.9, optimiser = Adam(agent.e2.parameters()) only).
    - The raw per-step e3 prediction_error + frozen-window MEL accumulation instrument
      (fixed V3-EXQ-677's batch-training-loss wash-out) is KEPT UNCHANGED.
    - The R1 novelty-shock positive-control probe is KEPT UNCHANGED.
    - Commitment-free: NO sleep flags are set.
  The separate recon-only-vs-recon+contrastive recipe ablation 701b ran has done its
  job (the destabiliser is isolated + autopsy-confirmed); it is NOT re-run here. The
  per-seed R2 gate over the 12 recon-only main cells IS the multi-seed convergence
  certification for the recon-only base.

GATING (per the 701b autopsy routing):
  R2 (world_model_converged on the FROZEN battery, >= 2/3 seeds) MUST pass BEFORE R1
  and C1 are read. The interpretation grid below short-circuits on R2, then R1, then C1.

EXPERIMENT PURPOSE: diagnostic
  (Excluded from governance confidence/conflict scoring; self-route is a hypothesis
  adjudicated by the pipeline, not a verdict on INV-050. PROMOTES NOTHING.)

DESIGN:
  4 graded novelty arms x 3 seeds (42, 123, 456). Per (seed, arm) cell:
    P0 CONVERGENCE (RECON-ONLY; identical across arms for a given seed): train the
      E2 world-forward model on the STABLE base env for CONV_EPISODES with the PRIMARY
      reconstruction MSE ONLY (SD-056 contrastive auxiliary OMITTED).
    FROZEN-PROBE CONVERGENCE METRIC: a FIXED probe battery of one-step transitions
      sampled ONCE (frozen encoder) at cell entry; conv_rel_drop =
      (pe_probe_init - pe_probe_final) / pe_probe_init where pe_probe_* = mean
      world-forward recon error on the battery at P0 start vs P0 end.
    POSITIVE-CONTROL PROBES (frozen model): pe_stable (stable env) and pe_shock
      (max-novelty env). R1 routes on the PE RESPONSE RANGE (pe_shock vs pe_stable)
      -- the SAME kind of statistic the load-bearing C1 routes on (V3-EXQ-643 rule).
    P1 MEASUREMENT (frozen model, no training): MEAS_EPISODES on the arm's graded-
      novelty env, accumulating per-step prediction_error. MEL = mean per-step
      prediction_error over the measurement window.

  Arms (novelty via CausalGridWorldV2 env drift):
    ARM_0_NONE  drift_interval=999 drift_prob=0.00   (stable; matched-activity control)
    ARM_1_LOW   drift_interval=15  drift_prob=0.15
    ARM_2_MED   drift_interval=6   drift_prob=0.30
    ARM_3_HIGH  drift_interval=2   drift_prob=0.50

PRIMARY DEPENDENT VARIABLE:
  mel_mean_pe = mean per-step e3 prediction_error over the frozen measurement window
                (per (seed, arm) cell).

READINESS PRECONDITIONS (positive control; gate BEFORE trusting the verdict):
  R2 world_model_converged_p0_seed_fraction: fraction of seeds whose FROZEN-PROBE
       conv_rel_drop >= MIN_REL_CONV_DROP must be >= SEED_PASS_FRAC. Below floor ->
       substrate_not_ready_requeue (per-seed gate on the FIXED battery).
  R1 pe_response_range_to_novelty_shock (on R2-passing seeds only):
       (pe_shock / pe_stable - 1) >= MIN_REL_PE_RESPONSE -- the converged frozen
       model's PE actually RISES under a known novelty shock. Below floor on the
       ready seeds -> substrate_not_ready_requeue (probe cannot detect novelty).

ACCEPTANCE CRITERIA (evaluated on R2-passing seeds only):
  C1 (LOAD-BEARING) MEL is measurable + monotonic in novelty:
       monotone non-decreasing mel[NONE]<=mel[LOW]<=mel[MED]<=mel[HIGH]
       (per-arm means across ready seeds, MONO_TOL slack) AND
       mel[HIGH] >= mel[NONE] * (1 + MIN_REL_MEL_SPREAD) AND
       (mel[HIGH] - mel[NONE]) > ABS_MEL_FLOOR.
  C2 non-degenerate: per-arm MEL has cross-seed spread (not pinned) AND arm means
       are not all identical, over the ready seeds.

INTERPRETATION GRID (one row per outcome -> next action):
  R2 unmet on the FROZEN-PROBE metric (< SEED_PASS_FRAC seeds converged)
       -> self-route substrate_not_ready_requeue. This is the 701b autopsy BRAKE-LOCK
       condition: the recon-only base (which converged 3/3 in the 701b ablation) ALSO
       failing to converge here is a GENUINE recon-primary world-forward divergence,
       not a recipe defect -> FIRE the INV-050 re-derive brake at the 701c autopsy +
       route to /implement-substrate on the world-forward stabilisation substrate.
       NOT a verdict on INV-050; the 701c autopsy adjudicates the brake.
  R2 met + R1 unmet on ready seeds -> substrate_not_ready_requeue (converged base,
       but the frozen PE does not respond to a max-novelty shock; re-queue with a
       stronger novelty probe -- NOT a verdict).
  R2 met + R1 met + C1 PASS + C2 non-degen -> mel_measurable_monotonic
       INV-050 IV precondition MET on a CONVERGED, FROZEN-PROBE-CERTIFIED recon-only
       base. The deferred adaptive sleep-cadence enrichment now has a non-vacuous IV.
  R2 met + R1 met + C1 FAIL -> mel_not_modulated_by_novelty
       DECISIVE: with a FROZEN-PROBE-CONVERGED recon-only P0 (R2 met) and a PE that
       responds to a shock (R1 met), graded env-drift novelty does NOT produce a graded
       MEL signal -- the GENUINE MEL-measurability ceiling on V3, no longer the
       V3-EXQ-642/701/701a/701b convergence-instrument / recipe confound. This clean
       reading should FIRE the INV-050 re-derive brake at autopsy and route to substrate
       enrichment. NOT a falsification of INV-050 -- a measurability ceiling.

  RECURRENCE NOTE: V3-EXQ-677 (MECH-180, sleep-side), V3-EXQ-701 (diverged P0),
  V3-EXQ-701a (exploration-confounded convergence gate) and V3-EXQ-701b (SD-056
  contrastive destabiliser) each showed novelty failing to move PE under a confound.
  This recon-only re-run builds on the demonstrably-converging base + the frozen-probe
  instrument and is the DECISIVE MEL-measurability test.

claim_ids: ["INV-050"]
experiment_purpose: "diagnostic"
supersedes: "V3-EXQ-701b"
"""

import sys
import json
import math
import time
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
from _metrics import check_degeneracy
from experiments._lib.arm_fingerprint import arm_cell
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_701c_inv050_mel_measurability_recononly"
QUEUE_ID = "V3-EXQ-701c"
SUPERSEDES = "V3-EXQ-701b"
CLAIM_IDS = ["INV-050"]
EXPERIMENT_PURPOSE = "diagnostic"

# -- Design parameters -------------------------------------------------------
SEEDS = [42, 123, 456]
CONV_EPISODES = 60          # P0 world-model convergence on the STABLE base env
MEAS_EPISODES = 10          # P1 frozen-model measurement window (per arm)
STEPS_PER_EPISODE = 90
PROBE_STEPS = 100           # positive-control probe length (single episode each)
PROBE_BATTERY_SIZE = 64     # FIXED held-out probe-battery: # one-step transitions
EPISODES_PER_RUN = CONV_EPISODES + MEAS_EPISODES   # progress denominator M (per cell)

# E2 world-forward online training (SD-056). PRIMARY = reconstruction MSE. The
# RECON-ONLY P0 recipe OMITS the contrastive auxiliary term (use_contrastive=False);
# the 701b ablation isolated that auxiliary as the P0 destabiliser. SD056_WEIGHT is
# retained only so the agent config (and thus the world_forward module) is identical
# to the proven-converging recon-only ablation arm; the term is never added to the
# P0 loss in this experiment.
SD056_WEIGHT = 0.05          # auxiliary contrastive weight (UNUSED here; module parity)
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats) -------
MIN_REL_PE_RESPONSE = 0.25   # R1: pe_shock at least 25% above pe_stable (ready seeds)
MIN_REL_CONV_DROP = 0.10     # R2: per-seed FROZEN-PROBE PE drops at least 10%
SEED_PASS_FRAC = 2.0 / 3.0   # R2: at least 2/3 of seeds must converge
MIN_REL_MEL_SPREAD = 0.25    # C1: mel[HIGH] at least 25% above mel[NONE]
ABS_MEL_FLOOR = 1e-4         # C1: absolute HIGH-NONE spread floor (anti-pinned)
MONO_TOL = 0.02              # C1: monotonicity slack (relative to mel[NONE])

# -- Environment base (functional config; mirrors PASSing V3-EXQ-691) --------
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

# Convergence + probe drift settings.
STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)
SHOCK_DRIFT = dict(env_drift_interval=1, env_drift_prob=0.8)

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE", "level": 0, "env_drift_interval": 999, "env_drift_prob": 0.0},
    {"arm_id": "ARM_1_LOW",  "level": 1, "env_drift_interval": 15,  "env_drift_prob": 0.15},
    {"arm_id": "ARM_2_MED",  "level": 2, "env_drift_interval": 6,   "env_drift_prob": 0.30},
    {"arm_id": "ARM_3_HIGH", "level": 3, "env_drift_interval": 2,   "env_drift_prob": 0.50},
]


def _make_env(seed: int, env_drift_interval: int, env_drift_prob: float) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw["env_drift_interval"] = env_drift_interval
    kw["env_drift_prob"] = env_drift_prob
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Functional waking agent with a trainable world-forward model. NO sleep.
    The encoder (latent stack) is FROZEN -- only agent.e2 trains (Adam over
    agent.e2.parameters()) -- which is what makes the frozen-probe battery valid:
    the captured z0/z1 are constant across P0 checkpoints.

    Config is BYTE-IDENTICAL to V3-EXQ-701b: e2_action_contrastive_enabled stays True
    so the world_forward MODULE matches the proven-converging recon-only ablation arm
    exactly. The recon-only recipe omits the contrastive LOSS TERM at train time
    (use_contrastive=False), NOT the head -- so this is the same model that converged
    3/3 in the 701b ablation."""
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
        # E2 world-forward online contrastive HEAD present (module parity with 701b);
        # the loss term is omitted at train time (recon-only).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # MECH-205 surprise EMA (secondary corroborating signal; harmless OFF-sleep).
        surprise_gated_replay=True,
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
    k: int,
    rng: random.Random,
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
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
    use_contrastive: bool = False,
) -> Optional[float]:
    """One P0 world-forward training step. PRIMARY = reconstruction MSE on the
    buffered one-step (z0, a, z1) transitions (the convergence objective). The
    SD-056 contrastive auxiliary is added ONLY when use_contrastive is True; the
    701c main cells call with use_contrastive=False (recon-only, the converged base).

    Returns the reconstruction loss (None if the batch was too small)."""
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)

    optimiser.zero_grad(set_to_none=True)

    # PRIMARY: world-forward reconstruction. world_forward(z0, a) -> z1_pred;
    # minimise ||z1_pred - z1||^2 so the one-step prediction error CONVERGES
    # (the e3 prediction_error the MEL instrument reads is this same mismatch).
    z1_pred = agent.e2.world_forward(z0_K, actions_K)
    recon = F.mse_loss(z1_pred, z1_K)
    recon_val = float(recon.detach().item())
    if not math.isfinite(recon_val):
        return recon_val
    total = recon

    # AUXILIARY: SD-056 action-conditional contrastive (OFF in the recon-only base).
    if use_contrastive:
        closs = agent.e2.world_forward_contrastive_loss(
            z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K,
            simulation_mode=False,
        )
        if torch.is_tensor(closs) and closs.requires_grad:
            c_val = float(closs.detach().item())
            if math.isfinite(c_val) and c_val != 0.0:
                total = total + SD056_WEIGHT * closs

    total.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return recon_val


def _pe_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    pe = metrics.get("e3_prediction_error")
    if pe is None:
        return None
    val = float(pe.detach().item()) if torch.is_tensor(pe) else float(pe)
    return val if math.isfinite(val) else None


def _step_cycle(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_dict: Dict[str, Any],
    train: bool,
    buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer],
    sample_rng: Optional[random.Random],
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    use_contrastive: bool = False,
) -> Tuple[Optional[float], Dict[str, Any], bool]:
    """One waking step. Returns (per_step_pe, next_obs_dict, done).

    train=True: capture (z0,a,z1) transitions + run a P0 world-forward training
                step (recon-only by default). train=False: frozen model, no optimiser,
                no capture (P1 measurement / probes)."""
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
            _e2_train_step(agent, buffer, e2_opt, sample_rng, use_contrastive=use_contrastive)

    _, harm_signal, done, info, next_obs_dict = env.step(action)
    with torch.no_grad():
        metrics = agent.update_residue(
            harm_signal=float(harm_signal), world_delta=None,
            hypothesis_tag=False, owned=True,
        )
    pe = _pe_from_metrics(metrics)
    return pe, next_obs_dict, bool(done)


def _run_window(
    agent: REEAgent,
    env: CausalGridWorldV2,
    n_episodes: int,
    steps: int,
    train: bool,
    buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer],
    sample_rng: Optional[random.Random],
    ep_offset: int,
    arm_id: str,
    seed: int,
    use_contrastive: bool = False,
) -> Tuple[List[float], List[float]]:
    """Run n_episodes; return (all_step_pe, per_episode_mean_pe)."""
    all_pe: List[float] = []
    per_ep_mean: List[float] = []
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
        ep_pe: List[float] = []
        for _step in range(steps):
            pe, obs_dict, done = _step_cycle(
                agent, env, obs_dict, train, buffer, e2_opt, sample_rng,
                pending_capture_ref, use_contrastive=use_contrastive,
            )
            if pe is not None:
                all_pe.append(pe)
                ep_pe.append(pe)
            if done:
                break
        per_ep_mean.append(float(np.mean(ep_pe)) if ep_pe else 0.0)
    return all_pe, per_ep_mean


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


# -- FROZEN held-out probe-state convergence metric (the V3-EXQ-701b instrument) ----
def _sample_probe_battery(
    agent: REEAgent, seed: int, n_transitions: int, steps: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """FIXED held-out probe battery: a deterministic set of one-step
    (z0, action, z1) transitions on the stable base env, captured with the
    AGENT'S OWN (frozen) encoder so z0/z1 live in the same latent space the
    world-forward predicts and are CONSTANT across P0 checkpoints. The frozen-probe
    convergence metric measures world_forward error on this fixed battery,
    removing the exploration/coverage drift that confounded V3-EXQ-701a's
    episode-rollout conv_rel_drop.

    Uses a FIXED action policy (independent of training) so the battery is
    deterministic given (agent-init, seed). Pure read: senses + steps the env;
    never trains e2 and never calls update_residue."""
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
    agent: REEAgent,
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """Mean one-step world_forward reconstruction error over the FIXED probe
    battery. Same battery + frozen encoder -> reads ONLY world_forward
    convergence. This is the same per-element MSE the P0 recon loss minimises and
    the same one-step world mismatch the e3 prediction_error reads."""
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


def _train_p0_and_probe(
    seed: int, conv_eps: int, steps: int, probe_size: int,
    use_contrastive: bool, label: str,
) -> Dict[str, Any]:
    """Build agent, sample the FIXED probe battery (frozen encoder), measure
    frozen-probe PE at P0-start, train P0 (recon-only by default), measure
    frozen-probe PE at P0-end, compute conv_rel_drop on the battery. Returns the
    trained agent + the frozen-probe convergence record."""
    env_stable = _make_env(seed, **STABLE_DRIFT)
    agent = _make_agent(env_stable)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
        maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    # FIXED probe battery captured BEFORE training with the agent's own (frozen)
    # encoder -> z0/z1 are constant across the P0 checkpoints below.
    battery = _sample_probe_battery(agent, seed, probe_size, steps)
    pe_probe_init = _frozen_probe_pe(agent, battery)

    # P0 convergence on the STABLE base env (recon-only when use_contrastive=False).
    _run_window(
        agent, env_stable, conv_eps, steps, train=True, buffer=buffer,
        e2_opt=e2_opt, sample_rng=sample_rng, ep_offset=0, arm_id=label, seed=seed,
        use_contrastive=use_contrastive,
    )
    pe_probe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((pe_probe_init - pe_probe_final) / pe_probe_init)
                     if pe_probe_init > 1e-12 else 0.0)
    return {
        "agent": agent,
        "battery": battery,
        "pe_probe_init": pe_probe_init,
        "pe_probe_final": pe_probe_final,
        "conv_rel_drop": conv_rel_drop,
        "n_probe": len(battery),
    }


def run_cell(arm: Dict[str, Any], seed: int, conv_eps: int, meas_eps: int,
             steps: int, probe_steps: int, probe_size: int) -> Dict[str, Any]:
    """One (arm, seed) cell: RECON-ONLY P0 + FROZEN-PROBE convergence ->
    probes -> P1 measurement (all on the frozen converged model)."""
    arm_id = arm["arm_id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # RECON-ONLY P0 warmup (the 701b-ablation-validated converged base).
    p0 = _train_p0_and_probe(seed, conv_eps, steps, probe_size,
                             use_contrastive=False, label=arm_id)
    agent = p0["agent"]

    # Positive-control probes on the FROZEN model (no training).
    stable_pe, _ = _run_window(
        agent, _make_env(seed, **STABLE_DRIFT), 1, probe_steps, train=False,
        buffer=None, e2_opt=None, sample_rng=None, ep_offset=conv_eps,
        arm_id=arm_id, seed=seed,
    )
    shock_pe, _ = _run_window(
        agent, _make_env(seed, **SHOCK_DRIFT), 1, probe_steps, train=False,
        buffer=None, e2_opt=None, sample_rng=None, ep_offset=conv_eps,
        arm_id=arm_id, seed=seed,
    )
    pe_stable = _mean(stable_pe)
    pe_shock = _mean(shock_pe)
    pe_response_rel = ((pe_shock / pe_stable) - 1.0) if pe_stable > 1e-12 else 0.0

    # P1 measurement on the ARM's graded-novelty env (FROZEN model).
    env_meas = _make_env(seed, arm["env_drift_interval"], arm["env_drift_prob"])
    meas_pe, _ = _run_window(
        agent, env_meas, meas_eps, steps, train=False, buffer=None, e2_opt=None,
        sample_rng=None, ep_offset=conv_eps + 1, arm_id=arm_id, seed=seed,
    )
    mel_mean_pe = _mean(meas_pe)
    mel_sum_pe = float(np.sum(meas_pe)) if meas_pe else 0.0

    print("verdict: PASS", flush=True)
    return {
        "arm_id": arm_id,
        "level": arm["level"],
        "seed": seed,
        "mel_mean_pe": mel_mean_pe,
        "mel_sum_pe": mel_sum_pe,
        "pe_stable": pe_stable,
        "pe_shock": pe_shock,
        "pe_response_rel": pe_response_rel,
        "conv_metric": "frozen_probe_battery",
        "p0_recipe": "recon_only",
        "conv_pe_init": p0["pe_probe_init"],
        "conv_pe_final": p0["pe_probe_final"],
        "conv_rel_drop": p0["conv_rel_drop"],
        "n_probe": p0["n_probe"],
        "n_meas_pe": len(meas_pe),
    }


def evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm_order = [a["arm_id"] for a in sorted(ARMS, key=lambda a: a["level"])]
    seed_list = sorted({r["seed"] for r in rows})
    n_seeds = len(seed_list)

    # -- PER-SEED R2 gate on the FROZEN-PROBE conv_rel_drop. Conv is identical
    #    across arms for a seed (same seed + stable env + budget), so per-seed
    #    conv = mean over arms.
    per_seed_conv = {s: _mean([r["conv_rel_drop"] for r in rows if r["seed"] == s])
                     for s in seed_list}
    per_seed_pe_resp = {s: _mean([r["pe_response_rel"] for r in rows if r["seed"] == s])
                        for s in seed_list}
    ready_seeds = [s for s in seed_list if per_seed_conv[s] >= MIN_REL_CONV_DROP]
    conv_frac = (len(ready_seeds) / n_seeds) if n_seeds else 0.0
    r2_ok = conv_frac >= SEED_PASS_FRAC - 1e-9

    # R1 (pe_response) is interpreted ONLY on R2-passing seeds.
    pe_response_ready = (_mean([per_seed_pe_resp[s] for s in ready_seeds])
                         if ready_seeds else _mean([per_seed_pe_resp[s] for s in seed_list]))
    r1_ok = bool(pe_response_ready >= MIN_REL_PE_RESPONSE)

    # MEL per arm over the R2-passing seeds (fall back to all seeds for the
    # manifest record only when no seed converged; routing still self-routes below).
    score_seeds = ready_seeds if ready_seeds else seed_list
    by_arm = {a: [r for r in rows if r["arm_id"] == a and r["seed"] in score_seeds]
              for a in arm_order}
    mel_per_arm = {a: _mean([r["mel_mean_pe"] for r in by_arm[a]]) for a in arm_order}
    mel_groups = [[r["mel_mean_pe"] for r in by_arm[a]] for a in arm_order]

    mel_none = mel_per_arm[arm_order[0]]
    mel_high = mel_per_arm[arm_order[-1]]

    # C1 monotonicity (per-arm means over ready seeds, slack relative to mel_none).
    tol = MONO_TOL * max(mel_none, ABS_MEL_FLOOR)
    mono = all(
        mel_per_arm[arm_order[i + 1]] >= mel_per_arm[arm_order[i]] - tol
        for i in range(len(arm_order) - 1)
    )
    rel_spread_ok = mel_high >= mel_none * (1.0 + MIN_REL_MEL_SPREAD)
    abs_spread_ok = (mel_high - mel_none) > ABS_MEL_FLOOR
    c1_pass = bool(mono and rel_spread_ok and abs_spread_ok)

    # C2 non-degeneracy (cross-seed spread within arms + arm means differ).
    degeneracy = check_degeneracy({"mel_mean_pe": {"groups": mel_groups}})
    c2_non_degenerate = bool(degeneracy["non_degenerate"])

    arm_mean_vals = [mel_per_arm[a] for a in arm_order]
    arm_means_vary = (max(arm_mean_vals) - min(arm_mean_vals)) > ABS_MEL_FLOOR
    c1_non_degenerate = bool(arm_means_vary and c2_non_degenerate)

    # -- Readiness preconditions (positive control; recomputed by the indexer).
    #    R2 is the LOAD-BEARING per-seed gate on the FROZEN-PROBE metric (fraction
    #    of converged seeds vs the 2/3 floor -- the SAME statistic routing consumes).
    preconditions = [
        {"name": "world_model_converged_p0_seed_fraction",
         "description": "fraction of seeds whose RECON-ONLY FROZEN-PROBE world-forward "
                        "PE dropped >= MIN_REL_CONV_DROP on the FIXED held-out probe "
                        "battery (per-seed R2 gate; recon-only is the 701b-ablation-"
                        "validated converged base)",
         "measured": conv_frac, "threshold": SEED_PASS_FRAC,
         "direction": "lower", "kind": "readiness",
         "control": "per-seed P0 frozen-probe PE relative drop (fixed battery, "
                    "frozen encoder, recon-only P0)",
         "met": bool(r2_ok)},
        {"name": "pe_response_range_to_novelty_shock",
         "description": "converged frozen-model PE rises under a max-novelty shock "
                        "vs stable (relative spread -- same statistic C1 routes on; "
                        "evaluated on R2-passing seeds)",
         "measured": pe_response_ready, "threshold": MIN_REL_PE_RESPONSE,
         "direction": "lower", "kind": "readiness",
         "control": "pe_shock vs pe_stable on the converged frozen model (ready seeds)",
         "met": bool(r1_ok)},
    ]

    if not r2_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        note = ("RECON-ONLY FROZEN held-out probe-state R2: the recon-only P0 world-"
                f"forward did NOT converge on >= 2/3 seeds ({len(ready_seeds)}/{n_seeds}, "
                f"conv_frac={conv_frac:.2f}) on the FIXED probe battery. This is the "
                "701b autopsy BRAKE-LOCK condition: the recon-only base (which converged "
                "3/3 in the 701b matched ablation) ALSO failing here would be a GENUINE "
                "recon-primary world-forward divergence, NOT the SD-056 recipe defect -> "
                "FIRE the INV-050 re-derive brake at the 701c autopsy + route to "
                "/implement-substrate on the world-forward stabilisation substrate. "
                "Self-route substrate_not_ready_requeue (NOT a verdict on INV-050).")
    elif not r1_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        note = ("RECON-ONLY P0 converged on the FROZEN-PROBE metric (>= 2/3 seeds) BUT "
                "the converged frozen-model PE does not rise under a max-novelty shock on "
                f"the ready seeds (pe_response={pe_response_ready:.3f} < {MIN_REL_PE_RESPONSE}). "
                "The positive-control probe cannot detect novelty, so a graded MEL readout "
                "is uninterpretable. Re-queue with a stronger novelty probe. NOT a verdict.")
    elif c1_pass:
        label = "mel_measurable_monotonic"
        outcome = "PASS"
        evidence_direction = "supports"
        note = ("CONVERGED recon-only P0 certified on the FROZEN-PROBE metric (R2 met on "
                ">= 2/3 seeds; recon-only base, exploration + SD-056-recipe confounds "
                "removed) + PE responds to a shock (R1). MEL is a measurable, graded, "
                "monotonic quantity under graded waking novelty. INV-050 IV precondition "
                "MET; the deferred adaptive sleep-cadence enrichment now has a non-vacuous "
                "IV to respond to. PROMOTES NOTHING (diagnostic, scoring-excluded).")
    else:
        label = "mel_not_modulated_by_novelty"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        note = ("DECISIVE: CONVERGED recon-only P0 certified on the FROZEN-PROBE metric "
                "(R2 met on >= 2/3 seeds) + PE responds to a shock (R1) but graded env-"
                "drift novelty does NOT produce a graded MEL signal -- the GENUINE MEL-"
                "measurability ceiling on V3, no longer the V3-EXQ-642/701/701a/701b "
                "convergence-instrument / SD-056-recipe confound. Route to substrate "
                "enrichment / a stronger MEL manipulation; this clean reading should FIRE "
                "the INV-050 re-derive brake at autopsy. NOT a falsification of INV-050.")

    return {
        "outcome": outcome,
        "label": label,
        "evidence_direction": evidence_direction,
        "note": note,
        "mel_per_arm": mel_per_arm,
        "mel_groups": {arm_order[i]: mel_groups[i] for i in range(len(arm_order))},
        "arm_order": arm_order,
        "per_seed_conv_rel_drop": per_seed_conv,
        "per_seed_pe_response_rel": per_seed_pe_resp,
        "ready_seeds": ready_seeds,
        "conv_seed_fraction": conv_frac,
        "pe_response_ready_seeds": pe_response_ready,
        "preconditions": preconditions,
        "criteria_non_degenerate": {"C1": c1_non_degenerate, "C2": c2_non_degenerate},
        "criteria": [
            {"name": "C1_mel_measurable_monotonic", "load_bearing": True,
             "passed": c1_pass,
             "detail": {"monotone": mono, "rel_spread_ok": rel_spread_ok,
                        "abs_spread_ok": abs_spread_ok,
                        "mel_none": mel_none, "mel_high": mel_high,
                        "rel_spread_threshold": MIN_REL_MEL_SPREAD,
                        "scored_seeds": score_seeds}},
            {"name": "C2_non_degenerate", "load_bearing": False,
             "passed": c2_non_degenerate, "detail": degeneracy},
        ],
        "degeneracy": degeneracy,
        "readiness_ok": bool(r2_ok and r1_ok),
        "r2_ok": bool(r2_ok),
        "r1_ok": bool(r1_ok),
        "c1_pass": c1_pass,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [42]
        conv_eps, meas_eps, steps, probe_steps, probe_size = 3, 2, 12, 12, 8
    else:
        seeds = SEEDS
        conv_eps, meas_eps, steps, probe_steps, probe_size = (
            CONV_EPISODES, MEAS_EPISODES, STEPS_PER_EPISODE, PROBE_STEPS,
            PROBE_BATTERY_SIZE)

    cell_config = {
        "env_base": ENV_BASE, "conv_eps": conv_eps, "meas_eps": meas_eps,
        "steps": steps, "probe_steps": probe_steps, "probe_size": probe_size,
        "sd056_weight": SD056_WEIGHT, "e2_lr": E2_LR,
        "p0_loss": "recon_only",
        "conv_metric": "frozen_probe_battery",
    }

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            slice_cfg = dict(cell_config)
            slice_cfg.update({"arm_id": arm["arm_id"],
                              "env_drift_interval": arm["env_drift_interval"],
                              "env_drift_prob": arm["env_drift_prob"]})
            with arm_cell(seed, config_slice=slice_cfg,
                          script_path=Path(__file__)) as cell:
                row = run_cell(arm, seed, conv_eps, meas_eps, steps, probe_steps,
                               probe_size)
                cell.stamp(row)
            rows.append(row)

    ev = evaluate(rows)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": ev["outcome"],
        "result": ev["outcome"],
        "evidence_direction": ev["evidence_direction"],
        "evidence_direction_note": ev["note"],
        "interpretation": {
            "label": ev["label"],
            "preconditions": ev["preconditions"],
            "criteria_non_degenerate": ev["criteria_non_degenerate"],
        },
        "criteria": ev["criteria"],
        "summary": {
            "mel_mean_pe_per_arm": ev["mel_per_arm"],
            "mel_mean_pe_groups_per_arm": ev["mel_groups"],
            "arm_order_by_novelty": ev["arm_order"],
            "per_seed_conv_rel_drop": ev["per_seed_conv_rel_drop"],
            "per_seed_pe_response_rel": ev["per_seed_pe_response_rel"],
            "ready_seeds": ev["ready_seeds"],
            "conv_seed_fraction": ev["conv_seed_fraction"],
            "pe_response_ready_seeds": ev["pe_response_ready_seeds"],
            "readiness_ok": ev["readiness_ok"],
            "r2_ok": ev["r2_ok"],
            "r1_ok": ev["r1_ok"],
            "c1_pass": ev["c1_pass"],
            "conv_metric": "frozen_probe_battery",
            "p0_recipe": "recon_only",
            "thresholds": {
                "MIN_REL_PE_RESPONSE": MIN_REL_PE_RESPONSE,
                "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
                "SEED_PASS_FRAC": SEED_PASS_FRAC,
                "MIN_REL_MEL_SPREAD": MIN_REL_MEL_SPREAD,
                "ABS_MEL_FLOOR": ABS_MEL_FLOOR,
                "MONO_TOL": MONO_TOL,
            },
        },
        "config": {
            "seeds": seeds, "conv_episodes": conv_eps, "meas_episodes": meas_eps,
            "steps_per_episode": steps, "probe_steps": probe_steps,
            "probe_battery_size": probe_size,
            "episodes_per_run": EPISODES_PER_RUN, "arms": ARMS, "env_base": ENV_BASE,
            "stable_drift": STABLE_DRIFT, "shock_drift": SHOCK_DRIFT,
            "p0_loss": "recon_only", "sd056_weight": SD056_WEIGHT,
            "e2_lr": E2_LR, "conv_metric": "frozen_probe_battery", "sleep_used": False,
        },
        "arm_results": rows,
    }
    manifest.update(ev["degeneracy"])

    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"\nResults written to {out_path}")
    print(f"Outcome: {ev['outcome']}  Label: {ev['label']}")
    print(f"MEL per arm: {ev['mel_per_arm']}")
    print(f"Conv seed fraction (recon-only frozen-probe): {ev['conv_seed_fraction']:.2f}  "
          f"ready seeds: {ev['ready_seeds']}")
    print(f"PE response (ready): {ev['pe_response_ready_seeds']:.4f}  "
          f"R2: {ev['r2_ok']}  R1: {ev['r1_ok']}  C1 pass: {ev['c1_pass']}")
    return {"outcome": ev["outcome"], "out_path": str(out_path)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Minimal smoke test (1 seed, 3+2 episodes, short steps)")
    args = parser.parse_args()

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE} (supersedes {SUPERSEDES})")
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
