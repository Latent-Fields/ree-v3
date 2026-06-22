#!/opt/local/bin/python3
"""
V3-EXQ-701 -- INV-050 MEL-measurability diagnostic (commitment-free)

CLAIM UNDER TEST: INV-050
  "Sleep phase architecture is regulated by three drives -- circadian, homeostatic,
  and a learning/model-update demand drive proportional to daily prediction error
  accumulation (Model Error Load, MEL)." The novel, falsifiable content over the
  standard two-process model is the THIRD (MEL) drive.

WHY THIS SHAPE (read first):
  INV-050's directly-testable corollary MECH-180 (novelty adaptively upregulates
  the sleep learning drive) was tested by V3-EXQ-677 and FAILed non_contributory +
  degenerate (confirmed autopsy failure_autopsy_V3-EXQ-677_2026-06-14). TWO
  independent failures there:
    (a) the sleep-amount DV is scheduler-pinned (SWS/REM counts derive from
        sleep_interval, ZERO cross-arm variance) -- the substrate has NO channel
        by which wake prediction-error load scales offline-update resources. That
        adaptive-sleep-cadence enrichment build is DEFERRED (user, 2026-06-14), so
        the sleep DV is OFF-LIMITS here (it would only re-derive the ceiling).
    (b) the novelty manipulation never moved the prediction-error signal at all
        (HIGH-LOW mean E1 loss diff = 8.8e-7) -- so even the INDEPENDENT VARIABLE
        the whole INV-050 thesis rests on (MEL = accumulated prediction error
        during waking) was never shown to be a measurable, manipulable quantity.

  This experiment tests ONLY (b) -- the IV precondition the 677 autopsy flagged as
  the necessary next step ("any future MECH-180 test must instrument an *adaptive*
  sleep-resource variable, not the scheduled count" AND "the manipulation itself
  needs validation before it can drive a sleep response"). It touches NO sleep
  machinery (commitment-free), reads only the existing per-step world-model
  prediction error, and asks: does a GRADED waking novelty manipulation produce a
  measurable, MONOTONIC, accumulated prediction-error (MEL) signal?

  677's measurement bug: it averaged agent.compute_prediction_loss() (a batch
  TRAINING loss the encoder drives down) over the whole run, washing out the
  novelty transient. This experiment instead reads the RAW per-step world-model
  mismatch e3.post_action_update()["prediction_error"] (actual_z_world minus the
  E2 one-step rollout prediction; updated every step per ARC-016), accumulates it
  over a measurement window with the model FROZEN (no adaptation to wash it out),
  and compares the accumulated MEL across graded novelty levels.

EXPERIMENT PURPOSE: diagnostic
  (Excluded from governance confidence/conflict scoring; self-route is a
  hypothesis adjudicated by the pipeline, not a verdict.)

DESIGN:
  4 graded novelty arms x 3 seeds (42, 123, 456). Per (seed, arm) cell:
    P0 CONVERGENCE (identical across arms for a given seed): train the E2
      world-forward model (SD-056 online contrastive) on the STABLE base env
      (drift OFF) for CONV_EPISODES. Because the seed + stable env + step budget
      are identical across arms, the frozen model at the end of P0 is the SAME for
      every arm of a seed -- the controlled comparison isolates novelty exposure.
    POSITIVE-CONTROL PROBES (frozen model): short window on the stable env
      (pe_stable) and on a MAX-novelty env (pe_shock). The readiness gate routes on
      the PE RESPONSE RANGE (pe_shock vs pe_stable) -- the SAME kind of statistic the
      load-bearing C1 criterion routes on (cross-arm MEL spread), per the
      V3-EXQ-643 same-statistic rule.
    P1 MEASUREMENT (frozen model -- no training, no adaptation): run MEAS_EPISODES
      on the arm's graded-novelty env, accumulating per-step prediction_error.
      MEL = mean per-step prediction_error over the measurement window.

  Arms (novelty via CausalGridWorldV2 env drift):
    ARM_0_NONE  drift_interval=999 drift_prob=0.00   (stable; matched-activity control)
    ARM_1_LOW   drift_interval=15  drift_prob=0.15
    ARM_2_MED   drift_interval=6   drift_prob=0.30
    ARM_3_HIGH  drift_interval=2   drift_prob=0.50

  NO sleep flags are set (use_sleep_loop / sws_enabled / rem_enabled all default
  False) -- this is a waking-side IV probe only.

PRIMARY DEPENDENT VARIABLE:
  mel_mean_pe = mean per-step e3 prediction_error over the frozen measurement window
                (per (seed, arm) cell).

READINESS PRECONDITIONS (P0 positive control; gate BEFORE trusting the verdict):
  R1 pe_response_range: (pe_shock / pe_stable - 1) >= MIN_REL_PE_RESPONSE
       -- the frozen model's PE actually RISES under a known novelty shock (a
          spread, the same statistic C1 routes on). Below floor -> the PE signal
          does not respond to env novelty at all -> substrate_not_ready_requeue.
  R2 world_model_converged: P0 relative PE drop (pe_init - pe_final)/pe_init >=
       MIN_REL_CONV_DROP -- the world-forward model actually learned (else MEL is
       noise, not model error). Below floor -> substrate_not_ready_requeue.

ACCEPTANCE CRITERIA:
  C1 (LOAD-BEARING) MEL is measurable + monotonic in novelty:
       monotone non-decreasing mel[NONE]<=mel[LOW]<=mel[MED]<=mel[HIGH]
       (per-arm means across seeds, MONO_TOL slack) AND
       mel[HIGH] >= mel[NONE] * (1 + MIN_REL_MEL_SPREAD) AND
       (mel[HIGH] - mel[NONE]) > ABS_MEL_FLOOR.
  C2 non-degenerate: per-arm MEL has cross-seed spread (not pinned) AND arm means
       are not all identical (check_degeneracy groups).

INTERPRETATION GRID (one row per outcome -> next action):
  R1 or R2 unmet                 -> substrate_not_ready_requeue
       (frozen model did not converge, or PE does not respond to novelty;
        re-queue at a larger P0 / stronger probe -- NOT a verdict on INV-050).
  C1 PASS + C2 non-degenerate    -> mel_measurable_monotonic
       INV-050 IV precondition MET: MEL is a real, graded, manipulable quantity in
       V3. The adaptive sleep-cadence enrichment (sleep-cadence-pe-driven-
       upregulation; deferred 2026-06-14) is now worth building/queuing -- a
       PE-driven sleep DV would have a non-vacuous IV to respond to.
  C1 FAIL (substrate ready)      -> mel_not_modulated_by_novelty
       PE is measurable but env-drift novelty does not produce a graded MEL signal
       on this substrate. Before the sleep-cadence enrichment is worth building, a
       different / stronger MEL manipulation (causal-structure shift, reward-
       structure change) or finer PE instrumentation is needed. NOT a falsification
       of INV-050 -- a gap in the manipulability of the IV.

claim_ids: ["INV-050"]
experiment_purpose: "diagnostic"
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

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from _metrics import check_degeneracy
from experiments._lib.arm_fingerprint import arm_cell

EXPERIMENT_TYPE = "v3_exq_701_inv050_mel_measurability_diagnostic"
QUEUE_ID = "V3-EXQ-701"
CLAIM_IDS = ["INV-050"]
EXPERIMENT_PURPOSE = "diagnostic"

# -- Design parameters -------------------------------------------------------
SEEDS = [42, 123, 456]
CONV_EPISODES = 20          # P0 world-model convergence on the STABLE base env
MEAS_EPISODES = 10          # P1 frozen-model measurement window (per arm)
STEPS_PER_EPISODE = 90
PROBE_STEPS = 100           # positive-control probe length (single episode each)
EPISODES_PER_RUN = CONV_EPISODES + MEAS_EPISODES   # progress denominator M (per cell)

# E2 world-forward online contrastive training (SD-056) -- known-good 691 recipe.
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats) -------
MIN_REL_PE_RESPONSE = 0.25   # R1: pe_shock at least 25% above pe_stable
MIN_REL_CONV_DROP = 0.10     # R2: P0 PE drops at least 10% (model learned)
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
    """Functional waking agent with a trainable world-forward model. NO sleep."""
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
        # E2 world-forward online contrastive (the model whose PE we read).
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


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


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
) -> Tuple[Optional[float], Dict[str, Any], bool]:
    """One waking step. Returns (per_step_pe, next_obs_dict, done).

    train=True: capture (z0,a,z1) transitions + run an SD-056 contrastive step
                (P0 convergence). train=False: frozen model, no optimiser, no
                capture (P1 measurement / probes)."""
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)

    latent = agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=_obs(obs_dict, "harm_obs"),
        obs_harm_a=_obs(obs_dict, "harm_obs_a"),
        obs_harm_history=_obs(obs_dict, "harm_history"),
    )

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
            _e2_contrastive_step(agent, buffer, e2_opt, sample_rng)

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
                pending_capture_ref,
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


def run_cell(arm: Dict[str, Any], seed: int, conv_eps: int, meas_eps: int,
             steps: int, probe_steps: int) -> Dict[str, Any]:
    """One (arm, seed) cell: P0 convergence (stable) -> probes -> P1 measurement."""
    arm_id = arm["arm_id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    env_stable = _make_env(seed, **STABLE_DRIFT)
    agent = _make_agent(env_stable)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
        maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    # P0 convergence on the STABLE base env (identical across arms for a seed).
    _, conv_ep_means = _run_window(
        agent, env_stable, conv_eps, steps, train=True, buffer=buffer,
        e2_opt=e2_opt, sample_rng=sample_rng, ep_offset=0, arm_id=arm_id, seed=seed,
    )
    # Convergence readiness: relative PE drop across P0 (first vs last episode mean).
    pe_init = conv_ep_means[0] if conv_ep_means else 0.0
    pe_final = conv_ep_means[-1] if conv_ep_means else 0.0
    conv_rel_drop = ((pe_init - pe_final) / pe_init) if pe_init > 1e-12 else 0.0

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
        "conv_pe_init": pe_init,
        "conv_pe_final": pe_final,
        "conv_rel_drop": conv_rel_drop,
        "n_meas_pe": len(meas_pe),
    }


def evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {a["arm_id"]: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    arm_order = [a["arm_id"] for a in sorted(ARMS, key=lambda a: a["level"])]
    mel_per_arm = {a: _mean([r["mel_mean_pe"] for r in by_arm[a]]) for a in arm_order}
    mel_groups = [[r["mel_mean_pe"] for r in by_arm[a]] for a in arm_order]

    pe_response_mean = _mean([r["pe_response_rel"] for r in rows])
    conv_drop_mean = _mean([r["conv_rel_drop"] for r in rows])

    # -- Readiness preconditions (positive control; recomputed by the indexer) --
    preconditions = [
        {"name": "pe_response_range_to_novelty_shock",
         "description": "frozen-model PE rises under a max-novelty shock vs stable "
                        "(relative spread -- same statistic C1 routes on)",
         "measured": pe_response_mean, "threshold": MIN_REL_PE_RESPONSE,
         "direction": "lower", "control": "pe_shock vs pe_stable on the converged "
                                           "frozen model",
         "kind": "readiness",
         "met": bool(pe_response_mean >= MIN_REL_PE_RESPONSE)},
        {"name": "world_model_converged_p0",
         "description": "P0 world-forward PE dropped (model learned the base env)",
         "measured": conv_drop_mean, "threshold": MIN_REL_CONV_DROP,
         "direction": "lower", "control": "P0 first-vs-last episode mean PE",
         "kind": "readiness",
         "met": bool(conv_drop_mean >= MIN_REL_CONV_DROP)},
    ]
    ready = all(p["met"] for p in preconditions)

    mel_none = mel_per_arm[arm_order[0]]
    mel_high = mel_per_arm[arm_order[-1]]

    # C1 monotonicity (per-arm means, with slack relative to mel_none).
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

    # C1 discriminates iff arm means are not all identical AND there is cross-seed
    # spread somewhere (else C1 passed/failed trivially).
    arm_mean_vals = [mel_per_arm[a] for a in arm_order]
    arm_means_vary = (max(arm_mean_vals) - min(arm_mean_vals)) > ABS_MEL_FLOOR
    c1_non_degenerate = bool(arm_means_vary and c2_non_degenerate)

    if not ready:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        note = ("Readiness unmet (PE does not respond to novelty shock, or P0 world "
                "model did not converge). Re-queue at larger P0 / stronger probe. "
                "NOT a verdict on INV-050.")
    elif c1_pass:
        label = "mel_measurable_monotonic"
        outcome = "PASS"
        evidence_direction = "supports"
        note = ("MEL is a measurable, graded, monotonic quantity under graded waking "
                "novelty. INV-050 IV precondition MET; the deferred adaptive "
                "sleep-cadence enrichment (sleep-cadence-pe-driven-upregulation) now "
                "has a non-vacuous IV to respond to and is worth building.")
    else:
        label = "mel_not_modulated_by_novelty"
        outcome = "FAIL"
        evidence_direction = "unknown"
        note = ("PE is measurable but env-drift novelty did not produce a graded MEL "
                "signal. A different/stronger MEL manipulation or finer PE "
                "instrumentation is needed before the sleep-cadence enrichment is "
                "worth building. NOT a falsification of INV-050.")

    return {
        "outcome": outcome,
        "label": label,
        "evidence_direction": evidence_direction,
        "note": note,
        "mel_per_arm": mel_per_arm,
        "mel_groups": {arm_order[i]: mel_groups[i] for i in range(len(arm_order))},
        "arm_order": arm_order,
        "pe_response_mean": pe_response_mean,
        "conv_drop_mean": conv_drop_mean,
        "preconditions": preconditions,
        "criteria_non_degenerate": {"C1": c1_non_degenerate, "C2": c2_non_degenerate},
        "criteria": [
            {"name": "C1_mel_measurable_monotonic", "load_bearing": True,
             "passed": c1_pass,
             "detail": {"monotone": mono, "rel_spread_ok": rel_spread_ok,
                        "abs_spread_ok": abs_spread_ok,
                        "mel_none": mel_none, "mel_high": mel_high,
                        "rel_spread_threshold": MIN_REL_MEL_SPREAD}},
            {"name": "C2_non_degenerate", "load_bearing": False,
             "passed": c2_non_degenerate, "detail": degeneracy},
        ],
        "degeneracy": degeneracy,
        "readiness_ok": ready,
        "c1_pass": c1_pass,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [42]
        conv_eps, meas_eps, steps, probe_steps = 2, 2, 12, 12
    else:
        seeds = SEEDS
        conv_eps, meas_eps, steps, probe_steps = (
            CONV_EPISODES, MEAS_EPISODES, STEPS_PER_EPISODE, PROBE_STEPS)

    cell_config = {
        "env_base": ENV_BASE, "conv_eps": conv_eps, "meas_eps": meas_eps,
        "steps": steps, "probe_steps": probe_steps, "sd056_weight": SD056_WEIGHT,
        "e2_lr": E2_CONTRASTIVE_LR,
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
                row = run_cell(arm, seed, conv_eps, meas_eps, steps, probe_steps)
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
            "pe_response_rel_mean": ev["pe_response_mean"],
            "conv_rel_drop_mean": ev["conv_drop_mean"],
            "readiness_ok": ev["readiness_ok"],
            "c1_pass": ev["c1_pass"],
            "thresholds": {
                "MIN_REL_PE_RESPONSE": MIN_REL_PE_RESPONSE,
                "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
                "MIN_REL_MEL_SPREAD": MIN_REL_MEL_SPREAD,
                "ABS_MEL_FLOOR": ABS_MEL_FLOOR,
                "MONO_TOL": MONO_TOL,
            },
        },
        "config": {
            "seeds": seeds, "conv_episodes": conv_eps, "meas_episodes": meas_eps,
            "steps_per_episode": steps, "probe_steps": probe_steps,
            "episodes_per_run": EPISODES_PER_RUN, "arms": ARMS, "env_base": ENV_BASE,
            "stable_drift": STABLE_DRIFT, "shock_drift": SHOCK_DRIFT,
            "sleep_used": False,
        },
        "arm_results": rows,
    }
    manifest.update(ev["degeneracy"])

    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(f"Outcome: {ev['outcome']}  Label: {ev['label']}")
    print(f"MEL per arm: {ev['mel_per_arm']}")
    print(f"PE response (rel): {ev['pe_response_mean']:.4f}  "
          f"Conv drop (rel): {ev['conv_drop_mean']:.4f}")
    print(f"Readiness OK: {ev['readiness_ok']}  C1 pass: {ev['c1_pass']}")
    return {"outcome": ev["outcome"], "out_path": str(out_path)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Minimal smoke test (1 seed, 2+2 episodes, short steps)")
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
