"""V3-EXQ-691: Q-055 -- does SD-017 SWS-phase sleep consolidation PRESERVE or
ERODE the trajectory-class diversity ARC-065 achieves during waking?

SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a dedicated N_CYCLES wake-sleep-test loop)

Rung-3 persistence test (behavioral_diversity_acceptance_criteria.md "Rung 3 --
Persistence after training/replay"). Authoritative design source:
REE_assembly/evidence/planning/q054_q055_q056_buildability_triage_2026-06-19.md
"Q-055" section.

THE QUESTION (Q-047 in the acceptance-criteria doc): the monostrategy literature
(MECH-120) predicts Hebbian winner-take-all replay preferentially strengthens the
most-recently / most-heavily-weighted trajectories, so SD-017 sleep replay could
systematically UNDO the trajectory-class diversity ARC-065 generates during
waking. SD-017 (stable) is hypothesised to PRESERVE it (and INV-049 sleep
necessity). This is the FIRST direct empirical test of the SD-017 <-> MECH-120
interaction on the committed-action-class diversity DV.

SCIENTIFICALLY-INTERESTING PRIOR (triage doc): ARC-065's diversity now reaches
committed action via the 569i TOP-K shortlist conversion -- a SELECTION-LAYER
mechanism, NOT learned synaptic weights. Sleep replay strengthens WEIGHTS
(Hebbian). So the MECH-120 erosion pathway may not even touch a selection-layer
diversity source -- a clean dissociation of "diversity lives in weights" vs
"diversity lives in the selector". This experiment settles it.

DESIGN (3 time points x 3 arms x >= 3 seeds):
  Run to waking convergence with ARC-065 ON + the 569i top-k conversion stack
  armed (so diversity reaches committed action). Measure the committed-action-class
  diversity DV (TrajDiv / C_R1B = selected_action_class_entropy) at three points:
    t0 = post-waking-convergence
    t1 = after +T1_EPISODES episodes with FORCED EXPLORATION OFF (temperature=1.0,
         no scaffold, no extra diversity boost)
    t2 = after +N_SLEEP_CYCLES manual SD-017 sleep cycles (run_sleep_cycle())
  ARMS:
    (A) ARM_SLEEP_ON       sws_enabled=True,  rem_enabled=True   (the test arm)
    (B) ARM_SLEEP_OFF      sws_enabled=False, rem_enabled=False  (control: the
        "must hold WITH sleep active" criterion makes OFF the control; if t2
        holds only because sleep is disabled, that is NOT a PASS)
    (C) ARM_REPLAY_ABLATED sws/rem ON but replay_diversity_enabled=False AND
        random_replay_fraction=0.0 (tests the counter-mechanism that replay is
        DESIGNED to sample diverse trajectories -- if diversity is preserved only
        when diverse replay is on, that localises the protection to the replay
        sampler)

DV: committed-action-class diversity C_R1B = selected_action_class_entropy
    (Shannon entropy of the committed first-action class distribution over a
    measurement window). Measured at t0 / t1 / t2 per (seed x arm). >= 3 seeds.

PASS (arm A only is governance-load-bearing):
  t2 diversity within PERSIST_FRACTION (50%) of t0 AND not collapsed to a single
  dominant class (>= 2 committed classes AND entropy > COLLAPSE_FLOOR), WITH sleep
  ACTIVE (arm A). The 3-outcome reading:
    PRESERVES  -> supports SD-017 (+ INV-049); MECH-120-as-protective-risk-mitigated.
    ERODES     -> supports an erosion RISK; weakens SD-017-preserves-diversity and
                  SUPPORTS MECH-120 (Hebbian erosion is real; ARC-065 must also
                  operate during consolidation -> new MECH).
    INCREASES  -> surprising: broader episodic-replay sampling than the waking
                  policy; supports SD-017, neutral-to-supports MECH-120.

NON-VACUITY SELF-ROUTE GATES (substrate_not_ready_requeue if unmet; NEVER weaken):
  R1 sleep cycles actually FIRE: arm A cumulative consolidation write-passes
     (sws_n_writes + rem_n_rollouts) > 0 AND sws_enabled effective.
  R2 t0 diversity is SUPRA-FLOOR: arm A t0 committed-class entropy > T0_FLOOR
     (there IS diversity to erode -- else the persistence question is vacuous).
  R3 sleep-OFF differs from sleep-ON on SOME consolidation metric (the sleep knob
     is not inert): arm A cumulative write-passes > arm B cumulative write-passes
     (B is 0 by construction; A must be > 0).

claim_ids=["SD-017", "MECH-120"] -> MULTI-CLAIM, so evidence_direction_per_claim
is MANDATORY. experiment_purpose="evidence".

Usage:
  /opt/local/bin/python3 experiments/v3_exq_691_q055_sleep_consolidation_diversity_persistence.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_691_q055_sleep_consolidation_diversity_persistence.py
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_691_q055_sleep_consolidation_diversity_persistence"
QUEUE_ID = "V3-EXQ-691"
CLAIM_IDS: List[str] = ["SD-017", "MECH-120"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]

# Full-scale budget.
T0_CONVERGENCE_EPISODES = 60     # waking convergence with ARC-065 + 569i conversion armed
T1_EPISODES = 200                # +200 episodes, FORCED EXPLORATION OFF (acceptance-criteria Rung 3)
N_SLEEP_CYCLES = 5               # +5 manual SD-017 sleep cycles (acceptance-criteria Rung 3)
STEPS_PER_EPISODE = 200
MEASURE_EPISODES = 20            # measurement window length (episodes) at each of t0/t1/t2
MEASURE_STEPS = 200

# TOTAL per-seed-x-arm episode count run sequentially across the t0-convergence +
# t1 phases (the progress estimator denominator; sleep cycles are not episodes).
TOTAL_TRAIN_EPISODES = T0_CONVERGENCE_EPISODES + T1_EPISODES

# Dry-run (tiny: 1 seed, few episodes per phase, all 3 arms, 1-2 sleep cycles).
DRY_RUN_SEEDS = [42]
DRY_T0 = 3
DRY_T1 = 4
DRY_SLEEP_CYCLES = 2
DRY_STEPS = 30
DRY_MEASURE_EPISODES = 2
DRY_MEASURE_STEPS = 30

# Pre-registered thresholds (constants).
PERSIST_FRACTION = 0.5           # PASS: t2 entropy >= PERSIST_FRACTION * t0 entropy
COLLAPSE_FLOOR = 0.1             # not-collapsed: t2 entropy > this AND >= 2 classes
T0_FLOOR = 0.2                   # R2: t0 entropy must exceed this (there is diversity to erode)
INCREASE_FRACTION = 1.10         # t2 > 1.10*t0 -> INCREASES reading

# SD-056 online contrastive training (mirror 569i harness so the action-divergent
# pool is genuinely present -- the precondition for 569i top-k conversion).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# 569i TOP-K shortlist conversion config (the constant: arms diverging on sleep).
MODULATORY_SHORTLIST_K = 3
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

# Sleep config (manual-cycle-loop; SD-017 surface).
SWS_CONSOLIDATION_STEPS = 5
SWS_SCHEMA_WEIGHT = 0.1
REM_ATTRIBUTION_STEPS = 10

MEASURE_TEMPERATURE = 1.0        # forced exploration OFF at t1/t2 (and measurement)

# Behavioural-diversity env: SD-054 reef-bipartite hazard layout (matches 569i).
ENV_KWARGS: Dict[str, Any] = dict(
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
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_SLEEP_ON",
        "label": "sd017_sws_rem_on_diverse_replay_on",
        "sws_enabled": True,
        "rem_enabled": True,
        "replay_diversity_enabled": True,
        "random_replay_fraction": 0.2,
    },
    {
        "arm_id": "ARM_SLEEP_OFF",
        "label": "sd017_off_control",
        "sws_enabled": False,
        "rem_enabled": False,
        "replay_diversity_enabled": False,
        "random_replay_fraction": 0.0,
    },
    {
        "arm_id": "ARM_REPLAY_ABLATED",
        "label": "sd017_on_replay_diversity_ablated",
        "sws_enabled": True,
        "rem_enabled": True,
        "replay_diversity_enabled": False,
        "random_replay_fraction": 0.0,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM (ARC-065 Layer A) + 569i TOP-K shortlist conversion (so
    diversity reaches committed action) + SD-017 sleep surface per arm. SD-056
    online contrastive trained on every arm with the rollout-norm clamp."""
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
        # ARC-065 SP-CEM (Layer A) -- action-divergent candidate pool.
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # Shared E3-side bias channels consumed by the routed conversion.
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        # SD-056 substrate trained online (e2.world_forward action divergence).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # ARC-065 GAP-A: e2_world_forward consumed-summary source (diversity reaches commit).
        candidate_summary_source="e2_world_forward",
        # 569i TOP-K shortlist conversion (constant across arms -- diversity converts).
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        # SD-017 sleep surface (manual run_sleep_cycle()). Per-arm.
        sws_enabled=bool(arm["sws_enabled"]),
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        sws_schema_weight=SWS_SCHEMA_WEIGHT,
        rem_enabled=bool(arm["rem_enabled"]),
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        replay_diversity_enabled=bool(arm["replay_diversity_enabled"]),
        random_replay_fraction=float(arm["random_replay_fraction"]),
    )
    agent = REEAgent(cfg)
    return agent


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _first_actions_K(candidates) -> torch.Tensor:
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


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
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
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
        z_world_0=z0_K,
        actions=actions_K,
        z_world_1_targets=z1_K,
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


# ---------------------------------------------------------------------------
# Episode driver (one waking episode; trains SD-056 + collects committed classes)
# ---------------------------------------------------------------------------

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    e2_opt: torch.optim.Optimizer,
    sample_rng: random.Random,
    measure: bool,
    temperature: float,
) -> Tuple[Counter, Optional[str]]:
    """Run one waking episode. When measure=True, accumulate committed
    first-action class counts into the returned Counter (the C_R1B DV)."""
    selected_class_counts: Counter = Counter()
    error_note: Optional[str] = None

    _, obs_dict = env.reset()
    agent.reset()

    z_self_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None
    pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    tick_in_ep = 0

    for _step in range(steps):
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

        if pending_capture is not None:
            z0_prev, a_prev = pending_capture
            z1_obs = latent.z_world.detach().reshape(-1).clone()
            if (
                torch.isfinite(z0_prev).all()
                and torch.isfinite(a_prev).all()
                and torch.isfinite(z1_obs).all()
            ):
                transition_buffer.append((z0_prev, a_prev, z1_obs))
            pending_capture = None

        if z_self_prev is not None and action_prev is not None:
            agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

        ticks = agent.clock.advance()
        wdim = latent.z_world.shape[-1]
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick", False)
            else torch.zeros(1, wdim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)

        if agent.goal_state is not None:
            try:
                energy = float(body[0, 3].item())
            except Exception:
                energy = 1.0
            drive_level = max(0.0, 1.0 - energy)
            agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

        action = agent.select_action(candidates, ticks, temperature=temperature)

        if action is None:
            idx = int(np.random.randint(0, env.action_dim))
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, idx] = 1.0
            agent._last_action = action
        if not torch.isfinite(action).all():
            error_note = "non_finite_action"
            break

        if measure:
            committed_class = int(action[0].argmax().item())
            selected_class_counts[committed_class] += 1

        if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
            pending_capture = (
                latent.z_world.detach().reshape(-1).clone(),
                action.detach().reshape(-1).clone(),
            )

        # SD-056 online contrastive (train every tick; keeps the divergent pool).
        _e2_contrastive_step(
            agent=agent, buffer=transition_buffer, optimiser=e2_opt, rng=sample_rng,
        )

        _, harm_signal, done, info, next_obs_dict = env.step(action)
        with torch.no_grad():
            agent.update_residue(
                harm_signal=float(harm_signal),
                world_delta=None,
                hypothesis_tag=False,
                owned=True,
            )

        z_self_prev = latent.z_self.detach()
        action_prev = action
        obs_dict = next_obs_dict
        tick_in_ep += 1
        if done:
            break

    return selected_class_counts, error_note


def _measure_diversity(
    agent: REEAgent,
    env: CausalGridWorldV2,
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    e2_opt: torch.optim.Optimizer,
    sample_rng: random.Random,
    n_episodes: int,
    steps: int,
) -> Dict[str, Any]:
    """Run a measurement window (forced exploration OFF) and compute the
    committed-action-class diversity DV."""
    counts: Counter = Counter()
    err: Optional[str] = None
    for _ in range(n_episodes):
        c, e = _run_episode(
            agent, env, steps, transition_buffer, e2_opt, sample_rng,
            measure=True, temperature=MEASURE_TEMPERATURE,
        )
        counts.update(c)
        if e is not None and err is None:
            err = e
    entropy = _entropy_from_counts(dict(counts))
    return {
        "selected_action_class_entropy": round(entropy, 6),
        "selected_class_counts": dict(sorted(counts.items())),
        "selected_classes_n_unique": int(len(counts)),
        "n_committed_ticks": int(sum(counts.values())),
        "error_note": err,
    }


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    t0_episodes: int,
    t1_episodes: int,
    n_sleep_cycles: int,
    steps_per_episode: int,
    measure_episodes: int,
    measure_steps: int,
) -> Dict[str, Any]:
    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    error_note: Optional[str] = None
    eps_done = 0
    # TOTAL = the full per-seed-x-arm episode count run sequentially across the
    # t0-convergence + t1 phases (the progress-estimator denominator).
    total_train = int(t0_episodes + t1_episodes)
    log_every = 10 if total_train >= 10 else 1

    # --- Phase: waking convergence (t0 training) ---
    for ep in range(t0_episodes):
        _, e = _run_episode(
            agent, env, steps_per_episode, transition_buffer, e2_opt, sample_rng,
            measure=False, temperature=MEASURE_TEMPERATURE,
        )
        eps_done += 1
        if e is not None and error_note is None:
            error_note = e
        if (eps_done % log_every == 0) or (eps_done == total_train):
            print(
                f"  [train] convergence arm={arm['arm_id']} seed={seed} "
                f"ep {eps_done}/{total_train}",
                flush=True,
            )

    # t0 measurement (post-convergence).
    t0 = _measure_diversity(
        agent, env, transition_buffer, e2_opt, sample_rng,
        measure_episodes, measure_steps,
    )
    if t0.get("error_note") and error_note is None:
        error_note = t0["error_note"]

    # --- Phase: +t1 episodes, FORCED EXPLORATION OFF ---
    for ep in range(t1_episodes):
        _, e = _run_episode(
            agent, env, steps_per_episode, transition_buffer, e2_opt, sample_rng,
            measure=False, temperature=MEASURE_TEMPERATURE,
        )
        eps_done += 1
        if e is not None and error_note is None:
            error_note = e
        if (eps_done % log_every == 0) or (eps_done == total_train):
            print(
                f"  [train] no_explore arm={arm['arm_id']} seed={seed} "
                f"ep {eps_done}/{total_train}",
                flush=True,
            )

    # t1 measurement (after +t1 episodes, no forced exploration).
    t1 = _measure_diversity(
        agent, env, transition_buffer, e2_opt, sample_rng,
        measure_episodes, measure_steps,
    )
    if t1.get("error_note") and error_note is None:
        error_note = t1["error_note"]

    # --- Phase: +N sleep cycles (manual-cycle-loop run_sleep_cycle()) ---
    cumulative_sws_writes = 0.0
    cumulative_rem_rollouts = 0.0
    cumulative_sws_slot_diversity = 0.0
    cumulative_rem_terrain_variance = 0.0
    n_cycles_fired = 0
    for cyc in range(n_sleep_cycles):
        # SD-017 surface: no-op when sws/rem both disabled (arm B); fires for A/C.
        cyc_metrics = agent.run_sleep_cycle() or {}
        sws_w = float(cyc_metrics.get("sws_n_writes", 0.0) or 0.0)
        rem_r = float(cyc_metrics.get("rem_n_rollouts", 0.0) or 0.0)
        cumulative_sws_writes += sws_w
        cumulative_rem_rollouts += rem_r
        cumulative_sws_slot_diversity += float(
            cyc_metrics.get("sws_slot_diversity", 0.0) or 0.0
        )
        cumulative_rem_terrain_variance += float(
            cyc_metrics.get("rem_terrain_variance", 0.0) or 0.0
        )
        if sws_w > 0.0 or rem_r > 0.0:
            n_cycles_fired += 1

    # t2 measurement (after +N sleep cycles).
    t2 = _measure_diversity(
        agent, env, transition_buffer, e2_opt, sample_rng,
        measure_episodes, measure_steps,
    )
    if t2.get("error_note") and error_note is None:
        error_note = t2["error_note"]

    consolidation_write_passes = cumulative_sws_writes + cumulative_rem_rollouts

    # Per-cell verdict (ran-to-completion = PASS; crash = FAIL).
    verdict = "FAIL" if error_note is not None else "PASS"
    print(f"verdict: {verdict}", flush=True)

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "sws_enabled": bool(arm["sws_enabled"]),
        "rem_enabled": bool(arm["rem_enabled"]),
        "replay_diversity_enabled": bool(arm["replay_diversity_enabled"]),
        "random_replay_fraction": float(arm["random_replay_fraction"]),
        "error_note": error_note,
        # 3 time points -- the C_R1B / TrajDiv DV.
        "t0_selected_action_class_entropy": t0["selected_action_class_entropy"],
        "t0_selected_classes_n_unique": t0["selected_classes_n_unique"],
        "t0_selected_class_counts": t0["selected_class_counts"],
        "t0_n_committed_ticks": t0["n_committed_ticks"],
        "t1_selected_action_class_entropy": t1["selected_action_class_entropy"],
        "t1_selected_classes_n_unique": t1["selected_classes_n_unique"],
        "t1_selected_class_counts": t1["selected_class_counts"],
        "t1_n_committed_ticks": t1["n_committed_ticks"],
        "t2_selected_action_class_entropy": t2["selected_action_class_entropy"],
        "t2_selected_classes_n_unique": t2["selected_classes_n_unique"],
        "t2_selected_class_counts": t2["selected_class_counts"],
        "t2_n_committed_ticks": t2["n_committed_ticks"],
        # Sleep consolidation telemetry (R1 / R3).
        "n_sleep_cycles_requested": int(n_sleep_cycles),
        "n_sleep_cycles_fired": int(n_cycles_fired),
        "cumulative_sws_writes": round(cumulative_sws_writes, 6),
        "cumulative_rem_rollouts": round(cumulative_rem_rollouts, 6),
        "consolidation_write_passes": round(consolidation_write_passes, 6),
        "cumulative_sws_slot_diversity": round(cumulative_sws_slot_diversity, 6),
        "cumulative_rem_terrain_variance": round(cumulative_rem_terrain_variance, 6),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _n_seeds(rows: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for r in rows if predicate(r))


def _evaluate(arm_results: List[Dict[str, Any]], min_seeds: int) -> Dict[str, Any]:
    armA = _arm_rows(arm_results, "ARM_SLEEP_ON")
    armB = _arm_rows(arm_results, "ARM_SLEEP_OFF")
    armC = _arm_rows(arm_results, "ARM_REPLAY_ABLATED")

    T0 = "t0_selected_action_class_entropy"
    T2 = "t2_selected_action_class_entropy"

    # --- Non-vacuity gates (arm A is the governance-load-bearing arm) ---
    # R1: sleep cycles actually fire (write-passes > 0) on >= min_seeds, sws effective.
    r1_seeds = _n_seeds(
        armA,
        lambda r: float(r.get("consolidation_write_passes", 0.0)) > 0.0
        and bool(r.get("sws_enabled", False)),
    )
    r1_ok = bool(r1_seeds >= min_seeds)

    # R2: t0 diversity supra-floor (there IS diversity to erode) on >= min_seeds.
    r2_seeds = _n_seeds(armA, lambda r: float(r.get(T0, 0.0)) > T0_FLOOR)
    r2_ok = bool(r2_seeds >= min_seeds)

    # R3: sleep-OFF differs from sleep-ON on a consolidation metric (knob not inert).
    armA_writes = _mean_key(armA, "consolidation_write_passes")
    armB_writes = _mean_key(armB, "consolidation_write_passes")
    r3_ok = bool(armA_writes > armB_writes)

    readiness_ok = bool(r1_ok and r2_ok and r3_ok)

    # --- Persistence reading on arm A (per seed, then aggregate) ---
    def _persist_label(r: Dict[str, Any]) -> str:
        t0v = float(r.get(T0, 0.0))
        t2v = float(r.get(T2, 0.0))
        n2 = int(r.get("t2_selected_classes_n_unique", 0))
        collapsed = bool(t2v <= COLLAPSE_FLOOR or n2 < 2)
        if collapsed:
            return "erodes"
        if t0v <= 1e-9:
            # no t0 diversity to compare; treat as vacuous (handled by R2 gate)
            return "increases" if t2v > 0.0 else "erodes"
        ratio = t2v / t0v
        if ratio >= INCREASE_FRACTION:
            return "increases"
        if ratio >= PERSIST_FRACTION:
            return "preserves"
        return "erodes"

    armA_labels = [_persist_label(r) for r in armA]
    n_preserves = sum(1 for x in armA_labels if x == "preserves")
    n_increases = sum(1 for x in armA_labels if x == "increases")
    n_erodes = sum(1 for x in armA_labels if x == "erodes")

    # PASS (arm A) = t2 within PERSIST_FRACTION of t0 AND not collapsed, on >= min_seeds.
    # preserves OR increases both satisfy "within 50% and not collapsed".
    n_persist_or_increase = n_preserves + n_increases
    persistence_pass = bool(n_persist_or_increase >= min_seeds)

    # 3-outcome aggregate reading (which dominates arm A across seeds).
    if persistence_pass and n_increases > n_preserves:
        reading = "increases"
    elif persistence_pass:
        reading = "preserves"
    else:
        reading = "erodes"

    # Non-degeneracy: every measured arm produced committed ticks at all 3 points.
    all_arms = [armA, armB, armC]
    non_degenerate = bool(
        all(len(a) > 0 for a in all_arms)
        and all(
            int(r.get("t0_n_committed_ticks", 0)) > 0
            and int(r.get("t1_n_committed_ticks", 0)) > 0
            and int(r.get("t2_n_committed_ticks", 0)) > 0
            for a in all_arms for r in a
        )
    )

    # --- Verdict + evidence directions ---
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        ed_sd017 = "non_contributory"
        ed_mech120 = "non_contributory"
    elif reading in ("preserves", "increases"):
        label = (
            "sleep_preserves_diversity" if reading == "preserves"
            else "sleep_increases_diversity"
        )
        overall_pass = True
        # SD-017 sleep PRESERVES (supports SD-017 + INV-049).
        ed_sd017 = "supports"
        # MECH-120 Hebbian-erosion-as-mitigated-risk: preservation supports the
        # MECH-120-as-protective reading (replay resists monostrategy).
        ed_mech120 = "supports"
    else:  # erodes
        label = "sleep_erodes_diversity"
        overall_pass = False
        # SD-017 sleep ERODES diversity (weakens SD-017-preserves-diversity).
        ed_sd017 = "weakens"
        # Erosion observed -> Hebbian winner-take-all risk is real -> supports MECH-120.
        ed_mech120 = "supports"

    overall = "supports" if overall_pass else (
        "weakens" if label == "sleep_erodes_diversity" else "non_contributory"
    )

    return {
        "readiness": {
            "r1_sleep_cycles_fire": r1_ok,
            "r1_seeds_writes_positive": int(r1_seeds),
            "r2_t0_diversity_supra_floor": r2_ok,
            "r2_seeds_t0_above_floor": int(r2_seeds),
            "t0_floor": T0_FLOOR,
            "r3_sleep_knob_not_inert": r3_ok,
            "armA_consolidation_write_passes_mean": round(armA_writes, 6),
            "armB_consolidation_write_passes_mean": round(armB_writes, 6),
            "min_seeds_required": min_seeds,
            "readiness_ok": readiness_ok,
        },
        "persistence": {
            "armA_per_seed_labels": armA_labels,
            "n_preserves": int(n_preserves),
            "n_increases": int(n_increases),
            "n_erodes": int(n_erodes),
            "n_persist_or_increase": int(n_persist_or_increase),
            "persist_fraction": PERSIST_FRACTION,
            "increase_fraction": INCREASE_FRACTION,
            "collapse_floor": COLLAPSE_FLOOR,
            "persistence_pass": persistence_pass,
            "reading": reading,
        },
        "entropy_per_arm": {
            "ARM_SLEEP_ON": {
                "t0": round(_mean_key(armA, T0), 6),
                "t1": round(_mean_key(armA, "t1_selected_action_class_entropy"), 6),
                "t2": round(_mean_key(armA, T2), 6),
            },
            "ARM_SLEEP_OFF": {
                "t0": round(_mean_key(armB, T0), 6),
                "t1": round(_mean_key(armB, "t1_selected_action_class_entropy"), 6),
                "t2": round(_mean_key(armB, T2), 6),
            },
            "ARM_REPLAY_ABLATED": {
                "t0": round(_mean_key(armC, T0), 6),
                "t1": round(_mean_key(armC, "t1_selected_action_class_entropy"), 6),
                "t2": round(_mean_key(armC, T2), 6),
            },
        },
        "label": label,
        "evidence_direction": overall,
        "evidence_direction_per_claim": {
            "SD-017": ed_sd017,
            "MECH-120": ed_mech120,
        },
        "overall_pass": overall_pass,
        "non_degenerate": non_degenerate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = DRY_RUN_SEEDS
        t0_eps, t1_eps = DRY_T0, DRY_T1
        n_sleep = DRY_SLEEP_CYCLES
        steps = DRY_STEPS
        meas_eps, meas_steps = DRY_MEASURE_EPISODES, DRY_MEASURE_STEPS
        min_seeds = 1
    else:
        seeds = SEEDS
        t0_eps, t1_eps = T0_CONVERGENCE_EPISODES, T1_EPISODES
        n_sleep = N_SLEEP_CYCLES
        steps = STEPS_PER_EPISODE
        meas_eps, meas_steps = MEASURE_EPISODES, MEASURE_STEPS
        min_seeds = 2  # of 3

    full_config = {
        "env_kwargs": ENV_KWARGS,
        "t0_convergence_episodes": t0_eps,
        "t1_episodes": t1_eps,
        # episodes_per_run == TOTAL (t0-convergence + t1 phases run sequentially).
        "episodes_per_run": int(t0_eps + t1_eps),
        "n_sleep_cycles": n_sleep,
        "steps_per_episode": steps,
        "measure_episodes": meas_eps,
        "measure_steps": meas_steps,
        "sd056_weight": SD056_WEIGHT,
        "conversion_config": {
            "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        },
        "sleep_config": {
            "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
            "sws_schema_weight": SWS_SCHEMA_WEIGHT,
            "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        },
    }

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell_cfg = dict(full_config)
            cell_cfg["arm"] = {
                k: arm[k]
                for k in (
                    "arm_id", "sws_enabled", "rem_enabled",
                    "replay_diversity_enabled", "random_replay_fraction",
                )
            }
            with arm_cell(
                seed,
                config_slice=cell_cfg,
                script_path=Path(__file__),
                extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
            ) as cell:
                row = _run_seed_arm(
                    arm, seed, t0_eps, t1_eps, n_sleep, steps, meas_eps, meas_steps,
                )
                cell.stamp(row)
            arm_results.append(row)

    summary = _evaluate(arm_results, min_seeds)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    evidence_direction = summary["evidence_direction"]

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": summary["evidence_direction_per_claim"],
        "non_degenerate": summary.get("non_degenerate", True),
        "sleep_driver_pattern": (
            "manual-cycle-loop (run_sleep_cycle() called once per cycle in a "
            "dedicated N_CYCLES wake-sleep-test loop)"
        ),
        "evidence_direction_note": (
            "Q-055 Rung-3 persistence test: does SD-017 SWS-phase sleep consolidation "
            "PRESERVE or ERODE the committed-action-class diversity (TrajDiv / C_R1B = "
            "selected_action_class_entropy) ARC-065 achieves during waking, with the "
            "569i TOP-K shortlist conversion armed so diversity reaches committed "
            "action? 3 time points (t0 post-convergence / t1 +200 ep no forced "
            "exploration / t2 +5 SD-017 sleep cycles) x 3 arms (A sleep-ON / B "
            "sleep-OFF control / C replay-diversity-ablated) x >=3 seeds. PASS (arm A, "
            "the governance-load-bearing arm): t2 entropy >= 50% of t0 AND not "
            "collapsed (>=2 classes, entropy > 0.1) WITH sleep active -> PRESERVES -> "
            "supports SD-017 (+INV-049) and the MECH-120-as-protective-risk-mitigated "
            "reading. ERODES (t2 < 50% of t0 or collapsed) -> weakens "
            "SD-017-preserves-diversity AND supports the MECH-120 Hebbian-erosion risk "
            "(ARC-065 must also operate during consolidation -> new MECH). INCREASES "
            "(t2 > 1.1x t0) -> surprising broader episodic-replay sampling; supports "
            "SD-017. Non-vacuity self-route gates (substrate_not_ready_requeue, "
            "non_contributory, NEVER weaken): R1 arm-A consolidation write-passes "
            "(sws_n_writes + rem_n_rollouts) > 0 on >=2/3 seeds (sleep cycles fire); "
            "R2 arm-A t0 entropy > 0.2 on >=2/3 (there IS diversity to erode); R3 arm-A "
            "cumulative write-passes > arm-B (the sleep knob is not inert). "
            "claim_ids=[SD-017, MECH-120]; evidence_direction_per_claim MANDATORY."
        ),
        "interpretation": {
            "label": summary["label"],
            "readiness": summary["readiness"],
            "persistence": summary["persistence"],
            "entropy_per_arm": summary["entropy_per_arm"],
            "routing": {
                "sleep_preserves_diversity": "PASS -> SD-017 preserves trajectory-class diversity; supports SD-017 + INV-049; MECH-120 protective risk mitigated",
                "sleep_increases_diversity": "PASS -> sleep broadens episodic-replay sampling beyond the waking policy; supports SD-017 (surprising)",
                "sleep_erodes_diversity": "FAIL -> Hebbian winner-take-all erodes diversity; weakens SD-017-preserves-diversity, supports MECH-120 erosion risk; route to /implement-substrate (ARC-065 during consolidation)",
                "substrate_not_ready_requeue": "non-vacuity gate unmet (sleep did not fire / no t0 diversity / sleep knob inert); re-queue, do NOT weaken either claim",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "min_seeds_for_pass": min_seeds,
            **full_config,
            "arms": [
                {
                    k: a[k]
                    for k in (
                        "arm_id", "label", "sws_enabled", "rem_enabled",
                        "replay_diversity_enabled", "random_replay_fraction",
                    )
                }
                for a in ARMS
            ],
            "thresholds": {
                "persist_fraction": PERSIST_FRACTION,
                "increase_fraction": INCREASE_FRACTION,
                "collapse_floor": COLLAPSE_FLOOR,
                "t0_floor": T0_FLOOR,
                "measure_temperature": MEASURE_TEMPERATURE,
            },
        },
        "acceptance_criteria": {
            "R1_sleep_cycles_fire": summary["readiness"]["r1_sleep_cycles_fire"],
            "R2_t0_diversity_supra_floor": summary["readiness"]["r2_t0_diversity_supra_floor"],
            "R3_sleep_knob_not_inert": summary["readiness"]["r3_sleep_knob_not_inert"],
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "persistence_pass": summary["persistence"]["persistence_pass"],
            "reading": summary["persistence"]["reading"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(
        f"Outcome: {outcome} (label={summary['label']}, "
        f"evidence_direction={evidence_direction})",
        flush=True,
    )
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-691 Q-055 sleep-consolidation diversity-persistence (Rung-3)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
