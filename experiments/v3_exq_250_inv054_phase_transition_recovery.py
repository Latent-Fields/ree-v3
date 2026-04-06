#!/opt/local/bin/python3
"""
V3-EXQ-250 -- INV-054: Phase-Transition Recovery Test

Claims: INV-054
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

INV-054 (depressive maintenance loop) predicts that the depressive state is
self-maintaining: z_goal absent -> no terrain exploration -> terrain stays
collapsed. Recovery from this attractor requires accumulated terrain re-exposure
to cross a threshold (phase-transition), NOT immediate improvement upon
environment change.

This unique prediction distinguishes INV-054 from simple reward-signal
insufficiency accounts: under an insufficiency account, improving the reward
environment (LOW_HARM) should produce rapid, graded recovery. Under INV-054,
recovery requires the agent to re-explore benefit terrain long enough to
re-seed z_goal, which only occurs once accumulated exposure exceeds a threshold.
The expected signature is a latency of >50 Phase-2 episodes before z_goal
crosses the 0.3 recovery threshold.

=== DESIGN ===

Two-phase within-session experiment:

Phase 1 -- Establish depression attractor:
  - LONG_HORIZON environment (8x8, 1 resource, 3 hazards, 150 steps/ep, hazard_harm=0.02)
  - Warmup: 200 episodes (standard mixed-policy)
  - Then run until: z_goal_norm < 0.1 for 50 consecutive episodes (attractor confirmed),
    OR max 500 total episodes (whichever comes first)
  - If attractor not established: FAIL with note "attractor not established"
  - Record: phase1_attractor_episode (first ep where the 50-ep window closes)

Phase 2 -- Recovery test:
  - Switch to LOW_HARM environment (8x8, 3 resources, 1 hazard, 150 steps/ep, hazard_harm=0.02)
  - Agent state preserved (no reset) -- maintains depression attractor state
  - Run 300 additional training episodes
  - Record per-episode: z_goal_norm, resource_rate (rolling 20-ep), terrain_coverage
  - Record terrain_coverage_by_ep every 50 episodes: [ep50, ep100, ..., ep300]

=== TERRAIN COVERAGE METRIC ===

Operationalises the INV-054 re-exposure mechanism:
  terrain_coverage = unique (row, col) positions visited in Phase 2 / 64 (8x8 grid)
This tests whether z_goal recovery co-occurs with expanded terrain exploration,
as predicted by the INV-054 maintenance loop (re-exposure breaks the attractor).

=== CLINICAL ANCHOR (STAR*D) ===

STAR*D median response: ~5.4 weeks (~38 treatment-days).
50 Phase-2 episodes is a conservative threshold to ensure signal is not missed.
Graded recovery (latency <= 10) would be inconsistent with attractor dynamics.

=== PRE-REGISTERED CRITERIA ===

latency_to_recovery = first Phase-2 episode where z_goal_norm crosses 0.3
                      (or None if z_goal_norm never reaches 0.3 in 300 eps)

PASS: latency_to_recovery > 50 in >= 2/3 seeds
  -> supports INV-054 (delayed phase-transition recovery confirmed)
FAIL: latency_to_recovery <= 10 in >= 2/3 seeds
  -> does_not_support INV-054 (graded/immediate recovery)
INCONCLUSIVE: any other pattern (10 < latency <= 50, or never recovers)
  -> inconclusive (update evidence_direction to inconclusive)
"""

import sys
import random
import json
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_250_inv054_phase_transition_recovery"
CLAIM_IDS          = ["INV-054"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
ATTRACTOR_THRESHOLD      = 0.1   # z_goal_norm below this = depression attractor
ATTRACTOR_WINDOW         = 50    # consecutive episodes below threshold
RECOVERY_THRESHOLD       = 0.3   # z_goal_norm above this = recovered
LATENCY_PASS_MIN         = 50    # PASS: recovery latency > this
LATENCY_FAIL_MAX         = 10    # FAIL: recovery latency <= this (graded)
MAJORITY_THRESH          = 2     # criteria met in >= MAJORITY_THRESH/3 seeds

# ---------------------------------------------------------------------------
# Grid and episode parameters
# ---------------------------------------------------------------------------
GRID_SIZE              = 8
GRID_CELLS             = GRID_SIZE * GRID_SIZE   # 64

LONG_N_RESOURCES       = 1
LONG_N_HAZARDS         = 3
LONG_STEPS             = 150

LOW_HARM_N_RESOURCES   = 3
LOW_HARM_N_HAZARDS     = 1
LOW_HARM_STEPS         = 150

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
WARMUP_EPISODES      = 200
MAX_PHASE1_EPISODES  = 500    # hard cap including warmup
PHASE2_EPISODES      = 300
SEEDS                = [42, 7, 13]
GREEDY_FRAC          = 0.4
MAX_BUF              = 4000
WF_BUF_MAX           = 2000
WORLD_DIM            = 32
BATCH_SIZE           = 16
RESOURCE_RATE_WINDOW = 20     # rolling window for resource_rate
TERRAIN_SAMPLE_STEP  = 50     # sample terrain_coverage every N Phase-2 episodes

# Learning rates
LR_E1      = 1e-3
LR_E2_WF   = 1e-3
LR_HARM    = 1e-4
LR_BENEFIT = 1e-3

# SD-018: resource proximity supervision
LAMBDA_RESOURCE = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    """Greedy action: move toward nearest resource (Manhattan)."""
    ax, ay = env.agent_x, env.agent_y
    if not env.resources:
        return random.randint(0, env.action_dim - 1)
    best_d = float("inf")
    nearest = None
    for r in env.resources:
        rx, ry = int(r[0]), int(r[1])
        d = abs(ax - rx) + abs(ay - ry)
        if d < best_d:
            best_d = d
            nearest = (rx, ry)
    if nearest is None or best_d == 0:
        return random.randint(0, env.action_dim - 1)
    rx, ry = nearest
    dx, dy = rx - ax, ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    return 3 if dy > 0 else 2


def _dist_to_nearest_resource(env) -> int:
    if not env.resources:
        return 999
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


def _get_benefit_exposure(obs_body: torch.Tensor) -> float:
    """Extract benefit_exposure from body_state obs (index 11 in proxy mode)."""
    flat = obs_body.flatten()
    if flat.shape[0] > 11:
        return float(flat[11].item())
    return 0.0


def _get_energy(obs_body: torch.Tensor) -> float:
    """Extract energy from body_state obs (index 3)."""
    flat = obs_body.flatten()
    if flat.shape[0] > 3:
        return float(flat[3].item())
    return 1.0


def _update_z_goal(agent: REEAgent, obs_body: torch.Tensor) -> None:
    """Update z_goal from current step's benefit_exposure and drive_level."""
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


def _get_goal_norm(agent: REEAgent) -> float:
    """Read z_goal_norm from agent diagnostics."""
    diag = agent.compute_goal_maintenance_diagnostic()
    return float(diag["goal_norm"])


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _make_long_horizon_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=LONG_N_RESOURCES,
        num_hazards=LONG_N_HAZARDS,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=0.02,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )


def _make_low_harm_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=LOW_HARM_N_RESOURCES,
        num_hazards=LOW_HARM_N_HAZARDS,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=0.02,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    """PLANNED mode only (z_goal_enabled=True, drive_weight=2)."""
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=16,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=True,
        benefit_weight=0.5,
        z_goal_enabled=True,
        e1_goal_conditioned=True,
        goal_weight=1.0,
        drive_weight=2.0,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def _make_optimisers(agent: REEAgent):
    """Create standard set of optimisers for training."""
    e1_params    = list(agent.e1.parameters())
    e2_wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    e1_opt      = optim.Adam(e1_params, lr=LR_E1)
    e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
    harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_HARM)
    benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)
    return e1_opt, e2_wf_opt, harm_opt, benefit_opt, e1_params, e2_wf_params


# ---------------------------------------------------------------------------
# Training step helpers
# ---------------------------------------------------------------------------

def _train_step(
    agent: REEAgent,
    device,
    n_act: int,
    env,
    obs_dict: dict,
    z_world_prev: Optional[torch.Tensor],
    action_prev: Optional[torch.Tensor],
    wf_buf: list,
    harm_pos_buf: list,
    harm_neg_buf: list,
    ben_zw_buf: list,
    ben_lbl_buf: list,
    e1_opt, e2_wf_opt, harm_opt, benefit_opt,
    e1_params, e2_wf_params,
    step_i: int,
    warmup_mode: bool,
) -> Tuple[dict, float, bool, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single training step. Returns:
      (obs_dict_next, harm_signal, done, info, z_world_curr, action_oh, obs_body_post)
    """
    obs_body  = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    obs_harm  = obs_dict.get("harm_obs", None)

    # Sense
    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
    ticks  = agent.clock.advance()

    if ticks.get("e1_tick", False):
        _ = agent._e1_tick(latent)

    z_world_curr = latent.z_world.detach()

    # E2 world_forward buffer
    if z_world_prev is not None and action_prev is not None:
        # Detach action_prev: in non-warmup mode, select_action() returns a tensor
        # that may carry a live grad_fn through e2.world_transition (from trajectory
        # rollout). Storing it undetached would cause wf_loss.backward() to traverse
        # the already-freed old trajectory graph on the next step.
        wf_buf.append((z_world_prev, action_prev.detach(), z_world_curr))
        if len(wf_buf) > WF_BUF_MAX:
            wf_buf[:] = wf_buf[-WF_BUF_MAX:]

    # Action selection
    if warmup_mode:
        if random.random() < GREEDY_FRAC:
            action_idx = _greedy_toward_resource(env)
        else:
            action_idx = random.randint(0, n_act - 1)
        action_oh = _onehot(action_idx, n_act, device)
        agent._last_action = action_oh
    else:
        e1_prior = torch.zeros(1, WORLD_DIM, device=device)
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action_oh  = agent.select_action(candidates, ticks, temperature=1.0)
        if action_oh is None:
            action_oh = _onehot(random.randint(0, n_act - 1), n_act, device)
            agent._last_action = action_oh

    # Benefit proximity label (before step)
    dist    = _dist_to_nearest_resource(env)
    is_near = 1.0 if dist <= 2 else 0.0

    # Env step
    _, harm_signal, done, info, obs_dict_next = env.step(action_oh)

    # z_goal update (PLANNED always)
    _update_z_goal(agent, obs_dict_next["body_state"])

    # Train E1
    if len(agent._world_experience_buffer) >= 2:
        e1_loss = agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            e1_opt.zero_grad()
            e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
            e1_opt.step()

    # SD-018: resource proximity supervision
    rfv = obs_dict_next.get("resource_field_view", None)
    if rfv is not None:
        rp_target = max(rfv).item()
        rp_loss = agent.compute_resource_proximity_loss(
            rp_target, latent)
        if rp_loss.requires_grad:
            e1_opt.zero_grad()
            (LAMBDA_RESOURCE * rp_loss).backward()
            torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
            e1_opt.step()

    # Train E2 world_forward
    if len(wf_buf) >= BATCH_SIZE:
        idxs  = random.sample(range(len(wf_buf)), min(BATCH_SIZE, len(wf_buf)))
        zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
        a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
        zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
        wf_pred = agent.e2.world_forward(zw_b, a_b)
        wf_loss = F.mse_loss(wf_pred, zw1_b)
        if wf_loss.requires_grad:
            e2_wf_opt.zero_grad()
            wf_loss.backward()
            torch.nn.utils.clip_grad_norm_(e2_wf_params, 1.0)
            e2_wf_opt.step()
        with torch.no_grad():
            agent.e3.update_running_variance(
                (wf_pred.detach() - zw1_b).detach()
            )

    # Train E3 harm_eval (stratified)
    if float(harm_signal) < 0:
        harm_pos_buf.append(z_world_curr)
        if len(harm_pos_buf) > MAX_BUF:
            harm_pos_buf[:] = harm_pos_buf[-MAX_BUF:]
    else:
        harm_neg_buf.append(z_world_curr)
        if len(harm_neg_buf) > MAX_BUF:
            harm_neg_buf[:] = harm_neg_buf[-MAX_BUF:]

    if len(harm_pos_buf) >= 4 and len(harm_neg_buf) >= 4:
        k_p = min(BATCH_SIZE // 2, len(harm_pos_buf))
        k_n = min(BATCH_SIZE // 2, len(harm_neg_buf))
        pi  = torch.randperm(len(harm_pos_buf))[:k_p].tolist()
        ni  = torch.randperm(len(harm_neg_buf))[:k_n].tolist()
        zw_b = torch.cat(
            [harm_pos_buf[i] for i in pi] + [harm_neg_buf[i] for i in ni],
            dim=0,
        )
        tgt = torch.cat([
            torch.ones(k_p,  1, device=device),
            torch.zeros(k_n, 1, device=device),
        ], dim=0)
        pred = agent.e3.harm_eval(zw_b)
        hloss = F.binary_cross_entropy(pred, tgt)
        if hloss.requires_grad:
            harm_opt.zero_grad()
            hloss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.e3.harm_eval_head.parameters(), 0.5
            )
            harm_opt.step()

    # Train E3 benefit_eval
    ben_zw_buf.append(z_world_curr)
    ben_lbl_buf.append(is_near)
    if len(ben_zw_buf) > MAX_BUF:
        ben_zw_buf[:]  = ben_zw_buf[-MAX_BUF:]
        ben_lbl_buf[:] = ben_lbl_buf[-MAX_BUF:]

    if len(ben_zw_buf) >= 32 and step_i % 4 == 0:
        k    = min(32, len(ben_zw_buf))
        idxs = random.sample(range(len(ben_zw_buf)), k)
        zw_b = torch.cat([ben_zw_buf[i] for i in idxs], dim=0)
        lbl  = torch.tensor(
            [ben_lbl_buf[i] for i in idxs],
            dtype=torch.float32,
        ).unsqueeze(1).to(device)
        pred_b = agent.e3.benefit_eval(zw_b)
        bloss  = F.binary_cross_entropy(pred_b, lbl)
        if bloss.requires_grad:
            benefit_opt.zero_grad()
            bloss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.e3.benefit_eval_head.parameters(), 0.5
            )
            benefit_opt.step()
            agent.e3.record_benefit_sample(k)

    return obs_dict_next, float(harm_signal), done, info, z_world_curr, action_oh, obs_dict_next["body_state"]


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    warmup_episodes: int,
    max_phase1_episodes: int,
    phase2_episodes: int,
    dry_run: bool = False,
) -> Dict:
    """
    Run Phase 1 (attractor establishment) and Phase 2 (recovery test) for one seed.
    """
    random.seed(seed)
    print(f"\n[V3-EXQ-250] Seed {seed}", flush=True)

    # Build Phase 1 environment and agent
    env_p1  = _make_long_horizon_env(seed)
    agent   = _make_agent(env_p1, seed)
    device  = agent.device
    n_act   = env_p1.action_dim

    # Build optimisers
    e1_opt, e2_wf_opt, harm_opt, benefit_opt, e1_params, e2_wf_params = _make_optimisers(agent)

    # Shared experience buffers (persist across phases)
    wf_buf:       List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_pos_buf: List[torch.Tensor] = []
    harm_neg_buf: List[torch.Tensor] = []
    ben_zw_buf:   List[torch.Tensor] = []
    ben_lbl_buf:  List[float]        = []

    # -----------------------------------------------------------------------
    # Phase 1: warmup + attractor establishment
    # -----------------------------------------------------------------------
    print(f"[Phase1] seed={seed} starting warmup ({warmup_episodes} eps)", flush=True)
    agent.train()

    # Rolling window to detect attractor (z_goal_norm < ATTRACTOR_THRESHOLD for ATTRACTOR_WINDOW eps)
    goal_norm_window: deque = deque(maxlen=ATTRACTOR_WINDOW)
    phase1_attractor_episode: Optional[int] = None
    phase1_goal_norm_traj: List[float] = []   # record every 50 eps

    total_phase1_eps = max_phase1_episodes

    for ep in range(total_phase1_eps):
        is_warmup = ep < warmup_episodes
        _, obs_dict = env_p1.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for step_i in range(LONG_STEPS):
            obs_dict, harm_signal, done, info, z_world_curr, action_oh, obs_body_post = _train_step(
                agent, device, n_act, env_p1, obs_dict,
                z_world_prev, action_prev,
                wf_buf, harm_pos_buf, harm_neg_buf, ben_zw_buf, ben_lbl_buf,
                e1_opt, e2_wf_opt, harm_opt, benefit_opt, e1_params, e2_wf_params,
                step_i, warmup_mode=is_warmup,
            )
            z_world_prev = z_world_curr
            action_prev  = action_oh
            if done:
                break

        goal_norm = _get_goal_norm(agent)
        goal_norm_window.append(goal_norm)

        # Record trajectory every 50 eps
        if (ep + 1) % 50 == 0:
            phase1_goal_norm_traj.append(round(goal_norm, 4))
            print(
                f"[Phase1] seed={seed} ep {ep+1}/{total_phase1_eps}"
                f" z_goal_norm={goal_norm:.3f}"
                f" window_min={min(goal_norm_window):.3f}"
                f" window_len={len(goal_norm_window)}",
                flush=True,
            )

        # Check attractor condition (only after warmup)
        if ep >= warmup_episodes and phase1_attractor_episode is None:
            if len(goal_norm_window) == ATTRACTOR_WINDOW and all(
                v < ATTRACTOR_THRESHOLD for v in goal_norm_window
            ):
                phase1_attractor_episode = ep + 1  # 1-indexed
                print(
                    f"[Phase1] seed={seed} ATTRACTOR ESTABLISHED at ep {phase1_attractor_episode}"
                    f" (z_goal_norm={goal_norm:.3f})",
                    flush=True,
                )
                break

    # Check if attractor was established
    attractor_established = phase1_attractor_episode is not None
    if not attractor_established:
        print(
            f"[Phase1] seed={seed} attractor NOT established in {total_phase1_eps} eps"
            f" (final z_goal_norm={_get_goal_norm(agent):.3f})",
            flush=True,
        )
        return {
            "seed": seed,
            "attractor_established": False,
            "phase1_attractor_episode": None,
            "latency_to_recovery": None,
            "terrain_coverage_by_ep": [],
            "z_goal_norm_trajectory": phase1_goal_norm_traj,
            "phase2_resource_rate_traj": [],
            "phase2_goal_norm_traj": [],
            "note": "attractor not established",
        }

    # -----------------------------------------------------------------------
    # Phase 2: switch to LOW_HARM, measure recovery
    # -----------------------------------------------------------------------
    print(
        f"[Phase1->Phase2] seed={seed} switching to LOW_HARM env"
        f" (agent state preserved, no reset)",
        flush=True,
    )

    env_p2 = _make_low_harm_env(seed + 1000)   # different seed for env layout variety
    # Note: agent state is preserved (no agent.reset())

    latency_to_recovery: Optional[int] = None
    visited_cells: set = set()
    phase2_resource_counts: deque = deque(maxlen=RESOURCE_RATE_WINDOW)
    phase2_goal_norm_traj: List[float] = []
    phase2_resource_rate_traj: List[float] = []
    terrain_coverage_by_ep: List[float] = []  # sampled every TERRAIN_SAMPLE_STEP episodes

    for ep in range(phase2_episodes):
        _, obs_dict = env_p2.reset()
        # Note: agent is NOT reset -- maintains z_goal state from Phase 1 attractor

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_resources = 0

        for step_i in range(LOW_HARM_STEPS):
            # Track terrain coverage: record agent position before each step
            pos = (int(env_p2.agent_x), int(env_p2.agent_y))
            visited_cells.add(pos)

            obs_dict, harm_signal, done, info, z_world_curr, action_oh, obs_body_post = _train_step(
                agent, device, n_act, env_p2, obs_dict,
                z_world_prev, action_prev,
                wf_buf, harm_pos_buf, harm_neg_buf, ben_zw_buf, ben_lbl_buf,
                e1_opt, e2_wf_opt, harm_opt, benefit_opt, e1_params, e2_wf_params,
                step_i, warmup_mode=False,
            )
            z_world_prev = z_world_curr
            action_prev  = action_oh

            if info.get("transition_type") == "resource":
                ep_resources += 1
            if done:
                break

        # After episode: record metrics
        goal_norm = _get_goal_norm(agent)
        phase2_resource_counts.append(1 if ep_resources >= 1 else 0)
        rr = sum(phase2_resource_counts) / max(1, len(phase2_resource_counts))
        coverage = len(visited_cells) / GRID_CELLS

        # Check recovery latency
        if latency_to_recovery is None and goal_norm >= RECOVERY_THRESHOLD:
            latency_to_recovery = ep + 1  # 1-indexed Phase-2 episode
            print(
                f"[Phase2] seed={seed} RECOVERY at Phase-2 ep {latency_to_recovery}"
                f" z_goal_norm={goal_norm:.3f}",
                flush=True,
            )

        # Terrain coverage checkpoint every TERRAIN_SAMPLE_STEP episodes
        if (ep + 1) % TERRAIN_SAMPLE_STEP == 0:
            terrain_coverage_by_ep.append(round(coverage, 4))
            phase2_goal_norm_traj.append(round(goal_norm, 4))
            phase2_resource_rate_traj.append(round(rr, 4))
            print(
                f"[Phase2] seed={seed} ep {ep+1}/{phase2_episodes}"
                f" z_goal_norm={goal_norm:.3f}"
                f" resource_rate={rr:.3f}"
                f" coverage={coverage:.2f}",
                flush=True,
            )

    # Determine per-seed verdict
    if latency_to_recovery is None:
        seed_verdict = "INCONCLUSIVE"
    elif latency_to_recovery > LATENCY_PASS_MIN:
        seed_verdict = "PASS"
    elif latency_to_recovery <= LATENCY_FAIL_MAX:
        seed_verdict = "FAIL"
    else:
        seed_verdict = "INCONCLUSIVE"

    print(f"verdict: {seed_verdict} (seed={seed} latency={latency_to_recovery})", flush=True)

    return {
        "seed": seed,
        "attractor_established": True,
        "phase1_attractor_episode": phase1_attractor_episode,
        "latency_to_recovery": latency_to_recovery,
        "terrain_coverage_by_ep": terrain_coverage_by_ep,
        "z_goal_norm_trajectory": phase2_goal_norm_traj,  # Phase-2 traj (every 50 eps)
        "phase1_goal_norm_trajectory": phase1_goal_norm_traj,
        "phase2_resource_rate_traj": phase2_resource_rate_traj,
        "seed_verdict": seed_verdict,
    }


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(per_seed_results: List[Dict]) -> Dict:
    """Aggregate per-seed results into experiment-level verdict."""
    n_seeds = len(per_seed_results)

    # Count seeds where attractor was established
    established = [r for r in per_seed_results if r.get("attractor_established", False)]
    if len(established) < MAJORITY_THRESH:
        return {
            "per_seed": per_seed_results,
            "n_seeds": n_seeds,
            "n_established": len(established),
            "pass_count": 0,
            "fail_count": 0,
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "decision": "attractor_not_established",
            "note": f"Attractor established in only {len(established)}/{n_seeds} seeds",
            "recovery_latency_mean": None,
            "recovery_latency_std": None,
            "terrain_coverage_at_ep50_mean": None,
            "terrain_coverage_at_ep300_mean": None,
            "latencies": [r.get("latency_to_recovery") for r in per_seed_results],
        }

    # Latencies (None = never recovered)
    latencies = [r.get("latency_to_recovery") for r in established]

    # PASS: latency > 50 in >= 2/3 seeds
    pass_count = sum(
        1 for lat in latencies
        if lat is not None and lat > LATENCY_PASS_MIN
    )
    # FAIL: latency <= 10 in >= 2/3 seeds
    fail_count = sum(
        1 for lat in latencies
        if lat is not None and lat <= LATENCY_FAIL_MAX
    )

    # Compute latency stats (treat None as 301 for mean, indicating no recovery)
    lat_vals = [lat if lat is not None else PHASE2_EPISODES + 1 for lat in latencies]
    lat_mean = sum(lat_vals) / max(1, len(lat_vals))
    lat_std  = (sum((v - lat_mean) ** 2 for v in lat_vals) / max(1, len(lat_vals))) ** 0.5

    # Terrain coverage means (ep50, ep300 = first and last checkpoint)
    cov_at_ep50  = []
    cov_at_ep300 = []
    for r in established:
        cby = r.get("terrain_coverage_by_ep", [])
        if len(cby) >= 1:
            cov_at_ep50.append(cby[0])
        if len(cby) >= 6:
            cov_at_ep300.append(cby[5])
        elif len(cby) >= 1:
            cov_at_ep300.append(cby[-1])

    cov_ep50_mean  = sum(cov_at_ep50)  / max(1, len(cov_at_ep50))
    cov_ep300_mean = sum(cov_at_ep300) / max(1, len(cov_at_ep300))

    # Outcome
    if pass_count >= MAJORITY_THRESH:
        outcome   = "PASS"
        direction = "supports"
        decision  = "retain_ree"
    elif fail_count >= MAJORITY_THRESH:
        outcome   = "FAIL"
        direction = "does_not_support"
        decision  = "hybridize"
    else:
        outcome   = "INCONCLUSIVE"
        direction = "inconclusive"
        decision  = "hold_inconclusive"

    return {
        "per_seed": per_seed_results,
        "n_seeds": n_seeds,
        "n_established": len(established),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "outcome": outcome,
        "evidence_direction": direction,
        "decision": decision,
        "recovery_latency_mean": round(lat_mean, 2),
        "recovery_latency_std":  round(lat_std, 2),
        "terrain_coverage_at_ep50_mean":  round(cov_ep50_mean, 4),
        "terrain_coverage_at_ep300_mean": round(cov_ep300_mean, 4),
        "latencies": latencies,
        "note": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Dry-run overrides: 1 seed, Phase 1 max 10 eps (5 warmup), Phase 2 10 eps
    if args.dry_run:
        warmup    = 5
        max_p1    = 10
        phase2    = 10
        seeds     = [42]
    else:
        warmup    = WARMUP_EPISODES
        max_p1    = MAX_PHASE1_EPISODES
        phase2    = PHASE2_EPISODES
        seeds     = SEEDS

    print(
        f"[V3-EXQ-250] INV-054 Phase-Transition Recovery Test"
        f"  dry_run={args.dry_run}"
        f"  warmup={warmup} max_phase1={max_p1} phase2={phase2} seeds={seeds}",
        flush=True,
    )
    print(
        f"  Phase1 (LONG_HORIZON): {LONG_N_RESOURCES} resource,"
        f" {LONG_N_HAZARDS} hazards, {LONG_STEPS} steps/ep",
        flush=True,
    )
    print(
        f"  Phase2 (LOW_HARM):     {LOW_HARM_N_RESOURCES} resources,"
        f" {LOW_HARM_N_HAZARDS} hazard, {LOW_HARM_STEPS} steps/ep",
        flush=True,
    )
    print(
        f"  Attractor threshold: z_goal_norm<{ATTRACTOR_THRESHOLD} for {ATTRACTOR_WINDOW} consecutive eps",
        flush=True,
    )
    print(
        f"  Recovery threshold:  z_goal_norm>={RECOVERY_THRESHOLD}",
        flush=True,
    )
    print(
        f"  PASS criterion: latency > {LATENCY_PASS_MIN} in >= {MAJORITY_THRESH}/3 seeds (STAR*D anchor)",
        flush=True,
    )

    per_seed_results = []
    for seed in seeds:
        result = _run_seed(
            seed=seed,
            warmup_episodes=warmup,
            max_phase1_episodes=max_p1,
            phase2_episodes=phase2,
            dry_run=args.dry_run,
        )
        per_seed_results.append(result)

    agg = _aggregate(per_seed_results)

    print(f"\n[V3-EXQ-250] === Results ===", flush=True)
    print(
        f"  Attractor established: {agg['n_established']}/{agg['n_seeds']} seeds",
        flush=True,
    )
    if "latencies" in agg:
        print(f"  Recovery latencies (Phase-2 ep): {agg['latencies']}", flush=True)
        lat_mean_str = f"{agg['recovery_latency_mean']:.2f}" if agg['recovery_latency_mean'] is not None else "N/A"
        lat_std_str  = f"{agg['recovery_latency_std']:.2f}"  if agg['recovery_latency_std']  is not None else "N/A"
        print(
            f"  Latency mean={lat_mean_str}  std={lat_std_str}",
            flush=True,
        )
        cov50  = agg.get('terrain_coverage_at_ep50_mean')
        cov300 = agg.get('terrain_coverage_at_ep300_mean')
        cov50_str  = f"{cov50:.3f}"  if cov50  is not None else "N/A"
        cov300_str = f"{cov300:.3f}" if cov300 is not None else "N/A"
        print(
            f"  Terrain coverage @ ep50 (mean):  {cov50_str}",
            flush=True,
        )
        print(
            f"  Terrain coverage @ ep300 (mean): {cov300_str}",
            flush=True,
        )
    print(
        f"  PASS count (latency>{LATENCY_PASS_MIN}): {agg.get('pass_count', 'N/A')}/{agg['n_seeds']}",
        flush=True,
    )
    print(
        f"  FAIL count (latency<={LATENCY_FAIL_MAX}): {agg.get('fail_count', 'N/A')}/{agg['n_seeds']}",
        flush=True,
    )
    print(
        f"  -> {agg['outcome']} decision={agg['decision']} direction={agg['evidence_direction']}",
        flush=True,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    # Write output
    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    manifest = {
        "run_id":             f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":    EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            agg["outcome"],
        "evidence_direction": agg["evidence_direction"],
        "decision":           agg["decision"],
        "timestamp":          ts,
        "seeds":              seeds,
        # Parameters
        "warmup_episodes":          warmup,
        "max_phase1_episodes":      max_p1,
        "phase2_episodes":          phase2,
        "long_n_resources":         LONG_N_RESOURCES,
        "long_n_hazards":           LONG_N_HAZARDS,
        "long_steps":               LONG_STEPS,
        "low_harm_n_resources":     LOW_HARM_N_RESOURCES,
        "low_harm_n_hazards":       LOW_HARM_N_HAZARDS,
        "low_harm_steps":           LOW_HARM_STEPS,
        "hazard_harm":              0.02,
        # Thresholds
        "attractor_threshold":      ATTRACTOR_THRESHOLD,
        "attractor_window":         ATTRACTOR_WINDOW,
        "recovery_threshold":       RECOVERY_THRESHOLD,
        "latency_pass_min":         LATENCY_PASS_MIN,
        "latency_fail_max":         LATENCY_FAIL_MAX,
        # Aggregate metrics
        "recovery_latency_mean":              agg.get("recovery_latency_mean"),
        "recovery_latency_std":               agg.get("recovery_latency_std"),
        "terrain_coverage_at_ep50_mean":      agg.get("terrain_coverage_at_ep50_mean"),
        "terrain_coverage_at_ep300_mean":     agg.get("terrain_coverage_at_ep300_mean"),
        "n_established":                      agg.get("n_established"),
        "pass_count":                         agg.get("pass_count"),
        "fail_count":                         agg.get("fail_count"),
        "latencies":                          agg.get("latencies"),
        "note":                               agg.get("note", ""),
        # Per-seed detail
        "per_seed_results": [
            {
                "seed":                     r["seed"],
                "attractor_established":    r.get("attractor_established", False),
                "phase1_attractor_episode": r.get("phase1_attractor_episode"),
                "latency_to_recovery":      r.get("latency_to_recovery"),
                "terrain_coverage_by_ep":   r.get("terrain_coverage_by_ep", []),
                "z_goal_norm_trajectory":   r.get("z_goal_norm_trajectory", []),
                "phase2_resource_rate_traj":r.get("phase2_resource_rate_traj", []),
                "seed_verdict":             r.get("seed_verdict", "N/A"),
            }
            for r in per_seed_results
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-250] Written: {out_path}", flush=True)
    print(f"Status: {agg['outcome']}", flush=True)


if __name__ == "__main__":
    main()
