#!/opt/local/bin/python3
"""
V3-EXQ-432 -- SD-014: Replay Gate Prioritization (Surprise-Weighted vs Wanting-Weighted)

Claims: SD-014
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

Does SURPRISE_WEIGHTED replay (d=[0,0,0,1]) select categorically different z_world nodes
than WANTING_WEIGHTED replay (d=[1,0,0,0])? And does the selected start reflect the
correct valence component (surprise condition selects high-surprise nodes, wanting condition
selects high-wanting nodes)?

Directly tests the Carey, Tanaka & van der Meer (2019, Nat Neurosci) anti-preference
replay finding: after motivational revaluation, hippocampal SWR replay preferentially
targets the now-devalued (stale) nodes rather than the currently-valued nodes.
In REE terms: VALENCE_SURPRISE (staleness signal) should gate replay start selection
to the anti-preference (recently violated expectation) arm.

=== DESIGN ===

4-phase, 3 conditions, 3 seeds.

Phase 1 (D1 training, N_D1 episodes):
  Environment: 2 resources, 1 hazard.
  Conditions are identical in this phase (all train the same way).
  VALENCE_WANTING accrues at resource-adjacent z_world nodes (via tonic_5ht).
  VALENCE_SURPRISE accrues at hazard-adjacent nodes (via MECH-205 PE spikes).

Phase 2 (D2 revaluation, N_D2 episodes):
  Environment: 2 resources, 2 hazards (1 NOVEL hazard, different grid location).
  Same training loop. Novel hazard nodes generate unexpected harm PE spikes ->
  VALENCE_SURPRISE rises at novel hazard nodes.
  Resource nodes retain VALENCE_WANTING from D1. Novel hazard nodes have low VALENCE_WANTING.

Phase 3 (offline consolidation, after each D2 episode):
  For each condition, compute which theta_buffer entry WOULD be selected by the
  condition's drive_state vector. Record the valence at that node.
  Then call replay() with the condition's drive_state (generates trajectories for E3).

Phase 4 (test, N_TEST episodes):
  Same D2 environment. Evaluate harm_exposure as secondary behavioral metric.

Conditions differ ONLY in their replay drive_state:
  A_SURPRISE: drive_state = [0, 0, 0, 1]  -- pure VALENCE_SURPRISE prioritization
  B_WANTING:  drive_state = [1, 0, 0, 0]  -- pure VALENCE_WANTING prioritization
  C_ABLATION: drive_state = None           -- most recent buffer entry (baseline)

All conditions have identical valence write infrastructure (tonic_5ht + surprise_gated_replay
both enabled) so the VALENCE map is shared -- only the replay start selection policy differs.

=== PRE-REGISTERED CRITERIA (evaluated per seed, majority >= 2/3) ===

  P1: mean_surprise_at_start[A_SURPRISE] > mean_surprise_at_start[B_WANTING] + 0.001
      (SURPRISE condition selects nodes with measurably higher VALENCE_SURPRISE)

  P2: mean_wanting_at_start[B_WANTING] > mean_wanting_at_start[A_SURPRISE] + 0.001
      (WANTING condition selects nodes with measurably higher VALENCE_WANTING)

  P3: n_surprise_writes_d2 > 5 (VALENCE_SURPRISE is actually populated in D2 phase --
      if not, P1/P2 are trivially zero and the experiment is non-contributory)

  PASS: P1 AND P2 AND P3

Secondary (not pass criterion):
  S1: mean_harm_d2[A_SURPRISE] < mean_harm_d2[B_WANTING] (behavioral adaptation speed)
      -- expected but may not reach significance in N_TEST=20 episodes.

=== EVB REFERENCE ===

EVB-0132 (registered 2026-04-17): SD-014 replay gate -- test surprise-weighted vs
wanting-weighted replay prioritization. Literature anchor: Carey et al. (2019).
"""

import sys
import random
import json
import time
from datetime import datetime
from pathlib import Path
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
EXPERIMENT_TYPE    = "v3_exq_432_sd014_replay_gate_prioritization"
CLAIM_IDS          = ["SD-014"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
MAJORITY_THRESH       = 2       # >= 2 of 3 seeds
SURPRISE_MARGIN       = 0.001   # P1: SURPRISE condition > WANTING condition + this
WANTING_MARGIN        = 0.001   # P2: WANTING condition > SURPRISE condition + this
MIN_SURPRISE_WRITES   = 5       # P3: min VALENCE_SURPRISE writes in D2 phase

# ---------------------------------------------------------------------------
# Episode and training parameters
# ---------------------------------------------------------------------------
GRID_SIZE       = 8
N_RESOURCES     = 2
N_HAZARDS_D1    = 1
N_HAZARDS_D2    = 2   # adds 1 novel hazard for revaluation
STEPS_PER_EP    = 120
N_D1            = 100   # Phase 1: D1 training episodes
N_D2            = 50    # Phase 2: D2 revaluation episodes
N_REPLAY_PER_EP = 5     # Phase 3: replay calls after each D2 episode
N_TEST          = 20    # Phase 4: test episodes
SEEDS           = [42, 7, 13]

WORLD_DIM       = 32
BATCH_SIZE      = 16
MAX_BUF         = 4000
WF_BUF_MAX      = 2000
GREEDY_FRAC     = 0.4

LR_E1      = 1e-3
LR_E2_WF   = 1e-3
LR_BENEFIT = 1e-3
LR_HARM    = 1e-4

LAMBDA_RESOURCE = 0.5   # SD-018

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
# drive_state = [w_drive, l_drive, h_drive, s_drive] -> [WANTING, LIKING, HARM_DISC, SURPRISE]
CONDITIONS = [
    ("A_SURPRISE", torch.tensor([0.0, 0.0, 0.0, 1.0])),
    ("B_WANTING",  torch.tensor([1.0, 0.0, 0.0, 0.0])),
    ("C_ABLATION", None),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
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
    flat = obs_body.flatten()
    return float(flat[11].item()) if flat.shape[0] > 11 else 0.0


def _get_energy(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    return float(flat[3].item()) if flat.shape[0] > 3 else 1.0


def _get_valence_at_replay_start(
    agent: REEAgent,
    drive_state: Optional[torch.Tensor],
) -> Optional[List[float]]:
    """
    Compute which theta_buffer entry would be selected under drive_state and
    return the valence vector [wanting, liking, harm_disc, surprise] at that node.

    Mirrors the logic of HippocampalModule._select_valence_weighted_start()
    so we can measure the valence BEFORE calling replay().
    Returns None if theta_buffer is empty or too small.
    """
    recent = agent.theta_buffer.recent  # [T, batch, world_dim] or None
    if recent is None or recent.shape[0] < 1:
        return None

    T = recent.shape[0]
    best_t = T - 1  # default: most recent

    if drive_state is not None and T > 1:
        best_priority = -float("inf")
        with torch.no_grad():
            for t in range(T):
                z_w = recent[t]
                priority = agent.residue_field.get_valence_priority(
                    z_w, drive_state.to(z_w.device)
                )
                p_val = float(priority.sum().item())
                if p_val > best_priority:
                    best_priority = p_val
                    best_t = t

    selected_z = recent[best_t]  # [batch, world_dim]
    with torch.no_grad():
        valence = agent.residue_field.evaluate_valence(selected_z)  # [batch, 4]

    return valence[0].cpu().tolist()  # [wanting, liking, harm_disc, surprise]


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _make_env_d1(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=N_RESOURCES,
        num_hazards=N_HAZARDS_D1,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=0.02,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.003,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )


def _make_env_d2(seed: int) -> CausalGridWorldV2:
    # Novel hazard at different grid layout (seed offset creates different placement)
    return CausalGridWorldV2(
        seed=seed + 100,
        size=GRID_SIZE,
        num_resources=N_RESOURCES,
        num_hazards=N_HAZARDS_D2,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=0.02,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.003,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )


# ---------------------------------------------------------------------------
# Agent factory (identical for all conditions -- drive_state differs at replay time)
# ---------------------------------------------------------------------------

def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
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
        benefit_eval_enabled=True,
        benefit_weight=0.5,
        z_goal_enabled=True,
        e1_goal_conditioned=True,
        goal_weight=1.0,
        drive_weight=2.0,
        use_resource_proximity_head=True,
        resource_proximity_weight=LAMBDA_RESOURCE,
        # SD-014 valence writes: all enabled so valence map is built identically
        # across all conditions -- only replay drive_state differs
        tonic_5ht_enabled=True,           # VALENCE_WANTING via update_benefit_salience
        surprise_gated_replay=True,       # VALENCE_SURPRISE via MECH-205 PE tracking
        pe_ema_alpha=0.02,
        pe_surprise_threshold=0.001,
        valence_harm_enabled=True,        # VALENCE_HARM_DISCRIMINATIVE (new 2026-04-17)
        valence_liking_enabled=True,      # VALENCE_LIKING (new 2026-04-17)
        liking_threshold=0.1,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Training step (shared across phases)
# ---------------------------------------------------------------------------

def _train_step(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_dict: dict,
    device,
    e1_opt, e2_wf_opt, harm_opt, benefit_opt,
    e1_params, e2_wf_params,
    wf_buf: list, harm_pos_buf: list, harm_neg_buf: list,
    ben_zw_buf: list, ben_lbl_buf: list,
    z_world_prev, action_prev,
    use_greedy: bool = True,
) -> Tuple[dict, float, float, torch.Tensor, torch.Tensor, dict]:
    """
    One environment step + training updates. Returns (obs_dict, harm, benefit, z_world, action, metrics).
    """
    n_act = env.action_dim
    obs_body  = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    obs_harm  = obs_dict.get("harm_obs", None)

    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
    _ = agent.clock.advance()
    agent._e1_tick(latent)

    z_world_curr = latent.z_world.detach()

    # E2 world_forward buffer
    if z_world_prev is not None and action_prev is not None:
        wf_buf.append((z_world_prev, action_prev, z_world_curr))
        if len(wf_buf) > WF_BUF_MAX:
            wf_buf[:] = wf_buf[-WF_BUF_MAX:]

    # Policy: greedy toward resource or random
    if use_greedy and random.random() < GREEDY_FRAC:
        action_idx = _greedy_toward_resource(env)
    else:
        action_idx = random.randint(0, n_act - 1)
    action_oh = _onehot(action_idx, n_act, device)
    agent._last_action = action_oh

    dist = _dist_to_nearest_resource(env)
    is_near = 1.0 if dist <= 2 else 0.0

    _, harm_signal, done, info, new_obs_dict = env.step(action_oh)
    harm_val = float(harm_signal)
    b_exp    = _get_benefit_exposure(new_obs_dict["body_state"])

    # Valence writes (WANTING, LIKING, SURPRISE)
    energy = _get_energy(new_obs_dict["body_state"])
    drive  = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive)
    agent.serotonin_step(b_exp)
    agent.update_benefit_salience(b_exp)
    agent.update_liking(b_exp)

    # Residue update (triggers VALENCE_SURPRISE write via MECH-205)
    residue_metrics = agent.update_residue(harm_val)

    # Train E1
    if len(agent._world_experience_buffer) >= 2:
        e1_loss = agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            e1_opt.zero_grad(); e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
            e1_opt.step()

    # SD-018: resource proximity supervision
    rfv = new_obs_dict.get("resource_field_view", None)
    if rfv is not None:
        rp_target = rfv[12].item()
        rp_loss = agent.compute_resource_proximity_loss(rp_target, latent)
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
            e2_wf_opt.zero_grad(); wf_loss.backward()
            torch.nn.utils.clip_grad_norm_(e2_wf_params, 1.0)
            e2_wf_opt.step()
        with torch.no_grad():
            agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

    # Train E3 harm_eval (stratified)
    if harm_val < 0:
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
            [harm_pos_buf[i] for i in pi] + [harm_neg_buf[i] for i in ni], dim=0
        )
        tgt = torch.cat([
            torch.ones(k_p,  1, device=device),
            torch.zeros(k_n, 1, device=device),
        ], dim=0)
        pred = agent.e3.harm_eval(zw_b)
        hloss = F.binary_cross_entropy(pred, tgt)
        if hloss.requires_grad:
            harm_opt.zero_grad(); hloss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
            harm_opt.step()

    # Train E3 benefit_eval
    ben_zw_buf.append(z_world_curr)
    ben_lbl_buf.append(is_near)
    if len(ben_zw_buf) > MAX_BUF:
        ben_zw_buf[:] = ben_zw_buf[-MAX_BUF:]
        ben_lbl_buf[:] = ben_lbl_buf[-MAX_BUF:]

    if len(ben_zw_buf) >= 32:
        k    = min(32, len(ben_zw_buf))
        idxs = random.sample(range(len(ben_zw_buf)), k)
        zw_b = torch.cat([ben_zw_buf[i] for i in idxs], dim=0)
        lbl  = torch.tensor(
            [ben_lbl_buf[i] for i in idxs], dtype=torch.float32
        ).unsqueeze(1).to(device)
        pred_b = agent.e3.benefit_eval(zw_b)
        bloss  = F.binary_cross_entropy(pred_b, lbl)
        if bloss.requires_grad:
            benefit_opt.zero_grad(); bloss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e3.benefit_eval_head.parameters(), 0.5)
            benefit_opt.step()

    return new_obs_dict, harm_val, b_exp, z_world_curr, action_oh, residue_metrics


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(seed: int) -> Dict:
    random.seed(seed)
    results: Dict = {}

    env_d1 = _make_env_d1(seed)
    env_d2 = _make_env_d2(seed)
    device = torch.device("cpu")

    for label, drive_state in CONDITIONS:
        print(f"\n[V3-EXQ-432] Seed {seed} Condition {label}", flush=True)

        agent = _make_agent(env_d1, seed)
        agent.train()

        e1_params    = list(agent.e1.parameters())
        e2_wf_params = (
            list(agent.e2.world_transition.parameters()) +
            list(agent.e2.world_action_encoder.parameters())
        )
        e1_opt      = optim.Adam(e1_params, lr=LR_E1)
        e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
        harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_HARM)
        benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)

        wf_buf:       List[Tuple] = []
        harm_pos_buf: List        = []
        harm_neg_buf: List        = []
        ben_zw_buf:   List        = []
        ben_lbl_buf:  List        = []

        # === Phase 1: D1 training ===
        surprise_write_count_d1 = 0
        for ep in range(N_D1):
            _, obs_dict = env_d1.reset()
            agent.reset()
            z_world_prev = None
            action_prev  = None

            for step_i in range(STEPS_PER_EP):
                obs_dict, harm_val, b_exp, z_world_curr, action_oh, rm = _train_step(
                    agent, env_d1, obs_dict, device,
                    e1_opt, e2_wf_opt, harm_opt, benefit_opt,
                    e1_params, e2_wf_params,
                    wf_buf, harm_pos_buf, harm_neg_buf,
                    ben_zw_buf, ben_lbl_buf,
                    z_world_prev, action_prev,
                )
                surprise_write_count_d1 += agent._surprise_write_count
                agent._surprise_write_count = 0
                z_world_prev = z_world_curr
                action_prev  = action_oh
                if obs_dict.get("done", False):
                    break

            if (ep + 1) % 25 == 0:
                print(
                    f"  [D1] cond={label} seed={seed}"
                    f" ep={ep+1}/{N_D1}"
                    f" harm_pos={len(harm_pos_buf)}"
                    f" s_writes={surprise_write_count_d1}",
                    flush=True,
                )

        # === Phase 2: D2 revaluation + Phase 3: offline consolidation ===
        surprise_write_count_d2 = 0
        # Per-condition replay-start valence tracking
        wanting_at_start:  List[float] = []
        surprise_at_start: List[float] = []
        harm_d2: List[float] = []

        for ep in range(N_D2):
            _, obs_dict = env_d2.reset()
            agent.reset()
            z_world_prev = None
            action_prev  = None
            ep_harm = 0.0

            for step_i in range(STEPS_PER_EP):
                obs_dict, harm_val, b_exp, z_world_curr, action_oh, rm = _train_step(
                    agent, env_d2, obs_dict, device,
                    e1_opt, e2_wf_opt, harm_opt, benefit_opt,
                    e1_params, e2_wf_params,
                    wf_buf, harm_pos_buf, harm_neg_buf,
                    ben_zw_buf, ben_lbl_buf,
                    z_world_prev, action_prev,
                )
                surprise_write_count_d2 += agent._surprise_write_count
                agent._surprise_write_count = 0
                if harm_val < 0:
                    ep_harm += abs(harm_val)
                z_world_prev = z_world_curr
                action_prev  = action_oh
                if obs_dict.get("done", False):
                    break

            harm_d2.append(ep_harm)

            # Phase 3: measure valence at replay start, then call replay
            for _ in range(N_REPLAY_PER_EP):
                valence = _get_valence_at_replay_start(agent, drive_state)
                if valence is not None:
                    wanting_at_start.append(valence[0])   # VALENCE_WANTING
                    surprise_at_start.append(valence[3])  # VALENCE_SURPRISE

                # Actual replay call (generates trajectories, E3 evaluation)
                recent = agent.theta_buffer.recent
                if recent is not None and recent.shape[0] > 0:
                    ds = drive_state.to(device) if drive_state is not None else None
                    agent.hippocampal.replay(recent, num_replay_steps=3, drive_state=ds)

            if (ep + 1) % 10 == 0:
                mean_w = sum(wanting_at_start) / max(1, len(wanting_at_start))
                mean_s = sum(surprise_at_start) / max(1, len(surprise_at_start))
                print(
                    f"  [D2] cond={label} seed={seed}"
                    f" ep={ep+1}/{N_D2}"
                    f" s_writes={surprise_write_count_d2}"
                    f" mean_w@start={mean_w:.4f}"
                    f" mean_s@start={mean_s:.4f}",
                    flush=True,
                )

        # === Phase 4: test (eval only, no training) ===
        agent.eval()
        harm_test: List[float] = []
        for ep in range(N_TEST):
            _, obs_dict = env_d2.reset()
            agent.reset()
            ep_harm = 0.0

            for step_i in range(STEPS_PER_EP):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm  = obs_dict.get("harm_obs", None)
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)

                n_act = env_d2.action_dim
                action_idx = _greedy_toward_resource(env_d2)
                action_oh  = _onehot(action_idx, n_act, device)
                agent._last_action = action_oh

                _, harm_signal, done, info, obs_dict = env_d2.step(action_oh)
                if float(harm_signal) < 0:
                    ep_harm += abs(float(harm_signal))
                if done:
                    break

            harm_test.append(ep_harm)

        mean_w = sum(wanting_at_start)  / max(1, len(wanting_at_start))
        mean_s = sum(surprise_at_start) / max(1, len(surprise_at_start))
        mean_harm_test = sum(harm_test) / max(1, len(harm_test))

        print(
            f"  [RESULT] cond={label} seed={seed}"
            f" n_replay_starts={len(wanting_at_start)}"
            f" mean_wanting@start={mean_w:.5f}"
            f" mean_surprise@start={mean_s:.5f}"
            f" s_writes_d2={surprise_write_count_d2}"
            f" mean_harm_test={mean_harm_test:.4f}",
            flush=True,
        )

        results[label] = {
            "mean_wanting_at_start":  mean_w,
            "mean_surprise_at_start": mean_s,
            "n_replay_starts":        len(wanting_at_start),
            "n_surprise_writes_d2":   surprise_write_count_d2,
            "mean_harm_test":         mean_harm_test,
        }

    return results


# ---------------------------------------------------------------------------
# Criteria evaluation
# ---------------------------------------------------------------------------

def _check_criteria(results: Dict) -> Dict:
    """Evaluate P1, P2, P3 for a single seed."""
    A = results.get("A_SURPRISE", {})
    B = results.get("B_WANTING",  {})

    a_s = A.get("mean_surprise_at_start", 0.0)
    b_s = B.get("mean_surprise_at_start", 0.0)
    a_w = A.get("mean_wanting_at_start",  0.0)
    b_w = B.get("mean_wanting_at_start",  0.0)
    n_s = A.get("n_surprise_writes_d2",   0)

    p1 = (a_s - b_s) > SURPRISE_MARGIN
    p2 = (b_w - a_w) > WANTING_MARGIN
    p3 = n_s > MIN_SURPRISE_WRITES

    return {"P1": p1, "P2": p2, "P3": p3, "pass": p1 and p2 and p3,
            "a_surprise_at_start": a_s, "b_surprise_at_start": b_s,
            "a_wanting_at_start": a_w,  "b_wanting_at_start": b_w,
            "n_surprise_writes_d2": n_s}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.time()
    all_seed_results: List[Dict] = []
    criteria_per_seed: List[Dict] = []

    print(f"\n=== V3-EXQ-432 SD-014 Replay Gate Prioritization ===", flush=True)
    print(f"Conditions: {[c[0] for c in CONDITIONS]}", flush=True)
    print(f"Seeds: {SEEDS}  D1={N_D1}  D2={N_D2}  Test={N_TEST} eps", flush=True)

    for seed in SEEDS:
        seed_results = _run_seed(seed)
        all_seed_results.append(seed_results)
        crit = _check_criteria(seed_results)
        criteria_per_seed.append(crit)
        status = "PASS" if crit["pass"] else "FAIL"
        print(
            f"\n[SEED {seed}] {status}"
            f" P1={crit['P1']} P2={crit['P2']} P3={crit['P3']}"
            f" a_s@start={crit['a_surprise_at_start']:.5f}"
            f" b_s@start={crit['b_surprise_at_start']:.5f}"
            f" a_w@start={crit['a_wanting_at_start']:.5f}"
            f" b_w@start={crit['b_wanting_at_start']:.5f}"
            f" n_writes={crit['n_surprise_writes_d2']}",
            flush=True,
        )

    # Majority vote
    n_pass = sum(1 for c in criteria_per_seed if c["pass"])
    overall_pass = n_pass >= MAJORITY_THRESH
    evidence_direction = "supports" if overall_pass else "does_not_support"

    print(f"\n=== AGGREGATE ===", flush=True)
    print(f"Seeds passed: {n_pass}/{len(SEEDS)} (threshold >= {MAJORITY_THRESH})", flush=True)
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}", flush=True)

    elapsed = time.time() - run_start
    run_id = (
        EXPERIMENT_TYPE
        + "_"
        + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        + "_v3"
    )

    manifest = {
        "run_id":                run_id,
        "experiment_type":       EXPERIMENT_TYPE,
        "claim_ids":             CLAIM_IDS,
        "architecture_epoch":    "ree_hybrid_guardrails_v1",
        "experiment_purpose":    EXPERIMENT_PURPOSE,
        "evidence_direction":    evidence_direction,
        "overall_pass":          overall_pass,
        "seeds_passed":          n_pass,
        "n_seeds":               len(SEEDS),
        "majority_threshold":    MAJORITY_THRESH,
        "criteria_per_seed":     criteria_per_seed,
        "all_seed_results":      all_seed_results,
        "hyperparams": {
            "n_d1":              N_D1,
            "n_d2":              N_D2,
            "n_test":            N_TEST,
            "n_replay_per_ep":   N_REPLAY_PER_EP,
            "grid_size":         GRID_SIZE,
            "n_hazards_d1":      N_HAZARDS_D1,
            "n_hazards_d2":      N_HAZARDS_D2,
            "steps_per_ep":      STEPS_PER_EP,
            "world_dim":         WORLD_DIM,
            "surprise_margin":   SURPRISE_MARGIN,
            "wanting_margin":    WANTING_MARGIN,
            "min_surprise_writes": MIN_SURPRISE_WRITES,
        },
        "elapsed_seconds":       round(elapsed, 1),
        "timestamp_utc":         datetime.utcnow().isoformat() + "Z",
    }

    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
