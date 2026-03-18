"""
V3-EXQ-021 — lstsq Linear Reafference Correction (SD-007, MECH-098)

Claims: SD-007 (encoder.perspective_corrected_world_latent), MECH-098
(encoder.reafference_cancellation), SD-003 (causal attribution).

Motivation (2026-03-18):
  EXQ-016 used a SGD MLP ReafferencePredictor and achieved R²_test=0.118 vs
  the EXQ-014 lstsq benchmark of R²=0.333 (same feature space). Root causes:
    1. SGD MLP with 200 steps doesn't converge to the linear solution
    2. C3 metric bug: correction was applied to BOTH endpoints of Δz_world,
       causing cancellation: dz_corrected = (z_curr - pred) - (z_prev - pred)
       = z_curr - z_prev = dz_raw (correction had zero effect)

  This experiment fixes both issues:
    1. Replaces SGD MLP with torch.linalg.lstsq on the full collected dataset
       Feature matrix: X = [z_self | a_onehot | 1]  (self_dim + action_dim + 1)
       Target matrix:  Y = Δz_world  (world_dim)
       Solution: W = lstsq(X, Y)  →  pred = X @ W
       Expected R²_test ≈ 0.333 (matching EXQ-014 linear prediction benchmark)
    2. Fixed C3 metric: dz_corrected = dz_raw - pred (subtract ONCE from delta,
       not from each endpoint separately)

PASS criteria (ALL must hold):
  C1: calibration_gap > 0.05 using lstsq-corrected z_world in SD-003 probe
  C2: reafference_r2_test > 0.25 (lstsq should match EXQ-014 benchmark)
  C3: mean_dz_corrected(empty) < mean_dz_raw(empty)
      (correction reduced perspective shift — using fixed per-step delta metric)
  C4: warmup harm events > 100
  C5: No fatal errors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_021_lstsq_reafference"
CLAIM_IDS = ["SD-007", "MECH-098", "SD-003"]

E2_ROLLOUT_STEPS = 5
RECON_WEIGHT = 1.0


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _random_cf_action(actual_idx: int, num_actions: int) -> int:
    choices = [a for a in range(num_actions) if a != actual_idx]
    return random.choice(choices) if choices else 0


def _make_world_decoder(world_dim: int, world_obs_dim: int, hidden_dim: int = 64):
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, world_obs_dim),
    )


def _make_net_eval_head(world_dim: int, hidden_dim: int = 64) -> nn.Module:
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
        nn.Tanh(),
    )


def _compute_e2w_loss(agent, traj_buffer, rollout_steps, batch_size=8):
    if len(traj_buffer) < 2:
        return next(agent.e1.parameters()).sum() * 0.0
    n = min(batch_size, len(traj_buffer))
    idxs = torch.randperm(len(traj_buffer))[:n].tolist()
    total = next(agent.e1.parameters()).sum() * 0.0
    count = 0
    for idx in idxs:
        seg = traj_buffer[idx]
        if len(seg) < rollout_steps + 1:
            continue
        z = seg[0][0]
        z_target = seg[rollout_steps][0]
        for k in range(rollout_steps):
            z = agent.e2.world_forward(z, seg[k][1])
        total = total + F.mse_loss(z, z_target)
        count += 1
    return total / count if count > 0 else total


def _train_and_collect(
    agent: REEAgent,
    env: CausalGridWorld,
    world_decoder: nn.Module,
    net_eval_head: nn.Module,
    optimizer: optim.Optimizer,
    net_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Training phase + collect reafference data for lstsq fit."""
    agent.train()
    world_decoder.train()
    net_eval_head.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    # Reafference data: (z_self_prev, a_prev, dz_world_vec) for empty-space steps
    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_REAF_DATA = 5000  # More data for lstsq (vs 2000 for SGD)

    signal_buffer: List[Tuple[torch.Tensor, float]] = []
    MAX_SIGNAL_BUFFER = 1000

    total_harm = 0
    total_benefit = 0
    n_empty_steps = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        episode_traj: List = []

        z_world_prev = None
        z_self_prev  = None
        a_prev       = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_self_curr  = latent.z_self.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            episode_traj.append((latent.z_world.detach(), action.detach()))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if harm_signal < 0:
                total_harm += 1
            elif harm_signal > 0:
                total_benefit += 1

            # Collect reafference data: (z_self_prev, a_prev, Δz_world) on empty steps
            if (z_world_prev is not None and z_self_prev is not None and
                    a_prev is not None and ttype == "none"):
                dz_world = z_world_curr - z_world_prev  # [1, world_dim]
                reaf_data.append((z_self_prev.cpu(), a_prev.cpu(), dz_world.cpu()))
                n_empty_steps += 1
                if len(reaf_data) > MAX_REAF_DATA:
                    reaf_data = reaf_data[-MAX_REAF_DATA:]

            z_world_prev = z_world_curr
            z_self_prev  = z_self_curr
            a_prev       = action.detach()

            # Collect (z_world, signal) for net_eval
            if harm_signal != 0.0 or (step % 3 == 0):
                signal_buffer.append((latent.z_world.detach(), float(harm_signal)))
                if len(signal_buffer) > MAX_SIGNAL_BUFFER:
                    signal_buffer = signal_buffer[-MAX_SIGNAL_BUFFER:]

            # E1 + E2_self + E2_world + reconstruction
            e1_loss      = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            e2w_loss     = _compute_e2w_loss(agent, traj_buffer, E2_ROLLOUT_STEPS)

            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_w)
            recon_loss = F.mse_loss(recon, obs_w)

            total_loss = e1_loss + e2_self_loss + e2w_loss + RECON_WEIGHT * recon_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_decoder.parameters(), 1.0)
                optimizer.step()

            # net_eval regression
            if len(signal_buffer) >= 8:
                k = min(32, len(signal_buffer))
                idxs_sel = torch.randperm(len(signal_buffer))[:k].tolist()
                zw_b = torch.cat([signal_buffer[i][0] for i in idxs_sel], dim=0)
                sv_b = torch.tensor(
                    [signal_buffer[i][1] for i in idxs_sel], device=agent.device
                ).unsqueeze(1)
                pred = net_eval_head(zw_b)
                sv_norm = (sv_b / 0.5).clamp(-1.0, 1.0)
                net_loss = F.mse_loss(pred, sv_norm)
                if net_loss.requires_grad:
                    net_eval_optimizer.zero_grad()
                    net_loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_eval_head.parameters(), 0.5)
                    net_eval_optimizer.step()

            if done:
                break

        for start in range(0, len(episode_traj) - E2_ROLLOUT_STEPS):
            traj_buffer.append(episode_traj[start:start + E2_ROLLOUT_STEPS + 1])
        if len(traj_buffer) > MAX_TRAJ_BUFFER:
            traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  "
                  f"benefit={total_benefit}  empty_steps={n_empty_steps}", flush=True)

    return {
        "total_harm":    total_harm,
        "total_benefit": total_benefit,
        "n_empty_steps": n_empty_steps,
        "reaf_data":     reaf_data,
    }


def _fit_lstsq_predictor(
    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    self_dim: int,
    action_dim: int,
    world_dim: int,
) -> Tuple[Optional[torch.Tensor], float, float]:
    """
    Fit a linear reafference predictor using torch.linalg.lstsq.

    Feature matrix: X = [z_self | a_onehot | 1]  (n × (self_dim + action_dim + 1))
    Target matrix:  Y = Δz_world                  (n × world_dim)
    Solution:       W = lstsq(X, Y)               ((self_dim + action_dim + 1) × world_dim)
    Prediction:     Y_pred = X @ W

    Returns (W, r2_train, r2_test).
    """
    if len(reaf_data) < 20:
        print(f"  WARNING: only {len(reaf_data)} empty-step records — lstsq skipped",
              flush=True)
        return None, 0.0, 0.0

    n = len(reaf_data)
    n_train = int(n * 0.8)

    z_self_all = torch.cat([d[0] for d in reaf_data], dim=0)  # [n, self_dim]
    a_all      = torch.cat([d[1] for d in reaf_data], dim=0)  # [n, action_dim]
    dz_all     = torch.cat([d[2] for d in reaf_data], dim=0)  # [n, world_dim]

    ones_all = torch.ones(n, 1)
    X_all = torch.cat([z_self_all, a_all, ones_all], dim=-1)  # [n, feat_dim]

    X_train = X_all[:n_train]
    dz_train = dz_all[:n_train]
    X_test  = X_all[n_train:]
    dz_test = dz_all[n_train:]

    def _r2(X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor) -> float:
        with torch.no_grad():
            pred = X @ W
            ss_res = ((Y - pred) ** 2).sum()
            ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum()
            return float((1 - ss_res / (ss_tot + 1e-8)).item())

    # Fit with lstsq (closed-form, driver='gelsd' for numerical stability)
    with torch.no_grad():
        result = torch.linalg.lstsq(X_train, dz_train, driver="gelsd")
        W = result.solution  # [(self_dim + action_dim + 1), world_dim]

    r2_train = _r2(X_train[:256], dz_train[:256], W)
    r2_test  = _r2(X_test, dz_test, W)

    print(f"  lstsq predictor: n_train={n_train}  n_test={n-n_train}  "
          f"R²_train={r2_train:.3f}  R²_test={r2_test:.3f}", flush=True)

    return W, r2_train, r2_test


def _apply_lstsq_correction(
    z_world_raw: torch.Tensor,
    z_self: torch.Tensor,
    a: torch.Tensor,
    W: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Apply lstsq reafference correction to raw z_world.
    z_world_corrected = z_world_raw - W.T @ [z_self, a, 1]
    """
    batch = z_self.shape[0]
    ones = torch.ones(batch, 1, device=device)
    feat = torch.cat([z_self, a, ones], dim=-1)    # [batch, feat_dim]
    pred = feat @ W.to(device)                      # [batch, world_dim]
    return z_world_raw - pred


def _measure_dz_correction(
    agent: REEAgent,
    W: torch.Tensor,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Measure mean ||Δz_world|| before and after lstsq correction.

    FIXED (vs EXQ-016): correction is subtracted ONCE from the delta, not
    from each endpoint separately.
      dz_corrected = (z_world_curr - z_world_prev) - lstsq_pred(z_self_prev, a_prev)
    """
    agent.eval()

    dz_raw_empty: List[float] = []
    dz_cor_empty: List[float] = []
    dz_raw_env:   List[float] = []
    dz_cor_env:   List[float] = []

    device = agent.device

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev = None
        z_self_prev  = None
        a_prev       = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                z_world_curr = latent.z_world.detach()
                z_self_curr  = latent.z_self.detach()

                action_idx = random.randint(0, env.action_dim - 1)
                action = _action_to_onehot(action_idx, env.action_dim, device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if z_world_prev is not None and z_self_prev is not None and a_prev is not None:
                with torch.no_grad():
                    # Raw delta
                    dz_vec = z_world_curr - z_world_prev  # [1, world_dim]
                    dz_raw = float(torch.norm(dz_vec).item())

                    # Lstsq-corrected delta: subtract predicted locomotion component once
                    batch = z_self_prev.shape[0]
                    ones  = torch.ones(batch, 1, device=device)
                    feat  = torch.cat([z_self_prev, a_prev, ones], dim=-1)
                    pred  = feat @ W.to(device)
                    dz_vec_corrected = dz_vec - pred
                    dz_cor = float(torch.norm(dz_vec_corrected).item())

                    if ttype == "none":
                        dz_raw_empty.append(dz_raw)
                        dz_cor_empty.append(dz_cor)
                    elif ttype == "env_caused_hazard":
                        dz_raw_env.append(dz_raw)
                        dz_cor_env.append(dz_cor)

            z_world_prev = z_world_curr
            z_self_prev  = z_self_curr
            a_prev       = action.detach()

            if done:
                break

    mean_dz_raw_empty = float(sum(dz_raw_empty) / max(1, len(dz_raw_empty)))
    mean_dz_cor_empty = float(sum(dz_cor_empty) / max(1, len(dz_cor_empty)))
    mean_dz_raw_env   = float(sum(dz_raw_env)   / max(1, len(dz_raw_env)))
    mean_dz_cor_env   = float(sum(dz_cor_env)   / max(1, len(dz_cor_env)))

    print(f"  Δz correction (fixed C3) — empty: raw={mean_dz_raw_empty:.4f}  "
          f"corrected={mean_dz_cor_empty:.4f}  "
          f"reduction={mean_dz_raw_empty - mean_dz_cor_empty:.4f}", flush=True)
    print(f"  Δz correction — env_hazard: raw={mean_dz_raw_env:.4f}  "
          f"corrected={mean_dz_cor_env:.4f}", flush=True)

    return {
        "mean_dz_raw_empty":       mean_dz_raw_empty,
        "mean_dz_corrected_empty": mean_dz_cor_empty,
        "mean_dz_raw_env":         mean_dz_raw_env,
        "mean_dz_corrected_env":   mean_dz_cor_env,
        "n_empty":                 len(dz_raw_empty),
        "n_env":                   len(dz_raw_env),
    }


def _eval_corrected_probes(
    agent: REEAgent,
    net_eval_head: nn.Module,
    W: torch.Tensor,
    env: CausalGridWorld,
    num_resets: int,
) -> Dict:
    """SD-003 probe with lstsq reafference correction applied to z_world."""
    agent.eval()
    net_eval_head.eval()

    near_sigs: List[float] = []
    safe_sigs:  List[float] = []
    all_pred_vals: List[float] = []
    fatal_errors = 0

    wall_type   = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]
    device = agent.device

    def _probe(ax: int, ay: int, actual_idx: int) -> float:
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            latent  = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world = latent.z_world
            z_self  = latent.z_self

            a_act = _action_to_onehot(actual_idx, env.action_dim, device)
            cf_idx = _random_cf_action(actual_idx, env.action_dim)
            a_cf  = _action_to_onehot(cf_idx, env.action_dim, device)

            # Apply lstsq correction with current z_self as proxy for z_self_prev
            z_world_corr_act = _apply_lstsq_correction(z_world, z_self, a_act, W, device)
            z_world_corr_cf  = _apply_lstsq_correction(z_world, z_self, a_cf,  W, device)

            zw_act = agent.e2.world_forward(z_world_corr_act, a_act)
            zw_cf  = agent.e2.world_forward(z_world_corr_cf,  a_cf)

            v_act = net_eval_head(zw_act)
            v_cf  = net_eval_head(zw_cf)
            all_pred_vals.extend([float(v_act.item()), float(v_cf.item())])
            return float((v_act - v_cf).item())

    try:
        for _ in range(num_resets):
            env.reset()
            for hx, hy in env.hazards:
                for action_idx, (dx, dy) in env.ACTIONS.items():
                    if action_idx == 4:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < env.size and 0 <= ay < env.size:
                        cell = int(env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            near_sigs.append(_probe(ax, ay, action_idx))
            for px in range(env.size):
                for py in range(env.size):
                    if int(env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(abs(px - hx) + abs(py - hy) for hx, hy in env.hazards)
                    if min_dist > 3:
                        safe_sigs.append(_probe(px, py, random.randint(0, 3)))
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL probe: {traceback.format_exc()}", flush=True)

    mean_near = float(sum(near_sigs) / max(1, len(near_sigs)))
    mean_safe = float(sum(safe_sigs)  / max(1, len(safe_sigs)))
    gap = mean_near - mean_safe
    pred_std = float(torch.tensor(all_pred_vals).std().item()) if len(all_pred_vals) > 1 else 0.0

    print(f"  Corrected probe  n_near={len(near_sigs)} n_safe={len(safe_sigs)}  "
          f"near={mean_near:.4f}  safe={mean_safe:.4f}  gap={gap:.4f}  "
          f"pred_std={pred_std:.4f}", flush=True)

    return {
        "calibration_gap":      gap,
        "mean_causal_sig_near": mean_near,
        "mean_causal_sig_safe": mean_safe,
        "n_near_hazard_probes": len(near_sigs),
        "n_safe_probes":        len(safe_sigs),
        "net_eval_pred_std":    pred_std,
        "fatal_errors":         fatal_errors,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 300,
    eval_probe_resets: int = 10,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
    )
    agent = REEAgent(config)
    world_decoder = _make_world_decoder(world_dim, env.world_obs_dim)
    net_eval_head = _make_net_eval_head(world_dim)

    e12_params = [p for n_, p in agent.named_parameters() if "harm_eval" not in n_]
    e12_params += list(world_decoder.parameters())
    optimizer      = optim.Adam(e12_params, lr=lr)
    net_eval_optim = optim.Adam(net_eval_head.parameters(), lr=1e-4)

    print(f"[V3-EXQ-021] Warmup + collect: {warmup_episodes} eps (12×12, 15 hazards, drift)",
          flush=True)
    train_out = _train_and_collect(
        agent, env, world_decoder, net_eval_head,
        optimizer, net_eval_optim,
        warmup_episodes, steps_per_episode,
    )
    warmup_harm    = train_out["total_harm"]
    warmup_benefit = train_out["total_benefit"]
    n_empty_steps  = train_out["n_empty_steps"]
    reaf_data      = train_out["reaf_data"]

    print(f"  Warmup done. harm={warmup_harm}  benefit={warmup_benefit}  "
          f"empty_steps_collected={n_empty_steps}", flush=True)

    # Fit lstsq predictor
    print(f"[V3-EXQ-021] Fitting lstsq predictor on {len(reaf_data)} empty steps...",
          flush=True)
    W, r2_train, r2_test = _fit_lstsq_predictor(
        reaf_data, self_dim, env.action_dim, world_dim
    )

    if W is None:
        # Fallback: identity (no correction)
        feat_dim = self_dim + env.action_dim + 1
        W = torch.zeros(feat_dim, world_dim)
        r2_test = 0.0
    reafference_r2 = max(0.0, r2_test)

    # Measure Δz correction (fixed C3 metric)
    print(f"[V3-EXQ-021] Measuring Δz correction (20 episodes, fixed C3 metric)...",
          flush=True)
    dz_stats = _measure_dz_correction(agent, W, env, 20, steps_per_episode)

    # SD-003 probe with lstsq-corrected z_world
    print(f"[V3-EXQ-021] SD-003 probe ({eval_probe_resets} resets, lstsq-corrected)...",
          flush=True)
    probe = _eval_corrected_probes(agent, net_eval_head, W, env, eval_probe_resets)
    fatal_errors = probe["fatal_errors"]

    # PASS / FAIL
    c1_pass = probe["calibration_gap"] > 0.05
    c2_pass = reafference_r2 > 0.25
    c3_pass = dz_stats["mean_dz_corrected_empty"] < dz_stats["mean_dz_raw_empty"]
    c4_pass = warmup_harm > 100
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: calibration_gap={probe['calibration_gap']:.4f} <= 0.05"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: R²_test={reafference_r2:.3f} <= 0.25 "
            f"(EXQ-014 lstsq benchmark = 0.333)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: Δz_corrected(empty)={dz_stats['mean_dz_corrected_empty']:.4f} >= "
            f"Δz_raw(empty)={dz_stats['mean_dz_raw_empty']:.4f}"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: warmup_harm={warmup_harm} <= 100")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-021 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count":                  float(fatal_errors),
        "warmup_harm_events":                 float(warmup_harm),
        "warmup_benefit_events":              float(warmup_benefit),
        "n_empty_steps_collected":            float(n_empty_steps),
        "reafference_reconstruction_r2":      float(reafference_r2),
        "reafference_r2_train":               float(r2_train),
        "calibration_gap":                    float(probe["calibration_gap"]),
        "mean_causal_sig_near_hazard":        float(probe["mean_causal_sig_near"]),
        "mean_causal_sig_safe":               float(probe["mean_causal_sig_safe"]),
        "n_near_hazard_probes":               float(probe["n_near_hazard_probes"]),
        "n_safe_probes":                      float(probe["n_safe_probes"]),
        "net_eval_pred_std":                  float(probe["net_eval_pred_std"]),
        "mean_dz_world_raw_empty":            float(dz_stats["mean_dz_raw_empty"]),
        "mean_dz_world_corrected_empty":      float(dz_stats["mean_dz_corrected_empty"]),
        "dz_correction_reduction_empty":      float(dz_stats["mean_dz_raw_empty"] - dz_stats["mean_dz_corrected_empty"]),
        "mean_dz_world_raw_env_hazard":       float(dz_stats["mean_dz_raw_env"]),
        "mean_dz_world_corrected_env_hazard": float(dz_stats["mean_dz_corrected_env"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-021 — lstsq Linear Reafference Correction

**Status:** {status}
**Warmup:** {warmup_episodes} eps (RANDOM policy, 12×12, 15 hazards, drift_interval=3, drift_prob=0.5)
**Probe eval:** {eval_probe_resets} grid resets
**Seed:** {seed}

## Motivation (SD-007 / MECH-098, EXQ-016 bug fixes)

EXQ-016 failed on C2 (R²_test=0.118 vs threshold 0.2) and C3 (metric was broken).
This experiment fixes both issues:
  1. **lstsq**: torch.linalg.lstsq on [z_self|a|1] → Δz_world (expected R²≈0.333)
  2. **Fixed C3**: dz_corrected = dz_raw - pred (subtract ONCE, not from each endpoint)

## lstsq Predictor

| Metric | Value |
|---|---|
| Feature dim | {self_dim + env.action_dim + 1} = self_dim({self_dim}) + actions({env.action_dim}) + 1 |
| n_empty_steps | {n_empty_steps} |
| R²_train | {r2_train:.3f} |
| R²_test | {reafference_r2:.3f} |

## Δz_world Before/After Correction (Fixed Metric)

| Event Type | Raw Δz_world | Corrected Δz_world | Reduction |
|---|---|---|---|
| empty_move (locomotion) | {dz_stats["mean_dz_raw_empty"]:.4f} | {dz_stats["mean_dz_corrected_empty"]:.4f} | {dz_stats["mean_dz_raw_empty"] - dz_stats["mean_dz_corrected_empty"]:.4f} |
| env_caused_hazard | {dz_stats["mean_dz_raw_env"]:.4f} | {dz_stats["mean_dz_corrected_env"]:.4f} | {dz_stats["mean_dz_raw_env"] - dz_stats["mean_dz_corrected_env"]:.4f} |

## SD-003 Attribution (lstsq-Corrected z_world)

| Position | mean(causal_sig) |
|---|---|
| Near-hazard | {probe["mean_causal_sig_near"]:.4f} |
| Safe | {probe["mean_causal_sig_safe"]:.4f} |
| **calibration_gap** | **{probe["calibration_gap"]:.4f}** |

Warmup: harm={warmup_harm}  benefit={warmup_benefit}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap > 0.05 (lstsq-corrected) | {"PASS" if c1_pass else "FAIL"} | {probe["calibration_gap"]:.4f} |
| C2: R²_test > 0.25 | {"PASS" if c2_pass else "FAIL"} | {reafference_r2:.3f} |
| C3: Δz_corrected(empty) < Δz_raw(empty) | {"PASS" if c3_pass else "FAIL"} | {dz_stats["mean_dz_corrected_empty"]:.4f} vs {dz_stats["mean_dz_raw_empty"]:.4f} |
| C4: Warmup harm events > 100 | {"PASS" if c4_pass else "FAIL"} | {warmup_harm} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {fatal_errors} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--warmup",       type=int, default=300)
    parser.add_argument("--probe-resets", type=int, default=10)
    parser.add_argument("--steps",        type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
