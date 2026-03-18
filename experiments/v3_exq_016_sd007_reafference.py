"""
V3-EXQ-016 — SD-007 Reafference Correction

Claims: SD-007 (encoder.perspective_corrected_world_latent), MECH-098
(encoder.reafference_cancellation), SD-003 (causal attribution).

Motivation (2026-03-17):
  EXQ-014 quantified perspective_shift_dominance: locomotion explains > 30% of
  z_world variance (expected). The fix (SD-007) adds a ReafferencePredictor:
    z_world_corrected = z_world_raw - ReafferencePredictor(z_self_prev, a_prev)
  trained on empty-space steps to predict locomotion-induced z_world change.

  After correction, z_world_corrected retains only genuine world-state changes
  (exafference), not perspective shift (reafference). The SD-003 attribution
  pipeline then uses z_world_corrected throughout, giving E2_world and
  E3.harm_eval a cleaner input that actually varies with action choice.

  Implementation (MSTd-equivalent, z-space level):
  - ReafferencePredictor: Linear(self_dim + action_dim, 64) → ReLU → Linear(64, world_dim)
  - Trained on empty-space steps (transition_type == "none"):
    minimize MSE(Δz_world_predicted, z_world_actual - z_world_prev)
  - Applied at inference: z_world_corrected = z_world_raw - prediction
  - SD-003 probe uses z_world_corrected in both E2.world_forward and E3.harm_eval

PASS criteria (ALL must hold):
  C1: calibration_gap > 0.05 using reafference-corrected z_world in SD-003 probe
  C2: reafference_reconstruction_r2 > 0.2 on held-out empty steps
      (confirms predictor learned something real about locomotion → z_world change)
  C3: mean_dz_world_corrected(empty_move) < mean_dz_world_raw(empty_move)
      (correction reduced perspective shift in z_world)
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
from ree_core.predictors.e2_fast import ReafferencePredictor


EXPERIMENT_TYPE = "v3_exq_016_sd007_reafference"
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
    """
    Training phase + collect reafference data simultaneously.
    Returns training metrics and collected reafference data for predictor training.
    """
    agent.train()
    world_decoder.train()
    net_eval_head.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    # Reafference data: (z_self_prev, a_prev, dz_world_vec) for empty-space steps only
    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_REAF_DATA = 2000

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
            ttype = info["transition_type"]

            if harm_signal < 0:
                total_harm += 1
            elif harm_signal > 0:
                total_benefit += 1

            # Collect reafference data on empty-space steps
            if (z_world_prev is not None and z_self_prev is not None and
                    a_prev is not None and ttype == "none"):
                dz_world = (z_world_curr - z_world_prev)  # [1, world_dim]
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
            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            e2w_loss = _compute_e2w_loss(agent, traj_buffer, E2_ROLLOUT_STEPS)

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
        "total_harm": total_harm,
        "total_benefit": total_benefit,
        "n_empty_steps": n_empty_steps,
        "reaf_data": reaf_data,
    }


def _train_reafference_predictor(
    reaf_data: List,
    self_dim: int,
    action_dim: int,
    world_dim: int,
    device: str = "cpu",
) -> Tuple[ReafferencePredictor, float, float]:
    """
    Train ReafferencePredictor on empty-space steps.
    Returns (predictor, r2_train, r2_test).
    """
    if len(reaf_data) < 20:
        print(f"  WARNING: only {len(reaf_data)} empty-step records — skipping reaf fit",
              flush=True)
        rp = ReafferencePredictor(self_dim, action_dim, world_dim)
        return rp, 0.0, 0.0

    n = len(reaf_data)
    n_train = int(n * 0.8)

    z_self_all = torch.cat([d[0] for d in reaf_data], dim=0)  # [n, self_dim]
    a_all      = torch.cat([d[1] for d in reaf_data], dim=0)  # [n, action_dim]
    dz_all     = torch.cat([d[2] for d in reaf_data], dim=0)  # [n, world_dim]

    # Split train/test
    z_self_train = z_self_all[:n_train]
    a_train      = a_all[:n_train]
    dz_train     = dz_all[:n_train]
    z_self_test  = z_self_all[n_train:]
    a_test       = a_all[n_train:]
    dz_test      = dz_all[n_train:]

    rp = ReafferencePredictor(self_dim, action_dim, world_dim)
    rp_optimizer = optim.Adam(rp.parameters(), lr=1e-3)

    # Train for 200 mini-batch steps
    BATCH = 32
    for step_i in range(200):
        if n_train < BATCH:
            break
        idxs = torch.randperm(n_train)[:BATCH]
        z_s = z_self_train[idxs]
        a_s = a_train[idxs]
        dz_s = dz_train[idxs]
        pred = rp(z_s, a_s)
        loss = F.mse_loss(pred, dz_s)
        rp_optimizer.zero_grad()
        loss.backward()
        rp_optimizer.step()

    rp.eval()

    def _r2(z_s, a_s, dz_s):
        with torch.no_grad():
            pred = rp(z_s, a_s)
            ss_res = ((dz_s - pred) ** 2).sum()
            ss_tot = ((dz_s - dz_s.mean(dim=0, keepdim=True)) ** 2).sum()
            return float((1 - ss_res / (ss_tot + 1e-8)).item())

    r2_train = _r2(z_self_train[:256], a_train[:256], dz_train[:256])
    r2_test  = _r2(z_self_test,  a_test,  dz_test)

    print(f"  ReafferencePredictor: n_train={n_train}  n_test={n-n_train}  "
          f"R²_train={r2_train:.3f}  R²_test={r2_test:.3f}", flush=True)

    return rp, r2_train, r2_test


def _eval_corrected_probes(
    agent: REEAgent,
    net_eval_head: nn.Module,
    reaf_predictor: ReafferencePredictor,
    env: CausalGridWorld,
    num_resets: int,
    world_dim: int,
) -> Dict:
    """
    Probe using reafference-corrected z_world.

    For each position, we use the previous z_self as the correction context.
    Since probes are stateless (we teleport the agent), we approximate
    z_self_prev as z_self at the current position (zero-motion step), which
    gives zero correction. Instead, we use the "stay" action (action_idx=4)
    to get a representative z_self and apply correction assuming a move step.

    The key test: does mean(causal_sig at near-hazard) > mean(causal_sig at safe)?
    causal_sig = net_eval(E2.world_forward(z_world_corrected, a_actual))
               - net_eval(E2.world_forward(z_world_corrected, a_cf))
    """
    agent.eval()
    net_eval_head.eval()
    reaf_predictor.eval()

    near_sigs: List[float] = []
    safe_sigs:  List[float] = []
    near_dz_raw: List[float] = []
    near_dz_cor: List[float] = []
    empty_dz_raw: List[float] = []
    empty_dz_cor: List[float] = []
    all_pred_vals: List[float] = []
    fatal_errors = 0

    wall_type   = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _probe(ax: int, ay: int, actual_idx: int) -> float:
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world_raw = latent.z_world   # [1, world_dim]
            z_self_curr = latent.z_self    # [1, self_dim]

            # Use current z_self as the z_self_prev approximation for correction
            # (conservative: corrects as if we're about to step)
            a_act_t = _action_to_onehot(actual_idx, env.action_dim, agent.device)
            cf_idx  = _random_cf_action(actual_idx, env.action_dim)
            a_cf_t  = _action_to_onehot(cf_idx, env.action_dim, agent.device)

            z_world_corr = reaf_predictor.correct_z_world(z_world_raw, z_self_curr, a_act_t)
            z_world_cf_corr = reaf_predictor.correct_z_world(z_world_raw, z_self_curr, a_cf_t)

            zw_act = agent.e2.world_forward(z_world_corr,    a_act_t)
            zw_cf  = agent.e2.world_forward(z_world_cf_corr, a_cf_t)

            v_act = net_eval_head(zw_act)
            v_cf  = net_eval_head(zw_cf)
            all_pred_vals.extend([float(v_act.item()), float(v_cf.item())])
            return float((v_act - v_cf).item())

    try:
        for _ in range(num_resets):
            env.reset()

            # Near-hazard probes
            for hx, hy in env.hazards:
                for action_idx, (dx, dy) in env.ACTIONS.items():
                    if action_idx == 4:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < env.size and 0 <= ay < env.size:
                        cell = int(env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            near_sigs.append(_probe(ax, ay, action_idx))

            # Safe probes (min dist > 3)
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
    calibration_gap = mean_near - mean_safe
    pred_std = float(torch.tensor(all_pred_vals).std().item()) if len(all_pred_vals) > 1 else 0.0

    print(f"  Corrected probe — n_near={len(near_sigs)} n_safe={len(safe_sigs)}", flush=True)
    print(f"  near={mean_near:.4f}  safe={mean_safe:.4f}  gap={calibration_gap:.4f}  "
          f"pred_std={pred_std:.4f}", flush=True)

    return {
        "calibration_gap":         calibration_gap,
        "mean_causal_sig_near":    mean_near,
        "mean_causal_sig_safe":    mean_safe,
        "n_near_hazard_probes":    len(near_sigs),
        "n_safe_probes":           len(safe_sigs),
        "net_eval_pred_std":       pred_std,
        "fatal_errors":            fatal_errors,
    }


def _measure_dz_correction(
    agent: REEAgent,
    reaf_predictor: ReafferencePredictor,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Measure mean ||Δz_world|| before and after correction for empty-space steps."""
    agent.eval()
    reaf_predictor.eval()

    dz_raw_empty: List[float] = []
    dz_cor_empty: List[float] = []
    dz_raw_env:   List[float] = []
    dz_cor_env:   List[float] = []

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
                action = _action_to_onehot(action_idx, env.action_dim, agent.device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info["transition_type"]

            if z_world_prev is not None and z_self_prev is not None and a_prev is not None:
                with torch.no_grad():
                    dz_raw = float(torch.norm(z_world_curr - z_world_prev).item())
                    z_world_corr = reaf_predictor.correct_z_world(
                        z_world_curr, z_self_prev, a_prev
                    )
                    z_world_prev_corr = reaf_predictor.correct_z_world(
                        z_world_prev, z_self_prev, a_prev
                    )
                    dz_cor = float(torch.norm(z_world_corr - z_world_prev_corr).item())

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

    print(f"  Δz correction — empty: raw={mean_dz_raw_empty:.4f}  "
          f"corrected={mean_dz_cor_empty:.4f}  "
          f"reduction={mean_dz_raw_empty - mean_dz_cor_empty:.4f}", flush=True)
    print(f"  Δz correction — env_hazard: raw={mean_dz_raw_env:.4f}  "
          f"corrected={mean_dz_cor_env:.4f}", flush=True)

    return {
        "mean_dz_raw_empty":    mean_dz_raw_empty,
        "mean_dz_corrected_empty": mean_dz_cor_empty,
        "mean_dz_raw_env":      mean_dz_raw_env,
        "mean_dz_corrected_env": mean_dz_cor_env,
        "n_empty": len(dz_raw_empty),
        "n_env": len(dz_raw_env),
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
    world_decoder  = _make_world_decoder(world_dim, env.world_obs_dim)
    net_eval_head  = _make_net_eval_head(world_dim)

    e12_params = [p for n_, p in agent.named_parameters() if "harm_eval" not in n_]
    e12_params += list(world_decoder.parameters())
    optimizer       = optim.Adam(e12_params, lr=lr)
    net_eval_optim  = optim.Adam(net_eval_head.parameters(), lr=1e-4)

    print(f"[V3-EXQ-016] Warmup + collect: {warmup_episodes} eps (12×12, 15 hazards, drift)",
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

    # ── Train ReafferencePredictor ────────────────────────────────────────
    print(f"[V3-EXQ-016] Training ReafferencePredictor on {len(reaf_data)} empty steps...",
          flush=True)
    reaf_predictor, r2_train, r2_test = _train_reafference_predictor(
        reaf_data, self_dim, env.action_dim, world_dim
    )
    reafference_r2 = max(0.0, r2_test)

    # ── Measure z_world delta before/after correction ─────────────────────
    print(f"[V3-EXQ-016] Measuring Δz correction (20 episodes)...", flush=True)
    dz_stats = _measure_dz_correction(agent, reaf_predictor, env, 20, steps_per_episode)

    # ── SD-003 probe with corrected z_world ───────────────────────────────
    print(f"[V3-EXQ-016] SD-003 probe ({eval_probe_resets} resets, corrected z_world)...",
          flush=True)
    probe = _eval_corrected_probes(
        agent, net_eval_head, reaf_predictor, env, eval_probe_resets, world_dim
    )
    fatal_errors = probe["fatal_errors"]

    # ── PASS / FAIL ──────────────────────────────────────────────────────
    c1_pass = probe["calibration_gap"] > 0.05
    c2_pass = reafference_r2 > 0.2
    c3_pass = dz_stats["mean_dz_corrected_empty"] < dz_stats["mean_dz_raw_empty"]
    c4_pass = warmup_harm > 100
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: corrected calibration_gap={probe['calibration_gap']:.4f} <= 0.05"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: reafference R²_test={reafference_r2:.3f} <= 0.2"
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

    print(f"\nV3-EXQ-016 verdict: {status}  ({criteria_met}/5)", flush=True)
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

    summary_markdown = f"""# V3-EXQ-016 — SD-007 Reafference Correction

**Status:** {status}
**Warmup:** {warmup_episodes} eps (RANDOM policy, 12×12, 15 hazards, drift_interval=3, drift_prob=0.5)
**Probe eval:** {eval_probe_resets} grid resets
**Seed:** {seed}

## Motivation (SD-007 / MECH-098)

EXQ-012 revealed calibration_gap ≈ 0.0007 — E2 identity shortcut from perspective shift.
EXQ-014 (expected) confirmed locomotion explains > 30% of z_world variance.
This experiment applies reafference correction (SD-007) at z-space level:
  `z_world_corrected = z_world_raw - ReafferencePredictor(z_self_prev, a_prev)`
Then re-runs the SD-003 attribution probe using corrected z_world.

## ReafferencePredictor

- Architecture: Linear({self_dim + env.action_dim if hasattr(env, 'action_dim') else '?'}, 64) → ReLU → Linear(64, {world_dim})
- Trained on empty-space steps (transition_type == "none")
- R² train: {r2_train:.3f} | R² test: {reafference_r2:.3f}
- Empty-step data collected: {n_empty_steps}

## Δz_world Before/After Correction

| Event Type | Raw Δz_world | Corrected Δz_world | Reduction |
|---|---|---|---|
| empty_move (locomotion) | {dz_stats["mean_dz_raw_empty"]:.4f} | {dz_stats["mean_dz_corrected_empty"]:.4f} | {dz_stats["mean_dz_raw_empty"] - dz_stats["mean_dz_corrected_empty"]:.4f} |
| env_caused_hazard | {dz_stats["mean_dz_raw_env"]:.4f} | {dz_stats["mean_dz_corrected_env"]:.4f} | {dz_stats["mean_dz_raw_env"] - dz_stats["mean_dz_corrected_env"]:.4f} |

## SD-003 Attribution (Corrected z_world)

| Position | mean(causal_sig) |
|---|---|
| Near-hazard | {probe["mean_causal_sig_near"]:.4f} |
| Safe | {probe["mean_causal_sig_safe"]:.4f} |
| **calibration_gap** | **{probe["calibration_gap"]:.4f}** |

net_eval pred_std: {probe["net_eval_pred_std"]:.4f}
n_near={probe["n_near_hazard_probes"]}  n_safe={probe["n_safe_probes"]}
Warmup: harm={warmup_harm}  benefit={warmup_benefit}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap > 0.05 (corrected) | {"PASS" if c1_pass else "FAIL"} | {probe["calibration_gap"]:.4f} |
| C2: reafference R²_test > 0.2 | {"PASS" if c2_pass else "FAIL"} | {reafference_r2:.3f} |
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
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--warmup",         type=int, default=300)
    parser.add_argument("--probe-resets",   type=int, default=10)
    parser.add_argument("--steps",          type=int, default=200)
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
