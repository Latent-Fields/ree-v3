"""
V3-EXQ-024 — E2_world 1-Step Loss + Fixed SD-003 Probe

Claims: SD-003, SD-008, ARC-016.

Motivation (2026-03-18 diagnosis of EXQ-023):
  EXQ-023 confirmed SD-008 (alpha_world=0.9 restores event selectivity, C1 PASS).
  But C3 and C4 still FAIL due to two compounding problems identified in the diagnosis:

  Problem 1 — E2_world worse than identity baseline:
    With E2_ROLLOUT_STEPS=5 on a random policy:
      - ~50% of actions hit walls (no movement), producing unpredictable 5-step outcomes
      - E2_world can't converge to a useful model; it defaults to near-zero output
    EXQ-023 identity MSE = 0.072²/32 = 0.000162
    EXQ-023 reported e2w_mse = 0.0006–0.0008  (4–7× WORSE than identity)
    Fix: add a 1-step direct loss inline during training:
        e2w_1step_loss = MSE(E2.world_forward(z_world_t, a_t), z_world_{t+1})
    The 1-step signal is clean and immediately available from consecutive observations.
    The 5-step rollout loss is kept as auxiliary (rollout weight = 0.5, 1-step weight = 1.0).

  Problem 2 — SD-003 probe geometry broken with 15 hazards on 12×12 grid:
    15 hazards × ~25 cells at min_dist <= 3 = ~375 cell-exclusions on 144 total cells
    Result: only 27–59 valid "safe" cells — barely any safe space exists
    The "calibration_gap" compares near-hazard vs near-hazard, not near vs far
    Fix: use separate probe_env with num_hazards=6, min_dist threshold lowered to > 2

PASS criteria (ALL must hold):
  C1: event_selectivity_margin > 0.005
  C2: e2w_improvement_ratio > 2.0 (E2_world at least 2× better than identity baseline)
  C3: var_diff > 0.001 (ARC-016 precision fires with corrected E2_world)
  C4: calibration_gap > 0.05 (SD-003 threshold, probe_env with 6 hazards, min_dist > 2)
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


EXPERIMENT_TYPE = "v3_exq_024_e2w_1step_loss"
CLAIM_IDS = ["SD-003", "SD-008", "ARC-016"]

E2_ROLLOUT_STEPS = 5
E2W_1STEP_WEIGHT = 1.0    # direct 1-step loss weight
E2W_ROLLOUT_WEIGHT = 0.5  # 5-step rollout loss weight (kept as auxiliary)
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


def _compute_e2w_rollout_loss(agent, traj_buffer, rollout_steps, batch_size=8):
    """5-step rollout loss (auxiliary). Trains E2_world on multi-step consistency."""
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
    harm_norm_scale: float = 0.5,
) -> Dict:
    """Training with 1-step E2_world loss + collect diagnostic data."""
    agent.train()
    world_decoder.train()
    net_eval_head.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_REAF_DATA = 5000

    signal_buffer: List[Tuple[torch.Tensor, float]] = []
    MAX_SIGNAL_BUFFER = 1000

    dz_world_by_event: Dict[str, List[float]] = {
        "none": [], "env_caused_hazard": [], "agent_caused_hazard": []
    }
    MAX_DZ_PER_TYPE = 500

    # Track E2_world 1-step MSE for diagnostic
    e2w_1step_mses: List[float] = []
    identity_mses: List[float] = []
    MAX_MSE_TRACK = 2000

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
        prev_ttype   = None

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

            # Reafference data (empty steps only)
            if (z_world_prev is not None and z_self_prev is not None and
                    a_prev is not None and ttype == "none"):
                dz_world = z_world_curr - z_world_prev
                reaf_data.append((z_self_prev.cpu(), a_prev.cpu(), dz_world.cpu()))
                n_empty_steps += 1
                if len(reaf_data) > MAX_REAF_DATA:
                    reaf_data = reaf_data[-MAX_REAF_DATA:]

            # Event selectivity (use prev_ttype: transition that produced z_world_curr)
            if z_world_prev is not None and prev_ttype in dz_world_by_event:
                dz = float(torch.norm(z_world_curr - z_world_prev).item())
                if len(dz_world_by_event[prev_ttype]) < MAX_DZ_PER_TYPE:
                    dz_world_by_event[prev_ttype].append(dz)

            # Track E2_world MSE diagnostics (1-step prediction quality)
            if (z_world_prev is not None and a_prev is not None and
                    len(e2w_1step_mses) < MAX_MSE_TRACK):
                with torch.no_grad():
                    z_pred = agent.e2.world_forward(z_world_prev, a_prev)
                    mse_pred = float(F.mse_loss(z_pred, z_world_curr).item())
                    mse_identity = float(F.mse_loss(z_world_prev, z_world_curr).item())
                    e2w_1step_mses.append(mse_pred)
                    identity_mses.append(mse_identity)

            # Signal buffer for net_eval
            if harm_signal != 0.0 or (step % 3 == 0):
                signal_buffer.append((latent.z_world.detach(), float(harm_signal)))
                if len(signal_buffer) > MAX_SIGNAL_BUFFER:
                    signal_buffer = signal_buffer[-MAX_SIGNAL_BUFFER:]

            # --- Loss computation ---
            e1_loss      = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()

            # Rollout loss (5-step, auxiliary)
            e2w_rollout_loss = _compute_e2w_rollout_loss(
                agent, traj_buffer, E2_ROLLOUT_STEPS
            )

            # 1-step direct E2_world loss (KEY FIX vs EXQ-023)
            if z_world_prev is not None and a_prev is not None:
                z_world_pred_1s = agent.e2.world_forward(z_world_prev, a_prev)
                e2w_1step_loss_val = F.mse_loss(z_world_pred_1s, z_world_curr.detach())
            else:
                e2w_1step_loss_val = next(agent.e1.parameters()).sum() * 0.0

            # Reconstruction loss
            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_w)
            recon_loss = F.mse_loss(recon, obs_w)

            total_loss = (
                e1_loss
                + e2_self_loss
                + E2W_ROLLOUT_WEIGHT * e2w_rollout_loss
                + E2W_1STEP_WEIGHT  * e2w_1step_loss_val
                + RECON_WEIGHT      * recon_loss
            )
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_decoder.parameters(), 1.0)
                optimizer.step()

            # net_eval head regression
            if len(signal_buffer) >= 8:
                k = min(32, len(signal_buffer))
                idxs_sel = torch.randperm(len(signal_buffer))[:k].tolist()
                zw_b = torch.cat([signal_buffer[i][0] for i in idxs_sel], dim=0)
                sv_b = torch.tensor(
                    [signal_buffer[i][1] for i in idxs_sel], device=agent.device
                ).unsqueeze(1)
                pred = net_eval_head(zw_b)
                sv_norm = (sv_b / harm_norm_scale).clamp(-1.0, 1.0)
                net_loss = F.mse_loss(pred, sv_norm)
                if net_loss.requires_grad:
                    net_eval_optimizer.zero_grad()
                    net_loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_eval_head.parameters(), 0.5)
                    net_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = z_self_curr
            a_prev       = action.detach()
            prev_ttype   = ttype

            if done:
                break

        for start in range(0, len(episode_traj) - E2_ROLLOUT_STEPS):
            traj_buffer.append(episode_traj[start:start + E2_ROLLOUT_STEPS + 1])
        if len(traj_buffer) > MAX_TRAJ_BUFFER:
            traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            mean_e2w = sum(e2w_1step_mses[-200:]) / max(1, len(e2w_1step_mses[-200:]))
            mean_id  = sum(identity_mses[-200:])  / max(1, len(identity_mses[-200:]))
            ratio    = mean_id / max(mean_e2w, 1e-10)
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  "
                  f"e2w_mse={mean_e2w:.5f}  identity_mse={mean_id:.5f}  ratio={ratio:.2f}",
                  flush=True)

    mean_dz = {k: float(sum(v) / max(1, len(v))) for k, v in dz_world_by_event.items()}
    mean_e2w_mse = float(sum(e2w_1step_mses) / max(1, len(e2w_1step_mses)))
    mean_id_mse  = float(sum(identity_mses)  / max(1, len(identity_mses)))
    e2w_improvement_ratio = mean_id_mse / max(mean_e2w_mse, 1e-10)

    return {
        "total_harm":    total_harm,
        "total_benefit": total_benefit,
        "n_empty_steps": n_empty_steps,
        "reaf_data":     reaf_data,
        "mean_dz_world_by_event": mean_dz,
        "n_dz_by_event": {k: len(v) for k, v in dz_world_by_event.items()},
        "mean_e2w_1step_mse":       mean_e2w_mse,
        "mean_identity_mse":        mean_id_mse,
        "e2w_improvement_ratio":    e2w_improvement_ratio,
        "signal_buffer":            signal_buffer,
    }


def _fit_lstsq_predictor(reaf_data, self_dim, action_dim, world_dim):
    if len(reaf_data) < 20:
        return None, 0.0, 0.0
    n = len(reaf_data)
    n_train = int(n * 0.8)
    z_self_all = torch.cat([d[0] for d in reaf_data], dim=0)
    a_all      = torch.cat([d[1] for d in reaf_data], dim=0)
    dz_all     = torch.cat([d[2] for d in reaf_data], dim=0)
    ones_all   = torch.ones(n, 1)
    X_all = torch.cat([z_self_all, a_all, ones_all], dim=-1)

    with torch.no_grad():
        result = torch.linalg.lstsq(X_all[:n_train], dz_all[:n_train], driver="gelsd")
        W = result.solution

    def _r2(X, Y):
        with torch.no_grad():
            pred = X @ W
            ss_res = ((Y - pred) ** 2).sum()
            ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum()
            return float((1 - ss_res / (ss_tot + 1e-8)).item())

    r2_train = _r2(X_all[:min(256, n_train)], dz_all[:min(256, n_train)])
    r2_test  = _r2(X_all[n_train:], dz_all[n_train:])
    print(f"  lstsq: n_train={n_train}  n_test={n-n_train}  "
          f"R²_train={r2_train:.3f}  R²_test={r2_test:.3f}", flush=True)
    return W, r2_train, r2_test


def _apply_lstsq_correction(z_world_raw, z_self, a, W, device):
    batch = z_self.shape[0]
    ones  = torch.ones(batch, 1, device=device)
    feat  = torch.cat([z_self, a, ones], dim=-1)
    return z_world_raw - feat @ W.to(device)


def _eval_precision_dynamics(
    agent: REEAgent,
    env: CausalGridWorld,
    steps_per_episode: int,
    n_episodes_stable: int = 50,
    n_episodes_perturbed: int = 50,
    drift_prob_stable: float = 0.1,
    drift_prob_perturbed: float = 0.9,
) -> Dict:
    agent.eval()
    device = agent.device

    def _run_env(drift_prob, n_eps, label):
        env.env_drift_prob = drift_prob
        errors: List[float] = []
        for _ in range(n_eps):
            flat_obs, obs_dict = env.reset()
            agent.reset()
            z_world_prev = None
            for step in range(steps_per_episode):
                with torch.no_grad():
                    latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                    agent.clock.advance()
                    z_world_curr = latent.z_world.detach()
                    action_idx = random.randint(0, env.action_dim - 1)
                    action = _action_to_onehot(action_idx, env.action_dim, device)
                    agent._last_action = action

                flat_obs, _, done, _, obs_dict = env.step(action)

                if z_world_prev is not None:
                    with torch.no_grad():
                        z_pred = agent.e2.world_forward(z_world_prev, action)
                        err = float(F.mse_loss(z_pred, z_world_curr).item())
                        errors.append(err)

                z_world_prev = z_world_curr
                if done:
                    break

        mean_err = float(sum(errors) / max(1, len(errors)))
        var_err  = float(sum((e - mean_err) ** 2 for e in errors) / max(1, len(errors)))
        precision = 1.0 / (var_err + 1e-6)
        print(f"  [{label}] drift_prob={drift_prob}  mean_pred_err={mean_err:.5f}  "
              f"variance={var_err:.5f}  precision={precision:.2f}  n_steps={len(errors)}",
              flush=True)
        return {
            "mean_pred_err": mean_err,
            "variance":      var_err,
            "precision":     precision,
            "n_steps":       len(errors),
        }

    stable    = _run_env(drift_prob_stable,    n_episodes_stable,    "stable")
    perturbed = _run_env(drift_prob_perturbed, n_episodes_perturbed, "perturbed")

    var_diff = abs(stable["variance"] - perturbed["variance"])
    print(f"  var_diff={var_diff:.5f}  "
          f"(stable={stable['variance']:.5f}  perturbed={perturbed['variance']:.5f})", flush=True)

    return {
        "variance_stable":       stable["variance"],
        "variance_perturbed":    perturbed["variance"],
        "precision_stable":      stable["precision"],
        "precision_perturbed":   perturbed["precision"],
        "mean_pred_err_stable":  stable["mean_pred_err"],
        "mean_pred_err_perturbed": perturbed["mean_pred_err"],
        "var_diff":              var_diff,
    }


def _eval_sd003_probe(agent, net_eval_head, W, probe_env, num_resets, min_dist_safe=2):
    """
    SD-003 probe using probe_env (sparse hazards) with min_dist_safe > 2.
    Key fix vs EXQ-023:
      - probe_env has num_hazards=6 (not 15) → meaningful near vs safe distinction
      - min_dist_safe=2 (not 3) → more safe cells available
    """
    agent.eval()
    net_eval_head.eval()
    near_sigs, safe_sigs, all_pred_vals = [], [], []
    fatal_errors = 0
    wall_type   = probe_env.ENTITY_TYPES["wall"]
    hazard_type = probe_env.ENTITY_TYPES["hazard"]
    device = agent.device
    use_lstsq = (W is not None)

    def _probe(ax, ay, actual_idx):
        probe_env.agent_x = ax
        probe_env.agent_y = ay
        obs_dict = probe_env._get_observation_dict()
        with torch.no_grad():
            latent  = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world = latent.z_world
            z_self  = latent.z_self
            a_act   = _action_to_onehot(actual_idx, probe_env.action_dim, device)
            cf_idx  = _random_cf_action(actual_idx, probe_env.action_dim)
            a_cf    = _action_to_onehot(cf_idx, probe_env.action_dim, device)
            if use_lstsq:
                zw_a = _apply_lstsq_correction(z_world, z_self, a_act, W, device)
                zw_c = _apply_lstsq_correction(z_world, z_self, a_cf,  W, device)
            else:
                zw_a = z_world
                zw_c = z_world
            v_act = net_eval_head(agent.e2.world_forward(zw_a, a_act))
            v_cf  = net_eval_head(agent.e2.world_forward(zw_c, a_cf))
            all_pred_vals.extend([float(v_act.item()), float(v_cf.item())])
            return float((v_act - v_cf).item())

    try:
        for _ in range(num_resets):
            probe_env.reset()
            for hx, hy in probe_env.hazards:
                for action_idx, (dx, dy) in probe_env.ACTIONS.items():
                    if action_idx == 4:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < probe_env.size and 0 <= ay < probe_env.size:
                        cell = int(probe_env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            near_sigs.append(_probe(ax, ay, action_idx))
            for px in range(probe_env.size):
                for py in range(probe_env.size):
                    if int(probe_env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(
                        abs(px - hx) + abs(py - hy) for hx, hy in probe_env.hazards
                    )
                    if min_dist > min_dist_safe:
                        safe_sigs.append(_probe(px, py, random.randint(0, 3)))
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL probe: {traceback.format_exc()}", flush=True)

    mean_near = float(sum(near_sigs) / max(1, len(near_sigs)))
    mean_safe = float(sum(safe_sigs) / max(1, len(safe_sigs)))
    gap = mean_near - mean_safe
    pred_std = float(torch.tensor(all_pred_vals).std().item()) if len(all_pred_vals) > 1 else 0.0

    print(f"  SD-003 probe  n_near={len(near_sigs)} n_safe={len(safe_sigs)}  "
          f"near={mean_near:.4f}  safe={mean_safe:.4f}  gap={gap:.4f}  "
          f"pred_std={pred_std:.4f}  (min_dist_safe={min_dist_safe})", flush=True)

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
    warmup_episodes: int = 1000,
    eval_probe_resets: int = 10,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    probe_num_hazards: int = 6,
    probe_min_dist_safe: int = 2,
    train_hazard_harm: float = 0.02,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    # Training env: dense hazards for event data, low harm values so episodes are long.
    # EXQ-023 defaults: hazard_harm=0.5, contaminated_harm=0.4 → avg episode = 18 steps.
    # Root cause: contaminated cells (agent's own trail) also deplete health at 0.4/step.
    # With both=0.02: avg episode ≈ 182 steps, 46% hit the 200-step cap.
    # 1000 eps × ~182 steps = ~182,000 training steps (vs 5,500 in EXQ-023).
    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
        hazard_harm=train_hazard_harm,
        contaminated_harm=train_hazard_harm,
    )
    # Probe env: sparse hazards for meaningful near/safe distinction (KEY FIX)
    probe_env = CausalGridWorld(
        seed=seed + 100, size=12, num_hazards=probe_num_hazards, num_resources=5,
        hazard_harm=train_hazard_harm,
        contaminated_harm=train_hazard_harm,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    agent = REEAgent(config)
    world_decoder = _make_world_decoder(world_dim, env.world_obs_dim)
    net_eval_head = _make_net_eval_head(world_dim)

    e12_params  = [p for n_, p in agent.named_parameters() if "harm_eval" not in n_]
    e12_params += list(world_decoder.parameters())
    optimizer      = optim.Adam(e12_params, lr=lr)
    net_eval_optim = optim.Adam(net_eval_head.parameters(), lr=1e-4)

    print(f"[V3-EXQ-024] E2_world 1-step loss + fixed probe", flush=True)
    print(f"  alpha_world={alpha_world}  alpha_self={alpha_self}", flush=True)
    print(f"  Training env: 15 hazards, 12×12, hazard_harm={train_hazard_harm}", flush=True)
    print(f"  Probe env: {probe_num_hazards} hazards, min_dist > {probe_min_dist_safe}", flush=True)
    print(f"  warmup_episodes={warmup_episodes}  (EXQ-023 had 300 but avg ep=18 steps → only ~5500 training steps)", flush=True)
    print(f"  E2W_1STEP_WEIGHT={E2W_1STEP_WEIGHT}  E2W_ROLLOUT_WEIGHT={E2W_ROLLOUT_WEIGHT}",
          flush=True)

    train_out = _train_and_collect(
        agent, env, world_decoder, net_eval_head,
        optimizer, net_eval_optim,
        warmup_episodes, steps_per_episode,
        harm_norm_scale=train_hazard_harm,
    )
    warmup_harm    = train_out["total_harm"]
    warmup_benefit = train_out["total_benefit"]
    mean_dz        = train_out["mean_dz_world_by_event"]

    selectivity_margin = mean_dz["env_caused_hazard"] - mean_dz["none"]
    e2w_improvement_ratio = train_out["e2w_improvement_ratio"]

    print(f"  Warmup done. harm={warmup_harm}  benefit={warmup_benefit}", flush=True)
    print(f"  Δz_world — none={mean_dz['none']:.4f}  "
          f"env_hazard={mean_dz['env_caused_hazard']:.4f}  "
          f"selectivity={selectivity_margin:.4f}", flush=True)
    print(f"  E2_world: 1-step MSE={train_out['mean_e2w_1step_mse']:.5f}  "
          f"identity MSE={train_out['mean_identity_mse']:.5f}  "
          f"improvement_ratio={e2w_improvement_ratio:.2f}x", flush=True)

    # lstsq reafference predictor
    print(f"[V3-EXQ-024] Fitting lstsq predictor on {len(train_out['reaf_data'])} steps...",
          flush=True)
    W, r2_train, r2_test = _fit_lstsq_predictor(
        train_out["reaf_data"], self_dim, env.action_dim, world_dim
    )

    # ARC-016 precision dynamics (stable vs perturbed)
    print(f"[V3-EXQ-024] ARC-016 precision dynamics...", flush=True)
    env.env_drift_interval = 10
    prec_stats = _eval_precision_dynamics(
        agent, env, steps_per_episode,
        n_episodes_stable=50, n_episodes_perturbed=50,
        drift_prob_stable=0.1, drift_prob_perturbed=0.9,
    )
    env.env_drift_interval = 3
    env.env_drift_prob = 0.5

    # SD-003 probe (sparse probe_env, min_dist > 2)
    print(f"[V3-EXQ-024] SD-003 probe ({eval_probe_resets} resets, probe_env={probe_num_hazards} hazards)...",
          flush=True)
    probe = _eval_sd003_probe(
        agent, net_eval_head, W, probe_env,
        eval_probe_resets, min_dist_safe=probe_min_dist_safe
    )
    fatal_errors = probe["fatal_errors"]

    # PASS / FAIL
    c1_pass = selectivity_margin > 0.005
    c2_pass = e2w_improvement_ratio > 2.0
    c3_pass = prec_stats["var_diff"] > 0.001
    c4_pass = probe["calibration_gap"] > 0.05
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: event_selectivity_margin={selectivity_margin:.4f} <= 0.005"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: e2w_improvement_ratio={e2w_improvement_ratio:.2f}x <= 2.0 "
            f"(1-step MSE={train_out['mean_e2w_1step_mse']:.5f}, "
            f"identity={train_out['mean_identity_mse']:.5f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: var_diff={prec_stats['var_diff']:.5f} <= 0.001"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: calibration_gap={probe['calibration_gap']:.4f} <= 0.05"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-024 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count":             float(fatal_errors),
        "alpha_world":                   float(alpha_world),
        "alpha_self":                    float(alpha_self),
        "warmup_harm_events":            float(warmup_harm),
        "warmup_benefit_events":         float(warmup_benefit),
        "n_empty_steps_collected":       float(train_out["n_empty_steps"]),
        "event_selectivity_margin":      float(selectivity_margin),
        "mean_dz_world_none":            float(mean_dz["none"]),
        "mean_dz_world_env_hazard":      float(mean_dz["env_caused_hazard"]),
        "mean_dz_world_agent_hazard":    float(mean_dz["agent_caused_hazard"]),
        "e2w_1step_mse":                 float(train_out["mean_e2w_1step_mse"]),
        "identity_baseline_mse":         float(train_out["mean_identity_mse"]),
        "e2w_improvement_ratio":         float(e2w_improvement_ratio),
        "reafference_r2_train":          float(r2_train),
        "reafference_r2_test":           float(r2_test if r2_test is not None else 0.0),
        "variance_stable":               float(prec_stats["variance_stable"]),
        "variance_perturbed":            float(prec_stats["variance_perturbed"]),
        "precision_stable":              float(prec_stats["precision_stable"]),
        "precision_perturbed":           float(prec_stats["precision_perturbed"]),
        "mean_pred_err_stable":          float(prec_stats["mean_pred_err_stable"]),
        "mean_pred_err_perturbed":       float(prec_stats["mean_pred_err_perturbed"]),
        "var_diff":                      float(prec_stats["var_diff"]),
        "calibration_gap":               float(probe["calibration_gap"]),
        "mean_causal_sig_near_hazard":   float(probe["mean_causal_sig_near"]),
        "mean_causal_sig_safe":          float(probe["mean_causal_sig_safe"]),
        "n_near_hazard_probes":          float(probe["n_near_hazard_probes"]),
        "n_safe_probes":                 float(probe["n_safe_probes"]),
        "net_eval_pred_std":             float(probe["net_eval_pred_std"]),
        "probe_num_hazards":             float(probe_num_hazards),
        "probe_min_dist_safe":           float(probe_min_dist_safe),
        "train_hazard_harm":             float(train_hazard_harm),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-024 — E2_world 1-Step Loss + Fixed SD-003 Probe

**Status:** {status}
**alpha_world:** {alpha_world}  (SD-008 fix, same as EXQ-023)
**E2W_1STEP_WEIGHT:** {E2W_1STEP_WEIGHT}  (KEY FIX: direct 1-step loss added)
**E2W_ROLLOUT_WEIGHT:** {E2W_ROLLOUT_WEIGHT}  (5-step rollout kept as auxiliary)
**Probe env:** {probe_num_hazards} hazards, min_dist > {probe_min_dist_safe}  (FIX: was 15 hazards, min_dist > 3)
**Seed:** {seed}

## Motivation (from EXQ-023 diagnosis)

Two problems diagnosed from EXQ-023:

1. **E2_world worse than identity baseline**: With 5-step rollout on random policy,
   ~50% wall collisions produce unpredictable 5-step outcomes.
   EXQ-023 identity_MSE = 0.000162; reported E2_world MSE = 0.0006–0.0008 (4–7× worse).
   Fix: add 1-step direct loss `MSE(E2.world_forward(z_world_t, a_t), z_world_t+1)`.

2. **Probe geometry broken at 15 hazards**: 15 × ~25 cells exclusion = ~375 on 144 grid.
   Only 27–59 safe cells — measuring near-hazard vs near-hazard.
   Fix: probe_env with {probe_num_hazards} hazards + min_dist > {probe_min_dist_safe}.

## E2_world Quality

| Metric | Value |
|---|---|
| 1-step MSE | {train_out['mean_e2w_1step_mse']:.5f} |
| Identity baseline MSE | {train_out['mean_identity_mse']:.5f} |
| Improvement ratio | {e2w_improvement_ratio:.2f}× |

## Event Selectivity

| Event Type | mean Δz_world |
|---|---|
| none (locomotion) | {mean_dz['none']:.4f} |
| env_caused_hazard | {mean_dz['env_caused_hazard']:.4f} |
| agent_caused_hazard | {mean_dz['agent_caused_hazard']:.4f} |
| **selectivity margin** | **{selectivity_margin:.4f}** |

## ARC-016 Precision Dynamics

| Condition | mean_pred_err | variance | precision |
|---|---|---|---|
| stable (drift=0.1) | {prec_stats['mean_pred_err_stable']:.5f} | {prec_stats['variance_stable']:.5f} | {prec_stats['precision_stable']:.2f} |
| perturbed (drift=0.9) | {prec_stats['mean_pred_err_perturbed']:.5f} | {prec_stats['variance_perturbed']:.5f} | {prec_stats['precision_perturbed']:.2f} |
| **var_diff** | — | **{prec_stats['var_diff']:.5f}** | — |

## SD-003 Attribution

| Position | mean(causal_sig) |
|---|---|
| Near-hazard | {probe['mean_causal_sig_near']:.4f} |
| Safe (min_dist > {probe_min_dist_safe}) | {probe['mean_causal_sig_safe']:.4f} |
| **calibration_gap** | **{probe['calibration_gap']:.4f}** |

n_near={probe['n_near_hazard_probes']}  n_safe={probe['n_safe_probes']}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: event_selectivity_margin > 0.005 | {"PASS" if c1_pass else "FAIL"} | {selectivity_margin:.4f} |
| C2: e2w_improvement_ratio > 2.0× | {"PASS" if c2_pass else "FAIL"} | {e2w_improvement_ratio:.2f}× |
| C3: var_diff > 0.001 (ARC-016) | {"PASS" if c3_pass else "FAIL"} | {prec_stats['var_diff']:.5f} |
| C4: calibration_gap > 0.05 (SD-003) | {"PASS" if c4_pass else "FAIL"} | {probe['calibration_gap']:.4f} |
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
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--warmup",             type=int,   default=300)
    parser.add_argument("--probe-resets",       type=int,   default=10)
    parser.add_argument("--steps",              type=int,   default=200)
    parser.add_argument("--alpha-world",        type=float, default=0.9)
    parser.add_argument("--alpha-self",         type=float, default=0.3)
    parser.add_argument("--probe-hazards",      type=int,   default=6)
    parser.add_argument("--probe-min-dist",     type=int,   default=2)
    parser.add_argument("--train-hazard-harm",  type=float, default=0.02)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        probe_num_hazards=args.probe_hazards,
        probe_min_dist_safe=args.probe_min_dist,
        train_hazard_harm=args.train_hazard_harm,
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
