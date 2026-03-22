"""
V3-EXQ-023 -- SD-008 alpha_world Fix (alpha_world=0.9 vs 0.3 baseline)

Claims: SD-008 (encoder.z_world_alpha_correction), SD-003, MECH-098, ARC-016.

Motivation (2026-03-18):
  The entire EXQ-013-019 FAIL cluster shares a single root cause: alpha=0.3 in
  LatentStack.encode() means z_world = 0.3*z_new + 0.7*z_prev -- a ~3-step EMA
  that suppresses event responses to 30% per step, making z_world nearly constant.

  Consequences:
    - Event selectivity ≈ 0 (EXQ-013): dz_world barely differs by event type
    - E2_world trivially accurate (MSE ≈ 0.005 regardless of env drift) -- ARC-016
      precision stuck at ~188, commit_rate = 1.0 always (EXQ-018)
    - Reafference predictor R2=0.118 instead of 0.333 because the multi-step
      blended dz_world signal can't be predicted from single-step (z_self, a)
    - z_self more autocorrelated than z_world (EXQ-019): backwards from MECH-058

  The fix (SD-008): set alpha_world >= 0.9 (or 1.0 = no EMA for z_world).
  MECH-089 theta buffer already handles temporal integration; encoder EMA is
  redundant double-smoothing. alpha_self can remain at 0.3 (body state is
  genuinely highly autocorrelated).

  This experiment runs a comprehensive joint test:
    Phase 1: EXQ-013-style event selectivity (measures dz_world by event type)
    Phase 2: lstsq reafference fit (should now reach R2 ≈ 0.333)
    Phase 3: ARC-016-style precision dynamics (stable vs perturbed env)
    Phase 4: SD-003 attribution probe (calibration_gap)
    Phase 5: Timescale check (z_world more persistent than z_self -- MECH-058)

  A second condition (alpha_world=1.0 = pure current encoding, no EMA) is also
  tested if --alpha-world=1.0 is passed. Default is 0.9.

PASS criteria (ALL must hold):
  C1: event_selectivity_margin > 0.005 (dz_world(env_hazard) > dz_world(empty))
  C2: reafference_r2_test > 0.25 (lstsq predictor recovers EXQ-014 benchmark)
  C3: var_diff > 0.001 (running_variance responds to env stability change)
  C4: calibration_gap > 0.05 (SD-003 threshold, with lstsq-corrected z_world)
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


EXPERIMENT_TYPE = "v3_exq_023_sd008_alpha_world"
CLAIM_IDS = ["SD-008", "SD-003", "MECH-098", "ARC-016"]

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
    """Standard training + collect multi-purpose data (reaf + event selectivity + precision)."""
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

            # Reafference data (empty steps)
            if (z_world_prev is not None and z_self_prev is not None and
                    a_prev is not None and ttype == "none"):
                dz_world = z_world_curr - z_world_prev
                reaf_data.append((z_self_prev.cpu(), a_prev.cpu(), dz_world.cpu()))
                n_empty_steps += 1
                if len(reaf_data) > MAX_REAF_DATA:
                    reaf_data = reaf_data[-MAX_REAF_DATA:]

            # Event selectivity: use prev_ttype (transition that produced z_world_curr)
            if z_world_prev is not None and prev_ttype in dz_world_by_event:
                dz = float(torch.norm(z_world_curr - z_world_prev).item())
                if len(dz_world_by_event[prev_ttype]) < MAX_DZ_PER_TYPE:
                    dz_world_by_event[prev_ttype].append(dz)

            z_world_prev = z_world_curr
            z_self_prev  = z_self_curr
            a_prev       = action.detach()
            prev_ttype   = ttype

            # Collect signal for net_eval
            if harm_signal != 0.0 or (step % 3 == 0):
                signal_buffer.append((latent.z_world.detach(), float(harm_signal)))
                if len(signal_buffer) > MAX_SIGNAL_BUFFER:
                    signal_buffer = signal_buffer[-MAX_SIGNAL_BUFFER:]

            # Standard losses
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
                  f"empty_steps={n_empty_steps}", flush=True)

    mean_dz = {k: float(sum(v) / max(1, len(v))) for k, v in dz_world_by_event.items()}

    return {
        "total_harm":    total_harm,
        "total_benefit": total_benefit,
        "n_empty_steps": n_empty_steps,
        "reaf_data":     reaf_data,
        "mean_dz_world_by_event": mean_dz,
        "n_dz_by_event": {k: len(v) for k, v in dz_world_by_event.items()},
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
          f"R2_train={r2_train:.3f}  R2_test={r2_test:.3f}", flush=True)
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
    """
    ARC-016-style precision measurement (EXQ-018 methodology).
    Measures running_variance of E2_world prediction errors under stable vs perturbed envs.
    With alpha_world=0.9, z_world should track env changes -> prediction error rises
    in perturbed env -> running_variance rises -> var_diff > 0.
    """
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
        "variance_stable":    stable["variance"],
        "variance_perturbed": perturbed["variance"],
        "precision_stable":   stable["precision"],
        "precision_perturbed": perturbed["precision"],
        "mean_pred_err_stable":    stable["mean_pred_err"],
        "mean_pred_err_perturbed": perturbed["mean_pred_err"],
        "var_diff":           var_diff,
    }


def _eval_sd003_probe(agent, net_eval_head, W, env, num_resets):
    agent.eval()
    net_eval_head.eval()
    near_sigs, safe_sigs, all_pred_vals = [], [], []
    fatal_errors = 0
    wall_type   = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]
    device = agent.device
    use_lstsq = (W is not None)

    def _probe(ax, ay, actual_idx):
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            latent  = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world = latent.z_world
            z_self  = latent.z_self
            a_act   = _action_to_onehot(actual_idx, env.action_dim, device)
            cf_idx  = _random_cf_action(actual_idx, env.action_dim)
            a_cf    = _action_to_onehot(cf_idx, env.action_dim, device)
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
    mean_safe = float(sum(safe_sigs) / max(1, len(safe_sigs)))
    gap = mean_near - mean_safe
    pred_std = float(torch.tensor(all_pred_vals).std().item()) if len(all_pred_vals) > 1 else 0.0

    print(f"  SD-003 probe  n_near={len(near_sigs)} n_safe={len(safe_sigs)}  "
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
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
    )
    # SD-008: use configurable alpha_world (default 0.9 for this experiment)
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

    print(f"[V3-EXQ-023] SD-008 test: alpha_world={alpha_world}  alpha_self={alpha_self}",
          flush=True)
    print(f"  Warmup: {warmup_episodes} eps (12x12, 15 hazards, drift)", flush=True)

    train_out = _train_and_collect(
        agent, env, world_decoder, net_eval_head,
        optimizer, net_eval_optim,
        warmup_episodes, steps_per_episode,
    )
    warmup_harm    = train_out["total_harm"]
    warmup_benefit = train_out["total_benefit"]
    mean_dz        = train_out["mean_dz_world_by_event"]

    selectivity_margin = mean_dz["env_caused_hazard"] - mean_dz["none"]
    print(f"  Warmup done. harm={warmup_harm}  benefit={warmup_benefit}", flush=True)
    print(f"  dz_world -- none={mean_dz['none']:.4f}  "
          f"env_hazard={mean_dz['env_caused_hazard']:.4f}  "
          f"selectivity_margin={selectivity_margin:.4f}", flush=True)

    # lstsq reafference predictor
    print(f"[V3-EXQ-023] Fitting lstsq predictor on {len(train_out['reaf_data'])} steps...",
          flush=True)
    W, r2_train, r2_test = _fit_lstsq_predictor(
        train_out["reaf_data"], self_dim, env.action_dim, world_dim
    )
    reafference_r2 = max(0.0, r2_test if r2_test is not None else 0.0)

    # ARC-016 precision dynamics (stable vs perturbed)
    print(f"[V3-EXQ-023] ARC-016 precision dynamics (50 stable + 50 perturbed eps)...",
          flush=True)
    # Restore normal drift for precision test
    env.env_drift_interval = 10
    prec_stats = _eval_precision_dynamics(
        agent, env, steps_per_episode,
        n_episodes_stable=50, n_episodes_perturbed=50,
        drift_prob_stable=0.1, drift_prob_perturbed=0.9,
    )
    # Restore training drift params
    env.env_drift_interval = 3
    env.env_drift_prob = 0.5

    # SD-003 probe (with lstsq correction if available)
    print(f"[V3-EXQ-023] SD-003 probe ({eval_probe_resets} resets)...", flush=True)
    probe = _eval_sd003_probe(agent, net_eval_head, W, env, eval_probe_resets)
    fatal_errors = probe["fatal_errors"]

    # PASS / FAIL
    c1_pass = selectivity_margin > 0.005
    c2_pass = reafference_r2 > 0.25
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
            f"C2 FAIL: reafference_r2={reafference_r2:.3f} <= 0.25 "
            f"(EXQ-014 lstsq benchmark = 0.333)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: var_diff={prec_stats['var_diff']:.5f} <= 0.001 "
            f"(stable={prec_stats['variance_stable']:.5f}  "
            f"perturbed={prec_stats['variance_perturbed']:.5f})"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: calibration_gap={probe['calibration_gap']:.4f} <= 0.05"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-023 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count":               float(fatal_errors),
        "alpha_world":                     float(alpha_world),
        "alpha_self":                      float(alpha_self),
        "warmup_harm_events":              float(warmup_harm),
        "warmup_benefit_events":           float(warmup_benefit),
        "n_empty_steps_collected":         float(train_out["n_empty_steps"]),
        "event_selectivity_margin":        float(selectivity_margin),
        "mean_dz_world_none":              float(mean_dz["none"]),
        "mean_dz_world_env_hazard":        float(mean_dz["env_caused_hazard"]),
        "mean_dz_world_agent_hazard":      float(mean_dz["agent_caused_hazard"]),
        "reafference_reconstruction_r2":   float(reafference_r2),
        "reafference_r2_train":            float(r2_train),
        "variance_stable":                 float(prec_stats["variance_stable"]),
        "variance_perturbed":              float(prec_stats["variance_perturbed"]),
        "precision_stable":                float(prec_stats["precision_stable"]),
        "precision_perturbed":             float(prec_stats["precision_perturbed"]),
        "mean_pred_err_stable":            float(prec_stats["mean_pred_err_stable"]),
        "mean_pred_err_perturbed":         float(prec_stats["mean_pred_err_perturbed"]),
        "var_diff":                        float(prec_stats["var_diff"]),
        "calibration_gap":                 float(probe["calibration_gap"]),
        "mean_causal_sig_near_hazard":     float(probe["mean_causal_sig_near"]),
        "mean_causal_sig_safe":            float(probe["mean_causal_sig_safe"]),
        "n_near_hazard_probes":            float(probe["n_near_hazard_probes"]),
        "n_safe_probes":                   float(probe["n_safe_probes"]),
        "net_eval_pred_std":               float(probe["net_eval_pred_std"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-023 -- SD-008 alpha_world Fix Test

**Status:** {status}
**alpha_world:** {alpha_world}  (baseline was 0.3 -- root cause of EXQ-013-019 FAIL cluster)
**alpha_self:** {alpha_self}
**Warmup:** {warmup_episodes} eps (RANDOM policy, 12x12, 15 hazards, drift_interval=3, drift_prob=0.5)
**Probe eval:** {eval_probe_resets} grid resets
**Seed:** {seed}

## Motivation (SD-008)

LatentStack.encode() used alpha=0.3 for z_world -> ~3-step EMA -> event responses
suppressed to 30% per step. Root cause of all EXQ-013-019 FAILs:
- EXQ-013: dz_world barely differs by event type (selectivity ≈ 0)
- EXQ-014/016/021: reafference R2=0.118 (SGD) instead of 0.333 (lstsq)
- EXQ-018: ARC-016 precision ≈ 188 always (E2 trivially accurate)
- EXQ-019: z_self more autocorrelated than z_world (backwards from MECH-058)

This test uses alpha_world={alpha_world}: z_world tracks current observation much more
directly. MECH-089 theta buffer handles temporal integration; encoder EMA was redundant.

## Event Selectivity (SD-008 diagnostic)

| Event Type | mean dz_world |
|---|---|
| none (locomotion) | {mean_dz['none']:.4f} |
| env_caused_hazard | {mean_dz['env_caused_hazard']:.4f} |
| agent_caused_hazard | {mean_dz['agent_caused_hazard']:.4f} |
| **selectivity margin** (env - none) | **{selectivity_margin:.4f}** |

## lstsq Reafference (MECH-098)

| Metric | Value |
|---|---|
| n_empty_steps | {train_out['n_empty_steps']} |
| R2_train | {r2_train:.3f} |
| R2_test | {reafference_r2:.3f} |

## ARC-016 Precision Dynamics

| Condition | mean_pred_err | variance | precision |
|---|---|---|---|
| stable (drift_prob=0.1) | {prec_stats['mean_pred_err_stable']:.5f} | {prec_stats['variance_stable']:.5f} | {prec_stats['precision_stable']:.2f} |
| perturbed (drift_prob=0.9) | {prec_stats['mean_pred_err_perturbed']:.5f} | {prec_stats['variance_perturbed']:.5f} | {prec_stats['precision_perturbed']:.2f} |
| **var_diff** | -- | **{prec_stats['var_diff']:.5f}** | -- |

## SD-003 Attribution

| Position | mean(causal_sig) |
|---|---|
| Near-hazard | {probe['mean_causal_sig_near']:.4f} |
| Safe | {probe['mean_causal_sig_safe']:.4f} |
| **calibration_gap** | **{probe['calibration_gap']:.4f}** |

Warmup: harm={warmup_harm}  benefit={warmup_benefit}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: event_selectivity_margin > 0.005 | {"PASS" if c1_pass else "FAIL"} | {selectivity_margin:.4f} |
| C2: R2_test > 0.25 (lstsq reafference) | {"PASS" if c2_pass else "FAIL"} | {reafference_r2:.3f} |
| C3: var_diff > 0.001 (ARC-016 fires) | {"PASS" if c3_pass else "FAIL"} | {prec_stats['var_diff']:.5f} |
| C4: calibration_gap > 0.05 (SD-003) | {"PASS" if c4_pass else "FAIL"} | {probe['calibration_gap']:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {fatal_errors} |

Criteria met: {criteria_met}/5 -> **{status}**
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
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--warmup",       type=int,   default=300)
    parser.add_argument("--probe-resets", type=int,   default=10)
    parser.add_argument("--steps",        type=int,   default=200)
    parser.add_argument("--alpha-world",  type=float, default=0.9)
    parser.add_argument("--alpha-self",   type=float, default=0.3)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
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
