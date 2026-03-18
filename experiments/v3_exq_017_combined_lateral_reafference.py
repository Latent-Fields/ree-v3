"""
V3-EXQ-017 — Combined Lateral + Reafference (MECH-099 + MECH-098)

Claims: MECH-099, MECH-098, SD-007, SD-003.

Motivation (2026-03-17):
  If EXQ-015 (lateral head alone) and/or EXQ-016 (reafference alone) fail to
  produce calibration_gap > 0.05, this experiment combines both mechanisms, as
  specified in the architectural plan. The biological grounding (Haak & Beckmann
  2018): MSTd congruent/incongruent neurons (reafference cancellation) feed their
  output into the lateral stream (MT→FST→STS→TPJ, harm/agency detection). In REE
  terms: reafference correction removes perspective-shift noise from z_world, then
  the lateral head reads harm-salient channels into z_harm. Together:

  Combined attribution pipeline:
    z_w_corr_act = reaf_pred.correct_z_world(z_world_raw, z_self, a_act)
    z_w_corr_cf  = reaf_pred.correct_z_world(z_world_raw, z_self, a_cf)
    z_w_next_act = E2.world_forward(z_w_corr_act, a_act)
    z_w_next_cf  = E2.world_forward(z_w_corr_cf, a_cf)
    combined_act = combined_head([z_harm; z_w_next_act]) → scalar
    combined_cf  = combined_head([z_harm; z_w_next_cf])  → scalar
    causal_sig   = combined_act - combined_cf

  z_harm provides harm context (perspective-invariant; same for both actions but
  acts as nonlinear conditioning). z_world_corrected_next provides action-conditional
  world outcome. The combined head: "given this harm context, how much does my
  predicted world-state differ between action choices?"

Implementation:
  Phase 1 (warmup): train encoder + E2_world + E2_self + collect data
    - collect reafference data (empty-step z_self, a, Δz_world)
    - collect signal buffer (z_harm, z_world, z_self, action, harm_signal)
  Phase 2: train ReafferencePredictor on empty-step data (200 mini-batches)
  Phase 3: train combined_head on signal buffer using corrected z_world
  Phase 4: probe using combined attribution

PASS criteria (ALL must hold):
  C1: calibration_gap > 0.05 using combined attribution
  C2: z_harm selectivity margin > 0.01 (lateral head still responds to hazard proximity)
  C3: reafference_r2_test > 0.2 (predictor learned real locomotion-induced shift)
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


EXPERIMENT_TYPE = "v3_exq_017_combined_lateral_reafference"
CLAIM_IDS = ["MECH-099", "MECH-098", "SD-007", "SD-003"]

HARM_DIM = 16
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


def _make_combined_head(harm_dim: int, world_dim: int, hidden_dim: int = 64) -> nn.Module:
    """Combined net_eval head: [z_harm; z_world_next] → scalar ∈ [-1, 1]."""
    return nn.Sequential(
        nn.Linear(harm_dim + world_dim, hidden_dim),
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
    optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Train encoder/E2 and collect data for reafference predictor + combined head."""
    agent.train()
    world_decoder.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    # Reafference data: (z_self_prev, a_prev, dz_world) for empty-space steps
    reaf_data: List[Tuple] = []
    MAX_REAF_DATA = 2000

    # Signal buffer: (z_harm, z_world, z_self, action, harm_signal) for combined head
    signal_buffer: List[Tuple] = []
    MAX_SIGNAL_BUFFER = 1000

    total_harm = 0
    total_benefit = 0
    n_empty_steps = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        episode_traj: List = []

        z_world_prev = None
        z_self_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_self_curr = latent.z_self.detach()
            z_harm_curr = latent.z_harm.detach() if latent.z_harm is not None else None

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
            if (z_world_prev is not None and z_self_prev is not None
                    and a_prev is not None and ttype == "none"):
                dz_world = (z_world_curr - z_world_prev)
                reaf_data.append((z_self_prev.cpu(), a_prev.cpu(), dz_world.cpu()))
                n_empty_steps += 1
                if len(reaf_data) > MAX_REAF_DATA:
                    reaf_data = reaf_data[-MAX_REAF_DATA:]

            z_world_prev = z_world_curr
            z_self_prev = z_self_curr
            a_prev = action.detach()

            # Collect signal buffer for combined head training
            if z_harm_curr is not None and (harm_signal != 0.0 or step % 3 == 0):
                signal_buffer.append((
                    z_harm_curr.cpu(),
                    z_world_curr.cpu(),
                    z_self_curr.cpu(),
                    action.detach().cpu(),
                    float(harm_signal),
                ))
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
        "signal_buffer": signal_buffer,
    }


def _train_reafference_predictor(
    reaf_data: List,
    self_dim: int,
    action_dim: int,
    world_dim: int,
) -> Tuple[ReafferencePredictor, float, float]:
    if len(reaf_data) < 20:
        print(f"  WARNING: only {len(reaf_data)} empty-step records", flush=True)
        return ReafferencePredictor(self_dim, action_dim, world_dim), 0.0, 0.0

    n = len(reaf_data)
    n_train = int(n * 0.8)

    z_self_all = torch.cat([d[0] for d in reaf_data], dim=0)
    a_all = torch.cat([d[1] for d in reaf_data], dim=0)
    dz_all = torch.cat([d[2] for d in reaf_data], dim=0)

    z_self_train, z_self_test = z_self_all[:n_train], z_self_all[n_train:]
    a_train, a_test = a_all[:n_train], a_all[n_train:]
    dz_train, dz_test = dz_all[:n_train], dz_all[n_train:]

    rp = ReafferencePredictor(self_dim, action_dim, world_dim)
    rp_opt = optim.Adam(rp.parameters(), lr=1e-3)

    BATCH = 32
    for _ in range(200):
        if n_train < BATCH:
            break
        idxs = torch.randperm(n_train)[:BATCH]
        pred = rp(z_self_train[idxs], a_train[idxs])
        loss = F.mse_loss(pred, dz_train[idxs])
        rp_opt.zero_grad()
        loss.backward()
        rp_opt.step()

    rp.eval()

    def _r2(zs, a, dz):
        with torch.no_grad():
            pred = rp(zs, a)
            ss_res = ((dz - pred) ** 2).sum()
            ss_tot = ((dz - dz.mean(dim=0, keepdim=True)) ** 2).sum()
            return float((1 - ss_res / (ss_tot + 1e-8)).item())

    r2_train = _r2(z_self_train[:256], a_train[:256], dz_train[:256])
    r2_test = _r2(z_self_test, a_test, dz_test)
    print(f"  ReafferencePredictor: n_train={n_train}  R²_train={r2_train:.3f}  "
          f"R²_test={r2_test:.3f}", flush=True)
    return rp, r2_train, r2_test


def _train_combined_head(
    combined_head: nn.Module,
    signal_buffer: List,
    reaf_predictor: ReafferencePredictor,
    agent: REEAgent,
    harm_dim: int,
    world_dim: int,
    steps: int = 300,
    harm_scale: float = 0.02,
) -> None:
    """Train combined_head on signal buffer using reafference-corrected z_world."""
    if len(signal_buffer) < 8:
        print("  WARNING: insufficient signal buffer for combined head training", flush=True)
        return

    combined_head.train()
    optimizer = optim.Adam(combined_head.parameters(), lr=1e-4)

    BATCH = 32
    for step_i in range(steps):
        n = len(signal_buffer)
        k = min(BATCH, n)
        idxs = torch.randperm(n)[:k].tolist()

        z_harm_list = [signal_buffer[i][0] for i in idxs]
        z_world_list = [signal_buffer[i][1] for i in idxs]
        z_self_list = [signal_buffer[i][2] for i in idxs]
        action_list = [signal_buffer[i][3] for i in idxs]
        harm_vals = [signal_buffer[i][4] for i in idxs]

        z_harm_b = torch.cat(z_harm_list, dim=0)    # [k, harm_dim]
        z_world_b = torch.cat(z_world_list, dim=0)  # [k, world_dim]
        z_self_b = torch.cat(z_self_list, dim=0)    # [k, self_dim]
        action_b = torch.cat(action_list, dim=0)    # [k, action_dim]
        sv_b = torch.tensor(harm_vals).unsqueeze(1) # [k, 1]

        with torch.no_grad():
            reaf_predictor.eval()
            z_world_corr = reaf_predictor.correct_z_world(z_world_b, z_self_b, action_b)
            z_world_next = agent.e2.world_forward(z_world_corr, action_b)

        combined_input = torch.cat([z_harm_b, z_world_next], dim=-1)  # [k, harm_dim+world_dim]
        pred = combined_head(combined_input)
        sv_norm = (sv_b / harm_scale).clamp(-1.0, 1.0)
        loss = F.mse_loss(pred, sv_norm)

        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(combined_head.parameters(), 0.5)
            optimizer.step()

    print(f"  Combined head trained for {steps} steps on {len(signal_buffer)} samples",
          flush=True)


def _eval_combined_probes(
    agent: REEAgent,
    combined_head: nn.Module,
    reaf_predictor: ReafferencePredictor,
    env: CausalGridWorld,
    num_resets: int,
) -> Dict:
    """
    Combined attribution probe.

    causal_sig = combined_head([z_harm; E2.world_forward(z_w_corr_act, a_act)])
               - combined_head([z_harm; E2.world_forward(z_w_corr_cf, a_cf)])
    """
    agent.eval()
    combined_head.eval()
    reaf_predictor.eval()

    near_sigs: List[float] = []
    safe_sigs: List[float] = []
    near_harm_norms: List[float] = []
    safe_harm_norms: List[float] = []
    all_pred_vals: List[float] = []
    fatal_errors = 0

    wall_type = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _probe(ax: int, ay: int, actual_idx: int) -> Tuple[float, float]:
        """Returns (causal_sig, z_harm_norm)."""
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world_raw = latent.z_world
            z_self_curr = latent.z_self
            z_harm = latent.z_harm  # [1, harm_dim]
            if z_harm is None:
                return 0.0, 0.0

            a_act = _action_to_onehot(actual_idx, env.action_dim, agent.device)
            cf_idx = _random_cf_action(actual_idx, env.action_dim)
            a_cf = _action_to_onehot(cf_idx, env.action_dim, agent.device)

            z_w_corr_act = reaf_predictor.correct_z_world(z_world_raw, z_self_curr, a_act)
            z_w_corr_cf = reaf_predictor.correct_z_world(z_world_raw, z_self_curr, a_cf)

            z_w_next_act = agent.e2.world_forward(z_w_corr_act, a_act)
            z_w_next_cf = agent.e2.world_forward(z_w_corr_cf, a_cf)

            inp_act = torch.cat([z_harm, z_w_next_act], dim=-1)
            inp_cf = torch.cat([z_harm, z_w_next_cf], dim=-1)

            v_act = combined_head(inp_act)
            v_cf = combined_head(inp_cf)
            all_pred_vals.extend([float(v_act.item()), float(v_cf.item())])

            harm_norm = float(torch.norm(z_harm).item())
            return float((v_act - v_cf).item()), harm_norm

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
                            sig, hn = _probe(ax, ay, action_idx)
                            near_sigs.append(sig)
                            near_harm_norms.append(hn)

            for px in range(env.size):
                for py in range(env.size):
                    if int(env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(abs(px - hx) + abs(py - hy) for hx, hy in env.hazards)
                    if min_dist > 3:
                        sig, hn = _probe(px, py, random.randint(0, 3))
                        safe_sigs.append(sig)
                        safe_harm_norms.append(hn)

    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL probe: {traceback.format_exc()}", flush=True)

    mean_near = float(sum(near_sigs) / max(1, len(near_sigs)))
    mean_safe = float(sum(safe_sigs) / max(1, len(safe_sigs)))
    calibration_gap = mean_near - mean_safe

    mean_harm_near = float(sum(near_harm_norms) / max(1, len(near_harm_norms)))
    mean_harm_safe = float(sum(safe_harm_norms) / max(1, len(safe_harm_norms)))
    pred_std = float(torch.tensor(all_pred_vals).std().item()) if len(all_pred_vals) > 1 else 0.0

    print(f"  Combined probe — n_near={len(near_sigs)} n_safe={len(safe_sigs)}", flush=True)
    print(f"  near={mean_near:.4f}  safe={mean_safe:.4f}  gap={calibration_gap:.4f}  "
          f"pred_std={pred_std:.4f}", flush=True)
    print(f"  ||z_harm|| near={mean_harm_near:.4f}  safe={mean_harm_safe:.4f}  "
          f"margin={mean_harm_near-mean_harm_safe:.4f}", flush=True)

    return {
        "calibration_gap": calibration_gap,
        "mean_causal_sig_near": mean_near,
        "mean_causal_sig_safe": mean_safe,
        "z_harm_selectivity_margin": mean_harm_near - mean_harm_safe,
        "mean_z_harm_norm_near": mean_harm_near,
        "mean_z_harm_norm_safe": mean_harm_safe,
        "n_near_hazard_probes": len(near_sigs),
        "n_safe_probes": len(safe_sigs),
        "net_eval_pred_std": pred_std,
        "fatal_errors": fatal_errors,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 1000,
    eval_probe_resets: int = 10,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    harm_scale: float = 0.02,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
        hazard_harm=harm_scale, contaminated_harm=harm_scale,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        harm_dim=HARM_DIM,  # Enable lateral head
        alpha_world=alpha_world,
    )
    agent = REEAgent(config)
    world_decoder = _make_world_decoder(world_dim, env.world_obs_dim)
    combined_head = _make_combined_head(HARM_DIM, world_dim)

    e12_params = list(agent.parameters()) + list(world_decoder.parameters())
    optimizer = optim.Adam(e12_params, lr=lr)

    print(f"[V3-EXQ-017] Warmup + collect: {warmup_episodes} eps  "
          f"(harm_dim={HARM_DIM}, reafference+lateral combined)", flush=True)
    train_out = _train_and_collect(
        agent, env, world_decoder, optimizer, warmup_episodes, steps_per_episode
    )
    warmup_harm = train_out["total_harm"]
    warmup_benefit = train_out["total_benefit"]
    n_empty_steps = train_out["n_empty_steps"]
    reaf_data = train_out["reaf_data"]
    signal_buffer = train_out["signal_buffer"]

    print(f"  Warmup done. harm={warmup_harm}  benefit={warmup_benefit}  "
          f"empty_steps={n_empty_steps}  sig_buf={len(signal_buffer)}", flush=True)

    # Phase 2: train ReafferencePredictor
    print(f"[V3-EXQ-017] Training ReafferencePredictor ({len(reaf_data)} empty steps)...",
          flush=True)
    reaf_predictor, r2_train, r2_test = _train_reafference_predictor(
        reaf_data, self_dim, env.action_dim, world_dim
    )

    # Phase 3: train combined head
    print(f"[V3-EXQ-017] Training combined head ({len(signal_buffer)} samples)...",
          flush=True)
    _train_combined_head(combined_head, signal_buffer, reaf_predictor, agent,
                         HARM_DIM, world_dim, steps=300, harm_scale=harm_scale)

    # Phase 4: probe
    print(f"[V3-EXQ-017] Combined probe ({eval_probe_resets} resets)...", flush=True)
    probe = _eval_combined_probes(agent, combined_head, reaf_predictor, env, eval_probe_resets)
    fatal_errors = probe["fatal_errors"]

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1_pass = probe["calibration_gap"] > 0.05
    c2_pass = probe["z_harm_selectivity_margin"] > 0.01
    c3_pass = max(0.0, r2_test) > 0.2
    c4_pass = warmup_harm > 100
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(f"C1 FAIL: calibration_gap={probe['calibration_gap']:.4f} <= 0.05")
    if not c2_pass:
        failure_notes.append(f"C2 FAIL: z_harm selectivity={probe['z_harm_selectivity_margin']:.4f} <= 0.01")
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: reafference R²_test={r2_test:.3f} <= 0.2")
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: warmup_harm={warmup_harm} <= 100")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-017 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count": float(fatal_errors),
        "warmup_harm_events": float(warmup_harm),
        "warmup_benefit_events": float(warmup_benefit),
        "n_empty_steps_collected": float(n_empty_steps),
        "harm_dim": float(HARM_DIM),
        "reafference_reconstruction_r2": float(max(0.0, r2_test)),
        "reafference_r2_train": float(r2_train),
        "calibration_gap": float(probe["calibration_gap"]),
        "mean_causal_sig_near_hazard": float(probe["mean_causal_sig_near"]),
        "mean_causal_sig_safe": float(probe["mean_causal_sig_safe"]),
        "z_harm_selectivity_margin": float(probe["z_harm_selectivity_margin"]),
        "mean_z_harm_norm_near": float(probe["mean_z_harm_norm_near"]),
        "mean_z_harm_norm_safe": float(probe["mean_z_harm_norm_safe"]),
        "n_near_hazard_probes": float(probe["n_near_hazard_probes"]),
        "n_safe_probes": float(probe["n_safe_probes"]),
        "net_eval_pred_std": float(probe["net_eval_pred_std"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-017 — Combined Lateral + Reafference (MECH-099 + MECH-098)

**Status:** {status}
**Warmup:** {warmup_episodes} eps (RANDOM policy, 12×12, 15 hazards, drift_interval=3, drift_prob=0.5)
**Probe eval:** {eval_probe_resets} grid resets
**harm_dim:** {HARM_DIM}  **world_dim:** {world_dim}  **Seed:** {seed}

## Motivation (MECH-099 + MECH-098 / SD-007 / SD-003)

EXQ-015 (lateral head alone) and EXQ-016 (reafference alone) tested individual fixes
for the E2 identity shortcut (EXQ-012 calibration_gap ≈ 0.0007). This experiment
combines both: reafference correction removes perspective shift from z_world, while
the lateral head provides harm-salient z_harm context. The combined attribution:
  combined_act = combined_head([z_harm; E2.world_forward(z_w_corr_act, a_act)])
  combined_cf  = combined_head([z_harm; E2.world_forward(z_w_corr_cf, a_cf)])
  causal_sig   = combined_act - combined_cf

## ReafferencePredictor

- R² train: {r2_train:.3f} | R² test: {r2_test:.3f}
- Empty-step data: {n_empty_steps}

## Probe Results

| Metric | Near-Hazard | Safe | Margin |
|---|---|---|---|
| Combined causal_sig | {probe["mean_causal_sig_near"]:.4f} | {probe["mean_causal_sig_safe"]:.4f} | {probe["calibration_gap"]:.4f} |
| \|\|z_harm\|\| norm | {probe["mean_z_harm_norm_near"]:.4f} | {probe["mean_z_harm_norm_safe"]:.4f} | {probe["z_harm_selectivity_margin"]:.4f} |

pred_std: {probe["net_eval_pred_std"]:.4f}
Warmup: harm={warmup_harm}  benefit={warmup_benefit}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap > 0.05 | {"PASS" if c1_pass else "FAIL"} | {probe["calibration_gap"]:.4f} |
| C2: z_harm selectivity > 0.01 | {"PASS" if c2_pass else "FAIL"} | {probe["z_harm_selectivity_margin"]:.4f} |
| C3: reafference R²_test > 0.2 | {"PASS" if c3_pass else "FAIL"} | {r2_test:.3f} |
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
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--warmup",         type=int,   default=1000)
    parser.add_argument("--probe-resets",   type=int,   default=10)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
        harm_scale=args.harm_scale,
        alpha_world=args.alpha_world,
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
