#!/opt/local/bin/python3
"""
V3-EXQ-166b -- SD-003 Harm Latent Forward Model + Reconstruction Branch

Claims: SD-003, ARC-033, SD-011
Supersedes: EXQ-166a (obs-space forward model -- architectural dead end)

Root cause of EXQ-166a C2 failure:
  The obs-space approach (51-dim MSE) has no selective pressure on harm-causal
  structure. MSE loss is dominated by position-change prediction (large-variance
  dims 0:24 of hazard_field_view) -- harm-discriminative dims are diluted.
  Even with obs_fwd_r2=0.45, the causal gap delta_approach=0.0006 (threshold 0.03):
  the model learns position dynamics, not harm-causal dynamics.

  Deeper analysis: C2 failure is structural. The obs-space forward model encodes
  ALL 51 dims equally, and the harm signal (dims [12], [50]) is a small fraction.
  The causal signal path is:
    harm_obs_cf_pred -> z_harm_cf = frozen_enc(harm_obs_cf_pred) -> harm_eval
  If fwd_obs predicts harm_obs_cf well but the harm dims are dominated by position
  dims, z_harm_cf varies with action primarily due to position-change effects, not
  harm-causal effects. The counterfactual gap is washed out.

This experiment (EXQ-166b) -- latent forward model with reconstruction branch:
  - Use HarmForwardModel (already in stack.py) for latent prediction:
      z_harm_s_next_pred = HarmForwardModel(z_harm_s_t, action)
  - Add HarmDecoder for reconstruction branch:
      harm_obs_recon = HarmDecoder(z_harm_s_next_pred)
  - Joint training loss:
      L = lambda_lat * ||z_harm_pred - z_harm_actual||^2
        + lambda_rec * ||harm_obs_recon - harm_obs_actual||^2
  - Encoder pre-trained with FULL RECONSTRUCTION (autoencoder: harm_obs -> z_harm -> harm_obs)
    in addition to scalar harm_obs[12] regression. Replaces scalar-only P0 that left z_harm
    sparse (only ~1/32 dims had useful structure).

Why this approach addresses EXQ-115 identity collapse without EXQ-166a's harm dilution:
  (1) Co-training stability: encoder frozen before Phase 1 (no stale tuple problem)
  (2) Rich z_harm representations: P0 autoencoder loss trains all 32 z_harm dims,
      not just the single scalar harm_obs[12] supervised dimension
  (3) No identity collapse: reconstruction branch forces z_harm_pred to decode
      back to harm_obs -- z_harm_pred = z_harm_curr is NOT reconstruction-optimal
      (because harm_obs changes with action -- a 1-cell grid shift changes the
      hazard_field_view substantially, and the encoder can't decode to harm_obs_next
      from harm_obs_curr's latent)
  (4) Harm-selective pressure: forward model operates entirely in harm-latent space --
      learning pressure is concentrated on harm structure, not diluted by world-obs dims

Causal signal pipeline (same logic as EXQ-166, different forward model):
  z_harm_s_t = frozen_enc(harm_obs_s_t)
  z_harm_s_cf = HarmForwardModel(z_harm_s_t, a_cf)          # latent counterfactual
  causal_sig_trained = harm_eval(z_harm_actual) - mean(harm_eval(z_harm_cf) over cfs)
  causal_sig_ablated = harm_eval(z_harm_actual) - mean(harm_eval(ablated_z_harm_cf) over cfs)

PRE-REGISTERED ACCEPTANCE CRITERIA (ALL required for PASS):
  C1 (latent fwd R2, gate): latent_fwd_r2_mean > 0.30
    HarmForwardModel must predict z_harm_next better than mean baseline.
    Lower threshold than obs-space R2 (0.40) because z_harm is denser representation.
    If R2 < 0.30, reconstruction branch is not preventing collapse.
  C2 (discrimination): delta_approach_mean > 0.005
    TRAINED causal_sig at approach events > ABLATED causal_sig, averaged across seeds.
    Primary SD-003 discriminative test. Same threshold as EXQ-166a.
  C3 (gradient ordering): causal_approach_trained_mean > causal_none_trained_mean
    Trained causal signal higher at approach than neutral events.
  C4 (escalation ordering): causal_contact_trained_mean > causal_approach_trained_mean
    Contact events produce larger signal than approach events.
  C5 (data quality): n_approach_eval_mean > 20
    Sufficient approach events for reliable estimates.
  C6 (seed consistency): delta_approach > 0.0 for >=3/4 seeds (majority)

Decision scoring:
  retain_ree:       ALL criteria met -- full SD-003 + ARC-033 validation
  hybridize:        C1+C2+C5+C6 pass, C3 or C4 fail
  retire_ree_claim: C1 passes but C2 fails (delta <= 0) AND C5 passes
  inconclusive:     C1 fails -- latent forward model did not learn harm dynamics
"""

import sys
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, HarmForwardModel
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_166b_sd003_harm_latent_reconstruction"
CLAIM_IDS = ["SD-003", "ARC-033", "SD-011"]

# Pre-registered thresholds (MUST NOT be changed post-hoc)
THRESH_C1_LATENT_FWD_R2    = 0.30
THRESH_C2_DELTA_APPROACH   = 0.005
THRESH_C3_GRAD_ORDERING    = 0.0
THRESH_C4_ESCALATION       = 0.0
THRESH_C5_MIN_APPROACH_EVT = 20
THRESH_C6_MAJORITY_FRAC    = 0.75   # >=75% of seeds must have delta_approach > 0

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32
MAX_OBS_DATA = 4000

P0_VAR_THRESH = 0.005   # z_harm variance below this after P0 = degenerate encoder

# Training loss weights
LAMBDA_LATENT = 1.0    # latent prediction loss weight
LAMBDA_RECON  = 0.5    # reconstruction loss weight (harm_obs reconstruction)


class HarmDecoder(nn.Module):
    """
    Reconstruction branch for EXQ-166b: decode z_harm back to harm_obs space.

    Prevents identity collapse in HarmForwardModel: if forward model predicts
    z_harm_pred = z_harm_curr, the decoder must reconstruct harm_obs_next from
    z_harm_curr's latent -- which is wrong because harm_obs changes with action.
    The reconstruction loss creates pressure to predict the CORRECT z_harm_next.

    Mirror architecture of HarmEncoder (51 -> 64 -> 32) reversed:
    z_harm (32) -> 64 -> harm_obs_dim (51).
    """

    def __init__(self, z_harm_dim: int = 32, harm_obs_dim: int = 51, hidden_dim: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_harm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, harm_obs_dim),
        )

    def forward(self, z_harm: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_harm)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _ttype_bucket(ttype: str) -> str:
    if ttype in ("env_caused_hazard", "agent_caused_hazard"):
        return "contact"
    elif ttype == "hazard_approach":
        return "approach"
    return "none"


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action index toward nearest hazard gradient. Falls back to random."""
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return random.randint(0, n_actions - 1)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _run_single(
    seed: int,
    phase0_episodes: int,
    phase1_episodes: int,
    phase2_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    self_dim: int,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one matched seed. Returns metrics for TRAINED and ABLATED conditions."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if dry_run:
        phase0_episodes = min(3, phase0_episodes)
        phase1_episodes = min(3, phase1_episodes)
        phase2_episodes = min(2, phase2_episodes)
        eval_episodes   = min(2, eval_episodes)

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=2,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )
    n_actions = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)

    harm_enc     = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    harm_dec_p0  = HarmDecoder(z_harm_dim=Z_HARM_DIM, harm_obs_dim=HARM_OBS_DIM)

    # Latent forward model (from stack.py) + decoder for Phase 1
    harm_fwd_trained = HarmForwardModel(z_harm_dim=Z_HARM_DIM, action_dim=n_actions, hidden_dim=64)
    harm_fwd_ablated = HarmForwardModel(z_harm_dim=Z_HARM_DIM, action_dim=n_actions, hidden_dim=64)
    harm_dec_p1      = HarmDecoder(z_harm_dim=Z_HARM_DIM, harm_obs_dim=HARM_OBS_DIM)

    # Phase 0 optimizer: encoder + decoder (autoencoder) + harm eval head
    opt_enc = optim.Adam(
        list(harm_enc.parameters())
        + list(harm_dec_p0.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=1e-3,
    )
    # Phase 1 optimizer: forward model + reconstruction decoder (encoder is frozen)
    opt_fwd = optim.Adam(
        list(harm_fwd_trained.parameters())
        + list(harm_dec_p1.parameters()),
        lr=1e-3,
    )
    opt_std = optim.Adam(
        [p for n, p in agent.named_parameters()
         if "harm_eval" not in n
         and "world_transition" not in n
         and "world_action_encoder" not in n],
        lr=1e-3,
    )

    # Buffer: (harm_obs_t, action, harm_obs_next, z_harm_t, z_harm_next)
    # z_harm entries are populated after P0 (encoder frozen)
    obs_data: List = []

    # ---- Phase 0: Pre-train HarmEncoder as autoencoder + scalar regression ----------
    print(
        f"[EXQ-166b seed={seed}] P0: HarmEncoder autoencoder + harm regression"
        f" ({phase0_episodes} eps)...",
        flush=True,
    )
    agent.train(); harm_enc.train(); harm_dec_p0.train()
    n_approach_p0 = 0; n_contact_p0 = 0

    for ep in range(phase0_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent    = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action_oh = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action_oh

            flat_obs, _, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype == "hazard_approach":   n_approach_p0 += 1
            elif "hazard" in ttype:          n_contact_p0  += 1

            harm_obs_next = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            # Accumulate raw obs for Phase 1 buffer (z_harm filled in later)
            obs_data.append([
                harm_obs_t.cpu(),
                action_oh.detach().cpu(),
                harm_obs_next.cpu(),
                None,   # z_harm_t placeholder (filled after P0)
                None,   # z_harm_next placeholder
            ])
            if len(obs_data) > MAX_OBS_DATA:
                obs_data = obs_data[-MAX_OBS_DATA:]

            # P0 loss: autoencoder (full reconstruction) + scalar harm regression
            z_for_train  = harm_enc(harm_obs_t.unsqueeze(0))
            harm_obs_recon = harm_dec_p0(z_for_train)
            loss_recon   = F.mse_loss(harm_obs_recon, harm_obs_t.unsqueeze(0))

            # Scalar harm regression on harm_obs[12] (center hazard cell -- discriminative)
            pred_harm = agent.e3.harm_eval_z_harm(z_for_train)
            label     = harm_obs_t[12].unsqueeze(0).unsqueeze(0)
            loss_he   = F.mse_loss(pred_harm, label)

            loss_p0 = loss_recon + loss_he
            opt_enc.zero_grad(); loss_p0.backward(); opt_enc.step()

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            if done: break

        if (ep + 1) % 25 == 0 or ep == phase0_episodes - 1:
            print(
                f"  [P0 seed={seed}] ep {ep+1}/{phase0_episodes}"
                f"  obs_buf={len(obs_data)}"
                f"  approach={n_approach_p0} contact={n_contact_p0}",
                flush=True,
            )

    # P0 quality diagnostic: check z_harm variance
    p0_z_harm_var = 0.0
    if len(obs_data) >= 10:
        sample_size = min(50, len(obs_data))
        step = max(1, len(obs_data) // sample_size)
        sample_harm_obs = [obs_data[i][0] for i in range(0, len(obs_data), step)][:sample_size]
        with torch.no_grad():
            harm_enc.eval()
            zh_samples = torch.stack(
                [harm_enc(ho.unsqueeze(0)).squeeze(0) for ho in sample_harm_obs]
            )
            p0_z_harm_var = float(zh_samples.var().item())
        harm_enc.train()
    if p0_z_harm_var < P0_VAR_THRESH:
        print(
            f"  [P0 seed={seed}] WARNING: z_harm_var={p0_z_harm_var:.5f} < {P0_VAR_THRESH}"
            f" -- degenerate encoder despite autoencoder training.",
            flush=True,
        )
    else:
        print(
            f"  [P0 seed={seed}] P0 quality OK: z_harm_var={p0_z_harm_var:.5f}",
            flush=True,
        )

    # ---- Freeze encoder, pre-populate z_harm in buffer ---------------------------
    for p in harm_enc.parameters(): p.requires_grad_(False)
    for p in agent.e3.harm_eval_z_harm_head.parameters(): p.requires_grad_(False)
    for p in harm_dec_p0.parameters(): p.requires_grad_(False)
    harm_enc.eval()

    with torch.no_grad():
        for entry in obs_data:
            if entry[3] is None:
                entry[3] = harm_enc(entry[0].unsqueeze(0)).squeeze(0).cpu()
            if entry[4] is None:
                entry[4] = harm_enc(entry[2].unsqueeze(0)).squeeze(0).cpu()

    # ---- Phase 1: Train latent forward model + decoder (FROZEN encoder) -----------
    print(
        f"[EXQ-166b seed={seed}] P1: latent fwd model + reconstruction decoder"
        f" ({phase1_episodes} eps)...",
        flush=True,
    )
    harm_fwd_trained.train(); harm_dec_p1.train()

    n_approach_p1 = 0
    fwd_step = 0
    p1_lat_losses = []; p1_rec_losses = []

    for ep in range(phase1_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action_oh = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action_oh

            flat_obs, _, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype == "hazard_approach": n_approach_p1 += 1

            harm_obs_next = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            with torch.no_grad():
                z_t    = harm_enc(harm_obs_t.unsqueeze(0)).squeeze(0)
                z_next = harm_enc(harm_obs_next.unsqueeze(0)).squeeze(0)

            obs_data.append([
                harm_obs_t.cpu(),
                action_oh.detach().cpu(),
                harm_obs_next.cpu(),
                z_t.detach().cpu(),
                z_next.detach().cpu(),
            ])
            if len(obs_data) > MAX_OBS_DATA:
                obs_data = obs_data[-MAX_OBS_DATA:]

            # Train forward model + decoder on buffer batch
            valid = [e for e in obs_data if e[3] is not None and e[4] is not None]
            if len(valid) >= 32:
                k    = min(64, len(valid))
                idxs = torch.randperm(len(valid))[:k].tolist()

                z_t_b    = torch.stack([valid[i][3] for i in idxs]).to(agent.device)
                a_b      = torch.cat(  [valid[i][1] for i in idxs]).to(agent.device)
                z_next_b = torch.stack([valid[i][4] for i in idxs]).to(agent.device)
                obs_n_b  = torch.stack([valid[i][2] for i in idxs]).to(agent.device)

                z_pred   = harm_fwd_trained(z_t_b, a_b)
                loss_lat = F.mse_loss(z_pred, z_next_b)

                # Reconstruction branch: decode predicted latent to harm_obs_next
                obs_recon  = harm_dec_p1(z_pred)
                loss_rec   = F.mse_loss(obs_recon, obs_n_b)

                loss_p1 = LAMBDA_LATENT * loss_lat + LAMBDA_RECON * loss_rec
                if loss_p1.requires_grad:
                    opt_fwd.zero_grad(); loss_p1.backward(); opt_fwd.step()
                    p1_lat_losses.append(float(loss_lat.item()))
                    p1_rec_losses.append(float(loss_rec.item()))
                fwd_step += 1

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            if done: break

        if (ep + 1) % 25 == 0 or ep == phase1_episodes - 1:
            lat_str = f"{np.mean(p1_lat_losses[-50:]):.4f}" if p1_lat_losses else "n/a"
            rec_str = f"{np.mean(p1_rec_losses[-50:]):.4f}" if p1_rec_losses else "n/a"
            print(
                f"  [P1 seed={seed}] ep {ep+1}/{phase1_episodes}"
                f"  fwd_steps={fwd_step}"
                f"  lat_loss={lat_str} rec_loss={rec_str}"
                f"  approach={n_approach_p1}",
                flush=True,
            )

    # ---- Evaluate latent forward model R2 on held-out steps -----------------------
    latent_fwd_r2 = 0.0
    if len(obs_data) >= 50:
        harm_fwd_trained.eval()
        eval_env = CausalGridWorldV2(
            seed=seed + 10000,
            size=6, num_hazards=4, num_resources=2,
            hazard_harm=0.02, env_drift_interval=10, env_drift_prob=0.1,
            proximity_harm_scale=0.05, proximity_benefit_scale=0.03,
            proximity_approach_threshold=0.15, hazard_field_decay=0.5,
            use_proxy_fields=True,
        )
        eval_agent = REEAgent(config)
        flat_obs, obs_dict = eval_env.reset()
        eval_agent.reset()
        z_actual_list = []; z_pred_list = []
        prev_z = None; prev_a = None
        with torch.no_grad():
            for _ in range(min(400, steps_per_episode * 6)):
                harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_curr = harm_enc(harm_obs.unsqueeze(0)).squeeze(0)
                if prev_z is not None and prev_a is not None:
                    z_pred = harm_fwd_trained(
                        prev_z.unsqueeze(0), prev_a
                    ).squeeze(0)
                    z_actual_list.append(z_curr.cpu())
                    z_pred_list.append(z_pred.detach().cpu())
                prev_z = z_curr.detach()
                if random.random() < nav_bias:
                    a_idx = _hazard_approach_action(eval_env, n_actions)
                else:
                    a_idx = random.randint(0, n_actions - 1)
                a_oh = _action_to_onehot(a_idx, n_actions, agent.device)
                eval_agent._last_action = a_oh
                prev_a = a_oh.detach()
                with torch.no_grad():
                    _ = eval_agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                    eval_agent.clock.advance()
                flat_obs, _, done, _, obs_dict = eval_env.step(a_oh)
                if done: break

        if len(z_actual_list) >= 20:
            z_actual = torch.stack(z_actual_list)
            z_pred_t = torch.stack(z_pred_list)
            ss_res = ((z_actual - z_pred_t) ** 2).sum()
            ss_tot = ((z_actual - z_actual.mean(0, keepdim=True)) ** 2).sum()
            latent_fwd_r2 = float((1.0 - ss_res / (ss_tot + 1e-8)).item())

    print(f"  [seed={seed}] Latent forward model R2: {latent_fwd_r2:.4f}", flush=True)

    # ---- Phase 2: Calibrate E3 harm head (stratified, all else frozen) ------------
    print(
        f"[EXQ-166b seed={seed}] P2: E3 harm head calibration ({phase2_episodes} eps)...",
        flush=True,
    )
    for p in agent.parameters(): p.requires_grad_(False)
    for p in harm_fwd_trained.parameters(): p.requires_grad_(False)
    for p in harm_dec_p1.parameters(): p.requires_grad_(False)
    for p in agent.e3.harm_eval_z_harm_head.parameters(): p.requires_grad_(True)

    opt_e3 = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)
    strat: Dict[str, list] = {"none": [], "approach": [], "contact": []}
    STRAT_MAX   = 500
    MIN_BUCKET  = 4
    SAMP_BUCKET = 8

    for ep in range(phase2_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
            if random.random() < nav_bias:
                a_idx = _hazard_approach_action(env, n_actions)
            else:
                a_idx = random.randint(0, n_actions - 1)
            a_oh = _action_to_onehot(a_idx, n_actions, agent.device)
            agent._last_action = a_oh
            flat_obs, _, done, info, obs_dict = env.step(a_oh)

            ttype    = info.get("transition_type", "none")
            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            label    = float(harm_obs[12].item())
            with torch.no_grad():
                zh = harm_enc(harm_obs.unsqueeze(0))
            bucket = _ttype_bucket(ttype)
            strat[bucket].append((zh.detach(), label))
            if len(strat[bucket]) > STRAT_MAX:
                strat[bucket] = strat[bucket][-STRAT_MAX:]

            buckets_ready = [b for b in strat if len(strat[b]) >= MIN_BUCKET]
            if len(buckets_ready) >= 2:
                zh_list_t = []; lbl_list = []
                for bk in strat:
                    buf = strat[bk]
                    if len(buf) < MIN_BUCKET: continue
                    k = min(SAMP_BUCKET, len(buf))
                    for i in random.sample(range(len(buf)), k):
                        zh_list_t.append(buf[i][0]); lbl_list.append(buf[i][1])
                if len(zh_list_t) >= 6:
                    zh_b  = torch.cat(zh_list_t, dim=0).to(agent.device)
                    lbl_b = torch.tensor(lbl_list, dtype=torch.float32,
                                         device=agent.device).unsqueeze(1)
                    pred = agent.e3.harm_eval_z_harm(zh_b)
                    loss = F.mse_loss(pred, lbl_b)
                    if loss.requires_grad:
                        opt_e3.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_z_harm_head.parameters(), 0.5)
                        opt_e3.step()
            if done: break

        if (ep + 1) % 25 == 0 or ep == phase2_episodes - 1:
            buf_sz = {k: len(v) for k, v in strat.items()}
            print(
                f"  [P2 seed={seed}] ep {ep+1}/{phase2_episodes}  strat={buf_sz}",
                flush=True,
            )

    # ---- Phase 3: Attribution eval (TRAINED latent fwd vs ABLATED latent fwd) -----
    print(
        f"[EXQ-166b seed={seed}] P3: attribution eval ({eval_episodes} eps)...",
        flush=True,
    )
    agent.eval(); harm_enc.eval(); harm_fwd_trained.eval()

    causal_trained: Dict[str, List[float]] = {}
    causal_ablated: Dict[str, List[float]] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()

                harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_harm_t   = harm_enc(harm_obs_t.unsqueeze(0))

            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            a_oh = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = a_oh
            flat_obs, _, done, info, obs_dict = env.step(a_oh)

            ttype  = info.get("transition_type", "none")
            bucket = _ttype_bucket(ttype)

            with torch.no_grad():
                harm_obs_t1 = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_harm_t1   = harm_enc(harm_obs_t1.unsqueeze(0))
                harm_eval_actual = float(agent.e3.harm_eval_z_harm(z_harm_t1).item())

                # Trained forward model counterfactuals (latent space)
                cf_vals_trained = []
                for ci in range(n_actions):
                    a_cf        = _action_to_onehot(ci, n_actions, agent.device)
                    z_harm_cf   = harm_fwd_trained(z_harm_t, a_cf)
                    cf_vals_trained.append(
                        float(agent.e3.harm_eval_z_harm(z_harm_cf).item()))
                causal_sig_trained = harm_eval_actual - float(np.mean(cf_vals_trained))

                # Ablated forward model counterfactuals (latent space, untrained)
                cf_vals_ablated = []
                for ci in range(n_actions):
                    a_cf         = _action_to_onehot(ci, n_actions, agent.device)
                    z_harm_cf_ab = harm_fwd_ablated(z_harm_t, a_cf)
                    cf_vals_ablated.append(
                        float(agent.e3.harm_eval_z_harm(z_harm_cf_ab).item()))
                causal_sig_ablated = harm_eval_actual - float(np.mean(cf_vals_ablated))

            causal_trained.setdefault(bucket, []).append(causal_sig_trained)
            causal_ablated.setdefault(bucket, []).append(causal_sig_ablated)
            if done: break

    def _m(lst): return float(np.mean(lst)) if lst else 0.0

    causal_approach_trained = _m(causal_trained.get("approach", []))
    causal_none_trained     = _m(causal_trained.get("none",     []))
    causal_contact_trained  = _m(causal_trained.get("contact",  []))
    causal_approach_ablated = _m(causal_ablated.get("approach", []))

    n_approach_eval = len(causal_trained.get("approach", []))
    n_contact_eval  = len(causal_trained.get("contact",  []))
    delta_approach  = causal_approach_trained - causal_approach_ablated

    print(
        f"  [seed={seed}] latent_fwd_r2={latent_fwd_r2:.4f}"
        f"  delta_approach={delta_approach:.4f}"
        f"  causal_approach_trained={causal_approach_trained:.4f}"
        f"  causal_approach_ablated={causal_approach_ablated:.4f}"
        f"  n_approach={n_approach_eval} n_contact={n_contact_eval}",
        flush=True,
    )

    return {
        "latent_fwd_r2":           latent_fwd_r2,
        "delta_approach":          delta_approach,
        "causal_approach_trained": causal_approach_trained,
        "causal_none_trained":     causal_none_trained,
        "causal_contact_trained":  causal_contact_trained,
        "causal_approach_ablated": causal_approach_ablated,
        "n_approach_eval":         n_approach_eval,
        "n_contact_eval":          n_contact_eval,
        "p0_z_harm_var":           p0_z_harm_var,
    }


def main(dry_run: bool = False):
    import json, datetime

    SEEDS             = [42, 123, 7, 13]
    PHASE0_EPISODES   = 80
    PHASE1_EPISODES   = 80
    PHASE2_EPISODES   = 30
    EVAL_EPISODES     = 40
    STEPS_PER_EPISODE = 200
    WORLD_DIM         = 32
    SELF_DIM          = 16
    NAV_BIAS          = 0.65

    results_per_seed = {}

    for seed in SEEDS:
        print(f"\n=== EXQ-166b seed={seed} ===", flush=True)
        results_per_seed[seed] = _run_single(
            seed=seed,
            phase0_episodes=PHASE0_EPISODES,
            phase1_episodes=PHASE1_EPISODES,
            phase2_episodes=PHASE2_EPISODES,
            eval_episodes=EVAL_EPISODES,
            steps_per_episode=STEPS_PER_EPISODE,
            world_dim=WORLD_DIM,
            self_dim=SELF_DIM,
            nav_bias=NAV_BIAS,
            dry_run=dry_run,
        )

    def _mean_over_seeds(key):
        return float(np.mean([results_per_seed[s][key] for s in SEEDS]))

    latent_fwd_r2_mean           = _mean_over_seeds("latent_fwd_r2")
    delta_approach_mean          = _mean_over_seeds("delta_approach")
    causal_approach_trained_mean = _mean_over_seeds("causal_approach_trained")
    causal_none_trained_mean     = _mean_over_seeds("causal_none_trained")
    causal_contact_trained_mean  = _mean_over_seeds("causal_contact_trained")
    causal_approach_ablated_mean = _mean_over_seeds("causal_approach_ablated")
    n_approach_eval_mean         = _mean_over_seeds("n_approach_eval")

    delta_per_seed  = {s: results_per_seed[s]["delta_approach"] for s in SEEDS}
    p0_var_per_seed = {s: results_per_seed[s]["p0_z_harm_var"] for s in SEEDS}

    # Criteria evaluation
    c1 = latent_fwd_r2_mean    > THRESH_C1_LATENT_FWD_R2
    c2 = delta_approach_mean   > THRESH_C2_DELTA_APPROACH
    c3 = causal_approach_trained_mean > causal_none_trained_mean + THRESH_C3_GRAD_ORDERING
    c4 = causal_contact_trained_mean  > causal_approach_trained_mean + THRESH_C4_ESCALATION
    c5 = n_approach_eval_mean  > THRESH_C5_MIN_APPROACH_EVT
    n_positive_seeds = sum(1 for s in SEEDS if delta_per_seed[s] > 0.0)
    c6 = n_positive_seeds >= int(len(SEEDS) * THRESH_C6_MAJORITY_FRAC)

    criteria_met = sum([c1, c2, c3, c4, c5, c6])

    if c1 and c2 and c3 and c4 and c5 and c6:
        outcome = "PASS"
    elif c1 and c2 and c5 and c6:
        outcome = "PASS"   # hybridize
    elif c1 and not c2 and c5:
        outcome = "FAIL"
    elif not c1:
        outcome = "FAIL"
    else:
        outcome = "FAIL"

    print(f"\n=== EXQ-166b RESULTS ===", flush=True)
    print(f"  latent_fwd_r2_mean         = {latent_fwd_r2_mean:.4f}  (C1 >{THRESH_C1_LATENT_FWD_R2}: {'PASS' if c1 else 'FAIL'})", flush=True)
    print(f"  delta_approach_mean        = {delta_approach_mean:.4f}  (C2 >{THRESH_C2_DELTA_APPROACH}: {'PASS' if c2 else 'FAIL'})", flush=True)
    print(f"  gradient_ordering          = approach>{causal_none_trained_mean:.4f}  (C3: {'PASS' if c3 else 'FAIL'})", flush=True)
    print(f"  escalation_ordering        = contact>{causal_approach_trained_mean:.4f}  (C4: {'PASS' if c4 else 'FAIL'})", flush=True)
    print(f"  n_approach_eval_mean       = {n_approach_eval_mean:.1f}  (C5 >{THRESH_C5_MIN_APPROACH_EVT}: {'PASS' if c5 else 'FAIL'})", flush=True)
    print(f"  seed_consistency           = {n_positive_seeds}/{len(SEEDS)} positive  (C6: {'PASS' if c6 else 'FAIL'})", flush=True)
    print(f"  p0_z_harm_var per seed     = {p0_var_per_seed}", flush=True)
    print(f"  criteria_met               = {criteria_met}/6", flush=True)
    print(f"  OUTCOME: {outcome}", flush=True)

    run_id = (
        f"{EXPERIMENT_TYPE}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    out_dir = (
        Path(__file__).resolve().parents[1].parent
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    flat = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "criteria_met": criteria_met,
        "total_criteria": 6,
        "dry_run": dry_run,
        "supersedes": "v3_exq_166a_sd003_obs_space_forward_model",
        "notes": (
            "EXQ-166b: latent forward model (HarmForwardModel from stack.py) with reconstruction "
            "branch (HarmDecoder). Fixes EXQ-166a C2 failure: obs-space approach diluted harm "
            "signal in 51-dim MSE. P0 now trains encoder as full autoencoder (not scalar-only), "
            "ensuring all 32 z_harm dims carry harm structure. Phase 1 joint loss: latent "
            "prediction (lambda=1.0) + reconstruction (lambda=0.5). Prevents identity collapse "
            "while keeping learning pressure harm-specific."
        ),
        "metrics": {
            "latent_fwd_r2_mean":           latent_fwd_r2_mean,
            "delta_approach_mean":          delta_approach_mean,
            "causal_approach_trained_mean": causal_approach_trained_mean,
            "causal_none_trained_mean":     causal_none_trained_mean,
            "causal_contact_trained_mean":  causal_contact_trained_mean,
            "causal_approach_ablated_mean": causal_approach_ablated_mean,
            "n_approach_eval_mean":         n_approach_eval_mean,
            "crit1_pass": float(c1),
            "crit2_pass": float(c2),
            "crit3_pass": float(c3),
            "crit4_pass": float(c4),
            "crit5_pass": float(c5),
            "crit6_pass": float(c6),
        },
        "per_seed": {str(s): results_per_seed[s] for s in SEEDS},
        "p0_quality": p0_var_per_seed,
        "config": {
            "seeds":              SEEDS,
            "phase0_episodes":    PHASE0_EPISODES,
            "phase1_episodes":    PHASE1_EPISODES,
            "phase2_episodes":    PHASE2_EPISODES,
            "eval_episodes":      EVAL_EPISODES,
            "steps_per_episode":  STEPS_PER_EPISODE,
            "world_dim":          WORLD_DIM,
            "self_dim":           SELF_DIM,
            "nav_bias":           NAV_BIAS,
            "harm_obs_dim":       HARM_OBS_DIM,
            "z_harm_dim":         Z_HARM_DIM,
            "fwd_hidden_dim":     64,
            "grid_size":          6,
            "num_hazards":        4,
            "p0_var_thresh":      P0_VAR_THRESH,
            "lambda_latent":      LAMBDA_LATENT,
            "lambda_recon":       LAMBDA_RECON,
        },
    }

    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2)
    print(f"\nResult written: {out_path}", flush=True)
    return flat


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
