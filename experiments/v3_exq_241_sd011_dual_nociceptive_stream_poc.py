#!/opt/local/bin/python3
"""
V3-EXQ-241 -- SD-011 Dual Nociceptive Stream Proof-of-Concept

Claims: SD-011
EXPERIMENT_PURPOSE = "diagnostic"

Proof-of-concept probe for SD-011 (dual nociceptive streams):
  z_harm_s: sensory-discriminative (A-delta analog, immediate proximity)
  z_harm_a: affective-motivational (C-fiber analog, slow EMA accumulation)

SD-011 is already implemented in the V3 substrate:
  - HarmEncoder(harm_obs [51]) -> z_harm_s [32]
  - AffectiveHarmEncoder(harm_obs_a [50]) -> z_harm_a [16]
  - CausalGridWorldV2 emits both harm_obs and harm_obs_a in obs_dict

This experiment validates that the two streams have the expected properties:
  - Different temporal dynamics (immediate vs accumulated)
  - Approximate orthogonality
  - Differential forward-predictability from (z_world, action)

Design
------
Phase 1 (training, first TRAIN_EPS episodes per seed):
  - Run CausalGridWorldV2 with a nav-biased random policy
  - Encode harm_obs -> z_harm_s and harm_obs_a -> z_harm_a at each step
  - Train a forward model MLP(z_world_t, action_t) -> z_harm_s_t+1 (sensory)
  - Train a forward model MLP(z_world_t, action_t) -> z_harm_a_t+1 (affective)
  - Train a small REE agent (E1/E2/E3) for z_world encoding

Phase 2 (evaluation, last EVAL_EPS episodes):
  - Collect z_harm_s and z_harm_a vectors, compute cosine similarity
  - Compute R2 for both forward model predictions on held-out steps
  - Run three conditions (SINGLE_STREAM, DUAL_STREAM_SENSORY, DUAL_STREAM_AFFECTIVE)
    for harm_rate comparison

Conditions
----------
SINGLE_STREAM:        action selection uses E3 harm_eval on z_harm_s (baseline)
DUAL_STREAM_SENSORY:  action selection uses harm_eval on z_harm_s only
DUAL_STREAM_AFFECTIVE: action selection uses harm_eval on z_harm_a (projected to z_harm_s dim)

Success criteria (diagnostic) -- aggregate 3/5 seeds:
  D1: cos_sim(z_harm_s, z_harm_a) < 0.5  (streams orthogonal)
  D2: R2(z_harm_s prediction) > 0.3       (sensory stream forward-predictable)
  D3: R2(z_harm_a prediction) < R2(z_harm_s) (affective less predictable than sensory)
  D4: harm_rate_dual_s < harm_rate_single (bonus: fast stream helps)

PASS: D1 AND D2 AND D3 (all >= 3/5 seeds)
FAIL: any of D1, D2, D3 not met

Seeds: [42, 7, 13, 99, 17]
Env:   CausalGridWorldV2 size=10, 3 hazards, 2 resources, hazard_harm=0.5
Train: 80 eps x 150 steps (training), 20 eps x 150 steps (eval)
Est:   ~250 min (DLAPTOP-4.local) -- 5 seeds x 3 conditions x 100 eps x 150 steps
"""

import sys
import random
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_241_sd011_dual_nociceptive_stream_poc"
CLAIM_IDS          = ["SD-011"]
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_COSSIM     = 0.5    # D1: cosine sim below this = orthogonal
THRESH_R2_S       = 0.3    # D2: z_harm_s forward R2 must exceed this
# D3: R2(z_harm_a) < R2(z_harm_s) -- no fixed threshold, just comparison
SEED_PASS_QUOTA   = 3      # criteria must be met in >= this many seeds

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM   = 12
WORLD_OBS_DIM  = 250
ACTION_DIM     = 5
WORLD_DIM      = 32
SELF_DIM       = 32
HARM_OBS_DIM   = 51    # harm_obs (sensory-discriminative): [25 hazard + 25 resource + 1 exposure]
HARM_OBS_A_DIM = 50    # harm_obs_a (affective EMA): [25 hazard_ema + 25 resource_ema]
Z_HARM_S_DIM   = 32    # HarmEncoder output
Z_HARM_A_DIM   = 16    # AffectiveHarmEncoder output

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
SEEDS         = [42, 7, 13, 99, 17]
TRAIN_EPS     = 80
EVAL_EPS      = 20
STEPS_PER_EP  = 150


# ---------------------------------------------------------------------------
# Small forward model: MLP(z_world + action) -> z_harm_next
# ---------------------------------------------------------------------------
class HarmStreamForwardProbe(nn.Module):
    """Small MLP forward probe: MLP(z_world_t, action_t) -> z_harm_t+1."""

    def __init__(self, z_world_dim: int, action_dim: int, z_harm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_world_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_harm_dim),
        )

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_world, action], dim=-1))


# ---------------------------------------------------------------------------
# R2 helper
# ---------------------------------------------------------------------------
def _r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R2 between prediction and target tensors (flattened)."""
    p = preds.detach().float().reshape(-1)
    t = targets.detach().float().reshape(-1)
    ss_res = ((p - t) ** 2).sum().item()
    ss_tot = ((t - t.mean()) ** 2).sum().item()
    if ss_tot < 1e-9:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _cos_sim_batch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity over batch of vector pairs."""
    # a: [N, da], b: [N, db] -- project b to a's dim if needed via mean cos_sim on normalised
    a_n = F.normalize(a.float(), dim=-1)
    # If dims differ, pad shorter to longer dim
    if a.shape[-1] != b.shape[-1]:
        min_d = min(a.shape[-1], b.shape[-1])
        a_n = a_n[:, :min_d]
        b_n = F.normalize(b.float()[:, :min_d], dim=-1)
    else:
        b_n = F.normalize(b.float(), dim=-1)
    cos_vals = (a_n * b_n).sum(dim=-1)  # [N]
    return float(cos_vals.mean().item())


# ---------------------------------------------------------------------------
# Env / config factory
# ---------------------------------------------------------------------------
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=2,
        hazard_harm=0.5,
        resource_benefit=0.3,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.08,
        harm_obs_a_ema_alpha=0.05,
    )


def _make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
    )


# ---------------------------------------------------------------------------
# Nav-biased random policy: 50% toward random resource/away from hazard
# ---------------------------------------------------------------------------
def _nav_action(env: CausalGridWorldV2) -> int:
    """Simple proximity-biased action: move away from nearest hazard 50% of time."""
    ax, ay = env.agent_x, env.agent_y
    # Move toward nearest resource or away from nearest hazard
    if random.random() < 0.5 and env.resources:
        best_d = float("inf"); target = None
        for r in env.resources:
            rx, ry = int(r[0]), int(r[1])
            d = abs(ax - rx) + abs(ay - ry)
            if d < best_d:
                best_d = d; target = (rx, ry)
        if target:
            rx, ry = target
            dx, dy = rx - ax, ry - ay
            if abs(dx) >= abs(dy):
                return 1 if dx > 0 else 0
            else:
                return 3 if dy > 0 else 2
    elif env.hazards:
        # Move away from nearest hazard
        near_h = None; near_d = float("inf")
        for h in env.hazards:
            hx, hy = int(h[0]), int(h[1])
            d = abs(ax - hx) + abs(ay - hy)
            if d < near_d:
                near_d = d; near_h = (hx, hy)
        if near_h and near_d < 3:
            hx, hy = near_h
            dx, dy = ax - hx, ay - hy
            if abs(dx) >= abs(dy):
                return 1 if dx > 0 else 0
            else:
                return 3 if dy > 0 else 2
    return random.randint(0, ACTION_DIM - 1)


# ---------------------------------------------------------------------------
# Main per-seed function
# ---------------------------------------------------------------------------
def _run_seed(seed: int, n_train: int, n_eval: int, steps: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env    = _make_env(seed)
    config = _make_config()
    agent  = REEAgent(config)

    # Dedicated harm encoders (not inside agent -- standalone per SD-011 design)
    harm_enc_s = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_S_DIM)
    harm_enc_a = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM)

    # Forward probes: MLP(z_world, action) -> z_harm_s_next / z_harm_a_next
    fwd_s = HarmStreamForwardProbe(WORLD_DIM, ACTION_DIM, Z_HARM_S_DIM)
    fwd_a = HarmStreamForwardProbe(WORLD_DIM, ACTION_DIM, Z_HARM_A_DIM)

    e1_opt  = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt  = optim.Adam(agent.e2.parameters(), lr=3e-3)
    e3_opt  = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-3,
    )
    hs_opt  = optim.Adam(
        list(harm_enc_s.parameters()) + list(fwd_s.parameters()),
        lr=1e-3,
    )
    ha_opt  = optim.Adam(
        list(harm_enc_a.parameters()) + list(fwd_a.parameters()),
        lr=1e-3,
    )

    agent.train()
    harm_enc_s.train()
    harm_enc_a.train()
    fwd_s.train()
    fwd_a.train()

    print(f"  [seed={seed}] training {n_train} eps x {steps} steps...", flush=True)

    # Replay buffer for forward model training
    buf_s: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []  # (z_world, action, z_harm_s_next)
    buf_a: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []  # (z_world, action, z_harm_a_next)
    BUF_MAX = 5000

    # ---- Training phase ----
    for ep in range(n_train):
        _, obs_dict = env.reset()
        agent.reset()

        prev_z_world: Optional[torch.Tensor] = None
        prev_action:  Optional[torch.Tensor] = None

        for _ in range(steps):
            obs_body  = obs_dict["body_state"].unsqueeze(0)
            obs_world = obs_dict["world_state"].unsqueeze(0)
            harm_obs  = obs_dict["harm_obs"].unsqueeze(0)    # [1, 51]
            harm_obs_a = obs_dict["harm_obs_a"].unsqueeze(0) # [1, 50]

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()
            _      = agent._e1_tick(latent) if ticks["e1_tick"] else None

            # Encode harm streams
            with torch.no_grad():
                z_harm_s_cur = harm_enc_s(harm_obs)   # [1, 32]
                z_harm_a_cur = harm_enc_a(harm_obs_a)  # [1, 16]

            # Choose action
            action_idx = _nav_action(env)
            action_oh  = torch.zeros(1, ACTION_DIM)
            action_oh[0, action_idx] = 1.0

            # Record for forward model buffer (with previous step's world/action -> current harm)
            if prev_z_world is not None and prev_action is not None:
                if len(buf_s) >= BUF_MAX:
                    buf_s.pop(0)
                if len(buf_a) >= BUF_MAX:
                    buf_a.pop(0)
                buf_s.append((prev_z_world.detach().clone(),
                               prev_action.detach().clone(),
                               z_harm_s_cur.detach().clone()))
                buf_a.append((prev_z_world.detach().clone(),
                               prev_action.detach().clone(),
                               z_harm_a_cur.detach().clone()))

            # Train forward models in mini-batches
            if len(buf_s) >= 64:
                idxs = random.sample(range(len(buf_s)), 32)
                zw_b  = torch.cat([buf_s[i][0] for i in idxs], dim=0)  # [32, 32]
                ac_b  = torch.cat([buf_s[i][1] for i in idxs], dim=0)  # [32, 5]
                hs_b  = torch.cat([buf_s[i][2] for i in idxs], dim=0)  # [32, 32]
                pred_s = fwd_s(zw_b, ac_b)
                loss_s = F.mse_loss(pred_s, hs_b)
                hs_opt.zero_grad(); loss_s.backward(); hs_opt.step()

            if len(buf_a) >= 64:
                idxs = random.sample(range(len(buf_a)), 32)
                zw_b  = torch.cat([buf_a[i][0] for i in idxs], dim=0)  # [32, 32]
                ac_b  = torch.cat([buf_a[i][1] for i in idxs], dim=0)  # [32, 5]
                ha_b  = torch.cat([buf_a[i][2] for i in idxs], dim=0)  # [32, 16]
                pred_a = fwd_a(zw_b, ac_b)
                loss_a = F.mse_loss(pred_a, ha_b)
                ha_opt.zero_grad(); loss_a.backward(); ha_opt.step()

            # Store for next step
            prev_z_world = latent.z_world.detach().clone()
            prev_action  = action_oh.clone()

            # E1/E2/E3 training step (world model)
            z_self_prev = None
            if agent._current_latent is not None:
                z_self_prev = agent._current_latent.z_self.detach().clone()

            _, reward, done, info, obs_dict = env.step(action_oh)
            harm_signal = float(reward) if reward < 0 else 0.0

            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action_oh, latent.z_self.detach())

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                e1_opt.zero_grad(); e2_opt.zero_grad()
                total.backward()
                e1_opt.step(); e2_opt.step()

            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_t  = torch.tensor([[1.0 if harm_signal < 0 else 0.0]])
                hloss   = F.mse_loss(agent.e3.harm_eval(z_world), harm_t)
                if hloss.requires_grad:
                    e3_opt.zero_grad(); hloss.backward(); e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 40 == 0:
            print(f"  [seed={seed}] train ep {ep+1}/{n_train}", flush=True)

    # ---- Evaluation phase: collect streams, compute metrics ----
    print(f"  [seed={seed}] evaluating {n_eval} eps...", flush=True)
    harm_enc_s.eval()
    harm_enc_a.eval()
    fwd_s.eval()
    fwd_a.eval()
    agent.eval()

    all_z_harm_s: List[torch.Tensor] = []
    all_z_harm_a: List[torch.Tensor] = []
    all_fwd_s_pred: List[torch.Tensor] = []
    all_fwd_s_tgt:  List[torch.Tensor] = []
    all_fwd_a_pred: List[torch.Tensor] = []
    all_fwd_a_tgt:  List[torch.Tensor] = []

    harm_counts = {"single": 0, "dual_s": 0, "dual_a": 0}
    step_counts  = {"single": 0, "dual_s": 0, "dual_a": 0}
    conditions   = ["single", "dual_s", "dual_a"]

    # Projected affective evaluator (map z_harm_a to z_harm_s space for E3 comparison)
    affective_proj = nn.Linear(Z_HARM_A_DIM, Z_HARM_S_DIM, bias=False)
    nn.init.orthogonal_(affective_proj.weight)

    for cond in conditions:
        env_c = _make_env(seed + 1000)  # fresh env per condition for fair comparison
        _, obs_dict = env_c.reset()
        prev_z_world_c: Optional[torch.Tensor] = None
        prev_action_c:  Optional[torch.Tensor] = None

        for ep in range(n_eval):
            _, obs_dict = env_c.reset()
            for _ in range(steps):
                obs_body   = obs_dict["body_state"].unsqueeze(0)
                obs_world  = obs_dict["world_state"].unsqueeze(0)
                harm_obs   = obs_dict["harm_obs"].unsqueeze(0)
                harm_obs_a = obs_dict["harm_obs_a"].unsqueeze(0)

                with torch.no_grad():
                    latent      = agent.sense(obs_body, obs_world)
                    z_harm_s    = harm_enc_s(harm_obs)
                    z_harm_a    = harm_enc_a(harm_obs_a)

                all_z_harm_s.append(z_harm_s.detach())
                all_z_harm_a.append(z_harm_a.detach())

                # Collect forward model eval targets
                if prev_z_world_c is not None and prev_action_c is not None:
                    with torch.no_grad():
                        ps = fwd_s(prev_z_world_c, prev_action_c)
                        pa = fwd_a(prev_z_world_c, prev_action_c)
                    all_fwd_s_pred.append(ps.detach())
                    all_fwd_s_tgt.append(z_harm_s.detach())
                    all_fwd_a_pred.append(pa.detach())
                    all_fwd_a_tgt.append(z_harm_a.detach())

                # Action selection based on condition
                action_idx = _nav_action(env_c)

                if cond == "dual_s":
                    with torch.no_grad():
                        harm_score_s = agent.e3.harm_eval(z_harm_s).item()
                    if harm_score_s > 0.5 and random.random() < 0.7:
                        # Avoidance: reverse last action
                        action_idx = (action_idx + 2) % 4
                elif cond == "dual_a":
                    with torch.no_grad():
                        z_harm_a_proj = affective_proj(z_harm_a)
                        harm_score_a  = agent.e3.harm_eval(z_harm_a_proj).item()
                    if harm_score_a > 0.5 and random.random() < 0.7:
                        action_idx = (action_idx + 2) % 4

                action_oh  = torch.zeros(1, ACTION_DIM)
                action_oh[0, action_idx] = 1.0

                prev_z_world_c = latent.z_world.detach().clone()
                prev_action_c  = action_oh.clone()

                _, reward, done, info, obs_dict = env_c.step(action_oh)
                if reward < 0:
                    harm_counts[cond] += 1
                step_counts[cond] += 1

                if done:
                    break

    # ---- Compute metrics ----
    # Orthogonality: mean cos_sim over all eval steps
    if len(all_z_harm_s) > 0:
        zs_all = torch.cat(all_z_harm_s, dim=0)   # [N, 32]
        za_all = torch.cat(all_z_harm_a, dim=0)   # [N, 16]
        cos_sim_mean = _cos_sim_batch(zs_all, za_all)
    else:
        cos_sim_mean = 0.0

    # Forward model R2
    if len(all_fwd_s_pred) > 0:
        ps_all = torch.cat(all_fwd_s_pred, dim=0)
        ts_all = torch.cat(all_fwd_s_tgt,  dim=0)
        r2_s   = _r2_score(ps_all, ts_all)
    else:
        r2_s = 0.0

    if len(all_fwd_a_pred) > 0:
        pa_all = torch.cat(all_fwd_a_pred, dim=0)
        ta_all = torch.cat(all_fwd_a_tgt,  dim=0)
        r2_a   = _r2_score(pa_all, ta_all)
    else:
        r2_a = 0.0

    # Harm rates per condition
    def _harm_rate(cond: str) -> float:
        n = step_counts.get(cond, 0)
        return harm_counts.get(cond, 0) / max(1, n)

    hr_single = _harm_rate("single")
    hr_dual_s = _harm_rate("dual_s")
    hr_dual_a = _harm_rate("dual_a")

    # Diagnostic criteria per seed
    d1 = cos_sim_mean < THRESH_COSSIM
    d2 = r2_s > THRESH_R2_S
    d3 = r2_a < r2_s
    d4 = hr_dual_s < hr_single

    print(
        f"  [seed={seed}]"
        f" cos_sim={cos_sim_mean:.4f} (D1={'P' if d1 else 'F'})"
        f" r2_s={r2_s:.4f} (D2={'P' if d2 else 'F'})"
        f" r2_a={r2_a:.4f} (D3={'P' if d3 else 'F'})"
        f" hr_s={hr_single:.4f} hr_ds={hr_dual_s:.4f} (D4={'P' if d4 else 'F'})",
        flush=True,
    )

    return {
        "seed":        seed,
        "cos_sim_mean": float(cos_sim_mean),
        "r2_s_forward": float(r2_s),
        "r2_a_forward": float(r2_a),
        "harm_rate_single": float(hr_single),
        "harm_rate_dual_s": float(hr_dual_s),
        "harm_rate_dual_a": float(hr_dual_a),
        "d1_pass": bool(d1),
        "d2_pass": bool(d2),
        "d3_pass": bool(d3),
        "d4_pass": bool(d4),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run(dry_run: bool = False) -> dict:
    print(f"\n[EXQ-241] SD-011 Dual Nociceptive Stream POC (dry_run={dry_run})",
          flush=True)

    n_train = 3  if dry_run else TRAIN_EPS
    n_eval  = 2  if dry_run else EVAL_EPS
    n_steps = 20 if dry_run else STEPS_PER_EP
    seeds   = [42] if dry_run else SEEDS

    seed_results: List[Dict] = []
    for seed in seeds:
        print(f"\n--- seed={seed} ---", flush=True)
        r = _run_seed(seed, n_train, n_eval, n_steps)
        seed_results.append(r)

    # Aggregate pass counts
    d1_passes = sum(1 for r in seed_results if r["d1_pass"])
    d2_passes = sum(1 for r in seed_results if r["d2_pass"])
    d3_passes = sum(1 for r in seed_results if r["d3_pass"])
    d4_passes = sum(1 for r in seed_results if r["d4_pass"])
    n_seeds   = len(seeds)
    quota     = 1 if dry_run else SEED_PASS_QUOTA

    d1_agg = d1_passes >= quota
    d2_agg = d2_passes >= quota
    d3_agg = d3_passes >= quota
    d4_agg = d4_passes >= quota

    all_pass  = d1_agg and d2_agg and d3_agg
    status    = "PASS" if all_pass else "FAIL"
    crit_met  = sum([d1_agg, d2_agg, d3_agg])

    print(f"\n[EXQ-241] --- Aggregate Results ({n_seeds} seeds) ---", flush=True)
    for r in seed_results:
        print(
            f"  seed={r['seed']}"
            f" cos_sim={r['cos_sim_mean']:.4f}"
            f" r2_s={r['r2_s_forward']:.4f}"
            f" r2_a={r['r2_a_forward']:.4f}"
            f" hr_single={r['harm_rate_single']:.4f}"
            f" hr_dual_s={r['harm_rate_dual_s']:.4f}"
            f" D1={'P' if r['d1_pass'] else 'F'}"
            f" D2={'P' if r['d2_pass'] else 'F'}"
            f" D3={'P' if r['d3_pass'] else 'F'}"
            f" D4={'P' if r['d4_pass'] else 'F'}",
            flush=True,
        )
    print(
        f"  D1 passes={d1_passes}/{n_seeds} (>={quota}: {'PASS' if d1_agg else 'FAIL'})",
        flush=True,
    )
    print(
        f"  D2 passes={d2_passes}/{n_seeds} (>={quota}: {'PASS' if d2_agg else 'FAIL'})",
        flush=True,
    )
    print(
        f"  D3 passes={d3_passes}/{n_seeds} (>={quota}: {'PASS' if d3_agg else 'FAIL'})",
        flush=True,
    )
    print(
        f"  D4 passes={d4_passes}/{n_seeds} (bonus, not PASS criterion)",
        flush=True,
    )
    print(f"  Status: {status} ({crit_met}/3 diagnostic criteria)", flush=True)

    if all_pass:
        interpretation = (
            "SD-011 DUAL STREAM CONFIRMED: z_harm_s and z_harm_a are approximately "
            "orthogonal (D1), z_harm_s is forward-predictable from (z_world, action) "
            "(D2), and z_harm_a is less predictable than z_harm_s (D3). "
            "Dual-stream architecture behaves as expected."
        )
    elif crit_met >= 2:
        interpretation = (
            "SD-011 PARTIAL: "
            + ("D1 (orthogonality) " if d1_agg else "")
            + ("D2 (sensory predictability) " if d2_agg else "")
            + ("D3 (affective less predictable) " if d3_agg else "")
            + "met. Check failure criteria for diagnostic guidance."
        )
    else:
        interpretation = (
            "SD-011 NOT CONFIRMED: Dual-stream separation not observed. "
            "Possible causes: encoders not diverging (same input structure), "
            "insufficient training steps, or env too uniform for stream separation."
        )

    failure_notes: List[str] = []
    if not d1_agg:
        vals = [round(r["cos_sim_mean"], 4) for r in seed_results]
        failure_notes.append(
            f"D1 FAIL: cos_sim >= {THRESH_COSSIM} in {n_seeds - d1_passes} seeds: {vals}. "
            "Streams not orthogonal -- may share input structure too strongly."
        )
    if not d2_agg:
        vals = [round(r["r2_s_forward"], 4) for r in seed_results]
        failure_notes.append(
            f"D2 FAIL: R2(z_harm_s forward) < {THRESH_R2_S} in {n_seeds - d2_passes} seeds: {vals}. "
            "Sensory stream not forward-predictable from (z_world, action)."
        )
    if not d3_agg:
        pairs = [(round(r["r2_s_forward"], 4), round(r["r2_a_forward"], 4)) for r in seed_results]
        failure_notes.append(
            f"D3 FAIL: R2(z_harm_a) >= R2(z_harm_s) in {n_seeds - d3_passes} seeds: {pairs}. "
            "Affective stream should be less predictable than sensory stream."
        )
    for note in failure_notes:
        print(f"  [EXQ-241] {note}", flush=True)

    # Summary markdown
    summary_markdown = (
        f"# V3-EXQ-241 -- SD-011 Dual Nociceptive Stream POC\n\n"
        f"**Status:** {status}  **Criteria met:** {crit_met}/3\n"
        f"**Claims:** SD-011  **Purpose:** diagnostic\n\n"
        f"## Design\n\n"
        f"Standalone dual-stream harm encoder validation (no ree_core changes required).\n"
        f"HarmEncoder(harm_obs [51]) -> z_harm_s [32] (sensory-discriminative, A-delta analog).\n"
        f"AffectiveHarmEncoder(harm_obs_a [50]) -> z_harm_a [16] (affective-motivational, C-fiber analog).\n"
        f"Forward probes: MLP(z_world, action) -> z_harm_s_next and -> z_harm_a_next.\n\n"
        f"## Results by Seed\n\n"
        f"| Seed | cos_sim | R2_s | R2_a | hr_single | hr_dual_s | D1 | D2 | D3 | D4 |\n"
        f"|------|---------|------|------|-----------|-----------|----|----|----|----|"
    )
    for r in seed_results:
        summary_markdown += (
            f"\n| {r['seed']}"
            f" | {r['cos_sim_mean']:.4f}"
            f" | {r['r2_s_forward']:.4f}"
            f" | {r['r2_a_forward']:.4f}"
            f" | {r['harm_rate_single']:.4f}"
            f" | {r['harm_rate_dual_s']:.4f}"
            f" | {'PASS' if r['d1_pass'] else 'FAIL'}"
            f" | {'PASS' if r['d2_pass'] else 'FAIL'}"
            f" | {'PASS' if r['d3_pass'] else 'FAIL'}"
            f" | {'PASS' if r['d4_pass'] else 'FAIL'} |"
        )
    summary_markdown += f"\n\n## Interpretation\n\n{interpretation}\n"
    if failure_notes:
        summary_markdown += "\n## Failure Notes\n\n"
        summary_markdown += "\n".join(f"- {n}" for n in failure_notes) + "\n"

    metrics: Dict = {
        "d1_pass": 1.0 if d1_agg else 0.0,
        "d2_pass": 1.0 if d2_agg else 0.0,
        "d3_pass": 1.0 if d3_agg else 0.0,
        "d4_pass": 1.0 if d4_agg else 0.0,
        "criteria_met": float(crit_met),
        "d1_passes_n": float(d1_passes),
        "d2_passes_n": float(d2_passes),
        "d3_passes_n": float(d3_passes),
        "d4_passes_n": float(d4_passes),
    }
    for i, r in enumerate(seed_results):
        sfx = f"_seed{i}"
        metrics[f"cos_sim{sfx}"]       = float(r["cos_sim_mean"])
        metrics[f"r2_s_forward{sfx}"]  = float(r["r2_s_forward"])
        metrics[f"r2_a_forward{sfx}"]  = float(r["r2_a_forward"])
        metrics[f"harm_rate_single{sfx}"] = float(r["harm_rate_single"])
        metrics[f"harm_rate_dual_s{sfx}"] = float(r["harm_rate_dual_s"])
        metrics[f"harm_rate_dual_a{sfx}"] = float(r["harm_rate_dual_a"])

    return {
        "status":         status,
        "metrics":        metrics,
        "summary_markdown": summary_markdown,
        "claim_ids":      CLAIM_IDS,
        "evidence_direction_per_claim": {
            "SD-011": "supports" if all_pass else ("mixed" if crit_met >= 2 else "weakens"),
        },
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if crit_met >= 2 else "weakens")
        ),
        "experiment_type":    EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "fatal_error_count":  0,
        "seed_results":       seed_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(dry_run=args.dry_run)

    ts     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"v3_exq_241_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
