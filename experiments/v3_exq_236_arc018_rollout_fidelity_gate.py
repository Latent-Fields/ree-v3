#!/opt/local/bin/python3
"""
V3-EXQ-236 -- ARC-018: Rollout Fidelity Gate

=============================================================================
DESIGN NOTE -- CONFLICT RESOLUTION
=============================================================================
ARC-018 currently has mixed evidence that traces to three distinct failures:

  EXQ-120 PASS  (2026-03-28): VIABILITY_MAP_ON vs MAP_ABLATED (no planning at
    all). MAP_ON achieved 99.2% harm reduction. Strong support -- but the
    ablated condition had NO trajectory proposal, so this tested "does any
    planning help?" not "does multi-step rollout beat 1-step greedy?"

  EXQ-172 FAIL  (2026-03-30): ROLLOUT_VIABILITY (k=5) vs GREEDY_HARM (k=1).
    Used lightweight inline toy models (not ree_core). E2 quality r2=0.20
    (seed-averaged). Rollout 19% WORSE than greedy. Consistent with the
    hypothesis that k-step rollout amplifies E2 prediction error when
    r2 is marginal.

  EXQ-196 FAIL  (2026-04-01): VIABILITY_MAPPED vs VIABILITY_ABLATED with real
    ree_core modules. E2 r2=0.72 (good). But both conditions produced
    IDENTICAL harm_rate (0.848). Root cause: environment saturation.
    num_hazards=3 + proximity_harm_scale=0.05 + size=8 produced harm at 85%
    of steps regardless of strategy. Neither condition had room to improve.

WHAT THIS EXPERIMENT TESTS:
  The actual discriminative question -- within the trained V3 viability-map
  system, does k=5 step E2 rollout via HippocampalModule CEM outperform
  1-step greedy harm evaluation?

  The E2-quality-gating hypothesis: rollout advantage should be positively
  correlated with per-seed E2 r2. With r2~0.20 (EXQ-172), rollout amplifies
  error. With r2>0.4 and a navigable environment, rollout should show clear
  benefit.

COMPARISON:
  ROLLOUT_k5:  HippocampalModule CEM (horizon=5, 8 candidates, 3 CEM iters)
               + E3.select() scores each 5-step trajectory via J(zeta).
               Residue field ACTIVE. Same trained modules as GREEDY.

  GREEDY_k1:   Evaluate all 4 directional actions 1-step ahead via
               E2.world_forward + E3.harm_eval + residue_field.evaluate.
               Pick action minimising (lambda*harm + rho*residue).
               Residue field ACTIVE. Same trained modules as ROLLOUT.

  The ONLY experimental difference is planning horizon.
  Both conditions use the same E2, E3, ResidueField trained jointly.
  Both conditions have the viability map active.

E2-QUALITY DIAGNOSTIC:
  Per-seed E2 r2 computed after training. Pearson correlation between
  per-seed r2 and per-seed harm advantage (GREEDY - ROLLOUT harm rate)
  is reported as C5 diagnostic. Positive correlation supports E2-gating
  hypothesis; negative/zero correlation weakens it.

ENVIRONMENT FIX:
  EXQ-196 saturation root cause: proximity_harm_scale=0.05 + hazard
  proximity field covers most of 8x8 grid -> 85% harm regardless of strategy.
  Fix: proximity_harm_scale=0.0 (point-source hazard harm only).
  Harm counted as direct hazard cell contact (transition_type=="env_caused_hazard").
  Expected GREEDY harm rate ~25-50% (interpretable signal for comparison).

HOW THIS RESOLVES THE CONFLICT:
  PASS:          EXQ-172 failure was E2 quality artifact (r2=0.20 too low).
                 At adequate E2 quality, rollout adds value. ARC-018 supported.
  FAIL + C5_pos: E2-gating confirmed, but r2 still insufficient in this run.
                 ARC-018 conditionally supported subject to E2 quality floor.
  FAIL + C5_neg: Rollout provides no benefit even with quality E2.
                 ARC-018 weakened -- multi-step mechanism not load-bearing.
=============================================================================

Claim: ARC-018 -- hippocampus.rollout_viability_mapping
  "Hippocampus generates explicit rollouts and post-commitment viability mapping."

Dispatch mode: discriminative_pair
Seeds: [42, 7, 13, 100, 200]
"""

import sys
import copy
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import (
    E2Config, E3Config, HippocampalConfig, ResidueConfig,
)
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField
from ree_core.hippocampal.module import HippocampalModule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_236_arc018_rollout_fidelity_gate"
CLAIM_IDS = ["ARC-018"]
SEEDS = [42, 7, 13, 100, 200]

# Environment -- point-source harm only (no proximity saturation)
ENV_SIZE = 8
NUM_HAZARDS = 2          # sparser than EXQ-196 (was 3)
NUM_RESOURCES = 2
HAZARD_HARM = 0.4        # clear harm signal per direct contact
PROXIMITY_HARM_SCALE = 0.0   # KEY FIX: disable proximity harm to avoid saturation

# Architecture -- canonical V3 dims (same as EXQ-196)
WORLD_DIM = 32
SELF_DIM = 32
ACTION_DIM = 4           # directional only (no STAY) -- consistent with EXQ-196
ACTION_OBJECT_DIM = 16
BODY_OBS_DIM = 12        # use_proxy_fields=True adds harm_exposure + benefit_exposure
WORLD_OBS_DIM = 250      # use_proxy_fields=True adds hazard_field[25] + resource_field[25]

# Training
LR = 3e-4
N_TRAIN_EPISODES = 250   # reduced from 400 to allow r2 variation across seeds
STEPS_PER_EPISODE = 150

# Eval
N_EVAL_EPISODES = 30
EVAL_STEPS = 100

# HippocampalModule CEM parameters (ROLLOUT condition)
CEM_HORIZON = 5
CEM_CANDIDATES = 8
CEM_ITERATIONS = 3
CEM_ELITE_FRAC = 0.375   # 3/8 candidates as elite

# Pre-registered thresholds
THRESH_C2_ADVANTAGE = 0.03   # rollout must reduce direct-contact harm rate by >=3pp mean
THRESH_C3_E2_QUALITY = 0.20  # E2 must reach r2>=0.20 for valid test
THRESH_C4_MIN_CONTACTS = 3   # need >=3 direct hazard contacts in GREEDY per seed

# E3 scoring weights (from E3Config)
LAMBDA_ETHICAL = 1.0
RHO_RESIDUE = 0.5


# ---------------------------------------------------------------------------
# Simple encoders (same as EXQ-196)
# ---------------------------------------------------------------------------

class SimpleWorldEncoder(nn.Module):
    """Encode world_obs -> z_world."""

    def __init__(self, world_obs_dim: int = WORLD_OBS_DIM, world_dim: int = WORLD_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, world_dim),
            nn.LayerNorm(world_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleSelfEncoder(nn.Module):
    """Encode body_obs -> z_self."""

    def __init__(self, body_obs_dim: int = BODY_OBS_DIM, self_dim: int = SELF_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(body_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self_dim),
            nn.LayerNorm(self_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Build modules
# ---------------------------------------------------------------------------

def build_modules(seed: int) -> dict:
    """Construct all ree_core modules for one seed."""
    torch.manual_seed(seed)

    e2_cfg = E2Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=128,
        action_object_dim=ACTION_OBJECT_DIM,
        rollout_horizon=CEM_HORIZON,
    )
    e2 = E2FastPredictor(e2_cfg)

    res_cfg = ResidueConfig(
        world_dim=WORLD_DIM,
        hidden_dim=64,
        accumulation_rate=0.15,
        num_basis_functions=32,
        kernel_bandwidth=1.0,
    )
    residue_field = ResidueField(res_cfg)

    e3_cfg = E3Config(
        world_dim=WORLD_DIM,
        hidden_dim=64,
        lambda_ethical=LAMBDA_ETHICAL,
        rho_residue=RHO_RESIDUE,
    )
    e3 = E3TrajectorySelector(e3_cfg, residue_field=residue_field)

    hipp_cfg = HippocampalConfig(
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM,
        hidden_dim=128,
        horizon=CEM_HORIZON,
        num_candidates=CEM_CANDIDATES,
        num_cem_iterations=CEM_ITERATIONS,
        elite_fraction=CEM_ELITE_FRAC,
    )
    hippocampal = HippocampalModule(hipp_cfg, e2=e2, residue_field=residue_field)

    world_enc = SimpleWorldEncoder()
    self_enc = SimpleSelfEncoder()

    return {
        "e2": e2,
        "e3": e3,
        "residue_field": residue_field,
        "hippocampal": hippocampal,
        "world_enc": world_enc,
        "self_enc": self_enc,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(modules: dict, seed: int, dry_run: bool = False) -> dict:
    """
    Train E2 (world_forward), E3 (harm_eval), encoders on random exploration.

    Architecture assignments (ree-v3/CLAUDE.md):
      E2 trains on motor-sensory + world prediction MSE.
      E3 trains on harm_eval MSE.
      ResidueField accumulates at z_world locations where direct harm occurs.

    Returns: dict with e2_world_r2 and residue statistics.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed,
        proximity_harm_scale=PROXIMITY_HARM_SCALE,
    )

    e2 = modules["e2"]
    e3 = modules["e3"]
    residue_field = modules["residue_field"]
    world_enc = modules["world_enc"]
    self_enc = modules["self_enc"]

    all_params = (
        list(e2.parameters())
        + list(e3.parameters())
        + list(world_enc.parameters())
        + list(self_enc.parameters())
    )
    optimizer = optim.Adam(all_params, lr=LR)

    n_eps = N_TRAIN_EPISODES if not dry_run else 5
    steps = STEPS_PER_EPISODE if not dry_run else 20

    # Replay buffer: (ws, bs, action_oh, ws_next, bs_next, harm_val)
    buffer = []
    MAX_BUFFER = 15000

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()
        bs = obs_dict["body_state"].float()

        for _step in range(steps):
            a = rng.randint(0, ACTION_DIM - 1)
            action_oh = torch.zeros(ACTION_DIM)
            action_oh[a] = 1.0

            flat_next, reward, done, info, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float()
            bs_next = obs_dict_next["body_state"].float()

            # Direct contact harm only (proximity harm disabled)
            harm_val = max(-reward, 0.0)

            buffer.append((
                ws.clone(), bs.clone(), action_oh.clone(),
                ws_next.clone(), bs_next.clone(), float(harm_val),
            ))
            if len(buffer) > MAX_BUFFER:
                buffer.pop(0)

            # Accumulate residue at z_world on direct hazard contact
            if info.get("transition_type") == "env_caused_hazard" and harm_val > 0.0:
                with torch.no_grad():
                    z_w = world_enc(ws.unsqueeze(0))
                residue_field.accumulate(
                    z_w, harm_magnitude=harm_val, hypothesis_tag=False,
                )

            ws = ws_next
            bs = bs_next
            if done:
                break

        # Mini-batch training
        if len(buffer) >= 64:
            batch = random.sample(buffer, min(128, len(buffer)))
            ws_b = torch.stack([b[0] for b in batch])
            bs_b = torch.stack([b[1] for b in batch])
            act_b = torch.stack([b[2] for b in batch])
            ws_next_b = torch.stack([b[3] for b in batch])
            bs_next_b = torch.stack([b[4] for b in batch])
            harm_b = torch.tensor(
                [b[5] for b in batch], dtype=torch.float32
            ).unsqueeze(1)

            z_world = world_enc(ws_b)
            z_self = self_enc(bs_b)
            z_world_next_tgt = world_enc(ws_next_b).detach()
            z_self_next_tgt = self_enc(bs_next_b).detach()

            z_world_next_pred = e2.world_forward(z_world, act_b)
            loss_e2_world = F.mse_loss(z_world_next_pred, z_world_next_tgt)

            z_self_next_pred = e2.predict_next_self(z_self, act_b)
            loss_e2_self = F.mse_loss(z_self_next_pred, z_self_next_tgt)

            harm_pred = e3.harm_eval(z_world)
            loss_harm = F.mse_loss(harm_pred, harm_b)

            loss = loss_e2_world + loss_e2_self + loss_harm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (ep + 1) % 100 == 0:
            print(
                f"  [ep {ep+1}/{n_eps}] buffer={len(buffer)} "
                f"residue_events={residue_field.num_harm_events.item()}",
                flush=True,
            )

    # E2 quality diagnostic: held-out r2 on world_forward
    e2_world_r2 = 0.0
    world_enc.eval()
    self_enc.eval()
    e2.eval()
    e3.eval()

    if len(buffer) >= 100:
        eval_buf = buffer[-200:]
        ws_e = torch.stack([b[0] for b in eval_buf])
        act_e = torch.stack([b[2] for b in eval_buf])
        ws_next_e = torch.stack([b[3] for b in eval_buf])

        with torch.no_grad():
            z_e = world_enc(ws_e)
            z_pred = e2.world_forward(z_e, act_e)
            z_tgt = world_enc(ws_next_e)

        ss_res = ((z_pred - z_tgt) ** 2).sum().item()
        ss_tot = ((z_tgt - z_tgt.mean(0)) ** 2).sum().item()
        e2_world_r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    residue_stats = residue_field.get_statistics()
    train_metrics = {
        "e2_world_r2": e2_world_r2,
        "residue_total": float(residue_stats["total_residue"].item()),
        "residue_harm_events": int(residue_stats["num_harm_events"].item()),
        "residue_active_centers": int(residue_stats["active_centers"].item()),
    }

    world_enc.train()
    self_enc.train()
    e2.train()
    e3.train()

    return train_metrics


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def greedy_k1_select(
    z_world: torch.Tensor,
    e2: E2FastPredictor,
    e3: E3TrajectorySelector,
    residue_field: ResidueField,
) -> int:
    """
    1-step greedy action selection.

    For each of ACTION_DIM directional actions:
      z_world_next = E2.world_forward(z_world, action)
      score = lambda * E3.harm_eval(z_world_next) + rho * residue.evaluate(z_world_next)
    Returns argmin action index.

    This is equivalent to running J(zeta) with horizon=1 (no rollout),
    using the same harm_eval and residue_field as the ROLLOUT condition.
    """
    scores = []
    for a in range(ACTION_DIM):
        ao = torch.zeros(1, ACTION_DIM, device=z_world.device)
        ao[0, a] = 1.0
        with torch.no_grad():
            z_next = e2.world_forward(z_world, ao)       # [1, world_dim]
            harm = e3.harm_eval(z_next).item()           # scalar
            residue = residue_field.evaluate(z_next).item()  # scalar
        scores.append(LAMBDA_ETHICAL * harm + RHO_RESIDUE * residue)
    return int(torch.tensor(scores).argmin().item())


def rollout_k5_select(
    z_world: torch.Tensor,
    z_self: torch.Tensor,
    hippocampal: HippocampalModule,
    e3: E3TrajectorySelector,
    rng: random.Random,
) -> int:
    """
    k=5 step rollout via HippocampalModule CEM + E3.select().

    HippocampalModule proposes CEM_CANDIDATES trajectories of horizon=CEM_HORIZON
    steps in action-object space, guided by residue terrain.
    E3.select() scores each trajectory with J(zeta) = F + lambda*M + rho*Phi_R
    and returns the first action from the best trajectory.

    Returns: discrete action index [0, ACTION_DIM).
    Falls back to random if no candidates returned.
    """
    with torch.no_grad():
        candidates = hippocampal.propose_trajectories(
            z_world, z_self=z_self, num_candidates=CEM_CANDIDATES,
        )

    if not candidates:
        return rng.randint(0, ACTION_DIM - 1)

    with torch.no_grad():
        result = e3.select(candidates, temperature=0.5)

    action_vec = result.selected_action.squeeze(0)   # [action_dim]
    return int(action_vec.argmax().item())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_condition(
    condition: str,
    modules: dict,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """
    Evaluate ROLLOUT_k5 or GREEDY_k1 condition.

    Both conditions:
    - Use the same trained E2, E3, ResidueField (shared weights)
    - Have the residue field ACTIVE (viability map informs both)
    - Count harm as direct hazard cell contacts (transition_type == env_caused_hazard)

    Harm counting via transition_type avoids the proximity EMA saturation
    that produced identical results in EXQ-196.
    """
    rng = random.Random(seed + 8888)
    torch.manual_seed(seed + 8888)

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed + 8888,
        proximity_harm_scale=PROXIMITY_HARM_SCALE,
    )

    e2 = modules["e2"]
    e3 = modules["e3"]
    residue_field = modules["residue_field"]
    hippocampal = modules["hippocampal"]
    world_enc = modules["world_enc"]
    self_enc = modules["self_enc"]

    world_enc.eval()
    self_enc.eval()
    e2.eval()
    e3.eval()

    n_episodes = N_EVAL_EPISODES if not dry_run else 3
    eval_steps = EVAL_STEPS if not dry_run else 20
    harm_contacts = 0
    total_steps = 0

    for _ep in range(n_episodes):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()
        bs = obs_dict["body_state"].float()

        for _step in range(eval_steps):
            with torch.no_grad():
                z_world = world_enc(ws.unsqueeze(0))   # [1, world_dim]
                z_self = self_enc(bs.unsqueeze(0))     # [1, self_dim]

            if condition == "ROLLOUT_k5":
                a = rollout_k5_select(z_world, z_self, hippocampal, e3, rng)
            elif condition == "GREEDY_k1":
                a = greedy_k1_select(z_world, e2, e3, residue_field)
            else:
                raise ValueError(f"Unknown condition: {condition}")

            flat_next, reward, done, info, obs_dict_next = env.step(a)
            ws = obs_dict_next["world_state"].float()
            bs = obs_dict_next["body_state"].float()
            total_steps += 1

            # Count DIRECT hazard contacts only (not proximity EMA)
            if info.get("transition_type") == "env_caused_hazard":
                harm_contacts += 1

            if done:
                break

    harm_rate = harm_contacts / max(total_steps, 1)
    return {
        "harm_rate": harm_rate,
        "n_harm_contacts": harm_contacts,
        "total_steps": total_steps,
    }


# ---------------------------------------------------------------------------
# E2-quality correlation diagnostic
# ---------------------------------------------------------------------------

def pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Pearson r between xs and ys. Returns 0.0 if constant input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom = (
        sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5 *
        sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    )
    if denom < 1e-9:
        return 0.0
    return num / denom


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    per_seed_results = {}
    e2_r2_vals = []
    rollout_harm_rates = []
    greedy_harm_rates = []
    greedy_contacts_per_seed = []

    for seed in SEEDS:
        print(f"[seed {seed}] Building modules...", flush=True)
        modules = build_modules(seed)

        print(f"[seed {seed}] Training...", flush=True)
        train_metrics = train_agent(modules, seed, dry_run=dry_run)
        e2_r2 = train_metrics["e2_world_r2"]
        e2_r2_vals.append(e2_r2)
        print(
            f"[seed {seed}] e2_world_r2={e2_r2:.4f} "
            f"residue_events={train_metrics['residue_harm_events']} "
            f"residue_total={train_metrics['residue_total']:.3f}",
            flush=True,
        )

        # Eval ROLLOUT_k5 with live modules
        print(f"[seed {seed}] Evaluating ROLLOUT_k5...", flush=True)
        rollout_res = eval_condition("ROLLOUT_k5", modules, seed, dry_run=dry_run)

        # Deep-copy for GREEDY_k1 (identical weights, fresh eval env via same seed+8888)
        print(f"[seed {seed}] Deep-copying for GREEDY_k1...", flush=True)
        modules_greedy = copy.deepcopy(modules)

        print(f"[seed {seed}] Evaluating GREEDY_k1...", flush=True)
        greedy_res = eval_condition("GREEDY_k1", modules_greedy, seed, dry_run=dry_run)

        rollout_harm_rates.append(rollout_res["harm_rate"])
        greedy_harm_rates.append(greedy_res["harm_rate"])
        greedy_contacts_per_seed.append(greedy_res["n_harm_contacts"])

        harm_adv = greedy_res["harm_rate"] - rollout_res["harm_rate"]
        c1_seed_pass = rollout_res["harm_rate"] < greedy_res["harm_rate"]
        print(
            f"[seed {seed}] ROLLOUT={rollout_res['harm_rate']:.4f} "
            f"GREEDY={greedy_res['harm_rate']:.4f} "
            f"advantage={harm_adv:+.4f} "
            f"C1={'PASS' if c1_seed_pass else 'FAIL'}",
            flush=True,
        )

        per_seed_results[str(seed)] = {
            "e2_world_r2": e2_r2,
            "train_metrics": train_metrics,
            "ROLLOUT_k5": rollout_res,
            "GREEDY_k1": greedy_res,
            "harm_advantage": harm_adv,
            "c1_direction_pass": c1_seed_pass,
        }

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    e2_r2_mean = sum(e2_r2_vals) / len(e2_r2_vals)
    harm_advantage_mean = sum(
        greedy_harm_rates[i] - rollout_harm_rates[i] for i in range(len(SEEDS))
    ) / len(SEEDS)

    # E2-quality correlation diagnostic (Pearson r: r2 vs advantage per seed)
    advantages = [greedy_harm_rates[i] - rollout_harm_rates[i] for i in range(len(SEEDS))]
    quality_advantage_corr = pearson_correlation(e2_r2_vals, advantages)

    print(
        f"\nSummary: e2_r2_mean={e2_r2_mean:.4f} "
        f"harm_advantage_mean={harm_advantage_mean:+.4f} "
        f"quality_advantage_corr={quality_advantage_corr:+.4f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Criteria
    # -----------------------------------------------------------------------
    # C1: direction correct in >=4/5 seeds
    c1_seeds_pass = sum(1 for s in SEEDS if per_seed_results[str(s)]["c1_direction_pass"])
    c1_pass = c1_seeds_pass >= 4

    # C2: mean harm advantage >= threshold
    c2_pass = harm_advantage_mean >= THRESH_C2_ADVANTAGE

    # C3: E2 quality gate
    c3_pass = e2_r2_mean >= THRESH_C3_E2_QUALITY

    # C4: data quality -- enough direct hazard contacts in GREEDY condition
    c4_pass = all(
        greedy_contacts_per_seed[i] >= THRESH_C4_MIN_CONTACTS
        for i in range(len(SEEDS))
    )

    # C5: E2-quality-gating diagnostic (positive correlation)
    c5_quality_gating = quality_advantage_corr >= 0.0

    criteria = {
        "C1_direction_4of5": c1_pass,
        "C2_mean_advantage": c2_pass,
        "C3_e2_quality_gate": c3_pass,
        "C4_data_quality": c4_pass,
        "C5_quality_gating_corr": c5_quality_gating,
    }
    print(f"Criteria: {criteria}", flush=True)

    # -----------------------------------------------------------------------
    # Outcome
    # -----------------------------------------------------------------------
    if not c3_pass or not c4_pass:
        outcome = "INCONCLUSIVE"
        evidence_direction = "inconclusive"
        decision = "inconclusive"
    elif c1_pass and c2_pass:
        outcome = "PASS"
        evidence_direction = "supports"
        decision = "retain_ree"
    elif not c1_pass and c5_quality_gating:
        # Rollout doesn't help yet, but quality-gating signal present
        outcome = "FAIL"
        evidence_direction = "mixed"
        decision = "hybridize"
    else:
        # Rollout doesn't help and no quality-gating signal
        outcome = "FAIL"
        evidence_direction = "weakens"
        decision = "retire_ree_claim"

    print(
        f"Outcome: {outcome}  direction: {evidence_direction}  "
        f"decision: {decision}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Result pack
    # -----------------------------------------------------------------------
    run_id = (
        f"{EXPERIMENT_TYPE}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3"
    )

    pack = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "discriminative_pair",
        "decision": decision,
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_C2_ADVANTAGE": THRESH_C2_ADVANTAGE,
            "THRESH_C3_E2_QUALITY": THRESH_C3_E2_QUALITY,
            "THRESH_C4_MIN_CONTACTS": THRESH_C4_MIN_CONTACTS,
        },
        "summary_metrics": {
            "harm_rate_ROLLOUT_mean": sum(rollout_harm_rates) / len(rollout_harm_rates),
            "harm_rate_GREEDY_mean": sum(greedy_harm_rates) / len(greedy_harm_rates),
            "harm_advantage_mean": harm_advantage_mean,
            "e2_world_r2_mean": e2_r2_mean,
            "quality_advantage_corr": quality_advantage_corr,
            "greedy_contacts_per_seed": greedy_contacts_per_seed,
        },
        "seeds": SEEDS,
        "config": {
            "env_size": ENV_SIZE,
            "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "hazard_harm": HAZARD_HARM,
            "proximity_harm_scale": PROXIMITY_HARM_SCALE,
            "world_dim": WORLD_DIM,
            "self_dim": SELF_DIM,
            "action_dim": ACTION_DIM,
            "action_object_dim": ACTION_OBJECT_DIM,
            "n_train_episodes": N_TRAIN_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_steps": EVAL_STEPS,
            "cem_horizon": CEM_HORIZON,
            "cem_candidates": CEM_CANDIDATES,
            "cem_iterations": CEM_ITERATIONS,
            "lr": LR,
        },
        "scenario": (
            "discriminative_pair: ROLLOUT_k5 (HippocampalModule CEM horizon=5, "
            f"{CEM_CANDIDATES} candidates, {CEM_ITERATIONS} CEM iters; "
            "E3.select() scores 5-step trajectories; residue field ACTIVE) "
            "vs GREEDY_k1 (1-step E2.world_forward + E3.harm_eval + "
            "residue.evaluate for each of 4 directional actions; argmin; "
            "residue field ACTIVE). Shared trained modules. "
            f"CausalGridWorld size={ENV_SIZE} hazards={NUM_HAZARDS} "
            f"hazard_harm={HAZARD_HARM} proximity_harm_scale=0.0 "
            "(point-source harm only, no proximity saturation). "
            "Harm counted as direct env_caused_hazard contacts only. "
            f"Train {N_TRAIN_EPISODES} eps x {STEPS_PER_EPISODE} steps. "
            f"Eval {N_EVAL_EPISODES} eps x {EVAL_STEPS} steps."
        ),
        "interpretation": (
            "PASS supports ARC-018: k=5 rollout via HippocampalModule CEM reduces "
            "direct hazard contacts vs 1-step greedy with same residue field active. "
            "Multi-step planning adds value when E2 quality is adequate. "
            "FAIL + C5_quality_gating=True: rollout benefit is E2-quality-gated; "
            "ARC-018 conditionally supported (need higher r2 floor). "
            "FAIL + C5_quality_gating=False: multi-step rollout does not add value "
            "even with adequate E2; ARC-018 weakened. "
            "INCONCLUSIVE: E2 quality gate not met or insufficient hazard contacts."
        ),
        "per_seed_results": per_seed_results,
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry_run] Result pack NOT written.", flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
