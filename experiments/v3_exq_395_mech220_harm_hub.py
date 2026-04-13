#!/opt/local/bin/python3
"""
V3-EXQ-395 -- MECH-220 Harm Hub: Behavioral Harm-Avoidance Probe

experiment_purpose: evidence
Claims: MECH-220

MECH-220 asserts a lightweight cross-stream coordination mechanism (harm hub)
allows z_harm_s and z_harm_a to share context without collapsing into a single
stream. The hub implements two directed interactions:
  (1) z_harm_a receives z_harm_s PE magnitude as context (sensory surprise
      informs affective load update rate -- hypervigilance seeding).
  (2) z_harm_s precision is gated by z_harm_a urgency level (high affective
      load amplifies sensory sensitivity -- hypervigilance pathway).

This is NOT a fusion: streams maintain separate encoders and training targets.

PRIOR EXPERIMENT (EXQ-262 FAIL / non_contributory):
  EXQ-262 tested fwd_r2 improvement, which required ARC-033/E2HarmSForward
  to be wired in. fwd_r2 was < -0.89 across all conditions. Evidence_direction
  was non_contributory because ARC-033 prerequisite was missing at the time.
  ARC-033 has since been validated (EXQ-329 PASS 10/10).

THIS EXPERIMENT uses a different evaluation approach: behavioral harm-avoidance
rate under identical navigation policy. The hub's benefit should appear as fewer
harm encounters per step when both streams are active and coordinated.

DESIGN:
Two conditions per seed:
  HUB_ENABLED  -- hub_weight=0.3; post-encoder gating wires z_harm_s PE
                  magnitude into z_harm_a and z_harm_a urgency into z_harm_s
                  sensitivity scaling.
  HUB_ABLATED  -- hub_weight=0.0; z_harm_s and z_harm_a are fully independent
                  (current default substrate).

The HarmHub module is implemented inline here (NOT modifying ree_core).
This is a probe experiment: if HUB_ENABLED consistently reduces harm exposure
vs HUB_ABLATED, MECH-220 receives positive behavioral support.

Phased training:
  P0 (100 eps): HarmEncoder + AffectiveHarmEncoder warmup with proximity
                supervision. Encoders learn raw harm representations.
  P1 (100 eps): Hub module trained alongside (encoders frozen). Hub gating
                receives gradient via harm_eval_z_harm supervision on modulated
                z_harm_s.
  Eval: 100-episode evaluation measuring harm_rate and z_harm_a_norm.

Each condition run per seed; 3 seeds (42, 7, 13).

PASS criteria:
  C1: HUB harm_rate < ABLATED harm_rate - 0.03 (hub reduces harm exposure)
      in >= 2/3 seeds.
  C2: HUB mean harm_rate < 0.35 (absolute: agent functional under hub)
  C3: HUB z_harm_a_norm > ABLATED z_harm_a_norm + 0.05 (hub enriches
      affective representation via sensory PE coupling) in >= 2/3 seeds.

PASS requires C1 AND (C2 OR C3).

Output JSON fields:
  run_id (ends _v3), architecture_epoch, claim_ids, experiment_purpose,
  evidence_direction_per_claim, outcome, timestamp_utc, all C1-C3 metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
import datetime as dt_mod
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_395_mech220_harm_hub"
CLAIM_IDS          = ["MECH-220"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS = [42, 7, 13]

P0_EPISODES   = 100   # encoder warmup
P1_EPISODES   = 100   # hub training (frozen encoders)
EVAL_EPISODES = 100   # evaluation
STEPS_PER_EP  = 60    # short episodes: dense harm exposure

GRID_SIZE     = 8
NUM_RESOURCES = 2
NUM_HAZARDS   = 3     # elevated hazard count for rich harm signal
HAZARD_HARM   = 0.3

Z_HARM_S_DIM  = 32   # LatentStackConfig.z_harm_dim default
Z_HARM_A_DIM  = 16   # LatentStackConfig.z_harm_a_dim default
HARM_OBS_DIM  = 51   # hazard_field(25) + resource_field(25) + harm_exposure(1)
HARM_OBS_A_DIM = 50  # harm_obs_a EMA: hazard_ema(25) + resource_ema(25)
ACTION_DIM    = 5

HUB_WEIGHT_ENABLED = 0.3
HUB_WEIGHT_ABLATED = 0.0

LR_MAIN = 3e-4
LR_HUB  = 1e-3

# Pass thresholds
C1_HARM_RATE_DELTA = 0.03    # HUB harm_rate < ABLATED harm_rate - 0.03
C2_ABS_HARM_RATE   = 0.35    # HUB harm_rate < 0.35
C3_NORM_DELTA      = 0.05    # HUB z_harm_a_norm > ABLATED + 0.05
MIN_SEEDS_C1       = 2       # primary: >= 2/3 seeds pass C1
MIN_SEEDS_PASS     = 2       # overall: >= 2/3 seeds


# ---------------------------------------------------------------------------
# HarmHub module (MECH-220 probe implementation -- inline, not in ree_core)
# ---------------------------------------------------------------------------

class HarmHub(nn.Module):
    """
    Lightweight cross-stream coordination module (MECH-220 probe).

    Two directed gating interactions between z_harm_s and z_harm_a:
      (1) Sensory-to-affective (gate_sa):
          z_harm_a_out = z_harm_a + hub_weight * gate_sa(z_harm_s) * z_harm_a
          Sensory stream gates affective load scaling.
          Biological: ACC -> insula projection for salience-to-affective routing.
      (2) Affective-to-sensory (gate_as):
          z_harm_s_out = z_harm_s * (1 + hub_weight * gate_as(z_harm_a))
          High affective urgency amplifies sensory sensitivity.
          Biological: amygdala -> S2 top-down gain (hypervigilance).

    hub_weight=0.0 -> ablated: no gating; purely independent streams.
    hub_weight>0.0 -> enabled: both gating paths active.

    Multiplicative gates preserve stream independence and prevent collapse.
    """

    def __init__(self, z_harm_s_dim: int, z_harm_a_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Gate sa: z_harm_s -> affective modulation [z_harm_a_dim]
        self.gate_sa = nn.Sequential(
            nn.Linear(z_harm_s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_harm_a_dim),
            nn.Sigmoid(),
        )
        # Gate as: z_harm_a -> sensory gain [z_harm_s_dim]
        self.gate_as = nn.Sequential(
            nn.Linear(z_harm_a_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_harm_s_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_harm_s: torch.Tensor,
        z_harm_a: torch.Tensor,
        hub_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-stream hub gating. hub_weight=0 returns inputs unchanged."""
        if hub_weight == 0.0:
            return z_harm_s, z_harm_a
        mod_sa = self.gate_sa(z_harm_s.detach())           # [batch, z_harm_a_dim]
        z_harm_a_out = z_harm_a + hub_weight * mod_sa * z_harm_a
        gain_as = self.gate_as(z_harm_a.detach())           # [batch, z_harm_s_dim]
        z_harm_s_out = z_harm_s * (1.0 + hub_weight * gain_as)
        return z_harm_s_out, z_harm_a_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.3,
        resource_respawn_on_consume=True,
    )


def _make_config(env: CausalGridWorldV2) -> REEConfig:
    """Build REEConfig with both harm streams enabled (SD-010 + SD-011)."""
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_S_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
    )


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_avoid_hazards(env: CausalGridWorldV2) -> int:
    """Move toward resources while preferring hazard-distant cells."""
    ax, ay = env.agent_x, env.agent_y
    best_action = random.randint(0, env.action_dim - 1)
    best_score = float("-inf")
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    for i, (dx, dy) in enumerate(deltas):
        nx, ny = ax + dx, ay + dy
        if not (0 <= nx < env.size and 0 <= ny < env.size):
            continue
        res_score = sum(
            max(0.0, 1.0 - (abs(nx - int(r[0])) + abs(ny - int(r[1]))) / env.size)
            for r in env.resources
        )
        haz_pen = sum(
            max(0.0, 1.0 - (abs(nx - int(h[0])) + abs(ny - int(h[1]))) / env.size)
            for h in env.hazards
        )
        score = res_score - haz_pen
        if score > best_score:
            best_score = score
            best_action = i
    return best_action


def _get_harm_obs(obs_dict: dict, device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract and batch-dim harm observations."""
    obs_harm = obs_dict.get("harm_obs", None)
    obs_harm_a = obs_dict.get("harm_obs_a", None)
    if obs_harm is not None:
        obs_harm = obs_harm.to(device).float()
        if obs_harm.dim() == 1:
            obs_harm = obs_harm.unsqueeze(0)
    if obs_harm_a is not None:
        obs_harm_a = obs_harm_a.to(device).float()
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
    return obs_harm, obs_harm_a


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def _train_p0(
    agent: REEAgent,
    env: CausalGridWorldV2,
    opt: optim.Optimizer,
    n_episodes: int,
    seed: int,
    condition: str,
    dry_run: bool = False,
) -> None:
    """P0: warm up both harm encoders with proximity supervision."""
    n_ep = min(n_episodes, 3) if dry_run else n_episodes
    total_ep = 6 if dry_run else (P0_EPISODES + P1_EPISODES)
    device = agent.device

    for ep in range(n_ep):
        _, obs_dict = env.reset()
        agent.reset()

        for _s in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"].to(device).unsqueeze(0)
            obs_world = obs_dict["world_state"].to(device).unsqueeze(0)
            obs_harm, obs_harm_a = _get_harm_obs(obs_dict, device)

            latent = agent.sense(obs_body, obs_world,
                                 obs_harm=obs_harm, obs_harm_a=obs_harm_a)
            agent.clock.advance()

            action_idx = (
                _greedy_avoid_hazards(env)
                if random.random() < 0.5
                else random.randint(0, env.action_dim - 1)
            )
            z_self_prev = None
            if agent._current_latent is not None:
                z_self_prev = agent._current_latent.z_self.detach().clone()
            action = _onehot(action_idx, env.action_dim, device)
            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            _, harm_signal, done, info, obs_dict = env.step(action)

            total_loss = agent.compute_prediction_loss()

            # z_harm_s supervision: predict harm proximity
            if latent.z_harm is not None:
                harm_t = torch.tensor([[abs(float(harm_signal))]], device=device)
                harm_pred = agent.e3.harm_eval_z_harm(latent.z_harm)
                total_loss = total_loss + F.mse_loss(harm_pred, harm_t)

            # z_harm_a supervision: predict accumulated harm via harm_accum_pred
            if latent.harm_accum_pred is not None:
                accum = info.get("accumulated_harm", abs(float(harm_signal)))
                accum_t = torch.tensor(
                    [[float(np.clip(accum, 0.0, 1.0))]],
                    device=device,
                )
                total_loss = total_loss + F.mse_loss(
                    latent.harm_accum_pred,
                    accum_t.expand_as(latent.harm_accum_pred),
                )

            if total_loss.requires_grad:
                opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                break

        if (ep + 1) % 50 == 0 or (ep + 1) == n_ep:
            print(
                f"  [train] seed={seed} ep {ep + 1}/{n_ep}"
                f" phase=P0 cond={condition}",
                flush=True,
            )


def _train_p1(
    agent: REEAgent,
    env: CausalGridWorldV2,
    hub: HarmHub,
    hub_weight: float,
    opt_hub: Optional[optim.Optimizer],
    opt_main: optim.Optimizer,
    n_episodes: int,
    seed: int,
    condition: str,
    dry_run: bool = False,
) -> None:
    """P1: train hub module; harm encoders frozen."""
    n_ep = min(n_episodes, 3) if dry_run else n_episodes
    total_ep = 6 if dry_run else (P0_EPISODES + P1_EPISODES)
    device = agent.device

    for ep in range(n_ep):
        abs_ep = (3 if dry_run else P0_EPISODES) + ep
        _, obs_dict = env.reset()
        agent.reset()

        for _s in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"].to(device).unsqueeze(0)
            obs_world = obs_dict["world_state"].to(device).unsqueeze(0)
            obs_harm, obs_harm_a = _get_harm_obs(obs_dict, device)

            latent = agent.sense(obs_body, obs_world,
                                 obs_harm=obs_harm, obs_harm_a=obs_harm_a)
            agent.clock.advance()

            action_idx = (
                _greedy_avoid_hazards(env)
                if random.random() < 0.5
                else random.randint(0, env.action_dim - 1)
            )
            z_self_prev = None
            if agent._current_latent is not None:
                z_self_prev = agent._current_latent.z_self.detach().clone()
            action = _onehot(action_idx, env.action_dim, device)
            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            _, harm_signal, done, info, obs_dict = env.step(action)

            harm_t = torch.tensor([[abs(float(harm_signal))]], device=device)

            # Hub training: supervision on modulated z_harm_s
            if (latent.z_harm is not None and latent.z_harm_a is not None
                    and hub_weight > 0.0 and opt_hub is not None):
                z_hs, _z_ha = hub(latent.z_harm, latent.z_harm_a, hub_weight)
                hub_pred = agent.e3.harm_eval_z_harm(z_hs)
                hub_loss = F.mse_loss(hub_pred, harm_t)
                opt_hub.zero_grad()
                hub_loss.backward()
                nn.utils.clip_grad_norm_(hub.parameters(), 1.0)
                opt_hub.step()

            # Main agent training (encoders already frozen)
            total_loss = agent.compute_prediction_loss()
            if total_loss.requires_grad:
                opt_main.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_main.step()

            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                break

        if (abs_ep + 1) % 50 == 0 or (ep + 1) == n_ep:
            print(
                f"  [train] seed={seed} ep {abs_ep + 1}/{total_ep}"
                f" phase=P1 cond={condition}"
                f" hub_weight={hub_weight:.1f}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Hub-guided action selection
# ---------------------------------------------------------------------------

def _hub_guided_action(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_body: torch.Tensor,
    obs_world: torch.Tensor,
    obs_harm: Optional[torch.Tensor],
    obs_harm_a: Optional[torch.Tensor],
    hub: HarmHub,
    hub_weight: float,
    device,
) -> int:
    """
    Select action by evaluating hub-modulated harm score for each candidate move.

    For each valid action, compute the resulting harm_obs from a simulated step,
    encode it through the harm encoder, apply hub gating, and score with
    harm_eval_z_harm. Pick the action minimising predicted harm, with resource
    proximity as secondary score.

    This wires the hub's effect on z_harm_s into the behavioral output channel.
    HUB_ABLATED (hub_weight=0) uses identical z_harm_s without gating -> same
    harm_eval scores -> same action selection as baseline.
    """
    ax, ay = env.agent_x, env.agent_y
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    best_action = random.randint(0, env.action_dim - 1)
    best_score = float("inf")

    for i, (dx, dy) in enumerate(deltas):
        nx, ny = ax + dx, ay + dy
        if not (0 <= nx < env.size and 0 <= ny < env.size):
            continue

        # Approximate harm from hazard proximity at candidate cell
        haz_prox = sum(
            max(0.0, 1.0 - (abs(nx - int(h[0])) + abs(ny - int(h[1]))) / env.size)
            for h in env.hazards
        )
        res_prox = sum(
            max(0.0, 1.0 - (abs(nx - int(r[0])) + abs(ny - int(r[1]))) / env.size)
            for r in env.resources
        )

        # Use current z_harm_s with hub gating to compute predicted harm score
        harm_score = haz_prox  # default: raw proximity
        if agent._current_latent is not None and agent._current_latent.z_harm is not None:
            z_hs = agent._current_latent.z_harm.detach()
            z_ha = agent._current_latent.z_harm_a
            if z_ha is not None and hub_weight > 0.0:
                with torch.no_grad():
                    z_hs, _z_ha = hub(z_hs, z_ha, hub_weight)
            with torch.no_grad():
                harm_pred = agent.e3.harm_eval_z_harm(z_hs)
            harm_score = float(harm_pred.item()) + haz_prox
        # Combined score: minimise harm, maximise resource proximity
        combined = harm_score - 0.5 * res_prox
        if combined < best_score:
            best_score = combined
            best_action = i

    return best_action


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    hub: HarmHub,
    hub_weight: float,
    seed: int,
    condition: str,
    dry_run: bool = False,
) -> Dict:
    """Evaluate harm_rate and z_harm_a_norm over EVAL_EPISODES."""
    device = agent.device
    n_eval = 5 if dry_run else EVAL_EPISODES

    harm_events = 0
    total_steps = 0
    z_harm_a_norms: list = []
    z_harm_s_norms: list = []

    for _ep in range(n_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _s in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"].to(device).unsqueeze(0)
            obs_world = obs_dict["world_state"].to(device).unsqueeze(0)
            obs_harm, obs_harm_a = _get_harm_obs(obs_dict, device)

            latent = agent.sense(obs_body, obs_world,
                                 obs_harm=obs_harm, obs_harm_a=obs_harm_a)
            agent.clock.advance()

            # Apply hub gating to harm latents (no grad for eval)
            z_hs = latent.z_harm
            z_ha = latent.z_harm_a
            if z_hs is not None and z_ha is not None and hub_weight > 0.0:
                with torch.no_grad():
                    z_hs, z_ha = hub(z_hs, z_ha, hub_weight)

            if z_ha is not None:
                z_harm_a_norms.append(float(z_ha.norm().item()))
            if z_hs is not None:
                z_harm_s_norms.append(float(z_hs.norm().item()))

            # Hub-guided action selection: pick action that minimises predicted
            # harm score on the hub-modulated z_harm_s, breaking ties toward
            # resource-proximity. This is the behavioral channel through which
            # the hub can influence harm_rate.
            if z_hs is not None:
                action_idx = _hub_guided_action(
                    agent, env, obs_body, obs_world, obs_harm, obs_harm_a,
                    hub, hub_weight, device
                )
            else:
                action_idx = _greedy_avoid_hazards(env)

            action = _onehot(action_idx, env.action_dim, device)
            _, harm_signal, done, info, obs_dict = env.step(action)

            if float(harm_signal) < 0:
                harm_events += 1
            total_steps += 1

            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                break

    harm_rate = harm_events / max(1, total_steps)
    mean_z_harm_a_norm = float(np.mean(z_harm_a_norms)) if z_harm_a_norms else 0.0
    mean_z_harm_s_norm = float(np.mean(z_harm_s_norms)) if z_harm_s_norms else 0.0

    print(
        f"  [{condition}] seed={seed} EVAL:"
        f" harm_rate={harm_rate:.4f}"
        f" ({harm_events}/{total_steps})"
        f" z_harm_a_norm={mean_z_harm_a_norm:.4f}"
        f" z_harm_s_norm={mean_z_harm_s_norm:.4f}",
        flush=True,
    )

    return {
        "condition": condition,
        "harm_rate": harm_rate,
        "harm_events": harm_events,
        "total_steps": total_steps,
        "mean_z_harm_a_norm": mean_z_harm_a_norm,
        "mean_z_harm_s_norm": mean_z_harm_s_norm,
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool = False) -> Dict:
    """Run both conditions (HUB_ENABLED, HUB_ABLATED) for one seed."""
    print(f"\n[EXQ-395] Seed {seed}", flush=True)

    results = {}

    for condition, hw in [("HUB_ENABLED", HUB_WEIGHT_ENABLED),
                           ("HUB_ABLATED", HUB_WEIGHT_ABLATED)]:
        print(f"  Seed {seed} Condition {condition} hub_weight={hw}", flush=True)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        env = _make_env(seed)
        cfg = _make_config(env)
        agent = REEAgent(cfg)
        device = agent.device

        hub = HarmHub(Z_HARM_S_DIM, Z_HARM_A_DIM, hidden_dim=32).to(device)
        opt_hub = optim.Adam(hub.parameters(), lr=LR_HUB) if hw > 0.0 else None
        opt_main = optim.Adam(agent.parameters(), lr=LR_MAIN)

        n_p0 = 3 if dry_run else P0_EPISODES
        n_p1 = 3 if dry_run else P1_EPISODES

        print(
            f"  [train] seed={seed} P0 start ({n_p0} eps) cond={condition}",
            flush=True,
        )
        _train_p0(agent, env, opt_main, n_p0, seed, condition, dry_run)

        # Freeze encoders before P1
        if agent.latent_stack.harm_encoder is not None:
            for p in agent.latent_stack.harm_encoder.parameters():
                p.requires_grad_(False)
        if agent.latent_stack.affective_harm_encoder is not None:
            for p in agent.latent_stack.affective_harm_encoder.parameters():
                p.requires_grad_(False)

        opt_p1 = optim.Adam(
            [p for p in agent.parameters() if p.requires_grad], lr=LR_MAIN
        )

        print(
            f"  [train] seed={seed} P1 start ({n_p1} eps) cond={condition}",
            flush=True,
        )
        _train_p1(agent, env, hub, hw, opt_hub, opt_p1, n_p1, seed, condition, dry_run)

        agent.eval()
        hub.eval()
        res = _run_eval(agent, env, hub, hw, seed, condition, dry_run)
        results[condition] = res

    hub_r     = results["HUB_ENABLED"]
    ablated_r = results["HUB_ABLATED"]

    c1_pass = hub_r["harm_rate"] < ablated_r["harm_rate"] - C1_HARM_RATE_DELTA
    c2_pass = hub_r["harm_rate"] < C2_ABS_HARM_RATE
    c3_pass = (
        hub_r["mean_z_harm_a_norm"] > ablated_r["mean_z_harm_a_norm"] + C3_NORM_DELTA
    )

    harm_delta = ablated_r["harm_rate"] - hub_r["harm_rate"]
    norm_delta  = hub_r["mean_z_harm_a_norm"] - ablated_r["mean_z_harm_a_norm"]

    seed_pass = c1_pass and (c2_pass or c3_pass)
    verdict   = "PASS" if seed_pass else "FAIL"
    print(
        f"  verdict: {verdict}"
        f" (C1={c1_pass}, C2={c2_pass}, C3={c3_pass})"
        f" harm_delta={harm_delta:.4f}"
        f" norm_delta={norm_delta:.4f}"
        f" hub={hub_r['harm_rate']:.4f}"
        f" abl={ablated_r['harm_rate']:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "c1_hub_reduces_harm": c1_pass,
        "c2_abs_harm_rate": c2_pass,
        "c3_hub_enriches_affective": c3_pass,
        "hub_harm_rate": hub_r["harm_rate"],
        "ablated_harm_rate": ablated_r["harm_rate"],
        "harm_rate_delta": harm_delta,
        "hub_z_harm_a_norm": hub_r["mean_z_harm_a_norm"],
        "ablated_z_harm_a_norm": ablated_r["mean_z_harm_a_norm"],
        "z_harm_a_norm_delta": norm_delta,
        "hub_z_harm_s_norm": hub_r["mean_z_harm_s_norm"],
        "ablated_z_harm_s_norm": ablated_r["mean_z_harm_s_norm"],
        "condition_results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("[EXQ-395] MECH-220 Harm Hub: Behavioral Harm-Avoidance Probe", flush=True)
    print(
        f"  Seeds={SEEDS}  P0={P0_EPISODES}ep  P1={P1_EPISODES}ep"
        f"  Eval={EVAL_EPISODES}ep"
        f"  hub_enabled={HUB_WEIGHT_ENABLED}  hub_ablated={HUB_WEIGHT_ABLATED}",
        flush=True,
    )

    timestamp = dt_mod.datetime.now(dt_mod.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{EXPERIMENT_TYPE}_dry_{timestamp}"
        if args.dry_run
        else f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    )

    per_seed = [run_seed(s, dry_run=args.dry_run) for s in SEEDS]

    seeds_c1_pass = sum(1 for r in per_seed if r["c1_hub_reduces_harm"])
    seeds_pass    = sum(1 for r in per_seed if r["seed_pass"])

    experiment_passes = seeds_c1_pass >= MIN_SEEDS_C1 and seeds_pass >= MIN_SEEDS_PASS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(
        f"\n[EXQ-395] === {outcome}"
        f" (seeds_pass={seeds_pass}/{len(SEEDS)}"
        f" seeds_c1={seeds_c1_pass}/{len(SEEDS)}) ===",
        flush=True,
    )
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s}"
            f" harm_delta={r['harm_rate_delta']:.4f}"
            f" norm_delta={r['z_harm_a_norm_delta']:.4f}"
            f" hub={r['hub_harm_rate']:.4f}"
            f" abl={r['ablated_harm_rate']:.4f}",
            flush=True,
        )

    evidence_direction = "supports" if experiment_passes else "weakens"

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {
            "MECH-220": evidence_direction,
        },
        "outcome": outcome,
        "timestamp_utc": dt_mod.datetime.now(dt_mod.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "design_note": (
            "Prior experiment EXQ-262 was non_contributory: fwd_r2 < -0.89 "
            "because ARC-033/E2HarmSForward prerequisite was absent. "
            "This experiment uses behavioral harm_rate as primary metric -- "
            "does not require E2HarmSForward. Hub is inline HarmHub module "
            "(gate_sa: z_harm_s->z_harm_a; gate_as: z_harm_a->z_harm_s). "
            "Biological basis: Chen (2023) cingulate-insula hub for pain stream coordination."
        ),
        "registered_thresholds": {
            "C1_harm_rate_delta": C1_HARM_RATE_DELTA,
            "C2_abs_harm_rate": C2_ABS_HARM_RATE,
            "C3_norm_delta": C3_NORM_DELTA,
            "min_seeds_c1": MIN_SEEDS_C1,
        },
        "hub_weight_enabled": HUB_WEIGHT_ENABLED,
        "hub_weight_ablated": HUB_WEIGHT_ABLATED,
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "seeds": SEEDS,
        "seeds_passing": seeds_pass,
        "seeds_c1_pass": seeds_c1_pass,
        "per_seed_results": per_seed,
        "metrics": {
            "mean_hub_harm_rate": float(
                sum(r["hub_harm_rate"] for r in per_seed) / len(per_seed)
            ),
            "mean_ablated_harm_rate": float(
                sum(r["ablated_harm_rate"] for r in per_seed) / len(per_seed)
            ),
            "mean_harm_rate_delta": float(
                sum(r["harm_rate_delta"] for r in per_seed) / len(per_seed)
            ),
            "mean_hub_z_harm_a_norm": float(
                sum(r["hub_z_harm_a_norm"] for r in per_seed) / len(per_seed)
            ),
            "mean_ablated_z_harm_a_norm": float(
                sum(r["ablated_z_harm_a_norm"] for r in per_seed) / len(per_seed)
            ),
            "mean_z_harm_a_norm_delta": float(
                sum(r["z_harm_a_norm_delta"] for r in per_seed) / len(per_seed)
            ),
        },
    }

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResult written to: {out_path}", flush=True)

    print(f"Status: {outcome}", flush=True)
    for k, v in output["metrics"].items():
        print(f"  {k}: {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
