#!/opt/local/bin/python3
"""
V3-EXQ-433a: SD-029 Single-Pass Comparator -- Scripted-Eval (Fixed-Script Protocol)

experiment_purpose: evidence
supersedes: v3_exq_433_sd029_eventcond_comparator (event-distribution collapse)

Scientific question: Same as EXQ-433. Does the z_harm_s single-pass comparator
(residual = z_harm_s_observed - E2_harm_s(z_harm_s_{t-1}, a_actual)) exhibit the
Shergill/Blakemore partial-attenuation signature?

Why a successor: EXQ-433 FAIL was traced to event-distribution collapse driven by
bimodal policy modes (3/4 seeds either pure-exploit or pure-avoid; only seed 7
balanced). C4 controller could not escape because the bottleneck is policy mode,
not episode length. Reclassified non_contributory.

Fix (Option C, fixed-script protocol -- the gold-standard from Blakemore 1998 /
Shergill 2003 biology that grounds SD-029): training is unchanged from EXQ-433
(P0 + P1 with SD-013 interventional fraction=0.5). Evaluation replaces the
behavioural-rollout density controller with a *scripted trial harness*: each trial
deterministically constructs one event type via env.reset_to() + a forced action.
N trials per event type are guaranteed by construction; behavioural sample frequency
is decoupled from evaluation balance.

Methodological note: the env's "agent_caused" vs "env_caused" labels distinguish the
provenance of the harm source (agent-contaminated cell vs fresh env hazard). All
actions in the scripted protocol are forced by the harness; the comparator residual
asks whether the trained forward model predicts the consequence of the action it
is shown -- which is the Shergill comparator question.

Claim mapping (per SD-029 registration 2026-04-18):
  C1 forward_r2(E2_harm_s) >= 0.9 on reafferent z_harm_s stream
  C2 Self/external residual attenuation ratio in [0.3, 0.7]
     (Shergill partial-attenuation pattern, not a binary gate)
  C3 SNR on approach-to-harm events > 3.0
  C4 Per-event-type trial count >= 20 BY CONSTRUCTION (no density controller)

PASS: C1 AND C2 AND C3 AND C4 across >= 3/4 seeds.

Phased training (unchanged from EXQ-433):
  P0 (80 ep):  HarmEncoder + agent encoder warmup
  P1 (100 ep): E2HarmSForward interventional training on frozen z_harm_s

Evaluation (NEW):
  P2_scripted: build TRIALS_PER_TYPE trials of each event type via reset_to() +
               forced action; collect single-pass residuals; no density controller.

Single condition TRAINED at 4 seeds.

Claims tested: SD-029 (primary), MECH-256 (general comparator instantiation).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_433a_sd029_eventcond_comparator_scripted"
CLAIM_IDS          = ["SD-029", "MECH-256"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS      = [42, 7, 13, 91]
CONDITIONS = ["TRAINED"]

P0_EPISODES   = 80
P1_EPISODES   = 100
STEPS_PER_EP  = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 4
HAZARD_HARM   = 0.3

Z_HARM_DIM   = 32
HARM_OBS_DIM = 51
ACTION_DIM   = 5

LR_AGENT    = 3e-4
LR_HARM_FWD = 5e-4

REPLAY_BUF_MAX = 5000
BATCH_SIZE     = 32

# SD-013 interventional settings
INTERVENTIONAL_FRACTION = 0.5
INTERVENTIONAL_MARGIN   = 0.1

# Pass thresholds (per SD-029 criteria sketch -- unchanged from EXQ-433)
C1_FORWARD_R2_THRESH      = 0.9
C2_RATIO_LOW              = 0.3
C2_RATIO_HIGH             = 0.7
C3_APPROACH_SNR_THRESH    = 3.0
C4_MIN_EVENTS_PER_TYPE    = 20
MIN_SEEDS_PASS            = 3   # 3 of 4

# Scripted eval: trials per event type per seed (>= C4 floor).
# 30 trials gives slack above the C4 minimum of 20 and stable mean estimates.
TRIALS_PER_TYPE = 30

EVENT_TYPES = [
    "env_caused_hazard",      # externally-caused (fresh env hazard)
    "agent_caused_hazard",    # self-caused (agent steps into contaminated cell)
    "hazard_approach",        # approach-to-harm (proxy gradient, no contact)
    "none",                   # baseline (empty cell, no proximity)
]

# Action indices (must match CausalGridWorldV2.ACTIONS):
#   0=up(-x,0)  1=down(+x,0)  2=left(0,-x)  3=right(0,+x)  4=stay
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTION_STAY  = 4

DRY_RUN_EPISODES   = 3
DRY_RUN_STEPS      = 20
DRY_RUN_TRIALS     = 3


# ---------------------------------------------------------------------------
# Factories (unchanged from EXQ-433)
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
        proximity_harm_scale=0.1,
        proximity_approach_threshold=0.2,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_event_classifier=True,
    )
    return REEAgent(config)


def _make_harm_fwd(device) -> E2HarmSForward:
    cfg = E2HarmSConfig(
        use_e2_harm_s_forward=True,
        z_harm_dim=Z_HARM_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=128,
        use_interventional=True,
        interventional_fraction=INTERVENTIONAL_FRACTION,
        interventional_margin=INTERVENTIONAL_MARGIN,
    )
    return E2HarmSForward(cfg).to(device)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_to_device(obs_dict: Dict, device) -> Tuple:
    obs_body  = obs_dict["body_state"].to(device)
    obs_world = obs_dict["world_state"].to(device)
    obs_harm  = obs_dict.get("harm_obs", None)
    if obs_harm is not None:
        obs_harm = (obs_harm.to(device).unsqueeze(0)
                    if obs_harm.dim() == 1 else obs_harm.to(device))
    return obs_body, obs_world, obs_harm


# ---------------------------------------------------------------------------
# Training (P0 + P1) -- structurally identical to EXQ-433
# ---------------------------------------------------------------------------

def run_training(
    seed: int,
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_fwd: E2HarmSForward,
    dry_run: bool,
    total_training_episodes: int,
) -> float:
    """P0 + P1 training. Returns forward_r2 computed on the tail of P1."""
    total_p0  = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1  = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1

    agent_opt    = optim.Adam(list(agent.parameters()), lr=LR_AGENT)
    harm_fwd_opt = optim.Adam(harm_fwd.parameters(), lr=LR_HARM_FWD)
    replay_buf: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
    device     = agent.device
    prev_ttype = "none"

    eval_preds:   List[float] = []
    eval_targets: List[float] = []

    z_harm_s_prev:   Optional[torch.Tensor] = None
    action_prev_idx: Optional[int]          = None

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()
        z_harm_s_prev   = None
        action_prev_idx = None

        phase = "P0" if ep < total_p0 else "P1"
        in_p1      = (phase == "P1")
        p1_eval_ep = in_p1 and ep >= (total_p0 + total_p1 - 20)

        for step in range(steps_per):
            obs_body, obs_world, obs_harm = _obs_to_device(obs_dict, device)

            z_self_prev_t: Optional[torch.Tensor] = None
            if agent._current_latent is not None:
                z_self_prev_t = agent._current_latent.z_self.detach().clone()

            latent     = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks      = agent.clock.advance()
            e1_prior   = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", True)
                else torch.zeros(1, 32, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            z_harm_s_now = latent.z_harm
            if z_harm_s_now is not None:
                z_hs_d = z_harm_s_now.detach().clone()
                if z_harm_s_prev is not None and action_prev_idx is not None:
                    replay_buf.append((z_harm_s_prev, action_prev_idx, z_hs_d))
                    if len(replay_buf) > REPLAY_BUF_MAX:
                        replay_buf = replay_buf[-REPLAY_BUF_MAX:]

            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            agent.update_residue(float(harm_signal))
            if z_self_prev_t is not None:
                agent.record_transition(z_self_prev_t, action, latent.z_self.detach())

            if phase == "P0":
                agent_opt.zero_grad()
                loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    loss = loss + agent.compute_resource_proximity_loss(rp_t, latent)
                lat2 = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                loss = loss + agent.compute_event_contrastive_loss(prev_ttype, lat2)
                if loss.requires_grad:
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    agent_opt.step()

            if in_p1 and len(replay_buf) >= BATCH_SIZE:
                batch_idx  = random.sample(range(len(replay_buf)), BATCH_SIZE)
                z_s_b  = torch.cat([replay_buf[i][0] for i in batch_idx], dim=0).detach()
                a_idxs = [replay_buf[i][1] for i in batch_idx]
                z_s1_b = torch.cat([replay_buf[i][2] for i in batch_idx], dim=0).detach()

                a_b = torch.zeros(BATCH_SIZE, ACTION_DIM, device=device)
                for bi, ai in enumerate(a_idxs):
                    a_b[bi, ai] = 1.0

                harm_fwd_opt.zero_grad()
                z_pred = harm_fwd(z_s_b, a_b)
                fwd_loss = harm_fwd.compute_loss(z_pred, z_s1_b)

                if random.random() < INTERVENTIONAL_FRACTION:
                    a_cf_b = torch.zeros(BATCH_SIZE, ACTION_DIM, device=device)
                    for bi, ai in enumerate(a_idxs):
                        cfs = [j for j in range(ACTION_DIM) if j != ai]
                        a_cf_b[bi, random.choice(cfs)] = 1.0
                    fwd_loss = fwd_loss + harm_fwd.compute_interventional_loss(
                        z_s_b, a_b, a_cf_b
                    )

                fwd_loss.backward()
                harm_fwd_opt.step()

            if p1_eval_ep and z_harm_s_now is not None and z_harm_s_prev is not None:
                with torch.no_grad():
                    a_act = _onehot(action_idx, ACTION_DIM, device)
                    z_pred_act = harm_fwd(z_harm_s_prev.detach(), a_act)
                for d in range(Z_HARM_DIM):
                    eval_preds.append(float(z_pred_act[0, d].item()))
                    eval_targets.append(float(z_harm_s_now.detach()[0, d].item()))

            z_harm_s_prev   = z_harm_s_now.detach().clone() if z_harm_s_now is not None else None
            action_prev_idx = action_idx
            prev_ttype      = ttype
            obs_dict        = obs_dict_next

            if done:
                break

        if (ep + 1) % 50 == 0 or ep + 1 == total_eps:
            print(
                f"  [train] seed={seed} ep {ep+1}/{total_training_episodes} "
                f"phase={phase} replay={len(replay_buf)}",
                flush=True,
            )

    forward_r2 = 0.0
    if len(eval_preds) >= 10:
        try:
            tgt    = np.array(eval_targets)
            prd    = np.array(eval_preds)
            ss_res = float(np.sum((tgt - prd) ** 2))
            ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
            forward_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        except Exception:
            pass

    return forward_r2


# ---------------------------------------------------------------------------
# Scripted-eval trial builders
# ---------------------------------------------------------------------------
# Each builder returns (agent_pos, hazard_positions, resource_positions,
# extra_setup, forced_action, intended_event_type).
# extra_setup is a callable applied to env after reset_to() (e.g. set
# a contaminated cell). It may also return None.

def _trial_env_caused(rng: random.Random, size: int) -> Dict:
    """Agent adjacent to fresh env hazard; one-step into it."""
    while True:
        ax = rng.randint(2, size - 3)
        ay = rng.randint(2, size - 3)
        # Place hazard one step DOWN from agent
        hx, hy = ax + 1, ay
        # Filler hazards far away (proximity=0 at agent neighbourhood)
        # Place at corners to keep hazard_field at agent_pos+target near zero
        fillers = [(1, 1), (1, size - 2), (size - 2, 1)]
        # Avoid overlap with target hazard
        fillers = [f for f in fillers if f != (hx, hy) and f != (ax, ay)]
        return {
            "agent_pos":     (ax, ay),
            "hazards":       [(hx, hy)] + fillers[:NUM_HAZARDS - 1],
            "resources":     [],
            "post_reset":    None,
            "forced_action": ACTION_DOWN,
            "intended":      "env_caused_hazard",
        }


def _trial_agent_caused(rng: random.Random, size: int) -> Dict:
    """Agent adjacent to a pre-contaminated cell; one step into it."""
    while True:
        ax = rng.randint(2, size - 3)
        ay = rng.randint(2, size - 3)
        cx, cy = ax + 1, ay
        # Filler hazards far away
        fillers = [(1, 1), (1, size - 2), (size - 2, 1), (size - 2, size - 2)]
        fillers = [f for f in fillers if f not in [(cx, cy), (ax, ay)]]

        def post_reset(env, _cx=cx, _cy=cy):
            # Mark the target cell as contaminated (overrides any hazard there).
            env.grid[_cx, _cy] = env.ENTITY_TYPES["contaminated"]

        return {
            "agent_pos":     (ax, ay),
            "hazards":       fillers[:NUM_HAZARDS],
            "resources":     [],
            "post_reset":    post_reset,
            "forced_action": ACTION_DOWN,
            "intended":      "agent_caused_hazard",
        }


def _trial_hazard_approach(rng: random.Random, size: int) -> Dict:
    """Agent steps from outside-proxy region into proxy-gradient region.

    Hazard placed 2 cells away from target cell so target_cell hazard_field
    is above proximity_approach_threshold (0.2) but the cell itself is empty.
    """
    while True:
        ax = rng.randint(1, size - 4)
        ay = rng.randint(2, size - 3)
        hx, hy = ax + 3, ay  # 2 cells past the target step
        if not (1 <= hx <= size - 2):
            continue
        # Ensure no other hazards within proximity of agent or target
        fillers: List[Tuple[int, int]] = []
        for cand in [(1, size - 2), (size - 2, 1), (size - 2, size - 2)]:
            if abs(cand[0] - (ax + 1)) + abs(cand[1] - ay) > 4 and cand != (hx, hy):
                fillers.append(cand)
            if len(fillers) >= NUM_HAZARDS - 1:
                break
        return {
            "agent_pos":     (ax, ay),
            "hazards":       [(hx, hy)] + fillers,
            "resources":     [],
            "post_reset":    None,
            "forced_action": ACTION_DOWN,
            "intended":      "hazard_approach",
        }


def _trial_none(rng: random.Random, size: int) -> Dict:
    """Empty interior; all hazards in distant corners; step into empty cell."""
    while True:
        ax = rng.randint(2, size - 3)
        ay = rng.randint(2, size - 3)
        # Far corners only
        fillers = [(1, 1), (1, size - 2), (size - 2, 1), (size - 2, size - 2)]
        # Skip if too close to agent or target
        target = (ax + 1, ay)
        ok = all(abs(f[0] - ax) + abs(f[1] - ay) >= 4 and
                 abs(f[0] - target[0]) + abs(f[1] - target[1]) >= 4
                 for f in fillers)
        if not ok:
            continue
        return {
            "agent_pos":     (ax, ay),
            "hazards":       fillers[:NUM_HAZARDS],
            "resources":     [],
            "post_reset":    None,
            "forced_action": ACTION_DOWN,
            "intended":      "none",
        }


TRIAL_BUILDERS = {
    "env_caused_hazard":    _trial_env_caused,
    "agent_caused_hazard":  _trial_agent_caused,
    "hazard_approach":      _trial_hazard_approach,
    "none":                 _trial_none,
}


# ---------------------------------------------------------------------------
# Scripted evaluation: forced-action single-pass comparator residuals
# ---------------------------------------------------------------------------

def run_scripted_eval(
    seed: int,
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_fwd: E2HarmSForward,
    dry_run: bool,
) -> Tuple[Dict[str, List[float]], Dict[str, int], Dict[str, int]]:
    """For each event type, run TRIALS_PER_TYPE deterministic trials.

    Each trial: reset_to(scripted_state) -> sense() -> z_harm_s_prev ->
    forced_action -> step() -> sense() -> z_harm_s_now ->
    residual = ||z_harm_s_now - E2_harm_s(z_harm_s_prev, a_forced)||.

    Returns:
      residuals[event_type]      -> list of residual magnitudes (intended)
      counts[event_type]         -> N(intended) actually executed
      label_match[event_type]    -> N where env returned the intended ttype
    """
    device = agent.device
    n_trials = DRY_RUN_TRIALS if dry_run else TRIALS_PER_TYPE
    rng = random.Random(seed * 9973 + 7)

    residuals: Dict[str, List[float]] = {t: [] for t in EVENT_TYPES}
    counts:    Dict[str, int]         = {t: 0 for t in EVENT_TYPES}
    label_match: Dict[str, int]       = {t: 0 for t in EVENT_TYPES}

    for event_type in EVENT_TYPES:
        builder = TRIAL_BUILDERS[event_type]
        for trial_i in range(n_trials):
            spec = builder(rng, GRID_SIZE)

            _flat, obs_dict = env.reset_to(
                agent_pos=spec["agent_pos"],
                hazard_positions=spec["hazards"],
                resource_positions=spec["resources"],
            )
            if spec["post_reset"] is not None:
                spec["post_reset"](env)
                # Recompute proximity fields in case the post_reset changed grid
                if env.use_proxy_fields:
                    env._compute_proximity_fields()
                # Refresh obs_dict so the encoder sees the post_reset state
                obs_dict = env._get_observation_dict()

            agent.reset()

            # Encode pre-step
            obs_body, obs_world, obs_harm = _obs_to_device(obs_dict, device)
            latent_pre = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            z_pre = latent_pre.z_harm
            if z_pre is None:
                continue
            z_pre = z_pre.detach().clone()

            # Forced action
            a_idx = spec["forced_action"]
            _flat_n, harm_signal, done, info, obs_dict_next = env.step(a_idx)
            ttype = info.get("transition_type", "none")

            # Encode post-step
            obs_body_n, obs_world_n, obs_harm_n = _obs_to_device(obs_dict_next, device)
            latent_post = agent.sense(obs_body_n, obs_world_n, obs_harm=obs_harm_n)
            z_post = latent_post.z_harm
            if z_post is None:
                continue
            z_post = z_post.detach().clone()

            # Single-pass comparator residual
            with torch.no_grad():
                a_oh = _onehot(a_idx, ACTION_DIM, device)
                z_pred = harm_fwd(z_pre, a_oh)
            residual = float((z_post - z_pred).norm(dim=-1).mean().item())

            # Bin by INTENDED event type (the scripted protocol's design label).
            residuals[event_type].append(residual)
            counts[event_type] += 1
            if ttype == event_type or (event_type == "none" and ttype not in EVENT_TYPES):
                label_match[event_type] += 1

        if not dry_run:
            print(
                f"  [scripted] seed={seed} type={event_type} "
                f"n={counts[event_type]} env_label_match={label_match[event_type]}",
                flush=True,
            )

    return residuals, counts, label_match


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_condition(seed: int, condition: str, dry_run: bool,
                  total_training_episodes: int) -> Dict:
    print(f"Seed {seed} Condition {condition}", flush=True)
    env   = _make_env(seed)
    agent = _make_agent(env, seed)
    device = agent.device

    harm_fwd   = _make_harm_fwd(device)
    forward_r2 = run_training(seed, env, agent, harm_fwd, dry_run,
                              total_training_episodes)
    print(f"  [P1 done] seed={seed} forward_r2={forward_r2:.3f}", flush=True)

    eval_env   = _make_env(seed + 1000)
    eval_agent = _make_agent(eval_env, seed)
    eval_agent.load_state_dict(agent.state_dict())

    residuals, counts, label_match = run_scripted_eval(
        seed, eval_env, eval_agent, harm_fwd, dry_run
    )

    means = {t: (float(np.mean(residuals[t])) if residuals[t] else 0.0)
             for t in EVENT_TYPES}
    stds  = {t: (float(np.std(residuals[t])) if residuals[t] else 0.0)
             for t in EVENT_TYPES}

    n_self = counts.get("agent_caused_hazard", 0)
    n_ext  = counts.get("env_caused_hazard", 0)
    n_app  = counts.get("hazard_approach", 0)
    n_base = counts.get("none", 0)

    mean_self = means.get("agent_caused_hazard", 0.0)
    mean_ext  = means.get("env_caused_hazard", 0.0)
    attenuation_ratio = (mean_self / mean_ext) if mean_ext > 1e-9 else 0.0

    mean_app   = means.get("hazard_approach", 0.0)
    std_base   = stds.get("none", 0.0)
    approach_snr = (mean_app / std_base) if std_base > 1e-9 else 0.0

    print(
        f"  [eval] seed={seed} cond={condition} "
        f"n_env={n_ext} n_agent={n_self} n_app={n_app} n_base={n_base} "
        f"r_self/ext={attenuation_ratio:.3f} approach_snr={approach_snr:.3f}",
        flush=True,
    )

    return {
        "seed":                seed,
        "condition":           condition,
        "forward_r2":          forward_r2,
        "residual_means":      means,
        "residual_stds":       stds,
        "event_counts":        counts,
        "env_label_match":     label_match,
        "attenuation_ratio":   attenuation_ratio,
        "approach_snr":        approach_snr,
        "trials_per_type":     DRY_RUN_TRIALS if dry_run else TRIALS_PER_TYPE,
    }


# ---------------------------------------------------------------------------
# Criteria (identical to EXQ-433 except C4 is by-construction)
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    seeds_sorted = sorted(all_results, key=lambda x: x["seed"])

    c1_vals  = [r["forward_r2"] for r in seeds_sorted]
    c1_seeds = sum(v >= C1_FORWARD_R2_THRESH for v in c1_vals)
    c1_pass  = c1_seeds >= MIN_SEEDS_PASS

    c2_vals  = [r["attenuation_ratio"] for r in seeds_sorted]
    c2_seeds = sum(C2_RATIO_LOW <= v <= C2_RATIO_HIGH for v in c2_vals)
    c2_pass  = c2_seeds >= MIN_SEEDS_PASS

    c3_vals  = [r["approach_snr"] for r in seeds_sorted]
    c3_seeds = sum(v >= C3_APPROACH_SNR_THRESH for v in c3_vals)
    c3_pass  = c3_seeds >= MIN_SEEDS_PASS

    c4_seeds   = 0
    c4_details = []
    for r in seeds_sorted:
        n_self = r["event_counts"].get("agent_caused_hazard", 0)
        n_ext  = r["event_counts"].get("env_caused_hazard", 0)
        ok     = (n_self >= C4_MIN_EVENTS_PER_TYPE and
                  n_ext  >= C4_MIN_EVENTS_PER_TYPE)
        c4_details.append({
            "seed":   r["seed"],
            "n_self": n_self,
            "n_ext":  n_ext,
            "ok":     ok,
        })
        if ok:
            c4_seeds += 1
    c4_pass = c4_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass

    return {
        "c1_forward_r2_pass":        c1_pass,
        "c1_vals":                   c1_vals,
        "c1_seeds_pass":             c1_seeds,
        "c2_attenuation_ratio_pass": c2_pass,
        "c2_vals":                   c2_vals,
        "c2_seeds_pass":             c2_seeds,
        "c3_approach_snr_pass":      c3_pass,
        "c3_vals":                   c3_vals,
        "c3_seeds_pass":             c3_seeds,
        "c4_event_density_pass":     c4_pass,
        "c4_details":                c4_details,
        "c4_seeds_pass":             c4_seeds,
        "overall_pass":              overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_training_episodes = (
        DRY_RUN_EPISODES * 2 if args.dry_run else P0_EPISODES + P1_EPISODES
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{EXPERIMENT_TYPE}_dry_{ts}_v3"
        if args.dry_run
        else f"{EXPERIMENT_TYPE}_{ts}_v3"
    )
    print(f"EXQ-433a start: {run_id}")

    all_results: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition, args.dry_run,
                                   total_training_episodes)
            all_results.append(result)
            passed_seed = (
                result["forward_r2"] >= C1_FORWARD_R2_THRESH
                and C2_RATIO_LOW <= result["attenuation_ratio"] <= C2_RATIO_HIGH
                and result["approach_snr"] >= C3_APPROACH_SNR_THRESH
                and result["event_counts"].get("agent_caused_hazard", 0) >= C4_MIN_EVENTS_PER_TYPE
                and result["event_counts"].get("env_caused_hazard",   0) >= C4_MIN_EVENTS_PER_TYPE
            )
            print(f"verdict: {'PASS' if passed_seed else 'FAIL'}", flush=True)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-433a {outcome} ===")
    print(f"C1 forward_r2 (>={C1_FORWARD_R2_THRESH}): "
          f"{criteria['c1_forward_r2_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c1_vals']]})")
    print(f"C2 self/ext ratio in [{C2_RATIO_LOW},{C2_RATIO_HIGH}]: "
          f"{criteria['c2_attenuation_ratio_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c2_vals']]})")
    print(f"C3 approach SNR (>={C3_APPROACH_SNR_THRESH}): "
          f"{criteria['c3_approach_snr_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c3_vals']]})")
    print(f"C4 event density (by-construction >={C4_MIN_EVENTS_PER_TYPE}): "
          f"{criteria['c4_event_density_pass']} (details={criteria['c4_details']})")

    sd029_pass   = (criteria["c1_forward_r2_pass"]
                    and criteria["c2_attenuation_ratio_pass"]
                    and criteria["c3_approach_snr_pass"])
    mech256_pass = criteria["c1_forward_r2_pass"]

    output = {
        "run_id":             run_id,
        "experiment_type":    EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes":         "v3_exq_433_sd029_eventcond_comparator",
        "evidence_direction_per_claim": {
            "SD-029":   "supports" if sd029_pass   else "does_not_support",
            "MECH-256": "supports" if mech256_pass else "does_not_support",
        },
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome":            outcome,
        "criteria":           criteria,
        "results_per_condition": all_results,
        "config": {
            "seeds":                     SEEDS,
            "conditions":                CONDITIONS,
            "p0_episodes":               P0_EPISODES,
            "p1_episodes":               P1_EPISODES,
            "trials_per_type":           TRIALS_PER_TYPE,
            "steps_per_ep":              STEPS_PER_EP,
            "grid_size":                 GRID_SIZE,
            "num_hazards":               NUM_HAZARDS,
            "hazard_harm":               HAZARD_HARM,
            "interventional_fraction":   INTERVENTIONAL_FRACTION,
            "c1_forward_r2_thresh":      C1_FORWARD_R2_THRESH,
            "c2_ratio_low":              C2_RATIO_LOW,
            "c2_ratio_high":             C2_RATIO_HIGH,
            "c3_approach_snr_thresh":    C3_APPROACH_SNR_THRESH,
            "c4_min_events_per_type":    C4_MIN_EVENTS_PER_TYPE,
            "min_seeds_pass":            MIN_SEEDS_PASS,
            "eval_protocol":             "scripted_fixed_action_per_trial",
        },
        "timestamp_utc": ts,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
        EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
