#!/opt/local/bin/python3
"""
V3-EXQ-433: SD-029 Event-Conditioned Single-Pass Comparator Test

experiment_purpose: evidence

Scientific question: Does the z_harm_s single-pass comparator (residual =
z_harm_s_observed - E2_harm_s(z_harm_s_{t-1}, a_actual)) exhibit the
Shergill/Blakemore partial-attenuation signature on self-caused vs
externally-caused harm events?

Distinct from EXQ-431 (cf_gap-based SD-003 counterfactual test): this is
a SINGLE-PASS comparator. No E2(a_cf) call at evaluation. The residual
between the observed next z_harm_s and the forward-model prediction on the
actual action IS the agency signal.

Claim mapping (per SD-029 registration 2026-04-18):
  C1 forward_r2(E2_harm_s) >= 0.9 on reafferent z_harm_s stream
  C2 Self/external residual attenuation ratio in [0.3, 0.7]
     (Shergill partial-attenuation pattern, not a binary gate)
  C3 SNR on approach-to-harm events > 3.0
     (event-conditioned, not mean over all steps -- EXQ-431 lesson)
  C4 Event-density floor: >= 20 self-caused (agent_caused_hazard) events
     AND >= 20 externally-caused (env_caused_hazard) events per seed
     (density controller extends P2 episodes until floor met or cap reached)

PASS: C1 AND C2 AND C3 AND C4 across >= 3/4 seeds.

SD-013 interventional training (fraction=0.5) is REQUIRED during P1 so
E2_harm_s learns P(z_harm_s | do(a)) rather than correlational P(z_harm_s | a).

Phased training:
  P0 (80 ep):  HarmEncoder + agent encoder warmup
  P1 (100 ep): E2HarmSForward interventional training on frozen z_harm_s
  P2 (up to 200 ep): Evaluation with per-event-type residual accumulation;
                      density controller re-runs 25-ep extensions until
                      event-density floor is met or cap (200 ep) is reached.

Single condition TRAINED at 4 seeds. A RANDOM baseline is not required here
because SD-029's criteria are absolute (attenuation ratio, SNR) -- a random
model trivially fails C1 and can only weaken interpretability.

Claims tested: SD-029 (primary), MECH-256 (general comparator instantiation).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
from datetime import datetime
from collections import defaultdict
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
EXPERIMENT_TYPE    = "v3_exq_433_sd029_eventcond_comparator"
CLAIM_IDS          = ["SD-029", "MECH-256"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS      = [42, 7, 13, 91]
CONDITIONS = ["TRAINED"]

P0_EPISODES       = 80
P1_EPISODES       = 100
P2_BASE_EPISODES  = 100
P2_EXTEND_EPISODES = 25
P2_CAP_EPISODES   = 200
STEPS_PER_EP      = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 4      # higher than EXQ-431 (3) to boost event density
HAZARD_HARM   = 0.3
# contamination carries over via agent_caused_hazard transitions (self-caused)

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

# Pass thresholds (per SD-029 criteria sketch)
C1_FORWARD_R2_THRESH      = 0.9
C2_RATIO_LOW              = 0.3
C2_RATIO_HIGH             = 0.7
C3_APPROACH_SNR_THRESH    = 3.0
C4_MIN_EVENTS_PER_TYPE    = 20
MIN_SEEDS_PASS            = 3   # 3 of 4

EVENT_TYPES = [
    "env_caused_hazard",      # externally-caused (fresh env hazard)
    "agent_caused_hazard",    # self-caused (agent re-enters contaminated cell)
    "hazard_approach",        # approach-to-harm (proxy gradient)
    "none",                   # baseline
]

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 20


# ---------------------------------------------------------------------------
# Factories
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


# ---------------------------------------------------------------------------
# Training (P0 + P1) -- unchanged structure from EXQ-431
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
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            obs_harm  = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = (obs_harm.to(device).unsqueeze(0)
                            if obs_harm.dim() == 1 else obs_harm.to(device))

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

            # P0: encoder warmup
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

            # P1: E2HarmSForward interventional training
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

            # P1 forward_r2 eval slice (C1)
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
# Evaluation (single-pass comparator, event-conditioned with density control)
# ---------------------------------------------------------------------------

def _eval_one_episode(
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_fwd: E2HarmSForward,
    steps_per: int,
    residuals: Dict[str, List[float]],
):
    """One P2 episode. Accumulates single-pass residual magnitudes per event type."""
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()

    z_harm_s_prev:   Optional[torch.Tensor] = None
    action_prev_idx: Optional[int]          = None

    for step in range(steps_per):
        obs_body  = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)
        obs_harm  = obs_dict.get("harm_obs", None)
        if obs_harm is not None:
            obs_harm = (obs_harm.to(device).unsqueeze(0)
                        if obs_harm.dim() == 1 else obs_harm.to(device))

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

        # Single-pass comparator: residual at this step refers to the
        # previous (z_harm_s_prev, action_prev_idx) -> z_harm_s_now transition.
        if (
            z_harm_s_now is not None
            and z_harm_s_prev is not None
            and action_prev_idx is not None
        ):
            a_prev = _onehot(action_prev_idx, ACTION_DIM, device)
            with torch.no_grad():
                z_pred = harm_fwd(z_harm_s_prev.detach(), a_prev)
            residual = float((z_harm_s_now.detach() - z_pred).norm(dim=-1).mean().item())
            # Tag residual by the transition_type that PRODUCED z_harm_s_now.
            # env.step() below will tell us -- so we record after step().
            # We stage the residual and bind after we know ttype.
            staged_residual = residual
        else:
            staged_residual = None

        _flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
        ttype = info.get("transition_type", "none")
        agent.update_residue(float(harm_signal))

        if staged_residual is not None:
            key = ttype if ttype in EVENT_TYPES else "none"
            residuals[key].append(staged_residual)

        z_harm_s_prev   = z_harm_s_now.detach().clone() if z_harm_s_now is not None else None
        action_prev_idx = action_idx
        obs_dict        = obs_dict_next

        if done:
            break


def run_evaluation_with_density_floor(
    seed: int,
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_fwd: E2HarmSForward,
    dry_run: bool,
) -> Tuple[Dict, int]:
    """
    Runs P2 episodes, extending until event-density floor met or cap reached.

    Returns per-event-type residual lists and total episodes run.
    """
    steps_per = DRY_RUN_STEPS if dry_run else STEPS_PER_EP
    base_eps  = DRY_RUN_EPISODES if dry_run else P2_BASE_EPISODES
    extend_eps = DRY_RUN_EPISODES if dry_run else P2_EXTEND_EPISODES
    cap_eps   = DRY_RUN_EPISODES if dry_run else P2_CAP_EPISODES
    floor     = 3 if dry_run else C4_MIN_EVENTS_PER_TYPE

    residuals: Dict[str, List[float]] = {t: [] for t in EVENT_TYPES}
    episodes_run = 0

    def _density_met() -> bool:
        return (
            len(residuals["env_caused_hazard"])   >= floor
            and len(residuals["agent_caused_hazard"]) >= floor
        )

    # Initial burst
    for _ in range(base_eps):
        _eval_one_episode(env, agent, harm_fwd, steps_per, residuals)
        episodes_run += 1
        if episodes_run % 25 == 0 and not dry_run:
            print(
                f"  [eval] seed={seed} ep {episodes_run}/{cap_eps} "
                f"env={len(residuals['env_caused_hazard'])} "
                f"agent={len(residuals['agent_caused_hazard'])} "
                f"approach={len(residuals['hazard_approach'])}",
                flush=True,
            )

    # Density extensions
    while not _density_met() and episodes_run < cap_eps:
        for _ in range(extend_eps):
            if episodes_run >= cap_eps:
                break
            _eval_one_episode(env, agent, harm_fwd, steps_per, residuals)
            episodes_run += 1
        if not dry_run:
            print(
                f"  [eval-extend] seed={seed} ep {episodes_run}/{cap_eps} "
                f"env={len(residuals['env_caused_hazard'])} "
                f"agent={len(residuals['agent_caused_hazard'])}",
                flush=True,
            )

    return residuals, episodes_run


# ---------------------------------------------------------------------------
# Single-seed run (TRAINED condition)
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

    # Separate env for clean evaluation (different seed offset)
    eval_env   = _make_env(seed + 1000)
    eval_agent = _make_agent(eval_env, seed)
    eval_agent.load_state_dict(agent.state_dict())

    residuals, n_eval_eps = run_evaluation_with_density_floor(
        seed, eval_env, eval_agent, harm_fwd, dry_run
    )

    counts = {t: len(residuals[t]) for t in EVENT_TYPES}
    means  = {t: (float(np.mean(residuals[t])) if residuals[t] else 0.0)
              for t in EVENT_TYPES}
    stds   = {t: (float(np.std(residuals[t])) if residuals[t] else 0.0)
              for t in EVENT_TYPES}

    n_self = counts.get("agent_caused_hazard", 0)
    n_ext  = counts.get("env_caused_hazard", 0)
    n_app  = counts.get("hazard_approach", 0)
    n_base = counts.get("none", 0)

    # C2: self/external attenuation ratio
    mean_self = means.get("agent_caused_hazard", 0.0)
    mean_ext  = means.get("env_caused_hazard", 0.0)
    attenuation_ratio = (mean_self / mean_ext) if mean_ext > 1e-9 else 0.0

    # C3: SNR on approach -- mean_approach / std_baseline
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
        "attenuation_ratio":   attenuation_ratio,
        "approach_snr":        approach_snr,
        "n_eval_episodes":     n_eval_eps,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    # Only TRAINED condition -- all seeds apply.
    seeds_sorted = sorted(all_results, key=lambda x: x["seed"])

    # C1: forward_r2 >= 0.9
    c1_vals  = [r["forward_r2"] for r in seeds_sorted]
    c1_seeds = sum(v >= C1_FORWARD_R2_THRESH for v in c1_vals)
    c1_pass  = c1_seeds >= MIN_SEEDS_PASS

    # C2: self/external ratio in [0.3, 0.7]
    c2_vals  = [r["attenuation_ratio"] for r in seeds_sorted]
    c2_seeds = sum(C2_RATIO_LOW <= v <= C2_RATIO_HIGH for v in c2_vals)
    c2_pass  = c2_seeds >= MIN_SEEDS_PASS

    # C3: approach_snr > 3.0
    c3_vals  = [r["approach_snr"] for r in seeds_sorted]
    c3_seeds = sum(v >= C3_APPROACH_SNR_THRESH for v in c3_vals)
    c3_pass  = c3_seeds >= MIN_SEEDS_PASS

    # C4: event-density floor met per seed
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
        "c1_forward_r2_pass":       c1_pass,
        "c1_vals":                  c1_vals,
        "c1_seeds_pass":            c1_seeds,
        "c2_attenuation_ratio_pass": c2_pass,
        "c2_vals":                  c2_vals,
        "c2_seeds_pass":            c2_seeds,
        "c3_approach_snr_pass":     c3_pass,
        "c3_vals":                  c3_vals,
        "c3_seeds_pass":            c3_seeds,
        "c4_event_density_pass":    c4_pass,
        "c4_details":               c4_details,
        "c4_seeds_pass":            c4_seeds,
        "overall_pass":             overall_pass,
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
        f"v3_exq_433_sd029_eventcond_comparator_dry_{ts}_v3"
        if args.dry_run
        else f"v3_exq_433_sd029_eventcond_comparator_{ts}_v3"
    )
    print(f"EXQ-433 start: {run_id}")

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

    print(f"\n=== EXQ-433 {outcome} ===")
    print(f"C1 forward_r2 (>={C1_FORWARD_R2_THRESH}): "
          f"{criteria['c1_forward_r2_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c1_vals']]})")
    print(f"C2 self/ext ratio in [{C2_RATIO_LOW},{C2_RATIO_HIGH}]: "
          f"{criteria['c2_attenuation_ratio_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c2_vals']]})")
    print(f"C3 approach SNR (>={C3_APPROACH_SNR_THRESH}): "
          f"{criteria['c3_approach_snr_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c3_vals']]})")
    print(f"C4 event density: {criteria['c4_event_density_pass']} "
          f"(details={criteria['c4_details']})")

    # Per-claim direction
    #  SD-029:   attribution signature   -> C1 + C2 + C3
    #  MECH-256: general comparator      -> C1 (forward model fit suffices
    #            as comparator-substrate evidence; attenuation pattern is
    #            the stream-specific instantiation SD-029 carries)
    sd029_pass   = (criteria["c1_forward_r2_pass"]
                    and criteria["c2_attenuation_ratio_pass"]
                    and criteria["c3_approach_snr_pass"])
    mech256_pass = criteria["c1_forward_r2_pass"]

    output = {
        "run_id":            run_id,
        "experiment_type":   EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":         CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction_per_claim": {
            "SD-029":   "supports" if sd029_pass   else "does_not_support",
            "MECH-256": "supports" if mech256_pass else "does_not_support",
        },
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome":           outcome,
        "criteria":          criteria,
        "results_per_condition": all_results,
        "config": {
            "seeds":                     SEEDS,
            "conditions":                CONDITIONS,
            "p0_episodes":               P0_EPISODES,
            "p1_episodes":               P1_EPISODES,
            "p2_base_episodes":          P2_BASE_EPISODES,
            "p2_extend_episodes":        P2_EXTEND_EPISODES,
            "p2_cap_episodes":           P2_CAP_EPISODES,
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
