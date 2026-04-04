"""
V3-EXQ-223a -- Toroidal Minimal Vertebrate Cognition Ablation

Claims: None (hypothesis generation)

EXPERIMENT_PURPOSE = "diagnostic"

Replicates EXQ-223 (minimal E1/E2/hippocampus ablation) with a toroidal
grid: no walls, movement wraps at edges. Jellyfish drift freely through
the wrapping boundary.

SCIENTIFIC QUESTION:
  EXQ-223 passed with a degenerate wall-hugging strategy -- the agent
  learned to move in one direction until hitting a wall, then stay there.
  The wall provided a free refuge from drifting hazards.

  With toroidal wrapping there are no walls to exploit. If the wall-refuge
  strategy was the entire basis for the EXQ-223 PASS, we expect FAIL here.

  If the minimal mind loop has any genuine gradient-following beyond
  wall-hugging, we expect at least partial harm-avoidance to survive.

  SECONDARY QUESTION: does the agent develop a row or column preference
  (top/bottom/left/right half of the grid), and if so, does it correlate
  with hazard-sparse regions? This would indicate z_world encodes spatial
  structure beyond the immediate 5x5 view.

ABLATION (identical to EXQ-223):
  commitment_threshold=-1.0 (always uncommitted)
  z_goal_enabled=False, benefit_eval_enabled=False, goal_weight=0.0

PRE-REGISTERED ACCEPTANCE CRITERIA (same as EXQ-223):
  C1: ree_mean_reward > rand_mean_reward in >= 2/3 seeds
  C2: ree/rand harm ratio < 1.0 in >= 2/3 seeds
  C3: mean_n_cands >= 2 per e3 tick (informational)
  C4: warmup_last25 > warmup_first25 in >= 2/3 seeds

PASS = C1 + C2 + C4 (majority >= 2/3 seeds each). C3 informational only.

Estimated runtime: ~8 min (3 seeds x 150 eps x ~0.17 min/ep on Mac CPU)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE    = "v3_exq_223a_toroidal_minimal_vertebrate"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = []

# Pre-registered thresholds (identical to EXQ-223)
THRESH_C1 = 0.0
THRESH_C2 = 1.0
THRESH_C3 = 2
THRESH_C4 = 0.0

HARM_MODE_THRESH   = 0.25
EXPLORE_ERR_THRESH = 0.10

ENV_KWARGS = dict(
    size=10,
    num_hazards=3,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    toroidal=True,           # no walls; movement and hazard drift wrap at edges
)

WARMUP_EPISODES   = 100
EVAL_EPISODES     = 50
STEPS_PER_EPISODE = 200
WORLD_DIM         = 32
SELF_DIM          = 32
WF_BUF_MAX        = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE        = 32
LR_E1             = 1e-4
LR_E2_WF          = 3e-4
LR_E3_HARM        = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _make_agent_and_env(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        z_goal_enabled=False,
        benefit_eval_enabled=False,
        goal_weight=0.0,
    )
    config.e3.commitment_threshold = -1.0
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0: Warmup training
# ---------------------------------------------------------------------------

def _warmup_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    device = agent.device
    action_dim = env.action_dim

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM
    )

    wf_buf:        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]]               = []
    reward_log:    List[float] = []

    agent.train()

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent    = agent.sense(obs_body, obs_world)
            z_world_curr = latent.z_world.detach()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, action_dim - 1), action_dim, device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs  = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp   = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        reward_log.append(ep_reward)

        if (ep + 1) % 25 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [warmup] ep {ep+1}/{num_episodes}"
                f"  rv={rv:.4f}"
                f"  ep_reward={ep_reward:.4f}",
                flush=True,
            )

    first25 = float(np.mean(reward_log[:25]))  if len(reward_log) >= 25 else float(np.mean(reward_log))
    last25  = float(np.mean(reward_log[-25:])) if len(reward_log) >= 25 else float(np.mean(reward_log))

    return {
        "final_running_variance": agent.e3._running_variance,
        "warmup_first25_reward":  first25,
        "warmup_last25_reward":   last25,
    }


# ---------------------------------------------------------------------------
# Episode logging helpers
# ---------------------------------------------------------------------------

def _classify_mode(z_harm_norm: float, world_change_norm: float, harm_signal: float) -> str:
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


# ---------------------------------------------------------------------------
# Phase 1 / 2: Evaluation
# ---------------------------------------------------------------------------

def _eval_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    use_random: bool = False,
    label: str = "",
    record_episodes: bool = False,
) -> Dict:
    action_dim     = env.action_dim
    device         = agent.device
    episode_rewards: List[float] = []
    episode_harms:   List[float] = []
    n_cands_log:     List[int]   = []
    episodes_log:    List[Dict]  = []

    agent.eval()

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_self_prev:  Optional[torch.Tensor] = None
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        ep_reward = 0.0
        ep_harm   = 0.0

        if record_episodes and not use_random:
            ep_steps:         List[Dict] = []
            initial_hazards   = [list(h) for h in env.hazards]
            initial_resources = [list(r) for r in env.resources]
            current_hazards   = [list(h) for h in env.hazards]
            current_resources = [list(r) for r in env.resources]

        for step_idx in range(steps_per_episode):
            if use_random:
                action = _action_to_onehot(
                    random.randint(0, action_dim - 1), action_dim, device
                )
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            else:
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    if z_self_prev is not None and action_prev is not None:
                        agent.record_transition(
                            z_self_prev, action_prev, latent.z_self.detach()
                        )
                    ticks    = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, WORLD_DIM, device=device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    if ticks.get("e3_tick", True):
                        n_cands_log.append(len(candidates))
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, action_dim - 1), action_dim, device
                        )
                        agent._last_action = action

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                if record_episodes:
                    if info.get("env_drift_occurred", False):
                        current_hazards   = [list(h) for h in env.hazards]
                        current_resources = [list(r) for r in env.resources]
                    z_harm_norm = (
                        float(latent.z_harm.norm().item())
                        if latent.z_harm is not None else 0.0
                    )
                    z_beta_val = (
                        float(latent.z_beta.mean().item())
                        if latent.z_beta is not None else 0.0
                    )
                    world_change_norm = (
                        float((latent.z_world - z_world_prev).norm().item())
                        if z_world_prev is not None else 0.0
                    )
                    mode = _classify_mode(z_harm_norm, world_change_norm, float(harm_signal))
                    ep_steps.append({
                        "t":               step_idx,
                        "pos":             [int(env.agent_x), int(env.agent_y)],
                        "action":          int(action.argmax(dim=-1).item()),
                        "harm_signal":     float(harm_signal),
                        "z_harm_norm":     z_harm_norm,
                        "z_world_norm":    float(latent.z_world.norm().item()),
                        "z_beta_val":      z_beta_val,
                        "world_change_norm": world_change_norm,
                        "mode":            mode,
                        "transition_type": info.get("transition_type", "none"),
                        "health":          float(info.get("health", 1.0)),
                        "energy":          float(info.get("energy", 1.0)),
                        "harm_event":      float(harm_signal) < 0,
                        "n_cands":         len(candidates),
                        "hazards":         [list(h) for h in current_hazards],
                        "resources":       [list(r) for r in current_resources],
                    })

                z_self_prev  = latent.z_self.detach()
                z_world_prev = latent.z_world.detach()
                action_prev  = action.detach()

            ep_reward += float(harm_signal)
            if float(harm_signal) < 0:
                ep_harm += abs(float(harm_signal))
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_harms.append(ep_harm)

        if record_episodes and not use_random:
            episodes_log.append({
                "ep":               ep_idx,
                "initial_hazards":  initial_hazards,
                "initial_resources": initial_resources,
                "steps":            ep_steps,
            })

        if (ep_idx + 1) % 25 == 0 or ep_idx == num_episodes - 1:
            print(
                f"  [eval {label}] ep {ep_idx+1}/{num_episodes}"
                f"  mean_reward={np.mean(episode_rewards):.4f}",
                flush=True,
            )

    result = {
        "mean_reward":  float(np.mean(episode_rewards)),
        "mean_harm":    float(np.mean(episode_harms)),
        "rewards":      episode_rewards,
        "mean_n_cands": float(np.mean(n_cands_log)) if n_cands_log else 0.0,
    }
    if record_episodes:
        result["episodes"] = episodes_log
    return result


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool = False) -> Dict:
    warmup_eps = 5 if dry_run else WARMUP_EPISODES
    eval_eps   = 3 if dry_run else EVAL_EPISODES

    print(
        f"\n[EXQ-223a] Seed {seed}"
        f"  warmup={warmup_eps}  eval={eval_eps}"
        f"  dry_run={dry_run}",
        flush=True,
    )

    agent, env = _make_agent_and_env(seed)

    print(f"[EXQ-223a] Seed {seed} -- Phase 0: warmup", flush=True)
    warmup = _warmup_train(agent, env, warmup_eps, STEPS_PER_EPISODE)

    print(f"[EXQ-223a] Seed {seed} -- Phase 1: REE ablated eval", flush=True)
    ree = _eval_agent(agent, env, eval_eps, STEPS_PER_EPISODE,
                      use_random=False, label="REE-ablated", record_episodes=True)

    print(f"[EXQ-223a] Seed {seed} -- Phase 2: random baseline", flush=True)
    rnd = _eval_agent(agent, env, eval_eps, STEPS_PER_EPISODE,
                      use_random=True, label="random")

    improvement_delta = warmup["warmup_last25_reward"] - warmup["warmup_first25_reward"]
    harm_ratio = (
        ree["mean_harm"] / rnd["mean_harm"]
        if rnd["mean_harm"] > 1e-8 else 0.0
    )

    print(
        f"\n[EXQ-223a] Seed {seed} summary:\n"
        f"  warmup first25={warmup['warmup_first25_reward']:.4f}"
        f"  last25={warmup['warmup_last25_reward']:.4f}"
        f"  delta={improvement_delta:.4f}\n"
        f"  REE  reward={ree['mean_reward']:.4f}"
        f"  harm={ree['mean_harm']:.4f}"
        f"  n_cands={ree['mean_n_cands']:.1f}\n"
        f"  rand reward={rnd['mean_reward']:.4f}"
        f"  harm={rnd['mean_harm']:.4f}",
        flush=True,
    )

    return {
        "seed":                  seed,
        "warmup_first25_reward": warmup["warmup_first25_reward"],
        "warmup_last25_reward":  warmup["warmup_last25_reward"],
        "warmup_improvement":    improvement_delta,
        "warmup_final_rv":       warmup["final_running_variance"],
        "ree_mean_reward":       ree["mean_reward"],
        "ree_mean_harm":         ree["mean_harm"],
        "ree_mean_n_cands":      ree["mean_n_cands"],
        "rand_mean_reward":      rnd["mean_reward"],
        "rand_mean_harm":        rnd["mean_harm"],
        "harm_ratio":            harm_ratio,
        "episodes":              ree.get("episodes", []),
    }


# ---------------------------------------------------------------------------
# Aggregate + output
# ---------------------------------------------------------------------------

def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    print(
        f"[V3-EXQ-223a] Toroidal Minimal Vertebrate Cognition Ablation\n"
        f"  Seeds: {seeds}\n"
        f"  Warmup: {WARMUP_EPISODES} eps  Eval: {EVAL_EPISODES} eps"
        f"  Steps/ep: {STEPS_PER_EPISODE}\n"
        f"  TOROIDAL: no walls, movement wraps at edges\n"
        f"  ABLATION: commitment_threshold=-1.0 (always uncommitted)\n"
        f"            z_goal_enabled=False, benefit_eval_enabled=False\n"
        f"  Pre-registered thresholds:\n"
        f"    C1: ree_mean_reward > rand_mean_reward (>= 2/3 seeds)\n"
        f"    C2: ree/rand harm ratio < 1.0 (>= 2/3 seeds)\n"
        f"    C3: mean_n_cands >= {THRESH_C3} (informational)\n"
        f"    C4: warmup_last25 > warmup_first25 (>= 2/3 seeds)",
        flush=True,
    )

    seed_results = [run_seed(s, dry_run=dry_run) for s in seeds]
    n_seeds  = len(seeds)
    majority = max(2, (2 * n_seeds + 2) // 3)

    c1_seeds = sum(1 for r in seed_results
                   if r["ree_mean_reward"] > r["rand_mean_reward"] + THRESH_C1)
    c2_seeds = sum(1 for r in seed_results
                   if r["harm_ratio"] < THRESH_C2)
    c3_seeds = sum(1 for r in seed_results
                   if r["ree_mean_n_cands"] >= THRESH_C3)
    c4_seeds = sum(1 for r in seed_results
                   if r["warmup_improvement"] > THRESH_C4)

    c1_pass = c1_seeds >= majority
    c2_pass = c2_seeds >= majority
    c3_pass = c3_seeds >= majority
    c4_pass = c4_seeds >= majority

    all_pass     = c1_pass and c2_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c4_pass])
    status       = "PASS" if all_pass else "FAIL"

    failure_notes = []
    if not c1_pass:
        vals = [(r["ree_mean_reward"], r["rand_mean_reward"]) for r in seed_results]
        failure_notes.append(
            f"C1 FAIL: ree>rand in {c1_seeds}/{n_seeds} seeds (need {majority});"
            f" ree={[f'{v[0]:.4f}' for v in vals]}"
            f" rand={[f'{v[1]:.4f}' for v in vals]}"
        )
    if not c2_pass:
        vals = [r["harm_ratio"] for r in seed_results]
        failure_notes.append(
            f"C2 FAIL: harm_ratio < 1.0 in {c2_seeds}/{n_seeds} seeds (need {majority});"
            f" ratios={[f'{v:.4f}' for v in vals]}"
        )
    if not c4_pass:
        vals = [r["warmup_improvement"] for r in seed_results]
        failure_notes.append(
            f"C4 FAIL: warmup improvement > 0 in {c4_seeds}/{n_seeds} seeds (need {majority});"
            f" deltas={[f'{v:.4f}' for v in vals]}"
        )

    metrics: Dict = {
        "c1_seeds_pass":      float(c1_seeds),
        "c2_seeds_pass":      float(c2_seeds),
        "c3_seeds_pass":      float(c3_seeds),
        "c4_seeds_pass":      float(c4_seeds),
        "criteria_met":       float(criteria_met),
        "n_seeds":            float(n_seeds),
        "majority_threshold": float(majority),
    }
    for r in seed_results:
        s = r["seed"]
        for key in (
            "warmup_first25_reward", "warmup_last25_reward", "warmup_improvement",
            "warmup_final_rv",
            "ree_mean_reward", "ree_mean_harm", "ree_mean_n_cands",
            "rand_mean_reward", "rand_mean_harm", "harm_ratio",
        ):
            metrics[f"seed{s}_{key}"] = float(r[key])

    rows = ""
    for r in seed_results:
        rows += (
            f"| {r['seed']}"
            f" | {r['warmup_first25_reward']:.4f}"
            f" | {r['warmup_last25_reward']:.4f}"
            f" | {r['warmup_improvement']:.4f}"
            f" | {r['ree_mean_reward']:.4f}"
            f" | {r['rand_mean_reward']:.4f}"
            f" | {r['ree_mean_harm']:.4f}"
            f" | {r['rand_mean_harm']:.4f}"
            f" | {r['harm_ratio']:.4f}"
            f" | {r['ree_mean_n_cands']:.1f} |\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = f"""# V3-EXQ-223a -- Toroidal Minimal Vertebrate Cognition Ablation

**Status:** {status}
**Purpose:** Hypothesis generation -- does minimal mind survive without walls to exploit?
**Toroidal:** no walls; movement and hazard drift wrap at grid edges
**Ablation:** commitment_threshold=-1.0 (always uncommitted, no argmin, no BetaGate hold)
             z_goal_enabled=False, benefit_eval_enabled=False, goal_weight=0.0
**Warmup:** {WARMUP_EPISODES} eps | **Eval:** {EVAL_EPISODES} eps | **Seeds:** {seeds}

## Scientific Question

EXQ-223 passed via wall-hugging: agent moved in one direction to a wall
corner and stayed there. This provides free hazard avoidance because
jellyfish drift only within the interior. The minimal mind may have no
genuine gradient-following beyond exploiting this boundary effect.

With toroidal wrapping, walls are removed. If PASS: the minimal loop has
genuine spatial gradient-following beyond wall refuge. If FAIL: EXQ-223
result was entirely boundary-dependent.

Secondary question: does the agent develop a spatial preference (row/column
half of grid), indicating z_world encodes more than the immediate 5x5 view?

## Pre-registered Thresholds

| Criterion | Threshold | Seed requirement |
|-----------|-----------|-----------------|
| C1: ree_reward > rand_reward | > {THRESH_C1} | >= {majority}/{n_seeds} seeds |
| C2: harm_ratio (ree/rand) | < {THRESH_C2} | >= {majority}/{n_seeds} seeds |
| C3: n_cands per e3 tick (info) | >= {THRESH_C3} | informational |
| C4: warmup improvement delta | > {THRESH_C4} | >= {majority}/{n_seeds} seeds |

## Results by Seed

| Seed | W-first25 | W-last25 | W-delta | REE reward | rand reward | REE harm | rand harm | harm ratio | n_cands |
|------|-----------|----------|---------|------------|------------|---------|----------|-----------|---------|
{rows}
## PASS Criteria

| Criterion | Seeds passing | Required | Result |
|-----------|--------------|---------|--------|
| C1 ree > rand reward | {c1_seeds}/{n_seeds} | >={majority} | {"PASS" if c1_pass else "FAIL"} |
| C2 harm ratio < 1.0 | {c2_seeds}/{n_seeds} | >={majority} | {"PASS" if c2_pass else "FAIL"} |
| C3 n_cands >= {THRESH_C3} (info) | {c3_seeds}/{n_seeds} | informational | {"PASS" if c3_pass else "info"} |
| C4 warmup improvement > 0 | {c4_seeds}/{n_seeds} | >={majority} | {"PASS" if c4_pass else "FAIL"} |

Criteria met (C1+C2+C4): {criteria_met}/3 -> **{status}**
{failure_section}
"""

    evidence_direction = (
        "supports" if all_pass
        else ("mixed" if criteria_met >= 2 else "weakens")
    )

    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "env_config":      ENV_KWARGS,
        "phase":           "eval_ree",
        "toroidal":        True,
        "seeds": [
            {"seed": r["seed"], "episodes": r.get("episodes", [])}
            for r in seed_results
        ],
    }

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "experiment_type":    EXPERIMENT_TYPE,
        "episode_log":        episode_log,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"]          = CLAIM_IDS

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_log = result.pop("episode_log", None)
    if episode_log is not None:
        episode_log["run_id"] = result["run_id"]
        log_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}_episode_log.json"
        log_path.write_text(json.dumps(episode_log, indent=2) + "\n", encoding="utf-8")
        print(f"Episode log written to: {log_path}", flush=True)

    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
