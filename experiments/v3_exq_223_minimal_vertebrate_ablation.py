"""
V3-EXQ-223 -- Minimal Vertebrate Cognition Ablation

Claims: None (milestone architecture validation)

EXPERIMENT_PURPOSE = "diagnostic"

Tests whether the core E1/E2/hippocampus loop exhibits gradient-following
behavior when the commitment architecture is stripped and raw reward signals
drive action selection.

EVOLUTIONARY MOTIVATION:
  If REE's architecture correctly decomposes cognition, ablating the
  commitment layer (argmin selection, BetaGate action-holding, hypothesis
  tag, pre/post-commit channels) should leave a functional gradient-follower
  -- the minimal vertebrate cognition loop:
    E1 = association learning on sensory gradients
    E2 = fast transition model (movement prediction)
    hippocampus = chained state sequence proposals
    go/no-go = stochastic multinomial trajectory selection
    harm/goal = raw environmental signals

  Ablation mechanism: commitment_threshold=-1.0 forces committed=False
  always (_running_variance is always >= 0, so rv < -1.0 is always False).
  Effect:
    (1) E3.select() always uses multinomial (never argmin)
    (2) BetaGate always released -- no action-holding
    (3) _committed_trajectory never set
  z_goal_enabled=False and benefit_eval_enabled=False are also set,
  removing the latent goal/benefit substrates.

  This sidesteps SD-011 (z_world perp z_harm) and SD-012 (z_goal seeding
  failures) by using raw environmental harm/reward signals only.

  PASS = minimal mind loop is structurally sound before harm-stream
         and goal substrates are resolved.
  FAIL = commitment architecture is load-bearing for basic navigation.

PRE-REGISTERED ACCEPTANCE CRITERIA:
  C1 (ree_mean_reward > rand_mean_reward in >= 2/3 seeds):
    REE ablated agent outperforms random on net reward per episode.
  C2 (ree_mean_harm < rand_mean_harm in >= 2/3 seeds):
    REE ablated agent incurs less harm than random baseline.
  C3 (mean_n_cands_per_e3_tick >= 2 per seed, informational):
    Hippocampal module is generating multi-candidate trajectory proposals.
  C4 (warmup_last25_reward > warmup_first25_reward in >= 2/3 seeds):
    Agent improves across warmup episodes -- learning occurs.

PASS = C1 + C2 + C4 (majority >= 2/3 seeds each). C3 informational only.

Estimated runtime: ~80 min (3 seeds x 150 eps x ~0.17 min/ep on Mac CPU)
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


EXPERIMENT_TYPE    = "v3_exq_223_minimal_vertebrate_ablation"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = []

# Pre-registered thresholds
THRESH_C1 = 0.0   # ree_mean_reward > rand_mean_reward (any improvement over random)
THRESH_C2 = 1.0   # ree/rand harm ratio < 1.0 (any harm reduction vs random)
THRESH_C3 = 2     # informational: mean hippocampal candidates >= 2 per e3 tick
THRESH_C4 = 0.0   # warmup last-25 reward > warmup first-25 reward (positive delta)

# Episode logging thresholds (for mode classification)
HARM_MODE_THRESH   = 0.25   # z_harm norm -> avoid
EXPLORE_ERR_THRESH = 0.10   # z_world change norm -> explore (novelty proxy)

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
        z_goal_enabled=False,       # no latent goal substrate
        benefit_eval_enabled=False, # no benefit eval head
        goal_weight=0.0,            # no goal scoring in E3
    )
    # ABLATION: force always-uncommitted mode.
    # _running_variance is always >= 0; threshold=-1.0 means committed = False always.
    # Disables: argmin selection, BetaGate action-holding, _committed_trajectory.
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
    """
    Train E1 associations, E2 world forward model, and E3 harm_eval
    on the ablated agent. No commitment architecture active.

    Three simultaneous learning components:
      E1: long-horizon associative prediction (LSTM world model)
      E2: world forward model z_w -> z_w' (for trajectory scoring)
      E3 harm_eval: supervised on raw harm_signal from environment
    """
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

    wf_buf:       List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]]              = []
    reward_log:   List[float] = []

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

            # Buffer: E2 world forward (z_world domain)
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            # Buffer: E3 harm_eval (supervised by raw environmental signal)
            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            # Train E2 world forward
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

            # Train E3 harm_eval head (minimal vertebrate harm signal)
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

            # Train E1 (association learning -- populates via sense())
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

    first25 = float(np.mean(reward_log[:25]))   if len(reward_log) >= 25 else float(np.mean(reward_log))
    last25  = float(np.mean(reward_log[-25:]))  if len(reward_log) >= 25 else float(np.mean(reward_log))

    return {
        "final_running_variance": agent.e3._running_variance,
        "warmup_first25_reward":  first25,
        "warmup_last25_reward":   last25,
    }


# ---------------------------------------------------------------------------
# Episode logging helpers
# ---------------------------------------------------------------------------

def _classify_mode(z_harm_norm: float, world_change_norm: float, harm_signal: float) -> str:
    """Classify behavioural mode from latent signals (per step)."""
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
    """Evaluate REE ablated agent or random baseline."""
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
        f"\n[EXQ-223] Seed {seed}"
        f"  warmup={warmup_eps}  eval={eval_eps}"
        f"  dry_run={dry_run}",
        flush=True,
    )

    agent, env = _make_agent_and_env(seed)

    print(f"[EXQ-223] Seed {seed} -- Phase 0: warmup", flush=True)
    warmup = _warmup_train(agent, env, warmup_eps, STEPS_PER_EPISODE)

    print(f"[EXQ-223] Seed {seed} -- Phase 1: REE ablated eval", flush=True)
    ree = _eval_agent(agent, env, eval_eps, STEPS_PER_EPISODE,
                      use_random=False, label="REE-ablated", record_episodes=True)

    print(f"[EXQ-223] Seed {seed} -- Phase 2: random baseline", flush=True)
    rnd = _eval_agent(agent, env, eval_eps, STEPS_PER_EPISODE,
                      use_random=True, label="random")

    improvement_delta = warmup["warmup_last25_reward"] - warmup["warmup_first25_reward"]
    harm_ratio = (
        ree["mean_harm"] / rnd["mean_harm"]
        if rnd["mean_harm"] > 1e-8 else 0.0
    )

    print(
        f"\n[EXQ-223] Seed {seed} summary:\n"
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
        f"[V3-EXQ-223] Minimal Vertebrate Cognition Ablation\n"
        f"  Seeds: {seeds}\n"
        f"  Warmup: {WARMUP_EPISODES} eps  Eval: {EVAL_EPISODES} eps"
        f"  Steps/ep: {STEPS_PER_EPISODE}\n"
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
    c3_pass = c3_seeds >= majority  # informational
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

    summary_markdown = f"""# V3-EXQ-223 -- Minimal Vertebrate Cognition Ablation

**Status:** {status}
**Purpose:** Milestone validation -- core E1/E2/hippocampus loop as gradient-follower
**Ablation:** commitment_threshold=-1.0 (always uncommitted, no argmin, no BetaGate hold)
             z_goal_enabled=False, benefit_eval_enabled=False, goal_weight=0.0
**Warmup:** {WARMUP_EPISODES} eps | **Eval:** {EVAL_EPISODES} eps | **Seeds:** {seeds}

## Evolutionary Hypothesis

If REE correctly decomposes vertebrate cognition, the minimal loop
(E1 associations + E2 transitions + hippocampal sequences + multinomial
go/no-go + raw harm/goal signals) should produce gradient-following
behavior without the commitment architecture.

Sidesteps SD-011 (z_world perp z_harm) and SD-012 (z_goal seeding failures)
by using raw environmental signals, not latent harm/goal substrates.

## Ablation Config

- commitment_threshold = -1.0  (committed = rv < -1.0 = always False)
- z_goal_enabled = False
- benefit_eval_enabled = False
- goal_weight = 0.0

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

    # Write companion episode log (separate file; popped from result to keep metrics JSON lean)
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
