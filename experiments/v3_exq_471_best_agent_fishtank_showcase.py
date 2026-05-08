"""
V3-EXQ-471 -- Best-Available V3 Agent Fishtank Showcase

Claims: None (demo / hypothesis generation)

EXPERIMENT_PURPOSE = "diagnostic"

Showcase run for fishtank_viz.html. Derived from EXQ-223a (which was
the only script that already emitted the *_episode_log.json the fishtank
auto-discovers) but the minimal-vertebrate ablation is LIFTED. We turn
on the best-available V3 substrate -- everything that has landed and
smoke-passed in ree_core as of 2026-04-21:

  Perception / representation
    SD-008  alpha_world=0.9                          (encoder.z_world_alpha_correction)
    SD-007  reafference_action_dim=env.action_dim    (perspective correction)
    SD-010  use_harm_stream=True                     (sensory-discriminative z_harm_s)
    SD-011  use_affective_harm_stream=True, harm_history_len=10
                                                     (dual nociceptive streams)
    SD-018  use_resource_proximity_head=True         (resource proximity supervision)
    SD-015  (implicit via z_goal seeding; z_resource encoder NOT enabled --
             it requires its own phased training and is not load-bearing for the
             fishtank demo; z_goal is seeded from z_world at resource contact)

  Homeostatic / goal system
    SD-012  z_goal_enabled=True, drive_weight=2.0, goal_weight=0.5
                                                     (homeostatic drive modulation)
    benefit_eval_enabled=True
    e1_goal_conditioned=True (default)

  Commitment / control plane
    MECH-090 beta_gate_bistable=True                 (bistable commitment latch)
    commitment_threshold=0.5                         (not the -1.0 ablation)
    urgency_interrupt_threshold default              (MECH-091)

  Affective-sensory coupling
    SD-021  harm_descending_mod_enabled=True         (z_harm_s attenuated when beta elevated)

  Body / environment
    SD-022  limb_damage_enabled=True                 (directional limb damage + residual pain)
    SD-023  n_landmarks_b=2                          (predictive resource cues)

Everything that requires phased encoder warmup OR that defaults to False
and has its own dedicated landing diagnostic (SD-032 cingulate cluster,
SD-033a lateral PFC, SD-034 closure operator, MECH-266/267/268,
serotonergic sleep, SWS/REM passes, SHY, schema_wanting, SD-013
interventional, SD-019 non-redundancy, SD-020 surprise-PE target,
ARC-033 E2_harm_s forward, tonic_5ht) is LEFT OFF. This script is a
best-of-what-runs-out-of-the-box demo, not a claim-weighted experiment.

Env: CausalGridWorldV2, non-toroidal (walls present -- gives the
replay spatial structure viewers can follow), size=10, 3 hazards,
5 resources, resource_respawn_on_consume=True so the agent keeps
having goal cycles throughout each recorded episode.

Output:
  evidence/experiments/v3_exq_471_best_agent_fishtank_showcase/
    v3_exq_471_best_agent_fishtank_showcase_<ts>.json         (manifest)
    v3_exq_471_best_agent_fishtank_showcase_<ts>_episode_log.json  (fishtank feed)

The `_episode_log.json` file is what fishtank_viz.html/serve.py's
/api/fishtank/logs endpoint auto-discovers.

Estimated runtime: ~20 min on Mac CPU (3 seeds x [60 warmup + 5 eval] eps x 200 steps)
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


EXPERIMENT_TYPE    = "v3_exq_471_best_agent_fishtank_showcase"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = []

# Fishtank mode classifier thresholds (same as EXQ-223a so the colour
# scheme in the viz is consistent across runs).
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
    resource_respawn_on_consume=True,   # keeps goal cycling across a recorded episode
    use_proxy_fields=True,              # SD-022 body proxy mode (needed for 17-dim body_state)
    toroidal=False,
    # SD-011 second source: rolling harm-history window
    harm_history_len=10,
    # SD-022: directional limb damage
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
    # SD-023: predictive resource cue landmarks
    n_landmarks_b=2,
)

WARMUP_EPISODES   = 60
EVAL_EPISODES     = 5
STEPS_PER_EPISODE = 200
WORLD_DIM         = 32
SELF_DIM          = 32
HARM_DIM          = 32
HARM_A_DIM        = 16
HARM_HISTORY_LEN  = 10

WF_BUF_MAX        = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE        = 32
LR_E1             = 1e-4
LR_E2_WF          = 3e-4
LR_E3_HARM        = 1e-3
LR_ENC_AUX        = 5e-4    # resource-prox + harm-accum auxiliary heads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs")


def _obs_harm_a(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    # max over resource_field_view is SD-018's proxy proximity target.
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def _make_agent_and_env(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        # SD-010 / SD-011
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        # SD-018: resource-prox supervision on z_world
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        # SD-012: homeostatic drive-modulated benefit + goal system
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # SD-022: directional limb damage (env also has it enabled in ENV_KWARGS)
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
    )
    # Non-from_dims config tweaks.
    config.e3.commitment_threshold = 0.5              # MECH-090 realistic threshold
    config.heartbeat.beta_gate_bistable = True        # MECH-090 bistable latch
    config.harm_descending_mod_enabled = True         # SD-021
    config.descending_attenuation_factor = 0.5

    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0: Warmup training with dual-stream auxiliary losses
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

    # Auxiliary head optimiser: resource-prox + harm-accum heads live on
    # the latent stack (SplitEncoder + AffectiveHarmEncoder). Training
    # signal flows BACK through the encoders, which is the SD-018 +
    # SD-011 intent.
    aux_params: List[torch.nn.Parameter] = []
    for p in agent.latent_stack.parameters():
        aux_params.append(p)
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

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
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            prox_t    = _obs_resource_prox(obs_dict)
            accum_t   = _obs_accum(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h,
                obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            # SD-018 + SD-011 auxiliary losses. Both use the same optimiser
            # but the computation depends on the most recent sense()'s
            # latent_state, so run them BEFORE the next sense() clobbers it.
            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            accum_target_t = torch.tensor([[accum_t]], device=device)
            harm_accum_loss = agent.compute_harm_accum_loss(accum_target_t, latent)
            if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                aux_terms.append(harm_accum_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            # Re-sense to refresh latent cache for action selection
            # (aux backward didn't mutate _current_latent, but it did
            # consume the graph on the previous latent's tensors).
            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h,
                obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            # SD-012: seed / refresh z_goal from z_world when benefit exposure
            # crosses threshold (update_z_goal internally handles the gating).
            drive_level = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
            agent.update_z_goal(
                benefit_exposure=benefit_exposure,
                drive_level=drive_level,
            )

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

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [warmup] ep {ep+1}/{num_episodes}"
                f"  rv={rv:.4f}"
                f"  ep_reward={ep_reward:.4f}",
                flush=True,
            )

    first10 = float(np.mean(reward_log[:10]))  if len(reward_log) >= 10 else float(np.mean(reward_log))
    last10  = float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log))

    return {
        "final_running_variance": agent.e3._running_variance,
        "warmup_first10_reward":  first10,
        "warmup_last10_reward":   last10,
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
# Phase 1: Evaluation with per-step recording for fishtank
# ---------------------------------------------------------------------------

def _eval_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    label: str = "",
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

        ep_steps: List[Dict] = []
        initial_hazards   = [list(h) for h in env.hazards]
        initial_resources = [list(r) for r in env.resources]
        current_hazards   = [list(h) for h in env.hazards]
        current_resources = [list(r) for r in env.resources]

        for step_idx in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world,
                    obs_harm=obs_h,
                    obs_harm_a=obs_h_a,
                    obs_harm_history=obs_h_h,
                )
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

                drive_level = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

                if ticks.get("e3_tick", True):
                    n_cands_log.append(len(candidates))
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, action_dim - 1), action_dim, device
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

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
        episodes_log.append({
            "ep":                ep_idx,
            "initial_hazards":   initial_hazards,
            "initial_resources": initial_resources,
            "steps":             ep_steps,
        })

        print(
            f"  [eval {label}] ep {ep_idx+1}/{num_episodes}"
            f"  reward={ep_reward:.4f}  harm={ep_harm:.4f}  steps={len(ep_steps)}",
            flush=True,
        )

    return {
        "mean_reward":  float(np.mean(episode_rewards)),
        "mean_harm":    float(np.mean(episode_harms)),
        "rewards":      episode_rewards,
        "mean_n_cands": float(np.mean(n_cands_log)) if n_cands_log else 0.0,
        "episodes":     episodes_log,
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool = False) -> Dict:
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps   = 2 if dry_run else EVAL_EPISODES
    steps      = 30 if dry_run else STEPS_PER_EPISODE

    print(
        f"\n[EXQ-471] Seed {seed}  warmup={warmup_eps}  eval={eval_eps}"
        f"  steps/ep={steps}  dry_run={dry_run}",
        flush=True,
    )

    agent, env = _make_agent_and_env(seed)

    print(f"[EXQ-471] Seed {seed} -- Phase 0: warmup (dual-stream + SD-018 aux losses)", flush=True)
    warmup = _warmup_train(agent, env, warmup_eps, steps)

    print(f"[EXQ-471] Seed {seed} -- Phase 1: eval (recorded for fishtank)", flush=True)
    ree = _eval_agent(agent, env, eval_eps, steps, label="showcase")

    improvement_delta = warmup["warmup_last10_reward"] - warmup["warmup_first10_reward"]

    print(
        f"\n[EXQ-471] Seed {seed} summary:\n"
        f"  warmup first10={warmup['warmup_first10_reward']:.4f}"
        f"  last10={warmup['warmup_last10_reward']:.4f}"
        f"  delta={improvement_delta:.4f}\n"
        f"  eval reward={ree['mean_reward']:.4f}"
        f"  harm={ree['mean_harm']:.4f}"
        f"  n_cands={ree['mean_n_cands']:.1f}",
        flush=True,
    )

    return {
        "seed":                  seed,
        "warmup_first10_reward": warmup["warmup_first10_reward"],
        "warmup_last10_reward":  warmup["warmup_last10_reward"],
        "warmup_improvement":    improvement_delta,
        "warmup_final_rv":       warmup["final_running_variance"],
        "eval_mean_reward":      ree["mean_reward"],
        "eval_mean_harm":        ree["mean_harm"],
        "eval_mean_n_cands":     ree["mean_n_cands"],
        "episodes":              ree["episodes"],
    }


# ---------------------------------------------------------------------------
# Aggregate + output
# ---------------------------------------------------------------------------

def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    print(
        f"[V3-EXQ-471] Best-Available V3 Agent Fishtank Showcase\n"
        f"  Seeds: {seeds}\n"
        f"  Warmup: {WARMUP_EPISODES} eps  Eval: {EVAL_EPISODES} eps"
        f"  Steps/ep: {STEPS_PER_EPISODE}\n"
        f"  Substrate on: SD-007 SD-008 SD-010 SD-011 SD-012 SD-018 SD-021 SD-022 SD-023 MECH-090\n"
        f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/",
        flush=True,
    )

    seed_results = [run_seed(s, dry_run=dry_run) for s in seeds]

    metrics: Dict = {
        "n_seeds": float(len(seeds)),
    }
    for r in seed_results:
        s = r["seed"]
        for key in (
            "warmup_first10_reward", "warmup_last10_reward", "warmup_improvement",
            "warmup_final_rv",
            "eval_mean_reward", "eval_mean_harm", "eval_mean_n_cands",
        ):
            metrics[f"seed{s}_{key}"] = float(r[key])

    rows = ""
    for r in seed_results:
        rows += (
            f"| {r['seed']}"
            f" | {r['warmup_first10_reward']:.4f}"
            f" | {r['warmup_last10_reward']:.4f}"
            f" | {r['warmup_improvement']:.4f}"
            f" | {r['eval_mean_reward']:.4f}"
            f" | {r['eval_mean_harm']:.4f}"
            f" | {r['eval_mean_n_cands']:.1f} |\n"
        )

    summary_markdown = f"""# V3-EXQ-471 -- Best-Available V3 Agent Fishtank Showcase

**Status:** N/A (diagnostic demo -- not scored)
**Purpose:** Showcase a full-stack V3 agent in fishtank_viz.html. All smoke-passed
substrate flags turned on; phased-training-required or still-under-validation
substrates left off.

**Substrate enabled:** SD-007 (reafference), SD-008 (alpha_world=0.9), SD-010
(sensory harm stream), SD-011 (affective harm stream + history), SD-012
(drive-modulated goal), SD-018 (resource proximity supervision), SD-021
(descending modulation), SD-022 (directional limb damage), SD-023 (Landmark B),
MECH-090 (bistable beta gate).

**Warmup:** {WARMUP_EPISODES} eps | **Eval:** {EVAL_EPISODES} eps | **Steps/ep:** {STEPS_PER_EPISODE}

## Per-seed Metrics

| Seed | W-first10 | W-last10 | W-delta | eval reward | eval harm | n_cands |
|------|-----------|----------|---------|-------------|-----------|---------|
{rows}
The `_episode_log.json` companion file is the payload fishtank_viz.html
auto-discovers via `/api/fishtank/logs`.
"""

    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "env_config":      ENV_KWARGS,
        "phase":           "eval_showcase",
        "toroidal":        ENV_KWARGS.get("toroidal", False),
        "seeds": [
            {"seed": r["seed"], "episodes": r.get("episodes", [])}
            for r in seed_results
        ],
    }

    return {
        "status":             "N/A",
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
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
