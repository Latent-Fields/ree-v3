"""
V3-EXQ-475 -- SD-036 GABAergic Decay Unlocks EXQ-471

Claims: SD-036, MECH-279

EXPERIMENT_PURPOSE = "diagnostic"

Diagnostic re-run of V3-EXQ-471 (best-available V3 agent fishtank showcase)
with the SD-036 GABAergic cross-stream decay regulator and the MECH-279 PAG
freeze-gate enabled. Hypothesis (per
REE_assembly/docs/architecture/sd_036_gabaergic_decay_regulator.md):

  EXQ-471 fishtank reveals that the agent gets locked into a single
  monostrategy mode -- z_harm_a (and to a lesser extent z_beta) accumulate
  monotonically without a relaxation mechanism, so the salience coordinator
  / dACC bundle keeps re-hitting the same operating mode. SD-036 introduces
  a tonic GABAergic exponential decay on z_harm_s, z_harm_a, and z_beta.
  Predicted observable: mode flip ~t=50 once accumulated affective load
  decays below the commitment / mode-switch thresholds. MECH-279 PAG freeze-
  gate adds a committed catatonia state for sustained extreme z_harm_a;
  expected to be silent on EXQ-471 dynamics (entries should be rare under
  the default theta_freeze=2.0) but is wired so the new substrate is
  exercised end-to-end.

Reference: V3-EXQ-471 (best-available V3 agent fishtank showcase). This is
NOT a supersede -- EXQ-471 stands as the no-decay baseline. EXQ-475 is the
matched diagnostic with use_gabaergic_decay=True + use_pag_freeze_gate=True.

Output:
  evidence/experiments/v3_exq_475_sd036_decay_unlocks_exq471/
    v3_exq_475_sd036_decay_unlocks_exq471_<ts>.json         (manifest)
    v3_exq_475_sd036_decay_unlocks_exq471_<ts>_episode_log.json  (fishtank feed)

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
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE    = "v3_exq_475_sd036_decay_unlocks_exq471"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = ["SD-036", "MECH-279"]

# Fishtank mode classifier thresholds (same as EXQ-223a / EXQ-471 so the
# colour scheme in the viz is consistent across runs).
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
    resource_respawn_on_consume=True,
    use_proxy_fields=True,
    toroidal=False,
    harm_history_len=10,
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
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
LR_ENC_AUX        = 5e-4


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
        # SD-012
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # SD-022
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        # SD-036 / MECH-279 -- the diagnostic delta vs EXQ-471
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
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


def _classify_mode(z_harm_norm: float, world_change_norm: float, harm_signal: float) -> str:
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


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
    # SD-036 / MECH-279 diagnostic counters
    freeze_commit_count = 0
    freeze_active_steps = 0

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

            # SD-036/MECH-279 diagnostics: poll the freeze gate after select_action
            if agent.pag_freeze_gate is not None:
                last = agent.pag_freeze_gate.last_output
                if last.freeze_commit:
                    freeze_commit_count += 1
                if last.freeze_active:
                    freeze_active_steps += 1

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

    # SD-036 diagnostics from the regulator (final-state n_ticks etc.)
    decay_diag = (
        agent.gabaergic_decay.diagnostics if agent.gabaergic_decay is not None else {}
    )
    pag_diag = (
        agent.pag_freeze_gate.diagnostics if agent.pag_freeze_gate is not None else {}
    )

    return {
        "mean_reward":         float(np.mean(episode_rewards)),
        "mean_harm":           float(np.mean(episode_harms)),
        "rewards":             episode_rewards,
        "mean_n_cands":        float(np.mean(n_cands_log)) if n_cands_log else 0.0,
        "episodes":            episodes_log,
        "freeze_commit_count": int(freeze_commit_count),
        "freeze_active_steps": int(freeze_active_steps),
        "decay_n_ticks":       int(decay_diag.get("n_ticks", 0)),
        "pag_n_ticks":         int(pag_diag.get("n_ticks", 0)),
        "pag_n_commits":       int(pag_diag.get("n_commits", 0)),
        "pag_n_releases":      int(pag_diag.get("n_releases", 0)),
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps   = 2 if dry_run else EVAL_EPISODES
    steps      = 30 if dry_run else STEPS_PER_EPISODE

    print(
        f"\n[EXQ-475] Seed {seed}  warmup={warmup_eps}  eval={eval_eps}"
        f"  steps/ep={steps}  dry_run={dry_run}",
        flush=True,
    )

    agent, env = _make_agent_and_env(seed)

    print(f"[EXQ-475] Seed {seed} -- Phase 0: warmup (dual-stream + SD-018 aux + SD-036 decay)", flush=True)
    warmup = _warmup_train(agent, env, warmup_eps, steps)

    print(f"[EXQ-475] Seed {seed} -- Phase 1: eval (recorded for fishtank)", flush=True)
    ree = _eval_agent(agent, env, eval_eps, steps, label="sd036_decay")

    improvement_delta = warmup["warmup_last10_reward"] - warmup["warmup_first10_reward"]

    print(
        f"\n[EXQ-475] Seed {seed} summary:\n"
        f"  warmup first10={warmup['warmup_first10_reward']:.4f}"
        f"  last10={warmup['warmup_last10_reward']:.4f}"
        f"  delta={improvement_delta:.4f}\n"
        f"  eval reward={ree['mean_reward']:.4f}"
        f"  harm={ree['mean_harm']:.4f}"
        f"  n_cands={ree['mean_n_cands']:.1f}\n"
        f"  pag commits={ree['pag_n_commits']} releases={ree['pag_n_releases']}"
        f"  freeze_active_steps={ree['freeze_active_steps']}",
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
        "freeze_commit_count":   ree["freeze_commit_count"],
        "freeze_active_steps":   ree["freeze_active_steps"],
        "decay_n_ticks":         ree["decay_n_ticks"],
        "pag_n_ticks":           ree["pag_n_ticks"],
        "pag_n_commits":         ree["pag_n_commits"],
        "pag_n_releases":        ree["pag_n_releases"],
        "episodes":              ree["episodes"],
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    print(
        f"[V3-EXQ-475] SD-036 GABAergic Decay Unlocks EXQ-471\n"
        f"  Seeds: {seeds}\n"
        f"  Warmup: {WARMUP_EPISODES} eps  Eval: {EVAL_EPISODES} eps"
        f"  Steps/ep: {STEPS_PER_EPISODE}\n"
        f"  Substrate on (delta vs EXQ-471): SD-036 (use_gabaergic_decay), MECH-279 (use_pag_freeze_gate)\n"
        f"  Other substrate: SD-007 SD-008 SD-010 SD-011 SD-012 SD-018 SD-021 SD-022 SD-023 MECH-090\n"
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
            "freeze_commit_count", "freeze_active_steps",
            "decay_n_ticks", "pag_n_ticks", "pag_n_commits", "pag_n_releases",
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
            f" | {r['eval_mean_n_cands']:.1f}"
            f" | {r['pag_n_commits']}"
            f" | {r['pag_n_releases']}"
            f" | {r['freeze_active_steps']} |\n"
        )

    summary_markdown = f"""# V3-EXQ-475 -- SD-036 GABAergic Decay Unlocks EXQ-471

**Status:** N/A (diagnostic -- not scored)
**Purpose:** Diagnostic re-run of EXQ-471 with SD-036 GABAergic cross-stream decay
regulator and MECH-279 PAG freeze-gate enabled. Hypothesis: tonic decay on
z_harm_s / z_harm_a / z_beta unlocks the EXQ-471 monostrategy lock-in by
providing a relaxation mechanism that lets the salience coordinator / dACC
bundle re-enter alternative operating modes.

**Substrate enabled (delta vs EXQ-471):** SD-036 (use_gabaergic_decay=True),
MECH-279 (use_pag_freeze_gate=True). All other EXQ-471 flags retained.

**Reference:** V3-EXQ-471 (best-available agent fishtank showcase) -- NOT a
supersede. EXQ-471 stands as the no-decay baseline; EXQ-475 is the matched
diagnostic with SD-036 ON.

**Warmup:** {WARMUP_EPISODES} eps | **Eval:** {EVAL_EPISODES} eps | **Steps/ep:** {STEPS_PER_EPISODE}

## Per-seed Metrics

| Seed | W-first10 | W-last10 | W-delta | eval reward | eval harm | n_cands | pag_commits | pag_releases | freeze_active |
|------|-----------|----------|---------|-------------|-----------|---------|-------------|--------------|---------------|
{rows}
The `_episode_log.json` companion file is the payload fishtank_viz.html
auto-discovers via `/api/fishtank/logs`.
"""

    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "env_config":      ENV_KWARGS,
        "phase":           "eval_sd036_decay",
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

    # --- runner-conformance sentinel (added by retrofit_experiments.py) ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
