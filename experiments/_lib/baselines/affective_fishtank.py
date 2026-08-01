"""
Canonical baseline module for the `affective_fishtank` lineage.

Factored from V3-EXQ-664 (the affective fishtank showcase) so that its
default-configured OFF/harsh substrate arm -- the exact configuration that
produced the 2026-06-10 z_harm_a saturation-and-inversion observation -- can be
reused, by construction, across the lineage's consumers:

  * V3-EXQ-856 (SD-087): harm_surprise_pe_enabled OFF vs ON.
  * V3-EXQ-857 (Q-086):  harsh env vs gentler env.

Both consumers share the SAME substrate config, warmup training, and eval
collection; the only per-consumer difference is which axis is varied across arms
(the training-target flag for 856, the environment harshness for 857). Putting
that shared code here (a) removes ~400 lines of duplication and (b) makes the
OFF/harsh baseline cell's arm fingerprint match by construction for any future
different-driver sibling (mint-as-you-go, arm_reuse_fingerprint_plan.md sec 7b).

SCOPE: this is the RAW-WARMUP configuration -- a ~50-episode warmup with
scaffold_train_harm_pathway OFF and no scaffolded_sd054_onboarding, exactly as
V3-EXQ-664 ran. Conclusions drawn with this module are scoped to raw-warmup
agents (intake candidate 5 / SD-086 scope_note). The substrate config is a
verbatim port of V3-EXQ-664's `_make_agent_and_env` -- keeping it identical is
what lets the OFF arm reproduce the 664 signature (the non-degeneracy
precondition every consumer asserts before reading a verdict).

ASCII-only stdout throughout (Windows-runner safety).
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


LINEAGE               = "affective_fishtank"
CANONICAL_BASELINE_ID = "affective_fishtank_off_harsh_rawwarmup_v1"

# --- schedule (verbatim from V3-EXQ-664) -----------------------------------
WARMUP_EPISODES   = 50
EVAL_EPISODES     = 5
STEPS_PER_EPISODE = 200

# --- substrate dims (verbatim from V3-EXQ-664) -----------------------------
WORLD_DIM         = 32
SELF_DIM          = 32
HARM_DIM          = 32
HARM_A_DIM        = 16
HARM_HISTORY_LEN  = 10
PAG_MAX_FREEZE    = 8

# --- training hyperparameters (verbatim from V3-EXQ-664) -------------------
WF_BUF_MAX        = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE        = 32
LR_E1             = 1e-4
LR_E2_WF          = 3e-4
LR_E3_HARM        = 1e-3
LR_ENC_AUX        = 5e-4

# --- mode-classification thresholds (verbatim from V3-EXQ-664) -------------
HARM_MODE_THRESH    = 0.25
EXPLORE_ERR_THRESH  = 0.10
SHELTER_HARM_THRESH = 0.15
ASSERT_THRESH       = 0.10

# --- HARSH environment (the V3-EXQ-664 default; the observed-pathology env) --
HARSH_ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    scheduled_action_block_enabled=True,
    scheduled_action_block_interval=10,
    scheduled_action_block_prob=0.4,
)


# ---------------------------------------------------------------------------
# obs / latent helpers
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


def _norm(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(t.norm().item())


def _get_reef_cells_set(env: CausalGridWorldV2) -> Set:
    return getattr(env, "_reef_cells", set())


def classify_mode(
    z_harm_norm: float,
    world_change_norm: float,
    harm_signal: float,
    in_reef: bool,
    freeze: bool,
    z_block_assert: float,
) -> str:
    """Behavioural mode with affect precedence (verbatim from V3-EXQ-664):
    freeze > assert > shelter > avoid > approach > explore > neutral."""
    if freeze:
        return "freeze"
    if z_block_assert is not None and z_block_assert > ASSERT_THRESH:
        return "assert"
    if in_reef and z_harm_norm > SHELTER_HARM_THRESH:
        return "shelter"
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


def _read_freeze_block(agent: REEAgent, latent) -> Tuple[bool, float]:
    """Freeze (MECH-279) and blocked-agency assert pole (MECH-353) -- the two
    affect signals the mode classifier needs beyond the harm norms."""
    freeze = False
    pag_out = getattr(agent, "_pag_last_output", None)
    if pag_out is not None:
        freeze = bool(getattr(pag_out, "freeze_active", False))

    z_block = 0.0
    ba = getattr(agent, "blocked_agency", None)
    if ba is not None and getattr(ba, "_last_output", None) is not None:
        z_block = float(getattr(ba._last_output, "z_block_assert", 0.0))
    elif getattr(latent, "z_block", None) is not None:
        z_block = float(latent.z_block.abs().mean().item())
    return freeze, z_block


# ---------------------------------------------------------------------------
# Agent + env construction (verbatim substrate config from V3-EXQ-664)
# ---------------------------------------------------------------------------

def make_agent_and_env(
    seed: int,
    env_overrides: Optional[Dict[str, Any]] = None,
    harm_surprise_pe_enabled: bool = False,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Build the V3-EXQ-664 affective substrate for one arm.

    env_overrides:            merged over HARSH_ENV_KWARGS (used by V3-EXQ-857 to
                              make a gentler environment).
    harm_surprise_pe_enabled: SD-020 precision-weighted-PE training target for
                              z_harm_a (used by V3-EXQ-856; default False matches
                              the 664 default-trained substrate).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env_kwargs = dict(HARSH_ENV_KWARGS)
    if env_overrides:
        env_kwargs.update(env_overrides)
    env = CausalGridWorldV2(seed=seed, **env_kwargs)

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
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_tonic_vigor=True,
        use_blocked_agency=True,
        use_pag_freeze_gate=True,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5
    config.latent.use_harm_un = True
    config.pag_max_freeze_duration = PAG_MAX_FREEZE
    config.use_broadcast_override = True
    config.surprise_gated_replay = True
    config.use_mech307_split_surprise = True
    config.use_control_vector_logging = True

    # --- the SD-087 axis: SD-020 precision-weighted PE training target ---
    # Top-level REEConfig flag (config.py:2306). Default False -> z_harm_a trains
    # against the EMA accumulated-harm target; True -> against the affective
    # surprise (precision-weighted |actual - expected|) target. This flips the
    # TRAINING target inside agent.compute_harm_accum_loss (agent.py PE branch).
    config.harm_surprise_pe_enabled = bool(harm_surprise_pe_enabled)

    agent = REEAgent(config)
    return agent, env


def arm_config_slice(
    env_overrides: Optional[Dict[str, Any]],
    harm_surprise_pe_enabled: bool,
    p0: int = WARMUP_EPISODES,
    p1: int = EVAL_EPISODES,
    steps: int = STEPS_PER_EPISODE,
) -> Dict[str, Any]:
    """Declared config slice for this arm's fingerprint.

    Captures ONLY what the arm's build+collect path reads: the resolved env
    kwargs, the schedule, and the one substrate flag that differs between arms
    (harm_surprise_pe_enabled). The rest of the substrate config is fixed for the
    lineage and is folded into the substrate_hash via the ree_core glob, so it
    need not be restated here. The OFF/harsh baseline slice (env_overrides=None,
    harm_surprise_pe_enabled=False) is the reuse-eligible mint for the lineage.
    """
    env_kwargs = dict(HARSH_ENV_KWARGS)
    if env_overrides:
        env_kwargs.update(env_overrides)
    return {
        "baseline_id": CANONICAL_BASELINE_ID,
        "lineage": LINEAGE,
        "env_kwargs": env_kwargs,
        "schedule": {"p0": int(p0), "p1": int(p1), "steps_per_episode": int(steps)},
        "substrate_dims": {
            "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
            "harm_dim": HARM_DIM, "harm_a_dim": HARM_A_DIM,
            "harm_history_len": HARM_HISTORY_LEN, "pag_max_freeze": PAG_MAX_FREEZE,
        },
        "harm_surprise_pe_enabled": bool(harm_surprise_pe_enabled),
    }


# ---------------------------------------------------------------------------
# Phase 0: warmup training (verbatim dual-stream aux losses from V3-EXQ-664)
# ---------------------------------------------------------------------------

def warmup_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    tag: str = "",
) -> Dict[str, Any]:
    device     = agent.device
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
    aux_params: List[torch.nn.Parameter] = list(agent.latent_stack.parameters())
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
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            accum_target_t = torch.tensor([[accum_t]], device=device)
            # SD-087 axis lives HERE: with harm_surprise_pe_enabled, this loss
            # trains z_harm_a against the precision-weighted PE target instead of
            # the raw accumulated-harm target.
            harm_accum_loss = agent.compute_harm_accum_loss(accum_t, latent)
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
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level      = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
            agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)

            agent.update_residue(float(harm_signal))

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
                        list(agent.e2.world_action_encoder.parameters()), 1.0)
                    e2_wf_optimizer.step()

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs   = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b   = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b   = torch.stack([harm_eval_buf[i][1] for i in idxs]).to(device)
                harm_pred = agent.e3.harm_eval_head(zw_b)
                harm_loss = F.mse_loss(harm_pred, ht_b)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 1.0)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        reward_log.append(ep_reward)
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  [train]{tag} seed={seed} ep {ep+1}/{num_episodes} "
                  f"reward={ep_reward:.4f}", flush=True)

    first10 = float(np.mean(reward_log[:10])) if reward_log else 0.0
    last10  = float(np.mean(reward_log[-10:])) if reward_log else 0.0
    return {
        "warmup_first10_reward": first10,
        "warmup_last10_reward":  last10,
        "harm_obs_ema_after_warmup": float(getattr(agent, "_harm_obs_ema", 0.0)),
    }


# ---------------------------------------------------------------------------
# Phase 1: eval collection (per-step z_harm_s / z_harm_a / mode, per episode)
# ---------------------------------------------------------------------------

def eval_collect(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    tag: str = "",
) -> Dict[str, Any]:
    """Run eval episodes, returning per-episode per-step z_harm_a / z_harm_s and
    behavioural mode. This is the affective-fingerprint readout the consumers'
    DVs are computed from -- everything else (drive, vigor, valence, episode_log)
    that V3-EXQ-664 emitted for the viz is intentionally dropped."""
    device     = agent.device
    action_dim = env.action_dim
    agent.eval()

    episodes: List[Dict[str, Any]] = []
    ep_rewards: List[float] = []
    ep_harms:   List[float] = []

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        reef_cells_set = _get_reef_cells_set(env)

        step_z_harm_a: List[float] = []
        step_z_harm_s: List[float] = []
        step_mode:     List[str]   = []
        ep_reward = 0.0
        ep_harm   = 0.0

        for step_idx in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)

            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world,
                    obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
                )
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                drive_level      = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
                agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            with torch.no_grad():
                agent.update_residue(float(harm_signal))

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef   = agent_pos in reef_cells_set

            z_harm_s = _norm(latent.z_harm)
            z_harm_a = _norm(latent.z_harm_a)
            z_harm_s = z_harm_s if z_harm_s is not None else 0.0
            z_harm_a = z_harm_a if z_harm_a is not None else 0.0

            freeze, z_block = _read_freeze_block(agent, latent)
            world_change_norm = (
                float((latent.z_world - z_world_prev).norm().item())
                if z_world_prev is not None else 0.0
            )
            mode = classify_mode(
                z_harm_s, world_change_norm, float(harm_signal),
                in_reef, freeze, z_block,
            )

            step_z_harm_a.append(z_harm_a)
            step_z_harm_s.append(z_harm_s)
            step_mode.append(mode)

            z_self_prev  = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()
            ep_reward += float(harm_signal)
            if float(harm_signal) < 0:
                ep_harm += abs(float(harm_signal))
            if done:
                break

        episodes.append({
            "ep": ep_idx,
            "z_harm_a": step_z_harm_a,
            "z_harm_s": step_z_harm_s,
            "mode":     step_mode,
        })
        ep_rewards.append(ep_reward)
        ep_harms.append(ep_harm)
        print(f"  [eval]{tag} seed={seed} ep {ep_idx+1}/{num_episodes} "
              f"reward={ep_reward:.4f} harm={ep_harm:.4f} steps={len(step_z_harm_a)}",
              flush=True)

    return {
        "episodes":    episodes,
        "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "mean_harm":   float(np.mean(ep_harms)) if ep_harms else 0.0,
        "n_eval_steps": int(sum(len(e["z_harm_a"]) for e in episodes)),
    }
