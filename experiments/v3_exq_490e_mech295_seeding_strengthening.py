#!/opt/local/bin/python3
"""
V3-EXQ-490e -- MECH-295 seeding-strengthening successor to 490c.

V3-EXQ-490c answered the relevant branch of Q-040b: the V_s gate fired in both
arms, but MECH-295 cue fires, dACC score bias, and approach commits all stayed
at zero. The prewritten 490c successor tree routes that sub-signature to 490e:
the bridge was configured but silent, so the first discriminator is whether the
upstream activation floors are blocking MECH-295 before cue magnitude or E3
candidate generation can be tested.

Claims (claim_ids): ["Q-040"]. This is an architectural Q-040b config probe,
not a clean MECH-295 weak-reading test, because both arms keep MECH-295 ON and
vary activation floors / drive-to-liking gain.

Arms
----
    BASELINE: bridge ON with default MECH-295 knobs:
              min_drive_to_fire=0.1, min_z_goal_norm_to_fire=0.05,
              drive_to_liking_gain=1.0.
    RELAXED : bridge ON with relaxed upstream gates:
              min_drive_to_fire=0.01, min_z_goal_norm_to_fire=0.005,
              drive_to_liking_gain=2.0.

Both arms keep the same V_s stack and 490b threshold override used by 490c:
use_vs_rollout_gating=True, full per-stream/per-region V_s invalidation ON,
use_broadcast_override=True, use_dacc=True, drive_weight=2.0, PAG freeze ON,
GABAergic decay ON, and V_s thresholds 0.85/0.85/0.95.

Acceptance
----------
    C1: V_s gate fires in both arms.
    C2: RELAXED bridge_n_cue_fires_total >= 10x BASELINE, or >0 when
        BASELINE is zero.
    C3: RELAXED bridge_cue_bias_mean_max_abs > BASELINE.
    C4: RELAXED approach_commit_count > 0 in >= 2/3 seeds.

PASS = C1 AND C2 AND C3 AND C4.

FAIL routing:
    C2 FAIL with C1 PASS -> upstream gate is not the blocker; route to 490g
        instrumentation.
    C2 PASS and C4 FAIL -> bridge fires more often but still does not move
        behavior; compose with 490d cue-gain sweep.

experiment_purpose=evidence. This is a successor, not a supersession, of
V3-EXQ-490c.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_490e_mech295_seeding_strengthening"
CLAIM_IDS = ["Q-040"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 19]
ARMS = [
    {
        "id": "BASELINE",
        "mech295_min_drive_to_fire": 0.1,
        "mech295_min_z_goal_norm_to_fire": 0.05,
        "mech295_drive_to_liking_gain": 1.0,
        "mech295_liking_to_approach_cue_gain": 0.5,
    },
    {
        "id": "RELAXED",
        "mech295_min_drive_to_fire": 0.01,
        "mech295_min_z_goal_norm_to_fire": 0.005,
        "mech295_drive_to_liking_gain": 2.0,
        "mech295_liking_to_approach_cue_gain": 0.5,
    },
]

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

WARMUP_EPISODES   = 80
EVAL_EPISODES     = 5
STEPS_PER_EPISODE = 200

WORLD_DIM   = 32
SELF_DIM    = 32
HARM_DIM    = 32
HARM_A_DIM  = 16
HARM_HISTORY_LEN = 10

WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32
LR_E1      = 1e-4
LR_E2_WF   = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4

HARM_MODE_THRESH   = 0.25
EXPLORE_ERR_THRESH = 0.10


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict):       return obs_dict.get("harm_obs")
def _obs_harm_a(obs_dict):     return obs_dict.get("harm_obs_a")
def _obs_harm_history(obs_dict): return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def _clear_bridge_eval_diagnostics(agent: REEAgent) -> None:
    br = getattr(agent, "mech295_bridge", None)
    if br is None:
        return
    br._n_write_fires = 0
    br._n_cue_fires = 0
    br._exq_write_drive_log = []
    br._exq_write_z_goal_norm_log = []
    br._exq_write_value_log = []
    br._exq_cue_drive_log = []
    br._exq_cue_bias_max_abs_log = []
    br._exq_cue_bias_mean_abs_log = []
    br._exq_candidate_prox_max_log = []
    br._exq_candidate_prox_mean_log = []


def _instrument_mech295_bridge(agent: REEAgent) -> None:
    """Add experiment-local diagnostics without changing substrate code."""
    br = getattr(agent, "mech295_bridge", None)
    if br is None or getattr(br, "_exq490e_instrumented", False):
        return
    _clear_bridge_eval_diagnostics(agent)

    original_write = br.compute_anticipatory_liking_write
    original_cue = br.compute_approach_cue_score_bias

    def write_wrapper(*, drive_level: float, z_goal_norm: float,
                      simulation_mode: bool = False) -> float:
        value = original_write(
            drive_level=drive_level,
            z_goal_norm=z_goal_norm,
            simulation_mode=simulation_mode,
        )
        br._exq_write_drive_log.append(float(drive_level))
        br._exq_write_z_goal_norm_log.append(float(z_goal_norm))
        br._exq_write_value_log.append(float(value))
        return value

    def cue_wrapper(*, drive_level: float, candidate_proximities: torch.Tensor,
                    simulation_mode: bool = False) -> torch.Tensor:
        bias = original_cue(
            drive_level=drive_level,
            candidate_proximities=candidate_proximities,
            simulation_mode=simulation_mode,
        )
        if bias.numel() > 0:
            br._exq_cue_bias_max_abs_log.append(float(bias.abs().max().item()))
            br._exq_cue_bias_mean_abs_log.append(float(bias.abs().mean().item()))
        else:
            br._exq_cue_bias_max_abs_log.append(0.0)
            br._exq_cue_bias_mean_abs_log.append(0.0)
        if candidate_proximities.numel() > 0:
            cp = candidate_proximities.detach()
            br._exq_candidate_prox_max_log.append(float(cp.max().item()))
            br._exq_candidate_prox_mean_log.append(float(cp.mean().item()))
        else:
            br._exq_candidate_prox_max_log.append(0.0)
            br._exq_candidate_prox_mean_log.append(0.0)
        br._exq_cue_drive_log.append(float(drive_level))
        return bias

    br.compute_anticipatory_liking_write = write_wrapper
    br.compute_approach_cue_score_bias = cue_wrapper
    br._exq490e_instrumented = True


def _make_agent_and_env(seed: int, arm: Dict) -> Tuple[REEAgent, CausalGridWorldV2]:
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
        # SD-022 limb damage (matches env so harm_obs_a_dim auto-set to 7)
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        # SD-036 + MECH-279 + SD-037 (held ON in both arms; same as 490b)
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
        # SD-032b dACC + MECH-258 E2_harm_a (held ON in both arms)
        use_dacc=True,
        use_e2_harm_a=True,
        # Full V_s invalidation circuit (held ON in both arms)
        use_per_stream_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        use_per_region_vs=True,
        use_staleness_accumulator=True,
        use_mech284_hysteresis=True,
        use_vs_commit_release=True,
        # MECH-269b: HELD ON in BOTH arms (490b/490c gate-fire baseline).
        use_vs_rollout_gating=True,
        # 490b smoke threshold override (rationale carried verbatim).
        # At default thresholds the gate never fires under typical Phase 1
        # V_s dynamics; the override exercises the gate-firing precondition
        # so the surrounding stack is observable. Q-040a is already
        # answered PASS by 490b at this override -- we re-confirm here
        # only via C1, NOT a new claim.
        vs_gate_snapshot_refresh_threshold=0.95,
        vs_gate_e1_threshold=0.85,
        vs_gate_e2_threshold=0.85,
        # MECH-295: bridge ON in both arms; 490e varies only upstream
        # activation floors and drive-to-liking gain.
        use_mech295_liking_bridge=True,
        mech295_drive_to_liking_gain=arm["mech295_drive_to_liking_gain"],
        mech295_liking_to_approach_cue_gain=arm[
            "mech295_liking_to_approach_cue_gain"
        ],
        mech295_min_drive_to_fire=arm["mech295_min_drive_to_fire"],
        mech295_min_z_goal_norm_to_fire=arm[
            "mech295_min_z_goal_norm_to_fire"
        ],
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5

    agent = REEAgent(config)
    _instrument_mech295_bridge(agent)
    return agent, env


def _classify_mode(z_harm_norm: float, world_change_norm: float, harm_signal: float) -> str:
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


def _warmup_train(agent, env, num_episodes, steps_per_episode, label) -> Dict:
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
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    reward_log: List[float] = []

    agent.train()

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev = None
        z_self_prev = None
        action_prev = None
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
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            ticks = agent.clock.advance()
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
                hp = agent.e3.harm_eval(zw_b)
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
        if (ep + 1) % 20 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train] {label} ep {ep+1}/{num_episodes} rv={rv:.4f} ep_reward={ep_reward:.4f}",
                flush=True,
            )

    return {
        "final_running_variance": agent.e3._running_variance,
        "warmup_first10_reward":  float(np.mean(reward_log[:10])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
        "warmup_last10_reward":   float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
    }


def _eval_agent(agent, env, num_episodes, steps_per_episode) -> Dict:
    action_dim = env.action_dim
    device = agent.device

    approach_commit_count = 0
    freeze_commit_count = 0
    freeze_active_steps = 0
    pag_release_count = 0
    n_ticks = 0
    override_signal_log: List[float] = []
    dacc_score_bias_norms: List[float] = []
    n_dacc_bias_nonzero = 0
    bridge_n_write_fires_total = 0
    bridge_n_cue_fires_total = 0
    base_drive_log: List[float] = []
    goal_norm_log: List[float] = []
    goal_active_log: List[bool] = []

    agent.eval()
    _clear_bridge_eval_diagnostics(agent)

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_self_prev = None
        z_world_prev = None
        action_prev = None
        prev_freeze_active = False

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
                    agent.record_transition(
                        z_self_prev, action_prev, latent.z_self.detach()
                    )
                ticks = agent.clock.advance()
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
                base_drive_log.append(float(drive_level))
                if agent.goal_state is not None:
                    goal_norm = float(agent.goal_state.goal_norm())
                    goal_active = bool(agent.goal_state.is_active())
                else:
                    goal_norm = 0.0
                    goal_active = False
                goal_norm_log.append(goal_norm)
                goal_active_log.append(goal_active)

                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, action_dim - 1), action_dim, device
                    )
                    agent._last_action = action

            # PAG freeze gate diagnostics
            if agent.pag_freeze_gate is not None:
                last = agent.pag_freeze_gate.last_output
                if last is not None:
                    if getattr(last, "freeze_commit", False):
                        freeze_commit_count += 1
                    if getattr(last, "freeze_active", False):
                        freeze_active_steps += 1
                        if not prev_freeze_active:
                            pass
                        prev_freeze_active = True
                    else:
                        if prev_freeze_active:
                            pag_release_count += 1
                        prev_freeze_active = False

            if agent.broadcast_override is not None:
                override_signal_log.append(float(agent.broadcast_override.override_signal))

            # dACC score-bias magnitude (C3 metric).
            if agent.dacc is not None:
                bundle = getattr(agent.dacc, "_last_bundle", None)
                if bundle is not None:
                    sb = bundle.get("mode_ev")
                    if sb is None:
                        sb = bundle.get("harm_interaction")
                    if sb is not None:
                        try:
                            norm = float(torch.as_tensor(sb).norm().item())
                        except Exception:
                            norm = 0.0
                        dacc_score_bias_norms.append(norm)
                        if norm > 1e-6:
                            n_dacc_bias_nonzero += 1

            # MECH-295 bridge diagnostics (activation and cue-bias checks).
            if getattr(agent, "mech295_bridge", None) is not None:
                br = agent.mech295_bridge
                bridge_n_write_fires_total = int(getattr(br, "_n_write_fires", 0))
                bridge_n_cue_fires_total   = int(getattr(br, "_n_cue_fires", 0))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            n_ticks += 1

            z_harm_norm = (
                float(latent.z_harm.norm().item())
                if latent.z_harm is not None else 0.0
            )
            world_change_norm = (
                float((latent.z_world - z_world_prev).norm().item())
                if z_world_prev is not None else 0.0
            )
            mode = _classify_mode(z_harm_norm, world_change_norm, float(harm_signal))
            if mode == "approach":
                approach_commit_count += 1

            z_world_prev = latent.z_world.detach()
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

    mean_dacc_bias = (
        float(np.mean(dacc_score_bias_norms)) if dacc_score_bias_norms else 0.0
    )
    br = getattr(agent, "mech295_bridge", None)
    br_cfg = getattr(br, "config", None)
    min_drive = float(getattr(br_cfg, "min_drive_to_fire", 0.0))
    min_goal = float(getattr(br_cfg, "min_z_goal_norm_to_fire", 0.0))
    write_drive_log = list(getattr(br, "_exq_write_drive_log", [])) if br is not None else []
    write_goal_norm_log = list(getattr(br, "_exq_write_z_goal_norm_log", [])) if br is not None else []
    write_value_log = list(getattr(br, "_exq_write_value_log", [])) if br is not None else []
    cue_drive_log = list(getattr(br, "_exq_cue_drive_log", [])) if br is not None else []
    cue_bias_max_log = list(getattr(br, "_exq_cue_bias_max_abs_log", [])) if br is not None else []
    cue_bias_mean_log = list(getattr(br, "_exq_cue_bias_mean_abs_log", [])) if br is not None else []
    cand_prox_max_log = list(getattr(br, "_exq_candidate_prox_max_log", [])) if br is not None else []
    cand_prox_mean_log = list(getattr(br, "_exq_candidate_prox_mean_log", [])) if br is not None else []

    out = {
        "approach_commit_count": int(approach_commit_count),
        "freeze_commit_count":   int(freeze_commit_count),
        "freeze_active_steps":   int(freeze_active_steps),
        "pag_release_count":     int(pag_release_count),
        "n_ticks":               int(n_ticks),
        "override_mean":         float(np.mean(override_signal_log)) if override_signal_log else 0.0,
        "override_max":          float(np.max(override_signal_log)) if override_signal_log else 0.0,
        "dacc_score_bias_mean":  mean_dacc_bias,
        "dacc_score_bias_nonzero_ticks": int(n_dacc_bias_nonzero),
        "bridge_n_write_fires_total": int(bridge_n_write_fires_total),
        "bridge_n_cue_fires_total":   int(bridge_n_cue_fires_total),
        "bridge_cue_bias_mean_max_abs": float(np.mean(cue_bias_max_log)) if cue_bias_max_log else 0.0,
        "bridge_cue_bias_peak_max_abs": float(np.max(cue_bias_max_log)) if cue_bias_max_log else 0.0,
        "bridge_cue_bias_mean_abs": float(np.mean(cue_bias_mean_log)) if cue_bias_mean_log else 0.0,
        "bridge_write_value_mean": float(np.mean(write_value_log)) if write_value_log else 0.0,
        "bridge_write_value_max": float(np.max(write_value_log)) if write_value_log else 0.0,
        "bridge_write_drive_mean": float(np.mean(write_drive_log)) if write_drive_log else 0.0,
        "bridge_write_z_goal_norm_mean": float(np.mean(write_goal_norm_log)) if write_goal_norm_log else 0.0,
        "bridge_write_drive_ge_min_ticks": int(sum(v >= min_drive for v in write_drive_log)),
        "bridge_write_z_goal_norm_ge_min_ticks": int(sum(v >= min_goal for v in write_goal_norm_log)),
        "bridge_cue_drive_mean": float(np.mean(cue_drive_log)) if cue_drive_log else 0.0,
        "bridge_cue_drive_ge_min_ticks": int(sum(v >= min_drive for v in cue_drive_log)),
        "bridge_candidate_prox_max_mean": float(np.mean(cand_prox_max_log)) if cand_prox_max_log else 0.0,
        "bridge_candidate_prox_mean_mean": float(np.mean(cand_prox_mean_log)) if cand_prox_mean_log else 0.0,
        "base_drive_mean": float(np.mean(base_drive_log)) if base_drive_log else 0.0,
        "base_drive_max": float(np.max(base_drive_log)) if base_drive_log else 0.0,
        "goal_norm_mean": float(np.mean(goal_norm_log)) if goal_norm_log else 0.0,
        "goal_norm_max": float(np.max(goal_norm_log)) if goal_norm_log else 0.0,
        "goal_active_ticks": int(sum(1 for v in goal_active_log if v)),
    }

    if agent.vs_rollout_gate is not None:
        out.update(agent.vs_rollout_gate.get_diagnostics())
    else:
        out["vs_gate_total_held_e1"] = 0
        out["vs_gate_total_held_e2"] = 0
        out["vs_gate_n_snapshots"] = 0

    return out


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- MECH-295 seeding-strengthening probe", flush=True)
    print(f"Arms: {[a['id'] for a in ARMS]}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Warmup x Eval x steps_per_ep: {WARMUP_EPISODES} x {EVAL_EPISODES} x {STEPS_PER_EPISODE}", flush=True)
    print("Both arms hold use_vs_rollout_gating=True with smoke threshold override (0.85/0.85/0.95);",
          "use_broadcast_override=True, use_dacc=True, drive_weight=2.0, full V_s circuit ON;",
          "use_mech295_liking_bridge=True in both arms.",
          flush=True)
    print("Acceptance: C1 (gate fires both arms) AND C2 (RELAXED cue fires >=10x BASELINE,",
          "or >0 when BASELINE is 0) AND C3 (RELAXED cue-bias mean-max > BASELINE)",
          "AND C4 (RELAXED approach_commit >0 in >=2/3 seeds)",
          flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} MECH-295 seeding-strengthening probe"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit 0; do not execute.")
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for seed in SEEDS:
        for arm in ARMS:
            print(f"Seed {seed} Arm {arm['id']}", flush=True)
            agent, env = _make_agent_and_env(seed, arm)
            label = f"seed{seed}_{arm['id']}"
            warmup_stats = _warmup_train(
                agent, env,
                num_episodes=WARMUP_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                label=label,
            )
            eval_stats = _eval_agent(
                agent, env,
                num_episodes=EVAL_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
            )
            row = {
                "seed": seed,
                "arm":  arm["id"],
                "use_mech295_liking_bridge": True,
                "mech295_min_drive_to_fire": arm["mech295_min_drive_to_fire"],
                "mech295_min_z_goal_norm_to_fire": arm[
                    "mech295_min_z_goal_norm_to_fire"
                ],
                "mech295_drive_to_liking_gain": arm[
                    "mech295_drive_to_liking_gain"
                ],
                "mech295_liking_to_approach_cue_gain": arm[
                    "mech295_liking_to_approach_cue_gain"
                ],
                **warmup_stats,
                **eval_stats,
            }
            print(
                f"  -> approach={row['approach_commit_count']}"
                f"  freeze={row['freeze_commit_count']}"
                f"  pag_release={row['pag_release_count']}"
                f"  override_mean={row['override_mean']:.3f}"
                f"  dacc_bias_mean={row['dacc_score_bias_mean']:.4f}"
                f"  bridge_writes={row['bridge_n_write_fires_total']}"
                f"  bridge_cues={row['bridge_n_cue_fires_total']}"
                f"  cue_bias_mean_max={row['bridge_cue_bias_mean_max_abs']:.4f}"
                f"  goal_norm_max={row['goal_norm_max']:.4f}"
                f"  vs_held_e1={row.get('vs_gate_total_held_e1', 0)}"
                f"  vs_held_e2={row.get('vs_gate_total_held_e2', 0)}",
                flush=True,
            )
            verdict = "PASS" if row["approach_commit_count"] > 0 else "FAIL"
            print(f"verdict: {verdict}", flush=True)
            all_results.append(row)

    relaxed = [r for r in all_results if r["arm"] == "RELAXED"]
    baseline = [r for r in all_results if r["arm"] == "BASELINE"]

    relaxed_total_held = sum(
        r.get("vs_gate_total_held_e1", 0) + r.get("vs_gate_total_held_e2", 0)
        for r in relaxed
    )
    baseline_total_held = sum(
        r.get("vs_gate_total_held_e1", 0) + r.get("vs_gate_total_held_e2", 0)
        for r in baseline
    )

    relaxed_seeds_passing = sum(1 for r in relaxed if r["approach_commit_count"] > 0)
    relaxed_approach_mean = float(np.mean([r["approach_commit_count"] for r in relaxed])) if relaxed else 0.0
    baseline_approach_mean = float(np.mean([r["approach_commit_count"] for r in baseline])) if baseline else 0.0

    relaxed_dacc_mean = float(np.mean([r["dacc_score_bias_mean"] for r in relaxed])) if relaxed else 0.0
    baseline_dacc_mean = float(np.mean([r["dacc_score_bias_mean"] for r in baseline])) if baseline else 0.0
    relaxed_cue_bias_mean = float(np.mean([r["bridge_cue_bias_mean_max_abs"] for r in relaxed])) if relaxed else 0.0
    baseline_cue_bias_mean = float(np.mean([r["bridge_cue_bias_mean_max_abs"] for r in baseline])) if baseline else 0.0

    relaxed_cue_total = sum(r["bridge_n_cue_fires_total"] for r in relaxed)
    baseline_cue_total = sum(r["bridge_n_cue_fires_total"] for r in baseline)

    c1_pass = relaxed_total_held > 0 and baseline_total_held > 0
    if baseline_cue_total == 0:
        c2_pass = relaxed_cue_total > 0
    else:
        c2_pass = relaxed_cue_total >= (10 * baseline_cue_total)
    c3_pass = relaxed_cue_bias_mean > baseline_cue_bias_mean
    c4_pass = relaxed_seeds_passing >= 2

    outcome = "PASS" if (c1_pass and c2_pass and c3_pass and c4_pass) else "FAIL"

    summary = {
        "c1_pass": bool(c1_pass),
        "c1_relaxed_total_held": int(relaxed_total_held),
        "c1_baseline_total_held": int(baseline_total_held),
        "c2_pass": bool(c2_pass),
        "c2_relaxed_cue_fires_total": int(relaxed_cue_total),
        "c2_baseline_cue_fires_total": int(baseline_cue_total),
        "c2_rule": "relaxed > 0 when baseline == 0, else relaxed >= 10x baseline",
        "c3_pass": bool(c3_pass),
        "c3_relaxed_cue_bias_mean_max_abs": relaxed_cue_bias_mean,
        "c3_baseline_cue_bias_mean_max_abs": baseline_cue_bias_mean,
        "c4_pass": bool(c4_pass),
        "c4_relaxed_approach_seeds_passing": int(relaxed_seeds_passing),
        "c4_relaxed_approach_mean": relaxed_approach_mean,
        "c4_baseline_approach_mean": baseline_approach_mean,
        "c4_seeds_required": 2,
        "diagnostic_relaxed_dacc_mean": relaxed_dacc_mean,
        "diagnostic_baseline_dacc_mean": baseline_dacc_mean,
        "n_seeds": len(SEEDS),
    }

    per_claim = {
        cid: ("supports" if outcome == "PASS" else "weakens")
        for cid in CLAIM_IDS
    }

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"  C1 gate fires both arms:      {c1_pass}  (RELAXED held={relaxed_total_held}, BASELINE held={baseline_total_held})", flush=True)
    print(f"  C2 cue-fire lift:             {c2_pass}  (RELAXED cues={relaxed_cue_total}, BASELINE cues={baseline_cue_total})", flush=True)
    print(f"  C3 cue-bias lift:             {c3_pass}  (RELAXED mean-max={relaxed_cue_bias_mean:.4f}, BASELINE mean-max={baseline_cue_bias_mean:.4f})", flush=True)
    print(f"  C4 relaxed approach:          {c4_pass}  (RELAXED seeds>0: {relaxed_seeds_passing}/{len(SEEDS)}, mean={relaxed_approach_mean:.3f})", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": (
            "supports" if outcome == "PASS" else "weakens"
        ),
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_note": (
            "V3-EXQ-490e is the prewritten 490c successor for the bridge-"
            "silent branch. V3-EXQ-490c showed that the V_s gate fired but "
            "MECH-295 cue fires, dACC score bias, and approach commits stayed "
            "at zero. 490e keeps the full V_s stack and MECH-295 bridge ON in "
            "both arms, then compares BASELINE default activation floors "
            "(min_drive_to_fire=0.1, min_z_goal_norm_to_fire=0.05, "
            "drive_to_liking_gain=1.0) against RELAXED floors "
            "(0.01, 0.005, gain=2.0). PASS = C1 gate fires in both arms, "
            "C2 RELAXED cue fires lift by >=10x over BASELINE (or >0 when "
            "BASELINE is zero), C3 RELAXED cue-bias mean-max exceeds "
            "BASELINE, and C4 RELAXED approach_commit >0 in >=2/3 seeds. "
            "PASS supports the Q-040b route that the 490c inert phenotype was "
            "an upstream activation-floor problem. FAIL with C1 PASS and C2 "
            "FAIL routes to 490g instrumentation; FAIL with C2 PASS and C4 "
            "FAIL routes to 490d cue-gain sweep. Tagged Q-040 only because "
            "this is a config probe, not a standalone MECH-295 weak-reading "
            "test."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "arms": ARMS,
            "seeds": SEEDS,
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
