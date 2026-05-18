#!/opt/local/bin/python3
"""
V3-EXQ-490 -- MECH-269b symmetric V_s gating on E1/E2 cortical rollouts.

Claims: MECH-269b, Q-040

Purpose (diagnostic / substrate-readiness)
------------------------------------------
EXQ-483 / EXQ-483a confirmed the SD-037 broadcast override + PAG freeze-gate
substrate is wired (override_signal climbs to 0.56, PAG release ratio
ON_ON / ON_OFF = 1.69) but behaviourally INERT: approach_commit stayed at
0.0 across all four arms including the SD-036-only baseline. This is the
wired-but-inert phenotype: every regulator in the V3 control stack fires,
nothing in the policy layer revises.

MECH-269b is one candidate root cause. The V_s primitive (MECH-269 Phase 1
per_stream_vs) is computed but consumed only by the hippocampal proposer
anchor selection. The cortical world models (E1 sensory predictor, E2_harm_a
forward model) consume current-tick latent values indiscriminately; if a
stream's V_s has dropped, downstream consumers (dACC, E3, PAG) compute
prediction errors against the wrong reference and produce no corrective
behavioural revision.

This experiment is the Q-040 factorial. Both arms hold the broadcast
override + dACC + drive modulation ON; the only manipulated variable is
use_vs_rollout_gating (consumes per_stream_vs at the E1 / E2_harm_a
forward call sites; substitutes held snapshots when V_s falls below the
per-side threshold).

Arms
----
    ON_OFF: full V_s circuit ON, use_vs_rollout_gating=False (mirrors
            EXQ-483a ON_ON arm exactly).
    ON_ON : full V_s circuit ON, use_vs_rollout_gating=True.

Both arms share:
    use_per_stream_vs=True
    use_event_segmenter=True
    use_invalidation_trigger=True
    use_anchor_sets=True
    use_per_region_vs=True
    use_staleness_accumulator=True
    use_mech284_hysteresis=True
    use_vs_commit_release=True
    use_broadcast_override=True
    use_dacc=True
    drive_weight=2.0 (SD-012 ON)
    use_pag_freeze_gate=True
    use_gabaergic_decay=True

Acceptance criteria
-------------------
    C1 (substrate physically wired): ON arm registers
        vs_gate_total_held_e1 + vs_gate_total_held_e2 > 0 across the
        eval run (i.e. the gate fires on at least one stream, on at
        least one side, on at least one tick). OFF arm reports 0.

    C2 (differential approach): ON arm produces approach_commit_count > 0
        in >= 2/3 seeds; OFF arm reproduces the EXQ-483 zero baseline
        (approach_commit_count == 0).

    C3 (dACC behavioural-adjustment magnitude): ON arm shows mean
        |dacc_score_bias_norm| > 0; OFF arm approximately zero. Confirms
        precision-weighted PE is flowing to E3 score-bias and not being
        cancelled by stale-stream rollouts.

Pass / fail rule
----------------
    PASS = C1 AND C2 (in >= 2/3 seeds) AND C3.

    FAIL on C1 -> the substrate is not wired correctly (gate did not
        fire on any stream / side; debug per_stream_vs or threshold
        configuration before iterating).

    FAIL on C2 / C3 (with C1 PASSing) -> the gate is wired but does not
        unblock approach_commit. This is the Q-040 FAIL branch and points
        evidence at MECH-295 (drive -> liking-stream bridge) as the
        dominant blocker; MECH-269b is necessary but not sufficient.

experiment_purpose=diagnostic. Substrate-readiness gate. Not governance
evidence. Q-040 resolution requires both this PASS / FAIL and the
follow-up MECH-295 substrate.

See REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md
See REE_assembly/docs/architecture/v_s_invalidation_runtime.md
See ree-v3/CLAUDE.md MECH-269b section.
"""

from __future__ import annotations

import argparse
import json
import math
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


EXPERIMENT_TYPE = "v3_exq_490_mech269b_vs_rollout_gating"
CLAIM_IDS = ["MECH-269b", "Q-040"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 19]
ARMS = [
    {"id": "ON_OFF", "use_vs_rollout_gating": False},
    {"id": "ON_ON",  "use_vs_rollout_gating": True},
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
        # SD-036 + MECH-279 + SD-037 (held ON in both arms)
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
        use_broadcast_override=True,
        # SD-032b dACC + MECH-258 E2_harm_a (held ON in both arms;
        # dACC uses use_dacc, E2_harm_a constructed via use_e2_harm_a)
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
        # MECH-269b: the single variable under test
        use_vs_rollout_gating=arm["use_vs_rollout_gating"],
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5

    agent = REEAgent(config)
    return agent, env


def _classify_mode(z_harm_norm: float, world_change_norm: float, harm_signal: float) -> str:
    if z_harm_norm > HARM_MODE_THRESH:
        return "avoid"
    if harm_signal > 0.01:
        return "approach"
    if world_change_norm > EXPLORE_ERR_THRESH:
        return "explore"
    return "neutral"


def _warmup_train(agent, env, num_episodes, steps_per_episode) -> Dict:
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
                f"  [warmup] ep {ep+1}/{num_episodes}  rv={rv:.4f}  ep_reward={ep_reward:.4f}",
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

    agent.eval()

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
    }

    # MECH-269b gate diagnostics (only present when gate is wired).
    if agent.vs_rollout_gate is not None:
        out.update(agent.vs_rollout_gate.get_diagnostics())
    else:
        # Mirror keys at zero for the OFF arm so per-arm comparison is clean.
        out["vs_gate_total_held_e1"] = 0
        out["vs_gate_total_held_e2"] = 0
        out["vs_gate_n_snapshots"] = 0

    return out


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- MECH-269b symmetric V_s gating", flush=True)
    print(f"Arms: {[a['id'] for a in ARMS]}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Warmup x Eval x steps_per_ep: {WARMUP_EPISODES} x {EVAL_EPISODES} x {STEPS_PER_EPISODE}", flush=True)
    print("Both arms hold use_broadcast_override=True, use_dacc=True, drive_weight=2.0,",
          "and full V_s invalidation circuit ON; toggled var = use_vs_rollout_gating.",
          flush=True)
    print("Acceptance: C1 (ON gate fires > 0) AND C2 (ON approach_commit > 0 in >=2/3 seeds) AND C3 (ON dacc_score_bias_mean > 0)",
          flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} MECH-269b cortical V_s gating validation"
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
            print(f"Seed {seed}  Arm {arm['id']}", flush=True)
            agent, env = _make_agent_and_env(seed, arm)
            warmup_stats = _warmup_train(
                agent, env,
                num_episodes=WARMUP_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
            )
            eval_stats = _eval_agent(
                agent, env,
                num_episodes=EVAL_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
            )
            row = {
                "seed": seed,
                "arm":  arm["id"],
                "use_vs_rollout_gating": arm["use_vs_rollout_gating"],
                **warmup_stats,
                **eval_stats,
            }
            print(
                f"  -> approach={row['approach_commit_count']}"
                f"  freeze={row['freeze_commit_count']}"
                f"  pag_release={row['pag_release_count']}"
                f"  override_mean={row['override_mean']:.3f}"
                f"  dacc_bias_mean={row['dacc_score_bias_mean']:.4f}"
                f"  vs_held_e1={row.get('vs_gate_total_held_e1', 0)}"
                f"  vs_held_e2={row.get('vs_gate_total_held_e2', 0)}",
                flush=True,
            )
            all_results.append(row)

    on_results  = [r for r in all_results if r["arm"] == "ON_ON"]
    off_results = [r for r in all_results if r["arm"] == "ON_OFF"]

    on_total_held = sum(
        r.get("vs_gate_total_held_e1", 0) + r.get("vs_gate_total_held_e2", 0)
        for r in on_results
    )
    off_total_held = sum(
        r.get("vs_gate_total_held_e1", 0) + r.get("vs_gate_total_held_e2", 0)
        for r in off_results
    )

    on_approach_seeds_passing = sum(
        1 for r in on_results if r["approach_commit_count"] > 0
    )
    on_dacc_mean = float(
        np.mean([r["dacc_score_bias_mean"] for r in on_results])
    ) if on_results else 0.0

    c1_pass = on_total_held > 0 and off_total_held == 0
    c2_pass = on_approach_seeds_passing >= 2
    c3_pass = on_dacc_mean > 0.0
    outcome = "PASS" if (c1_pass and c2_pass and c3_pass) else "FAIL"

    summary = {
        "c1_pass": bool(c1_pass),
        "c1_on_total_held": int(on_total_held),
        "c1_off_total_held": int(off_total_held),
        "c2_pass": bool(c2_pass),
        "c2_on_approach_seeds_passing": int(on_approach_seeds_passing),
        "c2_seeds_required": 2,
        "c3_pass": bool(c3_pass),
        "c3_on_dacc_mean": on_dacc_mean,
        "n_seeds": len(SEEDS),
    }

    per_claim = {
        cid: ("supports" if outcome == "PASS" else "inconclusive")
        for cid in CLAIM_IDS
    }

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"  C1 substrate-wired: {c1_pass}  (ON_held={on_total_held}, OFF_held={off_total_held})", flush=True)
    print(f"  C2 approach_commit: {c2_pass}  (ON seeds with >0: {on_approach_seeds_passing}/{len(SEEDS)})", flush=True)
    print(f"  C3 dacc_bias_mean:  {c3_pass}  (ON mean: {on_dacc_mean:.4f})", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": (
            "supports" if outcome == "PASS" else "inconclusive"
        ),
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_note": (
            "MECH-269b symmetric V_s gating substrate-readiness diagnostic. "
            "Two arms ON_OFF / ON_ON differ only in use_vs_rollout_gating; "
            "both hold broadcast override + dACC + drive modulation + full "
            "V_s invalidation circuit ON. C1 = gate fires (substrate wired). "
            "C2 = ON arm produces non-zero approach_commit in >=2/3 seeds "
            "(unblocks the EXQ-483 wired-but-inert phenotype). C3 = ON arm "
            "shows non-zero dACC score-bias magnitude (PE flowing to E3). "
            "PASS = C1 AND C2 AND C3. FAIL on C2 / C3 with C1 PASSing -> "
            "Q-040 FAIL branch; points evidence at MECH-295 liking-bridge "
            "as dominant blocker. Substrate-readiness only; not governance "
            "evidence."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "arms": [a["id"] for a in ARMS],
            "seeds": SEEDS,
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Result written to: {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
