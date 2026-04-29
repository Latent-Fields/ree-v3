#!/opt/local/bin/python3
"""
V3-EXQ-490c -- MECH-269b VsRolloutGate + MECH-295 liking-bridge factorial (Q-040b).

Successor (NOT supersede) of V3-EXQ-490b. The 490b run delivered the Q-040
narrowing: C1 PASSed (gate fires under threshold-override smoke) but C2 + C3
FAILed across all seeds (zero approach_commit, zero dACC score-bias). Per the
governance walk 2026-04-28T22Z, that points evidence at MECH-295 liking-bridge
as the dominant remaining blocker. With the bridge substrate now landed
(V3-EXQ-493 UC1-UC6 PASS 2026-04-26), 490c tests whether wiring the bridge ON
top of the V_s gate ON arm unblocks behavioural approach_commit + dACC bias.

Claims (claim_ids): ["Q-040"]. We tag only Q-040 (per claim_ids accuracy rule
"err toward fewer tags"). MECH-269b is held ON in both arms, so the
experiment does not discriminate it. MECH-295 weak reading is implicated by
the bridge OFF/ON manipulation, but the accompanying V_s threshold override
(see 490b root-cause note) means a clean MECH-295 standalone read is also
not isolated. PASS supports the architectural reading underpinning Q-040b
(jointly: V_s gating + liking bridge unblock the EXQ-483 wired-but-inert
phenotype). FAIL routes evidence further upstream (additional substrate
gaps; see resolution branches below).

Purpose
-------
The 490b factorial held everything else ON and toggled use_vs_rollout_gating.
Result (2026-04-28): C1 PASS, C2 FAIL (approach_commit==0 in both arms),
C3 FAIL (dACC bias mean ~0). Q-040 narrowed:
    Q-040a (precondition): substrate-readiness, gate fires. ANSWERED PASS
        at threshold-overridden smoke (the 490b ON_ON arm hit C1).
    Q-040b (behavioural sufficiency): does MECH-269b produce non-zero
        approach_commit + dACC bias?
        Standing inference 2026-04-28: NOT alone. Bridge needed.

This experiment runs the natural successor: V_s gating ON in both arms;
toggle use_mech295_liking_bridge. Each arm holds the same V_s threshold
override carried from 490b so the gate-fire precondition (Q-040a) stays
PASS-able without re-running it.

Arms
----
    ON_OFF: full V_s circuit ON, use_vs_rollout_gating=True,
            use_mech295_liking_bridge=False (replicates 490b ON_ON arm
            exactly).
    ON_ON : full V_s circuit ON, use_vs_rollout_gating=True,
            use_mech295_liking_bridge=True. Bridge sub-knobs at default
            from REEConfig.from_dims (drive_to_liking_gain=1.0,
            liking_to_approach_cue_gain=0.5, min_drive_to_fire=0.1,
            min_z_goal_norm_to_fire=0.05).

Both arms share the 490b stack (held identical across arms):
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
    vs_gate_e1_threshold=0.85, vs_gate_e2_threshold=0.85,
    vs_gate_snapshot_refresh_threshold=0.95
    (smoke override carried from 490b root-cause analysis)

Acceptance criteria
-------------------
    C1 (substrate physically wired in both arms): both arms register
        vs_gate_total_held_e1 + vs_gate_total_held_e2 > 0 across the eval
        run. This is the same gate-fire precondition 490b answered; we
        re-verify it stays PASS-able when bridge is wired in.

    C2 (differential approach in favour of bridge ON):
        ON_ON arm produces approach_commit_count > 0 in >= 2/3 seeds AND
        ON_ON.mean(approach_commit) > ON_OFF.mean(approach_commit).
        FAIL on this with C1 PASSing means the bridge alone is also not
        sufficient; resolves Q-040b further upstream (residue/valence
        write geometry, GoalState seeding, downstream consumer wiring).

    C3 (differential dACC bias magnitude):
        ON_ON.mean(dacc_score_bias_norm) > ON_OFF.mean(dacc_score_bias_norm)
        AND ON_ON.mean(dacc_score_bias_norm) > 0. Confirms the bridge's
        per-candidate cue-side score_bias is composing additively with
        the dACC-adapter score_bias as designed.

    C4 (severed-bridge collapse, falsifiable signature):
        bridge_n_cue_fires_total in ON_OFF arm == 0 AND
        bridge_n_cue_fires_total in ON_ON arm > 0. UC5-style severance
        check at the behavioural level: ON_OFF (bridge off) must produce
        zero cue fires regardless of drive/goal_norm; ON_ON must produce
        cue fires when both upstream (drive>=0.1) and downstream
        (goal_norm>=0.05) gates are satisfied.

Pass / fail rule
----------------
    PASS = C1 AND C2 AND C3 AND C4.

    FAIL on C2/C3 with C1 PASSing AND C4 PASSing -> bridge wired but
        does not behaviourally unblock approach_commit. Routes evidence
        to: (a) ResidueField VALENCE_LIKING write geometry (writes go
        to z_goal latent but proximities computed against current
        z_world; spatial dissociation may degrade cue magnitude),
        (b) GoalState.is_active() / goal_norm dynamics (z_goal not
        being seeded sufficiently from z_resource), (c) downstream
        BG/E3 consumer wiring of dacc_score_bias (the additive
        composition might be too weak vs other E3 cost terms).
    FAIL on C4 with C2/C3 either way -> bridge is not severable as
        designed (use_mech295_liking_bridge=False arm still produced
        cue fires); points at a wiring bug in MECH-295 master switch
        or default config sub-knobs. Action: rebuild & re-run 493
        UC5 contract.

experiment_purpose=evidence. The bridge OFF/ON toggle is the directly
falsifiable test of MECH-295 weak reading severability AT THE
BEHAVIOURAL LEVEL (UC5 in V3-EXQ-493 was the substrate-side signature).
PASS supports Q-040b architectural reading; FAIL has interpretable
upstream-routing implications.

See REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md
See REE_assembly/docs/architecture/mech_295_drive_liking_approach_bridge.md
See REE_assembly/docs/architecture/v_s_invalidation_runtime.md
See ree-v3/CLAUDE.md MECH-269b + MECH-295 sections.
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


EXPERIMENT_TYPE = "v3_exq_490c_mech269b_with_liking_bridge"
CLAIM_IDS = ["Q-040"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 19]
ARMS = [
    {"id": "ON_OFF", "use_mech295_liking_bridge": False},
    {"id": "ON_ON",  "use_mech295_liking_bridge": True},
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
        # MECH-269b: HELD ON in BOTH arms (490b ON_ON arm baseline).
        # The toggled variable in 490c is the bridge.
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
        # MECH-295: the toggled variable.
        use_mech295_liking_bridge=arm["use_mech295_liking_bridge"],
        # Bridge sub-knobs at default values
        # (mech295_drive_to_liking_gain=1.0,
        #  mech295_liking_to_approach_cue_gain=0.5,
        #  mech295_min_drive_to_fire=0.1,
        #  mech295_min_z_goal_norm_to_fire=0.05).
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
            try:
                agent.update_z_goal(
                    latent,
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )
            except Exception:
                pass

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
    bridge_last_cue_bias_max_abs_log: List[float] = []

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
                try:
                    agent.update_z_goal(
                        latent,
                        benefit_exposure=benefit_exposure,
                        drive_level=drive_level,
                    )
                except Exception:
                    pass

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

            # MECH-295 bridge diagnostics (C4 falsifiable severance check).
            if getattr(agent, "mech295_bridge", None) is not None:
                br = agent.mech295_bridge
                bridge_n_write_fires_total = int(getattr(br, "_n_write_fires", 0))
                bridge_n_cue_fires_total   = int(getattr(br, "_n_cue_fires", 0))
                last_bias = getattr(br, "_last_cue_bias_max_abs", None)
                if last_bias is not None:
                    try:
                        bridge_last_cue_bias_max_abs_log.append(float(last_bias))
                    except Exception:
                        pass

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
        "bridge_n_write_fires_total": int(bridge_n_write_fires_total),
        "bridge_n_cue_fires_total":   int(bridge_n_cue_fires_total),
        "bridge_cue_bias_mean_max_abs": float(np.mean(bridge_last_cue_bias_max_abs_log)) if bridge_last_cue_bias_max_abs_log else 0.0,
    }

    if agent.vs_rollout_gate is not None:
        out.update(agent.vs_rollout_gate.get_diagnostics())
    else:
        out["vs_gate_total_held_e1"] = 0
        out["vs_gate_total_held_e2"] = 0
        out["vs_gate_n_snapshots"] = 0

    return out


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- MECH-269b V_s ON + MECH-295 bridge OFF/ON factorial", flush=True)
    print(f"Arms: {[a['id'] for a in ARMS]}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Warmup x Eval x steps_per_ep: {WARMUP_EPISODES} x {EVAL_EPISODES} x {STEPS_PER_EPISODE}", flush=True)
    print("Both arms hold use_vs_rollout_gating=True with smoke threshold override (0.85/0.85/0.95);",
          "use_broadcast_override=True, use_dacc=True, drive_weight=2.0, full V_s circuit ON;",
          "toggled var = use_mech295_liking_bridge.",
          flush=True)
    print("Acceptance: C1 (gate fires both arms) AND C2 (ON_ON approach > ON_OFF AND ON_ON approach > 0 in >=2/3 seeds)",
          "AND C3 (ON_ON dACC bias > ON_OFF AND ON_ON dACC bias > 0)",
          "AND C4 (bridge severance: cue fires == 0 in ON_OFF AND > 0 in ON_ON)",
          flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} MECH-269b + MECH-295 factorial"
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
                "use_mech295_liking_bridge": arm["use_mech295_liking_bridge"],
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
                f"  vs_held_e1={row.get('vs_gate_total_held_e1', 0)}"
                f"  vs_held_e2={row.get('vs_gate_total_held_e2', 0)}",
                flush=True,
            )
            verdict = "PASS" if row["approach_commit_count"] > 0 else "FAIL"
            print(f"verdict: {verdict}", flush=True)
            all_results.append(row)

    on_on  = [r for r in all_results if r["arm"] == "ON_ON"]
    on_off = [r for r in all_results if r["arm"] == "ON_OFF"]

    on_on_total_held = sum(
        r.get("vs_gate_total_held_e1", 0) + r.get("vs_gate_total_held_e2", 0)
        for r in on_on
    )
    on_off_total_held = sum(
        r.get("vs_gate_total_held_e1", 0) + r.get("vs_gate_total_held_e2", 0)
        for r in on_off
    )

    on_on_seeds_passing = sum(1 for r in on_on if r["approach_commit_count"] > 0)
    on_on_approach_mean  = float(np.mean([r["approach_commit_count"] for r in on_on])) if on_on else 0.0
    on_off_approach_mean = float(np.mean([r["approach_commit_count"] for r in on_off])) if on_off else 0.0

    on_on_dacc_mean  = float(np.mean([r["dacc_score_bias_mean"] for r in on_on])) if on_on else 0.0
    on_off_dacc_mean = float(np.mean([r["dacc_score_bias_mean"] for r in on_off])) if on_off else 0.0

    on_on_cue_total  = sum(r["bridge_n_cue_fires_total"] for r in on_on)
    on_off_cue_total = sum(r["bridge_n_cue_fires_total"] for r in on_off)

    c1_pass = on_on_total_held > 0 and on_off_total_held > 0
    c2_pass = (
        on_on_seeds_passing >= 2
        and on_on_approach_mean > on_off_approach_mean
    )
    c3_pass = (
        on_on_dacc_mean > on_off_dacc_mean
        and on_on_dacc_mean > 0.0
    )
    c4_pass = on_off_cue_total == 0 and on_on_cue_total > 0

    outcome = "PASS" if (c1_pass and c2_pass and c3_pass and c4_pass) else "FAIL"

    summary = {
        "c1_pass": bool(c1_pass),
        "c1_on_on_total_held": int(on_on_total_held),
        "c1_on_off_total_held": int(on_off_total_held),
        "c2_pass": bool(c2_pass),
        "c2_on_on_approach_seeds_passing": int(on_on_seeds_passing),
        "c2_on_on_approach_mean": on_on_approach_mean,
        "c2_on_off_approach_mean": on_off_approach_mean,
        "c2_seeds_required": 2,
        "c3_pass": bool(c3_pass),
        "c3_on_on_dacc_mean": on_on_dacc_mean,
        "c3_on_off_dacc_mean": on_off_dacc_mean,
        "c4_pass": bool(c4_pass),
        "c4_on_on_cue_fires_total": int(on_on_cue_total),
        "c4_on_off_cue_fires_total": int(on_off_cue_total),
        "n_seeds": len(SEEDS),
    }

    per_claim = {
        cid: ("supports" if outcome == "PASS" else "weakens")
        for cid in CLAIM_IDS
    }

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"  C1 substrate-wired both arms: {c1_pass}  (ON_ON_held={on_on_total_held}, ON_OFF_held={on_off_total_held})", flush=True)
    print(f"  C2 approach delta:            {c2_pass}  (ON_ON mean={on_on_approach_mean:.3f}, ON_OFF mean={on_off_approach_mean:.3f}, ON_ON seeds>0: {on_on_seeds_passing}/{len(SEEDS)})", flush=True)
    print(f"  C3 dACC bias delta:           {c3_pass}  (ON_ON mean={on_on_dacc_mean:.4f}, ON_OFF mean={on_off_dacc_mean:.4f})", flush=True)
    print(f"  C4 bridge severance:          {c4_pass}  (ON_ON cues={on_on_cue_total}, ON_OFF cues={on_off_cue_total})", flush=True)

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
            "MECH-269b VsRolloutGate + MECH-295 liking-bridge factorial. "
            "Successor (NOT supersede) of V3-EXQ-490b. ON_OFF replicates "
            "the 490b ON_ON arm exactly (V_s ON, bridge OFF). ON_ON adds "
            "the bridge with default sub-knobs. Both arms run with the "
            "same V_s threshold override (0.85/0.85/0.95) so the gate-fire "
            "precondition (Q-040a, answered PASS by 490b) is exercised "
            "consistently. PASS = C1 (gate fires both arms) AND C2 (ON_ON "
            "approach_commit > ON_OFF AND > 0 in >=2/3 seeds) AND C3 "
            "(ON_ON dACC bias > ON_OFF AND > 0) AND C4 (severance: cue "
            "fires == 0 in ON_OFF AND > 0 in ON_ON). PASS supports the "
            "Q-040b architectural reading that V_s gating + liking bridge "
            "jointly unblock the EXQ-483 wired-but-inert phenotype. FAIL "
            "with C1 and C4 both PASSing routes evidence further upstream "
            "(VALENCE_LIKING write geometry, GoalState seeding from "
            "z_resource, downstream BG/E3 score_bias composition). FAIL "
            "on C4 alone is a wiring bug indication (run 493 UC5 contract). "
            "MECH-295 weak reading severability is implicated by the C4 "
            "test but not isolated (V_s threshold override leaves "
            "residual confound on MECH-269b discrimination). Tagged "
            "Q-040 only per claim_ids accuracy rule."
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
