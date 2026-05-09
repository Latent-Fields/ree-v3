#!/opt/local/bin/python3
"""V3-EXQ-543: ARC-062 Phase 2a Monomodal-Collapse Falsifier on SD-054 reef.

GAP-B of arc_062_rule_apprehension_plan.md (REE_assembly/evidence/planning/).
Phase 2a single-arm core 2-arm contrast: ARM_0 (use_gated_policy=False, baseline
single-head E3) vs ARM_1c (use_gated_policy=True, gated heads + full 3-stream
context discriminator on (z_world, z_self, z_harm_a) per Pull A R1 verdict).
Same env config, same seeds, same training budget across arms; only manipulated
variable is use_gated_policy.

Phase 2a tests structural sufficiency of the ARC-062 substrate AT INIT (under
symmetry-broken head bias + disc_init_scale=0.1 sigmoid-near-0.5). Bias-head
phased training is deferred to Phase 3 per the plan-of-record (closes
commitment_closure GAP-1 by wiring discriminator -> SD-033a + adding bias-head
parameters to E3 optimiser). The Phase 2a question is: does the architectural
substrate (gated heads + 3-stream discriminator) produce structurally different
agent-environment dynamics than baseline E3 even at frozen-init head bias?

Substrate (matches V3-EXQ-522 ARM_1_med + V3-EXQ-524 baseline):
  SD-007 reafference perspective correction
  SD-008 alpha_world=0.9 encoder correction
  SD-010 use_harm_stream=True sensory-discriminative z_harm_s
  SD-011 use_affective_harm_stream=True + harm_history_len=10 affective z_harm_a
  SD-012 z_goal_enabled=True + drive_weight=2.0 homeostatic drive modulation
  SD-018 use_resource_proximity_head=True resource proximity supervision on z_world
  SD-021 harm_descending_mod_enabled=True descending pain modulation
  SD-054 reef_enabled=True + hazard_food_attraction=0.7 + n_reef_patches=3
  MECH-090 beta_gate_bistable=True bistable commitment latch
  ARC-062 use_gated_policy=True (ARM_1c only); OFF in ARM_0
Density: ARM_1_med = (num_hazards=4, hazard_food_attraction=0.7, size=12);
  matches V3-EXQ-522/524 baseline. Phase 2b density-gradient sub-arms
  (ARM_1_low / ARM_1_med / ARM_1_high) deferred until Phase 2a passes.

Acceptance criteria (per arc_062_rule_apprehension_plan.md Phase 2 deliverable
table, lines 230-243; Pull B R4 multi-signature tolerance window). PASS rule:
>=2 of {C2, C3, C4} hold across seeds with no contradictory signal. C1 is
NOT MEASURABLE in Phase 2a (single-density design); marked non_contributory
and deferred to Phase 2b density-gradient sub-arms.

  C1 density-tracking (NOT MEASURABLE in Phase 2a; non_contributory):
     monotone refuge-use response across hazard density. Deferred to Phase 2b
     (ARM_1_low / ARM_1_med / ARM_1_high density-gradient sub-arms). Reported
     here for downstream-tracker completeness.

  C2 state-dependence (Balaban-Feld 2019): refuge-use should track drive_level
     within episodes -- agents in higher-drive (depleted-energy) states should
     forage more readily; well-fed states should retreat to reef under matched
     hazard pressure. Operationalised as Spearman correlation between binned
     drive_level and binned reef_visit_fraction over eval steps; PASS = |rho|
     >= 0.2 in >=2/3 seeds in ARM_1c AND ARM_1c |rho| > ARM_0 |rho| on average.

  C3 risk-type dissociation (Eccard 2020): distinct response to feeding-risk
     (forage-zone hazard contacts) vs transit-risk (reef-forage transition
     hazard contacts). Operationalised as the per-arm ratio
     forage_hazard_rate / transit_hazard_rate; PASS = ARM_1c ratio differs from
     ARM_0 ratio by >= 50% relative magnitude (|ARM_1c - ARM_0| / max(ARM_0, eps)
     >= 0.5) in mean over seeds.

  C4 cross-seed variation (Eccard 2020 + Crowell 2016): non-zero coefficient of
     variation in reef_visit_fraction across seeds. Operationalised as the
     per-arm CoV (std/mean) of reef_visit_fraction across seeds; PASS = ARM_1c
     CoV >= 0.10 (i.e. observable individual variation, the diversity signature
     ARM_0 monomodal collapse predicts ARM_0 will lack).

Unambiguous FAIL signatures (any one is unambiguous; flips the per-claim
direction to weakens regardless of C2/C3/C4 PASS counts):
  F1 total invariance: ARM_0 and ARM_1c reef_visit_fraction differ by < 0.02
     in mean across seeds AND C2 |rho| < 0.05 in both arms AND C3 ratio
     differs by < 5% AND C4 CoV < 0.02 in both arms. The unambiguous
     monomodal-collapse signature MECH-309 predicts in the absence of a
     rule-creator at the policy layer.
  F2 biologically inverted: ARM_1c refuge-use *increases* monotonically with
     chronic high-drive regimes (Spearman rho > +0.4 with drive_level). Naive
     "always-flee-when-hazard-present" rather than relative-risk-pattern-
     dependent policy; rules out Phase 2a interpretation as substrate-
     sufficient and routes to Phase 3 / ARC-063 V4 strong reading.

Tagging:
  claim_ids: [MECH-309, ARC-062, SD-029]
  evidence_direction_per_claim:
    MECH-309 (primary, logical-necessity falsifier):
      "supports" if PASS (>=2 of C2/C3/C4 + no F1/F2)
      "weakens" if F1 (monomodal collapse confirmed in BOTH arms) or F2
      "non_contributory" if PASS rule unmet but no F1/F2 (mixed signal)
    ARC-062 (V3 weak-reading architectural validation):
      "supports" if PASS
      "weakens" if F1 or F2
      "non_contributory" if PASS rule unmet but no F1/F2
    SD-029 (monomodal-collapse measurement gate):
      "supports" if PASS (substrate enables the C2/C3/C4 measurements that
        SD-029 retests under trained policy needed)
      "weakens" if F1 (substrate ceiling reached -- measurements unobtainable)
      "non_contributory" if PASS rule unmet but no F1
  experiment_purpose: "evidence" (architecturally load-bearing falsifier)
  Phase 3 wiring (discriminator -> SD-033a, bias-head -> E3 optimiser) is
  GATED on this experiment's PASS at score_bias level (option iii in the plan
  R3 verdict). FAIL routes through the falsification chain documented in
  arc_062_rule_apprehension_plan.md Phase 2 deliverable 5.

Plan-of-record: REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md
GAP-B owner_exq -> V3-EXQ-543 (this experiment) on PASS.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_543_arc062_phase2a_monomodal_collapse_falsifier.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "v3_exq_543_arc062_phase2a_monomodal_collapse_falsifier"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-543"
CLAIM_IDS = ["MECH-309", "ARC-062", "SD-029"]

# ARM_1_med density: matches V3-EXQ-522 ARM_1_reef_food + V3-EXQ-524 baseline.
ENV_KWARGS = dict(
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
)

WARMUP_EPISODES = 40
EVAL_EPISODES = 8
STEPS_PER_EPISODE = 200
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4

# Drive-bin breakpoints for C2 state-dependence Spearman (3 bins).
DRIVE_BINS = (0.33, 0.67)

# Coefficients for acceptance thresholds (pre-registered).
C2_RHO_THRESHOLD = 0.20
C2_MIN_PASS_SEEDS = 2
C3_RELATIVE_DELTA_THRESHOLD = 0.50
C4_COV_THRESHOLD = 0.10
F1_REEF_DIFF_THRESHOLD = 0.02
F1_C2_RHO_THRESHOLD = 0.05
F1_C3_DELTA_THRESHOLD = 0.05
F1_C4_COV_THRESHOLD = 0.02
F2_INVERTED_RHO_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict):
    return obs_dict.get("harm_obs")


def _obs_harm_a(obs_dict):
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict):
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation. Returns 0.0 on degenerate (constant) input."""
    if len(x) < 4 or len(y) < 4:
        return 0.0
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _make_agent_and_env(seed: int, use_gated_policy: bool) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Build agent with the manipulated variable use_gated_policy + matched seed."""
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
        # ARC-062 manipulated variable.
        use_gated_policy=use_gated_policy,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5

    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0: Warmup training (identical pipeline across arms)
# ---------------------------------------------------------------------------

def _warmup_train(agent: REEAgent, env: CausalGridWorldV2,
                  num_episodes: int, steps_per_episode: int,
                  total_train_episodes: int, arm_label: str) -> Dict:
    device = agent.device
    action_dim = env.action_dim

    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
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

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)
            prox_t = _obs_resource_prox(obs_dict)
            accum_t = _obs_accum(obs_dict)

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
                    random.randint(0, action_dim - 1), action_dim, device,
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
                idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()), 1.0,
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
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        reward_log.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] {arm_label} ep {ep+1}/{total_train_episodes}"
                f"  rv={agent.e3._running_variance:.4f}"
                f"  ep_reward={ep_reward:.4f}",
                flush=True,
            )

    return {
        "final_running_variance": agent.e3._running_variance,
        "warmup_first10_reward": float(np.mean(reward_log[:10])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
        "warmup_last10_reward": float(np.mean(reward_log[-10:])) if len(reward_log) >= 10 else float(np.mean(reward_log)),
    }


# ---------------------------------------------------------------------------
# Phase 1: Eval with behavioural-metric collection per step
# ---------------------------------------------------------------------------

def _eval_collect_metrics(agent: REEAgent, env: CausalGridWorldV2,
                          num_episodes: int, steps_per_episode: int,
                          total_train_episodes: int,
                          arm_label: str) -> Dict:
    """Eval phase: collect per-step metrics for C2 / C3 / C4 acceptance.

    Per-step records (drive_level, in_reef, transition_event, harm_event,
    in_forage_zone) drive the C2 / C3 / C4 computations downstream.
    """
    device = agent.device
    action_dim = env.action_dim
    agent.eval()

    per_episode_reef_fractions: List[float] = []
    per_episode_drives: List[List[float]] = []
    per_episode_in_reef: List[List[bool]] = []
    forage_hazard_events = 0
    forage_total_steps = 0
    transit_hazard_events = 0
    transit_total_steps = 0

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        reef_cells_set = getattr(env, "_reef_cells", set())
        prev_in_reef = False
        ep_drive_log: List[float] = []
        ep_in_reef_log: List[bool] = []

        for step_idx in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)

            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world,
                    obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
                )
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(
                        z_self_prev, action_prev, latent.z_self.detach(),
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
                        random.randint(0, action_dim - 1), action_dim, device,
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef = agent_pos in reef_cells_set
            transition_event = (in_reef != prev_in_reef)
            harm_event = float(harm_signal) < 0

            ep_drive_log.append(float(drive_level))
            ep_in_reef_log.append(bool(in_reef))

            # Risk-type attribution (C3): feeding-risk vs transit-risk.
            # Forage zone = NOT in reef AND NOT in transition this step.
            # Transit = transition between reef and non-reef.
            if transition_event:
                transit_total_steps += 1
                if harm_event:
                    transit_hazard_events += 1
            elif not in_reef:
                forage_total_steps += 1
                if harm_event:
                    forage_hazard_events += 1

            prev_in_reef = in_reef
            z_self_prev = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev = action.detach()
            if done:
                break

        reef_frac = (
            sum(ep_in_reef_log) / max(len(ep_in_reef_log), 1)
        )
        per_episode_reef_fractions.append(reef_frac)
        per_episode_drives.append(ep_drive_log)
        per_episode_in_reef.append(ep_in_reef_log)

        if (ep_idx + 1) % 4 == 0 or ep_idx == num_episodes - 1:
            cur_ep = total_train_episodes - num_episodes + ep_idx + 1
            print(
                f"  [train] {arm_label} ep {cur_ep}/{total_train_episodes}"
                f"  reef_frac={reef_frac:.3f}"
                f"  steps={len(ep_in_reef_log)}",
                flush=True,
            )

    # C2 state-dependence: per-seed Spearman across binned drive vs binned reef-use.
    flat_drives: List[float] = []
    flat_in_reef: List[float] = []
    for d_log, r_log in zip(per_episode_drives, per_episode_in_reef):
        flat_drives.extend(d_log)
        flat_in_reef.extend([1.0 if r else 0.0 for r in r_log])
    rho_drive_reef = _spearman_rho(flat_drives, flat_in_reef)

    forage_hazard_rate = forage_hazard_events / max(forage_total_steps, 1)
    transit_hazard_rate = transit_hazard_events / max(transit_total_steps, 1)
    risk_type_ratio = forage_hazard_rate / max(transit_hazard_rate, 1e-6)

    return {
        "per_episode_reef_fractions": per_episode_reef_fractions,
        "mean_reef_fraction": float(np.mean(per_episode_reef_fractions)),
        "rho_drive_vs_reef": float(rho_drive_reef),
        "forage_hazard_rate": float(forage_hazard_rate),
        "transit_hazard_rate": float(transit_hazard_rate),
        "risk_type_ratio": float(risk_type_ratio),
        "n_forage_steps": int(forage_total_steps),
        "n_transit_steps": int(transit_total_steps),
        "n_forage_hazards": int(forage_hazard_events),
        "n_transit_hazards": int(transit_hazard_events),
    }


# ---------------------------------------------------------------------------
# Per-arm/seed run
# ---------------------------------------------------------------------------

def run_arm_seed(arm_label: str, use_gated_policy: bool, seed: int,
                 dry_run: bool) -> Dict:
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps = 2 if dry_run else EVAL_EPISODES
    steps_per_ep = 30 if dry_run else STEPS_PER_EPISODE
    total_train_eps = warmup_eps + eval_eps

    print(f"\nSeed {seed} Condition {arm_label}", flush=True)
    print(
        f"  use_gated_policy={use_gated_policy}"
        f"  warmup={warmup_eps}  eval={eval_eps}  steps/ep={steps_per_ep}",
        flush=True,
    )

    agent, env = _make_agent_and_env(seed, use_gated_policy)

    print(
        f"  world_obs_dim={env.world_obs_dim}"
        f"  agent.gated_policy={'on' if agent.gated_policy is not None else 'off'}",
        flush=True,
    )

    warmup = _warmup_train(
        agent, env, warmup_eps, steps_per_ep,
        total_train_episodes=total_train_eps, arm_label=arm_label,
    )

    eval_metrics = _eval_collect_metrics(
        agent, env, eval_eps, steps_per_ep,
        total_train_episodes=total_train_eps,
        arm_label=arm_label,
    )

    seed_summary = {
        "arm_label": arm_label,
        "seed": seed,
        "use_gated_policy": use_gated_policy,
        **warmup,
        **eval_metrics,
    }

    print(
        f"  seed={seed} arm={arm_label}"
        f"  reef_frac={eval_metrics['mean_reef_fraction']:.3f}"
        f"  rho={eval_metrics['rho_drive_vs_reef']:+.3f}"
        f"  forage_hr={eval_metrics['forage_hazard_rate']:.4f}"
        f"  transit_hr={eval_metrics['transit_hazard_rate']:.4f}"
        f"  ratio={eval_metrics['risk_type_ratio']:.3f}",
        flush=True,
    )

    # Per seed/arm verdict line for runner progress accounting.
    seed_pass = (
        eval_metrics["mean_reef_fraction"] > 0.0
        and eval_metrics["n_forage_steps"] >= 5
    )
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)
    return seed_summary


# ---------------------------------------------------------------------------
# Acceptance computation
# ---------------------------------------------------------------------------

def _aggregate_arm(seed_results: List[Dict]) -> Dict:
    rfs = [r["mean_reef_fraction"] for r in seed_results]
    rhos = [r["rho_drive_vs_reef"] for r in seed_results]
    forage_hrs = [r["forage_hazard_rate"] for r in seed_results]
    transit_hrs = [r["transit_hazard_rate"] for r in seed_results]
    ratios = [r["risk_type_ratio"] for r in seed_results]

    cov = (
        float(np.std(rfs) / max(abs(float(np.mean(rfs))), 1e-9))
        if len(rfs) > 1 else 0.0
    )
    return {
        "mean_reef_fraction": float(np.mean(rfs)),
        "std_reef_fraction": float(np.std(rfs)),
        "cov_reef_fraction": cov,
        "rhos_per_seed": rhos,
        "abs_rho_per_seed": [abs(r) for r in rhos],
        "n_rho_above_threshold": sum(1 for r in rhos if abs(r) >= C2_RHO_THRESHOLD),
        "mean_abs_rho": float(np.mean([abs(r) for r in rhos])),
        "mean_forage_hazard_rate": float(np.mean(forage_hrs)),
        "mean_transit_hazard_rate": float(np.mean(transit_hrs)),
        "mean_risk_type_ratio": float(np.mean(ratios)),
    }


def _compute_acceptance(arm_summaries: Dict[str, Dict]) -> Dict:
    """Apply the pre-registered acceptance grid to per-arm aggregates."""
    a0 = arm_summaries["ARM_0_baseline"]
    a1 = arm_summaries["ARM_1c_full_3stream"]

    # C2 state-dependence
    c2_a1_pass = (a1["n_rho_above_threshold"] >= C2_MIN_PASS_SEEDS)
    c2_a1_beats_a0 = (a1["mean_abs_rho"] > a0["mean_abs_rho"])
    c2_pass = bool(c2_a1_pass and c2_a1_beats_a0)

    # C3 risk-type dissociation: |ARM_1c - ARM_0| / max(ARM_0, eps) >= 0.50
    a0_ratio = a0["mean_risk_type_ratio"]
    a1_ratio = a1["mean_risk_type_ratio"]
    c3_relative_delta = abs(a1_ratio - a0_ratio) / max(abs(a0_ratio), 1e-6)
    c3_pass = bool(c3_relative_delta >= C3_RELATIVE_DELTA_THRESHOLD)

    # C4 cross-seed variation
    c4_pass = bool(a1["cov_reef_fraction"] >= C4_COV_THRESHOLD)

    n_criteria_passed = int(c2_pass) + int(c3_pass) + int(c4_pass)
    pass_rule_met = (n_criteria_passed >= 2)

    # F1 unambiguous monomodal-collapse signature.
    reef_diff = abs(a1["mean_reef_fraction"] - a0["mean_reef_fraction"])
    f1_signature = bool(
        reef_diff < F1_REEF_DIFF_THRESHOLD
        and a0["mean_abs_rho"] < F1_C2_RHO_THRESHOLD
        and a1["mean_abs_rho"] < F1_C2_RHO_THRESHOLD
        and c3_relative_delta < F1_C3_DELTA_THRESHOLD
        and a0["cov_reef_fraction"] < F1_C4_COV_THRESHOLD
        and a1["cov_reef_fraction"] < F1_C4_COV_THRESHOLD
    )

    # F2 biologically inverted signature: ARM_1c rho monotonically positive.
    a1_mean_rho_signed = float(np.mean(a1["rhos_per_seed"])) if a1["rhos_per_seed"] else 0.0
    f2_inverted = bool(a1_mean_rho_signed > F2_INVERTED_RHO_THRESHOLD)

    overall_pass = pass_rule_met and not f1_signature and not f2_inverted

    return {
        "C1_density_tracking": "non_contributory_phase2a_single_density",
        "C2_state_dependence_pass": c2_pass,
        "C2_a1_seeds_above_threshold": a1["n_rho_above_threshold"],
        "C2_a1_mean_abs_rho": a1["mean_abs_rho"],
        "C2_a0_mean_abs_rho": a0["mean_abs_rho"],
        "C3_risk_type_dissociation_pass": c3_pass,
        "C3_relative_delta": c3_relative_delta,
        "C3_a0_ratio": a0_ratio,
        "C3_a1_ratio": a1_ratio,
        "C4_cross_seed_variation_pass": c4_pass,
        "C4_a1_cov_reef_fraction": a1["cov_reef_fraction"],
        "C4_a0_cov_reef_fraction": a0["cov_reef_fraction"],
        "n_criteria_passed": n_criteria_passed,
        "pass_rule_met": pass_rule_met,
        "F1_monomodal_collapse_signature": f1_signature,
        "F2_biologically_inverted_signature": f2_inverted,
        "overall_pass": overall_pass,
    }


def _compute_per_claim_direction(acceptance: Dict) -> Tuple[str, Dict[str, str]]:
    """Determine outcome + per-claim direction grid.

    Per the docstring:
      MECH-309 (primary, logical-necessity falsifier):
        supports if PASS; weakens if F1 or F2; non_contributory if mixed.
      ARC-062 (V3 weak-reading architectural validation):
        supports if PASS; weakens if F1 or F2; non_contributory if mixed.
      SD-029 (monomodal-collapse measurement gate):
        supports if PASS; weakens if F1; non_contributory otherwise.
    """
    pass_ = acceptance["overall_pass"]
    f1 = acceptance["F1_monomodal_collapse_signature"]
    f2 = acceptance["F2_biologically_inverted_signature"]

    if pass_:
        outcome = "PASS"
        per_claim = {
            "MECH-309": "supports",
            "ARC-062": "supports",
            "SD-029": "supports",
        }
    elif f1 or f2:
        outcome = "FAIL"
        per_claim = {
            "MECH-309": "weakens",
            "ARC-062": "weakens",
            "SD-029": "weakens" if f1 else "non_contributory",
        }
    else:
        outcome = "FAIL"
        per_claim = {
            "MECH-309": "non_contributory",
            "ARC-062": "non_contributory",
            "SD-029": "non_contributory",
        }
    return outcome, per_claim


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]

    arms = [
        ("ARM_0_baseline", False),
        ("ARM_1c_full_3stream", True),
    ]

    print(
        f"[V3-EXQ-543] ARC-062 Phase 2a Monomodal-Collapse Falsifier"
        f"  seeds={seeds}  dry_run={dry_run}",
        flush=True,
    )

    seed_results_by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, use_gated in arms:
            r = run_arm_seed(arm_label, use_gated, seed, dry_run=dry_run)
            seed_results_by_arm[arm_label].append(r)

    arm_summaries = {
        arm_label: _aggregate_arm(seed_results_by_arm[arm_label])
        for arm_label, _ in arms
    }
    acceptance = _compute_acceptance(arm_summaries)

    return {
        "arm_summaries": arm_summaries,
        "seed_results_by_arm": seed_results_by_arm,
        "acceptance": acceptance,
    }


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acceptance = result["acceptance"]
    outcome, per_claim_direction = _compute_per_claim_direction(acceptance)
    overall_direction = (
        "supports" if outcome == "PASS"
        else (
            "weakens" if (
                acceptance["F1_monomodal_collapse_signature"]
                or acceptance["F2_biologically_inverted_signature"]
            )
            else "non_contributory"
        )
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": per_claim_direction,
        "metrics": {
            "arm_summaries": result["arm_summaries"],
            "acceptance": acceptance,
            "per_seed_per_arm": {
                k: [{kk: vv for kk, vv in s.items() if kk != "per_episode_reef_fractions"} | {"per_episode_reef_fractions": s["per_episode_reef_fractions"]} for s in v]
                for k, v in result["seed_results_by_arm"].items()
            },
        },
        "elapsed_seconds": elapsed,
        "dry_run": dry_run,
        "notes": (
            "ARC-062 Phase 2a single-arm core 2-arm contrast (ARM_0 baseline vs "
            "ARM_1c full 3-stream discriminator) at ARM_1_med density. C1 density-"
            "tracking marked non_contributory (single-density Phase 2a; deferred to "
            "Phase 2b density-gradient sub-arms). Phase 2 deliverable per "
            "REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md "
            "(GAP-B). PASS unblocks Phase 3 (closes commitment_closure GAP-1: "
            "wires discriminator -> SD-033a + adds bias-head parameters to E3 "
            "optimiser). FAIL routes through the falsification chain in Phase 2 "
            "deliverable 5: BG-side score-aggregation -> trajectory-proposal-level -> "
            "ARC-063 V4 strong reading."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path, outcome


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with reduced episodes/steps for smoke testing.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Override seed list (default [0, 1, 2]).")
    args = parser.parse_args()

    t0 = time.time()
    result = run(seeds=args.seeds, dry_run=args.dry_run)
    elapsed = time.time() - t0

    out_path, outcome = write_manifest(result, args.dry_run, elapsed)

    acc = result["acceptance"]
    print("\n=== V3-EXQ-543 SUMMARY ===", flush=True)
    print(
        f"  C2 state_dependence: pass={acc['C2_state_dependence_pass']}"
        f"  a1_mean_abs_rho={acc['C2_a1_mean_abs_rho']:.3f}"
        f"  a0_mean_abs_rho={acc['C2_a0_mean_abs_rho']:.3f}",
        flush=True,
    )
    print(
        f"  C3 risk_type:       pass={acc['C3_risk_type_dissociation_pass']}"
        f"  delta={acc['C3_relative_delta']:.3f}"
        f"  a0_ratio={acc['C3_a0_ratio']:.3f}"
        f"  a1_ratio={acc['C3_a1_ratio']:.3f}",
        flush=True,
    )
    print(
        f"  C4 cross_seed_cov:  pass={acc['C4_cross_seed_variation_pass']}"
        f"  a1_cov={acc['C4_a1_cov_reef_fraction']:.3f}"
        f"  a0_cov={acc['C4_a0_cov_reef_fraction']:.3f}",
        flush=True,
    )
    print(
        f"  n_criteria_passed:  {acc['n_criteria_passed']} (rule = >=2)"
        f"  F1={acc['F1_monomodal_collapse_signature']}"
        f"  F2={acc['F2_biologically_inverted_signature']}",
        flush=True,
    )
    print(f"  outcome:            {outcome}", flush=True)
    print(f"  elapsed:            {elapsed:.1f}s", flush=True)
    print(f"Result written to: {out_path}", flush=True)

    _outcome_raw = str(outcome).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
    )
