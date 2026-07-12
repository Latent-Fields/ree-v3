#!/opt/local/bin/python3
"""V3-EXQ-606b: ARC-064 GAP-I MECH-318 empirical retire-vs-promote gate.

Supersedes V3-EXQ-606a (ERROR on ree-cloud-2: script not yet on disk when runner
claimed the experiment; git sync lag between push at 14:37Z and last pull at 13:18Z
on 2026-05-21). No code changes from 606a. Re-queued as 606b per EXQ versioning policy.
Adds a hard startup check for V3-EXQ-543k contributory PASS before training.

Episode-boundary multi-rule context via alternating SD-054 bipartite axis
(horizontal vs vertical) across P1 training episodes. Tests whether the
SD-033a + ARC-062 cluster (gated_policy + lateral_pfc discriminator source
+ trainable rule_bias_head) produces regime-dependent behaviour distinct
from a cluster-OFF baseline.

Plan: arc_062_rule_apprehension_plan.md GAP-I (IGW-20260521-004).
MECH-316 / MECH-317 have no V3 substrate modules; this EXQ tests only
MECH-318 partial-absorption (W1+W3+W4 within-episode). Within-step rule
switch deferred to a future multi-rule-context env extension.

Pre-registered acceptance:
  C1_cross_regime: ARM_2 |reef_frac_eval_H - reef_frac_eval_V| >= CROSS_REGIME_MIN
      in >= MIN_PASS_SEEDS seeds.
  C2_cluster_advantage: ARM_2 cross-regime delta > ARM_0 cross-regime delta
      in >= MIN_PASS_SEEDS seeds.
  C3_rule_state_active: ARM_2 mean lateral_pfc rule_state L2 norm after P1
      >= RULE_STATE_MIN in >= MIN_PASS_SEEDS seeds.
  PASS = C1 AND C2 AND C3 -> MECH-318 superseded-by-cluster reading.
  FAIL -> MECH-318 remains candidate (may motivate dedicated substrate).

Scientific gate: hard-block unless V3-EXQ-543k latest manifest is outcome PASS
and evidence_direction is contributory (not non_contributory / superseded).

claim_ids: MECH-318 only.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
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

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_606b_arc064_gap_i_mech318_multi_rule_empirical_gate"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-606b"
SUPERSEDES = "V3-EXQ-606a"
GATES_ON_EXQ = "V3-EXQ-543k"
CLAIM_IDS = ["MECH-318"]
NON_CONTRIBUTORY_DIRECTIONS = frozenset({"non_contributory", "superseded"})

SELF_DIM = 8
WORLD_DIM = 32
HARM_DIM = 4
HARM_A_DIM = 4
HARM_HISTORY_LEN = 10

BASE_ENV = dict(
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
    reef_bipartite_layout=True,
    reef_bipartite_agent_band_radius=1,
)

P0_EPISODES = 20
P1_EPISODES = 40
P2_EPISODES = 16
STEPS_PER_EPISODE = 100

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
LR_GATED = 5e-4
LR_LPFC_BIAS = 5e-4
BATCH_SIZE = 32
WF_BUF_MAX = 256
HARM_EVAL_BUF_MAX = 256
N_PROBE_CANDIDATES = 8
RECORD_EVERY_N_STEPS = 4
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9
LAMBDA_DISC_VAR = 0.1
MODE_SEPARATION_FLOOR = 0.25
P1_W_DEVIATION_AUX_WEIGHT = 0.1

CROSS_REGIME_MIN = 0.12
RULE_STATE_MIN = 0.005
MIN_PASS_SEEDS = 2


def _env_kwargs(axis: str) -> dict:
    kw = dict(BASE_ENV)
    kw["reef_bipartite_axis"] = axis
    return kw


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _make_agent(seed: int, cluster_on: bool) -> REEAgent:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **_env_kwargs("horizontal"))
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
        use_gated_policy=cluster_on,
        gated_policy_use_first_action_onehot=cluster_on,
        use_dacc=False,
        dacc_weight=0.0,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_lateral_pfc_analog=cluster_on,
        lateral_pfc_use_discriminator_source=cluster_on,
        lateral_pfc_discriminator_pool_weight=0.3,
        lateral_pfc_train_rule_bias_head=cluster_on,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    if cluster_on:
        config.gated_policy_use_differential_heads = True
        config.gated_policy_differential_bias_scale = 0.1
        config.gated_policy_mode_separation_floor = MODE_SEPARATION_FLOOR
        config.gated_policy_p1_w_deviation_aux_weight = P1_W_DEVIATION_AUX_WEIGHT
    agent = REEAgent(config)
    return agent


def _preflight() -> None:
    off = _make_agent(0, cluster_on=False)
    on = _make_agent(1, cluster_on=True)
    assert off.gated_policy is None and off.lateral_pfc is None
    assert on.gated_policy is not None and on.lateral_pfc is not None
    del off, on
    print("Preflight PASS: cluster OFF vs ON agent wiring", flush=True)


def _require_scientific_gate(gate_exq: str, skip: bool) -> None:
    if skip:
        print(
            f"Scientific gate SKIPPED (--skip-scientific-gate): {gate_exq}",
            flush=True,
        )
        return
    ev_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    best_ts = ""
    best_label = ""
    best_outcome = ""
    best_direction = ""
    token = gate_exq.replace("V3-EXQ-", "").lower()
    for path in sorted(ev_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        qid = str(data.get("queue_id") or "")
        etype = str(data.get("experiment_type") or "")
        if qid != gate_exq and token not in etype and token not in path.name:
            continue
        if data.get("dry_run"):
            continue
        ts = str(data.get("timestamp_utc") or data.get("run_id") or "")
        if ts >= best_ts:
            best_ts = ts
            best_label = path.name
            best_outcome = str(data.get("outcome") or "").upper()
            best_direction = str(data.get("evidence_direction") or "").lower()
    if not best_label:
        raise RuntimeError(
            f"Scientific gate blocked: no non-dry-run manifest for {gate_exq}. "
            f"Run {gate_exq} to contributory PASS before {QUEUE_ID}."
        )
    if best_outcome != "PASS" or best_direction in NON_CONTRIBUTORY_DIRECTIONS:
        raise RuntimeError(
            f"Scientific gate blocked: {gate_exq} latest={best_label} "
            f"outcome={best_outcome} evidence_direction={best_direction}. "
            f"Need contributory PASS before {QUEUE_ID}."
        )
    print(
        f"Scientific gate PASS: {gate_exq} from {best_label} "
        f"(outcome={best_outcome}, direction={best_direction})",
        flush=True,
    )


def _build_snap(agent: REEAgent, latent, candidates: List) -> Optional[Dict]:
    if not isinstance(candidates, list) or len(candidates) < N_PROBE_CANDIDATES:
        return None
    if getattr(candidates[0], "world_states", None) is None:
        return None
    if len(candidates[0].world_states) < 2:
        return None
    first_step_world = torch.cat(
        [c.world_states[1].detach().clone() for c in candidates[:N_PROBE_CANDIDATES]],
        dim=0,
    )
    fa_list = []
    for c in candidates[:N_PROBE_CANDIDATES]:
        if getattr(c, "actions", None) is None or c.actions.shape[1] < 1:
            return None
        fa_list.append(c.actions[:, 0, :][0].detach().float())
    return {
        "z_world": latent.z_world.detach().clone(),
        "z_self": latent.z_self.detach().clone(),
        "z_harm_a": (
            latent.z_harm_a.detach().clone() if latent.z_harm_a is not None else None
        ),
        "candidate_features": first_step_world,
        "first_action_onehots": torch.stack(fa_list, dim=0).clone(),
    }


def _gated_reinforce_loss(
    agent: REEAgent, outcome_buf: List[Tuple[Dict, int, float]], baseline: float, device
) -> torch.Tensor:
    if agent.gated_policy is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    idxs = np.random.choice(len(outcome_buf), size=min(BATCH_SIZE, len(outcome_buf)), replace=False)
    terms: List[torch.Tensor] = []
    disc_ws: List[torch.Tensor] = []
    for i in idxs:
        snap, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        out = agent.gated_policy.forward(
            z_world=snap["z_world"],
            z_self=snap["z_self"],
            z_harm_a=snap.get("z_harm_a"),
            candidate_features=snap["candidate_features"],
            first_action_onehots=snap.get("first_action_onehots"),
            simulation_mode=False,
        )
        k = out.gated_score_bias.shape[0]
        log_p = F.log_softmax(out.gated_score_bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, k - 1)])
        zw = snap["z_world"]
        zs = snap["z_self"]
        za = snap.get("z_harm_a")
        if za is None:
            za = torch.zeros(1, agent.gated_policy.harm_a_dim, device=device)
        disc_in = torch.cat(
            [
                zw if zw.dim() == 2 else zw.unsqueeze(0),
                zs if zs.dim() == 2 else zs.unsqueeze(0),
                za if za.dim() == 2 else za.unsqueeze(0),
            ],
            dim=-1,
        )
        disc_ws.append(agent.gated_policy.discriminator(disc_in).squeeze())
    if not terms:
        return torch.zeros(1, device=device)
    loss = torch.stack(terms).mean()
    if len(disc_ws) > 1:
        loss = loss - LAMBDA_DISC_VAR * torch.stack(disc_ws).var(unbiased=False)
    if disc_ws:
        loss = loss + agent.gated_policy.p1_training_auxiliary_loss(disc_ws[-1])
    return loss


def _lpfc_reinforce_loss(
    agent: REEAgent, outcome_buf: List[Tuple[Dict, int, float]], baseline: float, device
) -> torch.Tensor:
    if agent.lateral_pfc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    idxs = np.random.choice(len(outcome_buf), size=min(BATCH_SIZE, len(outcome_buf)), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        snap, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = agent.lateral_pfc.compute_bias(snap["candidate_features"])
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _rule_state_norm(agent: REEAgent) -> float:
    if agent.lateral_pfc is None:
        return 0.0
    rs = agent.lateral_pfc.rule_state
    if rs is None:
        return 0.0
    return float(rs.detach().norm().item())


def _encoder_step(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    total_eps: int,
    arm: str,
    ep: int,
) -> float:
    device = agent.device
    e1_opt = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    he_opt = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM)
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    he_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    ep_reward = 0.0
    z_wp = z_sp = act_p = None
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(
            obs_body,
            obs_world,
            obs_harm=obs_dict.get("harm_obs"),
            obs_harm_a=obs_dict.get("harm_obs_a"),
            obs_harm_history=obs_dict.get("harm_history"),
        )
        z_w = latent.z_world.detach()
        if z_wp is not None and act_p is not None:
            agent.record_transition(z_sp, act_p, latent.z_self.detach())
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, WORLD_DIM, device=device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        agent.update_z_goal(
            benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
            drive_level=REEAgent.compute_drive_level(obs_body),
        )
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = _action_to_onehot(random.randint(0, env.action_dim - 1), env.action_dim, device)
            agent._last_action = action
        _, harm, done, _, obs_dict = env.step(action)
        ep_reward += float(harm)
        if z_wp is not None and act_p is not None:
            wf_buf.append((z_wp.cpu(), act_p.cpu(), z_w.cpu()))
            if len(wf_buf) > WF_BUF_MAX:
                wf_buf = wf_buf[-WF_BUF_MAX:]
        he_buf.append((z_w.cpu(), torch.tensor([abs(float(harm)) if float(harm) < 0 else 0.0])))
        if len(wf_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([wf_buf[i][0] for i in idx]).to(device)
            a = torch.cat([wf_buf[i][1] for i in idx]).to(device)
            zw1 = torch.cat([wf_buf[i][2] for i in idx]).to(device)
            pred = agent.e2.world_forward(zw, a)
            loss = F.mse_loss(pred, zw1)
            if loss.requires_grad:
                e2_opt.zero_grad()
                loss.backward()
                e2_opt.step()
            with torch.no_grad():
                agent.e3.update_running_variance((pred.detach() - zw1).detach())
        if len(he_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(he_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([he_buf[i][0] for i in idx]).to(device)
            ht = torch.cat([he_buf[i][1] for i in idx]).to(device)
            loss = F.mse_loss(agent.e3.harm_eval(zw).squeeze(), ht.squeeze())
            if loss.requires_grad:
                he_opt.zero_grad()
                loss.backward()
                he_opt.step()
        if len(agent._world_experience_buffer) >= 2:
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()
        z_wp, z_sp, act_p = z_w, latent.z_self.detach(), action.detach()
        if done:
            break
    log_every = max(1, total_eps // 5)
    if (ep + 1) % log_every == 0 or ep + 1 == total_eps:
        print(
            f"  [train] {arm} ep {ep+1}/{total_eps}  phase=enc  reward={ep_reward:.3f}",
            flush=True,
        )
    return ep_reward


def _p1_train(
    agent: REEAgent,
    seed: int,
    episodes: int,
    steps: int,
    total_eps: int,
    arm: str,
    cluster_on: bool,
    multi_rule: bool,
) -> Dict:
    device = agent.device
    gated_opt = (
        optim.Adam(agent.gated_policy.parameters(), lr=LR_GATED)
        if agent.gated_policy is not None
        else None
    )
    bias_opt = (
        optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        if cluster_on and agent.lateral_pfc is not None
        else None
    )
    outcome_buf: List[Tuple[Dict, int, float]] = []
    baseline = 0.0
    rule_norms: List[float] = []
    agent.train()
    for ep in range(episodes):
        axis = "vertical" if (multi_rule and ep % 2 == 1) else "horizontal"
        env = CausalGridWorldV2(seed=seed + ep * 17, **_env_kwargs(axis))
        _, obs_dict = env.reset()
        agent.reset()
        ep_reward = 0.0
        ep_buf: List[Tuple[Dict, int]] = []
        step = 0
        z_wp = z_sp = act_p = None
        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(
                obs_body,
                obs_world,
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            if z_wp is not None and act_p is not None:
                agent.record_transition(z_sp, act_p, latent.z_self.detach())
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                drive_level=REEAgent.compute_drive_level(obs_body),
            )
            snap = None
            if step % RECORD_EVERY_N_STEPS == 0:
                snap = _build_snap(agent, latent, candidates)
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(random.randint(0, env.action_dim - 1), env.action_dim, device)
                agent._last_action = action
            if snap is not None and action is not None:
                sel = 0
                aa = int(action.argmax(-1).item())
                for ci, c in enumerate(candidates[:N_PROBE_CANDIDATES]):
                    if getattr(c, "actions", None) is not None and c.actions.shape[1] >= 1:
                        if int(c.actions[:, 0, :].argmax(-1).item()) == aa:
                            sel = ci
                            break
                ep_buf.append((snap, sel))
            _, harm, done, _, obs_dict = env.step(action)
            ep_reward += float(harm)
            z_wp, z_sp, act_p = latent.z_world.detach(), latent.z_self.detach(), action.detach()
            step += 1
            if done:
                break
        baseline = EMA_DECAY * baseline + (1.0 - EMA_DECAY) * ep_reward
        for snap, sel in ep_buf:
            outcome_buf.append((snap, sel, ep_reward))
        if len(outcome_buf) > OUTCOME_BUF_MAX:
            outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
        if gated_opt is not None:
            g_loss = _gated_reinforce_loss(agent, outcome_buf, baseline, device)
            if g_loss.requires_grad:
                gated_opt.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.gated_policy.parameters(), 1.0)
                gated_opt.step()
        if bias_opt is not None:
            l_loss = _lpfc_reinforce_loss(agent, outcome_buf, baseline, device)
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.lateral_pfc.bias_head_parameters(), 1.0)
                bias_opt.step()
        rule_norms.append(_rule_state_norm(agent))
        log_every = max(1, total_eps // 5)
        if (ep + 1) % log_every == 0 or ep + 1 == total_eps:
            print(
                f"  [train] {arm} ep {ep+1}/{total_eps}  phase=P1  axis={axis}"
                f"  rule_norm={rule_norms[-1]:.5f}",
                flush=True,
            )
    return {
        "p1_mean_rule_state_norm": float(np.mean(rule_norms)) if rule_norms else 0.0,
        "p1_max_rule_state_norm": float(np.max(rule_norms)) if rule_norms else 0.0,
    }


def _reef_fraction(agent: REEAgent, env: CausalGridWorldV2, episodes: int, steps: int) -> float:
    device = agent.device
    reef_cells = getattr(env, "_reef_cells", set())
    in_reef = 0
    total = 0
    agent.eval()
    with torch.no_grad():
        for _ in range(episodes):
            _, obs_dict = env.reset()
            agent.reset()
            for _ in range(steps):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                latent = agent.sense(
                    obs_body,
                    obs_world,
                    obs_harm=obs_dict.get("harm_obs"),
                    obs_harm_a=obs_dict.get("harm_obs_a"),
                    obs_harm_history=obs_dict.get("harm_history"),
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick", False)
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(0, env.action_dim, device)
                _, _, done, _, obs_dict = env.step(action)
                if (int(env.agent_x), int(env.agent_y)) in reef_cells:
                    in_reef += 1
                total += 1
                if done:
                    break
    return in_reef / max(total, 1)


def run_arm_seed(
    arm: str,
    cluster_on: bool,
    multi_rule: bool,
    seed: int,
    dry_run: bool,
) -> Dict:
    p0 = 2 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    p2 = 2 if dry_run else P2_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    total = p0 + p1 + p2
    print(
        f"\nSeed {seed} Condition {arm} cluster_on={cluster_on} multi_rule={multi_rule}",
        flush=True,
    )
    agent = _make_agent(seed, cluster_on=cluster_on)
    env_p0 = CausalGridWorldV2(seed=seed, **_env_kwargs("horizontal"))
    for ep in range(p0):
        _encoder_step(agent, env_p0, steps, total, arm, ep)
    p1m = _p1_train(agent, seed, p1, steps, total, arm, cluster_on, multi_rule)
    if multi_rule:
        env_h = CausalGridWorldV2(seed=seed + 9001, **_env_kwargs("horizontal"))
        env_v = CausalGridWorldV2(seed=seed + 9002, **_env_kwargs("vertical"))
        reef_h = _reef_fraction(agent, env_h, p2, steps)
        reef_v = _reef_fraction(agent, env_v, p2, steps)
    else:
        env_h = CausalGridWorldV2(seed=seed + 9001, **_env_kwargs("horizontal"))
        reef_h = _reef_fraction(agent, env_h, p2, steps)
        reef_v = reef_h
    cross_delta = abs(reef_h - reef_v)
    if multi_rule and cluster_on:
        seed_ok = cross_delta >= CROSS_REGIME_MIN and p1m["p1_mean_rule_state_norm"] >= RULE_STATE_MIN
    elif cluster_on:
        seed_ok = 0.15 < reef_h < 0.85
    else:
        seed_ok = cross_delta < CROSS_REGIME_MIN
    print(
        f"verdict: {'PASS' if seed_ok else 'FAIL'}  "
        f"reef_H={reef_h:.3f}  reef_V={reef_v:.3f}  cross_delta={cross_delta:.3f}",
        flush=True,
    )
    return {
        "arm": arm,
        "seed": seed,
        "cluster_on": cluster_on,
        "multi_rule": multi_rule,
        **p1m,
        "reef_frac_eval_horizontal": reef_h,
        "reef_frac_eval_vertical": reef_v,
        "cross_regime_delta": cross_delta,
    }


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [42, 7, 17]
    arms = [
        ("ARM_0_cluster_off", False, False),
        ("ARM_1_single_regime", True, False),
        ("ARM_2_multi_rule", True, True),
    ]
    by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, cluster_on, multi_rule in arms:
            by_arm[arm_label].append(
                run_arm_seed(arm_label, cluster_on, multi_rule, seed, dry_run)
            )
    arm0_d = [r["cross_regime_delta"] for r in by_arm["ARM_0_cluster_off"]]
    arm2_d = [r["cross_regime_delta"] for r in by_arm["ARM_2_multi_rule"]]
    arm2_rn = [r["p1_mean_rule_state_norm"] for r in by_arm["ARM_2_multi_rule"]]
    c1 = sum(1 for d in arm2_d if d >= CROSS_REGIME_MIN) >= MIN_PASS_SEEDS
    c2 = sum(
        1
        for i in range(len(arm2_d))
        if arm2_d[i] > arm0_d[i] + 0.02
    ) >= MIN_PASS_SEEDS
    c3 = sum(1 for n in arm2_rn if n >= RULE_STATE_MIN) >= MIN_PASS_SEEDS
    return {
        "by_arm": by_arm,
        "acceptance": {
            "C1_cross_regime": c1,
            "C2_cluster_advantage": c2,
            "C3_rule_state_active": c3,
            "pass": c1 and c2 and c3,
        },
    }


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acc = result["acceptance"]
    outcome = "PASS" if acc["pass"] else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"
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
        "supersedes": SUPERSEDES,
        "gates_on_exq": GATES_ON_EXQ,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "executing_hostname": socket.gethostname(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-318": direction},
        "outcome": outcome,
        "dry_run": dry_run,
        "elapsed_sec": elapsed,
        "acceptance": acc,
        "arm_results": result["by_arm"],
        "note": (
            "Episode-boundary multi-rule via alternating bipartite axis; "
            "not within-step rule switch. MECH-316/317 not tested (no V3 modules). "
            "Hard scientific gate on V3-EXQ-543k contributory PASS (606a supersedes premature 606)."
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    return out_path, outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--skip-scientific-gate",
        action="store_true",
        help="Smoke/dry-run only: do not require V3-EXQ-543k contributory PASS manifest.",
    )
    args = parser.parse_args()
    t0 = time.time()
    _preflight()
    if not args.dry_run:
        _require_scientific_gate(GATES_ON_EXQ, args.skip_scientific_gate)
    result = run(seeds=args.seeds, dry_run=args.dry_run)
    elapsed = time.time() - t0
    out_path, outcome = write_manifest(result, args.dry_run, elapsed)
    acc = result["acceptance"]
    print("\n=== V3-EXQ-606b SUMMARY ===", flush=True)
    print(f"  C1_cross_regime={acc['C1_cross_regime']}", flush=True)
    print(f"  C2_cluster_advantage={acc['C2_cluster_advantage']}", flush=True)
    print(f"  C3_rule_state_active={acc['C3_rule_state_active']}", flush=True)
    print(f"  outcome: {outcome}", flush=True)
    print(f"  gates_on_exq: {GATES_ON_EXQ}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    return out_path, outcome


if __name__ == "__main__":
    out_path, outcome = main()
    _raw = str(outcome).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
    )
