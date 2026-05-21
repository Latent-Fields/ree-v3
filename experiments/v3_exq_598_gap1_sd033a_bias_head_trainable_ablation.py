#!/opt/local/bin/python3
"""V3-EXQ-598: commitment_closure GAP-1 -- SD-033a bias-head trainable ablation.

2-arm validation on the ARC-062 + SD-054 bipartite stack (post-SP-CEM defaults,
differential heads + mode_separation_floor from the 543i/543k substrate line).
Single varied factor: lateral_pfc_train_rule_bias_head (frozen-zero vs trainable).

Pre-registered acceptance (commitment_closure_plan Phase 1 / arc_062 Phase 3):
  C1_frozen_silent: ARM_0 mean |score_bias| after P1 < FROZEN_BIAS_MAX all seeds.
  C2_trainable_nonzero: ARM_1 mean |score_bias| after P1 >= TRAINABLE_BIAS_MIN
      in >= MIN_PASS_SEEDS of 3 seeds.
  C3_trainable_not_monomodal: ARM_1 P2 reef_visit_fraction in (REEF_LO, REEF_HI)
      in >= MIN_PASS_SEEDS seeds (non-trivial reef/forage split proxy).
  PASS = C1 AND C2 AND C3.

Scientific gate: interpret closure only after V3-EXQ-543k (arc_062 GAP-B retest)
returns a contributory result. Queued at priority 4 (below 543k priority 5).

claim_ids: SD-033a (primary); MECH-262 unblocked only if PASS closes GAP-1.

SLEEP DRIVER: not applicable (no sleep loop in this experiment).
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

EXPERIMENT_TYPE = "v3_exq_598_gap1_sd033a_bias_head_trainable_ablation"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-598"
GATES_ON_EXQ = "V3-EXQ-543k"
NON_CONTRIBUTORY_DIRECTIONS = frozenset({"non_contributory", "superseded"})
CLAIM_IDS = ["SD-033a"]

SELF_DIM = 8
WORLD_DIM = 32
HARM_DIM = 4
HARM_A_DIM = 4
HARM_HISTORY_LEN = 10

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
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

P0_EPISODES = 30
P1_EPISODES = 50
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

FROZEN_BIAS_MAX = 1e-4
TRAINABLE_BIAS_MIN = 0.002
REEF_LO = 0.20
REEF_HI = 0.80
MIN_PASS_SEEDS = 2


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _make_agent(seed: int, train_rule_bias_head: bool) -> Tuple[REEAgent, CausalGridWorldV2]:
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
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        use_dacc=False,
        dacc_weight=0.0,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_lateral_pfc_analog=True,
        lateral_pfc_use_discriminator_source=True,
        lateral_pfc_discriminator_pool_weight=0.3,
        lateral_pfc_train_rule_bias_head=train_rule_bias_head,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.gated_policy_use_differential_heads = True
    config.gated_policy_differential_bias_scale = 0.1
    config.gated_policy_mode_separation_floor = MODE_SEPARATION_FLOOR
    config.gated_policy_p1_w_deviation_aux_weight = P1_W_DEVIATION_AUX_WEIGHT
    agent = REEAgent(config)
    return agent, env


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


def _preflight() -> None:
    agent, _ = _make_agent(0, train_rule_bias_head=False)
    assert agent.gated_policy is not None
    assert agent.lateral_pfc is not None
    assert agent.gated_policy._use_diff is True
    frozen, _ = _make_agent(1, train_rule_bias_head=False)
    trainable, _ = _make_agent(2, train_rule_bias_head=True)
    assert frozen.lateral_pfc.config.train_rule_bias_head is False
    assert trainable.lateral_pfc.config.train_rule_bias_head is True
    del agent, frozen, trainable
    print("Preflight PASS: gated_policy + lateral_pfc + train_rule_bias_head flag", flush=True)


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


def _encoder_step(agent: REEAgent, env, steps: int, total_eps: int, arm: str, ep: int) -> float:
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
    for step_i in range(steps):
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
        drive = REEAgent.compute_drive_level(obs_body)
        agent.update_z_goal(
            benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
            drive_level=drive,
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
    if (ep + 1) % 10 == 0:
        print(
            f"  [train] {arm} ep {ep+1}/{total_eps}  phase=enc  reward={ep_reward:.3f}",
            flush=True,
        )
    return ep_reward


def _p1_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    episodes: int,
    steps: int,
    total_eps: int,
    arm: str,
    train_bias: bool,
) -> Dict:
    device = agent.device
    gated_opt = optim.Adam(agent.gated_policy.parameters(), lr=LR_GATED)
    bias_opt = (
        optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        if train_bias
        else None
    )
    outcome_buf: List[Tuple[Dict, int, float]] = []
    baseline = 0.0
    bias_samples: List[float] = []
    agent.train()
    for ep in range(episodes):
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
            if agent.lateral_pfc is not None:
                bias_samples.append(float(agent.lateral_pfc._last_bias_abs_mean))
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
        if (ep + 1) % 10 == 0:
            print(
                f"  [train] {arm} ep {ep+1}/{total_eps}  phase=P1"
                f"  bias_abs={bias_samples[-1] if bias_samples else 0:.5f}",
                flush=True,
            )
    return {
        "p1_mean_abs_lpfc_bias": float(np.mean(bias_samples)) if bias_samples else 0.0,
        "p1_max_abs_lpfc_bias": float(np.max(bias_samples)) if bias_samples else 0.0,
    }


def _p2_reef_fraction(agent: REEAgent, env: CausalGridWorldV2, episodes: int, steps: int) -> float:
    device = agent.device
    reef_cells = getattr(env, "_reef_cells", set())
    in_reef_steps = 0
    total_steps = 0
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
                pos = (int(env.agent_x), int(env.agent_y))
                if pos in reef_cells:
                    in_reef_steps += 1
                total_steps += 1
                if done:
                    break
    return in_reef_steps / max(total_steps, 1)


def run_arm_seed(arm: str, train_bias: bool, seed: int, dry_run: bool) -> Dict:
    p0 = 3 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    p2 = 2 if dry_run else P2_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    total = p0 + p1 + p2
    print(f"\nSeed {seed} Condition {arm} train_rule_bias_head={train_bias}", flush=True)
    agent, env = _make_agent(seed, train_rule_bias_head=train_bias)
    for ep in range(p0):
        _encoder_step(agent, env, steps, total, arm, ep)
    p1m = _p1_train(agent, env, p1, steps, total, arm, train_bias)
    reef_frac = _p2_reef_fraction(agent, env, p2, steps)
    passed = (
        (not train_bias and p1m["p1_mean_abs_lpfc_bias"] < FROZEN_BIAS_MAX)
        or (
            train_bias
            and p1m["p1_mean_abs_lpfc_bias"] >= TRAINABLE_BIAS_MIN
            and REEF_LO < reef_frac < REEF_HI
        )
    )
    print(
        f"verdict: {'PASS' if passed else 'FAIL'}  "
        f"bias_mean={p1m['p1_mean_abs_lpfc_bias']:.6f}  reef_frac={reef_frac:.3f}",
        flush=True,
    )
    return {
        "arm": arm,
        "seed": seed,
        "train_rule_bias_head": train_bias,
        **p1m,
        "p2_reef_visit_fraction": reef_frac,
        "seed_pass": passed,
    }


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]
    arms = [
        ("ARM_0_frozen_bias", False),
        ("ARM_1_trainable_bias", True),
    ]
    by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, train_bias in arms:
            by_arm[arm_label].append(run_arm_seed(arm_label, train_bias, seed, dry_run))
    frozen_biases = [r["p1_mean_abs_lpfc_bias"] for r in by_arm["ARM_0_frozen_bias"]]
    train_biases = [r["p1_mean_abs_lpfc_bias"] for r in by_arm["ARM_1_trainable_bias"]]
    train_reefs = [r["p2_reef_visit_fraction"] for r in by_arm["ARM_1_trainable_bias"]]
    c1 = all(b < FROZEN_BIAS_MAX for b in frozen_biases)
    c2 = sum(1 for b in train_biases if b >= TRAINABLE_BIAS_MIN) >= MIN_PASS_SEEDS
    c3 = sum(1 for r in train_reefs if REEF_LO < r < REEF_HI) >= MIN_PASS_SEEDS
    return {
        "by_arm": by_arm,
        "acceptance": {
            "C1_frozen_silent": c1,
            "C2_trainable_nonzero": c2,
            "C3_trainable_not_monomodal": c3,
            "pass": c1 and c2 and c3,
        },
    }


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acc = result["acceptance"]
    outcome = "PASS" if acc["pass"] else "FAIL"
    direction = "supports" if outcome == "PASS" else "mixed"
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
        "gates_on_exq": GATES_ON_EXQ,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "executing_hostname": socket.gethostname(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"SD-033a": direction},
        "outcome": outcome,
        "dry_run": dry_run,
        "elapsed_sec": elapsed,
        "acceptance": acc,
        "arm_results": result["by_arm"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
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
    print("\n=== V3-EXQ-598 SUMMARY ===", flush=True)
    print(f"  C1_frozen_silent={acc['C1_frozen_silent']}", flush=True)
    print(f"  C2_trainable_nonzero={acc['C2_trainable_nonzero']}", flush=True)
    print(f"  C3_trainable_not_monomodal={acc['C3_trainable_not_monomodal']}", flush=True)
    print(f"  outcome: {outcome}", flush=True)
    print(f"  gates_on_exq: {GATES_ON_EXQ} (interpret after contributory PASS)", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    return out_path, outcome


if __name__ == "__main__":
    out_path, outcome = main()
    _raw = str(outcome).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
    )
