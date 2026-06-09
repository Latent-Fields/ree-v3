#!/opt/local/bin/python3
"""V3-EXQ-485d: SD-033b OFC trainable-head substrate-readiness diagnostic.

Validates the SD-033b GAP-8 substrate enrichment landed 2026-06-09 (ree-v3 main
382db2c): OFCConfig.train_state_bias_head + OFCAnalog.bias_head_parameters() +
REEConfig.ofc_train_state_bias_head wiring -- the exact OFC mirror of the SD-033a
GAP-D rule_bias_head trainable enrichment (landed 2026-05-17, validated by
V3-EXQ-598b). This is the substrate PREREQUISITE gate for the deferred
trained-OFC-head behavioural arm that closes commitment_closure:GAP-8 (takes
SD-033b from PARTIAL -- 485b/485c representation-level diagnostics PASS,
reviewed 2026-06-04 -- to the full candidate->provisional behavioural
validation). The full behavioural arm (aversive-devaluation behaviour change)
follows as a SEPARATE /queue-experiment session; this run only confirms the
head is trainable and the frozen default stays silent.

EXPERIMENT_PURPOSE = diagnostic (substrate readiness; claim_ids=[]). Does NOT
weight governance confidence. The behavioural arm carries the SD-033b /
MECH-263 evidence.

2-arm frozen-vs-trainable ablation, OFC isolated (gated_policy / dACC /
lateral_pfc all OFF) on the SD-054 bipartite reef/forage env, single varied
factor ofc_train_state_bias_head. ofc_harm_dim>0 so z_harm enters the OFC
state_code (the aversive-devaluation-capable shape the behavioural arm needs --
the OFC reads only z_world + z_harm, no appetitive/drive input, per the GAP-8
substrate constraint). Phased training: P0 encoder warmup -> P1 head training
on the frozen-encoder state_code via E3-gradient REINFORCE (add
list(agent.ofc.bias_head_parameters()) to a P1 Adam optimizer) -> P2 eval.

Pre-registered acceptance:
  C1_frozen_silent  (load-bearing): ARM_0 mean |ofc score_bias| after P1 <
                    FROZEN_BIAS_MAX, all seeds. Default (zeroed-last-Linear)
                    head must remain exactly silent -- the bit-identical-OFF
                    guarantee.
  C2_head_trains    (load-bearing): ARM_1 state_bias_head weight-delta-from-init
                    L2 norm > HEAD_DELTA_MIN in >= MIN_PASS_SEEDS of 3 seeds.
                    This is the UNAMBIGUOUS substrate-readiness signal: the
                    gradient genuinely reached and moved the head. Routed on
                    head-weight movement, NOT the clamp-prone bias magnitude.
  PASS = C1 AND C2.

  C3_bias_nonzero   (informational, NOT a PASS gate): ARM_1 mean |ofc
                    score_bias| after P1 >= TRAINABLE_BIAS_MIN. The
                    behavioural-influence proxy. May read low even when C2
                    passes because compute_bias clamps to +/-ofc_bias_scale
                    (0.1) and the head's own Linear bias terms can push the
                    pre-clamp output past the rail for ~all candidates at
                    random init (zero grad in the saturated region). A low
                    C3 with C2 PASS is a CLAMP-CALIBRATION note for the
                    behavioural arm (consider larger ofc_bias_scale or a
                    pre-clamp training signal), NOT a substrate failure.

Substrate-readiness self-route (P0 positive-control gate): C2 reads a learned
quantity (head movement under gradient). If ARM_1's head receives zero gradient
(delta ~ 0), the substrate is not wired and the only correct route is
substrate_not_ready_requeue -- NEVER a substrate verdict.

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
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_485d_sd033b_ofc_trainable_head_readiness"
EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-485d"
CLAIM_IDS: List[str] = []

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
TRAIN_EPS = P0_EPISODES + P1_EPISODES  # progress denominator M

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
LR_OFC_BIAS = 5e-4
BATCH_SIZE = 32
WF_BUF_MAX = 256
HARM_EVAL_BUF_MAX = 256
N_PROBE_CANDIDATES = 8
RECORD_EVERY_N_STEPS = 4
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

FROZEN_BIAS_MAX = 1e-4
HEAD_DELTA_MIN = 1e-3
TRAINABLE_BIAS_MIN = 0.002
REEF_LO = 0.20
REEF_HI = 0.80
MIN_PASS_SEEDS = 2


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _make_agent(seed: int, train_head: bool) -> Tuple[REEAgent, CausalGridWorldV2]:
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
        # OFC isolated: no gated_policy / dACC / lateral_pfc bias channels.
        use_gated_policy=False,
        use_dacc=False,
        dacc_weight=0.0,
        use_lateral_pfc_analog=False,
        # SP-CEM main-path for first-action candidate diversity.
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # SD-033b OFC analog under test. ofc_harm_dim>0 -> z_harm enters the
        # state_code (aversive-devaluation-capable shape for the behavioural arm).
        use_ofc_analog=True,
        ofc_state_dim=16,
        ofc_harm_dim=HARM_DIM,
        ofc_bias_scale=0.1,
        ofc_train_state_bias_head=train_head,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config)
    return agent, env


def _preflight() -> None:
    frozen, _ = _make_agent(1, train_head=False)
    trainable, _ = _make_agent(2, train_head=True)
    assert frozen.ofc is not None and trainable.ofc is not None
    assert frozen.ofc.config.train_state_bias_head is False
    assert trainable.ofc.config.train_state_bias_head is True
    assert frozen.ofc.config.harm_dim == HARM_DIM
    # Frozen head last Linear must be exactly zeroed; trainable must not be.
    fz = frozen.ofc.state_bias_head[-1]
    tz = trainable.ofc.state_bias_head[-1]
    assert bool(torch.all(fz.weight == 0)) and bool(torch.all(fz.bias == 0))
    assert not (bool(torch.all(tz.weight == 0)) and bool(torch.all(tz.bias == 0)))
    assert len(list(trainable.ofc.bias_head_parameters())) == 4
    del frozen, trainable
    print(
        "Preflight PASS: OFC analog + harm_dim + train_state_bias_head flag "
        "+ bias_head_parameters",
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
    )  # [K, world_dim]
    return {"candidate_features": first_step_world}


def _ofc_reinforce_loss(
    agent: REEAgent, outcome_buf: List[Tuple[Dict, int, float]], baseline: float, device
) -> torch.Tensor:
    """REINFORCE on OFCAnalog.compute_bias -> state_bias_head.

    Gradient path: loss -> log_softmax(-ofc_bias) -> compute_bias() ->
    state_bias_head weights. Parallel to V3-EXQ-598b _lpfc_reinforce_loss.
    """
    if agent.ofc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    idxs = np.random.choice(
        len(outcome_buf), size=min(BATCH_SIZE, len(outcome_buf)), replace=False
    )
    terms: List[torch.Tensor] = []
    for i in idxs:
        snap, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = agent.ofc.compute_bias(snap["candidate_features"])  # [K]
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _encoder_step(agent: REEAgent, env, steps: int, arm: str, ep: int, ep_global: int) -> float:
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
        drive = REEAgent.compute_drive_level(obs_body)
        agent.update_z_goal(
            benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
            drive_level=drive,
        )
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = _action_to_onehot(
                random.randint(0, env.action_dim - 1), env.action_dim, device
            )
            agent._last_action = action
        _, harm, done, _, obs_dict = env.step(action)
        ep_reward += float(harm)
        if z_wp is not None and act_p is not None:
            wf_buf.append((z_wp.cpu(), act_p.cpu(), z_w.cpu()))
            if len(wf_buf) > WF_BUF_MAX:
                wf_buf = wf_buf[-WF_BUF_MAX:]
        he_buf.append(
            (z_w.cpu(), torch.tensor([abs(float(harm)) if float(harm) < 0 else 0.0]))
        )
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
    if ep_global == 1 or ep_global % 10 == 0:
        print(
            f"  [train] {arm} ep {ep_global}/{TRAIN_EPS}  phase=P0  reward={ep_reward:.3f}",
            flush=True,
        )
    return ep_reward


def _head_weight_vector(agent: REEAgent) -> torch.Tensor:
    return torch.cat(
        [p.detach().reshape(-1).cpu() for p in agent.ofc.bias_head_parameters()]
    )


def _p1_train(
    agent: REEAgent, env: CausalGridWorldV2, episodes: int, steps: int, arm: str, train_head: bool
) -> Dict:
    device = agent.device
    bias_opt = (
        optim.Adam(list(agent.ofc.bias_head_parameters()), lr=LR_OFC_BIAS)
        if train_head
        else None
    )
    head_init = _head_weight_vector(agent)
    grad_nonzero_updates = 0
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
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, device
                )
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
            if agent.ofc is not None:
                bias_samples.append(float(agent.ofc._last_bias_abs_mean))
            _, harm, done, _, obs_dict = env.step(action)
            ep_reward += float(harm)
            z_wp, z_sp, act_p = (
                latent.z_world.detach(),
                latent.z_self.detach(),
                action.detach(),
            )
            step += 1
            if done:
                break
        baseline = EMA_DECAY * baseline + (1.0 - EMA_DECAY) * ep_reward
        for snap, sel in ep_buf:
            outcome_buf.append((snap, sel, ep_reward))
        if len(outcome_buf) > OUTCOME_BUF_MAX:
            outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
        if bias_opt is not None:
            l_loss = _ofc_reinforce_loss(agent, outcome_buf, baseline, device)
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                gsum = sum(
                    float(p.grad.abs().sum())
                    for p in agent.ofc.bias_head_parameters()
                    if p.grad is not None
                )
                if gsum > 0:
                    grad_nonzero_updates += 1
                torch.nn.utils.clip_grad_norm_(
                    agent.ofc.bias_head_parameters(), 1.0
                )
                bias_opt.step()
        ep_global = P0_EPISODES + ep + 1
        if ep_global % 10 == 0:
            print(
                f"  [train] {arm} ep {ep_global}/{TRAIN_EPS}  phase=P1"
                f"  bias_abs={bias_samples[-1] if bias_samples else 0:.5f}",
                flush=True,
            )
    head_final = _head_weight_vector(agent)
    head_delta = float(torch.norm(head_final - head_init).item())
    return {
        "p1_mean_abs_ofc_bias": float(np.mean(bias_samples)) if bias_samples else 0.0,
        "p1_max_abs_ofc_bias": float(np.max(bias_samples)) if bias_samples else 0.0,
        "head_weight_delta_norm": head_delta,
        "n_grad_nonzero_updates": grad_nonzero_updates,
    }


def _p2_reef_fraction(
    agent: REEAgent, env: CausalGridWorldV2, episodes: int, steps: int
) -> float:
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


def run_arm_seed(arm: str, train_head: bool, seed: int, dry_run: bool) -> Dict:
    p0 = 3 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    p2 = 2 if dry_run else P2_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    print(f"\nSeed {seed} Condition {arm} ofc_train_state_bias_head={train_head}", flush=True)
    full_config = {
        "arm": arm,
        "train_head": train_head,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "steps": steps,
        "env_kwargs": ENV_KWARGS,
    }
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        agent, env = _make_agent(seed, train_head=train_head)
        for ep in range(p0):
            _encoder_step(agent, env, steps, arm, ep, ep + 1)
        p1m = _p1_train(agent, env, p1, steps, arm, train_head)
        reef_frac = _p2_reef_fraction(agent, env, p2, steps)
        # Verdict: C1 for frozen arm, C2 (head moved) for trainable arm.
        if not train_head:
            passed = p1m["p1_mean_abs_ofc_bias"] < FROZEN_BIAS_MAX
        else:
            passed = p1m["head_weight_delta_norm"] > HEAD_DELTA_MIN
        row = {
            "arm": arm,
            "seed": seed,
            "ofc_train_state_bias_head": train_head,
            **p1m,
            "p2_reef_visit_fraction": reef_frac,
            "seed_pass": passed,
        }
        cell.stamp(row)
    print(
        f"verdict: {'PASS' if passed else 'FAIL'}  "
        f"bias_mean={p1m['p1_mean_abs_ofc_bias']:.6f}  "
        f"head_delta={p1m['head_weight_delta_norm']:.6f}  reef_frac={reef_frac:.3f}",
        flush=True,
    )
    return row


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]
    arms = [
        ("ARM_0_frozen_head", False),
        ("ARM_1_trainable_head", True),
    ]
    by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, train_head in arms:
            by_arm[arm_label].append(run_arm_seed(arm_label, train_head, seed, dry_run))

    frozen_biases = [r["p1_mean_abs_ofc_bias"] for r in by_arm["ARM_0_frozen_head"]]
    train_deltas = [r["head_weight_delta_norm"] for r in by_arm["ARM_1_trainable_head"]]
    train_biases = [r["p1_mean_abs_ofc_bias"] for r in by_arm["ARM_1_trainable_head"]]
    train_grad_updates = [r["n_grad_nonzero_updates"] for r in by_arm["ARM_1_trainable_head"]]

    c1 = all(b < FROZEN_BIAS_MAX for b in frozen_biases)
    c2 = sum(1 for d in train_deltas if d > HEAD_DELTA_MIN) >= MIN_PASS_SEEDS
    c3 = sum(1 for b in train_biases if b >= TRAINABLE_BIAS_MIN) >= MIN_PASS_SEEDS

    # Substrate-readiness positive control: did the gradient reach the head at
    # all on the trainable arm? Same statistic the load-bearing C2 routes on
    # (head movement). Below-floor -> substrate not wired -> requeue, NOT a
    # substrate verdict.
    max_train_delta = max(train_deltas) if train_deltas else 0.0
    any_grad_flowed = any(g > 0 for g in train_grad_updates)
    head_grad_ready = max_train_delta > HEAD_DELTA_MIN and any_grad_flowed

    return {
        "by_arm": by_arm,
        "max_train_head_delta": max_train_delta,
        "any_grad_flowed": any_grad_flowed,
        "head_grad_ready": head_grad_ready,
        "acceptance": {
            "C1_frozen_silent": c1,
            "C2_head_trains": c2,
            "C3_bias_nonzero_informational": c3,
            "pass": c1 and c2,
        },
    }


def _interpretation(result: Dict) -> Dict:
    acc = result["acceptance"]
    ready = bool(result["head_grad_ready"])
    if not ready:
        label = "substrate_not_ready_requeue"
    elif acc["pass"]:
        label = (
            "ofc_trainable_head_substrate_ready"
            if acc["C3_bias_nonzero_informational"]
            else "ofc_trainable_head_substrate_ready_bias_clamp_saturated"
        )
    else:
        label = "ofc_trainable_head_not_substrate_ready"
    # C2 reads head movement (a learned/measured quantity) -> P0/training
    # positive-control readiness precondition, SAME statistic C2 routes on.
    preconditions = [
        {
            "name": "ofc_head_weight_delta_supra_floor",
            "description": (
                "ARM_1 state_bias_head weight-delta-from-init L2 norm clears "
                "HEAD_DELTA_MIN -- the gradient reached and moved the head "
                "(positive control: trainable head trained via E3 REINFORCE in P1)"
            ),
            "measured": float(result["max_train_head_delta"]),
            "threshold": HEAD_DELTA_MIN,
            "control": "trainable-arm head trained on frozen-encoder state_code in P1",
            "met": ready,
        }
    ]
    criteria_non_degenerate = {
        # C1 non-degenerate: frozen head genuinely zeroed (exactly 0), not a
        # coincidental small value -- the bit-identical-OFF guarantee.
        "C1": bool(acc["C1_frozen_silent"]),
        # C2 non-degenerate: head movement is real (gradient flowed) and the
        # arms differ structurally (trainable head un-zeroed vs frozen zeroed).
        "C2": bool(result["any_grad_flowed"]),
    }
    criteria = [
        {"name": "C1_frozen_silent", "load_bearing": True, "passed": bool(acc["C1_frozen_silent"])},
        {"name": "C2_head_trains", "load_bearing": True, "passed": bool(acc["C2_head_trains"])},
        {
            "name": "C3_bias_nonzero_informational",
            "load_bearing": False,
            "passed": bool(acc["C3_bias_nonzero_informational"]),
        },
    ]
    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": criteria,
    }


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acc = result["acceptance"]
    interpretation = _interpretation(result)
    # Substrate not ready -> the run did not validate; treat as FAIL outcome but
    # the self-route label routes it to requeue, not a substrate verdict.
    outcome = "PASS" if (acc["pass"] and result["head_grad_ready"]) else "FAIL"
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
        "executing_hostname": socket.gethostname(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "dry_run": dry_run,
        "elapsed_sec": elapsed,
        "acceptance": acc,
        "interpretation": interpretation,
        "max_train_head_delta": result["max_train_head_delta"],
        "head_grad_ready": result["head_grad_ready"],
        "arm_results": [r for arm in result["by_arm"].values() for r in arm],
        "substrate_under_test": "SD-033b train_state_bias_head (ree-v3 main 382db2c)",
        "unblocks": "commitment_closure:GAP-8 trained-OFC-head behavioural arm",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    t0 = time.time()
    _preflight()
    result = run(seeds=args.seeds, dry_run=args.dry_run)
    elapsed = time.time() - t0
    out_path, outcome = write_manifest(result, args.dry_run, elapsed)
    acc = result["acceptance"]
    print("\n=== V3-EXQ-485d SUMMARY ===", flush=True)
    print(f"  C1_frozen_silent={acc['C1_frozen_silent']}", flush=True)
    print(f"  C2_head_trains={acc['C2_head_trains']}", flush=True)
    print(f"  C3_bias_nonzero_informational={acc['C3_bias_nonzero_informational']}", flush=True)
    print(f"  head_grad_ready={result['head_grad_ready']} (max_delta={result['max_train_head_delta']:.6f})", flush=True)
    print(f"  interpretation: {_interpretation(result)['label']}", flush=True)
    print(f"  outcome: {outcome}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    return out_path, outcome


if __name__ == "__main__":
    out_path, outcome = main()
    _raw = str(outcome).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
    )
