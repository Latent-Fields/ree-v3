"""
V3-EXQ-036 — SD-003 Multi-Step Attribution on CausalGridWorldV2 (k=5 rollout)

Claims: SD-003, ARC-024, MECH-071

Root cause of EXQ-030b C3 FAIL (causal_sig_approach=0.003149 < 0.005):

    With 1-step E2.world_forward, action-conditional predictions are nearly identical:
    The agent moves 1 cell in a 12x12 grid per step. The hazard_field_view (proximity
    channels in z_world) changes only slightly for any single-step move. So
    E3(E2(z_world, a_actual)) ≈ E3(E2(z_world, a_cf)) for almost all timesteps —
    the causal signature is diluted to noise.

Multi-step rollout fix (k=5):

    After 5 steps from the same starting z_world, different first-action choices lead
    to meaningfully divergent world states. The causal footprint of the initial choice
    propagates and amplifies across the rollout.

    Protocol: at each eval step with k_rollout=5:
      - Sample k-1 shared random actions: [a_rand_1, ..., a_rand_{k-1}]
        (same for actual and all counterfactuals — only a_0 differs)
      - actual_rollout: z_w → E2(z_w, a_actual) → E2(·, a_rand_1) → ... → z_w_actual(k)
      - cf_rollout for a_cf: z_w → E2(z_w, a_cf) → E2(·, a_rand_1) → ... → z_w_cf(k)
      - causal_sig = E3(z_w_actual(k)) - mean_cf E3(z_w_cf(k))

    Using the SAME shared random tail makes the comparison fair: only a_0 differs.
    The k-step divergence compounds hazard proximity differences across 5 moves.

Training: identical to EXQ-030b (Fix 2: E3 trained on observed + E2-predicted states).
World: identical to EXQ-030b (CausalGridWorldV2, size=12, num_hazards=4).

PASS criteria (C3 threshold raised from 0.005 to 0.01 — 5-step rollout amplifies signal):
    C1: attribution_gap > 0  (approach attribution > env-caused attribution)
    C2: causal_sig_approach > causal_sig_none  (hazard approach more attributable)
    C3: causal_sig_approach > 0.01  (minimum detectable signal with 5x amplification)
    C4: world_forward_r2 > 0.05    (E2 functional)
    C5: n_approach_eval >= 50      (sufficient approach events)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_036_sd003_multistep_attribution"
CLAIM_IDS = ["SD-003", "ARC-024", "MECH-071"]
K_ROLLOUT = 5  # multi-step rollout horizon for causal_sig computation


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_world_forward_r2(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all  = torch.cat([d[0] for d in wf_data], dim=0)
        a_all   = torch.cat([d[1] for d in wf_data], dim=0)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0)
        pred_all = agent.e2.world_forward(zw_all, a_all)
        pred_test = pred_all[n_train:]
        tgt_test  = zw1_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R2 (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return r2


def _train(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    world_forward_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Training identical to EXQ-030b Fix 2:
    E3 harm_eval trained on both observed z_world AND E2-predicted z_world states.
    """
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    counts: Dict[str, int] = {}
    num_actions = env.action_dim

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            counts[ttype] = counts.get(ttype, 0) + 1

            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # Standard E1 + E2 self losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E2.world_forward training (separate optimizer — MECH-069)
            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred, zw1_b)
                if wf_loss.requires_grad:
                    world_forward_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    world_forward_optimizer.step()

            # E3 harm_eval: Fix 2 — observed + E2-predicted states
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                zw_pos_obs = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg_obs = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)

                with torch.no_grad():
                    a_rand_pos = torch.zeros(k_pos, num_actions, device=agent.device)
                    a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
                    a_rand_neg = torch.zeros(k_neg, num_actions, device=agent.device)
                    a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0
                    zw_pos_pred = agent.e2.world_forward(zw_pos_obs.to(agent.device), a_rand_pos)
                    zw_neg_pred = agent.e2.world_forward(zw_neg_obs.to(agent.device), a_rand_neg)

                zw_b = torch.cat([
                    zw_pos_obs.to(agent.device),
                    zw_neg_obs.to(agent.device),
                    zw_pos_pred,
                    zw_neg_pred,
                ], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}  wf_buf={len(wf_data)}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


def _multistep_rollout(
    agent: REEAgent,
    z_world_0: torch.Tensor,
    a_0: torch.Tensor,
    shared_tail: List[torch.Tensor],
) -> torch.Tensor:
    """
    Roll out k_rollout steps from z_world_0.
    First action is a_0; remaining are from shared_tail.
    Returns z_world after k_rollout steps.
    """
    z_w = z_world_0
    actions = [a_0] + shared_tail
    for a in actions:
        z_w = agent.e2.world_forward(z_w, a)
    return z_w


def _eval_multistep_attribution(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    k_rollout: int = K_ROLLOUT,
) -> Dict:
    """
    Multi-step SD-003 attribution eval.

    For each step:
      1. z_world_t (perspective-corrected)
      2. Sample k_rollout-1 shared random actions (same for actual and all cf)
      3. actual_rollout: E2^k(z_world, [a_actual, shared_tail...]) → z_w_actual(k)
      4. cf_rollout for each a_cf: E2^k(z_world, [a_cf, shared_tail...]) → z_w_cf(k)
      5. causal_sig = E3(z_w_actual(k)) - mean_cf E3(z_w_cf(k))

    The shared random tail ensures only a_0 differs across comparisons.
    """
    agent.eval()

    ttypes = ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]
    causal_sigs: Dict[str, List[float]] = {t: [] for t in ttypes}

    num_actions = env.action_dim

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                # Shared random tail actions (same for actual and all counterfactuals)
                shared_tail = [
                    _action_to_onehot(
                        random.randint(0, num_actions - 1),
                        num_actions,
                        agent.device,
                    )
                    for _ in range(k_rollout - 1)
                ]

                # Actual k-step rollout
                z_w_actual_k = _multistep_rollout(agent, z_world, action, shared_tail)
                harm_actual_k = float(agent.e3.harm_eval(z_w_actual_k).item())

                # Counterfactual k-step rollouts
                cf_harms = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_w_cf_k = _multistep_rollout(agent, z_world, a_cf, shared_tail)
                    cf_harms.append(float(agent.e3.harm_eval(z_w_cf_k).item()))

                mean_cf = float(np.mean(cf_harms)) if cf_harms else harm_actual_k
                causal_sig = harm_actual_k - mean_cf

            if ttype in causal_sigs:
                causal_sigs[ttype].append(causal_sig)

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    mean_sigs = {t: _mean(causal_sigs[t]) for t in ttypes}
    n_counts  = {t: len(causal_sigs[t]) for t in ttypes}

    attribution_gap = mean_sigs["hazard_approach"] - mean_sigs["env_caused_hazard"]

    print(f"\n  --- SD-003 Multi-Step Attribution Eval (EXQ-036, k={k_rollout}) ---", flush=True)
    for t in ttypes:
        print(
            f"  {t:28s}: causal_sig={mean_sigs[t]:.6f}  n={n_counts[t]}",
            flush=True,
        )
    print(f"  attribution_gap (approach-env): {attribution_gap:.6f}", flush=True)

    return {
        "mean_sigs":       mean_sigs,
        "n_counts":        n_counts,
        "attribution_gap": attribution_gap,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    k_rollout: int = K_ROLLOUT,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    print(
        f"[V3-EXQ-036] SD-003 Multi-Step Attribution (k={k_rollout}) on CausalGridWorldV2\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  alpha_world={alpha_world}  proximity_scale={proximity_scale}\n"
        f"  Fix: {k_rollout}-step rollout amplifies action-conditional divergence\n"
        f"  Training: EXQ-030b Fix 2 (E3 on observed + E2-predicted states)",
        flush=True,
    )

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    world_forward_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer               = optim.Adam(standard_params,      lr=lr)
    world_forward_optimizer = optim.Adam(world_forward_params, lr=1e-3)
    harm_eval_optimizer     = optim.Adam(harm_eval_params,     lr=1e-4)

    train_out = _train(
        agent, env, optimizer, harm_eval_optimizer, world_forward_optimizer,
        warmup_episodes, steps_per_episode,
    )

    wf_r2 = _compute_world_forward_r2(agent, train_out["wf_data"])

    print(f"\n[V3-EXQ-036] Eval ({eval_episodes} eps, k={k_rollout}-step rollout)...", flush=True)
    eval_out = _eval_multistep_attribution(
        agent, env, eval_episodes, steps_per_episode, k_rollout,
    )

    ms = eval_out["mean_sigs"]
    nc = eval_out["n_counts"]

    c1_pass = eval_out["attribution_gap"] > 0.0
    c2_pass = ms["hazard_approach"] > ms["none"]
    c3_pass = ms["hazard_approach"] > 0.01
    c4_pass = wf_r2 > 0.05
    c5_pass = nc["hazard_approach"] >= 50

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: attribution_gap={eval_out['attribution_gap']:.6f} <= 0"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: causal_sig_approach({ms['hazard_approach']:.6f}) <= causal_sig_none({ms['none']:.6f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: causal_sig_approach={ms['hazard_approach']:.6f} <= 0.01"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: n_approach={nc['hazard_approach']} < 50")

    print(f"\nV3-EXQ-036 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    tc = train_out["counts"]
    metrics = {
        "alpha_world":                   float(alpha_world),
        "proximity_scale":               float(proximity_scale),
        "k_rollout":                     float(k_rollout),
        "world_forward_r2":              float(wf_r2),
        "mean_causal_sig_none":          float(ms["none"]),
        "mean_causal_sig_approach":      float(ms["hazard_approach"]),
        "mean_causal_sig_env_hazard":    float(ms["env_caused_hazard"]),
        "mean_causal_sig_agent_hazard":  float(ms["agent_caused_hazard"]),
        "attribution_gap":               float(eval_out["attribution_gap"]),
        "n_approach_eval":               float(nc["hazard_approach"]),
        "n_env_hazard_eval":             float(nc["env_caused_hazard"]),
        "n_agent_hazard_eval":           float(nc["agent_caused_hazard"]),
        "n_none_eval":                   float(nc["none"]),
        "train_approach_events":         float(tc.get("hazard_approach", 0)),
        "train_contact_events":          float(tc.get("env_caused_hazard", 0) + tc.get("agent_caused_hazard", 0)),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(n_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-036 — SD-003 Multi-Step Attribution (k={k_rollout}) on CausalGridWorldV2

**Status:** {status}
**Claims:** SD-003, ARC-024, MECH-071
**World:** CausalGridWorldV2 (proximity_scale={proximity_scale})
**alpha_world:** {alpha_world}  (SD-008)
**k_rollout:** {k_rollout}
**Seed:** {seed}
**Predecessor:** EXQ-030b (4/5 FAIL — causal_sig_approach=0.003149 < 0.005)

## Root Cause Analysis and Fix

**EXQ-030b C3 FAIL — 1-step causal_sig too small:**
Single-step E2.world_forward produces nearly identical next-states for all actions.
Agent moves 1 cell/step in 12x12 grid → hazard_field_view changes ~1/12 of full range.
E3(E2(z, a_actual)) ≈ E3(E2(z, a_cf)) → causal_sig ≈ noise.

**Fix: k={k_rollout}-step rollout with shared random tail:**
Only the first action (a_0) differs. Remaining k-1 actions are sampled identically
for actual and all counterfactuals. After {k_rollout} steps, different a_0 choices have
propagated their position difference through E2's world model, amplifying proximity gaps.

## Attribution Results (k={k_rollout}-step)

| Transition Type | causal_sig | n |
|---|---|---|
| none (locomotion)    | {ms['none']:.6f} | {nc['none']} |
| hazard_approach      | {ms['hazard_approach']:.6f} | {nc['hazard_approach']} |
| env_caused_hazard    | {ms['env_caused_hazard']:.6f} | {nc['env_caused_hazard']} |
| agent_caused_hazard  | {ms['agent_caused_hazard']:.6f} | {nc['agent_caused_hazard']} |

- **world_forward R2**: {wf_r2:.4f}
- **attribution_gap** (approach − env): {eval_out['attribution_gap']:.6f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: attribution_gap > 0 | {"PASS" if c1_pass else "FAIL"} | {eval_out['attribution_gap']:.6f} |
| C2: causal_sig_approach > causal_sig_none | {"PASS" if c2_pass else "FAIL"} | {ms['hazard_approach']:.6f} vs {ms['none']:.6f} |
| C3: causal_sig_approach > 0.01 (minimum k-step signal) | {"PASS" if c3_pass else "FAIL"} | {ms['hazard_approach']:.6f} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} | {wf_r2:.4f} |
| C5: n_approach >= 50 | {"PASS" if c5_pass else "FAIL"} | {nc['hazard_approach']} |

Criteria met: {n_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--warmup",          type=int,   default=500)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--k-rollout",       type=int,   default=K_ROLLOUT)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
        k_rollout=args.k_rollout,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"]  = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
