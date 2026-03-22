"""
V3-EXQ-030c -- SD-003 Full Attribution: Large-Scale (n_hazards=8, warmup=2000, eval=200)

Claims: SD-003, ARC-024, MECH-071

Root cause of EXQ-030b seed-1 sign switch:
    n_agent_hazard_eval ≈ 32. With per-sample causal_sig std≈0.05,
    SE = 0.05/sqrt(32) ≈ 0.009. Seed-0 mean=+0.017 and seed-1 mean=-0.008
    are both within 2 SE of zero -- a statistical power failure, not a
    model failure. The sign structure cannot be distinguished from noise at N=32.

    EXQ-030c fixes this by scaling up:
      - n_hazards: 4 -> 8 (doubles contact rate from ~0.003 to ~0.006/step)
      - warmup: 500 -> 2000 episodes (E2 and E3 better calibrated on contact states)
      - eval: 50 -> 200 episodes (N ≈ 200+ contact events per type)
      - steps: 200 -> 300 (more time per episode near hazards)
    Target: n_agent_hazard_eval >= 100, n_env_hazard_eval >= 100.
    At N=100, SE ≈ 0.005 -- agent_caused mean of 0.010 would be 2 SE above zero.

Architecture basis:
    SD-003 (counterfactual self-attribution)
    ARC-024 (gradient world enables approach detection)
    MECH-071 (E3 harm_eval calibration gradient asymmetry)
    MECH-069 (three incommensurable optimizers)
    Fix 2 from EXQ-030b: E3 trained on observed + E2-predicted states

PASS criteria (same core as EXQ-030b, new N requirements):
    C1: attribution_gap > 0       (approach causal_sig > env_caused causal_sig)
    C2: causal_sig_approach > causal_sig_none
    C3: causal_sig_approach > 0.005
    C4: world_forward_r2 > 0.05
    C5: n_agent_hazard_eval >= 100 (NEW -- enough data for reliable contact stats)

Diagnostic (not PASS/FAIL):
    sign_structure_correct = (causal_sig_agent_caused > causal_sig_env_caused)
    This is what seed-0 showed and seed-1 could not resolve. EXQ-030c
    has the power to determine whether the sign structure is genuine.
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


EXPERIMENT_TYPE = "v3_exq_030c_sd003_attribution_largescale"
CLAIM_IDS = ["SD-003", "ARC-024", "MECH-071"]


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
        pred_all  = agent.e2.world_forward(zw_all, a_all)
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
    Random-policy warmup. E3 trained on observed + E2-predicted states (Fix 2).
    """
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 4000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 8000

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

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

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

            # E3 harm_eval: observed + E2-predicted (Fix 2 from EXQ-030b)
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

        if (ep + 1) % 200 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}  "
                f"pos_buf={len(harm_buf_pos)}  neg_buf={len(harm_buf_neg)}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


def _eval_attribution(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Full SD-003 attribution eval with mean + median per ttype.
    Tracks all four transition types; reports sign_structure_correct as diagnostic.
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
                z_world_actual = agent.e2.world_forward(z_world, action)
                harm_actual    = agent.e3.harm_eval(z_world_actual)

                sigs = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf    = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_cf    = agent.e2.world_forward(z_world, a_cf)
                    harm_cf = agent.e3.harm_eval(z_cf)
                    sigs.append(float((harm_actual - harm_cf).item()))

                mean_sig = float(np.mean(sigs)) if sigs else 0.0

            if ttype in causal_sigs:
                causal_sigs[ttype].append(mean_sig)

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    def _median(lst):
        return float(np.median(lst)) if lst else 0.0

    def _std(lst):
        return float(np.std(lst)) if len(lst) > 1 else 0.0

    mean_sigs   = {t: _mean(causal_sigs[t])   for t in ttypes}
    median_sigs = {t: _median(causal_sigs[t]) for t in ttypes}
    std_sigs    = {t: _std(causal_sigs[t])    for t in ttypes}
    n_counts    = {t: len(causal_sigs[t])     for t in ttypes}

    attribution_gap = mean_sigs["hazard_approach"] - mean_sigs["env_caused_hazard"]
    sign_correct    = mean_sigs["agent_caused_hazard"] > mean_sigs["env_caused_hazard"]

    print(f"\n  --- SD-003 Attribution Eval (EXQ-030c) ---", flush=True)
    print(f"  {'ttype':28s}  {'mean':>10}  {'median':>10}  {'std':>10}  {'n':>6}", flush=True)
    for t in ttypes:
        print(
            f"  {t:28s}  {mean_sigs[t]:>10.6f}  {median_sigs[t]:>10.6f}"
            f"  {std_sigs[t]:>10.6f}  {n_counts[t]:>6}",
            flush=True,
        )
    print(f"  attribution_gap (approach - env_caused): {attribution_gap:.6f}", flush=True)
    print(f"  sign_structure_correct (agent > env): {sign_correct}", flush=True)

    return {
        "mean_sigs":       mean_sigs,
        "median_sigs":     median_sigs,
        "std_sigs":        std_sigs,
        "n_counts":        n_counts,
        "attribution_gap": attribution_gap,
        "sign_correct":    sign_correct,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 2000,
    eval_episodes: int = 200,
    steps_per_episode: int = 300,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=8, num_resources=5,
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
        f"[V3-EXQ-030c] SD-003 Full Attribution -- Large Scale\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  num_hazards=8 (vs 4 in EXQ-030b)  warmup={warmup_episodes}  eval={eval_episodes}\n"
        f"  alpha_world={alpha_world}  proximity_scale={proximity_scale}\n"
        f"  Goal: N >= 100 contact events per type for reliable sign-structure test",
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

    print(f"\n[V3-EXQ-030c] Training ({warmup_episodes} eps)...", flush=True)
    train_out = _train(
        agent, env, optimizer, harm_eval_optimizer, world_forward_optimizer,
        warmup_episodes, steps_per_episode,
    )

    wf_r2 = _compute_world_forward_r2(agent, train_out["wf_data"])

    print(f"\n[V3-EXQ-030c] Eval ({eval_episodes} eps)...", flush=True)
    eval_out = _eval_attribution(agent, env, eval_episodes, steps_per_episode)

    ms = eval_out["mean_sigs"]
    md = eval_out["median_sigs"]
    sd = eval_out["std_sigs"]
    nc = eval_out["n_counts"]

    c1_pass = eval_out["attribution_gap"] > 0.0
    c2_pass = ms["hazard_approach"] > ms["none"]
    c3_pass = ms["hazard_approach"] > 0.005
    c4_pass = wf_r2 > 0.05
    c5_pass = nc["agent_caused_hazard"] >= 100

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
            f"C2 FAIL: causal_sig_approach={ms['hazard_approach']:.6f} <= causal_sig_none={ms['none']:.6f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: causal_sig_approach={ms['hazard_approach']:.6f} <= 0.005"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_agent_hazard_eval={nc['agent_caused_hazard']} < 100 (insufficient data for sign test)"
        )

    print(f"\nV3-EXQ-030c verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(f"  Diagnostic -- sign_structure_correct: {eval_out['sign_correct']}", flush=True)

    tc = train_out["counts"]
    metrics = {
        "alpha_world":                       float(alpha_world),
        "proximity_scale":                   float(proximity_scale),
        "world_forward_r2":                  float(wf_r2),
        # Mean causal_sig per ttype
        "mean_causal_sig_none":              float(ms["none"]),
        "mean_causal_sig_approach":          float(ms["hazard_approach"]),
        "mean_causal_sig_env_hazard":        float(ms["env_caused_hazard"]),
        "mean_causal_sig_agent_hazard":      float(ms["agent_caused_hazard"]),
        # Median causal_sig per ttype
        "median_causal_sig_none":            float(md["none"]),
        "median_causal_sig_approach":        float(md["hazard_approach"]),
        "median_causal_sig_env_hazard":      float(md["env_caused_hazard"]),
        "median_causal_sig_agent_hazard":    float(md["agent_caused_hazard"]),
        # Std
        "std_causal_sig_agent_hazard":       float(sd["agent_caused_hazard"]),
        "std_causal_sig_env_hazard":         float(sd["env_caused_hazard"]),
        # Attribution gap and sign diagnostic
        "attribution_gap":                   float(eval_out["attribution_gap"]),
        "sign_structure_correct":            float(1.0 if eval_out["sign_correct"] else 0.0),
        # N counts
        "n_approach_eval":                   float(nc["hazard_approach"]),
        "n_env_hazard_eval":                 float(nc["env_caused_hazard"]),
        "n_agent_hazard_eval":               float(nc["agent_caused_hazard"]),
        "n_none_eval":                       float(nc["none"]),
        # Training counts
        "train_approach_events":             float(tc.get("hazard_approach", 0)),
        "train_contact_events":              float(tc.get("env_caused_hazard", 0) + tc.get("agent_caused_hazard", 0)),
        # Criteria
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

    summary_markdown = f"""# V3-EXQ-030c -- SD-003 Full Attribution (Large Scale)

**Status:** {status}
**Claims:** SD-003, ARC-024, MECH-071
**World:** CausalGridWorldV2 (n_hazards=8, size=12, proximity_scale={proximity_scale})
**Warmup:** {warmup_episodes} eps | **Eval:** {eval_episodes} eps x {steps_per_episode} steps
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}

## Motivation: Resolving EXQ-030b Sign Switch

EXQ-030b seed-0 PASS (agent_caused=+0.017), seed-1 FAIL (agent_caused=-0.008).
Root cause: n_agent_caused_eval ≈ 32 -> SE ≈ 0.009. Both means within 2 SE of zero.
EXQ-030c scales up to N ≥ 100 contact events per type:
- n_hazards: 4 -> 8 (doubles contact frequency)
- warmup: 500 -> 2000 (better-calibrated E2 and E3 on contact states)
- eval: 50 -> 200 episodes (more data)
- steps: 200 -> 300 (more exposure per episode)

## Attribution Results

| Transition | mean causal_sig | median | std | n |
|---|---|---|---|---|
| none (locomotion)    | {ms['none']:.6f} | {md['none']:.6f} | {sd['none']:.6f} | {nc['none']} |
| hazard_approach      | {ms['hazard_approach']:.6f} | {md['hazard_approach']:.6f} | {sd['hazard_approach']:.6f} | {nc['hazard_approach']} |
| env_caused_hazard    | {ms['env_caused_hazard']:.6f} | {md['env_caused_hazard']:.6f} | {sd['env_caused_hazard']:.6f} | {nc['env_caused_hazard']} |
| agent_caused_hazard  | {ms['agent_caused_hazard']:.6f} | {md['agent_caused_hazard']:.6f} | {sd['agent_caused_hazard']:.6f} | {nc['agent_caused_hazard']} |

- **world_forward R2**: {wf_r2:.4f}
- **attribution_gap** (approach − env_caused): {eval_out['attribution_gap']:.6f}
- **sign_structure_correct** (agent > env): {eval_out['sign_correct']}  *(diagnostic -- not a pass criterion)*

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: attribution_gap > 0 | {"PASS" if c1_pass else "FAIL"} | {eval_out['attribution_gap']:.6f} |
| C2: causal_sig_approach > causal_sig_none | {"PASS" if c2_pass else "FAIL"} | {ms['hazard_approach']:.6f} vs {ms['none']:.6f} |
| C3: causal_sig_approach > 0.005 | {"PASS" if c3_pass else "FAIL"} | {ms['hazard_approach']:.6f} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} | {wf_r2:.4f} |
| C5: n_agent_hazard_eval >= 100 (reliable contact stats) | {"PASS" if c5_pass else "FAIL"} | {nc['agent_caused_hazard']} |

Criteria met: {n_met}/5 -> **{status}**
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
    parser.add_argument("--warmup",          type=int,   default=2000)
    parser.add_argument("--eval-eps",        type=int,   default=200)
    parser.add_argument("--steps",           type=int,   default=300)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
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
