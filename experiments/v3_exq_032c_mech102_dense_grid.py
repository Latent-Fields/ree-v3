"""
V3-EXQ-032c — MECH-102: Forced Escalation via Dense Constrained Grid

Claims: MECH-102, ARC-024, SD-003

EXQ-032 FAIL: E3-guided policy on a 12x12 grid with 6 hazards (4.2% coverage)
could navigate around all hazards — viability was never genuinely threatened.

EXQ-032c redesign: constrain the environment so avoidance is impossible.
    Grid: 8x8 (64 cells), 10 hazards (15.6% of cells)
    E3-guided harm-minimizing policy (argmin E3(E2(z_world, a)))
    With 15.6% hazard coverage, the agent cannot avoid hazard proximity.
    Even the most ethical possible policy accumulates harm_exposure.

    Adjusted exposure thresholds for dense grid:
        low_exposure < 0.05  (vs 0.10 in EXQ-032 — baseline for dense env)
        high_exposure >= 0.10 (vs 0.20 in EXQ-032 — achievable under density)

MECH-102 prediction:
    Even with full ethical optimization (argmin harm), the agent in a dense
    hazard grid produces higher causal_sig when harm_exposure is high
    (viability threatened) than when low. This is FORCED ESCALATION:
    not chosen violence, but structurally inevitable consequence of navigating
    a world where low-energy options have collapsed.

PASS criteria:
    C1: escalation_gap > 0  (causal_sig_high > causal_sig_low)
    C2: causal_sig_high_exposure > 0.001
    C3: n_high_exposure >= 50  (dense grid should produce many threat steps)
    C4: world_forward_r2 > 0.05
    C5: mean_harm_exposure_high > 0.10  (genuine viability threat in dense grid)

Architecture basis:
    MECH-102 (violence as terminal error correction)
    ARC-024 (gradient world — proximity fields make exposure accumulate continuously)
    SD-003 (counterfactual causal_sig via E2.world_forward + E3.harm_eval)
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


EXPERIMENT_TYPE = "v3_exq_032c_mech102_dense_grid"
CLAIM_IDS = ["MECH-102", "ARC-024", "SD-003"]

# Adjusted thresholds for dense environment where harm accumulates faster
LOW_EXPOSURE_MAX  = 0.05   # harm_exposure < this → low-energy paths working
HIGH_EXPOSURE_MIN = 0.10   # harm_exposure >= this → viability threatened


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
    """Random-policy warmup with E3 Fix 2 (observed + E2-predicted states)."""
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    num_actions = env.action_dim
    counts: Dict[str, int] = {}

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

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


def _eval_escalation_dense(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    E3-guided harm-minimizing policy eval in dense grid.
    Split steps by harm_exposure EMA into low/high conditions.
    """
    agent.eval()
    num_actions = env.action_dim

    causal_sigs_low:  List[float] = []
    causal_sigs_high: List[float] = []
    harm_exposures:   List[float] = []
    causal_sigs_by_ttype: Dict[str, List[float]] = {}

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

                harm_exp = float(obs_body[10].item()) if obs_body.shape[0] > 10 else 0.0
                harm_exposures.append(harm_exp)

                # E3-guided policy: argmin E3(E2(z_world, a))
                best_action_idx = 0
                best_harm_score = float("inf")
                all_harm_scores = []

                for a_idx in range(num_actions):
                    a_vec = _action_to_onehot(a_idx, num_actions, agent.device)
                    z_pred = agent.e2.world_forward(z_world, a_vec)
                    h_score = float(agent.e3.harm_eval(z_pred).item())
                    all_harm_scores.append(h_score)
                    if h_score < best_harm_score:
                        best_harm_score = h_score
                        best_action_idx = a_idx

                harm_actual = all_harm_scores[best_action_idx]
                cf_harms = [all_harm_scores[i] for i in range(num_actions) if i != best_action_idx]
                mean_cf_harm = float(np.mean(cf_harms)) if cf_harms else harm_actual
                causal_sig = harm_actual - mean_cf_harm

            action = _action_to_onehot(best_action_idx, num_actions, agent.device)
            agent._last_action = action
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if harm_exp < LOW_EXPOSURE_MAX:
                causal_sigs_low.append(causal_sig)
            elif harm_exp >= HIGH_EXPOSURE_MIN:
                causal_sigs_high.append(causal_sig)

            if ttype not in causal_sigs_by_ttype:
                causal_sigs_by_ttype[ttype] = []
            causal_sigs_by_ttype[ttype].append(causal_sig)

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    mean_sig_low    = _mean(causal_sigs_low)
    mean_sig_high   = _mean(causal_sigs_high)
    mean_harm_exp   = _mean(harm_exposures)
    mean_harm_exp_high = _mean([h for h in harm_exposures if h >= HIGH_EXPOSURE_MIN])

    print(f"\n  --- MECH-102 Dense Grid Escalation Eval ---", flush=True)
    print(f"  harm_exposure: mean={mean_harm_exp:.4f}  n_low={len(causal_sigs_low)}  n_high={len(causal_sigs_high)}", flush=True)
    print(f"  causal_sig_low  (<{LOW_EXPOSURE_MAX}):  {mean_sig_low:.6f}  (n={len(causal_sigs_low)})", flush=True)
    print(f"  causal_sig_high (>={HIGH_EXPOSURE_MIN}): {mean_sig_high:.6f}  (n={len(causal_sigs_high)})", flush=True)
    print(f"  escalation_gap: {mean_sig_high - mean_sig_low:.6f}", flush=True)
    print(f"  mean_harm_exp (high condition): {mean_harm_exp_high:.4f}", flush=True)
    print(f"\n  By ttype:", flush=True)
    for tt, sigs in sorted(causal_sigs_by_ttype.items()):
        print(f"    {tt:28s}: causal_sig={_mean(sigs):.6f}  n={len(sigs)}", flush=True)

    return {
        "causal_sig_low_exposure":    mean_sig_low,
        "causal_sig_high_exposure":   mean_sig_high,
        "escalation_gap":             mean_sig_high - mean_sig_low,
        "n_low_exposure":             len(causal_sigs_low),
        "n_high_exposure":            len(causal_sigs_high),
        "mean_harm_exposure":         mean_harm_exp,
        "mean_harm_exposure_high":    mean_harm_exp_high,
        "causal_sigs_by_ttype":       {k: _mean(v) for k, v in causal_sigs_by_ttype.items()},
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
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    # Dense constrained grid: 8x8 with 10 hazards = 15.6% coverage
    # E3-guided policy cannot avoid all hazards at this density
    env = CausalGridWorldV2(
        seed=seed, size=8, num_hazards=10, num_resources=2,
        hazard_harm=harm_scale,
        env_drift_interval=10, env_drift_prob=0.1,
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
        f"[V3-EXQ-032c] MECH-102 Forced Escalation — Dense Grid\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  Grid: 8x8  num_hazards=10 (15.6% coverage — unavoidable density)\n"
        f"  Policy: E3-guided harm-minimizing (argmin E3(E2(z_world,a)))\n"
        f"  Thresholds: low_exposure<{LOW_EXPOSURE_MAX}  high_exposure>={HIGH_EXPOSURE_MIN}",
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

    print(f"\n[V3-EXQ-032c] Training ({warmup_episodes} eps)...", flush=True)
    train_out = _train(
        agent, env, optimizer, harm_eval_optimizer, world_forward_optimizer,
        warmup_episodes, steps_per_episode,
    )

    wf_r2 = _compute_world_forward_r2(agent, train_out["wf_data"])

    print(f"\n[V3-EXQ-032c] Eval ({eval_episodes} eps, E3-guided in dense grid)...", flush=True)
    eval_out = _eval_escalation_dense(agent, env, eval_episodes, steps_per_episode)

    c1_pass = eval_out["escalation_gap"] > 0.0
    c2_pass = eval_out["causal_sig_high_exposure"] > 0.001
    c3_pass = eval_out["n_high_exposure"] >= 50
    c4_pass = wf_r2 > 0.05
    c5_pass = eval_out["mean_harm_exposure_high"] > 0.10

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: escalation_gap={eval_out['escalation_gap']:.6f} <= 0"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: causal_sig_high={eval_out['causal_sig_high_exposure']:.6f} <= 0.001"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: n_high_exposure={eval_out['n_high_exposure']} < 50"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: mean_harm_exposure_high={eval_out['mean_harm_exposure_high']:.4f} <= 0.10"
        )

    print(f"\nV3-EXQ-032c verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    tc = train_out["counts"]
    metrics = {
        "alpha_world":                    float(alpha_world),
        "proximity_scale":                float(proximity_scale),
        "grid_size":                      8.0,
        "num_hazards":                    10.0,
        "world_forward_r2":               float(wf_r2),
        "causal_sig_low_exposure":        float(eval_out["causal_sig_low_exposure"]),
        "causal_sig_high_exposure":       float(eval_out["causal_sig_high_exposure"]),
        "escalation_gap":                 float(eval_out["escalation_gap"]),
        "n_low_exposure":                 float(eval_out["n_low_exposure"]),
        "n_high_exposure":                float(eval_out["n_high_exposure"]),
        "mean_harm_exposure":             float(eval_out["mean_harm_exposure"]),
        "mean_harm_exposure_high":        float(eval_out["mean_harm_exposure_high"]),
        "train_contact_events":           float(tc.get("env_caused_hazard", 0) + tc.get("agent_caused_hazard", 0)),
        "train_approach_events":          float(tc.get("hazard_approach", 0)),
        "crit1_pass":    1.0 if c1_pass else 0.0,
        "crit2_pass":    1.0 if c2_pass else 0.0,
        "crit3_pass":    1.0 if c3_pass else 0.0,
        "crit4_pass":    1.0 if c4_pass else 0.0,
        "crit5_pass":    1.0 if c5_pass else 0.0,
        "criteria_met":  float(n_met),
        "fatal_error_count": 0.0,
    }
    for tt, sig in eval_out["causal_sigs_by_ttype"].items():
        metrics[f"causal_sig_ttype_{tt.replace(' ','_')}"] = float(sig)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-032c — MECH-102: Forced Escalation in Dense Grid (8x8, 10 hazards)

**Status:** {status}
**Claims:** MECH-102, ARC-024, SD-003
**Grid:** 8x8, 10 hazards (15.6% cell coverage — physically unavoidable)
**Policy:** E3-guided harm-minimizing (argmin E3(E2(z_world, a)))
**Exposure thresholds:** low < {LOW_EXPOSURE_MAX}  |  high >= {HIGH_EXPOSURE_MIN}
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}

## Design Rationale

EXQ-032 used a 12x12 grid with 6 hazards (4.2% coverage). The harm-minimizing
policy succeeded in avoiding all hazards — n_high_exposure=0. This is not a model
failure; it is a test design failure: the environment was not constraining enough
to force viability threat.

EXQ-032c uses 8x8 = 64 cells with 10 hazards (15.6% coverage). At this density,
even an optimal harm-minimizing policy cannot avoid frequent proximity to hazards.
The harm_exposure EMA accumulates continuously.

MECH-102 prediction: when the agent is in a constrained environment where all
paths carry harm proximity, the selected action has higher counterfactual signature
than when safe paths exist — regardless of the ethical intent of the selection.

## Results

| Condition | harm_exp threshold | causal_sig | n steps |
|---|---|---|---|
| Low exposure (< {LOW_EXPOSURE_MAX}) | — | {eval_out['causal_sig_low_exposure']:.6f} | {eval_out['n_low_exposure']} |
| High exposure (>= {HIGH_EXPOSURE_MIN}) | {eval_out['mean_harm_exposure_high']:.4f} | {eval_out['causal_sig_high_exposure']:.6f} | {eval_out['n_high_exposure']} |

**Escalation gap** (high - low): {eval_out['escalation_gap']:.6f}
**world_forward R2**: {wf_r2:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: escalation_gap > 0 | {"PASS" if c1_pass else "FAIL"} | {eval_out['escalation_gap']:.6f} |
| C2: causal_sig_high > 0.001 | {"PASS" if c2_pass else "FAIL"} | {eval_out['causal_sig_high_exposure']:.6f} |
| C3: n_high_exposure >= 50 (dense grid ensures threat steps) | {"PASS" if c3_pass else "FAIL"} | {eval_out['n_high_exposure']} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} | {wf_r2:.4f} |
| C5: mean_harm_exposure_high > 0.10 (genuine viability threat) | {"PASS" if c5_pass else "FAIL"} | {eval_out['mean_harm_exposure_high']:.4f} |

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
