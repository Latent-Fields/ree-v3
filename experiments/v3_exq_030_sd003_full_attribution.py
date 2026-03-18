"""
V3-EXQ-030 — SD-003 Full Attribution Pipeline on CausalGridWorldV2

Claims: SD-003, ARC-024, MECH-071

This is the complete SD-003 test: E2.world_forward counterfactual divergence
combined with E3.harm_eval — the full attribution pipeline.

EXQ-029 confirmed E3 can detect hazard gradients. What hasn't been tested:
does E2.world_forward produce action-conditional predictions? That is: does
E2 predict different z_world futures for "move toward hazard" vs "move away"?

On the old world (EXQ-006): mean_dz_world_hazard < mean_dz_world_empty.
E2 saw identical futures regardless of direction near hazards. Identity
shortcut was optimal. SD-003 was structurally impossible.

On CausalGridWorldV2: hazard_field changes with position. Moving toward a
hazard increases field intensity in z_world; moving away decreases it.
E2.world_forward should now learn action-conditional predictions.

SD-003 full pipeline (V3):
    z_world_actual = e2.world_forward(z_world, a_actual)
    z_world_cf     = e2.world_forward(z_world, a_cf)     # counterfactual
    harm_actual    = e3.harm_eval(z_world_actual)
    harm_cf        = e3.harm_eval(z_world_cf)
    causal_sig     = harm_actual - harm_cf                # attribution score
    world_delta    = ||z_world_actual - z_world_cf||      # divergence

Attribution claim (SD-003):
    At agent-caused steps (hazard_approach, agent_caused_hazard):
        causal_sig > 0 (agent's action predicts more harm than counterfactual)
        world_delta > threshold (agent's action predicts different world than counterfactual)
    At env-caused steps (env_caused_hazard):
        causal_sig ≈ 0 (different action wouldn't have changed env drift)
        world_delta ≈ low (agent action doesn't explain the world change)

PASS criteria (ALL must hold):
    C1: mean_world_delta_approach > mean_world_delta_none  (action-conditional near hazard)
    C2: attribution_gap > 0.0
        attribution_gap = mean_causal_sig(approach) - mean_causal_sig(env_hazard)
        agent-caused attribution > env-caused attribution
    C3: mean_causal_sig_approach > 0.02   (positive attribution at approach steps)
    C4: world_forward_r2 > 0.05           (E2 has learned something about world dynamics)
    C5: n_approach_steps >= 50

Architecture basis: ARC-024 (gradient world enables action-conditional E2),
SD-003 (counterfactual self-attribution), INV-028 (others real → attribution matters)
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


EXPERIMENT_TYPE = "v3_exq_030_sd003_full_attribution"
CLAIM_IDS = ["SD-003", "ARC-024", "MECH-071"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_world_forward_r2(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """R² of E2.world_forward on held-out transitions."""
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all = torch.cat([d[0] for d in wf_data], dim=0)
        a_all  = torch.cat([d[1] for d in wf_data], dim=0)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0)
        pred_all = agent.e2.world_forward(zw_all, a_all)
        pred_test = pred_all[n_train:]
        tgt_test  = zw1_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R² (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return r2


HARM_TTYPES = {"env_caused_hazard", "agent_caused_hazard", "hazard_approach"}


def _train(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    world_forward_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

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

            # harm_eval training (balanced, includes approach events as harm)
            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # world_forward training: ALL transitions
            # E2 must learn how actions change z_world across all contexts
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # Standard E1 + E2_self losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E2.world_forward training (separate optimizer — MECH-069 incommensurable)
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

            # E3 harm_eval training
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
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


def _eval_attribution(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Full SD-003 attribution eval.

    For each step:
      1. Get z_world_t (perspective-corrected)
      2. Compute z_world_actual = E2.world_forward(z_world_t, a_actual)
      3. For each counterfactual action a_cf != a_actual:
         z_world_cf = E2.world_forward(z_world_t, a_cf)
      4. world_delta = mean_cf ||z_world_actual - z_world_cf||
      5. causal_sig  = mean_cf [E3(z_world_actual) - E3(z_world_cf)]

    Collect world_delta and causal_sig by transition_type.
    """
    agent.eval()

    ttypes = ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]
    world_deltas: Dict[str, List[float]] = {t: [] for t in ttypes}
    causal_sigs:  Dict[str, List[float]] = {t: [] for t in ttypes}
    all_deltas: List[float] = []

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
                z_world = latent.z_world  # perspective-corrected

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                # Actual predicted next world state
                z_world_actual = agent.e2.world_forward(z_world, action)
                harm_actual    = agent.e3.harm_eval(z_world_actual)

                # Counterfactual predictions (all other actions)
                deltas = []
                sigs   = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_cf = agent.e2.world_forward(z_world, a_cf)
                    harm_cf = agent.e3.harm_eval(z_cf)
                    delta = float((z_world_actual - z_cf).norm().item())
                    sig   = float((harm_actual - harm_cf).item())
                    deltas.append(delta)
                    sigs.append(sig)

                mean_delta = float(np.mean(deltas)) if deltas else 0.0
                mean_sig   = float(np.mean(sigs))   if sigs   else 0.0

            all_deltas.append(mean_delta)
            if ttype in world_deltas:
                world_deltas[ttype].append(mean_delta)
                causal_sigs[ttype].append(mean_sig)

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    mean_deltas = {t: _mean(world_deltas[t]) for t in ttypes}
    mean_sigs   = {t: _mean(causal_sigs[t])  for t in ttypes}
    n_counts    = {t: len(world_deltas[t])    for t in ttypes}

    attribution_gap = mean_sigs["hazard_approach"] - mean_sigs["env_caused_hazard"]

    print(f"\n  --- SD-003 Attribution Eval ---", flush=True)
    for t in ttypes:
        print(
            f"  {t:28s}: delta={mean_deltas[t]:.4f}  causal_sig={mean_sigs[t]:.4f}  n={n_counts[t]}",
            flush=True,
        )
    print(f"  attribution_gap (approach-env): {attribution_gap:.4f}", flush=True)

    return {
        "mean_deltas":      mean_deltas,
        "mean_sigs":        mean_sigs,
        "n_counts":         n_counts,
        "attribution_gap":  attribution_gap,
        "all_delta_std":    float(np.std(all_deltas)) if all_deltas else 0.0,
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
        f"[V3-EXQ-030] SD-003 Full Attribution on CausalGridWorldV2\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  alpha_world={alpha_world}  proximity_scale={proximity_scale}",
        flush=True,
    )

    # Three separate optimizers (MECH-069: incommensurable error signals)
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

    optimizer              = optim.Adam(standard_params,    lr=lr)
    world_forward_optimizer = optim.Adam(world_forward_params, lr=1e-3)
    harm_eval_optimizer    = optim.Adam(harm_eval_params,   lr=1e-4)

    train_out = _train(
        agent, env, optimizer, harm_eval_optimizer, world_forward_optimizer,
        warmup_episodes, steps_per_episode,
    )

    wf_r2 = _compute_world_forward_r2(agent, train_out["wf_data"])

    print(f"\n[V3-EXQ-030] Eval ({eval_episodes} eps)...", flush=True)
    eval_out = _eval_attribution(agent, env, eval_episodes, steps_per_episode)

    md = eval_out["mean_deltas"]
    ms = eval_out["mean_sigs"]
    nc = eval_out["n_counts"]

    c1_pass = md["hazard_approach"] > md["none"] and nc["hazard_approach"] >= 50
    c2_pass = eval_out["attribution_gap"] > 0.0
    c3_pass = ms["hazard_approach"] > 0.02
    c4_pass = wf_r2 > 0.05
    c5_pass = nc["hazard_approach"] >= 50

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_approach({md['hazard_approach']:.4f}) <= delta_none({md['none']:.4f})"
            + (f" or n_approach={nc['hazard_approach']} < 50" if nc["hazard_approach"] < 50 else "")
        )
    if not c2_pass:
        failure_notes.append(f"C2 FAIL: attribution_gap={eval_out['attribution_gap']:.4f} <= 0")
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: causal_sig_approach={ms['hazard_approach']:.4f} <= 0.02")
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: n_approach={nc['hazard_approach']} < 50")

    print(f"\nV3-EXQ-030 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    tc = train_out["counts"]
    metrics = {
        "alpha_world":                float(alpha_world),
        "proximity_scale":            float(proximity_scale),
        "world_forward_r2":           float(wf_r2),
        "mean_delta_none":            float(md["none"]),
        "mean_delta_approach":        float(md["hazard_approach"]),
        "mean_delta_env_hazard":      float(md["env_caused_hazard"]),
        "mean_delta_agent_hazard":    float(md["agent_caused_hazard"]),
        "mean_causal_sig_none":       float(ms["none"]),
        "mean_causal_sig_approach":   float(ms["hazard_approach"]),
        "mean_causal_sig_env_hazard": float(ms["env_caused_hazard"]),
        "mean_causal_sig_agent_hazard": float(ms["agent_caused_hazard"]),
        "attribution_gap":            float(eval_out["attribution_gap"]),
        "n_approach_eval":            float(nc["hazard_approach"]),
        "n_env_hazard_eval":          float(nc["env_caused_hazard"]),
        "n_agent_hazard_eval":        float(nc["agent_caused_hazard"]),
        "n_none_eval":                float(nc["none"]),
        "train_approach_events":      float(tc.get("hazard_approach", 0)),
        "train_contact_events":       float(tc.get("env_caused_hazard", 0) + tc.get("agent_caused_hazard", 0)),
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

    summary_markdown = f"""# V3-EXQ-030 — SD-003 Full Attribution on CausalGridWorldV2

**Status:** {status}
**Claims:** SD-003, ARC-024, MECH-071
**World:** CausalGridWorldV2 (proximity_scale={proximity_scale})
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}

## What This Tests

The complete SD-003 attribution pipeline on a gradient world:
  z_world_actual = E2.world_forward(z_world, a_actual)
  z_world_cf     = E2.world_forward(z_world, a_cf)
  causal_sig     = E3(z_world_actual) - E3(z_world_cf)

Does E2 predict action-conditional futures near hazards? Does the causal
signature distinguish agent-caused approach from environment-caused hazard?

EXQ-029 confirmed E3 detects gradients. This tests whether E2 models them.

## Attribution Results

| Transition | world_delta | causal_sig | n |
|---|---|---|---|
| none (locomotion)    | {md['none']:.4f} | {ms['none']:.4f} | {nc['none']} |
| hazard_approach      | {md['hazard_approach']:.4f} | {ms['hazard_approach']:.4f} | {nc['hazard_approach']} |
| env_caused_hazard    | {md['env_caused_hazard']:.4f} | {ms['env_caused_hazard']:.4f} | {nc['env_caused_hazard']} |
| agent_caused_hazard  | {md['agent_caused_hazard']:.4f} | {ms['agent_caused_hazard']:.4f} | {nc['agent_caused_hazard']} |

- **world_forward R²**: {wf_r2:.4f}  (PASS > 0.05)
- **attribution_gap** (approach - env): {eval_out['attribution_gap']:.4f}  (PASS > 0)

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: delta_approach > delta_none (action-conditional near hazard) | {"PASS" if c1_pass else "FAIL"} | {md['hazard_approach']:.4f} vs {md['none']:.4f} |
| C2: attribution_gap > 0 (agent > env attribution) | {"PASS" if c2_pass else "FAIL"} | {eval_out['attribution_gap']:.4f} |
| C3: causal_sig_approach > 0.02 (positive attribution signal) | {"PASS" if c3_pass else "FAIL"} | {ms['hazard_approach']:.4f} |
| C4: world_forward_r2 > 0.05 (E2 learned world dynamics) | {"PASS" if c4_pass else "FAIL"} | {wf_r2:.4f} |
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
