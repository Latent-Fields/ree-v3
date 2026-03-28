"""
V3-EXQ-071d -- Rollout-Batched SD-003 Attribution (GPU-Optimized Variant)

Supersedes: V3-EXQ-071d (ModuleNotFoundError: scipy not installed on this machine)
Fix: replaced scipy.stats.spearmanr with pure-numpy rank correlation.

Claims: SD-003, ARC-024, MECH-071

The EXQ-030b PASS (sequential attribution) established the SD-003 pipeline as
working at world_dim=32 with 4 counterfactual actions evaluated one at a time:

    for cf_idx in range(num_actions):        # 3 sequential calls
        z_cf    = E2.world_forward(z, a_cf)  # [1, 32]
        harm_cf = E3.harm_eval(z_cf)         # [1, 1]

This experiment validates a batch-vectorized version:

    z_cf_batch    = E2.world_forward(z_batch, a_cf_batch)  # [3, 32] -- one call
    harm_cf_batch = E3.harm_eval(z_cf_batch)               # [3, 1]  -- one call

Scientific question: does batching the counterfactual rollouts preserve attribution
selectivity (rank ordering, gap sign, magnitude)? This is a prerequisite for
SD-010 experiments where attribution pipelines will be more expensive (larger
harm_obs channel, multiple distractors) and GPU batching will be the primary
amortization strategy.

Protocol:
  Phase 1 -- Train: 150 warmup episodes on standard env (4 hazards, alpha_world=0.9)
             Identical to EXQ-030b training loop (Fix 2: E3 trained on E2-predicted states)
  Phase 2 -- Paired eval: run BOTH sequential and batched attribution on the same
             steps in the same eval episodes (same random seed, deterministic)
             - Sequential: 3 calls to world_forward, 3 calls to harm_eval
             - Batched:    1 call to world_forward([3, d]), 1 call to harm_eval([3, 1])
  Phase 3 -- Distractor sweep: eval across 3 env configs (n_hazards = 2, 4, 6)
             same trained model; tests whether batched attr holds under different
             hazard densities

Pass criteria:
  C1: batched attribution_gap > 0 (agent approach > env attribution -- primary SD-003 criterion)
  C2: rank_corr(batched_causal_sigs, sequential_causal_sigs) > 0.90 per distractor level
      (batched gives same rank ordering as sequential)
  C3: batched causal_sig_approach > 0.005 (minimum signal, same threshold as EXQ-030b C3)
  C4: world_forward_r2 > 0.05 (E2 learned world dynamics)
  C5: n_approach_steps >= 20 per distractor level (sufficient events)

Machine affinity: Daniel-PC (GPU-batched variant; Mac handles sequential baseline EXQ-030b)
Estimated runtime: ~145 min on Daniel-PC CPU (GPU will be used for eval batching if available)
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


EXPERIMENT_TYPE = "v3_exq_071d_rollout_batched_attribution"
CLAIM_IDS       = ["SD-003", "ARC-024", "MECH-071"]

# Distractor sweep: different hazard counts applied at eval time
DISTRACTOR_LEVELS = [2, 4, 6]   # n_hazards


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_env(seed: int, num_hazards: int = 4) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=12, num_hazards=num_hazards, num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer,
    harm_eval_optimizer,
    world_forward_optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Identical training loop to EXQ-030b (Fix 2: E3 on observed + E2-predicted states).
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
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev       = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent    = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, num_actions - 1)
            action     = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action_idx)
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

            e1_loss   = agent.compute_prediction_loss()
            e2_loss   = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(wf_data) >= 16:
                k     = min(32, len(wf_data))
                idxs  = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                pred  = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred, zw1_b)
                if wf_loss.requires_grad:
                    world_forward_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e2.world_transition.parameters(), 0.5)
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
                    zw_pos_obs.to(agent.device), zw_neg_obs.to(agent.device),
                    zw_pos_pred,                  zw_neg_pred,
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
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            a_prev       = action.detach()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  "
                f"approach={counts.get('hazard_approach', 0)}  wf_buf={len(wf_data)}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


def _eval_paired(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    num_hazards_label: int,
) -> Dict:
    """
    Run sequential and batched attribution on the same steps (same seed path).
    Both methods evaluate the same z_world and action at each step.
    """
    agent.eval()
    num_actions = env.action_dim

    ttypes = ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]
    seq_sigs:    Dict[str, List[float]] = {t: [] for t in ttypes}
    batch_sigs:  Dict[str, List[float]] = {t: [] for t in ttypes}

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

            action_idx = random.randint(0, num_actions - 1)
            action     = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                z_world_actual = agent.e2.world_forward(z_world, action)
                harm_actual    = agent.e3.harm_eval(z_world_actual)

                cf_indices = [i for i in range(num_actions) if i != action_idx]

                # -- Sequential attribution (EXQ-030b style) --
                seq_sig_vals = []
                for cf_idx in cf_indices:
                    a_cf     = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_cf     = agent.e2.world_forward(z_world, a_cf)
                    harm_cf  = agent.e3.harm_eval(z_cf)
                    seq_sig_vals.append(float((harm_actual - harm_cf).item()))
                mean_seq_sig = float(np.mean(seq_sig_vals)) if seq_sig_vals else 0.0

                # -- Batched attribution --
                n_cf = len(cf_indices)
                a_cf_batch = torch.zeros(n_cf, num_actions, device=agent.device)
                for j, cf_idx in enumerate(cf_indices):
                    a_cf_batch[j, cf_idx] = 1.0
                z_world_batch = z_world.expand(n_cf, -1)           # [n_cf, world_dim]
                z_cf_batch    = agent.e2.world_forward(z_world_batch, a_cf_batch)
                harm_cf_batch = agent.e3.harm_eval(z_cf_batch)     # [n_cf, 1]
                harm_actual_b = harm_actual.expand(n_cf, -1)
                batch_sig_vals = (harm_actual_b - harm_cf_batch).squeeze(-1).tolist()
                mean_batch_sig = float(np.mean(batch_sig_vals)) if batch_sig_vals else 0.0

            if ttype in seq_sigs:
                seq_sigs[ttype].append(mean_seq_sig)
                batch_sigs[ttype].append(mean_batch_sig)

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    seq_mean   = {t: _mean(seq_sigs[t])   for t in ttypes}
    batch_mean = {t: _mean(batch_sigs[t]) for t in ttypes}
    n_counts   = {t: len(seq_sigs[t])     for t in ttypes}

    # Rank correlation between batched and sequential per-step signals
    all_seq   = []
    all_batch = []
    for t in ttypes:
        all_seq.extend(seq_sigs[t])
        all_batch.extend(batch_sigs[t])

    if len(all_seq) > 10:
        try:
            x = np.array(all_seq,   dtype=float)
            y = np.array(all_batch, dtype=float)
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            n  = float(len(x))
            d2 = float(((rx - ry) ** 2).sum())
            rank_corr = float(1.0 - 6.0 * d2 / (n * (n * n - 1.0)))
        except Exception:
            rank_corr = 0.0
    else:
        rank_corr = 0.0

    batch_gap = batch_mean["hazard_approach"] - batch_mean["env_caused_hazard"]

    print(f"\n  --- Paired Attribution Eval (n_hazards={num_hazards_label}) ---", flush=True)
    print(f"  {'type':28s}  seq_sig      batch_sig    n", flush=True)
    for t in ttypes:
        print(
            f"  {t:28s}  {seq_mean[t]:+.6f}    {batch_mean[t]:+.6f}  {n_counts[t]}",
            flush=True,
        )
    print(f"  rank_corr(batched, sequential): {rank_corr:.4f}", flush=True)
    print(f"  batched attribution_gap: {batch_gap:.6f}", flush=True)

    return {
        "seq_mean":   seq_mean,
        "batch_mean": batch_mean,
        "n_counts":   n_counts,
        "rank_corr":  rank_corr,
        "batch_attribution_gap": batch_gap,
    }


def _compute_wf_r2(agent, wf_data) -> float:
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all  = torch.cat([d[0] for d in wf_data], dim=0)
        a_all   = torch.cat([d[1] for d in wf_data], dim=0)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0)
        pred    = agent.e2.world_forward(zw_all, a_all)
        pred_t  = pred[n_train:]
        tgt_t   = zw1_all[n_train:]
        if pred_t.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_t - pred_t) ** 2).sum()
        ss_tot = ((tgt_t - tgt_t.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R2 (test n={pred_t.shape[0]}): {r2:.4f}", flush=True)
    return r2


def run(
    seed: int = 0,
    warmup_episodes: int = 150,
    eval_episodes: int = 30,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[V3-EXQ-071d] Rollout-Batched SD-003 Attribution\n"
        f"  warmup={warmup_episodes} eps  eval={eval_episodes} eps/level\n"
        f"  distractor_levels={DISTRACTOR_LEVELS} (n_hazards)\n"
        f"  Validates: batched world_forward([n_cf, d]) equiv to sequential",
        flush=True,
    )

    train_env = _make_env(seed, num_hazards=4)
    config    = REEConfig.from_dims(
        body_obs_dim=train_env.body_obs_dim,
        world_obs_dim=train_env.world_obs_dim,
        action_dim=train_env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=train_env.action_dim,
    )
    agent = REEAgent(config)

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

    print(f"\n[V3-EXQ-071d] Training ({warmup_episodes} episodes)...", flush=True)
    train_out = _train(
        agent, train_env, optimizer, harm_eval_optimizer, world_forward_optimizer,
        warmup_episodes, steps_per_episode,
    )

    wf_r2 = _compute_wf_r2(agent, train_out["wf_data"])

    # Eval at each distractor level
    level_results = {}
    for n_haz in DISTRACTOR_LEVELS:
        print(f"\n[V3-EXQ-071d] Eval: n_hazards={n_haz} ({eval_episodes} eps)...", flush=True)
        eval_env = _make_env(seed + 1000 + n_haz, num_hazards=n_haz)
        lr_out   = _eval_paired(agent, eval_env, eval_episodes, steps_per_episode, n_haz)
        level_results[n_haz] = lr_out

    # Aggregate criteria
    c1_vals = [level_results[h]["batch_attribution_gap"] > 0.0 for h in DISTRACTOR_LEVELS]
    c2_vals = [level_results[h]["rank_corr"] > 0.90             for h in DISTRACTOR_LEVELS]
    c3_vals = [level_results[h]["batch_mean"]["hazard_approach"] > 0.005 for h in DISTRACTOR_LEVELS]
    c4_pass = wf_r2 > 0.05
    c5_vals = [level_results[h]["n_counts"].get("hazard_approach", 0) >= 20 for h in DISTRACTOR_LEVELS]

    # Primary criteria across all distractor levels
    c1_pass = all(c1_vals)
    c2_pass = all(c2_vals)
    c3_pass = all(c3_vals)
    c5_pass = all(c5_vals)

    n_criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"

    failure_notes = []
    for h in DISTRACTOR_LEVELS:
        lr = level_results[h]
        if not lr["batch_attribution_gap"] > 0:
            failure_notes.append(f"C1 FAIL n_hazards={h}: batch_gap={lr['batch_attribution_gap']:.6f} <= 0")
        if not lr["rank_corr"] > 0.90:
            failure_notes.append(f"C2 FAIL n_hazards={h}: rank_corr={lr['rank_corr']:.4f} <= 0.90")
        if not lr["batch_mean"]["hazard_approach"] > 0.005:
            failure_notes.append(f"C3 FAIL n_hazards={h}: batch_sig_approach={lr['batch_mean']['hazard_approach']:.6f} <= 0.005")
        if not lr["n_counts"].get("hazard_approach", 0) >= 20:
            failure_notes.append(f"C5 FAIL n_hazards={h}: n_approach={lr['n_counts'].get('hazard_approach',0)} < 20")
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")

    print(f"\n[V3-EXQ-071d] Verdict: {status}  ({n_criteria_met}/5 criteria)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics: dict = {
        "world_forward_r2": float(wf_r2),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(n_criteria_met),
    }
    for h in DISTRACTOR_LEVELS:
        lr = level_results[h]
        prefix = f"haz{h}"
        metrics[f"{prefix}_batch_attribution_gap"] = float(lr["batch_attribution_gap"])
        metrics[f"{prefix}_rank_corr"]             = float(lr["rank_corr"])
        metrics[f"{prefix}_batch_sig_approach"]    = float(lr["batch_mean"]["hazard_approach"])
        metrics[f"{prefix}_seq_sig_approach"]      = float(lr["seq_mean"]["hazard_approach"])
        metrics[f"{prefix}_n_approach"]            = float(lr["n_counts"].get("hazard_approach", 0))

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    # Build distractor table rows
    dist_rows = ""
    for h in DISTRACTOR_LEVELS:
        lr = level_results[h]
        dist_rows += (
            f"| {h} | {lr['batch_mean']['hazard_approach']:+.6f} | "
            f"{lr['seq_mean']['hazard_approach']:+.6f} | "
            f"{lr['rank_corr']:.4f} | "
            f"{lr['batch_attribution_gap']:+.6f} | "
            f"{lr['n_counts'].get('hazard_approach', 0)} |\n"
        )

    summary_markdown = f"""# V3-EXQ-071d -- Rollout-Batched SD-003 Attribution

**Status:** {status}
**Claims:** SD-003, ARC-024, MECH-071
**Predecessor:** EXQ-030b PASS (sequential attribution baseline)
**Supersedes:** V3-EXQ-071b
**Warmup:** {warmup_episodes} eps | **Eval:** {eval_episodes} eps x {len(DISTRACTOR_LEVELS)} distractor levels
**alpha_world:** {alpha_world} | **world_forward R2:** {wf_r2:.4f}

## Attribution Results by Distractor Level

| n_hazards | batch_sig_approach | seq_sig_approach | rank_corr | batch_gap | n_approach |
|---|---|---|---|---|---|
{dist_rows}

## Pass Criteria

| Criterion | Result |
|---|---|
| C1: batched attribution_gap > 0 (all levels) | {"PASS" if c1_pass else "FAIL"} |
| C2: rank_corr(batched, sequential) > 0.90 (all levels) | {"PASS" if c2_pass else "FAIL"} |
| C3: batched causal_sig_approach > 0.005 (all levels) | {"PASS" if c3_pass else "FAIL"} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} ({wf_r2:.4f}) |
| C5: n_approach >= 20 per level | {"PASS" if c5_pass else "FAIL"} |

Criteria met: {n_criteria_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status":            status,
        "metrics":           metrics,
        "summary_markdown":  summary_markdown,
        "claim_ids":         CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_criteria_met >= 3 else "weakens")
        ),
        "experiment_type":   EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--warmup",       type=int,   default=150)
    parser.add_argument("--eval-eps",     type=int,   default=30)
    parser.add_argument("--steps",        type=int,   default=200)
    parser.add_argument("--alpha-world",  type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]

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
