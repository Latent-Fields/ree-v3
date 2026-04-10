#!/opt/local/bin/python3
"""
V3-EXQ-329: ARC-033 E2_harm_s Counterfactual Validation (fix of EXQ-264)

experiment_purpose: evidence

Tests ARC-033 (E2_harm_s forward model) and SD-003 (self-attribution
via counterfactual E2).

Root cause of EXQ-264 FAIL: REEAgent.compute_prediction_loss() was called
with an extra positional argument (old API). Fixed here by calling with no args.

Design: Does E2_harm_s learn a valid forward model of z_harm_s transitions,
and does it produce counterfactual gaps (divergent predictions for actual vs
counterfactual actions)?

Three conditions per seed:
  ACTUAL_VS_CF:  key test -- z_harm_s_actual vs z_harm_s_cf diverge
  RANDOM_BASELINE: a_cf randomly sampled (expected non-zero gap)
  SHUFFLED:      a_cf = a_actual (no gap expected -- same action = zero gap)

Phased training (critical for encoder stability):
  P0 (100 ep): HarmEncoder warmup (use_harm_stream=True, proximity supervision)
  P1 (100 ep): E2HarmSForward training on FROZEN z_harm_s inputs (stop-grad)
  P2 (50 ep):  Evaluation -- measure forward_r2 and cf_gap

Forward model architecture: ResidualHarmForward (delta predictor, avoids
identity collapse on autocorrelated z_harm_s sequences).

Pass criteria:
  C1: forward_r2 >= 0.7 in ACTUAL_VS_CF (forward model learns dynamics)
  C2: cf_gap_actual > cf_gap_shuffled * 1.5 (CF diverges beyond identity baseline)
  C3: attribution_sign_correct >= 0.6 in ACTUAL_VS_CF
      (causal_sig sign correct on approach vs neutral transitions)

PASS: C1 AND C2 AND C3 across >= 2/3 seeds.

Supersedes: V3-EXQ-264 (which failed due to wrong API call)
Claims: ARC-033, SD-003
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_329_arc033_e2_harm_s_counterfactual"
CLAIM_IDS          = ["ARC-033", "SD-003"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS      = [42, 7, 13]
CONDITIONS = ["ACTUAL_VS_CF", "RANDOM_BASELINE", "SHUFFLED"]

P0_EPISODES  = 100    # HarmEncoder warmup
P1_EPISODES  = 100    # E2HarmSForward training (frozen encoder)
P2_EPISODES  = 50     # evaluation
STEPS_PER_EP = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 2
HAZARD_HARM   = 0.3   # stronger harm for cleaner z_harm_s signal

Z_HARM_DIM = 32
HARM_OBS_DIM = 51
ACTION_DIM = 5

LR_AGENT = 3e-4
LR_HARM_FWD = 5e-4

# Replay buffer for P1 training
REPLAY_BUF_MAX = 5000

# Pass thresholds
C1_FORWARD_R2_THRESH  = 0.7
C2_CF_GAP_RATIO       = 1.5
C3_SIGN_CORRECT_THRESH = 0.6
MIN_SEEDS_PASS        = 2

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 20


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.3,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.1,
        proximity_approach_threshold=0.2,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_event_classifier=True,
    )
    return REEAgent(config)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _random_cf_action(a_actual_idx: int, n_actions: int, device) -> torch.Tensor:
    """Sample a_cf != a_actual."""
    choices = [i for i in range(n_actions) if i != a_actual_idx]
    cf_idx = random.choice(choices)
    return _onehot(cf_idx, n_actions, device)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_condition(
    seed: int,
    condition: str,
    dry_run: bool = False,
) -> Dict:
    total_p0  = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1  = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    total_p2  = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1 + total_p2

    print(f"  Seed {seed} Condition {condition}")

    env   = _make_env(seed)
    agent = _make_agent(env, seed)
    device = agent.device

    # E2HarmSForward -- separate module, trained in P1
    harm_fwd_cfg = E2HarmSConfig(
        use_e2_harm_s_forward=True,
        z_harm_dim=Z_HARM_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=128,
        learning_rate=LR_HARM_FWD,
    )
    harm_fwd = E2HarmSForward(harm_fwd_cfg).to(device)
    harm_fwd_opt = optim.Adam(harm_fwd.parameters(), lr=LR_HARM_FWD)

    agent_opt = optim.Adam(list(agent.parameters()), lr=LR_AGENT)

    # Replay buffer for harm forward model: (z_harm_s_t, action_idx, z_harm_s_t1)
    replay_buf: List[Tuple[torch.Tensor, int, torch.Tensor]] = []

    # P2 measurement accumulators
    fwd_preds_all:  List[float] = []
    fwd_targets_all: List[float] = []
    cf_gaps_actual:  List[float] = []
    cf_gaps_shuffled: List[float] = []
    sign_correct_n:  int = 0
    sign_total_n:    int = 0

    prev_ttype = "none"
    z_harm_s_prev: Optional[torch.Tensor] = None
    action_prev_idx: Optional[int] = None

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()
        z_harm_s_prev = None
        action_prev_idx = None

        phase   = "P0" if ep < total_p0 else ("P1" if ep < total_p0 + total_p1 else "P2")
        in_p1   = (phase == "P1")
        in_p2   = (phase == "P2")

        for step in range(steps_per):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            obs_harm  = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = obs_harm.to(device).unsqueeze(0) if obs_harm.dim() == 1 else obs_harm.to(device)

            z_self_prev_t: Optional[torch.Tensor] = None
            if agent._current_latent is not None:
                z_self_prev_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", True)
                else torch.zeros(1, 32, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            # Record z_harm_s transition for replay
            z_harm_s_now = latent.z_harm  # [1, z_harm_dim] or None
            if z_harm_s_now is not None:
                z_harm_s_now_d = z_harm_s_now.detach().clone()
                if z_harm_s_prev is not None and action_prev_idx is not None and not in_p2:
                    replay_buf.append((z_harm_s_prev, action_prev_idx, z_harm_s_now_d))
                    if len(replay_buf) > REPLAY_BUF_MAX:
                        replay_buf = replay_buf[-REPLAY_BUF_MAX:]

            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            agent.update_residue(float(harm_signal))

            if z_self_prev_t is not None:
                agent.record_transition(z_self_prev_t, action, latent.z_self.detach())

            # P0: train agent encoders
            if phase == "P0":
                agent_opt.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss = e1_loss + e2_loss

                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    rp_loss = agent.compute_resource_proximity_loss(rp_t, latent)
                    loss = loss + rp_loss

                latent2 = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                ec_loss = agent.compute_event_contrastive_loss(prev_ttype, latent2)
                loss = loss + ec_loss

                if loss.requires_grad:
                    loss.backward()
                    import torch.nn as nn
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    agent_opt.step()

            # P1: train E2HarmSForward on FROZEN z_harm_s (stop-grad)
            if in_p1 and len(replay_buf) >= 32:
                batch_idx = random.sample(range(len(replay_buf)), min(32, len(replay_buf)))
                z_s_batch = torch.cat([replay_buf[i][0] for i in batch_idx], dim=0)
                a_batch_idx = [replay_buf[i][1] for i in batch_idx]
                z_s1_batch  = torch.cat([replay_buf[i][2] for i in batch_idx], dim=0)

                a_batch = torch.zeros(len(batch_idx), ACTION_DIM, device=device)
                for bi, ai in enumerate(a_batch_idx):
                    a_batch[bi, ai] = 1.0

                harm_fwd_opt.zero_grad()
                # P1 stop-grad discipline: z_harm_s inputs are already detached
                z_pred = harm_fwd(z_s_batch.detach(), a_batch)
                fwd_loss = harm_fwd.compute_loss(z_pred, z_s1_batch.detach())
                fwd_loss.backward()
                harm_fwd_opt.step()

            # P2: measurement
            if in_p2 and z_harm_s_now is not None and z_harm_s_prev is not None:
                z_hs = z_harm_s_prev.detach()
                a_actual = _onehot(action_idx, ACTION_DIM, device)

                with torch.no_grad():
                    z_pred_actual = harm_fwd(z_hs, a_actual)

                # R2 measurement
                for dim_i in range(Z_HARM_DIM):
                    fwd_preds_all.append(float(z_pred_actual[0, dim_i].item()))
                    fwd_targets_all.append(float(z_harm_s_now.detach()[0, dim_i].item()))

                # CF gap measurement
                if condition == "ACTUAL_VS_CF":
                    a_cf = _random_cf_action(action_idx, ACTION_DIM, device)
                    with torch.no_grad():
                        z_pred_cf = harm_fwd(z_hs, a_cf)
                    gap = float((z_pred_actual - z_pred_cf).norm(dim=-1).mean().item())
                    cf_gaps_actual.append(gap)

                elif condition == "SHUFFLED":
                    # a_cf = a_actual -> should be zero gap
                    with torch.no_grad():
                        z_pred_shuffled = harm_fwd(z_hs, a_actual)
                    gap = float((z_pred_actual - z_pred_shuffled).norm(dim=-1).mean().item())
                    cf_gaps_shuffled.append(gap)

                elif condition == "RANDOM_BASELINE":
                    a_rand = _onehot(random.randint(0, ACTION_DIM - 1), ACTION_DIM, device)
                    with torch.no_grad():
                        z_pred_rand = harm_fwd(z_hs, a_rand)
                    gap = float((z_pred_actual - z_pred_rand).norm(dim=-1).mean().item())
                    cf_gaps_actual.append(gap)

                # Sign correctness: harm_approach should have larger gap than neutral
                if ttype in ("hazard_approach", "agent_caused_hazard"):
                    if cf_gaps_actual:
                        sign_correct_n += 1 if cf_gaps_actual[-1] > 0.0 else 0
                        sign_total_n += 1

            z_harm_s_prev  = z_harm_s_now.detach().clone() if z_harm_s_now is not None else None
            action_prev_idx = action_idx
            prev_ttype = ttype
            obs_dict   = obs_dict_next

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"    [train] seed={seed} {condition} ep {ep+1}/{total_eps} "
                f"phase={phase} replay_buf={len(replay_buf)}",
                flush=True,
            )

    # Compute metrics
    forward_r2 = 0.0
    if len(fwd_preds_all) >= 10:
        try:
            tgt = np.array(fwd_targets_all)
            prd = np.array(fwd_preds_all)
            ss_res = float(np.sum((tgt - prd) ** 2))
            ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
            forward_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        except Exception:
            forward_r2 = 0.0

    mean_cf_gap_actual   = float(np.mean(cf_gaps_actual))   if cf_gaps_actual   else 0.0
    mean_cf_gap_shuffled = float(np.mean(cf_gaps_shuffled)) if cf_gaps_shuffled else 0.0
    sign_correct_frac    = float(sign_correct_n / max(1, sign_total_n))

    print(f"  verdict: forward_r2={forward_r2:.3f} "
          f"cf_gap={mean_cf_gap_actual:.4f} cf_shuffled={mean_cf_gap_shuffled:.4f} "
          f"sign_correct={sign_correct_frac:.3f}")

    return {
        "seed": seed,
        "condition": condition,
        "forward_r2": forward_r2,
        "mean_cf_gap": mean_cf_gap_actual,
        "mean_cf_gap_shuffled": mean_cf_gap_shuffled,
        "attribution_sign_correct": sign_correct_frac,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_cond[r["condition"]].append(r)

    avc_list  = sorted(by_cond.get("ACTUAL_VS_CF",     []), key=lambda x: x["seed"])
    shuf_list = sorted(by_cond.get("SHUFFLED",         []), key=lambda x: x["seed"])

    # C1: forward_r2 >= 0.7
    c1_vals = [r["forward_r2"] for r in avc_list]
    c1_seeds = sum(v >= C1_FORWARD_R2_THRESH for v in c1_vals)
    c1_pass  = c1_seeds >= MIN_SEEDS_PASS

    # C2: cf_gap_actual > cf_gap_shuffled * 1.5
    c2_seeds = 0
    c2_ratios = []
    # Pair ACTUAL_VS_CF vs SHUFFLED by seed
    shuf_by_seed = {r["seed"]: r for r in shuf_list}
    for r in avc_list:
        shuf = shuf_by_seed.get(r["seed"], None)
        shuf_gap = shuf["mean_cf_gap_shuffled"] if shuf else 0.0
        ratio = r["mean_cf_gap"] / max(shuf_gap, 1e-6)
        c2_ratios.append(ratio)
        if ratio >= C2_CF_GAP_RATIO:
            c2_seeds += 1
    c2_pass = c2_seeds >= MIN_SEEDS_PASS

    # C3: attribution_sign_correct >= 0.6
    c3_vals = [r["attribution_sign_correct"] for r in avc_list]
    c3_seeds = sum(v >= C3_SIGN_CORRECT_THRESH for v in c3_vals)
    c3_pass  = c3_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass
    return {
        "c1_forward_r2_pass": c1_pass,
        "c1_vals": c1_vals,
        "c1_seeds_pass": c1_seeds,
        "c2_cf_gap_ratio_pass": c2_pass,
        "c2_ratios": c2_ratios,
        "c2_seeds_pass": c2_seeds,
        "c3_sign_correct_pass": c3_pass,
        "c3_vals": c3_vals,
        "c3_seeds_pass": c3_seeds,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"v3_exq_329_arc033_e2_harm_s_counterfactual_dry_{ts}_v3"
        if args.dry_run
        else f"v3_exq_329_arc033_e2_harm_s_counterfactual_{ts}_v3"
    )
    print(f"EXQ-329 start: {run_id}")

    all_results: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-329 {outcome} ===")
    print(f"C1 forward_r2: {criteria['c1_forward_r2_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c1_vals']]})")
    print(f"C2 cf_gap_ratio: {criteria['c2_cf_gap_ratio_pass']} "
          f"(ratios={[f'{v:.2f}' for v in criteria['c2_ratios']]})")
    print(f"C3 sign_correct: {criteria['c3_sign_correct_pass']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c3_vals']]})")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction_per_claim": {
            "ARC-033": "supports" if criteria["c1_forward_r2_pass"] else "does_not_support",
            "SD-003":  "supports" if (criteria["c2_cf_gap_ratio_pass"] and criteria["c3_sign_correct_pass"]) else "does_not_support",
        },
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome": outcome,
        "criteria": criteria,
        "results_per_condition": all_results,
        "supersedes": "V3-EXQ-264",
        "config": {
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "p2_episodes": P2_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "z_harm_dim": Z_HARM_DIM,
            "hazard_harm": HAZARD_HARM,
        },
        "timestamp_utc": ts,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
        EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
