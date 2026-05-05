#!/opt/local/bin/python3
"""
V3-EXQ-445g -- SD-032b dACC: Reef-Enriched Substrate (bias calibration fix).

Claims: SD-032b, MECH-258, MECH-260
Supersedes: V3-EXQ-445f

Root cause of 445f FAIL (C2/C3): score_bias was O(17-21), ~10-20x the natural
E3 inter-candidate variation (O(0.5-2.0)). Two fixes:

  1. dacc_bias_max_abs=2.0: clips total bias to +-2.0, keeping it within
     the range where E3 softmax can actually discriminate candidates.
  2. dacc_suppression_weight=4.0: raises the per-candidate suppression
     differential to ~2.0 (4.0 * 0.5 frequency), enough to dominate
     within the clipped budget.

Both fixes land via the new DACCConfig.dacc_bias_max_abs field and the
DACCtoE3Adapter.forward() clip. No other changes to env or training phases.

Phased training (unchanged from 445f):
  P0 (150 eps): Full pipeline + epsilon=0.1
  P1 (100 eps): E2_harm_a frozen-encoder training + epsilon=0.1
  P2 (30 eps): Eval, epsilon=0.0

PASS criteria (unchanged):
  C1 (MECH-258): ON harm_a_forward_r2 >= 0.3 in >=2/3 seeds
  C2 (SD-032b): |entropy_ON - entropy_OFF| >= 0.1 in >=2/3 seeds
  C3 (MECH-260): entropy_ON >= entropy_OFF in >=2/3 seeds

PASS = C1 AND C2 AND C3.

Per-claim direction:
  SD-032b: "supports" if c1 and c2 and c3
           "mixed"    if c1 but c2_wins==1 (partial signal)
           "does_not_support" if c2_wins==0 (zero entropy differential measured)
           "weakens"  if c1 fails
  MECH-258: "supports" if c1, "weakens" otherwise
  MECH-260: "supports" if c3, "weakens" otherwise

claim_ids: ["SD-032b", "MECH-258", "MECH-260"]
experiment_purpose: "evidence"
"""

import sys
import json
import argparse
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_445g_sd032b_dacc_reef"
CLAIM_IDS = ["SD-032b", "MECH-258", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 150
P0_EPS = 150
P1_EPS = 100
P2_EPS = 30
EPSILON_TRAIN = 0.1
EPSILON_EVAL = 0.0

CONDITIONS = ["OFF", "ON_INDEPENDENT"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=14,
        num_hazards=5,
        num_resources=7,
        hazard_harm=0.04,
        proximity_harm_scale=0.20,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
        # SD-050 reef enrichment -- breaks monostrategy on 14x14 grid
        reef_enabled=True,
        n_reef_patches=3,
        reef_patch_radius=2,
        hazard_food_attraction=0.7,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    use_dacc = condition != "OFF"
    use_e2 = condition != "OFF"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        use_e2_harm_a=use_e2,
        use_shared_harm_trunk=False,
        e2_harm_a_lr=5e-4,
        use_dacc=use_dacc,
        dacc_weight=0.5,
        dacc_interaction_weight=0.3,
        dacc_foraging_weight=0.2,
        # Fix 1: suppression_weight raised 0.5 -> 4.0 so differential is O(2.0)
        # within the clipped budget, enough to influence E3 selection.
        dacc_suppression_weight=4.0,
        dacc_suppression_memory=8,
        dacc_precision_scale=5000.0,
        dacc_effort_cost=0.1,
        dacc_drive_coupling=0.0,
        # Fix 2: clip total score_bias to +-2.0, keeping it within E3's natural
        # inter-candidate variation range (O(0.5-2.0)) so softmax can discriminate.
        dacc_bias_max_abs=2.0,
    )
    return REEAgent(cfg)


def _entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return ent


def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
    harm_hist = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
    return body, world, harm, harm_a, harm_hist


def _run_condition(seed: int, condition: str, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, condition)
    action_dim = env.action_dim
    world_dim = agent.config.latent.world_dim

    has_e2 = condition != "OFF" and agent.e2_harm_a is not None
    optim_e2_a = torch.optim.Adam(agent.e2_harm_a.parameters(), lr=5e-4) if has_e2 else None

    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()), lr=1e-3
    )
    harm_eval_opt = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)

    harm_pos: List[torch.Tensor] = []
    harm_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple] = []
    MAX_BUF = 2000

    total_eps = P0_EPS + P1_EPS + P2_EPS
    p1_start = P0_EPS
    p2_start = P0_EPS + P1_EPS

    action_counts: Dict[int, int] = {}
    bias_sum = 0.0
    bias_cnt = 0
    r2_pairs: List = []
    prev_z_harm_a: Optional[torch.Tensor] = None
    prev_z_world: Optional[torch.Tensor] = None
    prev_action: Optional[torch.Tensor] = None

    for ep_idx in range(total_eps):
        agent.reset()
        _obs, obs_dict = env.reset()
        prev_z_harm_a = None
        prev_z_world = None
        prev_action = None

        is_p0 = ep_idx < p1_start
        is_p1 = p1_start <= ep_idx < p2_start
        is_p2 = ep_idx >= p2_start
        epsilon = EPSILON_EVAL if is_p2 else EPSILON_TRAIN

        for step in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body, obs_world=world, obs_harm=harm,
                obs_harm_a=harm_a, obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            if epsilon > 0.0 and random.random() < epsilon:
                ai = random.randint(0, action_dim - 1)
                action = torch.zeros(1, action_dim, device=agent.device)
                action[0, ai] = 1.0
                a_idx = ai
            else:
                a_idx = int(action[0].argmax().item())

            z_world_curr = latent.z_world.detach()

            if is_p0:
                theta_z = agent.theta_buffer.summary()
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_opt.zero_grad(); e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_opt.step()
                if prev_z_world is not None and prev_action is not None:
                    wf_buf.append((prev_z_world.cpu(), prev_action.cpu(), z_world_curr.cpu()))
                    if len(wf_buf) > MAX_BUF:
                        wf_buf = wf_buf[-MAX_BUF:]
                if len(wf_buf) >= 16:
                    k = min(32, len(wf_buf))
                    idx = torch.randperm(len(wf_buf))[:k].tolist()
                    zw = torch.cat([wf_buf[i][0] for i in idx]).to(agent.device)
                    av = torch.cat([wf_buf[i][1] for i in idx]).to(agent.device)
                    zw1 = torch.cat([wf_buf[i][2] for i in idx]).to(agent.device)
                    wf_pred = agent.e2.world_forward(zw, av)
                    wf_l = F.mse_loss(wf_pred, zw1)
                    if wf_l.requires_grad:
                        wf_opt.zero_grad(); wf_l.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(agent.e2.world_transition.parameters()) +
                            list(agent.e2.world_action_encoder.parameters()), 1.0
                        )
                        wf_opt.step()
                _obs, harm_signal, done, _info, obs_dict = env.step(action)
                if harm_signal < 0:
                    harm_pos.append(z_world_curr)
                    if len(harm_pos) > MAX_BUF: harm_pos = harm_pos[-MAX_BUF:]
                else:
                    harm_neg.append(z_world_curr)
                    if len(harm_neg) > MAX_BUF: harm_neg = harm_neg[-MAX_BUF:]
                if len(harm_pos) >= 4 and len(harm_neg) >= 4:
                    kp = min(16, len(harm_pos)); kn = min(16, len(harm_neg))
                    pi = torch.randperm(len(harm_pos))[:kp].tolist()
                    ni = torch.randperm(len(harm_neg))[:kn].tolist()
                    zb = torch.cat([harm_pos[i] for i in pi] + [harm_neg[i] for i in ni])
                    tgt = torch.cat([
                        torch.ones(kp, 1, device=agent.device),
                        torch.zeros(kn, 1, device=agent.device),
                    ])
                    pred = agent.e3.harm_eval(theta_z.expand(zb.shape[0], -1))
                    hl = F.mse_loss(pred, tgt)
                    if hl.requires_grad:
                        harm_eval_opt.zero_grad(); hl.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                        harm_eval_opt.step()
            else:
                _obs, _, done, _, obs_dict = env.step(action)

            if is_p1 and has_e2 and prev_z_harm_a is not None and latent.z_harm_a is not None:
                z_pred = agent.e2_harm_a(prev_z_harm_a.detach(), action.detach())
                loss = agent.e2_harm_a.compute_loss(z_pred, latent.z_harm_a.detach())
                optim_e2_a.zero_grad(); loss.backward(); optim_e2_a.step()

            if is_p2:
                action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
                if agent._dacc_last_bias is not None:
                    bias_sum += float(agent._dacc_last_bias.abs().mean().item())
                    bias_cnt += 1
                if has_e2 and prev_z_harm_a is not None and latent.z_harm_a is not None:
                    with torch.no_grad():
                        z_pred_e = agent.e2_harm_a(prev_z_harm_a.detach(), action.detach())
                        r2_pairs.append((z_pred_e.detach().cpu(), latent.z_harm_a.detach().cpu()))

            prev_z_harm_a = latent.z_harm_a.detach().clone() if latent.z_harm_a is not None else None
            prev_z_world = z_world_curr
            prev_action = action.detach()
            if done:
                break

    if r2_pairs:
        preds = torch.cat([p for p, _ in r2_pairs])
        targets = torch.cat([t for _, t in r2_pairs])
        ss_res = float(((targets - preds) ** 2).sum())
        ss_tot = float(((targets - targets.mean(dim=0)) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    else:
        r2 = float("nan")

    entropy = _entropy(action_counts)
    mean_bias = bias_sum / bias_cnt if bias_cnt > 0 else 0.0

    result = {
        "seed": seed,
        "condition": condition,
        "harm_a_forward_r2": float(r2) if not math.isnan(r2) else None,
        "action_class_entropy": float(entropy),
        "mean_score_bias_abs": float(mean_bias),
        "action_counts": {str(k): v for k, v in action_counts.items()},
    }
    if verbose:
        r2s = f"{r2:.3f}" if not math.isnan(r2) else "n/a"
        print(f"  [seed={seed} {condition}] r2={r2s} entropy={entropy:.3f} bias={mean_bias:.2f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke: seed=42 P0=3/P1=3/P2=3 steps=20")
        global P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP
        P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 3, 3, 3, 20
        for cond in CONDITIONS:
            r = _run_condition(seed=42, condition=cond, verbose=True)
            assert r["action_class_entropy"] >= 0.0
        print("Smoke test PASSED")
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parents[1]
        out_dir = (
            script_dir.parent / "REE_assembly" / "evidence"
            / "experiments" / EXPERIMENT_TYPE
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        print(f"\nSeed {seed}", flush=True)
        for cond in CONDITIONS:
            print(f"  Running {cond}...", flush=True)
            r = _run_condition(seed=seed, condition=cond)
            all_results.append(r)

    off_r = [r for r in all_results if r["condition"] == "OFF"]
    on_r = [r for r in all_results if r["condition"] == "ON_INDEPENDENT"]

    c1_wins = sum(1 for r in on_r if (r["harm_a_forward_r2"] or 0.0) >= 0.3)
    c1 = c1_wins >= 2
    c2_wins = sum(1 for a, b in zip(on_r, off_r)
                  if abs(a["action_class_entropy"] - b["action_class_entropy"]) >= 0.1)
    c2 = c2_wins >= 2
    c3_wins = sum(1 for a, b in zip(on_r, off_r)
                  if a["action_class_entropy"] >= b["action_class_entropy"])
    c3 = c3_wins >= 2

    outcome = "PASS" if (c1 and c2 and c3) else "FAIL"
    summary = {
        "c1_mech258": {"wins": c1_wins, "pass": c1},
        "c2_sd032b": {"wins": c2_wins, "pass": c2},
        "c3_mech260": {"wins": c3_wins, "pass": c3},
    }

    # SD-032b direction: "does_not_support" if zero entropy differential was
    # measured across all seeds (monostrategy persists); "mixed" if partial;
    # "supports" if full criteria met.
    if c1 and c2 and c3:
        sd032b_dir = "supports"
    elif c2_wins == 0:
        sd032b_dir = "does_not_support"
    elif c1:
        sd032b_dir = "mixed"
    else:
        sd032b_dir = "weakens"

    per_claim = {
        "SD-032b": sd032b_dir,
        "MECH-258": "supports" if c1 else "weakens",
        "MECH-260": "supports" if c3 else "weakens",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": "v3_exq_445f_sd032b_dacc_reef",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": per_claim,
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS, "seeds": SEEDS,
            "p0_eps": P0_EPS, "p1_eps": P1_EPS, "p2_eps": P2_EPS,
            "steps_per_ep": STEPS_PER_EP,
            "env_size": 14, "num_hazards": 5, "num_resources": 7,
            "epsilon_train": EPSILON_TRAIN, "epsilon_eval": EPSILON_EVAL,
            "dacc_suppression_weight": 4.0,
            "dacc_bias_max_abs": 2.0,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
