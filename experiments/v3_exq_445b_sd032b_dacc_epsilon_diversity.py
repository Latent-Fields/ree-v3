#!/opt/local/bin/python3
"""
V3-EXQ-445b -- SD-032b dACC: Epsilon-Greedy Diversity During Training.

Claims: SD-032b, MECH-258, MECH-260

Follows up V3-EXQ-445 (FAIL: monostrategy collapse) with the minimal intervention:
add epsilon-greedy exploration (eps=0.1) during P0+P1 training to prevent the policy
from collapsing to a single action. In P2 evaluation, eps=0.0 (pure policy).

Compared to EXQ-445a (full pipeline), this experiment does NOT add full E3 training.
It tests the narrower hypothesis: is the monostrategy a pure training artifact that
can be broken without changing the training objective? If yes: ε-greedy is sufficient.
If no: the monostrategy is fundamental (environment too simple, E3 always picks one action).

Same 3-arm design as EXQ-445 (OFF, ON_INDEPENDENT, ON_SHARED) to keep ARC-033/ARC-058
arbitration alive. Same P0/P1/P2 episode counts.

Score_bias scale fix: dacc_precision_scale=5000.0, dacc_weight=0.5 (same as 445a).

Phased training:
  P0 (50 eps): EMA warmup with epsilon=0.1 random action injection
  P1 (100 eps): E2_harm_a frozen-encoder training with epsilon=0.1
  P2 (30 eps): Eval with epsilon=0.0 (pure policy)

PASS criteria (same as EXQ-445):
  C1 (MECH-258): both ON arms achieve harm_a_forward_r2 >= 0.3 in >=2/3 seeds
  C2 (SD-032b): either ON arm has |entropy_ON - entropy_OFF| >= 0.1 in >=2/3 seeds
  C3 (MECH-260): ON entropy >= OFF entropy in >=2/3 seeds for both arms

PASS = C1 AND C2 AND C3.

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
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_445b_sd032b_dacc_epsilon_diversity"
CLAIM_IDS = ["SD-032b", "MECH-258", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 120
P0_EPS = 50
P1_EPS = 100
P2_EPS = 30
EPSILON_TRAIN = 0.1   # fraction of steps using random action during P0/P1
EPSILON_EVAL = 0.0    # pure policy during P2

CONDITIONS = ["OFF", "ON_INDEPENDENT", "ON_SHARED"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    use_dacc = condition != "OFF"
    use_e2 = condition != "OFF"
    use_shared = condition == "ON_SHARED"
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
        use_shared_harm_trunk=use_shared,
        e2_harm_a_lr=5e-4,
        use_dacc=use_dacc,
        dacc_weight=0.5,
        dacc_interaction_weight=0.3,
        dacc_foraging_weight=0.2,
        dacc_suppression_weight=0.5,
        dacc_suppression_memory=8,
        dacc_precision_scale=5000.0,
        dacc_effort_cost=0.1,
        dacc_drive_coupling=0.0,
    )
    return REEAgent(cfg)


def _entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
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

    has_e2_harm_a = condition != "OFF" and agent.e2_harm_a is not None
    optim_e2_a = None
    if has_e2_harm_a:
        optim_e2_a = torch.optim.Adam(agent.e2_harm_a.parameters(), lr=5e-4)

    total_eps = P0_EPS + P1_EPS + P2_EPS
    phase_p1_start = P0_EPS
    phase_p2_start = P0_EPS + P1_EPS

    action_counts: Dict[int, int] = {}
    score_bias_abs_sum = 0.0
    score_bias_count = 0
    forward_r2_pairs: List = []

    prev_z_harm_a: Optional[torch.Tensor] = None

    for ep_idx in range(total_eps):
        agent.reset()
        _obs, obs_dict = env.reset()
        prev_z_harm_a = None

        phase_is_p1 = phase_p1_start <= ep_idx < phase_p2_start
        phase_is_p2 = ep_idx >= phase_p2_start
        # epsilon: 0.1 during training, 0.0 during eval
        epsilon = EPSILON_EVAL if phase_is_p2 else EPSILON_TRAIN

        for step in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body, obs_world=world, obs_harm=harm,
                obs_harm_a=harm_a, obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            # Epsilon-greedy: override action with random during training
            if epsilon > 0.0 and random.random() < epsilon:
                a_idx = random.randint(0, action_dim - 1)
                action = torch.zeros(1, action_dim, device=agent.device)
                action[0, a_idx] = 1.0
            else:
                a_idx = int(action[0].argmax().item())

            # P1: train E2_harm_a
            if phase_is_p1 and has_e2_harm_a and prev_z_harm_a is not None and latent.z_harm_a is not None:
                z_prev_det = prev_z_harm_a.detach()
                z_next_det = latent.z_harm_a.detach()
                a_det = action.detach()
                z_pred = agent.e2_harm_a(z_prev_det, a_det)
                loss = agent.e2_harm_a.compute_loss(z_pred, z_next_det)
                optim_e2_a.zero_grad()
                loss.backward()
                optim_e2_a.step()

            # P2: collect metrics
            if phase_is_p2:
                action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
                if agent._dacc_last_bias is not None:
                    score_bias_abs_sum += float(agent._dacc_last_bias.abs().mean().item())
                    score_bias_count += 1
                if has_e2_harm_a and prev_z_harm_a is not None and latent.z_harm_a is not None:
                    with torch.no_grad():
                        z_pred_eval = agent.e2_harm_a(
                            prev_z_harm_a.detach(), action.detach()
                        )
                        forward_r2_pairs.append(
                            (z_pred_eval.detach().cpu(), latent.z_harm_a.detach().cpu())
                        )

            prev_z_harm_a = latent.z_harm_a.detach().clone() if latent.z_harm_a is not None else None
            _obs, _, done, _, obs_dict = env.step(action)
            if done:
                break

    # Metrics
    if forward_r2_pairs:
        preds = torch.cat([p for p, _ in forward_r2_pairs])
        targets = torch.cat([t for _, t in forward_r2_pairs])
        ss_res = float(((targets - preds) ** 2).sum())
        ss_tot = float(((targets - targets.mean(dim=0)) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    else:
        r2 = float("nan")

    entropy = _entropy(action_counts)
    mean_bias = score_bias_abs_sum / score_bias_count if score_bias_count > 0 else 0.0

    result = {
        "seed": seed,
        "condition": condition,
        "harm_a_forward_r2": float(r2) if not math.isnan(r2) else None,
        "action_class_entropy": float(entropy),
        "mean_score_bias_abs": float(mean_bias),
        "action_counts": {str(k): v for k, v in action_counts.items()},
    }
    if verbose:
        r2_str = f"{r2:.3f}" if not math.isnan(r2) else "n/a"
        print(
            f"  [seed={seed} {condition}] r2={r2_str} entropy={entropy:.3f} bias={mean_bias:.2f}",
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke: seed=42 P0=2/P1=3/P2=3 steps=20")
        global P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP
        P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 2, 3, 3, 20
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

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    off_r = by_cond("OFF")
    indep_r = by_cond("ON_INDEPENDENT")
    shared_r = by_cond("ON_SHARED")

    c1_indep = sum(1 for r in indep_r if (r["harm_a_forward_r2"] or 0.0) >= 0.3)
    c1_shared = sum(1 for r in shared_r if (r["harm_a_forward_r2"] or 0.0) >= 0.3)
    c1 = (c1_indep >= 2) and (c1_shared >= 2)

    def entropy_delta_wins(on_rs, off_rs):
        return sum(
            1 for a, b in zip(on_rs, off_rs)
            if abs(a["action_class_entropy"] - b["action_class_entropy"]) >= 0.1
        )

    c2_indep = entropy_delta_wins(indep_r, off_r)
    c2_shared = entropy_delta_wins(shared_r, off_r)
    c2 = (c2_indep >= 2) or (c2_shared >= 2)

    def not_collapsed(on_rs, off_rs):
        return sum(1 for a, b in zip(on_rs, off_rs) if a["action_class_entropy"] >= b["action_class_entropy"])

    c3_indep = not_collapsed(indep_r, off_r)
    c3_shared = not_collapsed(shared_r, off_r)
    c3 = (c3_indep >= 2) and (c3_shared >= 2)

    outcome = "PASS" if (c1 and c2 and c3) else "FAIL"

    summary = {
        "c1_mech258": {"indep_wins": c1_indep, "shared_wins": c1_shared, "pass": c1},
        "c2_sd032b": {"indep_wins": c2_indep, "shared_wins": c2_shared, "pass": c2},
        "c3_mech260": {"indep_wins": c3_indep, "shared_wins": c3_shared, "pass": c3},
    }
    per_claim = {
        "SD-032b": "supports" if (c1 and c2) else ("mixed" if c1 else "weakens"),
        "MECH-258": "supports" if c1 else "weakens",
        "MECH-260": "supports" if c3 else "weakens",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
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
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "p0_eps": P0_EPS,
            "p1_eps": P1_EPS,
            "p2_eps": P2_EPS,
            "steps_per_ep": STEPS_PER_EP,
            "epsilon_train": EPSILON_TRAIN,
            "epsilon_eval": EPSILON_EVAL,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
