#!/opt/local/bin/python3
"""
V3-EXQ-445a -- SD-032b dACC: Extended Training with Full Agent Pipeline.

Claims: SD-032b, MECH-258, MECH-260
Supersedes: V3-EXQ-445 (FAIL: monostrategy collapse, entropy_ON == entropy_OFF == 0.0)

Root cause of V3-EXQ-445 C2 FAIL (identified in review):
  EXQ-445 trained ONLY E2_harm_a (P0=50 eps warmup + P1=100 eps frozen head). The core
  E3 policy (harm_eval, terrain_prior) was NEVER trained. With an untrained E3, the CEM
  selector degenerates -- always choosing the same action regardless of trajectory scoring.
  Adding dACC score_bias onto a degenerate selector has no effect: entropy_ON == entropy_OFF == 0.

This experiment adds FULL agent training in P0 (E1 sensory prediction, E3 harm_eval,
terrain_prior -- same pattern as EXQ-397) before the E2_harm_a training phase. With a
trained E3 policy, C2 (dACC behavioral effect) becomes testable.

Also fixes score_bias scale:
  - dacc_precision_scale: 5000.0 (was 500.0) -- reduces precision amplification
  - dacc_weight: 0.5 (was 1.0) -- reduces overall score_bias magnitude

2-arm design (OFF vs ON_INDEPENDENT) -- 3 seeds.
DROP ON_SHARED arm (ARC-033/ARC-058 arbitration from 445 was inconclusive; requires
separate experiment after behavioral gate passes).

Phased training per condition, per seed:
  P0 (150 eps): FULL agent training
    - E1: sensory prediction loss
    - E3 harm_eval: balanced harm/no-harm buffer, MSE loss
    - terrain_prior: train on lowest-residue candidate (FIXED vs EXQ-397's E3-selected)
    - World-forward: E2.world_transition on (z_world_t, a_t) -> z_world_{t+1}
  P1 (100 eps): E2_harm_a training on FROZEN z_harm_a targets
  P2 (30 eps): Evaluation -- same metrics as EXQ-445 (action_class_entropy, forward_r2)

PASS criteria (same as EXQ-445):
  C1 (MECH-258): ON arm achieves harm_a_forward_r2 >= 0.3 in >=2/3 seeds
  C2 (SD-032b): |entropy_ON - entropy_OFF| >= 0.1 nats in >=2/3 seeds
  C3 (MECH-260): action_class_entropy ON >= OFF in >=2/3 seeds

PASS = C1 AND C2 AND C3.

Additional diagnostic: mean_score_bias_abs (should be < 100.0 with fixed scale).

claim_ids: ["SD-032b", "MECH-258", "MECH-260"]
experiment_purpose: "evidence"
  Rationale: Re-tests behavioral effect of dACC with trained agent. PASS resolves 445 FAIL
  and confirms score_bias scale was the root cause. FAIL means dACC has no effect even
  with a trained policy (stronger claim against SD-032b).
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


EXPERIMENT_TYPE = "v3_exq_445a_sd032b_dacc_full_pipeline"
CLAIM_IDS = ["SD-032b", "MECH-258", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 120
P0_EPS = 150
P1_EPS = 100
P2_EPS = 30

CONDITIONS = ["OFF", "ON_INDEPENDENT"]


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
    use_e2_harm_a = condition != "OFF"
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
        use_e2_harm_a=use_e2_harm_a,
        use_shared_harm_trunk=False,
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


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_condition(seed: int, condition: str, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, condition)
    world_dim = agent.config.latent.world_dim

    has_e2_harm_a = condition != "OFF" and agent.e2_harm_a is not None
    optim_e2_a = None
    if has_e2_harm_a:
        optim_e2_a = torch.optim.Adam(agent.e2_harm_a.parameters(), lr=5e-4)

    # Full pipeline optimizers (P0)
    e1_optim = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optim = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    terrain_optim = optim.Adam(
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters()),
        lr=5e-4,
    )
    harm_eval_optim = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_BUF = 2000

    total_eps = P0_EPS + P1_EPS + P2_EPS
    phase_p1_start = P0_EPS
    phase_p2_start = P0_EPS + P1_EPS

    action_counts: Dict[int, int] = {}
    score_bias_abs_sum = 0.0
    score_bias_count = 0
    forward_r2_pairs: List = []

    prev_z_harm_a: Optional[torch.Tensor] = None
    prev_z_world: Optional[torch.Tensor] = None
    prev_action: Optional[torch.Tensor] = None

    for ep_idx in range(total_eps):
        agent.reset()
        _obs, obs_dict = env.reset()
        prev_z_harm_a = None
        prev_z_world = None
        prev_action = None

        phase_is_p0 = ep_idx < phase_p1_start
        phase_is_p1 = phase_p1_start <= ep_idx < phase_p2_start
        phase_is_p2 = ep_idx >= phase_p2_start

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
            a_idx = int(action[0].argmax().item())

            z_world_curr = latent.z_world.detach()

            # ---- P0: Full pipeline training ----
            if phase_is_p0:
                theta_z = agent.theta_buffer.summary()

                # E1 sensory prediction loss
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optim.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optim.step()

                # World-forward buffer + training
                if prev_z_world is not None and prev_action is not None:
                    wf_buf.append((prev_z_world.cpu(), prev_action.cpu(), z_world_curr.cpu()))
                    if len(wf_buf) > MAX_BUF:
                        wf_buf = wf_buf[-MAX_BUF:]
                if len(wf_buf) >= 16:
                    k = min(32, len(wf_buf))
                    idxs = torch.randperm(len(wf_buf))[:k].tolist()
                    zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                    a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                    zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                    wf_pred = agent.e2.world_forward(zw_b, a_b)
                    wf_loss = F.mse_loss(wf_pred, zw1_b)
                    if wf_loss.requires_grad:
                        wf_optim.zero_grad()
                        wf_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(agent.e2.world_transition.parameters()) +
                            list(agent.e2.world_action_encoder.parameters()),
                            1.0,
                        )
                        wf_optim.step()

                # Terrain prior (trained on LOWEST-residue candidate, not E3-selected)
                # This prevents hippocampus from learning to navigate toward residue
                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        residue_scores = []
                        for c in candidates:
                            ws = c.get_world_state_sequence()
                            if ws is not None and ws.numel() > 0:
                                s = float(agent.residue_field.evaluate_trajectory(ws).mean().item())
                            else:
                                s = float("inf")
                            residue_scores.append(s)
                        best_idx = min(range(len(residue_scores)), key=lambda i: residue_scores[i])
                    best_ao = candidates[best_idx].get_action_object_sequence()
                    if best_ao is not None:
                        ao_mean_pred = agent.hippocampal._get_terrain_action_object_mean(
                            theta_z, e1_prior=e1_prior.detach()
                        )
                        terrain_loss = F.mse_loss(ao_mean_pred, best_ao.detach())
                        terrain_optim.zero_grad()
                        terrain_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(agent.hippocampal.terrain_prior.parameters()) +
                            list(agent.hippocampal.action_object_decoder.parameters()),
                            1.0,
                        )
                        terrain_optim.step()

                # Harm eval (balanced)
                harm_action = _obs, harm_signal, done_local, _info, obs_dict = env.step(action)
                done = done_local
                if harm_signal < 0:
                    harm_buf_pos.append(z_world_curr)
                    if len(harm_buf_pos) > MAX_BUF:
                        harm_buf_pos = harm_buf_pos[-MAX_BUF:]
                else:
                    harm_buf_neg.append(z_world_curr)
                    if len(harm_buf_neg) > MAX_BUF:
                        harm_buf_neg = harm_buf_neg[-MAX_BUF:]
                if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                    k_p = min(16, len(harm_buf_pos))
                    k_n = min(16, len(harm_buf_neg))
                    pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                    ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                    zw_b = torch.cat(
                        [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                    )
                    target = torch.cat([
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ], dim=0)
                    pred = agent.e3.harm_eval(theta_z.expand(zw_b.shape[0], -1))
                    h_loss = F.mse_loss(pred, target)
                    if h_loss.requires_grad:
                        harm_eval_optim.zero_grad()
                        h_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                        harm_eval_optim.step()

            else:
                _obs, harm_signal, done, _info, obs_dict = env.step(action)

            # ---- P1: E2_harm_a training (frozen encoder) ----
            if phase_is_p1 and has_e2_harm_a and prev_z_harm_a is not None and latent.z_harm_a is not None:
                z_prev_det = prev_z_harm_a.detach()
                z_next_det = latent.z_harm_a.detach()
                a_det = action.detach()
                z_pred = agent.e2_harm_a(z_prev_det, a_det)
                loss = agent.e2_harm_a.compute_loss(z_pred, z_next_det)
                optim_e2_a.zero_grad()
                loss.backward()
                optim_e2_a.step()

            # ---- P2: Evaluation ----
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
            prev_z_world = z_world_curr
            prev_action = action.detach()
            if done:
                break

    # Compute metrics
    if forward_r2_pairs:
        preds = torch.cat([p for p, _ in forward_r2_pairs], dim=0)
        targets = torch.cat([t for _, t in forward_r2_pairs], dim=0)
        ss_res = float(((targets - preds) ** 2).sum().item())
        ss_tot = float(((targets - targets.mean(dim=0)) ** 2).sum().item())
        harm_a_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    else:
        harm_a_r2 = float("nan")

    action_entropy = _entropy(action_counts)
    mean_bias = score_bias_abs_sum / score_bias_count if score_bias_count > 0 else 0.0

    result = {
        "seed": seed,
        "condition": condition,
        "harm_a_forward_r2": float(harm_a_r2) if not math.isnan(harm_a_r2) else None,
        "action_class_entropy": float(action_entropy),
        "mean_score_bias_abs": float(mean_bias),
        "action_counts": {str(k): v for k, v in action_counts.items()},
    }

    if verbose:
        r2_str = f"{harm_a_r2:.3f}" if not math.isnan(harm_a_r2) else "n/a"
        print(
            f"  [seed={seed} {condition}] "
            f"forward_r2={r2_str} "
            f"entropy={action_entropy:.3f} "
            f"bias_abs={mean_bias:.2f}",
            flush=True,
        )
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

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    off_res = by_cond("OFF")
    on_res = by_cond("ON_INDEPENDENT")

    # C1: ON harm_a_forward_r2 >= 0.3 in >=2/3 seeds
    c1_wins = sum(1 for r in on_res if (r["harm_a_forward_r2"] or 0.0) >= 0.3)
    c1 = c1_wins >= 2

    # C2: |entropy_ON - entropy_OFF| >= 0.1 in >=2/3 seeds
    c2_wins = sum(
        1 for on_r, off_r in zip(on_res, off_res)
        if abs(on_r["action_class_entropy"] - off_r["action_class_entropy"]) >= 0.1
    )
    c2 = c2_wins >= 2

    # C3: entropy_ON >= entropy_OFF in >=2/3 seeds
    c3_wins = sum(
        1 for on_r, off_r in zip(on_res, off_res)
        if on_r["action_class_entropy"] >= off_r["action_class_entropy"]
    )
    c3 = c3_wins >= 2

    outcome = "PASS" if (c1 and c2 and c3) else "FAIL"

    summary = {
        "c1_mech258_forward_r2": {"on_wins": c1_wins, "pass": c1, "desc": "ON harm_a_forward_r2>=0.3 in >=2/3 seeds"},
        "c2_sd032b_behavioral_effect": {"wins": c2_wins, "pass": c2, "desc": "|entropy_ON - entropy_OFF|>=0.1 in >=2/3 seeds"},
        "c3_mech260_no_collapse": {"wins": c3_wins, "pass": c3, "desc": "entropy_ON >= entropy_OFF in >=2/3 seeds"},
        "score_bias_diagnostic": {
            "mean_bias_per_seed": [r["mean_score_bias_abs"] for r in on_res],
            "desc": "dACC score_bias magnitude (should be <100 with fixed precision_scale)",
        },
    }

    per_claim = {
        "SD-032b": "supports" if (c1 and c2) else ("mixed" if c1 else "weakens"),
        "MECH-258": "supports" if c1 else "weakens",
        "MECH-260": "supports" if c3 else "weakens",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for k, v in summary.items():
        if "diagnostic" not in k:
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
        "supersedes": "v3_exq_445_sd032b_dacc_analog",
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "p0_eps": P0_EPS,
            "p1_eps": P1_EPS,
            "p2_eps": P2_EPS,
            "steps_per_ep": STEPS_PER_EP,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
