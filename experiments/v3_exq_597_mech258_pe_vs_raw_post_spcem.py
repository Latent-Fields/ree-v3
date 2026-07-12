#!/opt/local/bin/python3
"""
V3-EXQ-597 -- MECH-258: precision-weighted affective PE vs raw magnitude (post-SP-CEM).

MECH-258 claims z_harm_a enters dACC action selection as precision-weighted
prediction error against E2_harm_a, not as raw ||z_harm_a|| magnitude. Historical
V3-EXQ-445h supported forward-model fit (C1) under monostrategy; this run
re-validates under main-path SP-CEM defaults (ARC-065) and adds a direct
PE-vs-raw discriminative read.

Arms (single-variable on the PE source):
  PE_FORWARD       -- use_e2_harm_a=True,  use_dacc=True (canonical MECH-258)
  RAW_NORM_ABLATION -- use_e2_harm_a=False, use_dacc=True (dACC falls back to
                      ||z_harm_a|| when z_harm_a_pred is None)

Phased training (same as 445h reef substrate):
  P0 (150 eps): encoder + world pipeline warmup
  P1 (100 eps): E2_harm_a on frozen z_harm_a (PE_FORWARD only)
  P2 (30 eps):  eval, epsilon=0

Pre-registered PASS (MECH-258 evidence):
  C0: action_class_entropy >= 0.10 in >=2/3 seeds (PE_FORWARD, diversity gate)
  C1: harm_a_forward_r2 >= 0.3 in >=2/3 seeds (PE_FORWARD)
  C2: corr(|bias|, model_pe) > corr(|bias|, raw_norm) + 0.05 in >=2/3 seeds
      (PE_FORWARD; model_pe = ||z - z_pred||, raw_norm = ||z||)

claim_ids: ["MECH-258"]
experiment_purpose: "evidence"
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_597_mech258_pe_vs_raw_post_spcem"
CLAIM_IDS = ["MECH-258"]
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-597"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 150
P0_EPS = 150
P1_EPS = 100
P2_EPS = 30
EPSILON_TRAIN = 0.1
EPSILON_EVAL = 0.0

CONDITIONS = ["PE_FORWARD", "RAW_NORM_ABLATION"]

# Pre-registered thresholds
ENTROPY_MIN = 0.10
FORWARD_R2_MIN = 0.30
CORR_DELTA_MIN = 0.05
CORR_MIN_STEPS = 12


def _pearsonr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < CORR_MIN_STEPS:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x < 1e-12 or den_y < 1e-12:
        return float("nan")
    return num / (den_x * den_y)


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
        reef_enabled=True,
        n_reef_patches=3,
        reef_patch_radius=2,
        hazard_food_attraction=0.7,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    use_e2 = condition == "PE_FORWARD"
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
        use_dacc=True,
        dacc_weight=0.5,
        dacc_interaction_weight=0.3,
        dacc_foraging_weight=0.2,
        dacc_suppression_weight=4.0,
        dacc_suppression_memory=8,
        dacc_precision_scale=5000.0,
        dacc_effort_cost=0.1,
        dacc_drive_coupling=0.0,
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

    has_e2 = condition == "PE_FORWARD" and agent.e2_harm_a is not None
    optim_e2_a = (
        torch.optim.Adam(agent.e2_harm_a.parameters(), lr=5e-4) if has_e2 else None
    )

    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_opt = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)

    harm_pos: List[torch.Tensor] = []
    harm_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple] = []
    max_buf = 2000

    total_training_eps = P0_EPS + P1_EPS
    p1_start = P0_EPS
    p2_start = P0_EPS + P1_EPS

    action_counts: Dict[int, int] = {}
    bias_sum = 0.0
    bias_cnt = 0
    r2_pairs: List = []

    bias_trace: List[float] = []
    model_pe_trace: List[float] = []
    raw_norm_trace: List[float] = []

    prev_z_harm_a: Optional[torch.Tensor] = None
    prev_z_world: Optional[torch.Tensor] = None
    prev_action: Optional[torch.Tensor] = None

    for ep_idx in range(P0_EPS + P1_EPS + P2_EPS):
        agent.reset()
        _obs, obs_dict = env.reset()
        prev_z_harm_a = None
        prev_z_world = None
        prev_action = None

        is_p0 = ep_idx < p1_start
        is_p1 = p1_start <= ep_idx < p2_start
        is_p2 = ep_idx >= p2_start
        epsilon = EPSILON_EVAL if is_p2 else EPSILON_TRAIN

        if verbose and (ep_idx + 1) % 50 == 0:
            print(
                f"  [train] {condition} seed={seed} ep {ep_idx + 1}/{total_training_eps}",
                flush=True,
            )

        for _step in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body,
                obs_world=world,
                obs_harm=harm,
                obs_harm_a=harm_a,
                obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
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
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_opt.step()
                if prev_z_world is not None and prev_action is not None:
                    wf_buf.append((prev_z_world.cpu(), prev_action.cpu(), z_world_curr.cpu()))
                    if len(wf_buf) > max_buf:
                        wf_buf = wf_buf[-max_buf:]
                if len(wf_buf) >= 16:
                    k = min(32, len(wf_buf))
                    idx = torch.randperm(len(wf_buf))[:k].tolist()
                    zw = torch.cat([wf_buf[i][0] for i in idx]).to(agent.device)
                    av = torch.cat([wf_buf[i][1] for i in idx]).to(agent.device)
                    zw1 = torch.cat([wf_buf[i][2] for i in idx]).to(agent.device)
                    wf_pred = agent.e2.world_forward(zw, av)
                    wf_l = F.mse_loss(wf_pred, zw1)
                    if wf_l.requires_grad:
                        wf_opt.zero_grad()
                        wf_l.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(agent.e2.world_transition.parameters())
                            + list(agent.e2.world_action_encoder.parameters()),
                            1.0,
                        )
                        wf_opt.step()
                _obs, harm_signal, done, _info, obs_dict = env.step(action)
                if harm_signal < 0:
                    harm_pos.append(z_world_curr)
                    if len(harm_pos) > max_buf:
                        harm_pos = harm_pos[-max_buf:]
                else:
                    harm_neg.append(z_world_curr)
                    if len(harm_neg) > max_buf:
                        harm_neg = harm_neg[-max_buf:]
                if len(harm_pos) >= 4 and len(harm_neg) >= 4:
                    kp = min(16, len(harm_pos))
                    kn = min(16, len(harm_neg))
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
                        harm_eval_opt.zero_grad()
                        hl.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                        harm_eval_opt.step()
            else:
                _obs, _, done, _, obs_dict = env.step(action)

            if is_p1 and has_e2 and prev_z_harm_a is not None and latent.z_harm_a is not None:
                z_pred = agent.e2_harm_a(prev_z_harm_a.detach(), action.detach())
                loss = agent.e2_harm_a.compute_loss(z_pred, latent.z_harm_a.detach())
                optim_e2_a.zero_grad()
                loss.backward()
                optim_e2_a.step()

            if is_p2:
                action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
                if agent._dacc_last_bias is not None:
                    b_abs = float(agent._dacc_last_bias.abs().mean().item())
                    bias_sum += b_abs
                    bias_cnt += 1
                    bias_trace.append(b_abs)
                    if latent.z_harm_a is not None:
                        z_a = latent.z_harm_a.squeeze(0) if latent.z_harm_a.dim() > 1 else latent.z_harm_a
                        raw_norm_trace.append(float(z_a.norm().item()))
                        if has_e2 and prev_z_harm_a is not None:
                            with torch.no_grad():
                                z_pred_e = agent.e2_harm_a(
                                    prev_z_harm_a.detach(), action.detach()
                                )
                                mp = float((z_a - z_pred_e.squeeze(0)).norm().item())
                            model_pe_trace.append(mp)
                        else:
                            model_pe_trace.append(float("nan"))
                if has_e2 and prev_z_harm_a is not None and latent.z_harm_a is not None:
                    with torch.no_grad():
                        z_pred_e = agent.e2_harm_a(prev_z_harm_a.detach(), action.detach())
                        r2_pairs.append((z_pred_e.detach().cpu(), latent.z_harm_a.detach().cpu()))

            prev_z_harm_a = (
                latent.z_harm_a.detach().clone() if latent.z_harm_a is not None else None
            )
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

    corr_pe = float("nan")
    corr_raw = float("nan")
    if len(bias_trace) >= CORR_MIN_STEPS and len(model_pe_trace) == len(bias_trace):
        pe_pairs = [
            (b, m)
            for b, m, r in zip(bias_trace, model_pe_trace, raw_norm_trace)
            if not math.isnan(m)
        ]
        raw_pairs = [(b, r) for b, r in zip(bias_trace, raw_norm_trace)]
        if len(pe_pairs) >= CORR_MIN_STEPS:
            corr_pe = _pearsonr([p[0] for p in pe_pairs], [p[1] for p in pe_pairs])
        if len(raw_pairs) >= CORR_MIN_STEPS:
            corr_raw = _pearsonr([p[0] for p in raw_pairs], [p[1] for p in raw_pairs])

    entropy = _entropy(action_counts)
    mean_bias = bias_sum / bias_cnt if bias_cnt > 0 else 0.0
    corr_delta = (
        corr_pe - corr_raw
        if not (math.isnan(corr_pe) or math.isnan(corr_raw))
        else float("nan")
    )

    result = {
        "seed": seed,
        "condition": condition,
        "harm_a_forward_r2": float(r2) if not math.isnan(r2) else None,
        "action_class_entropy": float(entropy),
        "mean_score_bias_abs": float(mean_bias),
        "corr_bias_model_pe": float(corr_pe) if not math.isnan(corr_pe) else None,
        "corr_bias_raw_norm": float(corr_raw) if not math.isnan(corr_raw) else None,
        "corr_delta_pe_minus_raw": float(corr_delta) if not math.isnan(corr_delta) else None,
        "p2_bias_samples": len(bias_trace),
        "action_counts": {str(k): v for k, v in action_counts.items()},
    }
    if verbose:
        r2s = f"{r2:.3f}" if not math.isnan(r2) else "n/a"
        print(
            f"  [seed={seed} {condition}] r2={r2s} ent={entropy:.3f} "
            f"corr_pe={corr_pe:.3f} corr_raw={corr_raw:.3f} delta={corr_delta:.3f}",
            flush=True,
        )
    return result


def main(dry_run: bool = False, output_dir: Optional[str] = None) -> Dict:
    args = argparse.Namespace(dry_run=dry_run, output_dir=output_dir)

    global P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP
    if args.dry_run:
        print("Smoke: seed=42 P0=3/P1=3/P2=3 steps=20", flush=True)
        P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 3, 3, 3, 20

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

    seeds_run = [42] if args.dry_run else SEEDS
    all_results: List[Dict] = []
    for seed in seeds_run:
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}", flush=True)
            r = _run_condition(seed=seed, condition=cond, verbose=True)
            all_results.append(r)
            passed = (
                (r.get("harm_a_forward_r2") or 0.0) >= FORWARD_R2_MIN
                if cond == "PE_FORWARD"
                else True
            ) and (r.get("action_class_entropy") or 0.0) >= 0.0
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    pe_rows = [r for r in all_results if r["condition"] == "PE_FORWARD"]

    c0_wins = sum(
        1 for r in pe_rows if (r.get("action_class_entropy") or 0.0) >= ENTROPY_MIN
    )
    c0 = c0_wins >= 2

    c1_wins = sum(
        1 for r in pe_rows if (r.get("harm_a_forward_r2") or 0.0) >= FORWARD_R2_MIN
    )
    c1 = c1_wins >= 2

    c2_wins = sum(
        1
        for r in pe_rows
        if (r.get("corr_delta_pe_minus_raw") or -999.0) >= CORR_DELTA_MIN
    )
    c2 = c2_wins >= 2

    outcome = "PASS" if (c0 and c1 and c2) else "FAIL"
    summary = {
        "c0_policy_entropy": {"wins": c0_wins, "pass": c0, "threshold": ENTROPY_MIN},
        "c1_forward_r2": {"wins": c1_wins, "pass": c1, "threshold": FORWARD_R2_MIN},
        "c2_pe_beats_raw_corr": {
            "wins": c2_wins,
            "pass": c2,
            "threshold_delta": CORR_DELTA_MIN,
        },
    }

    if c0 and c1 and c2:
        mech258_dir = "supports"
    elif c1:
        mech258_dir = "mixed"
    else:
        mech258_dir = "weakens"

    per_claim = {"MECH-258": mech258_dir}

    print(f"\nOutcome: {outcome}", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": "v3_exq_445h_sd032b_dacc_reef",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": mech258_dir,
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
            "sp_cem_defaults": True,
            "env_size": 14,
            "reef_enabled": True,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"\nOutput written to: {out_file}", flush=True)

    if args.dry_run:
        try:
            out_file.unlink()
        except OSError:
            pass
        print("Smoke test completed (criteria may FAIL on tiny run)", flush=True)

    return {
        "outcome": outcome,
        "manifest_path": out_file,
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    cli = parser.parse_args()
    result = main(dry_run=cli.dry_run, output_dir=cli.output_dir)
    if not result["dry_run"]:
        emit_outcome(
            outcome=result["outcome"],
            manifest_path=result["manifest_path"],
            run_id=result["run_id"],
            queue_id=QUEUE_ID,
        )
