#!/opt/local/bin/python3
"""
V3-EXQ-445 -- SD-032b dACC-analog Adaptive Control validation.

Claims: SD-032b, MECH-258, MECH-260, ARC-033, ARC-058

Three-arm ablation validating:
  (a) SD-032b activation shifts action selection vs dACC-OFF baseline.
  (b) E2_harm_a forward model (MECH-258) achieves usable forward_r2 under
      phased training in both competing architectures.
  (c) ARC-033 (independent-per-stream) vs ARC-058 (shared-trunk) arbitration:
      which path produces better per-stream forward_r2 and/or better
      downstream dACC-mediated behaviour diversity (MECH-260 target).
  (d) MECH-260 recency/monostrategy suppression visibly diversifies the
      action-class distribution at the ON arms.

Conditions (all at 3 seeds, same env):
  OFF:             use_dacc=False, use_e2_harm_a=False.
  ON_INDEPENDENT:  use_dacc=True, use_e2_harm_a=True,
                   use_shared_harm_trunk=False  (ARC-033 path).
  ON_SHARED:       use_dacc=True, use_e2_harm_a=True,
                   use_shared_harm_trunk=True   (ARC-058 path).

Phased training per condition (ON arms):
  P0: encoder warmup (AffectiveHarmEncoder + SD-020 surprise target). 50 eps.
  P1: freeze z_harm_a -- train E2_harm_a on .detach()ed targets. 100 eps.
  P2: eval. 30 eps (with dACC active; measure harm_a_forward_r2 +
      action-class entropy + mode_ev/score_bias trace).

OFF arm runs 180 eps straight (no dACC state; no E2_harm_a).

Metrics per arm per seed:
  - harm_a_forward_r2:    P2 evaluation MSE against detach()ed next-step targets.
  - action_class_entropy: Shannon entropy over action-argmax distribution.
                          MECH-260 target: ON arms > OFF arm.
  - mean_score_bias_abs:  avg |dACC score_bias| over P2 (0 for OFF).
  - harm_rate_per_ep:     mean harm exposure per episode in P2.

Acceptance checks:
  C1 (MECH-258): both ON arms achieve harm_a_forward_r2 >= 0.3 in >=2/3 seeds
                 (forward model learnable under phased training; low bar --
                  signal > noise, not mastery).
  C2 (SD-032b): either ON arm differs meaningfully in action_class_entropy
                from OFF (|entropy_on - entropy_off| >= 0.1 nats in >=2/3 seeds)
                -- dACC routing has an observable behavioural effect.
  C3 (MECH-260): ON arm action_class_entropy >= OFF arm entropy in >=2/3
                 seeds (suppression does not collapse diversity; ideally raises it).
  C4 (ARC-033 vs ARC-058 arbitration, DIAGNOSTIC only, never fails the run):
                 report which arm has higher mean harm_a_forward_r2.
                 Result recorded in pass_criteria_summary for governance
                 adjudication; not an acceptance gate.

PASS: C1 AND C2 AND C3.
FAIL otherwise.

Note: C1 threshold r2 >= 0.3 is deliberately generous. This is a substrate
validation experiment, not a mastery test. The forward model is being
trained on a fresh substrate with no pre-existing curriculum; any
statistically-significant fit against random is evidence of substrate
integrity.

claim_ids: ["SD-032b", "MECH-258", "MECH-260", "ARC-033", "ARC-058"]
experiment_purpose: "diagnostic"
  Rationale: substrate readiness validation. A PASS confirms the wiring
  runs end-to-end and the competing-claim switch functions; governance
  arbitration between ARC-033 and ARC-058 requires a larger evidence
  experiment after this substrate gate.

See REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
See ree-v3/CLAUDE.md "SD-032b / MECH-258 / MECH-260 / ARC-058 ..." section.
"""

import sys
import json
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_445_sd032b_dacc_analog"
CLAIM_IDS = ["SD-032b", "MECH-258", "MECH-260", "ARC-033", "ARC-058"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 120
P0_EPS = 50
P1_EPS = 100
P2_EPS = 30

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
    use_e2_harm_a = condition != "OFF"
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
        use_e2_harm_a=use_e2_harm_a,
        use_shared_harm_trunk=use_shared,
        e2_harm_a_lr=5e-4,
        use_dacc=use_dacc,
        dacc_weight=1.0 if use_dacc else 0.0,
        dacc_interaction_weight=0.3,
        dacc_foraging_weight=0.2,
        dacc_suppression_weight=0.5,
        dacc_suppression_memory=8,
        dacc_precision_scale=500.0,
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


def _run_condition(seed: int, condition: str, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, condition)

    has_e2_harm_a = condition != "OFF" and agent.e2_harm_a is not None
    optim_e2_a = None
    if has_e2_harm_a:
        optim_e2_a = torch.optim.Adam(
            agent.e2_harm_a.parameters(), lr=5e-4
        )

    total_eps = P0_EPS + P1_EPS + P2_EPS if has_e2_harm_a else P0_EPS + P1_EPS + P2_EPS
    action_counts: Dict[int, int] = {}
    score_bias_abs_sum = 0.0
    score_bias_count = 0
    harm_per_ep: List[float] = []
    forward_r2_eval_pairs: List = []  # (pred, target) pairs for P2 eval

    # P0 + P1: training (no dACC behaviour metrics counted).
    # P2: evaluation, collect action entropy, score_bias abs, harm rate, forward pairs.
    phase_boundaries = (P0_EPS, P0_EPS + P1_EPS)

    def _obs_tensors(obs_dict):
        body = obs_dict["body_state"].float().unsqueeze(0)
        world = obs_dict["world_state"].float().unsqueeze(0)
        harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
        harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
        harm_hist = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
        return body, world, harm, harm_a, harm_hist

    for ep_idx in range(total_eps):
        agent.reset()
        _obs, obs_dict = env.reset()

        phase_is_p2 = ep_idx >= phase_boundaries[1]
        phase_is_p1 = (ep_idx >= phase_boundaries[0]) and (ep_idx < phase_boundaries[1])

        ep_harm = 0.0
        prev_z_harm_a = None

        for step in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body,
                obs_world=world,
                obs_harm=harm,
                obs_harm_a=harm_a,
                obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            world_dim_local = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim_local, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            a_idx = int(action[0].argmax().item())

            # Collect metrics only during P2 (evaluation phase).
            if phase_is_p2:
                action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
                if agent._dacc_last_bias is not None:
                    score_bias_abs_sum += float(
                        agent._dacc_last_bias.abs().mean().item()
                    )
                    score_bias_count += 1

            _obs, harm_signal, done, _info, obs_dict = env.step(action)
            hs = float(harm_signal)
            ep_harm += -hs if hs < 0 else 0.0

            # P1 training: update E2_harm_a on the transition (z_harm_a_prev, action_prev -> z_harm_a).
            # Caller MUST .detach() targets.
            if phase_is_p1 and has_e2_harm_a and prev_z_harm_a is not None and latent.z_harm_a is not None:
                z_prev_det = prev_z_harm_a.detach()
                z_next_det = latent.z_harm_a.detach()
                a_prev = action.detach()
                z_pred = agent.e2_harm_a(z_prev_det, a_prev)
                loss = agent.e2_harm_a.compute_loss(z_pred, z_next_det)
                optim_e2_a.zero_grad()
                loss.backward()
                optim_e2_a.step()

            # P2 eval: accumulate (pred, target) for forward_r2.
            if phase_is_p2 and has_e2_harm_a and prev_z_harm_a is not None and latent.z_harm_a is not None:
                with torch.no_grad():
                    z_pred_eval = agent.e2_harm_a(
                        prev_z_harm_a.detach(), action.detach()
                    )
                    forward_r2_eval_pairs.append(
                        (z_pred_eval.detach().cpu(), latent.z_harm_a.detach().cpu())
                    )

            prev_z_harm_a = latent.z_harm_a.detach().clone() if latent.z_harm_a is not None else None

            if done:
                break

        harm_per_ep.append(ep_harm)

    # Metrics
    if forward_r2_eval_pairs:
        preds = torch.cat([p for p, _ in forward_r2_eval_pairs], dim=0)
        targets = torch.cat([t for _, t in forward_r2_eval_pairs], dim=0)
        ss_res = float(((targets - preds) ** 2).sum().item())
        ss_tot = float(((targets - targets.mean(dim=0)) ** 2).sum().item())
        harm_a_forward_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    else:
        harm_a_forward_r2 = float("nan")

    action_entropy = _entropy(action_counts)
    mean_bias_abs = (score_bias_abs_sum / score_bias_count) if score_bias_count > 0 else 0.0
    eval_harm_rate = sum(harm_per_ep[-P2_EPS:]) / max(1, len(harm_per_ep[-P2_EPS:]))

    result = {
        "seed": seed,
        "condition": condition,
        "harm_a_forward_r2": float(harm_a_forward_r2) if not math.isnan(harm_a_forward_r2) else None,
        "action_class_entropy": float(action_entropy),
        "mean_score_bias_abs": float(mean_bias_abs),
        "eval_harm_rate": float(eval_harm_rate),
        "action_counts": {str(k): v for k, v in action_counts.items()},
        "n_forward_r2_samples": len(forward_r2_eval_pairs),
    }

    if verbose:
        r2_str = f"{harm_a_forward_r2:.3f}" if not math.isnan(harm_a_forward_r2) else "n/a"
        print(
            f"  [seed={seed} {condition}] "
            f"forward_r2={r2_str} "
            f"entropy={action_entropy:.3f} "
            f"mean_bias_abs={mean_bias_abs:.4f} "
            f"eval_harm={eval_harm_rate:.4f}",
            flush=True,
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke: seed=42, tiny P0=3/P1=5/P2=3")
        global P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP
        P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 3, 5, 3, 20
        for cond in CONDITIONS:
            print(f"cond={cond}")
            r = _run_condition(seed=42, condition=cond, verbose=False)
            print(f"  {cond}: {r}")
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
        print(f"Seed {seed}")
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            r = _run_condition(seed=seed, condition=cond)
            all_results.append(r)

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    off = by_cond("OFF")
    indep = by_cond("ON_INDEPENDENT")
    shared = by_cond("ON_SHARED")

    # C1: both ON arms achieve r2 >= 0.3 in >=2/3 seeds.
    c1_indep_ok = sum(
        1 for r in indep if (r["harm_a_forward_r2"] or 0.0) >= 0.3
    )
    c1_shared_ok = sum(
        1 for r in shared if (r["harm_a_forward_r2"] or 0.0) >= 0.3
    )
    c1 = (c1_indep_ok >= 2) and (c1_shared_ok >= 2)

    # C2: |entropy_on - entropy_off| >= 0.1 nats in either ON arm, >=2/3 seeds.
    def entropy_delta(on_runs, off_runs):
        wins = 0
        for on_r, off_r in zip(on_runs, off_runs):
            if abs(on_r["action_class_entropy"] - off_r["action_class_entropy"]) >= 0.1:
                wins += 1
        return wins

    c2_indep_wins = entropy_delta(indep, off)
    c2_shared_wins = entropy_delta(shared, off)
    c2 = (c2_indep_wins >= 2) or (c2_shared_wins >= 2)

    # C3: ON entropy >= OFF entropy in >=2/3 seeds (across both ON arms).
    def entropy_not_collapsed(on_runs, off_runs):
        wins = 0
        for on_r, off_r in zip(on_runs, off_runs):
            if on_r["action_class_entropy"] >= off_r["action_class_entropy"]:
                wins += 1
        return wins

    c3_indep_wins = entropy_not_collapsed(indep, off)
    c3_shared_wins = entropy_not_collapsed(shared, off)
    c3 = (c3_indep_wins >= 2) and (c3_shared_wins >= 2)

    # C4 diagnostic (ARC-033 vs ARC-058 arbitration -- never fails run).
    def mean_r2(runs):
        vals = [r["harm_a_forward_r2"] for r in runs if r["harm_a_forward_r2"] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    mean_r2_indep = mean_r2(indep)
    mean_r2_shared = mean_r2(shared)
    c4_winner = "ARC-033_independent" if mean_r2_indep > mean_r2_shared else "ARC-058_shared"

    outcome = "PASS" if (c1 and c2 and c3) else "FAIL"

    summary = {
        "c1_forward_r2_substrate_ok": {
            "indep_pass_count": c1_indep_ok,
            "shared_pass_count": c1_shared_ok,
            "threshold_r2": 0.3,
            "pass": c1,
            "desc": "both ON arms harm_a_forward_r2 >= 0.3 in >=2/3 seeds",
        },
        "c2_dacc_behavioural_effect": {
            "indep_wins": c2_indep_wins,
            "shared_wins": c2_shared_wins,
            "pass": c2,
            "desc": "|entropy_on - entropy_off| >= 0.1 in either ON arm, >=2/3 seeds",
        },
        "c3_mech260_no_collapse": {
            "indep_wins": c3_indep_wins,
            "shared_wins": c3_shared_wins,
            "pass": c3,
            "desc": "ON entropy >= OFF entropy in >=2/3 seeds for both arms",
        },
        "c4_arc033_vs_arc058_diagnostic": {
            "mean_r2_independent": mean_r2_indep,
            "mean_r2_shared": mean_r2_shared,
            "winner_suggested_by_forward_r2": c4_winner,
            "desc": "diagnostic only -- informs governance arbitration, never fails run",
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    per_claim = {
        "SD-032b": "supports" if (c1 and c2) else ("mixed" if (c1 or c2) else "weakens"),
        "MECH-258": "supports" if c1 else "weakens",
        "MECH-260": "supports" if c3 else ("mixed" if c2 else "weakens"),
        "ARC-033": "supports" if (c1_indep_ok >= 2) else "weakens",
        "ARC-058": "supports" if (c1_shared_ok >= 2) else "weakens",
    }

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
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}")


if __name__ == "__main__":
    main()
