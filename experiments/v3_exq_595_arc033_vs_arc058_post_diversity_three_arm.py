#!/opt/local/bin/python3
"""
V3-EXQ-595 -- Post-diversity three-arm ARC-033 vs ARC-058 arbitration.

Claims: ARC-033, ARC-058

self_attribution:GAP-1 resume experiment (NOT a 445-letter iteration). The
445/445b/445h series ran under monostrategy (action_class_entropy=0.0) and
445h dropped the ON_SHARED arm; EXQ-445 + 445b showed bit-identical
ON_INDEPENDENT vs ON_SHARED forward_r2. This script re-issues the three-arm
ablation on the post-ARC-065 main path: REEConfig.from_dims() SP-CEM defaults
(use_support_preserving_cem=True, validated V3-EXQ-567 / main-path landing
2026-05-17).

Conditions (3 seeds each):
  OFF:             use_dacc=False, use_e2_harm_a=False.
  ON_INDEPENDENT:  use_dacc=True, use_e2_harm_a=True,
                   use_shared_harm_trunk=False  (ARC-033 path).
  ON_SHARED:       use_dacc=True, use_e2_harm_a=True,
                   use_shared_harm_trunk=True   (ARC-058 path).

Phased training per ON arm:
  P0 (50 eps): encoder warmup.
  P1 (100 eps): E2_harm_a on .detach()ed z_harm_a targets.
  P2 (30 eps): eval metrics.

Pre-registered acceptance (evidence for ARC-033 vs ARC-058):
  C0 policy diversity: min(action_class_entropy) across ON arms >= 0.10 in
     >=2/3 seeds (monostrategy -> non_contributory substrate ceiling).
  C1 balanced harm events: in ON arms, P2 steps with
     interoceptive_n_agent_caused_harm_events > 0 AND total tagged harm
     steps >= 10 in >=2/3 seeds.
  C2 forward learnability: both ON arms harm_a_forward_r2 >= 0.3 in >=2/3 seeds.
  C3 cross-arm discrimination: in >=2/3 seeds,
     |harm_a_forward_r2_indep - harm_a_forward_r2_shared| > R2_ABS_EPS (1e-5)
     AND relative gap > R2_REL_EPS (0.01).

PASS: C0 AND C1 AND C2 AND C3.

C4 diagnostic (never fails run): governance arbitration tag from mean P2 r2:
  ARC-058 wins if mean_r2_shared >= mean_r2_indep * (1 - ARC058_DEGRADATION_FRAC)
  with ARC058_DEGRADATION_FRAC=0.05 per substrate_queue Q2; else ARC-033.

claim_ids: ["ARC-033", "ARC-058"]
experiment_purpose: "evidence"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

MANIFEST_WRITER_EXEMPT = "archival early-era manifest (non-canonical filename not provably == run_id.json; superseded lineage, not re-run)"

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_TYPE = "v3_exq_595_arc033_vs_arc058_post_diversity_three_arm"
QUEUE_ID = "V3-EXQ-595"
CLAIM_IDS = ["ARC-033", "ARC-058"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES_QUEUE_NOTE = (
    "Fresh three-arm post-SP-CEM arbitration; supersedes the inconclusive "
    "445/445b monostrategy reads and the two-arm 445h template for "
    "self_attribution:GAP-1."
)

SEEDS = [42, 7, 13]
STEPS_PER_EP = 120
P0_EPS = 50
P1_EPS = 100
P2_EPS = 30
TOTAL_TRAINING_EPS = P0_EPS + P1_EPS + P2_EPS

CONDITIONS = ["OFF", "ON_INDEPENDENT", "ON_SHARED"]

# Pre-registered thresholds (not inferred post-hoc).
MIN_POLICY_ENTROPY = 0.10
MIN_FORWARD_R2 = 0.3
R2_ABS_EPS = 1e-5
R2_REL_EPS = 0.01
MIN_TAGGED_HARM_STEPS_P2 = 10
ARC058_DEGRADATION_FRAC = 0.05
SEEDS_PASS_MIN = 2


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=4,
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
        dacc_weight=0.5 if use_dacc else 0.0,
        dacc_interaction_weight=0.3,
        dacc_foraging_weight=0.2,
        dacc_suppression_weight=0.5,
        dacc_suppression_memory=8,
        dacc_precision_scale=5000.0,
        dacc_effort_cost=0.1,
        dacc_drive_coupling=0.0,
        # SP-CEM main-path defaults intentionally omitted (ARC-065 landed 2026-05-17).
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


def _obs_tensors(obs_dict: Dict) -> Tuple:
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = (
        obs_dict["harm_obs_a"].float().unsqueeze(0)
        if "harm_obs_a" in obs_dict
        else None
    )
    harm_hist = (
        obs_dict["harm_history"].float().unsqueeze(0)
        if "harm_history" in obs_dict
        else None
    )
    return body, world, harm, harm_a, harm_hist


def _compute_r2(pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    if not pairs:
        return float("nan")
    preds = torch.cat([p for p, _ in pairs], dim=0)
    targets = torch.cat([t for _, t in pairs], dim=0)
    ss_res = float(((targets - preds) ** 2).sum().item())
    ss_tot = float(((targets - targets.mean(dim=0)) ** 2).sum().item())
    if ss_tot <= 1e-8:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _run_condition(
    seed: int,
    condition: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, condition)
    has_e2 = condition != "OFF" and agent.e2_harm_a is not None
    optim_e2_a = None
    if has_e2:
        optim_e2_a = torch.optim.Adam(agent.e2_harm_a.parameters(), lr=5e-4)

    phase_p1_start = P0_EPS
    phase_p2_start = P0_EPS + P1_EPS

    action_counts: Dict[int, int] = {}
    forward_r2_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    n_agent_caused_p2 = 0
    n_body_noise_p2 = 0
    n_tagged_harm_p2 = 0

    prev_z_harm_a: Optional[torch.Tensor] = None

    for ep_idx in range(TOTAL_TRAINING_EPS):
        agent.reset()
        _obs, obs_dict = env.reset()
        prev_z_harm_a = None

        phase_is_p1 = phase_p1_start <= ep_idx < phase_p2_start
        phase_is_p2 = ep_idx >= phase_p2_start

        if verbose and (ep_idx + 1) % 50 == 0:
            print(
                f"  [train] seed={seed} {condition} ep {ep_idx + 1}/{TOTAL_TRAINING_EPS}",
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
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            a_idx = int(action[0].argmax().item())

            if phase_is_p2:
                action_counts[a_idx] = action_counts.get(a_idx, 0) + 1

            _obs, _harm_signal, done, info, obs_dict = env.step(action)

            if phase_is_p2:
                n_agent = int(info.get("interoceptive_n_agent_caused_harm_events", 0))
                n_body = int(info.get("interoceptive_n_body_noise_events", 0))
                if n_agent > 0:
                    n_agent_caused_p2 += 1
                    n_tagged_harm_p2 += 1
                if n_body > 0:
                    n_body_noise_p2 += 1
                    n_tagged_harm_p2 += 1

            if phase_is_p1 and has_e2 and prev_z_harm_a is not None and latent.z_harm_a is not None:
                z_pred = agent.e2_harm_a(prev_z_harm_a.detach(), action.detach())
                loss = agent.e2_harm_a.compute_loss(z_pred, latent.z_harm_a.detach())
                optim_e2_a.zero_grad()
                loss.backward()
                optim_e2_a.step()

            if phase_is_p2 and has_e2 and prev_z_harm_a is not None and latent.z_harm_a is not None:
                with torch.no_grad():
                    z_pred_eval = agent.e2_harm_a(
                        prev_z_harm_a.detach(), action.detach()
                    )
                    forward_r2_pairs.append(
                        (
                            z_pred_eval.detach().cpu(),
                            latent.z_harm_a.detach().cpu(),
                        )
                    )

            prev_z_harm_a = (
                latent.z_harm_a.detach().clone()
                if latent.z_harm_a is not None
                else None
            )
            if done:
                break

    r2 = _compute_r2(forward_r2_pairs)
    entropy = _entropy(action_counts)
    result = {
        "seed": seed,
        "condition": condition,
        "harm_a_forward_r2": float(r2) if not math.isnan(r2) else None,
        "action_class_entropy": float(entropy),
        "n_agent_caused_harm_p2": n_agent_caused_p2,
        "n_body_noise_p2": n_body_noise_p2,
        "n_tagged_harm_p2": n_tagged_harm_p2,
        "action_counts": {str(k): v for k, v in action_counts.items()},
        "use_shared_harm_trunk": condition == "ON_SHARED",
        "sp_cem_main_path_defaults": True,
    }
    if verbose:
        r2s = f"{r2:.4f}" if not math.isnan(r2) else "n/a"
        print(
            f"  [seed={seed} {condition}] r2={r2s} entropy={entropy:.3f} "
            f"agent_harm_p2={n_agent_caused_p2} body_noise_p2={n_body_noise_p2}",
            flush=True,
        )
    return result


def _by_cond(all_results: List[Dict], cond: str) -> List[Dict]:
    return [r for r in all_results if r["condition"] == cond]


def _evaluate(
    all_results: List[Dict],
) -> Tuple[str, Dict[str, Any], Dict[str, str], str]:
    off = _by_cond(all_results, "OFF")
    indep = _by_cond(all_results, "ON_INDEPENDENT")
    shared = _by_cond(all_results, "ON_SHARED")

    def seed_pass_c0(on_runs: List[Dict]) -> int:
        wins = 0
        for r in on_runs:
            if r["action_class_entropy"] >= MIN_POLICY_ENTROPY:
                wins += 1
        return wins

    c0_indep = seed_pass_c0(indep)
    c0_shared = seed_pass_c0(shared)
    c0 = (c0_indep >= SEEDS_PASS_MIN) and (c0_shared >= SEEDS_PASS_MIN)

    def seed_pass_c1(on_runs: List[Dict]) -> int:
        wins = 0
        for r in on_runs:
            if (
                r["n_agent_caused_harm_p2"] > 0
                and r["n_tagged_harm_p2"] >= MIN_TAGGED_HARM_STEPS_P2
            ):
                wins += 1
        return wins

    c1_indep = seed_pass_c1(indep)
    c1_shared = seed_pass_c1(shared)
    c1 = (c1_indep >= SEEDS_PASS_MIN) and (c1_shared >= SEEDS_PASS_MIN)

    def r2_ok(runs: List[Dict]) -> int:
        return sum(1 for r in runs if (r["harm_a_forward_r2"] or 0.0) >= MIN_FORWARD_R2)

    c1_indep_r2 = r2_ok(indep)
    c1_shared_r2 = r2_ok(shared)
    c2 = (c1_indep_r2 >= SEEDS_PASS_MIN) and (c1_shared_r2 >= SEEDS_PASS_MIN)

    def seed_discriminates() -> int:
        wins = 0
        for ri, rs in zip(indep, shared):
            ri_r2 = ri["harm_a_forward_r2"]
            rs_r2 = rs["harm_a_forward_r2"]
            if ri_r2 is None or rs_r2 is None:
                continue
            abs_gap = abs(ri_r2 - rs_r2)
            denom = max(abs(ri_r2), abs(rs_r2), 1e-8)
            rel_gap = abs_gap / denom
            if abs_gap > R2_ABS_EPS and rel_gap > R2_REL_EPS:
                wins += 1
        return wins

    c3_disc = seed_discriminates()
    c3 = c3_disc >= SEEDS_PASS_MIN

    def mean_r2(runs: List[Dict]) -> float:
        vals = [r["harm_a_forward_r2"] for r in runs if r["harm_a_forward_r2"] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    mean_indep = mean_r2(indep)
    mean_shared = mean_r2(shared)
    if mean_shared >= mean_indep * (1.0 - ARC058_DEGRADATION_FRAC):
        c4_winner = "ARC-058_shared"
    else:
        c4_winner = "ARC-033_independent"

    passed = c0 and c1 and c2 and c3
    outcome = "PASS" if passed else "FAIL"

    summary = {
        "c0_policy_diversity": {
            "indep_pass_count": c0_indep,
            "shared_pass_count": c0_shared,
            "min_entropy": MIN_POLICY_ENTROPY,
            "pass": c0,
        },
        "c1_balanced_harm_events": {
            "indep_pass_count": c1_indep,
            "shared_pass_count": c1_shared,
            "min_tagged_steps": MIN_TAGGED_HARM_STEPS_P2,
            "pass": c1,
        },
        "c2_forward_r2_learnable": {
            "indep_pass_count": c1_indep_r2,
            "shared_pass_count": c1_shared_r2,
            "threshold_r2": MIN_FORWARD_R2,
            "pass": c2,
        },
        "c3_cross_arm_discrimination": {
            "discriminating_seeds": c3_disc,
            "r2_abs_eps": R2_ABS_EPS,
            "r2_rel_eps": R2_REL_EPS,
            "pass": c3,
        },
        "c4_arbitration_diagnostic": {
            "mean_r2_independent": mean_indep,
            "mean_r2_shared": mean_shared,
            "winner_suggested": c4_winner,
            "degradation_frac": ARC058_DEGRADATION_FRAC,
        },
        "off_arm_entropy_mean": (
            sum(r["action_class_entropy"] for r in off) / len(off) if off else 0.0
        ),
    }

    if not passed:
        if not c0:
            arc033_dir = "non_contributory"
            arc058_dir = "non_contributory"
            note = "substrate_ceiling: policy diversity gate failed (monostrategy)"
        elif not c1:
            arc033_dir = "non_contributory"
            arc058_dir = "non_contributory"
            note = "substrate_ceiling: balanced harm-event gate failed"
        elif not c3:
            arc033_dir = "non_contributory"
            arc058_dir = "non_contributory"
            note = "substrate_ceiling: ON_INDEPENDENT vs ON_SHARED bit-identical or indistinguishable r2"
        else:
            arc033_dir = "weakens"
            arc058_dir = "weakens"
            note = "forward_r2 learnability failed"
    else:
        if c4_winner == "ARC-058_shared":
            arc033_dir = "weakens"
            arc058_dir = "supports"
        else:
            arc033_dir = "supports"
            arc058_dir = "weakens"
        note = "contributory arbitration under post-SP-CEM diversity"

    per_claim = {"ARC-033": arc033_dir, "ARC-058": arc058_dir}
    if outcome == "PASS":
        evidence_dir = "supports"
    elif arc033_dir == "non_contributory":
        evidence_dir = "non_contributory"
    else:
        evidence_dir = "weakens"

    summary["evidence_direction_note"] = note
    return outcome, summary, per_claim, evidence_dir


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP, TOTAL_TRAINING_EPS, SEEDS  # noqa: PLW0603

    if dry_run:
        P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 2, 3, 3, 20
        TOTAL_TRAINING_EPS = P0_EPS + P1_EPS + P2_EPS
        SEEDS = [42]
    else:
        TOTAL_TRAINING_EPS = P0_EPS + P1_EPS + P2_EPS

    all_results: List[Dict] = []
    for seed in SEEDS:
        print(f"Seed {seed}", flush=True)
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}", flush=True)
            r = _run_condition(seed=seed, condition=cond, verbose=not dry_run)
            all_results.append(r)
            passed_run = True
            if cond != "OFF":
                passed_run = (
                    r["action_class_entropy"] >= MIN_POLICY_ENTROPY * 0.5
                )
            print(f"verdict: {'PASS' if passed_run else 'FAIL'}", flush=True)

    outcome, summary, per_claim, evidence_dir = _evaluate(all_results)
    print(f"\nOutcome: {outcome}", flush=True)
    for k, v in summary.items():
        if k != "evidence_direction_note":
            print(f"  {k}: {v}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest: Dict[str, Any] = {
        "schema_version": "experiment_pack/v1",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_dir,
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_note": summary.get("evidence_direction_note", ""),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "queue_id": QUEUE_ID,
            "conditions": CONDITIONS,
            "seeds": list(SEEDS),
            "p0_eps": P0_EPS,
            "p1_eps": P1_EPS,
            "p2_eps": P2_EPS,
            "steps_per_ep": STEPS_PER_EP,
            "sp_cem_main_path_defaults": True,
            "supersedes_note": SUPERSEDES_QUEUE_NOTE,
        },
        "dry_run": dry_run,
    }

    manifest_path = None
    if not dry_run:
        out_dir = EVIDENCE_ROOT / EXPERIMENT_TYPE / "runs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {manifest_path}", flush=True)

    return {
        "outcome": outcome,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "run_id": run_id,
        "dry_run": dry_run,
        "all_pass": outcome == "PASS",
    }


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=QUEUE_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(f"{QUEUE_ID} -- post-diversity ARC-033 vs ARC-058 three-arm ablation", flush=True)
    print(SUPERSEDES_QUEUE_NOTE, flush=True)
    return run_experiment(dry_run=args.dry_run)


if __name__ == "__main__":
    result = main()
    if not result["dry_run"]:
        emit_outcome(
            outcome=result["outcome"],
            manifest_path=result["manifest_path"],
            run_id=result["run_id"],
        )
    sys.exit(0 if result["all_pass"] else 1)
