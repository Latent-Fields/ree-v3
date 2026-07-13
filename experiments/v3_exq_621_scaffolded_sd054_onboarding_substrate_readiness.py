"""
V3-EXQ-621: scaffolded_sd054_onboarding substrate-readiness diagnostic.

Substrate landed: 2026-05-31 via /implement-substrate-scaffolded-sd054-onboarding session
(IGW-20260531-029). Plan-of-record memo:
REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md.

4-arm design per memo Acceptance section:
    ARM_0 ALL_OFF_baseline                   -- 603c-style flat training (master OFF)
    ARM_1 SCAFFOLD_ONLY                      -- scaffolded P0 spawn ON; anneal endpoints at memo defaults
                                                but goal pipeline kept frozen across P1 -> P2 boundary
                                                (isolates the spatial-scaffolding effect alone)
    ARM_2 SCAFFOLD_AND_ANNEAL                -- full scheduler with default anneal endpoints
                                                (the substrate-design memo's primary path)
    ARM_3 SCAFFOLD_AND_ANNEAL_CONTROL_FROM_SCRATCH
                                              -- same as ARM_2 but agent reset per-cell (no
                                                 cross-cell weight reuse). Pins the substrate
                                                 effect to the scheduler design, not to
                                                 incidental warm-start carry-over.

3 seeds x 4 arms = 12 cells. P0=30 / P1=30 / P2=30 episodes x 200 steps default; reduced to
2/2/1 x 20 with --dry-run for smoke verification.

Acceptance (per memo section "Acceptance criteria"):
    C1 substrate-readiness (cells complete)
        P0+P1+P2 completes on >= total_cells / 2 (arm, seed) cells without hitting the Fix D
        survival gate. PASS = the scaffolded onboarding lets the agent survive the target env
        after the curriculum, FAIL = the scaffold is insufficient.
    C2 z_goal materially nonzero
        z_goal_norm_peak in P2 measurement cells achieves >= 0.1 on at least 2 of 3 seeds in
        at least one of ARM_1 / ARM_2 / ARM_3 (not ARM_0 baseline).
    C3 goal pipeline behaviourally consequential
        At least one of:
          - approach_commit_rate lift >= 0.10 (scaffolded > ARM_0 baseline)
          - bridge_cue_fires mean per episode >= 2 (scaffolded; ARM_0 expected near zero)
          - dacc_bias_nonzero_steps >= 1 per episode mean (scaffolded; ARM_0 expected zero)

Overall PASS = C1 AND (C2 OR C3). C2 OR C3 captures the two routes the substrate fix could
succeed: either z_goal_norm itself crosses the floor (direct probe) or the cascade fires
behaviourally even at modest z_goal_norm (the 493-anchored isolation result extending to
cascade). Joint pass would be ideal but is not required.

claim_ids = [Q-045, MECH-313, MECH-260] -- inherits the V3-EXQ-603c claim trio per the
substrate-design memo's "What does NOT change" disclaimer (the substrate addresses prereq (2)
of GAP-C; cleared prereq -> 603c successor becomes queueable as V3-EXQ-603d). The 591 family
(ARC-046) is transitively unblocked but NOT in claim_ids -- ARC-046 has its own pending
prereqs and its retest is V3-EXQ-591b not this experiment.

experiment_purpose = "diagnostic" (substrate-readiness; scoring-excluded from governance per
Phase-3 rules until V3-EXQ-603d behavioural successor runs).

Architecture epoch: ree_hybrid_guardrails_v1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# experiments package
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome

from experiments.scaffolded_sd054_onboarding import (
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_621_scaffolded_sd054_onboarding_substrate_readiness"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["Q-045", "MECH-313", "MECH-260"]
SEEDS = [42, 43, 44]
ARMS = ("ARM_0_ALL_OFF", "ARM_1_SCAFFOLD_ONLY", "ARM_2_SCAFFOLD_AND_ANNEAL", "ARM_3_SCAFFOLD_AND_ANNEAL_FROM_SCRATCH")


@dataclass
class CellResult:
    arm: str
    seed: int
    cell_completed: bool
    p0_n_episodes: int
    p0_mean_episode_length: float
    p1_n_episodes: int
    p1_median_last_window: float
    p1_survival_passed: bool
    p2_n_episodes: int
    z_goal_norm_peak_max: float
    approach_commit_rate: float
    bridge_cue_fires_total: int
    dacc_bias_nonzero_total: int
    mean_p2_episode_length: float
    abort_reason: str = ""


def build_agent(seed: int, device: torch.device, world_obs_dim: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = REEConfig.from_dims(
        world_obs_dim=world_obs_dim,
        body_obs_dim=17,
        harm_obs_dim=50,
        action_dim=5,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
    )
    agent = REEAgent(cfg).to(device)
    return agent


def make_scheduler_cfg(arm: str, p0_eps: int, p1_eps: int, p2_eps: int, steps_per_ep: int):
    """Build the scheduler config per arm."""
    if arm == "ARM_0_ALL_OFF":
        return ScaffoldedSD054OnboardingConfig(
            use_scaffolded_sd054_onboarding_scheduler=False,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_p0_episode_budget=p0_eps,
        scaffold_p1_episode_budget=p1_eps,
        scaffold_p2_episode_budget=p2_eps,
        scaffold_steps_per_episode=steps_per_ep,
        scaffold_p1_survival_gate_steps=75,
    )
    if arm == "ARM_1_SCAFFOLD_ONLY":
        # Hold goal-pipeline gates at the silent end of the anneal (drive_floor stays high,
        # z_beta_threshold stays high) so the spatial scaffold runs but the cascade does not
        # engage. Isolates the spatial-scaffolding contribution.
        cfg.scaffold_p1_anneal_mech295_min_drive_to_fire_min = 1.0
        cfg.scaffold_p1_anneal_mech295_min_drive_to_fire_max = 1.0
        cfg.scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_min = 0.6
        cfg.scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_max = 0.6
    return cfg


def baseline_flat_training(agent, device, steps_per_ep: int, episodes: int) -> Tuple[int, List[float]]:
    """
    ARM_0 ALL_OFF baseline: 603c-style flat random-policy training on the target env without
    the scaffolded onboarding scheduler. Returns (cells_attempted, per-episode lengths).
    """
    env = CausalGridWorldV2(
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_food_attraction=0.7,
        proximity_harm_scale=0.1,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
    )
    lengths = []
    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        for step in range(steps_per_ep):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, _harm, done, _, obs_dict = env.step(action_idx)
            if done:
                lengths.append(step + 1)
                break
        else:
            lengths.append(steps_per_ep)
    return len(lengths), lengths


def measure_arm0_baseline(agent, device, p2_eps: int, steps_per_ep: int) -> Dict[str, Any]:
    """ARM_0 P2 measurement on the target env after flat training."""
    env = CausalGridWorldV2(
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_food_attraction=0.7,
        proximity_harm_scale=0.1,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
    )
    z_goal_peak_max = 0.0
    approach_commit_steps = 0
    bridge_baseline = 0
    dacc_bias_nonzero_local = 0
    if hasattr(agent, "mech295_bridge") and agent.mech295_bridge is not None:
        bridge_baseline = int(getattr(agent.mech295_bridge, "_n_cue_fires", 0))
    total_steps = 0
    for ep in range(p2_eps):
        _, obs_dict = env.reset()
        agent.reset()
        ep_len = 0
        for step in range(steps_per_ep):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
            gs = getattr(agent, "goal_state", None)
            if gs is not None:
                try:
                    cur = float(gs.goal_norm())
                except TypeError:
                    cur = float(gs.goal_norm)
                if cur > z_goal_peak_max:
                    z_goal_peak_max = cur
            bg = getattr(agent, "beta_gate", None)
            if bg is not None and getattr(bg, "is_elevated", False):
                approach_commit_steps += 1
            dacc = getattr(agent, "dacc", None)
            if dacc is not None:
                bundle = getattr(dacc, "_last_bundle", None)
                if bundle is not None:
                    sb = bundle.get("mode_ev") or bundle.get("harm_interaction")
                    if sb is not None:
                        try:
                            if float(torch.as_tensor(sb).norm().item()) > 1e-6:
                                dacc_bias_nonzero_local += 1
                        except Exception:
                            pass
            action_idx = int(action.argmax(dim=-1).item())
            _, _h, done, _, obs_dict = env.step(action_idx)
            ep_len = step + 1
            if done:
                break
        total_steps += ep_len
    bridge_final = 0
    if hasattr(agent, "mech295_bridge") and agent.mech295_bridge is not None:
        bridge_final = int(getattr(agent.mech295_bridge, "_n_cue_fires", 0))
    rate = float(approach_commit_steps) / float(total_steps) if total_steps else 0.0
    return {
        "n_episodes": p2_eps,
        "z_goal_norm_peak_max": z_goal_peak_max,
        "approach_commit_rate": rate,
        "bridge_cue_fires": bridge_final - bridge_baseline,
        "dacc_bias_nonzero_steps": dacc_bias_nonzero_local,
        "mean_episode_length": float(total_steps) / float(p2_eps) if p2_eps else 0.0,
    }


def run_cell(arm: str, seed: int, device, world_obs_dim: int, p0_eps: int, p1_eps: int, p2_eps: int, steps_per_ep: int) -> CellResult:
    agent = build_agent(seed, device, world_obs_dim)
    sched_cfg = make_scheduler_cfg(arm, p0_eps, p1_eps, p2_eps, steps_per_ep)
    scheduler = ScaffoldedSD054OnboardingScheduler(sched_cfg)

    if arm == "ARM_0_ALL_OFF":
        # Flat training on the target env (no scheduler).
        n_eps_done, lengths = baseline_flat_training(agent, device, steps_per_ep, p0_eps + p1_eps)
        measurement = measure_arm0_baseline(agent, device, p2_eps, steps_per_ep)
        cell_completed = True  # baseline always "completes" -- the measurement IS the comparison
        return CellResult(
            arm=arm,
            seed=seed,
            cell_completed=cell_completed,
            p0_n_episodes=n_eps_done,
            p0_mean_episode_length=float(np.mean(lengths)) if lengths else 0.0,
            p1_n_episodes=0,
            p1_median_last_window=0.0,
            p1_survival_passed=True,  # not applicable to baseline
            p2_n_episodes=measurement["n_episodes"],
            z_goal_norm_peak_max=measurement["z_goal_norm_peak_max"],
            approach_commit_rate=measurement["approach_commit_rate"],
            bridge_cue_fires_total=measurement["bridge_cue_fires"],
            dacc_bias_nonzero_total=measurement["dacc_bias_nonzero_steps"],
            mean_p2_episode_length=measurement["mean_episode_length"],
        )

    p0 = scheduler.run_p0(agent, device)
    if p0.aborted:
        return CellResult(
            arm=arm, seed=seed, cell_completed=False,
            p0_n_episodes=p0.n_episodes, p0_mean_episode_length=p0.mean_episode_length,
            p1_n_episodes=0, p1_median_last_window=0.0, p1_survival_passed=False,
            p2_n_episodes=0, z_goal_norm_peak_max=0.0, approach_commit_rate=0.0,
            bridge_cue_fires_total=0, dacc_bias_nonzero_total=0,
            mean_p2_episode_length=0.0, abort_reason=p0.abort_reason or "p0_aborted",
        )
    p1 = scheduler.run_p1(agent, device)
    if not p1.survival_gate_passed:
        return CellResult(
            arm=arm, seed=seed, cell_completed=False,
            p0_n_episodes=p0.n_episodes, p0_mean_episode_length=p0.mean_episode_length,
            p1_n_episodes=p1.n_episodes, p1_median_last_window=p1.median_last_window_episode_length,
            p1_survival_passed=False, p2_n_episodes=0,
            z_goal_norm_peak_max=0.0, approach_commit_rate=0.0,
            bridge_cue_fires_total=0, dacc_bias_nonzero_total=0,
            mean_p2_episode_length=0.0, abort_reason="p1_survival_gate_failed",
        )
    metrics = scheduler.run_p2(agent, device)
    return CellResult(
        arm=arm, seed=seed, cell_completed=True,
        p0_n_episodes=p0.n_episodes, p0_mean_episode_length=p0.mean_episode_length,
        p1_n_episodes=p1.n_episodes, p1_median_last_window=p1.median_last_window_episode_length,
        p1_survival_passed=True, p2_n_episodes=metrics.n_episodes,
        z_goal_norm_peak_max=metrics.z_goal_norm_peak_max,
        approach_commit_rate=metrics.approach_commit_rate,
        bridge_cue_fires_total=metrics.bridge_cue_fires,
        dacc_bias_nonzero_total=metrics.dacc_bias_nonzero_steps,
        mean_p2_episode_length=metrics.mean_episode_length,
    )


def evaluate_acceptance(cells: List[CellResult]) -> Dict[str, Any]:
    n_total = len(cells)
    n_completed = sum(1 for c in cells if c.cell_completed)
    c1 = n_completed >= (n_total // 2)

    arm_cells = {arm: [c for c in cells if c.arm == arm] for arm in ARMS}
    arm_baseline = arm_cells.get("ARM_0_ALL_OFF", [])
    baseline_rate = (
        float(np.mean([c.approach_commit_rate for c in arm_baseline])) if arm_baseline else 0.0
    )

    c2_any_arm = False
    c2_arm = ""
    for arm in ("ARM_1_SCAFFOLD_ONLY", "ARM_2_SCAFFOLD_AND_ANNEAL", "ARM_3_SCAFFOLD_AND_ANNEAL_FROM_SCRATCH"):
        a_cells = [c for c in arm_cells.get(arm, []) if c.cell_completed]
        if len(a_cells) >= 2 and sum(1 for c in a_cells if c.z_goal_norm_peak_max >= 0.1) >= 2:
            c2_any_arm = True
            c2_arm = arm
            break

    c3_any_arm = False
    c3_reason = ""
    for arm in ("ARM_1_SCAFFOLD_ONLY", "ARM_2_SCAFFOLD_AND_ANNEAL", "ARM_3_SCAFFOLD_AND_ANNEAL_FROM_SCRATCH"):
        a_cells = [c for c in arm_cells.get(arm, []) if c.cell_completed]
        if not a_cells:
            continue
        rates = [c.approach_commit_rate for c in a_cells]
        bridges = [c.bridge_cue_fires_total / max(1, c.p2_n_episodes) for c in a_cells]
        daccs = [c.dacc_bias_nonzero_total / max(1, c.p2_n_episodes) for c in a_cells]
        if float(np.mean(rates)) - baseline_rate >= 0.10:
            c3_any_arm = True
            c3_reason = f"approach_commit_lift in {arm}"
            break
        if float(np.mean(bridges)) >= 2.0:
            c3_any_arm = True
            c3_reason = f"bridge_cue_fires_mean in {arm}"
            break
        if float(np.mean(daccs)) >= 1.0:
            c3_any_arm = True
            c3_reason = f"dacc_bias_nonzero_mean in {arm}"
            break

    overall = c1 and (c2_any_arm or c3_any_arm)
    return {
        "C1_cells_completed": c1,
        "C1_n_completed": n_completed,
        "C1_n_total": n_total,
        "C2_z_goal_floor_met": c2_any_arm,
        "C2_arm_passed": c2_arm,
        "C3_cascade_consequential": c3_any_arm,
        "C3_reason": c3_reason,
        "overall_pass": overall,
    }


def emit_manifest(cells: List[CellResult], acceptance: Dict[str, Any], out_dir: Path, dry_run: bool):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_621_scaffolded_sd054_onboarding_substrate_readiness_{ts}_v3"
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    evidence_direction = "supports" if acceptance["overall_pass"] else "non_contributory"

    manifest = {
        "run_id": run_id,
        "queue_id": "V3-EXQ-621",
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "diagnostic",
        "outcome": outcome,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {cid: evidence_direction for cid in CLAIM_IDS},
        "evidence_direction_note": "Substrate-readiness diagnostic for scaffolded_sd054_onboarding (substrate landed 2026-05-31 IGW-20260531-029). PASS clears prereq (2) of behavioral_diversity_isolation:GAP-C; behavioural cluster validation V3-EXQ-603d / 591b queueable. FAIL routes to a re-triage; the substrate-design memo Section 'What this memo does NOT do' bullets reserve A1 full-policy-replay-onto-reward-rich-trajectories and A3 hand-coded heuristic-pretrained agents as fallbacks if (A2) does not produce z_goal lift.",
        "scoring_excluded": "diagnostic_probe",
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "acceptance": acceptance,
        "cells": [asdict(c) for c in cells],
        "seeds": SEEDS,
        "arms": list(ARMS),
        "substrate_design_doc": "REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md",
        "triage_memo": "REE_assembly/evidence/planning/z_goal_collapse_triage_2026-05-31.md",
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"manifest written: {out_path}")
    print(f"outcome={outcome} c1={acceptance['C1_cells_completed']} c2={acceptance['C2_z_goal_floor_met']} c3={acceptance['C3_cascade_consequential']}")
    return out_path


def main(args: argparse.Namespace):
    device = torch.device("cpu")
    if args.dry_run:
        p0_eps, p1_eps, p2_eps, steps_per_ep = 2, 2, 1, 20
    else:
        p0_eps, p1_eps, p2_eps, steps_per_ep = 30, 30, 30, 200

    # World obs dim probed from a target env build (matches build_agent assumption).
    probe = CausalGridWorldV2(
        size=12,
        num_hazards=2,
        num_resources=3,
        reef_enabled=True,
        reef_bipartite_layout=True,
        limb_damage_enabled=True,
        seed=0,
    )
    probe.reset()
    world_obs_dim = probe.world_obs_dim

    cells: List[CellResult] = []
    for arm in ARMS:
        for seed in SEEDS:
            print(f"--- {arm} seed={seed} (P0={p0_eps} P1={p1_eps} P2={p2_eps} steps={steps_per_ep}) ---")
            result = run_cell(
                arm, seed, device, world_obs_dim, p0_eps, p1_eps, p2_eps, steps_per_ep
            )
            print(
                f"  completed={result.cell_completed} "
                f"z_goal_peak={result.z_goal_norm_peak_max:.4f} "
                f"approach_rate={result.approach_commit_rate:.3f} "
                f"bridge_fires={result.bridge_cue_fires_total}"
            )
            cells.append(result)

    acceptance = evaluate_acceptance(cells)
    out_dir = Path(args.output_dir)
    if args.dry_run:
        outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
        print(
            f"verdict: dry_run overall={outcome} "
            f"c1={acceptance['C1_cells_completed']} "
            f"c2={acceptance['C2_z_goal_floor_met']} "
            f"c3={acceptance['C3_cascade_consequential']}"
        )
        return outcome, None
    out_path = emit_manifest(cells, acceptance, out_dir, dry_run=False)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    print(f"Overall: {outcome}")
    return outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Reduced-budget smoke run.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
            / EXPERIMENT_TYPE
        ),
        help="Output dir for the flat-JSON manifest.",
    )
    args = parser.parse_args()
    outcome, manifest_path = main(args)
    if not args.dry_run and manifest_path:
        emit_outcome(outcome=outcome, manifest_path=manifest_path)
    sys.exit(0 if outcome == "PASS" else 1)
