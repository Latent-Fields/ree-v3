"""
V3-EXQ-267: ARC-038 Waking Consolidation Discriminative Evidence

MECHANISM UNDER TEST: ARC-038 (hippocampus.waking_consolidation_mode)
  "During waking immobility, hippocampal replay switches between task-focused
   forward sweeps (planning mode) and local consolidation replay (integration
   mode) based on task demand; this waking consolidation mode is architecturally
   necessary for viability map integration during experience."

EXPERIMENT PURPOSE: evidence

SCIENTIFIC QUESTION:
  Is waking quiescent consolidation replay (MECH-092: _do_replay on e3_quiescent
  cycles, no z_goal target) architecturally necessary for viability map integration?
  Specifically: does removing consolidation replay during waking experience
  produce higher harm rates (worse map generalisation) compared to an agent with
  consolidation replay enabled, across matched seeds?

DESIGN:
  Two conditions, 3 matched seeds each:

    CONSOLIDATION_ENABLED: Standard agent with waking replay active.
      On e3_quiescent ticks: explicit _do_replay() call runs goal-independent
      hippocampal.replay() to integrate trajectory experience into the viability
      map (residue field). This is the waking consolidation mode specified by ARC-038.

    CONSOLIDATION_ABLATED: Same agent, but e3_quiescent replay skipped entirely.
      All other processing (sensing, planning forward sweeps via propose_trajectories,
      action selection, residue accumulation on harm events) is identical.
      The only difference: no waking consolidation replay cycles.

  The key distinction from EXQ-191 (invalidated):
  - Target harm_rate 0.05-0.10 (hazard_harm=0.05, not 0.002 which was unreachable)
  - Tests actual waking consolidation mode (goal-free replay via MECH-092),
    NOT generic weight transfer or schema-transfer speedup
  - ARC-038 depends on MECH-092 which is implemented: _do_replay on quiescent cycles
  - Discrimination: consolidation-enabled vs consolidation-ablated (not primed vs naive)

  Training: 100 episodes total per seed.
    Phase 0 (warmup, ep 0-49): encoder + prediction loss training. Residue accumulates
      on harm events (harm_signal < 0). Both conditions accumulate residue identically.
    Phase 1 (evaluation, ep 50-99): same training, but report harm_rate separately.
      Primary metric: Phase 1 harm_rate (lower = better viability map generalisation).

  Environment: CausalGridWorldV2 with:
    hazard_harm=0.05 (target: harm_rate ~ 0.05-0.10 range, reachable)
    num_hazards=4, size=10, use_proxy_fields=True
    resource_respawn_on_consume=True (continued drive cycling)

PRE-REGISTERED THRESHOLDS:
  C1 (primary): In CONSOLIDATION_ENABLED, Phase 1 harm_rate < Phase 0 harm_rate
      in >= 2/3 seeds (consolidation improves map generalisation over training).
  C2 (primary): CONSOLIDATION_ENABLED Phase 1 harm_rate < CONSOLIDATION_ABLATED
      Phase 1 harm_rate by >=15% relative in >= 2/3 seeds
      (consolidation necessary: ablation degrades generalisation).
  C3 (secondary): CONSOLIDATION_ENABLED active_centers >= CONSOLIDATION_ABLATED
      active_centers by >=10% relative in >= 2/3 seeds at Phase 1 end
      (consolidation integrates more of the viability map).
  C4 (data quality): Both conditions, all seeds: total_residue > 0
      (harm events occurred and residue accumulated -- experiment is informative).

PASS: C1 AND C2 AND C4
  C3 is secondary (map coverage proxy).

Note: EXQ-191 was non_contributory (target_harm_rate=0.002 unreachable; tested
generic weight transfer not waking-consolidation mode). This experiment is the
redesign specified in the EXQ-191 post-mortem.

claim_ids: ["ARC-038"]
experiment_purpose: "evidence"
dispatch_mode: "discriminative_pair"
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


# ===== EXPERIMENT CONSTANTS =====
EXPERIMENT_TYPE = "v3_exq_267_arc038_waking_consolidation_discriminative"
CLAIM_IDS = ["ARC-038"]
EXPERIMENT_PURPOSE = "evidence"

WARMUP_EPISODES = 50          # Phase 0 (encoder warmup + residue accumulation)
EVAL_EPISODES = 50            # Phase 1 (evaluation window for primary metrics)
TOTAL_EPISODES = WARMUP_EPISODES + EVAL_EPISODES
STEPS_PER_EPISODE = 200       # Steps per episode
NUM_SEEDS = 3

# PRE-REGISTERED THRESHOLDS
HARM_RATE_IMPROVEMENT_REQ = 0.15   # C2: ENABLED must beat ABLATED by >=15% relative
MIN_CENTER_IMPROVEMENT = 0.10       # C3: ENABLED active_centers >= ABLATED by >=10%

# Environment parameters (target harm_rate ~0.05-0.10)
ENV_PARAMS = dict(
    size=10,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    contaminated_harm=0.04,
    resource_benefit=0.3,
    use_proxy_fields=True,
    resource_respawn_on_consume=True,
    proximity_harm_scale=0.02,
    proximity_benefit_scale=0.02,
)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_PARAMS)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Standard V3 agent with harm stream and world encoder."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,    # SD-008: event-responsive world encoding
        alpha_self=0.3,
        use_harm_stream=True,   # SD-010: z_harm nociceptive stream
    )
    return REEAgent(cfg)


def _get_residue_stats(agent: REEAgent) -> Dict[str, float]:
    """Extract viability map statistics from the residue field."""
    stats = agent.get_residue_statistics()
    return {
        "active_centers": float(stats.get("active_centers", torch.tensor(0)).item()),
        "total_residue": float(stats.get("total_residue", torch.tensor(0.0)).item()),
        "num_harm_events": float(stats.get("num_harm_events", torch.tensor(0)).item()),
        "mean_weight": float(stats.get("mean_weight", torch.tensor(0.0)).item()),
    }


def run_condition(
    condition_name: str,
    seed: int,
    consolidation_enabled: bool,
    dry_run: bool = False,
) -> Dict:
    """
    Run one condition x seed.

    consolidation_enabled=True: waking quiescent replay (_do_replay) active
    consolidation_enabled=False: replay skipped (ablation of MECH-092)
    """
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env)
    device = agent.device

    # Phased training: encoder + world predictor
    # Phase 0 warms up encoding before we measure harm_rate in Phase 1
    optimizer = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-4,
    )

    phase0_harm_rates: List[float] = []
    phase1_harm_rates: List[float] = []

    n_episodes = 2 if dry_run else TOTAL_EPISODES
    n_steps = 5 if dry_run else STEPS_PER_EPISODE
    n_warmup = 1 if dry_run else WARMUP_EPISODES

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        ep_harm = 0.0
        ep_steps = 0

        for step in range(n_steps):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            obs_harm  = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = obs_harm.to(device)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else \
                torch.zeros(1, agent.config.latent.world_dim, device=device)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            # ARC-038: waking consolidation replay on quiescent E3 cycles.
            # CONSOLIDATION_ENABLED: call _do_replay (goal-free, map integration).
            # CONSOLIDATION_ABLATED: skip replay entirely (ablation condition).
            if consolidation_enabled and ticks.get("e3_quiescent", False):
                agent._do_replay(latent)

            _, harm_signal, done, info, obs_dict = env.step(action)
            ep_harm += max(0.0, float(-harm_signal))

            # Accumulate residue on harm events (both conditions identical)
            if harm_signal < 0:
                agent.update_residue(
                    harm_signal=harm_signal,
                    hypothesis_tag=False,
                    owned=True,
                )

            # Training: prediction loss (phased -- encoder first)
            optimizer.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

            ep_steps += 1
            if done:
                break

        harm_rate = ep_harm / max(1, ep_steps)

        if ep < n_warmup:
            phase0_harm_rates.append(harm_rate)
        else:
            phase1_harm_rates.append(harm_rate)

        if (ep + 1) % 10 == 0 or dry_run:
            phase_label = "P0" if ep < n_warmup else "P1"
            print(f"  Ep {ep+1}/{n_episodes} [{condition_name} seed={seed} {phase_label}]"
                  f" harm_rate={harm_rate:.4f}")

    # Final residue field statistics
    residue_stats = _get_residue_stats(agent)

    mean_phase0 = float(sum(phase0_harm_rates) / max(1, len(phase0_harm_rates)))
    mean_phase1 = float(sum(phase1_harm_rates) / max(1, len(phase1_harm_rates)))

    verdict = "PASS" if mean_phase1 < mean_phase0 else "FAIL"
    print(f"  -- [{condition_name} seed={seed}] Phase0={mean_phase0:.4f}"
          f" Phase1={mean_phase1:.4f} active_centers={residue_stats['active_centers']:.0f}"
          f" verdict:{verdict}")

    return {
        "condition": condition_name,
        "seed": seed,
        "consolidation_enabled": consolidation_enabled,
        "mean_phase0_harm_rate": mean_phase0,
        "mean_phase1_harm_rate": mean_phase1,
        "residue_active_centers": residue_stats["active_centers"],
        "residue_total": residue_stats["total_residue"],
        "residue_num_harm_events": residue_stats["num_harm_events"],
        "residue_mean_weight": residue_stats["mean_weight"],
        "phase0_harm_rates": phase0_harm_rates,
        "phase1_harm_rates": phase1_harm_rates,
    }


def main(dry_run: bool = False):
    print("[EXQ-267] ARC-038 Waking Consolidation Discriminative Pair")
    print("=" * 60)
    print(f"Conditions: CONSOLIDATION_ENABLED vs CONSOLIDATION_ABLATED")
    print(f"Seeds: {NUM_SEEDS}, Episodes: {TOTAL_EPISODES}, Steps/ep: {STEPS_PER_EPISODE}")
    print(f"Harm threshold: hazard_harm={ENV_PARAMS['hazard_harm']}"
          f" (target harm_rate ~0.05-0.10)")
    print("=" * 60)

    enabled_results: List[Dict] = []
    ablated_results: List[Dict] = []

    seeds = [42 + i * 13 for i in range(NUM_SEEDS)]

    # Run matched seed pairs
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        print(f"  [CONSOLIDATION_ENABLED]")
        res_en = run_condition("CONSOLIDATION_ENABLED", seed,
                               consolidation_enabled=True, dry_run=dry_run)
        enabled_results.append(res_en)

        print(f"  [CONSOLIDATION_ABLATED]")
        res_ab = run_condition("CONSOLIDATION_ABLATED", seed,
                               consolidation_enabled=False, dry_run=dry_run)
        ablated_results.append(res_ab)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Per-seed pairwise comparisons
    c1_wins = 0   # enabled: Phase1 < Phase0
    c2_wins = 0   # enabled Phase1 < ablated Phase1 by >=15% relative
    c3_wins = 0   # enabled active_centers >= ablated by >=10% relative
    c4_ok_all = True  # all residue > 0

    seed_details = []
    for en_r, ab_r in zip(enabled_results, ablated_results):
        seed = en_r["seed"]
        en_p0 = en_r["mean_phase0_harm_rate"]
        en_p1 = en_r["mean_phase1_harm_rate"]
        ab_p1 = ab_r["mean_phase1_harm_rate"]
        en_ac = en_r["residue_active_centers"]
        ab_ac = ab_r["residue_active_centers"]

        # C1: enabled improves over warmup
        c1_seed = en_p1 < en_p0
        if c1_seed:
            c1_wins += 1

        # C2: enabled beats ablated by >=15% relative
        if ab_p1 > 0:
            rel_improvement = (ab_p1 - en_p1) / ab_p1
        else:
            rel_improvement = 0.0
        c2_seed = rel_improvement >= HARM_RATE_IMPROVEMENT_REQ
        if c2_seed:
            c2_wins += 1

        # C3: enabled has more active_centers
        if ab_ac > 0:
            rel_centers = (en_ac - ab_ac) / ab_ac
        else:
            rel_centers = float(en_ac > 0)
        c3_seed = rel_centers >= MIN_CENTER_IMPROVEMENT
        if c3_seed:
            c3_wins += 1

        # C4: data quality
        if en_r["residue_total"] <= 0 or ab_r["residue_total"] <= 0:
            c4_ok_all = False

        print(f"Seed {seed}:"
              f" EN p0={en_p0:.4f} p1={en_p1:.4f}"
              f" AB p1={ab_p1:.4f}"
              f" rel_improv={rel_improvement:.3f}"
              f" C1={c1_seed} C2={c2_seed} C3={c3_seed}")

        seed_details.append({
            "seed": seed,
            "enabled_phase0_harm_rate": en_p0,
            "enabled_phase1_harm_rate": en_p1,
            "ablated_phase1_harm_rate": ab_p1,
            "relative_improvement": rel_improvement,
            "enabled_active_centers": en_ac,
            "ablated_active_centers": ab_ac,
            "center_rel_improvement": rel_centers,
            "c1_pass": c1_seed,
            "c2_pass": c2_seed,
            "c3_pass": c3_seed,
        })

    # Aggregate
    def _mean_list(lst, key):
        vals = [r[key] for r in lst]
        return float(sum(vals) / len(vals)) if vals else 0.0

    agg_en = {
        "mean_phase0_harm_rate": _mean_list(enabled_results, "mean_phase0_harm_rate"),
        "mean_phase1_harm_rate": _mean_list(enabled_results, "mean_phase1_harm_rate"),
        "mean_active_centers": _mean_list(enabled_results, "residue_active_centers"),
        "mean_total_residue": _mean_list(enabled_results, "residue_total"),
    }
    agg_ab = {
        "mean_phase0_harm_rate": _mean_list(ablated_results, "mean_phase0_harm_rate"),
        "mean_phase1_harm_rate": _mean_list(ablated_results, "mean_phase1_harm_rate"),
        "mean_active_centers": _mean_list(ablated_results, "residue_active_centers"),
        "mean_total_residue": _mean_list(ablated_results, "residue_total"),
    }

    # Acceptance checks
    c1_pass = c1_wins >= 2   # enabled improves in >=2/3 seeds
    c2_pass = c2_wins >= 2   # enabled beats ablated by >=15% in >=2/3 seeds
    c3_pass = c3_wins >= 2   # enabled has more centers in >=2/3 seeds
    c4_pass = c4_ok_all

    outcome = "PASS" if (c1_pass and c2_pass and c4_pass) else "FAIL"

    print(f"\nC1 (enabled improves over warmup, >=2/3 seeds): {c1_pass} ({c1_wins}/3)")
    print(f"C2 (enabled beats ablated >=15%, >=2/3 seeds): {c2_pass} ({c2_wins}/3)")
    print(f"C3 (enabled has more centers >=10%, >=2/3 seeds): {c3_pass} ({c3_wins}/3)")
    print(f"C4 (data quality: residue>0 all conditions): {c4_pass}")
    print(f"Outcome: {outcome}")
    print(f"  ENABLED agg: p0={agg_en['mean_phase0_harm_rate']:.4f}"
          f" p1={agg_en['mean_phase1_harm_rate']:.4f}"
          f" centers={agg_en['mean_active_centers']:.1f}")
    print(f"  ABLATED agg: p0={agg_ab['mean_phase0_harm_rate']:.4f}"
          f" p1={agg_ab['mean_phase1_harm_rate']:.4f}"
          f" centers={agg_ab['mean_active_centers']:.1f}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "dispatch_mode": "discriminative_pair",
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_class": "waking_consolidation_necessity",
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "summary": (
            f"ARC-038 discriminative pair: waking consolidation replay (MECH-092) "
            f"vs ablation. ENABLED Phase1 harm={agg_en['mean_phase1_harm_rate']:.4f}, "
            f"ABLATED Phase1 harm={agg_ab['mean_phase1_harm_rate']:.4f}. "
            f"C1 (improvement over warmup): {c1_wins}/3 seeds. "
            f"C2 (>={int(HARM_RATE_IMPROVEMENT_REQ*100)}% relative advantage): {c2_wins}/3 seeds. "
            f"Outcome: {outcome}."
        ),
        "pre_registered_thresholds": {
            "C1": "Phase1 harm_rate < Phase0 in >=2/3 enabled seeds",
            "C2": f"Enabled Phase1 harm < Ablated Phase1 by >={int(HARM_RATE_IMPROVEMENT_REQ*100)}% relative in >=2/3 seeds",
            "C3": f"Enabled active_centers >= Ablated by >={int(MIN_CENTER_IMPROVEMENT*100)}% in >=2/3 seeds",
            "C4": "total_residue > 0 in all conditions (data quality)",
        },
        "acceptance_checks": {
            "C1_enabled_improves_over_warmup": c1_pass,
            "C1_wins": c1_wins,
            "C2_enabled_beats_ablated_by_threshold": c2_pass,
            "C2_wins": c2_wins,
            "C3_enabled_more_centers": c3_pass,
            "C3_wins": c3_wins,
            "C4_data_quality_residue_nonzero": c4_pass,
        },
        "aggregated": {
            "CONSOLIDATION_ENABLED": agg_en,
            "CONSOLIDATION_ABLATED": agg_ab,
        },
        "seed_details": seed_details,
        "per_seed_results": {
            "CONSOLIDATION_ENABLED": [
                {k: v for k, v in r.items() if k not in ("phase0_harm_rates", "phase1_harm_rates")}
                for r in enabled_results
            ],
            "CONSOLIDATION_ABLATED": [
                {k: v for k, v in r.items() if k not in ("phase0_harm_rates", "phase1_harm_rates")}
                for r in ablated_results
            ],
        },
        "params": {
            "total_episodes": TOTAL_EPISODES,
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "num_seeds": NUM_SEEDS,
            "env": ENV_PARAMS,
            "harm_rate_improvement_req": HARM_RATE_IMPROVEMENT_REQ,
            "min_center_improvement": MIN_CENTER_IMPROVEMENT,
        },
        "design_note": (
            "Redesign of EXQ-191 (non_contributory). Key changes: (1) hazard_harm=0.05 "
            "so harm_rate is in reachable 0.05-0.10 range; (2) discriminates actual "
            "waking-consolidation mode (goal-free MECH-092 replay) vs no replay, not "
            "generic weight transfer. ARC-038 depends_on ARC-018+ARC-007+MECH-092 -- "
            "all implemented in V3 substrate."
        ),
    }

    if dry_run:
        print("\n[DRY RUN] Skipping file write.")
        print(f"[DRY RUN] Outcome would be: {outcome}")
        return output

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments"
    )
    out_path = os.path.join(out_dir, f"{EXPERIMENT_TYPE}_{ts}_v3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to: {out_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V3-EXQ-267 ARC-038 Waking Consolidation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run minimal wiring check (1-2 episodes per condition)")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] V3-EXQ-267: checking ARC-038 wiring...")
        print("  Verifying: _do_replay callable, residue accumulation, harm stream")
        result = main(dry_run=True)
        c4 = result["acceptance_checks"]["C4_data_quality_residue_nonzero"]
        if c4:
            print("[DRY RUN] PASS - wiring OK (residue accumulates on harm events)")
        else:
            print("[DRY RUN] NOTE - no harm events in dry-run (expected with 1 ep, ok)")
        print("[DRY RUN] API wiring verified. Ready to run full experiment.")
        sys.exit(0)

    main(dry_run=False)
