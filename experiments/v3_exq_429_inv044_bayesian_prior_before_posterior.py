#!/opt/local/bin/python3
"""
V3-EXQ-429 -- INV-044: Bayesian Prior-Before-Posterior Necessity

EXPERIMENT_PURPOSE: evidence

CLAIM UNDER TEST: INV-044
  "Approximate Bayesian contextual inference is architecturally impossible to
  co-compute with online encoding: prior construction (schema formation) must
  precede posterior inference (attribution); a system attempting both online
  produces a degenerate prior that makes attribution uninformative regardless
  of training duration."

MECHANISM:
  In REE terms: SWS (run_sws_schema_pass) = slot-formation = prior construction.
  REM (run_rem_attribution_pass) = slot-filling = posterior inference.
  INV-044 predicts: slot diversity (ContextMemory differentiation) should be
  substantially higher in the ORDERED condition (SWS then REM) vs the
  CONCURRENT condition (REM without preceding SWS, i.e., posterior inference
  against a flat/degenerate prior).

DESIGN: Three conditions, 3 seeds each.
  ORDERED:    SWS_THEN_REM -- slot formation (SWS) runs every 10 eps,
              then REM attribution. Correct prior-before-posterior order.
  REM_ONLY:   No SWS, REM-only sleep. Attribution replay against undifferentiated
              slots. Concurrent/degenerate condition -- posterior without prior.
  WAKING_ONLY: No sleep at all. Pure online encoding baseline.

Pre-registered thresholds (discriminative_pair):
  C1: ORDERED mean_sws_slot_diversity > 0.05 in >= 2/3 seeds
      (SWS pass installs differentiated context attractors)
  C2: ORDERED mean_sws_slot_diversity > WAKING_ONLY final_slot_diversity
      in >= 2/3 seeds
      (ordering advantage over no-sleep baseline)
  C3: ORDERED mean_rem_terrain_variance >= REM_ONLY mean_rem_terrain_variance
      in >= 2/3 seeds
      (ordered condition produces more discriminating attribution than unordered)
  C4: REM_ONLY sws_slot_diversity < ORDERED sws_slot_diversity
      in ALL 3 seeds
      (absence of SWS produces less differentiated context representations)

PASS: C1 AND C2 (ordering installs differentiated priors)
  C3 and C4 are secondary confirmatory checks; their failure is informative
  but not a FAIL on the primary invariant.

If PASS: supports INV-044 -- SWS slot-formation builds context priors that
  REM slot-filling requires; omitting SWS leaves attribution uninformative
  because the context representation remains undifferentiated (flat prior).
If FAIL: weakens INV-044 -- slot diversity does not differ between ordered
  and degenerate conditions, suggesting the prior-before-posterior constraint
  may not be binding in the current substrate.

claim_ids: ["INV-044"]
experiment_purpose: "evidence"
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_429_inv044_bayesian_prior_before_posterior"
CLAIM_IDS = ["INV-044"]
EXPERIMENT_PURPOSE = "evidence"

TRAINING_EPISODES = 90
STEPS_PER_EPISODE = 150
SLEEP_INTERVAL = 10       # sleep cycle every N episodes (ORDERED and REM_ONLY)
CONTEXT_SWITCH_EVERY = 5  # alternate SAFE/DANGEROUS every N episodes
NUM_SEEDS = 3
SEEDS = [42, 7, 13]


def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=1,
        num_resources=3,
        hazard_harm=0.02,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=8,
        num_hazards=5,
        num_resources=3,
        hazard_harm=0.04,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, sws_enabled: bool, rem_enabled: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        sws_enabled=sws_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=6,
    )
    return REEAgent(cfg)


def _compute_slot_diversity(agent: REEAgent) -> float:
    """Mean pairwise cosine distance of ContextMemory slots (0=identical, 1=orthogonal)."""
    with torch.no_grad():
        mem = agent.e1.context_memory.memory  # [num_slots, memory_dim]
        n = mem.shape[0]
        if n < 2:
            return 0.0
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        dist = 1.0 - sim[mask]
        return float(dist.mean().item())


def run_condition(condition_name: str, sws_enabled: bool, rem_enabled: bool,
                  seed: int, dry_run: bool = False) -> Dict:
    """Run one condition x seed and return summary metrics."""
    torch.manual_seed(seed)

    ep_count = 2 if dry_run else TRAINING_EPISODES

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_enabled, rem_enabled)
    device = agent.device

    optimizer = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-4,
    )

    sws_diversity_per_cycle: List[float] = []
    rem_terrain_var_per_cycle: List[float] = []
    sws_writes_per_cycle: List[float] = []

    for ep in range(ep_count):
        use_dangerous = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe

        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            _, harm_signal, done, info, obs_dict = env.step(action)

            optimizer.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                pred_loss.backward()
                optimizer.step()

            if done:
                break

        # Sleep cycle every SLEEP_INTERVAL episodes
        should_sleep = (sws_enabled or rem_enabled) and (ep + 1) % SLEEP_INTERVAL == 0 and ep > 0
        if should_sleep:
            sleep_m = agent.run_sleep_cycle()
            sws_div = sleep_m.get("sws_slot_diversity", 0.0)
            sws_n = sleep_m.get("sws_n_writes", 0.0)
            rem_tv = sleep_m.get("rem_terrain_variance", 0.0)
            sws_diversity_per_cycle.append(sws_div)
            sws_writes_per_cycle.append(sws_n)
            rem_terrain_var_per_cycle.append(rem_tv)

        if (ep + 1) % 10 == 0 or dry_run:
            div = _compute_slot_diversity(agent)
            print(f"  ep {ep+1}/{ep_count} [{condition_name} seed={seed}]"
                  f" slot_div={div:.4f}")

    # Final slot diversity at end of training
    final_slot_div = _compute_slot_diversity(agent)

    # For WAKING_ONLY: record final diversity
    if not sws_enabled and not rem_enabled:
        sws_diversity_per_cycle.append(final_slot_div)

    mean_sws_diversity = (
        sum(sws_diversity_per_cycle) / len(sws_diversity_per_cycle)
        if sws_diversity_per_cycle else 0.0
    )
    mean_sws_writes = (
        sum(sws_writes_per_cycle) / len(sws_writes_per_cycle)
        if sws_writes_per_cycle else 0.0
    )
    mean_rem_terrain_var = (
        sum(rem_terrain_var_per_cycle) / len(rem_terrain_var_per_cycle)
        if rem_terrain_var_per_cycle else 0.0
    )

    print(f"  verdict [{condition_name} seed={seed}]:"
          f" sws_div={mean_sws_diversity:.4f}"
          f" final_div={final_slot_div:.4f}"
          f" rem_tv={mean_rem_terrain_var:.4f}"
          f" n_cycles={len(sws_diversity_per_cycle)}")

    return {
        "condition": condition_name,
        "seed": seed,
        "mean_sws_slot_diversity": mean_sws_diversity,
        "final_slot_diversity": final_slot_div,
        "mean_sws_n_writes": mean_sws_writes,
        "mean_rem_terrain_variance": mean_rem_terrain_var,
        "n_sleep_cycles": len(sws_diversity_per_cycle),
        "sws_diversity_series": sws_diversity_per_cycle,
        "rem_terrain_var_series": rem_terrain_var_per_cycle,
    }


def main(dry_run: bool = False):
    conditions = [
        ("ORDERED",     True,  True),   # SWS then REM -- correct ordering
        ("REM_ONLY",    False, True),   # REM without SWS -- degenerate prior
        ("WAKING_ONLY", False, False),  # no sleep -- online baseline
    ]

    all_results: Dict[str, List[Dict]] = {
        "ORDERED": [],
        "REM_ONLY": [],
        "WAKING_ONLY": [],
    }

    for cond_name, sws_en, rem_en in conditions:
        for seed in SEEDS:
            print(f"  ep 0/{TRAINING_EPISODES} [{cond_name} seed={seed}] starting...")
            res = run_condition(cond_name, sws_en, rem_en, seed, dry_run=dry_run)
            all_results[cond_name].append(res)

    def _agg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / len(vals)) if vals else 0.0

    agg = {}
    for cond_name in ["ORDERED", "REM_ONLY", "WAKING_ONLY"]:
        rs = all_results[cond_name]
        agg[cond_name] = {
            "mean_sws_slot_diversity": _agg(rs, "mean_sws_slot_diversity"),
            "mean_final_slot_diversity": _agg(rs, "final_slot_diversity"),
            "mean_sws_n_writes": _agg(rs, "mean_sws_n_writes"),
            "mean_rem_terrain_variance": _agg(rs, "mean_rem_terrain_variance"),
            "mean_n_sleep_cycles": _agg(rs, "n_sleep_cycles"),
        }

    ordered = all_results["ORDERED"]
    rem_only = all_results["REM_ONLY"]
    waking = all_results["WAKING_ONLY"]

    # C1: ORDERED mean_sws_slot_diversity > 0.05 in >= 2/3 seeds
    c1_wins = sum(1 for r in ordered if r["mean_sws_slot_diversity"] > 0.05)
    c1_pass = c1_wins >= 2

    # C2: ORDERED mean_sws_slot_diversity > WAKING_ONLY final_slot_diversity in >= 2/3 seeds
    c2_wins = sum(
        1 for o, w in zip(ordered, waking)
        if o["mean_sws_slot_diversity"] > w["final_slot_diversity"]
    )
    c2_pass = c2_wins >= 2

    # C3: ORDERED rem_terrain_variance >= REM_ONLY rem_terrain_variance in >= 2/3 seeds
    c3_wins = sum(
        1 for o, r in zip(ordered, rem_only)
        if o["mean_rem_terrain_variance"] >= r["mean_rem_terrain_variance"]
    )
    c3_pass = c3_wins >= 2

    # C4: REM_ONLY sws_slot_diversity < ORDERED sws_slot_diversity in ALL seeds
    c4_wins = sum(
        1 for o, r in zip(ordered, rem_only)
        if o["mean_sws_slot_diversity"] > r["mean_sws_slot_diversity"]
    )
    c4_pass = c4_wins == NUM_SEEDS

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    print(f"C1 (ORDERED div>0.05 in 2/3): {c1_pass} ({c1_wins}/3)")
    print(f"C2 (ORDERED div > WAKING baseline in 2/3): {c2_pass} ({c2_wins}/3)")
    print(f"C3 (ORDERED rem_tv >= REM_ONLY in 2/3): {c3_pass} ({c3_wins}/3)")
    print(f"C4 (REM_ONLY div < ORDERED in all): {c4_pass} ({c4_wins}/3)")
    print(f"verdict: {outcome}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "registered_thresholds": {
            "C1": "ORDERED mean_sws_slot_diversity > 0.05 in >= 2/3 seeds",
            "C2": "ORDERED mean_sws_slot_diversity > WAKING final_slot_diversity in >= 2/3 seeds",
            "C3": "ORDERED rem_terrain_variance >= REM_ONLY rem_terrain_variance in >= 2/3 seeds",
            "C4": "REM_ONLY slot_diversity < ORDERED slot_diversity in ALL seeds",
        },
        "acceptance_checks": {
            "C1_ordered_div_gt_threshold_2of3": c1_pass,
            "C2_ordered_div_gt_waking_2of3": c2_pass,
            "C3_ordered_rem_tv_ge_remonly_2of3": c3_pass,
            "C4_remonly_div_lt_ordered_all": c4_pass,
        },
        "aggregated": agg,
        "per_seed_results": {
            "ORDERED": ordered,
            "REM_ONLY": rem_only,
            "WAKING_ONLY": waking,
        },
        "params": {
            "training_episodes": TRAINING_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "num_seeds": NUM_SEEDS,
            "seeds": SEEDS,
            "sws_consolidation_steps": 8,
            "rem_attribution_steps": 6,
        },
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "summary": (
            "INV-044 test: ORDERED (SWS-then-REM) vs REM_ONLY vs WAKING_ONLY. "
            "C1 checks ordered condition installs differentiated context priors "
            "(slot_diversity > 0.05). C2 checks ordering advantage over no-sleep "
            "baseline. PASS = prior construction before posterior inference yields "
            "measurably more differentiated representations, supporting INV-044."
        ),
        "scenario": "Three-arm discriminative probe of prior-before-posterior ordering necessity",
        "interpretation": (
            "PASS supports INV-044: SWS slot-formation builds context attractors that "
            "online encoding and REM-without-SWS cannot build. The degenerate prior "
            "(REM_ONLY) leaves ContextMemory undifferentiated (cosine_sim near 1.0). "
            "FAIL would suggest the ordering constraint is not computationally binding "
            "in the current substrate, weakening the invariant claim."
        ),
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments"
    )
    out_path = os.path.join(out_dir, f"{EXPERIMENT_TYPE}_{ts}_v3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results written to {out_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 episodes per condition to verify wiring")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
