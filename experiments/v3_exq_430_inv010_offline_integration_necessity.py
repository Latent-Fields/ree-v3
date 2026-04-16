#!/opt/local/bin/python3
"""
V3-EXQ-430 -- INV-010: Offline Integration Necessity

experiment_purpose: evidence
claim_ids: ["INV-010"]

CLAIM UNDER TEST: INV-010
  "Offline integration exists and is required."

MECHANISM:
  INV-010 is the foundational sleep-necessity claim in REE architecture.
  With SD-017 implemented (run_sws_schema_pass + run_rem_attribution_pass +
  run_sleep_cycle), the claim is directly testable: does periodic offline
  integration (sleep cycles) produce measurably better behavioral outcomes
  than continuous online learning with no offline pass?

  The causal chain:
    SWS pass -> context memory slot differentiation (prior construction)
    REM pass -> attribution terrain evaluation (posterior replay)
    Together -> reduced harm accumulation and/or increased benefit gain
    in later episodes, because the agent has reorganised its world model.

DESIGN:
  Two conditions x 3 seeds (seeds = [42, 7, 13]).
  SLEEP:    sws_enabled=True, rem_enabled=True.
            Sleep cycle every SLEEP_INTERVAL episodes.
            run_sws_schema_pass() + run_rem_attribution_pass() each cycle.
  NO_SLEEP: sws_enabled=False, rem_enabled=False.
            Continuous online learning, no offline pass.

  Phase structure:
    Phase A (warm-up): episodes 1-20. Both conditions accumulate experience.
    Phase B (consolidation): episodes 21-70. SLEEP condition runs sleep cycles;
      NO_SLEEP continues online.
    Phase C (eval): episodes 71-90. Both conditions tested in the same env.
      Behavioral metrics computed from Phase C only.

  Environment: two-room grid with moderate hazard density.
    Early episodes: agent wanders freely, accumulates varied experience.
    Late episodes: agent is re-seeded to fixed evaluation layout.

Pre-registered thresholds:
  C1: SLEEP mean_late_harm_per_ep < NO_SLEEP mean_late_harm_per_ep
      in >= 2/3 seeds (primary criterion -- harm avoidance)
  C2: SLEEP late_benefit_rate >= NO_SLEEP late_benefit_rate
      in >= 2/3 seeds (secondary -- benefit seeking)
  C3: SLEEP mean_sws_slot_diversity > 0.02 in ALL 3 seeds
      (sanity: sleep cycle actually installed differentiated slots)

PASS: C1 AND C3 (harm reduction + evidence of functional offline pass).
  C2 is secondary; its failure is informative but not a FAIL for the primary claim.

If PASS: supports INV-010 -- periodic offline integration produces measurably
  better harm avoidance than no offline integration; and offline pass is
  functional (slot diversity > noise floor). Both 'exists' and 'is required'.
If FAIL: weakens INV-010 -- offline integration does not measurably improve
  harm avoidance in the current substrate. Could reflect insufficient training
  duration or insensitive behavioral metric; note in evidence_direction_note.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_430_inv010_offline_integration_necessity"
CLAIM_IDS = ["INV-010"]
EXPERIMENT_PURPOSE = "evidence"

WARMUP_EPISODES = 20
TRAIN_EPISODES = 50
EVAL_EPISODES = 20
TOTAL_EPISODES = WARMUP_EPISODES + TRAIN_EPISODES + EVAL_EPISODES  # = 90
STEPS_PER_EPISODE = 150
SLEEP_INTERVAL = 10   # sleep cycle every N episodes during training phase
NUM_SEEDS = 3
SEEDS = [42, 7, 13]

# Pre-registered thresholds
THRESH_C3_SLOT_DIV = 0.02    # SWS must produce slot diversity above noise floor
PASS_MIN_SEEDS_C1 = 2         # C1 must hold in at least 2/3 seeds
PASS_MIN_SEEDS_C2 = 2         # C2 must hold in at least 2/3 seeds


def _make_env(seed: int, eval_layout: bool = False) -> CausalGridWorldV2:
    """Create CausalGridWorldV2 with moderate hazard density."""
    s = seed + (9999 if eval_layout else 0)
    return CausalGridWorldV2(
        seed=s,
        size=10,
        num_hazards=4,
        num_resources=3,
        hazard_harm=0.04,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        env_drift_interval=15,
        env_drift_prob=0.1,
    )


def _make_agent(env: CausalGridWorldV2, sleep_enabled: bool) -> REEAgent:
    """Build agent with or without sleep."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        sws_enabled=sleep_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=sleep_enabled,
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
        normed = F.normalize(mem.float(), dim=-1)
        sim = torch.mm(normed, normed.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        dist = 1.0 - sim[mask]
        return float(dist.mean().item())


def _run_episode(agent: REEAgent, env: CausalGridWorldV2,
                 optimizer: Optional[torch.optim.Optimizer],
                 steps: int) -> Dict[str, float]:
    """Run one episode. Returns per-episode metrics."""
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    device = agent.device

    ep_harm = 0.0
    ep_benefit = 0.0

    for _ in range(steps):
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

        if optimizer is not None:
            optimizer.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                pred_loss.backward()
                optimizer.step()

        ep_harm += float(info.get("total_harm", 0.0))
        ep_benefit += float(info.get("total_benefit", 0.0))

        if done:
            break

    # Use final episode totals (cumulative from env)
    return {
        "total_harm": float(info.get("total_harm", ep_harm)),
        "total_benefit": float(info.get("total_benefit", ep_benefit)),
    }


def run_condition(condition_name: str, sleep_enabled: bool,
                  seed: int, dry_run: bool = False) -> Dict:
    """Run one condition x seed. Returns metrics."""
    torch.manual_seed(seed)

    ep_train = 2 if dry_run else (WARMUP_EPISODES + TRAIN_EPISODES)
    ep_eval = 1 if dry_run else EVAL_EPISODES
    total_ep = ep_train + ep_eval

    env = _make_env(seed)
    eval_env = _make_env(seed, eval_layout=True)
    agent = _make_agent(env, sleep_enabled)
    device = agent.device

    optimizer = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-4,
    )

    sws_diversities: List[float] = []
    train_harm_per_ep: List[float] = []
    train_benefit_per_ep: List[float] = []

    # Training phase (warm-up + consolidation)
    print(f"Seed {seed} Condition {condition_name}")
    for ep in range(ep_train):
        ep_metrics = _run_episode(agent, env, optimizer, STEPS_PER_EPISODE)
        # total_harm/total_benefit reset on env.reset() each episode
        train_harm_per_ep.append(float(ep_metrics["total_harm"]))
        train_benefit_per_ep.append(float(ep_metrics["total_benefit"]))

        # Sleep cycle during consolidation phase
        if (sleep_enabled and ep >= WARMUP_EPISODES and
                (ep - WARMUP_EPISODES + 1) % SLEEP_INTERVAL == 0):
            sleep_m = agent.run_sleep_cycle()
            sws_div = sleep_m.get("sws_slot_diversity", 0.0)
            sws_diversities.append(sws_div)

        ep_harm = train_harm_per_ep[-1]
        ep_benefit = train_benefit_per_ep[-1]
        # Progress line every ~50 training episodes
        if (ep + 1) % 50 == 0 or dry_run:
            print(f"  [train] label seed={seed} ep {ep+1}/{total_ep} "
                  f"cond={condition_name} harm={ep_harm:.4f} benefit={ep_benefit:.4f}",
                  flush=True)

    # Eval phase (fixed layout, no training)
    eval_harm_per_ep: List[float] = []
    eval_benefit_per_ep: List[float] = []

    for ep in range(ep_eval):
        ep_metrics = _run_episode(agent, eval_env, None, STEPS_PER_EPISODE)
        # total_harm/total_benefit reset on env.reset() each episode
        eval_harm_per_ep.append(float(ep_metrics["total_harm"]))
        eval_benefit_per_ep.append(float(ep_metrics["total_benefit"]))

        ep_harm = eval_harm_per_ep[-1]
        ep_benefit = eval_benefit_per_ep[-1]
        cur_ep = ep_train + ep + 1
        if (ep + 1) % 10 == 0 or dry_run:
            print(f"  [train] label seed={seed} ep {cur_ep}/{total_ep} "
                  f"cond={condition_name} eval harm={ep_harm:.4f} benefit={ep_benefit:.4f}",
                  flush=True)

    mean_eval_harm = sum(eval_harm_per_ep) / len(eval_harm_per_ep) if eval_harm_per_ep else 0.0
    mean_eval_benefit = sum(eval_benefit_per_ep) / len(eval_benefit_per_ep) if eval_benefit_per_ep else 0.0
    mean_sws_diversity = sum(sws_diversities) / len(sws_diversities) if sws_diversities else 0.0
    final_slot_div = _compute_slot_diversity(agent)

    # C1 check per seed: SLEEP harm < NO_SLEEP harm
    # C2 check per seed: SLEEP benefit >= NO_SLEEP benefit
    # C3 check per seed: sws_slot_diversity > 0.02

    passed = False  # set at main level after collecting both conditions
    print(f"  [cond_done] {condition_name} seed={seed} "
          f"eval_harm={mean_eval_harm:.4f} eval_benefit={mean_eval_benefit:.4f} "
          f"sws_div={mean_sws_diversity:.4f} final_div={final_slot_div:.4f}")
    print(f"verdict: {'N/A (per-condition partial result)'}")

    return {
        "condition": condition_name,
        "seed": seed,
        "mean_eval_harm_per_ep": mean_eval_harm,
        "mean_eval_benefit_per_ep": mean_eval_benefit,
        "mean_sws_slot_diversity": mean_sws_diversity,
        "final_slot_diversity": final_slot_div,
        "n_sleep_cycles": len(sws_diversities),
        "sws_diversity_series": sws_diversities,
        "eval_harm_series": eval_harm_per_ep,
        "eval_benefit_series": eval_benefit_per_ep,
    }


def main(dry_run: bool = False) -> Dict:
    """Run all conditions and seeds, compute PASS/FAIL."""
    sleep_results: List[Dict] = []
    nosleep_results: List[Dict] = []

    for seed in SEEDS:
        sleep_results.append(run_condition("SLEEP", True, seed, dry_run=dry_run))
        nosleep_results.append(run_condition("NO_SLEEP", False, seed, dry_run=dry_run))

    # C1: SLEEP mean_eval_harm < NO_SLEEP mean_eval_harm in >= 2/3 seeds
    c1_wins = sum(
        1 for s, n in zip(sleep_results, nosleep_results)
        if s["mean_eval_harm_per_ep"] < n["mean_eval_harm_per_ep"]
    )
    c1_pass = c1_wins >= PASS_MIN_SEEDS_C1

    # C2: SLEEP mean_eval_benefit >= NO_SLEEP mean_eval_benefit in >= 2/3 seeds
    c2_wins = sum(
        1 for s, n in zip(sleep_results, nosleep_results)
        if s["mean_eval_benefit_per_ep"] >= n["mean_eval_benefit_per_ep"]
    )
    c2_pass = c2_wins >= PASS_MIN_SEEDS_C2

    # C3: SLEEP mean_sws_slot_diversity > THRESH_C3_SLOT_DIV in ALL seeds
    c3_wins = sum(
        1 for s in sleep_results
        if s["mean_sws_slot_diversity"] > THRESH_C3_SLOT_DIV
    )
    c3_pass = c3_wins == NUM_SEEDS

    outcome = "PASS" if (c1_pass and c3_pass) else "FAIL"

    print(f"C1 (SLEEP harm < NO_SLEEP harm in {c1_wins}/3): {c1_pass}")
    print(f"C2 (SLEEP benefit >= NO_SLEEP benefit in {c2_wins}/3): {c2_pass}")
    print(f"C3 (SWS slot_div > {THRESH_C3_SLOT_DIV} in all seeds: {c3_wins}/3): {c3_pass}")
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
            "C1": f"SLEEP mean_eval_harm < NO_SLEEP in >= {PASS_MIN_SEEDS_C1}/3 seeds",
            "C2": f"SLEEP mean_eval_benefit >= NO_SLEEP in >= {PASS_MIN_SEEDS_C2}/3 seeds",
            "C3": f"SLEEP sws_slot_diversity > {THRESH_C3_SLOT_DIV} in ALL seeds",
        },
        "acceptance_checks": {
            "C1_sleep_harm_lt_nosleep_2of3": c1_pass,
            "C2_sleep_benefit_ge_nosleep_2of3": c2_pass,
            "C3_sws_slot_div_gt_threshold_all": c3_pass,
        },
        "criterion_details": {
            "C1_wins": c1_wins,
            "C2_wins": c2_wins,
            "C3_wins": c3_wins,
        },
        "per_seed_results": {
            "SLEEP": sleep_results,
            "NO_SLEEP": nosleep_results,
        },
        "params": {
            "warmup_episodes": WARMUP_EPISODES if not dry_run else 1,
            "train_episodes": TRAIN_EPISODES if not dry_run else 1,
            "eval_episodes": EVAL_EPISODES if not dry_run else 1,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "num_seeds": NUM_SEEDS,
            "seeds": SEEDS,
            "sws_consolidation_steps": 8,
            "rem_attribution_steps": 6,
            "harm_threshold_c3": THRESH_C3_SLOT_DIV,
        },
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "summary": (
            "INV-010 test: SLEEP (periodic sleep cycles) vs NO_SLEEP (online only). "
            f"C1 primary: SLEEP harm avoidance better in {c1_wins}/3 seeds. "
            f"C3 sanity: sleep cycle installs differentiated slots in {c3_wins}/3 seeds. "
            f"Outcome: {outcome}."
        ),
        "scenario": "Two-arm probe: periodic offline integration vs continuous online learning",
        "interpretation": (
            "PASS supports INV-010 ('Offline integration exists and is required'): "
            "agents with periodic sleep cycles accumulate less harm in eval episodes, "
            "and the SWS pass functionally installs differentiated context priors. "
            "Both halves of INV-010 confirmed: offline integration exists (C3) "
            "and improves behavioral outcomes (C1). "
            "FAIL would suggest the current substrate does not yet show measurable "
            "behavioral benefit from offline integration -- likely requires longer "
            "training, more complex environments, or fuller sleep pipeline (V4 scope)."
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
                        help="Run 2+1 episodes per condition to verify wiring")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
