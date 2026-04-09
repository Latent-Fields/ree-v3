"""
V3-EXQ-265: SD-017 Sleep Phase Methods Validation

MECHANISM UNDER TEST: SD-017 (sleep_phase.minimal_sleep_infrastructure_v3)
  run_sws_schema_pass() and run_rem_attribution_pass() added to REEAgent.
  These are the first-class SD-017 substrate methods (not proxy hooks as in EXQ-242).

EXPERIMENT PURPOSE: diagnostic

SCIENTIFIC QUESTION:
  Do the new SD-017 first-class methods activate, write to ContextMemory with
  non-trivial diversity gain (SWS), and produce non-trivial REM attribution
  metrics? Does periodic sleep cycling improve slot_diversity compared to
  waking-only training?

DESIGN:
  Two conditions, 3 seeds each:
    WITH_SLEEP: sws_enabled=True, rem_enabled=True, sleep_cycle every 10 episodes
    WITHOUT_SLEEP: sws_enabled=False, rem_enabled=False (waking only, baseline)
  Both conditions: standard encoder training (prediction loss), same architecture.
  Two-context alternation: SAFE (low hazards) vs DANGEROUS (high hazards) every
  5 episodes. Buffer fills with both context types before first sleep.

ACCEPTANCE CRITERIA (diagnostic):
  C1: In WITH_SLEEP, sws_n_writes > 0 in every seed
      (SWS pass activates and writes to ContextMemory)
  C2: In WITH_SLEEP, mean sws_slot_diversity > 0.05
      (SWS pass produces measurably differentiated ContextMemory slots;
       baseline undifferentiated = ~0.0 after random init)
  C3: In WITH_SLEEP, rem_n_rollouts > 0 in every seed
      (REM pass activates and generates attribution rollouts)
  C4: sws_slot_diversity in WITH_SLEEP > sws_slot_diversity in WITHOUT_SLEEP
      in >= 2/3 seeds
      (Sleep cycle differentiates slots more than waking-only encoding)

PASS: C1 AND C2 AND C3 (functional activation confirmed)
  C4 is secondary (behavioral differentiation quality -- failure here suggests
  the substrate works but the ablation design needs more episodes).

claim_ids: ["SD-017"]
experiment_purpose: "diagnostic"
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


EXPERIMENT_TYPE = "v3_exq_265_sd017_sleep_phase_methods_validation"
CLAIM_IDS = ["SD-017"]
EXPERIMENT_PURPOSE = "diagnostic"

SLEEP_INTERVAL = 10       # episodes between sleep cycles
TRAINING_EPISODES = 80
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5  # alternate SAFE/DANGEROUS every N episodes
NUM_SEEDS = 3


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
        # SD-017: sleep phase switches
        sws_enabled=sws_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=6,
    )
    return REEAgent(cfg)


def _compute_slot_diversity(agent: REEAgent) -> float:
    """Mean pairwise cosine distance between ContextMemory slots (0=same, 1=orthogonal)."""
    with torch.no_grad():
        mem = agent.e1.context_memory.memory  # [num_slots, memory_dim]
        n = mem.shape[0]
        if n < 2:
            return 0.0
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())  # [n, n]
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        dist = 1.0 - sim[mask]
        return float(dist.mean().item())


def run_condition(condition_name: str, sws_enabled: bool, rem_enabled: bool,
                  seed: int) -> Dict:
    """Run one condition x seed."""
    torch.manual_seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_enabled, rem_enabled)
    device = agent.device

    optimizer = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-4,
    )

    # Aggregated metrics across sleep calls
    sws_writes_total: List[float] = []
    sws_diversity_total: List[float] = []
    rem_rollouts_total: List[float] = []
    episode_harm_rates: List[float] = []

    for ep in range(TRAINING_EPISODES):
        # Alternate context every CONTEXT_SWITCH_EVERY episodes
        use_dangerous = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe

        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        ep_harm = 0.0
        ep_steps = 0

        for step in range(STEPS_PER_EPISODE):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else \
                torch.zeros(1, agent.config.latent.world_dim, device=device)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            _, harm_signal, done, info, obs_dict = env.step(action)
            ep_harm += max(0.0, float(-harm_signal))

            # Training: E1 prediction + encoder
            optimizer.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                pred_loss.backward()
                optimizer.step()

            ep_steps += 1
            if done:
                break

        episode_harm_rates.append(ep_harm / max(1, ep_steps))

        # Sleep cycle every SLEEP_INTERVAL episodes
        if sws_enabled or rem_enabled:
            if (ep + 1) % SLEEP_INTERVAL == 0 and ep > 0:
                sleep_metrics = agent.run_sleep_cycle()
                sws_writes_total.append(sleep_metrics.get("sws_n_writes", 0.0))
                sws_diversity_total.append(sleep_metrics.get("sws_slot_diversity", 0.0))
                rem_rollouts_total.append(sleep_metrics.get("rem_n_rollouts", 0.0))

    # For WITHOUT_SLEEP: measure slot diversity once at end to compare
    if not sws_enabled and not rem_enabled:
        div = _compute_slot_diversity(agent)
        sws_diversity_total.append(div)

    return {
        "condition": condition_name,
        "seed": seed,
        "sws_writes_total": sws_writes_total,
        "sws_diversity_values": sws_diversity_total,
        "rem_rollouts_total": rem_rollouts_total,
        "mean_sws_n_writes": float(sum(sws_writes_total) / len(sws_writes_total))
                             if sws_writes_total else 0.0,
        "mean_sws_slot_diversity": float(sum(sws_diversity_total) / len(sws_diversity_total))
                                   if sws_diversity_total else 0.0,
        "mean_rem_n_rollouts": float(sum(rem_rollouts_total) / len(rem_rollouts_total))
                               if rem_rollouts_total else 0.0,
        "mean_harm_rate_last20ep": float(sum(episode_harm_rates[-20:]) / max(1, len(episode_harm_rates[-20:]))),
    }


def main():
    all_results = {"WITH_SLEEP": [], "WITHOUT_SLEEP": []}

    conditions = [
        ("WITH_SLEEP",   True,  True),
        ("WITHOUT_SLEEP", False, False),
    ]

    for cond_name, sws_en, rem_en in conditions:
        for seed_i in range(NUM_SEEDS):
            seed = 42 + seed_i * 7
            print(f"  Ep 0/{TRAINING_EPISODES} [{cond_name} seed={seed}]...")
            res = run_condition(cond_name, sws_en, rem_en, seed)
            all_results[cond_name].append(res)
            print(f"  -> sws_writes={res['mean_sws_n_writes']:.1f}"
                  f" diversity={res['mean_sws_slot_diversity']:.4f}"
                  f" rem_rollouts={res['mean_rem_n_rollouts']:.1f}")

    # Aggregate per condition
    def _agg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results if r[key] is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0

    agg = {}
    for cond_name in ["WITH_SLEEP", "WITHOUT_SLEEP"]:
        rs = all_results[cond_name]
        agg[cond_name] = {
            "mean_sws_n_writes": _agg(rs, "mean_sws_n_writes"),
            "mean_sws_slot_diversity": _agg(rs, "mean_sws_slot_diversity"),
            "mean_rem_n_rollouts": _agg(rs, "mean_rem_n_rollouts"),
            "mean_harm_rate_last20ep": _agg(rs, "mean_harm_rate_last20ep"),
        }

    with_s = agg["WITH_SLEEP"]
    wo_s   = agg["WITHOUT_SLEEP"]

    # Acceptance checks
    # C1: all WITH_SLEEP seeds had sws_n_writes > 0
    c1_pass = all(r["mean_sws_n_writes"] > 0.0 for r in all_results["WITH_SLEEP"])
    # C2: mean sws_slot_diversity > 0.05 in WITH_SLEEP
    c2_pass = with_s["mean_sws_slot_diversity"] > 0.05
    # C3: all WITH_SLEEP seeds had rem_n_rollouts > 0
    c3_pass = all(r["mean_rem_n_rollouts"] > 0.0 for r in all_results["WITH_SLEEP"])
    # C4: WITH_SLEEP diversity > WITHOUT_SLEEP in >= 2/3 seeds
    c4_wins = 0
    for ws_r, wo_r in zip(all_results["WITH_SLEEP"], all_results["WITHOUT_SLEEP"]):
        if ws_r["mean_sws_slot_diversity"] > wo_r["mean_sws_slot_diversity"]:
            c4_wins += 1
    c4_pass = c4_wins >= 2

    outcome = "PASS" if (c1_pass and c2_pass and c3_pass) else "FAIL"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "aggregated": agg,
        "acceptance_checks": {
            "C1_sws_writes_all_seeds": c1_pass,
            "C2_sws_diversity_gt_0.05": c2_pass,
            "C3_rem_rollouts_all_seeds": c3_pass,
            "C4_diversity_higher_2of3_seeds": c4_pass,
        },
        "params": {
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "num_seeds": NUM_SEEDS,
            "sws_consolidation_steps": 8,
            "rem_attribution_steps": 6,
        },
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
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
    print(f"Outcome: {outcome}")
    print(f"C1 (sws_writes all seeds): {c1_pass}")
    print(f"C2 (diversity>0.05): {c2_pass} ({with_s['mean_sws_slot_diversity']:.4f})")
    print(f"C3 (rem_rollouts all seeds): {c3_pass}")
    print(f"C4 (diversity higher 2/3): {c4_pass} ({c4_wins}/3 seeds)")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 episode per condition to check wiring")
    args = parser.parse_args()

    if args.dry_run:
        # Minimal dry-run: verify API wiring without full training run
        print("[DRY RUN] Testing SD-017 API wiring...")

        env = _make_env_safe(42)

        # Check backward compat: disabled config returns zeros
        agent_off = _make_agent(env, sws_enabled=False, rem_enabled=False)
        m = agent_off.run_sws_schema_pass()
        assert m["sws_n_writes"] == 0.0, "Disabled SWS should have 0 writes"
        m = agent_off.run_rem_attribution_pass()
        assert m["rem_n_rollouts"] == 0.0, "Disabled REM should have 0 rollouts"
        print("  Backward compat OK: disabled config returns zeros")

        # Check activation: enabled config, fill buffer with experience first
        agent_on = _make_agent(env, sws_enabled=True, rem_enabled=True)
        _, obs_dict = env.reset()
        agent_on.reset()
        agent_on.e1.reset_hidden_state()
        device = agent_on.device

        for step in range(20):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)
            latent = agent_on.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent_on.clock.advance()
            e1_prior = agent_on._e1_tick(latent) if ticks.get("e1_tick", False) else \
                torch.zeros(1, agent_on.config.latent.world_dim, device=device)
            candidates = agent_on.generate_trajectories(latent, e1_prior, ticks)
            action = agent_on.select_action(candidates, ticks)
            _, harm_signal, done, info, obs_dict = env.step(action)
            if done:
                _, obs_dict = env.reset()

        print(f"  World experience buffer size: {len(agent_on._world_experience_buffer)}")
        sleep_m = agent_on.run_sleep_cycle()
        sws_writes = sleep_m.get("sws_n_writes", 0)
        sws_div    = sleep_m.get("sws_slot_diversity", 0.0)
        rem_rolls  = sleep_m.get("rem_n_rollouts", 0)
        print(f"  SWS n_writes: {sws_writes}")
        print(f"  SWS slot_diversity: {sws_div:.4f}")
        print(f"  REM n_rollouts: {rem_rolls}")

        c1 = sws_writes > 0
        c3 = rem_rolls > 0
        print(f"  C1 (sws_writes>0): {c1}")
        print(f"  C3 (rem_rollouts>0): {c3}")
        if c1 and c3:
            print("[DRY RUN] PASS - SD-017 methods activate correctly")
        else:
            print("[DRY RUN] FAIL - check experience buffer fill")
        sys.exit(0)

    main()
