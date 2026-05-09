"""
V3-EXQ-265a: SD-017 Sleep Phase Methods Validation -- Phase 2 retest

SUPERSEDES: V3-EXQ-265 (non_contributory under SD-016 attention-uniformity confound).

MECHANISM UNDER TEST: SD-017 (sleep_phase.minimal_sleep_infrastructure_v3)
  run_sws_schema_pass() and run_rem_attribution_pass() exercised under the
  Phase 2 substrate stack:
    - SD-016 Path 1 diversification loss ON (sd016_diversification_weight > 0,
      A2_div_only equivalent: writepath_mode=off so only the div loss fires).
    - MECH-269 Phase 1 per-stream V_s ON.
    - MECH-269 Phase 2 (ii) anchor sets ON (dual-trace).
    - SD-039 anchor goal-snapshot payload ON (substrate-level).
    - SD-017 SleepLoopManager ON (use_sleep_loop=True), SWS+REM enabled.

EXPERIMENT PURPOSE: diagnostic

WHY THE RETEST: original V3-EXQ-265 (2026-04-09) was reclassified non_contributory
because the SD-016 ContextMemory write path landed BEFORE the diversification
loss; ContextMemory slots collapsed and SWS schema diversity was unmeasurable.
The plan-of-record sleep_substrate_plan.md GAP-2 Tier 0 cleared 2026-04-27 when
EXQ-418e A2_div_only PASSed slot_diversity 1.0 mean / 0.9997 min in 3/3 seeds.
EXQ-265a re-runs the SD-017 methods validation under that fixed substrate.

SCIENTIFIC QUESTION:
  Under the Phase 2 substrate (SD-016 div loss + V_s + anchor sets + SD-039
  payload + sleep loop), do the SD-017 first-class methods produce measurably
  differentiated SWS schema writes and REM attribution rollouts? Does sleep
  cycling produce slot diversity ABOVE the SD-016-fixed waking baseline?

DESIGN:
  Two conditions, 3 seeds each (matched to 265 for direct comparability):
    WITH_SLEEP:    use_sleep_loop=True, sws_enabled=True, rem_enabled=True;
                   sleep cycle every SLEEP_INTERVAL episodes.
    WITHOUT_SLEEP: use_sleep_loop=False, sws_enabled=False, rem_enabled=False;
                   waking baseline under the SAME SD-016+V_s+anchor stack.
  Both conditions: SD-016 diversification loss ON, MECH-269 V_s ON,
  anchor sets ON, SD-039 payload ON. Two-context alternation SAFE/DANGEROUS
  every CONTEXT_SWITCH_EVERY episodes.

ACCEPTANCE CRITERIA (diagnostic):
  C1: In WITH_SLEEP, sws_n_writes > 0 in every seed.
      (SWS pass activates and writes to ContextMemory.)
  C2: In WITH_SLEEP, mean sws_slot_diversity > 0.10
      (Phase 2 threshold; under SD-016 diversification ON the baseline
       slot diversity should already be non-trivial. EXQ-418e A2_div_only
       hit slot_diversity=1.0; SWS writes operating on top of that should
       not collapse it. Threshold 0.10 is conservative -- catches collapse
       without requiring the extreme separation 418e measured.)
  C3: In WITH_SLEEP, rem_n_rollouts > 0 in every seed.
  C4: WITH_SLEEP slot diversity differs from WITHOUT_SLEEP in >= 2/3 seeds
      (Phase 2 acceptance: slot metrics differ between WITH_SLEEP and
       WITHOUT_SLEEP, vs the identical-across-conditions pattern observed
       in original 418/418a/436 runs that motivated this retest. Direction
       is open: SWS schema installation may either INCREASE diversity by
       writing diverse z_world prototypes, or DECREASE it relative to a
       waking-only attractor that climbs to 1.0 under div loss alone.
       Either signed difference is informative per the interpretation grid
       below.)

PASS: C1 AND C2 AND C3 AND C4.
  C1+C2+C3 = activation confirmed under Phase 2 substrate.
  C4 = sleep produces a measurable behavioural footprint on the substrate
       that EXQ-265 could not measure (SD-016 confound made all slots
       identical regardless of sleep).

INTERPRETATION GRID (for the discussant reviewing this):
  All PASS                  -> Phase 2 unblocks SD-017 retest cohort cleanly;
                               proceed to behavioural variants (418c, 436a,
                               500a, 503a) per sleep_substrate_plan.md.
  C1+C2+C3 PASS, C4 FAIL    -> SD-017 substrate activates but its effect on
                               ContextMemory diversity is below the
                               WITHOUT_SLEEP saturation ceiling. Investigate:
                               does WITH_SLEEP differ on rem_terrain_variance
                               or rem_n_reverse instead of slot diversity?
                               Possibly route to a metric-redesign successor
                               (265b) rather than a substrate fix.
  C1 or C3 FAIL             -> Substrate regression: the SD-017 methods stopped
                               firing under the Phase 2 stack. Diagnose via
                               /diagnose-errors. Likely candidates: sleep_loop
                               wiring incompatible with anchor_sets, or
                               world_experience_buffer empty under V_s gating.
  C2 FAIL only              -> SWS writes fire but produce a LOWER-diversity
                               substrate than EXQ-418e baseline. Indicates SWS
                               schema installation collapses the div-loss-
                               carved structure. Architectural concern: SWS
                               write should respect or improve, not flatten,
                               div-loss-trained slots.

claim_ids: ["SD-017"]
experiment_purpose: "diagnostic"
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "v3_exq_265a_sd017_sleep_phase_methods_validation_phase2"
QUEUE_ID = "V3-EXQ-265a"
SUPERSEDES = "V3-EXQ-265"
CLAIM_IDS = ["SD-017"]
EXPERIMENT_PURPOSE = "diagnostic"

SLEEP_INTERVAL = 10       # episodes between sleep cycles (matches EXQ-265)
TRAINING_EPISODES = 80    # matches EXQ-265 for direct comparability
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
NUM_SEEDS = 3

# SD-016 Path 1 diversification weight; A2_div_only equivalent (writes-off, div-on)
SD016_DIVERSIFICATION_WEIGHT = 0.5  # matches EXQ-418e LAMBDA_DIVERSIFY


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


def _make_agent(env: CausalGridWorldV2, sws_enabled: bool, rem_enabled: bool,
                use_sleep_loop: bool) -> REEAgent:
    """Phase 2 substrate stack: SD-016 div loss + MECH-269 V_s + anchor sets +
    SD-039 payload, with SD-017 sleep machinery toggled by caller.

    Note: anchor_sets requires per_stream_vs ON (precondition raised in
    HippocampalModule). They are wired together here.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # SD-016 Path 1 diversification (A2_div_only: writepath off, div on)
        sd016_writepath_mode="off",
        sd016_diversification_weight=SD016_DIVERSIFICATION_WEIGHT,
        # MECH-269 Phase 1 + Phase 2 (ii)
        use_per_stream_vs=True,
        use_anchor_sets=True,
        # SD-039 substrate-side anchor payload
        use_sd039_anchor_payload=True,
        # SD-017 sleep phases
        sws_enabled=sws_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=6,
        # SD-017 SleepLoopManager (Phase A scaffolding wraps run_sleep_cycle)
        use_sleep_loop=use_sleep_loop,
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
                  use_sleep_loop: bool, seed: int) -> Dict:
    """Run one condition x seed under Phase 2 substrate."""
    torch.manual_seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_enabled, rem_enabled, use_sleep_loop)
    device = agent.device

    # Train E1 + LatentStack as in EXQ-265. SD-016 diversification loss is
    # added inside compute_prediction_loss when sd016_diversification_weight > 0.
    optimizer = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-4,
    )

    sws_writes_total: List[float] = []
    sws_diversity_total: List[float] = []
    rem_rollouts_total: List[float] = []
    episode_harm_rates: List[float] = []
    final_slot_diversity = 0.0

    for ep in range(TRAINING_EPISODES):
        if ep % 10 == 0 or ep == TRAINING_EPISODES - 1:
            print(f"  [train] {condition_name} seed={seed} ep {ep+1}/{TRAINING_EPISODES}", flush=True)

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

            optimizer.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                pred_loss.backward()
                optimizer.step()

            ep_steps += 1
            if done:
                break

        episode_harm_rates.append(ep_harm / max(1, ep_steps))

        # Sleep cycle every SLEEP_INTERVAL episodes (WITH_SLEEP only)
        if sws_enabled or rem_enabled:
            if (ep + 1) % SLEEP_INTERVAL == 0 and ep > 0:
                sleep_metrics = agent.run_sleep_cycle()
                sws_writes_total.append(sleep_metrics.get("sws_n_writes", 0.0))
                sws_diversity_total.append(sleep_metrics.get("sws_slot_diversity", 0.0))
                rem_rollouts_total.append(sleep_metrics.get("rem_n_rollouts", 0.0))

    # Always measure final ContextMemory slot diversity at end-of-run for
    # WITHOUT_SLEEP comparison (and as cross-check for WITH_SLEEP).
    final_slot_diversity = _compute_slot_diversity(agent)
    if not (sws_enabled or rem_enabled):
        sws_diversity_total.append(final_slot_diversity)

    print(f"verdict: {'PASS' if sws_writes_total or not sws_enabled else 'FAIL'}", flush=True)

    return {
        "condition": condition_name,
        "seed": seed,
        "sws_writes_total": sws_writes_total,
        "sws_diversity_values": sws_diversity_total,
        "rem_rollouts_total": rem_rollouts_total,
        "final_slot_diversity": final_slot_diversity,
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
        ("WITH_SLEEP",    True,  True,  True),
        ("WITHOUT_SLEEP", False, False, False),
    ]

    for cond_name, sws_en, rem_en, sleep_loop in conditions:
        for seed_i in range(NUM_SEEDS):
            seed = 42 + seed_i * 7
            print(f"Seed {seed} Condition {cond_name}", flush=True)
            res = run_condition(cond_name, sws_en, rem_en, sleep_loop, seed)
            all_results[cond_name].append(res)
            print(f"  -> sws_writes={res['mean_sws_n_writes']:.1f}"
                  f" diversity={res['mean_sws_slot_diversity']:.4f}"
                  f" rem_rollouts={res['mean_rem_n_rollouts']:.1f}"
                  f" final_slot_div={res['final_slot_diversity']:.4f}", flush=True)

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
            "mean_final_slot_diversity": _agg(rs, "final_slot_diversity"),
            "mean_harm_rate_last20ep": _agg(rs, "mean_harm_rate_last20ep"),
        }

    with_s = agg["WITH_SLEEP"]
    wo_s   = agg["WITHOUT_SLEEP"]

    # Acceptance checks
    c1_pass = all(r["mean_sws_n_writes"] > 0.0 for r in all_results["WITH_SLEEP"])
    c2_pass = with_s["mean_sws_slot_diversity"] > 0.10
    c3_pass = all(r["mean_rem_n_rollouts"] > 0.0 for r in all_results["WITH_SLEEP"])

    # C4: WITH_SLEEP slot diversity DIFFERS from WITHOUT_SLEEP in >= 2/3 seeds.
    # Use signed difference > tolerance to count as "differs"; either direction
    # is informative per the interpretation grid in the docstring.
    DIFF_TOLERANCE = 0.05
    c4_diffs = []
    for ws_r, wo_r in zip(all_results["WITH_SLEEP"], all_results["WITHOUT_SLEEP"]):
        ws_div = ws_r["final_slot_diversity"]
        wo_div = wo_r["final_slot_diversity"]
        c4_diffs.append({
            "seed": ws_r["seed"],
            "with_sleep_final_slot_div": ws_div,
            "without_sleep_final_slot_div": wo_div,
            "abs_diff": abs(ws_div - wo_div),
            "differs": abs(ws_div - wo_div) > DIFF_TOLERANCE,
        })
    c4_diff_count = sum(1 for d in c4_diffs if d["differs"])
    c4_pass = c4_diff_count >= 2

    outcome = "PASS" if (c1_pass and c2_pass and c3_pass and c4_pass) else "FAIL"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "aggregated": agg,
        "acceptance_checks": {
            "C1_sws_writes_all_seeds": c1_pass,
            "C2_sws_diversity_gt_0.10": c2_pass,
            "C3_rem_rollouts_all_seeds": c3_pass,
            "C4_with_vs_without_differs_2of3": c4_pass,
            "C4_per_seed_diffs": c4_diffs,
            "C4_diff_count": c4_diff_count,
        },
        "params": {
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "num_seeds": NUM_SEEDS,
            "sws_consolidation_steps": 8,
            "rem_attribution_steps": 6,
            "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            "sd016_writepath_mode": "off",
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "use_sd039_anchor_payload": True,
            "use_sleep_loop_with_sleep_arm": True,
        },
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "evidence_direction_per_claim": {
            "SD-017": "supports" if outcome == "PASS" else "does_not_support",
        },
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
    print(f"C2 (with_sleep diversity > 0.10): {c2_pass} (mean={with_s['mean_sws_slot_diversity']:.4f})")
    print(f"C3 (rem_rollouts all seeds): {c3_pass}")
    print(f"C4 (with vs without differs in >= 2/3): {c4_pass} (n_differs={c4_diff_count}/3)")

    return output, out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run minimal end-to-end smoke (1 short episode per condition)")
    args = parser.parse_args()

    if args.dry_run:
        # Minimal dry-run: verify Phase 2 substrate wiring and SD-017 method
        # activation under the new flag stack.
        print("[DRY RUN] Phase 2 SD-017 + SD-016 div loss + V_s + anchor sets + SD-039")

        env = _make_env_safe(42)

        # Backward compat check: Phase 2 stack with sleep flags OFF still boots.
        agent_off = _make_agent(env, sws_enabled=False, rem_enabled=False,
                                use_sleep_loop=False)
        m = agent_off.run_sws_schema_pass()
        assert m["sws_n_writes"] == 0.0, "Disabled SWS should have 0 writes"
        m = agent_off.run_rem_attribution_pass()
        assert m["rem_n_rollouts"] == 0.0, "Disabled REM should have 0 rollouts"
        print("  [OK] Phase 2 stack + sleep OFF: backward compat preserved")

        # Activation: Phase 2 stack + sleep ON.
        agent_on = _make_agent(env, sws_enabled=True, rem_enabled=True,
                               use_sleep_loop=True)
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

        # Verify Phase 2 substrate is actually wired (not just defaulted off).
        print(f"  per_stream_vs populated: {len(agent_on.hippocampal.per_stream_vs)} streams")
        print(f"  anchor_set instantiated: {agent_on.hippocampal.anchor_set is not None}")
        if agent_on.hippocampal.anchor_set is not None:
            print(f"  anchor_set.use_sd039_payload: "
                  f"{agent_on.hippocampal.anchor_set.config.use_sd039_anchor_payload}")
        print(f"  sleep_loop_manager instantiated: "
              f"{getattr(agent_on, 'sleep_loop_manager', None) is not None}")

        c1 = sws_writes > 0
        c3 = rem_rolls > 0
        smoke_ok = c1 and c3
        print(f"  C1 (sws_writes>0): {c1}")
        print(f"  C3 (rem_rollouts>0): {c3}")
        if smoke_ok:
            print("[DRY RUN] PASS - Phase 2 stack + SD-017 methods activate correctly")
        else:
            print("[DRY RUN] FAIL - check experience buffer fill or substrate wiring")

        # Emit smoke outcome so the runner conformance contract is satisfied
        # even from --dry-run paths.
        emit_outcome(
            outcome="PASS" if smoke_ok else "FAIL",
            manifest_path=None,
        )
        sys.exit(0 if smoke_ok else 1)

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE}")

    result, out_path = main()

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
