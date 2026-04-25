#!/opt/local/bin/python3
"""
V3-EXQ-482 -- SD-029 baseline-monostrategy diagnostic.

Claims: SD-029

Purpose (diagnostic / substrate-readiness)
------------------------------------------
EXQ-480 (param sweep over MECH-284 staleness) revealed a SECONDARY
finding distinct from the V_s read-side gap that V3-EXQ-481 targets:
the OFF baseline arm reported action_class_entropy=0.0 in BOTH seeds
EVEN THOUGH the SD-029 balanced hazard-event curriculum was active
for all arms. This means either (a) the curriculum is not breaking
baseline monostrategy at all, or (b) its effect is not visible at
action-class granularity.

V3-EXQ-479 only validated the SD-029 hazard-injection MECHANISM
(injection_log populated, info keys exposed, n_steps_total in
expected range) -- it did not measure action_class_entropy or
behavioural diversity.

This experiment isolates the SD-029 effect on baseline behavioural
diversity by running a plain agent (no V_s circuit, no MECH-284
hysteresis, no fancy substrate flags) with SD-029 ON vs OFF. PASS
confirms the curriculum has a measurable effect on action-class
entropy at the granularity of the executed action stream; FAIL
flags the curriculum as either broken or operating at a granularity
invisible to action-class entropy (in which case the metric or the
curriculum scope needs revisiting before relying on the curriculum
to break baseline monostrategy).

Arms
----
    OFF: plain baseline agent, SD-029 disabled (no scheduled
         external-hazard injection).
    ON : plain baseline agent, SD-029 enabled (curriculum injection
         on at default interval/prob).

Both arms share the plain-agent default. None of the substrate flags
that EXQ-478/480/481 toggle are touched here -- this is a clean read
on whether SD-029 alone produces a behavioural delta.

Metrics
-------
    action_class_entropy: Shannon entropy over executed action-class
        histogram across the run. Primary metric.
    action_class_counts: per-class executions (sanity).
    n_external_hazard_events: cumulative scheduled-external injection
        count from CausalGridWorldV2 (sanity that SD-029 fired in ON).

Pass / fail rule
----------------
    PASS = entropy(SD-029=ON) > entropy(SD-029=OFF) + 0.1
           in >= 2/2 seeds.

    FAIL = curriculum produces no measurable entropy lift. SD-029
           cannot be relied upon to break baseline monostrategy at
           this granularity; either the curriculum needs to inject
           more aggressively, or the metric needs replacing
           (state-cell entropy, exploration coverage, etc.).

experiment_purpose=diagnostic. Substrate-readiness gate. NOT
governance evidence; SD-029 governance evidence requires a follow-up
behavioural EXQ once this diagnostic confirms (or refutes) the
curriculum effect on action-class entropy.

See ree-v3/CLAUDE.md SD-029 section.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_482_sd029_baseline_monostrategy_diagnostic"
CLAIM_IDS = ["SD-029"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
CONDITIONS = ["OFF", "ON"]
EPISODES = 6
STEPS_PER_EP = 200
ENTROPY_LIFT_MIN = 0.1

CURRICULUM_INTERVAL = 50
CURRICULUM_PROB = 0.5
CURRICULUM_ADJACENT_ONLY = True


def _make_env(seed: int, condition: str) -> CausalGridWorldV2:
    sd029_on = condition == "ON"
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
        scheduled_external_hazard_enabled=sd029_on,
        scheduled_external_hazard_interval=CURRICULUM_INTERVAL,
        scheduled_external_hazard_prob=CURRICULUM_PROB,
        scheduled_external_hazard_adjacent_only=CURRICULUM_ADJACENT_ONLY,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
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
        # Plain baseline: NO V_s circuit, NO MECH-284, NO fancy substrate.
    )
    return REEAgent(cfg)


def _shannon_entropy(counts: Dict[int, int]) -> float:
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


def _run_condition(seed: int, condition: str) -> Dict:
    torch.manual_seed(seed)
    env = _make_env(seed, condition)
    agent = _make_agent(env)

    action_counts: Dict[int, int] = {}
    n_ticks = 0
    n_external_hazard_events = 0

    for _ep in range(EPISODES):
        obs, _info = env.reset()
        for _step in range(STEPS_PER_EP):
            action = agent.act(obs)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            obs, _harm, done, info, _obs_dict = env.step(a_idx)
            n_ticks += 1
            if info.get("external_hazard_injected", False):
                n_external_hazard_events += 1
            if done:
                break

    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "n_ticks": n_ticks,
        "n_external_hazard_events": n_external_hazard_events,
    }


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- SD-029 baseline-monostrategy diagnostic",
          flush=True)
    print(f"Arms: {CONDITIONS} (plain baseline agent; SD-029 toggled)",
          flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Episodes x steps_per_ep: {EPISODES} x {STEPS_PER_EP}", flush=True)
    print("Metrics: action_class_entropy (primary), action_class_counts, "
          "n_external_hazard_events", flush=True)
    print(f"PASS = entropy(SD-029=ON) > entropy(SD-029=OFF) + "
          f"{ENTROPY_LIFT_MIN} in >= 2/2 seeds", flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} SD-029 baseline diagnostic"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit 0; do not execute.")
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}", flush=True)
            r = _run_condition(seed=seed, condition=cond)
            print(f"  -> entropy={r['action_class_entropy']:.4f} "
                  f"n_ext_haz={r['n_external_hazard_events']} "
                  f"counts={r['action_class_counts']} "
                  f"n_ticks={r['n_ticks']}", flush=True)
            all_results.append(r)

    off_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "OFF"}
    on_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "ON"}

    per_seed_gate = []
    seeds_passing = 0
    for seed in SEEDS:
        off_r = off_by_seed[seed]
        on_r = on_by_seed[seed]
        lift = on_r["action_class_entropy"] - off_r["action_class_entropy"]
        passed = lift > ENTROPY_LIFT_MIN
        per_seed_gate.append({
            "seed": seed,
            "off_entropy": off_r["action_class_entropy"],
            "on_entropy": on_r["action_class_entropy"],
            "entropy_lift": lift,
            "on_n_ext_hazard": on_r["n_external_hazard_events"],
            "passed": passed,
        })
        if passed:
            seeds_passing += 1

    outcome = "PASS" if seeds_passing >= len(SEEDS) else "FAIL"

    summary = {
        "gate_rule": (
            f"entropy(SD-029=ON) > entropy(SD-029=OFF) + "
            f"{ENTROPY_LIFT_MIN} in >= {len(SEEDS)}/{len(SEEDS)} seeds"
        ),
        "per_seed_gate": per_seed_gate,
        "seeds_passing": seeds_passing,
        "seeds_required": len(SEEDS),
        "pass": outcome == "PASS",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for row in per_seed_gate:
        print(
            f"  seed={row['seed']} "
            f"off_entropy={row['off_entropy']:.4f} "
            f"on_entropy={row['on_entropy']:.4f} "
            f"lift={row['entropy_lift']:.4f} "
            f"ext_haz={row['on_n_ext_hazard']} "
            f"passed={row['passed']}",
            flush=True,
        )

    per_claim = {
        cid: ("supports" if outcome == "PASS" else "weakens")
        for cid in CLAIM_IDS
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": (
            "supports" if outcome == "PASS" else "weakens"
        ),
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_note": (
            "SD-029 baseline-monostrategy diagnostic. EXQ-480 OFF arms "
            "produced action_class_entropy=0.0 across all seeds even "
            "with SD-029 active for all arms; EXQ-479 only validated "
            "the SD-029 mechanism, not its behavioural effect at "
            "action-class granularity. PASS confirms SD-029 lifts "
            "action-class entropy in a plain baseline agent. FAIL "
            "flags the curriculum as either too sparse to break "
            "baseline monostrategy at this granularity or as having "
            "an effect invisible to action-class entropy -- in which "
            "case the metric or the curriculum scope must be "
            "revisited before SD-029 is relied upon as the "
            "monostrategy-breaking lever."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "entropy_lift_min": ENTROPY_LIFT_MIN,
            "curriculum_interval": CURRICULUM_INTERVAL,
            "curriculum_prob": CURRICULUM_PROB,
            "curriculum_adjacent_only": CURRICULUM_ADJACENT_ONLY,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Result written to: {out_file}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
