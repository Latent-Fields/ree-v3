#!/opt/local/bin/python3
"""
V3-EXQ-476b -- MECH-269 V_s validation entropy probe (cascade gate).

Claims: MECH-269, MECH-287, MECH-288

Purpose (diagnostic)
--------------------
Cascade gate for the V_s-enabled validation experiments (V3-EXQ-445d /
V3-EXQ-449c / V3-EXQ-455a). Asks one narrow question:

    Does enabling MECH-269 + MECH-287 + MECH-288 Phase-2 runtime flags
    alone already break the monostrategy action-class lock, or does
    behaviour require the MECH-284 Phase-3 staleness-accumulator
    consumer to be landed first?

Phase-2 already wires an explicit per-region reset path
(HippocampalModule.apply_invalidation_broadcasts_to_regions) that drops
per_region_vs entries and mark_inactives matching anchors on broadcast
events. If that path produces measurable action-diversification in a
baseline agent, the downstream V_s cascade can proceed on Phase-2
substrate. If not, Phase-3 is the bottleneck.

Supersedes
----------
V3-EXQ-476a. EXQ-476a ran to completion but terminated every episode
after step 0 because the env.step 5-tuple was unpacked as
(obs, r, terminated, truncated, info) -- CausalGridWorldV2 actually
returns (flat_obs, harm_signal, done, info, obs_dict), so "truncated"
was the info dict (always truthy) and the break fired immediately.
Result: n_ticks=6 per seed, entropy=0 in every arm, uninterpretable
FAIL. 476b corrects the unpacking and uses `done` as the termination
flag. Same scientific question, same thresholds, same seeds.

Arms
----
    OFF:  baseline agent, all V_s runtime flags OFF.
    ON:   baseline agent, Phase-1 + Phase-2 V_s flags ON:
            use_per_stream_vs=True
            use_per_region_vs=True
            use_event_segmenter=True
            use_invalidation_trigger=True
            use_anchor_sets=True

Environment: CausalGridWorldV2 matched to V3-EXQ-473 / V3-EXQ-475 shape
(10x10, 2 hazards, 3 resources, harm stream on, harm history on).

Metric
------
    action_class_entropy per arm per seed (Shannon entropy over
    executed action-class histogram across the run).

Pass / fail rule
----------------
    PASS = action_class_entropy(ON) - action_class_entropy(OFF) >= 0.1
           in >= 2/2 seeds.
        -> cascade unblocked; V3-EXQ-445d / 449c / 455a may proceed.

    FAIL / INCONCLUSIVE = ON does not clear OFF by >= 0.1 in both seeds.
        -> MECH-284 Phase 3 consumer (staleness accumulator feeding
           action-selection bias) must land before any downstream
           cascade proceeds.

experiment_purpose=diagnostic. This is the narrowly-scoped gate probe;
full claim-evidence work happens downstream.

See REE_assembly/docs/architecture/v_s_invalidation_runtime.md
See ree-v3/CLAUDE.md MECH-269 / MECH-287 / MECH-288 sections.
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


EXPERIMENT_TYPE = "v3_exq_476b_mech269_vs_validation_probe"
CLAIM_IDS = ["MECH-269", "MECH-287", "MECH-288"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
CONDITIONS = ["OFF", "ON"]
EPISODES = 6
STEPS_PER_EP = 200
ENTROPY_DELTA_THRESHOLD = 0.1


def _make_env(seed: int) -> CausalGridWorldV2:
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
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    vs_on = condition == "ON"
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
        # V_s invalidation runtime under test (Phase 1 + Phase 2 only;
        # Phase 3 MECH-284 staleness accumulator is the gate hypothesis)
        use_per_stream_vs=vs_on,
        use_per_region_vs=vs_on,
        use_event_segmenter=vs_on,
        use_invalidation_trigger=vs_on,
        use_anchor_sets=vs_on,
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
    env = _make_env(seed)
    agent = _make_agent(env, condition)

    action_counts: Dict[int, int] = {}
    boundary_event_count = 0
    broadcast_event_count = 0
    anchor_active_peak = 0
    n_ticks = 0

    for ep in range(EPISODES):
        obs, _info = env.reset()
        for _step in range(STEPS_PER_EP):
            action = agent.act(obs)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            obs, _harm, done, _info, _obs_dict = env.step(a_idx)
            n_ticks += 1

            if condition == "ON":
                hc = agent.hippocampal
                be_q = getattr(hc, "_boundary_event_queue", None)
                if be_q is not None:
                    boundary_event_count += len(be_q)
                br_q = getattr(hc, "_broadcast_event_queue", None)
                if br_q is not None:
                    broadcast_event_count += len(br_q)
                anchor_set = getattr(hc, "anchor_set", None)
                if anchor_set is not None:
                    anchor_active_peak = max(
                        anchor_active_peak, len(anchor_set.active_anchors())
                    )

            if done:
                break

    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "n_ticks": n_ticks,
        "boundary_event_count": boundary_event_count,
        "broadcast_event_count": broadcast_event_count,
        "anchor_active_peak": anchor_active_peak,
    }


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- MECH-269 V_s validation entropy probe", flush=True)
    print(f"Arms: {CONDITIONS}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Episodes x steps_per_ep: {EPISODES} x {STEPS_PER_EP}", flush=True)
    print(f"Metric: action_class_entropy per arm per seed", flush=True)
    print(
        f"PASS = ON - OFF >= {ENTROPY_DELTA_THRESHOLD} in >=2/2 seeds "
        "-> cascade unblocked",
        flush=True,
    )
    print(
        "FAIL / INCONCLUSIVE -> MECH-284 Phase 3 consumer required before cascade",
        flush=True,
    )
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} MECH-269 V_s entropy probe"
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
                  f"n_actions={r['n_actions']} "
                  f"boundaries={r['boundary_event_count']} "
                  f"broadcasts={r['broadcast_event_count']} "
                  f"anchor_peak={r['anchor_active_peak']}", flush=True)
            all_results.append(r)

    off_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "OFF"}
    on_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "ON"}

    per_seed_delta = []
    seeds_passing = 0
    for seed in SEEDS:
        off_e = off_by_seed[seed]["action_class_entropy"]
        on_e = on_by_seed[seed]["action_class_entropy"]
        delta = on_e - off_e
        per_seed_delta.append({
            "seed": seed,
            "off_entropy": off_e,
            "on_entropy": on_e,
            "delta": delta,
            "cleared": delta >= ENTROPY_DELTA_THRESHOLD,
        })
        if delta >= ENTROPY_DELTA_THRESHOLD:
            seeds_passing += 1

    outcome = "PASS" if seeds_passing >= len(SEEDS) else "FAIL"

    summary = {
        "gate_rule": (
            f"action_class_entropy(ON) - action_class_entropy(OFF) "
            f">= {ENTROPY_DELTA_THRESHOLD} in >= {len(SEEDS)}/{len(SEEDS)} seeds"
        ),
        "per_seed_delta": per_seed_delta,
        "seeds_passing": seeds_passing,
        "seeds_required": len(SEEDS),
        "pass": outcome == "PASS",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for row in per_seed_delta:
        print(
            f"  seed={row['seed']} off={row['off_entropy']:.4f} "
            f"on={row['on_entropy']:.4f} delta={row['delta']:.4f} "
            f"cleared={row['cleared']}",
            flush=True,
        )

    per_claim = {
        cid: ("supports" if outcome == "PASS" else "inconclusive")
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
            "supports" if outcome == "PASS" else "inconclusive"
        ),
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_note": (
            "Cascade gate probe: narrowly tests whether Phase-1 + Phase-2 V_s "
            "runtime flags alone diversify action-class entropy relative to a "
            "fully-OFF baseline. A FAIL does not weaken MECH-269/287/288 "
            "substrate claims -- it indicates the MECH-284 Phase-3 consumer "
            "is required to convert the V_s observables into behaviour."
        ),
        "supersedes": "v3_exq_476a_mech269_vs_validation_probe",
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "entropy_delta_threshold": ENTROPY_DELTA_THRESHOLD,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
