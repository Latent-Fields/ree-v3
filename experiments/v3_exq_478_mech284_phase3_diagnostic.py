#!/opt/local/bin/python3
"""
V3-EXQ-478 -- MECH-284 Phase 3 staleness accumulator diagnostic.

Claims: MECH-284, MECH-269

Purpose (diagnostic)
--------------------
Phase 3 landing validation for the V_s invalidation runtime. Asks the
narrow substrate-readiness question:

    With the full V_s invalidation circuit ON (Phase 1 per-stream +
    Phase 2 event segmenter + invalidation trigger + anchor sets +
    Phase 2 iii per-region + Phase 3 staleness accumulator +
    MECH-269 online hysteresis consuming MECH-284), does the online
    arm (a) populate the accumulator with non-trivial staleness,
    (b) fire at least a handful of anchor-reset events, and
    (c) reduce freeze-recommit behaviour relative to a fully-OFF
    baseline?

This is substrate-readiness, not governance evidence. The full
four-arm dissociation (MECH-287 present/lesioned x MECH-284-online
present/lesioned) is V3-EXQ-476 and is outstanding for promotion.

Arms
----
    OFF: baseline agent; all V_s runtime flags OFF.
    ON : Phase 1 + Phase 2 + Phase 3 online arm all ON:
            use_per_stream_vs=True
            use_event_segmenter=True
            use_invalidation_trigger=True
            use_anchor_sets=True
            use_per_region_vs=True
            use_staleness_accumulator=True
            use_mech284_hysteresis=True

Environment: CausalGridWorldV2 shaped to exercise hazards (so MECH-288
event boundaries fire on harm-stream spikes) and resource respawns
(so boundary events and anchor installation happen across an episode).

Metrics
-------
    freeze_recommit_count: number of consecutive-repeat action-class
        runs of length >= 3 (proxy for stuck-in-monostrategy freezes;
        the fewer, the less the agent gets locked onto one action).
    anchor_reset_count: HippocampalModule-reported active->inactive
        transitions (hysteresis fires + broadcast-explicit resets).
    mean_staleness_peak: per-episode max staleness value observed in
        the accumulator snapshot, averaged across episodes.
    action_class_entropy: Shannon entropy over executed action-class
        histogram across the run.

Pass / fail rule
----------------
    PASS = (anchor_reset_count >= 2 in ON)
           AND (freeze_recommit_count(ON) < freeze_recommit_count(OFF))
           in >= 2/2 seeds.

    FAIL = either anchor-reset arm silent or freeze count does not
           drop -> Phase 3 substrate is wired but behaviourally inert
           on this env; revisit attribution weighting / leak rate /
           threshold before relying on the online arm.

experiment_purpose=diagnostic. Substrate-readiness gate.

See REE_assembly/docs/architecture/v_s_invalidation_runtime.md
See ree-v3/CLAUDE.md MECH-284 / MECH-269 Phase 3 section.
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


EXPERIMENT_TYPE = "v3_exq_478_mech284_phase3_diagnostic"
CLAIM_IDS = ["MECH-284", "MECH-269"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
CONDITIONS = ["OFF", "ON"]
EPISODES = 6
STEPS_PER_EP = 200
FREEZE_RUN_LEN = 3
ANCHOR_RESET_MIN = 2


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
        # Full V_s invalidation circuit (Phase 1 + 2 + 3 online arm)
        use_per_stream_vs=vs_on,
        use_event_segmenter=vs_on,
        use_invalidation_trigger=vs_on,
        use_anchor_sets=vs_on,
        use_per_region_vs=vs_on,
        use_staleness_accumulator=vs_on,
        use_mech284_hysteresis=vs_on,
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


def _count_freeze_runs(action_seq: List[int], run_len: int) -> int:
    """Count number of maximal runs of identical consecutive actions
    with length >= run_len (proxy for monostrategy freezes)."""
    if not action_seq:
        return 0
    runs = 0
    cur_val = action_seq[0]
    cur_len = 1
    for a in action_seq[1:]:
        if a == cur_val:
            cur_len += 1
        else:
            if cur_len >= run_len:
                runs += 1
            cur_val = a
            cur_len = 1
    if cur_len >= run_len:
        runs += 1
    return runs


def _run_condition(seed: int, condition: str) -> Dict:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, condition)

    action_counts: Dict[int, int] = {}
    action_seq: List[int] = []
    n_ticks = 0
    anchor_reset_count = 0
    prev_active_keys: set = set()
    staleness_peaks: List[float] = []

    for _ep in range(EPISODES):
        obs, _info = env.reset()
        ep_peak = 0.0
        prev_active_keys = set()
        for _step in range(STEPS_PER_EP):
            action = agent.act(obs)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            action_seq.append(a_idx)
            obs, _harm, done, _info, _obs_dict = env.step(a_idx)
            n_ticks += 1

            if condition == "ON":
                hc = agent.hippocampal
                anchor_set = getattr(hc, "anchor_set", None)
                if anchor_set is not None:
                    active_now = {a.key for a in anchor_set.active_anchors()}
                    gone = prev_active_keys - active_now
                    anchor_reset_count += len(gone)
                    prev_active_keys = active_now
                sa = getattr(hc, "staleness_accumulator", None)
                if sa is not None:
                    snap = sa.snapshot()
                    if snap:
                        ep_peak = max(ep_peak, max(snap.values()))

            if done:
                break

        if condition == "ON":
            staleness_peaks.append(ep_peak)

    mean_staleness_peak = (
        sum(staleness_peaks) / len(staleness_peaks)
        if staleness_peaks else 0.0
    )

    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "n_ticks": n_ticks,
        "freeze_recommit_count": _count_freeze_runs(action_seq, FREEZE_RUN_LEN),
        "anchor_reset_count": anchor_reset_count,
        "mean_staleness_peak": mean_staleness_peak,
    }


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- MECH-284 Phase 3 staleness accumulator diagnostic", flush=True)
    print(f"Arms: {CONDITIONS}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Episodes x steps_per_ep: {EPISODES} x {STEPS_PER_EP}", flush=True)
    print("Metrics: freeze_recommit_count, anchor_reset_count, "
          "mean_staleness_peak, action_class_entropy", flush=True)
    print(f"PASS = anchor_reset_count(ON) >= {ANCHOR_RESET_MIN} AND "
          "freeze_recommit_count(ON) < freeze_recommit_count(OFF) "
          "in >=2/2 seeds", flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} MECH-284 Phase 3 diagnostic"
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
                  f"freeze_runs={r['freeze_recommit_count']} "
                  f"anchor_resets={r['anchor_reset_count']} "
                  f"mean_staleness_peak={r['mean_staleness_peak']:.4f} "
                  f"n_ticks={r['n_ticks']}", flush=True)
            all_results.append(r)

    off_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "OFF"}
    on_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "ON"}

    per_seed_gate = []
    seeds_passing = 0
    for seed in SEEDS:
        off_r = off_by_seed[seed]
        on_r = on_by_seed[seed]
        reset_ok = on_r["anchor_reset_count"] >= ANCHOR_RESET_MIN
        freeze_ok = on_r["freeze_recommit_count"] < off_r["freeze_recommit_count"]
        passed = reset_ok and freeze_ok
        per_seed_gate.append({
            "seed": seed,
            "off_freeze_recommit": off_r["freeze_recommit_count"],
            "on_freeze_recommit": on_r["freeze_recommit_count"],
            "on_anchor_reset_count": on_r["anchor_reset_count"],
            "on_mean_staleness_peak": on_r["mean_staleness_peak"],
            "anchor_reset_ok": reset_ok,
            "freeze_drop_ok": freeze_ok,
            "passed": passed,
        })
        if passed:
            seeds_passing += 1

    outcome = "PASS" if seeds_passing >= len(SEEDS) else "FAIL"

    summary = {
        "gate_rule": (
            f"anchor_reset_count(ON) >= {ANCHOR_RESET_MIN} AND "
            f"freeze_recommit_count(ON) < freeze_recommit_count(OFF) "
            f"in >= {len(SEEDS)}/{len(SEEDS)} seeds"
        ),
        "per_seed_gate": per_seed_gate,
        "seeds_passing": seeds_passing,
        "seeds_required": len(SEEDS),
        "pass": outcome == "PASS",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for row in per_seed_gate:
        print(
            f"  seed={row['seed']} off_freeze={row['off_freeze_recommit']} "
            f"on_freeze={row['on_freeze_recommit']} "
            f"resets={row['on_anchor_reset_count']} "
            f"staleness_peak={row['on_mean_staleness_peak']:.4f} "
            f"passed={row['passed']}",
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
            "Phase 3 substrate-readiness diagnostic for MECH-284 + "
            "MECH-269 online-arm coupling. PASS confirms the "
            "accumulator populates, anchor resets fire, and freeze-"
            "recommit behaviour drops relative to a fully-OFF "
            "baseline. FAIL flags Phase 3 as wired-but-behaviourally-"
            "inert -- revisit attribution weighting / leak_factor / "
            "reset_threshold before relying on the online arm. Not "
            "governance evidence; full four-arm dissociation is "
            "V3-EXQ-476."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "freeze_run_len": FREEZE_RUN_LEN,
            "anchor_reset_min": ANCHOR_RESET_MIN,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Result written to: {out_file}", flush=True)

    from experiment_protocol import emit_outcome
    emit_outcome(outcome=outcome, manifest_path=str(out_file))

    return 0


if __name__ == "__main__":
    sys.exit(main())
