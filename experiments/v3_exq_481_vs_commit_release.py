#!/opt/local/bin/python3
"""
V3-EXQ-481 -- V_s -> commit release substrate validation.

Claims: MECH-269, MECH-090

Purpose (diagnostic / substrate-readiness)
------------------------------------------
EXQ-478 / EXQ-480 confirmed the V_s invalidation runtime is WIRED
(anchor_resets fire abundantly, staleness peaks at ~0.94/0.99) but
behaviourally INERT (freeze_recommit_count and action_class_entropy
were bit-identical between the OFF baseline and ON arms across all
parameter sweep variants). Diagnosis: the V_s pipeline write side
is complete but the read side -- the BetaGate consumer that turns
anchor invalidation into a commitment release -- was missing.

This experiment validates the read-side hook
(REEAgent.use_vs_commit_release): with the full V_s circuit ON for
both arms, the only difference is whether select_action() snapshots
active anchor keys at commit entry and releases beta when any
snapshot key drops out of the active set.

Arms
----
    OFF: full V_s circuit ON, use_vs_commit_release=False (mirrors
         EXQ-480 ON arms exactly).
    ON : full V_s circuit ON, use_vs_commit_release=True.

Both arms share:
    use_per_stream_vs=True
    use_event_segmenter=True
    use_invalidation_trigger=True
    use_anchor_sets=True
    use_per_region_vs=True
    use_staleness_accumulator=True
    use_mech284_hysteresis=True

Metrics
-------
    freeze_recommit_count: maximal-run count over executed action-class
        sequences with run length >= 3 (proxy for stuck monostrategy).
    anchor_reset_count: active->inactive transitions observed over the
        run (sanity check that V_s circuit is firing identically across
        arms; the OFF arm should report comparable resets to confirm
        the only architectural difference is the read-side hook).
    vs_commit_release_count: agent._vs_commit_release_count -- number
        of times the V_s -> commit release block fired in select_action().
        Must be non-zero in ON; zero in OFF.
    action_class_entropy: Shannon entropy over executed action-class
        histogram across the run.
    mean_staleness_peak: per-episode max staleness (cross-check).

Pass / fail rule
----------------
    PASS = (vs_commit_release_count(ON) > 0)
           AND (freeze_recommit_count(ON) < freeze_recommit_count(OFF))
           in >= 2/2 seeds.

    FAIL on any seed -> read-side hook is wired but does not produce a
    behavioural delta on this env. Revisit BoundaryEvent rate /
    anchor-key snapshot scope / commit-entry capture sites.

experiment_purpose=diagnostic. Substrate-readiness gate. NOT governance
evidence; Phase 3 + read-side combined four-arm dissociation belongs
to V3-EXQ-476 successors when this hook PASSes.

See REE_assembly/docs/architecture/v_s_invalidation_runtime.md
See ree-v3/CLAUDE.md MECH-269 / MECH-090 read-side section.
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_481_vs_commit_release"
CLAIM_IDS = ["MECH-269", "MECH-090"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
CONDITIONS = ["OFF", "ON"]
EPISODES = 6
STEPS_PER_EP = 200
FREEZE_RUN_LEN = 3


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
    release_on = condition == "ON"
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
        # Full V_s invalidation circuit, identical across both arms.
        use_per_stream_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        use_per_region_vs=True,
        use_staleness_accumulator=True,
        use_mech284_hysteresis=True,
        # The variable under test.
        use_vs_commit_release=release_on,
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

        staleness_peaks.append(ep_peak)

    mean_staleness_peak = (
        sum(staleness_peaks) / len(staleness_peaks)
        if staleness_peaks else 0.0
    )

    vs_release_count = int(getattr(agent, "_vs_commit_release_count", 0))

    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "n_ticks": n_ticks,
        "freeze_recommit_count": _count_freeze_runs(action_seq, FREEZE_RUN_LEN),
        "anchor_reset_count": anchor_reset_count,
        "vs_commit_release_count": vs_release_count,
        "mean_staleness_peak": mean_staleness_peak,
    }


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- V_s -> commit release substrate validation",
          flush=True)
    print(f"Arms: {CONDITIONS} (V_s circuit ON for both; "
          "use_vs_commit_release toggled)", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Episodes x steps_per_ep: {EPISODES} x {STEPS_PER_EP}", flush=True)
    print("Metrics: freeze_recommit_count, vs_commit_release_count, "
          "anchor_reset_count, mean_staleness_peak, action_class_entropy",
          flush=True)
    print("PASS = vs_commit_release_count(ON) > 0 AND "
          "freeze_recommit_count(ON) < freeze_recommit_count(OFF) "
          "in >= 2/2 seeds", flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} V_s -> commit release validation"
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
                  f"vs_release={r['vs_commit_release_count']} "
                  f"staleness_peak={r['mean_staleness_peak']:.4f} "
                  f"n_ticks={r['n_ticks']}", flush=True)
            all_results.append(r)

    off_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "OFF"}
    on_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "ON"}

    per_seed_gate = []
    seeds_passing = 0
    for seed in SEEDS:
        off_r = off_by_seed[seed]
        on_r = on_by_seed[seed]
        release_ok = on_r["vs_commit_release_count"] > 0
        freeze_ok = on_r["freeze_recommit_count"] < off_r["freeze_recommit_count"]
        passed = release_ok and freeze_ok
        per_seed_gate.append({
            "seed": seed,
            "off_freeze_recommit": off_r["freeze_recommit_count"],
            "on_freeze_recommit": on_r["freeze_recommit_count"],
            "off_vs_release": off_r["vs_commit_release_count"],
            "on_vs_release": on_r["vs_commit_release_count"],
            "off_anchor_resets": off_r["anchor_reset_count"],
            "on_anchor_resets": on_r["anchor_reset_count"],
            "vs_release_ok": release_ok,
            "freeze_drop_ok": freeze_ok,
            "passed": passed,
        })
        if passed:
            seeds_passing += 1

    outcome = "PASS" if seeds_passing >= len(SEEDS) else "FAIL"

    summary = {
        "gate_rule": (
            "vs_commit_release_count(ON) > 0 AND "
            "freeze_recommit_count(ON) < freeze_recommit_count(OFF) "
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
            f"  seed={row['seed']} "
            f"off_freeze={row['off_freeze_recommit']} "
            f"on_freeze={row['on_freeze_recommit']} "
            f"on_vs_release={row['on_vs_release']} "
            f"on_resets={row['on_anchor_resets']} "
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
            "V_s -> commit release substrate-readiness diagnostic. "
            "EXQ-478/480 showed the V_s invalidation circuit was "
            "wired-but-inert: anchor resets fired abundantly with no "
            "behavioural delta. This experiment isolates the read-side "
            "hook (REEAgent select_action() V_s release block) by "
            "running both arms with the full V_s circuit ON and "
            "toggling only use_vs_commit_release. PASS confirms the "
            "read-side hook produces a measurable freeze-recommit drop "
            "and that the release path is exercised. FAIL = the hook "
            "is wired but inert on this env -- revisit BoundaryEvent "
            "rate, anchor-key snapshot scope, or commit-entry capture "
            "sites. Substrate-readiness only; not governance evidence."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "freeze_run_len": FREEZE_RUN_LEN,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_file}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
