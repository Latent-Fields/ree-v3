#!/opt/local/bin/python3
"""
V3-EXQ-596 -- MECH-269 post-SP-CEM end-to-end behavioral validation.

Claims: MECH-269

Purpose (diagnostic / substrate-readiness)
------------------------------------------
MECH-269 (per-region V_s + anchor-selection cluster) is IMPLEMENTED but
governance still marks it wired_but_inert: V3-EXQ-476c tested Phase-1+2
flags on a monostrategy-collapsed agent (entropy 0.0 both arms).
V3-EXQ-481/481b validated the read-side commit-release hook in isolation.
ARC-065 SP-CEM is now on the main path (V3-EXQ-567 PASS).

This experiment re-tests the FULL V_s stack (Phase 1-3 + MECH-284 hysteresis
+ use_vs_commit_release) against a VS_OFF baseline, BOTH under default
SP-CEM so diversity is measurable at E3.

Supersedes (interpretation)
---------------------------
V3-EXQ-476c (and the 476a/b chain): same scientific family but confounded
by pre-SP-CEM monostrategy. Does not supersede V3-EXQ-481b (forced hook
unit test); complements it with natural env interaction.

Arms
----
    VS_OFF:  SP-CEM main-path defaults; all V_s runtime flags OFF.
    VS_FULL: SP-CEM defaults + full invalidation runtime:
             use_per_stream_vs, use_per_region_vs, use_event_segmenter,
             use_invalidation_trigger, use_anchor_sets,
             use_staleness_accumulator, use_mech284_hysteresis,
             use_vs_commit_release=True.

Environment: CausalGridWorldV2 (10x10, 4 hazards, 4 resources) -- richer
than the 476c/481 2h/3r grid to allow varied contexts.

Pre-registered acceptance (PASS = all required)
-----------------------------------------------
    C0 (diversity floor): VS_OFF action_class_entropy >= 0.10 in >=2/3 seeds.
        If C0 fails -> non_contributory (substrate ceiling; MECH-269 untestable).

    C1 (V_s wiring): VS_FULL anchor_active_peak >= 1 AND anchor_reset_count > 0
        in >=2/3 seeds.

    C2 (per-region readout): VS_FULL per_region_vs_peak_keys >= 1 in >=2/3 seeds.

    C3 (behavioral delta): (entropy(VS_FULL) - entropy(VS_OFF)) >= 0.05 OR
        (freeze_recommit(VS_OFF) - freeze_recommit(VS_FULL)) >= 1 in >=2/3 seeds.

    C4 (read-side exercised): VS_FULL vs_commit_release_count > 0 in >=1/3 seeds.
        Failure of C4 alone is inconclusive (natural commitment may not cross in
        short runs; see 481b) and does not weaken MECH-269 substrate.

experiment_purpose=diagnostic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_TYPE = "v3_exq_596_mech269_post_sp_cem_behavioral_validation"
QUEUE_ID = "V3-EXQ-596"
CLAIM_IDS = ["MECH-269"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES_QUEUE = "V3-EXQ-476c"

SEEDS = [42, 7, 13]
CONDITIONS = ["VS_OFF", "VS_FULL"]
EPISODES = 8
STEPS_PER_EP = 200
FREEZE_RUN_LEN = 3

# Pre-registered thresholds (not inferred post-hoc).
MIN_BASELINE_ENTROPY = 0.10
ENTROPY_DELTA_THRESHOLD = 0.05
FREEZE_DROP_MIN = 1
SEEDS_PASS_MIN = 2
VS_RELEASE_SEEDS_MIN = 1


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    vs_on = condition == "VS_FULL"
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
        # SP-CEM main-path defaults intentionally omitted (ARC-065).
        use_per_stream_vs=vs_on,
        use_per_region_vs=vs_on,
        use_event_segmenter=vs_on,
        use_invalidation_trigger=vs_on,
        use_anchor_sets=vs_on,
        use_staleness_accumulator=vs_on,
        use_mech284_hysteresis=vs_on,
        use_vs_commit_release=vs_on,
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


def _obs_tensors(obs_dict: Dict) -> Tuple:
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict.get("harm_obs")
    if harm is not None:
        harm = harm.float().unsqueeze(0)
    harm_a = obs_dict.get("harm_obs_a")
    if harm_a is not None:
        harm_a = harm_a.float().unsqueeze(0)
    harm_hist = obs_dict.get("harm_history")
    if harm_hist is not None:
        harm_hist = harm_hist.float().unsqueeze(0)
    return body, world, harm, harm_a, harm_hist


def _run_condition(seed: int, condition: str, episodes: int, steps_per_ep: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, condition)

    action_counts: Dict[int, int] = {}
    action_seq: List[int] = []
    n_ticks = 0
    anchor_reset_count = 0
    boundary_event_count = 0
    broadcast_event_count = 0
    anchor_active_peak = 0
    per_region_vs_peak_keys = 0
    prev_active_keys: Set[str] = set()
    staleness_peaks: List[float] = []

    for ep in range(episodes):
        _obs, obs_dict = env.reset()
        prev_active_keys = set()
        ep_staleness_peak = 0.0

        if (ep + 1) % 2 == 0 or ep == episodes - 1:
            print(
                f"  [train] seed={seed} {condition} ep {ep + 1}/{episodes}",
                flush=True,
            )

        for _step in range(steps_per_ep):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body,
                obs_world=world,
                obs_harm=harm,
                obs_harm_a=harm_a,
                obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            action_seq.append(a_idx)
            _obs, _harm_signal, done, _info, obs_dict = env.step(action)
            n_ticks += 1

            if condition == "VS_FULL":
                hc = agent.hippocampal
                be_q = getattr(hc, "_boundary_event_queue", None)
                if be_q is not None:
                    boundary_event_count += len(be_q)
                br_q = getattr(hc, "_broadcast_event_queue", None)
                if br_q is not None:
                    broadcast_event_count += len(br_q)
                anchor_set = getattr(hc, "anchor_set", None)
                if anchor_set is not None:
                    active_now = {a.key for a in anchor_set.active_anchors()}
                    gone = prev_active_keys - active_now
                    anchor_reset_count += len(gone)
                    prev_active_keys = active_now
                    anchor_active_peak = max(
                        anchor_active_peak, len(active_now)
                    )
                pr_vs = getattr(hc, "per_region_vs", None)
                if isinstance(pr_vs, dict):
                    per_region_vs_peak_keys = max(
                        per_region_vs_peak_keys, len(pr_vs)
                    )
                sa = getattr(hc, "staleness_accumulator", None)
                if sa is not None:
                    snap = sa.snapshot()
                    if snap:
                        ep_staleness_peak = max(
                            ep_staleness_peak, max(snap.values())
                        )

            if done:
                break

        staleness_peaks.append(ep_staleness_peak)

    mean_staleness_peak = (
        sum(staleness_peaks) / len(staleness_peaks) if staleness_peaks else 0.0
    )

    return {
        "condition": condition,
        "seed": seed,
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": {str(k): v for k, v in action_counts.items()},
        "n_actions": sum(action_counts.values()),
        "n_ticks": n_ticks,
        "freeze_recommit_count": _count_freeze_runs(action_seq, FREEZE_RUN_LEN),
        "anchor_reset_count": anchor_reset_count,
        "boundary_event_count": boundary_event_count,
        "broadcast_event_count": broadcast_event_count,
        "anchor_active_peak": anchor_active_peak,
        "per_region_vs_peak_keys": per_region_vs_peak_keys,
        "vs_commit_release_count": int(
            getattr(agent, "_vs_commit_release_count", 0)
        ),
        "mean_staleness_peak": float(mean_staleness_peak),
        "sp_cem_main_path_defaults": True,
    }


def _seeds_passing(predicate) -> int:
    return sum(1 for s in SEEDS if predicate(s))


def _evaluate(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    off_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "VS_OFF"}
    on_by_seed = {r["seed"]: r for r in all_results if r["condition"] == "VS_FULL"}

    c0_per_seed = {
        s: off_by_seed[s]["action_class_entropy"] >= MIN_BASELINE_ENTROPY
        for s in SEEDS
    }
    c0_pass = _seeds_passing(lambda s: c0_per_seed[s]) >= SEEDS_PASS_MIN

    c1_per_seed = {
        s: (
            on_by_seed[s]["anchor_active_peak"] >= 1
            and on_by_seed[s]["anchor_reset_count"] > 0
        )
        for s in SEEDS
    }
    c1_pass = _seeds_passing(lambda s: c1_per_seed[s]) >= SEEDS_PASS_MIN

    c2_per_seed = {
        s: on_by_seed[s]["per_region_vs_peak_keys"] >= 1 for s in SEEDS
    }
    c2_pass = _seeds_passing(lambda s: c2_per_seed[s]) >= SEEDS_PASS_MIN

    c3_per_seed: Dict[int, bool] = {}
    for s in SEEDS:
        ent_delta = (
            on_by_seed[s]["action_class_entropy"]
            - off_by_seed[s]["action_class_entropy"]
        )
        freeze_drop = (
            off_by_seed[s]["freeze_recommit_count"]
            - on_by_seed[s]["freeze_recommit_count"]
        )
        c3_per_seed[s] = (
            ent_delta >= ENTROPY_DELTA_THRESHOLD
            or freeze_drop >= FREEZE_DROP_MIN
        )
    c3_pass = _seeds_passing(lambda s: c3_per_seed[s]) >= SEEDS_PASS_MIN

    c4_per_seed = {
        s: on_by_seed[s]["vs_commit_release_count"] > 0 for s in SEEDS
    }
    c4_pass = _seeds_passing(lambda s: c4_per_seed[s]) >= VS_RELEASE_SEEDS_MIN

    if not c0_pass:
        outcome = "FAIL"
        direction = "non_contributory"
        note = (
            "C0 failed: VS_OFF baseline entropy < "
            f"{MIN_BASELINE_ENTROPY} in too many seeds -- MECH-269 not "
            "testable on this run (substrate ceiling / SP-CEM diversity "
            "not achieved)."
        )
    elif c1_pass and c2_pass and c3_pass:
        outcome = "PASS"
        direction = "supports"
        note = (
            "Full V_s stack under SP-CEM: wiring active (C1/C2), measurable "
            "behavioral delta vs VS_OFF (C3). C4 commit-release may still be 0 "
            "without forced commitment."
        )
    else:
        outcome = "FAIL"
        direction = "inconclusive"
        failed = []
        if not c1_pass:
            failed.append("C1_wiring")
        if not c2_pass:
            failed.append("C2_per_region")
        if not c3_pass:
            failed.append("C3_behavioral_delta")
        note = (
            "Post-SP-CEM behavioral validation inconclusive: "
            + ", ".join(failed)
            + ". Does not weaken MECH-269 implementation; may need longer "
            "runs, richer env, or GAP-11 committed-mode curriculum."
        )

    return {
        "c0_baseline_entropy": {"per_seed": c0_per_seed, "pass": c0_pass},
        "c1_vs_wiring": {"per_seed": c1_per_seed, "pass": c1_pass},
        "c2_per_region_readout": {"per_seed": c2_per_seed, "pass": c2_pass},
        "c3_behavioral_delta": {"per_seed": c3_per_seed, "pass": c3_pass},
        "c4_vs_commit_release": {"per_seed": c4_per_seed, "pass": c4_pass},
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_note": note,
    }


def _print_plan(episodes: int, steps_per_ep: int) -> None:
    print(f"{EXPERIMENT_TYPE} -- MECH-269 post-SP-CEM validation", flush=True)
    print(f"queue_id={QUEUE_ID} supersedes={SUPERSEDES_QUEUE}", flush=True)
    print(f"Arms: {CONDITIONS}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Episodes x steps_per_ep: {episodes} x {steps_per_ep}", flush=True)
    print(
        f"C0 VS_OFF entropy>={MIN_BASELINE_ENTROPY} in >={SEEDS_PASS_MIN}/{len(SEEDS)} seeds",
        flush=True,
    )
    print(
        f"C1 VS_FULL anchor_active>=1 and anchor_resets>0 in >={SEEDS_PASS_MIN} seeds",
        flush=True,
    )
    print(
        f"C2 VS_FULL per_region_vs_peak_keys>=1 in >={SEEDS_PASS_MIN} seeds",
        flush=True,
    )
    print(
        f"C3 entropy delta>={ENTROPY_DELTA_THRESHOLD} OR freeze_drop>={FREEZE_DROP_MIN}",
        flush=True,
    )
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def run_experiment(
    dry_run: bool = False,
    episodes: int = EPISODES,
    steps_per_ep: int = STEPS_PER_EP,
) -> Dict[str, Any]:
    if dry_run:
        _print_plan(episodes, steps_per_ep)
        print("DRY RUN OK", flush=True)
        return {
            "outcome": "PASS",
            "manifest_path": None,
            "run_id": None,
            "dry_run": True,
            "all_pass": True,
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = EVIDENCE_ROOT / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}", flush=True)
            r = _run_condition(seed, cond, episodes, steps_per_ep)
            print(
                f"  -> entropy={r['action_class_entropy']:.4f} "
                f"freeze={r['freeze_recommit_count']} "
                f"anchor_peak={r['anchor_active_peak']} "
                f"resets={r['anchor_reset_count']} "
                f"per_region_keys={r['per_region_vs_peak_keys']} "
                f"vs_release={r['vs_commit_release_count']}",
                flush=True,
            )
            cell_pass = False
            if cond == "VS_OFF":
                cell_pass = r["action_class_entropy"] >= MIN_BASELINE_ENTROPY
            else:
                cell_pass = (
                    r["anchor_active_peak"] >= 1
                    and r["per_region_vs_peak_keys"] >= 1
                )
            print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
            all_results.append(r)

    eval_summary = _evaluate(all_results)
    outcome = eval_summary["outcome"]

    output: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": eval_summary["evidence_direction"],
        "evidence_direction_note": eval_summary["evidence_direction_note"],
        "supersedes": SUPERSEDES_QUEUE,
        "acceptance": eval_summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "episodes": episodes,
            "steps_per_ep": steps_per_ep,
            "thresholds": {
                "min_baseline_entropy": MIN_BASELINE_ENTROPY,
                "entropy_delta": ENTROPY_DELTA_THRESHOLD,
                "freeze_drop_min": FREEZE_DROP_MIN,
                "seeds_pass_min": SEEDS_PASS_MIN,
            },
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
    print(f"Output written to: {out_file}", flush=True)
    print(f"Outcome: {outcome} ({eval_summary['evidence_direction']})", flush=True)

    return {
        "outcome": outcome if outcome in ("PASS", "FAIL") else "FAIL",
        "manifest_path": str(out_file),
        "run_id": run_id,
        "dry_run": False,
        "all_pass": outcome == "PASS",
    }


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EP)
    args = parser.parse_args()
    return run_experiment(
        dry_run=args.dry_run,
        episodes=args.episodes,
        steps_per_ep=args.steps,
    )


if __name__ == "__main__":
    result = main()
    if not result["dry_run"]:
        emit_outcome(
            outcome=result["outcome"],
            manifest_path=result["manifest_path"],
            run_id=result["run_id"],
        )
    sys.exit(0 if result["all_pass"] else 1)
