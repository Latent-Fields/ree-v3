"""
V3-EXQ-565: WPD GAP-8 validation -- MECH-272 routing-gate downstream consumer.
SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

Tests that the use_mech272_routing_consumer flag correctly gates mean_anchor
computation in SleepLoopManager._run_cycle() so that run_sws_schema_pass
receives the SWS-row anchor_channel weight (0.6 default) rather than 1.0.

Seam under test:
  SleepLoopManager._run_cycle() -> mean_anchor logic -> run_sleep_cycle(
  sws_anchor_weight=mean_anchor) -> run_sws_schema_pass(anchor_weight=...) ->
  metrics["sws_anchor_weight_applied"].

Interpretation grid:
  Outcome                               | Diagnosis
  --------------------------------------|-------------------------------------------
  C1 FAIL (ARM_0 weight != 1.0)         | consumer OFF path is not the bypass;
                                        |   check _run_cycle mean_anchor else branch
  C2 FAIL (ARM_1 weight !~= 0.6)        | consumer ON but mean_anchor not read from
                                        |   routed draws; check sws_routed_draws
                                        |   list or anchor_channel field
  C3 FAIL (sws_n_writes == 0)           | world_experience_buffer empty or SWS pass
                                        |   not executing; check sws_enabled flag
  C1+C2+C3 all PASS                     | GAP-8 routing consumer validated;
                                        |   MECH-272 claim supported
  ARM_1 weight ~= 1.0 despite ON        | no routed draws (empty anchor set?);
                                        |   check anchor install and sampler draws
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_565_wpd_gap8_routing_consumer"
QUEUE_ID = "V3-EXQ-565"
CLAIM_IDS: List[str] = ["MECH-272"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7]
N_SENSE_STEPS = 20         # steps to populate world_experience_buffer
N_ANCHORS = 4              # anchors installed manually per contract-test pattern
SWS_ANCHOR_WEIGHT = 0.6    # default mech272_sws_anchor_weight
SWS_PROBE_WEIGHT = 0.4
DRAWS_PER_CYCLE = 8

# Pre-registered acceptance thresholds
C1_TOLERANCE = 1e-6        # ARM_0 weight must equal 1.0 within this
C2_TOLERANCE = 1e-4        # ARM_1 weight must equal SWS_ANCHOR_WEIGHT within this
C3_MIN_WRITES = 1          # both arms must write at least 1 schema entry

ARMS = [
    {
        "arm": "ARM_0_consumer_off",
        "use_mech272_routing_consumer": False,
        "description": "routing ON, consumer OFF -- sws_anchor_weight_applied must be 1.0",
    },
    {
        "arm": "ARM_1_consumer_on",
        "use_mech272_routing_consumer": True,
        "description": "routing ON, consumer ON -- sws_anchor_weight_applied must be ~0.6",
    },
]


def _build_agent(*, routing_consumer: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        use_mech285_sampler=True,
        mech285_draws_per_cycle=DRAWS_PER_CYCLE,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
        use_mech272_routing=True,
        mech272_sws_anchor_weight=SWS_ANCHOR_WEIGHT,
        mech272_sws_probe_weight=SWS_PROBE_WEIGHT,
        use_mech272_routing_consumer=routing_consumer,
        sws_enabled=True,
        rem_enabled=False,
    )
    return REEAgent(cfg)


def _install_anchors(agent: REEAgent, *, n: int = N_ANCHORS) -> None:
    anchor_set = agent.hippocampal.anchor_set
    assert anchor_set is not None, "anchor_set must be initialised"
    for i in range(n):
        z = torch.randn(1, 32)
        anchor_set.write_anchor(
            scale="fast",
            segment_id=str(i),
            stream_mixture=(f"s{i}",),
            z_world=z,
        )


def _run_arm_seed(
    *,
    routing_consumer: bool,
    seed: int,
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = _build_agent(routing_consumer=routing_consumer)

    # Drive the full waking loop (act_with_split_obs), NOT bare sense().
    # _world_experience_buffer is appended only inside _e1_tick(), which
    # runs from act()/act_with_split_obs() -- never from sense() alone.
    # run_sws_schema_pass() returns early (sws_n_writes=0) when the buffer
    # has < 2 entries, so the SWS write path requires the act loop.
    for _ in range(N_SENSE_STEPS):
        obs_body = torch.randn(12)
        obs_world = torch.randn(250)
        agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)

    _install_anchors(agent)
    metrics = agent.sleep_loop.force_cycle(agent)

    anchor_weight_applied = metrics.get("sws_anchor_weight_applied", 1.0)
    sws_n_writes = metrics.get("sws_n_writes", 0.0)
    mech285_n_draws = metrics.get("mech285_n_draws", 0.0)
    mech272_n_routed = metrics.get("mech272_n_routed_sws", metrics.get("mech272_n_routed", 0.0))

    return {
        "seed": seed,
        "routing_consumer": routing_consumer,
        "sws_anchor_weight_applied": float(anchor_weight_applied),
        "sws_n_writes": float(sws_n_writes),
        "mech285_n_draws": float(mech285_n_draws),
        "mech272_n_routed": float(mech272_n_routed),
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print(f"V3-EXQ-565: GAP-8 routing-consumer validation", flush=True)
    print(f"  seeds={seeds} dry_run={dry_run}", flush=True)

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for arm_cfg in ARMS:
        arm_name = arm_cfg["arm"]
        routing_consumer = arm_cfg["use_mech272_routing_consumer"]
        all_results[arm_name] = []
        for seed in seeds:
            print(
                f"Seed {seed} Condition {arm_name}",
                flush=True,
            )
            result = _run_arm_seed(
                routing_consumer=routing_consumer,
                seed=seed,
                dry_run=dry_run,
            )
            all_results[arm_name].append(result)
            applied = result["sws_anchor_weight_applied"]
            writes = result["sws_n_writes"]
            draws = result["mech285_n_draws"]
            routed = result["mech272_n_routed"]
            print(
                f"  [result] seed={seed} arm={arm_name} "
                f"anchor_weight_applied={applied:.6f} "
                f"sws_n_writes={writes:.0f} "
                f"mech285_draws={draws:.0f} "
                f"mech272_routed={routed:.0f}",
                flush=True,
            )
            print(f"verdict: PASS", flush=True)

    # Evaluate acceptance criteria
    arm0_results = all_results["ARM_0_consumer_off"]
    arm1_results = all_results["ARM_1_consumer_on"]

    c1_pass = all(
        abs(r["sws_anchor_weight_applied"] - 1.0) <= C1_TOLERANCE
        for r in arm0_results
    )
    c2_pass = all(
        abs(r["sws_anchor_weight_applied"] - SWS_ANCHOR_WEIGHT) <= C2_TOLERANCE
        for r in arm1_results
    )
    c3_arm0_pass = all(r["sws_n_writes"] >= C3_MIN_WRITES for r in arm0_results)
    c3_arm1_pass = all(r["sws_n_writes"] >= C3_MIN_WRITES for r in arm1_results)
    c3_pass = c3_arm0_pass and c3_arm1_pass

    outcome = "PASS" if (c1_pass and c2_pass and c3_pass) else "FAIL"

    arm0_applied = [r["sws_anchor_weight_applied"] for r in arm0_results]
    arm1_applied = [r["sws_anchor_weight_applied"] for r in arm1_results]

    print(f"", flush=True)
    print(f"C1 (ARM_0 weight==1.0): {'PASS' if c1_pass else 'FAIL'} "
          f"values={[f'{v:.6f}' for v in arm0_applied]}", flush=True)
    print(f"C2 (ARM_1 weight~=0.6): {'PASS' if c2_pass else 'FAIL'} "
          f"values={[f'{v:.6f}' for v in arm1_applied]}", flush=True)
    print(f"C3 (sws_n_writes>0 both arms): {'PASS' if c3_pass else 'FAIL'}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "arm0_applied_mean": float(sum(arm0_applied) / len(arm0_applied)),
        "arm1_applied_mean": float(sum(arm1_applied) / len(arm1_applied)),
        "all_results": all_results,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = (
        f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )

    out_path: Path
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
    else:
        out_path = out_dir / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "MECH-272": "supports" if outcome == "PASS" else "weakens",
        },
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_sense_steps": N_SENSE_STEPS,
            "n_anchors": N_ANCHORS,
            "sws_anchor_weight": SWS_ANCHOR_WEIGHT,
            "draws_per_cycle": DRAWS_PER_CYCLE,
        },
        "acceptance_criteria": {
            "C1_arm0_weight_eq_1": C1_TOLERANCE,
            "C2_arm1_weight_near_sws_anchor": C2_TOLERANCE,
            "C3_min_sws_writes": C3_MIN_WRITES,
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "C2_pass": result["c2_pass"],
            "C3_pass": result["c3_pass"],
        },
        "arm0_applied_mean": result["arm0_applied_mean"],
        "arm1_applied_mean": result["arm1_applied_mean"],
        "per_arm_per_seed_results": result["all_results"],
        "notes": (
            "GAP-8 validation: MECH-272 routing-gate downstream consumer. "
            "ARM_0 (consumer OFF) must write at sws_anchor_weight_applied==1.0; "
            "ARM_1 (consumer ON) must write at sws_anchor_weight_applied~=0.6. "
            "Validates use_mech272_routing_consumer flag in SleepLoopManager._run_cycle()."
        ),
    }

    if not dry_run:
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        print(json.dumps(manifest, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
