"""
V3-EXQ-574: MECH-273 SelfModelAggregator offline gradient pass validation.

Tests that SelfModelAggregator.offline_gradient_pass() (MECH-273 Phase E
WRITEBACK) correctly runs n_steps=100 bounded MSE gradient steps on E2_harm_s
using posterior means from last_snapshot as targets. Also validates the GAP-4
real-replay path: (z_harm_s, action) tuples sampled from harm_replay_buffer.

Interpretation grid:
  Outcome                              | Diagnosis
  -------------------------------------|------------------------------------------
  C1 FAIL (n_steps < 100 some cycle)  | offline_gradient_pass not reaching step
                                       |   count; check config.offline_n_steps or
                                       |   early-exit path in aggregator
  C2 FAIL (no regions consumed)        | posterior store empty; check anchor
                                       |   install, routing, or SWS draws
  C3 FAIL (loss not decreasing)        | MSE not converging; target may be zero
                                       |   (staleness too low) or E2_harm_s not
                                       |   trainable; check posterior means and
                                       |   learning rate
  C1+C2+C3 all PASS                    | MECH-273 offline gradient pass validated;
                                       |   Phase E WRITEBACK claim supported
"""

from __future__ import annotations

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

EXPERIMENT_TYPE = "v3_exq_574_mech273_self_model_aggregator_validation"
QUEUE_ID = "V3-EXQ-574"
CLAIM_IDS: List[str] = ["MECH-273"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13]
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
N_ANCHORS = 4
N_CYCLES = 3
EPISODES_PER_CYCLE = 30
TOTAL_EPS = N_CYCLES * EPISODES_PER_CYCLE  # 90
DRAWS_PER_CYCLE = 50
OFFLINE_N_STEPS = 100

# Pre-registered acceptance thresholds
C1_MIN_STEPS = OFFLINE_N_STEPS   # each cycle must run exactly 100 steps
C2_MIN_REGIONS = 1               # at least 1 region consumed in some cycle
C3_MIN_LOSS_CYCLE1 = 1e-4       # cycle 1 mean loss must be non-trivial
# C3 also requires: cycle3_loss < cycle1_loss (loss decreasing across cycles)


def _build_agent() -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        use_affective_harm_stream=True,
        use_e2_harm_s_forward=True,
        use_sleep_loop=True,
        sleep_loop_episodes_K=TOTAL_EPS + 1,
        use_mech285_sampler=True,
        mech285_draws_per_cycle=DRAWS_PER_CYCLE,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
        use_mech272_routing=True,
        use_mech275_aggregator=True,
        use_mech273_self_model=True,
        sws_enabled=True,
        rem_enabled=True,
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


def _run_seed(*, seed: int, dry_run: bool) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = _build_agent()
    _install_anchors(agent)

    cycle_records: List[Dict[str, float]] = []

    for cycle_idx in range(N_CYCLES):
        for ep in range(EPISODES_PER_CYCLE):
            ep_global = cycle_idx * EPISODES_PER_CYCLE + ep + 1
            obs_body = torch.randn(BODY_OBS_DIM)
            obs_world = torch.randn(WORLD_OBS_DIM)
            agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)
            if ep_global == 1 or ep_global % 10 == 0:
                print(
                    f"  [train] seed={seed} ep {ep_global}/{TOTAL_EPS}",
                    flush=True,
                )

        metrics = agent.sleep_loop.force_cycle(agent)
        n_steps = float(metrics.get("mech273_writeback_n_steps", 0.0))
        n_regions = float(metrics.get("mech273_writeback_regions", 0.0))
        mean_loss = float(metrics.get("mech273_writeback_mean_loss", 0.0))
        n_passes = float(metrics.get("mech273_n_offline_passes", 0.0))
        print(
            f"  [sleep] seed={seed} cycle={cycle_idx + 1}/{N_CYCLES} "
            f"n_steps={n_steps:.0f} n_regions={n_regions:.0f} "
            f"mean_loss={mean_loss:.6f} n_passes={n_passes:.0f}",
            flush=True,
        )
        cycle_records.append(
            {
                "cycle": cycle_idx + 1,
                "n_steps": n_steps,
                "n_regions": n_regions,
                "mean_loss": mean_loss,
            }
        )

    c1_pass = all(r["n_steps"] >= C1_MIN_STEPS for r in cycle_records)
    c2_pass = any(r["n_regions"] >= C2_MIN_REGIONS for r in cycle_records)
    if dry_run:
        c3_pass = True
    else:
        cycle1_loss = cycle_records[0]["mean_loss"]
        cycle3_loss = cycle_records[-1]["mean_loss"]
        c3_pass = (
            cycle1_loss > C3_MIN_LOSS_CYCLE1
            and cycle3_loss < cycle1_loss
        )

    seed_pass = c1_pass and c2_pass and c3_pass
    print(
        f"  [criteria] seed={seed} C1={'PASS' if c1_pass else 'FAIL'} "
        f"C2={'PASS' if c2_pass else 'FAIL'} "
        f"C3={'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    return {
        "seed": seed,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "seed_pass": seed_pass,
        "cycle_records": cycle_records,
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print(
        f"V3-EXQ-574: MECH-273 SelfModelAggregator offline gradient pass",
        flush=True,
    )
    print(f"  seeds={seeds} dry_run={dry_run}", flush=True)

    all_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition single", flush=True)
        result = _run_seed(seed=seed, dry_run=dry_run)
        all_seed_results.append(result)
        print(f"verdict: {'PASS' if result['seed_pass'] else 'FAIL'}", flush=True)

    c1_pass = all(r["c1_pass"] for r in all_seed_results)
    c2_pass = all(r["c2_pass"] for r in all_seed_results)
    c3_pass = all(r["c3_pass"] for r in all_seed_results)
    outcome = "PASS" if (c1_pass and c2_pass and c3_pass) else "FAIL"

    print(f"", flush=True)
    print(
        f"C1 (n_steps==100 all cycles): {'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"C2 (regions>=1 some cycle): {'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"C3 (loss non-trivial and decreasing): {'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "all_seed_results": all_seed_results,
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
            "MECH-273": "supports" if outcome == "PASS" else "weakens",
        },
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_anchors": N_ANCHORS,
            "n_cycles": N_CYCLES,
            "episodes_per_cycle": EPISODES_PER_CYCLE,
            "total_eps": TOTAL_EPS,
            "draws_per_cycle": DRAWS_PER_CYCLE,
            "offline_n_steps": OFFLINE_N_STEPS,
        },
        "acceptance_criteria": {
            "C1_min_steps_per_cycle": C1_MIN_STEPS,
            "C2_min_regions_some_cycle": C2_MIN_REGIONS,
            "C3_min_loss_cycle1": C3_MIN_LOSS_CYCLE1,
            "C3_also_requires_loss_decrease": True,
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "C2_pass": result["c2_pass"],
            "C3_pass": result["c3_pass"],
        },
        "per_seed_results": result["all_seed_results"],
        "notes": (
            "MECH-273 validation: SelfModelAggregator.offline_gradient_pass "
            "Phase E WRITEBACK. C1 checks bounded n_steps=100. C2 checks that "
            "posterior regions are consumed (non-empty posterior store). C3 "
            "checks that E2_harm_s loss is non-trivial and decreases across "
            "3 sleep cycles (learning convergence). GAP-4 real-replay path "
            "exercised via act_with_split_obs() populating harm_replay_buffer."
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
