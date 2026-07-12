"""
V3-EXQ-599a: MECH-286 override-gated sleep-onset validation (substrate diagnostic).

Fix for V3-EXQ-599 ERROR: emit_outcome() used removed keyword args; no runner
sentinel written despite PASS/FAIL verdict.

3 arms x 1 seed: gate OFF (sleep at K=1), gate ON permissive, gate ON
hyperarousal lesion (saturated override blocks sleep).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_TYPE = "v3_exq_599a_mech286_sleep_onset_gate_validation"
QUEUE_ID = "V3-EXQ-599a"
CLAIM_IDS: List[str] = ["MECH-286"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "V3-EXQ-599"

SEEDS = [42]
OVERRIDE_TICKS = 50

ARMS = [
    {
        "arm": "ARM_0_gate_off",
        "use_mech286_sleep_onset_gate": False,
        "saturate_override": False,
        "staleness_bump": 0.8,
        "expect_permitted": None,
        "expect_cycle": True,
    },
    {
        "arm": "ARM_1_gate_on_permit",
        "use_mech286_sleep_onset_gate": True,
        "saturate_override": False,
        "staleness_bump": 0.8,
        "expect_permitted": 1.0,
        "expect_cycle": True,
    },
    {
        "arm": "ARM_2_gate_on_hyperarousal",
        "use_mech286_sleep_onset_gate": True,
        "saturate_override": True,
        "staleness_bump": 0.9,
        "expect_permitted": 0.0,
        "expect_cycle": False,
    },
]


def _build_agent(arm: Dict[str, Any]) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        sws_enabled=True,
        rem_enabled=False,
        use_staleness_accumulator=True,
        use_e2_harm_a=True,
        use_broadcast_override=bool(arm.get("saturate_override", False)),
        use_mech286_sleep_onset_gate=bool(arm["use_mech286_sleep_onset_gate"]),
    )
    return REEAgent(cfg)


def _bump_staleness(agent: REEAgent, value: float) -> None:
    acc = agent.hippocampal.staleness_accumulator
    if acc is None:
        raise RuntimeError("staleness accumulator required for MECH-286 validation")
    acc._staleness[("fast", "0.0")] = float(value)


def _saturate_override(agent: REEAgent) -> None:
    reg = agent.broadcast_override
    if reg is None:
        raise RuntimeError("broadcast_override required for hyperarousal arm")
    for _ in range(OVERRIDE_TICKS):
        reg.tick(drive_level=0.95, z_harm_norm=0.85)


def run_arm(arm: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    agent = _build_agent(arm)
    _bump_staleness(agent, float(arm["staleness_bump"]))
    if arm.get("saturate_override"):
        _saturate_override(agent)
    before = agent.sleep_loop.state.cycle_index
    agent.reset()
    after = agent.sleep_loop.state.cycle_index
    metrics = dict(agent.sleep_loop.state.last_metrics or {})
    permitted = float(metrics.get("mech286_sleep_permitted", 1.0))
    if not arm["use_mech286_sleep_onset_gate"]:
        permitted = 1.0
    cycle_fired = after > before
    return {
        "arm": arm["arm"],
        "seed": seed,
        "cycle_index_before": before,
        "cycle_index_after": after,
        "cycle_fired": cycle_fired,
        "mech286_sleep_permitted": permitted,
        "override_signal": float(metrics.get("mech286_override_signal", 0.0)),
        "staleness_max": float(metrics.get("mech286_staleness_max", 0.0)),
    }


def evaluate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    for arm in ARMS:
        row = next(r for r in results if r["arm"] == arm["arm"])
        if arm["expect_cycle"] is not None:
            checks[f"{arm['arm']}_cycle"] = row["cycle_fired"] == arm["expect_cycle"]
        if arm["expect_permitted"] is not None:
            checks[f"{arm['arm']}_permitted"] = (
                row["mech286_sleep_permitted"] == arm["expect_permitted"]
            )
    passed = all(checks.values())
    return {"checks": checks, "passed": passed}


def main(dry_run: bool = False, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    use_seeds = SEEDS if seeds is None else seeds
    results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in use_seeds:
            results.append(run_arm(arm, seed))

    verdict = evaluate(results)
    outcome = "PASS" if verdict["passed"] else "FAIL"
    print(f"verdict: {outcome}")
    for k, v in verdict["checks"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": "supports" if verdict["passed"] else "weakens",
        "evidence_direction_per_claim": {
            "MECH-286": "supports" if verdict["passed"] else "weakens",
        },
        "evidence_direction_note": (
            "MECH-286 sleep-onset gate substrate validation (diagnostic). "
            "Fix for V3-EXQ-599 emit_outcome API mismatch only; same arms."
        ),
        "outcome": outcome,
        "metrics": {"arm_results": results, "verdict": verdict},
        "dry_run": bool(dry_run),
    }

    out_path = None
    if not dry_run:
        out_dir = EVIDENCE_ROOT / EXPERIMENT_TYPE
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}", flush=True)

    return {
        "all_pass": bool(verdict["passed"]),
        "outcome": outcome,
        "manifest_path": str(out_path) if out_path is not None else None,
        "run_id": run_id,
        "dry_run": bool(dry_run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    result = main(dry_run=args.dry_run, seeds=args.seeds)
    if not result["dry_run"]:
        emit_outcome(
            outcome=result["outcome"],
            manifest_path=result["manifest_path"],
            run_id=result["run_id"],
            queue_id=QUEUE_ID,
        )
    sys.exit(0 if result["all_pass"] else 1)
