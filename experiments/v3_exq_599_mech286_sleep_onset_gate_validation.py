"""
V3-EXQ-599: MECH-286 override-gated sleep-onset validation (substrate diagnostic).

3 arms x 1 seed (dry-run friendly): gate OFF (sleep always fires at K=1),
gate ON permissive (low override + high staleness + low harm_a), gate ON
hyperarousal lesion (saturated override blocks sleep despite high staleness).

SLEEP DRIVER: K=1 via SleepLoopManager.notify_episode_end at agent.reset().
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_599_mech286_sleep_onset_gate_validation"
QUEUE_ID = "V3-EXQ-599"
CLAIM_IDS: List[str] = ["MECH-286"]
EXPERIMENT_PURPOSE = "diagnostic"

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    seeds = SEEDS if args.seeds is None else list(args.seeds)

    results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            results.append(run_arm(arm, seed))

    verdict = evaluate(results)
    outcome = "PASS" if verdict["passed"] else "FAIL"
    print(f"VERDICT: {outcome}")
    for k, v in verdict["checks"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    if args.dry_run:
        return

    emit_outcome(
        experiment_type=EXPERIMENT_TYPE,
        queue_id=QUEUE_ID,
        claim_ids=CLAIM_IDS,
        outcome=outcome,
        run_id=f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        architecture_epoch="ree_hybrid_guardrails_v1",
        results={"arm_results": results, "verdict": verdict},
        experiment_purpose=EXPERIMENT_PURPOSE,
    )


if __name__ == "__main__":
    main()
