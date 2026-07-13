#!/opt/local/bin/python3
"""V3-EXQ-563a -- scaffolded forced action-bias actuator retest.

Lettered successor to V3-EXQ-563. The scientific question is unchanged:
does a forced score-bias by first-action class move the selected/executed
action distribution? V3-EXQ-563 was useful but not interpretable for that
question because CEM returned a one-class candidate set before E3 selection.

This successor keeps the original six-arm actuator design but enables
HippocampalConfig.use_action_class_scaffold_candidates for every arm. That
injects one one-hot first-action candidate per action class before E3.select()
so the forced-score-bias hook is guaranteed a real candidate surface.

Changes from V3-EXQ-563:
  - queue_id: V3-EXQ-563a
  - experiment_type: v3_exq_563a_action_bias_scaffold_actuator_test
  - supersedes: v3_exq_563_action_bias_actuator_test_20260514T183416Z_v3
  - use_action_class_scaffold_candidates=True
  - forced nonzero vector corrected to five classes and strengthened:
      [-100.0, 0.0, 0.0, 0.0, 0.0]

Acceptance:
  P1 forced bias changes action distribution in the expected direction.
  P2 natural tonic-vigor v_t remains zero.
  P3 v_t_floor still produces nonzero v_t.
  P4 scaffold produces candidate_unique_first_action_classes >= 2 in all cells.
  P5 forced nonzero arm has forced_bias_abs_mean > 0.

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_563a_action_bias_scaffold_actuator_test.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments import v3_exq_563_action_bias_actuator_test as base  # noqa: E402

MANIFEST_WRITER_EXEMPT = "archival early-era manifest (non-canonical filename not provably == run_id.json; superseded lineage, not re-run)"


EXPERIMENT_TYPE = "v3_exq_563a_action_bias_scaffold_actuator_test"
QUEUE_ID = "V3-EXQ-563a"
SUPERSEDES_RUN_ID = "v3_exq_563_action_bias_actuator_test_20260514T183416Z_v3"
SUPERSEDES_QUEUE_ID = "V3-EXQ-563"
FORCED_NONZERO_BIAS = [-100.0, 0.0, 0.0, 0.0, 0.0]
FORCED_ZERO_BIAS = [0.0, 0.0, 0.0, 0.0, 0.0]

_BASE_MAKE_CONFIG = base._make_config


def _successor_arms() -> List[Dict[str, Any]]:
    arms = copy.deepcopy(base.ARMS)
    for arm in arms:
        fsb = arm.get("forced_score_bias_per_class")
        if fsb is None:
            continue
        if any(float(v) != 0.0 for v in fsb):
            arm["forced_score_bias_per_class"] = list(FORCED_NONZERO_BIAS)
        else:
            arm["forced_score_bias_per_class"] = list(FORCED_ZERO_BIAS)
    return arms


def _make_config_with_scaffold(env, arm):  # type: ignore[no-untyped-def]
    cfg = _BASE_MAKE_CONFIG(env, arm)
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    return cfg


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _augment_manifest(manifest: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    arm_results = manifest.get("arm_results", [])
    arm4_rows = [
        r for r in arm_results
        if r.get("arm") == "ARM_4_forced_nonzero"
    ]
    all_unique = [
        len(r.get("candidate_first_action_counts", {}))
        for r in arm_results
    ]
    p4 = bool(all_unique and min(all_unique) >= 2)
    arm4_forced_bias_abs_mean = _mean(arm4_rows, "forced_bias_abs_mean")
    p5 = bool(arm4_forced_bias_abs_mean > 0.0)

    manifest["supersedes"] = SUPERSEDES_RUN_ID
    manifest["supersedes_queue_id"] = SUPERSEDES_QUEUE_ID
    manifest["diagnostic_scaffold"] = {
        "use_action_class_scaffold_candidates": True,
        "forced_nonzero_bias_per_class": list(FORCED_NONZERO_BIAS),
        "p4_candidate_support_surface_present": p4,
        "p5_forced_nonzero_bias_applied": p5,
        "arm4_forced_bias_abs_mean": round(arm4_forced_bias_abs_mean, 6),
        "min_candidate_unique_first_action_classes": (
            min(all_unique) if all_unique else 0
        ),
    }
    manifest.setdefault("acceptance_criteria", {})
    manifest["acceptance_criteria"].update({
        "P4_candidate_support_surface_present": p4,
        "P5_forced_nonzero_bias_applied": p5,
    })
    manifest.setdefault("summary", {})
    manifest["summary"].update({
        "scaffold_enabled_all_arms": True,
        "forced_nonzero_bias_per_class": list(FORCED_NONZERO_BIAS),
        "p4_candidate_support_surface_present": p4,
        "p5_forced_nonzero_bias_applied": p5,
        "arm4_forced_bias_abs_mean": round(arm4_forced_bias_abs_mean, 6),
    })

    if not dry_run:
        out_path = Path(str(manifest["manifest_path"]))
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Result written to: {out_path}", flush=True)

    return manifest


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    base.EXPERIMENT_TYPE = EXPERIMENT_TYPE
    base.QUEUE_ID = QUEUE_ID
    base.ARMS = _successor_arms()
    base._make_config = _make_config_with_scaffold
    manifest = base.run_experiment(dry_run=dry_run)
    return _augment_manifest(manifest, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-563a scaffolded forced action-bias actuator retest"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run (1 seed, 1 episode, 50 steps); no manifest written.",
    )
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
