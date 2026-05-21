"""V3-EXQ-607: MECH-340 Persistence / Efficacy Gate -- Substrate Validation.

experiment_purpose: diagnostic
status_when_drafted: lands with the MECH-340 substrate (smallest step,
2026-05-21). Substrate readiness gate; NOT governance evidence.

CLAIM TESTED:
MECH-340 (hippocampal.persistence_efficacy_gate) -- ARC-079 / Q-053
front-runner. Persistence of a ghost-goal entry as an active re-probe
target is gated by a global control/efficacy unattainability appraisal;
disengagement is the default when the gate withholds license.

Pure-arithmetic validation over deterministic AnchorSet pools (no env,
no training). Three GhostGoalBankConfig conditions on the SAME pool.

DESIGN: 5 sub-tests.

  T1 backward_compat_gate_off: use_persistence_efficacy_gate=False (default)
     -> rank() is unchanged when a disengaging appraisal is passed (gate
     ignored); no persistence_license component key.

  T2 persistence_licensed_admits: control=1, unattainability=0 -> anchor
     admitted with persistence_license=1.0 in components.

  T3 disengaged_excludes: control=0 OR unattainability=1 -> empty bank;
     diagnostics n_below_persistence >= 1.

  T4 recoverability_invariant_under_gate: two anchors with identical z_goal
     match but different last_vs; same disengaging appraisal -> BOTH
     excluded (gate does not key on recoverability).

  T5 stuck_on_rumination_signature: with persistence_floor=0.5, appraisal
     control=1 unattainability=1 -> license 0 but both anchors still admitted
     (gate stuck open / wrong signal would admit under disengagement).

PASS CRITERIA: T1 AND T2 AND T3 AND T4 AND T5.

claim_ids: ['MECH-340']

Run with:
  /opt/local/bin/python3 experiments/v3_exq_607_mech340_persistence_efficacy_gate_validation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from ree_core.hippocampal.anchor_set import Anchor, AnchorGoalPayload, AnchorSet
from ree_core.hippocampal.ghost_goal_bank import (
    GhostGoalBank,
    GhostGoalBankConfig,
    PersistenceAppraisal,
)
from ree_core.utils.config import AnchorSetConfig

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_PURPOSE = "diagnostic"

CURRENT_Z_GOAL = torch.tensor([1.0, 0.0, 0.0, 0.0])
TOL = 1e-9


def _make_anchor(seg: str, zsnap: list[float], last_vs: float) -> Anchor:
    a = Anchor(key=("fast", seg, ("s",)), z_world=torch.zeros(4), active=False)
    a.goal_payload = AnchorGoalPayload(
        z_goal_snapshot=torch.tensor(zsnap, dtype=torch.float32).unsqueeze(0),
        wanting_strength=0.3,
        arousal_tag=0.1,
        last_vs=last_vs,
        staleness_at_write=0.0,
        payload_written_step=0,
    )
    return a


def _bank(anchors: list[Anchor], cfg: GhostGoalBankConfig) -> GhostGoalBank:
    s = AnchorSet(AnchorSetConfig())
    s._all = {a.key: a for a in anchors}
    return GhostGoalBank(cfg, s)


def _cfg_off() -> GhostGoalBankConfig:
    return GhostGoalBankConfig()


def _cfg_on() -> GhostGoalBankConfig:
    return GhostGoalBankConfig(use_persistence_efficacy_gate=True)


def run_t1_backward_compat() -> Dict[str, Any]:
    try:
        a = _make_anchor("A", [1.0, 0.0, 0.0, 0.0], 0.6)
        bank = _bank([a], _cfg_off())
        base = bank.rank(CURRENT_Z_GOAL)
        stressed = bank.rank(
            CURRENT_Z_GOAL,
            persistence_appraisal=PersistenceAppraisal(
                control_efficacy=0.0,
                goal_unattainability=1.0,
            ),
        )
        ok = (
            len(base) == 1
            and len(stressed) == 1
            and abs(base[0].ghost_priority - stressed[0].ghost_priority) < TOL
            and "persistence_license" not in base[0].components
        )
        return {"pass": bool(ok), "n_base": len(base), "n_stressed": len(stressed)}
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


def run_t2_licensed_admits() -> Dict[str, Any]:
    try:
        a = _make_anchor("A", [1.0, 0.0, 0.0, 0.0], 0.6)
        bank = _bank([a], _cfg_on())
        entries = bank.rank(
            CURRENT_Z_GOAL,
            persistence_appraisal=PersistenceAppraisal(
                control_efficacy=1.0,
                goal_unattainability=0.0,
            ),
        )
        lic = entries[0].components.get("persistence_license") if entries else None
        ok = len(entries) == 1 and lic is not None and abs(lic - 1.0) < TOL
        return {"pass": bool(ok), "n_entries": len(entries), "license": lic}
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


def run_t3_disengaged_excludes() -> Dict[str, Any]:
    try:
        a = _make_anchor("A", [1.0, 0.0, 0.0, 0.0], 0.6)
        bank = _bank([a], _cfg_on())
        entries = bank.rank(
            CURRENT_Z_GOAL,
            persistence_appraisal=PersistenceAppraisal(
                control_efficacy=0.0,
                goal_unattainability=1.0,
            ),
        )
        diag = bank.get_diagnostics()
        ok = len(entries) == 0 and int(diag.get("n_below_persistence", 0)) >= 1
        return {
            "pass": bool(ok),
            "n_entries": len(entries),
            "n_below_persistence": int(diag.get("n_below_persistence", 0)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


def run_t4_recoverability_invariant() -> Dict[str, Any]:
    try:
        hi = _make_anchor("Hi", [1.0, 0.0, 0.0, 0.0], 0.95)
        lo = _make_anchor("Lo", [1.0, 0.0, 0.0, 0.0], 0.05)
        appraisal = PersistenceAppraisal(
            control_efficacy=0.0,
            goal_unattainability=1.0,
        )
        bank = _bank([hi, lo], _cfg_on())
        entries = bank.rank(CURRENT_Z_GOAL, persistence_appraisal=appraisal)
        ok = len(entries) == 0
        return {"pass": bool(ok), "n_entries": len(entries)}
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


def run_t5_stuck_on_signature() -> Dict[str, Any]:
    """High unattainability with control=1 -> license 0; floor 0.5 excludes."""
    try:
        hi = _make_anchor("Hi", [1.0, 0.0, 0.0, 0.0], 0.95)
        lo = _make_anchor("Lo", [1.0, 0.0, 0.0, 0.0], 0.05)
        cfg = GhostGoalBankConfig(
            use_persistence_efficacy_gate=True,
            persistence_floor=0.5,
        )
        bank = _bank([hi, lo], cfg)
        entries = bank.rank(
            CURRENT_Z_GOAL,
            persistence_appraisal=PersistenceAppraisal(
                control_efficacy=1.0,
                goal_unattainability=1.0,
            ),
        )
        lic = bank.get_diagnostics().get("persistence_license")
        ok = len(entries) == 0 and lic is not None and float(lic) < 0.5
        return {
            "pass": bool(ok),
            "n_entries": len(entries),
            "persistence_license": lic,
        }
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], bool]:
    if dry_run:
        t1 = run_t1_backward_compat()
        return {"T1_backward_compat_gate_off": t1}, bool(t1["pass"])
    metrics = {
        "T1_backward_compat_gate_off": run_t1_backward_compat(),
        "T2_persistence_licensed_admits": run_t2_licensed_admits(),
        "T3_disengaged_excludes": run_t3_disengaged_excludes(),
        "T4_recoverability_invariant": run_t4_recoverability_invariant(),
        "T5_stuck_on_excludes": run_t5_stuck_on_signature(),
    }
    all_pass = all(m["pass"] for m in metrics.values())
    return metrics, all_pass


def main(dry_run: bool = False) -> Dict[str, Any]:
    print(
        "[v3_exq_607] MECH-340 persistence/efficacy gate validation...",
        flush=True,
    )
    metrics, all_pass = run_experiment(dry_run=dry_run)
    for name, m in metrics.items():
        print(f"  {name}: {'PASS' if m['pass'] else 'FAIL'}  {m}", flush=True)
    print(f"[v3_exq_607] overall: {'PASS' if all_pass else 'FAIL'}", flush=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"v3_exq_607_mech340_persistence_efficacy_gate_validation_{ts}_v3"
    )
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": (
            "v3_exq_607_mech340_persistence_efficacy_gate_validation"
        ),
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-340"],
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-340": "supports" if all_pass else "weakens",
        },
        "evidence_direction_note": (
            "MECH-340 persistence/efficacy gate substrate validation "
            "(diagnostic; NOT governance evidence). Confirms gate-off is "
            "bit-identical, licensed appraisal admits, disengaged appraisal "
            "excludes, recoverability does not drive the gate, and "
            "unattainability=1 with license below floor excludes (stuck-on "
            "failure signature). Agent appraisal wiring deferred."
        ),
        "outcome": "PASS" if all_pass else "FAIL",
        "metrics": metrics,
        "dry_run": bool(dry_run),
    }

    out_path = None
    if not dry_run:
        out_dir = (
            EVIDENCE_ROOT
            / "v3_exq_607_mech340_persistence_efficacy_gate_validation"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Result written to: {out_path}", flush=True)

    return {
        "all_pass": bool(all_pass),
        "outcome": "PASS" if all_pass else "FAIL",
        "manifest_path": str(out_path) if out_path is not None else None,
        "run_id": run_id,
        "dry_run": bool(dry_run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="T1 only; no manifest write.",
    )
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if not result["dry_run"]:
        emit_outcome(
            outcome=result["outcome"],
            manifest_path=result["manifest_path"],
            run_id=result["run_id"],
        )
    sys.exit(0 if result["all_pass"] else 1)
