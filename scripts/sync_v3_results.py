#!/usr/bin/env python3
"""
sync_v3_results.py — Sync V3 experiment results to REE_assembly run pack format.

Reads flat JSON results from ree-v3/evidence/experiments/*/*.json and creates
governance-compatible run packs in:
  REE_assembly/evidence/experiments/<experiment_type>/runs/<run_id>_v3/

Pattern mirrors sync_v2_results.py.

Governance requirements (spec §8):
  - architecture_epoch: "ree_hybrid_guardrails_v1"
  - run_id ends _v3
  - metrics.json includes fatal_error_count: 0.0 for PASS runs

Idempotent: skips run packs whose run_id already exists.

Usage:
    /opt/local/bin/python3 scripts/sync_v3_results.py
    /opt/local/bin/python3 scripts/sync_v3_results.py --dry-run
    /opt/local/bin/python3 scripts/sync_v3_results.py --v3-evidence-dir /path/to/ree-v3/evidence/experiments
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
V3_REPO_ROOT = SCRIPT_DIR.parent
REE_ASSEMBLY_DIR = V3_REPO_ROOT.parent / "REE_assembly"
EVIDENCE_EXPERIMENTS_DIR = REE_ASSEMBLY_DIR / "evidence" / "experiments"
DEFAULT_V3_EVIDENCE_DIR = V3_REPO_ROOT / "evidence" / "experiments"

ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SCHEMA_VERSION = "experiment_pack/v1"
METRICS_SCHEMA_VERSION = "experiment_pack_metrics/v1"
STOP_CRITERIA_VERSION = "stop_criteria/v1"
RUNNER_NAME = "ree-v3-harness"
RUNNER_VERSION = "3.0.0"


def _claim_to_experiment_type(claim_id: str) -> str:
    """SD-005 → claim_probe_sd_005, V3-EXQ-001 → v3_exq_001, etc."""
    if claim_id.upper().startswith("V3-EXQ"):
        # V3-EXQ-001 → v3_exq_001
        return claim_id.lower().replace("-", "_")
    normalized = claim_id.lower().replace("-", "_")
    return f"claim_probe_{normalized}"


def _run_id_from(run_timestamp: str, exp_name: str) -> str:
    """Build run_id ending _v3 from timestamp + experiment name."""
    ts = run_timestamp[:19].replace(":", "").replace("-", "").replace("T", "T")
    return f"{ts}_{exp_name}_v3"


def _normalize_timestamp(ts: str) -> str:
    """Normalise timestamp to RFC3339 UTC."""
    if not ts:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        normalized = ts
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_latest(exp_dir: Path) -> dict | None:
    """Load the most recent JSON result from an experiment directory."""
    json_files = sorted(exp_dir.glob("*.json"))
    if not json_files:
        return None
    latest: dict | None = None
    latest_ts = ""
    for jf in json_files:
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        ts = str(d.get("run_timestamp", ""))
        if ts > latest_ts:
            latest_ts = ts
            latest = d
    return latest


def _write_run_pack(run_dir: Path, data: dict, exp_name: str, dry_run: bool = False) -> str:
    """Write a governance-compatible run pack. Returns run_id."""
    claim_id = data.get("claim", exp_name)
    verdict = str(data.get("verdict", data.get("status", "UNKNOWN"))).upper()
    if verdict not in {"PASS", "FAIL"}:
        verdict = "FAIL"

    run_timestamp = data.get("run_timestamp", "")
    aggregate = data.get("aggregate", {})
    metrics_raw = data.get("metrics", aggregate)
    config = data.get("config", {})

    # Derive experiment_type from claim_id or exp_name
    experiment_type = _claim_to_experiment_type(claim_id) if claim_id != exp_name else exp_name

    run_id = run_dir.name

    # Ensure fatal_error_count present
    metrics_values = {k: v for k, v in metrics_raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if "fatal_error_count" not in metrics_values:
        metrics_values["fatal_error_count"] = 0.0

    evidence_direction = data.get("evidence_direction", "supports" if verdict == "PASS" else "weakens")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_type": experiment_type,
        "run_id": run_id,
        "status": verdict,
        "timestamp_utc": _normalize_timestamp(run_timestamp),
        "source_repo": {
            "name": "ree-v3",
            "commit": "",
            "branch": "main",
        },
        "runner": {"name": RUNNER_NAME, "version": RUNNER_VERSION},
        "artifacts": {
            "metrics_path": "metrics.json",
            "summary_path": "summary.md",
        },
        "stop_criteria_version": STOP_CRITERIA_VERSION,
        "claim_ids_tested": data.get("claim_ids", [claim_id] if claim_id else []),
        "evidence_class": data.get("evidence_class", "simulation"),
        "evidence_direction": evidence_direction,
        "producer_capabilities": {
            "sd005_split_latent": True,
            "sd004_action_objects": True,
            "sd006_multirate_clock": True,
        },
        "environment": {
            "env_id": "ree.causal_grid_world_v3",
            "env_version": "3.0.0",
            "dynamics_hash": "unknown",
            "reward_hash": "unknown",
            "observation_hash": "unknown",
            "config_hash": "unknown",
            "tier": "causal_grid_world_v3",
        },
        "failure_signatures": data.get("failure_signatures", []),
    }

    metrics_doc = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "values": metrics_values,
    }

    summary_md = data.get("summary_markdown", data.get("summary", f"# {verdict}\n"))

    if dry_run:
        print(f"[DRY RUN] Would write: {run_dir}")
        return run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_md.rstrip() + "\n", encoding="utf-8")

    return run_id


def sync_v3_results(
    v3_evidence_dir: Path,
    assembly_evidence_dir: Path,
    dry_run: bool = False,
    dir_filter=None,
) -> int:
    """
    Sync all V3 experiment results to REE_assembly.

    dir_filter: optional callable(Path) -> bool to restrict which subdirs are scanned.
    Returns number of run packs written.
    """
    if not v3_evidence_dir.exists():
        print(f"V3 evidence dir does not exist: {v3_evidence_dir}", file=sys.stderr)
        return 0

    written = 0
    skipped = 0

    for exp_dir in sorted(v3_evidence_dir.iterdir()):
        if dir_filter is not None and not dir_filter(exp_dir):
            continue
        if not exp_dir.is_dir():
            continue

        data = _load_latest(exp_dir)
        if data is None:
            print(f"  SKIP {exp_dir.name} — no JSON found")
            continue

        exp_name = exp_dir.name
        run_timestamp = data.get("run_timestamp", "")
        claim_id = data.get("claim", exp_name)
        experiment_type = _claim_to_experiment_type(claim_id) if claim_id != exp_name else exp_name

        run_id = _run_id_from(run_timestamp, exp_name)

        target_dir = assembly_evidence_dir / experiment_type / "runs" / run_id

        if target_dir.exists():
            print(f"  SKIP {run_id} — already exists")
            skipped += 1
            continue

        _write_run_pack(target_dir, data, exp_name, dry_run=dry_run)
        print(f"  {'[DRY] ' if dry_run else ''}WROTE {run_id} → {experiment_type}")
        written += 1

    print(f"\nSync complete: {written} written, {skipped} skipped")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3-evidence-dir",
        type=Path,
        default=DEFAULT_V3_EVIDENCE_DIR,
        help="Path to ree-v3/evidence/experiments/",
    )
    parser.add_argument(
        "--assembly-evidence-dir",
        type=Path,
        default=EVIDENCE_EXPERIMENTS_DIR,
        help="Path to REE_assembly/evidence/experiments/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing anything",
    )
    args = parser.parse_args()

    print(f"V3 evidence dir:      {args.v3_evidence_dir}")
    print(f"Assembly target dir:  {args.assembly_evidence_dir}")
    if args.dry_run:
        print("DRY RUN — nothing will be written\n")

    # Primary pass: ree-v3/evidence/experiments/ → REE_assembly runs/
    n = sync_v3_results(
        v3_evidence_dir=args.v3_evidence_dir,
        assembly_evidence_dir=args.assembly_evidence_dir,
        dry_run=args.dry_run,
    )

    # Secondary pass: flat JSON dirs already in REE_assembly (v3_exq_* dirs
    # written directly by the runner, not routed via ree-v3/evidence/).
    # Only scan directories starting with "v3_exq_" that contain flat JSONs.
    if args.assembly_evidence_dir.exists():
        flat_dirs = sorted(
            d for d in args.assembly_evidence_dir.iterdir()
            if d.is_dir() and d.name.startswith("v3_exq_")
        )
        if flat_dirs:
            print(f"\nSecondary pass: {len(flat_dirs)} v3_exq_* dirs in assembly")
            n += sync_v3_results(
                v3_evidence_dir=args.assembly_evidence_dir,
                assembly_evidence_dir=args.assembly_evidence_dir,
                dry_run=args.dry_run,
                dir_filter=lambda d: d.name.startswith("v3_exq_"),
            )

    sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()
