"""
Experiment Pack v1 writer — V3.

Identical to V2 pack_writer except:
- run_id must end _v3 (validated at write time)
- architecture_epoch "ree_hybrid_guardrails_v1" added to manifest

<output_root>/claim_probe_{claim_id}/runs/{run_id}_v3/
  manifest.json
  metrics.json
  summary.md

Governance requirements (spec §8):
  manifest.json must contain:
    architecture_epoch: "ree_hybrid_guardrails_v1"
    run_id: ends "_v3"
    status: "PASS" | "FAIL"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Optional, Union


MANIFEST_SCHEMA_VERSION = "experiment_pack/v1"
METRICS_SCHEMA_VERSION = "experiment_pack_metrics/v1"
STOP_CRITERIA_VERSION = "stop_criteria/v1"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
OUTPUT_ROOT_ENV = "REE_EXPERIMENT_OUTPUT_ROOT"
EVIDENCE_DIRECTIONS = {"supports", "weakens", "mixed", "unknown"}
REQUIRED_ENVIRONMENT_FIELDS = (
    "env_id", "env_version", "dynamics_hash",
    "reward_hash", "observation_hash", "config_hash", "tier",
)
DEFAULT_PRODUCER_CAPABILITIES = {
    "trajectory_integrity_channelized_bias": True,
    "mech056_dispatch_metric_set": True,
    "mech056_summary_escalation_trace": True,
    "sd005_split_latent": True,
    "sd004_action_objects": True,
    "sd006_multirate_clock": True,
}
DEFAULT_ENVIRONMENT = {
    "env_id": "ree.causal_grid_world_v3",
    "env_version": "3.0.0",
    "dynamics_hash": "unknown",
    "reward_hash": "unknown",
    "observation_hash": "unknown",
    "config_hash": "unknown",
    "tier": "causal_grid_world_v3",
}

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_EXPERIMENT_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class EmittedPack:
    run_dir: Path
    manifest_path: Path
    metrics_path: Path
    summary_path: Path


def normalize_timestamp_utc(timestamp_utc: Optional[str] = None) -> str:
    if timestamp_utc is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    normalized = timestamp_utc
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def deterministic_run_id(experiment_type: str, seed: int, timestamp_utc: str) -> str:
    """Build deterministic run id. Automatically appends _v3."""
    normalized = normalize_timestamp_utc(timestamp_utc)
    dt = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    ts = dt.strftime("%Y%m%dT%H%M%S")
    exp_slug = _EXPERIMENT_SLUG_RE.sub("_", experiment_type.lower()).strip("_")
    run_id = f"{ts}_{exp_slug}_seed{seed}_v3"
    return run_id


def stable_config_hash(config_payload: Mapping[str, Any]) -> str:
    payload = json.dumps(config_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def resolve_output_root(cli_output_root: Optional[str]) -> Path:
    if cli_output_root:
        return Path(cli_output_root)
    from_env = os.getenv(OUTPUT_ROOT_ENV)
    if from_env:
        return Path(from_env)
    return Path("runs")


def discover_source_repo(repo_root: Path) -> dict:
    source_repo: dict = {"name": repo_root.name}
    commit = _git_value(["rev-parse", "HEAD"], repo_root)
    source_repo["commit"] = commit or "unknown"
    branch = _git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if branch and branch != "HEAD":
        source_repo["branch"] = branch
    return source_repo


def _git_value(args: list, cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(cwd), check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class ExperimentPackWriter:
    """Reusable writer for Experiment Pack v1 artifacts (V3)."""

    def __init__(
        self,
        output_root: Path,
        repo_root: Path,
        runner_name: str,
        runner_version: str,
    ):
        self.output_root = output_root
        self.repo_root = repo_root
        self.runner_name = runner_name
        self.runner_version = runner_version

    def write_pack(
        self,
        experiment_type: str,
        run_id: str,
        timestamp_utc: str,
        status: str,
        metrics_values: Mapping[str, Any],
        summary_markdown: str,
        scenario: Optional[Mapping[str, Any]] = None,
        failure_signatures: Optional[list] = None,
        claim_ids_tested: Optional[list] = None,
        evidence_class: Optional[str] = None,
        evidence_direction: Optional[str] = None,
        producer_capabilities: Optional[Mapping[str, bool]] = None,
        environment: Optional[Mapping[str, Any]] = None,
        traces_dir: Optional[str] = None,
        media_dir: Optional[str] = None,
    ) -> EmittedPack:
        status = status.upper()
        if status not in {"PASS", "FAIL"}:
            raise ValueError(f"invalid status '{status}'")

        # V3 governance: run_id must end _v3
        if not run_id.endswith("_v3"):
            raise ValueError(
                f"V3 governance violation: run_id must end '_v3', got '{run_id}'"
            )

        normalized_ts = normalize_timestamp_utc(timestamp_utc)
        clean_metrics = _clean_numeric_metrics(metrics_values)
        clean_claim_ids = _clean_claim_ids(claim_ids_tested or [])
        clean_evidence_class = _clean_evidence_class(evidence_class)
        clean_evidence_direction = _clean_evidence_direction(evidence_direction)
        clean_producer_capabilities = _clean_producer_capabilities(producer_capabilities)
        clean_environment = _clean_environment(environment)

        run_dir = self.output_root / experiment_type / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        metrics_path = run_dir / "metrics.json"
        summary_path = run_dir / "summary.md"
        manifest_path = run_dir / "manifest.json"

        metrics_doc = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "values": clean_metrics,
        }
        metrics_path.write_text(json.dumps(metrics_doc, indent=2) + "\n", encoding="utf-8")
        summary_path.write_text(summary_markdown.rstrip() + "\n", encoding="utf-8")

        artifacts: dict = {"metrics_path": "metrics.json", "summary_path": "summary.md"}
        if traces_dir:
            (run_dir / traces_dir).mkdir(parents=True, exist_ok=True)
            artifacts["traces_dir"] = traces_dir
        if media_dir:
            (run_dir / media_dir).mkdir(parents=True, exist_ok=True)
            artifacts["media_dir"] = media_dir

        manifest_doc: dict = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "architecture_epoch": ARCHITECTURE_EPOCH,  # V3 governance field
            "experiment_type": experiment_type,
            "run_id": run_id,
            "status": status,
            "timestamp_utc": normalized_ts,
            "source_repo": discover_source_repo(self.repo_root),
            "runner": {"name": self.runner_name, "version": self.runner_version},
            "artifacts": artifacts,
            "stop_criteria_version": STOP_CRITERIA_VERSION,
            "claim_ids_tested": clean_claim_ids,
            "evidence_class": clean_evidence_class,
            "evidence_direction": clean_evidence_direction,
            "producer_capabilities": clean_producer_capabilities,
            "environment": clean_environment,
            "failure_signatures": _dedupe_strings(failure_signatures or []),
        }
        if scenario:
            manifest_doc["scenario"] = dict(scenario)

        manifest_path.write_text(json.dumps(manifest_doc, indent=2) + "\n", encoding="utf-8")

        return EmittedPack(
            run_dir=run_dir,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
        )


# --- Internal helpers ---

def _clean_numeric_metrics(metrics_values: Mapping[str, Any]) -> dict:
    clean: dict = {}
    for key, value in metrics_values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metrics keys must be non-empty strings")
        if not _SNAKE_CASE_RE.match(key):
            raise ValueError(f"metric key must be snake_case: {key}")
        clean[key] = _coerce_numeric(value)
    return clean


def _clean_claim_ids(claim_ids: list) -> list:
    return _dedupe_strings([str(c).strip() for c in claim_ids if str(c).strip()])


def _clean_evidence_class(evidence_class: Optional[str]) -> str:
    if not evidence_class:
        return "simulation"
    return evidence_class.strip() or "simulation"


def _clean_evidence_direction(evidence_direction: Optional[str]) -> str:
    if evidence_direction is None:
        return "unknown"
    cleaned = evidence_direction.strip().lower()
    if cleaned not in EVIDENCE_DIRECTIONS:
        raise ValueError(f"invalid evidence_direction '{evidence_direction}'")
    return cleaned


def _clean_producer_capabilities(
    producer_capabilities: Optional[Mapping[str, bool]],
) -> dict:
    clean = dict(DEFAULT_PRODUCER_CAPABILITIES)
    if producer_capabilities:
        for key, value in producer_capabilities.items():
            if not isinstance(value, bool):
                raise TypeError(f"producer capability '{key}' must be a boolean")
            clean[key] = value
    return clean


def _clean_environment(environment: Optional[Mapping[str, Any]]) -> dict:
    clean = dict(DEFAULT_ENVIRONMENT)
    if environment:
        for key, value in environment.items():
            clean[key] = str(value).strip() if value is not None else "unknown"
    for key in REQUIRED_ENVIRONMENT_FIELDS:
        clean[key] = clean.get(key, "").strip() or "unknown"
    return clean


def _coerce_numeric(value: Any) -> Union[float, int]:
    if isinstance(value, bool):
        raise TypeError("metrics values must be numeric; booleans are not allowed")
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "item"):
        item_value = value.item()
        if isinstance(item_value, bool):
            raise TypeError("metrics values must be numeric; booleans are not allowed")
        if isinstance(item_value, (int, float)):
            return item_value
    raise TypeError(f"metrics values must be int/float, got {type(value)!r}")


def _dedupe_strings(values: list) -> list:
    seen: set = set()
    result = []
    for v in values:
        if v not in seen:
            result.append(v)
            seen.add(v)
    return result
