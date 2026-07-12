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
from typing import Any, Callable, Mapping, Optional, Union


MANIFEST_SCHEMA_VERSION = "experiment_pack/v1"
METRICS_SCHEMA_VERSION = "experiment_pack_metrics/v1"
STOP_CRITERIA_VERSION = "stop_criteria/v1"

# Structured metrics sections (Experimental Recording Standard 3b/3c). The metrics
# doc historically carried ONLY `values` (scalar keys), and _clean_numeric_metrics
# coerced every entry to a scalar -- which structurally prevented the sanctioned
# writer from storing per-seed arrays, latent stats, config snapshots, or timing
# (standard section 4 "Deferred hardening"; the mechanical reason packs are flat and
# rich readouts survive only in hand-rolled manifests). These reserved sibling
# sections carry that rich payload UN-coerced alongside the scalar `values` block, so
# the indexer's scalar-only read of `values` is unaffected while nothing structured is
# dropped. Additive/forward-compatible: an older reader ignores unknown keys.
METRICS_STRUCTURED_SECTIONS = ("per_seed", "latent", "config", "timing")
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

# Reserved plumbing filenames that live alongside flat manifests under
# evidence/experiments/ -- a run_id must never collide with one, or downstream
# scanners (sync_v3_results / serve / scan_flat_vs_runs) would exclude it as
# non-run plumbing (their SKIP lists). Guarded by write_flat_manifest().
FLAT_MANIFEST_SKIP_NAMES = frozenset({
    "claim_evidence.v1.json",
    "claim_evidence_matrix.v1.json",
    "review_tracker.json",
    "runner_status.json",
    "substrate_status_snapshot.json",
    "pending_review.json",
})


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
        per_seed: Optional[Any] = None,
        latent: Optional[Any] = None,
        config: Optional[Any] = None,
        timing: Optional[Any] = None,
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
        # Structured recording-standard sections (per_seed / latent / config /
        # timing) carry rich readouts UN-coerced beside the scalar `values` block.
        # Only sections actually supplied are written (additive; a None section is
        # a no-op, so an existing single-arm scalar-only caller is bit-identical).
        structured = _clean_structured_sections(
            per_seed=per_seed, latent=latent, config=config, timing=timing,
        )
        metrics_doc.update(structured)
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


# --- Single flat-manifest writer (Experimental Recording Standard sec 4 chokepoint) ---

def _resolve_flat_status(manifest: Mapping[str, Any]) -> Optional[str]:
    """The status value the flat->pack converter (sync_v3_results.build_runpack_docs)
    will read: the first non-empty of status | overall_outcome | outcome. Returns
    None if none is present (a manifest that would sync to status 'UNKNOWN')."""
    for key in ("status", "overall_outcome", "outcome"):
        val = manifest.get(key)
        if val not in (None, ""):
            return str(val)
    return None


def write_flat_manifest(
    manifest: dict,
    out_dir: Union[str, Path],
    *,
    dry_run: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    seeds: Any = None,
    script_path: Optional[Union[str, Path]] = None,
    machine: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    started_at: Optional[float] = None,
    stamp: bool = True,
    overwrite_core: bool = False,
    require_v3: bool = True,
    json_default: Optional[Callable[[Any], Any]] = None,
) -> Path:
    """The single sanctioned writer for a FLAT V3 experiment manifest.

    This is the author-side chokepoint the Experimental Recording Standard
    (experimental_recording_standard_2026-07-12.md sec 4) names: instead of a
    hand-rolled ``json.dump(manifest, f)`` tail, an experiment calls this once and
    gets (a) the always-record core stamped via
    ``experiments/_lib/manifest_core.stamp_recording_core`` and (b) the identity
    invariants the whole downstream chain depends on, enforced at emission.

    It writes the flat manifest to ``<out_dir>/<run_id>.json`` (or
    ``_dry_<run_id>.json`` when ``dry_run``), which is the exact path/keying the
    coordinator commits verbatim and every flat consumer reads
    (build_experiment_indexes' governance overlay keys on ``<run_id>.json``;
    serve.py's explorer detail; sync_v3_results' flat->pack projection). Field
    NAMES are preserved verbatim -- this writer does NOT reshape ``claim_ids`` ->
    ``claim_ids_tested`` or collapse ``outcome`` -> ``status``, and it does NOT
    strip unknown/rich fields (``arm_results`` / ``interpretation`` / ``per_seed``
    must survive for the explorer catch-all and the adjudication overlay). It is a
    stamp-validate-write wrapper, not a schema projector; the pack projection stays
    the job of sync_v3_results.build_runpack_docs.

    Invariants enforced (the sync ``_is_flat_v3`` + coordinator ``POST /result`` +
    scoring hard constraints):
      * ``run_id`` present, a non-empty string, ending ``_v3`` (or carrying the
        mid-string ``_v3_<ts>`` evidence-grade form) unless ``require_v3=False``;
      * ``architecture_epoch`` present -- defaulted to ``ree_hybrid_guardrails_v1``
        if the caller omitted it (both gate ``_is_flat_v3``);
      * a resolvable status (one of ``status`` | ``overall_outcome`` | ``outcome``);
      * the filename does not collide with a reserved plumbing name.

    Parameters mirror stamp_recording_core for the always-core (config / seeds /
    script_path / machine / elapsed_seconds / started_at). ``stamp=False`` skips the
    always-core merge (for a manifest already stamped upstream). ``overwrite_core``
    forces the stamper to overwrite already-present core fields. Returns the written
    Path (hand it to ``experiment_protocol.emit_outcome(manifest_path=...)``).

    ASCII-only output (repo rule); stdlib + a lazy manifest_core import so a
    scalar-only caller needs no torch/ree_core.
    """
    if not isinstance(manifest, dict):
        raise TypeError(f"manifest must be a dict, got {type(manifest)!r}")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("flat manifest requires a non-empty string 'run_id'")
    run_id = run_id.strip()
    if require_v3 and not (run_id.endswith("_v3") or "_v3_" in run_id):
        raise ValueError(
            "V3 governance: run_id must end '_v3' (or carry a mid-string "
            f"'_v3_<ts>'), got '{run_id}' -- else sync_v3_results._is_flat_v3 "
            "silently never scores it"
        )

    # architecture_epoch is the other _is_flat_v3 gate; default-fill so a caller
    # may omit it, but never clobber a deliberate value.
    if not manifest.get("architecture_epoch"):
        manifest["architecture_epoch"] = ARCHITECTURE_EPOCH

    if _resolve_flat_status(manifest) is None:
        raise ValueError(
            "flat manifest requires a status/outcome -- set one of "
            "'status' | 'overall_outcome' | 'outcome' (else the pack syncs to "
            "status 'UNKNOWN')"
        )

    # Always-record core (standard 3b). Stamped AFTER the manifest (incl. any
    # arm_results) is assembled, so a multi-arm run hoists substrate_hash from the
    # per-cell fingerprints rather than recomputing a driver-inclusive hash that
    # would not match. Never let stamping crash a run (it is a soft-validate WARN).
    if stamp:
        stamp_fn = _import_stamp_recording_core()
        if stamp_fn is not None:
            try:
                stamp_fn(
                    manifest,
                    config=config,
                    seeds=seeds,
                    script_path=script_path,
                    machine=machine,
                    elapsed_seconds=elapsed_seconds,
                    started_at=started_at,
                    overwrite=overwrite_core,
                )
            except Exception:
                pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{run_id}.json"
    if fname in FLAT_MANIFEST_SKIP_NAMES:
        raise ValueError(
            f"run_id '{run_id}' collides with reserved plumbing filename '{fname}'"
        )
    if dry_run:
        fname = f"_dry_{fname}"
    out_path = out_dir / fname
    # json_default mirrors json.dump's ``default=`` (e.g. ``str``) for the small
    # early-era class whose manifest carries non-JSON-native values; None (the
    # default) is byte-identical to a plain json.dumps for every other caller.
    out_path.write_text(
        json.dumps(manifest, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    return out_path


def _import_stamp_recording_core():
    """Lazily import stamp_recording_core across the several ways experiment
    scripts put experiments/ on sys.path. Returns the callable or None."""
    try:
        from experiments._lib.manifest_core import stamp_recording_core  # type: ignore
        return stamp_recording_core
    except Exception:
        pass
    try:
        from _lib.manifest_core import stamp_recording_core  # type: ignore
        return stamp_recording_core
    except Exception:
        pass
    try:
        from experiments._lib import manifest_core  # type: ignore
        return manifest_core.stamp_recording_core
    except Exception:
        return None


# --- Internal helpers ---

def _clean_structured_sections(**sections: Any) -> dict:
    """Validate + collect the optional structured metrics sections.

    Each section (per_seed / latent / config / timing) is stored VERBATIM -- no
    scalar coercion (that is the whole point: the scalar-coercion of `values` is
    what previously made packs unable to carry rich readouts). A section is only
    included when it is not None. Validation is minimal and structural:
      * the section must be a JSON-serialisable dict or list (so the manifest stays
        diffable and the file round-trips);
      * a section name must be one of METRICS_STRUCTURED_SECTIONS (guards typos);
      * dict sections must have string keys.
    Large arrays should still be stored by reference (standard 3d), but that is the
    author's call -- this writer does not second-guess the payload's size.
    """
    clean: dict = {}
    for name, value in sections.items():
        if value is None:
            continue
        if name not in METRICS_STRUCTURED_SECTIONS:
            raise ValueError(
                f"unknown structured section '{name}'; allowed: "
                f"{', '.join(METRICS_STRUCTURED_SECTIONS)}"
            )
        if not isinstance(value, (dict, list)):
            raise TypeError(
                f"structured section '{name}' must be a dict or list, "
                f"got {type(value)!r}"
            )
        if isinstance(value, dict):
            for k in value:
                if not isinstance(k, str):
                    raise ValueError(
                        f"structured section '{name}' keys must be strings, "
                        f"got {type(k)!r}"
                    )
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"structured section '{name}' is not JSON-serialisable: {exc}"
            ) from exc
        clean[name] = value
    return clean


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
