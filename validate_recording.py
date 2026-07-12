#!/usr/bin/env python3
"""
validate_recording.py -- soft-validate linter for the Experimental Recording Standard.

Standard: REE_assembly/evidence/planning/experimental_recording_standard_2026-07-12.md
(section 3b "ALWAYS-record core"; section 4 "Deferred hardening" names this linter).

Mirrors validate_experiments.py in spirit: it WARNs on a manifest that is missing the
always-record core, and exits non-zero ONLY under --strict. It is additive and
forward-compatible per standard 3d -- an UNKNOWN `recording_schema` (newer than the one
this linter knows) WARNs and is interpreted best-effort; it never hard-fails on a field
it does not recognise.

What it checks (per manifest JSON)
----------------------------------
The always-core keys from experiments/_lib/manifest_core.ALWAYS_CORE_KEYS:
    recording_schema, substrate_hash, machine, machine_class,
    elapsed_seconds, config, seeds
For a PACK manifest (a runs/<run_id>/manifest.json with a sibling metrics.json), the
sibling metrics.json's top-level sections (values / per_seed / latent / config / timing)
are merged into the presence check, since pack_writer stores config/timing there.

Usage
-----
    /opt/local/bin/python3 validate_recording.py --paths a.json b.json
    /opt/local/bin/python3 validate_recording.py --dir <evidence/experiments>
    /opt/local/bin/python3 validate_recording.py --dir <dir> --strict   # exit 1 on any gap

Default mode is REPORT (exit 0). Point --dir at REE_assembly/evidence/experiments to
sweep the corpus, or --paths at a manifest a smoke test just produced. With neither
--paths nor --dir, prints usage and exits 0.

This file is ASCII-safe (cp1252 / Windows terminal compatible). Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent

# The canonical always-core key list + the empty-check live in manifest_core so this
# linter and the stamper cannot drift. Import robustly; fall back to a local copy only
# if the package layout prevents import (keeps the linter runnable standalone).
try:
    from experiments._lib.manifest_core import (  # type: ignore
        ALWAYS_CORE_KEYS,
        RECORDING_SCHEMA,
        missing_core_fields,
    )
except Exception:  # pragma: no cover - standalone fallback
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    try:
        from _lib.manifest_core import (  # type: ignore
            ALWAYS_CORE_KEYS,
            RECORDING_SCHEMA,
            missing_core_fields,
        )
    except Exception:
        ALWAYS_CORE_KEYS = (
            "recording_schema", "substrate_hash", "machine", "machine_class",
            "elapsed_seconds", "config", "seeds",
        )
        RECORDING_SCHEMA = "rec/v1"

        def _is_empty(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, (str, bytes, list, tuple, dict, set)) and len(value) == 0:
                return True
            return False

        def missing_core_fields(manifest):  # type: ignore
            return [k for k in ALWAYS_CORE_KEYS if _is_empty(manifest.get(k, None))]


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _merged_view(manifest: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """A combined presence view: the manifest, plus a sibling metrics.json's
    top-level sections for a pack (pack_writer stores config/timing/seeds-ish there).

    The sibling is only merged for keys the manifest itself lacks -- the manifest is
    authoritative; the metrics doc only fills provenance the pack shape keeps in
    metrics.json. Absent sibling => the manifest is used unchanged.
    """
    view = dict(manifest)
    metrics_path = path.parent / "metrics.json"
    if metrics_path.is_file():
        metrics = _load_json(metrics_path)
        if isinstance(metrics, dict):
            for k in ("config", "timing", "seeds", "per_seed", "latent"):
                if k in metrics and k not in view:
                    view[k] = metrics[k]
            # timing.elapsed_seconds satisfies elapsed_seconds if absent up top.
            timing = metrics.get("timing")
            if (isinstance(timing, dict) and "elapsed_seconds" in timing
                    and "elapsed_seconds" not in view):
                view["elapsed_seconds"] = timing["elapsed_seconds"]
    return view


def check_manifest(path: Path) -> Tuple[List[str], List[str]]:
    """Return (missing_fields, schema_warnings) for one manifest JSON.

    missing_fields: always-core keys absent/empty.
    schema_warnings: non-blocking notes (e.g. unknown recording_schema).
    A file that is not a JSON object yields a single sentinel missing entry.
    """
    doc = _load_json(path)
    if not isinstance(doc, dict):
        return ["<not-a-json-object>"], []

    view = _merged_view(doc, path)
    missing = missing_core_fields(view)

    schema_warnings: List[str] = []
    rec = view.get("recording_schema")
    if isinstance(rec, str) and rec and rec != RECORDING_SCHEMA:
        # Forward-compatible: a newer/unknown schema is interpreted best-effort,
        # never a hard failure (standard 3d).
        schema_warnings.append(
            f"recording_schema '{rec}' != known '{RECORDING_SCHEMA}' "
            f"-- interpreting best-effort (forward-compatible)")
    return missing, schema_warnings


def _candidate_paths(paths: Sequence[str], dir_arg: Optional[str]) -> List[Path]:
    out: List[Path] = []
    if paths:
        out.extend(Path(p).resolve() for p in paths)
    if dir_arg:
        base = Path(dir_arg).resolve()
        # flat manifests + pack manifests under the dir
        out.extend(sorted(base.glob("*.json")))
        out.extend(sorted(base.glob("**/runs/**/manifest.json")))
    # de-dup, preserve order
    seen: set = set()
    uniq: List[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Soft-validate manifests against the Experimental Recording Standard always-core.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any manifest is missing an always-core field. Default is report-only.")
    parser.add_argument("--paths", nargs="*", default=[],
                        help="Specific manifest JSON files to check.")
    parser.add_argument("--dir", default=None,
                        help="Directory to sweep (flat *.json + **/runs/**/manifest.json).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-manifest OK lines.")
    args = parser.parse_args()

    paths = _candidate_paths(args.paths, args.dir)
    if not paths:
        print("[validate_recording] no manifests specified; pass --paths or --dir", flush=True)
        print(f"[validate_recording] always-core: {', '.join(ALWAYS_CORE_KEYS)}", flush=True)
        return 0

    n_ok = 0
    gaps: List[Tuple[Path, List[str]]] = []
    warns: List[Tuple[Path, List[str]]] = []
    for p in paths:
        missing, schema_warnings = check_manifest(p)
        rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
        if schema_warnings:
            warns.append((p, schema_warnings))
        if missing:
            gaps.append((p, missing))
        else:
            n_ok += 1
            if not args.quiet:
                print(f"[validate_recording] OK   {rel}", flush=True)

    print("", flush=True)
    print(f"[validate_recording] checked {len(paths)} manifest(s): "
          f"{n_ok} complete, {len(gaps)} with always-core gaps, "
          f"{len(warns)} schema-warning(s)", flush=True)

    if warns:
        print("", flush=True)
        print("[validate_recording] Schema WARNINGS (advisory, forward-compatible):", flush=True)
        for p, ws in warns:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
            for w in ws:
                print(f"  - {rel}: {w}", flush=True)

    if gaps:
        print("", flush=True)
        label = "GAPS (strict: blocking)" if args.strict else "GAPS (advisory; --strict to block)"
        print(f"[validate_recording] Always-core {label}:", flush=True)
        for p, missing in gaps:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
            print(f"  - {rel}: missing {', '.join(missing)}", flush=True)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
