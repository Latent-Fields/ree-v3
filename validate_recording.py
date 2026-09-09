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
import math
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
        # Keep in sync with manifest_core.ALWAYS_CORE_KEYS -- pinned by
        # tests/contracts/test_substrate_commit.py so this copy cannot drift
        # silently the way it did when substrate_commit was added.
        ALWAYS_CORE_KEYS = (
            "recording_schema", "substrate_hash", "substrate_commit",
            "machine", "machine_class", "elapsed_seconds", "config", "seeds",
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


# The provenance subset of the always-core that the runs/ pack producer
# (sync_v3_results.build_runpack_docs) is responsible for carrying from the flat
# manifest into the pack. A pack that drops these while its flat sibling carries
# them is the 2026-07-16 thin-pack recording-provenance bug: the index-scored
# runs/ artifact reads machine_class=null / substrate_hash="" even though the
# provenance was recorded (in the flat manifest + coordinator DB). Kept narrow
# (NOT the full always-core) because this tuple names one specific producer
# regression, and widening it would conflate that regression with the ordinary
# always-core gap check_manifest already reports.
#
# CORRECTED 2026-09-09. This comment previously asserted that
# recording_schema/config/seeds/elapsed_seconds "live in the flat manifest +
# metrics.json by design and are absent from the pack schema". That was wrong in
# both halves and cost an autopsy a re-discovery (V3-EXQ-1014 / V3-EXQ-1015):
#   * They are NOT absent from the pack schema. The sanctioned writer
#     (experiments/pack_writer.write_pack, via stamp_recording_core /
#     MANDATORY_CORE_KEYS) stamps all four into the pack MANIFEST.
#   * They did not live in metrics.json either. _manifest_view below merges
#     metrics.json's config/timing/seeds sections into the presence check
#     precisely because pack_writer CAN store them there -- but the flat->pack
#     converter (sync_v3_results.build_runpack_docs), which produced 2915 of the
#     2931 packs in the tree, emitted only `values` and never those sections. So
#     the fields reached NO pack surface at all, and check_manifest's always-core
#     arm was red on 2929 of 2931 packs -- a check that could never pass.
# The converter now carries all four into the pack manifest (REE_assembly
# sync_v3_results.py, 2026-09-09), matching pack_writer. A residual always-core
# gap on a pack is therefore now a real finding again, not this known blind spot.
#
# What the converter still deliberately does NOT project into the pack manifest,
# so it does not get re-litigated: arm_results, readout, criteria,
# control_policies, bears_on, stage2_routing and the other per-run rich blocks.
# build_experiment_indexes reads arm_results from the FLAT manifests by its own
# glob (the arm-fingerprint index visits top-level *.json as well as pack
# manifests), and reads the rest nowhere at all -- so projecting them would
# inflate every pack with bytes no consumer reads. The scalar readouts do need to
# reach the pack, and their channel is metrics.json `values`, which the converter
# harvests from metrics / aggregates / summary_metrics / readout.
_PACK_PROVENANCE_KEYS = ("machine", "machine_class", "substrate_hash")


def _flat_sibling_path(pack_path: Path) -> Optional[Path]:
    """For a pack manifest .../<exp>/runs/<run_id>/manifest.json, return the flat
    sibling evidence/experiments/<run_id>.json, else None (non-pack path)."""
    if pack_path.name != "manifest.json":
        return None
    run_dir = pack_path.parent
    if run_dir.parent.name != "runs":
        return None
    run_id = run_dir.name
    evidence_dir = run_dir.parent.parent.parent  # .../<exp>/runs/<run_id> -> evidence/experiments
    return evidence_dir / f"{run_id}.json"


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def check_pack_provenance(path: Path) -> List[str]:
    """Return the provenance always-core keys the PACK drops but its flat sibling
    carries (the thin-pack recording-provenance regression). Empty for non-pack
    paths, a missing/unreadable flat sibling, or a pack that already carries the
    provenance (the healthy state once the producer fix has propagated)."""
    flat_path = _flat_sibling_path(path)
    if flat_path is None or not flat_path.is_file():
        return []
    pack = _load_json(path)
    flat = _load_json(flat_path)
    if not isinstance(pack, dict) or not isinstance(flat, dict):
        return []
    dropped: List[str] = []
    for k in _PACK_PROVENANCE_KEYS:
        if not _nonempty(pack.get(k)) and _nonempty(flat.get(k)):
            dropped.append(k)
    return dropped


# The four flat spellings the runpack converter harvests metrics.json `values`
# from (REE_assembly evidence/experiments/scripts/sync_v3_results.build_runpack_docs).
#
# THE ORDER IS THE CONVERTER'S OWN AND IS LOAD-BEARING, not cosmetic. The converter
# takes the FIRST non-empty of metrics -> aggregates -> summary_metrics -> readout
# and never looks at the rest. So a manifest carrying BOTH a nested, non-scalar
# `metrics` block AND a good flat `readout` scores from the `metrics` one and lands
# values with no numeric entries -- a checker that searched in any other order would
# find the healthy block first and report the manifest clean, which is precisely the
# false negative this check exists to prevent. Mirror the consumer, do not re-rank it.
#
# `readout` is nonetheless the spelling to PREFER in a new driver (it is the one name
# with no other meaning in the corpus); the other three are historical and are read
# identically once chosen.
_READOUT_SPELLINGS = ("metrics", "aggregates", "summary_metrics", "readout")


def _numeric_entry_count(block: Any) -> int:
    """Count the entries of `block` that build_experiment_indexes would actually
    read: `_is_number` (l.315) is `isinstance(v, (int, float)) and not
    isinstance(v, bool)`. Applied here with the extra non-finite exclusion the
    standard's second encoding rule requires -- a nan IS numeric to the indexer,
    which is exactly why emitting one is a defect rather than a neutral filler."""
    if not isinstance(block, dict):
        return 0
    n = 0
    for value in block.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            n += 1
    return n


def _readout_encoding_notes(block: Any) -> List[str]:
    """Entries that are PRESENT but inert or harmful under the standard's two
    encoding rules (3b "Machine-readable verdict readout"). Never a gap on its
    own -- reported alongside whichever verdict check_flat_scalar_readout gives."""
    notes: List[str] = []
    if not isinstance(block, dict):
        return notes
    bools = [k for k, v in block.items() if isinstance(v, bool)]
    nonfinite = [k for k, v in block.items()
                 if isinstance(v, float) and not math.isfinite(v)]
    nulls = [k for k, v in block.items() if v is None]
    if bools:
        notes.append(f"raw bool (emit as 0/1 int -- _is_number excludes bool): "
                     f"{', '.join(sorted(bools)[:6])}")
    if nonfinite:
        notes.append(f"non-finite (DROP the key -- nan is numeric to the indexer "
                     f"and pollutes deltas): {', '.join(sorted(nonfinite)[:6])}")
    if nulls:
        notes.append(f"null (DROP the key -- absent correctly reads as unmeasured): "
                     f"{', '.join(sorted(nulls)[:6])}")
    return notes


def check_flat_scalar_readout(path: Path) -> Optional[Tuple[str, str, List[str]]]:
    """Return (verdict, detail, encoding_notes) when the manifest lacks a usable
    flat scalar readout, else None.

    Standard 3b "Machine-readable verdict readout" (added 2026-09-09). The
    converter harvests metrics.json `values` from ONE of four FLAT spellings, and
    build_experiment_indexes reads only the NUMERIC entries of that block. A
    readout recorded only as a dict keyed by arm or by seed matches none of the
    four, so the pack scores with values=={} -- and then no `fail_if` stop
    threshold can fire (final_status silently falls back to the manifest's own
    self-declared status, which is what claim_evidence.v1.json records), the
    duplicate-emission supersession fingerprint is skipped entirely (both copies
    of a byte-identical re-emission score), and the index carries no deltas and
    no key-metrics columns. Measurement: REE_assembly evidence/planning/
    flat_scalar_readout_recording_gap_20260909.md.

    Verdicts:
      "absent"       -- none of the four spellings carries a non-empty dict.
      "no_numeric"   -- a spelling IS present but no entry survives _is_number.
                        This is usually the bool-only shape (82 flat manifests
                        measured 2026-09-09): recorded, and invisible.
      "encoding_only" -- the block DOES score, so this is not a gap; it just
                        also carries entries that are inert (raw bools) or
                        harmful (nan/inf, which _is_number happily accepts and
                        which then pollute a delta). Reported so the encoding
                        rules are actionable on a manifest that otherwise looks
                        healthy -- V3-EXQ-484 (4 numerics + 4 bool criterion
                        verdicts) and V3-EXQ-165 (16 numerics + 2 nan) are both
                        this shape, and both hide the defect behind a passing
                        presence check.

    ADVISORY IN EVERY MODE, INCLUDING --strict -- deliberately, and not an
    oversight to be tidied up later. /queue-experiment Step 3.5 runs this linter
    with --strict on the smoke-test manifest, and 802 of 1008 flat manifests
    (703 of 1358 manifest-writing drivers) lack the field today; blocking there
    would gate every new driver on a corpus-wide gap, and CLAUDE.md's standing
    guidance is that a gate firing on ordinary work gets disabled -- which is
    worse than no gate. It is likewise NOT added to manifest_core.
    ALWAYS_CORE_KEYS: that constant feeds missing_core_fields, which IS the
    --strict-blocking arm, and it is also the list stamp_recording_core is
    responsible for -- but this is the one always-core field the stamper
    structurally CANNOT compute, since only the driver knows which scalars its
    verdict turns on.

    Exempt (they record no verdict, so they have no scalars to pre-register --
    this exemption came out of the GOV-HELDOUT-1 check on the rule, which found
    the first draft WARNed on both):
      - a crash report: outcome == "ERROR", or a `*_runner_error_*` run_id;
      - a dry-run manifest: dry_run truthy.
    Non-manifest JSON (no `run_id`) is not gated at all.
    """
    doc = _load_json(path)
    if not isinstance(doc, dict):
        return None
    if not doc.get("run_id"):
        return None  # not a result manifest -- index/tracker/config JSON
    if doc.get("dry_run"):
        return None
    if doc.get("outcome") == "ERROR" or "_runner_error_" in str(doc.get("run_id")):
        return None

    # For a PACK manifest the readout does not live on the manifest at all -- the
    # converter has already projected it into the sibling metrics.json `values`,
    # which IS the surface build_experiment_indexes scores. Check that surface
    # directly rather than the pack manifest, or every pack reads as a false gap.
    metrics_path = path.parent / "metrics.json"
    if path.name == "manifest.json" and metrics_path.is_file():
        metrics = _load_json(metrics_path)
        values = metrics.get("values") if isinstance(metrics, dict) else None
        notes = _readout_encoding_notes(values)
        if _numeric_entry_count(values) > 0:
            return None if not notes else ("encoding_only", "metrics.json values", notes)
        if isinstance(values, dict) and values:
            return ("no_numeric", "metrics.json values", notes)
        return ("absent", "metrics.json values", notes)

    # Resolve exactly as the converter does: FIRST non-empty spelling wins, and
    # the others are never consulted -- see _READOUT_SPELLINGS on why searching
    # for the healthiest block instead would hide the very defect being checked.
    for spelling in _READOUT_SPELLINGS:
        block = doc.get(spelling)
        if not isinstance(block, dict) or not block:
            continue
        notes = _readout_encoding_notes(block)
        if _numeric_entry_count(block) > 0:
            return None if not notes else ("encoding_only", spelling, notes)
        return ("no_numeric", spelling, notes)
    return ("absent", "|".join(_READOUT_SPELLINGS), [])


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

    # substrate_commit: an EXPLAINED absence is not a gap. A manifest carrying
    # `substrate_commit_unavailable` has recorded WHY the commit could not be
    # taken (git-less staged tree, unborn HEAD -- see
    # manifest_core.substrate_commit_unavailable_reason), which is the honest
    # answer for that environment and strictly more information than the field's
    # presence would carry. WARNing on it would train readers to ignore the
    # always-core WARN, which is the one signal that has to stay meaningful.
    # A BARE omission -- neither field -- still WARNs, and is additionally a hard
    # error at write time (missing_mandatory_core_fields).
    if "substrate_commit" in missing and view.get("substrate_commit_unavailable"):
        missing = [k for k in missing if k != "substrate_commit"]

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
    thin_packs: List[Tuple[Path, List[str]]] = []
    readout_gaps: List[Tuple[Path, str, str, List[str]]] = []
    for p in paths:
        missing, schema_warnings = check_manifest(p)
        thin = check_pack_provenance(p)
        readout = check_flat_scalar_readout(p)
        if readout is not None:
            readout_gaps.append((p, readout[0], readout[1], readout[2]))
        rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
        if schema_warnings:
            warns.append((p, schema_warnings))
        if thin:
            thin_packs.append((p, thin))
        if missing:
            gaps.append((p, missing))
        else:
            n_ok += 1
            if not args.quiet:
                print(f"[validate_recording] OK   {rel}", flush=True)

    print("", flush=True)
    print(f"[validate_recording] checked {len(paths)} manifest(s): "
          f"{n_ok} complete, {len(gaps)} with always-core gaps, "
          f"{len(thin_packs)} thin-pack provenance drop(s), "
          f"{len(readout_gaps)} flat-scalar-readout finding(s), "
          f"{len(warns)} schema-warning(s)", flush=True)

    if warns:
        print("", flush=True)
        print("[validate_recording] Schema WARNINGS (advisory, forward-compatible):", flush=True)
        for p, ws in warns:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
            for w in ws:
                print(f"  - {rel}: {w}", flush=True)

    if thin_packs:
        print("", flush=True)
        label = "(strict: blocking)" if args.strict else "(advisory; --strict to block)"
        print(f"[validate_recording] THIN-PACK provenance drops {label} -- "
              f"the runs/ pack lost always-core the flat manifest carries "
              f"(sync_v3_results.build_runpack_docs must carry it; re-materialise "
              f"the pack to heal):", flush=True)
        for p, dropped in thin_packs:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
            print(f"  - {rel}: pack drops {', '.join(dropped)} (flat has them)", flush=True)

    if gaps:
        print("", flush=True)
        label = "GAPS (strict: blocking)" if args.strict else "GAPS (advisory; --strict to block)"
        print(f"[validate_recording] Always-core {label}:", flush=True)
        for p, missing in gaps:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
            print(f"  - {rel}: missing {', '.join(missing)}", flush=True)

    if readout_gaps:
        print("", flush=True)
        # ADVISORY IN EVERY MODE -- see check_flat_scalar_readout's docstring for
        # why this one does NOT harden under --strict while the two sections
        # above do.
        print("[validate_recording] FLAT-SCALAR-READOUT findings "
              "(advisory in ALL modes, including --strict) -- standard 3b "
              "\"Machine-readable verdict readout\": with no NUMERIC "
              "metrics.values, no `fail_if` stop threshold can fire (final_status "
              "falls back to the manifest's self-declared status), the "
              "duplicate-emission supersession fingerprint is skipped, and the "
              "index carries no deltas:", flush=True)
        for p, verdict, where, notes in readout_gaps:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
            if verdict == "absent":
                print(f"  - {rel}: NO flat scalar readout under any of "
                      f"{where} -- add one (see "
                      f"experiments/v3_exq_1015_mech465_zworld_warmup_budget_"
                      f"dispersion_sweep.py for the reference block)", flush=True)
            elif verdict == "no_numeric":
                print(f"  - {rel}: `{where}` present but NO entry survives "
                      f"_is_number -- recorded and invisible", flush=True)
            else:
                print(f"  - {rel}: `{where}` scores, but carries inert/harmful "
                      f"entries", flush=True)
            for n in notes:
                print(f"      * {n}", flush=True)

    if args.strict and (gaps or thin_packs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
