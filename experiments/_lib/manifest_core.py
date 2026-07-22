"""
Always-record core stamper for the Experimental Recording Standard (V3+).

Standard: REE_assembly/evidence/planning/experimental_recording_standard_2026-07-12.md
(section 3b "ALWAYS-record core"). This is the highest-value hardening item: the
2026-07-12 corpus audit found 0% of flat manifests record a substrate hash, which is
precisely why no historical baseline can be safely reused
(arm_reuse_fingerprint_plan.md:128-133).

Purpose
-------
`stamp_recording_core(manifest, config=..., seeds=..., script_path=...)` merges the
mandatory always-core provenance/reproducibility fields onto an experiment manifest in
ONE line, so an author cannot forget them and every manifest carries the same fixed
skeleton (standard principle 1: small mandatory identity+provenance core). It is a
NO-OP-SAFE additive merge: by default it fills only fields that are absent/empty, never
clobbering a value the script already set (pass `overwrite=True` to force).

Always-core fields it stamps (standard 3b)
------------------------------------------
  recording_schema : "rec/v1"  -- the self-declaring manifest-shape version.
  substrate_hash   : content hash over ree_core/** + env + _lib/** (+ the driver
                     script). For a MULTI-ARM run it is HOISTED from
                     arm_results[i].arm_fingerprint.substrate_hash (already computed by
                     the arm-fingerprint machinery) so the top-level value matches the
                     per-cell fingerprints by construction; for a single-arm run it is
                     computed fresh via experiments/_lib/arm_fingerprint.py.
  substrate_stable_across_run : bool -- False iff the substrate provably moved during
                     the run (per-cell hashes disagree, or the process snapshot no
                     longer matches disk at stamp time). Deliberately NOT in
                     ALWAYS_CORE_KEYS: the pre-2026-07-20 corpus cannot carry it, and
                     making it core would turn every legacy manifest into a WARN.
  arm_knobs_effective : bool -- False iff some pair of arms DECLARED distinct ran
                     bit-identically on every recorded per-cell field at matched seed,
                     i.e. the knob naming their difference was inert (the V3-EXQ-689d D2
                     defect, which silently degrades conjunctive acceptance criteria).
                     See experiments/_lib/inert_arm_knob.py. Deliberately NOT in
                     ALWAYS_CORE_KEYS, for the same legacy-corpus reason as above.
  dose_levels_separable : bool -- False iff two DECLARED DOSE LEVELS produced values
                     identical beyond float noise, i.e. the measured quantity saturated
                     before the dose could express itself (the V3-EXQ-794 defect, where
                     overconfidence_score was bit-identical at asymmetry 0.6 and 0.8).
                     Complementary to arm_knobs_effective: there the knob never reached a
                     live path, here it did and a bound erased its effect. See
                     experiments/_lib/dose_saturation.py. Also NOT in ALWAYS_CORE_KEYS.
  machine          : socket.gethostname() (or a caller override -- the hub records
                     "ree-cloud-1" although its hostname is "ree-worker-1").
  machine_class    : arm_fingerprint.machine_class() -- fingerprint equality is
                     machine-class-bound, so this is the class the substrate_hash is
                     valid within.
  elapsed_seconds  : wallclock. Pass it directly, or pass started_at (a perf_counter()
                     value captured at run start) and the helper computes the delta.
  config           : the full config snapshot (env params + hyperparameters + schedule).
  seeds            : the explicit seed LIST (a single int is coerced to [int]).

Design (standard principle 4: additive, forward-compatible)
-----------------------------------------------------------
Additive-only and non-destructive: unknown/older manifests keep every field they had;
new fields are only ADDED. Safe to call unconditionally at manifest-build time in any
experiment_purpose (evidence / diagnostic / baseline).

ASCII-only output (repo rule). Stdlib + arm_fingerprint (itself stdlib-only), so this is
importable without torch/ree_core.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

# arm_fingerprint is a sibling module in this package and stdlib-only. Import it
# robustly across the several ways experiment scripts put experiments/ on sys.path
# (package import, _lib-on-path, cwd=experiments/).
try:  # normal package import (scripts do `from experiments._lib... import ...`)
    from experiments._lib import arm_fingerprint as _afp  # type: ignore
except Exception:  # pragma: no cover - path-dependent fallbacks
    try:
        from . import arm_fingerprint as _afp  # type: ignore
    except Exception:
        import arm_fingerprint as _afp  # type: ignore

# Same triple-fallback import shape as arm_fingerprint above -- inert_arm_knob is a
# sibling module in this package and stdlib-only.
try:  # normal package import
    from experiments._lib import inert_arm_knob as _inert_arm_knob  # type: ignore
except Exception:  # pragma: no cover - path-dependent fallbacks
    try:
        from . import inert_arm_knob as _inert_arm_knob  # type: ignore
    except Exception:
        import inert_arm_knob as _inert_arm_knob  # type: ignore

# Same triple-fallback import shape -- dose_saturation is a sibling module in this
# package and stdlib-only.
try:  # normal package import
    from experiments._lib import dose_saturation as _dose_saturation  # type: ignore
except Exception:  # pragma: no cover - path-dependent fallbacks
    try:
        from . import dose_saturation as _dose_saturation  # type: ignore
    except Exception:
        import dose_saturation as _dose_saturation  # type: ignore

RECORDING_SCHEMA = "rec/v1"

# The always-core keys this helper is responsible for. Kept as a named tuple so a
# validator (validate_recording.py) can import the canonical list rather than
# re-hardcoding it.
ALWAYS_CORE_KEYS: Sequence[str] = (
    "recording_schema",
    "substrate_hash",
    "machine",
    "machine_class",
    "elapsed_seconds",
    "config",
    "seeds",
)


def _is_empty(value: Any) -> bool:
    """A field counts as absent-for-fill purposes when it is None or an empty
    container/string. A meaningful 0 / False is NOT empty (so an explicit
    elapsed_seconds=0.0 is respected)."""
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict, set)) and len(value) == 0:
        return True
    return False


def _coerce_seed_list(seeds: Any) -> Optional[List[int]]:
    """Normalise seeds to an explicit list of ints, or None if not derivable.

    Accepts a single int, an iterable of ints, or None. Non-int members are kept
    verbatim (best-effort) rather than dropped, so a caller passing e.g. string
    seeds still records something rather than nothing.
    """
    if seeds is None:
        return None
    if isinstance(seeds, bool):  # bool is an int subclass -- treat as a scalar seed
        return [int(seeds)]
    if isinstance(seeds, int):
        return [seeds]
    if isinstance(seeds, (list, tuple, set)):
        out: List[Any] = []
        for s in seeds:
            try:
                out.append(int(s))
            except (TypeError, ValueError):
                out.append(s)
        return out
    return None


def multi_arm_substrate_hashes(manifest: Mapping[str, Any]) -> List[str]:
    """Distinct arm_results[i].arm_fingerprint.substrate_hash values, first-seen order.

    Cardinality > 1 means the run's cells do NOT agree on which substrate they ran --
    the intra-run divergence defect (D3). Exposed publicly because arm_reuse needs
    exactly this test to refuse serving a cell out of a divergent run.
    """
    out: List[str] = []
    arm_results = manifest.get("arm_results")
    if not isinstance(arm_results, list):
        return out
    for cell in arm_results:
        if not isinstance(cell, dict):
            continue
        fp = cell.get("arm_fingerprint")
        if isinstance(fp, dict):
            sh = fp.get("substrate_hash")
            if isinstance(sh, str) and sh and sh not in out:
                out.append(sh)
    return out


def _hoist_multi_arm_substrate_hash(manifest: Mapping[str, Any]) -> Optional[str]:
    """Return the first arm's substrate_hash, or None if there is none to hoist.

    HOISTING IS LOSSY AND WAS ONCE A TRAP. This function's original contract asserted
    that "all arms of one run execute against the same substrate, so the first present
    hash is authoritative". The 2026-07-20 corpus sweep falsified that outright: 42 of
    164 fingerprinted runs (25.6%) changed substrate mid-run, and because this hoist
    keeps only the FIRST hash, every one of them recorded a single clean-looking value
    at the top level -- the per-run field actively HID the divergence it was meant to
    summarise (intra_run_substrate_divergence_sweep_2026-07-20.md sec 1).

    The hoist is kept as-is for backward compatibility (the top-level field still means
    "a substrate hash from this run"), but it is no longer the whole story: callers get
    `substrate_stable_across_run` beside it, and the authoritative per-cell set is
    available via multi_arm_substrate_hashes(). Two mitigations now sit upstream --
    arm_fingerprint resolves substrate identity once per process, so a stable run cannot
    silently split; and stamp_recording_core records the stability verdict below.
    """
    hashes = multi_arm_substrate_hashes(manifest)
    return hashes[0] if hashes else None


def compute_single_arm_substrate_hash(
    script_path: Optional[Union[str, Path]] = None,
    extra_substrate_paths: Optional[Iterable[Union[str, Path]]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> str:
    """Compute a top-level substrate_hash for a single-arm run.

    Hashes ree_core/** + env + _lib/** (the arm-fingerprint substrate glob) plus the
    driver script (so a driver edit correctly flips the hash), matching the
    include_driver_script_in_hash=True default of the arm-fingerprint machinery.
    """
    extra: List[Path] = []
    if script_path:
        extra.append(Path(script_path))
    if extra_substrate_paths:
        extra.extend(Path(p) for p in extra_substrate_paths)
    sub = _afp.compute_substrate_hash(
        extra_paths=extra or None,
        repo_root=Path(repo_root) if repo_root else None,
    )
    return str(sub["substrate_hash"])


def stamp_recording_core(
    manifest: Dict[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    seeds: Any = None,
    script_path: Optional[Union[str, Path]] = None,
    *,
    elapsed_seconds: Optional[float] = None,
    started_at: Optional[float] = None,
    machine: Optional[str] = None,
    extra_substrate_paths: Optional[Iterable[Union[str, Path]]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Merge the always-record core onto `manifest` in place and return it.

    NO-OP-SAFE: by default only absent/empty fields are filled, so an existing value
    the script deliberately set is preserved (pass overwrite=True to force). A
    meaningful 0/False is NOT treated as empty.

    Parameters
    ----------
    manifest
        The manifest dict being built (mutated in place; also returned for chaining).
    config
        The full config snapshot. Recorded verbatim under `config` (standard 3b
        reproducibility core). If None and the manifest already carries `config`, it
        is left as-is.
    seeds
        The explicit seed list (a single int is coerced to [int]).
    script_path
        The driver script -- `Path(__file__)` from the experiment. Folded into a
        freshly-computed single-arm substrate_hash. Ignored for the hoist path.
    elapsed_seconds
        Wallclock seconds. Takes precedence over started_at.
    started_at
        A time.perf_counter() value captured at run start; elapsed is computed as
        perf_counter() - started_at when elapsed_seconds is not given.
    machine
        Override for the recorded machine name (default socket.gethostname()).
    extra_substrate_paths, repo_root
        Passed through to the single-arm substrate-hash computation.
    overwrite
        Force-overwrite already-present fields (default False -> fill-only).

    Returns the same manifest dict.
    """

    def _fill(key: str, value: Any) -> None:
        if value is None:
            return
        if overwrite or _is_empty(manifest.get(key)):
            manifest[key] = value

    # recording_schema -- the self-declaring version primitive.
    _fill("recording_schema", RECORDING_SCHEMA)

    # substrate_hash -- hoist from the per-arm fingerprints for a multi-arm run,
    # else compute fresh for a single-arm run. Only compute if we would actually
    # fill (avoid the file-hashing cost when the field is already present).
    if overwrite or _is_empty(manifest.get("substrate_hash")):
        hoisted = _hoist_multi_arm_substrate_hash(manifest)
        if hoisted:
            _fill("substrate_hash", hoisted)
        else:
            try:
                _fill(
                    "substrate_hash",
                    compute_single_arm_substrate_hash(
                        script_path=script_path,
                        extra_substrate_paths=extra_substrate_paths,
                        repo_root=repo_root,
                    ),
                )
            except Exception:
                # Never let provenance stamping crash an experiment. A missing
                # substrate_hash is a soft-validate WARN, not a run failure.
                pass

    # substrate_stable_across_run -- did the substrate hold still for the whole run?
    # Two independent tests, either of which can only ever prove INSTABILITY:
    #   (a) the run's own per-cell fingerprints disagree (cardinality > 1) -- this is
    #       the D3 signature the 2026-07-20 sweep found on 42 of 164 runs, and it is
    #       decisive on its own even for a manifest built in some other process;
    #   (b) the substrate this process HASHED at its first cell no longer matches disk
    #       (arm_fingerprint.substrate_stability_report) -- catches a mid-run checkout
    #       move that (a) cannot see, precisely because the process-snapshot fix now
    #       keeps all cells agreeing.
    # False is the informative value: it records the checkout move as the instrument
    # event it is, and arm_reuse refuses to serve a cell out of such a run. Stamped
    # unconditionally (not only when empty) is WRONG -- an author who explicitly set it
    # must win, so it goes through _fill like everything else.
    if overwrite or _is_empty(manifest.get("substrate_stable_across_run")):
        try:
            cells_disagree = len(multi_arm_substrate_hashes(manifest)) > 1
            report = _afp.substrate_stability_report()
            stable = bool(report.get("substrate_stable_across_run", True)) and not cells_disagree
            # _fill() skips a meaningful False? No -- _is_empty treats False as present,
            # so assign directly rather than via _fill, which would refuse to write it.
            manifest["substrate_stable_across_run"] = stable
            if not stable:
                manifest["substrate_stability_detail"] = {
                    "per_cell_hashes_disagree": cells_disagree,
                    "distinct_cell_substrate_hashes": multi_arm_substrate_hashes(manifest),
                    "process_snapshot_drift": report.get("drift", []),
                    "checked_utc": report.get("checked_utc"),
                }
        except Exception:
            # Never let provenance stamping crash an experiment (same posture as the
            # substrate_hash branch above). An absent field is a soft WARN, and the
            # reuse path treats "absent" as unproven-but-not-disproven, falling back to
            # the per-cell cardinality test it can compute for itself.
            pass

    # arm_knobs_effective -- did every declared-distinct arm pair actually run differently?
    # Purely manifest-local (no substrate dependency): it compares recorded per-cell fields
    # at matched seed. Catches the V3-EXQ-689d D2 defect, where ARM_PROPOSER_CTRL and
    # ARM_MATCHED_NOISE were bit-identical on 26 of 27 fields at all three seeds and
    # differed only in the `temperature` field naming their intended difference -- so the
    # conjunctive C_PRIMARY silently degraded to one of its conjuncts and the run PASSED.
    # RECORD-AND-WARN, never a hard failure: by manifest-write time the compute is spent.
    # The helper is internally exception-safe; the guard here covers the import itself.
    try:
        _inert_arm_knob.stamp_inert_arm_knob(manifest)
    except Exception:
        pass

    # dose_levels_separable -- did two DECLARED DOSE LEVELS produce different values?
    # Sibling of the check above, catching the complementary defect: there the knob
    # never reached a live path (arms ran identically); here the knob DID move the
    # dynamics and a bound downstream erased the difference, so the arms differ while
    # the readouts are bit-identical. The V3-EXQ-794 defect: overconfidence_score was
    # -1.004111904519277 at BOTH asymmetry 0.6 and 0.8 because rv was clamped at an
    # absolute floor sitting above the operating point. Same RECORD-AND-WARN posture.
    try:
        _dose_saturation.stamp_dose_saturation(manifest)
    except Exception:
        pass

    # machine / machine_class -- where it ran + the class the hash is valid within.
    _fill("machine", machine if machine else socket.gethostname())
    try:
        _fill("machine_class", _afp.machine_class())
    except Exception:
        pass

    # elapsed_seconds -- explicit value wins; else derive from started_at.
    if elapsed_seconds is not None:
        _fill("elapsed_seconds", float(elapsed_seconds))
    elif started_at is not None:
        _fill("elapsed_seconds", float(time.perf_counter() - started_at))

    # config -- the full config snapshot (reproducibility + fingerprinting).
    if config is not None:
        _fill("config", dict(config))

    # seeds -- the explicit seed list.
    seed_list = _coerce_seed_list(seeds)
    if seed_list is not None:
        _fill("seeds", seed_list)

    return manifest


def missing_core_fields(manifest: Mapping[str, Any]) -> List[str]:
    """Return the always-core keys absent/empty on `manifest` (soft-validate helper).

    Used by validate_recording.py to WARN on an under-recorded manifest without
    re-hardcoding the key list. A field present but empty ([] / "" / {}) counts as
    missing; a meaningful 0/False does not.
    """
    return [k for k in ALWAYS_CORE_KEYS if _is_empty(manifest.get(k, None))]


__all__ = [
    "RECORDING_SCHEMA",
    "ALWAYS_CORE_KEYS",
    "compute_single_arm_substrate_hash",
    "multi_arm_substrate_hashes",
    "stamp_recording_core",
    "missing_core_fields",
]
