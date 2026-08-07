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
  substrate_identity : {source, resolved_at_utc, stamped_at_utc, lag_seconds,
                     drifted_since_resolved, stability_snapshots,
                     hash_on_disk_at_stamp?, commit_source, commit_resolved_at_utc,
                     commit_describes_recorded_hash?} -- WHEN
                     the two fields above were taken, not only what they were.
                     `source` is process_snapshot (frozen at the first arm cell or by
                     pin_recording_substrate -- the EXECUTED substrate),
                     per_cell_fingerprint_hoist, or manifest_write_disk_read (this
                     stamp was the first look, so the value describes the checkout as
                     it is NOW). The V3-EXQ-866b gap: a diagnostic whose whole purpose
                     was certifying substrate stability recorded a hash read 2h26m
                     after its seed cells, disagreeing with the cells' own snapshot,
                     with nothing in the manifest saying the two were taken at
                     different moments. Deliberately NOT in ALWAYS_CORE_KEYS, for the
                     same legacy-corpus reason as the fields below.
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
  z_goal_stream    : {ticks_total, ticks_active, writer_calls, active_frac,
                     writer_defect, goal_state_present, n_agents} -- was the z_goal stream
                     LIVE during the run, read off the agent's own per-tick counters.
                     active_frac 0.0 says the stream was dead (every consumer got
                     current_z_goal=None); writer_calls says WHY, and only
                     writer_calls == 0 is the missing-call defect -- a correctly-wired run
                     whose benefit gate never opened also reads 0.0. writer_defect is that
                     verdict precomputed. Requires the
                     caller to pass `agent=` (or `z_goal_stream_stats=`); the block is
                     OMITTED rather than zero-filled when they don't, so its presence
                     always means the run measured it. See experiments/_lib/z_goal_stream.py.
                     Also NOT in ALWAYS_CORE_KEYS -- the legacy corpus cannot carry it, and
                     it is unavailable to any manifest built outside the stepping process.
  episode_termination : {n_episodes, steps_configured, steps_mean/min/max,
                     full_budget_frac, truncated_frac, causes} -- did the episodes
                     actually SPEND the configured step budget, and if not, why did they
                     stop? A manifest records a configured n_steps and an aggregate
                     outcome and nothing in between, so a run whose episodes died on step
                     19 of a configured 400 reads exactly like one that ran the full
                     budget with worse numbers (the V3-EXQ-884 / SD-094 defect: 32/19/90
                     of 400 steps, all self-inflicted contamination death, established
                     only by re-running the experiment inside a failure autopsy).
                     Requires the caller to pass `episode_termination=` (an
                     EpisodeTerminationAccumulator or its stats); OMITTED rather than
                     zero-filled when they don't, so its presence always means the run
                     measured it. See experiments/_lib/episode_termination.py. Also NOT
                     in ALWAYS_CORE_KEYS, for the same legacy-corpus reason as above.
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

import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
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

# Same triple-fallback import shape -- z_goal_stream is a sibling module in this
# package, and duck-typed/stdlib-only so this module keeps its no-torch/no-ree_core
# import guarantee even though the block it stamps is read off a live agent.
try:  # normal package import
    from experiments._lib import z_goal_stream as _z_goal_stream  # type: ignore
except Exception:  # pragma: no cover - path-dependent fallbacks
    try:
        from . import z_goal_stream as _z_goal_stream  # type: ignore
    except Exception:
        import z_goal_stream as _z_goal_stream  # type: ignore

# Same triple-fallback import shape -- episode_termination is a sibling module in
# this package and stdlib-only.
try:  # normal package import
    from experiments._lib import episode_termination as _episode_termination  # type: ignore
except Exception:  # pragma: no cover - path-dependent fallbacks
    try:
        from . import episode_termination as _episode_termination  # type: ignore
    except Exception:
        import episode_termination as _episode_termination  # type: ignore

RECORDING_SCHEMA = "rec/v1"

# The always-core keys this helper is responsible for. Kept as a named tuple so a
# validator (validate_recording.py) can import the canonical list rather than
# re-hardcoding it.
ALWAYS_CORE_KEYS: Sequence[str] = (
    "recording_schema",
    "substrate_hash",
    "substrate_commit",
    "machine",
    "machine_class",
    "elapsed_seconds",
    "config",
    "seeds",
)

# Git repo-location env vars that a PARENT git process (a pre-commit hook shelling
# out to python, the runner's own git invocation) exports into our environment.
# They MUST be stripped before `git rev-parse` below, or the SHA resolves against
# whatever repo the parent was pointed at instead of the substrate this run
# actually executed -- recording a confidently WRONG provenance value, which is
# strictly worse than recording none. Same list and same reason as
# pack_writer._GIT_LOCATION_ENV_VARS and tests/contracts/test_arm_reuse.py's
# _resolve_ree_working_root; duplicated rather than imported because pack_writer
# imports THIS module (importing back would be circular) and this module keeps a
# stdlib-only guarantee.
_GIT_LOCATION_ENV_VARS = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX",
    "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_INDEX_VERSION",
    "GIT_CEILING_DIRECTORIES", "GIT_DISCOVERY_ACROSS_FILESYSTEM",
)

# Fallback if arm_fingerprint ever stops exposing _SUBSTRATE_GLOBS. Kept in sync
# by test_substrate_commit.py, which asserts the two agree rather than trusting
# this copy.
_SUBSTRATE_GLOBS_FALLBACK: Sequence[str] = (
    "ree_core/**/*.py",
    "experiments/_harness.py",
    "experiments/_metrics.py",
    "experiments/_lib/**/*.py",
)

# Cap on the dirty-path list recorded for diagnosis. A dirty substrate is already
# the exceptional case; the names are what make it actionable, but an unbounded
# list could balloon a manifest during a large refactor.
_MAX_DIRTY_PATHS = 20


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


def _first_cell_identity_resolved_at(manifest: Mapping[str, Any]) -> Optional[str]:
    """`substrate_identity_resolved_at` off the first arm cell that recorded one.

    A hoisted top-level hash inherits the CELL's provenance, so it inherits the
    cell's resolution time too -- which is the first cell of the run, not the
    manifest write. Absent on pre-2026-07-20 cells, and absent is reported as
    absent rather than back-filled with the stamp time.
    """
    arm_results = manifest.get("arm_results")
    if not isinstance(arm_results, list):
        return None
    for cell in arm_results:
        if not isinstance(cell, dict):
            continue
        fp = cell.get("arm_fingerprint")
        if isinstance(fp, dict):
            at = fp.get("substrate_identity_resolved_at")
            if isinstance(at, str) and at:
                return at
    return None


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


def _single_arm_extra_paths(
    script_path: Optional[Union[str, Path]] = None,
    extra_substrate_paths: Optional[Iterable[Union[str, Path]]] = None,
) -> Optional[List[Path]]:
    """The `extra_paths` half of the single-arm substrate-identity key.

    Factored out because THREE call sites must agree on it byte-for-byte or they
    address different process snapshots: the resolve below, the pin helper, and the
    pinned-vs-not probe. A key that disagrees by one path silently resolves a second
    snapshot at manifest-write time, which is the exact defect this module now
    records against.
    """
    extra: List[Path] = []
    if script_path:
        extra.append(Path(script_path))
    if extra_substrate_paths:
        extra.extend(Path(p) for p in extra_substrate_paths)
    return extra or None


def resolve_single_arm_substrate_identity(
    script_path: Optional[Union[str, Path]] = None,
    extra_substrate_paths: Optional[Iterable[Union[str, Path]]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """The top-level substrate identity for a single-arm run, with its provenance.

    Returns the `resolve_substrate_identity` dict plus `pinned_before_call`: True
    when this process had ALREADY frozen this identity (at its first arm cell, or
    via `pin_recording_substrate`), False when this call is the first disk read.

    THE PROCESS SNAPSHOT, NOT A FRESH DISK READ -- this is the fix, and the
    one-word difference is the whole bug. This helper previously called
    `compute_substrate_hash` directly, so on a single-arm run the top-level
    `substrate_hash` was a disk read taken at MANIFEST-WRITE time, hours after the
    seed cells it purports to describe, while the cells' own fingerprints were
    served from the process snapshot frozen at the first cell (the 2026-07-20
    executed-substrate fix, arm_fingerprint's "THE FIX, in two parts"). The two
    paths therefore disagreed by construction whenever the shared checkout moved
    mid-run -- routine on the hub, where sync_daemon writers and other sessions
    commit continuously.

    Confirmed on V3-EXQ-866b, a diagnostic whose entire purpose was certifying
    substrate stability: cells pinned `bb755658...` at 16:58:16Z, the manifest
    stamped `8e275408...` read at 19:24:05Z. Reproduced exactly by re-hashing the
    tree at the recorded `substrate_commit` (c4247794) -- the driver-INCLUSIVE hash
    there is `8e275408...`, i.e. the same key the cells had already pinned, read
    again 2h26m later. Corpus scan of `evidence/experiments/`: 10 of 249 manifests
    carrying a substrate_hash are in this state, with lags up to two days.

    Routing through the snapshot makes the recorded value the EXECUTED one for
    every driver that stamps at least one arm cell before writing its manifest --
    no driver change, since `arm_cell(script_path=Path(__file__))` pins exactly this
    key. A driver that builds no cells still resolves here for the first time; that
    case is unchanged in value and is now labelled honestly rather than silently
    (see `stamp_recording_core`'s `substrate_identity` block).
    """
    extra = _single_arm_extra_paths(script_path, extra_substrate_paths)
    root = Path(repo_root) if repo_root else None
    try:
        pinned = bool(_afp.substrate_identity_pinned(extra_paths=extra, repo_root=root))
    except Exception:  # pragma: no cover - older arm_fingerprint without the probe
        pinned = False
    sub = dict(_afp.resolve_substrate_identity(extra_paths=extra, repo_root=root))
    sub["pinned_before_call"] = pinned
    return sub


def compute_single_arm_substrate_hash(
    script_path: Optional[Union[str, Path]] = None,
    extra_substrate_paths: Optional[Iterable[Union[str, Path]]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> str:
    """The top-level substrate_hash for a single-arm run.

    Covers ree_core/** + env + _lib/** (the arm-fingerprint substrate glob) plus the
    driver script (so a driver edit correctly flips the hash), matching the
    include_driver_script_in_hash=True default of the arm-fingerprint machinery.

    Served from the process snapshot -- see resolve_single_arm_substrate_identity
    for why that matters and what it fixes.
    """
    return str(
        resolve_single_arm_substrate_identity(
            script_path=script_path,
            extra_substrate_paths=extra_substrate_paths,
            repo_root=repo_root,
        )["substrate_hash"]
    )


def _git_value(
    args: Sequence[str], cwd: Path, *, strip: bool = True
) -> Optional[str]:
    """Run a read-only git command in `cwd`. Returns stdout on success --
    POSSIBLY THE EMPTY STRING -- and None only on failure.

    ``strip=False`` right-strips ONLY, and is required for ``status --porcelain``:
    a porcelain v1 line is ``XY<space>PATH`` where column 0 is the staged status
    and is a SPACE for the common unstaged-modification case (`` M path``). A full
    ``.strip()`` silently eats that leading space on the FIRST line only, shifting
    the fixed path offset by one and yielding a truncated name (``xperiments/...``)
    for the first dirty file while every later line parses correctly -- a
    corrupted-but-plausible value, which is the worst failure shape for a
    provenance field.

    The empty-vs-None distinction is load-bearing and is deliberately NOT the
    ``stdout.strip() or None`` idiom used elsewhere in the repo. For
    ``git status --porcelain`` the CLEAN case *is* empty output, so collapsing ""
    to None would make a clean substrate indistinguishable from a failed probe and
    report ``dirty: null`` on virtually every run -- the common case, silently
    unmeasured. Callers that want falsy-on-empty can still just test the string.

    Never raises: a missing git binary, a non-repo directory, or a non-zero exit
    all yield None. Provenance stamping must never be able to fail an experiment
    that has already spent its compute.
    """
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_ENV_VARS}
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(cwd), check=True,
            capture_output=True, text=True, env=env,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return result.stdout.strip() if strip else result.stdout.rstrip("\n")


def _git_pathspecs_from_globs(globs: Sequence[str]) -> Sequence[str]:
    """Convert pathlib-style substrate globs into git pathspecs covering >= the same files.

    THE TWO GLOB DIALECTS DISAGREE, AND NAIVELY REUSING THE GLOBS UNDER-REPORTS.
    ``substrate_hash`` selects files with ``Path.glob``, where ``**`` matches ZERO
    or more directories -- so ``ree_core/**/*.py`` includes ``ree_core/agent.py``
    and ``experiments/_lib/**/*.py`` includes ``experiments/_lib/manifest_core.py``.
    Git's default pathspec matching treats the same string as wildmatch with a
    LITERAL ``/`` after ``**``, so both of those top-level files fail to match and
    ``git status -- <glob>`` reports a clean tree while the hashed substrate is in
    fact modified. Measured on this repo: 4 top-level ``ree_core/*.py`` files plus
    every top-level ``_lib`` module were invisible that way.

    So each glob is truncated at its first wildcard component, yielding a plain
    directory-or-file pathspec that git matches recursively. That is an
    OVER-approximation (a directory spec also covers non-.py files in those trees),
    and the direction is chosen deliberately: a spurious ``dirty: true`` is a cheap
    over-report that costs a reader one `git diff`, whereas a false ``dirty: false``
    is the silent under-report this whole field exists to prevent.
    """
    specs: List[str] = []
    for g in globs:
        keep: List[str] = []
        for part in str(g).split("/"):
            if any(ch in part for ch in "*?["):
                break
            keep.append(part)
        spec = "/".join(keep) if keep else "."
        if spec and spec not in specs:
            specs.append(spec)
    return tuple(specs)


def substrate_commit(repo_root: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Identify WHICH commit the substrate this run executed corresponds to.

    Returns ``{"commit", "dirty", "branch"?, "dirty_count"?, "dirty_paths"?}``, or
    None when the checkout cannot be resolved (no git binary, not a repo, unborn
    HEAD). ``dirty`` is None -- not False -- when HEAD resolved but the status
    probe did not, so "unverified" is never reported as "clean".

    WHY THIS EXISTS BESIDE ``substrate_hash``, WHICH ALREADY DETECTS CHANGE.
    ``substrate_hash`` is a content hash: it proves two runs executed different
    substrate, but it is opaque, so it cannot tell you WHAT differed. Diagnosis
    needs a commit you can `git diff`. The motivating case is V3-EXQ-614 vs 614a --
    bit-identical drivers, identical seeds, identical ``config_summary``, and a
    verdict that flipped FAIL -> PASS because ``e3_diversity_entropy_lambda``
    changed 0.05 -> 0.5 in ree-v3 `a45ca7f`, which landed between the two runs.
    A hash pair would have said "these differ"; the commit pair reduces it to one
    `git diff`. The two fields are complementary and both are stamped:
    hash = detection (and covers uncommitted edits a SHA cannot), commit = diagnosis.

    ``dirty`` IS DELIBERATELY SCOPED, and the scope is the whole point. It covers
    exactly the trees ``substrate_hash`` hashes (arm_fingerprint._SUBSTRATE_GLOBS),
    so it means "the substrate this run executed differs from the recorded commit".
    An unscoped ``git status --porcelain`` would report dirty on nearly every run in
    these shared multi-session checkouts -- one open ``experiment_queue.json`` edit
    by an unrelated session is enough -- and a flag that is true almost always
    carries no information. When dirty, up to ``_MAX_DIRTY_PATHS`` offending paths
    are recorded, because "dirty" alone is not actionable but the names are.

    The repo is resolved from the RUNNING CODE's own location (arm_fingerprint's
    ``_REPO_ROOT``, i.e. this file's ``parents[2]``), never from cwd and never from
    a hardcoded path: the hub and each cloud worker run from their own checkouts,
    and a worktree must report its own HEAD.
    """
    root = Path(repo_root).resolve() if repo_root else getattr(_afp, "_REPO_ROOT", Path(__file__).resolve().parents[2])
    root = Path(root)
    commit = _git_value(["rev-parse", "HEAD"], root)
    if not commit:
        # Not a repo, no git, or an unborn HEAD. Absent beats wrong.
        return None

    out: Dict[str, Any] = {"commit": commit}

    globs = getattr(_afp, "_SUBSTRATE_GLOBS", None) or _SUBSTRATE_GLOBS_FALLBACK
    status = _git_value(
        ["status", "--porcelain", "--", *_git_pathspecs_from_globs(globs)],
        root, strip=False,
    )
    if status is None:
        # HEAD resolved but status did not: report the commit without claiming
        # cleanliness we did not verify.
        out["dirty"] = None
    else:
        paths = []
        for line in status.splitlines():
            # porcelain v1: 2 status columns + a space, then the path.
            entry = line[3:].strip() if len(line) > 3 else ""
            # A rename/copy renders as "OLD -> NEW"; the NEW path is the live one.
            if " -> " in entry:
                entry = entry.split(" -> ", 1)[1]
            entry = entry.strip('"')
            if entry:
                paths.append(entry)
        out["dirty"] = bool(paths)
        if paths:
            uniq = sorted(set(paths))
            # Record the TOTAL beside the capped list. A silently truncated list
            # reads as the complete set, which would understate the blast radius
            # of a dirty substrate at exactly the moment someone is trying to
            # judge whether a run is trustworthy. len(dirty_paths) < dirty_count
            # is the self-evident truncation signal.
            out["dirty_count"] = len(uniq)
            out["dirty_paths"] = uniq[:_MAX_DIRTY_PATHS]

    branch = _git_value(["rev-parse", "--abbrev-ref", "HEAD"], root)
    if branch and branch != "HEAD":
        out["branch"] = branch
    return out


# Pinned substrate COMMITs, keyed by resolved repo root. Anchored on `sys` rather
# than on this module's globals for exactly the reason arm_fingerprint's snapshot
# state is (see its _STATE_ATTR block): this file is imported under at least two
# distinct names in a normal experiment process, and two module dicts would give
# two disagreeing pins.
_COMMIT_STATE_ATTR = "_ree_manifest_core_pinned_substrate_commit"
_PINNED_COMMITS: Dict[str, Dict[str, Any]] = getattr(sys, _COMMIT_STATE_ATTR, None) or {}
setattr(sys, _COMMIT_STATE_ATTR, _PINNED_COMMITS)


def _commit_pin_key(repo_root: Optional[Union[str, Path]]) -> str:
    root = repo_root or getattr(_afp, "_REPO_ROOT", Path(__file__).resolve().parents[2])
    return str(Path(root).resolve())


def _reset_recording_substrate_pins() -> None:
    """Clear the pinned commits. TEST-ONLY -- an experiment must never call this."""
    _PINNED_COMMITS.clear()


def pin_recording_substrate(
    script_path: Optional[Union[str, Path]] = None,
    extra_substrate_paths: Optional[Iterable[Union[str, Path]]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Freeze the substrate identity + commit NOW, for a manifest written later.

    Call ONCE at run start (before the first seed's work) from a driver that
    stamps NO arm cells. A driver using `arm_cell(script_path=Path(__file__))`
    already pins the hash half at its first cell and does not need this for the
    hash -- but calling it also pins the COMMIT, which nothing else does.

    WHY A COMMIT PIN IS NOT REDUNDANT WITH THE HASH PIN. The hash is recoverable
    late-but-correct (the snapshot survives in-process); the commit is not
    recoverable at all -- `git rev-parse HEAD` at manifest-write time answers about
    the checkout as it is THEN, and on V3-EXQ-866b that answer (c4247794) was the
    commit of the tree that had replaced the executed one, presented beside a hash
    the run never executed. Once the checkout has moved, no amount of care at write
    time can recover which commit the run actually ran; it has to be read while it
    is still true.

    Returns the pinned record ({substrate_hash, resolved_at_utc, substrate_commit}).
    Never raises: a failure to pin leaves the write-time behaviour in place, which
    is exactly the pre-fix behaviour, and provenance stamping must never be able to
    fail an experiment that has not yet spent its compute.
    """
    out: Dict[str, Any] = {}
    try:
        sub = resolve_single_arm_substrate_identity(
            script_path=script_path,
            extra_substrate_paths=extra_substrate_paths,
            repo_root=repo_root,
        )
        out["substrate_hash"] = sub.get("substrate_hash")
        out["resolved_at_utc"] = sub.get("resolved_at_utc")
    except Exception:
        pass
    try:
        key = _commit_pin_key(repo_root)
        if key not in _PINNED_COMMITS:
            commit = substrate_commit(repo_root=repo_root)
            if commit:
                record = dict(commit)
                record["resolved_at_utc"] = _utc_now()
                _PINNED_COMMITS[key] = record
        out["substrate_commit"] = _PINNED_COMMITS.get(key)
    except Exception:
        pass
    return out


def _utc_now() -> str:
    # ASCII, UTC, second resolution -- matches arm_fingerprint._utc_now and the repo
    # timestamp convention. Local rather than imported to keep this module's
    # stdlib-only guarantee independent of arm_fingerprint's private surface.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_delta_seconds(earlier: Optional[str], later: Optional[str]) -> Optional[int]:
    """Whole seconds between two "%Y-%m-%dT%H:%M:%SZ" stamps, or None if unparseable.

    Never raises and never guesses: an unparseable stamp yields None, so a missing
    lag is visibly missing rather than silently 0 (which would read as "stamped at
    the instant it was resolved" -- the exact claim this block exists to stop
    manifests making without evidence).
    """
    if not earlier or not later:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        t0 = datetime.strptime(str(earlier), fmt)
        t1 = datetime.strptime(str(later), fmt)
    except (ValueError, TypeError):
        return None
    return int((t1 - t0).total_seconds())


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
    agent: Any = None,
    z_goal_stream_stats: Optional[Mapping[str, Any]] = None,
    episode_termination: Any = None,
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
    agent
        The stepped REEAgent, or an iterable of them (a multi-arm run builds one per
        arm x seed). Read for the `z_goal_stream` liveness block. Omitting it simply
        omits the block -- it is never fabricated from zeros.
    z_goal_stream_stats
        A precomputed liveness block (from `z_goal_stream.stats_from_counts`) for a
        caller that keeps its own counters, e.g. a StepHarness accumulating across
        agent swaps. Takes precedence over `agent`.
    episode_termination
        Per-episode length + termination cause for the `episode_termination` block:
        an `episode_termination.EpisodeTerminationAccumulator`, a precomputed block,
        or a sequence of (steps, cause) pairs. Omitting it omits the block -- it is
        never fabricated, since "unmeasured" and "ran the full budget" must not look
        alike (that is the defect it exists to close).
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

    # WHEN the recorded substrate identity was taken, not just WHAT it was. Filled
    # by the branches below and stamped as `substrate_identity` at the end, once the
    # stability report (which is what knows the on-disk value at stamp time) has run.
    identity: Optional[Dict[str, Any]] = None

    # substrate_hash -- hoist from the per-arm fingerprints for a multi-arm run,
    # else resolve the single-arm process snapshot. Only resolve if we would actually
    # fill (avoid the file-hashing cost when the field is already present).
    if overwrite or _is_empty(manifest.get("substrate_hash")):
        hoisted = _hoist_multi_arm_substrate_hash(manifest)
        if hoisted:
            _fill("substrate_hash", hoisted)
            identity = {
                "source": "per_cell_fingerprint_hoist",
                "resolved_at_utc": _first_cell_identity_resolved_at(manifest),
            }
        else:
            try:
                sub = resolve_single_arm_substrate_identity(
                    script_path=script_path,
                    extra_substrate_paths=extra_substrate_paths,
                    repo_root=repo_root,
                )
                _fill("substrate_hash", str(sub["substrate_hash"]))
                identity = {
                    # The distinction the V3-EXQ-866b gap turned on. process_snapshot
                    # = frozen at the first arm cell (or by pin_recording_substrate),
                    # so it IS what executed. manifest_write_disk_read = this stamp is
                    # the first time the process looked, so the value describes the
                    # checkout as it is NOW and only coincides with what executed if
                    # nothing moved. Recorded either way: a reader must never have to
                    # assume which one a manifest is carrying.
                    "source": (
                        "process_snapshot" if sub.get("pinned_before_call")
                        else "manifest_write_disk_read"
                    ),
                    "resolved_at_utc": sub.get("resolved_at_utc"),
                }
            except Exception:
                # Never let provenance stamping crash an experiment. A missing
                # substrate_hash is a soft-validate WARN, not a run failure.
                pass

    # substrate_commit -- WHICH commit the substrate_hash above corresponds to.
    # Stamped beside the hash, never instead of it: the hash DETECTS a difference
    # (including uncommitted edits, which no SHA can express), the commit lets you
    # DIAGNOSE it with a git diff. See substrate_commit() for the V3-EXQ-614/614a
    # case that motivated it. Same never-crash posture as the hash branch above --
    # absent is a soft-validate WARN, and absent always beats a wrong SHA.
    #
    # A PIN WINS OVER A FRESH READ. If the driver called pin_recording_substrate at
    # run start, that record is the commit the run actually executed; re-reading HEAD
    # here would answer about the checkout as it is at manifest-write time, which on
    # a multi-hour run on the shared hub is a different tree (V3-EXQ-866b recorded
    # c4247794 for a run whose cells executed something else).
    if overwrite or _is_empty(manifest.get("substrate_commit")):
        try:
            pinned_commit = _PINNED_COMMITS.get(_commit_pin_key(repo_root))
            if pinned_commit:
                value = {k: v for k, v in pinned_commit.items() if k != "resolved_at_utc"}
                _fill("substrate_commit", value)
                if identity is not None:
                    identity["commit_source"] = "process_pin"
                    identity["commit_resolved_at_utc"] = pinned_commit.get("resolved_at_utc")
            else:
                _fill("substrate_commit", substrate_commit(repo_root=repo_root))
                if identity is not None:
                    identity["commit_source"] = "manifest_write_disk_read"
                    identity["commit_resolved_at_utc"] = _utc_now()
        except Exception:
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
    stability_report: Optional[Dict[str, Any]] = None
    if overwrite or _is_empty(manifest.get("substrate_stable_across_run")):
        try:
            cells_disagree = len(multi_arm_substrate_hashes(manifest)) > 1
            report = _afp.substrate_stability_report()
            stability_report = report
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

    # substrate_identity -- WHEN the recorded hash/commit were taken, and what the
    # tree looked like at stamp time if it had since moved. This is the readable half
    # of the V3-EXQ-866b fix: routing the hash through the process snapshot makes the
    # VALUE right, and this block makes the value's provenance CHECKABLE instead of
    # something a reader has to infer from a stability flag whose drift entries do not
    # say which snapshot key they belong to.
    #
    # `hash_on_disk_at_stamp` is lifted from the stability report already computed
    # above rather than re-hashing the tree -- same numbers, no second full scan, and
    # it is present ONLY when the disk really did move, so its presence alone is the
    # signal. Matched on the recorded hash, which is exact: the drift entries carry
    # repo_root and scope but not extra_paths, so matching on those would confuse a
    # driver-inclusive snapshot with a driver-exclusive one.
    if identity is not None:
        identity["stamped_at_utc"] = _utc_now()
        identity["lag_seconds"] = _utc_delta_seconds(
            identity.get("resolved_at_utc"), identity.get("stamped_at_utc")
        )
        recorded = manifest.get("substrate_hash")
        if isinstance(stability_report, dict) and isinstance(recorded, str):
            # HOW MANY snapshots the stability verdict beside this block was computed
            # over. 0 means `substrate_stable_across_run: true` is VACUOUS -- the
            # process never froze an identity to re-check, which is the state of every
            # driver that stamps no arm cell (560 of the 915 manifest-writing drivers
            # at the time of writing). A vacuous true and an earned true are identical
            # on the page otherwise, and a diagnostic that certifies substrate
            # stability must not be able to cite one for the other.
            identity["stability_snapshots"] = stability_report.get("n_snapshots")
            drifted = None
            for entry in stability_report.get("drift") or ():
                if isinstance(entry, dict) and entry.get("recorded") == recorded:
                    drifted = entry
                    break
            identity["drifted_since_resolved"] = drifted is not None
            if drifted is not None:
                identity["hash_on_disk_at_stamp"] = drifted.get("on_disk_now")
                # The commit was read from that MOVED tree, so it names a substrate
                # this run did not execute. Saying so explicitly is the point: on
                # V3-EXQ-866b the commit looked authoritative precisely because it
                # agreed with the (wrong) hash beside it.
                if identity.get("commit_source") == "manifest_write_disk_read":
                    identity["commit_describes_recorded_hash"] = False
        _fill("substrate_identity", identity)

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

    # z_goal_stream -- was the z_goal stream actually LIVE during the run?
    # The runtime half of the dead-z_goal-stream backstop, complementing the static
    # `dead_z_goal_stream` lint in validate_experiments.py, which is an AST scan and
    # so cannot see a config assembled inside a helper it can't follow (a _lib
    # builder, a **kwargs splat, a preset factory) -- it UNDER-fires by design.
    # `update_z_goal` is the SOLE z_goal writer in the substrate, so a driver that
    # hand-rolls its loop and omits it runs with z_goal pinned at zero-init and every
    # consumer silently no-opping, with nothing raised and (before this) no manifest
    # field showing it. Generalises the ad-hoc `zgoal_present_frac` readiness gate that
    # caught V3-EXQ-830, and would have caught V3-EXQ-626, which nothing did.
    # RECORD-ONLY, never a failure or a warning: an active_frac of 0.0 is CORRECT for a
    # goal-OFF parity arm or a negative control (V3-EXQ-626b's ARM_NO_BENEFIT), so this
    # is a field to read against the run's design, not a gate. Omitted entirely when no
    # counters were supplied -- absence means unmeasured, never "measured zero".
    # Same _fill posture as everything else: an explicit author value wins.
    if overwrite or _is_empty(manifest.get(_z_goal_stream.MANIFEST_KEY)):
        try:
            _z_goal_stream.stamp_z_goal_stream(
                manifest,
                agent,
                stats=dict(z_goal_stream_stats) if z_goal_stream_stats else None,
                overwrite=overwrite,
            )
        except Exception:
            pass

    # episode_termination -- did the episodes actually SPEND their step budget?
    # SD-094. A manifest records a configured n_steps and an aggregate outcome, and
    # nothing in between, so a run whose episodes died on step 19 of a configured 400
    # is indistinguishable on the page from one that ran the full budget: same config
    # block, same metric shape, merely worse numbers -- which every downstream reader
    # then interprets as a measurement of the mechanism rather than of an episode that
    # ended before the mechanism could express itself. Confirmed against V3-EXQ-884
    # (32/19/90 of 400 steps on seeds 42/43/44, all self-inflicted contamination
    # death); establishing that needed a live re-run inside a failure autopsy, for a
    # fact the run knew at every step and never wrote down.
    # RECORD-ONLY, same posture as z_goal_stream above: early termination is often the
    # legitimate measurement (a survival experiment, a deliberate lethal arm), so this
    # is a field to read against the run's design, not a gate. Omitted entirely when
    # the driver collected nothing -- absence means unmeasured, never "ran full".
    if overwrite or _is_empty(manifest.get(_episode_termination.MANIFEST_KEY)):
        try:
            _episode_termination.stamp_episode_termination(
                manifest, episode_termination, overwrite=overwrite
            )
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
    "resolve_single_arm_substrate_identity",
    "pin_recording_substrate",
    "multi_arm_substrate_hashes",
    "substrate_commit",
    "stamp_recording_core",
    "missing_core_fields",
]
