"""
Arm-reuse fingerprint (Phase 0 -- INSTRUMENT ONLY).

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md

Purpose
-------
Compute a content-addressed fingerprint for a single (seed x arm) experiment
cell so that, in a LATER phase, a cell that is provably the same random variable
as a previously-computed one need not be re-run. PHASE 0 EMITS THE FINGERPRINT
ONLY -- nothing here ever skips, caches, or reuses a computation. It is pure and
side-effect-free except for reading source files to hash them.

Governing asymmetry (see plan section 2): a false cache-HIT corrupts a scientific
conclusion; a false cache-MISS only wastes compute. So the fingerprint is
deliberately OVER-inclusive -- when unsure whether something affects the cell, it
goes into the hash. Over-inclusion only causes (cheap) false misses.

What the fingerprint binds (Regime A, plan section 2.3 / 3.2)
------------------------------------------------------------
  substrate_hash : sha256 over the CONTENT of the source the cell executes
                   (ree_core/**, the env module, _harness.py, _lib/**, and the
                   calling experiment script). Content hash, NOT git SHA --
                   this workflow runs dirty trees constantly, so a commit SHA
                   would falsely match across uncommitted edits.
  config_slice   : the resolved config the cell reads. WHOLE config by default
                   (decision 3, whole-config default); callers may pass a
                   narrowed dict, recorded via config_slice_declared.
  seed           : int.
  machine_class  : platform/arch tag -- Regime A only reuses within one class.
  regime         : "A".

reuse_eligible
--------------
A cell is only a candidate for FUTURE reuse if it is a pure function of
(substrate, config_slice, seed) -- independent of iteration order and global
mutable state (plan section 2.2: the 643 "ARM_A runs first" hazard). The caller
asserts this by passing rng_fully_reset=True AFTER calling reset_all_rng() (or an
equivalent complete reset) at cell entry. If it cannot, the fingerprint is still
emitted for observability but flagged reuse_eligible=False so no future phase can
ever serve it.

ASCII-only output (repo rule). Stdlib only (importable without ree_core).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

FINGERPRINT_SCHEMA = "arm_fp/v1"
REGIME = "A"

_REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/

# Source trees whose CONTENT defines a cell's computation. Globs are relative to
# the ree-v3 repo root. Kept deliberately broad (over-inclusion -> false misses
# only). The calling script's own path is added dynamically in compute_*.
_SUBSTRATE_GLOBS: Sequence[str] = (
    "ree_core/**/*.py",
    "experiments/_harness.py",
    "experiments/_metrics.py",
    "experiments/_lib/**/*.py",
)


def _canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace jitter, ASCII-escaped.

    Non-JSON-native values (e.g. numpy scalars, tuples) are coerced to a stable
    string form so the hash is reproducible across processes.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=_json_default)


def _json_default(o: Any) -> Any:
    # tuples -> lists handled by json natively; this catches numpy/torch scalars
    # and anything else without leaking memory addresses into the hash.
    for attr in ("item", "tolist"):
        fn = getattr(o, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return f"<{type(o).__name__}:{o!r}>"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_TORCH_TAG: Optional[str] = None
TORCH_ABSENT_TAG = "torchNA"


def torch_version_tag() -> str:
    """The torch build identity that joins machine_class(). Memoised.

    Returns torch.__version__ verbatim (e.g. "2.5.1+cu121") -- INCLUDING the local
    version segment, so a CUDA-build change is a different class too (over-inclusion
    -> false misses only, per the governing asymmetry). When torch cannot be imported
    at all, returns the reserved TORCH_ABSENT_TAG, which can never collide with a real
    version string; a torchless host therefore gets its own class rather than silently
    joining a torch-bearing one.

    Lazy + memoised on purpose. Module-level `import torch` would break this module's
    stdlib-only importability, which manifest_core.py documents and depends on ("safe to
    import without torch/ree_core"). Importing on FIRST CALL keeps that property while
    still forcing a real resolution -- deliberately NOT `sys.modules.get("torch")`, which
    would make the tag depend on whether torch happened to be imported yet and so make
    the fingerprint nondeterministic across call sites.
    """
    global _TORCH_TAG
    if _TORCH_TAG is None:
        try:
            import torch as _torch  # noqa
            _TORCH_TAG = str(_torch.__version__)
        except Exception:
            _TORCH_TAG = TORCH_ABSENT_TAG
    return _TORCH_TAG


def machine_class() -> str:
    """Coarse architecture tag. Regime A only reuses within one class.

    system + machine arch + py major.minor + TORCH VERSION. Two hosts matching on all
    four are treated as one class; float-rounding determinism is assumed stable within
    a class and NOT across classes.

    WHY TORCH IS IN THE TAG (added 2026-07-19; plan sections 7b/9). Without it the tag
    was blind to the single most likely source of float-behaviour drift in this fleet.
    Upgrading torch on the cloud workers leaves python at 3.10, so the old tag stayed
    BYTE-IDENTICAL across the upgrade -- every one of the 1170 banked linux fingerprints
    would have remained matchable, and a post-upgrade consumer would have compared its
    new-torch treatment arms against old-torch baselines with no cache miss and no
    warning. That is precisely the false HIT the whole design exists to prevent (a false
    hit corrupts a conclusion; a false miss only wastes compute). Putting torch in the
    tag converts that silent corruption into a visible, cheap re-run.

    This is a HARD CUT, and deliberately so: it invalidates every fingerprint minted
    before it. No migration was possible -- the fingerprint hashes `config_slice`, which
    is persisted NOWHERE (neither the index entry nor the stored per-cell payload keeps
    it), so an old fingerprint cannot be recomputed under a new tag by any means. Old
    entries simply stop matching, which is the correct and safe failure direction.

    The same tag also keys maturation_curriculum._prefix_key (frozen prefix TENSORS on
    disk) and probe_warmup._cache_key, so both inherit the torch discrimination from
    this one change rather than each needing its own guard.
    """
    return "{sys}-{arch}-py{maj}.{minr}-torch{torch}".format(
        sys=platform.system().lower(),
        arch=(platform.machine() or "unknown").lower(),
        maj=sys.version_info.major,
        minr=sys.version_info.minor,
        torch=torch_version_tag(),
    )


def compute_substrate_hash(
    extra_paths: Optional[Iterable[Path]] = None,
    repo_root: Optional[Path] = None,
    scope: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Hash the CONTENT of every substrate source file the cell can execute.

    Returns a dict: {"substrate_hash", "n_files", "missing", "scoped", "globs"}.
    Files are sorted by repo-relative path before hashing so the result is
    order-stable. A missing expected file is recorded (not silently skipped) so a
    stripped checkout produces a visibly different hash rather than a false match.

    scope
        Dependency-scoped substrate hashing (plan section 11). None (DEFAULT) hashes
        the WHOLE `_SUBSTRATE_GLOBS` trees -- today's behaviour, byte-for-byte
        unchanged, so every existing fingerprint is unaffected and the global
        section-9 arm_fingerprint path is untouched. A non-None `scope` is an
        author-declared iterable of repo-root-relative globs naming ONLY the
        closure the cell actually executes; then only those files are hashed.

        SAFETY (plan section 2 governing asymmetry): narrowing is sound ONLY if the
        declared scope is a provable SUPERSET of every file the cell can execute --
        an under-approximation (omitting an executed file) is a false-HIT bug that
        corrupts a conclusion, whereas over-inclusion only causes a (cheap) false
        miss. This function does NOT itself prove the superset property; the caller
        MUST establish it (the execution-trace + static data-closure guards in the
        shared experiments/_lib/substrate_scope_guard.verify_scope_conservatism).
        `scoped`/`globs` are returned so the caller can record the discriminator in
        provenance.
    """
    root = (repo_root or _REPO_ROOT).resolve()
    globs: Sequence[str] = _SUBSTRATE_GLOBS if scope is None else tuple(scope)
    paths: Dict[str, Path] = {}
    for g in globs:
        for p in root.glob(g):
            if p.is_file():
                paths[str(p.relative_to(root))] = p
    for ep in extra_paths or ():
        ep = Path(ep).resolve()
        try:
            rel = str(ep.relative_to(root))
        except ValueError:
            rel = ep.name  # outside repo root: name-only key, still hashed
        if ep.is_file():
            paths[rel] = ep

    h = hashlib.sha256()
    missing: List[str] = []
    for rel in sorted(paths):
        p = paths[rel]
        try:
            content = p.read_bytes()
        except OSError:
            missing.append(rel)
            continue
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_hex(content).encode("ascii"))
        h.update(b"\n")
    for rel in missing:
        h.update(b"MISSING\0")
        h.update(rel.encode("utf-8"))
        h.update(b"\n")
    return {
        "substrate_hash": h.hexdigest(),
        "n_files": len(paths) - len(missing),
        "missing": missing,
        "scoped": scope is not None,
        "globs": list(globs),
    }


def reset_all_rng(seed: int) -> None:
    """Complete per-cell RNG reset (plan section 2.2 hardening fix).

    Resets every RNG a cell can touch so the cell is order-independent:
    Python `random`, numpy, torch (+cuda), and the harness module-level
    `_action_random`. Call this at cell entry; then pass rng_fully_reset=True to
    compute_arm_fingerprint. Imports of numpy/torch/_harness are lazy so this
    module stays importable without them.
    """
    random.seed(seed)
    try:
        import numpy as _np  # noqa
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch  # noqa
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # Reseed the harness fallback RNG (module-level, unseeded by default:
    # experiments/_harness.py:70) so a cell that falls back to it is still a
    # pure function of `seed`.
    try:
        from experiments import _harness as _h  # type: ignore
        _h._action_random.seed(seed)
    except Exception:
        try:
            import _harness as _h2  # type: ignore
            _h2._action_random.seed(seed)
        except Exception:
            pass


def compute_arm_fingerprint(
    *,
    config_slice: Mapping[str, Any],
    seed: int,
    script_path: Optional[Path] = None,
    rng_fully_reset: bool,
    config_slice_declared: bool = False,
    extra_substrate_paths: Optional[Iterable[Path]] = None,
    repo_root: Optional[Path] = None,
    extra_ineligible_reasons: Optional[Sequence[str]] = None,
    include_driver_script_in_hash: bool = True,
    substrate_scope: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute the per-cell fingerprint payload to embed in a manifest arm row.

    Parameters
    ----------
    config_slice
        The resolved config the cell reads. Pass the WHOLE config for the
        default (safe) behaviour; pass a narrowed dict + config_slice_declared=
        True only if you have deliberately verified the OFF path reads nothing
        outside it.
    seed
        The cell seed.
    script_path
        The calling experiment script. With include_driver_script_in_hash=True
        (default) its content joins the substrate hash. With False, its content
        is recorded separately (driver_script_hash) for human triage but is NOT
        part of the reuse-critical hash. Pass `Path(__file__)` from the experiment.
    include_driver_script_in_hash
        DEFAULT True -- backward-compatible: the driver script's content folds
        into substrate_hash (over-inclusion -> false misses only, the original
        conservative behaviour; an existing mint's fingerprint is unchanged).
        Set False ONLY when the cell's OFF computation is fully contained in a
        canonical baseline module under experiments/_lib/** (which is already in
        the substrate-hash glob): then the driver script that merely orchestrates
        the cell is excluded from the reuse key, so a mint and a later consumer
        with DIFFERENT driver scripts -- both constructing the OFF arm from the
        same canonical module + config_slice + seed -- produce the SAME
        fingerprint and the automated consumer (arm_reuse.try_reuse_cell) can HIT.
        This is the deliberate "my OFF cell is a pure function of the hashed
        substrate, not of my driver" assertion. False fingerprints can never
        collide with True fingerprints (a discriminator enters the hash), so the
        two modes are isolated; mint and consumer MUST agree on the flag to HIT.
    rng_fully_reset
        True iff reset_all_rng(seed) (or equivalent) ran at cell entry. False
        forces reuse_eligible=False.
    extra_ineligible_reasons
        Any caller-known reasons the cell must never be reused (e.g.
        "shared_optimizer_across_arms"). Their presence forces
        reuse_eligible=False.
    substrate_scope
        Author-declared dependency scope (plan section 11). DEFAULT None hashes the
        WHOLE _SUBSTRATE_GLOBS trees (today's behaviour, byte-for-byte unchanged, so
        every existing fingerprint is unaffected -- this stays strictly opt-in and
        false-miss-only). A non-None value is an iterable of repo-root-relative globs
        naming ONLY the closure the cell actually executes+reads; then only those files
        join substrate_hash, so an edit to an unrelated ree_core module no longer busts
        the fingerprint and a later consumer that declares the SAME scope can HIT where
        it previously missed.

        SAFETY (plan section 2 governing asymmetry -- the one thing that must not be
        wrong): a declared scope MUST be a provable SUPERSET of every file the cell can
        execute or read a module-level constant from. An UNDER-approximation is a
        false-HIT bug that corrupts a conclusion; over-inclusion only causes a (cheap)
        false miss. This function does NOT prove the superset property -- the caller MUST
        establish it BEFORE trusting a scope, via the shared guards in
        experiments/_lib/substrate_scope_guard.verify_scope_conservatism (guard 1
        call-trace + guard 2 static data-closure), exactly as the maturation-curriculum
        prototype did. The declared scope is folded into the fingerprint hash (so
        narrowing/widening the reuse contract changes the key, and a scoped fingerprint
        can never collide with a whole-tree one) and recorded via substrate_scope_declared
        for audit -- precisely like config_slice_declared. Optionally, exporting
        REE_ARM_SCOPE_GUARD=1 runs the cheap static guard 2 here at emit time as a
        tripwire (the expensive call-trace guard 1 must run in the author's smoke/contract
        test since it executes a real cell).

    Returns a JSON-serialisable dict. Embed it as arm_results[i]["arm_fingerprint"].
    NOTHING here reads or consults any cache -- Phase 0 is emit-only.
    """
    scoped = substrate_scope is not None
    if scoped and os.environ.get("REE_ARM_SCOPE_GUARD", "").strip() not in ("", "0", "false", "False"):
        # Opt-in cheap static conservatism guard (guard 2). Lazily imported so the
        # default path stays stdlib-only and dependency-free; the guard module is itself
        # stdlib-only. Raises loudly if the declared scope is no longer data-closed or a
        # declared file is missing. The expensive call-trace guard (guard 1) must run in
        # the author's smoke/contract test (it executes a real cell). The _lib dir is put
        # on sys.path so the bare import resolves regardless of how the caller imported us.
        _libdir = str(Path(__file__).resolve().parent)
        if _libdir not in sys.path:
            sys.path.insert(0, _libdir)
        from substrate_scope_guard import verify_scope_static as _vss  # noqa: E402
        _vss(list(substrate_scope), label="arm_fingerprint substrate_scope")

    # Driver script folds into the reuse-critical hash only when the caller opts
    # IN (the default). When excluded, record its content separately for triage --
    # the canonical baseline module that actually defines the OFF computation is
    # already captured by the experiments/_lib/** glob in compute_substrate_hash.
    fold_script = bool(script_path) and include_driver_script_in_hash
    sub = compute_substrate_hash(
        extra_paths=[script_path] if fold_script else None,
        repo_root=repo_root,
        scope=substrate_scope,
    )
    if extra_substrate_paths:
        # fold any additional declared deps into the hash deterministically
        extra = compute_substrate_hash(extra_paths=extra_substrate_paths,
                                       repo_root=repo_root,
                                       scope=substrate_scope)
        sub["substrate_hash"] = _sha256_hex(
            (sub["substrate_hash"] + ":" + extra["substrate_hash"]).encode("ascii")
        )

    driver_script_hash: Optional[str] = None
    if script_path:
        try:
            driver_script_hash = _sha256_hex(Path(script_path).read_bytes())
        except OSError:
            driver_script_hash = None

    mc = machine_class()
    fp_input = {
        "schema": FINGERPRINT_SCHEMA,
        "substrate_hash": sub["substrate_hash"],
        "config_slice": dict(config_slice),
        "seed": int(seed),
        "machine_class": mc,
        "regime": REGIME,
    }
    # Discriminator: a driver-script-excluded fingerprint must NEVER collide with a
    # driver-script-included one for the same cell. Added only on the non-default
    # path so existing (include=True) fingerprints are byte-identical to before.
    if not include_driver_script_in_hash:
        fp_input["driver_script_excluded"] = True
    # Discriminator: a substrate-scoped fingerprint must NEVER collide with a whole-tree
    # (scope=None) one, and two DIFFERENT declared scopes must key differently (narrowing/
    # widening the reuse contract changes the key -- mirrors config_slice + the maturation
    # _prefix_key). Folded in ONLY on the non-default path, so an unscoped fingerprint is
    # byte-identical to before this feature.
    if scoped:
        fp_input["substrate_scope_declared"] = True
        fp_input["substrate_scope"] = list(substrate_scope)
    fingerprint = _sha256_hex(_canonical_json(fp_input).encode("utf-8"))

    reasons: List[str] = list(extra_ineligible_reasons or ())
    if not rng_fully_reset:
        reasons.append("incomplete_rng_reset")
    if sub["missing"]:
        reasons.append("substrate_files_missing")
    reuse_eligible = len(reasons) == 0

    return {
        "schema": FINGERPRINT_SCHEMA,
        "arm_fingerprint": fingerprint,
        "substrate_hash": sub["substrate_hash"],
        "substrate_n_files": sub["n_files"],
        "machine_class": mc,
        # Recorded SEPARATELY as well as inside machine_class, so a future miss can be
        # triaged ("missed because torch moved") instead of being an unexplained miss,
        # and so a later tag change has the data this one did not: the pre-2026-07-19
        # corpus records no torch anywhere, which is exactly why that cut could not be
        # migrated. Observability only -- machine_class is what enters the hash.
        "torch_version": torch_version_tag(),
        "regime": REGIME,
        "seed": int(seed),
        "config_slice_declared": bool(config_slice_declared),
        # Dependency-scoped substrate hashing (plan section 11). Recorded for audit like
        # config_slice_declared; the glob list lets a reviewer / consumer confirm both
        # sides of a reuse used the same declared scope. substrate_scope is None when
        # undeclared (whole-tree hash -- the safe default).
        "substrate_scope_declared": scoped,
        "substrate_scope": list(substrate_scope) if scoped else None,
        "reuse_eligible": reuse_eligible,
        "reuse_ineligible_reasons": reasons,
        # Reuse-key scoping: did the driver script join the reuse-critical hash?
        # Recorded so the index/consumer + a human reviewer can confirm both sides
        # of a reuse used the same convention. driver_script_hash is observability
        # only (never in the fingerprint when excluded) -- analogous to the git SHA
        # recorded alongside the authoritative content hash (plan section 3.2).
        "driver_script_in_substrate_hash": bool(include_driver_script_in_hash),
        "driver_script_hash": driver_script_hash,
        # NOTE: Phase 0 is emit-only. No reuse is ever performed off this value.
    }


class _ArmCell:
    """Context manager bundling the two per-cell fingerprint obligations.

    On `__enter__` it performs the complete per-cell RNG reset (`reset_all_rng`),
    so the cell is order-independent (plan section 2.2). `.stamp(row)` computes
    `compute_arm_fingerprint(...)` and writes it onto `row["arm_fingerprint"]`,
    returning the payload. This collapses the 4-line emit boilerplate to:

        with arm_cell(seed, config_slice=cfg, script_path=Path(__file__)) as cell:
            row = run_one_cell(seed)
            cell.stamp(row)

    `rng_fully_reset=True` is recorded automatically because `__enter__` did the
    reset. If you must skip the reset (you should not, for a reusable cell),
    construct with `do_reset=False` and the fingerprint is flagged
    `reuse_eligible=False`. Extra ineligibility reasons (shared optimiser/buffer
    across arms, etc.) pass through to keep the cell out of any future cache.
    """

    def __init__(
        self,
        seed: int,
        *,
        config_slice: Mapping[str, Any],
        script_path: Optional[Path] = None,
        config_slice_declared: bool = False,
        extra_substrate_paths: Optional[Iterable[Path]] = None,
        repo_root: Optional[Path] = None,
        extra_ineligible_reasons: Optional[Sequence[str]] = None,
        do_reset: bool = True,
        include_driver_script_in_hash: bool = True,
        substrate_scope: Optional[Sequence[str]] = None,
    ) -> None:
        self.seed = int(seed)
        self._config_slice = config_slice
        self._script_path = script_path
        self._config_slice_declared = config_slice_declared
        self._extra_substrate_paths = extra_substrate_paths
        self._repo_root = repo_root
        self._extra_ineligible_reasons = extra_ineligible_reasons
        self._do_reset = do_reset
        self._include_driver_script_in_hash = include_driver_script_in_hash
        self._substrate_scope = substrate_scope
        self._rng_reset = False
        self.fingerprint: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "_ArmCell":
        if self._do_reset:
            reset_all_rng(self.seed)
            self._rng_reset = True
        return self

    def stamp(self, row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute the fingerprint; if `row` is given, set row['arm_fingerprint']."""
        self.fingerprint = compute_arm_fingerprint(
            config_slice=self._config_slice,
            seed=self.seed,
            script_path=self._script_path,
            rng_fully_reset=self._rng_reset,
            config_slice_declared=self._config_slice_declared,
            extra_substrate_paths=self._extra_substrate_paths,
            repo_root=self._repo_root,
            extra_ineligible_reasons=self._extra_ineligible_reasons,
            include_driver_script_in_hash=self._include_driver_script_in_hash,
            substrate_scope=self._substrate_scope,
        )
        if row is not None:
            row["arm_fingerprint"] = self.fingerprint
        return self.fingerprint

    def __exit__(self, *exc: Any) -> bool:
        return False  # never suppress exceptions


def arm_cell(
    seed: int,
    *,
    config_slice: Mapping[str, Any],
    script_path: Optional[Path] = None,
    config_slice_declared: bool = False,
    extra_substrate_paths: Optional[Iterable[Path]] = None,
    repo_root: Optional[Path] = None,
    extra_ineligible_reasons: Optional[Sequence[str]] = None,
    do_reset: bool = True,
    include_driver_script_in_hash: bool = True,
    substrate_scope: Optional[Sequence[str]] = None,
) -> _ArmCell:
    """One-liner per-cell helper: resets RNG on enter, stamps fingerprint on .stamp().

    See `_ArmCell` for the usage pattern. This is the recommended path for new
    multi-arm experiments -- it discharges BOTH per-cell obligations (complete RNG
    reset + fingerprint emission) so an author cannot accidentally do one without
    the other. The low-level `reset_all_rng` + `compute_arm_fingerprint` pair
    remains available for scripts that need finer control.

    `substrate_scope` (plan section 11): pass an author-declared dependency scope to
    hash only the cell's closure instead of the whole substrate tree. DEFAULT None =
    whole-tree (byte-unchanged). It MUST be a provable superset of the cell's
    execute+read closure -- verify with substrate_scope_guard.verify_scope_conservatism
    first (see compute_arm_fingerprint).
    """
    return _ArmCell(
        seed,
        config_slice=config_slice,
        script_path=script_path,
        config_slice_declared=config_slice_declared,
        extra_substrate_paths=extra_substrate_paths,
        repo_root=repo_root,
        extra_ineligible_reasons=extra_ineligible_reasons,
        do_reset=do_reset,
        include_driver_script_in_hash=include_driver_script_in_hash,
        substrate_scope=substrate_scope,
    )


__all__ = [
    "FINGERPRINT_SCHEMA",
    "REGIME",
    "machine_class",
    "compute_substrate_hash",
    "reset_all_rng",
    "compute_arm_fingerprint",
    "arm_cell",
]
