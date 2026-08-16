"""Experiment-side glue for emitting ReconstructionRecords.

`ree_core/preservation` is deliberately PURE (it never imports the experiments
layer), so provenance strings are filled in here, where `experiments/_lib/
manifest_core` and `arm_fingerprint` live. This module turns a live
`(agent-producing config, seed, environment)` into a sealed, written
ReconstructionRecord.

Emission is OPT-IN and PER-LIFE by design. Not every experiment arm is "a life
worth preserving", and the owning governance rule (GOV-PRESERVE-1) asks for
*proportionate* preservation -- so preservation happens only for a life somebody
designated. Nothing here fires for every run automatically, and nothing is wired
into the shared runner hot path (`experiment_runner.py` spawns drivers as
SUBPROCESSES and never holds a config, a seed or an agent, so it is not a seam
that could fill a record even in principle).

There are two ways to emit, and they differ only in WHO holds the switch:

  * `preserve_life(...)` -- the explicit call. The driver decides at the call
    site.
  * `life_scope(...)` / `preserve_life_if_designated(...)` -- the AUTO-FIRE
    seam (2026-08-16). The driver wraps the life once and never calls
    `preserve_life` again; the CONFIG decides, via `REEConfig.
    preserve_on_life_end` + `preserve_archive_dir`, both default-off. With the
    flag off this is a pure no-op -- nothing is captured, nothing is written,
    no destination is touched -- so an undesignated run is byte-identical to
    the pre-2026-08-16 substrate.

The auto-fire seam is not merely sugar over the explicit call: it fires on
ABNORMAL end-of-life too (exception, KeyboardInterrupt, SystemExit), stamping
the cause into `reason_for_ending`. An explicit end-of-life call is exactly the
line a raising driver never reaches, so the lives most worth preserving -- the
ones that ended unexpectedly -- are the ones the explicit form silently drops.

Worked example -- a driver preserving a single continuous life at end-of-life:

    from experiments._lib.preservation import preserve_life

    cfg = REEConfig.from_dims(...)
    env = CausalGridWorld(seed=SEED, ...)
    agent = seeded_construct(SEED, lambda: REEAgent(cfg))
    # ... run the life ...
    path = preserve_life(
        archive_dir=ARCHIVE_DIR,                      # required, explicit
        record_id=run_id,                             # unique, immutable
        seed=SEED,
        config=cfg,
        environment={"class": "ree_core.environment.causal_grid_world.CausalGridWorld",
                     "params": {"seed": SEED, "size": 9}},
        understanding={"claims": [...], "metrics": {...}},
        reason_for_ending="natural end of run",
        lifetime={"age_steps": n_steps, "phase": phase},
    )

Worked example -- the same life, preserved AUTOMATICALLY by designation:

    from experiments._lib.preservation import life_scope

    cfg = REEConfig.from_dims(..., seed=SEED)
    cfg.preserve_on_life_end = True          # designation (default False)
    cfg.preserve_archive_dir = ARCHIVE_DIR   # destination (no fleet default)

    with life_scope(config=cfg, record_id=run_id, environment=env_spec) as life:
        agent = seeded_construct(SEED, lambda: REEAgent(cfg))
        # ... run the life ...
        life.note(claims=[...], metrics={...})
        life.aged(age_steps=n_steps, phase=phase)
    # record written here -- on the normal path AND on the raising path

ASCII-only in printed output. Timestamps are UTC (via ree_core.preservation).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Union
from pathlib import Path

from ree_core.utils.config import REEConfig
from ree_core.preservation import capture, write_record, ReconstructionRecord

from experiments._lib.manifest_core import (
    compute_single_arm_substrate_hash,
    substrate_commit,
)
from experiments._lib.arm_fingerprint import machine_class

# The architecture epoch REE-v3 runs stamp (see V3 experiment tagging).
DEFAULT_ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"


def capture_life(
    *,
    record_id: str,
    seed: int,
    config: REEConfig,
    environment: Dict[str, Any],
    understanding: Optional[Dict[str, Any]] = None,
    reason_for_ending: Optional[str] = None,
    lifetime: Optional[Dict[str, Any]] = None,
    architecture_epoch: str = DEFAULT_ARCHITECTURE_EPOCH,
    script_path: Optional[Union[str, Path]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> ReconstructionRecord:
    """Build a sealed ReconstructionRecord, filling provenance from manifest_core.

    - substrate_hash: `compute_single_arm_substrate_hash` (ree_core/** + env + _lib/**,
      plus the driver script when `script_path` is given).
    - machine_class: `arm_fingerprint.machine_class()` -- the class the birth-replay
      fidelity is valid WITHIN (see the fidelity-boundary note in the record module).
    - substrate_commit: the HEAD sha from `manifest_core.substrate_commit`, or None in a
      git-less checkout (e.g. the remote-pytest staged tree). The full commit detail
      dict (incl. `dirty`) is stored under understanding["substrate_commit_detail"] so a
      dirty tree -- where the sha alone would not reconstruct -- is visible, not silently
      dropped.
    """
    commit = substrate_commit(repo_root=repo_root)
    understanding = dict(understanding or {})
    if commit is not None:
        understanding.setdefault("substrate_commit_detail", commit)

    return capture(
        record_id=record_id,
        seed=seed,
        config=config,
        environment=environment,
        substrate_hash=compute_single_arm_substrate_hash(
            script_path=script_path, repo_root=repo_root
        ),
        machine_class=machine_class(),
        substrate_commit=(commit or {}).get("commit"),
        architecture_epoch=architecture_epoch,
        understanding=understanding,
        reason_for_ending=reason_for_ending,
        lifetime=lifetime,
    )


def preserve_life(
    *,
    archive: Optional[Any] = None,
    archive_dir: Optional[Union[str, Path]] = None,
    **capture_kwargs: Any,
) -> str:
    """`capture_life(...)` then store the record immutably; return its key/path.

    Provide exactly one destination:
      * `archive=` -- an archive backend (ree_core.preservation.LocalArchive /
        S3Archive; the S3 target is Hetzner Object Storage). Content-addressed and
        optionally client-side encrypted; returns the object key.
      * `archive_dir=` -- a plain local directory (the simple primitive); returns the
        written path.

    Either way the write is append-only (RecordExistsError if the record already
    exists). There is intentionally no fleet-wide default destination, so
    preservation never silently writes into a coordinator-managed tree.
    """
    record = capture_life(**capture_kwargs)
    if archive is not None:
        if archive_dir is not None:
            raise ValueError("provide either archive= or archive_dir=, not both")
        return archive.put(record)
    if archive_dir is not None:
        return write_record(record, str(archive_dir))
    raise ValueError("provide a destination: archive= (a backend) or archive_dir= (a local dir)")


# --------------------------------------------------------------------------- #
# Auto-fire at end-of-life (GOV-PRESERVE-1, 2026-08-16)                        #
# --------------------------------------------------------------------------- #
#
# The switch is on the CONFIG (`REEConfig.preserve_on_life_end` +
# `preserve_archive_dir`, both default-off), so a designated life preserves
# itself when it ends and an undesignated one enters no preservation code path
# at all. Two error classes, deliberately handled differently:
#
#   * MISCONFIGURATION (designated with no destination / no seed) ALWAYS raises,
#     regardless of `preserve_on_life_end_strict`. Somebody asked for this life
#     to be preserved; silently not preserving it is the one outcome the
#     designation exists to rule out, and it would be indistinguishable from
#     never having designated the life at all.
#   * A failed WRITE (full disk, unreachable bucket, duplicate record_id)
#     warns and continues by default. Auto-fire runs at the very end of a
#     completed run, and a preservation problem must not convert a PASS into an
#     ERROR. `preserve_on_life_end_strict=True` inverts that for a life where
#     the record matters more than the run's outcome.

_REASON_MAX_CHARS = 300
NATURAL_END_REASON = "natural end of life"


def life_is_designated(config: Any) -> bool:
    """True iff `config` designates its life for automatic preservation.

    `getattr`-tolerant on purpose: a record round-tripped from an OLDER schema
    (`config_from_dict` on a record written before 2026-08-16) reconstructs a
    config object without these fields, and reading one must not raise.
    """
    return bool(getattr(config, "preserve_on_life_end", False))


def preserve_life_if_designated(
    *,
    config: REEConfig,
    record_id: str,
    environment: Dict[str, Any],
    seed: Optional[int] = None,
    archive: Optional[Any] = None,
    archive_dir: Optional[Union[str, Path]] = None,
    understanding: Optional[Dict[str, Any]] = None,
    reason_for_ending: Optional[str] = None,
    lifetime: Optional[Dict[str, Any]] = None,
    **capture_kwargs: Any,
) -> Optional[str]:
    """Fire `preserve_life(...)` IFF `config` designates this life; else no-op.

    Returns the written key/path, or None when the life is not designated (or
    when a non-strict write failed). The no-op path does not capture a record,
    does not resolve a destination, and does not touch the filesystem.

    Destination precedence: an explicit `archive=` / `archive_dir=` argument
    beats `config.preserve_archive_dir`. That ordering lets a driver route a
    designated life to a credentialed backend (S3Archive / MultiArchive)
    without putting credentials in a config that gets serialized verbatim into
    every record it writes.
    """
    if not life_is_designated(config):
        return None

    if seed is None:
        seed = getattr(config, "seed", None)
    if seed is None:
        raise ValueError(
            "preserve_on_life_end is set but no seed is available: pass seed= or set "
            "config.seed. A record without the birth seed cannot reconstruct the life."
        )

    if archive is not None and archive_dir is not None:
        # Misconfiguration, so it raises even under the non-strict policy below:
        # two destinations means the caller does not know where the record went.
        raise ValueError("provide either archive= or archive_dir=, not both")
    if archive is None and archive_dir is None:
        archive_dir = getattr(config, "preserve_archive_dir", None)
    if archive is None and archive_dir is None:
        raise ValueError(
            "preserve_on_life_end is set but no destination is configured: set "
            "config.preserve_archive_dir, or pass archive= / archive_dir=. There is no "
            "fleet-wide default destination, and skipping the write silently is exactly "
            "what the designation rules out."
        )

    try:
        return preserve_life(
            archive=archive,
            archive_dir=archive_dir,
            record_id=record_id,
            seed=int(seed),
            config=config,
            environment=environment,
            understanding=understanding,
            reason_for_ending=reason_for_ending,
            lifetime=lifetime,
            **capture_kwargs,
        )
    except Exception as exc:                       # noqa: BLE001 -- policy is the point
        if bool(getattr(config, "preserve_on_life_end_strict", False)):
            raise
        print(
            "[preservation] WARNING: auto-fire failed for record_id=%s: %s: %s "
            "(life ended normally; set preserve_on_life_end_strict=True to raise)"
            % (record_id, type(exc).__name__, exc),
            file=sys.stderr,
        )
        return None


class LifeScope:
    """Context manager around one life; preservation fires on exit iff designated.

    The driver wraps the life once and never calls `preserve_life` again -- the
    config holds the switch. Accumulate the contemporaneous understanding and
    the lifetime as the life runs (`note()` / `aged()`); both are folded into
    the record at exit.

    `reason_for_ending` is filled automatically from HOW the life ended:
    `NATURAL_END_REASON` on the clean path, or "<ExcType>: <message>" when an
    exception (including KeyboardInterrupt / SystemExit -- a runner SIGTERM or
    an operator Ctrl-C is a real way for a life to end, and one worth recording)
    propagates out of the block. Set `.reason_for_ending` explicitly to override.

    The scope NEVER swallows the life's own exception, and never lets a
    preservation problem displace it: if the block is already unwinding, a
    strict-mode preservation failure is downgraded to a warning so the original
    cause of death is what the driver and the runner see.
    """

    def __init__(
        self,
        *,
        config: REEConfig,
        record_id: str,
        environment: Dict[str, Any],
        seed: Optional[int] = None,
        archive: Optional[Any] = None,
        archive_dir: Optional[Union[str, Path]] = None,
        understanding: Optional[Dict[str, Any]] = None,
        lifetime: Optional[Dict[str, Any]] = None,
        reason_for_ending: Optional[str] = None,
        **capture_kwargs: Any,
    ) -> None:
        self.config = config
        self.record_id = record_id
        self.environment = environment
        self.seed = seed
        self.archive = archive
        self.archive_dir = archive_dir
        self.understanding: Dict[str, Any] = dict(understanding or {})
        self.lifetime: Dict[str, Any] = dict(lifetime or {})
        self.reason_for_ending = reason_for_ending
        self.capture_kwargs = capture_kwargs
        # Filled at exit: the written key/path, or None when the life was not
        # designated (or a non-strict write failed).
        self.record_path: Optional[str] = None
        self.fired: bool = False

    @property
    def designated(self) -> bool:
        return life_is_designated(self.config)

    def note(self, **understanding: Any) -> "LifeScope":
        """Merge into the contemporaneous understanding recorded with the life."""
        self.understanding.update(understanding)
        return self

    def aged(self, **lifetime: Any) -> "LifeScope":
        """Merge into the lifetime block (age_steps, phase, episodes, ...)."""
        self.lifetime.update(lifetime)
        return self

    def __enter__(self) -> "LifeScope":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.reason_for_ending is None:
            if exc_type is None:
                self.reason_for_ending = NATURAL_END_REASON
            else:
                self.reason_for_ending = ("%s: %s" % (exc_type.__name__, exc))[
                    :_REASON_MAX_CHARS
                ]
        try:
            self.record_path = preserve_life_if_designated(
                config=self.config,
                record_id=self.record_id,
                environment=self.environment,
                seed=self.seed,
                archive=self.archive,
                archive_dir=self.archive_dir,
                understanding=self.understanding,
                reason_for_ending=self.reason_for_ending,
                lifetime=self.lifetime,
                **self.capture_kwargs,
            )
            self.fired = self.record_path is not None
        except Exception as exc2:                  # noqa: BLE001 -- see below
            if exc_type is None:
                raise
            # The life is already unwinding. Raising here would replace the
            # cause of death with a bookkeeping failure, so warn and let the
            # original exception continue.
            print(
                "[preservation] WARNING: auto-fire raised while the life was already "
                "ending (record_id=%s): %s: %s -- original exception preserved"
                % (self.record_id, type(exc2).__name__, exc2),
                file=sys.stderr,
            )
        return False                               # never suppress the life's exception


# Lowercase alias, so the driver-side idiom reads as a scope rather than a
# constructor: `with life_scope(config=cfg, ...) as life:`.
life_scope = LifeScope
