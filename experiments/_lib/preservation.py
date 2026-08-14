"""Experiment-side glue for emitting ReconstructionRecords.

`ree_core/preservation` is deliberately PURE (it never imports the experiments
layer), so provenance strings are filled in here, where `experiments/_lib/
manifest_core` and `arm_fingerprint` live. This module turns a live
`(agent-producing config, seed, environment)` into a sealed, written
ReconstructionRecord.

Emission is OPT-IN and PER-LIFE by design. Not every experiment arm is "a life
worth preserving", and the owning governance rule (GOV-PRESERVE-1) asks for
*proportionate* preservation -- so a driver calls `preserve_life(...)`
deliberately, at the point a life it cares about ends. Nothing here fires for
every run automatically, and nothing is wired into the shared runner hot path
(that would change fleet behaviour; see the plan doc for the deferred,
default-off runner-integration step).

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

ASCII-only in printed output. Timestamps are UTC (via ree_core.preservation).
"""

from __future__ import annotations

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
