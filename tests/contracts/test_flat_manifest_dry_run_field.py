"""Contract: a FLAT dry-run manifest must self-identify with top-level ``dry_run: true``.

GFLAG-0244 (2026-09-09), the FIFTH instance of the GFLAG-0117/0120 family. The
defect: ``write_flat_manifest`` applied the ``_dry_<run_id>.json`` FILENAME marker
but never wrote the ``dry_run`` FIELD, so a smoke landed in
``REE_assembly/evidence/experiments/`` looking, to every field-keyed consumer, like
a real scored run.

WHY THE FIELD AND NOT THE FILENAME. The two independent nets that exclude a smoke
from evidence both key on the top-level field, and therefore both miss the same
files together:

  * ``generate_pending_review.load_dry_run_run_ids()`` -- so the smoke surfaces in
    ``pending_review.md`` as an unclaimed manifest, inviting a mark-discussed;
  * the GOV-DRY-1 sweep (``REE_assembly/scripts/check_dry_run_adjudication_leak.py``,
    governance.sh Step 3i) -- so its "dry manifests carrying an ASSERTING stamp"
    count reads 0 while an asserting stamp is sitting right there.

A filename heuristic in the consumers is NOT the fix, and the corpus is the
evidence: it holds a ``_dry_``-PREFIXED convention (V3-EXQ-918a, this writer's own
marker), a ``_dry``-SUFFIXED convention baked into the run_id by older drivers
(V3-EXQ-259/318-331/353/431, and the still-live V3-EXQ-324d), and 78 manifests
carrying ``dry_run: true`` with no filename marker at all. The field is the only
carrier that spans all three, which is precisely why it has to be written at the
source.

SIBLINGS, and where this one sits. ``test_write_pack_dry_run_lint.py`` watches the
PACK writer (which gained its own self-identification on 2026-07-28 and is what the
indexer actually scores); ``test_hardcoded_dry_run_lint.py`` and
``test_emit_outcome_dry_run_lint.py`` watch the DRIVER-side threading of
``args.dry_run``. Those three are LINTS over driver source. This one is a
BEHAVIOURAL contract on the writer itself: given ``dry_run=True``, what lands on
disk. The distinction matters -- V3-EXQ-918a and V3-EXQ-324d both thread the flag
correctly (so every lint above is silent) and still produced an unflagged manifest,
because the gap was in the writer, not in them.

NOT A MANDATORY_CORE KEY, deliberately. ``manifest_core.MANDATORY_CORE_KEYS``
asserts PRESENCE on every manifest, and ``dry_run`` must be ABSENT on a real run --
``write_pack``'s "conditional add on a truthy value" posture, which keeps every
non-dry manifest byte-identical to what it was before the flag existed. The right
home is the writer's existing dry-run branch, where dry-ness is already known from
the caller's kwarg and already acted on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# conftest puts ree-v3 root on sys.path -> `experiments.*` importable.
from experiments import pack_writer as pw

_REE_V3_ROOT = Path(__file__).resolve().parents[2]


def _manifest(run_id: str = "v3_exq_test_dry_field_20260909T000000Z_v3") -> dict:
    """A minimal flat manifest with the mandatory-core fields pre-set, so the test
    can use ``stamp=False`` and stay free of a real git/subprocess dependency (the
    same shape ``test_pack_writer_worktree_resolution`` uses)."""
    return {
        "run_id": run_id,
        "outcome": "PASS",
        "evidence_direction": "supports",
        "recording_schema": "rec/v1",
        "substrate_hash": "0" * 64,
        "substrate_commit": {"commit": "0" * 40, "dirty": False},
        "machine": "test-host",
        "machine_class": "test-class",
    }


# ---- the defect, pinned ------------------------------------------------------

def test_dry_run_write_sets_top_level_field_on_disk(tmp_path):
    """THE REGRESSION. `dry_run=True` must put `dry_run: true` in the WRITTEN file,
    not merely in the filename. Read the artifact back off disk rather than the
    in-memory dict -- the field has to survive the serialisation, and the on-disk
    copy is the only thing the consumers ever see."""
    import json

    out = pw.write_flat_manifest(
        _manifest(), tmp_path, dry_run=True, stamp=False,
    )
    doc = json.loads(out.read_text())
    assert doc.get("dry_run") is True, (
        "flat dry-run manifest landed WITHOUT top-level dry_run -- "
        "generate_pending_review and GOV-DRY-1 are both blind to it (GFLAG-0244)"
    )


def test_dry_run_still_applies_the_filename_marker(tmp_path):
    """The field is ADDITIONAL to the `_dry_` prefix, not a replacement for it --
    the prefix is what keeps a smoke visually separable in the evidence dir and is
    load-bearing for anyone reading the directory by eye."""
    out = pw.write_flat_manifest(
        _manifest(), tmp_path, dry_run=True, stamp=False,
    )
    assert out.name.startswith("_dry_")


def test_real_run_does_not_gain_the_key(tmp_path):
    """A non-dry run must stay byte-identical to its pre-fix shape: the key is
    ADDED on a truthy value, never written as `dry_run: false`. Asserting absence
    (not falsity) is the point -- `write_pack` holds the same posture, and a
    `dry_run: false` on 4000+ real manifests would be pure diff noise."""
    import json

    out = pw.write_flat_manifest(
        _manifest(), tmp_path, dry_run=False, stamp=False,
    )
    doc = json.loads(out.read_text())
    assert "dry_run" not in doc
    assert not out.name.startswith("_dry_")


def test_explicit_caller_value_is_not_clobbered(tmp_path):
    """A driver that already sets the field itself (V3-EXQ-365/608/632 do) keeps its
    own value. The writer fills a GAP; it does not overrule an author who was
    explicit. Guards against a future 'just always assign True' simplification that
    would silently rewrite a deliberate `dry_run: false` on a `_dry_`-named
    diagnostic."""
    import json

    m = _manifest()
    m["dry_run"] = False
    out = pw.write_flat_manifest(m, tmp_path, dry_run=True, stamp=False)
    doc = json.loads(out.read_text())
    assert doc["dry_run"] is False


# ---- pin the live module -----------------------------------------------------

def test_live_writer_carries_the_fix():
    """Pin that the fix is in the COMMITTED file, not only in whatever `experiments`
    happened to import (sys.path can reach a different checkout -- e.g. an rsync'd
    staging tree; the same hazard the writer's own stamp_fn fallback guards)."""
    src = (_REE_V3_ROOT / "experiments" / "pack_writer.py").read_text()
    assert 'manifest["dry_run"] = True' in src, (
        "write_flat_manifest no longer sets the dry_run field -- GFLAG-0244 regression"
    )
    assert 'manifest_doc["dry_run"] = True' in src, (
        "write_pack no longer sets the dry_run field -- 2026-07-28 regression"
    )


def test_both_writers_mark_dry_runs():
    """Cross-writer symmetry, asserted rather than trusted. The 2026-07-28 repair
    gave the PACK writer self-identification and left the FLAT writer behind for
    fourteen months of corpus; the whole GFLAG-0117/0120/0244 family is that
    asymmetry. Both writers take the kwarg -- keep it that way."""
    import inspect

    flat_sig = inspect.signature(pw.write_flat_manifest)
    assert "dry_run" in flat_sig.parameters
    pack_sig = inspect.signature(pw.ExperimentPackWriter.write_pack)
    assert "dry_run" in pack_sig.parameters
