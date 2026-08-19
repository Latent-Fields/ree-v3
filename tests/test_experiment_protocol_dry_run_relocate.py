"""
Regression tests for dry-run companion-file relocation in experiment_protocol.py.

Motivated 2026-08-11: several fishtank-feed experiment drivers write a
`*_episode_log.json` sidecar directly to evidence/experiments/ *before* calling
emit_outcome(dry_run=...). _relocate_dry_run_manifest only ever moved the ONE
path it was given (the main manifest), so every --dry-run smoke of those
drivers left its episode_log sidecar behind permanently in the real evidence
tree. Confirmed twice: once caught live during a v3_exq_483 smoke test, and
once earlier misdiagnosed as "2 crashed runs with no manifest" for
v3_exq_913 (REE_assembly 0a71abc9d9, walked back by 468606c50a) before being
correctly identified as ordinary --dry-run leftovers.

Two driver shapes exist and both must be covered:
  (a) direct-write drivers (e.g. v3_exq_483, v3_exq_524, v3_exq_223): the
      manifest and its episode_log sidecar land in the SAME per-experiment-type
      directory, and no companion_files field is declared -- relies on
      auto-discovery of `*_episode_log.json` next to the manifest.
  (b) write_flat_manifest drivers (e.g. v3_exq_913, v3_exq_906, v3_exq_495):
      the manifest lands FLAT in evidence/experiments/ (the parent of the
      per-type directory) while the episode_log sits in the per-type
      subdirectory, and the manifest declares
      `companion_files: ["<experiment_type>/<name>_episode_log.json"]`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment_protocol as ep


def test_direct_write_style_relocates_sidecar_next_to_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ep.tempfile, "gettempdir", lambda: str(tmp_path / "systmp"))

    evidence_dir = tmp_path / "evidence_experiments" / "v3_exq_999_fake_fishtank"
    evidence_dir.mkdir(parents=True)
    manifest_path = evidence_dir / "v3_exq_999_fake_fishtank_20260811T000000Z.json"
    manifest_path.write_text(json.dumps({"run_id": "v3_exq_999_fake_fishtank_20260811T000000Z_v3"}))
    episode_log = evidence_dir / "v3_exq_999_fake_fishtank_20260811T000000Z_episode_log.json"
    episode_log.write_text(json.dumps({"episodes": []}))

    dest = ep._relocate_dry_run_manifest(manifest_path)

    assert not manifest_path.exists()
    assert not episode_log.exists()
    assert dest.exists()
    assert dest.parent == tmp_path / "systmp" / ep.DRY_RUN_SCRATCH_DIRNAME
    assert (dest.parent / episode_log.name).exists()


def test_unrelated_sibling_episode_log_is_never_touched(tmp_path, monkeypatch):
    """Regression for the bug caught live 2026-08-11: a directory-wide
    `*_episode_log.json` glob swept up a PRIOR REAL run's episode_log sitting
    in the same per-experiment-type directory and deleted it from evidence/
    while relocating an unrelated dry-run manifest. Auto-discovery must match
    only THIS manifest's own expected sidecar name.
    """
    monkeypatch.setattr(ep.tempfile, "gettempdir", lambda: str(tmp_path / "systmp"))

    evidence_dir = tmp_path / "evidence_experiments" / "v3_exq_483_fake"
    evidence_dir.mkdir(parents=True)

    # A real, unrelated prior run's manifest + sidecar -- must survive untouched.
    real_manifest = evidence_dir / "v3_exq_483_fake_20260425T204829Z.json"
    real_manifest.write_text(json.dumps({"run_id": "v3_exq_483_fake_20260425T204829Z_v3"}))
    real_episode_log = evidence_dir / "v3_exq_483_fake_20260425T204829Z_episode_log.json"
    real_episode_log.write_text(json.dumps({"episodes": ["real evidence"]}))

    # Today's dry-run smoke, same directory, its own sidecar.
    dry_manifest = evidence_dir / "v3_exq_483_fake_20260811T165741Z.json"
    dry_manifest.write_text(json.dumps({"run_id": "v3_exq_483_fake_20260811T165741Z_v3"}))
    dry_episode_log = evidence_dir / "v3_exq_483_fake_20260811T165741Z_episode_log.json"
    dry_episode_log.write_text(json.dumps({"episodes": ["smoke"]}))

    dest = ep._relocate_dry_run_manifest(dry_manifest)

    assert real_manifest.exists(), "unrelated prior run's manifest must survive"
    assert real_episode_log.exists(), "unrelated prior run's episode_log must survive"
    assert not dry_manifest.exists()
    assert not dry_episode_log.exists()
    assert dest.exists()
    assert (dest.parent / dry_episode_log.name).exists()


def test_write_flat_manifest_style_relocates_declared_companion(tmp_path, monkeypatch):
    monkeypatch.setattr(ep.tempfile, "gettempdir", lambda: str(tmp_path / "systmp"))

    evidence_root = tmp_path / "evidence_experiments"
    per_type_dir = evidence_root / "v3_exq_998_fake_fishtank"
    per_type_dir.mkdir(parents=True)

    episode_log = per_type_dir / "v3_exq_998_fake_fishtank_20260811T000000Z_episode_log.json"
    episode_log.write_text(json.dumps({"episodes": []}))

    # Manifest lands FLAT at evidence_root, one level above the episode log,
    # exactly as write_flat_manifest(result, out_dir.parent, ...) does.
    manifest_path = evidence_root / "_dry_v3_exq_998_fake_fishtank_20260811T000000Z_v3.json"
    manifest_path.write_text(json.dumps({
        "run_id": "v3_exq_998_fake_fishtank_20260811T000000Z_v3",
        "companion_files": ["v3_exq_998_fake_fishtank/v3_exq_998_fake_fishtank_20260811T000000Z_episode_log.json"],
    }))

    dest = ep._relocate_dry_run_manifest(manifest_path)

    assert not manifest_path.exists()
    assert not episode_log.exists(), "companion_files-declared sidecar in a different directory must still relocate"
    assert dest.exists()
    assert (dest.parent / episode_log.name).exists()


def test_manifest_with_no_companion_relocates_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(ep.tempfile, "gettempdir", lambda: str(tmp_path / "systmp"))

    evidence_dir = tmp_path / "evidence_experiments" / "v3_exq_997_fake_no_sidecar"
    evidence_dir.mkdir(parents=True)
    manifest_path = evidence_dir / "v3_exq_997_fake_no_sidecar_20260811T000000Z.json"
    manifest_path.write_text(json.dumps({"run_id": "v3_exq_997_fake_no_sidecar_20260811T000000Z_v3"}))

    dest = ep._relocate_dry_run_manifest(manifest_path)

    assert not manifest_path.exists()
    assert dest.exists()
    assert list(dest.parent.iterdir()) == [dest]


def test_missing_declared_companion_does_not_block_manifest_relocation(tmp_path, monkeypatch):
    """Best-effort: a companion_files entry pointing nowhere must not raise."""
    monkeypatch.setattr(ep.tempfile, "gettempdir", lambda: str(tmp_path / "systmp"))

    evidence_dir = tmp_path / "evidence_experiments" / "v3_exq_996_fake_missing_sidecar"
    evidence_dir.mkdir(parents=True)
    manifest_path = evidence_dir / "v3_exq_996_fake_missing_sidecar_20260811T000000Z.json"
    manifest_path.write_text(json.dumps({
        "run_id": "v3_exq_996_fake_missing_sidecar_20260811T000000Z_v3",
        "companion_files": ["does_not_exist_episode_log.json"],
    }))

    dest = ep._relocate_dry_run_manifest(manifest_path)

    assert not manifest_path.exists()
    assert dest.exists()


def test_manifest_already_in_scratch_dir_is_not_deleted(tmp_path, monkeypatch):
    """Regression, 2026-08-19: a driver whose --dry-run out_dir IS the scratch
    dir itself (e.g. v3_exq_937a) writes its manifest directly to `dest`, so
    `src == dest`. The old code did `if dest.exists(): dest.unlink()` before
    the move -- which deleted the manifest that was ALREADY at the
    destination, then failed to find a source to move (swallowed by the
    outer best-effort except), silently destroying the smoke manifest.
    Confirmed live while smoke-testing V3-EXQ-937b.
    """
    scratch_root = tmp_path / "systmp"
    monkeypatch.setattr(ep.tempfile, "gettempdir", lambda: str(scratch_root))

    scratch = scratch_root / ep.DRY_RUN_SCRATCH_DIRNAME
    scratch.mkdir(parents=True)
    manifest_path = scratch / "v3_exq_937a_fake_20260819T000000Z.json"
    manifest_path.write_text(json.dumps({"run_id": "v3_exq_937a_fake_20260819T000000Z_v3"}))

    dest = ep._relocate_dry_run_manifest(manifest_path)

    assert manifest_path.exists(), "manifest must survive a relocate onto itself"
    assert dest == manifest_path
    assert dest.exists()
    assert dest.read_text() == json.dumps({"run_id": "v3_exq_937a_fake_20260819T000000Z_v3"})


def test_companion_already_in_scratch_dir_is_not_deleted(tmp_path, monkeypatch):
    """Same hazard as above, for the companion-file loop: a declared
    companion already sitting at its own relocation destination must survive
    the unlink-then-move instead of being deleted with no move to replace it.
    """
    scratch_root = tmp_path / "systmp"
    monkeypatch.setattr(ep.tempfile, "gettempdir", lambda: str(scratch_root))

    evidence_dir = tmp_path / "evidence_experiments" / "v3_exq_994_fake"
    evidence_dir.mkdir(parents=True)
    manifest_path = evidence_dir / "v3_exq_994_fake_20260819T000000Z.json"

    scratch = scratch_root / ep.DRY_RUN_SCRATCH_DIRNAME
    scratch.mkdir(parents=True)
    companion = scratch / "v3_exq_994_fake_20260819T000000Z_episode_log.json"
    companion.write_text(json.dumps({"episodes": ["smoke"]}))

    manifest_path.write_text(json.dumps({
        "run_id": "v3_exq_994_fake_20260819T000000Z_v3",
        "companion_files": [str(companion)],
    }))

    dest = ep._relocate_dry_run_manifest(manifest_path)

    assert companion.exists(), "companion already at its destination must survive"
    assert dest.exists()
    assert (dest.parent / companion.name).exists()


def test_emit_outcome_dry_run_relocates_manifest_and_companion(tmp_path, monkeypatch):
    """End-to-end through the public emit_outcome() entry point."""
    monkeypatch.setattr(ep.tempfile, "gettempdir", lambda: str(tmp_path / "systmp"))
    signal_dir = tmp_path / "signals"

    evidence_dir = tmp_path / "evidence_experiments" / "v3_exq_995_fake_e2e"
    evidence_dir.mkdir(parents=True)
    manifest_path = evidence_dir / "v3_exq_995_fake_e2e_20260811T000000Z.json"
    manifest_path.write_text(json.dumps({"run_id": "v3_exq_995_fake_e2e_20260811T000000Z_v3"}))
    episode_log = evidence_dir / "v3_exq_995_fake_e2e_20260811T000000Z_episode_log.json"
    episode_log.write_text(json.dumps({"episodes": []}))

    sentinel_path = ep.emit_outcome(
        outcome="PASS",
        manifest_path=manifest_path,
        run_id="v3_exq_995_fake_e2e_20260811T000000Z_v3",
        dry_run=True,
        signal_dir=signal_dir,
    )

    sentinel = json.loads(sentinel_path.read_text())
    assert sentinel["dry_run"] is True
    assert not manifest_path.exists()
    assert not episode_log.exists()
    relocated_manifest = Path(sentinel["manifest_path"])
    assert relocated_manifest.exists()
    assert (relocated_manifest.parent / episode_log.name).exists()
