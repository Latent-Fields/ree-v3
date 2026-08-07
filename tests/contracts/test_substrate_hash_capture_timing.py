"""Contract tests for WHEN a manifest's top-level substrate identity is captured.

Defect (V3-EXQ-866b, found in its confirmed failure autopsy 2026-08-07):
`manifest_core.compute_single_arm_substrate_hash` called `compute_substrate_hash`
DIRECTLY, so on a single-arm run the manifest's top-level `substrate_hash` was a
disk read taken at MANIFEST-WRITE time -- while the run's own arm cells were served
the process snapshot frozen at the first cell by the 2026-07-20 executed-substrate
fix. On a multi-hour run against the shared hub checkout (which sync_daemon writers
and other sessions move continuously) the two disagree by construction.

Worked instance:
  run    v3_exq_866b_603q_substrate_regression_check_20260803T192405Z_v3
  cells  bb755658...  resolved 2026-08-03T16:58:16Z
  stamp  8e275408...  read     2026-08-03T19:24:05Z  (2h26m later)
Reproduced exactly by re-hashing the tree at the recorded substrate_commit
(c4247794): the driver-INCLUSIVE hash there is 8e275408..., i.e. the same snapshot
key the cells had already pinned, read a second time hours afterwards. The run's
entire purpose was certifying that the substrate still reproduced V3-EXQ-603q, so
it could not say which substrate produced its own numbers.

Corpus scale at the time of the fix: 10 of the 249 manifests in
REE_assembly/evidence/experiments/ carrying a substrate_hash record a top-level
value equal to the write-time disk read and unequal to the executed snapshot,
with lags up to two days.

Acceptance target these tests encode:
  (a) the recorded top-level substrate_hash is the EXECUTED (snapshot) one whenever
      the process froze an identity before the manifest write; and
  (b) when it is not -- a driver that stamps no cell and pins nothing -- the manifest
      SAYS SO, rather than presenting a write-time read as the run's identity.

ASCII-only. Run: pytest tests/contracts/test_substrate_hash_capture_timing.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments._lib import arm_fingerprint as afp
from experiments._lib import manifest_core as mc


@pytest.fixture
def fake_repo(tmp_path):
    """A minimal ree-v3-shaped tree the substrate globs actually match."""
    root = tmp_path / "ree-v3"
    (root / "ree_core").mkdir(parents=True)
    (root / "experiments" / "_lib").mkdir(parents=True)
    (root / "ree_core" / "agent.py").write_text("VERSION = 1\n")
    (root / "experiments" / "_lib" / "harness.py").write_text("H = 1\n")
    return root


@pytest.fixture(autouse=True)
def clean_state():
    """Every test starts from a cold process snapshot and leaves one behind."""
    afp._reset_substrate_snapshot()
    mc._reset_recording_substrate_pins()
    yield
    afp._reset_substrate_snapshot()
    mc._reset_recording_substrate_pins()


def _cell(repo_root, script_path=None, seed=42):
    """Stand-in for one arm cell: exactly what arm_cell stamps."""
    return afp.compute_arm_fingerprint(
        config_slice={"k": 1}, seed=seed, script_path=script_path,
        rng_fully_reset=True, repo_root=repo_root,
    )


def _move_the_checkout(repo_root):
    """The mid-run tree movement -- a sibling session landing a commit."""
    (repo_root / "experiments" / "_lib" / "harness.py").write_text("H = 2  # moved\n")


# --------------------------------------------------------------------------- #
# 1. The defect itself.
# --------------------------------------------------------------------------- #

def test_866b_replay_manifest_records_the_executed_hash_not_the_write_time_read(fake_repo):
    """The V3-EXQ-866b scenario end to end, on the same shape of driver.

    Driver stamps its seed cells with script_path=__file__ (arm_cell's default folds
    the driver into the hash), the checkout then moves, and only afterwards is the
    manifest written. Pre-fix the manifest carried the post-move disk read.
    """
    driver = fake_repo / "experiments" / "v3_exq_866b_driver.py"
    driver.write_text("# driver\n")

    executed = _cell(fake_repo, script_path=driver)["substrate_hash"]
    _move_the_checkout(fake_repo)
    on_disk_now = afp.compute_substrate_hash(
        extra_paths=[driver], repo_root=fake_repo
    )["substrate_hash"]
    assert executed != on_disk_now, "fixture failed to move the tree"

    manifest = {"run_id": "r"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], script_path=driver, repo_root=fake_repo
    )

    assert manifest["substrate_hash"] == executed, (
        "manifest recorded the manifest-write-time disk read, not the executed "
        "substrate -- the 866b gap is back"
    )
    assert manifest["substrate_hash"] != on_disk_now


def test_the_mismatch_is_visible_not_silently_overwritten(fake_repo):
    """Both hashes and both timestamps must be readable off the manifest.

    The autopsy's requirement: a reader must be able to see that the tree moved and
    what it moved TO, without inferring it from a stability flag whose drift entries
    do not name which snapshot key they belong to.
    """
    driver = fake_repo / "experiments" / "v3_exq_866b_driver.py"
    driver.write_text("# driver\n")
    executed = _cell(fake_repo, script_path=driver)["substrate_hash"]
    _move_the_checkout(fake_repo)
    on_disk_now = afp.compute_substrate_hash(
        extra_paths=[driver], repo_root=fake_repo
    )["substrate_hash"]

    manifest = {"run_id": "r"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], script_path=driver, repo_root=fake_repo
    )

    ident = manifest["substrate_identity"]
    assert ident["source"] == "process_snapshot"
    assert ident["resolved_at_utc"], "resolution time not recorded"
    assert ident["stamped_at_utc"], "stamp time not recorded"
    assert ident["drifted_since_resolved"] is True
    assert ident["hash_on_disk_at_stamp"] == on_disk_now
    assert manifest["substrate_hash"] == executed
    assert manifest["substrate_stable_across_run"] is False


def test_stable_run_records_no_drift_and_a_zero_lag(fake_repo):
    """The normal case must stay quiet: same hash, no drift claim, honest lag."""
    driver = fake_repo / "experiments" / "d.py"
    driver.write_text("# driver\n")
    executed = _cell(fake_repo, script_path=driver)["substrate_hash"]

    manifest = {"run_id": "r"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], script_path=driver, repo_root=fake_repo
    )
    ident = manifest["substrate_identity"]
    assert manifest["substrate_hash"] == executed
    assert ident["drifted_since_resolved"] is False
    assert "hash_on_disk_at_stamp" not in ident
    assert ident["lag_seconds"] is not None and ident["lag_seconds"] >= 0
    assert manifest["substrate_stable_across_run"] is True


# --------------------------------------------------------------------------- #
# 2. The honest-labelling half: a driver that pinned nothing.
# --------------------------------------------------------------------------- #

def test_a_driver_with_no_cells_is_labelled_write_time_not_snapshot(fake_repo):
    """No cell, no pin -> the stamp IS the first look, and must say so.

    This is the case the fix cannot make correct (nothing recorded what the tree
    looked like earlier), so the requirement is that it cannot be mistaken for the
    correct one. Value is unchanged from pre-fix; only the label is new.
    """
    driver = fake_repo / "experiments" / "d.py"
    driver.write_text("# driver\n")

    manifest = {"run_id": "r"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], script_path=driver, repo_root=fake_repo
    )
    ident = manifest["substrate_identity"]
    assert ident["source"] == "manifest_write_disk_read"
    assert manifest["substrate_hash"] == afp.compute_substrate_hash(
        extra_paths=[driver], repo_root=fake_repo
    )["substrate_hash"]


def test_a_vacuous_stability_verdict_is_countable(fake_repo):
    """`substrate_stable_across_run: true` over ZERO snapshots is not evidence.

    A cell-less driver's process never froze an identity to re-check, so its stable
    verdict is vacuous -- and looks identical on the page to an earned one. The
    snapshot count is what separates them.
    """
    driver = fake_repo / "experiments" / "d.py"
    driver.write_text("# driver\n")

    manifest = {"run_id": "r"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], script_path=driver, repo_root=fake_repo
    )
    # The stamp itself resolves one snapshot, seconds before the re-check -- so the
    # verdict is true, and the count plus the write_time source say why that is weak.
    assert manifest["substrate_stable_across_run"] is True
    ident = manifest["substrate_identity"]
    assert ident["source"] == "manifest_write_disk_read"
    assert ident["stability_snapshots"] == 1

    # A run that DID freeze at a cell earns its verdict over the same count but from
    # a snapshot taken before the work, which is the difference `source` records.
    afp._reset_substrate_snapshot()
    _cell(fake_repo, script_path=driver)
    m2 = {"run_id": "r2"}
    mc.stamp_recording_core(
        m2, config={"a": 1}, seeds=[42], script_path=driver, repo_root=fake_repo
    )
    assert m2["substrate_identity"]["source"] == "process_snapshot"
    assert m2["substrate_identity"]["stability_snapshots"] == 1


def test_pin_recording_substrate_makes_that_driver_correct_too(fake_repo):
    """One line at run start buys the cell-less driver the executed identity."""
    driver = fake_repo / "experiments" / "d.py"
    driver.write_text("# driver\n")

    pinned = mc.pin_recording_substrate(script_path=driver, repo_root=fake_repo)
    _move_the_checkout(fake_repo)

    manifest = {"run_id": "r"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], script_path=driver, repo_root=fake_repo
    )
    assert manifest["substrate_hash"] == pinned["substrate_hash"]
    assert manifest["substrate_identity"]["source"] == "process_snapshot"
    assert manifest["substrate_identity"]["drifted_since_resolved"] is True


# --------------------------------------------------------------------------- #
# 3. The key must not be confused with a neighbouring one.
# --------------------------------------------------------------------------- #

def test_a_driver_exclusive_cell_pin_does_not_answer_for_the_inclusive_hash(fake_repo):
    """A cell stamped with include_driver_script_in_hash=False pins the extras=()
    key. The manifest's top-level hash folds the driver, so it is a DIFFERENT key
    and must not be served -- nor reported as snapshot-sourced -- off that pin.

    Reading the drift report by repo_root/scope alone would confuse exactly these
    two (the entries carry no extra_paths), which is why the drift match is on the
    recorded hash value instead.
    """
    driver = fake_repo / "experiments" / "d.py"
    driver.write_text("# driver\n")
    afp.compute_arm_fingerprint(
        config_slice={"k": 1}, seed=42, script_path=driver, rng_fully_reset=True,
        repo_root=fake_repo, include_driver_script_in_hash=False,
    )

    manifest = {"run_id": "r"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], script_path=driver, repo_root=fake_repo
    )
    assert manifest["substrate_identity"]["source"] == "manifest_write_disk_read"


def test_substrate_identity_pinned_is_read_only(fake_repo):
    """The probe must never itself resolve -- asking must not change the answer."""
    assert afp.substrate_identity_pinned(repo_root=fake_repo) is False
    assert afp.substrate_identity_pinned(repo_root=fake_repo) is False
    afp.resolve_substrate_identity(repo_root=fake_repo)
    assert afp.substrate_identity_pinned(repo_root=fake_repo) is True
    # ... and keyed, not global.
    assert afp.substrate_identity_pinned(
        extra_paths=[fake_repo / "ree_core" / "agent.py"], repo_root=fake_repo
    ) is False


# --------------------------------------------------------------------------- #
# 4. Value compatibility -- this must NOT be a hard cut.
# --------------------------------------------------------------------------- #

def test_stable_run_hash_is_byte_identical_to_the_pre_fix_disk_read(fake_repo):
    """Whenever the substrate holds still -- the normal case -- the recorded value
    is exactly what the old direct-disk-read path produced, so no banked comparison
    and no historical manifest is invalidated by this change."""
    driver = fake_repo / "experiments" / "d.py"
    driver.write_text("# driver\n")
    raw = afp.compute_substrate_hash(
        extra_paths=[driver], repo_root=fake_repo
    )["substrate_hash"]
    assert mc.compute_single_arm_substrate_hash(
        script_path=driver, repo_root=fake_repo
    ) == raw


def test_multi_arm_hoist_still_wins_and_carries_the_cell_resolution_time(fake_repo):
    """A multi-arm manifest keeps hoisting from its cells (unchanged), and the
    hoisted value inherits the CELL's resolution time, not the stamp time."""
    cell = _cell(fake_repo)
    manifest = {
        "run_id": "r",
        "arm_results": [{"arm": "A", "arm_fingerprint": cell}],
    }
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], repo_root=fake_repo
    )
    assert manifest["substrate_hash"] == cell["substrate_hash"]
    ident = manifest["substrate_identity"]
    assert ident["source"] == "per_cell_fingerprint_hoist"
    assert ident["resolved_at_utc"] == cell["substrate_identity_resolved_at"]


def test_an_author_set_substrate_hash_is_never_relabelled(fake_repo):
    """No-op-safety: a manifest that already carries a hash keeps it AND gets no
    substrate_identity block, since this call did not determine the value and has
    no standing to describe its provenance."""
    manifest = {"run_id": "r", "substrate_hash": "author_set"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42], repo_root=fake_repo
    )
    assert manifest["substrate_hash"] == "author_set"
    assert "substrate_identity" not in manifest


def test_lag_seconds_is_none_rather_than_zero_when_unparseable(fake_repo):
    """A missing lag must be visibly missing: 0 would read as 'stamped at the
    instant it was resolved', which is the claim this block exists to stop a
    manifest making without evidence."""
    assert mc._utc_delta_seconds(None, "2026-08-03T19:24:05Z") is None
    assert mc._utc_delta_seconds("not-a-time", "2026-08-03T19:24:05Z") is None
    assert mc._utc_delta_seconds(
        "2026-08-03T16:58:16Z", "2026-08-03T19:24:05Z"
    ) == 8749  # the real 866b lag: 2h 25m 49s


def test_stamping_never_raises_when_the_repo_is_unreadable(tmp_path):
    """Provenance stamping must never fail an experiment that already spent its
    compute -- the standing posture of every branch in stamp_recording_core."""
    manifest = {"run_id": "r"}
    mc.stamp_recording_core(
        manifest, config={"a": 1}, seeds=[42],
        script_path=tmp_path / "nope.py", repo_root=tmp_path / "does_not_exist",
    )
    assert manifest["recording_schema"] == mc.RECORDING_SCHEMA
