"""Contract tests for the arm_cell accidental-bypass guard (2026-09-17,
chip-20260917-armcell-bypass-guard).

Defect: `_ArmCell.__enter__` (arm_fingerprint.py) is where `reset_all_rng(seed)`
runs. Any caller that constructs an `_ArmCell` (via `arm_cell(...)`) but never
enters it via `with` never gets that reset -- and, before this guard, `.stamp()`
would happily compute and return a fingerprint anyway, with nothing to say the
cell's numbers are off-contract. This mirrors the failure that invalidated
every full-scale figure taken for V3-EXQ-1048 (calling `run_cell(...)` directly
instead of running `main()`): see the module docstring on `_ArmCell` for the
narrower bypass shape (no `_ArmCell` ever constructed at all) that this guard
cannot detect, by construction -- that shape is out of reach for this module
and is mitigated only by the /queue-experiment skill's Arm fingerprint section,
not by a test here.

These tests assert on the fingerprint/eligibility CONTRACT (refuses / does not
refuse, and the recorded reuse_eligible + reason), never on a sampled action or
any RNG-drawn value -- per CLAUDE.md "Running the test suite" (never assert on
the discrete sampled action, only upstream of it; here there is no sampling at
all, only the bookkeeping around whether a reset happened).

ASCII-only. Run: pytest tests/contracts/test_arm_cell_bypass_guard.py -q
"""

from __future__ import annotations

import pytest

from experiments._lib import arm_fingerprint as afp


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
def clean_snapshot():
    """Every test starts from a cold process snapshot and leaves one behind."""
    afp._reset_substrate_snapshot()
    yield
    afp._reset_substrate_snapshot()


def test_stamp_without_entering_refuses(fake_repo):
    """Constructing arm_cell(...) and calling .stamp() WITHOUT `with` (the
    accidental-bypass shape: do_reset defaults True, __enter__ never ran) must
    raise, not silently emit a fingerprint for an unreset cell."""
    cell = afp.arm_cell(seed=7, config_slice={"k": 1}, repo_root=fake_repo)
    with pytest.raises(RuntimeError, match="reset_all_rng"):
        cell.stamp({})


def test_stamp_after_entering_succeeds_and_is_eligible(fake_repo):
    """The correct usage -- entering via `with` -- must be unaffected by the
    guard: __enter__ resets, .stamp() succeeds, and the cell is reuse_eligible."""
    row = {}
    with afp.arm_cell(seed=7, config_slice={"k": 1}, repo_root=fake_repo) as cell:
        cell.stamp(row)
    assert row["arm_fingerprint"]["reuse_eligible"] is True
    assert row["arm_fingerprint"]["reuse_ineligible_reasons"] == []


def test_deliberate_do_reset_false_opt_out_still_stamps(fake_repo):
    """A caller that explicitly declares do_reset=False (shared state across
    arms, etc.) must NOT be refused -- the guard only fires on the accidental
    shape (do_reset still True). The opt-out still emits, flagged ineligible,
    exactly as before this guard existed."""
    row = {}
    cell = afp.arm_cell(
        seed=7, config_slice={"k": 1}, repo_root=fake_repo, do_reset=False,
    )
    # Deliberately never entered via `with` -- the do_reset=False caller may
    # not use the context-manager form at all.
    cell.stamp(row)
    fp = row["arm_fingerprint"]
    assert fp["reuse_eligible"] is False
    assert any("rng" in r.lower() for r in fp["reuse_ineligible_reasons"])


def test_deliberate_do_reset_false_entered_via_with_also_stamps(fake_repo):
    """do_reset=False used WITH the context-manager form (enter is a no-op for
    the reset, __exit__ never suppresses) must also stamp without refusing."""
    row = {}
    with afp.arm_cell(
        seed=7, config_slice={"k": 1}, repo_root=fake_repo, do_reset=False,
    ) as cell:
        cell.stamp(row)
    assert row["arm_fingerprint"]["reuse_eligible"] is False


def test_second_stamp_after_proper_entry_still_succeeds(fake_repo):
    """.stamp() may legitimately be called more than once on an entered cell
    (e.g. re-stamping after enriching the row) -- the guard must not fire on
    a second call once _rng_reset is already True from __enter__."""
    with afp.arm_cell(seed=7, config_slice={"k": 1}, repo_root=fake_repo) as cell:
        cell.stamp({})
        cell.stamp({})  # must not raise
