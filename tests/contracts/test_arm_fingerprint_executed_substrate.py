"""Contract tests for the executed-substrate identity fix (2026-07-20).

Defect: `arm_cell` computed substrate_hash by hashing source files FROM DISK at cell
entry, while the cell executed in-memory bytecode frozen in sys.modules at first
import. A mid-run checkout move therefore stamped the DISK state, not the EXECUTED
state -- a FALSE-HIT channel, the one failure mode arm_fingerprint.py's governing
asymmetry says must never occur.

Origin: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-778a_2026-07-20.json
targets[0] (worked instance: 6 of 8 cells stamped c8d6d0e2 while provably executing
e9a22a91, all reuse_eligible: true). Corpus scale:
REE_assembly/evidence/planning/intra_run_substrate_divergence_sweep_2026-07-20.md
-- 42 of 164 fingerprinted runs, a 25.6% base rate.

Acceptance target from the failure record, which these tests encode directly:
    recorded per-cell substrate identity == executed substrate identity for 100% of
    cells, OR the run is stamped substrate_stable_across_run: false.

ASCII-only. Run: pytest tests/contracts/test_arm_fingerprint_executed_substrate.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments._lib import arm_fingerprint as afp
from experiments._lib import arm_reuse as ar
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
def clean_snapshot():
    """Every test starts from a cold process snapshot and leaves one behind."""
    afp._reset_substrate_snapshot()
    yield
    afp._reset_substrate_snapshot()


def _fp(repo_root, seed=42):
    return afp.compute_arm_fingerprint(
        config_slice={"k": 1}, seed=seed, script_path=None,
        rng_fully_reset=True, repo_root=repo_root,
    )


# --------------------------------------------------------------------------- #
# 1. The defect itself: cells must not split when the checkout moves mid-run.
# --------------------------------------------------------------------------- #

def test_cells_stamped_after_a_mid_run_edit_keep_the_executed_identity(fake_repo):
    """The 778a scenario. Cell 1 stamps, the tree moves, cells 2-3 stamp.

    Pre-fix this produced the observed 1/2 split. Post-fix all three carry the
    identity resolved at the first cell -- which is the one the frozen bytecode
    actually came from, since imports precede the first cell.
    """
    first = _fp(fake_repo)

    # The da873a1 analogue: one _lib file rewritten 85 seconds into the run.
    (fake_repo / "experiments" / "_lib" / "harness.py").write_text("H = 2  # edited\n")

    second = _fp(fake_repo, seed=43)
    third = _fp(fake_repo, seed=44)

    hashes = {first["substrate_hash"], second["substrate_hash"], third["substrate_hash"]}
    assert len(hashes) == 1, "cells split across substrates -- the D3 defect is back"

    # And it is pinned to what was on disk when the process started, not to the edit.
    assert first["substrate_hash"] == second["substrate_hash"] == third["substrate_hash"]


def test_a_fresh_process_does_see_the_edit(fake_repo):
    """The freeze must be per-PROCESS, not permanent -- a real code change must
    still bust the fingerprint, or the whole reuse key is worthless."""
    before = _fp(fake_repo)["substrate_hash"]
    (fake_repo / "ree_core" / "agent.py").write_text("VERSION = 2\n")
    afp._reset_substrate_snapshot()  # stands in for the next process
    after = _fp(fake_repo)["substrate_hash"]
    assert before != after


def test_snapshot_is_keyed_by_scope_and_root(fake_repo):
    """Two different substrate QUESTIONS must not share one snapshot."""
    whole = _fp(fake_repo)["substrate_hash"]
    scoped = afp.compute_arm_fingerprint(
        config_slice={"k": 1}, seed=42, script_path=None, rng_fully_reset=True,
        repo_root=fake_repo, substrate_scope=["ree_core/**/*.py"],
    )["substrate_hash"]
    assert whole != scoped


def test_fingerprint_value_is_unchanged_on_a_stable_run(fake_repo):
    """NOT a hard cut: with a stable substrate the emitted hash must equal what the
    raw pre-fix disk-read path produced, so every banked mint still matches."""
    raw = afp.compute_substrate_hash(repo_root=fake_repo)["substrate_hash"]
    assert _fp(fake_repo)["substrate_hash"] == raw


def test_payload_records_how_identity_was_obtained(fake_repo):
    p = _fp(fake_repo)
    assert p["substrate_identity_source"] == afp.SUBSTRATE_IDENTITY_SOURCE == "process_snapshot"
    assert p["substrate_identity_resolved_at"], "resolution time not recorded"


def test_snapshot_is_shared_across_import_aliases(fake_repo):
    """This file is imported under two names in a real experiment process --
    `experiments._lib.arm_fingerprint` (manifest_core) and bare `arm_fingerprint`
    (arm_reuse, maturation_curriculum). Python makes those two module objects with two
    sets of globals, so a per-module snapshot would let the two instances disagree and
    would blind manifest_core's stability report to drift only the other saw. The state
    is anchored on `sys` precisely to stop that; assert it, because the failure is silent.
    """
    import sys as _sys
    sys_path_added = str(Path(afp.__file__).resolve().parent)
    if sys_path_added not in _sys.path:
        _sys.path.insert(0, sys_path_added)
    import arm_fingerprint as bare  # noqa: E402  -- the second alias, on purpose

    assert bare is not afp, "aliases collapsed; this test no longer tests anything"
    assert bare._SUBSTRATE_SNAPSHOT is afp._SUBSTRATE_SNAPSHOT
    assert bare._DRIVER_SCRIPT_SNAPSHOT is afp._DRIVER_SCRIPT_SNAPSHOT

    # and the sharing is live, not just identity at import time
    _fp(fake_repo)
    assert len(bare._SUBSTRATE_SNAPSHOT) == 1
    afp._reset_substrate_snapshot()
    assert len(bare._SUBSTRATE_SNAPSHOT) == 0


# --------------------------------------------------------------------------- #
# 2. The adjunct: a checkout move is RECORDED as an instrument event.
# --------------------------------------------------------------------------- #

def test_stability_report_clean_when_nothing_moved(fake_repo):
    _fp(fake_repo)
    r = afp.substrate_stability_report()
    assert r["substrate_stable_across_run"] is True
    assert r["n_snapshots"] == 1 and r["drift"] == []


def test_stability_report_flags_a_mid_run_move(fake_repo):
    _fp(fake_repo)
    (fake_repo / "ree_core" / "agent.py").write_text("VERSION = 99\n")
    r = afp.substrate_stability_report()
    assert r["substrate_stable_across_run"] is False
    assert len(r["drift"]) == 1
    d = r["drift"][0]
    assert d["recorded"] != d["on_disk_now"]


def test_empty_process_is_stable_not_false(fake_repo):
    """A process that stamped no cells is vacuously stable; n_snapshots says so."""
    r = afp.substrate_stability_report()
    assert r["substrate_stable_across_run"] is True
    assert r["n_snapshots"] == 0


def test_stamp_records_false_when_the_tree_moved(fake_repo):
    _fp(fake_repo)
    (fake_repo / "ree_core" / "agent.py").write_text("VERSION = 3\n")
    m = mc.stamp_recording_core({"run_id": "r"}, config={}, seeds=[42])
    assert m["substrate_stable_across_run"] is False
    assert m["substrate_stability_detail"]["process_snapshot_drift"]


def test_stamp_records_false_when_cells_disagree(fake_repo):
    """The decisive test for a manifest assembled elsewhere: per-cell disagreement
    alone proves instability, with no process snapshot involved."""
    m = mc.stamp_recording_core({
        "run_id": "r",
        "arm_results": [
            {"seed": 42, "arm_fingerprint": {"substrate_hash": "e9a22a91"}},
            {"seed": 123, "arm_fingerprint": {"substrate_hash": "c8d6d0e2"}},
        ],
    }, config={}, seeds=[42, 123])
    assert m["substrate_stable_across_run"] is False
    assert m["substrate_stability_detail"]["per_cell_hashes_disagree"] is True
    assert sorted(m["substrate_stability_detail"]["distinct_cell_substrate_hashes"]) == \
        ["c8d6d0e2", "e9a22a91"]


def test_stamp_records_true_on_a_clean_multi_arm_run(fake_repo):
    _fp(fake_repo)
    m = mc.stamp_recording_core({
        "run_id": "r",
        "arm_results": [
            {"seed": 42, "arm_fingerprint": {"substrate_hash": "aaa"}},
            {"seed": 43, "arm_fingerprint": {"substrate_hash": "aaa"}},
        ],
    }, config={}, seeds=[42, 43])
    assert m["substrate_stable_across_run"] is True
    assert "substrate_stability_detail" not in m


def test_stamp_does_not_clobber_an_explicit_author_value(fake_repo):
    m = mc.stamp_recording_core(
        {"run_id": "r", "substrate_stable_across_run": False}, config={}, seeds=[42])
    assert m["substrate_stable_across_run"] is False


# --------------------------------------------------------------------------- #
# 3. The retroactive half: the reuse path refuses the 42 legacy divergent runs.
# --------------------------------------------------------------------------- #

def test_unstable_detector_on_the_778a_shape():
    """The worked instance verbatim: 2 seeds on e9a22a91, 6 on c8d6d0e2, no
    substrate_stable_across_run field at all (as every pre-fix manifest lacks it)."""
    manifest = {"arm_results": (
        [{"seed": s, "arm_fingerprint": {"substrate_hash": "e9a22a91"}} for s in (42, 7)]
        + [{"seed": s, "arm_fingerprint": {"substrate_hash": "c8d6d0e2"}}
           for s in (123, 2024, 99, 7777, 314, 1000)]
    )}
    assert ar.source_run_substrate_unstable(manifest) is True


def test_absent_stability_field_is_not_read_as_unstable():
    """The entire pre-2026-07-20 corpus lacks the field. Treating absence as failure
    would refuse every banked mint -- a false MISS so broad it breaks reuse outright."""
    manifest = {"arm_results": [
        {"seed": 42, "arm_fingerprint": {"substrate_hash": "same"}},
        {"seed": 43, "arm_fingerprint": {"substrate_hash": "same"}},
    ]}
    assert ar.source_run_substrate_unstable(manifest) is False


def test_explicit_false_is_honoured_even_with_agreeing_cells():
    """A lazily-imported module could move after the cells agreed; the stamped
    verdict outranks the cardinality test."""
    manifest = {
        "substrate_stable_across_run": False,
        "arm_results": [{"seed": 42, "arm_fingerprint": {"substrate_hash": "same"}}],
    }
    assert ar.source_run_substrate_unstable(manifest) is True


def test_manifest_without_arm_results_is_not_flagged():
    assert ar.source_run_substrate_unstable({"run_id": "single_arm"}) is False


def test_evaluate_reuse_refuses_a_divergent_source_run(tmp_path):
    """End-to-end: an otherwise perfectly reusable cell is REFUSED because its
    source run's other cells ran on a different substrate."""
    assembly = tmp_path / "REE_assembly"
    exp = assembly / "evidence" / "experiments"
    exp.mkdir(parents=True)

    cs = {"baseline_id": "x"}
    fp = afp.compute_arm_fingerprint(
        config_slice=cs, seed=42, script_path=None,
        rng_fully_reset=True, config_slice_declared=True,
    )["arm_fingerprint"]

    good_cell = {
        "arm": "off", "seed": 42, "mean_reward": 1.0,
        "arm_fingerprint": {"schema": afp.FINGERPRINT_SCHEMA, "arm_fingerprint": fp,
                            "substrate_hash": "aaaa", "reuse_eligible": True},
    }
    other_cell = {
        "arm": "off", "seed": 43, "mean_reward": 2.0,
        "arm_fingerprint": {"schema": afp.FINGERPRINT_SCHEMA, "arm_fingerprint": "other",
                            "substrate_hash": "bbbb", "reuse_eligible": True},
    }
    (exp / "mint.json").write_text(json.dumps({
        "run_id": "MINT-1", "outcome": "PASS",
        "arm_results": [good_cell, other_cell],
    }))
    (exp / "arm_fingerprint_index.json").write_text(json.dumps({
        "schema": ar.INDEX_SCHEMA, "regime": "A",
        "by_fingerprint": {fp: {
            "run_id": "MINT-1",
            "manifest_path": "evidence/experiments/mint.json",
            "reuse_eligible": True, "outcome": "PASS", "superseded": False,
            "fingerprint_schema": afp.FINGERPRINT_SCHEMA,
            "cell_keys": ["arm", "seed", "mean_reward"], "seed": 42,
        }},
    }))

    d = ar.evaluate_reuse(
        config_slice=cs, seed=42, script_path=None, needed_keys=["mean_reward"],
        cite_run_id="MINT-1",
        index_path=exp / "arm_fingerprint_index.json", assembly_root=assembly,
    )
    assert d.reused is False
    assert d.reason == ar.REFUSE_SUBSTRATE_UNSTABLE

    # Control: with the sibling cell on the SAME substrate, the identical request HITs.
    other_cell["arm_fingerprint"]["substrate_hash"] = "aaaa"
    (exp / "mint.json").write_text(json.dumps({
        "run_id": "MINT-1", "outcome": "PASS",
        "arm_results": [good_cell, other_cell],
    }))
    d2 = ar.evaluate_reuse(
        config_slice=cs, seed=42, script_path=None, needed_keys=["mean_reward"],
        cite_run_id="MINT-1",
        index_path=exp / "arm_fingerprint_index.json", assembly_root=assembly,
    )
    assert d2.reused is True, "the guard must refuse divergence, not reuse itself"
