"""Contract: substrate_stable_across_run compares LIKE SCOPES (2026-09-07).

Defect (chip-20260904-substrate-stability-flag-mint-scope): an experiment that
mints its OFF arm cross-driver reusable -- `include_driver_script_in_hash=(arm
!= OFF_ARM)`, the CLAUDE.md standing default -- produces two per-cell substrate
hashes BY CONSTRUCTION (OFF: globs only; ON: globs + driver). manifest_core's
bare cardinality test then stamped `substrate_stable_across_run: false` with an
empty `process_snapshot_drift`, and arm_reuse refused to serve the minted OFF
arm -- defeating the mint. Confirmed on V3-EXQ-976 and V3-EXQ-1000 (their
recorded hash pairs reproduce from origin with and without the driver folded).

Three cases, plus the two mirrors agreeing:
  1. driver-fold-only difference -> stable True, kind stamped, reuse allowed;
  2. same-scope difference       -> stable False (D3, the 42 legacy runs);
  3. real snapshot drift         -> stable False regardless of cell agreement.

976 and 1000 carry the old False on disk and are NOT retro-edited; arm_reuse's
explicit-False test (rule 1) still refuses them, by design.

ASCII-only. Run: pytest tests/contracts/test_substrate_stability_like_scope.py -q
"""

from __future__ import annotations

import pytest

from experiments._lib import arm_fingerprint as afp
from experiments._lib import arm_reuse as ar
from experiments._lib import manifest_core as mc


@pytest.fixture
def fake_repo(tmp_path):
    root = tmp_path / "ree-v3"
    (root / "ree_core").mkdir(parents=True)
    (root / "experiments" / "_lib").mkdir(parents=True)
    (root / "ree_core" / "agent.py").write_text("VERSION = 1\n")
    (root / "experiments" / "_lib" / "harness.py").write_text("H = 1\n")
    return root


@pytest.fixture(autouse=True)
def clean_snapshot():
    afp._reset_substrate_snapshot()
    yield
    afp._reset_substrate_snapshot()


def _cell(seed, substrate_hash, driver_folded):
    fp = {"substrate_hash": substrate_hash}
    if driver_folded is not None:
        fp["driver_script_in_substrate_hash"] = driver_folded
    return {"seed": seed, "arm_fingerprint": fp}


# The V3-EXQ-1000 shape: OFF cells hash globs only, ON cells hash globs + driver.
MINT_AS_YOU_GO = [
    _cell(42, "5a198a8e", False), _cell(43, "5a198a8e", False),
    _cell(42, "d7fe936d", True), _cell(43, "d7fe936d", True),
]
# The V3-EXQ-778a shape: pre-flag cells split across two substrates.
LEGACY_SPLIT = [_cell(42, "e9a22a91", None), _cell(123, "c8d6d0e2", None)]
# A post-flag run whose ON cells genuinely split.
SAME_SCOPE_SPLIT = [_cell(42, "aaa", True), _cell(43, "bbb", True), _cell(42, "off", False)]


def _snapshot(repo):
    afp.compute_arm_fingerprint(config_slice={"k": 1}, seed=42, script_path=None,
                                rng_fully_reset=True, repo_root=repo)


# --------------------------------------------------------------------------- #
# 1. driver-fold-only difference is NOT instability
# --------------------------------------------------------------------------- #

def test_classifier_names_the_driver_fold():
    assert mc.multi_arm_substrate_disagreement({"arm_results": MINT_AS_YOU_GO}) == \
        mc.DISAGREEMENT_DRIVER_FOLD_ONLY


def test_stamp_is_true_with_kind_recorded_on_a_mint_as_you_go_run(fake_repo):
    _snapshot(fake_repo)
    m = mc.stamp_recording_core({"run_id": "r", "arm_results": list(MINT_AS_YOU_GO)},
                                config={}, seeds=[42, 43])
    assert m["substrate_stable_across_run"] is True
    d = m["substrate_stability_detail"]
    assert d["disagreement_kind"] == mc.DISAGREEMENT_DRIVER_FOLD_ONLY
    assert d["per_cell_hashes_disagree"] is False
    assert sorted(d["distinct_cell_substrate_hashes"]) == ["5a198a8e", "d7fe936d"]
    assert d["process_snapshot_drift"] == []


def test_reuse_gate_does_not_refuse_a_mint_as_you_go_run():
    assert ar.source_run_substrate_unstable({"arm_results": MINT_AS_YOU_GO}) is False


# --------------------------------------------------------------------------- #
# 2. same-scope difference is still instability (D3)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cells", [LEGACY_SPLIT, SAME_SCOPE_SPLIT])
def test_same_scope_split_is_false(fake_repo, cells):
    assert mc.multi_arm_substrate_disagreement({"arm_results": cells}) == \
        mc.DISAGREEMENT_SAME_SCOPE
    m = mc.stamp_recording_core({"run_id": "r", "arm_results": list(cells)},
                                config={}, seeds=[42, 43])
    assert m["substrate_stable_across_run"] is False
    assert m["substrate_stability_detail"]["disagreement_kind"] == mc.DISAGREEMENT_SAME_SCOPE
    assert m["substrate_stability_detail"]["per_cell_hashes_disagree"] is True
    assert ar.source_run_substrate_unstable({"arm_results": cells}) is True


def test_clean_run_has_no_detail_block(fake_repo):
    _snapshot(fake_repo)
    cells = [_cell(42, "same", True), _cell(43, "same", True)]
    m = mc.stamp_recording_core({"run_id": "r", "arm_results": cells}, config={}, seeds=[42, 43])
    assert m["substrate_stable_across_run"] is True
    assert "substrate_stability_detail" not in m
    assert mc.multi_arm_substrate_disagreement({"arm_results": cells}) == mc.DISAGREEMENT_NONE


# --------------------------------------------------------------------------- #
# 3. real snapshot drift is instability regardless of the cell verdict
# --------------------------------------------------------------------------- #

def test_snapshot_drift_wins_even_on_a_driver_fold_only_run(fake_repo):
    _snapshot(fake_repo)
    (fake_repo / "ree_core" / "agent.py").write_text("VERSION = 2\n")
    m = mc.stamp_recording_core({"run_id": "r", "arm_results": list(MINT_AS_YOU_GO)},
                                config={}, seeds=[42, 43])
    assert m["substrate_stable_across_run"] is False
    d = m["substrate_stability_detail"]
    assert d["process_snapshot_drift"]
    assert d["disagreement_kind"] == mc.DISAGREEMENT_DRIVER_FOLD_ONLY


# --------------------------------------------------------------------------- #
# the two mirrors agree, and the explicit False on 976/1000 still refuses
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cells", [MINT_AS_YOU_GO, LEGACY_SPLIT, SAME_SCOPE_SPLIT, []])
def test_arm_reuse_mirror_agrees_with_manifest_core(cells):
    expected = mc.multi_arm_substrate_disagreement({"arm_results": cells}) == mc.DISAGREEMENT_SAME_SCOPE
    assert ar._same_scope_cell_disagreement({"arm_results": cells}) is expected


def test_recorded_false_on_landed_manifests_is_still_honoured():
    m = {"substrate_stable_across_run": False, "arm_results": list(MINT_AS_YOU_GO)}
    assert ar.source_run_substrate_unstable(m) is True
