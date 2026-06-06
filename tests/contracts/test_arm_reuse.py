"""Contract tests for the arm-reuse Phase 1 consumer (plan section 9).

Covers EVERY refuse branch of experiments/_lib/arm_reuse.try_reuse_cell plus the
happy path, and the indexer's arm_fingerprint_index.json writer
(non-double-count of reused cells, reverse-index, pending_reuse_revalidation,
collapse-prefer-newest, ERROR/superseded exclusion).

The governing asymmetry (plan section 2): a false cache-HIT corrupts science; a
false cache-MISS only wastes compute. These tests assert the helper REFUSES in
every doubtful case.

ASCII-only. Run: pytest tests/contracts/test_arm_reuse.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# conftest puts ree-v3 root on sys.path -> `experiments._lib.*` importable.
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, FINGERPRINT_SCHEMA
from experiments._lib import arm_reuse as ar

_REE_V3_ROOT = Path(__file__).resolve().parents[2]
_REE_WORKING = _REE_V3_ROOT.parent
_INDEXER_PATH = (
    _REE_WORKING / "REE_assembly" / "evidence" / "experiments" / "scripts"
    / "build_experiment_indexes.py"
)


# --------------------------------------------------------------------------- #
# Fixtures: a temp REE_assembly-shaped tree with an index + a mint manifest.
# --------------------------------------------------------------------------- #

def _fingerprint(config_slice, seed):
    """Recompute the fingerprint the SAME way try_reuse_cell will (script_path=None)."""
    return compute_arm_fingerprint(
        config_slice=config_slice,
        seed=seed,
        script_path=None,
        rng_fully_reset=True,
        config_slice_declared=True,
    )["arm_fingerprint"]


def _make_cell(fp, seed, *, extra_keys=None):
    """A minted OFF cell: metric keys + the arm_fingerprint sub-dict."""
    cell = {
        "arm": "authority_off_baseline",
        "seed": seed,
        "mean_reward": 0.123,
        "end_phase_2_entropy": 0.5,
        "end_phase_3_entropy": 0.4,
        "arm_fingerprint": {
            "schema": FINGERPRINT_SCHEMA,
            "arm_fingerprint": fp,
            "machine_class": "linux-x86_64-py3.13",
            "regime": "A",
            "seed": seed,
            "reuse_eligible": True,
            "reuse_ineligible_reasons": [],
        },
    }
    if extra_keys:
        cell.update(extra_keys)
    return cell


def _make_index_entry(fp, run_id, cell, *, eligible=True, outcome="PASS",
                      superseded=False, schema=FINGERPRINT_SCHEMA,
                      manifest_rel="evidence/experiments/mint.json"):
    return {
        "run_id": run_id,
        "manifest_path": manifest_rel,
        "experiment_type": "v3_exq_610_inv074_crystallization_baseline_mint",
        "machine_class": "linux-x86_64-py3.13",
        "regime": "A",
        "reuse_eligible": eligible,
        "outcome": outcome,
        "cell_keys": sorted(k for k in cell.keys() if k != "arm_fingerprint"),
        "superseded": superseded,
        "fingerprint_schema": schema,
        "seed": cell.get("seed"),
    }


@pytest.fixture
def reuse_tree(tmp_path):
    """Builds a temp assembly tree and returns a builder closure.

    builder(run_id="MINT-1", seed=42, config_slice=..., **entry_overrides) writes
    a manifest + an index and returns dict(index_path, assembly_root, fp,
    config_slice, seed, cell, run_id).
    """
    assembly_root = tmp_path / "REE_assembly"
    exp_dir = assembly_root / "evidence" / "experiments"
    exp_dir.mkdir(parents=True)

    def builder(run_id="MINT-1", seed=42, config_slice=None, cell_extra=None,
                **entry_overrides):
        cs = config_slice if config_slice is not None else {"baseline_id": "exq610", "seed_set": [42, 43, 44]}
        fp = _fingerprint(cs, seed)
        cell = _make_cell(fp, seed, extra_keys=cell_extra)
        manifest_rel = entry_overrides.pop("manifest_rel", "evidence/experiments/mint.json")
        manifest = {
            "run_id": run_id,
            "experiment_type": "v3_exq_610_inv074_crystallization_baseline_mint",
            "outcome": entry_overrides.get("outcome", "PASS"),
            "arm_results": [cell],
        }
        (assembly_root / manifest_rel).write_text(json.dumps(manifest, indent=2))
        entry = _make_index_entry(fp, run_id, cell, manifest_rel=manifest_rel,
                                  **entry_overrides)
        index = {
            "schema": ar.INDEX_SCHEMA,
            "regime": "A",
            "by_fingerprint": {fp: entry},
            "reverse_index": {},
            "pending_reuse_revalidation": [],
        }
        index_path = exp_dir / "arm_fingerprint_index.json"
        index_path.write_text(json.dumps(index, indent=2))
        return {
            "index_path": index_path, "assembly_root": assembly_root,
            "fp": fp, "config_slice": cs, "seed": seed, "cell": cell, "run_id": run_id,
        }

    return builder


def _call(t, *, needed_keys=("mean_reward",), cite_run_id="MINT-1", **kw):
    return ar.evaluate_reuse(
        config_slice=t["config_slice"], seed=t["seed"], script_path=None,
        needed_keys=needed_keys, cite_run_id=cite_run_id,
        index_path=t["index_path"], assembly_root=t["assembly_root"], **kw,
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_happy_path_reuses_and_stamps_provenance(reuse_tree):
    t = reuse_tree()
    d = _call(t, needed_keys=["mean_reward", "end_phase_3_entropy"])
    assert d.reused is True
    assert d.reason == "ok"
    assert d.source_run_id == "MINT-1"
    assert d.cell["reused_from_run_id"] == "MINT-1"
    assert d.cell["reused_fingerprint"] == t["fp"]
    assert "reused_at_utc" in d.cell
    # original metric is preserved for the caller to consume
    assert d.cell["mean_reward"] == 0.123


def test_try_reuse_cell_returns_cell_on_hit_and_logs(reuse_tree):
    t = reuse_tree()
    logs = []
    cell = ar.try_reuse_cell(
        t["config_slice"], t["seed"], None, ["mean_reward"], cite_run_id="MINT-1",
        index_path=t["index_path"], assembly_root=t["assembly_root"], logger=logs.append,
    )
    assert cell is not None
    assert cell["reused_from_run_id"] == "MINT-1"
    assert any("reuse_HIT" in line for line in logs)


def test_cite_none_allows_any_match(reuse_tree):
    """cite_run_id=None must still HIT (cite is an optional extra guard)."""
    t = reuse_tree()
    d = _call(t, cite_run_id=None)
    assert d.reused is True


# --------------------------------------------------------------------------- #
# Refuse branches (one per plan section-9.2 rule)
# --------------------------------------------------------------------------- #

def test_refuse_fingerprint_mismatch(reuse_tree):
    t = reuse_tree(seed=42)
    # request a different seed -> different fingerprint -> not in index
    d = ar.evaluate_reuse(
        config_slice=t["config_slice"], seed=43, script_path=None,
        needed_keys=["mean_reward"], cite_run_id="MINT-1",
        index_path=t["index_path"], assembly_root=t["assembly_root"],
    )
    assert d.reused is False
    assert d.reason == ar.REFUSE_FP_NOT_IN_INDEX


def test_refuse_config_slice_mismatch(reuse_tree):
    t = reuse_tree()
    d = ar.evaluate_reuse(
        config_slice={"baseline_id": "DIFFERENT"}, seed=t["seed"], script_path=None,
        needed_keys=["mean_reward"], cite_run_id="MINT-1",
        index_path=t["index_path"], assembly_root=t["assembly_root"],
    )
    assert d.reused is False
    assert d.reason == ar.REFUSE_FP_NOT_IN_INDEX


def test_refuse_ineligible(reuse_tree):
    t = reuse_tree(eligible=False)
    d = _call(t)
    assert d.reused is False
    assert d.reason == ar.REFUSE_NOT_ELIGIBLE


def test_refuse_error_parent(reuse_tree):
    t = reuse_tree(outcome="ERROR")
    d = _call(t)
    assert d.reused is False
    assert d.reason == ar.REFUSE_PARENT_ERROR


def test_refuse_superseded(reuse_tree):
    t = reuse_tree(superseded=True)
    d = _call(t)
    assert d.reused is False
    assert d.reason == ar.REFUSE_SUPERSEDED


def test_refuse_missing_needed_keys(reuse_tree):
    """The section-9.2 correctness trap: an OFF metric the mint did not record."""
    t = reuse_tree()
    d = _call(t, needed_keys=["mean_reward", "a_metric_the_mint_never_recorded"])
    assert d.reused is False
    assert d.reason == ar.REFUSE_NEEDED_KEYS


def test_refuse_schema_mismatch(reuse_tree):
    t = reuse_tree(schema="arm_fp/v0")
    d = _call(t)
    assert d.reused is False
    assert d.reason == ar.REFUSE_SCHEMA


def test_refuse_cite_mismatch(reuse_tree):
    t = reuse_tree(run_id="MINT-1")
    d = _call(t, cite_run_id="SOME-OTHER-RUN")
    assert d.reused is False
    assert d.reason == ar.REFUSE_CITE_MISMATCH


def test_refuse_no_index(reuse_tree, tmp_path):
    t = reuse_tree()
    d = ar.evaluate_reuse(
        config_slice=t["config_slice"], seed=t["seed"], script_path=None,
        needed_keys=["mean_reward"], cite_run_id="MINT-1",
        index_path=tmp_path / "does_not_exist.json", assembly_root=t["assembly_root"],
    )
    assert d.reused is False
    assert d.reason == ar.REFUSE_NO_INDEX


def test_refuse_manifest_unreadable(reuse_tree):
    """Index entry points at a manifest that is not on disk -> refuse, not crash."""
    t = reuse_tree(manifest_rel="evidence/experiments/mint.json")
    # delete the manifest the index points to
    (t["assembly_root"] / "evidence" / "experiments" / "mint.json").unlink()
    d = _call(t)
    assert d.reused is False
    assert d.reason == ar.REFUSE_MANIFEST_UNREADABLE


def test_try_reuse_cell_returns_none_on_refuse(reuse_tree):
    t = reuse_tree(outcome="ERROR")
    logs = []
    cell = ar.try_reuse_cell(
        t["config_slice"], t["seed"], None, ["mean_reward"], cite_run_id="MINT-1",
        index_path=t["index_path"], assembly_root=t["assembly_root"], logger=logs.append,
    )
    assert cell is None
    assert any("reuse_refused" in line for line in logs)


def test_empty_needed_keys_is_subset(reuse_tree):
    """set() is a subset of anything -> the needed_keys gate does not block."""
    t = reuse_tree()
    d = _call(t, needed_keys=[])
    assert d.reused is True


# --------------------------------------------------------------------------- #
# Indexer: arm_fingerprint_index.json writer (plan section 9.1 / 9.3)
# --------------------------------------------------------------------------- #

def _load_indexer():
    spec = importlib.util.spec_from_file_location("bei_test", _INDEXER_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bei_test"] = m
    spec.loader.exec_module(m)
    return m


def _write_manifest(exp_dir, run_id, cells, *, outcome="PASS", evidence_direction=None,
                    ts="20260606T120000Z"):
    manifest = {
        "run_id": run_id,
        "experiment_type": "exp_t",
        "outcome": outcome,
        "timestamp_utc": ts,
        "arm_results": cells,
    }
    if evidence_direction is not None:
        manifest["evidence_direction"] = evidence_direction
    (exp_dir / (run_id + ".json")).write_text(json.dumps(manifest, indent=2))


def _source_cell(fp, seed=42, eligible=True, schema=FINGERPRINT_SCHEMA):
    return {
        "arm": "off", "seed": seed, "mean_reward": 1.0,
        "arm_fingerprint": {
            "schema": schema, "arm_fingerprint": fp, "machine_class": "mc",
            "regime": "A", "seed": seed, "reuse_eligible": eligible,
        },
    }


@pytest.fixture
def indexer_tree(tmp_path):
    bei = _load_indexer()
    assembly = tmp_path / "REE_assembly"
    exp_dir = assembly / "evidence" / "experiments"
    exp_dir.mkdir(parents=True)
    return bei, exp_dir


def test_index_writer_indexes_eligible_source(indexer_tree):
    bei, exp_dir = indexer_tree
    _write_manifest(exp_dir, "MINT_A", [_source_cell("fp_aaa")])
    idx = bei._write_arm_fingerprint_index(exp_dir, "now")
    assert "fp_aaa" in idx["by_fingerprint"]
    assert idx["by_fingerprint"]["fp_aaa"]["run_id"] == "MINT_A"
    assert idx["n_source_cells"] == 1


def test_index_writer_excludes_ineligible_error_superseded(indexer_tree):
    bei, exp_dir = indexer_tree
    _write_manifest(exp_dir, "INELIG", [_source_cell("fp_inelig", eligible=False)])
    _write_manifest(exp_dir, "ERR", [_source_cell("fp_err")], outcome="ERROR")
    _write_manifest(exp_dir, "SUP", [_source_cell("fp_sup")], evidence_direction="superseded")
    idx = bei._write_arm_fingerprint_index(exp_dir, "now")
    assert idx["by_fingerprint"] == {}


def test_index_writer_no_double_count_reused_cell(indexer_tree):
    bei, exp_dir = indexer_tree
    # a fresh source ...
    _write_manifest(exp_dir, "MINT_A", [_source_cell("fp_shared")])
    # ... and a consumer that REUSED it (pointer, not a fresh source)
    reused = {"arm": "off", "mean_reward": 1.0, "reused_from_run_id": "MINT_A",
              "reused_fingerprint": "fp_shared"}
    _write_manifest(exp_dir, "CONSUMER", [reused, _source_cell("fp_treatment")])
    idx = bei._write_arm_fingerprint_index(exp_dir, "now")
    # the reused cell must NOT create a second source entry for fp_shared
    assert idx["by_fingerprint"]["fp_shared"]["run_id"] == "MINT_A"
    assert idx["n_reused_cells"] == 1
    assert idx["reverse_index"]["MINT_A"] == ["CONSUMER"]
    assert idx["pending_reuse_revalidation"] == []


def test_index_writer_flags_pending_when_source_superseded(indexer_tree):
    bei, exp_dir = indexer_tree
    _write_manifest(exp_dir, "MINT_A", [_source_cell("fp_shared")],
                    evidence_direction="superseded")
    reused = {"arm": "off", "reused_from_run_id": "MINT_A", "reused_fingerprint": "fp_shared"}
    _write_manifest(exp_dir, "CONSUMER", [reused])
    idx = bei._write_arm_fingerprint_index(exp_dir, "now")
    pend = idx["pending_reuse_revalidation"]
    assert len(pend) == 1
    assert pend[0]["consumer_run_id"] == "CONSUMER"
    assert pend[0]["source_run_id"] == "MINT_A"
    assert pend[0]["reason"] == "source_superseded"


def test_index_writer_flags_pending_when_source_missing(indexer_tree):
    bei, exp_dir = indexer_tree
    reused = {"arm": "off", "reused_from_run_id": "GONE", "reused_fingerprint": "fp_x"}
    _write_manifest(exp_dir, "CONSUMER", [reused])
    idx = bei._write_arm_fingerprint_index(exp_dir, "now")
    assert idx["pending_reuse_revalidation"][0]["reason"] == "source_run_missing"


def test_index_writer_collapse_prefers_newest(indexer_tree):
    bei, exp_dir = indexer_tree
    _write_manifest(exp_dir, "OLD", [_source_cell("fp_dupe")], ts="20260101T000000Z")
    _write_manifest(exp_dir, "NEW", [_source_cell("fp_dupe")], ts="20260601T000000Z")
    idx = bei._write_arm_fingerprint_index(exp_dir, "now")
    assert idx["by_fingerprint"]["fp_dupe"]["run_id"] == "NEW"
    assert idx["n_fingerprints"] == 1
