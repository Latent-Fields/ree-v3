"""Contract tests for the Experimental Recording Standard hardening (2026-07-12).

Covers the two ree-v3 code-plane pieces the standard's section-4 deferred-hardening
adds:

  1. experiments/_lib/manifest_core.stamp_recording_core -- the always-record core
     stamper (standard 3b): recording_schema, substrate_hash (multi-arm HOIST vs
     single-arm compute), machine/machine_class, elapsed_seconds, config, explicit
     seeds. Asserts the no-op-safe merge (fill-only unless overwrite) and that a
     meaningful 0/False is not treated as empty.
  2. experiments/pack_writer -- the relaxed metrics writer: the scalar `values`
     block still coerces to scalars (indexer-safe), while the structured sections
     (per_seed / latent / config / timing) are stored VERBATIM beside it, so the
     sanctioned writer can carry rich readouts instead of silently dropping them.

ASCII-only. Run: pytest tests/contracts/test_recording_standard.py -q
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# conftest puts ree-v3 root on sys.path -> `experiments.*` importable.
from experiments._lib import manifest_core as mc
from experiments import pack_writer as pw

_REE_V3_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# manifest_core.stamp_recording_core
# --------------------------------------------------------------------------- #

def test_stamp_fills_always_core_single_arm():
    m = {"run_id": "v3_exq_x_v3", "outcome": "PASS"}
    mc.stamp_recording_core(
        m, config={"lr": 0.01, "episodes": 10}, seeds=[0, 1, 2],
        script_path=str(_REE_V3_ROOT / "experiments" / "pack_writer.py"),
        elapsed_seconds=12.5,
    )
    assert m["recording_schema"] == mc.RECORDING_SCHEMA
    assert isinstance(m["substrate_hash"], str) and len(m["substrate_hash"]) == 64
    assert m["machine"]
    assert m["machine_class"]
    assert m["elapsed_seconds"] == 12.5
    assert m["config"] == {"lr": 0.01, "episodes": 10}
    assert m["seeds"] == [0, 1, 2]
    assert mc.missing_core_fields(m) == []
    json.dumps(m)  # serialisable


def test_stamp_hoists_multi_arm_substrate_hash():
    sh = "deadbeef" * 8
    m = {
        "run_id": "v3_exq_y_v3",
        "arm_results": [
            {"arm_id": "A", "arm_fingerprint": {"substrate_hash": sh}},
            {"arm_id": "B", "arm_fingerprint": {"substrate_hash": "cafef00d" * 8}},
        ],
    }
    mc.stamp_recording_core(m, config={"x": 1}, seeds=7)
    # first present per-cell hash is hoisted verbatim to the top level
    assert m["substrate_hash"] == sh
    assert m["seeds"] == [7]  # scalar seed coerced to list


def test_stamp_is_no_op_safe_fill_only():
    m = {"substrate_hash": "PRESET", "machine": "custom-host", "elapsed_seconds": 0.0}
    mc.stamp_recording_core(m, config={"a": 1}, seeds=[9], elapsed_seconds=99.0)
    assert m["substrate_hash"] == "PRESET"       # not clobbered
    assert m["machine"] == "custom-host"          # not clobbered
    assert m["elapsed_seconds"] == 0.0            # a meaningful 0 is NOT empty
    assert m["config"] == {"a": 1}                # absent field filled
    assert m["seeds"] == [9]


def test_stamp_overwrite_true_forces():
    m = {"substrate_hash": "OLD", "elapsed_seconds": 1.0}
    mc.stamp_recording_core(m, seeds=[1], elapsed_seconds=5.0, overwrite=True,
                            machine="h")
    assert m["substrate_hash"] != "OLD"
    assert m["elapsed_seconds"] == 5.0


def test_stamp_elapsed_from_started_at():
    import time
    m = {}
    mc.stamp_recording_core(m, started_at=time.perf_counter() - 3.0)
    assert m["elapsed_seconds"] >= 3.0


def test_missing_core_fields_reports_gaps():
    assert set(mc.missing_core_fields({})) == set(mc.ALWAYS_CORE_KEYS)
    # empty containers count as missing; a meaningful 0 does not
    partial = {"config": {}, "seeds": [], "elapsed_seconds": 0.0}
    gaps = set(mc.missing_core_fields(partial))
    assert "config" in gaps and "seeds" in gaps
    assert "elapsed_seconds" not in gaps


def test_stamp_never_crashes_on_bad_repo_root(tmp_path):
    # An unresolvable substrate glob must not raise -- provenance stamping is
    # best-effort (a missing substrate_hash is a soft-validate WARN, not a failure).
    m = {}
    mc.stamp_recording_core(m, config={"a": 1}, seeds=[0], repo_root=tmp_path)
    # everything except substrate_hash still stamped
    assert m["recording_schema"] == mc.RECORDING_SCHEMA
    assert m["config"] == {"a": 1}


# --------------------------------------------------------------------------- #
# pack_writer structured sections
# --------------------------------------------------------------------------- #

def _writer(tmp: Path) -> pw.ExperimentPackWriter:
    return pw.ExperimentPackWriter(
        output_root=tmp, repo_root=_REE_V3_ROOT, runner_name="test", runner_version="0")


def test_pack_scalar_only_is_backward_compatible():
    tmp = Path(tempfile.mkdtemp())
    pack = _writer(tmp).write_pack(
        "exp_a", "20260712T120000_exp_a_seed0_v3", "2026-07-12T12:00:00Z", "PASS",
        {"foraging_competence": 6.05, "n_episodes": 10}, "# s",
        claim_ids_tested=["MECH-457"])
    doc = json.loads(pack.metrics_path.read_text())
    # no structured sections when none supplied -- byte-shape identical to legacy
    assert set(doc.keys()) == {"schema_version", "values"}
    assert doc["values"] == {"foraging_competence": 6.05, "n_episodes": 10}


def test_pack_structured_sections_stored_verbatim():
    tmp = Path(tempfile.mkdtemp())
    pack = _writer(tmp).write_pack(
        "exp_b", "20260712T120001_exp_b_seed0_v3", "2026-07-12T12:00:01Z", "PASS",
        {"mean_return": 1.5}, "# s",
        per_seed=[{"seed": 0, "return": 1.4}, {"seed": 1, "return": 1.6}],
        latent={"zworld_eff_rank": 12, "e2_forward_r2": 0.83},
        config={"lr": 0.01, "episodes": 10},
        timing={"elapsed_seconds": 42.0})
    doc = json.loads(pack.metrics_path.read_text())
    assert doc["values"] == {"mean_return": 1.5}           # scalars still scalar
    assert doc["per_seed"] == [{"seed": 0, "return": 1.4}, {"seed": 1, "return": 1.6}]
    assert doc["latent"] == {"zworld_eff_rank": 12, "e2_forward_r2": 0.83}
    assert doc["config"] == {"lr": 0.01, "episodes": 10}
    assert doc["timing"] == {"elapsed_seconds": 42.0}


def test_pack_values_still_reject_nonscalar():
    # The scalar `values` contract is UNCHANGED -- a nested value there still raises,
    # forcing rich readouts into the structured sections where they belong.
    with pytest.raises(TypeError):
        pw._clean_numeric_metrics({"m": {"nested": 1}})


def test_clean_structured_sections_validation():
    with pytest.raises(TypeError):
        pw._clean_structured_sections(per_seed="notalist")
    with pytest.raises(ValueError):
        pw._clean_structured_sections(bogus={"x": 1})
    with pytest.raises(ValueError):
        pw._clean_structured_sections(latent={1: "int-key"})
    # None sections are dropped; valid ones pass through
    out = pw._clean_structured_sections(per_seed=[{"s": 0}], latent=None, config={"a": 1})
    assert out == {"per_seed": [{"s": 0}], "config": {"a": 1}}
