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

    # substrate_commit is the one always-core field with an ENVIRONMENTAL
    # dependency: it needs a real git checkout. Every production run has one (the
    # runner pulls before executing), but the tree `scripts/remote_pytest.sh`
    # stages deliberately excludes `.git/`, so the field is legitimately absent
    # there. Assert the strong property when git is present and the exact,
    # single-field shortfall when it is not -- rather than relaxing to a
    # tautology, which would stop this test noticing a real regression in any of
    # the other six core fields.
    if (_REE_V3_ROOT / ".git").exists():
        assert isinstance(m["substrate_commit"], dict)
        assert len(m["substrate_commit"]["commit"]) == 40
        assert mc.missing_core_fields(m) == []
    else:
        assert mc.missing_core_fields(m) == ["substrate_commit"]

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


# --------------------------------------------------------------------------- #
# MANDATORY_CORE_KEYS / missing_mandatory_core_fields (2026-08-12 hardening,
# chip-20260812-recording-standard-mandatory-provenance)
# --------------------------------------------------------------------------- #

def test_mandatory_core_keys_is_a_subset_of_always_core_keys():
    assert set(mc.MANDATORY_CORE_KEYS) <= set(mc.ALWAYS_CORE_KEYS)
    # the caller-supplied tier (needs a kwarg from the experiment script) is
    # deliberately NOT hard-enforced yet -- see MANDATORY_CORE_KEYS' docstring.
    assert "elapsed_seconds" not in mc.MANDATORY_CORE_KEYS
    assert "config" not in mc.MANDATORY_CORE_KEYS
    assert "seeds" not in mc.MANDATORY_CORE_KEYS
    # substrate_commit needs a real git checkout (unlike substrate_hash, a pure
    # file-content hash) and is legitimately absent on scripts/remote_pytest.sh's
    # staged tree, which deliberately excludes .git/ -- confirmed live 2026-08-12
    # when this WAS in MANDATORY_CORE_KEYS and broke 10 tests on exactly that tree.
    assert "substrate_commit" not in mc.MANDATORY_CORE_KEYS


def test_missing_mandatory_core_fields_reports_gaps():
    assert set(mc.missing_mandatory_core_fields({})) == set(mc.MANDATORY_CORE_KEYS)


def test_missing_mandatory_core_fields_empty_when_stamped():
    # Unlike missing_core_fields (the full ALWAYS_CORE_KEYS list, which includes
    # substrate_commit and so needs the git-presence guard seen above),
    # MANDATORY_CORE_KEYS excludes substrate_commit precisely so this holds
    # unconditionally -- no git-checkout guard needed, including on
    # scripts/remote_pytest.sh's staged tree which deliberately has none.
    m = {"run_id": "v3_exq_x_v3"}
    mc.stamp_recording_core(m, config={"a": 1}, seeds=[0], elapsed_seconds=1.0)
    assert mc.missing_mandatory_core_fields(m) == []


def test_stamp_never_crashes_on_bad_repo_root(tmp_path):
    # An unresolvable substrate glob must not raise -- provenance stamping is
    # best-effort (a missing substrate_hash is a soft-validate WARN, not a failure).
    m = {}
    mc.stamp_recording_core(m, config={"a": 1}, seeds=[0], repo_root=tmp_path)
    # everything except substrate_hash still stamped
    assert m["recording_schema"] == mc.RECORDING_SCHEMA
    assert m["config"] == {"a": 1}


# --------------------------------------------------------------------------- #
# enabled_default_off_flags (2026-08-03, substrate_stability_and_drift_detection_plan
# section 6 -- prospective-only recording of REEConfig's actual default-off values,
# consumed by REE_assembly/scripts/check_substrate_staleness_candidates.py)
# --------------------------------------------------------------------------- #

def _make_fake_dataclass():
    """Minimal nested-dataclass fixtures -- avoids a torch/ree_core REEConfig import
    in a test file that otherwise stays scalar-only, matching this module's own
    stdlib-only posture."""
    import dataclasses

    @dataclasses.dataclass
    class FakeSub:
        use_thing: bool = False
        level: int = 0

    @dataclasses.dataclass
    class FakeConfig:
        use_top: bool = False
        threshold: float = 0.0
        default_on: bool = True
        sub: FakeSub = dataclasses.field(default_factory=FakeSub)

    return FakeConfig, FakeSub


def test_enabled_default_off_flags_no_changes_is_empty():
    FakeConfig, _ = _make_fake_dataclass()
    assert mc.enabled_default_off_flags(FakeConfig()) == {}


def test_enabled_default_off_flags_top_level_change():
    FakeConfig, _ = _make_fake_dataclass()
    cfg = FakeConfig(use_top=True)
    assert mc.enabled_default_off_flags(cfg) == {"use_top": True}


def test_enabled_default_off_flags_ignores_default_on_field_changing():
    # default_on's coded default is True -- flipping it to False is a real change but
    # NOT a "default-off knob was enabled" event, so it must not appear here.
    FakeConfig, _ = _make_fake_dataclass()
    cfg = FakeConfig(default_on=False)
    assert mc.enabled_default_off_flags(cfg) == {}


def test_enabled_default_off_flags_recurses_into_nested_dataclass_with_dotted_name():
    FakeConfig, FakeSub = _make_fake_dataclass()
    cfg = FakeConfig(sub=FakeSub(use_thing=True))
    assert mc.enabled_default_off_flags(cfg) == {"sub.use_thing": True}


def test_enabled_default_off_flags_float_zero_default():
    FakeConfig, _ = _make_fake_dataclass()
    cfg = FakeConfig(threshold=0.5)
    assert mc.enabled_default_off_flags(cfg) == {"threshold": 0.5}


def test_enabled_default_off_flags_non_dataclass_input_is_empty():
    assert mc.enabled_default_off_flags(None) == {}
    assert mc.enabled_default_off_flags("not a dataclass") == {}
    assert mc.enabled_default_off_flags(object()) == {}


def test_enabled_default_off_flags_for_agents_single_and_none():
    FakeConfig, _ = _make_fake_dataclass()

    class A:
        def __init__(self, cfg):
            self.config = cfg

    # None -- deliberately, not {} -- distinguishing "never measured" from "measured,
    # nothing enabled" is the whole point of this function; see its own docstring.
    assert mc.enabled_default_off_flags_for_agents(None) is None
    assert mc.enabled_default_off_flags_for_agents(A(FakeConfig(use_top=True))) == {"use_top": True}


def test_enabled_default_off_flags_for_agents_measured_but_empty_is_not_none():
    FakeConfig, _ = _make_fake_dataclass()

    class A:
        def __init__(self, cfg):
            self.config = cfg

    result = mc.enabled_default_off_flags_for_agents(A(FakeConfig()))
    assert result == {}
    assert result is not None


def test_enabled_default_off_flags_for_agents_pools_multiple():
    FakeConfig, FakeSub = _make_fake_dataclass()

    class A:
        def __init__(self, cfg):
            self.config = cfg

    agents = [A(FakeConfig(use_top=True)), A(FakeConfig(sub=FakeSub(use_thing=True)))]
    pooled = mc.enabled_default_off_flags_for_agents(agents)
    assert pooled == {"use_top": True, "sub.use_thing": True}


def test_stamp_recording_core_records_enabled_flags_when_agent_given():
    FakeConfig, _ = _make_fake_dataclass()

    class A:
        def __init__(self, cfg):
            self.config = cfg

    m = {}
    mc.stamp_recording_core(m, config={"a": 1}, agent=A(FakeConfig(use_top=True)))
    assert m["enabled_default_off_flags"] == {"use_top": True}


def test_stamp_recording_core_omits_enabled_flags_without_agent():
    # Omitted, not {} -- presence must always mean "measured" (same convention as
    # z_goal_stream), so a caller that never passed an agent must see the key absent.
    m = {}
    mc.stamp_recording_core(m, config={"a": 1})
    assert "enabled_default_off_flags" not in m


def test_stamp_recording_core_records_empty_dict_when_agent_given_but_nothing_differs():
    # This is the case the earlier "omit on empty" draft got wrong: an agent WAS given,
    # so this must be recorded as {} (measured, nothing enabled), not omitted (which
    # would be indistinguishable from never having passed an agent at all).
    FakeConfig, _ = _make_fake_dataclass()

    class A:
        def __init__(self, cfg):
            self.config = cfg

    m = {}
    mc.stamp_recording_core(m, config={"a": 1}, agent=A(FakeConfig()))
    assert m["enabled_default_off_flags"] == {}


def test_stamp_recording_core_enabled_flags_no_op_safe_fill_only():
    FakeConfig, _ = _make_fake_dataclass()

    class A:
        def __init__(self, cfg):
            self.config = cfg

    m = {"enabled_default_off_flags": {"author_set": True}}
    mc.stamp_recording_core(m, config={"a": 1}, agent=A(FakeConfig(use_top=True)))
    assert m["enabled_default_off_flags"] == {"author_set": True}  # not clobbered

    mc.stamp_recording_core(
        m, config={"a": 1}, agent=A(FakeConfig(use_top=True)), overwrite=True)
    assert m["enabled_default_off_flags"] == {"use_top": True}  # overwrite=True wins


# --------------------------------------------------------------------------- #
# write_flat_manifest MANDATORY-core hard enforcement (2026-08-12 hardening,
# chip-20260812-recording-standard-mandatory-provenance)
# --------------------------------------------------------------------------- #

def test_write_flat_manifest_raises_when_mandatory_core_missing():
    # stamp=False with nothing pre-stamped is the one way a caller can legitimately
    # (if mistakenly) reach write_flat_manifest with the mandatory subset absent.
    tmp = Path(tempfile.mkdtemp())
    manifest = {"run_id": "wfm_missing_core_v3", "outcome": "PASS"}
    with pytest.raises(ValueError, match="mandatory always-core"):
        pw.write_flat_manifest(manifest, tmp, stamp=False)


def test_write_flat_manifest_still_writes_the_file_before_raising():
    """The whole point of checking AFTER the write: a run's already-spent compute
    must survive even when its provenance is incomplete (manifest_core.py's
    stated posture for every OTHER sub-stamper, now honoured here too)."""
    tmp = Path(tempfile.mkdtemp())
    manifest = {"run_id": "wfm_missing_core_survives_v3", "outcome": "PASS",
                "precious_result": 42}
    with pytest.raises(ValueError):
        pw.write_flat_manifest(manifest, tmp, stamp=False)
    out_path = tmp / "wfm_missing_core_survives_v3.json"
    assert out_path.is_file()
    assert json.loads(out_path.read_text())["precious_result"] == 42


def test_write_flat_manifest_passes_when_core_present():
    tmp = Path(tempfile.mkdtemp())
    manifest = {
        "run_id": "wfm_has_core_v3", "outcome": "PASS",
        "recording_schema": "rec/v1",
        "substrate_hash": "0" * 64,
        "machine": "test-host",
        "machine_class": "test-class",
        # substrate_commit deliberately absent -- it is NOT in MANDATORY_CORE_KEYS
        # (needs a real git checkout; see that constant's docstring), so this must
        # still pass without it.
    }
    out_path = pw.write_flat_manifest(manifest, tmp, stamp=False)
    assert out_path.is_file()


def test_write_flat_manifest_passes_when_stamped_from_real_repo():
    # The realistic path: stamp=True (the default) against this real checkout.
    if not (_REE_V3_ROOT / ".git").exists():
        pytest.skip("no real git checkout available (see staged-tree note above)")
    tmp = Path(tempfile.mkdtemp())
    manifest = {"run_id": "wfm_real_stamp_v3", "outcome": "PASS"}
    out_path = pw.write_flat_manifest(
        manifest, tmp, config={"a": 1}, seeds=[0], elapsed_seconds=1.0,
        script_path=_REE_V3_ROOT / "experiments" / "pack_writer.py",
    )
    assert out_path.is_file()


def test_write_flat_manifest_escape_hatch_downgrades_to_warning(monkeypatch, capsys):
    monkeypatch.setenv("REE_ALLOW_INCOMPLETE_PROVENANCE", "1")
    tmp = Path(tempfile.mkdtemp())
    manifest = {"run_id": "wfm_override_v3", "outcome": "PASS"}
    out_path = pw.write_flat_manifest(manifest, tmp, stamp=False)
    assert out_path.is_file()
    assert "WARNING (override)" in capsys.readouterr().out


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
