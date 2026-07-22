"""Contracts for the `dose_saturation` manifest-local lint.

Surfaces under test:
  (1) dose_saturation.check_dose_saturation -- fires when two DECLARED DOSE LEVELS
      produce values identical beyond float noise.
  (2) dose_saturation.stamp_dose_saturation -- the record-and-WARN half.
  (3) manifest_core.stamp_recording_core -- the chokepoint carries the lint, and carrying
      it NEVER raises.
  (4) The real V3-EXQ-794 per_level block replays to a FIRING verdict.

WHY THIS GATE EXISTS. V3-EXQ-794 ran SD-076's `waking_confidence_inflation_asymmetry` at
LO=0.6 and HI=0.8. Both levels returned `overconfidence_score` = -1.004111904519277 --
bit-identical to 15 significant figures -- and `calibration_ratio` = 2.7564936387545953.
`rv_final` finished at EXACTLY 0.010000 on all four inflation arms, because an absolute
floor (`waking_confidence_rv_floor` = 0.01) sat 1.8x ABOVE the substrate's un-inflated
operating point of 0.005420, so `max(floor, rv)` pinned every inflation arm from tick one.

The failure this prevents is NOT a crash and NOT a visible null: SD-076 was recorded
`does_not_support`, charging a refutation to a claim whose lever never moved, and
MECH-204's Phase-7 correction was left with no drift to correct. BOTH claims went
untested while appearing tested. It took a full autopsy to withdraw the direction and
revise it to `non_contributory` -- an outcome this ~40-line check produces at
manifest-write time. Recommended by failure_autopsy_V3-EXQ-794_2026-07-22.md sec 6 item 2.

WHY THE SIBLING LINT IS BLIND TO IT. `inert_arm_knob` catches declared-distinct arms that
RAN IDENTICALLY -- a knob that never reached a live path. Here the knob DID reach the path
and did move the dynamics (the 794 arms have different trajectories, and ARM_BOTH_LO
differs from ARM_INFL_LO in the 7th decimal), and a bound downstream erased the
difference. Complementary defects, so neither lint subsumes the other.

SCOPE / POSTURE. WARN-only, never a hard failure: by manifest-write time the compute is
spent, and 794's own green arms remained scorable (C4 confirmed the 774 ceiling). The
"refuse the dose-response criterion" half is honoured by emitting
`dose_levels_separable: False` for the experiment's own scoring to read.
"""
import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from _lib import dose_saturation  # noqa: E402
from _lib import manifest_core  # noqa: E402


# --------------------------------------------------------------------------------------
# Fixture: the real 794 per_level block. Values verbatim from
# evidence/experiments/v3_exq_794_mech204_phase7_sd076_calibration_loop_2x2_
# 20260721T113848Z_v3.json -- aggregates.per_level.
# --------------------------------------------------------------------------------------

_794_PER_LEVEL = {
    "LO": {
        "asymmetry": 0.6,
        "infl_arm": "ARM_INFL_LO",
        "both_arm": "ARM_BOTH_LO",
        "n_seeds_overconfident": 0,
        "clears_c1": False,
        "infl_score": -1.004111904519277,
        "calibration_ratio": 2.7564936387545953,
        "rv_final": 0.01,
    },
    "HI": {
        "asymmetry": 0.8,
        "infl_arm": "ARM_INFL_HI",
        "both_arm": "ARM_BOTH_HI",
        "n_seeds_overconfident": 0,
        "clears_c1": False,
        "infl_score": -1.004111904519277,
        "calibration_ratio": 2.7564936387545953,
        "rv_final": 0.01,
    },
}


def _manifest(per_level=None):
    return {
        "run_id": "v3_exq_794_test_v3",
        "aggregates": {"per_level": copy.deepcopy(per_level or _794_PER_LEVEL)},
    }


# --------------------------------------------------------------------------------------
# (1) The defect fires
# --------------------------------------------------------------------------------------

def test_794_saturation_signature_fires():
    """The real 794 per_level block must produce a NOT-separable verdict."""
    report = dose_saturation.check_dose_saturation(_manifest())
    assert report["checked"] is True, report["reason"]
    assert report["dose_levels_separable"] is False
    assert len(report["findings"]) == 1
    finding = report["findings"][0]
    assert finding["levels"] == ["HI", "LO"]
    assert finding["dose_key"] == "asymmetry"
    # All three tied float readouts must be named -- the autopsy called out exactly these.
    assert set(finding["tied_fields"]) == {
        "infl_score", "calibration_ratio", "rv_final"
    }


def test_dose_key_is_excluded_from_the_comparison():
    """The dose itself varies by construction; counting it would be circular."""
    report = dose_saturation.check_dose_saturation(_manifest())
    assert "asymmetry" not in report["findings"][0]["tied_fields"]


def test_separable_levels_do_not_fire():
    """A genuine dose-response -- including a genuinely NULL one -- must pass."""
    per_level = copy.deepcopy(_794_PER_LEVEL)
    # The repaired substrate's smoke values: distinct, dose-ordered, overconfident.
    per_level["LO"].update(infl_score=0.3141, calibration_ratio=1.458, rv_final=0.0025377)
    per_level["HI"].update(infl_score=0.4316, calibration_ratio=1.759, rv_final=0.0021031)
    report = dose_saturation.check_dose_saturation(_manifest(per_level))
    assert report["checked"] is True
    assert report["dose_levels_separable"] is True
    assert report["findings"] == []


def test_tiny_but_real_separation_does_not_fire():
    """A small separation is a WEAK effect, not a clamp. Only a tie is a clamp.

    This is the property the softplus floor buys: a residual saturation can shrink a
    separation but never collapse it to an exact tie, so the run stays adjudicable.
    """
    per_level = copy.deepcopy(_794_PER_LEVEL)
    per_level["HI"]["infl_score"] = -1.004111904519277 * (1 + 1e-6)
    per_level["HI"]["calibration_ratio"] = 2.7564936387545953 * (1 + 1e-6)
    per_level["HI"]["rv_final"] = 0.01 * (1 + 1e-6)
    report = dose_saturation.check_dose_saturation(_manifest(per_level))
    assert report["dose_levels_separable"] is True


# --------------------------------------------------------------------------------------
# (2) Deliberate non-firing: the false-positive sources the module docstring commits to
# --------------------------------------------------------------------------------------

def test_tied_integers_do_not_fire():
    """`n_seeds_overconfident` = 0 at both levels is how a COUNT reports 'no effect'."""
    per_level = {
        "LO": {"asymmetry": 0.6, "n_seeds_overconfident": 0, "score": 0.11},
        "HI": {"asymmetry": 0.8, "n_seeds_overconfident": 0, "score": 0.22},
    }
    report = dose_saturation.check_dose_saturation(_manifest(per_level))
    assert report["dose_levels_separable"] is True


def test_zero_ties_are_recorded_but_do_not_flip_the_verdict():
    """0.0/0.0 is overwhelmingly a not-applicable sentinel, not a clamp at zero."""
    per_level = {
        "LO": {"asymmetry": 0.6, "delta_mean": 0.0, "score": 0.11},
        "HI": {"asymmetry": 0.8, "delta_mean": 0.0, "score": 0.22},
    }
    report = dose_saturation.check_dose_saturation(_manifest(per_level))
    assert report["dose_levels_separable"] is True
    # But it must still be visible to a reader via the pairwise comparison.
    levels = dose_saturation._levels(_manifest(per_level))
    result = dose_saturation._compare_levels(levels["LO"], levels["HI"], "asymmetry")
    assert result["zero_tied_fields"] == ["delta_mean"]


def test_tied_strings_and_bools_do_not_fire():
    per_level = {
        "LO": {"asymmetry": 0.6, "mode": "soft", "clears_c1": False, "score": 0.11},
        "HI": {"asymmetry": 0.8, "mode": "soft", "clears_c1": False, "score": 0.22},
    }
    report = dose_saturation.check_dose_saturation(_manifest(per_level))
    assert report["dose_levels_separable"] is True


def test_single_level_is_out_of_scope():
    report = dose_saturation.check_dose_saturation(
        _manifest({"LO": {"asymmetry": 0.6, "score": 0.11}})
    )
    assert report["checked"] is False
    assert report["dose_levels_separable"] is True


def test_no_resolvable_dose_is_out_of_scope():
    """Without an authoritative dose the lint declines rather than guessing."""
    per_level = {
        "LO": {"score": 0.5, "other": 0.5},
        "HI": {"score": 0.5, "other": 0.5},
    }
    report = dose_saturation.check_dose_saturation(_manifest(per_level))
    assert report["checked"] is False
    assert "dose" in report["reason"]


def test_explicit_dose_key_declaration_wins():
    manifest = _manifest()
    manifest["dose_key"] = "asymmetry"
    report = dose_saturation.check_dose_saturation(manifest)
    assert report["checked"] is True
    assert report["findings"][0]["dose_source"] == "manifest.dose_key"


def test_expected_identical_opt_out_silences_a_named_pair_only():
    manifest = _manifest()
    manifest[dose_saturation.EXPECTED_IDENTICAL_KEY] = [["LO", "HI"]]
    report = dose_saturation.check_dose_saturation(manifest)
    assert report["dose_levels_separable"] is True


# --------------------------------------------------------------------------------------
# (3) Stamping, warning, and the never-raise contract
# --------------------------------------------------------------------------------------

def test_stamp_records_verdict_and_detail(capsys):
    manifest = _manifest()
    dose_saturation.stamp_dose_saturation(manifest)
    assert manifest["dose_levels_separable"] is False
    assert "dose_saturation_detail" in manifest
    out = capsys.readouterr()
    assert dose_saturation.WARN_PREFIX in out.out
    assert dose_saturation.WARN_PREFIX in out.err


def test_stamp_omits_detail_when_clean():
    per_level = {
        "LO": {"asymmetry": 0.6, "score": 0.11},
        "HI": {"asymmetry": 0.8, "score": 0.22},
    }
    manifest = _manifest(per_level)
    dose_saturation.stamp_dose_saturation(manifest)
    assert manifest["dose_levels_separable"] is True
    assert "dose_saturation_detail" not in manifest


def test_warning_text_is_ascii_only():
    """Printed output reaches Windows cp1252 terminals -- see CLAUDE.md."""
    report = dose_saturation.check_dose_saturation(_manifest())
    text = dose_saturation.format_warning(report["findings"])
    text.encode("ascii")  # raises UnicodeEncodeError on any non-ASCII character
    assert "SATURATION" in text
    assert "UNSCORED, not refuted" in text


@pytest.mark.parametrize("manifest", [
    {},
    {"aggregates": None},
    {"aggregates": {"per_level": "not-a-mapping"}},
    {"aggregates": {"per_level": {"LO": None, "HI": 3}}},
    {"aggregates": {"per_level": _794_PER_LEVEL}, "dose_key": 17},
])
def test_lint_never_raises_on_a_malformed_manifest(manifest):
    """A lint must never crash an experiment at manifest-write time."""
    report = dose_saturation.check_dose_saturation(manifest)
    assert isinstance(report["dose_levels_separable"], bool)
    dose_saturation.stamp_dose_saturation(copy.deepcopy(manifest))


# --------------------------------------------------------------------------------------
# (4) The chokepoint carries it
# --------------------------------------------------------------------------------------

def test_manifest_core_carries_the_lint():
    manifest = _manifest()
    manifest_core.stamp_recording_core(manifest)
    assert manifest["dose_levels_separable"] is False


def test_manifest_core_is_unaffected_when_out_of_scope():
    """No per_level block -> the key is not stamped at all, so the legacy corpus and
    every single-level experiment stay clean."""
    manifest = {"run_id": "v3_exq_000_test_v3"}
    manifest_core.stamp_recording_core(manifest)
    assert "dose_levels_separable" not in manifest


def test_lint_is_not_in_always_core_keys():
    """Making it core would turn every pre-2026-07-22 manifest into a WARN."""
    assert "dose_levels_separable" not in manifest_core.ALWAYS_CORE_KEYS
