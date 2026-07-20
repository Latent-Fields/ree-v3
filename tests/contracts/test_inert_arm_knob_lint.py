"""Contracts for the `inert_arm_knob` manifest-local lint.

Surfaces under test:
  (1) inert_arm_knob.check_inert_arm_knob -- fires on a declared-distinct arm pair that is
      bit-identical on every recorded per-cell field at matched seed.
  (2) inert_arm_knob.stamp_inert_arm_knob -- the record-and-WARN half.
  (3) manifest_core.stamp_recording_core -- the chokepoint carries the lint, and carrying
      it NEVER raises.

WHY THIS GATE EXISTS. V3-EXQ-689d declared four arms. `ARM_PROPOSER_CTRL` and
`ARM_MATCHED_NOISE` were bit-identical on 26 of 27 recorded per-cell fields on all three
seeds -- including `n_p1_ticks` 387/3616/224, meaning identical trajectories -- differing
only in the `temperature` field that NAMED their intended difference (1.0 vs 2.5).
`MATCHED_ENTROPY_TEMPERATURE = 2.5` was folded into `arm_fingerprint` but never reached a
sampling step, because `candidate_summary_source='proposer'` resolves by deterministic
argmin. The knob was inert.

The failure this prevents is NOT a crash, which is why it went unnoticed: the
"NOT noise-as-diversity" half of the conjunctive C_PRIMARY tested nothing, and the run
PASSED, because "strict above BOTH X and Y" degrades SILENTLY to "strict above X" when
X == Y. A conjunctive criterion does not announce that a conjunct became vacuous.

WHY THE ARM FINGERPRINT IS BLIND TO IT. `temperature` enters `config_slice`, so the two
cells' `arm_fingerprint` values DIFFER while every readout is byte-equal. The fingerprint
asserts "declared distinct"; the readouts prove "not distinct". That contradiction is the
signal, and only a lint that reads recorded VALUES can see it -- hence manifest-local,
not an AST lint in validate_experiments.py.

SCOPE / POSTURE. WARN-only, never a hard failure: by manifest-write time the compute is
spent, and refusing to write would destroy an expensive run over a defect that is
sometimes survivable (689d's other two arms were fine). Completed runs are re-adjudicated
via /failure-autopsy, never rewritten. Recommended by
failure_autopsy_V3-EXQ-689d-D3_2026-07-20.md sec 7 item 4 and
intra_run_substrate_divergence_sweep_2026-07-20.md sec 8(a) correction.
"""
import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from _lib import inert_arm_knob  # noqa: E402
from _lib import manifest_core  # noqa: E402


# --------------------------------------------------------------------------------------
# Fixtures: the real 689d shape, reduced to the two arms that collided.
# --------------------------------------------------------------------------------------

# Verbatim readout values from
# evidence/experiments/v3_exq_689d_mech448_f_eligibility_demotion_falsifier_...json,
# ARM_PROPOSER_CTRL / ARM_MATCHED_NOISE, seed 42. Every one of these was byte-equal
# across the two arms; that equality is the defect.
_689D_READOUTS_SEED42 = {
    "n_p1_ticks": 387,
    "n_contrastive_steps": 387,
    "error_note": None,
    "modulatory_channel_route_range_mean": 0.0,
    "modulatory_channel_route_range_max": 0.0,
    "modulatory_authority_active_ticks": 0,
    "modulatory_shortlist_size_mean": 3.0,
    "modulatory_shortlist_active_ticks": 387,
    "modulatory_shortlist_mode": "top_k",
    "cand_world_pairwise_dist_mean": 0.066511,
    "f_eligibility_demotion_active_ticks": 0,
    "f_eligibility_demotion_active_frac": 0.0,
    "f_eligibility_envelope_size_mean": 0.0,
    "f_eligibility_excluded_count_mean": 0.0,
    "f_eligibility_winner_neq_f_argmin_ticks": 0,
    "f_eligibility_rank_preserving_frac": 1.0,
    "selected_action_class_entropy": 1.27604,
    "selected_class_counts": {"0": 59, "1": 201, "2": 54, "3": 66, "4": 7},
    "selected_classes_n_unique": 5,
    "proposer_pool_class_entropy": 1.319557,
    "proposer_pool_classes_n_unique": 5,
    "harm_per_p1_tick_mean": 0.053824,
    "harm_p1_ticks": 387,
}

# n_p1_ticks per seed, as actually recorded -- identical BETWEEN the two arms at each
# seed, different ACROSS seeds. Pinned because identical trajectory lengths at three
# different lengths is what makes "same arm" decisive rather than coincidental.
_689D_N_P1_TICKS = {42: 387, 43: 3616, 44: 224}


def _cell(arm_id, label, seed, temperature, *, readouts=None, overrides=None):
    """One arm_results cell in the 689d shape."""
    cell = {
        "arm_id": arm_id,
        "label": label,
        "seed": seed,
        "candidate_summary_source": "proposer",
        "temperature": temperature,
        "use_f_eligibility_demotion": False,
    }
    body = dict(readouts if readouts is not None else _689D_READOUTS_SEED42)
    body["n_p1_ticks"] = _689D_N_P1_TICKS[seed]
    body["n_contrastive_steps"] = _689D_N_P1_TICKS[seed]
    body["harm_p1_ticks"] = _689D_N_P1_TICKS[seed]
    body["modulatory_shortlist_active_ticks"] = _689D_N_P1_TICKS[seed]
    cell.update(body)
    if overrides:
        cell.update(overrides)
    # The fingerprints DIFFER (temperature enters config_slice) even though every readout
    # is equal -- the contradiction the lint keys on.
    cell["arm_fingerprint"] = {
        "schema": "arm_fp/v1",
        "arm_fingerprint": "%s-%s-%s" % (arm_id, seed, temperature),
        "substrate_hash": "19b4073c41b9020256007c8378379164d854c0ac58def61b17c5afd6995afb05",
        "machine_class": "darwin-arm64-py3.13",
        "seed": seed,
    }
    return cell


_ARMS_SPEC = [
    {"arm_id": "ARM_PROPOSER_CTRL", "label": "proposer_collapsed_channel_baseline_control",
     "candidate_summary_source": "proposer", "temperature": 1.0,
     "use_f_eligibility_demotion": False},
    {"arm_id": "ARM_MATCHED_NOISE",
     "label": "proposer_matched_entropy_flat_temperature_negative_control",
     "candidate_summary_source": "proposer", "temperature": 2.5,
     "use_f_eligibility_demotion": False},
]


def _manifest_689d_signature():
    """The defect: two arms declared distinct by `temperature`, identical on everything."""
    cells = []
    for seed in (42, 43, 44):
        cells.append(_cell("ARM_PROPOSER_CTRL", _ARMS_SPEC[0]["label"], seed, 1.0))
        cells.append(_cell("ARM_MATCHED_NOISE", _ARMS_SPEC[1]["label"], seed, 2.5))
    return {
        "run_id": "v3_exq_689d_synthetic_v3",
        "config": {"arms": copy.deepcopy(_ARMS_SPEC)},
        "arm_results": cells,
    }


def _manifest_genuinely_differing():
    """The control: same declaration, but the knob actually moved the readouts."""
    manifest = _manifest_689d_signature()
    for cell in manifest["arm_results"]:
        if cell["arm_id"] == "ARM_MATCHED_NOISE":
            # A higher sampling temperature really does flatten the selected-class
            # distribution and change the trajectory.
            cell["selected_action_class_entropy"] = 1.55012
            cell["selected_class_counts"] = {"0": 88, "1": 121, "2": 79, "3": 71, "4": 28}
            cell["cand_world_pairwise_dist_mean"] = 0.094322
    return manifest


# --------------------------------------------------------------------------------------
# (1) The lint fires on the 689d signature.
# --------------------------------------------------------------------------------------

def test_fires_on_689d_signature():
    report = inert_arm_knob.check_inert_arm_knob(_manifest_689d_signature())
    assert report["checked"] is True
    assert report["arm_knobs_effective"] is False, (
        "the 689d D2 signature (two arms bit-identical except the knob naming their "
        "difference) MUST fire -- this is the whole point of the gate"
    )
    assert len(report["findings"]) == 1
    finding = report["findings"][0]
    assert sorted(finding["arm_ids"]) == ["ARM_MATCHED_NOISE", "ARM_PROPOSER_CTRL"]
    assert finding["seeds_compared"] == [42, 43, 44]


def test_names_the_inert_knob_and_only_it():
    """The report must NAME `temperature` -- an unattributed 'these arms match' warning
    does not tell the author which declared axis was inert."""
    report = inert_arm_knob.check_inert_arm_knob(_manifest_689d_signature())
    knobs = report["findings"][0]["differing_knobs"]
    assert list(knobs) == ["temperature"], (
        "temperature is the ONLY knob that differed; candidate_summary_source and "
        "use_f_eligibility_demotion were equal across the pair"
    )
    assert sorted(knobs["temperature"]) == [1.0, 2.5]


def test_identical_readouts_are_reported_not_just_counted():
    """The identical set is recorded so an adjudicator can see WHAT matched without
    re-deriving it from the manifest."""
    finding = inert_arm_knob.check_inert_arm_knob(_manifest_689d_signature())["findings"][0]
    identical = set(finding["identical_fields"])
    # n_p1_ticks matching at three DIFFERENT lengths is what makes this decisive.
    assert "n_p1_ticks" in identical
    assert "selected_action_class_entropy" in identical
    assert "selected_class_counts" in identical, "nested dicts must compare by content"
    # The knob itself is excluded from the comparison, never counted as a match.
    assert "temperature" not in identical
    assert finding["n_fields_compared"] == len(identical)
    assert finding["n_fields_compared"] >= inert_arm_knob.MIN_COMPARED_FIELDS


# --------------------------------------------------------------------------------------
# (2) The lint stays silent on a genuinely-differing pair.
# --------------------------------------------------------------------------------------

def test_silent_on_genuinely_differing_arms():
    report = inert_arm_knob.check_inert_arm_knob(_manifest_genuinely_differing())
    assert report["checked"] is True
    assert report["arm_knobs_effective"] is True
    assert report["findings"] == []


def test_one_diverging_seed_clears_the_pair():
    """A knob that moved the readouts at even ONE seed is not inert. Firing anyway would
    make the lint a nuisance on stochastic runs, and a nuisance warning gets ignored --
    taking the real case with it (the sweep's own argument against the whole-glob check)."""
    manifest = _manifest_689d_signature()
    for cell in manifest["arm_results"]:
        if cell["arm_id"] == "ARM_MATCHED_NOISE" and cell["seed"] == 44:
            cell["selected_action_class_entropy"] = 1.61
    report = inert_arm_knob.check_inert_arm_knob(manifest)
    assert report["arm_knobs_effective"] is True


def test_silent_when_no_knob_differs():
    """Two arms identical on readouts AND on every knob were never DECLARED distinct on
    any recorded field, so this lint has nothing to say about them."""
    manifest = _manifest_689d_signature()
    for cell in manifest["arm_results"]:
        cell["temperature"] = 1.0
    for spec in manifest["config"]["arms"]:
        spec["temperature"] = 1.0
    report = inert_arm_knob.check_inert_arm_knob(manifest)
    assert report["arm_knobs_effective"] is True


# --------------------------------------------------------------------------------------
# Scope, derivation and suppression.
# --------------------------------------------------------------------------------------

def test_absent_config_arms_is_out_of_scope_not_inferred():
    """PINS THE FALSE-POSITIVE FIX. Without an authoritative `config['arms']` declaration
    the lint must report out-of-scope rather than infer knobs from cell values.

    The tempting inference -- "constant within every arm, varying across arms" -- was
    built, swept over the 648-manifest corpus, and rejected: 5 fires, 4 false positives.
    It fails by self-fulfilling exclusion, misclassifying a READOUT as a knob, dropping it
    from the comparison, and then reporting the remainder as identical. 603j is the clean
    case: `mean_fed_safety_signal` 0.0 vs 0.89 proves the arms diverged loudly, yet the
    inference calls it a knob and reports the pair inert. Single-seed runs (590c) are
    degenerate -- "constant within arm" is vacuously true of every field.

    This is the sweep correction's own lesson: a warning that is usually wrong gets
    ignored and takes the real case with it. The resulting false negative is stated, not
    hidden -- 52 of 100 multi-arm manifests lack the declaration."""
    manifest = _manifest_689d_signature()
    manifest.pop("config")
    report = inert_arm_knob.check_inert_arm_knob(manifest)
    assert report["checked"] is False
    assert report["arm_knobs_effective"] is True
    assert report["findings"] == []
    assert "config['arms']" in report["reason"]


def test_readout_misclassified_as_knob_cannot_manufacture_a_finding():
    """The 603j shape: the arms differ loudly on readouts, and only the DECLARED knob set
    governs what is excluded. A readout is never excludable, so the pair clears."""
    manifest = _manifest_689d_signature()
    for cell in manifest["arm_results"]:
        # A readout that is constant within each arm and differs across them -- exactly
        # what the rejected inference would have mistaken for a knob.
        cell["mean_fed_safety_signal"] = 0.0 if cell["arm_id"] == "ARM_PROPOSER_CTRL" else 0.89274
    report = inert_arm_knob.check_inert_arm_knob(manifest)
    assert report["arm_knobs_effective"] is True, (
        "mean_fed_safety_signal is not in config['arms'], so it is compared, not excluded"
    )


def test_errored_cells_are_skipped():
    """Two crashed cells look identical for uninteresting reasons."""
    manifest = _manifest_689d_signature()
    for cell in manifest["arm_results"]:
        cell["error_note"] = "CUDA OOM"
    report = inert_arm_knob.check_inert_arm_knob(manifest)
    assert report["checked"] is False
    assert report["arm_knobs_effective"] is True


def test_single_arm_and_empty_manifests_are_out_of_scope():
    for manifest in ({}, {"arm_results": []}, {"arm_results": [_cell("A", "a", 42, 1.0)]}):
        report = inert_arm_knob.check_inert_arm_knob(manifest)
        assert report["checked"] is False
        assert report["arm_knobs_effective"] is True


def test_expected_identical_pairs_suppress_the_finding():
    """A duplicate control declared as such is a true statement about the manifest, not a
    defect. Suppression is per-PAIR and named -- a blanket off-switch would silence the
    real case too."""
    manifest = _manifest_689d_signature()
    manifest[inert_arm_knob.EXPECTED_IDENTICAL_KEY] = [
        ["ARM_PROPOSER_CTRL", "ARM_MATCHED_NOISE"]
    ]
    report = inert_arm_knob.check_inert_arm_knob(manifest)
    assert report["arm_knobs_effective"] is True


def test_thin_pairs_do_not_fire():
    """Agreeing on one or two bookkeeping fields is not evidence of a shared trajectory."""
    manifest = {
        "config": {"arms": [{"arm_id": "A", "temperature": 1.0},
                            {"arm_id": "B", "temperature": 2.5}]},
        "arm_results": [
            {"arm_id": "A", "seed": 42, "temperature": 1.0, "n_ticks": 10},
            {"arm_id": "B", "seed": 42, "temperature": 2.5, "n_ticks": 10},
        ],
    }
    report = inert_arm_knob.check_inert_arm_knob(manifest)
    assert report["arm_knobs_effective"] is True


def test_pairs_sharing_no_seed_do_not_fire():
    """Identity is only meaningful at MATCHED seed. Comparing arm A seed 42 against arm B
    seed 99 would be comparing two different draws."""
    manifest = {
        "config": {"arms": [{"arm_id": "A", "temperature": 1.0},
                            {"arm_id": "B", "temperature": 2.5}]},
        "arm_results": [
            {"arm_id": "A", "seed": 42, "temperature": 1.0, "x": 1, "y": 2, "z": 3},
            {"arm_id": "B", "seed": 99, "temperature": 2.5, "x": 1, "y": 2, "z": 3},
        ],
    }
    assert inert_arm_knob.check_inert_arm_knob(manifest)["arm_knobs_effective"] is True


def test_nan_readouts_compare_as_identical():
    """A TRAP worth pinning: Python's `nan != nan` would clear an otherwise-identical pair,
    silently converting the 689d defect into a pass on any run recording a NaN readout.
    Comparison goes through canonical JSON, so NaN is bit-identical to NaN."""
    manifest = {
        "config": {"arms": [{"arm_id": "A", "temperature": 1.0},
                            {"arm_id": "B", "temperature": 2.5}]},
        "arm_results": [
            {"arm_id": "A", "seed": 42, "temperature": 1.0,
             "x": None, "y": float("nan"), "z": 3},
            {"arm_id": "B", "seed": 42, "temperature": 2.5,
             "x": None, "y": float("nan"), "z": 3},
        ],
    }
    assert inert_arm_knob.check_inert_arm_knob(manifest)["arm_knobs_effective"] is False


def test_only_the_colliding_pair_is_reported():
    """A third arm that genuinely differs must not be dragged into the finding."""
    manifest = {
        "config": {"arms": [{"arm_id": "A", "t": 1}, {"arm_id": "B", "t": 2},
                            {"arm_id": "C", "t": 3}]},
        "arm_results": [
            {"arm_id": "A", "seed": 1, "t": 1, "x": 1, "y": 2, "z": 3},
            {"arm_id": "B", "seed": 1, "t": 2, "x": 1, "y": 2, "z": 3},
            {"arm_id": "C", "seed": 1, "t": 3, "x": 9, "y": 9, "z": 9},
        ],
    }
    report = inert_arm_knob.check_inert_arm_knob(manifest)
    assert [f["arm_ids"] for f in report["findings"]] == [["A", "B"]]


def test_lint_never_raises_on_malformed_input():
    for manifest in (
        {"arm_results": "not-a-list"},
        {"arm_results": [None, 42, "x"]},
        {"arm_results": [{"arm_id": "A", "seed": 1}], "config": "not-a-mapping"},
        {"arm_results": [{"no_arm_id": True}]},
    ):
        report = inert_arm_knob.check_inert_arm_knob(manifest)
        assert report["arm_knobs_effective"] is True


# --------------------------------------------------------------------------------------
# (3) Record-and-WARN posture, and the manifest_core chokepoint.
# --------------------------------------------------------------------------------------

def test_stamp_records_verdict_and_detail(capsys):
    manifest = _manifest_689d_signature()
    inert_arm_knob.stamp_inert_arm_knob(manifest)
    assert manifest["arm_knobs_effective"] is False
    assert "inert_arm_knob_detail" in manifest
    assert manifest["inert_arm_knob_detail"]["findings"][0]["differing_knobs"]
    captured = capsys.readouterr()
    assert inert_arm_knob.WARN_PREFIX in captured.out
    assert inert_arm_knob.WARN_PREFIX in captured.err, (
        "double print: stderr for the operator, stdout because the runner captures "
        "stdout into recent_lines"
    )


def test_stamp_emits_detail_only_on_the_bad_verdict():
    """Offenders-only, same shape discipline as `degenerate_metrics`."""
    manifest = _manifest_genuinely_differing()
    inert_arm_knob.stamp_inert_arm_knob(manifest)
    assert manifest["arm_knobs_effective"] is True
    assert "inert_arm_knob_detail" not in manifest


def test_warning_text_is_ascii():
    """Repo rule: anything reaching a terminal is ASCII (cp1252 mojibake otherwise)."""
    report = inert_arm_knob.check_inert_arm_knob(_manifest_689d_signature())
    text = inert_arm_knob.format_warning(report["findings"])
    text.encode("ascii")  # raises UnicodeEncodeError if violated
    assert "temperature" in text


def test_stamp_never_hard_fails_on_the_defect():
    """WARN, never raise. By manifest-write time the compute is spent, and 689d's other
    arms were fine -- refusing to write would destroy a survivable run."""
    manifest = _manifest_689d_signature()
    returned = inert_arm_knob.stamp_inert_arm_knob(manifest)  # must not raise
    assert returned is manifest


def test_manifest_core_chokepoint_carries_the_lint():
    """Every manifest reaching pack_writer.write_flat_manifest passes through
    stamp_recording_core, so the lint must be wired there and must not raise."""
    manifest = _manifest_689d_signature()
    manifest_core.stamp_recording_core(manifest, config=manifest.get("config"), seeds=[42, 43, 44])
    assert manifest["arm_knobs_effective"] is False
    assert manifest["inert_arm_knob_detail"]["findings"][0]["arm_ids"]


def test_manifest_core_chokepoint_clean_case():
    manifest = _manifest_genuinely_differing()
    manifest_core.stamp_recording_core(manifest, config=manifest.get("config"), seeds=[42, 43, 44])
    assert manifest["arm_knobs_effective"] is True
    assert "inert_arm_knob_detail" not in manifest


def test_lint_is_not_in_always_core_keys():
    """The pre-2026-07-20 corpus cannot carry it; making it core would turn every legacy
    manifest into a WARN. Same reasoning as substrate_stable_across_run."""
    assert "arm_knobs_effective" not in manifest_core.ALWAYS_CORE_KEYS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
