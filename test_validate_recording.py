"""Tests for validate_recording.check_pack_provenance -- the thin-pack
recording-provenance regression check (2026-07-16).

A runs/<run_id>/manifest.json PACK that drops machine/machine_class/substrate_hash
while its flat sibling evidence/experiments/<run_id>.json carries them is the
thin-pack bug: the index-scored pack reads machine_class=null even though the
provenance was recorded. check_pack_provenance flags exactly that case.

Also covers validate_recording.check_flat_scalar_readout -- the flat-scalar-readout
check (Experimental Recording Standard 3b "Machine-readable verdict readout",
2026-09-09). A manifest with no NUMERIC metrics.values can fire no `fail_if` stop
threshold (final_status falls back to its own self-declared status), is skipped by
the duplicate-emission supersession fingerprint, and carries no deltas.
"""
import json
import pathlib
import tempfile
import shutil
import unittest

import validate_recording as vr


class CheckPackProvenance(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="validate_rec_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.evidence = self.tmp / "evidence" / "experiments"
        self.evidence.mkdir(parents=True)

    def _make(self, run_id, exp, pack_extra, flat_extra):
        run_dir = self.evidence / exp / "runs" / run_id
        run_dir.mkdir(parents=True)
        pack = {"schema_version": "experiment_pack/v1", "run_id": run_id}
        pack.update(pack_extra)
        (run_dir / "manifest.json").write_text(json.dumps(pack), encoding="utf-8")
        flat = {"run_id": run_id}
        flat.update(flat_extra)
        (self.evidence / f"{run_id}.json").write_text(
            json.dumps(flat), encoding="utf-8")
        return run_dir / "manifest.json"

    def test_thin_pack_with_provenanced_flat_is_flagged(self):
        pack_path = self._make(
            "v3_exq_766_x_20260716T152044Z_v3", "v3_exq_766_x",
            pack_extra={},  # thin
            flat_extra={"machine": "ree-cloud-2",
                        "machine_class": "linux-x86_64-py3.10",
                        "substrate_hash": "f92a600cf17a"})
        dropped = vr.check_pack_provenance(pack_path)
        self.assertEqual(
            sorted(dropped), ["machine", "machine_class", "substrate_hash"])

    def test_healthy_pack_not_flagged(self):
        pack_path = self._make(
            "v3_exq_766_y_20260716T152044Z_v3", "v3_exq_766_y",
            pack_extra={"machine": "ree-cloud-2",
                        "machine_class": "linux-x86_64-py3.10",
                        "substrate_hash": "f92a600cf17a"},
            flat_extra={"machine": "ree-cloud-2",
                        "machine_class": "linux-x86_64-py3.10",
                        "substrate_hash": "f92a600cf17a"})
        self.assertEqual(vr.check_pack_provenance(pack_path), [])

    def test_legacy_flat_without_provenance_not_flagged(self):
        # A genuinely old run where NEITHER copy carries provenance is not the
        # bug -- nothing to recover, so it must not be flagged.
        pack_path = self._make(
            "v3_exq_100_z_20260328T120000Z_v3", "v3_exq_100_z",
            pack_extra={}, flat_extra={"evidence_direction": "supports"})
        self.assertEqual(vr.check_pack_provenance(pack_path), [])

    def test_partial_drop_flags_only_missing_keys(self):
        # Pack carries machine but drops machine_class/substrate_hash.
        pack_path = self._make(
            "v3_exq_200_p_20260328T120000Z_v3", "v3_exq_200_p",
            pack_extra={"machine": "ree-cloud-2"},
            flat_extra={"machine": "ree-cloud-2",
                        "machine_class": "linux-x86_64-py3.10",
                        "substrate_hash": "abc"})
        self.assertEqual(
            sorted(vr.check_pack_provenance(pack_path)),
            ["machine_class", "substrate_hash"])

    def test_missing_flat_sibling_is_noop(self):
        run_dir = self.evidence / "v3_exq_nf" / "runs" / "v3_exq_nf_20260328T120000Z_v3"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text('{"run_id": "x"}', encoding="utf-8")
        self.assertEqual(vr.check_pack_provenance(run_dir / "manifest.json"), [])

    def test_non_pack_path_is_noop(self):
        flat = self.evidence / "v3_exq_flat_20260328T120000Z_v3.json"
        flat.write_text('{"machine_class": "linux-x86_64-py3.10"}', encoding="utf-8")
        self.assertEqual(vr.check_pack_provenance(flat), [])


class CheckFlatScalarReadout(unittest.TestCase):
    """validate_recording.check_flat_scalar_readout -- standard 3b."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="validate_rec_ro_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.evidence = self.tmp / "evidence" / "experiments"
        self.evidence.mkdir(parents=True)

    def _flat(self, name, doc):
        p = self.evidence / f"{name}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def _pack(self, run_id, exp, values):
        run_dir = self.evidence / exp / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "outcome": "PASS"}), encoding="utf-8")
        if values is not None:
            (run_dir / "metrics.json").write_text(
                json.dumps({"schema_version": "metrics/v1", "values": values}),
                encoding="utf-8")
        return run_dir / "manifest.json"

    # --- the healthy shape -------------------------------------------------
    def test_numeric_readout_is_clean(self):
        p = self._flat("v3_exq_x_v3", {"run_id": "v3_exq_x_v3", "outcome": "PASS",
                                       "readout": {"max_sweep_dv": 0.31, "p2_bar": 0.25}})
        self.assertIsNone(vr.check_flat_scalar_readout(p))

    def test_each_recognised_spelling_satisfies(self):
        for spelling in ("readout", "metrics", "aggregates", "summary_metrics"):
            p = self._flat(f"v3_exq_{spelling}_v3",
                           {"run_id": f"v3_exq_{spelling}_v3", "outcome": "FAIL",
                            spelling: {"n": 3}})
            self.assertIsNone(vr.check_flat_scalar_readout(p), spelling)

    # --- absent ------------------------------------------------------------
    def test_nested_only_readout_is_absent(self):
        # The V3-EXQ-1015 shape: rich blocks, every one keyed by arm or seed, so
        # NONE of the four flat spellings matches and the pack scores values={}.
        p = self._flat("v3_exq_nested_v3", {
            "run_id": "v3_exq_nested_v3", "outcome": "FAIL",
            "arm_results": [{"arm_id": "A", "dv": 0.2}],
            "cell_summary": {"A": {"dv": 0.2}},
            "per_arm_gate": {"green_arms": ["A"]}})
        verdict = vr.check_flat_scalar_readout(p)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict[0], "absent")

    def test_empty_block_is_absent(self):
        p = self._flat("v3_exq_empty_v3",
                       {"run_id": "v3_exq_empty_v3", "outcome": "PASS", "readout": {}})
        self.assertEqual(vr.check_flat_scalar_readout(p)[0], "absent")

    # --- present but no numeric entry survives _is_number ------------------
    def test_bool_only_readout_is_no_numeric(self):
        # V3-EXQ-484 / V3-EXQ-177 shape: the criterion verdicts the run turns on,
        # recorded as raw bools. _is_number excludes bool, so they are recorded
        # and invisible -- the block scores as if it were empty.
        p = self._flat("v3_exq_bools_v3", {
            "run_id": "v3_exq_bools_v3", "outcome": "FAIL",
            "metrics": {"C1_task_informative": True, "C2_replay_frozen": False}})
        verdict, where, notes = vr.check_flat_scalar_readout(p)
        self.assertEqual(verdict, "no_numeric")
        self.assertEqual(where, "metrics")
        self.assertTrue(any("raw bool" in n for n in notes))

    def test_nan_only_readout_is_no_numeric(self):
        p = self._flat("v3_exq_nan_v3", {
            "run_id": "v3_exq_nan_v3", "outcome": "FAIL",
            "summary_metrics": {"spike": float("nan")}})
        verdict, where, notes = vr.check_flat_scalar_readout(p)
        self.assertEqual(verdict, "no_numeric")
        self.assertTrue(any("non-finite" in n for n in notes))

    # --- scores, but carries inert / harmful entries -----------------------
    def test_numeric_plus_bool_reports_encoding_only(self):
        # V3-EXQ-484's real shape: 4 numeric AND 4 raw bools. The block scores,
        # so this is not a gap -- but the four criterion verdicts are inert.
        p = self._flat("v3_exq_mixed_v3", {
            "run_id": "v3_exq_mixed_v3", "outcome": "FAIL",
            "metrics": {"delta": 0.4, "C1_task_informative": True}})
        verdict, where, notes = vr.check_flat_scalar_readout(p)
        self.assertEqual(verdict, "encoding_only")
        self.assertTrue(any("C1_task_informative" in n for n in notes))

    def test_numeric_plus_nan_reports_encoding_only(self):
        # V3-EXQ-165's real shape: 16 numeric + 2 nan. A nan IS numeric to the
        # indexer, so it pollutes any delta computed from it.
        p = self._flat("v3_exq_mixnan_v3", {
            "run_id": "v3_exq_mixnan_v3", "outcome": "FAIL",
            "summary_metrics": {"ok": 1.0, "spike": float("nan")}})
        verdict, _, notes = vr.check_flat_scalar_readout(p)
        self.assertEqual(verdict, "encoding_only")
        self.assertTrue(any("non-finite" in n for n in notes))

    def test_null_entry_reported(self):
        p = self._flat("v3_exq_null_v3", {
            "run_id": "v3_exq_null_v3", "outcome": "PASS",
            "aggregates": {"level": None, "n": 2}})
        verdict, _, notes = vr.check_flat_scalar_readout(p)
        self.assertEqual(verdict, "encoding_only")
        self.assertTrue(any("null" in n for n in notes))

    # --- resolution order must MIRROR the converter's ----------------------
    def test_first_spelling_wins_even_when_a_later_one_is_healthy(self):
        """The converter takes the FIRST non-empty of metrics -> aggregates ->
        summary_metrics -> readout and never looks further. A manifest with a
        nested, non-scalar `metrics` block AND a good flat `readout` therefore
        scores from `metrics` and lands values with no numeric entries. The
        check must report that, not search on for the healthy block."""
        p = self._flat("v3_exq_order_v3", {
            "run_id": "v3_exq_order_v3", "outcome": "FAIL",
            "metrics": {"per_arm": {"A": 0.2}},        # nested -> no numeric
            "readout": {"max_dv": 0.2, "bar": 0.25}})  # healthy, but unreachable
        verdict, where, _ = vr.check_flat_scalar_readout(p)
        self.assertEqual(verdict, "no_numeric")
        self.assertEqual(where, "metrics")

    def test_spelling_order_matches_the_converter(self):
        self.assertEqual(vr._READOUT_SPELLINGS,
                         ("metrics", "aggregates", "summary_metrics", "readout"))

    # --- exemptions (found by the GOV-HELDOUT-1 check on the rule) ---------
    def test_error_outcome_is_exempt(self):
        p = self._flat("v3_exq_boom_v3",
                       {"run_id": "v3_exq_boom_v3", "outcome": "ERROR",
                        "error": "boom"})
        self.assertIsNone(vr.check_flat_scalar_readout(p))

    def test_runner_error_run_id_is_exempt(self):
        p = self._flat("v3_v3_exq_591g_runner_error_v3",
                       {"run_id": "v3_v3_exq_591g_runner_error_20260902T201744Z_v3"})
        self.assertIsNone(vr.check_flat_scalar_readout(p))

    def test_dry_run_is_exempt(self):
        p = self._flat("v3_exq_dry_v3",
                       {"run_id": "v3_exq_dry_v3", "outcome": "PASS", "dry_run": True})
        self.assertIsNone(vr.check_flat_scalar_readout(p))

    def test_non_manifest_json_is_not_gated(self):
        p = self._flat("some_index", {"generated_at": "2026-09-09T00:00:00Z"})
        self.assertIsNone(vr.check_flat_scalar_readout(p))

    # --- pack manifests read the sibling metrics.json ----------------------
    def test_pack_with_numeric_values_is_clean(self):
        p = self._pack("v3_exq_p1_20260909T000000Z_v3", "v3_exq_p1", {"dv": 0.3})
        self.assertIsNone(vr.check_flat_scalar_readout(p))

    def test_pack_with_empty_values_is_absent(self):
        p = self._pack("v3_exq_p2_20260909T000000Z_v3", "v3_exq_p2", {})
        self.assertEqual(vr.check_flat_scalar_readout(p)[0], "absent")

    def test_pack_with_bool_only_values_is_no_numeric(self):
        p = self._pack("v3_exq_p3_20260909T000000Z_v3", "v3_exq_p3", {"ok": True})
        self.assertEqual(vr.check_flat_scalar_readout(p)[0], "no_numeric")


class ReadoutNeverBlocksUnderStrict(unittest.TestCase):
    """The readout arm is advisory in EVERY mode -- deliberately, because
    /queue-experiment Step 3.5 runs this linter with --strict and 802 of 1008
    flat manifests lack the field. Pinned so a later tidy-up cannot quietly
    fold it into the --strict exit path."""

    def test_strict_exit_ignores_readout_findings(self):
        import inspect
        src = inspect.getsource(vr.main)
        self.assertIn("if args.strict and (gaps or thin_packs):", src)
        self.assertNotIn("readout_gaps)", src.split("if args.strict")[1])


class SpellingsAgreeWithConverter(unittest.TestCase):
    """The four spellings must match the runpack converter's harvest order, and
    validate_experiments' copy of them. A drift here silently un-gates a whole
    recording shape."""

    def test_spelling_set(self):
        self.assertEqual(set(vr._READOUT_SPELLINGS),
                         {"readout", "metrics", "aggregates", "summary_metrics"})

    def test_matches_validate_experiments(self):
        import pathlib as _p
        import sys as _s
        root = _p.Path(vr.__file__).resolve().parent
        if str(root) not in _s.path:
            _s.path.insert(0, str(root))
        import validate_experiments as ve
        self.assertEqual(set(ve._READOUT_SPELLINGS), set(vr._READOUT_SPELLINGS))


class CheckCriteriaThresholds(unittest.TestCase):
    """Experimental Recording Standard 3b "Re-derivable criteria" (2026-09-09).

    A load-bearing criterion recorded `passed` with no measured value and no bar
    cannot be re-checked from the manifest, so /governance Step 2b's
    threshold-arithmetic clause forces a driver read every cycle. V3-EXQ-936a's
    absolute bar sat ~7,900x above the maximum attainable effect and was logged
    clean for three consecutive cycles on exactly this shape.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="validate_crit_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, doc, name="v3_exq_1_x_20260909T000000Z_v3.json"):
        p = self.tmp / name
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def _verdict(self, doc, name="v3_exq_1_x_20260909T000000Z_v3.json"):
        return vr.check_criteria_thresholds(self._write(doc, name))

    # --- the gap shapes -------------------------------------------------

    def test_bare_passed_is_none_rederivable(self):
        """The V3-EXQ-642b / 936a shape: name + load_bearing + passed, nothing else."""
        v = self._verdict({"run_id": "r_v3", "outcome": "FAIL", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": False}]})
        self.assertEqual(v[0], "none_rederivable")
        self.assertEqual(v[1], ["C1"])

    def test_threshold_without_measured_is_flagged(self):
        """A bar with nothing measured against it: you cannot tell how far off it landed."""
        v = self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": False, "threshold": 0.02}]})
        self.assertEqual(v[0], "none_rederivable")

    def test_measured_without_threshold_is_flagged(self):
        v = self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True, "measured": 0.25}]})
        self.assertEqual(v[0], "none_rederivable")

    def test_numbers_only_in_prose_do_not_count(self):
        """V3-EXQ-967 records "= 0.99 vs tol 1e-09" in `detail`; V3-EXQ-1014 puts
        its three bars in `description`. Unreadable to every consumer."""
        v = self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True,
             "detail": "max |a-b| over 2400 steps = 0.99 vs tol 1e-09"}]})
        self.assertEqual(v[0], "none_rederivable")

    def test_partial_when_only_some_carry_both(self):
        v = self._verdict({"run_id": "r_v3", "combination_rule": "C1 AND C2",
                           "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True,
             "measured": 0.7, "threshold": 0.5},
            {"name": "C2", "load_bearing": True, "passed": True}]})
        self.assertEqual(v[0], "partial")
        self.assertEqual(v[1], ["C2"])

    def test_combination_rule_without_criteria_is_flagged(self):
        """V3-EXQ-900: `criteria` is null while `combination_rule` reads
        "PASS iff C1 AND C2 AND C4 hold". The rule names what was never written down."""
        v = self._verdict({"run_id": "r_v3", "outcome": "PASS", "criteria": None,
                           "combination_rule": "PASS iff C1 AND C2 AND C4 hold"})
        self.assertEqual(v[0], "criteria_unrecorded")

    # --- the compliant shapes (no false positives) ----------------------

    def test_measured_and_threshold_is_clean(self):
        self.assertIsNone(self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True,
             "measured": 0.78, "threshold": 0.5}]}))

    def test_threshold_family_spellings_discharge(self):
        """The corpus does not use one spelling. A literal spelling list flagged
        26 criteria across 14 genuinely-compliant manifests -- these are real
        pairs from those manifests."""
        for meas, thr in (("mean", "requirement"),          # V3-EXQ-1001, 1013
                          ("measured_rho", "threshold_rho"),  # V3-EXQ-785
                          ("measured_auc", "bar"),            # V3-EXQ-950
                          ("measured_max", "threshold"),      # V3-EXQ-895
                          ("n_seeds", "required"),            # V3-EXQ-1004
                          ("seeds_clearing", "seeds_required"),
                          ("measured_gap", "threshold_gap"),
                          ("mean_rho_rv_vs_commit", "rho_floor")):  # V3-EXQ-818
            with self.subTest(measured=meas, threshold=thr):
                self.assertIsNone(self._verdict({"run_id": "r_v3", "criteria": [
                    {"name": "C1", "load_bearing": True, "passed": True,
                     meas: 0.7, thr: 0.5}]}))

    def test_non_load_bearing_criteria_need_no_bar(self):
        self.assertIsNone(self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True,
             "measured": 0.7, "threshold": 0.5},
            {"name": "C2", "load_bearing": False, "passed": False}]}))

    def test_explicitly_nothing_load_bearing_is_believed(self):
        """A block that carries the marker but sets nothing true is taken at its
        word -- it says no criterion is load-bearing."""
        self.assertIsNone(self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": False, "passed": True}]}))

    def test_bool_is_not_a_number(self):
        """`_is_number` excludes bool, matching build_experiment_indexes -- a
        `threshold: true` records no bar."""
        v = self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True,
             "threshold": True, "measured": True}]})
        self.assertEqual(v[0], "none_rederivable")

    def test_nan_threshold_is_not_a_number(self):
        v = self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True,
             "threshold": float("nan"), "measured": 0.5}]})
        self.assertEqual(v[0], "none_rederivable")

    # --- container shapes ------------------------------------------------

    def test_criteria_as_dict_of_dicts(self):
        """V3-EXQ-1004's shape: criteria keyed by name, values are dicts."""
        self.assertIsNone(self._verdict({"run_id": "r_v3", "criteria": {
            "C1_visit_lift": {"met": True, "load_bearing": True,
                              "n_seeds": 5, "required": 3, "threshold": 15.0}}}))

    def test_criteria_as_bare_bool_map_is_flagged(self):
        """V3-EXQ-149b / 193: `{"C1_fast_wins": false, ...}` with no load_bearing
        marker anywhere. The worst-recorded shape in the corpus (101 flat
        manifests) -- and a rule scoped to `load_bearing: true` alone would
        exempt it BY CONSTRUCTION, which is what the GOV-HELDOUT-1 check found."""
        v = self._verdict({"run_id": "r_v3", "outcome": "PASS", "criteria": {
            "C1_fast_wins": False, "C2_slow_wins": True, "C3_default_ranked": True}})
        self.assertEqual(v[0], "none_rederivable")
        self.assertEqual(sorted(v[1]),
                         ["C1_fast_wins", "C2_slow_wins", "C3_default_ranked"])

    def test_per_seed_siblings_are_not_criteria(self):
        """`C1_per_seed` is a criterion's per-seed expansion, not a criterion."""
        v = self._verdict({"run_id": "r_v3", "criteria": {
            "C1_x": True, "C1_per_seed": [True, True, False]}})
        self.assertEqual(v[1], ["C1_x"])

    # --- combination_rule -------------------------------------------------

    def test_multi_load_bearing_without_combination_rule(self):
        v = self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True,
             "measured": 0.7, "threshold": 0.5},
            {"name": "C2", "load_bearing": True, "passed": True,
             "measured": 0.9, "threshold": 0.5}]})
        self.assertEqual(v[0], "combination_rule_only")
        self.assertTrue(v[3])

    def test_single_load_bearing_needs_no_combination_rule(self):
        self.assertIsNone(self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1", "load_bearing": True, "passed": True,
             "measured": 0.7, "threshold": 0.5}]}))

    # --- exemptions -------------------------------------------------------

    def test_error_outcome_is_exempt(self):
        self.assertIsNone(self._verdict({"run_id": "r_v3", "outcome": "ERROR",
                                         "criteria": [{"name": "C1",
                                                       "load_bearing": True,
                                                       "passed": False}]}))

    def test_runner_error_run_id_is_exempt(self):
        self.assertIsNone(self._verdict(
            {"run_id": "v3_x_runner_error_20260909T0_v3",
             "criteria": [{"name": "C1", "load_bearing": True, "passed": False}]}))

    def test_dry_run_field_is_exempt(self):
        self.assertIsNone(self._verdict({"run_id": "r_v3", "dry_run": True,
                                         "criteria": [{"name": "C1",
                                                       "load_bearing": True,
                                                       "passed": False}]}))

    def test_dry_prefixed_filename_is_exempt(self):
        """24 dry manifests in the corpus and not all set the field -- the
        `_dry_` filename prefix is the other half of that signal."""
        self.assertIsNone(self._verdict(
            {"run_id": "r_v3",
             "criteria": [{"name": "C1", "load_bearing": True, "passed": False}]},
            name="_dry_v3_exq_1_x_20260909T000000Z_v3.json"))

    def test_criterion_opt_out_marker(self):
        """A genuine count-based negative existential: a bar is not the right
        shape and inventing one would be worse recording."""
        self.assertIsNone(self._verdict({"run_id": "r_v3", "criteria": [
            {"name": "C1_no_leakage", "load_bearing": True, "passed": True,
             "occurrences": 0, "calls": 2400,
             "threshold_not_applicable": "count-based negative existential"}]}))

    def test_no_criteria_and_no_rule_is_not_gated(self):
        self.assertIsNone(self._verdict({"run_id": "r_v3", "outcome": "PASS"}))

    def test_non_manifest_json_is_not_gated(self):
        self.assertIsNone(self._verdict({"criteria": [
            {"name": "C1", "load_bearing": True, "passed": False}]}))


class CriteriaNeverBlocksUnderStrict(unittest.TestCase):
    """Advisory in EVERY mode, by the same precedent as the readout arm: 184 of
    216 flat manifests carrying load-bearing criteria record none that is
    re-derivable, and /queue-experiment Step 3.5 runs this linter with --strict.
    Pinned so a later tidy-up cannot fold it into the --strict exit path."""

    def test_strict_exit_ignores_criteria_findings(self):
        import inspect
        src = inspect.getsource(vr.main)
        self.assertIn("if args.strict and (gaps or thin_packs):", src)
        tail = src.split("if args.strict")[1]
        self.assertNotIn("criteria_gaps)", tail)


class CriteriaThresholdTokensAgree(unittest.TestCase):
    """The threshold-family token set must match validate_experiments' copy.
    A drift silently un-gates a whole recording shape on one side only."""

    def test_matches_validate_experiments(self):
        import pathlib as _p
        import sys as _s
        root = _p.Path(vr.__file__).resolve().parent
        if str(root) not in _s.path:
            _s.path.insert(0, str(root))
        import validate_experiments as ve
        self.assertEqual(set(ve._CRITERION_THRESHOLD_TOKENS),
                         set(vr._THRESHOLD_TOKENS))

    def test_core_spellings_present(self):
        for token in ("threshold", "requirement", "required", "bar", "floor", "tol"):
            self.assertIn(token, vr._THRESHOLD_TOKENS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
