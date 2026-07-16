"""Contract tests for the Phase-3 git writer run-pack materialisation
(PHASE3_MATERIALIZE_RUNPACK), the belt-and-suspenders complement to the
2026-06-06 flat-only silent-drop fix (REE_assembly c92458c731).

Covers:
  - Golden byte-shape: sync_v3_results.runpack_for_flat reproduces the
    canonical runs/ pack for the sibling V3-EXQ-633 cloud-2 run (which DID
    sync both flat + pack) -- the pack the local governance.sh converter
    produced is byte-identical to what the hub writer would emit.
  - _materialize_runpacks back-fills a MISSING pack and git-adds it.
  - Skip-if-pack-exists: a runner-synced pack is NEVER clobbered.
  - Non-eligible flat manifests (wrong epoch / no run_id) materialise nothing.
  - Default-OFF guarantee: PHASE3_MATERIALIZE_RUNPACK is False unless the env
    knob is set.

Run: /opt/local/bin/python3 test_phase3_runpack_materialize.py
or:  /opt/local/bin/python3 -m unittest test_phase3_runpack_materialize

All printed text is ASCII-only (Windows cp1252 safety).
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sync_daemon  # noqa: E402

# REE_Working/ree-v3/coordinator -> REE_Working/REE_assembly
REE_ASSEMBLY = HERE.parent.parent / "REE_assembly"
SCRIPTS_DIR = REE_ASSEMBLY / "evidence" / "experiments" / "scripts"
EVIDENCE_DIR = REE_ASSEMBLY / "evidence" / "experiments"

# Make sync_v3_results importable for the golden-reference test.
if SCRIPTS_DIR.is_dir():
    sys.path.insert(0, str(SCRIPTS_DIR))

_EXQ633_FLAT = (
    EVIDENCE_DIR
    / "v3_exq_633_mech094_simulation_real_writegate_discriminative"
      "_20260603T072042Z_v3.json")
_EXQ633_RUN_DIR = (
    EVIDENCE_DIR
    / "v3_exq_633_mech094_simulation_real_writegate_discriminative"
    / "runs"
    / "v3_exq_633_mech094_simulation_real_writegate_discriminative"
      "_20260603T072042Z_v3")


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True)


def _make_flat(run_id, experiment_type, *,
               epoch="ree_hybrid_guardrails_v1",
               claim_ids=("MECH-094",),
               outcome="PASS",
               criteria=None):
    doc = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": experiment_type,
        "architecture_epoch": epoch,
        "timestamp_utc": "20260606T120000Z",
        "experiment_purpose": "evidence",
        "claim_ids": list(claim_ids),
        "outcome": outcome,
        "evidence_direction": "supports",
        "metrics": {"some_metric": 1.25, "n_seeds": 3},
        "criteria": criteria if criteria is not None
        else {"c1_pass": True, "c2_pass": True, "overall_pass": True},
    }
    return doc


class GoldenByteShape(unittest.TestCase):
    """runpack_for_flat must reproduce the on-disk V3-EXQ-633 pack exactly."""

    @unittest.skipUnless(
        _EXQ633_FLAT.is_file() and (_EXQ633_RUN_DIR / "manifest.json").is_file(),
        "V3-EXQ-633 sibling pack not present in this checkout")
    def test_runpack_for_flat_matches_exq633_pack(self):
        import sync_v3_results
        result = sync_v3_results.runpack_for_flat(_EXQ633_FLAT, EVIDENCE_DIR)
        self.assertIsNotNone(result)
        run_dir, manifest_doc, metrics_doc, summary = result

        # Directory derivation lands on the same runs/<run_id>/ path.
        self.assertEqual(
            pathlib.Path(run_dir).resolve(), _EXQ633_RUN_DIR.resolve())

        on_disk_manifest = json.loads(
            (_EXQ633_RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
        on_disk_metrics = json.loads(
            (_EXQ633_RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
        on_disk_summary = (_EXQ633_RUN_DIR / "summary.md").read_text(
            encoding="utf-8")

        self.assertEqual(manifest_doc, on_disk_manifest)
        self.assertEqual(metrics_doc, on_disk_metrics)
        self.assertEqual(summary, on_disk_summary)

        # And the serialised bytes match what the writer would commit.
        self.assertEqual(
            json.dumps(manifest_doc, indent=2) + "\n",
            (_EXQ633_RUN_DIR / "manifest.json").read_text(encoding="utf-8"))


class MaterializeRunpacks(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="phase3_runpack_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.asm = self.tmp / "REE_assembly"
        self.exp_dir = self.asm / "evidence" / "experiments"
        self.exp_dir.mkdir(parents=True)
        # The writer loads the builder from <asm>/evidence/experiments/scripts.
        scripts = self.asm / "evidence" / "experiments" / "scripts"
        scripts.mkdir()
        shutil.copy(SCRIPTS_DIR / "sync_v3_results.py",
                    scripts / "sync_v3_results.py")
        _git(self.asm, "init", "--quiet")
        _git(self.asm, "config", "user.email", "t@t")
        _git(self.asm, "config", "user.name", "t")

    def _write_flat(self, run_id, experiment_type, **kw):
        relpath = "evidence/experiments/%s.json" % run_id
        doc = _make_flat(run_id, experiment_type, **kw)
        (self.asm / relpath).write_text(
            json.dumps(doc), encoding="utf-8")
        _git(self.asm, "add", relpath)
        return run_id, relpath

    def _staged_paths(self):
        out = _git(self.asm, "diff", "--cached", "--name-only").stdout
        return set(out.split())

    def test_backfills_missing_pack_and_stages_it(self):
        run_id = "v3_exq_test_backfill_20260606T120000Z_v3"
        exp = "v3_exq_test_backfill"
        staged = [self._write_flat(run_id, exp)]
        n = sync_daemon._materialize_runpacks(str(self.asm), staged)
        self.assertEqual(n, 3)  # manifest.json + metrics.json + summary.md
        run_dir = self.exp_dir / exp / "runs" / run_id
        for fname in ("manifest.json", "metrics.json", "summary.md"):
            self.assertTrue((run_dir / fname).is_file(), fname)
        # All three are staged for the same commit as the flat manifest.
        staged_now = self._staged_paths()
        for fname in ("manifest.json", "metrics.json", "summary.md"):
            rel = "evidence/experiments/%s/runs/%s/%s" % (exp, run_id, fname)
            self.assertIn(rel, staged_now)
        # Field mapping: outcome PASS -> status PASS; claim_ids -> tested.
        manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["claim_ids_tested"], ["MECH-094"])
        self.assertEqual(manifest["run_id"], run_id)

    def test_skip_if_pack_already_exists(self):
        run_id = "v3_exq_test_existing_20260606T120000Z_v3"
        exp = "v3_exq_test_existing"
        staged = [self._write_flat(run_id, exp)]
        run_dir = self.exp_dir / exp / "runs" / run_id
        run_dir.mkdir(parents=True)
        sentinel = '{"runner_native": "do not clobber"}\n'
        (run_dir / "manifest.json").write_text(sentinel, encoding="utf-8")
        n = sync_daemon._materialize_runpacks(str(self.asm), staged)
        self.assertEqual(n, 0)
        # Untouched.
        self.assertEqual(
            (run_dir / "manifest.json").read_text("utf-8"), sentinel)
        self.assertFalse((run_dir / "metrics.json").exists())

    def test_non_eligible_flat_materializes_nothing(self):
        # Wrong architecture epoch -> not flat-v3 -> no pack.
        run_id = "v3_exq_test_badepoch_20260606T120000Z_v3"
        exp = "v3_exq_test_badepoch"
        staged = [self._write_flat(run_id, exp, epoch="ree_v2_legacy")]
        n = sync_daemon._materialize_runpacks(str(self.asm), staged)
        self.assertEqual(n, 0)
        self.assertFalse((self.exp_dir / exp / "runs").exists())

    def test_multiple_flats_in_one_pass(self):
        staged = [
            self._write_flat(
                "v3_exq_test_a_20260606T120000Z_v3", "v3_exq_test_a"),
            self._write_flat(
                "v3_exq_test_b_20260606T120000Z_v3", "v3_exq_test_b"),
        ]
        n = sync_daemon._materialize_runpacks(str(self.asm), staged)
        self.assertEqual(n, 6)


class FieldMappingProvenance(unittest.TestCase):
    """build_runpack_docs must carry always-core provenance from the flat manifest
    into the pack (2026-07-16 thin-pack fix) and fold `aggregates` into
    metrics.values when there is no top-level `metrics`."""

    def test_provenance_carried_into_pack(self):
        import sync_v3_results
        data = _make_flat("v3_exq_p_20260606T120000Z_v3", "v3_exq_p")
        data["machine"] = "ree-cloud-2"
        data["machine_class"] = "linux-x86_64-py3.10"
        data["substrate_hash"] = "f92a600cf17a"
        manifest, _metrics, _summary = sync_v3_results.build_runpack_docs(
            data, "v3_exq_p")
        self.assertEqual(manifest["machine"], "ree-cloud-2")
        self.assertEqual(manifest["machine_class"], "linux-x86_64-py3.10")
        self.assertEqual(manifest["substrate_hash"], "f92a600cf17a")

    def test_absent_provenance_omitted_not_nulled(self):
        """A flat without provenance produces a pack WITHOUT those keys (byte-
        identical to legacy output) -- never machine_class: null."""
        import sync_v3_results
        data = _make_flat("v3_exq_q_20260606T120000Z_v3", "v3_exq_q")
        manifest, _m, _s = sync_v3_results.build_runpack_docs(data, "v3_exq_q")
        self.assertNotIn("machine_class", manifest)
        self.assertNotIn("substrate_hash", manifest)
        self.assertNotIn("machine", manifest)

    def test_aggregates_folded_when_no_top_level_metrics(self):
        """766-style manifest: readouts under `aggregates`, no `metrics` key ->
        metrics.values carries the aggregates rather than staying empty."""
        import sync_v3_results
        data = _make_flat("v3_exq_r_20260606T120000Z_v3", "v3_exq_r")
        data.pop("metrics", None)
        data["aggregates"] = {"median_expansion_ratio": 2.40, "frac_ok": 0.917}
        _m, metrics_doc, _s = sync_v3_results.build_runpack_docs(data, "v3_exq_r")
        self.assertEqual(metrics_doc["values"]["median_expansion_ratio"], 2.40)
        self.assertEqual(metrics_doc["values"]["frac_ok"], 0.917)

    def test_top_level_metrics_win_over_aggregates(self):
        """When a top-level `metrics` dict is present it is used verbatim; the
        aggregates fallback only fills the empty case."""
        import sync_v3_results
        data = _make_flat("v3_exq_s_20260606T120000Z_v3", "v3_exq_s")
        data["metrics"] = {"some_metric": 1.25, "n_seeds": 3}
        data["aggregates"] = {"unused": 9.9}
        _m, metrics_doc, _s = sync_v3_results.build_runpack_docs(data, "v3_exq_s")
        self.assertEqual(metrics_doc["values"], {"some_metric": 1.25, "n_seeds": 3})


class FlagDefault(unittest.TestCase):

    def test_materialize_runpack_default_off(self):
        # Bit-identical guarantee: the feature is shadow/off unless the env
        # knob is explicitly set in the systemd unit.
        if "PHASE3_MATERIALIZE_RUNPACK" not in os.environ:
            self.assertFalse(sync_daemon.PHASE3_MATERIALIZE_RUNPACK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
