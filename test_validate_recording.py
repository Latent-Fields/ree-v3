"""Tests for validate_recording.check_pack_provenance -- the thin-pack
recording-provenance regression check (2026-07-16).

A runs/<run_id>/manifest.json PACK that drops machine/machine_class/substrate_hash
while its flat sibling evidence/experiments/<run_id>.json carries them is the
thin-pack bug: the index-scored pack reads machine_class=null even though the
provenance was recorded. check_pack_provenance flags exactly that case.
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
