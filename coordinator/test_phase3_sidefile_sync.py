"""Tests for the Phase 3 companion side-file sync.

A run may emit COMPANION artifacts alongside its result manifest -- most
commonly a `<type>_<ts>_episode_log.json` that fishtank_viz.html reads via
serve.py /api/fishtank/logs. This path carries those companions worker ->
coordinator (spool) -> origin (writer), so a fishtank showcase that runs on a
cloud worker lands its episode_log on origin/master instead of stranding it on
the worker's disk (the V3-EXQ-664 incident, 2026-06-10).

Covers:
  - manifest_spool companion API: round-trip, disabled no-op, unsafe-relpath
    rejection, idempotent overwrite.
  - sync_daemon.phase3_git_writer side-file materialisation:
      * flag ON  -> manifest + companion land in the SAME commit on origin;
      * flag OFF -> manifest lands, companion does NOT (bit-identical);
      * late companion (manifest already committed) lands on a follow-up tick.
  - coordinator_client.report_result_sidefile envelope round-trips through the
    /result/sidefile spool contract.
  - runner-side enumeration helpers (declared + auto-discovered companions,
    evidence-prefix boundary, default-OFF gate). The declared-companion
    fixture mirrors the real fishtank-driver on-disk layout -- manifest in
    evidence/experiments/, companions one level down in
    evidence/experiments/{EXPERIMENT_TYPE}/ -- not the same directory; a
    same-directory fixture previously masked the exact
    chip-20260809-sidefile-collection-glob-bug regression these tests now
    cover.

All printed text is ASCII-only.
"""

import base64
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REE_V3 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REE_V3))

import manifest_spool  # noqa: E402
import sync_daemon  # noqa: E402
import db  # noqa: E402

# Captured once at import time so every restore below puts the module
# global back to its REAL value (True post-Phase-3-cutover, 2026-05-29),
# not a stale hardcoded False. sync_daemon is cached in sys.modules for
# the whole pytest process, so a literal `= False` restore here used to
# leak into every OTHER test module collected afterward in the same
# `pytest coordinator/` run.
_ORIG_WRITER_READY = sync_daemon.PHASE3_GIT_WRITER_READY


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check)


# ---------------------------------------------------------------------------
# manifest_spool companion API
# ---------------------------------------------------------------------------

class SidefileSpoolDisabledTest(unittest.TestCase):
    """COORDINATOR_SPOOL_DIR unset -> every companion op is a no-op."""

    def setUp(self):
        self._saved = os.environ.pop("COORDINATOR_SPOOL_DIR", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["COORDINATOR_SPOOL_DIR"] = self._saved

    def test_write_returns_none(self):
        self.assertIsNone(manifest_spool.write_sidefile(
            "run_x", "evidence/experiments/t/x_episode_log.json", b"{}"))

    def test_list_run_ids_empty(self):
        self.assertEqual(
            list(manifest_spool.list_pending_sidefile_run_ids()), [])

    def test_list_for_run_empty(self):
        self.assertEqual(manifest_spool.list_sidefiles_for_run("run_x"), [])


class SidefileSpoolRoundtripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="sf_spool_")
        self._saved = os.environ.get("COORDINATOR_SPOOL_DIR")
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp

    def tearDown(self):
        if self._saved is not None:
            os.environ["COORDINATOR_SPOOL_DIR"] = self._saved
        else:
            os.environ.pop("COORDINATOR_SPOOL_DIR", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_write_list_read_delete(self):
        run_id = "v3_exq_664_x_20260610T000000Z_v3"
        rel = "evidence/experiments/v3_exq_664_x/x_20260610_episode_log.json"
        raw = b'{"frames": [1, 2, 3]}'
        spooled = manifest_spool.write_sidefile(
            run_id, rel, raw, received_at="2026-06-10T00:00:00Z",
            sha256_hex="abc")
        self.assertIsNotNone(spooled)
        self.assertIn(run_id, list(
            manifest_spool.list_pending_sidefile_run_ids()))
        entries = manifest_spool.list_sidefiles_for_run(run_id)
        self.assertEqual(len(entries), 1)
        got_rel, bin_path = entries[0]
        self.assertEqual(got_rel, rel)
        with open(bin_path, "rb") as fh:
            self.assertEqual(fh.read(), raw)
        self.assertTrue(manifest_spool.delete_sidefiles(run_id))
        self.assertEqual(manifest_spool.list_sidefiles_for_run(run_id), [])
        self.assertEqual(
            list(manifest_spool.list_pending_sidefile_run_ids()), [])

    def test_multiple_companions_one_run(self):
        run_id = "run_multi"
        for i in range(3):
            self.assertIsNotNone(manifest_spool.write_sidefile(
                run_id, "evidence/experiments/t/c%d_episode_log.json" % i,
                b"%d" % i))
        self.assertEqual(
            len(manifest_spool.list_sidefiles_for_run(run_id)), 3)

    def test_idempotent_overwrite(self):
        run_id = "run_idem"
        rel = "evidence/experiments/t/x_episode_log.json"
        manifest_spool.write_sidefile(run_id, rel, b"v1")
        manifest_spool.write_sidefile(run_id, rel, b"v2")
        entries = manifest_spool.list_sidefiles_for_run(run_id)
        self.assertEqual(len(entries), 1, "same relpath must not duplicate")
        with open(entries[0][1], "rb") as fh:
            self.assertEqual(fh.read(), b"v2")

    def test_unsafe_relpath_refused(self):
        run_id = "run_evil"
        # traversal
        self.assertIsNone(manifest_spool.write_sidefile(
            run_id, "evidence/experiments/../../.git/config", b"x"))
        # outside the evidence/experiments prefix
        self.assertIsNone(manifest_spool.write_sidefile(
            run_id, "scripts/evil.py", b"x"))
        self.assertEqual(
            list(manifest_spool.list_pending_sidefile_run_ids()), [])

    def test_unsafe_run_id_refused(self):
        self.assertIsNone(manifest_spool.write_sidefile(
            "../escape", "evidence/experiments/t/x_episode_log.json", b"x"))

    def test_safe_companion_relpath_helper(self):
        ok = manifest_spool.safe_companion_relpath(
            "/evidence/experiments/t/x_episode_log.json")
        self.assertEqual(ok, "evidence/experiments/t/x_episode_log.json")
        self.assertIsNone(manifest_spool.safe_companion_relpath("evidence/x"))
        self.assertIsNone(manifest_spool.safe_companion_relpath(
            "evidence/experiments/../x"))
        self.assertIsNone(manifest_spool.safe_companion_relpath(""))


# ---------------------------------------------------------------------------
# writer end-to-end
# ---------------------------------------------------------------------------

def _bare_remote(parent):
    remote = pathlib.Path(parent) / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seeded_clone(parent, remote, name="asm"):
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                   check=True)
    _git(repo, "config", "user.email", "test@example")
    _git(repo, "config", "user.name", "test")
    (repo / "README.md").write_text("seed\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "phase3: seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class WriterSidefileE2ETest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="sf_writer_")
        self._saved_spool = os.environ.get("COORDINATOR_SPOOL_DIR")
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp
        self._remote = _bare_remote(self._tmp)
        self._repo = _seeded_clone(self._tmp, self._remote)
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._queue = os.path.join(self._tmp, "queue.json")
        with open(self._queue, "w", encoding="utf-8") as fh:
            json.dump({"items": []}, fh)
        self._saved_sf = sync_daemon.PHASE3_SPOOL_SIDEFILES
        self._saved_max_bytes = sync_daemon.PHASE3_SIDEFILE_MAX_BYTES
        # Forced False as a known starting precondition for each test
        # (independent of the real module default); _run_writer() below
        # flips it True for the duration of a tick and restores the REAL
        # value afterward, not this forced False.
        sync_daemon.PHASE3_GIT_WRITER_READY = False

    def tearDown(self):
        sync_daemon.PHASE3_GIT_WRITER_READY = _ORIG_WRITER_READY
        sync_daemon.PHASE3_SPOOL_SIDEFILES = self._saved_sf
        sync_daemon.PHASE3_SIDEFILE_MAX_BYTES = self._saved_max_bytes
        self._conn.close()
        if self._saved_spool is not None:
            os.environ["COORDINATOR_SPOOL_DIR"] = self._saved_spool
        else:
            os.environ.pop("COORDINATOR_SPOOL_DIR", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run_writer(self):
        sync_daemon.PHASE3_GIT_WRITER_READY = True
        try:
            return sync_daemon.phase3_git_writer(
                self._conn, self._queue, ree_assembly_path=str(self._repo))
        finally:
            sync_daemon.PHASE3_GIT_WRITER_READY = _ORIG_WRITER_READY

    def _spool_manifest(self, run_id, relpath):
        body = {"run_id": run_id, "queue_id": "V3-EXQ-664",
                "outcome": "PASS", "machine": "test",
                "manifest_relpath": relpath}
        raw = json.dumps(body, sort_keys=True).encode("utf-8")
        db.record_result(self._conn, run_id, "V3-EXQ-664", "test", "PASS",
                         "sha", len(raw))
        manifest_spool.write_manifest(run_id, raw, manifest_relpath=relpath)
        return raw

    def _origin_files(self):
        return set(_git(self._remote, "ls-tree", "-r", "--name-only",
                        "master").stdout.split())

    def test_flag_on_manifest_and_companion_same_commit(self):
        sync_daemon.PHASE3_SPOOL_SIDEFILES = True
        run_id = "v3_exq_664_fish_20260610T010101Z_v3"
        mrel = "evidence/experiments/v3_exq_664_fish/fish_20260610T010101Z.json"
        crel = ("evidence/experiments/v3_exq_664_fish/"
                "fish_20260610T010101Z_episode_log.json")
        self._spool_manifest(run_id, mrel)
        comp_bytes = b'{"frames": [{"t": 0}, {"t": 1}]}'
        manifest_spool.write_sidefile(run_id, crel, comp_bytes)

        self.assertTrue(self._run_writer())

        origin = self._origin_files()
        self.assertIn(mrel, origin, "manifest must reach origin")
        self.assertIn(crel, origin, "episode_log must reach origin")
        # byte-correct on origin
        blob = _git(self._repo, "show", "master:" + crel).stdout
        self.assertEqual(json.loads(blob), json.loads(comp_bytes))
        # both spool entries drained
        self.assertEqual(
            list(manifest_spool.list_pending_sidefile_run_ids()), [])
        self.assertEqual(list(manifest_spool.list_pending_run_ids()), [])

    def test_flag_off_companion_not_committed(self):
        sync_daemon.PHASE3_SPOOL_SIDEFILES = False
        run_id = "v3_exq_664_off_20260610T020202Z_v3"
        mrel = "evidence/experiments/v3_exq_664_off/off_20260610T020202Z.json"
        crel = ("evidence/experiments/v3_exq_664_off/"
                "off_20260610T020202Z_episode_log.json")
        self._spool_manifest(run_id, mrel)
        manifest_spool.write_sidefile(run_id, crel, b'{"x": 1}')

        self.assertTrue(self._run_writer())

        origin = self._origin_files()
        self.assertIn(mrel, origin, "manifest still lands with flag off")
        self.assertNotIn(crel, origin,
                         "companion must NOT land when flag off")
        # companion spool retained (picked up once the flag is enabled)
        self.assertIn(run_id, list(
            manifest_spool.list_pending_sidefile_run_ids()))

    def test_late_companion_after_manifest_committed(self):
        """Manifest committed on tick 1; companion arrives after the manifest
        spool was drained. A follow-up tick must still commit the companion."""
        sync_daemon.PHASE3_SPOOL_SIDEFILES = True
        run_id = "v3_exq_664_late_20260610T030303Z_v3"
        mrel = "evidence/experiments/v3_exq_664_late/late_20260610T030303Z.json"
        crel = ("evidence/experiments/v3_exq_664_late/"
                "late_20260610T030303Z_episode_log.json")
        self._spool_manifest(run_id, mrel)
        self.assertTrue(self._run_writer())
        self.assertIn(mrel, self._origin_files())
        # companion arrives late, after the manifest is gone from the spool
        manifest_spool.write_sidefile(run_id, crel, b'{"late": true}')
        self.assertTrue(self._run_writer())
        self.assertIn(crel, self._origin_files(),
                      "late companion must reach origin on a follow-up tick")
        self.assertEqual(
            list(manifest_spool.list_pending_sidefile_run_ids()), [])

    # -----------------------------------------------------------------
    # Oversized-companion quarantine (2026-08-12, PHASE3_SIDEFILE_MAX_BYTES)
    # -----------------------------------------------------------------

    def test_sidefile_under_threshold_materialises_normally(self):
        """A companion at/under PHASE3_SIDEFILE_MAX_BYTES is written and
        committed byte-for-byte, exactly as before this size check existed --
        no placemarker, no different code path."""
        sync_daemon.PHASE3_SPOOL_SIDEFILES = True
        sync_daemon.PHASE3_SIDEFILE_MAX_BYTES = 1000
        run_id = "v3_exq_920_small_20260812T000000Z_v3"
        mrel = "evidence/experiments/v3_exq_920_small/small_20260812T000000Z.json"
        crel = ("evidence/experiments/v3_exq_920_small/"
                "small_20260812T000000Z_episode_log.json")
        self._spool_manifest(run_id, mrel)
        comp_bytes = b"x" * 999  # under the 1000-byte test threshold
        manifest_spool.write_sidefile(run_id, crel, comp_bytes)

        self.assertTrue(self._run_writer())

        origin = self._origin_files()
        self.assertIn(crel, origin)
        self.assertNotIn(
            crel + manifest_spool.PLACEMARKER_SUFFIX, origin,
            "under-threshold companion must not be quarantined")
        blob = _git(self._repo, "show", "master:" + crel).stdout
        self.assertEqual(blob.encode("utf-8"), comp_bytes)
        # spool fully drained, same as the pre-existing behaviour
        self.assertEqual(
            list(manifest_spool.list_pending_sidefile_run_ids()), [])

    def test_sidefile_over_threshold_quarantined_with_placemarker(self):
        """A companion over PHASE3_SIDEFILE_MAX_BYTES is NOT written into the
        working tree in full and NOT git-added -- a small placemarker JSON is
        committed in its place, and the writer completes without error (the
        2026-08-12 incident: every push of the raw 184.6MB bytes was
        rejected by GitHub's 100MB blob limit and the daemon crash-looped)."""
        sync_daemon.PHASE3_SPOOL_SIDEFILES = True
        sync_daemon.PHASE3_SIDEFILE_MAX_BYTES = 1000
        run_id = "v3_exq_920_big_20260812T010101Z_v3"
        mrel = "evidence/experiments/v3_exq_920_big/big_20260812T010101Z.json"
        crel = ("evidence/experiments/v3_exq_920_big/"
                "big_20260812T010101Z_episode_log.json")
        self._spool_manifest(run_id, mrel)
        comp_bytes = b"y" * 2000  # over the 1000-byte test threshold
        manifest_spool.write_sidefile(run_id, crel, comp_bytes)

        self.assertTrue(
            self._run_writer(),
            "writer must not crash-exit on an oversized companion")

        origin = self._origin_files()
        self.assertIn(mrel, origin, "manifest still lands")
        self.assertNotIn(
            crel, origin, "oversized companion's raw bytes must not land")
        placemarker_rel = crel + manifest_spool.PLACEMARKER_SUFFIX
        self.assertIn(placemarker_rel, origin,
                      "placemarker must be committed in its place")
        payload = json.loads(
            _git(self._repo, "show", "master:" + placemarker_rel).stdout)
        self.assertEqual(payload["artifact_type"],
                         "oversized_evidence_artifact_placemarker")
        self.assertEqual(payload["run_id"], run_id)
        self.assertEqual(payload["size_bytes"], len(comp_bytes))
        self.assertEqual(payload["sha256"],
                         hashlib.sha256(comp_bytes).hexdigest())

    def test_oversized_companion_bytes_remain_in_spool(self):
        """Quarantining a companion commits a placemarker but must NOT
        delete the real bytes from the spool -- they stay recoverable, the
        same way the manual 2026-08-12 incident fix recovered the file from
        the spool by hand."""
        sync_daemon.PHASE3_SPOOL_SIDEFILES = True
        sync_daemon.PHASE3_SIDEFILE_MAX_BYTES = 1000
        run_id = "v3_exq_920_retain_20260812T020202Z_v3"
        mrel = ("evidence/experiments/v3_exq_920_retain/"
                "retain_20260812T020202Z.json")
        crel = ("evidence/experiments/v3_exq_920_retain/"
                "retain_20260812T020202Z_episode_log.json")
        self._spool_manifest(run_id, mrel)
        comp_bytes = b"z" * 5000
        manifest_spool.write_sidefile(run_id, crel, comp_bytes)

        self.assertTrue(self._run_writer())

        entries = manifest_spool.list_sidefiles_for_run(run_id)
        self.assertEqual(
            len(entries), 1,
            "oversized companion's spool entry must be RETAINED")
        got_rel, bin_path = entries[0]
        self.assertEqual(got_rel, crel)
        with open(bin_path, "rb") as fh:
            self.assertEqual(fh.read(), comp_bytes)
        self.assertIn(run_id, list(
            manifest_spool.list_pending_sidefile_run_ids()))

    def test_mixed_run_oversized_companion_blocks_spool_wipe(self):
        """A run with BOTH a normal-size and an oversized companion: the
        normal one still lands on origin, but the run's whole spool dir is
        NOT wiped afterward (delete_sidefiles operates per-run_id, not
        per-file) -- so the oversized companion's bytes are never
        collaterally dropped alongside its already-committed sibling."""
        sync_daemon.PHASE3_SPOOL_SIDEFILES = True
        sync_daemon.PHASE3_SIDEFILE_MAX_BYTES = 1000
        run_id = "v3_exq_920_mixed_20260812T030303Z_v3"
        mrel = ("evidence/experiments/v3_exq_920_mixed/"
                "mixed_20260812T030303Z.json")
        crel_small = ("evidence/experiments/v3_exq_920_mixed/"
                     "mixed_20260812T030303Z_episode_log.json")
        crel_big = ("evidence/experiments/v3_exq_920_mixed/"
                    "mixed_20260812T030303Z_extra_episode_log.json")
        self._spool_manifest(run_id, mrel)
        manifest_spool.write_sidefile(run_id, crel_small, b"s" * 10)
        manifest_spool.write_sidefile(run_id, crel_big, b"b" * 5000)

        self.assertTrue(self._run_writer())

        origin = self._origin_files()
        self.assertIn(crel_small, origin)
        self.assertIn(crel_big + manifest_spool.PLACEMARKER_SUFFIX, origin)
        self.assertNotIn(crel_big, origin)
        # spool dir NOT wiped -- both entries still present
        self.assertEqual(
            len(manifest_spool.list_sidefiles_for_run(run_id)), 2)

    def test_oversized_quarantine_idempotent_across_ticks(self):
        """A second tick over an unchanged oversized companion must not
        crash and must not re-hash/re-commit -- the placemarker is already
        on origin, so the tick has nothing new to stage."""
        sync_daemon.PHASE3_SPOOL_SIDEFILES = True
        sync_daemon.PHASE3_SIDEFILE_MAX_BYTES = 1000
        run_id = "v3_exq_920_repeat_20260812T040404Z_v3"
        mrel = ("evidence/experiments/v3_exq_920_repeat/"
                "repeat_20260812T040404Z.json")
        crel = ("evidence/experiments/v3_exq_920_repeat/"
                "repeat_20260812T040404Z_episode_log.json")
        self._spool_manifest(run_id, mrel)
        manifest_spool.write_sidefile(run_id, crel, b"q" * 3000)
        self.assertTrue(self._run_writer())

        placemarker_rel = crel + manifest_spool.PLACEMARKER_SUFFIX
        before = _git(self._repo, "show", "master:" + placemarker_rel).stdout
        head_before = _git(self._repo, "rev-parse", "master").stdout.strip()

        # Second tick: manifest spool already drained, oversized companion
        # still sits in the spool by design -> nothing new to stage.
        self.assertFalse(self._run_writer())

        head_after = _git(self._repo, "rev-parse", "master").stdout.strip()
        self.assertEqual(
            head_before, head_after,
            "idempotent tick must not create an empty/duplicate commit")
        after = _git(self._repo, "show", "master:" + placemarker_rel).stdout
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# coordinator_client transport contract
# ---------------------------------------------------------------------------

class ClientEnvelopeTest(unittest.TestCase):
    """report_result_sidefile builds a gzipped {run_id, relpath, content_b64}
    envelope that the /result/sidefile endpoint decodes and spools. Verify the
    envelope round-trips into write_sidefile without a live server."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="sf_client_")
        self._saved_spool = os.environ.get("COORDINATOR_SPOOL_DIR")
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp
        # Force the client enabled + capture the outbound body.
        import coordinator_client
        self._cc = coordinator_client
        self._saved_enabled = coordinator_client._ENABLED
        coordinator_client._ENABLED = True
        self._captured = {}

        def _fake_post(path, payload, gzip_body=False):
            self._captured["path"] = path
            self._captured["payload"] = payload
            self._captured["gzip_body"] = gzip_body
            return {"ok": True}
        self._saved_post = coordinator_client._post
        coordinator_client._post = _fake_post

    def tearDown(self):
        self._cc._ENABLED = self._saved_enabled
        self._cc._post = self._saved_post
        if self._saved_spool is not None:
            os.environ["COORDINATOR_SPOOL_DIR"] = self._saved_spool
        else:
            os.environ.pop("COORDINATOR_SPOOL_DIR", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_envelope_decodes_and_spools(self):
        src = pathlib.Path(self._tmp) / "episode_log.json"
        content = b'{"frames": [1, 2, 3], "n": 753}'
        src.write_bytes(content)
        run_id = "v3_exq_664_env_20260610T040404Z_v3"
        rel = ("evidence/experiments/v3_exq_664_env/"
               "env_20260610T040404Z_episode_log.json")
        self._cc.report_result_sidefile(run_id, rel, str(src))

        self.assertEqual(self._captured["path"], "/result/sidefile")
        self.assertTrue(self._captured["gzip_body"])
        # decode exactly as the server's _read_body + endpoint would
        envelope = json.loads(gzip.decompress(self._captured["payload"]))
        self.assertEqual(envelope["run_id"], run_id)
        self.assertEqual(envelope["relpath"], rel)
        raw = base64.b64decode(envelope["content_b64"])
        self.assertEqual(raw, content)
        # feed into the spool the way the endpoint does -> byte-correct
        manifest_spool.write_sidefile(run_id, rel, raw)
        entries = manifest_spool.list_sidefiles_for_run(run_id)
        self.assertEqual(len(entries), 1)
        with open(entries[0][1], "rb") as fh:
            self.assertEqual(fh.read(), content)


# ---------------------------------------------------------------------------
# runner-side enumeration helpers
# ---------------------------------------------------------------------------

class RunnerHelperTest(unittest.TestCase):
    """Fixture mirrors the REAL fishtank-driver on-disk layout: the flat
    manifest lands in evidence/experiments/ (write_flat_manifest's out_dir
    argument is out_dir.parent in every affected driver), while the
    episode_log and any other companions land one level down in
    evidence/experiments/{EXPERIMENT_TYPE}/ (out_dir itself) -- NOT the same
    directory as the manifest. A same-directory fixture here previously
    masked the exact bug this class regression-tests
    (chip-20260809-sidefile-collection-glob-bug, 2026-08-09): a declared
    companion_files entry without the "{EXPERIMENT_TYPE}/" prefix resolves
    against the manifest's own directory and silently finds nothing."""

    def setUp(self):
        import experiment_runner
        self._r = experiment_runner
        self._tmp = tempfile.mkdtemp(prefix="sf_runner_")
        self._asm = pathlib.Path(self._tmp) / "REE_assembly"
        self._manifest_dir = self._asm / "evidence" / "experiments"
        self._companion_dir = self._manifest_dir / "v3_exq_664_x"
        self._companion_dir.mkdir(parents=True)
        self._mp = self._manifest_dir / "v3_exq_664_x_20260610T000000Z.json"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_collect_declared_resolves_into_subdirectory(self):
        """Regression test for the real fishtank-driver layout: a declared
        companion_files entry of the form "{EXPERIMENT_TYPE}/{filename}"
        must resolve, relative to the manifest's directory, into the
        EXPERIMENT_TYPE subdirectory where the episode_log actually lives."""
        self._mp.write_text(json.dumps({
            "run_id": "rid",
            "companion_files": [
                "v3_exq_664_x/v3_exq_664_x_20260610T000000Z_episode_log.json",
                "v3_exq_664_x/extra_companion.json",
            ],
        }))
        (self._companion_dir
         / "v3_exq_664_x_20260610T000000Z_episode_log.json").write_text("{}")
        (self._companion_dir / "extra_companion.json").write_text("{}")
        (self._companion_dir / "unrelated.txt").write_text("no")
        doc = json.loads(self._mp.read_text())
        names = sorted(p.name for p in self._r._collect_companion_files(
            self._mp, doc))
        self.assertEqual(names, [
            "extra_companion.json",
            "v3_exq_664_x_20260610T000000Z_episode_log.json"])

    def test_collect_declared_without_prefix_finds_nothing(self):
        """The exact bug: a declared entry naming only the bare filename
        (no "{EXPERIMENT_TYPE}/" prefix) resolves against the manifest's own
        directory, where the file does not exist -- so it is silently
        dropped, never raised. This is what shipped in the 906b driver
        before the 2026-08-09 fix, and in 664/665/906/906a without any
        companion_files declaration at all."""
        self._mp.write_text(json.dumps({
            "run_id": "rid",
            "companion_files": [
                "v3_exq_664_x_20260610T000000Z_episode_log.json"],
        }))
        (self._companion_dir
         / "v3_exq_664_x_20260610T000000Z_episode_log.json").write_text("{}")
        doc = json.loads(self._mp.read_text())
        comp = self._r._collect_companion_files(self._mp, doc)
        self.assertEqual(comp, [], "bare filename must NOT resolve across "
                          "the manifest/companion directory boundary")

    def test_collect_autodiscovery_does_not_recurse_into_subdirectory(self):
        """The *_episode_log.json glob auto-discovery scans the MANIFEST's
        own directory only -- it does not recurse into an EXPERIMENT_TYPE
        subdirectory, so it cannot rescue an undeclared companion that lives
        one level down (the real fishtank layout). This is why every
        fishtank driver must declare companion_files explicitly."""
        self._mp.write_text(json.dumps({"run_id": "rid"}))
        (self._companion_dir
         / "v3_exq_664_x_20260610T000000Z_episode_log.json").write_text("{}")
        doc = json.loads(self._mp.read_text())
        comp = self._r._collect_companion_files(self._mp, doc)
        self.assertEqual(comp, [])

    def test_collect_autodiscovery_same_directory_layout(self):
        """A driver that writes both the manifest and the episode_log into
        the SAME directory (e.g. v3_exq_495's out_dir passed directly to
        write_flat_manifest, not out_dir.parent) is still covered by
        auto-discovery with no companion_files declaration needed."""
        flat_mp = self._companion_dir / "v3_exq_664_x_20260610T010101Z.json"
        flat_mp.write_text(json.dumps({"run_id": "rid"}))
        (self._companion_dir
         / "v3_exq_664_x_20260610T010101Z_episode_log.json").write_text("{}")
        comp = self._r._collect_companion_files(
            flat_mp, json.loads(flat_mp.read_text()))
        self.assertEqual(
            [p.name for p in comp],
            ["v3_exq_664_x_20260610T010101Z_episode_log.json"])

    def test_collect_autodiscovery_ignores_unrelated_sibling_episode_log(self):
        """Regression for the shape of bug caught live 2026-08-11 in the
        sibling _relocate_dry_run_manifest path (experiment_protocol.py,
        _iter_companion_files): a directory-wide `*_episode_log.json` glob
        would sweep up a PRIOR REAL run's episode_log sitting in the same
        per-experiment-type directory alongside the manifest being processed.
        This call site is non-destructive (read/upload only, not delete), but
        the same over-collection would mis-attribute an unrelated run's
        sidecar to this manifest's coordinator spool. Auto-discovery must
        match only THIS manifest's own expected sidecar name."""
        flat_mp = self._companion_dir / "v3_exq_664_x_20260610T020202Z.json"
        flat_mp.write_text(json.dumps({"run_id": "rid"}))
        own_log = (self._companion_dir
                   / "v3_exq_664_x_20260610T020202Z_episode_log.json")
        own_log.write_text(json.dumps({"episodes": ["mine"]}))
        # Unrelated prior real run's sidecar in the SAME directory.
        (self._companion_dir
         / "v3_exq_664_x_20260425T204829Z_episode_log.json").write_text(
             json.dumps({"episodes": ["real evidence, not mine"]}))
        comp = self._r._collect_companion_files(
            flat_mp, json.loads(flat_mp.read_text()))
        self.assertEqual([p.name for p in comp], [own_log.name],
                          "only the manifest's own sidecar may be collected")

    def test_collect_excludes_manifest_itself(self):
        self._mp.write_text("{}")
        comp = self._r._collect_companion_files(self._mp, {})
        self.assertNotIn(self._mp.resolve(), [p.resolve() for p in comp])

    def test_evidence_relpath_boundary(self):
        cpath = self._companion_dir / "x_episode_log.json"
        cpath.write_text("{}")
        rel = self._r._evidence_relpath(self._asm, cpath)
        self.assertEqual(
            rel, "evidence/experiments/v3_exq_664_x/x_episode_log.json")
        # outside the assembly tree -> None
        self.assertIsNone(self._r._evidence_relpath(
            self._asm, pathlib.Path("/etc/passwd")))
        self.assertIsNone(self._r._evidence_relpath(None, cpath))

    def test_gate_default_off(self):
        saved = os.environ.pop("PHASE3_SPOOL_SIDEFILES", None)
        try:
            self.assertFalse(self._r._phase3_sidefiles_enabled())
            os.environ["PHASE3_SPOOL_SIDEFILES"] = "1"
            self.assertTrue(self._r._phase3_sidefiles_enabled())
        finally:
            if saved is not None:
                os.environ["PHASE3_SPOOL_SIDEFILES"] = saved
            else:
                os.environ.pop("PHASE3_SPOOL_SIDEFILES", None)


if __name__ == "__main__":
    unittest.main()
