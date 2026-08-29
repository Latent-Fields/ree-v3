"""Contracts for POST /chip/episode + db.record_chip_episode (PHASE-4
slice, 2026-08-29 fleet-wedge campaign -- W5a recurrence-collapse
precondition C3).

Pins:
  1. IDEMPOTENT ACCUMULATION: episodes append into the standing chip's
     entry_json keyed by episode_key; a replayed observation is a no-op.
  2. OPEN-ONLY: a resolved chip refuses (the minting client owns the
     reopen-vs-escalate judgment, not the storage verb).
  3. BOUNDED GROWTH: past CHIP_EPISODE_CAP the oldest episodes drop,
     episodes_truncated preserves the magnitude, episode_count stays the
     TRUE total (kept + dropped).
  4. RENDER TRANSPARENCY: the materializer's chips render carries the
     episodes verbatim (entry_json doctrine) with zero renderer changes.

Time-independent. ASCII-only.
"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import app  # noqa: E402
import db  # noqa: E402
import task_claim_chip_git_writer as writer  # noqa: E402

REF = "chip-queuefloor-testbox-standing"


def _chip(status="open"):
    return {
        "chip_ref": REF, "task_id": None, "session_id": "tick",
        "session_label": "", "title": "Experiment queue STARVED (standing)",
        "tldr": "standing class chip", "prompt": "[chip_ref: %s] watch" % REF,
        "cwd": "/x", "origin": "hygiene_tick", "kind": "report",
        "urgency": False, "spawned_at": "2026-08-29T00:00:00Z",
        "status": status, "resolved_at": None,
        "resolved_by_session_id": None, "resolution_note": None,
        "resolution_note_auto": None, "claimed_by": None, "claimed_at": None,
        "claim_note": None, "attached_by_session_id": None,
        "attached_at": None,
    }


def _episode(key, **payload):
    ep = {"episode_key": key}
    ep.update(payload)
    return ep


class _Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="chip_episode_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        verdict, _ = db.record_chip(self._conn, _chip())
        self.assertEqual(verdict, "ok")

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _entry(self):
        row = self._conn.execute(
            "SELECT entry_json FROM chip_ledger WHERE chip_ref=?",
            (REF,)).fetchone()
        return json.loads(row["entry_json"])


class TestEpisodeDb(_Fixture):

    def test_append_and_idempotent(self):
        v, p = db.record_chip_episode(self._conn, REF,
                                      _episode("since-1", depth=1))
        self.assertEqual(v, "ok")
        self.assertEqual(p["episode_count"], 1)
        v, _ = db.record_chip_episode(self._conn, REF,
                                      _episode("since-1", depth=99))
        self.assertEqual(v, "idempotent")
        v, p = db.record_chip_episode(self._conn, REF,
                                      _episode("since-2", depth=2))
        self.assertEqual(v, "ok")
        self.assertEqual(p["episode_count"], 2)
        entry = self._entry()
        self.assertEqual([e["episode_key"] for e in entry["episodes"]],
                         ["since-1", "since-2"])
        self.assertEqual(entry["episodes"][0]["depth"], 1,
                         "a replay must never overwrite the stored episode")
        self.assertTrue(entry["last_episode_at"])

    def test_not_found_and_bad_episode(self):
        v, _ = db.record_chip_episode(self._conn, "chip-nope",
                                      _episode("k"))
        self.assertEqual(v, "not_found")
        for bad in (None, "x", {}, {"episode_key": ""},
                    {"episode_key": 3}):
            v, _ = db.record_chip_episode(self._conn, REF, bad)
            self.assertEqual(v, "bad_episode", repr(bad))

    def test_not_open_refuses(self):
        db.resolve_chip(self._conn, "done", chip_ref=REF,
                        note="class quiet", resolved_by_session_id="s",
                        note_auto=False, force=False)
        v, p = db.record_chip_episode(self._conn, REF, _episode("k"))
        self.assertEqual(v, "not_open")
        self.assertEqual(p["status"], "done")

    def test_cap_drops_oldest_and_preserves_magnitude(self):
        old_cap = db.CHIP_EPISODE_CAP
        db.CHIP_EPISODE_CAP = 3
        try:
            for i in range(5):
                v, _ = db.record_chip_episode(self._conn, REF,
                                              _episode("since-%d" % i))
                self.assertEqual(v, "ok")
        finally:
            db.CHIP_EPISODE_CAP = old_cap
        entry = self._entry()
        self.assertEqual([e["episode_key"] for e in entry["episodes"]],
                         ["since-2", "since-3", "since-4"])
        self.assertEqual(entry["episodes_truncated"], 2)
        self.assertEqual(entry["episode_count"], 5,
                         "episode_count is the TRUE total, kept + dropped")


class TestEndpointAndRender(_Fixture):

    def test_endpoint_codes(self):
        code, out = app._chip_episode(
            self._conn, {"chip_ref": REF, "episode": _episode("k1")}, "h")
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "ok")
        code, out = app._chip_episode(
            self._conn, {"chip_ref": REF, "episode": _episode("k1")}, "h")
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "idempotent")
        code, _ = app._chip_episode(
            self._conn, {"chip_ref": "chip-nope", "episode": _episode("k")},
            "h")
        self.assertEqual(code, 404)
        code, _ = app._chip_episode(self._conn, {"episode": _episode("k")},
                                    "h")
        self.assertEqual(code, 400)
        db.resolve_chip(self._conn, "done", chip_ref=REF, note="n",
                        resolved_by_session_id="s", note_auto=False,
                        force=False)
        code, _ = app._chip_episode(
            self._conn, {"chip_ref": REF, "episode": _episode("k2")}, "h")
        self.assertEqual(code, 409)

    def test_dispatch_table_membership(self):
        self.assertIn("/chip/episode", app._TASK_CLAIM_CHIP_POST)

    def test_render_carries_episodes_verbatim(self):
        db.record_chip_episode(self._conn, REF,
                               _episode("since-1", depth=1))
        source_doc = {"schema_version": "task_chips/v1",
                      "chips": [_chip()]}
        render_text, stats, snaps = writer.render_chips(
            self._conn, source_doc=source_doc)
        rendered = json.loads(render_text)
        row = next(c for c in rendered["chips"] if c["chip_ref"] == REF)
        self.assertEqual(row["episodes"][0]["episode_key"], "since-1")
        self.assertEqual(row["episode_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
