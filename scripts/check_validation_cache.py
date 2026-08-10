#!/opt/local/bin/python3
"""Regression tests for validation_cache.py (the precommit_contracts.sh Block 2
tree-content-hash-keyed validation-result cache).

Design record: REE_assembly/docs/architecture/landing_integration_worker_investigation.md
section 6/7. Covers the safety invariants that document lists as non-negotiable:
cache hit within TTL, miss after TTL expiry, miss on a different machine class /
toolchain, miss when the relevant path-set content changes, fail-open on a
corrupt or missing cache file, and that a failing run never writes an entry.

DELIBERATELY NAMED check_*, not test_*/*_test.py -- this must never be
collected by remote_pytest.sh's default args. validation_cache.py is Mac-only
infrastructure (it cross-imports REE_Working/scripts/task_claim.py, which
remote_pytest.sh's staging rsync does NOT ship to a cloud worker -- only
ree-v3/ and REE_assembly/evidence/experiments/scripts/ are staged), so this
file could not even import successfully there; `--selftest`'s tree-enumeration
check (CLAUDE.md "Full suite means SIX paths, not tests/") would otherwise
require it to be reachable from DEFAULT_PYTEST_ARGS, which would break every
remote run the first time this file collected. A test_*.py name would trip
that. Matches the established REE_Working/scripts/test_chip_ledger.py
convention for git-hook-tier tooling tests in spirit (real argparse, real git
repo, real ree_commit.py subprocess -- never stubbed -- against a throwaway
temp repo), just relocated into ree-v3/scripts/ where the naming escape hatch
is needed to keep it out of the suite.

Run: /opt/local/bin/python3 scripts/check_validation_cache.py
ASCII-only. Exits 0 on pass.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import validation_cache as vc  # noqa: E402

TIER = "test-tier"


class _ValidationCacheTestBase(unittest.TestCase):
    """Shared fixture: a throwaway git repo shaped like a minimal ree-v3
    checkout (ree_core/ + experiments/_lib/ present, both non-empty so the
    hash is over real content, not an empty-tree degenerate case)."""

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="validation_cache_test_"))
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        (self.repo / "ree_core").mkdir()
        (self.repo / "experiments" / "_lib").mkdir(parents=True)
        (self.repo / "ree_core" / "a.py").write_text("x = 1\n")
        (self.repo / "experiments" / "_lib" / "b.py").write_text("y = 2\n")
        self._git("init", "-q", ".")
        self._git("config", "user.name", "Test")
        self._git("config", "user.email", "test@example.com")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "base")
        self.cache_path = self.repo / vc.DEFAULT_CACHE_REL
        self._machine_class_patch("class-A")

    def _git(self, *args):
        p = subprocess.run(["git", "-C", str(self.repo)] + list(args),
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, "git %s failed: %s" % (args, p.stderr))
        return p.stdout.strip()

    def _machine_class_patch(self, value):
        old = vc.compute_machine_class
        self.addCleanup(setattr, vc, "compute_machine_class", old)
        vc.compute_machine_class = lambda: value

    def _record_pass(self, session_id="test-session", push=False):
        return vc._write_and_commit_pass(
            self.cache_path, self.repo, TIER, session_id, push, True, vc.REE_COMMIT_DEFAULT)

    def _lookup(self, ttl_minutes=45):
        return vc.lookup(self.cache_path, self.repo, TIER, ttl_minutes)

    def _committed_cache(self):
        raw = self._git("show", "HEAD:%s" % vc.DEFAULT_CACHE_REL)
        return json.loads(raw)

    def _backdate_only_record(self, minutes_ago):
        """Directly rewrite the single record's recorded_at, bypassing the
        normal write path -- the deterministic way to simulate TTL expiry
        without sleeping in a test."""
        data = json.loads(self.cache_path.read_text())
        (key,) = data["records"].keys()
        ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        data["records"][key]["recorded_at"] = ts
        self.cache_path.write_text(json.dumps(data, indent=2) + "\n")


# --------------------------------------------------------------------------
# Pure key/hash components
# --------------------------------------------------------------------------

class HashAndKeyTest(_ValidationCacheTestBase):
    def test_hash_is_stable_for_unchanged_content(self):
        h1 = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        h2 = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        self.assertEqual(h1, h2)

    def test_hash_changes_when_relevant_file_changes(self):
        h1 = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        (self.repo / "ree_core" / "a.py").write_text("x = 2\n")
        h2 = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        self.assertNotEqual(h1, h2)

    def test_hash_ignores_files_outside_relevant_trees(self):
        h1 = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        (self.repo / "README.md").write_text("irrelevant\n")
        h2 = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        self.assertEqual(h1, h2, "a file outside ree_core/ and experiments/_lib/ must not "
                                 "affect the key -- it would cause needless misses")

    def test_hash_reads_working_tree_not_head(self):
        """The suite validates on-disk content (incl. uncommitted edits), not
        HEAD -- both the local run and remote_pytest.sh's rsync ship the
        working tree. The hash must track that, not the last commit."""
        h_committed = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        (self.repo / "ree_core" / "a.py").write_text("uncommitted change\n")
        h_dirty = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        self.assertNotEqual(h_committed, h_dirty)

    def test_cache_key_differs_by_tier(self):
        rel = vc.compute_relevant_hash(self.repo)["relevant_hash"]
        k1 = vc.cache_key_for("tier-a", rel, "class-A")
        k2 = vc.cache_key_for("tier-b", rel, "class-A")
        self.assertNotEqual(k1, k2)


# --------------------------------------------------------------------------
# check/lookup
# --------------------------------------------------------------------------

class LookupTest(_ValidationCacheTestBase):
    def test_miss_when_no_cache_file_exists(self):
        self.assertFalse(self.cache_path.exists())
        hit, info = self._lookup()
        self.assertFalse(hit)
        self.assertNotIn("error", info)

    def test_hit_within_ttl(self):
        self._record_pass()
        hit, info = self._lookup(ttl_minutes=45)
        self.assertTrue(hit, info)
        self.assertEqual(info["reason"], "hit")
        self.assertIn("recorded_by_commit", info)

    def test_miss_after_ttl_expiry(self):
        self._record_pass()
        self._backdate_only_record(minutes_ago=90)
        hit, info = self._lookup(ttl_minutes=45)
        self.assertFalse(hit, info)
        self.assertIn("TTL", info["reason"])

    def test_boundary_just_inside_ttl_is_a_hit(self):
        self._record_pass()
        self._backdate_only_record(minutes_ago=44)
        hit, info = self._lookup(ttl_minutes=45)
        self.assertTrue(hit, info)

    def test_miss_on_different_machine_class(self):
        self._record_pass()  # recorded under class-A (setUp default)
        self._machine_class_patch("class-B")
        hit, info = self._lookup()
        self.assertFalse(hit, info)
        self.assertEqual(info["machine_class"], "class-B")

    def test_miss_on_changed_relevant_path_set_content(self):
        self._record_pass()
        (self.repo / "ree_core" / "a.py").write_text("x = 999  # changed\n")
        hit, info = self._lookup()
        self.assertFalse(hit, info)
        self.assertEqual(info["reason"], "no record for this key")

    def test_hit_survives_an_irrelevant_file_change(self):
        self._record_pass()
        (self.repo / "README.md").write_text("noise\n")
        hit, info = self._lookup()
        self.assertTrue(hit, info)


# --------------------------------------------------------------------------
# Fail-open
# --------------------------------------------------------------------------

class FailOpenTest(_ValidationCacheTestBase):
    def test_fail_open_on_corrupt_cache_file(self):
        self.cache_path.write_text("{ this is not valid json ]]]")
        hit, info = self._lookup()
        self.assertFalse(hit)
        self.assertNotIn("error", info, "a corrupt file is a clean MISS via load_cache(), "
                                        "not a caught exception")

    def test_fail_open_on_wrong_shaped_json(self):
        self.cache_path.write_text(json.dumps({"not": "the expected shape"}))
        hit, info = self._lookup()
        self.assertFalse(hit)

    def test_fail_open_on_empty_file(self):
        self.cache_path.write_text("")
        hit, info = self._lookup()
        self.assertFalse(hit)

    def test_fail_open_on_unreadable_repo_root(self):
        """An exception anywhere in the lookup path (e.g. a bogus repo_root)
        resolves to a miss with the error recorded, never a raise."""
        hit, info = vc.lookup(self.cache_path, "/this/path/does/not/exist/at/all", TIER, 45)
        self.assertFalse(hit)
        # compute_relevant_hash tolerates a missing tree (returns an empty hash)
        # rather than raising, so this specific case is a clean miss too -- the
        # real assertion is that NOTHING raised out of lookup().


# --------------------------------------------------------------------------
# record: pass writes + commits, fail never does
# --------------------------------------------------------------------------

class RecordTest(_ValidationCacheTestBase):
    def test_record_pass_writes_and_commits(self):
        head_before = self._git("rev-parse", "HEAD")
        status = self._record_pass()
        self.assertIn("written and committed", status, status)
        self.assertTrue(self.cache_path.exists())
        head_after = self._git("rev-parse", "HEAD")
        self.assertNotEqual(head_before, head_after, "record pass must create a commit")
        committed = self._committed_cache()
        self.assertEqual(len(committed["records"]), 1)
        (record,) = committed["records"].values()
        self.assertEqual(record["result"], "pass")
        self.assertEqual(record["tier"], TIER)
        self.assertEqual(record["recorded_by_session_id"], "test-session")

    def test_a_failing_run_never_writes_a_cache_entry(self):
        head_before = self._git("rev-parse", "HEAD")
        rc = vc.build_parser().parse_args(
            ["record", "--repo-root", str(self.repo), "--tier", TIER,
             "--cache-path", str(self.cache_path), "--result", "fail"])
        exit_code = vc.cmd_record(rc)
        self.assertEqual(exit_code, 0, "record must never itself signal a failure")
        self.assertFalse(self.cache_path.exists(), "a FAIL result must never create the cache file")
        self.assertEqual(self._git("rev-parse", "HEAD"), head_before,
                         "a FAIL result must never create a commit")

    def test_a_failing_run_does_not_disturb_an_existing_hit(self):
        """A later FAIL on the same content must not invalidate a prior PASS
        entry -- record --result fail is a pure no-op, not a cache clear."""
        self._record_pass()
        rc = vc.build_parser().parse_args(
            ["record", "--repo-root", str(self.repo), "--tier", TIER,
             "--cache-path", str(self.cache_path), "--result", "fail"])
        vc.cmd_record(rc)
        hit, info = self._lookup()
        self.assertTrue(hit, info)

    def test_record_pass_upserts_the_same_key_rather_than_appending(self):
        self._record_pass(session_id="session-1")
        self._record_pass(session_id="session-2")
        committed = self._committed_cache()
        self.assertEqual(len(committed["records"]), 1, "re-recording the same content/class/"
                                                        "tier must overwrite, not accumulate")
        (record,) = committed["records"].values()
        self.assertEqual(record["recorded_by_session_id"], "session-2")

    def test_record_pass_for_a_second_machine_class_keeps_both_entries(self):
        self._record_pass()
        self._machine_class_patch("class-B")
        self._record_pass()
        committed = self._committed_cache()
        self.assertEqual(len(committed["records"]), 2)

    def test_record_never_raises_even_with_a_broken_ree_commit_path(self):
        """A broken ree_commit.py must degrade to a logged failure, never an
        exception -- record() is called from a git hook and must not be able
        to take the caller's own `git commit` down with it. The write itself
        (atomic_write_text) happens before the commit attempt, so the file
        may exist on disk uncommitted afterward; what must never happen is a
        COMMIT for it (that would mean corrupt/partial content landed)."""
        status = vc._write_and_commit_pass(
            self.cache_path, self.repo, TIER, "s", False, True,
            Path("/does/not/exist/ree_commit.py"))
        self.assertIn("FAILED", status)
        self.assertFalse(self._git_has_commit_for_cache(),
                         "a broken ree_commit.py must never result in a commit")

    def _git_has_commit_for_cache(self):
        log = self._git("log", "--oneline", "--", vc.DEFAULT_CACHE_REL)
        return bool(log)


# --------------------------------------------------------------------------
# CLI (real argparse, via build_parser()/cmd_check/cmd_record directly --
# mirrors test_chip_ledger.py's _run() pattern of driving the real parser)
# --------------------------------------------------------------------------

class CliTest(_ValidationCacheTestBase):
    def _cli(self, *argv):
        args = vc.build_parser().parse_args(list(argv))
        return args.func(args)

    def test_cli_check_exit_codes(self):
        rc = self._cli("check", "--repo-root", str(self.repo), "--tier", TIER,
                       "--cache-path", str(self.cache_path))
        self.assertEqual(rc, 1, "miss must exit 1")
        self._record_pass()
        rc = self._cli("check", "--repo-root", str(self.repo), "--tier", TIER,
                       "--cache-path", str(self.cache_path))
        self.assertEqual(rc, 0, "hit must exit 0")

    def test_cli_record_always_exits_0(self):
        rc = self._cli("record", "--repo-root", str(self.repo), "--tier", TIER,
                       "--cache-path", str(self.cache_path), "--result", "pass",
                       "--session-id", "s")
        self.assertEqual(rc, 0)
        rc = self._cli("record", "--repo-root", str(self.repo), "--tier", TIER,
                       "--cache-path", str(self.cache_path), "--result", "fail")
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
