"""Contract: the burned-queue-entry detector, and the false positives that
made a naive version useless.

Background -- the "silently burned queue entry" defect (fixed in ree-v3
d09127bb7039f12ca5a6f6ddc9c4de8cb6e0ae69, "coordinator: refuse to absorb a
NEW queue item into a TERMINAL row"):

  reconcile_once(upsert_only=True) upserted a NEW git-queue item onto a DB
  row that was already terminal -- keeping status='completed' while
  overwriting note / item_json / priority / script. phase3_queue_writer
  materialises only NON-terminal rows, so the next snapshot DELETED the
  freshly-committed entry from experiment_queue.json. The experiment never
  ran and nothing errored.

  This is the sibling of the phantom-completion taxonomy defect pinned in
  test_queue_removal_reason_recorded.py. Both stem from `status='completed'`
  meaning "no longer claimable" rather than "ran to a scientific outcome".
  There, a terminal row lost the REASON it went terminal; here, a terminal
  row silently ate a NEW item aimed at it.

  The ingress guard now refuses the upsert, so no NEW burns occur. What
  remains is that at the GIT layer a burned entry's deletion is
  indistinguishable from a normal post-completion removal -- which is why
  the 2026-07-21 audit found four ids that sessions re-queued BLIND
  (V3-EXQ-683 three times, V3-EXQ-686 three times).

Unit under test: scripts/audit_burned_queue_entries.py.

The contracts below are split deliberately:

  C1-C3  pure-logic contracts on synthetic stints -- these are the four
         legs, and they run anywhere.
  C4-C7  KNOWN-TRUTH contracts replayed against the real ree-v3 history and
         the real REE_assembly evidence tree. They are the ones that
         actually protect the detector, because every false-positive mode
         below was found by running it, not by reasoning about it. They
         SKIP when either tree is absent (cloud workers, CI, a bare
         worktree) rather than failing.

THE FALSE-POSITIVE MODES (a naive "<30 min after an operator add" filter
gave 93 hits, 85 of them benign):

  FP1  Fast experiments legitimately get claimed and complete within
       minutes. The window alone is worthless; the never-produced-a-manifest
       leg is the discriminator. C5.
  FP2  A driver writes the PARENT queue id into its manifest, not the
       letter-suffixed id it ran under -- V3-EXQ-734a ran fine but filed
       under V3-EXQ-734. So "did it run" is keyed on the SCRIPT, never on
       results.queue_id. C6.
  FP3  Pre-Phase-3 history (before 2026-05-29) is full of manual queue
       churn and is out of scope. C3.
  FP4  The inverse of FP2, and the one that cost a real miss: a
       letter-suffixed entry commonly REUSES the parent's script file, so
       "does a manifest for this stem exist at all" returns True from a run
       months earlier and clears a genuine burn. V3-EXQ-728a's first stint
       ran and wrote v3_exq_728_..._20260720T155414Z_v3.json; its second
       stint -- re-queued after the driver was rewired for SD-070 -- was
       burned 108 seconds later. The manifest test must be scoped to the
       STINT WINDOW. C2, C7.
"""

import datetime as dt
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_burned_queue_entries as audit  # noqa: E402

EVIDENCE_DIR = Path("/Users/dgolden/REE_Working/REE_assembly/"
                    "evidence/experiments")

# The 2026-07-21 audit's known-truth set.
MUST_FLAG = [
    "V3-EXQ-673",   # added 13:43 +01:00, gone 4.6 min later; the mech180
                    # script never ran. Id 673 had EARLIER run a DIFFERENT
                    # script (v3_exq_673_mech171_...) six times -- that is
                    # what made its DB row terminal.
    "V3-EXQ-683",   # re-queued 3x, blind
    "V3-EXQ-686",   # re-queued 3x, blind
    "V3-EXQ-654e",
    "V3-EXQ-728a",  # 108 seconds; the FP4 case
]

# Ran normally, or were deliberately replaced. Must not appear at all.
MUST_NOT_FLAG = [
    "V3-EXQ-734a",  # FP2 -- ran, manifest filed under parent V3-EXQ-734
    "V3-EXQ-569b",  # single stint, deliberate operator cancel -> 569c ran
    "V3-EXQ-490j",  # FP1 -- fast completer
    "V3-EXQ-757",   # FP1 -- fast completer
]


def _stint(queue_id="V3-EXQ-999", script="experiments/v3_exq_999_probe.py",
           added="2026-06-01T10:00:00+00:00",
           removed="2026-06-01T10:02:00+00:00",
           added_by_operator=True, removed_by_queue_snapshot=True,
           prior_stints=1, supersedes=None):
    item = {"queue_id": queue_id, "script": script}
    if supersedes:
        item["supersedes"] = supersedes
    return {
        "queue_id": queue_id, "item": item,
        "added_sha": "a" * 40, "added_at": added,
        "added_subject": "queue: %s" % queue_id,
        "added_by_operator": added_by_operator,
        "removed_sha": "b" * 40, "removed_at": removed,
        "removed_subject": "phase3-queue: snapshot",
        "removed_by_queue_snapshot": removed_by_queue_snapshot,
        "prior_stints": prior_stints,
    }


class _FakeEvidence:
    """Stand-in for EvidenceIndex: stem -> list of manifest datetimes."""

    def __init__(self, by_stem=None):
        self.by_stem = by_stem or {}

    def ran_ever(self, stem):
        return bool(stem and self.by_stem.get(stem))

    def ran_between(self, stem, start, end):
        return any(start <= w <= end for w in self.by_stem.get(stem, ()))

    def ran_after(self, stem, when_after):
        return any(w > when_after for w in self.by_stem.get(stem, ()))


def _utc(text):
    return dt.datetime.fromisoformat(text)


CUTOVER = _utc(audit.PHASE3_CUTOVER)


def _find(stints, evidence, successors=None, window=60.0, grace=120.0,
          cutover=CUTOVER):
    return audit.find_burns(stints, successors or {}, evidence, window,
                            grace, cutover)


class BurnDetectorLogicTest(unittest.TestCase):
    """C1-C3: the four legs, on synthetic stints."""

    # C1 -----------------------------------------------------------------
    def test_c1_all_four_legs_are_load_bearing(self):
        """Each leg alone must be able to clear a finding.

        Written as one table rather than four tests so that ADDING a leg
        without adding its negative case is visibly incomplete.
        """
        evidence = _FakeEvidence()
        self.assertEqual(len(_find([_stint()], evidence)), 1,
                         "the canonical burn must be flagged")

        cleared = {
            # L1: a phase3 writer added it, not an operator -- that is the
            # writer materialising a row, not a session queueing work.
            "L1 non-operator add": _stint(added_by_operator=False),
            # L2: removed by something other than a queue snapshot (e.g. an
            # operator's own governance retire commit).
            "L2 non-snapshot removal": _stint(
                removed_by_queue_snapshot=False),
            # L2: outside the window -- a long-lived entry that was claimed.
            "L2 outside window": _stint(
                removed="2026-06-01T18:00:00+00:00"),
            # L4: no prior stint, so the DB row cannot have been terminal.
            # This is what clears V3-EXQ-569b.
            "L4 first stint": _stint(prior_stints=0),
        }
        for label, stint in cleared.items():
            with self.subTest(leg=label):
                self.assertEqual(_find([stint], evidence), [],
                                 "%s must clear the finding" % label)

        # L3 has its own case below because it needs a populated evidence
        # index; asserted here only that it is reachable.
        ran = _FakeEvidence({"v3_exq_999_probe": [
            _utc("2026-06-01T10:01:00+00:00")]})
        self.assertEqual(_find([_stint()], ran), [],
                         "L3: a manifest inside the stint must clear it")

    # C2 -- FP4 ----------------------------------------------------------
    def test_c2_manifest_leg_is_scoped_to_the_stint(self):
        """THE REGRESSION THAT MATTERS -- the V3-EXQ-728a shape.

        A letter-suffixed entry reusing the parent's script has manifests
        from earlier stints. An all-time "has this stem ever run" test
        clears the burn; a stint-scoped one does not. If this reverts, the
        detector reports OK on the exact case it was built for.
        """
        stem = "v3_exq_728_trained_allon_capability_point"
        script = "experiments/%s.py" % stem
        stint = _stint(queue_id="V3-EXQ-728a", script=script,
                       added="2026-07-20T19:00:11+00:00",
                       removed="2026-07-20T19:01:59+00:00")

        # Manifests from BEFORE this stint: a run from 11 days earlier, and
        # one from the entry's OWN first stint 3 hours earlier.
        evidence = _FakeEvidence({stem: [
            _utc("2026-07-09T22:45:33+00:00"),
            _utc("2026-07-20T15:54:14+00:00"),
        ]})
        self.assertTrue(evidence.ran_ever(stem),
                        "fixture must have an all-time manifest, else this "
                        "test cannot detect the regression")

        findings = _find([stint], evidence)
        self.assertEqual([f["queue_id"] for f in findings], ["V3-EXQ-728a"])
        self.assertAlmostEqual(findings[0]["minutes_alive"], 1.8, places=1)
        self.assertFalse(findings[0]["evidence_recovered"],
                         "nothing ran after the burn -- science was lost")

    # C3 -- FP3 ----------------------------------------------------------
    def test_c3_pre_cutover_is_out_of_scope_but_still_seeds_prior_life(self):
        pre = _stint(removed="2026-05-01T10:02:00+00:00",
                     added="2026-05-01T10:00:00+00:00")
        self.assertEqual(_find([pre], _FakeEvidence()), [],
                         "pre-Phase-3 churn must not be reported")
        self.assertEqual(len(_find([pre], _FakeEvidence(), cutover=None)), 1,
                         "--all-history must still be able to see it")

    # C3b ----------------------------------------------------------------
    def test_c3b_disposition_separates_lost_from_recovered(self):
        """`evidence_recovered` demotes but never suppresses.

        Both arms: a declared `supersedes` successor that ran, and the same
        script re-running later under a fresh letter. The second arm must
        look strictly AFTER the stint -- V4-EXQ-001 and V3-EXQ-669b both
        have manifests from BEFORE their burned re-add, and those runs are
        precisely what the re-add existed to supersede.
        """
        stem = "v3_exq_999_probe"
        stint = _stint()

        before_only = _FakeEvidence({stem: [_utc("2026-05-30T00:00:00+00:00")]})
        finding = _find([stint], before_only)[0]
        self.assertFalse(finding["evidence_recovered"],
                         "a manifest from BEFORE the burn is not recovery")

        after = _FakeEvidence({stem: [_utc("2026-06-02T00:00:00+00:00")]})
        finding = _find([stint], after)[0]
        self.assertTrue(finding["evidence_recovered"])

        heir = _stint(queue_id="V3-EXQ-999a",
                      script="experiments/v3_exq_999a_probe.py")
        successors = {"V3-EXQ-999": [("V3-EXQ-999a",
                                      "2026-06-03T00:00:00+00:00")]}
        heir_ran = _FakeEvidence({"v3_exq_999a_probe": [
            _utc("2026-06-03T01:00:00+00:00")]})
        finding = _find([stint, heir], heir_ran, successors)[0]
        self.assertTrue(finding["evidence_recovered"])
        self.assertIn("V3-EXQ-999a", finding["recovered_by"])


@unittest.skipUnless(
    (REPO_ROOT / "experiment_queue.json").exists() and EVIDENCE_DIR.is_dir()
    and (REPO_ROOT / ".git").exists(),
    "needs the real ree-v3 git history and the REE_assembly evidence tree")
class BurnDetectorKnownTruthTest(unittest.TestCase):
    """C4-C7: replay against real history. The 2026-07-21 audit's answers."""

    @classmethod
    def setUpClass(cls):
        cls.findings = audit.audit(repo=str(REPO_ROOT),
                                   evidence_dir=str(EVIDENCE_DIR))
        cls.ids = {f["queue_id"] for f in cls.findings}

    # C4 -----------------------------------------------------------------
    def test_c4_every_confirmed_burn_is_flagged(self):
        for queue_id in MUST_FLAG:
            with self.subTest(queue_id=queue_id):
                self.assertIn(queue_id, self.ids)

    # C5 -- FP1 ----------------------------------------------------------
    def test_c5_benign_removals_are_not_flagged(self):
        for queue_id in MUST_NOT_FLAG:
            with self.subTest(queue_id=queue_id):
                self.assertNotIn(queue_id, self.ids)

    # C6 -- FP2 ----------------------------------------------------------
    def test_c6_parent_filed_manifest_clears_the_child(self):
        """V3-EXQ-734a ran; its manifest is filed under V3-EXQ-734.

        Asserted specifically (rather than only via C5) because it is the
        single case that proves the "did it run" test is keyed on the
        SCRIPT and not on results.queue_id.
        """
        self.assertNotIn("V3-EXQ-734a", self.ids)
        stem = "v3_exq_734_env_difficulty_competence_recovery_sweep"
        self.assertTrue(
            audit.EvidenceIndex(str(EVIDENCE_DIR)).ran_ever(stem),
            "fixture drift: 734's manifest is gone from the evidence tree")

    # C7 -----------------------------------------------------------------
    def test_c7_signal_stays_small_and_the_lost_set_is_pinned(self):
        """Precision, pinned. The naive filter gave 93 hits, 85 benign.

        A cap rather than an equality on the total, so that a genuinely new
        burn does not fail the suite -- but the LOST set is pinned exactly,
        because that is the set a session is expected to act on and it
        should not drift silently.

        KNOWN DIVERGENCE from the 2026-07-21 audit: V3-EXQ-592c and
        V3-EXQ-610a appear here (demoted to RECOVERED) although the audit
        listed them as must-not-flag. They are mechanically identical to
        the confirmed burns -- an operator re-add of an already-terminal id,
        deleted 17-36 seconds later with no manifest -- and the audit's call
        was about HARM (the science was recovered under 592d / 610b), not
        about mechanism. Reporting-but-demoting keeps both readings on the
        record. `--require-lost` drops them.
        """
        self.assertLessEqual(len(self.ids), 20,
                             "detector has started producing noise")

        lost = sorted({f["queue_id"] for f in self.findings
                       if not f["evidence_recovered"]})
        self.assertEqual(lost, [
            "V3-EXQ-569a",
            "V3-EXQ-669b",
            "V3-EXQ-673",
            "V3-EXQ-683",
            "V3-EXQ-686",
            "V3-EXQ-728a",
            "V4-EXQ-001",
        ])

        recovered = {f["queue_id"] for f in self.findings
                     if f["evidence_recovered"]}
        for queue_id in ("V3-EXQ-592c", "V3-EXQ-610a"):
            with self.subTest(queue_id=queue_id):
                self.assertIn(queue_id, recovered,
                              "known divergence must stay demoted, not "
                              "promoted to LOST")

        # V3-EXQ-654e is a must-flag that ALSO has a successor which ran
        # (654f), so it lands in RECOVERED. It is reported -- which is the
        # audit's requirement -- but a future tightening of the disposition
        # rule must not silently drop it.
        self.assertIn("V3-EXQ-654e", self.ids)

    # C8 -----------------------------------------------------------------
    def test_c8_blind_retries_are_reported_per_stint(self):
        """683 and 686 were each re-queued three times.

        One finding per burned STINT, not one per id -- collapsing them
        would hide how many times a session retried blind, which is the
        cost the audit was measuring.
        """
        for queue_id in ("V3-EXQ-683", "V3-EXQ-686"):
            with self.subTest(queue_id=queue_id):
                stints = [f for f in self.findings
                          if f["queue_id"] == queue_id]
                self.assertGreaterEqual(len(stints), 2)


if __name__ == "__main__":
    unittest.main()
