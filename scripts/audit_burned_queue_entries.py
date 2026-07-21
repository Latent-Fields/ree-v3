#!/usr/bin/env python3
"""Detect silently-BURNED experiment queue entries from ree-v3 git history.

THE DEFECT (fixed in ree-v3 d09127bb7039f12ca5a6f6ddc9c4de8cb6e0ae69,
"coordinator: refuse to absorb a NEW queue item into a TERMINAL row"):

  reconcile_once(upsert_only=True) upserted a NEW git-queue item onto a DB
  row that was ALREADY terminal. The upsert kept status='completed' while
  overwriting note / item_json / priority / script. phase3_queue_writer
  materialises only NON-terminal rows, so the very next snapshot DELETED the
  freshly-committed queue entry from ree-v3/experiment_queue.json.

  The experiment never ran. Nothing errored. Nothing was logged.

The ingress guard now refuses that upsert, so no NEW burns occur. What this
script addresses is the RESIDUAL problem: at the git layer a burned entry's
deletion is byte-identical to a normal post-completion removal. The
2026-07-21 audit found four ids that sessions re-queued BLIND -- V3-EXQ-683
three times, V3-EXQ-686 three times -- because nothing ever surfaced the drop.

THE SIGNATURE, four legs (all must hold):

  L1  the queue_id was ADDED by an OPERATOR commit (subject does not start
      with "phase3" -- i.e. a human/session commit, not a writer snapshot);
  L2  it was REMOVED by a "phase3-queue: snapshot" commit within
      --window-minutes (the confirmed cases ran 0.1 - 25 min; V3-EXQ-728a
      was 108 seconds);
  L3  NO evidence manifest was produced for the entry's declared `script`
      while that stint was live;
  L4  the queue_id had a PRIOR stint in the queue -- it had been present and
      removed before. This is the defect's causal precondition: the DB row
      can only be terminal-at-upsert if the id already had a life. It is
      also the blind-retry signature the audit actually saw.

FALSE-POSITIVE MODES THIS MUST SURVIVE (a naive "<30 min" filter gave 93
hits, 85 of them benign):

  FP1 Fast experiments legitimately get claimed and complete within minutes.
      The time window alone is worthless -- L3 is what discriminates. The
      raw L1+L2 candidate set is ~410; L3 cuts it to ~41.

  FP2 A driver writes the PARENT queue id into its manifest, not the
      letter-suffixed id it ran under. V3-EXQ-734a ran fine but its manifest
      is filed under queue_id V3-EXQ-734. So the "did it run" test is keyed
      on the SCRIPT STEM, never on results.queue_id == the entry's id.

  FP3 Pre-Phase-3 history (before 2026-05-29) is full of manual queue churn.
      Findings are restricted to the post-cutover window. NOTE the walk
      itself still starts from the beginning of history, because L4 needs to
      know whether an id had a life BEFORE the cutover -- truncating the
      walk would lose that and under-report early-post-cutover burns.

  FP4 (the subtle one, found while validating V3-EXQ-728a) A letter-suffixed
      entry often REUSES the parent's script file. "does a manifest for this
      stem exist at all" then returns True from a run months earlier and
      silently clears a real burn. V3-EXQ-728a's first stint ran and wrote
      v3_exq_728_..._20260720T155414Z_v3.json; its SECOND stint -- re-queued
      after the driver was rewired for SD-070 -- was burned 108 s later. So
      L3 is scoped to the STINT WINDOW [add_time, removal_time + grace],
      not to all time.

WHY A STANDALONE SCRIPT AND NOT validate_queue.py:

  validate_queue.py runs as a PreToolUse hook on EVERY `git commit`. This
  check is a full-history git walk -- ~3.4k commits, ~14 MB of blobs, ~4 s
  measured on the Mac. That is far too slow to pay per commit, and it
  answers a question about HISTORY rather than about the queue file being
  committed. It belongs here, run by a session or a cron.

USAGE

  /opt/local/bin/python3 ree-v3/scripts/audit_burned_queue_entries.py
  ... --json                 machine-readable findings
  ... --require-lost         only findings whose science was never recovered
  ... --window-minutes 60    widen/narrow L2
  ... --all-history          report pre-cutover findings too (noisy, FP3)

Exit status: 0 = no findings, 1 = findings, 2 = usage/environment error.

DISPOSITION. Every finding carries `evidence_recovered`: true when some
LATER queue entry declared `supersedes: <this id>` AND that successor's
script produced a manifest. Those are burns whose science was eventually
done under a different id -- real defects, lower urgency. `--require-lost`
drops them. See KNOWN DIVERGENCE in the module tests for the two cases
(V3-EXQ-592c, V3-EXQ-610a) where the 2026-07-21 audit called this benign
and this detector still reports it, demoted.
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

# Phase 3 cut over on 2026-05-28 / 2026-05-29. Before that, queue churn was
# manual and the burn mechanism did not exist (FP3).
PHASE3_CUTOVER = "2026-05-29T00:00:00+00:00"

QUEUE_PATH = "experiment_queue.json"

# A manifest / run-dir is named "<script stem>_<UTC timestamp>_v3[.json]".
TIMESTAMP_RE = re.compile(r"_(\d{8}T\d{6}Z)")

DEFAULT_REPO = "/Users/dgolden/REE_Working/ree-v3"
DEFAULT_EVIDENCE = ("/Users/dgolden/REE_Working/REE_assembly/"
                    "evidence/experiments")


# --------------------------------------------------------------------------
# git plumbing
# --------------------------------------------------------------------------

def _git(repo, *args):
    proc = subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args),
                                                  proc.stderr.strip()))
    return proc.stdout


def queue_commits(repo):
    """Every commit touching the queue file, oldest first.

    %cI (committer date) is used rather than %aI: the writers and the
    sessions disagree on author timezone, and committer date is what
    actually orders the ref.
    """
    out = _git(repo, "log", "--reverse", "--format=%H%x00%cI%x00%s",
               "--", QUEUE_PATH)
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, when, subject = line.split("\0", 2)
        commits.append((sha, when, subject))
    return commits


def read_blobs(repo, commits):
    """Batch-read <sha>:experiment_queue.json for every commit.

    One `git cat-file --batch` for the whole history -- ~1.3 s for 3.4k
    commits, versus minutes for a `git show` per commit.
    """
    stdin = "".join("%s:%s\n" % (sha, QUEUE_PATH) for sha, _, _ in commits)
    proc = subprocess.run(["git", "-C", repo, "cat-file", "--batch"],
                          input=stdin.encode(), capture_output=True)
    out = proc.stdout
    blobs = {}
    pos = 0
    for sha, _, _ in commits:
        nl = out.find(b"\n", pos)
        if nl < 0:
            break
        header = out[pos:nl].decode("utf-8", "replace").split()
        if len(header) == 3:                      # <oid> blob <size>
            size = int(header[2])
            blobs[sha] = out[nl + 1:nl + 1 + size]
            pos = nl + 1 + size + 1               # trailing newline
        else:                                     # "missing" / "ambiguous"
            blobs[sha] = None
            pos = nl + 1
    return blobs


def parse_queue(blob):
    """queue_id -> item, or None if the blob is not a readable queue."""
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except (ValueError, UnicodeDecodeError):
        return None
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None
    out = {}
    for item in items:
        if isinstance(item, dict) and item.get("queue_id"):
            out[item["queue_id"]] = item
    return out


# --------------------------------------------------------------------------
# evidence side
# --------------------------------------------------------------------------

def script_stem(script):
    """experiments/v3_exq_673_foo.py -> v3_exq_673_foo"""
    if not script:
        return None
    base = os.path.basename(script)
    return base[:-3] if base.endswith(".py") else base


class EvidenceIndex:
    """stem -> sorted list of manifest timestamps.

    Keyed on the SCRIPT STEM, never on queue_id (FP2). Both the flat
    `<stem>_<ts>_v3.json` manifests and the per-run `<stem>_<ts>_v3/`
    directories count as "this script produced evidence".
    """

    def __init__(self, evidence_dir):
        self.dir = evidence_dir
        self.by_stem = {}
        self.unparsed = set()
        if not os.path.isdir(evidence_dir):
            raise RuntimeError("evidence dir not found: %s" % evidence_dir)
        for name in os.listdir(evidence_dir):
            match = TIMESTAMP_RE.search(name)
            if not match:
                continue
            stem = name[:match.start()]
            try:
                when = dt.datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
            except ValueError:
                self.unparsed.add(stem)
                continue
            when = when.replace(tzinfo=dt.timezone.utc)
            self.by_stem.setdefault(stem, []).append(when)
        for stamps in self.by_stem.values():
            stamps.sort()

    def ran_ever(self, stem):
        return bool(stem and (stem in self.by_stem or stem in self.unparsed))

    def ran_between(self, stem, start, end):
        """Did this script produce a manifest while the stint was live?

        Scoped to the stint rather than to all time because a
        letter-suffixed entry commonly reuses its parent's script file, and
        an all-time test then clears a real burn with a months-old run
        (FP4 -- V3-EXQ-728a).

        Unparseable timestamps count as "ran": failing OPEN here costs a
        missed finding, failing closed would manufacture noise.
        """
        if not stem:
            return False
        if stem in self.unparsed:
            return True
        return any(start <= when <= end for when in self.by_stem.get(stem, ()))

    def ran_after(self, stem, when_after):
        """Did this script produce a manifest strictly after the stint?

        Used only for disposition, never to suppress a finding outright.
        """
        if not stem or stem in self.unparsed:
            return False
        return any(when > when_after for when in self.by_stem.get(stem, ()))


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

def parse_iso(text):
    return dt.datetime.fromisoformat(text)


def _supersedes_of(item):
    value = item.get("supersedes")
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def walk(repo, commits, blobs):
    """Reduce history to per-queue_id stints plus the supersession graph.

    A "stint" is one absent -> present -> absent cycle: how the id was added
    and how it left. Everything the four legs need is a property of a stint.
    """
    stints = []
    open_add = {}                 # queue_id -> pending add record
    successors = {}               # superseded id -> [(successor id, when)]
    prior_stint_count = {}        # queue_id -> stints already closed
    previous = None

    for sha, when, subject in commits:
        current = parse_queue(blobs.get(sha))
        if current is None:
            # Unreadable revision (a malformed intermediate commit). Skip it
            # WITHOUT advancing `previous`, so the add/remove diff is taken
            # across the gap rather than reported as a mass delete + re-add.
            continue
        if previous is not None:
            is_operator = not subject.startswith("phase3")
            is_queue_snapshot = subject.startswith("phase3-queue")

            for qid in set(current) - set(previous):
                open_add[qid] = {
                    "queue_id": qid,
                    "added_sha": sha,
                    "added_at": when,
                    "added_subject": subject,
                    "added_by_operator": is_operator,
                    "item": current[qid],
                }
                for superseded in _supersedes_of(current[qid]):
                    successors.setdefault(superseded, []).append((qid, when))

            for qid in set(previous) - set(current):
                add = open_add.pop(qid, None)
                if add is None:
                    continue
                add.update({
                    "removed_sha": sha,
                    "removed_at": when,
                    "removed_subject": subject,
                    "removed_by_queue_snapshot": is_queue_snapshot,
                    "prior_stints": prior_stint_count.get(qid, 0),
                })
                prior_stint_count[qid] = prior_stint_count.get(qid, 0) + 1
                stints.append(add)
        else:
            # Seed state: everything already in the first revision counts as
            # present, and as a prior life for L4 if it later leaves.
            for qid, item in current.items():
                open_add[qid] = {
                    "queue_id": qid, "added_sha": sha, "added_at": when,
                    "added_subject": subject, "added_by_operator": False,
                    "item": item,
                }
        previous = current

    return stints, successors


def find_burns(stints, successors, evidence, window_minutes, grace_minutes,
               cutover):
    findings = []
    for stint in stints:
        # L1 -- operator add
        if not stint["added_by_operator"]:
            continue
        # L2 -- removed by a phase3-queue snapshot, promptly
        if not stint.get("removed_by_queue_snapshot"):
            continue
        added = parse_iso(stint["added_at"])
        removed = parse_iso(stint["removed_at"])
        minutes = (removed - added).total_seconds() / 60.0
        if not 0 <= minutes <= window_minutes:
            continue
        # FP3 -- post-cutover only
        if cutover is not None and removed < cutover:
            continue
        # L3 -- nothing ran, scoped to THIS stint (FP2 + FP4)
        stem = script_stem(stint["item"].get("script"))
        grace = dt.timedelta(minutes=grace_minutes)
        if evidence.ran_between(stem, added, removed + grace):
            continue
        # L4 -- the id had a prior life, so the DB row could be terminal
        if stint["prior_stints"] < 1:
            continue

        # Disposition. A burn is "recovered" when the science it was meant
        # to produce eventually got produced anyway -- either because the
        # SAME script ran later (typically re-queued under a fresh letter),
        # or because a declared `supersedes` successor ran. Still a real
        # defect; lower urgency. Note the "same script ran later" arm must
        # look strictly AFTER the stint: V4-EXQ-001 and V3-EXQ-669b both
        # have manifests from BEFORE their burned re-add, and those runs are
        # exactly what the re-add was trying to supersede.
        heirs = successors.get(stint["queue_id"], [])
        recovered = [
            qid for qid, _ in heirs
            if evidence.ran_ever(_successor_stem(stints, qid))
        ]
        rerun_later = evidence.ran_after(stem, removed + grace)
        if rerun_later:
            recovered.append("(same script re-ran)")
        findings.append({
            "queue_id": stint["queue_id"],
            "script": stint["item"].get("script"),
            "added_at": stint["added_at"],
            "added_sha": stint["added_sha"][:10],
            "added_subject": stint["added_subject"][:100],
            "removed_at": stint["removed_at"],
            "removed_sha": stint["removed_sha"][:10],
            "minutes_alive": round(minutes, 2),
            "prior_stints": stint["prior_stints"],
            "evidence_recovered": bool(recovered),
            "recovered_by": recovered,
        })
    return findings


def _successor_stem(stints, qid):
    for stint in stints:
        if stint["queue_id"] == qid:
            return script_stem(stint["item"].get("script"))
    return None


def audit(repo=DEFAULT_REPO, evidence_dir=DEFAULT_EVIDENCE,
          window_minutes=60.0, grace_minutes=120.0, all_history=False):
    """Full audit. Returns findings, newest last."""
    commits = queue_commits(repo)
    blobs = read_blobs(repo, commits)
    stints, successors = walk(repo, commits, blobs)
    cutover = None if all_history else parse_iso(PHASE3_CUTOVER)
    return find_burns(stints, successors, EvidenceIndex(evidence_dir),
                      window_minutes, grace_minutes, cutover)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect silently-burned ree-v3 experiment queue entries.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--window-minutes", type=float, default=60.0,
                        help="max add->removal gap (L2). Confirmed burns ran "
                             "0.1-25 min; the manifest leg is what actually "
                             "discriminates, so this only bounds the search.")
    parser.add_argument("--grace-minutes", type=float, default=120.0,
                        help="how long after removal a manifest still counts "
                             "as belonging to the stint")
    parser.add_argument("--all-history", action="store_true",
                        help="include pre-Phase-3 findings (noisy)")
    parser.add_argument("--require-lost", action="store_true",
                        help="only burns whose science was never recovered "
                             "under a declared successor")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        findings = audit(args.repo, args.evidence_dir, args.window_minutes,
                         args.grace_minutes, args.all_history)
    except RuntimeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    if args.require_lost:
        findings = [f for f in findings if not f["evidence_recovered"]]

    if args.json:
        print(json.dumps(findings, indent=2))
        return 1 if findings else 0

    lost = [f for f in findings if not f["evidence_recovered"]]
    recovered = [f for f in findings if f["evidence_recovered"]]

    if not findings:
        print("OK: no burned queue entries found "
              "(post-cutover, window %.0f min)." % args.window_minutes)
        return 0

    print("BURNED QUEUE ENTRIES -- operator-added, deleted by the next "
          "phase3-queue snapshot,")
    print("with NO manifest for the declared script while the entry was live.")
    print()
    for label, group in (("EVIDENCE LOST", lost),
                         ("EVIDENCE RECOVERED (re-ran, or a successor ran)",
                          recovered)):
        if not group:
            continue
        print("== %s (%d) ==" % (label, len(group)))
        for f in group:
            print("  %-14s alive %7.1f min  (prior stints: %d)"
                  % (f["queue_id"], f["minutes_alive"], f["prior_stints"]))
            print("      script  %s" % f["script"])
            print("      added   %s  %s" % (f["added_at"], f["added_sha"]))
            print("      removed %s  %s" % (f["removed_at"], f["removed_sha"]))
            if f["recovered_by"]:
                print("      recovered by %s" % ", ".join(f["recovered_by"]))
        print()

    ids = sorted({f["queue_id"] for f in findings})
    print("%d finding(s) across %d queue id(s): %s"
          % (len(findings), len(ids), ", ".join(ids)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
