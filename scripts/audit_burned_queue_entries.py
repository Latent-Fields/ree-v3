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

DISPOSITION. Every finding carries `evidence_recovered`: true when the
science was eventually produced anyway, by ANY of three routes, all still
real defects at lower urgency:
  (1) a LATER queue entry declared `supersedes: <this id>` AND that
      successor's script produced a manifest;
  (2) the SAME script stem produced a manifest strictly after the stint;
  (3) RENUMBER -- an id collision was resolved by copying the driver to a
      fresh exq number, so the SAME descriptive slug (stem minus the
      "v[34]_exq_<number>_" prefix) ran under a DIFFERENT number after the
      stint. Routes (1) and (2) both miss this: no `supersedes` is declared
      and the stem differs. It is the most common collision-recovery shape
      here -- e.g. V3-EXQ-893 -> v3_exq_894_..., V3-EXQ-673 -> v3_exq_677_...,
      V3-EXQ-728a -> v3_exq_728b_...
`--require-lost` drops the recovered ones. See KNOWN DIVERGENCE in the
module tests for the two cases (V3-EXQ-592c, V3-EXQ-610a) where the
2026-07-21 audit called this benign and this detector still reports it,
demoted.

THE ALREADY-RAN ADVISORY (`already_ran_identical_script`). The LOST set is
"the set a session is expected to act on", and it holds two structurally
different things:

  (a) NEVER RAN UNDER ANY NUMBER -- real, unrecovered science loss
      (V3-EXQ-569a, V3-EXQ-683, V3-EXQ-686);
  (b) THE DECLARED SCRIPT ALREADY PRODUCED A MANIFEST BEFORE THE BURNED
      STINT, so that stint cannot have lost that run (V3-EXQ-895,
      V4-EXQ-001, V3-EXQ-929). Whether the re-add WANTED a fresh rerun is a
      human reading of intent and is not machine-visible, so the
      disposition does NOT move.

Three separate sessions hand-derived (b) from commit subjects and manifest
timestamps before it was made machine-visible. The advisory records the
partition; it is DISPOSITION-NEUTRAL by construction -- it never feeds
`evidence_recovered` and never removes an id from the LOST set. Demotion is
the dangerous direction, so this route (like the three recovery routes)
only ever annotates.

  IT IS NOT A TIMESTAMP TEST. "The declared stem produced a manifest before
  the stint" alone is WRONG on FP4: V3-EXQ-728a declares the PARENT's
  script, which ran on 2026-07-20T15:54:14Z and was then REWIRED for SD-070
  before the burned second stint -- so the earlier manifest is a DIFFERENT
  experiment and an advisory saying "728a already ran" is actively false.
  The advisory therefore fires only when the declared script's BLOB is
  byte-identical at both points: at the revision current when that earlier
  manifest was written, and at the burned stint's own add commit. A false
  advisory on the LOST set is exactly the failure the pin exists to
  prevent, so every step of it fails SILENT (missing path, missing blob,
  unparseable manifest timestamp) rather than guessing.
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


def cat_file_batch(repo, specs, oid_only=False):
    """Batch-read many `<rev>:<path>` objects in ONE `git cat-file` call.

    Returns spec -> content bytes, or spec -> blob oid (str) when
    `oid_only`. A spec git cannot resolve maps to None. Duplicate specs are
    asked for once.

    `oid_only` runs `--batch-check`, which emits only the object header. An
    IDENTITY question ("is this the same file at both revisions?") is then
    answered with no content transfer at all, because a git oid IS a content
    address -- equal oids mean byte-identical blobs. That matters for the
    already-ran advisory, which compares experiment DRIVERS (tens of KB
    each) rather than the single queue file this batching was built for.
    """
    ordered = []
    seen = set()
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            ordered.append(spec)
    if not ordered:
        return {}
    mode = "--batch-check" if oid_only else "--batch"
    stdin = "".join(spec + "\n" for spec in ordered)
    proc = subprocess.run(["git", "-C", repo, "cat-file", mode],
                          input=stdin.encode(), capture_output=True)
    out = proc.stdout
    result = {}
    pos = 0
    for spec in ordered:
        nl = out.find(b"\n", pos)
        if nl < 0:                                # truncated / no output
            result[spec] = None
            continue
        header = out[pos:nl].decode("utf-8", "replace").split()
        if len(header) == 3:                      # <oid> <type> <size>
            if oid_only:
                result[spec] = header[0]
                pos = nl + 1
            else:
                size = int(header[2])
                result[spec] = out[nl + 1:nl + 1 + size]
                pos = nl + 1 + size + 1           # trailing newline
        else:                                     # "missing" / "ambiguous"
            result[spec] = None
            pos = nl + 1
    return result


def read_blobs(repo, commits):
    """Batch-read <sha>:experiment_queue.json for every commit.

    One `git cat-file --batch` for the whole history -- ~1.3 s for 3.4k
    commits, versus minutes for a `git show` per commit.
    """
    contents = cat_file_batch(
        repo, ["%s:%s" % (sha, QUEUE_PATH) for sha, _, _ in commits])
    return {sha: contents.get("%s:%s" % (sha, QUEUE_PATH))
            for sha, _, _ in commits}


class ScriptBlobs:
    """Blob identity for a queue entry's declared script, at two revisions.

    Two questions, both needed by the already-ran advisory:
      `commit_at(path, when)` -- which revision held `path` at time `when`;
      `blob_ids(specs)`       -- the blob oid of each `<rev>:<path>`.

    Cost discipline, because the whole audit is ~4 s and must stay that way:
    path history is fetched LAZILY, one `git log` per DISTINCT script, and
    only for the handful of findings that have a prior run at all -- never
    per commit. The oid reads go through `cat_file_batch` as a single call.
    """

    def __init__(self, repo):
        self.repo = repo
        self._commit_at = {}

    def commit_at(self, path, when, tip):
        """Last commit at or before `when` that TOUCHED `path`, walking back
        from `tip`. None if there is none.

        Reading the blob at that commit is exact rather than approximate:
        nothing changed the file between it and `when`, by construction.

        Two cost decisions, because a path-limited `git log` walks history
        rather than indexing it:

        `-1` stops at the first match instead of enumerating the path's
        whole history (0.025 s vs 0.260 s per script, measured).

        `tip` is the BURNED STINT'S OWN ADD COMMIT, not HEAD. Walking from
        HEAD costs the whole distance back to `when` -- 0.22 s for a
        three-month-old finding, and the corpus is mostly old ones. From the
        add commit the distance is `added - when`, hours rather than months.
        It is also the more faithful tip: that commit's ancestry is exactly
        the main-line history the earlier run pulled its driver from, so a
        side-branch commit merged in afterwards (dated before `when`, but
        never in either tree) is correctly invisible.

        `--before` filters on committer date, so the answer is always at or
        before `when`; under clock skew it could be an EARLIER such commit
        rather than the latest, which errs toward a different blob and hence
        toward a SILENT advisory. A path git knows nothing about, or a tip
        it cannot resolve, is silent for the same reason.
        """
        key = (path, when, tip)
        if key in self._commit_at:
            return self._commit_at[key]
        try:
            out = _git(self.repo, "log", "-1", "--format=%H",
                       "--before=%s" % when.isoformat(), tip, "--", path)
        except RuntimeError:
            out = ""
        sha = out.strip() or None
        self._commit_at[key] = sha
        return sha

    def blob_ids(self, specs):
        return cat_file_batch(self.repo, specs, oid_only=True)


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


# A stem is "v[34]_exq_<number><letters>_<descriptive slug>". The RENUMBER
# recovery route keys on the descriptive slug: an id COLLISION is resolved by
# copying the driver to a fresh exq number (v3_exq_893_foo -> v3_exq_894_foo),
# which declares no `supersedes` and has a DIFFERENT stem -- so neither the
# same-stem "re-ran" route nor the `supersedes` route ever sees the recovery.
STEM_SPLIT_RE = re.compile(r"^(v[34])_exq_(\d+[a-z]*)_(.+)$")


def split_stem(stem):
    """v3_exq_728b_trained_allon -> ('v3', '728b', 'trained_allon').

    The number token KEEPS its letter suffix, so a renumber that differs only
    by the letter (728 -> 728b, the V3-EXQ-728a case) still reads as a
    DIFFERENT number. Returns (None, None, None) for a non-conforming stem.
    """
    if not stem:
        return None, None, None
    match = STEM_SPLIT_RE.match(stem)
    if not match:
        return None, None, None
    return match.group(1), match.group(2), match.group(3)


def renumber_recovery(by_stem, stem, when_after):
    """RENUMBER recovery: did a DIFFERENT-numbered entry with the SAME
    descriptive slug produce a manifest strictly after the stint?

    Returns the recovering entry's EXQ id (e.g. "V3-EXQ-894") for the
    disposition label, or None. `by_stem` maps stem -> iterable of manifest
    datetimes -- both EvidenceIndex.by_stem and the module tests' fake share
    that shape, so the slug logic has exactly one home.

    Disposition ONLY, like the other two recovery routes: it demotes a
    finding to RECOVERED, it never suppresses one. The `when_after` bound is
    the same removed+grace the same-stem route uses, so a same-slug run that
    predates the burn (e.g. V3-EXQ-569c/d, which ran BEFORE V3-EXQ-569a was
    burned) does NOT count as recovery.
    """
    _, number, slug = split_stem(stem)
    if not slug:
        return None
    matches = []
    for other, stamps in by_stem.items():
        ogen, onum, oslug = split_stem(other)
        if oslug != slug or onum is None or onum == number:
            continue
        if any(when > when_after for when in stamps):
            matches.append((onum, "%s-EXQ-%s" % (ogen.upper(), onum)))
    if not matches:
        return None
    return sorted(matches)[0][1]


def prior_own_run(by_stem, stem, before):
    """Latest manifest THIS EXACT stem produced strictly BEFORE `before`.

    Half of the already-ran advisory -- the timestamp half, which on its own
    is the REFUTED route (it fires on V3-EXQ-728a, whose declared script was
    rewired between the two stints). The caller must gate it on blob
    identity; this function deliberately does not know about dispositions.

    `by_stem` maps stem -> iterable of manifest datetimes, the same shape
    EvidenceIndex.by_stem and the module tests' fake share, so the lookup has
    exactly one home (as with renumber_recovery). A stem whose timestamp
    could not be parsed is absent from `by_stem` and therefore silent, which
    is the safe direction for an advisory.
    """
    if not stem:
        return None
    stamps = [when for when in by_stem.get(stem, ()) if when < before]
    return max(stamps) if stamps else None


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

    def renumbered_run_after(self, stem, when_after):
        """RENUMBER recovery route -- see module-level renumber_recovery.

        Disposition only. Catches the collision-renumber recovery that the
        same-stem and `supersedes` routes both miss.
        """
        return renumber_recovery(self.by_stem, stem, when_after)


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


def annotate_already_ran(pairs, by_stem, script_blobs):
    """Fill the ADVISORY `already_ran_identical_script` on each finding.

    Fires when the entry's DECLARED SCRIPT produced a manifest before the
    burned stint AND that script's blob is byte-identical at both points --
    at the revision current when the earlier manifest was written, and at
    the burned stint's own add commit. Blob identity is what makes it safe:
    the timestamp alone fires on V3-EXQ-728a, which declares the PARENT's
    driver, ran it on 2026-07-20, and was then re-queued after that driver
    was REWIRED for SD-070. The earlier manifest is a different experiment
    there, so an advisory saying "728a already ran" would be actively false.

    STRICTLY ADVISORY: it never touches `evidence_recovered` and never
    removes a finding. Demotion is the dangerous direction (see the REFUTED
    ROUTE block in the module tests), so this route only annotates.

    `pairs` is [(finding, stint)]. Two phases so that every oid read goes
    through ONE batched `git cat-file` -- phase 1 resolves which revisions
    to compare, phase 2 reads them all at once.
    """
    if script_blobs is None:
        return
    wanted = []
    specs = []
    for finding, stint in pairs:
        script = stint["item"].get("script")
        added = parse_iso(stint["added_at"])
        when = prior_own_run(by_stem, script_stem(script), added)
        if when is None or not script:
            continue
        then_sha = script_blobs.commit_at(script, when, stint["added_sha"])
        if then_sha is None:
            continue
        then_spec = "%s:%s" % (then_sha, script)
        now_spec = "%s:%s" % (stint["added_sha"], script)
        wanted.append((finding, then_spec, now_spec, when))
        specs.extend((then_spec, now_spec))
    if not specs:
        return
    oids = script_blobs.blob_ids(specs)
    for finding, then_spec, now_spec, when in wanted:
        then_oid = oids.get(then_spec)
        now_oid = oids.get(now_spec)
        # Fail SILENT on anything unresolvable: a missing blob means the
        # script was not in one of the two trees, which is not evidence that
        # it is the same file.
        if not then_oid or not now_oid or then_oid != now_oid:
            continue
        finding["already_ran_identical_script"] = True
        finding["already_ran_identical_script_at"] = when.strftime(
            "%Y-%m-%dT%H:%M:%SZ")


def find_burns(stints, successors, evidence, window_minutes, grace_minutes,
               cutover, script_blobs=None):
    findings = []
    annotated = []
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
        # to produce eventually got produced anyway -- by any of THREE routes,
        # all still a real defect at lower urgency:
        #   (1) a declared `supersedes` successor ran;
        #   (2) the SAME script ran later (typically re-queued under a fresh
        #       letter that reuses the driver);
        #   (3) RENUMBER: the same descriptive slug ran under a DIFFERENT exq
        #       number after an id collision (v3_exq_893_foo -> v3_exq_894_foo),
        #       which declares no `supersedes` and has a different stem, so
        #       routes (1) and (2) both miss it. This is the route this
        #       codebase uses most often for collisions.
        # All three must look strictly AFTER the stint: V4-EXQ-001 and
        # V3-EXQ-669b have manifests from BEFORE their burned re-add (exactly
        # what the re-add was superseding), and V3-EXQ-569c/d ran BEFORE
        # V3-EXQ-569a was burned -- none of those are recovery.
        heirs = successors.get(stint["queue_id"], [])
        recovered = [
            qid for qid, _ in heirs
            if evidence.ran_ever(_successor_stem(stints, qid))
        ]
        rerun_later = evidence.ran_after(stem, removed + grace)
        if rerun_later:
            recovered.append("(same script re-ran)")
        # The renumber route uses plain `removed`, NOT removed+grace: the
        # recovering run has a different stem/number, so it can never be the
        # stint's OWN late-finishing run (the thing grace exists to exclude in
        # the same-stem route). A collision-renumber commonly lands minutes
        # after the burn -- V3-EXQ-893 was renumbered to 894 only 19 min
        # after removal, inside the 120 min grace -- so removed+grace would
        # miss the actual renumber and mis-attribute recovery to a later run.
        renumbered = evidence.renumbered_run_after(stem, removed)
        if renumbered:
            recovered.append("(renumbered: %s)" % renumbered)
        finding = {
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
            # ADVISORY, filled below. Present unconditionally so consumers
            # can read the field without knowing whether a blob provider
            # was supplied.
            "already_ran_identical_script": False,
            "already_ran_identical_script_at": None,
        }
        findings.append(finding)
        annotated.append((finding, stint))
    annotate_already_ran(annotated, getattr(evidence, "by_stem", {}),
                         script_blobs)
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
                      window_minutes, grace_minutes, cutover,
                      ScriptBlobs(repo))


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
            if f["already_ran_identical_script"]:
                print("      ADVISORY: this entry's declared script already "
                      "ran at %s"
                      % f["already_ran_identical_script_at"])
                print("      from a BYTE-IDENTICAL blob, so the burned "
                      "re-add cannot have lost")
                print("      that run. Disposition unchanged -- whether the "
                      "re-add wanted a")
                print("      RERUN is not machine-visible.")
        print()

    already_ran = sorted({f["queue_id"] for f in lost
                          if f["already_ran_identical_script"]})
    if already_ran:
        print("EVIDENCE LOST splits in two. %d of the %d lost id(s) already "
              "ran under their" % (len(already_ran),
                                   len({f["queue_id"] for f in lost})))
        print("own number from an identical script (advisory above): %s"
              % ", ".join(already_ran))
        print("The rest never ran under any number -- those are the "
              "unrecovered science.")
        print()

    ids = sorted({f["queue_id"] for f in findings})
    print("%d finding(s) across %d queue id(s): %s"
          % (len(findings), len(ids), ", ".join(ids)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
