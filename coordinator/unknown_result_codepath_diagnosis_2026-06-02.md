# Runner UNKNOWN-result code path — diagnosis (insights_report.md Recommendation 1)

Date: 2026-06-02T06:38Z
Author session: runner UNKNOWN code-path diagnosis (insights Rec-1)
Scope: **code-side only.** The data-quality backfill of the 193 historical
UNKNOWN rows in `runner_status.json` is owned by a separate active session
(`TASK_CLAIMS.json`: "triage UNKNOWN runner_status results (cloud silent-drop
manifest relink)", claimed 2026-06-02T06:35Z). This memo does not touch
`runner_status.json` or any evidence manifest.

---

## TL;DR

**The silent-drop leak is already fixed.** It was closed on **2026-05-08** by
commit `f36461d` ("runner: sentinel-file conformance contract replacing stdout
regex scraping"). The `experiment_runner.py:1394` citation in
`REE_assembly/insights_report.md` Recommendation 1 (and in the
`reference_cloud_workers` memory) is **stale** — that line number no longer
exists and the leak it described is closed. No new UNKNOWN row has been written
on **any** machine since the fix landed.

Recommended action: **do not change runner logic.** The defensive behavior the
task envisioned (don't let UNKNOWN silently remove a queue item; release the
claim and leave it in the queue) is already present and load-bearing. The only
remaining work is (a) the data-quality backfill already in flight in the
parallel session, and (b) correcting the two stale `:1394` citations.

---

## 1. What the UNKNOWN path is, and how it used to leak

`result_info["result"]` is initialised to `"UNKNOWN"`
(`experiment_runner.py:1889`). A run is supposed to be reclassified to
PASS / FAIL / ERROR / SUSPENDED before the queue-management block.

**Pre-2026-05-08 (the bug):** outcome was scraped from subprocess **stdout**
via a set of regexes (`RE_DONE_OUTCOME`, `RE_STATUS_LINE`, etc.). When a script
printed a verdict in a format none of the regexes matched (e.g. cloud scripts
emitting `**Status:** FAIL` markdown, or `[EXQ-056] PASS (5/5 criteria)`), the
result stayed `UNKNOWN`, and the old fall-through let `UNKNOWN` reach the
success/queue-removal block — the item was committed as "done" and dropped from
the queue with no manifest. This is the "line 1394" leak. The historical
fingerprint is visible in the residual rows: their `result_summary` fields
literally contain the verdict the regex missed, e.g.
`'... | Done. Outcome: FAIL'`, `'[EXQ-056] PASS (5/5 criteria)'`.

## 2. Current code path (post-`f36461d`) — leak closed three ways

Trace through `run_experiment()` after `proc.wait()`:

1. **Sentinel is authoritative** (`experiment_runner.py:2095-2116`). Each
   subprocess writes `evidence/experiments/_runner_signals/<queue_id>.json` via
   `experiment_protocol.emit_outcome()`. `_read_sentinel()` reads it; a valid
   `outcome ∈ {PASS, FAIL}` wins over any stdout guess; an invalid sentinel →
   ERROR. Stdout regex is now only a diagnostic cross-check.
2. **No sentinel → never UNKNOWN** (`:2117-2151`). If the sentinel is missing:
   stdout PASS/FAIL is trusted as a legacy path (with a retrofit NOTE);
   otherwise the result is set to **ERROR**. A belt-and-braces guard
   (`:2147`) converts any residual `exit_code != 0 ∧ UNKNOWN` to ERROR.
3. **UNKNOWN can never reach queue removal** (`:2937-2954`). Even if a result
   somehow remains `UNKNOWN` (sentinel absent **and** exit_code == 0 **and** no
   stdout verdict), an explicit guard logs loudly, **releases the claim**,
   leaves the item in the queue (`_pass_skip.add`), and `continue`s — **no
   completed row is written.**

In addition, the infra-crash interception (`:2773-2795`,
`_transient_exit_codes = {137, -9, -11, -15, 143}`) catches OOM/SIGKILL/SIGTERM
*before* the removal path and releases the claim, and the PASS/FAIL/ERROR
branches each enforce a manifest-existence contract (`_result_manifest_exists`)
that leaves the item in the queue if a named manifest is missing on disk
(`:2829-2837`, `:2903-2911`, `:2961-2969`).

## 3. Evidence the leak is closed

| Machine status file | completed | UNKNOWN | latest UNKNOWN `completed_at` |
|---|---|---|---|
| DLAPTOP-4.local | 608 | 168 | 2026-05-08T18:02:53Z |
| ree-cloud-1 | 245 | 27 | 2026-05-08T12:29:53Z |
| ree-cloud-2 | 190 | 10 | 2026-05-08T13:00:30Z |
| EWIN-PC | 77 | 26 | 2026-04-21T19:56:05Z |
| Daniel-PC | 28 | 5 | 2026-04-10T09:36:18Z |
| ree-cloud-3 / -4 / worker-3 | 150 / 147 / 133 | 2 / 2 / 2 | 2026-03-20T20:49:11Z |

**Every** machine's most recent UNKNOWN is on or before 2026-05-08 — the day
`f36461d` landed (14:36 +0100). Zero UNKNOWN rows since. The aggregate
`runner_status.json` (singular, legacy) shows 193 UNKNOWN, range
2026-03-20 → 2026-05-08T18:02:53Z. These are historical residue, not an active
leak.

## 4. Regression coverage

Existing source-text contract tests already lock in the sibling branches:
`test_runner_fail_branch_persists_result.py`,
`test_runner_sigterm_no_phantom_completion.py`,
`test_runner_post_result_align.py`. They reference the UNKNOWN guard only as a
**slice boundary**; none asserts the UNKNOWN branch's own behavior. This memo's
companion change adds `tests/contracts/test_runner_unknown_no_drop.py` to close
that gap (read-only source-text contract; **no runner logic change**):
the UNKNOWN guard must (a) appear before the manifest/PASS path, (b) call
`release_active_claim`, (c) `_pass_skip.add(queue_id)`, (d) NOT
`status["completed"].append` on its path, and (e) `continue`.

---

## Approach forks (for the user)

The task asked to surface forks rather than silently change runner behavior.

- **Fork A — runner code (recommended: no change).** The leak is closed and
  the defensive "log + release + leave-in-queue" behavior already exists behind
  the transient-exit-code handling. Implementing the proposed `log + raise`
  would be redundant and would *re-open* a worse failure mode (an exception in
  the result-classification block aborts the whole pass). Recommend: land only
  the additive regression test; leave logic untouched.

- **Fork B — data quality (owned by the parallel session).** 193 stale UNKNOWN
  rows still inflate iteration/error counts in `insights_report.md` and the
  explorer. Reclassify each from its `result_summary` verdict (many literally
  contain `Done. Outcome: PASS/FAIL`) or from the on-disk manifest. **Already
  claimed** by "triage UNKNOWN runner_status results" — do not duplicate.

- **Fork C — stale citation.** `insights_report.md` Recommendation 1 and the
  `reference_cloud_workers` memory both still say "fix `experiment_runner.py:1394`
  to stop the leak." That line no longer exists and the leak is closed. The
  insights generator likely re-derives the `:1394` string from the memory note
  each run — correcting the memory note stops it recurring. (User decision:
  these touch the insights report + auto-memory.)
