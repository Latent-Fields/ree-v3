"""Opt-in helper for a NEW driver to build a letter-drop-safe run_id.

THE DEFECT THIS ADDRESSES (confirmed failure_autopsy_V3-EXQ-920a_2026-08-16,
CLAUDE.md "run_id identifier hygiene"). Every existing driver builds its
run_id inline as `f"{EXPERIMENT_TYPE}_{ts}_v3"`, where `EXPERIMENT_TYPE` is a
module-level constant with no access to the runner-supplied `queue_id`. A
bug-fix re-queue under the EXQ versioning policy (CLAUDE.md "EXQ Versioning
and Supersession Policy") appends a letter to the queue_id (V3-EXQ-920 ->
V3-EXQ-920a) but reuses the driver byte-unchanged, so the run_id never
encodes the letter and two different runs share one run_id STEM. Measured
corpus-wide: 9 genuine letter-drops, 10 shared stems, spanning 6 families
from 2026-05 to 2026-08 -- rare but recurring. See
`REE_assembly/scripts/check_runid_letter_hygiene.py` for the detector.

WHY THIS IS OPT-IN, NOT A RETROFIT. ~1160 existing drivers each compute their
run_id inline; rewriting them is out of scope for a low-priority hygiene
fix and would touch actively-run code for no live benefit (their run_ids
are already committed and immutable -- CLAUDE.md "Narrow Edits Only": do
not retro-rename landed manifests). `make_run_id()` below is for a NEW
driver, or one being edited for another reason, to call INSTEAD OF the
inline f-string -- it costs nothing to adopt and closes the defect for any
future same-driver re-queue.

USAGE

    from experiments._lib.run_id import make_run_id
    ...
    result["run_id"] = make_run_id(EXPERIMENT_TYPE)

`queue_id` defaults to the runner-exported `REE_QUEUE_ID` env var (see
`experiment_protocol.py`'s docstring: "The runner sets REE_QUEUE_ID ..."),
so no change is needed to how the driver is invoked -- a lettered re-queue
under V3-EXQ-920a automatically yields
`v3_exq_920a_uncensored_survival_single_life_fishtank_<ts>_v3` instead of
`v3_exq_920_..._<ts>_v3`. The letter is inserted right after the
`v[34]_exq_<number>` segment, never appended at the very end, so the
run_id's own number/letter pair stays adjacent and greppable exactly like a
queue_id is -- and so `check_runid_letter_hygiene.py`'s detector (which
looks for `exq_<number><letters>_`) reads it as clean. The output
`evidence/experiments/<EXPERIMENT_TYPE>/` directory is UNCHANGED by this --
only the run_id (and therefore the manifest filename / run-pack subdir
name) picks up the letter, so nothing about the driver's existing output
layout needs to change to adopt this.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Optional

# V3-EXQ-920a -> "a"; V3-EXQ-920 -> "". Deliberately excludes the hyphenated /
# double-letter naming shapes ("742-m", "742m-b") -- those are a different,
# non-bug-fix convention and are out of scope for this helper.
_QUEUE_LETTER_RE = re.compile(r"^V\d+-EXQ-\d+([a-z])$")

# v3_exq_920_uncensored_survival_single_life_fishtank
#   -> group(1)="v3_exq_920", group(2)="_uncensored_survival_single_life_fishtank"
_EXPERIMENT_TYPE_NUMBER_RE = re.compile(r"^(v[34]_exq_\d+)(_.*)?$")


def queue_letter_suffix(queue_id: Optional[str] = None) -> str:
    """The bug-fix letter of `queue_id` (or REE_QUEUE_ID), "" if none/unlettered."""
    qid = queue_id if queue_id is not None else os.environ.get("REE_QUEUE_ID")
    if not qid:
        return ""
    m = _QUEUE_LETTER_RE.match(qid)
    return m.group(1) if m else ""


def make_run_id(experiment_type: str, ts: Optional[str] = None,
                 queue_id: Optional[str] = None) -> str:
    """Build a run_id for `experiment_type` that encodes the queue letter, if any.

    `ts` defaults to now (UTC, `%Y%m%dT%H%M%SZ` -- the corpus's compact form).
    Pass an explicit `ts` in a test for a deterministic result.
    """
    if ts is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    letter = queue_letter_suffix(queue_id)
    stamped_type = experiment_type
    if letter:
        m = _EXPERIMENT_TYPE_NUMBER_RE.match(experiment_type)
        if m:
            stamped_type = m.group(1) + letter + (m.group(2) or "")
        else:
            # Unrecognised EXPERIMENT_TYPE shape (no v3_exq_<number> prefix) --
            # append rather than silently drop the letter.
            stamped_type = experiment_type + letter
    return f"{stamped_type}_{ts}_v3"
