"""Regression pin for the WI-1 substrate-split dropped-records defect.

BACKGROUND. `WI-1` split `ree-v3/CLAUDE.md`'s "SD Design Decisions Implemented"
section into one file per feature under `docs/substrate/`, keyed on `## `
headings. About 55 records had no heading of their own in the pre-split file --
they were appended as top-level `- <ID>: ...` bullets under a PRE-EXISTING
heading for a DIFFERENT feature -- so the split folded each into the
neighbouring feature's file with no heading and no index entry, making it
undiscoverable by id (confirmed instance: `SD-WAYPOINT-FIELD`, folded into
`SD-018-amend-encoder-resource-field.md` with no trace in `CLAUDE.md`).

Recovered in `ree-v3` `dedcc2ce24` ("docs(substrate): un-fold 55 feature
records the WI-1 split buried under neighbouring headings"): every dropped
record now has its own `## ` heading in the correct per-feature file, and its
own index entry in `CLAUDE.md`'s "## Substrate feature index" section. This
file is the test pin `chip-20260908-wi1-substrate-split-dropped-records` asked
for -- it does not re-run the recovery, it prevents the defect's two
observable symptoms (a file untracked by the index; a record's bullet sitting
under someone else's heading) from returning silently.

Two independent checks, matching the chip's own framing:

  (1) INDEX <-> FILE bijection: every `docs/substrate/*.md` file is linked
      from CLAUDE.md's index section, and every link the index section makes
      into `docs/substrate/` resolves to a real file. This is the literal
      symptom from the incident ("grep SD-WAYPOINT-FIELD in CLAUDE.md returned
      nothing") stated as a link audit, matching the recovery commit's own
      "255/255/0 broken" verification.
  (2) NO FOREIGN BULLET: no top-level `- <SD|MECH|ARC>-...:` bullet inside a
      docs/substrate file names an id outside that file's own heading's id
      set -- the exact shape of the drop (SD-WAYPOINT-FIELD's bullet sitting
      under SD-018's heading). Deliberately scoped to files whose OWN heading
      names at least one SD-/MECH-/ARC- id: a roll-up-ledger file (heading
      "## SD Design Decisions Implemented", no id in the heading itself,
      documented in CLAUDE.md's index as "grep here first when no single file
      owns an sd_id") is BY DESIGN a many-ids-one-file bundle and is not the
      shape this check is for.

WHERE THIS FILE LIVES, AND WHY THE PREDICATE IS STILL STRICT (decided
2026-09-19, chip-20260919-wi1-index-contract-prose-false-positive).

Check (2) cannot tell an id DECLARATION from ordinary PROSE that happens to
open with an id. On 2026-09-19 ree-v3 65c1f72 landed the prose bullet
`- MECH-094 does **not** apply to the WRITEBACK call: ...` under MECH-018's
heading. This file then lived in tests/contracts/, which
scripts/precommit_contracts.sh Block 2 runs on every staged ree_core/** change,
so that one red blocked EVERY ree_core commit fleet-wide for ~6.5h (cleared by
ree-v3 1b4c78caaf, which prefixed the bullet with `Note: `).

The defect was the gate's POSITION, not its predicate. Measured: nothing that
ran this check fired on the commit able to break it. Block 2 keys on
ree_core/** and experiments/_lib/**; contract-tests.yml's `paths:` filter
excludes docs/** and CLAUDE.md. A docs-only commit (65c1f72's exact shape:
CLAUDE.md + one docs/substrate file, authored on a cloud worker, where the
commit guards are deliberately not installed) was checked by nobody, and the
first person to find out was whoever committed ree_core next. So:

  * MOVED out of tests/contracts/ into tests/docs_integrity/. A pure-text
    markdown lint must not be able to block a CODE commit. It is still
    collected by every `tests/`-rooted run (contract-tests.yml,
    remote_pytest.sh's default six roots, the integration-branch merge gate).
  * precommit_contracts.sh Block 1e runs it, locally and sub-second, when
    docs/substrate/*.md or CLAUDE.md is STAGED -- the block lands on the
    author, at the moment the fix costs one word. Pinned by
    tests/contracts/test_precommit_contracts_docs_integrity_scope.py.
  * .github/workflows/docs-integrity.yml runs it on any push touching those
    paths, so a commit from a box with no commit guards (the actual incident)
    still goes red ON THE PUSHED COMMIT ITSELF, within seconds, torch-free.

The predicate was deliberately NOT narrowed (the chip's option (b): require
`<id>:` or `<id> --` before calling a bullet a declaration). Two measurements:

  * Against the 35 true WI-1 defects at dedcc2ce24^ that wording catches 23
    and MISSES 12 -- `- ARC-071 / MECH-324: ...`, `- SD-070 ADOPTION in ...`,
    `- SD-e1-rollout-consistency-training ITEM 1: ...`, `- MECH-204 Phase 7 /
    Option B: ...`, `- SD-SLEEP-ENTRY-PRESSURE (sleep_substrate:GAP-9 ...`.
    SD-WAYPOINT-FIELD itself is among the 23, so the regression pin named
    below would have stayed green while a third of the gate's strength went.
    Nor is "prose starts lowercase" usable: legitimate record bullets read
    `- MECH-341 amend -- IMPLEMENTED`, `- MECH-091 names THREE salient ...`.
  * GOV-HELDOUT-1 (held-out check, REE_Working CLAUDE.md). Running the OLD
    predicate at all 24 commits that touch docs/substrate/ yields exactly ONE
    case where old and any prose-tolerant wording disagree -- the motivating
    incident. Three were required; one exists. Outcome recorded: the narrowing
    is scoped to its own incident and was NOT shipped.

With the block on the author a false positive costs a one-word reword, while a
false negative is a silently buried feature record -- the defect that took a
55-record recovery. That asymmetry is why the predicate stays strict. House
style for docs/substrate/ follows from it: do not OPEN a top-level bullet with
an SD-/MECH-/ARC- id the file's heading does not own. Lead with a word
(`Note: MECH-094 does not ...`) or reword. If the bullet really IS a feature
record, it needs its own file, its own `## ` heading, and its own index entry.
"""

import glob
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSTRATE_DIR = REPO_ROOT / "docs" / "substrate"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

_LINK_RE = re.compile(r"\(docs/substrate/([\w.\-]+\.md)\)")

# Matches a heading/bullet-leading id: SD-/MECH-/ARC- followed by alnum
# segments joined by '-' or '_' (covers SD-032b, MECH-090, ARC-071,
# SD-e1-rollout-consistency-training, SD-DECISIONS-IMPLEMENTED, etc.).
_ID_RE = re.compile(r"\b((?:SD|MECH|ARC)-[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*)\b")
_BULLET_ID_RE = re.compile(r"^- ((?:SD|MECH|ARC)-[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*)\b")


def _index_section_text() -> str:
    text = CLAUDE_MD.read_text(encoding="utf-8")
    marker = "## Substrate feature index"
    assert marker in text, (
        f"{CLAUDE_MD}: no {marker!r} section -- has the index been renamed or "
        "removed? This test cannot audit a section it cannot find.")
    return text[text.index(marker):]


def _substrate_files():
    files = sorted(Path(p) for p in glob.glob(str(SUBSTRATE_DIR / "*.md")))
    assert len(files) > 100, (
        f"only {len(files)} docs/substrate/*.md files found -- "
        "_substrate_files() going near-empty would make every check below "
        "vacuously pass.")
    return files


def test_every_substrate_file_is_linked_from_the_claude_md_index():
    """The literal symptom: a file present on disk but absent from the index
    is exactly SD-WAYPOINT-FIELD's original failure mode ('grep ... in
    CLAUDE.md returned nothing') -- except this check catches it as a whole
    FILE going unlinked, which is what a folded-in record with no heading of
    its own would have produced before dedcc2ce24 gave it one."""
    section = _index_section_text()
    linked = set(_LINK_RE.findall(section))
    actual = {p.name for p in _substrate_files()}

    missing = sorted(actual - linked)
    assert missing == [], (
        f"{len(missing)} docs/substrate/*.md file(s) exist but are not linked "
        f"from CLAUDE.md's index: {missing}")


def test_every_index_link_resolves_to_a_real_file():
    """The reverse direction: a stale link (renamed/deleted file, typo) is a
    broken pointer a reader would silently follow nowhere."""
    section = _index_section_text()
    linked = _LINK_RE.findall(section)
    actual = {p.name for p in _substrate_files()}

    broken = sorted(set(linked) - actual)
    assert broken == [], (
        f"{len(broken)} CLAUDE.md index link(s) point at a nonexistent "
        f"docs/substrate/*.md file: {broken}")


def test_the_index_links_each_file_exactly_once():
    """A file linked twice (or a link appearing twice) is not itself the WI-1
    defect, but it is the kind of drift a future split/merge produces --
    duplicate entries are a maintenance smell worth catching here rather than
    let the count silently stop meaning what section 1's audit claims."""
    section = _index_section_text()
    linked = _LINK_RE.findall(section)
    dupes = sorted({l for l in linked if linked.count(l) > 1})
    assert dupes == [], f"linked more than once in the index: {dupes}"


def _heading_ids(heading_line: str):
    return set(_ID_RE.findall(heading_line[3:]))


def test_no_bullet_declares_an_id_foreign_to_its_own_heading():
    """The structural shape of the drop itself: SD-WAYPOINT-FIELD's
    `- SD-WAYPOINT-FIELD: ...` bullet sat inside
    `SD-018-amend-encoder-resource-field.md`, under a heading naming only
    SD-018 -- an id neither equal to nor containing the bullet's own id.
    Verified as a regression pin against the pre-fix content
    (`dedcc2ce24^:docs/substrate/SD-018-amend-encoder-resource-field.md` line
    61 carries exactly this bullet under exactly that heading)."""
    offenders = []
    for path in _substrate_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        headings = [l for l in lines if l.startswith("## ")]
        assert headings, f"{path}: no '## ' heading at all"
        hids = _heading_ids(headings[0])
        if not hids:
            # A roll-up-ledger-shaped heading (no SD-/MECH-/ARC- id in the
            # heading text itself) is documented as a many-records bundle by
            # design (CLAUDE.md's index: "roll-up ledger ... grep here first
            # when no single file owns an sd_id") -- not the shape this
            # check is for.
            continue
        for line in lines:
            if not line.startswith("- "):
                continue
            m = _BULLET_ID_RE.match(line)
            if not m:
                continue
            bid = m.group(1)
            if bid not in hids:
                offenders.append((path.name, headings[0].strip(), bid))

    assert offenders == [], (
        f"{len(offenders)} bullet(s) name an id foreign to their file's own "
        f"heading (file, heading, foreign id): {offenders}\n"
        "FIX, one of: (a) the bullet is PROSE that merely opens with an id -- "
        "lead with a word instead (`- Note: MECH-094 does not ...`) or reword; "
        "this check cannot tell prose from a declaration and is strict on "
        "purpose (module docstring). (b) the bullet IS a feature record -- "
        "give it its own docs/substrate file with its own `## ` heading and "
        "its own entry in CLAUDE.md's Substrate feature index.")


def test_negative_control_a_genuinely_shared_heading_is_not_flagged():
    """Non-degeneracy check for the previous test: a file whose heading
    legitimately lists several ids (comma/slash/plus-separated -- e.g.
    'SD-032b / MECH-258 / MECH-260 / ARC-058') must NOT be flagged for
    bullets naming any of ITS OWN listed ids. Picks one such file that is
    known (from the WI-1 split's own design, predating the drop) to bundle
    more than one id under a single heading, and confirms it passes."""
    multi_id_files = []
    for path in _substrate_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        headings = [l for l in lines if l.startswith("## ")]
        hids = _heading_ids(headings[0]) if headings else set()
        if len(hids) > 1:
            multi_id_files.append((path, hids))

    assert multi_id_files, (
        "expected at least one docs/substrate file with a genuinely "
        "multi-id heading (e.g. SD-032b's dACC cluster) -- if none remain, "
        "this negative control has nothing left to guard and should be "
        "revisited rather than silently passing on an empty list.")

    for path, hids in multi_id_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.startswith("- "):
                continue
            m = _BULLET_ID_RE.match(line)
            if m:
                assert m.group(1) in hids, (
                    f"{path.name}: bullet id {m.group(1)!r} not in this "
                    f"file's own multi-id heading set {hids} -- either a "
                    "genuine drop or the heading parser under-counted.")
