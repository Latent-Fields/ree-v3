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
        f"heading (file, heading, foreign id): {offenders}")


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
