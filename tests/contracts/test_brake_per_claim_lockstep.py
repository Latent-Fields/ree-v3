"""Contract: the three MOVE-3 re-derive-brake predicates agree, INCLUDING step 0.

THE DIVERGENCE (found by the fable red-team of failure_autopsy_V3-EXQ-983a,
2026-09-06, REE_assembly 71694d5b01, recorded in that artifact's red_team_pass and
learning_extracted). The brake has three predicates the skills say to keep in
lockstep:

  (1) `.claude/skills/failure-autopsy/SKILL.md` Step 7, the `counts(t, claim)`
      recipe (mirrored byte-identically in `.agents/skills/`);
  (2) `.claude/skills/queue-experiment/SKILL.md` Step 2.5b, the same recipe;
  (3) `validate_queue._autopsy_counts_toward_brake`, the COMMIT-HOOK consumer.

A per-claim exclusion landed in (1) and (2) on 2026-07-21, for the peripheral
co-tag case -- a run that did not exercise one of its `claim_ids` declares
`recommended_epistemic_category_per_claim: {THAT_CLAIM: standard}`, and that
claim's brake must not be advanced by a run that never tested it. It did NOT land
in (3). So from 2026-07-21 to 2026-09-07, an artifact with a per-claim `standard`
stamp, a blanket `standard` and direction `non_contributory` counted toward that
claim's brake at the commit hook and did not count under either skill: 2 hits
under the tool, 1 under the written rule.

RESOLVED 2026-09-07 by USER DECISION: the per-claim value WINS, which is the
documented intent, so the tool was brought to the rule rather than the rule to the
tool. MEASURED before landing, over the whole autopsy corpus (56 targets carry a
per-claim stamp): 36 claims change hit count and 9 cross the threshold of 2 --
ARC-070, ARC-107, MECH-122, MECH-135, MECH-303, MECH-342, MECH-428, MECH-449,
Q-040. Those names were put to the user; the loosening is deliberate.

THE SECOND DIVERGENCE -- the SCAN SHAPE (resolved 2026-09-07, same day).

The 2026-09-07 reconciliation above covered the PREDICATE only. The SCAN around it
diverged three ways, and nothing pinned it: `validate_queue._scan_substrate_ceiling_autopsies`
counted one hit per FILE per claim, taking the first matching target and ``break``ing,
with no status filter and no supersession -- while both SKILL.md snippets keyed per
RUN with no break (R1) and let the latest adjudication supersede (R2). Its docstring
still described the superseded first-match recipe.

MEASURED over the 488-file corpus before landing: 70 claims change count and 14 cross
the threshold of 2 -- 12 TIGHTEN (ARC-033, INV-044, MECH-025 1->4, MECH-057b, MECH-071,
MECH-186, MECH-188, MECH-216 1->6, Q-040, SD-005, SD-008, SD-016 1->6, as a grandfathered
cluster file adjudicating N runs becomes N hits rather than 1) and 2 RELEASE via R2
(MECH-140 2->1, MECH-471 2->1). Both directions were put to the user; adopting R1+R2 was
the decision. A third sub-question -- the status filter -- was decided the OTHER way:
the recipe's literal ``!= 'confirmed'`` is over-broad, since the corpus carries
`confirmed_revised`, `applied` and `resolved` (all MORE settled than a bare `confirmed`)
plus grandfathered artifacts with no `status` at all, so both the tool and the two
recipes now skip only a positively-UNSETTLED status. Zero corpus delta today; it is a
guard against a silent future under-count.

WHAT THIS FILE PINS. The three synthetic targets the brief names, the fact that
`claim=None` preserves the old blanket-only meaning (many existing callers pass
one argument), the PER-CLAIM evaluation point in the scanner, and -- the drift
detector -- the SKILL.md text itself, extracted and EXECUTED: both the `counts()`
PREDICATE and the whole SCAN block, compared against the implementation on fixtures
and across the live corpus. If either side moves, this fails.
"""
import ast
import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]      # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

import validate_queue as VQ  # noqa: E402

# The umbrella checkout, where the two skills live. Not inside ree-v3.
UMBRELLA = REPO_ROOT.parent
SKILL_PATHS = (
    UMBRELLA / ".claude/skills/failure-autopsy/SKILL.md",
    UMBRELLA / ".claude/skills/queue-experiment/SKILL.md",
    UMBRELLA / ".agents/skills/failure-autopsy/SKILL.md",
    UMBRELLA / ".agents/skills/queue-experiment/SKILL.md",
)
CLAIM = "MECH-342"


def _target(*, blanket, direction, per_claim=None, action=None):
    t = {
        "claim_ids": [CLAIM, "MECH-999"],
        "recommended_epistemic_category": blanket,
        "recommended_evidence_direction": direction,
    }
    if per_claim is not None:
        t["recommended_epistemic_category_per_claim"] = per_claim
    if action is not None:
        t["recommended_substrate_queue_entry"] = {"action": action}
    return t


# --------------------------------------------------------------------------- #
# (1) the three targets the brief names
# --------------------------------------------------------------------------- #

def test_blanket_only_standard_non_contributory_counts():
    """The pre-existing shape. Unchanged by step 0 -- absent a per-claim value the
    blanket one applies, so historical artifacts count exactly as before."""
    t = _target(blanket="standard", direction="non_contributory")
    assert VQ._autopsy_counts_toward_brake(t, CLAIM) is True


def test_per_claim_standard_for_this_claim_does_not_count():
    """The peripheral co-tag: this run did not exercise MECH-342, so it must not
    advance MECH-342's brake. This is the case that diverged."""
    t = _target(blanket="standard", direction="non_contributory",
                per_claim={CLAIM: "standard"})
    assert VQ._autopsy_counts_toward_brake(t, CLAIM) is False


def test_per_claim_substrate_ceiling_for_this_claim_counts():
    """A per-claim CEILING reading is still a ceiling reading -- step 0 excludes
    only a non-ceiling per-claim value."""
    t = _target(blanket="standard", direction="non_contributory",
                per_claim={CLAIM: "substrate_ceiling"})
    assert VQ._autopsy_counts_toward_brake(t, CLAIM) is True


def test_a_per_claim_stamp_for_a_DIFFERENT_claim_does_not_exclude_this_one():
    """Step 0 is per claim, not per target. A stamp naming MECH-999 says nothing
    about MECH-342."""
    t = _target(blanket="standard", direction="non_contributory",
                per_claim={"MECH-999": "standard"})
    assert VQ._autopsy_counts_toward_brake(t, CLAIM) is True
    assert VQ._autopsy_counts_toward_brake(t, "MECH-999") is False


# --------------------------------------------------------------------------- #
# (2) the optional argument, and why it is optional
# --------------------------------------------------------------------------- #

def test_omitting_the_claim_preserves_the_old_blanket_only_meaning():
    """Many callers and contracts pass one argument. Without a claim there is no
    per-claim value to consult, so the blanket category applies -- which is what
    this predicate has always done."""
    t = _target(blanket="standard", direction="non_contributory",
                per_claim={CLAIM: "standard"})
    assert VQ._autopsy_counts_toward_brake(t) is True          # blanket-only
    assert VQ._autopsy_counts_toward_brake(t, CLAIM) is False  # per-claim wins


def test_a_malformed_per_claim_field_is_ignored_rather_than_raising():
    """Artifact data is hand-written; a string or a list where a dict was expected
    must not take the commit hook down."""
    for bad in ("standard", ["standard"], 3, None):
        t = _target(blanket="standard", direction="non_contributory", per_claim=bad)
        assert VQ._autopsy_counts_toward_brake(t, CLAIM) is True


# --------------------------------------------------------------------------- #
# (3) the scanner evaluates PER CLAIM -- the bug that made step 0 unreachable
# --------------------------------------------------------------------------- #

def test_the_scanner_calls_the_predicate_once_per_claim(monkeypatch, tmp_path):
    """Until 2026-09-07 the scanner tested the target ONCE and then attributed it
    to every claim_id. With step 0 that is not merely imprecise, it is inert: a
    per-claim exclusion can never be seen. Pinned by observing the call signature."""
    seen = []
    real = VQ._autopsy_counts_toward_brake

    def spy(target, claim=None):
        seen.append(claim)
        return real(target, claim)

    art = {"status": "confirmed", "targets": [
        _target(blanket="standard", direction="non_contributory",
                per_claim={CLAIM: "standard"})]}
    d = tmp_path / "planning"
    d.mkdir()
    (d / "failure_autopsy_V3-EXQ-000_2026-09-07.json").write_text(
        json.dumps(art), encoding="utf-8")
    monkeypatch.setattr(VQ, "_find_planning_dir", lambda: d)
    monkeypatch.setattr(VQ, "_autopsy_counts_toward_brake", spy)

    out = VQ._scan_substrate_ceiling_autopsies()
    assert seen and all(c is not None for c in seen), seen
    assert set(seen) == {CLAIM, "MECH-999"}
    assert CLAIM not in out           # excluded by its per-claim stamp
    assert "MECH-999" in out          # its sibling still counts


# --------------------------------------------------------------------------- #
# (4) the DRIFT DETECTOR -- execute the SKILL.md recipe and compare
# --------------------------------------------------------------------------- #

_RECIPE_RE = re.compile(r"^def counts\(t, claim\):\n(?:(?:[ \t].*)?\n)+?(?=\S)", re.M)


def _extract_recipe(path: Path):
    """Pull `counts(t, claim)` out of a SKILL.md and make it callable.

    Skips rather than fails when the file is absent: the ree-v3 suite runs from a
    staged tree on a cloud worker (remote_pytest.sh), and the umbrella's skills are
    not guaranteed to be in it. Failing closed there would be a phantom contract
    failure that looks exactly like real drift -- the documented
    validate_queue._is_tracked trap.
    """
    if not path.exists():
        pytest.skip(f"{path} not reachable from this tree")
    m = _RECIPE_RE.search(path.read_text(encoding="utf-8"))
    if m is None:
        pytest.fail(f"the `counts(t, claim)` recipe is no longer in {path} -- if it "
                    "moved, update this contract in the same commit")
    ns = {
        "re": re,
        "NEG": re.compile(r"not[_ -]+substrate_ceiling"),
        "INSTRUMENT": VQ.RE_DERIVE_INSTRUMENT_CATEGORY_MARKERS,
    }
    exec(compile(m.group(0), str(path), "exec"), ns)   # noqa: S102 - fixture text
    return ns["counts"]


_FIXTURES = [
    _target(blanket="standard", direction="non_contributory"),
    _target(blanket="standard", direction="non_contributory", per_claim={CLAIM: "standard"}),
    _target(blanket="standard", direction="non_contributory", per_claim={CLAIM: "substrate_ceiling"}),
    _target(blanket="substrate_ceiling", direction="weakens"),
    _target(blanket="not_substrate_ceiling", direction="non_contributory"),
    _target(blanket="measurement_degeneracy", direction="non_contributory"),
    _target(blanket="measurement_degeneracy", direction="non_contributory", action="create"),
]


@pytest.mark.parametrize("path", SKILL_PATHS, ids=lambda p: str(p).split("REE_Working/")[-1])
def test_the_skill_recipe_and_the_tool_agree_on_every_fixture(path):
    counts = _extract_recipe(path)
    for i, t in enumerate(_FIXTURES):
        assert counts(t, CLAIM) is VQ._autopsy_counts_toward_brake(t, CLAIM), (
            f"predicate drift on fixture {i} between {path} and "
            "validate_queue._autopsy_counts_toward_brake")


@pytest.mark.parametrize("path", SKILL_PATHS[:2], ids=lambda p: str(p).split("REE_Working/")[-1])
def test_the_skill_recipe_and_the_tool_agree_across_the_LIVE_corpus(path):
    """Fixtures pin the shapes someone thought of. The corpus is the check on the
    shapes nobody did -- 56 targets carry a per-claim stamp."""
    counts = _extract_recipe(path)
    planning = VQ._find_planning_dir()
    if planning is None:
        pytest.skip("REE_assembly/evidence/planning not reachable from this tree")
    files = sorted(planning.glob("failure_autopsy_*.json"))
    if len(files) < 20:
        pytest.skip("autopsy corpus not present")
    n_compared = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for t in data.get("targets", []) or []:
            if not isinstance(t, dict):
                continue
            for claim in t.get("claim_ids", []) or []:
                if not isinstance(claim, str):
                    continue
                n_compared += 1
                assert counts(t, claim) is VQ._autopsy_counts_toward_brake(t, claim), (
                    f"predicate drift on {f.name} / {claim}")
    assert n_compared > 500, f"only {n_compared} (target, claim) pairs compared"


def test_the_two_skill_recipes_are_code_identical():
    """They are allowed to differ in COMMENTS and must not differ in CODE.

    Compared as ASTs, not as stripped text: the two DO differ in comments today
    (measured 2026-09-07 -- three lines, one of them a TRAILING comment on a code
    line, which a "drop lines starting with #" filter silently keeps and then
    reports as code drift).
    """
    dumps = []
    for p in SKILL_PATHS[:2]:
        if not p.exists():
            pytest.skip(f"{p} not reachable from this tree")
        m = _RECIPE_RE.search(p.read_text(encoding="utf-8"))
        assert m, p
        dumps.append(ast.dump(ast.parse(m.group(0))))
    assert dumps[0] == dumps[1], "the two skill recipes' CODE has drifted apart"


def test_the_agents_mirrors_are_byte_identical_to_the_claude_copies():
    """CLAUDE.md 'Dual skill directories': the files should be identical."""
    for claude_p, agents_p in ((SKILL_PATHS[0], SKILL_PATHS[2]),
                               (SKILL_PATHS[1], SKILL_PATHS[3])):
        if not (claude_p.exists() and agents_p.exists()):
            pytest.skip("skills not reachable from this tree")
        assert claude_p.read_bytes() == agents_p.read_bytes(), (
            f"{agents_p} has drifted from {claude_p}")


# --------------------------------------------------------------------------- #
# (5) the SCAN SHAPE -- extract and execute the whole snippet, not just counts()
# --------------------------------------------------------------------------- #

# The snippet runs from the INSTRUMENT constant (the first line after the shell-
# interpolated `claim='$CLAIM'`, which must NOT be executed) to the first `print(`.
# Everything between is the preamble, `counts()`, and the scan that builds `hits`.
_SCAN_RE = re.compile(r"^INSTRUMENT=\(.*?(?=^print\()", re.M | re.S)


class _StubGlob:
    """Stands in for the `glob` module so the snippet's repo-relative corpus path
    resolves to the real planning dir wherever pytest is invoked from."""

    def __init__(self, files):
        self._files = list(files)

    def glob(self, _pattern):
        return list(self._files)


def _scan_hits(path: Path, claim: str, files) -> int:
    """Execute the SKILL.md scan block and return its hit count for `claim`."""
    m = _SCAN_RE.search(path.read_text(encoding="utf-8"))
    if m is None:
        pytest.fail(f"the brake SCAN block is no longer in {path} -- if it moved, "
                    "update this contract in the same commit")
    ns = {"re": re, "json": json, "glob": _StubGlob(files),
          "collections": collections, "claim": claim}
    exec(compile(m.group(0), str(path), "exec"), ns)   # noqa: S102 - fixture text
    return len(ns["hits"])


def _corpus_files():
    planning = VQ._find_planning_dir()
    if planning is None:
        pytest.skip("REE_assembly/evidence/planning not reachable from this tree")
    files = sorted(str(p) for p in planning.glob("failure_autopsy_*.json"))
    if len(files) < 20:
        pytest.skip("autopsy corpus not present")
    return files


@pytest.mark.parametrize("path", SKILL_PATHS[:2], ids=lambda p: str(p).split("REE_Working/")[-1])
def test_the_skill_SCAN_and_the_tool_agree_across_the_LIVE_corpus(path):
    """The predicate agreeing is not enough -- the scan AROUND it decides how many
    times the predicate's verdict is counted. That is where the 2026-09-07 second
    divergence lived: same `counts()`, 70 claims apart.

    Compares claim -> hit-count maps, which is exactly what the brake thresholds on.
    """
    if not path.exists():
        pytest.skip(f"{path} not reachable from this tree")
    files = _corpus_files()
    impl = {c: len(v) for c, v in VQ._scan_substrate_ceiling_autopsies().items()}
    assert impl, "the scanner found nothing -- corpus or planning dir is wrong"
    mismatches = []
    for claim in sorted(impl):
        got = _scan_hits(path, claim, files)
        if got != impl[claim]:
            mismatches.append((claim, got, impl[claim]))
    assert not mismatches, (
        f"SCAN drift between {path} and validate_queue._scan_substrate_ceiling_autopsies "
        f"-- (claim, skill_count, tool_count): {mismatches[:12]}"
        f"{' ...' if len(mismatches) > 12 else ''}")


@pytest.mark.parametrize("path", SKILL_PATHS[:2], ids=lambda p: str(p).split("REE_Working/")[-1])
def test_a_claim_the_tool_does_not_count_is_not_counted_by_the_skill_either(path):
    """The test above iterates the TOOL's keys, so a claim the tool drops entirely
    would never be compared. Check the other direction on a sample that includes the
    claims the 2026-09-07 measurement moved."""
    if not path.exists():
        pytest.skip(f"{path} not reachable from this tree")
    files = _corpus_files()
    impl = {c: len(v) for c, v in VQ._scan_substrate_ceiling_autopsies().items()}
    seen = set()
    for f in files:
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for t in data.get("targets", []) or []:
            if isinstance(t, dict):
                seen.update(c for c in (t.get("claim_ids") or []) if isinstance(c, str))
    uncounted = sorted(seen - set(impl))
    assert uncounted, "expected some tagged claims to have zero counted hits"
    for claim in uncounted[:40]:
        assert _scan_hits(path, claim, files) == 0, (
            f"{path} counts {claim} but the tool does not")


def test_the_scanner_skips_only_a_positively_unsettled_status(tmp_path, monkeypatch):
    """User decision 2026-09-07: an EXCLUSION list, not a `confirmed` allowlist.

    `applied` / `confirmed_revised` / `resolved` are all MORE settled than a bare
    `confirmed`, and grandfathered artifacts carry no `status` field at all -- an
    allowlist would silently stop counting every one of them, with no signal. All four
    of those shapes are live in the corpus today.
    """
    d = tmp_path / "planning"
    d.mkdir()
    target = _target(blanket="substrate_ceiling", direction="non_contributory")
    cases = {
        "confirmed": True, "confirmed_revised": True, "applied": True,
        "resolved": True, "__missing__": True,
        "proposed": False, "awaiting_human_confirmation": False,
    }
    for i, (status, should_count) in enumerate(cases.items()):
        art = {"targets": [dict(target, run_id=f"run-{i}")]}
        if status != "__missing__":
            art["status"] = status
        (d / f"failure_autopsy_case{i}_2026-09-0{i + 1}.json").write_text(
            json.dumps(art), encoding="utf-8")
    monkeypatch.setattr(VQ, "_find_planning_dir", lambda: d)
    out = VQ._scan_substrate_ceiling_autopsies()
    got = len(out.get(CLAIM, []))
    assert got == sum(1 for v in cases.values() if v), (
        f"expected {sum(1 for v in cases.values() if v)} counted, got {got}")


def test_R1_a_cluster_file_adjudicating_N_runs_is_N_hits(tmp_path, monkeypatch):
    """The tightening half of the 2026-09-07 scan decision: no break, key per RUN.
    Under the superseded shape this file was ONE hit."""
    d = tmp_path / "planning"
    d.mkdir()
    t = _target(blanket="substrate_ceiling", direction="non_contributory")
    (d / "failure_autopsy_cluster_2026-09-07.json").write_text(json.dumps({
        "status": "confirmed",
        "targets": [dict(t, run_id=f"v3_exq_{n}_v3") for n in (1, 2, 3)],
    }), encoding="utf-8")
    monkeypatch.setattr(VQ, "_find_planning_dir", lambda: d)
    assert len(VQ._scan_substrate_ceiling_autopsies().get(CLAIM, [])) == 3


def test_R1_id_less_targets_in_one_cluster_file_do_not_collide(tmp_path, monkeypatch):
    """The `<file>#<index>` fallback. Keying on the bare filename (as
    /queue-experiment's snippet did until 2026-09-07) merges these into one hit."""
    d = tmp_path / "planning"
    d.mkdir()
    t = _target(blanket="substrate_ceiling", direction="non_contributory")
    (d / "failure_autopsy_idless_2026-09-07.json").write_text(json.dumps({
        "status": "confirmed", "targets": [dict(t), dict(t)],
    }), encoding="utf-8")
    monkeypatch.setattr(VQ, "_find_planning_dir", lambda: d)
    assert len(VQ._scan_substrate_ceiling_autopsies().get(CLAIM, [])) == 2


def test_R2_a_re_adjudication_supersedes_rather_than_adding(tmp_path, monkeypatch):
    """The releasing half (MECH-140, MECH-471 in the corpus): the SAME run
    re-adjudicated later, withdrawing the ceiling reading, DECREMENTS to zero.
    Under the superseded shape both files counted, giving 2."""
    d = tmp_path / "planning"
    d.mkdir()
    run = "v3_exq_999_v3"
    (d / "failure_autopsy_first_2026-09-01.json").write_text(json.dumps({
        "status": "confirmed", "generated_utc": "2026-09-01T00:00:00Z",
        "targets": [dict(_target(blanket="substrate_ceiling",
                                 direction="non_contributory"), run_id=run)],
    }), encoding="utf-8")
    (d / "failure_autopsy_redo_2026-09-05.json").write_text(json.dumps({
        "status": "confirmed", "generated_utc": "2026-09-05T00:00:00Z",
        "targets": [dict(_target(blanket="measurement_invalid",
                                 direction="non_contributory"), run_id=run)],
    }), encoding="utf-8")
    monkeypatch.setattr(VQ, "_find_planning_dir", lambda: d)
    assert CLAIM not in VQ._scan_substrate_ceiling_autopsies()


def test_R2_falls_back_to_the_filename_date_when_generated_utc_is_absent(tmp_path, monkeypatch):
    """Grandfathered artifacts often lack `generated_utc`. Both snippets and the tool
    fall back to the trailing filename date -- an empty-string fallback would make
    every such artifact sort equal and decide R2 by iteration order."""
    d = tmp_path / "planning"
    d.mkdir()
    run = "v3_exq_998_v3"
    (d / "failure_autopsy_zzz_early_2026-09-01.json").write_text(json.dumps({
        "status": "confirmed",
        "targets": [dict(_target(blanket="substrate_ceiling",
                                 direction="non_contributory"), run_id=run)],
    }), encoding="utf-8")
    (d / "failure_autopsy_aaa_late_2026-09-05.json").write_text(json.dumps({
        "status": "confirmed",
        "targets": [dict(_target(blanket="measurement_invalid",
                                 direction="non_contributory"), run_id=run)],
    }), encoding="utf-8")
    monkeypatch.setattr(VQ, "_find_planning_dir", lambda: d)
    # The later DATE wins even though its filename sorts first.
    assert CLAIM not in VQ._scan_substrate_ceiling_autopsies()


def test_a_run_the_artifact_declared_a_dry_run_smoke_does_not_count(tmp_path, monkeypatch):
    """`excluded_dry_run_ids` is the DECLARED output of the /failure-autopsy Step 2a
    gate, and the brake counter must read it.

    Until 2026-09-07 it did not, and the corpus shows why that matters: MECH-216
    crossed the brake on six runs of one grandfathered cluster artifact -- an artifact
    that had already run the gate correctly, stamped `dry_run_checked: true`, and named
    all six in `excluded_dry_run_ids`. The artifact was right; the counter was not
    reading it. Honouring the field releases MECH-216 6 -> 0.

    Deliberately keyed on the DECLARED field, never on a run_id filename pattern: a
    heuristic would be the counter special-casing that Step 7 forbids.
    """
    d = tmp_path / "planning"
    d.mkdir()
    t = _target(blanket="substrate_ceiling", direction="non_contributory")
    (d / "failure_autopsy_smoke_2026-09-07.json").write_text(json.dumps({
        "status": "confirmed",
        "dry_run_checked": True,
        "excluded_dry_run_ids": ["v3_exq_1_dry_v3", "v3_exq_2_dry_v3"],
        "targets": [
            dict(t, run_id="v3_exq_1_dry_v3"),
            dict(t, run_id="v3_exq_2_dry_v3"),
            dict(t, run_id="v3_exq_3_real_v3"),
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(VQ, "_find_planning_dir", lambda: d)
    out = VQ._scan_substrate_ceiling_autopsies()
    kept = [e[2].get("run_id") for e in out.get(CLAIM, [])]
    assert kept == ["v3_exq_3_real_v3"], kept


def test_a_malformed_excluded_dry_run_ids_is_ignored_rather_than_raising(tmp_path, monkeypatch):
    """Artifact data is hand-written; the commit hook must not go down over it."""
    d = tmp_path / "planning"
    d.mkdir()
    t = _target(blanket="substrate_ceiling", direction="non_contributory")
    for i, bad in enumerate(("v3_exq_1_v3", {"a": 1}, 7, None)):
        art = {"status": "confirmed", "targets": [dict(t, run_id=f"run-{i}")]}
        art["excluded_dry_run_ids"] = bad
        (d / f"failure_autopsy_bad{i}_2026-09-0{i + 1}.json").write_text(
            json.dumps(art), encoding="utf-8")
    monkeypatch.setattr(VQ, "_find_planning_dir", lambda: d)
    assert len(VQ._scan_substrate_ceiling_autopsies().get(CLAIM, [])) == 4
