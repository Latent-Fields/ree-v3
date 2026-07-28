"""Corpus-wide contract: `experiments/` contains ZERO invalid escape sequences.

WHAT AN INVALID ESCAPE IS. `'\\|'` inside a non-raw string literal. `\\|` is not one
of Python's recognised escapes, so the tokenizer keeps the backslash verbatim AND
warns. It is a warning today and a **SyntaxError** in a future CPython. That is why
this is pinned rather than tolerated: once it hard-errors the file stops PARSING,
which takes out every lint and every import that touches it -- so it fails the whole
corpus gate, not one test. The blast radius grows with the corpus, and the fix is
one character.

WHY IT WENT UNNOTICED FOR SO LONG. The warning is raised by the tokenizer, not by a
lint, so it surfaced as an unattributed line on every pytest run that parses the
corpus:

    <unknown>:455: DeprecationWarning: invalid escape sequence '\\|'

`<unknown>` because `validate_queue.prereg_share_feasibility_lint` called
`ast.parse(source)` with no `filename=`. Nothing named the file, nothing failed, so
nobody chased it. Three real instances lived in
`v3_exq_015_three_stream_lateral.py` and `v3_exq_017_combined_lateral_reafference.py`
until 2026-07-28 (ree-v3 `37673f280b`). The corpus reached zero there; this contract
is what keeps it there.

THE CATEGORY IS NOT STABLE, THE MESSAGE IS. CPython raised this as
`DeprecationWarning` through 3.11 and as `SyntaxWarning` from 3.12 (verified on
3.13.9: `SyntaxWarning`). `conftest.INVALID_ESCAPE_MARKER` therefore matches the
message text; matching the category would silently stop firing on an interpreter
upgrade.

WHERE THE DETECTION LIVES. `conftest.invalid_escape_findings`, called from the
single shared corpus walk. It is deliberately NOT a `path_lints` entry -- a path
lint runs after the file is already parsed and cached, so its `ast.parse` is a cache
HIT, no real parse happens, no warning is raised, and the lint would never fire
while looking green. `test_detection_must_run_at_the_cache_miss` below pins exactly
that. See the `conftest` module docstring for the full reasoning.
"""

from __future__ import annotations

import ast
import warnings

import pytest

# `experiments/` is NOT enumerated here: the shared `corpus_scan` fixture walks it
# once for every corpus-wide contract. See tests/contracts/conftest.py.

_FIX_ADVICE = """
How to fix an invalid escape sequence:
  * The backslash is part of a REGEX -> make the literal raw: r"\\|" not "\\|".
  * A literal backslash is intended -> double it: "\\\\|" not "\\|".
  * The string is an f-string -> the same two options apply; rf"..." is available.

DO NOT "fix" it by deleting the backslash. The backslash is usually load-bearing:
  * in a regex, `\\|` matches a literal pipe while `|` means ALTERNATION -- deleting
    it silently changes what the pattern matches;
  * in a Markdown table cell, `\\|` is the correct escape for a literal pipe, and
    deleting it breaks the table by starting a new column.
Both of the 2026-07-28 corpus instances were the Markdown-table case -- they render
`||z_harm||` inside a results table -- and deleting the backslash would have
corrupted the rendered output while making the warning go away.
""".strip()


# ---- the pin ---------------------------------------------------------------------
def test_corpus_has_no_invalid_escape_sequences(corpus_scan):
    """THE CONTRACT: zero invalid escape sequences anywhere under `experiments/`."""
    found = corpus_scan.invalid_escapes
    assert found == [], (
        f"{len(found)} invalid escape sequence(s) in experiments/:\n  "
        + "\n  ".join(found)
        + "\n\n"
        + _FIX_ADVICE
    )


def test_the_pin_is_not_vacuous(corpus_scan):
    """A count pinned at 0 also passes when NOTHING was scanned.

    This repo hit that defect class three times on 2026-07-27 (a whole test root
    uncollected; ten gate assertions passing against an empty set), so the scan's
    own coverage is asserted rather than assumed. The escape check runs over the
    `rglob("*.py")` set -- every Python file under `experiments/`, not just the
    top-level `v3_exq_*.py` drivers.
    """
    assert corpus_scan.n_rglob_files > 500, (
        f"only {corpus_scan.n_rglob_files} files walked -- the corpus scan is not "
        f"covering experiments/, so the zero-pin above proves nothing")


# ---- NEGATIVE CONTROL: prove the check can actually fire --------------------------
def test_detector_fires_on_an_invalid_escape(tmp_path, invalid_escape_findings_fn):
    """A contract pinned at 0 that has never been shown to FAIL is indistinguishable
    from one that cannot fail. This is that demonstration, through the same function
    the corpus walk calls."""
    src = "s = 'a\\|b'\n"  # the exact shape found in the corpus on 2026-07-28
    p = tmp_path / "bad_escape.py"
    p.write_text(src, encoding="utf-8")

    found = invalid_escape_findings_fn(src, p)
    assert len(found) == 1, f"detector missed a known-bad escape: {found}"
    assert "bad_escape.py:1" in found[0], f"finding must name file and line: {found[0]}"
    assert "invalid escape sequence" in found[0]


def test_detector_reports_every_occurrence_and_its_line(tmp_path, invalid_escape_findings_fn):
    src = "a = 'x\\|y'\nb = 2\nc = \"p\\qr\"\n"
    p = tmp_path / "two_bad.py"
    p.write_text(src, encoding="utf-8")

    found = invalid_escape_findings_fn(src, p)
    assert len(found) == 2, f"expected 2 findings, got {found}"
    assert "two_bad.py:1" in found[0] and "two_bad.py:3" in found[1]


def test_detector_fires_on_an_f_string(tmp_path, invalid_escape_findings_fn):
    """The corpus instances were f-strings; a plain-literal-only check would miss them."""
    src = "v = 1\nrow = f'| {v} \\| done |'\n"
    p = tmp_path / "fstring.py"
    p.write_text(src, encoding="utf-8")
    assert len(invalid_escape_findings_fn(src, p)) == 1


@pytest.mark.parametrize("ambient", ["ignore", "error", "once", "default"])
def test_detector_is_immune_to_ambient_warning_filters(
    tmp_path, invalid_escape_findings_fn, ambient
):
    """Guards the `simplefilter("always")` inside the detector.

    MUTATION-VERIFIED, and be precise about which params do the work: with the
    `simplefilter` line deleted from `invalid_escape_findings`, the `ignore` and
    `error` params go to 0 findings and FAIL; `once` and `default` still pass. So
    the guard rests on those two, and `once`/`default` are baseline coverage that
    the detector behaves under ordinary configuration. Do not delete the first two
    thinking the other two still cover this -- they do not.

    An earlier version of this test tried to guard the same call by "warming"
    `__warningregistry__` with a prior parse, on the theory that the default
    once-per-location filter would then silence the detector. That test was
    VACUOUS: the tokenizer passes no registry for this warning, so repeated parses
    of the same file warn every time even under `default`, and removing
    `simplefilter("always")` did not make it fail. It was a contract that could not
    fail -- the exact defect class this file exists to prevent -- and it also leaked
    a stray warning into every suite run's summary.

    The real hazard is an ambient filter that suppresses the category, which is
    ordinary configuration: `-W ignore` / `-W error` on the command line, or a
    pytest `filterwarnings` ini entry. Measured on a file that HAS a bad escape,
    without the `simplefilter` call:

        ambient=ignore -> 0 findings        ambient=error -> 0 findings

    `error` is the nastier of the two: CPython turns a tokenizer warning-turned-
    error into a **SyntaxError**, which the detector's fail-soft `except` swallows,
    so the file reads as clean rather than raising. Either way the corpus pin would
    sit reassuringly at zero while real offenders went unreported.

    `catch_warnings()` restores the ambient filter state on exit, so this does not
    leak configuration into the rest of the session.
    """
    src = "s = 'a\\|b'\n"
    p = tmp_path / f"ambient_{ambient}.py"
    p.write_text(src, encoding="utf-8")

    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter(ambient)
        found = invalid_escape_findings_fn(src, p)

    assert len(found) == 1, (
        f"detector went blind under ambient filter '{ambient}' (got {found}). The "
        f"simplefilter('always') inside invalid_escape_findings is what makes it "
        f"independent of ambient warning configuration -- without it the corpus pin "
        f"passes vacuously under -W ignore / -W error or a pytest filterwarnings ini.")


# ---- FALSE-POSITIVE controls: the legitimate spellings must stay clean -------------
@pytest.mark.parametrize(
    "label, src",
    [
        ("raw string regex", 'import re\np = re.compile(r"a\\|b")\n'),
        ("escaped backslash", 's = "a\\\\|b"\n'),
        ("valid escapes", 's = "line\\nnext\\ttab\\\\end"\n'),
        ("raw f-string", "v = 1\ns = rf'{v}\\|x'\n"),
        ("bytes raw", 'b = rb"\\|"\n'),
        ("no backslash at all", 's = "plain text"\n'),
    ],
)
def test_detector_is_clean_on_legitimate_spellings(
    tmp_path, invalid_escape_findings_fn, label, src
):
    p = tmp_path / "ok.py"
    p.write_text(src, encoding="utf-8")
    assert invalid_escape_findings_fn(src, p) == [], f"false positive on {label}"


def test_detector_is_fail_soft_on_unparseable_source(tmp_path, invalid_escape_findings_fn):
    """An unparseable script is reported by `check_script`, not here. This must not
    raise -- a single broken file would otherwise error the whole corpus contract."""
    p = tmp_path / "broken.py"
    src = "def (:\n"
    p.write_text(src, encoding="utf-8")
    assert invalid_escape_findings_fn(src, p) == []


def test_detector_still_reports_an_escape_in_a_file_that_later_fails_to_parse(
    tmp_path, invalid_escape_findings_fn
):
    """Fail-soft must not mean fail-silent: the escape is real even if the file also
    has a syntax error further down.

    ALSO PINS THE DE-DUPLICATION, which is what makes this count portable. On a
    failed parse CPython's PEG parser re-tokenizes to build a better error message
    and the tokenizer re-emits every escape warning: this file yields the warning
    TWICE on linux/3.10 and ONCE on darwin/3.13. Without the dedupe in
    `invalid_escape_findings` the assertion below is machine-dependent -- it passed
    on the Mac and failed on `ree-worker-2` when first written, which is precisely
    the cross-machine-class trap this repo has been bitten by before.
    """
    p = tmp_path / "warned_then_broken.py"
    src = "s = 'a\\|b'\ndef (:\n"
    p.write_text(src, encoding="utf-8")
    found = invalid_escape_findings_fn(src, p)
    assert len(found) == 1, f"expected exactly one de-duplicated finding, got {found}"
    assert "warned_then_broken.py:1" in found[0]
    assert "invalid escape sequence" in found[0]


# ---- the wiring itself ------------------------------------------------------------
def test_detection_must_run_at_the_cache_miss(
    tmp_path, invalid_escape_findings_fn, last_parse_cache_cls
):
    """REGRESSION GUARD for why this is not a `path_lints` entry.

    The shared scan parses each file once and serves every later consumer from a
    one-entry cache. The tokenizer only warns on a REAL parse, so a consumer that
    runs after the file is cached sees nothing. A path lint is exactly such a
    consumer -- registering the escape check there would produce a contract that is
    permanently green and permanently blind.

    So the check must be the FIRST consumer of each file's source. If anyone ever
    moves it later in `scan_corpus`'s loop body, this is the test that explains why
    the corpus count silently went to zero.
    """
    src = "s = 'a\\|b'\n"
    p = tmp_path / "cached.py"
    p.write_text(src, encoding="utf-8")
    cache = last_parse_cache_cls(ast)

    at_the_miss = invalid_escape_findings_fn(src, p, cache.parse)
    after_the_hit = invalid_escape_findings_fn(src, p, cache.parse)

    assert len(at_the_miss) == 1, "the first (uncached) consumer must see the warning"
    assert after_the_hit == [], (
        "a cached parse raised the warning again -- if this ever becomes true the "
        "note above is stale, but while it holds, the escape check MUST run before "
        "any other consumer of the file")


def test_scan_pays_no_extra_parse_for_the_escape_check(corpus_scan):
    """The escape check takes the miss the pre-registration lint used to take; it
    does not add one. Pinned so a future refactor cannot quietly reintroduce a
    second full-corpus parse (~+6s, the cost the shared scan exists to remove)."""
    _RESIDUE_SLACK = 64  # helper-file parses evict the one-entry cache; see conftest
    assert corpus_scan.parse_misses <= corpus_scan.n_rglob_files + _RESIDUE_SLACK, (
        f"{corpus_scan.parse_misses} real parses for {corpus_scan.n_rglob_files} "
        f"files -- the escape check is parsing on top of the shared scan, not into it")
