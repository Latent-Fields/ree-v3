"""Shared, session-scoped scan of `experiments/` for the corpus-wide lint contracts.

WHAT THIS CHANGES. Six contract tests each assert a property over the WHOLE
`experiments/` corpus (~1160 `v3_exq_*.py` drivers, ~1240 `*.py` under the tree).
Each one used to enumerate the corpus itself and call its lint file-by-file, and
every one of those lints opens with the same prelude:

    src  = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

so the corpus was read and **parsed once per corpus lint per pytest session**.
This module makes that happen ONCE and hands every lint the same parse.

NOTHING IS SAMPLED, CAPPED OR SKIPPED. Each lint is still applied to exactly the
file set it applied to before -- the five `validate_experiments` lints to
`sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))`, the `validate_queue` pre-registration
lint to `sorted(EXPERIMENTS_DIR.rglob("*.py"))` -- in the same sorted order, with
the same per-file error handling. The lint functions themselves are untouched and
are called with the same arguments as before. This is "do the same work once
instead of once per lint", not a coverage reduction: the exact-count pins
(150 / 63 / 12 / 0 / 0 and the pre-registration hit list) are the evidence for
that, and they are unchanged. If a change here altered what is scanned, those
five counts would move.

WHY THE LOOP IS INVERTED (every lint inside one file walk, not one file walk per
lint). The obvious shape -- a fixture that returns `{path: tree}` for the whole
corpus -- was measured and REJECTED: holding all 1162 trees resident costs
**886 MiB** (measured on the hub, 2026-07-27). The hub is a 3.8 GB box that also
runs the coordinator, `sync_daemon` and a `serve.py` holding ~1.7 GB RSS, and had
916 MiB available when this was written -- so a session-scoped fixture of that
shape would have traded ~16s of CPU for an OOM risk across the whole run, and the
resulting swapping would likely have cost more time than the parse ever did.
Walking the corpus once and
applying every lint to each file as it is parsed keeps only ONE tree alive at a
time (`_LastParseCache`, below, holds exactly one), so peak memory is unchanged
from the old behaviour while the parse is still paid once.

WHY THE PARSE CACHE RATHER THAN A `tree=` PARAMETER ON EACH LINT. Threading a
pre-parsed tree through would mean changing every production lint signature in
`validate_experiments.py` / `validate_queue.py`. Caching the parse instead keeps
the production validators bit-identical, which matters for three reasons:
  (1) the gate's behaviour when run for real (`validate_experiments.py` on the
      command line, and the pre-commit contract hook) is provably unchanged;
  (2) a signature change has to be opted into by every FUTURE corpus lint, where
      the cache covers new ones for free -- there was a sixth corpus-scanning
      lint in flight in another session while this was written; and
  (3) `validate_experiments.py` is a high-contention file.

THAT SIXTH LINT LANDED, AND (2) IS WHY FOLDING IT IN WAS A THREE-LINE CHANGE.
`hardcoded_dry_run_lint` (ree-v3 `5a1e645`, backlog drawn to zero by `506c713`)
arrived from session `festive-bose-705787` with its own
`for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))` loop -- i.e. it re-added
exactly the sixth full-corpus walk+parse this module exists to remove. Wiring it
in needed only a `path_lints` entry and a fixture argument on its pinned test,
because the cache required no signature change of it. Its production function is
byte-identical before and after, and its pin is still 0. THIS IS THE STANDING
PATTERN: a new corpus-wide lint belongs in `path_lints` below, and its corpus
test takes `corpus_scan` -- it must not enumerate `experiments/` itself.

ONE CHECK RIDES THE PARSE ITSELF RATHER THAN `path_lints`: INVALID ESCAPE
SEQUENCES. `scan.invalid_escapes` pins `'\\|'`-style invalid escapes at zero
corpus-wide (`test_invalid_escape_sequence_lint.py`). It is deliberately NOT a
`path_lints` entry, and the reason is a trap worth stating outright: an invalid
escape is reported by the **tokenizer, as a warning raised during `ast.parse`**,
not by anything visible in the resulting tree. A `path -> Optional[str]` lint in
`path_lints` runs AFTER the pre-registration lint has already parsed that file,
so its `ast.parse` is a cache HIT -- no real parse, therefore no warning,
therefore a lint that silently never fires while looking perfectly green. So the
capture has to happen at the MISS, which is what `invalid_escape_findings` below
does: it is the FIRST consumer of each file's source, takes the miss under a
`warnings.catch_warnings` block, and leaves the parse in the cache for the
pre-registration lint and the path lints to hit as before. Total real parses per
file are unchanged (one); only which consumer pays for it moves.

SAFETY OF SHARING ONE TREE ACROSS LINTS. Sound only because no consumer mutates
the AST. Verified over both validators: no `NodeTransformer`, no
`fix_missing_locations`, no parent-pointer annotation, no `.body.append/insert/pop`,
no `setattr` on a node -- every consumer is a read-only `ast.walk` / `isinstance`
traversal. If a lint ever starts rewriting the tree it is handed, this cache
becomes unsound and must be given a per-consumer copy. `test_corpus_scan_is_transparent`
in `test_corpus_scan_sharing.py` is the standing check that the shared scan and an
independent uncached scan agree.

MEASUREMENT (2026-07-27; corpus 1162 top-level `v3_exq_*.py`, 1237 under `rglob`).
The shape of the cost: `read_text` over the corpus is only ~0.35s while `ast.parse`
is the whole prelude, and three of the five lints are almost entirely prelude --
`dead_z_goal_stream_lint` early-returns on its knob gate for nearly every file, so
its walk is close to free, and `e3_diagnostics_staleness_lint` /
`spearman_guard_shape_lint` are ~78% parse. Only `e3_hold_weighted_readout_lint` is
walk-dominated. So four of the five parses were pure waste.

The A/B, on an IDLE `ree-worker-2`, pristine vs this change, back-to-back on the
same base (`c7ef00c`), running exactly the five corpus tests:

    pair 1:  66.91s -> 50.34s   (-16.57s, -24.8%)
    pair 2:  64.99s -> 49.80s   (-15.19s, -23.4%)

i.e. **~-16s, ~-24%**, reproducible. Per-test before: hw 26.6-28.2s, spearman
10.6-11.1s, dzg 9.7-10.4s, e3s 9.6-10.1s, prereg 6.6-7.0s.

SECOND MEASUREMENT -- FOLDING IN THE SIXTH LINT (2026-07-28). The pair above is the
original five-lint share (`60a6131`). Folding `hardcoded_dry_run_lint` into
`path_lints` (`7328f1e`, the three-line change described above) was A/B'd
separately, on an IDLE, UNCONTENDED `ree-worker-4`, two pairs back-to-back, running
exactly the SIX corpus tests -- the five above plus `test_hardcoded_dry_run_lint.py`:

    pair 1:  68.69s -> 63.32s   (-5.37s, -7.8%)
    pair 2:  66.37s -> 63.26s   (-3.11s, -4.7%)

132 tests passed in BOTH halves of both pairs, so coverage is identical -- which is
the whole point: the fold-in deletes a sixth full-corpus walk+parse, not an
assertion. Pre-half = pristine `37673f2`; post-half = `37673f2` plus ONLY `7328f1e`'s
change to the three `tests/contracts` files, with one line removed that `7328f1e`
had swept in from another session (`config_slice_under_declaration_lint`, since
landed on its own as `25bf07e`), so the delta isolates the `hardcoded_dry_run`
fold-in alone. Both halves ran from throwaway DETACHED git worktrees, so neither
carried any other session's uncommitted work.

READ THAT PAIR AS A DELTA ONLY -- ITS ABSOLUTE SECONDS ARE NOT COMPARABLE TO THE
BLOCK ABOVE. `ree-worker-4` is a DIFFERENT BOX from the `ree-worker-2` the
66.91 -> 50.34s pair was taken on: the hub and `ree-worker-2` were both held by other
sessions' pytest runs at the time, which is why the surge worker was used. So
63.26s here is NOT "slower than" 49.80s above -- different machine, different day,
and a six-test set rather than a five-test set. Only the within-pair ratio carries
across. That is the same rule the next paragraph states, applied to these two blocks
with respect to each other. (Originally recorded in `WORKSPACE_STATE.md`
2026-07-28T06:17Z, session `elated-shamir-5ef201`; the docstring edit was deferred
because a third session had ~125 uncommitted lines in this file at the time -- the
CLAUDE.md read-modify-write hazard.)

TAKE THE MEASUREMENT PROTOCOL SERIOUSLY IF YOU REVISIT THIS. Three earlier
attempts at this number were wrong, all in the optimistic-looking direction until
checked: two were polluted by another session's full-suite run on the shared hub,
and one by an experiment (`v3_exq_833`) starting on the hub ~40s into the run.
`serve.py` alone holds ~1.7 GB RSS on that 3.8 GB box, so memory pressure moves
these numbers too. Verify the box is idle (`uptime`, no `pytest`, no experiment
process) IMMEDIATELY before and after each half of the pair, and run the halves
back-to-back on the same machine -- never compare against a figure from a
different box or a different day.
"""

from __future__ import annotations

import ast
import sys
import types
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
for _p in (str(REPO_ROOT), str(REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


class _LastParseCache:
    """Memoises the MOST RECENT `ast.parse`, and exposes it as a drop-in `ast` module.

    NEVER MAKE THIS A `__getattr__` PROXY. The obvious implementation -- an object
    that intercepts `parse` and forwards everything else via `__getattr__` -- was
    written, measured, and REVERTED: it made the corpus tests **48% slower**
    (63.14s -> 93.56s, idle hub, back-to-back on the same base), turning the whole
    optimisation into a regression. The lints resolve `ast.Name` / `ast.walk` /
    `ast.FunctionDef` through this same module global inside `ast.walk` loops over
    every node of every file, i.e. millions of attribute lookups per run. On a real
    module those are C-level `__dict__` hits; through `__getattr__` each becomes a
    Python-level call, and that overhead dwarfed the ~25s the shared parse saves.

    So `self.module` is a genuine `types.ModuleType` with `ast.__dict__` copied in
    and only `parse` rebound. Attribute access stays on the fast path and just
    `parse` is intercepted. That is what gets installed onto
    `validate_experiments.ast` / `validate_queue.ast`.

    Installed onto `validate_experiments.ast` / `validate_queue.ast` for the
    duration of the shared scan, so that the several lints applied to one file
    in immediate succession parse that file's source once between them.

    Deliberately holds exactly ONE entry. That is what keeps peak memory at one
    tree instead of the 886 MiB a whole-corpus cache measured at, and it is
    sufficient *because the scan is file-major*: every consumer of a given file
    runs before the walk moves on.

    Keyed on the source TEXT, not on (text, filename). The four
    `validate_experiments` lints call `ast.parse(src, filename=str(path))` while
    `validate_queue.prereg_share_feasibility_lint` calls `ast.parse(source)` with
    no filename, so keying on the pair would split the very entry we are trying
    to share. `filename` affects nothing but the message of a raised SyntaxError,
    and every one of these consumers catches SyntaxError and discards it
    (returning None / [] -- unreadable and unparseable scripts are reported by
    `check_script`, not by these lints), so the distinction is unobservable here.

    A raised exception is cached and re-raised alongside the success case, so a
    syntactically broken corpus file behaves identically for the second consumer
    as for the first.
    """

    def __init__(self, real_ast) -> None:
        self._real = real_ast
        self._key: Optional[str] = None
        self._tree = None
        self._exc: Optional[BaseException] = None
        self.hits = 0
        self.misses = 0
        # A real module, so `shim.Name` etc. stay C-level dict lookups. See the
        # class docstring for why a __getattr__ proxy is not an option here.
        shim = types.ModuleType(getattr(real_ast, "__name__", "ast"))
        shim.__dict__.update(real_ast.__dict__)
        shim.parse = self.parse
        self.module = shim

    def parse(self, source, filename="<unknown>", *args, **kwargs):
        # Only the plain str form is cacheable; anything exotic (bytes, an
        # explicit mode/flags) goes straight through uncached.
        if args or kwargs or not isinstance(source, str):
            return self._real.parse(source, filename, *args, **kwargs)

        if self._key is None or source != self._key:
            self.misses += 1
            self._key = source
            self._tree, self._exc = None, None
            try:
                self._tree = self._real.parse(source, filename)
            except (SyntaxError, ValueError) as exc:
                self._exc = exc
        else:
            self.hits += 1

        if self._exc is not None:
            raise self._exc
        return self._tree


# The compiler's own wording, stable across the categories it has been raised
# under. Matched on the MESSAGE rather than the category on purpose: CPython
# raised this as `DeprecationWarning` up to 3.11 and as `SyntaxWarning` from
# 3.12, and it is slated to become a `SyntaxError`. Keying on the category would
# make the contract quietly stop firing on an interpreter upgrade -- the exact
# failure mode this file exists to prevent.
INVALID_ESCAPE_MARKER = "invalid escape sequence"


def invalid_escape_findings(
    source: str,
    path: Path,
    parse: Optional[Callable] = None,
) -> List[str]:
    """Return one message per invalid escape sequence in `source` ([] == clean).

    An invalid escape (`'\\|'`, `'\\d'` outside a raw string) is not visible in the
    parsed tree -- the tokenizer reports it as a warning while parsing. So this
    parses under `catch_warnings(record=True)` + `simplefilter("always")` and
    keeps the warnings whose message carries `INVALID_ESCAPE_MARKER`.

    `simplefilter("always")` is load-bearing, but NOT for the reason first
    documented here. It is not about `__warningregistry__` dedup: the tokenizer
    passes no registry for this warning, so repeated parses of the SAME file warn
    every time even under the `default` filter (measured). The real hazard is an
    ambient filter that suppresses the category -- `-W ignore`, or a pytest
    `filterwarnings` ini entry. Measured, for a file that HAS a bad escape:

        ambient=ignore   without always -> 0 findings   with always -> 1
        ambient=error    without always -> 0 findings   with always -> 1

    Both are silent blindness, and `error` is the nastier one: CPython converts a
    warning-turned-error from the tokenizer into a **SyntaxError**, which the
    fail-soft `except` below then swallows. Either way the corpus pin would sit at
    zero while offenders went unreported. `simplefilter("always")` inside the
    `catch_warnings` block makes the detector independent of ambient
    configuration. Pinned by `test_detector_is_immune_to_ambient_warning_filters`,
    which FAILS if the `simplefilter` call is removed.

    `parse` defaults to the real `ast.parse`; the shared scan passes the one-entry
    cache's `parse` so the corpus is still parsed exactly once per file. The
    negative control passes the real one, which is what makes it a control -- it
    exercises this function through the same path the corpus does.

    Findings are collected even when the parse then RAISES. A file can carry a
    warned-about escape on one line and a genuine SyntaxError on another; the
    escape is real and still worth naming (an unparseable script is separately
    reported by `check_script`, so nothing is lost by being fail-soft here).

    DE-DUPLICATED, and that is not cosmetic. When the parse fails, CPython's PEG
    parser makes a SECOND tokenizing pass to build a better error message, and the
    tokenizer re-emits every escape warning it already emitted -- so an
    escape-plus-syntax-error file reports each escape TWICE. Observed on the fleet's
    linux/3.10 (2 findings) and NOT on darwin/3.13 (1), i.e. it is version-dependent
    and would make any exact count machine-dependent. Order is preserved so the
    first-reported line stays first. Two identical escapes on ONE line collapse to a
    single finding; that is intended -- the message points at a line, and the pin is
    at zero, so a collapsed duplicate cannot hide anything.
    """
    if parse is None:
        parse = ast.parse
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            parse(source, str(path))
        except (SyntaxError, ValueError):
            pass

    out: List[str] = []
    for w in caught:
        if INVALID_ESCAPE_MARKER not in str(w.message):
            continue
        finding = f"{path.name}:{w.lineno}: {w.message}"
        if finding not in out:
            out.append(finding)
    return out


class CorpusScan:
    """Result of the single shared walk: per-lint fire lists, in sorted order."""

    __slots__ = (
        "fires", "n_glob_files", "n_rglob_files", "parse_hits", "parse_misses",
        "invalid_escapes", "lint_parse_misses",
    )

    def __init__(self) -> None:
        self.fires: Dict[str, List] = {}
        self.n_glob_files = 0
        self.n_rglob_files = 0
        self.parse_hits = 0
        self.parse_misses = 0
        # Invalid escape sequences over the whole rglob set. Not in `fires`: it is
        # not a `path_lints` entry (see the module docstring for why it cannot be),
        # and `test_every_path_lint_is_covered_by_this_files_differential` treats
        # every `fires` key as one.
        self.invalid_escapes: List[str] = []
        # Real parses (cache MISSES) incurred INSIDE each path lint's own call,
        # summed over every driver it was handed. This is the attribution that
        # makes the sharing guard direct rather than a global budget: a healthy
        # lint's figure is O(number of helper modules it resolves) and does NOT
        # grow with the corpus, while the failure being guarded -- a lint that
        # re-parses the driver it was handed -- is exactly `n_glob_files`. See
        # `test_shared_scan_parses_each_file_once`.
        self.lint_parse_misses: Dict[str, int] = {}

    def __getitem__(self, lint_name: str):
        return self.fires[lint_name]


def scan_corpus() -> CorpusScan:
    """Walk `experiments/` once and apply every corpus-wide lint to each file.

    Ordering and file sets are chosen to reproduce, exactly, what the six tests
    did independently:

      * `sorted(EXPERIMENTS_DIR.rglob("*.py"))` drives the outer loop -- this is
        the pre-registration lint's own file set, and it is a SUPERSET of the
        others' (`glob("v3_exq_*.py")` is the top level of the same tree),
        so one walk covers both and each lint still sees its own set.
      * the `validate_experiments` lints are applied only to top-level
        `v3_exq_*.py`, i.e. exactly `EXPERIMENTS_DIR.glob("v3_exq_*.py")`.
        `is_driver` below is that glob spelled as a predicate, so a lint moved
        into `path_lints` keeps its own file set unchanged.
      * `rglob`/`glob` results are sorted, so each fire list comes out in the
        same order the old per-test `sorted(...)` comprehensions produced.
    """
    import validate_experiments as V
    import validate_queue as VQ

    path_lints = (
        ("e3_hold_weighted_readout_lint", V.e3_hold_weighted_readout_lint),
        ("e3_diagnostics_staleness_lint", V.e3_diagnostics_staleness_lint),
        ("dead_z_goal_stream_lint", V.dead_z_goal_stream_lint),
        ("spearman_guard_shape_lint", V.spearman_guard_shape_lint),
        ("hardcoded_dry_run_lint", V.hardcoded_dry_run_lint),
        ("emit_outcome_dry_run_lint", V.emit_outcome_dry_run_lint),
        ("write_pack_dry_run_lint", V.write_pack_dry_run_lint),
        ("dry_run_unreachable_criterion_lint", V.dry_run_unreachable_criterion_lint),
        ("config_slice_under_declaration_lint", V.config_slice_under_declaration_lint),
        ("inert_salience_dacc_bias_lint", V.inert_salience_dacc_bias_lint),
    )

    scan = CorpusScan()
    for name, _ in path_lints:
        scan.fires[name] = []
        scan.lint_parse_misses[name] = 0
    scan.fires["prereg_share_feasibility_lint"] = []

    cache = _LastParseCache(ast)
    v_ast_saved, vq_ast_saved = V.ast, VQ.ast
    V.ast, VQ.ast = cache.module, cache.module
    try:
        for p in sorted(EXPERIMENTS_DIR.rglob("*.py")):
            scan.n_rglob_files += 1
            is_driver = p.parent == EXPERIMENTS_DIR and p.name.startswith("v3_exq_") \
                and p.suffix == ".py"

            # (1) pre-registration feasibility -- takes SOURCE, over the rglob set.
            #     Reading first also primes the parse cache for the four below.
            try:
                src = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                src = None  # same `continue` semantics as the old test

            # (1a) invalid escape sequences -- must run BEFORE any other consumer
            #      of this file, because only the cache MISS does a real parse and
            #      only a real parse raises the tokenizer's warning. See the module
            #      docstring. This takes the miss that the pre-registration lint
            #      used to take; the parse count per file is unchanged.
            if src is not None:
                misses_before = cache.misses
                found = invalid_escape_findings(src, p, cache.parse)
                if cache.misses == misses_before:
                    # Cache HIT, i.e. the previously-parsed file had byte-identical
                    # source (the cache keys on TEXT, not on the path). No real
                    # parse happened, so no warning could fire and `found` is a
                    # false clean. Re-parse this one uncached rather than let a
                    # duplicated file smuggle an escape past the pin. Costs an
                    # extra parse only for genuine duplicates, of which the corpus
                    # currently has none.
                    found = invalid_escape_findings(src, p, ast.parse)
                scan.invalid_escapes.extend(found)

            if src is not None and VQ.prereg_share_feasibility_lint(src):
                scan.fires["prereg_share_feasibility_lint"].append(p.name)

            # (2) the four path-taking lints -- over the top-level v3_exq_* set only.
            if not is_driver:
                continue
            scan.n_glob_files += 1
            for name, fn in path_lints:
                # Charge every real parse this lint triggers to the lint itself.
                # Two things land in this bucket, and the distinction matters when
                # reading a failure:
                #   * the lint parsing a HELPER module (an import target it has to
                #     resolve to discharge its rule). Bounded per lint by the number
                #     of distinct helper modules, because those resolvers hold their
                #     own stat-keyed caches -- constant in corpus size.
                #   * the lint re-parsing THIS driver because the previous lint's
                #     helper parse evicted the one-entry cache. Charged here, to the
                #     lint that pays it, not to the one that caused the eviction.
                # A lint that stops sharing altogether lands `n_glob_files` here.
                misses_before = cache.misses
                fired = fn(p) is not None
                scan.lint_parse_misses[name] += cache.misses - misses_before
                if fired:
                    scan.fires[name].append(p)
    finally:
        V.ast, VQ.ast = v_ast_saved, vq_ast_saved

    scan.parse_hits, scan.parse_misses = cache.hits, cache.misses
    return scan


@pytest.fixture(scope="session")
def last_parse_cache_cls():
    """Hand `_LastParseCache` to its contract test via a fixture.

    Deliberately NOT `from conftest import _LastParseCache` on the test side:
    `tests/conftest.py` and `tests/contracts/conftest.py` would both import under
    the bare module name `conftest`, and which one a plain `import conftest`
    resolves to depends on collection order and sys.modules state. A fixture has
    no such ambiguity.
    """
    return _LastParseCache


@pytest.fixture(scope="session")
def invalid_escape_findings_fn():
    """Hand `invalid_escape_findings` to its contract test via a fixture.

    Same reason as `last_parse_cache_cls`: `tests/conftest.py` and
    `tests/contracts/conftest.py` both import under the bare module name
    `conftest`, so a plain `import conftest` on the test side resolves by
    collection order. A fixture is unambiguous.
    """
    return invalid_escape_findings


@pytest.fixture(scope="session")
def corpus_scan() -> CorpusScan:
    """The one shared corpus walk, computed on first use and reused thereafter.

    TRADE-OFF, stated so nobody is surprised by it: this computes ALL the
    corpus-wide lints in one pass, so running a SINGLE corpus test in isolation
    now pays for the others' walks too (dominated by
    `e3_hold_weighted_readout_lint`, the one genuinely walk-heavy lint). That is
    the deliberate direction: the full contract suite -- what the pre-commit gate
    and `scripts/remote_pytest.sh` actually run -- gets materially faster, at the
    cost of a slower single-test invocation. Per-lint laziness is not available
    without either re-parsing per lint (the thing being removed) or holding the
    whole corpus resident (the 886 MiB shape rejected above).
    """
    return scan_corpus()
