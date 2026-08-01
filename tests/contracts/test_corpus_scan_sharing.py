"""Contract tests for the shared corpus scan in `tests/contracts/conftest.py`.

WHAT THIS GUARDS. Six corpus-wide lint contracts used to enumerate and parse
`experiments/` independently, so the corpus was parsed once per lint per session.
`conftest.scan_corpus()` now walks it once and applies every lint to each file as
it is parsed, sharing the parse via a one-entry cache installed onto
`validate_experiments.ast` / `validate_queue.ast`.

That is a performance change that must be BEHAVIOURALLY INVISIBLE. The primary
evidence is the five exact-count pins themselves (150 / 63 / 12 / 0 / 0) plus the
pre-registration hit list, which are full-corpus assertions and would move if the
sharing altered what is scanned or what a lint sees. The tests here are the
second line: they check the cache's own semantics directly, prove the sharing
actually happened (rather than silently degrading to a re-parse per lint, which
would be green but pointless), and differentially re-run a bounded sample of
files UNCACHED to confirm identical verdicts.

The one soundness precondition is that no consumer mutates the AST it is handed.
That was verified over both validators when this landed -- no NodeTransformer, no
fix_missing_locations, no parent-pointer annotation, no node mutation; every
consumer is a read-only walk. If a lint ever starts rewriting its tree, the
sample check below is what should catch it.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402
import validate_queue as VQ  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Must mirror `conftest.scan_corpus()`'s `path_lints`. `test_every_path_lint_is_covered`
# below fails if a lint is added there and not here, so the differential in (3) cannot
# silently stop covering a lint.
_PATH_LINTS = (
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
    ("dacc_last_bundle_lint", V.dacc_last_bundle_lint),
    ("agent_construction_before_seed_lint", V.agent_construction_before_seed_lint),
)


# ---- (1) the cache's own semantics ---------------------------------------------------
def test_cache_returns_the_same_tree_for_repeated_source(last_parse_cache_cls):
    """The point of the cache: the second consumer of a file gets the first's parse."""
    cache = last_parse_cache_cls(ast)
    src = "x = 1\ndef f():\n    return x\n"
    first = cache.parse(src, filename="a.py")
    second = cache.parse(src, filename="b.py")  # different filename, same text
    assert first is second, "repeated parse of identical source must be shared"
    assert (cache.misses, cache.hits) == (1, 1)


def test_cache_reparses_when_source_changes(last_parse_cache_cls):
    cache = last_parse_cache_cls(ast)
    a = cache.parse("a = 1\n")
    b = cache.parse("b = 2\n")
    c = cache.parse("a = 1\n")  # evicted -- only ONE entry is held, by design
    assert a is not b
    assert c is not a, "cache must hold exactly one entry (memory bound)"
    assert cache.misses == 3 and cache.hits == 0


def test_cache_holds_only_one_entry(last_parse_cache_cls):
    """Peak memory is the reason this is a size-1 cache, not an lru_cache(None):
    holding all 1162 corpus trees was measured at 886 MiB on a 3.8 GB hub."""
    cache = last_parse_cache_cls(ast)
    cache.parse("x = 1\n")
    assert cache._key == "x = 1\n"
    cache.parse("y = 2\n")
    assert cache._key == "y = 2\n", "previous entry must be evicted, not retained"


def test_cache_replays_syntax_error_to_every_consumer(last_parse_cache_cls):
    """A broken corpus file must fail identically for the 2nd consumer as the 1st --
    otherwise a lint's `except SyntaxError: return None` would stop firing on it."""
    cache = last_parse_cache_cls(ast)
    bad = "def (:\n"
    with pytest.raises(SyntaxError):
        cache.parse(bad)
    with pytest.raises(SyntaxError):
        cache.parse(bad)
    assert cache.misses == 1 and cache.hits == 1


def test_shim_exposes_everything_else_as_the_real_ast(last_parse_cache_cls):
    """The lints call ast.walk/ast.Name/isinstance through the same module global,
    so the installed shim must be indistinguishable from `ast` for non-parse access."""
    cache = last_parse_cache_cls(ast)
    shim = cache.module
    assert shim.Name is ast.Name
    assert shim.walk is ast.walk
    assert shim.FunctionDef is ast.FunctionDef
    assert shim.parse is not ast.parse, "parse must be the cached one"
    tree = shim.parse("v = 1\n")
    assert any(isinstance(n, shim.Name) for n in shim.walk(tree))


def test_shim_is_a_real_module_not_a_getattr_proxy(last_parse_cache_cls):
    """REGRESSION GUARD -- this exact mistake was made and measured.

    A `__getattr__`-forwarding proxy is functionally correct and was the first
    implementation. It made the corpus tests 48% SLOWER (63.14s -> 93.56s, idle
    hub, back-to-back on the same base), because the lints resolve `ast.Name` /
    `ast.walk` / `ast.FunctionDef` through this global inside `ast.walk` loops
    over every node of every corpus file. On a module those are C-level dict
    lookups; through `__getattr__` every one becomes a Python call, which cost
    far more than the shared parse saves.

    So the installed object must be a genuine module with the attributes present
    in its own `__dict__` -- not resolved dynamically.
    """
    shim = last_parse_cache_cls(ast).module
    assert isinstance(shim, types.ModuleType), (
        "the installed ast stand-in must be a real module -- a __getattr__ proxy "
        "makes attribute lookup a Python call and regresses the suite")
    for name in ("Name", "walk", "FunctionDef", "AsyncFunctionDef", "Module"):
        assert name in shim.__dict__, (
            f"{name} must be a direct __dict__ entry on the shim, not forwarded")


def test_cache_passes_through_non_str_source(last_parse_cache_cls):
    """bytes / explicit mode arguments are not cacheable and must not be mangled."""
    cache = last_parse_cache_cls(ast)
    tree = cache.parse(b"z = 1\n")
    assert isinstance(tree, ast.Module)
    expr = cache.parse("1 + 1", "<t>", "eval")
    assert isinstance(expr, ast.Expression)
    assert cache.misses == 0 and cache.hits == 0, "pass-through must not populate the cache"


# ---- (2) the sharing actually happened ------------------------------------------------
def test_shared_scan_parses_each_file_once(corpus_scan):
    """Guards against a silent degradation to one parse per lint.

    Every file in the rglob set is parsed once for the pre-registration lint (a
    miss); each top-level `v3_exq_*.py` driver is then handed to every path lint,
    which must all HIT. Asserted as inequalities rather than exact numbers so an
    added or removed corpus lint does not make this brittle -- the property being
    pinned is 'the extra consumers reuse the parse', not a specific arithmetic.

    THE SMALL RESIDUE IS EXPECTED AND BOUNDED. A few lint discharge rules parse a
    LIBRARY file rather than the driver under test -- measured on the hub
    2026-07-27: `pack_writer.py`, `_metrics.py`, `_lib/arm_fingerprint.py`,
    `_lib/capability_eval.py`, `_lib/z_goal_stream.py`,
    `goal_stream_stages_sd054.py`, `scaffolded_sd054_onboarding.py`, once each.
    Each such parse transiently evicts the one-entry cache, so the next lint for
    the current driver re-parses it. Measured then, with FOUR path lints: 1252
    parses against an ideal 1237, and 4640 hits against an ideal 4648 -- a 0.25%
    residue on ~5900 would-be parses, not worth a second cache slot (which would
    reintroduce unbounded memory growth as the number of such helper files grows).

    THE RESIDUE IS NOT THE FAILURE. A real degradation -- a lint that stops
    reusing the shared parse and re-parses the DRIVER it was handed -- costs one
    extra parse per driver, i.e. `n_glob_files` (~1170), where the whole residue
    is a few dozen. The two are separated by more than an order of magnitude and
    the guard below only has to sit between them.

    WHY THE BUDGET IS A FRACTION OF THE CORPUS AND IS ATTRIBUTED PER LINT, AND NOT
    A FIXED `_RESIDUE_SLACK`. This used to be a single global allowance of 64
    parses over `n_rglob_files`, and it was on a trajectory to fail for a reason
    that is not a degradation. Measured 2026-07-29 at `dd3205f878`:
    `glob=1172 rglob=1251 hits=12950 misses=1296`, i.e. 45 of the 64 already
    consumed -- against 15 when the paragraph above was written with FOUR path
    lints, there now being TEN. `dd3205f878` alone (`dead_z_goal_stream_lint`
    learning to resolve a local config-builder helper) took distinct helper-module
    parses from 7 to 21 and accounted for +14 of that. Two more helper-parsing
    rules would have breached 64 and failed this test for doing exactly what the
    residue paragraph says is fine.

    The asymptotics are what settle it, so state them rather than re-tuning a
    number. Per lint, the residue is O(distinct helper modules it resolves) --
    those resolvers hold stat-keyed caches, so a helper is parsed at most once per
    session no matter how many drivers reference it -- and is therefore CONSTANT in
    corpus size. The failure is O(`n_glob_files`). A fixed constant separates the
    two with a margin that SHRINKS every time a lint is added; a budget
    proportional to the corpus separates them with a margin that GROWS. Hence:

      * per lint, real parses charged to that lint must stay under one in
        `_PER_LINT_MISS_DIVISOR` of the drivers it was handed. This is the direct
        statement of the property -- 'this lint reused the parse it was given' --
        and it names the offending lint instead of reporting one opaque total.
      * globally, the residue outside the path lints (the invalid-escape and
        pre-registration pass over the non-driver rglob files) is held under
        `_GLOBAL_RESIDUE_DIVISOR`-th of one extra parse per driver.

    NOT A TAUTOLOGY, AND THE RESIDUE IS FULLY ACCOUNTED FOR. Re-measured on an idle
    `ree-worker-4`, 2026-07-29 at `a21f487709` (`glob=1173 rglob=1252 hits=12961
    misses=1297` -- the same 45 of residue, one driver later). The per-lint charges
    are not a diffuse tax; they are four numbers with named causes, and they sum to
    exactly the 45:

        dead_z_goal_stream_lint             21  its own distinct local helper
                                                modules (`_LOCAL_MODULE_CACHE`, via
                                                `_local_module_knob_setters` and
                                                `_uses_a_z_goal_driving_helper`)
        spearman_guard_shape_lint           18  driver re-parses inherited from
                                                those 21 evictions -- it runs
                                                immediately after dead_z
        config_slice_under_declaration_lint  3  distinct baseline modules
                                                (`_BASELINE_MODULE_CACHE`, whose own
                                                comment reads "3 in the corpus today")
        inert_salience_dacc_bias_lint        3  inherited the same way, running
                                                immediately after config_slice
        the other six                        0

    So the worst lint sits at 21 against a budget of 146 -- 7x headroom -- while a
    lint that stopped sharing would land 1173, 8x the budget and 56x the current
    worst. The global budget is 293 against 45 used. Both bounds move with the
    corpus; neither moves with the lint count.

    ONE PROPERTY OF THE ATTRIBUTION TO KNOW BEFORE READING A FAILURE. The helper
    caches are module-global and shared across lints, so the FIRST lint to touch a
    given helper pays for it and the rest ride free, and an eviction is charged to
    the NEXT lint rather than to the one that caused it -- which is why the two
    largest figures above are adjacent pairs. Reordering `path_lints` would shuffle
    these numbers without changing any behaviour, or the total. The per-lint figure
    is an attribution of COST, not of misbehaviour; what it pins is that no single
    lint's cost scales with the corpus.
    """
    assert corpus_scan.n_glob_files > 500, "corpus unexpectedly small -- check the walk"
    assert corpus_scan.n_rglob_files >= corpus_scan.n_glob_files
    # N path-lints per driver, each of which must reuse the pre-registration parse.
    # Ideal is len(_PATH_LINTS) hits per driver; one lint's worth of slack absorbs the
    # helper-file evictions described above. Scale this with _PATH_LINTS rather than
    # leaving it constant -- a fixed floor silently stops guarding the lints added since.
    assert corpus_scan.parse_hits >= (len(_PATH_LINTS) - 1) * corpus_scan.n_glob_files, (
        f"parse sharing degraded: {corpus_scan.parse_hits} hits for "
        f"{corpus_scan.n_glob_files} drivers -- the lints are re-parsing")

    # The direct property: no lint re-parses the drivers it is handed. Indexing
    # `lint_parse_misses` by `_PATH_LINTS` also KeyErrors if the scan's lint set and
    # this file's mirror drift apart, so a lint cannot be added to one and silently
    # left unguarded here.
    _PER_LINT_MISS_DIVISOR = 8
    per_lint_budget = corpus_scan.n_glob_files // _PER_LINT_MISS_DIVISOR
    for name, _ in _PATH_LINTS:
        charged = corpus_scan.lint_parse_misses[name]
        assert charged <= per_lint_budget, (
            f"{name} triggered {charged} real parses across "
            f"{corpus_scan.n_glob_files} drivers, over the budget of "
            f"{per_lint_budget} (1 in {_PER_LINT_MISS_DIVISOR} drivers). Expected is "
            f"a couple of dozen at most -- one per distinct HELPER module it "
            f"resolves. A figure near {corpus_scan.n_glob_files} means the lint is "
            f"re-parsing the driver it was handed instead of reusing the shared "
            f"parse; a figure that merely drifted up means it now resolves a helper "
            f"PER DRIVER rather than once per module, which is a real doubling of "
            f"corpus parse work and wants a stat-keyed cache in the resolver.")

    # One miss per file walked, not one per (file, lint) pair. Residue budgeted as a
    # fraction of one extra parse per driver so it tracks the corpus rather than the
    # lint count -- see the docstring for why the old fixed slack could not.
    _GLOBAL_RESIDUE_DIVISOR = 4
    residue_budget = corpus_scan.n_glob_files // _GLOBAL_RESIDUE_DIVISOR
    assert corpus_scan.parse_misses <= corpus_scan.n_rglob_files + residue_budget, (
        f"more parses ({corpus_scan.parse_misses}) than files walked "
        f"({corpus_scan.n_rglob_files}) + residue budget ({residue_budget}) -- the "
        f"cache is not being hit. Per-lint charges: "
        f"{ {n: corpus_scan.lint_parse_misses[n] for n, _ in _PATH_LINTS} }")


def test_every_path_lint_is_covered_by_this_files_differential(corpus_scan):
    """`_PATH_LINTS` here must mirror `conftest.scan_corpus()`'s `path_lints`.

    Without this, adding a lint to the shared scan and forgetting to add it here
    would leave the differential in (3) -- the only check that a shared verdict
    equals an uncached one -- silently not covering it. The failure mode is the
    quiet kind: everything stays green while the new lint's transparency is simply
    never tested. Compared against the scan's own key set rather than by importing
    the tuple, because `path_lints` is a local of `scan_corpus()` and because
    `tests/conftest.py` and `tests/contracts/conftest.py` both import under the
    bare name `conftest` (see the `last_parse_cache_cls` fixture for that hazard).
    """
    scanned = set(corpus_scan.fires) - {"prereg_share_feasibility_lint"}
    covered = {name for name, _ in _PATH_LINTS}
    assert scanned == covered, (
        f"_PATH_LINTS is out of sync with conftest.scan_corpus(): "
        f"missing here {sorted(scanned - covered)}, "
        f"stale here {sorted(covered - scanned)}")


def test_scan_restores_the_real_ast_module(corpus_scan):
    """The proxy is installed only for the duration of the walk. If it leaked, every
    later test in the session would run against a stale one-entry cache."""
    assert V.ast is ast
    assert VQ.ast is ast


# ---- (3) differential: cached verdicts == uncached verdicts ---------------------------
def _sample_paths(corpus_scan, per_lint=6, stride=140):
    """A bounded, deterministic sample: files that FIRE each lint (the load-bearing
    ones -- a comparison over only-clean files would prove None == None), plus a
    fixed-stride spread of the rest to catch a false NEGATIVE."""
    picked = []
    for name, _ in _PATH_LINTS:
        picked.extend(corpus_scan[name][:per_lint])
    drivers = sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))
    picked.extend(drivers[::stride])
    seen, out = set(), []
    for p in picked:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def test_corpus_scan_is_transparent(corpus_scan):
    """Re-run every lint UNCACHED over a bounded sample and require identical verdicts.

    Deliberately a sample, not the whole corpus: re-running every lint over all
    ~1160 drivers is exactly the duplicated work this change removes, so a
    full-corpus differential would cost more than the optimisation saves. Full-corpus
    coverage is retained by the exact-count pins themselves -- one per lint in
    `_PATH_LINTS`, each living in that lint's own test file -- which are asserted over
    every file and would move under any behavioural drift. (Stated without the LITERAL
    LIST for the same reason the zero-pin count below is: it read "150 / 63 / 12 / 0 / 0"
    when there were five such lints and was never updated as the set grew to nine, so it
    understated the coverage it was offered as evidence of. The two remaining spellings of
    that list -- this file's module docstring and conftest's -- are left alone on purpose:
    both are anchored to "the five lints" as records of the original sharing change, and
    rewriting a measurement record to match today's set would falsify it.)

    ONE ASYMMETRY WORTH KNOWING, for the lints pinned at ZERO (deliberately stated
    without a COUNT -- an earlier version said "the two", which was stale as soon as
    a third zero-pinned lint landed, and is the kind of drift nothing checks).
    `_sample_paths` seeds itself from each lint's fire list, so a lint with no corpus
    fires contributes no positive witness and is compared only over the stride
    spread, i.e. only in the false-NEGATIVE direction (None == None). A zero pin
    likewise cannot detect a lint that has been silenced. What covers those is that
    each such lint has its own synthetic positive cases in its own test file, run
    UNCACHED (`test_hdr_fires_on_the_reachable_hardcoded_shape` and the spearman
    equivalent), so 'the lint still fires at all' is pinned independently of this
    scan. When a zero-pinned lint is first wired into `scan_corpus`, run a one-off
    full-corpus differential to confirm the fire SETS match -- done for
    `hardcoded_dry_run_lint` when it was folded in, and for `write_pack_dry_run_lint`
    (2026-07-28: both sides empty over all 1167 drivers, since ZERO of them call
    `write_pack` at all -- see that file's pin comment for why empty is not vacuous).
    """
    assert V.ast is ast, "must run uncached -- the proxy should not be installed here"
    sample = _sample_paths(corpus_scan)
    assert len(sample) >= 12, f"sample too small to be meaningful: {len(sample)}"

    for name, fn in _PATH_LINTS:
        fired = set(corpus_scan[name])
        for p in sample:
            uncached = fn(p) is not None
            assert uncached == (p in fired), (
                f"{name} disagrees on {p.name}: shared scan said "
                f"{'FIRE' if p in fired else 'clean'}, uncached said "
                f"{'FIRE' if uncached else 'clean'}")

    prereg_hits = set(corpus_scan["prereg_share_feasibility_lint"])
    for p in sample:
        src = p.read_text(encoding="utf-8", errors="ignore")
        uncached = bool(VQ.prereg_share_feasibility_lint(src))
        assert uncached == (p.name in prereg_hits), (
            f"prereg_share_feasibility_lint disagrees on {p.name}")
