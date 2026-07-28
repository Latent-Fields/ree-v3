"""Shared, session-scoped scan of `experiments/` for the corpus-wide lint contracts.

WHAT THIS CHANGES. Five contract tests each assert a property over the WHOLE
`experiments/` corpus (~1160 `v3_exq_*.py` drivers, ~1240 `*.py` under the tree).
Each one used to enumerate the corpus itself and call its lint file-by-file, and
every one of those lints opens with the same prelude:

    src  = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

so the corpus was read and **parsed five times per pytest session**. This module
makes that happen ONCE and hands every lint the same parse.

NOTHING IS SAMPLED, CAPPED OR SKIPPED. Each lint is still applied to exactly the
file set it applied to before -- the four `validate_experiments` lints to
`sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))`, the `validate_queue` pre-registration
lint to `sorted(EXPERIMENTS_DIR.rglob("*.py"))` -- in the same sorted order, with
the same per-file error handling. The lint functions themselves are untouched and
are called with the same arguments as before. This is "do the same work once
instead of five times", not a coverage reduction: the exact-count pins
(150 / 63 / 12 / 0 and the pre-registration hit list) are the evidence for that,
and they are unchanged. If a change here altered what is scanned, those four
counts would move.

WHY THE LOOP IS INVERTED (five lints inside one file walk, not one file walk per
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
pre-parsed tree through would mean changing five production lint signatures in
`validate_experiments.py` / `validate_queue.py`. Caching the parse instead keeps
the production validators bit-identical, which matters for three reasons:
  (1) the gate's behaviour when run for real (`validate_experiments.py` on the
      command line, and the pre-commit contract hook) is provably unchanged;
  (2) a signature change has to be opted into by every FUTURE corpus lint, where
      the cache covers new ones for free -- there was a sixth corpus-scanning
      lint in flight in another session while this was written; and
  (3) `validate_experiments.py` is a high-contention file.

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
from pathlib import Path
from typing import Dict, List, Optional

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


class CorpusScan:
    """Result of the single shared walk: per-lint fire lists, in sorted order."""

    __slots__ = ("fires", "n_glob_files", "n_rglob_files", "parse_hits", "parse_misses")

    def __init__(self) -> None:
        self.fires: Dict[str, List] = {}
        self.n_glob_files = 0
        self.n_rglob_files = 0
        self.parse_hits = 0
        self.parse_misses = 0

    def __getitem__(self, lint_name: str):
        return self.fires[lint_name]


def scan_corpus() -> CorpusScan:
    """Walk `experiments/` once and apply every corpus-wide lint to each file.

    Ordering and file sets are chosen to reproduce, exactly, what the five tests
    did independently:

      * `sorted(EXPERIMENTS_DIR.rglob("*.py"))` drives the outer loop -- this is
        the pre-registration lint's own file set, and it is a SUPERSET of the
        other four's (`glob("v3_exq_*.py")` is the top level of the same tree),
        so one walk covers both and each lint still sees its own set.
      * the four `validate_experiments` lints are applied only to top-level
        `v3_exq_*.py`, i.e. exactly `EXPERIMENTS_DIR.glob("v3_exq_*.py")`.
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
    )

    scan = CorpusScan()
    for name, _ in path_lints:
        scan.fires[name] = []
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
            if src is not None and VQ.prereg_share_feasibility_lint(src):
                scan.fires["prereg_share_feasibility_lint"].append(p.name)

            # (2) the four path-taking lints -- over the top-level v3_exq_* set only.
            if not is_driver:
                continue
            scan.n_glob_files += 1
            for name, fn in path_lints:
                if fn(p) is not None:
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
