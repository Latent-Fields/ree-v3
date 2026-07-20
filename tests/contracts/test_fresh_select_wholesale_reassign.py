"""Contract: e3_selector.select() must REASSIGN last_score_diagnostics wholesale.

This pins the load-bearing substrate assumption behind the shared
`experiments/_lib/fresh_select.py` sentinel-key instrumentation, which every
E3 per-selection experiment depends on to avoid the hold-weighted-readout
defect class (corpus sweep `hold_weighted_e3_readout_corpus_sweep_2026-07-20`,
the 699 / 689d autopsies).

WHY THIS TEST EXISTS -- the failure it catches is SILENT and NOT self-catching.
The sentinel detects a genuine E3 selection by stamping a private key into
`agent.e3.last_score_diagnostics` before `select_action()` and observing that
the key is GONE afterwards, which is true only because `select()` rebinds the
attribute to a fresh dict literal. An in-place refactor splits into two
directions, and -- contrary to the intuition that in-place mutation is the safe
failure -- ONE OF THEM IS SILENT (verified by
`test_in_place_clear_is_the_silent_direction` below):

  * writing keys individually WITHOUT clearing leaves the sentinel in place, so
    every tick reads as LATCHED. Loud: the `n_fresh_select` sufficiency gates
    fail the run.
  * `self.last_score_diagnostics.clear()` + `.update(...)` DELETES the sentinel
    along with everything else, so every tick reads as FRESH. Silent: counts
    look healthy, the yield looks like ~1.0, and the replicated readout returns
    with nothing downstream objecting.

The same silent direction is reached by any wholesale rebind on a NON-selecting
code path. So this test asserts the exact structural property the sentinel's
correctness rests on, covering both directions:

  1. `select()` contains exactly ONE wholesale rebind of the attribute;
  2. that rebind's RHS is a dict LITERAL (not `.clear()`/`.update()`, not a
     reference to a dict built elsewhere);
  3. it sits at the DIRECT statement level of `select()` -- not nested inside
     any `if`/`try`/loop that could skip it;
  4. NO `return` occurs anywhere in `select()` before it, so it is
     unconditionally reached on every call;
  5. no OTHER method of the class rebinds the attribute wholesale (only
     `__init__` may), so a non-selecting code path cannot clear the marker.

Structural (AST) assertions are used deliberately rather than a behavioural
probe: constructing a live E3Selector requires substantial substrate setup, and
the property under test is a property of the SOURCE, so a source-level
assertion is both cheaper and strictly more direct.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E3_SELECTOR = REPO_ROOT / "ree_core" / "predictors" / "e3_selector.py"

ATTR = "last_score_diagnostics"


def _module() -> ast.Module:
    assert E3_SELECTOR.is_file(), f"missing substrate file: {E3_SELECTOR}"
    return ast.parse(E3_SELECTOR.read_text(encoding="utf-8"))


def _wholesale_rebinds(node: ast.AST) -> list:
    """Assignments of the form `self.<ATTR> = <value>` within `node`.

    Item-mutations (`self.<ATTR>[k] = v`) have an ast.Subscript target and are
    excluded -- they leave pre-existing keys, including the sentinel, intact.
    """
    out = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Assign):
            continue
        for tgt in n.targets:
            if (
                isinstance(tgt, ast.Attribute)
                and tgt.attr == ATTR
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
            ):
                out.append(n)
    return out


def _select_fn() -> ast.FunctionDef:
    fns = [
        n for n in ast.walk(_module())
        if isinstance(n, ast.FunctionDef) and n.name == "select"
    ]
    assert len(fns) == 1, (
        f"expected exactly one `select` definition in {E3_SELECTOR.name}, "
        f"found {len(fns)} at lines {[f.lineno for f in fns]}. The fresh-select "
        "sentinel targets a single select(); resolve the ambiguity before "
        "relying on it."
    )
    return fns[0]


def test_select_rebinds_diagnostics_exactly_once_as_a_dict_literal():
    """(1) + (2): one wholesale rebind, RHS a dict literal."""
    fn = _select_fn()
    rebinds = _wholesale_rebinds(fn)

    assert len(rebinds) == 1, (
        f"e3_selector.select() must rebind self.{ATTR} exactly ONCE; found "
        f"{len(rebinds)} at lines {[r.lineno for r in rebinds]}. The shared "
        "sentinel in experiments/_lib/fresh_select.py detects a genuine E3 "
        "selection by that single wholesale rebind clearing its marker key."
    )

    assign = rebinds[0]
    assert isinstance(assign.value, ast.Dict), (
        f"self.{ATTR} must be rebound to a DICT LITERAL at line "
        f"{assign.lineno}, got {type(assign.value).__name__}. An in-place "
        "mutation (.clear()/.update()) or an aliased dict would break the "
        "fresh-select sentinel."
    )


def test_diagnostics_rebind_is_unconditional_in_select():
    """(3) + (4): direct statement level, and no earlier return."""
    fn = _select_fn()
    assign = _wholesale_rebinds(fn)[0]

    direct = [
        s for s in fn.body
        if isinstance(s, ast.Assign) and s.lineno == assign.lineno
    ]
    assert direct, (
        f"the self.{ATTR} rebind at line {assign.lineno} must sit at the DIRECT "
        "statement level of select(), not nested inside an if/try/loop. Nested, "
        "it could be skipped on some path, and the sentinel would then report a "
        "genuine selection as latched."
    )

    earlier_returns = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Return) and n.lineno < assign.lineno
    ]
    assert not earlier_returns, (
        f"select() must not `return` before rebinding self.{ATTR} (line "
        f"{assign.lineno}); found returns at {earlier_returns}. An early return "
        "would leave the sentinel marker in place after a call that DID select, "
        "under-counting n_fresh_select."
    )


def test_no_other_method_rebinds_diagnostics_wholesale():
    """(5): only __init__ and select() may rebind the attribute wholesale.

    This is the direction the sufficiency gates CANNOT catch: a wholesale
    rebind on a non-selecting path clears the marker without a selection, so
    every tick reads as FRESH and the replicated readout returns silently.
    """
    mod = _module()
    offenders = []
    for fn in ast.walk(mod):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name in ("__init__", "select"):
            continue
        for a in _wholesale_rebinds(fn):
            # attribute rebinds nested inside select() are already covered
            offenders.append((fn.name, a.lineno))

    assert not offenders, (
        f"self.{ATTR} is rebound wholesale outside __init__/select() at "
        f"{offenders}. Any such rebind on a NON-selecting path clears the "
        "fresh-select sentinel without a selection having occurred, making "
        "every tick read as FRESH. That failure is silent -- the n_fresh_select "
        "sufficiency gates only catch the opposite direction."
    )


# ---------------------------------------------------------------------------
# Helper-side behaviour: the sentinel round-trip, against a fake agent.
# ---------------------------------------------------------------------------

import sys

sys.path.insert(0, str(REPO_ROOT / "experiments"))

from _lib.fresh_select import (  # noqa: E402
    FreshSelectCounter,
    FreshSelectProbe,
    marker_key_for,
)


class _FakeE3:
    def __init__(self):
        self.last_score_diagnostics = {}

    def select(self):
        """Mimic the substrate: wholesale reassignment."""
        self.last_score_diagnostics = {"scored": True}


class _FakeAgent:
    def __init__(self):
        self.e3 = _FakeE3()


def test_probe_reports_fresh_only_when_select_ran():
    agent = _FakeAgent()
    probe = FreshSelectProbe("exqtest")

    with probe.watch(agent) as sel:
        agent.e3.select()
    assert sel.fresh is True, "wholesale reassign must read as FRESH"

    with probe.watch(agent) as sel:
        pass  # held action: select() never runs
    assert sel.fresh is False, "an untouched dict must read as LATCHED"


def test_in_place_write_without_clear_is_the_loud_direction():
    """Sentinel survives -> reads LATCHED -> sufficiency gates fail the run."""
    agent = _FakeAgent()
    probe = FreshSelectProbe("exqtest")
    with probe.watch(agent) as sel:
        agent.e3.last_score_diagnostics["scored"] = True
    assert sel.fresh is False


def test_in_place_clear_is_the_silent_direction():
    """`.clear()` DELETES the sentinel -> reads FRESH even though no rebind ran.

    This is why the structural assertions above demand a dict-LITERAL rebind
    rather than merely "the dict was replaced or refreshed somehow": a
    `clear()`-based in-place refactor is indistinguishable from a real
    selection at the sentinel level, and is caught ONLY at the source level.
    """
    agent = _FakeAgent()
    probe = FreshSelectProbe("exqtest")
    with probe.watch(agent) as sel:
        agent.e3.last_score_diagnostics.clear()
        agent.e3.last_score_diagnostics.update({"scored": True})
    assert sel.fresh is True


def test_namespaces_do_not_collide():
    agent = _FakeAgent()
    a = FreshSelectProbe("exq699b")
    b = FreshSelectProbe("exq689i")
    assert a.marker_key != b.marker_key
    a.mark_stale(agent)
    assert b.is_fresh(agent) is True, (
        "one driver's marker must not make another driver read as latched"
    )
    b.mark_stale(agent)
    assert a.is_fresh(agent) is False


def test_namespace_is_required_and_validated():
    with pytest.raises(TypeError):
        FreshSelectProbe()  # type: ignore[call-arg]
    for bad in ("", "has space", "has-dash"):
        with pytest.raises(ValueError):
            marker_key_for(bad)


def test_result_cannot_be_read_inside_the_block():
    agent = _FakeAgent()
    probe = FreshSelectProbe("exqtest")
    with pytest.raises(RuntimeError):
        with probe.watch(agent) as sel:
            bool(sel)


def test_counter_hold_bookkeeping_matches_migrated_drivers():
    c = FreshSelectCounter()
    # a latch before any fresh selection must NOT open a hold
    c.record(False)
    assert c.hold_durations == [] and c.n_latched == 1

    c.record(True)          # opens a hold at 1
    c.record(False)         # -> 2
    c.record(False)         # -> 3
    c.record(True)          # closes 3, opens 1
    assert c.hold_durations == [3]
    assert c.n_fresh_select == 2

    c.flush()               # episode boundary closes the open hold
    assert c.hold_durations == [3, 1]
    c.flush()               # idempotent: nothing open
    assert c.hold_durations == [3, 1]


def test_counter_derived_quantities():
    c = FreshSelectCounter()
    for _ in range(2):
        c.record(True)
        for _ in range(9):
            c.record(False)
    c.flush()
    d = c.as_dict(n_ticks=20)
    assert d["n_fresh_select"] == 2
    assert d["n_latched"] == 18
    assert d["fresh_select_yield"] == pytest.approx(0.1)
    assert d["replication_factor"] == pytest.approx(10.0)
    assert d["hold_duration_max"] == 10


def test_counter_derived_quantities_are_zero_safe():
    c = FreshSelectCounter()
    d = c.as_dict(n_ticks=0)
    assert d["fresh_select_yield"] == 0.0
    assert d["replication_factor"] == 0.0
    assert d["hold_duration_mean"] == 0.0
    assert d["hold_duration_max"] == 0
