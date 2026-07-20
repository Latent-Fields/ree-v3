"""Shared E3 "fresh selection" instrumentation (the 699 / 689d instrument repair).

WHAT THIS GUARDS AGAINST -- the hold-weighted-readout defect class
------------------------------------------------------------------
`agent.py:5430` returns the HELD action on `not ticks["e3_tick"]`, BEFORE
`e3.select()` is ever reached. E3 cadence defaults to 10
(`heartbeat.e3_steps_per_tick`), so any per-env-step read of `agent.e3.last_*`
is REPLICATED by the hold duration: a quantity sampled once per genuine
selection is silently counted ~10 times, once per env step of the hold.

That replication is disqualifying whenever the statistic is not invariant to
it -- an entropy, a class-occupancy distribution, a pairwise distance, or any
"active fraction" whose denominator is env steps. Replication reweights the
very distribution the statistic is computed over, and it does so UNEQUALLY
across arms when the arms differ in hold duration (which is exactly what an
arm that changes commitment dynamics does). See the corpus sweep
`hold_weighted_e3_readout_corpus_sweep_2026-07-20` and the 699 / 689d
autopsies.

HOW THE DETECTION WORKS -- the sentinel key
-------------------------------------------
`e3_selector.select()` reassigns `last_score_diagnostics` WHOLESALE to a fresh
dict literal (`e3_selector.py:2452`). That is the ONLY wholesale assignment
inside `select()` -- every other write to the attribute is an item-mutation
(`last_score_diagnostics[k] = v`), which leaves pre-existing keys intact, and
`__init__` (line 358) does not run per tick. So:

  * stamp a private key into `agent.e3.last_score_diagnostics` immediately
    BEFORE every `select_action()`;
  * after `select_action()`, the key is ABSENT iff `select()` actually ran.

The invariant is load-bearing and is pinned by
`tests/contracts/test_fresh_select_wholesale_reassign.py`. If `select()` were
ever changed to MUTATE the dict in place rather than reassign it, the sentinel
would report EVERY tick as fresh -- silently, and NOT self-catching, because
the inverse failure (everything reported as latched) is caught by the
sufficiency gate on `n_fresh_select` while this one is not.

WHY A SENTINEL KEY AND NOT `= None` (a deliberate, documented deviation from
the autopsy's routing_detail requirement 1 and the v3_exq_785a reference)
---------------------------------------------------------------------------
  * Nulling `_last_selected_trajectory` CHANGES SUBSTRATE BEHAVIOUR.
    `post_action_update` (`e3_selector.py:3224`) falls back to it when
    `_committed_trajectory` is None -- the ARC-016 deadlock fix -- and it runs
    on EVERY step via `update_residue` (`agent.py:8006`), not only on E3 ticks.
    Nulling it silently SKIPS the running-variance update and the
    prediction_error / dynamic_precision metrics on non-E3 ticks with no live
    commitment. That perturbs selection dynamics, which would make the run a
    DIFFERENT experiment rather than a repaired instrument.
  * Nulling `last_score_diagnostics` is None-SAFE but not inert either:
    `_assemble_control_vector` (`agent.py:9660`) reads it on non-E3 ticks and
    would fall back to authority defaults instead of the persisted E3 values.

The sentinel key preserves both attributes byte-for-byte -- the only ree_core
reader of the dict is `agent.py:9660`, which uses `.get()` with defaults, and
nothing in ree_core iterates its keys (verified) -- so the marker is INERT
while giving an exact per-tick freshness signal.

NAMESPACING (required)
----------------------
The marker key is namespaced per driver and the namespace is a REQUIRED
argument, so two concurrently-instrumented drivers sharing one agent can never
collide on the same dict.

USAGE
-----
    from _lib.fresh_select import FreshSelectProbe, FreshSelectCounter

    probe = FreshSelectProbe("exq699b")      # -> "_exq699b_stale_marker"
    counter = FreshSelectCounter()

    for ep in range(n_episodes):
        counter.flush()                       # no hold may span an episode
        for step in range(n_steps):
            with probe.watch(agent) as sel:
                action = agent.select_action(candidates, ticks)
            if in_measured_phase:
                counter.record(sel.fresh)
                if sel.fresh:
                    ...                       # accumulate per-SELECTION only
    counter.flush()                           # flush a hold left open at the end
    row.update(counter.as_dict(n_ticks))

`record()` is deliberately NOT called from inside `watch()`: every migrated
driver gates accumulation on a phase (`is_p1` / `is_p2`), so the caller decides
when a selection counts.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

__all__ = [
    "FreshSelectProbe",
    "FreshSelectCounter",
    "FreshSelectResult",
    "STALE_MARKER_SUFFIX",
    "marker_key_for",
    "FRESH_SELECT_RATIONALE",
]

# Marker keys are `_<namespace>_stale_marker`; the suffix is shared so
# validate_experiments.py can recognise the helper's key shape.
STALE_MARKER_SUFFIX = "_stale_marker"

# Canonical rationale string. Migrated drivers cite this so the lint exemption
# text (where still needed) stays in one place rather than drifting per copy.
FRESH_SELECT_RATIONALE = (
    "E3 staleness is handled via the SHARED helper experiments/_lib/fresh_select.py, "
    "which stamps a namespaced sentinel key into agent.e3.last_score_diagnostics "
    "before every select_action() and detects a genuine selection by that key's "
    "absence afterwards (select() reassigns the dict wholesale, e3_selector.py:2452). "
    "A sentinel is used INSTEAD of an `agent.e3.<attr> = None` clear because nulling "
    "_last_selected_trajectory changes substrate behaviour via post_action_update "
    "(the ARC-016 deadlock fallback, which runs on EVERY step through update_residue), "
    "which would make the run a different experiment rather than a repaired "
    "instrument. All accumulation is fresh-gated; n_fresh_select / n_latched / "
    "fresh_select_yield / replication_factor are emitted per cell."
)


def marker_key_for(namespace: str) -> str:
    """Return the sentinel key for `namespace`, validating it."""
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("fresh_select namespace must be a non-empty string")
    if not namespace.replace("_", "").isalnum():
        raise ValueError(
            "fresh_select namespace must be alphanumeric/underscore, "
            f"got {namespace!r}"
        )
    return "_" + namespace + STALE_MARKER_SUFFIX


class FreshSelectResult:
    """Freshness verdict for one `select_action()` call.

    `fresh` is False until the enclosing `watch()` block exits, so reading it
    inside the block is always a programming error rather than a silent
    always-latched reading.
    """

    __slots__ = ("fresh", "_closed")

    def __init__(self) -> None:
        self.fresh = False
        self._closed = False

    def __bool__(self) -> bool:
        if not self._closed:
            raise RuntimeError(
                "FreshSelectResult read before its watch() block closed -- move the "
                "read after the `with` block"
            )
        return self.fresh


class FreshSelectProbe:
    """Per-driver sentinel-key freshness probe. `namespace` is REQUIRED."""

    __slots__ = ("namespace", "marker_key")

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.marker_key = marker_key_for(namespace)

    def mark_stale(self, agent: Any) -> None:
        """Stamp the sentinel. Call IMMEDIATELY before `select_action()`."""
        diag = getattr(getattr(agent, "e3", None), "last_score_diagnostics", None)
        if isinstance(diag, dict):
            diag[self.marker_key] = True

    def is_fresh(self, agent: Any) -> bool:
        """True iff `select()` ran since the last `mark_stale()`."""
        diag = getattr(getattr(agent, "e3", None), "last_score_diagnostics", None)
        return isinstance(diag, dict) and self.marker_key not in diag

    @contextmanager
    def watch(self, agent: Any) -> Iterator[FreshSelectResult]:
        """Mark before the block, resolve freshness after it.

        Pairing the mark and the check in one construct is the point: a
        hand-copied pattern can drift into checking without marking (reports
        everything fresh) or marking without checking.
        """
        res = FreshSelectResult()
        self.mark_stale(agent)
        try:
            yield res
        finally:
            res.fresh = self.is_fresh(agent)
            res._closed = True

    def diagnostics(self, agent: Any, fresh: bool) -> Dict[str, Any]:
        """The diagnostics dict on a fresh tick, else `{}`.

        The empty-dict-on-latched return makes every downstream
        `diag.get(...)` fresh-gated without an `if` around each one. The
        sentinel key is never present on a fresh tick, so no stripping is
        needed.
        """
        if not fresh:
            return {}
        diag = getattr(getattr(agent, "e3", None), "last_score_diagnostics", None)
        return diag if isinstance(diag, dict) else {}


class FreshSelectCounter:
    """Accumulates fresh/latched counts and hold durations.

    Hold bookkeeping reproduces the migrated drivers exactly:
      * on a FRESH selection: close any open hold, then open a new one at 1;
      * on a LATCHED tick: extend an already-open hold (a latch before the
        first fresh selection does NOT open one);
      * `flush()` at every episode boundary and once at the end -- `agent.reset()`
        clears the commitment latch, so a hold cannot span episodes.
    """

    __slots__ = ("n_fresh_select", "n_latched", "hold_durations", "_cur_hold")

    def __init__(self) -> None:
        self.n_fresh_select = 0
        self.n_latched = 0
        self.hold_durations: List[int] = []
        self._cur_hold = 0

    def record(self, fresh: bool) -> None:
        if fresh:
            self.n_fresh_select += 1
            if self._cur_hold > 0:
                self.hold_durations.append(self._cur_hold)
            self._cur_hold = 1
        else:
            self.n_latched += 1
            if self._cur_hold > 0:
                self._cur_hold += 1

    def flush(self) -> None:
        """Close a hold left open at an episode boundary or at the end."""
        if self._cur_hold > 0:
            self.hold_durations.append(self._cur_hold)
            self._cur_hold = 0

    # ----- derived quantities -------------------------------------------------
    def fresh_select_yield(self, n_ticks: int) -> float:
        """n_fresh_select / n_ticks, where n_ticks counts ENV STEPS."""
        return float(self.n_fresh_select) / float(n_ticks) if n_ticks > 0 else 0.0

    def replication_factor(self, n_ticks: int) -> float:
        """n_ticks / n_fresh_select -- the measured hold-replication factor."""
        return (
            float(n_ticks) / float(self.n_fresh_select)
            if self.n_fresh_select > 0 else 0.0
        )

    def hold_duration_mean(self) -> float:
        hd = self.hold_durations
        return float(sum(hd)) / float(len(hd)) if hd else 0.0

    def hold_duration_median(self) -> float:
        hd = sorted(self.hold_durations)
        return float(hd[len(hd) // 2]) if hd else 0.0

    def hold_duration_max(self) -> int:
        return int(max(self.hold_durations)) if self.hold_durations else 0

    def hold_duration_hist(self) -> Dict[str, int]:
        return {
            str(k): int(v)
            for k, v in sorted(Counter(self.hold_durations).items())
        }

    def as_dict(self, n_ticks: int, include_hist: bool = True) -> Dict[str, Any]:
        """Canonical manifest row fragment.

        `n_ticks` is the ENV-STEP count for the measured phase; its ratio to
        n_fresh_select IS the replication factor.
        """
        out: Dict[str, Any] = {
            "n_fresh_select": int(self.n_fresh_select),
            "n_latched": int(self.n_latched),
            "fresh_select_yield": round(self.fresh_select_yield(n_ticks), 6),
            "replication_factor": round(self.replication_factor(n_ticks), 6),
            "hold_duration_mean": round(self.hold_duration_mean(), 6),
            "hold_duration_median": round(self.hold_duration_median(), 6),
            "hold_duration_max": int(self.hold_duration_max()),
            "hold_duration_n": int(len(self.hold_durations)),
        }
        if include_hist:
            out["hold_duration_hist"] = self.hold_duration_hist()
        return out
