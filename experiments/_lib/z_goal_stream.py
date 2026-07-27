"""
z_goal stream liveness -- the RUNTIME half of the dead-z_goal-stream backstop.

The defect
----------
``REEAgent.update_z_goal(...)`` is the SOLE writer of z_goal in the substrate.
``sense()`` / ``generate_trajectories()`` / ``select_action()`` / ``update_residue()``
all leave it untouched, and both ``GoalState`` mutators (``update`` / ``cue_pull``) are
reached only from inside it. A driver that hand-rolls its inner loop and omits the call
therefore runs with z_goal pinned at zero-init for the whole run: ``GoalState.is_active()``
returns False, ``agent.py`` passes ``current_z_goal=None`` to every consumer, and the E3
goal term, MECH-293 ghost probes, MECH-288's slow BOCPD scale, MECH-189 super-ordinal
anchors, the SD-057 incentive bank, the MECH-295 liking->approach bridge and the
frontopolar counterfactual read all silently no-op. Nothing raises, and until now no
manifest field showed it.

Confirmed twice. V3-EXQ-626's bespoke episode loop never called it, so all five of that
diagnostic's criteria were keyed on a z_goal that never left zero (superseded by 626a /
626b). V3-EXQ-830 reused the V3-EXQ-816 harness -- harmless for 816, which never reads
z_goal -- to measure MECH-288's slow BOCPD scale, which does; its readiness gate refused
the dry-run smoke twice on an ad-hoc ``zgoal_present_frac = 0.0`` before the cause was
traced. Generalising that ad-hoc gate is what this module is.

Why this exists ALONGSIDE the static lint
-----------------------------------------
``validate_experiments.dead_z_goal_stream_lint`` catches the same defect at AUTHORING
time, but it is an AST scan: a config assembled inside a helper it cannot follow (a
``_lib`` builder, a ``**kwargs`` splat, a preset factory) is invisible to its trigger
half, so it UNDER-fires by design -- the safe direction for a WARN, but a real blind
spot. These counters are read from the run itself and so do not share that limitation:
the condition lands in the run's own record whether or not anyone pre-registered a gate
for it.

Why a counter and NOT a warning
-------------------------------
A one-time stderr warning from the substrate was considered and rejected. A zero fraction
is a legitimate and common measurement -- goal-OFF parity arms, and negative controls like
V3-EXQ-626b's ARM_NO_BENEFIT -- so a warning would be noisy where a counter reading 0.0 is
simply correct. A print during a multi-hour run also scrolls past unread and leaves nothing
auditable afterwards, and it would fire across the ~1800-test contract suite. The counter
subsumes the warning's only real value; a ``--dry-run`` smoke can just print it.

That print now exists: ``pack_writer.write_flat_manifest`` emits one ``[smoke]
z_goal_stream:`` line, gated on ``dry_run`` -- so an author sees ``active_frac`` and
``writer_defect`` before committing to a multi-hour run, while every objection above still
holds for the real run and for the contract suite. It is a REPORT, never a gate.

Recording it from a driver
--------------------------
Pass ``agent=`` (one agent, or an iterable for a multi-arm run) or ``z_goal_stream_stats=``
to ``write_flat_manifest`` -- or, when the driver calls ``write_flat_manifest(...,
stamp=False)`` because it stamps upstream, to its own ``stamp_recording_core(...)``, since
``stamp=False`` skips the stamper entirely. For the usual hand-rolled shape -- a fresh agent
built INSIDE a per-cell function -- use ``ZGoalStreamAccumulator`` below rather than
collecting the agents into a run-level list, which would keep every arm x seed agent alive
until the last cell finishes.

Reading the recorded block
--------------------------
``z_goal_stream`` on the manifest::

    {"ticks_total": 12000, "ticks_active": 0, "writer_calls": 0,
     "active_frac": 0.0, "writer_defect": true,
     "goal_state_present": true, "n_agents": 6}

``active_frac`` 0.0 means the stream was DEAD -- every consumer got
``current_z_goal=None`` on every tick. It does NOT by itself mean the driver is buggy,
and ``writer_calls`` is what separates the two causes (this ambiguity is the reason the
third counter exists; it was found by measurement, not anticipated):

* ``writer_calls == 0`` with ``ticks_total > 0`` -> **the defect**, flagged as
  ``writer_defect: true``. The driver never called ``update_z_goal``, so z_goal could not
  have moved. This is V3-EXQ-626 and V3-EXQ-830.
* ``writer_calls > 0`` with ``ticks_active == 0`` -> correctly wired; ``GoalState.update``'s
  benefit gate simply never opened (``benefit_exposure`` below ``goal.benefit_threshold``,
  default 0.1) because the run met no resource. A StepHarness run -- which pins the call as
  invariant 2 and *cannot* carry the defect -- reads exactly this on an env where the agent
  never reaches benefit. Conflating it with the row above would send a reader hunting for a
  call that is already there.
* ``active_frac`` ``null`` means nothing was measured (``ticks_total`` 0): either the agent
  was never stepped, or z_goal_enabled is off. ``goal_state_present`` separates those.
* An intermediate fraction localises the run's WARM-UP PREFIX. ``REEAgent.reset()`` does
  NOT reset ``goal_state`` (z_goal is deliberately cross-episode; pinned by
  ``test_dead_z_goal_stream_lint.test_dzg_agent_reset_does_not_reset_goal_state``), so
  once the stream goes live it stays live for the rest of the run. The inactive ticks are
  therefore the ones BEFORE the first successful write, not per-episode re-zeroing --
  reading it as the latter would understate how long the stream took to come up.

A goal-OFF parity arm and a negative control like V3-EXQ-626b's ARM_NO_BENEFIT both read
0.0 correctly. Interpret against the run's design; this is never a gate.

Duck-typed and stdlib-only, so it imports without torch/ree_core -- matching its sibling
stampers (``inert_arm_knob``, ``dose_saturation``) and keeping ``manifest_core`` free of a
substrate dependency. Every entry point is exception-safe: a missing or odd agent yields
no block rather than an error, because provenance stamping must never crash a run.

ASCII-only output (repo rule).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

MANIFEST_KEY = "z_goal_stream"


def _iter_agents(agent: Any) -> List[Any]:
    """Normalise the `agent` argument to a list of candidate agent objects.

    Accepts a single agent, or any iterable of them (a multi-arm run constructs a
    fresh agent per arm x seed, and the run-level fraction is the pooled one). A
    string/bytes/mapping is NOT treated as an iterable of agents.
    """
    if agent is None:
        return []
    if isinstance(agent, (str, bytes, dict)):
        return []
    if isinstance(agent, Iterable):
        return [a for a in agent if a is not None]
    return [agent]


def _counts(one: Any) -> Optional[Dict[str, Any]]:
    """Pull (ticks_total, ticks_active, goal_state_present) off one agent.

    Returns None when the object does not carry the counters at all -- an object
    that is not a counter-bearing REEAgent contributes nothing rather than zeros,
    so a stray non-agent cannot dilute a real fraction.
    """
    total = getattr(one, "z_goal_ticks_total", None)
    active = getattr(one, "z_goal_ticks_active", None)
    if total is None or active is None:
        return None
    try:
        total_i = int(total)
        active_i = int(active)
        calls_i = int(getattr(one, "z_goal_writer_calls", 0) or 0)
    except (TypeError, ValueError):
        return None
    return {
        "ticks_total": total_i,
        "ticks_active": active_i,
        "writer_calls": calls_i,
        # `goal_state is not None` is the enablement signal: agent.py constructs
        # GoalState only when config.goal.z_goal_enabled is set. It is what tells a
        # reader whether ticks_total == 0 means "goal off" or "never stepped".
        "goal_state_present": getattr(one, "goal_state", None) is not None,
    }


def stats_from_counts(
    ticks_total: int,
    ticks_active: int,
    *,
    writer_calls: int = 0,
    goal_state_present: Optional[bool] = None,
    n_agents: int = 1,
) -> Dict[str, Any]:
    """Build the recorded block from raw counts.

    Split out from `z_goal_stream_stats` so a caller holding its own counters (the
    StepHarness, which accumulates across agent swaps) records the identical shape
    rather than a parallel one that drifts.

    `active_frac` is None when the denominator is zero -- 0.0 there would be a FALSE
    reading (nothing measured) rather than a weak one, and the two must not be
    confusable by a reader scanning for zeros.

    `writer_defect` is the machine-readable verdict, and the reason `writer_calls` is
    recorded at all: a zero `active_frac` alone cannot tell a driver that never called
    `update_z_goal` (the defect) from one that called it every tick while the benefit
    gate never opened (correct wiring, no benefit encountered -- what a StepHarness run
    reads on an env where the agent meets no resource). True ONLY for the former. None
    when nothing was measured.
    """
    total_i = max(0, int(ticks_total))
    active_i = max(0, int(ticks_active))
    calls_i = max(0, int(writer_calls))
    block: Dict[str, Any] = {
        "ticks_total": total_i,
        "ticks_active": active_i,
        "writer_calls": calls_i,
        "active_frac": (float(active_i) / float(total_i)) if total_i > 0 else None,
        "writer_defect": (total_i > 0 and calls_i == 0) if total_i > 0 else None,
        "n_agents": int(n_agents),
    }
    if goal_state_present is not None:
        block["goal_state_present"] = bool(goal_state_present)
    return block


def z_goal_stream_stats(agent: Any) -> Optional[Dict[str, Any]]:
    """Pooled z_goal-liveness stats for one agent or an iterable of them.

    Returns None when no counter-bearing agent was supplied (nothing to record --
    the block is omitted rather than written as zeros, so a manifest carrying it
    always means the run actually measured it).
    """
    agents = _iter_agents(agent)
    if not agents:
        return None
    total = 0
    active = 0
    calls = 0
    present = False
    n = 0
    for one in agents:
        c = _counts(one)
        if c is None:
            continue
        total += c["ticks_total"]
        active += c["ticks_active"]
        calls += c["writer_calls"]
        # ANY arm having goal_state live makes the run's z_goal path live -- a
        # deliberate goal-OFF control arm beside goal-ON arms must not mask it.
        present = present or bool(c["goal_state_present"])
        n += 1
    if n == 0:
        return None
    return stats_from_counts(
        total, active,
        writer_calls=calls, goal_state_present=present, n_agents=n,
    )


class ZGoalStreamAccumulator:
    """Run-level tally for a driver that builds a FRESH agent per arm x seed.

    Why this exists beside ``z_goal_stream_stats(agent)``. The pooled helper takes the
    agents themselves, which is right for a driver holding all of them at manifest time.
    But the landed hand-rolled corpus overwhelmingly builds its agent INSIDE a per-cell
    function (``_run_cell(arm, seed)`` -> row) and lets it fall out of scope when the
    cell returns. Wiring those to ``agent=[...]`` would mean appending every agent to a
    run-level list purely for provenance, so a 3-arm x 5-seed training run keeps fifteen
    fully-populated agents -- nets, optimiser state, replay buffers, hippocampal banks --
    alive until the last cell finishes. That is a real peak-memory change on a 3 GB
    cloud worker, imposed by a provenance field. This reads the three counters off the
    agent at end-of-cell and drops the reference.

    The same reason applies to a ``StepHarness`` driver: the harness is cell-local too,
    so its ``z_goal_stream_stats()`` block needs pooling across cells -- ``observe_stats``.

    ORDERING IS THE ONE TRAP: ``observe`` reads the counters AT CALL TIME, so it must be
    called AFTER the cell has finished stepping, not at construction. Called on a fresh
    agent it contributes ticks_total 0 and, worse, would make a live run look unmeasured.
    ``observe`` deliberately returns None rather than the agent, so the tempting
    ``agent = acc.observe(REEAgent(cfg))`` chain at the construction site does not
    typecheck as a drop-in and the mistake surfaces immediately.

    Usage::

        _ZG = ZGoalStreamAccumulator()

        def _run_cell(arm, seed):
            agent = REEAgent(cfg)
            ...                       # step the agent
            _ZG.observe(agent)        # AFTER stepping
            return row

        write_flat_manifest(manifest, ..., z_goal_stream_stats=_ZG.stats())
    """

    __slots__ = ("_total", "_active", "_calls", "_present", "_n")

    def __init__(self) -> None:
        self._total = 0
        self._active = 0
        self._calls = 0
        self._present = False
        self._n = 0

    def observe(self, agent: Any) -> None:
        """Fold one FINISHED cell's agent (or iterable of agents) into the tally.

        Exception-safe and duck-typed, matching the module's posture: a non-agent
        contributes nothing rather than raising or diluting the fraction with zeros.
        """
        try:
            for one in _iter_agents(agent):
                c = _counts(one)
                if c is None:
                    continue
                self._total += c["ticks_total"]
                self._active += c["ticks_active"]
                self._calls += c["writer_calls"]
                self._present = self._present or bool(c["goal_state_present"])
                self._n += 1
        except Exception:
            pass

    def observe_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """Fold one cell's precomputed block -- e.g. ``StepHarness.z_goal_stream_stats()``.

        ``n_agents`` is summed rather than counted as one, so a block that already
        pooled several agents keeps its weight in the run-level ``n_agents``.
        """
        try:
            if not isinstance(stats, dict):
                return
            self._total += max(0, int(stats.get("ticks_total") or 0))
            self._active += max(0, int(stats.get("ticks_active") or 0))
            self._calls += max(0, int(stats.get("writer_calls") or 0))
            self._present = self._present or bool(stats.get("goal_state_present"))
            self._n += max(1, int(stats.get("n_agents") or 1))
        except Exception:
            pass

    def stats(self) -> Optional[Dict[str, Any]]:
        """The run-level block, or None when nothing was observed.

        None (not a zero-filled block) so the manifest keeps the module's invariant:
        a recorded ``z_goal_stream`` always means the run actually measured it.
        """
        if self._n == 0:
            return None
        return stats_from_counts(
            self._total, self._active,
            writer_calls=self._calls,
            goal_state_present=self._present,
            n_agents=self._n,
        )


def stamp_z_goal_stream(
    manifest: Dict[str, Any],
    agent: Any = None,
    *,
    stats: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Merge the `z_goal_stream` block onto `manifest` in place; return the manifest.

    NO-OP-SAFE and additive, matching `stamp_recording_core`'s posture: an existing
    block a script set deliberately (e.g. counts pooled by hand across a driver's own
    agent bookkeeping) wins unless `overwrite=True`. Pass EITHER `agent` (one agent or
    an iterable) or a precomputed `stats` dict from `stats_from_counts`.

    Never raises: provenance stamping must not be able to crash an experiment at
    manifest-write time, when the compute is already spent.
    """
    try:
        if not isinstance(manifest, dict):
            return manifest
        if not overwrite and manifest.get(MANIFEST_KEY):
            return manifest
        block = stats if stats is not None else z_goal_stream_stats(agent)
        if block:
            manifest[MANIFEST_KEY] = block
    except Exception:
        pass
    return manifest


__all__ = [
    "MANIFEST_KEY",
    "ZGoalStreamAccumulator",
    "stats_from_counts",
    "z_goal_stream_stats",
    "stamp_z_goal_stream",
]
