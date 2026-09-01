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

The training-phase-agent false positive (V3-EXQ-874b, 2026-08-19)
-------------------------------------------------------------------
``writer_calls == 0`` with ``ticks_total > 0`` is documented above as "the defect" --
but that inference is WRONG when the observed object is a TRAINING-phase agent rather
than the one actually driving the run's eval loop. ``REEAgent.select_action`` bumps
``ticks_total`` on every call, including calls made from an ordinary supervised/RL
TRAINING loop (e.g. P0 world-model warmup via ``committed_mode_curriculum.run_p0_warmup``),
where ``update_z_goal`` is never called because z_goal has nothing to do with that phase.
V3-EXQ-874b's driver called ``zg_acc.observe(agent)`` on the P0-trained BASE agent, while
every cell actually stepped a CLONE from ``clone_trained_agent`` (a fresh object -- cloning
copies weight ``state_dict()``, not these counters) -- so the base agent's ``ticks_total``
was 100% P0-training ticks and its ``writer_calls`` was legitimately 0, and the accumulator
published ``writer_defect: true`` into the manifest and ``pending_review.md`` even though
``update_z_goal`` ran throughout eval on the (unobserved) clone.

**Why this cannot be auto-detected from object state alone.** A tempting fix is to key off
``agent.training`` (the standard ``nn.Module`` flag set by ``.train()``/``.eval()``): the
874b base agent really was left in ``.train()`` mode at observe time. But this is not a
general discriminator -- ``nn.Module.__init__`` defaults ``training`` to ``True``, and the
corpus does not consistently call ``.eval()`` before a genuine eval loop (``StepHarness``
itself never toggles it, and V3-EXQ-830's own driver -- the confirmed real defect this
module exists to catch -- never calls ``.eval()``/``.train()`` either). Gating on
``.training`` would therefore read V3-EXQ-830's true positive as an equally plausible
"training-phase" false positive: the same object-state signature, opposite verdict. No
purely-passive signal on the object distinguishes "ticks came from training, and the
writer is legitimately silent" from "ticks came from a genuinely broken eval loop" --
the two are only distinguishable by DRIVER INTENT, which this module cannot observe. That
is not an oversight; it is why this fix is an EXPLICIT opt-in rather than a heuristic.

**The fix: an explicit ``eval_stepped`` flag on every observation entry point**
(``z_goal_stream_stats``, ``ZGoalStreamAccumulator.observe``/``observe_stats``,
``stamp_z_goal_stream``), default ``True`` (unchanged behaviour for every existing call
site). A driver that knowingly observes a training-only object -- most commonly a P0 base
agent kept around after ``clone_trained_agent`` -- passes ``eval_stepped=False``. Those
ticks/calls are pooled SEPARATELY from the eval-facing counters, under
``training_phase_ticks_total`` / ``training_phase_ticks_active`` /
``training_phase_writer_calls`` / ``training_phase_n_agents``, and are EXCLUDED from
``writer_defect``: an observation that is training-only in its entirety reports
``writer_defect: null`` (unmeasured, exactly like ``ticks_total: 0``) rather than a false
``true``. This does not retroactively fix a driver that observes the wrong object
outright (874b's actual bug -- the fix there, landed in V3-EXQ-940/941, is to observe the
stepping clone instead); it gives a driver that DOES know it is holding a training-only
object a correct, explicit way to say so instead of silently mis-reporting.

The pinned-goal false positive (V3-EXQ-642b, 2026-09-01)
----------------------------------------------------------
A second, distinct cause of ``writer_calls == 0`` with ``ticks_total > 0``: a driver that
deliberately holds z_goal at a fixed magnitude as an experimental control, by writing
``agent.goal_state._z_goal`` directly instead of going through ``update_z_goal``
(V3-EXQ-642a/642b's ``_pin_goal``, called every tick before ``sense()``). This produces
``active_frac == 1.0`` (``GoalState.is_active()`` is a plain nonzero-check on ``_z_goal``,
so a constant nonzero pin reads active on every tick) alongside ``writer_calls == 0`` --
by the letter of the inference above, ``writer_defect: true``, and it reached
``pending_review.md`` as V3-EXQ-642b (failure_autopsy_V3-EXQ-642b_2026-09-01).

**Why ``writer_calls == 0`` with ``active_frac == 1.0`` is NOT a safe auto-detection
signature for "this was a deliberate pin", even though it is what a pin produces.**
``update_z_goal`` is documented above as the sole writer, but it is not the only
*entry point* that can move ``_z_goal``: ``REEAgent.cue_recall_wanting`` calls
``GoalState.cue_pull`` directly and is driver-callable on its own (see
``experiments/_harness.py``, ``scaffolded_sd054_onboarding.py`` and the SD-057
cue-recall drivers) -- so a run driven purely by cue-recall wanting, never calling
``update_z_goal`` at all, legitimately reads ``writer_calls == 0`` with a live,
non-degenerate active fraction. That is not a pin and not a defect; it is a third
shape the counter cannot distinguish from a pin by number alone. This is the same
shape of problem as the training-phase case just above -- the signature is
CONSISTENT with a pin, but not UNIQUE to one -- so the fix is the same: an explicit
opt-in, never a heuristic on the counters.

**The fix: an explicit ``goal_pinned`` flag on every observation entry point**
(``z_goal_stream_stats``, ``ZGoalStreamAccumulator.observe``/``observe_stats``,
``stamp_z_goal_stream``, ``stats_from_counts``), default ``False`` (unchanged
behaviour for every existing call site). A driver that deliberately pins z_goal
outside the writer passes ``goal_pinned=True``; the pooled ticks stay in the
ordinary eval counters (they are real eval-loop activity, unlike the training-phase
case, which is why this is pooled rather than moved to a side channel), but
``writer_defect`` reports ``null`` -- unmeasured/not-applicable, exactly like
``ticks_total: 0`` -- rather than a false ``true``, and the block carries
``goal_pinned: true`` so a reader can tell the null apart from the "nothing
measured" case. An unrecognised shape (the flag omitted) still reports the
defect: fixing V3-EXQ-642b's false positive must not create a false negative for
a driver that actually forgot the call.

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
    goal_pinned: bool = False,
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

    `goal_pinned=True` -- see the module docstring's "pinned-goal false positive"
    section -- suppresses that inference for a driver that KNOWINGLY writes z_goal
    outside `update_z_goal` as a deliberate experimental control. `writer_defect`
    then reports None (not-applicable) rather than a false `true`, and the block
    carries `goal_pinned: true` so a reader can tell this null apart from the
    "nothing measured" one. Default False -- unchanged behaviour for every
    existing call site; an unrecognised shape still reports the defect.
    """
    total_i = max(0, int(ticks_total))
    active_i = max(0, int(ticks_active))
    calls_i = max(0, int(writer_calls))
    pinned = bool(goal_pinned)
    if total_i <= 0:
        defect: Optional[bool] = None
    elif pinned:
        defect = None
    else:
        defect = calls_i == 0
    block: Dict[str, Any] = {
        "ticks_total": total_i,
        "ticks_active": active_i,
        "writer_calls": calls_i,
        "active_frac": (float(active_i) / float(total_i)) if total_i > 0 else None,
        "writer_defect": defect,
        "n_agents": int(n_agents),
    }
    if goal_state_present is not None:
        block["goal_state_present"] = bool(goal_state_present)
    if pinned:
        block["goal_pinned"] = True
    return block


def _training_phase_block(
    total: int, active: int, calls: int, n: int, present: bool,
) -> Dict[str, Any]:
    """An eval-shaped block relabelled as training-phase-only.

    Used when the caller has told us (``eval_stepped=False``) that every observed
    agent's ticks came from a phase where ``update_z_goal`` is legitimately never
    called (P0 warmup and the like), not from the driver's actual eval loop.
    Reporting these under ``ticks_total``/``writer_calls`` would still read
    ``writer_defect: true`` by the letter of the inference above -- exactly the
    V3-EXQ-874b false positive this exists to prevent -- so they are moved to
    ``training_phase_*`` keys and the eval-facing fields report UNMEASURED
    (``writer_defect: null``), the same as a genuine ``ticks_total: 0``.
    """
    return {
        "ticks_total": 0,
        "ticks_active": 0,
        "writer_calls": 0,
        "active_frac": None,
        "writer_defect": None,
        "goal_state_present": bool(present),
        "n_agents": 0,
        "training_phase_ticks_total": int(total),
        "training_phase_ticks_active": int(active),
        "training_phase_writer_calls": int(calls),
        "training_phase_n_agents": int(n),
    }


def z_goal_stream_stats(
    agent: Any, *, eval_stepped: bool = True, goal_pinned: bool = False,
) -> Optional[Dict[str, Any]]:
    """Pooled z_goal-liveness stats for one agent or an iterable of them.

    Returns None when no counter-bearing agent was supplied (nothing to record --
    the block is omitted rather than written as zeros, so a manifest carrying it
    always means the run actually measured it).

    ``eval_stepped=False`` -- pass when the caller KNOWS the given agent(s) are
    training-phase-only objects (see the module docstring's "training-phase-agent
    false positive" section), e.g. a P0 base agent kept around after
    ``clone_trained_agent``. The pooled counts are then reported under
    ``training_phase_*`` keys instead of ``ticks_total``/``writer_calls``, and
    ``writer_defect`` reports unmeasured rather than a false positive.

    ``goal_pinned=True`` -- pass when the driver KNOWINGLY writes z_goal outside
    ``update_z_goal`` as a deliberate experimental control (see the module
    docstring's "pinned-goal false positive" section). Passed through to
    ``stats_from_counts``; ignored when ``eval_stepped=False`` (a training-phase
    observation is already excluded from ``writer_defect`` on its own terms).
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
    if not eval_stepped:
        return _training_phase_block(total, active, calls, n, present)
    return stats_from_counts(
        total, active,
        writer_calls=calls, goal_state_present=present, n_agents=n,
        goal_pinned=goal_pinned,
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

    __slots__ = (
        "_total", "_active", "_calls", "_present", "_n", "_pinned",
        "_train_total", "_train_active", "_train_calls", "_train_n", "_train_present",
    )

    def __init__(self) -> None:
        self._total = 0
        self._active = 0
        self._calls = 0
        self._present = False
        self._n = 0
        # See the module docstring's "pinned-goal false positive" section.
        # OR-reduced across observations, same idiom as `_present`: any
        # pinned observation in the run suppresses `writer_defect` for the
        # whole pooled tally, since the pooled counters cannot be split back
        # apart by arm any more than `active_frac` can.
        self._pinned = False
        # Training-phase-only tally -- see the module docstring's
        # "training-phase-agent false positive" section. Kept fully separate from
        # the eval tally above so a training-only observation can never contribute
        # to writer_defect.
        self._train_total = 0
        self._train_active = 0
        self._train_calls = 0
        self._train_n = 0
        self._train_present = False

    def observe(
        self, agent: Any, *, eval_stepped: bool = True, goal_pinned: bool = False,
    ) -> None:
        """Fold one FINISHED cell's agent (or iterable of agents) into the tally.

        Exception-safe and duck-typed, matching the module's posture: a non-agent
        contributes nothing rather than raising or diluting the fraction with zeros.

        ``eval_stepped=False`` -- pass when this agent is known to be a
        training-phase-only object (e.g. the P0 base agent, as opposed to the
        ``clone_trained_agent`` clone that actually stepped the eval loop -- the
        V3-EXQ-874b shape). Its counts are pooled into a separate training-phase
        tally that ``stats()`` reports but never folds into ``writer_defect``.

        ``goal_pinned=True`` -- pass when this agent's z_goal was deliberately
        pinned outside ``update_z_goal`` (see the module docstring's "pinned-goal
        false positive" section). Ignored when ``eval_stepped=False``.
        """
        try:
            for one in _iter_agents(agent):
                c = _counts(one)
                if c is None:
                    continue
                if eval_stepped:
                    self._total += c["ticks_total"]
                    self._active += c["ticks_active"]
                    self._calls += c["writer_calls"]
                    self._present = self._present or bool(c["goal_state_present"])
                    self._n += 1
                    self._pinned = self._pinned or bool(goal_pinned)
                else:
                    self._train_total += c["ticks_total"]
                    self._train_active += c["ticks_active"]
                    self._train_calls += c["writer_calls"]
                    self._train_present = self._train_present or bool(c["goal_state_present"])
                    self._train_n += 1
        except Exception:
            pass

    def observe_stats(
        self,
        stats: Optional[Dict[str, Any]],
        *,
        eval_stepped: bool = True,
        goal_pinned: bool = False,
    ) -> None:
        """Fold one cell's precomputed block -- e.g. ``StepHarness.z_goal_stream_stats()``.

        ``n_agents`` is summed rather than counted as one, so a block that already
        pooled several agents keeps its weight in the run-level ``n_agents``.
        ``eval_stepped=False`` routes the block into the training-phase tally --
        see ``observe()``. ``goal_pinned=True`` marks the eval tally as pinned --
        see ``observe()`` and the module docstring's "pinned-goal false positive"
        section.
        """
        try:
            if not isinstance(stats, dict):
                return
            total = max(0, int(stats.get("ticks_total") or 0))
            active = max(0, int(stats.get("ticks_active") or 0))
            calls = max(0, int(stats.get("writer_calls") or 0))
            present = bool(stats.get("goal_state_present"))
            n = max(1, int(stats.get("n_agents") or 1))
            if eval_stepped:
                self._total += total
                self._active += active
                self._calls += calls
                self._present = self._present or present
                self._n += n
                self._pinned = self._pinned or bool(goal_pinned)
            else:
                self._train_total += total
                self._train_active += active
                self._train_calls += calls
                self._train_present = self._train_present or present
                self._train_n += n
        except Exception:
            pass

    def stats(self) -> Optional[Dict[str, Any]]:
        """The run-level block, or None when nothing was observed.

        None (not a zero-filled block) so the manifest keeps the module's invariant:
        a recorded ``z_goal_stream`` always means the run actually measured it.

        When only training-phase-only observations were folded in (``eval_stepped=
        False`` on every ``observe``/``observe_stats`` call), the eval-facing fields
        report unmeasured (``writer_defect: null``) and the raw counts are surfaced
        under ``training_phase_*`` -- see ``_training_phase_block``.
        """
        if self._n == 0 and self._train_n == 0:
            return None
        # goal_state_present answers "was z_goal even configured", which is true
        # regardless of which phase observed it -- merge both tallies for it, unlike
        # every other field, which stays strictly phase-separated.
        present = self._present or self._train_present
        if self._n > 0:
            block = stats_from_counts(
                self._total, self._active,
                writer_calls=self._calls,
                goal_state_present=present,
                n_agents=self._n,
                goal_pinned=self._pinned,
            )
            if self._train_n > 0:
                block["training_phase_ticks_total"] = self._train_total
                block["training_phase_ticks_active"] = self._train_active
                block["training_phase_writer_calls"] = self._train_calls
                block["training_phase_n_agents"] = self._train_n
            return block
        return _training_phase_block(
            self._train_total, self._train_active, self._train_calls,
            self._train_n, present,
        )


def stamp_z_goal_stream(
    manifest: Dict[str, Any],
    agent: Any = None,
    *,
    stats: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    eval_stepped: bool = True,
    goal_pinned: bool = False,
) -> Dict[str, Any]:
    """Merge the `z_goal_stream` block onto `manifest` in place; return the manifest.

    NO-OP-SAFE and additive, matching `stamp_recording_core`'s posture: an existing
    block a script set deliberately (e.g. counts pooled by hand across a driver's own
    agent bookkeeping) wins unless `overwrite=True`. Pass EITHER `agent` (one agent or
    an iterable) or a precomputed `stats` dict from `stats_from_counts`.

    `eval_stepped=False` passes through to `z_goal_stream_stats` when `agent` is
    given -- see the module docstring's "training-phase-agent false positive"
    section. `goal_pinned=True` likewise passes through -- see the "pinned-goal
    false positive" section. Both are ignored when a precomputed `stats` block is
    passed directly; the caller built that block itself and is responsible for
    its shape.

    Never raises: provenance stamping must not be able to crash an experiment at
    manifest-write time, when the compute is already spent.
    """
    try:
        if not isinstance(manifest, dict):
            return manifest
        if not overwrite and manifest.get(MANIFEST_KEY):
            return manifest
        block = (
            stats if stats is not None
            else z_goal_stream_stats(
                agent, eval_stepped=eval_stepped, goal_pinned=goal_pinned,
            )
        )
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
