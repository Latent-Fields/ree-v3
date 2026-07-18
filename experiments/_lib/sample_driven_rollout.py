"""Sample-driven cell rollout for the 2x2 read-only telemetry-probe family.

WHY THIS EXISTS
---------------
`REE_assembly/evidence/planning/failure_autopsy_MECH-063-777-779-cluster_2026-07-18.md`
found that V3-EXQ-777 and V3-EXQ-779 both failed for ONE structural reason, and
that the reason belongs to the shared probe TEMPLATE rather than to either
experiment:

    The template fixes its sampling budget in EPISODES, while its readiness gates
    and criterion bars are denominated in SELECTIONS / EVENT TICKS / PER-SEED
    COUNTS. Because these probes run an UNTRAINED agent in a HAZARD-TERMINATING
    environment, episodes terminate early and seed-dependently.

Measured `n_env_steps` per cell (of a 900-step budget), reproducing near-
identically across two structurally different experiments:

    seed 11 -> ~220-320   seed 17 -> ~100-240   seed 23 -> ~20 (!)
    seed 29 -> 900        seed 37 -> 900

A 40x spread in sample yield across one seed set. Both runs then scored exactly
3/5 seeds against a `min_seeds = 4` bar -- arithmetically unreachable regardless
of the hypothesis's truth value.

THE FIX THIS MODULE PROVIDES
----------------------------
Stopping is denominated in the SAME UNIT as the probe's readiness gates (fresh
E3 selections, event ticks, ...), not in episodes. A cell runs -- auto-resetting
the env across episodes -- until every declared sample floor is met, subject to a
hard total-step cap. When the cap binds before the floors are met, that fact is
returned EXPLICITLY so the caller can self-route `sample_starvation_requeue` and
NAME THE OFFENDING CELL, rather than emitting `substrate_not_ready_requeue` for
what is actually a sampling failure (autopsy learning 4).

THE SAME DEFECT, ONE LAYER DOWN (V3-EXQ-779a)
---------------------------------------------
Making the stopping RULE sample-driven does not make the CAP sample-driven.
779a adopted this module but passed an explicit `max_episodes=120` against a
2400-step cap; seed 23 / arm T1P1 dies in ~7 steps per episode, so the EPISODE
cap bound at 835 of its 2400 available steps and the run was withheld. The
explicit 120 was an override of behaviour that was already correct by default
(`__post_init__` derives `max_episodes` from `max_env_steps`). So the override
is now LOUD: `EpisodeCapWarning` at construction, plus
`rollout_episode_cap_can_bind` on every cell's manifest row. It is a warning and
not a ban because a tight episode cap is sometimes genuinely wanted -- declare
it with `allow_tight_episode_cap=True`.

Realised counts (`n_env_steps`, `n_episodes`, per-readout sample counts) are
returned for recording in the manifest. Per-cell `n_env_steps` is exactly what
made the original diagnosis possible without a re-run (autopsy learning 7) --
preserve it.

WHAT THIS MODULE DOES NOT DO
----------------------------
This is EXPERIMENT-HARNESS code, not a substrate change. It does not touch
`ree_core`. It does not decide criteria, guard DV saturation, or aggregate across
seeds -- those are per-probe concerns (autopsy learnings 2 and 3). It only
governs how long a cell runs and what it reports about its own sample yield.

ASCII-only in all printed output (project rule).

Run the contract tests: pytest tests/contracts/test_sample_driven_rollout.py -q
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "TickContext",
    "RolloutBudget",
    "RolloutOutcome",
    "EpisodeCapWarning",
    "run_cell_until_samples",
    "starvation_selfroute",
    "SELF_ROUTE_SAMPLE_STARVATION",
]


class EpisodeCapWarning(UserWarning):
    """An explicit `max_episodes` that can bind before `max_env_steps`.

    Raised at `RolloutBudget` construction, NOT at stop time -- by stop time the
    starved cell has already been produced. See the "DENOMINATION" note on
    `RolloutBudget.max_episodes` for why this is a warning rather than an error.
    """

# The self-route label a caller should emit when a readiness precondition is
# unmet because of SAMPLE COUNT rather than a substrate CAPABILITY check.
# Autopsy learning 4: V3-EXQ-779 emitted `substrate_not_ready_requeue` and sent
# readers hunting for a missing SD-069 capability that was present and firing.
SELF_ROUTE_SAMPLE_STARVATION = "sample_starvation_requeue"

# Stop reasons (also the vocabulary recorded in the manifest).
STOP_FLOORS_MET = "floors_met"
STOP_STEP_CAP = "step_cap"
STOP_EPISODE_CAP = "episode_cap"


@dataclass
class TickContext:
    """What the caller's `observe` callback sees for one env step.

    Attributes
    ----------
    step_result
        The `StepResult` returned by `StepHarness.step()` this tick.
    probs
        `agent.e3.last_precommit_probs` as read AFTER this tick, or None.
    fresh
        True when `probs` is a NEW tensor object relative to the previous tick
        of this episode -- i.e. E3 actually ran a selection rather than the
        previous distribution being left in place. This is the identity-based
        freshness test both 777 and 779 implemented inline, byte-identically.
    episode_index
        0-based index of the episode currently running.
    step_in_episode
        0-based index of this step within the current episode.
    n_env_steps
        Total env steps taken in this cell so far, INCLUDING this one.
    """

    step_result: Any
    probs: Any
    fresh: bool
    episode_index: int
    step_in_episode: int
    n_env_steps: int


@dataclass(frozen=True)
class RolloutBudget:
    """Sample-driven budget for one (arm, seed) cell.

    Parameters
    ----------
    sample_floors
        Mapping of readout name -> required fresh sample count, e.g.
        `{"selections": 20}` or `{"selections": 20, "event_ticks": 10}`.
        The rollout continues until EVERY floor is met.

        IMPORTANT: floors must be declared PER CELL, not per experiment. A
        readout that is zero by construction on some arm (e.g. `event_ticks` on
        a PHASIC-OFF arm in V3-EXQ-779) must NOT carry a floor on that arm --
        it would burn the whole step cap and report false starvation. Pass
        `{}` (or omit that key) for arms where the readout cannot fire.
    max_env_steps
        Hard cap on total env steps for this cell. Bounds runtime so a
        pathological seed cannot run unbounded. Required, must be > 0.
    steps_per_episode
        Cap on steps within a single episode before a forced reset.
    max_episodes
        Cap on episodes. Defensive backstop for the degenerate case where the
        env terminates on step 1 every time -- without it, a 2-step episode
        against a large `max_env_steps` would spin through thousands of resets.
        Defaults to a generous bound derived from the step cap.

        DENOMINATION -- read this before passing an explicit value. Every episode
        consumes at least one env step, so `max_episodes >= max_env_steps`
        guarantees the STEP cap is the binding constraint for every cell. Any
        smaller explicit value re-opens the defect this module exists to close:
        V3-EXQ-779a made the stopping RULE sample-driven but left this CAP
        episode-denominated at 120 against a 2400-step cap, and seed 23 / arm
        T1P1 -- which dies in ~7 steps per episode -- spent only 835 of its 2400
        available steps and starved one layer down.

        This is a WARNING (`EpisodeCapWarning`), not an error: a caller may
        genuinely want a tight episode cap as a runtime bound on a probe whose
        floors it expects to reach quickly. Pass
        `allow_tight_episode_cap=True` to declare that intent and silence it.
        Either way the fact is recorded on the `RolloutOutcome` and reaches the
        manifest as `rollout_episode_cap_can_bind`, so a reader can see that a
        cell's yield was episode-denominated without re-deriving it.
    allow_tight_episode_cap
        Opt out of `EpisodeCapWarning`. Does NOT suppress the manifest flag.
    """

    sample_floors: Mapping[str, int] = field(default_factory=dict)
    max_env_steps: int = 0
    steps_per_episode: int = 300
    max_episodes: int = 0
    allow_tight_episode_cap: bool = False

    @property
    def episode_cap_can_bind(self) -> bool:
        """True when the EPISODE cap can bind before the STEP cap.

        Since an episode costs at least one step, this is exactly
        `max_episodes < max_env_steps`. False under the default, where
        `max_episodes` is derived from `max_env_steps`.
        """
        return int(self.max_episodes) < int(self.max_env_steps)

    def __post_init__(self) -> None:
        if int(self.max_env_steps) <= 0:
            raise ValueError("RolloutBudget.max_env_steps must be > 0")
        if int(self.steps_per_episode) <= 0:
            raise ValueError("RolloutBudget.steps_per_episode must be > 0")
        for name, floor in self.sample_floors.items():
            if int(floor) < 0:
                raise ValueError(
                    "RolloutBudget.sample_floors[%s] must be >= 0" % name
                )
        if int(self.max_episodes) <= 0:
            # Generous default: enough episodes to spend the step cap even when
            # episodes terminate almost immediately, but still finite. This is
            # the step-denominated form -- the episode cap can never bind first.
            object.__setattr__(self, "max_episodes", int(self.max_env_steps))
            return

        # An EXPLICIT episode cap below the step cap can starve a short-episode
        # seed before the step budget is spent. Say so at construction time.
        if self.episode_cap_can_bind and not bool(self.allow_tight_episode_cap):
            warnings.warn(
                "RolloutBudget: max_episodes=%d < max_env_steps=%d, so the "
                "EPISODE cap can bind before the STEP cap. A seed whose "
                "episodes are shorter than %.1f steps will spend less than its "
                "full step budget (the V3-EXQ-779a seed-23 defect: 835 of 2400 "
                "steps at ~7 steps/episode). Derive max_episodes from "
                "max_env_steps, or pass allow_tight_episode_cap=True if the "
                "tight cap is intended."
                % (
                    int(self.max_episodes),
                    int(self.max_env_steps),
                    float(self.max_env_steps) / float(self.max_episodes),
                ),
                EpisodeCapWarning,
                stacklevel=3,
            )


@dataclass
class RolloutOutcome:
    """Realised sample yield for one cell. Record this in the manifest."""

    n_env_steps: int
    n_episodes: int
    counts: Dict[str, int]
    floors: Dict[str, int]
    stop_reason: str
    # The budget this cell actually ran under. `stop_reason` says which cap DID
    # bind; these say which cap COULD have. The distinction matters for the
    # near-miss case -- a cell that stopped on `floors_met` under a tight
    # episode cap is one short-episode seed away from the 779a defect, and
    # stop_reason alone cannot show that.
    max_env_steps: int = 0
    max_episodes: int = 0
    episode_cap_can_bind: bool = False

    @property
    def starved_readouts(self) -> List[str]:
        """Readout names whose realised count fell short of their floor."""
        return sorted(
            name
            for name, floor in self.floors.items()
            if self.counts.get(name, 0) < int(floor)
        )

    @property
    def floors_met(self) -> bool:
        """True when every declared sample floor was reached."""
        return not self.starved_readouts

    def as_manifest_fields(self) -> Dict[str, Any]:
        """Flat dict to merge into a per-cell manifest row.

        `n_env_steps` is the field that made the 777/779 diagnosis possible
        without a re-run. Keep it.
        """
        out: Dict[str, Any] = {
            "n_env_steps": int(self.n_env_steps),
            "n_episodes": int(self.n_episodes),
            "rollout_stop_reason": self.stop_reason,
            "rollout_floors_met": bool(self.floors_met),
            "rollout_sample_floors": {k: int(v) for k, v in self.floors.items()},
            "rollout_sample_counts": {k: int(v) for k, v in self.counts.items()},
            "rollout_max_env_steps": int(self.max_env_steps),
            "rollout_max_episodes": int(self.max_episodes),
            "rollout_episode_cap_can_bind": bool(self.episode_cap_can_bind),
        }
        if not self.floors_met:
            out["rollout_starved_readouts"] = list(self.starved_readouts)
        return out

    def summary_line(self, label: str = "") -> str:
        """One-line ASCII progress/summary string."""
        counts = " ".join(
            "%s=%d/%d" % (k, self.counts.get(k, 0), int(v))
            for k, v in sorted(self.floors.items())
        )
        if not counts:
            counts = " ".join(
                "%s=%d" % (k, v) for k, v in sorted(self.counts.items())
            )
        head = ("%s " % label) if label else ""
        tail = " episode_cap_can_bind=1" if self.episode_cap_can_bind else ""
        return "%senv_steps=%d episodes=%d %s stop=%s%s" % (
            head,
            self.n_env_steps,
            self.n_episodes,
            counts,
            self.stop_reason,
            tail,
        )


def run_cell_until_samples(
    *,
    env: Any,
    agent: Any,
    harness: Any,
    budget: RolloutBudget,
    observe: Callable[[TickContext], Optional[Mapping[str, int]]],
    probs_attr: str = "last_precommit_probs",
    progress_label: str = "",
    progress_fn: Optional[Callable[[str], None]] = None,
) -> RolloutOutcome:
    """Run one (arm, seed) cell until its sample floors are met or the cap binds.

    Replaces the `for ep in range(N_EPISODES): for step in range(...)` +
    `if r.done: break` pattern, whose sample yield is set by episode survival
    rather than by the budget.

    Parameters
    ----------
    env, agent, harness
        A constructed env, `REEAgent`, and `StepHarness` for THIS cell. The
        helper resets all three at each episode boundary (the canonical order:
        `env.reset()` -> `agent.reset()` -> `harness.reset()`), but does not own
        their construction or teardown.
    budget
        `RolloutBudget` -- floors in the same unit as the probe's readiness
        gates, plus the hard step cap.
    observe
        Called once per env step with a `TickContext`. Returns a mapping of
        readout name -> increment for this tick (e.g. `{"selections": 1}`, or
        `{"selections": 1, "event_ticks": 1}`), or None/`{}` for a tick that
        yielded nothing. The callback owns all data accumulation; this helper
        only counts. Readout names that appear in `sample_floors` drive
        stopping; other names are still counted and returned.
    probs_attr
        Attribute on `agent.e3` holding the pre-commit distribution used for
        the freshness test. Overridable for probes reading a different readout.
    progress_label
        Prefix for per-episode progress lines, e.g. "T1S0 seed=11".
    progress_fn
        Sink for progress lines. Defaults to `print(..., flush=True)`. Output is
        ASCII-only.

    Returns
    -------
    RolloutOutcome
        Realised counts plus the stop reason. Check `.floors_met`; when False,
        `.starved_readouts` names what fell short and the caller should
        self-route `SELF_ROUTE_SAMPLE_STARVATION` naming this cell.
    """
    emit = progress_fn if progress_fn is not None else _default_progress

    floors = {str(k): int(v) for k, v in budget.sample_floors.items()}
    counts: Dict[str, int] = {name: 0 for name in floors}
    n_env_steps = 0
    n_episodes = 0
    stop_reason = STOP_EPISODE_CAP

    def _floors_met() -> bool:
        return all(counts.get(name, 0) >= floor for name, floor in floors.items())

    # An empty floor set means "no sample-driven target" -- run the step cap.
    while n_episodes < int(budget.max_episodes):
        if _floors_met() and floors:
            stop_reason = STOP_FLOORS_MET
            break
        if n_env_steps >= int(budget.max_env_steps):
            stop_reason = STOP_STEP_CAP
            break

        _flat, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        n_episodes += 1
        prev_probs_id: Optional[int] = None

        for step_in_episode in range(int(budget.steps_per_episode)):
            if n_env_steps >= int(budget.max_env_steps):
                stop_reason = STOP_STEP_CAP
                break

            r = harness.step(obs_dict)
            n_env_steps += 1

            e3 = getattr(agent, "e3", None)
            probs = getattr(e3, probs_attr, None) if e3 is not None else None
            pid = id(probs) if probs is not None else None
            fresh = probs is not None and pid != prev_probs_id
            prev_probs_id = pid

            inc = observe(
                TickContext(
                    step_result=r,
                    probs=probs,
                    fresh=fresh,
                    episode_index=n_episodes - 1,
                    step_in_episode=step_in_episode,
                    n_env_steps=n_env_steps,
                )
            )
            if inc:
                for name, delta in inc.items():
                    counts[str(name)] = counts.get(str(name), 0) + int(delta)

            obs_dict = r.next_obs_dict

            if floors and _floors_met():
                stop_reason = STOP_FLOORS_MET
                break
            if r.done:
                break
        else:
            # Episode ran to the per-episode step cap without terminating.
            if n_env_steps >= int(budget.max_env_steps):
                stop_reason = STOP_STEP_CAP

        if stop_reason in (STOP_FLOORS_MET, STOP_STEP_CAP):
            break

    outcome = RolloutOutcome(
        n_env_steps=n_env_steps,
        n_episodes=n_episodes,
        counts=counts,
        floors=floors,
        stop_reason=stop_reason,
        max_env_steps=int(budget.max_env_steps),
        max_episodes=int(budget.max_episodes),
        episode_cap_can_bind=bool(budget.episode_cap_can_bind),
    )
    emit("  [rollout] " + outcome.summary_line(progress_label))
    return outcome


def starvation_selfroute(
    cells: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build a `sample_starvation_requeue` self-route record, or None.

    Autopsy learning 4: a readiness gate that reports only a min-across-cells
    number ("measured 5, threshold 10") hides WHICH cell was starved and WHY.
    V3-EXQ-779's single 19-step cell (seed 23 / T1P1) vetoed the whole run while
    the substrate was working everywhere else, and the emitted label pointed at
    a substrate capability that was in fact present and firing.

    Parameters
    ----------
    cells
        One mapping per cell, each with:
          - "arm"     : arm label
          - "seed"    : seed
          - "outcome" : the cell's `RolloutOutcome`

    Returns
    -------
    dict or None
        None when every cell met its floors (nothing to route). Otherwise a
        record naming each offending cell, its shortfall per readout, and its
        realised step count -- suitable for dropping into the manifest's
        precondition record alongside `self_route`.
    """
    offenders: List[Dict[str, Any]] = []
    for cell in cells:
        outcome = cell.get("outcome")
        if not isinstance(outcome, RolloutOutcome) or outcome.floors_met:
            continue
        offenders.append(
            {
                "arm": cell.get("arm"),
                "seed": cell.get("seed"),
                "n_env_steps": int(outcome.n_env_steps),
                "n_episodes": int(outcome.n_episodes),
                "stop_reason": outcome.stop_reason,
                "episode_cap_can_bind": bool(outcome.episode_cap_can_bind),
                "max_env_steps": int(outcome.max_env_steps),
                "max_episodes": int(outcome.max_episodes),
                "shortfall": {
                    name: {
                        "measured": int(outcome.counts.get(name, 0)),
                        "threshold": int(outcome.floors[name]),
                    }
                    for name in outcome.starved_readouts
                },
            }
        )

    if not offenders:
        return None

    return {
        "self_route": SELF_ROUTE_SAMPLE_STARVATION,
        "reason": (
            "One or more cells hit the total-step cap before reaching their "
            "sample floors. This is a SAMPLING failure, not a substrate "
            "capability failure -- do not route to substrate_not_ready_requeue "
            "without an independent capability check."
        ),
        "n_starved_cells": len(offenders),
        "starved_cells": offenders,
    }


def _default_progress(line: str) -> None:
    print(line, flush=True)
