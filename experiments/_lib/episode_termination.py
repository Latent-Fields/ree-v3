"""
Episode-termination recording -- did the run actually spend its step budget?

The defect
----------
A flat manifest records a CONFIGURED step budget (``config["n_steps"]``) and some
aggregate outcome, and nothing in between. It does not record how many steps the
episodes REALLY ran, or why they stopped. So a run whose episodes died on step 19 of a
configured 400 is, on the page, indistinguishable from one that spent the whole budget
-- same config block, same shape of metrics, merely worse numbers. Every downstream
reader (governance, a claim-synthesis pass, the next author reusing the harness) reads
those numbers as a measurement of the mechanism under test, when they are actually a
measurement of an episode that ended before the mechanism could express itself.

Confirmed 2026-08-03 (SD-094). V3-EXQ-884 configured 400 steps; its episodes terminated
at 32 / 19 / 90 steps on seeds 42 / 43 / 44, all via ``agent_health <= 0`` from
self-inflicted contamination in a config the author believed was hazard-free
(``num_hazards=0``; see the SD-094 note in ``ree_core/environment/causal_grid_world.py``).
Nothing in the manifest said so. Establishing it needed a live re-run of the experiment
inside a failure autopsy -- for a fact the run itself already knew at every step and
simply never wrote down.

What this records
-----------------
``episode_termination``: {n_episodes, steps_configured, steps_mean, steps_min,
steps_max, full_budget_frac, causes, truncated_frac} -- read from the per-step
``info["done_cause"]`` / ``info["episode_steps"]`` fields the env now always emits.

``full_budget_frac`` is the load-bearing one: 1.0 means every episode ran to its
configured budget, and anything below that is the amount of the run that measured a
shortened episode. ``causes`` is a count per distinct cause (``"health_depleted"``,
``"step_limit"``, ``""`` for an episode the driver cut off itself at its own budget
before the env terminated), which says WHY, and is what separates a scientifically
meaningful termination from an accident.

Why a counter and NOT a warning or a gate
-----------------------------------------
Same reasoning as ``z_goal_stream`` (see that module's header). Early termination is
often the legitimate measurement -- a survival experiment, a curriculum probe with a
deliberate lethal arm -- so a warning would be noisy, and a gate would refuse real
science. A recorded fraction is simply correct in both directions and lands in the
run's own record whether or not anyone pre-registered a criterion for it.

Not in ``ALWAYS_CORE_KEYS``, for the same reason as ``substrate_stable_across_run`` /
``arm_knobs_effective`` / ``z_goal_stream``: the legacy corpus cannot carry it, and
making it core would turn every pre-2026-08-03 manifest into a WARN. The block is
OMITTED rather than zero-filled when a driver does not collect it, so its PRESENCE
always means the run measured this.

Recording it from a driver
--------------------------
Accumulate one entry per episode as the run proceeds, then pass the accumulator (or its
``stats()``) to ``write_flat_manifest`` / ``stamp_recording_core``::

    from experiments._lib.episode_termination import EpisodeTerminationAccumulator

    acc = EpisodeTerminationAccumulator(steps_configured=N_STEPS)
    for seed in SEEDS:
        env.reset()
        for t in range(N_STEPS):
            _, _, done, info, _ = env.step(action)
            if done:
                break
        acc.record_from_info(info)          # reads done_cause + episode_steps
    ...
    write_flat_manifest(..., episode_termination=acc)

``record(steps=..., cause=...)`` is the explicit form for a driver whose loop does not
keep the last ``info`` around, or one stepping an env that does not emit these fields.

ASCII-only output (repo rule). Stdlib only, so this module keeps the no-torch /
no-ree_core import guarantee the rest of ``_lib``'s stamping chain relies on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

MANIFEST_KEY = "episode_termination"

# The env field names this reads. Emitted unconditionally by
# ree_core/environment/causal_grid_world.py step() as of SD-094.
_STEPS_FIELD = "episode_steps"
_CAUSE_FIELD = "done_cause"

# Recorded for an episode the DRIVER ended at its own budget before the env
# terminated -- info["done_cause"] is "" in that case, which is not a cause and
# must not be counted as one.
_DRIVER_BUDGET_CAUSE = "driver_budget"


def stats_from_episodes(
    episodes: Any,
    steps_configured: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Summarise a sequence of (steps, cause) pairs into the manifest block.

    `episodes` is an iterable of (steps, cause) tuples. Returns None for an empty
    or unusable input, so the caller omits the block rather than stamping a
    zero-filled one that would read as a real measurement.
    """
    try:
        pairs: List[Any] = list(episodes or [])
    except Exception:
        return None
    if not pairs:
        return None

    steps: List[int] = []
    causes: Dict[str, int] = {}
    for item in pairs:
        try:
            raw_steps, raw_cause = item
            n = int(raw_steps)
        except Exception:
            continue
        steps.append(n)
        cause = str(raw_cause or "") or _DRIVER_BUDGET_CAUSE
        causes[cause] = causes.get(cause, 0) + 1

    if not steps:
        return None

    n_ep = len(steps)
    block: Dict[str, Any] = {
        "n_episodes": n_ep,
        "steps_mean": float(sum(steps)) / float(n_ep),
        "steps_min": int(min(steps)),
        "steps_max": int(max(steps)),
        "causes": dict(sorted(causes.items())),
    }

    if steps_configured is None:
        # Without a declared budget there is nothing to be short OF. Report the
        # distribution and say so, rather than inventing a denominator (using
        # max(steps) would make every run look like it spent its budget).
        block["steps_configured"] = None
        block["full_budget_frac"] = None
        block["truncated_frac"] = None
        return block

    budget = int(steps_configured)
    block["steps_configured"] = budget
    if budget <= 0:
        block["full_budget_frac"] = None
        block["truncated_frac"] = None
        return block
    n_full = sum(1 for s in steps if s >= budget)
    block["full_budget_frac"] = float(n_full) / float(n_ep)
    block["truncated_frac"] = float(n_ep - n_full) / float(n_ep)
    return block


class EpisodeTerminationAccumulator:
    """Collects one (steps, cause) entry per episode across a whole run.

    Mirrors ZGoalStreamAccumulator: the usual driver shape builds a fresh env per
    (arm, seed) cell inside a helper, so there is no run-level object to read the
    totals off at manifest time. Holding two small scalars per episode keeps every
    env and agent collectable, which a list of envs would not.
    """

    def __init__(self, steps_configured: Optional[int] = None) -> None:
        self.steps_configured = (
            int(steps_configured) if steps_configured is not None else None
        )
        self.episodes: List[Any] = []

    def record(self, steps: Any, cause: Any = "") -> "EpisodeTerminationAccumulator":
        """Explicit form: record one episode's length and termination cause."""
        try:
            self.episodes.append((int(steps), str(cause or "")))
        except Exception:
            pass
        return self

    def record_from_info(
        self, info: Optional[Mapping[str, Any]]
    ) -> "EpisodeTerminationAccumulator":
        """Record from the LAST `info` dict of an episode.

        Reads info["episode_steps"] and info["done_cause"]. A no-op when either is
        absent -- an env predating SD-094, or a driver passing something else --
        so the block is omitted rather than silently wrong.
        """
        try:
            if not info or _STEPS_FIELD not in info:
                return self
            self.record(info[_STEPS_FIELD], info.get(_CAUSE_FIELD, ""))
        except Exception:
            pass
        return self

    def stats(self) -> Optional[Dict[str, Any]]:
        return stats_from_episodes(self.episodes, self.steps_configured)


def _coerce_stats(source: Any) -> Optional[Dict[str, Any]]:
    """Accept an accumulator, a precomputed block, or a sequence of pairs."""
    if source is None:
        return None
    if isinstance(source, EpisodeTerminationAccumulator):
        return source.stats()
    if isinstance(source, Mapping):
        return dict(source) or None
    return stats_from_episodes(source, None)


def stamp_episode_termination(
    manifest: Dict[str, Any],
    source: Any = None,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Merge the `episode_termination` block onto `manifest` in place; return it.

    NO-OP-SAFE and additive, matching `stamp_recording_core`'s posture: a block the
    script set deliberately wins unless `overwrite=True`. `source` may be an
    EpisodeTerminationAccumulator, a precomputed block, or a sequence of
    (steps, cause) pairs.

    Never raises: provenance stamping must not be able to crash an experiment at
    manifest-write time, when the compute is already spent.
    """
    try:
        if not isinstance(manifest, dict):
            return manifest
        if not overwrite and manifest.get(MANIFEST_KEY):
            return manifest
        block = _coerce_stats(source)
        if block:
            manifest[MANIFEST_KEY] = block
    except Exception:
        pass
    return manifest


__all__ = [
    "MANIFEST_KEY",
    "EpisodeTerminationAccumulator",
    "stats_from_episodes",
    "stamp_episode_termination",
]
