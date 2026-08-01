"""Standing default eval-phase episode/step budget for the INV-091 cross-stream-
similarity-band driver family (v3_exq_827 -> v3_exq_827a -> v3_exq_828 -> v3_exq_828a
lineage).

WHY THIS MODULE EXISTS
-----------------------
All three runs to date were executed with WARMUP_EPISODES=40, EVAL_EPISODES=10,
STEPS_PER_EPISODE=150 (1500 eval-phase steps per arm), and every one of them found
`null_validation.checked=False` in its manifest: the constrained-realisation
surrogate null (`q081_surrogate.plan_blocks`, this package) requires the eval-phase
array to support blocks of at least `safety_factor * tau_max` steps (see
`q081_surrogate.py` "CHOOSING W"), and none of the three runs supplied enough steps
for that bound to clear. The recorded deficits, read from each manifest's
`null_validation.reason`:

    827  (pre phase-sync fix):        1500 supplied,  12000 needed
         -- period_max-dominated; driven by the un-fixed lockstep clock-collapse
         bug that 827a's redesign fixed (see 827a's module docstring). Not used
         as the sizing basis below -- it is a bug artefact, not a property of a
         correctly-built driver.
    827a (phase-sync redesign):       1099 supplied,   2576 needed  (tau_max=161)
    828  (remaining-ablations sweep): 1500 supplied, 2464-2848 needed (tau_max=178,
         worst of 6 arms)

Registered as `substrate_queue.json` sd_id `INV091-NULL-VALIDATION-RUN-LENGTH`
("bump the default run length for this driver family to >=2848 steps as a
standing default so future runs both compare arms AND establish significance
against chance"). 2848 -- the worst POST-phase-sync-fix deficit observed (828)
-- was the sizing basis for the FIRST fix below; see AMENDMENT 2 for why a second,
static-only fix based on that number also failed.

WHAT THIS DOES AND DOES NOT CHANGE
-----------------------------------
This bumps the CALLER's run length so the existing q081_surrogate null-validation
threshold (`DEFAULT_SAFETY_FACTOR`, `DEFAULT_MIN_BLOCKS` in `q081_surrogate.py`,
both untouched here) can actually be met. It does not touch `q081_surrogate.py`,
any `ree_core/` substrate, or any already-run experiment's recorded result --
827/827a/828/828a stay exactly as they were reviewed and scored in claims.yaml.

AMENDMENT 1 (2026-07-31, IGW-20260730-192): bumped EVAL_EPISODES 10 -> 24 (1500 ->
3600 eval-phase steps), sized to clear 828's own 2848-step worst-case deficit with
26% margin. See "AMENDMENT 2" immediately below for why this undershot.

AMENDMENT 2 (2026-08-01, failure_autopsy_V3-EXQ-828a_2026-08-01, SECOND consecutive
miscalibration): V3-EXQ-828a re-ran the SAME six-arm design at the AMENDMENT-1
budget (3600 eval-phase steps) and `null_validation.checked=False` AGAIN --
828a's actual worst-case arm/seed cell needed 14,648 steps, roughly 5x the
2848-step deficit AMENDMENT 1 was sized against, and roughly 4x the 3600-step
budget it supplied. The full per-(arm, seed) breakdown, read from 828a's manifest
`surrogate_p_values[*][*].error` (18 cells: 6 arms x 3 seeds; `supplied` is the
n_steps that ACTUALLY landed in the eval-phase trace for that cell, which is
frequently far below the nominal eval_episodes*steps_per_episode=3600 because
individual episodes can terminate EARLY on a hazard death -- `if result.done:
break` in the driver's `_eval_pass` loop):

    seed    tau_steps_max   supplied   needed (=ceil(lower*min_blocks))
    0            44              761-1062        6088-8496
    1           183-193         1684-2721        2928-3088
    2            53-56          1462-1831       11696-14648   <- worst cell

Two things this table makes explicit, both of which the AMENDMENT-1 fix missed by
construction (it derived a single static number from a single prior run's single
worst arm):

  1. TAU_MAX VARIES BY SEED, substantially -- seed-2's ACF-first-crossing tau
     (53-56 steps) is not the extreme value; seed-1's is smaller in absolute
     tau (183-193) but its stream's PERIOD is smaller too, so seed-1 clears at
     a much shorter n_steps than seed-2 needs. A margin computed from one seed's
     (or one run's) worst arm has no reason to bound a DIFFERENT seed's or a
     DIFFERENT run's worst arm -- confirmed twice now.
  2. THE BINDING TERM IS OFTEN `period_max`, NOT `safety_factor * tau_max`.
     `q081_surrogate.plan_blocks`'s bound is
     `lower = max(safety_factor * tau_steps_max, period_max)`. In the worst 828a
     cell (seed 2), `2 * 56 = 112` is nowhere near the recorded `lower=1831` --
     `period_max` (the estimated recording period of the SLOWEST-FIRING stream,
     e.g. a commitment-style stream that only goes "fresh" occasionally) is what
     actually dominates. `estimate_tick_period` measures this from the recorded
     `__fresh` flags, so a stream that fires only once or twice in a short window
     reports a period close to the WHOLE window length -- which is itself a
     right-censored (underestimated) reading. A static margin keyed only to a
     "tau_max" reading, rather than the full `plan_blocks` bound, was never going
     to track this term.

Recommendation (per the autopsy's routing): "a per-run dynamic pilot pass that
measures actual tau_max before committing to a fixed eval-phase budget" -- see
`compute_dynamic_eval_budget` below.

WHY A DEDICATED PILOT CHUNK, NOT A REUSE OF THE WARMUP PHASE
--------------------------------------------------------------
Checked directly against the actual driver
(`experiments/v3_exq_828a_inv091_cross_stream_similarity_band_null_validated.py`)
before choosing this, rather than assumed: the WARMUP phase in this driver family
runs through `_lib.goal_pipeline_tier1.warmup_train`, a completely separate,
UNTRACED code path -- it never touches a `StreamTraceRecorder` / `TraceStore` and
produces no `arrays` mapping of the shape `plan_blocks` / `autocorrelation_time`
consume (`name` + `name__fresh`). Only the EVAL phase (`_eval_pass`, per-arm) is
traced. So there is no existing warmup array to repurpose -- a per-run pilot has to
be a short DEDICATED chunk run through the same traced eval-phase code path
(`_eval_pass` with a small `n_episodes`) before the real, dynamically-sized eval
pass for that seed/arm. This module intentionally does not implement that
driver-side loop (it belongs in the driver script, which this module does not own)
-- it implements the SIZING computation the driver calls once it has pilot arrays
in hand.

USAGE (static defaults, unchanged; still the correct import for a driver that
prefers a fixed budget, or as a floor under the dynamic path below):

    from experiments._lib.inv091_driver_defaults import (
        INV091_WARMUP_EPISODES, INV091_EVAL_EPISODES, INV091_STEPS_PER_EPISODE,
        INV091_MIN_EVAL_STEPS_REQUIRED,
    )

USAGE (dynamic pilot sizing -- the recommended path for any NEW driver in this
family; PILOT -> MEASURE -> COMPUTE -> RUN):

    from experiments._lib.inv091_driver_defaults import (
        INV091_WARMUP_EPISODES, INV091_STEPS_PER_EPISODE,
        INV091_PILOT_EPISODES, compute_dynamic_eval_budget,
    )

    # 1. PILOT: after warmup, run a SHORT chunk of the intact eval phase through the
    #    same traced code path (_eval_pass or equivalent) -- e.g. INV091_PILOT_EPISODES
    #    episodes at STEPS_PER_EPISODE each. Keep the returned `arrays` mapping.
    pilot_arrays = _eval_pass(agent, env, "intact", seed,
                              INV091_PILOT_EPISODES, INV091_STEPS_PER_EPISODE,
                              store)["_arrays"]

    # 2/3. MEASURE + COMPUTE: one call, reusing q081_surrogate.autocorrelation_time
    #    internally -- do not re-derive tau by hand.
    required_steps = compute_dynamic_eval_budget(
        pilot_arrays, SIMILARITY_STREAM_NAMES,
    )

    # 4. RUN: convert to an eval-phase episode count and run the REAL eval pass at
    #    that size. Do NOT divide by the nominal STEPS_PER_EPISODE -- episodes can
    #    terminate early (hazard death), so the nominal length is only an UPPER
    #    bound on steps actually recorded per episode (see AMENDMENT 2's table:
    #    supplied n_steps ranged 470-2721 against a 3600-step NOMINAL budget in the
    #    very run that motivated this fix). Divide by the SHORTEST per-episode
    #    length actually observed during the pilot (or warmup, if tracked) instead,
    #    or -- more robust still -- keep running eval episodes in a loop and stop
    #    once the recorded trace array length reaches `required_steps`, rather than
    #    fixing an episode count up front.
    eval_episodes = math.ceil(required_steps / min_observed_steps_per_episode)

`dry_run` paths in existing scripts (DRY_WARMUP_EPISODES=2 / DRY_EVAL_EPISODES=2 /
DRY_STEPS_PER_EPISODE=20) are unaffected either way -- a smoke test does not need to
clear the null-validation bound, only to confirm the arms execute.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from experiments._lib import q081_surrogate as surro

# ---------------------------------------------------------------------------
# Static defaults (AMENDMENT 1, 2026-07-31) -- kept as a documented FALLBACK/
# FLOOR under the dynamic path (AMENDMENT 2), not deleted. A driver that has no
# pilot data at all (or whose dynamic estimate comes back below this) still gets
# at least this much run length.
# ---------------------------------------------------------------------------

INV091_WARMUP_EPISODES = 40            # unchanged from 827/827a/828/828a
INV091_EVAL_EPISODES = 24              # was 10 in 827/827a/828
INV091_STEPS_PER_EPISODE = 150         # unchanged from 827/827a/828/828a

# Worst POST-phase-sync-fix deficit recorded across 827a/828 (see module
# docstring AMENDMENT 1). The static default above clears this with margin, and
# it remains the FLOOR that compute_dynamic_eval_budget() will never go below --
# confirmed insufficient ON ITS OWN by 828a (AMENDMENT 2), but still a real,
# recorded lower bound worth keeping as a sanity net under a dynamic estimate
# built from a possibly-noisy short pilot.
INV091_MIN_EVAL_STEPS_REQUIRED = 2848

INV091_EVAL_STEPS_TOTAL = INV091_EVAL_EPISODES * INV091_STEPS_PER_EPISODE

assert INV091_EVAL_STEPS_TOTAL >= INV091_MIN_EVAL_STEPS_REQUIRED, (
    "INV091_EVAL_EPISODES * INV091_STEPS_PER_EPISODE (%d) fell below the "
    "recorded worst-case null-validation requirement (%d) -- see this module's "
    "docstring before lowering either constant."
    % (INV091_EVAL_STEPS_TOTAL, INV091_MIN_EVAL_STEPS_REQUIRED)
)

# ---------------------------------------------------------------------------
# Dynamic pilot sizing (AMENDMENT 2, 2026-08-01)
# ---------------------------------------------------------------------------

# How many eval-phase episodes a driver should run for the PILOT chunk before
# calling compute_dynamic_eval_budget(). Independent of INV091_EVAL_EPISODES --
# the pilot only needs enough fresh samples per stream to measure period/tau, not
# enough steps to itself clear plan_blocks. 6 episodes at STEPS_PER_EPISODE=150
# is up to 900 steps (less if episodes end early) -- cheap relative to the
# eval-phase budgets this module has been sizing (2848-14648 steps).
INV091_PILOT_EPISODES = 6

# margin_multiplier default for compute_dynamic_eval_budget(): applied ON TOP of
# the bare plan_blocks-clearing minimum (safety_factor * tau_max, or period_max,
# whichever binds -- see AMENDMENT 2). This is now the THIRD sizing attempt for
# this driver family, so shipping a bound that JUST clears plan_blocks with zero
# margin repeats the same mistake in miniature: a slightly-larger tau_max or
# period_max on the REAL eval run (which is longer than the pilot, and is a
# DIFFERENT sample from the same seed's stochastic process) would undershoot
# again. 2.0x is chosen over the autopsy's fallback-alternative "3-5x headroom"
# figure because that fallback was proposed for a STATIC margin with no
# per-run measurement at all; a dynamic pilot measurement already removes the
# dominant error source (borrowing a bound from a DIFFERENT run/seed's worst
# case -- AMENDMENT 2 showed seed-to-seed tau varying up to ~4x and the binding
# term itself, period_max vs safety*tau_max, differing across seeds). 2.0x is
# headroom against the RESIDUAL risk that this run's own tau/period during the
# real eval phase modestly exceeds what a short pilot measured, not against a
# wholesale wrong-run estimate. See INV091_UNRELIABLE_PILOT_EXTRA_MULTIPLIER for
# the case where even that residual risk is elevated.
INV091_DYNAMIC_MARGIN_MULTIPLIER = 2.0

# Extra multiplier (STACKS with INV091_DYNAMIC_MARGIN_MULTIPLIER) applied when the
# pilot itself could not reliably measure at least one stream -- either the stream
# never went fresh in the pilot window at all, or it went fresh too few times for
# a trustworthy period/tau reading, or its ACF-first-crossing estimate SATURATED
# at the lag cap (see `_measure_stream` below). A saturated or under-sampled
# reading is RIGHT-CENSORED -- it can only under-report tau/period, never
# over-report -- which is exactly the failure mode that produced AMENDMENT 2's
# rare-firing-stream `period_max` blind spot in the first place, now on the
# PILOT side rather than the sizing-basis-run side. Combined with the base
# margin (2.0 * 1.5 = 3.0x), this lands in the autopsy's own "3-5x headroom"
# fallback range for exactly the case that fallback was meant to cover: a
# measurement good enough to act on cannot actually be trusted.
INV091_UNRELIABLE_PILOT_EXTRA_MULTIPLIER = 1.5

# A stream needs at least this many fresh (recorded) samples in the PILOT window
# for its period/tau reading to be treated as reliable. `autocorrelation_time`
# itself only requires >=4 samples to return a non-zero estimate, but at exactly
# that floor the estimate is noisy and, per AMENDMENT 2's table, streams that
# fire this rarely are also the ones whose PERIOD estimate is most likely to be
# right-censored by a short pilot window. Set well above the bare functional
# minimum for that reason.
INV091_PILOT_MIN_FRESH_SAMPLES = 8


class PilotTooShortError(ValueError):
    """Raised when NO stream in the pilot could be measured at all (every named
    stream was never fresh in the pilot window), so compute_dynamic_eval_budget
    would have nothing but the static floor to fall back on. Mirrors
    q081_surrogate.SurrogateDesignError's refuse-rather-than-silently-guess
    philosophy (explicitly called out in failure_autopsy_V3-EXQ-828a_2026-08-01
    as a property worth preserving): a pilot that measured NOTHING is a driver
    bug (wrong stream names, pilot too short even for the fastest stream, wiring
    error), not a case where widening the margin further would help."""


def _measure_stream(
    name: str,
    values: np.ndarray,
    fresh: np.ndarray,
    max_lag: Optional[int],
    min_pilot_fresh_samples: int,
) -> Optional[Dict[str, Any]]:
    """One stream's (period, tau_ticks, tau_steps) reading from pilot data, plus a
    `reliable` verdict. Returns None if the stream was never fresh in the pilot at
    all (cannot inform sizing). Reuses q081_surrogate.estimate_tick_period /
    autocorrelation_time / fresh_positions -- does not re-derive autocorrelation
    estimation."""
    fresh = np.asarray(fresh, dtype=bool)
    pos = surro.fresh_positions(fresh)
    if pos.size == 0:
        return None
    period = surro.estimate_tick_period(fresh)
    if not np.isfinite(period):
        return None

    lag_cap = int(max_lag) if max_lag is not None else max(4, int(pos.size) // 4)
    tau_ticks = surro.autocorrelation_time(values, fresh, max_lag=max_lag) if pos.size >= 4 else 0.0
    tau_steps = float(tau_ticks) * period

    # `autocorrelation_time` returns the lag cap itself when the ACF never crosses
    # the 1/e threshold inside the evaluated window -- i.e. the true tau could be
    # LONGER than what was measured. Treating a saturated reading as trustworthy
    # would silently repeat the right-censoring failure mode AMENDMENT 2 traced
    # `period_max` to, just for tau instead of period.
    saturated = pos.size >= 4 and tau_ticks >= lag_cap
    reliable = (int(pos.size) >= min_pilot_fresh_samples) and not saturated

    return {
        "period_steps": float(period),
        "tau_ticks": float(tau_ticks),
        "tau_steps": float(tau_steps),
        "n_fresh_in_pilot": int(pos.size),
        "saturated": bool(saturated),
        "reliable": bool(reliable),
    }


def compute_dynamic_eval_budget(
    pilot_arrays: Mapping[str, Any],
    stream_names: Sequence[str],
    *,
    safety_factor: float = surro.DEFAULT_SAFETY_FACTOR,
    min_blocks: int = surro.DEFAULT_MIN_BLOCKS,
    margin_multiplier: float = INV091_DYNAMIC_MARGIN_MULTIPLIER,
    unreliable_pilot_extra_multiplier: float = INV091_UNRELIABLE_PILOT_EXTRA_MULTIPLIER,
    min_pilot_fresh_samples: int = INV091_PILOT_MIN_FRESH_SAMPLES,
    max_lag: Optional[int] = None,
    floor_steps: int = INV091_MIN_EVAL_STEPS_REQUIRED,
    return_detail: bool = False,
) -> Any:
    """Required EVAL-PHASE STEP COUNT for the REAL eval pass, sized from a short
    PILOT pass's own measured tau_max / period_max, guaranteed to clear
    `q081_surrogate.plan_blocks` on data with the same correlation structure --
    with `margin_multiplier` headroom on top of the bare clearing minimum (see
    module docstring AMENDMENT 2 for why zero-margin sizing is not acceptable a
    third time).

    Mirrors `plan_blocks`'s own worst-case-over-streams logic exactly: for every
    named stream, measures `period_steps` (via `q081_surrogate.estimate_tick_period`)
    and `tau_steps` (via `q081_surrogate.autocorrelation_time`, tau_ticks * period),
    then takes

        lower = max(safety_factor * tau_steps_max, period_max)
        bare_minimum = lower * min_blocks

    -- exactly `plan_blocks`'s own `lower * min_blocks` bound (see that function's
    SurrogateDesignError message) -- so a real eval phase sized to
    `bare_minimum * margin_multiplier` steps is GUARANTEED to clear `plan_blocks`
    for data whose tau/period do not exceed what the pilot measured, with room to
    spare. `period_max`, not just `safety_factor * tau_steps_max`, is carried
    through deliberately: AMENDMENT 2 found `period_max` was the BINDING term in
    828a's worst cells, not tau.

    RELIABILITY. A stream that fires rarely in a short pilot yields a
    RIGHT-CENSORED (under-)estimate of its true period, and a stream whose
    autocorrelation never decays within the pilot's evaluated lag window yields a
    right-censored (under-)estimate of its true tau -- both are exactly the
    failure mode this function exists to avoid reintroducing, now on the pilot
    side. Any such stream (see `_measure_stream`) marks the WHOLE estimate
    unreliable and multiplies in `unreliable_pilot_extra_multiplier` on top of
    `margin_multiplier`. A stream that never went fresh in the pilot at all
    contributes nothing to `tau_steps_max` / `period_max` (there is nothing to
    measure) and also marks the estimate unreliable -- it does NOT raise by
    itself, since the pilot may simply be shorter than that stream's natural
    period; a caller that wants a hard failure in that case should inspect
    `return_detail=True`'s `per_stream` mapping.

    Raises `PilotTooShortError` only if NO named stream could be measured at all
    (the pilot informed nothing), since in that case widening the margin further
    would not help -- see that exception's docstring.

    Returns `max(dynamic_estimate, floor_steps)`, an int, unless
    `return_detail=True`, in which case a diagnostic dict is returned instead
    (suitable for folding into a manifest so a reader can see what the dynamic
    sizing decision was actually based on).
    """
    if not stream_names:
        raise ValueError("stream_names must be non-empty")

    tau_steps_max = 0.0
    period_max = 1.0
    per_stream: Dict[str, Any] = {}
    any_measured = False
    any_unreliable = False

    for name in stream_names:
        values = np.asarray(pilot_arrays[name])
        fresh = np.asarray(pilot_arrays[f"{name}__fresh"], dtype=bool)
        measured = _measure_stream(name, values, fresh, max_lag, min_pilot_fresh_samples)
        if measured is None:
            per_stream[name] = {"n_fresh_in_pilot": 0, "reliable": False}
            any_unreliable = True
            continue
        per_stream[name] = measured
        any_measured = True
        if not measured["reliable"]:
            any_unreliable = True
        tau_steps_max = max(tau_steps_max, measured["tau_steps"])
        period_max = max(period_max, measured["period_steps"])

    if not any_measured:
        raise PilotTooShortError(
            "none of the %d named stream(s) were ever fresh in the pilot window -- "
            "the pilot cannot inform eval-phase sizing at all (check stream names "
            "and pilot length, not just the margin). Streams: %s"
            % (len(stream_names), list(stream_names))
        )

    lower = max(safety_factor * tau_steps_max, period_max)
    bare_minimum = lower * float(min_blocks)
    effective_multiplier = float(margin_multiplier) * (
        float(unreliable_pilot_extra_multiplier) if any_unreliable else 1.0
    )
    dynamic_estimate = int(math.ceil(bare_minimum * effective_multiplier))
    required_steps = max(dynamic_estimate, int(floor_steps))

    if not return_detail:
        return required_steps

    return {
        "required_eval_steps": required_steps,
        "bare_plan_blocks_minimum_steps": int(math.ceil(bare_minimum)),
        "tau_steps_max": tau_steps_max,
        "period_max": period_max,
        "safety_factor": float(safety_factor),
        "min_blocks": int(min_blocks),
        "margin_multiplier": float(margin_multiplier),
        "unreliable_pilot_extra_multiplier_applied": bool(any_unreliable),
        "effective_multiplier": effective_multiplier,
        "floor_steps": int(floor_steps),
        "floor_applied": dynamic_estimate < int(floor_steps),
        "per_stream": per_stream,
    }
