"""
Q-081 constrained-realisation surrogate null for unequal-rate multi-stream traces.

Why this module exists
----------------------
Q-081's non-degeneracy guard (claims.yaml, sharpened 2026-07-22 in REE_assembly
ca8e3d7fc8) requires that any cross-stream statistic be tested against a surrogate
ensemble built as a CONSTRAINED REALISATION -- one preserving, per stream:

  (a) the tick times / configured update period,
  (b) the marginal distribution of the stream's values,
  (c) the WITHIN-STREAM temporal autocorrelation,

and destroying ONLY the relation BETWEEN streams. The named construction is a block
permutation applied within each stream's own tick grid (Lancaster et al. 2018, Physics
Reports 748:1-60, the standard surrogate-construction reference).

THE SHUFFLE THIS REPLACES. `stream_recorder.rate_matched_shuffle_index()` permutes each
stream's FRESH samples among themselves and leaves held samples in place. That satisfies
(a) and (b) but NOT (c): permuting fresh samples individually destroys within-stream
serial correlation, which was never the property under test. A cross-stream statistic can
then clear the null purely because each stream is individually smooth in time -- a FALSE
Outcome A. `test_q081_surrogate_null.py::test_naive_shuffle_produces_the_false_positive`
demonstrates that failure on two provably UNCOUPLED streams and is the reason this module
exists. Do not use the fresh-only shuffle to adjudicate Q-081.

THE UNEQUAL-RATE PROBLEM, AND THE RULE ADOPTED
----------------------------------------------
Lancaster treats signals on a COMMON REGULAR GRID. REE's streams do not share one: E1
ticks every step, E2 every 3, E3 every 10 (SD-006; config.py e1/e2/e3_steps_per_tick).
The claim text flags this as "the step where an error is easiest to make and hardest to
notice", so the rule is stated explicitly here rather than left implicit in the code.

The tempting rule -- one block length L in ticks, applied to every stream -- is WRONG. At
L = 20 ticks, E1 blocks span 20 steps and E3 blocks span 200 steps. The streams are then
scrambled at different temporal granularities: each coarse block spans ten fine blocks, so
the coarse stream's large-scale temporal layout survives largely intact relative to the
fine stream's, and cross-stream alignment is destroyed unevenly across the pair. The
ensemble is not a null for the same hypothesis on every pair.

THE RULE: choose ONE common block DURATION W in STEPS -- the only unit the streams share
-- and give each stream a block length in its OWN tick units:

    L_s = max(1, round(W / period_s))

Blocks are then temporally commensurate across streams, and cross-stream alignment is
destroyed at the same timescale for every pair, while each stream is still permuted on
its own grid. A useful consequence: the block COUNT is then approximately equal across
streams (n_blocks_s ~ m_s / L_s ~ (n_steps/period_s) / (W/period_s) = n_steps / W), so
the ensemble has the same richness for the fine and the coarse stream. That equalisation
is the point of measuring W in steps.

CHOOSING W. Two constraints, from opposite directions:

  LOWER: W must exceed every stream's autocorrelation time expressed in STEPS, or the
  blocks are shorter than the correlation they are supposed to carry and property (c)
  fails for the slowest-decorrelating stream. Block-bootstrap practice uses a safety
  factor; the default here is 2.
      W >= safety * max_s (tau_s * period_s)
  W is additionally floored at max_s period_s so the coarsest stream gets L_s >= 1.

  UPPER: the block COUNT must stay large enough for the permutation ensemble to be rich.
      W <= n_steps / min_blocks           (default min_blocks = 8)

If those cross, the run is TOO SHORT to support a valid surrogate for its own
autocorrelation structure, and `plan_blocks()` RAISES with the n_steps that would be
needed. It does not silently shrink the blocks. A surrogate that quietly violates (c) is
exactly the undetectable error the claim warns about, and refusing is the only safe
failure mode.

PERIODS AND TAUS ARE MEASURED FROM THE DATA, NEVER FROM CONFIG. `agent.clock` resets the
E3 phase under MECH-091 and modulates the rate under MECH-093, so a nominal `% 10` is
wrong on real traces. Periods come from the recorded `<name>__fresh` flags and taus from
the fresh subsequence itself.

WHAT THE SURROGATE CANNOT DO (repeated here because it is load-bearing)
----------------------------------------------------------------------
Clearing this null is NECESSARY for Q-081 Outcome A and nowhere near sufficient. Wired
coordination is real coordination and will correctly clear any surrogate test. Only the
ablation series -- and in particular the structure-destroying (landmark-removal) arm --
separates Outcome A from Outcome B. This module builds the null; it does not adjudicate
the claim.

CROSS-STREAM LAG IS A CONTROL, NOT A READOUT. A lag between streams ticking at 1/3/10
steps is guaranteed by the scheduler, so lag is closer to an Outcome-B (clock) detector
than an Outcome-A one. `cross_stream_xcorr()` therefore returns the lag alongside the
statistic and `lag_control_report()` labels it `role="control"`: it must be present and
must not explain the result. Do not promote it to the primary readout.

A-PRIORI FILTER ON THE STATISTIC. Q-081 rules out, before any run, any candidate statistic
that is a function of the configured update rates. `screen_statistic()` implements that
filter mechanically rather than by inspection: a statistic that is constant across the
whole surrogate ensemble is by construction a function only of what the surrogate
PRESERVES (tick times, marginal, within-stream autocorrelation), so it cannot discriminate
and is ruled out. Run it on every candidate statistic before the statistic adjudicates
anything.

Input format
------------
The `arrays` mapping is the one `stream_recorder.StreamTraceRecorder.finalize()` writes
into the trace blob: for each stream `name`, a float array `name` of shape (n_steps, width)
plus boolean `name__fresh` and `name__valid` of shape (n_steps,). This module reads only
`name` and `name__fresh`.

ASCII-only output (repo rule). numpy + stdlib only; no torch, no ree_core.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SURROGATE_SCHEMA = "q081_surrogate/v1"

# Defaults. See "CHOOSING W" in the module docstring for what each one guards.
DEFAULT_SAFETY_FACTOR = 2.0
DEFAULT_MIN_BLOCKS = 8
DEFAULT_N_SURROGATES = 199
ACF_THRESHOLD = 1.0 / math.e

# Hold-mode classification (see classify_hold_mode).
HOLD_CARRY = "carry"
HOLD_FILLER = "filler"
HOLD_UNSTRUCTURED = "unstructured"
HOLD_NONE = "none"


class SurrogateDesignError(ValueError):
    """Raised when a valid constrained-realisation surrogate cannot be built.

    Always a refusal, never a downgrade. A surrogate that silently violates the
    within-stream-autocorrelation constraint is undetectable downstream.
    """


# --------------------------------------------------------------------------
# Measurement of the tick grid and the within-stream correlation time
# --------------------------------------------------------------------------


def estimate_tick_period(fresh: np.ndarray) -> float:
    """Steps per tick for one stream, MEASURED from its recorded freshness flags.

    Uses the median inter-fresh gap, which is robust to the MECH-091 phase reset and the
    MECH-093 rate modulation that both make a nominal `step % period` wrong. Returns 1.0
    for a stream fresh at every step, and inf for a stream that never goes fresh.
    """
    pos = np.flatnonzero(np.asarray(fresh, dtype=bool))
    if pos.size == 0:
        return float("inf")
    if pos.size == 1:
        return float(len(fresh))
    return float(np.median(np.diff(pos)))


def fresh_positions(fresh: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(fresh, dtype=bool))


def _acf_first_crossing(x: np.ndarray, max_lag: int) -> float:
    """Lag (in samples of x) at which the autocorrelation first drops below 1/e.

    First-crossing rather than an integrated estimate: it is robust to the noisy ACF tail
    of a short fresh subsequence, which for an E3 stream can be only a few hundred
    samples long. Returns 0.0 for a constant series (no correlation time to preserve).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return 0.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    max_lag = int(min(max_lag, n - 2))
    for k in range(1, max_lag + 1):
        r = float(np.dot(x[:-k], x[k:])) / denom
        if not np.isfinite(r) or r < ACF_THRESHOLD:
            return float(k)
    return float(max_lag)


def autocorrelation_time(values: np.ndarray, fresh: np.ndarray,
                         max_lag: Optional[int] = None) -> float:
    """Within-stream correlation time in TICKS OF THIS STREAM (not in steps).

    Computed on the fresh subsequence -- the stream's own grid -- because that is the
    grid the block permutation operates on. Held samples are excluded: they are repeats
    and would inflate tau by the hold length, i.e. by the update rate, reintroducing the
    rate dependence this whole apparatus is built to keep out of the analysis.

    A vector stream takes the MAX over its columns, which is the conservative choice: the
    block must be long enough for the slowest-decorrelating dimension.
    """
    vals = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if vals.shape[0] == 1 and np.ndim(values) == 1:
        vals = vals.T
    pos = fresh_positions(fresh)
    if pos.size < 4:
        return 0.0
    sub = vals[pos]
    if sub.ndim == 1:
        sub = sub.reshape(-1, 1)
    lag_cap = int(max_lag) if max_lag is not None else max(4, pos.size // 4)
    taus = [_acf_first_crossing(sub[:, j], lag_cap) for j in range(sub.shape[1])]
    finite = [t for t in taus if np.isfinite(t)]
    return float(max(finite)) if finite else 0.0


def classify_hold_mode(values: np.ndarray, fresh: np.ndarray) -> str:
    """How this stream's HELD (non-fresh) steps relate to its fresh ones.

    The permutation reorders fresh samples; the held steps then have to be rebuilt, and
    rebuilding them wrongly changes the full-series marginal. The recorder writes two
    genuinely different hold semantics, so the mode is measured rather than assumed:

      "carry"   -- a held row repeats the previous row. The recorder's dominant case:
                   the cache object was not reassigned, so the recorded value is
                   literally the previous one (e3_commitment, e3_scores, operating_mode).
      "filler"  -- every held row is the same constant row, unrelated to the preceding
                   fresh value. The event streams: a non-event step records [0, nan]
                   regardless of what the last event was.
      "none"    -- no held steps at all (fresh every step; z_world, z_self, ...).
      "unstructured" -- neither. `block_permute_stream` REFUSES these rather than guess.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    f = np.asarray(fresh, dtype=bool)
    held = np.flatnonzero(~f)
    held = held[held > 0]
    if held.size == 0:
        return HOLD_NONE

    def _rows_equal(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))

    if all(_rows_equal(vals[i], vals[i - 1]) for i in held):
        return HOLD_CARRY
    first = vals[held[0]]
    if all(_rows_equal(vals[i], first) for i in held):
        return HOLD_FILLER
    return HOLD_UNSTRUCTURED


# --------------------------------------------------------------------------
# The block plan: one common block DURATION in steps, per-stream lengths in ticks
# --------------------------------------------------------------------------


class BlockPlan:
    """The chosen block duration plus each stream's derived tick-grid block length.

    Self-describing on purpose: this object is what a manifest should record so that a
    reader can tell, years later, what null a p-value was computed against.
    """

    def __init__(self, block_duration_steps: int, n_steps: int,
                 streams: Dict[str, Dict[str, Any]], safety_factor: float,
                 min_blocks: int):
        self.block_duration_steps = int(block_duration_steps)
        self.n_steps = int(n_steps)
        self.streams = streams
        self.safety_factor = float(safety_factor)
        self.min_blocks = int(min_blocks)

    def block_len_ticks(self, name: str) -> int:
        return int(self.streams[name]["block_len_ticks"])

    def hold_mode(self, name: str) -> str:
        return str(self.streams[name]["hold_mode"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "surrogate_schema": SURROGATE_SCHEMA,
            "construction": "block_permutation_within_own_tick_grid",
            "block_duration_steps": self.block_duration_steps,
            "n_steps": self.n_steps,
            "safety_factor": self.safety_factor,
            "min_blocks": self.min_blocks,
            "preserved": [
                "tick_times", "marginal_of_fresh_samples", "within_stream_autocorrelation",
            ],
            "destroyed": ["between_stream_relation"],
            "streams": self.streams,
        }

    def __repr__(self) -> str:  # ASCII only
        return ("BlockPlan(W=%d steps, n_steps=%d, streams=%d)"
                % (self.block_duration_steps, self.n_steps, len(self.streams)))


def plan_blocks(
    arrays: Mapping[str, np.ndarray],
    stream_names: Sequence[str],
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    max_lag: Optional[int] = None,
) -> BlockPlan:
    """Choose the common block duration W (steps) and each stream's block length (ticks).

    Raises SurrogateDesignError if the lower bound (carry the autocorrelation) and the
    upper bound (keep the ensemble rich) cross -- i.e. the run is too short to support a
    valid null for its own correlation structure. The message names the n_steps that
    would be needed, so the caller can lengthen the run rather than weaken the null.
    """
    if not stream_names:
        raise SurrogateDesignError("no streams given")

    n_steps = int(len(np.asarray(arrays[f"{stream_names[0]}__fresh"])))
    per_stream: Dict[str, Dict[str, Any]] = {}
    tau_steps_max = 0.0
    period_max = 1.0

    for name in stream_names:
        try:
            values = np.asarray(arrays[name])
            fresh = np.asarray(arrays[f"{name}__fresh"], dtype=bool)
        except KeyError as exc:
            raise SurrogateDesignError(
                "stream '%s' missing from arrays (need '%s' and '%s__fresh')"
                % (name, name, name)) from exc
        if len(fresh) != n_steps:
            raise SurrogateDesignError(
                "stream '%s' has %d steps, expected %d" % (name, len(fresh), n_steps))

        period = estimate_tick_period(fresh)
        if not np.isfinite(period):
            raise SurrogateDesignError(
                "stream '%s' never goes fresh: it carries no samples to permute and "
                "cannot participate in a surrogate null" % name)
        tau_ticks = autocorrelation_time(values, fresh, max_lag=max_lag)
        tau_steps = tau_ticks * period
        n_fresh = int(fresh.sum())
        hold_mode = classify_hold_mode(values, fresh)

        per_stream[name] = {
            "period_steps": period,
            "n_fresh": n_fresh,
            "tau_ticks": tau_ticks,
            "tau_steps": tau_steps,
            "hold_mode": hold_mode,
        }
        tau_steps_max = max(tau_steps_max, tau_steps)
        period_max = max(period_max, period)

    lower = max(safety_factor * tau_steps_max, period_max)
    upper = n_steps / float(min_blocks)
    if lower > upper:
        raise SurrogateDesignError(
            "run too short for a valid constrained-realisation surrogate: the slowest "
            "stream needs blocks of at least %.1f steps (safety %.1f x tau_max %.1f "
            "steps) but %d steps only supports %.1f-step blocks at %d blocks minimum. "
            "Need at least %d steps. Refusing rather than shortening the blocks, which "
            "would silently break the within-stream autocorrelation the null must "
            "preserve."
            % (lower, safety_factor, tau_steps_max, n_steps, upper, min_blocks,
               int(math.ceil(lower * min_blocks))))

    W = int(max(1, round(lower)))
    for name, info in per_stream.items():
        L = max(1, int(round(W / info["period_steps"])))
        info["block_len_ticks"] = L
        info["block_len_steps"] = L * info["period_steps"]
        info["n_blocks"] = int(math.ceil(info["n_fresh"] / L)) if info["n_fresh"] else 0

    return BlockPlan(W, n_steps, per_stream, safety_factor, min_blocks)


# --------------------------------------------------------------------------
# The permutation itself
# --------------------------------------------------------------------------


def _block_permutation(m: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of range(m) built by permuting contiguous blocks of length block_len.

    A random CIRCULAR offset is applied before cutting, so the block boundaries fall in a
    different place in every ensemble member. Without it every surrogate cuts the series
    at the same points and the ensemble understates the null's variability.
    """
    if m <= 1 or block_len >= m:
        return np.arange(m)
    offset = int(rng.integers(0, m))
    order = (np.arange(m) + offset) % m
    n_blocks = int(math.ceil(m / block_len))
    blocks = [order[i * block_len:(i + 1) * block_len] for i in range(n_blocks)]
    return np.concatenate([blocks[j] for j in rng.permutation(n_blocks)])


def block_permute_stream(
    values: np.ndarray,
    fresh: np.ndarray,
    block_len_ticks: int,
    rng: np.random.Generator,
    hold_mode: Optional[str] = None,
) -> np.ndarray:
    """One stream's constrained-realisation surrogate.

    Permutes blocks of the FRESH subsequence and writes them back to the SAME fresh
    positions, then rebuilds the held steps according to `hold_mode`. Tick times are
    therefore preserved exactly, the fresh-sample marginal exactly, and the within-stream
    autocorrelation everywhere except at the block junctions.
    """
    vals = np.asarray(values)
    squeeze = vals.ndim == 1
    if squeeze:
        vals = vals.reshape(-1, 1)
    f = np.asarray(fresh, dtype=bool)
    if hold_mode is None:
        hold_mode = classify_hold_mode(vals, f)
    if hold_mode == HOLD_UNSTRUCTURED:
        raise SurrogateDesignError(
            "hold pattern is neither carry-forward nor a constant filler, so held steps "
            "cannot be rebuilt after permutation without changing the marginal. Refusing "
            "rather than guessing.")

    pos = fresh_positions(f)
    out = vals.copy()
    if pos.size > 1:
        perm = _block_permutation(pos.size, int(block_len_ticks), rng)
        out[pos] = vals[pos][perm]

    if hold_mode == HOLD_CARRY:
        # Rebuild held rows by carrying the NEW value forward, so a held run keeps
        # repeating whatever now occupies its fresh position. Leaving the original held
        # rows in place instead would put values next to fresh samples they never
        # followed, and would change the full-series marginal.
        #
        # Vectorised forward fill: running max of "index of the last fresh step at or
        # before i". Safe because that index is always <= i and every fresh row is
        # already final, so no row is ever built from a not-yet-written one. Rows before
        # the first fresh step get -1 and keep their original value.
        src = np.where(f, np.arange(out.shape[0]), -1)
        np.maximum.accumulate(src, out=src)
        filled = src >= 0
        out[filled] = out[src[filled]]
    # HOLD_FILLER / HOLD_NONE: held rows are a constant unrelated to the fresh value, so
    # they are already correct in the copy and must NOT be carried.

    return out.reshape(-1) if squeeze else out


def make_surrogate(
    arrays: Mapping[str, np.ndarray],
    stream_names: Sequence[str],
    plan: BlockPlan,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """One ensemble member: every named stream independently block-permuted.

    Returns a copy of `arrays` with the named value streams replaced. The `__fresh` and
    `__valid` companions and every unnamed array are passed through UNCHANGED -- the tick
    grid is a preserved property, not something the surrogate is entitled to touch.
    """
    out: Dict[str, np.ndarray] = dict(arrays)
    for name in stream_names:
        out[name] = block_permute_stream(
            arrays[name], arrays[f"{name}__fresh"],
            plan.block_len_ticks(name), rng, hold_mode=plan.hold_mode(name),
        )
    return out


def surrogate_ensemble(
    arrays: Mapping[str, np.ndarray],
    stream_names: Sequence[str],
    n_surrogates: int = DEFAULT_N_SURROGATES,
    seed: int = 0,
    plan: Optional[BlockPlan] = None,
    **plan_kwargs: Any,
) -> Iterable[Dict[str, np.ndarray]]:
    """Yield `n_surrogates` constrained realisations. Lazy: one member in memory at a time."""
    if plan is None:
        plan = plan_blocks(arrays, stream_names, **plan_kwargs)
    rng = np.random.default_rng(seed)
    for _ in range(int(n_surrogates)):
        yield make_surrogate(arrays, stream_names, plan, rng)


# --------------------------------------------------------------------------
# Testing a statistic against the ensemble
# --------------------------------------------------------------------------


def surrogate_p_value(observed: float, surrogate_values: Sequence[float],
                      tail: str = "upper") -> float:
    """Rank-based p-value with the observed value included in the reference set.

    The (1 + count) / (1 + n) form is the standard conservative estimator: it cannot
    return 0, which would claim more resolution than an ensemble of finite size has.
    """
    s = np.asarray([v for v in surrogate_values if np.isfinite(v)], dtype=np.float64)
    n = s.size
    if n == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "upper":
        count = int(np.sum(s >= observed))
    elif tail == "lower":
        count = int(np.sum(s <= observed))
    else:
        raise ValueError("tail must be 'upper' or 'lower'")
    return float((1 + count) / (1 + n))


def evaluate_against_null(
    arrays: Mapping[str, np.ndarray],
    stream_names: Sequence[str],
    statistic: Callable[[Mapping[str, np.ndarray]], float],
    n_surrogates: int = DEFAULT_N_SURROGATES,
    seed: int = 0,
    tail: str = "upper",
    plan: Optional[BlockPlan] = None,
    **plan_kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate `statistic` on the data and on the surrogate ensemble.

    The returned dict carries the block plan, so the p-value travels with a full
    description of the null it was computed against.
    """
    if plan is None:
        plan = plan_blocks(arrays, stream_names, **plan_kwargs)
    observed = float(statistic(arrays))
    surr = [float(statistic(m)) for m in
            surrogate_ensemble(arrays, stream_names, n_surrogates, seed, plan=plan)]
    finite = np.asarray([v for v in surr if np.isfinite(v)], dtype=np.float64)
    return {
        "surrogate_schema": SURROGATE_SCHEMA,
        "observed": observed,
        "p_value": surrogate_p_value(observed, surr, tail=tail),
        "tail": tail,
        "n_surrogates": len(surr),
        "surrogate_mean": float(finite.mean()) if finite.size else float("nan"),
        "surrogate_std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "surrogate_n_distinct": int(np.unique(np.round(finite, 12)).size),
        "plan": plan.as_dict(),
    }


def screen_statistic(
    arrays: Mapping[str, np.ndarray],
    stream_names: Sequence[str],
    statistic: Callable[[Mapping[str, np.ndarray]], float],
    n_surrogates: int = 32,
    seed: int = 0,
    plan: Optional[BlockPlan] = None,
    rel_tol: float = 1e-9,
    **plan_kwargs: Any,
) -> Dict[str, Any]:
    """The A-PRIORI FILTER, applied mechanically. Run this BEFORE a statistic adjudicates.

    Q-081 rules out any candidate statistic that is a function of the configured update
    rates. Rather than rely on inspection, this checks the property that makes such a
    statistic useless: a statistic CONSTANT across the whole surrogate ensemble is a
    function only of what the surrogate PRESERVES -- tick times, marginal, within-stream
    autocorrelation -- and therefore cannot carry information about the between-stream
    relation, which is the only thing the ensemble varies. Its p-value is meaningless
    however extreme its value looks.

    Verdict is "ruled_out" (constant across the ensemble; not admissible) or "admissible"
    (varies; the surrogate can discriminate it). Note that "admissible" is a necessary
    condition, not an endorsement of the statistic's scientific meaning.
    """
    if plan is None:
        plan = plan_blocks(arrays, stream_names, **plan_kwargs)
    observed = float(statistic(arrays))
    surr = np.asarray(
        [float(statistic(m)) for m in
         surrogate_ensemble(arrays, stream_names, n_surrogates, seed, plan=plan)],
        dtype=np.float64,
    )
    finite = surr[np.isfinite(surr)]
    scale = max(abs(observed), float(np.max(np.abs(finite))) if finite.size else 0.0, 1.0)
    spread = float(np.max(finite) - np.min(finite)) if finite.size else 0.0
    constant = spread <= rel_tol * scale
    return {
        "surrogate_schema": SURROGATE_SCHEMA,
        "verdict": "ruled_out" if constant else "admissible",
        "reason": ("statistic is constant across the surrogate ensemble, so it is a "
                   "function only of properties the surrogate preserves (tick times / "
                   "marginal / within-stream autocorrelation) and cannot detect a "
                   "between-stream relation"
                   if constant else
                   "statistic varies across the surrogate ensemble, so the ensemble can "
                   "discriminate it"),
        "observed": observed,
        "ensemble_spread": spread,
        "ensemble_n_distinct": int(np.unique(np.round(finite, 12)).size),
        "n_surrogates": int(surr.size),
    }


# --------------------------------------------------------------------------
# Reference cross-stream statistic + the lag CONTROL
# --------------------------------------------------------------------------


def reduce_stream(values: np.ndarray, mode: str = "first") -> np.ndarray:
    """Vector stream -> 1-D series for a scalar cross-stream statistic.

    "first" takes column 0, "norm" the L2 norm across columns, "mean" the column mean.
    Provided for convenience only -- Q-081's actual question is about multi-dimensional
    CONFIGURATIONS, and any reduction to 1-D discards exactly that. Use it for controls
    and diagnostics, not as the primary readout.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        return v
    if mode == "first":
        return v[:, 0]
    if mode == "norm":
        return np.linalg.norm(np.nan_to_num(v, nan=0.0), axis=1)
    if mode == "mean":
        return np.nanmean(v, axis=1)
    raise ValueError("mode must be 'first', 'norm' or 'mean'")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 4:
        return float("nan")
    x = a[ok] - a[ok].mean()
    y = b[ok] - b[ok].mean()
    dx = float(np.sqrt(np.dot(x, x)))
    dy = float(np.sqrt(np.dot(y, y)))
    if dx <= 0.0 or dy <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / (dx * dy))


def cross_stream_xcorr(
    arrays: Mapping[str, np.ndarray],
    name_a: str,
    name_b: str,
    max_lag_ticks: int = 8,
    reduce: str = "first",
) -> Tuple[float, int]:
    """max over lags of |correlation| between two streams, plus the argmax lag.

    Evaluated on the COARSER stream's tick grid -- the only grid on which both streams
    carry an independently-updated sample -- with the finer stream read at those same
    steps. Lags are swept in units of the coarse stream's ticks and reported in STEPS.

    Returns (statistic, lag_steps). The statistic is the readout; THE LAG IS A CONTROL
    QUANTITY (see lag_control_report and the module docstring) -- with streams ticking at
    1/3/10 steps a nonzero lag is guaranteed by the scheduler, so it is closer to an
    Outcome-B detector than evidence for Q-081.

    This statistic is not a function of the configured update rates: the grid it is
    evaluated on is, but its VALUE depends on the recorded samples. `screen_statistic`
    confirms that empirically rather than taking this paragraph's word for it.
    """
    fa = np.asarray(arrays[f"{name_a}__fresh"], dtype=bool)
    fb = np.asarray(arrays[f"{name_b}__fresh"], dtype=bool)
    pa = estimate_tick_period(fa)
    pb = estimate_tick_period(fb)
    coarse_fresh, coarse_period = (fb, pb) if pb >= pa else (fa, pa)

    grid = fresh_positions(coarse_fresh)
    if grid.size < 8:
        return float("nan"), 0
    a = reduce_stream(np.asarray(arrays[name_a]), reduce)[grid]
    b = reduce_stream(np.asarray(arrays[name_b]), reduce)[grid]

    best, best_lag = float("nan"), 0
    for k in range(-int(max_lag_ticks), int(max_lag_ticks) + 1):
        if k >= 0:
            r = _pearson(a[: a.size - k] if k else a, b[k:] if k else b)
        else:
            r = _pearson(a[-k:], b[: b.size + k])
        if np.isfinite(r) and (not np.isfinite(best) or abs(r) > best):
            best, best_lag = abs(r), k
    return float(best), int(round(best_lag * coarse_period))


def lag_control_report(
    arrays: Mapping[str, np.ndarray],
    name_a: str,
    name_b: str,
    max_lag_ticks: int = 8,
    reduce: str = "first",
) -> Dict[str, Any]:
    """Report cross-stream lag EXPLICITLY as a control quantity, not as a readout.

    Q-081 records lag as an Outcome-B (clock) detector: a lag between streams ticking at
    1/3/10 steps is produced by the scheduler with no shared organisation whatever. The
    reporting contract is that it must be PRESENT and must NOT EXPLAIN the result. The
    `scheduler_expected_lag_steps` field is the amount attributable to the rate offset
    alone, so a reader can see at once whether the measured lag is anything more.
    """
    stat, lag_steps = cross_stream_xcorr(arrays, name_a, name_b, max_lag_ticks, reduce)
    pa = estimate_tick_period(np.asarray(arrays[f"{name_a}__fresh"], dtype=bool))
    pb = estimate_tick_period(np.asarray(arrays[f"{name_b}__fresh"], dtype=bool))
    return {
        "role": "control",
        "interpretation": ("Outcome-B (clock) detector. Must be present and must not "
                           "explain the result. NOT evidence for Q-081 Outcome A."),
        "pair": [name_a, name_b],
        "statistic_at_argmax": stat,
        "lag_steps": lag_steps,
        "period_steps": {name_a: pa, name_b: pb},
        "scheduler_expected_lag_steps": float(max(pa, pb) - min(pa, pb)),
    }


def artefactual_rate_statistic(arrays: Mapping[str, np.ndarray],
                               name_a: str, name_b: str) -> float:
    """A DELIBERATELY ARTEFACTUAL statistic: a pure function of the two update periods.

    Exists to validate the null, per Q-081's "validate the null before using it"
    requirement. The surrogate preserves every stream's freshness flags exactly, so this
    statistic is bit-identical on every ensemble member and the null kills it by
    construction: `screen_statistic` returns "ruled_out" and the p-value is 1.0.

    This is the reference case for the a-priori filter -- any candidate statistic that
    behaves like this one is a rate readout wearing a coupling costume.
    """
    pa = estimate_tick_period(np.asarray(arrays[f"{name_a}__fresh"], dtype=bool))
    pb = estimate_tick_period(np.asarray(arrays[f"{name_b}__fresh"], dtype=bool))
    lo, hi = min(pa, pb), max(pa, pb)
    return float(hi / lo) if lo > 0 else float("nan")
