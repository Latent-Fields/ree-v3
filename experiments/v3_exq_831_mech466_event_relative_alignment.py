#!/opt/local/bin/python3
"""V3-EXQ-831 -- MECH-466: does EVENT-RELATIVE cross-stream coordination
exceed CLOCK-RELATIVE coordination? (Outcome E of the Q-081 taxonomy.)

This is the FIRST MECH-466 run. It is the direct sibling of V3-EXQ-824
(Q-081 Outcome A vs B), which explicitly DEFERRED Outcome E ("event-
relative beats clock-relative") to a lettered follow-up once the primary
A-vs-B result was banked. This run picks that up.

No SLEEP DRIVER: use_sleep_loop is deliberately left False (scope: WAKING-
stream event grammar only, matching V3-EXQ-824). The claim names five
landmark classes -- event boundaries, commitment onset, interruption,
completion, offline transitions -- of which the first three are waking
(boundary pulses, the beta commitment gate, salience mode switches) and
the last (offline/sleep transitions) needs the sleep loop. Offline-relative
structure is therefore reported as an available-but-empty event type this
iteration and deferred to a lettered follow-up with use_sleep_loop=True.

FALSIFIER (claim-mandated, verbatim intent)
-------------------------------------------
An alignment statistic computed BOTH event-locked and clock-locked on the
SAME traces. PASS requires event-locked alignment to exceed clock-locked by
a margin scaled on the SD of the DELTA plus an absolute floor (the project
effect-size convention -- never the SD of either arm alone).

NON-DEGENERACY GUARD (claim-mandated): the boundary pulses must be non-
degenerate -- boundary rate NOT floor- or ceiling-pinned. A segmenter that
emits a boundary every step, or never, makes the event-vs-clock comparison
vacuous, and self-routes substrate_not_ready_requeue.

DESIGN
------
Per seed: a single shared substrate (E1 + E2-self P0 warmup, modelled on
V3-EXQ-824's run_p0) is trained once, then a StreamTraceRecorder
(q081_profile flags enabled, so the salience coordinator + event segmenter
are live) captures the full per-step multi-stream trace across REC_EPISODES.
There is NO manipulation arm -- the two "arms" of the falsifier are two
INDEXINGS of the one recorded trace, not two substrate runs.

PRIMARY STATISTIC -- the SAME multivariate RV coefficient (Robert &
Escoufier 1976) that V3-EXQ-824 uses for the (z_world, operating_mode)
pair, computed on the coarser (E3/operating_mode) stream's own fresh tick
grid so the sampling trap the telemetry audit (section 3) warns about
cannot manufacture a reading. It is computed TWICE on the one trace:

  RV_event : pool the coarse-grid samples lying in a peri-event window
             (+/- W_TICKS) around each EVENT ONSET, then RV(z_world,
             operating_mode) over that pool. Events = the UNION of waking
             landmark onsets: boundary pulses (MECH-288), commitment onset
             (beta_elevated rising edge), and salience mode switches
             (operating_mode.mode_switch_trigger). Offline transitions are
             included if present (empty this iteration; sleep off).
  RV_clock : pool equally-many, equal-width windows around CLOCK-ARBITRARY
             anchors -- coarse-grid positions NOT within W_TICKS of any
             event -- then the same RV. This is the "align by wall-clock
             timestep" reference.

Both pools are subsampled to a common size (seeded) so RV is not biased by
n, and both windows are episode-bounded (a window never crosses an episode
reset). delta = RV_event - RV_clock, aggregated across seeds under the
effect-size gate.

SECONDARY / CORROBORATING (not load-bearing): a per-seed circular-shift
surrogate on the event onsets (shift each onset by a random within-episode
offset, recompute RV_event) yields a null for RV_event that tests event-
LOCKING specifically -- a p-value that RV_event exceeds what the same
windowing gives when the events are decoupled from the trace. Reported per
seed; the load-bearing comparator remains the matched clock-anchored RV.

ADJUDICATION
------------
  Outcome E supported (event_relative_beats_clock_relative): mean delta
    clears the pre-registered effect-size floor -> the boundary stream is
    the coordination substrate, not merely a detector with a consumer.
  Outcome not-E (clock_relative_not_exceeded): the delta does not clear the
    floor -> coordination is no tighter around events than around arbitrary
    clock points; the boundary stream is a detector, not the federation's
    coordination substrate. Informative FAIL (weakens), per the project's
    non-standard-directions convention, NOT a null.
  substrate_not_ready_requeue: the boundary rate is floor/ceiling pinned,
    or too few events/samples/valid-seeds to compute the comparison. The
    claim's own non-degeneracy guard; re-queue at an adequate config.

DV-SYMMETRY DECLARATION (single measurement stream, no manipulation arm).
The DV is RV(z_world, operating_mode) computed over a SUBSET of the realised
joint trajectory. The "manipulation" is which time-indices the subset is
drawn from (event-anchored vs clock-anchored). RV centres and whitens each
block, so it is invariant to a broadcast additive constant and to any
orthogonal transform of either block -- but the manipulation here is NEITHER
of those: it is a re-SELECTION of realised sample rows, which changes the
empirical joint covariance being measured. A null delta is therefore a
MEASURED finding (Outcome not-E), not an arithmetic identity: nothing about
event-anchoring vs clock-anchoring forces the two RVs to be equal, and if
coordination were clock-organised rather than event-organised the delta
would come out <= 0 by measurement.

Provenance: env/agent/run_p0 pattern and the rv_coefficient math are taken
verbatim from experiments/v3_exq_824_q081_shared_organisation_landmark_removal.py;
per-tick mechanics via experiments/_harness.StepHarness (discharges the
q081_profile LOOP_DRIVEN_REQUIREMENTS -- sense harm kwargs + update_z_goal).

supersedes: none (first MECH-466 adjudication run).
ASCII-only output.
"""
from __future__ import annotations

import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.stream_recorder import (  # noqa: E402
    StreamTraceRecorder, OPERATING_MODES,
)
from experiments._lib.q081_profile import (  # noqa: E402
    q081_profile_kwargs, q081_substrate_declaration,
)
from experiments._lib.q081_surrogate import estimate_tick_period  # noqa: E402
from _harness import StepHarness  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_831_mech466_event_relative_alignment"
QUEUE_ID = "V3-EXQ-831"
CLAIM_IDS: List[str] = ["MECH-466"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES_RUN: Optional[str] = None

# ------------------------------------------------------------------ config
WORLD_DIM = SELF_DIM = 32
ALPHA_WORLD = 0.9   # SD-008: z_world-fidelity-dependent experiments need >= 0.9
ENV_SIZE = 6
SEEDS = [0, 1, 2, 3, 4]

WARMUP_EPISODES = 20
REC_EPISODES = 20
STEPS_PER_EPISODE = 200
WARMUP_LR = 1e-3

PRIMARY_PAIR = ("z_world", "operating_mode")

# ---------------------------------------------------------- pre-registered
# Peri-event window half-width, in COARSE-GRID TICKS (operating_mode is E3-
# rate, ~1 fresh sample per 10 env steps). W_TICKS=3 -> up to 7 coarse
# samples per window, episode-bounded.
W_TICKS = 3

# Non-degeneracy band on the boundary-pulse rate (fraction of env steps that
# carry at least one boundary onset). Floor: a segmenter that (almost) never
# fires cannot organise coordination. Ceiling: a segmenter firing on nearly
# every step makes "event-relative" indistinguishable from "clock-relative"
# by construction. Claim-mandated guard. Floor is a "segmenter is not silent"
# check (>= ~1 boundary per 1000 steps), NOT a "boundaries are frequent" gate:
# a legitimately-sparse-but-firing segmenter (V3-EXQ-824 seed2 measured ~0.0022,
# 4 boundaries in ~1800 steps) is non-degenerate and must NOT be excluded; only
# a truly floor-pinned (never fires) or ceiling-pinned (every step) rate is.
BOUNDARY_RATE_FLOOR = 0.001
BOUNDARY_RATE_CEIL = 0.60

# Minimum distinct UNION event anchors per seed, and minimum pooled coarse-
# grid samples per pool, for a valid RV on that seed.
MIN_EVENT_ANCHORS = 6
MIN_POOL_SAMPLES = 16

# Circular-shift surrogate ensemble size (secondary/diagnostic).
SURR_SHIFTS = 199

# Effect-size PASS gate (project convention: scale noise on the SD of the
# DELTA between the two indexings, plus an absolute floor). RV units, [0, 1].
EFFECT_SIZE_K = 1.5
EFFECT_SIZE_ABS_FLOOR = 0.03


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ============================================================================
# Substrate construction (verbatim pattern from V3-EXQ-824)
# ============================================================================

def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=ENV_SIZE, seed=seed)


def make_agent(env: CausalGridWorldV2) -> REEAgent:
    flags = q081_profile_kwargs()
    flags["use_sleep_loop"] = False  # scope: waking event grammar only
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        world_dim=WORLD_DIM,
        self_dim=SELF_DIM,
        **flags,
    )
    return REEAgent(cfg)


# ============================================================================
# P0 -- shared substrate warmup (E1 + E2-self only). StepHarness drives
# z_harm / z_harm_a / z_goal for free. Prints "step", not "ep", so the P1
# recording loop owns the canonical "ep N/M" progress pattern.
# ============================================================================

def run_p0(agent: REEAgent, env: CausalGridWorldV2, seed: int) -> Dict[str, Any]:
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=WARMUP_LR)
    harness = StepHarness(agent, env, train_mode=True)
    harm_events = 0
    total_harm = 0.0

    for ep in range(WARMUP_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(STEPS_PER_EPISODE):
            result = harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.harm_signal < 0:
                harm_events += 1
                total_harm += abs(result.harm_signal)

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            loss = e1_loss + e2_loss
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if result.done:
                break
        if (ep + 1) % 5 == 0 or ep == WARMUP_EPISODES - 1:
            print(f"  [warmup] seed={seed} step {ep + 1}/{WARMUP_EPISODES} "
                  f"harm={total_harm:.2f}", flush=True)

    return {"warmup_harm_events": harm_events, "warmup_total_harm": total_harm}


# ============================================================================
# P1 -- recording rollout for one seed (single substrate, no manipulation).
# ============================================================================

def run_p1_recording(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    rec: StreamTraceRecorder,
) -> Dict[str, Any]:
    agent.eval()
    harness = StepHarness(agent, env, train_mode=False)
    print(f"Seed {seed} Condition RECORD", flush=True)

    harm_events = 0
    reward_events = 0
    episode_lengths: List[int] = []
    n_steps = 0

    for ep in range(REC_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        ep_len = 0
        for _ in range(STEPS_PER_EPISODE):
            result = harness.step(obs_dict)
            rec.on_step(extras={"reward": result.harm_signal})
            if result.harm_signal < 0:
                harm_events += 1
            elif result.harm_signal > 0:
                reward_events += 1
            obs_dict = result.next_obs_dict
            ep_len += 1
            n_steps += 1
            if result.done:
                break
        rec.on_episode_end()
        episode_lengths.append(ep_len)
        if (ep + 1) % 4 == 0 or ep == REC_EPISODES - 1:
            print(f"  [rec] seed={seed} ep {ep + 1}/{REC_EPISODES}", flush=True)

    verdict = "PASS" if n_steps > 0 else "FAIL"
    print(f"verdict: {verdict}", flush=True)
    return {
        "harm_events": int(harm_events),
        "reward_events": int(reward_events),
        "n_steps": int(n_steps),
        "episode_lengths": episode_lengths,
    }


# ============================================================================
# Cross-stream RV + event/clock alignment
# ============================================================================

def _rv_rows(X: np.ndarray, Y: np.ndarray) -> float:
    """RV coefficient (Robert & Escoufier 1976) on two ALREADY-POOLED,
    row-aligned sample matrices. Same math as V3-EXQ-824.rv_coefficient,
    factored to operate on a pre-selected row set (the peri-event or peri-
    clock pool) instead of the full trace on its coarse grid."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    if X.shape[0] < 8 or Y.shape[0] < 8 or X.shape[0] != Y.shape[0]:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Cxy = Xc.T @ Yc
    Cxx = Xc.T @ Xc
    Cyy = Yc.T @ Yc
    num = float(np.trace(Cxy @ Cxy.T))
    den = float(np.sqrt(np.trace(Cxx @ Cxx) * np.trace(Cyy @ Cyy)))
    return num / den if den > 1e-30 else float("nan")


def _episode_of_steps(arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> np.ndarray:
    """Per-step episode index, read from the recorder clock block."""
    clock = np.asarray(arrays["clock"], dtype=np.float64)
    cols = list(meta.get("clock_columns", []))
    if "episode_index" in cols and clock.ndim == 2 and clock.shape[0] > 0:
        return clock[:, cols.index("episode_index")].astype(np.int64)
    return np.zeros(clock.shape[0] if clock.ndim == 2 else 0, dtype=np.int64)


def extract_event_onsets(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Step-indices of each WAKING landmark onset. Returns per-type arrays;
    the union is the primary event set."""
    n = int(np.asarray(arrays["z_world"]).shape[0])
    out: Dict[str, np.ndarray] = {}

    # boundary pulses (MECH-288): recorder marks boundary_events__fresh where
    # at least one event fired that step (count column > 0).
    bfresh = np.asarray(arrays.get("boundary_events__fresh",
                                   np.zeros(n, dtype=bool)), dtype=bool)
    out["boundary"] = np.flatnonzero(bfresh)

    # commitment onset: rising edge of the beta gate (0 -> 1).
    beta = np.asarray(arrays.get("beta_elevated", np.zeros((n, 1))),
                      dtype=np.float64).reshape(n, -1)[:, 0]
    beta_b = (beta > 0.5).astype(np.int64)
    comm = np.flatnonzero((beta_b[1:] == 1) & (beta_b[:-1] == 0)) + 1
    out["commitment"] = comm

    # salience mode switch: operating_mode.mode_switch_trigger column == 1 on
    # a fresh E3 tick. Layout: [soft probs over OPERATING_MODES], current_mode
    # index, mode_switch_trigger, salience_aggregate.
    om = np.asarray(arrays.get("operating_mode", np.zeros((n, 1))),
                    dtype=np.float64).reshape(n, -1)
    omfresh = np.asarray(arrays.get("operating_mode__fresh",
                                    np.zeros(n, dtype=bool)), dtype=bool)
    switch_col = len(OPERATING_MODES) + 1
    if om.shape[1] > switch_col:
        sw = (om[:, switch_col] > 0.5) & omfresh
        out["mode_switch"] = np.flatnonzero(sw)
    else:
        out["mode_switch"] = np.zeros(0, dtype=np.int64)

    # offline transitions: offline_mode value goes 0 -> 1 (empty with sleep off).
    off = np.asarray(arrays.get("offline_mode", np.zeros((n, 1))),
                     dtype=np.float64).reshape(n, -1)[:, 0]
    off_b = (off > 0.5).astype(np.int64)
    if n >= 2:
        offon = np.flatnonzero((off_b[1:] == 1) & (off_b[:-1] == 0)) + 1
    else:
        offon = np.zeros(0, dtype=np.int64)
    out["offline"] = offon

    union = np.unique(np.concatenate(
        [out["boundary"], out["commitment"], out["mode_switch"], out["offline"]]
    )) if n else np.zeros(0, dtype=np.int64)
    out["union"] = union.astype(np.int64)
    return out


def _windows_for_anchors(
    anchor_grid_idx: np.ndarray,
    grid: np.ndarray,
    grid_epi: np.ndarray,
    w_ticks: int,
) -> np.ndarray:
    """Given anchor positions expressed as INDICES into `grid`, return the set
    of unique STEP positions (values of `grid`) within +/- w_ticks grid ticks,
    clamped to the anchor's own episode."""
    keep: List[int] = []
    g_n = grid.size
    for j in anchor_grid_idx:
        j = int(j)
        epi = grid_epi[j]
        lo = max(0, j - w_ticks)
        hi = min(g_n - 1, j + w_ticks)
        for k in range(lo, hi + 1):
            if grid_epi[k] == epi:
                keep.append(int(grid[k]))
    return np.unique(np.asarray(keep, dtype=np.int64)) if keep else np.zeros(0, dtype=np.int64)


def analyze_event_vs_clock(
    arrays: Dict[str, np.ndarray], meta: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    """Compute RV_event, RV_clock and their delta on the coarse grid, plus the
    circular-shift surrogate p-value for RV_event."""
    name_a, name_b = PRIMARY_PAIR
    X = np.atleast_2d(np.asarray(arrays[name_a], dtype=np.float64))
    Y = np.atleast_2d(np.asarray(arrays[name_b], dtype=np.float64))
    n_steps = X.shape[0]
    fb = np.asarray(arrays[f"{name_b}__fresh"], dtype=bool)  # coarse (E3) grid

    out: Dict[str, Any] = {"seed": seed, "n_steps": int(n_steps)}

    grid = np.flatnonzero(fb)  # step-positions of coarse-grid samples
    epi = _episode_of_steps(arrays, meta)
    grid_epi = epi[grid] if grid.size else np.zeros(0, dtype=np.int64)

    onsets = extract_event_onsets(arrays)
    event_steps = onsets["union"]
    boundary_rate = float(onsets["boundary"].size) / max(1, n_steps)
    out["boundary_rate"] = boundary_rate
    out["event_type_counts"] = {k: int(v.size) for k, v in onsets.items() if k != "union"}
    out["n_union_events"] = int(event_steps.size)
    out["operating_mode_tick_period"] = float(estimate_tick_period(fb))

    # Map each event step to its nearest coarse-grid tick index.
    if grid.size < (2 * W_TICKS + 2) or event_steps.size < MIN_EVENT_ANCHORS:
        out.update({
            "rv_event": float("nan"), "rv_clock": float("nan"),
            "delta": float("nan"), "n_event_anchors": int(event_steps.size),
            "n_clock_anchors": 0, "n_event_pool": 0, "n_clock_pool": 0,
            "valid": False, "invalid_reason": "insufficient_grid_or_events",
            "surrogate_p_value": float("nan"),
        })
        return out

    pos_in_grid = np.searchsorted(grid, event_steps)
    pos_in_grid = np.clip(pos_in_grid, 0, grid.size - 1)
    # snap to the nearer of the two bracketing grid ticks
    for i, e in enumerate(event_steps):
        j = pos_in_grid[i]
        if j > 0 and abs(int(grid[j - 1]) - int(e)) < abs(int(grid[j]) - int(e)):
            pos_in_grid[i] = j - 1
    event_grid_idx = np.unique(pos_in_grid)

    # Clock anchors: coarse-grid ticks NOT within W_TICKS of any event tick.
    excluded = np.zeros(grid.size, dtype=bool)
    for j in event_grid_idx:
        lo = max(0, int(j) - W_TICKS)
        hi = min(grid.size - 1, int(j) + W_TICKS)
        excluded[lo:hi + 1] = True
    clock_candidates = np.flatnonzero(~excluded)

    rng = np.random.default_rng(1000 + seed)
    n_anchor = int(event_grid_idx.size)
    if clock_candidates.size >= n_anchor and n_anchor > 0:
        clock_grid_idx = np.sort(rng.choice(clock_candidates, size=n_anchor, replace=False))
    else:
        clock_grid_idx = clock_candidates

    event_pos = _windows_for_anchors(event_grid_idx, grid, grid_epi, W_TICKS)
    clock_pos = _windows_for_anchors(clock_grid_idx, grid, grid_epi, W_TICKS)

    # Match pooled sample counts so RV is not biased by n.
    m = int(min(event_pos.size, clock_pos.size))
    valid = bool(m >= MIN_POOL_SAMPLES)
    if valid:
        ev_sel = np.sort(rng.choice(event_pos, size=m, replace=False))
        ck_sel = np.sort(rng.choice(clock_pos, size=m, replace=False))
        rv_event = _rv_rows(X[ev_sel], Y[ev_sel])
        rv_clock = _rv_rows(X[ck_sel], Y[ck_sel])
    else:
        rv_event = rv_clock = float("nan")

    delta = (rv_event - rv_clock) if (np.isfinite(rv_event) and np.isfinite(rv_clock)) else float("nan")

    # Circular-shift surrogate for RV_event (secondary): shift each event step
    # by a random within-episode offset, recompute the matched-n event RV.
    surr_p = float("nan")
    if valid and np.isfinite(rv_event):
        ep_ids = epi
        ep_bounds: Dict[int, Tuple[int, int]] = {}
        for e in np.unique(ep_ids):
            idx = np.flatnonzero(ep_ids == e)
            ep_bounds[int(e)] = (int(idx[0]), int(idx[-1]))
        srng = np.random.default_rng(7000 + seed)
        ge = 0
        for _ in range(SURR_SHIFTS):
            shifted = []
            for e in event_steps:
                lo, hi = ep_bounds[int(ep_ids[e])]
                span = hi - lo + 1
                if span <= 1:
                    shifted.append(int(e))
                    continue
                shifted.append(lo + int((int(e) - lo + srng.integers(1, span)) % span))
            sh = np.unique(np.asarray(shifted, dtype=np.int64))
            spos = np.searchsorted(grid, sh)
            spos = np.clip(spos, 0, grid.size - 1)
            sgi = np.unique(spos)
            spool = _windows_for_anchors(sgi, grid, grid_epi, W_TICKS)
            if spool.size < m:
                continue
            ssel = np.sort(srng.choice(spool, size=m, replace=False))
            rv_s = _rv_rows(X[ssel], Y[ssel])
            if np.isfinite(rv_s) and rv_s >= rv_event:
                ge += 1
        surr_p = (ge + 1) / (SURR_SHIFTS + 1)

    out.update({
        "rv_event": float(rv_event), "rv_clock": float(rv_clock),
        "delta": float(delta),
        "n_event_anchors": int(event_grid_idx.size),
        "n_clock_anchors": int(clock_grid_idx.size),
        "n_event_pool": int(event_pos.size), "n_clock_pool": int(clock_pos.size),
        "n_matched_pool": int(m),
        "valid": valid,
        "invalid_reason": None if valid else "insufficient_matched_pool",
        "surrogate_p_value": float(surr_p),
    })
    return out


# ============================================================================
# Per-seed cell
# ============================================================================

def run_seed(seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_p0 = make_env(seed)
    agent = make_agent(env_p0)
    p0_stats = run_p0(agent, env_p0, seed)

    env_rec = make_env(seed)  # fresh env, byte-identical starting RNG
    rec = StreamTraceRecorder(
        agent, run_id=f"{EXPERIMENT_TYPE}_seed{seed}",
        substrate_declaration=q081_substrate_declaration(agent.config),
    )
    p1_stats = run_p1_recording(agent, env_rec, seed, rec)
    pointer = rec.finalize()
    blob = rec.store.get(pointer)
    analysis = analyze_event_vs_clock(blob["arrays"], blob["meta"], seed)

    print(
        f"  [seed={seed}] RV_event={analysis['rv_event']:.4f} "
        f"RV_clock={analysis['rv_clock']:.4f} delta={analysis['delta']:.4f} "
        f"boundary_rate={analysis['boundary_rate']:.4f} "
        f"n_events={analysis['n_union_events']} valid={analysis['valid']}",
        flush=True,
    )
    return {
        "seed": seed,
        "p0_stats": p0_stats,
        "p1_stats": p1_stats,
        "trace_pointer": pointer,
        "analysis": analysis,
    }


# ============================================================================
# Adjudication
# ============================================================================

def adjudicate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    analyses = [r["analysis"] for r in per_seed]
    n_seeds = len(analyses)

    boundary_rates = [a["boundary_rate"] for a in analyses]
    mean_boundary_rate = float(statistics.fmean(boundary_rates)) if boundary_rates else 0.0
    boundary_non_degenerate_fraction = (
        sum(1 for br in boundary_rates if BOUNDARY_RATE_FLOOR <= br <= BOUNDARY_RATE_CEIL)
        / max(1, n_seeds)
    )
    events_ok_fraction = (
        sum(1 for a in analyses if a["n_union_events"] >= MIN_EVENT_ANCHORS)
        / max(1, n_seeds)
    )

    valid_seeds: List[int] = []
    deltas: List[float] = []
    excluded: Dict[int, str] = {}
    for a in analyses:
        seed = a["seed"]
        if not (BOUNDARY_RATE_FLOOR <= a["boundary_rate"] <= BOUNDARY_RATE_CEIL):
            excluded[seed] = "boundary_rate_degenerate"
            continue
        if not a["valid"]:
            excluded[seed] = a.get("invalid_reason") or "invalid_analysis"
            continue
        if not (np.isfinite(a["rv_event"]) and np.isfinite(a["rv_clock"])):
            excluded[seed] = "non_finite_rv"
            continue
        valid_seeds.append(seed)
        deltas.append(float(a["delta"]))

    n_valid = len(valid_seeds)
    mean_delta = float(statistics.fmean(deltas)) if deltas else float("nan")
    sd_delta = float(statistics.stdev(deltas)) if len(deltas) >= 2 else 0.0
    effective_floor = max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR)
    delta_clears_floor = bool(n_valid >= 2 and np.isfinite(mean_delta)
                              and mean_delta >= effective_floor)

    preconditions = [
        {
            "name": "boundary_rate_non_degenerate_band",
            "description": "Claim-mandated non-degeneracy guard: the mean boundary-pulse "
                           "rate (fraction of env steps carrying a boundary onset) must lie "
                           "strictly inside the band -- neither floor-pinned (segmenter "
                           "silent, no event grammar to test) nor ceiling-pinned (a boundary "
                           "every step makes event-relative == clock-relative by "
                           "construction).",
            "measured": mean_boundary_rate,
            "threshold_low": BOUNDARY_RATE_FLOOR, "threshold_high": BOUNDARY_RATE_CEIL,
            "comparator_low": ">=", "comparator_high": "<=", "direction": "interval",
            "control": "q081 profile enables hippocampal.use_event_segmenter (MECH-288)",
            "met": bool(BOUNDARY_RATE_FLOOR <= mean_boundary_rate <= BOUNDARY_RATE_CEIL),
        },
        {
            "name": "boundary_non_degenerate_seed_fraction",
            "description": "Fraction of seeds whose own boundary rate is inside the band; "
                           "a single well-placed mean must not mask per-seed pinning.",
            "measured": boundary_non_degenerate_fraction, "threshold": 0.6,
            "direction": "lower", "control": None,
            "met": bool(boundary_non_degenerate_fraction >= 0.6),
        },
        {
            "name": "sufficient_union_events_seed_fraction",
            "description": f"Fraction of seeds with >= {MIN_EVENT_ANCHORS} distinct union "
                           "event onsets (boundary + commitment + mode-switch + offline), "
                           "so peri-event pooling is not starved.",
            "measured": events_ok_fraction, "threshold": 0.6, "direction": "lower",
            "control": None, "met": bool(events_ok_fraction >= 0.6),
        },
        {
            "name": "sufficient_valid_seeds_for_delta_sd",
            "description": "At least 2 seeds must clear every precondition (non-degenerate "
                           "boundary rate, valid matched pools, finite RVs) to compute an SD "
                           "of the event-minus-clock delta.",
            "measured": float(n_valid), "threshold": 2.0, "direction": "lower",
            "control": None, "met": bool(n_valid >= 2),
        },
    ]
    all_met = all(p["met"] for p in preconditions)

    criteria = [
        {
            "name": "C_event_relative_beats_clock_relative",
            "load_bearing": True,
            "passed": delta_clears_floor,
            "measured": mean_delta,
            "threshold": effective_floor,
            "note": (
                f"effective_floor = max({EFFECT_SIZE_K} * sd_delta={sd_delta:.4f}, "
                f"{EFFECT_SIZE_ABS_FLOOR}) = {effective_floor:.4f}; the project convention is "
                "to scale noise on the SD of the DELTA between the two indexings, never the "
                "SD of either alone."
            ),
        },
    ]
    criteria_non_degenerate = {
        "C_event_relative_beats_clock_relative": bool(all_met and n_valid >= 2),
    }

    if not all_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        degeneracy_reason = (
            "one or more readiness preconditions failed: "
            + "; ".join(f"{p['name']}={p['met']}" for p in preconditions if not p["met"])
        )
        summary = (
            "Boundary/event non-degeneracy or seed-count gate did not clear -- see "
            f"interpretation.preconditions. Excluded seeds: {excluded}. This is the claim's "
            "own non-degeneracy guard self-routing a re-queue, NOT a scientific verdict."
        )
    elif delta_clears_floor:
        label = "event_relative_beats_clock_relative"
        outcome = "PASS"
        evidence_direction = "supports"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"Outcome E supported: RV(z_world, operating_mode) measured event-locked exceeds "
            f"clock-locked by mean delta={mean_delta:.4f} (sd={sd_delta:.4f}) across {n_valid} "
            f"valid seeds, clearing the effective floor {effective_floor:.4f}. Cross-stream "
            "coordination is tighter around waking landmarks than around arbitrary clock "
            "points -- the boundary stream is the coordination substrate, not merely a "
            "detector with a consumer."
        )
    else:
        label = "clock_relative_not_exceeded"
        outcome = "FAIL"
        evidence_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"Outcome not-E: event-minus-clock RV delta mean={mean_delta:.4f} did not clear "
            f"the effective floor {effective_floor:.4f} across {n_valid} valid seeds. "
            "Coordination is no tighter around events than around arbitrary clock points -- "
            "the boundary stream reads as a detector, not the federation's coordination "
            "substrate. Informative FAIL (weakens), not a null result, per the project's "
            "non-standard-directions convention."
        )

    return {
        "label": label, "outcome": outcome, "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate, "degeneracy_reason": degeneracy_reason,
        "summary": summary, "preconditions": preconditions, "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "valid_seeds": valid_seeds, "excluded_seeds": excluded,
        "mean_delta": mean_delta, "sd_delta": sd_delta, "effective_floor": effective_floor,
        "mean_boundary_rate": mean_boundary_rate,
        "rv_event_all_seeds": [a["rv_event"] for a in analyses],
        "rv_clock_all_seeds": [a["rv_clock"] for a in analyses],
        "delta_all_seeds": [a["delta"] for a in analyses],
        "surrogate_p_all_seeds": [a["surrogate_p_value"] for a in analyses],
    }


# ============================================================================
# Main
# ============================================================================

def main(dry_run: bool = False) -> Any:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    global WARMUP_EPISODES, REC_EPISODES, STEPS_PER_EPISODE, SURR_SHIFTS
    if dry_run:
        WARMUP_EPISODES, REC_EPISODES, STEPS_PER_EPISODE, SURR_SHIFTS = 2, 3, 20, 19

    print(f"[{EXPERIMENT_TYPE}] dry_run={dry_run} seeds={seeds} "
          f"warmup_episodes={WARMUP_EPISODES} rec_episodes={REC_EPISODES} "
          f"steps_per_episode={STEPS_PER_EPISODE}", flush=True)
    t0 = time.time()

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(run_seed(seed))

    verdict = adjudicate(per_seed)
    elapsed = time.time() - t0
    print(f"[{EXPERIMENT_TYPE}] label={verdict['label']} outcome={verdict['outcome']} "
          f"mean_delta={verdict['mean_delta']:.4f} "
          f"mean_boundary_rate={verdict['mean_boundary_rate']:.4f} "
          f"elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest.", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    per_seed_results = [
        {
            "seed": r["seed"],
            "p0_stats": r["p0_stats"],
            "p1_stats": r["p1_stats"],
            "trace_pointer": r["trace_pointer"],
            "analysis": r["analysis"],
        }
        for r in per_seed
    ]

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": verdict["outcome"],
        "evidence_direction": verdict["evidence_direction"],
        "non_degenerate": verdict["non_degenerate"],
        "degeneracy_reason": verdict["degeneracy_reason"],
        "interpretation": {
            "label": verdict["label"],
            "summary": verdict["summary"],
            "preconditions": verdict["preconditions"],
            "criteria_non_degenerate": verdict["criteria_non_degenerate"],
            "scope_note": (
                "WAKING event grammar only (use_sleep_loop=False). Offline/sleep-phase "
                "transitions -- the fifth landmark class in MECH-466 -- are recorded as an "
                "available-but-empty event type this iteration and deferred to a lettered "
                "follow-up with the sleep loop enabled. Sibling of V3-EXQ-824, which "
                "deferred this Outcome E."
            ),
        },
        "criteria": verdict["criteria"],
        "elapsed_seconds": elapsed,
        "per_seed_results": per_seed_results,
        "delta_summary": {
            "valid_seeds": verdict["valid_seeds"],
            "excluded_seeds": verdict["excluded_seeds"],
            "mean_delta": verdict["mean_delta"],
            "sd_delta": verdict["sd_delta"],
            "effective_floor": verdict["effective_floor"],
            "mean_boundary_rate": verdict["mean_boundary_rate"],
            "rv_event_all_seeds": verdict["rv_event_all_seeds"],
            "rv_clock_all_seeds": verdict["rv_clock_all_seeds"],
            "delta_all_seeds": verdict["delta_all_seeds"],
            "surrogate_p_all_seeds": verdict["surrogate_p_all_seeds"],
        },
        "primary_pair": list(PRIMARY_PAIR),
        "method_note": (
            "RV_event vs RV_clock are the SAME multivariate RV coefficient computed on two "
            "indexings of ONE recorded trace, read on the operating_mode (E3) fresh tick "
            "grid to avoid the sampling trap. Event anchors = union of waking landmark onsets "
            "(MECH-288 boundary pulses, beta commitment-gate rising edge, salience "
            "mode_switch_trigger). Clock anchors = matched-count coarse-grid ticks disjoint "
            "from event neighbourhoods. Pools matched to a common n; windows episode-bounded. "
            "Secondary circular-shift surrogate tests event-locking specifically."
        ),
    }
    if SUPERSEDES_RUN:
        manifest["supersedes"] = SUPERSEDES_RUN

    out_path = write_flat_manifest(
        manifest, dry_run=False,
        config={
            "seeds": seeds, "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
            "alpha_world": ALPHA_WORLD, "env_size": ENV_SIZE,
            "warmup_episodes": WARMUP_EPISODES, "rec_episodes": REC_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE, "primary_pair": list(PRIMARY_PAIR),
            "w_ticks": W_TICKS, "boundary_rate_floor": BOUNDARY_RATE_FLOOR,
            "boundary_rate_ceil": BOUNDARY_RATE_CEIL,
            "min_event_anchors": MIN_EVENT_ANCHORS, "min_pool_samples": MIN_POOL_SAMPLES,
            "surr_shifts": SURR_SHIFTS, "effect_size_k": EFFECT_SIZE_K,
            "effect_size_abs_floor": EFFECT_SIZE_ABS_FLOOR,
            "use_sleep_loop": False,
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0,
    )
    print(f"Result written to: {out_path}", flush=True)
    return verdict["outcome"], out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="V3-EXQ-831 MECH-466 event-relative vs clock-relative cross-stream alignment")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        emit_outcome(outcome="PASS", manifest_path=None, dry_run=True)
        sys.exit(0)
    outcome, out_path = result
    # main() returns 0 before writing a manifest under --dry-run, so this call is
    # unreachable in a smoke run and the old literal False was correct. Threaded
    # anyway so the guarantee is LOCAL: it survives a future edit that lets the
    # dry-run path fall through to the writer. Inert on the evidence path.
    emit_outcome(outcome=outcome, manifest_path=out_path,
                 dry_run=bool(args.dry_run))
    sys.exit(0)
