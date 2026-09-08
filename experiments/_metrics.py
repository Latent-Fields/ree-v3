"""
Canonical metric extractors for ree-v3 experiment scripts.

Each extractor is a pure function: take ``agent`` (and sometimes a few
per-tick scalars) and return a dict the script can merge into per_seed_results.

These exist to avoid the kind of measurement bug found in EXQ-490c/490e
where the script read ``agent.dacc._last_bundle.get('mode_ev')`` and
reported its norm as "dacc_score_bias_mean" -- but score_bias is the [K]
tensor that DACCtoE3Adapter produces from the bundle, not the raw mode_ev
slice. Plus a ``try/except: norm=0.0`` was hiding shape mismatches.

NOTE ON THAT SPELLING (2026-07-29): ``agent.dacc._last_bundle`` above is
quoted as the ORIGINAL BUG and is doubly wrong -- there is no ``_last_bundle``
attribute on the dACC module at all. The bundle lives on the AGENT as
``agent._dacc_last_bundle`` (written ``ree_core/agent.py:6148``, canonical
read ``:10340``). Use that spelling. ``getattr(dacc, "_last_bundle", None)``
returns None on every tick, so any max/mean derived from it is pinned to 0.0
BY CONSTRUCTION rather than measured. ``validate_experiments.py``'s
``dacc_last_bundle`` lint now fails the corpus on the wrong spelling so the
class cannot silently return.

Single-source-of-truth extractors mean the script does not have to know
which substrate slot to read; bug fixes propagate through one place.

Conventions
-----------
All extractors:
- Return a flat dict[str, float | int | bool] with stable keys.
- Tolerate the substrate being disabled by returning a {} or zero-filled dict
  with the keys the experiment writer expects to merge.
- DO NOT silently swallow shape errors. If a substrate is enabled but its
  diagnostic shape does not match what we expect, raise -- the experiment
  author can decide whether to special-case it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# -- Goal / drive ------------------------------------------------------------

def extract_goal_diagnostics(agent) -> Dict[str, float]:
    """
    Snapshot of GoalState the moment this is called.

    Use inside an on_post_step hook to record per-tick history; the script
    aggregates the history at episode end.
    """
    gs = getattr(agent, "goal_state", None)
    if gs is None:
        return {"goal_active": 0, "goal_norm": 0.0, "drive_level": 0.0}
    norm = 0.0
    try:
        norm = float(gs.goal_norm())
    except Exception:
        # Fall through with norm=0; do not silently swallow elsewhere.
        norm = 0.0
    drive_cached = float(getattr(gs, "_last_drive_level", 0.0) or 0.0)
    return {
        "goal_active": int(bool(gs.is_active())),
        "goal_norm": norm,
        "drive_level": drive_cached,
    }


# -- dACC --------------------------------------------------------------------

def extract_dacc_score_bias(agent) -> Optional[torch.Tensor]:
    """
    Return the [K] score_bias tensor that DACCtoE3Adapter produced from the
    most recent dACC bundle, OR None if dACC is disabled / not yet ticked.

    This is the tensor passed into ``E3.select(score_bias=...)`` and is what
    the C3-style "did dACC actually bias action selection?" criterion is
    measuring. Reading the raw bundle slot ``mode_ev`` instead (as the
    EXQ-490 cohort did) measures something different.

    The score_bias is cached on the agent in ``_dacc_last_bias`` if present;
    otherwise we recompute via the adapter using the last bundle.

    INSTRUMENT REPAIR (2026-07-29). BOTH substrate attribute names in this
    function were wrong, so it returned None unconditionally whenever dACC was
    enabled and ``extract_dacc_diagnostics`` below therefore emitted its
    zero-filled dict on every tick -- a structural zero, not a measurement:
      * the cache is ``agent._dacc_last_bias`` (``ree_core/agent.py:2699``,
        written ``:6151``/``:6282``), never ``agent._last_dacc_score_bias``;
      * the bundle is ``agent._dacc_last_bundle`` (written ``:6148``, canonical
        read ``:10340``), never ``dacc._last_bundle`` -- the dACC module has no
        such attribute (``ree_core/cingulate/dacc.py`` defines none).
    Neither name existed anywhere in ``ree_core``, so ``getattr`` swallowed both
    silently. These extractors had no callers at the time of the repair, so no
    landed manifest is affected through this file.
    """
    dacc = getattr(agent, "dacc", None)
    if dacc is None:
        return None
    cached = getattr(agent, "_dacc_last_bias", None)
    if cached is not None:
        return cached
    bundle = getattr(agent, "_dacc_last_bundle", None)
    if bundle is None:
        return None
    adapter = getattr(agent, "dacc_adapter", None)
    if adapter is None:
        # Adapter not constructed (rare config). Caller should treat as None.
        return None
    return adapter.forward(bundle)


def extract_dacc_diagnostics(agent) -> Dict[str, float]:
    """
    Aggregate dACC diagnostics from the actual score_bias the adapter produced.

    Returns:
        dacc_score_bias_norm: L2 norm of the [K] vector this tick
        dacc_score_bias_max_abs: max |bias| across candidates
        dacc_score_bias_nonzero: 1 if any |bias| > 1e-6, else 0
    """
    sb = extract_dacc_score_bias(agent)
    if sb is None:
        return {
            "dacc_score_bias_norm": 0.0,
            "dacc_score_bias_max_abs": 0.0,
            "dacc_score_bias_nonzero": 0,
        }
    sb_t = torch.as_tensor(sb).detach().flatten()
    if sb_t.numel() == 0:
        return {
            "dacc_score_bias_norm": 0.0,
            "dacc_score_bias_max_abs": 0.0,
            "dacc_score_bias_nonzero": 0,
        }
    return {
        "dacc_score_bias_norm": float(sb_t.norm().item()),
        "dacc_score_bias_max_abs": float(sb_t.abs().max().item()),
        "dacc_score_bias_nonzero": int(sb_t.abs().max().item() > 1e-6),
    }


# -- PAG freeze gate ---------------------------------------------------------

def extract_pag_diagnostics(agent) -> Dict[str, float]:
    """
    PAG freeze gate state this tick.

    Returns:
        pag_freeze_active: 1/0
        pag_freeze_commit: 1/0  (entered freeze this tick)
        pag_exit_threshold: scaled exit_threshold (0 when gate disabled)
    """
    gate = getattr(agent, "pag_freeze_gate", None)
    if gate is None:
        return {
            "pag_freeze_active": 0,
            "pag_freeze_commit": 0,
            "pag_exit_threshold": 0.0,
        }
    last = getattr(gate, "last_output", None)
    if last is None:
        return {
            "pag_freeze_active": 0,
            "pag_freeze_commit": 0,
            "pag_exit_threshold": 0.0,
        }
    return {
        "pag_freeze_active": int(bool(getattr(last, "freeze_active", False))),
        "pag_freeze_commit": int(bool(getattr(last, "freeze_commit", False))),
        "pag_exit_threshold": float(getattr(last, "exit_threshold", 0.0) or 0.0),
    }


# -- Broadcast override (SD-037) --------------------------------------------

def extract_broadcast_override_diagnostics(agent) -> Dict[str, float]:
    bo = getattr(agent, "broadcast_override", None)
    if bo is None:
        return {"override_signal": 0.0}
    return {"override_signal": float(getattr(bo, "override_signal", 0.0) or 0.0)}


# -- V_s rollout gate (MECH-269b) -------------------------------------------

def extract_vs_gate_diagnostics(agent) -> Dict[str, Any]:
    gate = getattr(agent, "vs_rollout_gate", None)
    if gate is None:
        return {
            "vs_gate_total_held_e1": 0,
            "vs_gate_total_held_e2": 0,
            "vs_gate_n_snapshots": 0,
        }
    diag = gate.get_diagnostics()
    # Normalize numeric types so np.mean over a list of these dicts works.
    return {k: (int(v) if isinstance(v, bool) else v) for k, v in diag.items()}


# -- MECH-295 bridge --------------------------------------------------------

def extract_bridge_diagnostics(agent) -> Dict[str, float]:
    br = getattr(agent, "mech295_bridge", None)
    if br is None:
        return {
            "bridge_n_write_fires_total": 0,
            "bridge_n_cue_fires_total": 0,
        }
    return {
        "bridge_n_write_fires_total": int(getattr(br, "_n_write_fires", 0)),
        "bridge_n_cue_fires_total": int(getattr(br, "_n_cue_fires", 0)),
    }


# -- Residue / valence ------------------------------------------------------

def extract_residue_valence_summary(agent) -> Dict[str, float]:
    """
    Sample the four-component valence vector at the agent's current z_world.

    Returns 0s when valence is disabled or no current latent. Uses the agent's
    ``_current_latent.z_world`` (set by sense()).
    """
    field = getattr(agent, "residue_field", None)
    latent = getattr(agent, "_current_latent", None)
    if field is None or latent is None or not hasattr(field, "evaluate_valence"):
        return {
            "valence_wanting": 0.0,
            "valence_liking": 0.0,
            "valence_harm": 0.0,
            "valence_surprise": 0.0,
        }
    try:
        v = field.evaluate_valence(latent.z_world)
    except Exception:
        return {
            "valence_wanting": 0.0,
            "valence_liking": 0.0,
            "valence_harm": 0.0,
            "valence_surprise": 0.0,
        }
    v_t = torch.as_tensor(v).detach().flatten()
    if v_t.numel() < 4:
        return {
            "valence_wanting": 0.0,
            "valence_liking": 0.0,
            "valence_harm": 0.0,
            "valence_surprise": 0.0,
        }
    return {
        "valence_wanting": float(v_t[0].item()),
        "valence_liking": float(v_t[1].item()),
        "valence_harm": float(v_t[2].item()),
        "valence_surprise": float(v_t[3].item()),
    }


# -- Action mode classifier --------------------------------------------------

def classify_action_mode(
    *,
    z_harm_norm: float,
    world_change_norm: float,
    harm_signal: float,
    harm_mode_thresh: float = 0.25,
    explore_err_thresh: float = 0.10,
    benefit_signal_thresh: float = 0.01,
) -> str:
    """
    Heuristic 4-way mode label. Note that "approach" here is detected via the
    benefit signal (positive harm_signal), so this is closer to "made resource
    contact" than "committed to approach trajectory". Use sparingly and label
    metric output accordingly.
    """
    if z_harm_norm > harm_mode_thresh:
        return "avoid"
    if harm_signal > benefit_signal_thresh:
        return "approach"
    if world_change_norm > explore_err_thresh:
        return "explore"
    return "neutral"


# -- Aggregation helper -----------------------------------------------------

def aggregate_per_tick_logs(logs: Dict[str, list]) -> Dict[str, float]:
    """
    Convert a dict of per-tick scalar lists into mean/max summaries.

    Pass in {"override_signal": [...], "drive_level": [...], ...}; get back
    {"override_signal_mean": ..., "override_signal_max": ..., "drive_level_mean": ..., ...}.

    Empty lists produce zeros.
    """
    out: Dict[str, float] = {}
    for k, vals in logs.items():
        if not vals:
            out[f"{k}_mean"] = 0.0
            out[f"{k}_max"] = 0.0
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_max"] = float(arr.max())
    return out


# -- Non-degeneracy self-report + P0 readiness abort gate --------------------
#
# These two helpers let an experiment self-report the degenerate / vacuous-
# criterion failure mode that previously only a manual /failure-autopsy could
# catch (V3-EXQ-514m C_WL pinned at 0.0; V3-EXQ-642 z_block identically 0).
#
#   check_degeneracy(...)  -> writes manifest fields non_degenerate /
#       non_degenerate_per_claim / degeneracy_reason. The REE_assembly indexer
#       (build_experiment_indexes.py) treats non_degenerate=false as
#       scoring_excluded="degenerate" -- the run stays in the full log but does
#       NOT weight claim confidence/conflict, exactly like "superseded".
#
#   p0_readiness_gate(...) -> a pre-registered abort gate. Call it after P0
#       training and BEFORE the expensive measurement phase; on an unmet
#       precondition it raises P0NotReady carrying a manifest-ready
#       preconditions[] payload, so the script can write a
#       substrate_not_ready_requeue manifest and skip P1/P2 rather than burn
#       compute and emit a misleading FAIL.


def metric_is_degenerate(
    values,
    *,
    eps: float = 1e-9,
    floor: Optional[float] = None,
    ceiling: Optional[float] = None,
) -> tuple[bool, str]:
    """A discriminative metric is DEGENERATE when it has no usable spread across
    the observations its criterion compares -- pinned at a constant (zero
    cross-arm/cross-seed variance), floor-pinned, or ceiling-saturated on every
    observation -- so the criterion can never fire regardless of behaviour (the
    V3-EXQ-514m C_WL=0.0 / V3-EXQ-642 z_block=0 vacuous-criterion pattern).

    `values` is the list/array of the metric's observed values across the cells
    its criterion compares (e.g. per-arm-per-seed separations). Returns
    (degenerate: bool, reason: str). reason is "" when non-degenerate.

    `floor` / `ceiling` catch the *saturation* family the bare zero-spread test
    misses when a readout is pinned at a rail with tiny residual jitter. A metric
    is degenerate if every observation is <= floor (floor-pinned, e.g. an
    approach-rate that never lifts off 0) OR >= ceiling (ceiling-saturated, e.g.
    the V3-EXQ-651 goal_prox ~0.98 readout whose on-vs-off delta is below its own
    resolution -- spread alone leaves it uncaught because the jitter exceeds eps).
    Keep `eps` tight (1e-9): the bit-identical / exact-zero family is the safe
    catch, and widening eps would false-positive genuine small-but-real spreads
    (a near-miss separation is a weak result, NOT a vacuous criterion). Use the
    floor/ceiling rails -- keyed to the metric's own bounds -- for saturation,
    never a loosened eps.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return True, "no finite observations"
    if not np.all(np.isfinite(arr)):
        return True, "non-finite observation(s) present"
    spread = float(arr.max() - arr.min())
    if spread <= eps:
        return True, (f"zero spread (constant={float(arr.flat[0]):.6g}, "
                      f"spread={spread:.3g}<=eps={eps:.3g})")
    if floor is not None and float(arr.max()) <= float(floor):
        return True, (f"floor-pinned (max={float(arr.max()):.6g}<=floor="
                      f"{float(floor):.6g})")
    if ceiling is not None and float(arr.min()) >= float(ceiling):
        return True, (f"ceiling-saturated (min={float(arr.min()):.6g}>=ceiling="
                      f"{float(ceiling):.6g})")
    return False, ""


def metric_groups_are_degenerate(
    groups,
    *,
    eps: float = 1e-9,
    floor: Optional[float] = None,
    ceiling: Optional[float] = None,
) -> tuple[bool, str]:
    """Paired/within-group variant of :func:`metric_is_degenerate`.

    Use this when the criterion fires on a *within-group separation* -- e.g. an
    ARM_ON-vs-ARM_OFF difference measured per seed -- rather than on raw values
    pooled across cells. `groups` is a list of value-lists (one per seed / per
    comparison block). The run is degenerate when EVERY group is internally
    degenerate (its arms are bit-identical / pinned), even if the metric varies
    ACROSS groups. This is the V3-EXQ-603 / 543e bit-identical-arms family: pool
    the raw (seed x arm) values into one flat list and the cross-seed variance
    masks the within-seed zero-difference, so :func:`metric_is_degenerate` on the
    flat list wrongly passes; this checks each group in isolation.

    An equivalent and often simpler producer-side option is to feed
    :func:`metric_is_degenerate` the per-group SEPARATION directly (e.g.
    [arm_on_i - arm_off_i for each seed i]); this helper exists for when the raw
    per-cell values are what was logged.

    ARITY GUARD (do not remove): a group of length < 2 has no spread to
    measure and is skipped rather than treated as pinned -- passing this
    function a list of SINGLETON groups (e.g. one value per seed, wrapped as
    `[[v] for v in values]` instead of a flat `values` list) used to make
    every group read "zero spread" BY CONSTRUCTION and the whole metric
    report degenerate regardless of genuine cross-seed variation
    (V3-EXQ-961: a real, well-above-floor signal was reported degenerate this
    way and silently dropped from scoring). If a flat, ungrouped list of
    observations is what you have, call :func:`metric_is_degenerate` on it
    directly instead of wrapping each value in its own singleton group.
    """
    groups = list(groups)
    if not groups:
        return True, "no groups"
    reasons = []
    any_measurable = False
    for i, g in enumerate(groups):
        g = list(g)
        if len(g) < 2:
            # A group of length < 2 carries NO spread information -- there is
            # nothing to compare it against, so it can never be evidence of
            # pinning. Without this guard metric_is_degenerate reads a
            # singleton as "zero spread" and reports it pinned BY
            # CONSTRUCTION, not by measurement -- the exact bug that made
            # V3-EXQ-961 report a genuinely-graded metric (values
            # 1.20998/1.09383/1.04082, well above both the 0.5 criterion
            # floor and this function's own 1e-6 floor) as degenerate, which
            # then silently excluded a sound run from scoring
            # (build_experiment_indexes.py's non-degeneracy gate). Treat it
            # as unmeasurable and skip -- never as pinned.
            reasons.append(
                f"group[{i}]: insufficient arity (n={len(g)}, need >=2 to "
                f"measure spread) -- skipped, not treated as pinned")
            continue
        any_measurable = True
        is_deg, reason = metric_is_degenerate(
            g, eps=eps, floor=floor, ceiling=ceiling)
        if not is_deg:
            return False, ""
        reasons.append(f"group[{i}]: {reason}")
    if not any_measurable:
        # Every group was a singleton (or shorter): nothing here was ever
        # measurable, so this is NOT a degenerate verdict -- it is an
        # insufficient-arity finding. Reporting non_degenerate=False on this
        # would still (correctly) mark it as not-genuinely-tested, but the
        # DEGENERATE label specifically must never fire on arity alone.
        return False, (
            "no group had sufficient arity (>=2) to measure spread; " +
            "; ".join(reasons))
    return True, "every measurable group pinned -- " + "; ".join(reasons)


def check_degeneracy(
    load_bearing_metrics: Dict[str, Any],
    *,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """Aggregate non-degeneracy self-report for a run's manifest.

    `load_bearing_metrics` maps each load-bearing discriminative metric name to
    EITHER the list of its observed values (across the cells its criterion
    compares) OR a dict accepting any of:
        {"values": [...],                 # flat per-cell observations
         "floor":   <float>,              # degenerate if every value <= floor
         "ceiling": <float>,              # degenerate if every value >= ceiling
         "groups":  [[...], [...], ...]}  # per-seed/per-block arm values:
                                          #   degenerate if EVERY group is pinned
    Provide EITHER "values" OR "groups". A run is non_degenerate iff EVERY
    load-bearing metric has usable spread.

    Returns a dict to merge into the manifest:
        {"non_degenerate": bool,
         "degeneracy_reason": str,                       # "" when non-degenerate
         "degenerate_metrics": {name: reason, ...}}      # only the offenders

    Writing non_degenerate=false makes the REE_assembly indexer exclude the run
    from confidence/conflict scoring (scoring_excluded="degenerate").
    """
    degenerate: Dict[str, str] = {}
    for name, spec in load_bearing_metrics.items():
        if isinstance(spec, dict):
            floor = spec.get("floor")
            ceiling = spec.get("ceiling")
            groups = spec.get("groups")
            if groups is not None:
                is_deg, reason = metric_groups_are_degenerate(
                    groups, eps=eps, floor=floor, ceiling=ceiling)
            else:
                is_deg, reason = metric_is_degenerate(
                    spec.get("values", []), eps=eps, floor=floor, ceiling=ceiling)
        else:
            is_deg, reason = metric_is_degenerate(spec, eps=eps)
        if is_deg:
            degenerate[name] = reason
    non_degen = not degenerate
    reason = "" if non_degen else "; ".join(
        f"{k}: {v}" for k, v in degenerate.items())
    return {
        "non_degenerate": non_degen,
        "degeneracy_reason": reason,
        "degenerate_metrics": degenerate,
    }


class P0NotReady(Exception):
    """Raised by p0_readiness_gate when a pre-registered precondition is unmet.

    Carries the manifest-ready preconditions[] payload (each entry with
    measured/threshold/direction/met) so the caller can write a
    substrate_not_ready_requeue manifest and abort before the measurement phase.
    """

    def __init__(self, preconditions: list, reason: str):
        self.preconditions = preconditions
        self.reason = reason
        super().__init__(reason)


# -- Non-finite readiness measurements (the NaN hole) ------------------------ #
#
# The gate computes `met` with `>=` / `<=`; the REE_assembly indexer RECOMPUTES it
# from the reported (measured, threshold) pair with the negated `<` / `>`, and
# treats its own recompute as AUTHORITATIVE. For every finite measurement the two
# agree. For NaN they do NOT: every comparison against NaN is False, so the gate
# reads `nan >= t` as False (UNMET) while the indexer reads `nan < t` as False
# (MET). A genuine premise failure is silently cleared and the run is wrongly
# trusted -- the confirmed V3-EXQ-680c mis-scoring of
# `r1_grad_cosine_not_net_negative` (nan vs 0.0, met False, read as met).
#
# Fix: substitute a sentinel that sits beyond ANY plausible threshold on the
# UNMET side of the bound, so the entry recomputes to its own `met` on its own.
# Direction matters -- a large NEGATIVE sentinel is below any floor but would read
# as MET against a ceiling -- so the sentinel is chosen per resolved direction.
# The true value is not lost: it is preserved verbatim (as a string, since NaN is
# not valid JSON) on the non-bound diagnostic keys `measured_non_finite` /
# `non_finite`, which the indexer ignores and a human reader sees.
#
# +/-inf is deliberately NOT substituted: it already compares identically in the
# gate and in the recompute, on both bound directions.
#
# Relation to the 680-series precedent: 680/680a/680b/680c patched this
# per-driver with a local `_nan_floor_guard()` (same sentinel value) PLUS a
# sibling `r1_grad_cosine_finite` precondition. Those drivers are left untouched
# -- they pre-substitute a FINITE value, so this gate passes it through unchanged
# and their shipped manifests are bit-identical. The finiteness sibling is now
# redundant (it relied on the adjudicator returning at the FIRST unmet entry,
# which is ordering-dependent; per-entry recomputability does not), but it is
# harmless and gate-equivalent, and removing it would edit drivers with shipped
# manifests for no behavioural gain.
NON_FINITE_FLOOR_SENTINEL = -1e30    # below any floor  -> recomputes UNMET
NON_FINITE_CEILING_SENTINEL = 1e30   # above any ceiling -> recomputes UNMET

_UPPER_DIRECTIONS = ("upper", "ceiling", "max", "upper_bound")
_LOWER_DIRECTIONS = ("lower", "floor", "min", "lower_bound")
_VALID_COMPARATORS = (">=", ">", "<=", "<")


def _readiness_is_upper(direction: str, comparator: str) -> bool:
    """Resolve a readiness check's bound side, mirroring the indexer's
    _precondition_direction EXACTLY -- comparator first, then direction, then the
    "lower" default. Kept in lockstep so the gate's `met` and the indexer's
    recompute cannot disagree about which side of the bound a check lives on."""
    if comparator in ("<=", "<"):
        return True
    if comparator in (">=", ">"):
        return False
    return direction.strip().lower() in _UPPER_DIRECTIONS


# -- DV-headroom preconditions (dv-dynamic-range-precondition-class) --------- #
#
# Every readiness gate in this corpus certifies the INTERVENTION -- was the channel
# perturbed, did the head train, were there enough samples -- and NONE certifies
# that the DEPENDENT VARIABLE had room to move. Across the seven 2026-09-03
# pending-review runs, six passed all their preconditions and still could not
# discriminate, because the registered pass threshold lay outside the range the
# configuration could produce:
#
#   V3-EXQ-981  C1 threshold 1.154 on a DV bounded in [0,1]        (unsatisfiable)
#   V3-EXQ-981  precision-margin elevation 0.000195 vs floor 0.01        (51x)
#   V3-EXQ-951c gate_caused with zero reachable ticks                (no support)
#   V3-EXQ-983  decline_gap realised range 0.0468 vs C1 0.15            (3.2x)
#   V3-EXQ-993  max |calibration_gap| 0.00152 vs floor 0.02            (13.1x)
#   V3-EXQ-994  retention spread 0.00078 vs 0.02                       (25.6x)
#   V3-EXQ-978  arm-mean difference one third of the DV's 0.05 quantum
#
# A `dv_headroom` check answers one question the other kinds never ask: CAN this
# DV, in THIS configuration, produce a value the registered threshold would
# accept? It is expressed in the gate's existing single-bound shape --
#
#     measured  = what the DV can actually achieve here (the CONTROL arm's
#                 realised dynamic range, or the arithmetic room left above a
#                 saturated baseline)
#     threshold = what the criterion requires (its registered threshold, times an
#                 optional safety margin)
#     direction = "lower"  ->  met iff achievable >= required
#
# -- so an unmet entry raises P0NotReady like any other and the caller writes the
# substrate_not_ready_requeue manifest it already writes. Nothing downstream
# changes: the REE_assembly indexer recomputes `met` from (measured, threshold,
# direction) and is kind-agnostic, so an unmet dv_headroom entry adjudicates as
# `precondition_unmet` with no indexer change. That is deliberate -- governance
# governance-20260903T2013 scoped this build to the HARNESS (validate_experiments.py
# + p0_readiness_gate) precisely so it could not perturb the 1,201 drivers that
# import the substrate.
#
# OPT-IN, and byte-identical when not opted into. A driver acquires a dv_headroom
# entry only by building one; every existing driver's `kind` is "readiness" or one
# of the dozen bespoke labels, so the validation below cannot fire on shipped code.
#
# PRECEDENT, and why this is a harness abstraction rather than a per-driver idiom:
# V3-EXQ-777a hand-rolled exactly this guard as two local precondition entries
# (`baseline_entropy_headroom`, `score_dv_headroom_seeds` -- the latter's own
# `control` string calls it "the guard V3-EXQ-777 lacked"). One driver having
# invented it locally, after a run was lost for want of it, is the argument for
# putting it where the next driver inherits it instead of re-deriving it.
DV_HEADROOM_KIND = "dv_headroom"

# How "what the DV can achieve" is measured. The four statistics are not
# interchangeable -- each matches a different one of the corpus failures above,
# and picking the wrong one produces a gate that passes while the DV is pinned.
DV_HEADROOM_STATISTICS = ("range", "max_abs", "ceiling_headroom", "floor_headroom",
                          "explicit", "floor_separation")


def dv_achievable(
    control_values: Sequence[float],
    statistic: str = "range",
    dv_bounds: Optional[Tuple[float, float]] = None,
) -> float:
    """Measure what a DV actually achieved in the CONTROL arm. Returns a float.

    `statistic` selects the measurement, and the choice is scientific, not
    stylistic:

      "range"            max - min over the control arm's realised values. The
                         DV's demonstrated dynamic range. Use when the criterion
                         reads a SPREAD or a between-arm difference (983's
                         decline_gap 0.0468 vs 0.15; 994's retention spread).
      "max_abs"          max |v|. The largest magnitude the control arm produced.
                         Use when the criterion reads a signed deviation against
                         an absolute floor (993's max |calibration_gap| 0.00152
                         vs SEPARATED_SIGNAL_FLOOR 0.02).
      "ceiling_headroom" dv_bounds[1] - max(v). The arithmetic room left ABOVE a
                         saturated baseline. Use for an ELEVATION criterion on a
                         bounded DV (981's precision-margin: 0.000195 available
                         against a 0.01 floor, a 51x shortfall that no
                         intervention could have closed).
      "floor_headroom"   min(v) - dv_bounds[0]. The mirror, for a SUPPRESSION
                         criterion against a floored baseline.
      "explicit"         refused here -- pass `achievable=` to dv_headroom_check
                         instead, for a DV whose ceiling is analytic rather than
                         sampled (e.g. 951c's "zero reachable ticks", where the
                         achievable count is a property of the schedule).
      "floor_separation" refused here -- this label belongs to
                         `dv_floor_control_check`, not this function. It answers
                         a different question ("does an information-free floor
                         arm already satisfy the criterion") and computes its
                         own signed separation directly from the caller's
                         floor_values; there is nothing here for dv_achievable
                         to measure.

    A non-finite value anywhere in `control_values` yields NaN rather than an
    order-dependent max: the measurement is broken, and NaN is routed to UNMET by
    p0_readiness_gate's existing sentinel substitution. That is the honest
    outcome -- a headroom gate that cannot measure the DV must not certify it.
    """
    if statistic not in DV_HEADROOM_STATISTICS:
        raise ValueError(
            f"dv_achievable: statistic {statistic!r}; expected one of "
            f"{DV_HEADROOM_STATISTICS}.")
    if statistic == "explicit":
        raise ValueError(
            "dv_achievable: the 'explicit' statistic has nothing to measure; "
            "pass achievable=<float> to dv_headroom_check() instead.")
    if statistic == "floor_separation":
        raise ValueError(
            "dv_achievable: the 'floor_separation' statistic has nothing for "
            "dv_achievable to measure; call dv_floor_control_check(floor_values=...) "
            "instead -- it computes the signed separation directly.")
    vals = [float(v) for v in control_values]
    if not vals:
        # A headroom gate over an EMPTY control arm is the vacuity it exists to
        # catch, arriving one level up. Refuse loudly rather than certify it.
        raise ValueError(
            "dv_achievable: control_values is empty; there is no realised range "
            "to measure. Fix the caller (the control arm produced no readings) "
            "rather than gating on nothing.")
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        return float("nan")
    if statistic == "range":
        return max(vals) - min(vals)
    if statistic == "max_abs":
        return max(abs(v) for v in vals)
    if dv_bounds is None:
        raise ValueError(
            f"dv_achievable: statistic {statistic!r} needs dv_bounds=(low, high) "
            "-- headroom against a bound is undefined without the bound.")
    low, high = float(dv_bounds[0]), float(dv_bounds[1])
    if not high > low:
        raise ValueError(
            f"dv_achievable: dv_bounds=({low}, {high}) is not an interval.")
    if statistic == "ceiling_headroom":
        return high - max(vals)
    return min(vals) - low          # floor_headroom


def _dv_headroom_scope_phrase(entry: Dict[str, Any]) -> str:
    """How the achievable value was obtained -- said in terms the check can see.

    Deliberately does NOT name an arm, a seed set, or anything else the check was
    not given. A caller that filtered its own cells can pass `measured_cells` /
    `n_dropped_nonfinite` and have them named; a caller that does not gets an
    honest count instead of a confident misdescription. That asymmetry is the
    whole repair -- see dv_headroom_check's docstring.
    """
    stat = entry.get("achievable_statistic")
    if stat == "explicit":
        return "analytic ceiling supplied by the caller, not measured from a sample"
    n = int(entry.get("n_control_values") or 0)
    parts = ["%s over %d finite value(s)" % (stat, n)]
    cells = entry.get("measured_cells")
    if cells:
        shown = ", ".join(str(c) for c in cells[:8])
        if len(cells) > 8:
            shown += ", ..."
        parts.append("cells: %s" % shown)
    dropped = entry.get("n_dropped_nonfinite")
    if dropped:
        parts.append("%d non-finite cell(s) dropped" % int(dropped))
    return "; ".join(parts)


def _dv_headroom_reason(entry: Dict[str, Any]) -> str:
    """One sentence saying what was measured, over what, and against what.

    ASCII only (CLAUDE.md): this reaches stdout and lands in manifests.
    """
    dv = entry.get("dv_name")
    measured = float(entry.get("measured"))
    required = float(entry.get("threshold"))
    scope = _dv_headroom_scope_phrase(entry)
    against = ("against a required %.6g (criterion threshold %.6g x margin %g)"
               % (required, float(entry.get("criterion_threshold")),
                  float(entry.get("headroom_margin"))))
    if measured != measured:  # NaN
        return ("DV HEADROOM INDETERMINATE: %s achievable range is NaN (%s), %s. "
                "A NaN cannot be compared to the bar, so this is not a refusal on "
                "the DV's range -- find out why the input was non-finite."
                % (dv, scope, against))
    if measured >= required:
        return ("DV headroom met: %s can reach %.6g (%s), %s."
                % (dv, measured, scope, against))
    if measured <= 0.0:
        # The V3-EXQ-983a case. A ratio-based shortfall here divides by ~zero and
        # prints an absurd figure (983a's hand-rolled string said "1000000000000.0x");
        # say the true thing instead.
        return ("DV HEADROOM UNMET: %s does not move at all in this configuration "
                "-- achievable %.6g (%s), %s. No outcome of this run could have "
                "shown the registered effect."
                % (dv, measured, scope, against))
    return ("DV HEADROOM UNMET: %s can only reach %.6g in this configuration (%s), "
            "%s -- a %.1fx shortfall. No outcome of this run could have shown the "
            "registered effect."
            % (dv, measured, scope, against, required / measured))


def dv_headroom_check(
    name: str,
    *,
    dv_name: str,
    criterion_threshold: float,
    control_values: Optional[Sequence[float]] = None,
    achievable: Optional[float] = None,
    statistic: str = "range",
    dv_bounds: Optional[Tuple[float, float]] = None,
    margin: float = 1.0,
    measured_cells: Optional[Sequence[str]] = None,
    n_dropped_nonfinite: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build one `dv_headroom` check for p0_readiness_gate. Returns a check dict.

    `criterion_threshold` is the threshold the LOAD-BEARING criterion actually
    registers -- pass the same module constant the criterion reads, never a
    re-typed literal, so the gate cannot drift away from the science it guards.

    `margin` (>= 1.0) requires headroom to EXCEED the threshold by a factor
    rather than merely reach it. The default 1.0 asserts bare feasibility: the DV
    could, in principle, produce a passing value. A margin of 2.0 asserts the
    threshold sits at most half the achievable range away -- the honest setting
    when the criterion needs room to resolve an effect, not just to touch the
    bound. Both are defensible; the default is the weaker claim.

    Supply EITHER `control_values` (measured via `statistic`) or `achievable`
    (an analytic ceiling, for a DV whose reachable range is a property of the
    schedule rather than a sample -- 951c's zero reachable ticks).

    GUIDANCE (dv_headroom_floor_control_direction_20260907.md sub-direction
    2a): when a criterion's PASSING side requires movement AWAY FROM an
    information-free configuration (a collapsed latent, a random projection,
    an untrained head), `control_values` must be THAT CONFIGURATION'S OWN
    REALISED VALUES -- never a null. Passing a null control arm here answers
    "can the DV move at all", not "can it move far enough from the floor it
    would sit at by default", and the two questions have different answers
    (H2's motivating case: a random 275->32 projection already retains 78%
    of the raw decodability lift, so a null-control headroom check passes
    while the actual, floor-relative headroom is unsatisfiable). If the
    question is instead "does an information-free floor ALREADY satisfy the
    criterion" (a distinct defect -- the floor need not move at all to
    pass), use `dv_floor_control_check` below, not this constructor.

    The returned dict is a plain check; it does not gate anything until it is
    passed to p0_readiness_gate(), which is where an unmet entry raises
    P0NotReady and the caller self-routes to substrate_not_ready_requeue.

    THE HUMAN-READABLE REASON IS COMPOSED HERE, FROM THE VALUES THE CHECK
    ACTUALLY USED -- never from a caller's separate metadata. That is the point
    of moving it (2026-09-07, chip-20260906-dv-headroom-reason-string).
    V3-EXQ-983a hand-composed its own refusal text at the call site, from the
    control-arm NAME and the pooled-seed LIST, while the check had measured the
    range over every finite pooled cell of BOTH arms -- which, after dropping one
    seed's non-finite declines, was one seed's two arms. The landed manifest
    therefore describes the wrong arm scope and the wrong seed count while its
    own `preconditions[...]` entry is correct. The two could disagree because
    they had two different sources; now they have one. (Found by the fable
    red-team of failure_autopsy_V3-EXQ-983a_2026-09-06, hygiene finding 6;
    REE_assembly 71694d5b01. The landed manifest is NOT edited.)

    `measured_cells` and `n_dropped_nonfinite` are OPTIONAL and exist so a caller
    that filtered its own input can say what survived and what it dropped -- the
    check receives already-filtered floats and cannot otherwise know. When they
    are omitted the reason says only how many finite values it measured: it
    never invents an arm or seed scope it cannot see. That restraint is the
    actual fix, not the extra fields.
    """
    if (control_values is None) == (achievable is None):
        raise ValueError(
            f"dv_headroom_check: check {name!r} must supply exactly one of "
            "control_values= (measured) or achievable= (analytic).")
    if achievable is None:
        measured = dv_achievable(control_values, statistic=statistic,
                                 dv_bounds=dv_bounds)
        n_control = len(list(control_values))
    else:
        measured = float(achievable)
        statistic = "explicit"
        n_control = 0
    margin = float(margin)
    if not margin >= 1.0:
        # A margin below 1 would certify a DV that CANNOT reach the threshold --
        # it inverts the gate's meaning rather than loosening it.
        raise ValueError(
            f"dv_headroom_check: check {name!r} has margin {margin}; the margin "
            "scales the REQUIRED headroom and must be >= 1.0.")
    required = float(criterion_threshold) * margin
    entry: Dict[str, Any] = dict(extra)
    entry.update({
        "name": str(name),
        "kind": DV_HEADROOM_KIND,
        "measured": measured,
        "threshold": required,
        "direction": "lower",
        "dv_name": str(dv_name),
        "achievable_statistic": statistic,
        "criterion_threshold": float(criterion_threshold),
        "headroom_margin": margin,
        "n_control_values": n_control,
    })
    # The shortfall the autopsy table reports ("13.1x", "25.6x") is 1/ratio.
    # Recorded so a reader of the manifest sees HOW FAR out of range the
    # criterion was, not merely that it was.
    if required > 0 and measured == measured:
        entry["headroom_ratio"] = measured / required
    if measured_cells is not None:
        entry["measured_cells"] = [str(c) for c in measured_cells]
    if n_dropped_nonfinite is not None:
        entry["n_dropped_nonfinite"] = int(n_dropped_nonfinite)
    entry["headroom_reason"] = _dv_headroom_reason(entry)
    if dv_bounds is not None:
        entry["dv_bounds"] = [float(dv_bounds[0]), float(dv_bounds[1])]
    return entry


def _dv_floor_control_reason(entry: Dict[str, Any]) -> str:
    """One sentence saying whether the FLOOR ARM already satisfies the
    load-bearing criterion, and by how much. ASCII only (CLAUDE.md): this
    reaches stdout and lands in manifests.

    Mirrors `_dv_headroom_reason`'s shape (composed here, from the values
    the check itself used -- the 983a lesson dv_headroom_check's own
    docstring records) but describes a different quantity: that function
    asks whether the DV CAN REACH the bar; this asks whether an
    INFORMATION-FREE floor arm has ALREADY CLEARED it.
    """
    dv = entry.get("dv_name")
    measured = float(entry.get("measured"))   # signed separation, larger = safer
    margin = float(entry.get("threshold"))    # required separation_margin
    bar = float(entry.get("criterion_threshold"))
    sense = entry.get("criterion_sense")
    scope = _dv_headroom_scope_phrase(entry)
    if measured != measured:  # NaN
        return ("DV FLOOR CONTROL INDETERMINATE: %s floor-arm separation from the "
                "%s bar %.6g is NaN (%s). A NaN cannot be compared to the required "
                "separation -- find out why the input was non-finite."
                % (dv, sense, bar, scope))
    if measured >= margin:
        return ("DV floor control met: %s floor arm (%s) sits %.6g away from the "
                "%s bar %.6g, clearing the required separation %.6g -- the design "
                "can discriminate the manipulation from the information-free floor."
                % (dv, scope, measured, sense, bar, margin))
    if measured <= 0.0:
        return ("DV FLOOR CONTROL UNMET: %s floor arm (%s) already SATISFIES the "
                "%s bar %.6g by itself (separation %.6g) -- no outcome of this run "
                "could distinguish the manipulation from the information-free floor."
                % (dv, scope, sense, bar, measured))
    return ("DV FLOOR CONTROL UNMET: %s floor arm (%s) sits only %.6g away from "
            "the %s bar %.6g, short of the required separation %.6g -- the design "
            "leaves too little room to distinguish the manipulation from the floor."
            % (dv, scope, measured, sense, bar, margin))


def dv_floor_control_check(
    name: str,
    *,
    dv_name: str,
    criterion_threshold: float,
    floor_values: Sequence[float],
    criterion_sense: str,
    separation_margin: float = 0.0,
    measured_cells: Optional[Sequence[str]] = None,
    n_dropped_nonfinite: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build one `dv_headroom` check certifying that an information-free FLOOR
    ARM does NOT already satisfy the load-bearing criterion by itself --
    sub-direction (2b) of the DV-headroom class
    (REE_assembly/evidence/planning/dv_headroom_floor_control_direction_20260907.md
    section 6). The sibling to `dv_headroom_check`, which asks the OPPOSITE
    question -- "can the DV reach the bar" -- rather than this function's
    "has an information-free configuration already cleared it, without the
    manipulation under test doing anything at all".

    Four historical corpus cases this answers, none of which the existing
    `dv_headroom_check` / `criterion_exceeds_achievable_range_lint` catch
    (design doc section 4, GOV-HELDOUT-1 record section 7): V3-EXQ-622
    (collapsed z_goal already yields `approach_commit_rate` 1.0 against a
    0.01 bar), V3-EXQ-723 (any weak linear map clears both the compactness
    and retention conjuncts), V3-EXQ-884 (2 credits already clears
    `n_subgoal_credits > 0`), V3-EXQ-920a (a criterion that cannot fail when
    the mechanism it monitors never fires). The reference design that got
    this right on its own, and must NOT trip this check: V3-EXQ-1002's
    `untrained_control + UNTRAINED_CONTROL_MARGIN` conjunct, which puts its
    untrained floor at a genuine 0.105 separation from its 0.80 bar.

    `criterion_sense` says which side of the ORIGINAL criterion is a pass.
    REQUIRED, never inferred -- the two senses invert which floor-arm value
    (max vs min) is the dangerous one, and there is no safe default:

      "floor"    the criterion passes when measured >= criterion_threshold
                 (622's and 884's shape). Separation =
                 criterion_threshold - max(floor_values): POSITIVE means the
                 floor's best-case value still falls short of the bar
                 (safe); NEGATIVE means the floor already clears it (the
                 defect).
      "ceiling"  the criterion passes when measured <= criterion_threshold
                 (723's compactness conjunct). Separation =
                 min(floor_values) - criterion_threshold, same sign
                 convention (positive = safe).

    `separation_margin` (default 0.0) is the minimum signed separation the
    floor arm must keep from the bar. 0.0 asks only that the floor not
    already clear it; a positive value (as V3-EXQ-1002's own
    UNTRAINED_CONTROL_MARGIN does, hand-rolled) demands real daylight, not a
    boundary touch.

    `direction` is always "lower" (met iff separation >= separation_margin)
    -- expressing the check as a SIGNED SEPARATION rather than as a raw
    bound on the floor arm's value is what keeps this on the same
    floor-only rail `_validate_dv_headroom_check` already enforces for
    `dv_headroom_check` (an upper bound there would invert the gate's
    meaning; the same inversion risk exists here, so this constructor never
    exposes the choice).

    `kind` stays "dv_headroom" so the lint and the runtime gate remain ONE
    feature and the REE_assembly indexer -- which recomputes `met` from
    (measured, threshold, direction) and is kind-agnostic -- needs no
    change. `achievable_statistic` is the "floor_separation" label, refused
    by `dv_achievable()` exactly as "explicit" already is: there is nothing
    for `dv_achievable` to measure here either, since the caller supplies
    the floor arm's own realised values directly and this function computes
    the signed separation itself.

    THE REASON IS COMPOSED HERE, FROM THE VALUES THIS CHECK ACTUALLY USED --
    never re-derived by a caller from separate metadata (the 983a lesson;
    see `dv_headroom_check`'s own docstring for the incident this
    generalises from).

    Refuses an empty `floor_values`: a floor-control gate over no floor-arm
    readings is the vacuity this class exists to catch, arriving one level
    up -- fix the caller rather than gate on nothing.
    """
    if criterion_sense not in ("floor", "ceiling"):
        raise ValueError(
            f"dv_floor_control_check: check {name!r} has criterion_sense "
            f"{criterion_sense!r}; expected 'floor' or 'ceiling'. There is no "
            "safe default -- the two senses invert which floor-arm value (max "
            "vs min) is the dangerous one.")
    vals = [float(v) for v in floor_values]
    if not vals:
        raise ValueError(
            f"dv_floor_control_check: check {name!r} has empty floor_values; "
            "there is no floor-arm value to compare against the bar. Fix the "
            "caller (the floor arm produced no readings) rather than gating "
            "on nothing.")
    bar = float(criterion_threshold)
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        measured = float("nan")
    elif criterion_sense == "floor":
        measured = bar - max(vals)
    else:
        measured = min(vals) - bar
    margin = float(separation_margin)
    entry: Dict[str, Any] = dict(extra)
    entry.update({
        "name": str(name),
        "kind": DV_HEADROOM_KIND,
        "measured": measured,
        "threshold": margin,
        "direction": "lower",
        "dv_name": str(dv_name),
        "achievable_statistic": "floor_separation",
        "criterion_threshold": bar,
        "criterion_sense": criterion_sense,
        "headroom_margin": margin,
        "n_control_values": len(vals),
    })
    if measured_cells is not None:
        entry["measured_cells"] = [str(c) for c in measured_cells]
    if n_dropped_nonfinite is not None:
        entry["n_dropped_nonfinite"] = int(n_dropped_nonfinite)
    entry["headroom_reason"] = _dv_floor_control_reason(entry)
    return entry


DV_HEADROOM_OBSERVATION_FLAG = "headroom_ceiling_exceeded_by_observation"
# Float-noise guard. An observation must EXCEED the ceiling by more than this to
# count; a value equal to it (a DV that touched its own bound) is not evidence of
# a mis-specification.
DV_HEADROOM_OBSERVATION_TOLERANCE = 1e-12


def dv_headroom_observation_check(
    entry: Dict[str, Any],
    observed: Optional[Sequence[float]],
    *,
    observed_name: Optional[str] = None,
    tolerance: float = DV_HEADROOM_OBSERVATION_TOLERANCE,
) -> Dict[str, Any]:
    """Falsify a headroom ceiling with the run's OWN observations. Returns `entry`.

    Call this at MANIFEST-EMIT time, once the test statistic has actually been
    observed, passing the observed values of the SAME statistic the load-bearing
    criterion reads, in the criterion's own orientation. It annotates `entry`
    in place with `headroom_ceiling_exceeded_by_observation` and returns it, so it
    can wrap an existing preconditions[] entry without restructuring the emit.

    WHY THIS IS STRONGER THAN THE STATIC LINT, and why it is the generalisable
    lesson rather than a second guard. A headroom entry asserts a CEILING: the
    largest value the DV could produce in this configuration. That is a universal
    claim, so a single observation above it REFUTES it outright -- no distributional
    assumption, no judgement about which statistic was the right one, no access to
    the driver's source. validate_experiments.dv_headroom_statistic_mismatch_lint
    can only ask whether the entry and the criterion look like they read the same
    quantity; this KNOWS, from data the run already has, and it catches mismatches
    the static scan cannot see at all -- a threshold passed as a literal, a
    criterion assembled across functions, a statistic that is subtly wrong for
    reasons no name reveals.

    THE CONFIRMED CASE. V3-EXQ-972a's T3 entry asserted an achievable ceiling of
    0.0806 (`1 - max(per-seed control accuracy)`) on the mean paired difference.
    Two of the eight observed paired diffs were 0.16135 and 0.09297 -- both ABOVE
    the asserted ceiling, one of them clearing the criterion's own 0.15 threshold.
    The ceiling was not a ceiling. Nothing in the run said so, the entry was read
    as an instrument failure, and an adequately ranged null (the mean-matched
    ceiling is 0.155563, ratio 1.037) was adjudicated as a substrate limit. This
    check turns that into a recorded flag on the entry at the moment the manifest
    is written. (Confirmed cluster autopsy, REE_assembly cb4a71fbd9,
    evidence/planning/failure_autopsy_dv-headroom-diagnostics-cluster_2026-09-07.md.)

    IT NEVER RAISES, and that is deliberate. By emit time the compute is spent;
    an exception here would cost the manifest to report a problem WITH the
    manifest. It records and returns. The flag's `exceeded` boolean is what a
    reader, an autopsy, or a later gate reads.

    ORIENTATION IS THE CALLER'S JOB, and it is the one thing to get right. Pass
    the statistic as the CRITERION reads it (`mean_diff`'s per-seed inputs for a
    mean criterion, the per-cell values for a per-cell criterion). Headroom is a
    floor gate on a ceiling quantity -- both `achievable` and the criterion's
    statistic run in the same direction -- so the test is a plain `observed >
    achievable`. Passing magnitudes against a signed ceiling, or the wrong arm's
    values, produces a flag that means nothing.

    NON-FINITE observations are dropped and counted rather than compared; a NaN
    ceiling (`measured` is NaN) is INDETERMINATE, not met and not exceeded, and is
    recorded as such -- the same asymmetry dv_headroom_check applies, for the same
    reason: a measurement that failed must not certify anything.
    """
    if not isinstance(entry, dict):
        return entry
    flag: Dict[str, Any] = {
        "checked": True,
        "exceeded": False,
        "statistic_observed": str(observed_name) if observed_name
        else str(entry.get("dv_name") or "<unnamed>"),
    }
    try:
        ceiling = float(entry.get("measured"))
    except (TypeError, ValueError):
        flag.update({"checked": False,
                     "reason": ("no finite `measured` on this entry, so there is no "
                                "asserted ceiling to falsify")})
        entry[DV_HEADROOM_OBSERVATION_FLAG] = flag
        return entry
    vals_all = list(observed) if observed is not None else []
    vals: List[float] = []
    n_dropped = 0
    for v in vals_all:
        try:
            f = float(v)
        except (TypeError, ValueError):
            n_dropped += 1
            continue
        if f != f or f in (float("inf"), float("-inf")):
            n_dropped += 1
            continue
        vals.append(f)
    flag["n_observed"] = len(vals)
    if n_dropped:
        flag["n_observed_nonfinite_dropped"] = n_dropped
    if ceiling != ceiling:
        flag.update({"checked": False, "asserted_ceiling": None,
                     "reason": ("asserted ceiling is NaN, so no observation can "
                                "exceed it -- INDETERMINATE, not met. Find out why "
                                "the headroom input was non-finite.")})
        entry[DV_HEADROOM_OBSERVATION_FLAG] = flag
        return entry
    flag["asserted_ceiling"] = ceiling
    if not vals:
        flag.update({"checked": False,
                     "reason": ("no finite observations of the test statistic were "
                                "supplied, so the ceiling was not tested")})
        entry[DV_HEADROOM_OBSERVATION_FLAG] = flag
        return entry
    tol = float(tolerance)
    over = [v for v in vals if (v - ceiling) > tol]
    flag["max_observed"] = max(vals)
    if not over:
        flag["reason"] = (
            "%d observed value(s) of %s, max %.6g, all at or below the asserted "
            "ceiling %.6g -- the ceiling is not contradicted by this run's data."
            % (len(vals), flag["statistic_observed"], max(vals), ceiling))
        entry[DV_HEADROOM_OBSERVATION_FLAG] = flag
        return entry
    flag["exceeded"] = True
    flag["n_exceeding"] = len(over)
    flag["exceeding_values"] = [float(v) for v in sorted(over, reverse=True)[:8]]
    crit = entry.get("criterion_threshold")
    over_crit = None
    try:
        if crit is not None:
            over_crit = sum(1 for v in over if v >= float(crit))
    except (TypeError, ValueError):
        over_crit = None
    tail = ""
    if over_crit:
        tail = (" %d of them also clear the criterion's own threshold %.6g, so this "
                "run produced passing values of a statistic the gate called out of "
                "reach." % (over_crit, float(crit)))
    flag["reason"] = (
        "HEADROOM CEILING CONTRADICTED BY OBSERVATION: %d of %d observed value(s) of "
        "%s exceed the asserted achievable ceiling %.6g (largest %.6g). A ceiling is "
        "a universal claim, so this refutes it -- the entry is measuring a different "
        "quantity, a different order statistic, or the wrong arm, and its verdict "
        "should not be read as a substrate limit.%s"
        % (len(over), len(vals), flag["statistic_observed"], ceiling, max(over), tail))
    entry[DV_HEADROOM_OBSERVATION_FLAG] = flag
    return entry


def _validate_dv_headroom_check(name: str, check: dict, is_upper: bool) -> None:
    """Refuse a malformed `dv_headroom` entry. Returns None; raises ValueError.

    Only ever called for an entry whose `kind` IS "dv_headroom", so it cannot
    touch a driver that has not opted in.

    Two refusals, both because the failure they prevent is SILENT:

    (1) An UPPER bound inverts the gate. "Achievable must be at most the
        threshold" certifies exactly the runs this class exists to stop -- it
        passes when the DV is pinned and fails when it has room. A floor is not a
        stylistic preference here; it is the whole semantics.

    (2) A missing `dv_name` / `achievable_statistic` leaves the manifest entry
        unreadable after the fact. "measured 0.0468 vs threshold 0.15" does not
        say WHICH variable had no room, or what "achievable" was taken to mean --
        and the four statistics are not interchangeable. The 2026-09-03 cluster
        cost seven runs precisely because nothing recorded this; an entry that
        cannot be recomputed by a later reader repeats that.

    Both are author errors at wiring time, caught on the first run rather than in
    the next autopsy.
    """
    if is_upper:
        raise ValueError(
            f"p0_readiness_gate: dv_headroom check {name!r} resolves to an UPPER "
            "bound; headroom is a FLOOR (achievable >= required). An upper bound "
            "inverts the gate -- it would pass a pinned DV and fail a live one. "
            "Drop the direction/comparator override, or build a plain readiness "
            "check if you meant something else.")
    dv_name = check.get("dv_name")
    if not (isinstance(dv_name, str) and dv_name.strip()):
        raise ValueError(
            f"p0_readiness_gate: dv_headroom check {name!r} is missing a "
            "non-empty `dv_name`. Name the dependent variable whose range is "
            "being certified -- the manifest entry is not interpretable without "
            "it. dv_headroom_check() sets this for you.")
    stat = check.get("achievable_statistic")
    if stat not in DV_HEADROOM_STATISTICS:
        raise ValueError(
            f"p0_readiness_gate: dv_headroom check {name!r} has "
            f"achievable_statistic {stat!r}; expected one of "
            f"{DV_HEADROOM_STATISTICS}. The statistics are not interchangeable "
            "(a range, a max-abs and a ceiling-headroom answer different "
            "questions), so the entry must say which was measured.")


def p0_readiness_gate(checks: list) -> list:
    """Pre-registered P0 abort gate -- assert the substrate is trained enough to
    make the measurement non-vacuous BEFORE burning compute on P1/P2.

    `checks` is a list of dicts, each:
        {"name": str, "measured": float, "threshold": float,
         "direction": "lower"|"upper",    # lower=floor: met iff measured>=threshold
                                          # upper=ceiling: met iff measured<=threshold
         "comparator": ">="|">"|"<="|"<"} # OPTIONAL: the PASS comparison, i.e.
                                          # met == (measured <comparator> threshold)
    (direction defaults to "lower"; comparator defaults to the inclusive form of
    the resolved direction). The semantics mirror the REE_assembly indexer's
    _precondition_direction / _precondition_unmet so the recorded preconditions[]
    adjudicate consistently.

    STRICTNESS. A driver whose shipped predicate is strict (`>` / `<`) must pass
    `comparator`, otherwise its `met` is computed inclusively at the boundary and
    disagrees with its own science. `met` mirrors the comparator exactly, and the
    key is passed through to the manifest, where the indexer honours it. Absent a
    comparator the behaviour is the pre-existing inclusive one, bit-identical.

    EXTRA KEYS. Any key beyond name/measured/threshold/direction/comparator is
    passed through to the manifest entry untouched, so a driver can attach
    non-bound diagnostics (counts, per-seed detail, notes) in the same dict
    instead of re-attaching them after the call. `kind` defaults to "readiness"
    but may be overridden by the caller.

    DV_HEADROOM. `kind: "dv_headroom"` (build it with dv_headroom_check(), see
    above) certifies that the DEPENDENT VARIABLE has room to reach its own
    registered threshold -- the one thing every other kind in this corpus leaves
    unchecked, and the cause of six of the seven 2026-09-03 pending-review runs
    passing all preconditions and still discriminating nothing. It rides the
    ordinary single-bound path (measured = achievable, threshold = required,
    floor), so an unmet entry raises P0NotReady and self-routes exactly like any
    other; the only added behaviour is a wiring-time refusal of a malformed
    entry (see _validate_dv_headroom_check). Opt-in per driver: a caller that
    does not set this kind is unaffected, bit-identically.

    NON-FINITE measurements are sentinel-substituted -- see NON_FINITE_*_SENTINEL
    above for the mechanism and why it is needed.

    Returns a manifest-ready preconditions[] list when ALL checks are met. Raises
    P0NotReady (with the same payload) when any check fails, so the caller writes
    interpretation={"label": "substrate_not_ready_requeue", "preconditions": ...}
    and self-routes to non_contributory instead of a misleading FAIL.

    SINGLE-BOUND ONLY. A two-sided band (threshold_low/threshold_high) is not
    expressible here and is REFUSED rather than silently read as a one-legged
    floor -- that mis-read is the exact defect family this gate is being kept
    honest against. Build such an entry directly.
    """
    preconditions = []
    unmet = []
    for c in checks:
        name = str(c["name"])
        if "threshold_low" in c or "threshold_high" in c or \
                str(c.get("direction", "")).strip().lower() in \
                ("interval", "between", "band", "range", "two_sided", "two-sided"):
            raise ValueError(
                f"p0_readiness_gate: check {name!r} declares a two-sided band; "
                "the gate is single-bound only. Build the interval precondition "
                "entry directly (the indexer supports threshold_low/high).")

        comparator = c.get("comparator")
        comparator = comparator.strip() if isinstance(comparator, str) else ""
        if comparator and comparator not in _VALID_COMPARATORS:
            # Never fall back silently: an unrecognised comparator would default to
            # an inclusive bound, i.e. a typo would quietly loosen the gate.
            raise ValueError(
                f"p0_readiness_gate: check {name!r} has comparator "
                f"{comparator!r}; expected one of {_VALID_COMPARATORS}.")
        direction = str(c.get("direction", "lower"))
        is_upper = _readiness_is_upper(direction, comparator)
        strict = comparator in (">", "<")

        # DV-headroom entries carry extra structure the generic path cannot check.
        # Gated on the kind, so a driver that has not opted in is untouched.
        if str(c.get("kind", "")) == DV_HEADROOM_KIND:
            _validate_dv_headroom_check(name, c, is_upper)

        m = float(c["measured"])
        t = float(c["threshold"])
        entry = dict(c)   # pass through any caller diagnostics / unknown keys
        if m != m:        # NaN -- substitute so the entry recomputes to its own met
            entry["measured_non_finite"] = "nan"
            entry["non_finite"] = True
            m = NON_FINITE_CEILING_SENTINEL if is_upper else NON_FINITE_FLOOR_SENTINEL

        if is_upper:
            met = (m < t) if strict else (m <= t)
        else:
            met = (m > t) if strict else (m >= t)

        entry.update({
            "name": name,
            "measured": m,
            "threshold": t,
            "direction": direction,
            "met": bool(met),
            "kind": str(c.get("kind", "readiness")),
        })
        if comparator:
            entry["comparator"] = comparator
        else:
            entry.pop("comparator", None)
        preconditions.append(entry)
        if not met:
            unmet.append(name)
    if unmet:
        raise P0NotReady(preconditions, "P0 readiness unmet: " + ", ".join(unmet))
    return preconditions


# -- Crystallization-necessity harness guards (MECH-334 / INV-074; 610-655 lineage) --
#
# The INV-074 / MECH-334 crystallization-necessity test has a recurring harness
# no-op that wasted runs across V3-EXQ-610c/610d/610e/610f/655: either the policy
# was never genuinely trained before crystallize(), or the EWC penalty was never
# actually added to the optimized loss when closure was on, or the ARM_0 "control"
# carried diversity floors (entropy_bonus / noise floor / E3 diversity) that
# prevented it from collapsing -- so the D1/D2 control arms were non-discriminative
# BY CONSTRUCTION and every run self-routed to non_contributory.
#
# 655 fixed all three inline (its `_assert_fixes_wired` preflight), but that block
# lives in one script: the NEXT MECH-334 retest is a copy-and-modify of 655 and can
# silently re-introduce the no-op (a stale claim_ids / a dropped assertion / an
# ARM_0 that quietly turns a floor back on). These guards extract the three checks
# into the shared harness so the retest INHERITS them by import and cannot ship a
# no-op without a guard firing.
#
# Convention: a guard FAILURE means the EXPERIMENT IS MISWIRED -- it would produce
# a vacuous result -- so the run must NOT proceed. The guard raises HarnessGuardError
# (an AssertionError subclass, distinct from P0NotReady's scientific self-route).
# Call them in a fresh / dedicated-agent preflight BEFORE the real arms run.


class HarnessGuardError(AssertionError):
    """Raised by a pre-run harness guard when an experiment WIRING precondition is
    unmet -- i.e. the experiment as configured would silently produce a no-op /
    vacuous result (the 610c-655 crystallization-no-op family).

    Distinct from P0NotReady: P0NotReady is a *scientific* readiness self-route
    (the substrate is honestly not trained enough; the run writes a
    substrate_not_ready_requeue manifest). HarnessGuardError is a *wiring bug* --
    the loss is mis-built or the control arm is mis-configured -- and the run must
    be fixed, not requeued. Let it propagate; do NOT catch-and-continue.
    """


def assert_policy_trained(
    params,
    pre_train_snapshot,
    *,
    grad_seen: Optional[bool] = None,
    min_weight_delta: float = 1e-4,
    trained_action_entropy: Optional[float] = None,
    untrained_entropy_ceiling: Optional[float] = None,
    label: str = "policy",
) -> Dict[str, Any]:
    """Guard (1): assert the policy was GENUINELY TRAINED (non-trivial weight delta)
    BEFORE crystallize().

    The 610c/610d no-op signature: crystallize() fired on a policy whose parameters
    never moved, so there was no learned distribution for crystallization to
    preserve and D1 (crystallization-preserves-diversity) was unreadable.

    `params` is the live list of policy parameters (e.g.
    [p for p in agent.gated_policy.parameters() if p.requires_grad]); ``pre_train_snapshot``
    is the matching list of detached clones taken BEFORE the training loop
    (``[p.detach().clone() for p in params]``). The guard computes the total L1
    weight movement and requires it to exceed ``min_weight_delta``.

    Optional stronger checks (mirroring 655's FIX 1):
      - ``grad_seen``: if provided, must be True (a non-zero gradient was observed
        during training). False/None-with-no-movement is the dead-policy signature.
      - ``trained_action_entropy`` + ``untrained_entropy_ceiling``: if both provided,
        require trained_action_entropy < untrained_entropy_ceiling (the policy learned
        a NON-UNIFORM action distribution -- e.g. 655's UNTRAINED_BAND_LOW=1.04 below
        ln(5)). A trained weight delta with a still-uniform action distribution is a
        weaker but real no-op.

    Returns a diagnostics dict (merge into fix_verification). Raises HarnessGuardError
    on any failed check.
    """
    params = list(params)
    pre = list(pre_train_snapshot)
    if not params:
        raise HarnessGuardError(
            f"[{label}] assert_policy_trained: empty parameter list -- nothing to "
            f"train (no requires_grad params? wrong module?). This is the "
            f"610c/610d untrained-policy signature.")
    if len(params) != len(pre):
        raise HarnessGuardError(
            f"[{label}] assert_policy_trained: param/snapshot length mismatch "
            f"({len(params)} vs {len(pre)}) -- snapshot was taken over a different "
            f"parameter set than the one trained.")
    weight_delta = 0.0
    for p, p0 in zip(params, pre):
        weight_delta += float((p.detach() - p0.detach()).abs().sum().item())
    n_params = int(sum(p.numel() for p in params))

    trained = weight_delta >= float(min_weight_delta)
    out: Dict[str, Any] = {
        "policy_trained": bool(trained),
        "policy_weight_delta": weight_delta,
        "policy_n_params": n_params,
        "policy_min_weight_delta": float(min_weight_delta),
    }
    if grad_seen is not None:
        out["policy_grad_seen"] = bool(grad_seen)
    if trained_action_entropy is not None:
        out["policy_trained_action_entropy"] = float(trained_action_entropy)
        out["policy_untrained_entropy_ceiling"] = (
            None if untrained_entropy_ceiling is None
            else float(untrained_entropy_ceiling))

    if not trained:
        raise HarnessGuardError(
            f"[{label}] assert_policy_trained FAILED: weight delta "
            f"{weight_delta:.6g} < min {float(min_weight_delta):.6g} over "
            f"{n_params} params -- the policy did NOT move before crystallize(). "
            f"This is the 610c/610d harness no-op (crystallizing an untrained "
            f"policy). Do NOT queue.")
    if grad_seen is not None and not grad_seen:
        raise HarnessGuardError(
            f"[{label}] assert_policy_trained FAILED: grad_seen=False -- the "
            f"policy params never received a non-zero gradient (the optimizer "
            f"stepped over a detached / disconnected loss). Do NOT queue.")
    if (trained_action_entropy is not None
            and untrained_entropy_ceiling is not None
            and not (float(trained_action_entropy)
                     < float(untrained_entropy_ceiling))):
        raise HarnessGuardError(
            f"[{label}] assert_policy_trained FAILED: trained action entropy "
            f"{float(trained_action_entropy):.4f} is NOT below the untrained band "
            f"edge {float(untrained_entropy_ceiling):.4f} -- the policy weights "
            f"moved but the action distribution stayed ~uniform (no learned "
            f"preference). Do NOT queue.")
    return out


def assert_ewc_penalty_live(
    residue_field,
    *,
    perturb: bool = True,
    perturb_scale: float = 0.5,
    min_penalty: float = 1e-8,
    label: str = "ewc",
) -> Dict[str, Any]:
    """Guard (2a): assert the EWC penalty is a LIVE, DIFFERENTIABLE term when closure
    is on -- i.e. it can actually contribute to the optimized loss.

    The 610c/610d no-op signature for closure: ``ewc_penalty()`` returned exactly 0
    (anchor never snapshotted, or residue_ewc_lambda left at 0) so adding it to the
    loss was a no-op and MECH-334's write-protect was never exercised.

    Mirrors 655's FIX 3. Requires:
      - ``residue_field.ewc_anchored`` is True (snapshot_ewc_anchor() was called);
      - ``ewc_penalty()`` > ``min_penalty`` once the field differs from its anchor
        (the guard optionally PERTURBS the rbf weights to force a non-zero penalty,
        so it must be called on a FRESH / throwaway agent, never the training agent);
      - the penalty back-propagates a non-zero gradient onto the residue rbf params
        (centers + weights) -- proving it is a real optimization target.

    Returns a diagnostics dict. Raises HarnessGuardError on any failed check.
    """
    if not getattr(residue_field, "ewc_anchored", False):
        raise HarnessGuardError(
            f"[{label}] assert_ewc_penalty_live FAILED: residue_field.ewc_anchored "
            f"is False -- snapshot_ewc_anchor() was not called (or EWC is not armed: "
            f"check crystallize_at_phase3 -> ewc_enabled / residue_ewc_lambda). The "
            f"EWC penalty cannot be in the loss if no anchor exists. Do NOT queue.")
    rbf = residue_field.rbf_field
    if perturb:
        # Force the field off its anchor so the penalty is provably non-zero even on
        # a just-snapshotted field. Mutates rbf weights -> use a throwaway agent.
        with torch.no_grad():
            rbf.weights.add_(float(perturb_scale) * rbf.active_mask.float())

    penalty = residue_field.ewc_penalty()
    pv = float(penalty.detach().item())
    if not (pv > float(min_penalty)):
        raise HarnessGuardError(
            f"[{label}] assert_ewc_penalty_live FAILED: ewc_penalty()={pv:.6g} is "
            f"not > {float(min_penalty):.6g} after anchoring"
            + (" + perturbation" if perturb else "")
            + " -- the penalty is inert (residue_ewc_lambda=0 / anchor==current). "
              "Adding it to the loss is a no-op. Do NOT queue.")

    res_params = [rbf.centers, rbf.weights]
    for p in res_params:
        if getattr(p, "grad", None) is not None:
            p.grad = None
    penalty.backward()
    res_grad = float(sum(
        p.grad.abs().sum().item() for p in res_params if p.grad is not None))
    if not (res_grad > 0.0):
        raise HarnessGuardError(
            f"[{label}] assert_ewc_penalty_live FAILED: ewc_penalty().backward() "
            f"produced no gradient on the residue rbf params (grad_sum={res_grad:.6g}) "
            f"-- the penalty is not a real optimization target. Do NOT queue.")
    return {
        "ewc_penalty_live": True,
        "ewc_penalty_value": pv,
        "ewc_residue_grad_sum": res_grad,
    }


def assert_ewc_term_in_loss(
    loss_without_ewc,
    ewc_term,
    total_loss,
    *,
    atol: float = 1e-5,
    label: str = "ewc",
) -> Dict[str, Any]:
    """Guard (2b): assert the EWC penalty is ACTUALLY ADDED to the optimized loss --
    the loss-construction-site check that catches "penalty computed but dropped".

    Call this at the loss-summation site inside the training step (where closure is
    on), passing the loss BEFORE the EWC add, the EWC term itself, and the resulting
    total. Catches BOTH 610c/610d failure modes at the point they happen:
      - ``ewc_term`` is ~0 (the penalty was inert -> nothing real was added);
      - ``total_loss`` does not actually include the term (it was computed into a
        local and then forgotten, so total == loss_without_ewc).

    All three args are scalars (tensors or floats). Raises HarnessGuardError if the
    term is non-positive OR if total_loss is not (loss_without_ewc + ewc_term)
    within ``atol`` AND distinct from loss_without_ewc.
    """
    def _f(x) -> float:
        return float(x.detach().item()) if hasattr(x, "detach") else float(x)

    l0 = _f(loss_without_ewc)
    et = _f(ewc_term)
    lt = _f(total_loss)
    out = {
        "ewc_term_in_loss": True,
        "ewc_term_value": et,
        "loss_without_ewc": l0,
        "total_loss": lt,
    }
    if not (et > 0.0):
        raise HarnessGuardError(
            f"[{label}] assert_ewc_term_in_loss FAILED: ewc_term={et:.6g} is not "
            f"> 0 -- the penalty was inert at the loss site (the 610c/610d "
            f"closure no-op). Do NOT queue.")
    if abs(lt - (l0 + et)) > float(atol):
        raise HarnessGuardError(
            f"[{label}] assert_ewc_term_in_loss FAILED: total_loss={lt:.6g} != "
            f"loss_without_ewc({l0:.6g}) + ewc_term({et:.6g}) within atol="
            f"{float(atol):.3g} -- the EWC term was NOT added to the optimized "
            f"loss (computed but dropped). Do NOT queue.")
    if abs(lt - l0) <= float(atol):
        raise HarnessGuardError(
            f"[{label}] assert_ewc_term_in_loss FAILED: total_loss is "
            f"indistinguishable from loss_without_ewc (delta {abs(lt - l0):.6g} "
            f"<= atol {float(atol):.3g}) -- the EWC penalty made no difference to "
            f"the loss. Do NOT queue.")
    return out


# Canonical ARM_0 true-negative control config: a no-closure arm with every
# diversity floor OFF, so it can MEASURABLY COLLAPSE its action diversity under
# post-Phase-3 pressure (the D2 precondition). Keys map to the 655-lineage arm-config
# schema. assert_true_negative_arm0() validates an arm_config against this.
TRUE_NEGATIVE_ARM0_CONTRACT = {
    "crystallize": False,         # no closure -> nothing resists collapse
    "entropy_bonus_phase3": 0.0,  # no entropy-bonus diversity floor in Phase 3
    "use_noise_floor": False,     # MECH-313 exploration noise floor OFF
    "use_e3_diversity": False,    # MECH-341 E3 score-diversity floor OFF
}


def assert_true_negative_arm0(
    arm_config: Dict[str, Any],
    *,
    label: str = "ARM_0",
) -> Dict[str, Any]:
    """Guard (3): assert the ARM_0 control is a TRUE NEGATIVE -- no closure AND every
    diversity floor OFF -- so a no-closure arm can measurably collapse under
    post-Phase-3 pressure (the D2 control-collapse precondition, delta >= +0.10).

    The 610e no-op signature: ARM_0 quietly carried structured-curiosity / a noise
    floor / E3 diversity, so it never collapsed and D1 (crystallization preserves)
    had no contrast to measure -- a confounded control.

    Validates the arm_config against TRUE_NEGATIVE_ARM0_CONTRACT: crystallize=False,
    entropy_bonus_phase3==0.0, use_noise_floor=False, use_e3_diversity=False. Any
    floor left on is a confound. Returns diagnostics; raises HarnessGuardError listing
    every violation.
    """
    violations = []
    for key, want in TRUE_NEGATIVE_ARM0_CONTRACT.items():
        if key not in arm_config:
            violations.append(f"{key} MISSING (must be {want!r})")
            continue
        got = arm_config[key]
        if key == "entropy_bonus_phase3":
            if abs(float(got)) > 1e-12:
                violations.append(f"{key}={got!r} (must be 0.0 -- entropy floor on)")
        elif bool(got) != bool(want):
            violations.append(f"{key}={got!r} (must be {want!r} -- floor on)")
    out = {
        "arm0_is_true_negative": not violations,
        "arm0_violations": violations,
    }
    if violations:
        raise HarnessGuardError(
            f"[{label}] assert_true_negative_arm0 FAILED: the control arm is NOT a "
            f"true negative -- it carries diversity floors that prevent collapse, so "
            f"D2 is non-discriminative by construction (the 610e confound): "
            + "; ".join(violations) + ". Do NOT queue.")
    return out


def assert_d2_control_collapsed(
    end_phase2_entropy,
    end_phase3_entropy,
    *,
    min_delta: float = 0.10,
    label: str = "ARM_0",
) -> Dict[str, Any]:
    """Companion to guard (3): the POST-run D2 acceptance check -- the true-negative
    control's action entropy must measurably COLLAPSE from its Phase-2 peak under
    post-Phase-3 pressure (delta = end_p2 - end_p3 >= min_delta, default +0.10).

    Pre-run, guard (3) (assert_true_negative_arm0) guarantees the control CAN collapse
    (no floors on). Post-run, this checks that it DID -- the D2 precondition without
    which D1 (crystallization preserves diversity) is unreadable. A FAIL here is NOT a
    wiring bug; it is a genuine substrate-incapacity finding (the 655 substrate_ceiling
    verdict). So this returns the verdict in the dict (d2_collapsed) AND raises only
    when ``min_delta`` is treated as a hard gate by the caller -- by default it RAISES
    so a misconfigured collapse cannot pass silently; pass a try/except at the call
    site if you want to route a genuine non-collapse to /failure-autopsy instead.
    """
    p2 = float(end_phase2_entropy)
    p3 = float(end_phase3_entropy)
    delta = p2 - p3
    out = {
        "d2_collapsed": delta >= float(min_delta),
        "d2_delta": delta,
        "d2_min_delta": float(min_delta),
        "arm0_end_phase2_entropy": p2,
        "arm0_end_phase3_entropy": p3,
    }
    if delta < float(min_delta):
        raise HarnessGuardError(
            f"[{label}] assert_d2_control_collapsed: D2 control-collapse delta "
            f"{delta:.4f} < min {float(min_delta):.4f} (end_p2 {p2:.4f} -> end_p3 "
            f"{p3:.4f}) -- the true-negative control did NOT collapse, so D1 is "
            f"unreadable. If the arm is verified-clean (guard 3 passed) this is a "
            f"genuine substrate-incapacity finding, NOT a wiring bug: route to "
            f"/failure-autopsy rather than re-queueing blind.")
    return out
