"""
Canonical metric extractors for ree-v3 experiment scripts.

Each extractor is a pure function: take ``agent`` (and sometimes a few
per-tick scalars) and return a dict the script can merge into per_seed_results.

These exist to avoid the kind of measurement bug found in EXQ-490c/490e
where the script read ``agent.dacc._last_bundle.get('mode_ev')`` and
reported its norm as "dacc_score_bias_mean" -- but score_bias is the [K]
tensor that DACCtoE3Adapter produces from the bundle, not the raw mode_ev
slice. Plus a ``try/except: norm=0.0`` was hiding shape mismatches.

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

from typing import Any, Dict, Optional

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

    The score_bias is cached on the agent in ``_last_dacc_score_bias`` if
    present; otherwise we recompute via the adapter using the last bundle.
    """
    dacc = getattr(agent, "dacc", None)
    if dacc is None:
        return None
    cached = getattr(agent, "_last_dacc_score_bias", None)
    if cached is not None:
        return cached
    bundle = getattr(dacc, "_last_bundle", None)
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
    """
    groups = list(groups)
    if not groups:
        return True, "no groups"
    reasons = []
    for i, g in enumerate(groups):
        is_deg, reason = metric_is_degenerate(
            g, eps=eps, floor=floor, ceiling=ceiling)
        if not is_deg:
            return False, ""
        reasons.append(f"group[{i}]: {reason}")
    return True, "every group pinned -- " + "; ".join(reasons)


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
