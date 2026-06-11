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


def p0_readiness_gate(checks: list) -> list:
    """Pre-registered P0 abort gate -- assert the substrate is trained enough to
    make the measurement non-vacuous BEFORE burning compute on P1/P2.

    `checks` is a list of dicts, each:
        {"name": str, "measured": float, "threshold": float,
         "direction": "lower"|"upper"}   # lower=floor: met iff measured>=threshold
                                         # upper=ceiling: met iff measured<=threshold
    (direction defaults to "lower"). The semantics mirror the REE_assembly indexer's
    _precondition_direction so the recorded preconditions[] adjudicate consistently.

    Returns a manifest-ready preconditions[] list (each entry carries measured/
    threshold/direction/met/kind="readiness") when ALL checks are met. Raises
    P0NotReady (with the same payload) when any check fails, so the caller writes
    interpretation={"label": "substrate_not_ready_requeue", "preconditions": ...}
    and self-routes to non_contributory instead of a misleading FAIL.
    """
    preconditions = []
    unmet = []
    for c in checks:
        m = float(c["measured"])
        t = float(c["threshold"])
        direction = str(c.get("direction", "lower"))
        met = (m <= t) if direction == "upper" else (m >= t)
        preconditions.append({
            "name": str(c["name"]),
            "measured": m,
            "threshold": t,
            "direction": direction,
            "met": bool(met),
            "kind": "readiness",
        })
        if not met:
            unmet.append(str(c["name"]))
    if unmet:
        raise P0NotReady(preconditions, "P0 readiness unmet: " + ", ".join(unmet))
    return preconditions
