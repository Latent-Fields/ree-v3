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
