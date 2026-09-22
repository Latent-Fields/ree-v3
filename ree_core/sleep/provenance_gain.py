"""
SD-PP-4 `sleep.provenance_conditioned_consolidation_gain` -- the RULE.

Contract: `REE_assembly/docs/architecture/precision_provenance_substrate_spec.md`
section 5. This module is the pure-arithmetic half of SD-PP-4; the other two
halves are the `module_step_scale` hook in
`ree_core/sleep/cross_module_consolidation.py` and the `reduction=` kwarg on
`ree_core/predictors/e2_fast.py::world_forward_contrastive_loss`.

WHY A GAIN AT ALL (MECH-572)
----------------------------
`CrossModuleConsolidator.consolidate()` builds a FRESH `torch.optim.Adam` per
module per call. A fresh Adam's bias-corrected first step is ~`lr * sign(g)`
REGARDLESS of `|g|`, so a per-transition loss weight is normalised away in
MAGNITUDE -- it changes only the DIRECTION of the step. MECH-572 measured the
displacement pinned at that Adam bound in 6/6 cells. So the gain is applied in
two places, both pre-registered:

  (a) per-row loss weights `sum_i g_i l_i / sum_i g_i`  -> sets the DIRECTION
      (`weighted_row_loss` below);
  (b) the module's lr for that step scaled by `mean_i g_i`  -> moves the
      DISPLACEMENT (the consolidator's `module_step_scale` hook).

Replay CONTENT, ORDER and COUNT are untouched: the same `randperm` draw, the
same K rows, the same 8 steps. Nothing here draws RNG.

THE RULE (spec section 5; constants provisional until the preregistration's
freeze record). For row `i` with provenance packet `p_i`, current global
epistemic precision `pi_cur` (SD-PP-2 `current_read().pi_epi` at sleep entry):

    K_i    = p.evidence_precision_z / (p.evidence_precision_z + pi_cur)
             Kalman-form write authority in (0,1): reliable evidence vs the
             belief currently held.
    m_i    = sqrt( max(p.pe - noise_gain * p.evidence_variance_z, 0) / v_ref )
             epistemic innovation MAGNITUDE -- precisely the information a
             fresh Adam discards.
    r_i    = min( reopen_max, 1 + surprise_beta * max(0, ln(p.surprise)) )
             reopen factor. HISTORICAL precision enters ONLY here, and only as
             an interpreter of the surprise.
    gain_i = clip( gain_max * K_i * m_i * r_i, gain_min, gain_max )

ANTI-SELF-SEALING is structural, not incidental: `r_i >= 1` always, so a
confidently-held prediction that is reliably falsified is never PROTECTED by
having been confident. Historical precision can never appear as a multiplier
below 1.

PACKETS ARE DUCK-TYPED. This module never imports
`ree_core.hippocampal.replay_provenance`; it reads only
`.evidence_precision_z`, `.evidence_variance_z`, `.pe`, `.surprise`,
`.has_prev`. That keeps the rule testable with a `types.SimpleNamespace` and
keeps SD-PP-3 and SD-PP-4 independently landable.

Default OFF: `ProvenanceGainConfig.use_provenance_conditioned_consolidation_gain`
is False and the agent constructs nothing, so the default sleep pipeline is
bit-identical by STRUCTURAL ABSENCE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


GAIN_MODES = ("provenance", "provenance_nohist", "residual_only", "global")


@dataclass
class ProvenanceGainConfig:
    """Configuration for the provenance-conditioned consolidation gain.

    mode:
        "provenance"        -- the full rule (ARM C).
        "provenance_nohist" -- the rule with r_i == 1, i.e. historical model
            precision removed (ARM C-nohist; intake F1 asks whether historical
            precision is load-bearing at all).
        "residual_only"     -- g_i = clip(gain_max * sqrt(pe_cur_i / v_ref),
            gain_min, gain_max) from the CURRENT per-row residual of the head on
            the replayed triple (ARM D-residual: C's magnitude factor with
            K = 1, r = 1 and the CURRENT rather than the STORED innovation; no
            packet, no precision term, no global_scale). Redesigned pre-freeze
            2026-09-22 (V3-EXQ-1073 freeze record item 6): the earlier
            budget-matched form inherited a pooled global budget and could not
            reallocate across epistemic regimes, so it was not the current-
            residual rival the design needs.
        "global"            -- g_i = global_scale (ARM D-global: matched budget
            carrying no per-row information).
    gain_min / gain_max:
        Clip bounds for the "provenance" family. `gain_max` is also the rule's
        overall scale (see the formula: the product is multiplied by gain_max).
    surprise_beta / reopen_max:
        Reopen-factor slope and ceiling.
    v_ref:
        Reference innovation variance normalising `m_i` (z units).
    noise_gain:
        Multiplier converting per-frame z-noise variance into PE noise
        variance. 2.0 because observation noise enters the PE through BOTH the
        input (z_t) and the target (z_{t+1}) -- see SD-PP-2.
    global_scale:
        Budget for the two control modes.
    """

    use_provenance_conditioned_consolidation_gain: bool = False
    mode: str = "provenance"
    gain_min: float = 0.02
    gain_max: float = 2.0
    surprise_beta: float = 0.5
    reopen_max: float = 3.0
    v_ref: float = 1e-2
    noise_gain: float = 2.0
    global_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in GAIN_MODES:
            raise ValueError(
                f"mode must be one of {GAIN_MODES}; got {self.mode!r}"
            )
        if not (self.gain_min > 0.0):
            raise ValueError(f"gain_min must be > 0; got {self.gain_min}")
        if not (self.gain_min <= self.gain_max):
            raise ValueError(
                "gain_min must be <= gain_max; got "
                f"gain_min={self.gain_min}, gain_max={self.gain_max}"
            )
        if not (self.v_ref > 0.0):
            raise ValueError(f"v_ref must be > 0; got {self.v_ref}")
        if not (self.reopen_max >= 1.0):
            raise ValueError(f"reopen_max must be >= 1; got {self.reopen_max}")


def _finite(value: Any) -> bool:
    """True iff `value` coerces to a finite float."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _is_missing(packet: Any) -> bool:
    """A row has no usable provenance.

    Missing means: no packet at all, an episode-start placeholder
    (`has_prev` False), or any of the three numeric fields the rule needs being
    absent / non-finite (a nan `pe` is the canonical placeholder marker from
    SD-PP-3).
    """
    if packet is None:
        return True
    if not bool(getattr(packet, "has_prev", False)):
        return True
    if not _finite(getattr(packet, "pe", float("nan"))):
        return True
    if not _finite(getattr(packet, "evidence_precision_z", float("nan"))):
        return True
    if not _finite(getattr(packet, "evidence_variance_z", float("nan"))):
        return True
    return False


def _reopen_factor(surprise: Any, config: ProvenanceGainConfig) -> float:
    """r_i -- the reopen factor. Never below 1.0.

    `surprise <= 0` or non-finite is treated as r == 1 (no reopening claimed),
    which is also what `ln` would refuse to answer.
    """
    if not _finite(surprise):
        return 1.0
    s = float(surprise)
    if s <= 0.0:
        return 1.0
    r = 1.0 + float(config.surprise_beta) * max(0.0, math.log(s))
    return float(min(float(config.reopen_max), r))


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _max_or_nan(values: Sequence[float]) -> float:
    return float(max(values)) if values else float("nan")


def compute_provenance_gains(
    packets: Optional[Sequence[Any]],
    pi_cur: float,
    per_row_loss: Optional[torch.Tensor],
    config: ProvenanceGainConfig,
    per_row_residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-row consolidation gains plus a flat diagnostics dict.

    Args:
        packets: one entry per replay row, aligned with `per_row_loss`. An
            entry may be None (no packet for that row). Duck-typed: only
            `.evidence_precision_z`, `.evidence_variance_z`, `.pe`,
            `.surprise`, `.has_prev` are read. May be None entirely in the
            `residual_only` / `global` modes, in which case K is taken from
            `per_row_loss` and no row can be "missing".
        pi_cur: current global epistemic precision (SD-PP-2
            `current_read().pi_epi`), read at sleep entry.
        per_row_loss: [K] per-row current loss (InfoNCE CE). Used only to
            size K when `packets` is None. Always read detached.
        per_row_residual: [K] per-row CURRENT mean-squared residual of the
            head on the replayed triple, mean_d((world_forward(z0,a) - z1)^2).
            REQUIRED for `residual_only`; ignored by the other modes. Detached.
        config: `ProvenanceGainConfig`.

    Returns:
        (gains, diag)
        gains: float32 tensor [K], detached (`requires_grad` False).
        diag: flat float dict --
            gain_mean, gain_min, gain_max, gain_sd, n_missing,
            k_mean, m_mean, r_mean, r_max, surprise_max, pi_cur,
            mode (float index into GAIN_MODES).
            The K/m/r/surprise readouts are computed over the NON-MISSING rows
            only and are nan when there are none (including in the two control
            modes, which never evaluate the rule). `gain_sd` is the POPULATION
            sd, so a single row reports 0.0 rather than nan.

    Raises:
        ValueError: `residual_only` with `per_row_loss` None, or neither
            `packets` nor `per_row_loss` supplied (K undeterminable).

    No RNG is drawn.
    """
    mode = config.mode
    if mode not in GAIN_MODES:
        raise ValueError(f"mode must be one of {GAIN_MODES}; got {mode!r}")

    if mode == "residual_only" and per_row_residual is None:
        raise ValueError(
            "mode 'residual_only' requires per_row_residual (current per-row "
            "MSE of the head on the replayed triple); got None"
        )

    if packets is not None:
        n_rows = len(packets)
    elif per_row_loss is not None:
        n_rows = int(per_row_loss.numel())
    else:
        raise ValueError(
            "compute_provenance_gains needs packets or per_row_loss to "
            "determine K; both were None"
        )

    pi = float(pi_cur) if _finite(pi_cur) else 0.0

    # --- which rows have usable provenance ------------------------------
    # `global` is the information-free control: it never inspects a packet, so
    # by construction it reports no missing rows (spec section 5).
    if packets is None or mode == "global":
        missing = [False] * n_rows
    else:
        missing = [_is_missing(p) for p in packets]
    n_missing = sum(1 for flag in missing if flag)

    k_vals: List[float] = []
    m_vals: List[float] = []
    r_vals: List[float] = []
    s_vals: List[float] = []
    gains: List[float] = []

    if mode in ("provenance", "provenance_nohist"):
        use_hist = mode == "provenance"
        for idx in range(n_rows):
            if missing[idx]:
                gains.append(1.0)
                continue
            p = packets[idx]
            ev_prec = float(p.evidence_precision_z)
            ev_var = float(p.evidence_variance_z)
            pe = float(p.pe)

            denom = ev_prec + pi
            k_i = (ev_prec / denom) if denom > 0.0 else 0.0
            if not math.isfinite(k_i):
                k_i = 1.0 if ev_prec > 0.0 else 0.0
            k_i = min(max(k_i, 0.0), 1.0)

            innovation = max(pe - float(config.noise_gain) * ev_var, 0.0)
            m_i = math.sqrt(innovation / float(config.v_ref))
            if not math.isfinite(m_i):
                m_i = 0.0

            surprise = getattr(p, "surprise", float("nan"))
            r_i = _reopen_factor(surprise, config) if use_hist else 1.0

            raw = float(config.gain_max) * k_i * m_i * r_i
            if math.isnan(raw):
                g = float(config.gain_min)
            else:
                g = min(max(raw, float(config.gain_min)), float(config.gain_max))

            k_vals.append(k_i)
            m_vals.append(m_i)
            r_vals.append(r_i)
            if _finite(surprise):
                s_vals.append(float(surprise))
            gains.append(float(g))

    elif mode == "residual_only":
        res = per_row_residual.detach().reshape(-1).to(torch.float32)
        if int(res.numel()) != n_rows:
            raise ValueError(
                "per_row_residual length does not match the row count: "
                f"{int(res.numel())} vs {n_rows}"
            )
        for idx in range(n_rows):
            r_i = float(res[idx].item())
            if not math.isfinite(r_i) or r_i < 0.0:
                gains.append(1.0)
                continue
            m_i = math.sqrt(r_i / float(config.v_ref))
            g = float(config.gain_max) * m_i
            g = min(max(g, float(config.gain_min)), float(config.gain_max))
            gains.append(float(g))

    else:  # "global" -- matched budget, no per-row information
        gains = [float(config.global_scale)] * n_rows

    gains_t = torch.tensor(gains, dtype=torch.float32)
    gains_t.requires_grad_(False)

    if n_rows > 0:
        diag_gain_mean = float(gains_t.mean().item())
        diag_gain_min = float(gains_t.min().item())
        diag_gain_max = float(gains_t.max().item())
        # Population sd: K == 1 reports 0.0, not nan.
        diag_gain_sd = float(gains_t.std(unbiased=False).item())
    else:
        diag_gain_mean = float("nan")
        diag_gain_min = float("nan")
        diag_gain_max = float("nan")
        diag_gain_sd = float("nan")

    diag: Dict[str, float] = {
        "gain_mean": diag_gain_mean,
        "gain_min": diag_gain_min,
        "gain_max": diag_gain_max,
        "gain_sd": diag_gain_sd,
        "n_missing": float(n_missing),
        "k_mean": _mean_or_nan(k_vals),
        "m_mean": _mean_or_nan(m_vals),
        "r_mean": _mean_or_nan(r_vals),
        "r_max": _max_or_nan(r_vals),
        "surprise_max": _max_or_nan(s_vals),
        "pi_cur": float(pi),
        "mode": float(GAIN_MODES.index(mode)),
    }
    return gains_t, diag


def weighted_row_loss(
    per_row_loss: torch.Tensor,
    gains: torch.Tensor,
) -> torch.Tensor:
    """Gain-weighted mean of a per-row loss: `sum_i g_i l_i / sum_i g_i`.

    The gains are DETACHED before use, so the returned scalar carries gradient
    through `per_row_loss` only -- the gain is a weighting, never a trainable
    quantity. A degenerate (zero or non-finite) gain sum falls back to the
    plain mean so a consolidation step is never silently dropped.

    Args:
        per_row_loss: [K] per-row loss, typically
            `world_forward_contrastive_loss(..., reduction="none")`.
        gains: [K] from `compute_provenance_gains`.

    Returns:
        0-d tensor.
    """
    rows = per_row_loss.reshape(-1)
    g = gains.detach().reshape(-1).to(dtype=rows.dtype, device=rows.device)
    if int(g.numel()) != int(rows.numel()):
        raise ValueError(
            "gains and per_row_loss must have the same length; got "
            f"{int(g.numel())} vs {int(rows.numel())}"
        )
    denom = g.sum()
    denom_f = float(denom.item())
    if denom_f == 0.0 or not math.isfinite(denom_f):
        return rows.mean()
    return (g * rows).sum() / denom
