"""SD-049 Phase 3 helper: per-axis-drive vector accessor.

Shared, validated collapse-to-scalar helper for the seven SD-032 consumer
modules migrated by SD-049 Phase 3 (AIC, PCC, pACC, dACC, SalienceCoordinator,
BroadcastOverrideRegulator, MECH-295 liking-bridge). Each consumer either
collapses the per-axis vector into a scalar effective drive via a per-consumer
combiner (whole-organism control modules) or routes by axis index (MECH-295,
the axis-matched approach-cue case).

Schema (matches CausalGridWorldV2 obs_dict["per_axis_drive"]):
    Sequence of floats in [0, 1], length == n_resource_types.
    Index i corresponds to resource_type_drive_axes[i] (default order:
    hunger, thirst, curiosity for the default 3-type config).
    0.0 = sated, 1.0 = fully depleted on that axis.

Combiner modes:
    "max"  -- most-pressing-axis reading (default for whole-organism
              control consumers: AIC urgency, dACC control demand,
              SalienceCoordinator external-task affinity, BroadcastOverride
              recruitment, PAG freeze-gate, MECH-295 fallback when
              goal_axis_idx is None). Biologically: orexin / locus
              coeruleus recruit on the worst-deficit axis.
    "mean" -- average-deficit reading. Used as the PCC fatigue collapse
              default (whole-organism fatigue is the integration over axes,
              not the peak alarm).
    "sum"  -- accumulated-deficit reading (allostatic-load style). Used by
              pACC as default since multiple sustained deficits compound
              the autonomic write-back signal.

Bit-identical OFF guarantee: when SD-049 per-axis is disabled at the env
level (multi_resource_heterogeneity_enabled=False or per_axis_drive_enabled
=False), the env does not surface obs_dict["per_axis_drive"]; the agent
passes per_axis_drive=None to each consumer; each consumer's per_axis_drive
branch is skipped and the legacy scalar drive_level path is taken
unchanged.

See REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
(Phase 3 section) for the SD-032 consumer cascade design.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch


PerAxisDriveLike = Union[Sequence[float], np.ndarray, torch.Tensor]

_VALID_COMBINERS = ("max", "mean", "sum")


def _to_float_list(per_axis_drive: PerAxisDriveLike) -> list:
    """Normalise a per-axis drive container to list[float].

    Accepts python sequences, numpy arrays, and torch tensors. Returns
    an empty list on a length-zero input.
    """
    if isinstance(per_axis_drive, torch.Tensor):
        return [float(v) for v in per_axis_drive.detach().cpu().reshape(-1).tolist()]
    if isinstance(per_axis_drive, np.ndarray):
        return [float(v) for v in per_axis_drive.reshape(-1).tolist()]
    return [float(v) for v in per_axis_drive]


def collapse_per_axis_drive(
    per_axis_drive: Optional[PerAxisDriveLike],
    mode: str = "max",
) -> Optional[float]:
    """Collapse a per-axis drive vector to a scalar via the chosen combiner.

    Returns None when per_axis_drive is None or empty (caller falls back to
    the legacy scalar drive_level path). Otherwise returns a float in
    [0, 1] (computed value is clipped to that range so callers can use it
    as a drive_level surrogate without further bounds-checking).

    Raises ValueError on an unrecognised combiner mode.
    """
    if per_axis_drive is None:
        return None
    vals = _to_float_list(per_axis_drive)
    if not vals:
        return None
    if mode not in _VALID_COMBINERS:
        raise ValueError(
            f"per_axis_drive combiner '{mode}' not in {_VALID_COMBINERS}"
        )
    if mode == "max":
        out = max(vals)
    elif mode == "mean":
        out = sum(vals) / float(len(vals))
    else:  # mode == "sum"
        out = sum(vals)
    if out < 0.0:
        out = 0.0
    elif out > 1.0:
        out = 1.0
    return out


def select_axis(
    per_axis_drive: Optional[PerAxisDriveLike],
    axis_idx: Optional[int],
) -> Optional[float]:
    """Return per_axis_drive[axis_idx] in [0, 1], or None on missing inputs.

    Used by MECH-295 liking-bridge for axis-matched routing: the goal's
    current resource-type axis index selects the deficit channel that
    drives the anticipatory-liking write and the per-candidate approach
    cue.

    Returns None if per_axis_drive is None / empty / axis_idx is None /
    out-of-range. The caller falls back to the legacy scalar drive path
    in that case.
    """
    if per_axis_drive is None or axis_idx is None:
        return None
    vals = _to_float_list(per_axis_drive)
    if not vals:
        return None
    if axis_idx < 0 or axis_idx >= len(vals):
        return None
    v = vals[int(axis_idx)]
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return v


def validate_combiner(mode: str) -> str:
    """Raise ValueError if mode is not a recognised combiner; return mode."""
    if mode not in _VALID_COMBINERS:
        raise ValueError(
            f"per_axis_drive combiner '{mode}' not in {_VALID_COMBINERS}"
        )
    return mode
