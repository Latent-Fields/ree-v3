"""
MECH-286: Override-gated sleep-state transition (wake-stability axis).

Recruitment authority for offline (sleep) mode entry. Sleep fires only when
all three joint conditions hold (sd_037_broadcast_override_regulator.md):

  override_signal < theta_sleep_permit
  AND max(MECH-284 region staleness) > theta_sleep_recruit
  AND z_harm_a tonic norm < threat_tonic_threshold (permissive threat)

Master switch: REEConfig.use_mech286_sleep_onset_gate (default False).
Hyperarousal lesion: pin override high -> sleep blocked despite staleness.
Narcolepsy lesion: override ~0 -> sleep may intrude without staleness demand.

MECH-094: evaluated at episode boundary on waking state only (called from
SleepLoopManager.notify_episode_end, not from replay/simulation paths).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from ree_core.agent import REEAgent


@dataclass
class SleepOnsetGateConfig:
    """MECH-286 threshold knobs (mirrored on REEConfig)."""

    theta_sleep_permit: float = 0.5
    theta_sleep_recruit: float = 0.3
    threat_tonic_threshold: float = 0.4


def _override_signal(agent: "REEAgent") -> float:
    reg = getattr(agent, "broadcast_override", None)
    if reg is None:
        return 0.0
    return float(reg.override_signal)


def _max_staleness(agent: "REEAgent") -> float:
    hippocampal = getattr(agent, "hippocampal", None)
    if hippocampal is None:
        return 0.0
    acc = getattr(hippocampal, "staleness_accumulator", None)
    if acc is None:
        return 0.0
    snap = acc.snapshot()
    if not snap:
        return 0.0
    return float(max(snap.values()))


def _z_harm_a_tonic_norm(agent: "REEAgent") -> float:
    latent = getattr(agent, "_current_latent", None)
    if latent is None:
        return 0.0
    z = getattr(latent, "z_harm_a", None)
    if z is None:
        return 0.0
    try:
        import torch

        if not isinstance(z, torch.Tensor) or z.numel() == 0:
            return 0.0
        return float(z.norm().item())
    except Exception:
        return 0.0


def evaluate_sleep_onset_permit(
    agent: "REEAgent",
    config: SleepOnsetGateConfig | None = None,
) -> Tuple[bool, Dict[str, float]]:
    """Return (permitted, diagnostics) for sleep-mode entry at episode end."""
    cfg = config or SleepOnsetGateConfig(
        theta_sleep_permit=float(
            getattr(agent.config, "mech286_theta_sleep_permit", 0.5)
        ),
        theta_sleep_recruit=float(
            getattr(agent.config, "mech286_theta_sleep_recruit", 0.3)
        ),
        threat_tonic_threshold=float(
            getattr(agent.config, "mech286_threat_tonic_threshold", 0.4)
        ),
    )

    override = _override_signal(agent)
    staleness_max = _max_staleness(agent)
    harm_a_norm = _z_harm_a_tonic_norm(agent)

    override_ok = override < float(cfg.theta_sleep_permit)
    staleness_ok = staleness_max > float(cfg.theta_sleep_recruit)
    threat_ok = harm_a_norm < float(cfg.threat_tonic_threshold)

    permitted = override_ok and staleness_ok and threat_ok

    diagnostics = {
        "mech286_override_signal": override,
        "mech286_staleness_max": staleness_max,
        "mech286_z_harm_a_norm": harm_a_norm,
        "mech286_override_ok": 1.0 if override_ok else 0.0,
        "mech286_staleness_ok": 1.0 if staleness_ok else 0.0,
        "mech286_threat_ok": 1.0 if threat_ok else 0.0,
        "mech286_theta_sleep_permit": float(cfg.theta_sleep_permit),
        "mech286_theta_sleep_recruit": float(cfg.theta_sleep_recruit),
        "mech286_threat_tonic_threshold": float(cfg.threat_tonic_threshold),
        "mech286_sleep_permitted": 1.0 if permitted else 0.0,
    }
    return permitted, diagnostics
