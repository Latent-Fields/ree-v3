"""
MECH-282: LPB interoceptive harm routing (lateral parabrachial analog).

Parallel upstream broadcast arm to VMHdm-style external-threat harm: routes
metabolic / visceral distress (resource depletion, homeostatic deviation,
limb-damage interoception) into a separable z_harm_intero channel while
z_harm (HarmEncoder output) carries external-threat-only harm_obs.

When coupled with SD-037 (BroadcastOverrideRegulator), override recruitment
uses drive_level + interoceptive magnitude; external-threat magnitude feeds
PAG freeze-gate duration instead of the mixed z_harm_a proxy.

Non-trainable routing (no new encoder head). MECH-094: simulation_mode=True
ticks return cached outputs without advancing internal state.

See REE_assembly/docs/architecture/sd_037_broadcast_override_regulator.md
section "MECH-282: LPB interoceptive routing into harm-arbitration".
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


@dataclass
class LPBInteroceptiveRoutingConfig:
    """MECH-282 configuration."""

    enabled: bool = True
    # harm_obs layout (CausalGridWorldV2 use_proxy_fields=True)
    hazard_field_dim: int = 25
    resource_field_dim: int = 25
    # Latent vector broadcast from scalar intero magnitude (non-trainable).
    intero_z_dim: int = 16
    drive_weight: float = 1.0
    resource_weight: float = 1.0
    # Normalisation: intero_raw clipped to [0, 1] after weighted sum.
    intero_norm_scale: float = 1.0
    external_norm_scale: float = 1.0


@dataclass
class LPBInteroceptiveRoutingOutput:
    """Per-tick MECH-282 routing outputs."""

    intero_magnitude: float = 0.0
    external_magnitude: float = 0.0
    z_harm_intero: Optional[torch.Tensor] = None


class LPBInteroceptiveRouter:
    """MECH-282 LPB interoceptive harm router.

    Public API:
      mask_external_harm_obs(harm_obs) -> Tensor
        Zero the resource-field slice so HarmEncoder sees hazard + exposure only.

      tick(drive_level, harm_obs, harm_obs_a, device, batch_size,
           simulation_mode=False) -> LPBInteroceptiveRoutingOutput
        Compute intero / external scalar magnitudes and z_harm_intero.

      reset()
        Clear per-episode diagnostics counters.

      diagnostics -> dict
    """

    def __init__(self, config: Optional[LPBInteroceptiveRoutingConfig] = None):
        self.config = config or LPBInteroceptiveRoutingConfig()
        self._n_ticks: int = 0
        self._last_intero: float = 0.0
        self._last_external: float = 0.0

    def reset(self) -> None:
        self._n_ticks = 0
        self._last_intero = 0.0
        self._last_external = 0.0

    def mask_external_harm_obs(self, harm_obs: torch.Tensor) -> torch.Tensor:
        """Return harm_obs with resource-field channels zeroed (VMHdm-only path)."""
        if harm_obs is None:
            return harm_obs
        ho = harm_obs.detach().float().clone()
        hz = int(self.config.hazard_field_dim)
        rz = int(self.config.resource_field_dim)
        if ho.dim() == 1:
            if ho.numel() >= hz + rz:
                ho[hz : hz + rz] = 0.0
        elif ho.dim() == 2 and ho.shape[1] >= hz + rz:
            ho[:, hz : hz + rz] = 0.0
        return ho

    @staticmethod
    def _tensor_mean(t: torch.Tensor) -> float:
        if t is None or t.numel() == 0:
            return 0.0
        return float(t.detach().float().mean().item())

    def _external_scalar(self, harm_obs: Optional[torch.Tensor]) -> float:
        if harm_obs is None:
            return 0.0
        ho = harm_obs.detach().float()
        hz = int(self.config.hazard_field_dim)
        rz = int(self.config.resource_field_dim)
        if ho.dim() == 1:
            n = ho.numel()
            if n >= hz + rz + 1:
                hazard = ho[:hz]
                exposure = ho[hz + rz : hz + rz + 1]
                parts = [hazard, exposure]
            elif n >= hz:
                parts = [ho[:hz]]
            else:
                parts = [ho]
        else:
            n = ho.shape[1]
            if n >= hz + rz + 1:
                hazard = ho[:, :hz]
                exposure = ho[:, hz + rz : hz + rz + 1]
                parts = [hazard, exposure]
            elif n >= hz:
                parts = [ho[:, :hz]]
            else:
                parts = [ho]
        val = sum(self._tensor_mean(p) for p in parts) / max(len(parts), 1)
        scale = max(1e-6, float(self.config.external_norm_scale))
        return max(0.0, min(1.0, val / scale))

    def _intero_scalar(
        self,
        drive_level: float,
        harm_obs_a: Optional[torch.Tensor],
    ) -> float:
        drive = max(0.0, min(1.0, float(drive_level)))
        resource_signal = 0.0
        if harm_obs_a is not None:
            ha = harm_obs_a.detach().float()
            hz = int(self.config.hazard_field_dim)
            rz = int(self.config.resource_field_dim)
            if ha.dim() == 1:
                n = ha.numel()
                if n >= hz + rz:
                    resource_signal = self._tensor_mean(ha[hz : hz + rz])
                else:
                    resource_signal = self._tensor_mean(ha)
            else:
                n = ha.shape[1]
                if n >= hz + rz:
                    resource_signal = self._tensor_mean(ha[:, hz : hz + rz])
                else:
                    resource_signal = self._tensor_mean(ha)
        raw = (
            float(self.config.drive_weight) * drive
            + float(self.config.resource_weight) * resource_signal
        )
        scale = max(1e-6, float(self.config.intero_norm_scale))
        return max(0.0, min(1.0, raw / scale))

    def _build_z_harm_intero(
        self,
        intero_mag: float,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        dim = max(1, int(self.config.intero_z_dim))
        return torch.full(
            (batch_size, dim),
            float(intero_mag),
            dtype=torch.float32,
            device=device,
        )

    def tick(
        self,
        drive_level: float,
        harm_obs: Optional[torch.Tensor],
        harm_obs_a: Optional[torch.Tensor],
        device: torch.device,
        batch_size: int = 1,
        simulation_mode: bool = False,
    ) -> LPBInteroceptiveRoutingOutput:
        if not self.config.enabled:
            z = self._build_z_harm_intero(0.0, batch_size, device)
            return LPBInteroceptiveRoutingOutput(
                intero_magnitude=0.0,
                external_magnitude=0.0,
                z_harm_intero=z,
            )

        if simulation_mode:
            z = self._build_z_harm_intero(self._last_intero, batch_size, device)
            return LPBInteroceptiveRoutingOutput(
                intero_magnitude=self._last_intero,
                external_magnitude=self._last_external,
                z_harm_intero=z,
            )

        self._n_ticks += 1
        intero = self._intero_scalar(drive_level, harm_obs_a)
        external = self._external_scalar(harm_obs)
        self._last_intero = intero
        self._last_external = external
        z = self._build_z_harm_intero(intero, batch_size, device)
        return LPBInteroceptiveRoutingOutput(
            intero_magnitude=intero,
            external_magnitude=external,
            z_harm_intero=z,
        )

    @property
    def diagnostics(self) -> Dict[str, object]:
        return {
            "n_ticks": int(self._n_ticks),
            "last_intero_magnitude": float(self._last_intero),
            "last_external_magnitude": float(self._last_external),
            "intero_z_dim": int(self.config.intero_z_dim),
        }
