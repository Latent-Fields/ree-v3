"""
ThetaBuffer — Cross-Rate Integration for SD-006 (MECH-089)

E3 does NOT receive raw E1/E2 output. It receives temporally-abstracted
theta-cycle summaries of z_world.

MECH-089: After each E1 step, z_world estimate is pushed into the rolling
theta buffer. When the E3 heartbeat fires (every N_e3 env steps), E3 samples
the buffer summary — a theta-cycle-averaged world state — rather than the
instantaneous E1 output.

This implements the neural analogue:
  - Gamma-frequency updates (E1/E2 rate): individual z_world estimates
  - Theta-cycle summaries: averaged z_world over theta_buffer_size steps
  - E3 (delta/beta rate): consumes theta-cycle averages only

The buffer also stores recent z_self estimates for E2 to consume.
"""

from typing import Optional, List
from collections import deque

import torch


class ThetaBuffer:
    """
    Rolling theta-cycle buffer for cross-rate integration.

    Tracks the last theta_buffer_size z_world (and z_self) estimates
    from E1. E3 calls summary() at its heartbeat tick to get the
    theta-cycle-averaged world state.

    MECH-089 invariant: E3 never sees raw z_world from a single E1 step.
    It always receives a theta-cycle summary.
    """

    def __init__(
        self,
        self_dim: int,
        world_dim: int,
        buffer_size: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.self_dim = self_dim
        self.world_dim = world_dim
        self.buffer_size = buffer_size
        self.device = device or torch.device("cpu")

        # Rolling buffers — deque for O(1) append/pop
        self._z_self_buffer:  deque = deque(maxlen=buffer_size)
        self._z_world_buffer: deque = deque(maxlen=buffer_size)

        # Pointer to the most recent summary (for replay — MECH-092)
        self._last_summary_z_world: Optional[torch.Tensor] = None
        self._last_summary_z_self:  Optional[torch.Tensor] = None

        # MECH-122 / Phase 3 spindle coordination: consolidation mode flag.
        # When True, the buffer is in offline consolidation mode (cx->hip direction).
        # set_consolidation_mode(True) before Phase 3 offline pass.
        # consolidation_summary() returns reverse-temporal-order packaged E1 updates
        # for hippocampal transfer (bidirectional theta-packaging proxy).
        self._consolidation_mode: bool = False

    def update(
        self,
        z_world: torch.Tensor,
        z_self: torch.Tensor,
    ) -> None:
        """
        Push new E1 estimates into the buffer.

        Called once per E1 tick (every env step).

        Args:
            z_world: z_world estimate from E1 [batch, world_dim]
            z_self:  z_self estimate from E1  [batch, self_dim]
        """
        self._z_world_buffer.append(z_world.detach())
        self._z_self_buffer.append(z_self.detach())

    def summary(self) -> torch.Tensor:
        """
        Compute theta-cycle-averaged z_world summary for E3.

        Called at each E3 heartbeat tick (MECH-089).
        Returns the mean of all z_world estimates in the current buffer.

        Returns:
            Averaged z_world [batch, world_dim]
        """
        if not self._z_world_buffer:
            batch_size = 1
            result = torch.zeros(batch_size, self.world_dim, device=self.device)
        else:
            stacked = torch.stack(list(self._z_world_buffer), dim=0)  # [T, batch, world_dim]
            result = stacked.mean(dim=0)  # [batch, world_dim]

        self._last_summary_z_world = result
        return result

    def self_summary(self) -> torch.Tensor:
        """
        Compute theta-cycle-averaged z_self summary.

        Used by E2 when it needs a smoothed self estimate.

        Returns:
            Averaged z_self [batch, self_dim]
        """
        if not self._z_self_buffer:
            batch_size = 1
            result = torch.zeros(batch_size, self.self_dim, device=self.device)
        else:
            stacked = torch.stack(list(self._z_self_buffer), dim=0)
            result = stacked.mean(dim=0)

        self._last_summary_z_self = result
        return result

    @property
    def recent(self) -> Optional[torch.Tensor]:
        """
        Most recent z_world estimates (for MECH-092 replay).

        Returns stacked recent buffer content [T, batch, world_dim].
        Returns None if buffer is empty.
        """
        if not self._z_world_buffer:
            return None
        return torch.stack(list(self._z_world_buffer), dim=0)

    @property
    def last_summary(self) -> Optional[torch.Tensor]:
        """The last theta-cycle summary that was computed."""
        return self._last_summary_z_world

    def is_ready(self) -> bool:
        """True when buffer has at least one entry."""
        return len(self._z_world_buffer) > 0

    def is_full(self) -> bool:
        """True when buffer has reached capacity."""
        return len(self._z_world_buffer) == self.buffer_size

    def set_consolidation_mode(self, enabled: bool) -> None:
        """
        MECH-122 Phase 3: Toggle bidirectional theta-packaging mode.

        When enabled=True, the buffer is in offline consolidation mode (cx->hip).
        Use consolidation_summary() to get reverse-order packaged E1 updates
        for hippocampal transfer.

        Call set_consolidation_mode(True) before Phase 3 offline pass.
        Call set_consolidation_mode(False) to return to normal waking mode.
        """
        self._consolidation_mode = enabled

    def consolidation_summary(self) -> Optional[torch.Tensor]:
        """
        MECH-122 Phase 3: Return reverse-temporal-order packaged E1 updates.

        Theta-packages the buffered z_world estimates in reverse temporal order
        (newest-first) for hippocampal transfer (cx->hip direction proxy).

        Normal summary() averages oldest-to-newest (hip->cx, E3 consumption).
        consolidation_summary() weights RECENT states more heavily by reading
        in reverse: most recent z_world has highest weight (linear decay).

        This is a V3 proxy for the full bidirectional ThetaBuffer mode (V4).
        The reverse-ordering gives Phase 3 its distinct computational signature:
        E1 updates are packaged starting from the agent's current state and
        tracing backward, giving hippocampus recency-weighted terrain context.

        Returns:
            Recency-weighted z_world summary [batch, world_dim], or None if empty.
        """
        if not self._z_world_buffer:
            return None

        # Stack in REVERSE temporal order (newest first)
        entries = list(self._z_world_buffer)  # oldest to newest
        stacked = torch.stack(list(reversed(entries)), dim=0)  # [T, batch, world_dim]
        T = stacked.shape[0]

        # Linear recency weights: newest entry has weight T, oldest has weight 1
        weights = torch.arange(T, 0, -1, dtype=torch.float, device=stacked.device)
        weights = weights / weights.sum()
        weights = weights.view(T, 1, 1)

        result = (stacked * weights).sum(dim=0)  # [batch, world_dim]
        return result

    def reset(self) -> None:
        """Clear the buffer (episode reset)."""
        self._z_world_buffer.clear()
        self._z_self_buffer.clear()
        self._last_summary_z_world = None
        self._last_summary_z_self = None

    def __len__(self) -> int:
        return len(self._z_world_buffer)
