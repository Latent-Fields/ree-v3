"""
Multi-Rate Clock — SD-006 Phase 1 (ARC-023)

Implements time-multiplexed execution with explicit rate parameters.
Each loop operates at its own temporal grain:

  E1 (sensorium):     updates every e1_steps_per_tick = 1 env steps (every step)
  E2 (action-enact):  updates every e2_steps_per_tick = 3 env steps (motor rate)
  E3 (planning-gate): updates every e3_steps_per_tick = 10 env steps (deliberation rate)

This implements ARC-023 (three characteristic thalamic heartbeat rates) in
phase 1 form: time-multiplexed, explicit rate params.

Claims implemented:
  ARC-023: Three characteristic update rates
  MECH-091: phase_reset() on salient events (completion, unexpected harm, commit cross)
  MECH-093: e3_steps_per_tick varies with z_beta magnitude (arousal → faster E3)

Note: phase 2 (full HTA with separate goroutines/threads) is deferred per spec §3/SD-006.
"""

import torch
from typing import Optional


class MultiRateClock:
    """
    Multi-rate loop clock for SD-006 phase 1.

    Tracks the env step count and fires tick events for each loop
    at the configured rates. Supports phase reset (MECH-091) and
    beta-modulated E3 rate (MECH-093).
    """

    def __init__(
        self,
        e1_steps_per_tick: int = 1,
        e2_steps_per_tick: int = 3,
        e3_steps_per_tick: int = 10,
        theta_buffer_size: int = 10,
        beta_rate_min_steps: int = 5,
        beta_rate_max_steps: int = 20,
        beta_magnitude_scale: float = 1.0,
    ):
        self.e1_steps_per_tick = e1_steps_per_tick
        self.e2_steps_per_tick = e2_steps_per_tick
        self._e3_base_steps = e3_steps_per_tick
        self.theta_buffer_size = theta_buffer_size

        # MECH-093: arousal-modulated E3 rate
        self.beta_rate_min_steps = beta_rate_min_steps
        self.beta_rate_max_steps = beta_rate_max_steps
        self.beta_magnitude_scale = beta_magnitude_scale
        self._current_e3_steps = e3_steps_per_tick

        # Step counters (reset on phase_reset for E3)
        self._global_step: int = 0
        self._e3_phase_step: int = 0  # Steps since last E3 tick or phase reset

        # MECH-091: salient event flag
        self._pending_phase_reset: bool = False

        # MECH-092: quiescence tracking (no salient event this E3 cycle)
        self._salient_event_this_cycle: bool = False

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def e3_steps_per_tick(self) -> int:
        return self._current_e3_steps

    def advance(self) -> dict:
        """
        Advance clock by one env step.

        Returns dict of boolean tick flags:
          e1_tick:      True if E1 should update this step
          e2_tick:      True if E2 should update this step
          e3_tick:      True if E3 should update this step
          e3_quiescent: True if E3 ticked and no salient event (→ replay, MECH-092)
          theta_tick:   True if ThetaBuffer should summarise this step (MECH-089)
        """
        self._global_step += 1
        self._e3_phase_step += 1

        e1_tick = (self._global_step % self.e1_steps_per_tick == 0)
        e2_tick = (self._global_step % self.e2_steps_per_tick == 0)

        # E3 tick: phase-step based (allows phase reset to shift timing)
        e3_tick = False
        e3_quiescent = False

        if self._pending_phase_reset:
            # Salient event requested reset — fire E3 immediately and reset counter
            e3_tick = True
            self._e3_phase_step = 0
            self._pending_phase_reset = False
            e3_quiescent = False  # Phase reset is itself salient
            self._salient_event_this_cycle = False
        elif self._e3_phase_step >= self._current_e3_steps:
            e3_tick = True
            e3_quiescent = not self._salient_event_this_cycle  # MECH-092
            self._e3_phase_step = 0
            self._salient_event_this_cycle = False

        # Theta buffer update: once per theta_buffer_size steps (MECH-089)
        theta_tick = (self._global_step % self.theta_buffer_size == 0)

        return {
            "e1_tick": e1_tick,
            "e2_tick": e2_tick,
            "e3_tick": e3_tick,
            "e3_quiescent": e3_quiescent,
            "theta_tick": theta_tick,
            "global_step": self._global_step,
        }

    def phase_reset(self) -> None:
        """
        Phase reset on salient event (MECH-091).

        Salient events: task completion, unexpected harm, commitment crossing.
        Synchronises next E3 tick to the event — fires E3 at next advance().
        """
        self._pending_phase_reset = True
        self._salient_event_this_cycle = True

    def mark_salient(self) -> None:
        """
        Mark that a salient event occurred this E3 cycle (suppresses quiescence).

        Use this for events that don't warrant a full phase reset but should
        suppress MECH-092 replay (e.g., routine harm events within a cycle).
        """
        self._salient_event_this_cycle = True

    def update_e3_rate_from_beta(self, z_beta: torch.Tensor) -> None:
        """
        Update E3 heartbeat rate based on z_beta magnitude (MECH-093).

        High arousal (large |z_beta|) → faster E3 rate (fewer steps between ticks).
        Low arousal → slower E3 rate.

        Args:
            z_beta: affective latent [batch, beta_dim]
        """
        beta_mag = z_beta.detach().norm(dim=-1).mean().item()
        scaled = beta_mag * self.beta_magnitude_scale

        # Interpolate: high magnitude → min_steps, low magnitude → max_steps
        # clamp scaled to [0, 1] range
        t = min(1.0, max(0.0, scaled))
        new_steps = int(
            self.beta_rate_max_steps * (1 - t) + self.beta_rate_min_steps * t
        )
        new_steps = max(self.beta_rate_min_steps, min(self.beta_rate_max_steps, new_steps))
        self._current_e3_steps = new_steps

    def reset(self) -> None:
        """Reset clock state (episode reset)."""
        self._global_step = 0
        self._e3_phase_step = 0
        self._pending_phase_reset = False
        self._salient_event_this_cycle = False
        self._current_e3_steps = self._e3_base_steps

    def get_state(self) -> dict:
        return {
            "global_step": self._global_step,
            "e3_phase_step": self._e3_phase_step,
            "current_e3_steps": self._current_e3_steps,
            "pending_phase_reset": self._pending_phase_reset,
        }
