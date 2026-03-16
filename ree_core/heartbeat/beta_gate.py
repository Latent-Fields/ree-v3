"""
Beta Gate — MECH-090 Implementation

MECH-090: Beta oscillations gate E3→action_selection propagation.
Beta state controls whether E3's updated policy state propagates to
action selection, NOT whether E3 updates internally.

During committed sequence: beta_elevated=True
  → E3 continues updating internally
  → action_selector.update() is suppressed (policy output held)

At sequence completion / stop-change signal: beta_elevated=False
  → E3's current policy state propagates to action selection

This implements beta as a commitment gate, not a learning gate.
"""

import torch
from typing import Optional


class BetaGate:
    """
    Beta-gated policy propagation gate (MECH-090).

    Controls whether E3's current policy state propagates to action selection.
    E3 always updates internally. The gate only blocks propagation.
    """

    def __init__(self, initial_beta_elevated: bool = False):
        self._beta_elevated: bool = initial_beta_elevated
        self._held_policy_state: Optional[torch.Tensor] = None
        self._propagation_count: int = 0
        self._hold_count: int = 0

    @property
    def is_elevated(self) -> bool:
        """True when beta is elevated (commitment active, propagation suppressed)."""
        return self._beta_elevated

    def elevate(self) -> None:
        """
        Elevate beta state (sequence commitment begins).

        After this, propagate() will hold E3 state.
        """
        self._beta_elevated = True

    def release(self) -> None:
        """
        Release beta state (sequence complete or stop-change signal).

        After this, the next propagate() call will pass through E3 state.
        """
        self._beta_elevated = False

    def propagate(self, e3_policy_state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Gate E3 policy state propagation to action selection (MECH-090).

        If beta is elevated: hold state, return None (action selector not updated).
        If beta is not elevated: pass state through, return it.

        Args:
            e3_policy_state: E3's current internal policy state [batch, state_dim]

        Returns:
            e3_policy_state if gate is open (beta not elevated), else None.
        """
        if self._beta_elevated:
            self._held_policy_state = e3_policy_state.detach()
            self._hold_count += 1
            return None
        else:
            self._held_policy_state = None
            self._propagation_count += 1
            return e3_policy_state

    def get_held_state(self) -> Optional[torch.Tensor]:
        """Return the held policy state (if any)."""
        return self._held_policy_state

    def get_state(self) -> dict:
        return {
            "beta_elevated": self._beta_elevated,
            "propagation_count": self._propagation_count,
            "hold_count": self._hold_count,
            "has_held_state": self._held_policy_state is not None,
        }

    def reset(self) -> None:
        """Reset gate state (episode reset)."""
        self._beta_elevated = False
        self._held_policy_state = None
