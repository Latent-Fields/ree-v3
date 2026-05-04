"""MECH-302 substrate: rolling-window harm-norm descent detector.

Fires a relief_completion_event when the z_harm_a norm has sustained a downward
crossing of drop_threshold within a rolling window of window_length ticks.

Non-trainable. Pure arithmetic. No nn.Module inheritance.

Simulation gating: tick() with simulation_mode=True returns False without
advancing the buffer -- waking-path signal only (MECH-094 requirement).
"""


class SufferingDerivativeComparator:
    """Rolling-window harm-norm descent detector for MECH-302 relief-completion event.

    Reads z_harm_a.norm() scalar each tick. Fires when the window shows a
    sustained drop from initial to final norm >= drop_threshold, provided the
    initial norm was above min_initial_norm (prevents spurious fires on a stream
    that is already quiet).
    """

    def __init__(
        self,
        window_length: int = 5,
        drop_threshold: float = 0.10,
        min_initial_norm: float = 0.05,
    ):
        self.window_length = window_length
        self.drop_threshold = drop_threshold
        self.min_initial_norm = min_initial_norm
        self._norm_buffer: list = []

    def tick(self, z_harm_a_norm: float, simulation_mode: bool = False) -> bool:
        """Advance the comparator by one tick. Returns True when event fires.

        simulation_mode=True returns False immediately without buffer advance
        (MECH-094: waking-stream signal only; replay/DMN must not trigger events).
        """
        if simulation_mode:
            return False
        self._norm_buffer.append(z_harm_a_norm)
        if len(self._norm_buffer) > self.window_length:
            self._norm_buffer.pop(0)
        if len(self._norm_buffer) < self.window_length:
            return False
        initial_norm = self._norm_buffer[0]
        if initial_norm < self.min_initial_norm:
            return False
        total_drop = initial_norm - self._norm_buffer[-1]
        return total_drop >= self.drop_threshold

    def reset(self) -> None:
        self._norm_buffer = []
