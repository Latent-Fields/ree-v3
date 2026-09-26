"""MECH-302 substrate: rolling-window harm-norm descent detector.

Fires a relief_completion_event when the z_harm_a norm has sustained a downward
crossing of drop_threshold within a rolling window of window_length ticks.

Non-trainable. Pure arithmetic. No nn.Module inheritance.

Simulation gating: tick() with simulation_mode=True returns False without
advancing the buffer -- waking-path signal only (MECH-094 requirement).

Event latch (suffering-derivative-comparator-refractory, 2026-09-26, opt-in):
with latch_enabled=False (default) tick() fires on EVERY tick whose window drop
clears drop_threshold, so one healing trajectory emits a train of events
(~9-17 per scheduled injection in V3-EXQ-517d; failure_autopsy_gflag0452-D1).
With latch_enabled=True the comparator fires once per descent: after a fire it
latches, tracks the lowest norm seen since the fire, and re-arms only when the
norm rises at least rearm_rise above that trough (a new harm onset), restarting
the window from that peak. The first
event of each descent fires on exactly the tick the unlatched comparator would
have fired, so latched events are a strict subset of unlatched events. This
matches SD-050's "fires a relief-completion event when a sustained drop exceeds
a threshold" and the phasic relief-offset DA transient (Navratilova 2012), not
a sustained train across the whole descent.
"""

from typing import Optional


class SufferingDerivativeComparator:
    """Rolling-window harm-norm descent detector for MECH-302 relief-completion event.

    Reads z_harm_a.norm() scalar each tick. Fires when the window shows a
    sustained drop from initial to final norm >= drop_threshold, provided the
    initial norm was above min_initial_norm (prevents spurious fires on a stream
    that is already quiet).

    Diagnostic counters (read-only, never affect firing):
      fire_count        -- ticks that returned True since the last reset()
      suppressed_count  -- ticks whose window crossed threshold but the latch
                           held the event back (always 0 with latch disabled)
    """

    def __init__(
        self,
        window_length: int = 5,
        drop_threshold: float = 0.10,
        min_initial_norm: float = 0.05,
        latch_enabled: bool = False,
        rearm_rise: Optional[float] = None,
    ):
        self.window_length = window_length
        self.drop_threshold = drop_threshold
        self.min_initial_norm = min_initial_norm
        self.latch_enabled = bool(latch_enabled)
        # None -> symmetric default: a new rise the size of a counted descent.
        self.rearm_rise = float(drop_threshold if rearm_rise is None else rearm_rise)
        if self.rearm_rise < 0.0:
            raise ValueError(
                "suffering_rearm_rise must be >= 0 (got %r)" % (rearm_rise,)
            )
        self._norm_buffer: list = []
        self._latched: bool = False
        self._trough: float = 0.0
        self.fire_count: int = 0
        self.suppressed_count: int = 0

    @property
    def latched(self) -> bool:
        return self._latched

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
        if self.latch_enabled and self._latched:
            # Track the trough of the current descent; re-arm on a new rise.
            if z_harm_a_norm < self._trough:
                self._trough = z_harm_a_norm
            elif z_harm_a_norm - self._trough >= self.rearm_rise:
                # New harm onset. Re-arm AND restart the window from this
                # peak, so the next event needs a full fresh window of descent
                # (a mid-descent bump cannot re-fire on the re-arm tick).
                self._latched = False
                self._norm_buffer = [z_harm_a_norm]
        if len(self._norm_buffer) < self.window_length:
            return False
        initial_norm = self._norm_buffer[0]
        if initial_norm < self.min_initial_norm:
            return False
        total_drop = initial_norm - self._norm_buffer[-1]
        crossed = total_drop >= self.drop_threshold
        if not crossed:
            return False
        if self.latch_enabled:
            if self._latched:
                self.suppressed_count += 1
                return False
            self._latched = True
            self._trough = z_harm_a_norm
        self.fire_count += 1
        return True

    def reset(self) -> None:
        self._norm_buffer = []
        self._latched = False
        self._trough = 0.0
        self.fire_count = 0
        self.suppressed_count = 0
