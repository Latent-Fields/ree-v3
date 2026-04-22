"""
MECH-279: PAG (periaqueductal gray) freeze-gate.

Architectural commitment (see REE_assembly/docs/architecture/sd_036_gabaergic_decay_regulator.md
section 3 "PAG freeze-gating (MECH-279)"):

  Freeze is a *committed* behavioural state -- sustained motor immobility plus
  elevated autonomic arousal. Not just "no movement"; an active commitment to
  not-move with its own duration and exit criterion. Biologically gated by PAG,
  where descending inputs from amygdala / hypothalamus / medial PFC converge on
  freeze-promoting cells that are themselves GABAergic. Freeze termination
  requires GABAergic inhibition to wane.

Logic:

  freeze_commit(t) = (z_harm_a(t) * duration_above_threshold(t)) > theta_freeze
  freeze_active(t) = freeze_commit OR (freeze_active(t-1) AND z_harm_a(t) > exit_threshold)
  exit_threshold   = theta_freeze * gaba_tone(t)

  duration_above_threshold(t) = ticks since z_harm_a first crossed
                                duration_input_threshold (resets when z_harm_a
                                falls back below that threshold).

  When freeze is active, the action selector is constrained to no-op /
  minimal-movement actions. Exit requires z_harm_a to fall below
  exit_threshold, which depends on both SD-036 decay (z_harm_a returns toward
  baseline) and gaba_tone (GABA agonists raise the exit_threshold, making exit
  easier). The same neurotransmitter system gates BOTH entry (PAG freeze-cell
  commitment) and exit (SD-036 decay returning z_harm_a below threshold).

Non-trainable: pure arithmetic over scalars and a small counter. No gradient
flow. Reset per episode.

Master switch: REEConfig.use_pag_freeze_gate (default False) gates instantiation
and wiring. With the flag off, agents behave bit-identically to legacy.

MECH-094: simulation_mode=True ticks return a zeroed-output (freeze inactive)
without updating internal state. Replay / DMN content must not commit the
agent into a behavioural freeze state.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PAGFreezeGateConfig:
    """MECH-279 freeze-gate configuration.

    Defaults are conservative -- freeze fires on sustained, substantial
    z_harm_a load. With `theta_freeze=2.0` and `duration_input_threshold=0.4`,
    a z_harm_a magnitude of 0.5 needs at least 4 sustained ticks above 0.4 to
    accumulate (z_harm_a * duration) > theta_freeze. Tune `theta_freeze` per
    experiment to set the catatonia entry sensitivity.
    """

    # Master flag, mirrored on REEConfig. Held here too so the gate is
    # independently testable from the agent.
    enabled: bool = True

    # Commit threshold for entering freeze. Compared against
    # z_harm_a_norm * duration_above_threshold.
    theta_freeze: float = 2.0

    # Threshold above which the duration-above counter increments. When
    # z_harm_a_norm falls back below this value, the counter resets to zero.
    duration_input_threshold: float = 0.4

    # Optional minimum committed-freeze duration (in ticks). Once freeze
    # commits, the gate stays active for at least this many ticks regardless
    # of z_harm_a / exit_threshold dynamics. 0 = no minimum (exit on first
    # tick where z_harm_a drops below exit_threshold).
    min_freeze_duration: int = 0

    # Optional maximum freeze duration (in ticks). 0 = no cap. Useful to
    # prevent permanent locks during smoke / debug; in normal operation,
    # SD-036 decay should reliably bring z_harm_a below exit_threshold.
    max_freeze_duration: int = 0


@dataclass
class PAGFreezeGateOutput:
    """Per-tick PAGFreezeGate output."""

    freeze_active: bool = False
    # True on the tick where freeze commits this step (newly entered).
    freeze_commit: bool = False
    # True on the tick where freeze releases this step.
    freeze_release: bool = False
    # Accumulated duration counter at the end of this tick.
    duration_above_threshold: int = 0
    # The exit threshold actually used this tick (theta_freeze * gaba_tone).
    exit_threshold: float = 0.0
    # The z_harm_a magnitude observed this tick.
    z_harm_a_norm: float = 0.0
    # How many ticks freeze has been active (0 when inactive).
    ticks_in_freeze: int = 0


class PAGFreezeGate:
    """MECH-279 PAG-analog committed-freeze gate.

    Public API:
      tick(z_harm_a_norm, gaba_tone=1.0, simulation_mode=False) -> PAGFreezeGateOutput
        Compute the freeze state for this step.

      reset()
        Clear per-episode state.

      is_active        Convenience property mirroring last freeze_active.

      diagnostics      Dict of running counters.

    State (per episode):
      _freeze_active                   bool, current freeze status
      _duration_above_threshold        int, sustained-input counter
      _ticks_in_freeze                 int, how long freeze has been active
      _last_output                     PAGFreezeGateOutput, last tick's output
      _n_ticks, _n_commits, _n_releases  diagnostic counters
    """

    def __init__(self, config: Optional[PAGFreezeGateConfig] = None):
        self.config = config or PAGFreezeGateConfig()

        self._freeze_active: bool = False
        self._duration_above_threshold: int = 0
        self._ticks_in_freeze: int = 0
        self._last_output: PAGFreezeGateOutput = PAGFreezeGateOutput()

        self._n_ticks: int = 0
        self._n_commits: int = 0
        self._n_releases: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear per-episode state."""
        self._freeze_active = False
        self._duration_above_threshold = 0
        self._ticks_in_freeze = 0
        self._last_output = PAGFreezeGateOutput()

    @property
    def is_active(self) -> bool:
        return bool(self._freeze_active)

    # -- Tick: main per-step computation --

    def tick(
        self,
        z_harm_a_norm: float,
        gaba_tone: float = 1.0,
        simulation_mode: bool = False,
    ) -> PAGFreezeGateOutput:
        """Compute the freeze gate state for this step.

        Args:
            z_harm_a_norm: Magnitude (L2 norm) of the SD-011 affective harm
                stream this tick.
            gaba_tone: Global GABAergic tonic multiplier (matches the SD-036
                regulator's gaba_tone). Used to compute exit_threshold =
                theta_freeze * gaba_tone. Higher tone -> higher exit threshold
                -> easier exit (benzo-analog accelerates termination).
            simulation_mode: MECH-094 hypothesis-tag equivalent. True -> return
                a zeroed output and do not update internal state. Replay / DMN
                content must not commit the agent into freeze.

        Returns:
            PAGFreezeGateOutput with freeze_active, commit / release edges,
            duration counter, and diagnostic fields.
        """
        if not self.config.enabled:
            # Master-off path returns a zeroed output without touching state.
            out = PAGFreezeGateOutput(
                freeze_active=False,
                freeze_commit=False,
                freeze_release=False,
                duration_above_threshold=int(self._duration_above_threshold),
                exit_threshold=0.0,
                z_harm_a_norm=float(z_harm_a_norm),
                ticks_in_freeze=0,
            )
            self._last_output = out
            return out

        if simulation_mode:
            # MECH-094 gate: replay / simulation -- return zeroed and do NOT
            # update entry / exit counters.
            out = PAGFreezeGateOutput(
                freeze_active=False,
                freeze_commit=False,
                freeze_release=False,
                duration_above_threshold=int(self._duration_above_threshold),
                exit_threshold=0.0,
                z_harm_a_norm=float(z_harm_a_norm),
                ticks_in_freeze=int(self._ticks_in_freeze),
            )
            self._last_output = out
            return out

        self._n_ticks += 1

        z = float(z_harm_a_norm)
        # Clamp gaba_tone to non-negative; values <0 are not biologically
        # meaningful for the exit-threshold computation.
        tone = max(0.0, float(gaba_tone))

        # 1. Update the sustained-input duration counter. The counter only
        #    accumulates while the gate is INACTIVE -- once committed, the
        #    accumulator stops advancing (and is reset to zero on release).
        #    This implements the per-cycle "fresh accumulation" semantic:
        #    each commit requires a new run-up of sustained input above
        #    duration_input_threshold rather than re-firing on the same
        #    accumulator immediately after release.
        if not self._freeze_active:
            if z > float(self.config.duration_input_threshold):
                self._duration_above_threshold += 1
            else:
                self._duration_above_threshold = 0

        # 2. Compute exit threshold for this tick.
        exit_threshold = float(self.config.theta_freeze) * tone

        # 3. Edge detection.
        commit_this_tick = False
        release_this_tick = False

        # 4. Entry check (only when not already in freeze).
        if not self._freeze_active:
            commit_value = z * float(self._duration_above_threshold)
            if commit_value > float(self.config.theta_freeze):
                self._freeze_active = True
                self._ticks_in_freeze = 0  # will increment to 1 below
                commit_this_tick = True
                self._n_commits += 1

        # 5. Exit check (only when active and not just committed).
        if self._freeze_active and not commit_this_tick:
            below_exit = z < exit_threshold
            min_dur_met = self._ticks_in_freeze >= int(self.config.min_freeze_duration)
            max_dur_cap = (
                int(self.config.max_freeze_duration) > 0
                and self._ticks_in_freeze >= int(self.config.max_freeze_duration)
            )
            if (below_exit and min_dur_met) or max_dur_cap:
                self._freeze_active = False
                release_this_tick = True
                self._n_releases += 1
                # Reset both the freeze-duration counter and the sustained-
                # input accumulator on release so the next commit requires a
                # fresh run-up.
                self._ticks_in_freeze = 0
                self._duration_above_threshold = 0

        # 6. Tick the in-freeze counter when active.
        if self._freeze_active:
            self._ticks_in_freeze += 1

        out = PAGFreezeGateOutput(
            freeze_active=bool(self._freeze_active),
            freeze_commit=bool(commit_this_tick),
            freeze_release=bool(release_this_tick),
            duration_above_threshold=int(self._duration_above_threshold),
            exit_threshold=float(exit_threshold),
            z_harm_a_norm=float(z),
            ticks_in_freeze=int(self._ticks_in_freeze),
        )
        self._last_output = out
        return out

    # -- Read-only accessors --

    @property
    def last_output(self) -> PAGFreezeGateOutput:
        return self._last_output

    @property
    def diagnostics(self) -> dict:
        return {
            "n_ticks": int(self._n_ticks),
            "n_commits": int(self._n_commits),
            "n_releases": int(self._n_releases),
            "freeze_active": bool(self._freeze_active),
            "duration_above_threshold": int(self._duration_above_threshold),
            "ticks_in_freeze": int(self._ticks_in_freeze),
        }
