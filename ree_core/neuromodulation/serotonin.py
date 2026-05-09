"""
SerotoninModule -- tonic 5-HT state variable for REE-v3.

Claims: MECH-203 (benefit-salience tagging), MECH-204 (REM gate zero-point)
Design doc: REE_assembly/docs/architecture/sleep/serotonergic_cross_state_substrate.md

Substrate requirements implemented:
  SR-1: tonic_5ht slow-accumulating scalar [0, 1]
  SR-2: benefit_salience = tonic_5ht * benefit_exposure (tagging for replay)
  SR-3: REM zero-point hook (_precision_at_rem_entry captured on enter_rem)

Waking dynamics:
  - Rises when benefit_exposure > 0 (rate: rise_rate * benefit_exposure)
  - Decays toward baseline when no benefit (rate: decay_rate)
  - Suppressed by elevated z_harm_a (rate: harm_suppress_rate * z_harm_a_norm)

Sleep dynamics:
  - SWS: held at waking level (5-HT active during SWS)
  - REM: drops to 0.0 (dorsal raphe quiescence)
  - Wake: restored to pre-sleep level

Connection to existing substrate:
  - Modulates GoalConfig.z_goal_seeding_gain dynamically (MECH-187)
  - Modulates GoalConfig.valence_wanting_floor dynamically (MECH-186)
  - benefit_salience feeds SD-014 VALENCE_WANTING for replay prioritisation

Master switch: tonic_5ht_enabled=False (default) -- all methods are safe
no-ops when disabled. Existing experiments are fully unaffected.
"""

from dataclasses import dataclass


@dataclass
class SerotoninConfig:
    """Configuration for the serotonergic neuromodulation system.

    All rates are per-step scalars applied in serotonin_step().
    """
    # Master switch -- disabled by default for backward compatibility.
    tonic_5ht_enabled: bool = False

    # Initial / baseline tonic 5-HT level.
    tonic_5ht_baseline: float = 0.5

    # Waking dynamics rates.
    rise_rate: float = 0.01        # per-step rise per unit benefit_exposure
    decay_rate: float = 0.001      # per-step decay toward baseline (no benefit)
    harm_suppress_rate: float = 0.1  # suppression when z_harm_a elevated

    # Gain modulation ranges (SR-1 -> GoalConfig).
    # tonic_5ht maps linearly to z_goal_seeding_gain in [gain_min, gain_max].
    gain_min: float = 0.3   # seeding gain when tonic_5ht = 0
    gain_max: float = 1.5   # seeding gain when tonic_5ht = 1

    # Floor modulation: tonic_5ht maps to valence_wanting_floor in [floor_min, floor_max].
    floor_min: float = 0.0   # wanting floor when tonic_5ht = 0
    floor_max: float = 0.08  # wanting floor when tonic_5ht = 1

    # MECH-204 F1: cross-cycle persistent zero-point reference.
    # Each REM entry captures precision_at_rem_entry as the moment-snapshot
    # (unchanged for diagnostic continuity). Separately, _persistent_zero_point
    # is updated as an EMA across captures with this alpha:
    #   persistent <- (1 - alpha) * persistent + alpha * captured_precision
    # First capture cold-starts persistent = captured. Subsequent cycles slowly
    # track. compute_recalibration_target() returns persistent (NOT the raw
    # _precision_at_rem_entry), so the WRITEBACK consumer pulls rv toward a
    # stable long-horizon reference rather than the just-captured value (which
    # equals rv at REM entry by construction, making Option A a no-op within a
    # single cycle -- the V3-EXQ-541 finding 2026-05-09).
    # alpha=0.0 freezes on first capture (cold-start anchor). alpha=1.0 reverts
    # to legacy behaviour (persistent = current capture each cycle). Default
    # 0.1 ~= EMA over ~10 cycles, biologically defensible for slow REM-driven
    # setpoint drift. Tunable per experiment.
    precision_zero_point_ema_alpha: float = 0.1


class SerotoninModule:
    """
    Tonic serotonin state variable (SR-1) with benefit-salience tagging (SR-2).

    Lifecycle:
      1. Agent creates SerotoninModule from SerotoninConfig.
      2. Each waking step: call serotonin_step(benefit_exposure, z_harm_a_norm).
      3. Query current_seeding_gain() / current_wanting_floor() for GoalConfig modulation.
      4. Query benefit_salience(benefit_exposure) for replay tagging (SR-2).
      5. Sleep transitions: enter_sws() / enter_rem() / exit_sleep().

    When tonic_5ht_enabled=False, all accessors return static defaults and
    step/transition methods are no-ops.
    """

    def __init__(self, config: SerotoninConfig) -> None:
        self.config = config
        self._tonic_5ht: float = config.tonic_5ht_baseline
        self._phase: str = "wake"  # "wake", "sws", "rem"
        self._pre_sleep_5ht: float = config.tonic_5ht_baseline

        # SR-3: precision snapshot at REM entry (captured by agent).
        self._precision_at_rem_entry: float = 0.0

        # MECH-204 F1: persistent cross-cycle zero-point reference.
        # None until the first REM capture; thereafter EMA-tracked across
        # captures per config.precision_zero_point_ema_alpha. Survives
        # per-episode reset() so the long-horizon reference accumulates
        # across episodes within a session. Cleared only by hard_reset()
        # (i.e. agent reconstruction).
        self._persistent_zero_point: float | None = None

    @property
    def enabled(self) -> bool:
        return self.config.tonic_5ht_enabled

    @property
    def tonic_5ht(self) -> float:
        """Current tonic 5-HT level [0, 1]."""
        return self._tonic_5ht

    @property
    def phase(self) -> str:
        """Current phase: 'wake', 'sws', or 'rem'."""
        return self._phase

    @property
    def precision_at_rem_entry(self) -> float:
        """SR-3: precision snapshot captured when entering REM."""
        return self._precision_at_rem_entry

    def compute_recalibration_target(self) -> float:
        """
        MECH-204 Option A target: the cross-cycle persistent zero-point
        precision reference (F1, 2026-05-09). Consumed by
        E3Selector.recalibrate_precision_to() at the WRITEBACK phase of the
        sleep cycle.

        Returns the EMA-tracked _persistent_zero_point so the WRITEBACK
        consumer pulls rv toward a stable long-horizon reference rather
        than the just-captured precision_at_rem_entry (which equals rv at
        REM entry by construction, making Option A a no-op within a single
        cycle -- V3-EXQ-541 finding).

        Returns 0.0 when the module is disabled or when no REM phase has
        been entered yet (caller treats 0.0 as "no target available" and
        skips recalibration).
        """
        if not self.config.tonic_5ht_enabled:
            return 0.0
        if self._persistent_zero_point is None:
            return 0.0
        return float(self._persistent_zero_point)

    # -- Waking dynamics (SR-1) --

    def serotonin_step(
        self,
        benefit_exposure: float = 0.0,
        z_harm_a_norm: float = 0.0,
    ) -> None:
        """
        Update tonic_5ht for one waking step.

        No-op when disabled or in sleep phase.

        Args:
            benefit_exposure: scalar benefit this step (>= 0).
            z_harm_a_norm: L2 norm of z_harm_a (affective harm accumulator).
        """
        if not self.config.tonic_5ht_enabled or self._phase != "wake":
            return

        # Rise from benefit contact
        if benefit_exposure > 0:
            self._tonic_5ht += self.config.rise_rate * benefit_exposure
        else:
            # Decay toward baseline
            self._tonic_5ht += self.config.decay_rate * (
                self.config.tonic_5ht_baseline - self._tonic_5ht
            )

        # Harm suppression (MECH-186 interaction)
        if z_harm_a_norm > 0:
            self._tonic_5ht -= self.config.harm_suppress_rate * z_harm_a_norm

        # Clamp
        self._tonic_5ht = max(0.0, min(1.0, self._tonic_5ht))

    # -- Benefit-salience tagging (SR-2) --

    def benefit_salience(self, benefit_exposure: float) -> float:
        """
        Compute benefit salience tag for replay prioritisation (SR-2).

        benefit_salience = tonic_5ht * benefit_exposure

        Returns 0.0 when disabled.
        """
        if not self.config.tonic_5ht_enabled:
            return 0.0
        return self._tonic_5ht * max(0.0, benefit_exposure)

    # -- GoalConfig dynamic modulation --

    def current_seeding_gain(self) -> float:
        """
        Map tonic_5ht to z_goal_seeding_gain (MECH-187 dynamic).

        Linear interpolation: gain_min at 5ht=0, gain_max at 5ht=1.
        Returns 1.0 (identity) when disabled.
        """
        if not self.config.tonic_5ht_enabled:
            return 1.0
        cfg = self.config
        return cfg.gain_min + self._tonic_5ht * (cfg.gain_max - cfg.gain_min)

    def current_wanting_floor(self) -> float:
        """
        Map tonic_5ht to valence_wanting_floor (MECH-186 dynamic).

        Linear interpolation: floor_min at 5ht=0, floor_max at 5ht=1.
        Returns 0.0 (disabled) when disabled.
        """
        if not self.config.tonic_5ht_enabled:
            return 0.0
        cfg = self.config
        return cfg.floor_min + self._tonic_5ht * (cfg.floor_max - cfg.floor_min)

    # -- Sleep phase transitions --

    def enter_sws(self) -> None:
        """
        SWS entry: hold tonic_5ht at waking level.

        5-HT is active during SWS (dorsal raphe still firing).
        Store pre-sleep level for restoration on wake.
        """
        if not self.config.tonic_5ht_enabled:
            return
        self._pre_sleep_5ht = self._tonic_5ht
        self._phase = "sws"

    def enter_rem(self, current_precision: float = 0.0) -> None:
        """
        REM entry: tonic_5ht drops to 0 (dorsal raphe quiescence).

        SR-3: captures current precision as the zero-point reference.

        MECH-204 F1: also updates _persistent_zero_point as an EMA across
        captures (or cold-starts on first capture). The persistent value is
        what compute_recalibration_target() returns; _precision_at_rem_entry
        is preserved as the moment-snapshot for diagnostic continuity.

        Args:
            current_precision: agent's current precision (from E3).
        """
        if not self.config.tonic_5ht_enabled:
            return
        self._tonic_5ht = 0.0
        self._precision_at_rem_entry = float(current_precision)
        alpha = float(self.config.precision_zero_point_ema_alpha)
        if self._persistent_zero_point is None:
            self._persistent_zero_point = float(current_precision)
        else:
            self._persistent_zero_point = (
                (1.0 - alpha) * self._persistent_zero_point
                + alpha * float(current_precision)
            )
        self._phase = "rem"

    def exit_sleep(self) -> None:
        """
        Wake from sleep: restore tonic_5ht to pre-sleep level.
        """
        if not self.config.tonic_5ht_enabled:
            return
        self._tonic_5ht = self._pre_sleep_5ht
        self._phase = "wake"

    # -- State management --

    def reset(self) -> None:
        """Reset to initial state (new episode).

        Note: _persistent_zero_point (MECH-204 F1) survives per-episode reset
        so the long-horizon REM-driven precision reference accumulates across
        episodes within a session. Use hard_reset() to clear it explicitly
        (e.g. between training stages).
        """
        self._tonic_5ht = self.config.tonic_5ht_baseline
        self._phase = "wake"
        self._pre_sleep_5ht = self.config.tonic_5ht_baseline
        self._precision_at_rem_entry = 0.0

    def hard_reset(self) -> None:
        """Reset including cross-episode state (MECH-204 _persistent_zero_point).

        Use between distinct training stages or when serotonin state should
        be fully cleared. Per-episode resets should use reset().
        """
        self.reset()
        self._persistent_zero_point = None

    def get_state(self) -> dict:
        """Serialisable state snapshot."""
        return {
            "tonic_5ht": self._tonic_5ht,
            "phase": self._phase,
            "pre_sleep_5ht": self._pre_sleep_5ht,
            "precision_at_rem_entry": self._precision_at_rem_entry,
            "persistent_zero_point": self._persistent_zero_point,
        }

    def load_state(self, state: dict) -> None:
        """Restore from state snapshot."""
        self._tonic_5ht = state.get("tonic_5ht", self.config.tonic_5ht_baseline)
        self._phase = state.get("phase", "wake")
        self._pre_sleep_5ht = state.get("pre_sleep_5ht", self.config.tonic_5ht_baseline)
        self._precision_at_rem_entry = state.get("precision_at_rem_entry", 0.0)
        # MECH-204 F1: tolerate older state dicts that lack persistent_zero_point.
        if "persistent_zero_point" in state:
            self._persistent_zero_point = state["persistent_zero_point"]
