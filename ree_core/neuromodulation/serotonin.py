"""
SerotoninModule -- tonic 5-HT state variable for REE-v3.

Claims: MECH-203 (benefit-salience tagging), MECH-204 (REM gate zero-point)
Design doc: REE_assembly/docs/architecture/sleep/serotonergic_cross_state_substrate.md

Substrate requirements implemented:
  SR-1: tonic_5ht slow-accumulating scalar [0, 1]
  SR-2: benefit_salience = tonic_5ht * benefit_exposure (tagging for replay)
        harm_salience = (1 - tonic_5ht) * harm_exposure (harm-symmetric tag,
        2026-08-02 -- see harm_salience() docstring for the calibration
        rationale and the swamping failure mode it fixes)
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

    # MECH-204 F1 cold-start guard. Default False -> bit-identical OFF.
    # The canonical driver start pattern (env.reset(); agent.reset() before any
    # waking tick -- also StepHarness.run_episode) lets SleepLoopManager fire a
    # sleep cycle with ZERO waking ticks. enter_rem() then anchors
    # _persistent_zero_point on E3 precision_init (rv=0.5 -> precision 2.0),
    # which reflects no experience at all, and the F1 EMA carries that sentinel
    # for ~30+ cycles. Measured IGW-20260915-243 (real StepHarness loop,
    # CausalGridWorldV2 size 8, K=1, recal step 0.25): realized z_world PE
    # variance ~0.0039 and waking rv tracks it within ~20 ticks (precision
    # ~255), but the F1 target climbs 2.0 -> 27 -> 49 -> 69 -> 88 across
    # cycles, so WRITEBACK recalibration pushes an already-calibrated rv AWAY
    # from calibration (0.0039 -> 0.0121). V3-EXQ-541c shows the same seed
    # (cycle-1 target 2.148).
    # When True, a REM entry with no waking tick since the last capture does
    # NOT touch _persistent_zero_point. _precision_at_rem_entry is still
    # captured (diagnostic continuity), and on a first-ever cycle
    # compute_recalibration_target() keeps returning its EXISTING 0.0
    # "no target available" sentinel -- which the WRITEBACK consumer
    # (sleep/phase_manager.py `if target > 0.0:`) already skips on, emitting
    # mech204_recalibration_fired=0.0 and no target key, so no consumer change
    # is needed.
    #
    # Direction NOT taken (recorded 2026-09-18): "exclude the precision_init
    # sentinel from the first capture" by value. Rejected -- it needs float
    # equality against 1.0/rv_init, and a genuinely-converged agent that
    # happens to sit at precision_init would be silently skipped. A value test
    # cannot distinguish "no experience" from "experience that landed on the
    # sentinel"; the tick counter tests the actual predicate.
    precision_zero_point_require_waking: bool = False


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

        # MECH-204 F1 cold-start guard: waking ticks since the last capture
        # into _persistent_zero_point. Incremented by note_waking_tick(),
        # which is driven from REEAgent.sense() -- the one per-tick call EVERY
        # waking driver makes (StepHarness, act_with_split_obs, and the
        # hand-rolled per-tick loops alike). Zeroed on capture and by reset().
        # Read ONLY when config.precision_zero_point_require_waking is True.
        # CORRECTED 2026-09-20: this comment previously named
        # REEAgent.update_residue() as the producer. That was never true on
        # main -- update_residue() does not call note_waking_tick(), and a
        # hand-rolled driver that calls neither update_residue() nor
        # serotonin_step() (e.g. v3_exq_541c) would then have had the counter
        # pinned at 0, turning this guard into a permanent kill switch. The
        # shipped wiring in agent.py has always been sense(); only the comment
        # was wrong. Verified by call-site grep: agent.py:4939 is the sole
        # caller.
        self._waking_ticks_since_capture: int = 0

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

        # MECH-204 F1 cold-start guard. Below the enabled/phase guard at the
        # top of this method, so only genuine waking ticks count.
        self._waking_ticks_since_capture += 1

    def note_waking_tick(self) -> None:
        """MECH-204 F1 cold-start guard: record one waking tick.

        Driven from REEAgent.sense() (agent.py, gated on
        config.precision_zero_point_require_waking so the default-off path adds
        no call at all). sense() is the load-bearing producer because it is the
        ONE call every waking tick makes on every driver -- StepHarness,
        act_with_split_obs, and the hand-rolled per-tick loops (e.g.
        v3_exq_541c) alike. serotonin_step() is NOT a usable producer: it is
        called by experiment DRIVERS, never from inside ree_core, and
        StepHarness never calls it -- a counter fed only from there would stay
        at 0 forever on the canonical loop and turn the guard into a permanent
        kill switch for MECH-204 recalibration rather than a cold-start guard.

        No sleep pass calls sense() (its only internal callers are the waking
        wrappers sense_flat() and act_with_split_obs()), and this method
        additionally no-ops outside the waking phase, so replay / REM ticks
        cannot satisfy the guard.

        Over-counting would be harmless by construction -- the guard's
        predicate is only ever `_waking_ticks_since_capture == 0`, so it cannot
        change a decision, while under-counting silently suppresses every
        capture.

        CORRECTED 2026-09-20: this docstring previously named
        REEAgent.update_residue() as the producer and claimed serotonin_step()
        increments too. Neither was true on main. The shipped wiring has always
        been sense() (agent.py:4939, its sole call site); only the
        documentation was wrong. Recorded because a reader who trusted it would
        conclude this guard is inert on any driver that does not call
        update_residue() -- the exact wrong conclusion that nearly derailed the
        V3-EXQ-541d design.

        No-op when disabled or outside the waking phase.
        """
        if not self.config.tonic_5ht_enabled or self._phase != "wake":
            return
        self._waking_ticks_since_capture += 1

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

    def harm_salience(self, harm_exposure: float) -> float:
        """
        Compute harm salience tag for replay prioritisation (SR-2, harm-symmetric).

        harm_salience = (1 - tonic_5ht) * harm_exposure

        Biological/architectural grounding: the design doc (SR-2) describes
        benefit_salience as "complementary to the existing harm salience from
        the residue field" -- but no calibrated write into
        VALENCE_HARM_DISCRIMINATIVE (the SD-014 valence-vector slot
        benefit_salience's WANTING write shares RBF capacity with) actually
        existed. update_residue()'s accumulate() path only ever wrote the
        legacy scalar `weights` (residue density), never the valence vector;
        the SD-014 `valence_harm_enabled` path (agent.py sense()) writes raw
        post-attenuation z_harm.norm() and is a different, ungated-by-5HT
        signal. Neither is tonic-5HT-modulated the way benefit_salience is.

        The (1 - tonic_5ht) factor is not a new invention -- it is the SAME
        complementary weight the substrate already uses for harm at replay
        time: Agent._do_replay's drive_state vector is
        [tonic_5ht, 0.5, 1 - tonic_5ht, surprise_weight], i.e. harm is already
        defined as the (1 - tonic_5ht) counterpart of the wanting/benefit
        weight when the substrate scores replay priority. This mirrors SR-1's
        harm_suppress_rate coupling (elevated z_harm_a suppresses tonic_5ht)
        applied symmetrically on the write side: low tonic 5-HT (the
        depressive attractor, INV-053) amplifies harm salience; high tonic
        5-HT suppresses it.

        `harm_exposure` MUST be the same kind of signal as `benefit_exposure`
        -- an EMA'd, already-normalised nociceptive exposure channel (e.g.
        CausalGridWorldV2.harm_exposure / body_obs[10], the exact counterpart
        of body_obs[11]'s benefit_exposure, updated by the same
        nociception_ema_alpha and clipped to the same [0, 1] range) -- NOT
        raw instantaneous harm_signal. Feeding raw |harm_signal| (env-scale,
        ~0.05-0.13) instead of the EMA convention (~0.0037-0.0066, matching
        benefit_exposure's own scale) was tried and reverted during the
        MECH-203 gap investigation: it swamped the shared-capacity RBF field
        (ResidueField.update_valence -> nearest ACTIVE center, the same
        centers benefit_salience's WANTING write lands on) to the opposite
        degenerate extreme -- 100% harm, 0% benefit -- because the two
        components then accumulated at a ~13-20x scale mismatch. Using the
        matched EMA convention keeps both channels on a comparable scale by
        construction, without an invented calibration constant.

        Returns 0.0 when disabled.
        """
        if not self.config.tonic_5ht_enabled:
            return 0.0
        return (1.0 - self._tonic_5ht) * max(0.0, harm_exposure)

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
        # MECH-204 F1 cold-start guard (default off -> `skip` is always False
        # and this method is bit-identical to the pre-guard version). When on,
        # a REM entry with no waking experience since the last capture carries
        # no new information: current_precision is the E3 precision_init
        # sentinel (first cycle) or an unchanged re-read of the previous
        # capture (later cycles). Skip rather than anchor on / double-count it.
        skip = (
            self.config.precision_zero_point_require_waking
            and self._waking_ticks_since_capture == 0
        )
        if not skip:
            alpha = float(self.config.precision_zero_point_ema_alpha)
            if self._persistent_zero_point is None:
                self._persistent_zero_point = float(current_precision)
            else:
                self._persistent_zero_point = (
                    (1.0 - alpha) * self._persistent_zero_point
                    + alpha * float(current_precision)
                )
            self._waking_ticks_since_capture = 0
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
        # MECH-204 F1 cold-start guard: a new episode starts with no waking
        # experience, so a sleep cycle fired before this episode's first tick
        # must not capture either. (_persistent_zero_point itself deliberately
        # survives reset(); see this method's docstring. hard_reset() needs no
        # change -- it calls reset() first.)
        self._waking_ticks_since_capture = 0

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
            "waking_ticks_since_capture": self._waking_ticks_since_capture,
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
        # MECH-204 F1 cold-start guard: tolerate state dicts predating it.
        # Defaulting to 0 is the CONSERVATIVE direction -- a resume that lands
        # exactly on a REM entry before any tick skips one capture rather than
        # anchoring on a stale value. Inert under the default-off flag.
        self._waking_ticks_since_capture = int(
            state.get("waking_ticks_since_capture", 0)
        )
