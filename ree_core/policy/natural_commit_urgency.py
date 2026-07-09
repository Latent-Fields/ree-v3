"""Commit/release-DURATION lever: graded natural-commit-occupancy release.

The rung-6 lever of the `f_dominance_conversion_ceiling` substrate front -- the
COMMIT/RELEASE-DURATION face, PARALLEL to the selection-face MECH-448 (rank-
preserving F->eligibility demotion), NOT an escalation of it.

WHY THIS EXISTS (V3-EXQ-460h finding):
    The F-dominance front's selection-face levers (MECH-439 conflict-graded
    width / commit-temperature; MECH-448 F->eligibility demotion) act at E3
    SELECTION and do not shorten the F-driven NATURAL-COMMIT latch occupancy.
    On strong (F-decisive) seeds the bistable beta latch elevates once and then
    HOLDS for ~2400-2600 steps, because nothing releases it: readiness stays
    healthy (a decisive F-gap = "good options") so MECH-342 maintenance-release
    is silent, and no closure fires so SD-034 is silent. That monolithic hold
    SWAMPS the SD-034 closure de-commit, leaving MECH-445 commit-intent and
    MECH-446 de-commit-occupancy-drop disjoint across seeds (measurable only
    where the natural commit is WEAK -- the 460h disjoint-certifier problem).

    This lever makes the F-driven natural commit LESS MONOLITHIC so weak-
    natural-commit becomes the norm across seeds, dissolving the 460h problem.

BINDING DESIGN CONSTRAINT FROM THE BIOLOGY (BG-3 SYNTHESIS divergence D1,
load-bearing):
    Biology does NOT set commitment DURATION with a separate fixed refractory
    clock. It times the hold with a GRADED BG/pallidal urgency signal that
    rises over the held epoch (Thura, Cabana, Feghaly & Cisek 2022, PLoS Biol
    10.1371/journal.pbio.3001861) and/or makes maintenance CO-EXTENSIVE with
    the executing action (Jin, Tecuapetla & Costa 2014, Nat Neurosci
    10.1038/nn.3632). REE's existing committed-run-scaled beta-gate refractory
    is the "tuned, not bio-sourced" divergence D1 names. So this lever is a
    GRADED release, NOT another fixed refractory constant.

TWO D1-FAITHFUL RELEASE MODES (both togglable; the sequenced 460i-successor
falsifier discriminates which lifts):

  (1) URGENCY mode (Thura/Cisek). A graded release-urgency accumulates each
      maintenance tick the latch is held by a natural commit:
          decisiveness_scale = 1 + gap_entry_sensitivity * gap_norm_at_entry
          urgency += urgency_rate * decisiveness_scale
          fire when urgency >= release_bound
      gap_norm_at_entry in [0, 1] is the normalised top-F decisiveness captured
      at commit entry (1 = a decisive F-gap = the kind of commit that
      monopolises the latch). The gap-scaling is the LOAD-BEARING piece: an
      F-decisive natural commit accrues release-urgency FASTER, so the
      strongest-F holds -- the ones that swamp the de-commit -- are shortened
      most. This attacks the F-dominance directly in the duration domain and
      FOLDS IN the "gap-scaled commit-entry threshold" impl_hint candidate (the
      commit-entry decisiveness sets the release rate). gap_entry_sensitivity=0
      reduces the urgency to a flat fixed-rate timeout -- the contrasted
      "another fixed refractory" control the D1 falsifier compares against.

  (2) ACTION-EXTENT mode (Jin). Release the natural commit when the committed
      trajectory's executed action sequence COMPLETES (the agent has stepped
      through all of trajectory.actions rather than repeating the last action
      indefinitely). Renders the "maintenance co-extensive with the executing
      action" biology + the "natural-commit run-length cap" impl_hint candidate
      as a behaviourally-grounded cap (the trajectory horizon), NOT a tuned
      constant. Fires regardless of urgency when the sequence is complete.

WHAT THIS IS NOT (falsifiable distinctions; the substrate must not collapse
into any of these):

  * NOT MECH-342 maintenance-release. MECH-342 fires on DEGRADED readiness
    (poor/low-decisiveness options) and is therefore SILENT on the healthy-but-
    prolonged F-decisive commit that actually monopolises the latch -- exactly
    why strong seeds hold ~2400 steps. This fires on a healthy, prolonged
    natural commit (the duration-urgency face MECH-342 does not cover).
  * NOT the SD-034 Leg-B committed-run-scaled refractory (MECH-446). That holds
    the latch DOWN post-closure (how long to keep it released). This shortens
    the natural commit's occupancy UP (how long it stays elevated). It does NOT
    install a refractory; it releases.
  * NOT MECH-091 urgency-interrupt. MECH-091 fires on z_harm_a/z_harm_un
    threat. This is a DURATION urgency with no harm-stream input.
  * NOT ARC-028/MECH-105 completion. That releases on a HIGH completion signal
    (a good plan was found). This releases on accumulated held-duration urgency
    or executed-sequence completion regardless of plan quality.

ACCUMULATOR DYNAMICS (per maintenance tick, only while beta is elevated by a
natural commit the lever armed via note_commit_entry):

    if committed_run_length > onset_ticks and urgency_mode:
        decisiveness_scale = 1 + gap_entry_sensitivity * gap_norm_at_entry
        urgency += urgency_rate * decisiveness_scale
        urgency  = clip(urgency, 0, urgency_cap)
        fire if urgency >= release_bound                  # urgency release
    if (not fire) and action_extent_mode and action_sequence_complete:
        fire                                               # Jin behaviour-co-extensive

The caller (REEAgent.select_action release block) acts on the returned bool:
beta_gate.release(); _committed_step_idx = 0; _committed_anchor_keys = None;
e3._committed_trajectory = None. Urgency is reset to 0 at commit ENTRY (each
natural committed program accumulates independently) and on fire.

MECH-094

  tick(simulation_mode=True) returns False without advancing urgency. A replay
  / DMN tick must not abort a committed motor program. Matches the
  SD-035 / MECH-279 / MECH-313 / MECH-320 / MECH-342 simulation_mode pattern.

INTEGRATION SITE

  REEAgent.__init__: instantiates self.natural_commit_urgency when
  config.use_natural_commit_urgency_release=True. None otherwise (bit-identical
  OFF). REEAgent.select_action bistable elevate site: on a fresh NATURAL commit
  (result.committed) calls note_commit_entry(gap_norm) -- arms the lever and
  resets urgency. REEAgent.select_action release block (alongside the MECH-342
  block): when natural_commit_urgency is not None AND beta_gate.is_elevated,
  computes action_sequence_complete from _committed_step_idx vs the committed
  trajectory horizon and calls tick(); on fire releases the committed program.
  REEAgent.reset: calls reset() per-episode.

See REE_assembly/docs/architecture/natural_commit_occupancy_release.md and
ree_core/policy/commit_maintenance_release.py (the readiness-degradation
release sibling).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NaturalCommitUrgencyReleaseConfig:
    """Configuration for the graded natural-commit-occupancy release lever.

    Attributes:
        use_natural_commit_urgency_release : master switch. False = disabled
            (default, backward-compatible). REEAgent does not instantiate
            NaturalCommitUrgencyRelease when False.
        urgency_mode : enable the Thura/Cisek graded-urgency release. Consulted
            only when the master switch is on.
        action_extent_mode : enable the Jin maintenance-co-extensive release
            (fire when the executed action sequence completes). Consulted only
            when the master switch is on.
        urgency_rate : per-tick base increment of the release-urgency
            accumulator (before the gap-scaled decisiveness multiplier).
        release_bound : urgency threshold at which the urgency-mode release
            fires.
        urgency_cap : hard clamp on accumulated urgency (numerical guard).
            Must be >= release_bound.
        gap_entry_sensitivity : scales how strongly the commit-entry
            decisiveness (gap_norm_at_entry in [0, 1]) raises the per-tick
            urgency increment. 0.0 = a flat fixed-rate timeout (the contrasted
            "another fixed refractory" control). >0 makes an F-decisive commit
            accrue urgency faster -> the strongest-F (most monopolising) holds
            are shortened most. This is the load-bearing D1 piece.
        onset_ticks : grace ticks at the start of a committed run before urgency
            begins accruing (a short execution window the hold is allowed before
            release-urgency builds). 0 = urgency accrues from the first tick.
    """

    use_natural_commit_urgency_release: bool = False
    urgency_mode: bool = True
    action_extent_mode: bool = True
    urgency_rate: float = 0.01
    release_bound: float = 1.0
    urgency_cap: float = 1.5
    gap_entry_sensitivity: float = 1.0
    onset_ticks: int = 0


class NaturalCommitUrgencyRelease:
    """Graded natural-commit-occupancy release regulator (waking-only).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance. Reuses the
    BetaGate's committed_run_length (the MECH-090 commit-gate machinery) rather
    than maintaining its own latch (ARC-106 guardrail G2: reuse, do not
    duplicate the latch).

    Diagnostics tracked:
        _urgency                  : float (the accumulator itself)
        _gap_norm_at_entry        : float (entry decisiveness; the gap-scaling)
        _last_decisiveness_scale  : float
        _natural_commit_armed     : bool  (set at a natural commit entry)
        _n_ticks                  : int   (maintenance ticks evaluated)
        _n_urgency_releases       : int   (urgency-mode release events)
        _n_action_extent_releases : int   (Jin-mode release events)
        _last_occupancy_at_release: int   (committed_run_length at the last fire)
        _n_simulation_skips       : int   (MECH-094 skip count)
    """

    def __init__(
        self, config: "NaturalCommitUrgencyReleaseConfig | None" = None
    ) -> None:
        self.config = (
            config
            if config is not None
            else NaturalCommitUrgencyReleaseConfig()
        )
        c = self.config
        if c.urgency_rate <= 0.0:
            raise ValueError(f"urgency_rate must be > 0. Got {c.urgency_rate}.")
        if c.release_bound <= 0.0:
            raise ValueError(
                f"release_bound must be > 0. Got {c.release_bound}."
            )
        if c.urgency_cap < c.release_bound:
            raise ValueError(
                "urgency_cap must be >= release_bound. Got "
                f"cap={c.urgency_cap}, bound={c.release_bound}."
            )
        if c.gap_entry_sensitivity < 0.0:
            raise ValueError(
                "gap_entry_sensitivity must be >= 0. Got "
                f"{c.gap_entry_sensitivity}."
            )
        if c.onset_ticks < 0:
            raise ValueError(f"onset_ticks must be >= 0. Got {c.onset_ticks}.")
        self._urgency: float = 0.0
        self._gap_norm_at_entry: float = 0.0
        self._last_decisiveness_scale: float = 0.0
        self._natural_commit_armed: bool = False
        self._n_ticks: int = 0
        self._n_urgency_releases: int = 0
        self._n_action_extent_releases: int = 0
        self._last_occupancy_at_release: int = 0
        self._n_simulation_skips: int = 0
        # SD-033e frontopolar de-commit lever diagnostics (all inert at 0.0 unless
        # the caller passes a nonzero frontopolar_pressure into tick()).
        self._fp_release_count: int = 0
        self._fp_last_pressure: float = 0.0
        self._fp_pressure_accum: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def note_commit_entry(self, gap_norm: float) -> None:
        """Arm the lever at a fresh NATURAL commit entry.

        Resets the urgency accumulator (each natural committed program
        accumulates release-urgency independently) and stores the entry
        decisiveness gap_norm (clipped to [0, 1]) for the urgency-rate scaling.
        Called by REEAgent at the bistable elevate site only when result.committed
        (a natural F-driven commit); a purely closure-coupled elevation does not
        arm this lever (its occupancy is governed by the SD-034 closure
        machinery).
        """
        self._urgency = 0.0
        self._gap_norm_at_entry = max(0.0, min(1.0, float(gap_norm)))
        self._natural_commit_armed = True

    def get_urgency(self) -> float:
        """Return the current release-urgency accumulator."""
        return self._urgency

    @property
    def is_armed(self) -> bool:
        """True when a natural committed run is being tracked by the lever."""
        return self._natural_commit_armed

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def tick(
        self,
        committed_run_length: int,
        action_sequence_complete: bool,
        simulation_mode: bool = False,
        frontopolar_pressure: float = 0.0,
    ) -> bool:
        """Advance the lever one maintenance tick; return True iff release fires.

        Call only while beta is elevated (the caller guards on
        beta_gate.is_elevated). No-op (returns False) when the lever was not
        armed by a natural commit entry -- a purely closure-coupled committed
        run is left to the SD-034 machinery.

        Args:
            committed_run_length : ticks the latch has been continuously
                elevated since the most recent entry (BetaGate.committed_run_length).
            action_sequence_complete : True when the committed trajectory's
                executed action sequence has been fully stepped (the agent would
                otherwise repeat the last action). Drives the action-extent mode.
            simulation_mode : MECH-094 gate. When True, no state advance and only
                the simulation-skip counter increments; returns False.
            frontopolar_pressure : SD-033e V3-narrow de-commit lever contribution.
                An entry-relative, NON-F counterfactual-improvement release
                pressure (frontopolar_gain * max(0, cfv_now - cfv_at_entry))
                computed by the caller (REEAgent + FrontopolarAnalog). Added to
                the SAME urgency accumulator, so it fires on the same
                urgency >= release_bound. 0.0 (default) => bit-identical to the
                pure NaturalCommitUrgencyRelease lever (the frontopolar-OFF /
                gain=0 contrast arm). See ree_core/pfc/frontopolar_analog.py.

        Returns:
            True iff this tick fires a release.
        """
        if simulation_mode:
            self._n_simulation_skips += 1
            return False
        if not self._natural_commit_armed:
            return False

        self._n_ticks += 1
        fire = False

        c = self.config
        # (1) URGENCY mode (Thura/Cisek graded release) + SD-033e frontopolar
        # de-commit pressure. Both feed the SAME urgency accumulator and fire on
        # the same urgency >= release_bound. Structured so that with
        # frontopolar_pressure == 0.0 (OFF default) this is bit-identical to the
        # original urgency-mode block: when urgency_mode is on the increment is
        # exactly urgency_rate * decisiveness_scale, and when both urgency_mode is
        # off and frontopolar_pressure is 0.0 nothing accrues (as before).
        if committed_run_length > c.onset_ticks:
            _inc = 0.0
            if c.urgency_mode:
                decisiveness_scale = 1.0 + (
                    c.gap_entry_sensitivity * self._gap_norm_at_entry
                )
                self._last_decisiveness_scale = decisiveness_scale
                _inc += float(c.urgency_rate) * decisiveness_scale
            _fp = float(frontopolar_pressure)
            if _fp != 0.0:
                _inc += _fp
                self._fp_last_pressure = _fp
                self._fp_pressure_accum += _fp
            if _inc != 0.0:
                self._urgency += _inc
                self._urgency = max(
                    0.0, min(float(c.urgency_cap), self._urgency)
                )
                if self._urgency >= float(c.release_bound):
                    fire = True
                    self._n_urgency_releases += 1
                    # SD-033e attribution: this fire had a positive frontopolar
                    # de-commit contribution (cfv_now > cfv_at_entry at the firing
                    # tick) -- a REAL switch toward an improved foregone option,
                    # NOT a flat timeout or F noise.
                    if _fp > 0.0:
                        self._fp_release_count += 1

        # (2) ACTION-EXTENT mode (Jin maintenance-co-extensive release).
        if (not fire) and c.action_extent_mode and action_sequence_complete:
            fire = True
            self._n_action_extent_releases += 1

        if fire:
            self._last_occupancy_at_release = int(committed_run_length)
            self._urgency = 0.0
            self._natural_commit_armed = False
        return fire

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode state and diagnostic counters."""
        self._urgency = 0.0
        self._gap_norm_at_entry = 0.0
        self._last_decisiveness_scale = 0.0
        self._natural_commit_armed = False
        self._n_ticks = 0
        self._n_urgency_releases = 0
        self._n_action_extent_releases = 0
        self._last_occupancy_at_release = 0
        self._n_simulation_skips = 0
        self._fp_release_count = 0
        self._fp_last_pressure = 0.0
        self._fp_pressure_accum = 0.0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "urgency": self._urgency,
            "gap_norm_at_entry": self._gap_norm_at_entry,
            "last_decisiveness_scale": self._last_decisiveness_scale,
            "natural_commit_armed": self._natural_commit_armed,
            # Latch-occupancy length at the most recent release (the lever's
            # primary readout: how long the natural commit held before release).
            "ncur_last_occupancy_at_release": self._last_occupancy_at_release,
            "ncur_n_ticks": self._n_ticks,
            # Release-event counts, split by mode.
            "ncur_n_urgency_releases": self._n_urgency_releases,
            "ncur_n_action_extent_releases": self._n_action_extent_releases,
            "ncur_n_releases_total": (
                self._n_urgency_releases + self._n_action_extent_releases
            ),
            "ncur_n_simulation_skips": self._n_simulation_skips,
            # SD-033e frontopolar de-commit attribution: count of urgency-mode
            # fires that had a POSITIVE frontopolar pressure contribution at the
            # firing tick (cfv_now > cfv_at_entry) -- the primary "real switch"
            # readout. >0 with the gate-drop proves the de-commit is attributable
            # to the frontopolar term, not F noise or a flat timeout.
            "frontopolar_release_count": self._fp_release_count,
            "frontopolar_last_pressure": self._fp_last_pressure,
            "frontopolar_pressure_accum": self._fp_pressure_accum,
        }
