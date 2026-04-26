"""
MECH-295: drive -> liking-stream -> approach_cue bridge (weak reading).

Architectural commitment (see REE_assembly/docs/architecture/
mech_295_drive_liking_approach_bridge.md):

  Functional restatement (claims.yaml MECH-295):
      drive_level (SD-012) ->
        liking_gain on goal-congruent outcome predictions ->
          approach_cue_signal at action selection (E3 / BG)

  This module is the substrate for the bridge. It does NOT instantiate the
  liking-stream itself (that lives in ResidueField VALENCE_LIKING per SD-014;
  the existing consummatory write path is REEAgent.update_liking()). It
  also does NOT directly modulate goal seeding (that is SD-012 in
  GoalState.update() and SD-037 in REEAgent.update_z_goal()). What this
  module DOES is the missing wiring between them:

    (a) anticipatory liking write at the goal location, gated on
        drive_level * z_goal_norm (the "goal-congruent outcome" surface),
    (b) per-candidate liking-readout at action selection, converted into
        an additive negative score_bias on E3 (E3 lower-is-better, so
        liking favours approach by reducing the score).

  Weak reading commitment (MECH-295 functional_restatement and
  evidence_quality_note):
      Strong necessity: elevated approach_commit requires elevated
        liking-stream activation (level-coupled).
      Weak necessity: baseline liking-stream activation is sufficient,
        but the bridge wiring must be intact -- if the link is severed,
        drive amplification produces no approach regardless of drive
        magnitude.

  The DAT-knockdown finding (Pecina 2003: more wanting, unchanged liking)
  is incompatible with the strong reading but compatible with the weak
  reading. MECH-295 commits to the WEAK reading provisionally.

  This module deliberately does NOT couple level. The cue-side gain is a
  function of drive * goal_proximity (the "is the bridge intact?"
  surface), not of drive * residue.liking (which would be the
  level-coupled strong reading). Setting
  mech295_liking_to_approach_cue_gain=0.0 from a config is the
  "severed bridge" arm of the falsifiable test:
      drive elevated AND mech295_drive_to_liking_gain > 0 AND
      mech295_liking_to_approach_cue_gain = 0
  -- liking-stream activity rises (anticipatory write) but cannot reach
  action selection. Approach_commit is predicted to collapse.

Biological reading: NAc shell hedonic hotspot (Pecina & Berridge 2005,
Castro & Berridge 2014) and ventral pallidum (Smith, Berridge & Aldridge
2011) produce hedonic-coding pulses on goal-congruent outcome predictions
under elevated drive; these supply the cue-side approach pull through
descending striatal projections to the BG action-selection circuit
(Berridge & Kringelbach 2015). The architectural articulation in
Berridge & Kringelbach 2015 is the most explicit theoretical statement
of this bridge.

MECH-094: simulation_mode argument is honoured throughout. When True the
write is skipped (no liking pulse during DMN / replay) and the cue-side
readout is invoked with the existing (non-simulation) residue field.
This is the same call-site-scoping pattern used by MECH-269 / MECH-287.

Backward compat: the bridge is constructed and ticked only when
REEConfig.use_mech295_liking_bridge=True. With the master flag off the
agent does not instantiate this module; integration sites in agent.py
are no-ops. With the flag on but both gain knobs at zero the bridge
runs but produces zero write and zero score_bias -- still bit-identical
to OFF for action selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class MECH295LikingBridgeConfig:
    """Configuration for the MECH-295 drive -> liking -> approach_cue bridge."""

    # Multiplier on drive_level * z_goal_norm for the anticipatory liking
    # write at the goal location. Setting to 0 disables the write side
    # without touching the cue side.
    drive_to_liking_gain: float = 1.0

    # Multiplier converting the per-candidate liking signal into the
    # additive approach-side score_bias. Setting to 0 disables the cue
    # side -- the "severed bridge" arm of the falsifiability test.
    liking_to_approach_cue_gain: float = 0.5

    # Drive floor below which the bridge is silent.
    min_drive_to_fire: float = 0.1

    # Goal-norm floor below which the bridge does not fire.
    min_z_goal_norm_to_fire: float = 0.05


@dataclass
class MECH295LikingBridgeOutput:
    """Diagnostic output from a single tick."""

    fired_write: bool = False
    fired_cue: bool = False
    drive_level: float = 0.0
    z_goal_norm: float = 0.0
    liking_write_value: float = 0.0
    approach_cue_signal_max: float = 0.0
    approach_cue_signal_mean: float = 0.0
    n_candidates: int = 0


class MECH295LikingBridge:
    """Drive -> liking-stream -> approach_cue bridge.

    Pure arithmetic over scalars and small vectors. No trainable
    parameters. No phased training needed.

    Two operations:
        compute_anticipatory_liking_write(drive_level, z_goal_norm) ->
            scalar value to be written to VALENCE_LIKING at the goal
            location (caller resolves the location).
        compute_approach_cue_score_bias(drive_level, candidate_proximities,
            simulation_mode) ->
            per-candidate score_bias [K] (negative; lower-is-better).

    The agent integrates: (a) is called from REEAgent.update_z_goal()
    after the drive computation, before the actual residue write;
    (b) is called from REEAgent.select_action() after the per-candidate
    z_world summaries are built (cand_world_summaries from the
    lateral_pfc / ofc blocks). Wiring details: see agent.py and
    REE_assembly/docs/architecture/mech_295_drive_liking_approach_bridge.md.
    """

    def __init__(self, config: MECH295LikingBridgeConfig) -> None:
        self.config = config
        # Diagnostics for the most recent tick. None until first tick.
        self._last_output: Optional[MECH295LikingBridgeOutput] = None
        # Counters; persist across episodes for end-of-run reporting.
        self._n_write_fires = 0
        self._n_cue_fires = 0

    # -- Reset hooks ---------------------------------------------------

    def reset(self) -> None:
        """Clear the per-tick diagnostic cache. Counters persist."""
        self._last_output = None

    # -- Anticipatory liking write (a) ---------------------------------

    def compute_anticipatory_liking_write(
        self,
        drive_level: float,
        z_goal_norm: float,
        simulation_mode: bool = False,
    ) -> float:
        """Compute the magnitude of the anticipatory liking write.

        Returns 0.0 (no-op) when:
            - simulation_mode is True (MECH-094 gate).
            - drive_level < min_drive_to_fire.
            - z_goal_norm < min_z_goal_norm_to_fire.
            - drive_to_liking_gain is 0.0.

        Otherwise:
            value = drive_to_liking_gain * drive_level * z_goal_norm

        The caller is responsible for actually writing this to the
        residue field at the appropriate location (the goal location,
        not the agent's current location -- this is the cue-side
        anticipatory pulse, not consummatory contact).
        """
        if simulation_mode:
            return 0.0
        if self.config.drive_to_liking_gain == 0.0:
            return 0.0
        d = float(drive_level)
        g = float(z_goal_norm)
        if d < self.config.min_drive_to_fire:
            return 0.0
        if g < self.config.min_z_goal_norm_to_fire:
            return 0.0
        value = self.config.drive_to_liking_gain * d * g
        value = float(max(0.0, value))
        if value > 0.0:
            self._n_write_fires += 1
        return value

    # -- Approach cue at action selection (b) --------------------------

    def compute_approach_cue_score_bias(
        self,
        drive_level: float,
        candidate_proximities: torch.Tensor,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """Compute the per-candidate approach-side score_bias.

        Args:
            drive_level: scalar in [0, 1].
            candidate_proximities: [K] tensor in [0, 1] giving each
                candidate trajectory's first-step proximity to the
                current goal latent. Caller computes via
                GoalState.goal_proximity(...).
            simulation_mode: when True returns zeros (MECH-094 gate).

        Returns:
            score_bias: [K] tensor. NEGATIVE values (E3 lower-is-better,
            so liking favours approach by reducing the score). Zero when
            disabled / sub-floor / severed.
        """
        K = int(candidate_proximities.shape[0])
        zero = torch.zeros(K, dtype=candidate_proximities.dtype,
                           device=candidate_proximities.device)
        if simulation_mode:
            return zero
        if self.config.liking_to_approach_cue_gain == 0.0:
            return zero
        d = float(drive_level)
        if d < self.config.min_drive_to_fire:
            return zero
        # Per-candidate liking signal: drive * goal_proximity scaled
        # by drive_to_liking_gain. Even when the write side is off
        # (drive_to_liking_gain==0) the cue side STILL fires off
        # drive*goal_proximity directly -- this matches the weak
        # reading: baseline liking-stream activation is sufficient,
        # the bridge does not require the level-coupled write to
        # produce the cue. The drive_to_liking_gain only controls the
        # anticipatory write magnitude, which is observed via residue.
        liking_signal = d * candidate_proximities.clamp(min=0.0, max=1.0)
        # Negate: E3 lower-is-better, so liking-driven approach
        # reduces the score.
        bias = -self.config.liking_to_approach_cue_gain * liking_signal
        if bool((bias.abs() > 0.0).any().item()):
            self._n_cue_fires += 1
        return bias

    # -- Combined tick (orchestration helper) --------------------------

    def tick(
        self,
        drive_level: float,
        z_goal_norm: float,
        candidate_proximities: Optional[torch.Tensor] = None,
        simulation_mode: bool = False,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """Convenience tick that computes BOTH (a) and (b) and caches
        diagnostics. The agent normally calls (a) and (b) separately at
        their respective integration sites; this helper exists for the
        contract tests and future direct callers.

        Returns:
            (write_value, score_bias_or_None).
        """
        write_value = self.compute_anticipatory_liking_write(
            drive_level=drive_level,
            z_goal_norm=z_goal_norm,
            simulation_mode=simulation_mode,
        )
        if candidate_proximities is None:
            score_bias = None
        else:
            score_bias = self.compute_approach_cue_score_bias(
                drive_level=drive_level,
                candidate_proximities=candidate_proximities,
                simulation_mode=simulation_mode,
            )
        # Cache diagnostics.
        out = MECH295LikingBridgeOutput()
        out.drive_level = float(drive_level)
        out.z_goal_norm = float(z_goal_norm)
        out.liking_write_value = float(write_value)
        out.fired_write = (write_value > 0.0)
        if score_bias is not None:
            out.n_candidates = int(score_bias.shape[0])
            cue_abs = score_bias.abs()
            out.approach_cue_signal_max = float(cue_abs.max().item())
            out.approach_cue_signal_mean = float(cue_abs.mean().item())
            out.fired_cue = (out.approach_cue_signal_max > 0.0)
        self._last_output = out
        if out.fired_write:
            self._n_write_fires += 1
        if out.fired_cue:
            self._n_cue_fires += 1
        return write_value, score_bias

    # -- Diagnostics ---------------------------------------------------

    def get_last_output(self) -> Optional[MECH295LikingBridgeOutput]:
        return self._last_output

    def get_diagnostics(self) -> dict:
        return {
            "n_write_fires": int(self._n_write_fires),
            "n_cue_fires": int(self._n_cue_fires),
        }
