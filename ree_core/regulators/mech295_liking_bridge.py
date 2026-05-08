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

    # MECH-307 Path B: consumer-side conjunction read (registered 2026-05-08).
    # When enabled, compute_conjunction_score_bias() reads the SD-014 valence
    # vector at each candidate's predicted-imminent location plus a global
    # z_beta arousal scalar and returns a per-candidate negative score_bias
    # whenever the four-way conjunction (wanting + liking + signed-positive
    # surprise + z_beta arousal) holds. This gives the MECH-307 substrate fix
    # (commit 65d4e46, EXQ-539 substrate-readiness PASS, behavioural FAIL on
    # C5 lift) a downstream commit-gating consumer that actually reads the
    # conjunction signal, instead of the legacy is_active()-only gate that
    # ignores the conjunction. Default False -> bit-identical OFF.
    use_mech307_conjunction_read: bool = False
    # Threshold knobs match the doc's is_excitement_state_at(...) predicate
    # at REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
    # lines 128-137. Wanting and z_beta share a default; liking is half
    # (anticipatory liking is partial per the doc).
    mech307_conjunction_wanting_threshold: float = 0.6
    mech307_conjunction_liking_threshold: float = 0.3
    mech307_conjunction_z_beta_threshold: float = 0.6
    # Negative-score gain when the conjunction holds. The bias term is
    # -conjunction_gain * drive_level for each candidate whose conjunction
    # predicate fires; zero otherwise.
    mech307_conjunction_gain: float = 1.0


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
        # MECH-307 Path B conjunction-read counters.
        self._n_conjunction_reads = 0  # ticks where the read fired (any K)
        self._n_conjunction_fires = 0  # cumulative per-candidate fires
        self._last_conjunction_count = 0  # K-fires this tick
        self._last_conjunction_score_max = 0.0

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

    # -- MECH-307 Path B: conjunction-aware approach cue ---------------

    def compute_conjunction_score_bias(
        self,
        candidate_z_locs: torch.Tensor,
        residue_field,
        z_beta_arousal: float,
        drive_level: float,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """MECH-307 Path B: conjunction-aware approach cue.

        Reads the SD-014 valence vector at each candidate's predicted-
        imminent z_world location and gates a fixed-magnitude approach
        bias on the four-way conjunction predicate from the doc:
            v[VALENCE_WANTING]   > wanting_threshold
            v[VALENCE_LIKING]    > liking_threshold
            v[VALENCE_SURPRISE]  > 0     (signed; positive PE only --
                                          requires Gap 1 substrate)
            z_beta_arousal       > z_beta_threshold
        When all four hold, the candidate gets a negative score bias of
        -mech307_conjunction_gain * drive_level (E3 lower-is-better, so
        this favours approach). Otherwise the per-candidate bias is 0.

        This is the consumer-side fix for the EXQ-539 PARTIAL-PASS
        finding: substrate counters fired (C1-C4 PASS) but the legacy
        MECH-295 cue path -- drive * goal_proximity -- did not lift
        approach_commit_rate (C5 FAIL) because nothing in that path
        actually read the conjunction signal. This method DOES.

        Args:
            candidate_z_locs: [K, world_dim] -- per-candidate predicted-
                imminent locations. Caller normally supplies the
                first-step z_world summary from each E2-rolled-out
                trajectory (the same tensor used by compute_approach_cue_score_bias).
            residue_field: object exposing evaluate_valence(z) -> [B, 4]
                in component order (wanting, liking, harm, surprise).
                Pass the agent's residue_field; None disables the read.
            z_beta_arousal: scalar global z_beta arousal (e.g. agent's
                self._current_latent.z_beta[..., 0] item, or its abs-mean).
            drive_level: scalar in [0, 1]. Conjunction bias scales with
                drive (silent when drive < min_drive_to_fire).
            simulation_mode: when True returns zeros (MECH-094).

        Returns:
            score_bias: [K] tensor on candidate_z_locs' device/dtype.
            Strictly <= 0. Zero when disabled / sub-floor / no
            conjunction. Bit-identical zero when use_mech307_conjunction_read
            is False on the bridge config.
        """
        K = int(candidate_z_locs.shape[0])
        zero = torch.zeros(K, dtype=candidate_z_locs.dtype,
                           device=candidate_z_locs.device)
        if simulation_mode:
            return zero
        if not self.config.use_mech307_conjunction_read:
            return zero
        if self.config.mech307_conjunction_gain == 0.0:
            return zero
        if residue_field is None or not hasattr(residue_field, "evaluate_valence"):
            return zero
        d = float(drive_level)
        if d < self.config.min_drive_to_fire:
            return zero

        # Evaluate residue valence at each candidate location. evaluate_valence
        # accepts a [batch, world_dim] tensor and returns [batch, 4].
        try:
            v = residue_field.evaluate_valence(candidate_z_locs)
        except Exception:
            return zero
        if v is None or v.shape[0] != K or v.shape[1] < 4:
            return zero

        # Component indices: 0=wanting, 1=liking, 2=harm, 3=surprise.
        v_w = v[:, 0]
        v_l = v[:, 1]
        v_s = v[:, 3]

        w_thr = float(self.config.mech307_conjunction_wanting_threshold)
        l_thr = float(self.config.mech307_conjunction_liking_threshold)
        b_thr = float(self.config.mech307_conjunction_z_beta_threshold)
        beta = float(z_beta_arousal)

        # Hard four-way conjunction (matches doc predicate). The signed-
        # surprise check (v_s > 0) is what makes Gap 1 (signed PE) load-
        # bearing for this consumer: under unsigned VALENCE_SURPRISE the
        # surprise channel accumulates magnitude regardless of sign, so
        # harm-paired surprise can falsely satisfy v_s > 0.
        cond = (
            (v_w > w_thr)
            & (v_l > l_thr)
            & (v_s > 0.0)
            & (torch.full_like(v_w, beta) > b_thr)
        )
        cond_f = cond.to(dtype=candidate_z_locs.dtype)
        gain = float(self.config.mech307_conjunction_gain)
        bias = -gain * d * cond_f

        # Update diagnostics counters.
        n_fires = int(cond.sum().item())
        if n_fires > 0:
            self._n_conjunction_reads += 1
            self._n_conjunction_fires += n_fires
        self._last_conjunction_count = n_fires
        self._last_conjunction_score_max = float(bias.abs().max().item())
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
            # MECH-307 Path B counters (zero when conjunction read disabled).
            "n_conjunction_reads": int(self._n_conjunction_reads),
            "n_conjunction_fires": int(self._n_conjunction_fires),
            "last_conjunction_count": int(self._last_conjunction_count),
            "last_conjunction_score_max": float(self._last_conjunction_score_max),
        }
