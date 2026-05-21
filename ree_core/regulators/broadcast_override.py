"""
SD-037: Broadcast Override Regulator (orexin-analog).

Architectural commitment (see REE_assembly/docs/architecture/sd_037_broadcast_override_regulator.md):

  Third regulatory layer of the V3 control stack, alongside 5-HT goal-pipeline
  gain (MECH-186/187/188) and SD-036 GABAergic cross-stream decay. The
  override_signal is a scalar in [0, 1] driven by:

    drive_level (SD-012, homeostatic depletion, [0, 1])
    sustained_threat (rolling-window magnitude over z_harm, normalised to [0, 1])

  Combined linearly through a sigmoid recruitment threshold, EMA-smoothed,
  and clamped to [0, 1]. The signal is consumed at three sites:

    PAG freeze-gate (MECH-279):
      exit_threshold scaled by (1 + alpha_override * override_signal) so
      that strong override raises the bar for entering / staying in the
      committed-freeze state. Aligns with the orexin -> arousal /
      escape-from-freeze story (Carter et al. 2009; LH -> PAG projections).

    SalienceCoordinator (SD-032a):
      override_signal injected as a salience reweight, biasing the
      operating-mode aggregate toward external_task / internal_planning
      via update_signal("override_signal", ...). MECH-261 generalises
      MECH-094 here -- the registry is the gating point.

    GoalState (SD-012):
      drive -> z_goal seeding is GATED by override_signal. When
      override_signal < recruitment_threshold the legacy SD-012 path is
      used; when >= threshold the seeding gain is amplified to push
      drive into z_goal recruitment. This is the "drive becomes
      action-orienting only when the override system has recruited"
      semantic.

Biological analogue: orexinergic (hypocretin) hub in the lateral
hypothalamus (LH). Persistent depletion (SD-012) plus sustained
nociceptive signal (z_harm window) recruits LH orexin neurons; their
broad projections (PAG, BLA, LC, VTA, mPFC) then gate downstream
arousal / escape / motivational systems.

Master switch: REEConfig.use_broadcast_override (default False). With
the flag off, agent.broadcast_override is None and every consumer site
is a no-op. Backward compatible.

MECH-094: simulation_mode=True ticks return the cached signal unchanged
and do not advance internal counters / threat window. Replay / DMN
content must not recruit the override system.

Non-trainable: pure scalar arithmetic, no gradient flow.

Reset per episode (drive_level itself is owned by GoalState; this
regulator caches only its own threat window and EMA state).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import math


@dataclass
class BroadcastOverrideConfig:
    """SD-037 configuration.

    Defaults are conservative seeds from the SD-037 design doc; orexin
    kinetics lit-pull will refine them post-validation.
    """

    # Master flag mirror; the agent-level use_broadcast_override gates
    # instantiation, but holding it here makes the regulator independently
    # testable.
    enabled: bool = True

    # Sigmoid recruitment threshold (subtracted before the squash).
    # override_raw = sigmoid(drive_weight*drive + harm_weight*sustained_threat
    #                        - recruitment_threshold)
    recruitment_threshold: float = 0.5

    # PAG freeze-gate scaling on theta_freeze:
    #   exit_threshold_eff = theta_freeze * (1 + alpha_override * override_signal) * gaba_tone
    # Read by PAGFreezeGate.tick(); kept here for ablation symmetry.
    alpha_override: float = 0.5

    # SalienceCoordinator reweight magnitude. Kept for documentation; the
    # actual weight is wired in the coordinator config (salience_weights).
    salience_reweight_alpha: float = 0.3

    # Linear coefficients on the two driving signals before the sigmoid.
    drive_weight: float = 1.0
    harm_weight: float = 1.0

    # Sustained-threat magnitude window: rolling mean of z_harm.norm() over
    # the last sustained_threat_window ticks; the window is normalised by
    # sustained_threat_threshold so values >= threshold map to ~1.0.
    sustained_threat_window: int = 12
    sustained_threat_threshold: float = 0.4

    # EMA decay rate on the override_signal output (smooths fast-spiking
    # input variation; ~20-tick EMA at 0.05).
    decay_rate: float = 0.05


class BroadcastOverrideRegulator:
    """SD-037 broadcast override regulator (orexin-analog).

    Public API:
      tick(drive_level, z_harm_norm, simulation_mode=False) -> float
        Advance the threat window with the current z_harm_norm scalar,
        compute override_raw via sigmoid(drive_weight*drive +
        harm_weight*sustained_threat - recruitment_threshold), EMA-smooth
        into override_signal, clip to [0, 1], cache, and return. With
        simulation_mode=True (MECH-094) the cached value is returned
        unchanged and no state advances.

      reset()
        Clear per-episode state (threat window, EMA, last signal).

      override_signal -> float
        Cached scalar in [0, 1]; reflects last waking tick.

      diagnostics -> dict
        n_ticks, override_signal, sustained_threat_norm, drive_input,
        last raw activation, recruitment threshold.
    """

    def __init__(self, config: Optional[BroadcastOverrideConfig] = None):
        self.config = config or BroadcastOverrideConfig()
        win = max(1, int(self.config.sustained_threat_window))
        self._threat_window: Deque[float] = deque(maxlen=win)
        self._override_signal: float = 0.0
        self._last_raw: float = 0.0
        self._last_drive: float = 0.0
        self._last_sustained: float = 0.0
        self._n_ticks: int = 0

    # -- Per-episode state --

    def reset(self) -> None:
        """Clear per-episode threat window, EMA state, and diagnostics."""
        self._threat_window.clear()
        self._override_signal = 0.0
        self._last_raw = 0.0
        self._last_drive = 0.0
        self._last_sustained = 0.0
        self._n_ticks = 0

    # -- Tick --

    def tick(
        self,
        drive_level: float,
        z_harm_norm: float,
        simulation_mode: bool = False,
        z_harm_intero_norm: Optional[float] = None,
        lpb_split_recruitment: bool = False,
    ) -> float:
        """Advance state and return the new override_signal.

        Args:
            drive_level: SD-012 homeostatic depletion in [0, 1].
            z_harm_norm: Current waking external-threat magnitude (e.g.
                LatentState.z_harm.norm().item() or LPB external scalar).
                Pass 0.0 if z_harm is unavailable; the threat window degrades
                gracefully. When lpb_split_recruitment is True this tracks
                external threat only (freeze/avoidance path), not override
                recruitment.
            simulation_mode: MECH-094 hypothesis-tag equivalent. True ->
                return the cached override_signal unchanged and do not
                advance counters / threat window. Replay / DMN content
                must not recruit the override system.
            z_harm_intero_norm: MECH-282 interoceptive distress magnitude in
                [0, 1]. When lpb_split_recruitment is True, the sustained-
                threat window uses this signal instead of z_harm_norm so
                override recruitment is driven by metabolic distress + drive,
                not external predator proximity.
            lpb_split_recruitment: MECH-282 coupling flag. True -> intero
                magnitude feeds the sigmoid harm_weight term; external
                magnitude is ignored for override recruitment.

        Returns:
            override_signal in [0, 1].
        """
        if not self.config.enabled:
            return float(self._override_signal)

        if simulation_mode:
            return float(self._override_signal)

        self._n_ticks += 1

        # Sustained-threat normalised magnitude in [0, 1]:
        # rolling-mean(threat_input) / threshold, clipped at 1.0.
        # MECH-282: interoceptive distress recruits override; external threat
        # does not (external still available via z_harm_norm for diagnostics).
        if lpb_split_recruitment and z_harm_intero_norm is not None:
            threat_input = float(z_harm_intero_norm)
        else:
            threat_input = float(z_harm_norm)
        self._threat_window.append(threat_input)
        if len(self._threat_window) > 0:
            mean_threat = sum(self._threat_window) / float(len(self._threat_window))
        else:
            mean_threat = 0.0
        threshold = max(1e-6, float(self.config.sustained_threat_threshold))
        sustained = max(0.0, min(1.0, mean_threat / threshold))

        drive_input = max(0.0, min(1.0, float(drive_level)))

        # Sigmoid recruitment: override_raw = sigmoid(drive_w*drive +
        # harm_w*sustained - recruitment_threshold).
        z = (
            float(self.config.drive_weight) * drive_input
            + float(self.config.harm_weight) * sustained
            - float(self.config.recruitment_threshold)
        )
        # Stable sigmoid.
        if z >= 0.0:
            ez = math.exp(-z)
            raw = 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            raw = ez / (1.0 + ez)

        # EMA smooth.
        alpha = max(0.0, min(1.0, float(self.config.decay_rate)))
        new_signal = (1.0 - alpha) * float(self._override_signal) + alpha * raw
        # Clip [0, 1].
        new_signal = max(0.0, min(1.0, new_signal))

        self._override_signal = new_signal
        self._last_raw = raw
        self._last_drive = drive_input
        self._last_sustained = sustained
        return float(new_signal)

    # -- Read API --

    @property
    def override_signal(self) -> float:
        return float(self._override_signal)

    @property
    def diagnostics(self) -> Dict[str, object]:
        return {
            "n_ticks": int(self._n_ticks),
            "override_signal": float(self._override_signal),
            "last_raw": float(self._last_raw),
            "last_drive": float(self._last_drive),
            "last_sustained": float(self._last_sustained),
            "recruitment_threshold": float(self.config.recruitment_threshold),
            "sustained_threat_window_size": int(len(self._threat_window)),
        }
