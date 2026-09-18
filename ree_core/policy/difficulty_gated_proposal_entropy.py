"""SD-061 component 2: difficulty-gated proposal-entropy regulator (MECH-343 2b).

The regulator half of the difficulty-gated proposal-entropy substrate. Given a
``stuck_score`` in [0, 1] from the SD-061 ``StuckStateDetector``, it maps the
impasse onto a transient gain on the PROPOSAL-generation layer (ARC-018
hippocampal / CEM): a wider candidate set plus a higher within-class sampling
temperature, scaled by the stuck_score, decaying back to baseline as the
impasse clears (the decay is carried by the detector's asymmetric EMA).

This is the V3 rendering of the MECH-343 main loop: *stuck-state signal ->
proposal-entropy gain (ARC-018 hippocampal/CEM candidate-set widening +
within-class temperature) -> goal/harm scoring preserved -> E3 arbitration over
the widened pool -> commitment only when score margin clears -> entropy decay on
goal progress.* This module owns only the PROPOSAL-WIDENING gain; scoring,
commitment thresholds (MECH-090 / MECH-342), and selection authority (569i
top-k conversion / MECH-341) are untouched -- a hard problem must trigger WIDER
INTERNAL PROPOSALS, not random behaviour.

WHAT THIS IS NOT
----------------
  * NOT MECH-313 (noise_floor). MECH-313 lifts the action-SELECTION softmax
    temperature, state-independently, every tick. This lifts PROPOSAL-layer
    entropy (num_candidates + within-class CEM temperature), only when stuck.
  * NOT a commitment lever. It widens proposals; the existing MECH-090 /
    MECH-342 commitment predicates decide when to commit over the widened pool.

GAIN MAPPING (gated by stuck_score s in [0, 1]):
    extra_candidates  = round(candidate_widen_max * s)            # >= 0
    temperature_gain  = 1.0 + temperature_gain_max * s            # >= 1.0

The caller (REEAgent._e3_tick) applies extra_candidates to the
HippocampalModule.propose_trajectories ``num_candidates`` and (optionally)
scales the differentiable-CEM within-class temperature by ``temperature_gain``
for the duration of the proposal, restoring it afterward. With s = 0 (not
stuck) the gain is identity (0 extra candidates, 1.0x temperature) ->
bit-identical to the un-regulated proposal.

THE TEMPERATURE HALF IS INERT UNLESS SD-055 IS ON (recorded 2026-09-18)
----------------------------------------------------------------------
``differentiable_cem_temperature`` has exactly ONE consumer in ``ree_core``:
``HippocampalModule``'s CEM refit at ``hippocampal/module.py:2416``, which sits
inside ``if getattr(self.config, "use_differentiable_cem", False):`` -- SD-055
substrate, default ``False``. So with SD-055 off, ``REEAgent._e3_tick``'s
temperature mutation writes a value nothing reads, and the regulator's effective
manipulation is CANDIDATE-COUNT WIDENING ONLY.

That was recorded nowhere until 2026-09-18, and it had already misled the
record: V3-EXQ-694 (the SD-061 readiness diagnostic) never set
``use_differentiable_cem``, so its C2 "regulator load-bearing" PASS certified the
COUNT half only -- while SD-061's own ``what_would_answer`` criterion (2) reads
as certifying that the regulator "lifts differentiable_cem_temperature
transiently". See GFLAG-0352 and
``REE_assembly/evidence/planning/exq1056_mech343_q056_upstream_leg_design_refusal_20260918.md``.

Two things close that gap, both default-off and bit-identical when off:

  * ``REEConfig.dgpe_enable_differentiable_cem`` (default False) -- when True
    alongside the SD-061 master flag, ``REEAgent.__init__`` turns SD-055's
    consumer on so the temperature half actually acts.
  * ``temperature_lever_consumer_live`` below -- DIAGNOSTIC ONLY, set by
    ``REEAgent`` from the live hippocampal config. It changes no arithmetic; it
    exists so ``get_state()`` -- and therefore any manifest that records it --
    says whether the temperature half was live. A run reporting
    ``sd061_temperature_half_inert: True`` measured count-widening, whatever its
    docstring claims.

MECH-094
--------
``compute_proposal_gain(simulation_mode=True)`` returns the identity gain
(0, 1.0) without advancing diagnostics. A replay / DMN proposal must not be
widened by waking impasse. Matches the SD-035 / MECH-279 / MECH-313 / MECH-320
pattern.

See REE_assembly/docs/architecture/sd_061_difficulty_gated_proposal_entropy.md
and ree_core/cingulate/stuck_state_detector.py (the detector half).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DifficultyGatedProposalEntropyConfig:
    """SD-061 difficulty-gated proposal-entropy regulator configuration.

    Attributes:
        use_difficulty_gated_proposal_entropy : master switch. False = disabled
            (default, backward-compatible). REEAgent does not instantiate the
            regulator when False.
        candidate_widen_max : maximum EXTRA CEM candidates added at full
            stuck_score (s=1). extra = round(candidate_widen_max * s).
        temperature_gain_max : maximum FRACTIONAL lift on the within-class CEM
            sampling temperature at full stuck_score. temperature_gain =
            1 + temperature_gain_max * s. 0.0 disables the temperature lever
            (candidate-widening only).
        temperature_lever_consumer_live : DIAGNOSTIC ONLY -- does the lifted
            ``differentiable_cem_temperature`` have a live consumer this run
            (i.e. is SD-055's ``hippocampal.use_differentiable_cem`` True)?
            REEAgent sets it from the live hippocampal config. It participates
            in NO arithmetic; ``compute_proposal_gain`` is bit-identical with it
            True or False. It exists only so ``get_state()`` can report whether
            the temperature half of the gain actually acted. See the module
            docstring section "THE TEMPERATURE HALF IS INERT UNLESS SD-055 IS ON".
    """

    use_difficulty_gated_proposal_entropy: bool = False
    candidate_widen_max: int = 8
    temperature_gain_max: float = 1.0
    temperature_lever_consumer_live: bool = False


class DifficultyGatedProposalEntropy:
    """SD-061 difficulty-gated proposal-entropy regulator (waking-only).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance. Stateless
    between calls except for diagnostic counters; the decay of the gain is
    carried by the upstream StuckStateDetector's asymmetric EMA.

    Diagnostics tracked:
        _last_stuck_score        : float
        _last_extra_candidates   : int
        _last_temperature_gain   : float
        _n_calls                 : int
        _n_active                : int  (calls with a non-identity gain)
        _n_simulation_skips      : int
    """

    def __init__(
        self, config: "DifficultyGatedProposalEntropyConfig | None" = None
    ) -> None:
        self.config = (
            config
            if config is not None
            else DifficultyGatedProposalEntropyConfig()
        )
        c = self.config
        if c.candidate_widen_max < 0:
            raise ValueError(
                f"candidate_widen_max must be >= 0. Got {c.candidate_widen_max}."
            )
        if c.temperature_gain_max < 0.0:
            raise ValueError(
                "temperature_gain_max must be >= 0. Got "
                f"{c.temperature_gain_max}."
            )
        self._last_stuck_score: float = 0.0
        self._last_extra_candidates: int = 0
        self._last_temperature_gain: float = 1.0
        self._n_calls: int = 0
        self._n_active: int = 0
        self._n_simulation_skips: int = 0

    def compute_proposal_gain(
        self, stuck_score: float, simulation_mode: bool = False
    ) -> "tuple[int, float]":
        """Map a stuck_score to (extra_candidates, temperature_gain).

        Args:
            stuck_score : impasse score in [0, 1] from the StuckStateDetector.
                Clamped to [0, 1] defensively.
            simulation_mode : MECH-094 gate. When True, returns the identity
                gain (0, 1.0) and increments only the simulation-skip counter.

        Returns:
            (extra_candidates, temperature_gain):
              extra_candidates  -- non-negative int extra CEM candidates.
              temperature_gain  -- >= 1.0 multiplier on the within-class CEM
                                    sampling temperature.
        """
        if simulation_mode:
            self._n_simulation_skips += 1
            return (0, 1.0)
        s = max(0.0, min(1.0, float(stuck_score)))
        extra = int(round(float(self.config.candidate_widen_max) * s))
        temp_gain = 1.0 + float(self.config.temperature_gain_max) * s
        self._last_stuck_score = s
        self._last_extra_candidates = extra
        self._last_temperature_gain = temp_gain
        self._n_calls += 1
        if extra > 0 or temp_gain > 1.0:
            self._n_active += 1
        return (extra, temp_gain)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode diagnostic counters."""
        self._last_stuck_score = 0.0
        self._last_extra_candidates = 0
        self._last_temperature_gain = 1.0
        self._n_calls = 0
        self._n_active = 0
        self._n_simulation_skips = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "last_stuck_score": self._last_stuck_score,
            "last_extra_candidates": self._last_extra_candidates,
            "last_temperature_gain": self._last_temperature_gain,
            "sd061_dgpe_n_calls": self._n_calls,
            "sd061_dgpe_n_active": self._n_active,
            "sd061_dgpe_n_simulation_skips": self._n_simulation_skips,
            # Was the temperature half of the gain actually consumed this run?
            # See the module docstring: the lifted differentiable_cem_temperature
            # is read ONLY by SD-055's CEM refit, so with SD-055 off the
            # regulator's effective manipulation is count-widening alone.
            "sd061_temperature_lever_consumer_live": bool(
                self.config.temperature_lever_consumer_live
            ),
            "sd061_temperature_half_inert": bool(
                float(self.config.temperature_gain_max) > 0.0
                and not self.config.temperature_lever_consumer_live
            ),
        }
