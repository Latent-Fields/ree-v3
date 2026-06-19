"""SD-061 component 1: stuck-state / impasse detector (MECH-343 blocker part 2a).

The detector half of the difficulty-gated proposal-entropy regulator. It
integrates signals REE already computes -- goal-progress stall, dACC choice
difficulty, E3 score margin, committed-action-class diversity -- into a graded
``stuck_score`` in [0, 1] plus a binary ``is_stuck``, GUARDED by goal salience
so it fires on *stuck-with-a-goal* (impasse), not goal-abandonment.

WHY THIS EXISTS
---------------
MECH-343 (difficulty_gated_proposal_entropy) is ``substrate_conditional`` and
``v3_pending`` because two upstream pieces were unbuilt: (1) the
modulatory-bias-selection-authority gap -- now IMPLEMENTED (569i top-k
shortlist conversion) -- and (2) a difficulty-gated proposal-entropy regulator
(*stuck-state detector + transient CEM temperature/candidate-count gain +
decay*) -- NOT designed. SD-061 builds piece (2). This module is its detector;
``ree_core/policy/difficulty_gated_proposal_entropy.py`` is the regulator that
maps ``stuck_score`` to a transient gain on the ARC-018 hippocampal/CEM
proposal layer.

WHAT THIS IS NOT (falsifiable distinctions; the substrate must not collapse
into any of these):

  * NOT MECH-313 (noise_floor). MECH-313 lifts the *action-selection* softmax
    temperature *state-independently* every waking tick. This detector is
    state-DEPENDENT (only an impasse-with-goal raises it) and its consumer
    acts on the *proposal-generation* layer, not action-selection softmax.
  * NOT a raw dACC choice-difficulty readout. dACC emits ``choice_difficulty``
    (std of per-candidate EVs); this detector INTEGRATES it with goal-progress
    stall + score margin + committed diversity into a single graded impasse
    state, gated by goal salience.
  * NOT the MECH-342 release-pressure accumulator. MECH-342 accumulates toward
    *releasing* an already-committed beta latch when readiness DEGRADES. This
    detector accumulates impasse evidence to *widen proposals* upstream of
    commitment; the two operate at opposite ends of the commitment loop.

SIGNAL -> DEFICIT MAPPING
-------------------------
Each axis is an Optional input; a ``None`` axis is inert (contributes neither
impasse evidence nor a recovery vote). Each present axis maps to a [0, 1]
deficit (higher = more stuck-evidence):

  goal-progress stall : internal short-window of goal_proximity; stall deficit
      rises when the recent improvement (max-over-window minus current... see
      _progress_deficit) falls at/below stall_eps. Goal pursued but not
      advancing -> impasse.
  score margin        : low E3 first-action margin (REE lower-is-better) =
      candidates indistinguishable = decision impasse. deficit =
      clip((margin_floor - margin)/margin_floor, 0, 1).
  committed diversity : low unique-class fraction over a recent committed-action
      window = behavioural lock-in. deficit = clip((div_floor - frac)/div_floor,
      0, 1).
  choice difficulty   : dACC choice_difficulty is the std of per-candidate EVs
      -- SMALL spread = HARD (ambiguous) choice. deficit =
      clip((diff_ref - choice_difficulty)/diff_ref, 0, 1) (inverted).

The present-axis deficits are combined by ``mean`` (default) or ``max``
(``combine_mode``). The combined impasse evidence is GATED by goal salience:
when goal_salient is False (no active goal / drive), the tick contributes 0 --
absence of progress without a goal is not impasse, it is rest. The gated
evidence drives an EMA accumulator (rise faster than it falls, giving the
substrate a brief hysteretic memory of recent impasse), clamped to [0, 1].
``is_stuck = stuck_score >= stuck_threshold``.

MECH-094
--------
``update(simulation_mode=True)`` returns the unchanged ``stuck_score`` without
advancing state and increments only the simulation-skip counter. A replay /
DMN tick must not accumulate waking impasse evidence. Matches the SD-035 /
MECH-279 / MECH-313 / MECH-320 / MECH-342 / commit_readiness pattern.

See REE_assembly/docs/architecture/sd_061_difficulty_gated_proposal_entropy.md
and ree_core/policy/difficulty_gated_proposal_entropy.py (the consumer).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class StuckStateDetectorConfig:
    """SD-061 stuck-state detector configuration.

    Attributes:
        use_stuck_state_detector : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate the detector
            when False.
        progress_window : number of recent goal_proximity readings retained for
            the goal-progress-stall axis.
        progress_stall_eps : recent goal-proximity improvement at/below this is
            treated as a stall (full deficit on that axis).
        score_margin_floor : E3 first-action margin (REE lower-is-better) at/below
            this contributes a decisiveness/impasse deficit.
        committed_diversity_window : recent committed-action-class window for the
            behavioural-lock-in axis.
        committed_diversity_floor : unique-class fraction at/below this
            contributes a committed-diversity deficit.
        choice_difficulty_ref : dACC choice_difficulty reference; readings
            at/below this contribute an ambiguity deficit (inverted -- small EV
            spread = hard choice).
        goal_salience_floor : goal_salience at/above this counts the tick as
            goal-pursuing (impasse-eligible). Below it the tick contributes 0
            (no goal -> not stuck).
        ema_alpha_rise : EMA rate when impasse evidence exceeds the current
            stuck_score (accumulation).
        ema_alpha_fall : EMA rate when impasse evidence is below the current
            stuck_score (decay/recovery). ema_alpha_rise >> ema_alpha_fall gives
            the hysteretic "stuck persists briefly after relief" behaviour and
            implements the entropy DECAY half of MECH-343.
        stuck_threshold : stuck_score at/above this sets is_stuck=True.
        combine_mode : "mean" (default) or "max" over the present-axis deficits.
    """

    use_stuck_state_detector: bool = False
    progress_window: int = 8
    progress_stall_eps: float = 0.01
    score_margin_floor: float = 0.05
    committed_diversity_window: int = 8
    committed_diversity_floor: float = 0.34
    choice_difficulty_ref: float = 0.05
    goal_salience_floor: float = 0.05
    ema_alpha_rise: float = 0.3
    ema_alpha_fall: float = 0.05
    stuck_threshold: float = 0.5
    combine_mode: str = "mean"


class StuckStateDetector:
    """SD-061 stuck-state / impasse detector (waking-only).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance. Maintains
    a [0, 1] ``stuck_score`` EMA plus small recent-history windows for the
    goal-progress and committed-diversity axes.

    Diagnostics tracked:
        _stuck_score             : float (the accumulator)
        _last_is_stuck           : bool
        _last_combined_deficit   : float
        _last_deficit_progress   : float
        _last_deficit_margin     : float
        _last_deficit_diversity  : float
        _last_deficit_difficulty : float
        _last_goal_salient       : bool
        _n_ticks                 : int
        _n_stuck_ticks           : int
        _n_simulation_skips      : int
    """

    def __init__(self, config: "StuckStateDetectorConfig | None" = None) -> None:
        self.config = config if config is not None else StuckStateDetectorConfig()
        c = self.config
        if c.progress_window < 2:
            raise ValueError(f"progress_window must be >= 2. Got {c.progress_window}.")
        if c.score_margin_floor < 0.0:
            raise ValueError(
                f"score_margin_floor must be >= 0. Got {c.score_margin_floor}."
            )
        if c.committed_diversity_window < 1:
            raise ValueError(
                "committed_diversity_window must be >= 1. Got "
                f"{c.committed_diversity_window}."
            )
        if not (0.0 <= c.committed_diversity_floor <= 1.0):
            raise ValueError(
                "committed_diversity_floor must be in [0, 1]. Got "
                f"{c.committed_diversity_floor}."
            )
        if c.choice_difficulty_ref < 0.0:
            raise ValueError(
                f"choice_difficulty_ref must be >= 0. Got {c.choice_difficulty_ref}."
            )
        if not (0.0 < c.ema_alpha_rise <= 1.0):
            raise ValueError(
                f"ema_alpha_rise must be in (0, 1]. Got {c.ema_alpha_rise}."
            )
        if not (0.0 < c.ema_alpha_fall <= 1.0):
            raise ValueError(
                f"ema_alpha_fall must be in (0, 1]. Got {c.ema_alpha_fall}."
            )
        if not (0.0 <= c.stuck_threshold <= 1.0):
            raise ValueError(
                f"stuck_threshold must be in [0, 1]. Got {c.stuck_threshold}."
            )
        if c.combine_mode not in ("mean", "max"):
            raise ValueError(
                f"combine_mode must be 'mean' or 'max'. Got {c.combine_mode!r}."
            )
        self._progress: Deque[float] = deque(maxlen=int(c.progress_window))
        self._committed_classes: Deque[int] = deque(
            maxlen=int(c.committed_diversity_window)
        )
        self._stuck_score: float = 0.0
        self._last_is_stuck: bool = False
        self._last_combined_deficit: float = 0.0
        self._last_deficit_progress: float = 0.0
        self._last_deficit_margin: float = 0.0
        self._last_deficit_diversity: float = 0.0
        self._last_deficit_difficulty: float = 0.0
        self._last_goal_salient: bool = False
        self._n_ticks: int = 0
        self._n_stuck_ticks: int = 0
        self._n_simulation_skips: int = 0

    # ------------------------------------------------------------------
    # Per-axis deficits
    # ------------------------------------------------------------------
    def _progress_deficit(self, goal_proximity: Optional[float]) -> Optional[float]:
        """Goal-progress-stall deficit in [0, 1], or None when no signal.

        Appends the current proximity to the window, then measures the best
        improvement available within the window (max proximity minus the oldest
        proximity). A small/negative improvement = stall = full deficit.
        """
        if goal_proximity is None:
            return None
        self._progress.append(float(goal_proximity))
        if len(self._progress) < 2:
            return None  # not enough history yet
        improvement = max(self._progress) - self._progress[0]
        eps = float(self.config.progress_stall_eps)
        if eps <= 0.0:
            return 0.0
        # improvement >= eps -> deficit 0 (advancing); <= 0 -> deficit 1 (stalled).
        deficit = (eps - improvement) / eps
        return max(0.0, min(1.0, deficit))

    def _margin_deficit(
        self, score_margin: Optional[float], n_candidates: int
    ) -> Optional[float]:
        """Decision-impasse deficit in [0, 1], or None when no signal."""
        if score_margin is None or n_candidates < 2:
            return None
        floor = float(self.config.score_margin_floor)
        if floor <= 0.0:
            return 0.0
        deficit = (floor - float(score_margin)) / floor
        return max(0.0, min(1.0, deficit))

    def _diversity_deficit(
        self, committed_action_class: Optional[int]
    ) -> Optional[float]:
        """Committed-action-class lock-in deficit in [0, 1], or None when no signal."""
        if committed_action_class is None:
            return None
        self._committed_classes.append(int(committed_action_class))
        n = len(self._committed_classes)
        if n < 1:
            return None
        frac = len(set(self._committed_classes)) / float(n)
        floor = float(self.config.committed_diversity_floor)
        if floor <= 0.0:
            return 0.0
        deficit = (floor - frac) / floor
        return max(0.0, min(1.0, deficit))

    def _difficulty_deficit(
        self, choice_difficulty: Optional[float]
    ) -> Optional[float]:
        """dACC choice-difficulty (inverted) deficit in [0, 1], or None.

        dACC choice_difficulty is the std of per-candidate EVs -- SMALL spread
        = HARD (ambiguous) choice -> high deficit.
        """
        if choice_difficulty is None:
            return None
        ref = float(self.config.choice_difficulty_ref)
        if ref <= 0.0:
            return 0.0
        deficit = (ref - float(choice_difficulty)) / ref
        return max(0.0, min(1.0, deficit))

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def update(
        self,
        goal_proximity: Optional[float] = None,
        score_margin: Optional[float] = None,
        n_candidates: int = 0,
        committed_action_class: Optional[int] = None,
        choice_difficulty: Optional[float] = None,
        goal_salience: Optional[float] = None,
        simulation_mode: bool = False,
    ) -> float:
        """Advance the impasse accumulator one waking tick; return stuck_score.

        Args:
            goal_proximity : GoalState.goal_proximity(z_world) in [0, 1] (or
                None when no active goal). Drives the goal-progress-stall axis.
            score_margin : per-candidate first-action margin
                (sorted(scores)[1] - sorted(scores)[0], REE lower-is-better) or
                None. Drives the decision-impasse axis.
            n_candidates : candidate-pool size the margin came from.
            committed_action_class : the action class committed this tick (or
                None). Drives the committed-diversity axis over a rolling window.
            choice_difficulty : dACC bundle choice_difficulty (std of per-
                candidate EVs) or None. Drives the (inverted) ambiguity axis.
            goal_salience : a [0, 1] proxy for "a goal is being pursued"
                (e.g. drive_level, or goal_norm). When below goal_salience_floor
                the tick contributes 0 impasse evidence (no goal -> not stuck).
                None is treated as not-salient (gates evidence to 0).
            simulation_mode : MECH-094 gate. When True, no state advance; returns
                the unchanged stuck_score.

        Returns:
            The updated stuck_score in [0, 1].
        """
        if simulation_mode:
            self._n_simulation_skips += 1
            return self._stuck_score

        d_prog = self._progress_deficit(goal_proximity)
        d_marg = self._margin_deficit(score_margin, n_candidates)
        d_div = self._diversity_deficit(committed_action_class)
        d_diff = self._difficulty_deficit(choice_difficulty)
        self._last_deficit_progress = d_prog if d_prog is not None else 0.0
        self._last_deficit_margin = d_marg if d_marg is not None else 0.0
        self._last_deficit_diversity = d_div if d_div is not None else 0.0
        self._last_deficit_difficulty = d_diff if d_diff is not None else 0.0

        present = [d for d in (d_prog, d_marg, d_div, d_diff) if d is not None]
        if present:
            if self.config.combine_mode == "max":
                combined = max(present)
            else:
                combined = sum(present) / float(len(present))
        else:
            combined = 0.0

        # Goal-salience guard: impasse only counts while a goal is pursued.
        salient = (
            goal_salience is not None
            and float(goal_salience) >= float(self.config.goal_salience_floor)
        )
        self._last_goal_salient = salient
        evidence = combined if salient else 0.0
        self._last_combined_deficit = evidence

        # Asymmetric EMA: rise faster than fall -> hysteretic decay (MECH-343
        # "entropy narrows once a workable candidate is found").
        if evidence >= self._stuck_score:
            alpha = float(self.config.ema_alpha_rise)
        else:
            alpha = float(self.config.ema_alpha_fall)
        self._stuck_score = (1.0 - alpha) * self._stuck_score + alpha * evidence
        self._stuck_score = max(0.0, min(1.0, self._stuck_score))

        self._n_ticks += 1
        self._last_is_stuck = self._stuck_score >= float(self.config.stuck_threshold)
        if self._last_is_stuck:
            self._n_stuck_ticks += 1
        return self._stuck_score

    def get_stuck_score(self) -> float:
        """Return the current stuck_score in [0, 1]."""
        return self._stuck_score

    def is_stuck(self) -> bool:
        """Return the binary stuck gate from the most recent update."""
        return self._last_is_stuck

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode state and diagnostic counters."""
        self._progress.clear()
        self._committed_classes.clear()
        self._stuck_score = 0.0
        self._last_is_stuck = False
        self._last_combined_deficit = 0.0
        self._last_deficit_progress = 0.0
        self._last_deficit_margin = 0.0
        self._last_deficit_diversity = 0.0
        self._last_deficit_difficulty = 0.0
        self._last_goal_salient = False
        self._n_ticks = 0
        self._n_stuck_ticks = 0
        self._n_simulation_skips = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "stuck_score": self._stuck_score,
            "is_stuck": self._last_is_stuck,
            "last_combined_deficit": self._last_combined_deficit,
            "last_deficit_progress": self._last_deficit_progress,
            "last_deficit_margin": self._last_deficit_margin,
            "last_deficit_diversity": self._last_deficit_diversity,
            "last_deficit_difficulty": self._last_deficit_difficulty,
            "last_goal_salient": self._last_goal_salient,
            "sd061_n_ticks": self._n_ticks,
            "sd061_n_stuck_ticks": self._n_stuck_ticks,
            "sd061_n_simulation_skips": self._n_simulation_skips,
        }
