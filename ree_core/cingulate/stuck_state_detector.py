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

WHICH AXES ACTUALLY ARRIVE (measured 2026-09-18, GFLAG-0352)
------------------------------------------------------------
Two of the four declared axes are absent under ordinary configurations, and the
combine below is a mean over PRESENT axes only -- so WHICH axes exist sets the
attainable maximum of ``stuck_score``, not just its value. Measured over an
ecological loop (``REEConfig.goal_stream`` + ``CausalGridWorldV2``, agent
selecting its own actions): ``goal_proximity`` 100/100 ticks, ``goal_salience``
100/100, ``score_margin`` 99/100, ``committed_action_class`` **0/100**,
``choice_difficulty`` **0/100**. With the progress axis saturated at 1.0 and the
margin axis at 0.0 the evidence is exactly ``mean(1.0, 0.0) = 0.5`` -- identical
to the default ``stuck_threshold``, approached from below by the EMA, so
``is_stuck`` never fires.

``choice_difficulty`` (the SD-032b dACC axis) needs FOUR conditions, not one.
``REEAgent.select_action`` writes ``_dacc_last_bundle`` only inside
``if self.dacc is not None and z_harm_a is not None:`` (``agent.py:7549``), so:

  1. ``use_dacc=True``            -- constructs ``agent.dacc``;
  2. ``use_affective_harm_stream=True`` -- constructs the ``AffectiveHarmEncoder``
     that produces ``z_harm_a`` (``latent/stack.py:1217``);
  3. the environment must emit ``harm_obs_a`` (``CausalGridWorldV2`` does);
  4. **the driver must forward it** -- ``agent.sense(..., obs_harm_a=...)``.
     ``act_with_split_obs`` calls ``sense(obs_body, obs_world)`` with no harm
     channel, so a driver using that convenience interface can NEVER populate
     the axis, at any config. ``experiments/_harness.py``,
     ``experiments/_lib/allon_training.py`` and the ``_lib/baselines/*`` modules
     all forward it correctly; a hand-rolled ``act_with_split_obs`` loop does not.

``committed_action_class`` needs a commitment to have occurred
(``e3._committed_trajectory``), i.e. a beta elevation -- which runs into
MECH-342's registered open failure ("no natural commit when score margins are
flat", V3-EXQ-629).

Neither condition is a defect in THIS module; both are recorded here because
MECH-343's ``what_would_answer`` MANDATES reporting the SD-032b contribution to
``stuck_score`` separately, and that is impossible on a run where the axis never
arrives. ``get_state()``'s ``sd061_n_present_*`` counters are what make the
difference legible in a manifest.

RESOLVED 2026-09-19 (user decision) -- THE AXIS MASK. What the detector should
do when axes are absent was an open design decision; it is now settled as
``declared_axes`` (see ``StuckStateDetectorConfig``). A run DECLARES its trigger;
the combination is taken over exactly the declared set; a declared axis that is
not wired REFUSES the run rather than silently rescaling. Threshold
recalibration was considered and REJECTED -- it would let ``is_stuck`` fire
without making the trigger attributable, redefining "stuck" as a function of
instrumentation rather than of the agent's state.
``declared_axes=None`` (the default) keeps the legacy mean-over-present
behaviour bit-identical, so nothing already recorded changes meaning.

Measured effect of the mask on the very loop above (2026-09-19, same seed):
legacy ``declared_axes=None`` reproduces the baseline exactly -- score pinned at
0.5000, duty 0.000. Declaring ``("progress",)`` alone lifts the score to 1.0000
with duty 0.980. **Both are unusable as a TRIGGER and for opposite reasons**: the
first never fires (G9 pole A), the second never stops firing and never decays
(G9 pole B), and MECH-343 requires a peak that exceeds threshold AND THEN
DECAYS. Which axes Q-056 should declare is therefore a live scientific question
and NOT settled by this build -- it is the subject of a separate decision chip.

The combined deficits are combined by ``mean`` (default) or ``max``
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
from typing import Deque, Optional, Tuple

# SD-061 (c): the axis names a run may DECLARE. Order is the canonical report
# order; membership is what `declared_axes` is validated against.
AXIS_NAMES: Tuple[str, ...] = ("progress", "margin", "diversity", "difficulty")


class StuckStateAxisUnavailable(RuntimeError):
    """A DECLARED detector axis received no input on a waking tick.

    SD-061 (c), user decision 2026-09-19. Raised -- never swallowed, never
    silently rescaled around -- because the alternative is the failure this
    decision exists to end: `combine_mode="mean"` over whichever axes happen to
    be present silently changes the attainable MAXIMUM of stuck_score, so a
    null cannot be attributed between "the agent was not stuck" and "the axis
    that would have said so never arrived".

    If you are seeing this for `difficulty`, the SD-032b dACC axis needs FOUR
    conditions and `use_dacc=True` is only the first:
      1. `use_dacc=True`                     -- constructs `agent.dacc`;
      2. `use_affective_harm_stream=True`    -- constructs the
         `AffectiveHarmEncoder` that produces `z_harm_a`;
      3. an environment that emits `harm_obs_a` (`CausalGridWorldV2` does);
      4. a driver that FORWARDS it: `agent.sense(..., obs_harm_a=...)`.
         `act_with_split_obs` calls `sense(obs_body, obs_world)` with no harm
         channel, so a driver on that convenience interface can never populate
         this axis at any config.
    If you are seeing it for `diversity`, the axis needs a commitment to have
    occurred (`e3._committed_trajectory`), which runs into MECH-342's open
    failure (V3-EXQ-629, "no natural commit when score margins are flat").
    """


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
        combine_mode : "mean" (default) or "max" over the combined axis set --
            the PRESENT axes when declared_axes is None, the DECLARED axes
            otherwise (see below).
        declared_axes : SD-061 (c), user decision 2026-09-19 -- the AXIS MASK.
            None (default) = legacy behaviour, bit-identical: combine over
            whichever axes happen to be present this tick.
            A tuple of axis names (any non-empty subset of
            ("progress", "margin", "diversity", "difficulty"), no duplicates)
            = this run DECLARES its trigger. Then:
              * the combination is taken over exactly the DECLARED axes -- an
                undeclared axis is IGNORED even when its input arrives, so the
                denominator is fixed by the declaration and cannot drift;
              * a declared axis whose INPUT is absent is an ERROR
                (``StuckStateAxisUnavailable``), never a silent rescale. That
                is the whole point of the decision: with mean-over-present,
                a missing axis silently changes the attainable MAXIMUM of
                stuck_score, and a null becomes unattributable between "not
                stuck" and "the axis that would have said so never arrived".
            WHY A MASK AND NOT A RECALIBRATED THRESHOLD: rescaling the
            threshold by the present-axis count would let is_stuck fire again
            without making the trigger attributable, and would redefine "stuck"
            as a function of instrumentation rather than of the agent's state.
            Rejected by the user, 2026-09-19.
        declared_axis_grace_ticks : how many waking ticks a DECLARED axis
            may go unseen before the detector concludes it is NOT WIRED
            and refuses. Inert when declared_axes is None. Exists because
            some axes are legitimately absent on the first tick(s) --
            `score_margin` is None until the first E3 selection has
            happened -- and refusing those would make the axis
            undeclarable. Inside the window a missing axis is
            UNDETERMINED (no advance, no partial combination), so this
            knob can only change WHEN a mis-wired run is told, never any
            measured quantity: a correctly-wired run never reaches
            either branch.
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
    declared_axes: Optional[Tuple[str, ...]] = None
    declared_axis_grace_ticks: int = 8


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
        # SD-061 (c): validate the declared axis set at CONSTRUCTION, so a
        # typo'd or empty declaration fails before any compute is spent rather
        # than mid-run. None = legacy mean-over-present (bit-identical).
        self._declared: Optional[Tuple[str, ...]] = None
        if c.declared_axes is not None:
            declared = tuple(c.declared_axes)
            if not declared:
                raise ValueError(
                    "declared_axes must name at least one axis (or be None for "
                    f"the legacy mean-over-present behaviour). Got {declared!r}."
                )
            unknown = [a for a in declared if a not in AXIS_NAMES]
            if unknown:
                raise ValueError(
                    f"declared_axes contains unknown axis name(s) {unknown!r}. "
                    f"Valid names are {list(AXIS_NAMES)}."
                )
            if len(set(declared)) != len(declared):
                raise ValueError(
                    f"declared_axes must not repeat an axis. Got {declared!r}."
                )
            # Canonicalise to AXIS_NAMES order so the recorded declaration and
            # the combination order do not depend on how the caller spelled it.
            self._declared = tuple(a for a in AXIS_NAMES if a in set(declared))
            if c.declared_axis_grace_ticks < 0:
                raise ValueError(
                    "declared_axis_grace_ticks must be >= 0. Got "
                    f"{c.declared_axis_grace_ticks}."
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
        # AXIS-PRESENCE diagnostics (added 2026-09-18, GFLAG-0352). The
        # _last_deficit_* fields above are 0.0 for BOTH "axis absent this tick"
        # and "axis present and measuring zero deficit", so they cannot
        # distinguish the two -- and the combine below is a mean over PRESENT
        # axes only, so WHICH axes exist sets the attainable maximum of
        # stuck_score. Without these counters a null is unattributable between
        # "the agent was not stuck" and "the axes that would have said so never
        # arrived". Pure diagnostics: nothing here feeds update()'s arithmetic.
        self._last_present_progress: bool = False
        self._last_present_margin: bool = False
        self._last_present_diversity: bool = False
        self._last_present_difficulty: bool = False
        self._n_present_progress: int = 0
        self._n_present_margin: int = 0
        self._n_present_diversity: int = 0
        self._n_present_difficulty: int = 0
        # SD-061 (c) mask diagnostics. A tick on which a DECLARED axis has its
        # input but cannot yet form a deficit (the progress window needs two
        # samples; the margin needs a pool of >= 2) is UNDETERMINED: the
        # detector does not advance. That is neither a refusal (the driver is
        # wiring the axis correctly) nor a rescale (no partial combination is
        # formed) -- it is "the declared combination is not computable yet",
        # the same no-advance shape as the MECH-094 simulation_mode path.
        self._n_undetermined_ticks: int = 0
        self._last_undetermined: bool = False
        self._last_undetermined_axes: Tuple[str, ...] = ()
        # Wiring-detection state for the declared-axis grace window.
        self._n_waking_calls: int = 0
        self._axis_seen = {a: False for a in AXIS_NAMES}

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

        # SD-061 (c) AXIS MASK -- decide BEFORE any state advances, so a refused
        # tick leaves the detector exactly as it was (this runs ahead of
        # _progress_deficit / _diversity_deficit appending to their windows).
        #
        # A declared axis whose INPUT is absent must never be silently combined
        # around. But "absent" has two causes and they need different answers:
        #
        #   NOT WIRED   -- the driver or config never supplies it (the
        #                  `difficulty` case: act_with_split_obs has no harm
        #                  channel at all). This is a mis-declared run and is a
        #                  REFUSAL.
        #   NOT YET     -- correctly wired, but this tick is too early. Measured
        #                  2026-09-19: `score_margin` is None on the FIRST tick
        #                  only, because agent.select_action leaves it None until
        #                  e3.last_scores exists, i.e. until the first E3
        #                  selection -- 99/100 ticks thereafter. Refusing that
        #                  would make `margin` undeclarable by any driver, which
        #                  is plainly not the decision's intent.
        #
        # A single tick's inputs cannot tell those apart; the RUN can. So:
        # an axis not yet seen inside the grace window is UNDETERMINED (no
        # advance, no partial combination -- the anti-rescale guarantee holds
        # absolutely); still unseen at the end of it is NOT WIRED and refuses;
        # and an axis that HAS been seen and then goes missing refuses at once,
        # because that is wiring breaking mid-run rather than warming up.
        #
        # declared_axis_grace_ticks only decides WHEN a mis-wired run is told;
        # it cannot affect any measured quantity, because a correctly-wired run
        # never reaches either branch.
        if self._declared is not None:
            _inputs = {
                "progress": goal_proximity,
                "margin": score_margin,
                "diversity": committed_action_class,
                "difficulty": choice_difficulty,
            }
            self._n_waking_calls += 1
            grace = int(self.config.declared_axis_grace_ticks)
            refuse: list = []
            for a in self._declared:
                if _inputs[a] is not None:
                    self._axis_seen[a] = True
                    continue
                if self._axis_seen[a]:
                    refuse.append((a, "arrived earlier in this run and has now stopped"))
                elif self._n_waking_calls > grace:
                    refuse.append(
                        (a, f"has never arrived in {self._n_waking_calls} waking ticks")
                    )
            if refuse:
                detail = "; ".join(f"{a} ({why})" for a, why in refuse)
                raise StuckStateAxisUnavailable(
                    f"SD-061 declared axis/axes {[a for a, _ in refuse]} "
                    f"unavailable: {detail}. "
                    f"Declared set: {list(self._declared)}; "
                    f"grace window: {grace} waking ticks. "
                    "Either wire the axis (see this exception's class docstring "
                    "-- 'difficulty' needs four conditions, of which use_dacc is "
                    "only the first) or do not declare it. The detector "
                    "deliberately does NOT fall back to a mean over the axes "
                    "that did arrive: that would silently change the attainable "
                    "maximum of stuck_score and make a null unattributable."
                )

        d_prog = self._progress_deficit(goal_proximity)
        d_marg = self._margin_deficit(score_margin, n_candidates)
        d_div = self._diversity_deficit(committed_action_class)
        d_diff = self._difficulty_deficit(choice_difficulty)
        self._last_deficit_progress = d_prog if d_prog is not None else 0.0
        self._last_deficit_margin = d_marg if d_marg is not None else 0.0
        self._last_deficit_diversity = d_div if d_div is not None else 0.0
        self._last_deficit_difficulty = d_diff if d_diff is not None else 0.0

        # AXIS-PRESENCE diagnostics -- see __init__. The four assignments above
        # collapse "absent" onto 0.0; these keep the distinction. Counters only,
        # read by get_state(); they feed nothing below.
        self._last_present_progress = d_prog is not None
        self._last_present_margin = d_marg is not None
        self._last_present_diversity = d_div is not None
        self._last_present_difficulty = d_diff is not None
        self._n_present_progress += int(self._last_present_progress)
        self._n_present_margin += int(self._last_present_margin)
        self._n_present_diversity += int(self._last_present_diversity)
        self._n_present_difficulty += int(self._last_present_difficulty)

        if self._declared is not None:
            # SD-061 (c): combine over exactly the DECLARED set. An undeclared
            # axis is ignored even when present, so the denominator is fixed by
            # the declaration and cannot drift with instrumentation.
            _deficits = {
                "progress": d_prog,
                "margin": d_marg,
                "diversity": d_div,
                "difficulty": d_diff,
            }
            undetermined = tuple(
                a for a in self._declared if _deficits[a] is None
            )
            self._last_undetermined = bool(undetermined)
            self._last_undetermined_axes = undetermined
            if undetermined:
                # The inputs ARE wired (the refusal above already proved that);
                # the declared combination simply is not computable yet -- the
                # progress window needs two samples, the margin needs a pool of
                # >= 2. Do NOT form a partial combination (that is the rescale
                # this decision forbids) and do NOT advance the EMA. The tick is
                # recorded and skipped, exactly like the simulation_mode path.
                self._n_undetermined_ticks += 1
                return self._stuck_score
            declared_deficits = [_deficits[a] for a in self._declared]
            if self.config.combine_mode == "max":
                combined = max(declared_deficits)
            else:
                combined = sum(declared_deficits) / float(len(declared_deficits))
        else:
            self._last_undetermined = False
            self._last_undetermined_axes = ()
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
        self._last_present_progress = False
        self._last_present_margin = False
        self._last_present_diversity = False
        self._last_present_difficulty = False
        self._n_present_progress = 0
        self._n_present_margin = 0
        self._n_present_diversity = 0
        self._n_present_difficulty = 0
        self._n_undetermined_ticks = 0
        self._last_undetermined = False
        self._last_undetermined_axes = ()
        self._n_waking_calls = 0
        self._axis_seen = {a: False for a in AXIS_NAMES}

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
            # AXIS PRESENCE (see __init__). last_deficit_* is 0.0 for BOTH an
            # absent axis and a present-but-zero one; these separate them, and
            # n_axes_present_last is what sets the attainable maximum of
            # stuck_score under combine_mode="mean" (a mean over PRESENT axes).
            "sd061_last_present_progress": self._last_present_progress,
            "sd061_last_present_margin": self._last_present_margin,
            "sd061_last_present_diversity": self._last_present_diversity,
            "sd061_last_present_difficulty": self._last_present_difficulty,
            "sd061_n_present_progress": self._n_present_progress,
            "sd061_n_present_margin": self._n_present_margin,
            "sd061_n_present_diversity": self._n_present_diversity,
            "sd061_n_present_difficulty": self._n_present_difficulty,
            "sd061_n_axes_present_last": int(
                self._last_present_progress
                + self._last_present_margin
                + self._last_present_diversity
                + self._last_present_difficulty
            ),
            # SD-061 (c) AXIS MASK. declared_axes is None on a legacy
            # mean-over-present run; a list names this run's declared trigger,
            # which is what makes a null attributable to a named axis set.
            # n_undetermined_ticks counts ticks on which the declared
            # combination was not yet computable and the EMA did NOT advance --
            # a large value means the run is measuring far fewer ticks than it
            # appears to, and should be read before any verdict.
            "sd061_declared_axes": (
                list(self._declared) if self._declared is not None else None
            ),
            "sd061_n_undetermined_ticks": self._n_undetermined_ticks,
            "sd061_last_undetermined": self._last_undetermined,
            "sd061_last_undetermined_axes": list(self._last_undetermined_axes),
            "sd061_axes_ever_seen": [
                a for a in AXIS_NAMES if self._axis_seen[a]
            ],
        }
