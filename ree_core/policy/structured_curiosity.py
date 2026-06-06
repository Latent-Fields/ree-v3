"""MECH-314: structured_curiosity_bonus (frontopolar exploration / EFE analog).

Child claim of ARC-065 (behavioral_diversity_generation_pathway). Sibling
to MECH-313 stochastic_noise_floor: where MECH-313 lifts the softmax
temperature uniformly (state-independent), MECH-314 produces a per-
candidate (or broadcast-scalar in Phase 1) score-bias on E3 candidate
scoring that rewards information gain or recency-of-encounter.

Three sub-flavours registered separately as MECH-314a/b/c (Pull 1
SYNTHESIS R3 explicitly recommended NOT collapsing them prematurely):

  MECH-314a: striatal novelty bonus (Wittmann et al. 2008 Neuron;
             ventral striatum responds to novelty independent of RPE).
             Sub-flavour where the bonus tracks recency-of-encounter.
             Phase 1 signal source: minimum distance from candidate's
             first-step z_world to the nearest ACTIVE ResidueField RBF
             center. Per-candidate (genuinely [K]).

  MECH-314b: frontopolar uncertainty-driven curiosity (Daw et al. 2006
             Nature; rostrolateral PFC exploration bonus). Sub-flavour
             where the bonus tracks epistemic value -- "how much would
             I learn." Phase 1 signal source: e3._running_variance
             (scalar). Phase 1 instantiation: BROADCAST scalar across
             [K]. Per-candidate refinement (E1 LSTM forward variance
             over candidates) deferred to Phase 2.

  MECH-314c: learning-progress curiosity (Schmidhuber 1991 compression-
             progress; Pathak et al. 2017 forward-model PE).
             Sub-flavour where the bonus tracks the rate-of-improvement
             of the agent's predictive model. Phase 1 signal source:
             EMA of |PE_t - PE_{t-K}| where PE is fed in via
             update_prediction_error() each tick. Phase 1 instantiation:
             BROADCAST scalar across [K]. Per-candidate refinement
             requires per-candidate learning-progress estimation,
             deferred to Phase 2.

PHASE 1 HONEST SCOPING

  314a is genuinely per-candidate; 314b and 314c are state-dependent
  global scalars broadcast across [K] in Phase 1. The architectural
  shape is correct (bonus magnitude varies with global uncertainty /
  learning-progress; the substrate exposes the falsification surface).
  Q-044's three-arm ablation is a FLAG-SET decision -- the substrate
  guarantees each sub-flavour can be turned on/off independently.
  Distinguishable behavioural signatures per sub-flavour at the
  candidate-selection level require Phase 2 per-candidate refinement
  (forward-variance head; per-candidate learning-progress estimation),
  which is OUT OF SCOPE for this substrate landing.

INTEGRATION SITE

  REEAgent.select_action(): after the MECH-295 liking-bridge score_bias
  block, BEFORE the MECH-313 noise_floor temperature lift (curiosity
  affects scores; noise floor affects temperature; orthogonal). Reuses
  the per-candidate first-step z_world summary tensor (cand_world_
  summaries / m295_summaries) when prior consumers built it; else
  builds fresh from candidates. Composes additively into
  dacc_score_bias (E3 lower-is-better convention; bonus is non-positive
  so curiosity makes novel/uncertain/learning-progress-rich candidates
  more attractive).

LIT-PULL VERDICTS (resolved defaults; see Pull 1 SYNTHESIS at
evidence/literature/targeted_review_arc_065_behavioral_diversity_generation/
SYNTHESIS.md, 9 entries, lit_conf 0.78-0.82, supports-direction)

  R1 BOTH-CHANNELS-NEEDED (conf 0.85): noise floor (MECH-313) AND
  structured curiosity (this substrate) both required. Wilson 2014 +
  Faisal 2008 + Friston 2015.

  R3 PROMOTE-TO-CLUSTER, sub-flavour split (conf 0.82 / 0.78): three
  sub-flavours registered as separate sub-claims rather than collapsed.
  314a anchored Wittmann 2008 striatal-novelty; 314b anchored Daw 2006
  frontopolar-exploration + Friston 2010/2015 EFE; 314c anchored
  Schmidhuber 1991 / Pathak 2017 (least biologically anchored;
  potentially-discardable-if-314a+314b-suffice per Q-044).

  R4 continuous in computation, triggered in dominance (conf 0.80):
  bonus computed every tick; magnitude scales with model uncertainty /
  novelty. Phase 1 instantiation is fully continuous (no triggering
  predicate); Phase 2 may add an MECH-104 volatility-surprise gate on
  the 314b weight.

  Magnitudes intentionally NOT pinned by the lit-pull. Q-043
  (relative-weight calibration MECH-313 vs MECH-314) and Q-044 (sub-
  flavour independence) are the empirical resolution paths via
  parametric sweep on V3-EXQ-543b/c and three-arm ablation.

  Default values (all sub-flavour weights 0.05; bias_scale 0.1) are
  conservative starting points that produce non-trivial bias under any
  realistic sub-signal magnitude without dominating the existing
  dACC / lateral_pfc / ofc / mech295 score_bias chain. Q-043 / Q-044
  calibrate.

DISTINCTION FROM MECH-313

  MECH-313: state-independent softmax temperature lift. After argmax
  behaviour, raises probability mass on every NON-A class equally.
  Architecturally affects the temperature kwarg into e3.select(...).

  MECH-314: state-dependent score-bias (314a per-candidate; 314b/c
  broadcast scalar). After argmax behaviour, makes novel / uncertain /
  learning-progress-rich candidates relatively more attractive.
  Architecturally affects the score_bias kwarg into e3.select(...).

  Both can coexist independently: MECH-313 OFF + MECH-314 ON is a
  valid configuration (per-candidate bias without softmax lift), as is
  the converse. Q-043 calibrates relative weights via parametric sweep.

MECH-094

  compute_score_bias(simulation_mode=True) returns zeros[K] and
  increments only the simulation-skip counter; the action-selection
  signal does not propagate to replay / DMN paths.
  update_prediction_error(simulation_mode=True) is a no-op on the EMA
  buffer (the LP signal does not advance on simulation paths).
  Match the SD-035 / MECH-279 / gated_policy / MECH-313 simulation_mode
  pattern.

ARCHITECTURAL-PLACEMENT OPEN QUESTION

  Phase 1 chooses a SEPARATE module at the e3.select() call site, in
  parallel with MECH-313 NoiseFloor and the GatedPolicy bias
  composition chain. The same Phase 1 placement-vs-consolidation note
  that MECH-313 carries applies here: whether MECH-313 / MECH-314 /
  MECH-318 / MECH-319 ultimately consolidate into one policy-layer
  module is OPEN pending the other ARC-065 substrates landing. Q-043
  / Q-045 calibration may motivate consolidation; for now the
  separate-module choice keeps each substrate independently togglable
  (which is what Q-044 needs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


@dataclass
class StructuredCuriosityConfig:
    """MECH-314 structured-curiosity-bonus configuration.

    Master flag and three independent sub-flavour switches. Sub-flavour
    switches are consulted only when use_structured_curiosity=True.

    Attributes:
        use_structured_curiosity : master switch. False = disabled
            (default, backward-compatible). REEAgent does not
            instantiate StructuredCuriosity when False.
        use_curiosity_novelty : MECH-314a switch. When True (and master
            is True), the novelty sub-flavour contributes to the
            score-bias output. Default True (so flag-set Q-044 ablation
            is "turn the master on, then flip individual sub-flavours
            off").
        use_curiosity_uncertainty : MECH-314b switch. Default True.
        use_curiosity_learning_progress : MECH-314c switch. Default
            True.
        curiosity_novelty_weight : MECH-314a magnitude. Default 0.05.
            Q-044 calibrates.
        curiosity_uncertainty_weight : MECH-314b magnitude. Default
            0.05.
        curiosity_learning_progress_weight : MECH-314c magnitude.
            Default 0.05.
        curiosity_bias_scale : hard clamp on the absolute value of the
            output bias [K]. Default 0.1, mirrors lateral_pfc_bias_scale
            so Phase 1 magnitudes are comparable to existing PFC-side
            score_bias contributions.
        curiosity_lp_ema_alpha : EMA smoothing for the 314c learning-
            progress signal. Default 0.1, matches the existing
            pe_ema_alpha pattern (~10-tick window).
        curiosity_lp_window_k : Lag K in |PE_t - PE_{t-K}|. Default 5
            ticks. Schmidhuber 1991 first-difference framing; the
            actual window is the convolution of K + the EMA alpha.
    """

    use_structured_curiosity: bool = False
    use_curiosity_novelty: bool = True
    use_curiosity_uncertainty: bool = True
    use_curiosity_learning_progress: bool = True
    curiosity_novelty_weight: float = 0.05
    curiosity_uncertainty_weight: float = 0.05
    curiosity_learning_progress_weight: float = 0.05
    curiosity_bias_scale: float = 0.1
    curiosity_lp_ema_alpha: float = 0.1
    curiosity_lp_window_k: int = 5

    # MECH-314a Phase 2 (Candidate 5A) novelty-source + augmentation knobs.
    # See REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
    # section 6. ALL DEFAULTS REPRODUCE PHASE 1 (bit-identical OFF): the
    # novelty source stays the harm-coupled ResidueField centers and no
    # first-action augmentation is applied.
    #
    # novelty_source : "residue" (Phase-1 ResidueField centers; default),
    #     "visitation" (rolling z_world buffer supplied by the caller),
    #     "auto" (use the visitation buffer when non-empty, else residue).
    # use_first_action_onehot : master switch for the augmentation leg.
    # first_action_augmentation_policy : "never" (default), "always", "auto"
    #     ("auto" engages after the un-augmented per-candidate spread stays
    #     below min_spread_threshold for min_spread_consecutive_ticks, and
    #     disengages once it recovers to/above the threshold).
    # min_spread_threshold / min_spread_consecutive_ticks : auto-policy gate.
    novelty_source: str = "residue"
    use_first_action_onehot: bool = False
    first_action_augmentation_policy: str = "never"
    min_spread_threshold: float = 0.01
    min_spread_consecutive_ticks: int = 5


class StructuredCuriosity:
    """MECH-314 structured-curiosity-bonus regulator (parent + 3 sub-flavours).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance.
    Each waking tick exposes:

        compute_score_bias(candidate_world_summaries, residue_field=None,
                           e3=None, simulation_mode=False) -> [K] tensor
        update_prediction_error(pe_scalar, simulation_mode=False) -> None
        get_state() -> dict
        reset() -> None

    Diagnostics tracked per call:
        _last_n_simulation_skips           : int
        _n_waking_calls                    : int
        _last_novelty_norm                 : float (mean ||delta|| over candidates)
        _last_uncertainty_signal           : float (e3 running variance read)
        _last_learning_progress_signal     : float (current LP EMA value)
        _last_n_active_residue_centers     : int
        _last_n_subflavours_fired          : int (in [0, 3])
        _last_bias_max_abs                 : float
    """

    def __init__(
        self,
        config: "Optional[StructuredCuriosityConfig]" = None,
    ) -> None:
        self.config = config if config is not None else StructuredCuriosityConfig()
        cfg = self.config
        # Validation
        if cfg.curiosity_novelty_weight < 0.0:
            raise ValueError(
                "curiosity_novelty_weight must be >= 0 (it is an additive "
                "magnitude on a non-negative novelty distance). Got "
                f"{cfg.curiosity_novelty_weight}."
            )
        if cfg.curiosity_uncertainty_weight < 0.0:
            raise ValueError(
                "curiosity_uncertainty_weight must be >= 0. Got "
                f"{cfg.curiosity_uncertainty_weight}."
            )
        if cfg.curiosity_learning_progress_weight < 0.0:
            raise ValueError(
                "curiosity_learning_progress_weight must be >= 0. Got "
                f"{cfg.curiosity_learning_progress_weight}."
            )
        if cfg.curiosity_bias_scale <= 0.0:
            raise ValueError(
                "curiosity_bias_scale must be > 0 (clamp magnitude). "
                f"Got {cfg.curiosity_bias_scale}."
            )
        if not (0.0 < cfg.curiosity_lp_ema_alpha <= 1.0):
            raise ValueError(
                "curiosity_lp_ema_alpha must be in (0, 1]. Got "
                f"{cfg.curiosity_lp_ema_alpha}."
            )
        if cfg.curiosity_lp_window_k < 1:
            raise ValueError(
                "curiosity_lp_window_k must be >= 1. Got "
                f"{cfg.curiosity_lp_window_k}."
            )
        # MECH-314a Phase 2 (Candidate 5A) validation.
        if cfg.novelty_source not in ("residue", "visitation", "auto"):
            raise ValueError(
                "novelty_source must be one of {'residue','visitation','auto'}. "
                f"Got {cfg.novelty_source!r}."
            )
        if cfg.first_action_augmentation_policy not in ("never", "auto", "always"):
            raise ValueError(
                "first_action_augmentation_policy must be one of "
                "{'never','auto','always'}. Got "
                f"{cfg.first_action_augmentation_policy!r}."
            )
        if cfg.min_spread_threshold < 0.0:
            raise ValueError(
                "min_spread_threshold must be >= 0. Got "
                f"{cfg.min_spread_threshold}."
            )
        if cfg.min_spread_consecutive_ticks < 1:
            raise ValueError(
                "min_spread_consecutive_ticks must be >= 1. Got "
                f"{cfg.min_spread_consecutive_ticks}."
            )

        # 314c learning-progress state
        self._pe_ring: list[float] = []        # ring buffer of last (K+1) PE scalars
        self._lp_ema: float = 0.0              # EMA of |PE_t - PE_{t-K}|
        self._lp_seeded: bool = False          # whether _lp_ema has been seeded

        # MECH-314a Phase 2 (Candidate 5A) auto-augmentation state.
        self._below_threshold_streak: int = 0  # consecutive low-spread ticks
        self._augmentation_engaged: bool = False  # auto-policy latch

        # Diagnostics
        self._last_n_simulation_skips: int = 0
        self._n_waking_calls: int = 0
        self._last_novelty_norm: float = 0.0
        self._last_uncertainty_signal: float = 0.0
        self._last_learning_progress_signal: float = 0.0
        self._last_n_active_residue_centers: int = 0
        self._last_n_subflavours_fired: int = 0
        self._last_bias_max_abs: float = 0.0
        # MECH-314a Phase 2 diagnostics.
        self._last_novelty_source_used: str = "none"
        self._last_candidate_spread: float = 0.0
        self._last_augmentation_engaged: bool = False
        self._last_n_visitation_entries: int = 0
        self._n_augmentation_engaged_ticks: int = 0

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def compute_score_bias(
        self,
        candidate_world_summaries: torch.Tensor,
        residue_field: object = None,
        e3: object = None,
        simulation_mode: bool = False,
        visitation_source: Any = None,
        first_action_onehots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-candidate curiosity score-bias [K].

        Lower-is-better convention: the returned tensor is non-positive
        wherever any active sub-flavour fires (curiosity reduces the
        candidate's score so it becomes more attractive). Composed
        additively with dacc_score_bias / lateral_pfc / ofc / gated_policy
        / mech295 in REEAgent.select_action.

        Args:
            candidate_world_summaries : [K, world_dim] tensor of
                per-candidate first-step z_world summaries (built by
                the caller; reused from the lateral_pfc / ofc / mech295
                composition chain when present).
            residue_field : optional ResidueField. Required for 314a
                novelty signal; when None or has no active centers,
                314a contributes zero.
            e3 : optional E3TrajectorySelector. Required for 314b
                uncertainty signal; when None, 314b contributes zero.
            simulation_mode : MECH-094 gate. When True, returns
                zeros[K] and increments only the simulation-skip
                counter.
            visitation_source : MECH-314a Phase 2. Optional rolling
                z_world visitation buffer (collections.deque / list /
                [N, world_dim] tensor) used as the 314a novelty
                comparison set when novelty_source is "visitation" or
                "auto". Ignored when novelty_source is "residue".
            first_action_onehots : MECH-314a Phase 2. Optional
                [K, action_dim] per-candidate first-action one-hot.
                Augments the candidate signatures before the RBF
                distance when the augmentation policy engages
                (substrate-robustness bypass). Ignored when the
                augmentation policy is "never" or
                use_first_action_onehot is False.

        Returns:
            [K] tensor on the same device/dtype as
            candidate_world_summaries. All-zero when the master would
            be off (caller should not invoke this path), when every
            sub-flavour is off, or under simulation_mode.
        """
        K = int(candidate_world_summaries.shape[0])
        device = candidate_world_summaries.device
        dtype = candidate_world_summaries.dtype

        if simulation_mode:
            self._last_n_simulation_skips += 1
            return torch.zeros(K, device=device, dtype=dtype)

        cfg = self.config
        self._n_waking_calls += 1

        total = torch.zeros(K, device=device, dtype=dtype)
        n_fired = 0

        # --------------------------------------------------------------
        # 314a striatal novelty: minimum distance to nearest ACTIVE
        # ResidueField RBF center, normalised by mean distance for
        # numerical stability.
        # --------------------------------------------------------------
        if cfg.use_curiosity_novelty:
            novelty = self._compute_novelty_phase2(
                candidate_world_summaries,
                residue_field=residue_field,
                visitation_source=visitation_source,
                first_action_onehots=first_action_onehots,
            )
            if novelty is not None:
                # Curiosity bonus is NEGATIVE in the lower-is-better
                # convention (high novelty -> more attractive ->
                # smaller score).
                total = total - cfg.curiosity_novelty_weight * novelty
                n_fired += 1
                self._last_novelty_norm = float(novelty.mean().item())
            else:
                self._last_novelty_norm = 0.0
        else:
            self._last_novelty_norm = 0.0
            self._last_novelty_source_used = "none"
            self._last_candidate_spread = 0.0
            self._last_augmentation_engaged = False

        # --------------------------------------------------------------
        # 314b frontopolar uncertainty: e3.running_variance scalar
        # broadcast across [K] (Phase 1 -- per-candidate refinement is
        # a Phase 2 follow-on requiring an E1 forward-variance head).
        # --------------------------------------------------------------
        if cfg.use_curiosity_uncertainty:
            unc = self._compute_uncertainty(e3)
            if unc is not None:
                total = total - cfg.curiosity_uncertainty_weight * unc * torch.ones(
                    K, device=device, dtype=dtype
                )
                n_fired += 1
                self._last_uncertainty_signal = float(unc)
            else:
                self._last_uncertainty_signal = 0.0
        else:
            self._last_uncertainty_signal = 0.0

        # --------------------------------------------------------------
        # 314c learning progress: EMA of |PE_t - PE_{t-K}| broadcast
        # across [K] (Phase 1 -- per-candidate refinement is a Phase 2
        # follow-on).
        # --------------------------------------------------------------
        if cfg.use_curiosity_learning_progress:
            lp = float(self._lp_ema)
            if lp != 0.0:
                total = total - cfg.curiosity_learning_progress_weight * lp * torch.ones(
                    K, device=device, dtype=dtype
                )
                n_fired += 1
            self._last_learning_progress_signal = lp
        else:
            self._last_learning_progress_signal = 0.0

        # Clamp to bias_scale so curiosity cannot dominate the existing
        # dACC / lateral_pfc / ofc / mech295 score-bias chain even at
        # extreme sub-signal magnitudes.
        total = torch.clamp(
            total, min=-cfg.curiosity_bias_scale, max=cfg.curiosity_bias_scale,
        )

        self._last_n_subflavours_fired = n_fired
        self._last_bias_max_abs = float(total.abs().max().item()) if K > 0 else 0.0
        return total

    # ------------------------------------------------------------------
    # 314a sub-flavour (per-candidate) -- MECH-314a Phase 2 orchestration
    # ------------------------------------------------------------------
    def _compute_novelty_phase2(
        self,
        candidate_world_summaries: torch.Tensor,
        residue_field: object,
        visitation_source: Any,
        first_action_onehots: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """MECH-314a Phase 2 (Candidate 5A) novelty orchestrator.

        Resolves the novelty comparison set (residue centers vs rolling
        z_world visitation buffer) per config.novelty_source, then applies
        the first-action one-hot augmentation per config.first_action_
        augmentation_policy. Delegates the actual RBF-distance arithmetic to
        _compute_novelty. With the default config (novelty_source="residue",
        use_first_action_onehot=False, first_action_augmentation_policy=
        "never") this is bit-identical to the Phase-1 single call
        self._compute_novelty(candidate_world_summaries, residue_field).
        """
        cfg = self.config

        # 1. Resolve the comparison source.
        src = cfg.novelty_source
        if src == "visitation":
            active_source = visitation_source
            source_used = "visitation"
        elif src == "auto":
            coerced = (
                self._coerce_source_to_centers(
                    visitation_source, candidate_world_summaries
                )
                if visitation_source is not None
                else None
            )
            if coerced is not None and coerced.shape[0] > 0:
                active_source = coerced
                source_used = "visitation"
            else:
                active_source = None
                source_used = "residue"
        else:  # "residue" (default)
            active_source = None
            source_used = "residue"
        self._last_novelty_source_used = source_used

        # Diagnostic: number of visitation entries in the comparison set.
        if isinstance(active_source, torch.Tensor):
            self._last_n_visitation_entries = int(active_source.shape[0])
        elif active_source is not None:
            try:
                self._last_n_visitation_entries = len(active_source)  # type: ignore[arg-type]
            except TypeError:
                self._last_n_visitation_entries = 0
        else:
            self._last_n_visitation_entries = 0

        # 2. Resolve augmentation engagement.
        policy = cfg.first_action_augmentation_policy
        can_augment = (
            cfg.use_first_action_onehot
            and first_action_onehots is not None
            and policy != "never"
        )
        if not can_augment:
            # Default / disabled path. Single un-augmented call. For "auto"
            # we still track the spread streak so an enabled-later toggle is
            # warm, but never augment here.
            engaged = False
            if policy == "auto":
                self._last_candidate_spread = self._candidate_spread(
                    candidate_world_summaries
                )
                self._update_augmentation_streak(self._last_candidate_spread)
            else:
                self._last_candidate_spread = 0.0
            self._last_augmentation_engaged = engaged
            return self._compute_novelty(
                candidate_world_summaries,
                residue_field,
                active_source=active_source,
                first_action_onehots=None,
            )

        # can_augment is True: policy is "always" or "auto".
        if policy == "always":
            self._last_candidate_spread = self._candidate_spread(
                candidate_world_summaries
            )
            engaged = True
        else:  # "auto"
            self._last_candidate_spread = self._candidate_spread(
                candidate_world_summaries
            )
            engaged = self._update_augmentation_streak(self._last_candidate_spread)
        self._last_augmentation_engaged = engaged
        if engaged:
            self._n_augmentation_engaged_ticks += 1
        return self._compute_novelty(
            candidate_world_summaries,
            residue_field,
            active_source=active_source,
            first_action_onehots=first_action_onehots if engaged else None,
        )

    def _candidate_spread(self, summaries: torch.Tensor) -> float:
        """Mean pairwise L2 distance across candidate z_world summaries.

        This is the per-candidate spread the auto-augmentation policy keys
        on (the same quantity as the SD-056 substrate-readiness diagnostic
        cand_world_pairwise_dist). Returns 0.0 for K < 2.
        """
        K = int(summaries.shape[0])
        if K < 2:
            return 0.0
        with torch.no_grad():
            diffs = summaries.unsqueeze(1) - summaries.unsqueeze(0)  # [K, K, D]
            dists = diffs.norm(dim=-1)  # [K, K]; diagonal is 0
            mean_pairwise = dists.sum() / float(K * (K - 1))
        return float(mean_pairwise.item())

    def _update_augmentation_streak(self, spread: float) -> bool:
        """Advance the auto-policy below-threshold streak and latch.

        Engages augmentation once spread stays below min_spread_threshold for
        min_spread_consecutive_ticks consecutive ticks; disengages
        immediately on a tick at-or-above the threshold (spread recovered).
        Returns the current engaged latch state.
        """
        cfg = self.config
        if spread < cfg.min_spread_threshold:
            self._below_threshold_streak += 1
        else:
            self._below_threshold_streak = 0
            self._augmentation_engaged = False
        if self._below_threshold_streak >= cfg.min_spread_consecutive_ticks:
            self._augmentation_engaged = True
        return self._augmentation_engaged

    def _coerce_source_to_centers(
        self, source: Any, like: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Coerce a visitation source to a [N, world_dim] centers tensor.

        Accepts a [N, D] tensor, a [D] tensor, or a deque / list / sequence
        of [D] (or [1, D]) tensors. Returns None when the source is empty or
        cannot be coerced. The result is detached and matched to like's
        dtype / device.
        """
        if source is None:
            return None
        if isinstance(source, torch.Tensor):
            t = source
        else:
            try:
                items = list(source)
            except TypeError:
                return None
            if len(items) == 0:
                return None
            try:
                t = torch.stack(
                    [
                        x.reshape(-1)
                        if isinstance(x, torch.Tensor)
                        else torch.as_tensor(x).reshape(-1)
                        for x in items
                    ],
                    dim=0,
                )
            except (RuntimeError, TypeError, ValueError):
                return None
        t = t.detach().to(dtype=like.dtype, device=like.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def _compute_novelty(
        self,
        candidate_world_summaries: torch.Tensor,
        residue_field: object,
        active_source: Any = None,
        first_action_onehots: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Per-candidate RBF-distance novelty score [K] in [0, +inf).

        Comparison set: when active_source is None (Phase-1 default), the
        active ResidueField RBF centers; otherwise active_source coerced to a
        [N, world_dim] tensor (the MECH-314a Phase 2 visitation buffer).
        Returns None when no comparison centers are available (314a
        contributes zero in that case).

        When first_action_onehots ([K, action_dim]) is provided (augmentation
        engaged), the candidate signatures are augmented with the one-hot and
        the comparison centers are zero-padded to the augmented dimension
        (architecture doc section 9: pad, do not project).

        With active_source=None and first_action_onehots=None this is
        bit-identical to the Phase-1 implementation.
        """
        # Resolve the comparison centers tensor.
        if active_source is not None:
            centers = self._coerce_source_to_centers(
                active_source, candidate_world_summaries
            )
            if centers is None or centers.shape[0] == 0:
                return None
        else:
            if residue_field is None:
                self._last_n_active_residue_centers = 0
                return None
            rbf = getattr(residue_field, "rbf_field", None)
            if rbf is None:
                self._last_n_active_residue_centers = 0
                return None
            active_mask = getattr(rbf, "active_mask", None)
            raw_centers = getattr(rbf, "centers", None)
            if active_mask is None or raw_centers is None:
                self._last_n_active_residue_centers = 0
                return None
            active_mask_bool = active_mask.bool()
            n_active = int(active_mask_bool.sum().item())
            self._last_n_active_residue_centers = n_active
            if n_active == 0:
                return None
            centers = raw_centers[active_mask_bool].detach()

        with torch.no_grad():
            sig = candidate_world_summaries
            cmp_centers = centers
            if first_action_onehots is not None:
                # Augment candidate signatures with the first-action one-hot;
                # zero-pad the comparison centers to the augmented dimension.
                oh = first_action_onehots.to(dtype=sig.dtype, device=sig.device)
                sig = torch.cat([sig, oh], dim=-1)
                pad = torch.zeros(
                    cmp_centers.shape[0],
                    oh.shape[-1],
                    dtype=cmp_centers.dtype,
                    device=cmp_centers.device,
                )
                cmp_centers = torch.cat([cmp_centers, pad], dim=-1)
            # [K, N] pairwise distances.
            diffs = sig.unsqueeze(1) - cmp_centers.unsqueeze(0)
            dists = diffs.norm(dim=-1)
            min_dists = dists.min(dim=-1).values  # [K]
            # Normalise by the candidate-pool mean norm to keep magnitudes
            # comparable across world_dim choices and to avoid swamping
            # the bias_scale clamp.
            mean_norm = sig.norm(dim=-1).mean().clamp(min=1e-6)
            novelty = min_dists / mean_norm
        return novelty

    # ------------------------------------------------------------------
    # 314b sub-flavour (broadcast scalar)
    # ------------------------------------------------------------------
    def _compute_uncertainty(self, e3: object) -> Optional[float]:
        """Global uncertainty signal: e3._running_variance.

        Phase 1 broadcast scalar; Phase 2 per-candidate refinement
        deferred. Returns None when e3 is None or running_variance is
        not exposed.
        """
        if e3 is None:
            return None
        rv = getattr(e3, "_running_variance", None)
        if rv is None:
            return None
        return float(rv)

    # ------------------------------------------------------------------
    # 314c sub-flavour (broadcast scalar; updated externally by agent)
    # ------------------------------------------------------------------
    def update_prediction_error(
        self, pe_scalar: float, simulation_mode: bool = False,
    ) -> None:
        """Feed the current waking tick's PE scalar into the 314c LP EMA.

        Called by REEAgent each tick after the e3.select(...) cycle.
        Maintains a ring buffer of the last (K+1) PE scalars; on each
        update with a full ring, computes |PE_t - PE_{t-K}| and folds
        it into the EMA at curiosity_lp_ema_alpha.

        Args:
            pe_scalar : float, e.g. e3._running_variance or PE-MSE per
                tick. The architectural contract is "any internal scalar
                that decreases as the agent's predictive model improves";
                Phase 1 callers pass running_variance.
            simulation_mode : MECH-094 gate. When True, no-op (the LP
                signal does not advance on simulation / replay paths).
        """
        if simulation_mode:
            self._last_n_simulation_skips += 1
            return
        K = int(self.config.curiosity_lp_window_k)
        # Maintain ring of last (K + 1) scalars.
        self._pe_ring.append(float(pe_scalar))
        if len(self._pe_ring) > K + 1:
            self._pe_ring = self._pe_ring[-(K + 1):]
        # Compute LP signal once we have enough history.
        if len(self._pe_ring) >= K + 1:
            lp_step = abs(self._pe_ring[-1] - self._pe_ring[0])
            alpha = float(self.config.curiosity_lp_ema_alpha)
            if not self._lp_seeded:
                self._lp_ema = lp_step
                self._lp_seeded = True
            else:
                self._lp_ema = (1.0 - alpha) * self._lp_ema + alpha * lp_step

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode diagnostic counters AND 314c LP buffer.

        The 314c learning-progress EMA is per-episode (a fresh task /
        environment can have a fresh learning curve). MECH-313 noise-
        floor is stateless across ticks; MECH-314 carries the LP buffer
        only. No persistent cross-tick state otherwise.
        """
        self._pe_ring = []
        self._lp_ema = 0.0
        self._lp_seeded = False
        self._last_n_simulation_skips = 0
        self._n_waking_calls = 0
        self._last_novelty_norm = 0.0
        self._last_uncertainty_signal = 0.0
        self._last_learning_progress_signal = 0.0
        self._last_n_active_residue_centers = 0
        self._last_n_subflavours_fired = 0
        self._last_bias_max_abs = 0.0
        # MECH-314a Phase 2 auto-augmentation state + diagnostics (per-episode).
        self._below_threshold_streak = 0
        self._augmentation_engaged = False
        self._last_novelty_source_used = "none"
        self._last_candidate_spread = 0.0
        self._last_augmentation_engaged = False
        self._last_n_visitation_entries = 0
        self._n_augmentation_engaged_ticks = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "n_waking_calls": self._n_waking_calls,
            "last_n_simulation_skips": self._last_n_simulation_skips,
            "last_novelty_norm": self._last_novelty_norm,
            "last_uncertainty_signal": self._last_uncertainty_signal,
            "last_learning_progress_signal": self._last_learning_progress_signal,
            "last_n_active_residue_centers": self._last_n_active_residue_centers,
            "last_n_subflavours_fired": self._last_n_subflavours_fired,
            "last_bias_max_abs": self._last_bias_max_abs,
            "lp_seeded": self._lp_seeded,
            "lp_ring_size": len(self._pe_ring),
            # MECH-314a Phase 2 diagnostics.
            "novelty_source_used": self._last_novelty_source_used,
            "last_candidate_spread": self._last_candidate_spread,
            "augmentation_engaged": self._last_augmentation_engaged,
            "below_threshold_streak": self._below_threshold_streak,
            "last_n_visitation_entries": self._last_n_visitation_entries,
            "n_augmentation_engaged_ticks": self._n_augmentation_engaged_ticks,
        }
