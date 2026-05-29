"""
MECH-341: E3 Score Diversity Preservation.

Layer-B (scoring) diversity-preservation substrate. Triggered by V3-EXQ-608 P2
majority R2a_e3_collapse_confirmed_large_gap finding (2026-05-26): CEM delivers
>=2 first-action classes (frac_pre_ge2=1.0) but E3 scoring collapses to a single
first-action class with mean_top2_class_gap 0.27-0.60 (LARGE-gap, structurally
rules out option 3 jittered tie-breaking near top).

Per behavioral_diversity_isolation_plan.md "Substrate design options for MECH-341"
section, two sub-flavours land together under one master, mirroring MECH-314a/b/c
precedent. Q-054 falsifier (subsequent /queue-experiment session) can dissociate
which one carries the load.

  Option 1: ENTROPY BONUS over candidate first-action classes.
    E3 score_bias chain receives a POSITIVE per-candidate addition proportional
    to the frequency of each candidate's first-action class in the pool. Strongly-
    represented classes get a homogenisation penalty. Composed BEFORE softmax;
    interacts naturally with the rest of the score-bias chain.

  Option 2: CLASS-STRATIFIED SELECTION.
    Partition candidates by first-action class, pick argmin (best score) within
    each class, then softmax-sample across class-representatives. Forces >= 2
    first-action classes to survive whenever the pool contains >= 2 classes.
    Replaces argmin in the committed selection path.

Both options bit-identical OFF by default. Pure-arithmetic regulator; no learned
parameters, no nn.Module inheritance. Sibling pattern to MECH-313 NoiseFloor,
MECH-314 StructuredCuriosity, MECH-320 TonicVigor.

Biological grounding:
- Rigotti et al. (2013) Nature 497:585 -- mixed selectivity in PFC encodes
  diverse trajectory contingencies; preservation across scoring layers is
  required for downstream behavioural flexibility.
- Padoa-Schioppa & Conen (2017) Neuron 96:736 -- OFC value comparison preserves
  option-distinct value signals through the comparison stage; collapse to single
  rank is pathological.

Option 1 maps to the soft-bias / entropy-pressure reading (Mnih 2016 A3C-style
entropy bonus on the candidate-class distribution). Option 2 maps to OFC
categorical preservation (Padoa-Schioppa explicit class-distinct retention).

MECH-094: both methods accept simulation_mode; when True, apply_entropy_bonus
returns zeros[K] and stratified_select falls through to legacy argmin. Inline
gates are defensive -- the call site (E3Selector.select) is currently invoked
only from waking REEAgent.select_action paths; no replay/DMN consumer exists.

See MECH-341 (this claim), ARC-065 (parent diversity pathway), Q-054 (entropy
floor calibration), INV-076 (diversity-as-counterfactual prerequisite),
behavioral_diversity_isolation_plan.md (isolation framework + R2.c rule).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class E3ScoreDiversityConfig:
    """Static config for MECH-341 substrate.

    Mirrors the flat REEConfig flags so callers can construct the module
    either from a REEConfig or from a standalone dataclass instance.
    """

    use_entropy_bonus: bool = True
    use_stratified_select: bool = True
    entropy_lambda: float = 0.5
    entropy_bias_scale: float = 1.0
    stratified_temperature: float = 1.0
    min_classes_for_stratification: int = 2


@dataclass
class E3ScoreDiversityDiagnostics:
    """Per-call diagnostics surfaced for the validation experiment."""

    last_n_candidates: int = 0
    last_n_unique_classes: int = 0
    last_entropy_bonus_max_abs: float = 0.0
    last_entropy_bonus_fired: bool = False
    last_stratified_fired: bool = False
    last_stratified_n_class_representatives: int = 0
    n_calls_total: int = 0
    n_entropy_bonus_fired: int = 0
    n_stratified_fired: int = 0
    n_simulation_skipped: int = 0


class E3ScoreDiversity:
    """MECH-341 substrate: two diversity-preservation sub-flavours under one master.

    Pure arithmetic. No learned parameters. No nn.Module inheritance.
    Sibling to MECH-313 NoiseFloor, MECH-314 StructuredCuriosity, MECH-320
    TonicVigor.

    Wiring: instantiated in E3Selector when REEConfig.use_e3_score_diversity is
    True; called from E3Selector.select() at two points:
        (1) apply_entropy_bonus(...) AFTER score_bias composition, BEFORE
            last_scores write and softmax.
        (2) stratified_select(...) replaces argmin in the committed selection
            path when both candidates and class-stratification preconditions
            hold; otherwise falls through to legacy argmin.
    """

    def __init__(self, config: Optional[E3ScoreDiversityConfig] = None) -> None:
        self.config = config if config is not None else E3ScoreDiversityConfig()
        self.diagnostics = E3ScoreDiversityDiagnostics()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _first_action_classes(candidates: List) -> List[int]:
        """Extract per-candidate first-action argmax class index.

        Mirrors the helper used by MECH-320 TonicVigor.compute_score_bias.
        Robust to candidates whose `actions` field has shape [batch, horizon, A]
        or [batch, A].
        """
        classes: List[int] = []
        for c in candidates:
            actions = c.actions
            if actions.dim() >= 3:
                first_step = actions[:, 0, :]
            else:
                first_step = actions
            first_step = first_step[0] if first_step.dim() >= 2 else first_step
            classes.append(int(first_step.argmax().item()))
        return classes

    # ------------------------------------------------------------------ #
    # Option 1: entropy bonus
    # ------------------------------------------------------------------ #

    def apply_entropy_bonus(
        self,
        scores: Tensor,
        candidates: List,
        simulation_mode: bool = False,
    ) -> Tensor:
        """Return per-candidate POSITIVE bias on over-represented first-action classes.

        Algorithm:
            1. Extract first-action class per candidate.
            2. Compute per-class frequency p_c = count(c) / K.
            3. Per-candidate bonus = lambda * p_c   (E3 lower-is-better, so a
               POSITIVE bias on over-represented classes makes them less
               attractive -- penalises pool homogenisation at scoring).
            4. Clamp to [-bias_scale, +bias_scale].

        Returns a per-candidate [K] tensor on the same device/dtype as `scores`.
        Returns zeros[K] when (a) master sub-flavour is off, (b) simulation_mode
        is True (MECH-094), (c) K < 2, or (d) all candidates share one class
        (no homogenisation gradient available).
        """
        self.diagnostics.n_calls_total += 1

        K = scores.shape[0]
        self.diagnostics.last_n_candidates = K
        self.diagnostics.last_entropy_bonus_fired = False
        self.diagnostics.last_entropy_bonus_max_abs = 0.0

        if not self.config.use_entropy_bonus:
            return scores.new_zeros(K)

        if simulation_mode:
            self.diagnostics.n_simulation_skipped += 1
            return scores.new_zeros(K)

        if K < 2:
            self.diagnostics.last_n_unique_classes = K
            return scores.new_zeros(K)

        classes = self._first_action_classes(candidates)
        unique_classes = sorted(set(classes))
        self.diagnostics.last_n_unique_classes = len(unique_classes)

        if len(unique_classes) <= 1:
            return scores.new_zeros(K)

        counts = {c: 0 for c in unique_classes}
        for c in classes:
            counts[c] += 1
        K_float = float(K)

        bonus_list = [
            self.config.entropy_lambda * (counts[c] / K_float)
            for c in classes
        ]
        bonus = torch.tensor(
            bonus_list, dtype=scores.dtype, device=scores.device
        )
        bonus = bonus.clamp(
            min=-self.config.entropy_bias_scale,
            max=self.config.entropy_bias_scale,
        )

        max_abs = float(bonus.abs().max().item())
        if max_abs > 0.0:
            self.diagnostics.last_entropy_bonus_fired = True
            self.diagnostics.n_entropy_bonus_fired += 1
            self.diagnostics.last_entropy_bonus_max_abs = max_abs

        return bonus

    # ------------------------------------------------------------------ #
    # Option 2: class-stratified selection
    # ------------------------------------------------------------------ #

    def stratified_select(
        self,
        scores: Tensor,
        candidates: List,
        simulation_mode: bool = False,
    ) -> Optional[int]:
        """Stratified sampling across first-action class representatives.

        Algorithm:
            1. Partition candidates by first-action class.
            2. Per class, pick the argmin-score candidate as class representative.
            3. Sample across class representatives with probability
               softmax(-best_in_class_score / stratified_temperature).
            4. Return the global candidate index of the sampled representative.

        Returns None when (a) master sub-flavour is off, (b) simulation_mode is
        True, or (c) the number of unique classes is below
        min_classes_for_stratification. Caller falls through to legacy argmin in
        those cases.
        """
        if not self.config.use_stratified_select:
            return None

        if simulation_mode:
            self.diagnostics.n_simulation_skipped += 1
            return None

        K = scores.shape[0]
        if K < self.config.min_classes_for_stratification:
            return None

        classes = self._first_action_classes(candidates)
        unique_classes = sorted(set(classes))
        if len(unique_classes) < self.config.min_classes_for_stratification:
            return None

        scores_detached = scores.detach()
        best_idx_per_class: List[Tuple[int, float]] = []
        for cls in unique_classes:
            class_idxs = [i for i, c in enumerate(classes) if c == cls]
            class_scores = scores_detached[class_idxs]
            local_best = int(class_scores.argmin().item())
            global_idx = class_idxs[local_best]
            best_idx_per_class.append(
                (global_idx, float(class_scores[local_best].item()))
            )

        rep_indices = [t[0] for t in best_idx_per_class]
        rep_scores = torch.tensor(
            [t[1] for t in best_idx_per_class],
            dtype=scores.dtype,
            device=scores.device,
        )

        temp = max(float(self.config.stratified_temperature), 1e-6)
        probs = torch.softmax(-rep_scores / temp, dim=0)
        sampled_rep = int(torch.multinomial(probs, 1).item())
        selected_idx = rep_indices[sampled_rep]

        self.diagnostics.last_stratified_fired = True
        self.diagnostics.n_stratified_fired += 1
        self.diagnostics.last_stratified_n_class_representatives = len(rep_indices)

        return selected_idx

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Per-episode reset of diagnostic counters.

        Module has no persistent state across ticks; only diagnostics are cleared.
        """
        self.diagnostics = E3ScoreDiversityDiagnostics()

    def get_state(self) -> dict:
        d = self.diagnostics
        return {
            "mech341_last_n_candidates": d.last_n_candidates,
            "mech341_last_n_unique_classes": d.last_n_unique_classes,
            "mech341_last_entropy_bonus_fired": d.last_entropy_bonus_fired,
            "mech341_last_entropy_bonus_max_abs": d.last_entropy_bonus_max_abs,
            "mech341_last_stratified_fired": d.last_stratified_fired,
            "mech341_last_stratified_n_class_representatives": (
                d.last_stratified_n_class_representatives
            ),
            "mech341_n_calls_total": d.n_calls_total,
            "mech341_n_entropy_bonus_fired": d.n_entropy_bonus_fired,
            "mech341_n_stratified_fired": d.n_stratified_fired,
            "mech341_n_simulation_skipped": d.n_simulation_skipped,
        }


def build_from_ree_config(ree_config) -> Optional[E3ScoreDiversity]:
    """Construct E3ScoreDiversity from REEConfig flat flags or return None.

    Returns None when use_e3_score_diversity is False. Caller branch protects
    callers that hold ree_config without explicit defaults.
    """
    if not getattr(ree_config, "use_e3_score_diversity", False):
        return None
    cfg = E3ScoreDiversityConfig(
        use_entropy_bonus=getattr(
            ree_config, "use_e3_diversity_entropy_bonus", True
        ),
        use_stratified_select=getattr(
            ree_config, "use_e3_diversity_stratified_select", True
        ),
        entropy_lambda=getattr(ree_config, "e3_diversity_entropy_lambda", 0.5),
        entropy_bias_scale=getattr(
            ree_config, "e3_diversity_entropy_bias_scale", 1.0
        ),
        stratified_temperature=getattr(
            ree_config, "e3_diversity_stratified_temperature", 1.0
        ),
        min_classes_for_stratification=getattr(
            ree_config, "e3_diversity_min_classes_for_stratification", 2
        ),
    )
    return E3ScoreDiversity(cfg)
