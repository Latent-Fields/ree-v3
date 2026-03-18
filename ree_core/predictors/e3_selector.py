"""
E3 Trajectory Selector — V3 Implementation

V3 extensions:
1. harm_eval(z_world) — new method; E3 is the correct locus for harm prediction,
   NOT E2. Used in SD-003 V3 attribution pipeline:
     harm_actual = e3.harm_eval(z_world_actual)
     harm_cf     = e3.harm_eval(z_world_cf)
     causal_sig  = harm_actual - harm_cf

2. Dynamic precision (ARC-016):
   Precision is derived from E3's own prediction error variance (EMA of MSE),
   NOT hardcoded. This is required to test ARC-016 (dynamic precision
   behavioural distinction).
   committed = running_variance < commit_threshold  (variance-space, fixed 2026-03-18)

3. Operates over z_world (SD-005):
   E3's input domain is z_world (exteroceptive world model), not z_gamma.

E3 trains on harm + goal error over z_world.

Scoring equation J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ) remains a working
hypothesis — see ARCHITECTURE NOTE below. All weights are placeholder.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E3Config
from ree_core.predictors.e2_fast import Trajectory
from ree_core.residue.field import ResidueField


@dataclass
class SelectionResult:
    """Result of trajectory selection.

    selected_trajectory: chosen trajectory
    selected_index:      index within candidates
    selected_action:     first action from selected trajectory
    scores:              all trajectory scores [num_candidates]
    precision:           current dynamic precision value
    committed:           whether commitment threshold was met
    log_prob:            log-probability for REINFORCE (connected to computation graph)
    """
    selected_trajectory: Trajectory
    selected_index: int
    selected_action: torch.Tensor
    scores: torch.Tensor
    precision: float
    committed: bool
    log_prob: Optional[torch.Tensor] = None


def variance_commit_threshold(config_threshold: float) -> float:
    """
    Return the variance-space commit threshold (ARC-016).

    Commitment fires when running_variance < threshold (low variance = confident).
    Threshold lives in variance space [~0.01–0.05], NOT precision space (~100).
    (Prior precision_to_threshold() was on the wrong scale: returning ~0.703
    vs current_precision ~95 → always committed. Fixed 2026-03-18.)
    """
    return config_threshold


class E3TrajectorySelector(nn.Module):
    """
    E3 Trajectory Selector — V3.

    Evaluates candidate trajectories and selects one by minimising:
        J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)   [working hypothesis]

    V3 additions:
    - harm_eval(z_world) for SD-003 attribution
    - Dynamic precision via prediction error variance (ARC-016)
    - All scoring operates over z_world (SD-005)

    # ARCHITECTURE NOTE (V3)
    # -----------------------------------------------------------------
    # The scoring equation is a WORKING HYPOTHESIS. The weights
    # lambda_ethical and rho_residue are placeholder parameters.
    # The scoring function as a whole requires calibration experiments.
    #
    # In V3:
    # - F(ζ): proxy (smoothness + final-state viability). Not settled.
    # - M(ζ): uses harm_eval() on z_world trajectory states. This is
    #   categorically distinct from V2's E2.predict_harm (which belonged
    #   to E2, not E3). harm_eval() is the correct locus per spec §5.4.
    # - Φ_R(ζ): residue field now operates over z_world (SD-005).
    #   Self-change (z_self_delta) does not contribute to residue.
    # -----------------------------------------------------------------
    """

    def __init__(
        self,
        config: Optional[E3Config] = None,
        residue_field: Optional[ResidueField] = None,
    ):
        super().__init__()
        self.config = config or E3Config()
        self.residue_field = residue_field

        world_dim = self.config.world_dim

        # Reality constraint scorer F(ζ): operates on z_world
        self.reality_scorer = nn.Sequential(
            nn.Linear(world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

        # Ethical cost scorer M(ζ): operates on z_world
        self.ethical_scorer = nn.Sequential(
            nn.Linear(world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

        # harm_eval head (SD-003 V3 attribution):
        # Predicts harm of a z_world state. This belongs to E3, NOT E2.
        self.harm_eval_head = nn.Sequential(
            nn.Linear(world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid(),  # harm in [0, 1]
        )

        # Dynamic precision state (ARC-016)
        # Maintained as EMA of prediction error MSE across committed trajectories.
        self._running_variance: float = self.config.precision_init
        self._ema_alpha: float = self.config.precision_ema_alpha

        # Commitment state
        self._committed_trajectory: Optional[Trajectory] = None
        self.last_scores: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Dynamic precision (ARC-016)                                         #
    # ------------------------------------------------------------------ #

    @property
    def current_precision(self) -> float:
        """Current precision estimate (inverse of running variance)."""
        return 1.0 / (self._running_variance + 1e-6)

    @property
    def commit_threshold(self) -> float:
        """Variance-space commit threshold (ARC-016). Committed when variance < threshold."""
        return variance_commit_threshold(self.config.commitment_threshold)

    def update_running_variance(self, prediction_error: torch.Tensor) -> None:
        """Update EMA of prediction error variance (ARC-016 dynamic precision)."""
        error_var = prediction_error.pow(2).mean().item()
        self._running_variance = (
            (1 - self._ema_alpha) * self._running_variance
            + self._ema_alpha * error_var
        )

    # ------------------------------------------------------------------ #
    # harm_eval (SD-003 V3 pipeline)                                      #
    # ------------------------------------------------------------------ #

    def harm_eval(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Evaluate harm of a world-state (SD-003 V3 attribution pipeline).

        This is the correct locus for harm prediction — NOT E2.
        Used externally as:
            harm_actual = e3.harm_eval(z_world_actual)
            harm_cf     = e3.harm_eval(z_world_cf)
            causal_sig  = harm_actual - harm_cf

        Args:
            z_world: [batch, world_dim]

        Returns:
            harm estimate [batch, 1] in [0, 1]
        """
        return self.harm_eval_head(z_world)

    def harm_eval_lateral(
        self,
        z_harm: torch.Tensor,
        lateral_head: nn.Module,
    ) -> torch.Tensor:
        """
        Evaluate harm using the lateral encoder head's z_harm embedding (MECH-099).

        The lateral head produces a harm-salient embedding directly from
        hazard + contamination channels, bypassing E2_world's identity shortcut.
        This method applies a small projection from z_harm to a scalar harm estimate.

        Args:
            z_harm: [batch, harm_dim] — output of SplitEncoder.lateral_head
            lateral_head: nn.Module that maps z_harm [harm_dim] → harm scalar [1]

        Returns:
            harm estimate [batch, 1]
        """
        return lateral_head(z_harm)

    # ------------------------------------------------------------------ #
    # Trajectory scoring                                                   #
    # ------------------------------------------------------------------ #

    def _get_world_states(self, trajectory: Trajectory) -> torch.Tensor:
        """Extract z_world trajectory. Falls back to z_self if no world_states."""
        if trajectory.world_states is not None:
            ws = trajectory.get_world_state_sequence()
            if ws is not None:
                return ws
        # Fallback: use z_self states (for V3-EXQ-001 before full SD-005 wiring)
        return trajectory.get_state_sequence()

    def compute_reality_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """Reality constraint F(ζ) — proxy: smoothness + viability of z_world final state."""
        world_seq = self._get_world_states(trajectory)  # [batch, horizon, world_dim]

        if world_seq.shape[1] > 1:
            transitions = world_seq[:, 1:, :] - world_seq[:, :-1, :]
            coherence_cost = transitions.pow(2).sum(dim=-1).mean(dim=-1)
        else:
            coherence_cost = torch.zeros(world_seq.shape[0], device=world_seq.device)

        final_world = world_seq[:, -1, :]  # [batch, world_dim]
        viability = self.reality_scorer(final_world).squeeze(-1)
        return coherence_cost - viability

    def compute_ethical_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Ethical cost M(ζ) via harm_eval over z_world trajectory.

        V3 change: uses harm_eval_head on z_world states rather than
        E2.harm_predictions (which belonged to E2 in V2, incorrectly).
        """
        world_seq = self._get_world_states(trajectory)  # [batch, horizon+1, world_dim]
        batch, horizon_p1, _ = world_seq.shape

        # Evaluate harm at each step
        flat = world_seq.reshape(batch * horizon_p1, -1)
        harm_flat = self.harm_eval_head(flat)              # [batch*horizon, 1]
        harm = harm_flat.reshape(batch, horizon_p1)        # [batch, horizon+1]
        harm_cost = harm.sum(dim=-1)                       # [batch]

        # Additional scoring from final z_world state
        final_world = world_seq[:, -1, :]
        ethical_score = self.ethical_scorer(final_world).squeeze(-1)

        return harm_cost - ethical_score

    def compute_residue_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """Residue field cost Φ_R(ζ) — evaluated over z_world (SD-005)."""
        if self.residue_field is None:
            return torch.zeros(
                trajectory.states[0].shape[0],
                device=trajectory.states[0].device,
            )
        world_seq = self._get_world_states(trajectory)   # [batch, horizon+1, world_dim]
        return self.residue_field.evaluate_trajectory(world_seq)

    def score_trajectory(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Total score J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ). Lower is better.
        """
        f = self.compute_reality_cost(trajectory)
        m = self.compute_ethical_cost(trajectory)
        phi = self.compute_residue_cost(trajectory)
        return f + self.config.lambda_ethical * m + self.config.rho_residue * phi

    # ------------------------------------------------------------------ #
    # Selection                                                            #
    # ------------------------------------------------------------------ #

    def select(
        self,
        candidates: List[Trajectory],
        temperature: float = 1.0,
    ) -> SelectionResult:
        """
        Select the best trajectory from candidates.

        Uses dynamic precision (ARC-016) to determine commit threshold.

        Args:
            candidates:  list of Trajectory objects
            temperature: softmax temperature (exploration vs exploitation)

        Returns:
            SelectionResult
        """
        if not candidates:
            raise ValueError("No candidate trajectories provided")

        scores = torch.stack([self.score_trajectory(t) for t in candidates])
        scores = scores.mean(dim=-1)
        self.last_scores = scores.detach()

        probs = F.softmax(-scores / temperature, dim=0)

        # Dynamic commit threshold (ARC-016): commit when variance is LOW
        # (confident predictions → greedy selection)
        committed = self._running_variance < self.commit_threshold
        if committed:
            selected_idx = int(scores.argmin().item())
        else:
            selected_idx = int(torch.multinomial(probs, 1).item())

        selected_trajectory = candidates[selected_idx]
        selected_action = selected_trajectory.actions[:, 0, :]

        log_probs = F.log_softmax(-scores / temperature, dim=0)
        log_prob = log_probs[selected_idx]

        if committed:
            self._committed_trajectory = selected_trajectory

        return SelectionResult(
            selected_trajectory=selected_trajectory,
            selected_index=selected_idx,
            selected_action=selected_action,
            scores=scores,
            precision=self.current_precision,
            committed=committed,
            log_prob=log_prob,
        )

    # ------------------------------------------------------------------ #
    # Post-action update                                                   #
    # ------------------------------------------------------------------ #

    def post_action_update(
        self,
        actual_z_world: torch.Tensor,
        harm_occurred: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Update E3 after action execution (ARC-016 dynamic precision update).

        Args:
            actual_z_world: Observed z_world after action [batch, world_dim]
            harm_occurred:  Whether harm occurred (for residue gating)

        Returns:
            Dictionary of update metrics
        """
        metrics: Dict[str, torch.Tensor] = {}

        if self._committed_trajectory is not None and self._committed_trajectory.world_states is not None:
            predicted_world = self._committed_trajectory.world_states[1]
            prediction_error = actual_z_world - predicted_world

            # ARC-016: update running variance → dynamic precision
            self.update_running_variance(prediction_error)

            metrics["prediction_error"] = prediction_error.pow(2).mean()
            metrics["running_variance"] = torch.tensor(self._running_variance)
            metrics["dynamic_precision"] = torch.tensor(self.current_precision)

            if harm_occurred and self.residue_field is not None:
                # Residue accumulates on z_world (SD-005), not z_gamma
                self.residue_field.accumulate(
                    actual_z_world, harm_magnitude=1.0, hypothesis_tag=False
                )
                metrics["residue_updated"] = torch.tensor(1.0)

        self._committed_trajectory = None
        return metrics

    def get_commitment_state(self) -> Dict[str, float]:
        return {
            "precision": self.current_precision,
            "running_variance": self._running_variance,
            "commit_threshold": self.commit_threshold,
            "committed_now": self._running_variance < self.commit_threshold,
            "is_committed": self._committed_trajectory is not None,
        }

    def forward(
        self,
        candidates: List[Trajectory],
        temperature: float = 1.0,
    ) -> SelectionResult:
        return self.select(candidates, temperature)
