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
from ree_core.goal import GoalState


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
    urgency: float = 0.0  # SD-011: z_harm_a urgency applied to commit threshold


def variance_commit_threshold(config_threshold: float) -> float:
    """
    Return the variance-space commit threshold (ARC-016).

    Commitment fires when running_variance < threshold (low variance = confident).
    Threshold lives in variance space [~0.01-0.05], NOT precision space (~100).
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

        # ARC-030 / MECH-112: benefit_eval head — Go channel (symmetric to harm_eval NoGo).
        # Evaluates resource/goal proximity from z_world states.
        # Instantiated unconditionally (adds ~4K params); only receives gradients
        # in experiments that call benefit_eval() or enable benefit_eval_enabled.
        # Biological basis: D1 (Go) pathway evaluates same proposals as D2 (NoGo).
        self.benefit_eval_head = nn.Sequential(
            nn.Linear(world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid(),  # benefit in [0, 1]
        )

        # MECH-111: novelty EMA — tracks recent E1 prediction error per trajectory.
        # Stored as a scalar EMA; decays toward zero between updates.
        self._novelty_ema: float = 0.0
        self._novelty_ema_alpha: float = 0.1

        # SD-010: harm_eval head operating on z_harm (dedicated nociceptive stream).
        # z_harm comes from HarmEncoder(harm_obs), NOT from z_world.
        # This head is always instantiated (adds ~4K params); it only receives
        # gradients in SD-010 experiments that explicitly call harm_eval_z_harm().
        # Fallback z_harm_dim = world_dim if config does not carry z_harm_dim
        # (all existing E3Config instances, which lack the field).
        _z_harm_dim = getattr(self.config, "z_harm_dim", world_dim)
        self.harm_eval_z_harm_head = nn.Sequential(
            nn.Linear(_z_harm_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            # Sigmoid removed (2026-03-20): this is a linear regression head for hazard
            # proximity, trained with normalized labels in [0,1] (harm_obs[12]).
            # MSE loss with [0,1] labels constrains outputs naturally without Sigmoid.
            # Sigmoid was causing saturation when raw hazard_field_at_agent labels (>1)
            # were used — root cause of SD-010 collapse in EXQ-056 original.
        )

        # Dynamic precision state (ARC-016)
        # Maintained as EMA of prediction error MSE across committed trajectories.
        self._running_variance: float = self.config.precision_init
        self._ema_alpha: float = self.config.precision_ema_alpha

        # Q-007: volatility estimate = var(rv) over a sliding window.
        # Raw rv tracks moment-to-moment prediction error (fast EMA, half-life
        # ~14 steps).  In both stable and volatile environments rv converges to
        # near-zero once E2 adapts, losing the between-condition signal.
        # var(rv) captures how much rv *fluctuates*: stable env -> rv flat ->
        # var(rv) ~ 0; volatile env -> rv spikes on hazard moves -> var(rv) high.
        # This is the LC-NE tonic firing analog (Yu & Dayan 2005).
        from collections import deque
        self._rv_history: deque = deque(maxlen=100)
        self._volatility_estimate: float = 0.0

        # Commitment state
        self._committed_trajectory: Optional[Trajectory] = None
        # ARC-016: store last selected trajectory for rv updates regardless of
        # commitment.  Without this, rv only updates when committed, creating a
        # deadlock: rv starts above commit_threshold -> agent never commits ->
        # rv never updates -> agent can never commit.
        self._last_selected_trajectory: Optional[Trajectory] = None
        self.last_scores: Optional[torch.Tensor] = None

        # ARC-030: benefit_eval warmup gate.
        # benefit_eval_head starts at random init — scoring with it before training
        # converges adds harmful noise to trajectory selection. Gate is lifted once
        # enough benefit samples have been seen (call record_benefit_sample() from
        # the experiment training loop when adding to the benefit buffer).
        self._benefit_samples_seen: int = 0
        self._BENEFIT_WARMUP_SAMPLES: int = 50

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

    @property
    def volatility_estimate(self) -> float:
        """Q-007: var(rv) over sliding window -- LC-NE tonic volatility signal."""
        return self._volatility_estimate

    def update_running_variance(self, prediction_error: torch.Tensor) -> None:
        """Update EMA of prediction error variance (ARC-016 dynamic precision)."""
        error_var = prediction_error.pow(2).mean().item()
        self._running_variance = (
            (1 - self._ema_alpha) * self._running_variance
            + self._ema_alpha * error_var
        )
        # Q-007: track rv history and compute volatility estimate
        self._rv_history.append(self._running_variance)
        if len(self._rv_history) >= 10:
            vals = list(self._rv_history)
            mean = sum(vals) / len(vals)
            self._volatility_estimate = sum((v - mean) ** 2 for v in vals) / len(vals)
        else:
            self._volatility_estimate = 0.0

    def record_benefit_sample(self, n: int = 1) -> None:
        """Record that n benefit training samples have been added to the buffer.

        Call from the experiment training loop each time a positive benefit
        sample is added. Once _benefit_samples_seen >= _BENEFIT_WARMUP_SAMPLES,
        benefit scoring is activated in score_trajectory().
        """
        self._benefit_samples_seen += n

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

    def benefit_eval(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Evaluate benefit/goal proximity from a world-state (ARC-030 Go channel).

        Symmetric to harm_eval() — both evaluate the same z_world trajectory
        proposals. This is the D1 (Go) pathway counterpart to harm_eval()'s
        D2 (NoGo) role (Bariselli 2018). Commit threshold is the balance point.

        Args:
            z_world: [batch, world_dim]

        Returns:
            benefit estimate [batch, 1] in [0, 1]
        """
        return self.benefit_eval_head(z_world)

    def update_novelty_ema(self, e1_prediction_error_mse: float) -> None:
        """
        MECH-111: Update novelty EMA from E1 prediction error.

        Higher E1 error = more novel state. Used in score_trajectory()
        when novelty_bonus_weight > 0 to reward unexplored regions.
        """
        self._novelty_ema = (
            (1 - self._novelty_ema_alpha) * self._novelty_ema
            + self._novelty_ema_alpha * e1_prediction_error_mse
        )

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

    def harm_eval_z_harm(self, z_harm: torch.Tensor) -> torch.Tensor:
        """
        SD-010: Evaluate harm using the dedicated nociceptive stream latent.

        z_harm is the output of HarmEncoder(harm_obs), NOT of the z_world encoder.
        Because HarmEncoder is instantiated outside LatentStack.encode(), z_harm
        is never subject to reafference correction — which resolves the EXQ-027b
        over-correction paradox (ReafferencePredictor was subtracting hazard signal
        when it was fused into z_world).

        Used in SD-003 attribution with SD-010:
            z_harm_actual = harm_enc(harm_obs_actual)
            z_harm_cf     = harm_enc(harm_bridge(E2.world_forward(z_world, a_cf)))
            causal_sig    = harm_eval_z_harm(z_harm_actual) - harm_eval_z_harm(z_harm_cf)

        Args:
            z_harm: [batch, z_harm_dim] — output of HarmEncoder

        Returns:
            harm estimate [batch, 1] in [0, 1]
        """
        return self.harm_eval_z_harm_head(z_harm)

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

    def compute_benefit_score(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Go channel benefit score B(ζ) — ARC-030 / MECH-112.

        Evaluates resource/goal proximity across z_world trajectory.
        Returns summed benefit signal (higher = more beneficial trajectory).
        Subtracted from total score (lower J = better, so benefit reduces J).
        """
        world_seq = self._get_world_states(trajectory)  # [batch, horizon+1, world_dim]
        batch, horizon_p1, _ = world_seq.shape
        flat = world_seq.reshape(batch * horizon_p1, -1)
        benefit_flat = self.benefit_eval_head(flat)         # [batch*horizon, 1]
        benefit = benefit_flat.reshape(batch, horizon_p1)   # [batch, horizon+1]
        return benefit.sum(dim=-1)                          # [batch]

    def compute_goal_score(
        self, trajectory: Trajectory, goal_state: GoalState
    ) -> torch.Tensor:
        """
        Goal proximity score across trajectory (wanting signal, MECH-112/117).
        Shape: [batch]. Higher = trajectory ends closer to z_goal.
        """
        world_seq = self._get_world_states(trajectory)
        batch, horizon_p1, _ = world_seq.shape
        flat = world_seq.reshape(batch * horizon_p1, -1)
        prox_flat = goal_state.goal_proximity(flat)
        prox = prox_flat.reshape(batch, horizon_p1)
        return prox.sum(dim=-1)

    def compute_harm_stream_cost(
        self,
        trajectory: Trajectory,
        harm_bridge: "nn.Module",
    ) -> torch.Tensor:
        """
        SD-010: Ethical cost M(ζ) via harm stream (z_harm) rather than z_world.

        Applies HarmBridge to each z_world state in the trajectory to get z_harm
        approximations, then evaluates via harm_eval_z_harm_head. This is the
        SD-010 replacement for compute_ethical_cost() when the nociceptive stream
        is available.

        Used in select() when harm_bridge is provided.

        Args:
            trajectory:  candidate trajectory with world_states
            harm_bridge: HarmBridge instance (z_world -> z_harm)

        Returns:
            harm_cost: [batch] — summed harm cost over trajectory horizon
        """
        world_seq = self._get_world_states(trajectory)  # [batch, horizon+1, world_dim]
        batch, horizon_p1, world_dim = world_seq.shape

        flat = world_seq.reshape(batch * horizon_p1, world_dim)
        z_harm_flat = harm_bridge(flat)                              # [batch*horizon, z_harm_dim]
        harm_flat = self.harm_eval_z_harm_head(z_harm_flat)         # [batch*horizon, 1]
        harm = harm_flat.reshape(batch, horizon_p1)                  # [batch, horizon+1]
        return harm.sum(dim=-1)                                      # [batch]

    def compute_harm_forward_cost(
        self,
        trajectory: Trajectory,
        harm_forward_model: "nn.Module",
        z_harm_s_current: torch.Tensor,
    ) -> torch.Tensor:
        """
        SD-011/ARC-033: Ethical cost M(zeta) via ResidualHarmForward rollout.

        Replaces compute_harm_stream_cost (HarmBridge-based, deprecated).
        Rolls out z_harm_s step-by-step through the trajectory actions using
        the residual harm forward model, then evaluates each predicted state
        via harm_eval_z_harm_head.

        Args:
            trajectory:          candidate trajectory with actions
            harm_forward_model:  ResidualHarmForward instance
            z_harm_s_current:    [batch, z_harm_dim] current sensory-discriminative
                                 harm latent

        Returns:
            harm_cost: [batch] -- summed harm cost over trajectory horizon
        """
        actions = trajectory.actions  # [batch, horizon, action_dim]
        batch, horizon, _ = actions.shape

        z_harm_step = z_harm_s_current  # [batch, z_harm_dim]
        harm_total = torch.zeros(batch, device=z_harm_step.device)

        for t in range(horizon):
            a_t = actions[:, t, :]  # [batch, action_dim]
            z_harm_step = harm_forward_model(z_harm_step, a_t)  # [batch, z_harm_dim]
            harm_t = self.harm_eval_z_harm_head(z_harm_step)    # [batch, 1]
            harm_total = harm_total + harm_t.squeeze(-1)

        return harm_total

    def score_trajectory(
        self,
        trajectory: Trajectory,
        goal_state: Optional[GoalState] = None,
        harm_bridge: Optional["nn.Module"] = None,
        terrain_weight: Optional[torch.Tensor] = None,
        harm_forward_model: Optional["nn.Module"] = None,
        z_harm_s_current: Optional[torch.Tensor] = None,
        z_harm_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Total score J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ) - β·B(ζ) - η·novelty.
        Lower is better.

        - F(ζ): reality cost (smoothness + viability)
        - M(ζ): ethical cost via harm_eval (NoGo channel, D2 pathway)
        - Φ_R(ζ): residue field cost
        - B(ζ): benefit score (Go channel, D1 pathway) — subtracted when enabled
        - novelty: E1 error EMA bonus — subtracted when novelty_bonus_weight > 0

        M(ζ) evaluation priority:
        1. harm_forward_model + z_harm_s_current (SD-011/ARC-033, preferred)
        2. harm_bridge (SD-010, deprecated but backward compat)
        3. harm_eval_head on z_world (default)

        SD-011: z_harm_a amplifies lambda_ethical when affective_harm_scale > 0.
        SD-016 (MECH-152): optional terrain_weight scales M/B after evaluation.
        """
        f = self.compute_reality_cost(trajectory)
        if harm_forward_model is not None and z_harm_s_current is not None:
            m = self.compute_harm_forward_cost(
                trajectory, harm_forward_model, z_harm_s_current
            )
        elif harm_bridge is not None:
            m = self.compute_harm_stream_cost(trajectory, harm_bridge)
        else:
            m = self.compute_ethical_cost(trajectory)
        phi = self.compute_residue_cost(trajectory)

        # SD-011: z_harm_a amplification of ethical cost.
        # When accumulated threat is high, harm costs weigh more.
        lambda_eff = self.config.lambda_ethical
        if z_harm_a is not None and self.config.affective_harm_scale > 0.0:
            z_harm_a_norm = z_harm_a.norm(dim=-1).mean().item()
            lambda_eff = lambda_eff * (1.0 + self.config.affective_harm_scale * z_harm_a_norm)

        # SD-016 (MECH-152): scale harm cost by w_harm from terrain_weight
        if terrain_weight is not None:
            w_harm = terrain_weight[:, 0]  # [batch]
            m = m * w_harm

        score = f + lambda_eff * m + self.config.rho_residue * phi

        # ARC-030 / MECH-112: Go channel — subtract benefit from cost.
        # Gated until _benefit_samples_seen >= _BENEFIT_WARMUP_SAMPLES to prevent
        # random-init noise from corrupting trajectory selection early in training.
        if (self.config.benefit_eval_enabled
                and self.config.benefit_weight > 0.0
                and self._benefit_samples_seen >= self._BENEFIT_WARMUP_SAMPLES):
            b = self.compute_benefit_score(trajectory)
            # SD-016 (MECH-152): scale benefit by w_goal from terrain_weight
            if terrain_weight is not None:
                w_goal = terrain_weight[:, 1]  # [batch]
                b = b * w_goal
            score = score - self.config.benefit_weight * b

        # MECH-111: novelty bonus — subtract EMA novelty signal
        if self.config.novelty_bonus_weight > 0.0:
            # Scalar EMA novelty applies uniformly across trajectory batch dim
            device = score.device
            novelty_bonus = torch.tensor(self._novelty_ema, device=device)
            score = score - self.config.novelty_bonus_weight * novelty_bonus

        # MECH-112 / MECH-117: wanting signal via z_goal distance
        if (goal_state is not None
                and goal_state.is_active()
                and self.config.goal_weight > 0.0):
            g = self.compute_goal_score(trajectory, goal_state)
            # SD-016 (MECH-152): scale goal proximity by w_goal from terrain_weight
            if terrain_weight is not None:
                w_goal = terrain_weight[:, 1]  # [batch]
                g = g * w_goal
            score = score - self.config.goal_weight * g

        return score

    # ------------------------------------------------------------------ #
    # Selection                                                            #
    # ------------------------------------------------------------------ #

    def select(
        self,
        candidates: List[Trajectory],
        temperature: float = 1.0,
        goal_state: Optional[GoalState] = None,
        harm_bridge: Optional["nn.Module"] = None,
        use_harm_variance_commit: bool = False,
        terrain_weight: Optional[torch.Tensor] = None,
        sweep_threshold_reduction: float = 0.0,
        z_harm_a: Optional[torch.Tensor] = None,
        harm_forward_model: Optional["nn.Module"] = None,
        z_harm_s_current: Optional[torch.Tensor] = None,
    ) -> SelectionResult:
        """
        Select the best trajectory from candidates.

        Uses dynamic precision (ARC-016) to determine commit threshold.

        Args:
            candidates:             list of Trajectory objects
            temperature:            softmax temperature (exploration vs exploitation)
            harm_bridge:            optional HarmBridge (SD-010); if provided, M(zeta) uses
                                    harm stream scores
            use_harm_variance_commit: if True, commit decision uses variance of harm
                                    scores across candidates rather than z_world running
                                    variance (ARC-016 reframe).
            terrain_weight:         [batch, 2] or None (SD-016 MECH-152).
            sweep_threshold_reduction: MECH-108 BreathOscillator threshold reduction.
            z_harm_a:               SD-011 affective-motivational harm latent [batch, z_harm_a_dim].
                                    When provided and urgency_weight > 0, lowers effective
                                    commit threshold under accumulated threat (D2 avoidance).
                                    When provided and affective_harm_scale > 0, amplifies
                                    lambda_ethical in score_trajectory().
            harm_forward_model:     SD-011/ARC-033 ResidualHarmForward instance. When
                                    provided with z_harm_s_current, replaces harm_bridge
                                    for M(zeta) computation via step-by-step rollout.
            z_harm_s_current:       [batch, z_harm_dim] current sensory-discriminative harm
                                    latent. Required when harm_forward_model is provided.

        Returns:
            SelectionResult
        """
        if not candidates:
            raise ValueError("No candidate trajectories provided")

        scores = torch.stack([
            self.score_trajectory(
                t, goal_state=goal_state, harm_bridge=harm_bridge,
                terrain_weight=terrain_weight,
                harm_forward_model=harm_forward_model,
                z_harm_s_current=z_harm_s_current,
                z_harm_a=z_harm_a,
            )
            for t in candidates
        ])
        scores = scores.mean(dim=-1)
        self.last_scores = scores.detach()

        probs = F.softmax(-scores / temperature, dim=0)

        # ARC-016 commit decision: two modes
        # Default: commit when z_world running_variance is LOW (world-stability signal)
        # Reframe: commit when variance of harm scores across candidates is LOW
        #
        # MECH-108: BreathOscillator sweep reduces effective threshold, creating
        # periodic uncommitted windows even after training converges variance below
        # base threshold. Without this, the agent becomes permanently committed.
        effective_threshold = self.commit_threshold
        if sweep_threshold_reduction > 0.0:
            effective_threshold = effective_threshold * (1.0 - sweep_threshold_reduction)

        # SD-011: z_harm_a urgency modulation.
        # High accumulated threat -> LOWER effective threshold -> commit faster.
        # D2 avoidance escape response. Capped by urgency_max to prevent
        # threshold collapse to zero (which would produce permanent commitment).
        urgency_applied = 0.0
        if z_harm_a is not None and self.config.urgency_weight > 0.0:
            z_harm_a_norm = z_harm_a.norm(dim=-1).mean().item()
            urgency_applied = min(
                z_harm_a_norm * self.config.urgency_weight,
                self.config.urgency_max,
            )
            effective_threshold = effective_threshold * (1.0 - urgency_applied)

        if use_harm_variance_commit and harm_bridge is not None:
            harm_scores = torch.stack([
                self.compute_harm_stream_cost(t, harm_bridge).mean()
                for t in candidates
            ])
            harm_score_variance = harm_scores.var().item()
            committed = harm_score_variance < effective_threshold
        else:
            committed = self._running_variance < effective_threshold
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
        # Always store for rv updates (ARC-016 deadlock fix)
        self._last_selected_trajectory = selected_trajectory

        return SelectionResult(
            selected_trajectory=selected_trajectory,
            selected_index=selected_idx,
            selected_action=selected_action,
            scores=scores,
            precision=self.current_precision,
            committed=committed,
            log_prob=log_prob,
            urgency=urgency_applied,
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

        # ARC-016: update running variance from any selected trajectory, not
        # just committed ones.  Previous code gated on _committed_trajectory,
        # creating a deadlock: rv starts at precision_init (0.5), above
        # commit_threshold (0.40), so the agent never commits, rv never
        # updates, and the agent can never commit.
        ref_trajectory = self._committed_trajectory or self._last_selected_trajectory
        if ref_trajectory is not None and ref_trajectory.world_states is not None:
            predicted_world = ref_trajectory.world_states[1]
            prediction_error = actual_z_world - predicted_world

            self.update_running_variance(prediction_error)

            metrics["prediction_error"] = prediction_error.pow(2).mean()
            metrics["running_variance"] = torch.tensor(self._running_variance)
            metrics["dynamic_precision"] = torch.tensor(self.current_precision)

        # Residue accumulation stays commitment-gated (only accumulate for
        # actions the agent was committed to)
        if self._committed_trajectory is not None and harm_occurred and self.residue_field is not None:
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
        goal_state: Optional[GoalState] = None,
        harm_bridge: Optional["nn.Module"] = None,
        use_harm_variance_commit: bool = False,
        terrain_weight: Optional[torch.Tensor] = None,
    ) -> SelectionResult:
        return self.select(
            candidates,
            temperature,
            goal_state=goal_state,
            harm_bridge=harm_bridge,
            use_harm_variance_commit=use_harm_variance_commit,
            terrain_weight=terrain_weight,
        )
