"""
HippocampalModule — V3 Implementation

V3 changes (SD-004, Q-020 ARC-007 strict):

SD-004 — Action-object space navigation:
  HippocampalModule now navigates ACTION-OBJECT space O, not raw z_world.
  E2.action_object(z_world, a) produces o_t — a compressed representation
  of the world-effect of action a from z_world. The CEM search operates
  over action-object proposals, not raw action sequences.

  This extends the effective planning horizon because:
  a) action-object space is lower-dimensional than z_world
  b) action objects are semantically grounded (encode world-effects)
  c) the hippocampal map is indexed over world-effects, not states

Q-020 — ARC-007 STRICT (decided 2026-03-16):
  HippocampalModule generates VALUE-FLAT proposals.
  Terrain sensitivity is a CONSEQUENCE of navigating a residue-shaped
  z_world, not an independent hippocampal value computation.
  No separate value head. Residue field curvature in z_world IS the terrain.
  E3 introduces all weighting via Φ_R(ζ) in J(ζ).

SD-002 (preserved from V2): E1 world-domain prior wired into terrain search.
  terrain_prior: (z_world, e1_prior, residue_val) → action_object_mean

MECH-092 (replay):
  replay() is called during quiescent E3 heartbeat cycles. Carries
  hypothesis_tag=True — replay cannot produce residue (MECH-094).
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ree_core.utils.config import HippocampalConfig
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.residue.field import ResidueField, VALENCE_WANTING


class HippocampalModule(nn.Module):
    """
    HippocampalModule — action-object space trajectory proposal.

    V3 role:
    - Uses E2.action_object() to build proposals in action-object space O
    - CEM refinement over action-object means (not raw z_world or z_self)
    - Residue field scores terrain; proposals naturally avoid high-residue regions
    - No independent value computation (ARC-007 strict, Q-020 resolved)

    SD-002 preserved: E1 world-domain prior conditions initial terrain proposal.

    replay() method for MECH-092 SWR-equivalent consolidation.
    """

    def __init__(
        self,
        config: HippocampalConfig,
        e2: E2FastPredictor,
        residue_field: ResidueField,
    ):
        super().__init__()
        self.config = config
        self.e2 = e2
        self.residue_field = residue_field

        # SD-004 + SD-002: terrain prior maps (z_world, e1_prior, residue_val[, benefit_val])
        # → action_object_mean (in action-object space O).
        # Output: action_object_dim * horizon (flattened)
        # Input: 2*world_dim + 1 (harm only) or 2*world_dim + 2 (harm + benefit, ARC-030)
        # Benefit channel added when residue_field.benefit_terrain_enabled=True.
        # Wanting/liking distinction (MECH-117): benefit terrain = liking (hippocampal);
        # z_goal = wanting (frontal). Both are separate inputs to trajectory scoring.
        self._benefit_terrain = getattr(residue_field, "benefit_terrain_enabled", False)
        terrain_input_dim = config.world_dim * 2 + 1 + (1 if self._benefit_terrain else 0)
        self.terrain_prior = nn.Sequential(
            nn.Linear(terrain_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_object_dim * config.horizon),
        )

        # Action-object → action decoder: used to generate actual action sequences
        # from action-object proposals (needed for E2 rollout).
        # Maps o_t back to an action a_t.
        self.action_object_decoder = nn.Sequential(
            nn.Linear(config.action_object_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        # ARC-028 / MECH-105: last completion signal cache
        self._last_completion_signal: float = 0.0

    def _get_terrain_action_object_mean(
        self,
        z_world: torch.Tensor,
        e1_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute terrain-informed action-object distribution mean (SD-004).

        Queries residue field at z_world and combines with E1 prior (SD-002)
        to bias the initial action-object proposal toward low-residue regions.

        Args:
            z_world:  [batch, world_dim]
            e1_prior: world-domain E1 prior [batch, world_dim]. If None, zeros.

        Returns:
            action_object_mean [batch, horizon, action_object_dim]
        """
        with torch.no_grad():
            residue_val = self.residue_field.evaluate(z_world).unsqueeze(-1)  # [batch, 1]

        if e1_prior is None:
            e1_prior = torch.zeros_like(z_world)

        if self._benefit_terrain:
            with torch.no_grad():
                benefit_val = self.residue_field.evaluate_benefit(z_world).unsqueeze(-1)
            combined = torch.cat([z_world, e1_prior, residue_val, benefit_val], dim=-1)
        else:
            combined = torch.cat([z_world, e1_prior, residue_val], dim=-1)
        mean_flat = self.terrain_prior(combined)  # [batch, action_object_dim * horizon]
        return mean_flat.view(
            z_world.shape[0], self.config.horizon, self.config.action_object_dim
        )

    def _decode_action_objects(
        self,
        action_objects: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode action-object sequence to action sequence.

        Args:
            action_objects: [batch, horizon, action_object_dim]

        Returns:
            actions [batch, horizon, action_dim]
        """
        batch, horizon, ao_dim = action_objects.shape
        flat = action_objects.reshape(batch * horizon, ao_dim)
        actions_flat = self.action_object_decoder(flat)
        return actions_flat.reshape(batch, horizon, self.config.action_dim)

    def _score_trajectory(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Score a trajectory for CEM elite selection.

        ARC-007 STRICT: primary scoring is terrain-only (residue field over z_world).
        No independent harm prediction here -- E3 introduces all value weighting.

        VALENCE_WANTING extension (config.wanting_weight > 0):
        Subtracts the mean VALENCE_WANTING value along the trajectory from the
        terrain score, biasing CEM selection toward high-wanting (resource-proximal)
        regions. Subtraction because CEM minimises the score (lower = better).
        The wanting gradient is the residue field's accumulated VALENCE_WANTING
        component -- populated by SerotoninModule.update_benefit_salience() during
        waking steps when tonic_5ht_enabled=True.

        Returns a scalar score (lower = better).
        """
        if trajectory.world_states is not None:
            world_seq = trajectory.get_world_state_sequence()  # [batch, horizon, world_dim]
            if world_seq is not None:
                terrain_score = self.residue_field.evaluate_trajectory(world_seq).sum()

                if self.config.wanting_weight > 0:
                    # Reshape to [batch*horizon, world_dim] for evaluate_valence
                    batch, horizon, world_dim = world_seq.shape
                    flat = world_seq.reshape(batch * horizon, world_dim)
                    with torch.no_grad():
                        valence_flat = self.residue_field.evaluate_valence(flat)  # [B*H, 4]
                    wanting_flat = valence_flat[..., VALENCE_WANTING]  # [B*H]
                    wanting_score = wanting_flat.mean()
                    return terrain_score - self.config.wanting_weight * wanting_score

                return terrain_score

        # Fallback: residue over z_self states (pre-SD-005 wiring)
        states = trajectory.get_state_sequence()
        return self.residue_field.evaluate_trajectory(states).sum()

    def propose_trajectories(
        self,
        z_world: torch.Tensor,
        z_self: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        e1_prior: Optional[torch.Tensor] = None,
        action_bias: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        """
        Propose candidate trajectories via terrain-guided CEM in action-object space.

        V3 algorithm:
        1. Initialise action-object distribution mean from terrain prior
           (SD-004: in action-object space O, conditioned on E1 prior — SD-002)
        2. For each CEM iteration:
           a. Sample action-object sequences from current distribution
           b. Decode to action sequences via action_object_decoder
           c. Roll out through E2 (self + world tracks)
           d. Score via residue field terrain (ARC-007 strict — no value head)
           e. Refit distribution to elite (lowest-residue) samples
        3. Return final candidates for E3 to evaluate

        SD-016 (MECH-151): optional action_bias [batch, action_object_dim] from
        E1.extract_cue_context(). When provided, passed through to every
        E2.rollout_with_world() call so that each action-object in the CEM search
        is shifted by the cue-indexed contextual bias. When None (all existing
        callers), behaviour is unchanged.

        Args:
            z_world:        Current z_world [batch, world_dim]
            z_self:         Current z_self [batch, self_dim] (for E2 rollouts)
            num_candidates: Number of candidates per CEM iteration
            e1_prior:       E1 world-domain prior [batch, world_dim] (SD-002)
            action_bias:    [batch, action_object_dim] or None (SD-016)

        Returns:
            List of Trajectory objects
        """
        n = num_candidates or self.config.num_candidates
        num_elite = max(1, int(n * self.config.elite_fraction))
        batch_size = z_world.shape[0]
        device = z_world.device

        # Fallback z_self (zeros) when not provided
        if z_self is None:
            z_self = torch.zeros(batch_size, self.e2.config.self_dim, device=device)

        # Initialise in action-object space (SD-004)
        ao_mean = self._get_terrain_action_object_mean(z_world, e1_prior=e1_prior)
        ao_std  = torch.ones_like(ao_mean)

        all_trajectories: List[Trajectory] = []

        for _iteration in range(self.config.num_cem_iterations):
            trajectories: List[Trajectory] = []
            scores: List[torch.Tensor] = []

            for _ in range(n):
                noise = torch.randn_like(ao_mean)
                action_objects_sample = ao_mean + ao_std * noise  # [batch, H, ao_dim]

                # Decode action objects → actions for E2 rollout
                actions = self._decode_action_objects(action_objects_sample)

                # Roll out: track both z_self and z_world for scoring.
                # SD-016: action_bias is passed through to action_object() calls.
                traj = self.e2.rollout_with_world(
                    z_self, z_world, actions,
                    compute_action_objects=True,
                    action_bias=action_bias,
                )
                trajectories.append(traj)
                scores.append(self._score_trajectory(traj))

            scores_tensor = torch.stack(scores)
            elite_indices = torch.argsort(scores_tensor)[:num_elite]

            # Refit distribution to elite action-object sequences
            # Extract action objects from trajectories
            elite_ao = []
            for i in elite_indices:
                ao_seq = trajectories[i].get_action_object_sequence()
                if ao_seq is not None:
                    elite_ao.append(ao_seq)

            if elite_ao:
                elite_ao_tensor = torch.stack(elite_ao)  # [elite, batch, H, ao_dim]
                ao_mean = elite_ao_tensor.mean(dim=0)
                ao_std  = elite_ao_tensor.std(dim=0) + 1e-6
            # else: keep previous distribution

            all_trajectories = trajectories

        return all_trajectories

    def replay(
        self,
        theta_buffer_recent: torch.Tensor,
        num_replay_steps: int = 5,
        drive_state: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        """
        SWR-equivalent replay for viability map consolidation (MECH-092).

        Called during quiescent E3 heartbeat cycles (no salient event pending).
        All content carries hypothesis_tag=True — replay cannot produce residue
        (MECH-094 invariant).

        MECH-203 extension: when drive_state is provided (from serotonergic
        system), replay start point is selected by valence-weighted priority
        using ResidueField.get_valence_priority(). Without drive_state, falls
        back to most-recent z_world (original behavior, fully backward compat).

        Args:
            theta_buffer_recent: Recent theta-cycle buffer content
                [T, batch, world_dim]
            num_replay_steps: Number of replay trajectories to generate
            drive_state: Optional [4] drive weights for valence-weighted start
                point selection. When provided, each buffer entry is scored
                by dot(valence, drive_state) and the highest-priority entry
                is used as replay start. When None, uses most recent entry.

        Returns:
            List of Trajectory objects (all hypothesis_tag=True on caller side)
        """
        if theta_buffer_recent is None or theta_buffer_recent.shape[0] == 0:
            return []

        # Select replay start point
        if drive_state is not None and theta_buffer_recent.shape[0] > 1:
            # MECH-203: valence-weighted replay start selection
            z_world_replay = self._select_valence_weighted_start(
                theta_buffer_recent, drive_state
            )
        else:
            # Default: most recent z_world from buffer
            z_world_replay = theta_buffer_recent[-1]   # [batch, world_dim]

        batch_size = z_world_replay.shape[0]
        device = z_world_replay.device
        z_self_zeros = torch.zeros(batch_size, self.e2.config.self_dim, device=device)

        # Generate random action sequences for replay rollouts
        replay_trajectories = []
        for _ in range(num_replay_steps):
            actions = self.e2.generate_random_actions(
                batch_size, self.config.horizon, device
            )
            traj = self.e2.rollout_with_world(
                z_self_zeros, z_world_replay, actions, compute_action_objects=False
            )
            replay_trajectories.append(traj)

        return replay_trajectories

    def _select_valence_weighted_start(
        self,
        theta_buffer_recent: torch.Tensor,
        drive_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        MECH-203: Select replay start point by valence priority.

        Scores each buffer entry via ResidueField.get_valence_priority()
        and returns the highest-priority entry. Falls back to most recent
        if valence scoring is unavailable.

        Args:
            theta_buffer_recent: [T, batch, world_dim]
            drive_state: [4] drive weights

        Returns:
            z_world_replay: [batch, world_dim]
        """
        if not hasattr(self.residue_field, 'get_valence_priority'):
            return theta_buffer_recent[-1]

        T = theta_buffer_recent.shape[0]
        best_idx = T - 1  # default: most recent
        best_priority = -float('inf')

        with torch.no_grad():
            for t in range(T):
                z_w = theta_buffer_recent[t]  # [batch, world_dim]
                priority = self.residue_field.get_valence_priority(
                    z_w, drive_state
                )
                # Sum across batch for comparison
                p_val = float(priority.sum().item())
                if p_val > best_priority:
                    best_priority = p_val
                    best_idx = t

        return theta_buffer_recent[best_idx]

    def compute_completion_signal(self, trajectories: List[Trajectory]) -> float:
        """
        Compute the hippocampal trajectory completion signal (ARC-028, MECH-105).

        This is the dopamine-analog value fed to BetaGate.receive_hippocampal_completion().
        It implements the subiculum->NAc->VP->VTA loop described in Lisman & Grace (2005):
        a high-quality trajectory (low residue cost) drives a dopamine-analog release that
        causes the BG beta gate to drop, allowing E3 state to propagate to action selection.

        Formula:
            best_score = min residue cost across trajectories (lower = better)
            completion_signal = sigmoid(-best_score * 0.5)
            -> maps [0, inf) residue to [0.5, 1.0) signal
            -> high completion (near 0 residue) -> signal near 1.0
            -> poor completion (high residue) -> signal near 0.5
            -> empty list -> 0.0

        The result is cached in self._last_completion_signal.

        Args:
            trajectories: List of Trajectory objects (as returned by propose_trajectories).

        Returns:
            float in [0.0, 1.0): dopamine-analog completion quality signal.
        """
        if not trajectories:
            self._last_completion_signal = 0.0
            return 0.0

        scores = [self._score_trajectory(t) for t in trajectories]
        best_score = min(s.item() if isinstance(s, torch.Tensor) else float(s) for s in scores)
        signal = float(torch.sigmoid(torch.tensor(-best_score * 0.5)).item())
        self._last_completion_signal = signal
        return signal

    def forward(
        self,
        z_world: torch.Tensor,
        z_self: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        e1_prior: Optional[torch.Tensor] = None,
        action_bias: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        """Forward pass: propose trajectories from z_world."""
        return self.propose_trajectories(
            z_world, z_self=z_self, num_candidates=num_candidates,
            e1_prior=e1_prior, action_bias=action_bias,
        )
