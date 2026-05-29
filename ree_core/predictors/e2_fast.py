"""
E2 Fast Predictor — V3 Implementation

V3 extensions (SD-004, SD-005):

SD-005 — Self/World latent split:
  forward()       — motor-sensory: z_self_t + a → z_self_{t+1}   (E2 primary domain)
  world_forward() — world-state:   z_world_t + a → z_world_{t+1} (for SD-003 attribution)

SD-004 — Action objects:
  action_object() — compressed world-effect: (z_world_t, a_t) → o_t
  o_t is a low-dimensional representation of the world-effect of taking
  action a_t from z_world_t. HippocampalModule builds its map over
  action-object space O, not raw z_world.

SD-003 V3 attribution pipeline:
    z_world_actual = e2.world_forward(z_world, a_actual)
    z_world_cf     = e2.world_forward(z_world, a_cf)
    harm_actual    = e3.harm_eval(z_world_actual)
    harm_cf        = e3.harm_eval(z_world_cf)
    causal_sig     = harm_actual - harm_cf
    world_delta    = ||z_world_actual - z_world_cf||

E2 trains ONLY on motor-sensory prediction error (z_self). NOT harm/goal error.
harm_predict head from V2 is REMOVED — harm evaluation belongs to E3.

rollout_horizon must exceed E1.prediction_horizon (30 > 20).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E2Config
from ree_core.latent.stack import LatentState


@dataclass
class Trajectory:
    """Candidate trajectory through latent space.

    V3: states are z_self sequences (E2 primary domain).
    world_states (if present) are z_world projections for SD-003 use.
    action_objects (if present) are o_t sequences for HippocampalModule.
    harm_predictions removed — E3 handles harm evaluation.
    """
    states: List[torch.Tensor]           # z_self: List of [batch, self_dim]
    actions: torch.Tensor                # [batch, horizon, action_dim]
    world_states: Optional[List[torch.Tensor]] = None  # z_world: for SD-003 attribution
    action_objects: Optional[List[torch.Tensor]] = None  # o_t: for HippocampalModule
    is_reverse: bool = False             # MECH-165: True when trajectory is reverse-replayed
    memory_strength: float = 1.0         # BLA write-strength proxy for replay sampling
    arousal_tag: float = 0.0             # BLA retrieval tag written at encoding time
    # MECH-094 / MECH-293: provenance tag for trajectory routing.
    # True for ghost-seeded probes (MECH-293) and any other simulation /
    # replay-derived trajectory that downstream write paths must treat as
    # hypothetical. False for waking value-flat CEM proposals AND for the
    # executed committed trajectory (record_committed_trajectory sets it
    # back to False explicitly even if the source proposal was a ghost).
    hypothesis_tag: bool = False
    # MECH-293: per-trajectory provenance dict. None for value-flat CEM
    # proposals; ghost probes carry {"source": "mech293_ghost_probe",
    # "anchor_key": ..., "ghost_priority": ..., "goal_match": ...} so
    # diagnostics can attribute downstream behaviour to the bank entry
    # that seeded the probe. Optional[Dict] keeps existing call sites
    # bit-identical when omitted.
    metadata: Optional[Dict[str, Any]] = None

    @property
    def total_length(self) -> int:
        return len(self.states)

    def get_final_state(self) -> torch.Tensor:
        return self.states[-1]

    def get_state_sequence(self) -> torch.Tensor:
        """Stack z_self states [batch, horizon+1, self_dim]."""
        return torch.stack(self.states, dim=1)

    def get_world_state_sequence(self) -> Optional[torch.Tensor]:
        """Stack z_world states [batch, horizon+1, world_dim]. None if not tracked."""
        if self.world_states is None:
            return None
        return torch.stack(self.world_states, dim=1)

    def get_action_object_sequence(self) -> Optional[torch.Tensor]:
        """Stack action objects [batch, horizon, action_object_dim]. None if not computed."""
        if self.action_objects is None:
            return None
        return torch.stack(self.action_objects, dim=1)



class E2FastPredictor(nn.Module):
    """
    E2 Fast Predictor — V3.

    Primary role: fast motor-sensory transition model over z_self.

    Extended interface (SD-004, SD-005):
      forward(z_self, a)          → z_self_next  (motor-sensory prediction)
      world_forward(z_world, a)   → z_world_next (world prediction for attribution)
      action_object(z_world, a)   → o_t          (compressed world-effect, SD-004)
      forward_counterfactual(z_self, a_cf) → z_self_cf (SD-003 self-attr)

    NOT a harm predictor. harm_eval() belongs on E3Selector.
    """

    def __init__(self, config: Optional[E2Config] = None):
        super().__init__()
        self.config = config or E2Config()

        # --- Primary: motor-sensory transition model (z_self domain) ---
        # z_self_{t+1} = z_self_t + delta(z_self_t, a_t)
        self.self_transition = nn.Sequential(
            nn.Linear(self.config.self_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.self_dim),
        )
        self.self_action_encoder = nn.Linear(
            self.config.action_dim, self.config.action_dim
        )

        # --- SD-005: world_forward (z_world domain, for attribution only) ---
        # z_world_{t+1} = z_world_t + delta_w(z_world_t, a_t)
        # Lightweight head — may share hidden layers in future iterations.
        self.world_transition = nn.Sequential(
            nn.Linear(self.config.world_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.world_dim),
        )
        self.world_action_encoder = nn.Linear(
            self.config.action_dim, self.config.action_dim
        )

        # --- SD-004: action_object head ---
        # Compressed representation of (z_world_t, a_t) world-effect.
        # Bottleneck ensures action_object_dim << world_dim.
        self.action_object_head = nn.Sequential(
            nn.Linear(self.config.world_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_object_dim),
        )

    # ------------------------------------------------------------------ #
    # Core interface                                                       #
    # ------------------------------------------------------------------ #

    def predict_next_self(
        self,
        z_self: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Motor-sensory prediction: z_self_t + a_t → z_self_{t+1}.

        This is E2's primary operation. Residual connection.

        Args:
            z_self:  [batch, self_dim]
            action:  [batch, action_dim]

        Returns:
            z_self_next [batch, self_dim]
        """
        a_enc = self.self_action_encoder(action)
        z_a = torch.cat([z_self, a_enc], dim=-1)
        delta = self.self_transition(z_a)
        return z_self + delta

    def world_forward(
        self,
        z_world: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        World-state prediction: z_world_t + a_t → z_world_{t+1}.

        Used in SD-003 V3 attribution pipeline only.
        NOT trained as E2's primary objective.

        Args:
            z_world: [batch, world_dim]
            action:  [batch, action_dim]

        Returns:
            z_world_next [batch, world_dim]
        """
        a_enc = self.world_action_encoder(action)
        z_a = torch.cat([z_world, a_enc], dim=-1)
        delta = self.world_transition(z_a)
        return z_world + delta

    def cand_world_pairwise_dist(
        self,
        z_world_0: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        """SD-056 diagnostic: mean pairwise L2 between K predicted z_world_1 outputs.

        For a batch of K CEM candidates sharing z_world_0 but differing in
        first action a_i, predict K z_world_1[i] via world_forward and return
        the mean pairwise L2 distance across the K predictions. Headline
        metric for V3-EXQ-571 root-cause / SD-056 substrate-readiness; named
        by the 2026-05-28 lit-pull SYNTHESIS verdict 3 as a methodological
        gap worth publishing as a standalone diagnostic.

        Under the pre-SD-056 substrate this returns ~0.0 across K candidates
        differing only in first action (V3-EXQ-571 measurement, 2026-05-16).
        After SD-056 training under contrastive loss it should rise above the
        substrate-readiness threshold (calibrated by V3-EXQ-NEW-1).

        Args:
            z_world_0:         [world_dim] OR [1, world_dim] starting state
            candidate_actions: [K, action_dim] action batch (typically first-step
                               actions from K sibling CEM candidates)

        Returns:
            0-d Tensor: mean pairwise L2 distance across K predictions.
            Returns tensor(0.0) on K < 2 (no pairs).
        """
        if candidate_actions.dim() != 2:
            raise ValueError(
                f"candidate_actions must be [K, action_dim]; got shape "
                f"{tuple(candidate_actions.shape)}"
            )
        K = candidate_actions.shape[0]
        if K < 2:
            return torch.zeros((), device=candidate_actions.device,
                               dtype=candidate_actions.dtype)
        if z_world_0.dim() == 1:
            z0 = z_world_0.unsqueeze(0).expand(K, -1)
        elif z_world_0.dim() == 2 and z_world_0.shape[0] == 1:
            z0 = z_world_0.expand(K, -1)
        elif z_world_0.dim() == 2 and z_world_0.shape[0] == K:
            z0 = z_world_0
        else:
            raise ValueError(
                f"z_world_0 must be [world_dim] OR [1, world_dim] OR [K, world_dim]; "
                f"got shape {tuple(z_world_0.shape)} with K={K}"
            )
        preds = self.world_forward(z0, candidate_actions)  # [K, world_dim]
        dists = torch.cdist(preds, preds)  # [K, K]
        K_int = int(K)
        eye = torch.eye(K_int, dtype=torch.bool, device=preds.device)
        off_diag = dists[~eye]
        return off_diag.mean()

    def world_forward_contrastive_loss(
        self,
        z_world_0: torch.Tensor,
        actions: torch.Tensor,
        z_world_1_targets: torch.Tensor,
        weight: Optional[float] = None,
        temperature: Optional[float] = None,
        min_batch_classes: Optional[int] = None,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """SD-056 auxiliary InfoNCE contrastive loss on world_forward.

        For K sibling CEM candidates sharing z_world_0 but differing in
        first action a_i, enforce that predicted z_world_1[i] is closer to
        target i than to target j != i in latent L2:

            logits[i, j] = -||pred_j - target_i||^2 / tau
            L_contrast = cross_entropy(logits, label=arange(K))

        Asymmetric anchor-to-prediction form per the design memo
        (REE_assembly/evidence/planning/e2_action_divergence_substrate_design.md):
        symmetric InfoNCE doubles cost without architectural gain.

        Helper does NOT modify autograd state externally; caller is
        responsible for adding `weight * L_contrast` to the total E2 loss
        and backpropping. Negatives are computed within this method by
        running world_forward over the K actions (sibling-CEM-candidate
        negatives are informative by construction).

        Returns tensor(0.0) when:
            (a) simulation_mode=True (MECH-094 standard pattern -- replay /
                DMN training paths cannot recruit the contrastive signal);
            (b) K < 2 (no negatives);
            (c) fewer than min_batch_classes distinct first-action classes
                in the batch (degenerate; would produce uninformative
                gradients).

        Args:
            z_world_0:         [world_dim] OR [1, world_dim] OR [K, world_dim]
            actions:           [K, action_dim]; argmax over last dim taken as
                               the first-action class for the min_batch_classes
                               floor check.
            z_world_1_targets: [K, world_dim] -- observed next-state targets
                               for each (z_world_0, a_i) pair.
            weight:            optional override for w_contrast scaling. The
                               helper returns the unweighted CE; caller scales
                               by config.e2.e2_action_contrastive_weight when
                               composing into L_E2. Argument retained for
                               caller-side override convenience.
            temperature:       optional InfoNCE tau override.
            min_batch_classes: optional first-action-class-floor override.
            simulation_mode:   MECH-094 gate. True -> tensor(0.0); no state
                               advance (helper is stateless beyond the
                               weights it shares with world_forward).

        Returns:
            0-d Tensor: contrastive loss (unweighted CE). Caller multiplies
            by w_contrast before adding to L_E2.
        """
        device = actions.device
        dtype = z_world_1_targets.dtype

        if simulation_mode:
            return torch.zeros((), device=device, dtype=dtype)

        if actions.dim() != 2:
            raise ValueError(
                f"actions must be [K, action_dim]; got shape {tuple(actions.shape)}"
            )
        K = actions.shape[0]
        if K < 2:
            return torch.zeros((), device=device, dtype=dtype)
        if z_world_1_targets.dim() != 2 or z_world_1_targets.shape[0] != K:
            raise ValueError(
                f"z_world_1_targets must be [K, world_dim] with K={K}; got "
                f"shape {tuple(z_world_1_targets.shape)}"
            )

        cfg = self.config
        if temperature is None:
            temperature = float(getattr(cfg, "e2_action_contrastive_temperature", 0.1))
        if min_batch_classes is None:
            min_batch_classes = int(
                getattr(cfg, "e2_action_contrastive_min_batch_classes", 2)
            )
        # Note: `weight` is accepted for API symmetry but the helper returns
        # the unweighted CE -- caller composes weight at the loss-summation
        # site (matches the SD-019 / MECH-258 / MECH-273 pattern of
        # auxiliary-loss helpers returning unweighted terms).

        # Min-batch-classes floor: count distinct first-action classes (argmax
        # over action_dim). One-hot inputs collapse to integer class labels;
        # continuous actions still produce class labels via argmax (the
        # min_batch_classes guard is structural -- it would not fire on a
        # well-formed K-candidate sibling batch).
        first_action_classes = actions.argmax(dim=-1)  # [K]
        n_classes = int(first_action_classes.unique().numel())
        if n_classes < int(min_batch_classes):
            return torch.zeros((), device=device, dtype=dtype)

        # Broadcast z_world_0 to [K, world_dim] so world_forward sees K
        # candidates with shared starting state but distinct actions.
        if z_world_0.dim() == 1:
            z0 = z_world_0.unsqueeze(0).expand(K, -1)
        elif z_world_0.dim() == 2 and z_world_0.shape[0] == 1:
            z0 = z_world_0.expand(K, -1)
        elif z_world_0.dim() == 2 and z_world_0.shape[0] == K:
            z0 = z_world_0
        else:
            raise ValueError(
                f"z_world_0 must be [world_dim] OR [1, world_dim] OR [K, world_dim]; "
                f"got shape {tuple(z_world_0.shape)} with K={K}"
            )

        preds = self.world_forward(z0, actions)  # [K, world_dim]

        # logits[i, j] = -||pred_j - target_i||^2 / tau
        # cdist^2 in [K (targets), K (preds)] form (anchors index i, preds index j).
        diffs = preds.unsqueeze(0) - z_world_1_targets.unsqueeze(1)  # [K, K, world_dim]
        sq_dists = diffs.pow(2).sum(dim=-1)  # [K, K]
        logits = -sq_dists / float(temperature)

        labels = torch.arange(K, device=device)
        return F.cross_entropy(logits, labels)

    def action_object(
        self,
        z_world: torch.Tensor,
        action: torch.Tensor,
        action_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Produce compressed world-effect action object o_t (SD-004).

        o_t = E2.action_object(z_world_t, a_t) encodes the world-effect
        of taking action a_t from world-state z_world_t.

        HippocampalModule builds its trajectory map in action-object space O.
        Planning horizon extends because action-object space is lower-dimensional
        and semantically grounded.

        SD-016 (MECH-151): optional action_bias [batch, action_object_dim] from
        E1.extract_cue_context(). When provided, added to o_t AFTER the base
        computation. This shifts the apparent affordance of each action-object
        by a cue-indexed contextual signal without replacing the base output.
        When None (all existing callers), behaviour is unchanged.

        Args:
            z_world:     [batch, world_dim]
            action:      [batch, action_dim]
            action_bias: [batch, action_object_dim] or None (SD-016)

        Returns:
            o_t [batch, action_object_dim]
        """
        z_a = torch.cat([z_world, action], dim=-1)
        o_t = self.action_object_head(z_a)
        if action_bias is not None:
            o_t = o_t + action_bias
        return o_t

    def forward_counterfactual(
        self,
        z_self: torch.Tensor,
        counterfactual_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Counterfactual query: what z_self under a different action?

        Used in the old z_gamma-level SD-003 (V2 style).
        In V3, counterfactual attribution uses world_forward() instead.

        Args:
            z_self: [batch, self_dim]
            counterfactual_action: [batch, action_dim]

        Returns:
            z_self_cf [batch, self_dim]
        """
        return self.predict_next_self(z_self, counterfactual_action)

    # ------------------------------------------------------------------ #
    # Rollout and candidate generation                                     #
    # ------------------------------------------------------------------ #

    def rollout_self(
        self,
        initial_z_self: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> "Trajectory":
        """
        Roll out z_self trajectory given an action sequence.

        Args:
            initial_z_self: [batch, self_dim]
            action_sequence: [batch, horizon, action_dim]

        Returns:
            Trajectory (z_self states only)
        """
        horizon = action_sequence.shape[1]
        states = [initial_z_self]
        current = initial_z_self

        for t in range(horizon):
            action = action_sequence[:, t, :]
            current = self.predict_next_self(current, action)
            states.append(current)

        return Trajectory(states=states, actions=action_sequence)

    def rollout_with_world(
        self,
        initial_z_self: torch.Tensor,
        initial_z_world: torch.Tensor,
        action_sequence: torch.Tensor,
        compute_action_objects: bool = True,
        action_bias: Optional[torch.Tensor] = None,
    ) -> "Trajectory":
        """
        Roll out trajectory tracking both z_self and z_world.

        Used by HippocampalModule for action-object proposals and
        by SD-003 attribution pipeline.

        SD-016 (MECH-151): optional action_bias [batch, action_object_dim] from
        E1.extract_cue_context(). Applied to each action_object() call within
        the rollout. When None (all existing callers), behaviour is unchanged.

        Args:
            initial_z_self:        [batch, self_dim]
            initial_z_world:       [batch, world_dim]
            action_sequence:       [batch, horizon, action_dim]
            compute_action_objects: whether to compute action objects (SD-004)
            action_bias:           [batch, action_object_dim] or None (SD-016)

        Returns:
            Trajectory with states (z_self), world_states (z_world),
            and optionally action_objects (o_t)
        """
        horizon = action_sequence.shape[1]
        states = [initial_z_self]
        world_states = [initial_z_world]
        action_objects = []

        z_self  = initial_z_self
        z_world = initial_z_world

        for t in range(horizon):
            action = action_sequence[:, t, :]

            if compute_action_objects:
                o_t = self.action_object(z_world, action, action_bias=action_bias)
                action_objects.append(o_t)

            z_self  = self.predict_next_self(z_self, action)
            z_world = self.world_forward(z_world, action)

            states.append(z_self)
            world_states.append(z_world)

        return Trajectory(
            states=states,
            actions=action_sequence,
            world_states=world_states,
            action_objects=action_objects if compute_action_objects else None,
        )

    def generate_random_actions(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.randn(batch_size, horizon, self.config.action_dim, device=device)

    def generate_candidates_random(
        self,
        initial_z_self: torch.Tensor,
        initial_z_world: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        horizon: Optional[int] = None,
        compute_action_objects: bool = True,
        action_bias: Optional[torch.Tensor] = None,
    ) -> List["Trajectory"]:
        """
        Generate candidate trajectories using random shooting.

        Used by HippocampalModule during CEM iterations.

        SD-016 (MECH-151): optional action_bias [batch, action_object_dim] passed
        through to each rollout's action_object() calls.

        Args:
            initial_z_self:        [batch, self_dim]
            initial_z_world:       [batch, world_dim] (optional; needed for SD-004 objects)
            num_candidates:        number of trajectories
            horizon:               rollout steps
            compute_action_objects: whether to compute action objects
            action_bias:           [batch, action_object_dim] or None (SD-016)

        Returns:
            List of Trajectory objects
        """
        n = num_candidates or self.config.num_candidates
        h = horizon or self.config.rollout_horizon
        device = initial_z_self.device
        batch_size = initial_z_self.shape[0]

        candidates = []
        for _ in range(n):
            actions = self.generate_random_actions(batch_size, h, device)
            if initial_z_world is not None:
                traj = self.rollout_with_world(
                    initial_z_self, initial_z_world, actions,
                    compute_action_objects, action_bias=action_bias,
                )
            else:
                traj = self.rollout_self(initial_z_self, actions)
            candidates.append(traj)

        return candidates

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compatibility shim: maps V2 API to V3.

        In V2 this operated on z_gamma. In V3, treats input as z_self.
        For new code, use predict_next_self() directly.
        """
        return self.predict_next_self(current_state, action)

    def forward(
        self,
        initial_z_self: torch.Tensor,
        num_candidates: Optional[int] = None,
    ) -> List["Trajectory"]:
        """Forward pass: generate random candidate trajectories."""
        return self.generate_candidates_random(
            initial_z_self, num_candidates=num_candidates, compute_action_objects=False
        )
