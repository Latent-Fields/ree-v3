"""
Residue Field φ(z_world) — V3 Implementation

V3 change (SD-005): ResidueField now operates exclusively over z_world
(the exteroceptive world-state latent). It does NOT operate over z_gamma
or z_self. This is architecturally correct: residue tracks the persistent
ethical cost of world-changes the agent caused, not body-state changes.

Accumulation contract (V3):
  - accumulate(z_world, world_delta, harm_magnitude) — called after action
    execution; world_delta = ||z_world_actual - z_world_cf|| from SD-003
    attribution pipeline. In early V3 experiments before SD-003 is fully
    wired, harm_magnitude alone drives accumulation (backward compat).
  - hypothesis_tag check (MECH-094): accumulate() refuses to accumulate
    when hypothesis_tag=True. Simulated/replay content cannot produce
    residue; only real world-outcomes can.
  - Residue cannot be erased (architectural invariant, preserved from V1/V2).

Benefit terrain (ARC-030, MECH-117):
  - When ResidueConfig.benefit_terrain_enabled=True, a separate RBFLayer
    accumulates positive attractors at resource contact/approach events via
    accumulate_benefit(). evaluate_benefit() returns a scalar >= 0; higher
    means closer to a previously-beneficial z_world region.
  - HippocampalModule reads both harm residue AND benefit terrain, giving the
    hippocampal landscape repellers (harm) and attractors (benefit).
  - Wanting/liking distinction (MECH-117): benefit terrain = LIKING (where
    benefit was received, hippocampal/contact-based). z_goal = WANTING
    (frontal attractor, persistent). These remain separate channels.

The ResidueField is an input to multiple modules (not just E3's Φ_R cost):
  - HippocampalModule: residue field is the terrain that action objects navigate
  - E3 scoring: Φ_R term in J(ζ)
  - (future) Sensorium gate: residue as attentional prior
"""

from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import ResidueConfig

# SD-014 / ARC-036: Valence vector component indices.
# Each hippocampal map node stores a 4-component valence vector updated incrementally.
# Component 0: wanting        -- z_goal signal (frontal goal attractor, drives approach)
# Component 1: liking         -- benefit terrain signal (where benefit was received)
# Component 2: harm_discriminative -- z_harm_s signal (sensory-discriminative, A-delta analog)
# Component 3: surprise       -- prediction error (novelty / unexpectedness signal)
VALENCE_WANTING: int = 0
VALENCE_LIKING: int = 1
VALENCE_HARM_DISCRIMINATIVE: int = 2
VALENCE_SURPRISE: int = 3
VALENCE_DIM: int = 4
VALENCE_COMPONENTS = [VALENCE_WANTING, VALENCE_LIKING, VALENCE_HARM_DISCRIMINATIVE, VALENCE_SURPRISE]


class RBFLayer(nn.Module):
    """RBF field over z_world — unchanged from V2 except latent_dim → world_dim."""

    def __init__(self, world_dim: int, num_centers: int, bandwidth: float = 1.0):
        super().__init__()
        self.world_dim = world_dim
        self.num_centers = num_centers
        self.bandwidth = bandwidth

        self.centers = nn.Parameter(torch.randn(num_centers, world_dim) * 0.1)
        self.weights = nn.Parameter(torch.zeros(num_centers))
        self.register_buffer("active_mask", torch.zeros(num_centers, dtype=torch.bool))
        self.register_buffer("next_center_idx", torch.tensor(0))
        # SD-014 / ARC-036: 4-component valence vector per center.
        # Shape [num_centers, VALENCE_DIM]: [wanting, liking, harm_discriminative, surprise].
        # Sparse by design -- most centers start at zeros and are updated incrementally.
        self.register_buffer("valence_vecs", torch.zeros(num_centers, VALENCE_DIM))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate residue field at z_world points.

        Args:
            z: [batch, world_dim] or [batch, seq, world_dim]

        Returns:
            residue values [batch] or [batch, seq]
        """
        original_shape = z.shape
        if z.dim() == 3:
            batch_size, seq_len, world_dim = z.shape
            z = z.reshape(-1, world_dim)
        else:
            batch_size = z.shape[0]
            seq_len = None

        diffs = z.unsqueeze(1) - self.centers.unsqueeze(0)
        distances_sq = (diffs ** 2).sum(dim=-1)
        rbf_values = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))
        active_weights = self.weights * self.active_mask.float()
        field_values = (rbf_values * active_weights.unsqueeze(0)).sum(dim=-1)

        if seq_len is not None:
            field_values = field_values.reshape(batch_size, seq_len)

        return field_values

    def add_residue(self, location: torch.Tensor, intensity: float = 1.0) -> int:
        if location.dim() == 2:
            location = location.mean(dim=0)

        idx = self.next_center_idx.item()
        with torch.no_grad():
            self.centers.data[idx] = location
            self.weights.data[idx] = self.weights.data[idx] + intensity
            self.active_mask[idx] = True
            self.next_center_idx = (self.next_center_idx + 1) % self.num_centers
        return idx

    def update_valence(
        self, center_idx: int, valence_component: int, value: float
    ) -> None:
        """
        Incrementally update a single valence component at a center (SD-014).

        Does NOT replace the existing value -- adds to it so the vector accumulates
        across visits. Callers are responsible for scaling (e.g. EMA step size).

        Args:
            center_idx:        Index of the RBF center to update (0-based).
            valence_component: One of VALENCE_WANTING/LIKING/HARM_DISCRIMINATIVE/SURPRISE.
            value:             Signed scalar to add to the component.
        """
        with torch.no_grad():
            self.valence_vecs[center_idx, valence_component] += value

    def evaluate_valence(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return weighted valence vector at z_world query points (SD-014).

        Uses the same RBF activations as forward() but sums valence_vecs instead
        of scalar weights. Only active centers contribute.

        Args:
            z: [batch, world_dim]

        Returns:
            valence: [batch, VALENCE_DIM]  -- component order [wanting, liking,
                     harm_discriminative, surprise]
        """
        if not self.active_mask.any():
            return torch.zeros(z.shape[0], VALENCE_DIM, device=z.device, dtype=z.dtype)

        # [batch, num_centers]
        diffs = z.unsqueeze(1) - self.centers.unsqueeze(0)
        distances_sq = (diffs ** 2).sum(dim=-1)
        rbf_values = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))

        # Zero out inactive centers
        active_rbf = rbf_values * self.active_mask.float().unsqueeze(0)  # [batch, num_centers]

        # Weighted sum of valence vecs: [batch, num_centers] x [num_centers, VALENCE_DIM]
        # -> [batch, VALENCE_DIM]
        valence = torch.matmul(active_rbf, self.valence_vecs)
        return valence


class ResidueField(nn.Module):
    """
    Persistent residue field φ(z_world) — V3.

    Key properties (all preserved from V2):
    1. PERSISTENCE: Residue cannot be erased
    2. PATH DEPENDENCE: Cost depends on trajectory through z_world
    3. ACCUMULATION: Harm adds to residue, never subtracts
    4. INTEGRATION: Offline processing contextualises but cannot remove

    V3 additions:
    - Operates over z_world (SD-005): self-state changes do not drive residue
    - hypothesis_tag check (MECH-094): simulation cannot produce residue
    - world_delta accumulation: magnitude of world-state change drives
      accumulation strength (requires SD-003 pipeline to be wired)

    SD-014 / ARC-036 — 4-component valence vector:
    Each RBF center also stores a 4-component valence vector [wanting, liking,
    harm_discriminative, surprise] that is updated incrementally as the agent
    visits different z_world regions.  The valence map enables drive-weighted
    replay prioritisation:  priority(node) = dot(V_node, d_current) + epsilon
    where d_current = [w_drive, l_drive, h_drive, s_drive] is the current
    drive-state vector.  API: update_valence(), evaluate_valence(),
    get_valence_priority().  Controlled by ResidueConfig.valence_enabled.
    """

    def __init__(self, config: Optional[ResidueConfig] = None):
        super().__init__()
        self.config = config or ResidueConfig()

        self.rbf_field = RBFLayer(
            world_dim=self.config.world_dim,
            num_centers=self.config.num_basis_functions,
            bandwidth=self.config.kernel_bandwidth,
        )

        self.neural_field = nn.Sequential(
            nn.Linear(self.config.world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Softplus(),
        )

        self.register_buffer("total_residue", torch.tensor(0.0))
        self.register_buffer("num_harm_events", torch.tensor(0))
        self._harm_history: List[torch.Tensor] = []

        # ARC-030 / MECH-117: benefit terrain (liking -- where benefit was received)
        # Separate from z_goal (wanting -- frontal goal attractor).
        self.benefit_terrain_enabled: bool = getattr(
            self.config, "benefit_terrain_enabled", False
        )
        if self.benefit_terrain_enabled:
            self.benefit_rbf_field = RBFLayer(
                world_dim=self.config.world_dim,
                num_centers=self.config.num_basis_functions,
                bandwidth=self.config.kernel_bandwidth,
            )
            self.register_buffer("total_benefit", torch.tensor(0.0))
            self.register_buffer("num_benefit_events", torch.tensor(0))

    def evaluate(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Evaluate φ(z_world) at a point.

        Args:
            z_world: [batch, world_dim]

        Returns:
            residue values [batch]
        """
        rbf_value = self.rbf_field(z_world)
        neural_value = self.neural_field(z_world).squeeze(-1)
        return rbf_value + neural_value * 0.1

    def evaluate_trajectory(self, trajectory_world_states: torch.Tensor) -> torch.Tensor:
        """
        Evaluate total residue cost along a trajectory through z_world.

        Args:
            trajectory_world_states: [batch, horizon, world_dim]

        Returns:
            total residue cost [batch]
        """
        field_values = self.rbf_field(trajectory_world_states)          # [batch, horizon]
        neural_values = self.neural_field(trajectory_world_states).squeeze(-1)
        total_values = field_values + neural_values * 0.1
        return total_values.sum(dim=-1)                                  # [batch]

    def accumulate(
        self,
        z_world: torch.Tensor,
        harm_magnitude: float = 1.0,
        world_delta: Optional[float] = None,
        hypothesis_tag: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate residue at a z_world location (MECH-094 gated).

        MECH-094 invariant: if hypothesis_tag=True, refuse accumulation.
        Simulated/replay content must not produce real residue.

        Args:
            z_world:        Location in z_world space [batch, world_dim]
            harm_magnitude: Base harm magnitude (positive = bad)
            world_delta:    Optional |z_world_actual - z_world_cf| from SD-003.
                            When provided, scales accumulation (agent-caused
                            world-change drives residue, not just harm signal).
            hypothesis_tag: MECH-094 gate. If True, no accumulation occurs.

        Returns:
            Dictionary of accumulation metrics
        """
        # MECH-094: simulation cannot produce residue
        if hypothesis_tag:
            return {
                "residue_added": torch.tensor(0.0),
                "total_residue": self.total_residue,
                "skipped_hypothesis": torch.tensor(1.0),
            }

        magnitude = abs(harm_magnitude) * self.config.accumulation_rate

        # SD-003 V3: world_delta scales accumulation when attribution is wired
        if world_delta is not None:
            magnitude *= min(2.0, world_delta)  # cap at 2× to prevent explosion

        center_idx = self.rbf_field.add_residue(z_world, magnitude)
        self.total_residue = self.total_residue + magnitude
        self.num_harm_events = self.num_harm_events + 1
        self._harm_history.append(z_world.detach().clone())

        return {
            "residue_added": torch.tensor(magnitude),
            "total_residue": self.total_residue,
            "center_idx": torch.tensor(center_idx),
            "num_harm_events": self.num_harm_events,
        }

    def evaluate_benefit(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Evaluate benefit terrain at z_world (ARC-030 / MECH-117 liking signal).

        Returns benefit attraction value [batch]. Higher = closer to a
        previously-beneficial region. Returns zeros if benefit terrain disabled.
        """
        if not self.benefit_terrain_enabled:
            return torch.zeros(z_world.shape[0], device=z_world.device)
        return self.benefit_rbf_field(z_world)

    def accumulate_benefit(
        self,
        z_world: torch.Tensor,
        benefit_magnitude: float = 1.0,
        hypothesis_tag: bool = False,
    ) -> None:
        """
        Accumulate benefit attractor at a z_world location (ARC-030).

        MECH-094: hypothesis_tag=True blocks accumulation (replay cannot
        create benefit terrain, only real reward contact can).

        Args:
            z_world:           Location in z_world space [batch, world_dim]
            benefit_magnitude: Strength of attractor (default 1.0)
            hypothesis_tag:    MECH-094 gate -- if True, no accumulation
        """
        if not self.benefit_terrain_enabled or hypothesis_tag:
            return
        if z_world.dim() == 2:
            loc = z_world.mean(dim=0)
        else:
            loc = z_world
        with torch.no_grad():
            self.benefit_rbf_field.add_residue(loc, float(benefit_magnitude))
            self.total_benefit = self.total_benefit + benefit_magnitude
            self.num_benefit_events = self.num_benefit_events + 1

    # ------------------------------------------------------------------
    # SD-014 / ARC-036: 4-component valence vector API
    # ------------------------------------------------------------------

    def update_valence(
        self,
        z_world: torch.Tensor,
        component: int,
        value: float,
        hypothesis_tag: bool = False,
    ) -> None:
        """
        Update valence at the nearest active RBF center (SD-014).

        MECH-094 gate: if hypothesis_tag=True, skip (simulated/replay content
        cannot update real valence, mirroring the accumulate() invariant).

        If no active centers exist yet, skips silently (avoids a crash during
        early training before any residue has been accumulated).

        Args:
            z_world:        Location in z_world space [batch, world_dim] or [world_dim].
            component:      Valence component index (use VALENCE_* constants).
            value:          Signed scalar to add (incremental, not replace).
            hypothesis_tag: MECH-094 gate -- if True, no update occurs.
        """
        if hypothesis_tag:
            return
        if not getattr(self.config, "valence_enabled", True):
            return
        if not self.rbf_field.active_mask.any():
            return

        # Reduce batch to a single representative point
        if z_world.dim() == 2:
            z_point = z_world.mean(dim=0, keepdim=True)   # [1, world_dim]
        else:
            z_point = z_world.unsqueeze(0)                 # [1, world_dim]

        # Find nearest active center
        active_idxs = self.rbf_field.active_mask.nonzero(as_tuple=True)[0]  # [n_active]
        active_centers = self.rbf_field.centers[active_idxs]                  # [n_active, world_dim]
        diffs = z_point - active_centers                                       # [n_active, world_dim]
        dists_sq = (diffs ** 2).sum(dim=-1)                                   # [n_active]
        nearest_local = dists_sq.argmin().item()
        nearest_global = active_idxs[nearest_local].item()

        self.rbf_field.update_valence(nearest_global, component, value)

    def evaluate_valence(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Return [batch, 4] valence vector from the RBF field (SD-014).

        Component order: [wanting, liking, harm_discriminative, surprise].
        Returns zeros when no centers are active or valence_enabled=False.

        Args:
            z_world: [batch, world_dim]

        Returns:
            valence: [batch, VALENCE_DIM]
        """
        if not getattr(self.config, "valence_enabled", True):
            return torch.zeros(z_world.shape[0], VALENCE_DIM,
                               device=z_world.device, dtype=z_world.dtype)
        return self.rbf_field.evaluate_valence(z_world)

    def get_valence_priority(
        self, z_world: torch.Tensor, drive_state: torch.Tensor, epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute drive-weighted replay priority (SD-014).

        priority(node) = dot(V_node, d_current) + epsilon
        where V_node = evaluate_valence(z_world) [batch, 4]
        and   d_current = drive_state [4] current drive weights.

        A high priority means the node is strongly relevant to current drive.
        All priorities are strictly positive (epsilon floor) so they can be
        used directly as sampling weights.

        Args:
            z_world:     [batch, world_dim] query points
            drive_state: [4] current drive weight vector
                         [w_drive, l_drive, h_drive, s_drive]
            epsilon:     small constant for numerical stability (default 1e-6)

        Returns:
            priority: [batch] scalar priority per query point
        """
        valence = self.evaluate_valence(z_world)           # [batch, 4]
        # drive_state may be on a different device; move to match
        d = drive_state.to(valence.device).to(valence.dtype)  # [4]
        priority = (valence * d.unsqueeze(0)).sum(dim=-1) + epsilon  # [batch]
        return priority

    def integrate(self, num_steps: int = 10) -> Dict[str, float]:
        """
        Offline integration of residue (contextualisation, no erasure).
        """
        if not self._harm_history:
            return {"integration_loss": 0.0, "steps": 0}

        harm_locations = torch.stack(self._harm_history[-100:])
        total_loss = 0.0

        for _ in range(num_steps):
            noise = torch.randn_like(harm_locations) * self.config.kernel_bandwidth
            sample_points = harm_locations + noise
            with torch.no_grad():
                targets = self.rbf_field(sample_points)
            predictions = self.neural_field(sample_points).squeeze(-1)
            loss = F.mse_loss(predictions, targets)
            total_loss += loss.item()

        return {
            "integration_loss": total_loss / num_steps,
            "steps": num_steps,
            "history_size": len(self._harm_history),
        }

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        return {
            "total_residue": self.total_residue,
            "num_harm_events": self.num_harm_events,
            "active_centers": self.rbf_field.active_mask.sum(),
            "mean_weight": (
                self.rbf_field.weights[self.rbf_field.active_mask].mean()
                if self.rbf_field.active_mask.any()
                else torch.tensor(0.0)
            ),
        }

    def visualize_field(
        self,
        z_range: Tuple[float, float] = (-3, 3),
        resolution: int = 50,
        slice_dims: Tuple[int, int] = (0, 1),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2D visualization over two z_world dimensions."""
        x = torch.linspace(z_range[0], z_range[1], resolution)
        y = torch.linspace(z_range[0], z_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        z = torch.zeros(resolution * resolution, self.config.world_dim)
        z[:, slice_dims[0]] = X.flatten()
        z[:, slice_dims[1]] = Y.flatten()
        with torch.no_grad():
            values = self.evaluate(z)
        return X, Y, values.reshape(resolution, resolution)

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z_world)
