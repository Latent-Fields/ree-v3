"""
CausalGridWorld V3 — Split Observation Channels (SD-005)

V3 change: observation is split into explicit body_state and world_state channels,
matching the SD-005 z_self/z_world split in the latent stack.

Observation structure (V3, use_proxy_fields=False — default):
  body_state:        position_local (2), health, energy, footprint_density (1),
                     heading (4 one-hot), episode_progress (1) — proprioceptive/interoceptive
                     → fed to z_self encoder
  world_state:       local_view (5×5×7 = 175), contamination_view (5×5 = 25) flattened
                     → fed to z_world encoder

  body_obs_dim  = 10
  world_obs_dim = 175 + 25 = 200

Observation structure (V3, use_proxy_fields=True — CausalGridWorldV2 mode):
  body_state adds: harm_exposure (1), benefit_exposure (1)
                   → nociceptive/hedonic EMA interoceptive channels
  world_state adds: hazard_field_view (5×5 = 25), resource_field_view (5×5 = 25)
                    → proximity gradient fields visible exteroceptively

  body_obs_dim  = 12
  world_obs_dim = 250

  Extra obs_dict keys (not in flat observation):
    harm_obs   [51]: SD-010 sensory-discriminative harm stream (Adelta-pathway analog).
                     hazard_field_view[25] + resource_field_view[25] + harm_exposure[1].
                     Immediate proximity; forward-predictable from action (HarmForwardModel).
    harm_obs_a [50]: SD-011 affective-motivational harm stream (C-fiber analog).
                     EMA of proximity fields at tau~20 steps. Accumulated homeostatic
                     deviation. NOT forward-predicted -- feeds E3 directly (ARC-033).

Proxy-gradient rationale (ARC-024 / INV-025-029):
  Harm and benefit signals are proxies along gradients pointing toward asymptotic
  limits (death / complete union) that are unreachable from within experience.
  A world that generates harm only at contact models the endpoint, not the gradient.
  CausalGridWorldV2 (use_proxy_fields=True) generates observable gradient fields
  that precede contact events, allowing E2.world_forward to learn action-conditional
  dynamics and E3.harm_eval to learn gradient-approach patterns.

Ground truth transition_type (for V3-EXQ-002, SD-003) is preserved.
New types (proxy mode only): "hazard_approach", "benefit_approach"

Sub-goal mode preserved unchanged from V2.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class CausalGridWorld:
    """
    2D grid world with persistent agent causal footprint — V3.

    V3 key change: _get_observation() returns a dict AND a flat tensor.
    The dict has:
      "body_state":   [body_obs_dim]  — proprioceptive channels → z_self
      "world_state":  [world_obs_dim] — exteroceptive channels  → z_world
      "contamination_view": [25]      — subset of world_state (for convenience)

    With use_proxy_fields=True (CausalGridWorldV2 mode):
      "hazard_field_view":   [25] — proximity gradient (subset of world_state)
      "resource_field_view": [25] — resource gradient (subset of world_state)

    Actions (unchanged from V2):
        0: up, 1: down, 2: left, 3: right, 4: stay

    Transition types (ground truth for SD-003 attribution):
        "agent_caused_hazard", "env_caused_hazard", "resource", "none"
        "waypoint", "sequence_complete" (subgoal_mode only)
        "hazard_approach", "benefit_approach" (use_proxy_fields=True only)
    """

    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0),
    }

    ENTITY_TYPES: Dict[str, int] = {
        "empty": 0, "wall": 1, "resource": 2, "hazard": 3,
        "contaminated": 4, "agent": 5, "waypoint": 6,
    }
    NUM_ENTITY_TYPES = 7

    def __init__(
        self,
        size: int = 10,
        num_hazards: int = 3,
        num_resources: int = 5,
        contamination_spread: float = 0.5,
        contamination_threshold: float = 2.0,
        env_drift_interval: int = 5,
        env_drift_prob: float = 0.3,
        hazard_harm: float = 0.5,
        contaminated_harm: float = 0.4,
        resource_benefit: float = 0.3,
        energy_decay: float = 0.01,
        seed: Optional[int] = None,
        # Sub-goal mode (preserved from V2 for MECH-057a)
        subgoal_mode: bool = False,
        num_waypoints: int = 3,
        waypoint_visit_reward: float = 0.2,
        waypoint_completion_reward: float = 0.8,
        sequence_commitment_timeout: int = 20,
        # Proxy-gradient field mode (ARC-024, CausalGridWorldV2)
        use_proxy_fields: bool = False,
        # SD-012: resource respawn for repeated drive-reduction cycles
        resource_respawn_on_consume: bool = False,
        hazard_field_decay: float = 0.5,
        resource_field_decay: float = 0.5,
        proximity_harm_scale: float = 0.05,
        proximity_benefit_scale: float = 0.03,
        nociception_ema_alpha: float = 0.1,
        harm_obs_a_ema_alpha: float = 0.05,
        proximity_approach_threshold: float = 0.15,
    ):
        self.size = size
        self.num_hazards = num_hazards
        self.num_resources = num_resources
        self.contamination_spread = contamination_spread
        self.contamination_threshold = contamination_threshold
        self.env_drift_interval = env_drift_interval
        self.env_drift_prob = env_drift_prob
        self.hazard_harm = hazard_harm
        self.contaminated_harm = contaminated_harm
        self.resource_benefit = resource_benefit
        self.energy_decay = energy_decay

        self.subgoal_mode = subgoal_mode
        self.num_waypoints = num_waypoints
        self.waypoint_visit_reward = waypoint_visit_reward
        self.waypoint_completion_reward = waypoint_completion_reward
        self.sequence_commitment_timeout = sequence_commitment_timeout

        # Proxy-gradient parameters
        self.use_proxy_fields = use_proxy_fields
        self.resource_respawn_on_consume = resource_respawn_on_consume
        self.hazard_field_decay = hazard_field_decay
        self.resource_field_decay = resource_field_decay
        self.proximity_harm_scale = proximity_harm_scale
        self.proximity_benefit_scale = proximity_benefit_scale
        self.nociception_ema_alpha = nociception_ema_alpha
        self.harm_obs_a_ema_alpha = harm_obs_a_ema_alpha
        self.proximity_approach_threshold = proximity_approach_threshold

        self._rng = np.random.default_rng(seed)
        # SD-011: harm_obs_a_ema persists across episodes (homeostatic accumulator).
        # Initialized here, NOT in reset(), so accumulated threat state carries over.
        # Resetting per-episode destroys autocorrelation (EXQ-106 C4 FAIL root cause).
        self.harm_obs_a_ema: np.ndarray = np.zeros(50, dtype=np.float32)
        self.reset()

    # ------------------------------------------------------------------ #
    # Dimension properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def body_obs_dim(self) -> int:
        """SD-005 body (proprioceptive/interoceptive) observation dimension."""
        # position_local (2) + health (1) + energy (1) + footprint_density (1)
        # + heading one-hot (4) + episode_progress (1) = 10
        # + harm_exposure (1) + benefit_exposure (1) = 12 (proxy mode)
        return 12 if self.use_proxy_fields else 10

    @property
    def world_obs_dim(self) -> int:
        """SD-005 world (exteroceptive) observation dimension."""
        local_view_dim = 5 * 5 * self.NUM_ENTITY_TYPES  # 175
        contamination_view_dim = 5 * 5                   # 25
        base = local_view_dim + contamination_view_dim   # 200
        if self.use_proxy_fields:
            hazard_field_dim = 5 * 5                     # 25
            resource_field_dim = 5 * 5                   # 25
            return base + hazard_field_dim + resource_field_dim  # 250
        return base                                      # 200

    @property
    def observation_dim(self) -> int:
        """Total flat observation dimension (body + world)."""
        return self.body_obs_dim + self.world_obs_dim

    @property
    def action_dim(self) -> int:
        return len(self.ACTIONS)

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset environment. Returns (flat_obs, obs_dict)."""
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.grid[0, :] = self.ENTITY_TYPES["wall"]
        self.grid[-1, :] = self.ENTITY_TYPES["wall"]
        self.grid[:, 0] = self.ENTITY_TYPES["wall"]
        self.grid[:, -1] = self.ENTITY_TYPES["wall"]

        self.contamination_grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.footprint_grid = np.zeros((self.size, self.size), dtype=np.int32)

        available = [
            (i, j)
            for i in range(1, self.size - 1)
            for j in range(1, self.size - 1)
        ]
        self._rng.shuffle(available)

        ax, ay = available.pop()
        self.agent_x = ax
        self.agent_y = ay
        self.agent_health = 1.0
        self.agent_energy = 1.0
        self.grid[ax, ay] = self.ENTITY_TYPES["agent"]
        self._last_action = 4  # stay

        self.hazards: List[List[int]] = []
        for _ in range(min(self.num_hazards, len(available))):
            hx, hy = available.pop()
            self.grid[hx, hy] = self.ENTITY_TYPES["hazard"]
            self.hazards.append([hx, hy])

        self.resources: List[List[int]] = []
        for _ in range(min(self.num_resources, len(available))):
            rx, ry = available.pop()
            self.grid[rx, ry] = self.ENTITY_TYPES["resource"]
            self.resources.append([rx, ry])

        self.waypoints: List[List[int]] = []
        self._next_waypoint_idx: int = 0
        self._sequence_in_progress: bool = False
        self._sequence_step: int = 0
        self._steps_since_waypoint: int = 0
        self._sequences_completed: int = 0

        if self.subgoal_mode:
            for _ in range(min(self.num_waypoints, len(available))):
                wx, wy = available.pop()
                self.grid[wx, wy] = self.ENTITY_TYPES["waypoint"]
                self.waypoints.append([wx, wy])

        self.steps = 0
        self.total_harm = 0.0
        self.total_benefit = 0.0

        # Proxy-gradient state
        self.harm_exposure: float = 0.0
        self.benefit_exposure: float = 0.0
        self.hazard_field = np.zeros((self.size, self.size), dtype=np.float32)
        self.resource_field = np.zeros((self.size, self.size), dtype=np.float32)
        # harm_obs_a_ema is NOT reset here -- it persists across episodes (see __init__).
        if self.use_proxy_fields:
            self._compute_proximity_fields()

        obs_dict = self._get_observation_dict()
        flat_obs = self._dict_to_flat(obs_dict)
        return flat_obs, obs_dict

    # ------------------------------------------------------------------ #
    # Step                                                                 #
    # ------------------------------------------------------------------ #

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, bool, Dict, Dict]:
        """
        Execute one step.

        Returns:
            flat_obs:        flat observation tensor [observation_dim]
            harm_signal:     float — negative = harm, positive = benefit
            done:            bool
            info:            dict with transition_type, contamination_delta, etc.
            obs_dict:        SD-005 split observation dict
        """
        if isinstance(action, torch.Tensor):
            action = action.argmax().item() if action.dim() > 0 else action.item()
        action = int(action) % len(self.ACTIONS)
        self._last_action = action

        dx, dy = self.ACTIONS[action]
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy

        harm_signal = 0.0
        transition_type = "none"
        contamination_delta = 0.0
        env_drift_occurred = False

        # Move agent if not wall
        if self.grid[new_x, new_y] != self.ENTITY_TYPES["wall"]:
            old_x, old_y = self.agent_x, self.agent_y

            if self.contamination_grid[old_x, old_y] >= self.contamination_threshold:
                self.grid[old_x, old_y] = self.ENTITY_TYPES["contaminated"]
            else:
                self.grid[old_x, old_y] = self.ENTITY_TYPES["empty"]

            target_type = self.grid[new_x, new_y]

            if target_type == self.ENTITY_TYPES["hazard"]:
                contact_harm = self.hazard_harm
                if self.use_proxy_fields:
                    # Proximity harm is additive to contact harm at the hazard cell
                    proximity_harm = self.proximity_harm_scale * float(
                        self.hazard_field[new_x, new_y]
                    )
                    harm_signal = -(contact_harm + proximity_harm)
                else:
                    harm_signal = -contact_harm
                self.agent_health = max(0.0, self.agent_health - abs(harm_signal))
                transition_type = "env_caused_hazard"
                self.total_harm += abs(harm_signal)

            elif target_type == self.ENTITY_TYPES["contaminated"]:
                harm_signal = -self.contaminated_harm
                self.agent_health = max(0.0, self.agent_health - self.contaminated_harm)
                transition_type = "agent_caused_hazard"
                self.total_harm += self.contaminated_harm

            elif target_type == self.ENTITY_TYPES["resource"]:
                contact_benefit = self.resource_benefit
                if self.use_proxy_fields:
                    proximity_benefit = self.proximity_benefit_scale * float(
                        self.resource_field[new_x, new_y]
                    )
                    harm_signal = contact_benefit + proximity_benefit
                else:
                    harm_signal = contact_benefit
                self.agent_health = min(1.0, self.agent_health + contact_benefit * 0.5)
                self.agent_energy = min(1.0, self.agent_energy + contact_benefit * 0.5)
                transition_type = "resource"
                self.total_benefit += harm_signal
                self.resources = [
                    r for r in self.resources if not (r[0] == new_x and r[1] == new_y)
                ]
                # SD-012: optional resource respawn for repeated drive-reduction cycles
                if self.resource_respawn_on_consume:
                    self._respawn_resource()
                # Resource consumed (or respawned) — recompute resource field
                if self.use_proxy_fields:
                    self._compute_proximity_fields()

            elif target_type == self.ENTITY_TYPES["waypoint"] and self.subgoal_mode:
                wp_idx = next(
                    (i for i, w in enumerate(self.waypoints)
                     if w[0] == new_x and w[1] == new_y),
                    None,
                )
                if wp_idx == self._next_waypoint_idx:
                    self._next_waypoint_idx += 1
                    self._sequence_step = wp_idx
                    self._steps_since_waypoint = 0
                    if not self._sequence_in_progress:
                        self._sequence_in_progress = True

                    if self._next_waypoint_idx >= len(self.waypoints):
                        # Sequence complete
                        harm_signal += self.waypoint_completion_reward
                        self.total_benefit += self.waypoint_completion_reward
                        transition_type = "sequence_complete"
                        self._sequences_completed += 1
                        self._sequence_in_progress = False
                        self._next_waypoint_idx = 0
                        self._respawn_waypoints()
                    else:
                        harm_signal += self.waypoint_visit_reward
                        self.total_benefit += self.waypoint_visit_reward
                        transition_type = "waypoint"

            elif self.use_proxy_fields and transition_type == "none":
                # Proxy-gradient approach transitions (only when no contact event)
                h_field_val = float(self.hazard_field[new_x, new_y])
                r_field_val = float(self.resource_field[new_x, new_y])
                if h_field_val >= self.proximity_approach_threshold:
                    harm_signal = -self.proximity_harm_scale * h_field_val
                    transition_type = "hazard_approach"
                    self.total_harm += abs(harm_signal)
                    self.agent_health = max(0.0, self.agent_health - abs(harm_signal))
                elif r_field_val >= self.proximity_approach_threshold:
                    harm_signal = self.proximity_benefit_scale * r_field_val
                    transition_type = "benefit_approach"
                    self.total_benefit += harm_signal

            # Move agent
            self.agent_x = new_x
            self.agent_y = new_y
            self.grid[new_x, new_y] = self.ENTITY_TYPES["agent"]

            # Update causal footprint
            self.footprint_grid[new_x, new_y] += 1
            old_cont = self.contamination_grid[new_x, new_y]
            self.contamination_grid[new_x, new_y] += self.contamination_spread
            contamination_delta = self.contamination_grid[new_x, new_y] - old_cont

        # Energy decay
        self.agent_energy = max(0.0, self.agent_energy - self.energy_decay)

        # Update interoceptive EMA channels
        if self.use_proxy_fields:
            alpha = self.nociception_ema_alpha
            if harm_signal < 0:
                self.harm_exposure = (1 - alpha) * self.harm_exposure + alpha * abs(harm_signal)
            else:
                self.harm_exposure = (1 - alpha) * self.harm_exposure
            if harm_signal > 0:
                self.benefit_exposure = (1 - alpha) * self.benefit_exposure + alpha * harm_signal
            else:
                self.benefit_exposure = (1 - alpha) * self.benefit_exposure
            # SD-011: update affective harm accumulator (C-fiber / paleospinothalamic analog).
            # EXQ-102 confirmed the prior spatial-window EMA had autocorr~0: as the agent
            # moves, the 5x5 local window content changes each step, destroying temporal
            # persistence. C-fiber / affective harm signal represents accumulated homeostatic
            # unpleasantness -- it follows the agent's trajectory, not the current grid view.
            # Fix (2026-03-28): EMA of the agent's current-cell hazard/resource scalar,
            # replicated uniformly across all 25 dims per channel. This gives autocorr ~
            # (1-alpha)^lag (e.g. lag=10 -> ~0.60 >> threshold 0.30). Interface stays 50-dim.
            alpha_a = self.harm_obs_a_ema_alpha
            ax2, ay2 = int(self.agent_x), int(self.agent_y)
            hazard_at_agent = float(np.clip(self.hazard_field[ax2, ay2], 0.0, 1.0))
            resource_at_agent = float(np.clip(self.resource_field[ax2, ay2], 0.0, 1.0))
            self.harm_obs_a_ema[:25] = (1.0 - alpha_a) * self.harm_obs_a_ema[:25] + alpha_a * hazard_at_agent
            self.harm_obs_a_ema[25:] = (1.0 - alpha_a) * self.harm_obs_a_ema[25:] + alpha_a * resource_at_agent

        # Env-caused drift
        if self.steps % self.env_drift_interval == 0 and self.steps > 0:
            self._drift_hazards()
            env_drift_occurred = True

        # Subgoal timeout
        if self.subgoal_mode and self._sequence_in_progress:
            self._steps_since_waypoint += 1
            if self._steps_since_waypoint > self.sequence_commitment_timeout:
                self._sequence_in_progress = False
                self._next_waypoint_idx = 0
                self._steps_since_waypoint = 0
                self._respawn_waypoints()

        self.steps += 1
        done = self.agent_health <= 0.0 or self.steps >= 500

        obs_dict = self._get_observation_dict()
        flat_obs = self._dict_to_flat(obs_dict)

        info = {
            "transition_type": transition_type,
            "contamination_delta": contamination_delta,
            "env_drift_occurred": env_drift_occurred,
            "footprint_at_cell": int(self.footprint_grid[self.agent_x, self.agent_y]),
            "health": self.agent_health,
            "energy": self.agent_energy,
            "steps": self.steps,
            "total_harm": self.total_harm,
            "total_benefit": self.total_benefit,
            "sequence_in_progress": self._sequence_in_progress,
            "sequence_step": self._sequence_step,
        }
        if self.use_proxy_fields:
            info["hazard_field_at_agent"] = float(
                self.hazard_field[self.agent_x, self.agent_y]
            )
            info["resource_field_at_agent"] = float(
                self.resource_field[self.agent_x, self.agent_y]
            )
            info["harm_exposure"] = self.harm_exposure
            info["benefit_exposure"] = self.benefit_exposure
        return flat_obs, harm_signal, done, info, obs_dict

    # ------------------------------------------------------------------ #
    # SD-005 Observation construction                                      #
    # ------------------------------------------------------------------ #

    def _get_observation_dict(self) -> Dict[str, torch.Tensor]:
        """
        Build the SD-005 split observation dict.

        Returns dict with keys:
          "body_state":   [body_obs_dim]  — proprioceptive/interoceptive
          "world_state":  [world_obs_dim] — exteroceptive
          "contamination_view": [25]      — convenience subset

        body_state channels (use_proxy_fields=False):
          [0]: agent_x / size  (normalised)
          [1]: agent_y / size  (normalised)
          [2]: agent_health
          [3]: agent_energy
          [4]: footprint_density at current cell
          [5-8]: last action one-hot (4 actions: up/down/left/right)
          [9]: steps / 500 (normalised episode progress)

        Additional body_state channels (use_proxy_fields=True):
          [10]: harm_exposure (nociceptive EMA)
          [11]: benefit_exposure (hedonic EMA)

        world_state channels (use_proxy_fields=False):
          [0:175]:   local_view (5×5×7 entity types, one-hot flattened)
          [175:200]: contamination_view (5×5 float, normalised)

        Additional world_state channels (use_proxy_fields=True):
          [200:225]: hazard_field_view (5×5, normalised by max field value)
          [225:250]: resource_field_view (5×5, normalised by max field value)
        """
        ax, ay = self.agent_x, self.agent_y

        # --- body_state ---
        body = torch.zeros(self.body_obs_dim)
        body[0] = ax / self.size
        body[1] = ay / self.size
        body[2] = self.agent_health
        body[3] = self.agent_energy
        max_vis = max(1, self.footprint_grid.max())
        body[4] = float(self.footprint_grid[ax, ay]) / max_vis
        action_enc = self._last_action if self._last_action < 4 else 0
        body[5 + action_enc] = 1.0  # one-hot last action (indices 5,6,7,8)
        body[9] = min(1.0, self.steps / 500.0)
        if self.use_proxy_fields:
            body[10] = float(np.clip(self.harm_exposure, 0.0, 1.0))
            body[11] = float(np.clip(self.benefit_exposure, 0.0, 1.0))

        # --- local_view (5×5×7) → world_state part 1 ---
        local_view = torch.zeros(5, 5, self.NUM_ENTITY_TYPES)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = ax + di, ay + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    etype = self.grid[ni, nj]
                else:
                    etype = self.ENTITY_TYPES["wall"]
                local_view[di + 2, dj + 2, etype] = 1.0
        local_view_flat = local_view.reshape(-1)  # [175]

        # --- contamination_view (5×5) → world_state part 2 ---
        cont_view = torch.zeros(5, 5)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = ax + di, ay + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    cont_view[di + 2, dj + 2] = float(self.contamination_grid[ni, nj])
        cont_view_flat = (cont_view / (self.contamination_threshold + 1e-6)).reshape(-1)  # [25]

        world_parts = [local_view_flat, cont_view_flat]

        # --- proxy-gradient field views (use_proxy_fields=True only) ---
        hazard_field_flat = torch.zeros(25)
        resource_field_flat = torch.zeros(25)
        if self.use_proxy_fields:
            hazard_max = float(self.hazard_field.max()) + 1e-6
            resource_max = float(self.resource_field.max()) + 1e-6
            h_view = torch.zeros(5, 5)
            r_view = torch.zeros(5, 5)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = ax + di, ay + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        h_view[di + 2, dj + 2] = float(self.hazard_field[ni, nj]) / hazard_max
                        r_view[di + 2, dj + 2] = float(self.resource_field[ni, nj]) / resource_max
            hazard_field_flat = h_view.reshape(-1)    # [25]
            resource_field_flat = r_view.reshape(-1)  # [25]
            world_parts.extend([hazard_field_flat, resource_field_flat])

        world_state = torch.cat(world_parts)

        result = {
            "body_state": body.float(),
            "world_state": world_state.float(),
            "contamination_view": cont_view_flat.float(),
        }
        if self.use_proxy_fields:
            result["hazard_field_view"] = hazard_field_flat.float()
            result["resource_field_view"] = resource_field_flat.float()
            # SD-010: dedicated harm_obs for HarmEncoder (nociceptive separation).
            # Sensory-discriminative stream (z_harm_s, Adelta-pathway analog):
            # Layout: hazard_field_view[25] + resource_field_view[25] + harm_exposure[1]
            result["harm_obs"] = torch.cat([
                hazard_field_flat.float(),
                resource_field_flat.float(),
                torch.tensor([float(np.clip(self.harm_exposure, 0.0, 1.0))]),
            ], dim=0)  # [51]
            # SD-011: harm_obs_a for AffectiveHarmEncoder (affective-motivational stream,
            # C-fiber/paleospinothalamic analog). EMA of proximity fields at slower tau
            # (~20 steps vs ~10 for harm_exposure). Represents accumulated homeostatic
            # threat state, not immediate proximity. Does NOT need a forward model.
            result["harm_obs_a"] = torch.from_numpy(self.harm_obs_a_ema.copy()).float()  # [50]
        return result

    def _dict_to_flat(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate body_state + world_state into flat observation vector."""
        return torch.cat([obs_dict["body_state"], obs_dict["world_state"]]).float()

    # V2 backward-compat method
    def _get_observation(self) -> torch.Tensor:
        return self._dict_to_flat(self._get_observation_dict())

    # ------------------------------------------------------------------ #
    # Proxy-gradient field computation (ARC-024)                          #
    # ------------------------------------------------------------------ #

    def _compute_proximity_fields(self) -> None:
        """
        Compute hazard and resource proximity fields across the full grid.

        Field value at (i,j): sum over all sources of 1 / (1 + dist * decay).
        Uses Manhattan distance. Peaks at 1.0 at source cell (dist=0).

        Called after placement, after drift (hazards), after consumption (resources).
        """
        self.hazard_field = np.zeros((self.size, self.size), dtype=np.float32)
        for hx, hy in self.hazards:
            for i in range(self.size):
                for j in range(self.size):
                    dist = abs(i - hx) + abs(j - hy)
                    self.hazard_field[i, j] += 1.0 / (1.0 + dist * self.hazard_field_decay)

        self.resource_field = np.zeros((self.size, self.size), dtype=np.float32)
        for rx, ry in self.resources:
            for i in range(self.size):
                for j in range(self.size):
                    dist = abs(i - rx) + abs(j - ry)
                    self.resource_field[i, j] += 1.0 / (1.0 + dist * self.resource_field_decay)

    # ------------------------------------------------------------------ #
    # Internal helpers (unchanged from V2)                                #
    # ------------------------------------------------------------------ #

    def _drift_hazards(self) -> None:
        """Drift environment-caused hazards randomly."""
        available_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        drifted = False
        for hazard in self.hazards:
            if self._rng.random() < self.env_drift_prob:
                self._rng.shuffle(available_dirs)
                for dx, dy in available_dirs:
                    nx, ny = hazard[0] + dx, hazard[1] + dy
                    if (0 < nx < self.size - 1 and 0 < ny < self.size - 1 and
                            self.grid[nx, ny] == self.ENTITY_TYPES["empty"]):
                        self.grid[hazard[0], hazard[1]] = self.ENTITY_TYPES["empty"]
                        hazard[0], hazard[1] = nx, ny
                        self.grid[nx, ny] = self.ENTITY_TYPES["hazard"]
                        drifted = True
                        break
        # Recompute hazard field after any drift
        if self.use_proxy_fields and drifted:
            self._compute_proximity_fields()

    def _respawn_resource(self) -> None:
        """SD-012: Spawn one new resource at a random empty cell after consumption."""
        available = [
            (i, j)
            for i in range(1, self.size - 1)
            for j in range(1, self.size - 1)
            if self.grid[i, j] == self.ENTITY_TYPES["empty"]
        ]
        if not available:
            return
        self._rng.shuffle(available)
        rx, ry = available[0]
        self.grid[rx, ry] = self.ENTITY_TYPES["resource"]
        self.resources.append([rx, ry])

    def _respawn_waypoints(self) -> None:
        """Respawn waypoints after sequence completion or timeout."""
        for wp in self.waypoints:
            if self.grid[wp[0], wp[1]] == self.ENTITY_TYPES["waypoint"]:
                self.grid[wp[0], wp[1]] = self.ENTITY_TYPES["empty"]

        available = [
            (i, j)
            for i in range(1, self.size - 1)
            for j in range(1, self.size - 1)
            if self.grid[i, j] == self.ENTITY_TYPES["empty"]
        ]
        self._rng.shuffle(available)
        self.waypoints = []
        for _ in range(min(self.num_waypoints, len(available))):
            wx, wy = available.pop()
            self.grid[wx, wy] = self.ENTITY_TYPES["waypoint"]
            self.waypoints.append([wx, wy])

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def get_subgoal_state(self) -> dict:
        return {
            "sequence_in_progress": self._sequence_in_progress,
            "sequence_step": self._sequence_step,
            "next_waypoint_idx": self._next_waypoint_idx,
            "sequences_completed": self._sequences_completed,
        }

    def get_contamination_map(self) -> np.ndarray:
        return self.contamination_grid.copy()

    def get_footprint_map(self) -> np.ndarray:
        return self.footprint_grid.copy()

    def get_agent_position(self) -> Tuple[int, int]:
        return (self.agent_x, self.agent_y)

    def get_hazard_field(self) -> np.ndarray:
        """Return hazard proximity field (proxy mode only)."""
        return self.hazard_field.copy()

    def get_resource_field(self) -> np.ndarray:
        """Return resource proximity field (proxy mode only)."""
        return self.resource_field.copy()

    def render(self, mode: str = "text") -> Optional[str]:
        if mode != "text":
            return None
        symbols = {
            self.ENTITY_TYPES["empty"]: ".",
            self.ENTITY_TYPES["wall"]: "#",
            self.ENTITY_TYPES["resource"]: "R",
            self.ENTITY_TYPES["hazard"]: "X",
            self.ENTITY_TYPES["contaminated"]: "c",
            self.ENTITY_TYPES["agent"]: "A",
            self.ENTITY_TYPES["waypoint"]: "W",
        }
        lines = []
        for i in range(self.size):
            row = "".join(symbols.get(self.grid[i, j], "?") for j in range(self.size))
            lines.append(row)
        lines.append(
            f"\nHealth: {self.agent_health:.2f} | Energy: {self.agent_energy:.2f} | "
            f"Steps: {self.steps}"
        )
        lines.append(
            f"Harm: {self.total_harm:.2f} | Benefit: {self.total_benefit:.2f} | "
            f"Max contamination: {self.contamination_grid.max():.2f}"
        )
        if self.use_proxy_fields:
            lines.append(
                f"Harm exposure: {self.harm_exposure:.3f} | "
                f"Benefit exposure: {self.benefit_exposure:.3f} | "
                f"Hazard field @ agent: {self.hazard_field[self.agent_x, self.agent_y]:.3f}"
            )
        return "\n".join(lines)


# Convenience alias for CausalGridWorldV2 mode
def CausalGridWorldV2(**kwargs) -> CausalGridWorld:
    """
    CausalGridWorldV2: CausalGridWorld with proxy-gradient fields enabled.

    Implements ARC-024 proxy-gradient structure:
    - Hazard proximity field generates continuous harm signal before contact
    - Resource proximity field generates continuous benefit signal before collection
    - Interoceptive EMA channels (harm_exposure, benefit_exposure) in body_state
    - Field views (hazard_field_view, resource_field_view) in world_state
    - New transition types: "hazard_approach", "benefit_approach"

    body_obs_dim = 12, world_obs_dim = 250
    """
    kwargs.setdefault("use_proxy_fields", True)
    return CausalGridWorld(**kwargs)
