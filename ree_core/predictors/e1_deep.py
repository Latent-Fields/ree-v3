"""
E1 Deep Predictor — V3 Implementation

Unchanged in function from V2: slow world model, LSTM-based, trains on
sensory prediction error.

V3 change (SD-005): E1 produces predictions over BOTH z_self and z_world
channels (associative prior). In V2, E1 predicted over unified z_gamma.
In V3, E1 reads the concatenated [z_self, z_world] vector and predicts
the next [z_self, z_world] jointly.

Module contract:
  - Input:  z_self + z_world (concatenated [batch, self_dim + world_dim])
  - Output: predicted (z_self_next, z_world_next) + prior for HippocampalModule
  - E1 is READ-ONLY with respect to z_self and z_world:
    it contributes predictions (associative prior) but does not drive
    the primary write path for either channel.

Episode-boundary semantics (preserved from V2):
E1 maintains self._hidden_state across episode steps. Reset at episode
start via reset_hidden_state(). The prediction-loss computation in REEAgent
uses a save/restore pattern so training replays do not disturb inference.
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E1Config
from ree_core.latent.stack import LatentState


class ContextMemory(nn.Module):
    """Context memory for E1's long-horizon predictions (unchanged from V2)."""

    def __init__(self, latent_dim: int, memory_dim: int = 128, num_slots: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_dim = memory_dim
        self.num_slots = num_slots

        self.memory = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.01)
        self.query_proj = nn.Linear(latent_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        self.output_proj = nn.Linear(memory_dim, latent_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(latent_dim, memory_dim),
            nn.Sigmoid()
        )

    def read(self, query: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(memory)
        v = self.value_proj(memory)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.memory_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, v).squeeze(1)
        return self.output_proj(context)

    def write(self, state: torch.Tensor) -> None:
        write_signal = self.write_gate(state)
        with torch.no_grad():
            query = self.query_proj(state)
            scores = torch.mm(query, self.memory.t())
            min_idx = scores.mean(0).argmin()
            self.memory.data[min_idx] = (
                0.9 * self.memory.data[min_idx] + 0.1 * write_signal.mean(0)
            )


class E1DeepPredictor(nn.Module):
    """
    E1 Deep Predictor — V3.

    Long-horizon world model, LSTM-based. Trains on sensory prediction error
    over [z_self, z_world] concatenated latent.

    Provides associative prior into HippocampalModule (SD-002, preserved from V2).
    Prior is now over world_dim (for HippocampalModule terrain conditioning).
    """

    def __init__(self, config: Optional[E1Config] = None):
        super().__init__()
        self.config = config or E1Config()

        # V3: total latent dim = self_dim + world_dim
        total_dim = self.config.self_dim + self.config.world_dim

        self.context_memory = ContextMemory(
            latent_dim=total_dim,
            memory_dim=self.config.hidden_dim,
            num_slots=16,
        )

        self.transition_rnn = nn.LSTM(
            input_size=total_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=0.1 if self.config.num_layers > 1 else 0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, total_dim),
        )

        # Prior generator: for HippocampalModule (SD-002).
        # In V3 the prior is over world_dim (terrain conditioning).
        self.prior_generator = nn.Sequential(
            nn.Linear(total_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.world_dim),
        )

        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # MECH-116: optional goal conditioning projection
        # Projects [z_self, z_world, z_goal] -> latent_dim before LSTM
        goal_dim = getattr(config, 'goal_dim', 0)
        self._goal_dim = goal_dim
        if goal_dim > 0:
            self.goal_input_proj = nn.Linear(
                config.latent_dim + goal_dim, config.latent_dim
            )
        else:
            self.goal_input_proj = None

    def reset_hidden_state(self) -> None:
        """Reset hidden state for a new episode."""
        self._hidden_state = None

    @property
    def total_latent_dim(self) -> int:
        return self.config.self_dim + self.config.world_dim

    def predict_long_horizon(
        self,
        current_state: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict latent states over a long horizon.

        Args:
            current_state: Concatenated [z_self, z_world] [batch, self_dim+world_dim]
            horizon: Number of steps to predict

        Returns:
            Predicted states [batch, horizon, self_dim+world_dim]
        """
        horizon = horizon or self.config.prediction_horizon
        batch_size = current_state.shape[0]
        device = current_state.device

        context = self.context_memory.read(current_state)
        combined = torch.cat([current_state, context], dim=-1)
        prior = self.prior_generator(combined)

        # prior is world_dim; expand back to total_dim for LSTM input
        # by concatenating with zeros for z_self part
        prior_self = torch.zeros(batch_size, self.config.self_dim, device=device)
        prior_full = torch.cat([prior_self, prior], dim=-1)  # [batch, total_dim]

        if self._hidden_state is None or self._hidden_state[0].shape[1] != batch_size:
            h0 = torch.zeros(
                self.config.num_layers, batch_size, self.config.hidden_dim, device=device
            )
            c0 = torch.zeros(
                self.config.num_layers, batch_size, self.config.hidden_dim, device=device
            )
            self._hidden_state = (h0, c0)

        predictions = []
        hidden = self._hidden_state
        input_state = prior_full.unsqueeze(1)  # [batch, 1, total_dim]

        for _ in range(horizon):
            output, hidden = self.transition_rnn(input_state, hidden)
            predicted = self.output_proj(output.squeeze(1))
            predictions.append(predicted)
            input_state = predicted.unsqueeze(1)

        self._hidden_state = (hidden[0].detach(), hidden[1].detach())
        return torch.stack(predictions, dim=1)  # [batch, horizon, total_dim]

    def generate_prior(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Generate world-domain prior for HippocampalModule conditioning (SD-002).

        Args:
            current_state: Concatenated [z_self, z_world] [batch, total_dim]

        Returns:
            Prior over z_world [batch, world_dim]
        """
        context = self.context_memory.read(current_state)
        combined = torch.cat([current_state, context], dim=-1)
        return self.prior_generator(combined)

    def split_prediction(
        self, prediction: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a prediction tensor into (z_self_pred, z_world_pred)."""
        z_self_pred  = prediction[:, :self.config.self_dim]
        z_world_pred = prediction[:, self.config.self_dim:]
        return z_self_pred, z_world_pred

    def update_from_observation(
        self,
        observation_state: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.context_memory.write(observation_state)
        return {
            "e1_error_magnitude": prediction_error.pow(2).mean(),
            "context_updated": torch.tensor(1.0),
        }

    def integrate_experience(
        self,
        experience_buffer: List[torch.Tensor],
        num_iterations: int = 10,
    ) -> Dict[str, float]:
        if len(experience_buffer) < 2:
            return {"integration_loss": 0.0}

        total_loss = 0.0
        for _ in range(num_iterations):
            self.reset_hidden_state()
            start_idx = int(torch.randint(0, len(experience_buffer) - 1, (1,)).item())
            end_idx = min(start_idx + self.config.prediction_horizon, len(experience_buffer))
            sequence = torch.stack(experience_buffer[start_idx:end_idx])
            if sequence.dim() == 2:
                sequence = sequence.unsqueeze(0)
            initial = sequence[:, 0, :]
            predictions = self.predict_long_horizon(initial, horizon=sequence.shape[1] - 1)
            targets = sequence[:, 1:, :]
            loss = F.mse_loss(predictions[:, :targets.shape[1], :], targets)
            total_loss += loss.item()

        return {"integration_loss": total_loss / num_iterations}

    def forward(
        self,
        current_state: torch.Tensor,
        horizon: Optional[int] = None,
        z_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            current_state: [z_self, z_world] concatenated [batch, total_dim]
            z_goal:        optional goal latent [1, goal_dim] for MECH-116 conditioning

        Returns:
            predictions: [batch, horizon, total_dim]
            prior:       world-domain prior [batch, world_dim] for HippocampalModule
        """
        # MECH-116: apply goal conditioning if provided
        total_state = current_state
        if self.goal_input_proj is not None and z_goal is not None:
            goal_exp = z_goal.expand(total_state.shape[0], -1)
            total_state = self.goal_input_proj(
                torch.cat([total_state, goal_exp], dim=-1)
            )
        predictions = self.predict_long_horizon(total_state, horizon)
        prior = self.generate_prior(current_state)
        return predictions, prior
