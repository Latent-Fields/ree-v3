"""Post-603i trainable relief/safety escape-affordance learner.

This module is a feature-flagged successor substrate for the SD-059 arithmetic
EscapeAffordanceBridge. It does not replace the active V3-EXQ-603i validation
path and is OFF by default at the agent/config layer.

The enabled path contains actual PyTorch relief and safety heads. Inputs are
compact detached latent/context features plus an action embedding, so local
optimizer steps do not backpropagate into E1/E2/E3 encoders by default.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainableEscapeAffordanceLearnerConfig:
    """Configuration for the post-603i trainable learner.

    `enabled` is a module-local guard used by tests and direct construction.
    REEConfig's `use_trainable_escape_affordance_learner` is the agent-level
    master switch and defaults False, so existing runs never instantiate this
    learner unless an experiment opts in.
    """

    enabled: bool = True
    n_action_classes: int = 5
    use_relief_critic: bool = True
    use_safety_predictor: bool = True
    bias_scale: float = 0.1
    relief_learn_rate: float = 0.1
    safety_learn_rate: float = 0.1
    optimizer_lr: float = 0.03
    leak_rate: float = 0.01
    relief_reward_floor: float = 1e-4
    relief_target_scale: float = 0.3
    threat_floor: float = 0.1
    noop_class: int = 0
    hidden_dim: int = 32
    action_embedding_dim: int = 8
    prediction_floor: float = 0.02
    max_grad_norm: float = 5.0


@dataclass
class TrainableEscapeAffordanceLearnerOutput:
    """Readout from the most recent update or bias computation."""

    threat_scale: float = 0.0
    relief_target: float = 0.0
    safety_target: float = 0.0
    relief_value: float = 0.0
    safety_value: float = 0.0
    combined_value: float = 0.0
    relief_loss: float = 0.0
    safety_loss: float = 0.0
    bias_max_abs: float = 0.0
    updated: bool = False
    optimizer_step: bool = False
    simulation_skipped: bool = False


class _EscapeAffordanceHeads(nn.Module):
    """Small shared trunk with action embedding and bounded heads."""

    def __init__(
        self,
        state_dim: int,
        n_action_classes: int,
        hidden_dim: int,
        action_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = max(1, int(state_dim))
        self.n_action_classes = max(1, int(n_action_classes))
        hidden = max(4, int(hidden_dim))
        emb_dim = max(1, int(action_embedding_dim))

        self.action_embedding = nn.Embedding(self.n_action_classes, emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim + emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.relief_head = nn.Linear(hidden, 1)
        self.safety_head = nn.Linear(hidden, 1)
        self._init_low_prior()

    def _init_low_prior(self) -> None:
        # Keep an untrained enabled learner from emitting arbitrary escape bias.
        with torch.no_grad():
            self.relief_head.bias.fill_(-4.0)
            self.safety_head.bias.fill_(-4.0)

    def forward(
        self,
        state_vector: torch.Tensor,
        action_class: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)
        action_class = action_class.to(dtype=torch.long, device=state_vector.device)
        action_class = action_class.clamp(min=0, max=self.n_action_classes - 1)
        if action_class.dim() == 0:
            action_class = action_class.unsqueeze(0)
        action_class = action_class.flatten()
        emb = self.action_embedding(action_class)
        x = torch.cat([state_vector, emb], dim=-1)
        h = self.trunk(x)
        return torch.sigmoid(self.relief_head(h)), torch.sigmoid(self.safety_head(h))


class TrainableEscapeAffordanceLearner:
    """Trainable relief critic + safety predictor.

    Representation:
      - shared trunk over detached compact state features plus action embedding;
      - Q_relief(state, action, threat_context), sigmoid bounded to [0, 1];
      - P_safety(state, context, action), sigmoid bounded to [0, 1].

    The model is lazily initialized from the first trainable transition so direct
    disabled construction and untrained bias checks instantiate no neural state.
    Learned model weights persist across episode reset; reset clears only the
    one-tick traces used for temporal credit assignment.
    """

    def __init__(
        self,
        config: Optional[TrainableEscapeAffordanceLearnerConfig] = None,
    ) -> None:
        self.config = config or TrainableEscapeAffordanceLearnerConfig()
        k = max(1, int(self.config.n_action_classes))

        self._model: Optional[_EscapeAffordanceHeads] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._state_dim: Optional[int] = None

        self._relief_seen: List[bool] = [False] * k
        self._safety_seen: List[bool] = [False] * k

        self._prev_z_harm_a_norm: Optional[float] = None
        self._prev_threat_scale: float = 0.0
        self._prev_state_vector: Optional[torch.Tensor] = None
        self._last_state_vector: Optional[torch.Tensor] = None

        self._n_updates: int = 0
        self._n_optimizer_steps: int = 0
        self._n_relief_positive: int = 0
        self._n_relief_negative: int = 0
        self._n_safety_positive: int = 0
        self._n_safety_negative: int = 0
        self._n_weight_decay_steps: int = 0
        self._n_bias_fires: int = 0
        self._n_noop_skipped: int = 0
        self._n_sim_skipped: int = 0

        self._max_relief_prediction: float = 0.0
        self._max_safety_prediction: float = 0.0
        self._last_output = TrainableEscapeAffordanceLearnerOutput()

    # -- State management --

    def reset(self) -> None:
        """Clear within-episode traces while preserving learned head weights."""
        self._prev_z_harm_a_norm = None
        self._prev_threat_scale = 0.0
        self._prev_state_vector = None

    # -- Feature hooks --

    @staticmethod
    def _flatten_optional(
        value: Optional[Union[torch.Tensor, Sequence[float], float]]
    ) -> torch.Tensor:
        if value is None:
            return torch.empty(0, dtype=torch.float32)
        if isinstance(value, torch.Tensor):
            return value.detach().flatten().to(dtype=torch.float32, device="cpu")
        return torch.as_tensor(value, dtype=torch.float32).flatten()

    def build_state_vector(
        self,
        z_world: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        z_self: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        z_harm_a: Optional[Union[torch.Tensor, Sequence[float], float]] = None,
        z_harm_a_norm: Optional[float] = None,
        threat_scale: Optional[float] = None,
        action_class: Optional[int] = None,
        recent_outcome_features: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    ) -> torch.Tensor:
        """Build compact detached state/context features.

        The action is represented by the model's action embedding during
        prediction/training. The `action_class` argument is accepted for
        compatibility with earlier callers but is not appended to the cached
        state vector, preventing stale previous-action leakage across ticks.
        """
        del action_class
        z_norm = (
            float(z_harm_a_norm)
            if z_harm_a_norm is not None
            else float(self._flatten_optional(z_harm_a).norm().item())
        )
        parts = [
            self._flatten_optional(z_world),
            self._flatten_optional(z_self),
            self._flatten_optional(z_harm_a),
            self._flatten_optional(recent_outcome_features),
            torch.tensor(
                [
                    z_norm,
                    float(threat_scale) if threat_scale is not None else 0.0,
                ],
                dtype=torch.float32,
            ),
        ]
        non_empty = [p for p in parts if p.numel() > 0]
        if not non_empty:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(non_empty).detach().clone()

    def _coerce_state_vector(self, state_vector: torch.Tensor) -> torch.Tensor:
        s = state_vector.detach().flatten().to(dtype=torch.float32, device="cpu")
        if s.numel() == 0:
            s = torch.zeros(1, dtype=torch.float32)
        if self._state_dim is None:
            return s
        if s.numel() == self._state_dim:
            return s
        if s.numel() > self._state_dim:
            return s[: self._state_dim].clone()
        out = torch.zeros(self._state_dim, dtype=torch.float32)
        out[: s.numel()] = s
        return out

    # -- Prediction helpers --

    def threat_scale(
        self,
        z_harm_a_norm: float,
        threat_scale: Optional[float] = None,
    ) -> float:
        """Return caller-supplied threat scale or a conservative scalar ramp."""
        if threat_scale is not None:
            return float(max(0.0, min(1.0, threat_scale)))
        z = float(z_harm_a_norm)
        floor = float(self.config.threat_floor)
        if z <= floor:
            return 0.0
        denom = max(floor, 1e-6)
        return float(max(0.0, min(1.0, (z - floor) / denom)))

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _valid_action(self, action_class: int) -> bool:
        return 0 <= int(action_class) < max(1, int(self.config.n_action_classes))

    def _ensure_model(self, state_vector: torch.Tensor) -> Optional[_EscapeAffordanceHeads]:
        if not self.config.enabled:
            return None
        if self._model is not None:
            return self._model
        s = state_vector.detach().flatten().to(dtype=torch.float32, device="cpu")
        self._state_dim = max(1, int(s.numel()))
        self._model = _EscapeAffordanceHeads(
            state_dim=self._state_dim,
            n_action_classes=max(1, int(self.config.n_action_classes)),
            hidden_dim=int(self.config.hidden_dim),
            action_embedding_dim=int(self.config.action_embedding_dim),
        )
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(max(0.0, self.config.optimizer_lr)),
            weight_decay=float(max(0.0, self.config.leak_rate)),
        )
        return self._model

    def _raw_predict(
        self,
        action_class: int,
        state_vector: Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        if (
            not self.config.enabled
            or self._model is None
            or state_vector is None
            or not self._valid_action(int(action_class))
        ):
            return 0.0, 0.0
        s = self._coerce_state_vector(state_vector).unsqueeze(0)
        a = torch.tensor([int(action_class)], dtype=torch.long)
        with torch.no_grad():
            relief, safety = self._model(s, a)
        return float(relief.item()), float(safety.item())

    def _masked_prediction(self, raw: float, seen: bool) -> float:
        if seen or raw >= float(max(0.0, self.config.prediction_floor)):
            return self._clamp01(raw)
        return 0.0

    def predict_relief(
        self,
        action_class: int,
        state_vector: Optional[torch.Tensor] = None,
    ) -> float:
        if not self.config.enabled or not self.config.use_relief_critic:
            return 0.0
        state = state_vector if state_vector is not None else self._last_state_vector
        raw, _ = self._raw_predict(int(action_class), state)
        seen = self._relief_seen[int(action_class)] if self._valid_action(action_class) else False
        return self._masked_prediction(raw, seen)

    def predict_safety(
        self,
        action_class: int,
        state_vector: Optional[torch.Tensor] = None,
    ) -> float:
        if not self.config.enabled or not self.config.use_safety_predictor:
            return 0.0
        state = state_vector if state_vector is not None else self._last_state_vector
        _, raw = self._raw_predict(int(action_class), state)
        seen = self._safety_seen[int(action_class)] if self._valid_action(action_class) else False
        return self._masked_prediction(raw, seen)

    def _train_heads(
        self,
        state_vector: torch.Tensor,
        action_class: int,
        relief_target: Optional[float],
        safety_target: Optional[float],
    ) -> Tuple[float, float, bool]:
        model = self._ensure_model(state_vector)
        if model is None or self._optimizer is None:
            return 0.0, 0.0, False

        s = self._coerce_state_vector(state_vector).unsqueeze(0)
        a = torch.tensor([int(action_class)], dtype=torch.long)
        relief_pred, safety_pred = model(s, a)
        losses: List[torch.Tensor] = []
        relief_loss_value = 0.0
        safety_loss_value = 0.0

        if relief_target is not None and self.config.use_relief_critic:
            target = torch.tensor(
                [[self._clamp01(float(relief_target))]], dtype=torch.float32
            )
            relief_loss = F.binary_cross_entropy(relief_pred, target)
            relief_loss_value = float(relief_loss.detach().item())
            losses.append(relief_loss * float(max(0.0, self.config.relief_learn_rate)))

        if safety_target is not None and self.config.use_safety_predictor:
            target = torch.tensor(
                [[self._clamp01(float(safety_target))]], dtype=torch.float32
            )
            safety_loss = F.binary_cross_entropy(safety_pred, target)
            safety_loss_value = float(safety_loss.detach().item())
            losses.append(safety_loss * float(max(0.0, self.config.safety_learn_rate)))

        if not losses:
            return relief_loss_value, safety_loss_value, False

        self._optimizer.zero_grad(set_to_none=True)
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        max_norm = float(max(0.0, self.config.max_grad_norm))
        if max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        self._optimizer.step()
        self._n_optimizer_steps += 1
        if float(max(0.0, self.config.leak_rate)) > 0.0:
            self._n_weight_decay_steps += 1
        return relief_loss_value, safety_loss_value, True

    def _update_prediction_maxima(self, state_vector: Optional[torch.Tensor]) -> None:
        if self._model is None or state_vector is None:
            return
        relief_vals = []
        safety_vals = []
        for c in range(max(1, int(self.config.n_action_classes))):
            relief_vals.append(self.predict_relief(c, state_vector))
            safety_vals.append(self.predict_safety(c, state_vector))
        self._max_relief_prediction = max(self._max_relief_prediction, max(relief_vals))
        self._max_safety_prediction = max(self._max_safety_prediction, max(safety_vals))

    # -- Learning update --

    def update(
        self,
        z_harm_a_norm: float,
        last_action_class: Optional[int],
        z_world: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        z_self: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        z_harm_a: Optional[Union[torch.Tensor, Sequence[float], float]] = None,
        threat_scale: Optional[float] = None,
        last_action_directed: bool = True,
        recent_outcome_features: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        simulation_mode: bool = False,
        hypothesis_tag: bool = False,
    ) -> TrainableEscapeAffordanceLearnerOutput:
        """Advance relief/safety predictions by one waking tick.

        Relief credit requires previous threat, a directed non-noop action, and
        a harm drop above `relief_reward_floor`. The target is continuous:
        clamp((prev_z_harm_a_norm - current_z_harm_a_norm) / target_scale, 0, 1).

        Safety credit requires previous threat, a directed non-noop action, and
        subsequent threat absence. It is therefore response-produced safety, not
        generic low harm. Failed relief and threat recurrence train extinction
        targets toward zero.
        """
        if (not self.config.enabled) or simulation_mode or hypothesis_tag:
            if simulation_mode or hypothesis_tag:
                self._n_sim_skipped += 1
            self._last_output = TrainableEscapeAffordanceLearnerOutput(
                simulation_skipped=bool(simulation_mode or hypothesis_tag)
            )
            return self._last_output

        z_now = float(z_harm_a_norm)
        ts_now = self.threat_scale(z_now, threat_scale=threat_scale)
        state_vec = self.build_state_vector(
            z_world=z_world,
            z_self=z_self,
            z_harm_a=z_harm_a if z_harm_a is not None else [z_now],
            z_harm_a_norm=z_now,
            threat_scale=ts_now,
            recent_outcome_features=recent_outcome_features,
        )

        self._n_updates += 1

        relief_target = 0.0
        safety_target = 0.0
        relief_target_for_loss: Optional[float] = None
        safety_target_for_loss: Optional[float] = None
        relief_loss = 0.0
        safety_loss = 0.0
        optimizer_step = False
        relief_value = 0.0
        safety_value = 0.0

        prev_z = self._prev_z_harm_a_norm
        prev_under_threat = bool(prev_z is not None and self._prev_threat_scale > 0.0)
        action_ok = (
            last_action_class is not None
            and bool(last_action_directed)
            and self._valid_action(int(last_action_class))
            and int(last_action_class) != int(self.config.noop_class)
        )

        if prev_under_threat and last_action_class is not None and not action_ok:
            self._n_noop_skipped += 1

        if prev_under_threat and action_ok and self._prev_state_vector is not None:
            a = int(last_action_class)
            prev_state = self._prev_state_vector
            relief_pred_prev = self.predict_relief(a, prev_state)
            safety_pred_prev = self.predict_safety(a, prev_state)

            if self.config.use_relief_critic:
                harm_drop = float(prev_z) - z_now
                if harm_drop > float(self.config.relief_reward_floor):
                    scale = max(float(self.config.relief_target_scale), 1e-6)
                    relief_target = self._clamp01(harm_drop / scale)
                    relief_target_for_loss = relief_target
                    self._relief_seen[a] = True
                    self._n_relief_positive += 1
                elif relief_pred_prev > 0.0 or self._relief_seen[a]:
                    relief_target_for_loss = 0.0
                    self._n_relief_negative += 1

            if self.config.use_safety_predictor:
                if ts_now <= 0.0:
                    safety_target = 1.0
                    safety_target_for_loss = 1.0
                    self._safety_seen[a] = True
                    self._n_safety_positive += 1
                elif safety_pred_prev > 0.0 or self._safety_seen[a]:
                    safety_target_for_loss = 0.0
                    self._n_safety_negative += 1

            relief_loss, safety_loss, optimizer_step = self._train_heads(
                prev_state,
                a,
                relief_target_for_loss,
                safety_target_for_loss,
            )
            relief_value = self.predict_relief(a, state_vec)
            safety_value = self.predict_safety(a, state_vec)

        elif action_ok and prev_z is not None and self._prev_state_vector is not None:
            # Extinguish predicted safety when a previously safe/action-bound
            # context is followed by threat recurrence. Positive safety still
            # requires prior threat; this branch only removes stale predictions.
            a = int(last_action_class)
            safety_pred_prev = self.predict_safety(a, self._prev_state_vector)
            if (
                self.config.use_safety_predictor
                and ts_now > 0.0
                and (safety_pred_prev > 0.0 or self._safety_seen[a])
            ):
                safety_target_for_loss = 0.0
                self._n_safety_negative += 1
                relief_loss, safety_loss, optimizer_step = self._train_heads(
                    self._prev_state_vector,
                    a,
                    relief_target=None,
                    safety_target=safety_target_for_loss,
                )
            relief_value = self.predict_relief(a, state_vec)
            safety_value = self.predict_safety(a, state_vec)

        self._prev_z_harm_a_norm = z_now
        self._prev_threat_scale = ts_now
        self._prev_state_vector = state_vec.detach().clone()
        self._last_state_vector = state_vec.detach().clone()
        self._update_prediction_maxima(state_vec)

        combined = self._clamp01(relief_value + safety_value)
        self._last_output = TrainableEscapeAffordanceLearnerOutput(
            threat_scale=float(ts_now),
            relief_target=float(relief_target),
            safety_target=float(safety_target),
            relief_value=float(relief_value),
            safety_value=float(safety_value),
            combined_value=float(combined),
            relief_loss=float(relief_loss),
            safety_loss=float(safety_loss),
            bias_max_abs=0.0,
            updated=bool(optimizer_step),
            optimizer_step=bool(optimizer_step),
            simulation_skipped=False,
        )
        return self._last_output

    # -- E3 score bias --

    def compute_approach_bias(
        self,
        z_harm_a_norm: float,
        action_classes: Union[Sequence[int], torch.Tensor],
        z_world: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        z_self: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        z_harm_a: Optional[Union[torch.Tensor, Sequence[float], float]] = None,
        threat_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        simulation_mode: bool = False,
        hypothesis_tag: bool = False,
    ) -> torch.Tensor:
        """Return bounded negative score-bias for predicted escape actions."""
        if isinstance(action_classes, torch.Tensor):
            classes = action_classes.detach().flatten().tolist()
        else:
            classes = list(action_classes)
        bias = torch.zeros(len(classes), dtype=dtype, device=device)

        if (
            not self.config.enabled
            or self._model is None
            or simulation_mode
            or hypothesis_tag
            or len(classes) == 0
        ):
            if simulation_mode or hypothesis_tag:
                self._n_sim_skipped += 1
            return bias

        ts = self.threat_scale(float(z_harm_a_norm), threat_scale=threat_scale)
        if ts <= 0.0:
            return bias

        state_vec = self.build_state_vector(
            z_world=z_world,
            z_self=z_self,
            z_harm_a=z_harm_a if z_harm_a is not None else [float(z_harm_a_norm)],
            z_harm_a_norm=float(z_harm_a_norm),
            threat_scale=ts,
        )

        scale = float(max(0.0, self.config.bias_scale))
        fired = False
        for i, cls in enumerate(classes):
            c = int(cls)
            if c == int(self.config.noop_class):
                continue
            relief = self.predict_relief(c, state_vec)
            safety = self.predict_safety(c, state_vec)
            combined = self._clamp01(relief + safety)
            if combined <= 0.0:
                continue
            mag = float(min(scale, scale * ts * combined))
            if mag > 0.0:
                bias[i] = -mag
                fired = True
        if fired:
            self._n_bias_fires += 1
        self._last_output.bias_max_abs = float(bias.abs().max().item()) if bias.numel() else 0.0
        self._last_output.threat_scale = float(ts)
        return bias

    # -- Read-only accessors --

    @property
    def model(self) -> Optional[nn.Module]:
        return self._model

    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self._optimizer

    @property
    def relief_values(self) -> List[float]:
        if self._last_state_vector is None:
            return [0.0] * max(1, int(self.config.n_action_classes))
        return [
            self.predict_relief(c, self._last_state_vector)
            for c in range(max(1, int(self.config.n_action_classes)))
        ]

    @property
    def safety_values(self) -> List[float]:
        if self._last_state_vector is None:
            return [0.0] * max(1, int(self.config.n_action_classes))
        return [
            self.predict_safety(c, self._last_state_vector)
            for c in range(max(1, int(self.config.n_action_classes)))
        ]

    def best_escape_class(self) -> int:
        best_c, best_v = -1, 0.0
        for c in range(max(1, int(self.config.n_action_classes))):
            if c == int(self.config.noop_class):
                continue
            v = self._clamp01(
                self.predict_relief(c, self._last_state_vector)
                + self.predict_safety(c, self._last_state_vector)
            )
            if v > best_v:
                best_c, best_v = c, v
        return best_c

    def last_output(self) -> TrainableEscapeAffordanceLearnerOutput:
        return self._last_output

    def get_state(self) -> dict:
        relief_values = self.relief_values
        safety_values = self.safety_values
        relief_max = float(max(relief_values) if relief_values else 0.0)
        safety_max = float(max(safety_values) if safety_values else 0.0)
        return {
            "trainable_escape_enabled": bool(self.config.enabled),
            "trainable_escape_model_instantiated": bool(self._model is not None),
            "trainable_escape_relief_max": relief_max,
            "trainable_escape_safety_max": safety_max,
            "trainable_escape_relief_max_prediction": relief_max,
            "trainable_escape_safety_max_prediction": safety_max,
            "trainable_escape_relief_max_prediction_ever": float(self._max_relief_prediction),
            "trainable_escape_safety_max_prediction_ever": float(self._max_safety_prediction),
            "trainable_escape_best_class": int(self.best_escape_class()),
            "trainable_escape_relief_loss": float(self._last_output.relief_loss),
            "trainable_escape_safety_loss": float(self._last_output.safety_loss),
            "trainable_escape_n_updates": int(self._n_updates),
            "trainable_escape_n_optimizer_steps": int(self._n_optimizer_steps),
            "trainable_escape_n_relief_positive": int(self._n_relief_positive),
            "trainable_escape_n_relief_negative": int(self._n_relief_negative),
            "trainable_escape_n_safety_positive": int(self._n_safety_positive),
            "trainable_escape_n_safety_negative": int(self._n_safety_negative),
            "trainable_escape_n_leak": int(self._n_weight_decay_steps),
            "trainable_escape_n_bias_fires": int(self._n_bias_fires),
            "trainable_escape_n_noop_skipped": int(self._n_noop_skipped),
            "trainable_escape_n_sim_skipped": int(self._n_sim_skipped),
        }

    @property
    def diagnostics(self) -> dict:
        return self.get_state()
