"""Post-603i trainable-ready relief/safety escape-affordance learner.

This module is a feature-flagged successor scaffold for the SD-059 arithmetic
EscapeAffordanceBridge. It does not replace the active V3-EXQ-603i validation
path and is OFF by default at the agent/config layer.

The first implementation is intentionally compact: it exposes explicit
state-vector hooks, per-action update targets, bounded predictions, and clear
extinction/leak behaviour without adding a neural head to the live agent loop.
Future experiments can replace the scalar/prototype tables with small PyTorch
heads while keeping the same update and bias contracts.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F


@dataclass
class TrainableEscapeAffordanceLearnerConfig:
    """Configuration for the post-603i trainable-ready learner.

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
    leak_rate: float = 0.01
    relief_reward_floor: float = 1e-4
    threat_floor: float = 0.1
    noop_class: int = 0


@dataclass
class TrainableEscapeAffordanceLearnerOutput:
    """Readout from the most recent update or bias computation."""

    threat_scale: float = 0.0
    relief_target: float = 0.0
    safety_target: float = 0.0
    relief_value: float = 0.0
    safety_value: float = 0.0
    combined_value: float = 0.0
    bias_max_abs: float = 0.0
    updated: bool = False
    simulation_skipped: bool = False


class TrainableEscapeAffordanceLearner:
    """Trainable-ready relief critic + safety predictor scaffold.

    Current representation:
      - Q_relief(state, action, threat_context): scalar per action class,
        optionally state-gated by a compact prototype for that action.
      - P_safety(state, context, action): scalar per action class, likewise
        prototype-gated.

    The state vector is a hook, not an architectural commitment. Callers may pass
    compact z_world, z_self, z_harm_a, and recent outcome features. The learner
    stores EMA prototypes on positive targets so a later neural head can consume
    the same compact features without changing the update contract.
    """

    def __init__(
        self,
        config: Optional[TrainableEscapeAffordanceLearnerConfig] = None,
    ):
        self.config = config or TrainableEscapeAffordanceLearnerConfig()
        k = max(1, int(self.config.n_action_classes))
        self._relief_value: List[float] = [0.0] * k
        self._safety_value: List[float] = [0.0] * k
        self._relief_proto: List[Optional[torch.Tensor]] = [None] * k
        self._safety_proto: List[Optional[torch.Tensor]] = [None] * k

        self._prev_z_harm_a_norm: Optional[float] = None
        self._prev_threat_scale: float = 0.0
        self._prev_state_vector: Optional[torch.Tensor] = None

        self._n_updates: int = 0
        self._n_relief_positive: int = 0
        self._n_relief_negative: int = 0
        self._n_safety_positive: int = 0
        self._n_safety_negative: int = 0
        self._n_leak: int = 0
        self._n_bias_fires: int = 0
        self._n_noop_skipped: int = 0
        self._n_sim_skipped: int = 0

        self._last_output = TrainableEscapeAffordanceLearnerOutput()

    # -- State management --

    def reset(self) -> None:
        """Clear within-episode traces while preserving learned tables."""
        self._prev_z_harm_a_norm = None
        self._prev_threat_scale = 0.0
        self._prev_state_vector = None

    # -- Feature hooks --

    @staticmethod
    def _flatten_optional(value: Optional[Union[torch.Tensor, Sequence[float], float]]) -> torch.Tensor:
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
        threat_scale: Optional[float] = None,
        action_class: Optional[int] = None,
        recent_outcome_features: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    ) -> torch.Tensor:
        """Compact trainable-ready state vector.

        It is deliberately permissive: callers can pass any currently available
        latent surfaces. Missing inputs produce a small scalar context rather than
        failing, which keeps the scaffold usable in narrow unit tests.
        """
        parts = [
            self._flatten_optional(z_world),
            self._flatten_optional(z_self),
            self._flatten_optional(z_harm_a),
            self._flatten_optional(recent_outcome_features),
            torch.tensor([
                float(threat_scale) if threat_scale is not None else 0.0,
                float(action_class) if action_class is not None else -1.0,
            ], dtype=torch.float32),
        ]
        non_empty = [p for p in parts if p.numel() > 0]
        if not non_empty:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(non_empty).detach().clone()

    # -- Prediction helpers --

    def threat_scale(self, z_harm_a_norm: float, threat_scale: Optional[float] = None) -> float:
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

    @staticmethod
    def _prototype_similarity(
        proto: Optional[torch.Tensor],
        state_vector: Optional[torch.Tensor],
    ) -> float:
        if proto is None or state_vector is None or proto.numel() == 0 or state_vector.numel() == 0:
            return 1.0
        if proto.numel() != state_vector.numel():
            return 1.0
        p = proto.detach().flatten().to(dtype=torch.float32)
        s = state_vector.detach().flatten().to(dtype=torch.float32)
        if float(p.norm().item()) <= 1e-9 or float(s.norm().item()) <= 1e-9:
            return 1.0
        return TrainableEscapeAffordanceLearner._clamp01(
            float(F.cosine_similarity(p, s, dim=0).item())
        )

    def _predict(
        self,
        values: Sequence[float],
        protos: Sequence[Optional[torch.Tensor]],
        action_class: int,
        state_vector: Optional[torch.Tensor],
    ) -> float:
        if action_class < 0 or action_class >= len(values):
            return 0.0
        base = self._clamp01(float(values[action_class]))
        sim = self._prototype_similarity(protos[action_class], state_vector)
        return self._clamp01(base * sim)

    def predict_relief(
        self,
        action_class: int,
        state_vector: Optional[torch.Tensor] = None,
    ) -> float:
        if not self.config.enabled or not self.config.use_relief_critic:
            return 0.0
        return self._predict(self._relief_value, self._relief_proto, int(action_class), state_vector)

    def predict_safety(
        self,
        action_class: int,
        state_vector: Optional[torch.Tensor] = None,
    ) -> float:
        if not self.config.enabled or not self.config.use_safety_predictor:
            return 0.0
        return self._predict(self._safety_value, self._safety_proto, int(action_class), state_vector)

    def _ema_value(self, values: List[float], idx: int, target: float, rate: float) -> None:
        r = float(max(0.0, min(1.0, rate)))
        values[idx] = self._clamp01(values[idx] + r * (float(target) - values[idx]))

    def _ema_proto(
        self,
        protos: List[Optional[torch.Tensor]],
        idx: int,
        state_vector: torch.Tensor,
        rate: float,
    ) -> None:
        if state_vector.numel() == 0:
            return
        r = float(max(0.0, min(1.0, rate)))
        s = state_vector.detach().flatten().to(dtype=torch.float32, device="cpu")
        prev = protos[idx]
        if prev is None or prev.numel() != s.numel():
            protos[idx] = s.clone()
        else:
            protos[idx] = (1.0 - r) * prev + r * s

    def _apply_leak(self) -> None:
        leak = float(max(0.0, min(1.0, self.config.leak_rate)))
        if leak <= 0.0:
            return
        for i in range(len(self._relief_value)):
            self._relief_value[i] = self._clamp01(self._relief_value[i] * (1.0 - leak))
            self._safety_value[i] = self._clamp01(self._safety_value[i] * (1.0 - leak))
        self._n_leak += 1

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

        Positive relief target: previous tick was under threat, the last action
        was directed/non-noop, and z_harm_a dropped more than the floor.

        Positive safety target: previous tick was under threat, the last action
        was directed/non-noop, and the current tick is threat-absent. This avoids
        learning "safety = merely low harm" in contexts that were never under
        threat.

        Negative/extinction targets: predicted relief without a harm drop decays
        the relief critic; predicted safety followed by threat recurrence decays
        the safety predictor. Leak handles stale predictions even without a
        targeted negative event.
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
            threat_scale=ts_now,
            action_class=last_action_class,
            recent_outcome_features=recent_outcome_features,
        )

        self._n_updates += 1
        self._apply_leak()

        relief_target = 0.0
        safety_target = 0.0
        updated = False
        relief_value = 0.0
        safety_value = 0.0

        prev_z = self._prev_z_harm_a_norm
        prev_under_threat = bool(prev_z is not None and self._prev_threat_scale > 0.0)
        action_ok = (
            last_action_class is not None
            and bool(last_action_directed)
            and 0 <= int(last_action_class) < len(self._relief_value)
            and int(last_action_class) != int(self.config.noop_class)
        )

        if prev_under_threat and last_action_class is not None and not action_ok:
            self._n_noop_skipped += 1

        if prev_under_threat and action_ok:
            a = int(last_action_class)
            prev_state = self._prev_state_vector
            relief_pred_prev = self.predict_relief(a, prev_state)
            safety_pred_prev = self.predict_safety(a, prev_state)

            if self.config.use_relief_critic:
                harm_drop = float(prev_z) - z_now
                if harm_drop > float(self.config.relief_reward_floor):
                    relief_target = 1.0
                    self._ema_value(
                        self._relief_value,
                        a,
                        target=1.0,
                        rate=float(self.config.relief_learn_rate),
                    )
                    if prev_state is not None:
                        self._ema_proto(
                            self._relief_proto,
                            a,
                            prev_state,
                            rate=float(self.config.relief_learn_rate),
                        )
                    self._n_relief_positive += 1
                    updated = True
                elif relief_pred_prev > 0.0:
                    self._ema_value(
                        self._relief_value,
                        a,
                        target=0.0,
                        rate=float(self.config.relief_learn_rate),
                    )
                    self._n_relief_negative += 1
                    updated = True

            if self.config.use_safety_predictor:
                if ts_now <= 0.0:
                    safety_target = 1.0
                    self._ema_value(
                        self._safety_value,
                        a,
                        target=1.0,
                        rate=float(self.config.safety_learn_rate),
                    )
                    if prev_state is not None:
                        self._ema_proto(
                            self._safety_proto,
                            a,
                            prev_state,
                            rate=float(self.config.safety_learn_rate),
                        )
                    self._n_safety_positive += 1
                    updated = True
                elif safety_pred_prev > 0.0:
                    self._ema_value(
                        self._safety_value,
                        a,
                        target=0.0,
                        rate=float(self.config.safety_learn_rate),
                    )
                    self._n_safety_negative += 1
                    updated = True

            relief_value = self.predict_relief(a, state_vec)
            safety_value = self.predict_safety(a, state_vec)

        elif action_ok and prev_z is not None and self.config.use_safety_predictor:
            # Extinguish predicted safety when a previously safe/action-bound
            # context is followed by threat recurrence. This is intentionally
            # separate from the positive safety target, which still requires
            # previous threat so "safety = merely low harm" is not learned.
            a = int(last_action_class)
            safety_pred_prev = self.predict_safety(a, self._prev_state_vector)
            if ts_now > 0.0 and safety_pred_prev > 0.0:
                self._ema_value(
                    self._safety_value,
                    a,
                    target=0.0,
                    rate=float(self.config.safety_learn_rate),
                )
                self._n_safety_negative += 1
                updated = True
            relief_value = self.predict_relief(a, state_vec)
            safety_value = self.predict_safety(a, state_vec)

        self._prev_z_harm_a_norm = z_now
        self._prev_threat_scale = ts_now
        self._prev_state_vector = state_vec.detach().clone()

        combined = self._clamp01(relief_value + safety_value)
        self._last_output = TrainableEscapeAffordanceLearnerOutput(
            threat_scale=float(ts_now),
            relief_target=float(relief_target),
            safety_target=float(safety_target),
            relief_value=float(relief_value),
            safety_value=float(safety_value),
            combined_value=float(combined),
            bias_max_abs=0.0,
            updated=bool(updated),
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

        scale = float(max(0.0, self.config.bias_scale))
        fired = False
        for i, cls in enumerate(classes):
            c = int(cls)
            if c == int(self.config.noop_class):
                continue
            state_vec = self.build_state_vector(
                z_world=z_world,
                z_self=z_self,
                z_harm_a=z_harm_a if z_harm_a is not None else [float(z_harm_a_norm)],
                threat_scale=ts,
                action_class=c,
            )
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
    def relief_values(self) -> List[float]:
        return list(self._relief_value)

    @property
    def safety_values(self) -> List[float]:
        return list(self._safety_value)

    def best_escape_class(self) -> int:
        best_c, best_v = -1, 0.0
        for c in range(len(self._relief_value)):
            if c == int(self.config.noop_class):
                continue
            v = self._clamp01(self._relief_value[c] + self._safety_value[c])
            if v > best_v:
                best_c, best_v = c, v
        return best_c

    def last_output(self) -> TrainableEscapeAffordanceLearnerOutput:
        return self._last_output

    def get_state(self) -> dict:
        return {
            "trainable_escape_enabled": bool(self.config.enabled),
            "trainable_escape_relief_max": float(max(self._relief_value) if self._relief_value else 0.0),
            "trainable_escape_safety_max": float(max(self._safety_value) if self._safety_value else 0.0),
            "trainable_escape_best_class": int(self.best_escape_class()),
            "trainable_escape_n_updates": int(self._n_updates),
            "trainable_escape_n_relief_positive": int(self._n_relief_positive),
            "trainable_escape_n_relief_negative": int(self._n_relief_negative),
            "trainable_escape_n_safety_positive": int(self._n_safety_positive),
            "trainable_escape_n_safety_negative": int(self._n_safety_negative),
            "trainable_escape_n_leak": int(self._n_leak),
            "trainable_escape_n_bias_fires": int(self._n_bias_fires),
            "trainable_escape_n_noop_skipped": int(self._n_noop_skipped),
            "trainable_escape_n_sim_skipped": int(self._n_sim_skipped),
        }

    @property
    def diagnostics(self) -> dict:
        return self.get_state()
