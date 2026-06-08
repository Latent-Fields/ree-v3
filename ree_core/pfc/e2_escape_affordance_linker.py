"""Post-603i E2 escape-affordance linker / readout (reuse, NOT a duplicate predictor).

V3-EXQ-603i landed FAIL / route ``substrate_not_ready_requeue`` /
``evidence_direction=non_contributory``. It did NOT weaken SD-059 / MECH-358. It
surfaced a prerequisite-representation gap: the relief/safety escape-affordance
bridge could not be adjudicated because the upstream representation needed for
escape-affordance credit -- "where out is" under threat -- was not ready (the
manifest states the detector needs a trained encoder / world-forward).

This module is that missing *linkage* layer. It is a READOUT over the EXISTING E2
(cerebellar-analog) action-consequence forward model -- it is NOT a new standalone
forward predictor. ``E2FastPredictor.world_forward`` / ``E2WorldForward`` already
predict action consequences; this linker indexes that geometry into escape
affordance viability and exposes:

  - ``escape_affordance_features``  -- a trained compact representation the
    relief/safety affect heads (TrainableEscapeAffordanceLearner) can optionally
    consume;
  - viability readouts -- predicted harm delta / threat termination / safety
    transition / refuge reachability / optional survival-step;
  - a per-action-class viability index (hippocampal-style scaffold: "where can
    this action sequence lead?" -- it does NOT compute reward and does NOT select
    actions);
  - a bounded, threat-gated E3 score-bias (behind a separate config flag).

Biological framing (preserve -- this layer must not absorb its neighbours):
  E2 / cerebellum     -> fast forward prediction of action consequences (REUSED;
                         inputs are taken from E2, never re-predicted here).
  Hippocampus         -> relational viability map over action-consequence
                         coordinates (scaffolded here as a readout/index, not a
                         trajectory generator).
  Amygdala / affect   -> relief and safety learning, kept DISTINCT (the heads).
  PFC gate / PAG      -> freeze suppression / defensive execution (NOT rebuilt;
                         SD-058 / MECH-357 / MECH-279 already own these).
  Basal ganglia / E3  -> selection + commitment (receives a bounded bias only).

REVISITABLE BET (reuse vs duplicate): "reuse E2, do not duplicate a forward
predictor" is an *assumption*, not a settled fact. The brain often duplicates the
forward-model motif across structures wired into different functional circuits
(cerebellar / cortical / striatal / hippocampal / defensive prediction), so a
dedicated escape-affordance forward circuit could be biologically faithful rather
than a duplication error. This module hedges that: ``e2_features`` is an ARGUMENT,
so the feature source can be swapped from the shared ``E2.world_forward`` to a
dedicated escape-specialised predictor later without re-deriving the readout
heads. See the "Architectural assumption" section of
docs/substrate_plans/post_603i_e2_escape_affordance_linkage.md for the falsifiable
revert trigger.

Guarantees (all enforced below):
  - OFF by default at the agent/config layer; disabled construction instantiates
    no neural state and emits zero bias.
  - Inputs from E2 are DETACHED -- no backprop into E1/E2/E3 encoders by default.
  - No learning under simulation / hypothesis_tag mode.
  - No credit to the no-op/freeze class by default.
  - Relief and safety remain DISTINCT readouts.
  - The bias is threat-gated (exactly zero when safe) and clamped to bias_scale.
  - Learned head weights persist across episode reset; reset clears only the
    one-tick traces used for action-contingent credit assignment.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


FeatureLike = Optional[Union[torch.Tensor, Sequence[float], float]]


@dataclass
class E2EscapeAffordanceLinkerConfig:
    """Configuration for the post-603i E2 escape-affordance linker.

    ``enabled`` is a module-local guard used by tests and direct construction.
    REEConfig's ``use_e2_escape_affordance_linker`` is the agent-level master
    switch and defaults False, so existing runs never instantiate this linker
    unless an experiment opts in.
    """

    enabled: bool = True
    n_action_classes: int = 5
    hidden_dim: int = 32
    action_embedding_dim: int = 8
    learn_rate: float = 0.1
    optimizer_lr: float = 0.03
    leak_rate: float = 0.01
    bias_scale: float = 0.1
    threat_floor: float = 0.1
    threat_ref: float = 0.5
    noop_class: int = 0
    relief_reward_floor: float = 1e-4
    harm_delta_scale: float = 0.3
    prediction_floor: float = 0.02
    max_grad_norm: float = 5.0
    block_hypothesis_learning: bool = True
    # Hippocampal-style viability-index scaffold (EMA of per-action-class escape
    # success). A readout/index only -- it does not generate trajectories or
    # compute reward.
    use_viability_index: bool = True
    viability_alpha: float = 0.1


@dataclass
class E2EscapeAffordanceLinkerOutput:
    """Readout from the most recent update or bias computation."""

    threat_scale: float = 0.0
    # Relief-side readouts (action-contingent aversive offset).
    predicted_harm_delta: float = 0.0
    predicted_threat_termination: float = 0.0
    # Safety-side readouts (learned threat-absence / response-produced safety).
    predicted_safety_transition: float = 0.0
    predicted_refuge_reachability: float = 0.0
    # Optional survival-horizon readout.
    predicted_survival_step: float = 0.0
    # Linker representation magnitude (the feature handed to the relief/safety heads).
    escape_affordance_norm: float = 0.0
    used_e2_features: bool = False
    bias_max_abs: float = 0.0
    updated: bool = False
    optimizer_step: bool = False
    simulation_skipped: bool = False


class _EscapeAffordanceReadout(nn.Module):
    """Small shared trunk over detached E2/compact features plus action embedding.

    The trunk's hidden activation IS ``escape_affordance_features`` (exposed for
    the relief/safety heads). Five bounded sigmoid readout heads sit on top. The
    heads are biased to a low prior so an untrained-but-enabled linker emits
    near-zero affordance, never arbitrary escape salience.
    """

    HEAD_NAMES = (
        "harm_delta",
        "threat_termination",
        "safety_transition",
        "refuge_reachability",
        "survival_step",
    )

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
        self.harm_delta_head = nn.Linear(hidden, 1)
        self.threat_termination_head = nn.Linear(hidden, 1)
        self.safety_transition_head = nn.Linear(hidden, 1)
        self.refuge_reachability_head = nn.Linear(hidden, 1)
        self.survival_step_head = nn.Linear(hidden, 1)
        self.hidden_dim = hidden
        self._init_low_prior()

    def _init_low_prior(self) -> None:
        with torch.no_grad():
            for name in self.HEAD_NAMES:
                getattr(self, name + "_head").bias.fill_(-4.0)

    def features(
        self,
        state_vector: torch.Tensor,
        action_class: torch.Tensor,
    ) -> torch.Tensor:
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)
        action_class = action_class.to(dtype=torch.long, device=state_vector.device)
        action_class = action_class.clamp(min=0, max=self.n_action_classes - 1)
        if action_class.dim() == 0:
            action_class = action_class.unsqueeze(0)
        action_class = action_class.flatten()
        emb = self.action_embedding(action_class)
        x = torch.cat([state_vector, emb], dim=-1)
        return self.trunk(x)

    def forward(
        self,
        state_vector: torch.Tensor,
        action_class: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(state_vector, action_class)
        heads = torch.cat(
            [
                torch.sigmoid(self.harm_delta_head(h)),
                torch.sigmoid(self.threat_termination_head(h)),
                torch.sigmoid(self.safety_transition_head(h)),
                torch.sigmoid(self.refuge_reachability_head(h)),
                torch.sigmoid(self.survival_step_head(h)),
            ],
            dim=-1,
        )
        return heads, h


class E2EscapeAffordanceLinker:
    """Readout/linker over detached E2 action-consequence features.

    The linker learns ONLY action-contingent viability labels on top of the
    existing E2 forward geometry. It does not duplicate E2 next-state prediction,
    hippocampal trajectory generation, E3 selection, or ilPFC/PAG freeze
    suppression.

    The model is lazily initialized from the first trainable transition, so
    direct disabled construction and untrained readiness checks instantiate no
    neural state.
    """

    HEAD_INDEX = {name: i for i, name in enumerate(_EscapeAffordanceReadout.HEAD_NAMES)}
    RELIEF_HEADS = ("harm_delta", "threat_termination")
    SAFETY_HEADS = ("safety_transition", "refuge_reachability")

    def __init__(
        self,
        config: Optional[E2EscapeAffordanceLinkerConfig] = None,
    ) -> None:
        self.config = config or E2EscapeAffordanceLinkerConfig()
        k = max(1, int(self.config.n_action_classes))

        self._model: Optional[_EscapeAffordanceReadout] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._state_dim: Optional[int] = None

        # Per-head "seen" gating (suppress untrained head leakage below a floor).
        self._head_seen = {name: [False] * k for name in _EscapeAffordanceReadout.HEAD_NAMES}

        # One-tick traces for action-contingent credit assignment.
        self._prev_z_harm_a_norm: Optional[float] = None
        self._prev_threat_scale: float = 0.0
        self._prev_state_vector: Optional[torch.Tensor] = None
        self._prev_refuge_dist: Optional[float] = None
        self._prev_safety_predicted: List[bool] = [False] * k
        self._last_state_vector: Optional[torch.Tensor] = None

        # Hippocampal-style viability index (EMA escape-success per action class).
        self._viability: List[float] = [0.0] * k

        # Diagnostics.
        self._n_updates: int = 0
        self._n_optimizer_steps: int = 0
        self._n_positive: int = 0
        self._n_negative: int = 0
        self._n_noop_skipped: int = 0
        self._n_sim_skipped: int = 0
        self._n_bias_fires: int = 0
        self._n_e2_feature_ticks: int = 0
        self._n_weight_decay_steps: int = 0
        self._max_escape_prediction: float = 0.0

        self._last_output = E2EscapeAffordanceLinkerOutput()

    # -- State management -----------------------------------------------------

    def reset(self) -> None:
        """Clear within-episode traces while preserving learned head weights."""
        k = max(1, int(self.config.n_action_classes))
        self._prev_z_harm_a_norm = None
        self._prev_threat_scale = 0.0
        self._prev_state_vector = None
        self._prev_refuge_dist = None
        self._prev_safety_predicted = [False] * k

    # -- Feature assembly -----------------------------------------------------

    @staticmethod
    def _flatten_optional(value: FeatureLike) -> torch.Tensor:
        if value is None:
            return torch.empty(0, dtype=torch.float32)
        if isinstance(value, torch.Tensor):
            return value.detach().flatten().to(dtype=torch.float32, device="cpu")
        return torch.as_tensor(value, dtype=torch.float32).flatten()

    def build_state_vector(
        self,
        e2_features: FeatureLike = None,
        z_world: FeatureLike = None,
        z_self: FeatureLike = None,
        z_harm_a: FeatureLike = None,
        z_harm_a_norm: Optional[float] = None,
        threat_scale: Optional[float] = None,
        refuge_features: FeatureLike = None,
    ) -> torch.Tensor:
        """Assemble the compact detached linker input.

        ``e2_features`` (detached E2 action-consequence outputs, e.g. predicted
        next z_world and its delta) are the PRIMARY representation. Compact
        ``z_world`` / ``z_self`` / ``z_harm_a`` are FALLBACK features used only
        when E2 features are unavailable, so the linker degrades gracefully on a
        substrate that has not yet exposed an E2 forward model. The action is
        represented by the model's action embedding during prediction/training,
        not appended here (prevents stale previous-action leakage across ticks).
        """
        e2 = self._flatten_optional(e2_features)
        used_e2 = e2.numel() > 0
        if used_e2:
            parts = [e2]
        else:
            parts = [
                self._flatten_optional(z_world),
                self._flatten_optional(z_self),
                self._flatten_optional(z_harm_a),
            ]
        parts.append(self._flatten_optional(refuge_features))
        z_norm = (
            float(z_harm_a_norm)
            if z_harm_a_norm is not None
            else float(self._flatten_optional(z_harm_a).norm().item())
        )
        parts.append(
            torch.tensor(
                [z_norm, float(threat_scale) if threat_scale is not None else 0.0],
                dtype=torch.float32,
            )
        )
        non_empty = [p for p in parts if p.numel() > 0]
        if not non_empty:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(non_empty).detach().clone()

    def _coerce_state_vector(self, state_vector: torch.Tensor) -> torch.Tensor:
        s = state_vector.detach().flatten().to(dtype=torch.float32, device="cpu")
        if s.numel() == 0:
            s = torch.zeros(1, dtype=torch.float32)
        if self._state_dim is None or s.numel() == self._state_dim:
            return s
        if s.numel() > self._state_dim:
            return s[: self._state_dim].clone()
        out = torch.zeros(self._state_dim, dtype=torch.float32)
        out[: s.numel()] = s
        return out

    # -- Threat scale ---------------------------------------------------------

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
        ref = max(float(self.config.threat_ref), floor + 1e-6)
        return float(max(0.0, min(1.0, (z - floor) / (ref - floor))))

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _valid_action(self, action_class: int) -> bool:
        return 0 <= int(action_class) < max(1, int(self.config.n_action_classes))

    # -- Model + prediction ---------------------------------------------------

    def _ensure_model(self, state_vector: torch.Tensor) -> Optional[_EscapeAffordanceReadout]:
        if not self.config.enabled:
            return None
        if self._model is not None:
            return self._model
        s = state_vector.detach().flatten().to(dtype=torch.float32, device="cpu")
        self._state_dim = max(1, int(s.numel()))
        self._model = _EscapeAffordanceReadout(
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

    def _raw_heads(
        self,
        action_class: int,
        state_vector: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            not self.config.enabled
            or self._model is None
            or state_vector is None
            or not self._valid_action(int(action_class))
        ):
            return None
        s = self._coerce_state_vector(state_vector).unsqueeze(0)
        a = torch.tensor([int(action_class)], dtype=torch.long)
        with torch.no_grad():
            heads, _ = self._model(s, a)
        return heads.flatten()

    def _masked(self, raw: float, seen: bool) -> float:
        if seen or raw >= float(max(0.0, self.config.prediction_floor)):
            return self._clamp01(raw)
        return 0.0

    def predict_head(
        self,
        head: str,
        action_class: int,
        state_vector: Optional[torch.Tensor] = None,
    ) -> float:
        if not self.config.enabled or head not in self.HEAD_INDEX:
            return 0.0
        state = state_vector if state_vector is not None else self._last_state_vector
        heads = self._raw_heads(int(action_class), state)
        if heads is None:
            return 0.0
        raw = float(heads[self.HEAD_INDEX[head]].item())
        seen = (
            self._head_seen[head][int(action_class)]
            if self._valid_action(action_class)
            else False
        )
        return self._masked(raw, seen)

    def escape_salience(
        self,
        action_class: int,
        state_vector: Optional[torch.Tensor] = None,
    ) -> float:
        """Relief-side escape salience for an action: harm-drop + threat-end.

        Relief and safety are kept distinct; this is the relief/aversive-offset
        readout used to bias E3 toward a directed escape. Safety readouts are
        exposed separately via predict_head and are not folded in here.
        """
        relief = sum(self.predict_head(h, action_class, state_vector) for h in self.RELIEF_HEADS)
        return self._clamp01(relief / max(1, len(self.RELIEF_HEADS)))

    def escape_affordance_features(
        self,
        action_class: int,
        state_vector: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Detached trunk activation -- the representation the relief/safety heads
        (TrainableEscapeAffordanceLearner) can optionally consume. None when the
        linker is disabled / untrained."""
        if not self.config.enabled or self._model is None:
            return None
        state = state_vector if state_vector is not None else self._last_state_vector
        if state is None or not self._valid_action(int(action_class)):
            return None
        s = self._coerce_state_vector(state).unsqueeze(0)
        a = torch.tensor([int(action_class)], dtype=torch.long)
        with torch.no_grad():
            h = self._model.features(s, a)
        return h.flatten().detach().clone()

    # -- Training -------------------------------------------------------------

    def _train(
        self,
        state_vector: torch.Tensor,
        action_class: int,
        targets: dict,
    ) -> Tuple[float, bool]:
        model = self._ensure_model(state_vector)
        if model is None or self._optimizer is None or not targets:
            return 0.0, False
        s = self._coerce_state_vector(state_vector).unsqueeze(0)
        a = torch.tensor([int(action_class)], dtype=torch.long)
        heads, _ = model(s, a)
        losses: List[torch.Tensor] = []
        for name, target in targets.items():
            if target is None:
                continue
            idx = self.HEAD_INDEX[name]
            pred = heads[..., idx : idx + 1]
            tgt = torch.tensor([[self._clamp01(float(target))]], dtype=torch.float32)
            losses.append(F.binary_cross_entropy(pred, tgt))
        if not losses:
            return 0.0, False
        loss = torch.stack(losses).sum() * float(max(0.0, self.config.learn_rate))
        loss_value = float(loss.detach().item())
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        max_norm = float(max(0.0, self.config.max_grad_norm))
        if max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        self._optimizer.step()
        self._n_optimizer_steps += 1
        if float(max(0.0, self.config.leak_rate)) > 0.0:
            self._n_weight_decay_steps += 1
        return loss_value, True

    def update(
        self,
        z_harm_a_norm: float,
        last_action_class: Optional[int],
        e2_features: FeatureLike = None,
        z_world: FeatureLike = None,
        z_self: FeatureLike = None,
        z_harm_a: FeatureLike = None,
        threat_scale: Optional[float] = None,
        last_action_directed: bool = True,
        refuge_dist: Optional[float] = None,
        survival_improved: Optional[bool] = None,
        refuge_features: FeatureLike = None,
        simulation_mode: bool = False,
        hypothesis_tag: bool = False,
    ) -> E2EscapeAffordanceLinkerOutput:
        """Advance escape-affordance viability readouts by one waking tick.

        Positive escape target when the previous state was under threat, the
        action was directed (non-noop), AND harm dropped / threat terminated /
        refuge became closer-or-reached / survival improved. Negative
        (extinction) target when a directed escape attempt under prior threat did
        not reduce harm/threat, or when predicted safety was followed by threat
        recurrence. No credit to no-op/freeze, simulation/hypothesis ticks, or
        when no threat transition occurred.
        """
        block_hyp = bool(self.config.block_hypothesis_learning) and bool(hypothesis_tag)
        if (not self.config.enabled) or simulation_mode or block_hyp:
            if simulation_mode or block_hyp:
                self._n_sim_skipped += 1
            self._last_output = E2EscapeAffordanceLinkerOutput(
                simulation_skipped=bool(simulation_mode or block_hyp)
            )
            return self._last_output

        z_now = float(z_harm_a_norm)
        ts_now = self.threat_scale(z_now, threat_scale=threat_scale)
        used_e2 = self._flatten_optional(e2_features).numel() > 0
        if used_e2:
            self._n_e2_feature_ticks += 1
        state_vec = self.build_state_vector(
            e2_features=e2_features,
            z_world=z_world,
            z_self=z_self,
            z_harm_a=z_harm_a if z_harm_a is not None else [z_now],
            z_harm_a_norm=z_now,
            threat_scale=ts_now,
            refuge_features=refuge_features,
        )

        self._n_updates += 1
        loss = 0.0
        optimizer_step = False

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
            harm_drop = float(prev_z) - z_now
            threat_terminated = ts_now <= 0.0
            refuge_closer = (
                refuge_dist is not None
                and self._prev_refuge_dist is not None
                and (self._prev_refuge_dist - float(refuge_dist)) > 0.0
            )
            survival_up = bool(survival_improved) if survival_improved is not None else False
            positive = (
                harm_drop > float(self.config.relief_reward_floor)
                or threat_terminated
                or refuge_closer
                or survival_up
            )

            targets: dict = {}
            # Relief side -- action-contingent aversive offset.
            if harm_drop > float(self.config.relief_reward_floor):
                scale = max(float(self.config.harm_delta_scale), 1e-6)
                targets["harm_delta"] = self._clamp01(harm_drop / scale)
                self._head_seen["harm_delta"][a] = True
            elif self._head_seen["harm_delta"][a]:
                targets["harm_delta"] = 0.0
            if threat_terminated:
                targets["threat_termination"] = 1.0
                self._head_seen["threat_termination"][a] = True
            elif self._head_seen["threat_termination"][a]:
                targets["threat_termination"] = 0.0
            # Safety side -- learned threat-absence / response-produced safety
            # (kept distinct from relief: keyed on threat absence, not harm drop).
            if threat_terminated:
                targets["safety_transition"] = 1.0
                self._head_seen["safety_transition"][a] = True
                self._prev_safety_predicted[a] = True
            elif self._head_seen["safety_transition"][a]:
                targets["safety_transition"] = 0.0
            # Refuge reachability -- only trained when a refuge signal exists.
            if refuge_dist is not None and self._prev_refuge_dist is not None:
                if refuge_closer:
                    targets["refuge_reachability"] = 1.0
                    self._head_seen["refuge_reachability"][a] = True
                elif self._head_seen["refuge_reachability"][a]:
                    targets["refuge_reachability"] = 0.0
            # Survival horizon -- only trained when a survival signal exists.
            if survival_improved is not None:
                targets["survival_step"] = 1.0 if survival_up else 0.0
                self._head_seen["survival_step"][a] = True

            loss, optimizer_step = self._train(prev_state, a, targets)
            if positive:
                self._n_positive += 1
            else:
                self._n_negative += 1
            if self.config.use_viability_index:
                alpha = float(self.config.viability_alpha)
                self._viability[a] = (1.0 - alpha) * self._viability[a] + alpha * (
                    1.0 if positive else 0.0
                )

        elif action_ok and prev_z is not None and self._prev_state_vector is not None:
            # Extinguish predicted safety on threat recurrence (was safe/bound,
            # now threatened again). Positive safety still requires prior threat;
            # this branch only removes stale safety predictions.
            a = int(last_action_class)
            if ts_now > 0.0 and (
                self._head_seen["safety_transition"][a] or self._prev_safety_predicted[a]
            ):
                loss, optimizer_step = self._train(
                    self._prev_state_vector, a, {"safety_transition": 0.0}
                )
                self._n_negative += 1
                self._prev_safety_predicted[a] = False

        # Advance traces.
        self._prev_z_harm_a_norm = z_now
        self._prev_threat_scale = ts_now
        self._prev_state_vector = state_vec.detach().clone()
        self._prev_refuge_dist = float(refuge_dist) if refuge_dist is not None else None
        self._last_state_vector = state_vec.detach().clone()

        out = E2EscapeAffordanceLinkerOutput(
            threat_scale=float(ts_now),
            used_e2_features=bool(used_e2),
            updated=bool(optimizer_step),
            optimizer_step=bool(optimizer_step),
            simulation_skipped=False,
        )
        if self._model is not None and last_action_class is not None and self._valid_action(int(last_action_class)):
            a = int(last_action_class)
            out.predicted_harm_delta = self.predict_head("harm_delta", a, state_vec)
            out.predicted_threat_termination = self.predict_head("threat_termination", a, state_vec)
            out.predicted_safety_transition = self.predict_head("safety_transition", a, state_vec)
            out.predicted_refuge_reachability = self.predict_head("refuge_reachability", a, state_vec)
            out.predicted_survival_step = self.predict_head("survival_step", a, state_vec)
            feats = self.escape_affordance_features(a, state_vec)
            out.escape_affordance_norm = float(feats.norm().item()) if feats is not None else 0.0
            self._max_escape_prediction = max(
                self._max_escape_prediction, self.escape_salience(a, state_vec)
            )
        self._last_output = out
        return out

    # -- E3 score-bias --------------------------------------------------------

    def compute_approach_bias(
        self,
        z_harm_a_norm: float,
        action_classes: Union[Sequence[int], torch.Tensor],
        e2_features: FeatureLike = None,
        z_world: FeatureLike = None,
        z_self: FeatureLike = None,
        z_harm_a: FeatureLike = None,
        threat_scale: Optional[float] = None,
        refuge_features: FeatureLike = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        simulation_mode: bool = False,
        hypothesis_tag: bool = False,
    ) -> torch.Tensor:
        """Return a bounded, threat-gated negative score-bias for escape actions.

        REE is lower-is-better, so a negative bias FAVOURS the predicted escape
        action. The bias is exactly zero when safe (threat scale 0), clamped to
        ``bias_scale``, and never applied to the no-op/freeze class.
        """
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
            e2_features=e2_features,
            z_world=z_world,
            z_self=z_self,
            z_harm_a=z_harm_a if z_harm_a is not None else [float(z_harm_a_norm)],
            z_harm_a_norm=float(z_harm_a_norm),
            threat_scale=ts,
            refuge_features=refuge_features,
        )

        scale = float(max(0.0, self.config.bias_scale))
        use_viab = bool(self.config.use_viability_index)
        fired = False
        for i, cls in enumerate(classes):
            c = int(cls)
            if c == int(self.config.noop_class):
                continue
            salience = self.escape_salience(c, state_vec)
            if use_viab and self._valid_action(c):
                salience = self._clamp01(salience * (0.5 + 0.5 * self._viability[c]))
            if salience <= 0.0:
                continue
            mag = float(min(scale, scale * ts * salience))
            if mag > 0.0:
                bias[i] = -mag
                fired = True
        if fired:
            self._n_bias_fires += 1
        self._last_output.bias_max_abs = (
            float(bias.abs().max().item()) if bias.numel() else 0.0
        )
        self._last_output.threat_scale = float(ts)
        return bias

    # -- Read-only accessors --------------------------------------------------

    @property
    def model(self) -> Optional[nn.Module]:
        return self._model

    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self._optimizer

    @property
    def viability_index(self) -> List[float]:
        return list(self._viability)

    def best_escape_class(self) -> int:
        best_c, best_v = -1, 0.0
        for c in range(max(1, int(self.config.n_action_classes))):
            if c == int(self.config.noop_class):
                continue
            v = self.escape_salience(c, self._last_state_vector)
            if v > best_v:
                best_c, best_v = c, v
        return best_c

    def last_output(self) -> E2EscapeAffordanceLinkerOutput:
        return self._last_output

    def get_state(self) -> dict:
        return {
            "e2_escape_linker_enabled": bool(self.config.enabled),
            "e2_escape_linker_model_instantiated": bool(self._model is not None),
            "e2_escape_linker_n_updates": int(self._n_updates),
            "e2_escape_linker_n_optimizer_steps": int(self._n_optimizer_steps),
            "e2_escape_linker_n_positive": int(self._n_positive),
            "e2_escape_linker_n_negative": int(self._n_negative),
            "e2_escape_linker_n_noop_skipped": int(self._n_noop_skipped),
            "e2_escape_linker_n_sim_skipped": int(self._n_sim_skipped),
            "e2_escape_linker_n_bias_fires": int(self._n_bias_fires),
            "e2_escape_linker_n_e2_feature_ticks": int(self._n_e2_feature_ticks),
            "e2_escape_linker_n_leak": int(self._n_weight_decay_steps),
            "e2_escape_linker_best_class": int(self.best_escape_class()),
            "e2_escape_linker_max_escape_prediction": float(self._max_escape_prediction),
            "e2_escape_linker_viability_max": float(
                max(self._viability) if self._viability else 0.0
            ),
        }

    @property
    def diagnostics(self) -> dict:
        return self.get_state()
