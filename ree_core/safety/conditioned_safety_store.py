"""SD-051 / MECH-304 substrate: cue-specific conditioned safety prediction store.

Maintains a prototype of the world latent (z_world) at MECH-302 relief-completion
event times. At each step, returns a cosine-similarity safety prediction.

Encoding pathway (dorsal striatum / dlPFC analog): EMA of z_world at event ticks.
Expression pathway (IL->CeA analog): cosine similarity -> commitment-release gate.

Non-trainable. Pure arithmetic. No nn.Module inheritance.

Simulation gating: update() with sim_mode=True returns 0.0 without advancing the
prototype -- waking-path signal only (MECH-094 requirement).

V4-deferred:
  - Approach attractor (output 2 of MECH-304 spec): requires V4 multi-step planning.
  - Contrastive cue-specific learning: requires trainable encoder head + phased training.
  See sd_051_conditioned_safety_store.md and v3_v4_transition_boundary.md.
"""

import math


class ConditionedSafetyStore:
    """EMA-prototype conditioned safety predictor for SD-051 / MECH-304.

    Learns which z_world states predict suffering absence via MECH-302 events.
    Outputs a safety_prediction scalar consumed by select_action() to gate
    avoidance commitment release (IL->CeA expression pathway).
    """

    def __init__(
        self,
        world_dim: int,
        ema_alpha: float = 0.1,
        decay_rate: float = 0.001,
        min_norm: float = 0.1,
        threshold: float = 0.5,
        gain: float = 4.0,
    ):
        """
        Args:
            world_dim: dimensionality of z_world latent.
            ema_alpha: prototype update rate per MECH-302 event tick (0,1].
            decay_rate: per-step prototype L2 decay toward zero.
            min_norm: prototype L2 norm below which query returns 0.0.
            threshold: cosine-similarity threshold used externally for release gate.
                       Stored here for reference; the agent reads it from config.
            gain: sigmoid gain applied to cosine similarity before returning.
        """
        self.world_dim = world_dim
        self.ema_alpha = ema_alpha
        self.decay_rate = decay_rate
        self.min_norm = min_norm
        self.threshold = threshold
        self.gain = gain
        self._prototype = [0.0] * world_dim

    def update(
        self,
        z_world: "torch.Tensor",  # noqa: F821
        event_fired: bool,
        sim_mode: bool = False,
    ) -> float:
        """Advance the store by one tick and return the current safety prediction.

        Args:
            z_world: current world latent tensor, shape [world_dim] or [1, world_dim].
            event_fired: True when a MECH-302 relief-completion event occurred this tick.
            sim_mode: if True, returns 0.0 without modifying the prototype (MECH-094).

        Returns:
            safety_prediction in [0, 1]; 0.0 when store is empty or sim_mode is True.
        """
        if sim_mode:
            return 0.0

        # Flatten to 1D python list for arithmetic independence from torch graph.
        vec = z_world
        if hasattr(vec, "detach"):
            vec = vec.detach()
        if hasattr(vec, "squeeze"):
            vec = vec.squeeze(0)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if not isinstance(vec, list):
            vec = list(vec)

        # Per-step decay (forgetting without reinforcement).
        self._prototype = [v * (1.0 - self.decay_rate) for v in self._prototype]

        # EMA update when MECH-302 relief event fired this tick.
        if event_fired:
            # Normalise incoming vector.
            norm = math.sqrt(sum(x * x for x in vec)) + 1e-8
            normed = [x / norm for x in vec]
            alpha = self.ema_alpha
            self._prototype = [
                (1.0 - alpha) * p + alpha * n
                for p, n in zip(self._prototype, normed)
            ]

        return self._query(vec)

    def _query(self, vec: list) -> float:
        """Cosine similarity between vec and prototype, sigmoid-compressed."""
        proto_norm_sq = sum(p * p for p in self._prototype)
        proto_norm = math.sqrt(proto_norm_sq) + 1e-8
        if proto_norm < self.min_norm:
            return 0.0
        vec_norm = math.sqrt(sum(x * x for x in vec)) + 1e-8
        dot = sum(p * v for p, v in zip(self._prototype, vec))
        cos_sim = dot / (proto_norm * vec_norm)
        # Sigmoid with gain: safety_prediction in (0, 1).
        return 1.0 / (1.0 + math.exp(-self.gain * cos_sim))

    def predict(self, z_world: "torch.Tensor") -> float:  # noqa: F821
        """Read-only safety prediction for z_world WITHOUT advancing the store.

        Same cosine->sigmoid query as update()'s return value but with no decay
        and no EMA write -- safe to call mid-tick (e.g. from the SD-059 escape-
        affordance bridge) to read the cue-specific safety prediction for the
        CURRENT post-action state without double-updating the prototype.

        Returns safety_prediction in [0, 1]; 0.0 when the store is empty.
        """
        vec = z_world
        if hasattr(vec, "detach"):
            vec = vec.detach()
        if hasattr(vec, "squeeze"):
            vec = vec.squeeze(0)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if not isinstance(vec, list):
            vec = list(vec)
        return self._query(vec)

    def reset(self) -> None:
        """Clear prototype (call at episode boundary)."""
        self._prototype = [0.0] * self.world_dim
