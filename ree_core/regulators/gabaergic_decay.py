"""
SD-036: GABAergic cross-stream decay regulator.

Architectural commitment (see REE_assembly/docs/architecture/sd_036_gabaergic_decay_regulator.md):

  Decay is NOT a property of each individual stream's update rule but of a regulatory
  layer that touches multiple streams simultaneously. Biological analogue: the
  GABAergic system as a broadly-projecting tonic inhibitory neuromodulator applying
  decay across cortical and subcortical sites in parallel.

  For each registered stream s in S_decay, in the absence of above-threshold input:

      z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone(t))

  where:
    tau_s         per-stream baseline decay rate
    gaba_tone(t)  global multiplier in [0, 2] representing tonic GABAergic level
                  (1.0 baseline; >1.0 = benzo-analog faster decay; <1.0 = withdrawal-
                  analog slower decay; 0.0 = decay suspended).

  Decay is suspended for a stream on any tick where its observed input crossed
  threshold (i.e. salient input drove the update); otherwise decay proceeds.

Initial coverage (per design doc):

  z_harm_s (sensory harm, SD-010)        tau=0.05  (~20-step half-life)
  z_harm_a (affective harm, SD-011)      tau=0.02  (~50-step half-life)
  z_beta   (precision weight, MECH-090)  tau=0.03  (~30-step half-life)

  Drive accumulator (SD-012) is intentionally NOT covered -- the homeostatic-override
  mechanism (separate, V4-or-late-V3) provides drive dynamics.

Non-trainable: pure arithmetic, no gradient flow. The regulator detaches each
stream tensor, scales it out-of-place by exp(-tau * gaba_tone), and writes the
new tensor back onto the LatentState attribute. Out-of-place is required so a
concurrent auxiliary loss head on an encoder output (e.g. SD-018 resource-prox
or SD-011 harm-accum) can still backward without a version-tracking error.
Reset per episode.

Master switch: REEConfig.use_gabaergic_decay (default False) gates instantiation
and tick wiring -- backward-compatibility requirement. With the flag off, agents
behave bit-identically to legacy (no decay applied).

MECH-094: simulation_mode=True ticks return the input unchanged and do not advance
internal counters. Replay / DMN content must not be subject to waking decay
dynamics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import torch


# Stream names (must match LatentState attribute names where applicable).
STREAM_Z_HARM_S = "z_harm"      # SD-010 sensory-discriminative harm latent
STREAM_Z_HARM_A = "z_harm_a"    # SD-011 affective-motivational harm latent
STREAM_Z_BETA = "z_beta"        # MECH-090 precision/affective shared latent


@dataclass
class StreamRegistration:
    """Per-stream decay parameters."""
    name: str
    tau: float
    # Above-threshold input gate: decay is suspended on a tick where the
    # magnitude change between consecutive observations exceeds this value
    # (a salient input drove the update). Defaults to 0.0 -- decay always
    # proceeds unless the caller sets a per-stream threshold.
    input_threshold: float = 0.0


@dataclass
class GABAergicDecayConfig:
    """SD-036 configuration.

    Defaults match the per-stream tau values from the design doc:
      tau_z_harm_s = 0.05  (~20-step half-life)
      tau_z_harm_a = 0.02  (~50-step half-life)
      tau_z_beta   = 0.03  (~30-step half-life)

    gaba_tone defaults to 1.0 (baseline); pharmacological / pathological
    perturbations map to values away from 1.0:
      >1.0  benzo-agonist analog (faster decay)
      <1.0  withdrawal / chronic-stress analog (slower decay)
      0.0   decay suspended

    Initial-coverage flags are True by default once the master switch is on;
    set False per stream to ablate just that stream while leaving the others
    decaying.

    Input thresholds default to 0.0 (always decay) so the simplest baseline
    is "exponential decay each tick toward zero." A non-zero threshold is the
    suspend-on-input gate: |z(t) - z(t-1)| > threshold -> no decay this step.
    """

    # Master flag: enabled when REEConfig.use_gabaergic_decay is True. Held
    # here too so the regulator is independently testable from the agent.
    enabled: bool = True

    # Global tonic multiplier in [0, 2]. 1.0 = baseline; clipping happens at tick.
    gaba_tone: float = 1.0
    gaba_tone_min: float = 0.0
    gaba_tone_max: float = 2.0

    # Per-stream baseline decay rates.
    tau_z_harm_s: float = 0.05
    tau_z_harm_a: float = 0.02
    tau_z_beta: float = 0.03

    # Per-stream coverage flags (used only by `register_default_streams`).
    decay_z_harm_s: bool = True
    decay_z_harm_a: bool = True
    decay_z_beta: bool = True

    # Per-stream input thresholds. Default 0.0 = no suspend-on-input gating.
    # Set to a small positive value (e.g. 1e-3) to suppress decay on ticks
    # where the stream actually received input.
    input_threshold_z_harm_s: float = 0.0
    input_threshold_z_harm_a: float = 0.0
    input_threshold_z_beta: float = 0.0


class GABAergicDecayRegulator:
    """SD-036 GABAergic cross-stream decay regulator.

    Public API:
      register(name, tau, input_threshold=0.0)
        Add a stream to the decay registry. Idempotent (re-registering
        replaces the prior entry). The `name` must match a LatentState
        attribute that holds an Optional[torch.Tensor].

      tick(latent_state, simulation_mode=False) -> LatentState
        For each registered stream present on the LatentState (not None),
        apply per-stream decay z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone)
        unless the suspend-on-input gate fires this tick. The decay is
        applied out-of-place (detach, multiply, swap the attribute) so
        concurrent autograd users of the source tensor (SD-018 resource-
        prox, SD-011 harm-accum aux heads) can still backward without
        a version-tracking error. Returns the same LatentState object
        with the (possibly swapped) stream attributes.

      register_default_streams(config)
        Convenience helper: registers the three SD-036 default streams
        (z_harm, z_harm_a, z_beta) using the config's tau / input_threshold
        defaults and per-stream coverage flags.

      reset()
        Clear per-episode state (last-observed magnitudes for the input
        gate). Stream registrations and gaba_tone are preserved.

      set_gaba_tone(value)
        Update the global multiplier. Clipped to [gaba_tone_min, gaba_tone_max].

    State (per episode):
      _last_norms: Dict[str, float]   per-stream L2 norm at the previous tick,
                                      used by the suspend-on-input gate.
      _n_ticks:    int                diagnostic counter.
      _n_decays:   Dict[str, int]     diagnostic per-stream count of ticks where
                                      decay actually applied (i.e. gate did not fire).
    """

    def __init__(self, config: Optional[GABAergicDecayConfig] = None):
        self.config = config or GABAergicDecayConfig()
        # Clip the configured tone into the bounds at construction time.
        self._gaba_tone: float = self._clip_tone(float(self.config.gaba_tone))

        self._streams: Dict[str, StreamRegistration] = {}
        self._last_norms: Dict[str, float] = {}
        self._n_ticks: int = 0
        self._n_decays: Dict[str, int] = {}

    # -- Configuration helpers --

    def _clip_tone(self, value: float) -> float:
        lo = float(self.config.gaba_tone_min)
        hi = float(self.config.gaba_tone_max)
        return max(lo, min(hi, value))

    def set_gaba_tone(self, value: float) -> None:
        """Set the global GABAergic tonic multiplier."""
        self._gaba_tone = self._clip_tone(float(value))

    @property
    def gaba_tone(self) -> float:
        return float(self._gaba_tone)

    # -- Stream registration --

    def register(
        self,
        name: str,
        tau: float,
        input_threshold: float = 0.0,
    ) -> None:
        """Register a stream for decay.

        Args:
            name: LatentState attribute name (must be Optional[torch.Tensor]).
            tau:  Per-stream baseline decay rate. Must be >= 0; values <= 0 are
                  treated as "no decay" (registration is recorded but tick is a
                  no-op for that stream).
            input_threshold: Magnitude-change threshold above which decay is
                  suspended for that step. Default 0.0 = always decay.
        """
        self._streams[name] = StreamRegistration(
            name=str(name),
            tau=float(tau),
            input_threshold=float(input_threshold),
        )
        # Initialise diagnostics for this stream.
        self._n_decays.setdefault(name, 0)
        self._last_norms.setdefault(name, 0.0)

    def register_default_streams(
        self, config: Optional[GABAergicDecayConfig] = None
    ) -> None:
        """Register the three SD-036 default streams from the config flags.

        Convenience for agent initialisation. Reads the per-stream coverage
        flags (decay_z_harm_s, decay_z_harm_a, decay_z_beta), tau values
        (tau_z_harm_s, tau_z_harm_a, tau_z_beta), and input thresholds.
        """
        cfg = config or self.config
        if cfg.decay_z_harm_s:
            self.register(
                STREAM_Z_HARM_S,
                tau=float(cfg.tau_z_harm_s),
                input_threshold=float(cfg.input_threshold_z_harm_s),
            )
        if cfg.decay_z_harm_a:
            self.register(
                STREAM_Z_HARM_A,
                tau=float(cfg.tau_z_harm_a),
                input_threshold=float(cfg.input_threshold_z_harm_a),
            )
        if cfg.decay_z_beta:
            self.register(
                STREAM_Z_BETA,
                tau=float(cfg.tau_z_beta),
                input_threshold=float(cfg.input_threshold_z_beta),
            )

    @property
    def registered_streams(self) -> List[str]:
        return list(self._streams.keys())

    # -- Per-episode state --

    def reset(self) -> None:
        """Clear per-episode state. Registrations and gaba_tone preserved."""
        self._last_norms = {name: 0.0 for name in self._streams}
        self._n_ticks = 0
        self._n_decays = {name: 0 for name in self._streams}

    # -- Tick: main per-step computation --

    def tick(
        self,
        latent_state,  # LatentState (not annotated here to avoid circular import)
        simulation_mode: bool = False,
    ):
        """Apply per-stream decay to the LatentState in-place and return it.

        Args:
            latent_state: LatentState instance. Stream attributes that match a
                registered name will be decayed when present (not None).
            simulation_mode: MECH-094 hypothesis-tag equivalent. True -> return
                the input unchanged and do not advance counters / norm cache.
                Replay / DMN content must not be subject to waking decay.

        Returns:
            The same LatentState object with decayed stream tensors.
        """
        if not self.config.enabled:
            return latent_state

        if simulation_mode:
            return latent_state

        self._n_ticks += 1

        tone = self._gaba_tone
        # If tone is exactly zero, decay is suspended globally (corresponds
        # biologically to total GABAergic blockade -- nothing returns to baseline).
        if tone <= 0.0:
            # Still update the last-norms so a subsequent non-zero tone tick
            # has a sensible baseline.
            for name in self._streams:
                z = getattr(latent_state, name, None)
                if z is not None:
                    self._last_norms[name] = float(z.detach().norm().item())
            return latent_state

        for name, reg in self._streams.items():
            z = getattr(latent_state, name, None)
            if z is None:
                continue
            tau = float(reg.tau)
            if tau <= 0.0:
                continue

            cur_norm = float(z.detach().norm().item())
            prev_norm = float(self._last_norms.get(name, 0.0))
            delta = abs(cur_norm - prev_norm)

            # Suspend-on-input gate: if the magnitude change exceeded the
            # per-stream threshold, the stream received salient input this
            # step -- skip decay (the input drives the update).
            suspend = (
                reg.input_threshold > 0.0 and delta > float(reg.input_threshold)
            )

            if not suspend:
                # z_s(t+1) = z_s(t) * exp(-tau * gaba_tone)
                factor = math.exp(-tau * tone)
                # Out-of-place to preserve autograd version tracking on the
                # source tensor: encoder outputs may participate in concurrent
                # auxiliary losses (e.g. SD-018 resource-prox / SD-011 harm-
                # accum heads) and an in-place mul_() on a graph-bearing
                # encoder output triggers RuntimeError on backward(). The
                # regulator is a non-trainable arithmetic substrate, so we
                # detach + multiply, then write the new tensor back onto the
                # LatentState attribute. Downstream consumers (E3 harm_eval,
                # AIC, dACC) take the latent slice directly, so the swap is
                # transparent.
                with torch.no_grad():
                    new_z = z.detach() * factor
                setattr(latent_state, name, new_z)
                self._n_decays[name] = self._n_decays.get(name, 0) + 1
                # Cache the post-decay norm as the new baseline.
                self._last_norms[name] = float(new_z.norm().item())
            else:
                # Decay suspended; cache the as-observed norm as the new baseline.
                self._last_norms[name] = cur_norm

        return latent_state

    # -- Diagnostics --

    @property
    def diagnostics(self) -> Dict[str, object]:
        return {
            "n_ticks": int(self._n_ticks),
            "n_decays": {k: int(v) for k, v in self._n_decays.items()},
            "gaba_tone": float(self._gaba_tone),
            "registered_streams": list(self._streams.keys()),
        }
