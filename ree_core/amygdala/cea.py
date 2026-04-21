"""
CeAAnalog -- SD-035 central-amygdala-analog (fast salience classification ->
mode prior, fast subcortical priming).

CeA is the low-latency / scalar subdivision of the amygdala analogue. Where
BLA (ree_core/amygdala/bla.py) writes multi-field structured output to the
hippocampus, CeA writes two scalars into the SalienceCoordinator:

  (1) MECH-046 mode_prior -- pre-softmax additive log-odds bias biasing the
      coordinator's operating-mode distribution toward harm/threat modes on
      fast threat detection. Fires within 1-2 sim steps (~75 ms biological;
      Mendez-Bertolo 2016) when a low-frequency projection of z_harm_a
      crosses theta_cea_fast. Distinct from AICAnalog urgency (SD-032c):
      AIC modulates mode-SWITCH threshold; CeA mode_prior biases mode
      SELECTION.

  (2) MECH-074c fast_prime -- scalar candidate-prior pulse distinct from
      mode_prior. Biases the salience weighting over candidate
      trajectories/actions BEFORE cortical confirmation arrives. Override
      window: cortical signals can flip fast_prime within 5-10 sim steps
      (~300-400 ms; Pessoa & Adolphs 2010 many-roads framing). Unconfirmed
      fast_prime decays with tau = 3-5 steps.

Architectural scope (SD-035 v3 minimum-viable):

  Inputs (per tick, called once per agent.sense() after z_harm_a is
  produced):
    z_harm_a         torch.Tensor [z_harm_a_dim]  SD-011 affective stream
    cue_features     Optional[torch.Tensor]       reserved for future raw
                       exteroceptive cue input (Mendez-Bertolo 2016 fast
                       pulvinar/subcortical route); None in V3.
    cortical_confirmation
                     Optional[float]              scalar in [0, 1]
                       indicating whether cortical analysis has confirmed
                       or contradicted the fast classification. 1.0 = full
                       confirmation (fast_prime held); 0.0 = no signal
                       (fast_prime decays); negative values (if callers
                       pass them) are clamped -- no unilateral override
                       from the fast path.
    escapability_hint Optional[float]             Q-036 placeholder. V3
                       no-op: accepted and stored in the output so MECH-219
                       can consume it without a module-interface refactor.
                       See docs/architecture/sd_035_amygdala_analog.md
                       section "Q-036 follow-up".
    simulation_mode  bool                         MECH-094 gate. True ->
                       return zeroed CeAOutput; do not update baselines
                       or decay counters.

  Outputs (CeAOutput):
    mode_prior       float   pre-softmax additive log-odds bias to be
                             written onto SalienceCoordinator.
                             affinity_weights for the harm-relevant modes
                             via salience.update_signal("cea_mode_prior",
                             value). Zero when |LowFreq(z_harm_a)| <
                             fast_route_threshold AND no residual window
                             remains. Bounded by
                             mode_prior_log_odds_max.
    fast_prime       float   scalar candidate-prior pulse. Zero at rest;
                             jumps to fast_prime_amplitude on threshold
                             crossing; decays exponentially with tau =
                             fast_prime_decay_tau_steps unless a cortical
                             confirmation signal arrives inside the
                             override window.
    urgency_fire     bool    diagnostic: True on the tick where the fast
                             gate crossed threshold this step. Distinct
                             from AICAnalog.urgency_signal (which is a
                             different biological circuit).
    escapability_hint Optional[float]
                             Q-036 pass-through. No-op in V3.
    low_freq_magnitude float  diagnostic: |LowFreq(z_harm_a)| this tick.
    steps_since_fire int     diagnostic: ticks since the last fast-route
                             fire (resets to 0 on fire). Bounded.
    override_window_remaining int
                             diagnostic: ticks of override window still
                             open for cortical confirmation.

Selectivity constraint (synthesis.md): CeA must fire on harm-affective
valence, NOT generic arousal. The low-frequency projection is the
magnitude L1 norm of z_harm_a -- because z_harm_a is the SD-011 affective
stream, magnitude there IS affective valence load. A BLAAnalog-style
retrieval of arousal from a richer latent would re-introduce content
specificity and double-count what BLA already does.

Override principle (synthesis.md): CeA biases the coordinator; it does
NOT unilaterally set mode. The fast_prime and mode_prior values are
additive log-odds biases; cortical AIC/dACC contributions pass through
the same coordinator and can dominate. |mode_prior| is bounded by
mode_prior_log_odds_max (default 0.8) which must be <= the AIC/dACC
ceiling so CeA never over-rules cortex.

Non-trainable: pure arithmetic over scalars. No gradient flow. Single EMA
for cortical-confirmation tracking (optional); decay counters for the
fast_prime pulse and override window.

MECH-094: simulation_mode=True (replay / rehearsal) returns a zeroed
CeAOutput with no state change so downstream consumers do not
accidentally apply threat biases to simulated rollouts.

Falsification signatures (per sub-claim):

  MECH-046: if EXQ-A (CeA mode-prior ablation) shows time-to-mode-switch
  on threat-cue onset is identical between amygdala-OFF and CeA-ON,
  the fast-route arithmetic has failed (either threshold never fires,
  or bias is too small to influence coordinator).

  MECH-074c: if the fast_prime amplitude is NOT larger than the first
  cortical AIC/dACC log-odds contribution within the override window
  (5-10 sim steps), the fast prime is not functionally a 'prime' -- it
  arrives too weak or too late to prefigure cortical output.

Biological grounding:
  Mendez-Bertolo et al. 2016 (Nat Neurosci) -- fast pulvinar-amygdala
  route with ~74 ms latency for fearful-face responses, distinct from
  cortical visual routes.
  Pessoa & Adolphs 2010 (Nat Rev Neurosci) -- many-roads framing:
  multiple parallel pathways feed amygdala; no single 'low road';
  cortical confirmation must be able to override in <=400 ms.
  Craig 2009 + Menon & Uddin 2010 -- AIC/dACC cortical route for
  comparison (takes ~5-10 sim steps / ~400 ms).

See CLAUDE.md: SD-035. Spec:
REE_assembly/docs/architecture/sd_035_amygdala_analog.md
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CeAConfig:
    """Configuration for SD-035 CeA-analog.

    Defaults chosen from synthesis.md:
      - fast_route_threshold 0.5 on |LowFreq(z_harm_a)|
      - mode_prior magnitude ceiling 0.8 (below AIC/dACC log-odds ceiling)
      - fast_prime decay tau 4 steps (~150-250 ms biological half-life)
      - override window 8 steps (~300-400 ms)

    All defaults produce backward-compatible no-op behaviour at rest:
    below threshold, mode_prior and fast_prime are exactly 0.0 and no
    state change occurs.
    """

    # -- Fast route gate (MECH-046) --

    # Threshold on the low-frequency / magnitude projection of z_harm_a.
    # Below this, no fast-route fire. Synthesis default 0.5.
    fast_route_threshold: float = 0.5

    # Use L1 magnitude (low-frequency / coarse projection) of z_harm_a
    # for the fast-route gate. Trades specificity for speed per
    # Mendez-Bertolo 2016. Set False to drive the gate on a caller-
    # supplied scalar for ablation studies.
    fast_route_input_is_lowfreq: bool = True

    # -- Mode prior (MECH-046) --

    # Maximum absolute value of the pre-softmax additive log-odds bias
    # emitted as mode_prior. MUST be <= the AIC/dACC log-odds ceiling so
    # CeA never over-rules cortex. Synthesis default 0.8.
    mode_prior_log_odds_max: float = 0.8

    # Emission scale: fraction of the above cap emitted at threshold.
    # 1.0 means mode_prior jumps straight to mode_prior_log_odds_max on
    # first crossing. Below 1.0 scales the fast-route jump linearly in
    # (low_freq_mag - threshold).
    mode_prior_gain: float = 1.0

    # Gating placement: pre-softmax additive log-odds, NOT post-softmax
    # multiplicative. Held as a config flag for documentation / ablation.
    # Flipping this to False is a named failure signature (synthesis).
    pre_softmax_additive: bool = True

    # -- Fast prime pulse (MECH-074c) --

    # Peak amplitude of the fast_prime pulse on threshold crossing.
    # Scaled on the same log-odds axis as mode_prior; bounded by
    # mode_prior_log_odds_max so CeA does not over-rule cortex via the
    # candidate-prior path either.
    fast_prime_amplitude: float = 0.6

    # Decay half-life of the fast_prime pulse in sim steps when cortical
    # confirmation is absent. Synthesis range 3-5; default 4
    # (~150-250 ms biological half-life at 100 ms per step).
    fast_prime_decay_tau_steps: int = 4

    # Override window: number of sim steps cortical confirmation has to
    # confirm / override the fast pulse. After this window, the pulse
    # decays toward baseline regardless. Synthesis range 5-10;
    # default 8 (~300-400 ms biological). Cortical confirmation arriving
    # inside the window extends the pulse; outside, it is ignored (the
    # pulse has already decayed).
    fast_prime_override_window_steps: int = 8

    # Weight on cortical_confirmation input during the override window.
    # 0.0 = cortex cannot sustain the pulse (pure fast-route timing);
    # 1.0 = full cortical confirmation holds the pulse at full
    # amplitude for the remainder of the override window. Intermediate
    # values linearly blend.
    cortical_confirmation_weight: float = 1.0


@dataclass
class CeAOutput:
    """Per-tick CeAAnalog output."""

    mode_prior: float = 0.0
    fast_prime: float = 0.0
    urgency_fire: bool = False
    escapability_hint: Optional[float] = None
    low_freq_magnitude: float = 0.0
    steps_since_fire: int = 0
    override_window_remaining: int = 0


class CeAAnalog:
    """SD-035 central-amygdala-analog fast-route / mode-prior module.

    Stateful:
      _fast_prime_value       current fast_prime amplitude (decays each
                              tick when outside override window or when
                              cortical_confirmation is absent).
      _steps_since_fire       ticks since the last fast-route fire.
      _override_remaining     ticks of override window still open.
      _n_ticks                diagnostic counter.
      _n_fires                diagnostic counter of fast-route fires.

    No gradient flow. Reset per episode via .reset().
    """

    def __init__(self, config: Optional[CeAConfig] = None):
        self.config = config or CeAConfig()

        self._fast_prime_value: float = 0.0
        self._steps_since_fire: int = 10_000_000  # effectively +inf
        self._override_remaining: int = 0

        self._last_mode_prior: float = 0.0
        self._last_fast_prime: float = 0.0
        self._last_low_freq_mag: float = 0.0

        self._n_ticks: int = 0
        self._n_fires: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear per-episode state. Call on env.reset()."""
        self._fast_prime_value = 0.0
        self._steps_since_fire = 10_000_000
        self._override_remaining = 0
        self._last_mode_prior = 0.0
        self._last_fast_prime = 0.0
        self._last_low_freq_mag = 0.0

    # -- Tick: main per-step computation --

    def tick(
        self,
        z_harm_a: torch.Tensor,
        cue_features: Optional[torch.Tensor] = None,
        cortical_confirmation: Optional[float] = None,
        escapability_hint: Optional[float] = None,
        simulation_mode: bool = False,
    ) -> CeAOutput:
        """Compute CeAOutput for this step.

        Args:
            z_harm_a: SD-011 affective stream latent [batch=1,
                z_harm_a_dim] or [z_harm_a_dim]. CeA reads a
                low-frequency / magnitude projection (L1 norm by
                default). Direction is not consumed here.
            cue_features: Optional raw exteroceptive cue tensor. V3 no-op
                placeholder for Mendez-Bertolo 2016 pulvinar / subcortical
                fast-route input. Reserved; not used in V3.
            cortical_confirmation: Optional scalar in [0, 1]. None (default)
                treated as 0 (no cortical signal -- fast_prime decays per
                its time constant). Values outside [0, 1] are clipped.
            escapability_hint: Q-036 placeholder. Stored on the output for
                MECH-219 consumers; V3 no-op.
            simulation_mode: MECH-094 hypothesis_tag equivalent. True -> do
                not update internal state, return zeroed output.

        Returns:
            CeAOutput with mode_prior, fast_prime, urgency_fire, and
            diagnostics.
        """
        self._n_ticks += 1

        if simulation_mode:
            # MECH-094 gate: replay / simulation ticks do not update
            # fast-route state. Return explicit no-op output.
            return CeAOutput(
                mode_prior=0.0,
                fast_prime=0.0,
                urgency_fire=False,
                escapability_hint=escapability_hint,
                low_freq_magnitude=0.0,
                steps_since_fire=int(self._steps_since_fire),
                override_window_remaining=int(self._override_remaining),
            )

        # --- 1. Low-frequency projection of z_harm_a ---

        if self.config.fast_route_input_is_lowfreq:
            # L1 magnitude is the coarse / low-frequency summary. Faster
            # than L2 (no sqrt), aligns with Mendez-Bertolo's coarse-
            # feature fast-route framing.
            low_freq_mag = float(
                torch.linalg.vector_norm(
                    z_harm_a.detach().flatten(), ord=1
                ).item()
            )
            # Normalise by dimensionality so threshold default 0.5 is
            # comparable across different z_harm_a_dim settings.
            n = int(z_harm_a.detach().flatten().shape[0])
            if n > 0:
                low_freq_mag = low_freq_mag / float(n)
        else:
            # Caller-supplied path: treat z_harm_a as a pre-computed
            # scalar (ablation convenience).
            low_freq_mag = float(
                torch.linalg.vector_norm(
                    z_harm_a.detach().flatten(), ord=2
                ).item()
            )
        self._last_low_freq_mag = low_freq_mag

        # --- 2. Fast-route gate (MECH-046) ---

        thr = float(self.config.fast_route_threshold)
        cap = float(self.config.mode_prior_log_odds_max)
        urgency_fire = low_freq_mag > thr

        if urgency_fire:
            # Fire: reset the fast-path counters.
            self._steps_since_fire = 0
            self._override_remaining = int(
                self.config.fast_prime_override_window_steps
            )
            self._n_fires += 1
            # Scale mode_prior and fast_prime in proportion to how far
            # over threshold the input is, clipped to cap.
            over = low_freq_mag - thr
            over_norm = over / max(1e-6, cap)  # use cap as scale
            gain = float(self.config.mode_prior_gain)
            raw_mode_prior = over_norm * cap * gain
            mode_prior = float(
                max(-cap, min(cap, raw_mode_prior))
            )
            # fast_prime pulse: immediate jump to configured amplitude,
            # scaled the same way as mode_prior to keep ratio stable.
            amp = float(self.config.fast_prime_amplitude)
            amp_bounded = min(amp, cap)  # never over-rule cortex via cap
            raw_pulse = over_norm * amp_bounded
            new_pulse = float(max(-cap, min(cap, raw_pulse)))
            # Take the larger-in-magnitude of any existing pulse and the
            # new one -- re-firing during a decay should not suppress an
            # earlier higher pulse.
            if abs(new_pulse) >= abs(self._fast_prime_value):
                self._fast_prime_value = new_pulse
        else:
            # No fire this step: advance counters.
            self._steps_since_fire = min(
                self._steps_since_fire + 1, 10_000_000
            )
            if self._override_remaining > 0:
                self._override_remaining -= 1
            # mode_prior is emission-only on active fire + short window
            # tail. Implement as a same-shape decay keyed off
            # fast_prime_value magnitude so the coordinator does not get
            # a mode-prior bias long after the pulse has decayed.
            # Tail magnitude scales to cap when the pulse is at its peak.
            amp_peak = max(1e-6, float(self.config.fast_prime_amplitude))
            tail_scale = self._fast_prime_value / amp_peak
            mode_prior = float(max(-cap, min(cap, tail_scale * cap)))

        # --- 3. fast_prime decay (MECH-074c) ---

        # Update fast_prime value for decay only when we did NOT just
        # fire (fire path already set _fast_prime_value above).
        if not urgency_fire and self._fast_prime_value != 0.0:
            # Cortical-confirmation handling:
            if cortical_confirmation is not None:
                cc = float(max(0.0, min(1.0, float(cortical_confirmation))))
            else:
                cc = 0.0

            in_window = self._override_remaining > 0
            weight = float(self.config.cortical_confirmation_weight)

            if in_window and cc > 0.0:
                # Inside override window and cortex is confirming: blend
                # toward holding the pulse. Held amplitude = cc * weight
                # * current magnitude (preserves sign).
                hold_factor = cc * weight
                # Pure decay with reduced rate proportional to hold_factor.
                tau = max(1.0, float(self.config.fast_prime_decay_tau_steps))
                decay_rate = (1.0 - hold_factor) * (1.0 / tau)
                decay_rate = max(0.0, min(1.0, decay_rate))
                self._fast_prime_value = float(
                    self._fast_prime_value * (1.0 - decay_rate)
                )
            else:
                # Outside override window OR no cortical confirmation:
                # exponential decay with tau_steps half-life.
                tau = max(1.0, float(self.config.fast_prime_decay_tau_steps))
                # Decay per step as 0.5 ** (1 / tau) -> half-life tau.
                self._fast_prime_value = float(
                    self._fast_prime_value * (0.5 ** (1.0 / tau))
                )

            # Zero-snap small residuals (avoid persistent tiny biases).
            if abs(self._fast_prime_value) < 1e-6:
                self._fast_prime_value = 0.0

        fast_prime = float(self._fast_prime_value)

        # Recompute mode_prior tail after decay so the two scalars
        # remain consistent within a tick.
        if not urgency_fire:
            amp_peak = max(1e-6, float(self.config.fast_prime_amplitude))
            tail_scale = self._fast_prime_value / amp_peak
            mode_prior = float(max(-cap, min(cap, tail_scale * cap)))

        self._last_mode_prior = mode_prior
        self._last_fast_prime = fast_prime

        return CeAOutput(
            mode_prior=float(mode_prior),
            fast_prime=float(fast_prime),
            urgency_fire=bool(urgency_fire),
            escapability_hint=escapability_hint,
            low_freq_magnitude=float(low_freq_mag),
            steps_since_fire=int(self._steps_since_fire),
            override_window_remaining=int(self._override_remaining),
        )

    # -- Read-only accessors --

    @property
    def mode_prior(self) -> float:
        return float(self._last_mode_prior)

    @property
    def fast_prime(self) -> float:
        return float(self._last_fast_prime)

    @property
    def low_freq_magnitude(self) -> float:
        return float(self._last_low_freq_mag)

    @property
    def diagnostics(self) -> dict:
        return {
            "n_ticks": int(self._n_ticks),
            "n_fires": int(self._n_fires),
            "steps_since_fire": int(self._steps_since_fire),
            "override_window_remaining": int(self._override_remaining),
        }
