"""
BLAAnalog -- SD-035 basolateral-amygdala-analog (encoding gain, retrieval
bias, remap on PE spike).

BLA is the slow / content-processing subdivision of the amygdala analogue.
Where CeA (ree_core/amygdala/cea.py) operates on low-latency scalar
channels into the SalienceCoordinator, BLA operates on the hippocampal
map:

  (1) MECH-074a encoding_gain -- arousal-dependent inverted-U multiplier
      on HippocampalModule write strength. At ||z_harm_a|| ~ 0.4 the gain
      is 1.0 (baseline); at ~ 0.7 the gain peaks at encoding_gain_max
      (default 2.5, Roozendaal & McGaugh 2011); above the peak the gain
      falls off (inverted-U). A post-event decay window extends the
      elevated gain for ~30 min biological (~18000 sim steps @ 100 ms
      per step) with a ~6 min half-life.

  (2) MECH-074b retrieval_bias -- a CONTENT-SELECTIVE per-trace weight
      VECTOR (NOT a scalar gain). This is a hard design decision:
      LaBar & Cabeza 2006 rule out a uniform retrieval boost because
      it cannot reproduce the central/peripheral dissociation that is
      the hallmark of amygdala-lesion studies. The arithmetic is
      w_i = 1 + alpha * arousal_tag_i with alpha in [0.3, 1.0].
      BLAAnalog.tick emits both (a) an encoding-time arousal_tag
      scalar that callers write onto each hippocampal trace, and (b)
      a per-trace retrieval_bias vector consulted on readout.

  (3) MECH-074d remap_signal -- step-function remap on harm-PE spike
      with a predictor-attribution gate. remap_signal fires when
      ||z_harm_a - z_harm_a_pred|| exceeds
      remap_pe_sigma_threshold standard deviations of the running
      harm-PE distribution AND an attribution head flags at least one
      candidate latent code. Both conditions required. Per-code binary
      shape; approximately one-third of predictor-candidate codes are
      perturbed on fire (Moita 2004 ~30-35% overwrite on training-box
      correlation drop 0.57 -> 0.39).

  Architectural scope (SD-035 v3 minimum-viable):

  Inputs (per tick, called once per agent.sense()):
    z_harm_a         torch.Tensor [z_harm_a_dim]  SD-011 affective stream
    z_harm_a_pred    torch.Tensor [z_harm_a_dim]  ARC-033 / ARC-058
                       one-step-ahead prediction; None on first tick
                       (remap_signal cannot fire without a prediction)
    candidate_code_contributions
                     Optional dict[int, float] from the attribution step
                       in the caller. If None, remap cannot fire
                       (conservative default; Moita 2004 attribution
                       gate requirement).

  Outputs (BLAOutput):
    encoding_gain    float   scalar >= 1.0; multiplier on HippocampalModule
                             write strength. Backward-compat contract:
                             with use_bla_analog=False, this module is
                             never instantiated; with it True and
                             z_harm_a below arousal_threshold, encoding_gain
                             is exactly 1.0 (no-op).
    arousal_tag      float   per-trace tag to be written at encoding time
                             onto each new hippocampal trace (read by
                             retrieval_bias at readout time; LaBar 2006
                             requires tag-at-encoding).
    retrieval_bias   torch.Tensor [num_traces] or None
                             per-trace weight vector w_i = 1 + alpha *
                             arousal_tag_i. Computed lazily -- only when
                             hippocampal_context carries arousal_tag
                             values from prior encoding ticks.
    remap_signal     Dict[int, float] or {}
                             per-code remap amplitudes. Empty dict when
                             PE is below threshold or attribution produces
                             no candidates. Keys are code indices from the
                             attribution input.
    pe_magnitude     float   diagnostic: magnitude of the current harm-PE
    pe_baseline_std  float   diagnostic: running std-dev of harm-PE
    encoding_window_steps_remaining  int  diagnostic

Non-trainable: pure arithmetic over scalars + tensors. No gradient flow.
Running mean + running variance on harm-PE; post-event decay counter on
encoding_gain window.

MECH-094: not applicable to the encoding_gain / retrieval_bias paths (they
modulate waking write / read strength). remap_signal is also a waking
signal (fired on live PE spikes, not replay) so MECH-094 does not gate it
either. Callers that invoke tick() from simulation/replay paths must
supply hypothesis_tag via the simulation_mode flag to skip ALL writes.

Falsification signatures (per sub-claim):

  MECH-074a: if EXQ-B (BLA encoding + remap) shows threat-context recall
  does NOT improve under BLA-modulated gain relative to gain=1, OR if
  neutral recall is harmed by gain > 1, the encoding-gain arithmetic is
  mis-specified.

  MECH-074b: if central/gist items are NOT retrieved preferentially over
  peripheral/neutral items under BLA retrieval-bias ON, or if a uniform
  retrieval boost produces the same behaviour (scalar-equivalent), the
  content-selective form has collapsed.

  MECH-074d: if remap_signal fires on sub-threshold PE, OR perturbs
  untagged codes uniformly (attribution gate broken), OR amplitude is
  wholesale (>>33% of codes), the remap logic is mis-specified.

Biological grounding:
  Roozendaal & McGaugh 2011 (Behav Neurosci) -- inverted-U BLA
  modulation of hippocampal LTP / consolidation; 30-min post-event
  window.
  McGaugh 2004 (Annu Rev Neurosci) -- canonical BLA-memory modulation
  review.
  LaBar & Cabeza 2006 (Nat Rev Neurosci) -- retrieval bias is
  content-selective (central/peripheral dissociation); BLA-MTL
  connectivity grows with trace age.
  Dolcos, LaBar & Cabeza 2004 -- fMRI evidence for BLA-MTL retrieval
  bias.
  Nader, Schafe & LeDoux 2000 (Nature) -- reconsolidation necessity
  for remap.
  Moita et al 2004 (J Neurosci) -- place-cell remapping contextual-vs-
  auditory dissociation (Z = -1.36 vs -0.34, p = 0.02) mandates the
  attribution gate.

See CLAUDE.md: SD-035. Spec:
REE_assembly/docs/architecture/sd_035_amygdala_analog.md
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class BLAConfig:
    """Configuration for SD-035 BLA-analog.

    All defaults produce backward-compatible no-op behaviour at rest:
    with z_harm_a magnitude below arousal_threshold_on, encoding_gain
    is exactly 1.0 and no arousal_tag is emitted. Under threat, the
    inverted-U kicks in and the post-event window decays back to 1.0.
    """

    # -- Encoding gain (MECH-074a) --

    # Peak multiplier on HippocampalModule write strength at the peak
    # of the inverted-U. 2.5x matches Roozendaal & McGaugh 2011
    # canonical estimate.
    encoding_gain_max: float = 2.5

    # Baseline gain (below threshold). Explicitly 1.0 -- BLA analogue
    # does not SUPPRESS memory writes below threshold; it only
    # enhances them at elevated arousal. Inverted-U means the gain
    # returns to 1.0 above the peak (very-high arousal degrades
    # encoding).
    encoding_gain_floor: float = 1.0

    # Arousal magnitude (||z_harm_a||_2) at which encoding_gain first
    # rises above floor. Below this: flat 1.0 (no-op). Per synthesis
    # default 0.4.
    arousal_threshold_on: float = 0.4

    # Peak of the inverted-U. At arousal_peak, encoding_gain =
    # encoding_gain_max. Above this, the gain falls off symmetrically
    # toward encoding_gain_floor. Per synthesis default 0.7.
    arousal_peak: float = 0.7

    # Post-event decay window length in sim steps. After z_harm_a
    # crosses arousal_threshold_on, encoding_gain stays elevated for
    # this many steps even after arousal drops, with a half-life
    # window_half_life_steps. 18000 steps ~= 30 min biological at
    # 100 ms per step (Roozendaal 2011).
    window_steps: int = 18000

    # Exponential decay half-life within the post-event window.
    # 3600 steps ~= 6 min biological (synthesis default).
    window_half_life_steps: int = 3600

    # -- Retrieval bias (MECH-074b) --

    # Per-trace weight scaling: w_i = 1 + alpha * arousal_tag_i.
    # Midpoint of 0.3-1.0 (LaBar & Cabeza 2006).
    retrieval_bias_alpha: float = 0.6

    # Optional zero-sum compensation: untagged traces are suppressed
    # by this fraction to compensate for elevated retrieval of tagged
    # traces. Default 0.0 (simpler; enable 0.1-0.3 for the full
    # zero-sum form). LaBar 2006 notes the effect is optional and
    # sometimes omitted in simpler implementations.
    retrieval_bias_compensation: float = 0.0

    # Required-by-design: LaBar 2006 mandates that the arousal_tag be
    # written AT ENCODING, not reconstructed at retrieval. Exposed as
    # a config flag for ablation studies (setting False reproduces the
    # named failure signature -- scalar retrieval bias with no tag).
    retrieval_tag_at_encoding: bool = True

    # -- Remap signal (MECH-074d) --

    # PE threshold in units of running std-dev. remap fires when
    # ||z_harm_a - z_harm_a_pred|| > threshold * running_std(PE).
    # 1.0 is the synthesis default (one SD above running mean).
    remap_pe_sigma_threshold: float = 1.0

    # EMA alpha for running harm-PE mean and variance. 0.02 matches
    # MECH-205 pe_ema_alpha convention (~50-step window).
    remap_pe_ema_alpha: float = 0.02

    # Initial running std-dev estimate. Too-small value causes
    # spurious early remaps before the EMA converges; a reasonable
    # starting interval. Absolute scale depends on z_harm_a
    # magnitude; callers can tune if z_harm_a_dim is unusually large.
    remap_pe_std_init: float = 0.1

    # Fraction of predictor-candidate codes to perturb on remap fire.
    # Moita 2004: training-box correlation drop 0.57 -> 0.39 implies
    # ~30-35% overwrite. 0.33 is the synthesis default.
    remap_code_fraction: float = 0.33

    # Hard requirement per Moita 2004 (contextual-vs-auditory
    # dissociation). Set False ONLY for deliberate broadcast-remap
    # ablation.
    remap_requires_attribution: bool = True


@dataclass
class BLAOutput:
    """Per-tick BLAAnalog output."""

    encoding_gain: float = 1.0
    arousal_tag: float = 0.0
    retrieval_bias: Optional[torch.Tensor] = None
    remap_signal: Dict[int, float] = field(default_factory=dict)
    pe_magnitude: float = 0.0
    pe_baseline_std: float = 0.0
    encoding_window_steps_remaining: int = 0


class BLAAnalog:
    """SD-035 basolateral-amygdala-analog encoding/retrieval/remap module.

    Stateful:
      _pe_mean              running EMA of harm-PE magnitude
      _pe_var               running EMA of harm-PE squared magnitude
      _pe_std               sqrt(var); lazily updated on tick
      _window_onset_step    sim step at which the last
                            arousal_threshold_on crossing occurred;
                            None if no crossing has been seen yet.
      _n_ticks              diagnostic counter (episode-local)
      _n_remap              diagnostic counter of remap_signal fires
      _n_gain_elevations    diagnostic counter of ticks with
                            encoding_gain > 1

    No gradient flow. Reset per episode via .reset().
    """

    def __init__(self, config: Optional[BLAConfig] = None):
        self.config = config or BLAConfig()

        self._pe_mean: float = 0.0
        self._pe_var: float = float(self.config.remap_pe_std_init) ** 2
        self._pe_std: float = float(self.config.remap_pe_std_init)

        self._window_onset_step: Optional[int] = None
        self._last_encoding_gain: float = float(self.config.encoding_gain_floor)
        self._last_arousal_tag: float = 0.0
        self._last_pe_magnitude: float = 0.0

        self._n_ticks: int = 0
        self._n_remap: int = 0
        self._n_gain_elevations: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear per-episode state. Call on env.reset()."""
        self._pe_mean = 0.0
        self._pe_var = float(self.config.remap_pe_std_init) ** 2
        self._pe_std = float(self.config.remap_pe_std_init)
        self._window_onset_step = None
        self._last_encoding_gain = float(self.config.encoding_gain_floor)
        self._last_arousal_tag = 0.0
        self._last_pe_magnitude = 0.0

    # -- Tick: main per-step computation --

    def tick(
        self,
        z_harm_a: torch.Tensor,
        z_harm_a_pred: Optional[torch.Tensor] = None,
        candidate_code_contributions: Optional[Dict[int, float]] = None,
        arousal_tags_in_context: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        simulation_mode: bool = False,
    ) -> BLAOutput:
        """Compute BLAOutput for this step.

        Args:
            z_harm_a: SD-011 affective stream latent [batch=1, z_harm_a_dim]
                or [z_harm_a_dim]. BLA reads the 2-norm magnitude; the
                direction of z_harm_a is not consumed here.
            z_harm_a_pred: One-step-ahead prediction of z_harm_a from the
                ARC-033 / ARC-058 forward model. If None, remap_signal
                cannot fire (no PE available).
            candidate_code_contributions: Dict[int, float] from the
                caller's attribution step. Each entry is (code_index,
                contribution_score). Only codes with attribution above
                zero are candidates for remap. If None AND
                remap_requires_attribution=True: remap_signal cannot
                fire (MECH-074d attribution gate per Moita 2004).
            arousal_tags_in_context: Optional [num_traces] tensor of
                per-trace arousal tags previously written at encoding
                time by earlier BLAAnalog.tick() calls. When provided,
                retrieval_bias is computed as
                w_i = 1 + alpha * arousal_tags_in_context[i]. None
                (default) means the retrieval-bias readout path is
                not yet wired in the caller.
            step_index: Sim step index for the post-event window
                countdown. If None, the internal tick counter is used
                (which assumes ticks are emitted at every sim step).
            simulation_mode: MECH-094 hypothesis_tag equivalent. When
                True, this call is a replay / simulation tick and MUST
                NOT update running statistics OR emit arousal_tag
                writes. Returns a zeroed BLAOutput (encoding_gain=1.0,
                remap_signal={}, retrieval_bias=None) so callers do
                not accidentally apply amygdala modulation to simulated
                rollouts.

        Returns:
            BLAOutput with encoding_gain, arousal_tag, retrieval_bias,
            remap_signal, and diagnostics.
        """
        self._n_ticks += 1

        if simulation_mode:
            # MECH-094 gate: replay / simulation ticks do not write
            # arousal tags or update running statistics. Return
            # explicit no-op output so callers get a consistent shape.
            return BLAOutput(
                encoding_gain=1.0,
                arousal_tag=0.0,
                retrieval_bias=None,
                remap_signal={},
                pe_magnitude=0.0,
                pe_baseline_std=float(self._pe_std),
                encoding_window_steps_remaining=0,
            )

        # --- 1. Encoding gain (MECH-074a) ---

        # Compute scalar arousal magnitude.
        z_norm = float(torch.linalg.norm(z_harm_a.detach().flatten()).item())

        # Track window onset: if we cross the threshold, reset the
        # post-event countdown. Below threshold, the countdown decays
        # via the exponential half-life model.
        now = int(step_index) if step_index is not None else self._n_ticks
        if z_norm >= float(self.config.arousal_threshold_on):
            self._window_onset_step = now

        steps_remaining = 0
        window_decay = 0.0  # fraction of post-event-window contribution
        if self._window_onset_step is not None:
            elapsed = now - int(self._window_onset_step)
            if elapsed <= int(self.config.window_steps):
                steps_remaining = int(self.config.window_steps) - elapsed
                # Half-life exponential decay on the post-event window.
                half_life = max(1.0, float(self.config.window_half_life_steps))
                window_decay = float(0.5 ** (elapsed / half_life))

        # Inverted-U over arousal magnitude: peaks at arousal_peak,
        # falls off symmetrically. Expressed as a piecewise linear
        # interp from threshold_on (-> floor) to peak (-> max), then
        # peak to (2*peak - threshold_on) -> floor, clipped to floor
        # above.
        peak = float(self.config.arousal_peak)
        thr = float(self.config.arousal_threshold_on)
        gmax = float(self.config.encoding_gain_max)
        gfloor = float(self.config.encoding_gain_floor)

        if z_norm < thr:
            # Below threshold: use the post-event window contribution
            # only.
            immediate_gain = gfloor
        elif z_norm <= peak:
            # Rising arm of the inverted-U.
            frac = (z_norm - thr) / max(1e-6, peak - thr)
            immediate_gain = gfloor + frac * (gmax - gfloor)
        else:
            # Falling arm; symmetric around peak.
            upper_edge = peak + (peak - thr)
            if z_norm >= upper_edge:
                immediate_gain = gfloor
            else:
                frac = (upper_edge - z_norm) / max(1e-6, upper_edge - peak)
                immediate_gain = gfloor + frac * (gmax - gfloor)

        # Post-event window contribution: residual decayed peak gain
        # acting as a floor above the immediate gain. This models the
        # tail where the hippocampus remains sensitised for ~30 min
        # after a threat event even as z_harm_a returns to baseline.
        window_tail = gfloor + window_decay * (gmax - gfloor)
        encoding_gain = max(immediate_gain, window_tail)

        if encoding_gain > gfloor + 1e-9:
            self._n_gain_elevations += 1

        self._last_encoding_gain = float(encoding_gain)

        # --- 2. Arousal tag (MECH-074b encoding-time write) ---

        # Arousal tag written at encoding time is a scalar per trace.
        # Simplest faithful mapping: normalised z_harm_a magnitude
        # above the threshold, scaled so that arousal_peak maps to
        # tag=1.0. Below threshold the tag is zero (no-op on retrieval
        # bias).
        if self.config.retrieval_tag_at_encoding and z_norm >= thr:
            denom = max(1e-6, peak - thr)
            raw = (z_norm - thr) / denom
            # Clamp to [0, 1] -- above-peak arousal still writes a
            # full-strength tag (the falling arm of the inverted-U
            # concerns ENCODING strength, not the semantic tag).
            arousal_tag = float(max(0.0, min(1.0, raw)))
        else:
            arousal_tag = 0.0
        self._last_arousal_tag = arousal_tag

        # --- 3. Retrieval bias (MECH-074b readout path) ---

        retrieval_bias: Optional[torch.Tensor] = None
        if arousal_tags_in_context is not None:
            # w_i = 1 + alpha * arousal_tag_i per trace.
            alpha = float(self.config.retrieval_bias_alpha)
            tags = arousal_tags_in_context.detach().to(torch.float32)
            bias = 1.0 + alpha * tags
            comp = float(self.config.retrieval_bias_compensation)
            if comp > 0.0:
                # Zero-sum compensation: untagged traces suppressed by
                # `comp`. Tagged traces (tag > 0) unaffected beyond
                # their own w_i.
                zero_mask = (tags <= 0.0).to(torch.float32)
                bias = bias - comp * zero_mask
            retrieval_bias = bias

        # --- 4. Remap signal (MECH-074d) ---

        remap_signal: Dict[int, float] = {}
        pe_magnitude = 0.0

        if z_harm_a_pred is not None:
            residual = (z_harm_a.detach().flatten() -
                        z_harm_a_pred.detach().flatten())
            pe_magnitude = float(torch.linalg.norm(residual).item())
            self._last_pe_magnitude = pe_magnitude

            # Running mean + variance (EMA) on harm-PE magnitude.
            alpha_pe = float(self.config.remap_pe_ema_alpha)
            delta = pe_magnitude - self._pe_mean
            self._pe_mean = self._pe_mean + alpha_pe * delta
            # EMA on squared deviation gives running variance.
            self._pe_var = (1.0 - alpha_pe) * self._pe_var + alpha_pe * (delta * delta)
            self._pe_std = float(max(1e-6, self._pe_var)) ** 0.5

            # PE threshold in units of running std-dev above the
            # running mean.
            sigma_thr = float(self.config.remap_pe_sigma_threshold)
            pe_z = (pe_magnitude - self._pe_mean) / max(1e-6, self._pe_std)

            # Gate 1: PE magnitude above threshold.
            pe_above = pe_z > sigma_thr

            # Gate 2: attribution candidates available (Moita 2004).
            attribution_ok = (
                candidate_code_contributions is not None
                and len(candidate_code_contributions) > 0
            )
            if self.config.remap_requires_attribution and not attribution_ok:
                pe_above = False

            if pe_above:
                # Select ~remap_code_fraction of the attribution
                # candidates, ordered by contribution magnitude
                # (highest first). Remap amplitude for each selected
                # code = 1.0 (binary per-code shape; population
                # gradedness comes from which codes are selected).
                if candidate_code_contributions is not None and len(candidate_code_contributions) > 0:
                    # Sort candidates by contribution, descending.
                    sorted_codes = sorted(
                        candidate_code_contributions.items(),
                        key=lambda kv: -abs(float(kv[1])),
                    )
                    n_total = len(sorted_codes)
                    n_select = max(1, int(round(float(self.config.remap_code_fraction) * n_total)))
                    for code_idx, _ in sorted_codes[:n_select]:
                        remap_signal[int(code_idx)] = 1.0
                    self._n_remap += 1
                elif not self.config.remap_requires_attribution:
                    # Broadcast-remap ablation path. Deliberately kept
                    # empty here -- caller must supply codes for
                    # remap to produce any output. This keeps the
                    # named-failure-signature behaviour (broadcast
                    # without codes = no-op) observable rather than
                    # silently writing nothing.
                    pass

        return BLAOutput(
            encoding_gain=float(encoding_gain),
            arousal_tag=float(arousal_tag),
            retrieval_bias=retrieval_bias,
            remap_signal=remap_signal,
            pe_magnitude=float(pe_magnitude),
            pe_baseline_std=float(self._pe_std),
            encoding_window_steps_remaining=int(steps_remaining),
        )

    # -- Read-only accessors --

    @property
    def encoding_gain(self) -> float:
        return float(self._last_encoding_gain)

    @property
    def arousal_tag(self) -> float:
        return float(self._last_arousal_tag)

    @property
    def pe_magnitude(self) -> float:
        return float(self._last_pe_magnitude)

    @property
    def pe_running_std(self) -> float:
        return float(self._pe_std)

    @property
    def diagnostics(self) -> Dict[str, int]:
        return {
            "n_ticks": int(self._n_ticks),
            "n_remap": int(self._n_remap),
            "n_gain_elevations": int(self._n_gain_elevations),
        }
