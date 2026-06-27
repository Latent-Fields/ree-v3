"""MECH-440: state_conditioned_self_annealing_noise_floor (NoisyNet analog).

Child claim of ARC-065 (behavioral_diversity_generation_pathway) and a
mechanistic refinement of MECH-313 (the state-INDEPENDENT post-softmax
temperature floor). MECH-313 lifts the softmax temperature *before* the commit;
V3-EXQ-687 found that floor NON-PROPAGATING -- the temperature was invisible to
the committed argmax (selected_action_entropy=0.0, the r1a_entropy_only_artefact
the degeneracy gate caught).

MECH-440 instead injects LEARNED PER-PARAMETER factorised-Gaussian WEIGHT NOISE
at the E3 selection head (NoisyNet; Fortunato et al. 2018), built as a
per-candidate additive bias from each candidate's first-action vector:

    bias[k] = ((sigma_w (x) eps_w) @ x_k) + (sigma_b (x) eps_b)        (mu == 0)

added INTO the modulatory accumulator BEFORE the within-eligible argmin (and
into the segregated-loop final), so it:

  1. PROPAGATES into the committed action -- it changes which candidate is the
     within-eligible argmin, not a discarded pre-commit temperature (the
     V3-EXQ-687 fix);
  2. is STATE-CONDITIONED -- the per-candidate action activation x_k modulates
     the noisy weight, so different states are perturbed differently with no
     hand-set per-state schedule;
  3. SELF-ANNEALS -- sigma is scaled by a LOCAL confidence EMA: it falls where
     the committed F-gap is consistently decisive (the policy is confident) and
     holds where near-ties recur (exploration pays), so the floor does not wash
     out committed-action diversity uniformly.

DESIGN CHOICE -- PURE NOISE INJECTOR (mu frozen at 0). NoisyNet's mu is the
ordinary (mean) policy weight; in REE the "mean" selection is already the F +
modulatory pipeline. Adding a second learned mean head would confound the
falsifier (is a committed-entropy lift the noise or a new learned mean?). So mu
is frozen at 0 and only sigma is the learned object -- the head adds ONLY
state-conditioned propagating noise. This isolates the MECH-440 falsifier.

FACTORISED GAUSSIAN (NoisyNet). One input-noise vector eps_in (size in_features)
and one output-noise vector eps_out (size out_features=1) per WAKING tick:
    f(x)   = sign(x) * sqrt(|x|)
    eps_w  = outer(f(eps_out), f(eps_in))        # [1, in_features]
    eps_b  = f(eps_out)                          # [1]
Resampled ONCE per select() call (shared across the K candidates, exactly as
NoisyNet shares noise across a batch), so per-candidate differentiation comes
from the distinct action inputs x_k and tick-to-tick rotation from resampling.

LOCAL, NOT GRADIENT (ARC-106 divergence #2). REE does not backprop through E3
selection (cf. the ARC-108 w_chan register_buffer / three-factor local rule).
So sigma here is a register_buffer annealed by a local confidence EMA, NOT a
gradient-trained nn.Parameter. The per-parameter sigma structure is preserved;
the annealing signal is shared and local. Disclosed in the grounding ledger.

ARC-106 divergence #1 (from the decision record): per-parameter sigma is one
description-level below biology's systems-level tonic/phasic LC-NE mode gate.

MECH-094. simulation_mode=True returns a zero perturbation and does NOT advance
the anneal EMA -- replay / DMN selection cannot inherit a waking exploration
floor that biologically belongs only to active behaviour (matches the MECH-313
noise_floor and SD-035 gated_policy simulation_mode pattern).

NO-OP DEFAULT. use_noisy_selection_head=False -> the head is never instantiated
(bit-identical). use_noisy_selection_head=True with sigma_init=0.0 -> every
sigma is 0 -> the output is exactly 0 -> bit-identical (the sd stub's
"sigma_init=0 = bit-identical" contract).

Biology (lit-pull before registration). LC-NE tonic exploration is itself
state/uncertainty-conditioned and annealed by controllability: Aston-Jones &
Cohen 2005 adaptive-gain theory; Tervo et al. 2014 (causal LC->ACC gating of
stochastic choice under uncertainty, suppressed when a model-based strategy
wins). MECH-313's state-independent framing is biologically under-specified; the
NoisyNet refinement is the biologically-correct shape.

See docs/architecture/state_conditioned_exploration_noise_floor.md (#mech-440)
and REE_assembly/docs/claims/claims.yaml (MECH-440).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class NoisySelectionHeadConfig:
    """MECH-440 NoisyNet selection-head configuration.

    Attributes:
        action_dim : width of the per-candidate first-action input vector.
        sigma_init : initial value for every per-parameter sigma. 0.0 (default)
            => zero output => bit-identical even when the master switch is on.
        weight : overall scale on the per-candidate noisy bias before it enters
            the modulatory accumulator.
        anneal : when True, sigma is scaled by the local confidence EMA below.
            False freezes sigma at sigma_init (the no-anneal control arm).
        anneal_floor : lower bound on the anneal scale (sigma never decays below
            anneal_floor * sigma_init).
        anneal_ema_alpha : EMA rate on the per-tick confidence signal
            (1 - gap_norm).
    """

    action_dim: int = 4
    sigma_init: float = 0.0
    weight: float = 1.0
    anneal: bool = True
    anneal_floor: float = 0.1
    anneal_ema_alpha: float = 0.01


class NoisySelectionHead(nn.Module):
    """MECH-440 NoisyNet propagating selection-head weight-noise injector.

    Pure noise injector: mu frozen at 0, only per-parameter sigma is learned
    (locally annealed). Produces a per-candidate additive bias from candidate
    first-action vectors that PROPAGATES into the committed within-eligible
    argmin.

    Diagnostics tracked per call (read into manifests via get_state()):
        _last_bias_range            : float (cross-candidate range of the bias)
        _last_sigma_scale           : float (current anneal scale on sigma)
        _last_n_simulation_skips    : int
        _n_waking_calls             : int
    """

    def __init__(self, config: "NoisySelectionHeadConfig | None" = None) -> None:
        super().__init__()
        self.config = config if config is not None else NoisySelectionHeadConfig()
        in_features = int(self.config.action_dim)
        if in_features < 1:
            raise ValueError(
                f"action_dim must be >= 1 for the noisy selection head; got {in_features}."
            )
        if self.config.sigma_init < 0.0:
            raise ValueError(
                f"sigma_init must be >= 0 (noise scale); got {self.config.sigma_init}."
            )
        out_features = 1
        # mu FROZEN at 0 (pure noise injector). register_buffer, never an
        # nn.Parameter -- no autograd surface (REE does not backprop selection).
        self.register_buffer("mu_w", torch.zeros(out_features, in_features))
        self.register_buffer("mu_b", torch.zeros(out_features))
        # Per-parameter sigma (the LEARNED object), locally annealed. Init flat
        # at sigma_init -> sigma_init=0 gives exactly-zero output (bit-identical).
        self.register_buffer(
            "sigma_w", torch.full((out_features, in_features), float(self.config.sigma_init))
        )
        self.register_buffer(
            "sigma_b", torch.full((out_features,), float(self.config.sigma_init))
        )
        # Local confidence EMA of (1 - gap_norm): high near-ties, low decisive.
        # Init 1.0 so a freshly-on head starts at full (un-annealed) noise.
        self._confidence_ema: float = 1.0
        self._last_bias_range: float = 0.0
        self._last_sigma_scale: float = 1.0
        self._last_n_simulation_skips: int = 0
        self._n_waking_calls: int = 0

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """NoisyNet factorised-noise transform f(x) = sign(x) * sqrt(|x|)."""
        return x.sign() * x.abs().sqrt()

    def _sigma_scale(self) -> float:
        """Current local-anneal scale on sigma in [anneal_floor, 1.0]."""
        if not self.config.anneal:
            return 1.0
        floor = float(self.config.anneal_floor)
        return floor + (1.0 - floor) * float(self._confidence_ema)

    def observe_gap(self, gap_norm: float, simulation_mode: bool = False) -> None:
        """Advance the local confidence EMA from the per-tick normalized F-gap.

        gap_norm in [0, 1] (0 = near-tie/uncertain, 1 = decisive/confident).
        The EMA tracks (1 - gap_norm): it stays high where near-ties recur
        (keep noise) and falls where the gap is consistently decisive (anneal
        sigma down). MECH-094: no update on a simulation tick.
        """
        if simulation_mode:
            return
        # Defensive: never let a non-finite gap (an untrained / unstable tick can
        # produce a nan raw_score_range upstream) poison the EMA -- a nan here would
        # propagate to every subsequent bias. Skip the update and self-heal the EMA.
        if not math.isfinite(float(gap_norm)):
            if not math.isfinite(self._confidence_ema):
                self._confidence_ema = 1.0
            return
        if not math.isfinite(self._confidence_ema):
            self._confidence_ema = 1.0
        g = max(0.0, min(1.0, float(gap_norm)))
        a = float(self.config.anneal_ema_alpha)
        self._confidence_ema = (1.0 - a) * self._confidence_ema + a * (1.0 - g)

    def forward(
        self,
        action_features: torch.Tensor,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """Per-candidate noisy bias [K] from candidate first-action vectors [K, action_dim].

        Resamples factorised Gaussian noise once per call (shared across the K
        candidates). Returns weighted, anneal-scaled per-candidate bias. Returns
        a zero vector under simulation_mode (MECH-094) and when sigma is all-zero
        (the sigma_init=0 bit-identical contract).
        """
        if action_features.dim() == 1:
            action_features = action_features.unsqueeze(0)
        k = int(action_features.shape[0])
        device = action_features.device
        dtype = self.mu_w.dtype
        x = action_features.to(device=self.mu_w.device, dtype=dtype)  # [K, in]

        if simulation_mode:
            self._last_n_simulation_skips += 1
            return torch.zeros(k, device=device, dtype=action_features.dtype)

        self._n_waking_calls += 1
        scale = self._sigma_scale()
        self._last_sigma_scale = scale

        # Fast bit-identical exit: no noise scale at all.
        if float(self.sigma_w.abs().max().item()) == 0.0 and float(self.sigma_b.abs().max().item()) == 0.0:
            self._last_bias_range = 0.0
            return torch.zeros(k, device=device, dtype=action_features.dtype)

        in_features = x.shape[1]
        eps_in = self._f(torch.randn(in_features, device=self.mu_w.device, dtype=dtype))
        eps_out = self._f(torch.randn(1, device=self.mu_w.device, dtype=dtype))
        eps_w = torch.outer(eps_out, eps_in)          # [1, in]
        eps_b = eps_out                               # [1]

        # mu == 0 -> pure noise. w_eff [1, in], b_eff [1].
        w_eff = (self.sigma_w * eps_w) * scale
        b_eff = (self.sigma_b * eps_b) * scale
        y = (x @ w_eff.t()).squeeze(-1) + b_eff       # [K]
        bias = float(self.config.weight) * y
        if k >= 2:
            self._last_bias_range = float((bias.max() - bias.min()).item())
        else:
            self._last_bias_range = 0.0
        return bias.to(dtype=action_features.dtype)

    def reset(self) -> None:
        """Reset per-episode diagnostic counters (sigma + EMA persist across episodes)."""
        self._last_bias_range = 0.0
        self._last_n_simulation_skips = 0
        self._n_waking_calls = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "noisy_selection_last_bias_range": self._last_bias_range,
            "noisy_selection_sigma_scale": self._last_sigma_scale,
            "noisy_selection_confidence_ema": self._confidence_ema,
            "noisy_selection_sigma_w_mean": float(self.sigma_w.abs().mean().item()),
            "noisy_selection_n_simulation_skips": self._last_n_simulation_skips,
            "noisy_selection_n_waking_calls": self._n_waking_calls,
        }
