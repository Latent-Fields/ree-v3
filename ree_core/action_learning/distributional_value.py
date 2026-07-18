"""MECH-457 H-retention-critic: a distributional (two-hot / HL-Gauss) value target.

An alternative VALUE ESTIMATOR for the actor-critic critic head. The scalar critic
regresses V(z) toward the GAE return under MSE; the distributional critic predicts a
CATEGORICAL distribution over a fixed bin support and is trained by cross-entropy
against a projection of the same return. The scalar value consumed downstream is the
expectation of that distribution, so GAE, the bootstrap value, the credit-replay TD
priority and every eval path keep consuming a scalar and are untouched.

WHY (measured, not speculative): V3-EXQ-782 R-(b) found the shared CTRL critic flat and
uninformed -- std(V)/std(G) = 0.041 against a 0.25 collapse threshold, and a
pre-reward-vs-far separation ratio of 0.016 against a 0.25 floor. An uninformed baseline
yields an unbaselined, high-variance advantage, which is a candidate mechanism for the
observed erosion of an installed behavioral prior (V3-EXQ-780: raw_view 20.933 -> 11.667
under RL refinement).

ANTI-ALIAS (load-bearing): this module changes what the baseline KNOWS. It does not
constrain how far the policy may MOVE. Nothing here touches the policy head, the
log-prob, the entropy term or the advantage weighting -- that locus belongs to
mech457_policy_kl_anchor / H-retention-consolidation, and a leg changing both would make
neither readable.

Biological grounding: a ventral-striatal value critic taught by a dopaminergic
reward-prediction error is not committed to a point estimate. Distributional coding of
value is the better-supported reading of dopaminergic populations (Dabney et al. 2020,
"A distributional code for value in dopamine-based reinforcement learning") -- individual
dopamine neurons carry heterogeneous reversal points that jointly encode a distribution
over reward rather than a single mean. The scalar head was the simplification; this is
the less simplified form, not an ML import.

ML statement (engineering counsel only): symlog two-hot bin encoding follows Hafner et al.
2023 (DreamerV3); the HL-Gauss target smoothing follows Farebrother et al. 2024
("Stop Regressing"), which finds a discretised-Gaussian target outperforms two-hot as a
value-regression objective. The engineering problem both solve is the same one 782 R-(b)
measured: scalar regression onto a heavy-tailed, wide-dynamic-range return collapses to
its conditional mean and stops discriminating states. Neither dictates the architecture --
the head width, the trunk and the reward channel are fixed by the SD.

MECH-094: not applicable -- this module performs no memory writes; it is a loss/decode
transform over the critic head's output.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|) -- compresses a wide-dynamic-range return onto a bounded
    support without discarding sign or small-magnitude structure."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


class ValueBins(nn.Module):
    """Fixed bin support in symlog space, plus target projection and expectation decode.

    Args:
        n_bins: number of categorical bins (the critic head's output width).
        limit: support half-width in SYMLOG space. The support therefore covers returns
            up to symexp(limit) in magnitude; limit=10 covers ~2.2e4, far beyond the
            reward-std-scaled GAE returns the MECH-457 loop produces.
        sigma_ratio: HL-Gauss target smoothing, in units of bin width. 0.0 -> pure
            two-hot (DreamerV3). 0.75 is the Farebrother et al. 2024 recommendation and
            the MECH-457 ON default.
    """

    def __init__(self, n_bins: int = 41, limit: float = 10.0, sigma_ratio: float = 0.75) -> None:
        super().__init__()
        if int(n_bins) < 2:
            raise ValueError("ValueBins requires n_bins >= 2.")
        if float(limit) <= 0.0:
            raise ValueError("ValueBins requires limit > 0.")
        if float(sigma_ratio) < 0.0:
            raise ValueError("ValueBins requires sigma_ratio >= 0.")
        self.n_bins = int(n_bins)
        self.limit = float(limit)
        self.sigma_ratio = float(sigma_ratio)
        self.bin_width = 2.0 * self.limit / float(self.n_bins)

        edges = torch.linspace(-self.limit, self.limit, self.n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("edges", edges)
        self.register_buffer("support", centers)

    # -- target projection ---------------------------------------------------
    def project(self, targets: torch.Tensor) -> torch.Tensor:
        """[B] scalar returns -> [B, n_bins] target probability mass.

        Two-hot when sigma_ratio == 0; otherwise the HL-Gauss discretised Gaussian.
        """
        t = symlog(targets.reshape(-1).to(self.support.dtype))
        if self.sigma_ratio <= 0.0:
            return self._two_hot(t)
        return self._hl_gauss(t)

    def _two_hot(self, t: torch.Tensor) -> torch.Tensor:
        lo_c, hi_c = float(self.support[0].item()), float(self.support[-1].item())
        t = torch.clamp(t, lo_c, hi_c)
        # Position in units of bin index along the CENTERS grid.
        pos = (t - lo_c) / self.bin_width
        lower = torch.clamp(torch.floor(pos).long(), 0, self.n_bins - 1)
        upper = torch.clamp(lower + 1, 0, self.n_bins - 1)
        w_upper = (pos - lower.to(pos.dtype)).clamp(0.0, 1.0)
        out = torch.zeros(t.shape[0], self.n_bins, dtype=t.dtype, device=t.device)
        out.scatter_add_(1, lower.unsqueeze(1), (1.0 - w_upper).unsqueeze(1))
        out.scatter_add_(1, upper.unsqueeze(1), w_upper.unsqueeze(1))
        return out

    def _hl_gauss(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma_ratio * self.bin_width
        # Mass in bin i = Phi((edge_{i+1} - t)/sigma) - Phi((edge_i - t)/sigma), renormalised
        # over the finite support so a target outside the support still yields a valid
        # distribution concentrated at the nearest end.
        z = (self.edges.unsqueeze(0) - t.unsqueeze(1)) / (sigma * math.sqrt(2.0))
        cdf = 0.5 * (1.0 + torch.erf(z))
        mass = cdf[:, 1:] - cdf[:, :-1]
        total = mass.sum(dim=1, keepdim=True)
        # Degenerate only if the target is astronomically outside the support; fall back
        # to a one-hot at the nearest end rather than dividing by ~0.
        safe = total.squeeze(1) > 1e-12
        out = torch.zeros_like(mass)
        if bool(safe.any()):
            out[safe] = mass[safe] / total[safe]
        if bool((~safe).any()):
            far = (~safe).nonzero(as_tuple=True)[0]
            end = torch.where(t[far] > 0, self.n_bins - 1, 0)
            out[far, end] = 1.0
        return out

    # -- decode / loss -------------------------------------------------------
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """[B, n_bins] logits -> [B] scalar value = symexp(E_p[support]).

        This is what keeps the distributional critic drop-in: every downstream consumer
        (GAE, bootstrap, credit-replay TD priority, eval) still receives a scalar.
        """
        probs = torch.softmax(logits, dim=-1)
        t_hat = (probs * self.support.unsqueeze(0)).sum(dim=-1)
        return symexp(t_hat)

    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy of the predicted categorical against the projected return.
        `targets` are raw scalar returns; the projection is done here and detached (it is
        a target, not a differentiable path)."""
        with torch.no_grad():
            target_p = self.project(targets)
        log_p = torch.log_softmax(logits, dim=-1)
        return -(target_p * log_p).sum(dim=-1).mean()
