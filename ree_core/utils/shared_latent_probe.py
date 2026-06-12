"""
shared_latent_gradient_probe: per-module gradient coupling readout on a shared
latent (MECH-423 R1 readiness instrumentation; ARC-004 shared L-space latent).

R1 (the COUPLING readiness check of the EXP-0380 / MECH-423 super-additivity
ablation) asks whether a shared latent fed jointly to several modules is doing
genuine, non-destructive cross-module work:

  * NON-ZERO shared-latent gradient norm in EACH module (||grad_z|| > 0; if ~0 the
    latent is uncoupled and the integrated arm equals the isolated arm by
    construction); AND
  * the mean pairwise cosine similarity between per-module gradients on the shared
    latent is NOT NET-NEGATIVE (mean inter-module gradient cosine >= 0; a
    net-negative cosine is the negative-transfer regime where sub-additivity is
    the EXPECTED consequence of gradient conflict -- re-scope / route
    substrate_not_ready, never FAIL).

R1 is constructible inside an experiment arm, so it is the least urgent of the
three readouts and is provided here as a small, reusable, pure function (no
substrate state, no hot-path touch) so the EXP-0380 integrated arm does not have
to reinvent the retain_grad + per-module grad-norm + cosine boilerplate.

Grounds Yu et al. (PCGrad, NeurIPS 2020) -- conflicting gradients (negative
cosine) cause negative transfer -- and Caruana 1997 -- shared-representation
multi-task learning beats isolated learning only when the tasks are related.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch


def shared_latent_gradient_probe(
    z_shared: torch.Tensor,
    module_losses: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Measure per-module gradient coupling on a shared latent.

    Args:
        z_shared: the shared L-space latent fed jointly to the modules. Must be a
            float tensor that participates in each module's loss graph (the caller
            constructs each loss FROM z_shared). It does NOT need requires_grad set
            -- this probe uses torch.autograd.grad(..., retain_graph=True) so the
            caller's own training backward is unaffected.
        module_losses: name -> callable(z_shared) -> scalar loss tensor. Each
            callable must build its loss so that it is differentiable w.r.t.
            z_shared (i.e. the module consumes z_shared, not a detached copy).
        eps: numerical floor for the cosine denominators.

    Returns:
        Flat dict of pure-arithmetic readouts:
            per_module_grad_norm  -- {name: float} L2 norm of d(loss_name)/d(z_shared)
            min_grad_norm         -- the smallest per-module grad norm (R1 "in EACH
                                     module" reduces to min > 0)
            mean_pairwise_cosine  -- mean cosine over all distinct module pairs
                                     (1.0 when fewer than 2 modules contribute a
                                     non-degenerate gradient -- no conflict possible)
            n_modules             -- number of modules with a non-degenerate gradient
            coupled               -- bool: min_grad_norm > 0 AND
                                     mean_pairwise_cosine >= 0 (the R1 readiness
                                     verdict; non-negative cosine, not net-negative)
    """
    if not z_shared.requires_grad:
        # Allow callers to pass a non-leaf z_shared; the grad still flows through
        # the loss graph. If it is a detached leaf, autograd.grad will raise, which
        # is the correct loud failure (the caller fed a detached latent).
        z_shared = z_shared

    grads: Dict[str, torch.Tensor] = {}
    grad_norms: Dict[str, float] = {}
    for name, loss_fn in module_losses.items():
        loss = loss_fn(z_shared)
        (g,) = torch.autograd.grad(
            loss, z_shared, retain_graph=True, create_graph=False, allow_unused=True
        )
        if g is None:
            g = torch.zeros_like(z_shared)
        g_flat = g.detach().reshape(-1)
        norm = float(g_flat.norm().item())
        grad_norms[name] = norm
        grads[name] = g_flat

    # Mean pairwise cosine over modules with a non-degenerate gradient.
    contributing = [n for n, v in grad_norms.items() if v > eps]
    if len(contributing) < 2:
        mean_cos = 1.0  # no pair -> no conflict possible
    else:
        cos_vals = []
        for i in range(len(contributing)):
            for j in range(i + 1, len(contributing)):
                gi = grads[contributing[i]]
                gj = grads[contributing[j]]
                denom = gi.norm() * gj.norm() + eps
                cos_vals.append(float((torch.dot(gi, gj) / denom).item()))
        mean_cos = sum(cos_vals) / len(cos_vals)

    min_norm = min(grad_norms.values()) if grad_norms else 0.0
    return {
        "per_module_grad_norm": grad_norms,
        "min_grad_norm": float(min_norm),
        "mean_pairwise_cosine": float(mean_cos),
        "n_modules": int(len(contributing)),
        "coupled": bool(min_norm > eps and mean_cos >= 0.0),
    }
