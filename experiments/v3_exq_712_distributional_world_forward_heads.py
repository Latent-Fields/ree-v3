#!/opt/local/bin/python3
"""
V3-EXQ-712 -- DISTRIBUTIONAL E2 WORLD-FORWARD HEADS (T0-alpha-inspired probe).

EXPERIMENT_PURPOSE = diagnostic. PROMOTES NOTHING (claim_ids = []). This is an
exploratory head-formulation probe, NOT a governance falsifier: it discriminates
WHICH of several probabilistic forward-model head formulations is worth promoting
into the E2 substrate, and whether ANY of them beats the current point predictor.

THE QUESTION
------------
REE-v3's world/forward models are pure POINT predictors trained with MSE
(E2FastPredictor.world_forward -> single [batch, world_dim] vector, loss
F.mse_loss; e2_fast.py / e2_world.py). Predictive uncertainty is only ever
POST-HOC: E3.current_precision = 1/(running_variance+eps) is a RETROSPECTIVE EMA
of past point-prediction MSE (e3_selector.py:512); SplitEncoder.precision_logit
is a STATIC per-channel encoder gate (stack.py:913). Neither is a forward-looking,
state-conditional predictive DISTRIBUTION emitted by the world model.

The T0-alpha time-series foundation model (The Forecasting Company, 2026) makes
the opposite commitment: the forward model emits a predictive DISTRIBUTION (nine
quantile heads scored by CRPS) rather than a point. The transferable idea (the
rest of T0 -- patching, cross-series group attention, pretraining -- does NOT map
onto REE's single compact-latent stream) is: give the forward head a predictive
distribution and score it with a proper scoring rule.

This probe holds the encoder + transition set FIXED and swaps only the forward
HEAD + its loss, comparing four formulations on the SAME held-out (z_world_t, a_t)
-> z_world_{t+1} transitions:

THE 4 ARMS (same input [z_world_t, a_t]; differ only in output head + loss)
--------------------------------------------------------------------------
  mse_point       : point MLP -> mean; MSE loss. For DISTRIBUTIONAL scoring it is
                    given a global homoscedastic sigma fitted on the train
                    residuals -- the honest "constant-variance Gaussian" null, and
                    the direct analog of E3's state-BLIND running-variance EMA.
  hetero_gaussian : MLP -> mean + logvar (per dim); Gaussian NLL loss.
  quantile_pinball: MLP -> 9 quantiles/dim (0.1..0.9); pinball loss. Direct
                    T0-alpha analog; predictive spread = interquantile range.
  mixture_gaussian: per-dim mixture density network, K=3; mixture NLL loss.
                    Targets multimodal / branching futures.

METRICS (per arm x seed, on held-out transitions)
-------------------------------------------------
  crps_mean            : sample-based CRPS (E|X-y| - 0.5 E|X-X'|), the common
                         currency all four heads can produce (mixture/quantile via
                         sampling; Gaussians via sampling too, for a uniform
                         estimator). LOWER is better. This is the ranking metric.
  nll_mean             : held-out Gaussian/mixture NLL where a density exists
                         (point/hetero/mixture; NaN for quantile). LOWER better.
  coverage_err_80      : |empirical central-80%-interval coverage - 0.80|. Lower
                         = better-calibrated predictive intervals.
  precision_error_corr : Pearson corr between the head's per-point predictive
                         VARIANCE (mean over dims) and the realized squared error.
                         A well-calibrated FORWARD uncertainty is HIGH exactly when
                         it is about to be wrong -> POSITIVE corr. The point head's
                         variance is constant (corr := 0.0) -- this IS the E3
                         running-variance-EMA null (state-blind), so C3 measures
                         whether a distributional head carries commitment-relevant
                         precision the EMA structurally cannot.

PRE-REGISTERED READINESS + OUTCOME MAP (diagnostic self-route -- falsifiable)
----------------------------------------------------------------------------
  LOAD-BEARING READINESS (the vacuity guard): the held-out transitions must carry
  conditional aleatoric uncertainty a point predictor cannot remove -- else all
  heads collapse to the mean and the comparison is vacuous ("distributional heads
  add nothing" would be UNMEASURABLE, not a finding). Operationalised as the
  point (MSE) arm's NORMALISED held-out residual (point_rmse / target_std) >= a
  floor, measured on the point arm as the positive control (SAME residual-spread
  statistic the CRPS/NLL criteria consume). If UNMET (dynamics effectively
  deterministic in z_world), self-route to substrate_not_ready_requeue
  (non_contributory, non_degenerate=False) -- NEVER a "distributional heads don't
  help" verdict.

  If readiness MET, the load-bearing criterion:
  C1 (PASS): the best distributional head's held-out CRPS is STRICT-BELOW the
     mse_point CRPS by >= CRPS_IMPROVE_FRAC on a strict-majority of seeds -- i.e.
     modeling the predictive distribution demonstrably helps.
     -> outcome PASS, label distributional_head_ranking (winner = lowest mean CRPS).
  If readiness MET but C1 fails (no distributional head beats the point baseline):
     -> outcome FAIL, label distributional_heads_no_benefit_over_point. This is a
     REAL informative negative (latent-space dynamics do not need distributional
     modeling), consistent with the caveat that latent-space MSE is less
     pathological than observation-space MSE.

  Secondary (reported, not gating): C2 calibration (best coverage_err_80 vs point),
  C3 downstream commitment (best precision_error_corr vs the point/EMA null ~0).

Diagnostic; PROMOTES NOTHING. A PASS here would motivate an /implement-substrate
of the winning head into E2 (with the SD-031 phased stop-gradient caveat: a
variance/logvar head must be trained so it cannot "explain away" the E2WorldForward
residual that carries the agency signal). Connects to: imagination-learning gap,
conversion-ceiling / committed-action-diversity, escape-forward, ARC-016 / Q-007
(LC-NE) precision story.

See towardsdatascience.com/time-series-llms-explained-with-t0-alpha (source),
    ree_core/predictors/e2_fast.py + e2_world.py (the point forward models),
    ree_core/predictors/e3_selector.py (current_precision EMA -- the C3 null).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_712_distributional_world_forward_heads"
QUEUE_ID = "V3-EXQ-712"
SUPERSEDES = None
BACKLOG_ID = None
CLAIM_IDS: List[str] = []                 # diagnostic: promotes nothing, tests no registered claim
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ----- Latent / model dims -----
WORLD_DIM = 32
SELF_DIM = 32
ALPHA_WORLD = 0.9        # high: z_world must faithfully encode world obs (SD-008 default 0.3 too low)
ALPHA_SELF = 0.3
HEAD_HIDDEN = 128
MIXTURE_K = 3
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NOMINAL_COVERAGE = 0.80  # central interval [q0.1, q0.9]
CRPS_SAMPLES = 100       # sample-based CRPS estimator draws per test point

# ----- Schedule -----
SEEDS = [42, 43, 44, 45, 46]
COLLECT_EPISODES = 40            # P0: transition-collection episodes per seed (runner denominator M)
STEPS_PER_EPISODE = 120
HEAD_TRAIN_EPOCHS = 40
HEAD_BATCH = 256
HEAD_LR = 1e-3
TRAIN_FRAC = 0.8

DRY_RUN_SEEDS = [42]
DRY_RUN_COLLECT = 3
DRY_RUN_STEPS = 20
DRY_RUN_EPOCHS = 3

# ----- Pre-registered thresholds -----
# Readiness: point arm's normalised held-out residual must clear this floor
# (there is genuine conditional spread to model). Relative to target std, so
# robust to z_world scaling.
READINESS_NORM_RESIDUAL_FLOOR = 0.05
READINESS_MIN_TEST_TRANSITIONS = 200
# C1: best distributional CRPS must beat point CRPS by this fraction, per seed.
CRPS_IMPROVE_FRAC = 0.02
# Strict-majority of seeds gate.
def _majority(n: int) -> int:
    return n // 2 + 1


ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
)


ARMS: List[Dict[str, Any]] = [
    {"arm_id": "mse_point", "head": "point"},
    {"arm_id": "hetero_gaussian", "head": "hetero"},
    {"arm_id": "quantile_pinball", "head": "quantile"},
    {"arm_id": "mixture_gaussian", "head": "mixture"},
]


# ---------------------------------------------------------------------------
# Forward-head modules. All take x = concat([z_world_t, a_onehot]) -> predict
# the distribution of z_world_{t+1} (WORLD_DIM independent marginals).
# ---------------------------------------------------------------------------


def _trunk(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, HEAD_HIDDEN),
        nn.ReLU(),
        nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN),
        nn.ReLU(),
    )


class PointHead(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.trunk = _trunk(in_dim)
        self.mean = nn.Linear(HEAD_HIDDEN, d)

    def forward(self, x):
        return self.mean(self.trunk(x))  # [B, D]


class HeteroHead(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.trunk = _trunk(in_dim)
        self.mean = nn.Linear(HEAD_HIDDEN, d)
        self.logvar = nn.Linear(HEAD_HIDDEN, d)

    def forward(self, x):
        h = self.trunk(x)
        # clamp logvar for numerical stability (predictive std in ~[6e-3, 22])
        return self.mean(h), torch.clamp(self.logvar(h), min=-10.0, max=6.0)


class QuantileHead(nn.Module):
    def __init__(self, in_dim: int, d: int, q: int):
        super().__init__()
        self.d = d
        self.q = q
        self.trunk = _trunk(in_dim)
        self.out = nn.Linear(HEAD_HIDDEN, d * q)

    def forward(self, x):
        return self.out(self.trunk(x)).view(-1, self.d, self.q)  # [B, D, Q]


class MixtureHead(nn.Module):
    def __init__(self, in_dim: int, d: int, k: int):
        super().__init__()
        self.d = d
        self.k = k
        self.trunk = _trunk(in_dim)
        self.mean = nn.Linear(HEAD_HIDDEN, d * k)
        self.logvar = nn.Linear(HEAD_HIDDEN, d * k)
        self.logit = nn.Linear(HEAD_HIDDEN, d * k)

    def forward(self, x):
        h = self.trunk(x)
        b = x.shape[0]
        mean = self.mean(h).view(b, self.d, self.k)
        logvar = torch.clamp(self.logvar(h).view(b, self.d, self.k), min=-10.0, max=6.0)
        logit = self.logit(h).view(b, self.d, self.k)
        return mean, logvar, logit


def _build_head(head_type: str, in_dim: int) -> nn.Module:
    if head_type == "point":
        return PointHead(in_dim, WORLD_DIM)
    if head_type == "hetero":
        return HeteroHead(in_dim, WORLD_DIM)
    if head_type == "quantile":
        return QuantileHead(in_dim, WORLD_DIM, len(QUANTILE_LEVELS))
    if head_type == "mixture":
        return MixtureHead(in_dim, WORLD_DIM, MIXTURE_K)
    raise ValueError(f"unknown head_type {head_type!r}")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _gaussian_nll(mean, logvar, target):
    # 0.5 * (logvar + (y-mu)^2 / var) + const; drop const (log 2pi) -- comparison-invariant.
    inv_var = torch.exp(-logvar)
    return (0.5 * (logvar + (target - mean) ** 2 * inv_var)).mean()


def _pinball_loss(quantiles, target, levels):
    # quantiles [B, D, Q]; target [B, D]; levels [Q]
    err = target.unsqueeze(-1) - quantiles  # [B, D, Q]
    lv = levels.view(1, 1, -1)
    return torch.maximum(lv * err, (lv - 1.0) * err).mean()


def _mixture_nll(mean, logvar, logit, target):
    # per-dim independent mixture. mean/logvar/logit [B, D, K]; target [B, D]
    log_w = F.log_softmax(logit, dim=-1)                     # [B, D, K]
    var = torch.exp(logvar)
    comp_logp = -0.5 * (logvar + math.log(2 * math.pi)
                        + (target.unsqueeze(-1) - mean) ** 2 / var)  # [B, D, K]
    logp = torch.logsumexp(log_w + comp_logp, dim=-1)        # [B, D]
    return (-logp).mean()


# ---------------------------------------------------------------------------
# Predictive sampling (uniform CRPS / coverage estimator across heads)
# ---------------------------------------------------------------------------


def _samples_point(mean, sigma, n):
    # mean [B, D]; sigma scalar tensor [D] (global homoscedastic). -> [B, D, n]
    b, d = mean.shape
    eps = torch.randn(b, d, n)
    return mean.unsqueeze(-1) + sigma.view(1, d, 1) * eps


def _samples_hetero(mean, logvar, n):
    b, d = mean.shape
    std = torch.exp(0.5 * logvar).unsqueeze(-1)  # [B, D, 1]
    return mean.unsqueeze(-1) + std * torch.randn(b, d, n)


def _samples_mixture(mean, logvar, logit, n):
    # per-dim mixture sampling. [B, D, K] -> [B, D, n]
    b, d, k = mean.shape
    w = F.softmax(logit, dim=-1)                              # [B, D, K]
    comp = torch.multinomial(w.reshape(b * d, k), n, replacement=True).view(b, d, n)
    std = torch.exp(0.5 * logvar)
    m_sel = torch.gather(mean, 2, comp)                       # [B, D, n]
    s_sel = torch.gather(std, 2, comp)
    return m_sel + s_sel * torch.randn(b, d, n)


def _samples_quantile(quantiles, levels, n):
    # inverse-CDF sampling from monotone-rearranged quantiles. quantiles [B,D,Q].
    b, d, q = quantiles.shape
    qs, _ = torch.sort(quantiles, dim=-1)                     # rearrangement fixes crossing
    lv = levels.to(qs.device)
    u = torch.rand(b, d, n)
    # piecewise-linear interpolation of u over (levels -> qs); flat-clamp outside.
    out = torch.zeros(b, d, n)
    for qi in range(q - 1):
        lo, hi = lv[qi].item(), lv[qi + 1].item()
        mask = (u >= lo) & (u < hi)
        if mask.any():
            frac = (u - lo) / max(hi - lo, 1e-9)
            interp = qs[..., qi:qi + 1] + frac * (qs[..., qi + 1:qi + 2] - qs[..., qi:qi + 1])
            out = torch.where(mask, interp, out)
    out = torch.where(u < lv[0].item(), qs[..., 0:1].expand_as(out), out)
    out = torch.where(u >= lv[-1].item(), qs[..., -1:].expand_as(out), out)
    return out


def _crps_from_samples(samples, target):
    # samples [B, D, n]; target [B, D]. CRPS = E|X-y| - 0.5 E|X-X'|, per (B,D), meaned.
    y = target.unsqueeze(-1)
    term1 = (samples - y).abs().mean(dim=-1)                  # [B, D]
    n = samples.shape[-1]
    # E|X-X'| via a single random permutation (unbiased-ish, cheap)
    perm = samples[..., torch.randperm(n)]
    term2 = 0.5 * (samples - perm).abs().mean(dim=-1)         # [B, D]
    return float((term1 - term2).mean().item())


def _coverage_from_samples(samples, target, nominal):
    # central-interval empirical coverage. samples [B,D,n]; target [B,D].
    lo_q = (1.0 - nominal) / 2.0
    hi_q = 1.0 - lo_q
    lo = torch.quantile(samples, lo_q, dim=-1)                # [B, D]
    hi = torch.quantile(samples, hi_q, dim=-1)
    inside = ((target >= lo) & (target <= hi)).float().mean()
    return float(inside.item())


def _predictive_variance(head_type, out, sigma_global) -> torch.Tensor:
    # per-point predictive variance averaged over dims -> [B]
    if head_type == "point":
        # constant: the state-blind EMA null. Return a constant vector.
        return (sigma_global ** 2).mean().expand(out.shape[0]).clone()
    if head_type == "hetero":
        mean, logvar = out
        return torch.exp(logvar).mean(dim=-1)                 # [B]
    if head_type == "quantile":
        quantiles = out
        qs, _ = torch.sort(quantiles, dim=-1)
        # spread proxy: variance implied by the interquantile range (IQR/1.349)^2
        iqr = qs[..., -1] - qs[..., 0]                        # [B, D] (q0.9 - q0.1)
        std = iqr / 2.563                                     # z0.9 - z0.1 = 2.563
        return (std ** 2).mean(dim=-1)
    if head_type == "mixture":
        mean, logvar, logit = out
        w = F.softmax(logit, dim=-1)
        var = torch.exp(logvar)
        mix_mean = (w * mean).sum(dim=-1, keepdim=True)
        # law of total variance per dim
        total = (w * (var + (mean - mix_mean) ** 2)).sum(dim=-1)  # [B, D]
        return total.mean(dim=-1)                             # [B]
    raise ValueError(head_type)


def _mean_prediction(head_type, out, sigma_global) -> torch.Tensor:
    if head_type == "point":
        return out
    if head_type == "hetero":
        return out[0]
    if head_type == "quantile":
        qs, _ = torch.sort(out, dim=-1)
        mid = len(QUANTILE_LEVELS) // 2
        return qs[..., mid]                                   # median (q0.5)
    if head_type == "mixture":
        mean, logvar, logit = out
        w = F.softmax(logit, dim=-1)
        return (w * mean).sum(dim=-1)
    raise ValueError(head_type)


# ---------------------------------------------------------------------------
# P0: collect (z_world_t, a_t, z_world_{t+1}) transitions from a frozen encoder
# ---------------------------------------------------------------------------


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
    )
    return REEAgent(cfg)


def _collect_transitions(
    seed: int, collect_episodes: int, steps_per_episode: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Roll out random-action episodes; return (Z0 [N,WORLD_DIM], A [N,action_dim],
    Z1 [N,WORLD_DIM], action_dim). The encoder is a FIXED feature map (never
    trained here) -- all arms see the SAME transitions, so the comparison isolates
    the forward-head formulation."""
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    agent = _make_agent(env)
    action_dim = int(env.action_dim)

    z0s: List[torch.Tensor] = []
    acts: List[torch.Tensor] = []
    z1s: List[torch.Tensor] = []

    for ep in range(collect_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        pending: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
            latent = agent.sense(obs_body=body, obs_world=world)
            z = latent.z_world.detach().reshape(-1).clone()
            if pending is not None:
                z0_prev, a_prev = pending
                if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                        and torch.isfinite(z).all()):
                    z0s.append(z0_prev)
                    acts.append(a_prev)
                    z1s.append(z.clone())
                pending = None
            a_idx = int(np.random.randint(0, action_dim))
            action = torch.zeros(1, action_dim)
            action[0, a_idx] = 1.0
            pending = (z.clone(), action.reshape(-1).clone())
            # Ignore `done` and run the full step budget (matches the established
            # collection pattern, e.g. v3_exq_711): the env keeps stepping, so we
            # get ~STEPS_PER_EPISODE transitions/episode instead of terminating at
            # the first harm/terminal ~step 12 and starving the dataset.
            _, _harm, _done, _info, obs_dict = env.step(action)
        if (ep + 1) % 10 == 0 or (ep + 1) == collect_episodes:
            print(
                f"  [train] collect seed={seed} ep {ep + 1}/{collect_episodes} "
                f"n_trans={len(z0s)}",
                flush=True,
            )

    if not z0s:
        return (torch.zeros(0, WORLD_DIM), torch.zeros(0, action_dim),
                torch.zeros(0, WORLD_DIM), action_dim)
    return (torch.stack(z0s), torch.stack(acts), torch.stack(z1s), action_dim)


# ---------------------------------------------------------------------------
# Train + evaluate one head on a fixed transition split
# ---------------------------------------------------------------------------


def _train_head(head_type, X_tr, Y_tr, epochs, batch):
    head = _build_head(head_type, X_tr.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    levels = torch.tensor(QUANTILE_LEVELS)
    n = X_tr.shape[0]
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = X_tr[idx], Y_tr[idx]
            opt.zero_grad(set_to_none=True)
            out = head(xb)
            if head_type == "point":
                loss = F.mse_loss(out, yb)
            elif head_type == "hetero":
                loss = _gaussian_nll(out[0], out[1], yb)
            elif head_type == "quantile":
                loss = _pinball_loss(out, yb, levels)
            else:
                loss = _mixture_nll(out[0], out[1], out[2], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
    return head


def _global_sigma(head, X_tr, Y_tr) -> torch.Tensor:
    # homoscedastic per-dim sigma fitted on train residuals of the point head.
    with torch.no_grad():
        pred = head(X_tr)                                     # [N, D] (point head)
        resid = Y_tr - pred
        var = resid.var(dim=0, unbiased=True).clamp(min=1e-8)
    return torch.sqrt(var)                                    # [D]


def _eval_head(head_type, head, X_te, Y_te, sigma_global) -> Dict[str, Any]:
    levels = torch.tensor(QUANTILE_LEVELS)
    with torch.no_grad():
        out = head(X_te)
        mean_pred = _mean_prediction(head_type, out, sigma_global)
        rmse = float(torch.sqrt(((mean_pred - Y_te) ** 2).mean()).item())

        # NLL where a density exists
        if head_type == "hetero":
            nll = float(_gaussian_nll(out[0], out[1], Y_te).item())
        elif head_type == "mixture":
            nll = float(_mixture_nll(out[0], out[1], out[2], Y_te).item())
        elif head_type == "point":
            lv = torch.log(sigma_global ** 2).view(1, -1).expand_as(Y_te)
            mu = out
            nll = float(_gaussian_nll(mu, lv, Y_te).item())
        else:
            nll = float("nan")                               # quantile: no density

        # sample-based CRPS + coverage (uniform estimator)
        if head_type == "point":
            samples = _samples_point(out, sigma_global, CRPS_SAMPLES)
        elif head_type == "hetero":
            samples = _samples_hetero(out[0], out[1], CRPS_SAMPLES)
        elif head_type == "quantile":
            samples = _samples_quantile(out, levels, CRPS_SAMPLES)
        else:
            samples = _samples_mixture(out[0], out[1], out[2], CRPS_SAMPLES)
        crps = _crps_from_samples(samples, Y_te)
        coverage = _coverage_from_samples(samples, Y_te, NOMINAL_COVERAGE)

        # C3 downstream: per-point predictive variance vs realized squared error
        pvar = _predictive_variance(head_type, out, sigma_global)         # [B]
        sq_err = ((mean_pred - Y_te) ** 2).mean(dim=-1)                   # [B]
        pvar_np = pvar.cpu().numpy()
        err_np = sq_err.cpu().numpy()
        if float(np.std(pvar_np)) < 1e-12 or float(np.std(err_np)) < 1e-12:
            prec_err_corr = 0.0                              # constant variance = EMA null
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                _c = float(np.corrcoef(pvar_np, err_np)[0, 1])
            prec_err_corr = _c if math.isfinite(_c) else 0.0

    return {
        "point_rmse": rmse,
        "nll_mean": nll,
        "crps_mean": crps,
        "coverage_80": coverage,
        "coverage_err_80": abs(coverage - NOMINAL_COVERAGE),
        "precision_error_corr": prec_err_corr,
    }


def _arm_config_slice(arm, seed, collect_episodes, steps_per_episode, epochs):
    return {
        "arm_id": arm["arm_id"],
        "head": arm["head"],
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "alpha_world": ALPHA_WORLD,
        "head_hidden": HEAD_HIDDEN,
        "mixture_k": MIXTURE_K,
        "quantile_levels": list(QUANTILE_LEVELS),
        "crps_samples": CRPS_SAMPLES,
        "head_lr": HEAD_LR,
        "head_batch": HEAD_BATCH,
        "epochs": int(epochs),
        "train_frac": TRAIN_FRAC,
        "collect_episodes": int(collect_episodes),
        "steps_per_episode": int(steps_per_episode),
        "env_kwargs": dict(ENV_KWARGS),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    collect_episodes = DRY_RUN_COLLECT if dry_run else COLLECT_EPISODES
    steps_per_episode = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    epochs = DRY_RUN_EPOCHS if dry_run else HEAD_TRAIN_EPOCHS

    arm_results: List[Dict[str, Any]] = []
    # per-seed, per-arm metric tables for aggregation
    crps_by_arm: Dict[str, List[float]] = {a["arm_id"]: [] for a in ARMS}
    nll_by_arm: Dict[str, List[float]] = {a["arm_id"]: [] for a in ARMS}
    coverr_by_arm: Dict[str, List[float]] = {a["arm_id"]: [] for a in ARMS}
    corr_by_arm: Dict[str, List[float]] = {a["arm_id"]: [] for a in ARMS}
    # per-seed point-arm readiness statistic + C1 seed wins
    point_norm_resid_per_seed: List[float] = []
    c1_seed_win = 0
    n_seeds_scored = 0
    n_test_total = 0

    for seed in seeds:
        print(f"Seed {seed} Condition all_heads", flush=True)
        # collect transitions once per seed; shared frozen encoder across all arms
        reset_all_rng(seed)
        Z0, A, Z1, action_dim = _collect_transitions(seed, collect_episodes, steps_per_episode)
        n = Z0.shape[0]
        if n < 8:
            print(f"  seed {seed}: too few transitions ({n}); skipping", flush=True)
            print("verdict: FAIL", flush=True)
            continue
        X = torch.cat([Z0, A], dim=1)
        Y = Z1
        n_tr = max(1, int(round(TRAIN_FRAC * n)))
        X_tr, Y_tr = X[:n_tr], Y[:n_tr]
        X_te, Y_te = X[n_tr:], Y[n_tr:]
        if X_te.shape[0] < 2:
            X_te, Y_te = X_tr, Y_tr
        n_test_total += X_te.shape[0]
        target_std = float(Y_te.std(unbiased=True).clamp(min=1e-8).item())

        # point arm first -> supplies sigma_global + readiness statistic
        seed_metrics: Dict[str, Dict[str, Any]] = {}
        sigma_global = None
        point_head = None
        for ai, arm in enumerate(ARMS):
            with arm_cell(
                seed,
                config_slice=_arm_config_slice(arm, seed, collect_episodes, steps_per_episode, epochs),
                script_path=Path(__file__),
                config_slice_declared=True,
                extra_ineligible_reasons=["shared_frozen_encoder_and_transition_set_across_head_arms"],
            ) as cell:
                head = _train_head(arm["head"], X_tr, Y_tr, epochs, HEAD_BATCH)
                if arm["head"] == "point":
                    point_head = head
                    sigma_global = _global_sigma(head, X_tr, Y_tr)
                sig = sigma_global if sigma_global is not None else torch.ones(WORLD_DIM)
                m = _eval_head(arm["head"], head, X_te, Y_te, sig)
                seed_metrics[arm["arm_id"]] = m
                row = {
                    "arm_id": arm["arm_id"],
                    "head": arm["head"],
                    "seed": seed,
                    "n_train": int(n_tr),
                    "n_test": int(X_te.shape[0]),
                    "target_std": target_std,
                    **m,
                }
                cell.stamp(row)
                arm_results.append(row)
                crps_by_arm[arm["arm_id"]].append(m["crps_mean"])
                if math.isfinite(m["nll_mean"]):
                    nll_by_arm[arm["arm_id"]].append(m["nll_mean"])
                coverr_by_arm[arm["arm_id"]].append(m["coverage_err_80"])
                corr_by_arm[arm["arm_id"]].append(m["precision_error_corr"])

        # readiness statistic (point arm normalised residual)
        point_rmse = seed_metrics["mse_point"]["point_rmse"]
        norm_resid = point_rmse / (target_std + 1e-12)
        point_norm_resid_per_seed.append(norm_resid)

        # C1 per-seed: best distributional CRPS beats point CRPS by frac
        point_crps = seed_metrics["mse_point"]["crps_mean"]
        dist_crps = [seed_metrics[a["arm_id"]]["crps_mean"]
                     for a in ARMS if a["head"] != "point"]
        best_dist = min(dist_crps)
        if best_dist <= point_crps * (1.0 - CRPS_IMPROVE_FRAC):
            c1_seed_win += 1
        n_seeds_scored += 1
        print(
            f"  seed {seed}: norm_resid={norm_resid:.4f} point_crps={point_crps:.5f} "
            f"best_dist_crps={best_dist:.5f}",
            flush=True,
        )
        print("verdict: PASS", flush=True)

    # ---- aggregate ----
    def _mean(xs):
        return float(statistics.fmean(xs)) if xs else float("nan")

    arm_summary = {}
    for a in ARMS:
        aid = a["arm_id"]
        arm_summary[aid] = {
            "crps_mean": _mean(crps_by_arm[aid]),
            "nll_mean": _mean(nll_by_arm[aid]),
            "coverage_err_80_mean": _mean(coverr_by_arm[aid]),
            "precision_error_corr_mean": _mean(corr_by_arm[aid]),
            "n_seeds": len(crps_by_arm[aid]),
        }

    # readiness (post-hoc, but formatted via the shared gate helper)
    median_norm_resid = (float(statistics.median(point_norm_resid_per_seed))
                         if point_norm_resid_per_seed else 0.0)
    readiness_checks = [
        {"name": "heldout_conditional_spread_present",
         "measured": median_norm_resid,
         "threshold": READINESS_NORM_RESIDUAL_FLOOR, "direction": "lower"},
        {"name": "sufficient_test_transitions",
         "measured": float(n_test_total),
         "threshold": float(READINESS_MIN_TEST_TRANSITIONS), "direction": "lower"},
    ]

    # winner (lowest mean CRPS among all arms)
    ranked = sorted(
        [(aid, s["crps_mean"]) for aid, s in arm_summary.items()
         if not math.isnan(s["crps_mean"])],
        key=lambda t: t[1],
    )
    winner = ranked[0][0] if ranked else None
    best_dist_arms = [aid for aid, _ in ranked if aid != "mse_point"]
    point_mean_crps = arm_summary["mse_point"]["crps_mean"]
    best_dist_mean_crps = min(
        (arm_summary[aid]["crps_mean"] for aid in best_dist_arms
         if not math.isnan(arm_summary[aid]["crps_mean"])),
        default=float("nan"),
    )

    # degeneracy: the load-bearing CRPS comparison must have cross-arm spread,
    # and realized errors must vary (else C3 corr is meaningless).
    per_arm_mean_crps = [arm_summary[a["arm_id"]]["crps_mean"] for a in ARMS]
    degen = check_degeneracy({
        "cross_arm_crps": [c for c in per_arm_mean_crps if not math.isnan(c)],
    })

    # readiness routing
    try:
        preconditions = p0_readiness_gate(readiness_checks)
        readiness_met = True
        readiness_note = ""
    except P0NotReady as e:
        preconditions = e.preconditions
        readiness_met = False
        readiness_note = e.reason

    # C1 (load-bearing) evaluated only if readiness met
    c1_pass = False
    if readiness_met and n_seeds_scored > 0:
        c1_pass = c1_seed_win >= _majority(n_seeds_scored)

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = ("readiness unmet: " + readiness_note
                             if readiness_note else "readiness unmet")
    elif not degen["non_degenerate"]:
        outcome = "FAIL"
        label = "distributional_head_comparison_degenerate"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = degen["degeneracy_reason"]
    elif c1_pass:
        outcome = "PASS"
        label = "distributional_head_ranking"
        evidence_direction = "non_contributory"   # diagnostic: promotes nothing
        non_degenerate = True
        degeneracy_reason = ""
    else:
        outcome = "FAIL"
        label = "distributional_heads_no_benefit_over_point"
        evidence_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = ""

    criteria_non_degenerate = {
        "C1_best_dist_crps_beats_point": (not math.isnan(best_dist_mean_crps)
                                          and not math.isnan(point_mean_crps)
                                          and degen["non_degenerate"]),
    }

    interpretation = {
        "label": label,
        "winner_arm": winner,
        "c1_seed_wins": c1_seed_win,
        "n_seeds_scored": n_seeds_scored,
        "c1_majority_needed": _majority(n_seeds_scored) if n_seeds_scored else None,
        "point_mean_crps": point_mean_crps,
        "best_distributional_mean_crps": best_dist_mean_crps,
        "preconditions": preconditions,
        "criteria": [
            {"name": "C1_best_dist_crps_beats_point", "load_bearing": True,
             "passed": bool(c1_pass)},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "readiness_note": readiness_note,
    }

    run_id = "{}_{}_v3".format(
        EXPERIMENT_TYPE, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "degenerate_metrics": degen.get("degenerate_metrics", {}),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "interpretation": interpretation,
        "arm_summary": arm_summary,
        "arm_results": arm_results,
        "config": {
            "seeds": seeds,
            "collect_episodes": collect_episodes,
            "steps_per_episode": steps_per_episode,
            "head_train_epochs": epochs,
            "world_dim": WORLD_DIM,
            "mixture_k": MIXTURE_K,
            "quantile_levels": QUANTILE_LEVELS,
            "crps_improve_frac": CRPS_IMPROVE_FRAC,
            "readiness_norm_residual_floor": READINESS_NORM_RESIDUAL_FLOOR,
        },
        "dry_run": dry_run,
    }
    return manifest


def _write_manifest(manifest: Dict[str, Any]) -> Path:
    evidence_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_path = evidence_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = run_experiment(dry_run=args.dry_run)
    out_path = _write_manifest(manifest)

    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"winner: {manifest['interpretation']['winner_arm']}", flush=True)
    print(f"saved: {out_path}", flush=True)

    _outcome = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=manifest["run_id"],
        dry_run=args.dry_run,
    )
