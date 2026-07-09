#!/opt/local/bin/python3
"""
V3-EXQ-716a -- SD-063 conditional predictive-uncertainty head: CLAIM-TAGGED
evidence-confirmation run.

EXPERIMENT_PURPOSE = evidence. TAGS SD-063 (claim_ids = ["SD-063"]). This is the
complementary claim-scoring companion to the V3-EXQ-716 diagnostic
(v3_exq_716_sd063_conditional_uncertainty_validation), which PASSED all three
load-bearing criteria and cleared SD-063's v3_pending gate (2026-07-09) but was
tagged experiment_purpose=diagnostic / claim_ids=[] / non_contributory and so
deposited NO claim-scoring evidence. SD-063 therefore sits at status=candidate
with no supporting evidence entry. This run reproduces the SAME three criteria
(C1/C2/C3) under the SAME frozen-encoder-warmup + joint-training protocol so a
PASS deposits a proper `supports` evidence_direction on SD-063, enabling a
candidate->provisional promotion. It is NOT a correction of 716 (no supersedes);
716's diagnostic verdict stands and this confirms it for governance weight.

WHAT SD-063 BUILT (confirmed here)
----------------------------------
ree_core/predictors/e2_world_uncertainty.py: E2WorldUncertaintyHead, a
distribution-free quantile/pinball head over (z_world, action) emitting a
per-input predictive spread (the V3-EXQ-712 winner form). It feeds E3 commitment
gating (E3Config.use_conditional_precision_gate) in place of the state-blind
running-variance EMA. The head is a SEPARATE module reading DETACHED z_world so it
cannot explain away the SD-031 E2WorldForward agency residual.

THE TWO LOAD-BEARING QUESTIONS (from the SD-063 doc "What v3_pending gates on")
------------------------------------------------------------------------------
(a) Does the head's per-point predictive variance improve E3 commitment gating
    over the EMA, and does the advantage SURVIVE JOINT TRAINING? The V3-EXQ-712
    diagnostic trained heads on a FIXED transition set from a FIXED (untrained)
    encoder -- effectively single-phase P1. Here the encoder is TRAINED (P0) and
    KEPT LEARNING while the head trains (the joint phase), so the head must chase a
    MOVING z_world target. Operationalised (same statistics as 712, which is what
    makes the E3 commit gate better -- the gate reads predictive_variance):
      - CRPS(quantile) < CRPS(point)                 [C1, distributional benefit]
      - precision_error_corr(quantile) > EMA null ~0 [C2, state-conditional signal
        the running-variance EMA structurally cannot carry -- corr(point) := 0]
(b) Is the SD-031 agency residual PRESERVED? E2 is an agency detector via
    residual = z_world_observed - E2WorldForward(z_world_prev, a_actual). A
    variance head could absorb the agent-caused component of next-state variance
    into "expected spread," collapsing the action-discrimination the forward model
    carries. The FALSIFIER: the E2WorldForward action-discrimination gap
    ||E2WF(z,a_actual) - E2WF(z,a_cf)|| with the head trained jointly (HEAD_ON)
    must NOT collapse relative to no head (HEAD_OFF) [C3]. Because the head is a
    separate detached module the structural prediction is "preserved"; C3 confirms
    it empirically under joint training (the contract asserts the gradient
    isolation; this asserts the behavioural consequence).

Note on operationalisation: (a) measures precision_error_corr (the per-point
signal the commit gate consumes) rather than running a full behavioural E3.select
commit-quality ablation. corr > 0 is exactly the property that lets the gate
withhold commitment where THIS prediction is unreliable, which the EMA (corr 0)
cannot. A behavioural commit-quality ablation is a sensible follow-on, not needed
to falsify the substrate.

DESIGN (identical to V3-EXQ-716)
--------------------------------
world_dim = 128 (REEConfig.large): REQUIRED by E2WorldForward's discriminative
arm (the 2026-06-06 cluster autopsy dim-guard); the uncertainty head is
dim-agnostic. alpha_world = 0.9 (z_world must faithfully encode world obs).
Two conditions per seed (arm grid):
  HEAD_ON : P0 encoder warmup (SD-009 event-contrastive + SD-018 resource
            proximity); P1 JOINT (encoder KEEPS training) + train E2WorldForward
            + point head + the SD-063 quantile head, all on DETACHED z_world
            transitions; P2 frozen held-out eval.
  HEAD_OFF: identical P0 + P1 (encoder + E2WorldForward + point head) but the
            quantile head is constructed (RNG parity) and NEVER trained -- the
            C3 agency-gap control.
The point head + its global homoscedastic sigma is the EMA/state-blind null (its
per-point predictive variance is constant -> precision_error_corr := 0).

PRE-REGISTERED READINESS + SELF-ROUTE (evidence, falsifiable)
-------------------------------------------------------------
Readiness (load-bearing preconditions, measured on positive controls):
  R1 heldout_conditional_spread_present: point-arm normalised held-out residual
     (point_rmse / target_std) >= READINESS_NORM_RESIDUAL_FLOOR. If the dynamics
     are effectively deterministic in z_world there is nothing for a distributional
     head to model and the comparison is vacuous.
  R2 sufficient_test_transitions: total held-out transitions >= floor.
  R3 e2_world_forward_discriminates_off: HEAD_OFF action-discrimination gap
     (cf_gap) >= READINESS_CF_GAP_FLOOR -- the SD-031 residual must actually
     discriminate for "preserved" (C3) to be a meaningful test.
If any readiness precondition is UNMET -> self-route substrate_not_ready_requeue
(non_contributory, non_degenerate=False), NEVER a substrate-verdict label.

If readiness MET:
  C1 (load-bearing): CRPS(quantile) <= CRPS(point) * (1 - CRPS_IMPROVE_FRAC) on a
     strict majority of seeds.
  C2 (load-bearing): precision_error_corr(quantile) >= CORR_FLOOR on a strict
     majority of seeds (and, by construction, > the point/EMA null of 0).
  C3 (load-bearing): agency gap preserved -- cf_gap(HEAD_ON) >=
     cf_gap(HEAD_OFF) * C3_PRESERVE_FRAC on a strict majority of seeds (one-sided:
     the failure mode is COLLAPSE via explaining-away).
  PASS (all three) -> label sd063_conditional_uncertainty_confirmed,
     evidence_direction = supports (SD-063).
  else FAIL with the specific failing-criterion label,
     evidence_direction = weakens (SD-063).

Claim-tagged evidence run: a PASS deposits `supports` on SD-063 (enabling
candidate->provisional); a criteria-FAIL deposits `weakens`; a readiness-unmet or
degenerate self-route deposits `non_contributory` (scoring-excluded).

See ree_core/predictors/e2_world_uncertainty.py (module under test),
    ree_core/predictors/e2_world.py (E2WorldForward -- the SD-031 residual),
    docs/architecture/sd_063_e2_conditional_uncertainty_head.md,
    experiments/v3_exq_716_sd063_conditional_uncertainty_validation.py (the
      diagnostic this confirms; same design),
    experiments/v3_exq_712_distributional_world_forward_heads.py (motivating
      diagnostic; CRPS/quantile-sampling patterns reused here).
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
import torch.optim as optim

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.predictors.e2_world import E2WorldForward, E2WorldConfig
from ree_core.predictors.e2_world_uncertainty import (
    E2WorldUncertaintyHead, E2WorldUncertaintyConfig, QUANTILE_LEVELS)


EXPERIMENT_TYPE = "v3_exq_716a_sd063_conditional_uncertainty_confirmation"
QUEUE_ID = "V3-EXQ-716a"
SUPERSEDES = None                         # complementary confirmation, NOT a correction of 716
CONFIRMS = "v3_exq_716_sd063_conditional_uncertainty_validation"  # the diagnostic this scores
CLAIM_IDS: List[str] = ["SD-063"]         # evidence: claim-tagged so a PASS deposits scoring evidence
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ----- Latent / model dims -----
WORLD_DIM = 128          # REQUIRED by E2WorldForward discriminative arm (dim-guard)
SELF_DIM = 32
ALPHA_WORLD = 0.9        # z_world must faithfully encode world obs (SD-008)
HEAD_HIDDEN = 128
CRPS_SAMPLES = 100

# ----- Schedule -----
SEEDS = [42, 43, 44]
CONDITIONS = ["HEAD_ON", "HEAD_OFF"]
P0_EPISODES = 60         # encoder warmup (SD-009 + SD-018)
P1_EPISODES = 80         # joint: encoder keeps training + E2WF + heads train
P2_EPISODES = 30         # frozen held-out eval
STEPS_PER_EP = 120
TOTAL_EPISODES = P0_EPISODES + P1_EPISODES + P2_EPISODES  # episodes_per_run
TRAIN_FRAC = 0.8         # split of the P2 held-out transitions for head fit vs eval

REPLAY_BUF_MAX = 6000
BATCH_SIZE = 128
LR_AGENT = 3e-4
LR_FWD = 3e-4
LR_HEAD = 1e-3

DRY_RUN_SEEDS = [42]
DRY_RUN_CONDITIONS = ["HEAD_ON", "HEAD_OFF"]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 20

# ----- Pre-registered thresholds (identical to V3-EXQ-716) -----
READINESS_NORM_RESIDUAL_FLOOR = 0.05     # 712 floor: conditional spread must exist
READINESS_MIN_TEST_TRANSITIONS = 200
READINESS_CF_GAP_FLOOR = 1e-3            # E2WF action-discrimination must be non-trivial at OFF
CRPS_IMPROVE_FRAC = 0.02                 # C1: quantile CRPS below point by this frac
CORR_FLOOR = 0.15                        # C2: precision_error_corr floor (< 712's 0.379)
C3_PRESERVE_FRAC = 0.75                  # C3: ON gap must retain >=75% of OFF gap


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
    use_proxy_fields=True,          # needed for resource_field_view (SD-018) + event labels
)


# ---------------------------------------------------------------------------
# Point head (the EMA / state-blind null): MLP -> mean; MSE loss; constant sigma.
# The SD-063 quantile head is imported from the substrate (E2WorldUncertaintyHead).
# ---------------------------------------------------------------------------


class PointHead(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN), nn.ReLU(),
        )
        self.mean = nn.Linear(HEAD_HIDDEN, d)

    def forward(self, x):
        return self.mean(self.trunk(x))


def _global_sigma(point_head: PointHead, X_tr, Y_tr) -> torch.Tensor:
    with torch.no_grad():
        resid = Y_tr - point_head(X_tr)
        var = resid.var(dim=0, unbiased=True).clamp(min=1e-8)
    return torch.sqrt(var)


# ----- CRPS / sampling estimators (uniform across heads; reused from 712) -----


def _samples_point(mean, sigma, n):
    b, d = mean.shape
    return mean.unsqueeze(-1) + sigma.view(1, d, 1) * torch.randn(b, d, n)


def _samples_quantile(quantiles, levels, n):
    b, d, q = quantiles.shape
    qs, _ = torch.sort(quantiles, dim=-1)
    lv = levels.to(qs.device)
    u = torch.rand(b, d, n)
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
    y = target.unsqueeze(-1)
    term1 = (samples - y).abs().mean(dim=-1)
    n = samples.shape[-1]
    perm = samples[..., torch.randperm(n)]
    term2 = 0.5 * (samples - perm).abs().mean(dim=-1)
    return float((term1 - term2).mean().item())


def _quantile_median(quantiles) -> torch.Tensor:
    qs, _ = torch.sort(quantiles, dim=-1)
    mid = qs.shape[-1] // 2
    return qs[..., mid]


def _corr(pvar_np, err_np) -> float:
    if float(np.std(pvar_np)) < 1e-12 or float(np.std(err_np)) < 1e-12:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = float(np.corrcoef(pvar_np, err_np)[0, 1])
    return c if math.isfinite(c) else 0.0


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_event_classifier=True,
    )
    return REEAgent(cfg)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# One (seed, condition) run
# ---------------------------------------------------------------------------


def run_condition(seed: int, condition: str, dry_run: bool = False) -> Dict[str, Any]:
    p0 = DRY_RUN_P0 if dry_run else P0_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_EPISODES
    p2 = DRY_RUN_P2 if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS if dry_run else STEPS_PER_EP
    total_eps = p0 + p1 + p2

    head_on = (condition == "HEAD_ON")
    print(f"Seed {seed} Condition {condition}", flush=True)

    env = _make_env(seed)
    agent = _make_agent(env, seed)
    device = agent.device
    action_dim = int(env.action_dim)

    # E2WorldForward (SD-031 comparator) -- trained in BOTH conditions.
    fwd = E2WorldForward(E2WorldConfig(
        use_e2_world_forward=True, z_world_dim=WORLD_DIM, action_dim=action_dim,
        hidden_dim=HEAD_HIDDEN)).to(device)
    # Point head (EMA/state-blind null) -- trained in BOTH conditions.
    point_head = PointHead(WORLD_DIM + action_dim, WORLD_DIM).to(device)
    # SD-063 quantile head (module under test). CONSTRUCTED in both (RNG parity);
    # TRAINED only in HEAD_ON.
    unc_head = E2WorldUncertaintyHead(E2WorldUncertaintyConfig(
        use_e2_world_uncertainty=True, z_world_dim=WORLD_DIM, action_dim=action_dim,
        hidden_dim=HEAD_HIDDEN)).to(device)

    agent_opt = optim.Adam(list(agent.parameters()), lr=LR_AGENT)
    fwd_opt = optim.Adam(fwd.parameters(), lr=LR_FWD)
    point_opt = optim.Adam(point_head.parameters(), lr=LR_HEAD)
    unc_opt = optim.Adam(unc_head.parameters(), lr=LR_HEAD)
    levels = torch.tensor(list(QUANTILE_LEVELS))

    # Replay of (z_world_t, action_idx, z_world_{t+1}) for P1 head/fwd training.
    replay: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
    # Held-out P2 transitions (collected under the frozen final encoder).
    z0_te: List[torch.Tensor] = []
    a_te: List[torch.Tensor] = []
    z1_te: List[torch.Tensor] = []

    p0_loss_first: Optional[float] = None
    p0_loss_last: Optional[float] = None
    unc_loss_first: Optional[float] = None
    unc_loss_last: Optional[float] = None

    prev_ttype = "none"
    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev_idx: Optional[int] = None
        phase = "P0" if ep < p0 else ("P1" if ep < p0 + p1 else "P2")
        in_p2 = (phase == "P2")

        for _step in range(steps_per):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            if obs_body.dim() == 1:
                obs_body = obs_body.unsqueeze(0)
            if obs_world.dim() == 1:
                obs_world = obs_world.unsqueeze(0)

            latent = agent.sense(obs_body=obs_body, obs_world=obs_world)
            z_world_now = latent.z_world.detach().reshape(1, -1).clone()

            # random action (transition collection); action selection is not under test
            a_idx = int(np.random.randint(0, action_dim))

            # record transition for replay (P0/P1) or held-out (P2)
            if z_world_prev is not None and action_prev_idx is not None:
                if in_p2:
                    z0_te.append(z_world_prev.reshape(-1).clone())
                    a_te.append(_onehot(action_prev_idx, action_dim, device).reshape(-1).clone())
                    z1_te.append(z_world_now.reshape(-1).clone())
                else:
                    replay.append((z_world_prev.clone(), action_prev_idx, z_world_now.clone()))
                    if len(replay) > REPLAY_BUF_MAX:
                        replay = replay[-REPLAY_BUF_MAX:]

            # ENCODER TRAINING (P0 AND P1 -- the joint phase keeps z_world moving).
            # SD-009 event-contrastive + SD-018 resource-proximity are the load-bearing
            # z_world discriminative signals (they read `latent` directly; no trajectory
            # bookkeeping needed). This is the phased-training P0 warmup the SD docs
            # specify; keeping it live through P1 is the "joint training" the head must
            # survive.
            if not in_p2:
                agent_opt.zero_grad()
                rfv = obs_dict.get("resource_field_view", None)
                rp_t = float(rfv.max().item()) if rfv is not None else 0.0
                loss = (agent.compute_resource_proximity_loss(rp_t, latent)
                        + agent.compute_event_contrastive_loss(prev_ttype, latent))
                if loss.requires_grad:
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    agent_opt.step()
                    lval = float(loss.detach().item())
                    if phase == "P0":
                        if p0_loss_first is None:
                            p0_loss_first = lval
                        p0_loss_last = lval

            # HEAD / FORWARD-MODEL TRAINING (P1 only), on DETACHED transitions
            if phase == "P1" and len(replay) >= BATCH_SIZE:
                idx = random.sample(range(len(replay)), BATCH_SIZE)
                z0b = torch.cat([replay[i][0] for i in idx], dim=0).detach()
                z1b = torch.cat([replay[i][2] for i in idx], dim=0).detach()
                ab = torch.zeros(BATCH_SIZE, action_dim, device=device)
                for bi, i in enumerate(idx):
                    ab[bi, replay[i][1]] = 1.0

                # E2WorldForward (SD-031): stop-grad target
                fwd_opt.zero_grad()
                fwd.compute_loss(fwd(z0b, ab), z1b).backward()
                fwd_opt.step()

                # point head (EMA null baseline)
                xb = torch.cat([z0b, ab], dim=1)
                point_opt.zero_grad()
                F.mse_loss(point_head(xb), z1b).backward()
                point_opt.step()

                # SD-063 quantile head -- HEAD_ON only; detached inputs + target
                if head_on:
                    unc_opt.zero_grad()
                    u_loss = unc_head.compute_loss(unc_head(z0b, ab), z1b)
                    u_loss.backward()
                    unc_opt.step()
                    uval = float(u_loss.detach().item())
                    if unc_loss_first is None:
                        unc_loss_first = uval
                    unc_loss_last = uval

            # Do NOT break on done -- follow V3-EXQ-712: run the full step budget so the
            # held-out set is not starved by early harm-terminals (~step 12).
            _, _harm, _done, info, obs_dict_next = env.step(a_idx)
            prev_ttype = info.get("transition_type", "none")
            z_world_prev = z_world_now
            action_prev_idx = a_idx
            obs_dict = obs_dict_next

        if (ep + 1) % 50 == 0 or (ep + 1) == total_eps:
            print(f"  [train] label seed={seed} cond={condition} ep {ep+1}/{total_eps} "
                  f"phase={phase} replay={len(replay)} test={len(z0_te)}", flush=True)

    # ---- P2 evaluation on held-out transitions ----
    n_te = len(z0_te)
    metrics: Dict[str, Any] = {
        "seed": seed, "condition": condition, "n_test": n_te,
        "point_rmse": float("nan"), "target_std": float("nan"),
        "norm_resid": float("nan"),
        "crps_point": float("nan"), "crps_quantile": float("nan"),
        "precision_error_corr_quantile": 0.0, "precision_error_corr_point": 0.0,
        "cf_gap": float("nan"), "world_forward_r2": float("nan"),
        "p0_loss_first": p0_loss_first, "p0_loss_last": p0_loss_last,
        "unc_loss_first": unc_loss_first, "unc_loss_last": unc_loss_last,
    }
    if n_te < 8:
        print(f"verdict: FAIL (seed {seed} {condition}: too few held-out transitions {n_te})", flush=True)
        metrics["insufficient_transitions"] = True
        return metrics

    Z0 = torch.stack(z0_te)
    A = torch.stack(a_te)
    Z1 = torch.stack(z1_te)
    X = torch.cat([Z0, A], dim=1)
    n_tr = max(1, int(round(TRAIN_FRAC * n_te)))
    X_tr, Y_tr = X[:n_tr], Z1[:n_tr]
    X_te, Y_te, Z0_te2, A_te2 = X[n_tr:], Z1[n_tr:], Z0[n_tr:], A[n_tr:]
    if X_te.shape[0] < 2:
        X_te, Y_te, Z0_te2, A_te2 = X_tr, Y_tr, Z0[:n_tr], A[:n_tr]
    target_std = float(Y_te.std(unbiased=True).clamp(min=1e-8).item())

    with torch.no_grad():
        # point head: rmse, sigma, CRPS, corr(:=0 by construction)
        sigma = _global_sigma(point_head, X_tr, Y_tr)
        p_mean = point_head(X_te)
        point_rmse = float(torch.sqrt(((p_mean - Y_te) ** 2).mean()).item())
        crps_point = _crps_from_samples(_samples_point(p_mean, sigma, CRPS_SAMPLES), Y_te)
        pvar_point = (sigma ** 2).mean().expand(X_te.shape[0]).clone()
        sq_err_point = ((p_mean - Y_te) ** 2).mean(dim=-1)
        corr_point = _corr(pvar_point.cpu().numpy(), sq_err_point.cpu().numpy())

        # E2WorldForward: world_forward_r2 + action-discrimination gap (SD-031).
        # Z0_te2 / A_te2 / Y_te are the aligned held-out test slice.
        z_pred_fwd = fwd(Z0_te2, A_te2)
        ss_res = float(((Y_te - z_pred_fwd) ** 2).sum().item())
        ss_tot = float(((Y_te - Y_te.mean(dim=0)) ** 2).sum().item())
        world_forward_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        # cf action for the gap
        cf = torch.zeros_like(A_te2)
        for bi in range(A_te2.shape[0]):
            aidx = int(A_te2[bi].argmax().item())
            choices = [j for j in range(action_dim) if j != aidx]
            cf[bi, random.choice(choices)] = 1.0
        gap = (fwd(Z0_te2, A_te2) - fwd(Z0_te2, cf)).norm(dim=-1).mean()
        cf_gap = float(gap.item())

        # quantile head (HEAD_ON): CRPS + precision_error_corr
        crps_quantile = float("nan")
        corr_quantile = 0.0
        if head_on:
            q = unc_head(Z0_te2, A_te2)                       # [B, D, Q]
            crps_quantile = _crps_from_samples(_samples_quantile(q, levels, CRPS_SAMPLES), Y_te)
            pvar_q = unc_head.predictive_variance(Z0_te2, A_te2)   # [B]
            q_mean = _quantile_median(q)
            sq_err_q = ((q_mean - Y_te) ** 2).mean(dim=-1)
            corr_quantile = _corr(pvar_q.cpu().numpy(), sq_err_q.cpu().numpy())

    metrics.update({
        "point_rmse": point_rmse,
        "target_std": target_std,
        "norm_resid": point_rmse / (target_std + 1e-12),
        "crps_point": crps_point,
        "crps_quantile": crps_quantile,
        "precision_error_corr_quantile": corr_quantile,
        "precision_error_corr_point": corr_point,
        "cf_gap": cf_gap,
        "world_forward_r2": world_forward_r2,
        "n_test_eval": int(X_te.shape[0]),
    })
    print(f"  seed {seed} {condition}: norm_resid={metrics['norm_resid']:.4f} "
          f"crps_point={crps_point:.5f} crps_quant={crps_quantile if head_on else float('nan'):.5f} "
          f"corr_quant={corr_quantile:.3f} cf_gap={cf_gap:.4f} wf_r2={world_forward_r2:.3f}", flush=True)
    print("verdict: PASS", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# Aggregate + self-route
# ---------------------------------------------------------------------------


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    conditions = DRY_RUN_CONDITIONS if dry_run else CONDITIONS
    p0 = DRY_RUN_P0 if dry_run else P0_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_EPISODES
    p2 = DRY_RUN_P2 if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS if dry_run else STEPS_PER_EP

    arm_results: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for seed in seeds:
        for cond in conditions:
            config_slice = {
                "condition": cond, "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
                "alpha_world": ALPHA_WORLD, "head_hidden": HEAD_HIDDEN,
                "quantile_levels": list(QUANTILE_LEVELS), "crps_samples": CRPS_SAMPLES,
                "p0_episodes": p0, "p1_episodes": p1, "p2_episodes": p2,
                "steps_per_episode": steps_per, "batch_size": BATCH_SIZE,
                "lr_agent": LR_AGENT, "lr_fwd": LR_FWD, "lr_head": LR_HEAD,
                "train_frac": TRAIN_FRAC, "env_kwargs": dict(ENV_KWARGS),
            }
            with arm_cell(
                seed,
                config_slice=config_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                extra_ineligible_reasons=["joint_agent_forward_head_training_shared_encoder"],
            ) as cell:
                m = run_condition(seed, cond, dry_run=dry_run)
                cell.stamp(m)
            arm_results.append(m)
            by_key[(seed, cond)] = m

    # ---- readiness (positive-control preconditions) ----
    on_rows = [by_key[(s, "HEAD_ON")] for s in seeds if (s, "HEAD_ON") in by_key]
    off_rows = [by_key[(s, "HEAD_OFF")] for s in seeds if (s, "HEAD_OFF") in by_key]
    norm_resids = [r["norm_resid"] for r in on_rows if r.get("norm_resid") == r.get("norm_resid")]
    n_test_total = sum(int(r.get("n_test_eval", 0) or 0) for r in on_rows)
    off_cf_gaps = [r["cf_gap"] for r in off_rows if r.get("cf_gap") == r.get("cf_gap")]
    median_norm_resid = float(statistics.median(norm_resids)) if norm_resids else 0.0
    median_off_cf_gap = float(statistics.median(off_cf_gaps)) if off_cf_gaps else 0.0

    readiness_checks = [
        {"name": "heldout_conditional_spread_present", "measured": median_norm_resid,
         "threshold": READINESS_NORM_RESIDUAL_FLOOR, "direction": "lower",
         "control": "point-arm normalised held-out residual (same statistic C1/C2 consume)"},
        {"name": "sufficient_test_transitions", "measured": float(n_test_total),
         "threshold": float(READINESS_MIN_TEST_TRANSITIONS), "direction": "lower",
         "control": "total held-out transitions across HEAD_ON seeds"},
        {"name": "e2_world_forward_discriminates_off", "measured": median_off_cf_gap,
         "threshold": READINESS_CF_GAP_FLOOR, "direction": "lower",
         "control": "HEAD_OFF action-discrimination gap -- the SD-031 residual C3 tests"},
    ]
    try:
        preconditions = p0_readiness_gate(readiness_checks)
        readiness_met = True
        readiness_note = ""
    except P0NotReady as e:
        preconditions = e.preconditions
        readiness_met = False
        readiness_note = e.reason

    # ---- per-seed criteria ----
    c1_win = c2_win = c3_win = 0
    n_scored = 0
    per_seed = []
    for s in seeds:
        on = by_key.get((s, "HEAD_ON"))
        off = by_key.get((s, "HEAD_OFF"))
        if not on or not off:
            continue
        if on.get("insufficient_transitions") or off.get("insufficient_transitions"):
            continue
        n_scored += 1
        cp, cq = on.get("crps_point"), on.get("crps_quantile")
        c1 = (cp == cp and cq == cq and cq <= cp * (1.0 - CRPS_IMPROVE_FRAC))
        c2 = on.get("precision_error_corr_quantile", 0.0) >= CORR_FLOOR
        og, ng = off.get("cf_gap"), on.get("cf_gap")
        c3 = (og == og and ng == ng and og > 0 and ng >= og * C3_PRESERVE_FRAC)
        c1_win += int(bool(c1)); c2_win += int(bool(c2)); c3_win += int(bool(c3))
        per_seed.append({"seed": s, "c1": bool(c1), "c2": bool(c2), "c3": bool(c3),
                         "crps_point": cp, "crps_quantile": cq,
                         "corr_quantile": on.get("precision_error_corr_quantile"),
                         "cf_gap_on": ng, "cf_gap_off": og})

    maj = _majority(n_scored) if n_scored else 1
    c1_pass = readiness_met and n_scored > 0 and c1_win >= maj
    c2_pass = readiness_met and n_scored > 0 and c2_win >= maj
    c3_pass = readiness_met and n_scored > 0 and c3_win >= maj

    # ---- degeneracy net (the load-bearing comparisons must have spread) ----
    on_crps_quant = [r.get("crps_quantile") for r in on_rows
                     if r.get("crps_quantile") == r.get("crps_quantile")]
    on_corr = [r.get("precision_error_corr_quantile") for r in on_rows]
    p0_moved = [((r.get("p0_loss_first") or 0.0) - (r.get("p0_loss_last") or 0.0)) for r in on_rows
                if r.get("p0_loss_first") is not None and r.get("p0_loss_last") is not None]
    unc_moved = [((r.get("unc_loss_first") or 0.0) - (r.get("unc_loss_last") or 0.0)) for r in on_rows
                 if r.get("unc_loss_first") is not None and r.get("unc_loss_last") is not None]
    degen = check_degeneracy({
        "crps_quantile_across_seeds": on_crps_quant,
        "cf_gap_off_across_seeds": off_cf_gaps,
    })
    # non-degeneracy also requires the encoder + head actually moved (else vacuous)
    encoder_moved = bool(p0_moved) and (statistics.fmean(p0_moved) > 0)
    head_moved = bool(unc_moved) and (statistics.fmean(unc_moved) > 0)
    non_degenerate = bool(degen["non_degenerate"]) and encoder_moved and head_moved
    degeneracy_reason = degen.get("degeneracy_reason", "")
    if not encoder_moved:
        degeneracy_reason = (degeneracy_reason + "; encoder P0 loss did not decrease").strip("; ")
    if not head_moved:
        degeneracy_reason = (degeneracy_reason + "; quantile head loss did not decrease").strip("; ")

    # ---- self-route (evidence: PASS -> supports, criteria-FAIL -> weakens) ----
    if not readiness_met:
        outcome, label, evidence_direction = "FAIL", "substrate_not_ready_requeue", "non_contributory"
        non_degenerate = False
        if not degeneracy_reason:
            degeneracy_reason = "readiness unmet: " + (readiness_note or "precondition")
    elif not non_degenerate:
        outcome, label, evidence_direction = "FAIL", "sd063_confirmation_degenerate", "non_contributory"
    elif c1_pass and c2_pass and c3_pass:
        outcome, label, evidence_direction = "PASS", "sd063_conditional_uncertainty_confirmed", "supports"
    else:
        fails = []
        if not c1_pass:
            fails.append("crps_no_benefit_under_joint_training")
        if not c2_pass:
            fails.append("precision_error_corr_below_floor")
        if not c3_pass:
            fails.append("sd031_agency_residual_collapsed")
        outcome, label, evidence_direction = "FAIL", "sd063_" + "+".join(fails), "weakens"

    criteria_non_degenerate = {
        "C1_crps_quantile_beats_point": bool(degen["non_degenerate"] and encoder_moved and head_moved),
        "C2_precision_error_corr": bool(head_moved and float(np.std(on_corr)) > 1e-9
                                        if len(on_corr) > 1 else head_moved),
        "C3_agency_gap_preserved": bool(encoder_moved and median_off_cf_gap > READINESS_CF_GAP_FLOOR),
    }

    interpretation = {
        "label": label,
        "readiness_met": readiness_met,
        "readiness_note": readiness_note,
        "n_seeds_scored": n_scored,
        "c1_seed_wins": c1_win, "c2_seed_wins": c2_win, "c3_seed_wins": c3_win,
        "majority_needed": maj,
        "preconditions": preconditions,
        "criteria": [
            {"name": "C1_crps_quantile_beats_point", "load_bearing": True, "passed": bool(c1_pass)},
            {"name": "C2_precision_error_corr_over_ema_null", "load_bearing": True, "passed": bool(c2_pass)},
            {"name": "C3_sd031_agency_residual_preserved", "load_bearing": True, "passed": bool(c3_pass)},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "per_seed": per_seed,
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
        "confirms": CONFIRMS,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "degenerate_metrics": degen.get("degenerate_metrics", {}),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "interpretation": interpretation,
        "arm_results": arm_results,
        "config": {
            "seeds": seeds, "conditions": conditions,
            "world_dim": WORLD_DIM, "self_dim": SELF_DIM, "alpha_world": ALPHA_WORLD,
            "p0_episodes": p0, "p1_episodes": p1, "p2_episodes": p2,
            "steps_per_episode": steps_per, "quantile_levels": list(QUANTILE_LEVELS),
            "crps_improve_frac": CRPS_IMPROVE_FRAC, "corr_floor": CORR_FLOOR,
            "c3_preserve_frac": C3_PRESERVE_FRAC,
            "readiness_norm_residual_floor": READINESS_NORM_RESIDUAL_FLOOR,
            "readiness_cf_gap_floor": READINESS_CF_GAP_FLOOR,
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
    print(f"saved: {out_path}", flush=True)

    _outcome = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=manifest["run_id"],
        dry_run=args.dry_run,
    )
