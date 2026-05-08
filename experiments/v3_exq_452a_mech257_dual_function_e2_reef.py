#!/opt/local/bin/python3
"""
V3-EXQ-452a -- MECH-257 Dual-Function E2_harm_s Diagnostic: Reef-Enriched Substrate.

Claims: MECH-257, SD-013, ARC-033
Supersedes: V3-EXQ-452 (non_contributory due to monostrategy: agent adopts single
fixed route, preventing behavioral diversity for comparator separation measurement).

Reef enrichment (SD-054): reef_enabled=True adds coral-reef safe zones + food-attracted
hazard drift. Creates two behavioral attractors breaking monostrategy on 10x10 grid.

MECH-257 hypothesis: a SINGLE E2_harm_s forward-model substrate can serve
two roles without one degrading the other:

  (a) COMPARATOR (SD-003 counterfactual attribution):
      causal_sig = E2(z_harm_s, a_actual) vs E2(z_harm_s, a_cf)
      -- what matters is DELTA SEPARATION in prediction space.

  (b) EVALUATOR (MECH-258 precision-weighted pain PE):
      pe = ||z_harm_s_pred - z_harm_s_next||
      -- what matters is per-step L2 ERROR MAGNITUDE correlating with surprise.

The question: does training a single E2_harm_s on the standard forward loss
degrade either role? Or are the two roles compatible?

Hypothesis: the forward-model L2 objective is neutral with respect to both
downstream uses. Comparator uses rotation in delta-space; evaluator uses
error magnitude. Both should emerge from ||z_pred - z_next||^2 training.

Conditions (3 arms, 1-condition-per-column):
  DUAL_STD           -- single E2_harm_s, standard forward loss only.
  DUAL_INTERVENTIONAL -- single E2_harm_s, forward + SD-013 contrastive
                         margin loss (use_interventional=True,
                         interventional_fraction=0.3, margin=0.1).
  DUAL_OFF_BASELINE  -- no E2_harm_s training. An untrained random
                        projection acts as comparator; PE path uses
                        raw z_harm_s identity as prediction.

Phased training (per condition):
  P0: encoder warmup (HarmEncoder + AffectiveHarmEncoder, SD-018 proximity
      + SD-020-style accumulated-harm supervision). 30 eps.
  P1: encoders frozen; train E2_harm_s (skipped in DUAL_OFF_BASELINE).
      80 eps.
  P2: evaluation only. 20 eps.

Metrics (collected in P2):
  (a) counterfactual_separation: mean ||E2(z_harm_s, a_actual) -
      E2(z_harm_s, a_cf)||_2 across P2 steps; a_cf is a uniformly-random
      alternative action != a_actual.
  (b) pe_harm_surprise_corr: Pearson correlation between per-step E2
      prediction error ||z_harm_s_pred - z_harm_s_next||_2 and a
      harm_surprise scalar derived inline from an EMA of accumulated_harm
      (SD-020 style; computed locally, does not require the flag).
  (c) forward_r2: reference forward-model goodness-of-fit in P2.
  (d) mean_dacc_score_bias: only populated if use_dacc=True; we keep
      use_dacc=False for this diagnostic to keep E2 isolated.

Acceptance criteria:
  C1 comparator_valid: counterfactual_separation > 0.05 in >=2/3 seeds
     for DUAL_STD AND DUAL_INTERVENTIONAL.
  C2 evaluator_valid: pe_harm_surprise_corr > 0.15 in >=2/3 seeds for
     DUAL_STD AND DUAL_INTERVENTIONAL.
  C3 interventional_improves_comparator (diagnostic refinement):
     DUAL_INTERVENTIONAL counterfactual_separation > DUAL_STD
     counterfactual_separation in >=2/3 seeds.
  C4 interventional_does_not_harm_evaluator (diagnostic refinement):
     DUAL_INTERVENTIONAL pe_harm_surprise_corr >=
     DUAL_STD pe_harm_surprise_corr - 0.1 in >=2/3 seeds.

Overall PASS = C1 AND C2 (dual-function viable).
C3 / C4 are reported but do not gate the outcome.

experiment_purpose: "diagnostic"
  Rationale: this is root-cause discrimination about whether a single
  substrate supports two roles simultaneously. Experiment_purpose=diagnostic
  is excluded from governance confidence scoring.

run_id: v3_exq_452_mech257_dual_function_e2_<timestamp>_v3
architecture_epoch: ree_hybrid_guardrails_v1
claim_ids: ["MECH-257", "SD-013", "ARC-033"]

See REE_assembly/docs/architecture/self_attribution_per_stream.md
See ree-v3/CLAUDE.md "SD-032b / MECH-258 / MECH-260 / ARC-058" and
"ARC-033: E2_harm_s Forward Model" sections.
"""

import sys
import json
import math
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder
from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig


EXPERIMENT_TYPE = "v3_exq_452a_mech257_dual_function_e2_reef"
CLAIM_IDS = ["MECH-257", "SD-013", "ARC-033"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 120
P0_EPS = 30
P1_EPS = 80
P2_EPS = 20

CONDITIONS = ["DUAL_STD", "DUAL_INTERVENTIONAL", "DUAL_OFF_BASELINE"]

# Dimensions / env layout
HARM_OBS_DIM = 51
HARM_OBS_A_DIM = 50
HARM_HISTORY_LEN = 10
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16
ACTION_DIM = 4

# Training
LR_ENC = 1e-3
LR_FWD = 5e-4

# Acceptance thresholds (pre-registered)
C1_SEP_THRESHOLD = 0.05
C2_CORR_THRESHOLD = 0.15
C4_CORR_SLACK = 0.10


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        use_proxy_fields=True,
        harm_history_len=HARM_HISTORY_LEN,
        # SD-054 reef enrichment -- breaks monostrategy on 10x10 grid
        reef_enabled=True,
        n_reef_patches=3,
        reef_patch_radius=2,
        hazard_food_attraction=0.7,
    )


def _make_harm_encoder(device) -> HarmEncoder:
    return HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)


def _make_aff_encoder(device) -> AffectiveHarmEncoder:
    return AffectiveHarmEncoder(
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
    ).to(device)


def _make_e2(use_interventional: bool, device) -> E2HarmSForward:
    cfg = E2HarmSConfig(
        use_e2_harm_s_forward=True,
        z_harm_dim=Z_HARM_DIM,
        action_dim=ACTION_DIM,
        use_interventional=use_interventional,
        interventional_fraction=0.3,
        interventional_margin=0.1,
    )
    return E2HarmSForward(cfg).to(device)


def _one_hot(idx: int, dim: int, device) -> torch.Tensor:
    a = torch.zeros(1, dim, device=device)
    a[0, idx] = 1.0
    return a


def _obs_harm(obs_dict) -> Optional[torch.Tensor]:
    ho = obs_dict.get("harm_obs")
    return ho if ho is not None else None


def _obs_harm_a(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict) -> Optional[torch.Tensor]:
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    val = obs_dict.get("accumulated_harm")
    return float(val) if val is not None else 0.0


def _phase_log(phase: str, seed: int, cond: str, ep: int, total: int):
    if ep % 10 == 0 or ep == total - 1:
        print(
            f"[train] seed={seed} cond={cond} ep {ep + 1}/{total} phase={phase}",
            flush=True,
        )


def _run_p0(
    harm_enc: HarmEncoder,
    aff_enc: AffectiveHarmEncoder,
    env: CausalGridWorldV2,
    seed: int,
    cond: str,
    device,
    n_eps: int,
    total_eps_for_log: int,
) -> None:
    """P0: warm up encoders with proximity + accumulated-harm supervision."""
    prox_head = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)
    # SD-018 proxy on the sensory stream (harm_exposure scalar at harm_obs[-1]).
    params_s = list(harm_enc.parameters()) + list(prox_head.parameters())
    opt_s = torch.optim.Adam(params_s, lr=LR_ENC)

    # SD-020-style target: aux head on AffectiveHarmEncoder already regresses
    # accumulated harm in [0, 1] when harm_history_len > 0.
    opt_a = torch.optim.Adam(aff_enc.parameters(), lr=LR_ENC)

    for ep in range(n_eps):
        _phase_log("P0", seed, cond, ep, total_eps_for_log)
        env.reset()
        for step in range(STEPS_PER_EP):
            obs_dict = env._get_observation_dict()
            ho = _obs_harm(obs_dict)
            hoa = _obs_harm_a(obs_dict)
            hh = _obs_harm_history(obs_dict)
            if ho is None or hoa is None:
                break

            ho_b = ho.float().unsqueeze(0).to(device)
            hoa_b = hoa.float().unsqueeze(0).to(device)
            hh_b = hh.float().unsqueeze(0).to(device) if hh is not None else None

            # Sensory: predict harm_exposure scalar (last channel of harm_obs).
            z_harm = harm_enc(ho_b)
            prox_pred = prox_head(z_harm)
            prox_target = ho_b[:, -1:].detach()
            loss_s = F.mse_loss(prox_pred, prox_target)
            opt_s.zero_grad()
            loss_s.backward()
            opt_s.step()

            # Affective: aux head already produces harm_accum_pred.
            _, accum_pred = aff_enc(hoa_b, hh_b)
            if accum_pred is not None:
                accum_tgt = torch.tensor(
                    [[_obs_accum(obs_dict)]], dtype=torch.float32, device=device
                )
                loss_a = F.mse_loss(accum_pred, accum_tgt)
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()

            a_idx = random.randint(0, ACTION_DIM - 1)
            _, _, done, _, _ = env.step(a_idx)
            if done:
                break


def _run_p1(
    harm_enc: HarmEncoder,
    fwd: Optional[E2HarmSForward],
    env: CausalGridWorldV2,
    seed: int,
    cond: str,
    device,
    n_eps: int,
    total_eps_for_log: int,
    offset: int,
) -> None:
    """P1: freeze encoder; train E2_harm_s on detached targets.

    DUAL_OFF_BASELINE arm skips forward-model training entirely.
    """
    if fwd is None:
        for ep in range(n_eps):
            _phase_log("P1", seed, cond, ep + offset, total_eps_for_log)
            env.reset()
            for step in range(STEPS_PER_EP):
                a_idx = random.randint(0, ACTION_DIM - 1)
                _, _, done, _, _ = env.step(a_idx)
                if done:
                    break
        return

    opt = torch.optim.Adam(fwd.parameters(), lr=LR_FWD)
    for ep in range(n_eps):
        _phase_log("P1", seed, cond, ep + offset, total_eps_for_log)
        env.reset()
        z_prev = None
        a_prev_onehot = None
        a_prev_idx = None
        for step in range(STEPS_PER_EP):
            obs_dict = env._get_observation_dict()
            ho = _obs_harm(obs_dict)
            if ho is None:
                z_prev = None
                break
            ho_b = ho.float().unsqueeze(0).to(device)
            with torch.no_grad():
                z_curr = harm_enc(ho_b)

            a_idx = random.randint(0, ACTION_DIM - 1)
            a_onehot = _one_hot(a_idx, ACTION_DIM, device)

            if z_prev is not None and a_prev_onehot is not None:
                z_pred = fwd(z_prev.detach(), a_prev_onehot)
                target = z_curr.detach()
                loss = fwd.compute_loss(z_pred, target)

                if fwd.config.use_interventional and random.random() < fwd.config.interventional_fraction:
                    cf_candidates = [i for i in range(ACTION_DIM) if i != a_prev_idx]
                    cf_idx = random.choice(cf_candidates)
                    a_cf = _one_hot(cf_idx, ACTION_DIM, device)
                    int_loss = fwd.compute_interventional_loss(
                        z_prev.detach(),
                        a_prev_onehot.detach(),
                        a_cf.detach(),
                    )
                    loss = loss + int_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fwd.parameters(), 1.0)
                opt.step()

            z_prev = z_curr
            a_prev_onehot = a_onehot
            a_prev_idx = a_idx

            _, _, done, _, _ = env.step(a_idx)
            if done:
                z_prev = None
                break


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.std() < 1e-8 or ya.std() < 1e-8:
        return 0.0
    c = np.corrcoef(xa, ya)[0, 1]
    if math.isnan(c):
        return 0.0
    return float(c)


def _run_p2_eval(
    harm_enc: HarmEncoder,
    fwd: Optional[E2HarmSForward],
    env: CausalGridWorldV2,
    seed: int,
    cond: str,
    device,
    n_eps: int,
    total_eps_for_log: int,
    offset: int,
    random_proj: Optional[nn.Linear],
) -> Dict[str, float]:
    """P2: evaluation only. Collect (a), (b), (c).

    For DUAL_OFF_BASELINE (fwd is None):
      - comparator: an untrained random Linear(z_harm_dim+action_dim, z_harm_dim)
        is treated as the "forward" model for counterfactual_separation.
      - evaluator: prediction = z_harm_s (identity). pe = ||z_curr - z_prev||.
    """
    seps: List[float] = []
    pes: List[float] = []
    surprises: List[float] = []
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    # Inline SD-020-style EMA of accumulated_harm (does NOT require the flag).
    harm_acc_ema: float = 0.0
    ema_alpha: float = 0.1

    for ep in range(n_eps):
        _phase_log("P2", seed, cond, ep + offset, total_eps_for_log)
        env.reset()
        z_prev = None
        a_prev_onehot = None
        a_prev_idx = None
        for step in range(STEPS_PER_EP):
            obs_dict = env._get_observation_dict()
            ho = _obs_harm(obs_dict)
            if ho is None:
                z_prev = None
                break
            ho_b = ho.float().unsqueeze(0).to(device)
            with torch.no_grad():
                z_curr = harm_enc(ho_b)

            a_idx = random.randint(0, ACTION_DIM - 1)
            a_onehot = _one_hot(a_idx, ACTION_DIM, device)

            # Harm surprise: |accum - ema| then update EMA.
            accum = _obs_accum(obs_dict)
            harm_surprise = abs(accum - harm_acc_ema)
            harm_acc_ema = (1.0 - ema_alpha) * harm_acc_ema + ema_alpha * accum

            if z_prev is not None and a_prev_onehot is not None:
                with torch.no_grad():
                    if fwd is not None:
                        z_pred = fwd(z_prev, a_prev_onehot)
                        cf_candidates = [i for i in range(ACTION_DIM) if i != a_prev_idx]
                        cf_idx = random.choice(cf_candidates)
                        a_cf = _one_hot(cf_idx, ACTION_DIM, device)
                        z_pred_cf = fwd(z_prev, a_cf)
                    else:
                        # DUAL_OFF_BASELINE: untrained random projection as comparator,
                        # identity prediction for evaluator.
                        assert random_proj is not None
                        inp_actual = torch.cat([z_prev, a_prev_onehot], dim=-1)
                        cf_candidates = [i for i in range(ACTION_DIM) if i != a_prev_idx]
                        cf_idx = random.choice(cf_candidates)
                        a_cf = _one_hot(cf_idx, ACTION_DIM, device)
                        inp_cf = torch.cat([z_prev, a_cf], dim=-1)
                        z_pred_cf = random_proj(inp_cf)
                        z_pred_actual_proj = random_proj(inp_actual)
                        # Evaluator: identity prediction of next z_harm_s.
                        z_pred = z_prev
                        # Override comparator target pairs with random-proj outputs.
                        sep_local = float(
                            (z_pred_actual_proj - z_pred_cf).norm(dim=-1).mean().item()
                        )
                        seps.append(sep_local)

                    if fwd is not None:
                        sep = float((z_pred - z_pred_cf).norm(dim=-1).mean().item())
                        seps.append(sep)

                    pe = float((z_pred - z_curr).norm(dim=-1).mean().item())
                    pes.append(pe)
                    surprises.append(float(harm_surprise))
                    preds.append(z_pred.detach().cpu())
                    targets.append(z_curr.detach().cpu())

            z_prev = z_curr
            a_prev_onehot = a_onehot
            a_prev_idx = a_idx

            _, _, done, _, _ = env.step(a_idx)
            if done:
                z_prev = None
                break

    counterfactual_separation = float(np.mean(seps)) if seps else 0.0
    pe_harm_surprise_corr = _pearson(pes, surprises)

    if preds and targets:
        P = torch.cat(preds, dim=0)
        T = torch.cat(targets, dim=0)
        ss_res = float(((T - P) ** 2).sum().item())
        ss_tot = float(((T - T.mean(dim=0)) ** 2).sum().item())
        forward_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    else:
        forward_r2 = 0.0

    return {
        "counterfactual_separation": counterfactual_separation,
        "pe_harm_surprise_corr": pe_harm_surprise_corr,
        "forward_r2": float(forward_r2),
        "n_steps": len(pes),
    }


def _run_unit(seed: int, cond: str, dry_run: bool) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    p0 = 2 if dry_run else P0_EPS
    p1 = 2 if dry_run else P1_EPS
    p2 = 2 if dry_run else P2_EPS
    total = p0 + p1 + p2

    env = _make_env(seed)
    harm_enc = _make_harm_encoder(device)
    aff_enc = _make_aff_encoder(device)

    if cond == "DUAL_STD":
        fwd = _make_e2(use_interventional=False, device=device)
    elif cond == "DUAL_INTERVENTIONAL":
        fwd = _make_e2(use_interventional=True, device=device)
    elif cond == "DUAL_OFF_BASELINE":
        fwd = None
    else:
        raise ValueError(f"unknown condition: {cond}")

    # DUAL_OFF_BASELINE: untrained random comparator projection.
    random_proj = None
    if cond == "DUAL_OFF_BASELINE":
        random_proj = nn.Linear(Z_HARM_DIM + ACTION_DIM, Z_HARM_DIM).to(device)
        for p in random_proj.parameters():
            p.requires_grad_(False)

    print(f"Seed {seed} Condition {cond}", flush=True)

    # P0
    _run_p0(harm_enc, aff_enc, env, seed, cond, device, p0, total)

    # P1: freeze encoders for forward-model training.
    for p in harm_enc.parameters():
        p.requires_grad_(False)
    for p in aff_enc.parameters():
        p.requires_grad_(False)
    _run_p1(harm_enc, fwd, env, seed, cond, device, p1, total, offset=p0)

    # P2
    metrics = _run_p2_eval(
        harm_enc, fwd, env, seed, cond, device, p2, total,
        offset=p0 + p1, random_proj=random_proj,
    )

    # Per-unit verdict: C1 and C2 both satisfied for this unit.
    c1_unit = metrics["counterfactual_separation"] > C1_SEP_THRESHOLD
    c2_unit = metrics["pe_harm_surprise_corr"] > C2_CORR_THRESHOLD
    # For DUAL_OFF_BASELINE we still print a verdict; overall acceptance
    # is evaluated across units, not per baseline unit.
    verdict = "PASS" if (c1_unit and c2_unit) else "FAIL"
    print(
        f"  [seed={seed} cond={cond}] "
        f"cf_sep={metrics['counterfactual_separation']:.4f} "
        f"pe_corr={metrics['pe_harm_surprise_corr']:.4f} "
        f"fwd_r2={metrics['forward_r2']:.4f}",
        flush=True,
    )
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "condition": cond,
        "metrics": metrics,
        "verdict": verdict,
    }


def _aggregate(all_units: List[Dict]) -> Dict:
    by_cond = {c: [u for u in all_units if u["condition"] == c] for c in CONDITIONS}

    def sep(c):
        return [u["metrics"]["counterfactual_separation"] for u in by_cond[c]]

    def corr(c):
        return [u["metrics"]["pe_harm_surprise_corr"] for u in by_cond[c]]

    # C1: comparator valid in STD and INTERVENTIONAL.
    c1_std_wins = sum(1 for v in sep("DUAL_STD") if v > C1_SEP_THRESHOLD)
    c1_int_wins = sum(1 for v in sep("DUAL_INTERVENTIONAL") if v > C1_SEP_THRESHOLD)
    c1 = (c1_std_wins >= 2) and (c1_int_wins >= 2)

    # C2: evaluator valid in STD and INTERVENTIONAL.
    c2_std_wins = sum(1 for v in corr("DUAL_STD") if v > C2_CORR_THRESHOLD)
    c2_int_wins = sum(1 for v in corr("DUAL_INTERVENTIONAL") if v > C2_CORR_THRESHOLD)
    c2 = (c2_std_wins >= 2) and (c2_int_wins >= 2)

    # C3: INTERVENTIONAL comparator > STD comparator in >=2/3 seeds.
    c3_wins = 0
    for u_int, u_std in zip(by_cond["DUAL_INTERVENTIONAL"], by_cond["DUAL_STD"]):
        if u_int["metrics"]["counterfactual_separation"] > u_std["metrics"]["counterfactual_separation"]:
            c3_wins += 1
    c3 = c3_wins >= 2

    # C4: INTERVENTIONAL evaluator >= STD evaluator - slack in >=2/3 seeds.
    c4_wins = 0
    for u_int, u_std in zip(by_cond["DUAL_INTERVENTIONAL"], by_cond["DUAL_STD"]):
        if (
            u_int["metrics"]["pe_harm_surprise_corr"]
            >= u_std["metrics"]["pe_harm_surprise_corr"] - C4_CORR_SLACK
        ):
            c4_wins += 1
    c4 = c4_wins >= 2

    overall_pass = c1 and c2

    # Baseline diagnostic: show DUAL_OFF values for contrast.
    baseline_sep_mean = float(np.mean(sep("DUAL_OFF_BASELINE"))) if by_cond["DUAL_OFF_BASELINE"] else 0.0
    baseline_corr_mean = float(np.mean(corr("DUAL_OFF_BASELINE"))) if by_cond["DUAL_OFF_BASELINE"] else 0.0

    summary = {
        "c1_comparator_valid": {
            "std_wins": c1_std_wins,
            "interventional_wins": c1_int_wins,
            "threshold": C1_SEP_THRESHOLD,
            "pass": c1,
            "desc": "counterfactual_separation > 0.05 in >=2/3 seeds for STD AND INTERVENTIONAL",
        },
        "c2_evaluator_valid": {
            "std_wins": c2_std_wins,
            "interventional_wins": c2_int_wins,
            "threshold": C2_CORR_THRESHOLD,
            "pass": c2,
            "desc": "pe_harm_surprise_corr > 0.15 in >=2/3 seeds for STD AND INTERVENTIONAL",
        },
        "c3_interventional_improves_comparator": {
            "wins": c3_wins,
            "pass": c3,
            "desc": "interventional cf_sep > std cf_sep in >=2/3 seeds (diagnostic refinement)",
        },
        "c4_interventional_does_not_harm_evaluator": {
            "wins": c4_wins,
            "slack": C4_CORR_SLACK,
            "pass": c4,
            "desc": "interventional pe_corr >= std pe_corr - 0.1 in >=2/3 seeds (diagnostic refinement)",
        },
        "baseline_contrast": {
            "off_baseline_cf_sep_mean": baseline_sep_mean,
            "off_baseline_pe_corr_mean": baseline_corr_mean,
            "desc": "DUAL_OFF_BASELINE reference: untrained projection + identity predictor",
        },
    }
    return {"pass": overall_pass, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke: seed=42, cond=DUAL_STD only, tiny P0=2/P1=2/P2=2", flush=True)
        unit = _run_unit(seed=42, cond="DUAL_STD", dry_run=True)
        print(f"Smoke unit: {unit}")
        print("Smoke test PASSED")
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parents[1]
        out_dir = (
            script_dir.parent / "REE_assembly" / "evidence"
            / "experiments" / EXPERIMENT_TYPE
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(SEEDS) * len(CONDITIONS)
    print(
        f"EXQ-452 start: run_id={run_id} total_runs={total_runs} "
        f"episodes_per_run={P0_EPS + P1_EPS + P2_EPS}",
        flush=True,
    )

    all_units: List[Dict] = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            unit = _run_unit(seed=seed, cond=cond, dry_run=False)
            all_units.append(unit)

    agg = _aggregate(all_units)
    outcome = "PASS" if agg["pass"] else "FAIL"
    print(f"\nOutcome: {outcome}", flush=True)
    for k, v in agg["summary"].items():
        print(f"  {k}: {v}")

    # evidence_direction_per_claim: mandatory for multi-claim manifest.
    if outcome == "PASS":
        per_claim = {
            "MECH-257": "supports",
            "SD-013": "supports",
            "ARC-033": "supports",
        }
    else:
        per_claim = {
            "MECH-257": "weakens",
            "SD-013": "mixed",
            "ARC-033": "mixed",
        }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": "v3_exq_452_mech257_dual_function_e2",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        # experiment_purpose=diagnostic is excluded from governance confidence
        # scoring; evidence_direction remains "diagnostic" at the overall level,
        # with per-claim directions recorded for bookkeeping.
        "evidence_direction": "diagnostic",
        "evidence_direction_per_claim": per_claim,
        "pass_criteria_summary": agg["summary"],
        "per_unit_results": all_units,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "p0_eps": P0_EPS,
            "p1_eps": P1_EPS,
            "p2_eps": P2_EPS,
            "steps_per_ep": STEPS_PER_EP,
            "harm_obs_dim": HARM_OBS_DIM,
            "harm_obs_a_dim": HARM_OBS_A_DIM,
            "harm_history_len": HARM_HISTORY_LEN,
            "z_harm_dim": Z_HARM_DIM,
            "z_harm_a_dim": Z_HARM_A_DIM,
            "action_dim": ACTION_DIM,
            "env": {
                "size": 10,
                "num_hazards": 3,
                "num_resources": 3,
                "hazard_harm": 0.04,
                "proximity_harm_scale": 0.12,
            },
            "thresholds": {
                "c1_sep": C1_SEP_THRESHOLD,
                "c2_corr": C2_CORR_THRESHOLD,
                "c4_slack": C4_CORR_SLACK,
            },
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
