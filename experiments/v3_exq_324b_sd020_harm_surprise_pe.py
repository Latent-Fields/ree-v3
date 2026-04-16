#!/opt/local/bin/python3
"""
V3-EXQ-324b: SD-020 Affective Harm Surprise PE -- Multi-Episode Eval Fix

experiment_purpose: evidence

CHANGE vs EXQ-324a (one bug fixed):
  eval_surprise_tracking() contained `if done: break` which terminated
  the 400-step evaluation loop when the episode ended (typically after 6-10
  steps once all resources are collected). With only 6-10 eval steps the
  Pearson correlation between z_harm_a_norm and harm_pe was noise-dominated.
  EXQ-324a results: n_eval_steps=6-10, most seeds produced near-zero or
  sign-flipping correlation.

  Fix: replace `if done: break` with `if done: _, obs_dict = env.reset()`
  This allows the eval loop to continue for the full EVAL_STEPS=400 across
  multiple episodes, providing sufficient data for a reliable Pearson r.

Everything else identical to EXQ-324a.

SCIENTIFIC QUESTION (unchanged):
  Does training enc_a on PE = |mean_damage - ema_mean_damage| (unexpected
  damage jump) produce z_harm_a representations that track harm surprise
  better than training on raw mean_damage (EMA_TARGET condition)?

  SD-020: anterior insula (AIC) encodes unsigned intensity PE as aversive
  surprise, not raw magnitude.

Pass criterion (unchanged):
  C1: surprise_corr_PE > surprise_corr_EMA
  C2: surprise_corr_PE >= 0.15
  C3: z_harm_a_norm_PE > 0.01
  PASS: >= 3/5 seeds satisfy C1, C2, and C3.

Claims: SD-020
Supersedes: V3-EXQ-324a
"""

import json
import sys
import random
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE    = "v3_exq_324b_sd020_harm_surprise_pe"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS          = ["SD-020"]
SUPERSEDES_ID      = "V3-EXQ-324a"

# Pre-registered thresholds
C2_threshold   = 0.15
C3_min_norm    = 0.01
PASS_MIN_SEEDS = 3

# SD-022 substrate dims
HARM_OBS_DIM    = 51
HARM_OBS_A_DIM  = 7
MEAN_DAMAGE_IDX = 5
BODY_OBS_DIM    = 17
Z_HARM_DIM      = 32
Z_HARM_A_DIM    = 16

SEEDS               = [42, 43, 44, 45, 46]
TRAIN_EPISODES      = 200
PRE_DAMAGE_TRANSITS = 20
EVAL_STEPS          = 400
STEPS_PER_EPISODE   = 200
LR                  = 1e-3
HARM_EMA_ALPHA      = 0.1

TOTAL_EPS_PER_SEED  = TRAIN_EPISODES * 2   # 400


def make_env(seed: int) -> CausalGridWorldV2:
    """SD-022 substrate: limb_damage_enabled=True, 8 hazards, slow healing."""
    return CausalGridWorldV2(
        size=10,
        num_hazards=8,
        num_resources=3,
        hazard_harm=0.1,
        resource_benefit=0.05,
        use_proxy_fields=True,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.001,
        seed=seed,
    )


def make_config() -> REEConfig:
    """REEConfig matching SD-022 substrate dims (used for dim validation only)."""
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=250,
        action_dim=4,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        limb_damage_enabled=True,
    )


def _find_hazard_cell(env: CausalGridWorldV2) -> Optional[Tuple[int, int]]:
    for hx, hy in env.hazards:
        return (int(hx), int(hy))
    return None


def _teleport_agent(env: CausalGridWorldV2, tx: int, ty: int) -> None:
    if env.contamination_grid[env.agent_x, env.agent_y] >= env.contamination_threshold:
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["contaminated"]
    else:
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["empty"]
    env.agent_x = tx
    env.agent_y = ty
    env.grid[tx, ty] = env.ENTITY_TYPES["agent"]


def run_pre_damage(env: CausalGridWorldV2, n_transits: int, dry_run: bool) -> float:
    """Force hazard transits to seed non-zero limb damage before eval."""
    n = 3 if dry_run else n_transits
    hcell = _find_hazard_cell(env)
    if hcell is None:
        return 0.0
    hx, hy = hcell
    for _ in range(n):
        _teleport_agent(env, hx, hy)
        env.step(4)
        env.step(0)
        env.step(1)
        env.step(4)
    return float(np.mean(env.limb_damage))


def run_training(
    enc_s: HarmEncoder,
    enc_a: AffectiveHarmEncoder,
    prox_head: nn.Module,
    dmg_head: nn.Module,
    env: CausalGridWorldV2,
    device,
    n_eps: int,
    pe_enabled: bool,
    seed: int,
    condition: str,
    ep_offset: int,
) -> int:
    """
    Train enc_s (harm proximity supervision) and enc_a (PE or EMA target).

    PE_TARGET:  target = |mean_damage - ema_mean_damage|
    EMA_TARGET: target = mean_damage

    Returns n_harm_events (steps where mean_damage > EMA).
    """
    all_params = (
        list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(prox_head.parameters())
        + list(dmg_head.parameters())
    )
    opt = optim.Adam(all_params, lr=LR)

    dmg_ema       = 0.0
    n_harm_events = 0

    _, obs_dict = env.reset()
    for ep in range(n_eps):
        _, obs_dict = env.reset()
        for _ in range(STEPS_PER_EPISODE):
            harm_obs   = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs   = harm_obs.to(device)
            harm_obs_a = harm_obs_a.to(device) if harm_obs_a is not None else None

            z_harm_s    = enc_s(harm_obs.unsqueeze(0))
            prox_pred   = prox_head(z_harm_s)
            prox_target = harm_obs[-1:].unsqueeze(0)
            loss_s = F.mse_loss(prox_pred, prox_target)

            loss_a = torch.tensor(0.0, device=device)
            if harm_obs_a is not None:
                res      = enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res

                mean_dmg = float(harm_obs_a[MEAN_DAMAGE_IDX].item())

                if pe_enabled:
                    harm_pe    = abs(mean_dmg - dmg_ema)
                    target_val = harm_pe
                    if mean_dmg > dmg_ema + 1e-5:
                        n_harm_events += 1
                else:
                    target_val = mean_dmg

                dmg_ema = (1.0 - HARM_EMA_ALPHA) * dmg_ema + HARM_EMA_ALPHA * mean_dmg

                dmg_target = torch.tensor(
                    [[target_val]], dtype=torch.float32, device=device
                )
                loss_a = F.mse_loss(dmg_head(z_harm_a), dmg_target)

            total_loss = loss_s + loss_a
            opt.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()

            action_idx = random.randint(0, 3)
            _, _, done, _, obs_dict = env.step(action_idx)
            if done:
                break

        if (ep + 1) % 50 == 0:
            cum_ep = ep + ep_offset + 1
            print(
                f"  [train] {condition} seed={seed} ep {cum_ep}/{TOTAL_EPS_PER_SEED}",
                flush=True,
            )

    return n_harm_events


def eval_surprise_tracking(
    enc_s: HarmEncoder,
    enc_a: AffectiveHarmEncoder,
    env: CausalGridWorldV2,
    device,
) -> Dict:
    """
    Measure Pearson r between z_harm_a norm and harm_pe / mean_damage.

    Fix vs EXQ-324a: `if done: break` replaced with `if done: env.reset()`.
    This runs the full EVAL_STEPS across multiple episodes (instead of exiting
    at episode end after ~6-10 steps), providing sufficient data for a
    reliable Pearson correlation estimate.
    """
    z_harm_a_norms: List[float] = []
    harm_pe_vals:   List[float] = []
    mean_dmg_vals:  List[float] = []
    dmg_ema = 0.0

    _, obs_dict = env.reset()
    for _ in range(EVAL_STEPS):
        harm_obs   = obs_dict.get("harm_obs")
        harm_obs_a = obs_dict.get("harm_obs_a")
        if harm_obs is None:
            break
        harm_obs   = harm_obs.to(device)
        harm_obs_a = harm_obs_a.to(device) if harm_obs_a is not None else None

        mean_dmg = 0.0
        harm_pe  = 0.0
        if harm_obs_a is not None:
            mean_dmg = float(harm_obs_a[MEAN_DAMAGE_IDX].item())
            harm_pe  = abs(mean_dmg - dmg_ema)
            dmg_ema  = (1.0 - HARM_EMA_ALPHA) * dmg_ema + HARM_EMA_ALPHA * mean_dmg

        with torch.no_grad():
            if harm_obs_a is not None:
                res      = enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res
                z_harm_a_norms.append(float(z_harm_a.norm().item()))
            else:
                z_harm_a_norms.append(0.0)

        harm_pe_vals.append(harm_pe)
        mean_dmg_vals.append(mean_dmg)

        action_idx = random.randint(0, 3)
        _, _, done, _, obs_dict = env.step(action_idx)
        # Fix vs EXQ-324a: reset on done instead of breaking
        if done:
            _, obs_dict = env.reset()

    def safe_corr(x: List[float], y: List[float]) -> float:
        if len(x) < 5:
            return 0.0
        xa = np.array(x, dtype=np.float64)
        ya = np.array(y, dtype=np.float64)
        if xa.std() < 1e-8 or ya.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(xa, ya)[0, 1])

    return {
        "surprise_corr":      safe_corr(z_harm_a_norms, harm_pe_vals),
        "magnitude_corr":     safe_corr(z_harm_a_norms, mean_dmg_vals),
        "z_harm_a_mean_norm": float(np.mean(z_harm_a_norms)) if z_harm_a_norms else 0.0,
        "harm_pe_mean":       float(np.mean(harm_pe_vals))   if harm_pe_vals   else 0.0,
        "mean_damage_mean":   float(np.mean(mean_dmg_vals))  if mean_dmg_vals  else 0.0,
        "n_eval_steps":       len(z_harm_a_norms),
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device  = torch.device("cpu")
    n_train = 5 if dry_run else TRAIN_EPISODES

    print(f"Seed {seed}", flush=True)

    condition_results = {}

    for cond_idx, condition in enumerate(["PE_TARGET", "EMA_TARGET"]):
        pe_enabled = (condition == "PE_TARGET")
        ep_offset  = cond_idx * TRAIN_EPISODES

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env       = make_env(seed)
        enc_s     = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)
        enc_a     = AffectiveHarmEncoder(
            harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM
        ).to(device)
        prox_head = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)
        dmg_head  = nn.Sequential(nn.Linear(Z_HARM_A_DIM, 1)).to(device)

        n_harm_events = run_training(
            enc_s, enc_a, prox_head, dmg_head, env, device,
            n_train, pe_enabled, seed, condition, ep_offset,
        )

        dmg_mean = run_pre_damage(env, PRE_DAMAGE_TRANSITS, dry_run)
        print(
            f"  {condition}: pre-damage mean_limb_damage={dmg_mean:.4f} "
            f"n_harm_events(train)={n_harm_events}",
            flush=True,
        )

        metrics = eval_surprise_tracking(enc_s, enc_a, env, device)
        condition_results[condition] = metrics

        print(
            f"  {condition}: surprise_corr={metrics['surprise_corr']:.4f} "
            f"magnitude_corr={metrics['magnitude_corr']:.4f} "
            f"z_harm_a_norm={metrics['z_harm_a_mean_norm']:.4f} "
            f"n_eval_steps={metrics['n_eval_steps']}",
            flush=True,
        )

    pe  = condition_results["PE_TARGET"]
    ema = condition_results["EMA_TARGET"]

    c1_pass   = pe["surprise_corr"] > ema["surprise_corr"]
    c2_pass   = pe["surprise_corr"] >= C2_threshold
    c3_pass   = pe["z_harm_a_mean_norm"] > C3_min_norm
    seed_pass = c1_pass and c2_pass and c3_pass

    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    return {
        "seed":                    seed,
        "seed_pass":               seed_pass,
        "surprise_corr_pe":        pe["surprise_corr"],
        "surprise_corr_ema":       ema["surprise_corr"],
        "magnitude_corr_pe":       pe["magnitude_corr"],
        "magnitude_corr_ema":      ema["magnitude_corr"],
        "z_harm_a_norm_pe":        pe["z_harm_a_mean_norm"],
        "z_harm_a_norm_ema":       ema["z_harm_a_mean_norm"],
        "harm_pe_mean_pe":         pe["harm_pe_mean"],
        "mean_damage_mean":        pe["mean_damage_mean"],
        "n_eval_steps_pe":         pe["n_eval_steps"],
        "n_eval_steps_ema":        ema["n_eval_steps"],
        "c1_surprise_corr_higher": c1_pass,
        "c2_abs_threshold":        c2_pass,
        "c3_no_collapse":          c3_pass,
        "condition_results":       condition_results,
    }


def main():
    parser = argparse.ArgumentParser(description="EXQ-324b: SD-020 harm surprise PE (multi-ep eval fix)")
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test (1 seed)")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_324b_sd020_harm_surprise_pe_dry"
        if args.dry_run
        else f"v3_exq_324b_sd020_harm_surprise_pe_{timestamp}_v3"
    )
    print(f"EXQ-324b start: {run_id}", flush=True)

    seeds_to_run = [SEEDS[0]] if args.dry_run else SEEDS
    per_seed = [run_seed(s, dry_run=args.dry_run) for s in seeds_to_run]

    seeds_passing       = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes   = seeds_passing >= PASS_MIN_SEEDS
    outcome             = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-324b {outcome} ===", flush=True)
    print(f"Seeds pass: {seeds_passing}/{len(per_seed)}", flush=True)
    for r in per_seed:
        flag = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {flag} "
            f"surp_pe={r['surprise_corr_pe']:.4f} "
            f"surp_ema={r['surprise_corr_ema']:.4f} "
            f"n_eval={r['n_eval_steps_pe']} "
            f"c1={r['c1_surprise_corr_higher']} "
            f"c2={r['c2_abs_threshold']} "
            f"c3={r['c3_no_collapse']}",
            flush=True,
        )

    evidence_direction = "supports" if experiment_passes else "does_not_support"

    output = {
        "run_id":             run_id,
        "experiment_type":    EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "outcome":            outcome,
        "timestamp_utc":      datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "supersedes":         SUPERSEDES_ID,
        "experiment_version": "b",
        "fix_description": (
            "Multi-episode eval fix. EXQ-324a used `if done: break` which exited "
            "eval after 6-10 steps (episode end when resources collected). "
            "Fix: `if done: env.reset()` continues for full EVAL_STEPS=400 "
            "across multiple episodes. This provides sufficient n for a reliable "
            "Pearson correlation between z_harm_a_norm and harm_pe."
        ),
        "substrate_notes":    "SD-022 active: limb_damage_enabled=True, 7-dim harm_obs_a, 8 hazards",
        "registered_thresholds": {
            "C1": "surprise_corr_PE > surprise_corr_EMA",
            "C2": f"surprise_corr_PE >= {C2_threshold}",
            "C3": f"z_harm_a_norm_PE > {C3_min_norm}",
            "pass_min_seeds": PASS_MIN_SEEDS,
        },
        "seeds_passing":  seeds_passing,
        "total_seeds":    len(per_seed),
        "per_seed_results": per_seed,
        "aggregate": {
            "mean_surprise_corr_pe":  float(np.mean([r["surprise_corr_pe"]  for r in per_seed])),
            "mean_surprise_corr_ema": float(np.mean([r["surprise_corr_ema"] for r in per_seed])),
            "mean_magnitude_corr_pe": float(np.mean([r["magnitude_corr_pe"] for r in per_seed])),
            "mean_z_harm_a_norm_pe":  float(np.mean([r["z_harm_a_norm_pe"]  for r in per_seed])),
            "mean_n_eval_steps_pe":   float(np.mean([r["n_eval_steps_pe"]   for r in per_seed])),
        },
    }

    out_dir  = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
