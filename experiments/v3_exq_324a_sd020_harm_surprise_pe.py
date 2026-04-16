#!/opt/local/bin/python3
"""
V3-EXQ-324a: SD-020 Affective Harm Surprise PE -- SD-022 Substrate Fix

experiment_purpose: evidence

Corrected version of EXQ-324. Two compounding root causes of EXQ-324 failure:

1. HARM_OBS_A_DIM=50 (EMA proximity): harm_obs_a was structurally identical to harm_obs_s.
   z_harm_a variance was near zero (0.0003-0.0007) -- the PE signal was drowned in noise.
   The AffectiveHarmEncoder had no discriminative input, so PE vs EMA training produced
   nearly identical representations.

2. Too few harm events: 10-20 events per seed (60 eps, 4 hazards). The surprise
   correlation estimate was noise-dominated, sign-flipping across seeds.

Fix: SD-022 substrate (limb_damage_enabled=True, heal_rate=0.001).
harm_obs_a is now 7-dim directional body damage state, causally independent of current
world proximity (confirmed EXQ-241b: r2_s_to_a=0.996 structural ceiling, not calibration).
With 8 hazards and 200 episodes per condition, ~3000+ damage-step signals per run.
PE is computed from mean_damage (harm_obs_a[5]), not the old EMA-proximity scalar.

Two conditions per seed (run sequentially, 200 episodes each):
  PE_TARGET  -- enc_a trained on |mean_damage - ema_mean_damage| (unexpected damage jump)
  EMA_TARGET -- enc_a trained on mean_damage directly (raw accumulated damage level)

Both conditions also train enc_s on harm proximity (pred harm_obs[-1] from z_harm_s)
to give enc_s a meaningful gradient and provide a z_harm_s baseline.
PE EMA (alpha=0.1, ~10-step window) tracks expected mean_damage; hazard contacts produce
non-zero PE when damage jumps above the rolling expectation.

Progress instrumentation:
  episodes_per_run = 400  (200 eps x 2 conditions per seed, cumulative denominator)
  seeds = 5, conditions = 1  -> 5 total runs, 5 verdict lines
  [train] ep N/400 cumulative across both conditions within each seed

Pass criterion (pre-registered):
  C1: surprise_corr_PE > surprise_corr_EMA  (PE training tracks surprise more than EMA)
  C2: surprise_corr_PE >= 0.15              (absolute threshold: reliable PE tracking)
  C3: z_harm_a_norm_PE > 0.01              (encoder not collapsed)

Experiment PASS: >= 3/5 seeds satisfy C1, C2, and C3.

Scientific dependency: SD-020 evidence is most interpretable when SD-019 (nonredundancy)
is confirmed on the same substrate (EXQ-323a PASS). EXQ-324a can run first but governance
weighting should be deferred until EXQ-323a resolves.

Biological grounding: anterior insula (AIC) encodes unsigned intensity PE as an
aversive surprise signal, not raw magnitude (Chen 2023; Hoskin 2023; Geuter 2017;
Horing 2022; Iannetti & Mouraux 2010).

Claims: SD-020 (affective harm surprise PE). SD-011 and Q-036 are prerequisites,
not variables under test -- see claim_ids accuracy rule.
Supersedes: V3-EXQ-324 (EMA substrate, too few harm events -- both confounds invalidated).
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


EXPERIMENT_TYPE    = "v3_exq_324a_sd020_harm_surprise_pe"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS          = ["SD-020"]

# Pre-registered thresholds
C2_threshold = 0.15   # surprise_corr_PE absolute minimum
C3_min_norm  = 0.01   # z_harm_a must have non-zero norm (no collapse)
PASS_MIN_SEEDS = 3

# SD-022 substrate dims
HARM_OBS_DIM    = 51   # hazard_field(25) + resource_field(25) + harm_exposure(1)
HARM_OBS_A_DIM  = 7    # SD-022: damage[4] + max_damage + mean_damage + residual_pain
MEAN_DAMAGE_IDX = 5    # index of mean_damage in harm_obs_a
BODY_OBS_DIM    = 17   # SD-022: proxy(12) + damage[4] + residual_pain(1)
Z_HARM_DIM      = 32
Z_HARM_A_DIM    = 16

SEEDS               = [42, 43, 44, 45, 46]
TRAIN_EPISODES      = 200   # per condition; 400 total per seed
PRE_DAMAGE_TRANSITS = 20    # hazard transits before eval (ensure non-zero damage)
EVAL_STEPS          = 400
STEPS_PER_EPISODE   = 200
LR                  = 1e-3
HARM_EMA_ALPHA      = 0.1   # ~10-step window for expected damage tracking

# Total episodes per seed (both conditions combined) -- equals episodes_per_run in queue
TOTAL_EPS_PER_SEED = TRAIN_EPISODES * 2   # 400


def make_env(seed: int) -> CausalGridWorldV2:
    """SD-022 substrate: limb_damage_enabled=True, 8 hazards, slow healing."""
    return CausalGridWorldV2(
        size=10,
        num_hazards=8,           # 8 vs EXQ-324's 4 -- more damage-generating contacts
        num_resources=3,
        hazard_harm=0.1,
        resource_benefit=0.05,
        use_proxy_fields=True,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.001,         # slow healing -- damage persists across steps
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

    PE_TARGET:  target = |mean_damage - ema_mean_damage| (surprise: unexpected damage jump)
    EMA_TARGET: target = mean_damage directly             (raw accumulated level)

    Progress printed as cumulative ep (ep + ep_offset + 1) / TOTAL_EPS_PER_SEED.
    Returns n_harm_events: steps where mean_damage exceeded EMA (diagnostic).
    """
    all_params = (
        list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(prox_head.parameters())
        + list(dmg_head.parameters())
    )
    opt = optim.Adam(all_params, lr=LR)

    dmg_ema      = 0.0
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

            # enc_s: harm proximity supervision
            z_harm_s    = enc_s(harm_obs.unsqueeze(0))
            prox_pred   = prox_head(z_harm_s)
            prox_target = harm_obs[-1:].unsqueeze(0)
            loss_s = F.mse_loss(prox_pred, prox_target)

            # enc_a: PE or EMA target from mean_damage (SD-022 signal)
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

                # EMA updated every step for both conditions (consistent eval baseline)
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

        # Cumulative episode progress (both conditions share a 400-ep denominator)
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
    Measure Pearson r between z_harm_a norm and:
      (a) PE = |mean_damage - ema_mean_damage|  (surprise signal)
      (b) mean_damage                            (raw accumulated load)
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
        if done:
            break

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
    device = torch.device("cpu")
    n_train = 5 if dry_run else TRAIN_EPISODES

    # Boundary line for runner: resets episode counter for this seed
    print(f"Seed {seed}", flush=True)

    condition_results = {}

    for cond_idx, condition in enumerate(["PE_TARGET", "EMA_TARGET"]):
        pe_enabled = (condition == "PE_TARGET")
        ep_offset  = cond_idx * TRAIN_EPISODES   # 0 for PE, 200 for EMA

        # Re-seed both conditions identically for clean within-seed comparison
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
            f"z_harm_a_norm={metrics['z_harm_a_mean_norm']:.4f}",
            flush=True,
        )

    pe  = condition_results["PE_TARGET"]
    ema = condition_results["EMA_TARGET"]

    c1_pass = pe["surprise_corr"] > ema["surprise_corr"]
    c2_pass = pe["surprise_corr"] >= C2_threshold
    c3_pass = pe["z_harm_a_mean_norm"] > C3_min_norm
    seed_pass = c1_pass and c2_pass and c3_pass

    # Single verdict per seed (runner seeds=5, conditions=1 -> 5 total verdict lines)
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
        "c1_surprise_corr_higher": c1_pass,
        "c2_abs_threshold":        c2_pass,
        "c3_no_collapse":          c3_pass,
        "condition_results":       condition_results,
    }


def main():
    parser = argparse.ArgumentParser(description="EXQ-324a: SD-020 harm surprise PE")
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test (1 seed)")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_324a_sd020_harm_surprise_pe_dry"
        if args.dry_run
        else f"v3_exq_324a_sd020_harm_surprise_pe_{timestamp}_v3"
    )
    print(f"EXQ-324a start: {run_id}", flush=True)

    seeds_to_run = [SEEDS[0]] if args.dry_run else SEEDS
    per_seed = [run_seed(s, dry_run=args.dry_run) for s in seeds_to_run]

    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-324a {outcome} ===", flush=True)
    print(f"Seeds pass: {seeds_passing}/{len(per_seed)}", flush=True)
    for r in per_seed:
        flag = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {flag} "
            f"surp_pe={r['surprise_corr_pe']:.4f} "
            f"surp_ema={r['surprise_corr_ema']:.4f} "
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
        "supersedes":         "v3_exq_324_sd020_harm_surprise_pe",
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
        },
    }

    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
