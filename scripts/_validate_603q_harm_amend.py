"""Proof-of-fix validation for the 603p harm-pathway stabilization amend.

Replicates the 603p POSITIVE-CONTROL cell (proximity_harm=0.10, harm_lr=1e-3,
BASE arm, Stage-H only) faithfully by importing 603p's own config builders, and
runs it with the new stabilization levers (decoupled encoder LR + LR warmup) ON.
Reports per-seed harm_eval_range and whether >=2/3 seeds clear the 0.02 floor --
the readiness gate the autopsy requires the amend to establish.

Budgets shrinkable via env vars for a fast smoke; defaults = full 603p budgets.
Not a queued experiment -- a manual probe (the 643a/680d proof-of-fix precedent).
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
)
import experiments.v3_exq_603p_base_harm_landscape_discriminativeness_diagnostic as p  # noqa: E402

PROX = 0.10
HARM_LR = 1e-3
FLOOR = 0.02
SEEDS = [42, 43, 44]

# Stabilization levers (the amend under test). None/0 -> OFF arm (legacy).
ENC_LR = float(os.environ.get("HARM_ENC_LR", "3e-4"))
WARMUP = int(os.environ.get("HARM_WARMUP", "200"))
ARM_OFF = os.environ.get("HARM_ARM_OFF", "0") == "1"  # run legacy 1e-3 single-LR

# Budget overrides (default = full 603p positive-control budget).


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_seed(seed: int) -> dict:
    _seed_all(seed)
    cfg = p._make_scaffold_cfg(dry_run=False, proximity_harm=PROX, harm_lr=HARM_LR)
    # Optional budget shrink for a fast smoke.
    if "P0_BUDGET" in os.environ:
        cfg.scaffold_p0_episode_budget = int(os.environ["P0_BUDGET"])
    if "HAZARD_BUDGET" in os.environ:
        cfg.scaffold_hazard_stage_episode_budget = int(os.environ["HAZARD_BUDGET"])
    if "STEPS" in os.environ:
        cfg.scaffold_steps_per_episode = int(os.environ["STEPS"])
    if "STAGE0_BUDGET" in os.environ:
        cfg.scaffold_stage0_episode_budget = int(os.environ["STAGE0_BUDGET"])
    # Apply the stabilization levers (the amend).
    if not ARM_OFF:
        cfg.scaffold_harm_pathway_encoder_lr = ENC_LR
        cfg.scaffold_harm_pathway_warmup_steps = WARMUP
    device = torch.device("cpu")
    env = _build_env(cfg, "hazard")
    env.reset()
    agent = p.REEAgent(p._make_config(env)).to(device)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    s0 = sched.run_stage0_nursery(agent, device)
    sched.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    sched.run_p0(agent, device)
    hz = sched.run_hazard_avoidance(agent, device)
    hd = dict(hz.harm_discriminativeness or {})
    rng = float(hd.get("harm_eval_range", 0.0))
    corr = hd.get("harm_eval_prox_corr", float("nan"))
    try:
        corr = float(corr)
    except (TypeError, ValueError):
        corr = float("nan")
    rec = {
        "seed": seed, "harm_eval_range": rng, "harm_eval_prox_corr": corr,
        "pass": bool(rng >= FLOOR), "mean_len": float(hz.mean_episode_length),
        "stage0_zgoal": float(s0.z_goal_norm_peak),
        "n_train_steps": int((hz.harm_pathway_diag or {}).get("n_train_steps", 0)),
    }
    arm = "OFF(legacy 1e-3)" if ARM_OFF else f"ON(enc_lr={ENC_LR},warmup={WARMUP})"
    print(f"[{arm}] seed={seed} harm_eval_range={rng:.4f} "
          f"({'PASS' if rec['pass'] else 'FAIL'} vs floor {FLOOR}) "
          f"corr={corr:.3f} mean_len={rec['mean_len']:.1f} "
          f"stage0_zgoal={rec['stage0_zgoal']:.3f} hz_steps={rec['n_train_steps']}",
          flush=True)
    return rec


def main() -> None:
    arm = "OFF(legacy single 1e-3 LR)" if ARM_OFF else f"ON(enc_lr={ENC_LR}, warmup={WARMUP})"
    print(f"=== 603q harm-pathway amend validation: proximity_harm={PROX} {arm} ===",
          flush=True)
    recs = [run_seed(s) for s in SEEDS]
    n_pass = sum(1 for r in recs if r["pass"])
    frac = n_pass / len(recs)
    ranges = [round(r["harm_eval_range"], 4) for r in recs]
    print(f"=== RESULT: harm_disc_frac={n_pass}/{len(recs)} ({frac:.3f}) "
          f"ranges={ranges} floor={FLOOR} "
          f"{'CLEARS >=2/3' if frac >= 2/3 else 'BELOW 2/3'} ===", flush=True)


if __name__ == "__main__":
    main()
