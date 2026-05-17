#!/opt/local/bin/python3
"""
V3-EXQ-592: GAP-11 pilot -- committed-mode curriculum harness validation.

Diagnostic pilot proving that committed_mode_curriculum.py (GAP-11 harness
helper) elicits genuine emergent committed mode, unblocking the OCD test
battery (V3-EXQ-460b/463b/464b/466b/467b/468b).

Three arms per seed:
  EMERGENT -- run_p0_warmup on easy env, then run_p2_eval on target env.
              Emergent commitment: agent drives running_variance below gate
              through training, no scripted rv injection.
  FORCED_RV -- clone trained agent, force rv=0.001, run_p2_eval.
               Control arm (O-2 mandatory contrast): committed state without
               emergent training. Validates the measurement works regardless
               of HOW commitment was entered.
  STARVED -- fresh agent with a near-unachievable commit gate
             (commitment_threshold=1e-4, below equilibrium E2 MSE).
             run_p0_warmup expected to abort at mid-probe (ep ~12/20).
             Validates the R1 escalation gate works.

Acceptance criteria (pre-registered):
  C1: EMERGENT arm: mean_committed_steps_per_ep > 100, median across seeds >= 2/3.
      (Emergent commitment is elicited and sustained; curriculum works.)
  C2: FORCED_RV arm: mean_committed_steps_per_ep > 100 across all seeds.
      (Measurement is valid; committed state produces computable metric.)
  C3: STARVED arm: run_p0_warmup aborts with abort_reason == 'commitment_not_elicited'
      across all seeds. (Abort gate detects rv still above near-unachievable threshold
      at mid-probe and escalates.)

PASS = C1 AND C2 AND C3.

experiment_purpose: diagnostic -- harness-helper validation, not governance
  evidence. Unblocks behavioural governance arms on MECH-090 / SD-034.

Interpretation grid:
  Outcome              | Diagnosis
  ---------------------|------------------------------------------------------
  C1 FAIL              | R1 risk: commit gate mis-calibrated vs achievable
  (emergent < 100)     | world-model error on this env. Check P0 probe log
                       | for rv convergence. If rv converged but committed
                       | steps still 0: check SD-034 closure-operator wiring
                       | (stable_ticks, rule_state norm).
  C2 FAIL              | Measurement wiring broken. Force-set rv=0.001 should
  (forced < 100)       | always produce committed mode. Check clone_trained_agent
                       | + run_p2_eval path in committed_mode_curriculum.py.
  C3 FAIL              | Abort gate not firing. Check mid_probe_frac and probe
  (starved not aborted)| logic in run_p0_warmup (threshold_relaxation=0.0,
                       | commitment_threshold=STARVED_THRESHOLD=1e-4).
                       | If rv drops below 1e-4 by mid-probe, the E2 model is
                       | achieving near-perfect world prediction -- unusual.
  All PASS             | Curriculum harness validated. Queue governance arms:
                       | V3-EXQ-460b (SD-034 closure), V3-EXQ-463b (MECH-268),
                       | V3-EXQ-464b (MECH-266), V3-EXQ-466b/467b/468b (SD-021).

Supersedes: V3-EXQ-461 (diagnostic substrate-readiness synthetic test, 2026-05-12,
  PASS -- that run used scripted delay windows, not emergent training).
"""

import argparse
import copy
import datetime
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from committed_mode_curriculum import (  # noqa: E402
    run_p0_warmup,
    run_p2_eval,
    clone_trained_agent,
    P0Result,
    CommittedModeMetrics,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_592_gap11_pilot_committed_mode_curriculum"
QUEUE_ID = "V3-EXQ-592"
CLAIM_IDS: List[str] = ["MECH-090"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ---------------------------------------------------------------------------
# Constants (pre-registered)
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]
PASS_SEEDS_REQUIRED = 2          # C1 requires >= 2/3 seeds

# C1: emergent commitment floor
C1_COMMITTED_STEPS_FLOOR = 100  # mean_committed_steps_per_ep in P2 eval

# C2: forced-rv sanity floor (same metric, easier to meet)
C2_COMMITTED_STEPS_FLOOR = 100

# P0 training config
P0_BUDGET = 400                  # max episodes for warmup
P0_STEPS_PER_EPISODE = 200
P0_PROBE_INTERVAL = 40           # probe every 40 eps (10% of budget)
P0_MID_PROBE_FRAC = 0.60         # abort at 60% of budget if not converging

# Easy env config (P0 warmup)
EASY_ENV_SIZE = 10
EASY_ENV_HAZARDS = 2
EASY_ENV_RESOURCES = 2
EASY_TOLERANCE_FRAC = 0.30       # generous tolerance band for P0

# Target env config (P2 eval)
TARGET_ENV_SIZE = 10
TARGET_ENV_HAZARDS = 4
TARGET_ENV_RESOURCES = 3
TARGET_TOLERANCE_FRAC = 0.15     # tighter band for eval

# P2 eval config
P2_EPISODES = 50
P2_STEPS_PER_EPISODE = 200

# STARVED arm config (abort gate validation).
# STARVED_THRESHOLD must be below the equilibrium E2 world-forward MSE so that
# rv >= threshold stays True at mid-probe (causing abort).  z_world MSE typically
# settles at ~0.005-0.01 in limited training; 1e-4 is tight enough to never
# be achieved by the mid-probe checkpoint (ep 12 of 20).
STARVED_THRESHOLD = 1e-4         # near-unachievable gate (rv stays above this)
STARVED_BUDGET = 20              # short run; abort fires at ep ~12 (60% of budget)
STARVED_PROBE_INTERVAL = 4


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def make_easy_env(seed: int) -> CausalGridWorldV2:
    """Easy env for P0 warmup -- low hazards, generous tolerance."""
    return CausalGridWorldV2(
        size=EASY_ENV_SIZE,
        num_hazards=EASY_ENV_HAZARDS,
        num_resources=EASY_ENV_RESOURCES,
        hazard_harm=0.02,
        resource_benefit=0.05,
        use_proxy_fields=True,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=EASY_TOLERANCE_FRAC,
        seed=seed,
    )


def make_target_env(seed: int) -> CausalGridWorldV2:
    """Target env for P2 eval -- standard difficulty, tighter tolerance."""
    return CausalGridWorldV2(
        size=TARGET_ENV_SIZE,
        num_hazards=TARGET_ENV_HAZARDS,
        num_resources=TARGET_ENV_RESOURCES,
        hazard_harm=0.02,
        resource_benefit=0.05,
        use_proxy_fields=True,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=TARGET_TOLERANCE_FRAC,
        seed=seed + 1000,
    )


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def make_standard_cfg() -> REEConfig:
    """Standard agent config. alpha_world=0.9 per SD-008."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        alpha_world=0.9,
        use_harm_stream=False,
    )
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def make_starved_cfg() -> REEConfig:
    """Starved agent: commit gate set to unreachable threshold."""
    cfg = make_standard_cfg()
    cfg.e3.commitment_threshold = STARVED_THRESHOLD
    return cfg


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool = False) -> Dict:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    p0_budget = 5 if dry_run else P0_BUDGET
    p2_eps = 3 if dry_run else P2_EPISODES
    p2_steps = 20 if dry_run else P2_STEPS_PER_EPISODE
    p0_steps = 20 if dry_run else P0_STEPS_PER_EPISODE
    starved_budget = 6 if dry_run else STARVED_BUDGET
    starved_probe = 2 if dry_run else STARVED_PROBE_INTERVAL

    # ------------------------------------------------------------------
    # ARM_0: EMERGENT
    # ------------------------------------------------------------------
    print(f"Seed {seed} Condition EMERGENT", flush=True)
    agent_main = REEAgent(make_standard_cfg()).to(device)
    easy_env = make_easy_env(seed)

    p0: P0Result = run_p0_warmup(
        agent_main, easy_env, device,
        budget=p0_budget,
        steps_per_episode=p0_steps,
        probe_interval=P0_PROBE_INTERVAL if not dry_run else 2,
        mid_probe_frac=P0_MID_PROBE_FRAC,
        convergence_stable_checkpoints=3 if not dry_run else 1,
        threshold_relaxation=0.0,
    )

    print(
        f"[seed={seed}] P0 EMERGENT: converged={p0.converged}"
        f" aborted={p0.aborted} final_rv={p0.final_rv:.5f}"
        f" n_eps={p0.n_episodes}",
        flush=True,
    )

    if p0.aborted:
        # R1 risk -- escalate as substrate finding
        print(
            f"[seed={seed}] EMERGENT P0 ABORTED: {p0.abort_reason}. "
            f"R1 escalation -- commit gate may be mis-calibrated vs achievable "
            f"world-model error. C1 = FAIL.",
            flush=True,
        )
        metrics_emergent = CommittedModeMetrics(
            total_committed_steps=0, total_beta_elevated=0,
            hold_rate=0.0, mean_committed_steps_per_ep=0.0,
            rule_state_norm=0.0, n_eval_episodes=0, per_episode=[],
        )
        c1_pass = False
    else:
        target_env_em = make_target_env(seed)
        metrics_emergent = run_p2_eval(
            agent_main, target_env_em, device,
            n_eps=p2_eps, steps_per_episode=p2_steps,
        )
        c1_pass = metrics_emergent.mean_committed_steps_per_ep > C1_COMMITTED_STEPS_FLOOR
        print(
            f"[seed={seed}] EMERGENT P2:"
            f" mean_committed/ep={metrics_emergent.mean_committed_steps_per_ep:.1f}"
            f" hold_rate={metrics_emergent.hold_rate:.3f}"
            f" C1={'PASS' if c1_pass else 'FAIL'}",
            flush=True,
        )

    verdict_em = "PASS" if c1_pass else "FAIL"
    print(f"verdict: {verdict_em}", flush=True)

    # ------------------------------------------------------------------
    # ARM_1: FORCED_RV (O-2 mandatory contrast)
    # ------------------------------------------------------------------
    print(f"Seed {seed} Condition FORCED_RV", flush=True)
    if p0.aborted:
        # Can't clone an unconverged agent meaningfully; use a fresh agent
        # with forced rv instead -- this still validates the measurement path.
        agent_forced = REEAgent(make_standard_cfg()).to(device)
    else:
        agent_forced = clone_trained_agent(agent_main, bistable=True, device=device)
    agent_forced.e3._running_variance = 0.001  # force below threshold

    target_env_fo = make_target_env(seed)
    metrics_forced = run_p2_eval(
        agent_forced, target_env_fo, device,
        n_eps=p2_eps, steps_per_episode=p2_steps,
    )
    c2_pass = metrics_forced.mean_committed_steps_per_ep > C2_COMMITTED_STEPS_FLOOR
    print(
        f"[seed={seed}] FORCED_RV P2:"
        f" mean_committed/ep={metrics_forced.mean_committed_steps_per_ep:.1f}"
        f" hold_rate={metrics_forced.hold_rate:.3f}"
        f" C2={'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    verdict_fo = "PASS" if c2_pass else "FAIL"
    print(f"verdict: {verdict_fo}", flush=True)

    # ------------------------------------------------------------------
    # ARM_2: STARVED -- abort gate validation (C3)
    # ------------------------------------------------------------------
    print(f"Seed {seed} Condition STARVED", flush=True)
    agent_starved = REEAgent(make_starved_cfg()).to(device)
    easy_env_st = make_easy_env(seed + 500)

    p0_starved: P0Result = run_p0_warmup(
        agent_starved, easy_env_st, device,
        budget=starved_budget,
        steps_per_episode=p0_steps,
        probe_interval=starved_probe,
        mid_probe_frac=P0_MID_PROBE_FRAC,
        convergence_stable_checkpoints=999,  # should never converge
        threshold_relaxation=0.0,
    )
    c3_pass = (
        p0_starved.aborted
        and p0_starved.abort_reason == "commitment_not_elicited"
    )
    print(
        f"[seed={seed}] STARVED P0:"
        f" aborted={p0_starved.aborted}"
        f" reason={p0_starved.abort_reason!r}"
        f" final_rv={p0_starved.final_rv:.5f}"
        f" C3={'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    verdict_st = "PASS" if c3_pass else "FAIL"
    print(f"verdict: {verdict_st}", flush=True)

    # ------------------------------------------------------------------
    # Seed summary
    # ------------------------------------------------------------------
    seed_pass = c1_pass and c2_pass and c3_pass
    print(
        f"[seed={seed}] C1={'PASS' if c1_pass else 'FAIL'}"
        f" C2={'PASS' if c2_pass else 'FAIL'}"
        f" C3={'PASS' if c3_pass else 'FAIL'}"
        f" => {'PASS' if seed_pass else 'FAIL'}",
        flush=True,
    )

    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "p0_emergent": {
            "converged": p0.converged,
            "aborted": p0.aborted,
            "abort_reason": p0.abort_reason,
            "n_episodes": p0.n_episodes,
            "final_rv": p0.final_rv,
            "commit_threshold_used": p0.commit_threshold_used,
            "probe_log": p0.probe_log,
        },
        "p2_emergent": {
            "total_committed_steps": metrics_emergent.total_committed_steps,
            "mean_committed_steps_per_ep": metrics_emergent.mean_committed_steps_per_ep,
            "hold_rate": metrics_emergent.hold_rate,
            "rule_state_norm": metrics_emergent.rule_state_norm,
            "n_eval_episodes": metrics_emergent.n_eval_episodes,
        },
        "p2_forced": {
            "total_committed_steps": metrics_forced.total_committed_steps,
            "mean_committed_steps_per_ep": metrics_forced.mean_committed_steps_per_ep,
            "hold_rate": metrics_forced.hold_rate,
            "rule_state_norm": metrics_forced.rule_state_norm,
            "n_eval_episodes": metrics_forced.n_eval_episodes,
        },
        "p0_starved": {
            "aborted": p0_starved.aborted,
            "abort_reason": p0_starved.abort_reason,
            "n_episodes": p0_starved.n_episodes,
            "final_rv": p0_starved.final_rv,
        },
        "c1_emergent_committed_floor": c1_pass,
        "c2_forced_committed_floor": c2_pass,
        "c3_starved_abort_gate": c3_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> tuple:
    """Run experiment. Returns (outcome_str, manifest_path_or_None)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Short run (2 eps P0, 3 eps P2) to test wiring without writing evidence.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    seeds = args.seeds

    print(f"[{QUEUE_ID}] GAP-11 pilot -- committed-mode curriculum harness validation", flush=True)
    print(f"  dry_run={dry_run}  seeds={seeds}", flush=True)
    print(f"  P0_BUDGET={5 if dry_run else P0_BUDGET}  P2_EPISODES={3 if dry_run else P2_EPISODES}", flush=True)
    print(f"  C1_floor={C1_COMMITTED_STEPS_FLOOR}  C2_floor={C2_COMMITTED_STEPS_FLOOR}", flush=True)
    print(f"  STARVED_THRESHOLD={STARVED_THRESHOLD}", flush=True)

    per_seed_results = []
    for seed in seeds:
        result = run_seed(seed, dry_run=dry_run)
        per_seed_results.append(result)

    seeds_passing = sum(1 for r in per_seed_results if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_SEEDS_REQUIRED

    # Aggregate C1/C2/C3 across seeds
    c1_vals = [r["p2_emergent"]["mean_committed_steps_per_ep"] for r in per_seed_results]
    c2_vals = [r["p2_forced"]["mean_committed_steps_per_ep"] for r in per_seed_results]
    c3_all = all(r["c3_starved_abort_gate"] for r in per_seed_results)

    print(f"\n[{QUEUE_ID}] Seeds passing: {seeds_passing}/{len(seeds)}", flush=True)
    print(
        f"[{QUEUE_ID}] C1 emergent mean/ep: {[round(v,1) for v in c1_vals]}"
        f"  median={round(float(np.median(c1_vals)),1)}",
        flush=True,
    )
    print(
        f"[{QUEUE_ID}] C2 forced mean/ep: {[round(v,1) for v in c2_vals]}",
        flush=True,
    )
    print(f"[{QUEUE_ID}] C3 starved abort gate: {c3_all}", flush=True)
    print(f"[{QUEUE_ID}] Experiment: {'PASS' if experiment_passes else 'FAIL'}", flush=True)

    outcome = "PASS" if experiment_passes else "FAIL"

    if dry_run:
        print(f"[{QUEUE_ID}] DRY RUN -- not writing evidence.", flush=True)
        return outcome, None

    # Write evidence
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = EVIDENCE_DIR / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": "supports" if experiment_passes else "does_not_support",
        "evidence_direction_per_claim": {
            "MECH-090": "supports" if experiment_passes else "does_not_support",
        },
        "supersedes": "v3_exq_461_mech090_sd033a_delayed_reward_persistence",
        "thresholds": {
            "C1_committed_steps_floor": C1_COMMITTED_STEPS_FLOOR,
            "C2_committed_steps_floor": C2_COMMITTED_STEPS_FLOOR,
            "C3_abort_gate_required": True,
            "pass_seeds_required": PASS_SEEDS_REQUIRED,
            "p0_budget": P0_BUDGET,
            "p0_mid_probe_frac": P0_MID_PROBE_FRAC,
        },
        "aggregate": {
            "seeds_passing": seeds_passing,
            "c1_emergent_mean_per_ep_by_seed": c1_vals,
            "c1_emergent_median": float(np.median(c1_vals)),
            "c2_forced_mean_per_ep_by_seed": c2_vals,
            "c3_starved_abort_all_seeds": c3_all,
        },
        "per_seed_results": per_seed_results,
        "notes": (
            "GAP-11 pilot: validates committed_mode_curriculum.py harness. "
            "EMERGENT arm uses run_p0_warmup + run_p2_eval (no scripted rv). "
            "FORCED_RV arm (O-2 contrast) uses clone_trained_agent + forced rv=0.001. "
            "STARVED arm validates abort gate (R1 escalation). "
            "On PASS: queue V3-EXQ-460b/463b/464b/466b/467b/468b governance arms."
        ),
    }

    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{QUEUE_ID}] Evidence written -> {out_path}", flush=True)

    return outcome, out_path


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
