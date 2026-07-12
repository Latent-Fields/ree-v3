#!/opt/local/bin/python3
"""
V3-EXQ-592b: MECH-090 R-c commit-entry readiness conjunction substrate validation.

Falsifier-grade 2-arm diagnostic for the commit-entry readiness gate landed
2026-05-28 (commitment_closure_plan.md GAP-4; see design doc
REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md).

V3-EXQ-592 seed 42 showed rv-only commit-entry is satisfiable by degenerate
trivial-predictability (running_variance=2.7e-5 with nav_competence=0.0).
The R-c conjunction substrate added a readiness predicate at BetaGate entry:
beta_gate.elevate() admitted iff E3SelectionResult.committed (rv-low) AND
per-candidate first-action score margin >= commit_readiness_floor.

This experiment validates that:
  ARM_0 GATED: in the V3-EXQ-592 EMERGENT-arm curriculum with the gate ON,
    seed 42's degenerate basin triggers gate-blocks (mech090_n_elevation_blocked
    >= 1) AND no committed steps accumulate (total_committed_steps == 0) AND
    rv crosses commitment_threshold (confirming the gate is the load-bearing
    block, not an upstream rv failure).
  ARM_1 GATED_FORCED_READY: in the V3-EXQ-592 FORCED_RV-arm scenario (clone
    a trained agent + force rv=0.001) with the gate ON, the trained agent's
    differentiated candidate scores produce margins above floor, gate admits
    (mech090_n_elevation_admitted >= 1), and committed_steps > 0 (confirming
    the gate does not permanently lock out commitment when readiness clears).

PASS = (ARM_0 falsifier signature on >= 1 seed in 3) AND
       (ARM_1 admit signature on >= 2 seeds in 3).

experiment_purpose: diagnostic -- substrate-readiness validation of the
  MECH-090 R-c commit-entry conjunction, not standalone governance evidence.
  Joint PASS unblocks commitment_closure:GAP-4 (partial -> done substrate-
  side) and queues the Phase 4/5 *b cohort. Joint FAIL routes to the
  interpretation grid below.

Interpretation grid:
  Outcome                                    | Diagnosis
  -------------------------------------------|---------------------------------
  ARM_0 PASS + ARM_1 PASS (joint PASS)      | R-c substrate validated. GAP-4
                                            | partial -> done substrate-side.
                                            | Queue *b cohort behavioural arms.
  ARM_0 PASS + ARM_1 FAIL                   | Gate too restrictive -- blocks
                                            | even clean-margin cases. Calibrate
                                            | commit_readiness_floor downward
                                            | (sweep {0.01, 0.02, 0.05, 0.10}).
  ARM_0 FAIL + ARM_1 PASS                   | Unexpected. Either seed 42 no
                                            | longer reaches degenerate basin
                                            | (curriculum-drift) or gate
                                            | doesn't see the degenerate
                                            | scores. /diagnose-errors required.
  ARM_0 FAIL + ARM_1 FAIL                   | Substrate retest. Check
                                            | use_commit_readiness_gate flag
                                            | propagation through agent.__init__
                                            | and verify result.scores -> margin
                                            | computation at the elevate sites.

Supersedes: none. V3-EXQ-592 remains the curriculum-harness validation; this
  is the conjunction-substrate validation that closes the gap V3-EXQ-592
  surfaced.

Predecessor synthesis: REE_assembly/evidence/literature/targeted_review_
  connectome_mech_090/synthesis.md commit 9e68c5ca8a.
Predecessor session: implement-substrate-mech090-commit-predicate-20260528T172800Z.
"""

import argparse
import datetime
import json
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_592b_mech090_commit_readiness_gate_validation"
QUEUE_ID = "V3-EXQ-592b"
CLAIM_IDS: List[str] = ["MECH-090"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ---------------------------------------------------------------------------
# Constants (pre-registered)
# ---------------------------------------------------------------------------

# Seed 42 is the V3-EXQ-592 degenerate-basin signature seed; 43 and 44 are the
# matched seeds from V3-EXQ-592 SEEDS so this experiment is directly
# comparable on identical RNG roots.
SEEDS = [42, 43, 44]

# MECH-090 R-c gate config
COMMIT_READINESS_FLOOR = 0.05
COMMIT_READINESS_GATE_ENABLED = True

# Acceptance criteria
# ARM_0 falsifier: at least one seed must trigger the gate-block path.
ARM_0_MIN_SEEDS_WITH_BLOCKS = 1
# ARM_1 admit: at least two seeds must show successful elevation admission.
ARM_1_MIN_SEEDS_WITH_ADMITS = 2

# Curriculum config (matched to V3-EXQ-592)
P0_BUDGET = 400
P0_STEPS_PER_EPISODE = 200
P0_PROBE_INTERVAL = 40
P0_MID_PROBE_FRAC = 0.60

EASY_ENV_SIZE = 10
EASY_ENV_HAZARDS = 2
EASY_ENV_RESOURCES = 2
EASY_TOLERANCE_FRAC = 0.30

TARGET_ENV_SIZE = 10
TARGET_ENV_HAZARDS = 4
TARGET_ENV_RESOURCES = 3
TARGET_TOLERANCE_FRAC = 0.15

P2_EPISODES = 50
P2_STEPS_PER_EPISODE = 200


# ---------------------------------------------------------------------------
# Env factories (matched to V3-EXQ-592)
# ---------------------------------------------------------------------------

def make_easy_env(seed: int) -> CausalGridWorldV2:
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
# Agent factories: MECH-090 R-c gate ON, otherwise matched to V3-EXQ-592.
# ---------------------------------------------------------------------------

def make_gated_cfg() -> REEConfig:
    """Standard agent config + MECH-090 R-c commit-readiness gate ON."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        alpha_world=0.9,
        use_harm_stream=False,
    )
    cfg.heartbeat.beta_gate_bistable = True
    # MECH-090 R-c (2026-05-28): commit-entry conjunction.
    cfg.heartbeat.use_commit_readiness_gate = COMMIT_READINESS_GATE_ENABLED
    cfg.heartbeat.commit_readiness_floor = COMMIT_READINESS_FLOOR
    return cfg


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def _gate_diagnostics(agent: REEAgent) -> Dict:
    """Extract BetaGate MECH-090 R-c diagnostics."""
    s = agent.beta_gate.get_state()
    return {
        "n_admitted": int(s.get("mech090_n_elevation_admitted", 0)),
        "n_blocked": int(s.get("mech090_n_elevation_blocked", 0)),
        "n_single_candidate": int(s.get("mech090_n_elevation_single_candidate", 0)),
        "last_margin": float(s.get("mech090_last_readiness_score_margin", 0.0)),
        "use_commit_readiness_gate": bool(s.get("use_commit_readiness_gate", False)),
        "commit_readiness_floor": float(s.get("commit_readiness_floor", 0.0)),
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    p0_budget = 5 if dry_run else P0_BUDGET
    p2_eps = 3 if dry_run else P2_EPISODES
    p2_steps = 20 if dry_run else P2_STEPS_PER_EPISODE
    p0_steps = 20 if dry_run else P0_STEPS_PER_EPISODE

    # ------------------------------------------------------------------
    # ARM_0: GATED (degenerate basin falsifier)
    # Reproduce the V3-EXQ-592 EMERGENT-arm curriculum with gate ON.
    # Seed 42 is the degenerate signature; seeds 43/44 may or may not
    # exhibit the basin -- only the population minimum matters for
    # ARM_0 acceptance.
    # ------------------------------------------------------------------
    print(f"Seed {seed} Condition ARM_0_GATED", flush=True)
    agent_main = REEAgent(make_gated_cfg()).to(device)
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

    if p0.aborted:
        # Abort means rv never crossed -- gate is upstream-bypassed; not a
        # falsifier signature. Record zero metrics for the P2 stage.
        metrics_gated = CommittedModeMetrics(
            total_committed_steps=0, total_beta_elevated=0,
            hold_rate=0.0, mean_committed_steps_per_ep=0.0,
            rule_state_norm=0.0, n_eval_episodes=0, per_episode=[],
        )
    else:
        target_env_a = make_target_env(seed)
        metrics_gated = run_p2_eval(
            agent_main, target_env_a, device,
            n_eps=p2_eps, steps_per_episode=p2_steps,
        )
    gate_diag_arm0 = _gate_diagnostics(agent_main)
    final_rv_arm0 = float(getattr(agent_main.e3, "_running_variance", 0.0))
    commit_threshold_arm0 = float(getattr(agent_main.e3, "commit_threshold", 0.40))

    # ARM_0 falsifier: rv crossed (final_rv < commit_threshold) AND
    # gate blocked at least once AND no committed steps in P2.
    rv_crossed = final_rv_arm0 < commit_threshold_arm0
    arm0_falsifier = (
        rv_crossed
        and gate_diag_arm0["n_blocked"] >= 1
        and metrics_gated.total_committed_steps == 0
    )
    print(
        f"[seed={seed}] ARM_0_GATED:"
        f" rv_final={final_rv_arm0:.5f} (threshold={commit_threshold_arm0:.3f}, crossed={rv_crossed})"
        f" n_admitted={gate_diag_arm0['n_admitted']}"
        f" n_blocked={gate_diag_arm0['n_blocked']}"
        f" last_margin={gate_diag_arm0['last_margin']:.4f}"
        f" committed_steps={metrics_gated.total_committed_steps}"
        f" falsifier_fired={arm0_falsifier}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # ARM_1: GATED_FORCED_READY (admit-when-ready)
    # Clone the trained agent + force rv=0.001. Trained-agent candidate
    # scores should produce non-trivial margins (the substrate finding the
    # gate is supposed to admit on). If P0 aborted (untrained), use a fresh
    # agent with forced rv to still exercise the measurement path -- in
    # that case ARM_1 may itself surface a near-tie margin signature, which
    # is a documented secondary diagnostic.
    # ------------------------------------------------------------------
    print(f"Seed {seed} Condition ARM_1_GATED_FORCED_READY", flush=True)
    if p0.aborted:
        agent_forced = REEAgent(make_gated_cfg()).to(device)
    else:
        agent_forced = clone_trained_agent(agent_main, bistable=True, device=device)
        # Ensure gate config survives the clone -- clone_trained_agent
        # may not propagate the HeartbeatConfig knobs; re-wire defensively.
        agent_forced.config.heartbeat.use_commit_readiness_gate = (
            COMMIT_READINESS_GATE_ENABLED
        )
        agent_forced.config.heartbeat.commit_readiness_floor = COMMIT_READINESS_FLOOR
        # The BetaGate instance itself is constructed at agent __init__ and
        # captures the knobs there. The clone path may have produced a fresh
        # BetaGate that already captured the right knobs from the cloned
        # config; if not, fall through to a defensive re-wire of the gate.
        agent_forced.beta_gate._use_commit_readiness_gate = (
            COMMIT_READINESS_GATE_ENABLED
        )
        agent_forced.beta_gate._commit_readiness_floor = COMMIT_READINESS_FLOOR
    agent_forced.e3._running_variance = 0.001  # force below threshold

    target_env_b = make_target_env(seed)
    metrics_forced = run_p2_eval(
        agent_forced, target_env_b, device,
        n_eps=p2_eps, steps_per_episode=p2_steps,
    )
    gate_diag_arm1 = _gate_diagnostics(agent_forced)
    arm1_admit = (
        gate_diag_arm1["n_admitted"] >= 1
        and metrics_forced.total_committed_steps > 0
    )
    print(
        f"[seed={seed}] ARM_1_GATED_FORCED_READY:"
        f" n_admitted={gate_diag_arm1['n_admitted']}"
        f" n_blocked={gate_diag_arm1['n_blocked']}"
        f" last_margin={gate_diag_arm1['last_margin']:.4f}"
        f" committed_steps={metrics_forced.total_committed_steps}"
        f" admit_fired={arm1_admit}",
        flush=True,
    )

    verdict_seed = arm0_falsifier or arm1_admit  # any positive signature per seed
    return {
        "seed": seed,
        "arm0_gated": {
            "p0_converged": p0.converged,
            "p0_aborted": p0.aborted,
            "p0_abort_reason": p0.abort_reason,
            "p0_n_episodes": p0.n_episodes,
            "final_rv": final_rv_arm0,
            "commit_threshold": commit_threshold_arm0,
            "rv_crossed": rv_crossed,
            "total_committed_steps": metrics_gated.total_committed_steps,
            "mean_committed_steps_per_ep": metrics_gated.mean_committed_steps_per_ep,
            "gate_diagnostics": gate_diag_arm0,
            "falsifier_fired": arm0_falsifier,
        },
        "arm1_gated_forced_ready": {
            "total_committed_steps": metrics_forced.total_committed_steps,
            "mean_committed_steps_per_ep": metrics_forced.mean_committed_steps_per_ep,
            "gate_diagnostics": gate_diag_arm1,
            "admit_fired": arm1_admit,
        },
        "seed_pass": verdict_seed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Short run (5 ep P0, 3 ep P2) to test wiring without writing evidence.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    seeds = args.seeds

    print(f"[{QUEUE_ID}] MECH-090 R-c commit-readiness gate validation", flush=True)
    print(f"  dry_run={dry_run}  seeds={seeds}", flush=True)
    print(
        f"  COMMIT_READINESS_FLOOR={COMMIT_READINESS_FLOOR}"
        f"  GATE_ENABLED={COMMIT_READINESS_GATE_ENABLED}",
        flush=True,
    )

    per_seed_results = []
    for seed in seeds:
        result = run_seed(seed, dry_run=dry_run)
        per_seed_results.append(result)

    arm0_blocks_seeds = sum(
        1 for r in per_seed_results if r["arm0_gated"]["falsifier_fired"]
    )
    arm1_admits_seeds = sum(
        1 for r in per_seed_results if r["arm1_gated_forced_ready"]["admit_fired"]
    )

    arm0_pass = arm0_blocks_seeds >= ARM_0_MIN_SEEDS_WITH_BLOCKS
    arm1_pass = arm1_admits_seeds >= ARM_1_MIN_SEEDS_WITH_ADMITS
    experiment_passes = arm0_pass and arm1_pass

    print(f"\n[{QUEUE_ID}] ARM_0 falsifier seeds: {arm0_blocks_seeds}/{len(seeds)} (need >= {ARM_0_MIN_SEEDS_WITH_BLOCKS})", flush=True)
    print(f"[{QUEUE_ID}] ARM_1 admit seeds:    {arm1_admits_seeds}/{len(seeds)} (need >= {ARM_1_MIN_SEEDS_WITH_ADMITS})", flush=True)
    print(f"[{QUEUE_ID}] ARM_0={'PASS' if arm0_pass else 'FAIL'}  ARM_1={'PASS' if arm1_pass else 'FAIL'}", flush=True)
    print(f"[{QUEUE_ID}] Experiment: {'PASS' if experiment_passes else 'FAIL'}", flush=True)

    outcome = "PASS" if experiment_passes else "FAIL"
    print(f"verdict: {outcome}", flush=True)

    if dry_run:
        print(f"[{QUEUE_ID}] DRY RUN -- not writing evidence.", flush=True)
        return outcome, None

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
        "thresholds": {
            "commit_readiness_floor": COMMIT_READINESS_FLOOR,
            "arm0_min_seeds_with_blocks": ARM_0_MIN_SEEDS_WITH_BLOCKS,
            "arm1_min_seeds_with_admits": ARM_1_MIN_SEEDS_WITH_ADMITS,
        },
        "aggregate": {
            "arm0_blocks_seeds": arm0_blocks_seeds,
            "arm1_admits_seeds": arm1_admits_seeds,
            "arm0_pass": arm0_pass,
            "arm1_pass": arm1_pass,
        },
        "per_seed_results": per_seed_results,
        "notes": (
            "MECH-090 R-c commit-entry readiness conjunction substrate validation. "
            "Predecessor synthesis: REE_assembly/evidence/literature/targeted_review_"
            "connectome_mech_090/synthesis.md commit 9e68c5ca8a. "
            "Predecessor session: implement-substrate-mech090-commit-predicate-"
            "20260528T172800Z. Substrate landed 2026-05-28 in ree-v3/ree_core/"
            "heartbeat/beta_gate.py + ree-v3/ree_core/agent.py + ree-v3/ree_core/"
            "utils/config.py. Joint PASS unblocks commitment_closure:GAP-4 "
            "substrate-side (partial -> done) and queues Phase 4/5 *b cohort "
            "behavioural arms. Joint FAIL routes per the interpretation grid in "
            "the experiment docstring."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"[{QUEUE_ID}] Evidence written -> {out_path}", flush=True)

    return outcome, out_path


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
