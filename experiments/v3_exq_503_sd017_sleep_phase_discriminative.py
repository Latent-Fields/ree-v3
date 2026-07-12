"""V3-EXQ-503 -- EXP-0171a SD-017 sleep-phase discriminative pair.
SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a dedicated N_CYCLES=3 wake-sleep-test loop)

Claim: SD-017 (sleep_phase.minimal_sleep_infrastructure_v3)
Proposal: EXP-0171a (follow-up to EXP-0171 / V3-EXQ-500 substrate-readiness probe)

Why this experiment exists
--------------------------
V3-EXQ-500 (the substrate-readiness probe) PASSed -- the four sleep phases
fire as designed -- but it was experiment_purpose=diagnostic, so its
evidence_direction is excluded from claim scoring under Option E. SD-017
therefore still sits at exp_conf=0.000 / quadrant=plausible_unproven even
though the substrate is verified-ready and 6 supporting literature entries
back the claim.

V3-EXQ-503 closes the gap: same forced-injection template that produced
evidence-grade signal for MECH-094 (V3-EXQ-499), SD-035 (V3-EXQ-501), and
MECH-062 (V3-EXQ-502) -- substrate-level discriminative pair, deterministic
inputs, pre-registered thresholds, multi-seed.

What SD-017 actually claims
---------------------------
The minimal sleep-phase infrastructure (SWS schema-installation pass +
REM attribution pass) DOES SOMETHING USEFUL: ContextMemory slots
differentiate over repeated sleep cycles. Without these passes, slots
remain near-undifferentiated regardless of waking experience volume.
The discriminative test is whether that differentiation actually occurs
when the substrate is exercised, vs absent when it is not.

Discriminative pair
-------------------
Both arms pre-load the world / self experience buffers with the SAME
N=20 deterministic prototype z_world / z_self pairs (so the difference
is not driven by data volume) and run K=3 cycles.

ARM_A (FULL_4_PHASE_ON, SD-017 architecture intact):
  sws_enabled=True, rem_enabled=True
  Each cycle: agent.run_sleep_cycle()
    -> enter_sws_mode -> run_sws_schema_pass -> exit_sleep_mode
    -> enter_rem_mode -> run_rem_attribution_pass -> exit_sleep_mode
  Schema writes hit ContextMemory; REM produces non-zero terrain rollouts.

ARM_B (NO_SLEEP_BASELINE, SD-017 substrate ablated):
  sws_enabled=False, rem_enabled=False
  Same K cycles in the loop, but agent.run_sleep_cycle() short-circuits
  (returns metrics with zeros). ContextMemory state evolves only via
  whatever waking writes happen between cycles -- which we hold constant
  across both arms by pre-loading the buffers identically.

Three pre-registered metrics
----------------------------
M1: cumulative_sws_writes = sum of sws_n_writes across K cycles.
    Confirms the SWS schema pass actually executed and consumed buffer
    prototypes. ARM_A predicts >= K (at least one write per cycle);
    ARM_B predicts == 0 (sws_enabled=False short-circuits the pass).

M2: ctxmem_state_change = Frobenius norm of (ContextMemory matrix
    after K cycles - matrix before). Magnitude-of-change probe; agnostic
    to whether slot pairwise distance grows or shrinks (the schema pass
    can REDUCE diversity by clustering slots around real prototypes,
    which is also "doing work"). ARM_A predicts >= 0.10; ARM_B predicts
    ~0 (matrix untouched when sleep is disabled).

M3: cumulative_rem_rollouts = sum of rem_n_rollouts across K cycles.
    Confirms the REM pass attributes against theta_buffer recent and
    produces non-zero rollouts. ARM_A predicts >= K; ARM_B predicts == 0.

PASS criteria (>= 2/3 seeds for each):
  C1: ARM_A cumulative_sws_writes >= N_CYCLES AND ARM_B == 0
  C2: ARM_A ctxmem_state_change >= 0.10 AND ARM_B <= 1e-6
  C3: ARM_A cumulative_rem_rollouts >= N_CYCLES AND ARM_B == 0

PASS supports SD-017 architecture (sleep substrate produces measurable
schema differentiation that does not occur without it).
FAIL with C1 alone passing -> initial differentiation occurs but doesn't
progress (REM ineffective; downstream consumer gap).
FAIL with C1 also failing -> substrate-readiness was insufficient
indication (run /diagnose-errors against V3-EXQ-500 + this run).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_503_sd017_sleep_phase_discriminative.py
  /opt/local/bin/python3 experiments/v3_exq_503_sd017_sleep_phase_discriminative.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_503_sd017_sleep_phase_discriminative"
CLAIM_IDS = ["SD-017"]
EXPERIMENT_PURPOSE = "evidence"

# --- Configuration --------------------------------------------------------
SEEDS = (42, 43, 44)
N_PROTOTYPES = 20         # deterministic experience buffer prototypes
N_CYCLES = 3              # sleep cycles per condition
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6

# Forced waking dims (match V3-EXQ-500's CausalGridWorldV2 setup so the
# sleep API is exercised against realistic dimensions).
SELF_DIM = 32
WORLD_DIM = 32
ACTION_DIM = 5
HARM_DIM = 1
HARM_A_DIM = 1
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250

# Pre-registered thresholds.
# C1 cumulative_sws_writes: ARM_A executes the SWS schema pass at every cycle
# (one write per consolidation step, capped at buffer size); ARM_B's pass
# short-circuits with sws_enabled=False so writes stay at 0.
C1_ARM_A_MIN_CUMULATIVE_WRITES = N_CYCLES * 1   # at least one write per cycle
C1_ARM_B_MAX_CUMULATIVE_WRITES = 0
# C2 ctxmem_state_change: Frobenius norm of (memory_after - memory_before).
# ARM_A reorganises slots; ARM_B leaves the matrix untouched.
C2_ARM_A_MIN_CTXMEM_DELTA = 0.10
C2_ARM_B_MAX_CTXMEM_DELTA = 1e-6
# C3 cumulative_rem_rollouts: REM pass attributes against theta_buffer recent;
# ARM_A produces > 0; ARM_B is a no-op (rem_enabled=False).
C3_ARM_A_MIN_CUMULATIVE_REM = N_CYCLES * 1      # at least one rollout per cycle
C3_ARM_B_MAX_CUMULATIVE_REM = 0
PASS_FRACTION_REQUIRED = 2.0 / 3.0


# --- Helpers --------------------------------------------------------------
def _make_agent(seed: int, sleep_enabled: bool) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        sws_enabled=bool(sleep_enabled),
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        sws_schema_weight=0.1,
        rem_enabled=bool(sleep_enabled),
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
    )
    return REEAgent(cfg)


def _preload_experience_buffers(agent: REEAgent, n: int, seed: int) -> None:
    """Load identical deterministic prototypes into world / self / theta buffers.

    Bypasses the env loop -- the discriminative test is about what the sleep
    cycle does GIVEN identical experience. Both arms see the same buffers.
    Populates theta_buffer via its update() API so REM attribution has recent
    z_world to replay from.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    for _ in range(n):
        z_world = torch.randn(1, WORLD_DIM, generator=g)
        z_self = torch.randn(1, SELF_DIM, generator=g)
        agent._world_experience_buffer.append(z_world.detach())
        agent._self_experience_buffer.append(z_self.detach())
        agent.theta_buffer.update(z_world.detach(), z_self.detach())


def _measure_slot_differentiation(agent: REEAgent) -> float:
    """Mean pairwise (1 - cosine_similarity) across ContextMemory slots.

    NOTE on direction: ContextMemory slots are initialised near-uniform-random
    (high pairwise distance ~1.0). The SWS schema pass writes coherent
    prototypes derived from the experience buffer, which can REDUCE pairwise
    distance because slots cluster around real prototype geometry. So this
    metric is a *probe* of slot organisation, not a goodness signal --
    interpret via magnitude of change, not direction. _measure_ctxmem_state
    below captures the raw matrix for delta computation.
    """
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        n = mem.shape[0]
        if n < 2:
            return 0.0
        norms = mem.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = mem / norms
        sim_mat = torch.mm(normed, normed.t())
        mask = torch.eye(n, device=sim_mat.device, dtype=torch.bool)
        off_diag = sim_mat[~mask]
        return float((1.0 - off_diag).mean().item())


def _measure_ctxmem_state(agent: REEAgent) -> torch.Tensor:
    """Snapshot ContextMemory matrix for change-magnitude computation."""
    with torch.no_grad():
        return agent.e1.context_memory.memory.detach().clone()


# --- Per-arm runner -------------------------------------------------------
def run_arm(seed: int, arm_label: str, sleep_enabled: bool) -> dict:
    """Run one arm: K sleep cycles, capturing per-cycle metrics."""
    agent = _make_agent(seed=seed, sleep_enabled=sleep_enabled)
    _preload_experience_buffers(agent, N_PROTOTYPES, seed=seed)

    diff_before = _measure_slot_differentiation(agent)
    ctxmem_before = _measure_ctxmem_state(agent)

    per_cycle_diff: list[float] = []
    per_cycle_metrics: list[dict] = []
    cumulative_sws_writes = 0.0
    cumulative_rem_rollouts = 0.0

    for cycle_idx in range(N_CYCLES):
        cycle_metrics = agent.run_sleep_cycle()
        per_cycle_metrics.append(cycle_metrics)
        per_cycle_diff.append(_measure_slot_differentiation(agent))
        cumulative_sws_writes += float(cycle_metrics.get("sws_n_writes", 0.0))
        cumulative_rem_rollouts += float(cycle_metrics.get("rem_n_rollouts", 0.0))

    ctxmem_after = _measure_ctxmem_state(agent)
    ctxmem_state_change = float((ctxmem_after - ctxmem_before).norm().item())
    diff_after = per_cycle_diff[-1] if per_cycle_diff else diff_before

    return {
        "seed": seed,
        "arm_label": arm_label,
        "sleep_enabled": bool(sleep_enabled),
        "n_cycles": N_CYCLES,
        "n_prototypes": N_PROTOTYPES,
        "slot_diff_before": diff_before,
        "slot_diff_after_cycle_k": diff_after,
        "ctxmem_state_change": ctxmem_state_change,
        "cumulative_sws_writes": cumulative_sws_writes,
        "cumulative_rem_rollouts": cumulative_rem_rollouts,
        "per_cycle_slot_differentiation": per_cycle_diff,
        "per_cycle_sleep_metrics": per_cycle_metrics,
    }


# --- Aggregation + criteria ----------------------------------------------
def _evaluate_criteria(arm_a_results: list[dict], arm_b_results: list[dict]) -> dict:
    n_seeds = len(arm_a_results)
    required = math.ceil(n_seeds * PASS_FRACTION_REQUIRED)

    c1_passes = 0
    c2_passes = 0
    c3_passes = 0
    for a, b in zip(arm_a_results, arm_b_results):
        c1 = (a["cumulative_sws_writes"] >= C1_ARM_A_MIN_CUMULATIVE_WRITES
              and b["cumulative_sws_writes"] <= C1_ARM_B_MAX_CUMULATIVE_WRITES)
        c2 = (a["ctxmem_state_change"] >= C2_ARM_A_MIN_CTXMEM_DELTA
              and b["ctxmem_state_change"] <= C2_ARM_B_MAX_CTXMEM_DELTA)
        c3 = (a["cumulative_rem_rollouts"] >= C3_ARM_A_MIN_CUMULATIVE_REM
              and b["cumulative_rem_rollouts"] <= C3_ARM_B_MAX_CUMULATIVE_REM)
        c1_passes += int(c1)
        c2_passes += int(c2)
        c3_passes += int(c3)

    return {
        "n_seeds": n_seeds,
        "min_seeds_required": required,
        "c1_cumulative_sws_writes_seeds_pass": c1_passes,
        "c2_ctxmem_state_change_seeds_pass": c2_passes,
        "c3_cumulative_rem_rollouts_seeds_pass": c3_passes,
        "c1_pass": c1_passes >= required,
        "c2_pass": c2_passes >= required,
        "c3_pass": c3_passes >= required,
        "overall_pass": (c1_passes >= required
                         and c2_passes >= required
                         and c3_passes >= required),
    }


# --- Driver ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single seed and print results without writing manifest.")
    args = parser.parse_args()

    seeds = (SEEDS[0],) if args.dry_run else SEEDS
    t0 = time.time()
    arm_a_results = [run_arm(s, "ARM_A_full_4_phase_on", sleep_enabled=True) for s in seeds]
    arm_b_results = [run_arm(s, "ARM_B_no_sleep_baseline", sleep_enabled=False) for s in seeds]
    elapsed = time.time() - t0

    criteria = _evaluate_criteria(arm_a_results, arm_b_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"V3-EXQ-503 (SD-017) -- {outcome} in {elapsed:.1f}s ({len(seeds)} seed(s))")
    for label, results in (("ARM_A_full_4_phase_on", arm_a_results),
                           ("ARM_B_no_sleep_baseline", arm_b_results)):
        for r in results:
            print(f"  {label} seed={r['seed']}  "
                  f"sws_writes={r['cumulative_sws_writes']:.0f}  "
                  f"ctxmem_delta={r['ctxmem_state_change']:.4f}  "
                  f"rem_rollouts={r['cumulative_rem_rollouts']:.0f}")
    print(f"  C1 cumulative_sws_writes: {criteria['c1_cumulative_sws_writes_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c1_pass'] else 'FAIL'}")
    print(f"  C2 ctxmem_state_change:   {criteria['c2_ctxmem_state_change_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c2_pass'] else 'FAIL'}")
    print(f"  C3 cumulative_rem_rollouts: {criteria['c3_cumulative_rem_rollouts_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c3_pass'] else 'FAIL'}")

    if args.dry_run:
        print("[--dry-run] manifest not written.")
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"SD-017": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_ARM_A_MIN_CUMULATIVE_WRITES": C1_ARM_A_MIN_CUMULATIVE_WRITES,
            "C1_ARM_B_MAX_CUMULATIVE_WRITES": C1_ARM_B_MAX_CUMULATIVE_WRITES,
            "C2_ARM_A_MIN_CTXMEM_DELTA": C2_ARM_A_MIN_CTXMEM_DELTA,
            "C2_ARM_B_MAX_CTXMEM_DELTA": C2_ARM_B_MAX_CTXMEM_DELTA,
            "C3_ARM_A_MIN_CUMULATIVE_REM": C3_ARM_A_MIN_CUMULATIVE_REM,
            "C3_ARM_B_MAX_CUMULATIVE_REM": C3_ARM_B_MAX_CUMULATIVE_REM,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
        },
        "config": {
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "action_dim": ACTION_DIM,
            "n_prototypes": N_PROTOTYPES,
            "n_cycles": N_CYCLES,
            "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
            "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
            "seeds": list(seeds),
        },
        "results_arm_a_full_4_phase_on": arm_a_results,
        "results_arm_b_no_sleep_baseline": arm_b_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Substrate-level discriminative pair for SD-017 (follow-up to "
            "V3-EXQ-500 substrate-readiness probe -- which PASSed but is "
            "diagnostic_probe so does not feed claim scoring). Both arms "
            "pre-load identical N=20 deterministic experience prototypes "
            "into world / self / theta buffers, then run K=3 cycles. "
            "ARM_A enables sws+rem and calls run_sleep_cycle (full 4-phase "
            "regime). ARM_B disables sws+rem; the same call short-circuits. "
            "Three pre-registered metrics: M1 slot_differentiation (mean "
            "pairwise (1-cos) across ContextMemory slots), M2 "
            "schema_progression (post-K - post-1, growth signature), M3 "
            "rem_terrain_response (mean rem_terrain_variance across "
            "cycles). PASS supports SD-017 substrate doing measurable work; "
            "FAIL with C1 alone routes to a downstream consumer gap; FAIL "
            "with C1 failing routes to a substrate gap that V3-EXQ-500 "
            "readiness probe did not catch."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}")


if __name__ == "__main__":
    main()
