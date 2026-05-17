#!/opt/local/bin/python3
"""V3-EXQ-503a -- SD-017 sleep-phase discriminative pair, Phase 2 retest.
SLEEP DRIVER: K=1 single-fire (SleepLoopManager, default cycle_every_k_episodes=1, fires every episode)

SUPERSEDES: V3-EXQ-503 (ran on the pre-Phase-2 substrate; the SD-016 confounded
baseline meant ContextMemory state was already collapsed prior to the SWS schema
pass, so the discriminative |diff| signal between FULL_4_PHASE_ON and
NO_SLEEP_BASELINE was confounded by the SD-016 substrate gap rather than driven
purely by the SD-017 sleep cycle).

MECHANISM UNDER TEST: SD-017 (sleep_phase.minimal_sleep_infrastructure_v3)
  Substrate-level discriminative pair -- ARM_A enables sws+rem and runs the full
  four-phase sleep cycle; ARM_B short-circuits sleep. Both arms preload identical
  N=20 deterministic experience prototypes and run K=3 cycles in identical loops.
  The only manipulated variable is the SD-017 substrate's enable state. NOW under
  the Phase 2 substrate stack (5 flags applied to BOTH arms; use_sleep_loop only
  on the FULL_4_PHASE_ON arm).

EXPERIMENT_PURPOSE: evidence

WHY THE RETEST: V3-EXQ-265a PASS (2026-05-09T20:12Z) validated the Phase 2
substrate template end-to-end on the SD-017 methods-validation experiment.
V3-EXQ-500a was queued 2026-05-09T20:43Z under the same template. Per
sleep_substrate_plan.md GAP-2 plan-of-record (decision-log entries
2026-05-09T19:49Z and 2026-05-09T20:14Z), the Tier-1 successor cohort
mechanically applies the validated 5-flag template to each base script.
503a is the discriminative-pair successor; its acceptance shape differs from
265a/500a/418c because it is an arm-A-vs-arm-B comparison rather than a
within-experiment slot-diversity contrast.

CLAIM_IDS RE-EVALUATION (per CLAUDE.md accuracy rule): the original 503
tagged ["SD-017"] for the SD-017 sleep-substrate discriminative pair. The
mechanism under test is unchanged -- still SD-017 first-class methods
(run_sws_schema_pass / run_rem_attribution_pass / run_sleep_cycle), exercised
in a discriminative-pair design under the Phase 2 substrate stack on both
arms. claim_ids=["SD-017"] preserved. evidence_direction_per_claim is not
strictly required (single claim) but the original 503 emits it; preserved for
indexer consistency.

PHASE 2 SUBSTRATE TEMPLATE (5 flags applied to BOTH arms):
  sd016_writepath_mode = "off"           (SD-016 Path 1, A2_div_only)
  sd016_diversification_weight = 0.5
  use_per_stream_vs = True               (MECH-269 Phase 1)
  use_anchor_sets = True                 (MECH-269 Phase 2 ii dual-trace)
  use_sd039_anchor_payload = True        (substrate-level)

ARM-SPECIFIC SLEEP FLAGS:
  ARM_A FULL_4_PHASE_ON:  sws_enabled=True, rem_enabled=True, use_sleep_loop=True
  ARM_B NO_SLEEP_BASELINE: sws_enabled=False, rem_enabled=False, use_sleep_loop=False

THREE PRE-REGISTERED METRICS (preserved from EXQ-503):
  M1: cumulative_sws_writes -- ARM_A predicts >= K, ARM_B predicts == 0.
  M2: ctxmem_state_change   -- Frobenius norm of (ContextMemory after K cycles
                               minus ContextMemory before). ARM_A predicts a
                               measurable change; ARM_B predicts ~0 (matrix
                               untouched without sleep).
  M3: cumulative_rem_rollouts -- ARM_A predicts >= K, ARM_B predicts == 0.

ACCEPTANCE CRITERIA (>= 2/3 seeds for each):
  C1: ARM_A cumulative_sws_writes >= N_CYCLES AND ARM_B == 0
  C2: ARM_A ctxmem_state_change >= 0.10 AND ARM_B <= 1e-6
  C3: ARM_A cumulative_rem_rollouts >= N_CYCLES AND ARM_B == 0
  C4: cross-arm signed |arm_A.M2 - arm_B.M2| > C4_CTXMEM_DIFF_TOLERANCE
      (calibrated from smoke-test output, NOT reused from 265a/500a slot-
      diversity 0.05; ctxmem_state_change Frobenius norm has a different
      metric scale than slot pairwise distance).

PASS: C1 AND C2 AND C3 AND C4 across the >= 2/3 seeds threshold.

INTERPRETATION GRID (for the discussant reviewing this):
  All PASS                    -> Phase 2 substrate preserves SD-017 sleep
                                 cycle's discriminative signature cleanly.
                                 SD-017 first-class methods produce a
                                 measurable ContextMemory state change in
                                 ARM_A that is absent in ARM_B; |diff| clears
                                 the calibrated tolerance. Roll
                                 sleep_substrate_plan.md GAP-2 owner-EXQ
                                 list forward; 503a closes.

  C1/C3 FAIL only             -> SD-017 substrate writes/rollouts did not
                                 fire as expected in ARM_A. Likely a Phase 2
                                 interaction (SD-016 div loss + V_s gating
                                 suppressing prototype sampling, or
                                 anchor_set query producing empty seeds for
                                 the SWS pass). Route via /diagnose-errors.

  C2 FAIL only                -> SWS/REM activate (M1/M3 pass) but
                                 ContextMemory change in ARM_A falls below
                                 0.10 threshold under Phase 2. SD-016 div
                                 loss may be reorganising slots before the
                                 SWS pass writes, leaving less room for
                                 schema installation to move them further.
                                 Architectural concern; flag to
                                 sleep-substrate cluster.

  C4 FAIL with C1/C2/C3 pass  -> Cross-arm discriminability collapsed --
                                 ARM_B's ctxmem also moves under Phase 2
                                 (e.g. SD-016 div loss writing to slots
                                 even without sleep). Indicates the
                                 discriminative-pair design is now
                                 confounded by Phase 2 substrate side
                                 effects independent of the SD-017
                                 manipulation. Successor 503b would need
                                 to control for div-loss baseline drift.

  C2+C3 FAIL                  -> Joint-mode failure: substrate
                                 incompatibility with SD-017 cycle. Would
                                 invalidate the 265a-validated template
                                 propagation assumption for discriminative-
                                 pair successors and require re-examination
                                 of the per-EXQ template diff.

claim_ids: ["SD-017"]
experiment_purpose: "evidence"
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
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_503a_sd017_sleep_phase_discriminative_phase2"
QUEUE_ID = "V3-EXQ-503a"
SUPERSEDES = "V3-EXQ-503"
CLAIM_IDS = ["SD-017"]
EXPERIMENT_PURPOSE = "evidence"

# --- Configuration --------------------------------------------------------
SEEDS = (42, 43, 44)
N_PROTOTYPES = 20         # deterministic experience buffer prototypes
N_CYCLES = 3              # sleep cycles per condition
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6

# Forced waking dims (match V3-EXQ-503 / V3-EXQ-500a setup so the sleep
# API is exercised against realistic dimensions).
SELF_DIM = 32
WORLD_DIM = 32
ACTION_DIM = 5
HARM_DIM = 1
HARM_A_DIM = 1
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250

# Phase 2 substrate template (validated by V3-EXQ-265a PASS 2026-05-09T20:12Z;
# propagated to V3-EXQ-500a 2026-05-09T20:43Z). Same five flags applied across
# both arms of this discriminative-pair experiment.
SD016_DIVERSIFICATION_WEIGHT = 0.5

# Pre-registered thresholds.
# C1 cumulative_sws_writes (preserved from EXQ-503): ARM_A executes the SWS
# schema pass at every cycle (one write per consolidation step, capped at
# buffer size); ARM_B's pass short-circuits with sws_enabled=False so
# writes stay at 0.
C1_ARM_A_MIN_CUMULATIVE_WRITES = N_CYCLES * 1
C1_ARM_B_MAX_CUMULATIVE_WRITES = 0
# C2 ctxmem_state_change (preserved from EXQ-503): Frobenius norm of
# (memory_after - memory_before). ARM_A reorganises slots; ARM_B is
# expected to leave the matrix untouched. NOTE: under the Phase 2
# substrate the SD-016 diversification loss is enabled on BOTH arms but
# only fires under training pressure (this experiment runs zero training
# steps -- buffer-only preload + sleep cycles -- so the div loss has no
# training signal to backprop and ARM_B's matrix remains untouched).
C2_ARM_A_MIN_CTXMEM_DELTA = 0.10
C2_ARM_B_MAX_CTXMEM_DELTA = 1e-6
# C3 cumulative_rem_rollouts (preserved from EXQ-503): REM pass attributes
# against theta_buffer recent; ARM_A produces > 0; ARM_B is a no-op.
C3_ARM_A_MIN_CUMULATIVE_REM = N_CYCLES * 1
C3_ARM_B_MAX_CUMULATIVE_REM = 0
# C4 cross-arm |diff| on M2 (NEW vs EXQ-503): calibrated from the dry-run
# smoke output. The Frobenius-norm scale of ctxmem_state_change is
# different from slot pairwise distance (which is what 265a/500a's 0.05
# was set against), so we calibrate from the actual smoke value rather
# than reusing 0.05. Smoke (1 seed dry-run on Mac, 2026-05-09T21:31Z):
# ARM_A ctxmem_state_change = 5.0323, ARM_B = 0.0. |diff| = 5.0323.
# Setting tolerance to 0.20 gives a ~25x safety margin against the
# observed value: well above the existing C2 ARM_A threshold (0.10) and
# far below the smoke magnitude, so cross-seed variation can absorb
# ARM_A regressions of up to ~96% before C4 fails. Picked above the
# existing C2 floor rather than as a percentage of smoke so the C4
# logic remains interpretable across seeds with very different ARM_A
# magnitudes.
C4_CTXMEM_DIFF_TOLERANCE = 0.20
PASS_FRACTION_REQUIRED = 2.0 / 3.0


# --- Helpers --------------------------------------------------------------
def _make_agent(seed: int, sleep_enabled: bool) -> REEAgent:
    """Build agent with Phase 2 substrate template on BOTH arms.

    Phase 2 stack (sd016 div loss + MECH-269 V_s + anchor sets + SD-039
    payload) applies independent of sleep on/off so it is configured
    identically across arms. Only sws_enabled / rem_enabled / use_sleep_loop
    differentiate ARM_A (FULL_4_PHASE_ON) from ARM_B (NO_SLEEP_BASELINE).

    Note: anchor_sets requires per_stream_vs ON (precondition raised in
    HippocampalModule). They are wired together here, matching 265a/500a.
    """
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        # Phase 2 substrate template (5 flags, applied to BOTH arms).
        # SD-016 Path 1 diversification (A2_div_only: writepath off, div on).
        sd016_writepath_mode="off",
        sd016_diversification_weight=SD016_DIVERSIFICATION_WEIGHT,
        # MECH-269 Phase 1 + Phase 2 (ii)
        use_per_stream_vs=True,
        use_anchor_sets=True,
        # SD-039 substrate-side anchor payload
        use_sd039_anchor_payload=True,
        # SD-017 sleep flags (arm-specific).
        sws_enabled=bool(sleep_enabled),
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        sws_schema_weight=0.1,
        rem_enabled=bool(sleep_enabled),
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        # SD-017 SleepLoopManager (Phase A scaffolding) -- only on the
        # FULL_4_PHASE_ON arm; bit-identical OFF on the baseline arm.
        use_sleep_loop=bool(sleep_enabled),
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

    Reported as a diagnostic; the discriminative metric is M2
    (ctxmem_state_change Frobenius norm via _measure_ctxmem_state).
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
def run_arm(seed: int, arm_label: str, sleep_enabled: bool, progress_denominator: int) -> dict:
    """Run one arm: K sleep cycles, capturing per-cycle metrics."""
    print(f"Seed {seed} Condition {arm_label}", flush=True)
    agent = _make_agent(seed=seed, sleep_enabled=sleep_enabled)
    _preload_experience_buffers(agent, N_PROTOTYPES, seed=seed)

    diff_before = _measure_slot_differentiation(agent)
    ctxmem_before = _measure_ctxmem_state(agent)

    per_cycle_diff: list[float] = []
    per_cycle_metrics: list[dict] = []
    cumulative_sws_writes = 0.0
    cumulative_rem_rollouts = 0.0

    for cycle_idx in range(N_CYCLES):
        print(
            f"  [train] {arm_label} seed={seed} ep {cycle_idx + 1}/{progress_denominator}",
            flush=True,
        )
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
    c4_passes = 0
    per_seed_m2_diffs: list[float] = []
    for a, b in zip(arm_a_results, arm_b_results):
        c1 = (a["cumulative_sws_writes"] >= C1_ARM_A_MIN_CUMULATIVE_WRITES
              and b["cumulative_sws_writes"] <= C1_ARM_B_MAX_CUMULATIVE_WRITES)
        c2 = (a["ctxmem_state_change"] >= C2_ARM_A_MIN_CTXMEM_DELTA
              and b["ctxmem_state_change"] <= C2_ARM_B_MAX_CTXMEM_DELTA)
        c3 = (a["cumulative_rem_rollouts"] >= C3_ARM_A_MIN_CUMULATIVE_REM
              and b["cumulative_rem_rollouts"] <= C3_ARM_B_MAX_CUMULATIVE_REM)
        m2_diff = abs(a["ctxmem_state_change"] - b["ctxmem_state_change"])
        per_seed_m2_diffs.append(m2_diff)
        c4 = m2_diff > C4_CTXMEM_DIFF_TOLERANCE
        c1_passes += int(c1)
        c2_passes += int(c2)
        c3_passes += int(c3)
        c4_passes += int(c4)

    return {
        "n_seeds": n_seeds,
        "min_seeds_required": required,
        "c1_cumulative_sws_writes_seeds_pass": c1_passes,
        "c2_ctxmem_state_change_seeds_pass": c2_passes,
        "c3_cumulative_rem_rollouts_seeds_pass": c3_passes,
        "c4_cross_arm_m2_diff_seeds_pass": c4_passes,
        "per_seed_m2_diffs": per_seed_m2_diffs,
        "c1_pass": c1_passes >= required,
        "c2_pass": c2_passes >= required,
        "c3_pass": c3_passes >= required,
        "c4_pass": c4_passes >= required,
        "overall_pass": (c1_passes >= required
                         and c2_passes >= required
                         and c3_passes >= required
                         and c4_passes >= required),
    }


# --- Driver ---------------------------------------------------------------
def main() -> tuple[str, str | None]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single seed and print results without writing manifest.")
    args = parser.parse_args()

    seeds = (SEEDS[0],) if args.dry_run else SEEDS
    progress_denominator = N_CYCLES
    t0 = time.time()
    arm_a_results = [
        run_arm(s, "ARM_A_full_4_phase_on", sleep_enabled=True,
                progress_denominator=progress_denominator)
        for s in seeds
    ]
    arm_b_results = [
        run_arm(s, "ARM_B_no_sleep_baseline", sleep_enabled=False,
                progress_denominator=progress_denominator)
        for s in seeds
    ]
    elapsed = time.time() - t0

    criteria = _evaluate_criteria(arm_a_results, arm_b_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    # Per-seed verdict lines (one per seed across both arms; runner expects
    # seeds * conditions verdict lines, but our 2-arm structure is two
    # passes through SEEDS so emit one verdict per seed pairing).
    for a, b in zip(arm_a_results, arm_b_results):
        m2_diff = abs(a["ctxmem_state_change"] - b["ctxmem_state_change"])
        seed_pass = (
            a["cumulative_sws_writes"] >= C1_ARM_A_MIN_CUMULATIVE_WRITES
            and b["cumulative_sws_writes"] <= C1_ARM_B_MAX_CUMULATIVE_WRITES
            and a["ctxmem_state_change"] >= C2_ARM_A_MIN_CTXMEM_DELTA
            and b["ctxmem_state_change"] <= C2_ARM_B_MAX_CTXMEM_DELTA
            and a["cumulative_rem_rollouts"] >= C3_ARM_A_MIN_CUMULATIVE_REM
            and b["cumulative_rem_rollouts"] <= C3_ARM_B_MAX_CUMULATIVE_REM
            and m2_diff > C4_CTXMEM_DIFF_TOLERANCE
        )
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    print(f"{QUEUE_ID} (SD-017) -- {outcome} in {elapsed:.1f}s ({len(seeds)} seed(s))", flush=True)
    for label, results in (("ARM_A_full_4_phase_on", arm_a_results),
                           ("ARM_B_no_sleep_baseline", arm_b_results)):
        for r in results:
            print(f"  {label} seed={r['seed']}  "
                  f"sws_writes={r['cumulative_sws_writes']:.0f}  "
                  f"ctxmem_delta={r['ctxmem_state_change']:.4f}  "
                  f"rem_rollouts={r['cumulative_rem_rollouts']:.0f}", flush=True)
    print(f"  C1 cumulative_sws_writes:    {criteria['c1_cumulative_sws_writes_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c1_pass'] else 'FAIL'}", flush=True)
    print(f"  C2 ctxmem_state_change:      {criteria['c2_ctxmem_state_change_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c2_pass'] else 'FAIL'}", flush=True)
    print(f"  C3 cumulative_rem_rollouts:  {criteria['c3_cumulative_rem_rollouts_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c3_pass'] else 'FAIL'}", flush=True)
    print(f"  C4 cross-arm |diff| on M2:   {criteria['c4_cross_arm_m2_diff_seeds_pass']}/{criteria['n_seeds']} "
          f"-> {'PASS' if criteria['c4_pass'] else 'FAIL'} "
          f"(tolerance={C4_CTXMEM_DIFF_TOLERANCE:.4f}; "
          f"per-seed diffs={[round(d, 4) for d in criteria['per_seed_m2_diffs']]})", flush=True)

    if args.dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return outcome, None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
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
            "C4_CTXMEM_DIFF_TOLERANCE": C4_CTXMEM_DIFF_TOLERANCE,
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
            # Phase 2 substrate template flags (recorded for indexer audit).
            "sd016_writepath_mode": "off",
            "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "use_sd039_anchor_payload": True,
            # use_sleep_loop is arm-specific (True on ARM_A only); recorded
            # for completeness but not a single scalar across the run.
        },
        "results_arm_a_full_4_phase_on": arm_a_results,
        "results_arm_b_no_sleep_baseline": arm_b_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Phase 2 retest of SD-017 sleep-substrate discriminative pair. "
            "Successor to V3-EXQ-503 under the validated 5-flag substrate "
            "template (SD-016 div loss + MECH-269 per-stream V_s + anchor "
            "sets + SD-039 payload). Both arms preload identical N=20 "
            "deterministic experience prototypes and run K=3 sleep cycles. "
            "ARM_A enables sws+rem and use_sleep_loop (full 4-phase regime); "
            "ARM_B disables sws+rem and use_sleep_loop (NO_SLEEP_BASELINE). "
            "Phase 2 flags applied independent of sleep on/off (apply to "
            "BOTH arms). Four pre-registered metrics: M1 cumulative SWS "
            "writes, M2 ctxmem_state_change Frobenius norm, M3 cumulative "
            "REM rollouts, plus C4 cross-arm |M2 diff| > calibrated "
            "tolerance. C4 tolerance calibrated from --dry-run smoke output "
            "(5.0323 ARM_A vs 0.0 ARM_B); set to 0.20 (above the existing "
            "C2 ARM_A threshold 0.10 and ~25x below the smoke magnitude). "
            "NOT reused from 265a/500a's 0.05 (slot-diversity scale). "
            "PASS supports SD-017 architecture "
            "preserving discriminative substrate signal under Phase 2 "
            "stack; FAIL routes per the docstring interpretation grid."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)

    return outcome, str(out_path)


if __name__ == "__main__":
    _outcome, _out_path = main()
    _outcome_clean = str(_outcome).upper() if _outcome in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome_clean,
        manifest_path=_out_path,
    )
