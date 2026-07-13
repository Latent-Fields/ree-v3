"""V3-EXQ-670 -- INV-048 pharmacological sleep phase disruption equivalence.
SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called K times with phase-specific ablations)

Claim: INV-048 (pharmacological sleep disruption causes attribution pipeline
degradation equivalent to behavioral sleep deprivation, proportional to which
phases are disrupted)

Proposal: EXP-0145

Why this experiment exists
--------------------------
INV-048 predicts that pharmacological agents that disrupt specific sleep phases
(REM-suppressing medications like antidepressants, or SWS-suppressing agents
like benzodiazepines) should produce attribution pipeline degradation equivalent
to behavioral sleep deprivation that affects the same phases. The key claim is
mechanism-agnostic equivalence: the pipeline responds only to phase fidelity,
not the cause of disruption.

In the REE substrate, we test this by comparing:
  1. Normal sleep (both SWS and REM enabled) - baseline attribution performance
  2. Pharmacological REM suppression (SWS only) - simulates antidepressants
  3. Pharmacological SWS suppression (REM only) - simulates benzodiazepines
  4. Complete sleep deprivation (neither phase) - behavioral baseline

INV-048 predicts that selective phase suppression (#2 and #3) should produce
attribution degradation intermediate between normal sleep and complete
deprivation, proportional to which phases are missing.

Discriminative design
---------------------
Four arms, each with N=3 seeds, running K=5 sleep cycles over a two-context
task (safe vs dangerous contexts). All arms receive identical waking experience
(deterministic z_world/z_self prototypes loaded into buffers).

ARM_A (NORMAL_SLEEP, baseline):
  sws_enabled=True, rem_enabled=True
  Full sleep architecture intact. Predicts best context discrimination and
  lowest harm rate in dangerous context.

ARM_B (REM_SUPPRESSED, pharmacological REM suppression analog):
  sws_enabled=True, rem_enabled=False
  SWS schema installation occurs, but REM attribution pass is skipped.
  Simulates medications that suppress REM (SSRIs, MAOIs, TCAs).
  Predicts: schema slots differentiate (SWS runs), but harm attribution to
  dangerous context is impaired (REM precision recalibration missing).

ARM_C (SWS_SUPPRESSED, pharmacological SWS suppression analog):
  sws_enabled=False, rem_enabled=True
  REM attribution runs, but SWS schema differentiation is skipped.
  Simulates benzodiazepines/Z-drugs (SWS depth suppression).
  Predicts: context slots remain undifferentiated (no schema installation),
  even though REM tries to attribute harm.

ARM_D (NO_SLEEP, complete behavioral sleep deprivation):
  sws_enabled=False, rem_enabled=False
  No offline consolidation. Baseline for comparison.
  Predicts worst performance (no schema differentiation, no attribution).

Pre-registered metrics (per seed, after K cycles)
--------------------------------------------------
M1: slot_differentiation = mean pairwise (1 - cosine_similarity) of
    ContextMemory slots. Higher = more differentiated contexts.
    Prediction: ARM_A > ARM_B > {ARM_C, ARM_D}
    (SWS is necessary for schema differentiation; REM alone insufficient)

M2: harm_discrimination = mean(harm_eval(z_dangerous)) - mean(harm_eval(z_safe))
    Positive = correctly assigns higher harm to dangerous context.
    Prediction: ARM_A > {ARM_B, ARM_C} > ARM_D
    (REM enhances harm attribution; but needs differentiated slots from SWS)

M3: cumulative_sws_writes = sum of sws_n_writes across K cycles.
    Manipulation check: ARM_A and ARM_B should have >0; ARM_C and ARM_D should be 0.

M4: cumulative_rem_rollouts = sum of rem_n_rollouts across K cycles.
    Manipulation check: ARM_A and ARM_C should have >0; ARM_B and ARM_D should be 0.

PASS criteria (>= 2/3 seeds for each):
  C1: slot_differentiation(ARM_A) > slot_differentiation(ARM_C) + 0.05
      (SWS is necessary for schema differentiation)
  C2: slot_differentiation(ARM_A) > slot_differentiation(ARM_D) + 0.05
      (Sleep is better than no sleep)
  C3: harm_discrimination(ARM_A) > harm_discrimination(ARM_D) + 0.10
      (Full sleep improves harm attribution vs no sleep)
  C4: harm_discrimination(ARM_B) < harm_discrimination(ARM_A) - 0.05
      (REM suppression impairs harm attribution vs normal sleep)
  C5: Manipulation checks pass:
      ARM_A cumulative_sws_writes >= K AND cumulative_rem_rollouts >= K
      ARM_B cumulative_sws_writes >= K AND cumulative_rem_rollouts == 0
      ARM_C cumulative_sws_writes == 0 AND cumulative_rem_rollouts >= K
      ARM_D cumulative_sws_writes == 0 AND cumulative_rem_rollouts == 0

PASS: (C1 AND C2 AND C3) OR (C1 AND C4) -- supports INV-048
FAIL otherwise

Run with:
  /opt/local/bin/python3 experiments/v3_exq_670_inv048_pharmacological_sleep_disruption_equivalence.py
  /opt/local/bin/python3 experiments/v3_exq_670_inv048_pharmacological_sleep_disruption_equivalence.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_670_inv048_pharmacological_sleep_disruption_equivalence"
CLAIM_IDS = ["INV-048"]
EXPERIMENT_PURPOSE = "evidence"

# --- Configuration --------------------------------------------------------
SEEDS = (42, 43, 44)
N_PROTOTYPES = 30         # deterministic experience buffer prototypes per context
N_CYCLES = 5              # sleep cycles per condition
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6

# Forced waking dims (match typical V3 substrate)
SELF_DIM = 32
WORLD_DIM = 32
ACTION_DIM = 5
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250

# Pre-registered thresholds
C1_SLOT_DIFF_MARGIN = 0.05  # ARM_A must beat ARM_C by this much
C2_SLOT_DIFF_MARGIN = 0.05  # ARM_A must beat ARM_D by this much
C3_HARM_DISC_MARGIN = 0.10  # ARM_A must beat ARM_D by this much
C4_HARM_DISC_MARGIN = 0.05  # ARM_A must beat ARM_B by this much
PASS_FRACTION_REQUIRED = 2.0 / 3.0

# Arms
ARM_A_NORMAL = "ARM_A_NORMAL_SLEEP"
ARM_B_REM_SUPP = "ARM_B_REM_SUPPRESSED"
ARM_C_SWS_SUPP = "ARM_C_SWS_SUPPRESSED"
ARM_D_NO_SLEEP = "ARM_D_NO_SLEEP"


# --- Helpers --------------------------------------------------------------
def _make_agent(seed: int, sws_enabled: bool, rem_enabled: bool) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        sws_enabled=sws_enabled,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
    )
    return REEAgent(cfg)


def _preload_two_context_buffers(
    agent: REEAgent, n_per_context: int, seed: int
) -> None:
    """Load deterministic prototypes for two distinct contexts (safe vs dangerous).

    Creates differentiated z_world distributions so the sleep passes have
    something to organize around. safe_context has lower variance; dangerous
    has higher.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    for context_id in [0, 1]:  # 0=safe, 1=dangerous
        scale = 0.5 if context_id == 0 else 1.5  # dangerous has higher variance
        for _ in range(n_per_context):
            z_world = torch.randn(1, WORLD_DIM, generator=g) * scale
            z_self = torch.randn(1, SELF_DIM, generator=g)
            agent._world_experience_buffer.append(z_world.detach())
            agent._self_experience_buffer.append(z_self.detach())
            agent.theta_buffer.update(z_world.detach(), z_self.detach())


def _measure_slot_differentiation(agent: REEAgent) -> float:
    """Mean pairwise (1 - cosine_similarity) across ContextMemory slots."""
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


def _measure_harm_discrimination(agent: REEAgent, seed: int) -> float:
    """Evaluate harm_eval on safe vs dangerous context prototypes.

    Returns mean(harm_dangerous) - mean(harm_safe).
    Positive = agent correctly assigns higher harm to dangerous context.
    """
    with torch.no_grad():
        g = torch.Generator()
        g.manual_seed(seed + 1000)  # offset to get fresh samples
        # Generate test prototypes matching the context distributions
        safe_zs = [torch.randn(1, WORLD_DIM, generator=g) * 0.5 for _ in range(10)]
        dangerous_zs = [torch.randn(1, WORLD_DIM, generator=g) * 1.5 for _ in range(10)]

        harm_safe = []
        harm_dangerous = []
        for z in safe_zs:
            # Simplified: if agent has harm_eval_head, use it; else use 0
            if hasattr(agent.e3, "harm_eval_head") and agent.e3.harm_eval_head is not None:
                h = agent.e3.harm_eval_head(z).item()
            else:
                h = 0.0
            harm_safe.append(h)
        for z in dangerous_zs:
            if hasattr(agent.e3, "harm_eval_head") and agent.e3.harm_eval_head is not None:
                h = agent.e3.harm_eval_head(z).item()
            else:
                h = 0.0
            harm_dangerous.append(h)

        mean_safe = sum(harm_safe) / len(harm_safe) if harm_safe else 0.0
        mean_dangerous = sum(harm_dangerous) / len(harm_dangerous) if harm_dangerous else 0.0
        return mean_dangerous - mean_safe


def run_arm(
    arm_name: str,
    seed: int,
    sws_enabled: bool,
    rem_enabled: bool,
) -> Dict:
    """Run one arm (one seed) through K sleep cycles."""
    agent = _make_agent(seed, sws_enabled, rem_enabled)
    _preload_two_context_buffers(agent, N_PROTOTYPES, seed)

    cumulative_sws_writes = 0
    cumulative_rem_rollouts = 0

    for cycle_idx in range(N_CYCLES):
        metrics = agent.run_sleep_cycle()
        cumulative_sws_writes += metrics.get("sws_n_writes", 0)
        cumulative_rem_rollouts += metrics.get("rem_n_rollouts", 0)

    slot_diff = _measure_slot_differentiation(agent)
    harm_disc = _measure_harm_discrimination(agent, seed)

    return {
        "arm_name": arm_name,
        "seed": seed,
        "slot_differentiation": slot_diff,
        "harm_discrimination": harm_disc,
        "cumulative_sws_writes": cumulative_sws_writes,
        "cumulative_rem_rollouts": cumulative_rem_rollouts,
        "sws_enabled": sws_enabled,
        "rem_enabled": rem_enabled,
    }


def run_experiment(dry_run: bool = False) -> Dict:
    """Run all four arms across all seeds."""
    all_results = []

    arm_configs = [
        (ARM_A_NORMAL, True, True),
        (ARM_B_REM_SUPP, True, False),
        (ARM_C_SWS_SUPP, False, True),
        (ARM_D_NO_SLEEP, False, False),
    ]

    for seed in SEEDS:
        for arm_name, sws_en, rem_en in arm_configs:
            result = run_arm(arm_name, seed, sws_en, rem_en)
            all_results.append(result)
            if dry_run:
                print(f"  {arm_name} seed={seed}: slot_diff={result['slot_differentiation']:.3f}, "
                      f"harm_disc={result['harm_discrimination']:.3f}, "
                      f"sws_writes={result['cumulative_sws_writes']}, "
                      f"rem_rollouts={result['cumulative_rem_rollouts']}")

    # Aggregate by arm
    arm_agg = {}
    for arm_name, _, _ in arm_configs:
        arm_results = [r for r in all_results if r["arm_name"] == arm_name]
        arm_agg[arm_name] = {
            "slot_differentiation_mean": sum(r["slot_differentiation"] for r in arm_results) / len(arm_results),
            "harm_discrimination_mean": sum(r["harm_discrimination"] for r in arm_results) / len(arm_results),
            "cumulative_sws_writes_mean": sum(r["cumulative_sws_writes"] for r in arm_results) / len(arm_results),
            "cumulative_rem_rollouts_mean": sum(r["cumulative_rem_rollouts"] for r in arm_results) / len(arm_results),
            "results": arm_results,
        }

    # Evaluate PASS criteria
    n_seeds = len(SEEDS)
    c1_pass_count = 0  # ARM_A slot_diff > ARM_C + margin
    c2_pass_count = 0  # ARM_A slot_diff > ARM_D + margin
    c3_pass_count = 0  # ARM_A harm_disc > ARM_D + margin
    c4_pass_count = 0  # ARM_A harm_disc > ARM_B + margin
    c5_pass_count = 0  # Manipulation checks

    for seed in SEEDS:
        a_slot = [r["slot_differentiation"] for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_A_NORMAL][0]
        b_slot = [r["slot_differentiation"] for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_B_REM_SUPP][0]
        c_slot = [r["slot_differentiation"] for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_C_SWS_SUPP][0]
        d_slot = [r["slot_differentiation"] for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_D_NO_SLEEP][0]

        a_harm = [r["harm_discrimination"] for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_A_NORMAL][0]
        b_harm = [r["harm_discrimination"] for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_B_REM_SUPP][0]
        d_harm = [r["harm_discrimination"] for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_D_NO_SLEEP][0]

        if a_slot > c_slot + C1_SLOT_DIFF_MARGIN:
            c1_pass_count += 1
        if a_slot > d_slot + C2_SLOT_DIFF_MARGIN:
            c2_pass_count += 1
        if a_harm > d_harm + C3_HARM_DISC_MARGIN:
            c3_pass_count += 1
        if a_harm > b_harm + C4_HARM_DISC_MARGIN:
            c4_pass_count += 1

        # Manipulation checks for this seed
        a_res = [r for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_A_NORMAL][0]
        b_res = [r for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_B_REM_SUPP][0]
        c_res = [r for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_C_SWS_SUPP][0]
        d_res = [r for r in all_results if r["seed"] == seed and r["arm_name"] == ARM_D_NO_SLEEP][0]

        manip_ok = (
            a_res["cumulative_sws_writes"] >= N_CYCLES
            and a_res["cumulative_rem_rollouts"] >= N_CYCLES
            and b_res["cumulative_sws_writes"] >= N_CYCLES
            and b_res["cumulative_rem_rollouts"] == 0
            and c_res["cumulative_sws_writes"] == 0
            and c_res["cumulative_rem_rollouts"] >= N_CYCLES
            and d_res["cumulative_sws_writes"] == 0
            and d_res["cumulative_rem_rollouts"] == 0
        )
        if manip_ok:
            c5_pass_count += 1

    c1_pass = c1_pass_count >= (n_seeds * PASS_FRACTION_REQUIRED)
    c2_pass = c2_pass_count >= (n_seeds * PASS_FRACTION_REQUIRED)
    c3_pass = c3_pass_count >= (n_seeds * PASS_FRACTION_REQUIRED)
    c4_pass = c4_pass_count >= (n_seeds * PASS_FRACTION_REQUIRED)
    c5_pass = c5_pass_count >= (n_seeds * PASS_FRACTION_REQUIRED)

    overall_pass = (c1_pass and c2_pass and c3_pass) or (c1_pass and c4_pass)

    outcome = "PASS" if overall_pass else "FAIL"
    evidence_direction = "supports" if overall_pass else "weakens"

    return {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "experimental",
        "summary": (
            f"V3-EXQ-670 tested INV-048's prediction that pharmacological sleep phase "
            f"disruption produces attribution pipeline degradation equivalent to "
            f"behavioral sleep deprivation. Four arms: normal sleep (both SWS+REM), "
            f"REM-suppressed (SWS only), SWS-suppressed (REM only), no sleep (neither). "
            f"Results: C1={c1_pass} (SWS necessary for slot differentiation), "
            f"C2={c2_pass} (sleep better than no sleep), C3={c3_pass} (full sleep "
            f"improves harm attribution), C4={c4_pass} (REM suppression impairs harm "
            f"attribution). Manipulation checks C5={c5_pass}. "
            f"Outcome: {outcome}."
        ),
        "run_id": f"v3_exq_670_inv048_pharm_sleep_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "seeds": list(SEEDS),
        "n_cycles": N_CYCLES,
        "n_prototypes_per_context": N_PROTOTYPES,
        "arm_configs": {
            ARM_A_NORMAL: {"sws_enabled": True, "rem_enabled": True},
            ARM_B_REM_SUPP: {"sws_enabled": True, "rem_enabled": False},
            ARM_C_SWS_SUPP: {"sws_enabled": False, "rem_enabled": True},
            ARM_D_NO_SLEEP: {"sws_enabled": False, "rem_enabled": False},
        },
        "arm_aggregates": arm_agg,
        "all_seed_results": all_results,
        "criteria": {
            "C1_slot_diff_A_vs_C": {"passed": c1_pass, "pass_count": c1_pass_count, "required": int(n_seeds * PASS_FRACTION_REQUIRED)},
            "C2_slot_diff_A_vs_D": {"passed": c2_pass, "pass_count": c2_pass_count, "required": int(n_seeds * PASS_FRACTION_REQUIRED)},
            "C3_harm_disc_A_vs_D": {"passed": c3_pass, "pass_count": c3_pass_count, "required": int(n_seeds * PASS_FRACTION_REQUIRED)},
            "C4_harm_disc_A_vs_B": {"passed": c4_pass, "pass_count": c4_pass_count, "required": int(n_seeds * PASS_FRACTION_REQUIRED)},
            "C5_manipulation_checks": {"passed": c5_pass, "pass_count": c5_pass_count, "required": int(n_seeds * PASS_FRACTION_REQUIRED)},
        },
        "thresholds": {
            "c1_slot_diff_margin": C1_SLOT_DIFF_MARGIN,
            "c2_slot_diff_margin": C2_SLOT_DIFF_MARGIN,
            "c3_harm_disc_margin": C3_HARM_DISC_MARGIN,
            "c4_harm_disc_margin": C4_HARM_DISC_MARGIN,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="V3-EXQ-670 INV-048 pharmacological sleep disruption equivalence")
    parser.add_argument("--dry-run", action="store_true", help="Run experiment without writing results")
    args = parser.parse_args()

    print(f"V3-EXQ-670: INV-048 pharmacological sleep disruption equivalence")
    print(f"Seeds: {SEEDS}, Cycles: {N_CYCLES}, Prototypes per context: {N_PROTOTYPES}")
    print(f"Running {'DRY RUN' if args.dry_run else 'FULL EXPERIMENT'}...")

    start_time = time.time()
    result = run_experiment(dry_run=args.dry_run)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Outcome: {result['outcome']}")
    print(f"Evidence direction: {result['evidence_direction']}")

    if args.dry_run:
        print("\nDRY RUN complete. No files written.")
        return

    # Write manifest
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    manifest_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=False,
        config=result.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"\nWrote manifest: {manifest_path}")

    # Emit outcome for runner
    emit_outcome(outcome=result["outcome"], manifest_path=str(manifest_path))


if __name__ == "__main__":
    main()
