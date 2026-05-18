"""
V3-EXQ-586: infant_substrate:GAP-9 curriculum scheduler validation.

Diagnostic substrate-readiness test for the InfantCurriculumScheduler
(experiments/infant_curriculum.py).

Two arms, 3 seeds each:
  ARM_0 (hard-count only): no telemetry provided to update(); transitions
        governed solely by episode-count minimums.
  ARM_1 (telemetry-gated): synthetic telemetry passed to update() that
        deliberately crosses thresholds AFTER the hard-count minimum,
        verifying that the scheduler delays transitions until both the
        episode-count and the telemetry gate are satisfied.

        Synthetic telemetry schedule for ARM_1:
          ep 0-149:   h_pos=0.0 (below threshold) -> phase 0 held past ep 100
          ep 150+:    h_pos=above threshold        -> phase 0->1 at ep 150
          ep 500-649: z_goal_norm=0.0 (below)     -> phase 1 held past ep 500
          ep 650+:    z_goal_norm=above threshold  -> phase 1->2 at ep 650
          ep 2000-:   residue_coverage_pct above   -> phase 2->3 at ep 2000
        Expected ARM_1 transitions: (150, 1), (650, 2), (2000, 3).

Acceptance criteria:
  C0: ARM_0 seeds all produce exactly 3 phase transitions at episodes
      100, 500, 2000.
  C1: ARM_1 seeds all produce transitions at (150, 1), (650, 2), (2000, 3);
      first transition at ep 150 (delayed from hard count 100 by H_pos gate).
  C2: For each phase 0-2, env_kwargs() produces a valid CausalGridWorldV2
      (no constructor exception).  The resulting env steps 10 ticks without
      error.
  C3: Phase 1 env_kwargs contain harm_gradient_enabled=True;
      Phase 2+ contain both harm_gradient_enabled=True AND
      microhabitat_enabled=True.
  C4: config_overrides keys residue_scale_factor and
      offline_integration_frequency are strictly increasing across phases 0-3.

PASS = C0 AND C1 AND C2 AND C3 AND C4 across all seeds.

experiment_purpose: diagnostic (scheduler harness-helper validation; no
  governance weighting).
claim_ids: [] (harness helper, not a claim-bearing substrate component).
Unblocks: infant_substrate:GAP-9, DEV-NEED-008, ARC-046.

Interpretation grid:
  Outcome                                 | Diagnosis
  ----------------------------------------|---------------------------------------
  C0 FAIL (wrong transition episodes)     | InfantCurriculumScheduler PHASE_EP_MIN
                                          | mismatch; check constants in
                                          | experiments/infant_curriculum.py
  C1 FAIL (ARM_1 transitions wrong)       | Telemetry gating logic broken in
                                          | _try_phase_0_to_1 / _try_phase_1_to_2;
                                          | check h_pos and z_goal_norm gate paths
                                          | in infant_curriculum.py
  C2 FAIL (env constructor error)         | env_kwargs() returns invalid kwarg for
                                          | the current CausalGridWorldV2 API;
                                          | check that GAP-1/2/3 env kwargs still
                                          | match the constructor signature
  C3 FAIL (wrong feature flags)           | env_kwargs() phase->feature mapping
                                          | inconsistent with design spec
  C4 FAIL (config_overrides not ordered)  | config_overrides() table wrong; check
                                          | residue_scale_factor / offline_freq
                                          | values in infant_curriculum.py
  All PASS                                | GAP-9 scheduler validated;
                                          | EXQ-ISEF-005 (GAP-14) can now use
                                          | InfantCurriculumScheduler
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from infant_curriculum import (  # noqa: E402
    InfantCurriculumScheduler,
    H_POS_FRAC_OF_MAX,
)
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_586_gap9_curriculum_validation"
QUEUE_ID = "V3-EXQ-586"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [0, 1, 2]
SIZE = 12
NUM_HAZARDS = 2
NUM_RESOURCES = 2
# Episode ranges for each arm.
ARM_0_EPISODES = 2050   # covers all 3 hard-count boundaries (100, 500, 2000)
ARM_1_EPISODES = 2050   # covers delayed transitions (150, 650, 2000)

# ARM_1 synthetic telemetry schedule.
# H_pos threshold = 0.70 * ln(144) ~ 3.478.  We use 4.0 (above) / 0.0 (below).
# Transition plan: phase 0->1 at ep 150, phase 1->2 at ep 650.
ARM1_H_POS_HIGH = 4.0   # above the 0.70*ln(144) threshold
ARM1_H_POS_LOW = 0.0    # below threshold
ARM1_Z_GOAL_HIGH = 0.40  # above Z_GOAL_THRESHOLD=0.30
ARM1_Z_GOAL_LOW = 0.10   # below threshold
# Residue: pass None for 2->3 so hard count (ep 2000) governs.

EXPECTED_ARM1_TRANSITIONS = [(150, 1), (650, 2), (2000, 3)]


# ------------------------------------------------------------------
# ARM_0: hard-count transitions only
# ------------------------------------------------------------------

def run_arm_0(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 1000)
    sched = InfantCurriculumScheduler(grid_size=SIZE)
    transitions: List[tuple] = []

    for ep in range(ARM_0_EPISODES):
        # No telemetry passed -> hard count governs.
        sched.update(ep)
        if sched.phase_changed:
            transitions.append((ep, sched.current_phase))

    # C0: exactly transitions at ep 100->1, 500->2, 2000->3.
    expected = [(100, 1), (500, 2), (2000, 3)]
    c0_ok = transitions == expected

    print(
        f"  ARM_0 seed={seed}: transitions={transitions}"
        f" c0_ok={c0_ok}",
        flush=True,
    )
    return {
        "passed": c0_ok,
        "c0_ok": c0_ok,
        "transitions": transitions,
    }


# ------------------------------------------------------------------
# ARM_1: telemetry-gated transitions (synthetic telemetry)
# ------------------------------------------------------------------

def _arm1_telemetry(ep: int):
    """
    Return (h_pos, z_goal_norm, residue_coverage_pct) for episode ep.

    Designed to produce delayed transitions:
      phase 0->1 at ep 150 (not 100) -- h_pos below threshold until ep 150
      phase 1->2 at ep 650 (not 500) -- z_goal_norm below threshold until 650
      phase 2->3 at ep 2000 (hard)   -- residue_coverage_pct = None
    """
    h_pos = ARM1_H_POS_HIGH if ep >= 150 else ARM1_H_POS_LOW
    z_goal_norm = ARM1_Z_GOAL_HIGH if ep >= 650 else ARM1_Z_GOAL_LOW
    residue_coverage_pct = None  # rely on hard count for 2->3
    return h_pos, z_goal_norm, residue_coverage_pct


def run_arm_1(seed: int) -> Dict[str, Any]:
    """
    Synthetic-telemetry arm: pass phase-appropriate telemetry each episode
    to verify that the scheduler correctly delays transitions until BOTH
    the episode-count minimum AND the telemetry threshold are satisfied.
    """
    sched = InfantCurriculumScheduler(grid_size=SIZE)
    transitions: List[tuple] = []

    for ep in range(ARM_1_EPISODES):
        h_pos, z_goal_norm, residue_coverage_pct = _arm1_telemetry(ep)
        sched.update(
            ep,
            h_pos=h_pos,
            z_goal_norm=z_goal_norm,
            residue_coverage_pct=residue_coverage_pct,
        )
        if sched.phase_changed:
            transitions.append((ep, sched.current_phase))

    # C1: transitions must match the expected delayed schedule.
    c1_ok = transitions == EXPECTED_ARM1_TRANSITIONS

    print(
        f"  ARM_1 seed={seed}: transitions={transitions}"
        f" expected={EXPECTED_ARM1_TRANSITIONS} c1_ok={c1_ok}",
        flush=True,
    )
    return {
        "passed": c1_ok,
        "c1_ok": c1_ok,
        "transitions": transitions,
        "expected_transitions": EXPECTED_ARM1_TRANSITIONS,
    }


# ------------------------------------------------------------------
# C2 + C3: env constructor validation per phase
# ------------------------------------------------------------------

def check_env_per_phase() -> Dict[str, Any]:
    """For phases 0-2, build a real env and run 10 steps."""
    sched = InfantCurriculumScheduler(grid_size=SIZE)
    c2_ok = True
    c3_ok = True
    details: Dict[str, Any] = {}

    for phase in range(3):
        kw = sched.env_kwargs(phase=phase)
        try:
            env = CausalGridWorldV2(
                size=SIZE,
                seed=42,
                num_hazards=NUM_HAZARDS,
                num_resources=NUM_RESOURCES,
                **kw,
            )
            env.reset()
            rng = np.random.default_rng(0)
            for _ in range(10):
                _, _, done, _, _ = env.step(int(rng.integers(0, 5)))
                if done:
                    env.reset()
        except Exception as exc:
            c2_ok = False
            details[f"phase{phase}_error"] = str(exc)
            continue

        # C3 checks.
        if phase == 1:
            if not kw.get("harm_gradient_enabled", False):
                c3_ok = False
                details["phase1_harm_gradient_missing"] = True
        if phase == 2:
            if not kw.get("harm_gradient_enabled", False):
                c3_ok = False
                details["phase2_harm_gradient_missing"] = True
            if not kw.get("microhabitat_enabled", False):
                c3_ok = False
                details["phase2_microhabitat_missing"] = True

    print(
        f"  check_env_per_phase: c2_ok={c2_ok} c3_ok={c3_ok} details={details}",
        flush=True,
    )
    return {"c2_ok": c2_ok, "c3_ok": c3_ok, "details": details}


# ------------------------------------------------------------------
# C4: config_overrides ordering
# ------------------------------------------------------------------

def check_config_overrides_order() -> bool:
    sched = InfantCurriculumScheduler()
    ovs = [sched.config_overrides(phase=p) for p in range(4)]
    rsf = [ov["residue_scale_factor"] for ov in ovs]
    freq = [ov["offline_integration_frequency"] for ov in ovs]
    rsf_ok = all(rsf[i] < rsf[i + 1] for i in range(3))
    freq_ok = all(freq[i] < freq[i + 1] for i in range(3))
    c4_ok = rsf_ok and freq_ok
    print(
        f"  C4 config_overrides: rsf={rsf} freq={freq}"
        f" rsf_ok={rsf_ok} freq_ok={freq_ok}",
        flush=True,
    )
    return c4_ok


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_experiment():
    arm0_results: Dict[int, Dict] = {}
    arm1_results: Dict[int, Dict] = {}

    for seed in SEEDS:
        print(f"Seed {seed} Condition ARM_0", flush=True)
        arm0_results[seed] = run_arm_0(seed)
        print(f"Seed {seed} Condition ARM_1", flush=True)
        arm1_results[seed] = run_arm_1(seed)

    print("Checking env constructor per phase (C2+C3)...", flush=True)
    env_check = check_env_per_phase()

    print("Checking config_overrides ordering (C4)...", flush=True)
    c4_ok = check_config_overrides_order()

    c0_all = all(r["c0_ok"] for r in arm0_results.values())
    c1_all = all(r["c1_ok"] for r in arm1_results.values())
    c2_ok = env_check["c2_ok"]
    c3_ok = env_check["c3_ok"]

    all_pass = c0_all and c1_all and c2_ok and c3_ok and c4_ok
    outcome = "PASS" if all_pass else "FAIL"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "timestamp_utc": ts,
        "seeds": SEEDS,
        "arm0_episodes": ARM_0_EPISODES,
        "arm1_episodes": ARM_1_EPISODES,
        "c0_hard_count_transitions": c0_all,
        "c1_telemetry_transitions": c1_all,
        "c2_env_constructor_ok": c2_ok,
        "c3_feature_flags_ok": c3_ok,
        "c4_config_ordering_ok": c4_ok,
        "env_check_details": env_check["details"],
        "expected_arm1_transitions": EXPECTED_ARM1_TRANSITIONS,
        "arm1_h_pos_high": ARM1_H_POS_HIGH,
        "arm1_z_goal_high": ARM1_Z_GOAL_HIGH,
        "h_pos_frac_of_max": H_POS_FRAC_OF_MAX,
        "per_seed_arm0": {str(s): r for s, r in arm0_results.items()},
        "per_seed_arm1": {str(s): r for s, r in arm1_results.items()},
    }

    return manifest, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest, run_id = run_experiment()

    if args.dry_run:
        print("DRY RUN -- manifest preview:", flush=True)
        print(
            json.dumps(
                {k: v for k, v in manifest.items() if k not in ("per_seed_arm0", "per_seed_arm1")},
                indent=2,
            )
        )
        sys.exit(0 if manifest["outcome"] == "PASS" else 1)

    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "REE_assembly", "evidence", "experiments"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {out_path}", flush=True)
    print(f"Outcome: {manifest['outcome']}", flush=True)
    print(f"  C0 hard-count transitions:   {manifest['c0_hard_count_transitions']}", flush=True)
    print(f"  C1 telemetry transitions:    {manifest['c1_telemetry_transitions']}", flush=True)
    print(f"  C2 env constructor OK:       {manifest['c2_env_constructor_ok']}", flush=True)
    print(f"  C3 feature flags correct:    {manifest['c3_feature_flags_ok']}", flush=True)
    print(f"  C4 config overrides ordered: {manifest['c4_config_ordering_ok']}", flush=True)

    _outcome = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
