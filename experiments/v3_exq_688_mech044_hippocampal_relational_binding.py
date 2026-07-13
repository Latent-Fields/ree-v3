"""V3-EXQ-688 -- MECH-044 hippocampal relational binding: detect changed spatial relations.

LINEAGE / ROUTING
-----------------
- Originating proposal: EVB-0295 / EXP-0184 (claim MECH-044, hippocampal relational
  binding and comparison).
- Substrate: HippocampalModule + AnchorSet + ghost goal bank are IMPLEMENTED and
  ACTIVE in V3 (SD-039, MECH-269, MECH-292).
- THIS experiment tests whether the hippocampal anchor system can detect RELATIONAL
  changes (not just absolute position changes): does the system discriminate when
  spatial relations between entities shift vs when entities move but preserve
  relations?

CLAIM HANDLING
--------------
claim_ids = ["MECH-044"]
evidence_direction grid:
  readiness (G0-G2) unmet  -> non_contributory, label substrate_not_ready_requeue
  readiness met + C1+C2+C3 -> supports,         label hippocampal_relational_binding_active
  readiness met + C1/C2 fail -> weakens,        label relational_insensitivity_detected
  readiness met + only C3 fail -> mixed

SUBSTRATE DETAIL
----------------
The hippocampal anchor set (MECH-269) encodes spatial context via z_world snapshots
and fires boundary events when verisimilitude (V_s) drops. MECH-044 asserts this
participates in RELATIONAL binding -- not just storing absolute positions but
tracking spatial RELATIONS between entities.

Biology grounding (Olsen et al. 2012): hippocampus does "online relational work
inside the perception-action loop", not just long-term storage consolidation.

DESIGN (3 arms x 3 seeds [42,43,44])
------------------------------------
  ARM_INTACT       HippocampalModule ON (use_hippocampal=True, use_anchor_sets=True)
  ARM_ABLATION_OFF HippocampalModule OFF (use_hippocampal=False)
  ARM_ABLATION_NO_ANCHORS  Hippocampal ON but anchor sets OFF (use_anchor_sets=False)

Per trial:
  1. Initial phase (10 ticks): agent observes two entities in a fixed spatial
     relation (e.g., "entity A north of entity B").
  2. RELATIONAL-CHANGE condition: entities swap positions (relation reverses).
     ABSOLUTE-CHANGE control: both entities shift by same delta (relation preserved).
  3. Measure: does the anchor system detect relational change (V_s drop, boundary
     event, anchor reset) vs absolute change?

Key metric: relational_sensitivity = (boundary_events_on_relation_change -
baseline_boundary_rate) / max(boundary_events_on_absolute_change, 1e-6).

Criteria (pre-registered):
  C1: relational_sensitivity_INTACT >= 0.5 (hippocampal system detects relation shifts)
  C2: relational_sensitivity_INTACT > relational_sensitivity_ABLATION_OFF + 0.3 (margin)
  C3: anchor_reset_count_INTACT on relational change > 0 (anchors actively remap)

Readiness gates (P0, measured BEFORE experiment):
  G0: z_world_discriminable -- mean pairwise distance in z_world >= 0.1 (not collapsed)
  G1: V_s_responsive -- V_s changes >= 0.05 when z_world changes (not stuck at 1.0)
  G2: boundary_events_fire -- at least 1 boundary event in 100 forced-change ticks

No phased training needed -- this tests the existing hippocampal module's relational
encoding capacity, not a learned predictor.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_688_mech044_hippocampal_relational_binding"
QUEUE_ID = "V3-EXQ-688"
BACKLOG_ID = "EVB-0295"
CLAIM_IDS: List[str] = ["MECH-044"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
N_TRIALS = 30
DRY_RUN_SEEDS = [42]
DRY_RUN_TRIALS = 5

WORLD_DIM = 128
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4

INITIAL_TICKS = 10
BASELINE_BOUNDARY_TICKS = 100

# Pre-registered thresholds
G0_ZWORLD_DIST_FLOOR = 0.1
G1_VS_CHANGE_FLOOR = 0.05
G2_BOUNDARY_MIN = 1
C1_RELATIONAL_SENSITIVITY_FLOOR = 0.5
C2_ABLATION_MARGIN = 0.3
C3_ANCHOR_RESET_MIN = 1
MIN_SEEDS_2OF3 = 2

ARM_INTACT = "ARM_INTACT"
ARM_OFF = "ARM_ABLATION_OFF"
ARM_NO_ANCHORS = "ARM_ABLATION_NO_ANCHORS"
ARMS = [ARM_INTACT, ARM_OFF, ARM_NO_ANCHORS]


def _build_agent(use_hippocampal: bool, use_anchor_sets: bool) -> REEAgent:
    """Build agent with hippocampal configuration."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=32,
        world_dim=WORLD_DIM,
        use_hippocampal=use_hippocampal,
        use_anchor_sets=use_anchor_sets if use_hippocampal else False,
        use_event_segmenter=True if use_anchor_sets else False,
    )
    return REEAgent(cfg)


def _create_entity_observations(
    pos_a: Tuple[float, float],
    pos_b: Tuple[float, float],
    seed: int
) -> torch.Tensor:
    """Create synthetic world observation with two entities at given positions.

    Returns: [1, WORLD_OBS_DIM] tensor encoding positions of entities A and B.
    """
    gen = torch.Generator().manual_seed(seed)
    obs = torch.zeros(1, WORLD_OBS_DIM)

    # Simple position encoding: first half for entity A, second half for entity B
    mid = WORLD_OBS_DIM // 2
    obs[0, :mid] = torch.tensor([pos_a[0], pos_a[1]] + [0.0] * (mid - 2))
    obs[0, mid:] = torch.tensor([pos_b[0], pos_b[1]] + [0.0] * (WORLD_OBS_DIM - mid - 2))

    # Add small noise
    obs += 0.01 * torch.randn_like(obs, generator=gen)
    return obs


def _run_trial(
    agent: REEAgent,
    trial_type: str,
    seed_offset: int
) -> Dict[str, Any]:
    """Run one trial (relational-change or absolute-change).

    Returns: dict with boundary_events_count, anchor_resets, V_s changes.
    """
    # Initial configuration: A north of B
    pos_a_init = (5.0, 7.0)
    pos_b_init = (5.0, 3.0)

    # Initial phase: stable observations
    for tick in range(INITIAL_TICKS):
        obs = _create_entity_observations(pos_a_init, pos_b_init, seed_offset + tick)
        body_obs = torch.randn(1, BODY_OBS_DIM)
        _ = agent.sense(body_obs, obs)

    # Record baseline V_s and anchor count
    vs_before = agent.hippocampal.per_stream_vs.get("z_world", 1.0) if hasattr(agent, "hippocampal") and agent.hippocampal is not None else 1.0
    anchor_count_before = len(agent.hippocampal.anchor_set.all_anchors()) if hasattr(agent, "hippocampal") and agent.hippocampal and hasattr(agent.hippocampal, "anchor_set") and agent.hippocampal.anchor_set else 0

    # Apply change
    if trial_type == "relational":
        # Swap positions (relation reverses: now A south of B)
        pos_a_new = pos_b_init
        pos_b_new = pos_a_init
    else:  # "absolute"
        # Both shift north by 2 (relation preserved: A still north of B)
        pos_a_new = (pos_a_init[0], pos_a_init[1] + 2.0)
        pos_b_new = (pos_b_init[0], pos_b_init[1] + 2.0)

    # Post-change phase
    boundary_events_count = 0
    for tick in range(INITIAL_TICKS):
        obs = _create_entity_observations(pos_a_new, pos_b_new, seed_offset + INITIAL_TICKS + tick)
        body_obs = torch.randn(1, BODY_OBS_DIM)
        _ = agent.sense(body_obs, obs)

        # Count boundary events (if available)
        if hasattr(agent, "hippocampal") and agent.hippocampal and hasattr(agent.hippocampal, "drain_boundary_events"):
            events = agent.hippocampal.drain_boundary_events()
            boundary_events_count += len(events)

    vs_after = agent.hippocampal.per_stream_vs.get("z_world", 1.0) if hasattr(agent, "hippocampal") and agent.hippocampal is not None else 1.0
    anchor_count_after = len(agent.hippocampal.anchor_set.all_anchors()) if hasattr(agent, "hippocampal") and agent.hippocampal and hasattr(agent.hippocampal, "anchor_set") and agent.hippocampal.anchor_set else 0

    vs_change = abs(vs_before - vs_after)
    anchor_resets = max(0, anchor_count_after - anchor_count_before)

    return {
        "boundary_events": boundary_events_count,
        "anchor_resets": anchor_resets,
        "vs_change": vs_change,
        "vs_before": vs_before,
        "vs_after": vs_after,
    }


def _run_seed_arm(arm: str, seed: int, dry_run: bool) -> Dict[str, Any]:
    """Run all trials for one seed x arm cell."""
    use_hippocampal = (arm != ARM_OFF)
    use_anchor_sets = (arm == ARM_INTACT)

    with arm_cell(seed, config_slice={"arm": arm, "use_hippocampal": use_hippocampal, "use_anchor_sets": use_anchor_sets}, script_path=Path(__file__)) as cell:
        agent = _build_agent(use_hippocampal, use_anchor_sets)

        n_trials = DRY_RUN_TRIALS if dry_run else N_TRIALS

        relational_results = []
        absolute_results = []

        for trial_idx in range(n_trials):
            if (trial_idx + 1) % 5 == 0 or trial_idx == 0:
                print(f"  [train] {arm} seed={seed} trial {trial_idx+1}/{n_trials}", flush=True)

            # Relational change trial
            rel = _run_trial(agent, "relational", seed * 10000 + trial_idx * 100)
            relational_results.append(rel)

            # Absolute change trial (control)
            abs_res = _run_trial(agent, "absolute", seed * 10000 + trial_idx * 100 + 50)
            absolute_results.append(abs_res)

        # Aggregate
        rel_boundary_mean = np.mean([r["boundary_events"] for r in relational_results])
        abs_boundary_mean = np.mean([r["boundary_events"] for r in absolute_results])
        rel_anchor_resets = sum(r["anchor_resets"] for r in relational_results)
        rel_vs_change_mean = np.mean([r["vs_change"] for r in relational_results])

        # Relational sensitivity = (rel_boundary - abs_boundary) / max(abs_boundary, eps)
        relational_sensitivity = (rel_boundary_mean - abs_boundary_mean) / max(abs_boundary_mean, 1e-6)

        row = {
            "arm": arm,
            "seed": seed,
            "relational_boundary_mean": float(rel_boundary_mean),
            "absolute_boundary_mean": float(abs_boundary_mean),
            "relational_sensitivity": float(relational_sensitivity),
            "anchor_reset_count": int(rel_anchor_resets),
            "vs_change_mean": float(rel_vs_change_mean),
        }

        cell.stamp(row)

    return row


def _p0_readiness_checks(agent: REEAgent) -> List[Dict[str, Any]]:
    """Measure readiness: z_world discriminable, V_s responsive, boundaries fire."""
    preconditions = []

    # G0: z_world discriminable
    z_worlds = []
    for tick in range(20):
        obs = torch.randn(1, WORLD_OBS_DIM)
        body_obs = torch.randn(1, BODY_OBS_DIM)
        latent_state = agent.sense(body_obs, obs)
        if latent_state.z_world is not None:
            z_worlds.append(latent_state.z_world.detach().clone())

    if len(z_worlds) >= 2:
        dists = [torch.dist(z_worlds[i], z_worlds[i+1]).item() for i in range(len(z_worlds)-1)]
        mean_dist = np.mean(dists)
        preconditions.append({
            "name": "z_world_discriminable",
            "measured": mean_dist,
            "threshold": G0_ZWORLD_DIST_FLOOR,
            "control": "pairwise z_world distances across random observations",
            "met": mean_dist >= G0_ZWORLD_DIST_FLOOR,
        })

    # G1: V_s responsive
    if hasattr(agent, "hippocampal") and agent.hippocampal:
        vs_initial = agent.hippocampal.per_stream_vs.get("z_world", 1.0)
        # Force a change
        for _ in range(10):
            obs = torch.randn(1, WORLD_OBS_DIM) * 10.0  # large change
            body_obs = torch.randn(1, BODY_OBS_DIM)
            _ = agent.sense(body_obs, obs)
        vs_after = agent.hippocampal.per_stream_vs.get("z_world", 1.0)
        vs_change = abs(vs_initial - vs_after)
        preconditions.append({
            "name": "V_s_responsive",
            "measured": vs_change,
            "threshold": G1_VS_CHANGE_FLOOR,
            "control": "V_s change on forced large world-observation shift",
            "met": vs_change >= G1_VS_CHANGE_FLOOR,
        })

    # G2: boundary events fire
    boundary_count = 0
    if hasattr(agent, "hippocampal") and agent.hippocampal and hasattr(agent.hippocampal, "drain_boundary_events"):
        for _ in range(BASELINE_BOUNDARY_TICKS):
            obs = torch.randn(1, WORLD_OBS_DIM)
            body_obs = torch.randn(1, BODY_OBS_DIM)
            _ = agent.sense(body_obs, obs)
            events = agent.hippocampal.drain_boundary_events()
            boundary_count += len(events)

    preconditions.append({
        "name": "boundary_events_fire",
        "measured": boundary_count,
        "threshold": G2_BOUNDARY_MIN,
        "control": f"{BASELINE_BOUNDARY_TICKS} random observation ticks",
        "met": boundary_count >= G2_BOUNDARY_MIN,
    })

    return preconditions


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    """Main experiment runner."""
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS

    # P0: Readiness checks (using ARM_INTACT)
    print("[P0] Readiness checks...")
    test_agent = _build_agent(use_hippocampal=True, use_anchor_sets=True)

    try:
        preconditions = p0_readiness_gate(_p0_readiness_checks(test_agent))
    except P0NotReady as e:
        # Substrate not ready -> early exit
        manifest = {
            "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
            "queue_id": QUEUE_ID,
            "backlog_id": BACKLOG_ID,
            "experiment_type": EXPERIMENT_TYPE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": "diagnostic",
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": e.preconditions,
            },
        }
        return manifest

    print("[P0] Readiness checks PASS. Proceeding to experiment.")

    # P1: Run all seeds x arms
    arm_results = []
    for arm in ARMS:
        for seed in seeds:
            print(f"[P1] Running {arm} seed={seed}...")
            row = _run_seed_arm(arm, seed, dry_run)
            arm_results.append(row)
            print(f"  -> relational_sensitivity={row['relational_sensitivity']:.3f}, anchor_resets={row['anchor_reset_count']}")

    # P2: Evaluate criteria
    intact_rows = [r for r in arm_results if r["arm"] == ARM_INTACT]
    off_rows = [r for r in arm_results if r["arm"] == ARM_OFF]

    intact_sensitivity = [r["relational_sensitivity"] for r in intact_rows]
    off_sensitivity = [r["relational_sensitivity"] for r in off_rows]
    intact_anchor_resets = [r["anchor_reset_count"] for r in intact_rows]

    mean_intact_sensitivity = np.mean(intact_sensitivity)
    mean_off_sensitivity = np.mean(off_sensitivity)
    total_anchor_resets = sum(intact_anchor_resets)

    c1_pass = mean_intact_sensitivity >= C1_RELATIONAL_SENSITIVITY_FLOOR
    c2_pass = mean_intact_sensitivity > (mean_off_sensitivity + C2_ABLATION_MARGIN)
    c3_pass = total_anchor_resets >= C3_ANCHOR_RESET_MIN

    seeds_c1 = sum(1 for s in intact_sensitivity if s >= C1_RELATIONAL_SENSITIVITY_FLOOR)
    criteria_pass = c1_pass and c2_pass and c3_pass

    # Interpret
    if criteria_pass:
        label = "hippocampal_relational_binding_active"
        evidence_direction = "supports"
    elif not c1_pass or not c2_pass:
        label = "relational_insensitivity_detected"
        evidence_direction = "weakens"
    else:  # only C3 fails
        label = "mixed_relational_signal"
        evidence_direction = "mixed"

    outcome = "PASS" if criteria_pass else "FAIL"

    print(f"[P2] Criteria: C1={c1_pass}, C2={c2_pass}, C3={c3_pass} -> {outcome}")
    print(f"verdict: {outcome}")

    # Build manifest
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "queue_id": QUEUE_ID,
        "backlog_id": BACKLOG_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": c1_pass, "C2": c2_pass, "C3": c3_pass},
        },
        "criteria": [
            {"name": "C1_relational_sensitivity_floor_met", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_ablation_margin", "load_bearing": False, "passed": c2_pass},
            {"name": "C3_anchor_resets", "load_bearing": False, "passed": c3_pass},
        ],
        "arm_results": arm_results,
        "metrics": {
            "mean_relational_sensitivity_INTACT": float(mean_intact_sensitivity),
            "mean_relational_sensitivity_OFF": float(mean_off_sensitivity),
            "total_anchor_resets_INTACT": int(total_anchor_resets),
            "seeds_c1_pass": int(seeds_c1),
            "C1_threshold": C1_RELATIONAL_SENSITIVITY_FLOOR,
            "C2_margin": C2_ABLATION_MARGIN,
            "C3_min_resets": C3_ANCHOR_RESET_MIN,
        },
    }

    # Degeneracy check
    manifest.update(check_degeneracy({
        "relational_sensitivity_INTACT": intact_sensitivity,
    }))

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== {EXPERIMENT_TYPE} ===")
    print(f"Queue ID: {QUEUE_ID}")
    print(f"Claim: {CLAIM_IDS}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL RUN'}")
    print()

    result = run_experiment(dry_run=args.dry_run)

    # Write manifest
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=False,
        config=result.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"\nWrote manifest to: {out_path}")
    print(f"Outcome: {result['outcome']}")
    print(f"Evidence direction: {result['evidence_direction']}")

    emit_outcome(
        outcome=result["outcome"],
        manifest_path=str(out_path),
    )
