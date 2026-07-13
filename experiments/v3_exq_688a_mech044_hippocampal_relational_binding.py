"""V3-EXQ-688a -- MECH-044 hippocampal relational binding: detect changed spatial relations.

LINEAGE / ROUTING
-----------------
- Supersedes V3-EXQ-688 (bug-fix to the SAME scientific question, MECH-044; new
  alphabetic suffix per the EXQ versioning rule). claim_ids = ["MECH-044"].
- Routed by the CONFIRMED failure autopsy
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-688_2026-06-18.{md,json}.
  V3-EXQ-688 (3 seeds, 2026-06-18) FAIL/non_contributory was a `precondition_unmet`
  VACUOUS self-route, NOT a substrate-readiness verdict -- MECH-044 was never put
  under test. Two harness-side defects:
    G1 `V_s_responsive` read `per_stream_vs.get("z_world", 1.0)` while the harness
       left `use_per_stream_vs=False`, forcing vs_change=0.0 BY CONSTRUCTION.
    G2 `boundary_events_fire` fed the MECH-288 segmenter i.i.d. white noise (no
       structured transition), so 0 boundary events fired.
  The MECH-269 per_stream_vs / MECH-288 EventSegmenter substrate is IMPLEMENTED and
  AVAILABLE in V3; 688 simply did not arm (G1) or stimulate (G2) it. NOT a substrate
  gap -> NO substrate_queue entry. The fix is entirely harness-side.

688a FIXES (full harness fix + non-vacuity guards)
--------------------------------------------------
1. ARM `use_per_stream_vs=True` in `_build_agent` so G1 measures the real MECH-269
   V_s signal instead of the constant default 1.0.
2. STRUCTURED-STIMULUS G2: drive the MECH-288 segmenter with a genuinely structured
   manipulation (sustained alternating step transitions on the experiment's own
   entity observations), not i.i.d. white noise. The measured count IS the
   boundary-event existence proof on the structured manipulation.
3. NON-VACUITY GUARD G3 (`z_world_relational_discriminative`): a discriminative-encoder
   readiness gate. z_world must separate the RELATION classes (A-north-of-B vs
   A-south-of-B) ACROSS many random absolute placements by at least the within-class
   scatter (Fisher-like ratio >= floor). A random-init encoder on synthetic obs is
   dominated by ABSOLUTE position (relation not specially represented) -> ratio < 1
   -> the run self-routes substrate_not_ready_requeue HONESTLY, never a spurious
   PASS/FAIL. This closes the "dimensionality necessary, not sufficient" gap (G0
   alone is the V3-EXQ-642 trap): an untrained agent + synthetic obs cannot fairly
   test relational binding even past G0/G1/G2, so the guard makes that condition
   self-route rather than weaken MECH-044.
4. DISPLACEMENT-MATCHED manipulations: relational change (swap A<->B, relation flips)
   and absolute change (translate both, relation preserved) now move each entity by
   the SAME distance, so any differential boundary response is attributable to the
   RELATION change, not motion magnitude (688's absolute=+2 vs relational-swap was a
   motion-magnitude confound).

CLAIM HANDLING / interpretation grid (only these paths)
-------------------------------------------------------
claim_ids = ["MECH-044"]
  readiness (G0-G3) unmet         -> non_contributory, substrate_not_ready_requeue
                                     (any precondition unmet -- NEVER a false weakens)
  readiness met + C1+C2+C3        -> supports,  hippocampal_relational_binding_active
  readiness met + C1/C2 fail      -> weakens,   relational_insensitivity_detected (honest)
  readiness met + only C3 fail    -> mixed,     mixed_relational_signal

SUBSTRATE DETAIL
----------------
The hippocampal anchor set (MECH-269) encodes spatial context via z_world snapshots
and fires boundary events (MECH-288) when verisimilitude (V_s) drops. MECH-044 asserts
this participates in RELATIONAL binding -- tracking spatial RELATIONS between entities,
not just absolute positions. Biology (Olsen et al. 2012): the hippocampus does "online
relational work inside the perception-action loop", not just long-term storage.

DESIGN (3 arms x 3 seeds [42,43,44])
------------------------------------
  ARM_INTACT       HippocampalModule ON (use_hippocampal=True, use_anchor_sets=True)
  ARM_ABLATION_OFF HippocampalModule OFF (use_hippocampal=False)
  ARM_ABLATION_NO_ANCHORS  Hippocampal ON but anchor sets OFF (use_anchor_sets=False)

Per trial (displacement-matched):
  1. Initial phase (10 ticks): two entities in a fixed spatial relation (A north of B).
  2. RELATIONAL-CHANGE condition: swap A<->B (relation reverses).
     ABSOLUTE-CHANGE control: translate both by the same vector (relation preserved),
     matched per-entity displacement.
  3. Measure: does the anchor system detect relational change (boundary events,
     anchor reset, V_s drop) vs absolute change?

Key metric: relational_sensitivity = (boundary_events_on_relation_change -
absolute_boundary_rate) / max(boundary_events_on_absolute_change, 1e-6).

Criteria (pre-registered):
  C1 (load-bearing): relational_sensitivity_INTACT >= 0.5
  C2: relational_sensitivity_INTACT > relational_sensitivity_ABLATION_OFF + 0.3
  C3: anchor_reset_count_INTACT on relational change > 0

Readiness gates (P0, measured BEFORE the experiment body on ARM_INTACT):
  G0: z_world_discriminable -- mean pairwise z_world distance >= 0.1 (not collapsed)
  G1: V_s_responsive -- per_stream_vs change >= 0.05 on a forced shift (ARMED)
  G2: boundary_events_fire -- >= 1 boundary event under a STRUCTURED step manipulation
  G3: z_world_relational_discriminative -- relation-class Fisher ratio >= floor
      (the discriminative-encoder non-vacuity guard)

No phased training -- this tests the existing hippocampal module's relational encoding
capacity. The G3 guard makes the random-init-encoder regime self-route honestly; a
fair PASS therefore requires a discriminative encoder (a future trained-encoder
successor is the carry if G3 cannot clear at the synthetic-obs scale).
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

EXPERIMENT_TYPE = "v3_exq_688a_mech044_hippocampal_relational_binding"
QUEUE_ID = "V3-EXQ-688a"
SUPERSEDES = "V3-EXQ-688"
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
BASELINE_BOUNDARY_TICKS = 240  # structured-stimulus probe length (>= pe_window for a stable z-baseline)
G2_BLOCK_TICKS = 40            # sustained-departure block length (segmenter needs SUSTAINED, not brief, shifts)
G2_SPIKE_MAGNITUDE = 30.0      # canonical 10x+ sustained spike (MECH-288 contract: "fires on 10x sustained spike")
G2_SPIKE_DIMS = 5              # concentrate the departure (a clean structured transition, not full-dim saturation)
G2_BASELINE_VALUE = 1.0

# Pre-registered thresholds
G0_ZWORLD_DIST_FLOOR = 0.1
G1_VS_CHANGE_FLOOR = 0.05
G2_BOUNDARY_MIN = 1
# G3: z_world must represent the RELATION at least as strongly as absolute position
# (between-relation-class distance >= within-class scatter). A random-init encoder,
# dominated by absolute position under wide placement variation, falls below 1.0.
G3_REL_DISCRIM_FLOOR = 1.0
G3_N_PER_CLASS = 12
G3_SEPARATION = 2.0

C1_RELATIONAL_SENSITIVITY_FLOOR = 0.5
C2_ABLATION_MARGIN = 0.3
C3_ANCHOR_RESET_MIN = 1
MIN_SEEDS_2OF3 = 2

# Displacement-matched manipulation geometry (each entity moves DISP units).
ENTITY_X = 5.0
ENTITY_B_Y = 4.0
ENTITY_SEP = 4.0          # A starts ENTITY_SEP above B
DISP = ENTITY_SEP          # absolute translation == swap displacement (matched)

ARM_INTACT = "ARM_INTACT"
ARM_OFF = "ARM_ABLATION_OFF"
ARM_NO_ANCHORS = "ARM_ABLATION_NO_ANCHORS"
ARMS = [ARM_INTACT, ARM_OFF, ARM_NO_ANCHORS]


def _build_agent(use_hippocampal: bool, use_anchor_sets: bool) -> REEAgent:
    """Build agent with hippocampal configuration.

    688a FIX (G1): arm use_per_stream_vs whenever the hippocampal module is on, so the
    MECH-269 V_s observable is populated by sense() instead of returning the constant
    default 1.0 (the 688 G1 vacuity).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=32,
        world_dim=WORLD_DIM,
        use_hippocampal=use_hippocampal,
        use_per_stream_vs=use_hippocampal,
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
    gen = torch.Generator().manual_seed(int(seed))
    obs = torch.zeros(1, WORLD_OBS_DIM)

    # Simple position encoding: first half for entity A, second half for entity B
    mid = WORLD_OBS_DIM // 2
    obs[0, :mid] = torch.tensor([pos_a[0], pos_a[1]] + [0.0] * (mid - 2))
    obs[0, mid:] = torch.tensor([pos_b[0], pos_b[1]] + [0.0] * (WORLD_OBS_DIM - mid - 2))

    # Add small noise (seeded; torch.randn_like does NOT accept a generator kwarg)
    obs += 0.01 * torch.randn(*obs.shape, generator=gen)
    return obs


def _manipulation_positions(trial_type: str) -> Tuple[Tuple[float, float], Tuple[float, float],
                                                      Tuple[float, float], Tuple[float, float]]:
    """Displacement-matched initial / post positions for a manipulation.

    Returns (pos_a_init, pos_b_init, pos_a_new, pos_b_new). Each entity moves DISP units
    in BOTH manipulations, so a differential boundary response is attributable to the
    relation change, not motion magnitude.
    """
    pos_a_init = (ENTITY_X, ENTITY_B_Y + ENTITY_SEP)   # A north of B
    pos_b_init = (ENTITY_X, ENTITY_B_Y)
    if trial_type == "relational":
        # Swap: A <-> B. A moves -SEP, B moves +SEP (relation reverses: A now south of B).
        pos_a_new = pos_b_init
        pos_b_new = pos_a_init
    else:  # "absolute"
        # Translate both by -DISP (relation preserved: A still north of B). Matched move.
        pos_a_new = (pos_a_init[0], pos_a_init[1] - DISP)
        pos_b_new = (pos_b_init[0], pos_b_init[1] - DISP)
    return pos_a_init, pos_b_init, pos_a_new, pos_b_new


def _run_trial(
    agent: REEAgent,
    trial_type: str,
    seed_offset: int
) -> Dict[str, Any]:
    """Run one trial (relational-change or absolute-change), displacement-matched.

    Returns: dict with boundary_events_count, anchor_resets, V_s changes.
    """
    pos_a_init, pos_b_init, pos_a_new, pos_b_new = _manipulation_positions(trial_type)

    # Initial phase: stable observations
    for tick in range(INITIAL_TICKS):
        obs = _create_entity_observations(pos_a_init, pos_b_init, seed_offset + tick)
        body_obs = torch.randn(1, BODY_OBS_DIM)
        _ = agent.sense(body_obs, obs)

    # Record baseline V_s and anchor count
    vs_before = agent.hippocampal.per_stream_vs.get("z_world", 1.0) if hasattr(agent, "hippocampal") and agent.hippocampal is not None else 1.0
    anchor_count_before = len(agent.hippocampal.anchor_set.all_anchors()) if hasattr(agent, "hippocampal") and agent.hippocampal and hasattr(agent.hippocampal, "anchor_set") and agent.hippocampal.anchor_set else 0

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


def _g2_structured_boundary_probe(agent: REEAgent) -> int:
    """G2 FIX: drive the MECH-288 segmenter with a STRUCTURED sustained step manipulation
    and count boundary events (the existence proof). Alternates a near-zero baseline with
    a canonical 10x+ sustained spike (the segmenter's own contract regime: "silent on
    constant baseline, fires on 10x sustained spike") every G2_BLOCK_TICKS ticks. Each
    block transition is a SUSTAINED structured z-scored departure the fast PE-threshold
    detector can fire on, over a window long enough to stabilise the z-baseline. Replaces
    688's i.i.d. white-noise probe, which can NEVER fire by construction (no structured
    transition). The returned count is the boundary-event existence proof; it verifies the
    segmenter mechanism is live and stimulated, not rigged-to-fail like 688's white noise.
    """
    if not (hasattr(agent, "hippocampal") and agent.hippocampal and hasattr(agent.hippocampal, "drain_boundary_events")):
        return 0

    baseline = torch.zeros(1, WORLD_OBS_DIM)
    baseline[0, :G2_SPIKE_DIMS] = G2_BASELINE_VALUE
    spike = torch.zeros(1, WORLD_OBS_DIM)
    spike[0, :G2_SPIKE_DIMS] = G2_SPIKE_MAGNITUDE

    boundary_count = 0
    for tick in range(BASELINE_BOUNDARY_TICKS):
        block = (tick // G2_BLOCK_TICKS) % 2  # 0 -> baseline, 1 -> sustained spike
        obs = (baseline if block == 0 else spike) + 0.01 * torch.randn(1, WORLD_OBS_DIM)
        body_obs = torch.randn(1, BODY_OBS_DIM)
        _ = agent.sense(body_obs, obs)
        events = agent.hippocampal.drain_boundary_events()
        boundary_count += len(events)
    return boundary_count


def _g3_relational_discriminability(agent: REEAgent, rng: np.random.Generator) -> float:
    """G3 non-vacuity guard: relation-class Fisher-like discriminability of z_world.

    Generates G3_N_PER_CLASS configs in each relation class (A-north-of-B and
    A-south-of-B) at RANDOM absolute placements, encodes each to z_world, and returns
    between-class-distance / within-class-scatter. A discriminative (trained) encoder
    represents the RELATION saliently -> ratio >> 1. A random-init encoder is dominated
    by absolute position (large within-class scatter from placement variation, small
    between-class relation signal) -> ratio < 1 -> the run self-routes
    substrate_not_ready_requeue honestly.
    """
    sep = G3_SEPARATION
    z_north: List[torch.Tensor] = []
    z_south: List[torch.Tensor] = []
    for _ in range(G3_N_PER_CLASS):
        cx = float(rng.uniform(1.5, 10.5))
        cy = float(rng.uniform(2.0, 10.0))
        s_north = int(rng.integers(0, 2 ** 31 - 1))
        s_south = int(rng.integers(0, 2 ** 31 - 1))
        # north: A above B at this center
        obs_n = _create_entity_observations((cx, cy + sep / 2.0), (cx, cy - sep / 2.0), s_north)
        ls_n = agent.sense(torch.randn(1, BODY_OBS_DIM), obs_n)
        # south: A below B at the SAME center (relation flipped, absolute placement matched)
        obs_s = _create_entity_observations((cx, cy - sep / 2.0), (cx, cy + sep / 2.0), s_south)
        ls_s = agent.sense(torch.randn(1, BODY_OBS_DIM), obs_s)
        if ls_n.z_world is None or ls_s.z_world is None:
            continue
        z_north.append(ls_n.z_world.detach().flatten())
        z_south.append(ls_s.z_world.detach().flatten())

    if len(z_north) < 2 or len(z_south) < 2:
        return 0.0

    north = torch.stack(z_north)
    south = torch.stack(z_south)
    between = float((north.mean(dim=0) - south.mean(dim=0)).norm().item())
    within = 0.5 * (float(north.std(dim=0).norm().item()) + float(south.std(dim=0).norm().item()))
    return between / (within + 1e-6)


def _p0_readiness_checks(rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Measure readiness: z_world discriminable, V_s responsive (ARMED), boundaries
    fire on a STRUCTURED stimulus, and z_world is relation-discriminative (G3 guard).

    Each state-dependent sub-probe (G2 segmenter, G3 encoder) runs on a FRESH agent so
    one probe's inputs do not pollute another's substrate state (G1's 10x forced shifts
    inflate the segmenter z-baseline; running G2 on the same agent suppresses its firing
    -- an order-dependence artifact, not a substrate signal). Fresh agents are
    representative: every P1 cell builds its own fresh agent too.
    """
    preconditions = []

    # G0 + G1 on a fresh agent.
    agent = _build_agent(use_hippocampal=True, use_anchor_sets=True)

    # G0: z_world discriminable (necessary -- not collapsed)
    z_worlds = []
    for tick in range(20):
        obs = torch.randn(1, WORLD_OBS_DIM)
        body_obs = torch.randn(1, BODY_OBS_DIM)
        latent_state = agent.sense(body_obs, obs)
        if latent_state.z_world is not None:
            z_worlds.append(latent_state.z_world.detach().clone())

    if len(z_worlds) >= 2:
        dists = [torch.dist(z_worlds[i], z_worlds[i + 1]).item() for i in range(len(z_worlds) - 1)]
        mean_dist = float(np.mean(dists))
        preconditions.append({
            "name": "z_world_discriminable",
            "measured": mean_dist,
            "threshold": G0_ZWORLD_DIST_FLOOR,
            "control": "pairwise z_world distances across random observations",
            "met": mean_dist >= G0_ZWORLD_DIST_FLOOR,
        })

    # G1: V_s responsive (ARMED -- use_per_stream_vs=True so this is a real measurement)
    if hasattr(agent, "hippocampal") and agent.hippocampal:
        vs_initial = agent.hippocampal.per_stream_vs.get("z_world", 1.0)
        for _ in range(10):
            obs = torch.randn(1, WORLD_OBS_DIM) * 10.0  # large forced shift
            body_obs = torch.randn(1, BODY_OBS_DIM)
            _ = agent.sense(body_obs, obs)
        vs_after = agent.hippocampal.per_stream_vs.get("z_world", 1.0)
        vs_change = abs(vs_initial - vs_after)
        preconditions.append({
            "name": "V_s_responsive",
            "measured": vs_change,
            "threshold": G1_VS_CHANGE_FLOOR,
            "control": "per_stream_vs change on a forced large world-observation shift (armed)",
            "met": vs_change >= G1_VS_CHANGE_FLOOR,
        })

    # G2: boundary events fire under a STRUCTURED step manipulation (existence proof).
    # Fresh agent -> clean segmenter z-baseline (not polluted by G1's forced shifts).
    agent_g2 = _build_agent(use_hippocampal=True, use_anchor_sets=True)
    boundary_count = _g2_structured_boundary_probe(agent_g2)
    preconditions.append({
        "name": "boundary_events_fire",
        "measured": boundary_count,
        "threshold": G2_BOUNDARY_MIN,
        "control": f"{BASELINE_BOUNDARY_TICKS} ticks of alternating baseline / 10x+ sustained spike (fresh agent)",
        "met": boundary_count >= G2_BOUNDARY_MIN,
    })

    # G3: z_world relation-discriminative (non-vacuity guard -- random encoder fails).
    # Fresh agent -> clean encoder/segmenter state.
    agent_g3 = _build_agent(use_hippocampal=True, use_anchor_sets=True)
    rel_discrim = _g3_relational_discriminability(agent_g3, rng)
    preconditions.append({
        "name": "z_world_relational_discriminative",
        "measured": rel_discrim,
        "threshold": G3_REL_DISCRIM_FLOOR,
        "control": "relation-class Fisher ratio (between A-north/A-south) / within-class scatter on a positive control",
        "met": rel_discrim >= G3_REL_DISCRIM_FLOOR,
    })

    return preconditions


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    """Main experiment runner."""
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS

    # P0: Readiness checks (using ARM_INTACT)
    print("[P0] Readiness checks (G0 discriminable / G1 V_s armed / G2 structured boundaries / G3 relational-discriminative)...")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    try:
        preconditions = p0_readiness_gate(_p0_readiness_checks(rng))
    except P0NotReady as e:
        # Substrate not ready / non-vacuity guard tripped -> honest early exit.
        print(f"[P0] Readiness UNMET: {e.reason} -> substrate_not_ready_requeue", flush=True)
        for pc in e.preconditions:
            print(f"      {pc['name']}: measured={pc['measured']:.4g} threshold={pc['threshold']:.4g} met={pc['met']}", flush=True)
        print("verdict: FAIL", flush=True)
        manifest = {
            "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
            "queue_id": QUEUE_ID,
            "supersedes": SUPERSEDES,
            "backlog_id": BACKLOG_ID,
            "experiment_type": EXPERIMENT_TYPE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": EXPERIMENT_PURPOSE,
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

    mean_intact_sensitivity = float(np.mean(intact_sensitivity))
    mean_off_sensitivity = float(np.mean(off_sensitivity))
    total_anchor_resets = int(sum(intact_anchor_resets))

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
        "supersedes": SUPERSEDES,
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
            "mean_relational_sensitivity_INTACT": mean_intact_sensitivity,
            "mean_relational_sensitivity_OFF": mean_off_sensitivity,
            "total_anchor_resets_INTACT": total_anchor_resets,
            "seeds_c1_pass": int(seeds_c1),
            "C1_threshold": C1_RELATIONAL_SENSITIVITY_FLOOR,
            "C2_margin": C2_ABLATION_MARGIN,
            "C3_min_resets": C3_ANCHOR_RESET_MIN,
        },
    }

    # Degeneracy check on the load-bearing metric (extra non-vacuity net for evidence runs)
    manifest.update(check_degeneracy({
        "relational_sensitivity_INTACT": intact_sensitivity,
    }))

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== {EXPERIMENT_TYPE} ===")
    print(f"Queue ID: {QUEUE_ID} (supersedes {SUPERSEDES})")
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

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
    )
