"""
V3-EXQ-636: SD-057 object-bound incentive-salience MECHANISM diagnostic (GAP-7 L2-L3-L4).

Scientific question: does the SD-057 incentive token bank make the WANTING target
(z_goal -> most-wanted object) dissociable from the LIKING target (the object just
contacted)? This is the structural prerequisite for the L9 wanting!=liking
dissociation (514l C6 = 0.0 on all seeds), which the legacy single-attractor goal
stream made impossible (wanting target == liking target by construction).

DECOUPLED FROM goal_pipeline:GAP-2 (foraging competence), mirroring how V3-EXQ-626b
decoupled the L1 positive control. Contacts are FORCED (the experiment calls
agent.update_z_goal(benefit, drive, resource_type=k) directly with a chosen type
tag and sets the per-axis drive state), so the dissociation does not depend on the
agent actually navigating to and consuming two resource types -- that ecological,
embedding-distinct behavioural L9 validation remains GAP-2-gated. This experiment
validates ONLY the L2-L4 MECHANISM: bind benefit to object identity (L2), accrue a
per-object drive-revaluable token (L3), and seed z_goal FROM the most-wanted
object's pointer (L4).

Protocol (per seed x arm):
  BUILD phase  -- forced contacts that lay down tokens for BOTH resource types:
                  feed WATER (type 2) under thirst, feed FOOD (type 1) under hunger.
  MEASURE phase -- forced contacts of FOOD (type 1) while the agent is THIRSTY
                  (per-axis drive favours water/type 2). With the bank ON, the
                  most-wanted type (argmax over base_value x per-axis-drive) should
                  be WATER (type 2) -- DIFFERENT from the just-contacted FOOD (type 1)
                  -- so wanting_target != liking_target. With the bank OFF there is
                  no per-object pointer (single attractor), so no such event exists.

Arms (2 arms x 3 seeds):
  ARM_BANK_ON   use_incentive_token_bank=True  -- the SD-057 object-bound layer.
  ARM_BANK_OFF  use_incentive_token_bank=False -- legacy single-attractor seeding
                (bit-identical pre-SD-057). resource_type is still passed but
                ignored (bank is None).
Both arms: cfg.latent.use_resource_encoder=True (set directly), z_goal_enabled=True,
drive_weight=2.0, SD-049 multi_resource_heterogeneity + per_axis_drive enabled.

Pre-registered acceptance criteria:
  C1 (L2/L3 bind):     ARM_BANK_ON binds >= 2 distinct resource types
                       (len(incentive_bank tokens) >= 2) on >= 2/3 seeds.
  C2 (L4 dissociation): ARM_BANK_ON produces >= 1 wanting_target != liking_target
                       event in the MEASURE phase on >= 2/3 seeds.
  C3 (OFF control):    ARM_BANK_OFF produces 0 wanting!=liking events on ALL seeds
                       (no bank -> no per-object pointer -> no dissociation).
  C4 (OFF seeding):    ARM_BANK_OFF still forms z_goal (z_goal_norm > floor on
                       contact) on ALL seeds (bit-identical legacy seeding intact).
  overall_pass = C1 AND C2 AND C3 AND C4.

Interpretation grid:
  PASS              -> SD-057 mechanism validated; unblocks the GAP-2-gated L9
                       behavioural retest of MECH-229 / MECH-117 / ARC-030.
  FAIL on C2 only   -> bank not redirecting the seed pointer (L4 wiring bug) -> /diagnose-errors.
  FAIL on C3 only   -> OFF arm leaking bank behaviour (flag-gating bug) -> /diagnose-errors.
  FAIL on C4 only   -> legacy seeding regressed -> /diagnose-errors.
  FAIL on C1        -> bank not binding object identity (L2 bug) -> /diagnose-errors.

claim_ids = []  (substrate-readiness diagnostic; NOT governance evidence weighting).
experiment_purpose = "diagnostic".

References:
- REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md
- REE_assembly/evidence/planning/goal_pipeline_plan.md (GAP-7, L0-L9 closure map)
- ree-v3/ree_core/goal.py (IncentiveTokenBank), ree_core/agent.py (update_z_goal resource_type)
- V3-EXQ-626b (the L1 positive control whose GAP-2-decoupling pattern this mirrors)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome

# -- Header constants --------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_636_sd057_object_bound_incentive_mechanism"
QUEUE_ID = "V3-EXQ-636"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SEEDS = [42, 43, 44]

# Resource type tags (SD-049 1-indexed): 1=food (hunger axis 0), 2=water (thirst axis 1).
TYPE_FOOD = 1
TYPE_WATER = 2

# Drive vectors over SD-049 axes (hunger, thirst, curiosity).
DRIVE_HUNGRY = [0.9, 0.1, 0.1]   # favours food (type 1)
DRIVE_THIRSTY = [0.1, 0.9, 0.1]  # favours water (type 2)

FORCED_BENEFIT = 1.0   # supra-threshold benefit on every forced contact
SCALAR_DRIVE = 1.0     # GoalState gate drive (separate from the per-axis recall drive)

# -- Pre-registered thresholds -----------------------------------------------
C1_MIN_TYPES_BOUND = 2            # bank must bind >= 2 distinct types
C4_ZGOAL_FLOOR = 1e-4             # OFF-arm legacy seeding must form non-trivial z_goal
PASS_SEED_FRACTION = 2.0 / 3.0    # >= 2/3 seeds clears C1/C2

# Phase budgets (one "contact" == one forced update_z_goal event).
BUILD_CONTACTS_PER_TYPE = 3       # lay down tokens for each type
MEASURE_CONTACTS = 6              # forced food contacts while thirsty
CONTACTS_PER_RUN = 2 * BUILD_CONTACTS_PER_TYPE + MEASURE_CONTACTS  # 12

WORLD_DIM = 32


@dataclass
class ArmConfig:
    name: str
    use_incentive_token_bank: bool


def arm_configs() -> List[ArmConfig]:
    return [
        ArmConfig(name="ARM_BANK_ON", use_incentive_token_bank=True),
        ArmConfig(name="ARM_BANK_OFF", use_incentive_token_bank=False),
    ]


def _build_env(seed: int) -> CausalGridWorldV2:
    """SD-049 multi-resource env. Contacts are FORCED in the harness (we call
    update_z_goal with a chosen type tag), so the env only needs to be a valid
    stepping substrate that emits a z_resource-bearing observation."""
    return CausalGridWorldV2(
        size=8,
        num_hazards=0,
        num_resources=4,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=3,
        per_axis_drive_enabled=True,
        seed=seed,
    )


def _build_agent(seed: int, device: torch.device, body_dim: int,
                 world_obs_dim: int, use_bank: bool) -> REEAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=body_dim,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        world_dim=WORLD_DIM,
        z_goal_enabled=True,
        drive_weight=2.0,
        use_incentive_token_bank=use_bank,
        incentive_decay=0.005,
        incentive_value_alpha=0.1,
        incentive_drive_kappa_weight=2.0,
        incentive_use_per_axis_drive=True,
    )
    # SD-015 pattern: use_resource_encoder is a LatentStackConfig field set
    # directly (not surfaced through from_dims). Required for z_resource to
    # populate, which SD-057's L2 bind stores per type.
    cfg.latent.use_resource_encoder = True
    return REEAgent(cfg).to(device)


def _body_world(obs: Dict[str, Any], device: torch.device):
    body = obs["body_state"]
    world = obs["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body.to(device), world.to(device)


def _forced_contact(agent: REEAgent, obs: Dict[str, Any], device: torch.device,
                    resource_type: int, drive_vec: List[float]) -> float:
    """One forced contact: sense (populate z_resource), set the per-axis drive
    state, then drive the goal pipeline with a forced supra-threshold benefit on
    the chosen resource_type. Returns z_goal_norm after the update."""
    body, world = _body_world(obs, device)
    agent.sense(obs_body=body, obs_world=world)
    # Set the recall-time per-axis drive AFTER sense (sense leaves _per_axis_drive
    # untouched when obs_per_axis_drive is not passed); the L4 bank read uses it.
    agent._per_axis_drive = torch.tensor(drive_vec, dtype=torch.float32, device=device)
    agent.update_z_goal(
        benefit_exposure=FORCED_BENEFIT,
        drive_level=SCALAR_DRIVE,
        resource_type=resource_type,
    )
    gs = getattr(agent, "goal_state", None)
    if gs is None or not hasattr(gs, "goal_norm"):
        return 0.0
    return float(gs.goal_norm())


def _wanting_target(agent: REEAgent, drive_vec: List[float]) -> Optional[int]:
    """The type the bank currently points z_goal at (argmax wanting). None when
    the bank is absent (OFF arm) or empty."""
    gs = getattr(agent, "goal_state", None)
    bank = getattr(gs, "incentive_bank", None) if gs is not None else None
    if bank is None:
        return None
    mw = bank.most_wanted(
        per_axis_drive=torch.tensor(drive_vec, dtype=torch.float32),
    )
    return None if mw is None else int(mw[0])


@dataclass
class CellSummary:
    seed: int
    arm: str
    types_bound: int
    wanting_ne_liking_events: int
    measure_events: int
    zgoal_norm_min_on_contact: float
    dissoc_detail: List[Dict[str, Any]] = field(default_factory=list)


def run_seed_arm(seed: int, arm: ArmConfig, device: torch.device,
                 body_dim: int, world_obs_dim: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm.name}", flush=True)
    agent = _build_agent(seed, device, body_dim, world_obs_dim, arm.use_incentive_token_bank)
    env = _build_env(seed)
    _, obs = env.reset()
    agent.reset()

    contact_i = 0
    zgoal_norms: List[float] = []
    dissoc_events = 0
    measure_events = 0
    dissoc_detail: List[Dict[str, Any]] = []

    def _tick(rtype: int, drive_vec: List[float], phase: str):
        nonlocal contact_i, dissoc_events, measure_events
        zg = _forced_contact(agent, obs, device, rtype, drive_vec)
        zgoal_norms.append(zg)
        contact_i += 1
        if phase == "measure":
            measure_events += 1
            wt = _wanting_target(agent, drive_vec)  # None on OFF arm
            if wt is not None and wt != rtype:
                # wanting target (most-wanted) differs from liking target (contacted)
                dissoc_events += 1
                dissoc_detail.append(
                    {"contact": contact_i, "liking_target": rtype, "wanting_target": wt}
                )
        if (contact_i % 4 == 0) or contact_i == CONTACTS_PER_RUN:
            print(f"  [train] seed={seed} arm={arm.name} ep {contact_i}/{CONTACTS_PER_RUN} "
                  f"phase={phase} z_goal={zg:.4f}", flush=True)

    # BUILD phase: lay down tokens for both types under matched drive.
    for _ in range(BUILD_CONTACTS_PER_TYPE):
        _tick(TYPE_WATER, DRIVE_THIRSTY, "build")
    for _ in range(BUILD_CONTACTS_PER_TYPE):
        _tick(TYPE_FOOD, DRIVE_HUNGRY, "build")
    # MEASURE phase: contact FOOD while THIRSTY -> most-wanted should be WATER.
    for _ in range(MEASURE_CONTACTS):
        _tick(TYPE_FOOD, DRIVE_THIRSTY, "measure")

    gs = agent.goal_state
    bank = getattr(gs, "incentive_bank", None)
    types_bound = len(bank._base_value) if bank is not None else 0

    summary = CellSummary(
        seed=seed, arm=arm.name,
        types_bound=types_bound,
        wanting_ne_liking_events=dissoc_events,
        measure_events=measure_events,
        zgoal_norm_min_on_contact=float(min(zgoal_norms)) if zgoal_norms else 0.0,
        dissoc_detail=dissoc_detail,
    )
    print(f"verdict: seed={seed} arm={arm.name} types_bound={types_bound} "
          f"dissoc_events={dissoc_events}/{measure_events} "
          f"zgoal_min={summary.zgoal_norm_min_on_contact:.4f}", flush=True)
    return summary.__dict__


def _frac(values: List[bool]) -> float:
    return float(sum(1 for v in values if v)) / float(len(values)) if values else 0.0


def evaluate_acceptance(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    on = [c for c in cells if c["arm"] == "ARM_BANK_ON"]
    off = [c for c in cells if c["arm"] == "ARM_BANK_OFF"]

    c1_flags = [c["types_bound"] >= C1_MIN_TYPES_BOUND for c in on]
    c1_pass = _frac(c1_flags) >= PASS_SEED_FRACTION

    c2_flags = [c["wanting_ne_liking_events"] >= 1 for c in on]
    c2_pass = _frac(c2_flags) >= PASS_SEED_FRACTION

    # C3 / C4 are structural -> require ALL seeds.
    c3_pass = bool(off) and all(c["wanting_ne_liking_events"] == 0 for c in off)
    c4_pass = bool(off) and all(c["zgoal_norm_min_on_contact"] > C4_ZGOAL_FLOOR for c in off)

    overall = bool(c1_pass and c2_pass and c3_pass and c4_pass)
    return {
        "C1_bind_object_identity": {
            "pass": c1_pass,
            "arm_bank_on_types_bound": [c["types_bound"] for c in on],
            "min_types": C1_MIN_TYPES_BOUND,
            "note": "L2/L3: the incentive bank binds benefit to >=2 distinct object identities.",
        },
        "C2_wanting_liking_dissociation": {
            "pass": c2_pass,
            "arm_bank_on_dissoc_events": [c["wanting_ne_liking_events"] for c in on],
            "measure_events": [c["measure_events"] for c in on],
            "note": "L4: under thirst, most-wanted=water differs from contacted=food -> wanting target != liking target.",
        },
        "C3_off_arm_no_dissociation": {
            "pass": c3_pass,
            "arm_bank_off_dissoc_events": [c["wanting_ne_liking_events"] for c in off],
            "note": "OFF control: no bank -> no per-object pointer -> wanting target undefined -> 0 events.",
        },
        "C4_off_arm_legacy_seeding_intact": {
            "pass": c4_pass,
            "arm_bank_off_zgoal_min": [c["zgoal_norm_min_on_contact"] for c in off],
            "floor": C4_ZGOAL_FLOOR,
            "note": "OFF control: legacy single-attractor seeding still forms z_goal on contact.",
        },
        "overall_pass": overall,
    }


def emit_manifest(cells: List[Dict[str, Any]], acceptance: Dict[str, Any],
                  out_dir: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": ts,
        "design_doc": "REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md",
        "plan_of_record": "REE_assembly/evidence/planning/goal_pipeline_plan.md",
        "decoupling_note": (
            "Forced-contact MECHANISM diagnostic decoupled from goal_pipeline:GAP-2 "
            "(foraging competence), mirroring V3-EXQ-626b's L1 decoupling. Validates the "
            "L2-L4 mechanism (wanting type pointer != liking/contacted type). The "
            "ecological, embedding-distinct behavioural L9 dissociation remains GAP-2-gated."
        ),
        "acceptance": acceptance,
        "cells": cells,
        "seeds": SEEDS,
        "protocol": {
            "build_contacts_per_type": BUILD_CONTACTS_PER_TYPE,
            "measure_contacts": MEASURE_CONTACTS,
            "contacts_per_run": CONTACTS_PER_RUN,
            "drive_hungry": DRIVE_HUNGRY,
            "drive_thirsty": DRIVE_THIRSTY,
            "forced_benefit": FORCED_BENEFIT,
        },
        "thresholds": {
            "C1_MIN_TYPES_BOUND": C1_MIN_TYPES_BOUND,
            "C4_ZGOAL_FLOOR": C4_ZGOAL_FLOOR,
            "PASS_SEED_FRACTION": PASS_SEED_FRACTION,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written: {out_path}")
    return out_path


def main(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    device = torch.device("cpu")
    probe = _build_env(seed=0)
    _, obs0 = probe.reset()
    b0, w0 = _body_world(obs0, device)
    body_dim, world_obs_dim = b0.shape[-1], w0.shape[-1]

    cells: List[Dict[str, Any]] = []
    for arm in arm_configs():
        for seed in SEEDS:
            cells.append(run_seed_arm(seed, arm, device, body_dim, world_obs_dim))

    acceptance = evaluate_acceptance(cells)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    print(
        f"Overall: {outcome} | "
        f"C1={acceptance['C1_bind_object_identity']['pass']} "
        f"C2={acceptance['C2_wanting_liking_dissociation']['pass']} "
        f"C3={acceptance['C3_off_arm_no_dissociation']['pass']} "
        f"C4={acceptance['C4_off_arm_legacy_seeding_intact']['pass']}",
        flush=True,
    )

    if args.dry_run:
        print(f"verdict: dry_run overall={outcome}", flush=True)
        return outcome, None

    out_path = emit_manifest(cells, acceptance, Path(args.output_dir))
    return outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE),
    )
    args = parser.parse_args()
    outcome, manifest_path = main(args)
    if not args.dry_run and manifest_path:
        emit_outcome(outcome=outcome, manifest_path=manifest_path)
    sys.exit(0 if outcome == "PASS" else 1)
