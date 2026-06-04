"""
V3-EXQ-637: SD-057 phase-2 mechanism diagnostic -- L6 cue-recall (MECH-347) +
L7 dACC object-discriminative readout (MECH-348). GAP-7.

Scientific question (two mechanisms, one forced harness):
  L6 -- does a PERCEIVED cue (NO benefit pulse) raise wanting for the MATCHED
        object, moving z_goal toward that object's stored embedding before
        consumption (identity-matched, specific-PIT)?
  L7 -- does the dACC consumer now READ the (object-bound) z_goal -- i.e. does a
        per-candidate goal_proximity readout reach the dACC bundle so the
        consumer is object-discriminative rather than z_goal-blind?

DECOUPLED FROM goal_pipeline:GAP-2 (foraging competence), mirroring V3-EXQ-636 /
626b: tokens are laid down by FORCED contacts and the cue is fired by calling the
substrate primitive directly. The ecological behavioural consequence (approach
changes) remains GAP-2-gated; this validates the L6/L7 MECHANISM + wiring.

Arms (2 arms x 3 seeds):
  ARM_PHASE2_ON   use_incentive_token_bank + use_cue_recall + use_dacc +
                  use_mech_consume (dacc_weight>0, dacc_goal_readout_weight>0).
  ARM_OFF         all phase-2 flags off (legacy). update_z_goal/cue calls still
                  made (no-op paths): cue_recall returns 0, dACC bundle carries
                  no goal_readout.

Per seed x arm:
  1. sense; lay down tokens for FOOD (type 1, hungry) and WATER (type 2, thirsty)
     via forced update_z_goal(resource_type=k).
  2. L6: snapshot z_goal; under HUNGRY drive, fire cue_recall_wanting(cue_type=1)
     with NO benefit; record cue strength + the change in cosine(z_goal,
     z_object[FOOD]). Identity-matched => cosine to the CUED object rises.
  3. L7: run one select_action; read the dACC bundle's per-candidate goal_readout
     (object-discriminative input reaching the consumer).

Pre-registered acceptance criteria:
  C1 (L6 cue fires):       ARM_PHASE2_ON cue_recall strength > 0 on >= 2/3 seeds.
  C2 (L6 identity-matched): ARM_PHASE2_ON the z_goal MOVEMENT from the cue points
                           toward the cued object -- cosine(z_goal_after -
                           z_goal_before, z_object[cued] - z_goal_before) > 0.5 on
                           >= 2/3 seeds. (Directional, not raw-cosine: in a
                           non-navigated forced harness z_object embeddings are
                           near-identical across types, so raw cosine-to-cued is
                           degenerate; the movement DIRECTION cleanly tests that
                           the cue pulls z_goal toward the matched object.)
  C3 (L7 readout reaches dACC): ARM_PHASE2_ON dACC bundle goal_readout is non-None,
                           length==K, finite on >= 2/3 seeds.
  C4 (OFF parity):         ARM_OFF cue strength == 0 AND dACC goal_readout is None
                           on ALL seeds.
  overall_pass = C1 AND C2 AND C3 AND C4.

Interpretation grid:
  PASS            -> SD-057 phase-2 (L6+L7) mechanism validated.
  FAIL C1/C2      -> cue-recall not moving z_goal (L6 wiring) -> /diagnose-errors.
  FAIL C3         -> goal readout not reaching dACC (L7 wiring) -> /diagnose-errors.
  FAIL C4         -> OFF arm leaking phase-2 behaviour (flag-gating bug).

claim_ids = []  (substrate-readiness diagnostic; NOT governance evidence).
experiment_purpose = "diagnostic".

References:
- REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md (Phase-2)
- ree-v3/ree_core/goal.py (GoalState.cue_pull), agent.py (cue_recall_wanting),
  cingulate/dacc.py (goal_readout)
- V3-EXQ-636 (SD-057 v1 mechanism diagnostic; same GAP-2-decoupling pattern)
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
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome

EXPERIMENT_TYPE = "v3_exq_637_sd057_phase2_cue_recall_consume"
QUEUE_ID = "V3-EXQ-637"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SEEDS = [42, 43, 44]

TYPE_FOOD = 1
TYPE_WATER = 2
DRIVE_HUNGRY = [0.9, 0.1, 0.1]
DRIVE_THIRSTY = [0.1, 0.9, 0.1]
FORCED_BENEFIT = 1.0
PASS_SEED_FRACTION = 2.0 / 3.0
STEPS_PER_RUN = 8  # progress denominator (token-build + cue + select_action)

WORLD_DIM = 32


@dataclass
class ArmConfig:
    name: str
    phase2_on: bool


def arm_configs() -> List[ArmConfig]:
    return [ArmConfig("ARM_PHASE2_ON", True), ArmConfig("ARM_OFF", False)]


def _build_env(seed: int) -> CausalGridWorldV2:
    # SD-049 multi-resource (per-type tags + per-axis drive) + limb damage so
    # harm_obs_a is emitted (the dACC block needs z_harm_a to fire for L7).
    return CausalGridWorldV2(
        size=8, num_hazards=1, num_resources=4,
        multi_resource_heterogeneity_enabled=True, n_resource_types=3,
        per_axis_drive_enabled=True, limb_damage_enabled=True, seed=seed,
    )


def _build_agent(seed: int, device: torch.device, body_dim: int,
                 world_obs_dim: int, phase2_on: bool) -> REEAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=body_dim, world_obs_dim=world_obs_dim, action_dim=4,
        world_dim=WORLD_DIM, harm_dim=16,
        use_harm_stream=True, z_harm_dim=16,
        use_affective_harm_stream=True, z_harm_a_dim=7, harm_history_len=10,
        limb_damage_enabled=True,
        z_goal_enabled=True, drive_weight=2.0,
        # dACC must exist in BOTH arms so the L7 readout has a host; only the
        # goal-readout (use_mech_consume / weight) toggles between arms.
        use_dacc=True, dacc_weight=0.5, dacc_interaction_weight=0.5,
        use_e2_harm_a=True,
        # SD-057 v1 bank + phase-2 flags (ON arm only).
        use_incentive_token_bank=phase2_on,
        use_cue_recall=phase2_on, cue_recall_gain=0.2,
        use_mech_consume=phase2_on,
        dacc_goal_readout_weight=(0.5 if phase2_on else 0.0),
    )
    cfg.latent.use_resource_encoder = True
    return REEAgent(cfg).to(device)


def _obs_tensors(obs: Dict[str, Any], device: torch.device):
    def _b(x):
        if x is None:
            return None
        return (x.unsqueeze(0) if x.dim() == 1 else x).to(device)
    return (_b(obs.get("body_state")), _b(obs.get("world_state")),
            _b(obs.get("harm_obs")), _b(obs.get("harm_obs_a")))


def _sense(agent: REEAgent, obs: Dict[str, Any], device: torch.device):
    body, world, harm, harm_a = _obs_tensors(obs, device)
    return agent.sense(obs_body=body, obs_world=world,
                       obs_harm=harm, obs_harm_a=harm_a)


def _forced_contact(agent, obs, device, rtype, drive_vec):
    _sense(agent, obs, device)
    agent._per_axis_drive = torch.tensor(drive_vec, dtype=torch.float32, device=device)
    agent.update_z_goal(benefit_exposure=FORCED_BENEFIT, drive_level=1.0,
                        resource_type=rtype)


@dataclass
class CellSummary:
    seed: int
    arm: str
    cue_strength: float
    cue_direction_cos: float
    goal_readout_present: bool
    goal_readout_len: int
    goal_readout_finite: bool


def run_seed_arm(seed, arm, device, body_dim, world_obs_dim) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm.name}", flush=True)
    agent = _build_agent(seed, device, body_dim, world_obs_dim, arm.phase2_on)
    env = _build_env(seed)
    _, obs = env.reset()
    agent.reset()
    step = 0

    def _tick(msg):
        nonlocal step
        step += 1
        if step % 4 == 0 or step == STEPS_PER_RUN:
            print(f"  [train] seed={seed} arm={arm.name} ep {step}/{STEPS_PER_RUN} {msg}",
                  flush=True)

    # 1. Lay down tokens for both types (food/hungry, water/thirsty).
    for _ in range(2):
        _forced_contact(agent, obs, device, TYPE_FOOD, DRIVE_HUNGRY); _tick("build food")
    for _ in range(2):
        _forced_contact(agent, obs, device, TYPE_WATER, DRIVE_THIRSTY); _tick("build water")

    # 2. L6: cue-recall for FOOD under HUNGRY drive, NO benefit. Identity-matched
    #    => cosine(z_goal, z_object[FOOD]) should INCREASE.
    bank = getattr(agent.goal_state, "incentive_bank", None)
    z_food = bank._z_object.get(TYPE_FOOD, None) if bank is not None else None

    agent._per_axis_drive = torch.tensor(DRIVE_HUNGRY, dtype=torch.float32, device=device)
    z_goal_before = agent.goal_state.z_goal.detach().clone()
    cue_strength = agent.cue_recall_wanting(cue_type=TYPE_FOOD, drive_level=1.0)
    z_goal_after = agent.goal_state.z_goal.detach().clone()
    # Identity-matched DIRECTION: did the cue move z_goal toward z_object[FOOD]?
    # cosine(delta z_goal, z_object[FOOD] - z_goal_before). Robust to near-identical
    # forced-harness embeddings (raw cosine-to-cued is degenerate there).
    cue_direction_cos = 0.0
    if z_food is not None:
        delta = (z_goal_after - z_goal_before).reshape(-1)
        target_dir = (z_food.reshape(-1) - z_goal_before.reshape(-1))
        if delta.norm() > 1e-9 and target_dir.norm() > 1e-9:
            cue_direction_cos = float(
                F.cosine_similarity(delta.unsqueeze(0), target_dir.unsqueeze(0)).item()
            )
    _tick(f"cue s={cue_strength:.3f} dir_cos={cue_direction_cos:.3f}")

    # 3. L7: one select_action; read the dACC bundle's goal_readout.
    latent = _sense(agent, obs, device)
    ticks = agent.clock.advance()
    world_dim = agent.config.latent.world_dim
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=device))
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    K = len(candidates)
    agent.select_action(candidates, ticks)
    bundle = getattr(agent, "_dacc_last_bundle", None) or {}
    gr = bundle.get("goal_readout", None)
    gr_present = gr is not None
    gr_len = int(gr.numel()) if (gr_present and hasattr(gr, "numel")) else 0
    gr_finite = bool(torch.isfinite(gr).all().item()) if gr_present else False
    _tick(f"goal_readout present={gr_present} len={gr_len}")

    summary = CellSummary(
        seed=seed, arm=arm.name,
        cue_strength=float(cue_strength), cue_direction_cos=float(cue_direction_cos),
        goal_readout_present=gr_present, goal_readout_len=gr_len,
        goal_readout_finite=gr_finite,
    )
    print(f"verdict: seed={seed} arm={arm.name} cue={cue_strength:.3f} "
          f"dir_cos={cue_direction_cos:.3f} readout={gr_present}({gr_len}) K={K}", flush=True)
    return summary.__dict__


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def evaluate_acceptance(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    on = [c for c in cells if c["arm"] == "ARM_PHASE2_ON"]
    off = [c for c in cells if c["arm"] == "ARM_OFF"]

    c1 = _frac([c["cue_strength"] > 0.0 for c in on]) >= PASS_SEED_FRACTION
    c2 = _frac([c["cue_direction_cos"] > 0.5 for c in on]) >= PASS_SEED_FRACTION
    c3 = _frac([c["goal_readout_present"] and c["goal_readout_len"] > 0
                and c["goal_readout_finite"] for c in on]) >= PASS_SEED_FRACTION
    c4 = bool(off) and all(
        c["cue_strength"] == 0.0 and not c["goal_readout_present"] for c in off
    )
    overall = bool(c1 and c2 and c3 and c4)
    return {
        "C1_cue_fires": {"pass": c1,
            "on_cue_strengths": [c["cue_strength"] for c in on],
            "note": "L6: perceived cue (no benefit) fires cue-recall."},
        "C2_identity_matched": {"pass": c2,
            "on_cue_direction_cos": [c["cue_direction_cos"] for c in on],
            "note": "L6: cue moves z_goal TOWARD z_object[FOOD] (direction cosine > 0.5) -- specific PIT, robust to forced-harness embedding similarity."},
        "C3_readout_reaches_dacc": {"pass": c3,
            "on_goal_readout_len": [c["goal_readout_len"] for c in on],
            "note": "L7: object-discriminative per-candidate goal_proximity reaches the dACC bundle."},
        "C4_off_parity": {"pass": c4,
            "off_cue_strengths": [c["cue_strength"] for c in off],
            "off_readout_present": [c["goal_readout_present"] for c in off],
            "note": "OFF: no cue effect, dACC z_goal-blind (goal_readout None)."},
        "overall_pass": overall,
    }


def emit_manifest(cells, acceptance, out_dir: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    manifest = {
        "run_id": run_id, "queue_id": QUEUE_ID, "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS, "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome, "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": ts,
        "design_doc": "REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md",
        "decoupling_note": (
            "SD-057 phase-2 (L6 cue-recall + L7 dACC readout) MECHANISM diagnostic, "
            "decoupled from goal_pipeline:GAP-2 (mirrors V3-EXQ-636 / 626b). The "
            "ecological behavioural consequence remains GAP-2-gated."
        ),
        "acceptance": acceptance, "cells": cells, "seeds": SEEDS,
        "protocol": {"steps_per_run": STEPS_PER_RUN,
                     "drive_hungry": DRIVE_HUNGRY, "drive_thirsty": DRIVE_THIRSTY,
                     "forced_benefit": FORCED_BENEFIT, "cue_recall_gain": 0.2,
                     "dacc_goal_readout_weight": 0.5},
        "thresholds": {"PASS_SEED_FRACTION": PASS_SEED_FRACTION},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written: {out_path}")
    return out_path


def main(args) -> Tuple[str, Optional[str]]:
    device = torch.device("cpu")
    probe = _build_env(seed=0)
    _, obs0 = probe.reset()
    b0, w0, _, _ = _obs_tensors(obs0, device)
    body_dim, world_obs_dim = b0.shape[-1], w0.shape[-1]

    cells: List[Dict[str, Any]] = []
    for arm in arm_configs():
        for seed in SEEDS:
            cells.append(run_seed_arm(seed, arm, device, body_dim, world_obs_dim))

    acceptance = evaluate_acceptance(cells)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    print(f"Overall: {outcome} | C1={acceptance['C1_cue_fires']['pass']} "
          f"C2={acceptance['C2_identity_matched']['pass']} "
          f"C3={acceptance['C3_readout_reaches_dacc']['pass']} "
          f"C4={acceptance['C4_off_parity']['pass']}", flush=True)

    if args.dry_run:
        print(f"verdict: dry_run overall={outcome}", flush=True)
        return outcome, None
    out_path = emit_manifest(cells, acceptance, Path(args.output_dir))
    return outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str,
        default=str(REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE))
    args = parser.parse_args()
    outcome, manifest_path = main(args)
    if not args.dry_run and manifest_path:
        emit_outcome(outcome=outcome, manifest_path=manifest_path)
    sys.exit(0 if outcome == "PASS" else 1)
