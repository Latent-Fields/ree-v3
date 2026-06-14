"""
V3-EXQ-681: SD-057 object-bound wanting!=liking CHANNEL-WRITE forced-contact
microdiagnostic. Precursor that de-risks the full V3-EXQ-514o GAP-2 successor
BEFORE sinking a multi-seed behavioural curriculum run.

claim_ids = []  (diagnostic positive control; NOT load-bearing evidence per the
                 CLAIM_IDS Accuracy Rule).
experiment_purpose = "diagnostic".

WHY THIS EXISTS
---------------
V3-EXQ-514n (the GAP-2 owner) FAILed substrate_not_ready_requeue: even on the
guard-passing seed42 it scored n_scored_wl_steps=0 / run_bank_populated=false
despite distinct_tokens_max=3 and drive_spread_max=0.012 (> floor). The co-timing
defect, code-confirmed:

  514n's eval sourced the LIKING target (last-consumed type) via
  scaffolded_sd054_onboarding._contacted_resource_type(obs_dict), which reads
  obs_dict["resource_type_at_agent"] = _resource_type_grid[agent_x, agent_y]
  (causal_grid_world.py). At a consumption tick the env REMOVES the resource and
  CLEARS that grid cell, so on the post-step obs_dict resource_type_at_agent == 0
  -> _contacted_resource_type returns None. The consumed-this-tick tag lives in
  info["sd049_consumed_type_tag_this_tick"] (causal_grid_world.py), which the
  eval never read (it passed obs_dict only, and that key is NOT in obs_dict).

  Net: rtype = None at every consumption step -> the WL scorer's
  (rtype is not None) gate never held -> n_scored=0; the bank's tokens came only
  from the curriculum Stage-0 _strongest_perceived_type binding, never from the
  eval's own contacts. This is a HARNESS sourcing bug, NOT a goal.py defect --
  IncentiveTokenBank.most_wanted / .update are correct.

THE FIX VALIDATED HERE (decoupled from foraging, 626b-style)
------------------------------------------------------------
This microdiagnostic FORCES contact (benefit + consumed-tag supplied directly,
NOT read from the post-consumption-cleared at-agent cell), exactly as V3-EXQ-626b
forced the L1 benefit pulse to decouple the positive control from GAP-2 foraging
competence. It proves the SD-057 channel-write + WL scoring machinery the full
V3-EXQ-514o successor relies on actually fires:

  C1 INSTRUMENT (constructed-bank, deterministic; = 514n readiness leg-1 parity):
     a 2-token IncentiveTokenBank built from this run's GoalConfig yields a
     drive-favored most_wanted that DIFFERS from a designated last-consumed type
     -- the SAME cross-target inequality statistic the C_WL criterion routes on.
     Separation must equal PC_SEPARATION_FLOOR (1.0).

  C2 DISSOCIATION (real agent + bank, forced contact): bind 2 tokens, then over
     SCORE_STEPS force the LIKING target directly (alternating type1/type2) with a
     fixed per-axis drive favoring type2 (large spread). The MECH-346 most_wanted
     pointer (wanting target) is read at each forced contact and compared to the
     forced liking target via the substrate's own agent.update_z_goal(resource_type=)
     L2-bind/L4-pointer path. Proves: bank populates >= 2 distinct tokens,
     drive_spread > floor, n_scored_wl_steps >= MIN_SCORED_STEPS, and
     wl_dissoc_fraction >= WL_FRACTION (wanting differs from liking on the steps
     where the forced liking != the drive-favored wanting).

  C3 NULL / PARITY (real agent + bank, forced contact): same machinery, but the
     forced liking target == the drive-favored wanting target every step ->
     wl_dissoc_fraction ~ 0. Proves the C2 dissociation is a genuine
     wanting!=liking read, not a scoring artifact (the arms are NOT bit-identical).

  C4 CO-TIMING (structural, deterministic): one real SD-049 env step; assert
     "sd049_consumed_type_tag_this_tick" is NOT a key in obs_dict (it lives only
     in info) and "resource_type_at_agent" IS a key in obs_dict. The literal root
     cause of why _contacted_resource_type(obs_dict) could never see the consumed
     tag -- and the reason V3-EXQ-514o must source the liking target from info.

overall_pass = C1 AND C2 AND C3 AND C4.

A PASS confirms the SD-057 WL channel-write + scoring machinery fires under forced
contact with the info-sourced consumed tag, de-risking the full V3-EXQ-514o
multi-seed behavioural run. It does NOT close GAP-2 (it is decoupled from foraging
contact; the contact guard is the separate 514o concern) and weights NO claim.

DIAGNOSTIC ADJUDICATION (self-route is a hypothesis, not a verdict)
------------------------------------------------------------------
The load-bearing criterion is C2 wl_dissoc_fraction >= WL_FRACTION (the WL
instrument fires AND dissociates). Its readiness precondition is the SAME
cross-target inequality statistic on a positive control (C1 constructed-bank
separation == 1.0) PLUS the channel-write non-vacuity (C2 bank populated >= 2
distinct tokens at differing per-axis drive). Below readiness -> the WL
instrument / channel-write is structurally degenerate -> self-route
substrate_not_ready_requeue, NEVER a substrate verdict.

References:
- ree_core/goal.py IncentiveTokenBank (update / most_wanted -- the substrate)
- ree_core/agent.py REEAgent.update_z_goal (L2 bind + L4 pointer path)
- ree_core/environment/causal_grid_world.py (info vs obs_dict key split)
- experiments/v3_exq_514n_sd049_phase2_mech229_object_bound_wanting_liking.py (the FAIL)
- experiments/v3_exq_626b_goal_pipeline_forced_seed_positive_control.py (the decoupling precedent)
- REE_assembly/evidence/planning/goal_pipeline_plan.md (GAP-2, GAP-7)

SLEEP DRIVER: N/A (no sleep loop; forced-contact WL channel-write microdiagnostic).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.goal import IncentiveTokenBank  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# Single deterministic forced-contact positive control; legs are run sequentially
# with full RNG reset, NOT a (seed x arm) training grid -- no arm_results written.
ARM_FINGERPRINT_EXEMPT = (
    "single deterministic forced-contact positive control; no seed x arm training "
    "grid (legs C1-C4 run sequentially with RNG reset; results under 'legs')"
)

EXPERIMENT_TYPE = "v3_exq_681_sd057_wl_channel_write_forced_contact_microdiagnostic"
QUEUE_ID = "V3-EXQ-681"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SEED = 42

# --- Dims (mirror the SD-057 substrate config used by 514n) ---
WORLD_DIM = 32
SELF_DIM = 32
N_RESOURCE_TYPES = 3
DRIVE_WEIGHT = 2.0

# --- Forced-contact budget ---
POPULATE_STEPS = 8        # steps to bind both tokens before scoring
SCORE_STEPS = 40          # forced-contact scoring steps per WL arm
FORCED_BENEFIT = 1.0      # supra-threshold benefit forced every step (>> threshold)

# --- Pre-registered thresholds (constants, NOT derived from the run) ---
PC_SEPARATION_FLOOR = 1.0   # C1: constructed-bank instrument must separate (= 514n leg-1)
MIN_SCORED_STEPS = 5        # C2/C3: min scored WL consumption events
DRIVE_SPREAD_FLOOR = 1e-3   # C2: min per-axis-drive spread (= 514n DRIVE_SPREAD_FLOOR)
WL_FRACTION = 0.3           # C2: object-bound wanting!=liking dissociation floor (= 514n WL_FRACTION)
NULL_DISSOC_CEILING = 0.1   # C3: parity arm dissociation ceiling (wanting == liking)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_env(seed: int) -> CausalGridWorldV2:
    """SD-049-enabled stepping substrate so the agent's ResourceEncoder produces a
    non-None z_resource (the L2-bind embedding) and the env emits the SD-049
    per-axis drive + the info["sd049_consumed_type_tag_this_tick"] tag the C4
    co-timing leg inspects. Env CONTACT is irrelevant to the forced arms by
    construction (benefit + consumed-tag are forced, not read from the env)."""
    return CausalGridWorldV2(
        size=12,
        num_hazards=0,
        num_resources=6,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=N_RESOURCE_TYPES,
        per_axis_drive_enabled=True,
        reef_enabled=True,
        seed=seed,
    )


def _build_agent(env: CausalGridWorldV2, device: torch.device) -> REEAgent:
    """Minimal SD-057 agent: incentive token bank + z_resource encoder + z_goal.
    No harm streams / curriculum machinery -- the micro forces contact directly."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        # SD-057 object-bound incentive-salience layer (the substrate under test)
        use_incentive_token_bank=True,
    )
    cfg.latent.use_resource_encoder = True  # SD-015 z_resource -> bank L2 bind requires it
    return REEAgent(cfg).to(device)


def _drive_favoring(fav_type: int, device: torch.device) -> torch.Tensor:
    """Per-axis drive [n_types] favoring SD-049 type `fav_type` (1-based) with a
    large spread. bank._drive_axis_for maps tag k -> axis k-1, so to favor tag k
    we set axis k-1 high."""
    pad = torch.zeros(N_RESOURCE_TYPES, device=device)
    pad[fav_type - 1] = 1.0
    return pad


# ---------------------------------------------------------------------------
# Leg C1: constructed-bank instrument (= 514n readiness leg-1 parity)
# ---------------------------------------------------------------------------


def leg_c1_instrument(agent: REEAgent, device: torch.device) -> Dict[str, Any]:
    """Build a 2-token IncentiveTokenBank from THIS run's GoalConfig and verify the
    cross-target inequality the C_WL criterion routes on CAN fire: a drive-favored
    most_wanted must differ from a designated last-consumed type. Two deterministic
    probes -> separation fraction (1.0 == instrument works)."""
    print("Seed 42 Condition C1_INSTRUMENT", flush=True)
    print("  [train] C1_instrument ep 1/1 constructed-bank cross-target probe", flush=True)
    goal_cfg = agent.config.goal
    d = int(getattr(goal_cfg, "goal_dim", WORLD_DIM))
    pc_bank = IncentiveTokenBank(goal_cfg, device)
    z_a = torch.zeros(1, d, device=device); z_a[0, 0] = 1.0  # type 1 identity embedding
    z_b = torch.zeros(1, d, device=device); z_b[0, 1] = 1.0  # type 2 identity embedding
    pc_bank.update(1, 1.0, z_a)  # bind type-1 token
    pc_bank.update(2, 1.0, z_b)  # bind type-2 token
    # probe 1: drive favors type 2 -> most-wanted should be 2; designated liking = 1.
    mw1 = pc_bank.most_wanted(per_axis_drive=_drive_favoring(2, device), scalar_drive=1.0)
    sep1 = 1.0 if (mw1 is not None and int(mw1[0]) != 1) else 0.0
    # probe 2: drive favors type 1 -> most-wanted should be 1; designated liking = 2.
    mw2 = pc_bank.most_wanted(per_axis_drive=_drive_favoring(1, device), scalar_drive=1.0)
    sep2 = 1.0 if (mw2 is not None and int(mw2[0]) != 2) else 0.0
    separation = float((sep1 + sep2) / 2.0)
    passed = bool(separation >= PC_SEPARATION_FLOOR)
    print(f"verdict: {'PASS' if passed else 'FAIL'} C1_INSTRUMENT separation={separation:.3f}",
          flush=True)
    return {"leg": "C1_INSTRUMENT", "separation": separation,
            "floor": PC_SEPARATION_FLOOR, "passed": passed}


# ---------------------------------------------------------------------------
# Legs C2 / C3: forced-contact WL scoring (real agent + bank)
# ---------------------------------------------------------------------------


def _forced_wl_arm(
    label: str,
    liking_schedule: str,        # "alternate" (C2) or "fixed_type2" (C3)
    fav_type: int,               # drive-favored (wanting) type
    device: torch.device,
    dry_run: bool,
) -> Dict[str, Any]:
    """Force contact directly (benefit + consumed-tag supplied, NOT read from the
    post-consumption-cleared at-agent cell -- the 514n co-timing fix). At each
    forced contact: set the cached per-axis drive favoring `fav_type`, call the
    substrate's own agent.update_z_goal(resource_type=liking_tag) (L2 bind + L4
    pointer), then read the MECH-346 most_wanted pointer (wanting target) and
    compare to the forced liking target."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = _build_env(SEED)
    agent = _build_agent(env, device)
    bank = getattr(agent.goal_state, "incentive_bank", None)
    assert bank is not None, "incentive_bank must be present (use_incentive_token_bank=True)"

    pad = _drive_favoring(fav_type, device)
    drive_spread = float(pad.max().item() - pad.min().item())

    score_steps = 6 if dry_run else SCORE_STEPS
    pop_steps = 4 if dry_run else POPULATE_STEPS

    print(f"Seed 42 Condition {label}", flush=True)
    _, obs_dict = env.reset()
    agent.reset()

    scored = 0
    dissoc = 0
    distinct_tokens_max = 0
    contact_steps = 0

    def _step_env_and_sense() -> None:
        """Advance the env one step and run sense() so _current_latent.z_resource
        is populated (the L2-bind embedding). Action is irrelevant (forced arm)."""
        nonlocal obs_dict
        obs_body = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)
        with torch.no_grad():
            agent.sense(obs_body, obs_world)
        # step with a fixed action; env contact is ignored (forced contact below).
        _, _harm, done, _info, obs_dict = env.step(0)
        if done:
            _, obs_dict = env.reset()

    # --- Populate phase: bind BOTH tokens (alternate type1/type2) ---
    for i in range(pop_steps):
        _step_env_and_sense()
        liking_tag = 1 if (i % 2 == 0) else 2
        agent._per_axis_drive = pad.reshape(-1)
        with torch.no_grad():
            agent.update_z_goal(float(FORCED_BENEFIT), drive_level=1.0, resource_type=liking_tag)

    # --- Score phase: forced contact, read most_wanted vs forced liking ---
    for step in range(score_steps):
        _step_env_and_sense()
        if liking_schedule == "alternate":
            liking_tag = 1 if (step % 2 == 0) else 2
        else:  # "fixed_type2"
            liking_tag = 2
        agent._per_axis_drive = pad.reshape(-1)
        contact_steps += 1
        with torch.no_grad():
            agent.update_z_goal(float(FORCED_BENEFIT), drive_level=1.0, resource_type=liking_tag)
            if not bank.is_empty():
                n_distinct = len(bank.wanting(per_axis_drive=pad.reshape(-1), scalar_drive=1.0))
                distinct_tokens_max = max(distinct_tokens_max, n_distinct)
                mw = bank.most_wanted(per_axis_drive=pad.reshape(-1), scalar_drive=1.0)
                if mw is not None and n_distinct >= 2:
                    wanting_tag = int(mw[0])
                    scored += 1
                    if wanting_tag != liking_tag:
                        dissoc += 1
        if ((step + 1) % 10 == 0) or (step + 1) == score_steps:
            print(f"  [train] {label} ep {step + 1}/{score_steps} "
                  f"scored={scored} distinct={distinct_tokens_max}", flush=True)

    wl_fraction = (float(dissoc) / float(scored)) if scored > 0 else 0.0
    run_populated = bool(
        distinct_tokens_max >= 2 and scored >= MIN_SCORED_STEPS and drive_spread > DRIVE_SPREAD_FLOOR
    )
    return {
        "leg": label,
        "liking_schedule": liking_schedule,
        "fav_type": fav_type,
        "n_scored_wl_steps": scored,
        "n_wl_dissoc_steps": dissoc,
        "wl_dissoc_fraction": wl_fraction,
        "distinct_tokens_max": distinct_tokens_max,
        "drive_spread": drive_spread,
        "run_bank_populated": run_populated,
        "contact_steps": contact_steps,
    }


def leg_c2_dissociation(device: torch.device, dry_run: bool) -> Dict[str, Any]:
    """Forced contact with liking ALTERNATING type1/type2 while drive favors type2
    -> wanting (most_wanted) is type2; dissociation on the steps liking == type1."""
    r = _forced_wl_arm("C2_DISSOCIATION", "alternate", fav_type=2, device=device, dry_run=dry_run)
    passed = bool(
        r["run_bank_populated"]
        and r["n_scored_wl_steps"] >= MIN_SCORED_STEPS
        and r["wl_dissoc_fraction"] >= WL_FRACTION
    )
    r["passed"] = passed
    print(f"verdict: {'PASS' if passed else 'FAIL'} C2_DISSOCIATION "
          f"scored={r['n_scored_wl_steps']} wl_frac={r['wl_dissoc_fraction']:.3f} "
          f"distinct={r['distinct_tokens_max']} populated={r['run_bank_populated']}", flush=True)
    return r


def leg_c3_null(device: torch.device, dry_run: bool) -> Dict[str, Any]:
    """Forced contact with liking FIXED to type2 == drive-favored wanting -> no
    dissociation (proves C2's dissociation is a genuine read, not an artifact)."""
    r = _forced_wl_arm("C3_NULL_PARITY", "fixed_type2", fav_type=2, device=device, dry_run=dry_run)
    passed = bool(
        r["n_scored_wl_steps"] >= MIN_SCORED_STEPS
        and r["wl_dissoc_fraction"] <= NULL_DISSOC_CEILING
    )
    r["passed"] = passed
    print(f"verdict: {'PASS' if passed else 'FAIL'} C3_NULL_PARITY "
          f"scored={r['n_scored_wl_steps']} wl_frac={r['wl_dissoc_fraction']:.3f} "
          f"(ceiling={NULL_DISSOC_CEILING})", flush=True)
    return r


# ---------------------------------------------------------------------------
# Leg C4: structural co-timing (the 514n bug demonstration)
# ---------------------------------------------------------------------------


def leg_c4_cotiming(device: torch.device) -> Dict[str, Any]:
    """One real SD-049 env step; assert the consumed-this-tick tag is NOT in
    obs_dict (it lives only in info) while resource_type_at_agent IS in obs_dict --
    the literal reason _contacted_resource_type(obs_dict) could never read the
    consumed tag, and why V3-EXQ-514o must source the liking target from info."""
    print("Seed 42 Condition C4_COTIMING", flush=True)
    print("  [train] C4_cotiming ep 1/1 obs_dict-vs-info key split check", flush=True)
    env = _build_env(SEED)
    _, obs_dict = env.reset()
    _, _harm, _done, info, obs_dict2 = env.step(0)
    consumed_in_obs = "sd049_consumed_type_tag_this_tick" in obs_dict2
    consumed_in_info = "sd049_consumed_type_tag_this_tick" in info
    at_agent_in_obs = "resource_type_at_agent" in obs_dict2
    # The 514n bug: _contacted_resource_type(obs_dict) looks for the consumed tag
    # (absent from obs_dict) then falls back to resource_type_at_agent (present but
    # 0 at a consumption tick). The fix: source the consumed tag from info.
    passed = bool((not consumed_in_obs) and consumed_in_info and at_agent_in_obs)
    print(f"verdict: {'PASS' if passed else 'FAIL'} C4_COTIMING "
          f"consumed_in_obs={consumed_in_obs} consumed_in_info={consumed_in_info} "
          f"at_agent_in_obs={at_agent_in_obs}", flush=True)
    return {
        "leg": "C4_COTIMING",
        "consumed_tag_in_obs_dict": consumed_in_obs,
        "consumed_tag_in_info": consumed_in_info,
        "resource_type_at_agent_in_obs_dict": at_agent_in_obs,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Run + manifest
# ---------------------------------------------------------------------------


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    device = torch.device("cpu")

    # C1 needs an agent only for its GoalConfig (constructed bank).
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env0 = _build_env(SEED)
    agent0 = _build_agent(env0, device)

    c1 = leg_c1_instrument(agent0, device)
    c2 = leg_c2_dissociation(device, dry_run)
    c3 = leg_c3_null(device, dry_run)
    c4 = leg_c4_cotiming(device)

    legs = {"C1": c1, "C2": c2, "C3": c3, "C4": c4}

    # --- Readiness (same-statistic non-vacuity for the load-bearing C2 DV) ---
    pc_met = bool(c1["separation"] >= PC_SEPARATION_FLOOR)
    channel_write_met = bool(c2["run_bank_populated"])  # >=2 tokens, >=MIN_SCORED, spread > floor
    readiness_met = bool(pc_met and channel_write_met)

    # --- Non-degeneracy: C2 is a fair test only if it scored AND differs from C3 ---
    arms_differ = bool(abs(c2["wl_dissoc_fraction"] - c3["wl_dissoc_fraction"]) > NULL_DISSOC_CEILING)
    c2_non_degenerate = bool(c2["n_scored_wl_steps"] >= MIN_SCORED_STEPS and arms_differ)

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route_reason = ("wl_positive_control_degenerate" if not pc_met
                        else "wl_channel_not_written")
    else:
        c2p = bool(c2["passed"])
        c3p = bool(c3["passed"])
        c1p = bool(c1["passed"])
        c4p = bool(c4["passed"])
        overall = bool(c1p and c2p and c3p and c4p)
        outcome = "PASS" if overall else "FAIL"
        label = ("sd057_wl_channel_write_validated" if overall
                 else "wl_channel_write_residual_open")
        route_reason = "all_legs_pass" if overall else "leg_unmet"

    overall_pass = bool(outcome == "PASS")

    print(f"[{EXPERIMENT_TYPE}] readiness={readiness_met} "
          f"(pc={pc_met} channel_write={channel_write_met}) "
          f"C1={c1['passed']} C2={c2['passed']} C3={c3['passed']} C4={c4['passed']} "
          f"-> outcome={outcome} route={label}", flush=True)

    acceptance = {
        "C1_instrument_separates": bool(c1["passed"]),
        "C2_forced_dissociation": bool(c2["passed"]),
        "C3_null_parity": bool(c3["passed"]),
        "C4_cotiming_structural": bool(c4["passed"]),
        "readiness_met": readiness_met,
        "c2_wl_dissoc_fraction": c2["wl_dissoc_fraction"],
        "c3_wl_dissoc_fraction": c3["wl_dissoc_fraction"],
        "c2_n_scored_wl_steps": c2["n_scored_wl_steps"],
        "c2_distinct_tokens_max": c2["distinct_tokens_max"],
        "c2_drive_spread": c2["drive_spread"],
        "route_reason": route_reason,
        "overall_pass": overall_pass,
    }

    interpretation = {
        "label": label,
        "readiness_route": label if not readiness_met else "n/a",
        "preconditions": [
            {
                "name": "wl_instrument_positive_control_separates",
                "description": "leg C1: a constructed 2-token IncentiveTokenBank (this run's "
                               "GoalConfig) yields a drive-favored most_wanted that DIFFERS from a "
                               "designated last-consumed type -- the SAME cross-target inequality "
                               "statistic C2 routes on, on a positive control.",
                "control": "2 deterministic probes: drive-favored most_wanted vs designated "
                           "last-consumed (instrument-can-fire).",
                "measured": float(c1["separation"]),
                "threshold": PC_SEPARATION_FLOOR,
                "met": pc_met,
            },
            {
                "name": "wl_channel_written_forced_contact",
                "description": "leg C2: under FORCED contact the substrate's own update_z_goal "
                               "L2-bind/L4-pointer path populated the bank with >= 2 distinct "
                               "tokens at differing per-axis drive AND scored >= MIN_SCORED_STEPS "
                               "consumption events -- the exact channels-written non-vacuity the "
                               "514n eval lacked (consumed tag never sourced).",
                "control": "forced consumed-tag (NOT the post-consumption-cleared at-agent cell) "
                           "+ fixed drive spread.",
                "measured": float(c2["n_scored_wl_steps"]),
                "threshold": float(MIN_SCORED_STEPS),
                "met": channel_write_met,
            },
        ],
        "criteria": [
            {"name": "C2_forced_dissociation", "load_bearing": True, "passed": bool(c2["passed"])},
        ],
        "criteria_non_degenerate": {
            "C2_forced_dissociation": c2_non_degenerate,
        },
        "wl_readiness_gate": {
            "definition": "SAME-STATISTIC non-vacuity for the forced WL DV: C1 constructed-bank "
                          "cross-target separation == 1.0 AND C2 in-run bank populated >= 2 "
                          "distinct tokens at differing per-axis drive with >= MIN_SCORED_STEPS "
                          "scored. Below floor -> substrate_not_ready_requeue.",
            "pc_separation_floor": PC_SEPARATION_FLOOR,
            "min_scored_steps": MIN_SCORED_STEPS,
            "drive_spread_floor": DRIVE_SPREAD_FLOOR,
            "wl_fraction": WL_FRACTION,
            "null_dissoc_ceiling": NULL_DISSOC_CEILING,
        },
    }

    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",  # diagnostic; weights no claim
        "acceptance": acceptance,
        "interpretation": interpretation,
        "legs": legs,
    }


def main(dry_run: bool = False, output_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written. outcome={outcome}",
              flush=True)
        return outcome, None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (Path(output_dir) if output_dir
               else REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": ts,
        "outcome": outcome,
        "sleep_driver_pattern": "N/A (no sleep loop; forced-contact WL channel-write microdiagnostic)",
        "substrate": "SD-057 object-bound incentive-salience layer (use_incentive_token_bank; "
                     "MECH-344/345/346) + SD-015 z_resource encoder. Forced contact, decoupled "
                     "from the scaffolded_sd054_onboarding foraging curriculum (626b-style).",
        "method_note": "Precursor de-risking the full V3-EXQ-514o GAP-2 successor. Proves the "
                       "SD-057 WL channel-write + scoring machinery fires under FORCED contact "
                       "with the info-sourced consumed (liking) tag. 514n sourced the liking "
                       "target via _contacted_resource_type(obs_dict), which reads the "
                       "post-consumption-cleared resource_type_at_agent; the consumed tag lives "
                       "in info[sd049_consumed_type_tag_this_tick] (C4 demonstrates the split).",
        "seeds": [SEED],
        "pre_registered_thresholds": {
            "pc_separation_floor": PC_SEPARATION_FLOOR,
            "min_scored_steps": MIN_SCORED_STEPS,
            "drive_spread_floor": DRIVE_SPREAD_FLOOR,
            "wl_fraction": WL_FRACTION,
            "null_dissoc_ceiling": NULL_DISSOC_CEILING,
            "populate_steps": POPULATE_STEPS,
            "score_steps": SCORE_STEPS,
        },
        "related_queue_ids": ["V3-EXQ-514n", "V3-EXQ-514o", "V3-EXQ-626b"],
    }
    manifest.update(result)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {outcome}", flush=True)
    return outcome, str(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run, output_dir=args.output_dir)
    if not args.dry_run and _manifest_path:
        _raw = str(_outcome).upper()
        emit_outcome(
            outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_manifest_path,
        )
    sys.exit(0 if _outcome == "PASS" else 1)
