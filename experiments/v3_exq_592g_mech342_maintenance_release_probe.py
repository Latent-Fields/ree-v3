#!/opt/local/bin/python3
"""
V3-EXQ-592g: MECH-342 maintenance-time commitment-release validation probe.

SLEEP DRIVER: K=never (SleepLoopManager disabled; experiment does not
exercise sleep aggregation cluster). use_sleep_loop=False default.

EXPERIMENT_PURPOSE: diagnostic. This is not an ecological behavioural run.
It is the controlled state-machine successor to V3-EXQ-592f
(FAIL_NO_RELEASE_AUTHORITY), re-run with the NEW MECH-342 maintenance-time
release substrate ENABLED.

Purpose:
  V3-EXQ-592f confirmed (controlled probe) that the MECH-090 R-c readiness
  conjunction is ADMISSION-ONLY: with beta forced elevated + E3 committed
  pointer present + degraded readiness sustained, state-occupancy suppression
  and decommit transitions were exactly ZERO. The MECH-090 release-path audit
  (B1 ruled out) + the motor-cessation lit-pull (verdict B3b) routed a NEW
  substrate -- MECH-342 -- a graded, bounded-accumulation, hysteretic release
  of an already-elevated beta latch driven by the SAME two R-c readiness
  signals (score_margin decisiveness + nav_competence). 592g verifies the new
  coupling now produces the suppression + decommit transitions that 592f
  measured as zero, with no false abort under healthy readiness.

Harness:
  Identical to 592f -- real REEAgent.select_action(), real BetaGate, real
  CommitReadiness, real MECH-342 CommitMaintenanceRelease; stub only
  E3TrajectorySelector.select() so controlled SelectionResult objects force
  result.committed=True and controlled per-candidate score margins. ADDED:
  the MECH-342 release branch reads decisiveness from agent.e3.last_scores
  (which the stub does NOT update), so each tick sets agent.e3.last_scores to
  the same controlled score tensor the stub returns -- this exercises the
  decisiveness axis under control.

Stages (forced committed entry, sustained STAGE_TICKS ticks each):
  A baseline pass: score margin above floor, nav readiness above floor.
    Expect NO release (healthy readiness -> the premature-abort guard holds).
  B score failure: score margin below floor, nav readiness above floor.
    Expect a decommit (the decisiveness axis drives release).
  C nav failure: score margin above floor, nav readiness below floor.
    Expect a decommit (the nav_competence axis drives release).
  D both fail: score margin below floor, nav readiness below floor.
    Expect a decommit; d_supp strictly POSITIVE (592f C4 vacuity hardening).
  E recovery: after D, restore both predicates above floor. Expect the agent
    to re-commit (admission re-elevates) and stay committed (no false abort).

INTERPRETATION GRID:
  PASS (C1..C5 with positive decommit in B/C/D, C6 inputs crossed, MECH-342
    fired in the fail stages): MECH-342 maintenance-time release validated.
    Clears the V3-EXQ-592f reach gap; lets governance flip the 592f manifest
    pending_retest_after_substrate to false.
  FAIL_NO_MAINTENANCE_RELEASE with mech342_n_fires == 0 in the fail stages:
    the accumulator never crossed the bound -> diagnose accumulator wiring /
    signal plumbing (e3.last_scores not reaching the branch, or
    commit_readiness not degrading) via /diagnose-errors.
  FAIL_NO_MAINTENANCE_RELEASE with mech342_n_fires >= 1 but no decommit
    transition: the regulator fired but the release-branch state clearing
    (beta_gate.release / e3._committed_trajectory) did not take -> diagnose
    the select_action release branch.
  INVALID_HARNESS_INPUTS: forced inputs did not cross the pre-registered
    thresholds -> claim_ids suppressed, non_contributory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.predictors.e3_selector import SelectionResult  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_592g_mech342_maintenance_release_probe"
QUEUE_ID = "V3-EXQ-592g"
CLAIM_IDS = ["MECH-342"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "K=never"

SEED = 42
ACTION_DIM = 4
SELF_DIM = 8
WORLD_DIM = 8
BODY_OBS_DIM = 4
WORLD_OBS_DIM = 8
TRAJECTORY_HORIZON = 3
# 10 ticks per stage: the MECH-342 accumulator (accumulation_rate 0.2,
# release_bound 1.0) fires after ~5 sustained max-deficit ticks; 10 leaves
# headroom for the post-release suppression to register and for sub-maximal
# deficits to still cross within the stage.
STAGE_TICKS = 10
TOTAL_STAGES = 5

SCORE_MARGIN_FLOOR = 0.05
NAV_READINESS_FLOOR = 0.30
PASS_SCORE_MARGIN = 0.10
FAIL_SCORE_MARGIN = 0.01
PASS_NAV_READINESS = 1.00
FAIL_NAV_READINESS = 0.00

# MECH-342 substrate knobs (passed explicitly for clarity; these are the
# REEConfig defaults).
MR_SCORE_MARGIN_FLOOR = 0.05
MR_SCORE_MARGIN_REENGAGE = 0.10
MR_NAV_FLOOR = 0.30
MR_NAV_REENGAGE = 0.50
MR_ACCUMULATION_RATE = 0.20
MR_LEAK_RATE = 0.10
MR_RELEASE_BOUND = 1.00
MR_PRESSURE_CAP = 1.50

BASELINE_MIN_OCCUPANCY = 0.80
DECOMMIT_REQUIRED = 1  # >=1 decommit transition required in each fail stage

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"


STAGES = [
    {
        "id": "A_forced_committed_baseline",
        "label": "baseline",
        "score_margin": PASS_SCORE_MARGIN,
        "nav_readiness": PASS_NAV_READINESS,
        "fresh_agent": True,
        "force_already_committed": True,
        "expect_release": False,
    },
    {
        "id": "B_score_margin_failure_while_committed",
        "label": "score_fail",
        "score_margin": FAIL_SCORE_MARGIN,
        "nav_readiness": PASS_NAV_READINESS,
        "fresh_agent": True,
        "force_already_committed": True,
        "expect_release": True,
    },
    {
        "id": "C_nav_competence_failure_while_committed",
        "label": "nav_fail",
        "score_margin": PASS_SCORE_MARGIN,
        "nav_readiness": FAIL_NAV_READINESS,
        "fresh_agent": True,
        "force_already_committed": True,
        "expect_release": True,
    },
    {
        "id": "D_both_gates_fail_while_committed",
        "label": "both_fail",
        "score_margin": FAIL_SCORE_MARGIN,
        "nav_readiness": FAIL_NAV_READINESS,
        "fresh_agent": True,
        "force_already_committed": True,
        "expect_release": True,
    },
    {
        "id": "E_recovery",
        "label": "recovery",
        "score_margin": PASS_SCORE_MARGIN,
        "nav_readiness": PASS_NAV_READINESS,
        "fresh_agent": False,
        "force_already_committed": False,
        "expect_release": False,
    },
]


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_agent() -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        # MECH-090 admission conjunction (so admission gates block re-commit
        # under degraded readiness after MECH-342 releases -- clean suppression).
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=NAV_READINESS_FLOOR,
        commit_readiness_initial=PASS_NAV_READINESS,
        # MECH-342 maintenance-time release substrate (under test).
        use_maintenance_release=True,
        maintenance_release_score_margin_floor=MR_SCORE_MARGIN_FLOOR,
        maintenance_release_score_margin_reengage=MR_SCORE_MARGIN_REENGAGE,
        maintenance_release_nav_floor=MR_NAV_FLOOR,
        maintenance_release_nav_reengage=MR_NAV_REENGAGE,
        maintenance_release_accumulation_rate=MR_ACCUMULATION_RATE,
        maintenance_release_leak_rate=MR_LEAK_RATE,
        maintenance_release_bound=MR_RELEASE_BOUND,
        maintenance_release_pressure_cap=MR_PRESSURE_CAP,
        use_sleep_loop=False,
        sws_enabled=False,
        rem_enabled=False,
        use_sleep_aggregation_cluster=False,
    )
    cfg.heartbeat.beta_gate_bistable = True
    cfg.heartbeat.use_commit_readiness_gate = True
    cfg.heartbeat.commit_readiness_floor = SCORE_MARGIN_FLOOR
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def one_hot_action(action_idx: int) -> torch.Tensor:
    action = torch.zeros(1, TRAJECTORY_HORIZON, ACTION_DIM)
    action[:, :, action_idx] = 1.0
    return action


def make_trajectory(action_idx: int, offset: float) -> Trajectory:
    states = [
        torch.full((1, SELF_DIM), offset + 0.01 * i)
        for i in range(TRAJECTORY_HORIZON + 1)
    ]
    world_states = [
        torch.full((1, WORLD_DIM), offset + 0.02 * i)
        for i in range(TRAJECTORY_HORIZON + 1)
    ]
    return Trajectory(
        states=states,
        actions=one_hot_action(action_idx),
        world_states=world_states,
    )


def make_candidates() -> List[Trajectory]:
    return [
        make_trajectory(action_idx=1, offset=0.1),
        make_trajectory(action_idx=2, offset=0.2),
    ]


def scores_for_margin(margin: float) -> torch.Tensor:
    # REE lower-is-better: winner 0.0, runner-up = margin.
    return torch.tensor([0.0, float(margin)], dtype=torch.float32)


def score_margin(scores: torch.Tensor) -> Optional[float]:
    flat = scores.detach().float().reshape(-1)
    if flat.numel() < 2:
        return None
    sorted_scores, _ = torch.sort(flat)
    return float(sorted_scores[1].item() - sorted_scores[0].item())


class ControlledSelector:
    """Stub E3 selection while preserving REEAgent.select_action semantics."""

    def __init__(self) -> None:
        self.scores = scores_for_margin(PASS_SCORE_MARGIN)
        self.committed = True
        self.n_calls = 0
        self.last_result: Optional[SelectionResult] = None

    def set_inputs(self, margin: float, committed: bool = True) -> None:
        self.scores = scores_for_margin(margin)
        self.committed = bool(committed)

    def select(self, candidates, temperature: float = 1.0, **kwargs) -> SelectionResult:
        del temperature, kwargs
        selected = candidates[0]
        result = SelectionResult(
            selected_trajectory=selected,
            selected_index=0,
            selected_action=selected.actions[:, 0, :],
            scores=self.scores.clone(),
            precision=1.0,
            committed=self.committed,
            log_prob=torch.tensor(0.0),
            urgency=0.0,
        )
        self.n_calls += 1
        self.last_result = result
        return result


def force_committed_state(agent: REEAgent, trajectory: Trajectory) -> None:
    agent.beta_gate.elevate()
    agent.e3._committed_trajectory = trajectory
    agent.e3._running_variance = 0.0
    agent._committed_step_idx = 0
    if agent.commit_readiness is not None:
        agent.commit_readiness.notify_outcome(PASS_NAV_READINESS)
    # MECH-342: start each forced commitment with zero release pressure
    # (mirrors the commit-entry reset_pressure in the real agent loop, which
    # is bypassed here because we force the elevated state directly).
    if agent.maintenance_release is not None:
        agent.maintenance_release.reset_pressure()


def counter_state(agent: REEAgent) -> Dict[str, int]:
    beta = agent.beta_gate.get_state()
    ready = agent.commit_readiness.get_state() if agent.commit_readiness is not None else {}
    mr = agent.maintenance_release.get_state() if agent.maintenance_release is not None else {}
    return {
        "score_blocks": int(beta.get("mech090_n_elevation_blocked", 0)),
        "score_admits": int(beta.get("mech090_n_elevation_admitted", 0)),
        "score_single_candidate": int(beta.get("mech090_n_elevation_single_candidate", 0)),
        "policy_hold_count": int(beta.get("hold_count", 0)),
        "policy_propagation_count": int(beta.get("propagation_count", 0)),
        "nav_blocks": int(ready.get("n_blocks_emitted", 0)),
        "nav_updates": int(ready.get("n_updates", 0)),
        "mech342_fires": int(mr.get("mech342_n_fires", 0)),
        "mech342_accumulate": int(mr.get("mech342_n_accumulate", 0)),
        "mech342_leak": int(mr.get("mech342_n_leak", 0)),
        "mech342_hold": int(mr.get("mech342_n_hold", 0)),
        "mech342_ticks": int(mr.get("mech342_n_ticks", 0)),
    }


def delta_counts(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in after}


def bool_fraction(values: List[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v) / len(values))


def run_stage(
    agent: REEAgent,
    selector: ControlledSelector,
    candidates: List[Trajectory],
    stage: Dict,
) -> Tuple[REEAgent, Dict]:
    selector.set_inputs(float(stage["score_margin"]), committed=True)
    if agent.commit_readiness is not None:
        agent.commit_readiness.notify_outcome(float(stage["nav_readiness"]))

    before = counter_state(agent)
    beta_values: List[bool] = []
    e3_pointer_values: List[bool] = []
    result_committed_values: List[bool] = []
    score_margins: List[Optional[float]] = []
    readiness_values: List[float] = []
    pressure_values: List[float] = []
    actions: List[int] = []

    beta_release_count = 0
    beta_reentry_count = 0
    e3_pointer_drop_count = 0
    step_index_reset_count = 0

    prev_beta = bool(agent.beta_gate.is_elevated)
    prev_pointer = agent.e3._committed_trajectory is not None
    prev_step_idx = int(getattr(agent, "_committed_step_idx", 0))

    forced_scores = scores_for_margin(float(stage["score_margin"]))

    for _tick in range(STAGE_TICKS):
        if agent.commit_readiness is not None:
            agent.commit_readiness.notify_outcome(float(stage["nav_readiness"]))
        # MECH-342 reads decisiveness from agent.e3.last_scores (NOT
        # result.scores); the stub does not update it, so set it directly to
        # the controlled margin BEFORE select_action exercises the branch.
        agent.e3.last_scores = forced_scores.clone()

        action = agent.select_action(candidates, {"e3_tick": True})
        result = selector.last_result
        margin = score_margin(result.scores) if result is not None else None
        readiness = (
            float(agent.commit_readiness.get_readiness())
            if agent.commit_readiness is not None
            else 1.0
        )
        pressure = (
            float(agent.maintenance_release.get_pressure())
            if agent.maintenance_release is not None
            else 0.0
        )
        beta_now = bool(agent.beta_gate.is_elevated)
        pointer_now = agent.e3._committed_trajectory is not None
        step_now = int(getattr(agent, "_committed_step_idx", 0))

        if prev_beta and not beta_now:
            beta_release_count += 1
        if (not prev_beta) and beta_now:
            beta_reentry_count += 1
        if prev_pointer and not pointer_now:
            e3_pointer_drop_count += 1
        if step_now < prev_step_idx:
            step_index_reset_count += 1

        beta_values.append(beta_now)
        e3_pointer_values.append(pointer_now)
        result_committed_values.append(bool(result.committed) if result is not None else False)
        score_margins.append(margin)
        readiness_values.append(readiness)
        pressure_values.append(pressure)
        actions.append(int(action.argmax(dim=-1).item()))

        prev_beta = beta_now
        prev_pointer = pointer_now
        prev_step_idx = step_now

    after = counter_state(agent)
    deltas = delta_counts(after, before)

    direct_score_below = sum(
        1 for m in score_margins if m is not None and m < SCORE_MARGIN_FLOOR
    )
    direct_nav_below = sum(1 for r in readiness_values if r < NAV_READINESS_FLOOR)

    metrics = {
        "stage_id": stage["id"],
        "label": stage["label"],
        "expect_release": bool(stage["expect_release"]),
        "forced_score_margin": float(stage["score_margin"]),
        "forced_nav_readiness": float(stage["nav_readiness"]),
        "ticks": STAGE_TICKS,
        "score_margin_floor": SCORE_MARGIN_FLOOR,
        "nav_readiness_floor": NAV_READINESS_FLOOR,
        "observed_score_margins": score_margins,
        "observed_readiness_values": readiness_values,
        "observed_pressure_values": pressure_values,
        "direct_score_margin_below_floor_count": direct_score_below,
        "direct_nav_readiness_below_floor_count": direct_nav_below,
        "official_gate_counters_delta": deltas,
        "mech342_fires_delta": deltas["mech342_fires"],
        "mech342_accumulate_delta": deltas["mech342_accumulate"],
        "state_occupancy": {
            "beta_elevated_fraction": bool_fraction(beta_values),
            "e3_committed_pointer_fraction": bool_fraction(e3_pointer_values),
            "result_committed_fraction": bool_fraction(result_committed_values),
            "policy_hold_count_delta": deltas["policy_hold_count"],
            "policy_propagation_count_delta": deltas["policy_propagation_count"],
        },
        "transition_counts": {
            "beta_true_to_false_release_count": beta_release_count,
            "beta_false_to_true_reentry_count": beta_reentry_count,
            "e3_pointer_true_to_false_drop_count": e3_pointer_drop_count,
            "step_index_reset_count": step_index_reset_count,
        },
        "decommit_transition_count": beta_release_count + e3_pointer_drop_count,
        "selected_action_classes": actions,
        "final_beta_elevated": bool(agent.beta_gate.is_elevated),
        "final_e3_committed_pointer_present": agent.e3._committed_trajectory is not None,
    }
    return agent, metrics


def suppression_from_baseline(baseline: Dict, stage: Dict) -> Dict[str, float]:
    base_state = baseline["state_occupancy"]
    stage_state = stage["state_occupancy"]
    beta_drop = float(
        base_state["beta_elevated_fraction"] - stage_state["beta_elevated_fraction"]
    )
    e3_drop = float(
        base_state["e3_committed_pointer_fraction"]
        - stage_state["e3_committed_pointer_fraction"]
    )
    return {
        "beta_drop": beta_drop,
        "e3_pointer_drop": e3_drop,
        "max_drop": max(beta_drop, e3_drop),
    }


def evaluate_acceptance(stage_metrics: Dict[str, Dict]) -> Tuple[Dict, str, str]:
    a = stage_metrics["A_forced_committed_baseline"]
    b = stage_metrics["B_score_margin_failure_while_committed"]
    c = stage_metrics["C_nav_competence_failure_while_committed"]
    d = stage_metrics["D_both_gates_fail_while_committed"]
    e = stage_metrics["E_recovery"]

    base_beta = a["state_occupancy"]["beta_elevated_fraction"]
    base_e3 = a["state_occupancy"]["e3_committed_pointer_fraction"]
    b_supp = suppression_from_baseline(a, b)
    c_supp = suppression_from_baseline(a, c)
    d_supp = suppression_from_baseline(a, d)

    b_decommit = b["decommit_transition_count"]
    c_decommit = c["decommit_transition_count"]
    d_decommit = d["decommit_transition_count"]

    a_decommit = a["decommit_transition_count"]

    # C1: forced baseline genuinely committed at stage start.
    c1 = base_beta >= BASELINE_MIN_OCCUPANCY and base_e3 >= BASELINE_MIN_OCCUPANCY

    # C2: decisiveness axis has RELEASE authority -- stage B (score margin
    # below floor, nav healthy) produces >=1 decommit transition. This is the
    # quantity V3-EXQ-592f measured as ZERO.
    c2 = b["direct_score_margin_below_floor_count"] > 0 and b_decommit >= DECOMMIT_REQUIRED

    # C3: nav_competence axis has RELEASE authority -- stage C (nav below
    # floor, margin healthy) produces >=1 decommit transition.
    c3 = c["direct_nav_readiness_below_floor_count"] > 0 and c_decommit >= DECOMMIT_REQUIRED

    # C4: conjunction -- stage D (both fail) produces a decommit, with d_supp
    # STRICTLY POSITIVE (heeds the 592f C4 vacuous-PASS note: require
    # max_drop > 0 AND >= max(b_supp, c_supp), not 0 >= max(0, 0)).
    c4 = (
        d["direct_score_margin_below_floor_count"] > 0
        and d["direct_nav_readiness_below_floor_count"] > 0
        and d_decommit >= DECOMMIT_REQUIRED
        and d_supp["max_drop"] > 0.0
        and d_supp["max_drop"] >= max(b_supp["max_drop"], c_supp["max_drop"])
    )

    # C5: no false abort -- the healthy baseline stage A produces ZERO decommit
    # transitions (the premature-abort guard holds), and the recovery stage E
    # ends committed (admission re-elevated after the degraded stages).
    c5 = a_decommit == 0 and e["final_beta_elevated"]

    # C6: no-vacuity -- forced inputs actually crossed the pre-registered
    # thresholds where intended, AND the MECH-342 accumulator actually fired
    # in each release stage (rules out a release produced by some OTHER
    # pathway rather than the substrate under test).
    c6_parts = {
        "A_score_above_floor": a["direct_score_margin_below_floor_count"] == 0,
        "A_nav_above_floor": a["direct_nav_readiness_below_floor_count"] == 0,
        "B_score_below_floor": b["direct_score_margin_below_floor_count"] > 0,
        "C_nav_below_floor": c["direct_nav_readiness_below_floor_count"] > 0,
        "D_score_below_floor": d["direct_score_margin_below_floor_count"] > 0,
        "D_nav_below_floor": d["direct_nav_readiness_below_floor_count"] > 0,
        "E_score_above_floor": e["direct_score_margin_below_floor_count"] == 0,
        "E_nav_above_floor": e["direct_nav_readiness_below_floor_count"] == 0,
        "B_mech342_fired": b["mech342_fires_delta"] >= 1,
        "C_mech342_fired": c["mech342_fires_delta"] >= 1,
        "D_mech342_fired": d["mech342_fires_delta"] >= 1,
        "A_mech342_silent": a["mech342_fires_delta"] == 0,
    }
    c6 = all(c6_parts.values())

    acceptance = {
        "C1_forced_baseline": {
            "pass": c1,
            "baseline_beta_elevated_fraction": base_beta,
            "baseline_e3_committed_pointer_fraction": base_e3,
            "threshold": BASELINE_MIN_OCCUPANCY,
        },
        "C2_score_margin_release_authority": {
            "pass": c2,
            "direct_score_margin_below_floor_count": b["direct_score_margin_below_floor_count"],
            "suppression": b_supp,
            "decommit_transition_count": b_decommit,
            "mech342_fires": b["mech342_fires_delta"],
        },
        "C3_nav_competence_release_authority": {
            "pass": c3,
            "direct_nav_readiness_below_floor_count": c["direct_nav_readiness_below_floor_count"],
            "nav_competence_blocks": c["official_gate_counters_delta"]["nav_blocks"],
            "suppression": c_supp,
            "decommit_transition_count": c_decommit,
            "mech342_fires": c["mech342_fires_delta"],
        },
        "C4_conjunction_authority": {
            "pass": c4,
            "direct_score_margin_below_floor_count": d["direct_score_margin_below_floor_count"],
            "direct_nav_readiness_below_floor_count": d["direct_nav_readiness_below_floor_count"],
            "suppression": d_supp,
            "decommit_transition_count": d_decommit,
            "mech342_fires": d["mech342_fires_delta"],
            "strongest_single_gate_suppression": max(b_supp["max_drop"], c_supp["max_drop"]),
            "requires_strictly_positive_d_supp": True,
        },
        "C5_no_false_abort": {
            "pass": c5,
            "baseline_decommit_transition_count": a_decommit,
            "recovery_final_beta_elevated": e["final_beta_elevated"],
        },
        "C6_no_vacuity": {
            "pass": c6,
            "parts": c6_parts,
        },
    }

    if not c6_parts["A_score_above_floor"] or not c6_parts["A_nav_above_floor"] or not (
        c6_parts["B_score_below_floor"]
        and c6_parts["C_nav_below_floor"]
        and c6_parts["D_score_below_floor"]
        and c6_parts["D_nav_below_floor"]
        and c6_parts["E_score_above_floor"]
        and c6_parts["E_nav_above_floor"]
    ):
        # Forced inputs themselves did not cross thresholds -> harness invalid.
        return acceptance, "FAIL", "INVALID_HARNESS_INPUTS"
    if c1 and c2 and c3 and c4 and c5 and c6:
        return acceptance, "PASS", "PASS_MAINTENANCE_RELEASE"
    return acceptance, "FAIL", "FAIL_NO_MAINTENANCE_RELEASE"


def run_experiment(dry_run: bool = False) -> Tuple[Dict, Path]:
    set_seed(SEED)
    candidates = make_candidates()
    selector: Optional[ControlledSelector] = None
    agent: Optional[REEAgent] = None
    stage_metrics: Dict[str, Dict] = {}

    print(f"Seed {SEED} Condition controlled_state_transition", flush=True)
    for stage_idx, stage in enumerate(STAGES, start=1):
        if stage["fresh_agent"] or agent is None or selector is None:
            agent = make_agent()
            selector = ControlledSelector()
            agent.e3.select = selector.select
        if stage["force_already_committed"]:
            force_committed_state(agent, candidates[0])

        agent, metrics = run_stage(agent, selector, candidates, stage)
        stage_metrics[stage["id"]] = metrics
        print(
            f"  [train] diagnostic seed={SEED} ep {stage_idx}/{TOTAL_STAGES} "
            f"stage={stage['label']} beta={metrics['state_occupancy']['beta_elevated_fraction']:.3f} "
            f"e3_pointer={metrics['state_occupancy']['e3_committed_pointer_fraction']:.3f} "
            f"decommit={metrics['decommit_transition_count']} "
            f"mr_fires={metrics['mech342_fires_delta']} "
            f"score_below={metrics['direct_score_margin_below_floor_count']} "
            f"nav_below={metrics['direct_nav_readiness_below_floor_count']}",
            flush=True,
        )

    acceptance, outcome, diagnostic_outcome = evaluate_acceptance(stage_metrics)

    claim_ids = CLAIM_IDS if diagnostic_outcome != "INVALID_HARNESS_INPUTS" else []
    if outcome == "PASS":
        evidence_direction = "supports"
        evidence_note = (
            "Controlled state-machine probe with the MECH-342 maintenance-time "
            "release substrate ENABLED: degraded execution readiness "
            "(score_margin and/or nav_competence) sustained while beta-elevated "
            "now produces >=1 decommit transition per fail stage (the quantity "
            "V3-EXQ-592f measured as zero), with no false abort under healthy "
            "readiness and re-commit on recovery. Validates the MECH-342 "
            "release-side coupling; clears the 592f reach gap."
        )
    elif diagnostic_outcome == "INVALID_HARNESS_INPUTS":
        evidence_direction = "non_contributory"
        evidence_note = (
            "Harness invalid: forced inputs did not cross pre-registered "
            "score/readiness thresholds, so the run is non-contributory."
        )
    else:
        evidence_direction = "weakens"
        evidence_note = (
            "MECH-342 substrate ENABLED but degraded readiness while "
            "beta-elevated did not produce the expected decommit transitions "
            "(see C6 parts: mech342 fires vs decommit). Diagnose per the "
            "interpretation grid -- no fires => accumulator wiring / signal "
            "plumbing; fires-but-no-decommit => release-branch state clearing."
        )

    timestamp = utc_stamp()
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    manifest = {
        "schema_version": "experiment_result/v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": timestamp,
        "machine": platform.node(),
        "dry_run": bool(dry_run),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "claim_ids": claim_ids,
        "outcome": outcome,
        "diagnostic_outcome": diagnostic_outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": (
            {claim_id: evidence_direction for claim_id in claim_ids}
            if len(claim_ids) > 1
            else {}
        ),
        "evidence_direction_note": evidence_note,
        "thresholds": {
            "score_margin_floor": SCORE_MARGIN_FLOOR,
            "nav_readiness_floor": NAV_READINESS_FLOOR,
            "pass_score_margin": PASS_SCORE_MARGIN,
            "fail_score_margin": FAIL_SCORE_MARGIN,
            "pass_nav_readiness": PASS_NAV_READINESS,
            "fail_nav_readiness": FAIL_NAV_READINESS,
            "baseline_min_occupancy": BASELINE_MIN_OCCUPANCY,
            "decommit_required": DECOMMIT_REQUIRED,
            "mech342": {
                "score_margin_floor": MR_SCORE_MARGIN_FLOOR,
                "score_margin_reengage": MR_SCORE_MARGIN_REENGAGE,
                "nav_floor": MR_NAV_FLOOR,
                "nav_reengage": MR_NAV_REENGAGE,
                "accumulation_rate": MR_ACCUMULATION_RATE,
                "leak_rate": MR_LEAK_RATE,
                "release_bound": MR_RELEASE_BOUND,
                "pressure_cap": MR_PRESSURE_CAP,
            },
        },
        "direct_gate_inputs": {
            stage_id: {
                "forced_score_margin": metrics["forced_score_margin"],
                "forced_nav_readiness": metrics["forced_nav_readiness"],
                "observed_score_margins": metrics["observed_score_margins"],
                "observed_readiness_values": metrics["observed_readiness_values"],
                "observed_pressure_values": metrics["observed_pressure_values"],
                "direct_score_margin_below_floor_count": metrics[
                    "direct_score_margin_below_floor_count"
                ],
                "direct_nav_readiness_below_floor_count": metrics[
                    "direct_nav_readiness_below_floor_count"
                ],
            }
            for stage_id, metrics in stage_metrics.items()
        },
        "official_gate_counters": {
            stage_id: metrics["official_gate_counters_delta"]
            for stage_id, metrics in stage_metrics.items()
        },
        "state_occupancy": {
            stage_id: metrics["state_occupancy"]
            for stage_id, metrics in stage_metrics.items()
        },
        "transition_counts": {
            stage_id: metrics["transition_counts"]
            for stage_id, metrics in stage_metrics.items()
        },
        "mech342_fires_by_stage": {
            stage_id: metrics["mech342_fires_delta"]
            for stage_id, metrics in stage_metrics.items()
        },
        "stage_metrics": stage_metrics,
        "acceptance": acceptance,
        "vacuity_checks": acceptance["C6_no_vacuity"],
        "diagnostic_interpretation": {
            "summary": diagnostic_outcome,
            "failure_routing": (
                "diagnose MECH-342 accumulator wiring (no fires) or release-branch "
                "state clearing (fires but no decommit) per the docstring grid"
                if diagnostic_outcome == "FAIL_NO_MAINTENANCE_RELEASE"
                else "none"
            ),
            "uses_real_select_action": True,
            "uses_real_beta_gate": True,
            "uses_real_commit_readiness": True,
            "uses_real_maintenance_release": True,
            "stubbed_component": "E3TrajectorySelector.select SelectionResult only",
        },
        "notes": [
            "592g is the MECH-342 maintenance-release validation successor to "
            "the V3-EXQ-592f gap probe (FAIL_NO_RELEASE_AUTHORITY).",
            "Each tick sets agent.e3.last_scores to the controlled margin so the "
            "MECH-342 decisiveness axis (which reads last_scores, not the stub's "
            "result.scores) is exercised under control.",
            "PASS requires positive decommit transitions in the fail stages "
            "(zero in 592f), no false abort under healthy readiness, and the "
            "MECH-342 accumulator firing in each fail stage (C6).",
            "On PASS, governance may flip the V3-EXQ-592f manifest's "
            "pending_retest_after_substrate to false (the reach gap is closed).",
        ],
    }

    out_dir = Path(tempfile.gettempdir()) / "ree_v3_dry_runs" if dry_run else EVIDENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"verdict: {outcome}", flush=True)
    print(f"Experiment: {outcome}", flush=True)
    print(f"Saved manifest: {out_path}", flush=True)
    return manifest, out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Write manifest under /tmp.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest, out_path = run_experiment(dry_run=args.dry_run)
    signal_dir = None
    if args.dry_run:
        signal_dir = Path(tempfile.gettempdir()) / "ree_runner_signals"
    emit_outcome(
        outcome=str(manifest["outcome"]).upper(),
        manifest_path=out_path,
        run_id=str(manifest["run_id"]),
        queue_id=QUEUE_ID,
        exit_reason="ok" if manifest["outcome"] == "PASS" else "fail",
        signal_dir=signal_dir,
    )
