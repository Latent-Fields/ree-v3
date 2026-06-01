#!/opt/local/bin/python3
"""
V3-EXQ-592f: MECH-090 commitment-state transition authority probe.

SLEEP DRIVER: K=never (SleepLoopManager disabled; experiment does not
exercise sleep aggregation cluster). use_sleep_loop=False default.

EXPERIMENT_PURPOSE: diagnostic. This is not an ecological behavioural run.
It is a controlled state-machine probe following
failure_autopsy_V3-EXQ-592e_2026-06-01.

Purpose:
  V3-EXQ-592e showed that MECH-090 score-margin and nav-readiness predicates
  fire, but did not demonstrate authority over an already-held committed
  state. In bistable mode, score-margin admission is only consulted when beta
  is not elevated; nav-readiness can increment block counters while beta is
  elevated, but is only AND-composed into the elevation call.

Harness:
  Use real REEAgent.select_action(), real BetaGate, and real CommitReadiness.
  Stub only E3TrajectorySelector.select() so that controlled SelectionResult
  objects force result.committed=True and controlled per-candidate score
  margins. Before perturbation, beta is elevated and E3's committed pointer is
  present. The stub does not refresh that pointer each tick; if the real
  select_action path clears it, the drop is observable.

Stages:
  A baseline pass: score margin above floor, nav readiness above floor.
  B score failure: score margin below floor, nav readiness above floor.
  C nav failure: score margin above floor, nav readiness below floor.
  D both fail: score margin below floor, nav readiness below floor.
  E recovery: after D, restore both predicates above floor.

PASS requires direct suppression or release of already-held beta/E3
commitment state under failed readiness, plus recovery after readiness returns.
If forced inputs do not cross thresholds, the diagnostic outcome is INVALID
and claim_ids are suppressed in the manifest.
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


EXPERIMENT_TYPE = "v3_exq_592f_mech090_commitment_state_transition_probe"
QUEUE_ID = "V3-EXQ-592f"
CLAIM_IDS = ["MECH-090"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "V3-EXQ-592e"
SLEEP_DRIVER_PATTERN = "K=never"

SEED = 42
ACTION_DIM = 4
SELF_DIM = 8
WORLD_DIM = 8
BODY_OBS_DIM = 4
WORLD_OBS_DIM = 8
TRAJECTORY_HORIZON = 3
STAGE_TICKS = 6
TOTAL_STAGES = 5

SCORE_MARGIN_FLOOR = 0.05
NAV_READINESS_FLOOR = 0.30
PASS_SCORE_MARGIN = 0.10
FAIL_SCORE_MARGIN = 0.01
PASS_NAV_READINESS = 1.00
FAIL_NAV_READINESS = 0.00

BASELINE_MIN_OCCUPANCY = 0.80
SUPPRESSION_DROP_THRESHOLD = 0.50
RECOVERY_FRACTION_OF_BASELINE = 0.80

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"


STAGES = [
    {
        "id": "A_forced_committed_baseline",
        "label": "baseline",
        "score_margin": PASS_SCORE_MARGIN,
        "nav_readiness": PASS_NAV_READINESS,
        "fresh_agent": True,
        "force_already_committed": True,
    },
    {
        "id": "B_score_margin_failure_while_committed",
        "label": "score_fail",
        "score_margin": FAIL_SCORE_MARGIN,
        "nav_readiness": PASS_NAV_READINESS,
        "fresh_agent": True,
        "force_already_committed": True,
    },
    {
        "id": "C_nav_competence_failure_while_committed",
        "label": "nav_fail",
        "score_margin": PASS_SCORE_MARGIN,
        "nav_readiness": FAIL_NAV_READINESS,
        "fresh_agent": True,
        "force_already_committed": True,
    },
    {
        "id": "D_both_gates_fail_while_committed",
        "label": "both_fail",
        "score_margin": FAIL_SCORE_MARGIN,
        "nav_readiness": FAIL_NAV_READINESS,
        "fresh_agent": True,
        "force_already_committed": True,
    },
    {
        "id": "E_recovery",
        "label": "recovery",
        "score_margin": PASS_SCORE_MARGIN,
        "nav_readiness": PASS_NAV_READINESS,
        "fresh_agent": False,
        "force_already_committed": False,
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
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=NAV_READINESS_FLOOR,
        commit_readiness_initial=PASS_NAV_READINESS,
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

    def select(self, candidates: List[Trajectory], temperature: float = 1.0, **kwargs) -> SelectionResult:
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


def counter_state(agent: REEAgent) -> Dict[str, int]:
    beta = agent.beta_gate.get_state()
    ready = agent.commit_readiness.get_state() if agent.commit_readiness is not None else {}
    return {
        "score_blocks": int(beta.get("mech090_n_elevation_blocked", 0)),
        "score_admits": int(beta.get("mech090_n_elevation_admitted", 0)),
        "score_single_candidate": int(beta.get("mech090_n_elevation_single_candidate", 0)),
        "policy_hold_count": int(beta.get("hold_count", 0)),
        "policy_propagation_count": int(beta.get("propagation_count", 0)),
        "nav_blocks": int(ready.get("n_blocks_emitted", 0)),
        "nav_updates": int(ready.get("n_updates", 0)),
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
    actions: List[int] = []

    beta_release_count = 0
    beta_reentry_count = 0
    e3_pointer_drop_count = 0
    step_index_reset_count = 0

    prev_beta = bool(agent.beta_gate.is_elevated)
    prev_pointer = agent.e3._committed_trajectory is not None
    prev_step_idx = int(getattr(agent, "_committed_step_idx", 0))

    for _tick in range(STAGE_TICKS):
        if agent.commit_readiness is not None:
            agent.commit_readiness.notify_outcome(float(stage["nav_readiness"]))

        action = agent.select_action(candidates, {"e3_tick": True})
        result = selector.last_result
        margin = score_margin(result.scores) if result is not None else None
        readiness = (
            float(agent.commit_readiness.get_readiness())
            if agent.commit_readiness is not None
            else 1.0
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
        "forced_score_margin": float(stage["score_margin"]),
        "forced_nav_readiness": float(stage["nav_readiness"]),
        "ticks": STAGE_TICKS,
        "score_margin_floor": SCORE_MARGIN_FLOOR,
        "nav_readiness_floor": NAV_READINESS_FLOOR,
        "observed_score_margins": score_margins,
        "observed_readiness_values": readiness_values,
        "direct_score_margin_below_floor_count": direct_score_below,
        "direct_nav_readiness_below_floor_count": direct_nav_below,
        "official_gate_counters_delta": deltas,
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

    b_decommit = (
        b["transition_counts"]["beta_true_to_false_release_count"]
        + b["transition_counts"]["e3_pointer_true_to_false_drop_count"]
    )
    c_decommit = (
        c["transition_counts"]["beta_true_to_false_release_count"]
        + c["transition_counts"]["e3_pointer_true_to_false_drop_count"]
    )
    d_decommit = (
        d["transition_counts"]["beta_true_to_false_release_count"]
        + d["transition_counts"]["e3_pointer_true_to_false_drop_count"]
    )

    c1 = base_beta >= BASELINE_MIN_OCCUPANCY and base_e3 >= BASELINE_MIN_OCCUPANCY
    c2 = (
        b["direct_score_margin_below_floor_count"] > 0
        and (
            b_supp["beta_drop"] >= SUPPRESSION_DROP_THRESHOLD
            or b_supp["e3_pointer_drop"] >= SUPPRESSION_DROP_THRESHOLD
            or b_decommit >= 1
        )
    )
    c3 = (
        c["official_gate_counters_delta"]["nav_blocks"] > 0
        and (
            c_supp["beta_drop"] >= SUPPRESSION_DROP_THRESHOLD
            or c_supp["e3_pointer_drop"] >= SUPPRESSION_DROP_THRESHOLD
            or c_decommit >= 1
        )
    )
    c4 = (
        d["direct_score_margin_below_floor_count"] > 0
        and d["direct_nav_readiness_below_floor_count"] > 0
        and d_supp["max_drop"] >= max(b_supp["max_drop"], c_supp["max_drop"])
    )
    prior_suppression = max(b_supp["max_drop"], c_supp["max_drop"], d_supp["max_drop"]) >= SUPPRESSION_DROP_THRESHOLD or max(
        b_decommit, c_decommit, d_decommit
    ) >= 1
    c5 = (
        prior_suppression
        and e["state_occupancy"]["beta_elevated_fraction"]
        >= base_beta * RECOVERY_FRACTION_OF_BASELINE
    )

    c6_parts = {
        "A_score_above_floor": a["direct_score_margin_below_floor_count"] == 0,
        "A_nav_above_floor": a["direct_nav_readiness_below_floor_count"] == 0,
        "B_score_below_floor": b["direct_score_margin_below_floor_count"] > 0,
        "C_nav_below_floor": c["direct_nav_readiness_below_floor_count"] > 0,
        "D_score_below_floor": d["direct_score_margin_below_floor_count"] > 0,
        "D_nav_below_floor": d["direct_nav_readiness_below_floor_count"] > 0,
        "E_score_above_floor": e["direct_score_margin_below_floor_count"] == 0,
        "E_nav_above_floor": e["direct_nav_readiness_below_floor_count"] == 0,
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
        },
        "C3_nav_competence_release_authority": {
            "pass": c3,
            "nav_competence_blocks": c["official_gate_counters_delta"]["nav_blocks"],
            "suppression": c_supp,
            "decommit_transition_count": c_decommit,
        },
        "C4_conjunction_authority": {
            "pass": c4,
            "direct_score_margin_below_floor_count": d["direct_score_margin_below_floor_count"],
            "direct_nav_readiness_below_floor_count": d["direct_nav_readiness_below_floor_count"],
            "suppression": d_supp,
            "decommit_transition_count": d_decommit,
            "strongest_single_gate_suppression": max(b_supp["max_drop"], c_supp["max_drop"]),
        },
        "C5_recovery": {
            "pass": c5,
            "prior_suppression_observed": prior_suppression,
            "recovery_beta_elevated_fraction": e["state_occupancy"]["beta_elevated_fraction"],
            "required_fraction_of_baseline": RECOVERY_FRACTION_OF_BASELINE,
            "note": (
                "not_evaluable_no_prior_suppression"
                if not prior_suppression
                else "evaluated"
            ),
        },
        "C6_no_vacuity": {
            "pass": c6,
            "parts": c6_parts,
        },
    }

    if not c6:
        return acceptance, "FAIL", "INVALID_HARNESS_INPUTS"
    if c1 and c2 and c3 and c4 and c5:
        return acceptance, "PASS", "PASS_STATE_AUTHORITY"
    return acceptance, "FAIL", "FAIL_NO_RELEASE_AUTHORITY"


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
            f"score_below={metrics['direct_score_margin_below_floor_count']} "
            f"nav_below={metrics['direct_nav_readiness_below_floor_count']}",
            flush=True,
        )

    acceptance, outcome, diagnostic_outcome = evaluate_acceptance(stage_metrics)

    claim_ids = CLAIM_IDS if diagnostic_outcome != "INVALID_HARNESS_INPUTS" else []
    if outcome == "PASS":
        evidence_direction = "supports"
        evidence_note = (
            "Controlled state-machine probe showed failed MECH-090 readiness "
            "can suppress or release already-held commitment state and recover."
        )
    elif diagnostic_outcome == "INVALID_HARNESS_INPUTS":
        evidence_direction = "non_contributory"
        evidence_note = (
            "Harness invalid: forced inputs did not cross pre-registered "
            "score/readiness thresholds, so the run is non-contributory."
        )
    else:
        evidence_direction = "does_not_support"
        evidence_note = (
            "Forced score/readiness failures crossed thresholds while using "
            "real REEAgent.select_action, BetaGate, and CommitReadiness, but "
            "already-held beta/E3 commitment state did not suppress or release. "
            "This supports the 592e autopsy read that current MECH-090 R-c "
            "integration is admission-side only in bistable mode."
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
        "supersedes": SUPERSEDES,
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
            "suppression_drop_threshold": SUPPRESSION_DROP_THRESHOLD,
            "recovery_fraction_of_baseline": RECOVERY_FRACTION_OF_BASELINE,
        },
        "direct_gate_inputs": {
            stage_id: {
                "forced_score_margin": metrics["forced_score_margin"],
                "forced_nav_readiness": metrics["forced_nav_readiness"],
                "observed_score_margins": metrics["observed_score_margins"],
                "observed_readiness_values": metrics["observed_readiness_values"],
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
        "stage_metrics": stage_metrics,
        "acceptance": acceptance,
        "vacuity_checks": acceptance["C6_no_vacuity"],
        "diagnostic_interpretation": {
            "summary": diagnostic_outcome,
            "failure_routing": (
                "implement-substrate audit for readiness-to-release coupling "
                "before further ecological retests"
                if diagnostic_outcome == "FAIL_NO_RELEASE_AUTHORITY"
                else "none"
            ),
            "uses_real_select_action": True,
            "uses_real_beta_gate": True,
            "uses_real_commit_readiness": True,
            "stubbed_component": "E3TrajectorySelector.select SelectionResult only",
        },
        "notes": [
            "592f is a controlled state-machine probe, not an ecological behavioural run.",
            "The E3 stub preserves the pre-forced committed pointer unless select_action clears it.",
            "PASS requires already-held commitment suppression/release plus recovery.",
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
