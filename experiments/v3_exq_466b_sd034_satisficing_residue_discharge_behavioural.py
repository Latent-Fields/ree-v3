"""V3-EXQ-466b (EXP-0162 behavioural): SD-034 satisficing / residue discharge.

Purpose: evidence. Full-agent-loop behavioural successor to the substrate-
readiness diagnostic V3-EXQ-466 (which hand-poked discharge_domain across
UC1-UC5). This arm exercises the SD-034 ClosureOperator's residue-discharge
signal inside the real committed_mode_curriculum (P0 warmup -> P1 consolidation
-> P2 eval) on a CausalGridWorldV2 with the GAP-3 adaptive tolerance-band
completion primitive, and measures the ocd4 "satisficing / over-checking"
dissociation in a live loop:

  - With the closure operator PRESENT, when the agent tolerance-completes a
    waypoint with a stable rule_state, the ClosureOperator fires and calls
    ResidueField.discharge_domain -- rule-domain residue is multiplicatively
    attenuated. Closure fires accumulate (closure_operator._n_closures > 0)
    and discharge events are counted (discharge_events >= 1).
  - With the closure operator ABSENT (same trained weights), the agent
    accumulates rule-domain residue without discharge -- the non-satisficing
    "over-checking" failure mode SD-034 closes.

Arms (one P0->P1 training run, three P2 evals):

  ARM_EMERGENT_ON     -- emergent-trained agent, closure operator ON. Positive
                         arm. Expect n_closures > 0 and discharge_events >= 1.
  ARM_FORCED_RV_ON    -- O-2 mandatory contrast (committed_mode_curriculum):
                         clone_trained_agent(bistable=True) with running_variance
                         forced to 0.001, closure ON. Isolates whether closure
                         firing and discharge need EMERGENT commitment or merely
                         the committed state.
  ARM_CLOSURE_OFF     -- same trained weights loaded into a closure-OFF agent.
                         Over-checking contrast: zero closure fires, zero residue
                         discharge. Residue accumulates unchecked.

Pre-registered acceptance (PASS = all three criteria per seed, majority of seeds):
  C1  ARM_EMERGENT_ON  n_closures >= 1          (closure fires in a live loop)
  C2  ARM_EMERGENT_ON  discharge_events >= 1    (residue actually discharged)
  C3  ARM_CLOSURE_OFF  n_closures == 0 AND discharge_events == 0
                       (over-checking: closure absent, no discharge)

Interpretation grid (one row per plausible outcome -> next action):
  C1-C3 all PASS .................. SD-034 satisficing/residue-discharge
                                    dissociation confirmed in a live loop;
                                    closure is the discharge pathway.
                                    -> governance: behavioural support for
                                    SD-034 satisficing arm / MECH-094.
  C1 FAIL (n_closures == 0 ON) .... commitment/rule_state never stabilised under
                                    the curriculum at this budget. NOT substrate
                                    falsification -> /diagnose-errors on the
                                    curriculum budget / rule_state seeding
                                    (mirror the V3-EXQ-592 commitment-not-
                                    elicited routing), not a re-run under 466b.
  C1 PASS, C2 FAIL ................ closure fires but no discharge event ->
                                    residue field has no active centers near
                                    the closure z_world (agent may not have
                                    accumulated residue at waypoint locations
                                    within the eval budget) -> increase eval
                                    episodes or confirm residue seeding before
                                    re-running.
  C1 PASS, C3 FAIL (OFF fires) ... discharge fires without the operator ->
                                    SD-034 is over-specification. Route to
                                    governance review, not a re-run.

Run:
  /opt/local/bin/python3 experiments/v3_exq_466b_sd034_satisficing_residue_discharge_behavioural.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_466b_sd034_satisficing_residue_discharge_behavioural.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.committed_mode_curriculum import (  # noqa: E402
    clone_trained_agent,
    run_p0_warmup,
    run_p1_consolidation,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_466b_sd034_satisficing_residue_discharge_behavioural"
QUEUE_ID = "V3-EXQ-466b"
CLAIM_IDS = ["SD-034", "MECH-094"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]

# Pre-registered thresholds (constants, not derived from the run).
C1_MIN_CLOSURES = 1
C2_MIN_DISCHARGE_EVENTS = 1
PASS_FRACTION_REQUIRED = 2.0 / 3.0  # majority of seeds


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_env(size: int) -> CausalGridWorldV2:
    """Target env with GAP-3 tolerance-band completion primitive on."""
    return CausalGridWorldV2(
        size=size,
        num_hazards=3,
        num_resources=3,
        num_waypoints=2,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.15,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
    )


def _build_easy_env(size: int) -> CausalGridWorldV2:
    """P0 warmup env: fewer hazards, tolerance still on."""
    return CausalGridWorldV2(
        size=size,
        num_hazards=1,
        num_resources=3,
        num_waypoints=2,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.15,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
    )


def _build_agent(world_obs_dim: int, use_closure: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=use_closure,
    )
    cfg.heartbeat.beta_gate_bistable = True
    return REEAgent(cfg)


def _eval_residue_discharge_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> dict:
    """Frozen-policy eval instrumented for SD-034 residue-discharge behaviour.

    Counts across all eval episodes:
      n_closures           -- closure_operator._n_closures delta (0 if OFF).
      discharge_events     -- closures where residue_centers_discharged >= 1.
      mean_weight_reduction -- mean active-weight-sum reduction per discharge.
      total_committed_steps -- commitment count across all episodes.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    discharge_events = 0
    weight_reductions: list = []
    total_committed_steps = 0

    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)

                n_closures_before = (
                    int(agent.closure_operator._n_closures) if has_closure else 0
                )

                # Snapshot active-weight sum before this tick
                if has_closure and agent.residue_field.rbf_field.active_mask.any():
                    w_sum_before = float(
                        agent.residue_field.rbf_field.weights.data[
                            agent.residue_field.rbf_field.active_mask
                        ].sum().item()
                    )
                else:
                    w_sum_before = 0.0

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                # Did closure fire this tick?
                if has_closure:
                    fired_now = (
                        int(agent.closure_operator._n_closures) - n_closures_before
                    )
                    if fired_now > 0 and agent.closure_operator._event_log:
                        last_event = agent.closure_operator._event_log[-1]
                        if last_event.residue_centers_discharged >= 1:
                            discharge_events += 1
                            # Measure weight reduction from the discharge
                            if agent.residue_field.rbf_field.active_mask.any():
                                w_sum_after = float(
                                    agent.residue_field.rbf_field.weights.data[
                                        agent.residue_field.rbf_field.active_mask
                                    ].sum().item()
                                )
                                weight_reductions.append(w_sum_before - w_sum_after)

                if agent.e3._committed_trajectory is not None:
                    total_committed_steps += 1

                _, _, done, _, obs_dict = env.step(action_idx)
                if done:
                    break

    n_closures = (
        int(agent.closure_operator._n_closures) - closures_pre
        if has_closure else 0
    )
    mean_reduction = float(np.mean(weight_reductions)) if weight_reductions else 0.0

    return {
        "n_closures": n_closures,
        "discharge_events": discharge_events,
        "mean_residue_weight_reduction": mean_reduction,
        "total_committed_steps": total_committed_steps,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
    }


def _clone_closure_off(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone trained weights into a closure-OFF agent (same parameters).

    The closure operator carries no trainable parameters, so the trained
    state_dict loads cleanly into a closure-disabled config. This is the
    over-checking (non-satisficing) contrast without retraining.
    """
    import copy
    from ree_core.heartbeat.beta_gate import BetaGate

    cfg_off = copy.deepcopy(trained_agent.config)
    cfg_off.use_closure_operator = False
    cfg_off.heartbeat.beta_gate_bistable = True
    agent_off = REEAgent(cfg_off).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent_off.load_state_dict(state)
    except RuntimeError:
        agent_off.load_state_dict(state, strict=False)
    agent_off.e3._running_variance = float(trained_agent.e3._running_variance)
    agent_off.beta_gate = BetaGate(completion_release_threshold=2.0)
    return agent_off


def run_seed(seed: int, device: torch.device, smoke: bool) -> dict:
    print(f"Seed {seed} Condition train_closure_on", flush=True)
    torch.manual_seed(seed)

    size = 8 if smoke else 10
    p0_budget = 3 if smoke else 200
    p1_budget = 3 if smoke else 150
    steps_per_ep = 20 if smoke else 200
    eval_eps = 2 if smoke else 30

    easy_env = _build_easy_env(size)
    target_env = _build_env(size)
    world_obs_dim = easy_env.world_obs_dim

    agent = _build_agent(world_obs_dim, use_closure=True).to(device)

    p0 = run_p0_warmup(
        agent, easy_env, device,
        budget=p0_budget, steps_per_episode=steps_per_ep,
    )
    print(
        f"  [train] seed={seed} P0 ep {p0.n_episodes}/{p0_budget}"
        f" converged={p0.converged} aborted={p0.aborted} rv={p0.final_rv:.5f}",
        flush=True,
    )
    if p0.aborted:
        print(f"verdict: FAIL", flush=True)
        return {
            "seed": seed,
            "outcome": "commitment_not_elicited",
            "p0_aborted": True,
            "p0_abort_reason": p0.abort_reason,
            "pass": False,
        }

    p1 = run_p1_consolidation(
        agent, target_env, device,
        budget=p1_budget, steps_per_episode=steps_per_ep,
    )
    print(
        f"  [train] seed={seed} P1 ep {p1.n_episodes}/{p1_budget}"
        f" emerged={p1.commitment_emerged}"
        f" committed/ep={p1.final_committed_steps_per_ep:.1f}",
        flush=True,
    )

    # ARM_EMERGENT_ON
    arm_on = _eval_residue_discharge_behaviour(
        agent, target_env, device, eval_eps, steps_per_ep,
    )

    # ARM_FORCED_RV_ON (O-2 mandatory contrast)
    agent_forced = clone_trained_agent(agent, bistable=True, device=device)
    agent_forced.e3._running_variance = 0.001
    arm_forced = _eval_residue_discharge_behaviour(
        agent_forced, target_env, device, eval_eps, steps_per_ep,
    )

    # ARM_CLOSURE_OFF (over-checking / non-satisficing contrast)
    agent_off = _clone_closure_off(agent, device)
    agent_off.e3._running_variance = float(agent.e3._running_variance)
    arm_off = _eval_residue_discharge_behaviour(
        agent_off, target_env, device, eval_eps, steps_per_ep,
    )

    c1 = arm_on["n_closures"] >= C1_MIN_CLOSURES
    c2 = arm_on["discharge_events"] >= C2_MIN_DISCHARGE_EVENTS
    c3 = (
        arm_off["n_closures"] == 0
        and arm_off["discharge_events"] == 0
    )
    seed_pass = bool(c1 and c2 and c3)

    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)
    return {
        "seed": seed,
        "ARM_EMERGENT_ON": arm_on,
        "ARM_FORCED_RV_ON": arm_forced,
        "ARM_CLOSURE_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "p1_commitment_emerged": p1.commitment_emerged,
        "pass": seed_pass,
    }


def build_manifest(seed_results: list, smoke: bool) -> dict:
    n_pass = sum(1 for r in seed_results if r.get("pass"))
    n_seeds = len(seed_results)
    overall_pass = (n_pass / max(1, n_seeds)) >= PASS_FRACTION_REQUIRED
    outcome = "PASS" if overall_pass else "FAIL"
    direction = "supports" if overall_pass else "weakens"
    run_id = f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3"
    return {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_iso(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            cid: direction for cid in CLAIM_IDS
        },
        "thresholds": {
            "C1_min_closures": C1_MIN_CLOSURES,
            "C2_min_discharge_events": C2_MIN_DISCHARGE_EVENTS,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": n_pass,
        "smoke": smoke,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "V3-EXQ-466b behavioural successor to the V3-EXQ-466 substrate-"
            "readiness diagnostic (which is NOT superseded -- its UC1-UC5 "
            "residue-discharge API evidence stands). This arm validates the "
            "ocd4 satisficing/residue-discharge dissociation in a live "
            "committed_mode_curriculum loop on CausalGridWorldV2 with GAP-3 "
            "tolerance-band completion: closure operator ON discharges rule-"
            "domain residue at tolerance-completed waypoints; closure OFF "
            "(same weights) accumulates residue without discharge (over-checking "
            "signature). O-2 forced-rv contrast included per the GAP-11 "
            "committed_mode_curriculum mandatory-contrast rule. Pilot lineage: "
            "the GAP-11 curriculum was proven by V3-EXQ-592. Plan: "
            "commitment_closure_plan.md GAP-4 Phase 4/5 cohort."
        ),
    }


def main(smoke: bool):
    device = torch.device("cpu")
    seeds = SEEDS[:1] if smoke else SEEDS
    seed_results = [run_seed(s, device, smoke) for s in seeds]
    manifest = build_manifest(seed_results, smoke)

    print(f"=== {QUEUE_ID} {EXPERIMENT_TYPE} ===", flush=True)
    print(
        f"outcome: {manifest['outcome']} "
        f"({manifest['n_seeds_pass']}/{manifest['n_seeds']} seeds pass)",
        flush=True,
    )

    if smoke:
        return None

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Smoke run (tiny budgets, no manifest written).",
    )
    args = parser.parse_args()
    result = main(smoke=args.dry_run)
    if args.dry_run or result is None:
        sys.exit(0)
    _outcome, _out_path, _run_id = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        queue_id=QUEUE_ID,
        exit_reason="ok" if _outcome == "PASS" else "fail",
    )
    sys.exit(0)
