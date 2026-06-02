"""V3-EXQ-460b (EXP-0156 behavioural): SD-034 verified-but-not-released.

Purpose: evidence. Full-agent-loop behavioural successor to the substrate-
readiness diagnostic V3-EXQ-460 (which hand-poked emit_closure across UC1-UC6).
This arm exercises the SD-034 ClosureOperator inside the real
committed_mode_curriculum (P0 warmup -> P1 consolidation -> P2 eval) on a
CausalGridWorldV2 with the GAP-3 adaptive tolerance-band completion primitive,
and measures the ocd4 "verified-but-not-released" dissociation in a live loop:

  - With the closure operator PRESENT, a committed sequence that reaches a
    (tolerance-band) completed waypoint with a stable rule_state releases the
    MECH-090 beta latch and installs a targeted MECH-260 No-Go. Closure fires
    accumulate (closure_operator._n_closures > 0).
  - With the closure operator ABSENT (same trained weights), beta stays latched
    after the same completion -- the agent verifies completion but does not
    release. This is the failure mode SD-034 closes.

Arms (one P0->P1 training run, three P2 evals):

  ARM_EMERGENT_ON     -- emergent-trained agent, closure operator ON. Positive
                         arm. Expect n_closures > 0, beta-release transitions,
                         and No-Go installs at closure events.
  ARM_FORCED_RV_ON    -- O-2 mandatory contrast (committed_mode_curriculum):
                         clone_trained_agent(bistable=True) with running_variance
                         forced to 0.001, closure ON. Isolates whether closure
                         firing needs EMERGENT commitment or merely the committed
                         state.
  ARM_CLOSURE_OFF     -- same trained weights loaded into a closure-OFF agent.
                         Verified-but-not-released contrast: beta stays elevated
                         after completion; zero closure releases.

Pre-registered acceptance (PASS = all):
  C1  ARM_EMERGENT_ON  n_closures >= 1            (closure fires in a live loop)
  C2  ARM_EMERGENT_ON  beta_release_events >= 1   (latch actually drops)
  C3  ARM_EMERGENT_ON  nogo_installed_total >= 1  (targeted MECH-260 No-Go)
  C4  ARM_CLOSURE_OFF  n_closures == 0 AND mean_beta_elevated_steps >=
                       ARM_EMERGENT_ON mean_beta_elevated_steps
                       (verified-but-not-released: OFF holds the latch at least
                        as long as ON, and never releases via closure)

Interpretation grid (one row per plausible outcome -> next action):
  C1-C4 all PASS .................. SD-034 behavioural dissociation confirmed
                                    in a live loop; closure is the release
                                    pathway. -> governance: behavioural support
                                    for SD-034/MECH-260/MECH-261.
  C1 FAIL (n_closures == 0 ON) .... commitment/rule_state never stabilised under
                                    the curriculum at this budget. NOT substrate
                                    falsification -> /diagnose-errors on the
                                    curriculum budget / rule_state seeding
                                    (mirror the V3-EXQ-592 commitment-not-
                                    elicited routing), not a re-run under 460b.
  C1 PASS, C2 FAIL ................ closure fires but beta does not drop ->
                                    MECH-090 release wiring regression ->
                                    /diagnose-errors on closure_operator ->
                                    beta_gate.release().
  C1 PASS, C4 FAIL (OFF releases) . the dissociation collapses: beta drops in
                                    the OFF arm without closure -> SD-034 is
                                    over-specification (the ocd4 falsifiability
                                    guard). Route to governance review, not a
                                    re-run.

Run:
  /opt/local/bin/python3 experiments/v3_exq_460b_sd034_verified_but_not_released_behavioural.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_460b_sd034_verified_but_not_released_behavioural.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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


EXPERIMENT_TYPE = "v3_exq_460b_sd034_verified_but_not_released_behavioural"
QUEUE_ID = "V3-EXQ-460b"
CLAIM_IDS = ["SD-034", "MECH-260", "MECH-261"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]

# Pre-registered thresholds (constants, not derived from the run).
C1_MIN_CLOSURES = 1
C2_MIN_BETA_RELEASES = 1
C3_MIN_NOGO = 1
PASS_FRACTION_REQUIRED = 2.0 / 3.0  # majority of seeds


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_env(size: int) -> CausalGridWorldV2:
    """Target env with the GAP-3 tolerance-band completion primitive on."""
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


def _eval_closure_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> dict:
    """Frozen-policy eval instrumented for SD-034 closure behaviour.

    Counts, across all eval episodes:
      n_closures           -- closure_operator._n_closures delta (0 if operator
                              is None / closure OFF).
      beta_release_events  -- elevated->not-elevated transitions of the latch.
      nogo_installed_total -- growth of dacc._action_history attributable to
                              closure No-Go injection at fire ticks.
      total_committed_steps / total_beta_elevated -- standard commitment metrics.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_dacc = getattr(agent, "dacc", None) is not None

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    beta_release_events = 0
    nogo_installed_total = 0
    total_committed_steps = 0
    total_beta_elevated = 0

    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_beta = bool(agent.beta_gate.is_elevated)

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)

                n_closures_before = (
                    int(agent.closure_operator._n_closures) if has_closure else 0
                )
                dacc_hist_before = (
                    len(agent.dacc._action_history) if has_dacc else 0
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                # Closure fired this tick? (closure tick runs inside select_action)
                if has_closure:
                    fired_now = (
                        int(agent.closure_operator._n_closures) - n_closures_before
                    )
                    if fired_now > 0 and has_dacc:
                        nogo_installed_total += (
                            len(agent.dacc._action_history) - dacc_hist_before
                        )

                if agent.e3._committed_trajectory is not None:
                    total_committed_steps += 1
                cur_beta = bool(agent.beta_gate.is_elevated)
                if cur_beta:
                    total_beta_elevated += 1
                if prev_beta and not cur_beta:
                    beta_release_events += 1
                prev_beta = cur_beta

                _, _, done, _, obs_dict = env.step(action_idx)
                if done:
                    break

    n_closures = (
        int(agent.closure_operator._n_closures) - closures_pre if has_closure else 0
    )
    return {
        "n_closures": n_closures,
        "beta_release_events": beta_release_events,
        "nogo_installed_total": nogo_installed_total,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "mean_beta_elevated_steps": total_beta_elevated / max(1, n_eps),
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
    }


def _clone_closure_off(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone trained weights into a closure-OFF agent (same parameters).

    The closure operator carries no trainable parameters, so the trained
    state_dict loads cleanly into a closure-disabled config. This is the
    verified-but-not-released contrast without retraining.
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
        verdict = "FAIL"
        print(f"verdict: {verdict}", flush=True)
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
        f" emerged={p1.commitment_emerged} committed/ep={p1.final_committed_steps_per_ep:.1f}",
        flush=True,
    )

    # ARM_EMERGENT_ON
    arm_on = _eval_closure_behaviour(
        agent, target_env, device, eval_eps, steps_per_ep
    )

    # ARM_FORCED_RV_ON (O-2 mandatory contrast)
    agent_forced = clone_trained_agent(agent, bistable=True, device=device)
    agent_forced.e3._running_variance = 0.001
    arm_forced = _eval_closure_behaviour(
        agent_forced, target_env, device, eval_eps, steps_per_ep
    )

    # ARM_CLOSURE_OFF (verified-but-not-released contrast)
    agent_off = _clone_closure_off(agent, device)
    agent_off.e3._running_variance = float(agent.e3._running_variance)
    arm_off = _eval_closure_behaviour(
        agent_off, target_env, device, eval_eps, steps_per_ep
    )

    c1 = arm_on["n_closures"] >= C1_MIN_CLOSURES
    c2 = arm_on["beta_release_events"] >= C2_MIN_BETA_RELEASES
    c3 = arm_on["nogo_installed_total"] >= C3_MIN_NOGO
    c4 = (
        arm_off["n_closures"] == 0
        and arm_off["mean_beta_elevated_steps"] >= arm_on["mean_beta_elevated_steps"]
    )
    seed_pass = bool(c1 and c2 and c3 and c4)

    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)
    return {
        "seed": seed,
        "ARM_EMERGENT_ON": arm_on,
        "ARM_FORCED_RV_ON": arm_forced,
        "ARM_CLOSURE_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4},
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
            "C2_min_beta_releases": C2_MIN_BETA_RELEASES,
            "C3_min_nogo": C3_MIN_NOGO,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": n_pass,
        "smoke": smoke,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "V3-EXQ-460b behavioural successor to the V3-EXQ-460 substrate-"
            "readiness diagnostic (which is NOT superseded -- its UC1-UC6 "
            "arithmetic wiring evidence stands). This arm validates the ocd4 "
            "verified-but-not-released dissociation in a live committed_mode_"
            "curriculum loop on CausalGridWorldV2 with GAP-3 tolerance-band "
            "completion: closure operator ON releases the MECH-090 latch and "
            "installs a MECH-260 No-Go at completed-with-stable-rule_state "
            "ticks; closure OFF (same weights) holds the latch (verified-but-"
            "not-released). O-2 forced-rv contrast included per the GAP-11 "
            "committed_mode_curriculum mandatory-contrast rule. Pilot lineage: "
            "the GAP-11 curriculum was proven by V3-EXQ-592 (supersedes the "
            "synthetic V3-EXQ-461). Plan: commitment_closure_plan.md GAP-4 "
            "Phase 4/5 cohort."
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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Smoke run (tiny budgets, no manifest)."
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
