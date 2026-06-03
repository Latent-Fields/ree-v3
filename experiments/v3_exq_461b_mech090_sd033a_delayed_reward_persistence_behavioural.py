"""V3-EXQ-461b (EXP-0157 behavioural): MECH-090 + SD-033a + SD-034 delayed-reward persistence.

Purpose: evidence. Full-agent-loop behavioural successor to the substrate-readiness
diagnostic V3-EXQ-461 (which hand-poked BetaGate hold windows, SD-033a replay-gate
rule-state preservation, and SD-034 terminal closure via explicit API calls). This
arm exercises the full delayed-reward persistence loop inside the real
committed_mode_curriculum (P0 warmup -> P1 consolidation -> P2 eval) on a
CausalGridWorldV2 with the GAP-3 tolerance-band completion primitive, and measures
the ocd4 "delayed-reward persistence" row behaviourally:

  - During a committed Hold window (MECH-090 beta latch elevated), the SD-033a
    rule_state PERSISTS across the delay (it does not wash out before resolution).
  - SD-034 closure fires within a bounded window (<= 2 ticks) of the delayed
    RESOLUTION (the tick the beta latch releases) -- the closure-pulse signature
    that marks the held commitment as resolved.
  - A no-Hold contrast (bistable OFF, legacy per-tick beta) produces materially
    shorter Hold windows and far fewer closure-coupled resolutions: without the
    MECH-090 latch there is no sustained delay window for the rule_state to
    persist across.

Arms (one P0->P1 training run, three instrumented P2 evals on cloned agents):

  ARM_EMERGENT_ON   -- emergent-trained agent, full substrate (MECH-090 bistable
                       latch + SD-033a lateral_pfc rule_state + SD-034 closure +
                       dACC + salience). Positive arm.
  ARM_FORCED_RV_ON  -- O-2 mandatory contrast (committed_mode_curriculum):
                       clone_trained_agent(bistable=True) with running_variance
                       forced to 0.001. Isolates whether persistence + closure-
                       coupling needs EMERGENT commitment or merely the committed
                       state.
  ARM_NO_HOLD_OFF   -- clone_trained_agent(bistable=False): legacy per-tick beta,
                       no sustained Hold latch. Under-binding contrast -- short
                       windows, no persistent rule_state across a delay.

Pre-registered acceptance (PASS = majority 2/3 seeds with all criteria):
  C1  rule_state persists across the delay: ARM_EMERGENT_ON has >= C1_MIN_WINDOWS
      committed windows of length >= MIN_DELAY_TICKS, AND mean rule_state
      persistence (norm at release / norm at commit-entry) >= C1_PERSIST_FLOOR
      (the rule did not wash out during the Hold).
  C2  closure-pulse within 2 ticks of resolution: ARM_EMERGENT_ON has
      >= C2_MIN_COUPLED resolutions with an SD-034 closure fire within
      CLOSURE_WINDOW_TICKS of the beta-release tick.
  C3  Hold contrast: ARM_EMERGENT_ON mean committed-window length is strictly
      greater than ARM_NO_HOLD_OFF mean committed-window length (the MECH-090
      latch creates the sustained delay window the no-Hold arm lacks).

Interpretation grid (one row per plausible outcome -> next action):
  C1-C3 all PASS .......... delayed-reward persistence confirmed in a live loop:
                            the MECH-090 Hold latch sustains the SD-033a rule_state
                            across the delay and SD-034 closure marks the delayed
                            resolution. -> governance: behavioural support for
                            MECH-090 / SD-033a / SD-034.
  C1 FAIL (no long windows) commitment / Hold never sustained under the curriculum
                            at this budget. NOT substrate falsification ->
                            /diagnose-errors on the curriculum budget / commit
                            elicitation (mirror the V3-EXQ-592 commitment-not-
                            elicited routing), not a re-run under 461b.
  C1 PASS, C2 FAIL ........ rule_state persists but closure does not couple to the
                            release -> SD-034 closure detector not firing at
                            resolution -> /diagnose-errors on the closure
                            completion-detection path.
  C1 PASS, C3 FAIL (no-Hold holds as long) the dissociation collapses: legacy
                            per-tick beta sustains windows as long as the latch ->
                            MECH-090 Hold is not load-bearing for persistence ->
                            governance review, not a re-run.

Run:
  /opt/local/bin/python3 experiments/v3_exq_461b_mech090_sd033a_delayed_reward_persistence_behavioural.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_461b_mech090_sd033a_delayed_reward_persistence_behavioural.py --dry-run
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


EXPERIMENT_TYPE = "v3_exq_461b_mech090_sd033a_delayed_reward_persistence_behavioural"
QUEUE_ID = "V3-EXQ-461b"
CLAIM_IDS = ["MECH-090", "SD-033a", "SD-034"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]

# Pre-registered thresholds (constants, not derived from the run).
MIN_DELAY_TICKS = 3          # a committed window must be at least this long to count as a "delay"
CLOSURE_WINDOW_TICKS = 2     # closure must fire within this many ticks of resolution
C1_MIN_WINDOWS = 1           # at least this many qualifying delay windows in ARM_ON
C1_PERSIST_FLOOR = 0.5       # rule_state norm at release / norm at entry must be >= this
C2_MIN_COUPLED = 1           # at least this many closure-coupled resolutions in ARM_ON
PASS_FRACTION_REQUIRED = 2.0 / 3.0  # majority of seeds


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _build_easy_env(size: int) -> CausalGridWorldV2:
    """P0 warmup env: fewer hazards, GAP-3 tolerance-band completion ON."""
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


def _build_target_env(size: int) -> CausalGridWorldV2:
    """P1 + P2 env: GAP-3 tolerance-band completion primitive ON."""
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


def _build_agent(world_obs_dim: int) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=True,
    )
    cfg.heartbeat.beta_gate_bistable = True
    return REEAgent(cfg)


def _eval_delayed_persistence(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> dict:
    """Frozen-policy eval instrumented for delayed-reward persistence.

    Detects committed Hold windows via beta-latch transitions and measures,
    per window:
      - window length (delay duration in ticks);
      - rule_state norm at commit-entry vs at release (SD-033a persistence);
      - whether an SD-034 closure fired within CLOSURE_WINDOW_TICKS of the
        release tick (closure-pulse coupling to the delayed resolution).
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None

    window_lengths = []          # all committed-window lengths
    delay_windows = []           # windows with len >= MIN_DELAY_TICKS
    persistence_ratios = []      # rule_norm_release / rule_norm_entry for delay windows
    n_closure_coupled = 0        # resolutions with closure fire within window
    n_resolutions = 0            # total beta releases observed

    global_tick = 0
    closure_fire_ticks = []      # global ticks where closure._n_closures incremented

    def _rule_norm() -> float:
        if not has_lpfc or agent.lateral_pfc.rule_state is None:
            return 0.0
        return float(agent.lateral_pfc.rule_state.norm().item())

    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_beta = bool(agent.beta_gate.is_elevated)
            window_len = 0
            rule_norm_entry = 0.0

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)

                closures_before = (
                    int(agent.closure_operator._n_closures) if has_closure else 0
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                if has_closure:
                    if int(agent.closure_operator._n_closures) - closures_before > 0:
                        closure_fire_ticks.append(global_tick)

                cur_beta = bool(agent.beta_gate.is_elevated)

                # Commit entry: not-elevated -> elevated.
                if cur_beta and not prev_beta:
                    window_len = 1
                    rule_norm_entry = _rule_norm()
                # Sustained Hold.
                elif cur_beta and prev_beta:
                    window_len += 1
                # Resolution: elevated -> not-elevated (delayed reward arrives).
                elif (not cur_beta) and prev_beta:
                    n_resolutions += 1
                    rule_norm_release = _rule_norm()
                    window_lengths.append(window_len)
                    if window_len >= MIN_DELAY_TICKS:
                        delay_windows.append(window_len)
                        denom = max(rule_norm_entry, 1e-6)
                        persistence_ratios.append(rule_norm_release / denom)
                    # closure-coupled resolution?
                    if any(
                        abs(global_tick - c) <= CLOSURE_WINDOW_TICKS
                        for c in closure_fire_ticks
                    ):
                        n_closure_coupled += 1
                    window_len = 0

                prev_beta = cur_beta
                global_tick += 1

                _, _, done, _, obs_dict = env.step(action_idx)
                if done:
                    break

    mean_window_len = (
        float(sum(window_lengths)) / len(window_lengths) if window_lengths else 0.0
    )
    mean_persistence = (
        float(sum(persistence_ratios)) / len(persistence_ratios)
        if persistence_ratios else 0.0
    )
    return {
        "n_windows": len(window_lengths),
        "n_delay_windows": len(delay_windows),
        "mean_window_len": round(mean_window_len, 3),
        "mean_persistence_ratio": round(mean_persistence, 4),
        "n_resolutions": n_resolutions,
        "n_closure_coupled_resolutions": n_closure_coupled,
        "closure_present": has_closure,
        "n_eval_episodes": n_eps,
    }


def run_seed(seed: int, device: torch.device, smoke: bool) -> dict:
    torch.manual_seed(seed)

    size = 8 if smoke else 10
    p0_budget = 3 if smoke else 200
    p1_budget = 3 if smoke else 150
    steps_per_ep = 20 if smoke else 200
    eval_eps = 2 if smoke else 30

    easy_env = _build_easy_env(size)
    target_env = _build_target_env(size)
    world_obs_dim = easy_env.world_obs_dim

    agent = _build_agent(world_obs_dim).to(device)

    print(f"Seed {seed} Condition warmup P0", flush=True)
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
        print("verdict: FAIL", flush=True)
        return {
            "seed": seed,
            "outcome": "commitment_not_elicited",
            "p0_aborted": True,
            "p0_abort_reason": p0.abort_reason,
            "pass": False,
        }

    print(f"Seed {seed} Condition consolidation P1", flush=True)
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
    print(f"Seed {seed} Condition ARM_EMERGENT_ON", flush=True)
    arm_on = _eval_delayed_persistence(
        agent, target_env, device, eval_eps, steps_per_ep
    )

    # ARM_FORCED_RV_ON (O-2 mandatory contrast)
    print(f"Seed {seed} Condition ARM_FORCED_RV_ON", flush=True)
    agent_forced = clone_trained_agent(agent, bistable=True, device=device)
    agent_forced.e3._running_variance = 0.001
    arm_forced = _eval_delayed_persistence(
        agent_forced, target_env, device, eval_eps, steps_per_ep
    )

    # ARM_NO_HOLD_OFF (under-binding contrast: legacy per-tick beta)
    print(f"Seed {seed} Condition ARM_NO_HOLD_OFF", flush=True)
    agent_off = clone_trained_agent(agent, bistable=False, device=device)
    arm_off = _eval_delayed_persistence(
        agent_off, target_env, device, eval_eps, steps_per_ep
    )

    c1 = (
        arm_on["n_delay_windows"] >= C1_MIN_WINDOWS
        and arm_on["mean_persistence_ratio"] >= C1_PERSIST_FLOOR
    )
    c2 = arm_on["n_closure_coupled_resolutions"] >= C2_MIN_COUPLED
    c3 = arm_on["mean_window_len"] > arm_off["mean_window_len"]
    seed_pass = bool(c1 and c2 and c3)

    print(
        f"verdict: {'PASS' if seed_pass else 'FAIL'} C1={c1} C2={c2} C3={c3}",
        flush=True,
    )
    return {
        "seed": seed,
        "ARM_EMERGENT_ON": arm_on,
        "ARM_FORCED_RV_ON": arm_forced,
        "ARM_NO_HOLD_OFF": arm_off,
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
        "evidence_direction_per_claim": {cid: direction for cid in CLAIM_IDS},
        "thresholds": {
            "min_delay_ticks": MIN_DELAY_TICKS,
            "closure_window_ticks": CLOSURE_WINDOW_TICKS,
            "C1_min_windows": C1_MIN_WINDOWS,
            "C1_persist_floor": C1_PERSIST_FLOOR,
            "C2_min_coupled": C2_MIN_COUPLED,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": n_pass,
        "smoke": smoke,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "V3-EXQ-461b behavioural successor to the V3-EXQ-461 substrate-"
            "readiness diagnostic (which is NOT superseded -- its UC1-UC6 "
            "hand-poked hold-window / replay-gate / closure evidence stands). "
            "This is the FULL delayed-reward persistence behavioural arm the "
            "GAP-2 / GAP-4 plan owed once the GAP-3 CausalGridWorldV2 tolerance-"
            "band completion env + the GAP-11 committed_mode_curriculum landed. "
            "It measures, in a live curriculum loop: SD-033a rule_state PERSISTS "
            "across the MECH-090 Hold window (delay), and SD-034 closure fires "
            "within 2 ticks of the delayed RESOLUTION (beta release). The "
            "no-Hold contrast (bistable OFF) loses the sustained delay window. "
            "O-2 forced-rv contrast included per the GAP-11 committed_mode_"
            "curriculum mandatory-contrast rule. New ID (not a re-run of 461): "
            "V3-EXQ-461 was a substrate-readiness diagnostic and V3-EXQ-592 "
            "superseded it as the GAP-11 curriculum pilot. Plan: "
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
        f"outcome: {manifest['outcome']}"
        f" ({manifest['n_seeds_pass']}/{manifest['n_seeds']} seeds pass)",
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
