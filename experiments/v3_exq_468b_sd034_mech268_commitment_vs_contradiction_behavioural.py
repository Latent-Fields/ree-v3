"""V3-EXQ-468b (EXP-0164 behavioural): commitment vs contradiction.

Purpose: evidence. Full-agent-loop behavioural arm of the EXP-0164
commitment-vs-contradiction row (commitment_closure_plan.md GAP-4,
Phase 4/5 OCD cohort). Successor to the substrate-readiness diagnostic
V3-EXQ-468, which validated that SD-034 closure + MECH-268 dACC PE
saturation are mechanistically separable via the 4-arm A/B/C/D
arithmetic test.

Scientific question: when counter-evidence contradicts an active
commitment, a healthy agent RELEASES the commitment (MECH-090 beta drops,
coordinated by SD-034 closure + MECH-268 dACC PE saturation preventing
runaway conflict accumulation). Without the substrate the agent
PERSEVERATES -- stays committed despite contradiction -- the OCD-like
signature.

Arms (one P0->P1 training run per seed, two P2 evals):

  ARM_SUBSTRATE_ON  -- full closure + dACC-saturation + bistable agent.
                       Counter-evidence fires DURING committed sequences;
                       dACC PE saturation attenuates the conflicting PE and
                       closure releases the MECH-090 beta latch. Expect
                       beta releases that occur near contradiction events
                       AND a drop in committed-step fraction after injection.

  ARM_SUBSTRATE_OFF -- clone_trained_agent with closure OFF and dACC
                       saturation disabled. Same trained weights; same env.
                       Expect perseveration: committed steps sustained despite
                       counter-evidence, no substrate-mediated release.

Pre-registered acceptance (PASS = all 3 criteria, majority 2/3 seeds):
  C1  ARM_SUBSTRATE_ON  beta_release_near_contradiction >= 1
                        (at least one beta release within RELEASE_WINDOW
                         steps of a counter-evidence injection event)
  C2  ARM_SUBSTRATE_ON  committed_frac_post < committed_frac_pre * C2_DROP_FACTOR
                        (committed-step fraction drops after first contradiction)
  C3  ARM_SUBSTRATE_OFF committed_frac_post >= committed_frac_pre * C3_PERSIST_FACTOR
                        (OFF arm perseverates -- fraction is sustained)

Interpretation grid (one row per plausible outcome -> next action):
  C1+C2+C3 all PASS ....... SD-034 + MECH-268 + MECH-090 confirmed.
                             Substrate coordinates contradiction-driven
                             decommit; OFF arm shows OCD-like perseveration.
                             -> governance: behavioural evidence for all
                             three claims.
  C1 FAIL (ON no release) . commitment never formed OR counter-evidence
                             too weak to trigger dACC PE accumulation.
                             NOT substrate falsification -> /diagnose-errors
                             on curriculum budget / counter-evidence config.
  C2 FAIL (ON no drop) .... releases occur but not near contradiction ticks.
                             dACC PE saturation may not be accumulating on
                             counter-evidence specifically -> /diagnose-errors
                             on MECH-268 PE tracking path.
  C3 FAIL (OFF releases) .. dissociation collapses: OFF arm also releases
                             under contradiction -> substrate is not the
                             load-bearing pathway -> governance review,
                             do not re-run.

Run:
  /opt/local/bin/python3 experiments/v3_exq_468b_sd034_mech268_commitment_vs_contradiction_behavioural.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_468b_sd034_mech268_commitment_vs_contradiction_behavioural.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
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
from ree_core.heartbeat.beta_gate import BetaGate  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.committed_mode_curriculum import (  # noqa: E402
    clone_trained_agent,
    run_p0_warmup,
    run_p1_consolidation,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = (
    "v3_exq_468b_sd034_mech268_commitment_vs_contradiction_behavioural"
)
QUEUE_ID = "V3-EXQ-468b"
CLAIM_IDS = ["SD-034", "MECH-268", "MECH-090"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]

# Pre-registered thresholds -- do NOT derive from run data.
# Steps after a counter-evidence injection within which a beta release
# counts as contradiction-triggered (C1).
RELEASE_WINDOW = 20
# C2: ON arm committed fraction post-contradiction must be below this
# fraction of the pre-contradiction committed fraction.
C2_DROP_FACTOR = 0.85
# C3: OFF arm committed fraction post-contradiction must remain at or
# above this fraction of its pre-contradiction committed fraction.
C3_PERSIST_FACTOR = 0.70
# Majority-of-seeds threshold.
PASS_FRACTION_REQUIRED = 2.0 / 3.0


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------

def _build_easy_env(size: int) -> CausalGridWorldV2:
    """P0 warmup: fewer hazards, NO counter-evidence (learn to commit first)."""
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


def _build_target_env(size: int, smoke: bool) -> CausalGridWorldV2:
    """P1 + P2 eval env: GAP-3 counter-evidence primitive ON."""
    return CausalGridWorldV2(
        size=size,
        num_hazards=3,
        num_resources=3,
        num_waypoints=2,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.15,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
        counter_evidence_enabled=True,
        counter_evidence_interval=20 if smoke else 50,
        counter_evidence_prob=0.6,
        counter_evidence_degrade_step=0.2,
        counter_evidence_degrade_floor=0.0,
        counter_evidence_requires_persistent_rule=True,
    )


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def _build_agent(world_obs_dim: int) -> REEAgent:
    """Build a substrate-ON agent (closure + dACC + bistable)."""
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
    agent = REEAgent(cfg)
    # MECH-268: enable dACC PE saturation (not surfaced through from_dims).
    if agent.dacc is not None:
        agent.dacc.config.dacc_saturation_enabled = True
        agent.dacc.config.dacc_saturation_window = 8
        agent.dacc.config.dacc_saturation_strength = 0.3
        agent.dacc.config.dacc_saturation_grace = 2
    return agent


# ---------------------------------------------------------------------------
# Substrate-OFF clone (verified-but-perseverates contrast)
# ---------------------------------------------------------------------------

def _clone_substrate_off(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone trained weights into a substrate-OFF agent.

    Disables:
      - SD-034 ClosureOperator (use_closure_operator=False)
      - MECH-268 dACC PE saturation (dacc_saturation_enabled=False)

    The trained state_dict loads cleanly -- the closure operator carries
    no trainable parameters.  This is the perseveration-without-substrate
    contrast arm.
    """
    cfg_off = copy.deepcopy(trained_agent.config)
    cfg_off.use_closure_operator = False
    cfg_off.heartbeat.beta_gate_bistable = True
    agent_off = REEAgent(cfg_off).to(device)

    # Disable dACC saturation on the cloned agent.
    if agent_off.dacc is not None:
        agent_off.dacc.config.dacc_saturation_enabled = False

    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent_off.load_state_dict(state)
    except RuntimeError:
        agent_off.load_state_dict(state, strict=False)

    agent_off.e3._running_variance = float(trained_agent.e3._running_variance)
    agent_off.beta_gate = BetaGate(completion_release_threshold=2.0)
    return agent_off


# ---------------------------------------------------------------------------
# Instrumented P2 eval -- contradiction-aware
# ---------------------------------------------------------------------------

def _eval_contradiction_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> dict:
    """Frozen-policy eval instrumented for commitment-vs-contradiction.

    Tracks per-episode:
      - committed steps BEFORE the first counter-evidence injection
      - committed steps AFTER the first counter-evidence injection
      - beta release events occurring within RELEASE_WINDOW steps of any
        counter-evidence injection (contradiction-triggered releases)
      - total beta release events and committed steps

    Returns a dict with aggregate and per-episode metrics.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None

    total_committed_steps = 0
    total_beta_elevated = 0
    total_beta_release_events = 0
    total_beta_release_near_contradiction = 0
    total_committed_pre = 0
    total_committed_post = 0
    total_episodes_with_contradiction = 0

    per_episode = []

    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_beta = bool(agent.beta_gate.is_elevated)

            ep_committed = 0
            ep_elevated = 0
            ep_release_events = 0
            ep_release_near_contradiction = 0
            ep_committed_pre = 0
            ep_committed_post = 0
            first_contradiction_step = -1
            # steps_since_last_injection[i] = steps since injection i fired
            # We track a rolling window counter for each recent injection.
            recent_injection_timers: list = []

            for step in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                # Committed/elevated tracking.
                cur_committed = agent.e3._committed_trajectory is not None
                cur_beta = bool(agent.beta_gate.is_elevated)

                if cur_committed:
                    ep_committed += 1
                if cur_beta:
                    ep_elevated += 1

                # Beta release event (elevated -> not elevated).
                if prev_beta and not cur_beta:
                    ep_release_events += 1
                    # Check if any injection was within RELEASE_WINDOW steps.
                    if any(t <= RELEASE_WINDOW for t in recent_injection_timers):
                        ep_release_near_contradiction += 1

                prev_beta = cur_beta

                # Advance recent injection timers.
                recent_injection_timers = [t + 1 for t in recent_injection_timers]
                # Prune timers that have exceeded the window.
                recent_injection_timers = [
                    t for t in recent_injection_timers if t <= RELEASE_WINDOW + 1
                ]

                _, _, done, info, obs_dict = env.step(action_idx)

                # Check for counter-evidence injection this tick (in info dict).
                injected = bool(
                    info.get("counter_evidence_injected_this_tick", False)
                )
                if injected:
                    recent_injection_timers.append(0)
                    if first_contradiction_step < 0:
                        first_contradiction_step = step

                # Accumulate pre/post committed steps.
                if first_contradiction_step >= 0:
                    if cur_committed:
                        ep_committed_post += 1
                else:
                    if cur_committed:
                        ep_committed_pre += 1

                if done:
                    break

            had_contradiction = first_contradiction_step >= 0
            if had_contradiction:
                total_episodes_with_contradiction += 1
                total_committed_pre += ep_committed_pre
                total_committed_post += ep_committed_post

            total_committed_steps += ep_committed
            total_beta_elevated += ep_elevated
            total_beta_release_events += ep_release_events
            total_beta_release_near_contradiction += ep_release_near_contradiction

            per_episode.append({
                "committed_steps": ep_committed,
                "beta_elevated_steps": ep_elevated,
                "release_events": ep_release_events,
                "release_near_contradiction": ep_release_near_contradiction,
                "committed_pre": ep_committed_pre,
                "committed_post": ep_committed_post,
                "had_contradiction": had_contradiction,
                "first_contradiction_step": first_contradiction_step,
            })

    # Aggregate committed fraction before/after contradiction (across episodes
    # that had at least one injection).
    eps_with_c = max(1, total_episodes_with_contradiction)
    mean_committed_pre = total_committed_pre / eps_with_c
    mean_committed_post = total_committed_post / eps_with_c

    # Committed fraction: post vs pre (use mean steps as denominator proxy).
    # Avoid division-by-zero: if pre is 0 set frac to 1.0 (no drop possible).
    committed_frac_pre = mean_committed_pre / max(1.0, mean_committed_pre)
    committed_frac_post = (
        mean_committed_post / mean_committed_pre
        if mean_committed_pre > 0 else 1.0
    )

    return {
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "total_beta_release_events": total_beta_release_events,
        "beta_release_near_contradiction": total_beta_release_near_contradiction,
        "episodes_with_contradiction": total_episodes_with_contradiction,
        "mean_committed_pre": mean_committed_pre,
        "mean_committed_post": mean_committed_post,
        "committed_frac_post_vs_pre": committed_frac_post,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
        "per_episode": per_episode,
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, device: torch.device, smoke: bool) -> dict:
    print(f"Seed {seed} Condition train_substrate_on", flush=True)
    torch.manual_seed(seed)

    size = 8 if smoke else 10
    p0_budget = 3 if smoke else 200
    p1_budget = 3 if smoke else 150
    steps_per_ep = 20 if smoke else 200
    eval_eps = 2 if smoke else 30

    easy_env = _build_easy_env(size)
    target_env = _build_target_env(size, smoke)
    world_obs_dim = easy_env.world_obs_dim

    agent = _build_agent(world_obs_dim).to(device)

    # P0: world-model warmup on easy env.
    p0 = run_p0_warmup(
        agent, easy_env, device,
        budget=p0_budget, steps_per_episode=steps_per_ep,
    )
    print(
        f"  [train] ep {p0.n_episodes}/{p0_budget}"
        f" converged={p0.converged} aborted={p0.aborted}"
        f" rv={p0.final_rv:.5f}",
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

    # P1: consolidation on target env (counter-evidence active).
    p1 = run_p1_consolidation(
        agent, target_env, device,
        budget=p1_budget, steps_per_episode=steps_per_ep,
    )
    print(
        f"  [train] ep {p1.n_episodes}/{p1_budget}"
        f" emerged={p1.commitment_emerged}"
        f" committed/ep={p1.final_committed_steps_per_ep:.1f}",
        flush=True,
    )

    # ARM_SUBSTRATE_ON eval.
    print(f"Seed {seed} Condition ARM_SUBSTRATE_ON", flush=True)
    arm_on = _eval_contradiction_behaviour(
        agent, target_env, device, eval_eps, steps_per_ep
    )

    # O-2 mandatory contrast: forced-rv clone, substrate ON.
    agent_forced = clone_trained_agent(agent, bistable=True, device=device)
    agent_forced.e3._running_variance = 0.001
    if agent_forced.dacc is not None:
        agent_forced.dacc.config.dacc_saturation_enabled = True
    print(f"Seed {seed} Condition ARM_FORCED_RV_ON", flush=True)
    arm_forced = _eval_contradiction_behaviour(
        agent_forced, target_env, device, eval_eps, steps_per_ep
    )

    # ARM_SUBSTRATE_OFF: closure OFF + saturation disabled.
    print(f"Seed {seed} Condition ARM_SUBSTRATE_OFF", flush=True)
    agent_off = _clone_substrate_off(agent, device)
    agent_off.e3._running_variance = float(agent.e3._running_variance)
    arm_off = _eval_contradiction_behaviour(
        agent_off, target_env, device, eval_eps, steps_per_ep
    )

    # --- Criteria evaluation ---
    # C1: ON arm had at least one beta release triggered near contradiction.
    c1 = arm_on["beta_release_near_contradiction"] >= 1

    # C2: ON arm committed fraction dropped after first contradiction.
    c2 = arm_on["committed_frac_post_vs_pre"] < C2_DROP_FACTOR

    # C3: OFF arm committed fraction is sustained (perseveration).
    c3 = arm_off["committed_frac_post_vs_pre"] >= C3_PERSIST_FACTOR

    seed_pass = bool(c1 and c2 and c3)
    verdict = "PASS" if seed_pass else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "ARM_SUBSTRATE_ON": arm_on,
        "ARM_FORCED_RV_ON": arm_forced,
        "ARM_SUBSTRATE_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "p1_commitment_emerged": p1.commitment_emerged,
        "pass": seed_pass,
    }


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(seed_results: list, smoke: bool) -> dict:
    n_pass = sum(1 for r in seed_results if r.get("pass"))
    n_seeds = len(seed_results)
    overall_pass = (n_pass / max(1, n_seeds)) >= PASS_FRACTION_REQUIRED
    outcome = "PASS" if overall_pass else "FAIL"
    direction = "supports" if overall_pass else "weakens"
    run_id = f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3"

    # Per-claim evidence direction based on which criteria bear on each claim.
    def _claim_dir(claim: str) -> str:
        if claim == "SD-034":
            # SD-034 closure drives the release path (C1 + C2 load bearing).
            ok = all(
                r.get("criteria", {}).get("C1") and r.get("criteria", {}).get("C2")
                for r in seed_results if r.get("pass") is not None
            )
        elif claim == "MECH-268":
            # MECH-268 PE saturation accumulates on contradiction;
            # contributes to C1 + C2 alongside closure.
            ok = all(
                r.get("criteria", {}).get("C1")
                for r in seed_results if r.get("pass") is not None
            )
        else:  # MECH-090
            # MECH-090 beta gate release is directly measured by C1 + C2.
            ok = overall_pass
        return "supports" if ok else "weakens"

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
            cid: _claim_dir(cid) for cid in CLAIM_IDS
        },
        "thresholds": {
            "RELEASE_WINDOW": RELEASE_WINDOW,
            "C2_DROP_FACTOR": C2_DROP_FACTOR,
            "C3_PERSIST_FACTOR": C3_PERSIST_FACTOR,
            "pass_fraction_required": PASS_FRACTION_REQUIRED,
        },
        "n_seeds": n_seeds,
        "n_seeds_pass": n_pass,
        "smoke": smoke,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "V3-EXQ-468b behavioural arm of EXP-0164 (commitment vs "
            "contradiction, commitment_closure_plan.md GAP-4, Phase 4/5 "
            "OCD cohort). Successor to the V3-EXQ-468 substrate-readiness "
            "diagnostic (which validated the 4-arm A/B/C/D arithmetic "
            "separability of SD-034 and MECH-268 -- that evidence stands "
            "and is not superseded). This arm exercises all three claims in "
            "the live committed_mode_curriculum loop on CausalGridWorldV2 "
            "with the GAP-3 counter-evidence-injection primitive: "
            "ARM_SUBSTRATE_ON (closure+dACC-saturation+bistable) releases "
            "commitment under sustained contradiction; ARM_SUBSTRATE_OFF "
            "(same weights, closure off, saturation disabled) perseverates. "
            "C1 tests contradiction-triggered beta release; C2 tests "
            "committed-step fraction drop after first injection; C3 tests "
            "OFF arm perseveration. O-2 forced-rv contrast included per "
            "GAP-11 committed_mode_curriculum mandatory-contrast rule. "
            "Curriculum proven by V3-EXQ-592 (supersedes the synthetic "
            "V3-EXQ-461). Plan: commitment_closure_plan.md GAP-4."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        "--dry-run",
        action="store_true",
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
