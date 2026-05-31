#!/opt/local/bin/python3
"""
V3-EXQ-592d: MECH-090 R-c commit-entry readiness conjunction validation (4-arm).

Supersedes V3-EXQ-592c (2-arm; FAILed on ree-cloud-3 2026-05-30). 592b/592c
both ran a 2-arm script that exercised ONLY the within-tick score_margin axis
of the R-c conjunction; the across-tick motor-program readiness axis (landed
2026-05-29 as ree_core/policy/commit_readiness.py + REEConfig
use_mech090_readiness_conjunction master + AND-composition in agent.py
select_action) has NEVER been validated. 592d expands the falsifier to 4 arms
toggling the two axes independently, so the nav_competence axis is exercised
for the first time and the conjunction-vs-axis-alone discrimination is
measurable.

SLEEP DRIVER: K=never (SleepLoopManager disabled; experiment does not exercise
sleep aggregation cluster). use_sleep_loop=False default.

EXPERIMENT_PURPOSE: diagnostic (substrate-readiness validation of the R-c
conjunction; behavioural validation of MECH-090 cluster downstream is the
governance-weighting signal, NOT this).

Substrate landed in two passes:
  2026-05-28 (within-tick decisiveness axis): BetaGate.should_admit_elevation
    + HeartbeatConfig.use_commit_readiness_gate / commit_readiness_floor. R-a
    rv-only commit-entry is supplemented by a per-candidate first-action score
    margin gate; elevation admitted iff E3SelectionResult.committed (rv-low)
    AND sorted(scores)[1] - sorted(scores)[0] >= commit_readiness_floor.
  2026-05-29 (across-tick motor-program readiness axis): CommitReadiness EMA
    + REEConfig.use_mech090_readiness_conjunction master flag (auto-arms
    use_commit_readiness=True via __post_init__) + REEConfig.mech090_readiness_floor.
    AND-composed at both elevate sites in agent.py L3681-3731.

Arm matrix (4 arms x 3 seeds = 12 cells):
  ARM_0 BASELINE_BOTH_OFF: use_commit_readiness_gate=False,
    use_mech090_readiness_conjunction=False. Pure legacy rv-only commit entry.
    Reference baseline for commit-rate / false-commit-rate / time-to-commit.
  ARM_1 SCORE_MARGIN_ONLY: use_commit_readiness_gate=True (floor=0.05),
    use_mech090_readiness_conjunction=False. Within-tick decisiveness axis
    alone. Replicates 592b/c substrate but at full 4-arm context.
  ARM_2 NAV_COMPETENCE_ONLY: use_commit_readiness_gate=False,
    use_mech090_readiness_conjunction=True (floor=0.3, initial=0.0). Across-
    tick motor-program readiness axis alone. FIRST-EVER validation.
    Initial=0.0 ensures gate starts blocking (default 1.0 would be fail-open).
  ARM_3 BOTH_GATES_ON: use_commit_readiness_gate=True (floor=0.05),
    use_mech090_readiness_conjunction=True (floor=0.3, initial=0.0). Full R-c
    conjunction (AND-composed at both elevate sites). Intended-production
    config.

ARM_2 / ARM_3 per-tick nav_competence proxy push: the substrate-side
CommitReadiness.update() is NOT wired into agent.sense() in Phase 1 (per
commit_readiness.py docstring SCOPE LIMITS); harnesses must push readiness
via notify_outcome(). 592d uses a custom P2 inner loop that pushes
nav_competence_proxy = clip([0,1], 1.0 - rv/commit_threshold) after every
env.step(). When rv is at threshold (i.e. agent is at the boundary of being
willing to commit), proxy=0; when rv=0 (world-model converged), proxy=1.
This is the simplest proxy derivable from existing agent telemetry and
matches the chip's spirit of "use the simplest proxy available". Initial=0.0
+ no-push-during-P0 means readiness stays at 0.0 during P0; P2 pushes lift
it as the agent's world-model confidence rises.

Per-arm per-seed metrics:
  total_committed_steps: int (P2)
  n_commit_entries: int = beta_gate.mech090_n_elevation_admitted (P2-only;
    P0 gate consultation is rare since rv is high)
  n_commit_blocks_score_margin: int = beta_gate.mech090_n_elevation_blocked
  n_commit_blocks_nav_competence: int = commit_readiness.n_blocks_emitted
  mean_time_to_commit_from_p2_start: float (steps elapsed to first commit;
    inf if no commit fired during P2)
  false_commit_rate: float = n_false_commits / max(1, n_commit_entries)
    where false_commit = commit fired while (rv < threshold AND
    proxy < 0.3) -- the V3-EXQ-592-seed-42 degenerate-basin signature.
  final_readiness_value: float = commit_readiness.readiness at P2 end
  final_running_variance: float = e3._running_variance at P2 end
  mean_p2_hold_rate: float = beta_elevated / committed (0 if 0 committed)
  rv_crossed_in_p0: bool = final_rv < commit_threshold at P0 end

Aggregate per arm:
  commit_rate_across_seeds: float = sum(n_commit_entries) / total_steps_p2
  false_commit_rate_across_seeds: float = mean(false_commit_rate)
  time_to_commit_p50_p90: tuple

Acceptance criteria:
  C1 BASELINE_FIRES: ARM_0 commit_rate_across_seeds > 0.001. Test
    environment is producing commit attempts (rv-only baseline IS firing).
    If this fails the curriculum / env is wrong, not the gate.
  C2 SCORE_MARGIN_DISCRIMINATES: ARM_1 false_commit_rate_across_seeds <
    ARM_0 false_commit_rate_across_seeds * 0.7. Score-margin gate
    suppresses >= 30% of degenerate commits vs baseline.
  C3 NAV_COMPETENCE_FIRES: ARM_2 sum(n_commit_blocks_nav_competence) >= 1
    AND ARM_2 commit_rate_across_seeds < ARM_0 commit_rate_across_seeds.
    The nav_competence gate blocks at least one elevation attempt; total
    commit rate is net-down vs baseline (gate is active).
  C4 CONJUNCTION_SUPPRESSES_DEGENERATE: ARM_3 false_commit_rate_across_seeds
    < 0.10 (conjunction suppresses V3-EXQ-592-seed-42 false commits below
    10%) AND ARM_3 commit_rate_across_seeds > 0 (conjunction does not
    permanently lock out commitment when readiness clears).

  PASS = C1 AND (C2 OR C3) AND C4. FAIL otherwise.

Interpretation grid:
  Outcome                          | Diagnosis
  ---------------------------------|--------------------------------------
  PASS                             | R-c substrate (both axes) validated.
                                   | commitment_closure:GAP-4
                                   | substrate_landed_pending_validation
                                   | -> done substrate-side. Queue *b cohort
                                   | (V3-EXQ-460b/461/463b/464b/466b/467b/
                                   | 468b) Phase 4/5 behavioural arms.
  C1 FAIL                          | Test setup broken (rv-only baseline
                                   | isn't firing). /diagnose-errors on
                                   | curriculum harness, NOT the gate.
  C2 PASS + C3 FAIL                | score_margin axis works, nav_competence
                                   | axis does NOT block. Route to
                                   | nav_competence-only diagnostic: proxy
                                   | may be wrong OR floor=0.3 too low.
                                   | Sweep floor {0.5, 0.7, 0.9}.
  C2 FAIL + C3 PASS                | nav_competence axis works, score_margin
                                   | axis does NOT discriminate. Route to
                                   | score_margin-only diagnostic: floor
                                   | sweep {0.01, 0.02, 0.05, 0.10}.
  All FAIL                         | R-c substrate retest. /diagnose-errors
                                   | on agent.py wiring -- check getattr-
                                   | fallback flag propagation through both
                                   | REEConfig and HeartbeatConfig.

Supersedes: V3-EXQ-592c. claim_ids=["MECH-090"] only per CLAUDE.md
claim_ids Accuracy Rule -- SD-034/MECH-266/267/268 are transitively
unblocked by GAP-4 substrate completion but not directly tested by
this substrate-readiness diagnostic.

Cross-link IGW-20260531-021. Predecessor synthesis:
REE_assembly/evidence/literature/targeted_review_connectome_mech_090/
synthesis.md commit 9e68c5ca8a.
"""

import argparse
import datetime
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from committed_mode_curriculum import (  # noqa: E402
    run_p0_warmup,
    P0Result,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_592d_mech090_readiness_conjunction_validation"
QUEUE_ID = "V3-EXQ-592d"
CLAIM_IDS: List[str] = ["MECH-090"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]

# R-c gate floors (pre-registered; see SCOPE_NOTE below for rationale)
SCORE_MARGIN_FLOOR = 0.05      # within-tick decisiveness axis
NAV_COMPETENCE_FLOOR = 0.3     # across-tick motor-program readiness axis
NAV_COMPETENCE_INITIAL_FAILCLOSED = 0.0  # ARM_2/ARM_3: fail-closed start so
                                          # the gate actually has the
                                          # opportunity to block. The default
                                          # initial=1.0 (fail-open) would
                                          # make the gate permanently admit
                                          # absent any harness push.

# False-commit detection: a commit-entry that fired while rv was low AND
# proxy_nav_competence was also low (the V3-EXQ-592-seed-42 signature).
FALSE_COMMIT_PROXY_THRESHOLD = 0.3

# Acceptance criteria thresholds
C1_BASELINE_MIN_COMMIT_RATE = 0.001
C2_SCORE_MARGIN_REDUCTION_RATIO = 0.7  # ARM_1 false-commit < ARM_0 * 0.7
C3_MIN_NAV_BLOCKS = 1
C4_MAX_CONJUNCTION_FALSE_COMMIT_RATE = 0.10

# Curriculum config (matched to V3-EXQ-592c)
P0_BUDGET = 400
P0_STEPS_PER_EPISODE = 200
P0_PROBE_INTERVAL = 40
P0_MID_PROBE_FRAC = 0.60

EASY_ENV_SIZE = 10
EASY_ENV_HAZARDS = 2
EASY_ENV_RESOURCES = 2
EASY_TOLERANCE_FRAC = 0.30

TARGET_ENV_SIZE = 10
TARGET_ENV_HAZARDS = 4
TARGET_ENV_RESOURCES = 3
TARGET_TOLERANCE_FRAC = 0.15

P2_EPISODES = 50
P2_STEPS_PER_EPISODE = 200

ARMS: List[Dict] = [
    {
        "name": "ARM_0_BASELINE_BOTH_OFF",
        "use_score_margin_gate": False,
        "use_nav_competence_conjunction": False,
        "commit_readiness_initial": 1.0,  # irrelevant when conjunction off
    },
    {
        "name": "ARM_1_SCORE_MARGIN_ONLY",
        "use_score_margin_gate": True,
        "use_nav_competence_conjunction": False,
        "commit_readiness_initial": 1.0,
    },
    {
        "name": "ARM_2_NAV_COMPETENCE_ONLY",
        "use_score_margin_gate": False,
        "use_nav_competence_conjunction": True,
        "commit_readiness_initial": NAV_COMPETENCE_INITIAL_FAILCLOSED,
    },
    {
        "name": "ARM_3_BOTH_GATES_ON",
        "use_score_margin_gate": True,
        "use_nav_competence_conjunction": True,
        "commit_readiness_initial": NAV_COMPETENCE_INITIAL_FAILCLOSED,
    },
]

TOTAL_RUNS = len(ARMS) * len(SEEDS)


# ---------------------------------------------------------------------------
# Env factories (matched to V3-EXQ-592c)
# ---------------------------------------------------------------------------

def make_easy_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=EASY_ENV_SIZE,
        num_hazards=EASY_ENV_HAZARDS,
        num_resources=EASY_ENV_RESOURCES,
        hazard_harm=0.02,
        resource_benefit=0.05,
        use_proxy_fields=True,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=EASY_TOLERANCE_FRAC,
        seed=seed,
    )


def make_target_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=TARGET_ENV_SIZE,
        num_hazards=TARGET_ENV_HAZARDS,
        num_resources=TARGET_ENV_RESOURCES,
        hazard_harm=0.02,
        resource_benefit=0.05,
        use_proxy_fields=True,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=TARGET_TOLERANCE_FRAC,
        seed=seed + 1000,
    )


# ---------------------------------------------------------------------------
# Agent factory: arm-parameterised
# ---------------------------------------------------------------------------

def make_arm_cfg(arm: Dict) -> REEConfig:
    """Standard 592c-matched cfg + per-arm R-c gate flags."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        alpha_world=0.9,
        use_harm_stream=False,
        # MECH-090 R-c (2026-05-29): nav_competence axis. master flag
        # auto-arms use_commit_readiness=True via REEConfig.__post_init__.
        use_mech090_readiness_conjunction=bool(arm["use_nav_competence_conjunction"]),
        mech090_readiness_floor=NAV_COMPETENCE_FLOOR,
        commit_readiness_initial=float(arm["commit_readiness_initial"]),
    )
    cfg.heartbeat.beta_gate_bistable = True
    # MECH-090 R-c (2026-05-28): score_margin within-tick axis.
    cfg.heartbeat.use_commit_readiness_gate = bool(arm["use_score_margin_gate"])
    cfg.heartbeat.commit_readiness_floor = SCORE_MARGIN_FLOOR
    return cfg


# ---------------------------------------------------------------------------
# Per-tick nav_competence proxy push helper
# ---------------------------------------------------------------------------

def compute_nav_competence_proxy(agent: REEAgent) -> float:
    """Proxy nav_competence from agent's world-model confidence.

    Returns clip_[0,1](1.0 - rv / commit_threshold). When rv is at the commit
    threshold (agent at the boundary of willingness to commit), proxy=0.
    When rv=0 (world-model fully converged), proxy=1. This is the simplest
    proxy derivable from existing agent telemetry without env-side hooks.
    """
    rv = float(getattr(agent.e3, "_running_variance", 1.0))
    threshold = float(getattr(agent.e3, "commit_threshold", 0.40))
    if threshold <= 0:
        return 1.0
    raw = 1.0 - (rv / threshold)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Custom P2 eval loop with per-tick metrics + nav_competence push
# ---------------------------------------------------------------------------

def run_p2_with_metrics(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
    push_nav_competence: bool,
) -> Dict:
    """Frozen-policy P2 eval with per-tick metric capture.

    When push_nav_competence=True AND agent.commit_readiness is not None,
    calls agent.commit_readiness.notify_outcome(proxy) AFTER each env.step()
    so the next tick's commit-entry consultation sees the updated readiness.

    Returns a dict with all per-arm-per-seed metrics needed downstream
    (commit attempts, blocks, false-commits, hold rate, final state).
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim

    total_committed_steps = 0
    total_beta_elevated = 0
    n_false_commits = 0
    # NOTE: BetaGate.reset() and CommitReadiness.reset() clear the cumulative
    # admit/block counters at every episode boundary (per the substrate's
    # per-episode scope -- cross-episode counters would conflate distinct
    # trials' degeneracy signatures). A naive pre-P2 / post-P2 snapshot
    # would therefore only capture the LAST episode's delta. We accumulate
    # the per-episode deltas explicitly below.
    n_admits_p2_accum = 0
    n_blocks_score_p2_accum = 0
    n_blocks_nav_p2_accum = 0
    time_to_first_commit = math.inf
    p2_step_global = 0
    total_p2_steps = 0
    per_episode_summary: List[Dict] = []

    rv_at_first_commit: Optional[float] = None
    proxy_at_first_commit: Optional[float] = None

    with torch.no_grad():
        for ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()  # clears BetaGate + CommitReadiness counters
            ep_committed = 0
            ep_elevated = 0
            ep_false_commits = 0
            # After reset(), the per-episode counters start at 0.
            admits_before_ep = 0

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                # Snapshot pre-select admits to detect a fresh commit this tick.
                admits_pre = int(agent.beta_gate.get_state().get("mech090_n_elevation_admitted", 0))
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())
                admits_post = int(agent.beta_gate.get_state().get("mech090_n_elevation_admitted", 0))

                # Track time-to-first-commit AND false-commit detection.
                if admits_post > admits_pre:
                    if time_to_first_commit == math.inf:
                        time_to_first_commit = float(p2_step_global)
                        rv_at_first_commit = float(getattr(agent.e3, "_running_variance", 0.0))
                        proxy_at_first_commit = compute_nav_competence_proxy(agent)
                    rv_now = float(getattr(agent.e3, "_running_variance", 0.0))
                    proxy_now = compute_nav_competence_proxy(agent)
                    thr_now = float(getattr(agent.e3, "commit_threshold", 0.40))
                    if rv_now < thr_now and proxy_now < FALSE_COMMIT_PROXY_THRESHOLD:
                        ep_false_commits += 1
                        n_false_commits += 1

                if agent.e3._committed_trajectory is not None:
                    ep_committed += 1
                if agent.beta_gate.is_elevated:
                    ep_elevated += 1

                _, _, done, _, obs_dict = env.step(action_idx)

                # Push nav_competence proxy AFTER env.step so the next tick's
                # commit-entry consultation reads the updated readiness. With
                # use_mech090_readiness_conjunction=False, agent.commit_readiness
                # is None and we skip the push entirely.
                if push_nav_competence and agent.commit_readiness is not None:
                    proxy = compute_nav_competence_proxy(agent)
                    agent.commit_readiness.notify_outcome(proxy)

                p2_step_global += 1
                total_p2_steps += 1
                if done:
                    break

            # Capture per-episode counter snapshots BEFORE the next agent.reset
            # zeroes them out.
            ep_admits = int(agent.beta_gate.get_state().get(
                "mech090_n_elevation_admitted", 0
            )) - admits_before_ep
            ep_blocks_score = int(agent.beta_gate.get_state().get(
                "mech090_n_elevation_blocked", 0
            ))
            ep_blocks_nav = (
                int(agent.commit_readiness.get_state().get("n_blocks_emitted", 0))
                if agent.commit_readiness is not None else 0
            )
            n_admits_p2_accum += ep_admits
            n_blocks_score_p2_accum += ep_blocks_score
            n_blocks_nav_p2_accum += ep_blocks_nav

            total_committed_steps += ep_committed
            total_beta_elevated += ep_elevated
            per_episode_summary.append({
                "episode": ep,
                "committed_steps": ep_committed,
                "beta_elevated_steps": ep_elevated,
                "false_commits": ep_false_commits,
                "n_admits_during_ep": ep_admits,
                "n_blocks_score_during_ep": ep_blocks_score,
                "n_blocks_nav_during_ep": ep_blocks_nav,
            })

    n_admits_p2 = n_admits_p2_accum
    n_blocks_score_p2 = n_blocks_score_p2_accum
    n_blocks_nav_p2 = n_blocks_nav_p2_accum

    hold_rate = (
        total_beta_elevated / total_committed_steps
        if total_committed_steps > 0 else 0.0
    )
    false_commit_rate = (
        n_false_commits / n_admits_p2 if n_admits_p2 > 0 else 0.0
    )
    final_readiness = (
        float(agent.commit_readiness.get_readiness())
        if agent.commit_readiness is not None else 1.0
    )
    final_rv = float(getattr(agent.e3, "_running_variance", 0.0))

    return {
        "total_p2_steps": total_p2_steps,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "hold_rate": hold_rate,
        "n_commit_entries": n_admits_p2,
        "n_commit_blocks_score_margin": n_blocks_score_p2,
        "n_commit_blocks_nav_competence": n_blocks_nav_p2,
        "n_false_commits": n_false_commits,
        "false_commit_rate": false_commit_rate,
        "mean_time_to_commit_from_p2_start": (
            time_to_first_commit if time_to_first_commit != math.inf
            else float("inf")
        ),
        "rv_at_first_commit": rv_at_first_commit,
        "proxy_at_first_commit": proxy_at_first_commit,
        "final_readiness_value": final_readiness,
        "final_running_variance": final_rv,
        "per_episode": per_episode_summary,
    }


# ---------------------------------------------------------------------------
# Per-arm per-seed runner
# ---------------------------------------------------------------------------

def run_arm_seed(
    arm: Dict,
    seed: int,
    dry_run: bool,
    total_run_idx: int,
) -> Dict:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    p0_budget = 5 if dry_run else P0_BUDGET
    p2_eps = 2 if dry_run else P2_EPISODES
    p2_steps = 20 if dry_run else P2_STEPS_PER_EPISODE
    p0_steps = 20 if dry_run else P0_STEPS_PER_EPISODE

    arm_label = arm["name"]
    push_nav = bool(arm["use_nav_competence_conjunction"])

    # Boundary line for runner progress display.
    print(f"Seed {seed} Condition {arm_label}", flush=True)

    agent = REEAgent(make_arm_cfg(arm)).to(device)
    easy_env = make_easy_env(seed)

    # Train denominator for runner progress: total training episodes per arm
    # x seed run (just P0 here; P2 is frozen eval).
    train_total = p0_budget

    print(
        f"  [train] {arm_label} seed={seed} ep 0/{train_total} (P0 warmup start)",
        flush=True,
    )
    p0: P0Result = run_p0_warmup(
        agent, easy_env, device,
        budget=p0_budget,
        steps_per_episode=p0_steps,
        probe_interval=P0_PROBE_INTERVAL if not dry_run else 2,
        mid_probe_frac=P0_MID_PROBE_FRAC,
        convergence_stable_checkpoints=3 if not dry_run else 1,
        threshold_relaxation=0.0,
    )
    print(
        f"  [train] {arm_label} seed={seed} ep {p0.n_episodes}/{train_total}"
        f"  (P0 done: converged={p0.converged} aborted={p0.aborted}"
        f" final_rv={p0.final_rv:.5f})",
        flush=True,
    )

    rv_crossed_in_p0 = (
        p0.final_rv < float(getattr(agent.e3, "commit_threshold", 0.40))
    )

    if p0.aborted:
        # Abort: rv never crossed. The gate cannot fire (commit-entry never
        # attempted). Record zero P2 metrics for this seed.
        p2_metrics = {
            "total_p2_steps": 0,
            "total_committed_steps": 0,
            "total_beta_elevated": 0,
            "hold_rate": 0.0,
            "n_commit_entries": 0,
            "n_commit_blocks_score_margin": 0,
            "n_commit_blocks_nav_competence": 0,
            "n_false_commits": 0,
            "false_commit_rate": 0.0,
            "mean_time_to_commit_from_p2_start": float("inf"),
            "rv_at_first_commit": None,
            "proxy_at_first_commit": None,
            "final_readiness_value": (
                float(agent.commit_readiness.get_readiness())
                if agent.commit_readiness is not None else 1.0
            ),
            "final_running_variance": float(getattr(agent.e3, "_running_variance", 0.0)),
            "per_episode": [],
        }
    else:
        target_env = make_target_env(seed)
        p2_metrics = run_p2_with_metrics(
            agent, target_env, device,
            n_eps=p2_eps,
            steps_per_episode=p2_steps,
            push_nav_competence=push_nav,
        )

    # Total committed steps across P0 + P2 (P0 doesn't expose per-ep
    # commit count; only P2 is measured here, matching the chip spec).
    total_committed_steps = p2_metrics["total_committed_steps"]
    mean_p2_hold_rate = p2_metrics["hold_rate"]

    cell_result = {
        "arm_name": arm_label,
        "seed": seed,
        "arm_flags": {
            "use_score_margin_gate": bool(arm["use_score_margin_gate"]),
            "use_nav_competence_conjunction": bool(arm["use_nav_competence_conjunction"]),
            "commit_readiness_initial": float(arm["commit_readiness_initial"]),
            "score_margin_floor": SCORE_MARGIN_FLOOR,
            "nav_competence_floor": NAV_COMPETENCE_FLOOR,
        },
        "p0": {
            "converged": bool(p0.converged),
            "aborted": bool(p0.aborted),
            "abort_reason": p0.abort_reason,
            "n_episodes": int(p0.n_episodes),
            "final_rv": float(p0.final_rv),
            "commit_threshold_used": float(p0.commit_threshold_used),
            "rv_crossed": bool(rv_crossed_in_p0),
        },
        "p2": p2_metrics,
        "summary": {
            "total_committed_steps": int(total_committed_steps),
            "n_commit_entries": int(p2_metrics["n_commit_entries"]),
            "n_commit_blocks_score_margin": int(p2_metrics["n_commit_blocks_score_margin"]),
            "n_commit_blocks_nav_competence": int(p2_metrics["n_commit_blocks_nav_competence"]),
            "false_commit_rate": float(p2_metrics["false_commit_rate"]),
            "mean_time_to_commit_from_p2_start": float(p2_metrics["mean_time_to_commit_from_p2_start"]),
            "final_readiness_value": float(p2_metrics["final_readiness_value"]),
            "final_running_variance": float(p2_metrics["final_running_variance"]),
            "mean_p2_hold_rate": float(mean_p2_hold_rate),
        },
    }

    # Verdict per cell -- progress instrumentation requirement. Per-cell verdict
    # is purely informational (the experiment verdict is computed at aggregate
    # level); use PASS sentinel here to mark cell completion.
    print(
        f"[run {total_run_idx + 1}/{TOTAL_RUNS}] {arm_label} seed={seed}"
        f"  commits={cell_result['summary']['n_commit_entries']}"
        f"  blocks_score={cell_result['summary']['n_commit_blocks_score_margin']}"
        f"  blocks_nav={cell_result['summary']['n_commit_blocks_nav_competence']}"
        f"  false_commit_rate={cell_result['summary']['false_commit_rate']:.3f}"
        f"  final_readiness={cell_result['summary']['final_readiness_value']:.3f}"
        f"  final_rv={cell_result['summary']['final_running_variance']:.5f}",
        flush=True,
    )
    print(f"verdict: PASS", flush=True)
    return cell_result


# ---------------------------------------------------------------------------
# Aggregation + acceptance evaluation
# ---------------------------------------------------------------------------

def aggregate_arm(cells: List[Dict]) -> Dict:
    total_steps = sum(c["p2"]["total_p2_steps"] for c in cells)
    total_commits = sum(c["summary"]["n_commit_entries"] for c in cells)
    total_blocks_nav = sum(c["summary"]["n_commit_blocks_nav_competence"] for c in cells)
    total_blocks_score = sum(c["summary"]["n_commit_blocks_score_margin"] for c in cells)

    false_rates = [c["summary"]["false_commit_rate"] for c in cells]
    times = [
        c["summary"]["mean_time_to_commit_from_p2_start"]
        for c in cells
        if c["summary"]["mean_time_to_commit_from_p2_start"] != float("inf")
    ]
    p50 = float(np.percentile(times, 50)) if times else float("inf")
    p90 = float(np.percentile(times, 90)) if times else float("inf")

    return {
        "n_seeds": len(cells),
        "total_p2_steps": int(total_steps),
        "total_commits": int(total_commits),
        "total_blocks_score_margin": int(total_blocks_score),
        "total_blocks_nav_competence": int(total_blocks_nav),
        "commit_rate_across_seeds": (
            float(total_commits) / total_steps if total_steps > 0 else 0.0
        ),
        "false_commit_rate_across_seeds": (
            float(np.mean(false_rates)) if false_rates else 0.0
        ),
        "time_to_commit_p50": p50,
        "time_to_commit_p90": p90,
    }


def evaluate_acceptance(arm_aggs: Dict[str, Dict]) -> Tuple[bool, Dict]:
    arm0 = arm_aggs.get("ARM_0_BASELINE_BOTH_OFF", {})
    arm1 = arm_aggs.get("ARM_1_SCORE_MARGIN_ONLY", {})
    arm2 = arm_aggs.get("ARM_2_NAV_COMPETENCE_ONLY", {})
    arm3 = arm_aggs.get("ARM_3_BOTH_GATES_ON", {})

    # C1 baseline fires
    c1 = arm0.get("commit_rate_across_seeds", 0.0) > C1_BASELINE_MIN_COMMIT_RATE

    # C2 score_margin discriminates
    arm0_false = arm0.get("false_commit_rate_across_seeds", 0.0)
    arm1_false = arm1.get("false_commit_rate_across_seeds", 0.0)
    # If baseline has no false commits the ratio test is vacuously NA; treat
    # as a non-failing observation (cannot discriminate what isn't there).
    if arm0_false <= 0.0:
        c2 = False
        c2_note = "baseline false_commit_rate=0; cannot demonstrate reduction"
    else:
        c2 = arm1_false < arm0_false * C2_SCORE_MARGIN_REDUCTION_RATIO
        c2_note = f"arm1={arm1_false:.4f} vs arm0*{C2_SCORE_MARGIN_REDUCTION_RATIO}={arm0_false * C2_SCORE_MARGIN_REDUCTION_RATIO:.4f}"

    # C3 nav_competence fires
    arm2_blocks = arm2.get("total_blocks_nav_competence", 0)
    arm2_commit_rate = arm2.get("commit_rate_across_seeds", 0.0)
    arm0_commit_rate = arm0.get("commit_rate_across_seeds", 0.0)
    c3 = (
        arm2_blocks >= C3_MIN_NAV_BLOCKS
        and arm2_commit_rate < arm0_commit_rate
    )

    # C4 conjunction suppresses degenerate
    arm3_false = arm3.get("false_commit_rate_across_seeds", 0.0)
    arm3_commit_rate = arm3.get("commit_rate_across_seeds", 0.0)
    c4 = (
        arm3_false < C4_MAX_CONJUNCTION_FALSE_COMMIT_RATE
        and arm3_commit_rate > 0.0
    )

    passes = c1 and (c2 or c3) and c4
    detail = {
        "C1_baseline_fires": {
            "passed": bool(c1),
            "arm0_commit_rate": arm0.get("commit_rate_across_seeds", 0.0),
            "threshold": C1_BASELINE_MIN_COMMIT_RATE,
        },
        "C2_score_margin_discriminates": {
            "passed": bool(c2),
            "arm0_false_commit_rate": arm0_false,
            "arm1_false_commit_rate": arm1_false,
            "ratio_threshold": C2_SCORE_MARGIN_REDUCTION_RATIO,
            "note": c2_note,
        },
        "C3_nav_competence_fires": {
            "passed": bool(c3),
            "arm2_blocks_nav": arm2_blocks,
            "arm2_commit_rate": arm2_commit_rate,
            "arm0_commit_rate": arm0_commit_rate,
            "min_blocks_threshold": C3_MIN_NAV_BLOCKS,
        },
        "C4_conjunction_suppresses_degenerate": {
            "passed": bool(c4),
            "arm3_false_commit_rate": arm3_false,
            "arm3_commit_rate": arm3_commit_rate,
            "max_false_rate_threshold": C4_MAX_CONJUNCTION_FALSE_COMMIT_RATE,
        },
        "rule": "PASS = C1 AND (C2 OR C3) AND C4",
        "overall_pass": bool(passes),
    }
    return passes, detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Tuple[str, Optional[Path]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Short run (5 ep P0, 2 ep P2 single seed) to test wiring.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Override SEEDS list (default: 42 43 44).",
    )
    parser.add_argument(
        "--arms", nargs="+", type=int, default=None,
        help="Subset of arm indices [0..3] (default: all).",
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    if args.seeds is not None:
        seeds = args.seeds
    elif dry_run:
        seeds = SEEDS[:1]
    else:
        seeds = SEEDS

    if args.arms is not None:
        arms_to_run = [ARMS[i] for i in args.arms if 0 <= i < len(ARMS)]
    else:
        arms_to_run = ARMS

    print(f"[{QUEUE_ID}] MECH-090 R-c commit-entry readiness conjunction (4-arm)", flush=True)
    print(f"  dry_run={dry_run}  seeds={seeds}  n_arms={len(arms_to_run)}", flush=True)
    print(
        f"  SCORE_MARGIN_FLOOR={SCORE_MARGIN_FLOOR}"
        f"  NAV_COMPETENCE_FLOOR={NAV_COMPETENCE_FLOOR}"
        f"  NAV_COMPETENCE_INITIAL_FAILCLOSED={NAV_COMPETENCE_INITIAL_FAILCLOSED}",
        flush=True,
    )

    per_cell_results: List[Dict] = []
    total_run_idx = 0
    for arm in arms_to_run:
        for seed in seeds:
            cell = run_arm_seed(arm, seed, dry_run, total_run_idx)
            per_cell_results.append(cell)
            total_run_idx += 1

    # Aggregate per arm
    arm_aggs: Dict[str, Dict] = {}
    for arm in arms_to_run:
        cells = [c for c in per_cell_results if c["arm_name"] == arm["name"]]
        arm_aggs[arm["name"]] = aggregate_arm(cells)

    # Acceptance
    passes, accept_detail = evaluate_acceptance(arm_aggs)
    outcome = "PASS" if passes else "FAIL"

    # Summary print
    print(f"\n[{QUEUE_ID}] === ARM AGGREGATES ===", flush=True)
    for name, agg in arm_aggs.items():
        print(
            f"  {name}: commit_rate={agg['commit_rate_across_seeds']:.5f}"
            f" false_rate={agg['false_commit_rate_across_seeds']:.3f}"
            f" blocks_score={agg['total_blocks_score_margin']}"
            f" blocks_nav={agg['total_blocks_nav_competence']}",
            flush=True,
        )
    print(f"\n[{QUEUE_ID}] === ACCEPTANCE ===", flush=True)
    for cname, cdetail in accept_detail.items():
        if isinstance(cdetail, dict) and "passed" in cdetail:
            print(f"  {cname}: {'PASS' if cdetail['passed'] else 'FAIL'}", flush=True)
    print(f"[{QUEUE_ID}] Experiment: {outcome}", flush=True)

    if dry_run:
        print(f"[{QUEUE_ID}] DRY RUN -- not writing evidence.", flush=True)
        return outcome, None

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = EVIDENCE_DIR / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "K=never (SleepLoopManager disabled)",
        "claim_ids": CLAIM_IDS,
        "supersedes": "V3-EXQ-592c",
        "outcome": outcome,
        "evidence_direction": "supports" if passes else "does_not_support",
        "thresholds": {
            "score_margin_floor": SCORE_MARGIN_FLOOR,
            "nav_competence_floor": NAV_COMPETENCE_FLOOR,
            "nav_competence_initial_failclosed": NAV_COMPETENCE_INITIAL_FAILCLOSED,
            "false_commit_proxy_threshold": FALSE_COMMIT_PROXY_THRESHOLD,
            "c1_baseline_min_commit_rate": C1_BASELINE_MIN_COMMIT_RATE,
            "c2_score_margin_reduction_ratio": C2_SCORE_MARGIN_REDUCTION_RATIO,
            "c3_min_nav_blocks": C3_MIN_NAV_BLOCKS,
            "c4_max_conjunction_false_commit_rate": C4_MAX_CONJUNCTION_FALSE_COMMIT_RATE,
        },
        "arm_aggregates": arm_aggs,
        "acceptance": accept_detail,
        "per_cell_results": per_cell_results,
        "notes": (
            "MECH-090 R-c commit-entry readiness conjunction validation, "
            "expanded 4-arm successor to V3-EXQ-592b/c (both 2-arm, both FAIL). "
            "First-ever validation of the nav_competence axis (landed 2026-05-29 "
            "as ree_core/policy/commit_readiness.py + REEConfig "
            "use_mech090_readiness_conjunction master + AND-composition at both "
            "elevate sites in agent.py select_action). claim_ids=[MECH-090] only "
            "per CLAUDE.md claim_ids Accuracy Rule -- SD-034/MECH-266/267/268 "
            "are transitively unblocked by GAP-4 substrate completion but not "
            "directly tested by this substrate-readiness diagnostic. "
            "PASS unblocks commitment_closure:GAP-4 (substrate_landed_pending_validation "
            "-> done substrate-side) and queues *b cohort behavioural arms "
            "(V3-EXQ-460b/461/463b/464b/466b/467b/468b). FAIL routes per the "
            "interpretation grid in the experiment docstring. Cross-link "
            "IGW-20260531-021. Predecessor synthesis: REE_assembly/evidence/"
            "literature/targeted_review_connectome_mech_090/synthesis.md commit "
            "9e68c5ca8a."
        ),
    }

    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{QUEUE_ID}] Evidence written -> {out_path}", flush=True)

    return outcome, out_path


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
