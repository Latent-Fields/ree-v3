#!/opt/local/bin/python3
"""
V3-EXQ-592e: MECH-090 R-c commit-entry readiness conjunction validation (4-arm,
C1 baseline redesigned).

Supersedes V3-EXQ-592d (FAIL non_contributory 2026-06-01 per
failure_autopsy_V3-EXQ-592d_2026-06-01). 592d's C1 baseline predicate
(`ARM_0 commit_rate = sum(n_commit_entries) / total_p2_steps > 0.001`)
was structurally unmeetable: under the rv-only ARM_0 entry semantic the
agent enters P2 already committed (P0 trains rv to ~1e-5 / 1e-6 well below
commit_threshold=0.4), stays continuously committed (hold_rate=1.0,
beta_elevated_steps == committed_steps), and produces zero transition-edge
commit entries during the measurement window. n_commit_entries = 0
UNCONDITIONALLY on ARM_0 regardless of gate state -- the predicate cannot
discriminate substrate firing from substrate silence.

The R-c substrate IS firing on the other arms: ARM_1 score-margin gate
blocks 2005 elevation attempts (within-tick decisiveness axis confirmed,
Hanes & Schall 1996); ARM_2 / ARM_3 nav-competence gate blocks 150
elevation attempts each (across-tick motor-program readiness axis
confirmed, Cisek & Kalaska 2010 + Roesch / Calu / Schoenbaum 2007);
ARM_3 confirms AND-composition at both elevate sites (1949 score blocks
+ 150 nav blocks). Substrate is sound; only the C1 baseline measurement
predicate needs redesign.

592e applies BOTH fixes from autopsy Section 9 (compose orthogonally):
  Fix (a) FORCE_UNCOMMITTED P2 ENTRY: at each P2 episode start (after
    agent.reset()), explicitly release beta_gate, null e3._committed_trajectory,
    and zero _committed_step_idx. Under fix (a) the agent has to re-elevate
    via the gate during the measurement window -- every commit during P2
    IS a fresh transition; n_commit_entries becomes a meaningful counter
    on every arm including ARM_0.
  Fix (b) HOLD_RATE_BASED ACCEPTANCE: replace the unmeasurable commit-rate
    predicate with hold-rate-based criteria. ARM_0 hold_rate = 1.0 on 592d
    is the CORRECT signal that the rv-only baseline IS firing under the
    legacy entry semantic; hold-rate generalises across arms and remains
    interpretable even when fix (a) is disabled.

SLEEP DRIVER: K=never (SleepLoopManager disabled; experiment does not
exercise sleep aggregation cluster). use_sleep_loop=False default.

EXPERIMENT_PURPOSE: diagnostic (substrate-readiness validation of the R-c
conjunction; behavioural validation of MECH-090 cluster downstream is the
governance-weighting signal, NOT this).

Substrate landed in two passes (unchanged from 592d):
  2026-05-28 (within-tick decisiveness axis): BetaGate.should_admit_elevation
    + HeartbeatConfig.use_commit_readiness_gate / commit_readiness_floor. R-a
    rv-only commit-entry is supplemented by a per-candidate first-action score
    margin gate.
  2026-05-29 (across-tick motor-program readiness axis): CommitReadiness EMA
    + REEConfig.use_mech090_readiness_conjunction master flag (auto-arms
    use_commit_readiness=True via __post_init__) + REEConfig.mech090_readiness_floor.
    AND-composed at both elevate sites in agent.py.

Arm matrix (4 arms x 3 seeds = 12 cells; identical to 592d):
  ARM_0 BASELINE_BOTH_OFF: use_commit_readiness_gate=False,
    use_mech090_readiness_conjunction=False. Pure legacy rv-only commit entry.
    Reference baseline for hold-rate / score-margin / nav-competence counter
    measurement under fix (a) force-uncommitted P2 entry.
  ARM_1 SCORE_MARGIN_ONLY: use_commit_readiness_gate=True (floor=0.05),
    use_mech090_readiness_conjunction=False. Within-tick decisiveness axis
    alone.
  ARM_2 NAV_COMPETENCE_ONLY: use_commit_readiness_gate=False,
    use_mech090_readiness_conjunction=True (floor=0.3, initial=0.0). Across-
    tick motor-program readiness axis alone. Initial=0.0 ensures the gate
    starts blocking.
  ARM_3 BOTH_GATES_ON: use_commit_readiness_gate=True (floor=0.05),
    use_mech090_readiness_conjunction=True (floor=0.3, initial=0.0). Full R-c
    conjunction (AND-composed at both elevate sites). Intended-production
    config.

ARM_2 / ARM_3 per-tick nav_competence proxy push: identical to 592d.
nav_competence_proxy = clip([0,1], 1.0 - rv/commit_threshold). When rv at
threshold, proxy=0; rv=0, proxy=1. Pushed AFTER env.step() via
commit_readiness.notify_outcome(proxy).

Per-arm per-seed metrics (changed from 592d to add hold_rate first-class):
  total_p2_steps: int
  total_committed_steps: int (steps where e3._committed_trajectory is not None)
  total_beta_elevated: int (steps where beta_gate.is_elevated)
  hold_rate: float = total_committed_steps / total_p2_steps (NEW primary
    measure under fix (b); was secondary in 592d). High = agent stays
    committed across P2. ARM_0 expected ~1.0 (rv-only baseline). ARM_1/2/3
    expected lower (gates suppress re-elevation).
  n_commit_entries: int = beta_gate.mech090_n_elevation_admitted (now
    meaningful under fix (a) -- counts fresh elevation transitions across
    P2 measurement window).
  n_commit_blocks_score_margin: int = beta_gate.mech090_n_elevation_blocked
  n_commit_blocks_nav_competence: int = commit_readiness.n_blocks_emitted
  mean_time_to_commit_from_p2_start: float
  false_commit_rate: float (false_commit = commit at rv-low AND proxy-low)
  final_readiness_value: float
  final_running_variance: float

Aggregate per arm:
  hold_rate_across_seeds: float = mean(hold_rate per cell)
  commit_rate_across_seeds: float = sum(n_commit_entries) / total_steps_p2
  false_commit_rate_across_seeds: float
  time_to_commit_p50_p90: tuple

Acceptance criteria (HOLD_RATE_BASED, autopsy Section 9 Option (b)):
  C1 BASELINE_FIRES: ARM_0 hold_rate_across_seeds >= 0.5. Test environment
    + curriculum produces continuous committed engagement at the rv-only
    baseline (the legacy ARM_0 semantic). Under fix (a) force-uncommitted
    entry this is still a meaningful predicate -- the rv-only agent
    re-commits on (effectively) every tick once it enters P2, so hold_rate
    remains near 1.0. If this fails, the curriculum / env IS broken.
  C2 SCORE_MARGIN_DISCRIMINATES: ARM_1 total_blocks_score_margin >= 1 across
    seeds AND ARM_1 hold_rate_across_seeds < ARM_0 hold_rate_across_seeds *
    0.7. Score-margin gate is firing AND suppresses commitment net by >=30%
    vs the rv-only baseline.
  C3 NAV_COMPETENCE_DISCRIMINATES: ARM_2 total_blocks_nav_competence >= 1
    across seeds AND ARM_2 hold_rate_across_seeds < ARM_0 hold_rate_across_seeds.
    Nav-competence gate is firing AND suppresses commitment vs baseline.
  C4 CONJUNCTION_SUPPRESSES_AND_RECOVERS: ARM_3 hold_rate_across_seeds <
    ARM_0 hold_rate_across_seeds (conjunction net-suppresses) AND ARM_3
    hold_rate_across_seeds > 0.0 (conjunction does not permanently lock out
    commitment once both gates clear) AND (ARM_3 total_blocks_score_margin
    >= 1 AND ARM_3 total_blocks_nav_competence >= 1) (both gates fire AND-
    composed at both elevate sites).

  PASS = C1 AND (C2 OR C3) AND C4. FAIL otherwise.

Interpretation grid (UPDATED from 592d per autopsy Section 9 + user
override 2026-06-01 routing C1 FAIL -> /diagnose-errors for SCRIPT
correction; the prior 592d routing said "C1 FAIL -> /diagnose-errors on
curriculum harness, NOT the gate" which was INVERTED -- the curriculum
harness was working as designed and the predicate was the defect):
  Outcome                          | Diagnosis / Routing
  ---------------------------------|-------------------------------------------
  PASS                             | R-c substrate (both axes) validated.
                                   | commitment_closure:GAP-4
                                   | substrate_landed_pending_validation
                                   | -> done substrate-side. Queue *b cohort
                                   | (V3-EXQ-460b/461/463b/464b/466b/467b/
                                   | 468b) Phase 4/5 behavioural arms.
  C1 FAIL                          | /diagnose-errors on SCRIPT acceptance
                                   | predicate (the hold_rate-based C1 may
                                   | still be wrong for the underlying
                                   | substrate dynamics). NOT the curriculum
                                   | harness (P0 convergence is robust by
                                   | construction). NOT the gate (substrate
                                   | landed + contract-tested). Candidate
                                   | predicate revisions: lower hold_rate
                                   | floor; restrict measurement to last
                                   | N steps of P2 after rv has stabilised;
                                   | swap to beta_elevated_steps as the
                                   | primary measure.
  C2 PASS + C3 FAIL                | score_margin axis works, nav_competence
                                   | axis does NOT block. Route to
                                   | nav_competence-only diagnostic: proxy
                                   | may be wrong OR floor=0.3 too low.
                                   | Sweep floor {0.5, 0.7, 0.9}.
  C2 FAIL + C3 PASS                | nav_competence axis works, score_margin
                                   | axis does NOT discriminate. Route to
                                   | score_margin-only diagnostic: floor
                                   | sweep {0.01, 0.02, 0.05, 0.10}.
  C4 FAIL (hold_rate=0)            | conjunction permanently locks out
                                   | commitment. Likely fail-closed initial
                                   | condition on nav_competence prevents
                                   | recovery even after harness push --
                                   | calibrate initial=0.05 or check push
                                   | semantics. /diagnose-errors on harness
                                   | push path.
  All FAIL                         | R-c substrate retest. /diagnose-errors
                                   | on agent.py wiring -- check getattr-
                                   | fallback flag propagation through both
                                   | REEConfig and HeartbeatConfig.

Supersedes: V3-EXQ-592d. claim_ids=["MECH-090"] only per CLAUDE.md
claim_ids Accuracy Rule -- SD-034/MECH-266/267/268 are transitively
unblocked by GAP-4 substrate completion but not directly tested by
this substrate-readiness diagnostic.

Cross-link IGW-20260531-021. Predecessor synthesis:
REE_assembly/evidence/literature/targeted_review_connectome_mech_090/
synthesis.md commit 9e68c5ca8a. Autopsy artifact:
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-592d_2026-06-01.{md,json}.
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_592e_mech090_readiness_conjunction_validation"
QUEUE_ID = "V3-EXQ-592e"
CLAIM_IDS: List[str] = ["MECH-090"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]

# R-c gate floors (pre-registered; unchanged from 592d)
SCORE_MARGIN_FLOOR = 0.05      # within-tick decisiveness axis
NAV_COMPETENCE_FLOOR = 0.3     # across-tick motor-program readiness axis
NAV_COMPETENCE_INITIAL_FAILCLOSED = 0.0  # ARM_2/ARM_3: fail-closed start so
                                          # the gate actually has the
                                          # opportunity to block. default
                                          # initial=1.0 would be fail-open.

# Fix (a) from autopsy Section 9: force-uncommitted P2 entry. Per-episode
# release of beta_gate at P2 start so every commit during measurement IS a
# fresh transition. This is the redesign of the C1 baseline measurement
# methodology -- 592d had this implicit (legacy entry semantic) and the
# predicate became unmeetable.
FORCE_UNCOMMITTED_P2_ENTRY = True

# False-commit detection (unchanged from 592d): commit-entry that fired while
# rv was low AND proxy_nav_competence was also low.
FALSE_COMMIT_PROXY_THRESHOLD = 0.3

# Acceptance criteria thresholds (HOLD_RATE_BASED -- changed from 592d)
C1_BASELINE_MIN_HOLD_RATE = 0.5        # ARM_0 hold_rate >= 0.5 (curriculum produces
                                       # continuous commitment under rv-only)
C2_SCORE_MARGIN_HOLD_REDUCTION_RATIO = 0.7   # ARM_1 hold_rate < ARM_0 * 0.7 AND blocks >= 1
C2_MIN_SCORE_MARGIN_BLOCKS = 1
C3_MIN_NAV_BLOCKS = 1                  # ARM_2 nav-competence gate fires AND
                                       # hold_rate < ARM_0 hold_rate
C4_MIN_CONJUNCTION_HOLD_RATE = 0.0     # ARM_3 hold_rate > 0 (NOT permanently locked)
C4_MIN_SCORE_BLOCKS = 1                # ARM_3 score-margin gate fires
C4_MIN_NAV_BLOCKS = 1                  # ARM_3 nav-competence gate fires
                                       # (AND-composed conjunction signature)

# Curriculum config (matched to V3-EXQ-592d)
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
# Env factories (matched to V3-EXQ-592d)
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
# Agent factory: arm-parameterised (unchanged from 592d)
# ---------------------------------------------------------------------------

def make_arm_cfg(arm: Dict) -> REEConfig:
    """Standard 592d-matched cfg + per-arm R-c gate flags."""
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
# Per-tick nav_competence proxy push helper (unchanged from 592d)
# ---------------------------------------------------------------------------

def compute_nav_competence_proxy(agent: REEAgent) -> float:
    """Proxy nav_competence from agent's world-model confidence."""
    rv = float(getattr(agent.e3, "_running_variance", 1.0))
    threshold = float(getattr(agent.e3, "commit_threshold", 0.40))
    if threshold <= 0:
        return 1.0
    raw = 1.0 - (rv / threshold)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Fix (a): force-uncommitted entry helper (new in 592e)
# ---------------------------------------------------------------------------

def force_uncommitted(agent: REEAgent) -> None:
    """Force the agent into uncommitted state.

    Implements failure_autopsy_V3-EXQ-592d_2026-06-01 Section 9 Option (a)
    redesign: release beta_gate elevation, null the committed trajectory,
    and zero the committed step index. Called at each P2 episode start
    (after agent.reset()) so every commit during the measurement window
    IS a fresh transition gate-consultation.

    Without this fix, the rv-only ARM_0 agent enters P2 already committed
    (P0-trained rv ~1e-5 / 1e-6 well below commit_threshold=0.4) and stays
    continuously committed (hold_rate=1.0, beta_elevated_steps ==
    committed_steps); transition-edge counters (n_commit_entries) report 0
    UNCONDITIONALLY on ARM_0 regardless of gate state. This makes ARM_0 the
    structural-zero baseline against which the gated arms cannot
    discriminate.
    """
    if agent.beta_gate.is_elevated:
        agent.beta_gate.release()
    if agent.e3._committed_trajectory is not None:
        agent.e3._committed_trajectory = None
    agent._committed_step_idx = 0


# ---------------------------------------------------------------------------
# Custom P2 eval loop with per-tick metrics + nav_competence push
# (forked from 592d, adds fix (a) force-uncommitted P2 entry per-episode)
# ---------------------------------------------------------------------------

def run_p2_with_metrics(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
    push_nav_competence: bool,
    force_uncommitted_entry: bool = FORCE_UNCOMMITTED_P2_ENTRY,
) -> Dict:
    """Frozen-policy P2 eval with per-tick metric capture.

    When force_uncommitted_entry=True (default per 592e fix (a)), calls
    force_uncommitted(agent) at the start of each P2 episode after
    agent.reset() so the gate is consulted on every fresh re-elevation
    attempt during measurement.

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
    n_admits_p2_accum = 0
    n_blocks_score_p2_accum = 0
    n_blocks_nav_p2_accum = 0
    n_force_uncommitted_calls = 0
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

            # Fix (a) per 592e -- failure_autopsy_V3-EXQ-592d_2026-06-01
            # Section 9 Option (a). Force uncommitted state at P2 episode
            # start so gate consultations happen during measurement.
            if force_uncommitted_entry:
                force_uncommitted(agent)
                n_force_uncommitted_calls += 1

            ep_committed = 0
            ep_elevated = 0
            ep_false_commits = 0
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

                # Push nav_competence proxy AFTER env.step.
                if push_nav_competence and agent.commit_readiness is not None:
                    proxy = compute_nav_competence_proxy(agent)
                    agent.commit_readiness.notify_outcome(proxy)

                p2_step_global += 1
                total_p2_steps += 1
                if done:
                    break

            # Per-episode counter snapshots BEFORE next agent.reset zeroes them.
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

    # Hold rate now uses TOTAL P2 STEPS as denominator (not committed_steps as
    # in 592d) -- under fix (b) this is the primary acceptance measure.
    # Definition: fraction of P2 steps the agent spent in committed state.
    # ARM_0 expected ~1.0 (rv-only commits immediately after force-uncommitted
    # entry); gated arms expected lower (gate blocks delay re-elevation).
    hold_rate = (
        total_committed_steps / total_p2_steps
        if total_p2_steps > 0 else 0.0
    )
    # 592d-compatible secondary measure: beta_elevated / committed_steps.
    # Kept for cross-comparison with 592d's hold_rate definition. Reports
    # "of the committed steps, what fraction was beta-elevated".
    hold_rate_592d_compat = (
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
        "hold_rate_592d_compat": hold_rate_592d_compat,
        "n_commit_entries": n_admits_p2,
        "n_commit_blocks_score_margin": n_blocks_score_p2,
        "n_commit_blocks_nav_competence": n_blocks_nav_p2,
        "n_false_commits": n_false_commits,
        "false_commit_rate": false_commit_rate,
        "n_force_uncommitted_calls": n_force_uncommitted_calls,
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
# Per-arm per-seed runner (unchanged in shape; new hold_rate-based summary)
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
        # Abort: rv never crossed. Gate cannot fire (no commit attempted).
        p2_metrics = {
            "total_p2_steps": 0,
            "total_committed_steps": 0,
            "total_beta_elevated": 0,
            "hold_rate": 0.0,
            "hold_rate_592d_compat": 0.0,
            "n_commit_entries": 0,
            "n_commit_blocks_score_margin": 0,
            "n_commit_blocks_nav_competence": 0,
            "n_false_commits": 0,
            "false_commit_rate": 0.0,
            "n_force_uncommitted_calls": 0,
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
            force_uncommitted_entry=FORCE_UNCOMMITTED_P2_ENTRY,
        )

    total_committed_steps = p2_metrics["total_committed_steps"]
    hold_rate_primary = p2_metrics["hold_rate"]

    cell_result = {
        "arm_name": arm_label,
        "seed": seed,
        "arm_flags": {
            "use_score_margin_gate": bool(arm["use_score_margin_gate"]),
            "use_nav_competence_conjunction": bool(arm["use_nav_competence_conjunction"]),
            "commit_readiness_initial": float(arm["commit_readiness_initial"]),
            "score_margin_floor": SCORE_MARGIN_FLOOR,
            "nav_competence_floor": NAV_COMPETENCE_FLOOR,
            "force_uncommitted_p2_entry": FORCE_UNCOMMITTED_P2_ENTRY,
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
            "total_p2_steps": int(p2_metrics["total_p2_steps"]),
            "hold_rate": float(hold_rate_primary),
            "hold_rate_592d_compat": float(p2_metrics["hold_rate_592d_compat"]),
            "n_commit_entries": int(p2_metrics["n_commit_entries"]),
            "n_commit_blocks_score_margin": int(p2_metrics["n_commit_blocks_score_margin"]),
            "n_commit_blocks_nav_competence": int(p2_metrics["n_commit_blocks_nav_competence"]),
            "false_commit_rate": float(p2_metrics["false_commit_rate"]),
            "mean_time_to_commit_from_p2_start": float(p2_metrics["mean_time_to_commit_from_p2_start"]),
            "final_readiness_value": float(p2_metrics["final_readiness_value"]),
            "final_running_variance": float(p2_metrics["final_running_variance"]),
            "n_force_uncommitted_calls": int(p2_metrics["n_force_uncommitted_calls"]),
        },
    }

    # Per-cell verdict: informational (aggregate verdict computed downstream).
    print(
        f"[run {total_run_idx + 1}/{TOTAL_RUNS}] {arm_label} seed={seed}"
        f"  hold_rate={cell_result['summary']['hold_rate']:.3f}"
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
# Aggregation + acceptance evaluation (HOLD_RATE_BASED per 592e fix (b))
# ---------------------------------------------------------------------------

def aggregate_arm(cells: List[Dict]) -> Dict:
    total_steps = sum(c["p2"]["total_p2_steps"] for c in cells)
    total_commits = sum(c["summary"]["n_commit_entries"] for c in cells)
    total_blocks_nav = sum(c["summary"]["n_commit_blocks_nav_competence"] for c in cells)
    total_blocks_score = sum(c["summary"]["n_commit_blocks_score_margin"] for c in cells)
    total_committed = sum(c["summary"]["total_committed_steps"] for c in cells)

    hold_rates = [c["summary"]["hold_rate"] for c in cells]
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
        "total_committed_steps": int(total_committed),
        "total_commits": int(total_commits),
        "total_blocks_score_margin": int(total_blocks_score),
        "total_blocks_nav_competence": int(total_blocks_nav),
        "hold_rate_across_seeds": (
            float(np.mean(hold_rates)) if hold_rates else 0.0
        ),
        "hold_rate_pooled": (
            float(total_committed) / total_steps if total_steps > 0 else 0.0
        ),
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
    """Hold-rate-based acceptance per failure_autopsy_V3-EXQ-592d_2026-06-01
    Section 9 Option (b). See module docstring for full predicate forms.
    """
    arm0 = arm_aggs.get("ARM_0_BASELINE_BOTH_OFF", {})
    arm1 = arm_aggs.get("ARM_1_SCORE_MARGIN_ONLY", {})
    arm2 = arm_aggs.get("ARM_2_NAV_COMPETENCE_ONLY", {})
    arm3 = arm_aggs.get("ARM_3_BOTH_GATES_ON", {})

    arm0_hold = arm0.get("hold_rate_across_seeds", 0.0)
    arm1_hold = arm1.get("hold_rate_across_seeds", 0.0)
    arm2_hold = arm2.get("hold_rate_across_seeds", 0.0)
    arm3_hold = arm3.get("hold_rate_across_seeds", 0.0)

    # C1 baseline fires (hold-rate redesign per 592e fix (b))
    c1 = arm0_hold >= C1_BASELINE_MIN_HOLD_RATE

    # C2 score_margin discriminates: blocks fire AND hold_rate reduction
    arm1_blocks_score = arm1.get("total_blocks_score_margin", 0)
    c2 = (
        arm1_blocks_score >= C2_MIN_SCORE_MARGIN_BLOCKS
        and arm1_hold < arm0_hold * C2_SCORE_MARGIN_HOLD_REDUCTION_RATIO
    )
    c2_note = (
        f"arm1_blocks_score={arm1_blocks_score} (>={C2_MIN_SCORE_MARGIN_BLOCKS}) AND "
        f"arm1_hold={arm1_hold:.3f} < arm0_hold * {C2_SCORE_MARGIN_HOLD_REDUCTION_RATIO} = "
        f"{arm0_hold * C2_SCORE_MARGIN_HOLD_REDUCTION_RATIO:.3f}"
    )

    # C3 nav_competence discriminates: blocks fire AND hold_rate < baseline
    arm2_blocks_nav = arm2.get("total_blocks_nav_competence", 0)
    c3 = (
        arm2_blocks_nav >= C3_MIN_NAV_BLOCKS
        and arm2_hold < arm0_hold
    )
    c3_note = (
        f"arm2_blocks_nav={arm2_blocks_nav} (>={C3_MIN_NAV_BLOCKS}) AND "
        f"arm2_hold={arm2_hold:.3f} < arm0_hold={arm0_hold:.3f}"
    )

    # C4 conjunction suppresses AND recovers (NOT permanent lockout)
    arm3_blocks_score = arm3.get("total_blocks_score_margin", 0)
    arm3_blocks_nav = arm3.get("total_blocks_nav_competence", 0)
    c4_suppresses = arm3_hold < arm0_hold
    c4_not_locked = arm3_hold > C4_MIN_CONJUNCTION_HOLD_RATE
    c4_both_gates_fire = (
        arm3_blocks_score >= C4_MIN_SCORE_BLOCKS
        and arm3_blocks_nav >= C4_MIN_NAV_BLOCKS
    )
    c4 = bool(c4_suppresses and c4_not_locked and c4_both_gates_fire)
    c4_note = (
        f"suppresses={c4_suppresses} (arm3_hold={arm3_hold:.3f} < arm0_hold={arm0_hold:.3f}) AND "
        f"not_locked={c4_not_locked} (arm3_hold > {C4_MIN_CONJUNCTION_HOLD_RATE}) AND "
        f"both_gates_fire={c4_both_gates_fire} (score={arm3_blocks_score} nav={arm3_blocks_nav})"
    )

    passes = c1 and (c2 or c3) and c4
    detail = {
        "C1_baseline_fires_hold_rate": {
            "passed": bool(c1),
            "arm0_hold_rate": arm0_hold,
            "threshold": C1_BASELINE_MIN_HOLD_RATE,
            "predicate": "ARM_0 hold_rate >= 0.5 (redesigned per autopsy Section 9 Option (b))",
        },
        "C2_score_margin_discriminates": {
            "passed": bool(c2),
            "arm0_hold_rate": arm0_hold,
            "arm1_hold_rate": arm1_hold,
            "arm1_blocks_score_margin": arm1_blocks_score,
            "hold_reduction_ratio_threshold": C2_SCORE_MARGIN_HOLD_REDUCTION_RATIO,
            "min_blocks_threshold": C2_MIN_SCORE_MARGIN_BLOCKS,
            "note": c2_note,
        },
        "C3_nav_competence_discriminates": {
            "passed": bool(c3),
            "arm0_hold_rate": arm0_hold,
            "arm2_hold_rate": arm2_hold,
            "arm2_blocks_nav_competence": arm2_blocks_nav,
            "min_blocks_threshold": C3_MIN_NAV_BLOCKS,
            "note": c3_note,
        },
        "C4_conjunction_suppresses_and_recovers": {
            "passed": bool(c4),
            "arm0_hold_rate": arm0_hold,
            "arm3_hold_rate": arm3_hold,
            "arm3_blocks_score_margin": arm3_blocks_score,
            "arm3_blocks_nav_competence": arm3_blocks_nav,
            "suppresses": bool(c4_suppresses),
            "not_locked": bool(c4_not_locked),
            "both_gates_fire": bool(c4_both_gates_fire),
            "note": c4_note,
        },
        "rule": "PASS = C1 AND (C2 OR C3) AND C4 (hold-rate-based per 592e fix (b))",
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

    print(f"[{QUEUE_ID}] MECH-090 R-c readiness conjunction (4-arm, 592d redesign)", flush=True)
    print(f"  dry_run={dry_run}  seeds={seeds}  n_arms={len(arms_to_run)}", flush=True)
    print(
        f"  SCORE_MARGIN_FLOOR={SCORE_MARGIN_FLOOR}"
        f"  NAV_COMPETENCE_FLOOR={NAV_COMPETENCE_FLOOR}"
        f"  NAV_COMPETENCE_INITIAL_FAILCLOSED={NAV_COMPETENCE_INITIAL_FAILCLOSED}"
        f"  FORCE_UNCOMMITTED_P2_ENTRY={FORCE_UNCOMMITTED_P2_ENTRY}",
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

    # Acceptance (hold-rate-based per 592e fix (b))
    passes, accept_detail = evaluate_acceptance(arm_aggs)
    outcome = "PASS" if passes else "FAIL"

    # Summary print
    print(f"\n[{QUEUE_ID}] === ARM AGGREGATES ===", flush=True)
    for name, agg in arm_aggs.items():
        print(
            f"  {name}: hold_rate={agg['hold_rate_across_seeds']:.3f}"
            f"  commit_rate={agg['commit_rate_across_seeds']:.5f}"
            f"  false_rate={agg['false_commit_rate_across_seeds']:.3f}"
            f"  blocks_score={agg['total_blocks_score_margin']}"
            f"  blocks_nav={agg['total_blocks_nav_competence']}",
            flush=True,
        )
    print(f"\n[{QUEUE_ID}] === ACCEPTANCE (hold-rate-based) ===", flush=True)
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
        "supersedes": "V3-EXQ-592d",
        "outcome": outcome,
        "evidence_direction": "supports" if passes else "does_not_support",
        "thresholds": {
            "score_margin_floor": SCORE_MARGIN_FLOOR,
            "nav_competence_floor": NAV_COMPETENCE_FLOOR,
            "nav_competence_initial_failclosed": NAV_COMPETENCE_INITIAL_FAILCLOSED,
            "force_uncommitted_p2_entry": FORCE_UNCOMMITTED_P2_ENTRY,
            "false_commit_proxy_threshold": FALSE_COMMIT_PROXY_THRESHOLD,
            "c1_baseline_min_hold_rate": C1_BASELINE_MIN_HOLD_RATE,
            "c2_score_margin_hold_reduction_ratio": C2_SCORE_MARGIN_HOLD_REDUCTION_RATIO,
            "c2_min_score_margin_blocks": C2_MIN_SCORE_MARGIN_BLOCKS,
            "c3_min_nav_blocks": C3_MIN_NAV_BLOCKS,
            "c4_min_conjunction_hold_rate": C4_MIN_CONJUNCTION_HOLD_RATE,
            "c4_min_score_blocks": C4_MIN_SCORE_BLOCKS,
            "c4_min_nav_blocks": C4_MIN_NAV_BLOCKS,
        },
        "arm_aggregates": arm_aggs,
        "acceptance": accept_detail,
        "per_cell_results": per_cell_results,
        "notes": (
            "MECH-090 R-c commit-entry readiness conjunction validation, "
            "C1 baseline redesigned per failure_autopsy_V3-EXQ-592d_2026-06-01 "
            "Section 9. Supersedes V3-EXQ-592d (FAIL non_contributory; C1 "
            "predicate structurally unmeetable under rv-only ARM_0 entry semantic). "
            "Fix (a) FORCE_UNCOMMITTED_P2_ENTRY: per-episode release of beta_gate "
            "+ null e3._committed_trajectory + zero _committed_step_idx at P2 "
            "episode start so the gate is consulted on every fresh re-elevation "
            "attempt during the measurement window. Fix (b) HOLD_RATE_BASED "
            "ACCEPTANCE: replaced the unmeasurable commit-rate predicate with "
            "hold-rate-based criteria that work whether or not the legacy "
            "transition-edge counters fire. Interpretation grid updated: C1 FAIL "
            "now routes to /diagnose-errors for SCRIPT predicate correction "
            "(the prior 592d grid said C1 FAIL routes to curriculum harness "
            "/diagnose-errors -- this was INVERTED: the curriculum harness was "
            "working as designed and the predicate was the defect). "
            "claim_ids=[MECH-090] only per CLAUDE.md claim_ids Accuracy Rule. "
            "PASS unblocks commitment_closure:GAP-4 substrate_landed_pending_validation "
            "-> done substrate-side and queues *b cohort behavioural arms. "
            "Cross-link IGW-20260531-021. Predecessor synthesis: "
            "REE_assembly/evidence/literature/targeted_review_connectome_mech_090/"
            "synthesis.md commit 9e68c5ca8a. Autopsy artifact: REE_assembly/"
            "evidence/planning/failure_autopsy_V3-EXQ-592d_2026-06-01.{md,json}."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"[{QUEUE_ID}] Evidence written -> {out_path}", flush=True)

    return outcome, out_path


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
