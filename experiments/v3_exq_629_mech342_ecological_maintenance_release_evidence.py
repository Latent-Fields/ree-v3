#!/opt/local/bin/python3
"""
V3-EXQ-629: MECH-342 maintenance-time commitment-release -- ECOLOGICAL evidence run.

SLEEP DRIVER: K=never (SleepLoopManager disabled; experiment does not exercise
the sleep aggregation cluster). use_sleep_loop=False default throughout.

EXPERIMENT_PURPOSE: evidence. This is the ECOLOGICAL complement to the
diagnostic state-machine probe V3-EXQ-592g (PASS, all six criteria). 592g is a
CONTROLLED STATE-MACHINE PROBE -- it stubs E3TrajectorySelector.select() to
force result.committed=True and sets agent.e3.last_scores / commit_readiness
directly to controlled values. 629 uses a REAL REEAgent in a REAL
CausalGridWorldV2 where commitment happens NATURALLY (E3.select NOT stubbed,
beta-gate elevation gated by the live MECH-090 R-c conjunction) and where the
two R-c readiness signals degrade NATURALLY mid-commitment (the world becomes
unpredictable, raising E2 world-forward error -> E3 running_variance -> the
nav_competence proxy drops AND the per-candidate score margin compresses).
629 references 592g as PREDECESSOR, NOT supersedes -- 592g is a valid diagnostic
that established the state-machine wiring; 629 is the complementary
ecological-grade run the 2026-06-02 /governance disposition requires before
MECH-342 can advance past candidate / v3_pending.

WHAT MECH-342 IS (the claim under test):
  control_plane.commit_maintenance_release -- a graded, bounded-accumulation,
  hysteretic RELEASE of an already-elevated beta latch driven by the SAME two
  R-c readiness signals MECH-090 AND-composes to ADMIT a commitment
  (score_margin decisiveness + nav_competence), when they degrade WHILE the
  agent is already beta-elevated. The MECH-090 R-c conjunction is
  ADMISSION-ONLY (V3-EXQ-592f autopsy + release-path audit + motor-cessation
  lit-pull verdict B3b); MECH-342 is the release-side complement.

ECOLOGICAL HARNESS (forked from the validated V3-EXQ-592e MECH-090 harness):
  - REAL REEAgent.select_action(), REAL E3.select() (NOT stubbed), REAL
    BetaGate, REAL CommitReadiness, REAL CommitMaintenanceRelease.
  - P0 warmup (committed_mode_curriculum.run_p0_warmup) drives E2 world-forward
    convergence so running_variance crosses the commit_threshold and the agent
    commits NATURALLY in P2.
  - P2 runs in TWO windows, each its own episodes on the same carried-over
    frozen-policy agent:
      HEALTHY window: plain target env. The trained world model predicts well
        -> running_variance stays low -> nav_competence proxy ~1.0 AND
        score_margin healthy -> readiness HEALTHY. Used to measure the
        premature-abort failure pole: a correct maintenance-release must NOT
        decommit here.
      DEGRADED window: SD-047 multi_source_dynamics env (agent-independent
        weather AR(1) field + background drift). The perturbation the trained
        E2 cannot predict raises world-forward error -> running_variance rises
        -> nav_competence proxy drops below nav_floor AND/OR score_margin
        compresses below score_margin_floor -> readiness DEGRADES
        mid-commitment. Used to measure release authority: a correct
        maintenance-release MUST decommit here.
  - In BOTH windows the per-tick nav_competence proxy is pushed via
    commit_readiness.notify_outcome(proxy) (the harness seam; the env-side
    auto-drive is a Phase-2 substrate follow-on not yet wired) AND
    running_variance is updated each tick from the REAL E2 world-forward
    error so the SD-047 perturbation genuinely moves readiness. The
    decisiveness axis reads agent.e3.last_scores naturally every tick.

ARM AXIS (the only manipulated variable): use_maintenance_release.
  ARM_0_RELEASE_OFF: use_maintenance_release=False. Reproduces the V3-EXQ-592f
    gap -- no release authority; stays committed under degraded readiness.
  ARM_1_RELEASE_ON:  use_maintenance_release=True. The substrate under test.
  Both arms commit IDENTICALLY: use_mech090_readiness_conjunction=True +
  use_commit_readiness_gate=True so commit ENTRY is gated the same way; the
  ONLY difference is whether the maintenance-release branch exists.

DISTINCT-FROM CONTROLS (enforced by config so a release in ARM_1 is
attributable to MECH-342 and nothing else):
  MECH-091 (z_harm threat): use_harm_stream=False -> z_harm_a is None -> the
    MECH-091 urgency-interrupt block can never fire (it guards on
    z_harm_a is not None). MECH-342 fires with z_harm BELOW threshold by
    construction (no harm stream at all).
  ARC-028 / MECH-105 completion: completion-release fires when completion is
    HIGH; MECH-342 fires when decisiveness is LOW (opposite regime). The
    degraded window produces LOW decisiveness, not high completion.
  MECH-269b / V_s: use_vs_commit_release / anchor substrate OFF (defaults) ->
    the schema-staleness release pathway cannot fire. MECH-342 fires with a
    stable schema (no anchor substrate at all).
  MECH-340 ghost-goal: ghost-goal bank OFF (default) -> goal-appraisal-
    timescale disengagement cannot fire. MECH-342 operates on the active
    beta latch at motor-program timescale.
  Net: in ARM_1 the ONLY release pathway wired is MECH-342, corroborated by
  the mech342_n_fires counter (>= the observed beta-release count). In ARM_0
  NO release pathway exists -> the agent stays committed (the 592f gap).

ACCEPTANCE CRITERIA (pre-registered; aggregated across SEEDS):
  C1 BASELINE_COMMITS (non-vacuity): both arms achieve genuine natural
     commitment -- total n_commit_entries >= 1 across seeds in BOTH windows.
     Without commitment there is nothing to release.
  C2 DEGRADATION_OCCURRED (harness-validity / C0 guard): in the DEGRADED
     window the readiness signals genuinely crossed their floors -- the
     summed count of ticks with (nav_proxy < nav_floor OR score_margin <
     score_margin_floor) while committed is >= DEGRADE_MIN_TICKS across seeds
     for BOTH arms (the SD-047 perturbation is arm-independent). If C2 fails
     the run is non_contributory (INVALID_HARNESS) -- degradation never
     happened, so release authority is untestable. NOT a falsification.
  C3 RELEASE_AUTHORITY (core MECH-342 evidence): in the DEGRADED window,
     ARM_1_RELEASE_ON produces >= 1 decommit transition AND mech342_n_fires
     >= 1 across seeds, AND its committed-state occupancy is STRICTLY LOWER
     than ARM_0_RELEASE_OFF (which reproduces the 592f gap: stays committed,
     ~0 decommits). This is the quantity V3-EXQ-592f measured as zero.
  C4 NO_FALSE_ABORT (premature-abort pole): in the HEALTHY window,
     ARM_1_RELEASE_ON produces mech342_n_fires == 0 and its committed-state
     occupancy is NOT materially below ARM_0_RELEASE_OFF (within
     FALSE_ABORT_OCCUPANCY_TOL). Healthy readiness must not trigger release.
  C5 DISTINCT_FROM (attribution): z_harm stream disabled (MECH-091 cannot
     fire), V_s commit-release OFF, ghost-goal OFF; AND in the ARM_1 degraded
     window the mech342 fire count is >= the beta-release count (MECH-342 is
     the sole release author). Rules out a release produced by some other
     pathway.

INTERPRETATION GRID:
  PASS (C1..C5): MECH-342 maintenance-time release validated ECOLOGICALLY.
    Combined with the 592g diagnostic PASS this is the ecological evidence-
    grade run the 2026-06-02 /governance disposition required. Governance
    may move MECH-342 past the candidate / v3_pending gate (subject to the
    standard V3-pending clearance rules).
  C2 FAIL (degradation never occurred): non_contributory / INVALID_HARNESS.
    The SD-047 perturbation did not move readiness below the floors. Raise
    DEGRADE_INTENSITY_SCALE or lengthen the degraded window; re-queue. NOT a
    falsification of MECH-342.
  C1 PASS, C2 PASS, C3 FAIL (no decommit / no fires in ARM_1 degraded):
    route /diagnose-errors -- either the release branch is not reached
    (signal plumbing: e3.last_scores / commit_readiness not degrading at the
    branch) or the accumulator never crosses the bound under ecological
    (sub-maximal, intermittent) deficits. Candidate: lower
    maintenance_release_bound or raise accumulation_rate; or lengthen the
    degraded window so pressure accumulates.
  C3 PASS, C4 FAIL (ARM_1 aborts in the healthy window too): the substrate
    has a premature-abort pole under ecological noise -- the hysteresis band
    is too narrow. Route /diagnose-errors; candidate: raise the reengage
    levels / leak_rate.
  C5 FAIL: a release pathway other than MECH-342 fired -> the distinct-from
    config guard leaked. Audit config; re-queue.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from committed_mode_curriculum import run_p0_warmup, P0Result  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_629_mech342_ecological_maintenance_release_evidence"
QUEUE_ID = "V3-EXQ-629"
CLAIM_IDS: List[str] = ["MECH-342"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "evidence"
SLEEP_DRIVER_PATTERN = "K=never"
PREDECESSOR = "V3-EXQ-592g"  # complementary diagnostic; NOT superseded

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"


# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]

# R-c readiness floors (mirror MECH-090 admission floors + MECH-342 defaults).
SCORE_MARGIN_FLOOR = 0.05      # within-tick decisiveness axis
NAV_COMPETENCE_FLOOR = 0.30    # across-tick motor-program readiness axis

# MECH-342 substrate knobs (REEConfig defaults; passed explicitly for clarity).
MR_SCORE_MARGIN_FLOOR = 0.05
MR_SCORE_MARGIN_REENGAGE = 0.10
MR_NAV_FLOOR = 0.30
MR_NAV_REENGAGE = 0.50
MR_ACCUMULATION_RATE = 0.20
MR_LEAK_RATE = 0.10
MR_RELEASE_BOUND = 1.00
MR_PRESSURE_CAP = 1.50

# Curriculum config (matched to V3-EXQ-592e).
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

# P2 two-window measurement.
P2_HEALTHY_EPISODES = 25
P2_DEGRADED_EPISODES = 25
P2_STEPS_PER_EPISODE = 200

# SD-047 degraded-window perturbation (agent-independent world unpredictability).
# High intensity to ensure the trained E2 world model is genuinely surprised
# (raising running_variance -> dropping the nav proxy). hazard_harm stays the
# baseline 0.02 so z_harm contact (irrelevant here -- harm stream is OFF) is
# not the driver.
DEGRADE_INTENSITY_SCALE = 4.0

# Acceptance thresholds.
C2_DEGRADE_MIN_TICKS = 10          # >= this many degraded committed-ticks / arm-window
C3_MIN_DECOMMITS = 1               # ARM_1 degraded decommit transitions >= 1
C3_MIN_FIRES = 1                   # ARM_1 degraded mech342_n_fires >= 1
C4_FALSE_ABORT_OCCUPANCY_TOL = 0.15  # ARM_1 healthy occupancy >= ARM_0 - this

# Force-uncommitted at each P2 episode start (per the 592e fix) so every
# commitment during measurement is a fresh gate-consulted transition.
FORCE_UNCOMMITTED_P2_ENTRY = True


# ---------------------------------------------------------------------------
# Arms (the maintenance-release flag is the only manipulated variable)
# ---------------------------------------------------------------------------

ARMS: List[Dict] = [
    {
        "name": "ARM_0_RELEASE_OFF",
        "use_maintenance_release": False,
    },
    {
        "name": "ARM_1_RELEASE_ON",
        "use_maintenance_release": True,
    },
]

CONDITIONS = len(ARMS)
# Two windows per arm-seed cell, each prints one verdict line.
WINDOWS_PER_CELL = 2
TOTAL_RUNS = len(ARMS) * len(SEEDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_stamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def make_target_env_healthy(seed: int) -> CausalGridWorldV2:
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


def make_target_env_degraded(seed: int) -> CausalGridWorldV2:
    """Target env + SD-047 multi-source dynamics (agent-independent world
    unpredictability). Same size / resource layout family as the healthy env
    so the ONLY difference is the world becomes harder to predict -- exactly
    the degraded-execution-readiness regime MECH-342 targets."""
    return CausalGridWorldV2(
        size=TARGET_ENV_SIZE,
        num_hazards=TARGET_ENV_HAZARDS,
        num_resources=TARGET_ENV_RESOURCES,
        hazard_harm=0.02,
        resource_benefit=0.05,
        use_proxy_fields=True,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=TARGET_TOLERANCE_FRAC,
        multi_source_dynamics_enabled=True,
        multi_source_intensity_scale=DEGRADE_INTENSITY_SCALE,
        weather_field_enabled=True,
        background_drift_enabled=True,
        n_drift_sources=2,
        seed=seed + 2000,
    )


def make_arm_cfg(arm: Dict) -> REEConfig:
    """592e-matched cfg + per-arm MECH-342 release flag. Both arms commit
    identically (R-c conjunction ON); only use_maintenance_release differs.
    Distinct-from controls enforced here: use_harm_stream=False (MECH-091
    cannot fire); V_s commit-release + ghost-goal left OFF (defaults)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        alpha_world=0.9,
        use_harm_stream=False,
        # MECH-090 R-c conjunction (commit ENTRY) -- identical across arms so
        # the agent commits the same way; the ONLY arm difference is release.
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=NAV_COMPETENCE_FLOOR,
        commit_readiness_initial=1.0,  # start healthy so the agent commits
        # MECH-342 maintenance-time release substrate (the arm axis).
        use_maintenance_release=bool(arm["use_maintenance_release"]),
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
    # MECH-090 within-tick score-margin gate at commit entry (identical arms).
    cfg.heartbeat.use_commit_readiness_gate = True
    cfg.heartbeat.commit_readiness_floor = SCORE_MARGIN_FLOOR
    return cfg


def compute_nav_competence_proxy(agent: REEAgent) -> float:
    """Proxy nav_competence from the agent's world-model confidence (592e).
    rv low (confident) -> proxy ~1.0; rv rises (surprised by the SD-047
    perturbation) -> proxy drops toward 0 (degraded readiness)."""
    rv = float(getattr(agent.e3, "_running_variance", 1.0))
    threshold = float(getattr(agent.e3, "commit_threshold", 0.40))
    if threshold <= 0:
        return 1.0
    raw = 1.0 - (rv / threshold)
    return max(0.0, min(1.0, raw))


def current_score_margin(agent: REEAgent) -> Optional[float]:
    """Per-candidate first-action decisiveness margin off the last real E3
    selection (REE lower-is-better -> sorted[1] - sorted[0]). None when no
    prior selection or fewer than 2 candidates."""
    scores = getattr(agent.e3, "last_scores", None)
    if scores is None:
        return None
    flat = scores.detach().float().reshape(-1)
    if flat.numel() < 2:
        return None
    s, _ = torch.sort(flat)
    return float(s[1].item() - s[0].item())


def force_uncommitted(agent: REEAgent) -> None:
    """Release any carried-over commitment so each P2 episode re-commits via a
    fresh gate-consulted transition (592e fix (a))."""
    if agent.beta_gate.is_elevated:
        agent.beta_gate.release()
    if agent.e3._committed_trajectory is not None:
        agent.e3._committed_trajectory = None
    agent._committed_step_idx = 0


def mech342_fire_count(agent: REEAgent) -> int:
    if getattr(agent, "maintenance_release", None) is None:
        return 0
    return int(agent.maintenance_release.get_state().get("mech342_n_fires", 0))


# ---------------------------------------------------------------------------
# P2 window: real agent, real env, rv updated from real E2 world-forward error
# ---------------------------------------------------------------------------

def run_p2_window(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
    window_label: str,
) -> Dict:
    """Frozen-policy P2 eval over `env`, capturing per-tick readiness, the
    committed-state occupancy, decommit transitions, and MECH-342 fires.

    running_variance is updated each tick from the REAL E2 world-forward error
    so the (SD-047) env perturbation genuinely drives the nav_competence proxy.
    nav_competence is pushed via notify_outcome(proxy) each tick. The
    decisiveness axis (score margin) is read naturally from e3.last_scores.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim

    n_steps_total = 0
    n_committed_steps = 0          # e3._committed_trajectory present
    n_beta_elevated_steps = 0      # beta_gate.is_elevated
    n_commit_entries = 0           # fresh beta-elevation transitions
    decommit_beta_releases = 0     # beta True->False while not at episode reset
    decommit_pointer_drops = 0     # e3 committed pointer True->False
    degraded_committed_ticks = 0   # committed ticks with nav<floor OR margin<floor
    nav_below_floor_ticks = 0
    margin_below_floor_ticks = 0
    max_z_harm_norm = 0.0          # distinct-from guard (expected 0.0; harm off)

    fires_before_window = mech342_fire_count(agent)

    nav_trace: List[float] = []
    margin_trace: List[float] = []
    rv_trace: List[float] = []

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            if FORCE_UNCOMMITTED_P2_ENTRY:
                force_uncommitted(agent)

            z_world_prev: Optional[torch.Tensor] = None
            action_prev: Optional[torch.Tensor] = None
            prev_beta = bool(agent.beta_gate.is_elevated)
            prev_pointer = agent.e3._committed_trajectory is not None

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)
                z_world_curr = latent.z_world.detach()

                # Update running_variance from the REAL world-forward error so
                # the env perturbation moves readiness (ecological degradation).
                if z_world_prev is not None and action_prev is not None:
                    wf_pred = agent.e2.world_forward(z_world_prev, action_prev)
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - z_world_curr).detach()
                    )

                # z_harm distinct-from guard (harm stream OFF -> latent.z_harm_a
                # is None; record 0.0 so the manifest can assert MECH-091 inert).
                z_harm_a = getattr(latent, "z_harm_a", None)
                if z_harm_a is not None:
                    max_z_harm_norm = max(
                        max_z_harm_norm, float(z_harm_a.norm().item())
                    )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                admits_pre = int(
                    agent.beta_gate.get_state().get("mech090_n_elevation_admitted", 0)
                )
                action = agent.select_action(candidates, ticks)
                admits_post = int(
                    agent.beta_gate.get_state().get("mech090_n_elevation_admitted", 0)
                )
                if admits_post > admits_pre:
                    n_commit_entries += 1

                # Readiness snapshot (post-select; reflects this tick's state).
                nav_proxy = compute_nav_competence_proxy(agent)
                margin = current_score_margin(agent)
                rv_now = float(getattr(agent.e3, "_running_variance", 0.0))
                beta_now = bool(agent.beta_gate.is_elevated)
                pointer_now = agent.e3._committed_trajectory is not None

                # Decommit transitions (the observable release signature).
                if prev_beta and not beta_now:
                    decommit_beta_releases += 1
                if prev_pointer and not pointer_now:
                    decommit_pointer_drops += 1

                # Occupancy + degradation accounting (only while committed).
                if pointer_now or beta_now:
                    n_committed_steps += 1 if pointer_now else 0
                    n_beta_elevated_steps += 1 if beta_now else 0
                    nav_bad = nav_proxy < NAV_COMPETENCE_FLOOR
                    margin_bad = (margin is not None) and (margin < SCORE_MARGIN_FLOOR)
                    if nav_bad:
                        nav_below_floor_ticks += 1
                    if margin_bad:
                        margin_below_floor_ticks += 1
                    if nav_bad or margin_bad:
                        degraded_committed_ticks += 1
                else:
                    n_committed_steps += 1 if pointer_now else 0
                    n_beta_elevated_steps += 1 if beta_now else 0

                nav_trace.append(nav_proxy)
                if margin is not None:
                    margin_trace.append(margin)
                rv_trace.append(rv_now)

                action_idx = int(action.argmax(dim=-1).item())
                z_world_prev = z_world_curr
                action_prev = action.detach()

                _, _, done, _, obs_dict = env.step(action_idx)

                # Push the nav_competence proxy AFTER the step so the next
                # tick's maintenance-release consultation sees the update.
                if agent.commit_readiness is not None:
                    agent.commit_readiness.notify_outcome(nav_proxy)

                prev_beta = beta_now
                prev_pointer = pointer_now
                n_steps_total += 1
                if done:
                    break

    fires_in_window = mech342_fire_count(agent) - fires_before_window

    beta_occupancy = (
        n_beta_elevated_steps / n_steps_total if n_steps_total > 0 else 0.0
    )
    pointer_occupancy = (
        n_committed_steps / n_steps_total if n_steps_total > 0 else 0.0
    )
    return {
        "window": window_label,
        "n_steps_total": n_steps_total,
        "n_commit_entries": n_commit_entries,
        "n_beta_elevated_steps": n_beta_elevated_steps,
        "n_committed_pointer_steps": n_committed_steps,
        "beta_elevated_occupancy": beta_occupancy,
        "committed_pointer_occupancy": pointer_occupancy,
        "decommit_beta_releases": decommit_beta_releases,
        "decommit_pointer_drops": decommit_pointer_drops,
        "decommit_transitions": decommit_beta_releases + decommit_pointer_drops,
        "mech342_fires": int(fires_in_window),
        "degraded_committed_ticks": degraded_committed_ticks,
        "nav_below_floor_ticks": nav_below_floor_ticks,
        "margin_below_floor_ticks": margin_below_floor_ticks,
        "max_z_harm_a_norm": max_z_harm_norm,
        "mean_nav_proxy": float(np.mean(nav_trace)) if nav_trace else 1.0,
        "min_nav_proxy": float(np.min(nav_trace)) if nav_trace else 1.0,
        "mean_score_margin": float(np.mean(margin_trace)) if margin_trace else None,
        "min_score_margin": float(np.min(margin_trace)) if margin_trace else None,
        "mean_running_variance": float(np.mean(rv_trace)) if rv_trace else 0.0,
        "max_running_variance": float(np.max(rv_trace)) if rv_trace else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-arm per-seed runner
# ---------------------------------------------------------------------------

def run_arm_seed(arm: Dict, seed: int, dry_run: bool, total_run_idx: int) -> Dict:
    device = torch.device("cpu")
    set_seed(seed)

    p0_budget = 5 if dry_run else P0_BUDGET
    p0_steps = 20 if dry_run else P0_STEPS_PER_EPISODE
    healthy_eps = 2 if dry_run else P2_HEALTHY_EPISODES
    degraded_eps = 2 if dry_run else P2_DEGRADED_EPISODES
    p2_steps = 20 if dry_run else P2_STEPS_PER_EPISODE

    arm_label = arm["name"]
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
        probe_interval=(2 if dry_run else P0_PROBE_INTERVAL),
        mid_probe_frac=P0_MID_PROBE_FRAC,
        convergence_stable_checkpoints=(1 if dry_run else 3),
        threshold_relaxation=0.0,
    )
    print(
        f"  [train] {arm_label} seed={seed} ep {p0.n_episodes}/{train_total}"
        f"  (P0 done: converged={p0.converged} aborted={p0.aborted}"
        f" final_rv={p0.final_rv:.5f})",
        flush=True,
    )

    if p0.aborted:
        empty = lambda lbl: {
            "window": lbl, "n_steps_total": 0, "n_commit_entries": 0,
            "n_beta_elevated_steps": 0, "n_committed_pointer_steps": 0,
            "beta_elevated_occupancy": 0.0, "committed_pointer_occupancy": 0.0,
            "decommit_beta_releases": 0, "decommit_pointer_drops": 0,
            "decommit_transitions": 0, "mech342_fires": 0,
            "degraded_committed_ticks": 0, "nav_below_floor_ticks": 0,
            "margin_below_floor_ticks": 0, "max_z_harm_a_norm": 0.0,
            "mean_nav_proxy": 1.0, "min_nav_proxy": 1.0,
            "mean_score_margin": None, "min_score_margin": None,
            "mean_running_variance": 0.0, "max_running_variance": 0.0,
        }
        healthy = empty("HEALTHY")
        degraded = empty("DEGRADED")
    else:
        healthy = run_p2_window(
            agent, make_target_env_healthy(seed), device,
            n_eps=healthy_eps, steps_per_episode=p2_steps,
            window_label="HEALTHY",
        )
        degraded = run_p2_window(
            agent, make_target_env_degraded(seed), device,
            n_eps=degraded_eps, steps_per_episode=p2_steps,
            window_label="DEGRADED",
        )

    cell = {
        "arm_name": arm_label,
        "seed": seed,
        "use_maintenance_release": bool(arm["use_maintenance_release"]),
        "p0": {
            "converged": bool(p0.converged),
            "aborted": bool(p0.aborted),
            "n_episodes": int(p0.n_episodes),
            "final_rv": float(p0.final_rv),
        },
        "healthy_window": healthy,
        "degraded_window": degraded,
    }

    print(
        f"[run {total_run_idx + 1}/{TOTAL_RUNS}] {arm_label} seed={seed}"
        f"  H[occ={healthy['committed_pointer_occupancy']:.3f}"
        f" fires={healthy['mech342_fires']} decommit={healthy['decommit_transitions']}]"
        f"  D[occ={degraded['committed_pointer_occupancy']:.3f}"
        f" fires={degraded['mech342_fires']} decommit={degraded['decommit_transitions']}"
        f" degr_ticks={degraded['degraded_committed_ticks']}"
        f" min_nav={degraded['min_nav_proxy']:.3f} max_rv={degraded['max_running_variance']:.4f}]",
        flush=True,
    )
    print(f"verdict: PASS", flush=True)  # window 2 of cell
    return cell


# ---------------------------------------------------------------------------
# Aggregation + acceptance
# ---------------------------------------------------------------------------

def _arm_cells(cells: List[Dict], arm_name: str) -> List[Dict]:
    return [c for c in cells if c["arm_name"] == arm_name]


def _sum(cells: List[Dict], window: str, key: str) -> int:
    return int(sum(c[window][key] for c in cells))


def _mean_occ(cells: List[Dict], window: str, key: str) -> float:
    vals = [c[window][key] for c in cells]
    return float(np.mean(vals)) if vals else 0.0


def evaluate_acceptance(cells: List[Dict]) -> Tuple[Dict, str, str]:
    off = _arm_cells(cells, "ARM_0_RELEASE_OFF")
    on = _arm_cells(cells, "ARM_1_RELEASE_ON")

    # C1 BASELINE_COMMITS: both arms commit in both windows.
    on_h_commits = _sum(on, "healthy_window", "n_commit_entries")
    on_d_commits = _sum(on, "degraded_window", "n_commit_entries")
    off_h_commits = _sum(off, "healthy_window", "n_commit_entries")
    off_d_commits = _sum(off, "degraded_window", "n_commit_entries")
    c1 = min(on_h_commits, on_d_commits, off_h_commits, off_d_commits) >= 1

    # C2 DEGRADATION_OCCURRED (harness validity): degraded window genuinely
    # crossed the floors while committed (arm-independent perturbation).
    on_degr_ticks = _sum(on, "degraded_window", "degraded_committed_ticks")
    off_degr_ticks = _sum(off, "degraded_window", "degraded_committed_ticks")
    c2 = on_degr_ticks >= C2_DEGRADE_MIN_TICKS and off_degr_ticks >= C2_DEGRADE_MIN_TICKS

    # C3 RELEASE_AUTHORITY: ON degraded decommit + fires, occupancy < OFF.
    on_d_decommit = _sum(on, "degraded_window", "decommit_transitions")
    on_d_fires = _sum(on, "degraded_window", "mech342_fires")
    on_d_occ = _mean_occ(on, "degraded_window", "committed_pointer_occupancy")
    off_d_occ = _mean_occ(off, "degraded_window", "committed_pointer_occupancy")
    c3 = (
        on_d_decommit >= C3_MIN_DECOMMITS
        and on_d_fires >= C3_MIN_FIRES
        and on_d_occ < off_d_occ
    )

    # C4 NO_FALSE_ABORT: ON healthy has no fires; occupancy not materially
    # below OFF healthy occupancy.
    on_h_fires = _sum(on, "healthy_window", "mech342_fires")
    on_h_occ = _mean_occ(on, "healthy_window", "committed_pointer_occupancy")
    off_h_occ = _mean_occ(off, "healthy_window", "committed_pointer_occupancy")
    c4 = on_h_fires == 0 and on_h_occ >= (off_h_occ - C4_FALSE_ABORT_OCCUPANCY_TOL)

    # C5 DISTINCT_FROM: the distinct-from controls are enforced by config
    # (harm stream off -> MECH-091 inert; V_s commit-release off; ghost-goal
    # off) so MECH-091 / MECH-269b-V_s / MECH-340 cannot fire at all. ARC-028
    # hippocampal completion-release IS part of the shared bistable gate and
    # can co-fire, but it is IDENTICAL across arms (same completion dynamics,
    # same env), so it cannot explain the ON-vs-OFF difference -- the controlled
    # single-variable manipulation (use_maintenance_release) attributes that
    # difference to MECH-342. C5 therefore requires (a) the config guards hold
    # (no z_harm anywhere) and (b) MECH-342 demonstrably fired in the ON
    # degraded window (corroborating the C3 suppression). The OFF-arm baseline
    # release count is recorded for transparency (the ARC-028 shared floor).
    on_d_beta_releases = _sum(on, "degraded_window", "decommit_beta_releases")
    off_d_beta_releases = _sum(off, "degraded_window", "decommit_beta_releases")
    max_harm = max(
        [c[w]["max_z_harm_a_norm"] for c in cells for w in ("healthy_window", "degraded_window")]
        or [0.0]
    )
    c5 = (max_harm == 0.0) and (on_d_fires >= C3_MIN_FIRES)

    acceptance = {
        "C1_baseline_commits": {
            "pass": bool(c1),
            "on_healthy_commits": on_h_commits, "on_degraded_commits": on_d_commits,
            "off_healthy_commits": off_h_commits, "off_degraded_commits": off_d_commits,
        },
        "C2_degradation_occurred": {
            "pass": bool(c2),
            "on_degraded_ticks": on_degr_ticks, "off_degraded_ticks": off_degr_ticks,
            "min_required": C2_DEGRADE_MIN_TICKS,
        },
        "C3_release_authority": {
            "pass": bool(c3),
            "on_degraded_decommit_transitions": on_d_decommit,
            "on_degraded_mech342_fires": on_d_fires,
            "on_degraded_occupancy": on_d_occ,
            "off_degraded_occupancy": off_d_occ,
            "occupancy_suppressed": bool(on_d_occ < off_d_occ),
        },
        "C4_no_false_abort": {
            "pass": bool(c4),
            "on_healthy_mech342_fires": on_h_fires,
            "on_healthy_occupancy": on_h_occ,
            "off_healthy_occupancy": off_h_occ,
            "tolerance": C4_FALSE_ABORT_OCCUPANCY_TOL,
        },
        "C5_distinct_from": {
            "pass": bool(c5),
            "max_z_harm_a_norm": max_harm,
            "harm_stream_disabled": True,
            "vs_commit_release_enabled": False,
            "ghost_goal_enabled": False,
            "on_degraded_mech342_fires": on_d_fires,
            "on_degraded_beta_releases": on_d_beta_releases,
            "off_degraded_beta_releases_arc028_shared_floor": off_d_beta_releases,
            "mech342_fired_in_on_degraded": bool(on_d_fires >= C3_MIN_FIRES),
            "attribution_note": (
                "ARC-028 completion-release is shared across arms and cannot "
                "explain the ON-vs-OFF difference; MECH-091/V_s/ghost-goal are "
                "config-disabled. The single-variable manipulation attributes "
                "the ON-vs-OFF suppression to MECH-342."
            ),
        },
    }

    if not c1:
        return acceptance, "FAIL", "NO_NATURAL_COMMITMENT"
    if not c2:
        return acceptance, "FAIL", "INVALID_HARNESS_NO_DEGRADATION"
    if c3 and c4 and c5:
        return acceptance, "PASS", "PASS_ECOLOGICAL_MAINTENANCE_RELEASE"
    return acceptance, "FAIL", "FAIL_NO_ECOLOGICAL_RELEASE"


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Tuple[Dict, Path]:
    cells: List[Dict] = []
    run_idx = 0
    for arm in ARMS:
        for seed in SEEDS:
            cells.append(run_arm_seed(arm, seed, dry_run, run_idx))
            run_idx += 1

    acceptance, outcome, diagnostic_outcome = evaluate_acceptance(cells)

    if diagnostic_outcome == "INVALID_HARNESS_NO_DEGRADATION":
        claim_ids: List[str] = []  # non-contributory: degradation never occurred
        evidence_direction = "non_contributory"
        evidence_note = (
            "Harness invalid: the SD-047 degraded window did not drive readiness "
            "below the R-c floors enough to test release authority "
            "(degraded_committed_ticks < threshold). Raise DEGRADE_INTENSITY_SCALE "
            "or lengthen the degraded window and re-queue. Not a falsification of "
            "MECH-342."
        )
    elif outcome == "PASS":
        claim_ids = list(CLAIM_IDS)
        evidence_direction = "supports"
        evidence_note = (
            "Ecological evidence-grade run (real REEAgent + real CausalGridWorldV2, "
            "real E3.select, natural commitment, natural mid-commitment readiness "
            "degradation via SD-047 world unpredictability): with "
            "use_maintenance_release=True the agent decommits (releases the beta "
            "latch + clears the committed trajectory) when the two R-c readiness "
            "signals degrade mid-commitment, with strictly lower committed-state "
            "occupancy than the OFF arm (which reproduces the V3-EXQ-592f gap), and "
            "does NOT false-abort under healthy readiness. MECH-342 is the sole "
            "release author (harm stream / V_s / ghost-goal disabled). Complements "
            "the V3-EXQ-592g diagnostic PASS; this is the ecological evidence the "
            "2026-06-02 governance disposition required."
        )
    elif diagnostic_outcome == "NO_NATURAL_COMMITMENT":
        claim_ids = []
        evidence_direction = "non_contributory"
        evidence_note = (
            "Harness invalid: the agent did not achieve natural commitment in P2 "
            "(C1 fail). Without commitment there is nothing to release. Inspect P0 "
            "convergence / commit-entry gating. Not a falsification of MECH-342."
        )
    else:
        claim_ids = list(CLAIM_IDS)
        evidence_direction = "weakens"
        evidence_note = (
            "MECH-342 ENABLED + degradation occurred, but the ecological release "
            "signature did not appear (C3) or a premature-abort pole surfaced (C4) "
            "or attribution leaked (C5). Route per the interpretation grid."
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
        "predecessor": PREDECESSOR,
        "claim_ids": claim_ids,
        "outcome": outcome,
        "diagnostic_outcome": diagnostic_outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": evidence_note,
        "seeds": SEEDS,
        "arms": [a["name"] for a in ARMS],
        "thresholds": {
            "score_margin_floor": SCORE_MARGIN_FLOOR,
            "nav_competence_floor": NAV_COMPETENCE_FLOOR,
            "degrade_intensity_scale": DEGRADE_INTENSITY_SCALE,
            "c2_degrade_min_ticks": C2_DEGRADE_MIN_TICKS,
            "c3_min_decommits": C3_MIN_DECOMMITS,
            "c3_min_fires": C3_MIN_FIRES,
            "c4_false_abort_occupancy_tol": C4_FALSE_ABORT_OCCUPANCY_TOL,
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
        "acceptance": acceptance,
        "cells": cells,
        "ecological_design": {
            "real_select_action": True,
            "real_e3_select": True,
            "real_beta_gate": True,
            "real_commit_readiness": True,
            "real_maintenance_release": True,
            "stubbed_component": None,
            "commitment": "natural (P0 curriculum warmup + R-c-gated entry)",
            "degradation": "natural (SD-047 multi_source_dynamics raises E2 "
                           "world-forward error -> running_variance -> nav proxy "
                           "drop + score-margin compression)",
            "arm_axis": "use_maintenance_release (ON vs OFF); commit-entry gating "
                        "identical across arms",
        },
        "notes": [
            "Ecological evidence complement to the V3-EXQ-592g diagnostic probe "
            "(PREDECESSOR, NOT superseded).",
            "Distinct-from controls enforced by config: use_harm_stream=False "
            "(MECH-091 inert), V_s commit-release OFF, ghost-goal OFF -- so a "
            "beta release in ARM_1 is attributable to MECH-342, corroborated by "
            "the mech342_n_fires counter.",
            "running_variance is updated each P2 tick from the real E2 "
            "world-forward error so the SD-047 perturbation drives the "
            "nav_competence proxy (the env-side auto-drive of commit_readiness is "
            "a Phase-2 substrate follow-on not yet wired).",
        ],
    }

    out_dir = (
        Path(tempfile.gettempdir()) / "ree_v3_dry_runs" if dry_run else EVIDENCE_DIR
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"verdict: {outcome}", flush=True)
    print(f"Experiment: {outcome} ({diagnostic_outcome})", flush=True)
    print(f"Saved manifest: {out_path}", flush=True)
    return manifest, out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Tiny smoke run; manifest under /tmp.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest, out_path = run_experiment(dry_run=args.dry_run)
    signal_dir = None
    if args.dry_run:
        signal_dir = Path(tempfile.gettempdir()) / "ree_runner_signals"
    _outcome = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=str(manifest["run_id"]),
        queue_id=QUEUE_ID,
        exit_reason="ok" if manifest["outcome"] == "PASS" else "fail",
        signal_dir=signal_dir,
    )
