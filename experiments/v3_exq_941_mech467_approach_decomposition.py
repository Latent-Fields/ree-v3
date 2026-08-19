"""V3-EXQ-941: MECH-467 leg H-denominator -- decompose "0 consumption events" into
NEVER-APPROACHED vs APPROACHED-AND-FAILED-TO-ARRIVE.

EXPERIMENT_PURPOSE: see EXPERIMENT_PURPOSE constant below.

GOV-FANOUT-1 PORTFOLIO MEMBER (leg 2 of 2). Sibling: V3-EXQ-940 (H-energy).
Authority: failure_autopsy_V3-EXQ-874b_2026-08-17 (confirmed; disposition applied to
MECH-467 by the /governance cycle of 2026-08-18). Design + the 4-leg -> 2-leg scope
shrink: REE_assembly/evidence/planning/mech467_govfanout_portfolio_staged_2026-08-19.md

THIS IS NOT A LETTERED RE-POSE OF 874b. The autopsy explicitly refuses one. It is a
DIAGNOSTIC, excluded from governance confidence scoring by construction.

WHY THIS LEG EXISTS
-------------------
V3-EXQ-874b recorded consumption events and NOTHING UPSTREAM OF THEM, so its 1/3/4/0
pooled events were uninterpretable: a zero could mean the agent never headed for a
target, or headed for one and never arrived, and the manifest could not tell those
apart. Every readout in 874b was DOWNSTREAM of a consumption event, which is precisely
why a zero denominator destroyed the whole run.

This leg builds the missing upstream instrument. Its deliverable is the DECOMPOSITION
itself, and -- the key design property -- every counter here is measurable on a window
in which NOTHING IS EVER EATEN. That is what routes this leg to the `measurement` axis
rather than to another attempt at producing events.

WHAT THE PRIOR EVIDENCE ALREADY SETTLES, AND WHAT IT DOES NOT
--------------------------------------------------------------
The 2026-08-18 navigation-immobility scoping spike
(REE_assembly/evidence/planning/navigation_immobility_scoping_2026-08-18.md) established
at SUBSTRATE level that the near-immobility is the plain-navigation instance of MECH-439
(F-dominance conversion ceiling, ceiling_decision: exhausted, awaiting ARC-107),
amplified by the E3 heartbeat hold-and-repeat cadence (e3_steps_per_tick, default 10,
MECH-093-modulated 5-20), so ~85-90% of env ticks are not re-selection at all. It also
ruled OUT a degenerate-proposer bug and identified wall-blocking as a distinct additive
contributor (causal_grid_world.py: a move into a wall is silently absorbed as a no-op).

That is why this portfolio does NOT queue the autopsy's `H-commitment` and `H-cadence`
legs: their hypotheses are answered at substrate level, and the deeper causal question
is owned by an ACTIVE hypothesis-space line (qid e3_fdominance_causal_discrimination,
H0-H5, with a discovery event as recent as 2026-08-18). Re-posing them as MECH-467-
specific probes would duplicate an owned, in-flight investigation.

What the spike does NOT supply is the BATTERY-LEVEL decomposition under 874b's own
config (6x6, 12 resources, pinned operating modes, per-cell clones). The approach-run
and cadence COUNTERS those two legs would have produced are therefore instrumented HERE,
as measurement, not adjudicated as hypotheses. Recording the number is not duplicating
the investigation.

ARMS -- the claim's two timing regimes (2 arms x 3 seeds = 6 cells)
-------------------------------------------------------------------
  ARM_PRECOMMIT  operating_mode pinned internal_planning (active rule-update regime)
  ARM_REPLAY     operating_mode pinned internal_replay   (protected regime)
The during-commitment arm is EXCLUDED per MECH-467's SEQUENCING CAUTION and
[memory] feedback_dont_queue_commitment_dependent_behavioural. The RULE-SET COMPLEXITY
axis is NOT varied -- see defect (2) below.

THE COUNTERS -- this leg's actual deliverable
----------------------------------------------
  n_move_actions            action was a movement action
  n_position_changes        the agent's cell actually changed
    -> their difference is the WALL-BLOCKING / no-op rate, the additive contributor the
       scoping spike flags as distinct from selection collapse
  n_approach_initiations    a step that STRICTLY REDUCED Chebyshev distance to the
                            nearest benefit-bearing (goal-tag) resource
  approach_run_lengths      consecutive strictly-decreasing steps toward the SAME target
                            cell -- the H-commitment measurement, retained as
                            instrumentation per the scope note above
  n_arrivals                distance to the tracked target reached 0
  min_distance_to_goal      closest the agent ever got
  n_e3_ticks / n_latched_ticks   the cadence denominator (H-cadence's measurement)
  done_cause / window_completeness / n_benefit_bearing_resource_cells

DECOMPOSITION DVs
  approach_initiation_rate = n_approach_initiations / n_realised_ticks
  approach_completion_rate = n_arrivals / max(n_approach_initiations, 1)

DISCRIMINATION GRID
  initiation ~ 0                      -> NEVER-APPROACHED. The agent does not head for
                                         targets at all; confirms MECH-439 at battery
                                         level.
  initiation > 0, completion ~ 0      -> APPROACHED-AND-FAILED-TO-ARRIVE. The
                                         move-vs-position-change gap then separates
                                         wall-blocking from commitment abandonment, and
                                         approach_run_lengths says how far runs get.
  both healthy                        -> DECLARED NULL: the decomposition is
                                         uninformative because approaches are initiated
                                         and complete at the observed rate, and the
                                         denominator problem lies outside anything this
                                         decomposition can see.

THE THREE 874b DEFECTS, AND HOW THIS DRIVER AVOIDS EACH
--------------------------------------------------------
(1) UNRECORDED TRUNCATION. Every cell records done_cause (the env supplies it in info;
    874b never read it), a truncated flag, n_realised_ticks, n_budgeted_ticks and
    window_completeness, and the run feeds the shared SD-094
    EpisodeTerminationAccumulator. EVERY RATE IS NORMALISED BY REALISED TICKS.
(2) RULE-SET COMPLEXITY CONFOUNDED WITH NUTRITIVE DENSITY. 874b held num_resources at 12
    while only type 0 carries benefit, so SIMPLE had 6/12 benefit-bearing cells and
    COMPLEX 4/12 -- adding distractor TYPES silently removed a third of the food. Here
    the ruleset axis is NOT VARIED (SIMPLE only), so the confound cannot arise, and
    n_benefit_bearing_resource_cells is measured off the env's own type grid and gated
    by a readiness precondition.
(3) FALSE z_goal writer_defect. 874b called zg_acc.observe(agent) on the P0 BASE agent
    while every cell stepped a CLONE. Here observe() is called on the STEPPING CLONE.

MECH-262 CONSTRAINT (from the same autopsy). Storage-site rule drift and selection-path
operative-rule fidelity DISSOCIATE in 6 of 9 live cells, so a storage-site read does not
track the operative rule. This driver reads rule state at NEITHER site -- it takes no
rule measurement at all. MECH-262 is not tagged.

READINESS, SAME-STATISTIC (the V3-EXQ-643 rule). The load-bearing criterion routes on
COUNTS OF POSITION CHANGE, so the readiness check asserts THAT SAME STATISTIC on a
positive control: in setup, forced movement actions from a known-free cell must register
position changes. A below-floor reading means the movement instrument cannot see motion,
and self-routes substrate_not_ready_requeue -- NEVER a substrate verdict. (Measured
2026-08-19 on this env: 4 of 4 distinct movement actions changed position.)

DV-SYMMETRY DECLARATION (mandatory, per arm -- the V3-EXQ-604c rule). The DV is a ratio
of counts of distance-decreasing steps; its symmetry group is permutation of ticks (a
set-aggregate). The manipulation is the operating-mode pin, which changes the SELECTION
POLICY and hence which actions are taken and which distances occur. It is NOT a
broadcast scalar added uniformly across candidates (it does not shift every candidate's
score by a constant), NOT a monotone rescaling (it does not preserve candidate order by
construction), and NOT a permutation of interchangeable units. It is therefore NOT
INVARIANT under the DV's symmetry group in either ARM_PRECOMMIT or ARM_REPLAY.

PSEUDO-REPLICATION. This driver reads NO agent.e3.last_* latched diagnostic. Ticks on
which the E3 cadence held the previous action are counted in n_latched_ticks and
contribute nothing to any per-selection quantity.

CONTAINMENT. Diagnostic over EXISTING substrate. No new module, no substrate change;
every counter is derived from the env's own published position / type grid / info.

Run:
  /opt/local/bin/python3 experiments/v3_exq_941_mech467_approach_decomposition.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_941_mech467_approach_decomposition.py --dry-run
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
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
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.episode_termination import EpisodeTerminationAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_941_mech467_approach_decomposition"
QUEUE_ID = "V3-EXQ-941"
CLAIM_IDS = ["MECH-467"]
EXPERIMENT_PURPOSE = "diagnostic"
SEEDS = [42, 43, 44]

GOAL_TAG = 1  # SD-049 tag = type_idx + 1; type_idx 0 is the benefit-bearing goal

TIMINGS = {"PRECOMMIT": "internal_planning", "REPLAY": "internal_replay"}
ARMS = tuple(f"ARM_{t}" for t in TIMINGS)

# ---- pre-registered constants (NOT derived from the run's own statistics) ----
# Geometry is 874b's VERBATIM: this leg decomposes 874b's own zero, so changing the
# geometry would change the question.
GRID_SIZE = 6
NUM_RESOURCES = 12
MAX_EPISODE_STEPS = 1500
N_WARMUP_STEPS = 80
N_EVAL_STEPS = 900
P0_BUDGET = 60
P0_STEPS_PER_EPISODE = 80

# Movement actions in CausalGridWorldV2 are the 4 cardinal moves (action_dim=4).
MOVE_ACTIONS = (0, 1, 2, 3)

# Criterion thresholds, pre-registered.
APPROACH_INITIATION_FLOOR = 0.02   # below this: the agent effectively never approaches
APPROACH_COMPLETION_FLOOR = 0.05   # below this: approaches are initiated but never land
# Readiness floors.
MIN_REALISED_TICKS = 20
MIN_POSITION_CHANGES_CONTROL = 1   # positive control: forced moves must move the agent
MIN_DISTANCE_SAMPLES = 20

# Smoke budgets
SMOKE_P0_BUDGET = 3
SMOKE_P0_STEPS = 15
SMOKE_WARMUP = 5
SMOKE_EVAL = 40
SMOKE_MIN_REALISED_TICKS = 5
SMOKE_MIN_DISTANCE_SAMPLES = 5


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ------------------------------------------------------------------ env / agent
def _env_kwargs(distractors_present: bool) -> dict:
    """874b's SIMPLE-ruleset env kwargs, verbatim. The ruleset axis is NOT varied."""
    dist = (1.0, 1.0) if distractors_present else (1.0, 0.0)
    return {
        "size": GRID_SIZE,
        "num_hazards": 0,
        "num_resources": NUM_RESOURCES,
        "num_waypoints": 2,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "resource_respawn_on_consume": True,
        "multi_resource_heterogeneity_enabled": True,
        "per_axis_drive_enabled": False,
        "n_resource_types": 2,
        "resource_type_names": ("goal", "distractor_a"),
        "resource_type_drive_axes": ("goal_need", "d_a_need"),
        "resource_type_benefit_curves": ("sigmoidal_saturating",) * 2,
        "resource_type_distribution": dist,
        "resource_type_benefit_amplitudes": (1.0, 0.0),
        "dual_cue_enabled": distractors_present,
        "dual_cue_min_active_ticks": 10,
        "dual_cue_replace_on_early_consume": False,
        "dual_cue_type_tags": (GOAL_TAG, GOAL_TAG + 1),
    }


def _build_env(distractors_present: bool) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=None, **_env_kwargs(distractors_present))


def _build_agent(world_obs_dim: int, body_obs_dim: int = 12) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        z_goal_enabled=True,
    )
    return REEAgent(cfg)


def _pin_operating_mode_across_ticks(agent: REEAgent, mode: str) -> None:
    """V3-EXQ-874's SUBSTRATE-API FINDING, carried forward -- see V3-EXQ-940."""
    coord = agent.salience
    real_tick = coord.tick

    def _pinned_tick(*args, **kwargs):
        result = real_tick(*args, **kwargs)
        coord._operating_mode = {
            m: (1.0 if m == mode else 0.0) for m in coord.mode_names
        }
        coord._current_mode = mode
        result["operating_mode"] = dict(coord._operating_mode)
        result["current_mode"] = mode
        return result

    coord.tick = _pinned_tick


# --------------------------------------------------------------- geometry utils
def _agent_cell(env: CausalGridWorldV2) -> tuple:
    return (int(getattr(env, "agent_x", -1)), int(getattr(env, "agent_y", -1)))


def _goal_cells(env: CausalGridWorldV2) -> list:
    """Live benefit-bearing (goal-tag) resource cells, read off the env's type grid.

    Measured, not inferred from nominal spawn weights: the SD-049 allocator is
    stochastic (2026-08-19 probe measured 4 goal / 6 distractor cells at nominal
    (1.0, 1.0) over num_resources=12), which is exactly why 874b's ruleset axis
    silently varied nutritive density.
    """
    grid = getattr(env, "_resource_type_grid", None)
    if grid is None:
        return []
    arr = torch.as_tensor(grid)
    if arr.dim() != 2:
        return []
    out = []
    n_rows, n_cols = arr.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if int(arr[i, j].item()) == GOAL_TAG:
                out.append((i, j))
    return out


def _chebyshev(a: tuple, b: tuple) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _nearest_goal(pos: tuple, goals: list):
    """(cell, distance) of the nearest benefit-bearing resource, or (None, None)."""
    if not goals:
        return (None, None)
    best = min(goals, key=lambda g: _chebyshev(pos, g))
    return (best, _chebyshev(pos, best))


def _movement_positive_control(env_kwargs: dict, seed: int) -> int:
    """Positive control for the movement instrument -- the SAME statistic the
    load-bearing criterion routes on (a COUNT OF POSITION CHANGES), not a proxy.

    Steps a throwaway env through each cardinal action and counts how many changed the
    agent's cell. A below-floor reading means the instrument cannot see motion at all,
    which is a SUBSTRATE-NOT-READY condition, never a finding about the agent.
    """
    probe = CausalGridWorldV2(seed=seed, **env_kwargs)
    probe.reset()
    changed = 0
    for a in MOVE_ACTIONS:
        before = _agent_cell(probe)
        probe.step(a)
        if _agent_cell(probe) != before:
            changed += 1
    return changed


# ------------------------------------------------------------------- arm runner
def _run_arm(
    agent_base: REEAgent,
    arm: str,
    timing: str,
    env: CausalGridWorldV2,
    device: torch.device,
    smoke: bool,
    zg_acc: ZGoalStreamAccumulator,
    ep_acc: EpisodeTerminationAccumulator,
) -> dict:
    """One (seed, arm) cell: clone -> unpinned warmup -> pinned decomposition window."""
    agent = clone_trained_agent(agent_base, bistable=False, device=device)
    agent.eval()

    n_warmup = SMOKE_WARMUP if smoke else N_WARMUP_STEPS
    n_eval = SMOKE_EVAL if smoke else N_EVAL_STEPS

    agent.reset()
    _, obs_dict = env.reset()
    world_dim = agent.config.latent.world_dim
    last_info = {}

    def _step(obs):
        latent = agent.sense(
            obs["body_state"].to(device), obs["world_state"].to(device)
        )
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick")
            else torch.zeros(1, world_dim, device=device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)
        action_idx = int(action.argmax(dim=-1).item())
        _, _, done, info, next_obs = env.step(action_idx)
        agent.update_z_goal(
            float(info.get("benefit_exposure", 0.0) or 0.0), drive_level=1.0
        )
        return next_obs, info, done, ticks, action_idx

    # ---- P1: unpinned warmup ----
    truncated_in_warmup = False
    with torch.no_grad():
        for _ in range(n_warmup):
            obs_dict, info, done, _t, _a = _step(obs_dict)
            last_info = info
            if done:
                truncated_in_warmup = True
                break

    goal_state = agent.goal_state
    goal_live_at_warmup_end = bool(
        goal_state is not None and goal_state.is_active()
    )

    _pin_operating_mode_across_ticks(agent, timing)

    # ---- P2: pinned decomposition window ----
    n_ticks = 0
    n_e3_ticks = 0
    n_latched_ticks = 0
    n_move_actions = 0
    n_position_changes = 0
    n_approach_initiations = 0
    n_retreats = 0
    n_arrivals = 0
    n_events = 0
    distances = []
    approach_run_lengths = []
    benefit_cell_samples = []
    unique_cells = set()
    done_cause = ""
    truncated = False

    current_run = 0
    tracked_target = None
    prev_dist = None

    with torch.no_grad():
        for _step_i in range(n_eval):
            pos_before = _agent_cell(env)
            goals = _goal_cells(env)
            benefit_cell_samples.append(len(goals))
            unique_cells.add(pos_before)

            # Track the nearest benefit-bearing target BEFORE the step.
            target, dist_before = _nearest_goal(pos_before, goals)
            if dist_before is not None:
                distances.append(int(dist_before))

            obs_dict, info, done, ticks, action_idx = _step(obs_dict)
            last_info = info
            n_ticks += 1

            if ticks.get("e3_tick"):
                n_e3_ticks += 1
            else:
                n_latched_ticks += 1

            if action_idx in MOVE_ACTIONS:
                n_move_actions += 1

            pos_after = _agent_cell(env)
            if pos_after != pos_before:
                n_position_changes += 1

            consumed_tag = int(info.get("sd049_consumed_type_tag_this_tick", 0))
            if consumed_tag > 0:
                n_events += 1

            # --- approach bookkeeping, against the SAME target across the run ---
            if target is None or dist_before is None:
                # No benefit-bearing target on the grid: no approach is definable.
                if current_run > 0:
                    approach_run_lengths.append(current_run)
                current_run = 0
                tracked_target = None
                prev_dist = None
                if done:
                    done_cause = str(info.get("done_cause", "") or "")
                    truncated = True
                    break
                continue

            if tracked_target != target:
                # Target switched (consumed / respawned / a nearer one appeared):
                # close any open run and start tracking the new one.
                if current_run > 0:
                    approach_run_lengths.append(current_run)
                current_run = 0
                tracked_target = target
                prev_dist = dist_before

            dist_after = _chebyshev(pos_after, target)
            if dist_after < (prev_dist if prev_dist is not None else dist_before):
                n_approach_initiations += 1
                current_run += 1
                if dist_after == 0:
                    n_arrivals += 1
                    approach_run_lengths.append(current_run)
                    current_run = 0
                    tracked_target = None
            elif dist_after > (prev_dist if prev_dist is not None else dist_before):
                n_retreats += 1
                if current_run > 0:
                    approach_run_lengths.append(current_run)
                current_run = 0
            prev_dist = dist_after

            if done:
                done_cause = str(info.get("done_cause", "") or "")
                truncated = True
                break

    if current_run > 0:
        approach_run_lengths.append(current_run)

    ep_acc.record_from_info(last_info)
    # DEFECT (3) FIX: observe the STEPPING CLONE, never the P0 base agent.
    zg_acc.observe(agent)

    def _mean(xs, default=0.0):
        return round(statistics.fmean(xs), 6) if xs else default

    window_completeness = round(n_ticks / float(n_eval), 6) if n_eval else 0.0
    approach_initiation_rate = (
        round(n_approach_initiations / float(n_ticks), 6) if n_ticks else 0.0
    )
    approach_completion_rate = (
        round(n_arrivals / float(n_approach_initiations), 6)
        if n_approach_initiations > 0 else 0.0
    )
    move_efficacy = (
        round(n_position_changes / float(n_move_actions), 6)
        if n_move_actions else 0.0
    )

    return {
        "arm": arm,
        "timing": timing,
        "pinned_mode": timing,
        # --- defect (1): the window, recorded ---
        "n_budgeted_ticks": int(n_eval),
        "n_realised_ticks": int(n_ticks),
        "window_completeness": window_completeness,
        "truncated": bool(truncated),
        "truncated_in_warmup": bool(truncated_in_warmup),
        "done_cause": done_cause,
        # --- the decomposition ---
        "n_move_actions": int(n_move_actions),
        "n_position_changes": int(n_position_changes),
        "move_efficacy": move_efficacy,          # 1 - wall-blocking / no-op rate
        "n_blocked_moves": int(max(n_move_actions - n_position_changes, 0)),
        "n_approach_initiations": int(n_approach_initiations),
        "n_retreats": int(n_retreats),
        "n_arrivals": int(n_arrivals),
        "approach_initiation_rate": approach_initiation_rate,
        "approach_completion_rate": approach_completion_rate,
        "approach_run_lengths": [int(x) for x in approach_run_lengths],
        "approach_run_length_mean": _mean(approach_run_lengths),
        "approach_run_length_max": int(max(approach_run_lengths))
        if approach_run_lengths else 0,
        "n_consumption_events": int(n_events),
        # --- geometry ---
        "min_distance_to_goal": int(min(distances)) if distances else -1,
        "mean_distance_to_goal": _mean(distances),
        "n_distance_samples": len(distances),
        "n_unique_cells_visited": len(unique_cells),
        "n_benefit_bearing_resource_cells_mean": _mean(benefit_cell_samples),
        # --- cadence denominator (H-cadence's measurement, kept as instrumentation) ---
        "n_e3_ticks": int(n_e3_ticks),
        "n_latched_ticks": int(n_latched_ticks),
        "e3_tick_fraction": (
            round(n_e3_ticks / float(n_ticks), 6) if n_ticks else 0.0
        ),
        "goal_live_at_warmup_end": bool(goal_live_at_warmup_end),
    }


def run_seed(
    seed: int,
    device: torch.device,
    smoke: bool,
    zg_acc: ZGoalStreamAccumulator,
    ep_acc: EpisodeTerminationAccumulator,
) -> dict:
    torch.manual_seed(seed)
    p0_budget = SMOKE_P0_BUDGET if smoke else P0_BUDGET
    p0_steps = SMOKE_P0_STEPS if smoke else P0_STEPS_PER_EPISODE

    easy_env = _build_env(distractors_present=False)
    target_env = _build_env(distractors_present=True)
    agent = _build_agent(target_env.world_obs_dim).to(device)

    # Positive control for the movement instrument, measured per seed.
    control_changes = _movement_positive_control(_env_kwargs(True), seed)

    print(f"Seed {seed} Condition P0", flush=True)
    p0 = run_p0_warmup(
        agent, easy_env, device, budget=p0_budget, steps_per_episode=p0_steps
    )
    print(
        f"  [train] p0 seed={seed} ep {p0.n_episodes}/{p0_budget}"
        f" converged={p0.converged} aborted={p0.aborted} rv={p0.final_rv:.5f}"
        f" move_control={control_changes}/{len(MOVE_ACTIONS)}",
        flush=True,
    )
    p0_summary = {
        "n_episodes": int(p0.n_episodes),
        "converged": bool(p0.converged),
        "aborted": bool(p0.aborted),
        "final_rv": float(p0.final_rv),
    }

    arm_rows = {}
    if p0.aborted:
        for timing_name in TIMINGS:
            arm = f"ARM_{timing_name}"
            print(f"Seed {seed} Condition {arm}", flush=True)
            print("verdict: FAIL", flush=True)
            arm_rows[arm] = {
                "arm": arm, "timing": TIMINGS[timing_name],
                "pinned_mode": TIMINGS[timing_name],
                "p0_aborted": True, "p0_abort_reason": p0.abort_reason,
                "n_budgeted_ticks": 0, "n_realised_ticks": 0,
                "window_completeness": 0.0, "truncated": True,
                "truncated_in_warmup": False, "done_cause": "p0_aborted",
                "n_move_actions": 0, "n_position_changes": 0, "move_efficacy": 0.0,
                "n_blocked_moves": 0, "n_approach_initiations": 0, "n_retreats": 0,
                "n_arrivals": 0, "approach_initiation_rate": 0.0,
                "approach_completion_rate": 0.0, "approach_run_lengths": [],
                "approach_run_length_mean": 0.0, "approach_run_length_max": 0,
                "n_consumption_events": 0, "min_distance_to_goal": -1,
                "mean_distance_to_goal": 0.0, "n_distance_samples": 0,
                "n_unique_cells_visited": 0,
                "n_benefit_bearing_resource_cells_mean": 0.0,
                "n_e3_ticks": 0, "n_latched_ticks": 0, "e3_tick_fraction": 0.0,
                "goal_live_at_warmup_end": False,
                "movement_control_changes": int(control_changes),
            }
        return {"seed": seed, "p0": p0_summary,
                "movement_control_changes": int(control_changes), "arms": arm_rows}

    for timing_name, mode in TIMINGS.items():
        arm = f"ARM_{timing_name}"
        print(f"Seed {seed} Condition {arm}", flush=True)
        with arm_cell(
            seed,
            config_slice={
                "arm": arm,
                "timing": timing_name,
                "operating_mode": mode,
                "env": _env_kwargs(True),
                "n_warmup_steps": N_WARMUP_STEPS,
                "n_eval_steps": N_EVAL_STEPS,
                "p0_budget": P0_BUDGET,
                "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
            },
            script_path=Path(__file__),
            config_slice_declared=True,
            # Both timing arms clone the SAME p0-trained agent, so neither cell is a
            # pure function of (seed, arm config) from a fresh RNG reset. Honestly
            # ineligible rather than falsely marked reusable.
            extra_ineligible_reasons=["shared_p0_warmup_across_timing_arms"],
        ) as cell:
            row = _run_arm(
                agent, arm, mode, target_env, device, smoke, zg_acc, ep_acc
            )
            row["p0_aborted"] = False
            row["movement_control_changes"] = int(control_changes)
            cell.stamp(row)
        arm_rows[arm] = row
        print(
            f"verdict: {'PASS' if row['n_realised_ticks'] > 0 else 'FAIL'}"
            f" window={row['window_completeness']:.4f}"
            f" ticks={row['n_realised_ticks']}/{row['n_budgeted_ticks']}"
            f" cause={row['done_cause'] or 'none'}"
            f" init_rate={row['approach_initiation_rate']:.4f}"
            f" completion={row['approach_completion_rate']:.4f}"
            f" moves={row['n_position_changes']}/{row['n_move_actions']}"
            f" runmax={row['approach_run_length_max']}"
            f" events={row['n_consumption_events']}",
            flush=True,
        )

    return {"seed": seed, "p0": p0_summary,
            "movement_control_changes": int(control_changes), "arms": arm_rows}


# ------------------------------------------------------------ precondition gate
def _precondition_specs(smoke: bool):
    min_ticks = SMOKE_MIN_REALISED_TICKS if smoke else MIN_REALISED_TICKS
    min_dist = SMOKE_MIN_DISTANCE_SAMPLES if smoke else MIN_DISTANCE_SAMPLES
    return [
        PreconditionSpec(
            name="movement_instrument_live",
            description=(
                "Forced cardinal movement actions from a fresh env DO change the "
                "agent's cell, so a count of position changes measures motion rather "
                "than instrument failure."
            ),
            control=(
                "Positive control: a throwaway env stepped through each of the 4 "
                "cardinal actions, counting cell changes. This is the SAME statistic "
                "the load-bearing criterion routes on (a COUNT OF POSITION CHANGES), "
                "not a magnitude or distance proxy for it -- the V3-EXQ-643 "
                "same-statistic rule. Measured 2026-08-19 on this env: 4 of 4."
            ),
            threshold=float(MIN_POSITION_CHANGES_CONTROL - 1),  # strict >, so >=1 passes
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="window_instrument_live",
            description=(
                "The cell stepped the decomposition window at all. Below this floor no "
                "rate normalised by realised ticks is measurable."
            ),
            control="n_realised_ticks counted directly in the eval loop.",
            threshold=float(min_ticks),
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="benefit_bearing_resources_present",
            description=(
                "Benefit-bearing (goal-tag) resource cells were live on the grid. "
                "874b defect (2) made into a gate: an APPROACH is undefined without a "
                "target, and 874b let this count vary silently with its ruleset axis."
            ),
            control=(
                "Mean live goal-tag cells read off the env's own _resource_type_grid "
                "each tick -- measured, not inferred from nominal spawn weights."
            ),
            threshold=0.0,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="approach_is_definable",
            description=(
                "The agent was not already sitting on / adjacent to its nearest target "
                "for the whole window, so there was somewhere to approach FROM."
            ),
            control=(
                "Mean Chebyshev distance to the nearest benefit-bearing cell over the "
                "window. Step 2.5b adversarial design audit (2026-08-19): with 12 "
                "resources on a 6x6 grid the mean distance is small, and at distance 0 "
                "no STRICT decrease is possible -- so a near-zero mean distance would "
                "produce an approach_initiation_rate of ~0 that ALIASES onto the "
                "`never_approached` verdict while meaning the opposite (the agent was "
                "already there). Same statistic the initiation count is derived from."
            ),
            threshold=0.0,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="distance_instrument_live",
            description=(
                "A distance-to-nearest-target was computable on the window's ticks, so "
                "an approach/retreat classification is defined."
            ),
            control="n_distance_samples counted whenever a goal-tag cell existed.",
            threshold=float(min_dist),
            direction="lower",
            kind="readiness",
        ),
    ]


def _arm_contexts():
    return {
        f"ARM_{t}": {"arm_id": f"ARM_{t}", "timing": t, "operating_mode": m}
        for t, m in TIMINGS.items()
    }


def _pool_arm(seed_results, arm) -> dict:
    rows = [r["arms"][arm] for r in seed_results if arm in r["arms"]]
    live = [r for r in rows if not r.get("p0_aborted")]

    def _mean(key, default=0.0):
        vals = [float(r.get(key, 0.0)) for r in live]
        return round(statistics.fmean(vals), 6) if vals else default

    def _sum(key):
        return sum(int(r.get(key, 0)) for r in live)

    total_ticks = _sum("n_realised_ticks")
    total_init = _sum("n_approach_initiations")
    total_arrivals = _sum("n_arrivals")
    total_moves = _sum("n_move_actions")
    total_pos = _sum("n_position_changes")

    causes = {}
    for r in live:
        c = r.get("done_cause") or ("none" if not r.get("truncated") else "unknown")
        causes[c] = causes.get(c, 0) + 1

    all_runs = []
    for r in live:
        all_runs.extend(list(r.get("approach_run_lengths") or []))

    return {
        "arm": arm,
        "n_cells": len(rows),
        "n_live_cells": len(live),
        "window_completeness_mean": _mean("window_completeness"),
        "n_truncated_cells": sum(1 for r in live if r.get("truncated")),
        "done_causes": causes,
        "pooled_realised_ticks": int(total_ticks),
        "pooled_move_actions": int(total_moves),
        "pooled_position_changes": int(total_pos),
        "pooled_blocked_moves": int(max(total_moves - total_pos, 0)),
        "pooled_move_efficacy": (
            round(total_pos / float(total_moves), 6) if total_moves else 0.0
        ),
        "pooled_approach_initiations": int(total_init),
        "pooled_arrivals": int(total_arrivals),
        "pooled_approach_initiation_rate": (
            round(total_init / float(total_ticks), 6) if total_ticks else 0.0
        ),
        "pooled_approach_completion_rate": (
            round(total_arrivals / float(total_init), 6) if total_init else 0.0
        ),
        "pooled_consumption_events": _sum("n_consumption_events"),
        "approach_run_length_max": int(max(all_runs)) if all_runs else 0,
        "approach_run_length_mean": (
            round(statistics.fmean(all_runs), 6) if all_runs else 0.0
        ),
        "n_approach_runs": len(all_runs),
        "min_distance_to_goal_min": (
            int(min(int(r.get("min_distance_to_goal", -1)) for r in live))
            if live else -1
        ),
        "mean_distance_to_goal_mean": _mean("mean_distance_to_goal"),
        "n_unique_cells_visited_mean": _mean("n_unique_cells_visited"),
        "n_benefit_bearing_resource_cells_mean": _mean(
            "n_benefit_bearing_resource_cells_mean"
        ),
        "e3_tick_fraction_mean": _mean("e3_tick_fraction"),
        "n_realised_ticks_mean": _mean("n_realised_ticks"),
        "n_distance_samples_mean": _mean("n_distance_samples"),
        "movement_control_changes_min": (
            int(min(int(r.get("movement_control_changes", 0)) for r in live))
            if live else 0
        ),
        "goal_live_at_warmup_end_frac": (
            round(
                sum(1 for r in live if r.get("goal_live_at_warmup_end"))
                / float(len(live)), 6
            ) if live else 0.0
        ),
    }


def build_manifest(seed_results, smoke: bool, started_at: float) -> dict:
    pooled = {arm: _pool_arm(seed_results, arm) for arm in ARMS}
    specs = _precondition_specs(smoke)
    contexts = _arm_contexts()

    arm_gates = []
    for arm in ARMS:
        p = pooled[arm]
        measured = {
            "movement_instrument_live": float(p["movement_control_changes_min"]),
            "window_instrument_live": float(p["n_realised_ticks_mean"]),
            "benefit_bearing_resources_present": float(
                p["n_benefit_bearing_resource_cells_mean"]
            ),
            "distance_instrument_live": float(p["n_distance_samples_mean"]),
            "approach_is_definable": float(p["mean_distance_to_goal_mean"]),
        }
        arm_gates.append(evaluate_arm_gate(arm, contexts[arm], specs, measured))
    aggregate = aggregate_arm_gates(arm_gates)

    # ---- the decomposition, pooled across arms -------------------------------
    total_init = sum(pooled[a]["pooled_approach_initiations"] for a in ARMS)
    total_ticks = sum(pooled[a]["pooled_realised_ticks"] for a in ARMS)
    total_arrivals = sum(pooled[a]["pooled_arrivals"] for a in ARMS)
    overall_init_rate = (
        round(total_init / float(total_ticks), 6) if total_ticks else 0.0
    )
    overall_completion_rate = (
        round(total_arrivals / float(total_init), 6) if total_init else 0.0
    )

    total_events = sum(pooled[a]["pooled_consumption_events"] for a in ARMS)
    c1_pass = bool(overall_init_rate >= APPROACH_INITIATION_FLOOR)
    c2_pass = bool(overall_completion_rate >= APPROACH_COMPLETION_FLOOR)
    # C3 closes a VERDICT-ALIASING gap found by the Step 2.5b adversarial design audit
    # (2026-08-19, pre-queue). Arrival is measured geometrically (Chebyshev distance to
    # a goal-tag cell reaching 0) while a consumption EVENT is the env's own
    # sd049_consumed_type_tag_this_tick. Those can come apart. Without C3, a run in
    # which the agent reliably ARRIVES and never EATS would satisfy C1 and C2 and route
    # to `approach_pipeline_intact` -- the DECLARED NULL -- which would be exactly
    # wrong: the denominator would be lost at the consumption step, downstream of
    # everything this leg was built to see, and the null would hide it.
    c3_pass = bool(total_events > 0)
    # C2 compares arrivals AGAINST initiations. With no initiation anywhere there is
    # no completion rate to read -- degenerate, not a null.
    c2_non_degenerate = bool(total_init > 0)
    # C3 is only readable once the agent actually arrived somewhere.
    c3_non_degenerate = bool(total_arrivals > 0)

    criteria_by_arm = {
        ARMS[0]: ["C1_approaches_are_initiated", "C3_arrivals_convert_to_consumption"],
        ARMS[1]: ["C2_initiated_approaches_complete"],
    }
    criteria_nd = arm_criteria_non_degenerate(
        criteria_by_arm,
        aggregate,
        extra={
            "C2_initiated_approaches_complete": c2_non_degenerate,
            "C3_arrivals_convert_to_consumption": c3_non_degenerate,
        },
    )

    # ---- self-route ---------------------------------------------------------
    if not aggregate["non_degenerate"]:
        label = "substrate_not_ready_requeue"
        decomposition = "undetermined"
        interpretation_note = (
            "No arm cleared its readiness gate -- in particular the movement/distance "
            "instruments. Nothing here is a verdict on MECH-467 or on where the "
            "denominator is lost."
        )
    elif not c1_pass:
        label = "denominator_lost_at_approach_initiation"
        decomposition = "never_approached"
        interpretation_note = (
            "Approaches are essentially never initiated: the agent does not head for "
            "benefit-bearing targets at all, so the missing consumption events are lost "
            "UPSTREAM of arrival. This confirms at battery level the substrate-level "
            "reading of the 2026-08-18 navigation-immobility scoping spike (MECH-439 "
            "F-dominance conversion ceiling, amplified by the E3 hold-and-repeat "
            "cadence). Read pooled_move_efficacy and e3_tick_fraction_mean alongside: "
            "they say how much of the immobility is wall-blocking and how much is "
            "cadence rather than selection."
        )
    elif c1_pass and not c2_pass:
        label = "denominator_lost_between_initiation_and_arrival"
        decomposition = "approached_and_failed_to_arrive"
        interpretation_note = (
            "Approaches ARE initiated but essentially never complete. The denominator "
            "is lost in transit, not at intention. pooled_move_efficacy separates "
            "wall-blocking (a blocked move is absorbed as a no-op by the env) from "
            "commitment abandonment; approach_run_length_max says how far runs get "
            "before breaking."
        )
    elif not c3_pass:
        label = "denominator_lost_at_consumption"
        decomposition = "arrived_without_consuming"
        interpretation_note = (
            "Approaches are initiated AND complete -- the agent reaches benefit-bearing "
            "cells -- but no consumption event fires. The denominator is lost at the "
            "consumption step, DOWNSTREAM of the whole approach pipeline. This branch "
            "exists because without it this reading would have satisfied C1 and C2 and "
            "routed to the declared null, hiding the defect (Step 2.5b adversarial "
            "design audit, 2026-08-19). Compare pooled_arrivals against "
            "pooled_consumption_events, and check the env's consummatory-act "
            "configuration before reading anything else."
        )
    else:
        label = "approach_pipeline_intact"
        decomposition = "null_decomposition_uninformative"
        interpretation_note = (
            "DECLARED NULL, in the autopsy's own wording: the decomposition is "
            "uninformative because approaches are initiated and complete at the "
            "observed rate. The leg-(c) denominator problem is not visible anywhere in "
            "the approach pipeline, and lies outside what this decomposition can see."
        )

    outcome = (
        "PASS"
        if (aggregate["non_degenerate"] and c1_pass and c2_pass and c3_pass)
        else "FAIL"
    )

    return {
        "run_id": f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": _utc_stamp(),
        "generated_utc": _utc_iso(),
        "smoke": bool(smoke),
        "portfolio": {
            "name": "GOV-FANOUT-1 MECH-467 leg-(c) denominator discrimination",
            "leg": "H-denominator",
            "axis_family": "measurement",
            "sibling_legs": ["V3-EXQ-940 (H-energy, environment)"],
            "legs_not_queued": {
                "H-commitment": (
                    "answered at substrate level by "
                    "navigation_immobility_scoping_2026-08-18.md -- MECH-439 "
                    "F-dominance conversion ceiling, ceiling_decision exhausted. Its "
                    "MEASUREMENT (approach-run length) is instrumented HERE."
                ),
                "H-cadence": (
                    "answered at substrate level (E3 heartbeat hold-and-repeat, "
                    "e3_steps_per_tick); deeper cause owned by the ACTIVE "
                    "hypothesis-space line e3_fdominance_causal_discrimination. Its "
                    "MEASUREMENT (e3_tick vs latched fraction) is instrumented HERE."
                ),
            },
        },
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "decomposition": {
            "verdict": decomposition,
            "overall_approach_initiation_rate": overall_init_rate,
            "overall_approach_completion_rate": overall_completion_rate,
            "pooled_approach_initiations": int(total_init),
            "pooled_arrivals": int(total_arrivals),
            "pooled_realised_ticks": int(total_ticks),
            "pooled_consumption_events": int(total_events),
        },
        "per_arm_pooled": pooled,
        "per_seed_results": seed_results,
        "criteria": [
            {
                "name": "C1_approaches_are_initiated",
                "load_bearing": True,
                "passed": bool(c1_pass),
                "measured": overall_init_rate,
                "threshold": APPROACH_INITIATION_FLOOR,
                "owned_by_arms": [ARMS[0]],
            },
            {
                "name": "C2_initiated_approaches_complete",
                "load_bearing": True,
                "passed": bool(c2_pass),
                "measured": overall_completion_rate,
                "threshold": APPROACH_COMPLETION_FLOOR,
                "owned_by_arms": [ARMS[1]],
                "note": (
                    "Degenerate when no approach was ever initiated: a completion rate "
                    "over zero initiations discriminates nothing."
                ),
            },
            {
                "name": "C3_arrivals_convert_to_consumption",
                "load_bearing": True,
                "passed": bool(c3_pass),
                "measured": int(total_events),
                "threshold": 1,
                "owned_by_arms": [ARMS[0]],
                "note": (
                    "Added by the Step 2.5b adversarial design audit (2026-08-19) to "
                    "close a verdict-aliasing gap: without it, arriving reliably and "
                    "never eating would have routed to the declared null. Degenerate "
                    "when the agent never arrived anywhere."
                ),
            },
        ],
        "combination_rule": (
            "This leg's DELIVERABLE is the decomposition label, not the PASS. outcome "
            "PASS iff the gate is non-degenerate AND C1 AND C2 AND C3 -- i.e. the approach "
            "pipeline is intact end to end, which is the DECLARED NULL. A FAIL here is "
            "the INFORMATIVE outcome: it localises where the denominator is lost, "
            "which is the whole purpose of routing this leg to the measurement axis. "
            "Read interpretation.label / decomposition.verdict, not outcome alone."
        ),
        "interpretation": {
            "label": label,
            "note": interpretation_note,
            "preconditions": aggregate["adjudication_preconditions"],
            "criteria_non_degenerate": criteria_nd,
            "preconditions_scope_note": aggregate.get("preconditions_scope_note", ""),
        },
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": bool(aggregate["non_degenerate"]),
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "dv_symmetry_declaration": {
            "dv": "approach_initiation_rate and approach_completion_rate",
            "symmetry_group": "permutation of ticks (both DVs are set-aggregates)",
            "manipulation_invariant_under_it": False,
            "per_arm": {
                arm: (
                    "NOT invariant: the operating-mode pin changes the SELECTION POLICY "
                    "and hence which actions are taken and which distances occur. It is "
                    "not a broadcast scalar added uniformly across candidates, not a "
                    "monotone rescaling preserving candidate order, and not a "
                    "permutation of interchangeable units."
                )
                for arm in ARMS
            },
        },
        "defects_of_874b_addressed": {
            "unrecorded_truncation": (
                "done_cause, truncated, n_realised_ticks, n_budgeted_ticks and "
                "window_completeness recorded per cell; SD-094 "
                "EpisodeTerminationAccumulator fed; every rate normalised by REALISED "
                "ticks."
            ),
            "ruleset_nutritive_density_confound": (
                "The ruleset axis is not varied at all (SIMPLE only), so the confound "
                "cannot arise; n_benefit_bearing_resource_cells measured off the env's "
                "type grid and gated by a readiness precondition."
            ),
            "false_z_goal_writer_defect": (
                "ZGoalStreamAccumulator.observe() called on the stepping CLONE inside "
                "the per-cell arm function, never on the P0 base agent."
            ),
        },
        "mech262_constraint": (
            "This driver reads rule state at NEITHER the storage site nor the selection "
            "path -- it is a denominator diagnostic and takes no rule measurement. "
            "MECH-262 is not tagged."
        ),
        "sleep_driver_pattern": "none (sleep not used)",
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
            "note": "V3 diagnostic; pure instrumentation over an existing env.",
        },
    }


def _full_config() -> dict:
    return {
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "num_hazards": 0,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "n_warmup_steps": N_WARMUP_STEPS,
        "n_eval_steps": N_EVAL_STEPS,
        "p0_budget": P0_BUDGET,
        "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
        "timings": dict(TIMINGS),
        "move_actions": list(MOVE_ACTIONS),
        "env_kwargs": _env_kwargs(True),
        "thresholds": {
            "approach_initiation_floor": APPROACH_INITIATION_FLOOR,
            "approach_completion_floor": APPROACH_COMPLETION_FLOOR,
            "min_realised_ticks": MIN_REALISED_TICKS,
            "min_position_changes_control": MIN_POSITION_CHANGES_CONTROL,
            "min_distance_samples": MIN_DISTANCE_SAMPLES,
        },
        "seeds": list(SEEDS),
    }


def main(smoke: bool):
    started_at = time.perf_counter()
    device = torch.device("cpu")

    assert_no_structurally_unsatisfiable_gate(
        _precondition_specs(smoke),
        list(_arm_contexts().values()),
        arm_id_key="arm_id",
    )

    zg_acc = ZGoalStreamAccumulator()
    ep_acc = EpisodeTerminationAccumulator(
        steps_configured=(SMOKE_EVAL if smoke else N_EVAL_STEPS)
    )

    seed_results = []
    seeds = SEEDS[:1] if smoke else SEEDS
    for seed in seeds:
        seed_results.append(run_seed(seed, device, smoke, zg_acc, ep_acc))

    manifest = build_manifest(seed_results, smoke, started_at)
    out_path = write_flat_manifest(
        manifest,
        dry_run=smoke,
        config=_full_config(),
        seeds=list(SEEDS),
        script_path=Path(__file__),
        started_at=started_at,
        z_goal_stream_stats=zg_acc.stats(),
        episode_termination=ep_acc,
    )

    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"decomposition: {manifest['decomposition']['verdict']}", flush=True)
    print(f"non_degenerate: {manifest['non_degenerate']}", flush=True)
    print(f"manifest: {out_path}", flush=True)
    return manifest, out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="short smoke run")
    args = ap.parse_args()
    _manifest, _out_path = main(smoke=args.dry_run)
    _outcome_raw = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=args.dry_run,
    )
