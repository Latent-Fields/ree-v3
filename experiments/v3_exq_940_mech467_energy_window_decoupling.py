"""V3-EXQ-940: MECH-467 leg H-energy -- is the leg-(c) exposure window destroyed by
SELF-INFLICTED CONTAMINATION or by genuine consumption starvation?

EXPERIMENT_PURPOSE: see EXPERIMENT_PURPOSE constant below.

GOV-FANOUT-1 PORTFOLIO MEMBER (leg 1 of 2). Sibling: V3-EXQ-941 (H-denominator).
Authority: failure_autopsy_V3-EXQ-874b_2026-08-17 (confirmed; disposition applied to
MECH-467 by the /governance cycle of 2026-08-18). Design + the 4-leg -> 2-leg scope
shrink: REE_assembly/evidence/planning/mech467_govfanout_portfolio_staged_2026-08-19.md

THIS IS NOT A LETTERED RE-POSE OF 874b. The autopsy explicitly refuses one. 874b asked
whether REE resists distractors; this asks why its measurement window collapsed. It is a
DIAGNOSTIC, excluded from governance confidence scoring by construction.

WHY THIS LEG EXISTS
-------------------
V3-EXQ-874b produced 1/3/4/0 pooled consumption events per arm against a pre-registered
floor of 15, all four arms gate-RED, non_degenerate false. 7 of 12 cells terminated early
on health_depleted and only 63.1% of the pre-registered exposure window was realised --
and NO cell recorded that. The autopsy reads those terminations as the agent having
"starved to death", i.e. as a CONSEQUENCE of not eating.

There is a second, fully documented mechanism 874b was exposed to and did not opt out of.
CausalGridWorldV2 applies contamination_spread (DEFAULT 0.5) to EVERY cell the agent
enters, regardless of num_hazards. Once a cell crosses contamination_threshold (default
2.0, i.e. four entries) it becomes ENTITY_TYPES["contaminated"] and drains
contaminated_harm (default 0.4) per contact -- so roughly three contacts are lethal from
full health (1.0). The env's own module docstring names this a footgun and cites the
V3-EXQ-884 precedent, where episodes died at 32/19/90 steps of a configured 400.

874b set num_hazards=0 and set NEITHER contamination_spread=0.0 NOR
hazard_free_contamination_gate=True. Measured directly on 874b's exact env kwargs
(2026-08-19 pre-authoring probe, this substrate):

    stock 874b config            -> contamination_spread = 0.5   (the exposure)
    + hazard_free_contamination_gate=True -> contamination_spread = 0.0

On a 6x6 grid with an agent that revisits a small cell set, self-poisoning is a live
alternative to consumption starvation WITH AN ENTIRELY DIFFERENT REMEDY. That is the
discrimination this leg runs.

ARMS -- one axis only (3 arms x 3 seeds = 9 cells)
--------------------------------------------------
  ARM_STOCK             874b's env kwargs verbatim; contamination at stock defaults.
                        Reproduction arm.
  ARM_CONTAM_OFF        identical + hazard_free_contamination_gate=True. Isolates
                        self-contamination from every other property of the geometry.
  ARM_HEALTH_DECOUPLED  contamination gated off AND agent_health clamped to
                        HEALTH_CLAMP_FLOOR after every step, so the window CANNOT
                        terminate for health reasons. This is the autopsy's
                        "decouple survival from the measurement window (health decay
                        disabled or satiety clamped) at identical geometry".

operating_mode is pinned to internal_planning in ALL THREE arms, so the energy
manipulation is the only axis varying. The internal_replay regime is deliberately NOT
covered here; the sibling leg V3-EXQ-941 covers both regimes.

THE THREE 874b DEFECTS, AND HOW THIS DRIVER AVOIDS EACH
--------------------------------------------------------
(1) UNRECORDED TRUNCATION. 874b's eval loop ended `if done: break` with no reset and no
    re-entry, so a 60-tick cell and a 900-tick cell entered the pooled denominator
    indistinguishably. Here: every cell records done_cause (the env supplies it in
    info; 874b never read it), a truncated flag, n_realised_ticks, n_budgeted_ticks and
    window_completeness; the run additionally feeds the shared SD-094
    EpisodeTerminationAccumulator, whose motivating incident IS V3-EXQ-884's
    contamination death. EVERY RATE DV IS NORMALISED BY REALISED TICKS, NEVER BY THE
    BUDGET -- that normalisation is the whole point of the leg.
(2) RULE-SET COMPLEXITY CONFOUNDED WITH NUTRITIVE DENSITY. 874b held num_resources at 12
    while only type 0 carries benefit, so SIMPLE had 6/12 benefit-bearing cells and
    COMPLEX 4/12: adding distractor TYPES silently removed a third of the food, and
    COMPLEX realised 41% of its window against SIMPLE's 85%. Here the ruleset axis is
    NOT VARIED AT ALL -- SIMPLE only. A confound between two axes cannot arise when one
    is not manipulated. n_benefit_bearing_resource_cells is measured off the env's own
    type grid and recorded per cell, so the quantity that was silently varying in 874b
    is now an explicit number with a readiness precondition on it.
(3) FALSE z_goal writer_defect. 874b called zg_acc.observe(agent) on the P0 BASE agent
    (n_agents: 6 = 2 rulesets x 3 seeds) while every cell stepped a CLONE, publishing a
    false writer_defect into pending_review.md. Here observe() is called on the STEPPING
    CLONE, inside the per-cell arm function, after the eval window.

MECH-262 CONSTRAINT (from the same autopsy). Storage-site rule drift and selection-path
operative-rule fidelity DISSOCIATE in 6 of 9 live cells, so a storage-site read does not
track the operative rule. This driver reads rule state at NEITHER site -- it is a
denominator diagnostic and takes no rule measurement at all. MECH-262 is not tagged.

DV-SYMMETRY DECLARATION (mandatory, per arm -- the V3-EXQ-604c rule). The DV is a ratio
of counts over REALISED ticks; its symmetry group is permutation of ticks (a
set-aggregate). The manipulation (contamination gating / health clamping) changes which
ticks exist and whether the terminal condition fires. It is NOT a broadcast constant
added across candidates, NOT a monotone rescaling of candidate scores, and NOT a
permutation of interchangeable units. It is therefore NOT INVARIANT under the DV's
symmetry group in ARM_STOCK, ARM_CONTAM_OFF or ARM_HEALTH_DECOUPLED.

STRUCTURAL-VACUITY DECLARATION (the V3-EXQ-785 rule). In ARM_HEALTH_DECOUPLED,
window_completeness is forced to ~1.0 BY THE MANIPULATION -- it is a manipulation check
there, not a measurement. C1 is therefore owned only by the two arms in which it can
move, and ARM_HEALTH_DECOUPLED is scored on C2 alone. This is disposition (a) (the
criterion is not meaningful for that regime), NOT a vacuous arm: the arm is green and
scorable on its own DV.

PSEUDO-REPLICATION. This driver reads NO agent.e3.last_* latched diagnostic. Ticks on
which the E3 cadence held the previous action are counted in n_latched_ticks and
contribute nothing to any per-selection quantity.

CONTAINMENT. Diagnostic over EXISTING substrate. No new module, no substrate change; the
contamination gate and the health attribute are both existing, documented env surface.

Run:
  /opt/local/bin/python3 experiments/v3_exq_940_mech467_energy_window_decoupling.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_940_mech467_energy_window_decoupling.py --dry-run
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


EXPERIMENT_TYPE = "v3_exq_940_mech467_energy_window_decoupling"
QUEUE_ID = "V3-EXQ-940"
CLAIM_IDS = ["MECH-467"]
EXPERIMENT_PURPOSE = "diagnostic"
# Not a reef-config env, so the seed-44 early-truncation instability does not apply
# (V3-EXQ-874's own note, carried forward and re-checked here).
SEEDS = [42, 43, 44]

GOAL_TAG = 1  # SD-049 tag = type_idx + 1; type_idx 0 is the benefit-bearing goal

# ---- arms: contamination / health manipulation ONLY -------------------------
ARM_STOCK = "ARM_STOCK"
ARM_CONTAM_OFF = "ARM_CONTAM_OFF"
ARM_HEALTH_DECOUPLED = "ARM_HEALTH_DECOUPLED"
ARMS = (ARM_STOCK, ARM_CONTAM_OFF, ARM_HEALTH_DECOUPLED)
ARM_SPECS = {
    ARM_STOCK: {"contam_gate": False, "clamp_health": False},
    ARM_CONTAM_OFF: {"contam_gate": True, "clamp_health": False},
    ARM_HEALTH_DECOUPLED: {"contam_gate": True, "clamp_health": True},
}
PINNED_MODE = "internal_planning"

# ---- pre-registered constants (NOT derived from the run's own statistics) ----
# Geometry is 874b's VERBATIM, on purpose: this leg asks what killed 874b's window, so
# changing the geometry would change the question.
GRID_SIZE = 6
NUM_RESOURCES = 12
MAX_EPISODE_STEPS = 1500
N_WARMUP_STEPS = 80
N_EVAL_STEPS = 900
P0_BUDGET = 60
P0_STEPS_PER_EPISODE = 80
HEALTH_CLAMP_FLOOR = 0.5   # env terminal is agent_health <= 0.0; 0.5 cannot reach it

# Criterion thresholds, pre-registered.
WINDOW_COMPLETENESS_LIFT_MIN = 0.15  # C1: absolute lift in pooled window_completeness
EVENT_RATE_LIFT_MIN = 1.5            # C2: multiplicative lift in events / realised tick
# Readiness floors.
MIN_REALISED_TICKS = 20              # below this nothing about the window is measurable
MIN_HEALTH_SAMPLES = 20
CONTAM_GATE_CEILING = 0.001          # gate applied => contamination_spread == 0.0

# Smoke budgets
SMOKE_P0_BUDGET = 3
SMOKE_P0_STEPS = 15
SMOKE_WARMUP = 5
SMOKE_EVAL = 40
SMOKE_MIN_REALISED_TICKS = 5
SMOKE_MIN_HEALTH_SAMPLES = 5


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ------------------------------------------------------------------ env / agent
def _env_kwargs(distractors_present: bool, contam_gate: bool) -> dict:
    """874b's SIMPLE-ruleset env kwargs, verbatim, plus the contamination gate.

    The ruleset axis is deliberately NOT varied in this leg -- see defect (2) in the
    module docstring. resource_type_distribution (1.0, 1.0) over 2 types is 874b's
    SIMPLE arm exactly.
    """
    dist = (1.0, 1.0) if distractors_present else (1.0, 0.0)
    kwargs = {
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
    if contam_gate:
        # SD-094 / V3-EXQ-884 opt-out. Zeroes contamination_spread when num_hazards==0.
        kwargs["hazard_free_contamination_gate"] = True
    return kwargs


def _build_env(distractors_present: bool, contam_gate: bool) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=None, **_env_kwargs(distractors_present, contam_gate))


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
    """Wrap agent.salience.tick so the FORCED mode survives every subsequent call.

    V3-EXQ-874's SUBSTRATE-API FINDING, carried forward: SalienceCoordinator.tick()
    unconditionally recomputes _operating_mode on every call and agent.py calls it
    inside select_action(), so a one-shot attribute assignment is clobbered before
    write_gate() is read. This runs the REAL tick (preserving bookkeeping) then forces
    the pin.
    """
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


def _live_resource_composition(env: CausalGridWorldV2) -> tuple:
    """(n_benefit_bearing_cells, n_distractor_cells) currently live on the grid.

    Defect (2)'s antidote: 874b never recorded this, so the nutritive density that was
    silently varying with its ruleset axis was invisible. Measured off the env's own
    type grid, not inferred from the nominal spawn weights (the SD-049 allocator is
    stochastic -- a 2026-08-19 probe measured 4 goal / 6 distractor cells at nominal
    (1.0, 1.0) weights over num_resources=12).
    """
    grid = getattr(env, "_resource_type_grid", None)
    if grid is None:
        return (0, 0)
    tags = torch.as_tensor(grid).reshape(-1).tolist()
    n_goal = sum(1 for t in tags if int(t) == GOAL_TAG)
    n_distr = sum(1 for t in tags if int(t) > GOAL_TAG)
    return (n_goal, n_distr)


# ------------------------------------------------------------------- arm runner
def _run_arm(
    agent_base: REEAgent,
    arm: str,
    env: CausalGridWorldV2,
    device: torch.device,
    smoke: bool,
    zg_acc: ZGoalStreamAccumulator,
    ep_acc: EpisodeTerminationAccumulator,
) -> dict:
    """One (seed, arm) cell: clone -> unpinned warmup -> pinned exposure window.

    ONE continuous episode; no mid-run agent.reset() (874's RULE-STATE-RESET FINDING:
    agent.reset() zeroes lateral_pfc.rule_state).
    """
    spec = ARM_SPECS[arm]
    agent = clone_trained_agent(agent_base, bistable=False, device=device)
    agent.eval()

    n_warmup = SMOKE_WARMUP if smoke else N_WARMUP_STEPS
    n_eval = SMOKE_EVAL if smoke else N_EVAL_STEPS

    agent.reset()
    _, obs_dict = env.reset()
    world_dim = agent.config.latent.world_dim

    # Manipulation check: what contamination_spread did the env ACTUALLY bind?
    realised_contam_spread = float(getattr(env, "contamination_spread", -1.0))

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
        # MECH-112/117: update_z_goal is the SOLE writer of z_goal. Omitting it pins
        # z_goal at zero-init and silently disables the E3 goal channel.
        agent.update_z_goal(
            float(info.get("benefit_exposure", 0.0) or 0.0), drive_level=1.0
        )
        if spec["clamp_health"]:
            # ARM_HEALTH_DECOUPLED: the window cannot terminate for health reasons.
            # Applied AFTER step so the tick's harm is felt and then undone -- the
            # measurement window is decoupled, the dynamics are not silently removed.
            env.agent_health = max(float(env.agent_health), HEALTH_CLAMP_FLOOR)
        return next_obs, info, done, ticks

    # ---- P1: unpinned warmup ----
    truncated_in_warmup = False
    with torch.no_grad():
        for _ in range(n_warmup):
            obs_dict, info, done, _t = _step(obs_dict)
            last_info = info
            if done:
                truncated_in_warmup = True
                break

    goal_state = agent.goal_state
    goal_live_at_warmup_end = bool(
        goal_state is not None and goal_state.is_active()
    )

    _pin_operating_mode_across_ticks(agent, PINNED_MODE)

    # ---- P2: pinned exposure window ----
    n_ticks = 0
    n_e3_ticks = 0
    n_latched_ticks = 0
    n_events = 0
    n_wrong_target_events = 0
    health_samples = []
    benefit_cell_samples = []
    distractor_cell_samples = []
    done_cause = ""
    truncated = False

    with torch.no_grad():
        for _step_i in range(n_eval):
            n_goal_cells, n_distr_cells = _live_resource_composition(env)
            benefit_cell_samples.append(n_goal_cells)
            distractor_cell_samples.append(n_distr_cells)

            obs_dict, info, done, ticks = _step(obs_dict)
            last_info = info
            n_ticks += 1

            if ticks.get("e3_tick"):
                n_e3_ticks += 1
            else:
                n_latched_ticks += 1

            health_samples.append(float(info.get("health", 0.0) or 0.0))

            consumed_tag = int(info.get("sd049_consumed_type_tag_this_tick", 0))
            if consumed_tag > 0:
                n_events += 1
                if consumed_tag > GOAL_TAG:
                    n_wrong_target_events += 1

            if done:
                # DEFECT (1) FIX: record WHY, not just that the loop stopped.
                done_cause = str(info.get("done_cause", "") or "")
                truncated = True
                break

    # SD-094 shared accumulator -- its motivating incident is V3-EXQ-884's
    # contamination death, the exact mechanism under test here.
    ep_acc.record_from_info(last_info)
    # DEFECT (3) FIX: observe the STEPPING CLONE, never the P0 base agent.
    zg_acc.observe(agent)

    def _mean(xs, default=0.0):
        return round(statistics.fmean(xs), 6) if xs else default

    window_completeness = round(n_ticks / float(n_eval), 6) if n_eval else 0.0
    events_per_realised_tick = round(n_events / float(n_ticks), 8) if n_ticks else 0.0

    return {
        "arm": arm,
        "contam_gate": bool(spec["contam_gate"]),
        "clamp_health": bool(spec["clamp_health"]),
        "realised_contamination_spread": realised_contam_spread,
        "pinned_mode": PINNED_MODE,
        # --- defect (1): the window, recorded ---
        "n_budgeted_ticks": int(n_eval),
        "n_realised_ticks": int(n_ticks),
        "window_completeness": window_completeness,
        "truncated": bool(truncated),
        "truncated_in_warmup": bool(truncated_in_warmup),
        "done_cause": done_cause,
        # --- the DV, normalised by REALISED ticks ---
        "n_consumption_events": int(n_events),
        "n_wrong_target_events": int(n_wrong_target_events),
        "events_per_realised_tick": events_per_realised_tick,
        # --- health ---
        "final_health": round(health_samples[-1], 6) if health_samples else 0.0,
        "min_health": round(min(health_samples), 6) if health_samples else 0.0,
        "mean_health": _mean(health_samples),
        "n_health_samples": len(health_samples),
        "health_trajectory_sampled": [
            round(h, 4) for h in health_samples[::25]
        ],
        # --- defect (2): the nutritive density, recorded ---
        "n_benefit_bearing_resource_cells_mean": _mean(benefit_cell_samples),
        "n_distractor_resource_cells_mean": _mean(distractor_cell_samples),
        # --- cadence denominator (instrumentation, not a hypothesis leg) ---
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

    # P0 is trained on the STOCK (ungated) easy env so all three arms share one
    # baseline competence; the manipulation is confined to the exposure window.
    easy_env = _build_env(distractors_present=False, contam_gate=False)
    probe_env = _build_env(distractors_present=True, contam_gate=False)
    agent = _build_agent(probe_env.world_obs_dim).to(device)

    print(f"Seed {seed} Condition P0", flush=True)
    p0 = run_p0_warmup(
        agent, easy_env, device, budget=p0_budget, steps_per_episode=p0_steps
    )
    print(
        f"  [train] p0 seed={seed} ep {p0.n_episodes}/{p0_budget}"
        f" converged={p0.converged} aborted={p0.aborted} rv={p0.final_rv:.5f}",
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
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            print("verdict: FAIL", flush=True)
            arm_rows[arm] = {
                "arm": arm, "p0_aborted": True,
                "p0_abort_reason": p0.abort_reason,
                "n_budgeted_ticks": 0, "n_realised_ticks": 0,
                "window_completeness": 0.0, "truncated": True,
                "truncated_in_warmup": False, "done_cause": "p0_aborted",
                "n_consumption_events": 0, "n_wrong_target_events": 0,
                "events_per_realised_tick": 0.0,
                "final_health": 0.0, "min_health": 0.0, "mean_health": 0.0,
                "n_health_samples": 0, "health_trajectory_sampled": [],
                "n_benefit_bearing_resource_cells_mean": 0.0,
                "n_distractor_resource_cells_mean": 0.0,
                "n_e3_ticks": 0, "n_latched_ticks": 0, "e3_tick_fraction": 0.0,
                "goal_live_at_warmup_end": False,
                "contam_gate": bool(ARM_SPECS[arm]["contam_gate"]),
                "clamp_health": bool(ARM_SPECS[arm]["clamp_health"]),
                "realised_contamination_spread": -1.0,
                "pinned_mode": PINNED_MODE,
            }
        return {"seed": seed, "p0": p0_summary, "arms": arm_rows}

    for arm in ARMS:
        print(f"Seed {seed} Condition {arm}", flush=True)
        spec = ARM_SPECS[arm]
        env = _build_env(distractors_present=True, contam_gate=spec["contam_gate"])
        with arm_cell(
            seed,
            config_slice={
                "arm": arm,
                "contam_gate": bool(spec["contam_gate"]),
                "clamp_health": bool(spec["clamp_health"]),
                "health_clamp_floor": HEALTH_CLAMP_FLOOR,
                "pinned_mode": PINNED_MODE,
                "env": _env_kwargs(True, spec["contam_gate"]),
                "n_warmup_steps": N_WARMUP_STEPS,
                "n_eval_steps": N_EVAL_STEPS,
                "p0_budget": P0_BUDGET,
                "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
            },
            script_path=Path(__file__),
            config_slice_declared=True,
            # The P0 warmup is SHARED across all three arms of a given seed: each
            # clones the same p0-trained agent, so no cell is a pure function of
            # (seed, arm config) from a fresh RNG reset. Honestly ineligible rather
            # than falsely marked reusable -- the sanctioned shared-mutable-state
            # reason, not a "one-off" excuse.
            extra_ineligible_reasons=["shared_p0_warmup_across_arms"],
        ) as cell:
            row = _run_arm(agent, arm, env, device, smoke, zg_acc, ep_acc)
            row["p0_aborted"] = False
            cell.stamp(row)
        arm_rows[arm] = row
        print(
            f"verdict: {'PASS' if row['n_realised_ticks'] > 0 else 'FAIL'}"
            f" window={row['window_completeness']:.4f}"
            f" ticks={row['n_realised_ticks']}/{row['n_budgeted_ticks']}"
            f" cause={row['done_cause'] or 'none'}"
            f" events={row['n_consumption_events']}"
            f" rate={row['events_per_realised_tick']:.6f}"
            f" minhealth={row['min_health']:.3f}"
            f" contam={row['realised_contamination_spread']:.3f}",
            flush=True,
        )

    return {"seed": seed, "p0": p0_summary, "arms": arm_rows}


# ------------------------------------------------------------ precondition gate
def _precondition_specs(smoke: bool):
    min_ticks = SMOKE_MIN_REALISED_TICKS if smoke else MIN_REALISED_TICKS
    min_health = SMOKE_MIN_HEALTH_SAMPLES if smoke else MIN_HEALTH_SAMPLES
    return [
        PreconditionSpec(
            name="window_instrument_live",
            description=(
                "The cell stepped the exposure window at all. Below this floor nothing "
                "about window completeness or any rate normalised by it is measurable."
            ),
            control=(
                "n_realised_ticks counted directly in the eval loop; the SAME statistic "
                "the load-bearing C1 window-completeness criterion routes on (a count of "
                "realised ticks), not a proxy for it."
            ),
            threshold=float(min_ticks),
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="health_instrument_live",
            description=(
                "info['health'] was read on the window's ticks, so a health-driven "
                "termination is attributable rather than inferred."
            ),
            control="n_health_samples counted from info['health'] each tick.",
            threshold=float(min_health),
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="benefit_bearing_resources_present",
            description=(
                "Benefit-bearing (goal-tag) resource cells were live on the grid. This "
                "is 874b defect (2) made into a gate: consumption is impossible without "
                "them, and 874b let their count vary silently with its ruleset axis."
            ),
            control=(
                "Mean live goal-tag cells read off the env's own _resource_type_grid "
                "each tick -- measured, not inferred from nominal spawn weights, which a "
                "2026-08-19 probe showed the stochastic SD-049 allocator does not honour "
                "exactly."
            ),
            threshold=0.0,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="contamination_gate_applied",
            description=(
                "In the gated arms the env really bound contamination_spread to 0.0. "
                "A manipulation check: without it, a null in ARM_CONTAM_OFF would be "
                "uninterpretable (gate ineffective vs contamination irrelevant)."
            ),
            control=(
                "env.contamination_spread read off the constructed env. Pre-authoring "
                "probe 2026-08-19: stock 874b kwargs bind 0.5, "
                "+hazard_free_contamination_gate=True binds 0.0."
            ),
            threshold=CONTAM_GATE_CEILING,
            direction="upper",
            kind="readiness",
            # NOT meaningful for ARM_STOCK, whose whole purpose is to run at the stock
            # 0.5 spread 874b was exposed to. Asserting it there would make that arm
            # structurally un-passable and collapse the design to two arms -- the
            # V3-EXQ-785 regime-conditioning rule.
            applies_to=lambda ctx: bool(ctx.get("contam_gate")),
            applies_note=(
                "Scoped to the contamination-gated arms; ARM_STOCK is the "
                "un-gated reproduction arm by design."
            ),
        ),
    ]


def _arm_contexts():
    return {
        arm: {
            "arm_id": arm,
            "contam_gate": bool(ARM_SPECS[arm]["contam_gate"]),
            "clamp_health": bool(ARM_SPECS[arm]["clamp_health"]),
        }
        for arm in ARMS
    }


def _pool_arm(seed_results, arm) -> dict:
    rows = [r["arms"][arm] for r in seed_results if arm in r["arms"]]
    live = [r for r in rows if not r.get("p0_aborted")]
    total_ticks = sum(int(r["n_realised_ticks"]) for r in live)
    total_events = sum(int(r["n_consumption_events"]) for r in live)

    def _mean(key, default=0.0):
        vals = [float(r.get(key, 0.0)) for r in live]
        return round(statistics.fmean(vals), 6) if vals else default

    causes = {}
    for r in live:
        c = r.get("done_cause") or ("none" if not r.get("truncated") else "unknown")
        causes[c] = causes.get(c, 0) + 1

    return {
        "arm": arm,
        "n_cells": len(rows),
        "n_live_cells": len(live),
        "window_completeness_mean": _mean("window_completeness"),
        "window_completeness_per_seed": [
            float(r.get("window_completeness", 0.0)) for r in live
        ],
        "n_truncated_cells": sum(1 for r in live if r.get("truncated")),
        "done_causes": causes,
        "pooled_realised_ticks": int(total_ticks),
        "pooled_budgeted_ticks": sum(int(r["n_budgeted_ticks"]) for r in live),
        "pooled_consumption_events": int(total_events),
        "pooled_events_per_realised_tick": (
            round(total_events / float(total_ticks), 8) if total_ticks else 0.0
        ),
        "min_health_mean": _mean("min_health"),
        "final_health_mean": _mean("final_health"),
        "n_benefit_bearing_resource_cells_mean": _mean(
            "n_benefit_bearing_resource_cells_mean"
        ),
        "e3_tick_fraction_mean": _mean("e3_tick_fraction"),
        "realised_contamination_spread": (
            float(live[0]["realised_contamination_spread"]) if live else -1.0
        ),
        "n_realised_ticks_mean": _mean("n_realised_ticks"),
        "n_health_samples_mean": _mean("n_health_samples"),
        "goal_live_at_warmup_end_frac": (
            round(
                sum(1 for r in live if r.get("goal_live_at_warmup_end"))
                / float(len(live)),
                6,
            )
            if live else 0.0
        ),
    }


def build_manifest(seed_results, smoke: bool, started_at: float,
                   zg_acc, ep_acc) -> dict:
    pooled = {arm: _pool_arm(seed_results, arm) for arm in ARMS}
    specs = _precondition_specs(smoke)
    contexts = _arm_contexts()

    arm_gates = []
    for arm in ARMS:
        p = pooled[arm]
        measured = {
            "window_instrument_live": float(p["n_realised_ticks_mean"]),
            "health_instrument_live": float(p["n_health_samples_mean"]),
            "benefit_bearing_resources_present": float(
                p["n_benefit_bearing_resource_cells_mean"]
            ),
        }
        if contexts[arm]["contam_gate"]:
            measured["contamination_gate_applied"] = float(
                max(p["realised_contamination_spread"], 0.0)
            )
        arm_gates.append(
            evaluate_arm_gate(arm, contexts[arm], specs, measured)
        )
    aggregate = aggregate_arm_gates(arm_gates)

    # ---- criteria -----------------------------------------------------------
    stock = pooled[ARM_STOCK]
    gated = pooled[ARM_CONTAM_OFF]
    decoupled = pooled[ARM_HEALTH_DECOUPLED]

    window_lift = round(
        gated["window_completeness_mean"] - stock["window_completeness_mean"], 6
    )
    c1_pass = bool(window_lift >= WINDOW_COMPLETENESS_LIFT_MIN)

    stock_rate = stock["pooled_events_per_realised_tick"]
    best_decoupled_rate = max(
        gated["pooled_events_per_realised_tick"],
        decoupled["pooled_events_per_realised_tick"],
    )
    rate_lift_ratio = (
        round(best_decoupled_rate / stock_rate, 6) if stock_rate > 0 else None
    )
    c2_pass = bool(
        rate_lift_ratio is not None and rate_lift_ratio >= EVENT_RATE_LIFT_MIN
    )

    total_events_all_arms = sum(
        pooled[a]["pooled_consumption_events"] for a in ARMS
    )
    # C2 compares RATES. With no events anywhere there is no rate to compare, so C2
    # discriminates nothing -- that is a degenerate criterion, not a null result.
    c2_non_degenerate = bool(total_events_all_arms > 0)
    # C1 NON-DEGENERACY -- found by the Step 2.5b adversarial design audit (2026-08-19).
    # C1 reads a LIFT in window_completeness from gating contamination. If ARM_STOCK
    # never truncated in the first place, that lift is ~0 and C1 reads FALSE -- which
    # ALIASES "self-contamination was not what truncated the window" onto "there was no
    # truncation to explain". Those are opposite findings. C1 can only discriminate when
    # the reproduction arm actually reproduced 874b's truncation (7 of 12 cells), so its
    # non-degeneracy is keyed to that.
    c1_non_degenerate = bool(stock["n_truncated_cells"] > 0)

    criteria_by_arm = {
        # C1 is owned ONLY by the two arms in which window_completeness can move.
        # In ARM_HEALTH_DECOUPLED it is forced to ~1.0 by the manipulation and is a
        # manipulation check, not a measurement (V3-EXQ-785 disposition (a)).
        ARM_STOCK: ["C1_window_completeness_lifts_when_contamination_gated"],
        ARM_CONTAM_OFF: ["C1_window_completeness_lifts_when_contamination_gated"],
        ARM_HEALTH_DECOUPLED: ["C2_event_rate_lifts_when_window_decoupled"],
    }
    criteria_nd = arm_criteria_non_degenerate(
        criteria_by_arm,
        aggregate,
        extra={
            "C1_window_completeness_lifts_when_contamination_gated": c1_non_degenerate,
            "C2_event_rate_lifts_when_window_decoupled": c2_non_degenerate,
        },
    )

    # ---- self-route ---------------------------------------------------------
    if not aggregate["non_degenerate"]:
        label = "substrate_not_ready_requeue"
        interpretation_note = (
            "No arm cleared its readiness gate; nothing here is a verdict on MECH-467 "
            "or on the energy hypothesis."
        )
    elif not c1_non_degenerate:
        label = "truncation_not_reproduced_c1_undiscriminating"
        interpretation_note = (
            "ARM_STOCK -- the reproduction arm, run on 874b's env kwargs verbatim -- did "
            "not truncate in any cell, so there was no window collapse to explain and C1 "
            "cannot discriminate. This is NOT the H-energy null: it means 874b's "
            "truncation did not reproduce here, which is itself a finding about "
            "874b (its 7-of-12 health_depleted cells) and must be read before anything "
            "else in this manifest. Compare per_arm_pooled done_causes against the "
            "autopsy's table. Step 2.5b adversarial design audit, 2026-08-19."
        )
    elif c1_pass and c2_pass:
        label = "energy_window_self_contamination_confirmed"
        interpretation_note = (
            "Gating self-contamination both restores the exposure window AND lifts the "
            "consumption rate per realised tick. 874b's window collapse was "
            "self-inflicted contamination, not consumption starvation, and the "
            "denominator is recoverable by an env flag."
        )
    elif c1_pass and not c2_pass:
        label = "window_restored_rate_unchanged"
        interpretation_note = (
            "The window is restored by gating contamination -- so the truncation WAS "
            "self-poisoning -- but the event rate per realised tick does not lift. This "
            "is the autopsy's declared H-energy null in its informative form: "
            "starvation only shortened the window, it did not suppress eating. The "
            "residual rate problem is not an energy problem."
        )
    elif (not c1_pass) and c2_pass:
        label = "rate_lift_without_window_lift"
        interpretation_note = (
            "The rate lifts without the window lifting -- contamination was not what "
            "truncated the window, but something in the decoupled condition still "
            "raises the rate. Read the per-arm done_causes before interpreting."
        )
    else:
        label = "energy_not_the_constraint"
        interpretation_note = (
            "Neither the window nor the rate responds to decoupling survival from the "
            "measurement window. H-energy is eliminated: the leg-(c) denominator "
            "problem is not an energy problem, which leaves the rate constraint with "
            "the MECH-439 F-dominance conversion ceiling the 2026-08-18 navigation-"
            "immobility scoping spike already owns."
        )

    outcome = "PASS" if (aggregate["non_degenerate"] and (c1_pass or c2_pass)) else "FAIL"

    manifest = {
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
            "leg": "H-energy",
            "axis_family": "environment",
            "sibling_legs": ["V3-EXQ-941 (H-denominator, measurement)"],
            "legs_not_queued": {
                "H-commitment": (
                    "answered at substrate level by "
                    "navigation_immobility_scoping_2026-08-18.md -- MECH-439 "
                    "F-dominance conversion ceiling, ceiling_decision exhausted"
                ),
                "H-cadence": (
                    "answered at substrate level (E3 heartbeat hold-and-repeat, "
                    "e3_steps_per_tick); deeper cause owned by the ACTIVE "
                    "hypothesis-space line e3_fdominance_causal_discrimination"
                ),
            },
        },
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "per_arm_pooled": pooled,
        "per_seed_results": seed_results,
        "criteria": [
            {
                "name": "C1_window_completeness_lifts_when_contamination_gated",
                "load_bearing": True,
                "passed": bool(c1_pass),
                "non_degenerate": bool(c1_non_degenerate),
                "non_degeneracy_basis": (
                    "ARM_STOCK truncated in at least one cell, so there was a window "
                    "collapse for the gate to explain."
                ),
                "measured_lift": window_lift,
                "threshold": WINDOW_COMPLETENESS_LIFT_MIN,
                "owned_by_arms": [ARM_STOCK, ARM_CONTAM_OFF],
                "note": (
                    "Excludes ARM_HEALTH_DECOUPLED, where window_completeness is forced "
                    "to ~1.0 by the manipulation and is a manipulation check rather "
                    "than a measurement."
                ),
            },
            {
                "name": "C2_event_rate_lifts_when_window_decoupled",
                "load_bearing": False,
                "passed": bool(c2_pass),
                "measured_ratio": rate_lift_ratio,
                "threshold": EVENT_RATE_LIFT_MIN,
                "owned_by_arms": [ARM_HEALTH_DECOUPLED],
                "note": (
                    "Degenerate when no arm produced any consumption event: a rate "
                    "comparison with no events discriminates nothing."
                ),
            },
        ],
        "combination_rule": (
            "PASS iff the aggregate gate is non-degenerate (ANY arm green) AND at least "
            "one of C1 / C2 fires. C1 is additionally only READABLE when ARM_STOCK "
            "actually truncated (see its non_degenerate field); a non-reproducing stock "
            "arm routes to truncation_not_reproduced_c1_undiscriminating rather than to "
            "a null. C1 and C2 are NOT ANDed: they answer different "
            "halves of the question (did the window survive; did the rate move), and "
            "the autopsy's declared null is precisely C1-true / C2-false."
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
            "dv": "events_per_realised_tick and window_completeness",
            "symmetry_group": "permutation of ticks (both DVs are set-aggregates)",
            "manipulation_invariant_under_it": False,
            "per_arm": {
                arm: (
                    "NOT invariant: contamination gating / health clamping changes "
                    "which ticks exist and whether the terminal condition fires. It is "
                    "not a broadcast constant across candidates, not a monotone "
                    "rescaling, and not a permutation of interchangeable units."
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
            "note": (
                "V3 diagnostic. The manipulation REDUCES a harm source (self-inflicted "
                "contamination) rather than introducing one."
            ),
        },
    }
    return manifest


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
        "health_clamp_floor": HEALTH_CLAMP_FLOOR,
        "pinned_mode": PINNED_MODE,
        "arms": {a: dict(ARM_SPECS[a]) for a in ARMS},
        "env_kwargs_stock": _env_kwargs(True, False),
        "env_kwargs_gated": _env_kwargs(True, True),
        "thresholds": {
            "window_completeness_lift_min": WINDOW_COMPLETENESS_LIFT_MIN,
            "event_rate_lift_min": EVENT_RATE_LIFT_MIN,
            "min_realised_ticks": MIN_REALISED_TICKS,
            "min_health_samples": MIN_HEALTH_SAMPLES,
            "contam_gate_ceiling": CONTAM_GATE_CEILING,
        },
        "seeds": list(SEEDS),
    }


def main(smoke: bool):
    started_at = time.perf_counter()
    device = torch.device("cpu")

    # Design-time refusal: no precondition may be structurally unsatisfiable from the
    # PRE-REGISTERED config (the V3-EXQ-785 check, run before compute is spent).
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

    manifest = build_manifest(seed_results, smoke, started_at, zg_acc, ep_acc)
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
