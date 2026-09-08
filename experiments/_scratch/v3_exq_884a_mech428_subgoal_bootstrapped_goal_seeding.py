#!/opt/local/bin/python3
"""
V3-EXQ-884a -- MECH-428 Subgoal-Bootstrapped Goal Seeding: 3-arm
NO-SUBGOAL / SUBGOAL-BOOTSTRAP / FORCED-SEED comparison.
SUPERSEDES V3-EXQ-884 (same scientific question; environment-configuration fix).

Claim: MECH-428 (subgoal_bootstrapped_goal_seeding, formation-direction sibling
of MECH-427). Proposal: EXP-0390.

WHY 884a EXISTS (the ONLY substantive change from 884)
---------------------------------------------------------------------------
The confirmed autopsy failure_autopsy_V3-EXQ-884_2026-08-03 found V3-EXQ-884
non-contributory: all three seeds terminated 77-95% short of the configured
400-step budget (32 / 19 / 90 steps) via agent_health <= 0, because of two
CausalGridWorld defects, both since fixed under SD-094 but both OPT-IN and
therefore OFF in the 884 driver:

  (a) waypoint arrival was detected by the destination cell's GRID TYPE, which
      the agent's own position marker destroys on first transit -- so later
      legitimate arrivals registered no transition_type at all;
  (b) contamination_spread defaults to 0.5 and applies on every entered cell
      regardless of num_hazards, so this "hazard-free" (num_hazards=0) probe
      contaminated its own trail and killed its own agent.

884 therefore starved the accumulation-vs-decay dynamics the claim's own
non-degeneracy clause requires; G0/G1 passed vacuously. A verbatim re-queue
would reproduce the identical failure, because neither flag is on by default.

884a sets BOTH SD-094 flags in _build_env (subgoal_arrival_position_check=True,
hazard_free_contamination_gate=True) and changes NOTHING else about the
scientific design -- same arms, same seeds, same pre-registered thresholds,
same DV, same C1/C2 criteria. Measured on this substrate before queuing
(2026-09-08, seeds 42/43/44):

  flags OFF: 32 / 19 / 90 steps, done_cause=health_depleted, 0 / 0 / 3
             sequence_complete events  (reproduces the 884 failure exactly)
  flags ON : 400 / 400 / 400 steps, done=False, health 1.000,
             19 / 19 / 21 sequence_complete events, 40 / 39 / 44 waypoints

Two recording additions the same autopsy required (its
recommended_substrate_queue_entry names them as a recording gap): every cell
now records the REAL episode length and the env's own done_cause, and a new
readiness gate G2 makes that machine-checkable rather than something a later
reader has to re-run the experiment to discover.

SD-094 gate assertion: _build_env asserts the contamination gate actually
FIRED (env._contamination_gate_applied) rather than inferring it from a zero,
so a future default change cannot silently un-fix this run.

DESIGN CHOICE (documented per /queue-experiment Step-2.5, mandatory): this
script does NOT use experiments/scaffolded_sd054_onboarding.py (the heavier
curriculum the registered proposal references) or any LEARNED navigation
policy. Reasons, stated explicitly:

  1. The scaffolded curriculum's P2 benefit-contact leg requires the agent to
     autonomously forage and commit to a multi-step approach to reach real
     resources. That is a sustained-multi-step-commitment behavioural DV, and
     the basal-ganglia / action-commitment layer is still under active
     construction (the F-dominance conversion ceiling: ARC-107/MECH-448/
     MECH-449, commit/release-duration falsifiers 460j/485i/654h pending --
     see memory feedback_dont_queue_commitment_dependent_behavioural). A FAIL
     on that substrate would most likely re-derive the known conversion
     ceiling, not a verdict on MECH-428.
  2. MECH-428's own text states MECH-427 and MECH-428 are "the SAME call
     against different starting states of _z_goal_parent" (goal.py
     credit_subgoal_attainment docstring) -- the mechanism under test is the
     EMA-pull/decay ARITHMETIC of repeated credit events, not navigation
     competence. A scripted, deterministic trajectory isolates exactly that,
     exactly as V3-EXQ-883 (MECH-427) did for the maintenance direction.
  3. The claim's own "GAP-2 seeding-sparse condition" is reproduced NOT by
     simulating ecological foraging failure (agent incompetence) but by the
     substrate fact that SD-092's _z_goal_parent has EXACTLY ONE write path
     (GoalState.credit_subgoal_attainment) -- there is no other mechanism
     that can seed it "directly". ARM NO-SUBGOAL (below) exercises that fact
     directly rather than re-deriving it through a foraging failure.

This is a MATERIALLY DIFFERENT design from V3-EXQ-883, not a copy: 883 tests
ONE attainment event vs none, in a short single-leg episode (already-clear
maintenance case). 884 runs a LONG, multi-waypoint-sequence episode (env
subgoal_mode loops back to waypoint 0 after each full sequence -- see
causal_grid_world.py step(), "sequence_complete" branch) so that MANY discrete
credit events occur, spaced ~one every several steps, while GoalState.update's
parent_goal_decay (default 0.005/tick) is ticking every step in between. The
scientific question is therefore about ACCUMULATION-VS-DECAY dynamics under
a REALISTIC intermittent event rate -- whether bottom-up bootstrapping nets
out to a materially structured parent attractor, not whether a single event
can move the needle at all (883's question).

DV-SYMMETRY DECLARATION (Step 3.5, mandatory per-arm statement): ARM
NO-SUBGOAL's parent_goal_norm is arithmetically PINNED near zero by
construction -- credit_subgoal_attainment is the substrate's only write path
to _z_goal_parent, and this arm never calls it (see reason 3 above), so its
value is a decaying zero-init, not a measurement of anything. This arm is
NOT itself a discriminating comparison and is not scored as one; it exists
only to anchor the C1 fraction-of-ceiling calculation and to give the
registered proposal's "NO-SUBGOAL baseline" a concrete value. The load-bearing
test is C1 (below): whether ARM SUBGOAL-BOOTSTRAP's steady-state parent norm
clears a PRE-REGISTERED FRACTION of ARM FORCED-SEED's ceiling -- that fraction
is NOT structurally guaranteed by any arm's construction (it depends on the
actual alpha/decay/event-rate dynamics), so it is the genuine test.

SCOPE (behaviour, not just representation): per reason 1 above, this script
does not exercise "goal-directed behaviour" driven by a learned policy reading
the seeded parent -- the claim's own wording ("produces goal-directed
behaviour the no-subgoal control does not") is a claim for a LATER,
commitment-substrate-dependent follow-on, explicitly out of scope here. This
script establishes the representational precondition (a structured parent
CAN be bootstrapped bottom-up) that any later behavioural test would need.

Arms (3 arms x 3 seeds; each cell is one long scripted-trajectory episode)
---------------------------------------------------------------------------
ARM_NO_SUBGOAL       Scripted greedy walk through the waypoint sequence occurs
                     (so the ENV mechanics are IDENTICAL across arms), but
                     agent.notify_subgoal_attainment(...) is NEVER called.
                     Structural floor (see DV-symmetry declaration above).
ARM_SUBGOAL_BOOTSTRAP Identical scripted walk; agent.notify_subgoal_attainment
                     (info["transition_type"]) IS called every step (no-op
                     unless "waypoint"/"sequence_complete"), with
                     use_hierarchical_goal_credit=True and default credit=1.0.
                     This is the claim's own bootstrap case: repeated
                     credit_subgoal_attainment calls against a
                     near-zero-starting _z_goal_parent.
ARM_FORCED_SEED      Same env/walk; agent.notify_subgoal_attainment is called
                     EVERY step with an artificially large credit
                     (FORCED_CREDIT=20.0, so a=min(1, parent_goal_alpha*credit)
                     clamps to 1.0 -- a full-replacement pull every step,
                     independent of whether a real waypoint/sequence-complete
                     event occurred this step). This is the "626b-style"
                     positive control adapted to the PARENT level (626b itself
                     forces the CHILD-level z_goal via benefit_exposure, which
                     does not touch _z_goal_parent at all -- see goal.py
                     GoalState.update(); the parent has no equivalent forced-
                     benefit-exposure argument, so the only way to force it is
                     through its own sole write path, credit_subgoal_attainment,
                     at maximal, event-rate-independent strength). Proves the
                     wiring CAN sustain a structured parent when driven hard,
                     decoupled from the SUBGOAL_BOOTSTRAP arm's realistic,
                     intermittent event cadence.

child_representation is left at its default in all crediting arms (the
current latent's z_world at the moment of the call -- see
REEAgent.notify_subgoal_attainment), sensed on the POST-step observation
before crediting (same ordering rationale as V3-EXQ-883).

Z_GOAL (CHILD-LEVEL) NOTE: agent.update_z_goal(benefit_exposure=0.0,
drive_level=0.0) is called every step in ALL THREE arms. This is not optional
bookkeeping: GoalState.update() is where BOTH the child _z_goal decay AND the
parent parent_goal_decay tick fire (see goal.py -- the parent decay branch is
inside update(), gated only on use_hierarchical_goal_credit, not on any
benefit signal). Without this call every step, the parent would never decay
in any arm and the whole accumulation-vs-decay comparison would be vacuous.
benefit_exposure=0.0 means the child _z_goal itself never seeds -- by design;
this experiment is about the PARENT attractor exclusively, exactly as 883 was.

PRE-REGISTERED GATES
--------------------
G2 non-degeneracy (readiness, NEW in 884a -- the 884 autopsy's own finding
   made machine-checkable): EVERY cell (all arms, all seeds) must run its
   full configured step budget without the env terminating the episode
   (done False, or done_cause == "step_limit" at exactly the budget). This
   is the gate that would have caught 884: a cell that dies at 19/400 steps
   has not exercised the accumulation-vs-decay dynamics at all, so its
   C1/C2 readings are starved rather than negative. Failure ->
   substrate_not_ready_requeue / non_contributory, NEVER "weakens".
G0 non-degeneracy (readiness, claim's own precondition (a)): ARM_SUBGOAL_
   BOOTSTRAP's n_subgoal_credits > 0 for EVERY seed (structurally near-
   guaranteed by the scripted, looping walk; failure means a harness/env
   defect, not a scientific finding -> substrate_not_ready_requeue).
G1 non-degeneracy (readiness, claim's own precondition (b)): ARM_FORCED_SEED's
   steady-state parent_goal_norm (median over the last WINDOW steps) clears
   FORCED_STRUCTURED_FLOOR on >= 2/3 seeds -- "the FORCED-SEED positive
   control must itself show a structured z_goal in the SAME harness". If this
   fails, the substrate (not the bootstrap hypothesis) is what is not ready.
C1 (the registered acceptance check, load-bearing): for each seed, compute
   lift_fraction = (steady_state[SUBGOAL_BOOTSTRAP] - steady_state[NO_SUBGOAL])
                   / (steady_state[FORCED_SEED] - steady_state[NO_SUBGOAL])
   PASS if lift_fraction >= C1_FRACTION_FLOOR on >= 2/3 seeds. This is the
   genuine test (see DV-symmetry declaration): NOT structurally guaranteed.
C2 (measurability floor): ARM_SUBGOAL_BOOTSTRAP's steady-state parent_goal_norm
   clears an absolute PARENT_NORM_ABS_FLOOR on >= 2/3 seeds -- guards against
   a technically-positive-fraction-but-numerically-negligible reading (both
   numerator and denominator near the float noise floor) counting as a lift.

PASS (supports MECH-428) = G2 AND G0 AND G1 AND C1 AND C2.
G2 fails -> non_contributory / substrate_not_ready_requeue (the 884 failure
  mode: the episode did not run, so nothing about accumulation-vs-decay was
  measured).
G0 or G1 fails -> non_contributory / substrate_not_ready_requeue (the GAP-2-
  analog trap: if even the forced positive control cannot produce a structured
  parent, or attainment itself never fires, nothing here speaks to bootstrap-
  ing).
G0 AND G1 pass, C1/C2 fail -> weakens (attainment occurred and the wiring CAN
  sustain a structured parent under forcing, but repeated realistic-cadence
  credit did not accumulate a materially useful fraction of that ceiling --
  genuine evidence that bottom-up bootstrapping is too weak against decay at
  this event rate/alpha).

Biological basis (unchanged from the claim): Bandura & Schunk (1981) --
proximal subgoals CREATED intrinsic interest / goal pursuit in initially-
uninterested learners (a FORMATION result, distinct from MECH-426/427's
maintenance framing).

SLEEP DRIVER: N/A (no sleep loop; scripted-trajectory representational probe).

ethics_preflight:
  involves_negative_valence: false        # no hazards/resources in this config; harm stream unused
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from ree_core.environment.causal_grid_world import CausalGridWorld

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

# --------------------------------------------------------------------- #
# Experiment metadata
# --------------------------------------------------------------------- #
EXPERIMENT_TYPE = "v3_exq_884a_mech428_subgoal_bootstrapped_goal_seeding"
QUEUE_ID = "V3-EXQ-884a"
SUPERSEDES = "V3-EXQ-884"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-428"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
ARMS = ["NO_SUBGOAL", "SUBGOAL_BOOTSTRAP", "FORCED_SEED"]
STAY_ACTION = 4  # (dx, dy) = (0, 0) in CausalGridWorld.ACTIONS -- a true no-op

# --- Env config (shared across all arms -- same seed => identical waypoint
#     placement + agent start; no hazards/resources so the ONLY env mechanic
#     exercised is subgoal_mode's looping waypoint sequence). -----------
GRID_SIZE = 12
NUM_WAYPOINTS = 3        # env default; a full sequence = 3 legs, then loops
N_STEPS = 400             # long enough for several full sequence loops
N_STEPS_DRY = 80
MEASUREMENT_WINDOW = 60   # last-N-steps window for the steady-state median

WORLD_DIM = 32
PARENT_GOAL_ALPHA = 0.05        # GoalConfig default (a = min(1, alpha*credit))
PARENT_GOAL_DECAY = 0.005       # GoalConfig default
FORCED_CREDIT = 20.0             # min(1, 0.05*20.0) = 1.0 -- full replacement every step

# Pre-registered thresholds. FORCED_STRUCTURED_FLOOR / PARENT_NORM_ABS_FLOOR
# are calibrated against this harness's own encoder-output norm scale (this
# is the dry-run-observed z_world norm magnitude for an UNTRAINED encoder on
# this env/config, NOT a threshold derived from the scored run's own
# statistics -- both arms' construction is fixed before any seed is scored).
FORCED_STRUCTURED_FLOOR = 0.05
PARENT_NORM_ABS_FLOOR = 0.01
C1_FRACTION_FLOOR = 0.3   # SUBGOAL_BOOTSTRAP must recover >= 30% of the
                          # FORCED_SEED - NO_SUBGOAL ceiling range

MIN_SEEDS_PASS = 2  # of 3 -- ">= 2/3 seeds"


# --------------------------------------------------------------------- #
# Env + agent builders
# --------------------------------------------------------------------- #

def _build_env(seed: int) -> CausalGridWorld:
    env = CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=0,
        num_resources=0,
        subgoal_mode=True,
        num_waypoints=NUM_WAYPOINTS,
        seed=seed,
        # --- SD-094, the ONLY change from V3-EXQ-884. Both flags default OFF,
        #     which is exactly why a verbatim re-queue would reproduce 884's
        #     failure. See the WHY 884a EXISTS block in the module docstring.
        subgoal_arrival_position_check=True,
        hazard_free_contamination_gate=True,
    )
    # Assert the gate actually FIRED rather than inferring it from a zero that
    # a future default change might supply for an unrelated reason. If this
    # ever trips, the run must not proceed -- it would silently be 884 again.
    assert env.subgoal_arrival_position_check is True, (
        "SD-094 (a) arrival-by-position check did not take effect"
    )
    assert getattr(env, "_contamination_gate_applied", False) is True, (
        "SD-094 (b) hazard-free contamination gate did not fire "
        f"(contamination_spread={env.contamination_spread})"
    )
    assert float(env.contamination_spread) == 0.0, (
        f"SD-094 (b) contamination_spread is {env.contamination_spread}, expected 0.0"
    )
    return env


def _build_agent(env: CausalGridWorld) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        world_dim=WORLD_DIM,
        z_goal_enabled=True,
    )
    agent = REEAgent(cfg)
    agent.goal_state.config.use_hierarchical_goal_credit = True
    return agent


def _scripted_action(env: CausalGridWorld) -> int:
    """Ground-truth greedy walk toward the currently-targeted waypoint.
    Reads the env's own state directly -- no agent perception involved --
    so the trajectory is fully decoupled from the agent's own action-
    selection competence (see the DESIGN CHOICE note in the module
    docstring re. the basal-ganglia commitment confound)."""
    sub = env.get_subgoal_state()
    idx = sub["next_waypoint_idx"]
    if not env.waypoints or idx >= len(env.waypoints):
        return STAY_ACTION
    wx, wy = env.waypoints[idx]
    ax, ay = env.get_agent_position()
    if ax != wx:
        return 1 if wx > ax else 0
    if ay != wy:
        return 3 if wy > ay else 2
    return STAY_ACTION  # already on the target cell (should not occur mid-step)


# --------------------------------------------------------------------- #
# One seed x arm cell
# --------------------------------------------------------------------- #

def _run_cell(arm: str, seed: int, zg: ZGoalStreamAccumulator, dry_run: bool) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    n_steps = N_STEPS_DRY if dry_run else N_STEPS

    env = _build_env(seed)
    agent = _build_agent(env)

    obs_flat, _obs_dict = env.reset()
    agent.act(obs_flat)  # populate _current_latent before the first possible credit call

    n_subgoal_credits = 0
    n_waypoint_events = 0
    n_sequence_complete_events = 0
    parent_norm_trace: List[float] = []
    # SD-094 recording gap (the 884 autopsy's own requirement): the REAL
    # episode length and the env's own reason for ending it. 884 recorded
    # only the CONFIGURED budget, which is why "ran 400 steps" and "died on
    # step 19 of 400" were indistinguishable in its manifest.
    steps_taken = 0
    done = False
    done_cause = ""

    for _step in range(n_steps):
        action_idx = _scripted_action(env)
        obs_flat, _harm_signal, done, info, _obs_dict = env.step(action_idx)
        steps_taken += 1
        done_cause = str(info.get("done_cause", "") or "")
        # Sense the POST-step observation before crediting, so the default
        # child_representation (current latent's z_world) reflects the
        # agent's arrival state at the moment of attainment.
        agent.act(obs_flat)
        agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)

        ttype = info.get("transition_type", "none")
        if ttype == "waypoint":
            n_waypoint_events += 1
        elif ttype == "sequence_complete":
            n_sequence_complete_events += 1

        if arm == "NO_SUBGOAL":
            credit_result = {}  # never call notify_subgoal_attainment (see DV-symmetry declaration)
        elif arm == "SUBGOAL_BOOTSTRAP":
            credit_result = agent.notify_subgoal_attainment(ttype)  # default credit=1.0
        else:  # FORCED_SEED
            credit_result = agent.notify_subgoal_attainment(
                "sequence_complete", credit=FORCED_CREDIT
            )  # forced literal transition_type every step -- event-rate-independent

        if credit_result:
            n_subgoal_credits = credit_result["n_subgoal_credits"]

        parent_norm_trace.append(float(agent.goal_state.parent_goal_norm()))

        if done:
            break

    window = parent_norm_trace[-MEASUREMENT_WINDOW:] if parent_norm_trace else [0.0]
    steady_state_norm = float(sorted(window)[len(window) // 2])  # median, no numpy dependency
    final_parent_norm = float(agent.goal_state.parent_goal_norm())
    zg.observe(agent)

    # G2 (884a): the cell ran its full configured budget. A cell whose episode
    # the env terminated early has not exercised the accumulation-vs-decay
    # dynamics, so its C1/C2 readings are starved, not negative.
    budget_frac = float(steps_taken) / float(n_steps) if n_steps else 0.0
    budget_reached = bool(steps_taken >= n_steps)

    print(
        f"  [eval] {arm} seed={seed} n_subgoal_credits={n_subgoal_credits} "
        f"n_waypoint_events={n_waypoint_events} "
        f"n_sequence_complete_events={n_sequence_complete_events} "
        f"steady_state_parent_norm={steady_state_norm:.4f} "
        f"steps_actual={steps_taken}/{n_steps} "
        f"done={done} done_cause='{done_cause or 'none'}' "
        f"agent_health={float(env.agent_health):.3f}",
        flush=True,
    )
    print(f"  [train] {arm} seed={seed} ep 1/1", flush=True)
    print("verdict: PASS", flush=True)  # cell ran to completion; scientific verdict is aggregate-level

    return {
        "seed": seed,
        "arm": arm,
        "n_subgoal_credits": n_subgoal_credits,
        "n_waypoint_events": n_waypoint_events,
        "n_sequence_complete_events": n_sequence_complete_events,
        "parent_goal_norm_final": final_parent_norm,
        "parent_goal_norm_steady_state": steady_state_norm,
        "parent_goal_norm_trace": parent_norm_trace,
        "n_steps": n_steps,                     # configured budget (as in 884)
        # --- SD-094 recording gap, new in 884a ---
        "n_steps_configured": n_steps,
        "n_steps_actual": steps_taken,
        "episode_budget_frac": budget_frac,
        "episode_budget_reached": budget_reached,
        "episode_done": bool(done),
        "episode_done_cause": done_cause,
        "agent_health_final": float(env.agent_health),
    }


# --------------------------------------------------------------------- #
# Aggregate + acceptance criteria
# --------------------------------------------------------------------- #

def run(dry_run: bool = False) -> tuple:
    print(f"\n[{QUEUE_ID}] MECH-428 Subgoal-Bootstrapped Goal Seeding "
          f"(NO_SUBGOAL / SUBGOAL_BOOTSTRAP / FORCED_SEED)", flush=True)

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict] = []
    per_seed: Dict[str, Dict[int, Dict]] = {a: {} for a in ARMS}

    for seed in SEEDS:
        for arm in ARMS:
            config_slice = {
                "arm": arm, "seed": seed, "dry_run": dry_run,
                "grid_size": GRID_SIZE, "num_waypoints": NUM_WAYPOINTS,
                "n_steps": N_STEPS_DRY if dry_run else N_STEPS,
                "world_dim": WORLD_DIM, "forced_credit": FORCED_CREDIT,
                # SD-094 flags are part of the cell's identity: a cell run with
                # them off is the 884 configuration, not this one, and must
                # never fingerprint-match a cell run with them on.
                "subgoal_arrival_position_check": True,
                "hazard_free_contamination_gate": True,
            }
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                row = _run_cell(arm, seed, zg, dry_run=dry_run)
                cell.stamp(row)
            arm_results.append(row)
            per_seed[arm][seed] = row

    # --- G2 (884a): every cell, every arm, ran its full configured budget. ---
    g2_failures = [
        f"{r['arm']}/seed{r['seed']}: {r['n_steps_actual']}/{r['n_steps_configured']} "
        f"(done_cause={r['episode_done_cause'] or 'none'}, "
        f"health={r['agent_health_final']:.3f})"
        for r in arm_results
        if not r["episode_budget_reached"]
    ]
    g2_pass = not g2_failures
    g2_min_budget_frac = min(float(r["episode_budget_frac"]) for r in arm_results)

    g0_per_seed, g1_per_seed, c1_per_seed, c2_per_seed = [], [], [], []
    lift_fractions: List[float] = []
    ceiling_ranges: List[float] = []
    for seed in SEEDS:
        no_sub = per_seed["NO_SUBGOAL"][seed]
        boot = per_seed["SUBGOAL_BOOTSTRAP"][seed]
        forced = per_seed["FORCED_SEED"][seed]

        g0_per_seed.append(boot["n_subgoal_credits"] > 0)
        g1_per_seed.append(forced["parent_goal_norm_steady_state"] >= FORCED_STRUCTURED_FLOOR)

        ceiling_range = forced["parent_goal_norm_steady_state"] - no_sub["parent_goal_norm_steady_state"]
        ceiling_ranges.append(float(ceiling_range))
        # NOTE (884a, dv_headroom reasoning): the degenerate branch below is
        # unreachable whenever G1 passes. G1 requires the FORCED_SEED arm's
        # steady state to clear FORCED_STRUCTURED_FLOOR (0.05) while
        # NO_SUBGOAL's is structurally pinned at ~0 (it never calls the sole
        # write path), so a G1-passing seed always has ceiling_range >= ~0.05.
        # A degenerate ceiling therefore routes through G1's
        # non_contributory / substrate_not_ready branch, NOT through a
        # lift_fraction of 0.0 read as "weakens". Kept as a defensive floor.
        if ceiling_range > 1e-12:
            lift_fraction = (
                boot["parent_goal_norm_steady_state"] - no_sub["parent_goal_norm_steady_state"]
            ) / ceiling_range
        else:
            lift_fraction = 0.0  # degenerate ceiling range -- cannot support a fraction claim
        lift_fractions.append(lift_fraction)
        c1_per_seed.append(lift_fraction >= C1_FRACTION_FLOOR)
        c2_per_seed.append(boot["parent_goal_norm_steady_state"] >= PARENT_NORM_ABS_FLOOR)

    threshold = MIN_SEEDS_PASS / len(SEEDS)
    g0_pass = all(g0_per_seed)  # readiness: EVERY seed, not a fraction
    g0_frac = sum(1 for g in g0_per_seed if g) / len(g0_per_seed)
    g1_frac = sum(1 for g in g1_per_seed if g) / len(g1_per_seed)
    g1_pass = g1_frac >= threshold
    c1_frac = sum(1 for c in c1_per_seed if c) / len(c1_per_seed)
    c2_frac = sum(1 for c in c2_per_seed if c) / len(c2_per_seed)
    c1_pass = c1_frac >= threshold
    c2_pass = c2_frac >= threshold

    non_degenerate = True
    degeneracy_reason = None

    if not g2_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G2 readiness gate failed: at least one cell's episode was "
            "terminated by the environment before its configured step budget, "
            "so the accumulation-vs-decay dynamics this experiment measures "
            "were never exercised. This is the EXACT V3-EXQ-884 failure mode "
            "(confirmed failure_autopsy_V3-EXQ-884_2026-08-03: 32/19/90 of 400 "
            "steps via agent_health<=0). Starved cells: "
            + "; ".join(g2_failures)
            + ". Not evidence about MECH-428; diagnose the environment "
            "configuration (SD-094 flags, contamination, step cap) and "
            "re-queue under a new letter."
        )
        interpretation = (
            "Readiness (G2) failed -- one or more episodes ended early, so "
            "the intended long, multi-waypoint accumulation regime did not "
            "run. Not evidence against MECH-428; substrate/config not ready."
        )
    elif not g0_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G0 readiness gate failed: ARM_SUBGOAL_BOOTSTRAP's scripted "
            "looping walk did not produce a subgoal-attainment credit on "
            "every seed. This is a harness/environment defect (the scripted "
            "walk is deterministic and should always reach the waypoint "
            "sequence repeatedly within N_STEPS), not evidence about "
            "MECH-428."
        )
        interpretation = (
            "Readiness (G0) failed -- the scripted trajectory did not "
            "reliably trigger subgoal-attainment events, so the bootstrap "
            "mechanism was never exercised. Not evidence against MECH-428; "
            "re-queue after fixing the harness (check N_STEPS vs grid size, "
            "or the greedy-walk action mapping)."
        )
    elif not g1_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G1 readiness gate failed: ARM_FORCED_SEED (maximal, event-rate-"
            "independent credit every step) did not itself produce a "
            "steady-state parent_goal_norm clearing FORCED_STRUCTURED_FLOOR "
            "on >= 2/3 seeds. This is the GAP-2-analog trap: if even the "
            "forced positive control cannot sustain a structured parent in "
            "this harness, the substrate (not the bootstrap hypothesis) is "
            "what is not ready -- re-queue after diagnosing the parent-level "
            "wiring, do not read this as a MECH-428 verdict."
        )
        interpretation = (
            "Readiness (G1) failed -- the forced-seed positive control did "
            "not itself produce a structured parent attractor in this "
            "harness, so the SUBGOAL_BOOTSTRAP arm cannot be meaningfully "
            "compared against a reference ceiling. Not evidence against "
            "MECH-428; diagnose the parent-level wiring/config in this "
            "harness before re-queuing."
        )
    else:
        all_pass = c1_pass and c2_pass
        status = "PASS" if all_pass else "FAIL"
        evidence_direction = "supports" if all_pass else "weakens"
        route_reason = "clean_scripted_bootstrap_scored"
        if all_pass:
            interpretation = (
                "MECH-428 SUPPORTED: in a regime where the parent "
                "(superordinate) attractor has exactly one write path "
                "(GoalState.credit_subgoal_attainment) and direct seeding is "
                "therefore structurally sparse (ARM_NO_SUBGOAL stays near "
                "zero), repeated REALISTIC-cadence subgoal-attainment credit "
                "(ARM_SUBGOAL_BOOTSTRAP) bootstrapped the parent to a "
                f"material fraction (>= {C1_FRACTION_FLOOR:.0%}) of the "
                "maximal forced-seed ceiling in the SAME harness -- bottom-up "
                "accumulation nets out ahead of the per-tick parent_goal_decay "
                "despite intermittent (not continuous) crediting events."
            )
        else:
            interpretation = (
                "MECH-428 WEAKENED: subgoal attainment occurred at a "
                "non-zero rate (G0 passed) and the forced-seed positive "
                "control demonstrated the wiring CAN sustain a structured "
                "parent (G1 passed), but repeated realistic-cadence credit "
                "did not accumulate a material fraction of that ceiling on "
                ">= 2/3 seeds -- genuine evidence that bottom-up "
                "bootstrapping at this alpha/decay/event-rate combination is "
                "too weak to net out against decay between sparse events."
            )

    metrics: Dict[str, float] = {
        "g2_pass": 1.0 if g2_pass else 0.0,
        "g2_min_episode_budget_frac": g2_min_budget_frac,
        "n_cells_starved": float(len(g2_failures)),
        "g0_frac_seeds": g0_frac,
        "g1_frac_seeds": g1_frac,
        "c1_frac_seeds": c1_frac,
        "c2_frac_seeds": c2_frac,
        "g0_pass": 1.0 if g0_pass else 0.0,
        "g1_pass": 1.0 if g1_pass else 0.0,
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "lift_fraction_mean": sum(lift_fractions) / len(lift_fractions),
    }
    for arm in ARMS:
        pn = [per_seed[arm][s]["parent_goal_norm_steady_state"] for s in SEEDS]
        nc = [per_seed[arm][s]["n_subgoal_credits"] for s in SEEDS]
        sa = [per_seed[arm][s]["n_steps_actual"] for s in SEEDS]
        metrics[f"parent_goal_norm_steady_state_mean_{arm}"] = sum(pn) / len(pn)
        metrics[f"n_subgoal_credits_mean_{arm}"] = sum(nc) / len(nc)
        metrics[f"n_steps_actual_mean_{arm}"] = sum(sa) / len(sa)

    evidence_direction_per_claim = {"MECH-428": evidence_direction}

    # --- dv_headroom (RECORDED, deliberately NOT a gate) -------------------
    # The load-bearing criterion C1 reads lift_fraction, which is normalised by
    # the FORCED_SEED - NO_SUBGOAL ceiling range measured IN THIS RUN, so the
    # DV's achievable ceiling is 1.0 BY CONSTRUCTION (the FORCED_SEED arm sits
    # at exactly 1.0 by definition of the denominator) and C1_FRACTION_FLOOR
    # (0.3) can never exceed it. There is therefore no headroom failure mode to
    # gate on -- the honest statement is that the bar is reachable and the run
    # measures whether the bootstrap arm reaches it. The block is recorded so a
    # later reader can see the measured denominators rather than re-derive them
    # (this is the recording half of the 884 autopsy's instrumentation
    # finding). Measured at probe scale 2026-09-08 (80 steps, seeds 42/43/44):
    # lift_fraction 0.243 / 0.297 / 0.259 against a bar of 0.300 -- the
    # criterion is discriminating and close to the boundary, not pinned.
    dv_headroom = {
        "dv_name": "lift_fraction",
        "criterion": "C1",
        "criterion_threshold": C1_FRACTION_FLOOR,
        "achievable": 1.0,
        "achievable_basis": (
            "FORCED_SEED defines the denominator, so its own lift_fraction is "
            "exactly 1.0; the DV is bounded [0, 1] whenever G1 holds"
        ),
        "statistic": "ratio_of_measured_ranges",
        "ceiling_range_per_seed": ceiling_ranges,
        "ceiling_range_min": min(ceiling_ranges) if ceiling_ranges else 0.0,
        "lift_fraction_per_seed": lift_fractions,
        "gates": False,
        "gate_rationale": (
            "A collapsed ceiling range is already caught by G1 "
            "(FORCED_STRUCTURED_FLOOR), which routes non_contributory; a "
            "second gate on the same condition would double-count it."
        ),
    }

    summary_markdown = (
        f"# {QUEUE_ID} -- MECH-428 Subgoal-Bootstrapped Goal Seeding\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}\n"
        f"**Route reason:** {route_reason}\n"
        f"**Claims:** MECH-428\n\n"
        f"## Gates\n\n"
        f"| Gate | Frac seeds | Pass |\n|---|---|---|\n"
        f"| G2 readiness (episode budget reached, every cell) | "
        f"{g2_min_budget_frac:.2f} (min) | {g2_pass} |\n"
        f"| G0 readiness (attainment, every seed) | {g0_frac:.2f} | {g0_pass} |\n"
        f"| G1 readiness (forced-seed structured, >=2/3) | {g1_frac:.2f} | {g1_pass} |\n"
        f"| C1 lift-fraction-of-ceiling (>= {C1_FRACTION_FLOOR:.0%}) | {c1_frac:.2f} | {c1_pass} |\n"
        f"| C2 measurability floor | {c2_frac:.2f} | {c2_pass} |\n\n"
        f"## Interpretation\n\n{interpretation}\n"
    )

    result: Dict[str, Any] = {
        "status": status,
        "outcome": status,
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "per_seed_results": per_seed,
        "dv_headroom": dv_headroom,
        "diagnostics": {"dv_headroom": dv_headroom},
        "episode_budget_audit": {
            "g2_pass": g2_pass,
            "min_episode_budget_frac": g2_min_budget_frac,
            "starved_cells": g2_failures,
            "per_cell": [
                {
                    "arm": r["arm"], "seed": r["seed"],
                    "n_steps_actual": r["n_steps_actual"],
                    "n_steps_configured": r["n_steps_configured"],
                    "done": r["episode_done"],
                    "done_cause": r["episode_done_cause"],
                    "agent_health_final": r["agent_health_final"],
                }
                for r in arm_results
            ],
        },
        "supersedes": SUPERSEDES,
        "fatal_error_count": 0,
    }
    if not non_degenerate:
        result["non_degenerate"] = False
        result["degeneracy_reason"] = degeneracy_reason

    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "seeds": SEEDS,
        "arms": ARMS,
        "grid_size": GRID_SIZE,
        "num_waypoints": NUM_WAYPOINTS,
        "n_steps": N_STEPS,
        "measurement_window": MEASUREMENT_WINDOW,
        "world_dim": WORLD_DIM,
        "parent_goal_alpha": PARENT_GOAL_ALPHA,
        "parent_goal_decay": PARENT_GOAL_DECAY,
        "forced_credit": FORCED_CREDIT,
        "forced_structured_floor": FORCED_STRUCTURED_FLOOR,
        "parent_norm_abs_floor": PARENT_NORM_ABS_FLOOR,
        "c1_fraction_floor": C1_FRACTION_FLOOR,
        # SD-094 -- the ONLY configuration difference from V3-EXQ-884.
        "subgoal_arrival_position_check": True,
        "hazard_free_contamination_gate": True,
        "supersedes": SUPERSEDES,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
