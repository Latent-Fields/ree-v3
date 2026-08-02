#!/opt/local/bin/python3
"""
V3-EXQ-884 -- MECH-428 Subgoal-Bootstrapped Goal Seeding: 3-arm
NO-SUBGOAL / SUBGOAL-BOOTSTRAP / FORCED-SEED comparison.

Claim: MECH-428 (subgoal_bootstrapped_goal_seeding, formation-direction sibling
of MECH-427). Proposal: EXP-0390.

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

PASS (supports MECH-428) = G0 AND G1 AND C1 AND C2.
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
EXPERIMENT_TYPE = "v3_exq_884_mech428_subgoal_bootstrapped_goal_seeding"
QUEUE_ID = "V3-EXQ-884"
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
    return CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=0,
        num_resources=0,
        subgoal_mode=True,
        num_waypoints=NUM_WAYPOINTS,
        seed=seed,
    )


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

    for _step in range(n_steps):
        action_idx = _scripted_action(env)
        obs_flat, _harm_signal, done, info, _obs_dict = env.step(action_idx)
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

    print(
        f"  [eval] {arm} seed={seed} n_subgoal_credits={n_subgoal_credits} "
        f"n_waypoint_events={n_waypoint_events} "
        f"n_sequence_complete_events={n_sequence_complete_events} "
        f"steady_state_parent_norm={steady_state_norm:.4f}",
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
        "n_steps": n_steps,
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
            }
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                row = _run_cell(arm, seed, zg, dry_run=dry_run)
                cell.stamp(row)
            arm_results.append(row)
            per_seed[arm][seed] = row

    g0_per_seed, g1_per_seed, c1_per_seed, c2_per_seed = [], [], [], []
    lift_fractions: List[float] = []
    for seed in SEEDS:
        no_sub = per_seed["NO_SUBGOAL"][seed]
        boot = per_seed["SUBGOAL_BOOTSTRAP"][seed]
        forced = per_seed["FORCED_SEED"][seed]

        g0_per_seed.append(boot["n_subgoal_credits"] > 0)
        g1_per_seed.append(forced["parent_goal_norm_steady_state"] >= FORCED_STRUCTURED_FLOOR)

        ceiling_range = forced["parent_goal_norm_steady_state"] - no_sub["parent_goal_norm_steady_state"]
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

    if not g0_pass:
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
        metrics[f"parent_goal_norm_steady_state_mean_{arm}"] = sum(pn) / len(pn)
        metrics[f"n_subgoal_credits_mean_{arm}"] = sum(nc) / len(nc)

    evidence_direction_per_claim = {"MECH-428": evidence_direction}

    summary_markdown = (
        f"# {QUEUE_ID} -- MECH-428 Subgoal-Bootstrapped Goal Seeding\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}\n"
        f"**Route reason:** {route_reason}\n"
        f"**Claims:** MECH-428\n\n"
        f"## Gates\n\n"
        f"| Gate | Frac seeds | Pass |\n|---|---|---|\n"
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
