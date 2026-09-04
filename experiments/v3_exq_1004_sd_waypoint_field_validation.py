"""V3-EXQ-1004 -- SD-WAYPOINT-FIELD diagnostic ablation.

Does the landed waypoint proximity field (SD-WAYPOINT-FIELD, ree-v3 main; design doc
REE_assembly docs/architecture/sd_waypoint_proximity_field.md) make navigation-dependent
DVs REACHABLE?

THE BLOCKED FINDING THIS VALIDATES. In `subgoal_mode` a waypoint reaches the agent ONLY
as entity-type channel 6 of the 5x5x7 local view (radius 2), so a target more than two
cells away is absent from the observation entirely. The V3-EXQ-977 probe measured the
consequence: 3 waypoints, 12x12, 400 steps, seeds 42/43/44 -- the agent's own policy
visited 0/0/1 waypoints and completed 0 sequences, i.e. a random walk. Every
navigation-dependent DV in a `subgoal_mode` driver is therefore pinned at chance while
the manifest reads as a clean "this mechanism has no effect on goal maintenance" null.
That is the `corrupting` severity on the substrate_queue entry.

SD-WAYPOINT-FIELD appends a 25-dim agent-centred `1/(1 + decay*d)` field over the PENDING
waypoint (`obs_dict["waypoint_proximity_field_view"]`, and the trailing 25 dims of
`world_state`), so the target is perceptible at range. This run asks whether that
perceptibility CONVERTS into navigation.

=== ARMS (learner, objective, budget and seeds held FIXED across the two learner arms) ===

  ARM_RANDOM   uniform-random policy                   FLOOR reference (no learning)
  ARM_ORACLE   scripted greedy step toward the pending
               waypoint, from env ground truth         CEILING reference + DEMONSTRATOR
  ARM_OFF      cloned reader, radius-2 waypoint
               channel only (25 dims)                  THE BLOCKED BASELINE
  ARM_ON       cloned reader, + the 25-dim field
               (50 dims)                               THE POSITIVE DEMONSTRATION

The ONLY difference between ARM_OFF and ARM_ON is whether the 25-dim waypoint proximity
field is present in the reader's input -- same learner class, same trunk width, same
optimiser, same demonstrations, same batch schedule, same seeds, same env layout seed.

WHY THE READER IS CLONED FROM THE ORACLE RATHER THAN REINFORCEMENT-TRAINED. An A2C reader
on this task was built first and MEASURED not to work, and the measurement is the reason
for the change rather than a preference: on this geometry a random policy visits 0-1
waypoints per episode, so the +0.2 visit reward is almost never experienced and the
learner has nothing to bootstrap from. At 400 training episodes it reached 0.00 (OFF) and
0.10 (ON) visits/ep on seed 42 (measured BEFORE the SD-094 contamination gate below, so
on shorter episodes -- but the sparsity argument is independent of episode length) -- i.e.
the run would have returned an uninformative `learner_capacity_not_field_reach` at real
budget, spending compute to learn that A2C cannot solve a sparse-reward navigation task.
That is a property of the OBJECTIVE, not of the channel under test.

Cloning removes the exploration problem WITHOUT touching the question. The DV stays exactly
what the SD doc registers -- waypoints visited and sequences completed by the reader's OWN
policy, rolled out greedily with no scripted walk and no oracle in the loop. The
demonstrations are IDENTICAL across the two arms (same oracle, same seeds, same states);
the ONLY thing that differs is whether the reader can SEE the target it is being asked to
imitate. Where the oracle's action is not predictable from the observation -- which beyond
radius 2 is precisely the V3-EXQ-977 finding -- the clone cannot reproduce it. That makes
this a direct test of perceivability-converting-to-navigation.

The oracle is deliberately BOTH the ceiling reference and the demonstrator. It is not a
confound, because it is held identical across arms and never reads the field (see
`_oracle_action`); it is the fixed target both readers are measured against.

SECONDARY MECHANISTIC READOUT. `bc_accuracy` -- how often the cloned reader reproduces the
demonstrator's action on held-in states -- measures DECODABILITY directly, upstream of
behaviour. If the field carries the target's direction, the ON reader's accuracy must
exceed the OFF reader's; if accuracy moves but visits do not, the information arrived and
did not convert, which is a different finding and is labelled differently.

WHY THE REWARD IS NON-ZERO HERE, WHEN V3-EXQ-977's PROBE USED ZERO. 977 set
`waypoint_visit_reward=0` because it was measuring whether an INTRINSICALLY motivated REE
agent navigates unprompted. Reproducing that constant here would make this ablation
VACUOUS BY CONSTRUCTION: with a zero-reward objective NEITHER learner arm has a gradient
toward a waypoint, so both return the floor and the null would be a property of the
objective rather than of the channel. That is the same structural-null defect that
BLOCKED V3-EXQ-963b at this skill's Step 4.5 on 2026-09-04 -- a manipulation with no path
by which the treatment could differ. `WAYPOINT_VISIT_REWARD` is therefore non-zero and
IDENTICAL in both learner arms; what varies is only whether the target is PERCEIVABLE.
ARM_RANDOM still reproduces the 977 floor, so the blocked baseline is still anchored.

WHY HAZARDS AND RESOURCES ARE ZEROED. Measured on this geometry, the stock reward is
dominated by terms that have nothing to do with waypoints (hazard_approach -0.745,
env_caused_hazard -0.578, resource +0.712, against waypoint +0.400), so a reward-maximising
learner rationally ignores waypoints in BOTH arms and the ablation returns a null produced
by the objective. Zeroing them makes the waypoint visit the ONLY reward term -- the
learner's objective and the DV become the same quantity, so a null is attributable to the
channel. See the NAVIGATION ISOLATION note beside the constants for the measured numbers.

WHY THE ORACLE ARM EXISTS, AND WHY IT IS NOT DECORATIVE. The registered acceptance rule
is that the bar must be pre-registered INSIDE the DV's measured range. A range measured
off the TREATMENT arm is the treatment-as-control category error (`_metrics.dv_headroom_check`
documents `measured` as what the CONTROL can achieve). ARM_ORACLE supplies a genuine
achievable ceiling from ground truth -- it does NOT read the field, so it cannot launder
the manipulation into its own reference -- and ARM_RANDOM supplies the floor. The
dv_headroom precondition is denominated on the ORACLE arm, on the SAME statistic C1
routes on.

SAME-STATISTIC RULE. C1 routes on a SEED COUNT (>= MIN_SEEDS of N seeds show the lift),
so the achievable statistic is the MIN_SEEDS-th LARGEST per-seed oracle value, NOT the
oracle mean and NOT its max. A headroom gate denominated on a statistic the criterion
does not read certifies exactly the runs the dv_headroom class exists to stop.

PRE-REGISTERED OFF-RAMP (a negative result must not over-claim). If the ON arm does not
clear C1, two states are distinguishable and are labelled differently:
  * ON <= OFF but BOTH learner arms are at/below the ARM_RANDOM floor while ARM_ORACLE is
    well above it -- the task is solvable and the target was perceivable, but this learner
    at this budget never learned to navigate at all. Recorded as
    `learner_capacity_not_field_reach`, and it is NOT a verdict against the channel; it
    routes to a learner-capacity follow-up (more episodes, or the x734/948 reader family).
  * ON <= OFF while at least one learner arm IS above the floor -- the learner can navigate
    and the field did not help. Recorded as `waypoint_field_does_not_convert`.
Neither is scored against INV-086 or MECH-428; both record `non_contributory`, because a
diagnostic that could not separate reach from capacity has not tested either claim.

MEASURED AT FULL BUDGET (seeds 42/43, sampled eval, all fixes applied) -- recorded here
so the pre-registered bars can be audited against the data they were set from, and so a
reader can see this is not a hopeful design:

  arm                 visits/ep      sequences/ep   distinct cells/ep   BC accuracy
  random_floor        1.25 / 1.55    0.00           67.3                --
  cloned_field_off    3.85 / 0.35    0.00 / 0.00    75.1                0.573
  cloned_field_on     58.75 / 58.60  19.20 / 19.25  88.5                0.839
  oracle_ceiling      60.15 / 59.50  19.75 / 19.50  91.0                --

The ON clone reaches ~98% of the oracle it was cloned from; the OFF clone stays at the
blocked baseline while moving MORE than a random walk (75.1 cells vs 67.3), so its low
count is a navigation failure and not a stationary-policy artefact.

experiment_purpose: diagnostic -- this validates the BUILD (is the channel usable), not
INV-086's or MECH-428's own hypotheses. claim_ids are carried as read-across only: a PASS
lifts the substrate block those claims' experiments were failing under, it does not score
them.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import dv_headroom_check, p0_readiness_gate, P0NotReady  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from ree_core.action_learning.actor_critic import ActorCriticPolicy  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1004_sd_waypoint_field_validation"
QUEUE_ID = "V3-EXQ-1004"
CLAIM_IDS = ["INV-086", "MECH-428"]
EXPERIMENT_PURPOSE = "diagnostic"

# --- env (the V3-EXQ-977 probe geometry, except the reward -- see the docstring) -------
GRID_SIZE = 12
N_WAYPOINTS = 3
STEPS_PER_EPISODE = 400
WAYPOINT_VISIT_REWARD = 0.2      # env default; IDENTICAL in both learner arms
WAYPOINT_FIELD_DECAY = 0.25      # SD-WAYPOINT-FIELD default
# NAVIGATION ISOLATION -- a measured deviation from the stock env, not a convenience.
# Instrumented on this exact geometry (oracle policy, seed 42, 400-step budget) the stock
# reward composition is: hazard_approach -0.745, env_caused_hazard -0.578, resource
# +0.712, waypoint +0.400. A reward-maximising learner therefore rationally prioritises
# hazard avoidance and foraging OVER waypoint navigation, so BOTH arms can ignore
# waypoints and the DV returns a null caused by the OBJECTIVE rather than by the channel
# under test -- the same structural-null class that BLOCKED V3-EXQ-963b at Step 4.5.
# Zeroing hazards and resources makes the waypoint visit the ONLY reward term, so the
# learner's objective and the DV are the same thing and the ON/OFF contrast measures
# exactly perceivability. `energy_decay=0.0` because with no resources the agent would
# otherwise starve out the episode. Measured effect on the window (seeds 42/43/44):
# oracle visits/ep 5-7 -> 16-24 while random stays 0-1, i.e. a WIDER separation, and the
# random floor still reproduces the V3-EXQ-977 blocked baseline. (Those figures predate
# the SD-094 contamination gate below; with the gate the episode runs its full 400 steps
# and the oracle reaches ~60 visits/ep. Both measurements are recorded because the
# isolation and the gate are separate fixes and each is load-bearing on its own.)
NUM_HAZARDS = 0
NUM_RESOURCES = 0
ENERGY_DECAY = 0.0
# SD-094 self-contamination gate. Zeroing hazards is NOT sufficient: causal_grid_world.py
# documents that a hazard-free probe contaminates itself to death (V3-EXQ-884 terminated
# at 32/19/90 steps). MEASURED on this exact env before the gate: every arm ended
# `health_depleted` at 44-134 steps rather than the declared 400, and the per-episode
# reward was ~-1.2 contamination against ~+0.04 waypoint -- so "the waypoint visit is the
# only reward term" was false by 30:1 IN THE WRONG DIRECTION, and wall-pinning (a blocked
# move skips the movement block) was survival-optimal. The gate restores the declared
# 400-step episode and makes the isolation claim true.
HAZARD_FREE_CONTAMINATION_GATE = True
# F8: both are LIVE and readout-affecting -- the completion bonus is 4x the visit reward,
# and the commitment timeout respawns the whole waypoint set mid-episode, which moves the
# field's target. Declared so a reader can reproduce a cell.
WAYPOINT_COMPLETION_REWARD = 0.8
SEQUENCE_COMMITMENT_TIMEOUT = 20

# --- learner (identical across ARM_OFF / ARM_ON) ---------------------------------------
HIDDEN_DIM = 128
LR = 1e-3
GRAD_CLIP = 1.0
DEMO_EPISODES = 60        # oracle demonstration episodes collected per seed
BC_STEPS = 300            # cross-entropy steps over the demonstration buffer
BC_BATCH = 256
EVAL_EPISODES = 20

SEEDS = [42, 43, 44, 45, 46]
MIN_SEEDS = 3                    # C1 routes on this SEED COUNT

# --- pre-registered bars ---------------------------------------------------------------
# C1: the ON arm must beat the OFF arm's own per-seed value by this margin, in waypoints
# visited per eval episode. 977's floor is ~0.33 visits/ep (1 visit across 3 seeds); a
# margin of 1.0 asserts the ON arm reaches at least one MORE waypoint per episode than the
# blocked baseline -- inside the oracle's achievable range, which the readiness gate
# verifies rather than assumes.
# MEASURED at full budget on this exact geometry (cloned readers, sampled eval, seeds
# 42/43): random 1.25-1.55 visits/ep, field-OFF 0.35-3.85, field-ON 58.6-58.75, oracle
# 59.5-60.15. The realised ON-minus-OFF lift is ~55. 15.0 is ~27% of that achievable lift
# and roughly 4x the OFF arm's own best value -- deep inside the range the readiness gate
# verifies against the oracle control, and far enough above the OFF arm that no draw
# clears it. Deliberately NOT set near the realised 55: a bar tuned to the observed
# effect would be unfalsifiable by construction.
C1_VISIT_LIFT = 15.0
# C2 (secondary): sequences completed per eval episode, ON above OFF (measured 0.10 -> 6.30).
C2_SEQ_LIFT = 5.0   # measured 0.00-0.30 (OFF) vs 19.15-19.25 (ON)
# C3 (secondary, MECHANISTIC): imitation accuracy, upstream of behaviour. Measured
# 0.573 (OFF) -> 0.839 (ON) at full budget: the demonstrator's action is decodable from
# the field and is not decodable from the radius-2 channel alone.
C3_BC_ACC_LIFT = 0.05
# R1 (F5): the OFF arm must still be at the blocked baseline, judged against THIS run's
# own random arm. If it already navigates, the premise is false and the run reports
# `baseline_not_blocked` rather than looping on a requeue.
# Expressed as a FRACTION of this run's own measured (oracle - random) span rather than
# an absolute count: the OFF clone legitimately beats a random walk a little (the radius-2
# channel is real, just short-range -- measured 3.85 vs random 1.25 against an oracle of
# 60.15), and a tight absolute margin would misfire on that. The premise this guards is
# "the OFF arm does not already NAVIGATE", i.e. does not approach the achievable ceiling.
BASELINE_BLOCKED_FRAC = 0.25
# Separate, smaller margin for the off-ramp's "did this learner navigate at all" test.
LEARNER_MOVES_MARGIN = 1.0
# R2 (F2): both cloned readers must actually explore. An argmax policy on an
# under-trained net emits a constant action and pins against a wall; that is a
# degenerate EVAL PROTOCOL, not a channel verdict, and must self-route.
MIN_DISTINCT_CELLS = 5.0

# --- readiness-anchor predicates (the SHIPPED callables, scored per SEED) --------------
# Anchor-kind preconditions self-route to substrate_not_ready_requeue, so a predicate
# NARROWER than the state it anchors to would report unmet on every run forever and
# mislabel an instrument gap as a substrate verdict. Each is asserted reachable below.
def _off_at_floor_cell(off_visits: float, random_visits: float,
                       oracle_visits: float) -> bool:
    """Is the field-OFF arm still at the blocked baseline? Judged against THIS run's own
    random AND oracle arms, so the test self-calibrates to the achievable span instead of
    importing a literal from a probe run at a different reward and contamination regime."""
    span = max(0.0, float(oracle_visits) - float(random_visits))
    return float(off_visits) <= float(random_visits) + BASELINE_BLOCKED_FRAC * span


def _moves_cell(distinct_cells: float) -> bool:
    return float(distinct_cells) >= MIN_DISTINCT_CELLS


# Reachability of the two anchor-kind preconditions, asserted at setup against frozen
# reference cells with the SHIPPED predicate -- not a copy that could drift from it.
# F6: the previous `waypoint_field_live` / `waypoint_field_carries_gradient` anchors were
# DELETED rather than re-tuned. Both were tautologies -- the pending index is -1 only when
# no waypoint is pending (and arrival re-points within the same step(), so an observation
# never sees it), and any 5x5 patch of a strictly decreasing kernel has non-zero range --
# and both were anchored to CONSTANTS I wrote rather than recorded values of a control,
# which readiness_anchor.py rule 2 forbids. A gate that cannot fail certifies nothing. The
# field's liveness and range are still RECORDED (see `field_means`), just not gated on.

# BASELINE anchor: a random walk on this geometry, which is what the field-OFF reader must
# not beat. Cells are per-seed (off_visits, random_visits) pairs from THIS design measured
# on seeds 42/43/44 -- both terms from the same run, so the comparison is commensurable.
# (field-OFF, random, oracle) visits/ep, MEASURED on this design at full budget.
BASELINE_REFERENCE_CELLS = ((3.85, 1.25, 60.15), (0.35, 1.55, 59.50))
# STATIONARITY anchor: distinct cells per episode reached by a policy that does move.
# A wall-pinned argmax policy reaches ~1.
MOVEMENT_REFERENCE_CELLS = (75.1, 88.5, 67.3)   # measured: OFF / ON / random, sampled eval

ANCHOR_REACHABILITY = [
    assert_anchor_reachable(
        anchor_name="baseline_not_blocked",
        reference_cells=BASELINE_REFERENCE_CELLS,
        score_fn=lambda c: _off_at_floor_cell(c[0], c[1], c[2]),
        threshold=1.0,
        reference_source=("per-seed (field-OFF, random, oracle) visit triples measured "
                          "on this design at full budget, seeds 42/43 -- all three terms "
                          "from the same run, so the comparison is commensurable."),
    ),
    assert_anchor_reachable(
        anchor_name="eval_policy_not_stationary",
        reference_cells=MOVEMENT_REFERENCE_CELLS,
        score_fn=_moves_cell,
        threshold=1.0,
        reference_source=("distinct cells per episode reached by a moving policy on this "
                          "geometry; a wall-pinned argmax policy reaches ~1."),
    ),
]

ARM_RANDOM = "random_floor"
ARM_ORACLE = "oracle_ceiling"
ARM_OFF = "cloned_field_off"
ARM_ON = "cloned_field_on"
LEARNER_ARMS = (ARM_OFF, ARM_ON)
ARMS = (ARM_RANDOM, ARM_ORACLE, ARM_OFF, ARM_ON)

DEVICE = torch.device("cpu")

# The readout-affecting constants, declared unconditionally so the config_slice lint can
# see every knob a reader would need to reproduce a cell.
CONFIG_SLICE_KEYS = {
    "grid_size_declared": GRID_SIZE,
    "n_waypoints_declared": N_WAYPOINTS,
    "steps_per_episode_declared": STEPS_PER_EPISODE,
    "waypoint_visit_reward_declared": WAYPOINT_VISIT_REWARD,
    "waypoint_field_decay_declared": WAYPOINT_FIELD_DECAY,
    "num_hazards_declared": NUM_HAZARDS,
    "num_resources_declared": NUM_RESOURCES,
    "energy_decay_declared": ENERGY_DECAY,
    "hazard_free_contamination_gate_declared": HAZARD_FREE_CONTAMINATION_GATE,
    "waypoint_completion_reward_declared": WAYPOINT_COMPLETION_REWARD,
    "sequence_commitment_timeout_declared": SEQUENCE_COMMITMENT_TIMEOUT,
    "hidden_dim_declared": HIDDEN_DIM,
    "lr_declared": LR,
    "grad_clip_declared": GRAD_CLIP,
    "demo_episodes_declared": DEMO_EPISODES,
    "bc_steps_declared": BC_STEPS,
    "bc_batch_declared": BC_BATCH,
    "eval_episodes_declared": EVAL_EPISODES,
    "c1_visit_lift_declared": C1_VISIT_LIFT,
    "c2_seq_lift_declared": C2_SEQ_LIFT,
    "c3_bc_acc_lift_declared": C3_BC_ACC_LIFT,
    "min_seeds_declared": MIN_SEEDS,
}


def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else 0.0


def _build_env(field_on: bool, seed: Optional[int]) -> CausalGridWorldV2:
    """The V3-EXQ-977 probe geometry. `use_proxy_fields` and `subgoal_mode` are both
    PRECONDITIONS of the field flag (the env raises without them);
    `subgoal_arrival_position_check` is set because without it the SD-094 grid-marker
    defect suppresses arrival detection and the DV would read zero for a reason that has
    nothing to do with this channel."""
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        use_proxy_fields=True,
        subgoal_mode=True,
        num_waypoints=N_WAYPOINTS,
        waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
        subgoal_arrival_position_check=True,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        energy_decay=ENERGY_DECAY,
        hazard_free_contamination_gate=HAZARD_FREE_CONTAMINATION_GATE,
        waypoint_completion_reward=WAYPOINT_COMPLETION_REWARD,
        sequence_commitment_timeout=SEQUENCE_COMMITMENT_TIMEOUT,
        waypoint_proximity_field_enabled=bool(field_on),
        waypoint_field_decay=WAYPOINT_FIELD_DECAY,
    )


# The 5x5x7 local view occupies the first 175 dims of `world_state`, entity-major, and
# channel 6 is "waypoint" (CausalGridWorld.ENTITY_TYPES). `[:175][6::7]` is therefore the
# agent-centred radius-2 waypoint channel -- EXACTLY the whole of what a subgoal_mode
# agent could perceive about its target before SD-WAYPOINT-FIELD landed, and so exactly
# the right OFF-arm input. Both readers are agent-centred and small (25 / 50 dims),
# following the V3-EXQ-948 precedent (z_world 32 vs z_world+field 57) rather than feeding
# a 250-dim raw vector a shallow reader cannot exploit.
LOCAL_VIEW_DIMS = 175
N_ENTITY_TYPES = 7
WAYPOINT_ENTITY_CHANNEL = 6
FIELD_DIMS = 25


def _obs_vector(obs_dict: Dict[str, Any], field_on: bool) -> torch.Tensor:
    ws = np.asarray(obs_dict["world_state"], dtype=np.float32).reshape(-1)
    wp_local = ws[:LOCAL_VIEW_DIMS][WAYPOINT_ENTITY_CHANNEL::N_ENTITY_TYPES]
    # F7: the OFF arm is ZERO-PADDED to the ON width. Without this the two readers are
    # Linear(25,H) and Linear(50,H), which consume different init RNG, so the per-seed
    # pairing C1 relies on would be nominal rather than matched. Padded, both nets are
    # byte-identical at init for a given seed and the manipulation is PURELY the CONTENT
    # of the trailing 25 dims -- field values under ON, zeros under OFF.
    if field_on:
        fv = np.asarray(obs_dict["waypoint_proximity_field_view"],
                        dtype=np.float32).reshape(-1)
    else:
        fv = np.zeros(FIELD_DIMS, dtype=np.float32)
    vec = np.concatenate([wp_local, fv])
    return torch.as_tensor(vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)


def _field_diagnostics(env: CausalGridWorldV2, obs_dict: Dict[str, Any]) -> Tuple[bool, float]:
    """(has a pending target, cross-cell range of the field patch). Reads the obs key when
    present; a field with a target but zero range would be a constant, not a gradient."""
    fv = obs_dict.get("waypoint_proximity_field_view")
    if fv is None:
        return False, 0.0
    arr = np.asarray(fv, dtype=np.float32).reshape(-1)
    has_target = int(getattr(env, "_waypoint_field_target_idx", -1)) >= 0
    return bool(has_target), float(arr.max() - arr.min())


def _oracle_action(env: CausalGridWorldV2) -> int:
    """Greedy step toward the pending waypoint, read from env GROUND TRUTH -- deliberately
    NOT from the field, so this ceiling reference cannot launder the manipulation into its
    own number."""
    idx = int(getattr(env, "_next_waypoint_idx", 0))
    wps = getattr(env, "waypoints", []) or []
    if not wps or idx >= len(wps):
        return int(np.random.randint(0, int(env.action_dim)))
    wx, wy = int(wps[idx][0]), int(wps[idx][1])
    ax, ay = int(env.agent_x), int(env.agent_y)
    dx, dy = wx - ax, wy - ay
    # ACTIONS order is (up, down, left, right, stay) in CausalGridWorld.
    if abs(dx) >= abs(dy) and dx != 0:
        return 1 if dx > 0 else 0
    if dy != 0:
        return 3 if dy > 0 else 2
    return 4


def _rollout_counts(env: CausalGridWorldV2, act_fn, n_episodes: int,
                    collect_field: bool = False) -> Dict[str, Any]:
    """Count waypoint arrivals and completed sequences under `act_fn`, plus (optionally)
    the field-liveness diagnostics. NO scripted walk: `act_fn` is the arm's own policy."""
    visits, seqs, cells, steps = [], [], [], []
    field_live_ticks = 0
    field_ticks = 0
    field_ranges: List[float] = []
    for _ep in range(n_episodes):
        _flat, obs = env.reset()
        v = s = 0
        seen = {(int(env.agent_x), int(env.agent_y))}
        n_steps = 0
        for _t in range(STEPS_PER_EPISODE):
            n_steps += 1
            a = act_fn(env, obs)
            # CausalGridWorldV2.step -> (flat_obs, harm_signal, done, info, obs_dict).
            # harm_signal IS the reward channel (negative = harm, positive = benefit);
            # waypoint_visit_reward is added into it on arrival.
            _flat, _r, done, info, obs = env.step(a)
            tt = str(info.get("transition_type", "") or "")
            if tt == "waypoint":
                v += 1
            elif tt == "sequence_complete":
                v += 1
                s += 1
            seen.add((int(env.agent_x), int(env.agent_y)))
            if collect_field:
                field_ticks += 1
                live, rng = _field_diagnostics(env, obs)
                if live:
                    field_live_ticks += 1
                field_ranges.append(rng)
            if done:
                break
        visits.append(float(v))
        seqs.append(float(s))
        cells.append(float(len(seen)))
        steps.append(float(n_steps))
    out = {
        "waypoints_visited_per_ep": _mean(visits),
        "sequences_completed_per_ep": _mean(seqs),
        "n_eval_episodes": int(n_episodes),
        # F2: a policy that argmaxes into a wall is stationary. Distinct cells visited
        # separates "did not navigate" from "did not move at all", so a degenerate eval
        # policy self-routes instead of being scored as a channel verdict.
        "distinct_cells_per_ep": _mean(cells),
        "steps_per_ep": _mean(steps),
    }
    if collect_field:
        out["field_live_frac"] = (field_live_ticks / field_ticks) if field_ticks else 0.0
        out["field_range_mean"] = _mean(field_ranges)
    return out


def _collect_demonstrations(env: CausalGridWorldV2, field_on: bool,
                            n_episodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll the ORACLE and record (observation, oracle action) pairs. The observation is
    the ARM's own view, so the two arms see the SAME states and the SAME target actions
    and differ only in what of that state they can perceive."""
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    for _ep in range(n_episodes):
        _flat, obs = env.reset()
        for _t in range(STEPS_PER_EPISODE):
            a = _oracle_action(env)
            xs.append(_obs_vector(obs, field_on))
            ys.append(int(a))
            _flat, _r, done, _info, obs = env.step(a)
            if done:
                break
    if not xs:
        return torch.zeros((0, 1), device=DEVICE), torch.zeros((0,), dtype=torch.long,
                                                               device=DEVICE)
    return torch.cat(xs), torch.as_tensor(ys, dtype=torch.long, device=DEVICE)


def _clone(policy: ActorCriticPolicy, xs: torch.Tensor, ys: torch.Tensor,
           n_steps: int) -> Dict[str, Any]:
    """Cross-entropy imitation of the demonstrator. Identical schedule in both arms."""
    if xs.shape[0] == 0:
        return {"bc_steps": 0, "bc_accuracy": 0.0, "bc_final_loss": float("nan")}
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    last = float("nan")
    n = int(xs.shape[0])
    for _i in range(n_steps):
        idx = torch.randperm(n, device=DEVICE)[:min(BC_BATCH, n)]
        logits, _v, _phi, _psi = policy.forward(xs[idx])
        loss = torch.nn.functional.cross_entropy(logits, ys[idx])
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        opt.step()
        last = float(loss.detach())
    with torch.no_grad():
        pred = policy.forward(xs)[0].argmax(dim=-1)
        acc = float((pred == ys).float().mean())
    return {"bc_steps": int(n_steps), "bc_accuracy": acc, "bc_final_loss": last,
            "n_demonstrations": n}


def _eval_policy(policy: ActorCriticPolicy, field_on: bool, sample: bool):
    """The cloned reader's OWN policy -- no oracle in the loop.

    F2 (red-team, MEASURED): argmax eval on a reader whose input does not determine the
    demonstrator's action emits a near-constant action and PINS against a wall. Measured
    at full budget: the field-OFF clone reached only 5.8-6.5 distinct cells per episode
    and 0.25 visits/ep, i.e. BELOW the 1.55 random floor -- a stationary policy, not a
    navigating one. Under argmax the DV would partly measure the eval protocol rather
    than the channel. SAMPLED eval is therefore the PRIMARY DV (a stochastic policy
    cannot pin), with argmax recorded alongside as a secondary readout."""
    def _act(env: CausalGridWorldV2, obs: Dict[str, Any]) -> int:
        with torch.no_grad():
            logits, _v, _phi, _psi = policy.forward(_obs_vector(obs, field_on))
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        if sample:
            probs = torch.softmax(logits.reshape(-1), dim=-1)
            return int(torch.multinomial(probs, 1).item())
        return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
    return _act


def _random_policy(env: CausalGridWorldV2, obs: Dict[str, Any]) -> int:
    return int(np.random.randint(0, int(env.action_dim)))


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _run_seed(seed: int, dry_run: bool) -> Dict[str, Any]:
    demo_eps = 3 if dry_run else DEMO_EPISODES
    bc_steps = 20 if dry_run else BC_STEPS
    eval_eps = 2 if dry_run else EVAL_EPISODES
    row: Dict[str, Any] = {"seed": int(seed)}

    for arm in ARMS:
        _seed_all(seed)
        field_on = arm == ARM_ON
        env = _build_env(field_on=field_on, seed=seed)
        if arm == ARM_RANDOM:
            res = _rollout_counts(env, _random_policy, eval_eps)
            res["n_updates"] = 0
        elif arm == ARM_ORACLE:
            res = _rollout_counts(env, lambda e, _o: _oracle_action(e), eval_eps)
            res["n_updates"] = 0
        else:
            xs, ys = _collect_demonstrations(env, field_on, demo_eps)
            obs_dim = int(xs.shape[-1]) if xs.shape[0] else 1
            policy = ActorCriticPolicy(
                world_dim=obs_dim, action_dim=int(env.action_dim),
                hidden_dim=HIDDEN_DIM, use_sf_critic=False,
            ).to(DEVICE)
            bc = _clone(policy, xs, ys, bc_steps)
            # PRIMARY: sampled eval (cannot pin). SECONDARY: argmax, recorded so the two
            # protocols can be compared rather than one silently standing for the other.
            res = _rollout_counts(env, _eval_policy(policy, field_on, sample=True),
                                  eval_eps, collect_field=field_on)
            argmax_res = _rollout_counts(env, _eval_policy(policy, field_on, sample=False),
                                         eval_eps)
            res["argmax_waypoints_visited_per_ep"] = argmax_res["waypoints_visited_per_ep"]
            res["argmax_sequences_completed_per_ep"] = argmax_res["sequences_completed_per_ep"]
            res["argmax_distinct_cells_per_ep"] = argmax_res["distinct_cells_per_ep"]
            res.update(bc)
            res["n_updates"] = int(bc.get("bc_steps", 0))
            res["obs_dim"] = obs_dim
        res["arm"] = arm
        res["field_enabled"] = bool(field_on)
        row[arm] = res
        print(f"[{EXPERIMENT_TYPE}] seed {seed} {arm}: "
              f"visits/ep={res['waypoints_visited_per_ep']:.3f} "
              f"seqs/ep={res['sequences_completed_per_ep']:.3f}", flush=True)
    return row


def _score(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    def col(arm: str, key: str) -> List[float]:
        return [float(r[arm][key]) for r in per_seed]

    off_v = col(ARM_OFF, "waypoints_visited_per_ep")
    on_v = col(ARM_ON, "waypoints_visited_per_ep")
    orc_v = col(ARM_ORACLE, "waypoints_visited_per_ep")
    rnd_v = col(ARM_RANDOM, "waypoints_visited_per_ep")
    off_s = col(ARM_OFF, "sequences_completed_per_ep")
    on_s = col(ARM_ON, "sequences_completed_per_ep")

    n = len(per_seed)
    need = min(MIN_SEEDS, n)

    c1_seeds = [i for i in range(n) if (on_v[i] - off_v[i]) >= C1_VISIT_LIFT]
    c2_seeds = [i for i in range(n) if (on_s[i] - off_s[i]) >= C2_SEQ_LIFT]
    off_cells = [float(r[ARM_OFF].get("distinct_cells_per_ep", 0.0)) for r in per_seed]
    on_cells = [float(r[ARM_ON].get("distinct_cells_per_ep", 0.0)) for r in per_seed]
    off_a = [float(r[ARM_OFF].get("bc_accuracy", 0.0)) for r in per_seed]
    on_a = [float(r[ARM_ON].get("bc_accuracy", 0.0)) for r in per_seed]
    c3_seeds = [i for i in range(n) if (on_a[i] - off_a[i]) >= C3_BC_ACC_LIFT]
    c1 = len(c1_seeds) >= need
    c2 = len(c2_seeds) >= need
    c3 = len(c3_seeds) >= need

    # R1 -- the blocked baseline is still blocked.
    r1_val = _mean(off_v)
    # R2 -- the field is live and carries a gradient in the ON arm.
    live = [float(r[ARM_ON].get("field_live_frac", 0.0)) for r in per_seed]
    rng = [float(r[ARM_ON].get("field_range_mean", 0.0)) for r in per_seed]
    r2_live, r2_range = _mean(live), _mean(rng)
    _ = (r1_val, r2_live, r2_range)  # reported in arm_means / field_means, not the gate

    def seed_count_achievable(vals: List[float]) -> float:
        """The value the MIN_SEEDS-of-N count actually turns on: the need-th LARGEST.
        NOT the mean and NOT the max -- the criterion routes on a seed count, so a
        headroom gate denominated on any other statistic certifies the wrong thing."""
        mags = sorted((float(v) for v in vals), reverse=True)
        if len(mags) < need:
            return float("nan")
        return mags[need - 1]

    # The achievable LIFT the oracle demonstrates over the blocked baseline, on the same
    # per-seed-then-seed-count statistic C1 reads. The control is the ORACLE arm, which
    # never reads the field.
    orc_lift = [orc_v[i] - off_v[i] for i in range(n)]
    checks = [
        dv_headroom_check(
            "dv_headroom_waypoint_visit_lift",
            dv_name="waypoints_visited_per_ep lift over the field-OFF baseline",
            criterion_threshold=C1_VISIT_LIFT,
            achievable=seed_count_achievable(orc_lift),
            margin=1.0,
            n_seed_values=n,
            seed_count_required=need,
            control=("MIN_SEEDS-th largest per-seed (ORACLE - OFF) visit lift on this run "
                     "-- the value C1's seed count turns on, measured on a control arm "
                     "that navigates from env ground truth and never reads the field."),
        ),
        # Scored as the FRACTION OF SEEDS clearing the shipped per-seed predicate --
        # the same callable ANCHOR_REACHABILITY asserted against the frozen reference,
        # not a second copy that could drift from it.
        # F5: anchored to THIS RUN's own random arm, per seed, not to a literal carried
        # over from a probe run at a different reward and contamination regime. A
        # violation means the premise is false (the baseline already navigates), which is
        # `baseline_not_blocked` -- NOT "the substrate is unready", which would requeue in
        # a loop.
        {"name": "baseline_not_blocked",
         "measured": sum(1 for i in range(n)
                         if _off_at_floor_cell(off_v[i], rnd_v[i], orc_v[i])) / max(1, n),
         "threshold": 1.0, "direction": "lower",
         "control": ("fraction of seeds whose field-OFF arm stays within "
                     "BASELINE_FLOOR_MARGIN of THIS run's own random arm.")},
        # F2: the eval policy must actually move. An argmax policy on an under-trained
        # net emits one constant action and pins against a wall, which reads as "did not
        # navigate" while measuring nothing about the channel.
        {"name": "eval_policy_not_stationary",
         "measured": sum(1 for i in range(n) if _moves_cell(on_cells[i])
                         and _moves_cell(off_cells[i])) / max(1, n),
         "threshold": 1.0, "direction": "lower",
         "control": ("fraction of seeds where BOTH cloned readers explore more than "
                     "MIN_DISTINCT_CELLS distinct cells per episode.")},
    ]

    sample_unmet = False
    unmet: List[str] = []
    try:
        gated = p0_readiness_gate(checks)
    except P0NotReady as exc:
        sample_unmet = True
        gated = list(getattr(exc, "preconditions", checks) or checks)
        unmet = [c.get("name", "?") for c in gated if not c.get("met", False)]

    measurable = not sample_unmet
    outcome = "PASS" if (measurable and c1) else "FAIL"
    # Separate "the field did not help a learner that CAN navigate" from "this learner
    # never navigated at all" -- see the docstring's pre-registered off-ramp. The oracle
    # arm is what makes the distinction non-circular: it establishes the task is solvable
    # WITHOUT reading the field, so a both-arms-at-floor result is attributable to learner
    # capacity rather than to the channel under test.
    # F3: a PER-SEED, MARGINED, SEED-COUNTED population test -- not an existence test
    # over all cells, which one lucky visit on one seed would satisfy (readiness_anchor.py
    # rule 3, the existence-vs-population distinction).
    learner_navigating_seeds = [
        i for i in range(n)
        if max(off_v[i], on_v[i]) > rnd_v[i] + LEARNER_MOVES_MARGIN
    ]
    oracle_navigating_seeds = [
        i for i in range(n) if orc_v[i] > rnd_v[i] + LEARNER_MOVES_MARGIN
    ]
    learner_above_floor = len(learner_navigating_seeds) >= need
    oracle_above_floor = len(oracle_navigating_seeds) >= need
    if sample_unmet:
        label = "substrate_not_ready_requeue"
    elif c1:
        label = "waypoint_field_converts_to_navigation"
    elif c3 and not c1:
        # The information ARRIVED (the field made the demonstrator's action decodable) but
        # did not convert into navigation. A distinct finding from "the field carries
        # nothing", and it must not be recorded as the latter.
        label = "field_decodable_but_did_not_convert"
    elif oracle_above_floor and not learner_above_floor:
        label = "learner_capacity_not_field_reach"
    else:
        label = "waypoint_field_does_not_convert"

    return {
        "outcome": outcome,
        "interpretation": {"label": label, "measurable": measurable,
                           "unmet_preconditions": unmet},
        "preconditions": gated,
        "criteria": {
            "C1_visit_lift": {"met": bool(c1), "load_bearing": True,
                              "n_seeds": len(c1_seeds), "required": need,
                              "threshold": C1_VISIT_LIFT},
            "C2_sequence_lift": {"met": bool(c2), "load_bearing": False,
                                 "n_seeds": len(c2_seeds), "required": need,
                                 "threshold": C2_SEQ_LIFT},
            "C3_decodability_lift": {"met": bool(c3), "load_bearing": False,
                                     "n_seeds": len(c3_seeds), "required": need,
                                     "threshold": C3_BC_ACC_LIFT},
        },
        "bc_accuracy_means": {"off": _mean(off_a), "on": _mean(on_a)},
        "arm_means": {
            "random_visits_per_ep": _mean(rnd_v),
            "oracle_visits_per_ep": _mean(orc_v),
            "off_visits_per_ep": _mean(off_v),
            "on_visits_per_ep": _mean(on_v),
            "off_sequences_per_ep": _mean(off_s),
            "on_sequences_per_ep": _mean(on_s),
        },
        "off_ramp": {"oracle_navigating_seeds": len(oracle_navigating_seeds),
                     "learner_navigating_seeds": len(learner_navigating_seeds),
                     "required": need,
                     "margin": LEARNER_MOVES_MARGIN},
        "field_means": {"field_live_frac": r2_live, "field_range_mean": r2_range,
                        "off_visits_per_ep_mean": r1_val},
        "anchor_reachability": list(ANCHOR_REACHABILITY),
        # Directions are read-across only and are withheld entirely when the run is not
        # measurable -- an unmet readiness gate must never be recorded as evidence.
        "claim_directions": (
            {cid: ("supports" if c1 else "non_contributory") for cid in CLAIM_IDS}
            if measurable else {cid: "non_contributory" for cid in CLAIM_IDS}
        ),
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:2] if dry_run else SEEDS
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    per_seed = [_run_seed(s, dry_run) for s in seeds]
    result = _score(per_seed)

    print(f"[{EXPERIMENT_TYPE}] outcome={result['outcome']} "
          f"label={result['interpretation']['label']}", flush=True)
    if dry_run:
        return {"outcome": result["outcome"], "manifest_path": None}

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "per_seed": per_seed,
        "config": dict(CONFIG_SLICE_KEYS),
    }
    manifest.update(result)
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=False, config=manifest.get("config"),
        seeds=list(seeds), script_path=Path(__file__),
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    _o = str(_res["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_res.get("manifest_path"), dry_run=args.dry_run)
