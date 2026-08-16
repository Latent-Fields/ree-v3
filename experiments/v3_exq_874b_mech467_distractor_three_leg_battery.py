"""V3-EXQ-874b (EXP-0398, supersedes V3-EXQ-874): MECH-467 distractor battery --
REDESIGN that (a) guarantees a leg-(c) event denominator, (b) instruments rule
state AT THE POINT OF ACTION SELECTION rather than at a storage site, and
(c) adds an ENCODING-SIDE index for leg 1 so the claim's non-degeneracy guard can
tell registered-then-suppressed from never-encoded.

EXPERIMENT_PURPOSE: see EXPERIMENT_PURPOSE constant below.

WHY A REDESIGN (V3-EXQ-874 -> 874b)
-----------------------------------
V3-EXQ-874 (autopsy failure_autopsy_V3-EXQ-874_2026-08-03, confirmed, governance
2026-08-03) fairly tested legs (a) sensory-capture and (b) rule-drift -- clear
dissociation, ARM_PRECOMMIT drift ~0.22 vs ARM_REPLAY drift ~0.015 -- but leg (c)
(wrong-target selection) had a 0/0 event denominator in ALL 6 cells: zero
target-consumption events of either kind across 900 real navigation ticks. Not a
measured null; the decisive leg never fired. Classified
non_contributory/measurement_test_design_defect, routed to /queue-experiment for a
battery that GUARANTEES target-consumption opportunities occur.

MEASURED ROOT CAUSE OF THE 0/0 DENOMINATOR -- NOT WHAT THE AUTOPSY ASSUMED
--------------------------------------------------------------------------
The autopsy hypothesised the 10x10 grid / 150-tick window under internal-only
operating modes was too small a budget for an approach-and-consume to complete, and
routed "denser targets and/or a longer horizon". A pre-authoring probe (2026-08-16,
scratch, P0-trained agent, this substrate) reproduced the zero-event outcome and
found the binding constraint is NARROWER and more severe than horizon:

    10x10, num_resources=5, 300 eval ticks, P0-trained, mode pinned:
        internal_planning: moved 14/300 ticks, 9 unique cells, 0 events
        internal_replay  : moved 14/300 ticks, 11 unique cells, 0 events

The agent does not fail to COMPLETE an approach; it barely MOVES (~5% of ticks
change its cell, ~10 distinct cells visited in 300 ticks, action histogram
dominated by one action). A longer horizon on the SAME geometry multiplies a ~0
event rate by a constant and would very likely have reproduced the defect. What
actually moved the number was shrinking the arena so the few moves that DO occur
land on resources:

    6x6, num_resources=6, 300 eval ticks, P0-trained, mode pinned:
        internal_planning: moved 9/300, 3 consumption events, z_goal LIVE
        internal_replay  : moved 9/300, 3 consumption events, z_goal LIVE

so this redesign takes the arena down to 6x6 at ~3x the resource density and
extends the window to 900 ticks, rather than only extending the window.

Two rejected alternatives, recorded so they are not re-tried:
  * TOROIDAL wrap (removes the walls that trap the agent) restores movement to
    100% of ticks -- and kills the agent: agent_health hits 0.0 by tick ~22-27
    even with num_hazards=0. Rejected.
  * An external_task (committed-pursuit) arm would navigate, but that IS the
    during-commitment arm the claim's SEQUENCING CAUTION excludes, and
    [memory] feedback_dont_queue_commitment_dependent_behavioural. Rejected.

The same probe established the second load-bearing fact: z_goal only goes LIVE
(goal_state.is_active() True, goal_norm 0.0156-0.0765) once consumption events
occur, because GoalState.update's benefit gate needs benefit_exposure above
goal.benefit_threshold. The event floor and the selection-path rule instrument
below are therefore the SAME precondition, not two independent ones.

THE THREE LEGS, AS REDESIGNED
------------------------------
leg 1  SENSORY CAPTURE -- now an ENCODING-SIDE index, not a proximity proxy.
       (2026-08-16 lit: 2026-08-16_mech_467_salient_distractor_suppression_gaspelin2017)
       874 read `resource_field_view_<distractor>` from obs_dict, which measures
       whether the distractor is PRESENT, not whether it was ENCODED. Gaspelin &
       Luck's finding is that registered-then-SUPPRESSED and never-encoded produce
       the same floor capture rate, so a guard built on that quantity fires
       substrate_not_ready on exactly the case where the substrate handled the
       distractor CORRECTLY -- the most expensive way to be wrong, because a
       self-route looks clean. Humans separate the two with the Pd ERP component.
       REE's analogue, built here: at sampled ticks where a distractor is proximal,
       re-encode the SAME observation with the distractor field-view block(s) zeroed
       and measure how far z_world moves --
           distractor_encoding_index = ||z_world_full - z_world_ablated|| / ||z_world_full||
       This is an encoding-side read: it asks whether the distractor's features
       entered the active representation, independently of whether they altered
       selection. High index + low behavioural capture = registered-then-suppressed
       (substrate working); low index = never encoded (guard fires legitimately).
       Probe-measured live and non-degenerate on this substrate: index ranged
       0.000-0.706 across sampled ticks. Reported as a TRIAL-LEVEL distribution
       (per-tick array + quantiles), never a block mean -- suppression waxes and
       wanes within a session, and a block mean can describe no tick that occurred.
       The 874 proximity measure is retained as `distractor_proximity_rate` for
       continuity, but it no longer carries the non-degeneracy guard.

leg 2  RULE STATE -- read at TWO sites, and the SELECTION-PATH one is load-bearing.
       (2026-08-16 lit: 2026-08-16_mech_467_goal_neglect_rule_reportable_duncan2008,
        corroborated from the other direction by ..._oculomotor_capture_awareness_adams2021)
       Duncan's goal neglect IS leg 3 -- "a person ignores some task requirement
       though being able to describe it" -- but his MECHANISM is that task-model
       components are lost from the OPERATIVE model under representational
       competition, so what remains describable and what remains operative are
       different objects. Put that against MECH-467's PASS criterion (rule drift at
       floor while wrong-target selection is elevated): if drift is read off a
       STORED rule vector -- the analogue of asking the participant -- then plain
       leg-2 rule corruption in the selection path produces that exact signature.
       Leg 2 wearing leg 3's clothes; the dissociation would be a property of where
       the probe sat, not of the substrate. Both lit entries independently land on
       the same requirement, and Adams & Gaspelin note REE can do what the human
       experimenter cannot: inspect the rule representation in the selection path.
       So:
         (i) STORAGE-SITE (874's measure, kept for comparability):
             ||lateral_pfc.rule_state - rule_state_at_warmup_end||   (SD-033a)
         (ii) SELECTION-PATH (new, load-bearing): at each tick E3 actually selects,
             score the SAME candidate set under the CURRENT z_goal and under the
             z_goal snapshotted at warmup end, and take the Spearman rank
             correlation of the two per-candidate score vectors:
                 operative_rule_fidelity = rho(goal_scores_now, goal_scores_ref)
             This asks whether the rule STEERING SELECTION still ranks the options
             the way the established rule did. Probe-confirmed non-degenerate: 32
             candidates with genuinely distinct goal scores (3.659-3.699), so the
             rank correlation is meaningful rather than tied.
       WHY NOT THE SD-033a BIAS HEAD, which looks like the obvious selection-path
       read: with train_rule_bias_head=False (default) the head's last Linear is
       zeroed, so compute_bias() returns exactly 0 and rule_state has NO authority
       over selection at all; and with the SD-082 centering+tanh consumer engaged
       V3-EXQ-822/822a still measured rule_state->action-bias propagation at exactly
       0.0. A counterfactual built on that read is identically zero -- structurally
       vacuous. The z_goal/goal-proximity channel is the rule signal that demonstrably
       DOES reach the E3 comparator (compute_goal_score, MECH-112/117), so it is the
       one instrumented here.

leg 3  BEHAVIOURAL CAPTURE -- wrong-target selection EXCESS OVER MEASURED CHANCE,
       conditioned on the SELECTION-PATH rule read being intact.
       874 scored a raw conditioned wrong-target rate against a flat 0.10 threshold.
       At the resource densities needed to guarantee an event denominator that is a
       DV-symmetry hazard: if enough cells carry a distractor, the wrong-target rate
       approaches the distractor share of live cells REGARDLESS of what selection
       does, and the "effect" is an arithmetic property of the spawn, fixed before
       the run. The SD-049 spawn allocator is also stochastic -- probe-measured
       goal:distractor splits varied run to run at fixed nominal weights -- so the
       baseline cannot be pinned by config. It is therefore MEASURED at each event
       (live distractor cells / live resource cells, read off the env's own type
       grid) and the DV is the EXCESS:
           wrong_target_excess = wrong_target_rate_intact - mean_chance_baseline
       That is selection-attributable by construction, and the composition is
       recorded per arm so a later reader can audit it.
       (2026-08-16 lit: ..._oculomotor_capture_awareness_adams2021 additionally
       predicts that any mechanism proposing to close leg 3 by DETECTING wrong-target
       selection and feeding it back into rule maintenance will be insufficient --
       humans have that monitoring signal and it does not fix the behaviour. This
       battery accordingly contains NO detect-and-correct arm; leg 3 is not treated
       as a detection problem.)

ARMS -- 2 x 2, and the second axis is Duncan's, not a power bump
-----------------------------------------------------------------
  TIMING (from the claim, unchanged from 874):
    PRECOMMIT -- operating_mode pinned internal_planning (active rule-update regime)
    REPLAY    -- operating_mode pinned internal_replay   (protected regime, gate ~0.05)
    The during-commitment arm is EXCLUDED per the claim's SEQUENCING CAUTION.
  RULE-SET COMPLEXITY (new; Duncan 2008's load result):
    SIMPLE  -- 2 SD-049 resource types (1 goal + 1 distractor)
    COMPLEX -- 4 SD-049 resource types (1 goal + 3 distinct zero-benefit distractors)
    Duncan's result is that goal neglect is driven by TASK-MODEL COMPLEXITY, not by
    moment-to-moment demand: increasing the real-time demand of one component does
    not promote neglect of another. A battery that cranks perceptual distraction
    while holding the rule set at two categories may therefore never elicit
    behavioural capture and return a floor that reflects the DESIGN, not the
    substrate -- and MECH-467's own non-degeneracy guard watches for an unregistered
    distractor but not for an under-complex rule set. The distractor CELL DENSITY is
    not what changes here: what changes is how many distinct rule categories those
    distractor cells are split into. (The realized per-type composition is stochastic
    in the SD-049 allocator, which is exactly why leg 3's baseline is measured per
    event rather than assumed from the nominal weights.)
  4 arms x 3 seeds = 12 cells.

CONTAINMENT (standing rule, still in force from the 2026-06-04 attention analysis):
REE-v3 attention is DISTRIBUTED precision-selection control, not a missing module.
This is a DIAGNOSTIC OVER EXISTING SUBSTRATE, not a build. No new attention module;
no expansion of the V3 green-board closure path. Every instrument above reads
telemetry the substrate already produces (SD-049 per-type field views and consumed
tag, SD-033a rule_state, MECH-112/117 goal_proximity) or re-runs an existing forward
pass (the encoding-side ablation re-calls agent.sense()).

OWNERSHIP: MECH-262 continues to own leg 2 (rule corruption); MECH-467 owns legs 1
and 3. Only MECH-467 is tagged.

PSEUDO-REPLICATION: this driver reads NO `agent.e3.last_*` latched diagnostic. The
selection-path read is computed by the driver over the candidate list it holds, and
is recorded ONLY on ticks where ticks["e3_tick"] is True (E3 genuinely selected);
ticks where the E3 cadence held the previous action are counted in
`n_latched_ticks` and contribute nothing. See the queue-experiment skill's
sample-size-integrity block.

Run:
  /opt/local/bin/python3 experiments/v3_exq_874b_mech467_distractor_three_leg_battery.py
Smoke:
  /opt/local/bin/python3 experiments/v3_exq_874b_mech467_distractor_three_leg_battery.py --dry-run
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
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_874b_mech467_distractor_three_leg_battery"
QUEUE_ID = "V3-EXQ-874b"
SUPERSEDES = "V3-EXQ-874"
CLAIM_IDS = ["MECH-467"]
EXPERIMENT_PURPOSE = "evidence"
# Kept identical to V3-EXQ-874 for direct comparability. Not a reef-config env, so
# the seed-44 early-truncation instability does not apply (874's own note).
SEEDS = [42, 43, 44]

GOAL_TAG = 1  # SD-049 tag = type_idx + 1; type_idx 0 is always the benefit-bearing goal

# ---- rule-set complexity axis (Duncan 2008 task-model complexity) -------------
RULESETS = {
    "SIMPLE": {
        "n_resource_types": 2,
        "resource_type_names": ("goal", "distractor_a"),
        "resource_type_drive_axes": ("goal_need", "d_a_need"),
        "resource_type_distribution": (1.0, 1.0),
        "resource_type_benefit_amplitudes": (1.0, 0.0),
    },
    "COMPLEX": {
        "n_resource_types": 4,
        "resource_type_names": ("goal", "distractor_a", "distractor_b", "distractor_c"),
        "resource_type_drive_axes": ("goal_need", "d_a_need", "d_b_need", "d_c_need"),
        "resource_type_distribution": (1.5, 1.0, 1.0, 1.0),
        "resource_type_benefit_amplitudes": (1.0, 0.0, 0.0, 0.0),
    },
}
TIMINGS = {"PRECOMMIT": "internal_planning", "REPLAY": "internal_replay"}
ARMS = tuple(
    f"ARM_{timing}_{ruleset}" for ruleset in ("SIMPLE", "COMPLEX") for timing in TIMINGS
)

# ---- pre-registered constants (NOT derived from the run's own statistics) -----
GRID_SIZE = 6              # probe-measured: the 10x10 arena is what starved leg (c)
NUM_RESOURCES = 12         # ~3x the 874 density; keeps the arena a choice, not a floor
MAX_EPISODE_STEPS = 1500   # comfortably above N_WARMUP_STEPS + N_EVAL_STEPS
N_WARMUP_STEPS = 80        # in-episode rule + z_goal establishment, mode UNPINNED
N_EVAL_STEPS = 900         # distractor-exposure window, mode pinned per arm
P0_BUDGET = 60
P0_STEPS_PER_EPISODE = 80
ENC_PROBE_EVERY = 5        # encoding-side ablation costs one extra sense() per probe

RULE_DRIFT_FLOOR = 0.05           # storage-site "near-frozen" (874's threshold, kept)
OPERATIVE_RULE_FIDELITY_FLOOR = 0.90   # Spearman rho vs the warmup-established rule
WRONG_TARGET_EXCESS_MIN = 0.10    # leg (c) elevation ABOVE the measured chance baseline
MIN_EVENTS_POOLED = 15            # leg (c) event floor -- the gap 874's guard left open
ENCODING_INDEX_FLOOR = 0.01       # leg (a) encoding-side registration floor
CHANCE_BASELINE_CEILING = 0.95    # a near-1.0 baseline forces wrong-target arithmetically
# Positive control for the SELECTION-PATH instrument. The load-bearing leg-(b)
# criterion routes on a Spearman rank correlation, so the readiness check must assert
# that SAME statistic on a control where it is known to be low -- not a magnitude
# proxy for it (the V3-EXQ-643 same-statistic rule). A --dry-run smoke measured
# operative_rule_fidelity at 0.9994-1.0000 in every cell, which is the correct reading
# when z_goal has barely moved from its warmup-established value (the operative rule
# really IS the established rule) but is INDISTINGUISHABLE from an instrument that
# cannot move at all. So each fidelity sample is paired with the identical computation
# against a RANDOMISED reference z_goal; if that control is also near 1.0 the ranking
# is insensitive to the rule content, "rule intact" carries no information, and the
# arm must self-route substrate_not_ready_requeue rather than score a dissociation.
FIDELITY_CONTROL_CEILING = 0.70

# Smoke budgets
SMOKE_P0_BUDGET = 3
SMOKE_P0_STEPS = 15
SMOKE_WARMUP = 5
SMOKE_EVAL = 40
SMOKE_MIN_EVENTS = 1


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ------------------------------------------------------------------ env / agent
def _env_kwargs(ruleset: str, distractors_present: bool) -> dict:
    spec = RULESETS[ruleset]
    n = spec["n_resource_types"]
    dist = (
        spec["resource_type_distribution"]
        if distractors_present
        else tuple([1.0] + [0.0] * (n - 1))
    )
    return {
        "size": GRID_SIZE,
        "num_hazards": 0,
        "num_resources": NUM_RESOURCES,
        "num_waypoints": 2,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "resource_respawn_on_consume": True,
        "multi_resource_heterogeneity_enabled": True,
        "per_axis_drive_enabled": False,
        "n_resource_types": n,
        "resource_type_names": spec["resource_type_names"],
        "resource_type_drive_axes": spec["resource_type_drive_axes"],
        "resource_type_benefit_curves": ("sigmoidal_saturating",) * n,
        "resource_type_distribution": dist,
        "resource_type_benefit_amplitudes": spec["resource_type_benefit_amplitudes"],
        "dual_cue_enabled": distractors_present,
        "dual_cue_min_active_ticks": 10,
        "dual_cue_replace_on_early_consume": False,
        "dual_cue_type_tags": (GOAL_TAG, GOAL_TAG + 1),
    }


def _build_env(ruleset: str, distractors_present: bool) -> CausalGridWorldV2:
    """P0 env (distractors_present=False) keeps the SAME resource-type schema as the
    target env so world_obs_dim matches and the P0-trained agent transfers; only the
    spawn distribution changes. Mirrors V3-EXQ-874's easy/target pair."""
    return CausalGridWorldV2(seed=None, **_env_kwargs(ruleset, distractors_present))


def _build_agent(world_obs_dim: int, body_obs_dim: int = 12) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=4,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        # MECH-112/117: the selection-path rule read is the E3 goal channel, so the
        # z_goal stream must be live. Probe-confirmed it activates once consumption
        # events occur (GoalState.update's benefit gate).
        z_goal_enabled=True,
    )
    return REEAgent(cfg)


def _pin_operating_mode_across_ticks(agent: REEAgent, mode: str) -> None:
    """Wrap agent.salience.tick so the FORCED mode survives every subsequent call.

    Carried over verbatim in spirit from V3-EXQ-874's SUBSTRATE-API FINDING:
    SalienceCoordinator.tick() unconditionally recomputes _operating_mode from
    softmax(logits) on every call, and agent.py calls it inside select_action(), so a
    one-shot attribute assignment before the loop is clobbered before write_gate() is
    ever read. This still runs the REAL tick (preserving _n_ticks / salience_aggregate
    / hysteresis bookkeeping) and then forces the pin, so every downstream
    write_gate() read within that same tick sees it.
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


# --------------------------------------------------------------- instruments
def _spearman(a, b) -> float:
    """Spearman rank correlation, stdlib only. Returns 0.0 on a degenerate input."""
    n = len(a)
    if n < 3 or len(b) != n:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, idx in enumerate(order):
            r[idx] = float(pos)
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return num / (da * db)


def _candidate_world_states(agent: REEAgent, traj) -> torch.Tensor:
    """[steps, world_dim] world-state sequence for one candidate trajectory."""
    ws = agent.e3._get_world_states(traj)
    return ws.reshape(-1, ws.shape[-1])


def _goal_score_vectors(agent: REEAgent, candidates, z_goal: torch.Tensor):
    """Per-candidate goal scores under an ARBITRARY z_goal: (full, rule_attributable).

    `full` is E3's own quantity: goal_proximity = 1 / (1 + MSE(z_world, z_goal))
    (ree_core/goal.py) summed over the trajectory horizon, exactly as
    E3Selector.compute_goal_score does. It is reported for faithfulness.

    `rule_attributable` is the LOAD-BEARING one, and the reason is measured, not
    stylistic. Expanding the squared error,
        MSE(z_w, z_g) = (||z_w||^2 - 2<z_w, z_g> + ||z_g||^2) / d
    the ||z_g||^2 term is constant across candidates and the ||z_w||^2 term is a
    candidate-MAGNITUDE term that is there for ANY goal -- neither carries rule
    content. Only the cross term <z_w, z_g> depends on WHICH rule is in force. On
    this substrate the magnitude term dominates the ranking (the SD-008 ~0.98-cosine
    z_world cone), so the full-proximity RANK ORDER is largely reference-independent:
    a --dry-run smoke measured rank correlation 0.65-0.93 between the true z_goal and
    a RANDOMISED one, i.e. the full-proximity ranking cannot tell the established
    rule from noise and a fidelity of ~1.0 read off it would have carried almost no
    information. The rule-attributable score sum_t <z_w_t, z_goal> keeps only the
    reference-dependent component, so the positive control can actually fail.

    Note the centering that SD-082 applies to the bias head is NOT the fix here: a
    common-mode subtraction is a uniform additive shift across candidates and
    therefore leaves every rank -- and every argmax -- exactly unchanged (the
    broadcast-scalar row of the DV-symmetry table).
    """
    ref = z_goal.reshape(1, -1)
    full, rule_attr = [], []
    for traj in candidates:
        ws = _candidate_world_states(agent, traj)
        mse = ((ws - ref) ** 2).mean(dim=-1)
        full.append(float((1.0 / (1.0 + mse)).sum().item()))
        rule_attr.append(float((ws @ ref.reshape(-1)).sum().item()))
    return full, rule_attr


def _locate_field_blocks(obs_dict: dict, field_keys) -> dict:
    """Map each distractor field-view key to its (offset, length) inside world_state.

    The SD-049 per-type field views are concatenated into world_state by
    causal_grid_world (world_parts); there is no public index, so the block is
    located by value match once and re-verified before each use. A key that cannot
    be located is omitted, and the arm records enc_block_located=False -- which
    fails the leg-(a) precondition rather than silently reporting a zero index.
    """
    world_flat = obs_dict["world_state"].reshape(-1)
    found = {}
    for key in field_keys:
        dv = obs_dict.get(key)
        if dv is None:
            continue
        dvf = dv.reshape(-1).to(world_flat.dtype)
        n = dvf.numel()
        if n == 0 or n > world_flat.numel():
            continue
        for start in range(world_flat.numel() - n + 1):
            if torch.allclose(world_flat[start:start + n], dvf):
                found[key] = (start, n)
                break
    return found


def _encoding_index(agent: REEAgent, obs_dict: dict, blocks: dict) -> float:
    """Leg (a): how far z_world moves when the distractor field block(s) are zeroed.

    Returns -1.0 when the located blocks no longer verify (caller re-locates).
    """
    ob = obs_dict["body_state"]
    ow = obs_dict["world_state"]
    ow_abl = ow.clone()
    flat = ow_abl.reshape(-1)
    for key, (start, n) in blocks.items():
        dv = obs_dict.get(key)
        if dv is None:
            return -1.0
        dvf = dv.reshape(-1).to(flat.dtype)
        if dvf.numel() != n or not torch.allclose(ow.reshape(-1)[start:start + n], dvf):
            return -1.0
        flat[start:start + n] = 0.0
    z_full = agent.sense(ob, ow).z_world
    z_abl = agent.sense(ob, ow_abl).z_world
    denom = float(z_full.norm().item())
    if denom <= 0.0:
        return 0.0
    return float((z_full - z_abl).norm().item()) / denom


def _live_resource_composition(env: CausalGridWorldV2) -> tuple:
    """(n_goal_cells, n_distractor_cells) currently live on the grid."""
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
    ruleset: str,
    timing: str,
    env: CausalGridWorldV2,
    device: torch.device,
    smoke: bool,
) -> dict:
    """One (seed, arm) cell: clone -> unpinned warmup -> snapshot -> pinned window.

    ONE continuous episode; no mid-run agent.reset(). agent.reset() zeroes
    lateral_pfc.rule_state (SD-033a spec), so a per-episode-reset drift metric would
    measure "does the rule re-establish from zero" rather than distractor resistance
    -- V3-EXQ-874's RULE-STATE-RESET FINDING, unchanged.
    """
    agent = clone_trained_agent(agent_base, bistable=False, device=device)
    agent.eval()

    n_warmup = SMOKE_WARMUP if smoke else N_WARMUP_STEPS
    n_eval = SMOKE_EVAL if smoke else N_EVAL_STEPS

    agent.reset()
    _, obs_dict = env.reset()
    world_dim = agent.config.latent.world_dim

    distractor_keys = [
        f"resource_field_view_{name}"
        for name in RULESETS[ruleset]["resource_type_names"][1:]
    ]

    def _step(obs):
        """One live env step through the standard REEAgent pipeline.
        Returns (next_obs, info, done, ticks, candidates)."""
        latent = agent.sense(obs["body_state"].to(device), obs["world_state"].to(device))
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
        # z_goal at zero-init, which silently disables the E3 goal channel and with it
        # this experiment's whole selection-path rule read.
        agent.update_z_goal(
            float(info.get("benefit_exposure", 0.0) or 0.0), drive_level=1.0
        )
        return next_obs, info, done, ticks, candidates

    # ---- P1: unpinned warmup -- establish the rule and let z_goal go live ----
    with torch.no_grad():
        for _ in range(n_warmup):
            obs_dict, _info, done, _t, _c = _step(obs_dict)
            if done:
                break

    rule_pre = agent.lateral_pfc.rule_state.clone()
    goal_state = agent.goal_state
    z_goal_ref = (
        goal_state.z_goal.clone() if goal_state is not None
        else torch.zeros(1, world_dim, device=device)
    )
    goal_live_at_warmup_end = bool(goal_state is not None and goal_state.is_active())
    # Positive control for the selection-path instrument (see FIDELITY_CONTROL_CEILING).
    # Deterministic given the cell's RNG state, which arm_cell() has fully reset.
    z_goal_control = torch.randn_like(z_goal_ref) * float(
        max(z_goal_ref.abs().max().item(), 1.0)
    )

    _pin_operating_mode_across_ticks(agent, TIMINGS[timing])

    # ---- P2: pinned distractor-exposure window ----
    blocks = _locate_field_blocks(obs_dict, distractor_keys)
    enc_block_located = len(blocks) == len(distractor_keys) and len(blocks) > 0

    n_ticks = 0
    n_e3_ticks = 0
    n_latched_ticks = 0
    n_proximal = 0
    enc_index_trials = []          # trial-level, per Gaspelin (never a block mean only)
    fidelity_trials = []
    fidelity_control_trials = []   # same statistic, randomised-reference positive control
    fidelity_full_trials = []      # E3's own full-proximity ranking, reported not gated
    storage_drift_trials = []
    events = []                    # one dict per consumption event
    write_gate_samples = []
    final_drift = 0.0

    with torch.no_grad():
        for step_i in range(n_eval):
            # --- leg (a): proximity (874 continuity) + encoding-side index (new) ---
            proximal = False
            for key in distractor_keys:
                dv = obs_dict.get(key)
                if dv is not None and float(dv.max().item()) > 0.0:
                    proximal = True
                    break
            n_proximal += int(proximal)
            if proximal and enc_block_located and (step_i % ENC_PROBE_EVERY == 0):
                idx = _encoding_index(agent, obs_dict, blocks)
                if idx < 0.0:
                    blocks = _locate_field_blocks(obs_dict, distractor_keys)
                    enc_block_located = (
                        len(blocks) == len(distractor_keys) and len(blocks) > 0
                    )
                    if enc_block_located:
                        idx = _encoding_index(agent, obs_dict, blocks)
                if idx >= 0.0:
                    enc_index_trials.append(round(idx, 6))

            n_goal_cells, n_distr_cells = _live_resource_composition(env)

            obs_dict, info, done, ticks, candidates = _step(obs_dict)
            n_ticks += 1

            # --- leg (b) storage-site read (874's measure) ---
            drift_now = float((agent.lateral_pfc.rule_state - rule_pre).norm().item())
            final_drift = drift_now
            storage_drift_trials.append(round(drift_now, 6))

            # --- leg (b) SELECTION-PATH read, only on genuine E3 selections ---
            fidelity = None
            if ticks.get("e3_tick"):
                n_e3_ticks += 1
                if goal_state is not None and len(candidates) >= 3:
                    now_full, now_ra = _goal_score_vectors(
                        agent, candidates, goal_state.z_goal)
                    ref_full, ref_ra = _goal_score_vectors(
                        agent, candidates, z_goal_ref)
                    _ctl_full, ctl_ra = _goal_score_vectors(
                        agent, candidates, z_goal_control)
                    fidelity = _spearman(now_ra, ref_ra)
                    fidelity_trials.append(round(fidelity, 6))
                    fidelity_control_trials.append(round(_spearman(now_ra, ctl_ra), 6))
                    fidelity_full_trials.append(
                        round(_spearman(now_full, ref_full), 6))
            else:
                n_latched_ticks += 1

            write_gate_samples.append(float(agent.salience.write_gate("sd_033a")))

            # --- leg (c): consumption event, with its own chance baseline ---
            consumed_tag = int(info.get("sd049_consumed_type_tag_this_tick", 0))
            if consumed_tag > 0:
                live_total = n_goal_cells + n_distr_cells
                baseline = (n_distr_cells / live_total) if live_total > 0 else 0.0
                # The most recent selection-path reading in force at this event. On a
                # latched tick the operative rule is whatever the last genuine E3
                # selection established, so the last fidelity reading is the correct
                # one to condition on.
                last_fid = fidelity if fidelity is not None else (
                    fidelity_trials[-1] if fidelity_trials else None
                )
                events.append({
                    "tick": step_i,
                    "wrong_target": consumed_tag > GOAL_TAG,
                    "chance_baseline": round(baseline, 6),
                    "storage_drift": round(drift_now, 6),
                    "operative_rule_fidelity": (
                        round(last_fid, 6) if last_fid is not None else None
                    ),
                    "selection_path_rule_intact": bool(
                        last_fid is not None
                        and last_fid >= OPERATIVE_RULE_FIDELITY_FLOOR
                    ),
                    "storage_rule_intact": bool(drift_now < RULE_DRIFT_FLOOR),
                })

            if done:
                break

    n_events = len(events)
    n_wrong = sum(1 for e in events if e["wrong_target"])
    intact_events = [e for e in events if e["selection_path_rule_intact"]]
    n_intact_events = len(intact_events)
    n_wrong_intact = sum(1 for e in intact_events if e["wrong_target"])
    both_intact_events = [e for e in intact_events if e["storage_rule_intact"]]
    n_both_intact = len(both_intact_events)
    n_wrong_both_intact = sum(1 for e in both_intact_events if e["wrong_target"])

    def _mean(xs, default=0.0):
        return round(statistics.fmean(xs), 6) if xs else default

    def _quantiles(xs):
        if not xs:
            return {"n": 0}
        s = sorted(xs)
        def q(p):
            return round(s[min(len(s) - 1, max(0, int(p * (len(s) - 1))))], 6)
        return {
            "n": len(s), "min": round(s[0], 6), "p25": q(0.25), "median": q(0.5),
            "p75": q(0.75), "max": round(s[-1], 6), "mean": _mean(xs),
        }

    chance_all = _mean([e["chance_baseline"] for e in events])
    chance_intact = _mean([e["chance_baseline"] for e in intact_events])
    wt_rate_all = (n_wrong / n_events) if n_events else 0.0
    wt_rate_intact = (n_wrong_intact / n_intact_events) if n_intact_events else 0.0
    wt_rate_both = (n_wrong_both_intact / n_both_intact) if n_both_intact else 0.0

    return {
        "arm": arm,
        "ruleset": ruleset,
        "timing": timing,
        "operating_mode": TIMINGS[timing],
        "n_ticks": n_ticks,
        "n_e3_ticks": n_e3_ticks,
        "n_latched_ticks": n_latched_ticks,
        # leg (a)
        "distractor_proximity_rate": round(n_proximal / max(1, n_ticks), 6),
        "enc_block_located": enc_block_located,
        "distractor_encoding_index_mean": _mean(enc_index_trials),
        "distractor_encoding_index_quantiles": _quantiles(enc_index_trials),
        "distractor_encoding_index_trials": enc_index_trials,
        # leg (b)
        "final_rule_drift": round(final_drift, 6),
        "storage_drift_quantiles": _quantiles(storage_drift_trials),
        "goal_live_at_warmup_end": goal_live_at_warmup_end,
        "operative_rule_fidelity_mean": _mean(fidelity_trials),
        "operative_rule_fidelity_quantiles": _quantiles(fidelity_trials),
        "operative_rule_fidelity_trials": fidelity_trials,
        "n_fidelity_samples": len(fidelity_trials),
        "operative_rule_fidelity_control_mean": _mean(fidelity_control_trials),
        "operative_rule_fidelity_control_quantiles": _quantiles(fidelity_control_trials),
        "operative_rule_fidelity_control_trials": fidelity_control_trials,
        "operative_rule_fidelity_full_mean": _mean(fidelity_full_trials),
        "operative_rule_fidelity_full_trials": fidelity_full_trials,
        # leg (c)
        "n_consumption_events": n_events,
        "n_wrong_target_events": n_wrong,
        "n_intact_events": n_intact_events,
        "n_wrong_target_intact": n_wrong_intact,
        "n_both_reads_intact_events": n_both_intact,
        "n_wrong_target_both_intact": n_wrong_both_intact,
        "chance_baseline_mean": chance_all,
        "chance_baseline_mean_intact": chance_intact,
        "wrong_target_rate_all": round(wt_rate_all, 6),
        "wrong_target_rate_intact": round(wt_rate_intact, 6),
        "wrong_target_rate_both_intact": round(wt_rate_both, 6),
        "wrong_target_excess_intact": round(wt_rate_intact - chance_intact, 6),
        "wrong_target_excess_both_intact": round(wt_rate_both - chance_intact, 6),
        "events": events,
        "mean_write_gate_sd033a": _mean(write_gate_samples),
    }


# --------------------------------------------------------------------- driver
def run_seed(seed: int, device: torch.device, smoke: bool, zg_acc) -> dict:
    torch.manual_seed(seed)
    p0_budget = SMOKE_P0_BUDGET if smoke else P0_BUDGET
    p0_steps = SMOKE_P0_STEPS if smoke else P0_STEPS_PER_EPISODE

    arm_rows = {}
    p0_summary = {}
    for ruleset in ("SIMPLE", "COMPLEX"):
        easy_env = _build_env(ruleset, distractors_present=False)
        target_env = _build_env(ruleset, distractors_present=True)
        agent = _build_agent(target_env.world_obs_dim).to(device)

        print(f"Seed {seed} Condition P0_{ruleset}", flush=True)
        p0 = run_p0_warmup(
            agent, easy_env, device, budget=p0_budget, steps_per_episode=p0_steps
        )
        print(
            f"  [train] {ruleset} seed={seed} ep {p0.n_episodes}/{p0_budget}"
            f" converged={p0.converged} aborted={p0.aborted} rv={p0.final_rv:.5f}",
            flush=True,
        )
        p0_summary[ruleset] = {
            "n_episodes": p0.n_episodes,
            "converged": bool(p0.converged),
            "aborted": bool(p0.aborted),
            "final_rv": float(p0.final_rv),
        }
        if p0.aborted:
            for timing in TIMINGS:
                arm = f"ARM_{timing}_{ruleset}"
                print(f"Seed {seed} Condition {arm}", flush=True)
                print("verdict: FAIL", flush=True)
                arm_rows[arm] = {
                    "arm": arm, "ruleset": ruleset, "timing": timing,
                    "p0_aborted": True, "p0_abort_reason": p0.abort_reason,
                    "n_consumption_events": 0, "n_intact_events": 0,
                    "n_fidelity_samples": 0, "enc_block_located": False,
                    "distractor_encoding_index_mean": 0.0,
                    "chance_baseline_mean_intact": 0.0,
                    "wrong_target_excess_intact": 0.0,
                    "wrong_target_excess_both_intact": 0.0,
                    "n_both_reads_intact_events": 0,
                    "final_rule_drift": 0.0,
                    "operative_rule_fidelity_mean": 0.0,
                    "distractor_encoding_index_trials": [],
                    "operative_rule_fidelity_trials": [],
                    "operative_rule_fidelity_control_trials": [],
                    "operative_rule_fidelity_control_mean": 1.0,
                    "n_wrong_target_events": 0,
                    "n_wrong_target_both_intact": 0,
                    "n_latched_ticks": 0,
                    "events": [],
                }
            continue

        for timing in TIMINGS:
            arm = f"ARM_{timing}_{ruleset}"
            print(f"Seed {seed} Condition {arm}", flush=True)
            with arm_cell(
                seed,
                config_slice={
                    "arm": arm,
                    "ruleset": ruleset,
                    "timing": timing,
                    "operating_mode": TIMINGS[timing],
                    "env": _env_kwargs(ruleset, True),
                    "n_warmup_steps": N_WARMUP_STEPS,
                    "n_eval_steps": N_EVAL_STEPS,
                    "p0_budget": P0_BUDGET,
                    "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
                },
                script_path=Path(__file__),
                config_slice_declared=True,
                # The P0 warmup is SHARED across both timing arms of a given
                # (seed, ruleset): both clone the same p0-trained agent, so neither
                # cell is a pure function of (seed, arm config) from a fresh RNG
                # reset. Honestly ineligible rather than falsely marked reusable --
                # the sanctioned shared-mutable-state reason, not a "one-off" excuse.
                extra_ineligible_reasons=["shared_p0_warmup_across_timing_arms"],
            ) as cell:
                row = _run_arm(agent, arm, ruleset, timing, target_env, device, smoke)
                row["p0_aborted"] = False
                cell.stamp(row)
            arm_rows[arm] = row
            min_events = SMOKE_MIN_EVENTS if smoke else MIN_EVENTS_POOLED
            cell_pass = (
                row["n_consumption_events"] > 0
                and row["n_both_reads_intact_events"] > 0
                and row["wrong_target_excess_both_intact"] >= WRONG_TARGET_EXCESS_MIN
            )
            row["cell_dissociation_pass"] = bool(cell_pass)
            print(
                f"verdict: {'PASS' if cell_pass else 'FAIL'}"
                f" events={row['n_consumption_events']}"
                f" excess={row['wrong_target_excess_both_intact']:.4f}"
                f" fidelity={row['operative_rule_fidelity_mean']:.4f}"
                f" drift={row['final_rule_drift']:.4f}"
                f" enc={row['distractor_encoding_index_mean']:.4f}"
                f" (min_events_pooled={min_events})",
                flush=True,
            )

        zg_acc.observe(agent)

    return {
        "seed": seed,
        "p0": p0_summary,
        "arms": arm_rows,
        "pass": any(r.get("cell_dissociation_pass") for r in arm_rows.values()),
    }


# ------------------------------------------------------------ precondition gate
def _precondition_specs(smoke: bool):
    min_events = SMOKE_MIN_EVENTS if smoke else MIN_EVENTS_POOLED
    return [
        PreconditionSpec(
            name="leg_c_event_floor",
            description=(
                "pooled target-consumption events (correct OR incorrect) in this arm. "
                "V3-EXQ-874 had 0/0 in all 6 cells and scored a 0.000 rate; the claim's "
                "own guard checked leg (a) only and had no event floor for leg (c). "
                "This is that floor."
            ),
            control=(
                "consumption events counted from the env's own "
                "sd049_consumed_type_tag_this_tick, pooled across seeds within the arm"
            ),
            threshold=float(min_events),
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="distractor_encoded_in_active_representation",
            description=(
                "encoding-side registration: ||z_world_full - z_world_distractor_ablated|| "
                "/ ||z_world_full|| on ticks where a distractor is proximal. Replaces the "
                "874 proximity proxy, which cannot separate registered-then-suppressed "
                "from never-encoded (Gaspelin & Luck 2018)."
            ),
            control=(
                "same observation re-encoded with the SD-049 distractor field block(s) "
                "zeroed; probe-measured 0.000-0.706 on this substrate"
            ),
            threshold=ENCODING_INDEX_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="selection_path_rule_read_live",
            description=(
                "number of genuine E3 selections at which the per-candidate goal-score "
                "vector could be scored under both the current and the reference z_goal. "
                "Zero means the selection-path rule read never ran, so leg (b)'s "
                "load-bearing measure is absent and leg (c) cannot be conditioned on it."
            ),
            control="ticks where ticks['e3_tick'] is True and z_goal is live",
            threshold=1.0,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="operative_rule_fidelity_instrument_sensitive",
            description=(
                "mean Spearman rho of the per-candidate goal-score ranking under the "
                "CURRENT z_goal against a RANDOMISED reference z_goal -- the same "
                "statistic the load-bearing leg-(b) criterion routes on, measured on a "
                "control where it must be LOW. A --dry-run smoke read the real fidelity "
                "at 0.9994-1.0000, which is correct when z_goal has barely left its "
                "warmup value but is indistinguishable from a ranking insensitive to "
                "rule content. If the control is ALSO near 1.0, 'operative rule intact' "
                "carries no information and no dissociation may be scored off it."
            ),
            control=(
                "randomised z_goal of comparable scale, scored over the identical "
                "candidate set at the identical tick"
            ),
            threshold=FIDELITY_CONTROL_CEILING,
            direction="upper",
            kind="readiness",
        ),
        PreconditionSpec(
            name="chance_baseline_not_saturated",
            description=(
                "mean live-distractor share of live resource cells at event time. At a "
                "saturated baseline the wrong-target rate is forced by the spawn "
                "composition and the leg-(c) DV is an arithmetic identity, not a "
                "measurement."
            ),
            control="env _resource_type_grid composition read at each consumption event",
            threshold=CHANCE_BASELINE_CEILING,
            direction="upper",
            kind="readiness",
        ),
    ]


def _pool_arm(seed_results, arm) -> dict:
    rows = [
        r["arms"][arm] for r in seed_results
        if arm in r.get("arms", {}) and not r["arms"][arm].get("p0_aborted")
    ]
    if not rows:
        return {
            "arm": arm, "n_seeds": 0, "n_events": 0, "n_intact_events": 0,
            "n_both_reads_intact_events": 0, "n_wrong_target_both_intact": 0,
            "encoding_index_mean": 0.0, "n_fidelity_samples": 0,
            "chance_baseline_mean_intact": 0.0, "wrong_target_rate_both_intact": 0.0,
            "wrong_target_excess_both_intact": 0.0, "mean_final_rule_drift": 0.0,
            "operative_rule_fidelity_mean": 0.0, "storage_rule_at_floor": False,
            "selection_path_rule_intact": False, "n_latched_ticks": 0,
            "n_wrong_target_events": 0, "encoding_index_max": 0.0,
            "n_encoding_samples": 0,
            # No samples -> the control cannot certify the instrument. 1.0 is the
            # FAILING value for this upper-bound precondition, so an arm with no
            # fidelity samples is red here as well as on the sample-count floor,
            # rather than inheriting a spuriously-passing 0.0.
            "operative_rule_fidelity_control_mean": 1.0,
        }
    n_events = sum(r["n_consumption_events"] for r in rows)
    n_intact = sum(r["n_intact_events"] for r in rows)
    n_both = sum(r["n_both_reads_intact_events"] for r in rows)
    n_wrong_both = sum(r["n_wrong_target_both_intact"] for r in rows)
    enc_all = [v for r in rows for v in r.get("distractor_encoding_index_trials", [])]
    fid_all = [v for r in rows for v in r.get("operative_rule_fidelity_trials", [])]
    ctl_all = [
        v for r in rows
        for v in r.get("operative_rule_fidelity_control_trials", [])
    ]
    intact_events = [
        e for r in rows for e in r.get("events", [])
        if e.get("selection_path_rule_intact")
    ]
    chance_intact = (
        statistics.fmean([e["chance_baseline"] for e in intact_events])
        if intact_events else 0.0
    )
    wt_both = (n_wrong_both / n_both) if n_both else 0.0
    mean_drift = statistics.fmean([r["final_rule_drift"] for r in rows])
    mean_fid = statistics.fmean(fid_all) if fid_all else 0.0
    return {
        "arm": arm,
        "n_seeds": len(rows),
        "n_events": n_events,
        "n_intact_events": n_intact,
        "n_both_reads_intact_events": n_both,
        "n_wrong_target_both_intact": n_wrong_both,
        "n_wrong_target_events": sum(r["n_wrong_target_events"] for r in rows),
        "encoding_index_mean": round(statistics.fmean(enc_all), 6) if enc_all else 0.0,
        "encoding_index_max": round(max(enc_all), 6) if enc_all else 0.0,
        "n_encoding_samples": len(enc_all),
        "n_fidelity_samples": len(fid_all),
        "n_latched_ticks": sum(r["n_latched_ticks"] for r in rows),
        "chance_baseline_mean_intact": round(chance_intact, 6),
        "wrong_target_rate_both_intact": round(wt_both, 6),
        "wrong_target_excess_both_intact": round(wt_both - chance_intact, 6),
        "mean_final_rule_drift": round(mean_drift, 6),
        "operative_rule_fidelity_mean": round(mean_fid, 6),
        # 1.0 (the FAILING value for this upper-bound precondition) when the control
        # never sampled -- an uncertified instrument must not read as certified.
        "operative_rule_fidelity_control_mean": (
            round(statistics.fmean(ctl_all), 6) if ctl_all else 1.0
        ),
        "n_fidelity_control_samples": len(ctl_all),
        "storage_rule_at_floor": bool(mean_drift < RULE_DRIFT_FLOOR),
        "selection_path_rule_intact": bool(mean_fid >= OPERATIVE_RULE_FIDELITY_FLOOR),
    }


def build_manifest(seed_results, smoke: bool, started_at: float) -> dict:
    specs = _precondition_specs(smoke)
    pooled = {arm: _pool_arm(seed_results, arm) for arm in ARMS}

    arm_gates = []
    for arm in ARMS:
        p = pooled[arm]
        ctx = {"id": arm, "ruleset": arm.split("_")[-1], "timing": arm.split("_")[1]}
        gate = evaluate_arm_gate(
            arm, ctx, specs,
            measured={
                "leg_c_event_floor": float(p["n_events"]),
                "distractor_encoded_in_active_representation": float(
                    p["encoding_index_mean"]
                ),
                "selection_path_rule_read_live": float(p["n_fidelity_samples"]),
                "operative_rule_fidelity_instrument_sensitive": float(
                    p["operative_rule_fidelity_control_mean"]
                ),
                "chance_baseline_not_saturated": float(p["chance_baseline_mean_intact"]),
            },
        )
        arm_gates.append(gate)

    aggregate = aggregate_arm_gates(arm_gates)
    green_arms = set(aggregate["green_arms"])

    # PASS: dissociability shown in >= 1 GREEN arm. Per the claim, "rule drift at
    # floor while wrong-target selection is elevated" -- with rule integrity verified
    # at the SELECTION PATH, not only at the storage site, so leg 2 cannot wear leg
    # 3's clothes (Duncan 2008). BOTH reads must say intact, and the leg-(c)
    # elevation is the EXCESS over the measured chance baseline.
    dissociating = [
        arm for arm in ARMS
        if arm in green_arms
        and pooled[arm]["n_both_reads_intact_events"] > 0
        and pooled[arm]["storage_rule_at_floor"]
        and pooled[arm]["selection_path_rule_intact"]
        and pooled[arm]["wrong_target_excess_both_intact"] >= WRONG_TARGET_EXCESS_MIN
    ]
    dissociation_found = bool(dissociating)

    if not aggregate["non_degenerate"]:
        outcome = "FAIL"
        direction = "non_contributory"
        interpretation_label = "substrate_not_ready_requeue"
    elif dissociation_found:
        outcome = "PASS"
        direction = "supports"
        interpretation_label = "behavioural_capture_dissociated_from_rule_corruption"
    else:
        outcome = "FAIL"
        direction = "does_not_support"
        interpretation_label = "no_behavioural_capture_with_operative_rule_intact"

    criteria_by_arm = {arm: [f"{arm}::dissociation"] for arm in ARMS}
    criteria_nd = arm_criteria_non_degenerate(criteria_by_arm, aggregate)

    run_id = f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3"
    arm_results = []
    for r in seed_results:
        for arm, row in r.get("arms", {}).items():
            row_copy = dict(row)
            row_copy["seed"] = r["seed"]
            arm_results.append(row_copy)

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_iso(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-467": direction},
        "non_degenerate": bool(aggregate["non_degenerate"]),
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "per_arm_gate": aggregate["per_arm_gate"],
        "interpretation": {
            "label": interpretation_label,
            "combination_rule": (
                "PASS = dissociability in >= 1 GREEN arm, where an arm dissociates iff "
                "(storage-site rule drift < RULE_DRIFT_FLOOR) AND (selection-path "
                "operative_rule_fidelity >= OPERATIVE_RULE_FIDELITY_FLOOR) AND "
                "(wrong_target_rate among both-reads-intact events MINUS the measured "
                "chance baseline >= WRONG_TARGET_EXCESS_MIN). An arm's gate being RED "
                "excludes it from scoring but NEVER vacates a green arm "
                "(aggregate non_degenerate = ANY arm green)."
            ),
            "preconditions": aggregate["adjudication_preconditions"],
            "preconditions_scope_note": aggregate.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_nd,
            "dissociating_arms": dissociating,
        },
        "thresholds": {
            "rule_drift_floor": RULE_DRIFT_FLOOR,
            "operative_rule_fidelity_floor": OPERATIVE_RULE_FIDELITY_FLOOR,
            "wrong_target_excess_min": WRONG_TARGET_EXCESS_MIN,
            "min_events_pooled": SMOKE_MIN_EVENTS if smoke else MIN_EVENTS_POOLED,
            "encoding_index_floor": ENCODING_INDEX_FLOOR,
            "chance_baseline_ceiling": CHANCE_BASELINE_CEILING,
        },
        "n_seeds": len(seed_results),
        "n_seeds_pass": sum(1 for r in seed_results if r.get("pass")),
        "smoke": smoke,
        "arm_results": arm_results,
        "pooled_by_arm": pooled,
        "metrics": {"per_seed": seed_results},
        "notes": (
            "MECH-467 three-leg distractor battery, REDESIGN superseding V3-EXQ-874 "
            "(0/0 leg-(c) denominator in all 6 cells). Three changes, each traceable: "
            "(1) leg (c) gets a real event denominator -- a 6x6 arena at ~3x resource "
            "density over a 900-tick window, because a pre-authoring probe showed the "
            "874 agent moved on only ~5% of ticks and visited ~10 cells in 300, so a "
            "longer horizon alone multiplies a ~0 rate by a constant; a pre-registered "
            "event floor now gates leg (c), closing the gap the claim's leg-(a)-only "
            "guard left open. (2) leg (b) is read at the SELECTION PATH (Spearman "
            "fidelity of the per-candidate goal-score ranking under the current vs the "
            "warmup-established z_goal) as well as the storage site, because Duncan "
            "2008's goal-neglect mechanism means a storage-site-only read lets plain "
            "rule corruption produce leg 3's exact signature. (3) leg (a) becomes an "
            "ENCODING-SIDE index (z_world displacement under distractor-field ablation) "
            "because registered-then-suppressed and never-encoded give the same "
            "proximity reading (Gaspelin & Luck 2018), so the old guard could "
            "self-route substrate_not_ready on a substrate that handled the distractor "
            "correctly. Arms cross the claim's timing axis (internal_planning / "
            "internal_replay; the during-commitment arm stays EXCLUDED) with a Duncan "
            "rule-set-complexity axis (2 vs 4 SD-049 resource types). Diagnostic over "
            "existing substrate -- no new attention module, per the standing "
            "containment rule. MECH-262 continues to own leg 2."
        ),
    }
    return manifest


def _full_config() -> dict:
    return {
        "env_p0_kwargs": {r: _env_kwargs(r, False) for r in RULESETS},
        "env_target_kwargs": {r: _env_kwargs(r, True) for r in RULESETS},
        "agent_kwargs": {
            "body_obs_dim": 12,
            "action_dim": 4,
            "use_dacc": True,
            "use_salience_coordinator": True,
            "use_lateral_pfc_analog": True,
            "z_goal_enabled": True,
            "beta_gate_bistable": False,
        },
        "arms": {
            arm: {
                "operating_mode": TIMINGS[arm.split("_")[1]],
                "ruleset": arm.split("_")[-1],
            }
            for arm in ARMS
        },
        "schedule": {
            "p0_budget": P0_BUDGET,
            "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
            "n_warmup_steps": N_WARMUP_STEPS,
            "n_eval_steps": N_EVAL_STEPS,
            "enc_probe_every": ENC_PROBE_EVERY,
            "max_episode_steps": MAX_EPISODE_STEPS,
        },
        "thresholds": {
            "rule_drift_floor": RULE_DRIFT_FLOOR,
            "operative_rule_fidelity_floor": OPERATIVE_RULE_FIDELITY_FLOOR,
            "wrong_target_excess_min": WRONG_TARGET_EXCESS_MIN,
            "min_events_pooled": MIN_EVENTS_POOLED,
            "encoding_index_floor": ENCODING_INDEX_FLOOR,
            "chance_baseline_ceiling": CHANCE_BASELINE_CEILING,
        },
    }


def main(smoke: bool):
    device = torch.device("cpu")
    seeds = SEEDS[:1] if smoke else SEEDS
    started_at = time.perf_counter()

    # Design-time refusal: no arm may carry a precondition it cannot satisfy from its
    # own pre-registered config. Every threshold here is measured at run time from
    # quantities no pre-registered constant bounds, so no structural_max/min applies
    # and this asserts the specs are at least all applicable and mutually satisfiable.
    assert_no_structurally_unsatisfiable_gate(
        _precondition_specs(smoke),
        [
            {"id": arm, "ruleset": arm.split("_")[-1], "timing": arm.split("_")[1]}
            for arm in ARMS
        ],
    )

    zg_acc = ZGoalStreamAccumulator()
    seed_results = [run_seed(s, device, smoke, zg_acc) for s in seeds]
    manifest = build_manifest(seed_results, smoke, started_at)

    print(f"=== {QUEUE_ID} {EXPERIMENT_TYPE} ===", flush=True)
    # n_seeds_pass counts seeds with a PRE-GATE dissociating cell; the run-level
    # outcome additionally requires that cell's arm to have passed its precondition
    # gate on POOLED counts, so the two can legitimately disagree. Label it so the
    # disagreement does not read as a contradiction.
    green_arm_ids = [g.get("arm") for g in manifest["per_arm_gate"].get("green", [])]
    print(
        f"outcome: {manifest['outcome']}"
        f" (pre-gate: {manifest['n_seeds_pass']}/{manifest['n_seeds']} seeds have a"
        f" dissociating cell) non_degenerate={manifest['non_degenerate']}"
        f" green_arms={green_arm_ids}"
        f" dissociating_arms={manifest['interpretation']['dissociating_arms']}",
        flush=True,
    )
    for arm in ARMS:
        p = manifest["pooled_by_arm"][arm]
        print(
            f"  {arm}: events={p['n_events']} both_intact={p['n_both_reads_intact_events']}"
            f" excess={p['wrong_target_excess_both_intact']:.4f}"
            f" chance={p['chance_baseline_mean_intact']:.4f}"
            f" fidelity={p['operative_rule_fidelity_mean']:.4f}"
            f" fid_control={p['operative_rule_fidelity_control_mean']:.4f}"
            f" drift={p['mean_final_rule_drift']:.4f}"
            f" enc={p['encoding_index_mean']:.4f} (n_enc={p['n_encoding_samples']})"
            f" latched={p['n_latched_ticks']}",
            flush=True,
        )

    if smoke:
        # Smoke assertions: the decisive readouts must be non-trivially engaged
        # BEFORE committing to the full 12-cell grid (the V3-EXQ-475b lesson --
        # a structural zero on an evidence run's decisive readout is exactly as
        # expensive to discover after a multi-hour run as on a diagnostic).
        tot_events = sum(p["n_events"] for p in manifest["pooled_by_arm"].values())
        tot_enc = sum(p["n_encoding_samples"] for p in manifest["pooled_by_arm"].values())
        tot_fid = sum(p["n_fidelity_samples"] for p in manifest["pooled_by_arm"].values())
        ctl_means = [
            p["operative_rule_fidelity_control_mean"]
            for p in manifest["pooled_by_arm"].values()
            if p["n_fidelity_samples"] > 0
        ]
        print(
            f"[smoke] decisive-readout engagement: consumption_events={tot_events} "
            f"encoding_samples={tot_enc} fidelity_samples={tot_fid} "
            f"fidelity_control_means={[round(v, 4) for v in ctl_means]}",
            flush=True,
        )
        assert tot_enc > 0, "leg (a) encoding-side index never sampled"
        assert tot_fid > 0, "leg (b) selection-path rule read never sampled"
        assert tot_events > 0, "leg (c) consumption-event denominator still zero"
        print("[smoke] OK", flush=True)
        return None

    out_path = write_flat_manifest(
        manifest,
        out_dir=None,
        dry_run=False,
        config=_full_config(),
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=started_at,
        z_goal_stream_stats=zg_acc.stats(),
    )
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
        dry_run=args.dry_run,
    )
    sys.exit(0)
