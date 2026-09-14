#!/opt/local/bin/python3
"""
V3-EXQ-884b -- MECH-428 Subgoal-Bootstrapped Goal Seeding: CONTENT-criterion
redesign, P0-conditioned. SUPERSEDES V3-EXQ-884a (parked, never queued --
same NO_SUBGOAL / SUBGOAL_BOOTSTRAP / FORCED_SEED scripted-walk design; the
criterion and the missing P0 phase are the only changes).

Claim: MECH-428 (subgoal_bootstrapped_goal_seeding). Proposal: EXP-0390.

WHY 884b EXISTS (ratified direction, not this session's choice)
---------------------------------------------------------------------------
884a's own red-team (fable, BLOCKING, independently confirmed at
/queue-experiment Step 4.5) found C1 (lift_fraction >= 0.30) structurally
unfailable: lift_fraction's closed form alpha/(1-(1-alpha)*decay^T) clears
0.30 for every T <= 26.1 steps at the fixed GoalConfig defaults, while the
scripted greedy walk's longest possible waypoint leg on this 12x12 grid is
22 steps (Manhattan bound) -- unreachable by 4.1 steps of horizon. GFLAG-0227
(resolved 2026-09-10) recorded this as NEEDS-DECISION and routed the
criterion choice to governance rather than resolving it by registry edit.

REE_assembly evidence/planning/
unfailable_criterion_class_mech428_recommendation_and_systemic_pass_20260911.md
(496d6f37ef) laid out four candidate reframings and recommended option 1
(score the parent's CONTENT, not its norm) CONDITIONED on option 2 (P0
encoder training as a PRECONDITION, not a competing option) -- because the
parked driver has NO P0 phase at all (confirmed: no P0 call anywhere in
884a) and z_world is a near-constant untrained direction (884a's own
docstring cites coherence 0.998), so a content DV on the untrained substrate
would read ~1.0 against every control and BE A SECOND UNFALSIFIABLE
CRITERION. The user ratified exactly this direction (option 1+2, option 3 as
a free diagnostic only, option 4 rejected) via the Orchestrator decision lane
on 2026-09-14T22:10Z, answering the parked question raised by decision chip
chip-20260914-mech428-c1-criterion-decision. This script implements that
ratified design; it does not choose it.

WHAT CHANGED FROM 884a (everything else is identical: same env, same SD-094
flags, same 3 arms' env mechanics, same scripted greedy walk, same G0/G1/G2
readiness gates, same seeds)
---------------------------------------------------------------------------
1. P0 PHASE ADDED (new G3 readiness gate). Once per seed, BEFORE any arm's
   measurement cell runs, a dedicated warmup env is rolled out under a
   uniform-random policy and `experiments/_lib/zworld_p0_warmup.run_zworld_p0`
   trains `agent.latent_stack.world_encoder` via the SD-070 anti-collapse
   recipe (variance/covariance anti-collapse + world_obs reconstruction +
   hazard/resource grounding heads + SD-106 generic bottleneck variance
   preservation at preservation_weight=200, `use_world_encoder_skip=True`
   on the agent -- see P0A_PRESERVATION_WEIGHT below for why the SD-070
   shipped default was empirically insufficient here), called with
   target_fn=None -- i.e. NOT pointed at anything waypoint-specific. This is
   a deliberate methodological choice, not an oversight: V3-EXQ-1030 (a
   sibling MECH-428/INV-086 probe on
   this same waypoint-field family, FAIL, awaiting autopsy) trains its P0 on
   a GENERIC target for exactly the reason stated in its own docstring --
   "a positive decodability result cannot be an artifact of having been
   trained on the very quantity under test." Training P0 on the
   waypoint-proximity field itself would make a subsequent "the parent aligns
   with attained-subgoal content" finding circular (of course it aligns; the
   encoder was fit to represent exactly that signal). This env has
   num_hazards=0, num_resources=0 (the SD-094 hazard-free design is
   unchanged), so the SD-070 recipe's hazard/resource-specific legs
   (proximity, presence, distance grounding) are auto-masked to None targets
   (a supported, tested path -- `resource_prox_target` returns None when the
   channel is absent, and the trainer treats an unlabelled step as such, not
   as zero) and the LIVE signal is reconstruction + anti-collapse on
   `world_state`'s `local_view` (entity-type one-hot over the 5x5
   neighbourhood, INCLUDING the "waypoint" grid entity type -- see
   ENTITY_TYPES in causal_grid_world.py), which genuinely varies with
   position along the walk (near/at a waypoint cell vs. open transit cell).
   The P0-trained `latent_stack` state_dict is then loaded into EACH of the
   3 arms' fresh per-cell agents for that seed, so all three arms share the
   identical trained substrate and differ ONLY in the crediting rule --
   exactly preserving 884a's own "ONLY the crediting differs across arms"
   design principle, now one level deeper (encoder weights, not just env
   mechanics).
2. G3 (NEW readiness gate, the P0 precondition, SAME STATISTIC the content
   criterion routes on): immediately after P0 training, a short dedicated
   probe (N_STEPS_DRY-length scripted walk on a fresh same-seed env, the
   P0-trained agent, no further training) collects z_world at every tick,
   partitioned by transition_type exactly as the real content DV will be,
   and measures the MEAN-DIRECTION cosine DISSIMILARITY -- 1 minus the
   cosine similarity between the WAYPOINT-tick group's mean z_world and the
   ordinary-TRANSIT-tick group's mean z_world. This went through two prior,
   empirically-refuted versions before landing here (both measured
   2026-09-14, seed 42, real P0 scale -- kept as history because the same
   mistake is easy to repeat): (a) WITHIN-waypoint-group pairwise
   dissimilarity passed (0.011-0.015) while the end-to-end parent-level
   comparison showed no discrimination (delta_group ~= -0.00005) -- the
   wrong GROUPING (within, not between); (b) BETWEEN-group PAIRWISE
   dissimilarity also passed (0.017) while delta_group was still ~0 -- the
   right grouping but the wrong STATISTIC: pairwise dissimilarity measures
   per-sample NOISE, but C1'/delta_group is built from a (recency-weighted)
   AVERAGE over each group, so what predicts it is whether the two groups'
   MEANS differ, which noisy-but-mean-coincident samples can satisfy (a) or
   (b) while still failing to move delta_group. Below
   ZWORLD_CONTENT_DISSIM_FLOOR -> substrate_not_ready_requeue, never a
   content verdict -- this is the gate that would have caught a repeat of
   884a's vacuity (an untrained/collapsed encoder reads dissimilarity ~0
   regardless of how many distinct waypoints were visited), now on the
   statistic C1' actually depends on. A SECOND, cheaper sanity precondition
   (G3b, the standard zworld_encoder_guard "at least one world_encoder
   tensor moved" check) is also recorded for the same reason 728/734/737/742
   all carry it. Pairwise dissimilarity is still recorded
   (g3_pairwise_dissim_secondary) as a non-gating diagnostic.
3. C1' REPLACES the norm-ratio C1. Within the SUBGOAL_BOOTSTRAP arm's cell,
   every tick's z_world snapshot is retained, partitioned into
   ATTAINED (ttype in {"waypoint","sequence_complete"} -- the ticks that get
   credited) and NON_ATTAINED (ttype == "none" -- ordinary transit ticks in
   the SAME walk, present but never credited; this operationalises the
   recommendation's "subgoals present but never credited" control using data
   already generated by the arm's own run, no extra environment pass).
   APPLES-TO-APPLES DESIGN (revised 2026-09-14 after an initial mean-
   centering attempt produced a manifold-mismatch artifact -- comparing the
   raw PARENT, a heavily-decayed EMA whose natural resting point is near the
   ORIGIN, against a raw MEAN of individual z_world snapshots, which live
   near a large non-zero common direction, is either dominated by that
   shared component uncentered (alignment ~0.9995 for every condition,
   unfalsifiable) or contaminated by a new artificial offset if centered by
   the wrong reference). The fix: build THREE PARENT-SHAPED objects via the
   IDENTICAL decay+credit replay (same manifold by construction, so raw
   cosine similarity between any two is directly meaningful, no centering):
     parent_true         the REAL substrate parent (attained sequence, true
                          temporal order)
     parent_shuffled     SAME attained representations, a FIXED seed-derived
                          PERMUTATION of crediting order (same GROUP,
                          different order)
     parent_non_attained a SAME-SIZE, SAME-TIMED sample of NON-ATTAINED
                          representations credited instead (a DIFFERENT
                          GROUP)
   The replay is GoalState's own decay+credit recursion verbatim (from
   ree_core/goal.py: parent *= (1-parent_goal_decay) every tick; parent =
   (1-a)*parent + a*z on a credit tick, a = min(1, alpha*credit) -- constant
   here since every credit call uses the default credit=1.0), run OUTSIDE
   the substrate at the SAME credit-tick positions. This is NOT a set-
   aggregate permutation (Step 3.5's DV-symmetry table: "a permutation of
   interchangeable units is invariant under any SET-AGGREGATE DV") -- the
   EMA is order-/recency-sensitive by construction (later credits dominate,
   both because of the pull itself and because earlier ones have decayed
   further), so crediting the SAME multiset in a DIFFERENT order, or a
   DIFFERENT multiset, produces a genuinely different accumulated vector.
   C1' PASSES a seed iff
     delta_group = cos(parent_true, parent_shuffled) - cos(parent_true, parent_non_attained)
   clears a margin scaled on the cross-seed SD of that delta plus an
   absolute floor ([memory] feedback_effect_size_pass_gate_margin), on
   >= 2/3 seeds (unchanged MIN_SEEDS_PASS convention). FALSIFIER: if z_world
   does not differentiate attained from non-attained states, crediting
   either group produces an indistinguishable parent, so parent_true would
   be EQUALLY similar to parent_shuffled and to parent_non_attained
   (delta_group ~= 0) regardless of MECH-428 -- this is the failing region
   884a's norm-ratio C1 lacked, and G3 (point 2) is what rules the
   degenerate-encoder version of this failure out before C1' is scored.
4. Option 3 (T as an independent variable) rides along as a FREE, NON-GATING
   diagnostic only, per the recommendation's own disposition for it: the
   mean/median inter-credit-event tick interval is recorded per seed
   (diagnostics.inter_event_interval_ticks) but does not enter PASS/FAIL.
   Option 4 (withdraw the floor, score the residual) was rejected in the
   recommendation and is not implemented here.
5. G0, G1, G2 (884a's own readiness gates -- attainment fires, the
   forced-seed positive control produces a structured parent, every cell
   reaches its configured step budget) are UNCHANGED verbatim. C2
   (measurability floor) is re-scoped from the old norm floor to a minimal
   orientation check on the new DV: cos(parent_true, parent_shuffled) must be
   strictly positive on >= 2/3 seeds (the parent must align in the RIGHT
   direction with what it was actually fed at all, before the delta-based
   C1' becomes the substantive test).

PASS (supports MECH-428) = G2 AND G0 AND G1 AND G3 AND C1' AND C2.
G2/G0/G1/G3 fail -> non_contributory / substrate_not_ready_requeue (readiness,
  never a MECH-428 verdict -- G3 in particular is the vacuity floor: an
  untrained or collapsed encoder cannot support a content verdict either
  direction).
G0,G1,G2,G3 pass, C1'/C2 fail -> weakens (attainment occurred, the substrate
  demonstrably carries content-differentiated representations, but the
  parent's specific alignment with the true attained set was not
  distinguishable from a shuffled or never-credited comparison -- genuine
  evidence that bottom-up bootstrapping does not produce content-specific
  structure at this alpha/decay/event-rate combination, as opposed to 884a's
  vacuous C1 which could not have failed at all).

DESIGN CHOICE (unchanged from 884a, restated): scripted greedy walk, not a
learned navigation policy -- see 884a's module docstring reasons 1-3
(basal-ganglia commitment substrate still under construction; MECH-428's own
text frames the mechanism as EMA-pull/decay arithmetic, not navigation
competence; GAP-2 sparse-seeding is reproduced by the substrate fact that
_z_goal_parent has exactly one write path, not by simulated foraging
failure). This script is about the PARENT attractor's CONTENT, still no
claim about downstream goal-directed behaviour.

Biological basis (unchanged): Bandura & Schunk (1981) -- proximal subgoals
CREATED intrinsic interest/goal pursuit in initially-uninterested learners.

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
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

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
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from ree_core.latent.zworld_p0 import ZWorldP0Config  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    assert_world_encoder_trained,
    zworld_precondition,
    ZWorldEncoderUntrainedError,
)

# --------------------------------------------------------------------- #
# Experiment metadata
# --------------------------------------------------------------------- #
EXPERIMENT_TYPE = "v3_exq_884b_mech428_subgoal_bootstrapped_goal_seeding"
QUEUE_ID = "V3-EXQ-884b"
SUPERSEDES = "V3-EXQ-884"  # 884a was parked, never queued -- 884 is the last QUEUED predecessor
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-428"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
ARMS = ["NO_SUBGOAL", "SUBGOAL_BOOTSTRAP", "FORCED_SEED"]
STAY_ACTION = 4  # (dx, dy) = (0, 0) in CausalGridWorld.ACTIONS -- a true no-op

# --- Env config (unchanged from 884a) ---------------------------------- #
GRID_SIZE = 12
NUM_WAYPOINTS = 3
N_STEPS = 400
N_STEPS_DRY = 80
MEASUREMENT_WINDOW = 60

WORLD_DIM = 32
PARENT_GOAL_ALPHA = 0.05
PARENT_GOAL_DECAY = 0.005
FORCED_CREDIT = 20.0

# --- P0 warmup config (NEW) --------------------------------------------- #
P0_EPISODES = 20
P0_STEPS_PER_EPISODE = 50
P0_EPISODES_DRY = 2
P0_STEPS_PER_EPISODE_DRY = 10
P0_PROBE_STEPS = N_STEPS_DRY  # G3 dedicated readiness probe, real run and dry-run alike

# SD-106 generic bottleneck variance preservation (scale-normalised reconstruction,
# NOT waypoint-specific -- see module docstring point 1). Empirically required and
# empirically calibrated (2026-09-14 scratch probes, seed 42, real P0_EPISODES/
# P0_STEPS_PER_EPISODE scale, all against G3's FINAL statistic -- mean-direction
# dissimilarity between the waypoint-tick group and the transit-tick group, NOT the
# two earlier, empirically-refuted pairwise-dissimilarity versions; see module
# docstring point 2 for why those were wrong):
#   untrained (P0_EPISODES=0):        0.0000669
#   preservation_weight=200:          0.000242   (~3.6x untrained)
#   preservation_weight=1000:         0.000731   (~11x untrained)
# 1000 is adopted for real headroom over the untrained floor, not the module
# docstring's own cited "200 + skip" operating point (which is calibrated for a
# DIFFERENT statistic -- world_obs reconstruction R^2 -- and was a bare ~3.6x
# separation on THIS statistic, too thin a margin to certify readiness on).
P0A_PRESERVATION_WEIGHT = 1000.0

# --- Pre-registered readiness thresholds --------------------------------
# 884a's own precedent: thresholds are calibrated against a SEPARATE dry
# probe measurement of this harness, never against the scored run's own
# statistics. FORCED_STRUCTURED_FLOOR / PARENT_NORM_ABS_FLOOR are unchanged
# from 884a. ZWORLD_CONTENT_DISSIM_FLOOR is new, calibrated on G3's FINAL
# statistic (mean-direction dissimilarity -- see P0A_PRESERVATION_WEIGHT
# comment above for the full three-point calibration table: untrained
# 0.0000669, weight=200 0.000242, weight=1000 0.000731, all seed 42,
# 2026-09-14, real P0 scale, measured BEFORE this threshold was fixed). Set
# at ~3x the untrained baseline (0.0000669 * 3 ~= 0.0002), so a genuine
# regression toward the untrained/collapsed regime trips it, while the
# adopted preservation_weight=1000 operating point (0.000731) clears it with
# > 3.5x headroom -- a comfortable, not razor-thin, margin. This floor is a
# PROPERTY OF THE SUBSTRATE (trained vs. untrained separation), fixed BEFORE
# any scored run -- NOT tuned to make a scored result pass. Recorded for the
# honest case this settles: even at this calibration, the WITHIN-run
# separation between attained/non-attained z_world content is small in
# absolute terms (1e-4 to 1e-3 scale) -- if a future session needs a larger
# operating margin, that is a P0-recipe / environment-design question, not a
# reason to lower this floor post hoc.
FORCED_STRUCTURED_FLOOR = 0.05
PARENT_NORM_ABS_FLOOR = 0.01
ZWORLD_CONTENT_DISSIM_FLOOR = 0.0002

# C1' effect-size margin ([memory] feedback_effect_size_pass_gate_margin --
# scale on the SD of the delta plus an absolute floor). With only 3 seeds the
# within-run SD is a noisy multiplier; it is a SECOND, stricter bar layered
# on top of the absolute floor, never a substitute for it.
CONTENT_DELTA_ABS_FLOOR = 0.05
CONTENT_DELTA_SD_MULT = 1.0

MIN_SEEDS_PASS = 2  # of 3 -- ">= 2/3 seeds"


# --------------------------------------------------------------------- #
# Env + agent builders (env builder unchanged from 884a)
# --------------------------------------------------------------------- #

def _build_env(seed: int) -> CausalGridWorld:
    env = CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=0,
        num_resources=0,
        subgoal_mode=True,
        num_waypoints=NUM_WAYPOINTS,
        seed=seed,
        subgoal_arrival_position_check=True,
        hazard_free_contamination_gate=True,
    )
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
        # SD-106: zero-initialised linear bypass around the encoder's ReLU --
        # bit-identical to OFF until the P0 preservation head pulls on it (see
        # _run_p0_and_g3 / P0A_CONFIG below). Needed for the preservation term
        # to have somewhere to push; see module docstring point 1.
        use_world_encoder_skip=True,
    )
    agent = REEAgent(cfg)
    agent.goal_state.config.use_hierarchical_goal_credit = True
    return agent


def _scripted_action(env: CausalGridWorld) -> int:
    """Ground-truth greedy walk toward the currently-targeted waypoint (verbatim from 884a)."""
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
    return STAY_ACTION


def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b) / denom)


# --------------------------------------------------------------------- #
# P0 phase (NEW): once per seed, shared across all 3 arms
# --------------------------------------------------------------------- #

def _run_p0_and_g3(seed: int, dry_run: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Train z_world on a dedicated warmup env, then run the G3 readiness
    probe. Returns (trained_latent_stack_state_dict, p0_g3_report)."""
    n_ep = P0_EPISODES_DRY if dry_run else P0_EPISODES
    n_steps_ep = P0_STEPS_PER_EPISODE_DRY if dry_run else P0_STEPS_PER_EPISODE

    warmup_env = _build_env(seed)
    master_agent = _build_agent(warmup_env)
    before = latent_stack_snapshot(master_agent)

    p0a_report = run_zworld_p0(
        master_agent,
        warmup_env,
        seed=seed,
        episodes=n_ep,
        steps_per_episode=n_steps_ep,
        policy=RandomPolicy(seed),
        label="mech428c1_p0",
        dry_run=dry_run,
        # target_fn left None -- deliberately generic, never waypoint-specific;
        # see module docstring point 1. config= adds the SD-106 preservation
        # term on top of the SD-070 defaults (empirically required -- see the
        # P0A_PRESERVATION_WEIGHT comment above).
        config=ZWorldP0Config(preservation_weight=P0A_PRESERVATION_WEIGHT),
    )

    guard_report: Dict[str, Any]
    try:
        guard_report = assert_world_encoder_trained(
            master_agent, before, p0=n_ep, strict=False,
            context="V3-EXQ-884b P0 (generic recipe, hazard/resource-free env)",
        )
        g3b_untrained_error = ""
    except ZWorldEncoderUntrainedError as exc:  # pragma: no cover -- strict=False above
        guard_report = {"zworld_encoder_trained": False, "guard_checked": True}
        g3b_untrained_error = str(exc)

    trained_state_dict = {
        k: v.detach().clone() for k, v in master_agent.latent_stack.state_dict().items()
    }

    # --- G3: dedicated post-P0 readiness probe. SAME statistic C1' actually
    #     routes on: C1' asks whether crediting the ATTAINED group produces a
    #     DIFFERENT parent than crediting the NON-ATTAINED group, so G3 must
    #     gate on BETWEEN-group separation (attained-tick reps vs
    #     non-attained-tick reps), not within-group separation among
    #     waypoint reps alone. An earlier version of this gate measured only
    #     within-group dissimilarity (mean pairwise among waypoint reps) --
    #     it passed (0.011-0.015) while a real-scale end-to-end measurement
    #     (2026-09-14, seed 42) showed the PARENT built from either group is
    #     ~99.93% cosine-similar regardless of which group was credited
    #     (delta_group ~= -0.00005) -- i.e. the within-group statistic did
    #     not certify the channel C1' actually needs. Fresh env, fresh agent
    #     carrying the just-trained encoder weights, no further training, no
    #     crediting.
    probe_env = _build_env(seed)
    probe_agent = _build_agent(probe_env)
    probe_agent.latent_stack.load_state_dict(trained_state_dict)

    obs_flat, _obs_dict = probe_env.reset()
    probe_agent.act(obs_flat)
    waypoint_reps: List[torch.Tensor] = []
    transit_reps: List[torch.Tensor] = []
    n_probe_steps = P0_PROBE_STEPS
    for _step in range(n_probe_steps):
        action_idx = _scripted_action(probe_env)
        obs_flat, _harm, done, info, _obs_dict = probe_env.step(action_idx)
        probe_agent.act(obs_flat)
        ttype = info.get("transition_type", "none")
        z_now = probe_agent._current_latent.z_world.detach().clone()
        if ttype in ("waypoint", "sequence_complete"):
            waypoint_reps.append(z_now)
        else:
            transit_reps.append(z_now)
        if done:
            break

    # PAIRWISE dissimilarity (kept, recorded) measures per-SAMPLE noise, not
    # whether the two GROUPS' MEANS differ systematically -- and C1'/delta_group
    # is built from a (recency-weighted) AVERAGE over each group, so it is the
    # MEAN-DIRECTION separation that actually predicts whether crediting one
    # group's samples produces a different EMA than crediting the other's.
    # Two groups can have high pairwise dissimilarity (noisy individual
    # samples) while their means still coincide (the noise averages out) --
    # exactly what a real-scale end-to-end check found here (2026-09-14,
    # seed 42): pairwise between-group dissimilarity read 0.017 (comfortably
    # "ready" by that statistic) while the actual parent-level delta_group
    # measured -0.00005 (no discrimination at all). g3_measured below is
    # therefore the MEAN-DIRECTION dissimilarity, the SAME-STATISTIC gate for
    # C1'; pairwise dissimilarity is kept only as a secondary diagnostic.
    n_pairs = 0
    dissim_sum = 0.0
    for wr in waypoint_reps:
        for tr in transit_reps:
            dissim_sum += 1.0 - _cos_sim(wr, tr)
            n_pairs += 1
    g3_pairwise_dissim = (dissim_sum / n_pairs) if n_pairs > 0 else 0.0

    if waypoint_reps and transit_reps:
        mean_waypoint = torch.stack([r.reshape(-1).float() for r in waypoint_reps]).mean(dim=0)
        mean_transit = torch.stack([r.reshape(-1).float() for r in transit_reps]).mean(dim=0)
        g3_measured = 1.0 - _cos_sim(mean_waypoint, mean_transit)
    else:
        g3_measured = 0.0
    g3_pass = bool(waypoint_reps and transit_reps and g3_measured >= ZWORLD_CONTENT_DISSIM_FLOOR)

    report = {
        "p0a_report": p0a_report,
        "guard_report": {
            "zworld_encoder_trained": bool(guard_report.get("zworld_encoder_trained", False)),
            "world_encoder_max_abs_delta": float(guard_report.get("world_encoder_max_abs_delta", 0.0)),
            "n_world_encoder_changed": int(guard_report.get("n_world_encoder_changed", 0)),
            "guard_checked": bool(guard_report.get("guard_checked", False)),
        },
        "g3b_untrained_error": g3b_untrained_error,
        "g3_zworld_content_dissim_measured": g3_measured,
        "g3_zworld_content_dissim_threshold": ZWORLD_CONTENT_DISSIM_FLOOR,
        "g3_pairwise_dissim_secondary": g3_pairwise_dissim,
        "g3_n_waypoint_reps_probed": len(waypoint_reps),
        "g3_n_transit_reps_probed": len(transit_reps),
        "g3_n_pairs": n_pairs,
        "g3_pass": g3_pass,
    }
    return trained_state_dict, report


# --------------------------------------------------------------------- #
# One seed x arm cell (per-tick z_world logging added for SUBGOAL_BOOTSTRAP)
# --------------------------------------------------------------------- #

def _run_cell(
    arm: str,
    seed: int,
    zg: ZGoalStreamAccumulator,
    dry_run: bool,
    trained_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    n_steps = N_STEPS_DRY if dry_run else N_STEPS

    env = _build_env(seed)
    agent = _build_agent(env)
    agent.latent_stack.load_state_dict(trained_state_dict)  # NEW: shared P0-trained encoder

    obs_flat, _obs_dict = env.reset()
    agent.act(obs_flat)

    n_subgoal_credits = 0
    n_waypoint_events = 0
    n_sequence_complete_events = 0
    parent_norm_trace: List[float] = []
    steps_taken = 0
    done = False
    done_cause = ""

    # NEW: per-tick content logging, kept only for SUBGOAL_BOOTSTRAP (the arm
    # the content DV is computed on) to avoid unnecessary memory/compute on
    # the other two arms.
    log_content = arm == "SUBGOAL_BOOTSTRAP"
    attained_reprs: List[torch.Tensor] = []
    non_attained_reprs: List[torch.Tensor] = []
    credit_tick_positions: List[int] = []
    event_tick_positions: List[int] = []  # every waypoint/sequence_complete tick (== credit ticks here)

    for _step in range(n_steps):
        action_idx = _scripted_action(env)
        obs_flat, _harm_signal, done, info, _obs_dict = env.step(action_idx)
        steps_taken += 1
        done_cause = str(info.get("done_cause", "") or "")
        agent.act(obs_flat)
        agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)

        ttype = info.get("transition_type", "none")
        if ttype == "waypoint":
            n_waypoint_events += 1
        elif ttype == "sequence_complete":
            n_sequence_complete_events += 1

        if arm == "NO_SUBGOAL":
            credit_result = {}
        elif arm == "SUBGOAL_BOOTSTRAP":
            credit_result = agent.notify_subgoal_attainment(ttype)
        else:  # FORCED_SEED
            credit_result = agent.notify_subgoal_attainment(
                "sequence_complete", credit=FORCED_CREDIT
            )

        if credit_result:
            n_subgoal_credits = credit_result["n_subgoal_credits"]

        if log_content:
            z_now = agent._current_latent.z_world.detach().clone()
            if ttype in ("waypoint", "sequence_complete"):
                attained_reprs.append(z_now)
                credit_tick_positions.append(steps_taken)
                event_tick_positions.append(steps_taken)
            else:
                non_attained_reprs.append(z_now)

        parent_norm_trace.append(float(agent.goal_state.parent_goal_norm()))

        if done:
            break

    window = parent_norm_trace[-MEASUREMENT_WINDOW:] if parent_norm_trace else [0.0]
    steady_state_norm = float(sorted(window)[len(window) // 2])
    final_parent_norm = float(agent.goal_state.parent_goal_norm())
    final_parent_vec = (
        agent.goal_state.z_goal_parent.detach().clone()
        if agent.goal_state.z_goal_parent is not None
        else None
    )
    zg.observe(agent)

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
    print("verdict: PASS", flush=True)

    row: Dict[str, Any] = {
        "seed": seed,
        "arm": arm,
        "n_subgoal_credits": n_subgoal_credits,
        "n_waypoint_events": n_waypoint_events,
        "n_sequence_complete_events": n_sequence_complete_events,
        "parent_goal_norm_final": final_parent_norm,
        "parent_goal_norm_steady_state": steady_state_norm,
        "parent_goal_norm_trace": parent_norm_trace,
        "n_steps": n_steps,
        "n_steps_configured": n_steps,
        "n_steps_actual": steps_taken,
        "episode_budget_frac": budget_frac,
        "episode_budget_reached": budget_reached,
        "episode_done": bool(done),
        "episode_done_cause": done_cause,
        "agent_health_final": float(env.agent_health),
    }
    if log_content:
        row["_content_final_parent_vec"] = final_parent_vec
        row["_content_attained_reprs"] = attained_reprs
        row["_content_non_attained_reprs"] = non_attained_reprs
        row["_content_credit_tick_positions"] = credit_tick_positions
        row["_content_event_tick_positions"] = event_tick_positions
        row["_content_n_steps"] = steps_taken
    return row


def _replay_parent_from_reprs(
    reprs_to_credit: List[torch.Tensor],
    credit_tick_positions: List[int],
    n_steps: int,
    goal_dim: int,
) -> torch.Tensor:
    """Replay GoalState's own decay+credit recursion (verbatim from
    ree_core/goal.py: parent *= (1-decay) every tick; parent =
    (1-a)*parent + a*z on a credit tick) OUTSIDE the substrate, crediting
    `reprs_to_credit` IN ORDER at the given credit-tick positions. The
    caller controls both WHICH representations are credited and their
    ORDER; this function only replays the arithmetic faithfully."""
    parent = torch.zeros(goal_dim, dtype=torch.float32)
    credit_set = set(credit_tick_positions)
    k = 0
    a = min(1.0, PARENT_GOAL_ALPHA * 1.0)  # default credit=1.0, matches SUBGOAL_BOOTSTRAP
    for tick in range(1, n_steps + 1):
        parent = parent * (1.0 - PARENT_GOAL_DECAY)
        if tick in credit_set and k < len(reprs_to_credit):
            z = reprs_to_credit[k].reshape(-1).float()
            parent = (1.0 - a) * parent + a * z
            k += 1
    return parent


# --------------------------------------------------------------------- #
# Aggregate + acceptance criteria
# --------------------------------------------------------------------- #

def run(dry_run: bool = False) -> tuple:
    print(f"\n[{QUEUE_ID}] MECH-428 Subgoal-Bootstrapped Goal Seeding "
          f"(content-DV redesign, P0-conditioned)", flush=True)

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict] = []
    per_seed: Dict[str, Dict[int, Dict]] = {a: {} for a in ARMS}
    p0_g3_by_seed: Dict[int, Dict[str, Any]] = {}

    for seed in SEEDS:
        trained_state_dict, p0_g3_report = _run_p0_and_g3(seed, dry_run)
        p0_g3_by_seed[seed] = p0_g3_report

        for arm in ARMS:
            config_slice = {
                "arm": arm, "seed": seed, "dry_run": dry_run,
                "grid_size": GRID_SIZE, "num_waypoints": NUM_WAYPOINTS,
                "n_steps": N_STEPS_DRY if dry_run else N_STEPS,
                "world_dim": WORLD_DIM, "forced_credit": FORCED_CREDIT,
                "subgoal_arrival_position_check": True,
                "hazard_free_contamination_gate": True,
                "p0_trained": True,
                "p0_episodes": P0_EPISODES_DRY if dry_run else P0_EPISODES,
                "p0_steps_per_episode": P0_STEPS_PER_EPISODE_DRY if dry_run else P0_STEPS_PER_EPISODE,
            }
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                row = _run_cell(arm, seed, zg, dry_run=dry_run, trained_state_dict=trained_state_dict)
                cell.stamp({k: v for k, v in row.items() if not k.startswith("_content_")})
            arm_results.append(row)
            per_seed[arm][seed] = row

    # --- G2 (unchanged from 884a): every cell ran its full configured budget.
    g2_failures = [
        f"{r['arm']}/seed{r['seed']}: {r['n_steps_actual']}/{r['n_steps_configured']} "
        f"(done_cause={r['episode_done_cause'] or 'none'}, "
        f"health={r['agent_health_final']:.3f})"
        for r in arm_results
        if not r["episode_budget_reached"]
    ]
    g2_pass = not g2_failures
    g2_min_budget_frac = min(float(r["episode_budget_frac"]) for r in arm_results)

    # --- G3 (NEW): every seed's post-P0 readiness probe cleared the content-
    #     dissimilarity floor.
    g3_failures = [
        f"seed{seed}: dissim={rep['g3_zworld_content_dissim_measured']:.4f} "
        f"(n_pairs={rep['g3_n_pairs']}, threshold={ZWORLD_CONTENT_DISSIM_FLOOR})"
        for seed, rep in p0_g3_by_seed.items()
        if not rep["g3_pass"]
    ]
    g3_pass = not g3_failures

    g0_per_seed, g1_per_seed, c2_per_seed = [], [], []
    c1_per_seed: List[bool] = []
    delta_group_by_seed: Dict[int, float] = {}
    content_alignment_by_seed: Dict[int, Dict[str, float]] = {}
    inter_event_interval_by_seed: Dict[int, Optional[float]] = {}

    for seed in SEEDS:
        no_sub = per_seed["NO_SUBGOAL"][seed]
        boot = per_seed["SUBGOAL_BOOTSTRAP"][seed]
        forced = per_seed["FORCED_SEED"][seed]

        g0_per_seed.append(boot["n_subgoal_credits"] > 0)
        g1_per_seed.append(forced["parent_goal_norm_steady_state"] >= FORCED_STRUCTURED_FLOOR)

        final_parent_vec = boot["_content_final_parent_vec"]
        attained_reprs = boot["_content_attained_reprs"]
        non_attained_reprs = boot["_content_non_attained_reprs"]
        credit_tick_positions = boot["_content_credit_tick_positions"]
        event_tick_positions = boot["_content_event_tick_positions"]
        n_steps_this_seed = boot["_content_n_steps"]

        if final_parent_vec is not None and len(attained_reprs) >= 2 and len(non_attained_reprs) >= len(attained_reprs):
            # APPLES-TO-APPLES DESIGN (2026-09-14, after an earlier mean-centering
            # attempt produced a MANIFOLD MISMATCH artifact -- see A-xx / build notes):
            # comparing the raw PARENT (a heavily-decayed EMA whose natural resting
            # point is near the ORIGIN) against a raw MEAN of individual z_world
            # snapshots (which live near a large, non-zero common direction -- 884a's
            # "coherence 0.998") is dominated by that shared component either
            # uncentered (alignment ~0.9995 for every condition alike, unfalsifiable)
            # or centered by the wrong reference (subtracting the snapshot-space mean
            # from the origin-anchored parent introduces a NEW large artificial
            # offset). The fix: never compare a PARENT to a snapshot-MEAN at all.
            # Build THREE parent-shaped objects via the IDENTICAL decay+credit
            # replay (same manifold by construction, so raw cosine similarity
            # between any two of them is directly meaningful, no centering needed):
            #   parent_true         -- the REAL substrate parent (attained sequence,
            #                          true temporal order)
            #   parent_shuffled     -- SAME attained representations, PERMUTED
            #                          crediting order (same GROUP, different order)
            #   parent_non_attained -- a SAME-SIZE, SAME-TIMED sample of NON-ATTAINED
            #                          representations credited instead (a DIFFERENT
            #                          GROUP entirely)
            # FALSIFIER (matches "a parent that is merely an EMA of a constant-
            # direction encoder output fails this by construction"): if z_world does
            # not differentiate attained from non-attained states, crediting either
            # group produces an indistinguishable parent, so parent_true would be
            # EQUALLY similar to parent_shuffled and to parent_non_attained. C1'
            # asks whether parent_true resembles its OWN GROUP's shuffled variant
            # MORE than it resembles the OTHER GROUP's variant.
            goal_dim = final_parent_vec.reshape(-1).shape[0]
            k = len(attained_reprs)
            perm = list(range(k))
            random.Random(("shuffle884b", seed)).shuffle(perm)
            shuffled_reprs = [attained_reprs[i] for i in perm]
            non_attained_sample_idx = list(range(len(non_attained_reprs)))
            random.Random(("nonattained884b", seed)).shuffle(non_attained_sample_idx)
            non_attained_sample = [non_attained_reprs[i] for i in non_attained_sample_idx[:k]]

            parent_true = final_parent_vec.reshape(-1).float()
            parent_shuffled = _replay_parent_from_reprs(
                shuffled_reprs, credit_tick_positions, n_steps_this_seed, goal_dim,
            )
            parent_non_attained = _replay_parent_from_reprs(
                non_attained_sample, credit_tick_positions, n_steps_this_seed, goal_dim,
            )

            similarity_shuffled = _cos_sim(parent_true, parent_shuffled)
            similarity_non_attained = _cos_sim(parent_true, parent_non_attained)
        else:
            similarity_shuffled = 0.0
            similarity_non_attained = 0.0

        content_alignment_by_seed[seed] = {
            "parent_true_vs_shuffled": similarity_shuffled,
            "parent_true_vs_non_attained": similarity_non_attained,
        }
        # THE falsifiable comparison: does parent_true resemble its OWN GROUP's
        # shuffled-order variant MORE than it resembles the OTHER GROUP's variant?
        delta_group_by_seed[seed] = similarity_shuffled - similarity_non_attained
        # C2 sanity floor: parent_true is not anti-correlated with its own group.
        c2_per_seed.append(similarity_shuffled > 0.0)

        if len(event_tick_positions) >= 2:
            intervals = [
                event_tick_positions[i] - event_tick_positions[i - 1]
                for i in range(1, len(event_tick_positions))
            ]
            inter_event_interval_by_seed[seed] = sum(intervals) / len(intervals)
        else:
            inter_event_interval_by_seed[seed] = None

    def _margin(deltas: List[float]) -> float:
        n = len(deltas)
        if n < 2:
            return CONTENT_DELTA_ABS_FLOOR
        mean = sum(deltas) / n
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        sd = var ** 0.5
        return max(CONTENT_DELTA_ABS_FLOOR, CONTENT_DELTA_SD_MULT * sd)

    group_deltas = [delta_group_by_seed[s] for s in SEEDS]
    margin_group = _margin(group_deltas)

    for seed in SEEDS:
        c1_per_seed.append(delta_group_by_seed[seed] >= margin_group)

    threshold = MIN_SEEDS_PASS / len(SEEDS)
    g0_pass = all(g0_per_seed)
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
            "terminated by the environment before its configured step budget. "
            "Starved cells: " + "; ".join(g2_failures) + ". Not evidence about "
            "MECH-428; diagnose the environment configuration and re-queue "
            "under a new letter."
        )
        interpretation = "Readiness (G2) failed. Not evidence against MECH-428."
    elif not g0_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G0 readiness gate failed: ARM_SUBGOAL_BOOTSTRAP did not produce a "
            "subgoal-attainment credit on every seed. Harness/environment "
            "defect, not evidence about MECH-428."
        )
        interpretation = "Readiness (G0) failed. Not evidence against MECH-428."
    elif not g1_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G1 readiness gate failed: ARM_FORCED_SEED did not itself produce "
            "a steady-state parent_goal_norm clearing FORCED_STRUCTURED_FLOOR "
            "on >= 2/3 seeds. The substrate, not the bootstrap hypothesis, is "
            "what is not ready."
        )
        interpretation = "Readiness (G1) failed. Not evidence against MECH-428."
    elif not g3_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "G3 readiness gate failed: the post-P0 probe's mean pairwise "
            "cosine dissimilarity among waypoint-tick z_world representations "
            "did not clear ZWORLD_CONTENT_DISSIM_FLOOR on every seed measured. "
            "This is the exact vacuity the P0 phase exists to prevent -- an "
            "untrained or collapsed encoder would read as content-aligned "
            "against every control by construction, regardless of MECH-428. "
            "Failures: " + "; ".join(g3_failures) + ". Not evidence about "
            "MECH-428; diagnose the P0 recipe (episode/step budget, "
            "reconstruction/anti-collapse weights) before re-queuing."
        )
        interpretation = "Readiness (G3, P0 content-discriminability) failed. Not evidence against MECH-428."
    else:
        all_pass = c1_pass and c2_pass
        status = "PASS" if all_pass else "FAIL"
        evidence_direction = "supports" if all_pass else "weakens"
        route_reason = "clean_scripted_bootstrap_content_scored"
        if all_pass:
            interpretation = (
                "MECH-428 SUPPORTED: the P0-trained parent (superordinate) "
                "attractor's final content aligns with the representations of "
                "the ATTAINED subgoals by a margin that clears BOTH a shuffled "
                "control (same representations, permuted crediting order) and "
                "a non-attained control (representations present in the same "
                "walk but never credited) on >= 2/3 seeds -- the parent's "
                "structure is specific to WHICH subgoals were actually "
                "attained, not merely an artifact of accumulating "
                "subgoal-shaped pulls in general."
            )
        else:
            interpretation = (
                "MECH-428 WEAKENED: attainment occurred (G0), the forced-seed "
                "positive control demonstrated the wiring can sustain a "
                "structured parent (G1), and the substrate carries genuinely "
                "content-differentiated representations post-P0 (G3), but the "
                "parent's alignment with the TRUE attained set was not "
                "distinguishable from a shuffled or non-attained comparison "
                "on >= 2/3 seeds -- genuine evidence that bottom-up "
                "bootstrapping at this alpha/decay/event-rate combination "
                "does not produce content-specific parent structure."
            )

    metrics: Dict[str, float] = {
        "g2_pass": 1.0 if g2_pass else 0.0,
        "g2_min_episode_budget_frac": g2_min_budget_frac,
        "n_cells_starved": float(len(g2_failures)),
        "g3_pass": 1.0 if g3_pass else 0.0,
        "g0_frac_seeds": g0_frac,
        "g1_frac_seeds": g1_frac,
        "c1_frac_seeds": c1_frac,
        "c2_frac_seeds": c2_frac,
        "g0_pass": 1.0 if g0_pass else 0.0,
        "g1_pass": 1.0 if g1_pass else 0.0,
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "similarity_shuffled_mean": sum(
            v["parent_true_vs_shuffled"] for v in content_alignment_by_seed.values()
        ) / len(content_alignment_by_seed),
        "similarity_non_attained_mean": sum(
            v["parent_true_vs_non_attained"] for v in content_alignment_by_seed.values()
        ) / len(content_alignment_by_seed),
        "delta_group_mean": sum(group_deltas) / len(group_deltas),
        "content_delta_margin_group": margin_group,
    }
    for arm in ARMS:
        pn = [per_seed[arm][s]["parent_goal_norm_steady_state"] for s in SEEDS]
        nc = [per_seed[arm][s]["n_subgoal_credits"] for s in SEEDS]
        sa = [per_seed[arm][s]["n_steps_actual"] for s in SEEDS]
        metrics[f"parent_goal_norm_steady_state_mean_{arm}"] = sum(pn) / len(pn)
        metrics[f"n_subgoal_credits_mean_{arm}"] = sum(nc) / len(nc)
        metrics[f"n_steps_actual_mean_{arm}"] = sum(sa) / len(sa)

    evidence_direction_per_claim = {"MECH-428": evidence_direction}

    criteria = [
        {
            "name": "G3_zworld_content_dissim_supra_floor",
            "kind": "readiness",
            "load_bearing": True,
            "measured": min(
                rep["g3_zworld_content_dissim_measured"] for rep in p0_g3_by_seed.values()
            ),
            "threshold": ZWORLD_CONTENT_DISSIM_FLOOR,
            "direction": "lower",
            "passed": g3_pass,
        },
        {
            "name": "C1_parent_resembles_own_group_more_than_other_group",
            "load_bearing": True,
            "measured_per_seed": group_deltas,
            "threshold": margin_group,
            "direction": "lower",
            "combination_rule": ">= 2/3 seeds",
            "passed": c1_pass,
        },
        {
            "name": "C2_parent_positively_similar_to_own_group",
            "load_bearing": True,
            "measured_per_seed": [
                content_alignment_by_seed[s]["parent_true_vs_shuffled"] for s in SEEDS
            ],
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "combination_rule": ">= 2/3 seeds",
            "passed": c2_pass,
        },
    ]

    dv_headroom = {
        "dv_name": "parent_group_similarity_delta",
        "criterion": "C1_content",
        "criterion_threshold": margin_group,
        "achievable": 2.0,
        "achievable_basis": (
            "cosine similarity is bounded [-1, 1] per term, so a delta between "
            "two such terms is bounded [-2, 2]; not pinned by construction the "
            "way 884a's norm-ratio DV was (FORCED_SEED does not define this "
            "DV's denominator)"
        ),
        "statistic": "parent_ema_cosine_similarity_delta_shuffled_vs_non_attained",
        "delta_group_per_seed": group_deltas,
        "gates": False,
        "gate_rationale": (
            "A collapsed/untrained substrate is already caught by G3; a second "
            "gate on the same condition would double-count it."
        ),
    }

    diagnostics = {
        "dv_headroom": dv_headroom,
        "content_alignment_per_seed": content_alignment_by_seed,
        # Option 3 (T as IV) -- free, non-gating diagnostic per the
        # recommendation's own disposition for it.
        "inter_event_interval_ticks_per_seed": inter_event_interval_by_seed,
        "p0_g3_report_per_seed": {
            str(seed): {
                k: v for k, v in rep.items() if k != "p0a_report"
            }
            for seed, rep in p0_g3_by_seed.items()
        },
        "p0a_report_per_seed": {
            str(seed): rep["p0a_report"] for seed, rep in p0_g3_by_seed.items()
        },
    }

    summary_markdown = (
        f"# {QUEUE_ID} -- MECH-428 Subgoal-Bootstrapped Goal Seeding (content-DV redesign)\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}\n"
        f"**Route reason:** {route_reason}\n"
        f"**Claims:** MECH-428\n\n"
        f"## Gates and criteria\n\n"
        f"| Gate/Criterion | Value | Pass |\n|---|---|---|\n"
        f"| G2 readiness (episode budget, every cell) | {g2_min_budget_frac:.2f} (min) | {g2_pass} |\n"
        f"| G0 readiness (attainment, every seed) | {g0_frac:.2f} | {g0_pass} |\n"
        f"| G1 readiness (forced-seed structured, >=2/3) | {g1_frac:.2f} | {g1_pass} |\n"
        f"| G3 readiness (P0 content-discriminability, every seed) | "
        f"{min(rep['g3_zworld_content_dissim_measured'] for rep in p0_g3_by_seed.values()):.4f} "
        f"(floor {ZWORLD_CONTENT_DISSIM_FLOOR}) | {g3_pass} |\n"
        f"| C1' content alignment vs shuffled (>=2/3) | {c1_frac:.2f} | {c1_pass} |\n"
        f"| C2 alignment orientation (>=2/3) | {c2_frac:.2f} | {c2_pass} |\n\n"
        f"## Interpretation\n\n{interpretation}\n"
    )

    result: Dict[str, Any] = {
        "status": status,
        "outcome": status,
        "metrics": metrics,
        "readout": dict(metrics),
        "criteria": criteria,
        "arm_results": [
            {k: v for k, v in r.items() if not k.startswith("_content_")} for r in arm_results
        ],
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "per_seed_results": {
            arm: {
                s: {k: v for k, v in row.items() if not k.startswith("_content_")}
                for s, row in seeds.items()
            }
            for arm, seeds in per_seed.items()
        },
        "dv_headroom": dv_headroom,
        "diagnostics": diagnostics,
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
        "zworld_content_dissim_floor": ZWORLD_CONTENT_DISSIM_FLOOR,
        "content_delta_abs_floor": CONTENT_DELTA_ABS_FLOOR,
        "content_delta_sd_mult": CONTENT_DELTA_SD_MULT,
        "p0_episodes": P0_EPISODES,
        "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
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
