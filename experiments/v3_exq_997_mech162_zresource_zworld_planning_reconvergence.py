"""V3-EXQ-997: MECH-162 z_resource / z_world hippocampal-planning RECONVERGENCE
probe (EVIDENCE).

SLEEP DRIVER: N/A (no sleep loop; single-episode-batch measurement run).

CLAIM TESTED (claims.yaml MECH-162, verbatim):
"z_resource (SD-015, object-identity encoding) and z_world (spatial-contextual
encoding) separate at the feedforward representation stage but must re-converge
at the hippocampal planning stage for goal-directed navigation." The claim's own
notes pose the concrete architectural question: "whether z_resource feeds into
the hippocampal planning module alongside z_world, enabling the planner to reason
about 'where resources are likely to be' given current spatial context."

MECHANISM UNDER TEST (code-confirmed, Step 2.5/2.5a; see below)
-----------------------------------------------------------------
`REEAgent.update_z_goal()` (agent.py ~11374-11511) seeds `GoalState.z_goal` from
EITHER `z_resource` (SD-015, object-identity latent, gated by
`config.latent.use_resource_encoder`) OR `z_world` (full-scene spatial latent,
the legacy fallback) at each benefit-triggered contact -- never both at once.
That `z_goal` is then threaded, every E3 tick, into
`HippocampalModule.propose_trajectories(current_z_goal=...)`
(`REEAgent._e3_tick`, confirmed by source read: `_current_z_goal = goal_state.z_goal
if goal_state.is_active() else None`). Its SOLE consumer inside
`propose_trajectories` is the MECH-293 waking ghost-goal-probe branch
(`_propose_ghost_seeded`), which ranks the MECH-292 `GhostGoalBank` -- a derived
view over the SD-039 dual-trace anchor pool -- via
`Anchor.goal_match(current_z_goal, baseline=...) = cosine(z_goal_snapshot - baseline,
current_z_goal - baseline)` (SD-079 common-mode-centered cosine; clamped to
[0, 1]). This IS the literal "z_resource feeds the hippocampal planning module"
pathway MECH-162 asks about: it is the only call site in the current substrate
where a z_resource-derived (or z_world-derived) tensor reaches trajectory
PROPOSAL as a retrieval CUE over remembered goal-relevant locations, rather than
as a raw scoring input. (V3-EXQ-914/MECH-236 established the SAME `current_z_goal`
-> `propose_trajectories` -> MECH-293 channel exists and is wired end-to-end;
this run reuses that confirmed wiring but asks the ORTHOGONAL question MECH-236
does not: which LATENT seeds z_goal, and does that choice change what the
retrieval cue can discriminate.)

WHY GOAL_MATCH IS THE RIGHT DV, NOT DOWNSTREAM NAVIGATION (Step 2.5a + design choice)
--------------------------------------------------------------------------------------
V3-EXQ-914 (MECH-236, PASS/supports 2026-09-xx) already measured whether the
ghost-probe CHANNEL changes downstream navigation outcome (mean_resource_proximity)
and found the effect small and noisy at 5-seed scale (gaps -0.011/-0.054/+0.071)
-- a real but weak signal, consistent with an untrained-substrate architectural
probe rather than a learned-competence one. MECH-162's question is more specific
and upstream of that noise: does the CONTENT of the retrieval cue (identity vs
position) change what gets matched, independent of whether matching changes
BEHAVIOUR yet. This run therefore reads `goal_match` directly off
`GhostGoalBank.rank()` -- the retrieval-cue computation itself -- rather than
inferring it from trajectory choice, which removes an entire additional
stochastic layer (CEM sampling, E3 selection, env transition) between the
manipulation and the DV.

DESIGN -- 2 ARMS x 5 SEEDS, TYPE-STRATIFIED RETROSPECTIVE RETRIEVAL DISCRIMINATION
------------------------------------------------------------------------------------
Env: CausalGridWorldV2, multi_resource_heterogeneity_enabled=True,
n_resource_types=2, resources RESPAWN on consumption at randomized locations
(env default -- "resources respawn at random locations, so z_world at contact
has no predictive value for future resource locations", agent.py's own SD-015
docstring). NUM_HAZARDS=0 (matches V3-EXQ-914's Step-2.5a finding: an untrained
agent with hazards present dies in ~12 steps/episode, pinning every DV at a
survival-confound floor regardless of the channel under test -- irrelevant here
too, since this run tests a representational property, not harm-survival
tradeoffs). No gradient training anywhere (matches V3-EXQ-914's own justification:
"the claim under test is an ARCHITECTURAL coupling ... untrained predictors are
the correct substrate state" -- MECH-162 asks whether the WIRING exists and
carries type-discriminative content when it reaches the planner, not whether a
trained policy exploits it).

  ARM_ZRESOURCE: use_resource_encoder=True. z_goal seeded from z_resource
                 (object-identity latent) at every genuine contact.
  ARM_ZWORLD:    use_resource_encoder=False. z_goal seeded from z_world (full
                 scene latent) at every genuine contact -- the pre-SD-015
                 legacy default this run treats as the comparison condition,
                 NOT an ablation of goal-seeding itself (both arms have a live,
                 identically-wired MECH-292/293 channel; only the SOURCE latent
                 differs).

Both arms carry the FULL MECH-292/293 prerequisite stack live and IDENTICAL
between arms (use_per_stream_vs, use_event_segmenter, use_invalidation_trigger,
use_anchor_sets, use_sd039_anchor_payload, use_mech292_ghost_bank,
use_mech293_ghost_probes -- all True), `wanting_weight=0.0` (isolates the
MECH-293 channel from the separate VALENCE_WANTING/`update_benefit_salience`
pathway, matching V3-EXQ-914's scoping note), `mech292_goal_match_floor` set to
a maximally permissive value so the RAW discrimination spectrum is visible
rather than only whatever already clears an arbitrary bar (this run cares about
the SPREAD between same-type and different-type matches, not a pass/fail floor
on any single one).

ANCHOR POPULATION (Step 2.5a empirical probe finding -- reused from V3-EXQ-914):
no production call site writes a new SD-039 anchor automatically (agent.py only
ever READS `anchor_set.active_anchors()`; confirmed by source grep, same finding
V3-EXQ-914 documents). Anchors are therefore DRIVER-ISSUED here, at every
GENUINE at-cell resource contact (`obs_dict["resource_type_at_agent"] > 0` --
the authoritative per-tick "agent is standing on a resource of type T" signal;
NOT the post-consumption-cleared cell, and NOT `info["sd049_consumed_type_tag_
this_tick"]`, which V3-EXQ-681 established is the CONSUMED-tag, only available
in info, one step removed from the at-agent obs_dict read this driver uses).
Each write tags the anchor with `goal_payload.z_goal_snapshot =
agent.goal_state.z_goal` (whichever latent just seeded it) and records, ON THE
DRIVER SIDE ONLY (not read back from the substrate), the contact's TRUE resource
type -- `_ANCHOR_TYPE_BY_KEY[anchor.key] = T`. `write_anchor`'s own dual-trace
semantics (anchor_set.py docstring) mean each new write under the same
(scale, stream_mixture) family demotes the PRIOR anchor to inactive rather than
deleting it -- exactly the accumulating "ghost pool" of past goal-relevant
locations MECH-292 ranks over, and `GhostGoalBankConfig` defaults include both
active and inactive anchors in that pool (confirmed default, V3-EXQ-914
docstring), so a full run's worth of past contacts (of BOTH types, at DIFFERENT,
respawn-randomized locations) accumulates for the retrieval query below.

EPISODE-BOUNDARY RESET (substrate-confirmed, code read of `REEAgent.reset()`,
NOT present in the V3-EXQ-914 precedent's own scope): `reset()` unconditionally
wipes the ENTIRE SD-039 anchor set (active + inactive) at every episode
boundary when `use_anchor_sets=True` (agent.py ~3762-3769, via
`hippocampal.reset_anchor_set()`). The accumulating "ghost pool" described
above therefore only ever accumulates WITHIN one episode, never across
episodes -- a discrimination-qualifying tick can only compare anchors written
since the current episode's own start. This is why `N_EPISODES`/
`STEPS_PER_EPISODE` below are set high enough (20 x 80, not V3-EXQ-914's 15 x
60) that dual-type contact within a single episode is reliably reached before
that episode's anchor pool is wiped; see the code comment beside those
constants for the exact figures.

RETRIEVAL QUERY / DECISIVE READOUT (the direct MECH-162 measurement):
At every E3 tick (`ticks["e3_tick"]`, whether or not this tick is itself a
contact tick) once at least one contact has occurred this run, the driver calls
`agent.hippocampal.ghost_goal_bank.rank(agent.goal_state.z_goal)` DIRECTLY (the
identical call `_propose_ghost_seeded` makes internally -- called directly here
for per-entry inspection that the propose_trajectories path does not expose;
`agent.generate_trajectories()` is ALSO run this tick via the standard pipeline,
so `propose_trajectories`'s own diagnostics -- `mech293_mean_goal_match_at_seed`,
`mech293_n_ghost_admitted` -- are captured for free as SECONDARY/CONTEXT
readouts confirming the discrimination measured here actually reaches probe
generation, not just the bank's internal ranking). For each returned entry,
`_ANCHOR_TYPE_BY_KEY[entry.anchor.key]` gives that anchor's TRUE original
contact type; `entry.components["goal_match"]` gives its match SCORE
(the WEIGHTED term -- `mech292_goal_match_weight=1.0` throughout, so this is
identity to the raw SD-079-centered cosine). Entries are split into
SAME_TYPE (anchor's original type == the type of the most recent genuine
contact -- the driver's own tracked "what am I currently wanting") and
DIFF_TYPE (all other typed anchors). A tick contributes to the decisive
readout only when BOTH sets are non-empty (both resource types must already
have been contacted at least once for the comparison to be meaningful).

  per-tick discrimination_delta = mean(SAME_TYPE goal_match) - mean(DIFF_TYPE goal_match)
  per-seed DISCRIM[arm][seed]   = mean(discrimination_delta over qualifying ticks)

PRE-REGISTERED HYPOTHESIS: under ARM_ZRESOURCE, because z_resource is an
object-IDENTITY embedding (SD-015 docstring: "encodes 'what kind of resource is
present' independent of spatial position"), SAME_TYPE anchors -- written at
DIFFERENT, respawn-randomized locations -- should still match the current
identity-seeded z_goal well, so DISCRIM[ZRESOURCE] should be reliably positive.
Under ARM_ZWORLD, z_goal is a raw SCENE embedding pulled toward wherever the
agent physically stood at contact; because resources respawn at randomized
locations (env default, uncorrelated with type), whether a stored anchor's
scene resembles the CURRENT scene is a POSITION question, not a type question,
so DISCRIM[ZWORLD] should be small/noisy relative to DISCRIM[ZRESOURCE]. A null
result (DISCRIM[ZRESOURCE] approx DISCRIM[ZWORLD]) would mean the architecture
already carries this wiring but it produces no representational advantage for
identity-invariant-to-position retrieval -- informative either way.

READINESS PRECONDITIONS (Step 2.5a / P0 readiness-assert -- same statistic the
load-bearing criterion reads, per-arm, MEASURED, self-routes substrate_not_ready
below floor rather than a false MECH-162 verdict):
  R1 TYPE_COVERAGE: per seed, per arm, BOTH resource types genuinely contacted
     (n_contacts_by_type[1] > 0 AND n_contacts_by_type[2] > 0) -- without this
     no discrimination tick is even possible, structurally.
  R2 ENGAGEMENT: per seed, per arm, fraction of post-first-contact E3 ticks that
     produced a qualifying (both-types-present) discrimination reading clears
     ENGAGEMENT_FLOOR.
  Both gated >= SEED_PASS_FRACTION (4/5) seeds per arm; unmet in EITHER arm ->
  whole run self-routes substrate_not_ready_requeue (FAIL, non_contributory),
  never a false weakens.
  R0 MECHANISM-OFF SANITY (single deterministic leg, run once, NOT part of the
     seed x arm grid -- mirrors V3-EXQ-681/626b's forced-contact microdiagnostic
     convention): with z_goal_enabled=False, confirm propose_trajectories's own
     diagnostics read `mech293_reason == "no_z_goal"` on every observed E3 tick,
     proving a genuinely-off channel reads as zero/absent rather than the metric
     being trivially always-nonzero by construction.

ACCEPTANCE CRITERIA (pre-registered; load-bearing statistic = DISCRIM, the SAME
statistic R1/R2 gate on):
  C1 (load-bearing): mean_over_seeds(DISCRIM[ZRESOURCE][s] - DISCRIM[ZWORLD][s])
     >= EFFECT_FLOOR (0.02) AND DISCRIM[ZRESOURCE][s] > DISCRIM[ZWORLD][s] on
     >= SEED_PASS_FRACTION (3/5) seeds.
  PASS = R1 AND R2 (both arms) AND C1. Readiness unmet (R1/R2) -> FAIL /
  non_contributory / substrate_not_ready_requeue. Readiness met, C1 unmet ->
  FAIL / weakens (a genuine result: the channel is live and measurable but does
  not discriminate identity from position the way MECH-162 predicts).

DV-SYMMETRY (Step 3 mandatory declaration, per the manipulation -> DV chain):
The manipulation (which latent seeds z_goal) changes the CONTENT of a vector fed
into a cosine similarity against stored anchor snapshots -- not a uniform
additive constant broadcast across candidates (the V3-EXQ-604c hazard: a
broadcast-scalar manipulation is invisible to an argmax/softmax DV) and not a
monotone rescaling (the DV here is a raw cosine-derived scalar, not a rank/
argmax/order statistic, so monotone-invariance does not apply). The SD-079
common-mode-centering baseline (`observe_goal_cue`) is a SLOW EMA shared across
both same-type and diff-type entries within one tick's query, so it cancels in
the same-type-minus-diff-type SUBTRACTION by construction -- the discrimination
statistic is deliberately a WITHIN-TICK, WITHIN-QUERY contrast for exactly this
reason (any residual common-mode drift affects both terms identically and
cannot manufacture the SAME_TYPE > DIFF_TYPE direction on its own). The one real
risk is that the RESOURCE FIELD ITSELF differs systematically by type in a way
that also differs by position (e.g. type-1 resources cluster spatially) --
env resource placement here is NOT type-clustered (`multi_resource_heterogeneity_
enabled` only changes per-cell resource IDENTITY, not spatial distribution), so
this confound does not apply by construction; recorded here rather than assumed.

GOV-REUSE-1 (Step 2.4): grep confirmed zero manifests, zero prior autopsies,
and zero existing experiment scripts reference MECH-162 anywhere in
REE_assembly/evidence/ or ree-v3/experiments/ (checked 2026-09-03). Not
recoverable -> run. Re-derive brake (Step 2.5b): 0 counting autopsies on
MECH-162 -> not braked.

SUBSTRATE-PATH OVERLAP GATE (Step 2.5c): two OPEN `corrupting` substrate_queue
entries file-match `ree_core/agent.py` at the whole-file level --
`mode-governance-engagement` (affinity-input box clamp / commitment-term
grading for MECH-266 competing-goals MODE SWITCHING) and `SD-082` (SD-033a
trained bias-head / SD-078 rule-pool action-bias propagation). Both concern
subsystems this driver's code paths do not exercise: neither
`mode_conditioning_enabled` nor `operating_mode` is ever set (mode-governance
pathway inert), and neither `action_bias` nor any rule-pool /
`lateral_pfc_rule_readout_consumer` flag is set (SD-082's bias-head pathway
inert). Confirmed by reading both entries' descriptions against this driver's
actual config surface before proceeding -- neither blocks this run.

claim_ids: ['MECH-162'] (single-claim evidence; MECH-236's wiring (does the
current_z_goal -> propose_trajectories -> MECH-293 channel exist and matter at
all) is a precondition this run relies on but does not re-test -- MECH-236 is
NOT tagged here per the claim_ids-accuracy rule; MECH-292/293/SD-039 substrate
correctness is likewise exercised, not re-verified, matching the V3-EXQ-807/
868/881/914 precedent family).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_997_mech162_zresource_zworld_planning_reconvergence.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.hippocampal.anchor_set import AnchorGoalPayload  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_997_mech162_zresource_zworld_planning_reconvergence"
QUEUE_ID = "V3-EXQ-997"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-162"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EVB-1447"

ARMS = ["ARM_ZRESOURCE", "ARM_ZWORLD"]
SEEDS: List[int] = [42, 43, 45, 46, 47]  # seed 44: recurring per-seed early-death
                                          # instability on this env config family
                                          # (see V3-EXQ-868/881/914 docstrings)

# --- Env ---
GRID_SIZE = 10
N_RESOURCE_TYPES = 2
NUM_RESOURCES = 10         # per-type (sd049_preserve_per_type_density=True) -> 20 total
NUM_HAZARDS = 0
DRIVE_WEIGHT = 2.0

# --- Schedule ---
# READINESS-CALIBRATION PASS (orchestrator-directed re-issue, post Step-4.5
# fix pass): the ORIGINAL schedule (8 resources/100 cells, 20 episodes x 80
# steps) was verified end-to-end against the corrected substrate (all three
# Step 4.5 fixes applied) and measured R1/R2 NOT met on either arm (internal
# 5-seed run: ARM_ZRESOURCE R1=0.6/0.8 R2=0.4/0.8; ARM_ZWORLD R1=0.8/0.8
# R2=0.4/0.8 -- see script history / V3-EXQ-997.report.md "Internal
# verification run" for the full per-seed breakdown). One seed (43) got
# ZERO contacts across the full 1600-step cell at the original density.
# NOTE (substrate-confirmed, code read of REEAgent.reset()): agent.reset() wipes
# the SD-039 anchor set (active + inactive) on EVERY episode boundary
# (use_anchor_sets branch, ~agent.py:3762-3769) -- so the discrimination query
# can only ever compare anchors written WITHIN the same episode, never across
# episodes. Because R2 (engagement) is keyed on WITHIN-episode dual-type
# co-presence, more TOTAL contacts help only if they land in the SAME
# episode -- so both STEPS_PER_EPISODE and resource DENSITY (not just
# N_EPISODES) were raised for this re-issue: NUM_RESOURCES 4->10/type
# (8%->20% grid density) and STEPS_PER_EPISODE 80->150 give each single
# episode a much higher chance of contacting BOTH types before the episode
# ends and its anchor pool is wiped; N_EPISODES 20->25 then gives more
# independent within-episode chances across the run. Re-verified (internal
# 5-seed run, this schedule): see V3-EXQ-997.report.md "Readiness
# calibration pass" for the resulting per-seed numbers.
N_EPISODES = 25
STEPS_PER_EPISODE = 150

# --- Forced goal-seeding at genuine contact (not a periodic force -- gated on
# obs_dict["resource_type_at_agent"] > 0, the authoritative at-cell signal) ---
FORCED_BENEFIT = 0.5
FORCED_DRIVE = 0.9

# --- MECH-292/293 config (identical both arms) ---
GOAL_MATCH_WEIGHT = 1.0          # components["goal_match"] == raw centered cosine
GOAL_MATCH_FLOOR = -1.0          # maximally permissive: see the FULL spread
GHOST_TOP_K = 200                # generous cap; realistic pool is far smaller

# --- Pre-registered thresholds ---
EFFECT_FLOOR = 0.02
ENGAGEMENT_FLOOR = 0.10
SEED_PASS_FRACTION = 3.0 / 5.0
READINESS_SEED_PASS_FRACTION = 4.0 / 5.0

# GhostGoalBankConfig defaults to include_inactive=True, include_active=False
# (config.py:2337-2338) -- the default queryable pool is GHOSTS (demoted/
# inactive anchors) only, never the live/active one. Combined with the
# per-resource-type family partitioning below (write_anchor's dual-trace
# demotion is scoped to (scale, stream_mixture); a type's FIRST contact is
# always "active" and therefore invisible to the default pool -- only a
# SECOND same-type contact demotes the first into a queryable ghost). So a
# type only becomes comparable once contacted >= 2 times in an episode, not
# >= 1; see the positive-control finding cited in the ANCHOR POPULATION /
# R1 sections below.
MIN_CONTACTS_PER_TYPE_FOR_GHOST = 2

# z_goal-stream liveness, pooled across arm x seed cells for the manifest block.
_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=N_RESOURCE_TYPES,
        sd049_preserve_per_type_density=True,
        resource_benefit=0.5,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _config_slice(arm: str) -> Dict[str, Any]:
    use_resource = arm == "ARM_ZRESOURCE"
    return {
        "env": {"size": GRID_SIZE, "num_resources": NUM_RESOURCES,
                "num_hazards": NUM_HAZARDS,
                "multi_resource_heterogeneity_enabled": True,
                "n_resource_types": N_RESOURCE_TYPES,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "z_goal_enabled": True,
        "use_resource_encoder": use_resource,
        "drive_weight": DRIVE_WEIGHT,
        "wanting_weight": 0.0,
        "use_per_stream_vs": True,
        "use_event_segmenter": True,
        "use_invalidation_trigger": True,
        "use_anchor_sets": True,
        "use_sd039_anchor_payload": True,
        "use_mech292_ghost_bank": True,
        "use_mech293_ghost_probes": True,
        "mech292_goal_match_weight": GOAL_MATCH_WEIGHT,
        "mech292_goal_match_floor": GOAL_MATCH_FLOOR,
        "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE,
    }


def _make_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    use_resource = arm == "ARM_ZRESOURCE"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,

        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        benefit_eval_enabled=True,
        benefit_weight=1.0,

        # Isolate the MECH-293 channel from the separate VALENCE_WANTING
        # pathway (V3-EXQ-914 "SCOPING NOTE").
        wanting_weight=0.0,

        # MECH-293's full prerequisite stack -- IDENTICAL between arms. Only
        # `use_resource_encoder` below differs between ARM_ZRESOURCE and
        # ARM_ZWORLD; every other flag is held constant so the contrast
        # isolates the goal-seeding SOURCE latent, not channel wiring.
        use_per_stream_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        use_mech292_ghost_bank=True,
        mech292_goal_match_weight=GOAL_MATCH_WEIGHT,
        mech292_goal_match_floor=GOAL_MATCH_FLOOR,
        mech292_top_k=GHOST_TOP_K,
        use_mech293_ghost_probes=True,
        mech293_ghost_fraction=0.2,

        # SD-079 common-mode-centered goal-cue matching (Step 4.5 red-team B2
        # fix). Without this, Anchor.goal_match is a RAW cosine and z_goal
        # vectors sit in a near-degenerate SD-008 common-mode cone (measured
        # 0.985-1.000 across independent contacts), compressing the entire
        # SAME_TYPE-vs-DIFF_TYPE discrimination_delta below EFFECT_FLOOR
        # regardless of any genuine identity-vs-position signal. Centering
        # subtracts a slow-EMA common-mode baseline from both the stored
        # snapshot and the query before the cosine, restoring real dynamic
        # range (SD-079 docstring; anchor_set.py). Live on BOTH arms
        # identically -- it is a readout correction, not part of the
        # manipulation under test.
        goal_cue_centering=True,
    )
    # SD-015: from_dims has no use_resource_encoder param -- set directly on the
    # latent config (matches V3-EXQ-531's established pattern). z_resource_dim
    # must equal goal.goal_dim for direct z_goal seeding (config.py comment,
    # SD-015 block); world_dim==goal_dim==32 already by from_dims defaults, so
    # this keeps every seeding source dimensionally consistent for
    # Anchor.goal_match's numel() equality check.
    cfg.latent.use_resource_encoder = use_resource
    cfg.latent.z_resource_dim = cfg.goal.goal_dim
    agent = REEAgent(cfg)
    # Step 4.5 red-team B3 fix: the default ghost pool is INACTIVE-ONLY
    # (GhostGoalBankConfig.include_active=False -- config.py:2337-2338). Under
    # the per-write-unique-family scheme this driver now uses (see the
    # write_anchor call site in _run_cell), dual-trace demotion never fires
    # at all -- nothing is ever marked inactive except by the 128-per-scale
    # FIFO cap, which this run's per-episode anchor counts never approach.
    # Without include_active=True the ghost pool would therefore stay EMPTY
    # for the whole run (every anchor is active, no anchor is ever
    # inactive), and R1/R2/the discrimination sampler would silently never
    # see a single entry. Flip it on both arms identically.
    if agent.hippocampal.ghost_goal_bank is not None:
        agent.hippocampal.ghost_goal_bank.config.include_active = True
        agent.hippocampal.ghost_goal_bank.config.include_inactive = True
    return agent


def _make_nogoal_agent(env: CausalGridWorldV2) -> REEAgent:
    """R0 sanity leg: z_goal entirely disabled -- the mechanism-off control."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        z_goal_enabled=False,
        drive_weight=0.0,
        benefit_eval_enabled=False,
        wanting_weight=0.0,
        use_per_stream_vs=False,
        use_event_segmenter=False,
        use_invalidation_trigger=False,
        use_anchor_sets=False,
        use_sd039_anchor_payload=False,
        use_mech292_ghost_bank=False,
        use_mech293_ghost_probes=False,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# R0: mechanism-off sanity leg (single deterministic run, NOT part of the grid)
# ---------------------------------------------------------------------------
def run_r0_sanity(dry_run: bool = False) -> Dict[str, Any]:
    steps = 6 if dry_run else 20
    torch.manual_seed(42)
    env = _make_env(42)
    agent = _make_nogoal_agent(env)
    device = agent.device
    wd = agent.config.latent.world_dim

    reasons: List[str] = []
    _, obs = env.reset()
    agent.reset()
    for t in range(steps):
        obs_body = obs["body_state"].to(device)
        obs_world = obs["world_state"].to(device)
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick", True)
            else torch.zeros(1, wd, device=device)
        )
        agent.generate_trajectories(latent, e1_prior, ticks)
        if ticks.get("e3_tick"):
            diag = getattr(agent.hippocampal, "_last_propose_diagnostics", {}) or {}
            reasons.append(str(diag.get("mech293_reason", "")))
        _, _, done, info, obs = env.step(0)
        if done:
            break

    n_ticks = len(reasons)
    n_no_z_goal = sum(1 for r in reasons if r == "no_z_goal")
    return {
        "n_e3_ticks_observed": n_ticks,
        "n_no_z_goal_reason": n_no_z_goal,
        "all_no_z_goal": bool(n_ticks > 0 and n_no_z_goal == n_ticks),
        "reasons_observed": sorted(set(reasons)),
    }


# ---------------------------------------------------------------------------
# Single arm x seed cell
# ---------------------------------------------------------------------------
def _run_cell(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    n_episodes = 3 if dry_run else N_EPISODES
    steps_per = 10 if dry_run else STEPS_PER_EPISODE

    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(arm),
                  script_path=Path(__file__)) as cell:
        env = _make_env(seed)
        agent = _make_agent(env, arm)
        device = agent.device
        wd = agent.config.latent.world_dim

        # Driver-side bookkeeping (NOT read back from the substrate -- the
        # ground-truth type each written anchor was contacted at).
        anchor_type_by_key: Dict[Tuple[str, str, Tuple[str, ...]], int] = {}
        n_contacts_by_type: Dict[int, int] = {}
        last_wanted_type: Optional[int] = None

        total_steps = 0
        total_contacts = 0
        zgoal_norms: List[float] = []
        ghost_admitted: List[int] = []
        ghost_goal_match_ctx: List[float] = []
        n_e3_ticks_total = 0
        n_e3_ticks_qualifying = 0
        discrim_samples: List[float] = []
        same_type_samples: List[float] = []
        diff_type_samples: List[float] = []

        for ep in range(n_episodes):
            _, obs = env.reset()
            agent.reset()
            # Step 4.5 red-team B1 fix: the LEGACY auto-consume-on-entry path
            # (env's default, consummatory_act_enabled=False) clears
            # obs_dict["resource_type_at_agent"] to 0 in the SAME step() call
            # that performs the consumption -- _consume_resource_at() runs
            # BEFORE _get_observation_dict() builds that field
            # (causal_grid_world.py). So the field is NEVER nonzero on any
            # obs_dict this driver ever reads; measured (Step 4.5 red-team,
            # confirmed by rerunning this exact driver): total_contacts == 0
            # in every cell at full length. The authoritative signal for
            # "a contact of type T happened on the step just taken" is
            # info["sd049_consumed_type_tag_this_tick"] (cached BEFORE the
            # grid cell is cleared; causal_grid_world.py:2127-2128,
            # exposed at :3300; V3-EXQ-681's own documented convention).
            # That tag is only available in the `info` dict env.step()
            # returns -- one step removed from the top-of-loop obs read this
            # driver otherwise uses. So the reaction is DEFERRED by exactly
            # one tick: pending_contact_type, set from the PREVIOUS
            # iteration's env.step() info, is consumed at the TOP of THIS
            # iteration using THIS iteration's freshly-sensed `latent`
            # (agent.sense() on the obs returned by that same step() call --
            # i.e. the agent is still standing at the now-emptied resource
            # cell; nothing has moved between the consuming step and this
            # reaction). This is a genuine, not corrupted, per-contact
            # z_world/z_goal snapshot.
            pending_contact_type = 0
            for t in range(steps_per):
                obs_body = obs["body_state"].to(device)
                obs_world = obs["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", True)
                    else torch.zeros(1, wd, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

                # --- Genuine at-cell contact (from the LAST step's consumed-
                # type info tag): seed z_goal + write an anchor ---
                rtype_at_agent = pending_contact_type
                pending_contact_type = 0
                if rtype_at_agent > 0:
                    agent.update_z_goal(
                        benefit_exposure=FORCED_BENEFIT,
                        drive_level=FORCED_DRIVE,
                        resource_type=rtype_at_agent,
                    )
                    last_wanted_type = rtype_at_agent
                    total_contacts += 1
                    n_contacts_by_type[rtype_at_agent] = (
                        n_contacts_by_type.get(rtype_at_agent, 0) + 1
                    )
                    zg = (
                        agent.goal_state.z_goal
                        if agent.goal_state is not None else None
                    )
                    if zg is not None and agent.hippocampal.anchor_set is not None:
                        zgoal_norms.append(float(zg.detach().norm().item()))
                        # Step 4.5 red-team B3 fix: stream_mixture now embeds
                        # a per-write-UNIQUE id (key_id) alongside the type
                        # tag, so no two writes -- not even two contacts of
                        # the SAME type -- ever share a (scale,
                        # stream_mixture) family. write_anchor's dual-trace
                        # demotion (anchor_set.py:288-293) therefore never
                        # fires from this driver, so it can never remap a
                        # prior anchor's goal_payload onto a LATER contact's
                        # z_goal (the corruption Step 4.5 found: every ghost
                        # in a shared-family pool held the SAME latest-
                        # contact snapshot, producing a trivial near-1.0
                        # SAME_TYPE match by construction, not by genuine
                        # identity generalisation). Each anchor now
                        # permanently holds its own true contact-time
                        # z_goal_snapshot. Consequence: nothing is ever
                        # demoted to inactive except by the 128-per-scale
                        # FIFO cap (anchor_set.py _enforce_scale_cap, which
                        # does NOT remap payloads either) -- see the
                        # include_active=True fix in _make_agent, required
                        # for these anchors to be visible in the default
                        # inactive-only ghost pool at all.
                        key_id = f"e{ep}s{t}"
                        anchor = agent.hippocampal.anchor_set.write_anchor(
                            "fast", key_id, (f"world_t{rtype_at_agent}", key_id),
                            z_world=latent.z_world.detach().reshape(-1),
                            goal_payload=AnchorGoalPayload(
                                z_goal_snapshot=zg.detach().reshape(-1).clone(),
                                wanting_strength=1.0,
                            ),
                        )
                        anchor_type_by_key[anchor.key] = rtype_at_agent

                # --- Retrieval query at every E3 tick (contact or not), once
                # at least one contact has occurred this run ---
                if ticks.get("e3_tick"):
                    n_e3_ticks_total += 1
                    diag = (
                        getattr(agent.hippocampal, "_last_propose_diagnostics", {})
                        or {}
                    )
                    ghost_admitted.append(int(diag.get("mech293_n_ghost_admitted", 0)))
                    if diag.get("mech293_reason") == "ok":
                        ghost_goal_match_ctx.append(
                            float(diag.get("mech293_mean_goal_match_at_seed", 0.0))
                        )

                    if (
                        last_wanted_type is not None
                        and agent.goal_state is not None
                        and agent.hippocampal.ghost_goal_bank is not None
                    ):
                        entries = agent.hippocampal.ghost_goal_bank.rank(
                            agent.goal_state.z_goal
                        )
                        same_gm: List[float] = []
                        diff_gm: List[float] = []
                        for entry in entries:
                            etype = anchor_type_by_key.get(entry.anchor.key)
                            if etype is None:
                                continue
                            gm = float(entry.components.get("goal_match", 0.0))
                            if etype == last_wanted_type:
                                same_gm.append(gm)
                            else:
                                diff_gm.append(gm)
                        if same_gm and diff_gm:
                            n_e3_ticks_qualifying += 1
                            delta = statistics.fmean(same_gm) - statistics.fmean(diff_gm)
                            discrim_samples.append(delta)
                            same_type_samples.extend(same_gm)
                            diff_type_samples.extend(diff_gm)

                _, _harm_signal, done, info, obs = env.step(
                    int(action.argmax(dim=-1).item())
                )
                total_steps += 1
                # Cache this step's consumed-type tag (B1 fix) for the TOP
                # of the next iteration; 0 when nothing was consumed. done
                # short-circuits below without a next iteration to react in
                # -- a consumption on the terminal step of an episode is
                # deliberately not reacted to (matches the pre-existing
                # per-episode anchor-pool-reset design: nothing carries
                # across an episode boundary anyway).
                pending_contact_type = int(
                    info.get("sd049_consumed_type_tag_this_tick", 0)
                )
                if done:
                    break
            if (ep + 1) % 5 == 0:
                print(
                    f"  [train] mech162 seed={seed} arm={arm} ep {ep+1}/{n_episodes} "
                    f"contacts={total_contacts}",
                    flush=True,
                )

        n_types_contacted = sum(
            1 for v in n_contacts_by_type.values()
            if v >= MIN_CONTACTS_PER_TYPE_FOR_GHOST
        )
        engagement_frac = (
            n_e3_ticks_qualifying / n_e3_ticks_total if n_e3_ticks_total else 0.0
        )
        mean_discrim = statistics.fmean(discrim_samples) if discrim_samples else 0.0
        row = {
            "arm": arm,
            "seed": seed,
            "total_steps": total_steps,
            "total_contacts": total_contacts,
            "n_contacts_by_type": dict(n_contacts_by_type),
            "n_types_contacted": n_types_contacted,
            "type_coverage_met": n_types_contacted >= N_RESOURCE_TYPES,
            "n_e3_ticks_total": n_e3_ticks_total,
            "n_e3_ticks_qualifying": n_e3_ticks_qualifying,
            "engagement_fraction": engagement_frac,
            "engagement_met": engagement_frac >= ENGAGEMENT_FLOOR,
            "mean_discrimination_delta": mean_discrim,
            "n_discrimination_samples": len(discrim_samples),
            "mean_same_type_goal_match": (
                statistics.fmean(same_type_samples) if same_type_samples else 0.0
            ),
            "mean_diff_type_goal_match": (
                statistics.fmean(diff_type_samples) if diff_type_samples else 0.0
            ),
            "zgoal_norm_mean": statistics.fmean(zgoal_norms) if zgoal_norms else 0.0,
            "n_zgoal_sampled": len(zgoal_norms),
            "mech293_n_ghost_admitted_mean": (
                statistics.fmean(ghost_admitted) if ghost_admitted else 0.0
            ),
            "mech293_n_ghost_admitted_nonzero_frac": (
                sum(1 for g in ghost_admitted if g > 0) / len(ghost_admitted)
                if ghost_admitted else 0.0
            ),
            "mech293_context_goal_match_mean": (
                statistics.fmean(ghost_goal_match_ctx) if ghost_goal_match_ctx else 0.0
            ),
        }
        cell.stamp(row)
    _ZG.observe(agent)
    cell_ok = total_steps > 0 and zgoal_norm_mean_ok(row)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def zgoal_norm_mean_ok(row: Dict[str, Any]) -> bool:
    return row.get("n_zgoal_sampled", 0) > 0 and row.get("zgoal_norm_mean", 0.0) > 0.0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r0 = run_r0_sanity(dry_run=dry_run)

    seeds = [SEEDS[0]] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed, dry_run=dry_run))

    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm"]].append(r)

    # --- R1/R2 readiness (per arm) ---
    readiness_checks = []
    readiness_ok = True
    for arm in ARMS:
        arm_rows = by_arm[arm]
        n = len(arm_rows)
        type_cov_frac = sum(1 for r in arm_rows if r["type_coverage_met"]) / n
        engagement_frac_seedwise = sum(1 for r in arm_rows if r["engagement_met"]) / n
        r1_met = type_cov_frac >= READINESS_SEED_PASS_FRACTION
        r2_met = engagement_frac_seedwise >= READINESS_SEED_PASS_FRACTION
        readiness_checks.append({
            "name": f"{arm}_R1_type_coverage_seed_fraction",
            "measured": type_cov_frac,
            "threshold": READINESS_SEED_PASS_FRACTION,
            "met": r1_met,
            "kind": "readiness",
        })
        readiness_checks.append({
            "name": f"{arm}_R2_engagement_seed_fraction",
            "measured": engagement_frac_seedwise,
            "threshold": READINESS_SEED_PASS_FRACTION,
            "met": r2_met,
            "kind": "readiness",
        })
        readiness_ok = readiness_ok and r1_met and r2_met

    # --- C1 (load-bearing) ---
    seed_deltas = []
    n_seed_zresource_gt = 0
    for seed in seeds:
        gm_zr = next(r for r in rows if r["arm"] == "ARM_ZRESOURCE" and r["seed"] == seed)
        gm_zw = next(r for r in rows if r["arm"] == "ARM_ZWORLD" and r["seed"] == seed)
        d = gm_zr["mean_discrimination_delta"] - gm_zw["mean_discrimination_delta"]
        seed_deltas.append(d)
        if gm_zr["mean_discrimination_delta"] > gm_zw["mean_discrimination_delta"]:
            n_seed_zresource_gt += 1

    mean_c1_delta = statistics.fmean(seed_deltas) if seed_deltas else 0.0
    seed_fraction_zr_gt = n_seed_zresource_gt / len(seeds) if seeds else 0.0
    c1_met = (
        mean_c1_delta >= EFFECT_FLOOR
        and seed_fraction_zr_gt >= SEED_PASS_FRACTION
    )

    degeneracy = check_degeneracy({
        "C1_discrimination_delta": {
            "groups": [
                [
                    next(r for r in rows if r["arm"] == "ARM_ZRESOURCE" and r["seed"] == s)[
                        "mean_discrimination_delta"
                    ],
                    next(r for r in rows if r["arm"] == "ARM_ZWORLD" and r["seed"] == s)[
                        "mean_discrimination_delta"
                    ],
                ]
                for s in seeds
            ]
        }
    })

    if not readiness_ok:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif c1_met:
        outcome = "PASS"
        evidence_direction = "supports"
        label = "mech162_zresource_discriminates_identity_invariant_to_position"
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        label = "mech162_zresource_no_discrimination_advantage_over_zworld"

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "arm_results": rows,
        "r0_sanity_leg": r0,
        "interpretation": {
            "label": label,
            "preconditions": readiness_checks,
            "criteria_non_degenerate": {
                "C1_discrimination_delta": not degeneracy.get(
                    "degenerate_metrics", {}
                ).get("C1_discrimination_delta", False),
            },
        },
        "criteria": [
            {
                "name": "C1_discrimination_delta_floor",
                "load_bearing": True,
                "passed": bool(c1_met),
                "measured_mean_delta": mean_c1_delta,
                "threshold": EFFECT_FLOOR,
                "measured_seed_fraction": seed_fraction_zr_gt,
                "seed_fraction_threshold": SEED_PASS_FRACTION,
            },
        ],
        **degeneracy,
        "n_seeds": len(seeds),
        "arms": ARMS,
    }
    manifest["elapsed_seconds_internal"] = time.perf_counter() - t0
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args, _unknown = parser.parse_known_args()

    t_start = time.perf_counter()
    manifest = run_experiment(dry_run=args.dry_run)

    agent_for_stamp = None  # cells build/discard their own agents; z_goal
                             # liveness is carried via the _ZG accumulator.
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config={
            "seeds": SEEDS,
            "arms": ARMS,
            "grid_size": GRID_SIZE,
            "n_resource_types": N_RESOURCE_TYPES,
            "num_resources": NUM_RESOURCES,
            "num_hazards": NUM_HAZARDS,
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "effect_floor": EFFECT_FLOOR,
            "engagement_floor": ENGAGEMENT_FLOOR,
        },
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t_start,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Wrote manifest: {out_path}")
    print(f"outcome: {manifest['outcome']}  evidence_direction: {manifest['evidence_direction']}")

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=manifest.get("run_id"),
        dry_run=args.dry_run,
    )
